/**
 Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 This work was (partially) funded by the DARPA XDATA program under
 AFRL Contract #FA8750-13-C-0002.

 This material is based upon work supported by the Defense Advanced
 Research Projects Agency (DARPA) under Contract No. D14PC00029.
 */

#include "mpi.h"
#include "kernel.cuh"
#include <GASengine/statistics.h>
#include <BitmapCompressor/Compressor.cuh>
#include <iostream>
#include <bitset>
#include <typeinfo>

#ifndef VALUECOMM_H_
#define VALUECOMM_H_
using namespace std;
using namespace MPI;
using namespace mpikernel;

template <class T> MPI_Datatype MyGetMPIType () { cerr << "ERROR in GetMPIType() -- no type found" << endl;return 0; }
template <> inline MPI_Datatype MyGetMPIType<int> () { return MPI_INT; }//模板显式特化
template <> inline MPI_Datatype MyGetMPIType<float> () { return MPI_FLOAT; }//模板显式特化

//MPI_Op_create中调用时，只有函数模板中的类型T和函数名能作为区分的依据，函数的参数的类型没法作为区分的依据，
//解决办法为：用函数模板中的类型T和函数名作为函数的区分依据，同时函数的形参用void*，在函数的内部用T*来强制转换成想要的类型T的指针
template <class GatherType, class Program> void MySumNew(void* in, void* inout, int* len, MPI_Datatype* type){
	for (int i = 0; i < *len; i++){
		//Program类型作为函数模板中的参数传入并被使用
		typename Program::gather_sum gather_sum_functor;
		((GatherType*)inout)[i] = gather_sum_functor( ((GatherType*)in)[i], ((GatherType*)inout)[i] );//Program::gather_sum(*in, *inout);
    }
}

template<typename GatherType, typename Program>
class ValueCommunicator
//frontier contraction in a 2-d partitioned graph
{
public:
  int pi; //row
  int pj; //column
  int p;
  int n;
  MPI_Group orig_group, new_row_group, new_col_group;
  MPI_Comm new_row_comm, new_col_comm;
  int new_row_rank, new_col_rank;
  double init_time, propagate_time, broadcast_time, compression_time, copy_time, bitunion_time, decompression_time, propagate_wait, broadcast_wait;
  double prop_row, prop_col;

  Statistics* stats;

  unsigned int *bitmap_compressed;
  unsigned char *bitmap_decompressed;
  unsigned char *out_copy, *assigned_temp, *prefix_temp;
  Compressor* comp;
  double compression_ratio_broadcast;
  double compression_ratio;
  unsigned int compressed_size;
public:

  ValueCommunicator(int l_pi, int l_pj, int l_p, int l_n, Statistics* l_stats) :
      init_time(0.0), compressed_size(0), propagate_time(0.0), propagate_wait(0.0), broadcast_time(0.0), broadcast_wait(0.0), compression_time(0.0), copy_time(0.0), bitunion_time(0.0), prop_row(0.0), prop_col(
          0.0)

  //l_pi is the x index
  //l_pj is the y index
  //l_p  is the number of partitions in 1d. usually, sqrt(number of processors)
  //l_n  is the size of the problem, number of vertices
  {
    double starttime, endtime;
    starttime = MPI_Wtime();
    pi = l_pi;
    pj = l_pj;
    p = l_p;
    n = l_n;
    stats = l_stats;

    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    //build original ranks for the processors

    //		int row_indices[p], col_indices[p + 1];
    int *row_indices = new int[p];
    int *col_indices = new int[p + 1];

    for (int i = 0; i < p; i++)
      row_indices[i] = pi * p + i;
    /*		for(int i=0;i<=pi-1;i++)
     row_indices[i+p] = i*p+pi;
     for(int i=pi+1;i<p;i++)
     row_indices[i+p-1] = i*p+pi;
     */for (int i = 0; i < p; i++)
      col_indices[i] = i * p + pj;
    /*              for(int i=0;i<=pj-1;i++)
     col_indices[i] = i*p+pj;
     for(int i=pj+1;i<p;i++)
     col_indices[i-1] = i*p+pj;
     col_indices[p-1] = pj*p+p-1;
     */
    MPI_Group_incl(orig_group, p, row_indices, &new_row_group);
    MPI_Group_incl(orig_group, p, col_indices, &new_col_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_row_group, &new_row_comm);
    MPI_Comm_create(MPI_COMM_WORLD, new_col_group, &new_col_comm);
    MPI_Group_rank(new_row_group, &new_row_rank);
    MPI_Group_rank(new_col_group, &new_col_rank);
    endtime = MPI_Wtime();
    //MPI_Barrier(new_row_comm);
    //MPI_Barrier(new_col_comm);
    init_time = endtime - starttime;
    propagate_time = 0;
    propagate_wait = 0;
    broadcast_time = 0;
    broadcast_wait = 0;
    copy_time = 0;
    bitunion_time = 0;

    util::B40CPerror(cudaMalloc((void**) &out_copy, ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(cudaMalloc((void**) &assigned_temp, ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(cudaMalloc((void**) &prefix_temp, ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(cudaMalloc((void**) &bitmap_compressed, (n + 31 - 1) / 31 * sizeof(unsigned int)), "CsrProblem cudaMalloc bitmap_compressed failed", __FILE__, __LINE__);

    util::B40CPerror(cudaMemset(bitmap_compressed, 0, (n + 31 - 1) / 31 * sizeof(unsigned int)), "Memset bitmap_compressed failed", __FILE__, __LINE__);

    util::B40CPerror(cudaMalloc((void**) &bitmap_decompressed, (n + 31 - 1) / 31 * sizeof(unsigned int)), "CsrProblem cudaMalloc bitmap_decompressed failed", __FILE__, __LINE__);

    util::B40CPerror(cudaMemset(bitmap_decompressed, 0, (n + 31 - 1) / 31 * sizeof(unsigned int)), "Memset bitmap_decompressed failed", __FILE__, __LINE__);

    comp = new Compressor(n);
    //    comp = new Compressor(186);
  }
  
  void valueReduce(GatherType* gatherValues,	GatherType* redResult, int &frontierSize, GatherType initValue, int nodes) {
	int tag = 0;
	MPI_Op myop;
    int commune = 0;
    MPI_Op_create( (MPI_User_function *) MySumNew<GatherType, Program>,
					commune, &myop);
	MPI_Reduce(gatherValues, redResult, frontierSize, MyGetMPIType<GatherType>(), myop, pi, new_row_comm);
	if (pi == pj) {
		for (int i = 0; i < p; i++) {
			if (i != pi) {
				MPI_Send(redResult, frontierSize, MyGetMPIType<GatherType>(), i * p + pi, tag,
						MPI_COMM_WORLD);
			}
		}
	} else {
		MPI_Status status;
		// Receive at most MAX_NUMBERS from process zero
		MPI_Recv(redResult, nodes, MyGetMPIType<GatherType>(), pj * p + pj, tag, MPI_COMM_WORLD, &status);

		// After receiving the message, check the status to determine
		// how many numbers were actually received
		MPI_Get_count(&status, MyGetMPIType<GatherType>(), &frontierSize);
	}
  }
};

#endif /* VALUECOMM_H_ */
