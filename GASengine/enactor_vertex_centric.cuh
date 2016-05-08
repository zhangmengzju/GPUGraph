/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/*THIS FILE HAS BEEN MODIFIED FROM THE ORIGINAL*/

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

#pragma once

#include <stdlib.h>
#include <config.h>

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/global_barrier.cuh>

#include <GASengine/problem_type.cuh>
#include <GASengine/csr_problem.cuh>
#include <GASengine/enactor_base.cuh>

//#include <GASengine/vertex_centric/gather/kernel.cuh>
//#include <GASengine/vertex_centric/gather/kernel_policy.cuh>
#include <GASengine/vertex_centric/mgpukernel/kernel.cuh>
#include <GASengine/vertex_centric/expand_atomic/kernel.cuh>
#include <GASengine/vertex_centric/expand_atomic/kernel_policy.cuh>
#include <GASengine/vertex_centric/expand_atomic_flag/kernel.cuh>
#include <GASengine/vertex_centric/expand_atomic_flag/kernel_policy.cuh>
#include <GASengine/vertex_centric/contract_atomic/kernel.cuh>
#include <GASengine/vertex_centric/contract_atomic/kernel_policy.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>

#include <vector>
#include <iterator>
#include <moderngpu.cuh>
//#include <util.h>
#include <util/mgpucontext.h>
#include <mgpuenums.h>

#include <MPI/wave.h>
#include <MPI/ValueComm.h>
#include <MPI/kernel.cuh>
#include <GASengine/statistics.h>

using namespace b40c;

using namespace std;

namespace GASengine
{

template<typename CsrProblem, typename Program, bool INSTRUMENT> // Whether or not to collect per-CTA clock-count statistics
class EnactorVertexCentric: public EnactorBase
{
  //---------------------------------------------------------------------
  // Members
  //---------------------------------------------------------------------

protected:

  //convenience

  void errorCheck(cudaError_t err, const char* file, int line)
  {
    if (err != cudaSuccess)
    {
      int rank_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
      printf("rank_id: %d, %s(%d): cuda error %d (%s)\n", rank_id, file, line, err, cudaGetErrorString(err));
      abort();
    }
  }

  //use only for debugging kernels
  //this slows stuff down a LOT

  void syncAndErrorCheck(const char* file, int line)
  {
    cudaDeviceSynchronize();
    errorCheck(cudaGetLastError(), file, line);
  }

  //this is undefined at the end of this template definition
#define CHECK(X) errorCheck(X, __FILE__, __LINE__)
#define SYNC_CHECK() syncAndErrorCheck(__FILE__, __LINE__)

  /**
   * CTA duty kernel stats
   */
  util::KernelRuntimeStatsLifetime expand_kernel_stats;
  util::KernelRuntimeStatsLifetime filter_kernel_stats;
  util::KernelRuntimeStatsLifetime contract_kernel_stats;
  util::KernelRuntimeStatsLifetime backward_sum_kernel_stats;
  util::KernelRuntimeStatsLifetime backward_contract_kernel_stats;

  unsigned long long total_runtimes; // Total time "worked" by each cta
  unsigned long long total_lifetimes; // Total time elapsed by each cta
  unsigned long long total_queued;

  volatile int *done;
  int *d_done;
  cudaEvent_t throttle_event;
  Config cfg;
  mgpu::ContextPtr m_mgpuContext;
  Statistics* stats;

  /**
   * Mechanism for implementing software global barriers from within
   * a single grid invocation
   */
  util::GlobalBarrierLifetime global_barrier;

  /**
   * Current iteration (mapped into GPU space so that it can
   * be modified by multi-iteration kernel launches)
   */
  volatile long long *iteration;
  long long *d_iteration;

  //---------------------------------------------------------------------
  // Methods
  //---------------------------------------------------------------------

protected:

  /**
   * Prepare enactor for search.  Must be called prior to each search.
   */
  cudaError_t Setup(CsrProblem &csr_problem, int expand_grid_size, int contract_grid_size, int iter)
  {
    typedef typename CsrProblem::SizeT SizeT;
    typedef typename CsrProblem::VertexId VertexId;
    typedef typename CsrProblem::VisitedMask VisitedMask;

    cudaError_t retval = cudaSuccess;

    do
    {

      // Make sure host-mapped "done" is initialized
      if (!done)
      {
        int flags = cudaHostAllocMapped;

        // Allocate pinned memory for done
        if (retval = util::B40CPerror(cudaHostAlloc((void **) &done, sizeof(int) * 1, flags), "EnactorContractExpand cudaHostAlloc done failed", __FILE__, __LINE__)) break;

        // Map done into GPU space
        if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &d_done, (void *) done, 0), "EnactorContractExpand cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

        // Create throttle event
        if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming), "EnactorContractExpand cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__))
          break;
      }

      // Make sure host-mapped "iteration" is initialized
      if (!iteration)
      {

        int flags = cudaHostAllocMapped;

        // Allocate pinned memory
        if (retval = util::B40CPerror(cudaHostAlloc((void **) &iteration, sizeof(long long) * 1, flags), "EnactorContractExpand cudaHostAlloc iteration failed", __FILE__, __LINE__)) break;

        // Map into GPU space
        if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &d_iteration, (void *) iteration, 0), "EnactorContractExpand cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__))
          break;
      }

      // Make sure software global barriers are initialized
      if (retval = global_barrier.Setup(expand_grid_size)) break;

      // Make sure our runtime stats are initialized
      if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
      if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;
      //            if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;
      if (retval = backward_sum_kernel_stats.Setup(expand_grid_size)) break;
      if (retval = backward_contract_kernel_stats.Setup(contract_grid_size)) break;

      // Reset statistics
      iteration[0] = iter;
      total_runtimes = 0;
      total_lifetimes = 0;
      total_queued = 0;
      done[0] = -1;

      // Single-gpu graph slice
      typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

      // Bind bitmask texture
      int bytes = (graph_slice->nodes + 8 - 1) / 8;
      cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
      if (retval = util::B40CPerror(cudaBindTexture(0, vertex_centric::contract_atomic::BitmaskTex<VisitedMask>::ref, graph_slice->d_visited_mask,
      //bitmask_desc,
          bytes), "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
    }
    while (0);

    return retval;
  }

public:
  int *m_hostMappedValue;
  int *m_deviceMappedValue;
  //    thrust::device_vector<int> d_vertex_ids;
  int* d_frontier_size;
  int* d_edge_frontier_size;
  int* h_edge_frontier_size_pinned;
  int frontier_size;
  int edge_frontier_size;

  typedef typename Program::VertexId VertexId;
  typedef typename Program::SizeT SizeT;
  typedef typename Program::DataType DataType;
  typedef typename Program::GatherType GatherType;

  GatherType *m_gatherMapTmp;
  GatherType *m_gatherTmp;
  GatherType *m_gatherTmp1;
  GatherType *m_gatherTmp2;
  GatherType *m_reduceResult;
  VertexId *m_gatherDstsTmp;
  std::auto_ptr<mgpu::ReduceByKeyPreprocessData> preprocessData;
  bool preComputed;

  /**
   * Constructor
   */
  EnactorVertexCentric(Config cfg, bool DEBUG = false) :
      cfg(cfg), EnactorBase(EDGE_FRONTIERS, DEBUG), iteration(NULL), d_iteration(NULL), total_queued(0), done(NULL), d_done(NULL), preComputed(false)
  {
    cudaMallocHost(&m_hostMappedValue, sizeof(SizeT), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&m_deviceMappedValue, m_hostMappedValue, 0);

    cudaMalloc((void**) &d_frontier_size, 2 * sizeof(SizeT));
    cudaMallocHost((void**) &h_edge_frontier_size_pinned, 2 * sizeof(SizeT));
    cudaMalloc((void**) &d_edge_frontier_size, 2 * sizeof(SizeT));

    if (util::B40CPerror(cudaMemset(d_edge_frontier_size, 0, 2 * sizeof(SizeT)), "CsrProblem cudaMemset d_edge_frontier_size failed", __FILE__, __LINE__)) exit(1);
  }

  /**
   * Destructor
   */
  virtual ~EnactorVertexCentric()
  {
    if (iteration)
    {
      util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorVertexCentric cudaFreeHost iteration failed", __FILE__, __LINE__);
    }
    if (done)
    {
      util::B40CPerror(cudaFreeHost((void *) done), "EnactorVertexCentric cudaFreeHost done failed", __FILE__, __LINE__);

      util::B40CPerror(cudaEventDestroy(throttle_event), "EnactorVertexCentric cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
    }
  }

  /**
   * Obtain statistics about the last BFS search enacted
   */
  void GetStatistics(long long &total_queued, VertexId &search_depth, double &avg_duty)
  {
    cudaThreadSynchronize();

    total_queued = this->total_queued;
    search_depth = this->iteration[0] - 1;

    avg_duty = (total_lifetimes > 0) ? double(total_runtimes) / total_lifetimes : 0.0;
  }

  struct bitcount
  {
    __host__ __device__ int operator()(const unsigned char& n1, const unsigned char& n2) const
    {
      int count1 = 0;
      int count2 = 0;
      unsigned char n11 = n1;
      unsigned char n22 = n2;
      while (n11)
      {
        count1 += n11 & 1;
        n11 >>= 1;
      }

      while (n22)
      {
        count2 += n22 & 1;
        n22 >>= 1;
      }
      return count1 + count2;
    }
  };

  struct degree_iterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    VertexId *offsets;
    VertexId *active_vertices;

    __device__
    int operator[](VertexId i) const
    {
      VertexId vertex_id = active_vertices[i];
      int degree = offsets[vertex_id + 1] - offsets[vertex_id];
      return degree == 0 ? 1 : degree;
    }

    __device__ degree_iterator operator +(int i) const
    {
      return degree_iterator(offsets, active_vertices + i);
    }

    __device__ __host__ degree_iterator(VertexId *offsets, VertexId *active_vertices) :
        offsets(offsets), active_vertices(active_vertices)
    {
    }
    ;
  };

  struct EdgeCountIterator: public std::iterator<std::input_iterator_tag, int>
  {
    int *m_offsets;
    int *m_active;

    __host__ __device__ EdgeCountIterator(int *offsets, int *active) :
        m_offsets(offsets), m_active(active)
    {
    }
    ;

    __device__
    int operator[](int i) const
    {
      int active = m_active[i];
      return max(m_offsets[active + 1] - m_offsets[active], 1);
    }

    __device__ EdgeCountIterator operator +(int i) const
    {
      return EdgeCountIterator(m_offsets, m_active + i);
    }
  };

  struct ReduceFunctor: std::binary_function<GatherType, GatherType, GatherType>
  {

    __device__ GatherType operator()(const GatherType &left, const GatherType & right)
    {
      typename Program::gather_sum gather_sum_functor;
      return gather_sum_functor(left, right);
    }
  };

  struct PredicatedEdgeCountIterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    VertexId *m_offsets;
    VertexId *m_active;
    typename Program::VertexType &m_vertex_list;
    typename Program::EdgeType &m_edge_list;
    char *m_predicates;

    __host__ __device__ PredicatedEdgeCountIterator(VertexId *offsets, VertexId *active, char * predicates, typename Program::VertexType &vertex_list, typename Program::EdgeType & edge_list) :
        m_offsets(offsets), m_active(active), m_predicates(predicates), m_vertex_list(vertex_list), m_edge_list(edge_list)
    {
    }
    ;

    __device__ VertexId operator[](VertexId i) const
    {
      VertexId active = m_active[i];
      typename Program::expand_vertex expand_vertex_functor;
      bool changed = expand_vertex_functor(active, m_predicates[active], m_vertex_list, m_edge_list);
      return changed ? m_offsets[active + 1] - m_offsets[active] : 0;
    }

    __device__ PredicatedEdgeCountIterator operator +(VertexId i) const
    {
      return PredicatedEdgeCountIterator(m_offsets, m_active + i, m_predicates, m_vertex_list, m_edge_list);
    }
  };

  struct ActivateGatherIterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    VertexId *m_offsets;
    VertexId *m_active;

    __host__ __device__ ActivateGatherIterator(VertexId* offsets, VertexId * active) :
        m_offsets(offsets), m_active(active)
    {
    }
    ;

    __device__ VertexId operator [](VertexId i)
    {
      return m_offsets[m_active[i]];
    }

    __device__ ActivateGatherIterator operator +(VertexId i) const
    {
      return ActivateGatherIterator(m_offsets, m_active + i);
    }
  };

  struct ActivateOutputIterator
  {
    int* m_flags;

    __host__ __device__ ActivateOutputIterator(int* flags) :
        m_flags(flags)
    {
    }

    __device__ ActivateOutputIterator& operator[](VertexId i)
    {
      return *this;
    }

    __device__
    void operator =(VertexId dst)
    {
      m_flags[dst] = 1;
    }

    __device__ ActivateOutputIterator operator +(VertexId i)
    {
      return ActivateOutputIterator(m_flags);
    }
  };

  struct reduce_functor: public std::binary_function<GatherType, GatherType, GatherType>
  {
    __device__ GatherType operator()(GatherType left, GatherType right)
    {
      typename Program::gather_sum gather_sum_functor;
      return gather_sum_functor(left, right);
    }
  };

  struct ReduceOutputIterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    GatherType *m_gather;
    VertexId *m_active;

    __host__ __device__ ReduceOutputIterator(GatherType *gatherTmp, VertexId * active) :
        m_gather(gatherTmp), m_active(active)
    {
    }
    ;

    __device__ GatherType& operator[](VertexId i) const
    {
      VertexId active = m_active[i];
      return m_gather[active];
    }

    __device__ ReduceOutputIterator operator +(VertexId i) const
    {
      return ReduceOutputIterator(m_gather, m_active + i);
    }

    __device__ ReduceOutputIterator& operator +=(const VertexId i)
    {
      m_active += i;
      return *this;
    }
  };

  struct changed_degree_iterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    VertexId *offsets;
    VertexId *active_vertices;
    char *changed;
    typename Program::VertexType &m_vertex_list;
    typename Program::EdgeType &m_edge_list;

    __device__ VertexId operator[](VertexId i) const
    {
      VertexId vertex_id = active_vertices[i];
      typename Program::expand_vertex expand_vertex_functor;
      bool changedv = expand_vertex_functor(vertex_id, changed[vertex_id], m_vertex_list, m_edge_list);
      return changedv ? offsets[vertex_id + 1] - offsets[vertex_id] : 0;
    }

    __device__ changed_degree_iterator operator +(VertexId i) const
    {
      return changed_degree_iterator(offsets, active_vertices + i, changed, m_vertex_list, m_edge_list);
    }

    __device__ __host__ changed_degree_iterator(VertexId *offsets, VertexId *active_vertices, char * changed, typename Program::VertexType &vertex_list, typename Program::EdgeType &edge_list) :
        offsets(offsets), active_vertices(active_vertices), changed(changed), m_vertex_list(vertex_list), m_edge_list(edge_list)
    {
    }
    ;
  };

  struct scatter_iterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    int* frontier_flags;

    __device__ scatter_iterator& operator[](VertexId i)
    {
      return *this;
    }

    __device__
    void operator =(VertexId vertex_id)
    {
      frontier_flags[vertex_id] = 1;
    }

    __device__ scatter_iterator operator +(VertexId i)
    {
      return scatter_iterator(frontier_flags);
    }

    __device__ __host__ scatter_iterator(int *frontier_flags) :
        frontier_flags(frontier_flags)
    {
    }
  };

  struct reduce_iterator: public std::iterator<std::input_iterator_tag, VertexId>
  {
    GatherType *gather;
    VertexId *active_vertices;

    __device__ GatherType& operator[](VertexId i) const
    {
      VertexId active = active_vertices[i];
      return gather[active];
    }

    __device__ reduce_iterator operator +(VertexId i) const
    {
      return reduce_iterator(gather, active_vertices + i);
    }

    __device__ reduce_iterator& operator +=(const VertexId i)
    {
      active_vertices += i;
      return *this;
    }

    __device__ __host__ reduce_iterator(GatherType *gather, VertexId *active_vertices) :
        gather(gather), active_vertices(active_vertices)
    {
    }
    ;
  };

  void scatter_mgpu(int frontier_selector, int* d_edge_frontier_size, int num_active, typename Program::SizeT* offsets, typename Program::VertexId* active_vertices,
      typename Program::VertexId* edge_count_scan, typename Program::VertexId* indices, typename Program::VertexId* edge_frontier, typename Program::VertexType& vertex_list,
      typename Program::EdgeType& edge_list, typename Program::VertexId* d_edgeCSC_indices, typename Program::MiscType* misc_values)
  {

    const int NT = 128;
    const int VT = 7;
    typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
    int2 launch = Tuning::GetLaunchParams(*m_mgpuContext);
    int move_count = edge_frontier_size;

    int NV = launch.x * launch.y;
    int numBlocks = MGPU_DIV_UP(move_count + num_active, NV);

    MGPU_MEM(int)partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper > (
        mgpu::counting_iterator<int>(0), move_count, edge_count_scan,
        num_active, NV, 0, mgpu::less<int>(), *m_mgpuContext);

    vertex_centric::mgpukernel::kernel_scatter_mgpu<Tuning, NT, VT, Program><<<
    numBlocks, launch.x, 0, m_mgpuContext->Stream()>>>(
        frontier_selector, move_count, num_active, d_edge_frontier_size,
        offsets, active_vertices, edge_count_scan, indices,
        partitionsDevice->get(), edge_frontier, vertex_list, edge_list,
        d_edgeCSC_indices, misc_values);

    //      MGPU_SYNC_CHECK("KernelIntervalMove");
//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
    __FILE__, __LINE__) != cudaSuccess) throw std::exception();
  }

  template<typename PredIt, typename OutputIt>
  void copy_if_mgpu(int num, PredIt pred, OutputIt output, int *d_total, int *h_total, mgpu::ContextPtr mgpuContext)
  {

    MGPU_MEM(int)d_map = mgpuContext->Malloc<int>(num);

    mgpu::Scan<mgpu::MgpuScanTypeExc > (pred
        , num
        , 0
        , mgpu::plus<int>()
        , d_total
        , h_total
        , d_map->get()
        , *mgpuContext);

    int threads = 256;
    int blocks = min((num + threads - 1) / threads, 256);

    mgpu::counting_iterator<int> input(0);

    vertex_centric::mgpukernel::kernel_copy_if << <blocks, threads >> >(input, num,
        pred,
        d_map->get(),
        output);
  }

  void expand_mgpu(typename CsrProblem::GraphSlice *graph_slice, int &selector, const int frontier_selector, const int directed)
  {
    int zero = 0;
    //counts = m_applyRet ? outEdgeCount[ m_active ] : 0
    //first scan the numbers of edges from the active list

    //      int* test_vid2 = new int[100];
    //      cudaMemcpy(test_vid2, graph_slice->frontier_queues.d_keys[selector ^ 1], 100 * sizeof(int), cudaMemcpyDeviceToHost);
    //      printf("frontier before scatter mgpu: ");
    //      for (int i = 0; i < 100; ++i)
    //      {
    //        printf("%d, ", test_vid2[i]);
    //      }
    //      printf("\n");
    //      delete[] test_vid2;
    //
    //      char*test_vid3 = new char[graph_slice->nodes];
    //      cudaMemcpy(test_vid3, graph_slice->d_changed, graph_slice->nodes * sizeof(char), cudaMemcpyDeviceToHost);
    //      printf("changed before scatter mgpu: ");
    //      for (int i = 0; i < graph_slice->nodes; ++i)
    //      {
    //        printf("%d, ", test_vid3[i]);
    //      }
    //      printf("\n");
    //      delete[] test_vid3;

    //      double startscatter = 0.0, startactive = 0.0, endscatter = 0.0, endactive = 0.0;
    //      startscatter = omp_get_wtime();

    //Gathers the dst vertex ids from m_dsts and writes a true for each
    //dst vertex into m_activeFlags

    if (Program::allow_duplicates)
    {

      if (directed == 0)
      {
        //          printf("Expand_mgpu:Dup:All\n");
        PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
        mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
            , frontier_size
            , 0
            , mgpu::plus<VertexId > ()
            , &d_edge_frontier_size[frontier_selector]
            , &edge_frontier_size
            , graph_slice->d_edgeCountScan
            , *m_mgpuContext);
        //				int rank_id;
        //				MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
        //				printf("rank_id=%d, frontier_size=%d, edge_frontier_size=%d\n", rank_id, frontier_size, edge_frontier_size);
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
        //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

        if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof (SizeT), cudaMemcpyDeviceToHost),
                "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
                __LINE__))
        exit(1);
        //
        //      printf("m_nActive = %d\n", frontier_size);
        //      printf("nActiveEdges = %d\n", edge_frontier_size);
        //          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
        //          {
        //            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
        //            exit(1);
        //          }
        //        SYNC_CHECK();

        scatter_mgpu(frontier_selector,
            d_edge_frontier_size,
            frontier_size,
            graph_slice->d_row_offsets,
            graph_slice->frontier_queues.d_keys[selector ^ 1],
            graph_slice->d_edgeCountScan,
            graph_slice->d_column_indices,
            graph_slice->frontier_queues.d_keys[selector],
            graph_slice->vertex_list,
            graph_slice->edge_list,
            NULL,
            graph_slice->frontier_queues.d_values[0]);
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();

        int edge_frontier_size1 = edge_frontier_size;

        PredicatedEdgeCountIterator ecIterator2(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
        mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator2
            , frontier_size
            , 0
            , mgpu::plus<VertexId > ()
            , &d_edge_frontier_size[frontier_selector]
            , &edge_frontier_size
            , graph_slice->d_edgeCountScan
            , *m_mgpuContext);
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();

        //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

        //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
        //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
        //          __LINE__))
        //        exit(1);
        //
        //      printf("m_nActive = %d\n", frontier_size);
        //          printf("edge_frontier_size1 = %d\n", edge_frontier_size1);
        //          printf("edge_frontier_size2 = %d\n", edge_frontier_size);
        if (edge_frontier_size1 + edge_frontier_size >= graph_slice->frontier_elements[selector])
        {
          printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size1 + edge_frontier_size, graph_slice->frontier_elements[selector]);
          exit(1);
        }
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
        scatter_mgpu(frontier_selector,
            d_edge_frontier_size,
            frontier_size,
            graph_slice->d_column_offsets,
            graph_slice->frontier_queues.d_keys[selector ^ 1],
            graph_slice->d_edgeCountScan,
            graph_slice->d_row_indices,
            graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            graph_slice->d_edgeCSC_indices,
            graph_slice->frontier_queues.d_values[0] + edge_frontier_size1);

        //          cudaMemcpy(
        //              graph_slice->frontier_queues.d_keys[selector]
        //              + edge_frontier_size,
        //              graph_slice->frontier_queues.d_keys[2],
        //              edge_frontier_size2 * sizeof(VertexId),
        //              cudaMemcpyDeviceToDevice);
        //
        //          cudaMemcpy(
        //              graph_slice->frontier_queues.d_values[0]
        //              + edge_frontier_size,
        //              graph_slice->frontier_queues.d_values[1],
        //              edge_frontier_size2 * sizeof(VertexId),
        //              cudaMemcpyDeviceToDevice);

        edge_frontier_size += edge_frontier_size1;
        //          printf("edge_frontier_final = %d\n", edge_frontier_size);
        cudaMemcpy(
            &d_edge_frontier_size[frontier_selector],
            &edge_frontier_size,
            sizeof (int),
            cudaMemcpyHostToDevice);
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();

        //          VertexId* test_vid = new VertexId[edge_frontier_size];
        //          cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector], edge_frontier_size * sizeof(VertexId), cudaMemcpyDeviceToHost);
        //          printf("Frontier after expansion: ");
        //          for (int i = 0; i < edge_frontier_size; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

      }
      else
      {
        if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
        {
          //					SYNC_CHECK();
          edge_frontier_size = 0;
          PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
//					SYNC_CHECK();
          if (util::B40CPerror(cudaDeviceSynchronize(), "Error after ecIterator"
                  __FILE__,
                  __LINE__))
          throw std::exception();

          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              //							, (int)NULL
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);

          //					int rank_id;
          //					MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
//            printf("frontier_size=%d, edge_frontier_size=%d\n", frontier_size, edge_frontier_size);
          if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                  __FILE__,
                  __LINE__) != cudaSuccess)
          throw std::exception();

          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //      printf("nActiveEdges = %d\n", edge_frontier_size);

          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }

          scatter_mgpu(frontier_selector,
              d_edge_frontier_size,
              frontier_size,
              graph_slice->d_row_offsets,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->d_edgeCountScan,
              graph_slice->d_column_indices,
              graph_slice->frontier_queues.d_keys[selector],
              graph_slice->vertex_list,
              graph_slice->edge_list,
              NULL,
              graph_slice->frontier_queues.d_values[0]);
          if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                  __FILE__,
                  __LINE__) != cudaSuccess)
          throw std::exception();
        }
        else if (Program::expandOverEdges() == EXPAND_IN_EDGES)
        {
          PredicatedEdgeCountIterator ecIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);
          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //      printf("nActiveEdges = %d\n", edge_frontier_size);
          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }
          //        SYNC_CHECK();

          scatter_mgpu(frontier_selector,
              d_edge_frontier_size,
              frontier_size,
              graph_slice->d_column_offsets,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->d_edgeCountScan,
              graph_slice->d_row_indices,
              graph_slice->frontier_queues.d_keys[selector],
              graph_slice->vertex_list,
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->frontier_queues.d_values[0]);

        }
        else if (Program::expandOverEdges() == EXPAND_ALL_EDGES)
        {
          //          printf("Expand_mgpu:Dup:All\n");
          PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);
          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //      printf("nActiveEdges = %d\n", edge_frontier_size);
          //          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          //          {
          //            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
          //            exit(1);
          //          }
          //        SYNC_CHECK();

          scatter_mgpu(frontier_selector,
              d_edge_frontier_size,
              frontier_size,
              graph_slice->d_row_offsets,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->d_edgeCountScan,
              graph_slice->d_column_indices,
              graph_slice->frontier_queues.d_keys[selector],
              graph_slice->vertex_list,
              graph_slice->edge_list,
              NULL,
              graph_slice->frontier_queues.d_values[0]);

          int edge_frontier_size1 = edge_frontier_size;

          PredicatedEdgeCountIterator ecIterator2(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator2
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);

          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //          printf("edge_frontier_size1 = %d\n", edge_frontier_size1);
          //          printf("edge_frontier_size2 = %d\n", edge_frontier_size);
          if (edge_frontier_size1 + edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size1 + edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }
          //        SYNC_CHECK();
          scatter_mgpu(frontier_selector,
              d_edge_frontier_size,
              frontier_size,
              graph_slice->d_column_offsets,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->d_edgeCountScan,
              graph_slice->d_row_indices,
              graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,
              graph_slice->vertex_list,
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->frontier_queues.d_values[0] + edge_frontier_size1);

          //          cudaMemcpy(
          //              graph_slice->frontier_queues.d_keys[selector]
          //              + edge_frontier_size,
          //              graph_slice->frontier_queues.d_keys[2],
          //              edge_frontier_size2 * sizeof(VertexId),
          //              cudaMemcpyDeviceToDevice);
          //
          //          cudaMemcpy(
          //              graph_slice->frontier_queues.d_values[0]
          //              + edge_frontier_size,
          //              graph_slice->frontier_queues.d_values[1],
          //              edge_frontier_size2 * sizeof(VertexId),
          //              cudaMemcpyDeviceToDevice);

          edge_frontier_size += edge_frontier_size1;
          //          printf("edge_frontier_final = %d\n", edge_frontier_size);
          cudaMemcpy(
              &d_edge_frontier_size[frontier_selector],
              &edge_frontier_size,
              sizeof (int),
              cudaMemcpyHostToDevice);

          //          VertexId* test_vid = new VertexId[edge_frontier_size];
          //          cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector], edge_frontier_size * sizeof(VertexId), cudaMemcpyDeviceToHost);
          //          printf("Frontier after expansion: ");
          //          for (int i = 0; i < edge_frontier_size; ++i)
          //          {
          //            printf("%d, ", test_vid[i]);
          //          }
          //          printf("\n");
          //          delete[] test_vid;

        }
        else
        {
          cout << "Invalid expandOverEdges value!" << endl;
          exit(1);
        }
      }

    }
    else
    {
      if (directed == 0)
      {
        //          int* test_vid = new int[graph_slice->nodes];
        //          cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(char), cudaMemcpyDeviceToHost);
        //          printf("d_changed before scatter out mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

        //          int* test_vid = new int[graph_slice->nodes];
        //          cudaMemcpy(test_vid, graph_slice->d_active_flags, graph_slice->nodes * sizeof(int), cudaMemcpyDeviceToHost);
        //          printf("flag before scatter out mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

        PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
        mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
            , frontier_size
            , 0
            , mgpu::plus<VertexId > ()
            , &d_edge_frontier_size[frontier_selector]
            , &edge_frontier_size
            , graph_slice->d_edgeCountScan
            , *m_mgpuContext);

        //          printf("edge_frontier_size1=%d\n", edge_frontier_size);

        IntervalGather(edge_frontier_size,
            ActivateGatherIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
            graph_slice->d_edgeCountScan,
            frontier_size,
            graph_slice->d_column_indices,
            ActivateOutputIterator(graph_slice->d_active_flags),
            *m_mgpuContext);

        //          test_vid = new int[graph_slice->nodes];
        //          cudaMemcpy(test_vid, graph_slice->d_active_flags, graph_slice->nodes * sizeof(int), cudaMemcpyDeviceToHost);
        //          printf("flag after scatter out mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

        PredicatedEdgeCountIterator ecIterator2(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
        mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator2
            , frontier_size
            , 0
            , mgpu::plus<VertexId > ()
            , &d_edge_frontier_size[frontier_selector]
            , &edge_frontier_size
            , graph_slice->d_edgeCountScan
            , *m_mgpuContext);

        //          printf("edge_frontier_size2=%d\n", edge_frontier_size);

        IntervalGather(edge_frontier_size,
            ActivateGatherIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
            graph_slice->d_edgeCountScan,
            frontier_size,
            graph_slice->d_row_indices,
            ActivateOutputIterator(graph_slice->d_active_flags),
            *m_mgpuContext);

        //          test_vid = new int[graph_slice->nodes];
        //          cudaMemcpy(test_vid, graph_slice->d_active_flags, graph_slice->nodes * sizeof(int), cudaMemcpyDeviceToHost);
        //          printf("flag after scatter in mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

        copy_if_mgpu(graph_slice->nodes,
            graph_slice->d_active_flags,
            graph_slice->frontier_queues.d_keys[selector],
            &d_frontier_size[frontier_selector],
            (int*)NULL,
            m_mgpuContext);

        //using memset is faster?
        if (util::B40CPerror(cudaMemcpy(&d_edge_frontier_size[frontier_selector], &zero, sizeof (SizeT), cudaMemcpyHostToDevice),
                "CsrProblem reset to zero d_edge_frontier_size failed", __FILE__,
                __LINE__))
        exit(1);

      }
      else
      {
        if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
        {
          PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);
          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //      printf("nActiveEdges = %d\n", edge_frontier_size);
          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }
          //        SYNC_CHECK();
          if (util::B40CPerror(cudaMemcpy(&d_edge_frontier_size[frontier_selector], &zero, sizeof (SizeT), cudaMemcpyHostToDevice),
                  "CsrProblem reset to zero d_edge_frontier_size failed", __FILE__,
                  __LINE__))
          exit(1);

          IntervalGather(edge_frontier_size,
              ActivateGatherIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
              graph_slice->d_edgeCountScan,
              frontier_size,
              graph_slice->d_column_indices,
              ActivateOutputIterator(graph_slice->d_active_flags),
              *m_mgpuContext);

          copy_if_mgpu(graph_slice->nodes,
              graph_slice->d_active_flags,
              graph_slice->frontier_queues.d_keys[selector],
              &d_frontier_size[frontier_selector],
              (int*)NULL,
              m_mgpuContext);
        }
        else if (Program::expandOverEdges() == EXPAND_IN_EDGES)
        {
          PredicatedEdgeCountIterator ecIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);
          //      cudaDeviceSynchronize();      //terminate scan kernel before read out m_hostMappedValue

          //      if (util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
          //          "CsrProblem cudaMemcpy d_edge_frontier_size failed", __FILE__,
          //          __LINE__))
          //        exit(1);
          //
          //      printf("m_nActive = %d\n", frontier_size);
          //      printf("nActiveEdges = %d\n", edge_frontier_size);
          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }
          //        SYNC_CHECK();
          if (util::B40CPerror(cudaMemcpy(&d_edge_frontier_size[frontier_selector], &zero, sizeof (SizeT), cudaMemcpyHostToDevice),
                  "CsrProblem reset to zero d_edge_frontier_size failed", __FILE__,
                  __LINE__))
          exit(1);

          IntervalGather(edge_frontier_size,
              ActivateGatherIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
              graph_slice->d_edgeCountScan,
              frontier_size,
              graph_slice->d_row_indices,
              ActivateOutputIterator(graph_slice->d_active_flags),
              *m_mgpuContext);

          copy_if_mgpu(graph_slice->nodes,
              graph_slice->d_active_flags,
              graph_slice->frontier_queues.d_keys[selector],
              &d_frontier_size[frontier_selector],
              (int*)NULL,
              m_mgpuContext);

        }
        else if (Program::expandOverEdges() == EXPAND_ALL_EDGES)
        {

          PredicatedEdgeCountIterator ecIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);

          IntervalGather(edge_frontier_size,
              ActivateGatherIterator(graph_slice->d_row_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
              graph_slice->d_edgeCountScan,
              frontier_size,
              graph_slice->d_column_indices,
              ActivateOutputIterator(graph_slice->d_active_flags),
              *m_mgpuContext);
          //
          //          int* test_vid = new int[graph_slice->nodes];
          //          cudaMemcpy(test_vid, graph_slice->d_active_flags, graph_slice->nodes * sizeof(int), cudaMemcpyDeviceToHost);
          //          printf("flag after scatter out mgpu: ");
          //          for (int i = 0; i < graph_slice->nodes; ++i)
          //          {
          //            printf("%d, ", test_vid[i]);
          //          }
          //          printf("\n");
          //          delete[] test_vid;

          PredicatedEdgeCountIterator ecIterator2(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_changed, graph_slice->vertex_list, graph_slice->edge_list);
          mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, VertexId, mgpu::plus<VertexId>, VertexId*>(ecIterator2
              , frontier_size
              , 0
              , mgpu::plus<VertexId > ()
              , &d_edge_frontier_size[frontier_selector]
              , &edge_frontier_size
              , graph_slice->d_edgeCountScan
              , *m_mgpuContext);

          IntervalGather(edge_frontier_size,
              ActivateGatherIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]),
              graph_slice->d_edgeCountScan,
              frontier_size,
              graph_slice->d_row_indices,
              ActivateOutputIterator(graph_slice->d_active_flags),
              *m_mgpuContext);

          //          test_vid = new int[graph_slice->nodes];
          //          cudaMemcpy(test_vid, graph_slice->d_active_flags, graph_slice->nodes * sizeof(int), cudaMemcpyDeviceToHost);
          //          printf("flag after scatter in mgpu: ");
          //          for (int i = 0; i < graph_slice->nodes; ++i)
          //          {
          //            printf("%d, ", test_vid[i]);
          //          }
          //          printf("\n");
          //          delete[] test_vid;

          copy_if_mgpu(graph_slice->nodes,
              graph_slice->d_active_flags,
              graph_slice->frontier_queues.d_keys[selector],
              &d_frontier_size[frontier_selector],
              (int*)NULL,
              m_mgpuContext);

          if (edge_frontier_size >= graph_slice->frontier_elements[selector])
          {
            printf("queue size: %d, Frontier queue overflow (%d).  Please increase queue-sizing factor.\n", edge_frontier_size, graph_slice->frontier_elements[selector]);
            exit(1);
          }

          //using memset is faster?
          if (util::B40CPerror(cudaMemcpy(&d_edge_frontier_size[frontier_selector], &zero, sizeof (SizeT), cudaMemcpyHostToDevice),
                  "CsrProblem reset to zero d_edge_frontier_size failed", __FILE__,
                  __LINE__))
          exit(1);

        }
        else
        {
          cout << "Invalid expandOverEdges value!" << endl;
          exit(1);
        }
        SYNC_CHECK();
      }
      CHECK(cudaMemset(graph_slice->d_active_flags, 0, sizeof (int)* graph_slice->nodes));
    }
    //      cudaDeviceSynchronize();
//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();
    selector ^= 1;

    //      endscatter = omp_get_wtime();

    //      int tmp_frontier_size;
    //      if (util::B40CPerror(cudaMemcpy(&tmp_frontier_size, &d_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
    //              "CsrProblem cudaMemcpy tmp_frontier_size failed", __FILE__,
    //              __LINE__))
    //      exit(1);
    //      printf("frontier_size after activation: %d\n", tmp_frontier_size);

    //      int* test_vid = new int[tmp_frontier_size];
    //      cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], tmp_frontier_size * sizeof(int), cudaMemcpyDeviceToHost);
    //      printf("frontier after scatter mgpu: ");
    //      for (int i = 0; i < tmp_frontier_size; ++i)
    //      {
    //        printf("%d, ", test_vid[i]);
    //      }
    //      printf("\n");
    //      delete[] test_vid;

    //      startactive = omp_get_wtime();
    //      selector ^= 1;

    //      cudaDeviceSynchronize();
    //      endactive = omp_get_wtime();
    //      selector ^= 1;

    //
    //      test_vid2 = new int[100];
    //      cudaMemcpy(test_vid2, graph_slice->d_edgeCountScan, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    //      printf("d_edgeCountScan after scatter: ");
    //      for (int i = 0; i < 100; ++i)
    //      {
    //        printf("%d, ", test_vid2[i]);
    //      }
    //      printf("\n");
    //      delete[] test_vid2;
    //
    //      printf("Scatter time: %f ms\n", (endscatter - startscatter) * 1000.0);
    //      printf("Activate time: %f ms\n", (endactive - startactive) * 1000.0);
  }

  void apply_mgpu(typename CsrProblem::GraphSlice *graph_slice, int selector, int rank_id)
  {
		if(rank_id == 0)
		{
		  DataType* test_vid = new DataType[graph_slice->nodes];
			if(util::B40CPerror(
							cudaMemcpy(test_vid, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof(DataType), cudaMemcpyDeviceToHost),
							"gather::reset_changed Kernel failed ",
							__FILE__, __LINE__))
			exit(1);

			printf("rank_id=%d, d_labels before apply: ", rank_id);
//			for (int i = 0; i < graph_slice->nodes; ++i)
			for (int i = 0; i < 20; ++i)
			{
				printf("%f, ", test_vid[i]);
			}
			printf("\n");
			delete[] test_vid;

			test_vid = new DataType[graph_slice->nodes];
			if(util::B40CPerror(
							cudaMemcpy(test_vid, m_reduceResult, graph_slice->nodes * sizeof(DataType), cudaMemcpyDeviceToHost),
							"m_reduceResult failed ",
							__FILE__, __LINE__))
			exit(1);

			printf("rank_id=%d, m_reduceResult before apply: ", rank_id);
//			for (int i = 0; i < graph_slice->nodes; ++i)
			for (int i = 0; i < 20; ++i)
			{
				printf("%f, ", test_vid[i]);
			}
			printf("\n");
			delete[] test_vid;
		}

    int nThreadsPerBlock = 128;
    SizeT nBlocks = MGPU_DIV_UP(frontier_size,
        nThreadsPerBlock);

//		printf("before apply: rank_id=%d, frontier_size=%d, nBlocks=%d, iteration[0]=%d\n", rank_id, frontier_size, nBlocks, iteration[0]);
    vertex_centric::mgpukernel::kernel_apply<Program> <<<nBlocks,nThreadsPerBlock>>>(rank_id,
        iteration[0],
        frontier_size,
        graph_slice->frontier_queues.d_keys[2],
        m_reduceResult,
        graph_slice->vertex_list,
        graph_slice->edge_list,
        graph_slice->d_changed);

    if (util::B40CPerror(cudaDeviceSynchronize(), "Error after apply"
            __FILE__,
            __LINE__))
    throw std::exception();

//		SYNC_CHECK();

		if(rank_id == 0)
		{
		  DataType* test_vid = new DataType[graph_slice->nodes];
			cudaMemcpy(test_vid, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof(DataType), cudaMemcpyDeviceToHost);
			printf("rank_id=%d, d_labels after apply: ", rank_id);
//			for (int i = 0; i < graph_slice->nodes; ++i)
			  for (int i = 0; i < 20; ++i)
			{
				printf("%f, ", test_vid[i]);
			}
			printf("\n");
			delete[] test_vid;

			char* test_vid2 = new char[graph_slice->nodes];
			cudaMemcpy(test_vid2, graph_slice->d_changed, graph_slice->nodes * sizeof(char), cudaMemcpyDeviceToHost);
			printf("rank_id=%d, changed after apply: ", rank_id);
//			for (int i = 0; i < graph_slice->nodes; ++i)
			for (int i = 0; i < 20; ++i)
			{
				printf("%d, ", test_vid2[i]);
			}
			printf("\n");
			delete[] test_vid2;
		}
  }

//	void gather_mgpu(typename CsrProblem::GraphSlice *graph_slice, const int selector, const int directed)
//	{
//		if (directed == 0)
//		//      if(0)
//		{
//			//          printf("Gather all edges --- gather in!\n");
//			EdgeCountIterator ecIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//			mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//					ecIterator,
//					frontier_size,
//					0,
//					mgpu::plus<int>(),
//					m_deviceMappedValue,
//					(int *)NULL,
//					graph_slice->d_edgeCountScan,
//					*m_mgpuContext);
//
//			//          int n_active_edges = *m_hostMappedValue;
//			int n_active_edges;
//			cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//					sizeof (int),
//					cudaMemcpyDeviceToHost);
//
//			SYNC_CHECK();
//			//      printf("n_active_edges = %d\n", n_active_edges);
//			//        printf("Gather all: int: n_active_edges = %d\n", n_active_edges);
//
//			const int nThreadsPerBlock = 128;
//			MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//			(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//					nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//			SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size, nThreadsPerBlock);
//			//          dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//			vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//			nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//					frontier_size,
//					graph_slice->frontier_queues.d_keys[selector ^ 1],
//					nBlocks,
//					n_active_edges,
//					graph_slice->d_edgeCountScan,
//					partitions->get(),
//					graph_slice->d_column_offsets,
//					graph_slice->d_row_indices,
//					graph_slice->vertex_list,
//					graph_slice->edge_list,
//					NULL,
//					graph_slice->m_gatherDstsTmp,
//					graph_slice->m_gatherMapTmp);
//
//			SYNC_CHECK();
//
//			//        VertexId* test_vid3 = new VertexId[n_active_edges];
//			//        cudaMemcpy(test_vid3, graph_slice->m_gatherDstsTmp,
//			//            n_active_edges * sizeof(VertexId),
//			//            cudaMemcpyDeviceToHost);
//			//        printf("graph_slice->m_gatherDstsTmp after gather-mgpu: ");
//			//        for (int i = 0; i < n_active_edges; ++i)
//			//        {
//			//          printf("%d, ", test_vid3[i]);
//			//        }
//			//        printf("\n");
//			//        delete[] test_vid3;
//			//
//			//        GatherType* test_vid2 = new GatherType[n_active_edges];
//			//        cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp,
//			//            n_active_edges * sizeof(GatherType),
//			//            cudaMemcpyDeviceToHost);
//			//        printf("graph_slice->m_gatherMapTmp after gather-mgpu: ");
//			//        for (int i = 0; i < n_active_edges; ++i)
//			//        {
//			//          printf("%f, ", test_vid2[i]);
//			//        }
//			//        printf("\n");
//			//        delete[] test_vid2;
//
//			mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//					, graph_slice->m_gatherMapTmp
//					, n_active_edges
//					, Program::INIT_VALUE
//					, ReduceFunctor()
//					, mgpu::equal_to<VertexId > ()
//					, (VertexId *)NULL
//					, ReduceOutputIterator(graph_slice->m_gatherTmp1, graph_slice->frontier_queues.d_keys[selector ^ 1])
//					, NULL
//					, NULL
//					, *m_mgpuContext);
//
//			SYNC_CHECK();
//
//			//          GatherType* test_vid1 = new GatherType[graph_slice->nodes];
//			//          cudaMemcpy(test_vid1, graph_slice->m_gatherTmp1,
//			//              graph_slice->nodes * sizeof(GatherType),
//			//              cudaMemcpyDeviceToHost);
//			//          printf("graph_slice->graph_slice->m_gatherTmp after gather-mgpu: ");
//			//          for (int i = 0; i < graph_slice->nodes; ++i)
//			//          {
//			//            printf("%f, ", test_vid1[i]);
//			//          }
//			//          printf("\n");
//			//          delete[] test_vid1;
//
//			//        printf("Gather all edges --- gather out!\n");
//
//			EdgeCountIterator ecIterator2(graph_slice->d_row_offsets,
//					graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//			mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//					ecIterator2,
//					frontier_size,
//					0,
//					mgpu::plus<int>(),
//					m_deviceMappedValue,
//					(int *)NULL,
//					graph_slice->d_edgeCountScan,
//					*m_mgpuContext);
//
//			SYNC_CHECK();
//
//			//          n_active_edges = *m_hostMappedValue;
//			cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//					sizeof (int),
//					cudaMemcpyDeviceToHost);
//			//        printf("Gather all: out: n_active_edges = %d\n", n_active_edges);
//
//			partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//			(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//					nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//			nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
//					nThreadsPerBlock);
//			//          grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//			vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//			nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//					frontier_size,
//					graph_slice->frontier_queues.d_keys[selector ^ 1],
//					nBlocks,
//					n_active_edges,
//					graph_slice->d_edgeCountScan,
//					partitions->get(),
//					graph_slice->d_row_offsets,
//					graph_slice->d_column_indices,
//					graph_slice->vertex_list,
//					graph_slice->edge_list,
//					graph_slice->d_edgeCSC_indices,
//					graph_slice->m_gatherDstsTmp,
//					graph_slice->m_gatherMapTmp);
//
//			SYNC_CHECK();
//
//			//          test_vid3 = new VertexId[n_active_edges];
//			//          cudaMemcpy(test_vid3, graph_slice->m_gatherDstsTmp,
//			//              n_active_edges * sizeof(VertexId),
//			//              cudaMemcpyDeviceToHost);
//			//          printf("graph_slice->m_gatherDstsTmp after gather-mgpu: ");
//			//          for (int i = 0; i < n_active_edges; ++i)
//			//          {
//			//            printf("%d, ", test_vid3[i]);
//			//          }
//			//          printf("\n");
//			//          delete[] test_vid3;
//			//
//			//          test_vid2 = new GatherType[n_active_edges];
//			//          cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp,
//			//              n_active_edges * sizeof(GatherType),
//			//              cudaMemcpyDeviceToHost);
//			//          printf("graph_slice->m_gatherMapTmp after gather-mgpu: ");
//			//          for (int i = 0; i < n_active_edges; ++i)
//			//          {
//			//            printf("%f, ", test_vid2[i]);
//			//          }
//			//          printf("\n");
//			//          delete[] test_vid2;
//
//			mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//					, graph_slice->m_gatherMapTmp
//					, n_active_edges
//					, Program::INIT_VALUE
//					, ReduceFunctor()
//					, mgpu::equal_to<VertexId > ()
//					, (VertexId *)NULL
//					, ReduceOutputIterator(graph_slice->m_gatherTmp2, graph_slice->frontier_queues.d_keys[selector ^ 1])
//					, NULL
//					, NULL
//					, *m_mgpuContext);
//
//			SYNC_CHECK();
//
//			//          test_vid1 = new GatherType[graph_slice->nodes];
//			//          cudaMemcpy(test_vid1, graph_slice->m_gatherTmp2,
//			//              graph_slice->nodes * sizeof(GatherType),
//			//              cudaMemcpyDeviceToHost);
//			//          printf("graph_slice->graph_slice->m_gatherTmp after gather-mgpu: ");
//			//          for (int i = 0; i < graph_slice->nodes; ++i)
//			//          {
//			//            printf("%f, ", test_vid1[i]);
//			//          }
//			//          printf("\n");
//			//          delete[] test_vid1;
//
//			thrust::device_ptr<GatherType> m_gatherTmp1_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp1);
//			thrust::device_ptr<GatherType> m_gatherTmp2_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp2);
//			thrust::device_ptr<GatherType> m_gatherTmp_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp);
//			typename Program::gather_sum gather_sum_functor;
//			thrust::transform(m_gatherTmp1_ptr,
//					m_gatherTmp1_ptr + graph_slice->nodes,
//					m_gatherTmp2_ptr,
//					m_gatherTmp_ptr,
//					gather_sum_functor);
//
//			//          test_vid1 = new GatherType[graph_slice->nodes];
//			//          cudaMemcpy(test_vid1, graph_slice->graph_slice->m_gatherTmp,
//			//              graph_slice->nodes * sizeof(GatherType),
//			//              cudaMemcpyDeviceToHost);
//			//          printf("final graph_slice->graph_slice->m_gatherTmp after gather-mgpu: ");
//			//          for (int i = 0; i < graph_slice->nodes; ++i)
//			//          {
//			//            printf("%f, ", test_vid1[i]);
//			//          }
//			//          printf("\n");
//			//          delete[] test_vid1;
//			//        EdgeCountIterator ecIterator(graph_slice->d_row_offsets,
//			//            graph_slice->frontier_queues.d_keys[selector ^ 1]);
//			//
//			//        mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//			//            ecIterator,
//			//            frontier_size,
//			//            0,
//			//            mgpu::plus<int>(),
//			//            m_deviceMappedValue,
//			//            (int *) NULL,
//			//            graph_slice->d_edgeCountScan,
//			//            *m_mgpuContext);
//			//
//			////        int n_active_edges = *m_hostMappedValue;
//			//        int n_active_edges;
//			//        cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//			//            sizeof(int),
//			//            cudaMemcpyDeviceToHost);
//			//        //      printf("n_active_edges = %d\n", n_active_edges);
//			//
//			//        const int nThreadsPerBlock = 128;
//			//        MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
//			//        (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//			//            nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//			//
//			//        SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
//			//            nThreadsPerBlock);
//			//        dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//			//
//			//        vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//			//        nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
//			//            frontier_size,
//			//            graph_slice->frontier_queues.d_keys[selector ^ 1],
//			//            nBlocks,
//			//            n_active_edges,
//			//            graph_slice->d_edgeCountScan,
//			//            partitions->get(),
//			//            graph_slice->d_row_offsets,
//			//            graph_slice->d_column_indices,
//			//            graph_slice->vertex_list,
//			//            graph_slice->edge_list,
//			//            NULL,
//			//            graph_slice->m_gatherDstsTmp,
//			//            graph_slice->m_gatherMapTmp);
//			//
//			//        mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//			//            , graph_slice->m_gatherMapTmp
//			//            , n_active_edges
//			//            , Program::INIT_VALUE
//			//            , ReduceFunctor()
//			//            , mgpu::equal_to<VertexId>()
//			//            , (VertexId *) NULL
//			//            , ReduceOutputIterator(graph_slice->graph_slice->m_gatherTmp, graph_slice->frontier_queues.d_keys[selector ^ 1])
//			//            , NULL
//			//            , NULL
//			//            , *m_mgpuContext);
//		}
//		else
//		{
//			if (Program::gatherOverEdges() == GATHER_IN_EDGES)
//			{
//				EdgeCountIterator ecIterator(graph_slice->d_column_offsets,
//						graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//				mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//						ecIterator,
//						frontier_size,
//						0,
//						mgpu::plus<int>(),
//						m_deviceMappedValue,
//						(int *)NULL,
//						graph_slice->d_edgeCountScan,
//						*m_mgpuContext);
//
//				//          int n_active_edges = *m_hostMappedValue;
//				int n_active_edges;
//				cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//						sizeof (int),
//						cudaMemcpyDeviceToHost);
//
//				//          printf("n_active_edges = %d, frontier_size = %d\n", n_active_edges, frontier_size);
//
//				const int nThreadsPerBlock = 128;
//				MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//				(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//						nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//				SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
//						nThreadsPerBlock);
//				dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//				vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//				nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						nBlocks,
//						n_active_edges,
//						graph_slice->d_edgeCountScan,
//						partitions->get(),
//						graph_slice->d_column_offsets,
//						graph_slice->d_row_indices,
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						NULL,
//						graph_slice->m_gatherDstsTmp,
//						graph_slice->m_gatherMapTmp);
//
//				mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//						, graph_slice->m_gatherMapTmp
//						, n_active_edges
//						, Program::INIT_VALUE
//						, ReduceFunctor()
//						, mgpu::equal_to<VertexId > ()
//						, (VertexId *)NULL
//						, ReduceOutputIterator(graph_slice->m_gatherTmp, graph_slice->frontier_queues.d_keys[selector ^ 1])
//						, NULL
//						, NULL
//						, *m_mgpuContext);
//
//				//          GatherType* test_vid2 = new GatherType[graph_slice->nodes];
//				//          cudaMemcpy(test_vid2, graph_slice->graph_slice->m_gatherTmp,
//				//              graph_slice->nodes * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->graph_slice->m_gatherTmp after gather-mgpu: ");
//				//          for (int i = 0; i < graph_slice->nodes; ++i)
//				//          {
//				//            printf("%f, ", test_vid2[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid2;
//			}
//			else if (Program::gatherOverEdges() == GATHER_OUT_EDGES)
//			{
//				EdgeCountIterator ecIterator(graph_slice->d_row_offsets,
//						graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//				mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//						ecIterator,
//						frontier_size,
//						0,
//						mgpu::plus<int>(),
//						m_deviceMappedValue,
//						(int *)NULL,
//						graph_slice->d_edgeCountScan,
//						*m_mgpuContext);
//
//				//          int n_active_edges = *m_hostMappedValue;
//				int n_active_edges;
//				cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//						sizeof (int),
//						cudaMemcpyDeviceToHost);
//				//      printf("n_active_edges = %d\n", n_active_edges);
//
//				const int nThreadsPerBlock = 128;
//				MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//				(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//						nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//				SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
//						nThreadsPerBlock);
//				dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//				vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//				nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						nBlocks,
//						n_active_edges,
//						graph_slice->d_edgeCountScan,
//						partitions->get(),
//						graph_slice->d_row_offsets,
//						graph_slice->d_column_indices,
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						graph_slice->d_edgeCSC_indices,
//						graph_slice->m_gatherDstsTmp,
//						graph_slice->m_gatherMapTmp);
//
//				mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//						, graph_slice->m_gatherMapTmp
//						, n_active_edges
//						, Program::INIT_VALUE
//						, ReduceFunctor()
//						, mgpu::equal_to<VertexId > ()
//						, (VertexId *)NULL
//						, ReduceOutputIterator(graph_slice->m_gatherTmp, graph_slice->frontier_queues.d_keys[selector ^ 1])
//						, NULL
//						, NULL
//						, *m_mgpuContext);
//			}
//			else if (Program::gatherOverEdges() == GATHER_ALL_EDGES)
//			{
//				//          printf("Gather all edges --- gather in!\n");
//				EdgeCountIterator ecIterator(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//				mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//						ecIterator,
//						frontier_size,
//						0,
//						mgpu::plus<int>(),
//						m_deviceMappedValue,
//						(int *)NULL,
//						graph_slice->d_edgeCountScan,
//						*m_mgpuContext);
//
//				//          int n_active_edges = *m_hostMappedValue;
//				int n_active_edges;
//				cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//						sizeof (int),
//						cudaMemcpyDeviceToHost);
//				//      printf("n_active_edges = %d\n", n_active_edges);
//				//          printf("Gather all: int: n_active_edges = %d\n", n_active_edges);
//
//				const int nThreadsPerBlock = 128;
//				MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//				(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//						nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//				SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size, nThreadsPerBlock);
//				//          dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//				vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//				nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						nBlocks,
//						n_active_edges,
//						graph_slice->d_edgeCountScan,
//						partitions->get(),
//						graph_slice->d_column_offsets,
//						graph_slice->d_row_indices,
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						NULL,
//						graph_slice->m_gatherDstsTmp,
//						graph_slice->m_gatherMapTmp);
//
//				//          VertexId* test_vid3 = new VertexId[n_active_edges];
//				//          cudaMemcpy(test_vid3, graph_slice->m_gatherDstsTmp,
//				//              n_active_edges * sizeof(VertexId),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherDstsTmp after gather-mgpu: ");
//				//          for (int i = 0; i < n_active_edges; ++i)
//				//          {
//				//            printf("%d, ", test_vid3[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid3;
//				//
//				//          GatherType* test_vid2 = new GatherType[n_active_edges];
//				//          cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp,
//				//              n_active_edges * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherMapTmp after gather-mgpu: ");
//				//          for (int i = 0; i < n_active_edges; ++i)
//				//          {
//				//            printf("%f, ", test_vid2[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid2;
//
//				mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//						, graph_slice->m_gatherMapTmp
//						, n_active_edges
//						, Program::INIT_VALUE
//						, ReduceFunctor()
//						, mgpu::equal_to<VertexId > ()
//						, (VertexId *)NULL
//						, ReduceOutputIterator(graph_slice->m_gatherTmp1, graph_slice->frontier_queues.d_keys[selector ^ 1])
//						, NULL
//						, NULL
//						, *m_mgpuContext);
//
//				//          GatherType* test_vid1 = new GatherType[graph_slice->nodes];
//				//          cudaMemcpy(test_vid1, graph_slice->m_gatherTmp1,
//				//              graph_slice->nodes * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherTmp after gather-mgpu: ");
//				//          for (int i = 0; i < graph_slice->nodes; ++i)
//				//          {
//				//            printf("%f, ", test_vid1[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid1;
//
//				//          printf("Gather all edges --- gather out!\n");
//
//				EdgeCountIterator ecIterator2(graph_slice->d_row_offsets,
//						graph_slice->frontier_queues.d_keys[selector ^ 1]);
//
//				mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
//						ecIterator2,
//						frontier_size,
//						0,
//						mgpu::plus<int>(),
//						m_deviceMappedValue,
//						(int *)NULL,
//						graph_slice->d_edgeCountScan,
//						*m_mgpuContext);
//
//				//          n_active_edges = *m_hostMappedValue;
//				cudaMemcpy(&n_active_edges, m_deviceMappedValue,
//						sizeof (int),
//						cudaMemcpyDeviceToHost);
//				//          printf("Gather all: out: n_active_edges = %d\n", n_active_edges);
//
//				partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
//				(mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
//						nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);
//
//				nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
//						nThreadsPerBlock);
//				//          grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);
//
//				vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
//				nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						nBlocks,
//						n_active_edges,
//						graph_slice->d_edgeCountScan,
//						partitions->get(),
//						graph_slice->d_row_offsets,
//						graph_slice->d_column_indices,
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						graph_slice->d_edgeCSC_indices,
//						graph_slice->m_gatherDstsTmp,
//						graph_slice->m_gatherMapTmp);
//
//				//          test_vid3 = new VertexId[n_active_edges];
//				//          cudaMemcpy(test_vid3, graph_slice->m_gatherDstsTmp,
//				//              n_active_edges * sizeof(VertexId),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherDstsTmp after gather-mgpu: ");
//				//          for (int i = 0; i < n_active_edges; ++i)
//				//          {
//				//            printf("%d, ", test_vid3[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid3;
//				//
//				//          test_vid2 = new GatherType[n_active_edges];
//				//          cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp,
//				//              n_active_edges * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherMapTmp after gather-mgpu: ");
//				//          for (int i = 0; i < n_active_edges; ++i)
//				//          {
//				//            printf("%f, ", test_vid2[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid2;
//
//				mgpu::ReduceByKey(graph_slice->m_gatherDstsTmp
//						, graph_slice->m_gatherMapTmp
//						, n_active_edges
//						, Program::INIT_VALUE
//						, ReduceFunctor()
//						, mgpu::equal_to<VertexId > ()
//						, (VertexId *)NULL
//						, ReduceOutputIterator(graph_slice->m_gatherTmp2, graph_slice->frontier_queues.d_keys[selector ^ 1])
//						, NULL
//						, NULL
//						, *m_mgpuContext);
//
//				//          test_vid1 = new GatherType[graph_slice->nodes];
//				//          cudaMemcpy(test_vid1, graph_slice->m_gatherTmp2,
//				//              graph_slice->nodes * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("graph_slice->m_gatherTmp after gather-mgpu: ");
//				//          for (int i = 0; i < graph_slice->nodes; ++i)
//				//          {
//				//            printf("%f, ", test_vid1[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid1;
//
//				thrust::device_ptr<GatherType> m_gatherTmp1_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp1);
//				thrust::device_ptr<GatherType> m_gatherTmp2_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp2);
//				thrust::device_ptr<GatherType> m_gatherTmp_ptr = thrust::device_pointer_cast(graph_slice->m_gatherTmp);
//				typename Program::gather_sum gather_sum_functor;
//				thrust::transform(m_gatherTmp1_ptr,
//						m_gatherTmp1_ptr + graph_slice->nodes,
//						m_gatherTmp2_ptr,
//						m_gatherTmp_ptr,
//						gather_sum_functor);
//
//				//          test_vid1 = new GatherType[graph_slice->nodes];
//				//          cudaMemcpy(test_vid1, graph_slice->m_gatherTmp,
//				//              graph_slice->nodes * sizeof(GatherType),
//				//              cudaMemcpyDeviceToHost);
//				//          printf("final graph_slice->m_gatherTmp after gather-mgpu: ");
//				//          for (int i = 0; i < graph_slice->nodes; ++i)
//				//          {
//				//            printf("%f, ", test_vid1[i]);
//				//          }
//				//          printf("\n");
//				//          delete[] test_vid1;
//			}
//			else
//			{
//				cout << "Error gatherOverEdges type!" << endl;
//				exit(1);
//			}
//		}
//		SYNC_CHECK();
//	}

  void gather_mgpu(typename CsrProblem::GraphSlice *graph_slice, int selector, const int directed, const int rank_id)
  {
    const int VT = 7;
    selector = 2; //use the 3rd frontier queue for gather
    if (directed == 0)
    //      if(0)
    {
      //          printf("Gather all edges --- gather in!\n");
      degree_iterator degree_iter(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector]);

      mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
          degree_iter,
          frontier_size,
          0,
          mgpu::plus<int>(),
          m_deviceMappedValue,
          (int *) NULL,
          graph_slice->d_edgeCountScan,
          *m_mgpuContext);

      //          int n_active_edges = *m_hostMappedValue;
      int n_active_edges;
      cudaMemcpy(&n_active_edges, m_deviceMappedValue,
          sizeof(int),
          cudaMemcpyDeviceToHost);

      //      printf("n_active_edges = %d\n", n_active_edges);
      //        printf("Gather all: int: n_active_edges = %d\n", n_active_edges);

      const int nThreadsPerBlock = 128;
      const int NV = nThreadsPerBlock * VT;
      MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
      (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
          NV, 0, mgpu::less<int>(), *m_mgpuContext);

      SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size, NV);
      //          dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);

      vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT,
      nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
          frontier_size,
          graph_slice->frontier_queues.d_keys[selector],
          nBlocks,
          n_active_edges,
          graph_slice->d_edgeCountScan,
          partitions->get(),
          graph_slice->d_column_offsets,
          graph_slice->d_row_indices,
          graph_slice->vertex_list,
          graph_slice->edge_list,
          NULL,
          m_gatherDstsTmp,
          m_gatherMapTmp);

      //        VertexId* test_vid3 = new VertexId[n_active_edges];
      //        cudaMemcpy(test_vid3, m_gatherDstsTmp,
      //            n_active_edges * sizeof(VertexId),
      //            cudaMemcpyDeviceToHost);
      //        printf("m_gatherDstsTmp after gather-mgpu: ");
      //        for (int i = 0; i < n_active_edges; ++i)
      //        {
      //          printf("%d, ", test_vid3[i]);
      //        }
      //        printf("\n");
      //        delete[] test_vid3;
      //
      //        GatherType* test_vid2 = new GatherType[n_active_edges];
      //        cudaMemcpy(test_vid2, m_gatherMapTmp,
      //            n_active_edges * sizeof(GatherType),
      //            cudaMemcpyDeviceToHost);
      //        printf("m_gatherMapTmp after gather-mgpu: ");
      //        for (int i = 0; i < n_active_edges; ++i)
      //        {
      //          printf("%f, ", test_vid2[i]);
      //        }
      //        printf("\n");
      //        delete[] test_vid2;

      mgpu::ReduceByKey(m_gatherDstsTmp
          , m_gatherMapTmp
          , n_active_edges
          , Program::INIT_VALUE
          , reduce_functor()
          , mgpu::equal_to<VertexId>()
          , (DataType *) NULL
          , reduce_iterator(m_gatherTmp1, graph_slice->frontier_queues.d_keys[selector])
          , NULL
          , NULL
          , *m_mgpuContext);

      //          GatherType* test_vid1 = new GatherType[graph_slice->nodes];
      //          cudaMemcpy(test_vid1, m_gatherTmp1,
      //              graph_slice->nodes * sizeof(GatherType),
      //              cudaMemcpyDeviceToHost);
      //          printf("m_gatherTmp after gather-mgpu: ");
      //          for (int i = 0; i < graph_slice->nodes; ++i)
      //          {
      //            printf("%f, ", test_vid1[i]);
      //          }
      //          printf("\n");
      //          delete[] test_vid1;

      //        printf("Gather all edges --- gather out!\n");

      degree_iterator degree_iter2(graph_slice->d_row_offsets,
          graph_slice->frontier_queues.d_keys[selector]);

      mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
          degree_iter2,
          frontier_size,
          0,
          mgpu::plus<int>(),
          m_deviceMappedValue,
          (int *) NULL,
          graph_slice->d_edgeCountScan,
          *m_mgpuContext);

      //          n_active_edges = *m_hostMappedValue;
      cudaMemcpy(&n_active_edges, m_deviceMappedValue,
          sizeof(int),
          cudaMemcpyDeviceToHost);
      //        printf("Gather all: out: n_active_edges = %d\n", n_active_edges);

      partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
      (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
          NV, 0, mgpu::less<int>(), *m_mgpuContext);

      nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
          NV);
      //          grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);

      vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT,
      nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
          frontier_size,
          graph_slice->frontier_queues.d_keys[selector],
          nBlocks,
          n_active_edges,
          graph_slice->d_edgeCountScan,
          partitions->get(),
          graph_slice->d_row_offsets,
          graph_slice->d_column_indices,
          graph_slice->vertex_list,
          graph_slice->edge_list,
          graph_slice->d_edgeCSC_indices,
          m_gatherDstsTmp,
          m_gatherMapTmp);

      //          test_vid3 = new VertexId[n_active_edges];
      //          cudaMemcpy(test_vid3, m_gatherDstsTmp,
      //              n_active_edges * sizeof(VertexId),
      //              cudaMemcpyDeviceToHost);
      //          printf("m_gatherDstsTmp after gather-mgpu: ");
      //          for (int i = 0; i < n_active_edges; ++i)
      //          {
      //            printf("%d, ", test_vid3[i]);
      //          }
      //          printf("\n");
      //          delete[] test_vid3;
      //
      //          test_vid2 = new GatherType[n_active_edges];
      //          cudaMemcpy(test_vid2, m_gatherMapTmp,
      //              n_active_edges * sizeof(GatherType),
      //              cudaMemcpyDeviceToHost);
      //          printf("m_gatherMapTmp after gather-mgpu: ");
      //          for (int i = 0; i < n_active_edges; ++i)
      //          {
      //            printf("%f, ", test_vid2[i]);
      //          }
      //          printf("\n");
      //          delete[] test_vid2;

      mgpu::ReduceByKey(m_gatherDstsTmp
          , m_gatherMapTmp
          , n_active_edges
          , Program::INIT_VALUE
          , reduce_functor()
          , mgpu::equal_to<VertexId>()
          , (VertexId *) NULL
          , reduce_iterator(m_gatherTmp2, graph_slice->frontier_queues.d_keys[selector])
          , NULL
          , NULL
          , *m_mgpuContext);

      //          test_vid1 = new GatherType[graph_slice->nodes];
      //          cudaMemcpy(test_vid1, m_gatherTmp2,
      //              graph_slice->nodes * sizeof(GatherType),
      //              cudaMemcpyDeviceToHost);
      //          printf("m_gatherTmp after gather-mgpu: ");
      //          for (int i = 0; i < graph_slice->nodes; ++i)
      //          {
      //            printf("%f, ", test_vid1[i]);
      //          }
      //          printf("\n");
      //          delete[] test_vid1;

      thrust::device_ptr<GatherType> m_gatherTmp1_ptr = thrust::device_pointer_cast(m_gatherTmp1);
      thrust::device_ptr<GatherType> m_gatherTmp2_ptr = thrust::device_pointer_cast(m_gatherTmp2);
      thrust::device_ptr<GatherType> m_gatherTmp_ptr = thrust::device_pointer_cast(m_gatherTmp);
      typename Program::gather_sum gather_sum_functor;
      thrust::transform(m_gatherTmp1_ptr,
          m_gatherTmp1_ptr + graph_slice->nodes,
          m_gatherTmp2_ptr,
          m_gatherTmp_ptr,
          gather_sum_functor);

      //          test_vid1 = new GatherType[graph_slice->nodes];
      //          cudaMemcpy(test_vid1, m_gatherTmp,
      //              graph_slice->nodes * sizeof(GatherType),
      //              cudaMemcpyDeviceToHost);
      //          printf("final m_gatherTmp after gather-mgpu: ");
      //          for (int i = 0; i < graph_slice->nodes; ++i)
      //          {
      //            printf("%f, ", test_vid1[i]);
      //          }
      //          printf("\n");
      //          delete[] test_vid1;
    }
    else
    {
      if (Program::gatherOverEdges() == GATHER_IN_EDGES)
      {
//				printf("rank_id=%d\n", rank_id);
        degree_iterator degree_iter(graph_slice->d_column_offsets,
            graph_slice->frontier_queues.d_keys[selector]);

        mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
            degree_iter,
            frontier_size,
            0,
            mgpu::plus<int>(),
            m_deviceMappedValue,
            (int *) NULL,
            graph_slice->d_edgeCountScan,
            *m_mgpuContext);

        //          int n_active_edges = *m_hostMappedValue;
        int n_active_edges;
        cudaMemcpy(&n_active_edges, m_deviceMappedValue,
            sizeof(int),
            cudaMemcpyDeviceToHost);

//				if(rank_id == 1)
//				printf("frontier_size=%d, n_active_edges=%d\n", frontier_size, n_active_edges);

        const int nThreadsPerBlock = 128;
        const int NV = nThreadsPerBlock * VT;
        MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
        (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
            NV, 0, mgpu::less<int>(), *m_mgpuContext);

        SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
            NV);

        vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT, nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
            frontier_size,
            graph_slice->frontier_queues.d_keys[selector],
            nBlocks,
            n_active_edges,
            graph_slice->d_edgeCountScan,
            partitions->get(),
            graph_slice->d_column_offsets,
            graph_slice->d_row_indices,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            NULL,
            m_gatherDstsTmp,
            m_gatherMapTmp);

				if(rank_id == 0)
				{
					VertexId* test_vid3 = new VertexId[n_active_edges];
					cudaMemcpy(test_vid3, m_gatherDstsTmp,
							n_active_edges * sizeof(VertexId),
							cudaMemcpyDeviceToHost);
					printf("m_gatherDstsTmp after gather-mgpu: n_active_edges=%d:", n_active_edges);
//					for (int i = 0; i < n_active_edges; ++i)
					for (int i = 0; i < 20; ++i)
					{
						printf("%d, ", test_vid3[i]);
					}
					printf("\n");
					delete[] test_vid3;

					GatherType* test_vid2 = new GatherType[n_active_edges];
					cudaMemcpy(test_vid2, m_gatherMapTmp,
							n_active_edges * sizeof(GatherType),
							cudaMemcpyDeviceToHost);
					printf("m_gatherMapTmp after gather-mgpu: ");
//					for (int i = 0; i < n_active_edges; ++i)
					for (int i = 0; i < 20; ++i)
					{
						printf("%f, ", test_vid2[i]);
					}
					printf("\n");
					delete[] test_vid2;
				}

        thrust::device_ptr<VertexId> m_gatherDstsTmp_ptr(m_gatherDstsTmp);
        thrust::device_ptr<GatherType> m_gatherMapTmp_ptr(m_gatherMapTmp);
        thrust::device_ptr<GatherType> m_gatherTmp_ptr(m_gatherTmp);

//				if(n_active_edges > 0)
//				{
//				mgpu::ReduceByKey(m_gatherDstsTmp,
//						m_gatherMapTmp,
//						n_active_edges,
//						Program::INIT_VALUE,
//						reduce_functor(),
//						mgpu::equal_to<VertexId>(),
//						(VertexId *) NULL,
//						 reduce_iterator(m_gatherTmp, graph_slice->frontier_queues.d_keys[selector]),
////						m_gatherTmp,
//						NULL,
//						NULL,
//						*m_mgpuContext);
//				}

        thrust::equal_to<GatherType> binary_pred;
//				thrust::device_vector<int> tmp_vector(n_active_edges);
        thrust::reduce_by_key(m_gatherDstsTmp_ptr, m_gatherDstsTmp_ptr + n_active_edges,
            m_gatherMapTmp_ptr, thrust::make_discard_iterator(), m_gatherTmp_ptr, binary_pred, reduce_functor());

				if(rank_id == 0)
				{
					GatherType* test_vid2 = new GatherType[graph_slice->nodes];
					cudaMemcpy(test_vid2, m_gatherTmp,
							graph_slice->nodes * sizeof(GatherType),
							cudaMemcpyDeviceToHost);
					printf("m_gatherTmp after gather-mgpu: ");
//					for (int i = 0; i < graph_slice->nodes; ++i)
					for (int i = 0; i < 20; ++i)
					{
						printf("%f, ", test_vid2[i]);
					}
					printf("\n");
					delete[] test_vid2;
				}
      }
      else if (Program::gatherOverEdges() == GATHER_OUT_EDGES)
      {
        degree_iterator degree_iter(graph_slice->d_row_offsets,
            graph_slice->frontier_queues.d_keys[selector]);

        mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
            degree_iter,
            frontier_size,
            0,
            mgpu::plus<int>(),
            m_deviceMappedValue,
            (int *) NULL,
            graph_slice->d_edgeCountScan,
            *m_mgpuContext);

        //          int n_active_edges = *m_hostMappedValue;
        int n_active_edges;
        cudaMemcpy(&n_active_edges, m_deviceMappedValue,
            sizeof(int),
            cudaMemcpyDeviceToHost);
        //      printf("n_active_edges = %d\n", n_active_edges);

        const int nThreadsPerBlock = 128;
        const int NV = nThreadsPerBlock * VT;
        MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
        (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
            NV, 0, mgpu::less<int>(), *m_mgpuContext);

        SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
            NV);

        vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT,
        nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
            frontier_size,
            graph_slice->frontier_queues.d_keys[selector],
            nBlocks,
            n_active_edges,
            graph_slice->d_edgeCountScan,
            partitions->get(),
            graph_slice->d_row_offsets,
            graph_slice->d_column_indices,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            graph_slice->d_edgeCSC_indices,
            m_gatherDstsTmp,
            m_gatherMapTmp);

        mgpu::ReduceByKey(m_gatherDstsTmp
            , m_gatherMapTmp
            , n_active_edges
            , Program::INIT_VALUE
            , reduce_functor()
            , mgpu::equal_to<VertexId>()
            , (VertexId *) NULL
            , reduce_iterator(m_gatherTmp, graph_slice->frontier_queues.d_keys[selector])
            , NULL
            , NULL
            , *m_mgpuContext);
      }
      else if (Program::gatherOverEdges() == GATHER_ALL_EDGES)
      {
        //          printf("Gather all edges --- gather in!\n");
        degree_iterator degree_iter(graph_slice->d_column_offsets, graph_slice->frontier_queues.d_keys[selector]);

        mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
            degree_iter,
            frontier_size,
            0,
            mgpu::plus<int>(),
            m_deviceMappedValue,
            (int *) NULL,
            graph_slice->d_edgeCountScan,
            *m_mgpuContext);

        //          int n_active_edges = *m_hostMappedValue;
        int n_active_edges;
        cudaMemcpy(&n_active_edges, m_deviceMappedValue,
            sizeof(int),
            cudaMemcpyDeviceToHost);
        //      printf("n_active_edges = %d\n", n_active_edges);
        //          printf("Gather all: int: n_active_edges = %d\n", n_active_edges);

        const int nThreadsPerBlock = 128;
        const int NV = nThreadsPerBlock * VT;
        MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
        (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
            NV, 0, mgpu::less<int>(), *m_mgpuContext);

        SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size, NV);
        //          dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);

        vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT,
        nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
            frontier_size,
            graph_slice->frontier_queues.d_keys[selector],
            nBlocks,
            n_active_edges,
            graph_slice->d_edgeCountScan,
            partitions->get(),
            graph_slice->d_column_offsets,
            graph_slice->d_row_indices,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            NULL,
            m_gatherDstsTmp,
            m_gatherMapTmp);

        //          VertexId* test_vid3 = new VertexId[n_active_edges];
        //          cudaMemcpy(test_vid3, m_gatherDstsTmp,
        //              n_active_edges * sizeof(VertexId),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherDstsTmp after gather-mgpu: ");
        //          for (int i = 0; i < n_active_edges; ++i)
        //          {
        //            printf("%d, ", test_vid3[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid3;
        //
        //          GatherType* test_vid2 = new GatherType[n_active_edges];
        //          cudaMemcpy(test_vid2, m_gatherMapTmp,
        //              n_active_edges * sizeof(GatherType),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherMapTmp after gather-mgpu: ");
        //          for (int i = 0; i < n_active_edges; ++i)
        //          {
        //            printf("%f, ", test_vid2[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid2;

        mgpu::ReduceByKey(m_gatherDstsTmp
            , m_gatherMapTmp
            , n_active_edges
            , Program::INIT_VALUE
            , reduce_functor()
            , mgpu::equal_to<VertexId>()
            , (VertexId *) NULL
            , reduce_iterator(m_gatherTmp1, graph_slice->frontier_queues.d_keys[selector])
            , NULL
            , NULL
            , *m_mgpuContext);

        //          GatherType* test_vid1 = new GatherType[graph_slice->nodes];
        //          cudaMemcpy(test_vid1, m_gatherTmp1,
        //              graph_slice->nodes * sizeof(GatherType),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherTmp after gather-mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%f, ", test_vid1[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid1;

        //          printf("Gather all edges --- gather out!\n");

        degree_iterator degree_iter2(graph_slice->d_row_offsets,
            graph_slice->frontier_queues.d_keys[selector]);

        mgpu::Scan<mgpu::MgpuScanTypeExc, degree_iterator, int, mgpu::plus<int>, int*>(
            degree_iter2,
            frontier_size,
            0,
            mgpu::plus<int>(),
            m_deviceMappedValue,
            (int *) NULL,
            graph_slice->d_edgeCountScan,
            *m_mgpuContext);

        //          n_active_edges = *m_hostMappedValue;
        cudaMemcpy(&n_active_edges, m_deviceMappedValue,
            sizeof(int),
            cudaMemcpyDeviceToHost);
        //          printf("Gather all: out: n_active_edges = %d\n", n_active_edges);

        partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
        (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
            NV, 0, mgpu::less<int>(), *m_mgpuContext);

        nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
            NV);
        //          grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);

        vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT,
        nThreadsPerBlock><<<nBlocks, nThreadsPerBlock>>>(
            frontier_size,
            graph_slice->frontier_queues.d_keys[selector],
            nBlocks,
            n_active_edges,
            graph_slice->d_edgeCountScan,
            partitions->get(),
            graph_slice->d_row_offsets,
            graph_slice->d_column_indices,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            graph_slice->d_edgeCSC_indices,
            m_gatherDstsTmp,
            m_gatherMapTmp);

        //          test_vid3 = new VertexId[n_active_edges];
        //          cudaMemcpy(test_vid3, m_gatherDstsTmp,
        //              n_active_edges * sizeof(VertexId),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherDstsTmp after gather-mgpu: ");
        //          for (int i = 0; i < n_active_edges; ++i)
        //          {
        //            printf("%d, ", test_vid3[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid3;
        //
        //          test_vid2 = new GatherType[n_active_edges];
        //          cudaMemcpy(test_vid2, m_gatherMapTmp,
        //              n_active_edges * sizeof(GatherType),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherMapTmp after gather-mgpu: ");
        //          for (int i = 0; i < n_active_edges; ++i)
        //          {
        //            printf("%f, ", test_vid2[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid2;

        mgpu::ReduceByKey(m_gatherDstsTmp
            , m_gatherMapTmp
            , n_active_edges
            , Program::INIT_VALUE
            , reduce_functor()
            , mgpu::equal_to<VertexId>()
            , (VertexId *) NULL
            , reduce_iterator(m_gatherTmp2, graph_slice->frontier_queues.d_keys[selector])
            , NULL
            , NULL
            , *m_mgpuContext);

        //          test_vid1 = new GatherType[graph_slice->nodes];
        //          cudaMemcpy(test_vid1, m_gatherTmp2,
        //              graph_slice->nodes * sizeof(GatherType),
        //              cudaMemcpyDeviceToHost);
        //          printf("m_gatherTmp after gather-mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%f, ", test_vid1[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid1;

        thrust::device_ptr<GatherType> m_gatherTmp1_ptr = thrust::device_pointer_cast(m_gatherTmp1);
        thrust::device_ptr<GatherType> m_gatherTmp2_ptr = thrust::device_pointer_cast(m_gatherTmp2);
        thrust::device_ptr<GatherType> m_gatherTmp_ptr = thrust::device_pointer_cast(m_gatherTmp);
        typename Program::gather_sum gather_sum_functor;
        thrust::transform(m_gatherTmp1_ptr,
            m_gatherTmp1_ptr + graph_slice->nodes,
            m_gatherTmp2_ptr,
            m_gatherTmp_ptr,
            gather_sum_functor);

        //          test_vid1 = new GatherType[graph_slice->nodes];
        //          cudaMemcpy(test_vid1, m_gatherTmp,
        //              graph_slice->nodes * sizeof(GatherType),
        //              cudaMemcpyDeviceToHost);
        //          printf("final m_gatherTmp after gather-mgpu: ");
        //          for (int i = 0; i < graph_slice->nodes; ++i)
        //          {
        //            printf("%f, ", test_vid1[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid1;
      }
      else
      {
        cout << "Error gatherOverEdges type!" << endl;
        throw std::exception();
      }
    }

    if (util::B40CPerror(cudaDeviceSynchronize(), "Error after gather"
            __FILE__,
            __LINE__))
    throw std::exception();

  }

  template<typename ExpandPolicy>
  void expand_dynamic(typename CsrProblem::GraphSlice *graph_slice, const int directed, const int selector, const int frontier_selector, const int queue_index, const int expand_grid_size)
  {
    //      printf("frontier_size before expand = %d\n", frontier_size);

    if (Program::allow_duplicates)
    {

      if (directed == 0)
      {
        //            printf("Expand_dynamic:Dup:All\n");
        vertex_centric::expand_atomic::Kernel<ExpandPolicy,
        Program> << <expand_grid_size,
        ExpandPolicy::THREADS >> >(iteration[0],
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            frontier_selector,
            0,
            d_frontier_size,
            d_edge_frontier_size,
            d_done,
            graph_slice->frontier_queues.d_keys[selector
            ^ 1],// vertex frontier in
            graph_slice->frontier_queues.d_keys[selector],// edge frontier out
            graph_slice->frontier_queues.d_values[selector],// predecessor out
            graph_slice->vertex_list,//
            graph_slice->edge_list,
            NULL,
            graph_slice->d_changed,
            graph_slice->d_column_indices,
            graph_slice->d_row_offsets,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
            graph_slice->frontier_elements[selector],// max edge frontier vertices
            this->expand_kernel_stats);

        int edge_frontier_size1;
        cudaMemcpy(
            &edge_frontier_size1,
            &d_edge_frontier_size[frontier_selector],
            sizeof (int),
            cudaMemcpyDeviceToHost);

        //            cudaDeviceSynchronize();
        //
        //            printf("frontier size1: %d: ", edge_frontier_size1);
        //            int* frontier = new int[edge_frontier_size1];
        //            cudaMemcpy(frontier,
        //                graph_slice->frontier_queues.d_keys[selector],
        //                sizeof(int) * edge_frontier_size1,
        //                cudaMemcpyDeviceToHost);
        //            for(int i=0; i<edge_frontier_size1; i++)
        //            {
        //              printf("%d, ", frontier[i]);
        //            }
        //            printf("\n");
        //            delete [] frontier;

        vertex_centric::expand_atomic::Kernel<ExpandPolicy,
        Program> << <expand_grid_size,
        ExpandPolicy::THREADS >> >(iteration[0],
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            frontier_selector,
            edge_frontier_size1,
            d_frontier_size,
            d_edge_frontier_size,
            d_done,
            graph_slice->frontier_queues.d_keys[selector
            ^ 1],// vertex frontier in
            graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,// edge frontier out
            graph_slice->frontier_queues.d_values[selector] + edge_frontier_size1,// predecessor out
            graph_slice->vertex_list,//
            graph_slice->edge_list,
            graph_slice->d_edgeCSC_indices,
            graph_slice->d_changed,
            graph_slice->d_row_indices,
            graph_slice->d_column_offsets,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
            graph_slice->frontier_elements[selector],// max edge frontier vertices
            this->expand_kernel_stats);
        //          vertex_centric::expand_atomic::Kernel<ExpandPolicy,
        //          Program><<<expand_grid_size,
        //          ExpandPolicy::THREADS>>>(iteration[0],
        //              queue_index,              // queue counter index
        //              queue_index,// steal counter index
        //              1,// number of GPUs
        //              frontier_selector,
        //              0,
        //              d_frontier_size,
        //              d_edge_frontier_size,
        //              d_done,
        //              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
        //              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
        //              graph_slice->frontier_queues.d_values[selector],// predecessor out
        //              graph_slice->vertex_list,//
        //              graph_slice->edge_list,
        //              NULL,
        //              graph_slice->d_changed,
        //              graph_slice->d_column_indices,
        //              graph_slice->d_row_offsets, this->work_progress,
        //              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
        //              graph_slice->frontier_elements[selector],// max edge frontier vertices
        //              this->expand_kernel_stats);

      }
      else
      {
        if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
        {

          vertex_centric::expand_atomic::Kernel<ExpandPolicy, Program> << <expand_grid_size,
          ExpandPolicy::THREADS >> >(iteration[0],
              queue_index, // queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              NULL,
              graph_slice->d_changed,
              graph_slice->d_column_indices,
              graph_slice->d_row_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);
        }
        else if (Program::expandOverEdges()
            == EXPAND_IN_EDGES)
        {
          vertex_centric::expand_atomic::Kernel<ExpandPolicy,
          Program> << <expand_grid_size,
          ExpandPolicy::THREADS >> >(iteration[0],
              queue_index, // queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector
              ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->d_changed,
              graph_slice->d_row_indices,
              graph_slice->d_column_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);
        }
        else if (Program::expandOverEdges() == EXPAND_ALL_EDGES)
        {

          //            printf("Expand_dynamic:Dup:All\n");
          vertex_centric::expand_atomic::Kernel<ExpandPolicy,
          Program> << <expand_grid_size,
          ExpandPolicy::THREADS >> >(iteration[0],
              queue_index,// queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector
              ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              NULL,
              graph_slice->d_changed,
              graph_slice->d_column_indices,
              graph_slice->d_row_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);

          int edge_frontier_size1;
          cudaMemcpy(
              &edge_frontier_size1,
              &d_edge_frontier_size[frontier_selector],
              sizeof (int),
              cudaMemcpyDeviceToHost);

          //            cudaDeviceSynchronize();
          //
          //            printf("frontier size1: %d: ", edge_frontier_size1);
          //            int* frontier = new int[edge_frontier_size1];
          //            cudaMemcpy(frontier,
          //                graph_slice->frontier_queues.d_keys[selector],
          //                sizeof(int) * edge_frontier_size1,
          //                cudaMemcpyDeviceToHost);
          //            for(int i=0; i<edge_frontier_size1; i++)
          //            {
          //              printf("%d, ", frontier[i]);
          //            }
          //            printf("\n");
          //            delete [] frontier;

          vertex_centric::expand_atomic::Kernel<ExpandPolicy,
          Program> << <expand_grid_size,
          ExpandPolicy::THREADS >> >(iteration[0],
              queue_index,// queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              edge_frontier_size1,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector
              ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,// edge frontier out
              graph_slice->frontier_queues.d_values[selector] + edge_frontier_size1,// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->d_changed,
              graph_slice->d_row_indices,
              graph_slice->d_column_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);
        }
      }
    }
    else
    {
      if (directed == 0)
      {
        //            printf("Expand dynamic: expand all\n");
        vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
            iteration[0],
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            frontier_selector,
            0,
            d_frontier_size,
            d_edge_frontier_size,
            d_done,
            graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
            graph_slice->frontier_queues.d_keys[selector],// edge frontier out
            graph_slice->frontier_queues.d_values[selector],// predecessor out
            graph_slice->vertex_list,//
            graph_slice->edge_list,
            NULL,
            graph_slice->d_changed,
            graph_slice->d_active_flags,
            graph_slice->d_column_indices,
            graph_slice->d_row_offsets,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
            graph_slice->frontier_elements[selector],// max edge frontier vertices
            this->expand_kernel_stats);

        int edge_frontier_size1;
        cudaMemcpy(
            &edge_frontier_size1,
            &d_edge_frontier_size[frontier_selector],
            sizeof (int),
            cudaMemcpyDeviceToHost);

        //          printf("frontier size1: %d: ", edge_frontier_size1);
        //          int* frontier = new int[edge_frontier_size1];
        //          cudaMemcpy(frontier,
        //              graph_slice->frontier_queues.d_keys[selector],
        //              sizeof(int) * edge_frontier_size1,
        //              cudaMemcpyDeviceToHost);
        //          for(int i=0; i<edge_frontier_size1; i++)
        //          {
        //            printf("%d, ", frontier[i]);
        //          }
        //          printf("\n");
        //          delete [] frontier;

        vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
            iteration[0],
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            frontier_selector,
            edge_frontier_size1,
            d_frontier_size,
            d_edge_frontier_size,
            d_done,
            graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
            graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,// edge frontier out
            graph_slice->frontier_queues.d_values[selector] + edge_frontier_size1,// predecessor out
            graph_slice->vertex_list,//
            graph_slice->edge_list,
            graph_slice->d_edgeCSC_indices,
            graph_slice->d_changed,
            graph_slice->d_active_flags,
            graph_slice->d_row_indices,
            graph_slice->d_column_offsets,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
            graph_slice->frontier_elements[selector],// max edge frontier vertices
            this->expand_kernel_stats);

        int edge_frontier_size2;
        cudaMemcpy(
            &edge_frontier_size2,
            &d_edge_frontier_size[frontier_selector],
            sizeof (int),
            cudaMemcpyDeviceToHost);

        //          printf("frontier size2: %d: ", edge_frontier_size2);
        //          frontier = new int[edge_frontier_size2];
        //          cudaMemcpy(frontier,
        //              graph_slice->frontier_queues.d_keys[selector],
        //              sizeof(int) * (edge_frontier_size2),
        //              cudaMemcpyDeviceToHost);
        //          for(int i=0; i<edge_frontier_size2; i++)
        //          {
        //            printf("%d, ", frontier[i]);
        //          }
        //          printf("\n");
        //          delete [] frontier;

        //reset the active flags
        cudaMemset(graph_slice->d_active_flags, 0, graph_slice->nodes * sizeof (int));

        //          vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program><<<expand_grid_size, ExpandPolicy::THREADS>>>(
        //              iteration[0],
        //              queue_index,              // queue counter index
        //              queue_index,// steal counter index
        //              1,// number of GPUs
        //              frontier_selector,
        //              0,
        //              d_frontier_size,
        //              d_edge_frontier_size,
        //              d_done,
        //              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
        //              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
        //              graph_slice->frontier_queues.d_values[selector],// predecessor out
        //              graph_slice->vertex_list,//
        //              graph_slice->edge_list,
        //              NULL,
        //              graph_slice->d_changed,
        //              graph_slice->d_active_flags,
        //              graph_slice->d_column_indices,
        //              graph_slice->d_row_offsets,
        //              this->work_progress,
        //              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
        //              graph_slice->frontier_elements[selector],// max edge frontier vertices
        //              this->expand_kernel_stats);

      }
      else
      {
        if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
        {

          //                printf("EXPAND_OUT_EDGES ...\n");
          vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
              iteration[0],
              queue_index,// queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              NULL,
              graph_slice->d_changed,
              graph_slice->d_active_flags,
              graph_slice->d_column_indices,
              graph_slice->d_row_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);
        }
        else if (Program::expandOverEdges() == EXPAND_IN_EDGES)
        {
          vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
              iteration[0],
              queue_index, // queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->d_changed,
              graph_slice->d_active_flags,
              graph_slice->d_row_indices,
              graph_slice->d_column_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);
        }
        else if (Program::expandOverEdges() == EXPAND_ALL_EDGES)
        {
          //            printf("Expand dynamic: expand all\n");
          vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
              iteration[0],
              queue_index,// queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              0,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              NULL,
              graph_slice->d_changed,
              graph_slice->d_active_flags,
              graph_slice->d_column_indices,
              graph_slice->d_row_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);

          int edge_frontier_size1;
          cudaMemcpy(
              &edge_frontier_size1,
              &d_edge_frontier_size[frontier_selector],
              sizeof (int),
              cudaMemcpyDeviceToHost);

          //            printf("frontier size1: %d: ", edge_frontier_size1);
          //            int* frontier = new int[edge_frontier_size1];
          //            cudaMemcpy(frontier,
          //                graph_slice->frontier_queues.d_keys[selector],
          //                sizeof (int)* edge_frontier_size1,
          //                cudaMemcpyDeviceToHost);
          //            for (int i = 0; i < edge_frontier_size1; i++)
          //            {
          //              printf("%d, ", frontier[i]);
          //            }
          //            printf("\n");
          //            delete [] frontier;

          vertex_centric::expand_atomic_flag::Kernel<ExpandPolicy, Program> << <expand_grid_size, ExpandPolicy::THREADS >> >(
              iteration[0],
              queue_index,// queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              frontier_selector,
              edge_frontier_size1,
              d_frontier_size,
              d_edge_frontier_size,
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector] + edge_frontier_size1,// edge frontier out
              graph_slice->frontier_queues.d_values[selector] + edge_frontier_size1,// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              graph_slice->d_edgeCSC_indices,
              graph_slice->d_changed,
              graph_slice->d_active_flags,
              graph_slice->d_row_indices,
              graph_slice->d_column_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector],// max edge frontier vertices
              this->expand_kernel_stats);

          int edge_frontier_size2;
          cudaMemcpy(
              &edge_frontier_size2,
              &d_edge_frontier_size[frontier_selector],
              sizeof (int),
              cudaMemcpyDeviceToHost);

          //            printf("frontier size2: %d: ", edge_frontier_size2);
          //            frontier = new int[edge_frontier_size2];
          //            cudaMemcpy(frontier,
          //                graph_slice->frontier_queues.d_keys[selector],
          //                sizeof (int)* (edge_frontier_size2),
          //                cudaMemcpyDeviceToHost);
          //            for (int i = 0; i < edge_frontier_size2; i++)
          //            {
          //              printf("%d, ", frontier[i]);
          //            }
          //            printf("\n");
          //            delete [] frontier;
        }
        //          thrust::device_ptr<int> active_flags_ptr = thrust::device_pointer_cast(graph_slice->d_active_flags);
        //          thrust::device_ptr<VertexId> frontier_queue_ptr = thrust::device_pointer_cast(graph_slice->frontier_queues.d_keys[selector]);
        //          thrust::sequence(d_vertex_ids.begin(), d_vertex_ids.end());
        //          thrust::device_ptr<VertexId> result_end = thrust::copy_if(d_vertex_ids.begin(), d_vertex_ids.end(),
        //              active_flags_ptr, frontier_queue_ptr, thrust::identity<int>());
        //          frontier_size = result_end - frontier_queue_ptr;
        //          work_progress.SetQueueLength(queue_index + 1, frontier_size);

        //reset the active flags
        cudaMemset(graph_slice->d_active_flags, 0, graph_slice->nodes * sizeof (int));
      }

    }

    //      if (util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))
    //        exit(1);

  }

  template<typename ExpandPolicy,
  typename ContractPolicy>
  cudaError_t EnactIterativeSearch(CsrProblem &csr_problem,
      typename CsrProblem::SizeT* h_row_offsets,
      int directed, int threshold, int expand_grid_size, int contract_grid_size, int &selector, int &frontier_selector, int pi, int pj, int rank_id)
  {
    typedef typename CsrProblem::SizeT SizeT;
    typedef typename CsrProblem::VertexId VertexId;
    typedef typename CsrProblem::EValue EValue;
    typedef typename CsrProblem::VisitedMask VisitedMask;

    // Single-gpu graph slice
    typename CsrProblem::GraphSlice *graph_slice =
    csr_problem.graph_slices[0];

    DEBUG = cfg.getParameter<int>("verbose");

    cudaError_t retval = cudaSuccess;

    // Determine grid size(s)
    //      int expand_occupancy = ExpandPolicy::CTA_OCCUPANCY;
    //      int expand_grid_size = MaxGridSize(expand_occupancy);
    //
    ////      int gather_occupancy = GatherPolicy::CTA_OCCUPANCY;
    ////      int gather_grid_size = MaxGridSize(gather_occupancy);
    //
    //      int contract_occupancy = ContractPolicy::CTA_OCCUPANCY;
    //      int contract_grid_size = MaxGridSize(contract_occupancy);

    //      cudaMallocHost(&m_hostMappedValue, sizeof(SizeT), cudaHostAllocMapped);
    //      cudaHostGetDevicePointer(&m_deviceMappedValue, m_hostMappedValue, 0);
    //      cudaMalloc((void**) &d_frontier_size, 2 * sizeof(SizeT));
    //      cudaMalloc((void**) &d_edge_frontier_size, 2 * sizeof(SizeT));
    //
    //      if (retval = util::B40CPerror(cudaMemset(d_edge_frontier_size, 0, 2 * sizeof(SizeT)),
    //              "CsrProblem cudaMemset d_edge_frontier_size failed", __FILE__,
    //              __LINE__))
    //      return retval;

    //      int tmp[2] =
    //      { num_srcs, 0};
    //      if (retval = util::B40CPerror(
    //              cudaMemcpy(d_frontier_size,
    //                  tmp,
    //                  2 * sizeof(int),
    //                  cudaMemcpyHostToDevice),
    //              "CsrProblem cudaMemcpy d_frontier_size failed",
    //              __FILE__, __LINE__))
    //      return retval;

//		cudaMalloc((void**) &graph_slice->m_gatherMapTmp, (graph_slice->edges + graph_slice->nodes) * sizeof(GatherType));
//		cudaMalloc((void**) &graph_slice->m_gatherTmp, graph_slice->nodes * sizeof(GatherType));
//		if ( (Program::gatherOverEdges() == GATHER_ALL_EDGES || directed == 0) && Program::gatherOverEdges() != NO_GATHER_EDGES)
//		{
//			cudaMalloc((void**) &graph_slice->m_gatherTmp1, graph_slice->nodes * sizeof(GatherType));
//			//        cudaMemset(graph_slice->m_gatherTmp1, 0, graph_slice->nodes * sizeof(GatherType) );
//			cudaMalloc((void**) &graph_slice->m_gatherTmp2, graph_slice->nodes * sizeof(GatherType));
//		}
//		cudaMalloc((void**) &graph_slice->m_gatherDstsTmp, (graph_slice->edges + graph_slice->nodes) * sizeof(VertexId));

    ////      thrust::device_vector<int> d_vertex_ids = thrust::device_vector<int>(graph_slice->nodes);
    ////      thrust::sequence(d_vertex_ids.begin(), d_vertex_ids.end());
    //      int memset_block_size = 256;
    //      int memset_grid_size_max = 32 * 1024;              // 32K CTAs
    //      int memset_grid_size;
    //
    //      memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slice->nodes + memset_block_size - 1) / memset_block_size);
    //
    ////init graph_slice->m_gatherTmp, necessary for CC!!!
    //      util::MemsetKernel<GatherType><<<memset_grid_size,
    //      memset_block_size, 0, graph_slice->stream>>>(
    //          graph_slice->m_gatherTmp, Program::INIT_VALUE,
    //          graph_slice->nodes);
    //
    //      if (retval = util::B40CPerror(cudaThreadSynchronize(),
    //              "MemsetKernel graph_slice->m_gatherTmp failed", __FILE__, __LINE__))
    //      return retval;

    //      double max_queue_sizing = cfg.getParameter<double>("max_queue_sizing");

    //// Reset data
    //      if (retval = csr_problem.Reset(GetFrontierType(),
    //              max_queue_sizing))
    //      return retval;
    //
    //      Program::Initialize(directed, graph_slice->nodes, graph_slice->edges, num_srcs,
    //          srcs, graph_slice->d_row_offsets, graph_slice->d_column_indices, graph_slice->d_column_offsets, graph_slice->d_row_indices,
    //          graph_slice->d_edge_values,
    //          graph_slice->vertex_list, graph_slice->edge_list,
    //          graph_slice->frontier_queues.d_keys,
    //          graph_slice->frontier_queues.d_values);
    //
    //      if (retval = Setup(csr_problem, expand_grid_size,
    //              contract_grid_size, 0))
    //      return retval;
    //
    ////          SizeT queue_length;
    VertexId queue_index = 0;// Work stealing/queue index
    //      int selector = 0;
    //      int frontier_selector = 0;
    //      frontier_size = num_srcs;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double startcontract, endcontract;
    double startgather, endgather;
    double startexpand, endexpand;

    double elapsedcontract = 0.0;
    double elapsedgather = 0.0;
    double elapsedexpand = 0.0;
//		SYNC_CHECK();

    cudaEventRecord(start);
    double startTime = omp_get_wtime();

    // Forward phase BC iterations
    //          while (done[0] < 0 || frontier_size > 0)
    for (int i = 0; i < 1; i++)
    {
      //        if (DEBUG)
      //        printf("Iteration: %lld, frontier_size: %d\n", (long long)iteration[0], frontier_size);

      //        int old_frontier_size = frontier_size; //for contract check

      ////            if (DEBUG)
      ////            {
      ////              printf("queue_length after contraction: %lld\n",
      ////                  (long long) queue_length);
      ////
      //////              VertexId* test_vid = new VertexId[graph_slice->nodes];
      //////              cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
      //////              printf("Frontier after contraction: ");
      //////              for (int i = 0; i < queue_length; ++i)
      //////              {
      //////                printf("%d, ", test_vid[i]);
      //////              }
      //////              printf("\n");
      //////              delete[] test_vid;
      ////
      //////                EValue *test_vid2 = new EValue[graph_slice->nodes];
      //////                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
      //////                printf("d_dists after contract: ");
      //////                for (int i = 0; i < graph_slice->nodes; ++i)
      //////                {
      //////                  printf("%d, ", test_vid2[i]);
      //////                }
      //////                printf("\n");
      //////                delete[] test_vid2;
      ////
      //////                test_vid2 = new EValue[graph_slice->nodes];
      //////                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
      //////                printf("d_min_dists after contract: ");
      //////                for (int i = 0; i < graph_slice->nodes; ++i)
      //////                {
      //////                  printf("%d, ", test_vid2[i]);
      //////                }
      //////                printf("\n");
      //////                delete[] test_vid2;
      ////            }
      ////
      ////            // Throttle
      ////            if (iteration[0] & 1)
      ////            {
      ////              if (retval =
      ////                  util::B40CPerror(
      ////                      cudaEventRecord(throttle_event),
      ////                      "EnactorVertexCentric cudaEventRecord throttle_event failed",
      ////                      __FILE__, __LINE__))
      ////                break;
      ////            }
      ////            else
      ////            {
      ////              if (retval =
      ////                  util::B40CPerror(
      ////                      cudaEventSynchronize(throttle_event),
      ////                      "EnactorVertexCentric cudaEventSynchronize throttle_event failed",
      ////                      __FILE__, __LINE__))
      ////                break;
      ////            };
      //
      //            // Check if done
      //            if (done[0] == 0)
      //              break;

//			if (DEBUG)
//			{
//				cudaDeviceSynchronize();
//				startgather = omp_get_wtime();
//			}
//			//
//			//Gather stage
//			//
//			if (Program::gatherOverEdges() != NO_GATHER_EDGES)
//			gather_mgpu(graph_slice, selector, directed);
//
//			if (DEBUG)
//			{
//				cudaDeviceSynchronize();
//				endgather = omp_get_wtime();
//				elapsedgather += endgather - startgather;
//				printf("Gather time: %f ms\n", (endgather - startgather) * 1000);
//			}
//
//			//            if (DEBUG)
//			//            {
//			//              if (work_progress.GetQueueLength(queue_index,
//			//                  queue_length))
//			//                break;
//			//              total_queued += queue_length;
//			//
//			//              //                  if (DEBUG) printf("queue_length after gather: %lld\n", (long long) queue_length);
//			//
//			//              EValue *test_vid2 = new EValue[graph_slice->nodes];
//			//              cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//			//              printf("d_min_dists after gather: ");
//			//              for (int i = 0; i < graph_slice->nodes; ++i)
//			//              {
//			//                printf("%f, ", test_vid2[i]);
//			//              }
//			//              printf("\n");
//			//              delete[] test_vid2;
//			//              //
//			//              //                  test_vid2 = new EValue[graph_slice->nodes];
//			//              //                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//			//              //                  printf("d_gather_results after gather: ");
//			//              //                  for (int i = 0; i < graph_slice->nodes; ++i)
//			//              //                  {
//			//              //                    printf("%f, ", test_vid2[i]);
//			//              //                  }
//			//              //                  printf("\n");
//			//              //                  delete[] test_vid2;
//			//
//			//              //                  VertexId* test_vid = new VertexId[graph_slice->nodes];
//			//              //                  cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//			//              //                  printf("changed after gather: ");
//			//              //                  for (int i = 0; i < graph_slice->nodes; ++i)
//			//              //                  {
//			//              //                    printf("%d, ", test_vid[i]);
//			//              //                  }
//			//              //                  printf("\n");
//			//              //                  delete[] test_vid;
//			//            }
//
//			if (Program::applyOverEdges() == APPLY_FRONTIER)
//			{
//				//
//				//apply stage
//				//
//				apply_mgpu(graph_slice, selector);
//
//				//              vertex_centric::gather::apply<GatherPolicy, Program><<<
//				//              gather_grid_size, GatherPolicy::THREADS>>>(
//				//                  iteration[0], queue_index, this->work_progress,
//				//                  graph_slice->frontier_queues.d_keys[selector ^ 1],
//				//                  graph_slice->vertex_list,
//				//                  graph_slice->edge_list);
//
//				//                  if (DEBUG)
//				//                  {
//				//                    int num_changed;
//				//                    thrust::device_ptr<int> changed_ptr = thrust::device_pointer_cast(graph_slice->vertex_list.d_changed);
//				//                    num_changed = thrust::reduce(changed_ptr, changed_ptr + graph_slice->nodes);
//				//                    printf("num_changed=%d\n", num_changed);
//				//                  }
//			}
//
//			if (Program::postApplyOverEdges() == POST_APPLY_FRONTIER)
//			{
//
//				//
//				//                //reset dists and gather_results
//				//                //
//				int nthreads = 256;
//				int nblocks = MGPU_DIV_UP(frontier_size, nthreads);
//				vertex_centric::mgpukernel::reset_gather_result<ExpandPolicy, Program> << <nblocks, nthreads >> >(iteration[0],
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						graph_slice->m_gatherTmp,
//						graph_slice->d_visited_mask);
//
//				if (DEBUG
//						&& (retval = util::B40CPerror(
//										cudaThreadSynchronize(),
//										"gather::reset_changed Kernel failed ",
//										__FILE__, __LINE__)))
//				break;
//
//				if (DEBUG)
//				{
//					//            EValue *test_vid2 = new EValue[graph_slice->nodes];
//					//            cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//					//            printf("d_dists after apply: ");
//					//            for (int i = 0; i < graph_slice->nodes; ++i)
//					//            {
//					//              printf("%f, ", test_vid2[i]);
//					//            }
//					//            printf("\n");
//					//            delete[] test_vid2;
//					//
//					//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//					//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//					//                  printf("changed after apply: ");
//					//                  for (int i = 0; i < graph_slice->nodes; ++i)
//					//                  {
//					//                    printf("%d, ", test_vid[i]);
//					//                  }
//					//                  printf("\n");
//					//                  delete[] test_vid;
//				}
//			}
//			else if (Program::postApplyOverEdges() == POST_APPLY_ALL)
//			{
//				int nthreads = 256;
//				int nblocks = MGPU_DIV_UP(graph_slice->nodes, nthreads);
//				vertex_centric::mgpukernel::reset_gather_result<
//				ExpandPolicy, Program> << <nblocks, nthreads >> >(iteration[0],
//						graph_slice->nodes, graph_slice->vertex_list,
//						graph_slice->edge_list,
//						graph_slice->m_gatherTmp,
//						graph_slice->d_visited_mask);
//
//				if (DEBUG)
//				{
//					//                EValue *test_vid2 = new EValue[graph_slice->nodes];
//					//                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//					//                printf("d_dists after apply: ");
//					//                for (int i = 0; i < graph_slice->nodes; ++i)
//					//                {
//					//                  printf("%f, ", test_vid2[i]);
//					//                }
//					//                printf("\n");
//					//                delete[] test_vid2;
//					//
//					//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//					//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//					//                  printf("changed after apply: ");
//					//                  for (int i = 0; i < graph_slice->nodes; ++i)
//					//                  {
//					//                    printf("%d, ", test_vid[i]);
//					//                  }
//					//                  printf("\n");
//					//                  delete[] test_vid;
//				}
//
//			}
//
//			//        if (util::B40CPerror(cudaMemset(d_edge_frontier_size, 0, sizeof(SizeT)),
//			//                "CsrProblem cudaMemset d_edge_frontier_size failed", __FILE__,
//			//                __LINE__))
//			//        exit(1);
//
//			if (DEBUG)
//			{
//				cudaDeviceSynchronize();
//				startexpand = omp_get_wtime();
//			}
//
//			SYNC_CHECK();

      //
      // Expansion
      //

      //        if (frontier_size > graph_slice->nodes / RATIO)

      if (frontier_size > threshold)
      {
        expand_mgpu(graph_slice, selector, frontier_selector, directed);
      }
      else
      {
        //This function call is fine but the contract function call slow things down a lot
        expand_dynamic < ExpandPolicy > (graph_slice, directed, selector, frontier_selector, queue_index, expand_grid_size);
        cudaMemcpyAsync(h_edge_frontier_size_pinned, d_edge_frontier_size, 2 * sizeof (int), cudaMemcpyDeviceToHost);

        selector ^= 1;

        //          // Throttle
        //          if (iteration[0] & 1)
        //          {
        //            if (retval =
        //                util::B40CPerror(cudaEventRecord(throttle_event),
        //                                 "EnactorVertexCentric cudaEventRecord throttle_event failed",
        //                                 __FILE__, __LINE__))
        //              break;
        //          }
        //          else
        //          {
        //            if (retval =
        //                util::B40CPerror(
        //                                 cudaEventSynchronize(throttle_event),
        //                                 "EnactorVertexCentric cudaEventSynchronize throttle_event failed",
        //                                 __FILE__, __LINE__))
        //              break;
        //          };
      }

      //        printf("iter=%d, rank_id=%d, frontier_size=%d, edge_frontier_size=%d\n", iteration[0], rank_id, frontier_size, edge_frontier_size);

      if (DEBUG)
      {

        //          if (retval = util::B40CPerror(cudaMemcpy(&edge_frontier_size, &d_edge_frontier_size[frontier_selector], sizeof(SizeT), cudaMemcpyDeviceToHost),
        //                  "CsrProblem cudaMemcpy frontier_size failed", __FILE__,
        //                  __LINE__))
        //          break;
        //          VertexId* test_vid = new VertexId[edge_frontier_size];
        //          cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], edge_frontier_size * sizeof(VertexId), cudaMemcpyDeviceToHost);
        //          printf("Frontier after expansion: ");
        //          for (int i = 0; i < edge_frontier_size; ++i)
        //          {
        //            printf("%d, ", test_vid[i]);
        //          }
        //          printf("\n");
        //          delete[] test_vid;

        //                test_vid = new VertexId[graph_slice->nodes];
        //                cudaMemcpy(test_vid, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
        //                printf("d_dists after expansion: ");
        //                for (int i = 0; i < graph_slice->nodes; ++i)
        //                {
        //                  printf("%d, ", test_vid[i]);
        //                }
        //                printf("\n");
        //                delete[] test_vid;

        //                test_vid = new VertexId[queue_length];
        //                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_values[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
        //                printf("d_predecesor after expansion: ");
        //                for (int i = 0; i < queue_length; ++i)
        //                {
        //                  printf("%d, ", test_vid[i]);
        //                }
        //                printf("\n");
        //                delete[] test_vid;
      }

      if (DEBUG)
      {
        cudaDeviceSynchronize();
        endexpand = omp_get_wtime();
        elapsedexpand += endexpand - startexpand;
        printf("Expand time: %f ms\n", (endexpand - startexpand) * 1000);
      }

      //        if (retval = util::B40CPerror(cudaMemset(d_frontier_size, 0, sizeof(SizeT)),
      //                "CsrProblem cudaMemset d_edge_frontier_size failed", __FILE__,
      //                __LINE__))
      //        break;

      //        if (frontier_size <= graph_slice->nodes / RATIO || Program::allow_duplicates)
      if (frontier_size <= threshold || Program::allow_duplicates)
      {
        if (DEBUG)
        {
          startcontract = omp_get_wtime();
        }
        //
        // Contraction
        //

        //Using this function call is much slower, probably due to function call overhead? It is wierd
        //            contract_dynamic<ContractPolicy>(graph_slice, directed, selector, expand_grid_size);

        vertex_centric::contract_atomic::Kernel<ContractPolicy,
        Program> << <contract_grid_size,
        ContractPolicy::THREADS >> >(rank_id,
            0,
            iteration[0],
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            frontier_selector,
            d_frontier_size,
            d_edge_frontier_size,
            d_done,
            graph_slice->frontier_queues.d_keys[selector ^ 1],// filtered edge frontier in
            graph_slice->frontier_queues.d_keys[selector],// vertex frontier out
            graph_slice->frontier_queues.d_values[0],// predecessor in
            graph_slice->m_gatherTmp,
            graph_slice->vertex_list,
            graph_slice->edge_list,
            graph_slice->d_changed,
            graph_slice->d_bitmap_visited,
            graph_slice->d_visited_mask,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max filtered edge frontier vertices
            graph_slice->frontier_elements[selector],// max vertex frontier vertices
            this->contract_kernel_stats);

        if (DEBUG)
        {
          cudaDeviceSynchronize();
          endcontract = omp_get_wtime();
          elapsedcontract += endcontract - startcontract;
          printf("Contract time: %f ms\n", (endcontract - startcontract) * 1000);
        }

        selector ^= 1;
        frontier_selector ^= 1;
        //          queue_index++;

        if (DEBUG)
        {

          if (retval = util::B40CPerror(cudaMemcpy(&frontier_size, &d_frontier_size[frontier_selector], sizeof (SizeT), cudaMemcpyDeviceToHost),
                  "CsrProblem cudaMemcpy frontier_size failed", __FILE__,
                  __LINE__))
          return retval;
          //            thrust::sort(graph_slice->frontier_queues.d_keys[selector^1], graph_slice->frontier_queues.d_keys[selector^1] + frontier_size);

        }

        //                    printf("iter=%d, rank_id=%d, Frontier size after contract: %d\n", iteration[0], rank_id, frontier_size);
        //
        //                    VertexId* test_vid = new VertexId[frontier_size];
        //                    cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector^1], frontier_size * sizeof (VertexId), cudaMemcpyDeviceToHost);
        //                    //          printf("Frontier after contract: ");
        //                    for (int i = 0; i < frontier_size; ++i)
        //                    {
        //                      printf("%d, ", test_vid[i]);
        //                    }
        //                    printf("\n");
        //                    delete[] test_vid;

        // Check if done
        //          if (done[0] == 0)
        //            break;
      }
      if (retval = util::B40CPerror(cudaMemcpy(&frontier_size, &d_frontier_size[frontier_selector], sizeof (SizeT), cudaMemcpyDeviceToHost),
              "CsrProblem cudaMemcpy frontier_size failed", __FILE__,
              __LINE__))
      return retval;

      //convert compacted frontier to bitmap
      if (frontier_size > 0)
      {

        //          if (rank_id == 0)
        //          {
        //            printf("iter=%d, rank_id=%d, Frontier size after contract: %d\n", iteration[0], rank_id, frontier_size);
        //
        //            VertexId* test_vid = new VertexId[frontier_size];
        //            cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector^1], frontier_size * sizeof (VertexId), cudaMemcpyDeviceToHost);
        //            //          printf("Frontier after contract: ");
        //            for (int i = 0; i < frontier_size; ++i)
        //            {
        //              printf("%d, ", test_vid[i]);
        //            }
        //            printf("\n");
        //            delete[] test_vid;
        //
        //            test_vid = new int[graph_slice->nodes];
        //            cudaMemcpy(test_vid, graph_slice->m_gatherTmp, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToHost);
        //            printf("pi=%d, pj=%d, m_gatherTmp: ", pi, pj);
        //            for (int i = 0; i < graph_slice->nodes; ++i)
        //            {
        //              printf("%d, ", test_vid[i]);
        //            }
        //            printf("\n");
        //            delete[] test_vid;
        //          }

        int nthreads = 256;
        int nblocks = (frontier_size + nthreads - 1) / nthreads;
        MPI::mpikernel::frontier2flag<Program> << <nblocks, nthreads >> >(frontier_size, graph_slice->nodes, graph_slice->frontier_queues.d_keys[selector^1], graph_slice->d_visit_flags);
        util::B40CPerror(cudaDeviceSynchronize(), "frontier2flag", __FILE__, __LINE__);
      }

      int nthreads = 256;
      int byte_size = (graph_slice->nodes + 8 - 1) / 8;
      int nblocks = (byte_size + nthreads - 1) / nthreads;
      MPI::mpikernel::flag2bitmap<Program> << <nblocks, nthreads >> >(graph_slice->nodes, byte_size, graph_slice->d_visit_flags, graph_slice->d_bitmap_out);
      util::B40CPerror(cudaDeviceSynchronize(), "flag2bitmap", __FILE__, __LINE__);

      if (DEBUG)
      {

        //            if(rank_id == 0)
        //            {
        //              int bitmap_cout = 0;
        //              int byte_size = (graph_slice->nodes + 8 - 1) / 8;
        //              char* test_vid = new char[byte_size];
        //              cudaMemcpy(test_vid, graph_slice->d_bitmap_out, byte_size, cudaMemcpyDeviceToHost);
        //              printf("bitmap after contract: ");
        //              for (int i = 0; i < byte_size; ++i)
        //              {
        //                for (int j = 0; j < 8; j++)
        //                {
        //                  if (test_vid[i] & (1 << j))
        //                  {
        //                    printf("%d, ", i * 8 + j);
        //                    bitmap_cout++;
        //                  }
        //                }
        //              }
        //              printf("\n");
        //              printf("bitmap_cout=%d\n", bitmap_cout);
        //              delete[] test_vid;
        //            }
      }

      //          if (retval = util::B40CPerror(cudaMemset(graph_slice->d_bitmap_out, 0, (graph_slice->nodes + 8 - 1) / 8),
      //                  "Memset d_bitmap_out failed", __FILE__, __LINE__))
      //          return retval;

      if (frontier_size > 0)
      {
        if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
                "Memset d_visit_flags failed", __FILE__, __LINE__))
        return retval;
      }

      //        iteration[0]++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    if (frontier_size <= threshold)
    edge_frontier_size = h_edge_frontier_size_pinned[frontier_selector];
    //      double endTime = omp_get_wtime();
    //      double elapsed_wall = (endTime - startTime) * 1000;

    //      float elapsed;
    //      cudaEventElapsedTime(&elapsed, start, stop);
    //      std::cout << "Kernel time took: " << elapsed << " ms"
    //              << std::endl;
    //      std::cout << "Wall time took: " << elapsed_wall << " ms"
    //              << std::endl;
    //      std::cout << "Contract time took: " << elapsedcontract * 1000
    //              << " ms" << std::endl;
    //      std::cout << "Gather time took: " << elapsedgather * 1000
    //              << " ms" << std::endl;
    //      std::cout << "Expand time took: " << elapsedexpand * 1000
    //              << " ms" << std::endl;
    //
    //      printf("Total iteration: %lld\n", (long long)iteration[0]);

    //          delete[] srcs;
    //      cudaFree (graph_slice->m_gatherMapTmp);
    //      cudaFree (graph_slice->m_gatherTmp);
    //      if ( (Program::gatherOverEdges() == GATHER_ALL_EDGES || directed == 0) && Program::gatherOverEdges() == NO_GATHER_EDGES)
    //      {
    //        cudaFree (graph_slice->graph_slice->m_gatherTmp1);
    //        cudaFree (graph_slice->m_gatherTmp2);
    //      }
    //
    //      cudaFree (graph_slice->m_gatherDstsTmp);

    return retval;
  }

  cudaError_t EnactIterativeSearch(CsrProblem &csr_problem,
      typename CsrProblem::SizeT* h_row_offsets,
      int directed, int num_srcs, int* srcs, int iter_num, int threshold, int np, int device_id, int rank_id)
  {
    typedef typename CsrProblem::VertexId VertexId;
    typedef typename CsrProblem::SizeT SizeT;

    // Expansion kernel config
    typedef vertex_centric::expand_atomic::KernelPolicy<Program,
    typename CsrProblem::ProblemType, 200,// CUDA_ARCH
    INSTRUMENT,// INSTRUMENT
    1,// CTA_OCCUPANCY
    9,// LOG_THREADS
    0,// LOG_LOAD_VEC_SIZE
    0,// LOG_LOADS_PER_TILE
    5,// LOG_RAKING_THREADS
    util::io::ld::cg,// QUEUE_READ_MODIFIER,
    util::io::ld::NONE,// COLUMN_READ_MODIFIER,
    util::io::ld::NONE,// EDGE_VALUES_READ_MODIFIER,
    util::io::ld::cg,// ROW_OFFSET_ALIGNED_READ_MODIFIER,
    util::io::ld::NONE,// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
    util::io::st::cg,// QUEUE_WRITE_MODIFIER,
    false,// WORK_STEALING
    32,// WARP_GATHER_THRESHOLD
    128 * 4,// CTA_GATHER_THRESHOLD,
    7 >// LOG_SCHEDULE_GRANULARITY
    ExpandPolicy;

    // Contraction kernel config
    typedef vertex_centric::contract_atomic::KernelPolicy<Program,
    typename CsrProblem::ProblemType, 200,// CUDA_ARCH
    INSTRUMENT,// INSTRUMENT
    0,// SATURATION_QUIT
    true,// DEQUEUE_PROBLEM_SIZE
    8,// CTA_OCCUPANCY
    7,// LOG_THREADS
    1,// LOG_LOAD_VEC_SIZE
    0,// LOG_LOADS_PER_TILE
    5,// LOG_RAKING_THREADS
    util::io::ld::NONE,// QUEUE_READ_MODIFIER,
    util::io::st::NONE,// QUEUE_WRITE_MODIFIER,
    false,// WORK_STEALING
    -1,// END_BITMASK_CULL 0 to never perform bitmask filtering, -1 to always perform bitmask filtering
    8 >// LOG_SCHEDULE_GRANULARITY
    ContractPolicy;

    int expand_occupancy = ExpandPolicy::CTA_OCCUPANCY;
    int expand_grid_size = MaxGridSize(expand_occupancy);

    int contract_occupancy = ContractPolicy::CTA_OCCUPANCY;
    int contract_grid_size = MaxGridSize(contract_occupancy);
    cudaError_t retval = cudaSuccess;
    typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

//		cudaMallocHost(&m_hostMappedValue, sizeof(SizeT), cudaHostAllocMapped);
//		cudaHostGetDevicePointer(&m_deviceMappedValue, m_hostMappedValue, 0);
    cudaMalloc((void**) &m_deviceMappedValue, sizeof(SizeT));
    cudaMalloc((void**) &d_frontier_size, 2 * sizeof(SizeT));
    cudaMalloc((void**) &d_edge_frontier_size, 2 * sizeof(SizeT));

    if (retval = util::B40CPerror(cudaMemset(d_edge_frontier_size, 0, 2 * sizeof(SizeT)),
            "CsrProblem cudaMemset d_edge_frontier_size failed", __FILE__,
            __LINE__))
    return retval;

    if(retval = util::B40CPerror(cudaMalloc((void**) &m_gatherMapTmp, (graph_slice->edges + graph_slice->nodes) * sizeof(GatherType)),
            "CsrProblem cudaMalloc m_gatherMapTmp failed", __FILE__,
            __LINE__))
    return retval;

    if(retval = util::B40CPerror(cudaMalloc((void**) &m_gatherTmp, graph_slice->nodes * sizeof(GatherType)),
            "CsrProblem cudaMalloc m_gatherTmp failed", __FILE__,
            __LINE__))
    return retval;

    if(retval = util::B40CPerror(cudaMalloc((void**) &m_reduceResult, graph_slice->nodes * sizeof(GatherType)),
            "CsrProblem cudaMalloc m_reduceResult failed", __FILE__,
            __LINE__))
    return retval;

    if(retval = util::B40CPerror(cudaMalloc((void**) &m_gatherDstsTmp, (graph_slice->edges + graph_slice->nodes) * sizeof(VertexId)),
            "CsrProblem cudaMalloc m_gatherDstsTmp failed", __FILE__,
            __LINE__))
    return retval;

    if ( (Program::gatherOverEdges() == GATHER_ALL_EDGES || directed == 0) && Program::gatherOverEdges() != NO_GATHER_EDGES)
    {
      cudaMalloc((void**) &m_gatherTmp1, graph_slice->nodes * sizeof(GatherType));
      //        cudaMemset(graph_slice->m_gatherTmp1, 0, graph_slice->nodes * sizeof(GatherType) );
      cudaMalloc((void**) &m_gatherTmp2, graph_slice->nodes * sizeof(GatherType));
    }

    int p = sqrt(np);// assuming that np is squre of an int
    int pi = rank_id / p;
    int pj = rank_id % p;

    //The src vertices range
    const int VERT_PER_NODE = graph_slice->nodes;
    int vertex_id_start = VERT_PER_NODE * pj;
    int vertex_id_end = vertex_id_start + VERT_PER_NODE;

    vector<int> local_srcs;
    local_srcs.reserve(num_srcs);

    for (int i = 0; i < num_srcs; i++)
    {
      if (srcs[i] >= vertex_id_start && srcs[i] < vertex_id_end)
      {
        local_srcs.push_back(srcs[i] - vertex_id_start);
      }
    }
    //for (int i = 0; i < num_srcs; i++)
    //{
    //  local_srcs.push_back(srcs[i]);
    //}
    frontier_size = local_srcs.size();

    int tmp[2] =
    { frontier_size, 0};
    if (retval = util::B40CPerror(
            cudaMemcpy(d_frontier_size,
                tmp,
                2 * sizeof (int),
                cudaMemcpyHostToDevice),
            "CsrProblem cudaMemcpy d_frontier_size failed",
            __FILE__, __LINE__))
    return retval;

    double max_queue_sizing = cfg.getParameter<double>("max_queue_sizing");
    // Reset data
    if (retval = csr_problem.Reset(GetFrontierType(),
            max_queue_sizing))
    return retval;

//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();

    thrust::device_vector<int> d_local_srcs = local_srcs;
    int byte_size = (graph_slice->nodes + 8 - 1) / 8;
    //check if there is a src belongs to this process column
    if (local_srcs.size() > 0)
    {
      int nthreads = 256;
      int nblocks = (local_srcs.size() + nthreads - 1) / nthreads;
      MPI::mpikernel::frontier2flag<Program> << <nblocks, nthreads >> >(local_srcs.size(), graph_slice->nodes, thrust::raw_pointer_cast(&d_local_srcs[0]), graph_slice->d_visit_flags);
      util::B40CPerror(cudaDeviceSynchronize(), "frontier2flag", __FILE__, __LINE__);

      nblocks = (byte_size + nthreads - 1) / nthreads;
      MPI::mpikernel::flag2bitmap<Program> << <nblocks, nthreads >> >(graph_slice->nodes, byte_size, graph_slice->d_visit_flags, graph_slice->d_bitmap_in);
      util::B40CPerror(cudaDeviceSynchronize(), "flag2bitmap", __FILE__, __LINE__);

      if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
              "Memset d_visit_flags failed", __FILE__, __LINE__))
      return retval;
    }

    Program::Initialize(directed, graph_slice->nodes, graph_slice->edges, frontier_size, &local_srcs[0],
        graph_slice->d_row_offsets, graph_slice->d_column_indices, graph_slice->d_column_offsets, graph_slice->d_row_indices,
        graph_slice->d_edge_values,
        graph_slice->vertex_list, graph_slice->edge_list,
        graph_slice->frontier_queues.d_keys,
        graph_slice->frontier_queues.d_values);

//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();


    //check if Vs is in Rj
    vertex_id_start = VERT_PER_NODE * pi;
    vertex_id_end = vertex_id_start + VERT_PER_NODE;

    local_srcs.clear();
    for (int i = 0; i < num_srcs; i++)
    {
      if (srcs[i] >= vertex_id_start && srcs[i] < vertex_id_end)
      {
        local_srcs.push_back(srcs[i] - vertex_id_start);
      }
    }

    d_local_srcs = local_srcs;

    //check if there is a src belongs to this process row
    if (local_srcs.size() > 0)
    {
      int nthreads = 256;
      int nblocks = (local_srcs.size() + nthreads - 1) / nthreads;
      MPI::mpikernel::frontier2flag<Program> << <nblocks, nthreads >> >(local_srcs.size(), graph_slice->nodes, thrust::raw_pointer_cast(&d_local_srcs[0]), graph_slice->d_visit_flags);
      util::B40CPerror(cudaDeviceSynchronize(), "frontier2flag", __FILE__, __LINE__);

      nblocks = (byte_size + nthreads - 1) / nthreads;
      MPI::mpikernel::flag2bitmap<Program> << <nblocks, nthreads >> >(graph_slice->nodes, byte_size, graph_slice->d_visit_flags, graph_slice->d_bitmap_visited);
      util::B40CPerror(cudaDeviceSynchronize(), "flag2bitmap", __FILE__, __LINE__);

      if (b40c::util::B40CPerror(
              cudaMemcpy(graph_slice->frontier_queues.d_keys[2], thrust::raw_pointer_cast(&d_local_srcs[0]), local_srcs.size() * sizeof (int),
                  cudaMemcpyDeviceToDevice),
              "CsrProblem cudaMemcpy d_frontier_keys[2] failed", __FILE__,
              __LINE__))
      exit(1);

      if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
              "Memset d_visit_flags failed", __FILE__, __LINE__))
      return retval;

      //init bitmap_out for the process row of the src vertex
      if (retval = cudaMemcpy(graph_slice->d_bitmap_out, graph_slice->d_bitmap_visited, byte_size, cudaMemcpyDeviceToDevice))
      return retval;

      frontier_size = local_srcs.size();
    }
    else
    {
      frontier_size = 0;
    }

		if (rank_id == 0)
		{
			unsigned char* test_vid = new unsigned char[byte_size];
			cudaMemcpy(test_vid, graph_slice->d_bitmap_visited, byte_size * sizeof (unsigned char), cudaMemcpyDeviceToHost);
			printf("pi=%d, pj=%d, initial d_bitmap_visited: ", pi, pj);
//			for (int i = 0; i < byte_size; ++i)
			  for (int i = 0; i < 20; ++i)
			{
				printf("%d, ", test_vid[i]);
			}
			printf("\n");
			delete[] test_vid;
		}

//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();

    if (rank_id == 0)
                  {
                    DataType* test_vid2 = new DataType[graph_slice->nodes];
                    cudaMemcpy(test_vid2, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (DataType), cudaMemcpyDeviceToHost);
                    printf("pi=%d, pj=%d, init d_labels before: ", pi, pj);
    //                for (int i = 0; i < graph_slice->nodes; ++i)
                    for (int i = 0; i < 20; ++i)
                    {
                      printf("%f, ", test_vid2[i]);
                    }
                    printf("\n");
                    delete[] test_vid2;
                  }


    //if (pj == p - 1)
    {
      int nthreads = 256;
      int nblocks = (graph_slice->nodes + nthreads - 1) / nthreads;
      update_BFS_labels<Program> << <nblocks, nthreads >> >(rank_id, 0, graph_slice->nodes, graph_slice->d_bitmap_in,
          graph_slice->d_bitmap_visited, graph_slice->vertex_list);

              if (rank_id == 0)
              {
                DataType* test_vid2 = new DataType[graph_slice->nodes];
                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (DataType), cudaMemcpyDeviceToHost);
                printf("pi=%d, pj=%d, init d_labels: ", pi, pj);
//                for (int i = 0; i < graph_slice->nodes; ++i)
                for (int i = 0; i < 20; ++i)
                {
                  printf("%f, ", test_vid2[i]);
                }
                printf("\n");
                delete[] test_vid2;
              }
    }
//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();

    if (retval = Setup(csr_problem, expand_grid_size,
            contract_grid_size, 0))
    return retval;

    //          SizeT queue_length;
    //      VertexId queue_index = 0;// Work stealing/queue index
    int selector = 0;
    int frontier_selector = 0;
    long long global_frontier_size = num_srcs;

    m_mgpuContext = mgpu::CreateCudaDevice(device_id);
    stats = new Statistics(rank_id, np);

    //			wave w1(pi, pj, p, graph_slice->nodes, stats);
    wave w(pi, pj, p, graph_slice->nodes, stats);
    ValueCommunicator<GatherType, Program> valComm(pi, pj, p, graph_slice->nodes, stats);
    //MPI_Barrier(MPI_COMM_WORLD);
    //      printf("\nmyid:%d Wave initiaized time:%lf\n", rank_id, MPI_Wtime());
    int NUM_WARMUP = cfg.getParameter<int>("warmup");
    int compressed = cfg.getParameter<int>("compressed");
    int dummy_frontier_size;
    int dummy_size = 1000;
    thrust::device_vector<int> dummy_d_frontier_size(1, 0);
    thrust::device_vector<int> dummy_d_result(dummy_size, 0);
    thrust::device_vector<int> dummy_flags(1000, 1);

//		SYNC_CHECK();
    if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
            __FILE__,
            __LINE__) != cudaSuccess)
    throw std::exception();
//      for (int i = 0; i < NUM_WARMUP; i++)
//      {
//        if (compressed)
//        {
//          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_out, graph_slice->d_bitmap_out);
//          w.broadcast_new_frontier(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        }
//        else
//        {
//          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
//          w.broadcast_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in, -1);
//        }
//        //        copy_if_mgpu(1000,
//        //                     thrust::raw_pointer_cast(&dummy_flags[0]),
//        //                     thrust::raw_pointer_cast(&dummy_d_result[0]),
//        //                     NULL,
//        //                     NULL,
//        //                     m_mgpuContext);
//      }
//      if (rank_id == 0)
//        printf("Warmup Done\n");

    Statistics::stats_per_iter iter_stat;
    double start_time, end_time, total_start, total_end;

//      MPI_Barrier(MPI_COMM_WORLD);
    double total_run_time_start = MPI_Wtime();
    int iter;

    for (iter = 0; iter < iter_num; iter++)
    {
			printf("iter=%d, rank_id=%d\n", iter, rank_id);
      total_start = MPI_Wtime();

      {
        int byte_size = (graph_slice->nodes + 8 - 1) / 8;
        ////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
        int nthreads = 256;
        int nblocks = (byte_size + nthreads - 1) / nthreads;

        MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_out, graph_slice->d_visit_flags);
        util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);

//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
        copy_if_mgpu(graph_slice->nodes,
            graph_slice->d_visit_flags,
            graph_slice->frontier_queues.d_keys[2],
            &d_frontier_size[frontier_selector],
            &frontier_size,
            m_mgpuContext);

        if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
                "Memset d_visit_flags failed", __FILE__, __LINE__))
        return retval;
//				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
      }

      if(rank_id == 0)
        printf("rank_id=%d, frontier_size=%d\n", rank_id, frontier_size);

      /**
       * Gather Stage
       */
      if (Program::gatherOverEdges() != NO_GATHER_EDGES)
      gather_mgpu(graph_slice, selector, directed, rank_id);

//			if (DEBUG)
//			{
//				cudaDeviceSynchronize();
//				endgather = time(NULL);
//				elapsedgather += difftime(endgather, startgather);
//				printf("Gather time: %f ms\n", difftime(endgather, startgather) * 1000);
//			}

      //            if (DEBUG)
      //            {
      //              if (work_progress.GetQueueLength(queue_index,
      //                  queue_length))
      //                break;
      //              total_queued += queue_length;
      //
      //              //                  if (DEBUG) printf("queue_length after gather: %lld\n", (long long) queue_length);
      //
      //              EValue *test_vid2 = new EValue[graph_slice->nodes];
      //              cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
      //              printf("d_min_dists after gather: ");
      //              for (int i = 0; i < graph_slice->nodes; ++i)
      //              {
      //                printf("%f, ", test_vid2[i]);
      //              }
      //              printf("\n");
      //              delete[] test_vid2;
      //              //
      //              //                  test_vid2 = new EValue[graph_slice->nodes];
      //              //                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
      //              //                  printf("d_gather_results after gather: ");
      //              //                  for (int i = 0; i < graph_slice->nodes; ++i)
      //              //                  {
      //              //                    printf("%f, ", test_vid2[i]);
      //              //                  }
      //              //                  printf("\n");
      //              //                  delete[] test_vid2;
      //
      //              //                  VertexId* test_vid = new VertexId[graph_slice->nodes];
      //              //                  cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
      //              //                  printf("changed after gather: ");
      //              //                  for (int i = 0; i < graph_slice->nodes; ++i)
      //              //                  {
      //              //                    printf("%d, ", test_vid[i]);
      //              //                  }
      //              //                  printf("\n");
      //              //                  delete[] test_vid;
      //            }

      /**
       * Value reduce
       */
      valComm.valueReduce(m_gatherTmp, m_reduceResult, frontier_size, Program::INIT_VALUE, graph_slice->nodes);

      /*
       * Prepare frontier for apply
       */
      {
        int byte_size = (graph_slice->nodes + 8 - 1) / 8;
        ////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
        int nthreads = 256;
        int nblocks = (byte_size + nthreads - 1) / nthreads;

        MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_in, graph_slice->d_visit_flags);
        util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);

        //				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
        copy_if_mgpu(graph_slice->nodes,
            graph_slice->d_visit_flags,
            graph_slice->frontier_queues.d_keys[2],
            &d_frontier_size[frontier_selector],
            &frontier_size,
            m_mgpuContext);

        if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
                "Memset d_visit_flags failed", __FILE__, __LINE__))
        return retval;
        //				SYNC_CHECK();
        if (util::B40CPerror(cudaDeviceSynchronize(), "Error!"
                __FILE__,
                __LINE__) != cudaSuccess)
        throw std::exception();
      }

      /**
       * Apply Stage
       */
//			printf("rank_id=%d, frontier_size before apply: %d\n", rank_id, frontier_size);
      if (Program::applyOverEdges() == APPLY_FRONTIER)
      {
        if(frontier_size > 0)
        apply_mgpu(graph_slice, selector, rank_id);
      }

//			/**
//			 * Post Apply Stage
//			 */
//			if (Program::postApplyOverEdges() == POST_APPLY_FRONTIER)
//			{
//				//
//				//                //reset dists and gather_results
//				//                //
//				int nthreads = 256;
//				int nblocks = MGPU_DIV_UP(frontier_size, nthreads);
//				vertex_centric::mgpukernel::reset_gather_result<ExpandPolicy, Program><<<nblocks, nthreads>>>(iteration[0],
//						frontier_size,
//						graph_slice->frontier_queues.d_keys[selector ^ 1],
//						graph_slice->vertex_list,
//						graph_slice->edge_list,
//						m_gatherTmp,
//						graph_slice->d_visited_mask);
//
//				if (DEBUG
//						&& (retval = util::B40CPerror(
//										cudaThreadSynchronize(),
//										"gather::reset_changed Kernel failed ",
//										__FILE__, __LINE__)))
//				break;
//
//				if (DEBUG)
//				{
//					//            EValue *test_vid2 = new EValue[graph_slice->nodes];
//					//            cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//					//            printf("d_dists after apply: ");
//					//            for (int i = 0; i < graph_slice->nodes; ++i)
//					//            {
//					//              printf("%f, ", test_vid2[i]);
//					//            }
//					//            printf("\n");
//					//            delete[] test_vid2;
//					//
//					//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//					//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//					//                  printf("changed after apply: ");
//					//                  for (int i = 0; i < graph_slice->nodes; ++i)
//					//                  {
//					//                    printf("%d, ", test_vid[i]);
//					//                  }
//					//                  printf("\n");
//					//                  delete[] test_vid;
//				}
//			}
//			else if (Program::postApplyOverEdges() == POST_APPLY_ALL)
//			{
//				int nthreads = 256;
//				int nblocks = MGPU_DIV_UP(graph_slice->nodes, nthreads);
//				vertex_centric::mgpukernel::reset_gather_result<
//				ExpandPolicy, Program><<<nblocks,nthreads>>>(iteration[0],
//						graph_slice->nodes, graph_slice->vertex_list,
//						graph_slice->edge_list,
//						m_gatherTmp,
//						graph_slice->d_visited_mask);
//
//				if (DEBUG)
//				{
//					//                EValue *test_vid2 = new EValue[graph_slice->nodes];
//					//                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//					//                printf("d_dists after apply: ");
//					//                for (int i = 0; i < graph_slice->nodes; ++i)
//					//                {
//					//                  printf("%f, ", test_vid2[i]);
//					//                }
//					//                printf("\n");
//					//                delete[] test_vid2;
//					//
//					//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//					//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//					//                  printf("changed after apply: ");
//					//                  for (int i = 0; i < graph_slice->nodes; ++i)
//					//                  {
//					//                    printf("%d, ", test_vid[i]);
//					//                  }
//					//                  printf("\n");
//					//                  delete[] test_vid;
//				}
//
//			}
//
      /*
       * prepare frontier for scatter
       */
      int byte_size = (graph_slice->nodes + 8 - 1) / 8;
      ////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
      int nthreads = 256;
      int nblocks = (byte_size + nthreads - 1) / nthreads;
      MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_in, graph_slice->d_visit_flags);
      //				util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);

      //				SYNC_CHECK();
      copy_if_mgpu(graph_slice->nodes,
          graph_slice->d_visit_flags,
          graph_slice->frontier_queues.d_keys[selector ^ 1],
          &d_frontier_size[frontier_selector],
          &frontier_size,
          m_mgpuContext);

//			{
//				char* test_vid3 = new char[graph_slice->nodes];
//				cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
//				printf("rank_id=%d, d_visit_flags in: ", rank_id);
//				for (int i = 0; i < graph_slice->nodes; ++i)
//				{
//					printf("%d, ", test_vid3[i]);
//				}
//				printf("\n");
//				delete[] test_vid3;
//
      if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
              "Memset d_visit_flags failed", __FILE__, __LINE__))
      return retval;
//			}

      //				SYNC_CHECK();

      /*
       * Scatter Stage
       */
      start_time = MPI_Wtime();
      if (frontier_size > 0)
      {
        retval = EnactIterativeSearch<ExpandPolicy, ContractPolicy > (csr_problem, h_row_offsets, directed, threshold,
            expand_grid_size, contract_grid_size, selector, frontier_selector, pi, pj, rank_id);
      }

//			{
//				char* test_vid3 = new char[graph_slice->nodes];
//				MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_out, graph_slice->d_visit_flags);
//				cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
//				printf("rank_id=%d, d_visit_flags out before wave: ", rank_id);
//				for (int i = 0; i < graph_slice->nodes; ++i)
//				{
//					printf("%d, ", test_vid3[i]);
//				}
//				printf("\n");
//				delete[] test_vid3;
//
//				if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
//								"Memset d_visit_flags failed", __FILE__, __LINE__))
//				return retval;
//			}

      byte_size = (graph_slice->nodes + 8 - 1) / 8;
      nthreads = 256;
      nblocks = (byte_size + nthreads - 1) / nthreads;
      bitsubstract << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_out, graph_slice->d_bitmap_visited, graph_slice->d_bitmap_out);
      util::B40CPerror(cudaDeviceSynchronize(), "bitsubstract", __FILE__, __LINE__);

      thrust::device_ptr<unsigned char> d_bitmap_out_ptr(graph_slice->d_bitmap_out);
      unsigned int red_result = (unsigned int)thrust::reduce(d_bitmap_out_ptr, d_bitmap_out_ptr+byte_size, (unsigned int)0, bitcount());
      end_time = MPI_Wtime();
      //        stats->total_GPU_time += end_time - start_time;
      iter_stat.GPU_time = end_time - start_time;

      // long long count = 0;
      // if (pj == p - 1)
      // {
      //  int mesg_size = (graph_slice->nodes + 8 - 1) / 8;
      //  char *in_h = (char*)malloc(mesg_size);
      //   cudaMemcpy(in_h, graph_slice->d_bitmap_out, mesg_size, cudaMemcpyDeviceToHost);
      //#pragma omp parallel for reduction(+:count)
      //     for (int i = 0; i < mesg_size; i++)
      //      {
      //        count += (int)(in_h[i] >> 0 & 1);
      //        count += (int)(in_h[i] >> 1 & 1);
      //        count += (int)(in_h[i] >> 2 & 1);
      //        count += (int)(in_h[i] >> 3 & 1);
      //        count += (int)(in_h[i] >> 4 & 1);
      //        count += (int)(in_h[i] >> 5 & 1);
      //        count += (int)(in_h[i] >> 6 & 1);
      //         count += (int)(in_h[i] >> 7 & 1);
      //        }
      //         free(in_h);
      //        }
      //
      frontier_size = red_result;
      iter_stat.frontier_size = frontier_size;
      iter_stat.edge_frontier_size = edge_frontier_size;

      iteration[0]++;

      //        long long tmp_frontier_size = frontier_size;
      //        //check if done
      //        start_time = MPI_Wtime();
      //        MPI_Allreduce(&tmp_frontier_size, &global_frontier_size, 1,
      //            MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      //
      //        end_time = MPI_Wtime();
      //        if (global_frontier_size == 0)
      //        {
      //          //update predecessors
      //          iter_stat.wave_time = 0.0;
      //          w.bitunion_time = 0.0;
      //          //break;
      //        }
      //        else
      //        {
//        start_time = MPI_Wtime();
//        //          if (iteration[0] != 1)
//        //          {
//        //            if (compressed)
//        //            {
//        //              w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
//        //              w.broadcast_new_frontier(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        //            }
//        //            else
//        //            {
//        //              w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
//        //              w.broadcast_send(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        //            }
//        //            //  w.reduce_frontier_GDR(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        //            //        w.reduce_frontier_CPU(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        //
//        //            end_time = MPI_Wtime();
//        //          }
//        //          else
//        //          {
//
//        if (compressed)
//        {
//
//          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
//          w.broadcast_new_frontier(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        }
//        else
//        {
//          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
//          w.broadcast_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in, iter);
//        }
//        //  w.reduce_frontier_GDR(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        //        w.reduce_frontier_CPU(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//
//        end_time = MPI_Wtime();
//
//        //          }
//        iter_stat.wave_time = end_time - start_time;
      //        }

      long long tmp_frontier_size = frontier_size;
      //check if done
      start_time = MPI_Wtime();
      MPI_Allreduce(&tmp_frontier_size, &global_frontier_size, 1,
          MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

      end_time = MPI_Wtime();
      iter_stat.allreduce_time = end_time - start_time;

       if(rank_id == 0) printf("rank_id=%d, global_frontier_size=%d\n", rank_id, global_frontier_size);
      if (global_frontier_size == 0)
      {
        //update predecessors
        iter_stat.wave_time = 0.0;
        w.bitunion_time = 0.0;
        //break;
      }
      else
      {
        start_time = MPI_Wtime();

        if (compressed)
        {
          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
          w.broadcast_new_frontier(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
        }
        else
        {
          w.propogate_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_assigned, graph_slice->d_bitmap_prefix);
          w.broadcast_tree(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in, iter);
        }
        end_time = MPI_Wtime();
        iter_stat.wave_time = end_time - start_time;
      }

//			{
//				int byte_size = (graph_slice->nodes + 8 - 1) / 8;
//				////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
//				int nthreads = 256;
//				int nblocks = (byte_size + nthreads - 1) / nthreads;
//				MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_out, graph_slice->d_visit_flags);
//				util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);
//
//				char* test_vid3 = new char[graph_slice->nodes];
//				cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
//				printf("rank_id=%d, d_visit_flags out after wave: ", rank_id);
//				for (int i = 0; i < graph_slice->nodes; ++i)
//				{
//					printf("%d, ", test_vid3[i]);
//				}
//				printf("\n");
//				delete[] test_vid3;
//				if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
//								"Memset d_visit_flags failed", __FILE__, __LINE__))
//				return retval;
//
//				MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_in, graph_slice->d_visit_flags);
//				util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);
//
//				test_vid3 = new char[graph_slice->nodes];
//				cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
//				printf("rank_id=%d, d_visit_flags in after wave: ", rank_id);
//				for (int i = 0; i < graph_slice->nodes; ++i)
//				{
//					printf("%d, ", test_vid3[i]);
//				}
//				printf("\n");
//				delete[] test_vid3;
//
//				if (retval = util::B40CPerror(cudaMemset(graph_slice->d_visit_flags, 0, graph_slice->nodes * sizeof (char)),
//								"Memset d_visit_flags failed", __FILE__, __LINE__))
//				return retval;
//			}

      //  w.reduce_frontier_GDR(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);
//        w.reduce_frontier_CPU(graph_slice->d_bitmap_out, graph_slice->d_bitmap_in);

      //          }

      //        }

      //        long long tmp_frontier_size = frontier_size;
      //        //check if done
      //        start_time = MPI_Wtime();
      //        MPI_Allreduce(&tmp_frontier_size, &global_frontier_size, 1,
      //                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      //
      //        end_time = MPI_Wtime();
      //
      //        if (global_frontier_size == 0)
      //        {
      //          //update predecessors
      //          iter_stat.wave_time = 0.0;
      //          w.bitunion_time = 0.0;
      //          //break;
      //        }

      iter_stat.propagate_time = w.propagate_time;
      iter_stat.broadcast_time = w.broadcast_time;
      iter_stat.propagate_wait = w.propagate_wait;
      iter_stat.broadcast_wait = w.broadcast_wait;
      iter_stat.prop_row = w.prop_row;
      iter_stat.prop_col = w.prop_col;
      iter_stat.copy_time = w.copy_time;
      iter_stat.bitunion_time = w.bitunion_time;
      iter_stat.compression_time = w.compression_time;
      iter_stat.compressed_size = w.compressed_size;

      start_time = MPI_Wtime();
      //update bitmap_visited
      {
        int nthreads = 256;
        int nblocks = (byte_size + nthreads - 1) / nthreads;
        bitunion << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_out, graph_slice->d_bitmap_visited, graph_slice->d_bitmap_visited);
        util::B40CPerror(cudaDeviceSynchronize(), "bitunion", __FILE__, __LINE__);
      }

//			//update labels
//			//        if (pj == p - 1)
//			{
//				int nthreads = 256;
//				int nblocks = (graph_slice->nodes + nthreads - 1) / nthreads;
//				update_BFS_labels<Program> << <nblocks, nthreads >> >(rank_id, iteration[0], graph_slice->nodes, graph_slice->d_bitmap_in, graph_slice->d_bitmap_out, graph_slice->vertex_list);
//				//          if (rank_id == 1)
//				//          {
//				//            int* test_vid2 = new int[graph_slice->nodes];
//				//            cudaMemcpy(test_vid2, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToHost);
//				//            printf("pi=%d, pj=%d, iter=%d, d_labels: ", pi, pj, iteration[0]);
//				//            for (int i = 0; i < graph_slice->nodes; ++i)
//				//            {
//				//              printf("%d, ", test_vid2[i]);
//				//            }
//				//            printf("\n");
//				//            delete[] test_vid2;
//				//
//				//          }
//			}

      //        fflush(stdout);
      //        usleep(200);
      //        MPI_Barrier(MPI_COMM_WORLD);

      //        if (rank_id == 1)
      //        {
      //          unsigned char* test_vid3 = new unsigned char[1];
      //          cudaMemcpy(test_vid3, graph_slice->d_bitmap_visited, 1, cudaMemcpyDeviceToHost);
      //          //        printf("pi=%d, pj=%d, iter=%d, d_bitmap_visited: %d ", pi, pj, iteration[0], test_vid3[0]);
      //          //        for (int i = 0; i < 1; ++i)
      //          //        {
      //          //          bitset < 8 > b(test_vid3[i]);
      //          //          cout << b;
      //          //        }
      //          bitset < 8 > b(test_vid3[0]);
      //          cout << "pi=" << pi << " pj=" << pj << " iter=" << iter << " d_bitmap_visited=" << b << endl;
      //          //        printf("\n");
      //          delete[] test_vid3;
      //        }
      //        fflush(stdout);
      //        usleep(200);
      //        MPI_Barrier(MPI_COMM_WORLD);

      end_time = MPI_Wtime();

      iter_stat.update_time = end_time - start_time;

      total_end = MPI_Wtime();
      iter_stat.iter_total = total_end - total_start;
      stats->iter_stats.push_back(iter_stat);

      if (global_frontier_size == 0)
      break;
    }

    /*
     //      cudaDeviceSynchronize();
     //      MPI_Barrier(MPI_COMM_WORLD);
     start_time = MPI_Wtime();
     ////      if (pi == pj)
     ////      {
     ////        cudaMemcpy(graph_slice->vertex_list.d_labels_src, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToDevice);
     ////      }
     ////
     ////      MPI_Bcast(graph_slice->vertex_list.d_labels_src, graph_slice->nodes, MPI_INT, pj, w.new_col_comm);

     byte_size = (graph_slice->nodes + 8 - 1) / 8;
     ////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
     int nthreads = 256;
     int nblocks = (byte_size + nthreads - 1) / nthreads;
     MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_assigned, graph_slice->d_visit_flags);
     util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);

     copy_if_mgpu(graph_slice->nodes,
     graph_slice->d_visit_flags,
     graph_slice->frontier_queues.d_keys[0],
     NULL,
     &frontier_size,
     m_mgpuContext);

     EdgeCountIterator ecIterator(graph_slice->d_column_offsets,
     graph_slice->frontier_queues.d_keys[0]);

     mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
     ecIterator,
     frontier_size,
     0,
     mgpu::plus<int>(),
     m_deviceMappedValue,
     (int *)NULL,
     graph_slice->d_edgeCountScan,
     *m_mgpuContext);

     int n_active_edges;
     cudaMemcpy(&n_active_edges, m_deviceMappedValue,
     sizeof (int),
     cudaMemcpyDeviceToHost);

     //          printf("n_active_edges = %d, frontier_size = %d\n", n_active_edges, frontier_size);



     const int nThreadsPerBlock = 128;
     MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
     (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, frontier_size,
     nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);

     SizeT nBlocks = MGPU_DIV_UP(n_active_edges + frontier_size,
     nThreadsPerBlock);
     dim3 grid = vertex_centric::mgpukernel::calcGridDim(nBlocks);

     vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VertexId,
     nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(
     frontier_size,
     graph_slice->frontier_queues.d_keys[0],
     nBlocks,
     n_active_edges,
     graph_slice->d_edgeCountScan,
     partitions->get(),
     graph_slice->d_column_offsets,
     graph_slice->d_row_indices,
     graph_slice->vertex_list,
     graph_slice->edge_list,
     NULL,
     graph_slice->m_gatherDstsTmp,
     graph_slice->m_gatherMapTmp);

     mgpu::ReduceByKey(
     graph_slice->m_gatherDstsTmp,
     graph_slice->m_gatherMapTmp,
     n_active_edges,
     Program::INIT_VALUE,
     ReduceFunctor(), mgpu::equal_to<VertexId > (),
     (VertexId *)NULL,
     ReduceOutputIterator(graph_slice->m_gatherTmp, graph_slice->frontier_queues.d_keys[0]),
     //                        graph_slice->m_gatherTmp,
     NULL,
     NULL,
     *m_mgpuContext);
     graph_slice->predecessor_size = frontier_size;
     end_time = MPI_Wtime();
     //      thrust::device_ptr<int> m_gatherTmp_ptr(graph_slice->m_gatherTmp);
     //      long long pred_sum = thrust::reduce(m_gatherTmp_ptr, m_gatherTmp_ptr + graph_slice->nodes);
     //      printf("rank_id %d pred_sum %lld\n", rank_id, pred_sum);

     //      if (rank_id == 1)
     //      {
     //        char* test_vid3 = new char[graph_slice->nodes];
     //        cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
     //        printf("d_visit_flags: ");
     //        for (int i = 0; i < graph_slice->nodes; ++i)
     //        {
     //          printf("%d, ", test_vid3[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid3;
     //
     //        int* test_vid2 = new int[n_active_edges];
     //        cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
     //        printf("m_gatherMapTmp: ");
     //        for (int i = 0; i < n_active_edges; ++i)
     //        {
     //          printf("%d, ", test_vid2[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid2;
     //
     //        test_vid2 = new int[n_active_edges];
     //        cudaMemcpy(test_vid2, graph_slice->m_gatherDstsTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
     //        printf("m_gatherDstsTmp: ");
     //        for (int i = 0; i < n_active_edges; ++i)
     //        {
     //          printf("%d, ", test_vid2[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid2;
     //
     //        test_vid2 = new int[graph_slice->nodes];
     //        cudaMemcpy(test_vid2, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToHost);
     //        printf("d_labels: ");
     //        for (int i = 0; i < graph_slice->nodes; ++i)
     //        {
     //          printf("%d, ", test_vid2[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid2;
     //
     //        test_vid2 = new int[frontier_size];
     //        cudaMemcpy(test_vid2, graph_slice->frontier_queues.d_keys[0], frontier_size * sizeof (int), cudaMemcpyDeviceToHost);
     //        printf("d_keys: ");
     //        for (int i = 0; i < frontier_size; ++i)
     //        {
     //          printf("%d, ", test_vid2[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid2;
     //
     //        test_vid2 = new int[frontier_size];
     //        cudaMemcpy(test_vid2, graph_slice->m_gatherTmp, frontier_size * sizeof (int), cudaMemcpyDeviceToHost);
     //        printf("m_gather: ");
     //        for (int i = 0; i < frontier_size; ++i)
     //        {
     //          printf("%d, ", test_vid2[i]);
     //        }
     //        printf("\n");
     //        delete[] test_vid2;
     //
     //      }
     //
     //      fflush(stdout);
     //      usleep(1000);
     */
    MPI_Barrier(MPI_COMM_WORLD);
    double total_run_time_end = MPI_Wtime();
    double pred_time = end_time - start_time;

//       for (int i = 0; i < stats->iter_stats.size(); i++)
//       {
//       double prop_max, prop_min, prop_avg, bcast_max, bcast_min, bcast_avg, GPUtime_max, GPUtime_min, GPUtime_avg, compression, decompression, compression_ratio, cr_bcast;
//       MPI_Reduce(&stats->iter_stats[i].propagate_time, &prop_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&stats->iter_stats[i].broadcast_time, &bcast_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&stats->iter_stats[i].propagate_time, &prop_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&stats->iter_stats[i].broadcast_time, &bcast_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&stats->iter_stats[i].propagate_time, &prop_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&stats->iter_stats[i].broadcast_time, &bcast_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//       if (rank_id == 0)
//       {
//       printf("%d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
//       i, prop_max, prop_min, prop_avg / 49.0, bcast_max, bcast_min, bcast_avg / 49.0, GPUtime_max, GPUtime_min, GPUtime_avg / 49.0);
//       }
//       }

    if (rank_id == 0)
    {
      printf("Iter+Rank_id GPU_time termination_check Propagate bcast_row bcast_col update_vertices pred_time bitunion_time copy_time iter_total_time wave-bitunion Broadcast PropagateWait BroadcastWait total_time frontier_size edge_frontier_size\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000);
    long long global_size = 0;

    for (int i = 0; i < stats->iter_stats.size(); i++)
    {
      double finel_pred_time = 0.0;
      if (i == stats->iter_stats.size() - 1)
      {
        finel_pred_time = pred_time;
      }
      printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lld %lld\n",
          i + rank_id / 100.0,
          stats->iter_stats[i].GPU_time,
          stats->iter_stats[i].allreduce_time,
          stats->iter_stats[i].propagate_time,
          stats->iter_stats[i].prop_row,
          stats->iter_stats[i].prop_col,
          stats->iter_stats[i].update_time,
          finel_pred_time,
          stats->iter_stats[i].bitunion_time,
          stats->iter_stats[i].copy_time,
          stats->iter_stats[i].iter_total,
          stats->iter_stats[i].wave_time - stats->iter_stats[i].bitunion_time,
          stats->iter_stats[i].broadcast_time,
          stats->iter_stats[i].propagate_wait,
          stats->iter_stats[i].broadcast_wait,
          stats->iter_stats[i].GPU_time + stats->iter_stats[i].wave_time + stats->iter_stats[i].allreduce_time + stats->iter_stats[i].update_time,
          stats->iter_stats[i].frontier_size,
          stats->iter_stats[i].edge_frontier_size);

      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
      usleep(1000);

      //        MPI_Reduce(&stats->iter_stats[i].frontier_size, &global_size, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      //        if (rank_id == 0)
      //          printf("%d %lld\n", i, global_size);

    }

    //      printf("rank_id %d predecessor update time %lf\n", rank_id, pred_time);
    //      fflush(stdout);
    //      MPI_Barrier(MPI_COMM_WORLD);
    //      usleep(200);

    if (rank_id == 0)
    {
      printf("total_run_time %lf\n", total_run_time_end - total_run_time_start);
    }

    //      stats->wave_setup_time = w.init_time;
    //      //      stats->total_propagate_time = w.propagate_time;
    //      //      stats->total_broadcast_time = w.broadcast_time;
    //      stats->total_iter = iteration[0];
    //      //	MPI_Barrier(MPI_COMM_WORLD);
    //
    //      total_end = MPI_Wtime();
    //      stats->total_time = total_end - total_start;
    //      stats->print_stats("results.csv");
    return retval;
  }
};

} // namespace GASengine

