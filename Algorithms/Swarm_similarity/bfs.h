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

#ifndef BFS_H_
#define BFS_H_

#include <GASengine/csr_problem.cuh>
#include <math.h>
struct bfs
{
  typedef float DataType;
  typedef DataType MiscType;
  typedef DataType GatherType;
  typedef int VertexId;
  typedef int SizeT;

  static const DataType INIT_VALUE = 0.0;
  static const DataType SRC_VALUE = 1.0;
  static const DataType TOL = 0.001;
  static const bool allow_duplicates = true;

  struct VertexType
  {
    DataType* d_labels;
    DataType* d_labels_src;
    int nodes;
    int edges;

    VertexType() :
        d_labels(NULL), d_labels_src(NULL), nodes(0), edges(0)
    {
    }
  };

  struct EdgeType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.
    DataType* d_weights;
    EdgeType() :
        nodes(0), edges(0), d_weights(NULL)
    {
    }
  };

  static void Initialize(const int directed, const SizeT nodes, const SizeT edges, int num_srcs, int* srcs, SizeT* d_row_offsets, VertexId* d_column_indices, SizeT* d_column_offsets,
      VertexId* d_row_indices, DataType* d_edge_values, VertexType &vertex_list, EdgeType &edge_list, VertexId* d_frontier_keys[3], MiscType * d_frontier_values[3])
  {
    vertex_list.nodes = nodes;
    vertex_list.edges = edges;
    DataType* h_init_labels = new DataType[nodes];

    for (int i = 0; i < nodes; i++)
    {
      h_init_labels[i] = INIT_VALUE;
    }
    for (int i = 0; i < num_srcs; i++)
    {
      h_init_labels[srcs[i]] = SRC_VALUE;
    }

    b40c::util::B40CPerror(cudaMalloc((void**) &vertex_list.d_labels, nodes * sizeof(DataType)), "cudaMalloc VertexType::d_labels failed", __FILE__, __LINE__);
    b40c::util::B40CPerror(cudaMalloc((void**) &vertex_list.d_labels_src, nodes * sizeof(DataType)), "cudaMalloc VertexType::d_labels_src failed", __FILE__, __LINE__);

    // Initialize d_labels
    if (b40c::util::B40CPerror(cudaMemcpy(vertex_list.d_labels, h_init_labels, nodes * sizeof(DataType), cudaMemcpyHostToDevice), "CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) exit(1);

    if (b40c::util::B40CPerror(cudaMemcpy(vertex_list.d_labels_src, h_init_labels, nodes * sizeof(DataType), cudaMemcpyHostToDevice), "CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__))
      exit(1);

    delete[] h_init_labels;

    if (edge_list.d_weights == NULL)
    {
      b40c::util::B40CPerror(cudaMalloc((void**) &edge_list.d_weights, edges * sizeof(DataType)), "cudaMalloc edge_list.d_weights failed", __FILE__, __LINE__);

      if (b40c::util::B40CPerror(cudaMemcpy(edge_list.d_weights, d_edge_values, edges * sizeof(DataType), cudaMemcpyDeviceToDevice), "CsrProblem cudaMemcpy edge d_weights failed", __FILE__, __LINE__))
        exit(0);
    }

    //    printf("Starting vertex: ");
    //    for (int i = 0; i < num_srcs; i++)
    //      printf("%d ", srcs[i]);
    //    printf("\n");

    if (b40c::util::B40CPerror(cudaMemcpy(d_frontier_keys[0], srcs, num_srcs * sizeof(VertexId), cudaMemcpyHostToDevice), "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__, __LINE__)) exit(0);

    if (b40c::util::B40CPerror(cudaMemcpy(d_frontier_keys[1], srcs, num_srcs * sizeof(VertexId), cudaMemcpyHostToDevice), "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__, __LINE__)) exit(0);

    //    int init_value[1] = { 0 };
    //    if (b40c::util::B40CPerror(
    //        cudaMemcpy(d_frontier_values[0], init_value,
    //            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
    //        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
    //        __LINE__))
    //      exit(0);
    //
    //    if (b40c::util::B40CPerror(
    //        cudaMemcpy(d_frontier_values[1], init_value,
    //            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
    //        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
    //        __LINE__))
    //      exit(0);

  }

  static SrcVertex srcVertex()
  {
    return SINGLE;
  }

  static GatherEdges gatherOverEdges()
  {
    return GATHER_IN_EDGES;
  }

  static ApplyVertices applyOverEdges()
  {
    return APPLY_FRONTIER;
  }

  static ExpandEdges expandOverEdges()
  {
    return EXPAND_OUT_EDGES;
  }

  static PostApplyVertices postApplyOverEdges()
  {
    return NO_POST_APPLY_VERTICES;
  }

  /**
   * the binary operator
   */
  struct gather_sum
  {

    __device__ __host__ GatherType operator()(GatherType left, GatherType right)
    {
      return max(left, right);
    }
  };

  static GatherType gather_sum_host(GatherType left, GatherType right)
  {
    return max(left, right);
  }

  /**
   * For each vertex in the frontier,
   */
//  struct gather_vertex
//  {
//
//    __device__
//            void operator()(const int vertex_id, const GatherType final_value,
//                            VertexType &vertex_list, EdgeType & edge_list)
//    {
//
//    }
//  };
  struct gather_edge
  {

    __device__
    void operator()(const VertexId vertex_id, const VertexId edge_id, const VertexId neighbor_id_in, VertexType &vertex_list, EdgeType &edge_list, GatherType & new_value)
    {
      DataType nb_label = vertex_list.d_labels[neighbor_id_in];
      new_value = nb_label * edge_list.d_weights[edge_id];
//			printf("vertex_id=%d, nb_label[%d]=%d\n", vertex_id,
//					neighbor_id_in, nb_label);
    }
  };

  struct apply
  {

    __device__
    void operator()(const int rank_id, const VertexId vertex_id, const int iteration, GatherType gathervalue, VertexType& vertex_list, EdgeType& edge_list, char& changed)
    {
      const DataType oldvalue = vertex_list.d_labels[vertex_id];
      const DataType newvalue = max(oldvalue, gathervalue);

			if(rank_id == 0 && vertex_id == 178)
				printf("rank_id=%d, vertex_id=%d, iteration=%d, oldvalue=%d, newvalue=%d\n", rank_id, vertex_id, iteration, oldvalue, newvalue);

      if (iteration == 0)
      {
        changed = 1;
      }
      else
      {
        if (fabs(oldvalue - newvalue) < TOL)
          changed = 0;
        else
        {
          changed = 1;
          printf("rank_id=%d, vertex_id=%d, iteration=%d, oldvalue=%d, newvalue=%d, changed=%d\n",
              rank_id, vertex_id, iteration, oldvalue, newvalue, changed);

        }
      }
      vertex_list.d_labels[vertex_id] = newvalue;
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {

    __device__
    void operator()(const VertexId vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType * gather_tmp)
    {
    }
  };

  struct expand_vertex
  {

    __device__
    bool operator()(const VertexId vertex_id, const char changed, VertexType &vertex_list, EdgeType & edge_list)
    {
      return changed;
    }
  };

  struct expand_edge
  {

    __device__
    void operator()(const bool changed, const int iteration, const VertexId vertex_id, const int neighbor_id_in, const int edge_id, VertexType& vertex_list, EdgeType& edge_list, VertexId& frontier,
        MiscType& misc_value)
    {
//      misc_value = vertex_id;
            if(iteration == 1)
              printf("Expand: vertex_id=%d, neighbor_id_in = %d, changed=%d\n", vertex_id, neighbor_id_in, changed);
      if (changed) frontier = neighbor_id_in;
    }
  };

  struct contract
  {

    __device__
    void operator()(const int rank_id, const int iteration, unsigned char *d_bitmap_visited, VertexId &vertex_id, VertexType &vertex_list, EdgeType &edge_list, GatherType* gather_tmp,
        MiscType& misc_value)
    {
    }
  };

  static void extractResult(VertexType& vertex_list, DataType * h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_labels, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }
  static void extractPred(int * pred_d, int nodes, int * h_output)
  {
    cudaMemcpy(h_output, pred_d, sizeof(int) * nodes, cudaMemcpyDeviceToHost);

  }

};

#endif /* BFS_H_ */
