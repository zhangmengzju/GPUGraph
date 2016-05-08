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
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>
#include <GASengine/problem_type.cuh>
#include <GASengine/Enums.h>
#include <config.h>
#include <util.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <vector>

// Graph definition
#include <MapGraph/Graph/Graph.cuh>
#include <MapGraph/Exceptions/CudaError.cuh>

using namespace b40c;
namespace GASengine {
   template<typename _Program,
      typename _VertexId,
      typename _SizeT,
      typename _EValue,
      bool MARK_PREDECESSORS,
      bool WITH_VALUE>
   struct GraphProblem {
      //---------------------------------------------------------------------
      // Typedefs and constants
      //---------------------------------------------------------------------
      typedef ProblemType<_Program,    // vertex type
         _VertexId,           // VertexId
         _SizeT,              // SizeT
         _EValue,          // Edge Value
         unsigned char,       // VisitedMask
         unsigned char,          // ValidFlag
         MARK_PREDECESSORS,      // MARK_PREDECESSORS
         WITH_VALUE>             // WITH_VALUE
      ProblemType;

      typedef typename ProblemType::Program::VertexType VertexType;
      typedef typename ProblemType::Program::EdgeType EdgeType;
      typedef typename ProblemType::Program::MiscType MiscType;
      typedef typename ProblemType::VertexId VertexId;
      typedef typename ProblemType::SizeT SizeT;
      typedef typename ProblemType::VisitedMask VisitedMask;
      typedef typename ProblemType::ValidFlag ValidFlag;
      typedef typename ProblemType::Program::DataType EValue;

      //---------------------------------------------------------------------
      // Members
      //---------------------------------------------------------------------
      Config* cfg;
      MapGraph::Graph<VertexId, EValue>* graph;
      VertexType vertex_list;
      EdgeType edge_list;
      int device;

      // From GraphSlice (probably should move into enactor)
      int directed;
      int* d_edgeCountScan;
      int* d_active_flags;
      char* d_changed;
      VisitedMask *d_visited_mask;
      SizeT *d_visit_flags;
      util::DoubleBuffer<VertexId, MiscType> frontier_queues;
      SizeT frontier_elements[2];
      SizeT predecessor_elements[2];
      ValidFlag *d_filter_mask;
      VertexId nodes;
      SizeT edges;

      //---------------------------------------------------------------------
      // Methods
      //---------------------------------------------------------------------

      /**
       * Constructor
       */
      GraphProblem(MapGraph::Graph<VertexId, EValue>* graph, Config* cfg) :
         graph(graph),
         cfg(cfg),
         d_filter_mask(NULL),
         d_visited_mask(NULL),
         d_visit_flags(NULL),
         d_edgeCountScan(NULL) {

         printf("GraphProblem()\n");

         for (int i = 0; i < 2; i++) {
            frontier_elements[i] = 0;
            predecessor_elements[i] = 0;
         }

         device = graph->getDevice();
         directed = cfg->getParameter<int>("directed");
         cudaError_t retval = cudaSuccess;
         nodes = graph->nodeCount();
         edges = graph->edgeCount();

         // Set the cuda device
         if (retval = util::B40CPerror(cudaSetDevice(device),
                                       "CsrProblem cudaSetDevice failed",
                                       __FILE__, __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         //Device mem allocations
         printf("GPU %d column_indices: %lld elements (%lld bytes)\n",
                device,
                (unsigned long long) (edges),
                (unsigned long long) (edges * sizeof(VertexId) * sizeof(SizeT)));

         printf("GPU %d row_offsets: %lld elements (%lld bytes)\n",
                device,
                (unsigned long long) (nodes + 1),
                (unsigned long long) (nodes + 1) * sizeof(SizeT));

         // Allocate active flags array
         if (retval = util::B40CPerror(cudaMalloc((void**) &d_active_flags, (nodes) * sizeof(int)),
                                       "CsrProblem cudaMalloc d_active_flags failed",
                                       __FILE__,
                                       __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         // Allocate changed array
         if (retval = util::B40CPerror(cudaMalloc((void**) &d_changed, (nodes) * sizeof(char)),
                                       "CsrProblem cudaMalloc d_changed failed",
                                       __FILE__,
                                       __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         //Initializations
         const time_t starttransfer = time(NULL);

         cudaDeviceSynchronize();
         const time_t endtransfer = time(NULL);
         printf("CPU to GPU memory transfer time: %f ms\n", difftime(endtransfer, starttransfer) * 1000.0);

         int memset_block_size = 256;
         int memset_grid_size_max = 32 * 1024;    // 32K CTAs
         int memset_grid_size;

         memset_grid_size = B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
         util::MemsetKernel<int>
         <<<memset_grid_size, memset_block_size, 0, NULL>>>(d_active_flags, 0, nodes);

         if (retval = util::B40CPerror(cudaThreadSynchronize(),
                                       "MemsetKernel d_active_flags failed",
                                       __FILE__,
                                       __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         util::MemsetKernel<char><<<memset_grid_size,
         memset_block_size, 0, NULL>>>(d_changed, 0, nodes);

         if (retval = util::B40CPerror(cudaThreadSynchronize(),
                                       "MemsetKernel d_changed failed",
                                       __FILE__, __LINE__))
            throw MapGraph::Exception::CudaError(retval);
      }

      /**
       * Destructor
       */
      virtual ~GraphProblem() {
         printf("~GraphProblem()\n");
         // Cleanup allocated memory arrays
         edge_list.Deallocate();
         vertex_list.Deallocate();
         // Set device
         util::B40CPerror(cudaSetDevice(device),
                          "GpuSlice cudaSetDevice failed",
                          __FILE__, __LINE__);

         if (d_edgeCountScan)
            util::B40CPerror(cudaFree(d_edgeCountScan),
                             "GpuSlice cudaFree d_edgeCountScan",
                             __FILE__,
                             __LINE__);

         if (d_active_flags)
            util::B40CPerror(cudaFree(d_active_flags),
                             "GpuSlice cudaFree d_active_flags",
                             __FILE__, __LINE__);

         if (d_changed)
            util::B40CPerror(cudaFree(d_changed),
                             "GpuSlice cudaFree d_changed",
                             __FILE__, __LINE__);

         if (d_visited_mask)
            util::B40CPerror(cudaFree(d_visited_mask),
                             "GpuSlice cudaFree d_visited_mask failed",
                             __FILE__,
                             __LINE__);
         if (d_filter_mask)
            util::B40CPerror(cudaFree(d_filter_mask),
                             "GpuSlice cudaFree d_filter_mask failed",
                             __FILE__,
                             __LINE__);
         if (d_visit_flags)
            util::B40CPerror(cudaFree(d_visit_flags),
                             "GpuSlice cudaFree d_visit_flags failed",
                             __FILE__,
                             __LINE__);
         for (int i = 0; i < 2; i++)
            {
            if (frontier_queues.d_keys[i])
               util::B40CPerror(cudaFree(frontier_queues.d_keys[i]),
                                "GpuSlice cudaFree frontier_queues.d_keys failed",
                                __FILE__, __LINE__);
            if (frontier_queues.d_values[i])
               util::B40CPerror(cudaFree(frontier_queues.d_values[i]),
                                "GpuSlice cudaFree frontier_queues.d_values failed",
                                __FILE__, __LINE__);
         }

      }

      /**
       * Extract into a single host vector the BFS results disseminated across
       * all GPUs
       */
      cudaError_t ExtractResults(EValue *h_values)
                                 {
         cudaError_t retval = cudaSuccess;

         // Set device
         if (retval = util::B40CPerror(cudaSetDevice(graph->getDevice()),
                                       "CsrProblem cudaSetDevice failed",
                                       __FILE__, __LINE__))
            return retval;

         _Program::extractResult(vertex_list, h_values);

         return retval;
      }

      /**
       * Performs any initialization work needed for this problem type.  Must be called
       * prior to each search
       */
      cudaError_t Reset() {
         FrontierType frontier_type = _Program::frontierType();
         double queue_sizing = cfg->getParameter<double>("max_queue_sizing");
         cudaError_t retval = cudaSuccess;
         // Set device
         if (retval = util::B40CPerror(cudaSetDevice(device),
                                       "CsrProblem cudaSetDevice failed",
                                       __FILE__, __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         // Allocate visited masks for the entire graph if necessary
         int visited_mask_bytes = ((nodes * sizeof(VisitedMask)) + 8 - 1) / 8;    // round up to the nearest VisitedMask
         int visited_mask_elements = visited_mask_bytes * sizeof(VisitedMask);
         if (!d_visited_mask) {
            if (retval = util::B40CPerror(cudaMalloc((void**) &d_visited_mask, visited_mask_bytes),
                                          "CsrProblem cudaMalloc d_visited_mask failed",
                                          __FILE__,
                                          __LINE__))
               throw MapGraph::Exception::CudaError(retval);
         }

         // Determine frontier queue sizes
         SizeT new_frontier_elements[2] = { 0, 0 };
         SizeT new_predecessor_elements[2] = { 0, 0 };
         switch (frontier_type)
         {
            case VERTEX_FRONTIERS:
               // O(n) ping-pong global vertex frontiers
               new_frontier_elements[0] = double(nodes) * queue_sizing;
               new_frontier_elements[1] = new_frontier_elements[0];
               break;

            case EDGE_FRONTIERS:
               // O(m) ping-pong global edge frontiers
               new_frontier_elements[0] = double(edges) * queue_sizing;
               new_frontier_elements[1] = new_frontier_elements[0];
               new_predecessor_elements[0] = new_frontier_elements[0];
               new_predecessor_elements[1] = new_frontier_elements[1];
               break;

            case MIXED_FRONTIERS:
               // O(n) global vertex frontier, O(m) global edge frontier
               new_frontier_elements[0] = double(nodes) * queue_sizing;
               new_frontier_elements[1] = double(edges) * queue_sizing;
               new_predecessor_elements[1] = new_frontier_elements[1];
               break;

            case MULTI_GPU_FRONTIERS:
               // O(n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier
               new_frontier_elements[0] = double(nodes) * MULTI_GPU_VERTEX_FRONTIER_SCALE
                                          * queue_sizing;
               new_frontier_elements[1] = double(edges) * queue_sizing;
               new_predecessor_elements[1] = new_frontier_elements[1];
               break;
         }

         // Iterate through global frontier queue setups
         for (int i = 0; i < 2; i++) {
            // Allocate frontier queue if not big enough
            if (frontier_elements[i] < new_frontier_elements[i]) {
               // Free if previously allocated
               if (frontier_queues.d_keys[i]) {
                  if (retval = util::B40CPerror(cudaFree(frontier_queues.d_keys[i]),
                                                "GpuSlice cudaFree frontier_queues.d_keys failed",
                                                __FILE__,
                                                __LINE__))
                     throw MapGraph::Exception::CudaError(retval);
               }
               frontier_elements[i] = new_frontier_elements[i];

               if (retval = util::B40CPerror(cudaMalloc((void**) &frontier_queues.d_keys[i],
                                                        frontier_elements[i] * sizeof(VertexId)),
                                             "CsrProblem cudaMalloc frontier_queues.d_keys failed",
                                             __FILE__,
                                             __LINE__))
                  throw MapGraph::Exception::CudaError(retval);
            }

            // Allocate predecessor queue if not big enough
            if (predecessor_elements[i] < new_predecessor_elements[i]) {
               // Free if previously allocated
               if (frontier_queues.d_values[i]) {
                  if (retval = util::B40CPerror(cudaFree(frontier_queues.d_values[i]),
                                                "GpuSlice cudaFree frontier_queues.d_values failed",
                                                __FILE__,
                                                __LINE__))
                     throw MapGraph::Exception::CudaError(retval);
               }

               predecessor_elements[i] = new_predecessor_elements[i];
               if (retval = util::B40CPerror(cudaMalloc((void**) &frontier_queues.d_values[i],
                                                        predecessor_elements[i] * sizeof(MiscType)),
                                             "CsrProblem cudaMalloc frontier_queues.d_values failed",
                                             __FILE__,
                                             __LINE__))
                  throw MapGraph::Exception::CudaError(retval);
            }
         }

         // Allocate edge count scan array
         if (retval = util::B40CPerror(cudaMalloc((void**) &d_edgeCountScan, (frontier_elements[0] + 1) * sizeof(SizeT)),
                                       "CsrProblem cudaMalloc d_edgeCountScan failed",
                                       __FILE__,
                                       __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         int memset_block_size = 256;
         int memset_grid_size_max = 32 * 1024;    // 32K CTAs
         int memset_grid_size;

         // Initialize d_visited_mask elements to 0
         memset_grid_size =
            B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);

         util::MemsetKernel<VisitedMask>
         <<<memset_grid_size, memset_block_size, 0, NULL>>>(d_visited_mask, 0, visited_mask_elements);

         if (retval = util::B40CPerror(cudaThreadSynchronize(),
                                       "MemsetKernel failed",
                                       __FILE__, __LINE__))
            throw MapGraph::Exception::CudaError(retval);

         return retval;
      }
   };
}    // namespace GASengine

