/**
 * \file GraphUtil.h
 * \brief Defines utility methods for examining and checking graphs
 */
#pragma once
#include <MapGraph/Graph/Types.cuh>
namespace MapGraph {
   namespace GraphUtil {
      template <typename VertexId, typename EdgeValue>
      bool isSymmetric(CsrGraph_h<VertexId, EdgeValue> graph, bool verbose){
         int unmatched = 0;
         // For every edge check to see if the reverse edge also exists
         for (int node = 0; node < graph.nodes; node++){
            for (int nIt = graph.row_offsets[node]; nIt < graph.row_offsets[node+1]; nIt++){
               int neighbor = graph.column_indices[nIt];
               bool matchFound = false;
               for (int nnIt = graph.row_offsets[neighbor]; nnIt < graph.row_offsets[neighbor+1]; nnIt++)
                  if (graph.column_indices[nnIt] == node)
                     matchFound = true;
               if (!matchFound){
                  unmatched++;
                  if (verbose && unmatched < 10)
                     printf("Edge (%d, %d) does not have a match.\n", node, neighbor);
               }
            }
         }
         if (verbose)
            printf("Graph checked and %d out of %d edges were unmatched.\n", unmatched, graph.nodes);
         return unmatched == 0;
      }
   }
}
