/**
 * \file Types.cuh
 * \brief Defines simple structs for host and device graphs
 */
#pragma once
/**
 * The MapGraph namespace contains data types and definitions for graphs and
 * other data structures
 */
namespace MapGraph {
   /**
    * Defines a basic CSR graph in host memory
    */
   template<typename VertexID, typename EdgeValue>
   struct CsrGraph_h {
      int nodes, edges;
      VertexID* row_offsets;
      VertexID* column_indices;
      EdgeValue* edge_values;

      CsrGraph_h(int nodes, int edges, VertexID* row_offsets, VertexID* column_indices, EdgeValue* edge_values) :
         nodes(nodes), edges(edges), row_offsets(row_offsets), column_indices(column_indices), edge_values(edge_values) {
      }

      /**
       * Print log-histogram
       */
      void PrintHistogram() {
         // Initialize
         int maxDegree = 0;
         VertexID maxDegreeVertexId;
         int log_counts[32];
         for (int i = 0; i < 32; i++) {
            log_counts[i] = 0;
         }

         // Scan
         int max_log_length = -1;
         for (VertexID i = 0; i < nodes; i++) {
            int length = row_offsets[i + 1] - row_offsets[i];
            if (length > maxDegree || i == 0) {
               maxDegree = length;
               maxDegreeVertexId = i;
            }
            int log_length = -1;
            while (length > 0) {
               length >>= 1;
               log_length++;
            }
            if (log_length > max_log_length) {
               max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
         }
         printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n", (long long) nodes, (long long) edges);
         for (int i = -1; i < max_log_length + 1; i++) {
            printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / nodes);
         }
         printf("maxDegree=%lld (vertexId=%lld).\n\n", (long long) maxDegree, (long long) maxDegreeVertexId);
      }

      void DisplayGraph() {
         printf("CSR Graph: %d nodes, %d edges\n", nodes, edges);
         for (VertexID node = 0; node < nodes; node++) {
            printf("%d", node);
            printf(": ");
            for (int edge = row_offsets[node]; edge < row_offsets[node + 1]; edge++) {
               printf("%d ", column_indices[edge]);
            }
            printf("\n");
         }
      }
   };

   /**
    * Defines a basic CSC graph in host memory
    */
   template<typename VertexID, typename EdgeValue>
   struct CscGraph_h {
      int nodes, edges;
      VertexID* column_offsets;
      VertexID* row_indices;
      VertexID* edge_ids;
      EdgeValue* edge_values;

      CscGraph_h(int nodes,
                 int edges,
                 VertexID* column_offsets,
                 VertexID* row_indices,
                 VertexID* edge_ids,
                 EdgeValue* edge_values) :
         nodes(nodes),
         edges(edges),
         column_offsets(column_offsets),
         row_indices(row_indices),
         edge_ids(edge_ids),
         edge_values(edge_values) {
      }

      void DisplayGraph() {
         printf("CSC Graph: %d nodes, %d edges\n", nodes, edges);
         for (VertexID node = 0; node < nodes; node++) {
            printf("%d", node);
            printf(": ");
            for (int edge = column_offsets[node]; edge < column_offsets[node + 1]; edge++) {
               printf("%d ", row_indices[edge]);
            }
            printf("\n");
         }
      }
   };
   /**
    * Defines a basic CSR graph in device memory, number of nodes and edges are not included
    * because it is intended to be used as a kernel argument and most kernels do not require
    * the sizes.
    */
   template<typename VertexID, typename EdgeValue>
   struct CsrGraph_d {
      VertexID* row_offsets;
      VertexID* column_indices;
      EdgeValue* edge_values;

      CsrGraph_d(VertexID* row_offsets, VertexID* column_indices, EdgeValue* edge_values) :
         row_offsets(row_offsets), column_indices(column_indices), edge_values(edge_values) {
      }
   };
   /**
    * Defines a basic CSC graph in device memory
    */
   template<typename VertexID, typename EdgeValue>
   struct CscGraph_d {
      VertexID* column_offsets;
      VertexID* row_indices;
      VertexID* edge_ids;
      EdgeValue* edge_values;

      CscGraph_d(VertexID* column_offsets, VertexID* row_indices, VertexID* edge_ids, EdgeValue* edge_values) :
         column_offsets(column_offsets), row_indices(row_indices), edge_ids(edge_ids), edge_values(edge_values) {
      }
   };
}
