/**
 * \file b40c_CsrGraph_convert.h
 * \brief Defines IO util method to convert CSR arrays into COO
 */
#pragma once
#include <thrust/host_vector.h>
namespace MapGraph {
   namespace IO {
      /**
       * This method reads the given b40c CsrGraph into the supplied COO arrays
       * the edgevalues parameter specifies whether edge values should be read.
       */
      template<typename VertexId, typename Value>
      void CsrToCoo(int nodes,
                    int edges,
                    VertexId* row_offsets,
                    VertexId* column_indices,
                    Value* edge_values,
                    bool edgevalues,
                    thrust::host_vector<VertexId>& rows_h,
                    thrust::host_vector<VertexId>& columns_h,
                    thrust::host_vector<Value>& edge_values_h) {
         printf("Starting to read CSR to COO\n");
         rows_h.resize(edges);
         columns_h.resize(edges);
         if (edgevalues)
            edge_values_h.resize(edges);
         int edgesRead = 0;
         for (int i = 0; i < nodes; i++) {
            for (int nIt = row_offsets[i]; nIt < row_offsets[i + 1]; nIt++) {
               rows_h[edgesRead] = i;
               columns_h[edgesRead] = column_indices[nIt];
               if (edgevalues)
                  edge_values_h[edgesRead] = edge_values[nIt];
               edgesRead++;
            }
         }
         printf("Done reading CSR to COO\n");
      }
   }
}
