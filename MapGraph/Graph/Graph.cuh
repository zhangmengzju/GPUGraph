/**
 * \file Graph.cuh
 * \brief Defines the Graph class
 */
/**
 Copyright 2013-2015 SYSTAP, LLC.  http://www.systap.com

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
#include <MapGraph/Graph/Types.cuh>
#include <MapGraph/Graph/MTX.h>
#include <MapGraph/Graph/CsrToCoo.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <sstream>
#include <MapGraph/Util/Timer.h>
namespace MapGraph {
   /**
    * The Graph class represents a graph and its associated edge and vertex values.
    * The graph data is stored internally in COO array form, with auxiliary arrays
    * created to return CSR and CSC representations.
    */
   template<typename VertexID, typename EdgeValue>
   class Graph {
   private:
      int nodes;           //!< The number of nodes in the graph
      int edges;           //!< The number of edges in the graph
      int device;          //!< The GPU to store graph on
      bool edge_values;    //!< Indicates whether the graph has edge values

      // COO structure
      thrust::device_vector<VertexID> rows_d;         //!< Stores row of COO data
      thrust::device_vector<VertexID> columns_d;      //!< Stores column of COO data
      thrust::device_vector<EdgeValue> edge_values_d;  //!< Stores edgeValues
      thrust::host_vector<VertexID> rows_h;           //!< Stores row of COO data
      thrust::host_vector<VertexID> columns_h;        //!< Stores column of COO data
      thrust::host_vector<EdgeValue> edge_values_h;    //!< Stores edgeValues

      // CSR structure
      thrust::device_vector<VertexID> row_offsets_d;   //!< Stores row offsets of CSR data
      thrust::host_vector<VertexID> row_offsets_h;     //!< Stores row offsets of CSR data

      // CSC structure
      thrust::device_vector<VertexID> column_offsets_d;   //!< Stores column offsets of CSC data
      thrust::host_vector<VertexID> column_offsets_h;     //!< Stores column offsets of CSC data
      thrust::device_vector<VertexID> row_indices_d;      //!< Stores row indices of CSC data
      thrust::host_vector<VertexID> row_indices_h;        //!< Stores row indices of CSC data
      thrust::device_vector<VertexID> edge_ids_d;         //!< Stores edge ids of CSC data
      thrust::host_vector<VertexID> edge_ids_h;           //!< Stores edge ids of CSC data

      // Methods
      /**
       * Clears all graph data.
       */
      void clearData() {
         nodes = 0;
         edges = 0;
         device = -1;
         rows_d.clear();
         columns_d.clear();
         edge_values_d.clear();
         rows_h.clear();
         columns_h.clear();
         edge_values_h.clear();
         row_offsets_d.clear();
         row_offsets_h.clear();
         column_offsets_d.clear();
         column_offsets_h.clear();
         row_indices_d.clear();
         row_indices_h.clear();
         edge_ids_d.clear();
         edge_ids_h.clear();
      }
      /**
       * Copies all graph data from GPU buffers to host buffers
       */
      void copyToHost() {
         rows_h = rows_d;
         columns_h = columns_d;
         edge_values_h = edge_values_d;
         row_offsets_h = row_offsets_d;
         column_offsets_h = column_offsets_d;
         row_indices_h = row_indices_d;
         edge_ids_h = edge_ids_d;
      }
      /**
       * Processes COO data from host buffers to construct CSR and CSC structures
       */
      void processCOO_h() {
         // Copy data from host to GPU
         Timer timmy;
         timmy.start();
         rows_d = rows_h;
         columns_d = columns_h;
         edge_values_d = edge_values_h;
         timmy.stop();
         printf("Copied host side COO arrays to GPU in %01.4fms\n", timmy.getElapsedTimeInMilliSec());

         timmy.start();
         // Sort edge_values with key COO by row,column
         if (edge_values)
            thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(rows_d.begin(), columns_d.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(rows_d.end(), columns_d.end())),
                                edge_values_d.begin());
         else
            thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(rows_d.begin(), columns_d.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(rows_d.end(), columns_d.end())));

         // Get row_offsets for CSR
         row_offsets_d.resize(nodes + 1);
         thrust::lower_bound(rows_d.begin(),
                             rows_d.end(),
                             thrust::counting_iterator<VertexID>(0),
                             thrust::counting_iterator<VertexID>(nodes + 1),
                             row_offsets_d.begin());

         // Copy rows to row_indices and make temp copy of columns
         row_indices_d = rows_d;
         thrust::device_vector<VertexID> colCopy = columns_d;

         // Make edge_ids array sequentially numbered
         if (edge_values) {
            edge_ids_d.resize(edges);
            thrust::copy(thrust::counting_iterator<VertexID>(0),
                         thrust::counting_iterator<VertexID>(edges),
                         edge_ids_d.begin());
         }

         // Sort edge_ids with key COO by column, row
         if (edge_values)
            thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(colCopy.begin(), row_indices_d.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(colCopy.end(), row_indices_d.end())),
                                edge_ids_d.begin());
         else
            thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(colCopy.begin(), row_indices_d.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(colCopy.end(), row_indices_d.end())));

         // Get row_offsets for CSR
         column_offsets_d.resize(nodes + 1);
         thrust::lower_bound(colCopy.begin(),
                             colCopy.end(),
                             thrust::counting_iterator<VertexID>(0),
                             thrust::counting_iterator<VertexID>(nodes + 1),
                             column_offsets_d.begin());
         timmy.stop();
         printf("Processed CSR and COO on GPU in %1.2fms\n", timmy.getElapsedTimeInMilliSec());
         timmy.start();
         copyToHost();
         timmy.stop();
         printf("Copied graph data to host in %1.2fms\n", timmy.getElapsedTimeInMilliSec());
      }

   public:
      int nodeCount() {
         return nodes;
      }
      int edgeCount() {
         return edges;
      }
      int getDevice() {
         return device;
      }
      /**
       * Returns host side CSR view of graph
       */
      CsrGraph_h<VertexID, EdgeValue> getCsrGraph_h() {
         return CsrGraph_h<VertexID, EdgeValue>(nodes,
                                                edges,
                                                &(row_offsets_h[0]),
                                                &(columns_h[0]),
                                                edge_values ? &(edge_values_h[0]) : NULL);
      }
      /**
       * Returns host side CSC view of graph
       */
      CscGraph_h<VertexID, EdgeValue> getCscGraph_h() {
         return CscGraph_h<VertexID, EdgeValue>(nodes,
                                                edges,
                                                &(column_offsets_h[0]),
                                                &(row_indices_h[0]),
                                                edge_values ? &(edge_ids_h[0]) : NULL,
                                                edge_values ? &(edge_values_h[0]) : NULL);
      }
      /**
       * Returns device side CSR view of graph
       */
      CsrGraph_d<VertexID, EdgeValue> getCsrGraph_d() {
         return CsrGraph_d<VertexID, EdgeValue>(thrust::raw_pointer_cast(row_offsets_d.data()),
                                                thrust::raw_pointer_cast(columns_d.data()),
                                                edge_values ? thrust::raw_pointer_cast(edge_values_d.data()) : NULL);
      }
      /**
       * Returns device side CSC view of graph
       */
      CscGraph_d<VertexID, EdgeValue> getCscGraph_d() {
         return CscGraph_d<VertexID, EdgeValue>(thrust::raw_pointer_cast(column_offsets_d.data()),
                                                thrust::raw_pointer_cast(row_indices_d.data()),
                                                edge_values ? thrust::raw_pointer_cast(edge_ids_d.data()) : NULL,
                                                edge_values ? thrust::raw_pointer_cast(edge_values_d.data()) : NULL);
      }
      /**
       * Fills graph from MTX file.
       * Fills the graph data by reading from the specified MTX file. The edgevalues parameter
       * specifies whether edge values should be read from the file.
       */
      void fromMTX(char* filename, int deviceToUse, bool edgevalues) {
         Timer timmy;
         device = deviceToUse;
         edge_values = edgevalues;
         timmy.start();
         IO::MTXtoArrays(filename, edgevalues, nodes, edges, rows_h, columns_h, edge_values_h);
         timmy.stop();
         printf("Read MTX file to COO arrays in %01.4fs\n", timmy.getElapsedTimeInSec());
         timmy.start();
         processCOO_h();
         timmy.stop();
         printf("Done processing CSR and CSC from COO in %01.4fs\n", timmy.getElapsedTimeInSec());
      }
      /**
       * Fills graph from CSR arrays
       */
      void fromCSR(int _nodes,
                   int _edges,
                   VertexID* row_offsets,
                   VertexID* column_indices,
                   EdgeValue* _edge_values,
                   int deviceToUse,
                   bool edgevalues) {
         nodes = _nodes;
         edges = _edges;
         device = deviceToUse;
         edge_values = edgevalues;
         IO::CsrToCoo(nodes,
                      edges,
                      row_offsets,
                      column_indices,
                      _edge_values,
                      edge_values,
                      rows_h,
                      columns_h,
                      edge_values_h);
         processCOO_h();
      }
   };
}
