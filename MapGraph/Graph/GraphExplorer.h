/**
 * \file GraphExplorer.h
 * \brief Defines a utility class for exploring a graph for debugging use.
 */
#pragma once
#include <MapGraph/Graph/Types.cuh>
#include <iostream>
#include <string>
#include <map>
namespace MapGraph {
   template<typename VertexId, typename EdgeValue>
   class GraphExplorer {
      CsrGraph_h<VertexId, EdgeValue> csr;
      CscGraph_h<VertexId, EdgeValue> csc;
      std::map<std::string, EdgeValue*> vertexValues;
      std::map<std::string, EdgeValue*> edgeValues;
      public:
      GraphExplorer(CsrGraph_h<VertexId, EdgeValue> csr, CscGraph_h<VertexId, EdgeValue> csc) :
         csr(csr), csc(csc) {
      }
      void addVertexData(std::string name, EdgeValue* array) {
         vertexValues[name] = array;
      }
      void addEdgeData(std::string name, EdgeValue* array) {
         edgeValues[name] = array;
      }
      void printColumns() {
         printf("%15s  ", "Vertex Id");
         for (auto m = vertexValues.begin(); m != vertexValues.end(); m++)
            printf("%15s  ", m->first.c_str());
         for (auto m = edgeValues.begin(); m != edgeValues.end(); m++)
            printf("%15s  ", m->first.c_str());
         printf("\n");
      }
      void printSelfRow(int vertex) {
         printf("%15s  ", "self");
         // Print vertex values
         for (auto m = vertexValues.begin(); m != vertexValues.end(); m++)
            printf("%15d   ", m->second[vertex]);

         printf("\n");
      }
      void printInRow(int index) {
         int vertex = csc.row_indices[index];
         printf("%15d  ", vertex);
         for (auto m = vertexValues.begin(); m != vertexValues.end(); m++)
            printf("%15d  ", m->second[vertex]);
         // Print edge values
         for (auto m = edgeValues.begin(); m != edgeValues.end(); m++) {
            printf("%15d  ", m->second[csc.edge_ids[index]]);
         }
         printf("\n");
      }
      void printOutRow(int index) {
         int vertex = csr.column_indices[index];
         printf("%15d  ", vertex);
         for (auto m = vertexValues.begin(); m != vertexValues.end(); m++)
            printf("%15d  ", m->second[vertex]);
         // Print edge values
         for (auto m = edgeValues.begin(); m != edgeValues.end(); m++) {
            printf("%15d  ", m->second[index]);
         }
         printf("\n");
      }
      void Explore(int node) {
         char option = 'h';
         while (option != 'q') {
            if (option == 'h') {
               printf("Current node is %d, options:\n", node);
               printf("  i - show in edges.\n");
               printf("  o - show out edges.\n");
               printf("  v <VertexId> - switch to vertex.\n");
               printf("  q - quit.\n");
            }
            if (option == 'i') {
               printf("In edges of vertex %d: \n", node);
               printColumns();
               printSelfRow(node);
               for (int i = csc.column_offsets[node]; i < csc.column_offsets[node + 1]; i++) {
                  printInRow(i);
               }
               printf("\n");
            }
            if (option == 'o') {
               printf("Out edges of vertex %d: \n", node);
               printColumns();
               printSelfRow(node);
               for (int i = csr.row_offsets[node]; i < csr.row_offsets[node + 1]; i++) {
                  printOutRow(i);
               }
               printf("\n");
            }
            if (option == 'v') {
               std::cin >> node;
               printf("Switched to vertex %d\n", node);
            }

            printf("Option: ");
            std::cin >> option;
         }
      }
   };
}
