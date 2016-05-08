/**
 * \file MTX.h
 * \brief Defines IO util methods for reading matrix market format files
 */
#pragma once
#include <thrust/host_vector.h>
#include <sstream>
namespace MapGraph {
   namespace IO {
      /**
       * This method reads the given MTX file into the supplied COO arrays
       * the edgevalues parameter specifies whether edge values should be read.
       */
      template<typename VertexID, typename EdgeValue>
      void MTXtoArrays(char* filename,
                       bool edgevalues,
                       int& nodes,
                       int& edges,
                       thrust::host_vector<VertexID>& rows_h,
                       thrust::host_vector<VertexID>& columns_h,
                       thrust::host_vector<EdgeValue>& edge_values_h) {
         // Open file to read in
         FILE *f_in = fopen(filename, "r");
         if (f_in) {
            printf("Parsing MARKET COO format from %s:\n", filename);
            int edges_read = -1;
            char line[1024];
            bool readSizes = false;
            std::istringstream ss;

            while (true) {
               // Throw out comment lines
               while (fscanf(f_in, "%%%[^\n]\n", line) > 0) {
               }
               // Read a data line
               if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
                  break;
               }

               if (edges_read == -1) {
                  // Problem description
                  int nodes_x, nodes_y, edgeCount;
                  if (sscanf(line, "%d %d %d", &nodes_x, &nodes_y, &edgeCount) != 3) {
                     fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
                     return;
                  }
                  if (nodes_x != nodes_y) {
                     fprintf(stderr, "Error parsing MARKET graph: not square (%d, %d)\n", nodes_x, nodes_y);
                     return;
                  }

                  nodes = nodes_x;
                  edges = edgeCount;

                  printf(" (%d nodes, %d directed edges)... ", nodes_x, edgeCount);
                  fflush(stdout);

                  // Allocate coo graph
                  rows_h.resize(edgeCount, 0);
                  columns_h.resize(edgeCount, 0);
                  if (edgevalues)
                     edge_values_h.resize(edgeCount, 0);
                  edges_read++;
                  readSizes = true;
               }
               else {

                  // Edge description (v -> w)
                  if (!readSizes) {
                     fprintf(stderr, "Error parsing MARKET graph: invalid format\n");
                     return;
                  }
                  if (edges_read >= edges) {
                     fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
                     edges = 0;
                     nodes = 0;
                     edge_values_h.clear();
                     rows_h.clear();
                     columns_h.clear();
                     return;
                  }

                  ss.clear();
                  ss.str(line);
//                  printf("Reading edge line: %s, ss fail()=%d\n", ss.str().c_str(), ss.fail());
                  bool readFailed = false;
                  if (!readFailed) {
                     ss >> rows_h[edges_read];
                     rows_h[edges_read]--;
                     readFailed = ss.fail();
//                     printf("Read row of: %d, ss fail()=%d\n", rows_h[edges_read], ss.fail());
                  }
                  if (!readFailed) {
                     ss >> columns_h[edges_read];
                     columns_h[edges_read]--;
                     readFailed = ss.fail();
//                     printf("Read column of: %d, ss fail()=%d\n", rows_h[edges_read], ss.fail());
                  }
                  if (!readFailed && edgevalues) {
                     // ToDo This reads in the edge value as a double, it should read it in as the
                     // edge value type. This matches the b40c behaviour but I don't think it's the
                     // right way to do it -James
                     double edgeVal;
                     ss >> edgeVal;
                     edge_values_h[edges_read] = edgeVal;
                     readFailed = ss.fail();

                     // If reading in the edge value from file failed set to default of 1
                     if (readFailed) {
                        edge_values_h[edges_read] = 1;
                        readFailed = false;
                     }
                  }
                  if (readFailed) {
                     fprintf(stderr, "Error parsing MARKET graph: badly formed edge\n");
                     edges = 0;
                     nodes = 0;
                     edge_values_h.clear();
                     rows_h.clear();
                     columns_h.clear();
                     return;
                  }
                  edges_read++;
               }
            }

            if (!readSizes) {
               fprintf(stderr, "No graph found\n");
               return;
            }

            if (edges_read != edges) {
               fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
               edges = 0;
               nodes = 0;
               edge_values_h.clear();
               rows_h.clear();
               columns_h.clear();
               return;
            }
            fclose(f_in);
         }
         else {
            perror("Unable to open file");
         }
      }
   }
}
