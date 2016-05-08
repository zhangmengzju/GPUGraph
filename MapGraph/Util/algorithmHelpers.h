/**
 * \file algorithmHelpers.h
 * \brief Defines helper methods used in each algorithm executable
 */
#pragma once
#include <MapGraph/Graph/Types.cuh>
#include <config.h>
namespace MapGraph {
   namespace Util {
      /**
       * Method to get to process config to get sources to use
       */
      template<typename VertexId, typename EdgeValue>
      void getSources(VertexId** sources,
                      Config& cfg,
                      char* source_file_name,
                      int& num_srcs,
                      MapGraph::CsrGraph_h<VertexId, EdgeValue>& graph) {
         int max_src_num = 1000;
         if (strcmp(source_file_name, "")) {
            if (strcmp(source_file_name, "RANDOM") == 0) {
               printf("Using random starting vertices!\n");
               num_srcs = cfg.getParameter<int>("num_src");
               *sources = new VertexId[num_srcs];
               printf("Using %d random starting vertices!\n", num_srcs);
               srand (time(NULL));
               int count = 0;

               while (count < num_srcs) {
                  int tmp_src = rand() % graph.nodes;
                  if (graph.row_offsets[tmp_src + 1] - graph.row_offsets[tmp_src] > 0) {
                     (*sources)[count++] = tmp_src;
                  }
               }
            }
            else
            {
               printf("Using source file: %s!\n", source_file_name);
               FILE* src_file;
               if ((src_file = fopen(source_file_name, "r")) == NULL) {
                  printf("Source file open error!\n");
                  exit(0);
               }

               *sources = new VertexId[max_src_num];
               for (num_srcs = 0; num_srcs < max_src_num; num_srcs++) {
                  if (fscanf(src_file, "%d\n", &(*sources)[num_srcs]) != EOF) {
                     if (cfg.getParameter<int>("origin") == 1)
                        (*sources)[num_srcs]--;  //0-based index
                  }
                  else
                     break;
               }
               printf("number of srcs used: %d\n", num_srcs);
            }

         }
         else {
            printf("From config (getSources): src=%d\n", cfg.getParameter<int>("src"));
            int src_node = cfg.getParameter<int>("src");
            int origin = cfg.getParameter<int>("origin");
            num_srcs = 1;
            *sources = new VertexId[1];
            (*sources)[0] = src_node;
            if (cfg.getParameter<int>("origin") == 1)
               (*sources)[0]--;
            printf("Single source vertex: %d\n", (*sources)[0]);
         }
      }
   }
}
