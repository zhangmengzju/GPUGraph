/**
 * \file testing.h
 * \brief utility methods used for correctness checking and CI
 */
#pragma once
#include <iostream>
namespace MapGraph {
   namespace Util {
      template<typename DataType>
      bool correctTest(const char* label, int count, DataType* gt_values, DataType* test_values) {
         bool pass = true;
         printf("Correctness testing (%s):\n", label);
         int errorCount = 0;
         for (int i = 0; i < count; i++) {
            if (gt_values[i] != test_values[i]) {
               if (errorCount < 10) {
                  std::cout << "Incorrect value for element " << i << ": GT value " << gt_values[i]
                  << ", Test value " << test_values[i] << "\n";
               }
               errorCount++;
               pass = false;
            }
         }
         if (pass)
            printf("Passed correctness testing (%s).\n", label);
         else
            printf("Failed correctness testing (%s) with %d errors.\n", label, errorCount);
         return pass;
      }

      template<typename DataType>
      void showResults(const char* label, const char* labelA, const char* labelB, int count, DataType* valuesA, DataType* valuesB) {
         printf("%s results:\n", label);
         printf("%15s  %15s  %15s\n", "Index", labelA, labelB);
         for (int i = 0; i < count; i++) {
            printf("%15d  %15d  %15d\n", i, valuesA[i], valuesB[i]);
         }
         printf("\n");
      }

      template<typename DataType>
      void showDeviceValues(const char* label, DataType* array, int start, int count){
         cudaDeviceSynchronize();
         DataType* vals = (DataType*)malloc(sizeof(DataType) * count);
         cudaError_t retval = cudaMemcpy(vals, array + start, sizeof(DataType) * count, cudaMemcpyDeviceToHost);
         if (retval != cudaSuccess){
            printf("Error copying values to host (%s):%s\n", label, cudaGetErrorString(retval));
         }

         printf("%s[%d:%d]: ", label, start, start + count);
         for (int i = 0; i < count; i++)
            std::cout << vals[i] << " ";
         std::cout << "\n";
         free(vals);
      }
   }
}
