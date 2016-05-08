/**
 * \file CudaError.cuh
 * \brief Defines an exception to be thrown when a CUDA error occurs
 */
#pragma once
#include <cstdio>
#include <string>
namespace MapGraph {
   namespace Exception {
      /**
       * Exception to be thrown when a CUDA error occurs
       */
      struct CudaError {
         cudaError_t error;
         CudaError(cudaError_t error) : error(error){
         }
         std::string what() {
            char message[100];
            sprintf(message, "CUDA Error Code: %d\n", error);
            return std::string(message);
         }
      };
   }
}
