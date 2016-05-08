#pragma once
#include <string>
namespace MapGraph {
   namespace Util {
      /**
       * Sets current cuda device and prints out some information about the device
       */
      bool cudaInit(int device) {
         int deviceCount = 0;
         cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

         if (error_id != cudaSuccess)
             {
            printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id,
                   cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit (EXIT_FAILURE);
         }

         // This function call returns 0 if there are no CUDA capable devices.
         if (deviceCount == 0)
             {
            printf("There are no available device(s) that support CUDA\n");
            return false;
         }
         else
         {
            printf("Detected %d CUDA Capable device(s)\n", deviceCount);
         }

         int dev, driverVersion = 0, runtimeVersion = 0;

         for (dev = 0; dev < deviceCount; ++dev)
            {
            if (dev == device)
                {
               cudaSetDevice(dev);
               cudaDeviceProp deviceProp;
               cudaGetDeviceProperties(&deviceProp, dev);

               printf("Running on this device:");
               printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

               // Console log
               cudaDriverGetVersion(&driverVersion);
               cudaRuntimeGetVersion(&runtimeVersion);
               printf(
                      "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
                      driverVersion / 1000, (driverVersion % 100) / 10,
                      runtimeVersion / 1000,
                      (runtimeVersion % 100) / 10);
               printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                      deviceProp.major,
                      deviceProp.minor);

               printf(
                      "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                      (float) deviceProp.totalGlobalMem / 1048576.0f,
                      (unsigned long long) deviceProp.totalGlobalMem);

               break;
            }
         }

         return true;
      }
      /**
       * Checks the current memory usage on the given device and prints to console
       * \param location The location string is printed as part of the output message to indicate where the
       * memory was checked.
       */
      void printCudaMemStats(std::string location, int deviceId) {
         cudaSetDevice(deviceId);
         size_t free, total;
         cudaError_t error = cudaMemGetInfo(&free, &total);
         if (error != cudaSuccess)
            std::cerr << "CUDA error getting memstats (" << location << "):" << cudaGetErrorString(error) << std::endl;
         else {
            size_t used = total - free;
            std::cerr << "CUDA memory usage (" << location << "): Used: " << (used / 1048576) << "MiB, Free: "
                      << (free / 1048576)
                      << "MiB, Total: " << (total / 1048576) << std::endl;
         }
      }
      /**
       * Helper method which synchronizes the device and checks for errors
       */
      void checkCudaError(const char* message) {
         cudaError_t error = cudaDeviceSynchronize();
         if (error) {
            printf("At: %s, CUDA error %d: %s\n", message, error, cudaGetErrorString(error));
         }
      }
   }
}
