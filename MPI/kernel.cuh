/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

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

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

namespace MPI
{
  namespace mpikernel
  {

    template<typename Program>
    __global__ void frontier2flag(int frontier_size, int nodes, typename Program::VertexId* frontier, char* flags)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < frontier_size; i += gridDim.x * blockDim.x)
      {
        typename Program::VertexId v = frontier[i];
        flags[v] = 1;
      }
    }

    template<typename Program>
    __global__ void flag2bitmap(int nodes, int byte_size, char* flags, unsigned char* bitmap)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        bitmap[i] = 0;
        for (int j = 0; j < 8; j++)
        {
          int v = i * 8 + j;
          if (v < nodes)
          {
            char f = flags[v];
            int byte_offset = i;

            if (f == 1)
            {
              unsigned char mask_byte = 1 << j;
              bitmap[byte_offset] |= mask_byte;
//                printf("v=%d, byte_offset=%d, mask_byte=%d, bitmap[byte_offset]=%d\n", v, byte_offset, mask_byte, bitmap[byte_offset]);
            }
            else
            {
              unsigned char mask_byte = ~(1 << j);
              bitmap[byte_offset] &= mask_byte;
            }
          }
        }

      }
    }

    template<typename Program>
    __global__ void bitmap2flag(int byte_size, unsigned char* bitmap, char* flags)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        unsigned char b = bitmap[i];
        unsigned char mask;
        for (int j = 0; j < 8; j++)
        {
          mask = 1;
          mask <<= j;
          if(b & mask)
          {
            flags[8*i + j] = 1;
          }
          else
            flags[8*i + j] = 0;
        }
      }
    }

    //c = a - b
    __global__ void bitsubstract(int byte_size, unsigned char* a, const unsigned char* b, unsigned char* c)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        unsigned char tmpa = a[i];
        unsigned char tmpb = b[i];
        c[i] = (~tmpb) & tmpa;
      }
    }

    //c = union(a, b)
    __global__ void bitunion(int byte_size, unsigned char* a, const unsigned char* b, unsigned char* c)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        unsigned char tmpa = a[i];
        unsigned char tmpb = b[i];
        c[i] = tmpb | tmpa;
      }
    }
    
    template<typename Program>
    __global__ void update_BFS_labels(int rank_id, int iter, typename Program::SizeT nodes, unsigned char* bitmap_in, unsigned char* bitmap_out, typename Program::VertexType vertex_list)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      for(int i = tid; i < nodes; i += blockDim.x * gridDim.x)
      {
        int byte_id = i / 8;
        int bit_off = i % 8;
        unsigned char mask = 1 << bit_off;
        if( (bitmap_out[byte_id] & mask) && vertex_list.d_labels[i] == Program::INIT_VALUE)
//        if(bitmap_out[byte_id] & mask)
        {
          vertex_list.d_labels[i] = Program::SRC_VALUE;
        }
        
        if( (bitmap_in[byte_id] & mask) && vertex_list.d_labels_src[i] == Program::INIT_VALUE)
        {
//          if(rank_id == 1)
//            printf("rank_id=%d, iter=$d, vertex_id=%d\n", rank_id, iter, i);
          vertex_list.d_labels_src[i] = Program::SRC_VALUE;
        }
      }
      
    }

  }
}
#endif /* KERNEL_CUH_ */
