#pragma once

namespace noxitu { namespace yolo
{
    void amp_convolution(float const * const input,
                          float const * const weights,
                          float const * const biases,
                          float * const output,
                          const int data_size, 
                          const int kernel_size,
                          const int depth,
                          const int kernels);
}}