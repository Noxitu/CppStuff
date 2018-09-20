#include <noxitu/yolo/cpu/fast_convolution.h>

namespace noxitu { namespace yolo { namespace cpu
{
    const int ARG = -1;

    template<int value>
    struct Value
    {
        typedef Value<value> Type;
        operator int() const { return value; }
    };

    template<>
    struct Value<ARG>
    {
        typedef int Type;
    };

    template<int DATA_SIZE = ARG, int KERNEL_SIZE = ARG, int DEPTH = ARG, int KERNELS = ARG>
    void fast_convolution_impl(float const * const input,
                          float const * const weights,
                          float const * const biases,
                          float * const output,
                          const typename Value<DATA_SIZE>::Type data_size,
                          const typename Value<KERNEL_SIZE>::Type kernel_size,
                          const typename Value<DEPTH>::Type depth,
                          const typename Value<KERNELS>::Type kernels)
    {
        const int r = kernel_size/2;
        #pragma omp parallel for
        for (int kernel = 0; kernel < kernels; ++kernel)
        {
            float const * const kernel_weights = weights + kernel*(depth*kernel_size*kernel_size);
            float * const kernel_output = output + kernel*data_size*data_size;
            const float bias = biases[kernel];

            for (int target_y = 0; target_y < data_size; ++target_y)
            for (int target_x = 0; target_x < data_size; target_x += (data_size-1))
            {
                float sum = bias;

                for (int z = 0; z < depth; ++z)
                {
                    float const * const z_input = input + z*data_size*data_size;
                    float const * const z_weights = kernel_weights + z*kernel_size*kernel_size;

                    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
                        for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                        {
                            const int source_y = target_y + kernel_y - r;
                            const int source_x = target_x + kernel_x - r;

                            if (source_x < 0 || source_y < 0 || source_x >= data_size || source_y >= data_size)
                                continue;

                            const float input_value = z_input[source_y*data_size + source_x];
                            const float weight = z_weights[kernel_y*kernel_size + kernel_x];

                            sum += input_value * weight;
                        }
                }

                kernel_output[target_y*data_size + target_x] = sum;
            }

            for (int target_y = 0; target_y < data_size; target_y+=(data_size-1))
            for (int target_x = 1; target_x < data_size-1; ++target_x)
            {
                float sum = bias;

                for (int z = 0; z < depth; ++z)
                {
                    float const * const z_input = input + z*data_size*data_size;
                    float const * const z_weights = kernel_weights + z*kernel_size*kernel_size;

                    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
                        for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                        {
                            const int source_y = target_y + kernel_y - r;
                            const int source_x = target_x + kernel_x - r;

                            if (source_x < 0 || source_y < 0 || source_x >= data_size || source_y >= data_size)
                                continue;

                            const float input_value = z_input[source_y*data_size + source_x];
                            const float weight = z_weights[kernel_y*kernel_size + kernel_x];

                            sum += input_value * weight;
                        }
                }

                kernel_output[target_y*data_size + target_x] = sum;
            }

            for (int target_y = 1; target_y < data_size-1; ++target_y)
            for (int target_x = 1; target_x < data_size-1; ++target_x)
            {
                float sum = bias;

                for (int z = 0; z < depth; ++z)
                {
                    float const * const z_input = input + z*data_size*data_size;
                    float const * const z_weights = kernel_weights + z*kernel_size*kernel_size;

                    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
                        for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                        {
                            const int source_y = target_y + kernel_y - r;
                            const int source_x = target_x + kernel_x - r;

                            const float input_value = z_input[source_y*data_size + source_x];
                            const float weight = z_weights[kernel_y*kernel_size + kernel_x];

                            sum += input_value * weight;
                        }
                }

                kernel_output[target_y*data_size + target_x] = sum;
            }
        }
    }

    void fast_convolution(float const * const input,
                          float const * const weights,
                          float const * const biases,
                          float * const output,
                          const int data_size, 
                          const int kernel_size,
                          const int depth,
                          const int kernels)
    {
        if (data_size == 13 && kernel_size == 3 && depth == 512 && kernels == 1024)
        {
            fast_convolution_impl<13, 3, 512, 1024>(input, weights, biases, output, {}, {}, {}, {});
            return;
        }
        if (data_size == 13 && kernel_size == 3 && depth == 1024 && kernels == 1024)
        {
            fast_convolution_impl<13, 3, 1024, 1024>(input, weights, biases, output, {}, {}, {}, {});
            return;
        }
        if (kernel_size == 3)
        {
            fast_convolution_impl<ARG, 3>(input, weights, biases, output, data_size, {}, depth, kernels);
            return;
        }
        if (kernel_size == 1)
        {
            fast_convolution_impl<ARG, 1>(input, weights, biases, output, data_size, {}, depth, kernels);
            return;
        }
        fast_convolution_impl<ARG>(input, weights, biases, output, data_size, kernel_size, depth, kernels);
    }
}}}