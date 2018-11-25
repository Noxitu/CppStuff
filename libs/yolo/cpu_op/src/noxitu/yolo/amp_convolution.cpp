#include <amp.h>

namespace noxitu { namespace yolo
{
    const int ARG = -1;

    template<int value>
    struct Value
    {
        typedef Value<value> Type;
        constexpr operator int() const { return value; }
    };

    template<>
    struct Value<ARG>
    {
        typedef int Type;
    };

    template<int DATA_SIZE = ARG, int KERNEL_SIZE = ARG, int DEPTH = ARG, int KERNELS = ARG>
    void amp_convolution_impl(float const * const input_ptr,
                          float const * const weights_ptr,
                          float const * const biases_ptr,
                          float * const output_ptr,
                          const typename Value<DATA_SIZE>::Type _data_size,
                          const typename Value<KERNEL_SIZE>::Type _kernel_size,
                          const typename Value<DEPTH>::Type _depth,
                          const typename Value<KERNELS>::Type _kernels)
    {
        using namespace concurrency;

        const int data_size = _data_size;
        const int kernel_size = _kernel_size;
        const int depth = _depth;
        const int kernels = _kernels;

        const int r = kernel_size/2;

        const extent<3> input_size(data_size, data_size, depth);
        const int weights_size_ptr[] = {kernels, kernel_size, kernel_size, depth};
        const extent<4> weights_size(weights_size_ptr);
        const extent<1> biases_size(kernels);
        const extent<3> output_size(data_size, data_size, kernels);

        array_view<const float, 3> input(input_size, input_ptr);
        array_view<const float, 4> weights(weights_size, weights_ptr);
        array_view<const float, 1> biases(biases_size, biases_ptr);
        array_view<float, 3> output(output_size, output_ptr);

        output.discard_data();

        parallel_for_each(output.extent, [=](index<3> idx) restrict(amp)
        {
            const int target_y = idx[0];
            const int target_x = idx[1];
            const int kernel = idx[2];

            const int kernel_y_first = max(0, r-target_y);
            const int kernel_y_size = min(kernel_size, data_size-(target_y-r));

            const int kernel_x_first = max(0, r-target_x);
            const int kernel_x_size = min(kernel_size, data_size-(target_x-r));
            float sum = biases(kernel);

            index<4> weight_idx;
            weight_idx[0] = kernel;

            for (int kernel_y = kernel_y_first; kernel_y < kernel_y_size; ++kernel_y)
            {
                const int source_y = target_y + kernel_y - r;
                weight_idx[1] = kernel_y;

                for (int kernel_x = kernel_x_first; kernel_x < kernel_x_size; ++kernel_x)
                {
                    const int source_x = target_x + kernel_x - r;
                    weight_idx[2] = kernel_x;

                    for (int z = 0; z < depth; ++z)
                    {
                        weight_idx[3] = z;

                        const float input_value = input(source_y, source_x, z);

                        const float weight = weights(weight_idx);
                        sum += input_value * weight;
                    }
                }
            }
            output[idx] = sum;
        });
    }

    void amp_convolution(float const * const input,
                          float const * const weights,
                          float const * const biases,
                          float * const output,
                          const int data_size, 
                          const int kernel_size,
                          const int depth,
                          const int kernels)
    {
        amp_convolution_impl<>(input, weights, biases, output, data_size, kernel_size, depth, kernels);
    }
}}