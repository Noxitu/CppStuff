
int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }

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
            float * const kernel_output = output + kernel;
            const float bias = biases[kernel];

            for (int target_y = 0; target_y < data_size; ++target_y)
            {
                float * const y_output = kernel_output + target_y*data_size*kernels;

                const int kernel_y_first = max(0, r-target_y);
                const int kernel_y_size = min(kernel_size, data_size-(target_y-r));

                for (int target_x = 0; target_x < data_size; ++target_x)
                {
                    float sum = bias;

                    const int kernel_x_first = max(0, r-target_x);
                    const int kernel_x_size = min(kernel_size, data_size-(target_x-r));

                    for (int kernel_y = kernel_y_first; kernel_y < kernel_y_size; ++kernel_y)
                    {
                        const int source_y = target_y + kernel_y - r;

                        float const * const y_input = input + source_y*data_size*depth;
                        float const * const y_weights = kernel_weights + kernel_y*kernel_size*depth;

                        for (int kernel_x = kernel_x_first; kernel_x < kernel_x_size; ++kernel_x)
                        {
                            const int source_x = target_x + kernel_x - r;

                            float const * const x_input = y_input + source_x*depth;
                            float const * const x_weights = y_weights + kernel_x*depth;

                            for (int z = 0; z < depth; ++z)
                            {
                                const float input_value = x_input[z];
                                const float weight = x_weights[z];

                                sum += input_value * weight;
                            }
                        }
                    }

                    y_output[target_x*kernels] = sum;
                }
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