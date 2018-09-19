#include <noxitu/yolo/cpu/NetworkBuilder.h>
#include <noxitu/yolo/common/ConfigurationEntry.h>
#include <noxitu/yolo/common/Utils.h>
#include <chrono>
#include <iostream>
#include <numeric>

using namespace noxitu::yolo::common::utils;

namespace noxitu { namespace yolo { namespace cpu
{
    enum class ActivationFunctionType 
    {
        Leaky, Linear
    };

    class Conv2dLayer : public Layer
    {
    private:
        int kernels;
        int depth;
        int size;

        bool batch_normalization;
        cv::Mat1f batch_normalization_beta;
        cv::Mat1f batch_normalization_gamma;
        cv::Mat1f batch_normalization_mean;
        cv::Mat1f batch_normalization_variance;
        const float batch_normalization_epsilon = 1e-3f;

        cv::Mat1f biases;
        cv::Mat1f weights;
        float (*activation_function)(float);

        static float leaky_activation(float x)
        {
            return std::max(x, .1f*x);
        }

        static float linear_activation(float x)
        {
            return x;
        }
    public:
        Conv2dLayer(cv::Mat1f batch_normalization_or_biases, cv::Mat1f weights, int stride, ActivationFunctionType activation_type) :
            weights(weights)
        {

            if (stride != 1)
                throw std::logic_error("Stride for conv layer must be 1.");

            if (weights.dims != 4) 
                throw std::logic_error("Expected 4d array of weights.");

            switch(activation_type)
            {
            case ActivationFunctionType::Leaky:
                activation_function = &leaky_activation;
                break;
            case ActivationFunctionType::Linear:
                activation_function = &linear_activation;
                break;
            default:
                throw std::logic_error("Unknown activation function.");
            }

            kernels = weights.size[0];

            depth = weights.size[1];

            if (weights.size[2] != weights.size[3])
                throw std::logic_error("Cant determine size of kernel.");

            size = weights.size[2];

            if (batch_normalization_or_biases.dims != 2)
                throw std::logic_error("Invalid batch_normalization_or_biases dims.");

            if (batch_normalization_or_biases.rows == 1)
            {
                batch_normalization = false;

                if (batch_normalization_or_biases.cols != kernels)
                    throw std::logic_error("Invalid biases length");

                biases = reshape(batch_normalization_or_biases, {kernels});
            }
            else if (batch_normalization_or_biases.rows == 4)
            {
                batch_normalization = true;
                biases = cv::Mat1f::zeros(kernels, 1);
                batch_normalization_beta = reshape(batch_normalization_or_biases.row(0), {kernels});
                batch_normalization_gamma = reshape(batch_normalization_or_biases.row(1), {kernels});
                batch_normalization_mean = reshape(batch_normalization_or_biases.row(2), {kernels});
                batch_normalization_variance = reshape(batch_normalization_or_biases.row(3), {kernels});
    #if 0
                batch_normalization = false;

                weights.forEach([&](float &value, const int *position)
                {
                    const int kernel = position[0];
                    const float beta = batch_normalization_beta(kernel);
                    const float gamma = batch_normalization_gamma(kernel);
                    const float mean = batch_normalization_mean(kernel);
                    const float variance = batch_normalization_variance(kernel);
                    const float stddev = std::sqrt(variance+batch_normalization_epsilon);

                    const float scale = gamma / stddev;
                    value = value * scale;
                });

                biases.forEach([&](float &value, const int *position)
                {
                    const int kernel = position[0];
                    const float beta = batch_normalization_beta(kernel);
                    const float gamma = batch_normalization_gamma(kernel);
                    const float mean = batch_normalization_mean(kernel);
                    const float variance = batch_normalization_variance(kernel);
                    const float stddev = std::sqrt(variance+batch_normalization_epsilon);

                    const float scale = gamma / stddev;
                    value = beta - mean * scale;
                });
    #endif
            }
            else
                throw std::logic_error("Invalid batch_normalization_or_biases shape.");
        }

        cv::Mat1f process(LayerInput const &input) const override
        {
            cv::Mat1f data = input.get();

            if (data.dims != 3)
                throw std::logic_error("Expected 3d input.");
            
            if (data.size[0] != depth)
                throw std::logic_error("Input has wrong depth.");

            const std::initializer_list<int> output_size = {kernels, data.size[1], data.size[2]};

            cv::Mat1f result = init_mat<float>(output_size);


            const cv::Rect2i roi = {{}, cv::Size{data.size[2], data.size[1]}};

            const int r = size/2;

            //std::cout << "processing:\n";
            //std::cout << " * kernels = " << kernels << '\n';
            //std::cout << " * depth = " << depth << '\n';
            //std::cout << " * size = " << size << '\n';
            //std::cout << " * input_size = " << print_size(data) << '\n';
            //std::cout << " * output_size = " << print_size(result) << '\n';
            //std::cout << " * r = " << r << '\n';
            //std::cout << std::flush;

            std::chrono::high_resolution_clock::time_point begin_ts = std::chrono::high_resolution_clock::now();

            result.forEach([&](float &value, const int *position)
            {
                const int kernel = position[0];
                const int target_y = position[1];
                const int target_x = position[2];

                float sum = biases(kernel);

                //cv::Mat1f kernel_weights = 

                for (int z = 0; z < depth; ++z)
                    for (int kernel_y = 0; kernel_y < size; ++kernel_y)
                        for (int kernel_x = 0; kernel_x < size; ++kernel_x)
                        {
                            const int source_y = target_y + kernel_y - r;
                            const int source_x = target_x + kernel_x - r;

                            if (!roi.contains({source_x, source_y}))
                                continue;

                            const float input = data(z, source_y, source_x);
                            const cv::Vec4i address = {kernel, z, kernel_y, kernel_x};

                            /*{
                                static std::mutex m;
                                std::lock_guard<std::mutex> _(m);
                                //std::cout << address << " of " << print_size(weight)
                            }*/
                            const float weight = weights(address);

                            sum += input * weight;
                        }

                value = sum;
            });

            if (batch_normalization)
            {
                result.forEach([&](float &value, const int *position)
                {
                    const int kernel = position[0];
                    const float beta = batch_normalization_beta(kernel);
                    const float gamma = batch_normalization_gamma(kernel);
                    const float mean = batch_normalization_mean(kernel);
                    const float variance = batch_normalization_variance(kernel);
                    const float stddev = std::sqrt(variance+batch_normalization_epsilon);

                    value = (value-mean) / stddev;
                    value = gamma * value + beta;
                });
            }

            std::chrono::high_resolution_clock::time_point end_ts = std::chrono::high_resolution_clock::now();
            //std::cout << " * Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ts-begin_ts).count() << "ms" << std::endl;

            result.forEach([&](float &value, const int *)
            {
                value = activation_function(value);
            });

            return result;
        }
    };

    class MaxPoolLayer : public Layer
    {
    private:
        int size, stride;
    public:
        MaxPoolLayer(int size, int stride) :
            size(size),
            stride(stride)
        {}

        cv::Mat1f process(LayerInput const &input) const override
        {
            cv::Mat1f data = input.get();

            if (data.dims != 3) throw std::runtime_error("Maxpool layer expected 3d input.");

            const int depth = data.size[0];

            int h, w;

            if (stride == size)
            {
                h = data.size[1]/size;
                w = data.size[2]/size;
            }
            else if (stride == 1)
            {
                h = data.size[1];
                w = data.size[2];
            }
            else 
                throw std::runtime_error("Can't handle size/stride combination.");

            cv::Mat1f result = init_mat<float>({depth, h, w});

            cv::Rect2i roi = {{}, cv::Size{data.size[2], data.size[1]}};

            result.forEach([&](float &value, const int *position)
            {
                value = -std::numeric_limits<float>::infinity();

                const int z = position[0];

                for (int dy = 0; dy < size; ++dy)
                    for (int dx = 0; dx < size; ++dx)
                    {
                        const int y = position[1]*stride + dy;
                        const int x = position[2]*stride + dx;

                        if (!roi.contains({x, y}))
                            continue;

                        value = std::max(value, data(z, y, x));
                    }
            });

            return result;
        }
    };


    cv::Mat1f NetworkBuilder::collect(const std::initializer_list<int> shape)
    {
        const int dims = static_cast<int>(shape.size());
        const int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        cv::Mat1f mat = weights.rowRange(offset, offset+total).reshape(1, dims, shape.begin());

        offset += static_cast<int>(mat.total());
        return mat;
    }

    NetworkBuilder::NetworkBuilder(cv::Mat1f weights) : weights(weights) {}

    void NetworkBuilder::setup(noxitu::yolo::common::NetConfigurationEntry const &entry)
    {
        previous_depth = entry.channels;
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::ConvolutionalConfigurationEntry const &entry)
    {
        cv::Mat1f biases_or_batch_normalization;
        
        if (entry.batch_normalize)
            biases_or_batch_normalization = collect({4, entry.filters});
        else
            biases_or_batch_normalization = collect({1, entry.filters});

        ActivationFunctionType activation_type = [&]()
        {
            if (entry.activation == "leaky") return ActivationFunctionType::Leaky;
            if (entry.activation == "linear") return ActivationFunctionType::Linear;

            throw std::logic_error("Unknown activation type.");
        }();

        auto weights = collect({entry.filters, previous_depth, entry.size, entry.size});

        net << std::make_shared<Conv2dLayer>(biases_or_batch_normalization, weights, entry.stride, activation_type);

        previous_depth = entry.filters;
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::MaxPoolConfigurationEntry const &entry)
    {
        net << std::make_shared<MaxPoolLayer>(entry.size, entry.stride);
    }

    void NetworkBuilder::finalize(noxitu::yolo::common::RegionConfigurationEntry const &entry)
    {
        if (offset != weights.total())
        {
            throw std::logic_error("Failed to extract weights (offset != weights.total()).");
        }

        net.anchors = cv::Mat(entry.anchors).reshape(2);
        net.number_of_boxes = entry.number_of_boxes;
        net.number_of_classes = entry.number_of_classes;
    }

    Network NetworkBuilder::build()
    {
        return net;
    }
}}}