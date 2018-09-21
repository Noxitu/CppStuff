#include <noxitu/yolo/cpu/fast_convolution.h>
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

                //weights.forEach([&](float &value, const int *position)
                // Bug in OpenCV ~3.1 #8447
                cv::Vec4i position = {};
                for (position[0] = 0; position[0] < weights.size[0]; ++position[0])
                for (position[1] = 0; position[1] < weights.size[1]; ++position[1])
                for (position[2] = 0; position[2] < weights.size[2]; ++position[2])
                for (position[3] = 0; position[3] < weights.size[3]; ++position[3])
                {
                    float &value = weights(position);
                    const int kernel = position[0];
                    const float beta = batch_normalization_beta(kernel);
                    const float gamma = batch_normalization_gamma(kernel);
                    const float mean = batch_normalization_mean(kernel);
                    const float variance = batch_normalization_variance(kernel);
                    const float stddev = std::sqrt(variance+batch_normalization_epsilon);

                    const float scale = gamma / stddev;
                    value = value * scale;
                }

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
            data = reorder<float, 3>(data, {2, 0, 1});

            if (data.dims != 3)
                throw std::logic_error("Expected 3d input.");
            
            if (data.size[0] != depth)
                throw std::logic_error("Input has wrong depth.");

            const std::initializer_list<int> output_size = {kernels, data.size[1], data.size[2]};

            cv::Mat1f result = init_mat<float>(output_size);

            const cv::Rect2i roi = {{}, cv::Size{data.size[2], data.size[1]}};

            const int r = size/2;
#if 1 // orig
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

                            const float weight = weights(address);

                            sum += input * weight;
                        }

                value = sum;
            });
#else
            //const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            fast_convolution(&data(0), &weights(0), &biases(0), &result(0), data.size[1], weights.size[2], depth, kernels);
            //const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            //std::cout << " * Took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" << std::endl;
#endif

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

            result.forEach([&](float &value, const int *)
            {
                value = activation_function(value);
            });

            result = reorder<float, 3>(result, {1, 2, 0});

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
            //data = reorder<float, 3>(data, {1, 2, 0});

            if (data.dims != 3) throw std::runtime_error("Maxpool layer expected 3d input.");

            const cv::Size2i input_size = {data.size[1], data.size[0]};
            const int depth = data.size[2];
            cv::Size2i output_size = input_size / stride;

            cv::Mat1f result = init_mat<float>({output_size.height, output_size.width, depth});

            cv::Rect2i roi = {{}, input_size};

            result.forEach([&](float &value, const int *position)
            {
                value = -std::numeric_limits<float>::infinity();

                const int z = position[2];

                for (int dy = 0; dy < size; ++dy)
                    for (int dx = 0; dx < size; ++dx)
                    {
                        const int y = position[0]*stride + dy;
                        const int x = position[1]*stride + dx;

                        if (!roi.contains({x, y}))
                            continue;

                        value = std::max(value, data(y, x, z));
                    }
            });

            //result = reorder<float, 3>(result, {2, 0, 1});
            return result;
        }
    };


    class RouteLayer : public Layer
    {
    private:
        std::vector<int> offsets;
    public:
        RouteLayer(std::vector<int> offsets) :
            offsets(offsets)
        {}

        cv::Mat1f process(LayerInput const &input) const override
        {
            std::vector<cv::Mat1f> data;

            int height, width;

            for (auto offset : offsets)
            {
                cv::Mat1f single_data = input.get(offset);
                //single_data = reorder<float, 3>(single_data, {1, 2, 0});

                height = single_data.size[0];
                width = single_data.size[1];
                const int current_depth = single_data.size[2];
                //std::cout << cv::Vec3i{single_data.size[0], height, width} << std::endl;

                single_data = reshape(single_data, {height*width, current_depth});

                data.push_back(single_data);
            }

            cv::Mat1f result;
            cv::hconcat(data, result);

            const int depth = result.cols;
            result = reshape(result, {height, width, depth});
            //result = reorder<float, 3>(result, {2, 0, 1});
            return result;
        }
    };

    class ReorgLayer : public Layer
    {
    private:
        int stride;
    public:
        ReorgLayer(int stride) :
            stride(stride)
        {}

        cv::Mat1f process(LayerInput const &input) const override
        {
            cv::Mat1f data = input.get();
            //data = reorder<float, 3>(data, {1, 2, 0});

            auto size = data.size;
            std::initializer_list<int> new_shape = {size[0]/stride, size[1]/stride, size[2]*stride*stride};

            cv::Mat1f result = reshape(data, new_shape);
            //result = reorder<float, 3>(result, {2, 0, 1});
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
        net.input_size = {entry.width, entry.height};
        sizes.push_back({entry.channels, entry.height, entry.width});
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::ConvolutionalConfigurationEntry const &entry)
    {
        const int previous_depth = sizes.back()[0];

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

        sizes.push_back({entry.filters, sizes.back()[1], sizes.back()[2]});
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::MaxPoolConfigurationEntry const &entry)
    {
        net << std::make_shared<MaxPoolLayer>(entry.size, entry.stride);

        sizes.push_back({sizes.back()[0], sizes.back()[1]/entry.stride, sizes.back()[2]/entry.stride});
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::RouteConfigurationEntry const &entry)
    {
        if (entry.layers.size() == 0) throw std::logic_error("Cant route 0 layers.");

        const auto get_size = [&](int offset) { return sizes.at(sizes.size()+offset); };

        int depth = 0;
        const int height = get_size(entry.layers.front())[1];
        const int width = get_size(entry.layers.front())[2];
        
        for (int offset : entry.layers)
        {
            if (height != get_size(offset)[1] || width != get_size(offset)[2])
                throw std::logic_error("Routing differently sized layers.");

            depth += get_size(offset)[0];
        }

        net << std::make_shared<RouteLayer>(entry.layers);

        sizes.push_back({depth, height, width});
    }

    void NetworkBuilder::add_layer(noxitu::yolo::common::ReorgConfigurationEntry const &entry)
    {
        const cv::Vec3i prev = sizes.back();
        const int k = entry.stride;

        net << std::make_shared<ReorgLayer>(entry.stride);

        sizes.push_back({prev[0]*k*k, prev[1]/k, prev[2]/k});
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