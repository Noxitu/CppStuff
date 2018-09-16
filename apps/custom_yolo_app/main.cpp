#include <numeric>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <atomic>
#include <chrono>
#include <iomanip>

template<typename T>
using sp = std::shared_ptr<T>;

struct Version
{
    int major;
    int minor;
    int revision;
#if 0
    size_t images_seen;
#else
    int images_seen;
#endif
};

std::string print_size(cv::Mat mat)
{
    std::stringstream ss;
    ss << "[ " << mat.size[0];

    for (int i = 1; i < mat.dims; ++i)
    {
        ss << " x " << mat.size[i];
    }

    ss << " ]";

    return ss.str();
}

template<typename T>
cv::Mat_<T> init_mat(std::initializer_list<int> shape)
{
    return cv::Mat_<T>(static_cast<int>(shape.size()), shape.begin());
}

cv::Mat reshape(cv::Mat image, std::initializer_list<int> new_shape)
{
    //std::cout << "reshape: " << print_size(image) << " -> (" << new_shape.size() << ") ";
    //for(int x : new_shape) std::cout << x << ' '; std::cout << std::endl;

    return image.reshape(1, static_cast<int>(new_shape.size()), new_shape.begin());
}

class Layer
{
public:
    virtual cv::Mat1f process(cv::Mat1f data) const = 0;
};

class Network
{
private:
    std::vector<sp<Layer>> layers;

public:
    void operator<< (sp<Layer> layer)
    {
        layers.push_back(layer);
    }

    cv::Mat1f process(cv::Mat1f data) const
    {
        for (auto layer : layers)
        {
            const auto input_size = print_size(data);

            data = layer->process(data);

            const auto output_size = print_size(data);

            std::cout << typeid(*layer).name() << "  " << input_size << " -> " << output_size << std::endl;

        }
        /*for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                for (int ch = 0; ch < 4; ++ch)       
                {
                    std::cout << data(ch, y, x) << ' ';
                }
                std::cout << '\n';
            }
            std::cout << std::endl;
        }

        exit(0);*/

        return data;
    }
};

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

    cv::Mat1f process(cv::Mat1f data) const override
    {
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
        std::cout << " * Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ts-begin_ts).count() << "ms" << std::endl;

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

    cv::Mat1f process(cv::Mat1f data) const override
    {
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

std::pair<Version, cv::Mat1f> load_weights()
{
    std::ifstream input("d:/sources/c++/data/yolov2-tiny-voc.weights", std::ifstream::binary);

    Version version;
    input.read(reinterpret_cast<char*>(&version.major), sizeof(int));
    input.read(reinterpret_cast<char*>(&version.minor), sizeof(int));
    input.read(reinterpret_cast<char*>(&version.revision), sizeof(int));
    input.read(reinterpret_cast<char*>(&version.images_seen), sizeof(version.images_seen));    

    std::cout << "Loading YOLO weights:" << '\n';
    std::cout << "  * Version: " << version.major << '.' << version.minor << "  rev " << version.revision << '\n';
    std::cout << "  * Images seen: " << version.images_seen << '\n';
    std::cout << std::flush;

    const std::ifstream::pos_type header_length = input.tellg();

    input.seekg(0, std::ifstream::end);
    const std::ifstream::pos_type data_length = input.tellg() - header_length;
    input.seekg(header_length, std::ifstream::beg);

    std::cout << "  * Header length: " << header_length << '\n';
    std::cout << "  * Data length: " << data_length << '\n';

    if (data_length % 4 != 0)
    {
        throw std::logic_error("Weights data length not divisible by 4 (size of float).");
    }

    const int weights_count = static_cast<int>(data_length / 4);
    std::cout << "  * Weights count: " << weights_count << '\n';
    std::cout << std::flush;

    std::initializer_list<int> size = {weights_count};
    cv::Mat1f weights(1, size.begin());
    input.read(reinterpret_cast<char*>(weights.ptr()), data_length);

    return {version, weights};
}

Network collect_yolov2_tiny_weights(cv::Mat1f weights)
{
    int offset = 0;
    const auto collect = [&](const std::initializer_list<int> shape) -> cv::Mat1f
    {
        const int dims = static_cast<int>(shape.size());
        const int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        cv::Mat1f mat = weights.rowRange(offset, offset+total).reshape(1, dims, shape.begin());

        offset += static_cast<int>(mat.total());
        return mat;
    };

    const auto collect1 = [&](const int size, const int depth, const int kernels) -> sp<Layer>
    {
        auto batch_normalization = collect({4, kernels});
        auto weights = collect({kernels, depth, size, size});

        return std::make_shared<Conv2dLayer>(batch_normalization, weights, 1, ActivationFunctionType::Leaky);
    };

    const auto collect2 = [&](const int size, const int depth, const int kernels) -> sp<Layer>
    {
        auto biases = collect({1, kernels});
        auto weights = collect({kernels, depth, size, size});

        return std::make_shared<Conv2dLayer>(biases, weights, 1, ActivationFunctionType::Linear);
    };

    Network net;

    net << collect1(3, 3, 16);
    net << std::make_shared<MaxPoolLayer>(2, 2);

    net << collect1(3, 16, 32);
    net << std::make_shared<MaxPoolLayer>(2, 2);

    net << collect1(3, 32, 64);
    net << std::make_shared<MaxPoolLayer>(2, 2);

    net << collect1(3, 64, 128);
    net << std::make_shared<MaxPoolLayer>(2, 2);

    net << collect1(3, 128, 256);
    net << std::make_shared<MaxPoolLayer>(2, 2);

    net << collect1(3, 256, 512);
    net << std::make_shared<MaxPoolLayer>(2, 1);

    net << collect1(3, 512, 1024);

    net << collect1(3, 1024, 1024);

    net << collect2(1, 1024, 125);


    if (offset != weights.total())
    {
        throw std::logic_error("Failed to extract weights (offset != weights.total()).");
    }

    return net;
}

cv::Mat1f reorder_image(cv::Mat3f rgb_image)
{
    const int H = rgb_image.rows;
    const int W = rgb_image.cols;
    cv::Mat1f input = reshape(rgb_image, {H, W, 3});
    cv::Mat1f output_image = init_mat<float>({3, H, W});

    input.forEach([&](float value, const int *pos)
    {
        output_image(pos[2], pos[0], pos[1]) = value;
    });

    return output_image;
}

struct BoundingBox
{
    cv::Rect2f box;

    float confidence;
    int class_id;
};

float sigmoid(float x)
{
  return 1.f / (1.f + std::exp(-x));
}

cv::Mat1f softmax(cv::Mat1f data)
{
    data = data.clone();
    const float max_value = *std::max_element(data.begin(), data.end());

    for (float &value : data)
        value = std::exp(value-max_value);

    const float sum = std::accumulate(data.begin(), data.end(), 0.0f);

    for (float &value : data)
        value /= sum;

    return data;
}

std::vector<BoundingBox> convert_result(cv::Mat1f mat, const float confidence_threshold)
{
    if (mat.dims != 4) throw std::logic_error("dims != 4");
    if (mat.size[0] != 5) throw std::logic_error("");
    if (mat.size[1] != 25) throw std::logic_error("");
    if (mat.size[2] != 13) throw std::logic_error("");
    if (mat.size[3] != 13) throw std::logic_error("");

    const std::vector<cv::Point2f> anchors = { {1.08f, 1.19f}, {3.42f, 4.41f}, {6.63f, 11.38f}, {9.42f, 5.11f}, {16.62f, 10.52f} };

    std::vector<BoundingBox> result;

    for (int y = 0; y < mat.size[2]; ++y)
        for (int x = 0; x < mat.size[3]; ++x)
            for (int i = 0; i < mat.size[0]; ++i)
            {
                std::initializer_list<cv::Range> range = {cv::Range(i, i+1), cv::Range::all(), cv::Range(y, y+1), cv::Range(x, x+1)};
                const cv::Mat1f entry = reshape(mat(range.begin()).clone(), {mat.size[1]});

                const cv::Mat1f probabilities = softmax(entry.rowRange(5, mat.size[1]));
                const int best_class = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

                const float object_confidence = sigmoid(entry(4));
                const float class_confidence = probabilities(best_class);

                const float confidence = object_confidence * class_confidence;

                if (confidence < confidence_threshold)
                    continue;


                cv::Point2f center;
                cv::Size2f size;

                center.x = (float(x) + sigmoid(entry(0))) * 32.0f;
                center.y = (float(y) + sigmoid(entry(1))) * 32.0f;

                size.width = std::exp(entry(2)) * anchors[i].x * 32.0f;
                size.height = std::exp(entry(3)) * anchors[i].y * 32.0f;

                const cv::Point2f offset = size*.5f;

                BoundingBox box;
                box.box = {center-offset, center+offset};
                box.confidence = confidence;
                box.class_id = best_class;

                result.push_back(box);

                //std::cout << "(" << x << ", " << y << ") " << entry.rowRange(0, 5).t() << " = " << box.box << std::endl;
            }

    return result;
}

float iou(const cv::Rect2f lhs, const cv::Rect2f rhs)
{
    const cv::Rect2f intersection = lhs & rhs;

    return intersection.area() / (lhs.area() + rhs.area() - intersection.area());
}

std::vector<BoundingBox> non_maximal_suppression(std::vector<BoundingBox> input, float iou_threshold)
{
    std::vector<BoundingBox> output;

    std::sort(input.begin(), input.end(), [](BoundingBox const &lhs, BoundingBox const &rhs) 
    { 
        return lhs.box.area() > rhs.box.area(); 
    });

    auto is_ok = [&](BoundingBox candidate)
    {
        for (auto added : output)
        {
            if (iou(candidate.box, added.box) > iou_threshold)
                return false;
        }

        return true;
    };

    for (auto box : input)
    {
        if (!is_ok(box)) continue;

        output.push_back(box);
    }

    return output;
}

int main() try
{
    //std::cout << std::setprecision(2) << std::fixed;
    auto weights = load_weights().second;
    Network net = collect_yolov2_tiny_weights(weights);

    //{
    
    cv::Mat3f input_img = (cv::Mat3f) cv::imread("d:/sources/c++/data/yolo-dog.jpg");
    cv::resize(input_img, input_img, {416, 416}, 0., 0., CV_INTER_CUBIC); 
    input_img /= 255;

    cv::Mat3f tmp_img;
    //cv::cvtColor(input_img, tmp_img, CV_BGR2RGB);
    tmp_img = input_img;

    cv::Mat1f img = reorder_image(tmp_img);
    std::cout << "input image " << print_size(img) << std::endl;

    cv::Mat1f result = net.process(img);
    result = reshape(result, {5, 25, 13, 13});

    std::cout << "result " << print_size(result) << std::endl;

    auto boxes = convert_result(result, 0.3);
    boxes = non_maximal_suppression(boxes, 0.3);

    const std::vector<cv::Scalar> COLORS = { {254.0, 254.0, 254}, {239.88888888888889, 211.66666666666669, 127}, {225.77777777777777, 169.33333333333334, 0}, {211.66666666666669, 127.0, 254},{197.55555555555557, 84.66666666666667, 127}, {183.44444444444443, 42.33333333333332, 0},{169.33333333333334, 0.0, 254}, {155.22222222222223, -42.33333333333335, 127},{141.11111111111111, -84.66666666666664, 0}, {127.0, 254.0, 254}, {112.88888888888889, 211.66666666666669, 127}, {98.77777777777777, 169.33333333333334, 0},{84.66666666666667, 127.0, 254}, {70.55555555555556, 84.66666666666667, 127},{56.44444444444444, 42.33333333333332, 0}, {42.33333333333332, 0.0, 254}, {28.222222222222236, -42.33333333333335, 127}, {14.111111111111118, -84.66666666666664, 0},{0.0, 254.0, 254}, {-14.111111111111118, 211.66666666666669, 127}};
    const std::vector<std::string> NAMES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    for (auto box : boxes)
    {
        int tickness = static_cast<float>(box.confidence/0.3)+1;
        cv::rectangle(input_img, box.box, COLORS.at(box.class_id), tickness);
        cv::Size2f text_size = cv::getTextSize(NAMES.at(box.class_id), cv::FONT_HERSHEY_PLAIN, 1, 2, nullptr) + cv::Size2i{4, 4};

        {
            cv::Point2f tl = box.box.tl() - cv::Point2f{0, text_size.height};
            cv::rectangle(input_img, {tl, text_size}, COLORS.at(box.class_id), CV_FILLED);
        }

        cv::putText(input_img, NAMES.at(box.class_id), box.box.tl()+cv::Point2f{2, -2}, cv::FONT_HERSHEY_PLAIN, 1, {}, 2);
        std::cout << box.box << ' ' << box.confidence << std::endl;
    }
    //std::initializer_list<int> shape = 
    //}

    //return 0;
#if 0
    cv::VideoCapture capture(0);

    if (!capture.isOpened())
    {
        throw std::runtime_error("Failed to open capture.");
    }

    while (true)
    {
        cv::Mat_<cv::Vec3b> img;
        capture >> img;
#else
    //cv::Mat_<cv::Vec3b> img = cv::imread("d:/sources/c++/data/yolo-dog.jpg");

    while (true)
    {
#endif

        cv::imshow("+", input_img);
        int key = cv::waitKey(1);

        switch(key&0xff)
        {
        case 'q':
        case 27:
            return 0;
        }
    }
}
catch(std::exception &ex)
{
    std::cerr << "main() failed with exception " << typeid(ex).name() << ": \"" << ex.what() << "\"." << std::endl;
}