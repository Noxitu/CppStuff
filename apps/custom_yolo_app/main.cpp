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
#include <noxitu/yolo/common/NetworkConfiguration.h>
#include <noxitu/yolo/cpu/NetworkBuilder.h>
#include <noxitu/yolo/common/Weights.h>
#include <noxitu/yolo/common/Utils.h>

using namespace noxitu::yolo::common::utils;

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

std::vector<BoundingBox> convert_result(cv::Mat1f mat, const float confidence_threshold, const cv::Mat2f anchors)
{
    if (mat.dims != 4) throw std::logic_error("dims != 4");
    if (mat.size[0] != 5) throw std::logic_error("");
    if (mat.size[1] != 25) throw std::logic_error("");
    if (mat.size[2] != 13) throw std::logic_error("");
    if (mat.size[3] != 13) throw std::logic_error("");

    std::vector<BoundingBox> result;

    for (int y = 0; y < mat.size[2]; ++y)
        for (int x = 0; x < mat.size[3]; ++x)
            for (int i = 0; i < mat.size[0]; ++i)
            {
                std::initializer_list<cv::Range> range = {cv::Range(i, i+1), cv::Range::all(), cv::Range(y, y+1), cv::Range(x, x+1)};
                const cv::Mat1f entry = reshape(mat(range.begin()).clone(), {mat.size[1]});

                const cv::Mat1f probabilities = softmax(entry.rowRange(5, mat.size[1]));
                const int best_class = (int) std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

                const float object_confidence = sigmoid(entry(4));
                const float class_confidence = probabilities(best_class);

                const float confidence = object_confidence * class_confidence;

                if (confidence < confidence_threshold)
                    continue;


                cv::Point2f center;
                cv::Size2f size;

                center.x = (float(x) + sigmoid(entry(0))) * 32.0f;
                center.y = (float(y) + sigmoid(entry(1))) * 32.0f;

                cv::Point2f anchor = anchors(i);

                size.width = std::exp(entry(2)) * anchor.x * 32.0f;
                size.height = std::exp(entry(3)) * anchor.y * 32.0f;

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
    //const std::string net_name = "yolov2-voc";
    const std::string net_name = "yolov2-tiny-voc";
    auto network_configuration = noxitu::yolo::common::read_network_configuration("d:/sources/c++/data/yolo/cfg/" + net_name + ".cfg");
    auto weights = noxitu::yolo::common::load_yolo_weights("d:/sources/c++/data/" + net_name + ".weights").weights;
    
    noxitu::yolo::cpu::Network net = [&]()
    {
        noxitu::yolo::cpu::NetworkBuilder builder(weights);
        noxitu::yolo::common::apply_network_configuration(builder, network_configuration);
        return builder.build();
    }();

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

    auto boxes = convert_result(result, 0.3f, net.anchors);
    boxes = non_maximal_suppression(boxes, 0.3f);

    const std::vector<cv::Scalar> COLORS = { {254.0, 254.0, 254}, {239.88888888888889, 211.66666666666669, 127}, {225.77777777777777, 169.33333333333334, 0}, {211.66666666666669, 127.0, 254},{197.55555555555557, 84.66666666666667, 127}, {183.44444444444443, 42.33333333333332, 0},{169.33333333333334, 0.0, 254}, {155.22222222222223, -42.33333333333335, 127},{141.11111111111111, -84.66666666666664, 0}, {127.0, 254.0, 254}, {112.88888888888889, 211.66666666666669, 127}, {98.77777777777777, 169.33333333333334, 0},{84.66666666666667, 127.0, 254}, {70.55555555555556, 84.66666666666667, 127},{56.44444444444444, 42.33333333333332, 0}, {42.33333333333332, 0.0, 254}, {28.222222222222236, -42.33333333333335, 127}, {14.111111111111118, -84.66666666666664, 0},{0.0, 254.0, 254}, {-14.111111111111118, 211.66666666666669, 127}};
    const std::vector<std::string> NAMES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    for (auto box : boxes)
    {
        int tickness = static_cast<int>(box.confidence/0.3f)+1;
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
            return EXIT_SUCCESS;
        }
    }
}
catch(std::exception &ex)
{
    std::cerr << "ERROR: main() failed with exception " << typeid(ex).name() << ": \"" << ex.what() << "\"." << std::endl;
    return EXIT_FAILURE;
}