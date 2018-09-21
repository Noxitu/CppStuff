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
#include <noxitu/yolo/common/Names.h>
#include <noxitu/yolo/common/NetworkConfiguration.h>
#include <noxitu/yolo/common/Weights.h>
#include <noxitu/yolo/common/Utils.h>
#include <noxitu/yolo/cpu/NetworkBuilder.h>

using namespace noxitu::yolo::common::utils;

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
    if (mat.size[0] != anchors.rows) throw std::logic_error("Inconsistent number of boxes.");
    //if (mat.size[1] != 25) throw std::logic_error("");
    //if (mat.size[2] != 13) throw std::logic_error("");
    //if (mat.size[3] != 13) throw std::logic_error("");

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
        return lhs.confidence > rhs.confidence; 
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

static cv::Vec3b name_to_color(std::string name)
{
    const uchar color_code = std::accumulate(name.begin(), name.end(), (uchar) 0, [](uchar sum, uchar value) ->uchar 
    { 
        return (sum*17) ^ ((sum*17) >> 8) ^ value;
    }) * 180 / 255;

    cv::Mat3b color(1, 1, {color_code, 200, 200});
    cv::cvtColor(color, color, CV_HSV2BGR);
    return color(0);
}

int main() try
{
    //const std::string net_name = "yolov2"; const std::string classes_name = "coco";
    //const std::string net_name = "yolov2-tiny"; const std::string classes_name = "coco";
    const std::string net_name = "yolov2-tiny-voc"; const std::string classes_name = "voc";
    const auto network_configuration = noxitu::yolo::common::read_network_configuration("d:/sources/c++/data/yolo/cfg/" + net_name + ".cfg");
    const auto weights = noxitu::yolo::common::load_yolo_weights("d:/sources/c++/data/" + net_name + ".weights").weights;
    const auto names = noxitu::yolo::common::load_yolo_names("d:/sources/c++/data/yolo/cfg/" + classes_name + ".names");
    
    noxitu::yolo::cpu::Network net = [&]()
    {
        noxitu::yolo::cpu::NetworkBuilder builder(weights);
        noxitu::yolo::common::apply_network_configuration(builder, network_configuration);
        return builder.build();
    }();

#define FILE_INPUT
#ifdef FILE_INPUT
    {
        cv::Mat3f input_img = (cv::Mat3f) cv::imread("d:/sources/c++/data/yolo-dog.jpg");
#else
    cv::VideoCapture capture(0);

    if (!capture.isOpened())
    {
        throw std::runtime_error("Failed to open capture.");
    }

    while (true)
    {
        cv::Mat_<cv::Vec3b> frame;
        capture >> frame;

        cv::Mat3f input_img = frame;
#endif
        cv::resize(input_img, input_img, net.input_size, 0., 0., CV_INTER_CUBIC); 
        input_img /= 255;

        cv::Mat3f tmp_img;
        cv::cvtColor(input_img, tmp_img, CV_BGR2RGB);
        tmp_img = input_img;

        cv::Mat1f img = reshape(tmp_img, {tmp_img.rows, tmp_img.cols, 3});
        //img = reorder<float, 3>(img, {2, 0, 1});

        std::cout << "input image " << print_size(img) << std::endl;

        const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        cv::Mat1f result = net.process(img);
        const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        std::cout << "Took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" << std::endl;

        std::cout << "result " << print_size(result) << std::endl;
        result = reorder<float, 3>(result, {2, 0, 1});
        std::cout << "result " << print_size(result) << std::endl;

        result = reshape(result, {net.number_of_boxes, net.number_of_classes+5, result.size[1], result.size[2]});

        std::cout << "result " << print_size(result) << std::endl;

        auto boxes = convert_result(result, 0.3f, net.anchors);
        boxes = non_maximal_suppression(boxes, 0.3f);

        for (auto box : boxes)
        {
            const std::string name = names.at(box.class_id);
            const cv::Vec3f color = name_to_color(name) / 255.0;

            const int tickness = static_cast<int>(box.confidence/0.3f)+1;
            cv::rectangle(input_img, box.box, color, tickness);
            cv::Size2f text_size = cv::getTextSize(name, cv::FONT_HERSHEY_PLAIN, 1, 2, nullptr) + cv::Size2i{4, 4};

            {
                cv::Point2f tl = box.box.tl() - cv::Point2f{0, text_size.height};
                cv::rectangle(input_img, {tl, text_size}, color, CV_FILLED);
            }

            cv::putText(input_img, name, box.box.tl()+cv::Point2f{2, -2}, cv::FONT_HERSHEY_PLAIN, 1, {}, 2);
            std::cout << box.box << ' ' << box.confidence << ' ' << name << std::endl;
        }

#ifdef FILE_INPUT
        while (true)
        {
#else
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
}
catch(std::exception &ex)
{
    std::cerr << "ERROR: main() failed with exception " << typeid(ex).name() << ": \"" << ex.what() << "\"." << std::endl;
    return EXIT_FAILURE;
}