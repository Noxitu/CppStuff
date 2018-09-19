#pragma once
#include <opencv2/core.hpp>
#include <memory>

namespace noxitu { namespace yolo { namespace cpu
{
    class LayerInput
    {
    private:
        std::vector<cv::Mat1f> const &data;
    public:
        LayerInput(std::vector<cv::Mat1f> const &data);
        cv::Mat1f get(int offset = 0) const;
    };

    class Layer
    {
    public:
        virtual cv::Mat1f process(LayerInput const &input) const = 0;
    };

    class Network
    {
    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        cv::Mat2f anchors;
        int number_of_boxes;
        int number_of_classes;

        void operator<< (std::shared_ptr<Layer> layer);
        cv::Mat1f process(cv::Mat1f data) const;
    };
}}}