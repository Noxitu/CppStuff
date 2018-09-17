#pragma once
#include <opencv2/core.hpp>
#include <memory>

namespace noxitu { namespace yolo { namespace cpu
{
    class Layer
    {
    public:
        virtual cv::Mat1f process(cv::Mat1f data) const = 0;
    };

    class Network
    {
    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        cv::Mat2f anchors;

        void operator<< (std::shared_ptr<Layer> layer);
        cv::Mat1f process(cv::Mat1f data) const;
    };
}}}