#pragma once
#include <opencv2/core.hpp>

namespace noxitu { namespace yolo { namespace common
{
    struct YoloWeights
    {
        struct Version
        {
            int major;
            int minor;
            int revision;
            size_t images_seen;
        };

        Version version;
        cv::Mat1f weights;
    };

    YoloWeights load_yolo_weights(const std::string input_path);
    YoloWeights load_yolo_weights(std::istream &in);
}}}