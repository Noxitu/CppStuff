#include <noxitu/yolo/cpu/Network.h>
#include <noxitu/yolo/common/Utils.h>
#include <iostream>

using namespace noxitu::yolo::common::utils;

namespace noxitu { namespace yolo { namespace cpu
{
    void Network::operator<< (sp<Layer> layer)
    {
        layers.push_back(layer);
    }

    cv::Mat1f Network::process(cv::Mat1f data) const
    {
        for (auto layer : layers)
        {
            const auto input_size = print_size(data);

            data = layer->process(data);

            const auto output_size = print_size(data);

            std::cout << typeid(*layer).name() << "  " << input_size << " -> " << output_size << std::endl;

        }

        return data;
    }
}}}