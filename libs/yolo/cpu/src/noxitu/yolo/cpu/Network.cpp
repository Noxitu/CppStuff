#include <noxitu/yolo/cpu/Network.h>
#include <noxitu/yolo/common/Utils.h>
#include <iostream>

using namespace noxitu::yolo::common::utils;

namespace noxitu { namespace yolo { namespace cpu
{
    LayerInput::LayerInput(std::vector<cv::Mat1f> const &data) : data(data) {}

    cv::Mat1f LayerInput::get(int offset) const
    {
        return data.at(data.size()-1+offset);
    }

    void Network::operator<< (sp<Layer> layer)
    {
        layers.push_back(layer);
    }

    cv::Mat1f Network::process(cv::Mat1f data) const
    {
        int layer_number = 0;

        std::vector<cv::Mat1f> outputs = {data};

        for (auto layer : layers)
        {
            const auto input_size = print_size(data);

            data = layer->process(outputs);
            outputs.push_back(data);

            const auto output_size = print_size(data);

            std::string name = typeid(*layer).name();

            if (name.size() > 25 && std::string(name.begin(), name.begin()+25) == "class noxitu::yolo::cpu::")
            {
                name = std::string(name.begin()+25, name.end());
            }

            std::cout << "#" << layer_number << "  " << name << "  " << input_size << " -> " << output_size << std::endl;

            layer_number += 1;
        }

        return data;
    }
}}}