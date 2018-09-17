#pragma once
#include <noxitu/yolo/common/NetworkBuilder.h>
#include <noxitu/yolo/cpu/Network.h>

namespace noxitu { namespace yolo { namespace cpu
{
    class NetworkBuilder : public common::NetworkBuilder
    {
    private:
        cv::Mat1f weights;
        int offset = 0;
        int previous_depth = 0;
        Network net;

        cv::Mat1f collect(const std::initializer_list<int> shape);

    public:

        NetworkBuilder(cv::Mat1f weights);

        void setup(noxitu::yolo::common::NetConfigurationEntry const &entry) final;
        void add_layer(noxitu::yolo::common::ConvolutionalConfigurationEntry const &entry) final;
        void add_layer(noxitu::yolo::common::MaxPoolConfigurationEntry const &entry) final;
        void finalize(noxitu::yolo::common::RegionConfigurationEntry const &entry) final;

        Network build();
    };
}}}