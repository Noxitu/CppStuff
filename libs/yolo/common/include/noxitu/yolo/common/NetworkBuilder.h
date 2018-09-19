#pragma once
#include <noxitu/yolo/common/NetworkConfiguration.h>

namespace noxitu { namespace yolo { namespace common
{
    struct NetConfigurationEntry;
    struct ConvolutionalConfigurationEntry;
    struct MaxPoolConfigurationEntry;
    struct RouteConfigurationEntry;
    struct ReorgConfigurationEntry;
    struct RegionConfigurationEntry;

    class NetworkBuilder
    {
    private:
    public:
        virtual void setup(NetConfigurationEntry const &entry) = 0;
        virtual void add_layer(ConvolutionalConfigurationEntry const &entry) = 0;
        virtual void add_layer(MaxPoolConfigurationEntry const &entry) = 0;
        virtual void add_layer(RouteConfigurationEntry const &entry) = 0;
        virtual void add_layer(ReorgConfigurationEntry const &entry) = 0;
        virtual void finalize(RegionConfigurationEntry const &entry) = 0;
    };

    void apply_network_configuration(NetworkBuilder &builder, NetworkConfiguration const &configuration);
}}}