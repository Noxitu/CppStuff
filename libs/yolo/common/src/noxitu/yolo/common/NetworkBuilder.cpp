#include <noxitu/yolo/common/NetworkBuilder.h>
#include <noxitu/yolo/common/ConfigurationEntry.h>

namespace noxitu { namespace yolo { namespace common
{
    void apply_network_configuration(NetworkBuilder &builder, NetworkConfiguration const &configuration)
    {
        for (const auto &entry_sp : configuration)
        {
            ConfigurationEntry const *entry = entry_sp.get();

            if (auto net_entry = dynamic_cast<NetConfigurationEntry const*>(entry))
            {
                builder.setup(*net_entry);
                continue;
            }

            if (auto conv_entry = dynamic_cast<ConvolutionalConfigurationEntry const*>(entry))
            {
                builder.add_layer(*conv_entry);
                continue;
            }

            if (auto maxpool_entry = dynamic_cast<MaxPoolConfigurationEntry const*>(entry))
            {
                builder.add_layer(*maxpool_entry);
                continue;
            }

            if (auto region_entry = dynamic_cast<RegionConfigurationEntry const*>(entry))
            {
                builder.finalize(*region_entry);
                continue;
            }

            throw std::logic_error("Unsupported layer while building network.");
        }
    }
}}}