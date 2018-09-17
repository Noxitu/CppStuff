#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace noxitu { namespace yolo { namespace common
{
    struct ConfigurationEntry 
    {
    public:
        virtual ~ConfigurationEntry() {}
    };

    struct GenericConfigurationEntry : public ConfigurationEntry
    {
        std::string name;
        std::map<std::string, std::string> settings;
    };

    struct NetConfigurationEntry : public ConfigurationEntry
    {
        int width;
        int height;
        int channels;

        NetConfigurationEntry(GenericConfigurationEntry const &entry);
    };

    struct ConvolutionalConfigurationEntry : public ConfigurationEntry
    {
        bool batch_normalize = false;
        int filters;
        int size;
        int stride;
        bool pad;
        std::string activation;

        ConvolutionalConfigurationEntry(GenericConfigurationEntry const &entry);
    };

    struct MaxPoolConfigurationEntry : public ConfigurationEntry
    {
        int size;
        int stride;

        MaxPoolConfigurationEntry(GenericConfigurationEntry const &entry);
    };

    struct RegionConfigurationEntry : public ConfigurationEntry
    {
        std::vector<float> anchors;

        RegionConfigurationEntry(GenericConfigurationEntry const &entry);
    };

    std::shared_ptr<ConfigurationEntry> create_configuration_entry(GenericConfigurationEntry const &entry);
}}}