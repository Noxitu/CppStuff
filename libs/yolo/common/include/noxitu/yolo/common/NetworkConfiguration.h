#pragma once
#include <memory>
#include <string>
#include <vector>

namespace noxitu { namespace yolo { namespace common
{
    struct ConfigurationEntry;
    typedef std::vector<std::shared_ptr<ConfigurationEntry>> NetworkConfiguration;

    NetworkConfiguration read_network_configuration(const std::string input_path);
    NetworkConfiguration read_network_configuration(std::istream &in);
}}}