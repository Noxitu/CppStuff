#pragma once
#include <vector>
#include <string>

namespace noxitu { namespace yolo { namespace common
{
    std::vector<std::string> load_yolo_names(const std::string input_path);
    std::vector<std::string> load_yolo_names(std::istream &in);
}}}