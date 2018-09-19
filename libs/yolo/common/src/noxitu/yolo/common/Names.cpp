#include <noxitu/yolo/common/Names.h>
#include <fstream>
#include <iostream>
#include <cctype>

namespace noxitu { namespace yolo { namespace common
{
    static void strip_whitespace(std::string &line)
    {
        auto first = line.begin();

        while (first != line.end() && std::isspace(*first))
            ++first;

        auto last = line.end();
        while (last != first && std::isspace(*std::prev(last)))
            --last;

        if (first != line.begin() || last != line.end())
            line = std::string(first, last);
    }

    std::vector<std::string> load_yolo_names(const std::string input_path)
    {
        return load_yolo_names(std::ifstream(input_path, std::ifstream::binary));
    }

    std::vector<std::string> load_yolo_names(std::istream &in)
    {
        std::vector<std::string> ret;

        while (!in.eof())
        {
            std::string line;
            std::getline(in, line);

            if (!in.eof() && in.fail())
                throw std::runtime_error("Failure occured while reading names.");
            
            strip_whitespace(line);

            if (line == "")
                continue;

            ret.push_back(line);
        }

        return ret;
    }
}}}

