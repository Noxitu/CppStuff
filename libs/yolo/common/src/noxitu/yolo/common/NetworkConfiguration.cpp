#include <noxitu/yolo/common/NetworkConfiguration.h>
#include <noxitu/yolo/common/ConfigurationEntry.h>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>

namespace noxitu { namespace yolo { namespace common
{
    NetworkConfiguration read_network_configuration(const std::string input_path)
    {
        return read_network_configuration(std::ifstream(input_path));
    }

    static void remove_comment(std::string &line)
    {
        auto it = std::find(line.begin(), line.end(), '#');

        if (it != line.end())
            line = std::string(line.begin(), it);
    }

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

    NetworkConfiguration read_network_configuration(std::istream &in)
    {
        NetworkConfiguration configuration;

        std::shared_ptr<GenericConfigurationEntry> current_entry;

        std::regex header_regex(R"(^\[(\w+)\])");
        std::regex mapping_regex(R"((\w+)\s*=\s*(\S.*))");

        auto add_entry = [&](const GenericConfigurationEntry &string_entry)
        {
            auto entry = create_configuration_entry(string_entry);
            configuration.push_back(entry);
        };

        auto handle_line = [&](const std::string line)
        {
            std::smatch match;

            if (std::regex_match(line, match, header_regex))
            {
                if (current_entry)
                    add_entry(*current_entry);

                current_entry = std::make_shared<GenericConfigurationEntry>();
                current_entry->name = match.str(1);
                return;
            }

            if (std::regex_match(line, match, mapping_regex))
            {
                if (current_entry == nullptr)
                    throw std::logic_error("Mapping without header found in .cfg file.");

                const std::string key = match.str(1);
                const std::string value = match.str(2);

                if (current_entry->settings.count(key) > 0)
                    throw std::logic_error("Duplicated entry found in .cfg.");
                
                current_entry->settings[key] = value;
                return;
            }

            throw std::logic_error("Unparsable line in .cfg.");
        };

        while (!in.eof())
        {
            std::string line;
            std::getline(in, line);

            if (!in.eof() && in.fail())
                throw std::runtime_error("Failure occured while reading configuration.");
            
            remove_comment(line);
            strip_whitespace(line);

            if (line == "")
                continue;

            handle_line(line);
        }

        if (current_entry)
            add_entry(*current_entry);

        return configuration;
    }
}}}