#include <noxitu/yolo/common/ConfigurationEntry.h>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iostream>

namespace noxitu { namespace yolo { namespace common
{
    class Parser
    {
    public:
        enum class ParseAllEntries { Yes, No };
        const static ParseAllEntries ignore_unused = ParseAllEntries::No;

        enum class MappingRequired { Yes, No };
        const static MappingRequired required = MappingRequired::Yes;
        const static MappingRequired optional = MappingRequired::No;
    private:
        std::map<std::string, std::function<void(std::map<std::string, std::string> const &)>> mappings;

        template<typename T>
        static T parse_value(std::string text);
    public:
        template<typename T>
        void map(std::string name, T &target, MappingRequired mapping_required);
        void parse(std::map<std::string, std::string> const &values, ParseAllEntries parse_all = ParseAllEntries::Yes) const;
    };

    template<typename T>
    T Parser::parse_value(std::string text)
    {
        std::stringstream ss(text);
        T ret;
        ss >> ret;
        return ret;
    }

    template<>
    std::string Parser::parse_value<std::string>(std::string text)
    {
        return text;
    }

    template<>
    std::vector<float> Parser::parse_value<std::vector<float>>(std::string text)
    {
        std::stringstream ss(text);
        std::vector<float> ret;
        
        while (!ss.eof())
        {
            std::string word;
            std::getline(ss, word, ',');
            ret.push_back(parse_value<float>(word));
        }

        return ret;
    }

    template<typename T>
    void Parser::map(std::string name, T &target, MappingRequired mapping_required)
    {
        mappings[name] = [=, &target](std::map<std::string, std::string> const &values)
        {
            const auto entry = values.find(name);
            const bool found = (entry != values.end());

            if (!found && mapping_required == MappingRequired::Yes)
                throw std::logic_error("Required entry was not present.");

            if (!found)
                return;

            target = parse_value<T>(entry->second);
        };
    }

    void Parser::parse(std::map<std::string, std::string> const &values, ParseAllEntries parse_all) const
    {
        if (parse_all == ParseAllEntries::Yes)
        {
            const bool all_mapped = std::none_of(values.begin(), values.end(), [&](const std::pair<std::string, std::string> &entry)
            {
                return mappings.count(entry.first) == 0;
            });

            if (!all_mapped)
                throw std::logic_error("Unmapped entry provided");
        }

        for (const auto &mapping : mappings)
        {
            mapping.second(values);
        }
    }

    std::shared_ptr<ConfigurationEntry> create_configuration_entry(GenericConfigurationEntry const &entry)
    {
        if (entry.name == "net")
            return std::make_shared<NetConfigurationEntry>(entry);

        if (entry.name == "convolutional")
            return std::make_shared<ConvolutionalConfigurationEntry>(entry);

        if (entry.name == "maxpool")
            return std::make_shared<MaxPoolConfigurationEntry>(entry);

        if (entry.name == "region")
            return std::make_shared<RegionConfigurationEntry>(entry);

        throw std::logic_error("Unknown entry header.");
        //return std::make_shared<GenericConfigurationEntry>(entry);
    }

    NetConfigurationEntry::NetConfigurationEntry(GenericConfigurationEntry const &entry)
    {
        Parser parser;

        parser.map("width", width, Parser::required);
        parser.map("height", height, Parser::required);
        parser.map("channels", channels, Parser::required);

        parser.parse(entry.settings, Parser::ignore_unused);
    }

    ConvolutionalConfigurationEntry::ConvolutionalConfigurationEntry(GenericConfigurationEntry const &entry)
    {
        Parser parser;

        parser.map("batch_normalize", batch_normalize, Parser::optional);
        parser.map("filters", filters, Parser::required);
        parser.map("size", size, Parser::required);
        parser.map("stride", stride, Parser::required);
        parser.map("pad", pad, Parser::required);
        parser.map("activation", activation, Parser::required);

        parser.parse(entry.settings);
    }

    MaxPoolConfigurationEntry::MaxPoolConfigurationEntry(GenericConfigurationEntry const &entry)
    {
        Parser parser;

        parser.map("size", size, Parser::required);
        parser.map("stride", stride, Parser::required);

        parser.parse(entry.settings);
    }

    RegionConfigurationEntry::RegionConfigurationEntry(GenericConfigurationEntry const &entry)
    {
        Parser parser;

        parser.map("anchors", anchors, Parser::required);
        parser.map("num", number_of_boxes, Parser::required);
        parser.map("classes", number_of_classes, Parser::required);

        parser.parse(entry.settings, Parser::ignore_unused);
    }

}}}

