#include <noxitu/yolo/common/Weights.h>
#include <fstream>
#include <iostream>

namespace noxitu { namespace yolo { namespace common
{
    YoloWeights load_yolo_weights(const std::string input_path)
    {
        return load_yolo_weights(std::ifstream(input_path, std::ifstream::binary));
    }

    template<typename T>
    static T read(std::istream &input)
    {
        T ret;
        input.read(reinterpret_cast<char*>(&ret), sizeof(T));
        return ret;
    }

    YoloWeights load_yolo_weights(std::istream &input)
    {
        YoloWeights ret;
        YoloWeights::Version &version = ret.version;

        version.major = read<int>(input);
        version.minor = read<int>(input);
        version.revision = read<int>(input);
        
        if (version.major >= 2)
            version.images_seen = read<size_t>(input);
        else
            version.images_seen = read<int>(input);

        std::cout << "Loading YOLO weights:" << '\n';
        std::cout << "  * Version: " << version.major << '.' << version.minor << "  rev " << version.revision << '\n';
        std::cout << "  * Images seen: " << version.images_seen << '\n';
        std::cout << std::flush;

        const std::ifstream::pos_type header_length = input.tellg();

        input.seekg(0, std::ifstream::end);
        const std::ifstream::pos_type data_length = input.tellg() - header_length;
        input.seekg(header_length, std::ifstream::beg);

        std::cout << "  * Header length: " << header_length << '\n';
        std::cout << "  * Data length: " << data_length << '\n';

        if (data_length % 4 != 0)
        {
            throw std::logic_error("Weights data length not divisible by 4 (size of float).");
        }

        const int weights_count = static_cast<int>(data_length / 4);
        std::cout << "  * Weights count: " << weights_count << '\n';
        std::cout << std::flush;

        std::initializer_list<int> size = {weights_count};
        ret.weights = cv::Mat1f(1, size.begin());
        input.read(reinterpret_cast<char*>(ret.weights.ptr()), data_length);

        return ret;
    }
}}}

