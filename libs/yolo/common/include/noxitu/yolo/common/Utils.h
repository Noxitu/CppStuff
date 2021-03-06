#pragma once
#include <opencv2/core.hpp>
#include <memory>

namespace noxitu { namespace yolo { namespace common { namespace utils
{
    template<typename T>
    using sp = std::shared_ptr<T>;

    inline std::string print_size(cv::Mat mat)
    {
        std::stringstream ss;
        ss << "[ " << mat.size[0];

        for (int i = 1; i < mat.dims; ++i)
        {
            ss << " x " << mat.size[i];
        }

        ss << " ]";

        return ss.str();
    }

    template<typename T>
    inline cv::Mat_<T> init_mat(std::initializer_list<int> shape)
    {
        return cv::Mat_<T>(static_cast<int>(shape.size()), shape.begin());
    }

    template<typename T>
    inline cv::Mat_<T> init_mat(std::vector<int> shape)
    {
        return cv::Mat_<T>(static_cast<int>(shape.size()), shape.data());
    }

    inline cv::Mat reshape(cv::Mat image, std::initializer_list<int> new_shape)
    {
        return image.reshape(1, static_cast<int>(new_shape.size()), new_shape.begin());
    }

    template<typename Type, int N>
    inline cv::Mat_<Type> reorder(cv::Mat_<Type> data, cv::Vec<int, N> order)
    {
        if (data.dims != N) throw std::logic_error("Invalid reorder args.");

        std::vector<int> new_shape;

        for (int i = 0; i < N; ++i) 
            new_shape.push_back(data.size[order[i]]);

        cv::Mat1f ret = init_mat<Type>(new_shape);

        ret.forEach([&](Type &value, int const * const position)
        {
            cv::Vec<int, N> source;

            for (int i = 0; i < N; ++i)
                source[order[i]] = position[i];

            value = data(source);
        });

        return ret;
    }
}}}}