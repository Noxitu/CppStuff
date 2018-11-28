#pragma once
#include <functional>
#include <opencv2/core/mat.hpp>

namespace noxitu { namespace matmul
{
    using Algorithm = std::function<void(cv::Mat1f, cv::Mat1f, cv::Mat1f)>;

    struct cv_tag {};
    struct cv_t_tag {};

    struct cpu_tag {};
    struct cpu_t_tag {};
    struct cpu_t_omp_tag {};

    struct amp_tag {};

    struct MatMulFactory
    {
        template<typename Tag>
        Algorithm create(Tag);
    };
}}