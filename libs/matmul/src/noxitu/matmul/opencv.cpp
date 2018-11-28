#include <noxitu/matmul/matmul.h>

namespace noxitu { namespace matmul
{
    void cv_matmul(cv::Mat1f a, cv::Mat1f b, cv::Mat1f c)
    {
        c = a * b;
    }

    void cv_matmul_transposed(cv::Mat1f a, cv::Mat1f b, cv::Mat1f c)
    {
        c = a * b.t();
    }
    
    template<>
    Algorithm MatMulFactory::create(cv_tag)
    {
        return cv_matmul;
    }

    template<>
    Algorithm MatMulFactory::create(cv_t_tag)
    {
        return cv_matmul_transposed;
    }
}}