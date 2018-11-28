#include <noxitu/matmul/matmul.h>

namespace noxitu { namespace matmul
{
    void cpu_matmul(cv::Mat1f a, cv::Mat1f b, cv::Mat1f c)
    {
        CV_Assert(a.isContinuous());
        CV_Assert(b.isContinuous());
        CV_Assert(c.isContinuous());
        CV_Assert(a.rows == c.rows);
        CV_Assert(b.cols == c.cols);
        CV_Assert(a.cols == b.rows);

        const int rows = c.rows;
        const int cols = c.cols;
        const int muls = a.cols;

        const float *A = &a(0);
        const float *B = &b(0);
        float *C = &c(0);

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                float result = 0;

                for (int k = 0; k < muls; ++k)
                {
                    result += A[y*muls + k] * B[k*cols + x];
                }

                C[y*cols + x] = result;
            }
        }
    }

    template<bool omp>
    void cpu_matmul_transposed(cv::Mat1f a, cv::Mat1f b, cv::Mat1f c)
    {
        CV_Assert(a.isContinuous());
        CV_Assert(b.isContinuous());
        CV_Assert(c.isContinuous());
        CV_Assert(a.rows == c.rows);
        CV_Assert(b.rows == c.cols);
        CV_Assert(a.cols == b.cols);

        const int rows = c.rows;
        const int cols = c.cols;
        const int muls = a.cols;

        const float *A = &a(0);
        const float *B = &b(0);
        float *C = &c(0);

        #pragma omp parallel for if (omp)
        for (int y = 0; y < rows; ++y)
        {
            const float *a_strip = A + y*muls;

            for (int x = 0; x < cols; ++x)
            {
                const float *b_strip = B + x*muls;

                float result = 0;

                for (int k = 0; k < muls; ++k)
                {
                    result += a_strip[k] * b_strip[k];
                }

                C[y*cols + x] = result;
            }
        }
    }
    
    template<>
    Algorithm MatMulFactory::create(cpu_tag)
    {
        return cpu_matmul;
    }

    template<>
    Algorithm MatMulFactory::create(cpu_t_tag)
    {
        return cpu_matmul_transposed<false>;
    }

    template<>
    Algorithm MatMulFactory::create(cpu_t_omp_tag)
    {
        return cpu_matmul_transposed<true>;
    }
}}