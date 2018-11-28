#include <noxitu/matmul/matmul.h>
#include <amp.h>

namespace noxitu { namespace matmul
{
    void amp_matmul(cv::Mat1f a, cv::Mat1f b, cv::Mat1f c)
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

        using namespace concurrency;

        const extent<2> a_size(rows, muls);
        const extent<2> b_size(muls, cols);
        const extent<2> c_size(rows, cols);

        array_view<const float, 2> a_array(a_size, &a(0));
        array_view<const float, 2> b_array(b_size, &b(0));
        array_view<float, 2> c_array(c_size, &c(0));

        c_array.discard_data();

        parallel_for_each(c_size, [=](index<2> idx) restrict(amp)
        {
            const int y = idx[0];
            const int x = idx[1];
            
            float result = 0;

            for (int k = 0; k < muls; ++k)
            {
                result += a_array(y, k) * b_array(k, x);
            }

            c_array[idx] = result;
        });

        c_array.synchronize();
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
                    result += A[k] * B[k];
                }

                C[y*cols + x] = result;
            }
        }
    }
    
    template<>
    Algorithm MatMulFactory::create(amp_tag)
    {
        return amp_matmul;
    }
}}