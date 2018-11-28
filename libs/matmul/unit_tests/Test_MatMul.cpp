#include <gtest/gtest.h>
#include <noxitu/matmul/matmul.h>
#include <random>
#include <opencv2/core.hpp>

using namespace noxitu::matmul;

static cv::Mat1f random(const int rows, const int cols, const int seed)
{
    cv::Mat1f ret(rows, cols);

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);

    const auto rand = [&]() { return distribution(generator); };

    std::generate(ret.begin(), ret.end(), rand);

    return ret;
}

static cv::Mat1f mat(cv::Size size)
{
    return cv::Mat1f::zeros(size);
}

static float maxdiff(cv::InputArray a, cv::InputArray b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);

    double max_value;
    cv::minMaxLoc(diff, nullptr, &max_value);

    return static_cast<float>(max_value);
}

template<typename Tag>
class Test_MatMul : public ::testing::Test {};

using Tags = ::testing::Types<cv_tag, cv_t_tag, cpu_tag, cpu_t_tag, cpu_t_omp_tag, amp_tag>;
TYPED_TEST_CASE(Test_MatMul, Tags);

template<typename Tag> bool transposeB = false;
template<> bool transposeB<cv_t_tag> = true;
template<> bool transposeB<cpu_t_tag> = true;
template<> bool transposeB<cpu_t_omp_tag> = true;

TYPED_TEST(Test_MatMul, IsCorrect)
{
    cv::Mat1f a = (cv::Mat1f(2, 3) << 1, 0, 2, 0, 1, 3); //random(2, 3, 1);
    cv::Mat1f b = (cv::Mat1f(3, 4) << 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1); //random(3, 4, 2);
    cv::Mat1f expected_c = a * b;

    if (transposeB<TypeParam>)
        b = b.t();

    auto c = mat(expected_c.size());

    auto mul = MatMulFactory().create(TypeParam{});

    mul(a, b, c);

    EXPECT_LT(maxdiff(expected_c, c), 1e-3) << expected_c << c;
}
