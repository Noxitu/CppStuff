#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <numeric>
#include <noxitu/yolo/fast_convolution.h>
#include <noxitu/yolo/amp_convolution.h>

struct fast_tag {};
struct amp_tag {};

template<typename T> 
struct BM_Convolution_Policy;

template<> 
struct BM_Convolution_Policy<fast_tag>
{
    constexpr static auto convolute = &noxitu::yolo::fast_convolution;
};

template<> 
struct BM_Convolution_Policy<amp_tag>
{
    constexpr static auto convolute = &noxitu::yolo::amp_convolution;
};

template <typename PolicyTag>
class Test_Convolution : public ::testing::Test 
{
};

static const std::vector<float> expectedOutput = {941, 2022, 3103, 1491, 3328, 5165, 1791, 4060, 6329, 2091, 4792, 7493, 1373, 3318, 5263, 2071, 4772, 7473, 3082, 7457, 11832, 3424, 8447, 13470, 3766, 9437, 15108, 2383, 6380, 10377, 3331, 8192, 13053, 4792, 12407, 20022, 5134, 13397, 21660, 5476, 14387, 23298, 3403, 9560, 15717, 4591, 11612, 18633, 6502, 17357, 28212, 6844, 18347, 29850, 7186, 19337, 31488, 4423, 12740, 21057, 2381, 7782, 13183, 3219, 11536, 19853, 3375, 12124, 20873, 3531, 12712, 21893, 2045, 8310, 14575};

using Test_ConvolutionTypes = ::testing::Types<fast_tag, amp_tag>;
TYPED_TEST_CASE(Test_Convolution, Test_ConvolutionTypes);

TYPED_TEST(Test_Convolution, SmallExample)
{
    const int data_size = 5;
    const int kernel_size = 3;
    const int depth = 2;
    const int kernels = 3;

    std::vector<float> input(data_size*data_size*depth);
    std::vector<float> weights(kernel_size*kernel_size*depth*kernels);
    std::vector<float> biases(kernels);
    std::vector<float> output(data_size*data_size*kernels);

    std::iota(input.begin(), input.end(), 1.f);
    std::iota(weights.begin(), weights.end(), 1.f);
    std::iota(biases.begin(), biases.end(), 1.f);

    noxitu::yolo::fast_convolution(input.data(), weights.data(), biases.data(), output.data(), data_size, kernel_size, depth, kernels);

    ASSERT_EQ(expectedOutput.size(), output.size());

    for (int i = 0; i < output.size(); ++i)
    {
        EXPECT_NEAR(expectedOutput.at(i), output.at(i), .01f) << " at index " << i;
    }
}