#include <benchmark/benchmark.h>
#include <noxitu/yolo/fast_convolution.h>
#include <noxitu/yolo/amp_convolution.h>
#include <vector>
#include <random>

namespace noxitu { namespace yolo
{
    template<int DATA_SIZE = -1, int KERNEL_SIZE = -1, int DEPTH = -1, int KERNELS = -1>
    void fast_convolution_impl(float const * const input,
                          float const * const weights,
                          float const * const biases,
                          float * const output,
                          const int data_size,
                          const int kernel_size,
                          const int depth,
                          const int kernels);
}}

std::vector<float> getRandomVector(const int size)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.f, 1.f);

    const auto rand = [&]() { return distribution(generator); };

    std::vector<float> ret(size);
    std::generate(ret.begin(), ret.end(), rand);

    return ret;
}

struct fast_tag {};
struct fast_impl_tag {};
struct amp_tag {};

template<typename T> 
struct BM_Convolution_Policy;

template<> 
struct BM_Convolution_Policy<fast_tag>
{
    constexpr static auto convolute = &noxitu::yolo::fast_convolution;
};

template<> 
struct BM_Convolution_Policy<fast_impl_tag>
{
    constexpr static auto convolute = &noxitu::yolo::fast_convolution_impl<>;
};

template<> 
struct BM_Convolution_Policy<amp_tag>
{
    constexpr static auto convolute = &noxitu::yolo::amp_convolution;
};

template<typename PolicyTag>
static void BM_Convolution(benchmark::State& state) {
  std::string x = "hello";

  const int data_size = (int) state.range(0);
  const int kernel_size = (int) state.range(1);
  const int depth = (int) state.range(2);
  const int kernels = (int) state.range(3);

  const std::vector<float> input = getRandomVector(data_size*data_size*depth);
  const std::vector<float> weights = getRandomVector(kernel_size*kernel_size*depth*kernels);
  const std::vector<float> biases = getRandomVector(kernels);
  std::vector<float> output(data_size*data_size*kernels, 0);

  for (auto _ : state)
  {
    BM_Convolution_Policy<PolicyTag>::convolute(input.data(), weights.data(), biases.data(), output.data(), data_size, kernel_size, depth, kernels);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_Convolution, fast_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({416, 3, 3, 16})
    ->Args({208, 3, 16, 32})
    ->Args({104, 3, 32, 64})
    ->Args({52, 3, 64, 128})
    ->Args({26, 3, 128, 256})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 512, 1024})
    ->Args({13, 3, 1024, 512})
    ->Args({13, 1, 512, 425});

BENCHMARK_TEMPLATE(BM_Convolution, fast_impl_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({416, 3, 3, 16})
    ->Args({208, 3, 16, 32})
    ->Args({104, 3, 32, 64})
    ->Args({52, 3, 64, 128})
    ->Args({26, 3, 128, 256})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 512, 1024})
    ->Args({13, 3, 1024, 512})
    ->Args({13, 1, 512, 425});

    

BENCHMARK_TEMPLATE(BM_Convolution, amp_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({416, 3, 3, 16})
    ->Args({208, 3, 16, 32})
    ->Args({104, 3, 32, 64})
    ->Args({52, 3, 64, 128})
    ->Args({26, 3, 128, 256})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 256, 512})
    ->Args({13, 3, 512, 1024})
    ->Args({13, 3, 1024, 512})
    ->Args({13, 1, 512, 425});

BENCHMARK_MAIN();