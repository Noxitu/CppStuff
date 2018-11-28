#include <benchmark/benchmark.h>
#include <noxitu/matmul/matmul.h>
#include <random>

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

template<typename Tag> bool transposeB = false;
template<> bool transposeB<cv_t_tag> = true;
template<> bool transposeB<cpu_t_tag> = true;
template<> bool transposeB<cpu_t_omp_tag> = true;

template<typename Tag>
static void BM_MatMul(benchmark::State& state) 
{
    const int m = (int) state.range(0);
    const int n = (int) state.range(0);
    const int k = (int) state.range(0);

    auto a = random(m, k, 1);
    auto b = random(k, n, 2);

    if (transposeB<Tag>)
        b = b.t();

    auto c = mat({m, n});

    auto mul = MatMulFactory().create(Tag{});


    for (auto _ : state)
    {
        mul(a, b, c);
        benchmark::ClobberMemory();
    }
}

BENCHMARK_TEMPLATE(BM_MatMul, cv_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024});

BENCHMARK_TEMPLATE(BM_MatMul, cv_t_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024});

BENCHMARK_TEMPLATE(BM_MatMul, cpu_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024});

BENCHMARK_TEMPLATE(BM_MatMul, cpu_t_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024});

BENCHMARK_TEMPLATE(BM_MatMul, cpu_t_omp_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048});

BENCHMARK_TEMPLATE(BM_MatMul, amp_tag)
    ->Unit(benchmark::kMillisecond)
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048});

BENCHMARK_MAIN();