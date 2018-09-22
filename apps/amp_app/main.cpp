#include <amp.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>

namespace noxitu
{
    static const int SIZE = 1024; //1024;
    static const int W = SIZE;
    static const int INNER = SIZE;
    static const int H = SIZE;

    std::chrono::high_resolution_clock::time_point now()
    {
        return std::chrono::high_resolution_clock::now();
    }

    std::ostream& operator<< (std::ostream &out, std::chrono::high_resolution_clock::duration duration)
    {
        return out << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms";
    }

    void multiply1(float const * const A, float const * const B, float * const C)
    {
    #pragma omp parallel for
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float sum = 0;
                for (int k = 0; k < INNER; ++k)
                {
                    sum += A[y*INNER+k] * B[k*W+x];
                }

                C[y*W+x] = sum;
            }
        }
    }

    void multiply2(float const * const ptrA, float const * const ptrB, float * const ptrC)
    {
        using namespace concurrency;
        extent<2> a_size(H, INNER);
        extent<2> b_size(INNER, W);
        extent<2> c_size(H, W);

        array_view<const float, 2> A(a_size, ptrA);
        array_view<const float, 2> B(b_size, ptrB);
        array_view<float, 2> C(c_size, ptrC);

        C.discard_data();

        parallel_for_each(C.extent, [=](index<2> idx) restrict(amp)
        {
            const int y = idx[0]; 
            const int x = idx[1];
            float sum = 0;
            for (int k = 0; k < INNER; ++k)
                sum += A(y, k) * B(k, x);
            C[idx] = sum;
        });
    }
}

int main() try
{
    using namespace noxitu;

    std::vector<concurrency::accelerator> accs = concurrency::accelerator::get_all();

    std::wcout << "Found " << accs.size() << " accelerators.\n";
    for (auto &acc : accs)
    {
        std::wcout << "Accelerator:\n"
                  << "  Description = " << acc.description << '\n'
                  << "  Device path = " << acc.device_path << '\n';
    }

    //concurrency::accelerator::set_default(accs[0].device_path);

    std::wcout << std::endl;


    std::vector<float> A(H*INNER); 
    std::vector<float> B(INNER*W);
    std::vector<float> C(H*W);

    const int LOOPS = 10;
    std::vector<float> durations;
    durations.reserve(LOOPS);

    for (int i = 0; i < LOOPS; ++i)
    {
        std::generate(A.begin(), A.end(), []() -> float { return 2.0f * rand() / RAND_MAX - 1.0f; });
        std::generate(B.begin(), B.end(), []() -> float { return 2.0f * rand() / RAND_MAX - 1.0f; });
            
        auto tp1 = now();

        //multiply1(A.data(), B.data(), C.data());
        multiply2(A.data(), B.data(), C.data());

        auto tp2 = now();

        std::cout << std::accumulate(C.begin(), C.end(), 0.0f) << std::endl;

        durations.push_back(std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(tp2-tp1).count());
    }

    const float min = *std::min_element(durations.begin(), durations.end());

    const float mean = std::accumulate(durations.begin(), durations.end(), 0.0f) / durations.size();

    const float stddev = std::sqrt(std::accumulate(durations.begin(), durations.end(), 0.0f, [mean](float value, float duration)
    {
        const float deviation = duration - mean;
        return value + deviation*deviation;
    }) / durations.size());

    std::cout << "N                  = " << durations.size() << std::endl;
    std::cout << "Min                = " << min << std::endl;
    std::cout << "Mean               = " << mean << std::endl;
    std::cout << "Standard Deviation = " << stddev << std::endl;
}
catch(std::exception &ex)
{
    std::cerr << "ERROR: main() failed with exception " << typeid(ex).name() << ": \"" << ex.what() << "\"." << std::endl;
    return EXIT_FAILURE;
}