#include <iostream>
#include <cstdlib>
#include <vector>

int main()
{
    const int N = 300000;
    std::vector<float> input(N);
    std::vector<float> weights(N);
    std::vector<float> output(N);

    for (int i = 0; i < N; ++i)
    {
        input[i] = 1.0f*rand()/RAND_MAX;
    }

    for (int i = 0; i < N; ++i)
    {
        weights[i] = 1.0f*rand()/RAND_MAX;
    }

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        float result = 0.0;
        for (auto w : weights)
        {
            result += input[i] * w;
        }

        output[i] = result;
    }

    for (int i = 0; i < 10; ++i)
    {
        std::cout << i << " = " << output[i] << std::endl;
    }

    return 0;
}