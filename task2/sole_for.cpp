#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <memory>

#define tau 0.01
#define epsilon 0.00001
#define N 1000

using int_vec = std::vector<int>;
using vec = std::unique_ptr<double[]>;

void generate_serial(vec &A, vec &b, vec &x, vec &x_0, int _);
void generate_parallel_threads(vec &A, vec &b, vec &x, vec &x_0, int NumThreads);
void generate_parallel_for(vec &A, vec &b, vec &x, vec &x_0, int _);

void step_serial(const vec &A, const vec &b, vec &x_0, vec &x, int _);
void step_parallel_threads(const vec &A, const vec &b, vec &x_0, vec &x, int NumThreads);
void step_parallel_for(const vec &A, const vec &b, vec &x_0, vec &x, int _);

int main()
{
    std::ofstream outputFile("data/sole_result.txt");
    
    
    outputFile.close();

    return 0;
}

void generate_serial(vec &A, vec &b, vec &x, vec &x_0, int _)
{
    for (int i = 0; i < N; i++)
    {
        b[i] = static_cast<double>(N);
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void generate_parallel_threads(vec &A, vec &b, vec &x, vec &x_0, int NumThreads)
{
#pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = N / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (N - 1) : (lower_bound + items_per_thread - 1);

        for (int i = lower_bound; i <= upper_bound; i++)
        {
            b[i] = static_cast<double>(N);
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = static_cast<double>((i == j) ? 2 : 1);
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void generate_parallel_for(vec &A, vec &b, vec &x, vec &x_0, int _)
{
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
        b[i] = static_cast<double>(N);
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }

    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void step_serial(const vec &A, const vec &b, const vec &x_0, vec &x, int _)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            x[i] = x_0[i] - tau * (A[i * N + j] * x_0[j] - b[j]);
        }
    }
}

void step_parallel_threads(const vec &A, const vec &b, vec &x_0, vec &x, int NumThreads)
{
}

void step_parallel_for(const vec &A, const vec &b, vec &x_0, vec &x, int _)
{
}
