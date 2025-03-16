#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <memory>

#include "for.hpp"

#define TAU 0.01
#define EPSILON 0.00001

using int_vec = std::vector<int>;

void generate_serial(vec &A, vec &b, vec &x, vec &x_0, int n, int _);
void generate_parallel_threads(vec &A, vec &b, vec &x, vec &x_0, int n, int NumThreads);

void step_serial(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int _);
void step_parallel_threads(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int NumThreads);

int main()
{
    std::ofstream outputFile("data/sole_result.txt");
    

    
    outputFile.close();

    return 0;
}

void generate_serial(vec &A, vec &b, vec &x, vec &x_0, int n, int _)
{
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>(n + 1);
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
}

void generate_parallel_threads(vec &A, vec &b, vec &x, vec &x_0, int n, int NumThreads)
{
#pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = n / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (n - 1) : (lower_bound + items_per_thread - 1);

        for (int i = lower_bound; i <= upper_bound; i++)
        {
            b[i] = static_cast<double>(n + 1);
            x[i] = static_cast<double>(0.0);
            x_0[i] = static_cast<double>(0.0);
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
            }
        }
    }
}

void step_serial(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += tau * (A[i * n + j] * x_0[j] - b[j]);
        }
        x[i] = x_0[i] - tmp;
    }
}

void step_parallel_threads(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int NumThreads)
{
    #pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = n / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (n - 1) : (lower_bound + items_per_thread - 1);

        for (int i = lower_bound; i <= upper_bound; i++)
        {
            double tmp = 0.0;
            for (int j = 0; j < n; j++)
            {
                tmp += tau * (A[i * n + j] * x_0[j] - b[j]);
            }
            x[i] = x_0[i] - tmp;
        }
    }
}
