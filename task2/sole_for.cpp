#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <memory>

#include "lib/for.hpp"

#define N 1900
#define ITER 10
#define TAU 0.001
#define EPSILON 0.00001

using int_vec = std::vector<int>;
using double_vec = std::vector<double>;

double run(void (*generate)(vec &, vec &, vec &, vec &, int, int),
           void (*step)(const vec &, const vec &, const vec &, vec &, double, int, int),
           double tau, int n, int NumThreads);

void generate_serial(vec &A, vec &b, vec &x, vec &x_0, int n, int _);
void generate_parallel_threads(vec &A, vec &b, vec &x, vec &x_0, int n, int NumThreads);

void step_serial(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _);
void step_parallel_threads(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumThreads);

double length(const vec &v, int n);
double mult_length(const vec &A, const vec &b, const vec &x, int n);

int main()
{
    std::ofstream outputFile("data/sole_result.txt");
    
    int_vec threads_list{1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << "Start serial...\n";
    double res_serial = 0.0;
    for (int i = 0; i < ITER; i++) res_serial += run(generate_serial, step_serial, TAU, N, 0);
    res_serial /= ITER;


    double res_threads = 0.0;
    double res_for_auto = 0.0;
    double res_for_static = 0.0;
    double res_dynamic = 0.0;
    double res_for_guided = 0.0;
    double_vec results_threads(threads_list.size());
    double_vec results_auto(threads_list.size());
    double_vec results_static(threads_list.size());
    double_vec results_dynamic(threads_list.size());
    double_vec results_guided(threads_list.size());
    for (int j = 0; j < threads_list.size(); j++)
    {
        std::cout << "Threads: (" << threads_list[j] << ")\n";
        std::cout << "Start threads...\n";
        for (int i = 0; i < ITER; i++)
        {
            res_threads += run(generate_parallel_threads, step_parallel_threads, TAU, N, threads_list[j]);
        }
        results_threads[j] = res_threads / ITER;
        res_threads = 0.0;

        std::cout << "Start for auto...\n";
        for (int i = 0; i < ITER; i++)
        {
            res_for_auto += run(generate_serial, step_parallel_for_auto, TAU, N, threads_list[j]);
        }
        results_auto[j] = res_for_auto / ITER;
        res_for_auto = 0.0;

        std::cout << "Start for static...\n";
        for (int i = 0; i < ITER; i++)
        {
            res_for_static += run(generate_serial, step_parallel_for_static, TAU, N, threads_list[j]);
        }
        results_static[j] = res_for_static / ITER;
        res_for_static = 0.0;

        std::cout << "Start dynamic...\n";
        for (int i = 0; i < ITER; i++)
        {
            res_dynamic += run(generate_serial, step_parallel_for_dynamic, TAU, N, threads_list[j]);
        }
        results_dynamic[j] = res_dynamic / ITER;
        res_dynamic = 0.0;

        std::cout << "Start for guided...\n";
        for (int i = 0; i < ITER; i++)
        {
            res_for_guided += run(generate_serial, step_parallel_for_guided, TAU, N, threads_list[j]);
        }
        results_guided[j] = res_for_guided / ITER;
        res_for_guided = 0.0;
    }
    for (int threads: threads_list) outputFile << threads << " ";
    outputFile << "\n";

    outputFile << "For_Auto";
    for (int i = 0; i < threads_list.size(); i++) outputFile << " " << results_auto[i] << "," << res_serial / results_auto[i];
    outputFile << "\n";

    outputFile << "For_Guided";
    for (int i = 0; i < threads_list.size(); i++) outputFile << " " << results_guided[i] << "," << res_serial / results_guided[i];
    outputFile << "\n";

    outputFile << "For_Static";
    for (int i = 0; i < threads_list.size(); i++) outputFile << " " << results_static[i] << "," << res_serial / results_static[i];
    outputFile << "\n";

    outputFile << "For_Dynamic";
    for (int i = 0; i < threads_list.size(); i++) outputFile << " " << results_dynamic[i] << "," << res_serial / results_dynamic[i];
    outputFile << "\n";

    outputFile << "Threads";
    for (int i = 0; i < threads_list.size(); i++) outputFile << " " << results_threads[i] << "," << res_serial / results_threads[i];
    outputFile << "\n";

    outputFile.close();

    return 0;
}

double run(void (*generate)(vec &, vec &, vec &, vec &, int, int),
           void (*step)(const vec &, const vec &, const vec &, vec &, double, int, int),
           double tau, int n, int NumThreads)
{
    vec A(new double[n * n]);
    vec b(new double[n]);
    vec x_0(new double[n]);
    vec x(new double[n]);
    
    generate(A, b, x, x_0, n, NumThreads);
    double b_length = length(b, n);
    double m_length = mult_length(A, b, x, n);
    double g = m_length / b_length;
    double time = 0.0;

    int i = 0; 
    while (g >= EPSILON)
    {
        if (i % 2 == 0) 
        {
            const auto start{std::chrono::steady_clock::now()};
            step(A, b, x_0, x, tau, n, NumThreads);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time_elapsed(end - start);
            time += time_elapsed.count();
            m_length = mult_length(A, b, x, n);
        }
        else
        {
            const auto start{std::chrono::steady_clock::now()};
            step(A, b, x, x_0, tau, n, NumThreads);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time_elapsed(end - start);
            time += time_elapsed.count();
            m_length = mult_length(A, b, x_0, n);
        }
        g = m_length / b_length;
        i++;
    }
    
    return time;
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
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void step_serial(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x_0[j];
        }
        x[i] = x_0[i] - tau * (tmp - b[i]);
    }
}

void step_parallel_threads(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumThreads)
{
    #pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = n / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (n - 1) : (lower_bound + items_per_thread - 1);

        double tmp;
        for (int i = lower_bound; i <= upper_bound; i++)
        {
            tmp = 0.0;
            for (int j = 0; j < n; j++)
            {
                tmp += A[i * n + j] * x_0[j];
            }
            x[i] = x_0[i] - tau * (tmp - b[i]);
        }
    }
}

double length(const vec &v, int n)
{
    double res = 0.0;
    for (int i = 0; i < n; i++)
    {
        res += v[i] * v[i];
    }
    res = std::sqrt(res);
    return res;
}

double mult_length(const vec &A, const vec &b, const vec &x, int n)
{
    double res = 0.0;
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x[j];
        }
        tmp -= b[i];
        res += tmp * tmp;
    }
    res = std::sqrt(res);
    return res;
}
