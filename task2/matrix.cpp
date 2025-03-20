#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <omp.h>
#include <fstream>

using int_vec = std::vector<int>;
using vec = std::unique_ptr<double[]>;

double run_test(void (*init)(vec &, vec &, vec &, int, int, int),
                void (*mult)(const vec &, const vec &, vec &, int, int, int), int n, int threads);

void generate_serial(vec &A, vec &b, vec &c, int m, int n, int _);
void generate_parallel(vec &A, vec &b, vec &c, int m, int n, int NumThreads);

void serial_mul(const vec &A, const vec &b, vec &c, int m, int n, int _);
void parallel_mul(const vec &A, const vec &b, vec &c, int m, int n, int NumThreads);

int main()
{
    std::ofstream outputFile("data/matrix_result1.txt");

    int_vec sizes{20000, 40000};
    int_vec threads_list{1, 2, 4, 7, 8, 16, 20, 40};
    
    for (int threads: threads_list)
    {
        outputFile << threads << " ";
    }
    outputFile << "\n";

    for (int N : sizes)
    {
        if (N == 20000) outputFile << "M=20k";
        else outputFile << "M=40k";

        std::cout << "Starting serial test.\n";
        double res_serial = 0.0;
        for (int i = 0; i < 10; i++)
        {
            res_serial += run_test(generate_serial, serial_mul, N, 0);
        }
        res_serial /= 10;
        std::cout << "Serial test finished.\n";

        double res = 0.0;
        for (int threads : threads_list)
        {
            std::cout << "Starting parallel test (" << threads << ").\n";
            for (int i = 0; i < 10; i++)
            {
                res += run_test(generate_parallel, parallel_mul, N, threads);
            }
            res /= 10;
            std::cout << res_serial << " " << res << " " << res_serial / res << "\n";
            outputFile << " " << res << "," << res_serial / res;
            res = 0.0;
        }
        std::cout << "Parallel test finished.\n";
        
        outputFile << "\n";
    }

    outputFile.close();

    return 0;
}

double run_test(void (*init)(vec &, vec &, vec &, int, int, int),
                void (*mult)(const vec &, const vec &, vec &, int, int, int), int n, int threads)
{
    std::unique_ptr<double[]> A(new double[n * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[n]);

    init(A, b, c, n, n, threads);

    const auto start{ std::chrono::steady_clock::now() };
    mult(A, b, c, n, n, threads);
    const auto end{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> time_elapsed(end - start);

    return time_elapsed.count();
}

void generate_serial(vec &A, vec &b, vec &c, int m, int n, int _)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = static_cast<double>(0.0);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>(i + j);
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = static_cast<double>(j);
    }
}

void generate_parallel(vec &A, vec &b, vec &c, int m, int n, int NumThreads)
{
#pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = m / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (m - 1) : (lower_bound + items_per_thread - 1);

        for (int i = lower_bound; i <= upper_bound; i++)
        {
            c[i] = static_cast<double>(0.0);
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = static_cast<double>(i + j);
            }
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = static_cast<double>(j);
    }
}

void serial_mul(const vec &A, const vec &b, vec &c, int m, int n, int _)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i] += A[i * n + j] * b[j];
        }
    }
}

void parallel_mul(const vec &A, const vec &b, vec &c, int m, int n, int NumThreads)
{
#pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        int items_per_thread = m / threads_count;
        int lower_bound = current_thread_number * items_per_thread;
        int upper_bound = (current_thread_number == threads_count - 1) ? (m - 1) : (lower_bound + items_per_thread - 1);

        double tmp;
        for (int i = lower_bound; i <= upper_bound; i++)
        {
            tmp = 0.0;
            for (int j = 0; j < n; j++)
            {
                tmp += A[i * n + j] * b[j];
            }
            c[i] = tmp;
        }
    }
}
