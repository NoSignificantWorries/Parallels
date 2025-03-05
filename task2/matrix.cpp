#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <fstream>

void write_lines(const std::pair<int, int> &serial, const std::pair<std::vector<int>, std::vector<int>> &results, std::ofstream &out);

std::pair<int, int> run_serial();
std::pair<std::vector<int>, std::vector<int>> run_parallel_only_mul();
std::pair<std::vector<int>, std::vector<int>> run_parallel_all();

void generate_serial(std::vector<double> &A, std::vector<double> &b, std::vector<double> &c, int m, int n);
void generate_parallel(std::vector<double> &A, std::vector<double> &b, std::vector<double> &c, int m, int n, int NumThreads);

void serial_mul(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &c, int m, int n);
void parallel_mul(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &c, int m, int n, int NumThreads);

int main()
{
    std::cout << "Start serial\n";
    std::pair<int, int> res_serial = run_serial();
    std::cout << "Serial finished\n";
    
    std::cout << "Start parallel (only multiplication)\n";
    std::pair<std::vector<int>, std::vector<int>> res_parallel_only_mul = run_parallel_only_mul();
    std::cout << "Parallel finished\n";
    
    std::cout << "Start parallel (all)\n";
    std::pair<std::vector<int>, std::vector<int>> res_parallel_all = run_parallel_all();
    std::cout << "Parallel finished\n";

    std::ofstream outputFile("matrix_result.txt");
    
    write_lines(res_serial, res_parallel_only_mul, outputFile);
    write_lines(res_serial, res_parallel_all, outputFile);
    
    outputFile.close();

    return 0;
}

void write_lines(const std::pair<int, int> &serial, const std::pair<std::vector<int>, std::vector<int>> &results, std::ofstream &out)
{
    for (int i = 0; i < results.first.size(); i++)
    {
        out << " " << results.first[i];
    }
    out << "\n";
    for (int i = 0; i < results.first.size(); i++)
    {
        out << " " << (double)serial.first / results.first[i];
    }
    out << "\n";
    for (int i = 0; i < results.first.size(); i++)
    {
        out << " " << results.second[i];
    }
    out << "\n";
    for (int i = 0; i < results.first.size(); i++)
    {
        out << " " << (double)serial.second / results.second[i];
    }
    out << "\n";
}

std::pair<int, int> run_serial()
{
    int N = 20000;
    int M = 40000;

    std::vector<double> A(N * N);
    std::vector<double> b(N);
    std::vector<double> c(N);

    std::vector<double> D(M * M);
    std::vector<double> e(M);
    std::vector<double> f(M);

    generate_serial(A, b, c, N, N);
    generate_serial(D, e, f, M, M);

    auto start = std::chrono::high_resolution_clock::now();
    serial_mul(A, b, c, N, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_20k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    serial_mul(D, e, f, M, M);
    end = std::chrono::high_resolution_clock::now();

    auto duration_40k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::pair<int, int> result{duration_20k.count(), duration_40k.count()};

    return result;
}

std::pair<std::vector<int>, std::vector<int>> run_parallel_only_mul()
{
    std::vector<int> idx{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> res_20k;
    std::vector<int> res_40k;
    for (int i = 0; i < idx.size(); i++)
    {
        int N = 20000;
        int M = 40000;

        std::vector<double> A(N * N);
        std::vector<double> b(N);
        std::vector<double> c(N);

        std::vector<double> D(M * M);
        std::vector<double> e(M);
        std::vector<double> f(M);

        generate_serial(A, b, c, N, N);
        generate_serial(D, e, f, M, M);

        auto start = std::chrono::high_resolution_clock::now();
        parallel_mul(A, b, c, N, N, idx[i]);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_20k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        res_20k.push_back(duration_20k.count());

        start = std::chrono::high_resolution_clock::now();
        parallel_mul(D, e, f, M, M, idx[i]);
        end = std::chrono::high_resolution_clock::now();

        auto duration_40k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        res_40k.push_back(duration_20k.count());
    }
    std::pair<std::vector<int>, std::vector<int>> result{res_20k, res_40k};
    return result;
}

std::pair<std::vector<int>, std::vector<int>> run_parallel_all()
{
    std::vector<int> idx{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> res_20k;
    std::vector<int> res_40k;
    for (int i = 0; i < idx.size(); i++)
    {
        int N = 20000;
        int M = 40000;

        std::vector<double> A(N * N);
        std::vector<double> b(N);
        std::vector<double> c(N);

        std::vector<double> D(M * M);
        std::vector<double> e(M);
        std::vector<double> f(M);

        generate_parallel(A, b, c, N, N, idx[i]);
        generate_parallel(D, e, f, M, M, idx[i]);

        auto start = std::chrono::high_resolution_clock::now();
        parallel_mul(A, b, c, N, N, idx[i]);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_20k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        res_20k.push_back(duration_20k.count());

        start = std::chrono::high_resolution_clock::now();
        parallel_mul(D, e, f, M, M, idx[i]);
        end = std::chrono::high_resolution_clock::now();

        auto duration_40k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        res_40k.push_back(duration_20k.count());
    }
    std::pair<std::vector<int>, std::vector<int>> result{res_20k, res_40k};
    return result;
}

void generate_serial(std::vector<double> &A, std::vector<double> &b, std::vector<double> &c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = j;
    }
}

void serial_mul(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            c[i] += A[i * n + j] * b[j];
        }
    }
}

void generate_parallel(std::vector<double> &A, std::vector<double> &b, std::vector<double> &c, int m, int n, int NumThreads)
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
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = i + j;
            }
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = j;
    }
}

void parallel_mul(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &c, int m, int n, int NumThreads)
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
