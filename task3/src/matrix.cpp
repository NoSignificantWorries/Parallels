#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <fstream>

#define ITERS 10

using uniqptr_array = std::unique_ptr<double[]>;

void generate_serial(uniqptr_array &A, uniqptr_array &b, uniqptr_array &c, int n);
void serial_mul(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int n);

void generate_part(uniqptr_array &A, uniqptr_array &c, int begin, int end, int n);
void generate_all(uniqptr_array &A, uniqptr_array &b, uniqptr_array &c, int n, int NumThreads);

void mul_part(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int begin, int end, int n);
void parallel_mul(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int n, int NumThreads);

int main()
{
    std::vector<int> threads_list{1, 2, 4, 7, 8, 16, 20, 40};
    
    auto file_deleter = [](std::fstream* fp) {
        if (fp && fp->is_open()) fp->close();
        delete fp;
    };

    std::shared_ptr<std::fstream> file_ptr(new std::fstream("matrix.txt", std::ios::out), file_deleter);

    for (int threads: threads_list) *file_ptr << threads << " ";
    *file_ptr << "\n";
    
    std::vector<int> sizes = { 20000, 40000 };
    for (auto n : sizes)
    {
        if (n == 20000) *file_ptr << "M=20k";
        else *file_ptr << "M=40k";

        std::cout << "Start " << n << "\nStart serial...\n";
        double res_serial = 0.0;
        for (int i = 0; i < ITERS; i++)
        {
            uniqptr_array A1(std::make_unique<double[]>(n * n));
            uniqptr_array b1(std::make_unique<double[]>(n));
            uniqptr_array c1(std::make_unique<double[]>(n));

            generate_serial(A1, b1, c1, n);

            const auto start{std::chrono::steady_clock::now()};
            serial_mul(A1, b1, c1, n);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time_elapsed(end - start);
            res_serial += time_elapsed.count();
        }
        res_serial /= ITERS;
        
        double res;
        for (auto threads_count: threads_list)
        {
            res = 0.0;
            std::cout << "Start parallel (" << threads_count << ")...\n";
            for (int i = 0; i < ITERS; i++)
            {
                uniqptr_array A2(std::make_unique<double[]>(n * n));
                uniqptr_array b2(std::make_unique<double[]>(n));
                uniqptr_array c2(std::make_unique<double[]>(n));

                generate_all(A2, b2, c2, n, threads_count);

                const auto start{std::chrono::steady_clock::now()};
                parallel_mul(A2, b2, c2, n, threads_count);
                const auto end{std::chrono::steady_clock::now()};
                const std::chrono::duration<double> time_elapsed(end - start);
                res += time_elapsed.count();
            }
            res /= ITERS;
            *file_ptr << " " << res << "," << res_serial / res;
        }
        *file_ptr << "\n";
    }

    return 0;
}

void generate_serial(uniqptr_array &A, uniqptr_array &b, uniqptr_array &c, int n)
{
    for (int i = 0; i < n; i++)
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

void serial_mul(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i] += A[i * n + j] * b[j];
        }
    }
}

void generate_part(uniqptr_array &A, uniqptr_array &c, int begin, int end, int n)
{
    for (int i = begin; i < end; i++)
    {
        c[i] = static_cast<double>(0.0);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>(i + j);
        }
    }
}

void generate_all(uniqptr_array &A, uniqptr_array &b, uniqptr_array &c, int n, int NumThreads)
{
    std::vector<std::thread> threads;
    int step = n / NumThreads;
    int begin = 0;
    int end = 0;
    for (int i = 0; i < NumThreads; i++)
    {
        begin = i * step;
        if (i == NumThreads - 1) end = n;
        else end = begin + step;
        threads.emplace_back(generate_part, std::ref(A), std::ref(c), begin, end, n);
    }
    for (int i = 0; i < n; i++) b[i] = static_cast<double>(i);
    
    for (auto &thread: threads)
    {
        thread.join();
    }
}

void mul_part(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int begin, int end, int n)
{
    double tmp = 0.0;
    for (int i = begin; i < end; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * b[j];            
        }
        c[i] = tmp;
    }
}

void parallel_mul(const uniqptr_array &A, const uniqptr_array &b, uniqptr_array &c, int n, int NumThreads)
{
    std::vector<std::thread> threads;
    int step = n / NumThreads;
    int begin = 0;
    int end = 0;
    for (int i = 0; i < NumThreads; i++)
    {
        begin = i * step;
        if (i == NumThreads - 1) end = n;
        else end = begin + step;
        threads.emplace_back(mul_part, std::ref(A), std::ref(b), std::ref(c), begin, end, n);
    }
    
    for (auto &thread: threads)
    {
        thread.join();
    }
}
