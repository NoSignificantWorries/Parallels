#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>

#define NSTEPS 40000000

typedef double myfunc(double);

void run_test(myfunc *func, double a, double b, std::ofstream &out);

int run_serial(myfunc *func, double a, double b);
std::vector<int> run_parallel(myfunc *func, double a, double b);

double integration_serial(myfunc *func, double a, double b);
double integration_parallel(myfunc *func, double a, double b, int NumThreads);

int main() {
    std::ofstream outputFile("result_integration.txt");
    
    run_test(std::sin, 0, 1000, outputFile);
    run_test(std::exp, 0, 1000, outputFile);
    run_test(std::sqrt, 0, 1000, outputFile);
    
    outputFile.close();

    return 0;
}

void run_test(myfunc *func, double a, double b, std::ofstream &out)
{
    int serial_time = run_serial(func, a, b);
    std::vector<int> parallel_time = run_parallel(func, a, b);
    
    for (int i = 0; i < parallel_time.size(); i++)
    {
        out << " " << parallel_time[i];
    }
    out << "\n";
    for (int i = 0; i < parallel_time.size(); i++)
    {
        out << " " << (double)serial_time / parallel_time[i];
    }
    out << "\n";
}

int run_serial(myfunc *func, double a, double b)
{
    auto start = std::chrono::high_resolution_clock::now();
    integration_serial(func, a, b);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

std::vector<int> run_parallel(myfunc *func, double a, double b)
{
    std::vector<int> idx{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> result;
    for (int i = 0; i < idx.size(); i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        integration_parallel(func, a, b, idx[i]);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        result.push_back(duration.count());
    }
    return result;
}

double integration_serial(myfunc *func, double a, double b)
{
    double step = (double)(b - a) / NSTEPS;

    double sum = 0.0;
    for (int i = 0; i < NSTEPS; i++)
    {
        double x = a + (i + 0.5) * step;
        sum += func(x) * step;
    }
    
    return sum;
}

double integration_parallel(myfunc *func, double a, double b, int NumThreads)
{
    double step = (double)(b - a) / NSTEPS;

    double sum = 0.0;
    
    #pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        for (int i = current_thread_number; i < NSTEPS; i += threads_count)
        {
            double x = a + (i + 0.5) * step;
            #pragma omp atomic
            sum += func(x) * step;
        }
    }

    return sum;
}
