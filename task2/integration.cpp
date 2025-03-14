#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>

#define NSTEPS 40000000

using int_vec = std::vector<int>;
typedef double (*func)(double);

double run_test(double (*integration)(func, int, int, int), func f, int a, int b, int NumThreads);

double integration_serial(func f, int a, int b, int _);
double integration_parallel(func f, int a, int b, int NumThreads);

int main()
{
    std::ofstream outputFile("data/integration_result.txt");

    int_vec threads_list{1, 2, 4, 7, 8, 16, 20, 40};

    for (int threads : threads_list)
    {
        outputFile << threads << " ";
    }
    outputFile << "\n";

    std::vector<std::pair<std::string, func>> functions{{"Exp", std::exp}, {"Sin", std::sin}, {"Cos", std::cos}};

    for (std::pair<std::string, func> elem : functions)
    {
        outputFile << elem.first;

        std::cout << "Starting serial test.\n";
        double res_serial = 0.0;
        for (int i = 0; i < 10; i++)
        {
            res_serial += run_test(integration_serial, elem.second, 0, 1000, 0);
        }
        res_serial /= 10;
        std::cout << "Serial test finished.\n";

        double res = 0.0;
        for (int threads : threads_list)
        {
            std::cout << "Starting parallel test (" << threads << ").\n";
            for (int i = 0; i < 10; i++)
            {
                res += run_test(integration_parallel, elem.second, 0, 1000, threads);
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

double run_test(double (*integration)(func, int, int, int), func f, int a, int b, int NumThreads)
{
    const auto start{std::chrono::steady_clock::now()};
    integration(f, a, b, NumThreads);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> time_elapsed(end - start);

    return time_elapsed.count();
}

double integration_serial(func f, int a, int b, int _)
{
    double step = (double)(b - a) / NSTEPS;

    double sum = 0.0;
    for (int i = 0; i < NSTEPS; i++)
    {
        double x = a + (i + 0.5) * step;
        sum += f(x) * step;
    }
    
    return sum;
}

double integration_parallel(func f, int a, int b, int NumThreads)
{
    double step = (double)(b - a) / NSTEPS;

    double sum = 0.0;
    
    #pragma omp parallel num_threads(NumThreads)
    {
        int current_thread_number = omp_get_thread_num();
        int threads_count = omp_get_num_threads();

        double local_sum = 0.0;
        for (int i = current_thread_number; i < NSTEPS; i += threads_count)
        {
            double x = a + (i + 0.5) * step;
            local_sum += f(x) * step;
        }
        #pragma omp atomic
        sum += local_sum;
    }

    return sum;
}
