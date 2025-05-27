#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <boost/program_options.hpp>

#define OFFSET(row, col, m) (((row) * (m)) + (col))

namespace po = boost::program_options;


void initialize_boundary(double* grid, int N)
{
    double top_left = 10.0;
    double top_right = 20.0;
    double bottom_right = 30.0;
    double bottom_left = 20.0;

    for (int j = 0; j < N; ++j) {
        double t = static_cast<double>(j) / (N - 1);
        grid[j] = top_left + t * (top_right - top_left);
    }

    for (int j = 0; j < N; ++j) {
        double t = static_cast<double>(j) / (N - 1);
        grid[OFFSET(N - 1, j, N)] = bottom_left + t * (bottom_right - bottom_left);
    }

    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        grid[OFFSET(i, 0, N)] = top_left + t * (bottom_left - top_left);
    }

    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        grid[OFFSET(i, N - 1, N)] = top_right + t * (bottom_right - top_right);
    }
}


void convolve_steps(double* u_1, double* u_2, int N, int steps)
{
    for (int k = 0; k < steps; ++k)
    {
        for (int i = 1; i < N - 1; ++i)
        {
            for (int j = 1; j < N - 1; ++j)
            {
                u_2[OFFSET(i, j, N)] = 0.25 * (u_1[OFFSET(i + 1, j, N)] + 
                                               u_1[OFFSET(i - 1, j, N)] +
                                               u_1[OFFSET(i, j + 1, N)] + 
                                               u_1[OFFSET(i, j - 1, N)]);
            }
        }
        std::swap(u_1, u_2);
    }
}

double error_step(double* u_1, double* u_2, int N, int steps)
{
    convolve_steps(u_1, u_2, N, steps);

    double error = 0.0;
    for (int i = 1; i < N - 1; ++i)
    {
        for (int j = 1; j < N - 1; ++j)
        {
            error = fmax(error, fabs(u_1[OFFSET(i, j, N)] - u_2[OFFSET(i, j, N)]));
        }
    }
    return error;
}


int main(int argc, char* argv[]) {
    int N;
    double tolerance;
    int max_iters;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("size,s", po::value<int>(&N)->default_value(128), "grid size (N x N)")
        ("tolerance,t", po::value<double>(&tolerance)->default_value(1e-6), "tolerance for convergence")
        ("max_iters,m", po::value<int>(&max_iters)->default_value(1000000), "maximum number of iterations")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (N < 2) {
        std::cerr << "Grid size must be at least 2\n";
        return -1;
    }

    std::unique_ptr<double[]> u_1_unique(std::make_unique<double[]>(N * N));
    std::unique_ptr<double[]> u_2_unique(std::make_unique<double[]>(N * N));

    double* u_1 = u_1_unique.get();
    double* u_2 = u_2_unique.get();

    initialize_boundary(u_1, N);
    initialize_boundary(u_2, N);
    
    int iter = 0;
    double max_error = 1.0 + tolerance;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    while (max_error > tolerance && iter < max_iters)
    {
        max_error = error_step(u_1, u_2, N, 1000);

        iter += 1000;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    std::cout << "Total iterations: " << iter << "\n";
    std::cout << "Error: " << max_error << "\n";
    std::cout << "Time: " << duration.count() << "\n";
    
    if (N >= 10 && N <= 20)
    {
       for (int i = 0; i < N; i++)
       {
            for (int j = 0; j < N; j++)
            {
                std::cout << std::fixed << std::setprecision(3) << std::setw(6) << std::setfill(' ') << u_1[OFFSET(i, j, N)] << " ";
            }
            std::cout << "\n";
       } 
    }

    return 0;
}
