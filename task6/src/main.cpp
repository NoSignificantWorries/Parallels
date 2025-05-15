#include <iostream>
#include <vector>
#include <cmath>
#include <boost/program_options.hpp>

#define OFFSET(row, col, m) (((row) * (m)) + (col))

namespace po = boost::program_options;

void algo_step(const std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &B, int n);

int main(int argc, char *argv[])
{
    po::options_description desc("Options for task6");
    desc.add_options()
    ("size", po::value<int>()->default_value(128), "Matrix size (N x N)")
    ("tolerance", po::value<double>()->default_value(1e-6), "Tolerance")
    ("max_iterations", po::value<int>()->default_value(1000000), "Maximum iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    int N = vm["size"].as<int>();
    double tolerance = vm["tolerance"].as<double>();
    int max_iterations = vm["max_iterations"].as<int>();

    std::cout << "Matrix size" << N << "x" << N << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "Maximum iterations: " << max_iterations << std::endl;

    std::unique_ptr<double[]> u_1(std::make_unique<double[]>(N * N));
    std::unique_ptr<double[]> u_2(std::make_unique<double[]>(N * N));

    double corner1 = 10.0;
    double corner2 = 20.0;
    double corner3 = 30.0;
    double corner4 = 20.0;

    for (int i = 0; i < N; ++i)
    {
        u_1[OFFSET(0, i, N)] = corner1 + (corner2 - corner1) * i / (N - 1.0);
        u_1[OFFSET(N - 1, i, N)] = corner4 + (corner3 - corner4) * i / (N - 1.0);
        u_1[OFFSET(i, 0, N)] = corner1 + (corner4 - corner1) * i / (N - 1.0);
        u_1[OFFSET(i, N - 1, N)] = corner2 + (corner3 - corner2) * i / (N - 1.0);
    }

    int iteration = 0;
    double error = 1.0;

    while (iteration < max_iterations && error > tolerance)
    {
        if (iteration % 2 == 0)
            algo_step(u_1, u_2, N);
        else
            algo_step(u_2, u_1, N);

        error = 0.0;
        for (int i = 1; i < N - 1; ++i)
        {
            for (int j = 1; j < N - 1; ++j)
            {
                error += std::abs(u_2[OFFSET(i, j, N)] - u_1[OFFSET(i, j, N)]);
            }
        }
        error /= ((N - 2) * (N - 2));

        iteration++;
    }

    std::cout << "Total iterations: " << iteration << std::endl;
    std::cout << "Final error: " << error << std::endl;
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            std::cout << u_1[OFFSET(i, j, N)] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

void algo_step(const std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &B, int n)
{
    for (int i = 0; i < n; ++i)
    {
        B[OFFSET(0, i, n)] = A[OFFSET(0, i, n)];
        B[OFFSET(n - 1, i, n)] = A[OFFSET(n - 1, i, n)];
        B[OFFSET(i, 0, n)] = A[OFFSET(i, 0, n)];
        B[OFFSET(i, n - 1, n)] = A[OFFSET(i, n - 1, n)];
    }

    for (int i = 1; i < n - 1; ++i)
    {
        for (int j = 1; j < n - 1; ++j)
        {
            B[OFFSET(i, j, n)] = 0.25 * (A[OFFSET(i + 1, j, n)] + A[OFFSET(i - 1, j, n)] + A[OFFSET(i, j + 1, n)] + A[OFFSET(i, j - 1, n)]);
        }
    }
}
