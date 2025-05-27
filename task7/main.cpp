#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>
#include <iomanip>
#include <openacc.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#define OFFSET(row, col, m) (((row) * (m)) + (col))

namespace po = boost::program_options;

void write_to_file(const double* arr, int size, std::string filename)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error to open file: " << filename << "\n";
        return;
    }

   for (int i = 0; i < size; i++)
   {
        for (int j = 0; j < size; j++)
        {
            outputFile << std::fixed << std::setprecision(3) << std::setw(6) << std::setfill(' ') << arr[OFFSET(i, j, size)] << " ";
        }
        outputFile << "\n";
   } 

    outputFile.close();
}

void convolve_steps(double *restrict u_1, double *restrict u_2, int size, int steps)
{
    for (int k = 0; k < steps; ++k)
    {
        #pragma acc kernels loop independent collapse(2) present(u_1, u_2) async
        for (int i = 1; i < size - 1; ++i)
        {
            for (int j = 1; j < size - 1; ++j)
            {
                u_2[OFFSET(i, j, size)] = 0.25 * (u_1[OFFSET(i + 1, j, size)] + 
                                              u_1[OFFSET(i - 1, j, size)] + 
                                              u_1[OFFSET(i, j + 1, size)] + 
                                              u_1[OFFSET(i, j - 1, size)]);
            }
        }
        std::swap(u_1, u_2);
    }
    #pragma acc wait
}

double cublas_diff(cublasHandle_t h, double *restrict u_1, double *restrict u_2, double *restrict diff, int size)
{
    double alpha = -1.0;

    int max_idx = 0;
    #pragma acc host_data use_device(u_1, u_2, diff)
    {
        cublasDcopy(h, size * size, u_2, 1, diff, 1);
        cublasDaxpy(h, size * size, &alpha, u_1, 1, diff, 1);
        cublasIdamax(h, size * size, diff, 1, &max_idx);
    }
    #pragma acc update host(diff[max_idx - 1])

    return fabs(diff[max_idx - 1]);
}

double error_step(cublasHandle_t h, double *restrict u_1, double *restrict u_2, double *restrict diff, int size, int steps)
{
    convolve_steps(u_1, u_2, size, steps);
    
    return cublas_diff(h, u_1, u_2, diff, size);
}

int main(int argc, char *argv[])
{
    po::options_description desc("Options for task6");
    desc.add_options()
    ("size", po::value<int>()->default_value(128), "Matrix size (N x N)")
    ("tolerance", po::value<double>()->default_value(1e-6), "Tolerance")
    ("max_iterations", po::value<int>()->default_value(1000000), "Maximum iterations")
    ("error_calc_step", po::value<int>()->default_value(1000), "Error calculation step");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    int N = vm["size"].as<int>();
    int error_calc_step = vm["error_calc_step"].as<int>();
    double tolerance = vm["tolerance"].as<double>();
    int max_iterations = vm["max_iterations"].as<int>();

    acc_set_device_num(2, acc_device_nvidia);

    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "Maximum iterations: " << max_iterations << std::endl;
    
    acc_device_t dev_type = acc_get_device_type();
    int dev_num = acc_get_device_num(dev_type);

    std::string dev_type_str;
    switch (dev_type) {
        case acc_device_none: dev_type_str = "none"; break;
        case acc_device_default: dev_type_str = "default"; break;
        case acc_device_host: dev_type_str = "host (CPU)"; break;
        case acc_device_not_host: dev_type_str = "not host"; break;
        case acc_device_nvidia: dev_type_str = "NVIDIA GPU"; break;
        default: dev_type_str = "unknown"; break;
    }

    std::cout << "OpenACC device type: " << dev_type_str << std::endl;
    std::cout << "OpenACC device number: " << dev_num << std::endl;

    std::unique_ptr<double[]> u_1_unique(std::make_unique<double[]>(N * N));
    std::unique_ptr<double[]> u_2_unique(std::make_unique<double[]>(N * N));
    std::unique_ptr<double[]> diff_unique(std::make_unique<double[]>(N * N));

    double corner1 = 10.0;
    double corner2 = 20.0;
    double corner3 = 30.0;
    double corner4 = 20.0;

    double* u_1 = u_1_unique.get();
    double* u_2 = u_2_unique.get();
    double* diff = diff_unique.get();
    
    nvtxRangePushA("init");
    std::memset(u_1, 0, N * N * sizeof(double));
    std::memset(u_2, 0, N * N * sizeof(double));
    std::memset(diff, 0, N * N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        u_1[OFFSET(0, i, N)] = corner1 + (corner2 - corner1) * i / (N - 1.0);
        u_1[OFFSET(N - 1, i, N)] = corner4 + (corner3 - corner4) * i / (N - 1.0);
        u_1[OFFSET(i, 0, N)] = corner1 + (corner4 - corner1) * i / (N - 1.0);
        u_1[OFFSET(i, N - 1, N)] = corner2 + (corner3 - corner2) * i / (N - 1.0);

        u_2[OFFSET(0, i, N)] = corner1 + (corner2 - corner1) * i / (N - 1.0);
        u_2[OFFSET(N - 1, i, N)] = corner4 + (corner3 - corner4) * i / (N - 1.0);
        u_2[OFFSET(i, 0, N)] = corner1 + (corner4 - corner1) * i / (N - 1.0);
        u_2[OFFSET(i, N - 1, N)] = corner2 + (corner3 - corner2) * i / (N - 1.0);
    }
    nvtxRangePop();
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    int iteration = 0;
    
    std::cout << "Start calculations.\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma acc data copyin(u_1[0:N*N], u_2[0:N*N]) create(diff[0:N*N])
    {
        double error = tolerance + 1.0;
        nvtxRangePushA("while");
        while (iteration < max_iterations && error > tolerance)
        {
            nvtxRangePushA("step");
            error = error_step(handle, u_1, u_2, diff, N, error_calc_step);
            iteration += error_calc_step;
            nvtxRangePop();
        }
        std::cout << "Error: " << error << "\n";
        nvtxRangePop();
        #pragma acc update host(u_1[0:N*N])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    std::cout << "Total iterations: " << iteration << std::endl;
    std::cout << "Time: " << duration.count() << "\n";
    
    write_to_file(u_1, N, "res_128.txt");
    
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