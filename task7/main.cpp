#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>
#include <iomanip>
#include <openacc.h>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

double calc_error_cublas(const double* d_u1, const double* d_u2, double* d_diff, int N, cublasHandle_t handle)
{
    int size = N * N;
    /*
    double* d_diff = nullptr;
    cudaMalloc((void**)&d_diff, size * sizeof(double));
    */

    const double alpha = -1.0;
    cudaMemcpy(d_diff, d_u2, size * sizeof(double), cudaMemcpyDeviceToDevice);
    cublasDaxpy(handle, size, &alpha, d_u1, 1, d_diff, 1);

    int idx = 0;
    cublasIdamax(handle, size, d_diff, 1, &idx);

    idx = idx - 1;

    double max_err = 0.0;
    cudaMemcpy(&max_err, d_diff + idx, sizeof(double), cudaMemcpyDeviceToHost);
    max_err = std::abs(max_err);

    // cudaFree(d_diff);
    return max_err;
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

    std::cout << "Matrix size" << N << "x" << N << std::endl;
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
    std::unique_ptr<double[]> u_diff_unique(std::make_unique<double[]>(N * N));

    double corner1 = 10.0;
    double corner2 = 20.0;
    double corner3 = 30.0;
    double corner4 = 20.0;

    nvtxRangePushA("init");
    for (int i = 0; i < N; ++i)
    {
        u_1_unique[OFFSET(0, i, N)] = corner1 + (corner2 - corner1) * i / (N - 1.0);
        u_1_unique[OFFSET(N - 1, i, N)] = corner4 + (corner3 - corner4) * i / (N - 1.0);
        u_1_unique[OFFSET(i, 0, N)] = corner1 + (corner4 - corner1) * i / (N - 1.0);
        u_1_unique[OFFSET(i, N - 1, N)] = corner2 + (corner3 - corner2) * i / (N - 1.0);

        u_2_unique[OFFSET(0, i, N)] = corner1 + (corner2 - corner1) * i / (N - 1.0);
        u_2_unique[OFFSET(N - 1, i, N)] = corner4 + (corner3 - corner4) * i / (N - 1.0);
        u_2_unique[OFFSET(i, 0, N)] = corner1 + (corner4 - corner1) * i / (N - 1.0);
        u_2_unique[OFFSET(i, N - 1, N)] = corner2 + (corner3 - corner2) * i / (N - 1.0);
    }
    nvtxRangePop();

    int iteration = 0;
    double error = 1.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    double* u_1 = u_1_unique.get();
    double* u_2 = u_2_unique.get();
    double* u_diff = u_diff_unique.get();
    
    nvtxRangePushA("while_with_copy");
    #pragma acc data copyin(error, u_diff[0:N*N], u_1[0:N*N], u_2[0:N*N])
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        nvtxRangePushA("while");
        while (iteration < max_iterations && error > tolerance)
        {
            nvtxRangePushA("calc_step");
            #pragma acc parallel loop independent collapse(2) present(u_1, u_2) // async(1)
            for (int i = 1; i < N - 1; ++i)
            {
                for (int j = 1; j < N - 1; ++j)
                {
                    u_2[OFFSET(i, j, N)] = 0.25 * (u_1[OFFSET(i + 1, j, N)] + u_1[OFFSET(i - 1, j, N)] + u_1[OFFSET(i, j + 1, N)] + u_1[OFFSET(i, j - 1, N)]);
                }
            }
            nvtxRangePop();
            
            if ((iteration + 1) % error_calc_step == 0)
            {
                nvtxRangePushA("calc_error");
                double* d_u1 = (double*)acc_deviceptr(u_1);
                double* d_u2 = (double*)acc_deviceptr(u_2);
                double* d_diff = (double*)acc_deviceptr(u_diff);
                error = calc_error_cublas(d_u1, d_u2, d_diff, N, handle);
                std::cout << iteration << " " << error << "\n";
                nvtxRangePop();
            }
            
            nvtxRangePushA("swap u_1 u_2");
            /*
            #pragma acc parallel loop independent collapse(2) present(u_1, u_2)
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    u_1[OFFSET(i, j, N)] = u_2[OFFSET(i, j, N)];
            */
            std::swap(u_1, u_2);
            nvtxRangePop();

            ++iteration;
        }
        #pragma acc update self(u_1[0:N*N])
        nvtxRangePop();
        cublasDestroy(handle);
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    std::cout << "Total iterations: " << iteration << std::endl;
    std::cout << "Final error: " << error << std::endl;
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
