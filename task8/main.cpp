#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

#define OFFSET(row, col, m) (((row) * (m)) + (col))
#define threadsPerBlock 1024

namespace po = boost::program_options;


void print(double* arr, int N)
{
   for (int i = 0; i < N; i++)
   {
        for (int j = 0; j < N; j++)
        {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << std::setfill(' ') << arr[OFFSET(i, j, N)] << " ";
        }
        std::cout << "\n";
   } 
}

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


__global__ void myKernel(const double* d_u_1, double* d_u_2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int i = idx / N;
    int j = idx % N;

    if (i >= 1 && i <= (N - 2) && j >= 1 && j <= (N - 2))
    {
        d_u_2[OFFSET(i, j, N)] = 0.25 * (d_u_1[OFFSET(i + 1, j, N)] + 
                                         d_u_1[OFFSET(i - 1, j, N)] +
                                         d_u_1[OFFSET(i, j + 1, N)] + 
                                         d_u_1[OFFSET(i, j - 1, N)]);
    }
}


template <int BLOCK_SIZE>
__global__ void maxErrorKernel(const double* d_u_1, const double* d_u_2, double* block_maxes, int size) {
    __shared__ typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double diff = 0.0f;
    if (idx < size)
    {
        diff = fabsf(d_u_2[idx] - d_u_1[idx]);
    }

    double block_max = cub::BlockReduce<double, BLOCK_SIZE>(temp_storage).Reduce(diff, cub::Max());

    if (threadIdx.x == 0)
    {
        block_maxes[blockIdx.x] = block_max;
    }
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

    nvtxRangePushA("init");
    initialize_boundary(u_1, N);
    initialize_boundary(u_2, N);
    nvtxRangePop();
    
    cudaSetDevice(3);
    
    nvtxRangePushA("Prepare graph");
    //========================================================================
    int steps = 1000;
    double *d_u_1;
    double *d_u_2;
    double* d_max_error;
    double* d_block_maxes;

    int size = N * N;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Threads per block: " << threadsPerBlock << "\n";
    std::cout << "Blocks: " << blocksPerGrid << "\n";

    cudaMalloc(&d_u_1, size * sizeof(double));
    cudaMalloc(&d_u_2, size * sizeof(double));
    cudaMalloc(&d_block_maxes, blocksPerGrid * sizeof(double));
    cudaMalloc(&d_max_error, sizeof(double));

    cudaMemcpy(d_u_1, u_1, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_2, u_2, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes, d_block_maxes, d_max_error, blocksPerGrid);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    double* h_max_error;
    cudaMallocHost(&h_max_error, sizeof(double));

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for (int i = 0; i < steps; ++i)
    {
        if (i % 2 == 0)
        {
            myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_u_1, d_u_2, N);
        }
        else
        {
            myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_u_2, d_u_1, N);
        }
    }

    maxErrorKernel<threadsPerBlock><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_u_1, d_u_2, d_block_maxes, size);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_block_maxes, d_max_error, blocksPerGrid, stream);
    cudaMemcpyAsync(h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    //========================================================================
    nvtxRangePop();
    
    int iter = 0;
    double max_error = 1.0 + tolerance;
    
    nvtxRangePushA("while");
    auto start = std::chrono::high_resolution_clock::now();
    
    while (max_error > tolerance && iter < max_iters)
    {
        // max_error = error_step(u_1, u_2, N, 1000);
        
        nvtxRangePushA("step");

        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        nvtxRangePop();
        
        /*
        for (int i = 0; i < steps; ++i)
        {
            if (i % 2 == 0)
            {
                myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_u_1, d_u_2, N);
            }
            else
            {
                myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_u_2, d_u_1, N);
            }
        }

        maxErrorKernel<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(d_u_1, d_u_2, d_block_maxes, size);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_block_maxes, d_max_error, blocksPerGrid);
        */
        
        // cudaMemcpy(h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost);
        max_error = *h_max_error;

        iter += steps;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    nvtxRangePop();

    cudaMemcpy(u_1, d_u_1, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_2, d_u_2, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    
    std::cout << "Total iterations: " << iter << "\n";
    std::cout << "Error: " << max_error << "\n";
    std::cout << "Time: " << duration.count() << "\n";
    
    if (N >= 10 && N <= 20)
    {
        print(u_1, N);
    }
    
    cudaFree(d_u_1);
    cudaFree(d_u_2);
    cudaFree(d_block_maxes);
    cudaFreeHost(h_max_error);
    cudaFree(d_max_error);
    cudaFree(d_temp_storage);

    return 0;
}
