#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <fstream>

#define N 10000

using matrix = std::vector<double>;

void init(matrix &A, matrix &b, matrix &x_0, matrix &x);
void init_parallel(matrix &A, matrix &b, matrix &x_0, matrix &x, int NumThreads);

double length(const matrix &vec);
double dot_length(const matrix &A, const matrix &x, const matrix &b);
void iter_func_parallel_for(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x);
void iter_func_parallel(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x, int NumThreads);
void iter_func_serial(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x);

int main()
{

    matrix A(N * N);
    matrix b(N);

    matrix x_0(N);
    matrix x(N);
    double tau = 0.01;
    double epsilon = 0.00001;
    
    init(A, b, x_0, x);

    std::ofstream outputFile("sole_error.txt");
    
    double err = 1.0;
    double last_err = 1.0;
    int i = 0;
    int last_i = 0;
    int step = 100;
    while (err >= epsilon)
    {
        iter_func_parallel_for(A, b, x_0, tau, x);
        err = dot_length(A, x, b) / length(b);
        i++;
        if (i - last_i >= step && last_err <= err) break;
        else if (i - last_i >= step)
        {
            last_err = err;
            last_i = i;
        }
        std::cout << i << ": " << err << "\n";
    }
    
    for (int i = 0; i < N; i++)
    {
        std::cout << x[i] << "\n";
    }
    
    outputFile.close();

    return 0;
}

void init(matrix &A, matrix &b, matrix &x_0, matrix &x)
{
    for (int i = 0; i < N; i++)
    {
        x_0[i] = 0.0;
        x[i] = 0.0;
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (i == j) ? 2 : 1;
        }
    }
    
    for (int i = 0; i < N; i++)
    {
        b[i] = N + 1;
    }
}

void init_parallel(matrix &A, matrix &b, matrix &x_0, matrix &x, int NumThreads)
{
    for (int i = 0; i < N; i++)
    {
        x_0[i] = 0.0;
        x[i] = 0.0;
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (i == j) ? 2 : 1;
        }
    }
    
    for (int i = 0; i < N; i++)
    {
        b[i] = N + 1;
    }
}

double length(const matrix &vec)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += vec[i] * vec[i];
    }
    sum = std::sqrt(sum);
    return sum;
}

double dot_length(const matrix &A, const matrix &x, const matrix &b)
{
    double res = 0.0;
    for (int i = 0; i < N; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j] * x[j];             
        }
        sum -= b[i];
        res += sum * sum;
    }
    res = std::sqrt(res);
    return res;
}

void iter_func_parallel_for(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x)
{
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j] * x_0[j];             
        }
        x[i] = x_0[i] - lr * (sum - b[i]);
    }
}

void iter_func_parallel(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x, int NumThreads)
{
    for (int i = 0; i < N; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j] * x_0[j];             
        }
        x[i] = x_0[i] - lr * (sum - b[i]);
    }
}

void iter_func_serial(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x)
{
    for (int i = 0; i < N; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j] * x_0[j]; 
        }
        x[i] = x_0[i] - lr * (sum - b[i]);
    }
}
