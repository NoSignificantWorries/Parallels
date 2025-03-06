#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

#define N 100

using matrix = std::vector<double>;

void parallel_init(matrix &A, matrix &b, matrix &x_0, matrix &x);

double length(const matrix &vec);
void dot(const matrix &A, const matrix &x, const matrix &b, matrix &c);
void iter_func(const matrix &A, const matrix &b, const matrix &x_0, double lr, matrix &x);

int main()
{

    matrix A(N * N);
    matrix b(N);

    matrix x_0(N);
    matrix x(N);
    double tau = 0.01;
    
    parallel_init(A, b, x_0, x);

    return 0;
}

void parallel_init(matrix &A, matrix &b, matrix &x_0, matrix &x)
{
    #pragma omp for
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
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
        #pragma atomic
        sum += vec[i] * vec[i];
    }
    sum = std::sqrt(sum);
    return sum;
}

void dot(const matrix &A, const matrix &x, const matrix &b, matrix &c)
{
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[i] += A[i * N + j] * x[j];             
        }
        c[i] -= b[i];
    }
}

void iter_func(const matrix &A, const matrix &b, const matrix &X_0, double lr, matrix &X)
{
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[i] += A[i * N + j] * x[j];             
        }
        c[i] -= b[i];
    }
}
