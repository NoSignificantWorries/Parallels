#include "for.hpp"
#include <omp.h>

void generate_parallel_auto(vec &A, vec &b, vec &x, vec &x_0, int n, int _)
{
    #pragma omp for schedule(auto)
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>(n + 1);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
    for (int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void generate_parallel_static(vec &A, vec &b, vec &x, vec &x_0, int n, int _)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>(n + 1);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
    for (int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void generate_parallel_dynamic(vec &A, vec &b, vec &x, vec &x_0, int n, int ChunkSize)
{
    #pragma omp for schedule(dynamic, ChunkSize)
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>(n + 1);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
    for (int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}

void generate_parallel_guided(vec &A, vec &b, vec &x, vec &x_0, int n, int _)
{
    #pragma omp for schedule(guided)
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>(n + 1);
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = static_cast<double>((i == j) ? 2 : 1);
        }
    }
    for (int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(0.0);
        x_0[i] = static_cast<double>(0.0);
    }
}


void step_parallel_for_auto(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
    #pragma omp for schedule(guided)
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x_0[j];
        }
        x[i] = x_0[i] - tau * (tmp - b[i]);
    }
}

void step_parallel_for_static(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
    #pragma omp for schedule(guided)
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x_0[j];
        }
        x[i] = x_0[i] - tau * (tmp - b[i]);
    }
}

void step_parallel_for_dynamic(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int ChunkSize)
{
    #pragma omp for schedule(guided)
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x_0[j];
        }
        x[i] = x_0[i] - tau * (tmp - b[i]);
    }
}

void step_parallel_for_guided(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
    #pragma omp for schedule(guided)
    for (int i = 0; i < n; i++)
    {
        double tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += A[i * n + j] * x_0[j];
        }
        x[i] = x_0[i] - tau * (tmp - b[i]);
    }
}
