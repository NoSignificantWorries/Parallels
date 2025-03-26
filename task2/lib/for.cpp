#include "for.hpp"
#include <omp.h>

void step_parallel_for_auto(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
#pragma omp parallel
    {
#pragma omp for schedule(auto)
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
}

void step_parallel_for_static(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int _)
{
#pragma omp parallel
    {
#pragma omp for schedule(static)
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
}

void step_parallel_for_dynamic(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int ChunkSize)
{
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, ChunkSize)
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
}

void step_parallel_for_guided(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int ChunkSize)
{
#pragma omp parallel
    {
#pragma omp for schedule(guided, ChunkSize)
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
}
