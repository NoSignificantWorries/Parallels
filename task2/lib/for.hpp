#pragma once

#include <memory>

#define Chunk 200

using vec = std::unique_ptr<double[]>;

void step_parallel_for_auto(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumThreads);
void step_parallel_for_static(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumThreads);
void step_parallel_for_dynamic(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumTreads);
void step_parallel_for_guided(const vec &A, const vec &b, const vec &x_0, vec &x, double tau, int n, int NumThreads);
