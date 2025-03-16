#pragma once

#include <memory>

using vec = std::unique_ptr<double[]>;

void generate_parallel_auto(vec &A, vec &b, vec &x, vec &x_0, int n, int _);
void generate_parallel_static(vec &A, vec &b, vec &x, vec &x_0, int n, int _);
void generate_parallel_dynamic(vec &A, vec &b, vec &x, vec &x_0, int n, int ChunkSize);
void generate_parallel_guided(vec &A, vec &b, vec &x, vec &x_0, int n, int _);

void step_parallel_for_auto(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int _);
void step_parallel_for_static(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int _);
void step_parallel_for_dynamic(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int ChunkSize);
void step_parallel_for_guided(const vec &A, const vec &b, vec &x_0, vec &x, double tau, int n, int _);
