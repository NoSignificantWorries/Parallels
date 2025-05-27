cmake_minimum_required(VERSION 3.9)
project(lab7)

set(ACC "gpu" CACHE STRING "device")

set(CMAKE_CXX_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/pgc++)
set(CMAKE_CXX_FLAGS "-O3 -fast -acc=${ACC} -cudalib=cublas -lboost_program_options")

add_executable(lab7 main.cpp)

set_property(TARGET lab7 PROPERTY CXX_STANDARD 20)
