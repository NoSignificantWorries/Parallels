cmake_minimum_required(VERSION 3.10)
project(Task2 LANGUAGES CXX)

add_library(for STATIC "lib/for.cpp" "lib/for.hpp")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

find_package(OpenMP REQUIRED)

find_package(Threads REQUIRED)

file(GLOB SOURCES "*.cpp")

foreach(SRC ${SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
    target_link_libraries(${EXE_NAME} PRIVATE Threads::Threads)
    target_link_libraries(${EXE_NAME} PRIVATE for)
endforeach()
