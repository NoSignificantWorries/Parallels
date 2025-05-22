cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "pgc++ -acc=gpu -Minfo=all")

project(Task6 LANGUAGES CXX)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

find_package(Boost REQUIRED COMPONENTS program_options)

file(GLOB SOURCES "src/*.cpp")

foreach(SRC ${SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_compile_options(${EXE_NAME} PRIVATE -O2) # -acc -Minfo=all
    target_link_libraries(${EXE_NAME} PRIVATE Boost::program_options)
endforeach()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
