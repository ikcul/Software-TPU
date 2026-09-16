cmake_minimum_required(VERSION 3.16)
project(software_tpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX2 / AVX-512 flags depending on compiler
if(MSVC)
    add_compile_options(/arch:AVX2 /W4)
else()
    add_compile_options(-mavx2 -mfma -Wall -Wextra -O3)
endif()

add_executable(software_tpu main.cpp)
