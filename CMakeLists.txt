# CMake minimum version
cmake_minimum_required(VERSION 3.18)

# Project Name and Language (C++ and CUDA)
project(GasSimulation LANGUAGES CXX CUDA)

# Enable CUDA debugging flags (fancy cuz cflow don't know sh*t about cuda)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")

# Create the executable by listing its source files with their paths
add_executable(gas_simulation
    src/main.cpp
    src/simulation.cu
)

# Link the filesystem library for C++17
target_link_libraries(gas_simulation PRIVATE stdc++fs)

# Tell the compiler to look for header files in the 'src' directory.
# This allows #include "particle_system.h" to work correctly from any file.
target_include_directories(gas_simulation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Tell CMake to use the C++17 standard
target_compile_features(gas_simulation PUBLIC cxx_std_17)
set_target_properties(gas_simulation PROPERTIES
    CUDA_STANDARD 17
)