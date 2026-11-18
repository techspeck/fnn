# 1. Specify the minimum version of CMake required
cmake_minimum_required(VERSION 4.3.4)

# 2. Define the project name and language
project(fnn LANGUAGES CXX)

# 3. Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Define the executable target and its source files
add_executable(
    fnn
    src/main.cpp
)
