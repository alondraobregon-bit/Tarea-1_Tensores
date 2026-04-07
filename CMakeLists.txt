cmake_minimum_required(VERSION 4.2)
project(proyectp)

set(CMAKE_CXX_STANDARD 20)

add_executable(proyectp main.cpp
        Tensor.cpp
        Tensor.h
        TensorTransform.cpp
        TensorTransform.h)
