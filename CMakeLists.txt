cmake_minimum_required(VERSION 3.5)

# Pybind11
if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
endif()

project(OptFus)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}) 

# Specify Python in conda env 
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_SYSTEM_VERSION OFF)
set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Conda Python root")
set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python3" CACHE PATH "Conda Python interpreter")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Pybind11
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
# Important: specify the arch of your CUDA hardware here (compute_89,code=sm_89)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -gencode arch=compute_80,code=sm_80)
# Add position independent code for shared library support
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")


set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++ -fPIC")


set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/randomized_optimization/include)
set(PROJECT_SRC_DIR ${PROJECT_SOURCE_DIR}/randomized_optimization/src)


include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SRC_DIR}/cuda/include")
file(GLOB OptFus_SRCS "${PROJECT_SRC_DIR}/cuda/*.cu" "${PROJECT_SRC_DIR}/*.cpp")
list(FILTER OptFus_SRCS EXCLUDE REGEX ".*bindings\\.cpp$")

cuda_add_library(OptFus_lib STATIC "${OptFus_SRCS}")

pybind11_add_module(optimization_fusion ${PROJECT_SRC_DIR}/bindings.cpp)

target_link_libraries(optimization_fusion PRIVATE
    OptFus_lib
    ${OpenCV_LIBS}
)
