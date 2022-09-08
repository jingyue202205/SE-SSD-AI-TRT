cmake_minimum_required(VERSION 2.6)

project(se-ssd-ai-trt)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)


find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS -G;-g)
include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/xxx/softwares/nvidia/TensorRT-8.0.1.6/include)
link_directories(/home/xxx/softwares/nvidia/TensorRT-8.0.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(voxelGeneratorlayer SHARED ${PROJECT_SOURCE_DIR}/voxelGenerator.cu)
target_link_libraries(voxelGeneratorlayer nvinfer cudart)

cuda_add_library(submConv3dlayer SHARED ${PROJECT_SOURCE_DIR}/submConv3dlayer.cu)
target_link_libraries(submConv3dlayer nvinfer cudart)

cuda_add_library(sparseConv3dlayer SHARED ${PROJECT_SOURCE_DIR}/sparseConv3dlayer.cu)
target_link_libraries(sparseConv3dlayer nvinfer cudart)

cuda_add_library(sparse2Denselayer SHARED ${PROJECT_SOURCE_DIR}/sparse2Dense.cu)
target_link_libraries(sparse2Denselayer nvinfer cudart)

cuda_add_library(zeroPad2dlayer SHARED ${PROJECT_SOURCE_DIR}/zeroPad2d.cu)
target_link_libraries(zeroPad2dlayer nvinfer cudart)

cuda_add_library(generateAnchorDecodelayer SHARED ${PROJECT_SOURCE_DIR}/generateAnchorDecode.cu)
target_link_libraries(generateAnchorDecodelayer nvinfer cudart)

cuda_add_library(filterBoxByScorelayer SHARED ${PROJECT_SOURCE_DIR}/filterBoxByScore.cu)
target_link_libraries(filterBoxByScorelayer nvinfer cudart)

add_executable(se-ssd-ai-trt ${PROJECT_SOURCE_DIR}/se-ssd-ai-trt.cpp)
target_link_libraries(se-ssd-ai-trt nvinfer)
target_link_libraries(se-ssd-ai-trt cudart)
target_link_libraries(se-ssd-ai-trt voxelGeneratorlayer)
target_link_libraries(se-ssd-ai-trt submConv3dlayer)
target_link_libraries(se-ssd-ai-trt sparseConv3dlayer)
target_link_libraries(se-ssd-ai-trt sparse2Denselayer)
target_link_libraries(se-ssd-ai-trt zeroPad2dlayer)
target_link_libraries(se-ssd-ai-trt generateAnchorDecodelayer)
target_link_libraries(se-ssd-ai-trt filterBoxByScorelayer)

add_definitions(-O2 -pthread)

