cmake_minimum_required(VERSION 3.24)
project(engine LANGUAGES CXX)

option(ENGINE_CUDA  "Build the CUDA backend"          OFF)
option(ENGINE_TESTS "Build unit tests"                ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps.cmake)

find_package(Threads REQUIRED)

# ---- core library: everything that does not need a GPU ----
add_library(engine_core
  src/config/config.cpp
  src/safetensors/safetensors.cpp
)
target_include_directories(engine_core PUBLIC include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json httplib::httplib Threads::Threads)
target_compile_options(engine_core PRIVATE -Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(engine_core PUBLIC -O3)
endif()

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "sm_89 = RTX 4090")
  find_package(CUDAToolkit REQUIRED)
  add_library(engine_cuda
    src/model/cuda/cuda_model.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/silu_mul.cu
    src/kernels/embedding.cu
    src/kernels/attention_decode.cu
    src/kernels/attention_prefill.cu
    src/kernels/sampling.cu
  )
  target_include_directories(engine_cuda PUBLIC include)
  target_link_libraries(engine_cuda PUBLIC engine_core CUDA::cudart CUDA::cublas)
  target_compile_definitions(engine_core PUBLIC ENGINE_CUDA=1)
endif()

add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE engine_core)
if(ENGINE_CUDA)
  target_link_libraries(engine PRIVATE engine_cuda)
endif()

if(ENGINE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
