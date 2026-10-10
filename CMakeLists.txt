cmake_minimum_required(VERSION 3.24)
project(engine LANGUAGES CXX)

option(ENGINE_CUDA  "Build the CUDA backend"          OFF)
option(ENGINE_TESTS "Build unit tests"                ON)
option(ENGINE_BENCH "Build benchmark tools"           ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps.cmake)

find_package(Threads REQUIRED)
if(APPLE)
  set(BLA_VENDOR Apple)
endif()
find_package(BLAS)

# ---- core library: everything that does not need a GPU ----
add_library(engine_core
  src/config/config.cpp
  src/safetensors/safetensors.cpp
  src/tokenizer/tokenizer.cpp
  src/kv_cache/block_allocator.cpp
  src/scheduler/scheduler.cpp
  src/model/cpu/cpu_model.cpp
  src/model/sampler.cpp
  src/api/server.cpp
  src/api/metrics.cpp
  src/api/engine.cpp
)
target_include_directories(engine_core PUBLIC include)
target_link_libraries(engine_core PUBLIC nlohmann_json::nlohmann_json httplib::httplib Threads::Threads)
target_compile_options(engine_core PRIVATE -Wall -Wextra -Wpedantic)
if(BLAS_FOUND)
  message(STATUS "CPU GEMMs via BLAS: ${BLAS_LIBRARIES}")
  target_link_libraries(engine_core PUBLIC ${BLAS_LIBRARIES})
  target_compile_definitions(engine_core PRIVATE ENGINE_HAVE_CBLAS=1)
  if(APPLE)
    target_compile_definitions(engine_core PRIVATE ACCELERATE_NEW_LAPACK=1)
  endif()
else()
  message(STATUS "No BLAS found; CPU GEMMs use naive loops (slow, still correct)")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(engine_core PUBLIC -O3)
endif()

# The CUDA backend.
#
# This list once doubled as an implementation check: configuring with
# ENGINE_CUDA=ON while a source was absent produced a message explaining that
# the backend was unwritten, rather than a confusing missing-file error. The
# check has been removed because it never measured what it claimed to. Both
# attention_prefill.cu and sampling.cu existed for months as one-line comments,
# which satisfies an existence test and compiles to an empty translation unit;
# the guard reported a complete backend while two of its kernels were absent.
# Whether a backend works is a question for the test suite, which compares
# against the CPU oracle, not for a file listing.
set(ENGINE_CUDA_SOURCES
  src/model/cuda/cuda_model.cu
  src/kernels/rmsnorm.cu
  src/kernels/rope.cu
  src/kernels/silu_mul.cu
  src/kernels/embedding.cu
  src/kernels/elementwise.cu
  src/kernels/store_kv.cu
  src/kernels/attention_decode.cu
  src/kernels/attention_prefill.cu
  src/kernels/sampling.cu
)
if(ENGINE_CUDA)
  # The architecture has to be chosen *before* enable_language(CUDA): that call
  # compiles a probe binary, and with no architecture set it tries to detect one
  # and fails outright on some driver/toolkit pairs. Setting it afterwards looks
  # correct and works only because every caller happened to pass the value on
  # the command line, which populates the cache earlier than this file runs.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "sm_89 = RTX 4090; 86 = A40/A6000")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  # An OBJECT library rather than a static one: engine_core calls
  # make_cuda_model, which lives here, while this needs engine_core's headers.
  # Two static archives in that cycle do not resolve in one link pass, so the
  # objects go straight into the final binaries instead.
  add_library(engine_cuda OBJECT ${ENGINE_CUDA_SOURCES})
  target_include_directories(engine_cuda PUBLIC include)
  target_link_libraries(engine_cuda PUBLIC CUDA::cudart CUDA::cublas
                        nlohmann_json::nlohmann_json)
  # Both targets need the define: engine_core because it chooses a backend, and
  # engine_cuda because the kernel declarations in cuda_kernels.h are guarded by
  # it. Setting it only on engine_core hid those declarations from the one file
  # that calls them.
  target_compile_definitions(engine_cuda PUBLIC ENGINE_CUDA=1)
  target_compile_definitions(engine_core PUBLIC ENGINE_CUDA=1)
  target_link_libraries(engine_core PUBLIC engine_cuda)
endif()

add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE engine_core)

if(ENGINE_BENCH)
  # Per-operation golden tensors, so each CUDA kernel can be verified on its own
  # rather than by diffing the end of a 24-layer forward pass.
  add_executable(golden_dump src/golden_dump.cpp)
  target_link_libraries(golden_dump PRIVATE engine_core)

  # KV-cache efficiency is a property of the allocator, not of the GPU, so it is
  # measured with a fake model on any machine.
  add_executable(kv_waste bench/kv_waste.cpp)
  target_link_libraries(kv_waste PRIVATE engine_core)
endif()

if(ENGINE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
