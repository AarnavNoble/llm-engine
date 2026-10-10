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

# The CUDA backend. Its sources are listed here before they exist so the build
# shape is settled, but configuring with ENGINE_CUDA=ON while they are missing
# would fail with a confusing missing-file error, so say what is actually wrong.
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
set(ENGINE_CUDA_MISSING "")
foreach(src ${ENGINE_CUDA_SOURCES})
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${src})
    list(APPEND ENGINE_CUDA_MISSING ${src})
  endif()
endforeach()

if(ENGINE_CUDA AND ENGINE_CUDA_MISSING)
  message(FATAL_ERROR
    "ENGINE_CUDA=ON but the CUDA backend is not implemented yet.\n"
    "Missing: ${ENGINE_CUDA_MISSING}\n"
    "The CPU backend is complete and is the default: drop -DENGINE_CUDA=ON.\n"
    "Status and the plan for the GPU work are in docs/gpu-setup.md.")
endif()

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "sm_89 = RTX 4090")
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
