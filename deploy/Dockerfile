# Multi-stage build. Two final targets:
#   --target gpu (default)  CUDA runtime image, needs an NVIDIA GPU at run time
#   --target cpu            slim image with the CPU backend only, useful for CI and local k8s
#
#   docker build -f deploy/Dockerfile --target gpu -t engine:gpu .
#   docker build -f deploy/Dockerfile --target cpu -t engine:cpu .
#
# Weights are NOT baked in: mount them at /models and point --model at the mount.

# ---------- builder (CUDA toolkit, builds both backends) ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-gpu
ARG CUDA_ARCH=89
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git g++ ca-certificates libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Copy the dependency manifest first so FetchContent downloads are cached across source edits.
COPY cmake/ cmake/
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DENGINE_TESTS=OFF \
 && cmake --build build --target engine

# ---------- builder (no CUDA, CPU backend only) ----------
FROM ubuntu:22.04 AS build-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git g++ ca-certificates libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cmake/ cmake/
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENGINE_TESTS=ON \
 && cmake --build build \
 && ./build/tests/engine_tests "~[model]"

# ---------- runtime: GPU ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 engine
COPY --from=build-gpu /src/build/engine /usr/local/bin/engine
USER 10001
EXPOSE 8000
ENV ENGINE_MODEL=/models/Qwen2.5-0.5B-Instruct
# SIGTERM makes the server stop accepting, finish in-flight sequences, then exit.
STOPSIGNAL SIGTERM
ENTRYPOINT ["/bin/sh", "-c", "exec engine serve --model \"$ENGINE_MODEL\" --backend cuda --host 0.0.0.0 --port 8000 $ENGINE_ARGS"]

# ---------- runtime: CPU ----------
FROM ubuntu:22.04 AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 engine
COPY --from=build-cpu /src/build/engine /usr/local/bin/engine
USER 10001
EXPOSE 8000
ENV ENGINE_MODEL=/models/Qwen2.5-0.5B-Instruct
STOPSIGNAL SIGTERM
ENTRYPOINT ["/bin/sh", "-c", "exec engine serve --model \"$ENGINE_MODEL\" --host 0.0.0.0 --port 8000 $ENGINE_ARGS"]
