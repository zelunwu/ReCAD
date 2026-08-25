cmake_minimum_required(VERSION 3.20)

project(recad_cuda VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optional CUDA ----------------------------------------------------------
# CUDA is enabled only when the toolkit is found, so CPU-only builds (machines
# without nvcc, or an nvcc/MSVC version mismatch) work out of the box. Force
# it off with -DRECAD_HAS_CUDA=OFF, or block the probe entirely with
# -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE.
option(RECAD_HAS_CUDA "Compile the CUDA device implementations" ON)
set(RECAD_CUDA_ENABLED OFF)
if(RECAD_HAS_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(RECAD_CUDA_ENABLED ON)
    message(STATUS "recad_cuda: CUDA toolkit found; building device kernels")
  else()
    message(STATUS "recad_cuda: CUDA toolkit not found; CPU-only build")
  endif()
else()
  message(STATUS "recad_cuda: CUDA disabled by RECAD_HAS_CUDA=OFF; CPU-only")
endif()

# The reference implementation always ships; the device file only when CUDA is
# enabled. RECAD_HAS_CUDA is PUBLIC so consumers know which path is compiled.
add_library(recad_cuda SHARED src/kernels_host.cpp)
if(RECAD_CUDA_ENABLED)
  target_sources(recad_cuda PRIVATE src/kernels_device.cu)
  target_compile_definitions(recad_cuda PUBLIC RECAD_HAS_CUDA=1)
endif()

target_include_directories(recad_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Compile options: /O2 (MSVC) or -O3 (GCC/Clang), with reasonable warnings
# forwarded through -Xcompiler for CUDA sources (nvcc drives the host
# compiler, so plain -Wall would not reach it).
if(MSVC)
  target_compile_options(recad_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2 -Xcompiler=/W3>)
else()
  target_compile_options(recad_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3 -Xcompiler=-Wall
      -Xcompiler=-Wextra>)
endif()

# --- Tests ------------------------------------------------------------------
option(RECAD_ENABLE_TESTS "Build the kernel test harness" ON)
if(RECAD_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()