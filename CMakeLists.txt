###############################################################################
#
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# All rights reserved.
#
# This file is part of the RAJA Proxy App Suite
#
###############################################################################

cmake_minimum_required(VERSION 3.4)

option(LLVM_BUILD "Path to llvm build directory" "/home/ubuntu/omp-llvm/build")
option(ENZYME "Path to ClangEnzyme.so" "/home/ubuntu/Enzyme/enzyme/buildomp/Enzyme/ClangEnzyme-15.so")
option(MPI_INCLUDE "Path to MPI include directory" "/usr/lib/x86_64-linux-gnu/openmpi/include") 
set(CMAKE_C_COMPILER   "${LLVM_BUILD}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${LLVM_BUILD}/bin/clang++" CACHE PATH "")
set(CMAKE_LINKER       "${LLVM_BUILD}/bin/clang++" CACHE PATH "")
set (ENABLE_OPENMP ON)
set(CMAKE_CXX_FLAGS " -I ${LLVM_BUILD}/projects/openmp/runtime/src -I ${MPI_INCLUDE} -flegacy-pass-manager -Xclang -load -Xclang ${ENZYME} -fopenmp -lmpi -O3 -ffast-math -mllvm -enzyme-loose-types -DLULESH_LIST_INDEXSET" CACHE STRING "")

set(BLT_CXX_STD "c++11")
project(RajaProxies LANGUAGES CXX)

option(ENABLE_LULESH_ONE "Build LULESH v1.0" OFF)
option(ENABLE_LULESH_TWO "Build LULESH v2.0" On)

set (ENABLE_TESTS Off CACHE Bool "Don't build tests")
set (ENABLE_EXAMPLES Off CACHE Bool "Don't build examples")
set (ENABLE_EXERCISES Off CACHE Bool "Don't build exercises")


################################
# BLT
################################
if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
  message(FATAL_ERROR "\
  The BLT submodule is not present. \
  If in git repository run the following two commands:\n \
  git submodule init\n \
  git submodule update")
endif()
include(${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)

add_subdirectory(tpl/RAJA)

if (ENABLE_LULESH_ONE)
  message(STATUS "Building LULESH v1.0")
  add_subdirectory(lulesh-v1.0)
endif ()

if (ENABLE_LULESH_TWO)
  message(STATUS "Building LULESH v2.0")
  add_subdirectory(lulesh-v2.0)
endif ()
