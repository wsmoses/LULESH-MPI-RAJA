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

set (lulesh_sources
  lulesh.cpp
  lulesh-init.cpp
  lulesh-util.cpp
  lulesh-viz.cpp)

set (lulesh_depends
  RAJA)

if (ENABLE_OPENMP)
  set (lulesh_depends
    ${lulesh_depends}
    openmp)
endif ()

if (ENABLE_CUDA)
  set (lulesh_depends
    ${lulesh_depends}
    cuda)
  set (lulesh_compile_options
    --expt-extended-lambda -arch=${CUDA_ARCH})
else()
  set (lulesh_compile_options
    -Wno-unknown-pragmas)
endif ()

blt_add_executable(
  NAME lulesh-v2.0-RAJA-seq.exe
  SOURCES ${lulesh_sources}
  DEPENDS_ON ${lulesh_depends})

target_compile_options(
  lulesh-v2.0-RAJA-seq.exe
  PRIVATE
  ${lulesh_compile_options})

target_compile_definitions(
  lulesh-v2.0-RAJA-seq.exe
  PRIVATE 
  -DUSE_MPI=0)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq.exe
  PRIVATE
  -DUSE_CASE=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq.exe
  PUBLIC
  -DLULESH_DEVICE= 
)

blt_add_executable(
  NAME lulesh-v2.0-RAJA-seq-grad.exe
  SOURCES ${lulesh_sources}
  DEPENDS_ON ${lulesh_depends})

target_compile_options(
  lulesh-v2.0-RAJA-seq-grad.exe
  PRIVATE
  ${lulesh_compile_options})

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-grad.exe
  PRIVATE 
  -DUSE_MPI=0)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-grad.exe
  PRIVATE
  -DUSE_CASE=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-grad.exe
  PUBLIC
  -DLULESH_DEVICE= 
)
target_compile_definitions(
  lulesh-v2.0-RAJA-seq-grad.exe
  PUBLIC
  -DGRADIENT=1 
)

if (ENABLE_OPENMP)
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-omp.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp.exe
    PRIVATE 
    -DUSE_MPI=0)

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_options(
    lulesh-v2.0-RAJA-omp.exe
    PRIVATE -mllvm -enzyme-omp-opt=0
    ${lulesh_compile_options})
  
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-ompOpt.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt.exe
    PRIVATE 
    -DUSE_MPI=0)

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_options(
    lulesh-v2.0-RAJA-ompOpt.exe
    PRIVATE -mllvm -enzyme-omp-opt=1
    ${lulesh_compile_options})
endif ()

if (ENABLE_OPENMP)
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-omp-gradient.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-gradient.exe
    PRIVATE 
    -DUSE_MPI=0)

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-gradient.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-gradient.exe
    PUBLIC
    -DGRADIENT=1 
  )
  target_compile_options(
    lulesh-v2.0-RAJA-omp-gradient.exe
    PRIVATE -mllvm -enzyme-omp-opt=0
    ${lulesh_compile_options})
  
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-ompOpt-gradient.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-gradient.exe
    PRIVATE 
    -DUSE_MPI=0)

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-gradient.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-gradient.exe
    PUBLIC
    -DGRADIENT=1 
  )
  target_compile_options(
    lulesh-v2.0-RAJA-ompOpt-gradient.exe
    PRIVATE -mllvm -enzyme-omp-opt=1
    ${lulesh_compile_options})
endif ()





blt_add_executable(
  NAME lulesh-v2.0-RAJA-seq-mpi.exe
  SOURCES ${lulesh_sources}
  DEPENDS_ON ${lulesh_depends})

target_compile_options(
  lulesh-v2.0-RAJA-seq-mpi.exe
  PRIVATE -mllvm -enzyme-omp-opt=0
  ${lulesh_compile_options})

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi.exe
  PRIVATE 
  -DUSE_MPI=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi.exe
  PRIVATE
  -DUSE_CASE=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi.exe
  PUBLIC
  -DLULESH_DEVICE= 
)

blt_add_executable(
  NAME lulesh-v2.0-RAJA-seq-mpi-grad.exe
  SOURCES ${lulesh_sources}
  DEPENDS_ON ${lulesh_depends})

target_compile_options(
  lulesh-v2.0-RAJA-seq-mpi-grad.exe
  PRIVATE -mllvm -enzyme-omp-opt=0
  ${lulesh_compile_options})

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi-grad.exe
  PRIVATE 
  -DUSE_MPI=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi-grad.exe
  PRIVATE
  -DUSE_CASE=1)

target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi-grad.exe
  PUBLIC
  -DLULESH_DEVICE= 
)
target_compile_definitions(
  lulesh-v2.0-RAJA-seq-mpi-grad.exe
  PUBLIC
  -DGRADIENT=1 
)

if (ENABLE_OPENMP)
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-omp-mpi.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-mpi.exe
    PRIVATE 
    -DUSE_MPI=1)

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-mpi.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_options(
  lulesh-v2.0-RAJA-omp-mpi.exe
  PRIVATE -mllvm -enzyme-omp-opt=0
  ${lulesh_compile_options})
  
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-ompOpt-mpi.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-mpi.exe
    PRIVATE 
    -DUSE_MPI=1)

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-mpi.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_options(
  lulesh-v2.0-RAJA-ompOpt-mpi.exe
  PRIVATE -mllvm -enzyme-omp-opt=1
  ${lulesh_compile_options})

endif ()

if (ENABLE_OPENMP)
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-omp-mpi-gradient.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-mpi-gradient.exe
    PRIVATE 
    -DUSE_MPI=1)

  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-mpi-gradient.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_definitions(
    lulesh-v2.0-RAJA-omp-mpi-gradient.exe
    PUBLIC
    -DGRADIENT=1 
  )
  target_compile_options(
  lulesh-v2.0-RAJA-omp-mpi-gradient.exe
  PRIVATE -mllvm -enzyme-omp-opt=0
  ${lulesh_compile_options})
  
  blt_add_executable(
    NAME lulesh-v2.0-RAJA-ompOpt-mpi-gradient.exe
    SOURCES ${lulesh_sources}
    DEPENDS_ON ${lulesh_depends})

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-mpi-gradient.exe
    PRIVATE 
    -DUSE_MPI=1)

  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-mpi-gradient.exe
    PUBLIC
    -DLULESH_DEVICE= 
  )
  target_compile_definitions(
    lulesh-v2.0-RAJA-ompOpt-mpi-gradient.exe
    PUBLIC
    -DGRADIENT=1 
  )
  target_compile_options(
  lulesh-v2.0-RAJA-ompOpt-mpi-gradient.exe
  PRIVATE -mllvm -enzyme-ompOpt-opt=1
  ${lulesh_compile_options})
endif ()




if (ENABLE_CUDA)
  configure_file(
    lulesh.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-cuda.cpp COPYONLY)

  configure_file(
    lulesh-comm.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-comm-cuda.cpp COPYONLY)
  configure_file(
    lulesh-init.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-init-cuda.cpp COPYONLY)
  configure_file(
    lulesh-util.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-util-cuda.cpp COPYONLY)
  configure_file(
    lulesh-viz.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-viz-cuda.cpp COPYONLY)

  blt_add_executable(
    NAME lulesh-v2.0-RAJA-cuda.exe
    SOURCES 
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-cuda.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-comm-cuda.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-init-cuda.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-util-cuda.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lulesh-viz-cuda.cpp
    DEPENDS_ON ${lulesh_depends})

  target_compile_options(
    lulesh-v2.0-RAJA-cuda.exe
    PRIVATE
    ${lulesh_compile_options})

  target_compile_definitions(
    lulesh-v2.0-RAJA-cuda.exe
    PRIVATE
    -DUSE_CASE=9)

  target_compile_definitions(
    lulesh-v2.0-RAJA-cuda.exe
    PRIVATE 
    -DUSE_MPI=0)

  target_compile_definitions(
    lulesh-v2.0-RAJA-cuda.exe
    PUBLIC
    -DLULESH_DEVICE=__device__
  )

  target_include_directories(
    lulesh-v2.0-RAJA-cuda.exe
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
endif ()
