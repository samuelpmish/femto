# PaStiX (https://gitlab.inria.fr/solverstack/pastix) provides multithreaded
# sparse direct solvers that, unlike MKL, can be built entirely from source,
# so it is the default backend for webassembly builds and the fallback CPU
# backend whenever MKL is unavailable (Eigen's built-in solvers are only
# used when neither is compiled in). The full stack (OpenBLAS for
# BLAS/LAPACKE, Scotch for ordering, then PaStiX itself) is downloaded and
# built here with no system dependencies.
#
# note: OpenBLAS is built for a GENERIC target (portable, required when
# cross-compiling to wasm), so a native FEMTO_ENABLE_PASTIX build trades
# some BLAS performance for reproducibility.
if (EMSCRIPTEN)
  set(FEMTO_PASTIX_DEFAULT ON)
else()
  set(FEMTO_PASTIX_DEFAULT OFF)
endif()
option(FEMTO_ENABLE_PASTIX "build the PaStiX sparse direct solver backend (from source)" ${FEMTO_PASTIX_DEFAULT})

# MKL_FOUND comes from mkl.cmake (included first); unset when MKL is disabled
if (NOT FEMTO_ENABLE_PASTIX AND NOT MKL_FOUND)
  message(STATUS "MKL not available: building PaStiX from source as the CPU sparse direct solver backend")
  set(FEMTO_ENABLE_PASTIX ON)
endif()

if (FEMTO_ENABLE_PASTIX)
  include(ExternalProject)

  set(PASTIX_STACK_PREFIX ${CMAKE_BINARY_DIR}/_deps/pastix-stack)

  set(pastix_stack_args
    -DCMAKE_INSTALL_PREFIX=${PASTIX_STACK_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )
  if (CMAKE_TOOLCHAIN_FILE)
    list(APPEND pastix_stack_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
  endif()
  if (EMSCRIPTEN AND FEMTO_WASM_THREADS)
    # wasm objects must be compiled with -pthread to link into a threaded
    # module (they need the atomics/bulk-memory features)
    list(APPEND pastix_stack_args -DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread)
  endif()

  # OpenBLAS keys its arch detection off CMAKE_SYSTEM_PROCESSOR, which
  # emscripten reports as x86 -- an arch whose kernels are assembly. Masquerade
  # as riscv64 to get the portable C kernel set instead.
  if (EMSCRIPTEN)
    set(openblas_target_args -DEMSCRIPTEN_SYSTEM_PROCESSOR=riscv64 -DTARGET=RISCV64_GENERIC)
  else()
    set(openblas_target_args -DTARGET=GENERIC)
  endif()

  # serial BLAS (PaStiX supplies the parallelism), with locking so that its
  # workspace buffers survive concurrent calls from PaStiX's worker threads
  ExternalProject_Add(openblas_external
    URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz
    CMAKE_ARGS ${pastix_stack_args} ${openblas_target_args}
      -DNOFORTRAN=ON -DC_LAPACK=ON -DBUILD_LAPACK_DEPRECATED=OFF
      -DUSE_THREAD=OFF -DUSE_LOCKING=ON -DDYNAMIC_ARCH=OFF
      -DBUILD_TESTING=OFF
    # OpenBLAS adds its BLAS test binaries to `all` regardless of
    # BUILD_TESTING (and one of them can't link against emscripten's fixed
    # initial memory in -pthread builds), so build just the library
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target openblas_static
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    BUILD_BYPRODUCTS ${PASTIX_STACK_PREFIX}/lib/libopenblas.a
  )

  # wasm builds run the solver single-threaded (see
  # sparse_factorization_pastix.cpp), so scotch's threading is disabled there
  if (EMSCRIPTEN)
    set(scotch_threads OFF)
  else()
    set(scotch_threads ON)
  endif()
  ExternalProject_Add(scotch_external
    URL https://gitlab.inria.fr/scotch/scotch/-/archive/v7.0.14/scotch-v7.0.14.tar.gz
    PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scotch_patch.cmake
    CMAKE_ARGS ${pastix_stack_args}
      -DBUILD_PTSCOTCH=OFF -DBUILD_LIBESMUMPS=OFF -DBUILD_LIBSCOTCHMETIS=OFF
      -DINSTALL_METIS_HEADERS=OFF -DBUILD_FORTRAN=OFF -DENABLE_TESTS=OFF
      -DTHREADS=${scotch_threads}
    BUILD_BYPRODUCTS ${PASTIX_STACK_PREFIX}/lib/libscotch.a
                     ${PASTIX_STACK_PREFIX}/lib/libscotcherr.a
  )

  if (EMSCRIPTEN)
    # emscripten restricts find_path/find_library to its own sysroot: let the
    # pastix configure also see the cblas/lapacke/scotch installs above
    set(pastix_find_args -DCMAKE_FIND_ROOT_PATH=${PASTIX_STACK_PREFIX})
  else()
    set(pastix_find_args)
  endif()

  ExternalProject_Add(pastix_external
    GIT_REPOSITORY https://gitlab.inria.fr/solverstack/pastix.git
    # post-v6.4.0 main branch: the v6.4.0 release always builds (and installs)
    # its examples and tests, some of which don't link without hwloc
    GIT_TAG ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2
    # fixes core-count detection (upstream tests PASTIX_OS_* macros that are
    # never defined) and silences a warning printed on every pastixInit()
    PATCH_COMMAND git reset --hard && git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pastix.patch
    CMAKE_ARGS ${pastix_stack_args} ${pastix_find_args}
      -DCMAKE_PREFIX_PATH=${PASTIX_STACK_PREFIX}
      -DPASTIX_WITH_MPI=OFF -DPASTIX_WITH_CUDA=OFF
      -DPASTIX_WITH_FORTRAN=OFF -DSPM_WITH_FORTRAN=OFF
      -DPASTIX_INT64=OFF -DSPM_INT64=OFF
      -DPASTIX_ORDERING_SCOTCH=ON
      -DPASTIX_BUILD_EXAMPLES=OFF -DPASTIX_BUILD_TESTS=OFF
      -DCMAKE_DISABLE_FIND_PACKAGE_HWLOC=ON
    DEPENDS openblas_external scotch_external
    BUILD_BYPRODUCTS ${PASTIX_STACK_PREFIX}/lib/libpastix.a
                     ${PASTIX_STACK_PREFIX}/lib/libpastix_kernels.a
                     ${PASTIX_STACK_PREFIX}/lib/libspm.a
  )

  add_library(pastix_stack INTERFACE)
  # pastix and spm install their headers into versioned subdirectories
  # (stable, since the pastix commit above is pinned)
  target_include_directories(pastix_stack SYSTEM INTERFACE
    ${PASTIX_STACK_PREFIX}/include
    ${PASTIX_STACK_PREFIX}/include/pastix/6.4
    ${PASTIX_STACK_PREFIX}/include/spm/1.2
  )
  target_link_libraries(pastix_stack INTERFACE
    ${PASTIX_STACK_PREFIX}/lib/libpastix.a
    ${PASTIX_STACK_PREFIX}/lib/libpastix_kernels.a
    ${PASTIX_STACK_PREFIX}/lib/libspm.a
    ${PASTIX_STACK_PREFIX}/lib/libscotch.a
    ${PASTIX_STACK_PREFIX}/lib/libscotcherr.a
    ${PASTIX_STACK_PREFIX}/lib/libopenblas.a
    m
  )
  if (NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
    target_link_libraries(pastix_stack INTERFACE Threads::Threads)
  endif()

  target_compile_definitions(femto PUBLIC FEMTO_ENABLE_PASTIX)
  target_link_libraries(femto PRIVATE pastix_stack)
  add_dependencies(femto pastix_external)
endif()
