# Intel MKL provides the default sparse direct solver backend (PARDISO),
# which is much faster than Eigen's built-in solvers. It is looked up
# through oneAPI's MKLConfig.cmake, e.g.
#
#   source /opt/intel/oneapi/setvars.sh          (oneAPI install), or
#   -DCMAKE_PREFIX_PATH=<venv>                   (pip install mkl-devel)
#
# When MKL isn't found, PaStiX is built from source as the CPU backend
# instead (see pastix.cmake) rather than failing to configure.
option(FEMTO_ENABLE_MKL "use MKL PARDISO as the sparse direct solver backend" ON)

if (FEMTO_ENABLE_MKL AND NOT EMSCRIPTEN)
  # Eigen's PARDISO wrappers use 32-bit indices, and gnu_thread (libgomp)
  # avoids a dependence on intel's OpenMP runtime (libiomp5); both
  # MKL_INTERFACE and MKL_THREADING can still be overridden on the command line
  if (NOT DEFINED MKL_INTERFACE)
    set(MKL_INTERFACE lp64)
  endif()
  if (NOT DEFINED MKL_THREADING)
    set(MKL_THREADING gnu_thread)
  endif()
  find_package(MKL CONFIG QUIET)
  if (MKL_FOUND)
    target_compile_definitions(femto PUBLIC FEMTO_ENABLE_MKL)
    target_link_libraries(femto PRIVATE MKL::MKL)
  else()
    message(STATUS "MKL not found: PaStiX will be built from source as the CPU sparse direct solver backend")
  endif()
endif()
