cmake_minimum_required(VERSION 3.24)
project(femto LANGUAGES C CXX)
include(cmake/circle_support.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_PREFIX_PATH "${DEP_PATH}/install/lib/cmake")

# webassembly builds (cmake configured through emscripten's emcmake wrapper)
# are CPU-only and skip the parts of femto that need sockets (femto::server).
#
# They use pthreads (Web Workers + SharedArrayBuffer) by default, so the thread
# pool works as it does natively. That needs the page to be cross-origin
# isolated: serve it with examples/html/serve.py, or let
# examples/html/enable-threads.js install the service worker that adds the
# headers on a static host like GitHub Pages.
# Configure with -DFEMTO_WASM_THREADS=OFF for a single-threaded module that any
# file server can host.
option(FEMTO_WASM_THREADS "build the webassembly target with pthread support" ON)
if (EMSCRIPTEN)
  set(FEMTO_ENABLE_CUDA OFF)
  if (FEMTO_WASM_THREADS)
    add_compile_options(-pthread)
    add_link_options(-pthread)
  endif()
endif()

if (FEMTO_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  if (NOT FEMTO_USING_CIRCLE)
    enable_language(CUDA)
  endif()
  # the constexpr shape function / quadrature point helpers in the element
  # headers are called from device code
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set(NDARRAY_ENABLE_CUDA ON)
endif()

option(FEMTO_TIME_TRACE "Emit compiler time traces/reports for C++ and CUDA sources" OFF)

################################################################################

file(GLOB_RECURSE femto_sources ${PROJECT_SOURCE_DIR}/inc/*.hpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
if (EMSCRIPTEN)
  list(REMOVE_ITEM femto_sources ${PROJECT_SOURCE_DIR}/src/server.cpp)
endif()
add_library(femto ${femto_sources})
target_include_directories(femto PUBLIC ${PROJECT_SOURCE_DIR}/inc)
target_include_directories(femto PUBLIC ${DEP_INSTALL_PATH}/include)

if (FEMTO_ENABLE_CUDA)
  file(GLOB_RECURSE femto_cu_sources ${PROJECT_SOURCE_DIR}/src/*.cu)
  target_sources(femto PRIVATE ${femto_cu_sources})
  target_include_directories(femto PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(femto PUBLIC FEMTO_ENABLE_CUDA NDARRAY_ENABLE_CUDA)
  target_compile_options(femto PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

  # cuDSS backs the device-memory sparse direct solver (sparse_factorization_cudss.cu)
  find_package(cudss REQUIRED)
  target_link_libraries(femto PUBLIC cudss)
endif()

if (NOT EMSCRIPTEN)
  target_precompile_headers(femto PRIVATE ${PROJECT_SOURCE_DIR}/src/pch.hpp)
endif()

if (FEMTO_ENABLE_PCH)
  target_precompile_headers(femto PRIVATE ${PROJECT_SOURCE_DIR}/inc/pch.hpp)
endif()

include(FetchContent)
include(cmake/fm.cmake)
include(cmake/zlib.cmake)
include(cmake/eigen.cmake)
include(cmake/mkl.cmake)
include(cmake/pastix.cmake)
include(cmake/cubql.cmake)

target_link_libraries(femto PUBLIC fm ZLIB::ZLIB)
target_link_libraries(femto PRIVATE Eigen3::Eigen cuBQL)

if (EMSCRIPTEN)
  if (NOT FEMTO_WASM_THREADS)
    target_compile_definitions(femto PUBLIC FEMTO_SINGLE_THREADED)
  endif()
  # stubbed <execinfo.h> for cuBQL, which assumes glibc when __GNUC__ is set
  target_include_directories(femto PRIVATE ${PROJECT_SOURCE_DIR}/cmake/shims)
else()
  include(cmake/nanothread.cmake)
  include(cmake/httplib.cmake)
  target_link_libraries(femto PUBLIC nanothread)
  target_link_libraries(femto PRIVATE httplib::httplib)
endif()
set_target_properties(femto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(femto PUBLIC "-DFEMTO_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data/\"")
target_compile_definitions(femto PUBLIC "-DFEMTO_MESH_DIR=\"${PROJECT_SOURCE_DIR}/data/meshes/\"")

if (APPLE)  
  # why do I have to set this manually
  # message(${CMAKE_SYSTEM_FRAMEWORK_PATH})
  target_link_options(femto PRIVATE "-Wl,-F/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/")

  target_link_options(femto PUBLIC -Wl,-no_compact_unwind)
  target_link_options(femto PUBLIC -Xlinker -no_warn_duplicate_libraries)
endif()

# prevent CMake error on macs: 
# Could NOT find Threads (missing: Threads_FOUND)
IF(APPLE)
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
ENDIF()

if (FEMTO_TIME_TRACE)
  target_compile_options(femto PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-ftime-trace>"
    #"$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-ftime-report>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--fdevice-time-trace=->"
  )
endif()

if (FEMTO_SAVE_TEMPS)
  target_compile_options(femto PUBLIC -save-temps)
endif()

if (FEMTO_ENABLE_ASAN)
  add_link_options(-fsanitize=address)
  add_compile_options(-fsanitize=address)
endif()

if (FEMTO_ENABLE_BOUNDS_CHECKING)
  target_compile_options(femto PUBLIC -DFEMTO_ENABLE_BOUNDS_CHECKING)
  target_compile_options(femto PUBLIC -DNDARRAY_ENABLE_BOUNDS_CHECKING)
endif()

# native executables (examples/cpp) or, under emscripten, the same physics
# examples as webassembly modules with browser viewers (examples/html)
if (FEMTO_ENABLE_EXAMPLES OR EMSCRIPTEN)
  add_subdirectory(examples)
endif()

if (FEMTO_ENABLE_TESTING)
  include(cmake/gtest.cmake)
  add_subdirectory(tests)
endif()

if (FEMTO_ENABLE_TOOLS)
  add_subdirectory(tools)
endif()
