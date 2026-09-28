# experimental support for circle (https://www.circle-lang.org), e.g.
#
#   cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=circle -DFEMTO_ENABLE_CUDA=ON ...
#
# CMake identifies circle as Clang, which is close enough for plain C++ but
# not for the Clang-specific features below. circle doesn't compile C, so
# leave CMAKE_C_COMPILER as gcc/clang.
#
# circle can't be CMake's CUDA compiler (compiler identification fails), and
# doesn't need to be: it's a single-source compiler that switches on CUDA with
# -sm_XX. So with circle, CMake's CUDA language stays disabled and the .cu
# files are compiled as CXX sources with the -sm_XX flags added.

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE _circle_version ERROR_QUIET)
if (NOT _circle_version MATCHES "^Circle")
  return()
endif()
set(FEMTO_USING_CIRCLE ON)
message(STATUS "femto: using circle as the C++/CUDA compiler")

# no -Xclang -emit-pch / -include-pch
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)

# no -flto, but some dependencies (nanothread) turn on INTERPROCEDURAL_OPTIMIZATION
set(CMAKE_CXX_COMPILE_OPTIONS_IPO "")
set(CMAKE_CXX_LINK_OPTIONS_IPO "")

# no -ftime-trace (the normal variable shadows the option() declared later)
if (FEMTO_TIME_TRACE)
  message(WARNING "circle has no -ftime-trace, ignoring FEMTO_TIME_TRACE")
  set(FEMTO_TIME_TRACE OFF)
endif()

if (FEMTO_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # otherwise cuBQL enables CMake's CUDA language itself (with nvcc)
  set(CUBQL_DISABLE_CUDA ON)
endif()

################################################################################

# -sm_XX flags from CMAKE_CUDA_ARCHITECTURES, e.g. "80;90-real" -> -sm_80;-sm_90
function(_circle_sm_flags out)
  set(archs ${CMAKE_CUDA_ARCHITECTURES})
  if (archs STREQUAL "native")
    # what CMake itself runs to resolve "native" for nvcc
    execute_process(COMMAND ${CUDAToolkit_BIN_DIR}/__nvcc_device_query OUTPUT_VARIABLE archs OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "," ";" archs "${archs}")
    list(REMOVE_DUPLICATES archs)
  endif()
  set(flags "")
  foreach(arch IN LISTS archs)
    if (NOT arch MATCHES "^([0-9]+[af]?)(-real|-virtual)?$")
      message(FATAL_ERROR "circle: unsupported CMAKE_CUDA_ARCHITECTURES entry '${arch}', use e.g. 80 or 80;90")
    endif()
    list(APPEND flags -sm_${CMAKE_MATCH_1})
  endforeach()
  set(${out} ${flags} PARENT_SCOPE)
endfunction()

# compile the .cu sources of every target in `dir` (and below) as circle CUDA
function(_circle_cuda_sources dir flags)
  get_property(targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t IN LISTS targets)
    get_target_property(srcs ${t} SOURCES)
    list(FILTER srcs INCLUDE REGEX "\\.cu$")
    if (srcs)
      set_source_files_properties(${srcs} TARGET_DIRECTORY ${t} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${flags}")
      target_link_libraries(${t} PUBLIC CUDA::cudart)
    endif()
  endforeach()
  get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir IN LISTS subdirs)
    _circle_cuda_sources(${subdir} "${flags}")
  endforeach()
endfunction()

# runs once the whole project (incl. dependencies, tests, examples) is configured
function(_circle_fixup)
  # nanothread's cmake-defaults add these for GNU|Clang, circle rejects them
  if (TARGET nanothread)
    get_target_property(opts nanothread COMPILE_OPTIONS)
    list(REMOVE_ITEM opts "$<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno>" "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>")
    set_target_properties(nanothread PROPERTIES COMPILE_OPTIONS "${opts}")
    # circle lowers its 16-byte CAS to a libatomic call (nanothread only links it for GNU)
    target_link_libraries(nanothread PRIVATE atomic)
  endif()

  # gtest leaves this undefined for a __clang__ without __has_attribute(disable_tail_calls)
  if (TARGET gtest)
    target_compile_definitions(gtest PRIVATE "GTEST_NO_TAIL_CALL_=")
  endif()

  # cuBQL uses the GNU "\e" escape extension, circle only accepts "\033"
  # ponytail: patches the fetched source in place (idempotent), drop once fixed upstream
  if (cubql_SOURCE_DIR)
    set(header ${cubql_SOURCE_DIR}/cuBQL/math/common.h)
    file(READ ${header} contents)
    string(REPLACE "\\e[" "\\033[" patched "${contents}")
    if (NOT patched STREQUAL contents)
      file(WRITE ${header} "${patched}")
    endif()
  endif()

  if (FEMTO_ENABLE_CUDA)
    _circle_sm_flags(sm_flags)
    message(STATUS "femto: compiling .cu sources with circle ${sm_flags}")
    _circle_cuda_sources(${CMAKE_SOURCE_DIR} "${sm_flags};-cuda-path=${CUDAToolkit_LIBRARY_ROOT}")
  endif()
endfunction()
cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _circle_fixup)
