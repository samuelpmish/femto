if (NDARRAY_ENABLE_CODE_COVERAGE) 
  include(CTest)

  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
    message(SEND_ERROR "code coverage only currently supported with gcc/g++")
  else()
    target_compile_options(ndarray INTERFACE -fprofile-arcs)
    target_compile_options(ndarray INTERFACE -ftest-coverage)
    target_link_options(ndarray INTERFACE -fprofile-arcs)
    target_link_options(ndarray INTERFACE -ftest-coverage)
  endif()

  add_custom_target(clean-gcov
    COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -exec rm {} "\;"
    COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcno" -exec rm {} "\;"
    COMMENT "Cleaning gcov files"
  )
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC ${CMAKE_PROJECT_NAME} GTest::gtest_main)

  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cpp_tests})

if (NDARRAY_ENABLE_CUDA) 

file(GLOB_RECURSE cu_tests ${PROJECT_SOURCE_DIR}/tests/*.cu)
foreach(filename ${cu_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname}_cuda ${filename})
  target_link_libraries(${testname}_cuda PUBLIC ${CMAKE_PROJECT_NAME} GTest::gtest_main)

  add_test(${testname}_cuda ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cu_tests})

endif()
