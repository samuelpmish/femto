enable_testing()

file(GLOB_RECURSE cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)
list(FILTER cpp_tests EXCLUDE REGEX "disabled")

if (FEMTO_ENABLE_PCH)
  add_executable(pch ${PROJECT_SOURCE_DIR}/src/pch.cc)
  target_link_libraries(pch PUBLIC femto GTest::gtest_main)
  target_precompile_headers(pch PRIVATE ${PROJECT_SOURCE_DIR}/inc/pch.hpp)
endif()

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC femto GTest::gtest_main)
  target_include_directories(${testname} PUBLIC ${PROJECT_SOURCE_DIR}/tests)
  if (FEMTO_ENABLE_PCH)
    target_precompile_headers(${testname} REUSE_FROM pch)
  endif()

  if (FEMTO_SAVE_TEMPS)
    target_compile_options(${testname} PUBLIC -save-temps)
  endif()

  if (NOT ${testname} MATCHES "perf")
    add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		
  endif()

endforeach(filename ${cpp_tests})

if (FEMTO_ENABLE_CUDA)

  file(GLOB_RECURSE cu_tests ${PROJECT_SOURCE_DIR}/tests/*.cu)

  foreach(filename ${cu_tests})		

    get_filename_component(testname ${filename} NAME_WE)

    add_executable(cuda_${testname} ${filename})
    target_link_libraries(cuda_${testname} PUBLIC femto GTest::gtest_main)
    target_include_directories(cuda_${testname} PUBLIC ${PROJECT_SOURCE_DIR}/tests)

    if (NOT ${testname} MATCHES "perf")
      add_test(cuda_${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda_${testname})		
    endif()

  endforeach(filename ${cpp_tests})
endif()