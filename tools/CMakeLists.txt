file(GLOB cpp_tools ${PROJECT_SOURCE_DIR}/tools/*.cpp)

add_executable(tools_pch ${PROJECT_SOURCE_DIR}/src/pch.cc)
target_link_libraries(tools_pch PRIVATE femto)
target_precompile_headers(tools_pch PRIVATE ${PROJECT_SOURCE_DIR}/inc/pch.hpp)

foreach(filename ${cpp_tools})		

  get_filename_component(toolname ${filename} NAME_WE)
  add_executable(${toolname} ${filename})
  target_link_libraries(${toolname} PUBLIC femto)
  target_precompile_headers(${toolname} REUSE_FROM tools_pch)

endforeach(filename ${cpp_tools})