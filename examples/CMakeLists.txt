# the 2D mesh generator the examples that mesh on the fly link
include(${PROJECT_SOURCE_DIR}/cmake/triangle.cmake)

if (EMSCRIPTEN)
  add_subdirectory(html)
else()
  add_subdirectory(cpp)
endif()
