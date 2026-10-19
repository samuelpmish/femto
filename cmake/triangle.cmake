# Shewchuk's Triangle, the 2D quality mesh generator, for the examples that
# mesh their geometry on the fly (examples/cpp/hall_sensor_2D.cpp).  Linked
# into those examples only, never into femto: Triangle is free for private,
# research and institutional use, but distributing it in a commercial product
# needs the author's permission (see the README in its source).
#
# EXTERNAL_TEST makes the -u switch call a triunsuitable() the example
# defines, so the element size follows a size function in one refinement
# pass.  An example that links this library has to define one.

FetchContent_Declare(
  triangle
  URL https://www.netlib.org/voronoi/triangle.zip
  URL_HASH SHA256=1766327add038495fa3499e9b7cc642179229750f7201b94f8e1b7bee76f8480
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(triangle)

add_library(triangle STATIC ${triangle_SOURCE_DIR}/triangle.c)
target_include_directories(triangle PUBLIC ${triangle_SOURCE_DIR})
target_compile_definitions(triangle PUBLIC ANSI_DECLARATORS PRIVATE TRILIBRARY NO_TIMER EXTERNAL_TEST)
target_compile_options(triangle PRIVATE -w)   # 2005-era C
