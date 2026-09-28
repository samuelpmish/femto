# cpp-httplib provides the embedded HTTP server used by femto::server
# (see inc/server.hpp) to stream meshes and solution fields to a browser

# don't let httplib pick up optional system libraries that would add
# link dependencies to femto (zlib is fine, femto already links it)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(httplib)
