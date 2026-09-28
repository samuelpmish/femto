# emscripten ships its own zlib port: asking for it on the compile and link
# lines is all that's needed, no download or system library involved
if (EMSCRIPTEN)
  add_library(zlib_emscripten INTERFACE)
  target_compile_options(zlib_emscripten INTERFACE "-sUSE_ZLIB=1")
  target_link_options(zlib_emscripten INTERFACE "-sUSE_ZLIB=1")
  add_library(ZLIB::ZLIB ALIAS zlib_emscripten)
  return()
endif()

find_package(ZLIB 1.2.3) # defines imported target ZLIB::ZLIB
if (NOT ZLIB_FOUND) 
  set(zlib_patch_command git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/zlib.patch)
  FetchContent_Declare(
    zlib
    #URL https://github.com/madler/zlib/releases/download/v1.3/zlib-1.3.tar.gz
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3
    GIT_PROGRESS TRUE
    PATCH_COMMAND ${zlib_patch_command}
    UPDATE_DISCONNECTED 1
  )
  FetchContent_MakeAvailable(zlib)
  add_library(ZLIB::ZLIB ALIAS zlibstatic)
  target_include_directories(zlibstatic INTERFACE ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR})
endif()