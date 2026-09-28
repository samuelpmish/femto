FetchContent_Declare(
  nanothread
  GIT_REPOSITORY https://github.com/mitsuba-renderer/nanothread.git
  GIT_TAG master
)
# nanothread's CMakeLists does find_library(atomic) under GCC. On Debian/Ubuntu
# the libatomic.so dev symlink lives only in the compiler's own directory
# (/usr/lib/gcc/<triple>/<version>/), which find_library does not search, so
# add the compiler's implicit link directories to the search path first.
list(APPEND CMAKE_LIBRARY_PATH ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
FetchContent_MakeAvailable(nanothread)
