# Eigen provides the sparse direct solver backends (SparseLU, SimplicialLLT/LDLT).
#
# Eigen is header-only, so we only download the sources and expose them through
# an interface target, deliberately skipping Eigen's own CMakeLists.txt (which
# would add doc/test/install targets and tweak global compiler flags). Pointing
# SOURCE_SUBDIR at a directory without a CMakeLists.txt makes
# FetchContent_MakeAvailable skip the add_subdirectory step.
#
# Some of Eigen's optional external solver backends (e.g. PaStiX, PARDISO)
# may be wired up here in the future.
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR cmake_subdir_intentionally_absent
)
FetchContent_MakeAvailable(eigen)

add_library(eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS eigen)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
