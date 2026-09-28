# NVIDIA/cuBQL: BVH construction and query library, used by femto::merge
# to find coincident vertices. Used header-only: src/mesh/merge.cu defines
# CUBQL_GPU_BUILDER_IMPLEMENTATION to instantiate the builders it needs, so
# we only link the `cuBQL` interface target (include paths, no libraries).

if (NOT FEMTO_ENABLE_CUDA)
  # keep cuBQL from probing for (and enabling) CUDA on its own
  set(CUBQL_DISABLE_CUDA ON)
endif()

FetchContent_Declare(
  cubql
  GIT_REPOSITORY https://github.com/NVIDIA/cuBQL.git
  GIT_TAG        20f9db19cdb160e2b83d8f05dfdb0437fe11fe24
)
FetchContent_MakeAvailable(cubql)
