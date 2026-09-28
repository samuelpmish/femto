# Scotch generates scotch.h at build time by *running* its dummysizes tool,
# which it invokes through $<TARGET_FILE:...>. That bypasses CMake's
# CMAKE_CROSSCOMPILING_EMULATOR (node, under emscripten), so cross builds
# fail trying to execute a .js file directly. Referring to the target by
# name makes CMake prepend the emulator automatically. Idempotent; run from
# the scotch source directory.
foreach(f src/libscotch/CMakeLists.txt src/check/CMakeLists.txt src/libscotchmetis/CMakeLists.txt)
  file(READ ${f} contents)
  string(REPLACE "$<TARGET_FILE:dummysizes>" "dummysizes" contents "${contents}")
  string(REPLACE "$<TARGET_FILE:ptdummysizes>" "ptdummysizes" contents "${contents}")
  file(WRITE ${f} "${contents}")
endforeach()

# emscripten sandboxes file i/o into an in-memory filesystem by default;
# dummysizes must read/write the real header files, so give it direct node
# filesystem access -- and EXIT_RUNTIME so stdio is flushed when main returns
file(READ src/libscotch/CMakeLists.txt contents)
set(marker "add_executable(dummysizes dummysizes.c)")
set(addition "${marker}
if (EMSCRIPTEN)
  target_link_options(dummysizes PRIVATE -sNODERAWFS=1 -sEXIT_RUNTIME=1)
endif()")
string(FIND "${contents}" "NODERAWFS" already_patched)
if (already_patched EQUAL -1)
  string(REPLACE "${marker}" "${addition}" contents "${contents}")
  file(WRITE src/libscotch/CMakeLists.txt "${contents}")
endif()
