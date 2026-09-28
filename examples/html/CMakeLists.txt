# the physics examples in the browser.  Every source in ../cpp is compiled into
# ONE webassembly module that all the pages share: each one's main() is
# compiled out and its EMSCRIPTEN_BINDINGS exposes its simulation object
# instead (see ../cpp/example_common.hpp).
#
# One module rather than one per page because the examples are the small part
# of it: femto, PaStiX and OpenBLAS come to ~2.3 MB, an example adds ~0.2 MB.
# Six separate modules were 16 MB on the server and 2.5 MB per page; the shared
# one is 3.1 MB, downloaded once and cached across the pages.
#
#   source /path/to/emsdk/emsdk_env.sh
#   emcmake cmake -B build-web-mt -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-web-mt --target femto_examples_html
# then serve build-web-mt/examples/html/ with python3 serve.py and open
# index.html.  The module is multithreaded, so the page has to be cross-origin
# isolated: serve.py sends the headers, and enable-threads.js installs a
# service worker that adds them where a host cannot (both are staged next to
# the pages).  node smoke_examples.js runs every example headlessly.
#
# To add an example: write ../cpp/<name>.cpp and <name>.html here, and add
# <name> to FEMTO_HTML_EXAMPLES below -- that puts it in the module and stages
# its page.  A mesh it loads goes in FEMTO_HTML_MESHES.

include(${PROJECT_SOURCE_DIR}/cmake/wasm_module.cmake)

set(FEMTO_HTML_EXAMPLES capacitor_2D heatsink_3D fracture_2D coffee_cup_modal magnetostatics_3D lid_driven_cavity_2D wave_disk_2D)

# meshes the examples load, staged next to the pages rather than embedded in
# the module: a page fetches only the one it needs and writes it into the
# module's filesystem (see loadMesh in viewer.js), which keeps the shared
# module small and a visit to one page from paying for every mesh
set(FEMTO_HTML_MESHES fracture_specimen.msh induction_coil.msh
                      to220_bare.msh to220_clip.msh to220_channel.msh to220_multiwatt.msh)

# the shared viewer, the smoke test, and the cross-origin isolation the
# threaded module needs
set(pages index.html viewer.js viewer.css smoke_examples.js enable-threads.js serve.py)
set(sources "")
foreach(example ${FEMTO_HTML_EXAMPLES})
  list(APPEND sources ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/${example}.cpp)
  list(APPEND pages ${example}.html)
endforeach()

add_executable(femto_examples_web ${sources})
set_target_properties(femto_examples_web PROPERTIES OUTPUT_NAME femto_examples)
femto_wasm_module(femto_examples_web createFemtoExamples)

set(staged femto_examples_web)
foreach(mesh_file ${FEMTO_HTML_MESHES})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${mesh_file}
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/data/meshes/${mesh_file} ${CMAKE_CURRENT_BINARY_DIR}/${mesh_file}
    DEPENDS ${PROJECT_SOURCE_DIR}/data/meshes/${mesh_file}
    COMMENT "Copying ${mesh_file}"
  )
  list(APPEND staged ${CMAKE_CURRENT_BINARY_DIR}/${mesh_file})
endforeach()
foreach(page_file ${pages})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${page_file}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${page_file} ${CMAKE_CURRENT_BINARY_DIR}/${page_file}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${page_file}
    COMMENT "Copying ${page_file}"
  )
  list(APPEND staged ${CMAKE_CURRENT_BINARY_DIR}/${page_file})
endforeach()

add_custom_target(femto_examples_html ALL DEPENDS ${staged})
