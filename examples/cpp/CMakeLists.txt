#file(GLOB_RECURSE cpp_examples ${PROJECT_SOURCE_DIR}/examples/*.cpp)
#
#foreach(filename ${cpp_examples})		
#  get_filename_component(example ${filename} NAME_WE)
#  add_executable(${example} ${filename})
#  target_link_libraries(${example} PUBLIC femto)
#endforeach(filename ${cpp_examples})

add_executable(hydro_2D hydro_2D.cpp)
target_link_libraries(hydro_2D PUBLIC femto)

add_executable(poisson_2D poisson_2D.cpp)
target_link_libraries(poisson_2D PUBLIC femto)

add_executable(elasticity_2D elasticity_2D.cpp)
target_link_libraries(elasticity_2D PUBLIC femto)

add_executable(elasticity_3D elasticity_3D.cpp)
target_link_libraries(elasticity_3D PUBLIC femto)

add_executable(viewer_demo viewer_demo.cpp)
target_link_libraries(viewer_demo PUBLIC femto)
# physics examples that also build as webassembly modules (see examples/html/CMakeLists.txt)
foreach(example capacitor_2D heatsink_3D fracture_2D coffee_cup_modal magnetostatics_3D lid_driven_cavity_2D wave_disk_2D)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PUBLIC femto)
endforeach()
