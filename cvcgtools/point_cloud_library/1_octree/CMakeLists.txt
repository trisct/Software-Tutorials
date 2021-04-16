project("PCL_octree")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(octree_search octree_search.cpp)
#target_include_directories(octree_search PRIVATE ${PCL_INCLUDE_DIRS})
#target_link_directories(pcd_compression PRIVATE ${PCL_LIBRARIES_DIRS})
target_link_libraries(octree_search PRIVATE ${PCL_LIBRARIES})
set_target_properties(octree_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")