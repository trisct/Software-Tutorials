# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (Tutorials)

find_package(OpenGL REQUIRED)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()


set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

# Tutorial 1
add_executable(cyclicly_occluding_triangles cyclicly_occluding_triangles.cpp ../common/shader.cpp)
target_include_directories(cyclicly_occluding_triangles PRIVATE ../)
target_link_libraries(cyclicly_occluding_triangles ${ALL_LIBS})
set_target_properties(cyclicly_occluding_triangles PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")