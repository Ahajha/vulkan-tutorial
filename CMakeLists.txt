cmake_minimum_required(VERSION 3.15)
project(vulkan-tutorial CXX)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

foreach(TARGET environment_test hello_triangle)
    add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET}.cpp")
    target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan glfw glm::glm)
    target_compile_features(${TARGET} PRIVATE cxx_std_20)
    if (NOT MSVC)
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

find_package(shaderc REQUIRED)

# Provided a base filename, reads `shaders/infile`
# and compiles the shader to spv/outfile when it changes.
macro(compile_shader infile outfile)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/spv/${outfile}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${infile}"
        COMMAND "${shaderc_INCLUDE_DIR}/../bin/glslc" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${infile}" -o "${CMAKE_CURRENT_SOURCE_DIR}/spv/${outfile}"
        COMMENT "Compiling ${infile}"
    )
endmacro()

compile_shader(shader.frag frag.spv)
compile_shader(shader.vert vert.spv)

target_sources(hello_triangle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/spv/frag.spv"
    "${CMAKE_CURRENT_SOURCE_DIR}/spv/vert.spv"
)
