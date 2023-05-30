cmake_minimum_required(VERSION 3.15)
project(vulkan-tutorial CXX)

option(ENABLE_VALIDATION_LAYERS "Enable Vulkan validation layers" ON)

find_package(glfwpp REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)
find_package(stb REQUIRED)

# STB requires the users of the package to compile the library
add_library(stb_impl STATIC src/stb_impl.cpp)
target_link_libraries(stb_impl PUBLIC stb::stb)

foreach(TARGET environment_test hello_triangle)
    add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET}.cpp")
    target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan glfwpp::glfwpp glm::glm stb_impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_20)
    if (NOT MSVC)
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
    endif()
    if (${ENABLE_VALIDATION_LAYERS})
        target_compile_definitions(${TARGET} PUBLIC ENABLE_VALIDATION_LAYERS)
    endif()
endforeach()

find_package(shaderc REQUIRED)

# Provided a base filename, reads `shaders/infile`
# and compiles the shader to ${CMAKE_BUILD_DIR}/shaders/outfile when it changes.
macro(compile_shader infile outfile)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/shaders/${outfile}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${infile}"
        COMMAND "${shaderc_INCLUDE_DIR}/../bin/glslc" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${infile}" -o "${CMAKE_BINARY_DIR}/shaders/${outfile}"
        COMMENT "Compiling ${infile}"
    )
endmacro()

compile_shader(shader.frag frag.spv)
compile_shader(shader.vert vert.spv)

target_sources(hello_triangle PRIVATE
    "${CMAKE_BINARY_DIR}/shaders/frag.spv"
    "${CMAKE_BINARY_DIR}/shaders/vert.spv"
)

#add_subdirectory(triangle_hpp)
