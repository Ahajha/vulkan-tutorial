cmake_minimum_required(VERSION 3.15)
project(vulkan-tutorial CXX)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

foreach(TARGET environment_test hello_triangle)
    add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET}.cpp")
    target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan glfw glm::glm)
    target_compile_features(${TARGET} PRIVATE cxx_std_20)
endforeach()
