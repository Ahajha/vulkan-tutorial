add_library(triangle_hpp_lib)
target_link_libraries(triangle_hpp_lib PUBLIC Vulkan::Vulkan glfw glm::glm)
target_include_directories(triangle_hpp_lib PUBLIC include)
target_sources(triangle_hpp_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/window.cpp"
)

add_executable(triangle_hpp main.cpp)
target_link_libraries(triangle_hpp PRIVATE triangle_hpp_lib)