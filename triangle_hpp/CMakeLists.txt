add_library(triangle_hpp_lib)
target_link_libraries(triangle_hpp_lib PUBLIC Vulkan::Vulkan glfw glm::glm)
target_include_directories(triangle_hpp_lib PUBLIC include)
target_sources(triangle_hpp_lib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/instance.cpp"
)
target_compile_features(triangle_hpp_lib PUBLIC cxx_std_20)
if (NOT MSVC)
    target_compile_options(triangle_hpp_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(triangle_hpp main.cpp)
target_link_libraries(triangle_hpp PRIVATE triangle_hpp_lib)