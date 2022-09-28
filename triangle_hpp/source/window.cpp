#include "window.hpp"

namespace triangle_hpp {

Window::Window() : window{createWindow(), &destroyWindow} {}

GLFWwindow *Window::createWindow() {
  glfwInit();

  // Disable OpenGL in GLFW
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  // For now, disallow sizing of windows
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  // Width, Height, window name, monitor, unused(OpenGL only)
  return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void Window::destroyWindow(GLFWwindow *window) {
  glfwDestroyWindow(window);

  glfwTerminate();
}

void Window::run() {
  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
  }
}

std::span<const char *> Window::getRequiredExtensions() {
  std::uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  return {glfwExtensions, glfwExtensionCount};
}

} // namespace triangle_hpp