#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "config.hpp"

#include <memory>
#include <span>

namespace triangle_hpp {
class Window {
public:
  Window();

  void run();

  static std::span<const char *> getRequiredExtensions();

private:
  static GLFWwindow *createWindow();
  static void destroyWindow(GLFWwindow *);

  std::unique_ptr<GLFWwindow, decltype(&destroyWindow)> window;
};
} // namespace triangle_hpp