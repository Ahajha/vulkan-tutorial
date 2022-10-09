#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "config.hpp"

#include <memory>

namespace triangle_hpp {
class Window {
public:
  Window();

  void run();

private:
  static GLFWwindow *createWindow();
  static void destroyWindow(GLFWwindow *);

  std::unique_ptr<GLFWwindow, decltype(&destroyWindow)> window;
};
} // namespace triangle_hpp