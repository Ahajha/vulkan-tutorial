#include "instance.hpp"
#include "window.hpp"

int main() {
  triangle_hpp::Window window;

  triangle_hpp::Instance instance{window.getRequiredExtensions()};

  window.run();
}