#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include <span>

namespace triangle_hpp {

class Instance {
public:
  Instance(std::span<const char *> requiredExtensions);

private:
  vk::raii::Context context;
  vk::raii::Instance instance;

  vk::raii::Instance createInstance(std::span<const char *> requiredExtensions);
};

} // namespace triangle_hpp