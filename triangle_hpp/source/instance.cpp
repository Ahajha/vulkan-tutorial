#include "instance.hpp"
#include "config.hpp"

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <iostream>

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              [[maybe_unused]] void *pUserData) {

  std::cerr << "validation layer: " << pCallbackData->pMessage
            << "\n\tSeverity: " << messageSeverity
            << "\n\tType: " << messageType << std::endl;

  return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT createDebugUtilsMessengerCreateInfo() {
  using enum vk::DebugUtilsMessageSeverityFlagBitsEXT;
  using enum vk::DebugUtilsMessageTypeFlagBitsEXT;
  return {
      .messageSeverity = eVerbose | eWarning | eError,
      .messageType = eGeneral | eValidation | ePerformance,
      .pfnUserCallback = debugCallback,
  };
}

} // namespace

namespace triangle_hpp {

Instance::Instance(std::span<const char *> requiredExtensions)
    : instance{createInstance(requiredExtensions)} {}

vk::raii::Instance
Instance::createInstance(std::span<const char *> requiredExtensions) {
  // Set app info (optional)
  const vk::ApplicationInfo appInfo{
      .pApplicationName = "Triangle - hpp",
      .applicationVersion = VK_MAKE_API_VERSION(1, 0, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_API_VERSION(1, 0, 0, 0),
      .apiVersion = VK_API_VERSION_1_0,
  };

  // Specify the required extensions
  std::vector<const char *> extensions(requiredExtensions.begin(),
                                       requiredExtensions.end());
  if constexpr (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // Initialize the vk::InstanceCreateInfo
  vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount =
          static_cast<std::uint32_t>(requiredExtensions.size()),
      .ppEnabledExtensionNames = requiredExtensions.data(),
  };

  vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if constexpr (enableValidationLayers) {
    // if (!checkValidationLayerSupport()) {
    //   throw std::runtime_error(
    //       "validation layers requested, but not available!");
    // } else {
    createInfo.enabledLayerCount =
        static_cast<std::uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    debugCreateInfo = createDebugUtilsMessengerCreateInfo();
    createInfo.pNext = &debugCreateInfo;

    std::cerr << "validation layers enabled\n";
  }

  return vk::raii::Instance{context, createInfo};
}

} // namespace triangle_hpp