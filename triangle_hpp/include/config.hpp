#pragma once

#include <array>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace triangle_hpp {

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;

constexpr std::array validationLayers = {"VK_LAYER_KHRONOS_validation"};

constexpr std::array deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

} // namespace triangle_hpp