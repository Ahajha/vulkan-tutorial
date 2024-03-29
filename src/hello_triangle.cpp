#include "stb_wrapper.hpp"

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>

// GLFWPP includes special functions if it detects that vulkan-hpp is included,
// so include after.
#define GLFW_INCLUDE_VULKAN
#include <glfwpp/glfwpp.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;

constexpr std::array deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef ENABLE_VALIDATION_LAYERS
constexpr std::array validationLayers = {"VK_LAYER_KHRONOS_validation"};
#endif

constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static constexpr vk::VertexInputBindingDescription getBindingDescription() {
    return {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  static constexpr std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    return {{
        {
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(Vertex, pos),
        },
        {
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, color),
        },
    }};
  }
};

const std::vector<Vertex> vertices{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
};

const std::vector<std::uint16_t> indices{
    0, 1, 2, 2, 3, 0,
};

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
  void run() { mainLoop(); }

private:
  // Initialized the GLFW window
  [[nodiscard]] glfw::Window initWindow() {
    glfw::WindowHints{.clientApi = glfw::ClientApi::None}.apply();
    glfw::Window window = {WIDTH, HEIGHT, "Vulkan"};

    window.framebufferSizeEvent.setCallback(
        [this](glfw::Window&, int, int) { framebufferResized = true; });

    return window;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                [[maybe_unused]] void* pUserData) {
    // This message is a known false positive with the validation layers, see
    // issue:
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1340
    if (pCallbackData->messageIdNumber == 2094043421) {
      return VK_FALSE;
    }

    std::cerr << "validation layer: " << pCallbackData->pMessage
              << "\n\tSeverity: " << messageSeverity
              << "\n\tType: " << messageType << std::endl;

    return VK_FALSE;
  }

  // Get a list of Vulkan extensions required/requested by the application
  [[nodiscard]] static std::vector<const char*> getRequiredExtensions() {
    // Extensions are needed for GLFW, we can query GLFW to get this info
    auto extensions = glfw::getRequiredInstanceExtensions();

#ifdef ENABLE_VALIDATION_LAYERS
    // Add "VK_EXT_debug_utils", macro is provided to prevent typos
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return extensions;
  }

  [[nodiscard]] constexpr static vk::DebugUtilsMessengerCreateInfoEXT
  createDebugMessengerCreateInfo() {
    using enum vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using enum vk::DebugUtilsMessageTypeFlagBitsEXT;
    return {
        .messageSeverity = eWarning | eError,
        .messageType = eGeneral | eValidation | ePerformance,
        .pfnUserCallback = debugCallback,
    };
  }

  // Creates the Vulkan instance
  [[nodiscard]] vk::raii::Instance createInstance() const {
    // Set app info (optional)
    const vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

#ifdef ENABLE_VALIDATION_LAYERS
    vk::StructureChain<vk::InstanceCreateInfo,
                       vk::DebugUtilsMessengerCreateInfoEXT>
        chain{vk::InstanceCreateInfo{.pApplicationInfo = &appInfo},
              createDebugMessengerCreateInfo()};

    auto& createInfo = chain.get<vk::InstanceCreateInfo>();
    createInfo.setPEnabledLayerNames(validationLayers);
    std::cerr << "validation layers enabled\n";
#else
    vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo};
#endif

    const auto requiredExtensions = getRequiredExtensions();
    createInfo.setPEnabledExtensionNames(requiredExtensions);

    return {m_context, createInfo};
  }

  struct QueueFamilyIndices {
    std::uint32_t graphicsFamily;
    std::uint32_t presentFamily;
  };

  struct OptionalQueueFamilyIndices {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }

    [[nodiscard]] std::optional<QueueFamilyIndices> finalize() const {
      if (isComplete()) {
        return QueueFamilyIndices{
            .graphicsFamily = *(this->graphicsFamily),
            .presentFamily = *(this->presentFamily),
        };
      } else {
        return {};
      }
    }
  };

  [[nodiscard]] OptionalQueueFamilyIndices
  findQueueFamilies(vk::PhysicalDevice device) const {
    OptionalQueueFamilyIndices queueIndices;
    // Logic to find queue family indices to populate struct with

    const auto queueFamilies = device.getQueueFamilyProperties();

    std::uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        queueIndices.graphicsFamily = i;
      }

      if (device.getSurfaceSupportKHR(i, *m_surface)) {
        queueIndices.presentFamily = i;
      }

      if (queueIndices.isComplete()) {
        break;
      }

      i++;
    }

    return queueIndices;
  }

  // Returns true if deviceExtensions is a subset of the available device
  // extensions.
  [[nodiscard]] bool
  checkDeviceExtensionSupport(vk::PhysicalDevice device) const {
    const auto availableExtensions =
        device.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  // Contains information about swap chain support for a given physical device
  struct SwapChainSupportDetails {
    SwapChainSupportDetails(vk::PhysicalDevice device, vk::SurfaceKHR surface)
        : capabilities{device.getSurfaceCapabilitiesKHR(surface)}
        , formats{device.getSurfaceFormatsKHR(surface)}
        , presentModes{device.getSurfacePresentModesKHR(surface)} {}

    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };

  bool isDeviceSuitable(vk::PhysicalDevice device) const {
    if (!findQueueFamilies(device).isComplete() ||
        !checkDeviceExtensionSupport(device))
      return false;

    const auto swapChainSupport = SwapChainSupportDetails(device, *m_surface);
    return !swapChainSupport.formats.empty() &&
           !swapChainSupport.presentModes.empty();
  }

  [[nodiscard]] static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::span<const vk::SurfaceFormatKHR> availableFormats) {
    constexpr vk::SurfaceFormatKHR desiredFormat{
        .format = vk::Format::eB8G8R8A8Srgb,
        .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
    };

    const auto iter = std::ranges::find(availableFormats, desiredFormat);

    return iter != availableFormats.end() ? desiredFormat
                                          : availableFormats.front();
  }

  [[nodiscard]] static vk::PresentModeKHR chooseSwapPresentMode(
      const std::span<const vk::PresentModeKHR> availablePresentModes) {
    constexpr vk::PresentModeKHR desiredPresentMode =
        // vk::PresentModeKHR::eMailbox;
        // Something about my dev environment is strange, it stutters terribly
        // when using mailbox. Something related to high resolution + high
        // refresh rates.
        vk::PresentModeKHR::eFifo;

    const auto iter =
        std::ranges::find(availablePresentModes, desiredPresentMode);

    return iter != availablePresentModes.end() ? desiredPresentMode
                                               : vk::PresentModeKHR::eFifo;
  }

  [[nodiscard]] vk::Extent2D
  chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {

    if (capabilities.currentExtent.width !=
        std::numeric_limits<std::uint32_t>::max()) {
      // Not sure about the reasoning about this branch
      return capabilities.currentExtent;
    } else {
      const auto [width, height] = m_window.getFramebufferSize();

      const auto [minWidth, minHeight] = capabilities.minImageExtent;
      const auto [maxWidth, maxHeight] = capabilities.maxImageExtent;

      // Bound dimensions between the allowed min and max supported by the
      // implementation.
      // clang-format off
      return {
          .width = std::clamp(static_cast<std::uint32_t>(width), minWidth, maxWidth),
          .height = std::clamp(static_cast<std::uint32_t>(height), minHeight, maxHeight),
      };
      // clang-format on
    }
  }

  // Returns a suitable physical device
  [[nodiscard]] vk::raii::PhysicalDevice pickPhysicalDevice() const {
    const auto devices = m_instance.enumeratePhysicalDevices();

    auto iter = std::ranges::find_if(
        devices, [this](auto& device) { return isDeviceSuitable(*device); });

    if (iter == devices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }

    return *iter;
  }

  [[nodiscard]] vk::raii::Device createLogicalDevice() const {
    const std::set<std::uint32_t> uniqueQueueFamilies = {
        m_queueFamilyIndices.graphicsFamily,
        m_queueFamilyIndices.presentFamily,
    };
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(uniqueQueueFamilies.size());

    float queuePriority = 1.0f;
    for (std::uint32_t queueFamily : uniqueQueueFamilies) {
      queueCreateInfos.push_back({
          .queueFamilyIndex = queueFamily,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      });
    }

    const vk::PhysicalDeviceFeatures deviceFeatures;
    vk::DeviceCreateInfo createInfo{
        .pEnabledFeatures = &deviceFeatures,
    };

    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledExtensionNames(deviceExtensions);

#ifdef ENABLE_VALIDATION_LAYERS
    createInfo.setPEnabledLayerNames(validationLayers);
#endif

    return {m_physicalDevice, createInfo};
  }

  [[nodiscard]] vk::raii::SurfaceKHR createSurface() {
    const auto surface = m_window.createSurface(*m_instance);
    return {m_instance, surface};
  }

  struct SwapChainAggreggate {
    vk::raii::SwapchainKHR swapChain;
    std::vector<vk::Image> images;
    vk::Format format;
    vk::Extent2D extent;
  };

  [[nodiscard]] SwapChainAggreggate createSwapChain() const {
    const auto swapChainSupport =
        SwapChainSupportDetails(*m_physicalDevice, *m_surface);
    const auto surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    const auto presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    const auto extent = chooseSwapExtent(swapChainSupport.capabilities);

    // We should use 1 more than the minimum as a basic default, but make sure
    // this doesn't exceed the maximum.
    std::uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    // 0 indicates no maximum
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{
        .surface = *m_surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        // We can specify a certain transform be applied to images, we don't
        // want any transformation here.
        .preTransform = swapChainSupport.capabilities.currentTransform,

        // Something about the alpha channel and blending with other windows
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,

        .presentMode = presentMode,

        // We don't care about obscured pixels
        .clipped = true,

        // Something about swap chain errors, will cover this later
        .oldSwapchain = nullptr,
    };

    const std::array queueFamilyIndicesArray = {
        m_queueFamilyIndices.graphicsFamily,
        m_queueFamilyIndices.presentFamily,
    };

    // We will draw images on the graphics queue, then present them with the
    // present queue. So we need to tell vulkan to enable concurrency between
    // two queues, if they differ.
    if (m_queueFamilyIndices.graphicsFamily !=
        m_queueFamilyIndices.presentFamily) {
      createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      createInfo.setQueueFamilyIndices(queueFamilyIndicesArray);
    } else {
      createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    vk::raii::SwapchainKHR swapChain{m_device, createInfo};
    std::vector<vk::Image> swapChainImages{swapChain.getImages()};

    return {
        .swapChain = std::move(swapChain),
        .images = std::move(swapChainImages),
        .format = surfaceFormat.format,
        .extent = extent,
    };
  }

  [[nodiscard]] vk::raii::ImageView createImageView(vk::Image image,
                                                    vk::Format format) const {
    const vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    return {m_device, createInfo};
  }

  [[nodiscard]] std::vector<vk::raii::ImageView> createSwapChainImageViews() {
    std::vector<vk::raii::ImageView> swapChainImageViews;
    swapChainImageViews.reserve(m_swapChainAggregate.images.size());

    for (const auto& image : m_swapChainAggregate.images) {
      swapChainImageViews.emplace_back(
          createImageView(image, m_swapChainAggregate.format));
    }

    return swapChainImageViews;
  }

  static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    std::size_t fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    return buffer;
  }

  [[nodiscard]] vk::raii::ShaderModule
  createShaderModule(const std::vector<char>& code) const {
    const vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
    };

    return {m_device, createInfo};
  }

  [[nodiscard]] vk::raii::RenderPass createRenderPass() {
    const vk::AttachmentDescription colorAttachment{
        .format = m_swapChainAggregate.format,
        .samples = vk::SampleCountFlagBits::e1,

        // Clear to black on load, and the value will be readable afterwards
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,

        // We're not using stencils, so we don't care what
        // happens before or after
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,

        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR,
    };

    const vk::AttachmentReference colorAttachmentRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
    };
    subpass.setColorAttachments(colorAttachmentRef);

    // Declare our single subpass to be dependent on the implicit beginning
    // subpass
    const vk::SubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,

        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,

        // Wait for the swap chain to finish reading from the image before we
        // can access it.
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
    };

    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachments(colorAttachment);
    renderPassInfo.setSubpasses(subpass);
    renderPassInfo.setDependencies(dependency);

    return {m_device, renderPassInfo};
  }

  [[nodiscard]] vk::raii::DescriptorSetLayout
  createDescriptorSetLayout() const {
    constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(uboLayoutBinding);

    return {m_device, layoutInfo};
  }

  [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout() const {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*m_descriptorSetLayout);
    return {m_device, pipelineLayoutInfo};
  }

  [[nodiscard]] vk::raii::Pipeline createGraphicsPipeline() const {
    const auto vertShaderCode = readFile("shaders/vert.spv");
    const auto fragShaderCode = readFile("shaders/frag.spv");

    const auto vertShaderModule = createShaderModule(vertShaderCode);
    const auto fragShaderModule = createShaderModule(fragShaderCode);

    // Vertex shader pipeline stage
    const vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = *vertShaderModule,
        .pName = "main",
    };

    // Fragment shader pipeline stage
    const vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = *fragShaderModule,
        .pName = "main",
    };

    const std::array shaderStages{
        vertShaderStageInfo,
        fragShaderStageInfo,
    };

    constexpr std::array dynamicStates{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicState;
    dynamicState.setDynamicStates(dynamicStates);

    constexpr auto bindingDescription = Vertex::getBindingDescription();
    constexpr auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.setVertexBindingDescriptions(bindingDescription);
    vertexInputInfo.setVertexAttributeDescriptions(attributeDescriptions);

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
    };

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_swapChainAggregate.extent.width),
        .height = static_cast<float>(m_swapChainAggregate.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = m_swapChainAggregate.extent,
    };

    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.setViewports(viewport);
    viewportState.setScissors(scissor);

    // Rasterizer
    constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    // Multisampling / anti-aliasing, will be revisited later

    constexpr vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
    };

    // Depth and stencil testing: None for now, will be revisited later

    // Color blending

    using enum vk::ColorComponentFlagBits;
    constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = false,
        .colorWriteMask = eR | eG | eB | eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = false,
        .logicOp = vk::LogicOp::eCopy,                        // Optional
        .blendConstants = std::array{0.0f, 0.0f, 0.0f, 0.0f}, // Optional
    };
    colorBlending.setAttachments(colorBlendAttachment);

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = *m_pipelineLayout,
        .renderPass = *m_renderPass,
        .subpass = 0,
        .basePipelineHandle = nullptr, // Optional
        .basePipelineIndex = -1,       // Optional
    };
    pipelineInfo.setStages(shaderStages);

    return {m_device, nullptr, pipelineInfo};
  }

  [[nodiscard]] std::vector<vk::raii::Framebuffer> createFramebuffers() const {
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
    swapChainFramebuffers.reserve(m_swapChainImageViews.size());

    for (const auto& imageView : m_swapChainImageViews) {
      vk::FramebufferCreateInfo framebufferInfo{
          .renderPass = *m_renderPass,
          .width = m_swapChainAggregate.extent.width,
          .height = m_swapChainAggregate.extent.height,
          .layers = 1,
      };
      framebufferInfo.setAttachments(*imageView);

      swapChainFramebuffers.emplace_back(m_device, framebufferInfo);
    }
    return swapChainFramebuffers;
  }

  [[nodiscard]] vk::raii::CommandPool createCommandPool() const {
    const vk::CommandPoolCreateInfo poolInfo{
        // Allow command buffers to be rerecorded individually
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = m_queueFamilyIndices.graphicsFamily,
    };

    return {m_device, poolInfo};
  }

  [[nodiscard]] static std::uint32_t
  findMemoryType(const vk::raii::PhysicalDevice& physicalDevice,
                 const std::uint32_t typeFilter,
                 const vk::MemoryPropertyFlags properties) {
    const auto memProperties = physicalDevice.getMemoryProperties();

    for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
        return i;
      }
    }

    return 0;
  }

  struct AllocatedBuffer {
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;

    [[nodiscard]] AllocatedBuffer(
        const vk::raii::Device& device,
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties)
        : buffer{createBuffer(device, size, usage)}
        , memory{allocateMemory(device, physicalDevice, properties)} {};

  private:
    [[nodiscard]] vk::raii::Buffer
    createBuffer(const vk::raii::Device& device, const vk::DeviceSize size,
                 const vk::BufferUsageFlags usage) {
      const vk::BufferCreateInfo bufferInfo{
          .size = size,
          .usage = usage,
          // These buffers will only be used from the graphics queue, so we can
          // give exclusive access
          .sharingMode = vk::SharingMode::eExclusive,
      };

      return {device, bufferInfo};
    }

    [[nodiscard]] vk::raii::DeviceMemory
    allocateMemory(const vk::raii::Device& device,
                   const vk::raii::PhysicalDevice& physicalDevice,
                   const vk::MemoryPropertyFlags properties) const {
      const vk::MemoryRequirements memRequirements =
          buffer.getMemoryRequirements();

      const vk::MemoryAllocateInfo allocInfo{
          .allocationSize = memRequirements.size,
          .memoryTypeIndex = findMemoryType(
              physicalDevice, memRequirements.memoryTypeBits, properties),
      };

      vk::raii::DeviceMemory bufferMemory{device, allocInfo};

      buffer.bindMemory(*bufferMemory, 0);
      return bufferMemory;
    }
  };

  // Create staging buffer and copy data into it
  [[nodiscard]] AllocatedBuffer stageData(const std::size_t size,
                                          const void* src) const {
    using enum vk::BufferUsageFlagBits;
    using enum vk::MemoryPropertyFlagBits;
    AllocatedBuffer stagingBuffer{m_device, m_physicalDevice, size,
                                  eTransferSrc, eHostVisible | eHostCoherent};

    void* dest = stagingBuffer.memory.mapMemory(0, size);
    std::memcpy(dest, src, size);
    stagingBuffer.memory.unmapMemory();

    return stagingBuffer;
  }

  struct PersistentMappedBuffer {
    AllocatedBuffer allocated_buffer;
    void* mapped_memory;

    [[nodiscard]] PersistentMappedBuffer(
        const vk::raii::Device& device,
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties)
        : allocated_buffer(device, physicalDevice, size, usage, properties)
        , mapped_memory(allocated_buffer.memory.mapMemory(0, size)) {}

    ~PersistentMappedBuffer() {
      if (mapped_memory != nullptr) {
        allocated_buffer.memory.unmapMemory();
      }
    }

    PersistentMappedBuffer& operator=(const PersistentMappedBuffer&) = delete;
    PersistentMappedBuffer(const PersistentMappedBuffer&) = delete;

    PersistentMappedBuffer& operator=(PersistentMappedBuffer&& other) {
      std::swap(allocated_buffer, other.allocated_buffer);
      std::swap(mapped_memory, other.mapped_memory);
      return *this;
    }
    PersistentMappedBuffer(PersistentMappedBuffer&& other)
        : allocated_buffer{std::move(other.allocated_buffer)}
        , mapped_memory{std::exchange(other.mapped_memory, nullptr)} {}
  };

  [[nodiscard]] AllocatedBuffer createVertexBuffer() const {
    using enum vk::BufferUsageFlagBits;
    using enum vk::MemoryPropertyFlagBits;

    const vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    auto stagingBuffer = stageData(bufferSize, vertices.data());

    AllocatedBuffer vertexBuffer{m_device, m_physicalDevice, bufferSize,
                                 eTransferDst | eVertexBuffer, eDeviceLocal};

    copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, bufferSize);

    return vertexBuffer;
  }

  [[nodiscard]] AllocatedBuffer createIndexBuffer() const {
    using enum vk::BufferUsageFlagBits;
    using enum vk::MemoryPropertyFlagBits;

    const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    auto stagingBuffer = stageData(bufferSize, indices.data());

    AllocatedBuffer vertexBuffer{m_device, m_physicalDevice, bufferSize,
                                 eTransferDst | eIndexBuffer, eDeviceLocal};

    copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, bufferSize);

    return vertexBuffer;
  }

  [[nodiscard]] std::vector<PersistentMappedBuffer>
  createUniformBuffers(std::uint32_t count) const {
    const vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    std::vector<PersistentMappedBuffer> uniformBuffers;
    uniformBuffers.reserve(count);

    for (std::uint32_t i = 0; i < count; ++i) {
      uniformBuffers.push_back(PersistentMappedBuffer{
          m_device, m_physicalDevice, bufferSize,
          vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible |
              vk::MemoryPropertyFlagBits::eHostCoherent});
    }

    return uniformBuffers;
  }

  struct AllocatedImage {
    vk::raii::Image image;
    vk::raii::DeviceMemory memory;

    [[nodiscard]] AllocatedImage(const vk::raii::Device& device,
                                 const vk::raii::PhysicalDevice& physicalDevice,
                                 const std::uint32_t width,
                                 const std::uint32_t height,
                                 const vk::Format format,
                                 const vk::ImageTiling tiling,
                                 const vk::ImageUsageFlags usage,
                                 const vk::MemoryPropertyFlags properties)
        : image{nullptr}
        , memory{nullptr} {
      const vk::ImageCreateInfo imageInfo{
          .imageType = vk::ImageType::e2D,
          .format = format,
          .extent =
              {
                  .width = width,
                  .height = height,
                  .depth = 1,
              },
          .mipLevels = 1,
          .arrayLayers = 1,
          .samples = vk::SampleCountFlagBits::e1, // Default for vulkan.hpp
          .tiling = tiling,
          .usage = usage,
          .sharingMode = vk::SharingMode::eExclusive,   // Default
          .initialLayout = vk::ImageLayout::eUndefined, // Default
      };

      image = {device, imageInfo};

      const auto memRequirements = image.getMemoryRequirements();

      const vk::MemoryAllocateInfo allocInfo{
          .allocationSize = memRequirements.size,
          .memoryTypeIndex = findMemoryType(
              physicalDevice, memRequirements.memoryTypeBits, properties),
      };

      memory = {device, allocInfo};

      image.bindMemory(*memory, 0);
    }
  };

  struct SingleUseCommandBuffer : vk::raii::CommandBuffer {

    SingleUseCommandBuffer(const vk::raii::Device& device,
                           const vk::Queue graphicsQueue,
                           const vk::raii::CommandPool& commandPool)
        : vk::raii::CommandBuffer(nullptr)
        , m_graphicsQueue(graphicsQueue) {
      const vk::CommandBufferAllocateInfo allocInfo{
          .commandPool = *commandPool,
          .level = vk::CommandBufferLevel::ePrimary,
          .commandBufferCount = 1,
      };

      static_cast<vk::raii::CommandBuffer&>(*this) =
          std::move(device.allocateCommandBuffers(allocInfo)[0]);

      constexpr vk::CommandBufferBeginInfo beginInfo{
          .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
      };

      begin(beginInfo);
    }

    ~SingleUseCommandBuffer() {
      end();

      vk::SubmitInfo submitInfo{};
      submitInfo.setCommandBuffers(**this);

      m_graphicsQueue.submit(submitInfo);
      m_graphicsQueue.waitIdle();
    }

    SingleUseCommandBuffer& operator=(const SingleUseCommandBuffer&) = delete;
    SingleUseCommandBuffer& operator=(SingleUseCommandBuffer&&) = delete;
    SingleUseCommandBuffer(const SingleUseCommandBuffer&) = delete;
    SingleUseCommandBuffer(SingleUseCommandBuffer&&) = delete;

  private:
    vk::Queue m_graphicsQueue;
  };

  void transitionImageLayout(vk::Image image, const vk::ImageLayout oldLayout,
                             const vk::ImageLayout newLayout) const {
    SingleUseCommandBuffer commandBuffer{m_device, *m_graphicsQueue,
                                         m_commandPool};

    vk::ImageMemoryBarrier barrier{
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;
    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr,
                                  nullptr, barrier);
  }

  void copyBufferToImage(vk::Buffer buffer, vk::Image image,
                         const std::uint32_t width,
                         const std::uint32_t height) const {
    SingleUseCommandBuffer commandBuffer{m_device, *m_graphicsQueue,
                                         m_commandPool};

    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset =
            {
                .x = 0,
                .y = 0,
                .z = 0,
            },
        .imageExtent =
            {
                .width = width,
                .height = height,
                .depth = 1,
            },
    };

    commandBuffer.copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }

  [[nodiscard]] AllocatedImage createTextureImage() const {
    auto result = stb::load("textures/pom.jpg", stb::Channels::rgb_alpha);

    const vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(result.width * result.height * 4);

    auto stagingBuffer = stageData(imageSize, result.data.get());

    auto image = AllocatedImage(
        m_device, m_physicalDevice, static_cast<std::uint32_t>(result.width),
        static_cast<std::uint32_t>(result.height), vk::Format::eR8G8B8A8Srgb,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    transitionImageLayout(*image.image, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(*stagingBuffer.buffer, *image.image,
                      static_cast<std::uint32_t>(result.width),
                      static_cast<std::uint32_t>(result.height));
    transitionImageLayout(*image.image, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);

    return image;
  }

  [[nodiscard]] vk::raii::ImageView createTextureImageView() {
    return createImageView(*m_texture_image.image, vk::Format::eR8G8B8A8Srgb);
  }

  [[nodiscard]] vk::raii::DescriptorPool createDescriptorPool() const {
    const vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = MAX_FRAMES_IN_FLIGHT,
    };

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
    };
    poolInfo.setPoolSizes(poolSize);

    return {m_device, poolInfo};
  }

  [[nodiscard]] std::vector<vk::raii::DescriptorSet>
  createDescriptorSets() const {
    const std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                       *m_descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *m_descriptorPool,
    };
    allocInfo.setSetLayouts(layouts);

    auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

    for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      const vk::DescriptorBufferInfo bufferInfo{
          .buffer = *m_uniformBuffers[i].allocated_buffer.buffer,
          .offset = 0,
          .range = sizeof(UniformBufferObject),
      };

      const vk::WriteDescriptorSet descriptorWrite{
          .dstSet = *descriptorSets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &bufferInfo,
      };

      m_device.updateDescriptorSets(descriptorWrite, nullptr);
    }

    return descriptorSets;
  }

  void copyBuffer(const vk::raii::Buffer& srcBuffer,
                  vk::raii::Buffer& dstBuffer,
                  const vk::DeviceSize size) const {
    SingleUseCommandBuffer commandBuffer{m_device, *m_graphicsQueue,
                                         m_commandPool};

    const vk::BufferCopy copyRegion{
        .size = size,
    };
    commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
  }

  void recordCommandBuffer(vk::CommandBuffer commandBuffer,
                           std::uint32_t imageIndex) const {
    const vk::CommandBufferBeginInfo beginInfo;

    commandBuffer.begin(beginInfo);

    // Set background color to black, 100% opacity
    const vk::ClearValue clearColor{
        .color = {std::array{0.0f, 0.0f, 0.0f, 1.0f}},
    };

    vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = *m_renderPass,
        .framebuffer = *(m_swapChainFramebuffers[imageIndex]),
        .renderArea =
            {
                .offset = {0, 0},
                .extent = m_swapChainAggregate.extent,
            },
    };
    renderPassInfo.setClearValues(clearColor);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               *m_graphicsPipeline);

    commandBuffer.bindVertexBuffers(0, *m_vertexBuffer.buffer, {0ull});

    commandBuffer.bindIndexBuffer(*m_indexBuffer.buffer, 0,
                                  vk::IndexType::eUint16);

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_swapChainAggregate.extent.width),
        .height = static_cast<float>(m_swapChainAggregate.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    commandBuffer.setViewport(0, viewport);

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = m_swapChainAggregate.extent,
    };
    commandBuffer.setScissor(0, scissor);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     *m_pipelineLayout, 0,
                                     *m_descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,
                              0);

    commandBuffer.endRenderPass();

    commandBuffer.end();
  }

  [[nodiscard]] std::vector<vk::raii::CommandBuffer>
  createCommandBuffers() const {
    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    return vk::raii::CommandBuffers{m_device, allocInfo};
  }

  [[nodiscard]] std::vector<vk::raii::Semaphore>
  createSemaphores(std::uint32_t count) {
    std::vector<vk::raii::Semaphore> semaphores;
    semaphores.reserve(count);

    constexpr vk::SemaphoreCreateInfo createInfo;
    for (std::uint32_t i = 0; i < count; ++i) {
      semaphores.emplace_back(m_device, createInfo);
    }
    return semaphores;
  }

  [[nodiscard]] std::vector<vk::raii::Fence>
  createFences(std::uint32_t count) const {
    std::vector<vk::raii::Fence> fences;
    fences.reserve(count);
    constexpr vk::FenceCreateInfo createInfo{
        // Start in the signaled state
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };

    for (std::uint32_t i = 0; i < count; ++i) {
      fences.emplace_back(m_device, createInfo);
    }
    return fences;
  }

  void recreateSwapChain() {
    // If the window was minimized, wait.
    while (m_window.getFramebufferSize() == std::tuple{0, 0}) {
      glfw::waitEvents();
    }

    m_device.waitIdle();

    m_swapChainAggregate = createSwapChain();
    m_swapChainImageViews = createSwapChainImageViews();
    m_swapChainFramebuffers = createFramebuffers();
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    const auto currentTime = std::chrono::high_resolution_clock::now();
    float elapsedTime =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime -
                                                                   startTime)
            .count();
    UniformBufferObject ubo{
        .model = glm::rotate(glm::mat4(1.0f), elapsedTime * glm::radians(90.0f),
                             glm::vec3(0.0f, 0.0f, 1.0f)),
        .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                            glm::vec3(0.0f, 0.0f, 0.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f)),
        .proj = glm::perspective(
            glm::radians(45.0f),
            static_cast<float>(m_swapChainAggregate.extent.width) /
                static_cast<float>(m_swapChainAggregate.extent.height),
            0.1f, 10.0f),
    };

    // Account for differences between OpenGL and Vulkan
    ubo.proj[1][1] *= -1;

    std::memcpy(m_uniformBuffers[currentImage].mapped_memory, &ubo,
                sizeof(ubo));
  }

  void drawFrame() {
    const auto waitResult =
        m_device.waitForFences(*m_inFlightFences[currentFrame], true,
                               std::numeric_limits<std::uint64_t>::max());

    if (waitResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence!");
    }

    const vk::AcquireNextImageInfoKHR acquireInfo{
        .swapchain = *(m_swapChainAggregate.swapChain),
        .timeout = std::numeric_limits<std::uint64_t>::max(),
        .semaphore = *m_imageAvailableSemaphores[currentFrame],
        .deviceMask = 1,
    };

    const auto [acquireResult, imageIndex] =
        m_device.acquireNextImage2KHR(acquireInfo);

    if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    } else if (acquireResult != vk::Result::eSuccess &&
               acquireResult != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("failed to acquire next image!");
    }

    m_device.resetFences(*m_inFlightFences[currentFrame]);

    m_commandBuffers[currentFrame].reset();

    recordCommandBuffer(*m_commandBuffers[currentFrame], imageIndex);

    constexpr vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    updateUniformBuffer(currentFrame);

    const auto submitInfo =
        vk::SubmitInfo{}
            .setWaitDstStageMask(waitStages)
            .setWaitSemaphores(*m_imageAvailableSemaphores[currentFrame])
            .setCommandBuffers(*m_commandBuffers[currentFrame])
            .setSignalSemaphores(*m_renderFinishedSemaphores[currentFrame]);

    m_graphicsQueue.submit(submitInfo, *m_inFlightFences[currentFrame]);

    const auto presentInfo =
        vk::PresentInfoKHR{}
            .setImageIndices(imageIndex)
            .setWaitSemaphores(*m_renderFinishedSemaphores[currentFrame])
            .setSwapchains(*(m_swapChainAggregate.swapChain));

    // Finally, present.
    const auto presentResult = presentQueue.presentKHR(presentInfo);

    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (presentResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void mainLoop() {
    while (!m_window.shouldClose()) {
      glfw::pollEvents();
      drawFrame();
    }

    // Wait for all async operations to finish
    m_device.waitIdle();
  }

  glfw::GlfwLibrary m_glfwLib{glfw::init()};
  glfw::Window m_window{initWindow()};
  vk::raii::Context m_context;
  vk::raii::Instance m_instance{createInstance()};
#ifdef ENABLE_VALIDATION_LAYERS
  vk::raii::DebugUtilsMessengerEXT m_debugMessenger{
      m_instance, createDebugMessengerCreateInfo()};
#endif
  vk::raii::SurfaceKHR m_surface{createSurface()};
  vk::raii::PhysicalDevice m_physicalDevice{pickPhysicalDevice()};
  // We unwrap this result, we are guaranteed this will succeed since we
  // validated that all the requested queues are available.
  QueueFamilyIndices m_queueFamilyIndices{
      findQueueFamilies(*m_physicalDevice).finalize().value()};
  vk::raii::Device m_device{createLogicalDevice()};
  vk::raii::Queue m_graphicsQueue{
      m_device.getQueue(m_queueFamilyIndices.graphicsFamily, 0)};
  vk::raii::Queue presentQueue{
      m_device.getQueue(m_queueFamilyIndices.presentFamily, 0)};
  vk::raii::CommandPool m_commandPool{createCommandPool()};
  AllocatedBuffer m_vertexBuffer{createVertexBuffer()};
  AllocatedBuffer m_indexBuffer{createIndexBuffer()};
  std::vector<PersistentMappedBuffer> m_uniformBuffers{
      createUniformBuffers(MAX_FRAMES_IN_FLIGHT)};
  AllocatedImage m_texture_image{createTextureImage()};
  vk::raii::ImageView m_texture_image_view{createTextureImageView()};
  vk::raii::DescriptorPool m_descriptorPool{createDescriptorPool()};
  vk::raii::DescriptorSetLayout m_descriptorSetLayout{
      createDescriptorSetLayout()};
  std::vector<vk::raii::DescriptorSet> m_descriptorSets{createDescriptorSets()};
  SwapChainAggreggate m_swapChainAggregate{createSwapChain()};
  std::vector<vk::raii::ImageView> m_swapChainImageViews{
      createSwapChainImageViews()};
  vk::raii::RenderPass m_renderPass{createRenderPass()};
  vk::raii::PipelineLayout m_pipelineLayout{createPipelineLayout()};
  vk::raii::Pipeline m_graphicsPipeline{createGraphicsPipeline()};
  std::vector<vk::raii::Framebuffer> m_swapChainFramebuffers{
      createFramebuffers()};
  std::vector<vk::raii::CommandBuffer> m_commandBuffers{createCommandBuffers()};
  std::vector<vk::raii::Semaphore> m_imageAvailableSemaphores{
      createSemaphores(MAX_FRAMES_IN_FLIGHT)};
  std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores{
      createSemaphores(MAX_FRAMES_IN_FLIGHT)};
  std::vector<vk::raii::Fence> m_inFlightFences{
      createFences(MAX_FRAMES_IN_FLIGHT)};
  std::uint32_t currentFrame{0};
  bool framebufferResized{false};
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
