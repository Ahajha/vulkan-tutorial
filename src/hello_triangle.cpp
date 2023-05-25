#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>

// GLFWPP includes special functions if it detects that vulkan-hpp is included,
// so include after.
#define GLFW_INCLUDE_VULKAN
#include <glfwpp/glfwpp.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
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
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
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
    OptionalQueueFamilyIndices indices;
    // Logic to find queue family indices to populate struct with

    const auto queueFamilies = device.getQueueFamilyProperties();

    std::uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphicsFamily = i;
      }

      if (device.getSurfaceSupportKHR(i, *m_surface)) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
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

  [[nodiscard]] std::vector<vk::raii::ImageView> createImageViews() {
    std::vector<vk::raii::ImageView> swapChainImageViews;
    swapChainImageViews.reserve(m_swapChainAggregate.images.size());

    for (const auto& image : m_swapChainAggregate.images) {
      const vk::ImageViewCreateInfo createInfo{
          .image = image,
          .viewType = vk::ImageViewType::e2D,
          .format = m_swapChainAggregate.format,
          .components =
              {
                  .r = vk::ComponentSwizzle::eIdentity,
                  .g = vk::ComponentSwizzle::eIdentity,
                  .b = vk::ComponentSwizzle::eIdentity,
                  .a = vk::ComponentSwizzle::eIdentity,
              },
          .subresourceRange =
              {
                  .aspectMask = vk::ImageAspectFlagBits::eColor,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };

      swapChainImageViews.emplace_back(m_device, createInfo);
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

  [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout() const {
    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
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
        .frontFace = vk::FrontFace::eClockwise,
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

    [[nodiscard]] std::uint32_t
    findMemoryType(const vk::raii::PhysicalDevice& physicalDevice,
                   const std::uint32_t typeFilter,
                   const vk::MemoryPropertyFlags properties) const {
      const auto memProperties = physicalDevice.getMemoryProperties();

      for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) ==
                properties) {
          return i;
        }
      }

      return 0;
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
  [[nodiscard]] AllocatedBuffer createVertexBuffer() const {
    using enum vk::BufferUsageFlagBits;
    using enum vk::MemoryPropertyFlagBits;

    const vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    AllocatedBuffer stagingBuffer{m_device, m_physicalDevice, bufferSize,
                                  eTransferSrc, eHostVisible | eHostCoherent};

    void* data = stagingBuffer.memory.mapMemory(0, bufferSize);
    std::memcpy(data, vertices.data(), bufferSize);
    stagingBuffer.memory.unmapMemory();

    AllocatedBuffer vertexBuffer{m_device, m_physicalDevice, bufferSize,
                                 eTransferDst | eVertexBuffer, eDeviceLocal};

    copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, bufferSize);

    return vertexBuffer;
  }

  void copyBuffer(const vk::raii::Buffer& srcBuffer,
                  vk::raii::Buffer& dstBuffer,
                  const vk::DeviceSize size) const {
    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    auto buffer = std::move(m_device.allocateCommandBuffers(allocInfo)[0]);

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    buffer.begin(beginInfo);

    const vk::BufferCopy copyRegion{
        .size = size,
    };
    buffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

    buffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(*buffer);

    m_graphicsQueue.submit(submitInfo);
    m_graphicsQueue.waitIdle();
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

    commandBuffer.draw(3, 1, 0, 0);

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
    m_swapChainImageViews = createImageViews();
    m_swapChainFramebuffers = createFramebuffers();
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
  SwapChainAggreggate m_swapChainAggregate{createSwapChain()};
  std::vector<vk::raii::ImageView> m_swapChainImageViews{createImageViews()};
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
