#define GLFW_INCLUDE_VULKAN
#include <glfwpp/glfwpp.h>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
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

constexpr std::array validationLayers = {"VK_LAYER_KHRONOS_validation"};

constexpr std::array deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class HelloTriangleApplication {
public:
  HelloTriangleApplication()
      : window{initWindow()}
      , instance{createInstance()}
      , debugMessenger{instance, createDebugMessengerCreateInfo()}
      , surface{createSurface()}
      , physicalDevice{pickPhysicalDevice()}
      // We unwrap this result, we are guaranteed this will succeed since we
      // validated that all the requested queues are available.
      , queueFamilyIndices{findQueueFamilies(*physicalDevice)
                               .finalize()
                               .value()}
      , device{createLogicalDevice()}
      , graphicsQueue{device.getQueue(queueFamilyIndices.graphicsFamily, 0)}
      , presentQueue{device.getQueue(queueFamilyIndices.presentFamily, 0)}
      , swapChainAggregate{createSwapChain()}
      , swapChainImageViews{createImageViews()}
      , renderPass{createRenderPass()}
      , pipelineLayout{createPipelineLayout()}
      , graphicsPipeline{createGraphicsPipeline()}
      , swapChainFramebuffers{createFramebuffers()}
      , commandPool{createCommandPool()}
      , commandBuffer{createCommandBuffer()}
      , imageAvailableSemaphore{device, vk::SemaphoreCreateInfo{}}
      , renderFinishedSemaphore{device, vk::SemaphoreCreateInfo{}}
      , inFlightFence{createFence()} {}

  void run() { mainLoop(); }

private:
  // Initialized the GLFW window
  [[nodiscard]] static glfw::Window initWindow() {
    glfw::WindowHints{.resizable = false, .clientApi = glfw::ClientApi::None}
        .apply();
    return {WIDTH, HEIGHT, "Vulkan"};
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

    if constexpr (enableValidationLayers) {
      // Add "VK_EXT_debug_utils", macro is provided to prevent typos
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  [[nodiscard]] constexpr static vk::DebugUtilsMessengerCreateInfoEXT
  createDebugMessengerCreateInfo() {
    using enum vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using enum vk::DebugUtilsMessageTypeFlagBitsEXT;
    return {
        .messageSeverity = eVerbose | eWarning | eError,
        .messageType = eGeneral | eValidation | ePerformance,
        .pfnUserCallback = debugCallback,
    };
  }

  // Creates the Vulkan instance
  [[nodiscard]] vk::raii::Instance createInstance() {
    // Set app info (optional)
    const vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    const auto requiredExtensions = getRequiredExtensions();

    if constexpr (enableValidationLayers) {
      vk::StructureChain<vk::InstanceCreateInfo,
                         vk::DebugUtilsMessengerCreateInfoEXT>
          chain{vk::InstanceCreateInfo{.pApplicationInfo = &appInfo},
                createDebugMessengerCreateInfo()};

      auto& createInfo = chain.get<vk::InstanceCreateInfo>();

      createInfo.setPEnabledExtensionNames(requiredExtensions);
      createInfo.setPEnabledLayerNames(validationLayers);

      std::cerr << "validation layers enabled\n";

      return {context, createInfo};
    } else {
      vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo};

      createInfo.setPEnabledExtensionNames(requiredExtensions);

      return {context, createInfo};
    }
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
  findQueueFamilies(vk::PhysicalDevice device) {
    OptionalQueueFamilyIndices indices;
    // Logic to find queue family indices to populate struct with

    const auto queueFamilies = device.getQueueFamilyProperties();

    std::uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphicsFamily = i;
      }

      if (device.getSurfaceSupportKHR(i, *surface)) {
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
  [[nodiscard]] bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
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

  bool isDeviceSuitable(vk::PhysicalDevice device) {
    if (!findQueueFamilies(device).isComplete() ||
        !checkDeviceExtensionSupport(device))
      return false;

    const auto swapChainSupport = SwapChainSupportDetails(device, *surface);
    return !swapChainSupport.formats.empty() &&
           !swapChainSupport.presentModes.empty();
  }

  [[nodiscard]] vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::span<const vk::SurfaceFormatKHR> availableFormats) {
    const vk::SurfaceFormatKHR desiredFormat{
        .format = vk::Format::eB8G8R8A8Srgb,
        .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
    };

    const auto iter = std::ranges::find(availableFormats, desiredFormat);

    return iter != availableFormats.end() ? desiredFormat
                                          : availableFormats.front();
  }

  [[nodiscard]] vk::PresentModeKHR chooseSwapPresentMode(
      const std::span<const vk::PresentModeKHR> availablePresentModes) {
    const vk::PresentModeKHR desiredPresentMode = vk::PresentModeKHR::eMailbox;

    const auto iter =
        std::ranges::find(availablePresentModes, desiredPresentMode);

    return iter != availablePresentModes.end() ? desiredPresentMode
                                               : vk::PresentModeKHR::eFifo;
  }

  [[nodiscard]] vk::Extent2D
  chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {

    if (capabilities.currentExtent.width !=
        std::numeric_limits<std::uint32_t>::max()) {
      // Not sure about the reasoning about this branch
      return capabilities.currentExtent;
    } else {
      const auto [width, height] = window.getFramebufferSize();

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
  [[nodiscard]] vk::raii::PhysicalDevice pickPhysicalDevice() {
    const auto devices = instance.enumeratePhysicalDevices();

    auto iter = std::ranges::find_if(
        devices, [this](auto& device) { return isDeviceSuitable(*device); });

    if (iter == devices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }

    return *iter;
  }

  [[nodiscard]] vk::raii::Device createLogicalDevice() {
    const std::set<std::uint32_t> uniqueQueueFamilies = {
        queueFamilyIndices.graphicsFamily, queueFamilyIndices.presentFamily};
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

    if constexpr (enableValidationLayers) {
      createInfo.setPEnabledLayerNames(validationLayers);
    }

    return {physicalDevice, createInfo};
  }

  [[nodiscard]] vk::raii::SurfaceKHR createSurface() {
    VkSurfaceKHR native_surface;
    if (window.createSurface(*instance, nullptr, &native_surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    return {instance, native_surface};
  }

  struct SwapChainAggreggate {
    vk::raii::SwapchainKHR swapChain;
    std::vector<vk::Image> images;
    vk::Format format;
    vk::Extent2D extent;
  };

  [[nodiscard]] SwapChainAggreggate createSwapChain() {
    const auto swapChainSupport =
        SwapChainSupportDetails(*physicalDevice, *surface);
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
        .surface = *surface,
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

    const std::uint32_t queueFamilyIndicesArray[] = {
        queueFamilyIndices.graphicsFamily, queueFamilyIndices.presentFamily};

    // We will draw images on the graphics queue, then present them with the
    // present queue. So we need to tell vulkan to enable concurrency between
    // two queues, if they differ.
    if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily) {
      createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      createInfo.setQueueFamilyIndices(queueFamilyIndicesArray);
    } else {
      createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    vk::raii::SwapchainKHR swapChain{device, createInfo};
    std::vector<vk::Image> swapChainImages{swapChain.getImages()};

    return {
        .swapChain = std::move(swapChain),
        .images = std::move(swapChainImages),
        .format = surfaceFormat.format,
        .extent = extent,
    };
  }

  [[nodiscard]] std::vector<vk::raii::ImageView> createImageViews() const {
    std::vector<vk::raii::ImageView> swapChainImageViews;
    swapChainImageViews.reserve(swapChainAggregate.images.size());

    for (const auto& image : swapChainAggregate.images) {
      const vk::ImageViewCreateInfo createInfo{
          .image = image,
          .viewType = vk::ImageViewType::e2D,
          .format = swapChainAggregate.format,
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

      swapChainImageViews.emplace_back(device, createInfo);
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
    file.read(buffer.data(), fileSize);

    return buffer;
  }

  [[nodiscard]] vk::raii::ShaderModule
  createShaderModule(const std::vector<char>& code) const {
    const vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
    };

    return {device, createInfo};
  }

  [[nodiscard]] vk::raii::RenderPass createRenderPass() {
    const vk::AttachmentDescription colorAttachment{
        .format = swapChainAggregate.format,
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

    return {device, renderPassInfo};
  }

  [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout() {
    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    return {device, pipelineLayoutInfo};
  }

  [[nodiscard]] vk::raii::Pipeline createGraphicsPipeline() {
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

    // We will revisit this later - we're currently hardcoding the triangle
    // data in the shader.
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
    };

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapChainAggregate.extent.width),
        .height = static_cast<float>(swapChainAggregate.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainAggregate.extent,
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

    constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = false,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
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
        .layout = *pipelineLayout,
        .renderPass = *renderPass,
        .subpass = 0,
        .basePipelineHandle = nullptr, // Optional
        .basePipelineIndex = -1,       // Optional
    };
    pipelineInfo.setStages(shaderStages);

    return {device, nullptr, pipelineInfo};
  }

  [[nodiscard]] std::vector<vk::raii::Framebuffer> createFramebuffers() {
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
    swapChainFramebuffers.reserve(swapChainImageViews.size());

    for (const auto& imageView : swapChainImageViews) {
      vk::FramebufferCreateInfo framebufferInfo{
          .renderPass = *renderPass,
          .width = swapChainAggregate.extent.width,
          .height = swapChainAggregate.extent.height,
          .layers = 1,
      };
      framebufferInfo.setAttachments(*imageView);

      swapChainFramebuffers.emplace_back(device, framebufferInfo);
    }
    return swapChainFramebuffers;
  }

  [[nodiscard]] vk::raii::CommandPool createCommandPool() {
    const vk::CommandPoolCreateInfo poolInfo{
        // Allow command buffers to be rerecorded individually
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily,
    };

    return {device, poolInfo};
  }

  void recordCommandBuffer(vk::CommandBuffer commandBuffer,
                           std::uint32_t imageIndex) {
    const vk::CommandBufferBeginInfo beginInfo;

    commandBuffer.begin(beginInfo);

    // Set background color to black, 100% opacity
    const vk::ClearValue clearColor{
        .color = std::array{0.0f, 0.0f, 0.0f, 1.0f},
    };

    vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = *renderPass,
        .framebuffer = *(swapChainFramebuffers[imageIndex]),
        .renderArea =
            {
                .offset = {0, 0},
                .extent = swapChainAggregate.extent,
            },
    };
    renderPassInfo.setClearValues(clearColor);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               *graphicsPipeline);

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapChainAggregate.extent.width),
        .height = static_cast<float>(swapChainAggregate.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    commandBuffer.setViewport(0, viewport);

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainAggregate.extent,
    };
    commandBuffer.setScissor(0, scissor);

    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRenderPass();

    commandBuffer.end();
  }

  [[nodiscard]] vk::raii::CommandBuffer createCommandBuffer() {

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    // Create a vector of buffers, then extract the sole element.
    // This could maybe be cleaner.
    return std::move(vk::raii::CommandBuffers{device, allocInfo}.front());
  }

  [[nodiscard]] vk::raii::Fence createFence() {
    constexpr vk::FenceCreateInfo fenceInfo{
        // Start in the signaled state
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };

    return {device, fenceInfo};
  }

  void drawFrame() {
    const auto waitResult = device.waitForFences(
        *inFlightFence, true, std::numeric_limits<std::uint64_t>::max());

    if (waitResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence!");
    }

    device.resetFences(*inFlightFence);

    const vk::AcquireNextImageInfoKHR acquireInfo{
        .swapchain = *(swapChainAggregate.swapChain),
        .timeout = std::numeric_limits<std::uint64_t>::max(),
        .semaphore = *imageAvailableSemaphore,
        .deviceMask = 1,
    };

    const auto [acquireResult, imageIndex] =
        device.acquireNextImage2KHR(acquireInfo);

    // By design, this function does not throw. For our purposes, we will throw.
    // https://github.com/KhronosGroup/Vulkan-Hpp/issues/150
    if (acquireResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to acquire next image!");
    }

    commandBuffer.reset();

    recordCommandBuffer(*commandBuffer, imageIndex);

    constexpr vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = waitStages,
    };
    submitInfo.setWaitSemaphores(*imageAvailableSemaphore);
    submitInfo.setCommandBuffers(*commandBuffer);
    submitInfo.setSignalSemaphores(*renderFinishedSemaphore);

    graphicsQueue.submit(submitInfo, *inFlightFence);

    vk::PresentInfoKHR presentInfo{
        .pImageIndices = &imageIndex,
    };
    presentInfo.setWaitSemaphores(*renderFinishedSemaphore);
    presentInfo.setSwapchains(*(swapChainAggregate.swapChain));

    // Finally, present.
    const auto presentResult = presentQueue.presentKHR(presentInfo);

    if (presentResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present!");
    }
  }

  void mainLoop() {
    while (!window.shouldClose()) {
      glfw::pollEvents();
      drawFrame();
    }

    // Wait for all async operations to finish
    device.waitIdle();
  }

  glfw::GlfwLibrary glfwLib{glfw::init()};
  glfw::Window window;
  vk::raii::Context context;
  vk::raii::Instance instance;
  vk::raii::DebugUtilsMessengerEXT debugMessenger;
  vk::raii::SurfaceKHR surface;
  vk::raii::PhysicalDevice physicalDevice;
  QueueFamilyIndices queueFamilyIndices;
  vk::raii::Device device;
  vk::raii::Queue graphicsQueue;
  vk::raii::Queue presentQueue;
  SwapChainAggreggate swapChainAggregate;
  std::vector<vk::raii::ImageView> swapChainImageViews;
  vk::raii::RenderPass renderPass;
  vk::raii::PipelineLayout pipelineLayout;
  vk::raii::Pipeline graphicsPipeline;
  std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
  vk::raii::CommandPool commandPool;
  vk::raii::CommandBuffer commandBuffer;
  vk::raii::Semaphore imageAvailableSemaphore;
  vk::raii::Semaphore renderFinishedSemaphore;
  vk::raii::Fence inFlightFence;
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
