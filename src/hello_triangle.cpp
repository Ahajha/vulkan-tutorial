#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanupVulkan();
    cleanupWindow();
  }

private:
  // Returns true iff all requested validation layers (in validationLayers) are
  // available.
  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    return std::ranges::all_of(validationLayers, [&](const char *layerName) {
      return std::ranges::find_if(availableLayers, [&](const auto &layerProps) {
               return std::strcmp(layerName, layerProps.layerName) == 0;
             }) != availableLayers.end();
    });
  }

  // Initialized the GLFW window
  void initWindow() {
    glfwInit();

    // Disable OpenGL in GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // For now, disallow sizing of windows
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Width, Height, window name, monitor, unused(OpenGL only)
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  // Returns the available Vulkan extensions
  std::vector<VkExtensionProperties> queryExtensions() {
    // For the first call, just get the number of extensions (last parameter
    // nullptr)
    std::uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                           extensions.data());

    return extensions;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                [[maybe_unused]] void *pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage
              << "\n\tSeverity: " << messageSeverity
              << "\n\tType: " << messageType << std::endl;

    return VK_FALSE;
  }

  // Get a list of Vulkan extensions required/requested by the application
  std::vector<const char *> getRequiredExtensions() {
    // Extensions are needed for GLFW, we can query GLFW to get this info
    std::uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions,
                                         glfwExtensions + glfwExtensionCount);

    if constexpr (enableValidationLayers) {
      // Add "VK_EXT_debug_utils", macro is provided to prevent typos
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const auto available_extensions = queryExtensions();

    // Vulkan will validate this for us, but just for fun:
    const auto valid = std::ranges::all_of(
        std::span{glfwExtensions, glfwExtensionCount},
        [&](const auto &extension_name) {
          return std::ranges::find_if(
                     available_extensions, [&](const auto &extension) {
                       return std::strcmp(extension.extensionName,
                                          extension_name) == 0;
                     }) != available_extensions.end();
        });

    if (!valid) {
      throw std::runtime_error(
          "Vulkan does not have all extensions necessary for glfw");
    }

    return extensions;
  }

  VkResult CreateDebugUtilsMessengerEXT(
      VkInstance instance,
      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
      const VkAllocationCallbacks *pAllocator,
      VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
      return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
  }

  void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                     VkDebugUtilsMessengerEXT debugMessenger,
                                     const VkAllocationCallbacks *pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(instance, debugMessenger, pAllocator);
    }
  }

  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
  }

  void setupDebugMessenger() {
    if constexpr (enableValidationLayers) {
      VkDebugUtilsMessengerCreateInfoEXT createInfo{};
      populateDebugMessengerCreateInfo(createInfo);

      if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                       &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
      }
    }
  }

  // Creates the Vulkan instance
  void createInstance() {
    // Set app info (optional)
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Specify extensions and validation layers we want to use (mandatory)
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if constexpr (enableValidationLayers) {
      if (!checkValidationLayerSupport()) {
        throw std::runtime_error(
            "validation layers requested, but not available!");
      } else {
        createInfo.enabledLayerCount =
            static_cast<std::uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext =
            (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;

        std::cerr << "validation layers enabled\n";
      }
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    auto requiredExtensions = getRequiredExtensions();

    createInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();

    // Create the instance. Second parameter is for custom allocator callbacks,
    // always nullptr for this tutorial.
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  struct QueueFamilyIndices {
    // The queue index that supports graphics
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    bool isComplete() const {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    // Logic to find queue family indices to populate struct with

    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());
    /*
    // Print queue info, for fun
    for (const auto &qfamily : queueFamilies) {
      std::cout << "Queue count: " << qfamily.queueCount << '\n';
      std::cout << "Flags: ";
      if (qfamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        std::cout << "Graphics ";
      }
      if (qfamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        std::cout << "Compute ";
      }
      if (qfamily.queueFlags & VK_QUEUE_TRANSFER_BIT) {
        std::cout << "Transfer ";
      }
      if (qfamily.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
        std::cout << "Sparse binding ";
      }
      std::cout << '\n';
    }
    */
    std::uint32_t i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

      if (presentSupport) {
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
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    std::uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  bool isDeviceSuitable(VkPhysicalDevice device) {
    // Query properties
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    // Print some info, for fun:
    std::cout << "Device name: " << deviceProperties.deviceName << '\n';

    const auto apiVersion = deviceProperties.apiVersion;
    std::cout << "Vulkan version supported by device: "
              << VK_API_VERSION_MAJOR(apiVersion) << '.'
              << VK_API_VERSION_MINOR(apiVersion) << '.'
              << VK_API_VERSION_MINOR(apiVersion) << '\n';

    const auto driverVersion = deviceProperties.driverVersion;
    std::cout << "Driver version: " << VK_API_VERSION_MAJOR(driverVersion)
              << '.' << VK_API_VERSION_MINOR(driverVersion) << '.'
              << VK_API_VERSION_MINOR(driverVersion) << '\n';

    const char *deviceType = "unknown";
    switch (deviceProperties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
      deviceType = "other";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      deviceType = "integrated";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      deviceType = "discrete";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      deviceType = "virtual";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      deviceType = "cpu";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM:
      break;
    }
    std::cout << "Device type: " << deviceType << '\n';

    // Query features (Currently unneeded)
    // VkPhysicalDeviceFeatures deviceFeatures;
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    auto indices = findQueueFamilies(device);

    const bool extensionsSupported = checkDeviceExtensionSupport(device);

    return indices.isComplete() && extensionsSupported;
  }

  // Sets physicalDevice to a suitable device
  void pickPhysicalDevice() {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    auto iter = std::ranges::find_if(
        devices, [this](auto &device) { return isDeviceSuitable(device); });

    if (iter == devices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }

    physicalDevice = *iter;
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::cout << "Graphics queue family index: "
              << indices.graphicsFamily.value() << '\n';
    std::cout << "Present queue family index: " << indices.presentFamily.value()
              << '\n';

    std::set<std::uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(), indices.presentFamily.value()};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(
        uniqueQueueFamilies.size());

    float queuePriority = 1.0f;
    std::uint32_t index = 0;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      auto &queueCreateInfo = queueCreateInfos[index];
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      ++index;
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount =
        static_cast<std::uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    VkPhysicalDeviceFeatures deviceFeatures{};
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if constexpr (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<std::uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }

  void createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanupVulkan() {

    vkDestroyDevice(device, nullptr);

    if constexpr (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkDestroyInstance(instance, nullptr);
  }

  void cleanupWindow() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  GLFWwindow *window;
  VkInstance instance;
  VkDebugUtilsMessengerEXT debugMessenger;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;
  VkQueue graphicsQueue;
  VkQueue presentQueue;
  VkSurfaceKHR surface;
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
