#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    queryExtensions();
    initVulkan();
    mainLoop();
    cleanupVulkan();
    cleanupWindow();
  }

private:
  void initWindow() {
    glfwInit();

    // Disable OpenGL in GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // For now, disallow sizing of windows
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Width, Height, window name, monitor, unused(OpenGL only)
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

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

    // Extensions are needed for GLFW, we can query GLFW to get this info
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    // Validation layers, for now none
    createInfo.enabledLayerCount = 0;

    // Create the instance. Second parameter is for custom allocator callbacks,
    // always nullptr for this tutorial.
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void queryExtensions() {
    // For the first call, just get the number of extensions (last parameter
    // nullptr)
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                           extensions.data());

    std::cout << "available extensions:\n";

    for (const auto &extension : extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  void initVulkan() { createInstance(); }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanupVulkan() { vkDestroyInstance(instance, nullptr); }

  void cleanupWindow() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  GLFWwindow *window;
  VkInstance instance;
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
