#include "stb_wrapper.hpp"

#include <stdexcept>

namespace stb {

Result load(const char* filename, Channels desired_channels) {
  int width, height, bytes_per_pixel;
  unsigned char* data = stbi_load(filename, &width, &height, &bytes_per_pixel,
                                  static_cast<int>(desired_channels));

  if (data == nullptr) {
    throw std::runtime_error(stbi_failure_reason());
  }

  return Result{
      .width = width,
      .height = height,
      .bytes_per_pixel = bytes_per_pixel,
      .data = PixelData{data},
  };
}
} // namespace stb