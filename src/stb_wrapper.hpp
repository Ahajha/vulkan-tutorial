#include <stb_image.h>

#include <utility>

namespace stb {

enum class Channels {
  default_ = STBI_default, // only used for desired_channels
  grey = STBI_grey,
  grey_alpha = STBI_grey_alpha,
  rgb = STBI_rgb,
  rgb_alpha = STBI_rgb_alpha,
};

// Like std::unique_ptr, but without the memory overhead of storing the custom
// deleter.
struct PixelData {
  unsigned char* data;

  operator unsigned char*() { return data; }
  operator const unsigned char*() const { return data; }

  PixelData(unsigned char* d)
      : data{d} {}

  ~PixelData() { stbi_image_free(data); }

  PixelData& operator=(const PixelData&) = delete;
  PixelData(const PixelData&) = delete;

  PixelData& operator=(PixelData&& other) {
    std::swap(data, other.data);
    return *this;
  }

  PixelData(PixelData&& other)
      : data{std::exchange(other.data, nullptr)} {}
};

struct Result {
  int width;
  int height;
  int bytes_per_pixel;
  PixelData data;
};

// Throws an exception on failure
Result load(const char* filename,
            Channels desired_channels = Channels::default_);

} // namespace stb
