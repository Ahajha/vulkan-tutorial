#include <stb_image.h>

#include <memory>
#include <utility>

namespace stb {

enum class Channels {
  default_ = STBI_default, // only used for desired_channels
  grey = STBI_grey,
  grey_alpha = STBI_grey_alpha,
  rgb = STBI_rgb,
  rgb_alpha = STBI_rgb_alpha,
};

namespace detail {
constexpr static auto image_deleter = [](unsigned char* data) {
  stbi_image_free(data);
};
}

using PixelData =
    std::unique_ptr<unsigned char, decltype(detail::image_deleter)>;

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
