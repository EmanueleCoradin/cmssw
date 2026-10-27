#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_EmbeddedFallbackModel_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_EmbeddedFallbackModel_h

// Read the CMSSW-specific fallback ONNX ModelProto stored in ModelProto.metadata_props.
// This deliberately reads only protobuf's outer length-delimited fields. 
// It skips the EPContext graph without materializing the TensorRT plan.
// ONNX ModelProto.metadata_props is field 14, StringStringEntryProto.key/value are 1/2.

#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace cms::Ort::alpakatools {
  namespace detail {
    inline std::uint64_t readPbVarint(std::istream& in) {
      std::uint64_t value = 0;
      for (unsigned shift = 0; shift <= 63; shift += 7) {
        const int c = in.get();
        if (c == EOF)
          throw std::runtime_error("Truncated ONNX protobuf varint");
        const auto byte = static_cast<unsigned char>(c);
        if (shift == 63 and byte > 1)
          throw std::runtime_error("Invalid ONNX protobuf varint");
        value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0)
          return value;
      }
      throw std::runtime_error("Overlong ONNX protobuf varint");
    }

    inline void skipPbBytes(std::istream& in, std::uint64_t length) {
      if (length > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("ONNX protobuf field is too large");
      const auto here = in.tellg();
      if (here < 0)
        throw std::runtime_error("Cannot find current position in ONNX model");
      in.seekg(0, std::ios::end);
      const auto end = in.tellg();
      if (end < 0 or (end - here) < static_cast<std::streamoff>(length))
        throw std::runtime_error("Truncated ONNX protobuf field");
      in.seekg(here + static_cast<std::streamoff>(length));
      if (!in)
        throw std::runtime_error("Failed to skip ONNX protobuf field");
    }

    inline std::string readPbBytes(std::istream& in, std::uint64_t length) {
      // Avoid allocating unbounded memory for malformed metadata.
      constexpr std::uint64_t maxMetadataBytes = 1024ULL * 1024 * 1024;
      if (length > maxMetadataBytes)
        throw std::runtime_error("Embedded ONNX fallback metadata exceeds 1 GiB");
      std::string data(static_cast<std::size_t>(length), '\0');
      if (!in.read(data.data(), static_cast<std::streamsize>(length)))
        throw std::runtime_error("Truncated ONNX metadata");
      return data;
    }

    inline void skipPbField(std::istream& in, std::uint64_t wireType) {
      if (wireType == 0) {
        (void)readPbVarint(in);
      } else if (wireType == 1) {
        skipPbBytes(in, 8);
      } else if (wireType == 2) {
        skipPbBytes(in, readPbVarint(in));
      } else if (wireType == 5) {
        skipPbBytes(in, 4);
      } else {
        throw std::runtime_error("Unsupported ONNX protobuf wire type");
      }
    }

    inline std::pair<std::string, std::string> parsePbStringPair(const std::string& bytes) {
      std::istringstream in(bytes, std::ios::in | std::ios::binary);
      std::pair<std::string, std::string> result;
      while (in.peek() != EOF) {
        const auto tag = readPbVarint(in);
        if (tag == 0)
          throw std::runtime_error("Invalid ONNX metadata entry");
        if ((tag >> 3) == 1 and (tag & 7) == 2) {
          result.first = readPbBytes(in, readPbVarint(in));
        } else if ((tag >> 3) == 2 and (tag & 7) == 2) {
          result.second = readPbBytes(in, readPbVarint(in));
        } else {
          skipPbField(in, tag & 7);
        }
      }
      return result;
    }

    inline std::string decodePbBase64(std::string_view encoded) {
      if (encoded.empty() or (encoded.size() % 4 != 0))
        throw std::runtime_error("Invalid Base64 encoded fallback model");
      auto value = [](char c) -> int {
        if (c >= 'A' and c <= 'Z') return c - 'A';
        if (c >= 'a' and c <= 'z') return c - 'a' + 26;
        if (c >= '0' and c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
      };
      std::string decoded;
      decoded.reserve(encoded.size() / 4 * 3);
      for (std::size_t i = 0; i < encoded.size(); i += 4) {
        const bool last = (i + 4 == encoded.size());
        const int a = value(encoded[i]);
        const int b = value(encoded[i + 1]);
        const int c = encoded[i + 2] == '=' ? -2 : value(encoded[i + 2]);
        const int d = encoded[i + 3] == '=' ? -2 : value(encoded[i + 3]);
        if (a < 0 or b < 0 or c == -1 or d == -1 or
            (c == -2 and d != -2) or ((c == -2 or d == -2) and not last) or
            (c == -2 and (b & 0xf) != 0) or (d == -2 and c >= 0 and (c & 0x3) != 0))
          throw std::runtime_error("Invalid Base64 encoded fallback model");
        decoded.push_back(static_cast<char>((a << 2) | (b >> 4)));
        if (c >= 0) {
          decoded.push_back(static_cast<char>(((b & 15) << 4) | (c >> 2)));
          if (d >= 0) decoded.push_back(static_cast<char>(((c & 3) << 6) | d));
        }
      }
      return decoded;
    }
  }  // namespace detail

  inline std::optional<std::string> readEmbeddedFallbackModel(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open ONNX model: " + path);
    std::optional<std::string> format;
    std::optional<std::string> payload;
    while (in.peek() != EOF) {
      const auto tag = detail::readPbVarint(in);
      if (tag == 0) throw std::runtime_error("Invalid top-level ONNX protobuf tag");
      if ((tag >> 3) == 14 and (tag & 7) == 2) {
        auto entry = detail::parsePbStringPair(detail::readPbBytes(in, detail::readPbVarint(in)));
        if (entry.first == "cmssw.tensorrt_fallback.format") {
          if (format) throw std::runtime_error("Duplicate fallback format metadata");
          format = std::move(entry.second);
        } else if (entry.first == "cmssw.tensorrt_fallback.onnx_base64") {
          if (payload) throw std::runtime_error("Duplicate fallback model metadata");
          payload = std::move(entry.second);
        }
      } else {
        detail::skipPbField(in, tag & 7);
      }
    }
    if (!format and !payload) return std::nullopt;
    if (format != "onnx-base64-v1" or !payload)
      throw std::runtime_error("Missing or unsupported CMSSW TensorRT fallback metadata");
    return detail::decodePbBase64(*payload);
  }
}  // namespace cms::Ort::alpakatools

#endif
