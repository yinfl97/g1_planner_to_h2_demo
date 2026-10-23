#include "robot_retargeter/npy_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace robot_retargeter {
namespace {

bool LittleEndianHost() {
  const uint16_t x = 1;
  return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

std::string ReadHeader(std::ifstream& in) {
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not a npy file");
  }
  uint8_t ver_major = 0;
  uint8_t ver_minor = 0;
  in.read(reinterpret_cast<char*>(&ver_major), 1);
  in.read(reinterpret_cast<char*>(&ver_minor), 1);
  uint32_t header_len = 0;
  if (ver_major == 1) {
    uint16_t len16 = 0;
    in.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    in.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  return header;
}

void ParseShape(const std::string& header, std::size_t& rows, std::size_t& cols, bool& fortran) {
  if (header.find("'descr': '<f8'") == std::string::npos &&
      header.find("\"descr\": \"<f8\"") == std::string::npos &&
      header.find("'descr': '<f8'") == std::string::npos) {
    if (header.find("<f8") == std::string::npos) {
      throw std::runtime_error("npy: expected little-endian float64 (<f8)");
    }
  }
  fortran = header.find("'fortran_order': True") != std::string::npos ||
            header.find("\"fortran_order\": True") != std::string::npos;

  const auto shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("npy: missing shape");
  }
  const auto lparen = header.find('(', shape_pos);
  const auto rparen = header.find(')', lparen);
  if (lparen == std::string::npos || rparen == std::string::npos) {
    throw std::runtime_error("npy: bad shape");
  }
  std::string inside = header.substr(lparen + 1, rparen - lparen - 1);
  for (char& c : inside) {
    if (c == ',') {
      c = ' ';
    }
  }
  std::istringstream iss(inside);
  std::size_t a = 0;
  std::size_t b = 1;
  if (!(iss >> a)) {
    throw std::runtime_error("npy: empty shape");
  }
  if (!(iss >> b)) {
    rows = 1;
    cols = a;
    return;
  }
  rows = a;
  cols = b;
}

}  // namespace

bool LoadNpyFloat64Matrix(
    const std::string& path,
    std::vector<double>& data,
    std::size_t& rows,
    std::size_t& cols) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  try {
    const std::string header = ReadHeader(in);
    bool fortran = false;
    ParseShape(header, rows, cols, fortran);
    data.resize(rows * cols);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
    if (!in) {
      return false;
    }
    if (fortran && rows > 1 && cols > 1) {
      std::vector<double> tmp = data;
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
          data[r * cols + c] = tmp[c * rows + r];
        }
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool SaveNpyFloat64Matrix(
    const std::string& path,
    const double* data,
    std::size_t rows,
    std::size_t cols) {
  if (!LittleEndianHost()) {
    return false;
  }
  std::ostringstream dict;
  dict << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << rows << ", " << cols
       << "), }";
  std::string header = dict.str();
  // npy v1: magic(6)+ver(2)+len(2)+header, padded to 64-byte alignment
  constexpr int prefix = 10;
  std::size_t pad = 64 - ((prefix + header.size() + 1) % 64);
  if (pad == 64) {
    pad = 0;
  }
  header.append(pad, ' ');
  header.push_back('\n');
  const uint16_t header_len = static_cast<uint16_t>(header.size());

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write("\x93NUMPY", 6);
  const uint8_t ver[2] = {1, 0};
  out.write(reinterpret_cast<const char*>(ver), 2);
  out.write(reinterpret_cast<const char*>(&header_len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(
      reinterpret_cast<const char*>(data),
      static_cast<std::streamsize>(rows * cols * sizeof(double)));
  return static_cast<bool>(out);
}

}  // namespace robot_retargeter
