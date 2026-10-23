#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace robot_retargeter {

bool LoadNpyFloat64Matrix(
    const std::string& path,
    std::vector<double>& data,
    std::size_t& rows,
    std::size_t& cols);

bool SaveNpyFloat64Matrix(
    const std::string& path,
    const double* data,
    std::size_t rows,
    std::size_t cols);

}  // namespace robot_retargeter
