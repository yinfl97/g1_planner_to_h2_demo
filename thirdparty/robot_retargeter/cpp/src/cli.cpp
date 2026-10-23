#include "robot_retargeter/g1_to_h2.hpp"
#include "robot_retargeter/npy_io.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  std::string in_path;
  std::string out_path;
  std::string g1_mjcf;
  std::string h2_mjcf;
  std::string root;
  int max_iter = 8;
  robot_retargeter::QpSolver qp_solver = robot_retargeter::QpSolver::kOsqp;
  bool dump_model = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--input-npy") {
      in_path = need("--input-npy");
    } else if (arg == "--output-npy") {
      out_path = need("--output-npy");
    } else if (arg == "--g1-mjcf") {
      g1_mjcf = need("--g1-mjcf");
    } else if (arg == "--h2-mjcf") {
      h2_mjcf = need("--h2-mjcf");
    } else if (arg == "--retargeter-root") {
      root = need("--retargeter-root");
    } else if (arg == "--max-iter") {
      max_iter = std::stoi(need("--max-iter"));
    } else if (arg == "--qp-solver") {
      qp_solver = robot_retargeter::ParseQpSolver(need("--qp-solver"));
    } else if (arg == "--dump-model") {
      dump_model = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout
          << "g1_to_h2_cli --input-npy in.npy --output-npy out.npy\n"
          << "  [--retargeter-root DIR] [--g1-mjcf PATH] [--h2-mjcf PATH]\n"
          << "  [--max-iter N] [--qp-solver osqp|qpoases]\n";
      return 0;
    } else {
      std::cerr << "unknown arg: " << arg << "\n";
      return 1;
    }
  }

  robot_retargeter::G1ToH2Options opt;
  opt.retargeter_root = root.empty() ? robot_retargeter::DiscoverRetargeterRoot() : root;
  opt.g1_mjcf = g1_mjcf.empty() ? robot_retargeter::ResolveDefaultG1Mjcf(opt.retargeter_root) : g1_mjcf;
  opt.h2_mjcf = h2_mjcf.empty() ? robot_retargeter::ResolveDefaultH2Mjcf(opt.retargeter_root) : h2_mjcf;
  opt.max_iter = max_iter;
  opt.qp_solver = qp_solver;

  if (dump_model) {
    std::cout << "root=" << opt.retargeter_root << "\n";
    std::cout << "g1=" << opt.g1_mjcf << "\n";
    std::cout << "h2=" << opt.h2_mjcf << "\n";
  }

  if (in_path.empty() || out_path.empty()) {
    std::cerr << "need --input-npy and --output-npy\n";
    return 1;
  }

  std::vector<double> in;
  std::size_t rows = 0;
  std::size_t cols = 0;
  if (!robot_retargeter::LoadNpyFloat64Matrix(in_path, in, rows, cols) || cols != 36) {
    std::cerr << "failed to load (N,36) float64 npy: " << in_path << "\n";
    return 1;
  }

  robot_retargeter::G1ToH2Retargeter retargeter(opt);
  std::vector<double> out(rows * 36, 0.0);
  const auto t0 = std::chrono::steady_clock::now();
  retargeter.StepBatch(in.data(), static_cast<int>(rows), out.data());
  const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!robot_retargeter::SaveNpyFloat64Matrix(out_path, out.data(), rows, 36)) {
    std::cerr << "failed to write " << out_path << "\n";
    return 1;
  }
  std::cout << "[g1_to_h2_cli] frames=" << rows << " in " << ms << " ms ("
            << (rows > 0 ? ms / static_cast<double>(rows) : 0.0) << " ms/frame) -> " << out_path
            << " solver=" << robot_retargeter::QpSolverName(qp_solver) << "\n";
  if (rows > 0) {
    std::cout << "  g1_z0=" << in[2] << " h2_z0=" << out[2] << "\n";
  }
  return 0;
}
