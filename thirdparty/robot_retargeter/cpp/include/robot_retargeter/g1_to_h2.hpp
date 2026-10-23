#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace robot_retargeter {

constexpr int kQposSize = 36;

enum class QpSolver {
  kOsqp,
  kQpOases,
};

const char* QpSolverName(QpSolver solver);
QpSolver ParseQpSolver(const std::string& name);

struct G1ToH2Options {
  std::string retargeter_root;
  std::string g1_mjcf;
  std::string h2_mjcf;
  int max_iter = 8;
  double damping = 1.0;
  double lm_damping = 1.0;
  bool ground_align = true;
  double ground_clearance = 0.02;
  /// Degrees added to the knee opening when blending feet. Python only applies
  /// this when `enable_knee_angle_offset_degrees` is true (default false).
  double knee_angle_offset_degrees = 0.0;
  double knee_limit_lower_offset_deg = 10.0;
  double elbow_limit_lower_offset_deg = 20.0;
  QpSolver qp_solver = QpSolver::kOsqp;
  /// Skip qpOASES when the unconstrained LDLT solution satisfies all box
  /// bounds. Disabled by default so qpOASES always validates/regularises the
  /// solve; enable only when explicitly trading robustness for lower latency.
  bool qpoases_use_ldlt_fast_path = false;
};

/// Online G1 MuJoCo qpos (wxyz free-joint) → H2-29DoF MuJoCo qpos (wxyz).
/// Kinematics: Pinocchio (same backend as Pink). QP: OSQP.
/// FrameTask / solve_ik follow Pink (https://github.com/stephane-caron/pink);
/// mink is the MuJoCo port of the same formulation. ConfigurationLimit gain=0.95
/// matches mink/Python, not Pink's default 0.5.
class G1ToH2Retargeter {
 public:
  explicit G1ToH2Retargeter(G1ToH2Options options);
  ~G1ToH2Retargeter();

  G1ToH2Retargeter(const G1ToH2Retargeter&) = delete;
  G1ToH2Retargeter& operator=(const G1ToH2Retargeter&) = delete;

  void Reset();

  /// One frame. Arrays are length 36 (MuJoCo wxyz free-joint convention).
  void Step(const double g1_qpos_wxyz[kQposSize], double h2_qpos_wxyz[kQposSize]);

  /// Batch: row-major (N, 36). Resets warm-start before the first frame.
  void StepBatch(const double* g1_qpos_frames, int num_frames, double* h2_qpos_frames);

  const G1ToH2Options& options() const { return options_; }

 private:
  struct Impl;
  G1ToH2Options options_;
  Impl* impl_;
};

std::string DiscoverRetargeterRoot();
std::string ResolveDefaultG1Mjcf(const std::string& root);
std::string ResolveDefaultH2Mjcf(const std::string& root);

}  // namespace robot_retargeter
