#include "robot_retargeter/g1_to_h2.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <osqp.h>

#ifdef ROBOT_RETARGETER_HAS_QPOASES
#include <qpOASES.hpp>
#endif

namespace robot_retargeter {
namespace {

namespace fs = std::filesystem;

constexpr int kNv = 35;
constexpr int kNReplay = 24;
constexpr double kEps = 1e-8;

using ProfileClock = std::chrono::steady_clock;
using VectorNv = Eigen::Matrix<double, kNv, 1>;
using MatrixNv = Eigen::Matrix<double, kNv, kNv>;
using FrameJacobian = Eigen::Matrix<double, 6, kNv>;

double ElapsedMs(ProfileClock::time_point start) {
  return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

ProfileClock::time_point ProfileStart(bool enabled) {
  return enabled ? ProfileClock::now() : ProfileClock::time_point{};
}

struct RetargetProfile {
  int frames = 0;
  int ik_iterations = 0;
  int ik_state_linearizations = 0;
  int qp_calls = 0;
  long long osqp_iterations = 0;
  int osqp_max_iterations = 0;
  long long qpoases_nwsr = 0;
  int qpoases_max_nwsr = 0;
  int qpoases_inits = 0;
  int qpoases_hotstarts = 0;
  int qpoases_reinits = 0;
  int qpoases_fast_paths = 0;
  int qpoases_unconstrained_feasible = 0;
  int qpoases_qps_with_active_bounds = 0;
  long long qpoases_active_bounds = 0;
  int qpoases_max_active_bounds = 0;
  double batch_ms = 0.0;
  double g1_fk_ms = 0.0;
  double keypoints_ms = 0.0;
  double ik_total_ms = 0.0;
  double ik_task_setup_ms = 0.0;
  double ik_state_linearize_ms = 0.0;
  double ik_jacobian_ms = 0.0;
  double ik_assembly_ms = 0.0;
  double ik_assembly_init_ms = 0.0;
  double ik_frame_jacobian_ms = 0.0;
  double ik_task_linearize_ms = 0.0;
  double ik_weighted_jacobian_ms = 0.0;
  double ik_normal_eq_ms = 0.0;
  double ik_constraints_ms = 0.0;
  double qp_csc_ms = 0.0;
  double qp_setup_ms = 0.0;
  double qp_solve_ms = 0.0;
  double qp_cleanup_ms = 0.0;
  double qp_fast_path_ms = 0.0;
  double qp_analysis_ms = 0.0;
  double ik_integrate_ms = 0.0;
  double output_ms = 0.0;
};

const char* kReplayNames[kNReplay] = {
    "hips",           "left_up_leg",  "left_leg",      "left_foot",     "left_toe",
    "right_up_leg",   "right_leg",    "right_foot",    "right_toe",     "spine1",
    "spine2",         "chest",        "neck",          "head",          "left_shoulder",
    "left_arm",       "left_fore_arm","left_hand",     "right_shoulder","right_arm",
    "right_fore_arm", "right_hand",   "hips_mean",     "shoulder_mean",
};

// Target robot_links key order from config/robot/h2_29dof.yaml
const char* kTargetLinkOrder[] = {
    "left_hip",      "left_thigh",    "left_calf",     "right_hip",     "right_thigh",
    "right_calf",    "neck",          "head",          "left_shoulder", "left_arm",
    "left_fore_arm", "right_shoulder","right_arm",     "right_fore_arm",
};
constexpr int kNTargetLinks = 14;

struct LinkPair {
  const char* name;
  const char* parent;
  const char* child;
};

// SOURCE_SEMANTIC_LINKS in robot_replay.py
const LinkPair kSemanticLinks[] = {
    {"left_hip", "hips_mean", "left_up_leg"},
    {"left_thigh", "left_up_leg", "left_leg"},
    {"left_calf", "left_leg", "left_foot"},
    {"right_hip", "hips_mean", "right_up_leg"},
    {"right_thigh", "right_up_leg", "right_leg"},
    {"right_calf", "right_leg", "right_foot"},
    {"neck", "hips_mean", "shoulder_mean"},
    {"head", "shoulder_mean", "head"},
    {"left_shoulder", "shoulder_mean", "left_arm"},
    {"left_arm", "left_arm", "left_fore_arm"},
    {"left_fore_arm", "left_fore_arm", "left_hand"},
    {"right_shoulder", "shoulder_mean", "right_arm"},
    {"right_arm", "right_arm", "right_fore_arm"},
    {"right_fore_arm", "right_fore_arm", "right_hand"},
};

struct IkTaskSpec {
  const char* keypoint;
  const char* h2_frame;
  double pos_w;
  double rot_w;
};

const IkTaskSpec kIkTasks[] = {
    {"hips_mean", "hips_sphere", 100.0, 0.0},
    {"left_hip", "left_hip_roll_link", 30.0, 3.0},
    {"left_thigh", "left_knee_link", 0.0, 3.0},
    {"left_calf", "left_ankle_pitch_link", 30.0, 3.0},
    {"right_hip", "right_hip_roll_link", 30.0, 3.0},
    {"right_thigh", "right_knee_link", 0.0, 3.0},
    {"right_calf", "right_ankle_pitch_link", 30.0, 3.0},
    {"head", "head_yaw_link", 0.0, 3.0},
    {"left_shoulder", "left_shoulder_roll_link", 30.0, 3.0},
    {"left_arm", "left_elbow_link", 10.0, 3.0},
    {"left_fore_arm", "left_wrist_yaw_link", 10.0, 1.0},
    {"right_shoulder", "right_shoulder_roll_link", 30.0, 3.0},
    {"right_arm", "right_elbow_link", 10.0, 3.0},
    {"right_fore_arm", "right_wrist_yaw_link", 10.0, 1.0},
};

// H2 robot_links body pairs for standing-pose length.
const LinkPair kH2LinkBodies[] = {
    {"left_hip", "hips_sphere", "left_hip_roll_link"},
    {"left_thigh", "left_hip_roll_link", "left_knee_link"},
    {"left_calf", "left_knee_link", "left_ankle_roll_link"},
    {"right_hip", "hips_sphere", "right_hip_roll_link"},
    {"right_thigh", "right_hip_roll_link", "right_knee_link"},
    {"right_calf", "right_knee_link", "right_ankle_roll_link"},
    {"neck", "hips_sphere", "neck_sphere"},
    {"head", "neck_sphere", "head_yaw_link"},
    {"left_shoulder", "neck_sphere", "left_shoulder_roll_link"},
    {"left_arm", "left_shoulder_roll_link", "left_elbow_link"},
    {"left_fore_arm", "left_elbow_link", "left_wrist_yaw_link"},
    {"right_shoulder", "neck_sphere", "right_shoulder_roll_link"},
    {"right_arm", "right_shoulder_roll_link", "right_elbow_link"},
    {"right_fore_arm", "right_elbow_link", "right_wrist_yaw_link"},
};

// G1 semantic → MJCF body (resolve_source_semantic_body_mapping + g1.yaml).
// Empty second name means a single body (duplicate first).
struct SemanticBody {
  const char* semantic;
  const char* a;
  const char* b;  // nullptr if not a mean
};

const SemanticBody kG1SemanticBodies[] = {
    {"hips_mean", "hips_sphere", "hips_sphere"},
    {"hips", "hips_sphere", nullptr},
    {"left_up_leg", "left_hip_roll_link", nullptr},
    {"left_leg", "left_knee_link", nullptr},
    {"left_foot", "left_ankle_roll_link", nullptr},
    {"right_up_leg", "right_hip_roll_link", nullptr},
    {"right_leg", "right_knee_link", nullptr},
    {"right_foot", "right_ankle_roll_link", nullptr},
    {"shoulder_mean", "neck_sphere", "neck_sphere"},
    {"spine1", "hips_sphere", nullptr},
    {"spine2", "hips_sphere", nullptr},
    {"chest", "neck_sphere", nullptr},
    {"neck", "neck_sphere", nullptr},
    {"head", "head_sphere", nullptr},
    {"left_shoulder", "left_shoulder_roll_link", nullptr},
    {"left_arm", "left_shoulder_roll_link", nullptr},
    {"left_fore_arm", "left_elbow_link", nullptr},
    {"left_hand", "left_wrist_yaw_link", nullptr},
    {"right_shoulder", "right_shoulder_roll_link", nullptr},
    {"right_arm", "right_shoulder_roll_link", nullptr},
    {"right_fore_arm", "right_elbow_link", nullptr},
    {"right_hand", "right_wrist_yaw_link", nullptr},
};

int ReplayIndex(const char* name) {
  for (int i = 0; i < kNReplay; ++i) {
    if (std::strcmp(kReplayNames[i], name) == 0) {
      return i;
    }
  }
  throw std::runtime_error(std::string("unknown replay body: ") + name);
}

bool ExistsFile(const std::string& p) {
  std::error_code ec;
  return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

Eigen::Quaterniond MatToQuat(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q;
  q = R;
  return q.normalized();
}

pinocchio::SE3 MujocoFreeToWorld(const double* mj) {
  const Eigen::Quaterniond quat(mj[3], mj[4], mj[5], mj[6]);
  return pinocchio::SE3(quat.normalized().toRotationMatrix(), Eigen::Vector3d(mj[0], mj[1], mj[2]));
}

void WorldToMujocoFree(const pinocchio::SE3& world, double* mj) {
  mj[0] = world.translation()[0];
  mj[1] = world.translation()[1];
  mj[2] = world.translation()[2];
  const Eigen::Quaterniond quat = MatToQuat(world.rotation());
  mj[3] = quat.w();
  mj[4] = quat.x();
  mj[5] = quat.y();
  mj[6] = quat.z();
}

pinocchio::SE3 PinocchioFreeToLocal(const Eigen::VectorXd& q) {
  const Eigen::Quaterniond quat(q[6], q[3], q[4], q[5]);
  return pinocchio::SE3(quat.normalized().toRotationMatrix(), q.head<3>());
}

void LocalToPinocchioFree(const pinocchio::SE3& local, Eigen::VectorXd& q) {
  q[0] = local.translation()[0];
  q[1] = local.translation()[1];
  q[2] = local.translation()[2];
  const Eigen::Quaterniond quat = MatToQuat(local.rotation());
  q[3] = quat.x();
  q[4] = quat.y();
  q[5] = quat.z();
  q[6] = quat.w();
}

// MJCF freejoint: Pinocchio bakes body pos into jointPlacements[1]; MuJoCo qpos is world.
// Pink's URDF+FreeFlyer path has identity placement so q is already world; MJCF is not.
void MujocoWorldToPinocchioQ(
    const pinocchio::Model& model, const double mj[kQposSize], Eigen::VectorXd& q) {
  q.resize(model.nq);
  const pinocchio::SE3 world = MujocoFreeToWorld(mj);
  const pinocchio::SE3 local = model.jointPlacements[1].actInv(world);
  LocalToPinocchioFree(local, q);
  for (int i = 0; i < 29; ++i) {
    q[7 + i] = mj[7 + i];
  }
}

void PinocchioQToMujocoWorld(
    const pinocchio::Model& model, const Eigen::VectorXd& q, double mj[kQposSize]) {
  const pinocchio::SE3 world = model.jointPlacements[1] * PinocchioFreeToLocal(q);
  WorldToMujocoFree(world, mj);
  for (int i = 0; i < 29; ++i) {
    mj[7 + i] = q[7 + i];
  }
}

Eigen::Quaterniond AverageWxyz(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  Eigen::Quaterniond bb = (a.dot(b) < 0.0) ? Eigen::Quaterniond(-b.w(), -b.x(), -b.y(), -b.z()) : b;
  Eigen::Quaterniond s(a.w() + bb.w(), a.x() + bb.x(), a.y() + bb.y(), a.z() + bb.z());
  s.normalize();
  return s;
}

Eigen::Quaterniond RotationBetween(const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
  Eigen::Vector3d a = from.normalized();
  Eigen::Vector3d b = to.normalized();
  Eigen::Vector3d axis = a.cross(b);
  const double axis_n = axis.norm();
  const double c = std::clamp(a.dot(b), -1.0, 1.0);
  const double angle = std::acos(c);
  if (axis_n < kEps) {
    axis = a.cross(Eigen::Vector3d::UnitX());
    if (axis.norm() < kEps) {
      axis = a.cross(Eigen::Vector3d::UnitY());
    }
  }
  axis.normalize();
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

Eigen::Vector3d TwoBoneKnee(
    const Eigen::Vector3d& hip,
    const Eigen::Vector3d& knee,
    const Eigen::Vector3d& foot,
    const Eigen::Vector3d& target_foot) {
  const Eigen::Vector3d upper = knee - hip;
  const Eigen::Vector3d lower = foot - knee;
  const double upper_len = upper.norm();
  const double lower_len = lower.norm();
  const Eigen::Vector3d orig_hf = foot - hip;
  const double orig_hf_len = orig_hf.norm();
  Eigen::Vector3d orig_u = orig_hf / std::max(orig_hf_len, kEps);
  const double knee_along = (knee - hip).dot(orig_u);
  const Eigen::Vector3d knee_proj = hip + knee_along * orig_u;
  Eigen::Vector3d bend_pref = knee - knee_proj;
  if (bend_pref.norm() <= kEps) {
    bend_pref = orig_u.cross(Eigen::Vector3d::UnitZ());
  }
  if (bend_pref.norm() <= kEps) {
    bend_pref = orig_u.cross(Eigen::Vector3d::UnitY());
  }
  bend_pref.normalize();

  Eigen::Vector3d plane_n = upper.cross(orig_hf);
  if (plane_n.norm() <= kEps) {
    plane_n = orig_u.cross(bend_pref);
  }
  if (plane_n.norm() <= kEps) {
    plane_n = orig_u.cross(Eigen::Vector3d::UnitZ());
  }
  if (plane_n.norm() <= kEps) {
    plane_n = orig_u.cross(Eigen::Vector3d::UnitY());
  }
  plane_n.normalize();

  const Eigen::Vector3d tgt_hf = target_foot - hip;
  const double tgt_len = tgt_hf.norm();
  Eigen::Vector3d tgt_u = tgt_hf / std::max(tgt_len, kEps);
  const double d = std::clamp(tgt_len, kEps, upper_len + lower_len - kEps);
  const double x = (upper_len * upper_len - lower_len * lower_len + d * d) / (2.0 * std::max(d, kEps));
  const double h = std::sqrt(std::max(upper_len * upper_len - x * x, 0.0));
  Eigen::Vector3d bend_dir = plane_n.cross(tgt_u);
  if (bend_dir.norm() <= kEps) {
    bend_dir = bend_pref - bend_pref.dot(tgt_u) * tgt_u;
  }
  if (bend_dir.norm() <= kEps) {
    bend_dir = tgt_u.cross(Eigen::Vector3d::UnitZ());
  }
  if (bend_dir.norm() <= kEps) {
    bend_dir = tgt_u.cross(Eigen::Vector3d::UnitY());
  }
  bend_dir.normalize();
  if (bend_dir.dot(bend_pref) < 0.0) {
    bend_dir = -bend_dir;
  }
  return hip + x * tgt_u + h * bend_dir;
}

pinocchio::FrameIndex RequireFrame(const pinocchio::Model& model, const std::string& name) {
  if (model.existFrame(name, pinocchio::BODY)) {
    return model.getFrameId(name, pinocchio::BODY);
  }
  if (model.existFrame(name)) {
    return model.getFrameId(name);
  }
  throw std::runtime_error("Pinocchio frame not found: " + name);
}

template <typename Derived>
void DenseToCscUpper(
    const Eigen::MatrixBase<Derived>& M,
    std::vector<c_float>& Ax,
    std::vector<c_int>& Ai,
    std::vector<c_int>& Ap) {
  const int n = static_cast<int>(M.rows());
  Ax.clear();
  Ai.clear();
  Ap.assign(n + 1, 0);
  for (int col = 0; col < n; ++col) {
    Ap[col] = static_cast<c_int>(Ax.size());
    for (int row = 0; row <= col; ++row) {
      const double v = 0.5 * (M(row, col) + M(col, row));
      if (std::abs(v) > 1e-18) {
        Ai.push_back(row);
        Ax.push_back(v);
      }
    }
  }
  Ap[n] = static_cast<c_int>(Ax.size());
}

template <typename Derived>
void DenseToCsc(
    const Eigen::MatrixBase<Derived>& M,
    std::vector<c_float>& Ax,
    std::vector<c_int>& Ai,
    std::vector<c_int>& Ap) {
  const int m = static_cast<int>(M.rows());
  const int n = static_cast<int>(M.cols());
  Ax.clear();
  Ai.clear();
  Ap.assign(n + 1, 0);
  for (int col = 0; col < n; ++col) {
    Ap[col] = static_cast<c_int>(Ax.size());
    for (int row = 0; row < m; ++row) {
      const double v = M(row, col);
      if (std::abs(v) > 1e-18) {
        Ai.push_back(row);
        Ax.push_back(v);
      }
    }
  }
  Ap[n] = static_cast<c_int>(Ax.size());
}

bool SolveQpOsqp(
    const MatrixNv& H,
    const VectorNv& c,
    const Eigen::MatrixXd& G,
    const Eigen::VectorXd& h,
    Eigen::VectorXd& x_out,
    RetargetProfile* profile) {
  const c_int n = static_cast<c_int>(H.rows());
  const c_int m = static_cast<c_int>(G.rows());
  std::vector<c_float> Px, Ax, q, l, u;
  std::vector<c_int> Pi, Pp, Ai, Ap;
  const auto csc_start = ProfileStart(profile != nullptr);
  DenseToCscUpper(H, Px, Pi, Pp);
  DenseToCsc(G, Ax, Ai, Ap);
  if (profile) {
    profile->qp_csc_ms += ElapsedMs(csc_start);
  }
  q.resize(n);
  for (c_int i = 0; i < n; ++i) {
    q[i] = c[i];
  }
  l.assign(m, -OSQP_INFTY);
  u.resize(m);
  for (c_int i = 0; i < m; ++i) {
    u[i] = h[i];
  }

  const auto setup_start = ProfileStart(profile != nullptr);
  OSQPData* data = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
  data->n = n;
  data->m = m;
  data->P = csc_matrix(n, n, static_cast<c_int>(Px.size()), Px.data(), Pi.data(), Pp.data());
  data->q = q.data();
  data->A = csc_matrix(m, n, static_cast<c_int>(Ax.size()), Ax.data(), Ai.data(), Ap.data());
  data->l = l.data();
  data->u = u.data();

  OSQPSettings* settings = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(settings);
  settings->verbose = 0;
  settings->warm_start = 0;
  settings->eps_abs = 1e-5;
  settings->eps_rel = 1e-5;
  settings->max_iter = 4000;
  settings->polish = 0;

  OSQPWorkspace* work = nullptr;
  const c_int setup_status = osqp_setup(&work, data, settings);
  if (profile) {
    profile->qp_setup_ms += ElapsedMs(setup_start);
  }
  bool ok = false;
  if (setup_status == 0 && work != nullptr) {
    const auto solve_start = ProfileStart(profile != nullptr);
    osqp_solve(work);
    if (profile) {
      profile->osqp_iterations += work->info->iter;
      profile->osqp_max_iterations = std::max(
          profile->osqp_max_iterations, static_cast<int>(work->info->iter));
    }
    if (work->info->status_val == OSQP_SOLVED || work->info->status_val == OSQP_SOLVED_INACCURATE) {
      x_out.resize(n);
      for (c_int i = 0; i < n; ++i) {
        x_out[i] = work->solution->x[i];
      }
      ok = true;
    }
    if (profile) {
      profile->qp_solve_ms += ElapsedMs(solve_start);
    }
  }
  const auto cleanup_start = ProfileStart(profile != nullptr);
  if (work) {
    osqp_cleanup(work);
  }
  if (data->P) {
    c_free(data->P);
  }
  if (data->A) {
    c_free(data->A);
  }
  c_free(data);
  c_free(settings);
  if (profile) {
    profile->qp_cleanup_ms += ElapsedMs(cleanup_start);
    ++profile->qp_calls;
  }
  return ok;
}

}  // namespace

const char* QpSolverName(QpSolver solver) {
  switch (solver) {
    case QpSolver::kOsqp:
      return "osqp";
    case QpSolver::kQpOases:
      return "qpoases";
  }
  return "unknown";
}

QpSolver ParseQpSolver(const std::string& name) {
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "osqp") {
    return QpSolver::kOsqp;
  }
  if (lower == "qpoases" || lower == "qp_oases" || lower == "qp-oases") {
    return QpSolver::kQpOases;
  }
  throw std::invalid_argument("unknown QP solver: " + name + " (expected osqp or qpoases)");
}

std::string DiscoverRetargeterRoot() {
  const auto resolve_root = [](const fs::path& candidate) -> std::string {
    if (ExistsFile((candidate / "asset/robot/g1_description/mjcf/g1.xml").string())) {
      return fs::weakly_canonical(candidate).string();
    }
    return {};
  };

  if (const char* env = std::getenv("ROBOT_RETARGETER_ROOT")) {
    if (const std::string resolved = resolve_root(env); !resolved.empty()) {
      return resolved;
    }
    throw std::runtime_error(
        "ROBOT_RETARGETER_ROOT does not contain asset/robot/g1_description/mjcf/g1.xml: " +
        std::string(env));
  }

#ifdef ROBOT_RETARGETER_SOURCE_ROOT
  if (const std::string resolved = resolve_root(ROBOT_RETARGETER_SOURCE_ROOT);
      !resolved.empty()) {
    return resolved;
  }
#endif

  const fs::path cwd = fs::current_path();
  for (fs::path p = cwd;; p = p.parent_path()) {
    if (const std::string resolved = resolve_root(p / "thirdparty/robot_retargeter");
        !resolved.empty()) {
      return resolved;
    }
    if (const std::string resolved = resolve_root(p); !resolved.empty()) {
      return resolved;
    }
    if (p == p.root_path()) {
      break;
    }
  }
  throw std::runtime_error("could not find robot_retargeter root (set ROBOT_RETARGETER_ROOT)");
}

std::string ResolveDefaultG1Mjcf(const std::string& root) {
  return (fs::path(root) / "asset/robot/g1_description/mjcf/g1.xml").string();
}

std::string ResolveDefaultH2Mjcf(const std::string& root) {
  return (fs::path(root) / "asset/robot/h2_29dof_sonic/h2_29dof.xml").string();
}

struct G1ToH2Retargeter::Impl {
  pinocchio::Model g1_model;
  pinocchio::Data g1_data;
  pinocchio::Model h2_model;
  pinocchio::Data h2_data;
  Eigen::VectorXd h2_q;
  bool has_prev = false;
  bool has_scaled_root = false;
  Eigen::Vector3d prev_unscaled_root = Eigen::Vector3d::Zero();
  Eigen::Vector3d prev_scaled_root = Eigen::Vector3d::Zero();
  double leg_disp_scale = 1.0;
  bool leg_disp_inited = false;
  bool profile_enabled = false;
  RetargetProfile profile;
#ifdef ROBOT_RETARGETER_HAS_QPOASES
  std::unique_ptr<qpOASES::SQProblem> qpoases_problem;
  bool qpoases_initialized = false;
  std::array<qpOASES::real_t, kNv * kNv> qpoases_h{};
  std::array<qpOASES::real_t, kNv> qpoases_g{};
  std::array<qpOASES::real_t, kNv> qpoases_lb{};
  std::array<qpOASES::real_t, kNv> qpoases_ub{};
#endif

  pinocchio::FrameIndex g1_body_a[kNReplay]{};
  pinocchio::FrameIndex g1_body_b[kNReplay]{};
  bool g1_is_mean[kNReplay]{};
  bool g1_has[kNReplay]{};

  std::unordered_map<std::string, double> h2_link_len;
  std::unordered_map<std::string, pinocchio::FrameIndex> h2_task_frames;
  std::unordered_map<std::string, std::vector<int>> h2_task_active_cols;
  struct LimitedDof {
    int iv;
    int iq;
  };
  std::vector<LimitedDof> limited_dofs;
  Eigen::VectorXd qmin;
  Eigen::VectorXd qmax;

  G1ToH2Options opt;

  explicit Impl(G1ToH2Options o) : opt(std::move(o)) {
#ifndef ROBOT_RETARGETER_HAS_QPOASES
    if (opt.qp_solver == QpSolver::kQpOases) {
      throw std::runtime_error(
          "qpOASES backend requested but robot_retargeter was built without "
          "ROBOT_RETARGETER_ENABLE_QPOASES=ON");
    }
#endif
    if (!ExistsFile(opt.g1_mjcf)) {
      throw std::runtime_error("G1 MJCF missing: " + opt.g1_mjcf);
    }
    if (!ExistsFile(opt.h2_mjcf)) {
      throw std::runtime_error("H2 MJCF missing: " + opt.h2_mjcf);
    }
    pinocchio::mjcf::buildModel(opt.g1_mjcf, g1_model);
    pinocchio::mjcf::buildModel(opt.h2_mjcf, h2_model);
    if (g1_model.nq != kQposSize || g1_model.nv != kNv) {
      throw std::runtime_error(
          "G1 pinocchio nq/nv expected 36/35, got " + std::to_string(g1_model.nq) + "/" +
          std::to_string(g1_model.nv));
    }
    if (h2_model.nq != kQposSize || h2_model.nv != kNv) {
      throw std::runtime_error(
          "H2 pinocchio nq/nv expected 36/35, got " + std::to_string(h2_model.nq) + "/" +
          std::to_string(h2_model.nv));
    }
    g1_data = pinocchio::Data(g1_model);
    h2_data = pinocchio::Data(h2_model);
    h2_q = pinocchio::neutral(h2_model);

    std::unordered_map<std::string, int> replay_slot;
    for (int i = 0; i < kNReplay; ++i) {
      replay_slot[kReplayNames[i]] = i;
    }
    for (const auto& sb : kG1SemanticBodies) {
      const int slot = replay_slot.at(sb.semantic);
      g1_has[slot] = true;
      g1_body_a[slot] = RequireFrame(g1_model, sb.a);
      if (sb.b != nullptr && std::strcmp(sb.a, sb.b) != 0) {
        g1_is_mean[slot] = true;
        g1_body_b[slot] = RequireFrame(g1_model, sb.b);
      } else if (sb.b != nullptr) {
        g1_is_mean[slot] = true;
        g1_body_a[slot] = RequireFrame(g1_model, sb.a);
        g1_body_b[slot] = RequireFrame(g1_model, sb.b);
      }
    }

    Eigen::VectorXd qn = pinocchio::neutral(h2_model);
    pinocchio::forwardKinematics(h2_model, h2_data, qn);
    pinocchio::updateFramePlacements(h2_model, h2_data);
    for (const auto& lp : kH2LinkBodies) {
      const auto fa = RequireFrame(h2_model, lp.parent);
      const auto fb = RequireFrame(h2_model, lp.child);
      h2_link_len[lp.name] = (h2_data.oMf[fb].translation() - h2_data.oMf[fa].translation()).norm();
    }

    for (const auto& t : kIkTasks) {
      if (t.pos_w == 0.0 && t.rot_w == 0.0) {
        continue;
      }
      const pinocchio::FrameIndex fid = RequireFrame(h2_model, t.h2_frame);
      h2_task_frames[t.keypoint] = fid;
      std::vector<int>& active_cols = h2_task_active_cols[t.keypoint];
      pinocchio::JointIndex joint = h2_model.frames[fid].parentJoint;
      while (joint > 0) {
        const int first = h2_model.idx_vs[joint];
        const int count = h2_model.nvs[joint];
        for (int col = first; col < first + count; ++col) {
          active_cols.push_back(col);
        }
        joint = h2_model.parents[joint];
      }
      std::sort(active_cols.begin(), active_cols.end());
    }

    qmin = h2_model.lowerPositionLimit;
    qmax = h2_model.upperPositionLimit;
    const double knee_off = opt.knee_limit_lower_offset_deg * M_PI / 180.0;
    const double elbow_off = opt.elbow_limit_lower_offset_deg * M_PI / 180.0;
    for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(h2_model.njoints); ++j) {
      if (h2_model.nqs[j] != 1) {
        continue;
      }
      const int iq = h2_model.idx_qs[j];
      const int iv = h2_model.idx_vs[j];
      const std::string& jn = h2_model.names[j];
      if (jn.find("knee_joint") != std::string::npos) {
        qmin[iq] += knee_off;
      }
      if (jn.find("elbow_joint") != std::string::npos) {
        qmin[iq] += elbow_off;
      }
      if (qmin[iq] > qmax[iq]) {
        std::swap(qmin[iq], qmax[iq]);
      }
      limited_dofs.push_back(LimitedDof{iv, iq});
    }

    std::cout << "[G1ToH2] pinocchio G1 nq=" << g1_model.nq << " H2 nq=" << h2_model.nq
              << " tasks=" << h2_task_frames.size() << " limited=" << limited_dofs.size()
              << " qp_solver=" << QpSolverName(opt.qp_solver)
              << std::endl;
  }

  void FkG1(
      const double g1_mj[kQposSize],
      Eigen::Matrix<double, kNReplay, 3>& pos,
      Eigen::Matrix<double, kNReplay, 4>& quat_wxyz) {
    Eigen::VectorXd q;
    MujocoWorldToPinocchioQ(g1_model, g1_mj, q);
    pinocchio::forwardKinematics(g1_model, g1_data, q);
    pinocchio::updateFramePlacements(g1_model, g1_data);
    pos.setZero();
    quat_wxyz.setZero();
    quat_wxyz.col(0).setOnes();
    for (int i = 0; i < kNReplay; ++i) {
      if (!g1_has[i]) {
        continue;
      }
      if (g1_is_mean[i]) {
        const Eigen::Vector3d pa = g1_data.oMf[g1_body_a[i]].translation();
        const Eigen::Vector3d pb = g1_data.oMf[g1_body_b[i]].translation();
        pos.row(i) = 0.5 * (pa + pb).transpose();
        const Eigen::Quaterniond qa = MatToQuat(g1_data.oMf[g1_body_a[i]].rotation());
        const Eigen::Quaterniond qb = MatToQuat(g1_data.oMf[g1_body_b[i]].rotation());
        const Eigen::Quaterniond qm = AverageWxyz(qa, qb);
        quat_wxyz(i, 0) = qm.w();
        quat_wxyz(i, 1) = qm.x();
        quat_wxyz(i, 2) = qm.y();
        quat_wxyz(i, 3) = qm.z();
      } else {
        pos.row(i) = g1_data.oMf[g1_body_a[i]].translation().transpose();
        const Eigen::Quaterniond qa = MatToQuat(g1_data.oMf[g1_body_a[i]].rotation());
        quat_wxyz(i, 0) = qa.w();
        quat_wxyz(i, 1) = qa.x();
        quat_wxyz(i, 2) = qa.y();
        quat_wxyz(i, 3) = qa.z();
      }
    }
  }

  void BuildKeypoints(
      const Eigen::Matrix<double, kNReplay, 3>& src_pos,
      const Eigen::Matrix<double, kNReplay, 4>& src_quat,
      Eigen::Matrix<double, 1 + kNTargetLinks, 3>& kps,
      Eigen::Matrix<double, 1 + kNTargetLinks, 4>& kq) {
    Eigen::Matrix<double, kNReplay, 3> scaled = src_pos;
    Eigen::Matrix<double, kNReplay, 4> sq = src_quat;

    std::unordered_map<std::string, Eigen::Vector3d> link_vec;
    std::unordered_map<std::string, double> link_len;
    for (const auto& lp : kSemanticLinks) {
      const int ip = ReplayIndex(lp.parent);
      const int ic = ReplayIndex(lp.child);
      const Eigen::Vector3d v = src_pos.row(ic).transpose() - src_pos.row(ip).transpose();
      const double n = v.norm();
      if (n <= kEps) {
        throw std::runtime_error(std::string("near-zero source link: ") + lp.name);
      }
      link_vec[lp.name] = v;
      link_len[lp.name] = n;
    }
    for (const auto& lp : kSemanticLinks) {
      const double tgt = h2_link_len.at(lp.name);
      const double scale = tgt / link_len[lp.name];
      const int ip = ReplayIndex(lp.parent);
      const int ic = ReplayIndex(lp.child);
      scaled.row(ic) = scaled.row(ip) + (scale * link_vec[lp.name]).transpose();
    }

    const int lhip = ReplayIndex("left_up_leg");
    const int lknee = ReplayIndex("left_leg");
    const int lfoot = ReplayIndex("left_foot");
    const int rhip = ReplayIndex("right_up_leg");
    const int rknee = ReplayIndex("right_leg");
    const int rfoot = ReplayIndex("right_foot");
    const Eigen::Vector3d left_h = scaled.row(lhip).transpose();
    const Eigen::Vector3d left_k = scaled.row(lknee).transpose();
    const Eigen::Vector3d left_f = scaled.row(lfoot).transpose();
    const Eigen::Vector3d right_h = scaled.row(rhip).transpose();
    const Eigen::Vector3d right_k = scaled.row(rknee).transpose();
    const Eigen::Vector3d right_f = scaled.row(rfoot).transpose();
    const Eigen::Vector3d lhk = left_k - left_h;
    const Eigen::Vector3d lkf = left_f - left_k;
    const Eigen::Vector3d rhk = right_k - right_h;
    const Eigen::Vector3d rkf = right_f - right_k;
    const double off = opt.knee_angle_offset_degrees * M_PI / 180.0;
    auto blend_foot = [&](const Eigen::Vector3d& hip, const Eigen::Vector3d& hk, const Eigen::Vector3d& kf,
                          const Eigen::Vector3d& foot) {
      const double a = std::acos(std::clamp(hk.normalized().dot(kf.normalized()), -1.0, 1.0));
      const double blended = a + off;
      const double hip_foot = std::sqrt(std::max(
          hk.squaredNorm() + kf.squaredNorm() + 2.0 * hk.norm() * kf.norm() * std::cos(blended),
          0.0));
      Eigen::Vector3d dir = foot - hip;
      const double dn = dir.norm();
      if (dn > kEps) {
        dir /= dn;
      }
      return Eigen::Vector3d(hip + dir * hip_foot);
    };
    const Eigen::Vector3d left_f_b = blend_foot(left_h, lhk, lkf, left_f);
    const Eigen::Vector3d right_f_b = blend_foot(right_h, rhk, rkf, right_f);
    const Eigen::Vector3d left_k_b = TwoBoneKnee(left_h, left_k, left_f, left_f_b);
    const Eigen::Vector3d right_k_b = TwoBoneKnee(right_h, right_k, right_f, right_f_b);

    const Eigen::Quaterniond lhip_d = RotationBetween(lhk, left_k_b - left_h);
    const Eigen::Quaterniond lknee_d = RotationBetween(lkf, left_f_b - left_k_b);
    const Eigen::Quaterniond rhip_d = RotationBetween(rhk, right_k_b - right_h);
    const Eigen::Quaterniond rknee_d = RotationBetween(rkf, right_f_b - right_k_b);
    auto apply_delta = [&](int slot, const Eigen::Quaterniond& d) {
      const Eigen::Quaterniond q(sq(slot, 0), sq(slot, 1), sq(slot, 2), sq(slot, 3));
      const Eigen::Quaterniond n = d * q;
      sq(slot, 0) = n.w();
      sq(slot, 1) = n.x();
      sq(slot, 2) = n.y();
      sq(slot, 3) = n.z();
    };
    apply_delta(lhip, lhip_d);
    apply_delta(lknee, lknee_d);
    apply_delta(rhip, rhip_d);
    apply_delta(rknee, rknee_d);
    scaled.row(lfoot) = left_f_b.transpose();
    scaled.row(rfoot) = right_f_b.transpose();
    scaled.row(lknee) = left_k_b.transpose();
    scaled.row(rknee) = right_k_b.transpose();

    if (!leg_disp_inited) {
      const double robot_thigh = 0.5 * (h2_link_len.at("left_thigh") + h2_link_len.at("right_thigh"));
      const double robot_calf = 0.5 * (h2_link_len.at("left_calf") + h2_link_len.at("right_calf"));
      const double robot_leg = std::sqrt(std::max(
          robot_thigh * robot_thigh + robot_calf * robot_calf +
              2.0 * robot_thigh * robot_calf * std::cos(opt.knee_angle_offset_degrees * M_PI / 180.0),
          0.0));
      const double skel_leg = 0.5 * (link_len.at("left_thigh") + link_len.at("left_calf") +
                                     link_len.at("right_thigh") + link_len.at("right_calf"));
      leg_disp_scale = robot_leg / std::max(skel_leg, kEps);
      leg_disp_inited = true;
    }

    const int hips_i = ReplayIndex("hips_mean");
    Eigen::Vector3d root = scaled.row(hips_i).transpose();
    Eigen::Vector3d scaled_root = root;
    if (!has_scaled_root) {
      scaled_root = root;
      prev_unscaled_root = root;
    } else {
      const Eigen::Vector3d delta = root - prev_unscaled_root;
      scaled_root = prev_scaled_root + leg_disp_scale * delta;
    }
    prev_unscaled_root = root;
    const Eigen::Vector3d translation = scaled_root - root;
    scaled.rowwise() += translation.transpose();
    prev_scaled_root = scaled_root;
    has_scaled_root = true;

    kps.row(0) = scaled.row(hips_i);
    kq.row(0) = sq.row(hips_i);
    for (int i = 0; i < kNTargetLinks; ++i) {
      const char* link = kTargetLinkOrder[i];
      const char* child = nullptr;
      for (const auto& lp : kSemanticLinks) {
        if (std::strcmp(lp.name, link) == 0) {
          child = lp.child;
          break;
        }
      }
      const int ic = ReplayIndex(child);
      kps.row(i + 1) = scaled.row(ic);
      kq.row(i + 1) = sq.row(ic);
    }

    if (opt.ground_align) {
      const int lc = 1 + 2;  // left_calf in target order (0-based after hips: left_hip, thigh, calf)
      const int rc = 1 + 5;  // right_calf
      const double min_z = std::min(kps(lc, 2), kps(rc, 2));
      kps.col(2).array() += opt.ground_clearance - min_z;
    }
  }

#ifdef ROBOT_RETARGETER_HAS_QPOASES
  void ResetQpOases() {
    qpoases_problem.reset();
    qpoases_initialized = false;
  }

  bool SolveQpQpoases(
      const MatrixNv& H,
      const VectorNv& c,
      Eigen::VectorXd& x_out,
      RetargetProfile* active_profile) {
    for (int row = 0; row < kNv; ++row) {
      qpoases_lb[row] = -qpOASES::INFTY;
      qpoases_ub[row] = qpOASES::INFTY;
    }
    for (const LimitedDof& limited : limited_dofs) {
      qpoases_ub[limited.iv] = static_cast<qpOASES::real_t>(
          0.95 * (qmax[limited.iq] - h2_q[limited.iq]));
      qpoases_lb[limited.iv] = static_cast<qpOASES::real_t>(
          -0.95 * (h2_q[limited.iq] - qmin[limited.iq]));
    }

    if (opt.qpoases_use_ldlt_fast_path) {
      // Most warm-started retarget QPs never touch a joint limit. For a
      // positive definite Hessian, the unconstrained LDLT solution is also the
      // exact box-QP solution whenever it lies inside all variable bounds.
      const auto fast_path_start = ProfileStart(active_profile != nullptr);
      const Eigen::LDLT<MatrixNv> ldlt(H);
      bool unconstrained_feasible = ldlt.info() == Eigen::Success;
      VectorNv unconstrained;
      if (unconstrained_feasible) {
        unconstrained = ldlt.solve(-c);
        unconstrained_feasible = unconstrained.allFinite();
      }
      if (unconstrained_feasible) {
        constexpr double kBoundTolerance = 1e-9;
        for (const LimitedDof& limited : limited_dofs) {
          const int iv = limited.iv;
          if (unconstrained[iv] < qpoases_lb[iv] - kBoundTolerance ||
              unconstrained[iv] > qpoases_ub[iv] + kBoundTolerance) {
            unconstrained_feasible = false;
            break;
          }
        }
      }
      if (active_profile) {
        active_profile->qp_fast_path_ms += ElapsedMs(fast_path_start);
        if (unconstrained_feasible) {
          ++active_profile->qpoases_unconstrained_feasible;
        }
      }
      if (unconstrained_feasible) {
        x_out = unconstrained;
        if (active_profile) {
          ++active_profile->qpoases_fast_paths;
          ++active_profile->qp_calls;
        }
        return true;
      }
    }

    const auto prepare_start = ProfileStart(active_profile != nullptr);
    for (int row = 0; row < kNv; ++row) {
      qpoases_g[row] = static_cast<qpOASES::real_t>(c[row]);
      for (int col = 0; col < kNv; ++col) {
        qpoases_h[static_cast<size_t>(row) * kNv + col] =
            static_cast<qpOASES::real_t>(0.5 * (H(row, col) + H(col, row)));
      }
    }
    if (active_profile) {
      active_profile->qp_csc_ms += ElapsedMs(prepare_start);
    }

    auto new_problem = [&]() {
      qpoases_problem = std::make_unique<qpOASES::SQProblem>(
          kNv, 0, qpOASES::HST_POSDEF);
      qpOASES::Options options;
      options.setToMPC();
      options.printLevel = qpOASES::PL_NONE;
      options.enableRegularisation = qpOASES::BT_TRUE;
      qpoases_problem->setOptions(options);
      qpoases_initialized = false;
    };
    if (!qpoases_problem) {
      new_problem();
    }

    qpOASES::int_t nwsr = 100;
    qpOASES::returnValue status;
    if (!qpoases_initialized) {
      const auto init_start = ProfileStart(active_profile != nullptr);
      status = qpoases_problem->init(
          qpoases_h.data(), qpoases_g.data(), nullptr,
          qpoases_lb.data(), qpoases_ub.data(), nullptr, nullptr, nwsr);
      if (active_profile) {
        active_profile->qp_setup_ms += ElapsedMs(init_start);
        ++active_profile->qpoases_inits;
      }
      qpoases_initialized = status == qpOASES::SUCCESSFUL_RETURN;
    } else {
      const auto hotstart_start = ProfileStart(active_profile != nullptr);
      status = qpoases_problem->hotstart(
          qpoases_h.data(), qpoases_g.data(), nullptr,
          qpoases_lb.data(), qpoases_ub.data(), nullptr, nullptr, nwsr);
      if (active_profile) {
        active_profile->qp_solve_ms += ElapsedMs(hotstart_start);
        ++active_profile->qpoases_hotstarts;
      }
      if (status != qpOASES::SUCCESSFUL_RETURN) {
        if (active_profile) {
          ++active_profile->qpoases_reinits;
        }
        new_problem();
        nwsr = 100;
        const auto reinit_start = ProfileStart(active_profile != nullptr);
        status = qpoases_problem->init(
            qpoases_h.data(), qpoases_g.data(), nullptr,
            qpoases_lb.data(), qpoases_ub.data(), nullptr, nullptr, nwsr);
        if (active_profile) {
          active_profile->qp_setup_ms += ElapsedMs(reinit_start);
          ++active_profile->qpoases_inits;
        }
        qpoases_initialized = status == qpOASES::SUCCESSFUL_RETURN;
      }
    }

    if (active_profile) {
      active_profile->qpoases_nwsr += nwsr;
      active_profile->qpoases_max_nwsr = std::max(
          active_profile->qpoases_max_nwsr, static_cast<int>(nwsr));
      ++active_profile->qp_calls;
    }
    if (status != qpOASES::SUCCESSFUL_RETURN) {
      return false;
    }
    x_out.resize(kNv);
    if (qpoases_problem->getPrimalSolution(x_out.data()) !=
        qpOASES::SUCCESSFUL_RETURN) {
      return false;
    }
    if (active_profile) {
      const auto analysis_start = ProfileStart(true);
      constexpr double kActiveTolerance = 1e-7;
      int active_bounds = 0;
      for (const LimitedDof& limited : limited_dofs) {
        const int iv = limited.iv;
        if (x_out[iv] <= qpoases_lb[iv] + kActiveTolerance ||
            x_out[iv] >= qpoases_ub[iv] - kActiveTolerance) {
          ++active_bounds;
        }
      }
      if (active_bounds > 0) {
        ++active_profile->qpoases_qps_with_active_bounds;
      }
      active_profile->qpoases_active_bounds += active_bounds;
      active_profile->qpoases_max_active_bounds = std::max(
          active_profile->qpoases_max_active_bounds, active_bounds);
      active_profile->qp_analysis_ms += ElapsedMs(analysis_start);
    }
    return true;
  }
#endif

    // Pink FrameTask + Task.compute_qp_objective + solve_ik (dt-cancelled):
    //   e = log(T_bt), J = -Jlog6(T_tb) * J_LOCAL
    //   H = WJᵀWJ + (damping + Σμ)I, c = -Weᵀ WJ, μ = lm_damping * ||We||²
    //   Δq from OSQP, q ← integrate(q, Δq)  ≡ Pink v=Δq/dt then integrate(v, dt)
    // ConfigurationLimit gain 0.95 matches mink (Pink default is 0.5).
    void SolveIk(
      const Eigen::Matrix<double, 1 + kNTargetLinks, 3>& kps,
      const Eigen::Matrix<double, 1 + kNTargetLinks, 4>& kq,
      double h2_mj[kQposSize]) {
    RetargetProfile* active_profile = profile_enabled ? &profile : nullptr;
    const auto task_setup_start = ProfileStart(active_profile != nullptr);
    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx["hips_mean"] = 0;
    for (int i = 0; i < kNTargetLinks; ++i) {
      name_to_idx[kTargetLinkOrder[i]] = i + 1;
    }

    if (!has_prev) {
      h2_q = pinocchio::neutral(h2_model);
      double seed_mj[kQposSize];
      PinocchioQToMujocoWorld(h2_model, h2_q, seed_mj);
      seed_mj[0] = kps(0, 0);
      seed_mj[1] = kps(0, 1);
      seed_mj[2] = kps(0, 2);
      seed_mj[3] = kq(0, 0);
      seed_mj[4] = kq(0, 1);
      seed_mj[5] = kq(0, 2);
      seed_mj[6] = kq(0, 3);
      MujocoWorldToPinocchioQ(h2_model, seed_mj, h2_q);
    }

    struct ActiveTask {
      pinocchio::FrameIndex fid;
      const std::vector<int>* active_cols;
      double pos_w;
      double rot_w;
      pinocchio::SE3 target;
      Eigen::Matrix<double, 6, 6> jlog;
      Eigen::Matrix<double, 6, 1> weighted_error;
    };
    std::vector<ActiveTask> tasks;
    tasks.reserve(h2_task_frames.size());
    for (const auto& spec : kIkTasks) {
      if (spec.pos_w == 0.0 && spec.rot_w == 0.0) {
        continue;
      }
      const auto it = name_to_idx.find(spec.keypoint);
      if (it == name_to_idx.end()) {
        continue;
      }
      const int idx = it->second;
      ActiveTask t;
      t.fid = h2_task_frames.at(spec.keypoint);
      t.active_cols = &h2_task_active_cols.at(spec.keypoint);
      t.pos_w = spec.pos_w;
      t.rot_w = spec.rot_w;
      const Eigen::Quaterniond target_quat =
          Eigen::Quaterniond(kq(idx, 0), kq(idx, 1), kq(idx, 2), kq(idx, 3)).normalized();
      t.target = pinocchio::SE3(target_quat.toRotationMatrix(), kps.row(idx).transpose());
      tasks.push_back(t);
    }
    if (active_profile) {
      active_profile->ik_task_setup_ms += ElapsedMs(task_setup_start);
    }

    auto linearize_state = [&]() {
      const auto state_start = ProfileStart(active_profile != nullptr);
      const auto jacobian_start = ProfileStart(active_profile != nullptr);
      pinocchio::computeJointJacobians(h2_model, h2_data, h2_q);
      pinocchio::updateFramePlacements(h2_model, h2_data);
      if (active_profile) {
        active_profile->ik_jacobian_ms += ElapsedMs(jacobian_start);
      }
      double acc = 0.0;
      for (auto& t : tasks) {
        const auto task_start = ProfileStart(active_profile != nullptr);
        const pinocchio::SE3& oMf = h2_data.oMf[t.fid];
        const pinocchio::SE3 T_tb = t.target.actInv(oMf);
        const pinocchio::Motion err = pinocchio::log6(oMf.actInv(t.target));
        Eigen::Matrix<double, 6, 1> e;
        e.head<3>() = err.linear();
        e.tail<3>() = err.angular();
        t.jlog = pinocchio::Jlog6(T_tb);
        t.weighted_error.head<3>() = -t.pos_w * e.head<3>();
        t.weighted_error.tail<3>() = -t.rot_w * e.tail<3>();
        acc += t.weighted_error.squaredNorm();
        if (active_profile) {
          active_profile->ik_task_linearize_ms += ElapsedMs(task_start);
        }
      }
      if (active_profile) {
        active_profile->ik_state_linearize_ms += ElapsedMs(state_start);
        ++active_profile->ik_state_linearizations;
      }
      return std::sqrt(acc);
    };

    double curr_err = linearize_state();
    FrameJacobian J;
    FrameJacobian weighted_jacobian;
    MatrixNv H;
    VectorNv c;
    for (int iter = 0; iter < opt.max_iter; ++iter) {
      if (active_profile) {
        ++active_profile->ik_iterations;
      }
      const auto assembly_start = ProfileStart(active_profile != nullptr);
      const auto assembly_init_start = ProfileStart(active_profile != nullptr);
      H.setZero();
      H.diagonal().setConstant(opt.damping);
      c.setZero();
      if (active_profile) {
        active_profile->ik_assembly_init_ms += ElapsedMs(assembly_init_start);
      }

      for (const auto& t : tasks) {
        const auto frame_jacobian_start = ProfileStart(active_profile != nullptr);
        J.setZero();
        pinocchio::getFrameJacobian(h2_model, h2_data, t.fid, pinocchio::LOCAL, J);
        if (active_profile) {
          active_profile->ik_frame_jacobian_ms += ElapsedMs(frame_jacobian_start);
        }
        const auto weighted_jacobian_start = ProfileStart(active_profile != nullptr);
        weighted_jacobian.noalias() = -t.jlog * J;
        weighted_jacobian.topRows<3>() *= t.pos_w;
        weighted_jacobian.bottomRows<3>() *= t.rot_w;
        if (active_profile) {
          active_profile->ik_weighted_jacobian_ms += ElapsedMs(weighted_jacobian_start);
        }
        const auto normal_eq_start = ProfileStart(active_profile != nullptr);

        // A frame Jacobian only has columns on the frame's ancestor joint
        // chain. Updating that structural non-zero block avoids a dense
        // 35x6 * 6x35 product for every task while preserving the same QP.
        const std::vector<int>& active_cols = *t.active_cols;
        for (std::size_t a = 0; a < active_cols.size(); ++a) {
          const int col = active_cols[a];
          c[col] -= weighted_jacobian.col(col).dot(t.weighted_error);
          for (std::size_t b = a; b < active_cols.size(); ++b) {
            const int row = active_cols[b];
            H(row, col) += weighted_jacobian.col(row).dot(weighted_jacobian.col(col));
          }
        }
        H.diagonal().array() += opt.lm_damping * t.weighted_error.squaredNorm();
        if (active_profile) {
          active_profile->ik_normal_eq_ms += ElapsedMs(normal_eq_start);
        }
      }
      H.template triangularView<Eigen::StrictlyUpper>() = H.transpose();
      if (active_profile) {
        active_profile->ik_assembly_ms += ElapsedMs(assembly_start);
      }

      Eigen::VectorXd dq;
      bool qp_ok = false;
      if (opt.qp_solver == QpSolver::kOsqp) {
        const auto constraints_start = ProfileStart(active_profile != nullptr);
        const int nl = static_cast<int>(limited_dofs.size());
        Eigen::MatrixXd G(2 * nl, h2_model.nv);
        Eigen::VectorXd hh(2 * nl);
        G.setZero();
        for (int i = 0; i < nl; ++i) {
          const int iv = limited_dofs[i].iv;
          const int iq = limited_dofs[i].iq;
          G(i, iv) = 1.0;
          G(nl + i, iv) = -1.0;
          const double dq_max = qmax[iq] - h2_q[iq];
          const double dq_min = h2_q[iq] - qmin[iq];
          hh[i] = 0.95 * dq_max;
          hh[nl + i] = 0.95 * dq_min;
        }
        if (active_profile) {
          active_profile->ik_constraints_ms += ElapsedMs(constraints_start);
        }
        qp_ok = SolveQpOsqp(H, c, G, hh, dq, active_profile);
      } else {
#ifdef ROBOT_RETARGETER_HAS_QPOASES
        qp_ok = SolveQpQpoases(H, c, dq, active_profile);
#endif
      }
      if (!qp_ok) {
        break;
      }
      const auto integrate_start = ProfileStart(active_profile != nullptr);
      Eigen::VectorXd q_next(h2_model.nq);
      pinocchio::integrate(h2_model, h2_q, dq, q_next);
      pinocchio::normalize(h2_model, q_next);
      h2_q = q_next;
      if (active_profile) {
        active_profile->ik_integrate_ms += ElapsedMs(integrate_start);
      }
      const double next_err = linearize_state();
      if (curr_err - next_err <= 0.001) {
        break;
      }
      curr_err = next_err;
    }
    has_prev = true;
    const auto output_start = ProfileStart(active_profile != nullptr);
    PinocchioQToMujocoWorld(h2_model, h2_q, h2_mj);
    if (active_profile) {
      active_profile->output_ms += ElapsedMs(output_start);
    }
  }
};

G1ToH2Retargeter::G1ToH2Retargeter(G1ToH2Options options) : options_(std::move(options)) {
  if (options_.retargeter_root.empty()) {
    options_.retargeter_root = DiscoverRetargeterRoot();
  }
  if (options_.g1_mjcf.empty()) {
    options_.g1_mjcf = ResolveDefaultG1Mjcf(options_.retargeter_root);
  }
  if (options_.h2_mjcf.empty()) {
    options_.h2_mjcf = ResolveDefaultH2Mjcf(options_.retargeter_root);
  }
  impl_ = new Impl(options_);
}

G1ToH2Retargeter::~G1ToH2Retargeter() {
  delete impl_;
}

void G1ToH2Retargeter::Reset() {
  impl_->has_prev = false;
  impl_->has_scaled_root = false;
  impl_->leg_disp_inited = false;
  impl_->h2_q = pinocchio::neutral(impl_->h2_model);
#ifdef ROBOT_RETARGETER_HAS_QPOASES
  impl_->ResetQpOases();
#endif
}

void G1ToH2Retargeter::Step(const double g1_qpos_wxyz[kQposSize], double h2_qpos_wxyz[kQposSize]) {
  RetargetProfile* active_profile = impl_->profile_enabled ? &impl_->profile : nullptr;
  const auto fk_start = ProfileStart(active_profile != nullptr);
  Eigen::Matrix<double, kNReplay, 3> pos;
  Eigen::Matrix<double, kNReplay, 4> quat;
  impl_->FkG1(g1_qpos_wxyz, pos, quat);
  if (active_profile) {
    active_profile->g1_fk_ms += ElapsedMs(fk_start);
  }
  const auto keypoints_start = ProfileStart(active_profile != nullptr);
  Eigen::Matrix<double, 1 + kNTargetLinks, 3> kps;
  Eigen::Matrix<double, 1 + kNTargetLinks, 4> kq;
  impl_->BuildKeypoints(pos, quat, kps, kq);
  if (active_profile) {
    active_profile->keypoints_ms += ElapsedMs(keypoints_start);
  }
  const auto ik_start = ProfileStart(active_profile != nullptr);
  impl_->SolveIk(kps, kq, h2_qpos_wxyz);
  if (active_profile) {
    active_profile->ik_total_ms += ElapsedMs(ik_start);
    ++active_profile->frames;
  }
}

void G1ToH2Retargeter::StepBatch(
    const double* g1_qpos_frames, int num_frames, double* h2_qpos_frames) {
  const char* profile_env = std::getenv("G1_TO_H2_PROFILE");
  impl_->profile_enabled = profile_env != nullptr && std::strcmp(profile_env, "0") != 0;
  impl_->profile = RetargetProfile{};
  const auto batch_start = ProfileStart(impl_->profile_enabled);
  Reset();
  for (int f = 0; f < num_frames; ++f) {
    Step(g1_qpos_frames + static_cast<std::size_t>(f) * kQposSize,
         h2_qpos_frames + static_cast<std::size_t>(f) * kQposSize);
  }
  if (impl_->profile_enabled) {
    RetargetProfile& p = impl_->profile;
    p.batch_ms = ElapsedMs(batch_start);
    const double denom = std::max(p.batch_ms, 1e-12);
    const double qp_total = p.qp_fast_path_ms + p.qp_csc_ms + p.qp_setup_ms +
                            p.qp_solve_ms + p.qp_cleanup_ms;
    auto print = [&](const char* name, double ms) {
      std::cout << "[G1ToH2 profile] " << std::left << std::setw(20) << name
                << std::right << std::fixed << std::setprecision(3) << std::setw(10) << ms
                << " ms  " << std::setw(6) << std::setprecision(1) << (100.0 * ms / denom)
                << "%\n";
    };
    std::cout << "[G1ToH2 profile] solver=" << QpSolverName(impl_->opt.qp_solver)
              << " frames=" << p.frames
              << " ik_iterations=" << p.ik_iterations
              << " (" << std::fixed << std::setprecision(2)
              << (p.frames > 0 ? static_cast<double>(p.ik_iterations) / p.frames : 0.0)
              << "/frame) state_linearizations=" << p.ik_state_linearizations
              << " qp_calls=" << p.qp_calls << " nv=" << kNv
              << " bounds=" << impl_->limited_dofs.size() << "\n";
    if (impl_->opt.qp_solver == QpSolver::kOsqp) {
      std::cout << "[G1ToH2 profile] osqp_iterations=" << p.osqp_iterations
                << " (" << std::fixed << std::setprecision(2)
                << (p.qp_calls > 0 ? static_cast<double>(p.osqp_iterations) / p.qp_calls : 0.0)
                << "/QP) max=" << p.osqp_max_iterations << "\n";
    } else {
      std::cout << "[G1ToH2 profile] qpoases_nwsr=" << p.qpoases_nwsr
                << " (" << std::fixed << std::setprecision(2)
                << (p.qp_calls > 0 ? static_cast<double>(p.qpoases_nwsr) / p.qp_calls : 0.0)
                << "/QP) max=" << p.qpoases_max_nwsr
                << " init=" << p.qpoases_inits
                << " hotstart=" << p.qpoases_hotstarts
                << " reinit=" << p.qpoases_reinits
                << " fast_path=" << p.qpoases_fast_paths
                << " solver_calls=" << (p.qp_calls - p.qpoases_fast_paths) << "\n";
      std::cout << "[G1ToH2 profile] unconstrained_feasible="
                << p.qpoases_unconstrained_feasible << "/" << p.qp_calls
                << " (" << std::fixed << std::setprecision(1)
                << (p.qp_calls > 0
                        ? 100.0 * p.qpoases_unconstrained_feasible / p.qp_calls
                        : 0.0)
                << "%) qps_with_active_bounds=" << p.qpoases_qps_with_active_bounds
                << " mean_active_bounds=" << std::setprecision(3)
                << (p.qp_calls > 0
                        ? static_cast<double>(p.qpoases_active_bounds) / p.qp_calls
                        : 0.0)
                << " max_active_bounds=" << p.qpoases_max_active_bounds << "\n";
    }
    print("batch total", p.batch_ms);
    print("G1 FK", p.g1_fk_ms);
    print("keypoint build", p.keypoints_ms);
    print("IK total", p.ik_total_ms);
    print("  task setup", p.ik_task_setup_ms);
    print("  state linearize", p.ik_state_linearize_ms);
    print("    Jacobians/FK", p.ik_jacobian_ms);
    print("    task error/Jlog", p.ik_task_linearize_ms);
    print("  H/c assembly", p.ik_assembly_ms);
    print("    H/c init", p.ik_assembly_init_ms);
    print("    frame Jacobian", p.ik_frame_jacobian_ms);
    print("    weighted Jacobian", p.ik_weighted_jacobian_ms);
    print("    normal equations", p.ik_normal_eq_ms);
    print("  bounds build", p.ik_constraints_ms);
    print("  QP total", qp_total);
    if (impl_->opt.qp_solver == QpSolver::kOsqp) {
      print("    dense->CSC", p.qp_csc_ms);
      print("    osqp_setup", p.qp_setup_ms);
      print("    osqp_solve", p.qp_solve_ms);
      print("    cleanup", p.qp_cleanup_ms);
    } else {
      print("    LDLT fast path", p.qp_fast_path_ms);
      print("    dense prepare", p.qp_csc_ms);
      print("    qpoases init", p.qp_setup_ms);
      print("    qpoases hotstart", p.qp_solve_ms);
      print("    profile analysis", p.qp_analysis_ms);
    }
    print("  integrate", p.ik_integrate_ms);
    print("  output convert", p.output_ms);
  }
}

}  // namespace robot_retargeter
