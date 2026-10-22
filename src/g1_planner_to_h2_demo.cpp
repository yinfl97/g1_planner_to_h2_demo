/**
 * @file g1_planner_to_h2_demo.cpp
 * @brief Standalone G1 kinematic-planner -> H2 retarget visualizer.
 *
 * This executable intentionally has no policy, encoder, DDS, controller, or
 * actuator output. It shares the production H2 deploy trajectory pipeline:
 *
 *   TensorRT G1 planner -> 30/50 Hz resample -> 8-frame G1 cross-fade
 *   -> G1ToH2Retargeter::StepBatch -> post-IK pelvis z offset
 *   -> overlapping-window H2 root alignment and pose cross-fade.
 *
 * A split-screen MuJoCo viewer displays the synchronized G1 and H2 kinematic
 * trajectories and exports both clips plus per-frame comparison metrics.
 */

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "g1_planner_defaults.hpp"
#include "g1_to_h2_retarget.hpp"
#include "glfw_minimal.hpp"
#include "localmotion_kplanner_tensorrt.hpp"
#include "planner_motion_pipeline.hpp"

#ifndef G1_PLANNER_TO_H2_DEMO_ROOT
#define G1_PLANNER_TO_H2_DEMO_ROOT "."
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kPlannerCapacity = 1500;
constexpr size_t kMaxRandomPlannerUpdates = 2000;
constexpr double kPlaybackFps = 50.0;
constexpr double kControlPeriodMs = 1000.0 / kPlaybackFps;

enum class RetargetExecution {
  kSync,
  kAsync,
};

enum class H2ContinuityMode {
  kNone,
  kRoot,
  kBlend,
};

const char* RetargetExecutionName(RetargetExecution execution) {
  return execution == RetargetExecution::kAsync ? "async" : "sync";
}

const char* H2ContinuityModeName(H2ContinuityMode mode) {
  if (mode == H2ContinuityMode::kRoot) return "root";
  if (mode == H2ContinuityMode::kBlend) return "blend";
  return "none";
}

struct Args {
  fs::path planner = fs::path(G1_PLANNER_TO_H2_DEMO_ROOT) /
                     "assets/planner/planner_sonic.onnx";
  fs::path g1_mjcf = fs::path(G1_PLANNER_TO_H2_DEMO_ROOT) /
                     "assets/g1/scene_29dof.xml";
  fs::path h2_mjcf = fs::path(G1_PLANNER_TO_H2_DEMO_ROOT) /
                     "assets/h2/scene_h2_29dof.xml";
  fs::path output;
  std::string retargeter_root;
  int mode = static_cast<int>(LocomotionMode::WALK);
  float speed = -1.0f;
  float height = -1.0f;
  std::array<float, 3> movement{1.0f, 0.0f, 0.0f};
  std::array<float, 3> facing{1.0f, 0.0f, 0.0f};
  int seed = 1234;
  int replans = 3;
  int replan_frames = -1;
  bool random_locomotion = false;
  bool live_random = false;
  bool random_core_only = false;
  int random_clips = 3;
  double random_duration_min_s = 4.0;
  double random_duration_max_s = 10.0;
  double random_gap_s = 0.0;
  float random_speed_max = 1.5f;
  int retarget_max_iter = 8;
  robot_retargeter::QpSolver retarget_solver =
      robot_retargeter::QpSolver::kOsqp;
  bool retarget_use_ldlt_fast_path = false;
  RetargetExecution retarget_execution = RetargetExecution::kSync;
  H2ContinuityMode h2_continuity = H2ContinuityMode::kBlend;
  int h2_blend_frames = 8;
  bool independent_g1_timeline = true;
  int h2_root_z_smooth_radius = 3;
  double root_z_offset = kH2RetargetRootZOffset;
  bool ground_align = true;
  bool fp16 = false;
  bool headless = false;
  bool autoplay = true;
  bool show_solve_stalls = false;
  int width = 1500;
  int height_px = 760;
};

std::string Timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&now_c, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

int ParseMode(const std::string& text) {
  char* end = nullptr;
  const long numeric = std::strtol(text.c_str(), &end, 10);
  if (end && *end == '\0') {
    if (numeric < 0 || numeric >= 27) {
      throw std::runtime_error("mode must be in [0, 26]");
    }
    return static_cast<int>(numeric);
  }
  static const std::map<std::string, int> names = {
      {"idle", 0},          {"slow_walk", 1},     {"walk", 2},
      {"run", 3},           {"squat", 4},         {"kneel_two", 5},
      {"kneel", 6},         {"lying", 7},         {"crawl", 8},
      {"boxing", 9},        {"walk_boxing", 10},  {"left_punch", 11},
      {"right_punch", 12},  {"random_punch", 13}, {"elbow_crawl", 14},
      {"left_hook", 15},    {"right_hook", 16},   {"jump", 17},
      {"stealth", 18},      {"injured", 19},      {"ledge", 20},
      {"carry", 21},        {"stealth2", 22},     {"happy", 23},
      {"zombie", 24},       {"gun", 25},          {"scared", 26},
  };
  const auto it = names.find(Lower(text));
  if (it == names.end()) {
    throw std::runtime_error("unknown locomotion mode: " + text);
  }
  return it->second;
}

std::array<float, 3> ParseVec3(std::string text) {
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream stream(text);
  std::array<float, 3> value{};
  if (!(stream >> value[0] >> value[1] >> value[2])) {
    throw std::runtime_error("expected x,y,z, got: " + text);
  }
  return value;
}

int DefaultReplanFrames(int mode) {
  if (mode == static_cast<int>(LocomotionMode::RUN)) return 5;       // 0.1 s
  if (mode == static_cast<int>(LocomotionMode::CRAWLING)) return 10; // 0.2 s
  return 50;                                                         // 1.0 s
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Generate G1 MotionBricks planner trajectories, run the production H2\n"
      << "retarget pipeline, and compare both kinematically (no policy/control).\n\n"
      << "Options:\n"
      << "  --planner PATH          planner_sonic.onnx\n"
      << "  --mode NAME|0..26       default: walk\n"
      << "  --speed MPS             default: -1 (planner default)\n"
      << "  --height M              default: -1 (planner default)\n"
      << "  --move X,Y,Z            default: 1,0,0\n"
      << "  --face X,Y,Z            default: 1,0,0\n"
      << "  --seed N                default: 1234\n"
      << "  --replans N             repeated live-style planner windows (default: 3)\n"
      << "  --replan-frames N       frames consumed before replan; mode default if omitted\n"
      << "  --random-locomotion     continuous weighted random locomotion commands\n"
      << "  --live-random           generate/retarget/display random commands online\n"
      << "  --random-clips N        random command segments (default: 3)\n"
      << "  --random-duration-min S minimum random segment duration (default: 4)\n"
      << "  --random-duration-max S maximum random segment duration (default: 10)\n"
      << "  --random-gap S          idle gap between segments (default: 0)\n"
      << "  --random-speed-max MPS  cap sampled target speed (default: 1.5)\n"
      << "  --random-core-only      sample only slowWalk/walk/run\n"
      << "  --root-z-offset M       post-IK H2 pelvis lift (default: 0.05)\n"
      << "  --retarget-max-iter N   IK iterations per frame (default: 8)\n"
      << "  --retarget-solver NAME  osqp|qpoases (default: osqp)\n"
      << "  --retarget-use-ldlt     enable experimental qpOASES LDLT fast path (default: off)\n"
      << "  --retarget-execution M  sync|async (default: sync)\n"
      << "  --async-retarget        alias for --retarget-execution async\n"
      << "  --h2-continuity M       none|root|blend (default: blend)\n"
      << "  --h2-blend-frames N     H2 overlap blend width (default: 8)\n"
      << "  --legacy-g1-blend-indexing  reproduce current deploy tail-freeze bug\n"
      << "  --h2-root-z-smooth-radius N  safe Gaussian radius (default: 3)\n"
      << "  --no-h2-root-z-smoothing     disable H2 pelvis-z smoothing\n"
      << "  --no-ground-align       disable retarget keypoint ground alignment\n"
      << "  --retargeter-root PATH  robot_retargeter root override\n"
      << "  --g1-mjcf PATH          G1 visualization model\n"
      << "  --h2-mjcf PATH          H2 visualization scene/model\n"
      << "  --output DIR            output directory (default: /tmp/g1_planner_to_h2_*)\n"
      << "  --fp16                  TensorRT planner FP16\n"
      << "  --headless              generate/export/measure without opening viewer\n"
      << "  --paused                open viewer paused\n"
      << "  --show-solve-stalls     add recorded sync IK stalls to viewer timing\n"
      << "  --width PX --height-px PX\n"
      << "  --help\n\n"
      << "Viewer: Space play/pause, Left/Right step, R reset, Up/Down speed,\n"
      << "        A/D orbit, W/S elevation, Q/E zoom, F toggle root-follow, Esc quit.\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(key + " requires a value");
      return argv[++i];
    };
    if (key == "--planner") args.planner = need();
    else if (key == "--mode") args.mode = ParseMode(need());
    else if (key == "--speed") args.speed = std::stof(need());
    else if (key == "--height") args.height = std::stof(need());
    else if (key == "--move") args.movement = ParseVec3(need());
    else if (key == "--face") args.facing = ParseVec3(need());
    else if (key == "--seed") args.seed = std::stoi(need());
    else if (key == "--replans") args.replans = std::stoi(need());
    else if (key == "--replan-frames") args.replan_frames = std::stoi(need());
    else if (key == "--random-locomotion") args.random_locomotion = true;
    else if (key == "--live-random") {
      args.live_random = true;
      args.random_locomotion = true;
    }
    else if (key == "--random-clips") args.random_clips = std::stoi(need());
    else if (key == "--random-duration-min") args.random_duration_min_s = std::stod(need());
    else if (key == "--random-duration-max") args.random_duration_max_s = std::stod(need());
    else if (key == "--random-gap") args.random_gap_s = std::stod(need());
    else if (key == "--random-speed-max") args.random_speed_max = std::stof(need());
    else if (key == "--random-core-only") args.random_core_only = true;
    else if (key == "--root-z-offset") args.root_z_offset = std::stod(need());
    else if (key == "--retarget-max-iter") args.retarget_max_iter = std::stoi(need());
    else if (key == "--retarget-solver") {
      args.retarget_solver = robot_retargeter::ParseQpSolver(need());
    }
    else if (key == "--retarget-use-ldlt") {
      args.retarget_use_ldlt_fast_path = true;
    }
    else if (key == "--retarget-execution") {
      const std::string mode = Lower(need());
      if (mode == "sync") args.retarget_execution = RetargetExecution::kSync;
      else if (mode == "async") args.retarget_execution = RetargetExecution::kAsync;
      else throw std::runtime_error("--retarget-execution must be sync or async");
    }
    else if (key == "--async-retarget") args.retarget_execution = RetargetExecution::kAsync;
    else if (key == "--h2-continuity") {
      const std::string mode = Lower(need());
      if (mode == "none") args.h2_continuity = H2ContinuityMode::kNone;
      else if (mode == "root") args.h2_continuity = H2ContinuityMode::kRoot;
      else if (mode == "blend") args.h2_continuity = H2ContinuityMode::kBlend;
      else throw std::runtime_error("--h2-continuity must be none, root, or blend");
    }
    else if (key == "--h2-blend-frames") args.h2_blend_frames = std::stoi(need());
    else if (key == "--legacy-g1-blend-indexing") args.independent_g1_timeline = false;
    else if (key == "--h2-root-z-smooth-radius") args.h2_root_z_smooth_radius = std::stoi(need());
    else if (key == "--no-h2-root-z-smoothing") args.h2_root_z_smooth_radius = 0;
    else if (key == "--no-ground-align") args.ground_align = false;
    else if (key == "--retargeter-root") args.retargeter_root = need();
    else if (key == "--g1-mjcf") args.g1_mjcf = need();
    else if (key == "--h2-mjcf") args.h2_mjcf = need();
    else if (key == "--output") args.output = need();
    else if (key == "--width") args.width = std::stoi(need());
    else if (key == "--height-px") args.height_px = std::stoi(need());
    else if (key == "--fp16") args.fp16 = true;
    else if (key == "--headless") args.headless = true;
    else if (key == "--paused") args.autoplay = false;
    else if (key == "--show-solve-stalls") args.show_solve_stalls = true;
    else if (key == "--help" || key == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + key);
    }
  }
  if (args.replans < 1 || args.replans > 200) {
    throw std::runtime_error("--replans must be in [1, 200]");
  }
  if (args.replan_frames == -1) {
    // Random turns update heading continuously; 10 frames gives a 5 Hz command
    // schedule without producing the recorder's much heavier 20 Hz solve load.
    args.replan_frames = args.random_locomotion ? 10
                                                 : DefaultReplanFrames(args.mode);
  }
  if (args.replan_frames < 1) {
    throw std::runtime_error("--replan-frames must be positive");
  }
  if (args.retarget_max_iter < 1 || args.retarget_max_iter > 100) {
    throw std::runtime_error("--retarget-max-iter must be in [1, 100]");
  }
  if (args.h2_blend_frames < 1 || args.h2_blend_frames > 50) {
    throw std::runtime_error("--h2-blend-frames must be in [1, 50]");
  }
  if (args.h2_root_z_smooth_radius < 0 || args.h2_root_z_smooth_radius > 20) {
    throw std::runtime_error("--h2-root-z-smooth-radius must be in [0, 20]");
  }
  if (args.random_clips < 1 || args.random_clips > 100) {
    throw std::runtime_error("--random-clips must be in [1, 100]");
  }
  if (args.random_duration_min_s <= 0.0 ||
      args.random_duration_max_s < args.random_duration_min_s ||
      args.random_duration_max_s > 120.0) {
    throw std::runtime_error(
        "random durations require 0 < min <= max <= 120 seconds");
  }
  if (args.random_gap_s < 0.0 || args.random_gap_s > 30.0) {
    throw std::runtime_error("--random-gap must be in [0, 30] seconds");
  }
  if (args.random_speed_max <= 0.0f || args.random_speed_max > 5.0f) {
    throw std::runtime_error("--random-speed-max must be in (0, 5] m/s");
  }
  if (args.live_random && args.headless) {
    throw std::runtime_error("--live-random requires a viewer (remove --headless)");
  }
  if (args.width < 640 || args.height_px < 360) {
    throw std::runtime_error("viewer size is too small");
  }
  if (args.output.empty()) {
    args.output = fs::temp_directory_path() / ("g1_planner_to_h2_" + Timestamp());
  }
  return args;
}

void ValidateInputs(const Args& args) {
  for (const auto& path : {args.planner, args.g1_mjcf, args.h2_mjcf}) {
    if (!fs::is_regular_file(path)) {
      throw std::runtime_error("missing input file: " + path.string());
    }
  }
}

std::shared_ptr<MotionSequence> NewMotion(int capacity, const std::string& name) {
  auto motion = std::make_shared<MotionSequence>();
  motion->ReserveCapacity(capacity, 29, 1, 1, 0, 0);
  motion->timesteps = 0;
  motion->name = name;
  return motion;
}

void AppendFrames(const MotionSequence& src, int begin, int count,
                  MotionSequence& dst) {
  begin = std::clamp(begin, 0, src.timesteps);
  count = std::clamp(count, 0, src.timesteps - begin);
  for (int i = 0; i < count; ++i) {
    CopyMotionSequenceFrame(src, begin + i, dst, dst.timesteps + i);
  }
  dst.timesteps += count;
}

struct PlannerCommand {
  int generation = 0;
  int clip = 0;
  int step_in_clip = 0;
  int steps_in_clip = 1;
  int mode = static_cast<int>(LocomotionMode::IDLE);
  float speed = -1.0f;
  float height = -1.0f;
  std::array<float, 3> movement{0.0f, 0.0f, 0.0f};
  std::array<float, 3> facing{1.0f, 0.0f, 0.0f};
  std::string style = "idle";
  std::string maneuver = "fixed";
  double face_yaw = 0.0;
  double spin_rad_s = 0.0;
  double body_move_angle = 0.0;
  double clip_duration_s = 0.0;
  bool idle_gap = false;
};

struct RandomStyle {
  int mode;
  const char* name;
  double speed_min;
  double speed_max;
  double weight;
  bool core;
};

struct RandomManeuver {
  const char* name;
  double weight;
};

constexpr std::array<RandomStyle, 12> kRandomStyles{{
    {1, "slowWalk", 0.15, 0.75, 3.0, true},
    {2, "walk", 0.80, 2.20, 4.0, true},
    {3, "run", 2.50, 4.50, 3.0, true},
    {10, "walkBoxing", 0.50, 1.50, 1.0, false},
    // zmq_record_g1_locomotion.py currently labels mode 17 as happy; the
    // planner enum defines 17=forward_jump and 23=happy_dance_walk.
    {23, "happy", 0.50, 1.80, 1.0, false},
    {18, "stealth", 0.40, 1.20, 1.0, false},
    {19, "injured", 0.30, 1.00, 1.0, false},
    {20, "ledge", 0.40, 1.20, 1.0, false},
    {21, "objectCarrying", 0.50, 1.40, 1.0, false},
    {22, "stealth2", 0.30, 1.00, 1.0, false},
    {24, "zombie", 0.30, 1.00, 1.0, false},
    {26, "scared", 0.40, 1.30, 1.0, false},
}};

constexpr std::array<RandomManeuver, 5> kRandomManeuvers{{
    {"move", 8.5},
    {"turn_ccw", 1.2},
    {"turn_cw", 1.2},
    {"arc_ccw", 1.0},
    {"arc_cw", 1.0},
}};

double Uniform(std::mt19937& rng, double low, double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return distribution(rng);
}

double WrapYaw(double yaw) {
  constexpr double kTwoPi = 2.0 * M_PI;
  yaw -= kTwoPi * std::floor(yaw / kTwoPi);
  return yaw;
}

std::array<float, 3> YawDirection(double yaw) {
  return {static_cast<float>(std::cos(yaw)),
          static_cast<float>(std::sin(yaw)), 0.0f};
}

std::array<float, 3> BodyMoveDirection(
    const std::array<float, 3>& facing, double body_angle) {
  const double c = std::cos(body_angle);
  const double s = std::sin(body_angle);
  return {static_cast<float>(c * facing[0] - s * facing[1]),
          static_cast<float>(s * facing[0] + c * facing[1]), 0.0f};
}

std::string ModeName(int mode) {
  static const std::array<const char*, 27> names{{
      "idle", "slowWalk", "walk", "run", "squat", "kneelTwo",
      "kneel", "lying", "crawl", "boxing", "walkBoxing",
      "leftPunch", "rightPunch", "randomPunch", "elbowCrawl",
      "leftHook", "rightHook", "forwardJump", "stealth", "injured",
      "ledge", "objectCarrying", "stealth2", "happy", "zombie",
      "gun", "scared",
  }};
  if (mode < 0 || mode >= static_cast<int>(names.size())) return "unknown";
  return names[mode];
}

std::vector<PlannerCommand> BuildFixedCommands(const Args& args) {
  std::vector<PlannerCommand> commands;
  commands.reserve(args.replans);
  for (int generation = 0; generation < args.replans; ++generation) {
    PlannerCommand command;
    command.generation = generation;
    command.step_in_clip = generation;
    command.steps_in_clip = args.replans;
    command.mode = args.mode;
    command.speed = args.speed;
    command.height = args.height;
    command.movement = args.movement;
    command.facing = args.facing;
    command.style = ModeName(args.mode);
    command.face_yaw = std::atan2(args.facing[1], args.facing[0]);
    commands.push_back(command);
  }
  return commands;
}

std::vector<PlannerCommand> BuildRandomCommands(const Args& args) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(args.seed));
  std::vector<const RandomStyle*> styles;
  std::vector<double> style_weights;
  for (const auto& style : kRandomStyles) {
    if (args.random_core_only && !style.core) continue;
    styles.push_back(&style);
    style_weights.push_back(style.weight);
  }
  std::discrete_distribution<size_t> choose_style(style_weights.begin(),
                                                   style_weights.end());
  std::array<double, kRandomManeuvers.size()> maneuver_weights{};
  for (size_t i = 0; i < kRandomManeuvers.size(); ++i) {
    maneuver_weights[i] = kRandomManeuvers[i].weight;
  }
  std::discrete_distribution<size_t> choose_maneuver(
      maneuver_weights.begin(), maneuver_weights.end());

  std::vector<PlannerCommand> commands;
  double heading_yaw = std::atan2(args.facing[1], args.facing[0]);
  const double command_period_s = args.replan_frames / kPlaybackFps;

  for (int clip = 1; clip <= args.random_clips; ++clip) {
    const RandomStyle& style = *styles[choose_style(rng)];
    const RandomManeuver& maneuver = kRandomManeuvers[choose_maneuver(rng)];
    const double duration_s = Uniform(rng, args.random_duration_min_s,
                                      args.random_duration_max_s);
    const int steps = std::max(1, static_cast<int>(
        std::ceil(duration_s / command_period_s)));
    const double scheduled_duration_s = steps * command_period_s;
    if (commands.size() + static_cast<size_t>(steps) >
        kMaxRandomPlannerUpdates) {
      throw std::runtime_error(
          "random schedule exceeds 2000 planner updates; reduce clips/duration "
          "or increase --replan-frames");
    }
    const double speed_low = std::min<double>(style.speed_min,
                                              args.random_speed_max);
    const double speed_high = std::min<double>(style.speed_max,
                                               args.random_speed_max);
    const bool use_default_speed =
        style.speed_max <= args.random_speed_max && Uniform(rng, 0.0, 1.0) < 0.4;
    const float speed = use_default_speed
                            ? -1.0f
                            : static_cast<float>(Uniform(
                                  rng, std::min(speed_low, speed_high),
                                  std::max(speed_low, speed_high)));
    double spin = 0.0;
    if (std::string(maneuver.name) == "turn_ccw" ||
        std::string(maneuver.name) == "arc_ccw") {
      spin = Uniform(rng, 0.35, 0.80);
    } else if (std::string(maneuver.name) == "turn_cw" ||
               std::string(maneuver.name) == "arc_cw") {
      spin = -Uniform(rng, 0.35, 0.80);
    }
    const bool pure_turn = std::string(maneuver.name) == "turn_ccw" ||
                           std::string(maneuver.name) == "turn_cw";
    const double body_angle = pure_turn ? 0.0 : Uniform(rng, 0.0, 2.0 * M_PI);
    const double yaw0 = WrapYaw(heading_yaw);

    for (int step = 0; step < steps; ++step) {
      PlannerCommand command;
      command.generation = static_cast<int>(commands.size());
      command.clip = clip;
      command.step_in_clip = step;
      command.steps_in_clip = steps;
      command.mode = style.mode;
      command.speed = speed;
      command.height = args.height;
      command.style = style.name;
      command.maneuver = maneuver.name;
      command.spin_rad_s = spin;
      command.body_move_angle = body_angle;
      command.clip_duration_s = scheduled_duration_s;
      command.face_yaw = WrapYaw(yaw0 + spin * step * command_period_s);
      command.facing = YawDirection(command.face_yaw);
      command.movement = pure_turn
                             ? std::array<float, 3>{0.0f, 0.0f, 0.0f}
                             : BodyMoveDirection(command.facing, body_angle);
      commands.push_back(std::move(command));
    }
    heading_yaw = WrapYaw(yaw0 + spin * scheduled_duration_s);

    if (clip < args.random_clips && args.random_gap_s > 0.0) {
      const int gap_steps = std::max(1, static_cast<int>(
          std::ceil(args.random_gap_s / command_period_s)));
      if (commands.size() + static_cast<size_t>(gap_steps) >
          kMaxRandomPlannerUpdates) {
        throw std::runtime_error(
            "random schedule exceeds 2000 planner updates; reduce random gap "
            "or increase --replan-frames");
      }
      for (int step = 0; step < gap_steps; ++step) {
        PlannerCommand command;
        command.generation = static_cast<int>(commands.size());
        command.clip = clip;
        command.step_in_clip = step;
        command.steps_in_clip = gap_steps;
        command.mode = static_cast<int>(LocomotionMode::IDLE);
        command.height = args.height;
        command.facing = YawDirection(heading_yaw);
        command.style = "idle";
        command.maneuver = "gap";
        command.face_yaw = heading_yaw;
        command.clip_duration_s = args.random_gap_s;
        command.idle_gap = true;
        commands.push_back(std::move(command));
      }
    }
  }
  if (commands.empty()) throw std::runtime_error("empty random command schedule");
  return commands;
}

std::vector<PlannerCommand> BuildPlannerCommands(const Args& args) {
  return args.random_locomotion ? BuildRandomCommands(args)
                                : BuildFixedCommands(args);
}

class LiveRandomCommandStream {
 public:
  explicit LiveRandomCommandStream(const Args& args)
      : args_(args), next_heading_yaw_(
                         std::atan2(args.facing[1], args.facing[0])) {}

  PlannerCommand Next(int generation) {
    if (batch_index_ >= batch_.size()) {
      if (gap_step_ < gap_steps_) return NextGap(generation);
      Refill();
    }
    PlannerCommand command = batch_[batch_index_++];
    command.generation = generation;
    command.clip = clip_id_;
    if (batch_index_ == batch_.size()) {
      const double command_period_s = args_.replan_frames / kPlaybackFps;
      next_heading_yaw_ = WrapYaw(
          command.face_yaw + command.spin_rad_s * command_period_s);
      gap_steps_ = args_.random_gap_s > 0.0
                       ? std::max(1, static_cast<int>(std::ceil(
                                         args_.random_gap_s / command_period_s)))
                       : 0;
      gap_step_ = 0;
    }
    return command;
  }

 private:
  PlannerCommand NextGap(int generation) {
    PlannerCommand command;
    command.generation = generation;
    command.clip = clip_id_;
    command.step_in_clip = gap_step_;
    command.steps_in_clip = gap_steps_;
    command.mode = static_cast<int>(LocomotionMode::IDLE);
    command.height = args_.height;
    command.facing = YawDirection(next_heading_yaw_);
    command.style = "idle";
    command.maneuver = "gap";
    command.face_yaw = next_heading_yaw_;
    command.clip_duration_s = args_.random_gap_s;
    command.idle_gap = true;
    ++gap_step_;
    return command;
  }

  void Refill() {
    Args batch_args = args_;
    batch_args.random_clips = 1;
    batch_args.random_gap_s = 0.0;
    batch_args.seed = args_.seed + 7919 * batch_number_;
    batch_args.facing = YawDirection(next_heading_yaw_);
    batch_ = BuildRandomCommands(batch_args);
    batch_index_ = 0;
    gap_steps_ = 0;
    gap_step_ = 0;
    ++clip_id_;
    ++batch_number_;
  }

  const Args& args_;
  std::vector<PlannerCommand> batch_;
  size_t batch_index_ = 0;
  int batch_number_ = 0;
  int clip_id_ = 0;
  int gap_steps_ = 0;
  int gap_step_ = 0;
  double next_heading_yaw_ = 0.0;
};

struct TrajectoryPair {
  std::shared_ptr<MotionSequence> g1;
  std::shared_ptr<MotionSequence> h2;
  RetargetExecution execution = RetargetExecution::kSync;
  H2ContinuityMode continuity = H2ContinuityMode::kNone;
  std::vector<int> replan_boundaries;
  std::vector<int> command_boundaries;
  std::vector<int> publish_boundaries;
  std::vector<int> applied_generation;
  std::vector<int> applied_source_frame;
  std::vector<double> frame_hold_ms;
  std::vector<double> planner_ms;
  std::vector<double> retarget_ms;
  std::vector<PlannerCommand> commands;
  double max_control_tick_ms = 0.0;
  int control_deadline_misses = 0;
  int dropped_generations = 0;
};

struct PlannedWindow {
  std::shared_ptr<MotionSequence> g1;
  double planner_ms = 0.0;
  PlannerCommand command;
};

struct RetargetEvent {
  int generation = 0;
  int scheduled_frame = 0;
  int publish_frame = -1;
  int aligned_source_frame = -1;
  double planner_ms = 0.0;
  double solve_ms = 0.0;
  double control_block_ms = 0.0;
  int missed_control_ticks = 0;
  double continuity_root_correction_m = 0.0;
  double continuity_planar_correction_m = 0.0;
  double continuity_vertical_correction_m = 0.0;
  double continuity_yaw_correction_rad = 0.0;
  double joint_anchor_rmse_before_rad = 0.0;
  double joint_anchor_rmse_after_rad = 0.0;
  int continuity_blended_frames = 0;
  int vertical_recovery_frames = 0;
  std::string status = "pending";
};

struct GeneratedTrajectories {
  TrajectoryPair trajectories;
  std::vector<RetargetEvent> events;
};

std::vector<PlannedWindow> GeneratePlannerWindows(const Args& args) {
  const std::vector<PlannerCommand> commands = BuildPlannerCommands(args);
  PlannerConfig config;
  config.model_path = args.planner.string();
  config.version = 2;
  config.default_height = G1_PLANNER_DEFAULT_HEIGHT;
  config.initial_random_seed = args.seed;

  std::cout << "[planner-test] Loading TensorRT G1 planner: " << args.planner << "\n";
  LocalMotionPlannerTensorRT planner(args.fp16, 0, config);
  planner.planner_state_.enabled = true;
  const std::array<double, 4> identity_quat{1.0, 0.0, 0.0, 0.0};
  if (!planner.Initialize(identity_quat, G1_PLANNER_DEFAULT_JOINTS_MUJOCO)) {
    throw std::runtime_error("planner initialization failed");
  }

  auto g1_current = NewMotion(kPlannerCapacity, "planner_g1_current");

  {
    std::lock_guard<std::mutex> lock(planner.planner_motion_mutex_);
    const auto result = ApplyGeneratedPlannerMotion(
        planner.planner_motion_50hz_, planner.gen_frame_, 0, *g1_current, 8,
        args.independent_g1_timeline);
    planner.motion_available_ = false;
    if (!result.success) throw std::runtime_error("failed to accept initial planner window");
  }
  std::vector<PlannedWindow> windows;
  windows.reserve(commands.size());
  std::cout << "[planner-test] command schedule: " << commands.size()
            << " planner updates @ " << args.replan_frames << " frames ("
            << args.replan_frames / kPlaybackFps << "s), mode="
            << (args.random_locomotion ? "random" : "fixed") << "\n";

  for (size_t replan = 0; replan < commands.size(); ++replan) {
    const PlannerCommand& command = commands[replan];
    const int current_frame = (replan == 0)
                                  ? 0
                                  : std::min(args.replan_frames,
                                             g1_current->timesteps - 1);
    const auto planning_start = std::chrono::steady_clock::now();
    if (!planner.UpdatePlanning(current_frame, g1_current, command.mode,
                                command.speed, command.height,
                                command.movement, command.facing,
                                args.seed + static_cast<int>(replan))) {
      throw std::runtime_error("planner UpdatePlanning failed at replan " +
                               std::to_string(replan));
    }
    const auto planning_end = std::chrono::steady_clock::now();

    PlannerMotionBlendResult blend_result;
    {
      std::lock_guard<std::mutex> lock(planner.planner_motion_mutex_);
      blend_result = ApplyGeneratedPlannerMotion(
          planner.planner_motion_50hz_, planner.gen_frame_, current_frame,
          *g1_current, 8, args.independent_g1_timeline);
      planner.motion_available_ = false;
    }
    if (!blend_result.success) {
      throw std::runtime_error("planner blend failed at replan " +
                               std::to_string(replan));
    }

    const double planning_ms = std::chrono::duration<double, std::milli>(
                                   planning_end - planning_start)
                                   .count();
    auto snapshot = std::make_shared<MotionSequence>(*g1_current);
    snapshot->name = "g1_planner_window_" + std::to_string(replan);
    windows.push_back({std::move(snapshot), planning_ms, command});
    std::cout << "[planner-test] replan " << (replan + 1) << "/"
              << commands.size() << " clip=" << command.clip
              << " style=" << command.style
              << " maneuver=" << command.maneuver
              << " speed=" << command.speed
              << " yaw=" << command.face_yaw
              << ": frames=" << g1_current->timesteps
              << " planner=" << std::fixed << std::setprecision(2)
              << planning_ms << "ms\n";
  }
  return windows;
}

robot_retargeter::G1ToH2Options MakeRetargetOptions(const Args& args) {
  robot_retargeter::G1ToH2Options options;
  options.max_iter = args.retarget_max_iter;
  options.qp_solver = args.retarget_solver;
  options.qpoases_use_ldlt_fast_path = args.retarget_use_ldlt_fast_path;
  options.ground_align = args.ground_align;
  if (!args.retargeter_root.empty()) options.retargeter_root = args.retargeter_root;
  return options;
}

std::shared_ptr<MotionSequence> RetargetWindow(
    robot_retargeter::G1ToH2Retargeter& retargeter,
    const MotionSequence& g1, const Args& args, double& solve_ms) {
  auto h2 = NewMotion(kPlannerCapacity, "h2_retarget_window");
  const auto start = std::chrono::steady_clock::now();
  if (!RetargetG1MotionSequenceToH2(retargeter, g1, *h2, g1.timesteps,
                                    args.root_z_offset)) {
    throw std::runtime_error("G1->H2 retarget failed");
  }
  const H2RootZSmoothingResult z_smoothing =
      SmoothH2RetargetRootZ(*h2, args.h2_root_z_smooth_radius);
  if (z_smoothing.applied) {
    std::cout << "[planner-test] H2 root-z smoothing radius="
              << args.h2_root_z_smooth_radius << " safety_lift="
              << z_smoothing.safety_lift_m * 1000.0 << "mm\n";
  }
  solve_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start)
                 .count();
  return h2;
}

void BuildDesiredG1(const Args& args, const std::vector<PlannedWindow>& windows,
                    TrajectoryPair& output) {
  const int capacity = static_cast<int>(windows.size()) * args.replan_frames +
                       kPlannerCapacity;
  output.g1 = NewMotion(capacity, "g1_planner_original");
  for (size_t generation = 0; generation < windows.size(); ++generation) {
    if (generation > 0 &&
        (windows[generation].command.clip !=
             windows[generation - 1].command.clip ||
         windows[generation].command.idle_gap !=
             windows[generation - 1].command.idle_gap)) {
      output.command_boundaries.push_back(output.g1->timesteps);
    }
    const bool last_generation = generation + 1 == windows.size();
    const bool append_prediction_tail =
        last_generation && !args.random_locomotion;
    const int count = append_prediction_tail
                          ? windows[generation].g1->timesteps
                          : std::min(args.replan_frames,
                                     windows[generation].g1->timesteps - 1);
    AppendFrames(*windows[generation].g1, 0, count, *output.g1);
    if (!last_generation) {
      output.replan_boundaries.push_back(output.g1->timesteps);
    }
    output.planner_ms.push_back(windows[generation].planner_ms);
    output.commands.push_back(windows[generation].command);
  }
  output.frame_hold_ms.assign(output.g1->timesteps, 0.0);
}

int MissedControlTicks(double block_ms) {
  return std::max(0, static_cast<int>(std::ceil(block_ms / kControlPeriodMs)) - 1);
}

int SegmentForFrame(int frame, const std::vector<int>& boundaries);

void ApplyConfiguredContinuity(const Args& args,
                               const MotionSequence& previous,
                               int previous_frame,
                               MotionSequence& next,
                               int next_frame,
                               RetargetEvent& event) {
  if (args.h2_continuity == H2ContinuityMode::kNone) return;
  const H2RetargetContinuityResult result =
      ApplyH2RetargetWindowContinuity(
          previous, previous_frame, next, next_frame, args.h2_blend_frames,
          args.h2_continuity == H2ContinuityMode::kBlend);
  if (!result.applied) return;
  event.continuity_root_correction_m =
      result.root_translation_correction_m;
  event.continuity_planar_correction_m = result.root_planar_correction_m;
  event.continuity_vertical_correction_m =
      result.root_vertical_correction_m;
  event.continuity_yaw_correction_rad = result.root_yaw_correction_rad;
  event.joint_anchor_rmse_before_rad =
      result.joint_anchor_rmse_before_rad;
  event.joint_anchor_rmse_after_rad =
      result.joint_anchor_rmse_after_rad;
  event.continuity_blended_frames = result.blended_frames;
  event.vertical_recovery_frames = result.vertical_recovery_frames;
}

GeneratedTrajectories GenerateSyncTrajectories(
    const Args& args, const std::vector<PlannedWindow>& windows,
    robot_retargeter::G1ToH2Retargeter& retargeter) {
  GeneratedTrajectories generated;
  auto& output = generated.trajectories;
  output.execution = RetargetExecution::kSync;
  output.continuity = args.h2_continuity;
  BuildDesiredG1(args, windows, output);
  output.h2 = NewMotion(output.g1->timesteps + 1, "h2_planner_retarget_sync");
  std::shared_ptr<MotionSequence> previous_h2;

  for (size_t generation = 0; generation < windows.size(); ++generation) {
    double solve_ms = 0.0;
    auto h2 = RetargetWindow(retargeter, *windows[generation].g1, args, solve_ms);
    RetargetEvent event;
    if (previous_h2) {
      const int previous_frame =
          std::min(args.replan_frames, previous_h2->timesteps - 1);
      ApplyConfiguredContinuity(args, *previous_h2, previous_frame, *h2, 0,
                                event);
    }
    const bool append_prediction_tail =
        generation + 1 == windows.size() && !args.random_locomotion;
    const int count = append_prediction_tail
                          ? h2->timesteps
                          : std::min(args.replan_frames, h2->timesteps - 1);
    const int scheduled_frame = output.h2->timesteps;
    AppendFrames(*h2, 0, count, *output.h2);
    output.retarget_ms.push_back(solve_ms);
    output.publish_boundaries.push_back(scheduled_frame);
    output.max_control_tick_ms = std::max(output.max_control_tick_ms, solve_ms);

    event.generation = static_cast<int>(generation);
    event.scheduled_frame = scheduled_frame;
    event.publish_frame = scheduled_frame;
    event.aligned_source_frame = 0;
    event.planner_ms = windows[generation].planner_ms;
    event.solve_ms = solve_ms;
    event.control_block_ms = solve_ms;
    event.missed_control_ticks = MissedControlTicks(solve_ms);
    event.status = generation == 0 ? "startup_sync" : "published_sync";
    output.control_deadline_misses += event.missed_control_ticks;
    generated.events.push_back(event);

    if (generation > 0 && scheduled_frame > 0) {
      output.frame_hold_ms[scheduled_frame - 1] = solve_ms;
    }
    std::cout << "[planner-test] sync retarget generation " << generation
              << ": " << h2->timesteps << " frames in " << solve_ms
              << "ms, missed 50Hz ticks=" << event.missed_control_ticks;
    if (event.continuity_root_correction_m > 0.0) {
      std::cout << ", planar anchor="
                << event.continuity_planar_correction_m * 100.0
                << "cm, transient z="
                << event.continuity_vertical_correction_m * 1000.0
                << "mm, joint anchor " << event.joint_anchor_rmse_before_rad
                << "->" << event.joint_anchor_rmse_after_rad << "rad";
    }
    std::cout << "\n";
    previous_h2 = std::move(h2);
  }
  output.applied_generation.resize(output.h2->timesteps, 0);
  output.applied_source_frame.resize(output.h2->timesteps, 0);
  for (int frame = 0; frame < output.h2->timesteps; ++frame) {
    const int generation = SegmentForFrame(frame, output.replan_boundaries);
    const int segment_start = generation == 0
                                  ? 0
                                  : output.replan_boundaries[generation - 1];
    output.applied_generation[frame] = generation;
    output.applied_source_frame[frame] = frame - segment_start;
  }
  return generated;
}

struct AsyncRetargetResult {
  int generation = -1;
  int scheduled_frame = -1;
  std::shared_ptr<MotionSequence> h2;
  double solve_ms = 0.0;
};

GeneratedTrajectories GenerateAsyncTrajectories(
    const Args& args, const std::vector<PlannedWindow>& windows,
    robot_retargeter::G1ToH2Retargeter& retargeter) {
  GeneratedTrajectories generated;
  auto& output = generated.trajectories;
  output.execution = RetargetExecution::kAsync;
  output.continuity = args.h2_continuity;
  BuildDesiredG1(args, windows, output);
  output.h2 = NewMotion(output.g1->timesteps + 1, "h2_planner_retarget_async");
  output.applied_generation.reserve(output.g1->timesteps);
  output.applied_source_frame.reserve(output.g1->timesteps);
  generated.events.resize(windows.size());
  output.retarget_ms.assign(windows.size(), 0.0);

  double startup_ms = 0.0;
  auto active_h2 = RetargetWindow(retargeter, *windows[0].g1, args, startup_ms);
  int active_generation = 0;
  int active_scheduled_frame = 0;
  generated.events[0].generation = 0;
  generated.events[0].scheduled_frame = 0;
  generated.events[0].publish_frame = 0;
  generated.events[0].aligned_source_frame = 0;
  generated.events[0].planner_ms = windows[0].planner_ms;
  generated.events[0].solve_ms = startup_ms;
  generated.events[0].status = "startup_before_playback";
  output.retarget_ms[0] = startup_ms;
  output.publish_boundaries.push_back(0);

  std::future<AsyncRetargetResult> pending;
  const auto control_period = std::chrono::duration<double, std::milli>(
      kControlPeriodMs);
  auto next_tick = std::chrono::steady_clock::now();
  size_t next_boundary = 0;

  for (int frame = 0; frame < output.g1->timesteps; ++frame) {
    const auto tick_start = std::chrono::steady_clock::now();

    if (pending.valid() &&
        pending.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      AsyncRetargetResult ready = pending.get();
      RetargetEvent& event = generated.events[ready.generation];
      event.solve_ms = ready.solve_ms;
      output.retarget_ms[ready.generation] = ready.solve_ms;
      const int aligned = frame - ready.scheduled_frame;
      if (aligned >= 0 && ready.h2 && aligned < ready.h2->timesteps) {
        const int previous_source_frame = std::clamp(
            frame - active_scheduled_frame, 0, active_h2->timesteps - 1);
        ApplyConfiguredContinuity(args, *active_h2, previous_source_frame,
                                  *ready.h2, aligned, event);
        active_h2 = std::move(ready.h2);
        active_generation = ready.generation;
        active_scheduled_frame = ready.scheduled_frame;
        event.publish_frame = frame;
        event.aligned_source_frame = aligned;
        event.status = "published_async";
        output.publish_boundaries.push_back(frame);
        std::cout << "[planner-test] async publish generation "
                  << ready.generation << " @ frame " << frame
                  << " (lag=" << aligned << ", planar anchor="
                  << event.continuity_planar_correction_m * 100.0
                  << "cm, transient z="
                  << event.continuity_vertical_correction_m * 1000.0
                  << "mm, joint anchor "
                  << event.joint_anchor_rmse_before_rad << "->"
                  << event.joint_anchor_rmse_after_rad << "rad)\n";
      } else {
        event.status = "expired_before_publish";
      }
    }

    if (next_boundary < output.replan_boundaries.size() &&
        frame == output.replan_boundaries[next_boundary]) {
      const int generation = static_cast<int>(next_boundary + 1);
      RetargetEvent& event = generated.events[generation];
      event.generation = generation;
      event.scheduled_frame = frame;
      event.planner_ms = windows[generation].planner_ms;
      if (pending.valid()) {
        event.status = "dropped_worker_busy";
        ++output.dropped_generations;
      } else {
        event.status = "solving_async";
        const auto source = windows[generation].g1;
        pending = std::async(std::launch::async,
            [&retargeter, source, &args, generation, frame]() {
              AsyncRetargetResult result;
              result.generation = generation;
              result.scheduled_frame = frame;
              result.h2 = RetargetWindow(retargeter, *source, args, result.solve_ms);
              return result;
            });
      }
      ++next_boundary;
    }

    const int source_frame = std::clamp(
        frame - active_scheduled_frame, 0, active_h2->timesteps - 1);
    CopyMotionSequenceFrame(*active_h2, source_frame, *output.h2,
                            output.h2->timesteps);
    ++output.h2->timesteps;
    output.applied_generation.push_back(active_generation);
    output.applied_source_frame.push_back(source_frame);

    const double tick_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - tick_start)
                               .count();
    output.max_control_tick_ms = std::max(output.max_control_tick_ms, tick_ms);
    if (tick_ms > kControlPeriodMs) ++output.control_deadline_misses;

    next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        control_period);
    std::this_thread::sleep_until(next_tick);
  }

  if (pending.valid()) {
    AsyncRetargetResult ready = pending.get();
    RetargetEvent& event = generated.events[ready.generation];
    event.solve_ms = ready.solve_ms;
    output.retarget_ms[ready.generation] = ready.solve_ms;
    if (event.status == "solving_async") event.status = "completed_after_playback";
  }
  return generated;
}

GeneratedTrajectories GenerateTrajectories(const Args& args) {
  const auto windows = GeneratePlannerWindows(args);
  if (windows.empty()) throw std::runtime_error("planner generated no windows");

  robot_retargeter::G1ToH2Retargeter retargeter(MakeRetargetOptions(args));
  std::cout << "[planner-test] Retarget G1: " << retargeter.options().g1_mjcf << "\n"
            << "[planner-test] Retarget H2: " << retargeter.options().h2_mjcf << "\n"
            << "[planner-test] max_iter=" << retargeter.options().max_iter
            << " qp_solver="
            << robot_retargeter::QpSolverName(retargeter.options().qp_solver)
            << " ground_align=" << retargeter.options().ground_align
            << " ground_clearance=" << retargeter.options().ground_clearance
            << " root_z_offset=" << args.root_z_offset
            << " execution=" << RetargetExecutionName(args.retarget_execution)
            << " h2_continuity=" << H2ContinuityModeName(args.h2_continuity)
            << " blend_frames=" << args.h2_blend_frames << "\n";

  GeneratedTrajectories generated =
      args.retarget_execution == RetargetExecution::kAsync
          ? GenerateAsyncTrajectories(args, windows, retargeter)
          : GenerateSyncTrajectories(args, windows, retargeter);
  auto& output = generated.trajectories;

  if (output.g1->timesteps != output.h2->timesteps ||
      output.g1->timesteps <= 0) {
    throw std::runtime_error("stitched trajectory length mismatch");
  }
  std::cout << "[planner-test] stitched " << output.g1->timesteps << " frames ("
            << output.g1->timesteps / kPlaybackFps << " s), boundaries="
            << output.replan_boundaries.size() << ", execution="
            << RetargetExecutionName(output.execution) << "\n";
  return generated;
}

struct LiveReplanRequest {
  int generation = 0;
  int64_t scheduled_frame = 0;
  int source_advance_frames = 0;
};

struct LiveWindow {
  int generation = 0;
  int64_t scheduled_frame = 0;
  PlannerCommand command;
  std::shared_ptr<MotionSequence> g1;
  std::shared_ptr<MotionSequence> h2;
  double planner_ms = 0.0;
  double solve_ms = 0.0;
};

class LivePlannerRetargetPipeline {
 public:
  explicit LivePlannerRetargetPipeline(const Args& args) : args_(args) {}
  ~LivePlannerRetargetPipeline() { Stop(); }

  void Start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_.joinable()) return;
    busy_ = true;
    worker_ = std::thread(&LivePlannerRetargetPipeline::WorkerMain, this);
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  bool Submit(const LiveReplanRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || busy_ || request_ready_ || !error_.empty()) return false;
    request_ = request;
    request_ready_ = true;
    busy_ = true;
    condition_.notify_one();
    return true;
  }

  std::shared_ptr<LiveWindow> TakePublished() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!published_) return {};
    auto result = std::move(published_);
    published_.reset();
    busy_ = false;
    return result;
  }

  bool Busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return busy_;
  }

  std::string Error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
  }

 private:
  void Publish(std::shared_ptr<LiveWindow> window) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stop_) published_ = std::move(window);
  }

  void WorkerMain() {
    try {
      PlannerConfig config;
      config.model_path = args_.planner.string();
      config.version = 2;
      config.default_height = G1_PLANNER_DEFAULT_HEIGHT;
      config.initial_random_seed = args_.seed;
      LocalMotionPlannerTensorRT planner(args_.fp16, 0, config);
      planner.planner_state_.enabled = true;
      const std::array<double, 4> identity_quat{1.0, 0.0, 0.0, 0.0};
      if (!planner.Initialize(identity_quat,
                              G1_PLANNER_DEFAULT_JOINTS_MUJOCO)) {
        throw std::runtime_error("live planner initialization failed");
      }

      auto g1_current = NewMotion(kPlannerCapacity, "live_g1_current");
      {
        std::lock_guard<std::mutex> lock(planner.planner_motion_mutex_);
        const auto accepted = ApplyGeneratedPlannerMotion(
            planner.planner_motion_50hz_, planner.gen_frame_, 0, *g1_current,
            8, args_.independent_g1_timeline);
        planner.motion_available_ = false;
        if (!accepted.success) {
          throw std::runtime_error("failed to accept live initial window");
        }
      }

      robot_retargeter::G1ToH2Retargeter retargeter(
          MakeRetargetOptions(args_));
      LiveRandomCommandStream commands(args_);

      auto generate = [&](const LiveReplanRequest& request) {
        const PlannerCommand command = commands.Next(request.generation);
        const int current_frame = std::clamp(
            request.source_advance_frames, 0, g1_current->timesteps - 1);
        const auto planner_start = std::chrono::steady_clock::now();
        if (!planner.UpdatePlanning(
                current_frame, g1_current, command.mode, command.speed,
                command.height, command.movement, command.facing,
                args_.seed + request.generation)) {
          throw std::runtime_error("live planner UpdatePlanning failed");
        }
        const double planner_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      planner_start)
                                      .count();
        {
          std::lock_guard<std::mutex> lock(planner.planner_motion_mutex_);
          const auto blended = ApplyGeneratedPlannerMotion(
              planner.planner_motion_50hz_, planner.gen_frame_, current_frame,
              *g1_current, 8, args_.independent_g1_timeline);
          planner.motion_available_ = false;
          if (!blended.success) {
            throw std::runtime_error("live planner blend failed");
          }
        }

        auto window = std::make_shared<LiveWindow>();
        window->generation = request.generation;
        window->scheduled_frame = request.scheduled_frame;
        window->command = command;
        window->g1 = std::make_shared<MotionSequence>(*g1_current);
        window->g1->name =
            "live_g1_window_" + std::to_string(request.generation);
        window->planner_ms = planner_ms;
        window->h2 = RetargetWindow(retargeter, *window->g1, args_,
                                    window->solve_ms);
        std::cout << "[planner-test/live] ready generation "
                  << request.generation << " clip=" << command.clip << " "
                  << command.style << "/" << command.maneuver
                  << " planner=" << planner_ms << "ms retarget="
                  << window->solve_ms << "ms\n";
        Publish(std::move(window));
      };

      generate({0, 0, 0});
      while (true) {
        LiveReplanRequest request;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock,
                          [&] { return stop_ || request_ready_; });
          if (stop_) break;
          request = request_;
          request_ready_ = false;
        }
        generate(request);
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = error.what();
      busy_ = false;
    }
  }

  const Args args_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::thread worker_;
  bool stop_ = false;
  bool busy_ = false;
  bool request_ready_ = false;
  LiveReplanRequest request_;
  std::shared_ptr<LiveWindow> published_;
  std::string error_;
};

struct MjModelDeleter {
  void operator()(mjModel* model) const { mj_deleteModel(model); }
};
struct MjDataDeleter {
  void operator()(mjData* data) const { mj_deleteData(data); }
};
using ModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;
using DataPtr = std::unique_ptr<mjData, MjDataDeleter>;

ModelPtr LoadModel(const fs::path& path) {
  char error[1024]{};
  ModelPtr model(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
  if (!model) {
    throw std::runtime_error("MuJoCo failed to load " + path.string() + ": " + error);
  }
  if (model->nq != 36) {
    throw std::runtime_error(path.string() + " has nq=" +
                             std::to_string(model->nq) + ", expected 36");
  }
  return model;
}

void SetFrame(const MotionSequence& motion, int frame, const mjModel* model,
              mjData* data) {
  std::array<double, 36> qpos{};
  MotionFrameToG1MujocoQpos(motion, frame, qpos);
  std::copy(qpos.begin(), qpos.end(), data->qpos);
  if (model->nv > 0) std::fill(data->qvel, data->qvel + model->nv, 0.0);
  mj_forward(model, data);
}

struct JointPair {
  std::string name;
  int g1_qadr = -1;
  int h2_qadr = -1;
};

struct BodyPair {
  std::string name;
  int g1_id = -1;
  int h2_id = -1;
};

std::vector<JointPair> CommonJoints(const mjModel* g1, const mjModel* h2) {
  std::vector<JointPair> pairs;
  for (int j = 0; j < g1->njnt; ++j) {
    if (g1->jnt_type[j] != mjJNT_HINGE) continue;
    const char* name = mj_id2name(g1, mjOBJ_JOINT, j);
    if (!name) continue;
    const int h2_j = mj_name2id(h2, mjOBJ_JOINT, name);
    if (h2_j < 0 || h2->jnt_type[h2_j] != mjJNT_HINGE) continue;
    pairs.push_back({name, g1->jnt_qposadr[j], h2->jnt_qposadr[h2_j]});
  }
  return pairs;
}

std::vector<BodyPair> CommonBodies(const mjModel* g1, const mjModel* h2) {
  std::vector<BodyPair> pairs;
  for (int b = 1; b < g1->nbody; ++b) {
    const char* name = mj_id2name(g1, mjOBJ_BODY, b);
    if (!name) continue;
    const int h2_b = mj_name2id(h2, mjOBJ_BODY, name);
    if (h2_b >= 0 && std::string(name) != "pelvis") {
      pairs.push_back({name, b, h2_b});
    }
  }
  return pairs;
}

std::array<double, 3> RootLocalPosition(const mjData* data, int root_id,
                                        int body_id) {
  std::array<double, 3> delta{};
  for (int i = 0; i < 3; ++i) {
    delta[i] = data->xpos[3 * body_id + i] - data->xpos[3 * root_id + i];
  }
  std::array<double, 3> local{};
  const mjtNum* rotation = data->xmat + 9 * root_id;
  for (int axis = 0; axis < 3; ++axis) {
    for (int world = 0; world < 3; ++world) {
      local[axis] += rotation[3 * world + axis] * delta[world];
    }
  }
  return local;
}

double Norm3(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

struct FrameMetrics {
  double root_xyz_delta = 0.0;
  double root_z_delta = 0.0;
  double root_angle_delta = 0.0;
  double joint_rmse = 0.0;
  double joint_max = 0.0;
  std::string joint_max_name;
  double body_rmse = 0.0;
  double left_foot = -1.0;
  double right_foot = -1.0;
  double left_hand = -1.0;
  double right_hand = -1.0;
};

FrameMetrics MeasureFrame(const mjModel* g1_model, const mjData* g1,
                          const mjModel* h2_model, const mjData* h2,
                          const std::vector<JointPair>& joints,
                          const std::vector<BodyPair>& bodies) {
  FrameMetrics metrics;
  std::array<double, 3> root_delta{};
  for (int i = 0; i < 3; ++i) root_delta[i] = h2->qpos[i] - g1->qpos[i];
  metrics.root_xyz_delta = Norm3(root_delta);
  metrics.root_z_delta = root_delta[2];

  double quat_dot = 0.0;
  for (int i = 0; i < 4; ++i) quat_dot += g1->qpos[3 + i] * h2->qpos[3 + i];
  quat_dot = std::clamp(std::abs(quat_dot), 0.0, 1.0);
  metrics.root_angle_delta = 2.0 * std::acos(quat_dot);

  double joint_squared = 0.0;
  for (const auto& pair : joints) {
    const double error = h2->qpos[pair.h2_qadr] - g1->qpos[pair.g1_qadr];
    joint_squared += error * error;
    if (std::abs(error) > metrics.joint_max) {
      metrics.joint_max = std::abs(error);
      metrics.joint_max_name = pair.name;
    }
  }
  if (!joints.empty()) {
    metrics.joint_rmse = std::sqrt(joint_squared / joints.size());
  }

  const int g1_root = mj_name2id(g1_model, mjOBJ_BODY, "pelvis");
  const int h2_root = mj_name2id(h2_model, mjOBJ_BODY, "pelvis");
  double body_squared = 0.0;
  for (const auto& pair : bodies) {
    const auto g1_local = RootLocalPosition(g1, g1_root, pair.g1_id);
    const auto h2_local = RootLocalPosition(h2, h2_root, pair.h2_id);
    std::array<double, 3> delta{};
    for (int i = 0; i < 3; ++i) delta[i] = h2_local[i] - g1_local[i];
    const double error = Norm3(delta);
    body_squared += error * error;
    if (pair.name == "left_ankle_pitch_link") metrics.left_foot = error;
    else if (pair.name == "right_ankle_pitch_link") metrics.right_foot = error;
    else if (pair.name == "left_wrist_yaw_link") metrics.left_hand = error;
    else if (pair.name == "right_wrist_yaw_link") metrics.right_hand = error;
  }
  if (!bodies.empty()) metrics.body_rmse = std::sqrt(body_squared / bodies.size());
  return metrics;
}

struct MeasuredTrajectory {
  std::vector<FrameMetrics> frames;
  std::vector<JointPair> joints;
  std::vector<BodyPair> bodies;
  double mean_joint_rmse = 0.0;
  double mean_body_rmse = 0.0;
  double mean_root_z_delta = 0.0;
};

MeasuredTrajectory MeasureTrajectory(const TrajectoryPair& trajectories,
                                     const mjModel* g1_model, mjData* g1_data,
                                     const mjModel* h2_model, mjData* h2_data) {
  MeasuredTrajectory measured;
  measured.joints = CommonJoints(g1_model, h2_model);
  measured.bodies = CommonBodies(g1_model, h2_model);
  measured.frames.reserve(trajectories.g1->timesteps);
  for (int frame = 0; frame < trajectories.g1->timesteps; ++frame) {
    SetFrame(*trajectories.g1, frame, g1_model, g1_data);
    SetFrame(*trajectories.h2, frame, h2_model, h2_data);
    measured.frames.push_back(MeasureFrame(
        g1_model, g1_data, h2_model, h2_data, measured.joints, measured.bodies));
  }
  for (const auto& frame : measured.frames) {
    measured.mean_joint_rmse += frame.joint_rmse;
    measured.mean_body_rmse += frame.body_rmse;
    measured.mean_root_z_delta += frame.root_z_delta;
  }
  const double n = std::max<size_t>(measured.frames.size(), 1);
  measured.mean_joint_rmse /= n;
  measured.mean_body_rmse /= n;
  measured.mean_root_z_delta /= n;
  return measured;
}

int SegmentForFrame(int frame, const std::vector<int>& boundaries) {
  return static_cast<int>(std::upper_bound(boundaries.begin(), boundaries.end(), frame) -
                          boundaries.begin());
}

struct BoundaryJump {
  double root_translation_m = 0.0;
  double root_velocity_delta_mps = 0.0;
  double joint_rmse_rad = 0.0;
  double joint_max_rad = 0.0;
  double joint_velocity_delta_rmse_rps = 0.0;
};

BoundaryJump MeasureBoundaryJump(const MotionSequence& motion, int boundary) {
  BoundaryJump jump;
  if (boundary <= 0 || boundary >= motion.timesteps) return jump;
  std::array<double, 3> root_delta{};
  for (int i = 0; i < 3; ++i) {
    root_delta[i] = motion.BodyPositions(boundary)[0][i] -
                    motion.BodyPositions(boundary - 1)[0][i];
  }
  jump.root_translation_m = Norm3(root_delta);
  double squared = 0.0;
  double velocity_delta_squared = 0.0;
  for (int joint = 0; joint < motion.GetNumJoints(); ++joint) {
    const double delta = motion.JointPositions(boundary)[joint] -
                         motion.JointPositions(boundary - 1)[joint];
    squared += delta * delta;
    jump.joint_max_rad = std::max(jump.joint_max_rad, std::abs(delta));
    if (boundary >= 2) {
      const double previous_delta =
          motion.JointPositions(boundary - 1)[joint] -
          motion.JointPositions(boundary - 2)[joint];
      const double velocity_delta =
          (delta - previous_delta) * kPlaybackFps;
      velocity_delta_squared += velocity_delta * velocity_delta;
    }
  }
  jump.joint_rmse_rad =
      std::sqrt(squared / std::max(motion.GetNumJoints(), 1));
  if (boundary >= 2) {
    std::array<double, 3> previous_root_delta{};
    std::array<double, 3> velocity_delta{};
    for (int axis = 0; axis < 3; ++axis) {
      previous_root_delta[axis] =
          motion.BodyPositions(boundary - 1)[0][axis] -
          motion.BodyPositions(boundary - 2)[0][axis];
      velocity_delta[axis] =
          (root_delta[axis] - previous_root_delta[axis]) * kPlaybackFps;
    }
    jump.root_velocity_delta_mps = Norm3(velocity_delta);
    jump.joint_velocity_delta_rmse_rps = std::sqrt(
        velocity_delta_squared / std::max(motion.GetNumJoints(), 1));
  }
  return jump;
}

struct MotionSmoothness {
  double max_root_step_m = 0.0;
  int max_root_step_frame = -1;
  double max_root_velocity_delta_mps = 0.0;
  int max_root_velocity_delta_frame = -1;
  double max_joint_step_rmse_rad = 0.0;
  int max_joint_step_frame = -1;
  double max_joint_velocity_delta_rmse_rps = 0.0;
  int max_joint_velocity_delta_frame = -1;
};

MotionSmoothness MeasureMotionSmoothness(const MotionSequence& motion) {
  MotionSmoothness smoothness;
  for (int frame = 1; frame < motion.timesteps; ++frame) {
    const BoundaryJump step = MeasureBoundaryJump(motion, frame);
    if (step.root_translation_m > smoothness.max_root_step_m) {
      smoothness.max_root_step_m = step.root_translation_m;
      smoothness.max_root_step_frame = frame;
    }
    if (step.joint_rmse_rad > smoothness.max_joint_step_rmse_rad) {
      smoothness.max_joint_step_rmse_rad = step.joint_rmse_rad;
      smoothness.max_joint_step_frame = frame;
    }
    if (step.root_velocity_delta_mps >
        smoothness.max_root_velocity_delta_mps) {
      smoothness.max_root_velocity_delta_mps =
          step.root_velocity_delta_mps;
      smoothness.max_root_velocity_delta_frame = frame;
    }
    if (step.joint_velocity_delta_rmse_rps >
        smoothness.max_joint_velocity_delta_rmse_rps) {
      smoothness.max_joint_velocity_delta_rmse_rps =
          step.joint_velocity_delta_rmse_rps;
      smoothness.max_joint_velocity_delta_frame = frame;
    }
  }
  return smoothness;
}

void ExportResults(const Args& args, const TrajectoryPair& trajectories,
                   const MeasuredTrajectory& measured,
                   const std::vector<RetargetEvent>& events) {
  fs::create_directories(args.output);
  const fs::path g1_dir = args.output / "g1_original";
  const fs::path h2_dir = args.output / "h2_retarget";
  if (!WriteMotionSequenceAsReferenceClip(*trajectories.g1, g1_dir.string(),
                                          "g1_planner_original",
                                          "G1 planner original")) {
    throw std::runtime_error("failed to export G1 motion");
  }
  if (!WriteMotionSequenceAsReferenceClip(*trajectories.h2, h2_dir.string(),
                                          "h2_planner_retarget",
                                          "G1 planner -> H2 retarget")) {
    throw std::runtime_error("failed to export H2 motion");
  }

  std::ofstream csv(args.output / "comparison_metrics.csv");
  csv << "frame,time_s,desired_segment,applied_generation,applied_source_frame,"
         "generation_lag,"
         "g1_root_x,g1_root_y,g1_root_z,"
         "h2_root_x,h2_root_y,h2_root_z,root_xyz_delta_m,root_z_delta_m,"
         "root_angle_delta_rad,joint_rmse_rad,joint_max_rad,joint_max_name,"
         "root_relative_body_rmse_m,left_foot_delta_m,right_foot_delta_m,"
         "left_hand_delta_m,right_hand_delta_m\n";
  for (int f = 0; f < trajectories.g1->timesteps; ++f) {
    const auto& m = measured.frames[f];
    const auto& gp = trajectories.g1->BodyPositions(f)[0];
    const auto& hp = trajectories.h2->BodyPositions(f)[0];
    const int desired_generation = SegmentForFrame(f, trajectories.replan_boundaries);
    const int applied_generation = trajectories.applied_generation.empty()
                                       ? desired_generation
                                       : trajectories.applied_generation[f];
    csv << f << ',' << std::fixed << std::setprecision(6) << f / kPlaybackFps
        << ',' << desired_generation << ',' << applied_generation << ','
        << trajectories.applied_source_frame[f] << ','
        << desired_generation - applied_generation << ','
        << gp[0] << ',' << gp[1] << ',' << gp[2] << ','
        << hp[0] << ',' << hp[1] << ',' << hp[2] << ','
        << m.root_xyz_delta << ',' << m.root_z_delta << ','
        << m.root_angle_delta << ',' << m.joint_rmse << ',' << m.joint_max
        << ',' << m.joint_max_name << ',' << m.body_rmse << ','
        << m.left_foot << ',' << m.right_foot << ',' << m.left_hand << ','
        << m.right_hand << '\n';
  }

  std::ofstream commands_csv(args.output / "command_schedule.csv");
  commands_csv
      << "generation,clip,start_frame,end_frame,start_time_s,end_time_s,"
         "step_in_clip,steps_in_clip,mode,style,maneuver,speed,height,"
         "move_x,move_y,move_z,face_x,face_y,face_z,face_yaw_rad,"
         "spin_rad_s,body_move_angle_rad,clip_duration_s,idle_gap\n";
  for (size_t generation = 0; generation < trajectories.commands.size();
       ++generation) {
    const PlannerCommand& command = trajectories.commands[generation];
    const int start_frame = generation == 0
                                ? 0
                                : trajectories.replan_boundaries[generation - 1];
    const int end_frame = generation < trajectories.replan_boundaries.size()
                              ? trajectories.replan_boundaries[generation] - 1
                              : trajectories.g1->timesteps - 1;
    commands_csv << generation << ',' << command.clip << ',' << start_frame
                 << ',' << end_frame << ',' << std::fixed
                 << std::setprecision(6) << start_frame / kPlaybackFps << ','
                 << end_frame / kPlaybackFps << ',' << command.step_in_clip
                 << ',' << command.steps_in_clip << ',' << command.mode << ','
                 << command.style << ',' << command.maneuver << ','
                 << command.speed << ',' << command.height << ','
                 << command.movement[0] << ',' << command.movement[1] << ','
                 << command.movement[2] << ',' << command.facing[0] << ','
                 << command.facing[1] << ',' << command.facing[2] << ','
                 << command.face_yaw << ',' << command.spin_rad_s << ','
                 << command.body_move_angle << ',' << command.clip_duration_s
                 << ',' << command.idle_gap << '\n';
  }

  std::ofstream timeline(args.output / "retarget_timeline.csv");
  timeline << "generation,scheduled_frame,publish_frame,publish_lag_frames,"
              "aligned_source_frame,planner_ms,solve_ms,control_block_ms,"
              "missed_control_ticks,continuity_root_correction_m,"
              "continuity_planar_correction_m,"
              "continuity_vertical_correction_m,"
              "continuity_yaw_correction_rad,joint_anchor_rmse_before_rad,"
              "joint_anchor_rmse_after_rad,continuity_blended_frames,"
              "vertical_recovery_frames,status\n";
  for (const auto& event : events) {
    const int publish_lag = event.publish_frame < 0
                                ? -1
                                : event.publish_frame - event.scheduled_frame;
    timeline << event.generation << ',' << event.scheduled_frame << ','
             << event.publish_frame << ',' << publish_lag << ','
             << event.aligned_source_frame << ',' << std::fixed
             << std::setprecision(6) << event.planner_ms << ','
             << event.solve_ms << ',' << event.control_block_ms << ','
             << event.missed_control_ticks << ','
             << event.continuity_root_correction_m << ','
             << event.continuity_planar_correction_m << ','
             << event.continuity_vertical_correction_m << ','
             << event.continuity_yaw_correction_rad << ','
             << event.joint_anchor_rmse_before_rad << ','
             << event.joint_anchor_rmse_after_rad << ','
             << event.continuity_blended_frames << ','
             << event.vertical_recovery_frames << ',' << event.status << '\n';
  }

  std::ofstream summary(args.output / "summary.txt");
  const MotionSmoothness g1_smoothness =
      MeasureMotionSmoothness(*trajectories.g1);
  const MotionSmoothness h2_smoothness =
      MeasureMotionSmoothness(*trajectories.h2);
  summary << "H2 kinematic planner retarget comparison\n"
          << "frames: " << trajectories.g1->timesteps << "\n"
          << "duration_s: " << trajectories.g1->timesteps / kPlaybackFps << "\n"
          << "replans: " << trajectories.commands.size() << "\n"
          << "replan_frames: " << args.replan_frames << "\n"
          << "command_mode: "
          << (args.random_locomotion ? "random" : "fixed") << "\n"
          << "random_clips: " << args.random_clips << "\n"
          << "random_duration_min_s: " << args.random_duration_min_s << "\n"
          << "random_duration_max_s: " << args.random_duration_max_s << "\n"
          << "random_gap_s: " << args.random_gap_s << "\n"
          << "random_speed_max: " << args.random_speed_max << "\n"
          << "random_core_only: " << args.random_core_only << "\n"
          << "viewer_show_solve_stalls: " << args.show_solve_stalls << "\n"
          << "retarget_execution: "
          << RetargetExecutionName(trajectories.execution) << "\n"
          << "h2_continuity: "
          << H2ContinuityModeName(trajectories.continuity) << "\n"
          << "h2_blend_frames: " << args.h2_blend_frames << "\n"
          << "h2_root_z_smooth_radius: "
          << args.h2_root_z_smooth_radius << "\n"
          << "independent_g1_timeline: "
          << args.independent_g1_timeline << "\n"
          << "control_period_ms: " << kControlPeriodMs << "\n"
          << "max_control_tick_ms: " << trajectories.max_control_tick_ms << "\n"
          << "control_deadline_misses: "
          << trajectories.control_deadline_misses << "\n"
          << "dropped_generations: " << trajectories.dropped_generations << "\n"
          << "mode: " << args.mode << "\n"
          << "speed: " << args.speed << "\n"
          << "root_z_offset: " << args.root_z_offset << "\n"
          << "retarget_max_iter: " << args.retarget_max_iter << "\n"
          << "retarget_solver: "
          << robot_retargeter::QpSolverName(args.retarget_solver) << "\n"
          << "retarget_ldlt_fast_path: "
          << args.retarget_use_ldlt_fast_path << "\n"
          << "ground_align: " << args.ground_align << "\n"
          << "common_joints: " << measured.joints.size() << "\n"
          << "common_bodies: " << measured.bodies.size() << "\n"
          << "mean_joint_rmse_rad: " << measured.mean_joint_rmse << "\n"
          << "mean_root_relative_body_rmse_m: " << measured.mean_body_rmse << "\n"
          << "mean_root_z_delta_m: " << measured.mean_root_z_delta << "\n";
  summary << "g1_max_root_step_m: " << g1_smoothness.max_root_step_m << "\n"
          << "g1_max_root_step_frame: "
          << g1_smoothness.max_root_step_frame << "\n"
          << "h2_max_root_step_m: " << h2_smoothness.max_root_step_m << "\n"
          << "h2_max_root_step_frame: "
          << h2_smoothness.max_root_step_frame << "\n"
          << "g1_max_root_velocity_delta_mps: "
          << g1_smoothness.max_root_velocity_delta_mps << "\n"
          << "g1_max_root_velocity_delta_frame: "
          << g1_smoothness.max_root_velocity_delta_frame << "\n"
          << "h2_max_root_velocity_delta_mps: "
          << h2_smoothness.max_root_velocity_delta_mps << "\n"
          << "h2_max_root_velocity_delta_frame: "
          << h2_smoothness.max_root_velocity_delta_frame << "\n"
          << "g1_max_joint_step_rmse_rad: "
          << g1_smoothness.max_joint_step_rmse_rad << "\n"
          << "g1_max_joint_step_frame: "
          << g1_smoothness.max_joint_step_frame << "\n"
          << "h2_max_joint_step_rmse_rad: "
          << h2_smoothness.max_joint_step_rmse_rad << "\n"
          << "h2_max_joint_step_frame: "
          << h2_smoothness.max_joint_step_frame << "\n"
          << "g1_max_joint_velocity_delta_rmse_rps: "
          << g1_smoothness.max_joint_velocity_delta_rmse_rps << "\n"
          << "g1_max_joint_velocity_delta_frame: "
          << g1_smoothness.max_joint_velocity_delta_frame << "\n"
          << "h2_max_joint_velocity_delta_rmse_rps: "
          << h2_smoothness.max_joint_velocity_delta_rmse_rps << "\n"
          << "h2_max_joint_velocity_delta_frame: "
          << h2_smoothness.max_joint_velocity_delta_frame << "\n";
  for (size_t i = 0; i < trajectories.planner_ms.size(); ++i) {
    summary << "replan_" << i << "_planner_ms: " << trajectories.planner_ms[i]
            << "\nreplan_" << i << "_retarget_ms: "
            << trajectories.retarget_ms[i] << '\n';
  }
  for (size_t i = 0; i < trajectories.replan_boundaries.size(); ++i) {
    const int boundary = trajectories.replan_boundaries[i];
    const BoundaryJump g1_jump = MeasureBoundaryJump(*trajectories.g1, boundary);
    const BoundaryJump h2_jump = MeasureBoundaryJump(*trajectories.h2, boundary);
    summary << "boundary_" << i << "_frame: " << boundary << '\n'
            << "boundary_" << i << "_g1_root_step_m: "
            << g1_jump.root_translation_m << '\n'
            << "boundary_" << i << "_h2_root_step_m: "
            << h2_jump.root_translation_m << '\n'
            << "boundary_" << i << "_g1_root_velocity_delta_mps: "
            << g1_jump.root_velocity_delta_mps << '\n'
            << "boundary_" << i << "_h2_root_velocity_delta_mps: "
            << h2_jump.root_velocity_delta_mps << '\n'
            << "boundary_" << i << "_g1_joint_step_rmse_rad: "
            << g1_jump.joint_rmse_rad << '\n'
            << "boundary_" << i << "_h2_joint_step_rmse_rad: "
            << h2_jump.joint_rmse_rad << '\n'
            << "boundary_" << i << "_h2_joint_step_max_rad: "
            << h2_jump.joint_max_rad << '\n'
            << "boundary_" << i << "_g1_joint_velocity_delta_rmse_rps: "
            << g1_jump.joint_velocity_delta_rmse_rps << '\n'
            << "boundary_" << i << "_h2_joint_velocity_delta_rmse_rps: "
            << h2_jump.joint_velocity_delta_rmse_rps << '\n';
    std::cout << "[planner-test] boundary " << i << " @ frame " << boundary
              << ": G1 joint step=" << g1_jump.joint_rmse_rad
              << " rad, H2 joint step=" << h2_jump.joint_rmse_rad
              << " rad (max=" << h2_jump.joint_max_rad
              << "), H2 root dv=" << h2_jump.root_velocity_delta_mps
              << " m/s\n";
  }
  for (size_t i = 0; i < trajectories.publish_boundaries.size(); ++i) {
    const int boundary = trajectories.publish_boundaries[i];
    if (boundary <= 0) continue;
    const BoundaryJump h2_jump = MeasureBoundaryJump(*trajectories.h2, boundary);
    summary << "publish_" << i << "_frame: " << boundary << '\n'
            << "publish_" << i << "_h2_root_step_m: "
            << h2_jump.root_translation_m << '\n'
            << "publish_" << i << "_h2_root_velocity_delta_mps: "
            << h2_jump.root_velocity_delta_mps << '\n'
            << "publish_" << i << "_h2_joint_step_rmse_rad: "
            << h2_jump.joint_rmse_rad << '\n'
            << "publish_" << i << "_h2_joint_step_max_rad: "
            << h2_jump.joint_max_rad << '\n'
            << "publish_" << i << "_h2_joint_velocity_delta_rmse_rps: "
            << h2_jump.joint_velocity_delta_rmse_rps << '\n';
  }
  std::cout << "[planner-test] output: " << args.output << "\n";
}

void TintModel(mjModel* model, const std::array<float, 4>& color) {
  for (int geom = 0; geom < model->ngeom; ++geom) {
    // Keep the scene's world geoms (notably the checkerboard floor) intact.
    if (model->geom_bodyid[geom] == 0) continue;
    if (model->geom_group[geom] == 1 || model->geom_contype[geom] == 0) {
      for (int c = 0; c < 4; ++c) model->geom_rgba[4 * geom + c] = color[c];
    }
  }
}

bool HasWorldPlane(const mjModel* model) {
  for (int geom = 0; geom < model->ngeom; ++geom) {
    if (model->geom_bodyid[geom] == 0 &&
        model->geom_type[geom] == mjGEOM_PLANE) {
      return true;
    }
  }
  return false;
}

bool HasSkybox(const mjModel* model) {
  for (int texture = 0; texture < model->ntex; ++texture) {
    if (model->tex_type[texture] == mjTEXTURE_SKYBOX) return true;
  }
  return false;
}

void AddLine(mjvScene* scene, const std::array<double, 3>& from,
             const std::array<double, 3>& to,
             const std::array<float, 4>& color, double width = 2.0) {
  if (scene->ngeom >= scene->maxgeom) return;
  mjvGeom* geom = &scene->geoms[scene->ngeom++];
  mjv_initGeom(geom, mjGEOM_LINE, nullptr, nullptr, nullptr, color.data());
  mjv_connector(geom, mjGEOM_LINE, width, from.data(), to.data());
}

void AddSphere(mjvScene* scene, const std::array<double, 3>& pos,
               const std::array<float, 4>& color, double radius) {
  if (scene->ngeom >= scene->maxgeom) return;
  const mjtNum size[3]{radius, radius, radius};
  mjv_initGeom(&scene->geoms[scene->ngeom++], mjGEOM_SPHERE, size,
               pos.data(), nullptr, color.data());
}

void AddFallbackFloor(mjvScene* scene) {
  if (scene->ngeom >= scene->maxgeom) return;
  const mjtNum size[3]{8.0, 8.0, 0.005};
  const mjtNum pos[3]{0.0, 0.0, -0.01};
  const float color[4]{0.18f, 0.20f, 0.23f, 0.7f};
  mjv_initGeom(&scene->geoms[scene->ngeom++], mjGEOM_BOX, size, pos,
               nullptr, color);
}

bool IsFinitePoint(const std::array<double, 3>& point) {
  return std::all_of(point.begin(), point.end(),
                     [](double value) { return std::isfinite(value); });
}

void AddTrajectoryRange(const MotionSequence& motion, int begin_frame,
                        const std::vector<int>& boundaries, mjvScene* scene,
                        const std::array<float, 4>& color) {
  begin_frame = std::clamp(begin_frame, 0, motion.timesteps);
  const int remaining = motion.timesteps - begin_frame;
  if (remaining < 2) return;
  const int stride = std::max(1, remaining / 900);
  // Even after downsampling, a human-scale 50 Hz root trajectory cannot move
  // this far between adjacent displayed samples. Skipping such a segment also
  // protects the viewer from NaN/Inf or a future alignment regression.
  const double max_segment_m = std::max(1.0, 0.2 * stride);
  const double max_segment_squared = max_segment_m * max_segment_m;
  for (int f = begin_frame + stride; f < motion.timesteps; f += stride) {
    const auto& from = motion.BodyPositions(f - stride)[0];
    const auto& to = motion.BodyPositions(f)[0];
    if (!IsFinitePoint(from) || !IsFinitePoint(to)) continue;
    double distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double delta = to[axis] - from[axis];
      distance_squared += delta * delta;
    }
    if (distance_squared > max_segment_squared) continue;
    AddLine(scene, from, to, color, 2.0);
  }
  const std::array<float, 4> boundary_color{1.0f, 0.78f, 0.15f, 1.0f};
  for (int boundary : boundaries) {
    if (boundary >= begin_frame && boundary < motion.timesteps) {
      const auto& position = motion.BodyPositions(boundary)[0];
      if (IsFinitePoint(position)) {
        AddSphere(scene, position, boundary_color, 0.025);
      }
    }
  }
}

void AddTrajectory(const MotionSequence& motion,
                   const std::vector<int>& boundaries, mjvScene* scene,
                   const std::array<float, 4>& color) {
  AddTrajectoryRange(motion, 0, boundaries, scene, color);
}

class KeyLatch {
 public:
  bool Pressed(GLFWwindow* window, int key) {
    const bool down = glfwGetKey(window, key) == GLFW_PRESS;
    const bool pressed = down && !states_[key];
    states_[key] = down;
    return pressed;
  }
 private:
  std::map<int, bool> states_;
};

void AddHistory(const std::vector<std::array<double, 3>>& history,
                mjvScene* scene, const std::array<float, 4>& color) {
  if (history.size() < 2) return;
  const size_t stride = std::max<size_t>(1, history.size() / 1000);
  for (size_t i = stride; i < history.size(); i += stride) {
    AddLine(scene, history[i - stride], history[i], color, 2.5);
  }
}

void RunLiveRandomViewer(const Args& args, mjModel* g1_model, mjData* g1_data,
                         mjModel* h2_model, mjData* h2_data) {
  if (!glfwInit()) {
    throw std::runtime_error("glfwInit failed (is DISPLAY available?)");
  }
  GLFWwindow* window = glfwCreateWindow(
      args.width, args.height_px, "H2 live planner retarget: initializing",
      nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  TintModel(g1_model, {0.86f, 0.25f, 0.22f, 1.0f});
  TintModel(h2_model, {0.20f, 0.78f, 0.34f, 1.0f});
  const bool g1_has_floor = HasWorldPlane(g1_model);
  const bool h2_has_floor = HasWorldPlane(h2_model);

  mjvCamera g1_camera;
  mjvCamera h2_camera;
  mjvOption g1_option;
  mjvOption h2_option;
  mjvPerturb g1_perturb;
  mjvPerturb h2_perturb;
  mjvScene g1_scene;
  mjvScene h2_scene;
  mjrContext g1_context;
  mjrContext h2_context;
  mjv_defaultCamera(&g1_camera);
  mjv_defaultCamera(&h2_camera);
  mjv_defaultOption(&g1_option);
  mjv_defaultOption(&h2_option);
  mjv_defaultPerturb(&g1_perturb);
  mjv_defaultPerturb(&h2_perturb);
  mjv_defaultScene(&g1_scene);
  mjv_defaultScene(&h2_scene);
  mjr_defaultContext(&g1_context);
  mjr_defaultContext(&h2_context);
  mjv_makeScene(g1_model, &g1_scene, 6000);
  mjv_makeScene(h2_model, &h2_scene, 6000);
  mjr_makeContext(g1_model, &g1_context, mjFONTSCALE_150);
  mjr_makeContext(h2_model, &h2_context, mjFONTSCALE_150);
  for (auto* camera : {&g1_camera, &h2_camera}) {
    camera->type = mjCAMERA_FREE;
    camera->azimuth = 145.0;
    camera->elevation = -18.0;
    camera->distance = 2.7;
    camera->lookat[2] = 0.65;
  }
  mj_forward(g1_model, g1_data);
  mj_forward(h2_model, h2_data);

  LivePlannerRetargetPipeline pipeline(args);
  pipeline.Start();
  std::shared_ptr<LiveWindow> active;
  int64_t global_frame = 0;
  int64_t last_request_frame = 0;
  int64_t next_request_frame = args.replan_frames;
  int next_generation = 1;
  int publish_lag_frames = 0;
  RetargetEvent last_continuity;
  bool playing = args.autoplay;
  bool follow_root = true;
  double playback_speed = 1.0;
  double playback_accumulator_s = 0.0;
  double last_wall_time = glfwGetTime();
  int64_t last_history_frame = -1;
  std::vector<std::array<double, 3>> g1_history;
  std::vector<std::array<double, 3>> h2_history;
  KeyLatch keys;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    const double now = glfwGetTime();
    const double wall_elapsed_s = std::max(0.0, now - last_wall_time);
    last_wall_time = now;
    bool reset_playback_clock = false;
    if (keys.Pressed(window, GLFW_KEY_ESCAPE)) {
      glfwSetWindowShouldClose(window, 1);
    }
    if (keys.Pressed(window, GLFW_KEY_SPACE)) {
      playing = !playing;
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_F)) follow_root = !follow_root;
    if (keys.Pressed(window, GLFW_KEY_LEFT) && active) {
      playing = false;
      global_frame = std::max(active->scheduled_frame, global_frame - 1);
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_RIGHT) && active) {
      playing = false;
      const int64_t last = active->scheduled_frame + active->g1->timesteps - 1;
      global_frame = std::min(last, global_frame + 1);
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_UP)) {
      playback_speed = std::min(4.0, playback_speed * 2.0);
    }
    if (keys.Pressed(window, GLFW_KEY_DOWN)) {
      playback_speed = std::max(0.125, playback_speed * 0.5);
    }
    if (keys.Pressed(window, GLFW_KEY_A)) g1_camera.azimuth -= 10.0;
    if (keys.Pressed(window, GLFW_KEY_D)) g1_camera.azimuth += 10.0;
    if (keys.Pressed(window, GLFW_KEY_W)) g1_camera.elevation += 5.0;
    if (keys.Pressed(window, GLFW_KEY_S)) g1_camera.elevation -= 5.0;
    if (keys.Pressed(window, GLFW_KEY_Q)) {
      g1_camera.distance = std::max(0.5, g1_camera.distance - 0.2);
    }
    if (keys.Pressed(window, GLFW_KEY_E)) g1_camera.distance += 0.2;
    h2_camera.azimuth = g1_camera.azimuth;
    h2_camera.elevation = g1_camera.elevation;
    h2_camera.distance = g1_camera.distance;

    if (auto ready = pipeline.TakePublished()) {
      if (!active) {
        active = std::move(ready);
        global_frame = active->scheduled_frame;
        last_request_frame = active->scheduled_frame;
        next_request_frame = global_frame + args.replan_frames;
        playback_accumulator_s = 0.0;
        last_history_frame = -1;
      } else {
        const int aligned_frame = static_cast<int>(
            global_frame - ready->scheduled_frame);
        if (aligned_frame >= 0 && aligned_frame < ready->h2->timesteps) {
          const int previous_frame = std::clamp(
              static_cast<int>(global_frame - active->scheduled_frame), 0,
              active->h2->timesteps - 1);
          last_continuity = {};
          ApplyConfiguredContinuity(args, *active->h2, previous_frame,
                                    *ready->h2, aligned_frame,
                                    last_continuity);
          publish_lag_frames = aligned_frame;
          active = std::move(ready);
          std::ostringstream publish_log;
          publish_log << "[planner-test/live] publish generation "
                      << active->generation << " @ global frame "
                      << global_frame << " lag=" << publish_lag_frames
                      << " frames planar_anchor=" << std::fixed
                      << std::setprecision(2)
                      << last_continuity.continuity_planar_correction_m * 100.0
                      << "cm transient_z="
                      << last_continuity.continuity_vertical_correction_m *
                             1000.0
                      << "mm joint_anchor="
                      << last_continuity.joint_anchor_rmse_before_rad << "->"
                      << last_continuity.joint_anchor_rmse_after_rad
                      << "rad\n";
          std::cout << publish_log.str();
        }
      }
    }

    if (reset_playback_clock || !playing || !active) {
      playback_accumulator_s = 0.0;
    }
    if (playing && active) {
      playback_accumulator_s +=
          (reset_playback_clock ? 0.0 : wall_elapsed_s) * playback_speed;
      const double frame_duration_s = 1.0 / kPlaybackFps;
      int catchup_frames = 0;
      while (playback_accumulator_s + 1e-12 >= frame_duration_s &&
             catchup_frames < 500) {
        playback_accumulator_s -= frame_duration_s;
        ++global_frame;
        ++catchup_frames;
      }
      if (catchup_frames == 500) playback_accumulator_s = 0.0;
    }

    if (active && global_frame >= next_request_frame && !pipeline.Busy()) {
      const int source_advance = static_cast<int>(std::clamp<int64_t>(
          global_frame - last_request_frame, 1, kPlannerCapacity - 1));
      if (pipeline.Submit(
              {next_generation, global_frame, source_advance})) {
        last_request_frame = global_frame;
        next_request_frame = global_frame + args.replan_frames;
        ++next_generation;
      }
    }

    int active_source_frame = 0;
    if (active) {
      active_source_frame = std::clamp(
          static_cast<int>(global_frame - active->scheduled_frame), 0,
          std::min(active->g1->timesteps, active->h2->timesteps) - 1);
      SetFrame(*active->g1, active_source_frame, g1_model, g1_data);
      SetFrame(*active->h2, active_source_frame, h2_model, h2_data);
      if (global_frame != last_history_frame) {
        g1_history.push_back(active->g1->BodyPositions(active_source_frame)[0]);
        h2_history.push_back(active->h2->BodyPositions(active_source_frame)[0]);
        last_history_frame = global_frame;
        if (g1_history.size() > 3000) {
          g1_history.erase(g1_history.begin(), g1_history.begin() + 1000);
          h2_history.erase(h2_history.begin(), h2_history.begin() + 1000);
        }
      }
      if (follow_root) {
        for (int i = 0; i < 3; ++i) {
          g1_camera.lookat[i] = g1_data->qpos[i];
          h2_camera.lookat[i] = h2_data->qpos[i];
        }
        g1_camera.lookat[2] = std::max(0.55, g1_camera.lookat[2] - 0.15);
        h2_camera.lookat[2] = std::max(0.65, h2_camera.lookat[2] - 0.25);
      }
    }

    mjv_updateScene(g1_model, g1_data, &g1_option, &g1_perturb, &g1_camera,
                    mjCAT_ALL, &g1_scene);
    mjv_updateScene(h2_model, h2_data, &h2_option, &h2_perturb, &h2_camera,
                    mjCAT_ALL, &h2_scene);
    if (!g1_has_floor) AddFallbackFloor(&g1_scene);
    if (!h2_has_floor) AddFallbackFloor(&h2_scene);
    AddHistory(g1_history, &g1_scene, {0.95f, 0.35f, 0.25f, 1.0f});
    AddHistory(h2_history, &h2_scene, {0.30f, 0.95f, 0.45f, 1.0f});
    if (active) {
      // Expired frames before active_source_frame can remain in a window's
      // pre-publication coordinate frame when an async solve arrives late.
      // History already shows the past, so render only the currently valid
      // prediction suffix and never connect that stale prefix to world space.
      AddTrajectoryRange(*active->g1, active_source_frame, {}, &g1_scene,
                         {0.95f, 0.35f, 0.25f, 0.35f});
      AddTrajectoryRange(*active->h2, active_source_frame, {}, &h2_scene,
                         {0.30f, 0.95f, 0.45f, 0.35f});
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const mjrRect left{0, 0, width / 2, height};
    const mjrRect right{width / 2, 0, width - width / 2, height};
    mjr_render(left, &g1_scene, &g1_context);
    mjr_render(right, &h2_scene, &h2_context);

    const std::string worker_error = pipeline.Error();
    std::ostringstream left_text;
    std::ostringstream right_text;
    if (!active) {
      left_text << "LIVE G1 planner\ninitializing TensorRT planner...";
      right_text << "LIVE H2 retarget\ninitializing retargeter...";
    } else {
      left_text << "LIVE G1 planner (red)\nglobal frame=" << global_frame
                << "  t=" << std::fixed << std::setprecision(2)
                << global_frame / kPlaybackFps << " s\ngeneration="
                << active->generation << " source=" << active_source_frame
                << "\ncommand clip=" << active->command.clip << " "
                << active->command.style << "/"
                << active->command.maneuver << " speed="
                << active->command.speed << "\nyaw="
                << active->command.face_yaw << " move=("
                << active->command.movement[0] << ","
                << active->command.movement[1] << ")";
      right_text << "LIVE H2 retarget (green)\ngeneration="
                 << active->generation << " source=" << active_source_frame
                 << " publish lag=" << publish_lag_frames
                 << " frames\nplanner=" << std::fixed << std::setprecision(2)
                 << active->planner_ms << " ms  retarget="
                 << active->solve_ms << " ms\nplanar anchor="
                 << last_continuity.continuity_planar_correction_m * 100.0
                 << " cm  transient z="
                 << last_continuity.continuity_vertical_correction_m * 1000.0
                 << " mm\njoint anchor="
                 << last_continuity.joint_anchor_rmse_before_rad << "->"
                 << last_continuity.joint_anchor_rmse_after_rad
                 << " rad\nworker=" << (pipeline.Busy() ? "busy" : "idle")
                 << "  clock=wall 50Hz\n"
                 << (playing ? "PLAY" : "PAUSE") << " speed="
                 << playback_speed << "x follow="
                 << (follow_root ? "on" : "off");
    }
    if (!worker_error.empty()) {
      right_text << "\nERROR: " << worker_error;
    }
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, left, left_text.str().c_str(),
                "Space pause | arrows step/speed | Esc quit", &g1_context);
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, right, right_text.str().c_str(),
                "A/D orbit | W/S elevation | Q/E zoom | F follow", &h2_context);

    std::ostringstream title;
    title << "H2 LIVE planner retarget | ";
    if (active) {
      title << active->command.style << "/" << active->command.maneuver
            << " | frame " << global_frame << " | generation "
            << active->generation << " | lag " << publish_lag_frames << "f";
    } else {
      title << (worker_error.empty() ? "initializing" : worker_error);
    }
    glfwSetWindowTitle(window, title.str().c_str());
    glfwSwapBuffers(window);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  pipeline.Stop();
  mjr_freeContext(&h2_context);
  mjr_freeContext(&g1_context);
  mjv_freeScene(&h2_scene);
  mjv_freeScene(&g1_scene);
  glfwDestroyWindow(window);
  glfwTerminate();
}

void RunViewer(const Args& args, const TrajectoryPair& trajectories,
               const MeasuredTrajectory& measured, mjModel* g1_model,
               mjData* g1_data, mjModel* h2_model, mjData* h2_data) {
  if (!glfwInit()) throw std::runtime_error("glfwInit failed (is DISPLAY available?)");
  GLFWwindow* window = glfwCreateWindow(args.width, args.height_px,
                                        "H2 planner retarget kinematics", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  TintModel(g1_model, {0.86f, 0.25f, 0.22f, 1.0f});
  TintModel(h2_model, {0.20f, 0.78f, 0.34f, 1.0f});
  const bool g1_has_floor = HasWorldPlane(g1_model);
  const bool h2_has_floor = HasWorldPlane(h2_model);
  std::cout << "[planner-test] viewer environment: G1 floor="
            << (g1_has_floor ? "MJCF" : "fallback")
            << " sky=" << (HasSkybox(g1_model) ? "MJCF" : "default")
            << ", H2 floor=" << (h2_has_floor ? "MJCF" : "fallback")
            << " sky=" << (HasSkybox(h2_model) ? "MJCF" : "default")
            << '\n';

  mjvCamera g1_camera;
  mjvCamera h2_camera;
  mjvOption g1_option;
  mjvOption h2_option;
  mjvPerturb g1_perturb;
  mjvPerturb h2_perturb;
  mjvScene g1_scene;
  mjvScene h2_scene;
  mjrContext g1_context;
  mjrContext h2_context;
  mjv_defaultCamera(&g1_camera);
  mjv_defaultCamera(&h2_camera);
  mjv_defaultOption(&g1_option);
  mjv_defaultOption(&h2_option);
  mjv_defaultPerturb(&g1_perturb);
  mjv_defaultPerturb(&h2_perturb);
  mjv_defaultScene(&g1_scene);
  mjv_defaultScene(&h2_scene);
  mjr_defaultContext(&g1_context);
  mjr_defaultContext(&h2_context);
  mjv_makeScene(g1_model, &g1_scene, 6000);
  mjv_makeScene(h2_model, &h2_scene, 6000);
  mjr_makeContext(g1_model, &g1_context, mjFONTSCALE_150);
  mjr_makeContext(h2_model, &h2_context, mjFONTSCALE_150);

  for (auto* camera : {&g1_camera, &h2_camera}) {
    camera->type = mjCAMERA_FREE;
    camera->azimuth = 145.0;
    camera->elevation = -18.0;
    camera->distance = 2.7;
    camera->lookat[2] = 0.65;
  }

  int frame = 0;
  bool playing = args.autoplay;
  bool follow_root = true;
  double playback_speed = 1.0;
  double playback_accumulator_s = 0.0;
  double last_wall_time = glfwGetTime();
  KeyLatch keys;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    const double now = glfwGetTime();
    const double wall_elapsed_s = std::max(0.0, now - last_wall_time);
    last_wall_time = now;
    bool reset_playback_clock = false;
    if (keys.Pressed(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, 1);
    if (keys.Pressed(window, GLFW_KEY_SPACE)) {
      playing = !playing;
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_R)) {
      frame = 0;
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_F)) follow_root = !follow_root;
    if (keys.Pressed(window, GLFW_KEY_LEFT)) {
      playing = false;
      frame = std::max(frame - 1, 0);
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_RIGHT)) {
      playing = false;
      frame = std::min(frame + 1, trajectories.g1->timesteps - 1);
      reset_playback_clock = true;
    }
    if (keys.Pressed(window, GLFW_KEY_UP)) playback_speed = std::min(4.0, playback_speed * 2.0);
    if (keys.Pressed(window, GLFW_KEY_DOWN)) playback_speed = std::max(0.125, playback_speed * 0.5);
    if (keys.Pressed(window, GLFW_KEY_A)) g1_camera.azimuth -= 10.0;
    if (keys.Pressed(window, GLFW_KEY_D)) g1_camera.azimuth += 10.0;
    if (keys.Pressed(window, GLFW_KEY_W)) g1_camera.elevation += 5.0;
    if (keys.Pressed(window, GLFW_KEY_S)) g1_camera.elevation -= 5.0;
    if (keys.Pressed(window, GLFW_KEY_Q)) g1_camera.distance = std::max(0.5, g1_camera.distance - 0.2);
    if (keys.Pressed(window, GLFW_KEY_E)) g1_camera.distance += 0.2;
    h2_camera.azimuth = g1_camera.azimuth;
    h2_camera.elevation = g1_camera.elevation;
    h2_camera.distance = g1_camera.distance;

    if (reset_playback_clock || !playing) playback_accumulator_s = 0.0;
    if (playing) {
      // Track trajectory time independently from the render/vsync cadence.
      // Keeping the remainder avoids 50 Hz being quantized to 30 Hz on a
      // 60 Hz display. Catch-up can advance multiple frames after a slow draw.
      playback_accumulator_s +=
          (reset_playback_clock ? 0.0 : wall_elapsed_s) * playback_speed;
      int catchup_frames = 0;
      const int max_catchup_frames = trajectories.g1->timesteps;
      while (catchup_frames < max_catchup_frames) {
        const double hold_s =
            args.show_solve_stalls && !trajectories.frame_hold_ms.empty()
                ? trajectories.frame_hold_ms[frame] / 1000.0
                : 0.0;
        const double frame_duration_s = 1.0 / kPlaybackFps + hold_s;
        if (playback_accumulator_s + 1e-12 < frame_duration_s) break;
        playback_accumulator_s -= frame_duration_s;
        frame = (frame + 1) % trajectories.g1->timesteps;
        ++catchup_frames;
      }
      if (catchup_frames == max_catchup_frames) {
        // A long window suspension should not make the next render spend
        // unbounded time replaying complete invisible trajectory loops.
        playback_accumulator_s = 0.0;
      }
    }

    SetFrame(*trajectories.g1, frame, g1_model, g1_data);
    SetFrame(*trajectories.h2, frame, h2_model, h2_data);
    if (follow_root) {
      for (int i = 0; i < 3; ++i) {
        g1_camera.lookat[i] = g1_data->qpos[i];
        h2_camera.lookat[i] = h2_data->qpos[i];
      }
      g1_camera.lookat[2] = std::max(0.55, g1_camera.lookat[2] - 0.15);
      h2_camera.lookat[2] = std::max(0.65, h2_camera.lookat[2] - 0.25);
    }

    mjv_updateScene(g1_model, g1_data, &g1_option, &g1_perturb, &g1_camera,
                    mjCAT_ALL, &g1_scene);
    mjv_updateScene(h2_model, h2_data, &h2_option, &h2_perturb, &h2_camera,
                    mjCAT_ALL, &h2_scene);
    // Scene MJCFs contribute their own textured plane. Robot-only MJCFs still
    // get a neutral fallback floor, without duplicating or z-fighting a plane.
    if (!g1_has_floor) AddFallbackFloor(&g1_scene);
    if (!h2_has_floor) AddFallbackFloor(&h2_scene);
    const auto& trajectory_markers =
        args.random_locomotion ? trajectories.command_boundaries
                               : trajectories.replan_boundaries;
    const auto& h2_trajectory_markers =
        args.random_locomotion ? trajectories.command_boundaries
                               : trajectories.publish_boundaries;
    AddTrajectory(*trajectories.g1, trajectory_markers, &g1_scene,
                  {0.95f, 0.35f, 0.25f, 1.0f});
    AddTrajectory(*trajectories.h2, h2_trajectory_markers, &h2_scene,
                  {0.30f, 0.95f, 0.45f, 1.0f});

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const mjrRect left{0, 0, width / 2, height};
    const mjrRect right{width / 2, 0, width - width / 2, height};
    mjr_render(left, &g1_scene, &g1_context);
    mjr_render(right, &h2_scene, &h2_context);

    const auto& metric = measured.frames[frame];
    const int desired_generation =
        SegmentForFrame(frame, trajectories.replan_boundaries);
    const int applied_generation = trajectories.applied_generation.empty()
                                       ? desired_generation
                                       : trajectories.applied_generation[frame];
    const PlannerCommand& desired_command = trajectories.commands.at(
        std::clamp(desired_generation, 0,
                   static_cast<int>(trajectories.commands.size()) - 1));
    const PlannerCommand& applied_command = trajectories.commands.at(
        std::clamp(applied_generation, 0,
                   static_cast<int>(trajectories.commands.size()) - 1));
    std::ostringstream left_text;
    left_text << "G1 planner original (red)\nframe " << frame << "/"
              << trajectories.g1->timesteps - 1 << "  t=" << std::fixed
              << std::setprecision(2) << frame / kPlaybackFps << " s\nroot z="
              << trajectories.g1->BodyPositions(frame)[0][2]
              << " m\ndesired generation=" << desired_generation
              << "  G1 timeline="
              << (args.independent_g1_timeline ? "independent" : "legacy")
              << "\ncommand clip=" << desired_command.clip << " "
              << desired_command.style << "/" << desired_command.maneuver
              << "  speed=" << desired_command.speed
              << "\nyaw=" << desired_command.face_yaw << " rad  move=("
              << desired_command.movement[0] << ","
              << desired_command.movement[1] << ")";
    std::ostringstream right_text;
    right_text << "H2 retarget (green)\nroot z=" << std::fixed
               << std::setprecision(3)
               << trajectories.h2->BodyPositions(frame)[0][2] << " m  dz="
               << metric.root_z_delta << "\njoint RMSE=" << metric.joint_rmse
               << " rad  max=" << metric.joint_max << " ("
               << metric.joint_max_name << ")\nbody RMSE(root-local)="
               << metric.body_rmse * 1000.0 << " mm\n"
               << "execution=" << RetargetExecutionName(trajectories.execution)
               << "  continuity="
               << H2ContinuityModeName(trajectories.continuity)
               << "  applied generation=" << applied_generation
               << " source frame=" << trajectories.applied_source_frame[frame]
               << "  lag=" << desired_generation - applied_generation << "\n"
               << "applied command=" << applied_command.style << "/"
               << applied_command.maneuver << " clip=" << applied_command.clip
               << "\n"
               << "max control tick=" << std::setprecision(2)
               << trajectories.max_control_tick_ms << " ms  misses="
               << trajectories.control_deadline_misses << "\n"
               << (playing ? "PLAY" : "PAUSE") << "  speed="
               << playback_speed << "x  follow=" << (follow_root ? "on" : "off")
               << "\nclock=wall 50Hz  solve stalls="
               << (args.show_solve_stalls ? "on" : "off");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, left, left_text.str().c_str(),
                "Space pause | arrows step/speed | R reset", &g1_context);
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, right, right_text.str().c_str(),
                "A/D orbit | W/S elevation | Q/E zoom | F follow", &h2_context);

    std::ostringstream title;
    title << "H2 planner retarget ["
          << RetargetExecutionName(trajectories.execution) << "/"
          << H2ContinuityModeName(trajectories.continuity) << "] | "
          << desired_command.style << "/" << desired_command.maneuver
          << " | frame "
          << frame << " | body RMSE "
          << std::fixed << std::setprecision(1) << metric.body_rmse * 1000.0
          << " mm | joint RMSE " << std::setprecision(3) << metric.joint_rmse
          << " rad";
    glfwSetWindowTitle(window, title.str().c_str());
    glfwSwapBuffers(window);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  mjr_freeContext(&h2_context);
  mjr_freeContext(&g1_context);
  mjv_freeScene(&h2_scene);
  mjv_freeScene(&g1_scene);
  glfwDestroyWindow(window);
  glfwTerminate();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    ValidateInputs(args);
    std::cout << "[planner-test] Pure kinematics: no policy, encoder, DDS, or motor commands\n"
              << "[planner-test] command_mode="
              << (args.live_random
                      ? "live_random"
                      : (args.random_locomotion ? "random" : "fixed"))
              << " mode=" << args.mode << " speed=" << args.speed
              << " replans=" << args.replans
              << " random_clips=" << args.random_clips
              << " replan_frames=" << args.replan_frames
              << " retarget_solver="
              << robot_retargeter::QpSolverName(args.retarget_solver)
              << " retarget_execution="
              << (args.live_random
                      ? "live_worker"
                      : RetargetExecutionName(args.retarget_execution))
              << " h2_continuity="
              << H2ContinuityModeName(args.h2_continuity)
              << " independent_g1_timeline="
              << args.independent_g1_timeline
              << " h2_root_z_smooth_radius="
              << args.h2_root_z_smooth_radius << "\n";

    if (args.live_random) {
      ModelPtr g1_model = LoadModel(args.g1_mjcf);
      ModelPtr h2_model = LoadModel(args.h2_mjcf);
      DataPtr g1_data(mj_makeData(g1_model.get()));
      DataPtr h2_data(mj_makeData(h2_model.get()));
      if (!g1_data || !h2_data) throw std::runtime_error("mj_makeData failed");
      std::cout << "[planner-test/live] viewer opens now; planner and retarget "
                   "initialize in the background\n";
      RunLiveRandomViewer(args, g1_model.get(), g1_data.get(), h2_model.get(),
                          h2_data.get());
      return 0;
    }

    GeneratedTrajectories generated = GenerateTrajectories(args);
    TrajectoryPair& trajectories = generated.trajectories;
    ModelPtr g1_model = LoadModel(args.g1_mjcf);
    ModelPtr h2_model = LoadModel(args.h2_mjcf);
    DataPtr g1_data(mj_makeData(g1_model.get()));
    DataPtr h2_data(mj_makeData(h2_model.get()));
    if (!g1_data || !h2_data) throw std::runtime_error("mj_makeData failed");

    MeasuredTrajectory measured = MeasureTrajectory(
        trajectories, g1_model.get(), g1_data.get(), h2_model.get(), h2_data.get());
    ExportResults(args, trajectories, measured, generated.events);
    std::cout << "[planner-test] common joints=" << measured.joints.size()
              << " common bodies=" << measured.bodies.size() << "\n"
              << "[planner-test] mean joint RMSE=" << measured.mean_joint_rmse
              << " rad, root-local body RMSE=" << measured.mean_body_rmse * 1000.0
              << " mm, root dz=" << measured.mean_root_z_delta * 1000.0
              << " mm\n";

    if (!args.headless) {
      RunViewer(args, trajectories, measured, g1_model.get(), g1_data.get(),
                h2_model.get(), h2_data.get());
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[planner-test] ERROR: " << error.what() << '\n';
    return 1;
  }
}
