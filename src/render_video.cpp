#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "policy_parameters.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path g1_mjcf;
  fs::path h2_mjcf;
  fs::path g1_clip;
  fs::path h2_clip;
  fs::path schedule;
  int width = 1280;
  int height = 720;
  int fps = 50;
  int max_frames = 0;
};

void PrintHelp() {
  std::cout
      << "Usage: g1_planner_to_h2_renderer --g1-mjcf FILE --h2-mjcf FILE\n"
         "       --g1-clip DIR --h2-clip DIR [--width 1280] [--height 720]\n"
         "       [--schedule FILE] [--fps 50] [--max-frames N]\n\n"
         "Writes headerless RGB24 frames to stdout. Diagnostics go to stderr.\n";
}

int ParsePositiveInt(const std::string& text, const char* option,
                     bool allow_zero = false) {
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size() || value < (allow_zero ? 0 : 1)) {
    throw std::runtime_error(std::string(option) + " requires " +
                             (allow_zero ? "a non-negative" : "a positive") +
                             " integer");
  }
  return value;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      PrintHelp();
      std::exit(0);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + option);
    }
    const std::string value = argv[++i];
    if (option == "--g1-mjcf") {
      args.g1_mjcf = value;
    } else if (option == "--h2-mjcf") {
      args.h2_mjcf = value;
    } else if (option == "--g1-clip") {
      args.g1_clip = value;
    } else if (option == "--h2-clip") {
      args.h2_clip = value;
    } else if (option == "--schedule") {
      args.schedule = value;
    } else if (option == "--width") {
      args.width = ParsePositiveInt(value, "--width");
    } else if (option == "--height") {
      args.height = ParsePositiveInt(value, "--height");
    } else if (option == "--fps") {
      args.fps = ParsePositiveInt(value, "--fps");
    } else if (option == "--max-frames") {
      args.max_frames = ParsePositiveInt(value, "--max-frames", true);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (args.g1_mjcf.empty() || args.h2_mjcf.empty() ||
      args.g1_clip.empty() || args.h2_clip.empty()) {
    throw std::runtime_error(
        "--g1-mjcf, --h2-mjcf, --g1-clip and --h2-clip are required");
  }
  if (args.width < 640 || args.height < 360 || args.width % 2 != 0 ||
      args.height % 2 != 0) {
    throw std::runtime_error(
        "video dimensions must be even and at least 640x360");
  }
  return args;
}

std::vector<std::vector<double>> ReadCsv(const fs::path& path,
                                         std::size_t columns) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("empty CSV: " + path.string());
  }
  std::vector<std::vector<double>> rows;
  int line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;
    std::vector<double> row;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
      std::size_t consumed = 0;
      const double value = std::stod(field, &consumed);
      if (consumed != field.size() || !std::isfinite(value)) {
        throw std::runtime_error("invalid number in " + path.string() +
                                 ":" + std::to_string(line_number));
      }
      row.push_back(value);
    }
    if (row.size() != columns) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_number) + " has " +
                               std::to_string(row.size()) + " columns; expected " +
                               std::to_string(columns));
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) throw std::runtime_error("no frames in " + path.string());
  return rows;
}

struct Clip {
  std::vector<std::vector<double>> joints;
  std::vector<std::vector<double>> positions;
  std::vector<std::vector<double>> quaternions;
};

Clip LoadClip(const fs::path& directory) {
  Clip clip{ReadCsv(directory / "joint_pos.csv", 29),
            ReadCsv(directory / "body_pos.csv", 3),
            ReadCsv(directory / "body_quat.csv", 4)};
  if (clip.joints.size() != clip.positions.size() ||
      clip.joints.size() != clip.quaternions.size()) {
    throw std::runtime_error("clip CSV row counts differ in " +
                             directory.string());
  }
  return clip;
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

struct ScheduleEntry {
  int clip = 0;
  int start_frame = 0;
  int end_frame = 0;
  std::string style;
  std::string maneuver;
  double speed = -1.0;
};

std::vector<ScheduleEntry> LoadSchedule(const fs::path& path) {
  if (path.empty()) return {};
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("empty schedule CSV: " + path.string());
  }
  const auto header = SplitCsvLine(line);
  auto column = [&](const std::string& name) {
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end()) {
      throw std::runtime_error("schedule CSV is missing column: " + name);
    }
    return static_cast<std::size_t>(std::distance(header.begin(), found));
  };
  const std::size_t clip_column = column("clip");
  const std::size_t start_column = column("start_frame");
  const std::size_t end_column = column("end_frame");
  const std::size_t style_column = column("style");
  const std::size_t maneuver_column = column("maneuver");
  const std::size_t speed_column = column("speed");
  const std::size_t required_columns =
      1 + std::max({clip_column, start_column, end_column, style_column,
                    maneuver_column, speed_column});

  std::vector<ScheduleEntry> entries;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = SplitCsvLine(line);
    if (fields.size() < required_columns) {
      throw std::runtime_error("short row in schedule CSV: " + path.string());
    }
    ScheduleEntry entry;
    entry.clip = std::stoi(fields[clip_column]);
    entry.start_frame = std::stoi(fields[start_column]);
    entry.end_frame = std::stoi(fields[end_column]);
    entry.style = fields[style_column];
    entry.maneuver = fields[maneuver_column];
    entry.speed = std::stod(fields[speed_column]);
    if (entry.start_frame < 0 || entry.end_frame < entry.start_frame) {
      throw std::runtime_error("invalid frame range in schedule CSV");
    }
    entries.push_back(std::move(entry));
  }
  if (entries.empty()) {
    throw std::runtime_error("no entries in schedule CSV: " + path.string());
  }
  return entries;
}

struct ModelDeleter {
  void operator()(mjModel* model) const { mj_deleteModel(model); }
};
struct DataDeleter {
  void operator()(mjData* data) const { mj_deleteData(data); }
};
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

ModelPtr LoadModel(const fs::path& path) {
  std::array<char, 4096> error{};
  ModelPtr model(mj_loadXML(path.c_str(), nullptr, error.data(), error.size()));
  if (!model) {
    throw std::runtime_error("failed to load " + path.string() + ": " +
                             error.data());
  }
  if (model->nq < 36) {
    throw std::runtime_error(path.string() +
                             " has fewer than the expected 36 qpos values");
  }
  return model;
}

class EglContext {
 public:
  EglContext() {
    const auto query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT"));
    const auto platform_display =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (query_devices && platform_display) {
      EGLDeviceEXT devices[16];
      EGLint count = 0;
      if (query_devices(16, devices, &count)) {
        for (int i = 0; i < count; ++i) {
          EGLDisplay candidate = platform_display(
              EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
          if (candidate != EGL_NO_DISPLAY &&
              eglInitialize(candidate, nullptr, nullptr)) {
            display_ = candidate;
            break;
          }
        }
      }
    }
    if (display_ == EGL_NO_DISPLAY) {
      EGLDisplay candidate = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (candidate != EGL_NO_DISPLAY &&
          eglInitialize(candidate, nullptr, nullptr)) {
        display_ = candidate;
      }
    }
    if (display_ == EGL_NO_DISPLAY || !eglBindAPI(EGL_OPENGL_API)) {
      throw std::runtime_error("cannot initialize headless EGL/OpenGL");
    }
    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE,     8,               EGL_GREEN_SIZE,     8,
        EGL_BLUE_SIZE,    8,               EGL_DEPTH_SIZE,     24,
        EGL_NONE};
    EGLConfig config = nullptr;
    EGLint count = 0;
    if (!eglChooseConfig(display_, config_attributes, &config, 1, &count) ||
        count == 0) {
      throw std::runtime_error("no compatible EGL framebuffer config");
    }
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, nullptr);
    if (context_ == EGL_NO_CONTEXT) {
      throw std::runtime_error("cannot create EGL OpenGL context");
    }
    if (!eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_)) {
      const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1,
                                           EGL_NONE};
      surface_ = eglCreatePbufferSurface(display_, config, surface_attributes);
      if (surface_ == EGL_NO_SURFACE ||
          !eglMakeCurrent(display_, surface_, surface_, context_)) {
        throw std::runtime_error("cannot activate EGL OpenGL context");
      }
    }
  }

  ~EglContext() {
    if (display_ == EGL_NO_DISPLAY) return;
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
    if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
    eglTerminate(display_);
  }

  EglContext(const EglContext&) = delete;
  EglContext& operator=(const EglContext&) = delete;

 private:
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLContext context_ = EGL_NO_CONTEXT;
  EGLSurface surface_ = EGL_NO_SURFACE;
};

void TintModel(mjModel* model, const std::array<float, 4>& color) {
  for (int geom = 0; geom < model->ngeom; ++geom) {
    if (model->geom_bodyid[geom] == 0) continue;
    if (model->geom_group[geom] == 1 || model->geom_contype[geom] == 0) {
      std::copy(color.begin(), color.end(), model->geom_rgba + 4 * geom);
    }
  }
}

void AddFallbackFloor(mjvScene* scene) {
  if (scene->ngeom >= scene->maxgeom) return;
  const mjtNum size[3]{8.0, 8.0, 0.005};
  const mjtNum position[3]{0.0, 0.0, -0.01};
  const float color[4]{0.18f, 0.20f, 0.23f, 0.7f};
  mjv_initGeom(&scene->geoms[scene->ngeom++], mjGEOM_BOX, size, position,
               nullptr, color);
}

void AddPath(const Clip& clip, std::size_t frame, mjvScene* scene,
             const std::array<float, 4>& color) {
  const std::size_t remaining = clip.positions.size() - frame;
  const std::size_t stride = std::max<std::size_t>(1, remaining / 300);
  for (std::size_t i = frame + stride; i < clip.positions.size(); i += stride) {
    if (scene->ngeom >= scene->maxgeom) return;
    const auto& from = clip.positions[i - stride];
    const auto& to = clip.positions[i];
    mjvGeom* geom = &scene->geoms[scene->ngeom++];
    mjv_initGeom(geom, mjGEOM_LINE, nullptr, nullptr, nullptr, color.data());
    mjv_connector(geom, mjGEOM_LINE, 2.0, from.data(), to.data());
  }
}

class PanelRenderer {
 public:
  PanelRenderer(const fs::path& model_path, const Clip& clip, int width,
                int height, std::string title,
                std::array<float, 4> tint)
      : model_(LoadModel(model_path)), data_(mj_makeData(model_.get())),
        clip_(clip), width_(width), height_(height), title_(std::move(title)),
        tint_(tint), bottom_up_(static_cast<std::size_t>(width) * height * 3),
        top_down_(bottom_up_.size()) {
    if (!data_) throw std::runtime_error("mj_makeData failed");
    TintModel(model_.get(), tint_);
    model_->vis.global.offwidth = width_;
    model_->vis.global.offheight = height_;
    mjv_defaultCamera(&camera_);
    mjv_defaultOption(&option_);
    // These MJCFs contain both visible meshes (group 1) and a second set of
    // collision geoms (group 0). Render only the presentation meshes; add a
    // neutral floor below after scene creation because the model floor also
    // belongs to group 0.
    option_.geomgroup[0] = 0;
    option_.geomgroup[1] = 1;
    mjv_defaultPerturb(&perturb_);
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    mjv_makeScene(model_.get(), &scene_, 6000);
    mjr_makeContext(model_.get(), &context_, mjFONTSCALE_150);
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    if (context_.currentBuffer != mjFB_OFFSCREEN) {
      throw std::runtime_error("MuJoCo offscreen framebuffer is unavailable");
    }
    camera_.type = mjCAMERA_FREE;
    camera_.azimuth = 145.0;
    camera_.elevation = -18.0;
    camera_.distance = 2.7;
    camera_.lookat[2] = 0.65;
  }

  ~PanelRenderer() {
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
  }

  const std::vector<unsigned char>& Render(std::size_t frame, int fps,
                                           const ScheduleEntry* command,
                                           int clip_count) {
    // Both panels share one EGL/OpenGL context but own different MuJoCo
    // offscreen framebuffers. Rebind this panel's FBO before every draw/read.
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    const auto& joints = clip_.joints.at(frame);
    const auto& position = clip_.positions.at(frame);
    const auto& quaternion = clip_.quaternions.at(frame);
    std::copy(position.begin(), position.end(), data_->qpos);
    std::copy(quaternion.begin(), quaternion.end(), data_->qpos + 3);
    for (std::size_t motor = 0; motor < isaaclab_to_mujoco.size(); ++motor) {
      data_->qpos[7 + motor] = joints[isaaclab_to_mujoco[motor]];
    }
    mj_forward(model_.get(), data_.get());
    for (int axis = 0; axis < 3; ++axis) camera_.lookat[axis] = data_->qpos[axis];
    camera_.lookat[2] = std::max(title_.front() == 'G' ? 0.55 : 0.65,
                                 camera_.lookat[2] -
                                     (title_.front() == 'G' ? 0.15 : 0.25));

    mjv_updateScene(model_.get(), data_.get(), &option_, &perturb_, &camera_,
                    mjCAT_ALL, &scene_);
    AddFallbackFloor(&scene_);
    std::array<float, 4> path_color = tint_;
    path_color[3] = 0.45f;
    AddPath(clip_, frame, &scene_, path_color);
    const mjrRect viewport{0, 0, width_, height_};
    mjr_render(viewport, &scene_, &context_);

    std::ostringstream overlay;
    overlay << title_ << "\nframe " << (frame + 1) << "/"
            << clip_.joints.size() << "  t=" << std::fixed
            << std::setprecision(2) << static_cast<double>(frame) / fps << " s";
    if (command) {
      overlay << "\naction " << command->style << " / " << command->maneuver;
      if (command->clip > 0) {
        overlay << "  segment " << command->clip << "/" << clip_count;
      }
      if (command->speed >= 0.0) {
        overlay << "\nspeed " << std::setprecision(2) << command->speed
                << " m/s";
      }
    }
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                overlay.str().c_str(), "50 Hz kinematic reference",
                &context_);
    mjr_readPixels(bottom_up_.data(), nullptr, viewport, &context_);
    const std::size_t row_bytes = static_cast<std::size_t>(width_) * 3;
    for (int y = 0; y < height_; ++y) {
      std::copy_n(bottom_up_.data() +
                      static_cast<std::size_t>(height_ - 1 - y) * row_bytes,
                  row_bytes,
                  top_down_.data() + static_cast<std::size_t>(y) * row_bytes);
    }
    return top_down_;
  }

 private:
  ModelPtr model_;
  DataPtr data_;
  const Clip& clip_;
  int width_;
  int height_;
  std::string title_;
  std::array<float, 4> tint_;
  mjvCamera camera_{};
  mjvOption option_{};
  mjvPerturb perturb_{};
  mjvScene scene_{};
  mjrContext context_{};
  std::vector<unsigned char> bottom_up_;
  std::vector<unsigned char> top_down_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const Clip g1 = LoadClip(args.g1_clip);
    const Clip h2 = LoadClip(args.h2_clip);
    const std::vector<ScheduleEntry> schedule = LoadSchedule(args.schedule);
    if (g1.joints.size() != h2.joints.size()) {
      throw std::runtime_error("G1 and H2 clips have different frame counts");
    }
    const std::size_t frame_count =
        args.max_frames > 0
            ? std::min<std::size_t>(g1.joints.size(), args.max_frames)
            : g1.joints.size();

    EglContext egl;
    const int panel_width = args.width / 2;
    int clip_count = 0;
    for (const auto& entry : schedule) {
      clip_count = std::max(clip_count, entry.clip);
    }
    PanelRenderer g1_renderer(args.g1_mjcf, g1, panel_width, args.height,
                              "G1 planner", {0.86f, 0.25f, 0.22f, 1.0f});
    PanelRenderer h2_renderer(args.h2_mjcf, h2, panel_width, args.height,
                              "H2 retarget", {0.20f, 0.78f, 0.34f, 1.0f});
    std::vector<unsigned char> frame(
        static_cast<std::size_t>(args.width) * args.height * 3);
    const std::size_t panel_row_bytes = static_cast<std::size_t>(panel_width) * 3;
    const std::size_t output_row_bytes = static_cast<std::size_t>(args.width) * 3;
    std::cerr << "[video-renderer] EGL offscreen " << args.width << "x"
              << args.height << " @ " << args.fps << " fps, " << frame_count
              << " frames\n";
    std::size_t schedule_index = 0;
    for (std::size_t index = 0; index < frame_count; ++index) {
      while (schedule_index + 1 < schedule.size() &&
             index > static_cast<std::size_t>(
                         schedule[schedule_index].end_frame)) {
        ++schedule_index;
      }
      const ScheduleEntry* command =
          !schedule.empty() &&
                  index >= static_cast<std::size_t>(
                               schedule[schedule_index].start_frame) &&
                  index <= static_cast<std::size_t>(
                               schedule[schedule_index].end_frame)
              ? &schedule[schedule_index]
              : nullptr;
      const auto& left =
          g1_renderer.Render(index, args.fps, command, clip_count);
      const auto& right =
          h2_renderer.Render(index, args.fps, command, clip_count);
      for (int y = 0; y < args.height; ++y) {
        const std::size_t panel_offset = static_cast<std::size_t>(y) * panel_row_bytes;
        const std::size_t output_offset = static_cast<std::size_t>(y) * output_row_bytes;
        std::copy_n(left.data() + panel_offset, panel_row_bytes,
                    frame.data() + output_offset);
        std::copy_n(right.data() + panel_offset, panel_row_bytes,
                    frame.data() + output_offset + panel_row_bytes);
      }
      std::cout.write(reinterpret_cast<const char*>(frame.data()), frame.size());
      if (!std::cout) throw std::runtime_error("failed to write RGB frame stream");
    }
    std::cout.flush();
    std::cerr << "[video-renderer] rendered " << frame_count << " frames\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[video-renderer] ERROR: " << error.what() << '\n';
    return 1;
  }
}
