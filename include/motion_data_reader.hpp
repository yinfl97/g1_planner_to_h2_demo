/**
 * @file motion_data_reader.hpp
 * @brief Motion data structures, CSV reader, and motion recorder.
 *
 * This file provides three main components:
 *
 * ## MotionSequence
 * Flat-array storage for a single motion clip: joint positions/velocities,
 * body positions/quaternions/velocities, and optional SMPL joint/pose data.
 * Frame accessors (e.g. `JointPositions(frame)`) return pointers into the
 * flat arrays at the correct stride.  The struct also supports forward
 * kinematics computation and Gaussian-filtered velocity estimation.
 *
 * ## MotionDataReader
 * Reads a directory of motion folders, each containing CSV files
 * (`joint_pos.csv`, `body_quat.csv`, etc.) and a `metadata.txt` file.
 * Auto-discovers available motions, validates frame-count consistency,
 * and returns a vector of shared_ptr<MotionSequence>.
 *
 * ## MotionRecorder
 * RAII-based writer that records live motion frames into timestamped
 * directories of CSV files (used for ZMQ stream recording).
 *
 * ## PlannerMotionState
 * A simple struct bundling a MotionSequence with a frame cursor, used
 * for thread-safe planner ↔ control-loop communication.
 */

#ifndef MOTION_DATA_READER_HPP
#define MOTION_DATA_READER_HPP

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <regex>

#include "../include/fk.hpp"
#include "../include/policy_parameters.hpp"
#include "../include/math_utils.hpp"
#include "io_debug_log/debug_log.hpp"

/**
 * @struct MotionSequence
 * @brief Stores one motion clip as flat arrays of per-frame data.
 *
 * Data layout:  each "row" in the flat arrays corresponds to one frame.
 * Frame accessors multiply the frame index by the per-frame stride
 * (e.g. `num_joints`) to yield a pointer to that frame's data.
 *
 * The `encode_mode` field is mutable so it can be toggled at runtime
 * (e.g. by keyboard 'Z' key) without requiring non-const access to
 * the MotionSequence.
 */
struct MotionSequence {
    std::string name = "";    ///< Human-readable motion name (folder name or "streamed" / "planner_motion").
    int timesteps = 0;        ///< Total number of frames in this motion.

    /// Encoding mode flag (mutable for runtime toggling from const contexts).
    ///  -2 = no token state configured
    ///  -1 = token state needed but encoder not loaded
    ///   0+ = active encoder mode index
    mutable int encode_mode = -2;

    using Point = std::array<double, 3>;
    using Quaternion = std::array<double, 4>;
    using Velocity = std::array<double, 3>;

    double *JointPositions(int frame) { return joint_positions_.data() + frame * num_joints; }
    double *JointVelocities(int frame) { return joint_velocities_.data() + frame * num_joints; }
    Point *BodyPositions(int frame) { return body_positions_.data() + frame * num_bodies; }
    Quaternion *BodyQuaternions(int frame) { return body_quaternions_.data() + frame * num_body_quaternions; }
    Velocity *BodyLinVelocities(int frame) { return body_lin_velocities_.data() + frame * num_bodies; }
    Velocity *BodyAngVelocities(int frame) { return body_ang_velocities_.data() + frame * num_bodies; }
    Point *SmplJoints(int frame) { return smpl_joints_.data() + frame * num_smpl_joints; }
    Point *SmplPoses(int frame) { return smpl_poses_.data() + frame * num_smpl_poses; }

    const double *JointPositions(int frame) const { return joint_positions_.data() + frame * num_joints; }
    const double *JointVelocities(int frame) const { return joint_velocities_.data() + frame * num_joints; }
    const Point *BodyPositions(int frame) const { return body_positions_.data() + frame * num_bodies; }
    const Quaternion *BodyQuaternions(int frame) const { return body_quaternions_.data() + frame * num_body_quaternions; }
    const Velocity *BodyLinVelocities(int frame) const { return body_lin_velocities_.data() + frame * num_bodies; }
    const Velocity *BodyAngVelocities(int frame) const { return body_ang_velocities_.data() + frame * num_bodies; }
    const Point *SmplJoints(int frame) const { return smpl_joints_.data() + frame * num_smpl_joints; }
    const Point *SmplPoses(int frame) const { return smpl_poses_.data() + frame * num_smpl_poses; }

    int GetNumJoints() const { return num_joints; }
    int GetNumBodies() const { return num_bodies; }
    int GetNumBodyQuaternions() const { return num_body_quaternions; }
    int GetNumSmplJoints() const { return num_smpl_joints; }
    int GetNumSmplPoses() const { return num_smpl_poses; }
    const std::vector<int> &BodyPartIndexes() const { return body_part_indexes; }

    // Encode mode accessors (const methods that modify mutable encode_mode)
    int GetEncodeMode() const { return encode_mode; }
    void SetEncodeMode(int mode) const { encode_mode = mode; }

    // Set body part indexes (used by ZMQ interface for streamed motions)
    void SetBodyPartIndexes(const std::vector<int>& indexes) { body_part_indexes = indexes; }

    // Pre-allocate capacity to avoid reallocations during extension
    void ReserveCapacity(int max_frames, int joints = 29, int bodies = 1,
                        int body_quaternions = 1, int smpl_joints = 0, int smpl_poses = 0) {
      num_joints = joints;
      num_bodies = bodies;
      num_body_quaternions = body_quaternions;
      num_smpl_joints = smpl_joints;
      num_smpl_poses = smpl_poses;

      joint_positions_.resize(max_frames * joints);
      joint_velocities_.resize(max_frames * joints);

      body_positions_.resize(max_frames * bodies);
      body_quaternions_.resize(max_frames * body_quaternions);
      body_lin_velocities_.resize(max_frames * bodies);
      body_ang_velocities_.resize(max_frames * bodies);

      smpl_joints_.resize(max_frames * smpl_joints);
      smpl_poses_.resize(max_frames * smpl_poses);

      positions_world_tmp.resize(joints + 1);
      rotations_world_tmp.resize(joints + 1);
      pos_filter_tmp.resize(max_frames);
      ang_filter_tmp.resize(max_frames);
    }

    void ComputeFK(const RobotFK &fk)
    {
      if(positions_world_tmp.size() != positions_world_tmp.size())
      {
        ioai::debug::log("MotionData").error("failed to compute fk - wrong number of joints");
        ioai::debug::log("MotionData").error("{} != {}", positions_world_tmp.size(), positions_world_tmp.size());
        return;
      }

      int nframes = timesteps;
      auto nj = GetNumJoints();
      for( int f=0; f < timesteps; ++f )
      {
        fk.DoFK(
          positions_world_tmp.data(),
          rotations_world_tmp.data(),
          BodyPositions(f)[0],
          BodyQuaternions(f)[0],
          JointPositions(f)
        );

        for(size_t body_idx = 0; body_idx < body_part_indexes.size(); ++body_idx)
        {
          auto i = body_part_indexes[body_idx];
          int mj_idx = 0;
          if(i > 0)
          {
            mj_idx = mujoco_to_isaaclab[i-1] + 1;
          }
          auto &p = positions_world_tmp[mj_idx];
          auto &q = rotations_world_tmp[mj_idx];

          BodyPositions(f)[body_idx] = p;
          BodyQuaternions(f)[body_idx] = q;
        }
      }
    }

    void ComputeGlobalVelocities(bool filter=true)
    {
      for( int f=0; f < timesteps; ++f )
      {
        int f0 = std::max(f - 1, 0);
        int f1 = std::min(timesteps-1, f+1);

        int dt = std::max(1, f1 - f0);
        for(size_t body_idx = 0; body_idx < body_part_indexes.size(); ++body_idx)
        {
          const auto &p1 = BodyPositions(f1)[body_idx];
          const auto &p0 = BodyPositions(f0)[body_idx];

          BodyLinVelocities(f)[body_idx] = {
            (p1[0] - p0[0]) * 50 / dt,
            (p1[1] - p0[1]) * 50 / dt,
            (p1[2] - p0[2]) * 50 / dt
          };

          const auto &q1 = BodyQuaternions(f1)[body_idx];
          const auto &q0 = BodyQuaternions(std::max(0, f1-1))[body_idx];
          auto dq = quat_mul(q1, quat_conjugate_d(q0));
          auto [diff_angle, diff_axis] = quat_to_angle_axis(dq);

          BodyAngVelocities(f)[body_idx] = {
            diff_axis[0] * diff_angle * 50,
            diff_axis[1] * diff_angle * 50,
            diff_axis[2] * diff_angle * 50
          };
        }
      }

      if(filter)
      {
        // trying to replicate scipy.ndimage.filters.gaussian_filter1d(sigma=2, mode="nearest")
        // like in the gr00t code:
        const double sigma = 2.0;
        const int lw = int(4.0 * sigma + 0.5);
        const double sigma2 = sigma * sigma;
        double kernel_sum = 0.0;
        for(int dx = -lw; dx <= lw; ++dx)
        {
          kernel_sum += exp(-0.5 / sigma2 * dx * dx);
        }
        if(pos_filter_tmp.size() < timesteps || ang_filter_tmp.size() < timesteps)
        {
          ioai::debug::log("MotionData").error("failed to filter velocities because buffer size is inadequate");
          return;
        }
        for(size_t body_idx = 0; body_idx < body_part_indexes.size(); ++body_idx)
        {
          for( int f=0; f < timesteps; ++f )
          {
            std::array<double, 3> pos_result = {0.0, 0.0, 0.0};
            std::array<double, 3> ang_result = {0.0, 0.0, 0.0};
            for(int dx = -lw; dx <= lw; ++dx)
            {
              auto fk = std::clamp(f + dx, 0, timesteps-1);
              double weight = exp(-0.5 / sigma2 * dx * dx);
              pos_result[0] += weight * BodyLinVelocities(fk)[body_idx][0];
              pos_result[1] += weight * BodyLinVelocities(fk)[body_idx][1];
              pos_result[2] += weight * BodyLinVelocities(fk)[body_idx][2];

              ang_result[0] += weight * BodyAngVelocities(fk)[body_idx][0];
              ang_result[1] += weight * BodyAngVelocities(fk)[body_idx][1];
              ang_result[2] += weight * BodyAngVelocities(fk)[body_idx][2];
            }
            pos_filter_tmp[f][0] = pos_result[0] / kernel_sum;
            pos_filter_tmp[f][1] = pos_result[1] / kernel_sum;
            pos_filter_tmp[f][2] = pos_result[2] / kernel_sum;

            ang_filter_tmp[f][0] = ang_result[0] / kernel_sum;
            ang_filter_tmp[f][1] = ang_result[1] / kernel_sum;
            ang_filter_tmp[f][2] = ang_result[2] / kernel_sum;
          }
          for( int f=0; f < timesteps; ++f )
          {
            BodyLinVelocities(f)[body_idx] = pos_filter_tmp[f];
            BodyAngVelocities(f)[body_idx] = ang_filter_tmp[f];
          }
        }
      }
    }

    private:

    friend class MotionDataReader;

    int num_joints = 0;
    int num_bodies = 0; // Used by BodyPositions, BodyLinVelocities, BodyAngVelocities
    int num_body_quaternions = 0; // Independent body count for BodyQuaternions
    int num_smpl_joints = 0;
    int num_smpl_poses = 0;

    // Body part mapping (crucial for alignment!)
    std::vector<int> body_part_indexes; // Maps 14 body parts to body indices

    // Joint data (timesteps x joints) - using double precision
    std::vector<double> joint_positions_; // [timestep][joint_id]
    std::vector<double> joint_velocities_; // [timestep][joint_id]

    // Body data (timesteps x body_parts x coordinates) - using double precision

    std::vector<Point> body_positions_; // [timestep][body_id][xyz]
    std::vector<Quaternion> body_quaternions_; // [timestep][body_id][wxyz]
    std::vector<Velocity> body_lin_velocities_; // [timestep][body_id][xyz]
    std::vector<Velocity> body_ang_velocities_; // [timestep][body_id][xyz]

    // SMPL data (separate from robot body positions/quaternions)
    std::vector<Point> smpl_joints_; // [timestep][smpl_joint_id][xyz] - SMPL joint positions
    std::vector<Point> smpl_poses_; // [timestep][smpl_pose_id][xyz] - SMPL body poses

    // temporary buffers for doing FK and filtering velocities,
    // to avoid allocating memory at runtime:
    std::vector<Point> positions_world_tmp;
    std::vector<Quaternion> rotations_world_tmp;
    std::vector<Velocity> pos_filter_tmp;
    std::vector<Velocity> ang_filter_tmp;

    // ==== PERFORMANCE OPTIMIZATIONS FOR SLIDING WINDOW OPERATIONS ====

    // ==== ORIGINAL METHODS ====

};

/**
 * @brief Bundles a MotionSequence with its playback cursor for thread-safe
 *        transfer between the planner and control threads via DataBuffer.
 */
struct PlannerMotionState {
  std::shared_ptr<const MotionSequence> motion;
  int current_frame;

  PlannerMotionState(std::shared_ptr<const MotionSequence> m = std::make_shared<const MotionSequence>(), int frame = 0)
      : motion(m), current_frame(frame) {}

  // Constructor for backwards compatibility
  PlannerMotionState(const MotionSequence& m, int frame = 0)
      : motion(std::make_shared<const MotionSequence>(m)), current_frame(frame) {}
};

/**
 * @class MotionRecorder
 * @brief RAII-based CSV writer for recording live motion frames to disk.
 *
 * Creates a timestamped directory under `base_dir` and writes one CSV file
 * per signal type (joint_pos, body_quat, etc.).  Call `StartSession()` to
 * begin, `WriteFrame()` each tick, and `Finalize()` (or let the destructor
 * do it) to close files and write metadata.
 */
class MotionRecorder {
public:
  MotionRecorder(const std::string& base_dir = "reference/recorded_motion")
    : base_dir_(base_dir), active_(false), frames_written_(0) {}

  ~MotionRecorder() {
    if (active_) {
      Finalize();
    }
  }

  // Start recording - creates session directory and opens CSV files
  // Directory structure: base_dir/YYYYMMDD/motion_name_HHMMSS/
  bool StartSession(const std::string& motion_name = "zmq_stream") {
    if (active_) {
      ioai::debug::log("MotionData").error("session already active, stopping previous session");
      Finalize();
    }

    try {
      auto now = std::chrono::system_clock::now();
      std::time_t now_c = std::chrono::system_clock::to_time_t(now);
      std::tm tm{};
#ifdef _WIN32
      localtime_s(&tm, &now_c);
#else
      localtime_r(&now_c, &tm);
#endif

      // Create date directory: YYYYMMDD
      std::ostringstream date_oss;
      date_oss << std::put_time(&tm, "%Y%m%d");
      std::string date_dir = base_dir_ + "/" + date_oss.str();
      std::filesystem::create_directories(date_dir);

      // Create motion directory with timestamp: motion_name_HHMMSS
      std::ostringstream time_oss;
      time_oss << std::put_time(&tm, "%H%M%S");
      motion_dir_ = date_dir + "/" + motion_name + "_" + time_oss.str();
      std::filesystem::create_directories(motion_dir_);

      motion_name_ = motion_name;
      frames_written_ = 0;
      active_ = true;

      ioai::debug::log("MotionData").info("recording started: {}", motion_dir_);
      return true;
    } catch (const std::exception& e) {
      ioai::debug::log("MotionData").error("failed to start session: {}", e.what());
      return false;
    }
  }

  // Write a single frame of motion data
  bool WriteFrame(std::shared_ptr<const MotionSequence> motion, int frame_idx) {
    if (!active_ || !motion || frame_idx >= motion->timesteps) {
      return false;
    }

    // Lazily open files on first write when we know dimensions
    if (!files_open_) {
      if (!OpenFiles(motion->GetNumJoints(), motion->GetNumBodies(), motion->GetNumBodyQuaternions(),
                     motion->GetNumSmplJoints(), motion->GetNumSmplPoses())) {
        return false;
      }
    }

    try {
      // Write joint positions
      const double* joint_pos = motion->JointPositions(frame_idx);
      for (int j = 0; j < motion->GetNumJoints(); ++j) {
        joint_pos_file_ << std::fixed << std::setprecision(9) << joint_pos[j];
        if (j + 1 < motion->GetNumJoints()) joint_pos_file_ << ",";
      }
      joint_pos_file_ << "\n";

      // Write joint velocities
      const double* joint_vel = motion->JointVelocities(frame_idx);
      for (int j = 0; j < motion->GetNumJoints(); ++j) {
        joint_vel_file_ << std::fixed << std::setprecision(9) << joint_vel[j];
        if (j + 1 < motion->GetNumJoints()) joint_vel_file_ << ",";
      }
      joint_vel_file_ << "\n";

      // Write body positions (for each body: x, y, z)
      const auto* body_pos = motion->BodyPositions(frame_idx);
      for (int b = 0; b < motion->GetNumBodies(); ++b) {
        for (int axis = 0; axis < 3; ++axis) {
          body_pos_file_ << std::fixed << std::setprecision(9) << body_pos[b][axis];
          if (b + 1 < motion->GetNumBodies() || axis + 1 < 3) body_pos_file_ << ",";
        }
      }
      body_pos_file_ << "\n";

      // Write body quaternions (for each body: w, x, y, z)
      const auto* body_quat = motion->BodyQuaternions(frame_idx);
      for (int b = 0; b < motion->GetNumBodyQuaternions(); ++b) {
        for (int q = 0; q < 4; ++q) {
          body_quat_file_ << std::fixed << std::setprecision(9) << body_quat[b][q];
          if (b + 1 < motion->GetNumBodyQuaternions() || q + 1 < 4) body_quat_file_ << ",";
        }
      }
      body_quat_file_ << "\n";

      // Write body linear velocities
      const auto* body_lin_vel = motion->BodyLinVelocities(frame_idx);
      for (int b = 0; b < motion->GetNumBodies(); ++b) {
        for (int axis = 0; axis < 3; ++axis) {
          body_lin_vel_file_ << std::fixed << std::setprecision(9) << body_lin_vel[b][axis];
          if (b + 1 < motion->GetNumBodies() || axis + 1 < 3) body_lin_vel_file_ << ",";
        }
      }
      body_lin_vel_file_ << "\n";

      // Write body angular velocities
      const auto* body_ang_vel = motion->BodyAngVelocities(frame_idx);
      for (int b = 0; b < motion->GetNumBodies(); ++b) {
        for (int axis = 0; axis < 3; ++axis) {
          body_ang_vel_file_ << std::fixed << std::setprecision(9) << body_ang_vel[b][axis];
          if (b + 1 < motion->GetNumBodies() || axis + 1 < 3) body_ang_vel_file_ << ",";
        }
      }
      body_ang_vel_file_ << "\n";

      // Write SMPL joints if present
      if (motion->GetNumSmplJoints() > 0 && smpl_joints_file_.is_open()) {
        const auto* smpl_joints = motion->SmplJoints(frame_idx);
        for (int b = 0; b < motion->GetNumSmplJoints(); ++b) {
          for (int axis = 0; axis < 3; ++axis) {
            smpl_joints_file_ << std::fixed << std::setprecision(9) << smpl_joints[b][axis];
            if (b + 1 < motion->GetNumSmplJoints() || axis + 1 < 3) smpl_joints_file_ << ",";
          }
        }
        smpl_joints_file_ << "\n";
      }

      // Write SMPL poses if present
      if (motion->GetNumSmplPoses() > 0 && smpl_poses_file_.is_open()) {
        const auto* smpl_poses = motion->SmplPoses(frame_idx);
        for (int p = 0; p < motion->GetNumSmplPoses(); ++p) {
          for (int axis = 0; axis < 3; ++axis) {
            smpl_poses_file_ << std::fixed << std::setprecision(9) << smpl_poses[p][axis];
            if (p + 1 < motion->GetNumSmplPoses() || axis + 1 < 3) smpl_poses_file_ << ",";
          }
        }
        smpl_poses_file_ << "\n";
      }

      frames_written_++;
      return true;
    } catch (const std::exception& e) {
      ioai::debug::log("MotionData").error("failed to write frame: {}", e.what());
      return false;
    }
  }

  // Finalize recording - closes files and writes metadata
  void Finalize() {
    if (!active_) return;

    try {
      // Close all open files
      if (joint_pos_file_.is_open()) joint_pos_file_.close();
      if (joint_vel_file_.is_open()) joint_vel_file_.close();
      if (body_pos_file_.is_open()) body_pos_file_.close();
      if (body_quat_file_.is_open()) body_quat_file_.close();
      if (body_lin_vel_file_.is_open()) body_lin_vel_file_.close();
      if (body_ang_vel_file_.is_open()) body_ang_vel_file_.close();
      if (smpl_joints_file_.is_open()) smpl_joints_file_.close();
      if (smpl_poses_file_.is_open()) smpl_poses_file_.close();

      // Write metadata file
      std::ofstream meta(motion_dir_ + "/metadata.txt");
      if (meta.is_open()) {
        meta << "Metadata for: " << motion_name_ << "\n";
        meta << std::string(30, '=') << "\n\n";
        meta << "Body part indexes:\n";
        meta << "[0]\n\n";  // Single body for ZMQ stream
        meta << "Total timesteps: " << frames_written_ << "\n";
        meta.close();
      }

      ioai::debug::log("MotionData").info("recording saved: {} frames at '{}'", frames_written_, motion_dir_);
    } catch (const std::exception& e) {
      ioai::debug::log("MotionData").error("failed to finalize: {}", e.what());
    }

    active_ = false;
    files_open_ = false;
    frames_written_ = 0;
  }

  bool IsActive() const { return active_; }
  int GetFramesWritten() const { return frames_written_; }

private:
  bool OpenFiles(int num_joints, int num_bodies, int num_body_quaternions,
                 int num_smpl_joints, int num_smpl_poses) {
    try {
      // Open joint position file with header
      joint_pos_file_.open(motion_dir_ + "/joint_pos.csv");
      if (!joint_pos_file_.is_open()) return false;
      for (int j = 0; j < num_joints; ++j) {
        joint_pos_file_ << "joint_" << j;
        if (j + 1 < num_joints) joint_pos_file_ << ",";
      }
      joint_pos_file_ << "\n";

      // Open joint velocity file with header
      joint_vel_file_.open(motion_dir_ + "/joint_vel.csv");
      if (!joint_vel_file_.is_open()) return false;
      for (int j = 0; j < num_joints; ++j) {
        joint_vel_file_ << "joint_vel_" << j;
        if (j + 1 < num_joints) joint_vel_file_ << ",";
      }
      joint_vel_file_ << "\n";

      // Open body position file with header
      body_pos_file_.open(motion_dir_ + "/body_pos.csv");
      if (!body_pos_file_.is_open()) return false;
      for (int b = 0; b < num_bodies; ++b) {
        body_pos_file_ << "body_" << b << "_x,body_" << b << "_y,body_" << b << "_z";
        if (b + 1 < num_bodies) body_pos_file_ << ",";
      }
      body_pos_file_ << "\n";

      // Open body quaternion file with header
      body_quat_file_.open(motion_dir_ + "/body_quat.csv");
      if (!body_quat_file_.is_open()) return false;
      for (int b = 0; b < num_body_quaternions; ++b) {
        body_quat_file_ << "body_" << b << "_w,body_" << b << "_x,body_" << b << "_y,body_" << b << "_z";
        if (b + 1 < num_body_quaternions) body_quat_file_ << ",";
      }
      body_quat_file_ << "\n";

      // Open body linear velocity file with header
      body_lin_vel_file_.open(motion_dir_ + "/body_lin_vel.csv");
      if (!body_lin_vel_file_.is_open()) return false;
      for (int b = 0; b < num_bodies; ++b) {
        body_lin_vel_file_ << "body_" << b << "_vel_x,body_" << b << "_vel_y,body_" << b << "_vel_z";
        if (b + 1 < num_bodies) body_lin_vel_file_ << ",";
      }
      body_lin_vel_file_ << "\n";

      // Open body angular velocity file with header
      body_ang_vel_file_.open(motion_dir_ + "/body_ang_vel.csv");
      if (!body_ang_vel_file_.is_open()) return false;
      for (int b = 0; b < num_bodies; ++b) {
        body_ang_vel_file_ << "body_" << b << "_angvel_x,body_" << b << "_angvel_y,body_" << b << "_angvel_z";
        if (b + 1 < num_bodies) body_ang_vel_file_ << ",";
      }
      body_ang_vel_file_ << "\n";

      // Open SMPL joints file if present
      if (num_smpl_joints > 0) {
        smpl_joints_file_.open(motion_dir_ + "/smpl_joint.csv");
        if (smpl_joints_file_.is_open()) {
          for (int b = 0; b < num_smpl_joints; ++b) {
            smpl_joints_file_ << "smpl_joint_" << b << "_x,smpl_joint_" << b << "_y,smpl_joint_" << b << "_z";
            if (b + 1 < num_smpl_joints) smpl_joints_file_ << ",";
          }
          smpl_joints_file_ << "\n";
        }
      }

      // Open SMPL poses file if present
      if (num_smpl_poses > 0) {
        smpl_poses_file_.open(motion_dir_ + "/smpl_pose.csv");
        if (smpl_poses_file_.is_open()) {
          for (int p = 0; p < num_smpl_poses; ++p) {
            smpl_poses_file_ << "smpl_pose_" << p << "_x,smpl_pose_" << p << "_y,smpl_pose_" << p << "_z";
            if (p + 1 < num_smpl_poses) smpl_poses_file_ << ",";
          }
          smpl_poses_file_ << "\n";
        }
      }

      files_open_ = true;
      return true;
    } catch (const std::exception& e) {
      ioai::debug::log("MotionData").error("failed to open files: {}", e.what());
      return false;
    }
  }

  std::string base_dir_;
  std::string motion_dir_;
  std::string motion_name_;
  bool active_;
  bool files_open_ = false;
  int frames_written_;

  std::ofstream joint_pos_file_;
  std::ofstream joint_vel_file_;
  std::ofstream body_pos_file_;
  std::ofstream body_quat_file_;
  std::ofstream body_lin_vel_file_;
  std::ofstream body_ang_vel_file_;
  std::ofstream smpl_joints_file_;
  std::ofstream smpl_poses_file_;
};

/// Write a MotionSequence as a keyboard-playable reference clip
/// (`joint_pos.csv` IsaacLab order, 50 Hz). Overwrites `clip_dir`.
inline bool WriteMotionSequenceAsReferenceClip(
    const MotionSequence& motion,
    const std::string& clip_dir,
    const std::string& motion_name = "h2_g1_planner_retarget",
    const std::string& source = "G1 planner -> H2 retarget") {
  if (motion.timesteps <= 0 || motion.GetNumJoints() <= 0) {
    return false;
  }
  try {
    std::filesystem::create_directories(clip_dir);
    const int n = motion.timesteps;
    const int nj = motion.GetNumJoints();
    const int nb = std::max(motion.GetNumBodies(), 1);
    const int nq = std::max(motion.GetNumBodyQuaternions(), 1);

    auto csv_header = [](std::ofstream& f, const std::string& prefix, int count) {
      for (int i = 0; i < count; ++i) {
        f << prefix << i;
        if (i + 1 < count) f << ",";
      }
      f << "\n";
    };

    std::ofstream jp(clip_dir + "/joint_pos.csv");
    std::ofstream jv(clip_dir + "/joint_vel.csv");
    std::ofstream bp(clip_dir + "/body_pos.csv");
    std::ofstream bq(clip_dir + "/body_quat.csv");
    std::ofstream bl(clip_dir + "/body_lin_vel.csv");
    if (!jp || !jv || !bp || !bq || !bl) {
      return false;
    }
    csv_header(jp, "joint_", nj);
    csv_header(jv, "joint_vel_", nj);
    bp << "body_0_x,body_0_y,body_0_z\n";
    bq << "body_0_w,body_0_x,body_0_y,body_0_z\n";
    bl << "body_0_x,body_0_y,body_0_z\n";

    for (int f = 0; f < n; ++f) {
      const double* q = motion.JointPositions(f);
      const double* qd = motion.JointVelocities(f);
      for (int j = 0; j < nj; ++j) {
        jp << std::fixed << std::setprecision(9) << q[j];
        jv << std::fixed << std::setprecision(9) << qd[j];
        if (j + 1 < nj) {
          jp << ",";
          jv << ",";
        }
      }
      jp << "\n";
      jv << "\n";

      const auto& pos = motion.BodyPositions(f)[0];
      bp << std::fixed << std::setprecision(9) << pos[0] << "," << pos[1] << "," << pos[2] << "\n";
      const auto& quat = motion.BodyQuaternions(f)[0];
      bq << std::fixed << std::setprecision(9) << quat[0] << "," << quat[1] << ","
         << quat[2] << "," << quat[3] << "\n";

      const int f1 = (f + 1 < n) ? f + 1 : f;
      const auto& pos1 = motion.BodyPositions(f1)[0];
      const double scale = (f1 > f) ? 50.0 : 0.0;
      double vx = (pos1[0] - pos[0]) * scale;
      double vy = (pos1[1] - pos[1]) * scale;
      double vz = (pos1[2] - pos[2]) * scale;
      if (f1 == f && f > 0) {
        const auto& prev = motion.BodyPositions(f - 1)[0];
        vx = (pos[0] - prev[0]) * 50.0;
        vy = (pos[1] - prev[1]) * 50.0;
        vz = (pos[2] - prev[2]) * 50.0;
      }
      bl << std::fixed << std::setprecision(9) << vx << "," << vy << "," << vz << "\n";
    }

    std::ofstream meta(clip_dir + "/metadata.txt");
    meta << "Metadata for: " << motion_name << "\n";
    meta << std::string(30, '=') << "\n\n";
    meta << "Body part indexes:\n[0]\n\n";
    meta << "Total timesteps: " << n << "\n";
    meta << "fps: 50.0\n";
    meta << "source: " << source << " (IsaacLab joint order)\n";
    meta << "num_joints: " << nj << " num_bodies: " << nb << " num_body_quat: " << nq << "\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[WriteMotionSequenceAsReferenceClip] " << e.what() << std::endl;
    return false;
  }
}

/**
 * @class MotionDataReader
 * @brief Discovers and loads motion clips from a directory of CSV files.
 *
 * Each sub-directory under the base directory represents one motion and may
 * contain: `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`,
 * `body_lin_vel.csv`, `body_ang_vel.csv`, `smpl_joint.csv`, `smpl_pose.csv`,
 * and `metadata.txt`.
 */
class MotionDataReader {
  public:
    std::vector<std::shared_ptr<MotionSequence>> motions;  ///< All loaded motion clips.
    int current_motion_index_ = 0;  ///< Index of the currently-selected motion.

    /// Load all motion sub-directories from @p base_directory.
    /// @return True if at least one motion was loaded successfully.
    bool ReadFromCSV(const std::string& base_directory) {
      ioai::debug::log("MotionData").info("reading motion data from CSV files in: {}", base_directory);

      // Auto-discover motion folders
      std::vector<std::string> motion_names;
      try {
        for (const auto& entry : std::filesystem::directory_iterator(base_directory)) {
          if (entry.is_directory()) {
            std::string folder_name = entry.path().filename().string();
            // Skip if it looks like a summary file or hidden folder
            if (folder_name[0] != '.' && folder_name != "motion_summary.txt") { motion_names.push_back(folder_name); }
          }
        }
      } catch (const std::exception& e) {
        ioai::debug::log("MotionData").error("error reading directory: {}", e.what());
        return false;
      }

      ioai::debug::log("MotionData").info("found {} motion folders", motion_names.size());

      for (const auto& motion_name : motion_names) {
        // Create motion as shared_ptr for proper ownership
        auto motion = std::make_shared<MotionSequence>();

        std::string motion_dir = base_directory + "/" + motion_name;

        // Read metadata first (contains body part indexes!)
        if (!ReadMetadata(motion_dir + "/metadata.txt", *motion)) {
          ioai::debug::log("MotionData").warn("could not read metadata for {}", motion_name);
        }

        // Collect missing file warnings to decide later which ones to show
        std::vector<std::string> missing_warnings;

        // Read joint data (optional)
        int num_frames = 0;
        int num_joints_pos = 0;
        int num_joints_vel = 0;

        // Read joint positions (optional)
        if (std::filesystem::exists(motion_dir + "/joint_pos.csv")) {
          int pos_frames = ReadCSV(motion_dir + "/joint_pos.csv", motion->joint_positions_);
          if (pos_frames > 0) {
            if (num_frames == 0) {
              num_frames = pos_frames;
            } else if (num_frames != pos_frames) {
              ioai::debug::log("MotionData").error("frame count mismatch in joint_pos.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, pos_frames);
              continue;
            }
            num_joints_pos = motion->joint_positions_.size() / pos_frames;
          }
        } else {
          missing_warnings.push_back("joint_pos.csv");
        }

        // Read joint velocities (optional)
        if (std::filesystem::exists(motion_dir + "/joint_vel.csv")) {
          int vel_frames = ReadCSV(motion_dir + "/joint_vel.csv", motion->joint_velocities_);
          if (vel_frames > 0) {
            if (num_frames == 0) {
              num_frames = vel_frames;
            } else if (num_frames != vel_frames) {
              ioai::debug::log("MotionData").error("frame count mismatch in joint_vel.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, vel_frames);
              continue;
            }
            num_joints_vel = motion->joint_velocities_.size() / vel_frames;
          }
        } else {
          missing_warnings.push_back("joint_vel.csv");
        }

        // Validate joint counts match if both are present
        if (num_joints_pos > 0 && num_joints_vel > 0 && num_joints_pos != num_joints_vel) {
          ioai::debug::log("MotionData").error("number of joints inconsistent between joint_pos.csv ({}) and joint_vel.csv ({}) for {} - SKIPPING MOTION", num_joints_pos, num_joints_vel, motion_name);
          continue;
        }

        // Use whichever joint count is available
        int num_joints = (num_joints_pos > 0) ? num_joints_pos : num_joints_vel;

        // Read body data (optional)
        int num_bodies = 0;
        int num_body_quaternions = 0;

        if (num_frames == 0 && std::filesystem::exists(motion_dir + "/body_pos.csv")) {
          num_frames = ReadCSV3D<3>(motion_dir + "/body_pos.csv", motion->body_positions_);
        } else if (std::filesystem::exists(motion_dir + "/body_pos.csv")) {
          int body_frames = ReadCSV3D<3>(motion_dir + "/body_pos.csv", motion->body_positions_);
          if (num_frames != body_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in body_pos.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, body_frames);
            continue;
          }
        } else {
          missing_warnings.push_back("body_pos.csv");
        }

        if (num_frames > 0 && !motion->body_positions_.empty()) {
          num_bodies = motion->body_positions_.size() / num_frames;
        }

        if (std::filesystem::exists(motion_dir + "/body_quat.csv")) {
          int quat_frames = ReadCSV3D<4>(motion_dir + "/body_quat.csv", motion->body_quaternions_);
          if (num_frames == 0) {
            num_frames = quat_frames;
          } else if (num_frames != quat_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in body_quat.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, quat_frames);
            continue;
          }
          if (num_frames > 0) {
            num_body_quaternions = motion->body_quaternions_.size() / num_frames;
          }
        } else {
          missing_warnings.push_back("body_quat.csv");
        }

        // Read body velocities (optional, must match body_pos count)
        if (std::filesystem::exists(motion_dir + "/body_lin_vel.csv")) {
          int vel_frames = ReadCSV3D<3>(motion_dir + "/body_lin_vel.csv", motion->body_lin_velocities_);
          if (num_frames != vel_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in body_lin_vel.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, vel_frames);
            continue;
          }
          if (motion->body_lin_velocities_.size() != num_frames * num_bodies) {
            ioai::debug::log("MotionData").error("body count mismatch between body_pos.csv and body_lin_vel.csv for {} - SKIPPING MOTION", motion_name);
            continue;
          }
        } else {
          missing_warnings.push_back("body_lin_vel.csv");
        }

        if (std::filesystem::exists(motion_dir + "/body_ang_vel.csv")) {
          int ang_vel_frames = ReadCSV3D<3>(motion_dir + "/body_ang_vel.csv", motion->body_ang_velocities_);
          if (num_frames != ang_vel_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in body_ang_vel.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, ang_vel_frames);
            continue;
          }
          if (motion->body_ang_velocities_.size() != num_frames * num_bodies) {
            ioai::debug::log("MotionData").error("body count mismatch between body_pos.csv and body_ang_vel.csv for {} - SKIPPING MOTION", motion_name);
            continue;
          }
        } else {
          missing_warnings.push_back("body_ang_vel.csv");
        }

        // Read SMPL data (optional)
        int num_smpl_joints = 0;
        int num_smpl_poses = 0;

        if (std::filesystem::exists(motion_dir + "/smpl_joint.csv")) {
          int smpl_frames = ReadCSV3D<3>(motion_dir + "/smpl_joint.csv", motion->smpl_joints_);
          if (num_frames > 0 && num_frames != smpl_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in smpl_joint.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, smpl_frames);
            continue;
          } else if (num_frames == 0) {
            num_frames = smpl_frames;
          }
          if (smpl_frames > 0) {
            num_smpl_joints = motion->smpl_joints_.size() / smpl_frames;
          }
        } else {
          missing_warnings.push_back("smpl_joint.csv");
        }

        if (std::filesystem::exists(motion_dir + "/smpl_pose.csv")) {
          int smpl_pose_frames = ReadCSV3D<3>(motion_dir + "/smpl_pose.csv", motion->smpl_poses_);
          if (num_frames > 0 && num_frames != smpl_pose_frames) {
            ioai::debug::log("MotionData").error("frame count mismatch in smpl_pose.csv for {} (expected {}, got {}) - SKIPPING MOTION", motion_name, num_frames, smpl_pose_frames);
            continue;
          } else if (num_frames == 0) {
            num_frames = smpl_pose_frames;
          }
          if (smpl_pose_frames > 0) {
            num_smpl_poses = motion->smpl_poses_.size() / smpl_pose_frames;
          }
        } else {
          missing_warnings.push_back("smpl_pose.csv");
        }

        // Must have at least some data to be valid
        if (num_frames == 0) {
          ioai::debug::log("MotionData").error("no valid data found for {} - SKIPPING MOTION", motion_name);
          continue;
        }

        // Decide which missing file warnings to show based on what data we have
        bool has_robot_data = (num_joints > 0 || num_bodies > 0 || num_body_quaternions > 0);
        bool has_smpl_data = (num_smpl_joints > 0 || num_smpl_poses > 0);

        // Show relevant warnings only for the type of data we have
        for (const auto& missing_file : missing_warnings) {
          bool is_smpl_file = (missing_file == "smpl_joint.csv" || missing_file == "smpl_pose.csv");
          bool is_robot_file = !is_smpl_file;

          // Show warning if it's relevant to the data type we have:
          // - Robot file warning only if we have robot data
          // - SMPL file warning only if we have SMPL data
          bool should_warn = (is_robot_file && has_robot_data) || (is_smpl_file && has_smpl_data);

          if (should_warn) {
            ioai::debug::log("MotionData").warn("missing {} for {}", missing_file, motion_name);
          }
        }

        motion->name = motion_name;
        motion->timesteps = num_frames;
        motion->num_joints = num_joints;
        motion->num_bodies = num_bodies;
        motion->num_body_quaternions = num_body_quaternions;
        motion->num_smpl_joints = num_smpl_joints;
        motion->num_smpl_poses = num_smpl_poses;

        motion->positions_world_tmp.resize(num_joints + 1);
        motion->rotations_world_tmp.resize(num_joints + 1);
        motion->pos_filter_tmp.resize(num_frames);
        motion->ang_filter_tmp.resize(num_frames);

        ioai::debug::log("MotionData").info("loaded {} ({} timesteps)", motion_name, motion->timesteps);
        motions.push_back(motion);
      }

      return !motions.empty();
    }

    void PrintSummary() const {
      ioai::debug::log("MotionData").info("motions loaded: {}", motions.size());
    }

    // Get a specific motion by name (returns shared_ptr for thread-safe shared ownership)
    std::shared_ptr<const MotionSequence> GetMotion(const std::string& name) const {
      for (const auto& motion : motions) {
        if (motion->name == name) { return motion; }
      }
      return nullptr;
    }

    // Get motion by index (returns shared_ptr for thread-safe shared ownership)
    std::shared_ptr<const MotionSequence> GetMotion(size_t index) const {
      if (index < motions.size()) { return motions[index]; }
      return nullptr;
    }

    // Alias for GetMotion - kept for backward compatibility
    std::shared_ptr<const MotionSequence> GetMotionShared(size_t index) const {
      return GetMotion(index);
    }
  private:
    // Read metadata.txt to get body part indexes and other info
    bool ReadMetadata(const std::string& filename, MotionSequence& motion) {
      std::ifstream file(filename);
      if (!file.is_open()) { return false; }

      std::string line;
      while (std::getline(file, line)) {
        if (line.find("Body part indexes:") != std::string::npos) {
          // Next line should contain the indexes
          if (std::getline(file, line)) {
            // Parse array like: [ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]
            std::regex number_regex("\\d+");
            std::sregex_iterator iter(line.begin(), line.end(), number_regex);
            std::sregex_iterator end;

            motion.body_part_indexes.clear();
            for (; iter != end; ++iter) { motion.body_part_indexes.push_back(std::stoi(iter->str())); }
          }
        } else if (line.find("Total timesteps:") != std::string::npos) {
          std::stringstream ss(line);
          std::string temp;
          ss >> temp >> temp >> motion.timesteps;
        }
      }

      return !motion.body_part_indexes.empty();
    }

    // Read 2D CSV data (joint positions, velocities) - using double precision
    int ReadCSV(const std::string& filename, std::vector<double>& data) {
      std::ifstream file(filename);
      if (!file.is_open()) { return false; }

      std::string line;
      int line_number = 0;

      while (std::getline(file, line)) {
        if (line_number == 0) {
          line_number++;
          continue; // Skip header
        }

        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
          try {
            double value = std::stod(cell); // Use stod for double precision
            data.push_back(value);
          } catch (...) {
            // Skip invalid values
          }
        }

        line_number++;
      }

      return line_number - 1;
    }

    // Read 3D CSV data (body positions, quaternions, velocities) - using double precision
    // CSV is flattened, need to reshape to [timestep][body_id][coordinates]
    template<int coords_per_body>
    int ReadCSV3D(const std::string& filename, std::vector<std::array<double, coords_per_body>>& data) {
      std::vector<double> flat_data;
      int num_lines = ReadCSV(filename, flat_data);
      if (num_lines == 0) { return 0; }

      int line_size = flat_data.size() / num_lines;
      if(line_size * num_lines != flat_data.size()) {
        ioai::debug::log("MotionData").warn("inconsistent line sizes in {}", filename);
        return 0;
      }

      int elements_per_line = line_size / coords_per_body;
      if(elements_per_line * coords_per_body != line_size) {
        ioai::debug::log("MotionData").warn("unexpected element count per line in {}", filename);
        return 0;
      }

      data.clear();
      std::array<double, coords_per_body> body_coords;
      for(int line=0; line < num_lines; line++) {
        for(int element=0; element < elements_per_line; element++) {
          for (int coord = 0; coord < coords_per_body; coord++) {
            body_coords[coord] = flat_data[line * line_size + element * coords_per_body + coord];
          }
          data.push_back(body_coords);
        }
      }

      return num_lines;
    }
};

#endif // MOTION_DATA_READER_HPP
