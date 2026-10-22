/**
 * @file robot_parameters.hpp
 * @brief Hardware-level constants for Unitree H2 29-DOF (head joints fixed).
 *
 * Motor indices follow MuJoCo / MJCF actuator order (ankle_roll before
 * ankle_pitch — differs from G1).
 */

#ifndef ROBOT_PARAMETERS_HPP
#define ROBOT_PARAMETERS_HPP

#include <array>
/// Policy / MuJoCo 29-DOF (head fixed). Real-robot LowCmd has 31 motors (head at 29/30).
const int H2_NUM_MOTOR = 29;
const int H2_HW_NUM_MOTOR = 31;
// Keep G1_NUM_MOTOR alias so shared observation / logger code compiles unchanged.
const int G1_NUM_MOTOR = H2_NUM_MOTOR;

struct MotorCommand {
    std::array<float, H2_NUM_MOTOR> q_target = {};
    std::array<float, H2_NUM_MOTOR> dq_target = {};
    std::array<float, H2_NUM_MOTOR> kp = {};
    std::array<float, H2_NUM_MOTOR> kd = {};
    std::array<float, H2_NUM_MOTOR> tau_ff = {};
};

struct HeadingState {
    std::array<double, 4> init_base_quat;
    double delta_heading;

    HeadingState(const std::array<double, 4>& quat = {1.0, 0.0, 0.0, 0.0}, double delta = 0.0)
        : init_base_quat(quat), delta_heading(delta) {}
};

struct OperatorState {
  bool stop = false;
  bool start = false;
  bool play = false;
};

enum class Mode {
  PR = 0,
  AB = 1
};

/// MuJoCo DFS / actuator order (h2.xml). Waist is yaw, roll, pitch.
/// Policy arrays (kps/kds/default_angles/action_scale) are in this order.
inline constexpr const char* H2_JOINT_NAMES[H2_NUM_MOTOR] = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};

/// Real-robot LowCmd/LowState motor ID order (31 motors).
/// Motors 0–28 match MuJoCo / Unitree H2JointIndex (wrist = roll, pitch, yaw).
/// Head is extra: 29=pitch, 30=yaw. See unitree_sdk2 example/h2/low_level.
inline constexpr const char* H2_HW_JOINT_NAMES[H2_HW_NUM_MOTOR] = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "head_pitch_joint",
    "head_yaw_joint",
};

/// hw_index -> mujoco_index. Identity: H2 LowCmd 0–28 == MuJoCo actuator order.
inline constexpr int H2_HW_TO_MJ[H2_NUM_MOTOR] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28};

/// mujoco_index -> hw_index (identity, same as H2_HW_TO_MJ).
inline constexpr int H2_MJ_TO_HW[H2_NUM_MOTOR] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28};

/// MuJoCo sim DDS uses lo/lo0 (or the "sim" alias). Real robot uses eth/enp*.
inline bool H2IsSimInterface(const std::string& iface) {
  return iface == "sim" || iface == "lo" || iface == "lo0";
}

/// LowCmd/LowState wire index → MuJoCo/policy index. Identity in sim.
inline int H2WireToMj(bool sim, int wire_i) {
  if (sim || wire_i < 0 || wire_i >= H2_NUM_MOTOR) {
    return wire_i;
  }
  return H2_HW_TO_MJ[wire_i];
}

/// Real-robot joint limits (rad), hardware motor order (wrist = roll, pitch, yaw).
inline constexpr float H2_HW_JOINT_LOWER[H2_HW_NUM_MOTOR] = {
    -2.4525975f, -0.467441f, -2.827f, -0.08725f, -0.3490658503f, -1.13446401358f,
    -2.4525975f, -2.1688605f, -2.827f, -0.08725f, -0.296705972783f, -1.13446401358f,
    -1.7453f, -0.5236f, -0.43633f,
    -2.6179939f, -0.517f, -2.6179939f, -0.986111f, -2.6179939f, -0.576f, -1.2217305f,
    -2.6179939f, -2.494f, -2.6179939f, -0.986111f, -2.6179939f, -0.576f, -1.2217305f,
    -0.5236f, -1.7453f};
inline constexpr float H2_HW_JOINT_UPPER[H2_HW_NUM_MOTOR] = {
    2.7754225f, 2.1688605f, 2.827f, 2.53025f, 0.296705972783f, 0.61086523808f,
    2.7754225f, 0.467441f, 2.827f, 2.53025f, 0.3490658503f, 0.61086523808f,
    1.7453f, 0.5236f, 0.5236f,
    1.8325957f, 2.494f, 2.6179939f, 3.0717795f, 2.6179939f, 0.576f, 1.2217305f,
    1.8325957f, 0.517f, 2.6179939f, 3.0717795f, 2.6179939f, 0.576f, 1.2217305f,
    0.83775f, 1.7453f};

/// H2 MuJoCo hinge indices (ankle_roll then ankle_pitch).
/// Waist IDs match the real robot: 12=yaw, 13=roll, 14=pitch.
enum H2JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnkleRoll = 4,
  LeftAnklePitch = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnkleRoll = 10,
  RightAnklePitch = 11,
  WaistYaw = 12,
  WaistRoll = 13,
  WaistPitch = 14,
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,
  LeftWristYaw = 21,
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,
  RightWristYaw = 28
};

// Compatibility alias used by observation / control code copied from G1.
using G1JointIndex = H2JointIndex;

#endif // ROBOT_PARAMETERS_HPP
