/**
 * @file policy_parameters.hpp
 * @brief H2 29-DOF policy constants: joint maps, PID gains, action scales,
 *        default standing pose.
 *
 * MuJoCo order uses ankle_roll before ankle_pitch (differs from G1).
 * IsaacLab↔MuJoCo DOF maps match G1 after head removal (see h2.py).
 */

#ifndef POLICY_PARAMETERS_HPP
#define POLICY_PARAMETERS_HPP

#include <array>
#include <vector>

const double ONE_DEGREE = 0.0174533;

const double ARMATURE_5020 = 0.003609725;
const double ARMATURE_7520_14 = 0.010177520;
const double ARMATURE_7520_22 = 0.025101925;
const double ARMATURE_4010 = 0.00425;

const double NATURAL_FREQ = 10 * 2.0 * 3.1415926535;
const double DAMPING_RATIO = 2;

const double STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;

const double DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
const double DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
const double DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
const double DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

// Effort limits from H2_CFG
const double EFFORT_HIP_PITCH_ROLL_KNEE = 417.0;
const double EFFORT_HIP_YAW_WAIST_YAW = 264.0;
const double EFFORT_ANKLE_WAIST_RP = 150.0;
const double EFFORT_ARM = 75.0;
const double EFFORT_WRIST_PY = 15.0;

const std::array<int, 5> vr_5point_index = {28, 29, 0, 18, 19};
const std::array<int, 3> vr_3point_index = {28, 29, 9};

const std::vector<int> upper_body_joint_mujoco_order_in_isaaclab_index = { 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
const std::vector<int> upper_body_joint_mujoco_order_in_mujoco_index = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
const std::vector<int> upper_body_joint_isaaclab_order_in_isaaclab_index = { 2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
const std::vector<int> upper_body_joint_isaaclab_order_in_mujoco_index = { 12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};
const std::vector<int> wrist_joint_mujoco_order_in_isaaclab_index = {23, 25, 27, 24, 26, 28};
const std::vector<int> wrist_joint_mujoco_order_in_mujoco_index = {19, 20, 21, 26, 27, 28};
const std::vector<int> wrist_joint_isaaclab_order_in_isaaclab_index = {23, 24, 25, 26, 27, 28};
const std::vector<int> wrist_joint_isaaclab_order_in_mujoco_index = {19, 26, 20, 27, 21, 28};
const std::vector<int> lower_body_joint_mujoco_order_in_isaaclab_index = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18};
const std::vector<int> lower_body_joint_mujoco_order_in_mujoco_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const std::vector<int> lower_body_joint_isaaclab_order_in_isaaclab_index = {0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18};
const std::vector<int> lower_body_joint_isaaclab_order_in_mujoco_index = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

// Indexed by mujoco motor index → isaaclab index
const std::array<int, 29> isaaclab_to_mujoco = {0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                                11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
// Indexed by isaaclab index → mujoco motor index
const std::array<int, 29> mujoco_to_isaaclab = {0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
                                                16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

// Kept name g1_action_scale for compatibility with observation/control code.
// MuJoCo / hardware ID order. Wrist roll is 5020+75Nm; pitch/yaw are 4010+15Nm.
const std::array<double, 29> g1_action_scale = {
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  0 left_hip_pitch
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  1 left_hip_roll
    0.25 * EFFORT_HIP_YAW_WAIST_YAW / STIFFNESS_7520_14,                //  2 left_hip_yaw
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  3 left_knee
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             //  4 left_ankle_roll
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             //  5 left_ankle_pitch
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  6 right_hip_pitch
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  7 right_hip_roll
    0.25 * EFFORT_HIP_YAW_WAIST_YAW / STIFFNESS_7520_14,                //  8 right_hip_yaw
    0.25 * EFFORT_HIP_PITCH_ROLL_KNEE / STIFFNESS_7520_22,              //  9 right_knee
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             // 10 right_ankle_roll
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             // 11 right_ankle_pitch
    0.25 * EFFORT_HIP_YAW_WAIST_YAW / STIFFNESS_7520_14,                // 12 waist_yaw
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             // 13 waist_roll
    0.25 * EFFORT_ANKLE_WAIST_RP / (2.0 * STIFFNESS_5020),             // 14 waist_pitch
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 15 left_shoulder_pitch
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 16 left_shoulder_roll
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 17 left_shoulder_yaw
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 18 left_elbow
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 19 left_wrist_roll
    0.25 * EFFORT_WRIST_PY / STIFFNESS_4010,                           // 20 left_wrist_pitch
    0.25 * EFFORT_WRIST_PY / STIFFNESS_4010,                           // 21 left_wrist_yaw
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 22 right_shoulder_pitch
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 23 right_shoulder_roll
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 24 right_shoulder_yaw
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 25 right_elbow
    0.25 * EFFORT_ARM / STIFFNESS_5020,                                // 26 right_wrist_roll
    0.25 * EFFORT_WRIST_PY / STIFFNESS_4010,                           // 27 right_wrist_pitch
    0.25 * EFFORT_WRIST_PY / STIFFNESS_4010,                           // 28 right_wrist_yaw
};

const std::array<float, 29> kps = {
    (float)STIFFNESS_7520_22, (float)STIFFNESS_7520_22, (float)STIFFNESS_7520_14, (float)STIFFNESS_7520_22,  // 0-3 L leg
    2.0f * (float)STIFFNESS_5020, 2.0f * (float)STIFFNESS_5020,                                            // 4-5 L ankle roll/pitch
    (float)STIFFNESS_7520_22, (float)STIFFNESS_7520_22, (float)STIFFNESS_7520_14, (float)STIFFNESS_7520_22,  // 6-9 R leg
    2.0f * (float)STIFFNESS_5020, 2.0f * (float)STIFFNESS_5020,                                            // 10-11 R ankle roll/pitch
    (float)STIFFNESS_7520_14,                    // 12 waist_yaw
    2.0f * (float)STIFFNESS_5020,                // 13 waist_roll
    2.0f * (float)STIFFNESS_5020,                // 14 waist_pitch
    (float)STIFFNESS_5020, (float)STIFFNESS_5020, (float)STIFFNESS_5020, (float)STIFFNESS_5020,  // 15-18 L arm
    (float)STIFFNESS_5020,                       // 19 left_wrist_roll
    (float)STIFFNESS_4010,                       // 20 left_wrist_pitch
    (float)STIFFNESS_4010,                       // 21 left_wrist_yaw
    (float)STIFFNESS_5020, (float)STIFFNESS_5020, (float)STIFFNESS_5020, (float)STIFFNESS_5020,  // 22-25 R arm
    (float)STIFFNESS_5020,                       // 26 right_wrist_roll
    (float)STIFFNESS_4010,                       // 27 right_wrist_pitch
    (float)STIFFNESS_4010,                       // 28 right_wrist_yaw
};

const std::array<float, 29> kds = {
    (float)DAMPING_7520_22, (float)DAMPING_7520_22, (float)DAMPING_7520_14, (float)DAMPING_7520_22,
    2.0f * (float)DAMPING_5020, 2.0f * (float)DAMPING_5020,
    (float)DAMPING_7520_22, (float)DAMPING_7520_22, (float)DAMPING_7520_14, (float)DAMPING_7520_22,
    2.0f * (float)DAMPING_5020, 2.0f * (float)DAMPING_5020,
    (float)DAMPING_7520_14,                      // 12 waist_yaw
    2.0f * (float)DAMPING_5020,                  // 13 waist_roll
    2.0f * (float)DAMPING_5020,                  // 14 waist_pitch
    (float)DAMPING_5020, (float)DAMPING_5020, (float)DAMPING_5020, (float)DAMPING_5020,
    (float)DAMPING_5020,                         // 19 left_wrist_roll
    (float)DAMPING_4010,                         // 20 left_wrist_pitch
    (float)DAMPING_4010,                         // 21 left_wrist_yaw
    (float)DAMPING_5020, (float)DAMPING_5020, (float)DAMPING_5020, (float)DAMPING_5020,
    (float)DAMPING_5020,                         // 26 right_wrist_roll
    (float)DAMPING_4010,                         // 27 right_wrist_pitch
    (float)DAMPING_4010,                         // 28 right_wrist_yaw
};

// Standing pose (MuJoCo order). Matches h2.py / H2_DEFAULT_MJ.
// H2 LowCmd 0–28 matches this MuJoCo order (no wrist remap). Head 29/30 is deploy-only.
const std::array<double, 29> default_angles = {
    -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
    -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
};

#endif // POLICY_PARAMETERS_HPP
