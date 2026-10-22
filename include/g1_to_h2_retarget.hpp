/**
 * @file g1_to_h2_retarget.hpp
 * @brief G1 MotionSequence ↔ MuJoCo qpos helpers + in-process C++ retarget (Pinocchio+OSQP / Pink FrameTask).
 */

#ifndef G1_TO_H2_RETARGET_HPP
#define G1_TO_H2_RETARGET_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "motion_data_reader.hpp"
#include "math_utils.hpp"
#include "policy_parameters.hpp"
#include "robot_retargeter/g1_to_h2.hpp"

inline void MotionFrameToG1MujocoQpos(
    const MotionSequence& motion,
    int frame,
    std::array<double, 36>& qpos)
{
    const auto& pos = motion.BodyPositions(frame)[0];
    const auto& quat = motion.BodyQuaternions(frame)[0];
    qpos[0] = pos[0];
    qpos[1] = pos[1];
    qpos[2] = pos[2];
    qpos[3] = quat[0];
    qpos[4] = quat[1];
    qpos[5] = quat[2];
    qpos[6] = quat[3];
    for (int mj = 0; mj < 29; ++mj) {
        qpos[7 + mj] = motion.JointPositions(frame)[isaaclab_to_mujoco[mj]];
    }
}

inline void H2MujocoQposToMotionFrame(
    const std::array<double, 36>& qpos,
    MotionSequence& motion,
    int frame)
{
    motion.BodyPositions(frame)[0] = {qpos[0], qpos[1], qpos[2]};
    motion.BodyQuaternions(frame)[0] = {qpos[3], qpos[4], qpos[5], qpos[6]};
    for (int mj = 0; mj < 29; ++mj) {
        motion.JointPositions(frame)[isaaclab_to_mujoco[mj]] = qpos[7 + mj];
    }
}

inline void CopyMotionSequenceFrame(
    const MotionSequence& src,
    int src_frame,
    MotionSequence& dst,
    int dst_frame)
{
    std::copy(
        src.JointPositions(src_frame),
        src.JointPositions(src_frame) + src.GetNumJoints(),
        dst.JointPositions(dst_frame));
    std::copy(
        src.JointVelocities(src_frame),
        src.JointVelocities(src_frame) + src.GetNumJoints(),
        dst.JointVelocities(dst_frame));
    dst.BodyPositions(dst_frame)[0] = src.BodyPositions(src_frame)[0];
    dst.BodyQuaternions(dst_frame)[0] = src.BodyQuaternions(src_frame)[0];
}

/**
 * Replace a motion window with a stationary copy of one of its frames.
 *
 * The G1 planner's first inference is a bootstrap used to establish context.
 * It is not yet driven by the manager command, so playing the sampled IDLE
 * continuation can move away from an otherwise-good initial standing pose.
 * Keeping the window length intact lets the normal planner timeline and later
 * replans continue to use the same code path.
 */
inline void FreezeMotionSequenceAtFrame(MotionSequence& motion, int frame)
{
    if (motion.timesteps <= 0) return;
    frame = std::clamp(frame, 0, motion.timesteps - 1);
    for (int dst_frame = 0; dst_frame < motion.timesteps; ++dst_frame) {
        if (dst_frame != frame) {
            CopyMotionSequenceFrame(motion, frame, motion, dst_frame);
        }
        std::fill(
            motion.JointVelocities(dst_frame),
            motion.JointVelocities(dst_frame) + motion.GetNumJoints(),
            0.0);
    }
}

// Matches g1_recorded_to_h2_reference.py --root-z-offset (default 0.05).
// Applied after C++ IK, not inside robot_retargeter (CLI must stay unlifted).
inline constexpr double kH2RetargetRootZOffset = 0.05;

struct H2RetargetContinuityResult {
    bool applied = false;
    int blended_frames = 0;
    int vertical_recovery_frames = 0;
    double root_translation_correction_m = 0.0;
    double root_planar_correction_m = 0.0;
    double root_vertical_correction_m = 0.0;
    double root_yaw_correction_rad = 0.0;
    double joint_anchor_rmse_before_rad = 0.0;
    double joint_anchor_rmse_after_rad = 0.0;
};

struct H2RootZSmoothingResult {
    bool applied = false;
    double safety_lift_m = 0.0;
};

/**
 * Smooth support-foot switching artifacts in the retargeted pelvis height.
 * The filtered curve receives one constant safety lift so it never falls
 * below the original ground-aligned root z at any frame.
 */
inline H2RootZSmoothingResult SmoothH2RetargetRootZ(
    MotionSequence& motion, int radius)
{
    H2RootZSmoothingResult result;
    if (radius <= 0 || motion.timesteps < 3) return result;
    radius = std::min(radius, motion.timesteps - 1);
    const double sigma = std::max(radius * 0.5, 0.5);
    std::vector<double> weights(2 * radius + 1, 0.0);
    double weight_sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double value = std::exp(
            -0.5 * offset * offset / (sigma * sigma));
        weights[offset + radius] = value;
        weight_sum += value;
    }
    for (double& weight : weights) weight /= weight_sum;

    std::vector<double> original(motion.timesteps, 0.0);
    std::vector<double> filtered(motion.timesteps, 0.0);
    for (int frame = 0; frame < motion.timesteps; ++frame) {
        original[frame] = motion.BodyPositions(frame)[0][2];
    }
    for (int frame = 0; frame < motion.timesteps; ++frame) {
        for (int offset = -radius; offset <= radius; ++offset) {
            const int sample = std::clamp(frame + offset, 0,
                                          motion.timesteps - 1);
            filtered[frame] +=
                weights[offset + radius] * original[sample];
        }
    }
    for (int frame = 0; frame < motion.timesteps; ++frame) {
        result.safety_lift_m = std::max(
            result.safety_lift_m, original[frame] - filtered[frame]);
    }
    for (int frame = 0; frame < motion.timesteps; ++frame) {
        motion.BodyPositions(frame)[0][2] =
            filtered[frame] + result.safety_lift_m;
    }
    result.applied = true;
    return result;
}

/**
 * Align a newly retargeted H2 window to the overlapping old H2 window.
 *
 * `previous_frame` and `next_frame` represent the same source time. A constant
 * planar SE(2) transform removes the per-StepBatch world-root re-anchoring.
 * Vertical position is different: ground alignment provides an absolute world
 * reference, so its anchor correction must decay to zero instead of being
 * integrated forever. When `blend_pose` is enabled, the overlapping old H2
 * future is cross-faded into the new IK solution, including that vertical
 * recovery, so a neutral-seed/local-minimum change cannot produce a joint or
 * root discontinuity at publication.
 *
 * This does not mutate the retargeter and is safe to apply after a background
 * solve, immediately before publishing the immutable window.
 */
inline H2RetargetContinuityResult ApplyH2RetargetWindowContinuity(
    const MotionSequence& previous,
    int previous_frame,
    MotionSequence& next,
    int next_frame,
    int blend_num_frames = 8,
    bool blend_pose = true)
{
    H2RetargetContinuityResult result;
    if (previous.timesteps <= 0 || next.timesteps <= 0 ||
        previous_frame < 0 || previous_frame >= previous.timesteps ||
        next_frame < 0 || next_frame >= next.timesteps) {
        return result;
    }

    const auto previous_anchor_pos = previous.BodyPositions(previous_frame)[0];
    const auto next_anchor_pos = next.BodyPositions(next_frame)[0];
    const auto previous_anchor_quat = previous.BodyQuaternions(previous_frame)[0];
    const auto next_anchor_quat = next.BodyQuaternions(next_frame)[0];
    double yaw_delta = calc_heading_d(previous_anchor_quat) -
                       calc_heading_d(next_anchor_quat);
    yaw_delta = std::atan2(std::sin(yaw_delta), std::cos(yaw_delta));
    const auto yaw_correction = euler_z_to_quat_d(yaw_delta);

    const std::array<double, 3> translation{
        previous_anchor_pos[0] - next_anchor_pos[0],
        previous_anchor_pos[1] - next_anchor_pos[1],
        previous_anchor_pos[2] - next_anchor_pos[2]};
    result.root_translation_correction_m = std::sqrt(
        translation[0] * translation[0] + translation[1] * translation[1] +
        translation[2] * translation[2]);
    result.root_planar_correction_m = std::hypot(translation[0], translation[1]);
    result.root_vertical_correction_m = translation[2];
    result.root_yaw_correction_rad = yaw_delta;

    double joint_squared_before = 0.0;
    const int num_joints = std::min(previous.GetNumJoints(), next.GetNumJoints());
    for (int joint = 0; joint < num_joints; ++joint) {
        const double error = next.JointPositions(next_frame)[joint] -
                             previous.JointPositions(previous_frame)[joint];
        joint_squared_before += error * error;
    }
    result.joint_anchor_rmse_before_rad =
        std::sqrt(joint_squared_before / std::max(num_joints, 1));

    // Rotate the new root trajectory around its anchor and permanently align
    // only x/y. Keeping the z translation here would integrate every
    // StepBatch reset residual and make the robot slowly sink or rise.
    for (int frame = next_frame; frame < next.timesteps; ++frame) {
        const auto position = next.BodyPositions(frame)[0];
        const std::array<double, 3> relative{
            position[0] - next_anchor_pos[0],
            position[1] - next_anchor_pos[1],
            position[2] - next_anchor_pos[2]};
        const auto rotated = quat_rotate_d(yaw_correction, relative);
        next.BodyPositions(frame)[0][0] = previous_anchor_pos[0] + rotated[0];
        next.BodyPositions(frame)[0][1] = previous_anchor_pos[1] + rotated[1];
        // Yaw does not change z. Preserve the new window's absolute,
        // ground-aligned height; the publication transition is handled below.
        next.BodyPositions(frame)[0][2] = position[2];
        next.BodyQuaternions(frame)[0] = quat_unit_d(quat_mul_d(
            yaw_correction, next.BodyQuaternions(frame)[0]));
    }

    blend_num_frames = std::max(blend_num_frames, 1);
    if (blend_pose) {
        const int available = std::min(previous.timesteps - previous_frame,
                                       next.timesteps - next_frame);
        const int count = std::min(available, blend_num_frames + 1);
        for (int offset = 0; offset < count; ++offset) {
            const int old_frame = previous_frame + offset;
            const int new_frame = next_frame + offset;
            const double w_new = std::clamp(
                static_cast<double>(offset) / blend_num_frames, 0.0, 1.0);
            const double w_old = 1.0 - w_new;
            for (int axis = 0; axis < 3; ++axis) {
                next.BodyPositions(new_frame)[0][axis] =
                    w_old * previous.BodyPositions(old_frame)[0][axis] +
                    w_new * next.BodyPositions(new_frame)[0][axis];
            }
            next.BodyQuaternions(new_frame)[0] = quat_slerp_d(
                previous.BodyQuaternions(old_frame)[0],
                next.BodyQuaternions(new_frame)[0], w_new);
            for (int joint = 0; joint < num_joints; ++joint) {
                next.JointPositions(new_frame)[joint] =
                    w_old * previous.JointPositions(old_frame)[joint] +
                    w_new * next.JointPositions(new_frame)[joint];
            }
        }
        result.blended_frames = count;
        result.vertical_recovery_frames = count;
    } else {
        // Root-only mode still needs an exact z anchor at publication, but the
        // correction tapers to zero so ground height remains absolute.
        const int count = std::min(next.timesteps - next_frame,
                                   blend_num_frames + 1);
        for (int offset = 0; offset < count; ++offset) {
            const int frame = next_frame + offset;
            const double correction_weight = std::clamp(
                1.0 - static_cast<double>(offset) / blend_num_frames,
                0.0, 1.0);
            next.BodyPositions(frame)[0][2] +=
                correction_weight * translation[2];
        }
        result.vertical_recovery_frames = count;
    }

    double joint_squared_after = 0.0;
    for (int joint = 0; joint < num_joints; ++joint) {
        const double error = next.JointPositions(next_frame)[joint] -
                             previous.JointPositions(previous_frame)[joint];
        joint_squared_after += error * error;
    }
    result.joint_anchor_rmse_after_rad =
        std::sqrt(joint_squared_after / std::max(num_joints, 1));

    for (int frame = 0; frame + 1 < next.timesteps; ++frame) {
        for (int joint = 0; joint < next.GetNumJoints(); ++joint) {
            next.JointVelocities(frame)[joint] =
                (next.JointPositions(frame + 1)[joint] -
                 next.JointPositions(frame)[joint]) * 50.0;
        }
    }
    if (next.timesteps > 1) {
        std::copy(next.JointVelocities(next.timesteps - 2),
                  next.JointVelocities(next.timesteps - 2) + next.GetNumJoints(),
                  next.JointVelocities(next.timesteps - 1));
    }
    result.applied = true;
    return result;
}

inline bool RetargetG1MotionSequenceToH2(
    robot_retargeter::G1ToH2Retargeter& retargeter,
    const MotionSequence& g1_motion,
    MotionSequence& h2_motion,
    int num_frames,
    double root_z_offset = kH2RetargetRootZOffset)
{
    if (num_frames <= 0) {
        return false;
    }

    std::vector<double> g1_qpos(static_cast<size_t>(num_frames) * 36);
    for (int f = 0; f < num_frames; ++f) {
        std::array<double, 36> frame_q{};
        MotionFrameToG1MujocoQpos(g1_motion, f, frame_q);
        std::copy(frame_q.begin(), frame_q.end(), g1_qpos.begin() + static_cast<size_t>(f) * 36);
    }

    std::vector<double> h2_qpos(static_cast<size_t>(num_frames) * 36, 0.0);
    try {
        retargeter.StepBatch(g1_qpos.data(), num_frames, h2_qpos.data());
    } catch (const std::exception& e) {
        std::cerr << "[G1→H2 retarget] C++ retarget failed: " << e.what() << std::endl;
        return false;
    }

    h2_motion.timesteps = num_frames;
    // First materialize every retargeted pose.  Joint velocity computation must
    // be a separate pass: reading frame f+1 while that frame still contains the
    // ReserveCapacity() value (or data from a previous window) produced bogus
    // reference velocities as large as tens of rad/s.
    for (int f = 0; f < num_frames; ++f) {
        std::array<double, 36> frame_q{};
        std::copy(
            h2_qpos.begin() + static_cast<size_t>(f) * 36,
            h2_qpos.begin() + static_cast<size_t>(f + 1) * 36,
            frame_q.begin());
        H2MujocoQposToMotionFrame(frame_q, h2_motion, f);
        if (root_z_offset != 0.0) {
            h2_motion.BodyPositions(f)[0][2] += root_z_offset;
        }
    }
    for (int f = 0; f + 1 < num_frames; ++f) {
        for (int j = 0; j < 29; ++j) {
            h2_motion.JointVelocities(f)[j] =
                (h2_motion.JointPositions(f + 1)[j] -
                 h2_motion.JointPositions(f)[j]) * 50.0;
        }
    }
    if (num_frames > 1) {
        for (int j = 0; j < 29; ++j) {
            h2_motion.JointVelocities(num_frames - 1)[j] =
                h2_motion.JointVelocities(num_frames - 2)[j];
        }
    }
    return true;
}

#endif // G1_TO_H2_RETARGET_HPP
