/**
 * @file planner_motion_pipeline.hpp
 * @brief Shared planner-motion copy/cross-fade used by H2 deploy and offline tests.
 */

#ifndef H2_PLANNER_MOTION_PIPELINE_HPP
#define H2_PLANNER_MOTION_PIPELINE_HPP

#include <algorithm>

#include "math_utils.hpp"
#include "motion_data_reader.hpp"

struct PlannerMotionBlendResult {
  bool success = false;
  bool first_motion = false;
  int timesteps = 0;
};

/**
 * Merge one newly-generated 50 Hz G1 planner window into the live planner
 * timeline. This is the exact pre-retarget operation used by h2_deploy:
 *
 *  - first window: copy as-is;
 *  - replans: rebase the old motion at current_frame and linearly cross-fade
 *    joint/root translation over eight frames, with quaternion slerp.
 *
 * The destination must already have enough capacity (h2_deploy reserves 1500
 * frames). On success, playback must restart at frame zero, matching deploy.
 * `independent_generated_timeline=false` preserves the current deploy helper's
 * tail behavior; tests can enable it to keep f_new advancing after f_old clamps.
 */
inline PlannerMotionBlendResult ApplyGeneratedPlannerMotion(
    const MotionSequence& generated,
    int generated_at_frame,
    int current_frame,
    MotionSequence& blended,
    int blend_num_frames = 8,
    bool independent_generated_timeline = false) {
  PlannerMotionBlendResult result;
  if (generated.timesteps <= 0 || generated.GetNumJoints() <= 0 ||
      generated.GetNumBodies() <= 0 || generated.GetNumBodyQuaternions() <= 0) {
    return result;
  }

  if (blended.timesteps == 0) {
    for (int f = 0; f < generated.timesteps; ++f) {
      std::copy(generated.JointPositions(f),
                generated.JointPositions(f) + generated.GetNumJoints(),
                blended.JointPositions(f));
      std::copy(generated.JointVelocities(f),
                generated.JointVelocities(f) + generated.GetNumJoints(),
                blended.JointVelocities(f));
      blended.BodyPositions(f)[0] = generated.BodyPositions(f)[0];
      blended.BodyQuaternions(f)[0] = generated.BodyQuaternions(f)[0];
    }
    blended.timesteps = generated.timesteps;
    result.success = true;
    result.first_motion = true;
    result.timesteps = blended.timesteps;
    return result;
  }

  const int new_anim_length =
      generated_at_frame - current_frame + generated.timesteps;
  if (new_anim_length <= 0) {
    return result;
  }

  const int blend_start_frame =
      std::max(0, generated_at_frame - current_frame);
  blend_num_frames = std::max(blend_num_frames, 1);

  for (int f = 0; f < new_anim_length; ++f) {
    const int timeline_frame = f + current_frame;
    int f_old = std::clamp(timeline_frame, 0, blended.timesteps - 1);
    // The production-compatible path preserves the current helper behavior.
    // In the corrected path, the new planner timeline must keep advancing
    // after the old timeline reaches its last frame; otherwise f_new freezes.
    const int new_timeline_frame = independent_generated_timeline
                                       ? timeline_frame
                                       : f_old;
    int f_new = std::clamp(new_timeline_frame - generated_at_frame, 0,
                           generated.timesteps - 1);

    double w_new =
        static_cast<double>(f - blend_start_frame) / blend_num_frames;
    w_new = std::clamp(w_new, 0.0, 1.0);
    const double w_old = 1.0 - w_new;

    for (int j = 0; j < blended.GetNumJoints(); ++j) {
      blended.JointPositions(f)[j] =
          w_old * blended.JointPositions(f_old)[j] +
          w_new * generated.JointPositions(f_new)[j];
      blended.JointVelocities(f)[j] =
          w_old * blended.JointVelocities(f_old)[j] +
          w_new * generated.JointVelocities(f_new)[j];
    }

    for (int axis = 0; axis < 3; ++axis) {
      blended.BodyPositions(f)[0][axis] =
          w_old * blended.BodyPositions(f_old)[0][axis] +
          w_new * generated.BodyPositions(f_new)[0][axis];
    }

    const std::array<double, 4> q_old =
        blended.BodyQuaternions(f_old)[0];
    const std::array<double, 4> q_new =
        generated.BodyQuaternions(f_new)[0];
    blended.BodyQuaternions(f)[0] = quat_slerp_d(q_old, q_new, w_new);
  }

  blended.timesteps = new_anim_length;
  result.success = true;
  result.timesteps = new_anim_length;
  return result;
}

#endif  // H2_PLANNER_MOTION_PIPELINE_HPP
