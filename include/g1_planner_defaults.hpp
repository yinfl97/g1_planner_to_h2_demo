/**
 * @file g1_planner_defaults.hpp
 * @brief G1 standing defaults for H2 deploy running the G1 MotionBricks planner (scheme A).
 *
 * Planner context / height use G1 shadow space; only policy-facing motion is retargeted to H2.
 */

#ifndef G1_PLANNER_DEFAULTS_HPP
#define G1_PLANNER_DEFAULTS_HPP

#include <array>

// G1 default_height from LocalMotionPlannerBase / planner_sonic training.
constexpr double G1_PLANNER_DEFAULT_HEIGHT = 0.788740;

// G1 default standing joint angles in **MuJoCo qpos[7:36] order** (matches g1 policy_parameters.hpp).
const std::array<double, 29> G1_PLANNER_DEFAULT_JOINTS_MUJOCO = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
};

#endif // G1_PLANNER_DEFAULTS_HPP
