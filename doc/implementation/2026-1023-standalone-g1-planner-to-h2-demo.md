<!-- INDEX
plan_id: 2026-1023-standalone-g1-planner-to-h2-demo
title: Extract G1 planner to H2 demo into an independent repository
type: implementation
status: COMPLETED
branch: integrate/standalone-g1-planner-to-h2-demo
plan: doc/plan/2026-1023-standalone-g1-planner-to-h2-demo.md
build: PASS — standalone Release build, validator tests, headless trajectory, and 1500-frame MP4 verified in sonic_sim_env; gear legacy H2 build PASS
deviations: Pinocchio's MJCF parser requires urdfdom headers supplied by the sonic_sim_env system prefix; no ROS 2, DDS, Unitree, or gear target is linked
date: 2026-10-23
-->

# Implementation: Extract G1 planner to H2 demo into an independent repository

## Checklist

### Part A — Standalone repository

- [x] Add independent CMake and dependency/bootstrap scripts.
- [x] Extract planner, retarget, renderer, launcher and validator sources.
- [x] Remove all ROS 2, Unitree SDK and gear source/link dependencies.
- [x] Add repository metadata, documentation and runtime asset layout.

### Part B — Demo behavior

- [x] Preserve interactive synchronized MuJoCo visualization.
- [x] Preserve headless generation and threshold validation.
- [x] Preserve the default 30-second labeled H.264 recorder.
- [x] Preserve CLI overrides for schedules, assets and output geometry.

### Part C — Source repository cleanup

- [x] Remove the misplaced example and documentation links.
- [x] Remove demo-only Just/CMake options and targets.
- [x] Remove the planner-test source after consumer verification.
- [x] Preserve production helpers and unrelated dirty state.

### Part D — Verification and commits

- [x] Configure and build exclusively from the new repository.
- [x] Run validator tests and a headless trajectory.
- [x] Record and probe the 30-second MP4.
- [x] Audit compile/link commands for gear dependencies.
- [x] Run repository checks and commit both repositories.

## Implementation notes

- Added a minimal standalone CMake graph for the original TensorRT planner,
  robot retargeter, MuJoCo viewer, EGL renderer, and their pinned dependencies.
- Pinned Pinocchio, OSQP, qpOASES, MuJoCo, and GLFW to public GitHub
  submodules. Vendored the internal-only C++ retarget subset and required MJCF
  assets. Dependency builds install beneath `build/deps/install`.
- Stored the planner and robot assets as regular Git LFS files; launchers also
  support explicit CLI/environment asset paths.
- Replaced the deployment logger include with a small local spdlog adapter, so
  the extracted planner has no ROS 2/ioai_msgs/Unitree dependency.
- The host launcher translates workspace absolute paths to `/root/workspace`.
  The recorder preserves the five-by-six-second random schedule and pipes EGL
  RGB frames directly into host FFmpeg.
- Removed the example directory, README links, Just recipes, CMake option and
  targets, and the unconsumed planner-test source from `gear_sonic_deploy`.
  Its production planner/retarget helpers remain unchanged.
- Preserved the pre-existing dirty source-repository submodules exactly as
  found. Generated files from the removed gear example were moved to the
  desktop trash.

## Verification

- `scripts/bootstrap_dependencies.sh Release 8`: PASS.
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: PASS.
- `cmake --build build -j8`: PASS; both executables linked.
- `python3 -m unittest discover -s tests -v`: PASS, 4 tests.
- `scripts/run_demo.sh --headless --output output/headless-verification`:
  PASS, 175 frames, 29 joints/bodies, zero misses/drops, mapping and continuity
  thresholds passed.
- Host path translation smoke test with an absolute planner and output path:
  PASS, 75 frames and all validator checks.
- `scripts/record_video.sh --output output/video-verification`: PASS; validator
  passed for 1500 frames and 150 replans.
- `ffprobe`: H.264, 1280x720, 50/1 fps, 1500 frames, 30.000000 seconds,
  7,337,577 bytes.
- Compile/link audit: no `gear_sonic_deploy` path in `compile_commands.json` or
  any `link.txt`; no rclcpp/ioai_msgs/Unitree include in standalone sources.
- `just build-h2-legacy Release` in `gear_sonic_deploy`: PASS after cleanup.
- `git diff --check`: PASS in both repositories.
