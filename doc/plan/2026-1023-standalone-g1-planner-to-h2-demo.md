<!-- INDEX
plan_id: 2026-1023-standalone-g1-planner-to-h2-demo
title: Extract G1 planner to H2 demo into an independent repository
type: plan
status: PLANNED
branch: integrate/standalone-g1-planner-to-h2-demo
created: 2026-10-23
implementation:
findings:
-->

# Extract G1 planner to H2 demo into an independent repository

## Goal

Create `/home/fox/ioenv_ws/iodev_ws/src/g1_planner_to_h2_demo` as an
independent Git repository containing the original C++ planner, G1-to-H2
retarget validation, MuJoCo visualization, headless renderer, and 30-second
continuous-action recording workflow. The repository must configure and build
without adding `gear_sonic_deploy` as a source or link dependency.

The finished repository remains a kinematic showcase: it does not start ROS 2,
DDS, a controller, a policy, a bridge, or actuator output.

## Current state and findings

- The original executable is
  `gear_sonic_deploy/src/h2/h2_deploy_onnx_ref/src/h2_planner_test.cpp`.
- Its direct reusable C++ dependencies are the planner/motion headers,
  `TRTInference`, and `robot_retargeter`; its demo support is currently under
  `gear_sonic_deploy/examples/g1_planner_to_h2/`.
- The existing CMake target inherits many deployment-only ROS 2 and Unitree
  dependencies. The standalone repository must define a minimal target graph
  containing only CUDA/TensorRT, MuJoCo/OpenGL/GLFW, Eigen, Pinocchio, OSQP,
  qpOASES, threads, and the extracted sources.
- The planner ONNX is about 739 MiB and robot mesh assets total more than 100
  MiB. They are runtime data, not source. Keep their paths configurable and
  provide a local asset bootstrap/convenience mechanism; do not hard-code a
  compile/link dependency on the source repository.
- `robot_retargeter`, qpOASES, Pinocchio, OSQP and MuJoCo are pinned upstream
  repositories in the current workspace. Preserve their pinning in the new
  repository using submodules or an equivalent explicit dependency manifest.
- `gear_sonic_deploy` has unrelated dirty submodules
  `ros2_msgs/ioai_msgs` and `thirdparty/io_debug_log`; they must remain
  untouched.

## Decisions

- **D1 — Separate repository and build graph.** The new repository owns its
  `CMakeLists.txt`, build scripts, extracted headers/sources, validation tests,
  documentation, and Git history. Its build must not pass include/library paths
  into `gear_sonic_deploy`.
- **D2 — Preserve the C++ algorithm.** Move the original planner/retarget and
  continuity logic rather than reimplementing it in Python. Python remains only
  for result validation.
- **D3 — Pin reusable dependencies.** Use the existing internal Git remotes and
  known commits for reusable third-party projects. Remove the demo's accidental
  ROS 2/Unitree deployment linkage.
- **D4 — Treat large models as runtime assets.** Default to paths under
  `assets/`; accept explicit CLI/environment overrides. A setup helper may copy
  or symlink assets from an existing workspace for local use, while documented
  operation supports independently supplied assets.
- **D5 — Keep one-command showcase workflows.** Preserve interactive viewing,
  deterministic headless validation, and the default 30-second/1500-frame
  continuous random locomotion MP4 recording with action labels.
- **D6 — Remove the misplaced integration.** After the standalone repository
  passes verification, remove the demo scripts/renderer and demo-only build and
  documentation hooks from `gear_sonic_deploy`. Remove the original
  `h2_planner_test.cpp` only after confirming no production target consumes it.
  Do not change shared production H2 helpers that remain in use.
- **D7 — Same implementation branch in both repositories.** Use
  `integrate/standalone-g1-planner-to-h2-demo` for the new repository and the
  `gear_sonic_deploy` cleanup commit.

## Task queue

### Part A — Standalone repository

- [ ] Add the independent CMake project and dependency/bootstrap scripts.
- [ ] Extract the original planner test, required planner/motion headers,
  TensorRT wrapper, retargeter integration, MuJoCo renderer, launcher,
  validator, and validator tests.
- [ ] Eliminate includes and link dependencies on ROS 2, Unitree SDK, and the
  `gear_sonic_deploy` source/build tree.
- [ ] Add `.gitignore`, `.gitattributes` where needed, README, and an explicit
  runtime asset layout.

### Part B — Demo behavior

- [ ] Preserve synchronized G1/H2 interactive MuJoCo visualization.
- [ ] Preserve headless result generation and threshold validation.
- [ ] Preserve default offline video generation at 1280x720, 50 fps, 30
  seconds, H.264, using five continuous six-second action segments.
- [ ] Keep CLI overrides for seed, segment schedule, paths, dimensions, frame
  count, and output location.

### Part C — Source repository cleanup

- [ ] Remove `examples/g1_planner_to_h2/` and its root/H2 README links.
- [ ] Remove demo-only `.justfile` and CMake targets/options.
- [ ] Remove `h2_planner_test.cpp` if it has no remaining production consumer.
- [ ] Preserve shared production planner/retarget helpers and all unrelated
  user changes.

### Part D — Verification and commits

- [ ] Configure and build only from the new repository in `sonic_sim_env`.
- [ ] Run validator unit tests.
- [ ] Generate and validate a headless trajectory from the new executable.
- [ ] Record and probe a 30-second, 1500-frame MP4 from the new repository.
- [ ] Confirm the new build does not mention `gear_sonic_deploy` in compile or
  link commands.
- [ ] Run `git diff --check` and inspect status in both repositories.
- [ ] Commit the standalone repository and the `gear_sonic_deploy` cleanup on
  the named branch. Do not push, tag, or create a remote unless requested.

## Acceptance criteria

1. `g1_planner_to_h2_demo` is a Git repository with an independent CMake build
   and no source/link dependency on `gear_sonic_deploy`.
2. The generated G1 and H2 clips pass the existing 29-joint/body structural,
   finite-value, mapping-error, continuity, and zero-drop checks.
3. The default recorder produces an H.264 1280x720, 50 fps, 30-second video
   containing 1500 frames and multiple labeled continuous action segments.
4. `gear_sonic_deploy` no longer contains the standalone example, renderer, or
   planner-test executable target; unrelated dirty submodules are unchanged.
5. Both repositories contain focused, reviewable commits on
   `integrate/standalone-g1-planner-to-h2-demo`.

## Verification commands

```bash
cd /root/workspace/src/g1_planner_to_h2_demo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
python3 -m unittest discover -s tests -v
./scripts/run_demo.sh --headless
./scripts/record_video.sh
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration -of default=noprint_wrappers=1 output/*/*.mp4

git diff --check
git status --short

cd /root/workspace/src/gear_sonic_deploy
git diff --check
git status --short
```

## Out of scope

- Publishing or creating a remote repository.
- Modifying retarget mathematics or validation thresholds.
- H2 policy/encoder inference, closed-loop physics control, DDS, ROS 2, bridge,
  or hardware behavior.
- Changing the planner model or robot mesh content.
