# G1 planner to H2 kinematic demo

This standalone C++ showcase runs the MotionBricks G1 TensorRT planner,
retargets each generated window to H2, validates continuity and mapping error,
and displays synchronized G1/H2 MuJoCo motion. It is kinematics only: it does
not start ROS 2, DDS, a policy, a bridge, a controller, or actuator output.

The source and link graph is self-contained in this repository. Reusable
libraries are pinned as Git submodules; the large planner and robot meshes are
runtime assets and are intentionally not committed.

## Setup and build

Use `sonic_sim_env`, which provides CUDA, TensorRT and the system development
packages used by the demo:

```bash
git submodule update --init --recursive
scripts/bootstrap_dependencies.sh Release
scripts/bootstrap_assets.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
python3 -m unittest discover -s tests -v
```

`bootstrap_assets.sh [WORKSPACE_ROOT]` is only a local convenience helper. For
an independently supplied asset bundle, populate the layout documented in
[`assets/README.md`](assets/README.md), or pass `--planner`, `--g1-mjcf`,
`--h2-mjcf`, and `--retargeter-root` explicitly. The environment equivalents
are `G1_H2_PLANNER`, `G1_H2_G1_MJCF`, `G1_H2_H2_MJCF`, and
`G1_H2_RETARGETER_ROOT`.

## Run

`run_demo.sh` works from the host or inside `sonic_sim_env`.
`record_video.sh` runs on the host so it can pipe container-rendered frames to
the host FFmpeg:

```bash
# Generate, validate, and open the synchronized viewer.
scripts/run_demo.sh

# Deterministic non-interactive generation and validation.
scripts/run_demo.sh --headless

# 30 seconds, 1500 frames, 1280x720, 50 fps, H.264.
scripts/record_video.sh
```

Host absolute paths below the ioenv workspace are translated to their
`/root/workspace` container paths. Paths elsewhere on the host are rejected;
copy or link those assets into the workspace first. Relative output paths are
resolved from this repository.

The default recording uses seed `1234` and five continuous six-second random
locomotion segments, replanning every ten frames. Overrides include
`--seed`, `--random-clips`, `--random-duration-min`,
`--random-duration-max`, `--replan-frames`, `--output`, `--video`, `--width`,
`--height`, `--fps`, `--crf`, and `--max-frames`. Use `--fixed-demo` together
with planner options such as `--mode`, `--speed`, and `--replans` for a short
explicit schedule.

Viewer controls: `Space` play/pause, `Left/Right` step, `Up/Down` speed,
`R` rewind, `A/D` yaw, `W/S` pitch, `Q/E` zoom, `F` follow root, and `Esc`
quit.

## Outputs and checks

Each run writes `g1_original/`, `h2_retarget/`, `comparison_metrics.csv`,
`retarget_timeline.csv`, `command_schedule.csv`, and `summary.txt`. The standard
library-only validator checks 29-joint/body structure, finite values, mapping
error, continuity, deadline misses, and dropped generations:

```bash
python3 scripts/verify_results.py output/RUN_DIRECTORY
```

Default limits are mean joint RMSE <= 0.15 rad, root-relative body RMSE <=
0.18 m, maximum H2 root step <= 0.06 m, and maximum H2 joint-step RMSE <=
0.08 rad.
