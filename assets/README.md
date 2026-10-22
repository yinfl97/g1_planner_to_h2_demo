# Runtime assets

The demo expects these runtime files by default:

- `planner/planner_sonic.onnx`
- `g1/scene_29dof.xml` and its referenced meshes/XML
- `h2/scene_h2_29dof.xml` and its referenced meshes/XML

They are intentionally not committed. Run `scripts/bootstrap_assets.sh` to
link assets from the standard ioenv workspace, pass another workspace root as
its first argument, or provide explicit `--planner`, `--g1-mjcf`, and
`--h2-mjcf` paths to `scripts/run_demo.sh`.
