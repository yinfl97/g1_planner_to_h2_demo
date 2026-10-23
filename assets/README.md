# Runtime assets

The demo expects these runtime files by default:

- `planner/planner_sonic.onnx`
- `g1/scene_29dof.xml` and its referenced meshes/XML
- `h2/scene_h2_29dof.xml` and its referenced meshes/XML

These are regular files committed through Git LFS, so a recursive clone plus
`git lfs pull` is enough to run the demo without another workspace checkout.
You can still provide explicit `--planner`, `--g1-mjcf`, and `--h2-mjcf`
paths to `scripts/run_demo.sh` when testing alternative assets.
