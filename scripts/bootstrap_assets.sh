#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"
SOURCE_WORKSPACE="${1:-${WORKSPACE_ROOT}}"

planner="${G1_H2_SOURCE_PLANNER:-${SOURCE_WORKSPACE}/src/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx}"
g1="${G1_H2_SOURCE_G1_ASSETS:-${SOURCE_WORKSPACE}/src/gear_sonic_deploy/g1}"
h2="${G1_H2_SOURCE_H2_ASSETS:-${SOURCE_WORKSPACE}/src/unitree_mujoco_sim/assets/h2}"

for path in "${planner}" "${g1}/scene_29dof.xml" "${h2}/scene_h2_29dof.xml"; do
  [[ -e "${path}" ]] || { echo "Missing source asset: ${path}" >&2; exit 1; }
done

mkdir -p "${REPO_ROOT}/assets/planner"
ln -sfn "$(realpath --relative-to="${REPO_ROOT}/assets/planner" "${planner}")" \
  "${REPO_ROOT}/assets/planner/planner_sonic.onnx"
ln -sfn "$(realpath --relative-to="${REPO_ROOT}/assets" "${g1}")" \
  "${REPO_ROOT}/assets/g1"
ln -sfn "$(realpath --relative-to="${REPO_ROOT}/assets" "${h2}")" \
  "${REPO_ROOT}/assets/h2"
echo "Runtime assets linked under ${REPO_ROOT}/assets"
