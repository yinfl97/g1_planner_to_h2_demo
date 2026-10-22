#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${SONIC_SIM_CONTAINER:-sonic_sim_env}"
CONTAINER_REPO="/root/workspace/src/g1_planner_to_h2_demo"

fail() {
  echo "[g1-planner-to-h2] ERROR: $*" >&2
  exit 1
}

for argument in "$@"; do
  if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    cat <<'EOF'
Usage: scripts/run_demo.sh [g1_planner_to_h2_demo options]

Runs the pure-kinematic G1 planner -> H2 retarget demo in sonic_sim_env.
Use --headless for automated generation/validation, or omit it for the viewer.
Outputs default to output/<timestamp>/.

Common options: --headless, --paused, --replans N, --mode NAME, --output DIR.
Asset paths may also be set with G1_H2_PLANNER, G1_H2_G1_MJCF,
G1_H2_H2_MJCF, and G1_H2_RETARGETER_ROOT.
EOF
    exit 0
  fi
done

if [[ "${G1_H2_DEMO_IN_CONTAINER:-0}" != "1" && "${REPO_ROOT}" != "${CONTAINER_REPO}" ]]; then
  command -v docker >/dev/null 2>&1 || fail "docker is required to enter ${CONTAINER_NAME}"
  running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  [[ "${running}" == "true" ]] || fail \
    "${CONTAINER_NAME} is not running; start it with: ioenv run sonic_sim"
  host_workspace="$(cd -- "${REPO_ROOT}/../.." && pwd)"
  translate_path() {
    local path="$1"
    if [[ "${path}" == "${host_workspace}" ]]; then
      echo /root/workspace
    elif [[ "${path}" == "${host_workspace}/"* ]]; then
      echo "/root/workspace/${path#"${host_workspace}/"}"
    elif [[ "${path}" == /* && "${path}" != /root/workspace && "${path}" != /root/workspace/* ]]; then
      fail "absolute host path must be under ${host_workspace}: ${path}"
    else
      echo "${path}"
    fi
  }
  translated_args=()
  while (($#)); do
    case "$1" in
      --output|--planner|--g1-mjcf|--h2-mjcf|--retargeter-root)
        (($# >= 2)) || fail "$1 requires a path"
        path_option="$1"
        option_path="$2"
        shift 2
        ;;
      --output=*|--planner=*|--g1-mjcf=*|--h2-mjcf=*|--retargeter-root=*)
        path_option="${1%%=*}"
        option_path="${1#*=}"
        shift
        ;;
      *)
        translated_args+=("$1")
        shift
        continue
        ;;
    esac
    translated_args+=("${path_option}" "$(translate_path "${option_path}")")
  done
  docker_env_args=()
  for variable in G1_H2_PLANNER G1_H2_G1_MJCF G1_H2_H2_MJCF G1_H2_RETARGETER_ROOT; do
    value="${!variable:-}"
    if [[ -n "${value}" ]]; then
      docker_env_args+=(-e "${variable}=$(translate_path "${value}")")
    fi
  done
  exec docker exec -i \
    -e G1_H2_DEMO_IN_CONTAINER=1 \
    "${docker_env_args[@]}" \
    "${CONTAINER_NAME}" \
    bash "${CONTAINER_REPO}/scripts/run_demo.sh" "${translated_args[@]}"
fi

cd "${REPO_ROOT}"

headless=0
output_dir=""
forward_args=()
while (($#)); do
  case "$1" in
    --headless)
      headless=1
      forward_args+=("$1")
      shift
      ;;
    --output)
      (($# >= 2)) || fail "--output requires a directory"
      output_dir="$2"
      shift 2
      ;;
    --output=*)
      output_dir="${1#--output=}"
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/output/$(date +%Y%m%d_%H%M%S)"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${REPO_ROOT}/${output_dir}"
fi

planner="${G1_H2_PLANNER:-${REPO_ROOT}/assets/planner/planner_sonic.onnx}"
retargeter="${G1_H2_RETARGETER_ROOT:-${REPO_ROOT}/thirdparty/robot_retargeter}"
g1_mjcf="${G1_H2_G1_MJCF:-${REPO_ROOT}/assets/g1/scene_29dof.xml}"
h2_mjcf="${G1_H2_H2_MJCF:-${REPO_ROOT}/assets/h2/scene_h2_29dof.xml}"
binary="${REPO_ROOT}/build/g1_planner_to_h2_demo"

for required in "${planner}" "${g1_mjcf}" "${h2_mjcf}" \
  "${retargeter}/asset/robot/g1_description/mjcf/g1.xml"; do
  [[ -f "${required}" ]] || fail "missing prerequisite: ${required}"
done

binary_ready=0
if [[ -x "${binary}" ]]; then
  ldd_output="$(ldd "${binary}" 2>&1 || true)"
  if [[ "${ldd_output}" != *"not found"* ]] && "${binary}" --help >/dev/null 2>&1; then
    binary_ready=1
  else
    echo "[g1-planner-to-h2] Existing binary is not runnable; rebuilding."
  fi
fi

if ((binary_ready == 0)); then
  command -v cmake >/dev/null 2>&1 || fail "cmake is required to rebuild the demo"
  echo "[g1-planner-to-h2] Building standalone demo..."
  if [[ ! -f "${REPO_ROOT}/build/deps/install/lib/cmake/mujoco/mujocoConfig.cmake" ]]; then
    "${SCRIPT_DIR}/bootstrap_dependencies.sh" Release
  fi
  cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${REPO_ROOT}/build" -j"$(nproc)"
  "${binary}" --help >/dev/null 2>&1 || fail \
    "rebuilt demo binary is still not runnable inside ${CONTAINER_NAME}"
fi

mkdir -p "${output_dir}"
echo "[g1-planner-to-h2] Pure kinematics only: no DDS, ROS control, policy, bridge, or motor commands."
echo "[g1-planner-to-h2] Output: ${output_dir}"

"${binary}" \
  --planner "${planner}" \
  --retargeter-root "${retargeter}" \
  --g1-mjcf "${g1_mjcf}" \
  --h2-mjcf "${h2_mjcf}" \
  --retarget-solver qpoases \
  --retarget-max-iter 8 \
  --root-z-offset 0.05 \
  --h2-root-z-smooth-radius 3 \
  --h2-blend-frames 8 \
  --output "${output_dir}" \
  "${forward_args[@]}"

python3 "${SCRIPT_DIR}/verify_results.py" "${output_dir}"
echo "[g1-planner-to-h2] Artifacts: ${output_dir}"
if ((headless == 0)); then
  echo "[g1-planner-to-h2] Viewer closed; exported results passed validation."
fi
