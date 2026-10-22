#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"
CONTAINER_NAME="${SONIC_SIM_CONTAINER:-sonic_sim_env}"
CONTAINER_REPO="/root/workspace/src/g1_planner_to_h2_demo"

fail() {
  echo "[h2-video] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/record_video.sh [recording options] [planner options]

Offline-generates and validates the G1 -> H2 clips, renders both robots with
headless MuJoCo/EGL, and saves an H.264 MP4 through the host FFmpeg.

By default it records a deterministic 30-second continuous demo: five random
locomotion styles/maneuvers, six seconds each, with 5 Hz replanning.

Recording options:
  --output DIR       Result directory (default: output/<timestamp>)
  --video FILE       MP4 path (default: RESULT_DIR/g1_to_h2.mp4)
  --width N          Video width, even and >= 640 (default: 1280)
  --height N         Video height, even and >= 360 (default: 720)
  --fps N            Playback/encoding rate (default: 50)
  --crf N            libx264 quality, 0-51; lower is better (default: 18)
  --max-frames N     Render only the first N frames (0 means all)
  --fixed-demo       Disable the 30-second random profile

Other options are forwarded to g1_planner_to_h2_demo, for example --replans 1,
--mode walk, and --speed 0.5. Passing one of those fixed-command options also
disables the default random profile.
EOF
}

width=1280
height=720
fps=50
crf=18
max_frames=0
fixed_demo=0
output_dir=""
video_path=""
g1_mjcf="${G1_H2_G1_MJCF:-${REPO_ROOT}/assets/g1/scene_29dof.xml}"
h2_mjcf="${G1_H2_H2_MJCF:-${REPO_ROOT}/assets/h2/scene_h2_29dof.xml}"
planner_args=()
while (($#)); do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --output|--video|--width|--height|--fps|--crf|--max-frames)
      (($# >= 2)) || fail "$1 requires a value"
      option="$1"
      value="$2"
      shift 2
      case "${option}" in
        --output) output_dir="${value}" ;;
        --video) video_path="${value}" ;;
        --width) width="${value}" ;;
        --height) height="${value}" ;;
        --fps) fps="${value}" ;;
        --crf) crf="${value}" ;;
        --max-frames) max_frames="${value}" ;;
      esac
      ;;
    --output=*|--video=*|--width=*|--height=*|--fps=*|--crf=*|--max-frames=*)
      option="${1%%=*}"
      value="${1#*=}"
      shift
      case "${option}" in
        --output) output_dir="${value}" ;;
        --video) video_path="${value}" ;;
        --width) width="${value}" ;;
        --height) height="${value}" ;;
        --fps) fps="${value}" ;;
        --crf) crf="${value}" ;;
        --max-frames) max_frames="${value}" ;;
      esac
      ;;
    --headless)
      shift
      ;;
    --fixed-demo)
      fixed_demo=1
      shift
      ;;
    --g1-mjcf|--h2-mjcf)
      (($# >= 2)) || fail "$1 requires a path"
      option="$1"
      value="$2"
      planner_args+=("${option}" "${value}")
      [[ "${option}" == "--g1-mjcf" ]] && g1_mjcf="${value}" || h2_mjcf="${value}"
      shift 2
      ;;
    --g1-mjcf=*|--h2-mjcf=*)
      option="${1%%=*}"
      value="${1#*=}"
      planner_args+=("${option}" "${value}")
      [[ "${option}" == "--g1-mjcf" ]] && g1_mjcf="${value}" || h2_mjcf="${value}"
      shift
      ;;
    *)
      case "$1" in
        --mode|--mode=*|--replans|--replans=*|--speed|--speed=*|--move|--move=*|--face|--face=*)
          fixed_demo=1
          ;;
      esac
      planner_args+=("$1")
      shift
      ;;
  esac
done

[[ "${width}" =~ ^[0-9]+$ && "${height}" =~ ^[0-9]+$ ]] || \
  fail "--width and --height must be integers"
((width >= 640 && height >= 360 && width % 2 == 0 && height % 2 == 0)) || \
  fail "video dimensions must be even and at least 640x360"
[[ "${fps}" =~ ^[1-9][0-9]*$ ]] || fail "--fps must be a positive integer"
[[ "${crf}" =~ ^[0-9]+$ ]] && ((crf <= 51)) || fail "--crf must be in [0, 51]"
[[ "${max_frames}" =~ ^[0-9]+$ ]] || fail "--max-frames must be non-negative"

command -v docker >/dev/null 2>&1 || fail "docker is required"
command -v ffmpeg >/dev/null 2>&1 || fail "host ffmpeg is required"
command -v ffprobe >/dev/null 2>&1 || fail "host ffprobe is required"
running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
[[ "${running}" == "true" ]] || fail \
  "${CONTAINER_NAME} is not running; start it with: ioenv run sonic_sim"

if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/output/$(date +%Y%m%d_%H%M%S)"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${REPO_ROOT}/${output_dir}"
fi
if [[ "${output_dir}" != "${WORKSPACE_ROOT}" && "${output_dir}" != "${WORKSPACE_ROOT}/"* ]]; then
  fail "--output must be inside workspace ${WORKSPACE_ROOT}"
fi
if [[ "${output_dir}" == "${WORKSPACE_ROOT}" ]]; then
  container_output="/root/workspace"
else
  container_output="/root/workspace/${output_dir#"${WORKSPACE_ROOT}/"}"
fi

to_container_path() {
  local path="$1"
  [[ "${path}" == /* ]] || path="${REPO_ROOT}/${path}"
  if [[ "${path}" == "${WORKSPACE_ROOT}" ]]; then
    echo /root/workspace
  elif [[ "${path}" == "${WORKSPACE_ROOT}/"* ]]; then
    echo "/root/workspace/${path#"${WORKSPACE_ROOT}/"}"
  elif [[ "${path}" == /root/workspace || "${path}" == /root/workspace/* ]]; then
    echo "${path}"
  else
    fail "asset path must be inside workspace ${WORKSPACE_ROOT}: ${path}"
  fi
}
container_g1_mjcf="$(to_container_path "${g1_mjcf}")"
container_h2_mjcf="$(to_container_path "${h2_mjcf}")"

if [[ -z "${video_path}" ]]; then
  video_path="${output_dir}/g1_to_h2.mp4"
elif [[ "${video_path}" != /* ]]; then
  video_path="${REPO_ROOT}/${video_path}"
fi

prepare_host_directory() {
  local host_dir="$1"
  if mkdir -p "${host_dir}" 2>/dev/null && [[ -w "${host_dir}" ]]; then
    return
  fi
  local container_dir
  if [[ "${host_dir}" == "${WORKSPACE_ROOT}" ]]; then
    container_dir="/root/workspace"
  elif [[ "${host_dir}" == "${WORKSPACE_ROOT}/"* ]]; then
    container_dir="/root/workspace/${host_dir#"${WORKSPACE_ROOT}/"}"
  else
    fail "cannot create host directory: ${host_dir}"
  fi
  docker exec "${CONTAINER_NAME}" mkdir -p "${container_dir}"
  docker exec "${CONTAINER_NAME}" chown "$(id -u):$(id -g)" "${container_dir}"
  [[ -w "${host_dir}" ]] || fail "host directory is not writable: ${host_dir}"
}

prepare_host_directory "${output_dir}"
prepare_host_directory "$(dirname -- "${video_path}")"

demo_args=()
if ((fixed_demo == 0)); then
  demo_args=(
    --random-locomotion
    --random-clips 5
    --random-duration-min 6
    --random-duration-max 6
    --replan-frames 10
    --random-speed-max 1.2
  )
  echo "[h2-video] Profile: 30 s continuous random locomotion (5 x 6 s)."
else
  echo "[h2-video] Profile: explicit/fixed planner command schedule."
fi

echo "[h2-video] Generating and validating kinematic clips..."
"${SCRIPT_DIR}/run_demo.sh" --headless --output "${output_dir}" \
  "${demo_args[@]}" "${planner_args[@]}"

renderer="${CONTAINER_REPO}/build/g1_planner_to_h2_renderer"
if ! docker exec "${CONTAINER_NAME}" test -x "${renderer}"; then
  echo "[h2-video] Building standalone EGL video renderer..."
  docker exec -i "${CONTAINER_NAME}" bash -lc \
    "cd '${CONTAINER_REPO}' && scripts/bootstrap_dependencies.sh Release && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
fi
docker exec "${CONTAINER_NAME}" "${renderer}" --help >/dev/null || \
  fail "video renderer is not runnable in ${CONTAINER_NAME}"

echo "[h2-video] Rendering ${width}x${height} RGB frames with MuJoCo/EGL..."
echo "[h2-video] Encoding H.264 MP4: ${video_path}"
docker exec -i "${CONTAINER_NAME}" "${renderer}" \
  --g1-mjcf "${container_g1_mjcf}" \
  --h2-mjcf "${container_h2_mjcf}" \
  --g1-clip "${container_output}/g1_original" \
  --h2-clip "${container_output}/h2_retarget" \
  --schedule "${container_output}/command_schedule.csv" \
  --width "${width}" --height "${height}" --fps "${fps}" \
  --max-frames "${max_frames}" |
  ffmpeg -hide_banner -loglevel warning -y \
    -f rawvideo -pixel_format rgb24 -video_size "${width}x${height}" \
    -framerate "${fps}" -i pipe:0 -an -c:v libx264 -preset medium \
    -crf "${crf}" -pix_fmt yuv420p -movflags +faststart "${video_path}"

[[ -s "${video_path}" ]] || fail "FFmpeg did not create a non-empty video"
echo "[h2-video] Video probe:"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 "${video_path}"
echo "[h2-video] Saved: ${video_path}"
