#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${DEMO_DEPENDENCY_BUILD_ROOT:-${REPO_ROOT}/build/deps}"
INSTALL_PREFIX="${BUILD_ROOT}/install"
BUILD_TYPE="${1:-Release}"
PARALLELISM="${2:-$(nproc)}"

declare -A pins=(
  [pinocchio]=187afafcfe22d7ac16a26241c0b13a76d04d82c1
  [osqp]=0dd00a578cf1c2691c5c379965d504c75bf6cfad
  [qpOASES]=9e40af7d170f440b7887fc4f9cf162f3f3ae24e8
  [mujoco]=13827e9ee56f097f57acf69ae52b078f9839682d
  [glfw]=7b6aead9fb88b3623e3b3725ebb42670cbe4c579
)

for dependency in pinocchio osqp qpOASES mujoco glfw; do
  path="${REPO_ROOT}/thirdparty/${dependency}"
  [[ -e "${path}/.git" ]] || {
    echo "Missing ${dependency}; run: git submodule update --init --recursive" >&2
    exit 1
  }
  actual="$(git -c safe.directory="${path}" -C "${path}" rev-parse HEAD)"
  [[ "${actual}" == "${pins[${dependency}]}" ]] || {
    echo "${dependency} is at ${actual}; expected ${pins[${dependency}]}" >&2
    exit 1
  }
done

mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"
cmake -S "${REPO_ROOT}/thirdparty/pinocchio" -B "${BUILD_ROOT}/pinocchio" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/ros/humble}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DBUILD_PYTHON_INTERFACE=OFF \
  -DBUILD_TESTING=OFF -DBUILD_DOCUMENTATION=OFF -DINSTALL_DOCUMENTATION=OFF \
  -DBUILD_BENCHMARK=OFF -DBUILD_UTILS=OFF -DBUILD_WITH_URDF_SUPPORT=ON \
  -DBUILD_WITH_COLLISION_SUPPORT=OFF -DBUILD_WITH_AUTODIFF_SUPPORT=OFF \
  -DBUILD_WITH_CODEGEN_SUPPORT=OFF
cmake --build "${BUILD_ROOT}/pinocchio" --target install -j"${PARALLELISM}"

cmake -S "${REPO_ROOT}/thirdparty/osqp" -B "${BUILD_ROOT}/osqp" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DPYTHON=OFF -DMATLAB=OFF -DR_LANG=OFF \
  -DUNITTESTS=OFF -DPROFILING=OFF -DENABLE_MKL_PARDISO=OFF
cmake --build "${BUILD_ROOT}/osqp" --target install -j"${PARALLELISM}"

cmake -S "${REPO_ROOT}/thirdparty/mujoco" -B "${BUILD_ROOT}/mujoco" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=OFF -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_STUDIO=OFF \
  -DMUJOCO_USE_FILAMENT=OFF -DMUJOCO_WITH_USD=OFF
cmake --build "${BUILD_ROOT}/mujoco" --target install -j"${PARALLELISM}"

echo "Pinned dependencies installed in ${INSTALL_PREFIX}"
