cmake_minimum_required(VERSION 3.22)
project(g1_planner_to_h2_demo VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEMO_DEPENDENCY_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/build/deps/install"
    CACHE PATH "Install prefix populated by scripts/bootstrap_dependencies.sh")
list(PREPEND CMAKE_PREFIX_PATH "${DEMO_DEPENDENCY_PREFIX}")
if(EXISTS "/opt/ros/humble")
  list(APPEND CMAKE_PREFIX_PATH "/opt/ros/humble")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

foreach(config IN ITEMS
    "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/pinocchio/pinocchioConfig.cmake"
    "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/osqp/osqp-config.cmake"
    "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/mujoco/mujocoConfig.cmake")
  if(NOT EXISTS "${config}")
    message(FATAL_ERROR
      "Pinned dependencies are missing from ${DEMO_DEPENDENCY_PREFIX}. "
      "Run scripts/bootstrap_dependencies.sh first.")
  endif()
endforeach()

set(pinocchio_DIR "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/pinocchio" CACHE PATH "" FORCE)
set(osqp_DIR "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/osqp" CACHE PATH "" FORCE)
set(mujoco_DIR "${DEMO_DEPENDENCY_PREFIX}/lib/cmake/mujoco" CACHE PATH "" FORCE)

find_package(Boost REQUIRED COMPONENTS filesystem serialization)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(ZLIB REQUIRED)
find_package(pinocchio 3.4.0 EXACT CONFIG REQUIRED)
find_package(osqp CONFIG REQUIRED)
find_package(mujoco 3.12.0 EXACT CONFIG REQUIRED)
find_package(spdlog REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glfw EXCLUDE_FROM_ALL)

set(QPOASES_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/qpOASES EXCLUDE_FROM_ALL)
target_compile_options(qpOASES PRIVATE -w)

add_library(g1_to_h2_retargeter STATIC
  thirdparty/robot_retargeter/cpp/src/g1_to_h2.cpp
  thirdparty/robot_retargeter/cpp/src/npy_io.cpp)
target_include_directories(g1_to_h2_retargeter PUBLIC
  thirdparty/robot_retargeter/cpp/include
  thirdparty/qpOASES/include)
target_compile_definitions(g1_to_h2_retargeter PRIVATE
  ROBOT_RETARGETER_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/robot_retargeter"
  ROBOT_RETARGETER_HAS_QPOASES=1)
target_compile_options(g1_to_h2_retargeter PRIVATE -O3 -ffast-math)
target_link_libraries(g1_to_h2_retargeter PUBLIC
  pinocchio::pinocchio Eigen3::Eigen osqp::osqpstatic qpOASES)

add_library(TRTInference STATIC
  src/TRTInference/InferenceEngine.cpp
  src/TRTInference/InferenceEngine.h
  src/TRTInference/InferenceEngine.inl
  src/TRTInference/Utility.h)
target_include_directories(TRTInference PUBLIC src ${TensorRT_INCLUDE_DIRS})
target_compile_options(TRTInference PRIVATE -Wno-deprecated-declarations)
target_link_libraries(TRTInference PUBLIC
  ${TensorRT_nvinfer_LIBRARY}
  ${TensorRT_nvonnxparser_LIBRARY}
  CUDA::cudart)

add_library(cnpy STATIC src/cnpy.cpp)
target_include_directories(cnpy PUBLIC include)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

add_executable(g1_planner_to_h2_demo src/g1_planner_to_h2_demo.cpp)
target_include_directories(g1_planner_to_h2_demo PRIVATE include src)
target_compile_definitions(g1_planner_to_h2_demo PRIVATE
  G1_PLANNER_TO_H2_DEMO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(g1_planner_to_h2_demo PRIVATE -O3 -ffast-math)
target_link_libraries(g1_planner_to_h2_demo PRIVATE
  TRTInference cnpy g1_to_h2_retargeter mujoco::mujoco glfw
  spdlog::spdlog OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads)

add_executable(g1_planner_to_h2_renderer src/render_video.cpp)
target_include_directories(g1_planner_to_h2_renderer PRIVATE include)
target_link_libraries(g1_planner_to_h2_renderer PRIVATE
  mujoco::mujoco OpenGL::EGL ${CMAKE_DL_LIBS})

set_target_properties(g1_planner_to_h2_demo g1_planner_to_h2_renderer PROPERTIES
  BUILD_RPATH "${DEMO_DEPENDENCY_PREFIX}/lib"
  INSTALL_RPATH "$ORIGIN")
