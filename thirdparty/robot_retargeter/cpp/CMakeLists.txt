cmake_minimum_required(VERSION 3.16)
project(g1_to_h2_retargeter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)
find_package(pinocchio 3.4.0 EXACT CONFIG REQUIRED)
find_package(osqp CONFIG REQUIRED)

option(ROBOT_RETARGETER_ENABLE_QPOASES
  "Enable the experimental qpOASES QP backend" OFF)
set(ROBOT_RETARGETER_QPOASES_ROOT
  ""
  CACHE PATH "Path to a pinned qpOASES source checkout")
if(ROBOT_RETARGETER_ENABLE_QPOASES)
  if(NOT EXISTS "${ROBOT_RETARGETER_QPOASES_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
      "ROBOT_RETARGETER_QPOASES_ROOT must point to qpOASES source at pinned "
      "commit 7f26b3247b90dbb9c0ed6c5dd18883da35e0ec72")
  endif()
  set(QPOASES_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    "${ROBOT_RETARGETER_QPOASES_ROOT}"
    "${CMAKE_CURRENT_BINARY_DIR}/qpoases")
  # qpOASES 3.2.2 emits legacy snprintf diagnostics under modern GCC. Keep
  # third-party warnings out of the retargeter build without changing ours.
  target_compile_options(qpOASES PRIVATE -w)
endif()

add_library(g1_to_h2_retargeter STATIC
  src/g1_to_h2.cpp
  src/npy_io.cpp
)
target_include_directories(g1_to_h2_retargeter
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_definitions(g1_to_h2_retargeter PRIVATE
  ROBOT_RETARGETER_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
if(ROBOT_RETARGETER_ENABLE_QPOASES)
  target_include_directories(g1_to_h2_retargeter PRIVATE
    "${ROBOT_RETARGETER_QPOASES_ROOT}/include")
  target_compile_definitions(g1_to_h2_retargeter PRIVATE
    ROBOT_RETARGETER_HAS_QPOASES=1)
endif()
target_link_libraries(g1_to_h2_retargeter
  PUBLIC
    pinocchio::pinocchio
    Eigen3::Eigen
    osqp::osqpstatic
)
if(ROBOT_RETARGETER_ENABLE_QPOASES)
  target_link_libraries(g1_to_h2_retargeter PUBLIC qpOASES)
endif()
target_compile_options(g1_to_h2_retargeter PRIVATE -O3 -ffast-math)

add_executable(g1_to_h2_cli src/cli.cpp)
target_link_libraries(g1_to_h2_cli PRIVATE g1_to_h2_retargeter)
set_target_properties(g1_to_h2_cli PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

add_executable(dump_mjcf src/dump_mjcf.cpp)
target_link_libraries(dump_mjcf PRIVATE pinocchio::pinocchio)
set_target_properties(dump_mjcf PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

install(TARGETS g1_to_h2_retargeter g1_to_h2_cli
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
