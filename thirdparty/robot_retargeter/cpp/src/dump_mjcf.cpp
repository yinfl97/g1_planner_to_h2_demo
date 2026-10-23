#include <algorithm>
#include <iostream>
#include <string>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: dump_mjcf <mjcf.xml>\n";
    return 1;
  }
  const std::string path = argv[1];
  pinocchio::Model model;
  try {
    pinocchio::mjcf::buildModel(path, model);
  } catch (const std::exception& e) {
    std::cerr << "buildModel failed: " << e.what() << "\n";
    return 2;
  }
  std::cout << "file=" << path << "\n";
  std::cout << "nq=" << model.nq << " nv=" << model.nv << " njoints=" << model.njoints
            << " nframes=" << model.nframes << "\n";
  std::cout << "name=" << model.name << "\n";
  std::cout << "joints:\n";
  for (pinocchio::JointIndex i = 0; i < static_cast<pinocchio::JointIndex>(model.njoints); ++i) {
    std::cout << "  [" << i << "] " << model.names[i] << " nq=" << model.nqs[i]
              << " nv=" << model.nvs[i] << " idx_q=" << model.idx_qs[i]
              << " idx_v=" << model.idx_vs[i] << " type=" << model.joints[i].shortname()
              << "\n";
  }
  std::cout << "frames (first 80):\n";
  const int nshow = std::min(80, model.nframes);
  for (int i = 0; i < nshow; ++i) {
    const auto& f = model.frames[static_cast<std::size_t>(i)];
    std::cout << "  [" << i << "] " << f.name << " type=" << f.type << " parent=" << f.parentJoint
              << "\n";
  }
  Eigen::VectorXd q = pinocchio::neutral(model);
  std::cout << "neutral q (size=" << q.size() << "): " << q.transpose() << "\n";
  if (model.njoints > 1) {
    const pinocchio::SE3& jp = model.jointPlacements[1];
    std::cout << "jointPlacements[1] (" << model.names[1] << ") translation="
              << jp.translation().transpose() << "\n";
  }
  pinocchio::Data data(model);
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  auto print_frame = [&](const std::string& name) {
    if (!model.existFrame(name)) {
      std::cout << "  missing frame " << name << "\n";
      return;
    }
    const auto fid = model.getFrameId(name);
    std::cout << "  " << name << " @neutral z=" << data.oMf[fid].translation()[2] << "\n";
  };
  print_frame("pelvis");
  print_frame("hips_sphere");
  print_frame("left_ankle_roll_link");
  print_frame("left_ankle_pitch_link");

  // MuJoCo-style q: copy body-world z into free-flyer q.z (common mistake).
  Eigen::VectorXd q_mj = q;
  if (model.nq >= 7) {
    q_mj[2] = 0.793;
    q_mj[6] = 1.0;  // qw
    pinocchio::forwardKinematics(model, data, q_mj);
    pinocchio::updateFramePlacements(model, data);
    std::cout << "if q.z=0.793 pelvis z=" << data.oMi[1].translation()[2];
    if (model.existFrame("pelvis")) {
      std::cout << " frame_pelvis z=" << data.oMf[model.getFrameId("pelvis")].translation()[2];
    }
    std::cout << "\n";
  }
  if (model.lowerPositionLimit.size() == model.nq) {
    std::cout << "qmin hinge[7:]: " << model.lowerPositionLimit.tail(std::max(0, model.nq - 7)).transpose()
              << "\n";
    std::cout << "qmax hinge[7:]: " << model.upperPositionLimit.tail(std::max(0, model.nq - 7)).transpose()
              << "\n";
  }
  return 0;
}
