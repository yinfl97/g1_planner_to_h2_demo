import csv
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_results import validate_results  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def make_clip(root: Path, frames: int = 2) -> None:
    root.mkdir()
    write_csv(root / "joint_pos.csv", [f"joint_{index}" for index in range(29)], [[0.0] * 29] * frames)
    write_csv(root / "joint_vel.csv", [f"joint_vel_{index}" for index in range(29)], [[0.0] * 29] * frames)
    write_csv(root / "body_pos.csv", ["body_0_x", "body_0_y", "body_0_z"], [[0.0, 0.0, 1.0]] * frames)
    write_csv(root / "body_quat.csv", ["body_0_w", "body_0_x", "body_0_y", "body_0_z"], [[1.0, 0.0, 0.0, 0.0]] * frames)
    write_csv(root / "body_lin_vel.csv", ["body_0_x", "body_0_y", "body_0_z"], [[0.0, 0.0, 0.0]] * frames)
    (root / "metadata.txt").write_text(
        "Total timesteps: 2\nfps: 50.0\nnum_joints: 29 num_bodies: 1 num_body_quat: 1\n",
        encoding="utf-8",
    )


def make_fixture(root: Path) -> None:
    (root / "summary.txt").write_text(
        "\n".join(
            [
                "H2 kinematic planner retarget comparison",
                "frames: 2",
                "replans: 1",
                "common_joints: 29",
                "common_bodies: 29",
                "control_deadline_misses: 0",
                "dropped_generations: 0",
                "mean_joint_rmse_rad: 0.08",
                "mean_root_relative_body_rmse_m: 0.12",
                "h2_max_root_step_m: 0.02",
                "h2_max_joint_step_rmse_rad: 0.03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_header = [
        "frame", "time_s", "desired_segment", "applied_generation", "applied_source_frame",
        "generation_lag", "g1_root_x", "g1_root_y", "g1_root_z", "h2_root_x",
        "h2_root_y", "h2_root_z", "root_xyz_delta_m", "root_z_delta_m",
        "root_angle_delta_rad", "joint_rmse_rad", "joint_max_rad", "joint_max_name",
        "root_relative_body_rmse_m", "left_foot_delta_m", "right_foot_delta_m",
        "left_hand_delta_m", "right_hand_delta_m",
    ]
    numeric_tail = [0.0] * 11 + ["left_knee"] + [0.0] * 5
    write_csv(
        root / "comparison_metrics.csv",
        comparison_header,
        [[0, 0.0, 0, 0, 0, 0, *numeric_tail], [1, 0.02, 0, 0, 1, 0, *numeric_tail]],
    )
    timeline_header = [
        "generation", "scheduled_frame", "publish_frame", "publish_lag_frames",
        "aligned_source_frame", "planner_ms", "solve_ms", "control_block_ms",
        "missed_control_ticks", "continuity_root_correction_m",
        "continuity_planar_correction_m", "continuity_vertical_correction_m",
        "continuity_yaw_correction_rad", "joint_anchor_rmse_before_rad",
        "joint_anchor_rmse_after_rad", "continuity_blended_frames",
        "vertical_recovery_frames", "status",
    ]
    write_csv(root / "retarget_timeline.csv", timeline_header, [[0] * 17 + ["startup_sync"]])
    schedule_header = [
        "generation", "clip", "start_frame", "end_frame", "start_time_s", "end_time_s",
        "step_in_clip", "steps_in_clip", "mode", "style", "maneuver", "speed", "height",
        "move_x", "move_y", "move_z", "face_x", "face_y", "face_z", "face_yaw_rad",
        "spin_rad_s", "body_move_angle_rad", "clip_duration_s", "idle_gap",
    ]
    write_csv(
        root / "command_schedule.csv",
        schedule_header,
        [[0, "fixed_0", 0, 1, 0.0, 0.02, 0, 1, 2, "walk", "none", 1.0, -1.0] + [0.0] * 11],
    )
    make_clip(root / "g1_original")
    make_clip(root / "h2_retarget")


class VerifyResultsTest(unittest.TestCase):
    def test_valid_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            self.assertTrue(validate_results(root).passed)

    def test_non_finite_csv_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            path = root / "h2_retarget" / "joint_pos.csv"
            path.write_text(path.read_text(encoding="utf-8").replace("0.0", "nan", 1), encoding="utf-8")
            self.assertFalse(validate_results(root).passed)

    def test_threshold_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            path = root / "summary.txt"
            path.write_text(path.read_text(encoding="utf-8").replace("mean_joint_rmse_rad: 0.08", "mean_joint_rmse_rad: 0.16"), encoding="utf-8")
            report = validate_results(root)
            self.assertFalse(report.passed)
            self.assertIn("joint mapping", [name for name, ok, _ in report.rows if not ok])

    def test_missing_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            (root / "retarget_timeline.csv").unlink()
            self.assertFalse(validate_results(root).passed)


if __name__ == "__main__":
    unittest.main()
