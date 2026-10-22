#!/usr/bin/env python3
"""Validate artifacts produced by the G1 planner to H2 kinematic demo."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Thresholds:
    mean_joint_rmse: float = 0.15
    mean_body_rmse: float = 0.18
    h2_root_step: float = 0.06
    h2_joint_step_rmse: float = 0.08


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.rows.append((name, condition, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def print(self) -> None:
        width = max((len(name) for name, _, _ in self.rows), default=5)
        for name, ok, detail in self.rows:
            print(f"{'PASS' if ok else 'FAIL':4}  {name:<{width}}  {detail}")
        print(f"\nRESULT: {'PASS' if self.passed else 'FAIL'}")


def _finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value {value!r}")
    return number


def _read_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _read_csv(path: Path, text_columns: Iterable[str] = ()) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("missing CSV header")
        rows = list(reader)
    text = set(text_columns)
    for row_number, row in enumerate(rows, start=2):
        for key, value in row.items():
            if key not in text:
                try:
                    _finite_number(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"row {row_number}, column {key}: {error}") from error
    return rows


def _int_field(values: dict[str, str], key: str) -> int:
    value = _finite_number(values[key])
    integer = int(value)
    if value != integer:
        raise ValueError(f"{key} must be an integer, got {value}")
    return integer


def _parse_clip_metadata(path: Path) -> dict[str, int | float]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "frames": r"Total timesteps:\s*(\d+)",
        "fps": r"fps:\s*([-+0-9.eE]+)",
        "joints": r"num_joints:\s*(\d+)",
        "bodies": r"num_bodies:\s*(\d+)",
        "quaternions": r"num_body_quat:\s*(\d+)",
    }
    parsed: dict[str, int | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing {key} in metadata")
        parsed[key] = _finite_number(match.group(1)) if key == "fps" else int(match.group(1))
    return parsed


def _validate_clip(path: Path, frames: int) -> tuple[bool, str]:
    expected_widths = {
        "joint_pos.csv": 29,
        "joint_vel.csv": 29,
        "body_pos.csv": 3,
        "body_quat.csv": 4,
        "body_lin_vel.csv": 3,
    }
    metadata_path = path / "metadata.txt"
    missing = [name for name in [*expected_widths, "metadata.txt"] if not (path / name).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)}"
    try:
        metadata = _parse_clip_metadata(metadata_path)
        if metadata["frames"] != frames or metadata["joints"] != 29:
            return False, f"metadata frames/joints={metadata['frames']}/{metadata['joints']}"
        if metadata["fps"] != 50.0 or metadata["bodies"] < 1 or metadata["quaternions"] < 1:
            return False, f"invalid metadata {metadata}"
        for filename, width in expected_widths.items():
            with (path / filename).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            if len(rows) != frames + 1:
                return False, f"{filename} has {len(rows) - 1} frames, expected {frames}"
            if any(len(row) != width for row in rows):
                return False, f"{filename} has an unexpected column count"
            for row in rows[1:]:
                for value in row:
                    _finite_number(value)
    except (OSError, ValueError) as error:
        return False, str(error)
    return True, f"{frames} frames, 29 joints, 50 Hz"


def validate_results(root: Path, thresholds: Thresholds = Thresholds()) -> Report:
    report = Report()
    required = [
        "summary.txt",
        "comparison_metrics.csv",
        "retarget_timeline.csv",
        "command_schedule.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    report.check("artifacts", not missing, "complete" if not missing else f"missing {', '.join(missing)}")
    if missing:
        return report

    try:
        summary = _read_summary(root / "summary.txt")
        required_summary = [
            "frames", "replans", "common_joints", "common_bodies",
            "control_deadline_misses", "dropped_generations",
            "mean_joint_rmse_rad", "mean_root_relative_body_rmse_m",
            "h2_max_root_step_m", "h2_max_joint_step_rmse_rad",
        ]
        absent = [key for key in required_summary if key not in summary]
        if absent:
            raise ValueError(f"summary missing {', '.join(absent)}")
        frames = _int_field(summary, "frames")
        replans = _int_field(summary, "replans")
        numbers = {key: _finite_number(summary[key]) for key in required_summary[2:]}
        if frames <= 0 or replans <= 0:
            raise ValueError("frames and replans must be positive")
        report.check("summary", True, f"{frames} frames, {replans} replans")
    except (OSError, KeyError, ValueError) as error:
        report.check("summary", False, str(error))
        return report

    correspondence_ok = numbers["common_joints"] == 29 and numbers["common_bodies"] == 29
    report.check(
        "correspondence", correspondence_ok,
        f"joints={numbers['common_joints']:.0f}, bodies={numbers['common_bodies']:.0f} (expected 29/29)",
    )
    timing_ok = numbers["control_deadline_misses"] == 0 and numbers["dropped_generations"] == 0
    report.check(
        "planner timing", timing_ok,
        f"missed_ticks={numbers['control_deadline_misses']:.0f}, dropped={numbers['dropped_generations']:.0f}",
    )

    metric_specs = [
        ("joint mapping", "mean_joint_rmse_rad", thresholds.mean_joint_rmse, "rad"),
        ("body mapping", "mean_root_relative_body_rmse_m", thresholds.mean_body_rmse, "m"),
        ("root continuity", "h2_max_root_step_m", thresholds.h2_root_step, "m"),
        ("joint continuity", "h2_max_joint_step_rmse_rad", thresholds.h2_joint_step_rmse, "rad"),
    ]
    for label, key, maximum, unit in metric_specs:
        value = numbers[key]
        report.check(label, value <= maximum, f"{value:.6f} {unit} <= {maximum:.6f} {unit}")

    try:
        comparison = _read_csv(root / "comparison_metrics.csv", ["joint_max_name"])
        sequential = [int(row["frame"]) for row in comparison] == list(range(frames))
        report.check("comparison CSV", len(comparison) == frames and sequential, f"{len(comparison)} rows")
    except (OSError, KeyError, ValueError) as error:
        report.check("comparison CSV", False, str(error))

    try:
        timeline = _read_csv(root / "retarget_timeline.csv", ["status"])
        generations = [int(row["generation"]) for row in timeline]
        statuses_ok = all(
            int(row["publish_frame"]) >= 0
            and not any(word in row["status"] for word in ("dropped", "expired", "solving"))
            for row in timeline
        )
        ticks = sum(int(row["missed_control_ticks"]) for row in timeline)
        ok = len(timeline) == replans and generations == list(range(replans)) and statuses_ok and ticks == 0
        report.check("retarget timeline", ok, f"{len(timeline)} rows, missed_ticks={ticks}")
    except (OSError, KeyError, ValueError) as error:
        report.check("retarget timeline", False, str(error))

    try:
        schedule = _read_csv(root / "command_schedule.csv", ["clip", "style", "maneuver"])
        generations = [int(row["generation"]) for row in schedule]
        contiguous = bool(schedule) and int(schedule[0]["start_frame"]) == 0
        for previous, current in zip(schedule, schedule[1:]):
            contiguous = contiguous and int(current["start_frame"]) == int(previous["end_frame"]) + 1
        contiguous = contiguous and int(schedule[-1]["end_frame"]) == frames - 1
        ok = len(schedule) == replans and generations == list(range(replans)) and contiguous
        report.check("command schedule", ok, f"{len(schedule)} contiguous rows")
    except (OSError, KeyError, ValueError) as error:
        report.check("command schedule", False, str(error))

    for clip_name in ("g1_original", "h2_retarget"):
        ok, detail = _validate_clip(root / clip_name, frames)
        report.check(clip_name, ok, detail)
    return report


def _positive_finite(value: str) -> float:
    number = _finite_number(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("threshold must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--max-mean-joint-rmse", type=_positive_finite, default=0.15)
    parser.add_argument("--max-mean-body-rmse", type=_positive_finite, default=0.18)
    parser.add_argument("--max-h2-root-step", type=_positive_finite, default=0.06)
    parser.add_argument("--max-h2-joint-step-rmse", type=_positive_finite, default=0.08)
    args = parser.parse_args(argv)
    thresholds = Thresholds(
        args.max_mean_joint_rmse,
        args.max_mean_body_rmse,
        args.max_h2_root_step,
        args.max_h2_joint_step_rmse,
    )
    report = validate_results(args.result_dir.resolve(), thresholds)
    report.print()
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
