#!/usr/bin/env python3
"""Run a speed-sweep comparison study for two ONNX policies.

The study evaluates each policy under the same map, seeds, speed caps, and
actuator model, then writes report-ready JSON/CSV/Markdown summaries plus a few
plots. It is intentionally close to evaluate_onnx_with_actuator.py, but keeps
per-step data so high-speed behavior can be quantified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
for parent in project_root.parents:
    gym_parent = parent / "f1tenth_gym_ros" / "f1tenth_gym"
    if gym_parent.exists():
        sys.path.insert(0, str(gym_parent))
        break


DEFAULT_ORIGINAL = project_root / "runs/levine_slam_ppo_cir_2026-04-28_11-14-54/final_model.onnx"
DEFAULT_FINETUNED = project_root / "runs/levine_slam_ppo_cir_actuator_ft_2026-05-10_11-25-53/final_model.onnx"
DEFAULT_ACTUATOR_DIR = (
    project_root.parent
    / "f1tenth-project/f1tenth_adaptive_server/offline_actuator_weights/offline_actuator_retrained_20260510"
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def apply_overrides(config: dict[str, Any], args: argparse.Namespace, speed_cap: float, actuator: bool) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("env", {})
    config.setdefault("experiment", {})
    config.setdefault("action", {})
    config.setdefault("expert", {}).setdefault("pure_pursuit", {})

    config["env"]["map_path"] = args.map
    config["env"]["num_envs"] = 1
    config["env"]["max_steps"] = args.max_steps
    config["env"]["randomize_direction"] = False
    config["experiment"]["device"] = args.device
    config["action"]["max_speed"] = float(speed_cap)
    config["expert"]["pure_pursuit"]["target_speed"] = float(speed_cap)

    config.setdefault("domain_randomization", {})
    config["domain_randomization"]["enabled"] = False
    config["domain_randomization"]["mode"] = "off"

    if actuator:
        config["actuator_model"] = {
            "model_path": str(args.actuator_model.resolve()),
            "scaler_X_path": str(args.actuator_scaler_x.resolve()),
            "scaler_y_path": str(args.actuator_scaler_y.resolve()),
            "history_steps": int(args.actuator_history),
        }
    else:
        config.pop("actuator_model", None)
    return config


def make_onnx_policy(onnx_path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    def predict(obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        outputs = session.run(output_names, {input_name: obs_batch})
        action = np.asarray(outputs[0])
        action = np.squeeze(action).astype(np.float32)
        if action.shape[0] > 2:
            action = action[:2]
        return np.clip(action, -1.0, 1.0)

    return predict, {
        "input_name": input_name,
        "input_shape": session.get_inputs()[0].shape,
        "outputs": output_names,
    }


def finite_stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    return finite_stats(values)


def nearest_centerline_dist(track, x: float, y: float) -> float:
    centerline = getattr(track, "centerline", None)
    if centerline is None:
        return float("nan")
    xs = np.asarray(centerline.xs, dtype=np.float64)
    ys = np.asarray(centerline.ys, dtype=np.float64)
    if xs.size == 0:
        return float("nan")
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    return float(np.sqrt(np.min(d2)))


def summarize_episode_steps(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    speed = [r["speed"] for r in step_rows]
    cmd_speed = [r["cmd_speed"] for r in step_rows]
    steer = [r["steer"] for r in step_rows]
    cte = [r["cross_track_error"] for r in step_rows]
    yaw_rate = [r["yaw_rate"] for r in step_rows]
    lat_vel = [r["lateral_vel"] for r in step_rows]
    delta_speed = [r["actuator_delta_speed"] for r in step_rows if r["actuator_delta_speed"] is not None]
    delta_yaw = [r["actuator_delta_yaw_rate"] for r in step_rows if r["actuator_delta_yaw_rate"] is not None]
    delta_lat = [r["actuator_delta_lateral_vel"] for r in step_rows if r["actuator_delta_lateral_vel"] is not None]
    steer_diff = np.diff(np.asarray(steer, dtype=np.float64)) if len(steer) > 1 else np.asarray([])
    speed_arr = np.asarray(speed, dtype=np.float64)
    high_speed_fraction = float(np.mean(speed_arr >= 0.85 * np.max(speed_arr))) if speed_arr.size else 0.0

    return {
        "avg_speed": float(np.mean(speed)) if speed else 0.0,
        "max_speed": float(np.max(speed)) if speed else 0.0,
        "avg_cmd_speed": float(np.mean(cmd_speed)) if cmd_speed else 0.0,
        "avg_abs_steer": float(np.mean(np.abs(steer))) if steer else 0.0,
        "rms_steer_rate": float(np.sqrt(np.mean(steer_diff ** 2))) if steer_diff.size else 0.0,
        "rms_cross_track_error": float(np.sqrt(np.nanmean(np.asarray(cte) ** 2))) if cte else 0.0,
        "p95_cross_track_error": float(np.nanpercentile(cte, 95)) if cte else 0.0,
        "max_cross_track_error": float(np.nanmax(cte)) if cte else 0.0,
        "avg_abs_yaw_rate": float(np.mean(np.abs(yaw_rate))) if yaw_rate else 0.0,
        "avg_abs_lateral_vel": float(np.mean(np.abs(lat_vel))) if lat_vel else 0.0,
        "mean_abs_actuator_delta_speed": float(np.mean(np.abs(delta_speed))) if delta_speed else None,
        "mean_abs_actuator_delta_yaw_rate": float(np.mean(np.abs(delta_yaw))) if delta_yaw else None,
        "mean_abs_actuator_delta_lateral_vel": float(np.mean(np.abs(delta_lat))) if delta_lat else None,
        "high_speed_fraction": high_speed_fraction,
    }


def run_one(
    policy_name: str,
    policy,
    onnx_path: Path,
    base_config: dict[str, Any],
    args: argparse.Namespace,
    speed_cap: float,
    actuator: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from f1tenth_rl.envs.wrapper import F1TenthWrapper

    config = apply_overrides(base_config, args, speed_cap, actuator)
    env = F1TenthWrapper(config, render_mode=None)
    control_every = int(getattr(env, "control_update_steps", 1))
    policy_every = 1 if args.policy_every_step else max(1, control_every)
    track = env.base_env.unwrapped.track

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            last_action = None
            info: dict[str, Any] = {}
            ep_steps: list[dict[str, Any]] = []

            while not done:
                step_idx = int(info.get("step", 0))
                if last_action is None or step_idx % policy_every == 0:
                    last_action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(last_action)
                total_reward += float(reward)
                done = bool(terminated or truncated)

                raw = info.get("raw_obs", {})
                x = float(raw.get("poses_x", [0.0])[0])
                y = float(raw.get("poses_y", [0.0])[0])
                physical = np.asarray(info.get("physical_action", [0.0, 0.0]), dtype=np.float64)
                row = {
                    "policy": policy_name,
                    "onnx": str(onnx_path),
                    "actuator": actuator,
                    "speed_cap": float(speed_cap),
                    "episode": ep + 1,
                    "seed": seed,
                    "step": int(info.get("step", 0)),
                    "time_s": float(info.get("step", 0)) * float(config["env"].get("timestep", 0.01)),
                    "progress": float(info.get("progress", 0.0)),
                    "collision": bool(info.get("ego_collision", False)),
                    "x": x,
                    "y": y,
                    "speed": float(info.get("ego_speed", 0.0)),
                    "cmd_speed": float(physical[1]),
                    "steer": float(physical[0]),
                    "yaw_rate": float(raw.get("ang_vels_z", [0.0])[0]),
                    "lateral_vel": float(raw.get("linear_vels_y", [0.0])[0]),
                    "cross_track_error": nearest_centerline_dist(track, x, y),
                    "actuator_warmed_up": bool(info.get("actuator_warmed_up", False)),
                    "actuator_delta_speed": info.get("actuator_delta_speed"),
                    "actuator_delta_yaw_rate": info.get("actuator_delta_yaw_rate"),
                    "actuator_delta_lateral_vel": info.get("actuator_delta_lateral_vel"),
                }
                ep_steps.append(row)
                step_rows.append(row)

            ep_summary = summarize_episode_steps(ep_steps)
            final_progress = float(info.get("progress", 0.0))
            ep_row = {
                "policy": policy_name,
                "onnx": str(onnx_path),
                "actuator": actuator,
                "speed_cap": float(speed_cap),
                "episode": ep + 1,
                "seed": seed,
                "return": total_reward,
                "steps": int(info.get("step", 0)),
                "progress": final_progress,
                "progress_per_second": final_progress / max(float(info.get("step", 0)) * float(config["env"].get("timestep", 0.01)), 1e-9),
                "collision": bool(info.get("ego_collision", False)),
                "lap_time": float(info.get("ego_lap_time", 0.0)),
                **ep_summary,
            }
            episode_rows.append(ep_row)
            print(
                f"{policy_name:10s} cap={speed_cap:.1f} ep={ep + 1:02d}/{args.episodes} "
                f"progress={ep_row['progress']:.2f} avg_v={ep_row['avg_speed']:.2f} "
                f"cte_rms={ep_row['rms_cross_track_error']:.3f} "
                f"{'CRASH' if ep_row['collision'] else 'OK'}"
            )
    finally:
        env.close()

    run_meta = {
        "policy": policy_name,
        "onnx": str(onnx_path),
        "actuator": actuator,
        "speed_cap": float(speed_cap),
        "policy_every_steps": policy_every,
    }
    return episode_rows, step_rows, run_meta


def aggregate_episodes(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["policy"], float(row["speed_cap"]), bool(row["actuator"]))].append(row)

    metric_names = [
        "return",
        "steps",
        "progress",
        "progress_per_second",
        "lap_time",
        "avg_speed",
        "max_speed",
        "avg_cmd_speed",
        "avg_abs_steer",
        "rms_steer_rate",
        "rms_cross_track_error",
        "p95_cross_track_error",
        "max_cross_track_error",
        "avg_abs_yaw_rate",
        "avg_abs_lateral_vel",
        "mean_abs_actuator_delta_speed",
        "mean_abs_actuator_delta_yaw_rate",
        "mean_abs_actuator_delta_lateral_vel",
        "high_speed_fraction",
    ]
    out = []
    for (policy, speed_cap, actuator), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        agg = {
            "policy": policy,
            "speed_cap": speed_cap,
            "actuator": actuator,
            "episodes": len(rows),
            "collision_rate": float(np.mean([float(r["collision"]) for r in rows])),
            "success_rate": float(np.mean([not bool(r["collision"]) and r["progress"] >= 0.95 for r in rows])),
        }
        for name in metric_names:
            stats = scalar_summary([r[name] for r in rows if r.get(name) is not None])
            agg[f"{name}_mean"] = stats["mean"]
            agg[f"{name}_std"] = stats["std"]
        out.append(agg)
    return out


def aggregate_speed_bins(step_rows: list[dict[str, Any]], bins: list[float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, bool, str], list[dict[str, Any]]] = defaultdict(list)
    edges = np.asarray(bins, dtype=np.float64)
    for row in step_rows:
        speed = float(row["speed"])
        idx = int(np.searchsorted(edges, speed, side="right") - 1)
        idx = max(0, min(idx, len(edges) - 2))
        label = f"{edges[idx]:.1f}-{edges[idx + 1]:.1f}"
        grouped[(row["policy"], float(row["speed_cap"]), bool(row["actuator"]), label)].append(row)

    out = []
    for (policy, speed_cap, actuator, label), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0], x[0][3])):
        speeds = [r["speed"] for r in rows]
        cte = [r["cross_track_error"] for r in rows]
        steer = [abs(r["steer"]) for r in rows]
        delta_speed = [abs(r["actuator_delta_speed"]) for r in rows if r["actuator_delta_speed"] is not None]
        delta_yaw = [abs(r["actuator_delta_yaw_rate"]) for r in rows if r["actuator_delta_yaw_rate"] is not None]
        delta_lat = [abs(r["actuator_delta_lateral_vel"]) for r in rows if r["actuator_delta_lateral_vel"] is not None]
        out.append({
            "policy": policy,
            "speed_cap": speed_cap,
            "actuator": actuator,
            "speed_bin": label,
            "samples": len(rows),
            "mean_speed": finite_stats(speeds)["mean"],
            "rms_cross_track_error": float(np.sqrt(np.nanmean(np.asarray(cte) ** 2))) if cte else None,
            "p95_cross_track_error": finite_stats(cte)["p95"],
            "mean_abs_steer": finite_stats(steer)["mean"],
            "mean_abs_actuator_delta_speed": finite_stats(delta_speed)["mean"],
            "mean_abs_actuator_delta_yaw_rate": finite_stats(delta_yaw)["mean"],
            "mean_abs_actuator_delta_lateral_vel": finite_stats(delta_lat)["mean"],
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.{digits}f}"
    return str(x)


def write_markdown(path: Path, aggregate_rows: list[dict[str, Any]], speed_bin_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    policies = sorted({r["policy"] for r in aggregate_rows})
    speed_caps = sorted({float(r["speed_cap"]) for r in aggregate_rows})
    by_key = {(r["policy"], float(r["speed_cap"]), bool(r["actuator"])): r for r in aggregate_rows}

    lines = [
        "# ONNX Policy Comparison With Learned Actuator Dynamics",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        f"- Episodes per policy/speed: {args.episodes}",
        f"- Max steps per episode: {args.max_steps}",
        f"- Map: `{args.map}`",
        f"- Actuator model: `{args.actuator_model}`",
        "",
        "## Episode-Level Results",
        "",
        "| Speed cap | Policy | Collision rate | Progress mean | Lap time mean | Avg speed | CTE RMS | P95 CTE | Mean abs delta speed | Mean abs delta yaw |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for speed_cap in speed_caps:
        for policy in policies:
            row = by_key.get((policy, speed_cap, True))
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join([
                    fmt(speed_cap, 1),
                    policy,
                    fmt(row["collision_rate"]),
                    fmt(row["progress_mean"]),
                    fmt(row["lap_time_mean"]),
                    fmt(row["avg_speed_mean"]),
                    fmt(row["rms_cross_track_error_mean"]),
                    fmt(row["p95_cross_track_error_mean"]),
                    fmt(row["mean_abs_actuator_delta_speed_mean"]),
                    fmt(row["mean_abs_actuator_delta_yaw_rate_mean"]),
                ])
                + " |"
            )

    lines += [
        "",
        "## Key Findings",
        "",
    ]
    for speed_cap in speed_caps:
        original = by_key.get(("original", speed_cap, True))
        tuned = by_key.get(("actuator_ft", speed_cap, True))
        if original is None or tuned is None:
            continue
        orig_cte = original.get("rms_cross_track_error_mean")
        tuned_cte = tuned.get("rms_cross_track_error_mean")
        orig_p95 = original.get("p95_cross_track_error_mean")
        tuned_p95 = tuned.get("p95_cross_track_error_mean")
        orig_steer_rate = original.get("rms_steer_rate_mean")
        tuned_steer_rate = tuned.get("rms_steer_rate_mean")
        if not orig_cte or not tuned_cte:
            continue
        cte_reduction = 100.0 * (orig_cte - tuned_cte) / orig_cte
        p95_reduction = 100.0 * (orig_p95 - tuned_p95) / orig_p95 if orig_p95 else None
        steer_reduction = (
            100.0 * (orig_steer_rate - tuned_steer_rate) / orig_steer_rate
            if orig_steer_rate
            else None
        )
        lines.append(
            f"- At speed cap {fmt(speed_cap, 1)} m/s, actuator fine-tuning reduced RMS cross-track error "
            f"from {fmt(orig_cte)} m to {fmt(tuned_cte)} m ({fmt(cte_reduction, 1)}%). "
            f"P95 cross-track error changed from {fmt(orig_p95)} m to {fmt(tuned_p95)} m "
            f"({fmt(p95_reduction, 1)}%), and RMS steering-rate changed by {fmt(steer_reduction, 1)}%."
        )

    lines += [
        "",
        "## High-Speed Interpretation",
        "",
        "Use the speed sweep to support the report claim in three ways:",
        "",
        "1. Collision/success vs. speed cap shows whether a policy remains reliable as the requested operating envelope rises.",
        "2. Cross-track error and steering-rate metrics show stability, not just whether the car eventually completes the lap.",
        "3. Speed-binned actuator deltas show where ideal simulator dynamics diverge from learned real-car response; the high-speed bins are the clearest evidence that dynamics-aware training matters.",
        "",
        "## Speed-Binned Actuator/Dynamics Metrics",
        "",
        "| Speed cap | Policy | Speed bin | Samples | Mean speed | CTE RMS | P95 CTE | Mean abs delta speed | Mean abs delta yaw |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speed_bin_rows:
        lines.append(
            "| "
            + " | ".join([
                fmt(row["speed_cap"], 1),
                row["policy"],
                row["speed_bin"],
                str(row["samples"]),
                fmt(row["mean_speed"]),
                fmt(row["rms_cross_track_error"]),
                fmt(row["p95_cross_track_error"]),
                fmt(row["mean_abs_actuator_delta_speed"]),
                fmt(row["mean_abs_actuator_delta_yaw_rate"]),
            ])
            + " |"
        )

    path.write_text("\n".join(lines) + "\n")


def make_plots(out_dir: Path, aggregate_rows: list[dict[str, Any]], speed_bin_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return

    policies = sorted({r["policy"] for r in aggregate_rows})
    colors = {"original": "#4C78A8", "actuator_ft": "#F58518"}

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(7.0, 4.2))
        for policy in policies:
            rows = sorted([r for r in aggregate_rows if r["policy"] == policy and r["actuator"]], key=lambda r: r["speed_cap"])
            xs = [r["speed_cap"] for r in rows]
            ys = [r.get(f"{metric}_mean") for r in rows]
            es = [r.get(f"{metric}_std") or 0.0 for r in rows]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=policy, color=colors.get(policy))
        plt.xlabel("Commanded speed cap (m/s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()

    plot_metric("progress", "Episode progress (laps)", "progress_vs_speed_cap.png")
    plot_metric("rms_cross_track_error", "RMS cross-track error (m)", "cte_vs_speed_cap.png")
    plot_metric("mean_abs_actuator_delta_yaw_rate", "Mean |actuator yaw-rate delta| (rad/s)", "actuator_yaw_delta_vs_speed_cap.png")
    plot_metric("rms_steer_rate", "RMS steering command change (rad/step)", "steer_rate_vs_speed_cap.png")

    plt.figure(figsize=(7.0, 4.2))
    for policy in policies:
        rows = [r for r in speed_bin_rows if r["policy"] == policy and r["actuator"]]
        labels = []
        values = []
        for row in rows:
            labels.append(f"{row['speed_cap']:.1f}:{row['speed_bin']}")
            values.append(row["mean_abs_actuator_delta_yaw_rate"])
        x = np.arange(len(values))
        plt.plot(x, values, marker="o", label=policy, color=colors.get(policy))
    plt.xticks([])
    plt.xlabel("Speed-cap / actual-speed bins")
    plt.ylabel("Mean |actuator yaw-rate delta| (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speed_binned_actuator_yaw_delta.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-onnx", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--finetuned-onnx", type=Path, default=DEFAULT_FINETUNED)
    parser.add_argument("--config", type=Path, default=project_root / "configs/sim2real_e2e.yaml")
    parser.add_argument("--map", type=str, default="maps/levine_slam/levine_slam")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--speed-caps", type=float, nargs="+", default=[3.5, 4.5, 5.5, 6.5])
    parser.add_argument("--speed-bins", type=float, nargs="+", default=[0.0, 2.0, 3.5, 5.0, 6.5, 8.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--include-no-actuator", action="store_true")
    parser.add_argument("--policy-every-step", action="store_true")
    parser.add_argument("--actuator-model", type=Path, default=DEFAULT_ACTUATOR_DIR / "actuator_net.pth")
    parser.add_argument("--actuator-scaler-x", type=Path, default=DEFAULT_ACTUATOR_DIR / "scaler_X.pkl")
    parser.add_argument("--actuator-scaler-y", type=Path, default=DEFAULT_ACTUATOR_DIR / "scaler_y.pkl")
    parser.add_argument("--actuator-history", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    args.original_onnx = args.original_onnx.resolve()
    args.finetuned_onnx = args.finetuned_onnx.resolve()
    args.config = args.config.resolve()
    args.actuator_model = args.actuator_model.resolve()
    args.actuator_scaler_x = args.actuator_scaler_x.resolve()
    args.actuator_scaler_y = args.actuator_scaler_y.resolve()
    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = project_root / "eval_results" / f"high_speed_actuator_study_{stamp}"
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(project_root)
    base_config = load_config(args.config)
    policies = {
        "original": (args.original_onnx, make_onnx_policy(args.original_onnx)[0]),
        "actuator_ft": (args.finetuned_onnx, make_onnx_policy(args.finetuned_onnx)[0]),
    }

    all_episode_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    run_meta = {
        "config": str(args.config),
        "map": args.map,
        "speed_caps": args.speed_caps,
        "speed_bins": args.speed_bins,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "actuator_model": str(args.actuator_model),
        "policies": {name: str(path) for name, (path, _) in policies.items()},
    }

    for speed_cap in args.speed_caps:
        for policy_name, (onnx_path, policy) in policies.items():
            for actuator in ([True, False] if args.include_no_actuator else [True]):
                eps, steps, meta = run_one(policy_name, policy, onnx_path, base_config, args, speed_cap, actuator)
                all_episode_rows.extend(eps)
                all_step_rows.extend(steps)
                run_meta.setdefault("runs", []).append(meta)

    aggregate_rows = aggregate_episodes(all_episode_rows)
    speed_bin_rows = aggregate_speed_bins(all_step_rows, args.speed_bins)

    write_csv(args.out_dir / "episode_metrics.csv", all_episode_rows)
    write_csv(args.out_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.out_dir / "speed_bin_metrics.csv", speed_bin_rows)
    (args.out_dir / "study_results.json").write_text(
        json.dumps(
            {
                "meta": run_meta,
                "aggregate": aggregate_rows,
                "speed_bins": speed_bin_rows,
                "episodes": all_episode_rows,
            },
            indent=2,
        )
        + "\n"
    )
    write_markdown(args.out_dir / "report_summary.md", aggregate_rows, speed_bin_rows, args)
    make_plots(args.out_dir, aggregate_rows, speed_bin_rows)

    print(f"\nWrote study outputs to: {args.out_dir}")
    print(f"Report summary: {args.out_dir / 'report_summary.md'}")


if __name__ == "__main__":
    main()
