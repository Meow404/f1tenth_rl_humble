#!/usr/bin/env python3
"""Evaluate a standalone ONNX policy in F1TenthWrapper with optional actuator model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
for parent in project_root.parents:
    gym_parent = parent / "f1tenth_gym_ros" / "f1tenth_gym"
    if gym_parent.exists():
        sys.path.insert(0, str(gym_parent))
        break


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    config.setdefault("env", {})
    config.setdefault("experiment", {})
    config.setdefault("actuator_model", {})

    config["env"]["map_path"] = args.map
    config["env"]["num_envs"] = 1
    config["experiment"]["device"] = args.device

    if args.max_steps is not None:
        config["env"]["max_steps"] = args.max_steps

    if args.actuator_model:
        config["actuator_model"]["model_path"] = str(args.actuator_model)
        config["actuator_model"]["scaler_X_path"] = str(args.actuator_scaler_x)
        config["actuator_model"]["scaler_y_path"] = str(args.actuator_scaler_y)
        config["actuator_model"]["history_steps"] = args.actuator_history
    else:
        config.pop("actuator_model", None)

    # Evaluation should be deterministic and not randomize dynamics unless requested.
    if args.disable_domain_randomization:
        config.setdefault("domain_randomization", {})
        config["domain_randomization"]["enabled"] = False
        config["domain_randomization"]["mode"] = "off"
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


def summarize(metrics: dict[str, list]) -> dict:
    summary = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        if key == "collision":
            summary[key] = {
                "rate": float(np.mean(arr)),
                "count": int(np.sum(arr)),
                "episodes": int(len(arr)),
            }
        else:
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return summary


def _maybe_record_actuator_diag(diag: dict[str, list], info: dict) -> None:
    if not info.get("actuator_enabled", False):
        return
    if not info.get("actuator_warmed_up", False):
        return
    # Some keys can be None depending on warm-up / legacy models.
    for key in (
        "actuator_delta_speed",
        "actuator_delta_yaw_rate",
        "actuator_delta_lateral_vel",
    ):
        val = info.get(key, None)
        if val is None:
            continue
        diag[key].append(float(val))


def _summarize_abs(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "count": int(arr.size),
        "mean_abs": float(np.mean(arr)),
        "std_abs": float(np.std(arr)),
        "max_abs": float(np.max(arr)),
    }


def _world_to_px(track, x: float, y: float, scale: float) -> tuple[int, int]:
    origin = track.spec.origin
    resolution = float(track.spec.resolution)
    h = track.occupancy_map.shape[0]
    px = int(round((x - origin[0]) / resolution * scale))
    py = int(round((h - (y - origin[1]) / resolution) * scale))
    return px, py


def _make_video_frame(env, trail: list[tuple[float, float]], pose: tuple[float, float, float], size: int = 900) -> np.ndarray:
    base_env = env.base_env.unwrapped
    track = base_env.track
    occ = np.asarray(track.occupancy_map, dtype=np.float32)
    gray = np.clip(occ, 0, 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray])
    img = Image.fromarray(rgb)

    scale = min(size / img.width, size / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)

    if getattr(track, "centerline", None) is not None:
        pts = [
            _world_to_px(track, float(x), float(y), scale)
            for x, y in zip(track.centerline.xs, track.centerline.ys)
        ]
        if len(pts) > 1:
            draw.line(pts, fill=(80, 160, 255), width=2)

    if len(trail) > 1:
        trail_pts = [_world_to_px(track, x, y, scale) for x, y in trail]
        draw.line(trail_pts, fill=(255, 180, 40), width=3)

    x, y, theta = pose
    cx, cy = _world_to_px(track, float(x), float(y), scale)
    car_len = max(10, int(0.58 / float(track.spec.resolution) * scale))
    car_wid = max(6, int(0.31 / float(track.spec.resolution) * scale))
    corners = np.array([
        [car_len / 2, car_wid / 2],
        [car_len / 2, -car_wid / 2],
        [-car_len / 2, -car_wid / 2],
        [-car_len / 2, car_wid / 2],
    ])
    rot = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    car = corners @ rot.T + np.array([cx, cy])
    draw.polygon([tuple(p) for p in car], fill=(255, 40, 40), outline=(40, 0, 0))
    nose = np.array([car_len / 2, 0.0]) @ rot.T + np.array([cx, cy])
    draw.line([(cx, cy), tuple(nose)], fill=(255, 255, 255), width=2)
    draw.rectangle((8, 8, 230, 62), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line((18, 24, 58, 24), fill=(80, 160, 255), width=3)
    draw.text((66, 16), "centerline", fill=(0, 0, 0))
    draw.line((18, 46, 58, 46), fill=(255, 180, 40), width=3)
    draw.text((66, 38), "car trajectory", fill=(0, 0, 0))

    return np.asarray(img)


def _make_comparison_frame(track, trails: dict[str, list[tuple[float, float]]], poses: dict[str, tuple[float, float, float]], size: int = 900) -> np.ndarray:
    occ = np.asarray(track.occupancy_map, dtype=np.float32)
    gray = np.clip(occ, 0, 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray])
    img = Image.fromarray(rgb)
    scale = min(size / img.width, size / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)

    if getattr(track, "centerline", None) is not None:
        pts = [_world_to_px(track, float(x), float(y), scale) for x, y in zip(track.centerline.xs, track.centerline.ys)]
        if len(pts) > 1:
            draw.line(pts, fill=(80, 160, 255), width=2)

    colors = {
        "no_actuator": (60, 210, 90),
        "actuator": (255, 90, 40),
    }
    labels = {
        "no_actuator": "without actuator",
        "actuator": "with actuator",
    }
    for key, trail in trails.items():
        if len(trail) > 1:
            draw.line([_world_to_px(track, x, y, scale) for x, y in trail], fill=colors[key], width=3)
        if key in poses:
            x, y, _ = poses[key]
            px, py = _world_to_px(track, x, y, scale)
            r = 5
            draw.ellipse((px - r, py - r, px + r, py + r), fill=colors[key], outline=(0, 0, 0))

    draw.rectangle((8, 8, 265, 84), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line((18, 24, 58, 24), fill=(80, 160, 255), width=3)
    draw.text((66, 16), "centerline", fill=(0, 0, 0))
    y = 46
    for key in ("no_actuator", "actuator"):
        draw.line((18, y, 58, y), fill=colors[key], width=3)
        draw.text((66, y - 8), labels[key], fill=(0, 0, 0))
        y += 22
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "sim2real_e2e.yaml")
    parser.add_argument("--map", type=str, default="maps/levine_slam/levine_slam")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-every", type=int, default=3)
    parser.add_argument("--video-episode", type=int, default=1)
    parser.add_argument("--comparison-video", type=Path, default=None)
    parser.add_argument("--comparison-steps", type=int, default=1000)
    parser.add_argument(
        "--policy-every-step",
        action="store_true",
        help=(
            "Run ONNX inference at every env.step (default: run inference only on control ticks if the env exposes control_update_steps > 1)."
        ),
    )
    parser.add_argument("--disable-domain-randomization", action="store_true", default=True)
    parser.add_argument("--keep-domain-randomization", dest="disable_domain_randomization", action="store_false")
    parser.add_argument("--actuator-model", type=Path, default=None)
    parser.add_argument("--actuator-scaler-x", type=Path, default=None)
    parser.add_argument("--actuator-scaler-y", type=Path, default=None)
    parser.add_argument("--actuator-history", type=int, default=15)
    args = parser.parse_args()

    from f1tenth_rl.envs.wrapper import F1TenthWrapper

    config = apply_overrides(load_config(args.config), args)
    policy, onnx_info = make_onnx_policy(args.onnx)
    os.chdir(project_root)
    env = F1TenthWrapper(config, render_mode="human" if args.render else None)

    print("ONNX:", args.onnx)
    print("Map:", config["env"]["map_path"])
    print("Actuator model:", config.get("actuator_model", {}).get("model_path", "disabled"))
    print("ONNX info:", onnx_info)

    # If control_freq_hz is configured, F1TenthWrapper can hold the last command
    # for multiple physics steps. In that case, avoid wasting compute by running
    # the policy only when the command is allowed to update.
    control_every = int(getattr(env, "control_update_steps", 1))
    if args.policy_every_step:
        policy_every = 1
    else:
        policy_every = max(1, control_every)
    print(f"Policy inference every {policy_every} env.step(s)")

    if args.comparison_video:
        compare_config = dict(config)
        no_actuator_config = dict(config)
        no_actuator_config.pop("actuator_model", None)
        env_no = F1TenthWrapper(no_actuator_config, render_mode=None)
        env_yes = F1TenthWrapper(compare_config, render_mode=None)
        trails = {"no_actuator": [], "actuator": []}
        poses = {}
        frames = []
        actuator_diag = defaultdict(list)
        try:
            obs_no, _ = env_no.reset(seed=42)
            obs_yes, _ = env_yes.reset(seed=42)
            control_every_no = int(getattr(env_no, "control_update_steps", 1))
            control_every_yes = int(getattr(env_yes, "control_update_steps", 1))
            policy_every_no = 1 if args.policy_every_step else max(1, control_every_no)
            policy_every_yes = 1 if args.policy_every_step else max(1, control_every_yes)
            last_action_no = None
            last_action_yes = None
            for step in range(args.comparison_steps):
                if last_action_no is None or step % policy_every_no == 0:
                    last_action_no = policy(obs_no)
                if last_action_yes is None or step % policy_every_yes == 0:
                    last_action_yes = policy(obs_yes)
                action_no = last_action_no
                action_yes = last_action_yes
                obs_no, _, done_no, trunc_no, info_no = env_no.step(action_no)
                obs_yes, _, done_yes, trunc_yes, info_yes = env_yes.step(action_yes)
                _maybe_record_actuator_diag(actuator_diag, info_no)
                _maybe_record_actuator_diag(actuator_diag, info_yes)
                for key, info in (("no_actuator", info_no), ("actuator", info_yes)):
                    raw = info.get("raw_obs", {})
                    x = float(raw.get("poses_x", [0.0])[0])
                    y = float(raw.get("poses_y", [0.0])[0])
                    th = float(raw.get("poses_theta", [0.0])[0])
                    trails[key].append((x, y))
                    poses[key] = (x, y, th)
                if step % max(args.video_every, 1) == 0:
                    frames.append(_make_comparison_frame(env_yes.base_env.unwrapped.track, trails, poses))
                if done_no or trunc_no or done_yes or trunc_yes:
                    break
        finally:
            env_no.close()
            env_yes.close()
        if frames:
            import imageio.v2 as imageio

            args.comparison_video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.comparison_video, frames, fps=args.video_fps)
            print("Wrote comparison video:", args.comparison_video)

        if actuator_diag:
            diag_summary = {
                "abs_delta_speed": _summarize_abs(actuator_diag.get("actuator_delta_speed", [])),
                "abs_delta_yaw_rate": _summarize_abs(actuator_diag.get("actuator_delta_yaw_rate", [])),
                "abs_delta_lateral_vel": _summarize_abs(actuator_diag.get("actuator_delta_lateral_vel", [])),
            }
            print("Actuator diagnostics (comparison run, abs deltas vs ideal sim):")
            print(json.dumps(diag_summary, indent=2))

    metrics = defaultdict(list)
    episode_rows = []
    video_frames = []
    actuator_diag = defaultdict(list)
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=42 + ep)
            done = False
            total_reward = 0.0
            speeds = []
            abs_steers = []
            actions = []
            info = {}
            trail = []
            last_action = None

            while not done:
                step_idx = int(info.get("step", 0))
                if last_action is None or step_idx % policy_every == 0:
                    last_action = policy(obs)
                action = last_action
                obs, reward, terminated, truncated, info = env.step(action)
                _maybe_record_actuator_diag(actuator_diag, info)
                total_reward += float(reward)
                done = bool(terminated or truncated)
                speeds.append(float(info.get("ego_speed", 0.0)))
                physical_action = np.asarray(info.get("physical_action", [0.0, 0.0]))
                abs_steers.append(abs(float(physical_action[0])))
                actions.append(physical_action)
                raw_obs = info.get("raw_obs", {})
                xs = raw_obs.get("poses_x", [None])
                ys = raw_obs.get("poses_y", [None])
                if xs[0] is not None and ys[0] is not None:
                    trail.append((float(xs[0]), float(ys[0])))
                if (
                    args.video
                    and ep + 1 == args.video_episode
                    and int(info.get("step", 0)) % max(args.video_every, 1) == 0
                ):
                    theta = float(raw_obs.get("poses_theta", [0.0])[0])
                    video_frames.append(_make_video_frame(env, trail, (trail[-1][0], trail[-1][1], theta)))
                if args.render:
                    env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            row = {
                "episode": ep + 1,
                "return": total_reward,
                "steps": int(info.get("step", 0)),
                "progress": float(info.get("progress", 0.0)),
                "collision": bool(info.get("ego_collision", False)),
                "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed": float(np.max(speeds)) if speeds else 0.0,
                "avg_abs_steer": float(np.mean(abs_steers)) if abs_steers else 0.0,
                "lap_time": float(info.get("ego_lap_time", 0.0)),
            }
            episode_rows.append(row)
            for key, value in row.items():
                if key != "episode":
                    metrics[key].append(float(value))
            print(
                f"Ep {ep + 1:02d}/{args.episodes}: "
                f"return={row['return']:.2f} progress={row['progress']:.1%} "
                f"avg_speed={row['avg_speed']:.2f} max_speed={row['max_speed']:.2f} "
                f"{'CRASH' if row['collision'] else 'OK'}"
            )
    finally:
        env.close()

    result = {
        "onnx": str(args.onnx),
        "config": str(args.config),
        "map": config["env"]["map_path"],
        "actuator_model": config.get("actuator_model", {}),
        "onnx_info": onnx_info,
        "episodes": episode_rows,
        "summary": summarize(metrics),
    }

    if actuator_diag:
        result["actuator_diagnostics"] = {
            "abs_delta_speed": _summarize_abs(actuator_diag.get("actuator_delta_speed", [])),
            "abs_delta_yaw_rate": _summarize_abs(actuator_diag.get("actuator_delta_yaw_rate", [])),
            "abs_delta_lateral_vel": _summarize_abs(actuator_diag.get("actuator_delta_lateral_vel", [])),
        }

        print("Actuator diagnostics (episodes, abs deltas vs ideal sim):")
        print(json.dumps(result["actuator_diagnostics"], indent=2))

    print(json.dumps(result["summary"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("Wrote:", args.out)
    if args.video and video_frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, video_frames, fps=args.video_fps)
        print("Wrote video:", args.video)


if __name__ == "__main__":
    main()
