#!/usr/bin/env python3
"""
8_actuator_curriculum_learning.py

Example: Train RL policy with optional actuator model and curriculum learning.

This example demonstrates the full closed-loop learning pipeline:
1. Load a trained actuator model that predicts real vehicle dynamics
2. Use progressive curriculum learning to gradually increase difficulty
3. Train RL policy that accounts for actual actuator response

Prerequisites:
  - Trained actuator model (actuator_net.pth)
  - Input/output scalers (scaler_X.pkl, scaler_y.pkl)
  - Python packages: torch, stable-baselines3, gymnasium, joblib
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from f1tenth_rl.envs.wrapper import make_env
from f1tenth_rl.agents.sb3_trainer import SB3Trainer
from f1tenth_rl.envs.actuator_model import ActuatorModel, CurriculumScheduler


def main():
    # ============================================================
    # Configuration
    # ============================================================
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update experiment name
    config["experiment"]["name"] = "actuator_curriculum_learning"
    
    # ============================================================
    # Setup Actuator Model (Optional)
    # ============================================================
    
    # In a real scenario, you would have a trained actuator model
    # For this example, we'll show the config structure:
    
    actuator_model_path = os.path.expanduser("~/.f1tenth/models/actuator_net.pth")
    scaler_X_path = os.path.expanduser("~/.f1tenth/models/scaler_X.pkl")
    scaler_y_path = os.path.expanduser("~/.f1tenth/models/scaler_y.pkl")
    
    # If the model files exist, configure them
    if Path(actuator_model_path).exists():
        config["actuator_model"] = {
            "model_path": actuator_model_path,
            "scaler_X_path": scaler_X_path,
            "scaler_y_path": scaler_y_path,
            "history_steps": 3,
        }
        print(f"✓ Configured actuator model from {actuator_model_path}")
    else:
        print(f"ℹ Actuator model not found at {actuator_model_path}")
        print("  Training will proceed with simulator-only dynamics")
    
    # ============================================================
    # Setup Curriculum Learning
    # ============================================================
    
    # Progressive difficulty schedule
    config["curriculum"] = {
        "enabled": True,
        "steps_per_phase": 100_000,  # Phase progression every 100k steps
        "speed_schedule": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],     # m/s
        "margin_schedule": [1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3],   # m
    }
    
    print("✓ Configured curriculum learning:")
    print(f"  Speed progression:  {config['curriculum']['speed_schedule']}")
    print(f"  Margin progression: {config['curriculum']['margin_schedule']}")
    print(f"  Steps per phase:    {config['curriculum']['steps_per_phase']:,}")
    
    # ============================================================
    # Training Configuration
    # ============================================================
    
    config["experiment"]["seed"] = 42
    config["experiment"]["device"] = "cuda"  # or "cpu"
    
    config["algorithm"]["type"] = "ppo"
    config["algorithm"]["total_timesteps"] = 1_200_000  # 7 phases × 100k steps
    
    config["env"]["num_envs"] = 8
    config["env"]["num_laps"] = 3
    config["env"]["max_steps"] = 3000
    
    # Optional: enable domain randomization too
    config["domain_randomization"]["enabled"] = False
    config["domain_randomization"]["mode"] = "off"
    
    # ============================================================
    # Create trainer and train
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Starting RL Training with Actuator Model + Curriculum")
    print("=" * 60)
    
    trainer = SB3Trainer(config)
    trainer.setup()
    
    # The trainer will automatically:
    # 1. Load the actuator model (if available) into the environment
    # 2. Initialize curriculum scheduler
    # 3. Progress curriculum at each training step
    # 4. Log curriculum info to callbacks
    
    print(f"Training for {config['algorithm']['total_timesteps']:,} steps...")
    print(f"Curriculum phases: {len(config['curriculum']['speed_schedule'])}")
    print()
    
    # Train with curriculum
    trainer.train()
    
    # ============================================================
    # Evaluation & Export
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Evaluate the trained policy
    print("\nEvaluating policy...")
    eval_env = make_env(config, rank=0, seed=42)()
    
    episode_reward = 0.0
    obs, info = eval_env.reset()
    
    for step in range(2000):
        action, _ = trainer.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Evaluation episode reward: {episode_reward:.2f}")
    print(f"Evaluation steps: {step+1}")
    
    if info.get("curriculum_phase") is not None:
        print(f"\nFinal curriculum state:")
        print(f"  Phase: {info.get('curriculum_phase')}")
        print(f"  Speed: {info.get('curriculum_max_speed', 0):.1f} m/s")
        print(f"  Margin: {info.get('curriculum_safety_margin', 0):.2f} m")
    
    # Export the policy for deployment
    print("\nExporting policy to ONNX...")
    trainer.export_onnx(f"runs/{config['experiment']['name']}/policy.onnx")
    print(f"✓ Policy exported to runs/{config['experiment']['name']}/policy.onnx")
    
    eval_env.close()
    trainer.close()


if __name__ == "__main__":
    main()
