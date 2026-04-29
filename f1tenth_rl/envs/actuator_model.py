"""
Actuator Model Integration for RL Training
===========================================

Optionally loads a trained actuator model (MLP) that predicts vehicle response
to steering commands. This model is injected into the environment so the RL
policy learns to account for actuator lag and nonlinearity.

Usage:
    # In wrapper.py __init__:
    self.actuator_model = ActuatorModel.from_env(config)

    # In wrapper.py step():
    if self.actuator_model is not None:
        corrected_action = self.actuator_model.predict(action, obs_history)
        use_action = corrected_action
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

try:
    import torch
    import torch.nn as nn
    import joblib
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ActuatorNet(nn.Module):
    """
    Dynamics net: predicts [yaw_rate[t+1], lateral_vel[t+1]] from:
      - cmd_steering[t]                              (current command)
      - history_steps x (cmd_steer, yaw_rate,
                         speed, lateral_vel)[t-h]   (4 cols per past step)

    Input dim  : 1 + history_steps * 4
    Output dim : 2  → [yaw_rate, lateral_vel]

    lateral_vel captures tyre-slip dynamics missing from yaw_rate alone,
    and including the current cmd ensures the model learns how commands
    drive the next state rather than just autocorrelating past state.
    """
    def __init__(self, in_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 2))  # [yaw_rate, lateral_vel]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActuatorModel:
    """
    Wrapper around trained actuator net for use during RL training.
    
    Predicts actual vehicle response (yaw_rate) given commanded steering
    and recent state history.
    """

    def __init__(
        self,
        model_path: str,
        scaler_X_path: Optional[str] = None,
        scaler_y_path: Optional[str] = None,
        history_steps: int = 3,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to saved TorchScript or PyTorch model (.pth or .pt)
            scaler_X_path: Path to joblib StandardScaler for inputs
            scaler_y_path: Path to joblib StandardScaler for outputs
            history_steps: Number of past timesteps in the model
            device: "cpu" or "cuda"
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for ActuatorModel. Install: pip install torch")

        self.device = device
        self.history_steps = history_steps
        # 5 state features per past step: (cmd_steer, yaw_rate, speed, lateral_vel, slip_angle)
        # Plus 2 current commands at t: (cmd_steering, cmd_speed)
        # Total input = 2 + history_steps * 5
        self._cols_per_step = 5
        self.feature_dim = 2 + history_steps * self._cols_per_step
        # Buffer holds (history_steps+1) slots of 5 cols; slot 0 = current (excluded from input)
        self._buffer_dim = (history_steps + 1) * self._cols_per_step

        # Load model
        try:
            self.model = torch.jit.load(model_path, map_location=device)
        except Exception:
            # Fallback: try to load as regular .pth if JIT fails
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                hidden_dims = checkpoint.get("hidden_dims", [64, 64, 32])
                self.model = ActuatorNet(self.feature_dim, hidden_dims)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model = checkpoint
        self.model.eval()

        # Load scalers
        self.scaler_X = None
        self.scaler_y = None
        if scaler_X_path and Path(scaler_X_path).exists():
            self.scaler_X = joblib.load(scaler_X_path)
        if scaler_y_path and Path(scaler_y_path).exists():
            self.scaler_y = joblib.load(scaler_y_path)

        # State history buffer (history_steps+1 slots of 5 cols; slot 0 = current)
        self.history = np.zeros(self._buffer_dim, dtype=np.float32)

    @classmethod
    def from_env(cls, config: Dict[str, Any]) -> Optional["ActuatorModel"]:
        """
        Factory method: create from config if actuator model is specified.
        Returns None if model_path not in config or not found.
        """
        model_path = config.get("actuator_model", {}).get("model_path")
        if not model_path or not Path(model_path).exists():
            return None

        try:
            return cls(
                model_path=model_path,
                scaler_X_path=config.get("actuator_model", {}).get("scaler_X_path"),
                scaler_y_path=config.get("actuator_model", {}).get("scaler_y_path"),
                history_steps=config.get("actuator_model", {}).get("history_steps", 3),
                device=config.get("experiment", {}).get("device", "cpu"),
            )
        except Exception as e:
            warnings.warn(f"Failed to load ActuatorModel: {e}")
            return None

    def predict(
        self,
        cmd_steering: float,
        actual_speed: float,
        actual_yaw_rate: float,
        actual_lateral_vel: float = 0.0,
        cmd_speed: float = 0.0,
    ) -> tuple:
        """
        Predict (yaw_rate[t+1], lateral_vel[t+1]) from current commands + history.

        Args:
            cmd_steering:       commanded steering angle (rad)
            actual_speed:       current longitudinal velocity (m/s)
            actual_yaw_rate:    current yaw rate (rad/s)
            actual_lateral_vel: current lateral slip velocity (m/s); 0.0 for old sims
            cmd_speed:          commanded speed (m/s); 0.0 default for old callers

        Returns:
            (predicted_yaw_rate, predicted_lateral_vel)
        """
        import math
        cols = self._cols_per_step  # 5
        slip = math.atan2(actual_lateral_vel, actual_speed + 1e-6)

        # Shift buffer right, insert current observation at front slot
        self.history = np.roll(self.history, cols)
        self.history[:cols] = [cmd_steering, actual_yaw_rate, actual_speed,
                               actual_lateral_vel, slip]

        # Feature: [cmd_steering[t], cmd_speed[t]] + history[cols:]  (skip slot 0)
        x = np.concatenate([[cmd_steering, cmd_speed],
                            self.history[cols:]]).reshape(1, -1).astype(np.float32)
        if self.scaler_X is not None:
            x = self.scaler_X.transform(x).astype(np.float32)

        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)

        return float(y_pred[0, 0]), float(y_pred[0, 1])

    def reset(self):
        """Reset history for new episode."""
        self.history.fill(0.0)


class CurriculumScheduler:
    """
    Progressive curriculum learning: increase speed and reduce safety margins
    as training progresses.

    Speed schedule:   [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0] m/s
    Margin schedule:  [1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3] m
    """

    def __init__(
        self,
        speed_schedule: Optional[list] = None,
        margin_schedule: Optional[list] = None,
        steps_per_phase: int = 100_000,
    ):
        """
        Args:
            speed_schedule: List of max speeds for each phase
            margin_schedule: List of safety margins for each phase
            steps_per_phase: Number of training steps per curriculum phase
        """
        self.speed_schedule = speed_schedule or [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        self.margin_schedule = margin_schedule or [1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]
        self.steps_per_phase = steps_per_phase
        self.total_steps = 0
        self.current_phase = 0

    def update(self, steps: int = 1) -> tuple:
        """
        Update curriculum state and return (current_max_speed, current_margin).
        
        Args:
            steps: Number of training steps completed since last update

        Returns:
            (max_speed, safety_margin) for current phase
        """
        self.total_steps += steps
        phase = min(self.total_steps // self.steps_per_phase, len(self.speed_schedule) - 1)
        if phase != self.current_phase:
            self.current_phase = phase
        return self.get_current()

    def get_current(self) -> tuple:
        """Return (max_speed, safety_margin) for current phase."""
        if self.current_phase >= len(self.speed_schedule):
            phase = len(self.speed_schedule) - 1
        else:
            phase = self.current_phase
        return (
            self.speed_schedule[phase],
            self.margin_schedule[phase],
        )

    def get_phase_info(self) -> Dict[str, Any]:
        """Return dict with curriculum info for logging."""
        speed, margin = self.get_current()
        return {
            "curriculum_phase": self.current_phase,
            "curriculum_max_speed": speed,
            "curriculum_safety_margin": margin,
            "curriculum_total_steps": self.total_steps,
        }
