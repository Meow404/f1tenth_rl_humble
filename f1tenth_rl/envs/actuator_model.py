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
    Predicts actual vehicle yaw_rate[t+1] from history of
    (cmd_steering, actual_yaw_rate, actual_speed) tuples.

    Input: concatenated history of steering commands + state
    Output: predicted actual yaw_rate
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ELU(),
            nn.Linear(64, 64),     nn.ELU(),
            nn.Linear(64, 32),     nn.ELU(),
            nn.Linear(32, 1),
        )

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
        self.feature_dim = history_steps * 3  # cmd_steer, yaw_rate, speed per step

        # Load model
        try:
            self.model = torch.jit.load(model_path, map_location=device)
        except Exception:
            # Fallback: try to load as regular .pth if JIT fails
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model = ActuatorNet(self.feature_dim)
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

        # State history buffer
        self.history = np.zeros(self.feature_dim, dtype=np.float32)

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
    ) -> float:
        """
        Predict actual yaw_rate given command and recent history.

        Args:
            cmd_steering: commanded steering angle [-max_steer, max_steer]
            actual_speed: current velocity (m/s)
            actual_yaw_rate: current yaw rate (rad/s)

        Returns:
            Predicted actual yaw_rate at the next timestep
        """
        # Shift history and add new sample
        self.history = np.roll(self.history, 3)
        self.history[:3] = [cmd_steering, actual_yaw_rate, actual_speed]

        # Prepare input
        x = self.history.reshape(1, -1).astype(np.float32)
        if self.scaler_X is not None:
            x = self.scaler_X.transform(x).astype(np.float32)

        # Predict
        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            y_pred_t = self.model(x_t)
            y_pred = y_pred_t.cpu().numpy().squeeze()

        # Inverse-scale if needed
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform([[y_pred]])[0, 0]

        return float(y_pred)

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
