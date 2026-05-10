"""
Actuator Model Integration for RL Training
===========================================

Optionally loads a trained actuator model (MLP) that predicts vehicle response
to steering commands. This model is injected into the environment so the RL
policy learns to account for actuator lag and nonlinearity.

Temporal contract
-----------------
predict(cmd_steering, actual_speed, actual_yaw_rate, actual_lateral_vel,
        cmd_speed)

  Call BEFORE env.step() with the commands you are about to send.
  Uses the car's state at time t (from prev_obs_dict) plus the current
  commands, and returns the predicted state at time t+1.

  The wrapper then injects those predictions into the obs returned by
  env.step(), so the RL policy always observes the corrected dynamics
  instead of the sim's idealized values.

  reset() MUST be called at the start of every episode to zero the
  history buffer.  Failure to do so leaks history across episodes and
  degrades model accuracy, especially on the first few steps.
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
    Dynamics net: predicts [speed[t+1], yaw_rate[t+1], lateral_vel[t+1]] from:
      - cmd_steering[t], cmd_speed[t]                     (current commands)
      - history_steps x (cmd_steer, yaw_rate,             (past state, newest first)
                         speed, lateral_vel, slip_angle)

    Input dim  : 2 + history_steps * 5
    Output dim : 3  → [speed, yaw_rate, lateral_vel]
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
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActuatorModel:
    """
    Wrapper around a trained actuator net for use during RL training.

    Predicts actual vehicle response (speed, yaw_rate, lateral_vel) given
    commanded steering/speed and a rolling window of recent state history.

    Critical: call reset() at the start of every episode.
    """

    # Feature layout per history slot: (cmd_steer, yaw_rate, speed, lat_vel, slip)
    _COLS_PER_STEP = 5

    def __init__(
        self,
        model_path: str,
        scaler_X_path: Optional[str] = None,
        scaler_y_path: Optional[str] = None,
        history_steps: int = 15,
        device: str = "cpu",
    ):
        """
        Args:
            model_path:     Path to saved TorchScript or PyTorch model (.pth/.pt)
            scaler_X_path:  Path to joblib StandardScaler for inputs
            scaler_y_path:  Path to joblib StandardScaler for outputs
            history_steps:  Number of past timesteps in the model  **must match
                            the value used during offline training**
            device:         "cpu" or "cuda"
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for ActuatorModel. Install: pip install torch")

        self.device = device
        self.history_steps = history_steps

        cols = self._COLS_PER_STEP
        # Feature vector: [cmd_steer[t], cmd_speed[t]]  +  history_steps × 5
        self.feature_dim = 2 + history_steps * cols
        # Rolling buffer: (history_steps + 1) slots so we can always shift by one
        self._buffer_size = (history_steps + 1) * cols

        # --- Load model ---
        try:
            self.model = torch.jit.load(model_path, map_location=device)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                hidden_dims = checkpoint.get("hidden_dims", [64, 64, 32])
                self.model = ActuatorNet(self.feature_dim, hidden_dims)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model = checkpoint
        self.model.eval()

        # --- Validate feature dimension against the loaded model ---
        self._validate_feature_dim(model_path)

        # --- Load scalers ---
        self.scaler_X = None
        self.scaler_y = None
        if scaler_X_path and Path(scaler_X_path).exists():
            self.scaler_X = joblib.load(scaler_X_path)
        if scaler_y_path and Path(scaler_y_path).exists():
            self.scaler_y = joblib.load(scaler_y_path)

        # Validate scaler dimensions if available
        if self.scaler_X is not None:
            scaler_n = self.scaler_X.n_features_in_
            if scaler_n != self.feature_dim:
                raise ValueError(
                    f"ActuatorModel: scaler_X expects {scaler_n} features but "
                    f"history_steps={history_steps} produces {self.feature_dim}. "
                    f"Make sure history_steps matches the value used during training."
                )

        # Rolling state buffer — slot 0 = most recent (excluded from feature);
        # slots 1..history_steps = the history fed into the model
        self._history = np.zeros(self._buffer_size, dtype=np.float32)
        self._steps_since_reset = 0

    def _validate_feature_dim(self, model_path: str) -> None:
        """Probe the model with a zero input to check feature dimension."""
        try:
            probe = torch.zeros(1, self.feature_dim)
            with torch.no_grad():
                self.model(probe)
        except Exception as e:
            raise ValueError(
                f"ActuatorModel: forward pass with feature_dim={self.feature_dim} "
                f"(history_steps={self.history_steps}) failed: {e}.\n"
                f"Check that history_steps matches the value used during training of {model_path}."
            ) from e

    @classmethod
    def from_env(cls, config: Dict[str, Any]) -> Optional["ActuatorModel"]:
        """
        Factory: create from config if actuator model is specified.
        Returns None if model_path not in config or file not found.
        """
        model_path = config.get("actuator_model", {}).get("model_path")
        if not model_path or not Path(model_path).exists():
            return None

        try:
            return cls(
                model_path=model_path,
                scaler_X_path=config.get("actuator_model", {}).get("scaler_X_path"),
                scaler_y_path=config.get("actuator_model", {}).get("scaler_y_path"),
                history_steps=config.get("actuator_model", {}).get("history_steps", 15),
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
        Predict (speed[t+1], yaw_rate[t+1], lateral_vel[t+1]).

        Call this BEFORE env.step() using the current state (from prev_obs_dict)
        and the commands you are about to send.  The history buffer is updated
        with the CURRENT state so the next call sees it as the most recent past.

        Args:
            cmd_steering:       commanded steering angle [rad]  (action being sent)
            actual_speed:       current longitudinal speed [m/s]  (from prev_obs_dict)
            actual_yaw_rate:    current yaw rate [rad/s]  (from prev_obs_dict)
            actual_lateral_vel: current lateral slip velocity [m/s]  (0.0 if unavailable)
            cmd_speed:          commanded speed [m/s]  (action being sent)

        Returns:
            (predicted_speed, predicted_yaw_rate, predicted_lateral_vel)
        """
        import math
        cols = self._COLS_PER_STEP

        # Compute slip angle from current state
        slip = math.atan2(actual_lateral_vel, actual_speed + 1e-6)

        # Shift buffer: make room for the new most-recent slot
        self._history = np.roll(self._history, cols)
        # Slot 0 = current state (will be history on the NEXT call)
        self._history[:cols] = [cmd_steering, actual_yaw_rate, actual_speed,
                                 actual_lateral_vel, slip]

        self._steps_since_reset += 1

        # Feature: [cmd_steering[t], cmd_speed[t]] + history slots 1..N
        # (slot 0 is "current" and is excluded, matching training convention)
        x = np.concatenate([[cmd_steering, cmd_speed],
                             self._history[cols:]]).reshape(1, -1).astype(np.float32)

        if self.scaler_X is not None:
            x = self.scaler_X.transform(x).astype(np.float32)

        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)

        if y_pred.shape[1] >= 3:
            return float(y_pred[0, 0]), float(y_pred[0, 1]), float(y_pred[0, 2])
        # Fallback for 2-output legacy models (speed, yaw_rate only)
        return float(actual_speed), float(y_pred[0, 0]), float(y_pred[0, 1])

    def reset(self):
        """
        Reset history for a new episode.

        MUST be called at the start of every episode.  Stale history from a
        previous episode causes the model to predict incorrectly for the first
        history_steps steps of each new episode.
        """
        self._history.fill(0.0)
        self._steps_since_reset = 0


class CurriculumScheduler:
    """
    Progressive curriculum: increase speed and reduce safety margins
    as training progresses.
    """

    def __init__(
        self,
        speed_schedule: Optional[list] = None,
        margin_schedule: Optional[list] = None,
        steps_per_phase: int = 100_000,
    ):
        self.speed_schedule = speed_schedule or [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        self.margin_schedule = margin_schedule or [1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]
        self.steps_per_phase = steps_per_phase
        self.total_steps = 0
        self.current_phase = 0

    def update(self, steps: int = 1) -> tuple:
        self.total_steps += steps
        phase = min(self.total_steps // self.steps_per_phase, len(self.speed_schedule) - 1)
        if phase != self.current_phase:
            self.current_phase = phase
        return self.get_current()

    def get_current(self) -> tuple:
        phase = min(self.current_phase, len(self.speed_schedule) - 1)
        return self.speed_schedule[phase], self.margin_schedule[phase]

    def get_phase_info(self) -> Dict[str, Any]:
        speed, margin = self.get_current()
        return {
            "curriculum_phase": self.current_phase,
            "curriculum_max_speed": speed,
            "curriculum_safety_margin": margin,
            "curriculum_total_steps": self.total_steps,
        }
