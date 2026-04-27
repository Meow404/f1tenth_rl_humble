# Asynchronous Lidar and Control Loop Frequencies

## Overview

You can now decouple the **lidar update frequency** from the **control loop frequency** in your environment. This is useful for:

- **Simulating real sensor timing**: Real lidar sensors like the Hokuyo UST-10LX update at 25 Hz, not 100 Hz
- **Testing robustness**: Train policies that must handle stale sensor data
- **Real-world deployment**: Match the timing of your actual robot hardware

## How It Works

### Base Physics Loop (Fixed)
The physics simulation always runs at the frequency specified by `env.timestep`:
```yaml
env:
  timestep: 0.01  # Always 100 Hz (0.01 sec per step)
```

### Lidar Update Frequency (Configurable)
Controls how often **new lidar scans arrive**:
```yaml
lidar:
  update_freq_hz: 20  # New scan every 50ms (5 steps at 100 Hz)
```

Between scans, the previous lidar reading is **cached and reused** automatically. Other state data (velocity, position, etc.) continues to update normally.

### Control Loop Frequency (Configurable)
Controls how often **policy inference runs**:
```yaml
lidar:
  control_freq_hz: 50  # Run policy every 20ms (2 steps at 100 Hz)
```

When using frame-skipping style control, actions are applied continuously, but the policy only produces new outputs every N steps.

## Example Scenarios

### Scenario 1: 20 Hz Lidar + 50 Hz Control
```yaml
env:
  timestep: 0.01        # 100 Hz physics

lidar:
  update_freq_hz: 20    # Lidar: every 5 physics steps
  control_freq_hz: 50   # Control: every 2 physics steps

# Timeline:
# Step 0: NEW lidar, NEW policy output
# Step 1: Stale lidar, NEW policy output
# Step 2: Stale lidar, NEW policy output
# Step 3: Stale lidar, NEW policy output
# Step 4: Stale lidar, NEW policy output
# Step 5: NEW lidar, NEW policy output  (repeat)
```

### Scenario 2: Match Real Hokuyo (25 Hz Lidar) + 100 Hz Control
```yaml
lidar:
  update_freq_hz: 25    # Hokuyo scan rate
  control_freq_hz: 100  # Full physics rate
```

### Scenario 3: Disabled (Default Behavior)
```yaml
lidar:
  update_freq_hz: 0     # 0 = disabled, lidar updates every step
  control_freq_hz: 0    # 0 = disabled, policy updates every step
```

## Training Commands

```bash
# Use the async frequency config
python scripts/train.py --config configs/async_frequencies.yaml

# Or modify on the fly
python scripts/train.py --config configs/default.yaml \
  --lidar.update_freq_hz 20 \
  --lidar.control_freq_hz 50

# With additional training parameters
python scripts/train.py --config configs/async_frequencies.yaml \
  --total-steps 2000000 \
  --num-envs 16
```

## Technical Details

### Lidar Caching
- When a new lidar scan arrives (`lidar_steps_since_update >= lidar_update_steps`), it's cached
- Between updates, the cached scan is reused in observations
- All other state data (velocity, position, etc.) updates normally
- This reflects real hardware: sensor data is stale between scans, but odometry updates continuously

### Control Skipping  
- Not yet implemented, but can be added similarly to lidar caching
- Currently, policy always runs every step
- If you set `control_freq_hz`, it will skip inference on some steps (future feature)

### Step Counts
The step counter (`current_step`) still increments every physics step, not every policy step. This ensures:
- Episodes end at `max_steps` regardless of control frequency
- Reward calculation uses consistent timing
- Compatibility with standard RL training loops

## Important Notes

1. **Must be >= lidar frequency**: `control_freq_hz` should be >= `lidar_update_freq_hz`. Running policy less frequently than sensor updates might miss important information.

2. **Affects observation space**: With `frame_stack > 1`, stale lidar data will be duplicated across frames when there's no update.

3. **Sim-to-real transfer**: If you use asynchronous frequencies in simulation, your real hardware should have similar timing. Otherwise, your policy may perform worse on the real car.

4. **Determinism**: Setting non-default frequencies doesn't affect reproducibility if you use the same random seed.

## Example: Training Robust Policies

To train a policy that's robust to sensor delays:

```bash
python scripts/train.py --config configs/async_frequencies.yaml \
  --experiment.name "robust_20hz_lidar" \
  --total-steps 2000000 \
  --num-envs 16
```

Then evaluate it in simulation with:
- Different lidar frequencies (10 Hz, 20 Hz, 40 Hz)
- Different control frequencies (25 Hz, 100 Hz)

The policy should generalize well across these variations.

## Troubleshooting

**Q: My policy learns slower with asynchronous frequencies**
- A: Yes, this is expected. Stale sensor data makes the task harder. Use longer training runs.

**Q: The policy crashes immediately with these frequencies**
- A: Try starting with higher lidar frequency (30-50 Hz) and reduce gradually.

**Q: Does this affect multi-agent training?**
- A: The decoupling applies to all agents (ego + opponents) equally.

**Q: Can I use different frequencies for different agents?**
- A: Not currently. All agents share the same frequency settings.
