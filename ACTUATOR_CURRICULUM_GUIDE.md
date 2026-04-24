"""
Actuator Model & Curriculum Learning Training Guide
====================================================

This guide shows how to train RL policies that account for real vehicle 
actuator dynamics and progressively increase difficulty via curriculum learning.

Overview of the 5-Step Pipeline
================================

1. TRAIN RL MODEL IN SIMULATION
   - Start with high safety margins, low speed
   - Collect driving data as policy improves
   - Export trained policy weights to ONNX

2. RUN POLICY ON ROBOT
   - Deploy the trained policy to Jetson
   - Collect ackermann commands + pose/velocity data
   - Bundle data and send to offline machine

3. TRAIN ACTUATOR MODEL (offline)
   - Use collected (cmd, actual_state) pairs
   - Train small MLP to predict: actual_yaw_rate[t+1] from history
   - Output: TorchScript model + scalers (scaler_X.pkl, scaler_y.pkl)

4. RETRAIN RL WITH ACTUATOR MODEL
   - Load the trained actuator model into the simulator
   - Use it to predict actual vehicle response to commands
   - Retrain RL policy from scratch with this model
   - Progressive curriculum: increase speed, reduce safety margins
   - Export updated policy weights

5. DEPLOY UPDATED POLICY
   - Send new policy weights back to robot
   - Car now drives better because policy accounts for real dynamics


Quick Start: Train with Actuator Model + Curriculum
====================================================

# Once you have a trained actuator model (actuator_net.pth, scaler_X.pkl, scaler_y.pkl):

python3 scripts/train.py \\
    --name actuator_curriculum_v1 \\
    --actuator-model path/to/actuator_net.pth \\
    --actuator-scaler-X path/to/scaler_X.pkl \\
    --actuator-scaler-y path/to/scaler_y.pkl \\
    --curriculum \\
    --curriculum-speeds 2.0 2.5 3.0 3.5 4.0 5.0 \\
    --curriculum-margins 1.5 1.2 1.0 0.8 0.6 0.4 \\
    --curriculum-steps-per-phase 200000 \\
    --total-steps 1200000 \\
    --num-envs 8 \\
    --device cuda


Command-Line Arguments
======================

Actuator Model Integration:
  --actuator-model PATH           Path to TorchScript/PyTorch model file
  --actuator-scaler-X PATH        Path to joblib StandardScaler for inputs
  --actuator-scaler-y PATH        Path to joblib StandardScaler for outputs

Curriculum Learning:
  --curriculum                    Enable curriculum (progressive difficulty)
  --curriculum-steps-per-phase N  Steps per curriculum phase (default: 100k)
  --curriculum-speeds S1 S2 ...   Speed schedule (m/s). Example: 2.0 2.5 3.0 4.0
  --curriculum-margins M1 M2 ...  Safety margin schedule (m). Example: 1.5 1.0 0.5


Example 1: Retrain with Actuator Model (No Curriculum)
=======================================================

python3 scripts/train.py \\
    --name actuator_baseline \\
    --actuator-model ~/.f1tenth/models/actuator_net.pth \\
    --actuator-scaler-X ~/.f1tenth/models/scaler_X.pkl \\
    --actuator-scaler-y ~/.f1tenth/models/scaler_y.pkl \\
    --total-steps 2000000


Example 2: Progressive Speed & Safety Curriculum
=================================================

python3 scripts/train.py \\
    --name progressive_curriculum \\
    --actuator-model ~/.f1tenth/models/actuator_net.pth \\
    --actuator-scaler-X ~/.f1tenth/models/scaler_X.pkl \\
    --actuator-scaler-y ~/.f1tenth/models/scaler_y.pkl \\
    --curriculum \\
    --curriculum-speeds 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \\
    --curriculum-margins 1.5 1.3 1.1 0.9 0.7 0.5 0.3 \\
    --curriculum-steps-per-phase 150000 \\
    --total-steps 1050000 \\
    --map maps/levine_slam/levine_slam


Example 3: Only Curriculum (No Actuator Model)
===============================================

python3 scripts/train.py \\
    --name curriculum_only \\
    --curriculum \\
    --curriculum-speeds 2.0 2.5 3.0 3.5 4.0 \\
    --curriculum-margins 1.5 1.0 0.8 0.6 0.4 \\
    --curriculum-steps-per-phase 200000 \\
    --total-steps 1000000


Configuration File Alternative
==============================

Instead of command-line flags, you can also create a YAML config file:

# configs/actuator_curriculum.yaml
experiment:
  name: actuator_curriculum_trial
  device: cuda
  
algorithm:
  type: ppo
  total_timesteps: 1200000

env:
  num_envs: 8
  map_path: maps/levine_slam/levine_slam
  max_steps: 3000

# Actuator model section
actuator_model:
  model_path: /home/user/.f1tenth/models/actuator_net.pth
  scaler_X_path: /home/user/.f1tenth/models/scaler_X.pkl
  scaler_y_path: /home/user/.f1tenth/models/scaler_y.pkl
  history_steps: 3

# Curriculum learning section
curriculum:
  enabled: true
  steps_per_phase: 150000
  speed_schedule: [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
  margin_schedule: [1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]

Then run:
  python3 scripts/train.py --config configs/actuator_curriculum.yaml


Monitoring Curriculum Progress
==============================

The training logs will show curriculum info at each step:

curriculum_phase: 0          # Current phase (0-indexed)
curriculum_max_speed: 2.0    # Current max speed (m/s)
curriculum_safety_margin: 1.5  # Current safety margin (m)
curriculum_total_steps: 1234  # Total training steps in current phase

To integrate with WandB:
  python3 scripts/train.py \\
    ... \\
    --wandb \\
    --wandb-project your-project


Full End-to-End Workflow
========================

# 1. Train initial policy (high safety, low speed)
python3 scripts/train.py \\
    --name initial_policy \\
    --total-steps 1000000

# 2. Export to ONNX for robot
python3 scripts/export_model.py --run runs/initial_policy_*

# 3. (Deploy to robot, collect actuator data, train actuator_net offline)

# 4. Retrain with actuator model + curriculum
python3 scripts/train.py \\
    --name retraining_iter1 \\
    --actuator-model /path/to/actuator_net.pth \\
    --actuator-scaler-X /path/to/scaler_X.pkl \\
    --actuator-scaler-y /path/to/scaler_y.pkl \\
    --curriculum \\
    --curriculum-speeds 2.0 2.5 3.0 3.5 4.0 5.0 \\
    --curriculum-margins 1.5 1.2 1.0 0.8 0.6 0.4 \\
    --curriculum-steps-per-phase 200000 \\
    --total-steps 1400000

# 5. Export updated policy
python3 scripts/export_model.py --run runs/retraining_iter1_*

# 6. Deploy updated policy to robot
# (Repeat steps 3-6 as needed to progressively improve)


Troubleshooting
==============

Q: "Failed to load ActuatorModel" warning
A: Check that:
   - Model file exists and is readable
   - Using torch-compatible format (.pth or .pt)
   - Scaler files exist if specified

Q: Curriculum not changing speed
A: Verify:
   - --curriculum flag is set
   - --curriculum-steps-per-phase is reasonable (e.g., < 50% of total steps)
   - Check WandB logs for curriculum_phase and curriculum_max_speed fields

Q: Policy doesn't improve with actuator model
A: Consider:
   - Actuator model may be poorly trained (collect more diverse data)
   - history_steps may be too short/long (try 3-5)
   - Increasing curriculum steps_per_phase to give policy more time per phase
   - Retraining from scratch rather than fine-tuning


References
==========

For more details on the data collection & actuator model training, see:
  f1tenth-project/README.md
  f1tenth-project/scripts/train_pipeline.py
"""

# This is a guide file, not executable Python code.
