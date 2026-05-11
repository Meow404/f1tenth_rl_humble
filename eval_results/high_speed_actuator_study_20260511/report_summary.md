# ONNX Policy Comparison With Learned Actuator Dynamics

Generated: 2026-05-11 16:43:33

## Setup

- Episodes per policy/speed: 10
- Max steps per episode: 3000
- Map: `maps/levine_slam/levine_slam`
- Actuator model: `/home/stfelix/Workspace/ese-6510/f1tenth-online-policy-and-actuator-model-learning/f1tenth-project/f1tenth_adaptive_server/offline_actuator_weights/offline_actuator_retrained_20260510/actuator_net.pth`

## Episode-Level Results

| Speed cap | Policy | Collision rate | Progress mean | Lap time mean | Avg speed | CTE RMS | P95 CTE | Mean abs delta speed | Mean abs delta yaw |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.5 | actuator_ft | 0.000 | 1.000 | 14.537 | 4.259 | 0.143 | 0.328 | 0.088 | 0.094 |
| 3.5 | original | 0.000 | 1.000 | 14.429 | 4.254 | 0.211 | 0.424 | 0.089 | 0.098 |
| 4.5 | actuator_ft | 0.000 | 1.000 | 12.321 | 5.022 | 0.132 | 0.292 | 0.085 | 0.101 |
| 4.5 | original | 0.000 | 1.000 | 12.213 | 5.018 | 0.206 | 0.405 | 0.085 | 0.105 |
| 5.5 | actuator_ft | 0.000 | 1.000 | 10.952 | 5.651 | 0.116 | 0.252 | 0.081 | 0.151 |
| 5.5 | original | 0.000 | 1.000 | 10.822 | 5.652 | 0.198 | 0.386 | 0.080 | 0.144 |
| 6.5 | actuator_ft | 0.000 | 1.000 | 9.884 | 6.249 | 0.102 | 0.226 | 0.044 | 0.164 |
| 6.5 | original | 0.000 | 1.000 | 9.734 | 6.254 | 0.183 | 0.371 | 0.041 | 0.171 |
| 7.5 | actuator_ft | 0.000 | 1.000 | 9.039 | 6.832 | 0.100 | 0.217 | 0.041 | 0.160 |
| 7.5 | original | 0.000 | 1.000 | 8.872 | 6.840 | 0.171 | 0.365 | 0.040 | 0.175 |
| 8.0 | actuator_ft | 0.000 | 1.000 | 8.681 | 7.117 | 0.103 | 0.208 | 0.051 | 0.164 |
| 8.0 | original | 0.000 | 1.000 | 8.508 | 7.126 | 0.167 | 0.361 | 0.050 | 0.178 |

## Key Findings

- At speed cap 3.5 m/s, actuator fine-tuning reduced RMS cross-track error from 0.211 m to 0.143 m (32.3%). P95 cross-track error changed from 0.424 m to 0.328 m (22.7%), and RMS steering-rate changed by 3.0%.
- At speed cap 4.5 m/s, actuator fine-tuning reduced RMS cross-track error from 0.206 m to 0.132 m (35.9%). P95 cross-track error changed from 0.405 m to 0.292 m (27.9%), and RMS steering-rate changed by 5.8%.
- At speed cap 5.5 m/s, actuator fine-tuning reduced RMS cross-track error from 0.198 m to 0.116 m (41.6%). P95 cross-track error changed from 0.386 m to 0.252 m (34.8%), and RMS steering-rate changed by 9.3%.
- At speed cap 6.5 m/s, actuator fine-tuning reduced RMS cross-track error from 0.183 m to 0.102 m (44.2%). P95 cross-track error changed from 0.371 m to 0.226 m (39.1%), and RMS steering-rate changed by 8.7%.
- At speed cap 7.5 m/s, actuator fine-tuning reduced RMS cross-track error from 0.171 m to 0.100 m (41.7%). P95 cross-track error changed from 0.365 m to 0.217 m (40.5%), and RMS steering-rate changed by 10.3%.
- At speed cap 8.0 m/s, actuator fine-tuning reduced RMS cross-track error from 0.167 m to 0.103 m (38.6%). P95 cross-track error changed from 0.361 m to 0.208 m (42.3%), and RMS steering-rate changed by 8.7%.

## High-Speed Interpretation

Use the speed sweep to support the report claim in three ways:

1. Collision/success vs. speed cap shows whether a policy remains reliable as the requested operating envelope rises.
2. Cross-track error and steering-rate metrics show stability, not just whether the car eventually completes the lap.
3. Speed-binned actuator deltas show where ideal simulator dynamics diverge from learned real-car response; the high-speed bins are the clearest evidence that dynamics-aware training matters.

## Speed-Binned Actuator/Dynamics Metrics

| Speed cap | Policy | Speed bin | Samples | Mean speed | CTE RMS | P95 CTE | Mean abs delta speed | Mean abs delta yaw |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3.5 | actuator_ft | 0.0-2.0 | 213 | 0.977 | 0.031 | 0.058 | 0.069 | 0.109 |
| 3.5 | actuator_ft | 2.0-3.5 | 340 | 2.883 | 0.049 | 0.080 | 0.016 | 0.038 |
| 3.5 | actuator_ft | 3.5-5.0 | 13994 | 4.342 | 0.146 | 0.333 | 0.090 | 0.095 |
| 3.5 | original | 0.0-2.0 | 216 | 0.986 | 0.032 | 0.061 | 0.060 | 0.149 |
| 3.5 | original | 2.0-3.5 | 335 | 2.881 | 0.062 | 0.105 | 0.017 | 0.036 |
| 3.5 | original | 3.5-5.0 | 13888 | 4.338 | 0.215 | 0.430 | 0.091 | 0.099 |
| 4.5 | actuator_ft | 0.0-2.0 | 183 | 0.876 | 0.030 | 0.055 | 0.205 | 0.153 |
| 4.5 | actuator_ft | 2.0-3.5 | 130 | 2.802 | 0.041 | 0.072 | 0.046 | 0.049 |
| 4.5 | actuator_ft | 3.5-5.0 | 2974 | 4.567 | 0.217 | 0.432 | 0.048 | 0.105 |
| 4.5 | actuator_ft | 5.0-6.5 | 9044 | 5.288 | 0.091 | 0.160 | 0.096 | 0.100 |
| 4.5 | original | 0.0-2.0 | 184 | 0.879 | 0.030 | 0.056 | 0.188 | 0.210 |
| 4.5 | original | 2.0-3.5 | 126 | 2.813 | 0.045 | 0.072 | 0.045 | 0.064 |
| 4.5 | original | 3.5-5.0 | 2868 | 4.556 | 0.309 | 0.542 | 0.046 | 0.120 |
| 4.5 | original | 5.0-6.5 | 9045 | 5.280 | 0.165 | 0.281 | 0.096 | 0.099 |
| 5.5 | actuator_ft | 0.0-2.0 | 170 | 0.815 | 0.029 | 0.057 | 0.352 | 0.184 |
| 5.5 | actuator_ft | 2.0-3.5 | 79 | 2.811 | 0.040 | 0.071 | 0.117 | 0.060 |
| 5.5 | actuator_ft | 3.5-5.0 | 982 | 4.699 | 0.175 | 0.314 | 0.052 | 0.074 |
| 5.5 | actuator_ft | 5.0-6.5 | 9731 | 5.854 | 0.109 | 0.196 | 0.079 | 0.159 |
| 5.5 | original | 0.0-2.0 | 170 | 0.813 | 0.029 | 0.058 | 0.328 | 0.279 |
| 5.5 | original | 2.0-3.5 | 73 | 2.793 | 0.043 | 0.071 | 0.125 | 0.084 |
| 5.5 | original | 3.5-5.0 | 921 | 4.701 | 0.346 | 0.525 | 0.052 | 0.069 |
| 5.5 | original | 5.0-6.5 | 9668 | 5.849 | 0.180 | 0.291 | 0.078 | 0.149 |
| 6.5 | actuator_ft | 0.0-2.0 | 163 | 0.785 | 0.029 | 0.059 | 0.518 | 0.206 |
| 6.5 | actuator_ft | 2.0-3.5 | 53 | 2.796 | 0.042 | 0.069 | 0.171 | 0.053 |
| 6.5 | actuator_ft | 3.5-5.0 | 98 | 4.414 | 0.038 | 0.071 | 0.101 | 0.072 |
| 6.5 | actuator_ft | 5.0-6.5 | 3272 | 5.875 | 0.152 | 0.358 | 0.046 | 0.138 |
| 6.5 | actuator_ft | 6.5-7.5 | 6308 | 6.642 | 0.067 | 0.121 | 0.028 | 0.178 |
| 6.5 | original | 0.0-2.0 | 160 | 0.762 | 0.029 | 0.060 | 0.499 | 0.320 |
| 6.5 | original | 2.0-3.5 | 53 | 2.752 | 0.042 | 0.070 | 0.173 | 0.080 |
| 6.5 | original | 3.5-5.0 | 84 | 4.364 | 0.048 | 0.080 | 0.108 | 0.072 |
| 6.5 | original | 5.0-6.5 | 3273 | 5.913 | 0.240 | 0.494 | 0.042 | 0.157 |
| 6.5 | original | 6.5-7.5 | 6174 | 6.632 | 0.150 | 0.255 | 0.026 | 0.177 |
| 7.5 | actuator_ft | 0.0-2.0 | 150 | 0.686 | 0.030 | 0.060 | 0.687 | 0.247 |
| 7.5 | actuator_ft | 2.0-3.5 | 46 | 2.665 | 0.035 | 0.063 | 0.322 | 0.089 |
| 7.5 | actuator_ft | 3.5-5.0 | 60 | 4.249 | 0.038 | 0.069 | 0.190 | 0.123 |
| 7.5 | actuator_ft | 5.0-6.5 | 1571 | 5.977 | 0.168 | 0.378 | 0.079 | 0.088 |
| 7.5 | actuator_ft | 6.5-7.5 | 7222 | 7.194 | 0.080 | 0.131 | 0.016 | 0.175 |
| 7.5 | original | 0.0-2.0 | 150 | 0.686 | 0.030 | 0.060 | 0.656 | 0.361 |
| 7.5 | original | 2.0-3.5 | 43 | 2.641 | 0.035 | 0.066 | 0.342 | 0.109 |
| 7.5 | original | 3.5-5.0 | 60 | 4.285 | 0.043 | 0.068 | 0.198 | 0.129 |
| 7.5 | original | 5.0-6.5 | 1400 | 5.996 | 0.307 | 0.528 | 0.078 | 0.118 |
| 7.5 | original | 6.5-7.5 | 7229 | 7.178 | 0.134 | 0.235 | 0.017 | 0.183 |
| 8.0 | actuator_ft | 0.0-2.0 | 150 | 0.687 | 0.030 | 0.060 | 0.769 | 0.268 |
| 8.0 | actuator_ft | 2.0-3.5 | 40 | 2.691 | 0.035 | 0.062 | 0.382 | 0.111 |
| 8.0 | actuator_ft | 3.5-5.0 | 52 | 4.254 | 0.037 | 0.060 | 0.243 | 0.128 |
| 8.0 | actuator_ft | 5.0-6.5 | 1112 | 6.068 | 0.107 | 0.195 | 0.092 | 0.077 |
| 8.0 | actuator_ft | 6.5-7.5 | 2259 | 7.120 | 0.162 | 0.352 | 0.043 | 0.187 |
| 8.0 | actuator_ft | 7.5-8.5 | 5078 | 7.599 | 0.062 | 0.101 | 0.019 | 0.171 |
| 8.0 | original | 0.0-2.0 | 150 | 0.687 | 0.030 | 0.060 | 0.738 | 0.380 |
| 8.0 | original | 2.0-3.5 | 40 | 2.721 | 0.036 | 0.063 | 0.379 | 0.117 |
| 8.0 | original | 3.5-5.0 | 46 | 4.285 | 0.042 | 0.059 | 0.246 | 0.135 |
| 8.0 | original | 5.0-6.5 | 964 | 6.085 | 0.260 | 0.460 | 0.092 | 0.121 |
| 8.0 | original | 6.5-7.5 | 2475 | 7.140 | 0.193 | 0.489 | 0.041 | 0.214 |
| 8.0 | original | 7.5-8.5 | 4843 | 7.590 | 0.129 | 0.222 | 0.019 | 0.166 |
