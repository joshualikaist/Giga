# Reward and gait design in open-source small biped/humanoid RL code bases (code-cited)

What was read (shallow clones of current default branches, read-only, nothing executed):
- hshi74/toddlerbot @ `e337f3b` (commit date 2026-04-19)
- google-deepmind/mujoco_playground @ `d59156e` (2026-10-07): locomotion/berkeley_humanoid, op3, t1, g1, h1 (joystick, joystick_gait_tracking), gait.py
- HybridRobotics/Berkeley-Humanoid-Lite @ `984741a` (2026-03-10) plus the asset submodule HybridRobotics/Berkeley-Humanoid-Lite-Assets @ `fc90fed` (2025-09-07)
- roboterax/humanoid-gym @ `ae46e20` (2024-11-01; this is the newest commit on main)
- unitreerobotics/unitree_rl_gym @ `276801e` (2025-07-25): g1, h1, h1_2
- leggedrobotics/legged_gym @ `8fa29ac` (2025-05-29): cassie
- Papers, read as summaries through a fetch tool, so check before quoting: ToddlerBot (arXiv 2502.00893), MuJoCo Playground (arXiv 2502.08844), Humanoid-Gym (arXiv 2404.05695), Berkeley Humanoid Lite (arXiv 2504.17249).

Conventions: all code links point to the commit above, with line anchors. "×dt" means the summed reward is multiplied by the control dt. ToddlerBot, Playground and legged_gym-family code all do this, so weights are roughly comparable "per second". Weight signs follow each repo's own convention. ToddlerBot penalty functions return negative values and take positive scales. Playground and legged_gym `_cost_*`/penalty functions return positive values and take negative scales.

---

## 1. Reward terms and weights for each walking/joystick task, and the base-orientation (roll/pitch) terms

### Takeaway
Every project uses an exponential velocity-tracking reward plus a small base-orientation term. Most use L2 on projected gravity / up-vector xy at weight −1 to −2 (OP3 and H1-joystick use −5), plus an xy angular-velocity penalty of −0.05 to −0.15. ToddlerBot, the closest in scale (~0.21 m leg, servo-driven), handles orientation with a bounded exp-kernel quaternion reward (exp(−20·θ²), weight 2.5) plus a squared xy-gyro penalty (weight 1.0), and it also has deadband roll/pitch rewards (±0.1 rad roll band). Humanoid-gym uses a positive exp-kernel orientation reward. Cassie in legged_gym, a biped without ankle roll, uses no orientation penalty at all.

### Cited Findings

**ToddlerBot (MJX + Brax-style PPO, gin configs). Current `walk.gin` block is labeled "Without ZMP reference motion".**
- Active weights: `lin_vel_xy` 2.0 with `lin_vel_tracking_sigma` 1000 (exp(−1000·‖e‖²)); `ang_vel_z` 1.5 with sigma 4.0; `penalty_ang_vel_xy` 1.0; `torso_quat` 2.5; `penalty_action_rate` 2.0; `penalty_pose` 0.5 with per-joint `pose_weights` [hip_pitch 0.01, hip_roll 1.0, hip_yaw 5.0, knee 0.01, ankle_roll 5.0, ankle_pitch 5.0] per leg; `penalty_close_feet_xy` 10.0 (threshold 0.06 m); `alive` 1.0; `feet_phase` 7.5; `penalty_feet_ori` 5.0; `swing_height` 0.04 m; `feet_phase_tracking_sigma` 0.0007. `penalty_orientation` 10.0 is present but commented out — [walk.gin L110-L131](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L110-L131)
- An older (commented-out) weight set that uses the ZMP reference: torso_pos_xy 1.0, torso_quat 5.0, torso_roll 0.5, torso_pitch 0.5, lin_vel_xy 5.0, lin_vel_z 1.0, ang_vel_xy 2.0, ang_vel_z 5.0, motor_pos 5.0, motor_torque 0.5, energy 0.05, action_rate 1.0, feet_contact 1.0, collision 0.1, survival 10.0, feet_air_time 500.0, feet_distance 1.0, feet_slip 0.05, feet_clearance 1.0, hip_yaw 1.0 — [walk.gin L55-L74](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L55-L74). A second commented block ("Dense foot-height tracking") sets torso_roll/torso_pitch to 0.0 and zeroes motor_pos, feet_air_time, feet_clearance, feet_contact and feet_slip — [walk.gin L76-L108](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L76-L108). `walk_fixed.gin` (fixed base) sets motor_pos 5.0 — [walk_fixed.gin L14](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_fixed.gin#L14)
- The ToddlerBot paper summary lists "Torso quaternion: 1.0", "Linear velocity (XY): 5.0", "Feet air time: 500.0" — [arXiv 2502.00893](https://arxiv.org/abs/2502.00893). This disagrees with the gin comment (torso_quat 5.0); see Gaps.
- Sigma defaults: pos 200, rot 20, lin_vel 200, ang_vel 0.5. Feet y-distance band 0.07–0.13 m, feet xy-distance band 0.07–0.20 m. `torso_roll_range` [−0.1, 0.1] rad, `torso_pitch_range` [−0.2, 0.2] rad — [mjx_config.py L111-L140](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L111-L140)
- `torso_quat` = exp(−rot_sigma·angle²), where the angle is the full rotation between torso and reference. The reference is `path_rot * default_rot`: upright, with the yaw obtained by integrating the commanded yaw rate. So this one term penalizes roll, pitch and yaw drift together — [mjx_env.py L2303-L2327](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2303-L2327); [walk_zmp_ref.py L228-L235](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/walk_zmp_ref.py#L228-L235)
- `penalty_ang_vel_xy` = −Σ(ω_xy)² in the torso frame. `penalty_orientation` = −Σ(g_xy)² with gravity projected into the torso frame. The docstrings say both "match holosoma's" terms — [mjx_env.py L2398-L2435](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2398-L2435)
- `torso_roll`/`torso_pitch` are deadband rewards: (exp(−100·|below-min|) + exp(−100·|above-max|))/2, so roll within ±0.1 rad is not penalized at all — [walk_env.py L324-L376](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L324-L376)
- Summed reward ×dt, with the clip to ≥0 commented out — [mjx_env.py L1700-L1702](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L1700-L1702)
- Command sampling: 20% zero, 20% pure turn, otherwise an elliptical x/y sample that includes backward. Walk ranges are x ±0.1 m/s, y ±0.05 m/s, yaw ±1.0 rad/s, with deadzone [0.05, 0.05, 0.2] — [walk_env.py L235-L321](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L235-L321); [walk.gin L42-L53](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L42-L53); [mjx_config.py L193-L199](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L193-L199)

**MuJoCo Playground, Berkeley Humanoid joystick**
- tracking_lin_vel 1.0, tracking_ang_vel 0.5 (tracking_sigma 0.5); lin_vel_z 0; ang_vel_xy −0.15; orientation −1.0; base_height 0; torques −2.5e-5; action_rate −0.01; feet_air_time 2.0; feet_slip −0.25; feet_phase 1.0; feet_clearance 0; feet_height 0; termination −1.0; joint_deviation_knee −0.1; joint_deviation_hip −0.25; dof_pos_limits −1.0; pose −1.0; max_foot_height 0.1 m — [berkeley_humanoid/joystick.py L32-L99](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L32-L99)
- The orientation cost is Σ(up-vector_xy)². The "gravity" sensor is a `framezaxis` up-vector at the IMU site. ang_vel_xy uses the global angular velocity — [joystick.py L528-L535](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L528-L535); [berkeley_humanoid_mjx_feetonly.xml L193](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/berkeley_humanoid_mjx_feetonly.xml#L193)
- Total reward is clipped to [0, 10000] after ×dt — [joystick.py L321-L324](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L321-L324)
- Pose-cost weights per leg [HR 1.0, HAA 1.0, HFE 0.01, KFE 0.01, FFE 1.0, FAA 1.0] (squared). Sagittal hip and knee are nearly free; hip rotation, abduction and both ankle joints are regularized — [joystick.py L144-L147](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L144-L147); [L598-L599](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L598-L599)
- `joint_deviation_hip` (L1 on HR/HAA) is multiplied by `|cmd_y| > 0.1`, i.e. it is active only when a lateral command is given — [joystick.py L582-L589](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L582-L589)
- Commands: x, y, yaw each U[−1, 1]; 10% all-zero; resampled every 500 steps — [joystick.py L667-L687](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L667-L687); [L333-L338](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L333-L338)

**MuJoCo Playground, OP3 joystick (Robotis OP3, small Dynamixel humanoid)**
- tracking_lin_vel 1.5, tracking_ang_vel 0.8 (sigma 0.25); lin_vel_z −2.0; ang_vel_xy −0.05; orientation −5.0; torques −0.0002; action_rate −0.01 (first plus second difference); zero_cmd −0.5 (Σaction² when |cmd|<0.1); termination −1.0; feet_slip −0.1; feet_clearance 0.0; energy −0.0001. No gait clock and no air-time term — [op3/joystick.py L30-L69](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L30-L69); [L279-L385](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L279-L385)
- Early termination when up-vector z < 0.85 (tilt above about 31.8°) or torso z < 0.21 m, and on any joint-limit exceedance — [op3/joystick.py L239-L255](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L239-L255)
- Commands: x [−0.6, 1.5], y [−0.8, 0.8], yaw [−0.7, 0.7] — [op3/joystick.py L43-L45](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L43-L45)

**MuJoCo Playground, Booster T1 joystick**
- tracking_lin_vel 1.0, tracking_ang_vel 0.5 (sigma 0.25); ang_vel_xy −0.15 (local gyro); orientation −1.0; feet_air_time 2.0; feet_slip −0.25; feet_phase 1.0; alive 0.25; termination 0.0; joint_deviation_knee −0.1; joint_deviation_hip −0.1; dof_pos_limits −1.0; pose −1.0; feet_distance −1.0; collision −1.0; max_foot_height 0.12 m; torques, action_rate and energy all 0 — [t1/joystick.py L32-L100](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L32-L100)
- Leg pose weights [hip_pitch 0.01, hip_roll 1.0, hip_yaw 1.0, knee 0.01, ankle_pitch 1.0, ankle_roll 1.0] — [t1/joystick.py L147-L153](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L147-L153)

**MuJoCo Playground, Unitree G1 joystick**
- tracking_lin_vel 1.0, tracking_ang_vel 0.75; ang_vel_xy −0.15 (torso); orientation −2.0; feet_air_time 2.0; feet_slip −0.25; feet_phase 1.0; stand_still −1.0; termination −100.0; collision −0.1; contact_force −0.01; joint_deviation_knee −0.1; joint_deviation_hip −0.25; dof_pos_limits −1.0; pose −0.1; max_foot_height 0.15 m — [g1/joystick.py L32-L107](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/g1/joystick.py#L32-L107)
- G1 orientation cost targets a non-zero lean: Σ(torso_zaxis − [0.073, 0, 1])² — [g1/joystick.py L699-L700](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/g1/joystick.py#L699-L700)

**MuJoCo Playground, H1**
- Plain joystick: tracking_lin_vel 1.5, tracking_ang_vel 0.8, lin_vel_z −2.0, ang_vel_xy −0.05, orientation −5.0, torques −0.0002, action_rate −0.2, stand_still −0.5, termination −1.0, feet_slip −0.1, feet_clearance −0.5 — [h1/joystick.py L32-L58](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick.py#L32-L58)
- Gait-tracking joystick: feet_phase 5.0, tracking_lin_vel 3.5, tracking_ang_vel 0.75, pose −2.5, action_rate −0.01; ang_vel_xy, lin_vel_z and foot_slip at 0. It has no orientation term. A multiplicative alternative r_pos·exp(0.2·r_neg) is left commented out — [h1/joystick_gait_tracking.py L36-L80](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py#L36-L80); [L247-L256](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py#L247-L256)

**Berkeley Humanoid Lite (Isaac Lab, biped config)**
- track_lin_vel_xy (yaw frame, std 0.25) 2.0; track_ang_vel_z (std 0.25) 1.0; termination −10.0; lin_vel_z_l2 −0.1; ang_vel_xy_l2 −0.05; flat_orientation_l2 −2.0; action_rate_l2 −0.01; joint_torques_l2 −2e-3; joint_acc_l2 −1e-6; joint_pos_limits −1.0; feet_air_time_positive_biped (threshold 0.4 s) 1.0; feet_slide −0.1; undesired_contacts (base/hip/knee) −1.0; joint_deviation_l1 on hip yaw/roll −0.2; on ankle roll −0.2 — [biped/env_cfg.py L104-L204](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L104-L204)
- Terminates on base tilt above 0.78 rad — [biped/env_cfg.py L215-L218](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L215-L218)
- Commands: x ±0.5, y ±0.25, yaw ±1.0, heading control on, 2% standing envs — [biped/env_cfg.py L25-L39](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L25-L39)
- The full humanoid config relaxes this: std 0.5, flat_orientation −1.0, air_time 2.0 with threshold 0.5, hip/ankle deviation −1.0 — [humanoid/env_cfg.py](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/humanoid/env_cfg.py)

**Humanoid-Gym (XBot-L)**
- joint_pos (reference tracking) 1.6; feet_clearance 1.0; feet_contact_number 1.2; feet_air_time 1.0; foot_slip −0.05; feet_distance 0.2; knee_distance 0.2; feet_contact_forces −0.01; tracking_lin_vel 1.2; tracking_ang_vel 1.1; vel_mismatch_exp 0.5; low_speed 0.2; track_vel_hard 0.5; default_joint_pos 0.5; orientation 1.0; base_height 0.2; base_acc 0.2; action_smoothness −0.002; torques −1e-5; dof_vel −5e-4; dof_acc −1e-7; collision −1.0. Also tracking_sigma 5, `only_positive_rewards = True` — [humanoid_config.py L175-L216](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L175-L216)
- The orientation term is a positive reward: (exp(−10·(|roll|+|pitch|)) + exp(−20·‖g_xy‖))/2 — [humanoid_env.py L346-L353](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L346-L353)
- vel_mismatch_exp = (exp(−10·v_z²) + exp(−5·‖ω_xy‖))/2. base_acc = exp(−3·‖Δroot_vel‖) — [humanoid_env.py L386-L406](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L386-L406)
- Scales ×dt — [legged_robot.py L528](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/base/legged_robot.py#L528)

**unitree_rl_gym (G1 12-DoF; H1 and H1_2 use the same weights)**
- tracking_lin_vel 1.0, tracking_ang_vel 0.5, lin_vel_z −2.0, ang_vel_xy −0.05, orientation −1.0, base_height −10.0 (target 0.78 m), dof_acc −2.5e-7, dof_vel −1e-3, feet_air_time 0, collision 0, action_rate −0.01, dof_pos_limits −5.0, alive 0.15, hip_pos −1.0, contact_no_vel −0.2, feet_swing_height −20.0, contact 0.18 — [g1_config.py L69-L89](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_config.py#L69-L89); [h1_config.py L83-L100](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/h1/h1_config.py#L83-L100)
- orientation = Σ(projected_gravity_xy)². ang_vel_xy = Σ(base ω_xy)² — [legged_robot.py L641-L647](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/base/legged_robot.py#L641-L647)
- `hip_pos` = Σ q² of G1 DoFs [1, 2, 7, 8], which are hip_roll and hip_yaw under the G1 URDF joint order — [g1_env.py L122-L123](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L122-L123)
- Base defaults: only_positive_rewards True, tracking_sigma 0.25 — [legged_robot_config.py L119-L120](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/base/legged_robot_config.py#L119-L120)

**legged_gym Cassie (biped without ankle roll)**
- termination −200, tracking_ang_vel 1.0, torques −5e-6, dof_acc −2e-7, lin_vel_z −0.5, feet_air_time 5.0, dof_pos_limits −1.0, no_fly 0.25, dof_vel 0, ang_vel_xy 0, feet_contact_forces 0. `only_positive_rewards = False` — [cassie_config.py L83-L100](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/cassie/cassie_config.py#L83-L100)
- Inherited defaults include orientation −0 (off), tracking_lin_vel 1.0, collision −1, action_rate −0.01 — [legged_robot_config.py L132-L146](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/base/legged_robot_config.py#L132-L146)
- `no_fly` gives 1 when exactly one foot is in contact — [cassie.py L43-L46](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/cassie/cassie.py#L43-L46)

### Inferences
- Typical scale of orientation penalties with ×dt is L2(g_xy) at weight 1–2, or 5 for OP3/H1-joystick, which have no gait shaping. A 25° roll gives g_xy² ≈ sin²25° ≈ 0.18, i.e. a cost of 0.18–0.9 per second against tracking rewards of 1–2 per second. None of these projects relies on a large linear roll penalty.
- Bounded kernels may help with "stronger penalty → the policy refuses backward/turn commands". ToddlerBot's exp(−20θ²) and deadband roll terms, and Humanoid-Gym's positive exp orientation reward, can only remove up to their weight; they cannot dominate the return. Unbounded quadratic or linear penalties can, especially when tracking a hard command (backward, in-place turn) means more sway. This is an inference; none of the repos documents the effect.
- ToddlerBot's current main config replaces sparse terms with dense ones (feet_phase instead of air_time/clearance) and lowers velocity weights relative to its commented v1 set (lin_vel 5→2, ang_vel_z 5→1.5). With a 1000 sigma, though, the velocity term is very sharp: a 0.03 m/s error gives exp(−0.9) ≈ 0.41.

### Gaps
- The ToddlerBot clone is shallow, so the date of the switch from ZMP imitation (motor_pos) to `feet_phase` could not be determined. CHANGELOG v2.0.0 (2025-08-25) still mentions "More stable ZMP walking" — [CHANGELOG.md](https://github.com/hshi74/toddlerbot/blob/e337f3b/CHANGELOG.md).
- The ToddlerBot paper's torso_quat weight (1.0, per the fetched summary) conflicts with the gin comment (5.0). Not resolved.
- H1's `hip_pos` indices [0, 1, 5, 6] were not checked against the H1 URDF joint order.

---

## 2. Foot swing shaping: periodic foot-height trajectories, contact schedules, air time, step width, slip, foot orientation

### Takeaway
Two families exist:
- Dense foot-height tracking against a phase clock (Playground `feet_phase`; ToddlerBot's modified `feet_phase`).
- Contact-schedule matching plus a swing-height target (Humanoid-Gym, unitree_rl_gym).

BHL and Cassie use neither and rely only on air-time / single-stance rewards. ToddlerBot's version is the most relevant to a small robot: each foot swings for half the cycle and stands for the other half, with swing apex 0.04 m and the tracking sigma rescaled to 0.0007 m² (Playground's 0.01 m² is tuned for 0.1–0.15 m swings). It also adds a foot-flatness penalty and a close-feet penalty.

### Cited Findings
- Playground `get_rz(phi, swing_height)`: with x = (phi+π)/(2π), the target rises with the smoothstep 3t²−2t³ from 0 to swing_height over x∈[0, 0.5], then falls back over x∈[0.5, 1]. The target is zero only at phi = ±π, so there is no flat stance window. Foot phases are offset by π — [gait.py L32-L43](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/gait.py#L32-L43); [berkeley_humanoid/joystick.py L228-L230](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L228-L230)
- Playground `feet_phase` = exp(−Σ_feet(z_foot − rz)²/0.01). BH and T1 do not gate it by command (a "TODO(kevin): Ensure no movement at 0 command" note) — [berkeley_humanoid/joystick.py L648-L665](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L648-L665); [t1/joystick.py L717-L734](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L717-L734). G1 multiplies it by a mask that is on when |cmd| > 0.01, ‖v_xy‖ > 0.1 or |ω_z| > 0.1 — [g1/joystick.py L788-L809](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/g1/joystick.py#L788-L809)
- The Playground paper writes the phase reward as r_phase = k·exp(−‖p_f,z − r_z(φ)‖²/σ) and says the phase is given to the policy so it can "synchronize its actions with the desired gait cycle" — [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)
- Playground air time: Σ clip(air_time − 0.2, max 0.3) at first contact, zeroed when |cmd| < 0.1 — [berkeley_humanoid/joystick.py L633-L646](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L633-L646)
- Playground `feet_height`: ((swing_peak/max_foot_height) − 1)² at first contact, i.e. the apex reached during the swing that just ended. `feet_clearance`: |z − max_h|·√‖v_xy‖. Both are 0 in BH/T1/G1 (H1 joystick uses feet_clearance −0.5) — [berkeley_humanoid/joystick.py L611-L631](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L611-L631)
- Playground BH and T1 `feet_slip` multiply the base (not foot) xy speed by contact — [berkeley_humanoid/joystick.py L603-L609](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L603-L609); [t1/joystick.py L672-L678](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L672-L678). OP3 uses the feet's xy speed² × contact — [op3/joystick.py L387-L403](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L387-L403)
- T1 `feet_distance` cost = clip(0.2 − lateral_foot_distance, 0, 0.1), with distance measured in the base-yaw frame — [t1/joystick.py L736-L748](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L736-L748)
- ToddlerBot `_expected_foot_height`: left foot swings for phase ∈ [0, π] and is grounded for [π, 2π], right foot the opposite. Within a swing the target rises then falls with the same smoothstep. The docstring says this guarantees "one foot is always on the ground" — [walk_env.py L583-L629](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L583-L629)
- ToddlerBot `feet_phase` = exp(−(e_L² + e_R²)/σ) × (1 + max_expected/swing_height), a bonus of up to 2× near the swing apex that "Incentivizes lifting feet". Both targets are 0 under a zero command. σ = 0.0007, commented as "Scaled for swing_height=0.03 (0.8 * 0.03²)" — [walk_env.py L631-L695](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L631-L695); [walk.gin L128-L131](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L128-L131)
- ToddlerBot `penalty_feet_ori` = −(‖g_xy in left-foot frame‖ + ‖g_xy in right-foot frame‖), "Prevent tippy-toe (keep feet flat)", weight 5.0 — [walk_env.py L697-L736](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L697-L736)
- ToddlerBot step-width terms:
  - `penalty_close_feet_xy` gives −1 when the lateral foot distance in the base-yaw frame is below 0.06 m (weight 10) — [mjx_env.py L2709-L2745](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2709-L2745)
  - `feet_distance` is a deadband reward on the torso-frame y distance within [0.07, 0.13] m — [walk_env.py L426-L453](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L426-L453)
  - `feet_xy_distance` is a binary reward for xy distance within [0.07, 0.20] m — [walk_env.py L455-L490](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L455-L490)
- ToddlerBot's sparse terms (now weight 0): feet_air_time (sum of air time at first contact, no threshold, zeroed at zero command), feet_clearance (accumulated swing height at touchdown), feet_slip (−Σ v_xy² × stance) — [walk_env.py L378-L511](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L378-L511). `feet_contact` = number of feet whose contact matches the reference stance mask — [mjx_env.py L2464-L2487](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2464-L2487)
- Humanoid-Gym:
  - `feet_contact_number` = mean over feet of (+1 if contact matches stance_mask, else −0.3) — [humanoid_env.py L336-L344](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L336-L344)
  - `feet_clearance` counts swing-phase feet whose accumulated lift is within 1 cm of target_feet_height 0.06 m — [humanoid_env.py L446-L467](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L446-L467)
  - `feet_air_time` = clamp(air, 0, 0.5) at first contact, with contacts OR-ed with the stance mask — [humanoid_env.py L320-L334](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L320-L334)
  - `feet_distance`/`knee_distance` are deadband rewards on Euclidean xy distance, feet within [0.2, 0.5] m and knees within [0.2, 0.25] m — [humanoid_env.py L282-L305](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L282-L305)
  - `foot_slip` = Σ√‖v_xy‖ × contact — [humanoid_env.py L308-L318](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L308-L318)
- unitree_rl_gym:
  - `contact` (0.18) counts feet where (contact == is_stance), with is_stance = leg_phase < 0.55 — [g1_env.py L98-L104](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L98-L104)
  - `feet_swing_height` (−20) = Σ(z_foot − 0.08)² for feet not in contact — [g1_env.py L106-L109](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L106-L109)
  - `contact_no_vel` (−0.2) = Σ v_foot² (including z) for feet in contact — [g1_env.py L115-L120](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L115-L120)
- BHL `feet_air_time_positive_biped`: during single stance, reward the min over feet of the current mode time (air or contact), clamped at 0.4 s and zeroed at small command. `feet_slide` = ‖v_xy‖ × contact — [rewards.py L36-L72](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py#L36-L72)
- legged_gym `feet_air_time` = Σ(air − 0.5) at first contact (Cassie weight 5.0) — [legged_robot.py L882-L893](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/base/legged_robot.py#L882-L893)

### Inferences
- On sigma, for a duck with a ~3–4 cm swing: Playground's /0.01 kernel is nearly flat at small-robot scale (4 cm error → exp(−0.16) ≈ 0.85), so it would barely shape swing. ToddlerBot's /0.0007 gives exp(−2.3) ≈ 0.10 for the same error. Scale σ with swing_height² as ToddlerBot does.
- On stance windows: the Playground curve keeps both feet at about h/2 at quarter-phase. ToddlerBot's half-stance/half-swing curve has an exact zero-height stance half and no double support. Humanoid-Gym and unitree build short explicit double-support bands into their contact masks (about 6% and 10% of the cycle in total; see §3). Our fixed double-support windows sit between these. A dense height target that is exactly 0 during stance (ToddlerBot style) penalizes foot lift and rocking onto foot edges during stance. For a robot without ankle roll, rocking onto an edge is one way to roll the torso.
- Foot flatness (ToddlerBot `penalty_feet_ori`, gravity in the foot frame) directly penalizes foot roll. With no ankle roll, a flat foot pins the shank's roll, so this term indirectly limits how much leg tilt can turn into torso roll. Our env has no equivalent.

### Gaps
- No project publishes measured foot-height or roll traces in code, so the effect of each swing-shaping style on torso roll is not quantified anywhere I read.

---

## 3. Gait clock: fixed vs commanded frequency, how phase is observed, speed scaling

### Takeaway
No project scales gait frequency with commanded speed:
- Fixed period: ToddlerBot 0.72 s, Humanoid-Gym 0.64 s, unitree 0.8 s.
- Per-episode random frequency, not observed: Playground BH/G1 1.25–1.5 Hz, T1 1.25–1.75 Hz.
- Observed command: Playground H1 gait-tracking randomizes frequency 0.5–2.0 Hz and swing height 0.08–0.4 m and gives both to the policy.
- No clock: OP3, BHL, Cassie.

Phase is always given as sin/cos. Most clocks keep running at zero command, and standing is handled by changing the reward target instead.

### Cited Findings
- ToddlerBot: `cycle_time` 0.72 s; phase signal = [sin(2πt/T), cos(2πt/T)] from episode time — [mjx_config.py L101-L107](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L101-L107); [walk_zmp_ref.py L70-L86](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/walk_zmp_ref.py#L70-L86). The two phase dimensions come first in the observation. With `use_phase_signal=False` the reference motor pos/vel are observed instead — [mjx_env.py L2102-L2124](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2102-L2124); [walk.gin L21-L37](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L21-L37)
- Playground BH: gait_freq ~ U[1.25, 1.5] Hz per episode; phase = [0, π]; obs = [cos(phase), sin(phase)], 4 dims; gait_freq is not in the observation — [berkeley_humanoid/joystick.py L227-L230](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L227-L230); [L400-L422](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L400-L422). T1: U[1.25, 1.75] Hz — [t1/joystick.py L247-L249](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L247-L249). G1: U[1.25, 1.5] Hz, with a commented-out option to freeze the phase at π under zero command "to make the policy stand still" — [g1/joystick.py L294-L296](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/g1/joystick.py#L294-L296); [L394-L401](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/g1/joystick.py#L394-L401)
- Playground H1 gait tracking: gait_frequency U[0.5, 2.0] Hz and foot_height U[0.08, 0.4] m sampled per episode, both in the observation — [h1/joystick_gait_tracking.py L77-L79](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py#L77-L79); [L164-L177](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py#L164-L177); [L341-L355](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py#L341-L355)
- Humanoid-Gym: phase = t/cycle_time (0.64 s). Stance mask: left when sin ≥ 0, right when sin < 0, both when |sin| < 0.1. Obs has [sin, cos, cmd×3]. The clock runs regardless of command — [humanoid_env.py L100-L118](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L100-L118); [L200-L244](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L200-L244); [humanoid_config.py L181](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L181). The paper describes "two double support phases (DS) and two single support phases (SS) within each gait cycle" — [arXiv 2404.05695](https://arxiv.org/abs/2404.05695)
- unitree_rl_gym: period 0.8 s, right offset 0.5, stance if phase < 0.55. Obs appends sin/cos of the left phase — [g1_env.py L55-L91](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L55-L91)
- ToddlerBot ZMP reference timing (used only when imitation is on): `ZMPWalk(robot, cycle_time, 2.0)`, i.e. single:double ratio 2.0. Double support = T/2/(ratio+1) = 0.12 s and single support 0.24 s at T = 0.72 s — [walk_zmp_ref.py L43](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/walk_zmp_ref.py#L43); [zmp_walk.py L27-L55](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/algorithms/zmp_walk.py#L27-L55)
- OP3, BHL and Cassie have no phase input. The OP3 obs is gyro, gravity, command, joint pos, last action (49 dims) with a 3-step history — [op3/joystick.py L257-L277](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L257-L277). BHL: command, ang vel, projected gravity, joint pos/vel, last action — [biped/env_cfg.py L42-L88](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L42-L88)
- History lengths: ToddlerBot frame_stack 15 (critic 15) — [mjx_config.py L83-L84](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L83-L84). Humanoid-Gym actor 15 frames, critic 3 — [humanoid_config.py L40-L45](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L40-L45). unitree uses an LSTM (64 hidden) with [32] MLPs — [g1_config.py L93-L105](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_config.py#L93-L105)

### Inferences
- Double-support fraction per cycle:
  - Humanoid-Gym: |sin(2πp)| < 0.1 gives 2 × 0.0319 ≈ 6.4% of the cycle, about 2 × 20 ms at 0.64 s.
  - unitree: 2 × 5% = 10%, about 2 × 40 ms at 0.8 s.
  - ToddlerBot's ZMP reference: 2 × 0.12 s = 33% of 0.72 s.
  - ToddlerBot's RL feet_phase target: 0%.
  Our fixed 0.6 s with fixed double-support windows can be compared against these bands.
- No repo ties period to speed. The Playground recipe (random per-episode frequency, unobserved) aims at robustness to timing. The H1 recipe (observed frequency) lets the operator choose a cadence per command at deployment, e.g. faster cadence for in-place turning. One possible lever for lateral sway: per LIPM intuition, a shorter single-support time means less lateral CoM excursion per step. That is my inference, not something any repo states.

### Gaps
- None of the repos documents why its period was chosen. No ablation of period vs roll was found.

---

## 4. Reference motions: who imitates what, and how the reference is generated

### Takeaway
ToddlerBot builds a closed-form ZMP reference: footsteps from the command, then LQR preview control of the ZMP, stored as a lookup table indexed by command. It historically tracked that reference with `motor_pos` (weight 5.0). Current main walking uses the reference only for the path, yaw and orientation targets, not joint imitation. Humanoid-Gym uses half-rectified joint sinusoids on hip pitch, knee and ankle pitch. Playground, unitree, BHL and Cassie use no reference motion.

### Cited Findings
- ToddlerBot `WalkZMPReference`: loads or builds `motion/walk_zmp_<robot>.lz4`. `get_state_ref` picks the nearest-command clip (after heading-error correction), indexes it by time, and uses the default pose when static. Root pos/rot come from command integration (`path_rot * default_rot`) — [walk_zmp_ref.py L36-L68](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/walk_zmp_ref.py#L36-L68); [L128-L250](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/walk_zmp_ref.py#L128-L250); [motion_ref.py L200-L235](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/reference/motion_ref.py#L200-L235)
- ZMPWalk: foot_step_height default 0.05 m, LQR cost Q 1.0 / R 0.1, CoM height = home z. Strides = command × total_time/(2·num_cycles − 1), with alternating left/right footsteps and turning steps. ZMP targets are the footstep xy, timed with DS/SS durations — [zmp_walk.py L27-L80](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/algorithms/zmp_walk.py#L27-L80); [L214-L354](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/algorithms/zmp_walk.py#L214-L354)
- The paper says walking uses "a closed-form ZMP (Zero Moment Point) solution" to generate references that an imitation reward tracks, alongside regularization and survival terms — [arXiv 2502.00893](https://arxiv.org/abs/2502.00893)
- `motor_pos` reward: exp-kernel (or MSE) of leg/arm/neck/waist motor-position error against `state_ref["motor_pos"]` — [mjx_env.py L2500-L2569](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L2500-L2569). In current `walk.gin` motor_pos is not set (default 0.0) and is listed under "Disabled ZMP-dependent" in the commented block — [walk.gin L103-L108](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L103-L108); [mjx_config.py L156](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L156)
- Humanoid-Gym `compute_ref_state`: with s = sin(2πφ), the left leg uses s clipped to ≤ 0 and the right leg s clipped to ≥ 0. Hip pitch and ankle pitch get s·0.17 rad, knee s·0.34 rad, and all are 0 when |s| < 0.1. `ref_action = 2·ref_dof_pos` is added to actions only if `use_ref_actions` (default False) — [humanoid_env.py L121-L142](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L121-L142); [humanoid_config.py L49](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L49); [L179](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L179)
- The Humanoid-Gym joint_pos reward = exp(−2‖q − q_ref‖) − 0.2·clamp(‖q − q_ref‖, 0, 0.5) (weight 1.6). The paper: "A sinusoidal wave is employed to generate reference motion, reflecting the repetitive nature of the gait cycle involving pitch, knee, and ankle movements" — [humanoid_env.py L272-L280](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L272-L280); [arXiv 2404.05695](https://arxiv.org/abs/2404.05695)

### Inferences
- The Humanoid-Gym sinusoid is purely sagittal (no hip roll) and command-independent, so the same stepping amplitude is requested when standing, walking backward or turning. Its lateral behavior comes entirely from the base reward terms. ToddlerBot's ZMP reference contains a lateral CoM plan, but the current main config no longer imitates joint angles. A plausible reading is that dense foot-height plus orientation/pose terms worked as well as imitation for them. I found no stated rationale.

### Gaps
- The ToddlerBot ZMP lateral CoM amplitude and any roll content of the reference were not extracted; the lookup tables are binary `.lz4` files that I did not decode.

---

## 5. Default pose, action scale, PD gains, control rate, domain randomization (small-servo relevant)

### Takeaway
Control is almost universally 50 Hz. Humanoid-Gym runs at 100 Hz and BHL at 25 Hz. Action scale is 0.25–0.5 rad in most projects; T1 uses 1.0. Small-servo projects model the servo explicitly. ToddlerBot has the richest servo model: sysID'd torque-speed envelope, passive/active asymmetry, backlash, colored IMU noise, 1-step action delay. Playground deliberately softens ankles (BH ankle-roll kp = 1; T1 ankle kp = 10 against 20–50 at the hips).

### Cited Findings
**ToddlerBot**
- 50 Hz control: sim timestep 0.005 s × n_frames 4. action_scale 0.25, n_steps_delay 1 — [mjx_config.py L21](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L21); [L101-L107](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L101-L107). Target = default + 0.25 × delayed action — [mjx_env.py L1537-L1546](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_env.py#L1537-L1546)
- Gains: Dynamixel kp 2100 (2XC430 hip pitch/roll, XM430 knee/ankle pitch) and 1500 (XC330 hip yaw, XC430 ankle roll), kd 0. Home pose: hip pitch ∓0.091, knee ∓0.381, ankle pitch ∓0.29 rad — [default.yml L43-L120](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/descriptions/default.yml#L43-L120). Sim kp = real kp / kp_ratio 150, i.e. 14 and 10 (Nm/rad); passive_active_ratio 3.0 — [default.yml L240-L243](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/descriptions/default.yml#L240-L243); [sim/robot.py L119-L122](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/sim/robot.py#L119-L122)
- Motor model: PD torque, then an asymmetric clamp. Constant tau_max up to q̇_tau_max, a linear taper to q̇_max, a separate brake limit, and a higher kp when back-driven (passive_active_ratio) — [sim/motor_control.py L40-L100](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/sim/motor_control.py#L40-L100). XM430 sysID: damping 0.08, armature 0.0012, frictionloss 0.134, tau_max 1.86 Nm, q̇_tau_max 4.22 rad/s, q̇_max 10 rad/s — [sysID_XM430/config_dynamics.json L3-L8](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/descriptions/sysID_XM430/config_dynamics.json#L3-L8)
- Size: 0.56 m tall, 3.4 kg, 6-DoF legs with a two-DoF parallel-linkage ankle — [arXiv 2502.00893](https://arxiv.org/abs/2502.00893). Leg segments: hip→knee 0.1015 m, knee→ankle 0.11 m — [toddlerbot_2xc/robot.yml L6-L11](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/descriptions/toddlerbot_2xc/robot.yml#L6-L11)
- Domain randomization:
  - Friction 0.4–1.0; damping, armature and frictionloss ×0.8–1.2; body mass ±0.2 (additive); kp, kd, tau_max, q̇ limits, kd_min and brake ×0.9–1.1; passive_active_ratio ×0.5–1.5
  - Backlash 0.02–0.1 rad (activation 0.1)
  - Initial torso roll/pitch ±0.1 rad; persistent torso pitch/roll calibration error (default 0, gin suggests ±0.05 / ±0.025)
  - Pushes 1–3 lasting 0.2 s every 2 s (off by default)
  - Head-pose randomization on in walk.gin
  - [mjx_config.py L201-L247](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L201-L247); [walk.gin L133-L144](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L133-L144)
- IMU noise is low-pass colored with bias random walk: gyro std 0.25 rad/s (fc 0.35 Hz), quaternion std 0.10 rad (fc 0.25 Hz), with amplitude scale ranges — [mjx_config.py L249-L268](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/mjx_config.py#L249-L268)

**MuJoCo Playground**
- BH: ctrl 0.02 s, sim 0.002 s, action_scale 0.5; obs noise hip 0.03, knee 0.05, ankle-pitch 0.08, ankle-roll 0.03 rad, joint vel 1.5, gravity 0.05, gyro 0.2. Pushes 0.1–2.0 m/s every 5–10 s — [berkeley_humanoid/joystick.py L32-L92](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py#L32-L92). Default position kp 35, joint damping 1.5. Ankle roll (FAA): kp 1, force ±5 Nm, damping 0.1 — [berkeley_humanoid_mjx_feetonly.xml L8-L32](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/berkeley_humanoid_mjx_feetonly.xml#L8-L32). Home pose: HR ∓0.071, HAA ±0.103, HFE −0.463, KFE 0.983, FFE −0.350, FAA ±0.126 — [scene_mjx_feetonly_flat_terrain.xml L29-L39](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/scene_mjx_feetonly_flat_terrain.xml#L29-L39). DR: friction U(0.4, 1.0), frictionloss ×0.9–1.1, armature ×1.0–1.05, link masses ×0.9–1.1, torso mass ±1 kg, qpos0 ±0.05 rad — [berkeley_humanoid/randomize.py L30-L66](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/berkeley_humanoid/randomize.py#L30-L66)
- OP3: ctrl 0.02 s, sim 0.004 s, action_scale 0.3, obs noise 0.05 uniform, 3-step history, no DR file — [op3/joystick.py L30-L45](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L30-L45). Actuators: kp 21.1, forcerange ±5 Nm; joint damping 1.084, armature 0.045, frictionloss 0.03 — [op3_mjx_feetonly.xml L64-L66](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml#L64-L66). Bent-knee default (hip pitch ∓0.534, knee ±0.880, ankle pitch ∓0.628) — [op3_mjx_feetonly.xml L294-L300](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml#L294-L300)
- T1: action_scale 1.0 — [t1/joystick.py L38](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/joystick.py#L38). Ankle class kp 10 with joint damping 3, ankle roll range ±0.44 rad and ±15 Nm — [t1_mjx_feetonly.xml L110-L119](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/xmls/t1_mjx_feetonly.xml#L110-L119). DR: friction 0.2–0.6, kp ×0.9–1.1, ankle damping ×0.5–2.0 ("Higher range on the ankles") — [t1/randomize.py L30-L82](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/t1/randomize.py#L30-L82)
- Paper: training takes "under 15 minutes for the Berkeley Humanoid, and under 30 minutes for the Unitree G1 and the Booster T1 on two RTX 4090". Zero-shot real walking is reported on G1, T1 and Berkeley Humanoid. The fetched summary found no OP3 hardware result — [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)

**Berkeley Humanoid Lite**
- 25 Hz control: sim dt 0.005 × decimation 8 — [velocity_env_cfg.py L68-L75](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/velocity_env_cfg.py#L68-L75); [biped/env_cfg.py L346-L348](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L346-L348). action_scale 0.25 — [biped/env_cfg.py L95-L101](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L95-L101)
- Implicit PD: stiffness 20, damping 2, effort 6 Nm, velocity limit 10 rad/s, armature 0.007 (legs) / 0.002 (ankles). Default pose: hip pitch −0.2, knee 0.4, ankle pitch −0.3, roll and yaw 0 — [berkeley_humanoid_lite.py L41-L100](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets/blob/fc90fed/berkeley_humanoid_lite_assets/robots/berkeley_humanoid_lite.py#L41-L100)
- DR: friction 0.4–1.2, base mass −1 to +2 kg, default joint pos ±0.05, stiffness/damping ×0.8–1.2, reset force/torque ±2. Interval pushes are commented out — [biped/env_cfg.py L221-L308](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L221-L308)
- Paper: 0.8 m tall, 16 kg, 3D-printed cycloidal-gearbox BLDC actuators, policy "runs onboard the robot at 25 Hz" — [arXiv 2504.17249](https://arxiv.org/abs/2504.17249)

**Humanoid-Gym (XBot-L)**
- 100 Hz control (dt 0.001 × decimation 10); action_scale 0.25; stiffness hip roll/yaw 200, pitch/knee 350, ankle 15; damping 10. All default joint angles 0 (straight legs) — [humanoid_config.py L101-L131](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L101-L131)
- DR: friction 0.1–2.0, added mass ±5 kg, push every 4 s (0.2 m/s, 0.4 rad/s), action delay blend up to 0.5, action noise 2% — [humanoid_config.py L150-L160](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_config.py#L150-L160); [humanoid_env.py L189-L197](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L189-L197)

**unitree_rl_gym G1**
- 50 Hz (sim dt 0.005 × decimation 4); action_scale 0.25; kp hip 100, knee 150, ankle 40; kd 2/4/2; default hip pitch −0.1, knee 0.3, ankle −0.2. DR: friction 0.1–1.25, mass −1 to +3 kg, push 1.5 m/s every 5 s — [g1_config.py L4-L57](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_config.py#L4-L57); [legged_robot_config.py L68](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/base/legged_robot_config.py#L68)

**legged_gym Cassie**
- action_scale 0.5, decimation 4, kp 100–200 (toe 40), kd 3–6 (toe 1). Default hip abduction ±0.1 rad — [cassie_config.py L45-L73](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/cassie/cassie_config.py#L45-L73)

### Inferences
- For a ~2 kg servo duck, the closest settings are ToddlerBot's (50 Hz, action_scale 0.25, 1-step delay, servo torque-speed envelope, backlash DR, colored IMU noise) and OP3's (kp ≈ 21 Nm/rad, ±5 Nm, 3-step history, no clock). Both projects use action scale ≤ 0.3 rad.
- Playground's soft ankle-roll (BH kp = 1) and soft T1 ankles suggest these designers prefer compliant ankle roll for stability. With no ankle roll at all, the duck is the limiting case of that choice. This is an inference from the gains; no comment in the code explains them.

### Gaps
- ToddlerBot's push magnitude units (N vs m/s) were not verified.
- The Open Duck servo (STS3215) parameters are out of scope here; none of these repos model that servo.

---

## 6. Reducing lateral sway or torso roll; robots without ankle roll

### Takeaway
No repo contains an explicit "lateral sway" or "head sway" term. Lateral behavior is shaped indirectly through:
- orientation (g_xy L2, exp quaternion, deadband roll) and xy-gyro penalties;
- step-width bounds (min lateral foot distance: ToddlerBot 0.06 m, T1 0.2 m, Humanoid-Gym 0.2–0.5 m);
- hip roll/yaw pose regularization (ToddlerBot, Playground, unitree, Humanoid-Gym, BHL);
- foot-flatness (ToddlerBot).

Among the robots here, only Cassie lacks ankle roll. Its legged_gym config uses no orientation penalty and relies on termination plus air-time / no-fly rewards. ToddlerBot, OP3, T1, G1, BHL and Berkeley Humanoid all have an ankle-roll joint, though BH's is nearly passive (kp 1).

### Cited Findings
- ToddlerBot keeps feet at least 6 cm apart laterally (`penalty_close_feet_xy` 10). Its pose penalty weights hip_roll 1.0, hip_yaw 5.0, ankle_roll 5.0 but hip_pitch/knee only 0.01. torso_quat (exp(−20θ²)) and penalty_ang_vel_xy are both active — [walk.gin L111-L131](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L111-L131)
- ToddlerBot's optional DR "Persistent calibration errors ... simulate miscalibrated motors where torso stays tilted during motion" (pitch ±0.05, roll ±0.025 rad, commented) — [walk.gin L141-L144](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk.gin#L141-L144)
- ToddlerBot `hip_yaw` reward penalizes hip yaw only when the turn command is zero — [walk_env.py L563-L581](https://github.com/hshi74/toddlerbot/blob/e337f3b/toddlerbot/locomotion/walk_env.py#L563-L581)
- ToddlerBot teleoperation (paper): "the lower body employs a two-layer PD controller to actively maintain balance", combining a CoM PD and an IMU torso-pitch controller — [arXiv 2502.00893](https://arxiv.org/abs/2502.00893)
- Humanoid-Gym `default_joint_pos` = exp(−100·max(‖Δq_yaw,roll,L‖ + ‖Δq_yaw,roll,R‖ − 0.1, 0)) − 0.01‖Δq‖, a 0.1 rad deadband on hip yaw+roll (weight 0.5) — [humanoid_env.py L362-L372](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L362-L372)
- Humanoid-Gym `base_height` uses stance-mask-weighted foot height as the ground reference: exp(−100·|h − 0.89|) — [humanoid_env.py L374-L384](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L374-L384)
- unitree `hip_pos` (−1.0) penalizes hip roll and yaw squared — [g1_env.py L122-L123](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/legged_gym/envs/g1/g1_env.py#L122-L123)
- BHL penalizes L1 deviation of hip yaw/roll (−0.2) and ankle roll (−0.2) — [biped/env_cfg.py L194-L204](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg.py#L194-L204)
- OP3 has ankle roll actuators (l/r_ank_roll_act) and the strongest orientation weight in the set (−5.0), plus a 31.8° tilt termination — [op3_mjx_feetonly.xml L259-L268](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml#L259-L268); [op3/joystick.py L52](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L52); [L248](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/mujoco_playground/_src/locomotion/op3/joystick.py#L248)
- Cassie's joints are hip abduction, rotation and flexion, thigh (knee), ankle and toe, all sagittal below the hip: no ankle/toe roll. Its reward has orientation −0 and ang_vel_xy −0.0 — [cassie_config.py L46-L60](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/cassie/cassie_config.py#L46-L60); [L89-L100](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/cassie/cassie_config.py#L89-L100); [legged_robot_config.py L137](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/legged_gym/envs/base/legged_robot_config.py#L137)
- The Playground paper lists "Angular Velocity in XY (Penalty)" and "Linear Velocity in Z (Penalty)" among its stabilization terms — [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)

### Inferences (mine, not stated in any repo)
- Mechanics for a 5-DoF leg without ankle roll (hip yaw, roll, pitch, knee, ankle pitch): when the stance foot is flat, the leg's roll relative to ground is fixed, so torso roll ≈ stance hip-roll angle, plus any foot-edge tilt. The torso CoM sits about half a hip width to the side of the stance foot unless the torso rolls toward it. The policy's ways to keep CoM support are:
  1. roll the torso, which is what we see at 24–30°;
  2. tilt the foot onto its edge (preventable with a ToddlerBot-style `penalty_feet_ori`);
  3. step narrower under the body (ToddlerBot allows down to 0.06 m lateral; T1's 0.2 m minimum is for a much larger robot);
  4. shorten single support (higher cadence) so the CoM does not need to sit over the stance foot.

  Reward terms that only penalize roll, without opening options 3 or 4, leave the policy with option 2 or with refusing hard commands.
- Several stability levers in these repos condition on the command rather than applying uniformly:
  - ToddlerBot zeroes target foot height at zero command and gates hip-yaw by turn command;
  - Playground G1 masks feet_phase at zero command;
  - Playground BH applies hip deviation only under lateral commands.

  Our head-sway gate for backward/turn commands fits this pattern. Gating step width or cadence by command type would extend it.
- Clipping the total reward at 0 (Playground BH/OP3/T1; legged_gym-family only_positive_rewards) hides penalties whenever tracking reward is low. ToddlerBot and Cassie do not clip. If our env clips, strong sway penalties mostly bind when tracking is already good; if it does not clip, they can make "do nothing" attractive for hard commands. This is a hypothesis to check against our env's reward code.

### Gaps
- No repo had a reward on head or upper-body lateral velocity or acceleration, and none reported torso roll amplitudes, so I have no quantitative benchmark for "acceptable roll" from these projects.
- Not examined: holosoma (Amazon FAR), which ToddlerBot's terms say they mirror, and Open Duck's own Playground env (covered by another researcher).

---

## 7. Licenses

### Takeaway
All the code is permissively licensed: MIT for ToddlerBot and BHL code, Apache-2.0 for MuJoCo Playground, BSD-3-Clause for Humanoid-Gym, unitree_rl_gym and legged_gym. The Berkeley Humanoid Lite asset repo is CC BY-SA 4.0 (share-alike).

### Cited Findings
- ToddlerBot: MIT License, "Copyright (c) 2023 Haochen Shi" — [LICENSE](https://github.com/hshi74/toddlerbot/blob/e337f3b/LICENSE)
- MuJoCo Playground: Apache License 2.0 — [LICENSE](https://github.com/google-deepmind/mujoco_playground/blob/d59156e/LICENSE)
- Berkeley Humanoid Lite code: MIT, "Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers" (file named LICENCE) — [LICENCE](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite/blob/984741a/LICENCE). Assets repo: "Attribution-ShareAlike 4.0 International" — [LICENCE](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets/blob/fc90fed/LICENCE)
- Humanoid-Gym: no top-level LICENSE file in the clone. Source headers carry "SPDX-License-Identifier: BSD-3-Clause" with "Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD.", and `humanoid/envs/base/LICENSE` is BSD-3-Clause (NVIDIA / ETH Zurich, inherited from legged_gym) — [humanoid_env.py header](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/custom/humanoid_env.py#L1-L32); [base/LICENSE](https://github.com/roboterax/humanoid-gym/blob/ae46e20/humanoid/envs/base/LICENSE)
- unitree_rl_gym: BSD 3-Clause, "Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD." — [LICENSE](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e/LICENSE)
- legged_gym: BSD-3-Clause, "Copyright (c) 2021, ETH Zurich, Nikita Rudin; NVIDIA" — [LICENSE](https://github.com/leggedrobotics/legged_gym/blob/8fa29ac/LICENSE)

### Inferences
- Reward-function ideas and short code patterns from all six code bases can be reused with attribution. Copying BHL meshes or URDFs would bring in CC BY-SA share-alike obligations; the code itself does not.

### Gaps
- Humanoid-Gym's README does not mention a license, so the repo-level license comes only from the per-file SPDX headers.
