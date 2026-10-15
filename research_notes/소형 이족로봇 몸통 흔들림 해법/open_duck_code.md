# How the Open Duck team trains the Open Duck Mini v2 walking policy (code-level notes)

Repos read (shallow clones, read-only; I did not run their code). Commit history was fetched for Playground and the reference generator.
- Open_Duck_Playground `main` @ b9be205 (2025-08-05). The last training-relevant change was 2025-03-24. `joystick.py`, `custom_rewards.py`, `randomize.py` and the backlash MJCF are byte-identical to commit 097875d, titled "CHECKPOINT pretty nice walk on the real robot" (verified by `git diff 097875d HEAD`, which came back empty). So the HEAD config is the real-robot config.
- Open_Duck_reference_motion_generator `main` @ 3d7bc6f (2025-04-03)
- Open_Duck_Mini_Runtime `v2` (default branch) @ 376de65 (2026-09-07)
- Open_Duck_Mini `v2` (default branch) @ b23317a (2026-01-31)
- apirrone/AWD `main` @ 7ef6015 (2024-09-16)
- Rhoban/placo @ b54affb (2026-09-30), read only to check what the parameters mean. The generator pins placo==0.6.3.
- mujoco_playground tag v0.0.3, for the PPO config the Playground imports.

Link prefixes used below:
- PG = https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3
- RMG = https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b
- RT = https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896
- ODM = https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c

---

## Q1. Reward functions (joystick and standing tasks): every term and weight, the imitation reward, and any orientation, roll or head terms

### Takeaway
The joystick walking policy is trained on a small set of terms:
- alive (+20, the largest)
- yaw-rate tracking (+6) and xy velocity tracking (+2.5), both with a very sharp σ = 0.01
- a Disney-BDX-style imitation reward (×1.0). It is dominated by a −15·Σ(q−q_ref)² joint-position term on the 10 leg joints. It also rewards matching the reference contacts and base linear and angular velocity.
- costs: action rate −0.5, torque −1e-3, stand-still −0.2

There is no active orientation, roll, base-height, foot-clearance, air-time, symmetry or head-stabilization term. An orientation cost (−0.5) and a torso-orientation imitation term were both tried and removed on 2025-03-07. Roll is regulated only implicitly:
- by imitating leg joint angles taken from a reference whose trunk roll is ≈0
- by a weak reward for matching the reference's ≈0 roll and pitch rates

### Cited Findings
**Joystick weights.** tracking_lin_vel=2.5, tracking_ang_vel=6.0, torques=−1.0e-3, action_rate=−0.5 (comment: "was −1.5"), stand_still=−0.2 (comment: "was −1.0"), alive=20.0, imitation=1.0, tracking_sigma=0.01 — [joystick.py L77-L88](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L77-L88)
- Flags: `USE_IMITATION_REWARD = True`, `USE_MOTOR_SPEED_LIMITS = True` — [joystick.py L45-L46](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L45-L46)
- Only these 7 terms are computed. "orientation" is commented out: `# "orientation": cost_orientation(...)` — [joystick.py L634-L667](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L634-L667)
- **How the total is formed.** `reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)`, with dt = 0.02. The step total is clipped to be ≥ 0 — [joystick.py L444-L447](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L444-L447)

**Velocity tracking.**
- Linear velocity: `exp(-(err_x² + err_y²)/σ)`. The y error has a dead-band, `y_tol = 0.1`: `error_y = clip(|v_y − cmd_y| − 0.1, 0)`. Lateral velocity error under 0.1 m/s is free — [rewards.py L11-L22](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L11-L22)
- The velocity input is the local IMU-site velocimeter (`local_linvel`) — [joystick.py L635-L639](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L635-L639); [sensors.xml L3](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/sensors.xml#L3)
- Angular velocity: `exp(-(cmd_yaw − gyro_z)²/σ)` — [rewards.py L25-L31](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L25-L31)

**Cost terms.**
- torques = Σ τ² over all 14 actuator forces; action_rate = Σ (a_t − a_{t−1})² over all 14 actions (head included); alive = 1 — [rewards.py L68-L79, L124-L125](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L68-L79)
- stand_still = (Σ|q − q_default| + Σ|q̇|) × (‖cmd[:3]‖ < 0.01). It is called with `ignore_head=False`, so head joints are included — [rewards.py L93-L117](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L93-L117); [joystick.py L659-L666](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L659-L666)

**Imitation reward** ([custom_rewards.py](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/custom_rewards.py#L20-L149)). Internal weights: w_torso_pos=1, w_torso_orientation=1, w_lin_vel_xy=1, w_lin_vel_z=1, w_ang_vel_xy=0.5, w_ang_vel_z=0.5, w_joint_pos=15.0, w_joint_vel=1e-3, w_contact=1.0 (L20-L28). The sub-terms are:
- base linear velocity xy: `exp(-8·‖v_xy − v_ref_xy‖²)`
- base linear velocity z: `exp(-8·(v_z − v_ref_z)²)`
- base angular velocity xy (roll and pitch rate): `0.5·exp(-2·‖ω_xy − ω_ref_xy‖²)`
- angular velocity z: `0.5·exp(-2·(ω_z − ω_ref_z)²)`
- joint position: `−15·Σ(q − q_ref)²`
- joint velocity: `−1e-3·Σ(q̇ − q̇_ref)²`
- contact: `Σ(contact == ref_contact)`. The reference contact is thresholded at 0.5, so this is worth 0–2 per step.

  Lines: L104-L135
- Neck, head and antenna joints are removed before comparison (`ref[:5] ++ ref[11:]` vs `q[:5] ++ q[9:]`), so only the 10 leg joints are imitated — [custom_rewards.py L78-L88](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/custom_rewards.py#L78-L88)
- The torso-orientation term is computed but excluded from the sum (`# + torso_orientation_rew`). The torso-position term is commented out. The whole imitation reward is multiplied by `(‖cmd[:3]‖ > 0.01)`, so there is no imitation for zero commands — [custom_rewards.py L104-L148](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/custom_rewards.py#L104-L148)

**History of orientation terms.**
- 2025-03-07, commit 171a7df "trying to add back base orientation in imitation reward" added `exp(-20·‖quat − quat_ref‖²)`. The same day, aa74fd6 "going back" commented it out again — [171a7df](https://github.com/apirrone/Open_Duck_Playground/commit/171a7df); [aa74fd6](https://github.com/apirrone/Open_Duck_Playground/commit/aa74fd6)
- 2025-03-07, commit c921496 "no orientation cost" removed `orientation=-0.5`, which was `Σ(upvector_xy)²` — [c921496](https://github.com/apirrone/Open_Duck_Playground/commit/c921496); [rewards.py L45-L46](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L45-L46)

**Defined but unused helpers in rewards.py.** Several are marked `# FIXME`: cost_lin_vel_z, cost_ang_vel_xy, cost_base_height, `reward_base_y_swing` (tracks a sinusoidal lateral speed `amplitude·sin(2π f t)`), cost_feet_slip, cost_feet_clearance, cost_feet_height, reward_feet_air_time (0.1–0.5 s), reward_feet_phase, cost_pose, cost_joint_deviation_hip/knee. None of these appear in the joystick reward dict — [rewards.py L37-L62, L151-L241](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L37-L62)

**Standing task weights.** orientation=−0.5, torques=−1e-3, action_rate=−0.375, stand_still=−0.3, alive=20, head_pos=−2.0. Imitation is off (`USE_IMITATION_REWARD = False`) — [standing.py L74-L84](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/standing.py#L74-L84)
- Standing's velocity commands are always 0.0 — [standing.py L646-L661](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/standing.py#L646-L661). `cost_head_pos` (Σ(head_q − head_cmd)²) is gated by `(‖cmd[:3]‖ > 0.01)` — [rewards.py L131-L147](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/rewards.py#L131-L147)

**Other reward details.**
- The PPO reward is computed from the pre-delay `action` but uses `info["last_act"]` for the action rate — [joystick.py L440-L455](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L440-L455)
- The README links the imitation reward to Disney's BD-X paper — [PG README L13-L17](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/README.md#L13-L17)

### Inferences
- **Approximate per-step budget (before ×dt).** Alive gives 20. Tracking gives at most 8.5. The positive imitation sub-terms give at most 5 (lin xy 1 + lin z 1 + ang xy 0.5 + ang z 0.5 + contacts 2). The −15·Σe² joint term costs about 1.5 at 0.1 rad error per joint. So the shaping that matters is the joint-angle imitation plus σ = 0.01 tracking. Everything else is weak.
- **No direct roll penalty.** The only roll-related signal is `0.5·exp(-2·(ω_x²+ω_y²))` versus a reference with ω_x ≈ 0 (see Q2). It is weak: a 1 rad/s roll rate loses only about 0.43 of 0.5. Roll is constrained mainly because the leg-angle reference keeps the trunk level (hip_roll swing of about 15° p2p; see Q2).
- **Likely bug: the lin_vel_xy imitation term is heading-dependent.** The sim side uses `base_qvel[:3]`, the MuJoCo free-joint linear velocity, which is in the world frame. The reference is also world-frame velocity, but computed in placo with the robot facing +x ([RMG gait_generator.py L251-L253](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/gait_generator.py#L251-L253)). Reset randomizes yaw over U(−3.14, 3.14) ([joystick.py L223-L231](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L223-L231)). So this term (including the reference lateral-sway velocity) is mostly misaligned except near yaw ≈ 0. The angular-velocity terms are fine: they are body-frame on both sides.
- **Likely bug: standing's head_pos cost never fires.** With velocity commands always 0, the `(‖cmd[:3]‖ > 0.01)` gate is always false, so the −2.0 weight has no effect at HEAD.
- **Compared with our setup.** They do not use gait-clock, air-time, step-length, symmetry, clearance, CoM-over-stance or head-velocity terms. That gait structure is supplied entirely by imitating a ZMP/LIPM reference.

### Gaps
- No ablations or logs show how much each term contributes. Commit messages are the only rationale.
- I found no measured roll or head-sway numbers for their policy, in sim or on hardware.

---

## Q2. How are reference motions generated, stored and interpolated? Do they depend on the commanded velocity?

### Takeaway
References come from Rhoban's placo ZMP/LIPM walk engine. Every reference shares one fixed gait:
- period 0.54 s (single support 0.18 s, double support 0.09 s ×2)
- CoM height 0.205 m, feet spacing 0.16 m, foot height 0.04 m, trunk pitch −4°
- ZMP target 3 cm inward of each foot center (`foot_zmp_target_y = −0.03`). This limits lateral CoM sway.

One clip is generated per (dx, dy, dθ) grid point: 6×4×10 = 240 clips. One period of each clip is fit with degree-15 polynomials (40 dims) and stored in a pickle. At train time the polynomials are evaluated at phase i/27 for the nearest grid point (nearest neighbor, no interpolation between grid points).

So step length and direction depend on the command, but step period does not. Evaluating the stored polynomials, the reference trunk stays level (roll rate ≈ 0) while translating about 38 mm peak-to-peak laterally. Hip roll swings about 14–15° p2p in straight walking.

### Cited Findings
**Generator.** The generator is "based on Placo". It is used both by the MuJoCo Playground work and by an Isaac Gym AWD fork (rimim/AWD) — [RMG README L6-L8](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/README.md#L6-L8). Pipeline: `auto_waddle.py --duck open_duck_mini_v2 --sweep`, then `fit_poly.py`, which writes `polynomial_coefficients.pkl` — [README L37-L50](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/README.md#L37-L50)

**Preset used.** `auto_waddle.py` only uses the "medium" preset (`preset_speeds = ["medium"]`) — [auto_waddle.py L98](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/scripts/auto_waddle.py#L98). v2 medium preset values — [medium.json L7-L51](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json#L7-L51):
- double_support_ratio 0.5, startend_double_support_ratio 1.0
- planned_timesteps 48, replan_timesteps 10
- walk_com_height 0.205, walk_foot_height 0.04, walk_trunk_pitch −4 (deg), walk_foot_rise_ratio 0.02
- single_support_duration 0.18, single_support_timesteps 10
- foot_length 0.06, feet_spacing 0.16, zmp_margin 0.0
- foot_zmp_target_x 0.0, foot_zmp_target_y −0.03
- walk_max_dtheta 1.0, walk_max_dy 0.1, walk_max_dx_forward 0.08, walk_max_dx_backward 0.03
- head joints: neck_pitch 20°, head_pitch −26°

**Preset history.** single_support_duration went from 0.22 to 0.18 and com_height from 0.21 to 0.205 on 2025-02-23 (c15eb08). Trunk pitch went from −3 to −4 on 2025-02-24 (b3586e2). Neck/head angles were added on 2025-03-18 (a9442f2). foot_zmp_target_y has been −0.03 since the first commit — [RMG commit c15eb08](https://github.com/apirrone/Open_Duck_reference_motion_generator/commit/c15eb08); [b3586e2](https://github.com/apirrone/Open_Duck_reference_motion_generator/commit/b3586e2); [a9442f2](https://github.com/apirrone/Open_Duck_reference_motion_generator/commit/a9442f2)

**Parameter semantics (placo).**
- `foot_zmp_target_y` is the ZMP target offset in the foot frame, "positive is 'outward'". It is applied as +y for left support and −y for right support, so −0.03 shifts the ZMP target 3 cm toward the midline — [placo humanoid_parameters.h L137-L145](https://github.com/Rhoban/placo/blob/b54affb692fb1fc80fe688ab50cb1c687d7c3ea6/src/placo/humanoid/humanoid_parameters.h#L137-L145); [walk_pattern_generator.cpp L428-L437](https://github.com/Rhoban/placo/blob/b54affb692fb1fc80fe688ab50cb1c687d7c3ea6/src/placo/humanoid/walk_pattern_generator.cpp#L428-L437)
- Timing: `dt = single_support_duration / single_support_timesteps`; `double_support_timesteps = round(ratio · ss_timesteps)`; `double_support_duration = ds_timesteps · dt` — [humanoid_parameters.cpp L8-L31](https://github.com/Rhoban/placo/blob/b54affb692fb1fc80fe688ab50cb1c687d7c3ea6/src/placo/humanoid/humanoid_parameters.cpp#L8-L31)
- `walk_foot_rise_ratio` is the "ratio of time spent at foot height during the step" — [humanoid_parameters.h L82-L85](https://github.com/Rhoban/placo/blob/b54affb692fb1fc80fe688ab50cb1c687d7c3ea6/src/placo/humanoid/humanoid_parameters.h#L82-L85)

**Walk engine.** `period = 2·single_support_duration + 2·double_support_duration()` — [placo_walk_engine.py L124-L128](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/placo_walk_engine.py#L124-L128). Other settings:
- IK runs at DT 0.01 with REFINE 10, velocity limit 12 rad/s, joint limits disabled; v2 knee limits [0.2, 0.01] — [L11-L53](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/placo_walk_engine.py#L11-L53)
- Foot orientation tasks are masked to the "yz" axes only (L61-L62)
- Head and antenna joints are held by a soft joints task (L71-L73)

**Recording.**
- 50 FPS ("50 for mujoco playground, 30 for AWD") — [gait_generator.py L85](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/gait_generator.py#L85)
- `set_traj(dx, dy, dtheta + 0.00955)`, a small yaw bias — [L143](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/gait_generator.py#L143)
- Base linear velocity is the finite-difference world-frame velocity of the trunk ("fbase"). Angular velocity is the relative-rotation rotvec/dt. Foot contacts come from the placo support phase ([1,1] double, [1,0] left, [0,1] right) — [L251-L263, L289; L230-L238 of placo_walk_engine.py](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/gait_generator.py#L251-L289)
- File naming uses `steps_to_vel(step, period) = 2·step/period` — [L189-L190, L447-L456](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/gait_generator.py#L447-L456)

**Sweep grid.** x step −0.04…0.06 m, y −0.03…0.03 m (granularity 0.02), θ −0.3…0.3 rad (granularity 0.07) — [auto_gait.json L8-L15](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/open_duck_reference_motion_generator/robots/open_duck_mini_v2/auto_gait.json#L8-L15); [auto_waddle.py L100-L133](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/scripts/auto_waddle.py#L100-L133)

**Polynomial fit.** One period is taken after the start double support (`start_offset = startend_ratio·fps`). Y = [16 joint positions, 16 joint velocities, 2 foot contacts, 3 base linear velocities, 3 base angular velocities] = 40 dims. X = linspace(0, 1). A degree-15 `np.polyfit` is fit per dimension — [fit_poly.py L15-L66](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/3d7bc6fd80e6c35587b2a5279314d946cf63456b/scripts/fit_poly.py#L15-L66)
- Runtime lookup: commands are clipped to the grid range, the nearest grid index is taken by `argmin` per axis, and the polynomials are evaluated at `t = (i mod nb_steps)/nb_steps` — [PG poly_reference_motion.py L148-L168](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/poly_reference_motion.py#L148-L168). The 40-dim layout is documented at L6-L51.
- The env calls `get_reference_motion(cmd_x, cmd_y, cmd_yaw, imitation_i)` every step, and `imitation_i` advances by 1 per 0.02 s step — [joystick.py L325-L355](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L325-L355)

**The shipped pkl.** It is the same blob at HEAD and at 097875d; its last change was d4601fc "new pkl", 2025-03-17. I read it by opcode disassembly plus a numpy-only restricted unpickler; none of their code was run. Contents:
- 240 clips
- dx keys (m/s): −0.148, −0.074, 0, 0.074, 0.148, 0.222
- dy keys: −0.111, −0.037, 0.037, 0.111
- dθ keys (rad/s): −1.111 … 1.222 in 10 steps (no 0)
- period 0.54 s, fps 50, nb_steps_in_period 27, start_offset 50, startend_double_support_ratio 1.0

  — [pkl blob](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl); [commit d4601fc](https://github.com/apirrone/Open_Duck_Playground/commit/d4601fc)

**Reference shape (my evaluation of the stored polynomials, not an author statement)** — [pkl blob](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl). For straight walking (dy −0.037, dθ −0.074, dx −0.148…0.222):
- Base lateral velocity swings between about −0.29 and +0.19 m/s. Integrated over a period, the lateral trunk excursion is about **37.6–37.9 mm p2p**, and 38.2 mm for dy = +0.111.
- Base roll and pitch rates are **≈0** (|ω_x| < 0.006 rad/s), meaning a level trunk. Base v_z is −0.09…+0.06 m/s.
- hip_roll is about 14–15° p2p per leg (L −0.10…+0.16 rad, R −0.20…+0.05 rad); 17–20° when turning or for lateral commands.
- hip_yaw is about 1° in straight walking and 21–25° for |dθ| ≈ 1.1–1.2.
- hip_pitch is 21–54° p2p, increasing with dx. knee is 40–52° p2p. ankle is 19–51° p2p.
- Contact pattern: about 37% double support and 63% single support, no flight phase.

**Earlier pkls and speed changes.**
- The first pkl (2025-02-24) and 907ad70 (2025-03-07) used a 5×4×10 grid (dx up to 0.185 m/s), with the same 0.54 s period — [commit 907ad70](https://github.com/apirrone/Open_Duck_Playground/commit/907ad70)
- On 2025-03-09, commit b706069 "faster ref motion and tracking targets" changed lin_vel_x from [−0.1, 0.15] to [−0.15, 0.2] — [b706069](https://github.com/apirrone/Open_Duck_Playground/commit/b706069). HEAD is [−0.15, 0.15] — [joystick.py L94](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L94)
- On hardware, the phase can be sped up: `imitation_i += phase_frequency_factor + offset`, where the factor is 1.0, or 1.3 while LB is held, and the offset is adjusted with the d-pad in ±0.05 steps — [RT v2_rl_walk_mujoco.py L213-L228, L257-L270](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/scripts/v2_rl_walk_mujoco.py#L213-L270). An unused helper would scale the frequency linearly from 1.0 to 1.2 with |vx|/0.15 — [L187-L197](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/scripts/v2_rl_walk_mujoco.py#L187-L197)

### Inferences
- Their reference has **0° trunk roll** and about **±19 mm lateral trunk translation**. A level trunk means the head (about 0.38 m above the feet on the same robot) would sway roughly the same ~4 cm p2p if tracked perfectly. That is less than half of our 8.5–12 cm. Most of our excess sway likely comes from the 24–30° roll range: 0.38 m × sin(±12–15°) ≈ ±8–10 cm, which is not lateral translation. Imitating a level-trunk reference (or adding a hip-roll/leg-angle reference) targets exactly that mode.
- Three placo choices that reduce sway are cheap to copy in our own reference or clock targets:
  - ZMP target 3 cm inward (ZMP at about ±5 cm with feet at ±8 cm)
  - a fairly long double support (33% of the cycle)
  - a short 0.54 s period
- Because there is no dy = 0 key, a zero lateral command maps to the dy = −0.037 m/s reference: jnp.argmin returns the first of two equal distances. Because there is no dθ = 0 key, zero yaw maps to −0.074 rad/s. This is a small built-in bias, inferred from the argmin code plus the key set.
- Commands of ±0.2 m/s lateral and ±1 rad/s yaw are clipped to the reference grid (±0.111 m/s, −1.111…1.222 rad/s).
- Backward and turning references come from the same placo engine with the same period. This is presumably why one policy handles all directions without direction-specific reward hacks.

### Gaps
- placo 0.6.3 sources were not checked. I assume its parameter semantics match the 2026 placo main I read. The pkl numbers match the expected 0.54 s period, which supports that.
- I did not regenerate references, and I computed no lateral CoM (as opposed to trunk-origin) trajectories.

---

## Q3. Action space, head control, action scaling, PD gains and servo model, control frequency, motor and backlash modelling

### Takeaway
The policy outputs **14 joint-position offsets**: 10 leg joints plus neck_pitch, head_pitch, head_yaw and head_roll. Antennas are not included.
- target = home pose + 0.25·action, rate-limited to 5.24 rad/s, at 50 Hz with a 500 Hz physics step.
- Head commands (4 values) are only an observation. In sim nothing makes the head follow them. On hardware, `head_target = head_cmd + policy_head_output`.
- Servos are MuJoCo position actuators with BAM-identified STS3215 parameters (kp 17.11, damping 0.56, armature 0.027, frictionloss 0.068, forcerange ±3.23 N·m).
- Each leg joint has an extra ±0.5° passive backlash hinge.
- The real robot uses firmware Kp 30 on the legs and 8 on the head, with Kd 0.

### Cited Findings
**Actuators and control loop.**
- 14 `<position class="sts3215">` actuators in order: L hip_yaw, hip_roll, hip_pitch, knee, ankle, neck_pitch, head_pitch, head_yaw, head_roll, R hip_yaw … ankle — [open_duck_mini_v2_backlash.xml L497-L512](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L497-L512). `action_size = mjx_model.nu` — [base.py L281-L283](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/base.py#L281-L283)
- `motor_targets = default_actuator + action_w_delay * 0.25`, then clipped to the previous target ± 5.24 rad/s × 0.02 s — [joystick.py L404-L417](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L404-L417)
- Head override is commented out (`# motor_targets.at[5:9].set(state.info["command"][3:])`) — [L419](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L419)
- ctrl_dt 0.02, sim_dt 0.002, action_scale 0.25, max_motor_velocity 5.24 — [L51-L59](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L51-L59)
- Action scale history: 0.5 on 2025-03-11, then 1.0, back to 0.25 on 2025-03-22 — [81d01c1](https://github.com/apirrone/Open_Duck_Playground/commit/81d01c1); [792258e](https://github.com/apirrone/Open_Duck_Playground/commit/792258e); [721317f](https://github.com/apirrone/Open_Duck_Playground/commit/721317f)
- Home pose (also the default ctrl): L 0.002, 0.053, −0.63, 1.368, −0.784; head 0 0 0 0; R −0.003, −0.065, 0.635, 1.379, −0.796. Base z 0.15 — [scene_flat_terrain_backlash.xml L39-L78](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml#L39-L78)

**Servo model (backlash scene, the "Current win" task).** `sts3215` joints: damping 0.56, frictionloss 0.068, armature 0.027; position kp 17.11, kv 0, forcerange ±3.23 — [open_duck_mini_v2_backlash.xml L45-L51](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L45-L51)
- The non-backlash MJCF uses kp 13.37 (17.12 commented out) — [open_duck_mini_v2.xml L45-L49](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml#L45-L49)
- The older `joints_properties.xml` has damping 0.60, frictionloss 0.052, armature 0.028, kp 17.8, forcerange ±3.35 — [joints_properties.xml L2-L6](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/joints_properties.xml#L2-L6)
- Solver: `iterations=1, ls_iterations=5`, `eulerdamp="disable"` — [backlash.xml L6-L8](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L6-L8)

**Backlash model.** A `backlash` class hinge with "+/- 0.5° of backlash": range ±0.00873 rad, damping 0.01, armature 0.01, frictionloss 0. It is added after each leg joint (e.g. `left_hip_yaw_backlash`); head joints have none — [backlash.xml L52-L56, L118-L119](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L52-L56). Observed joint angle = actuator joint + backlash joint — [joystick.py L534-L541](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L534-L541)

**Parameter provenance.**
- Values come from Rhoban BAM identification ("feetech_sts3215_7_4V"), exported with `bam.to_mujoco`: damping, kp, frictionloss, armature, forcerange — [ODM docs/sim2real.md L19-L31](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/docs/sim2real.md#L19-L31)
- 5.24 rad/s "comes from system identification" (BAM m1.json) — [Runtime issue #7](https://github.com/apirrone/Open_Duck_Mini_Runtime/issues/7)
- "17.8 in mujoco corresponds to 32 in the motors firmware units" — [Open_Duck_Mini issue #20](https://github.com/apirrone/Open_Duck_Mini/issues/20). This conflicts with an older MJCF comment, removed on 2025-03-07: "6.55 is equivalent to 32 firmware kp, 4.0 is equivalent to 20 firmware kp" — [171a7df](https://github.com/apirrone/Open_Duck_Playground/commit/171a7df)

**Hardware runtime** — [RT v2_rl_walk_mujoco.py](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/scripts/v2_rl_walk_mujoco.py):
- control_freq 50 Hz, pid [30, 0, 0], action_scale 0.25, num_dofs 14 (L30-L49)
- `kps[5:9] = [8, 8, 8, 8]` lowers the head Kp (L174-L183)
- `motor_targets = init_pos + action·action_scale`; the velocity clip is commented out on hardware (L290-L298)
- optional low-pass filter with `--cutoff_frequency` (default None) (L300-L306, L381)
- `head_motor_targets = last_commands[3:] + motor_targets[5:9]` (L310-L311)
- CLI default `-p 30` (L356)
- The HWI default is Kp 32 and Kd 0, with a low-torque Kp of 2 during startup — [rustypot_position_hwi.py L73-L101](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py#L73-L101)

**Head-control commit history.**
- "commands always include head, but not used for walking policy for now" (2025-03-07) — [c6b4263](https://github.com/apirrone/Open_Duck_Playground/commit/c6b4263)
- "no head command for now" removed head_pos=−0.5 (2025-03-17) — [a312a28](https://github.com/apirrone/Open_Duck_Playground/commit/a312a28)
- "trying head control again" (2025-03-22) added `motor_targets.at([5,6,7,8]).set(cmd[3:])` — [4f009fa](https://github.com/apirrone/Open_Duck_Playground/commit/4f009fa)
- "CHECKPOINT pretty nice. turns in place, catches itself, somewhat robust to independant head moves" (2025-03-23) — [c0c2b3e](https://github.com/apirrone/Open_Duck_Playground/commit/c0c2b3e)

### Inferences
- **Main difference from us.** Their policy also drives the four head joints. The head_roll body alone is about 0.41 kg ([backlash.xml L272-L273](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L272-L273)). It is only regularized by action-rate, torque and zero-command stand-still costs. So the head is free to act as a balance or counter-sway appendage during walking. This is a design point to test: give our policy the head joints, or at least head_roll. We could also add our head-velocity penalty explicitly; they have none.
- **The 2025-03-22 head override was a no-op.** `x.at[idx].set()` returns a new array and the result was not assigned. So in every committed version, sim training never moved the head per command. Hardware "robustness to head moves" comes from the policy tolerating offsets added open-loop at runtime.

### Gaps
- No documentation says whether the policy actually uses the head for balance; I saw no plots.
- I did not fetch the BAM parameter file itself (m1.json) to cross-check kp/damping.

---

## Q4. Observations, phase, history, domain randomization, terrain, episode length, PPO hyperparameters, training steps

### Takeaway
**Actor observation** (101 dims, no projected gravity, no base velocity): raw gyro, raw accelerometer, 7-dim command, 14 joint positions (relative to home), 14 joint velocities ×0.05, the last 3 actions, the last motor targets, 2 foot contacts, and the phase as (cos, sin) over 27 steps. The critic gets a privileged state.

**Noise, delay and pushes.** Uniform noise, action delay of 0–2 steps, and pushes of 0.1–1.0 m/s every 5–10 s.

**Domain randomization.**
- floor friction 0.5–1.0
- frictionloss ×0.9–1.1, armature ×1.0–1.05
- masses ×0.9–1.1, torso mass ±0.1 kg, torso CoM ±5 cm
- joint zero offsets ±0.03 rad, kp ×0.9–1.1

**Training setup.** Flat terrain with backlash. Episodes are 1000 steps (20 s) and commands are resampled every 10 s. PPO uses the mujoco_playground BerkeleyHumanoid config: 8192 envs, 512-256-128 networks, γ 0.97, lr 3e-4, entropy 0.005. Default training is 150M steps; the "Current win" command uses 300M.

### Cited Findings
**Actor observation `state`** — [joystick.py L570-L589](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L570-L589). In order:
- noisy_gyro (3), noisy_accelerometer (3), command (7: vx, vy, ωz, neck_pitch, head_pitch, head_yaw, head_roll)
- noisy joint angles − default (14), noisy joint velocity × 0.05 (14)
- last_act, last_last_act, last_last_last_act (14 each), motor_targets (14, absolute)
- contact (2), imitation_phase (2)

The reference motion itself is not in the observation ("trying without ref in obs", 2025-03-24; [309623e](https://github.com/apirrone/Open_Duck_Playground/commit/309623e)).

**Hardware observation.** Same order: gyro, accelero, cmds, dof_pos − init_pos, dof_vel × 0.05, the 3 last actions, motor_targets, feet_contacts (foot switches), imitation_phase — [RT v2_rl_walk_mujoco.py L156-L169](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/scripts/v2_rl_walk_mujoco.py#L156-L169). The IMU is a BNO055 in NDOF mode — [raw_imu.py L22-L27](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/376de65435c93486347cd601a31aa96951799896/mini_bdx_runtime/mini_bdx_runtime/raw_imu.py#L22-L27)

**IMU and phase details.**
- "use raw gyro and accelero instead of projected gravity" (2025-03-06) — [75ae5f0](https://github.com/apirrone/Open_Duck_Playground/commit/75ae5f0)
- Phase = [cos(2π·i/27), sin(2π·i/27)] — [joystick.py L325-L343](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L325-L343)

**Privileged (critic) state.** state + clean gyro, accelerometer, gravity, local linear velocity, global angular velocity, joint positions and velocities, root height, actuator forces, contacts, foot velocities, feet air time, the current reference frame, imitation_i and phase — [L596-L615](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L596-L615)

**Noise.** Uniform (2U−1)·scale with level 1.0: hip 0.03 rad, knee 0.05, ankle 0.08, joint velocity 2.5 rad/s ("was 1.5"), gyro 0.1, accelerometer 0.05 (gravity 0.1 and linvel 0.1 are unused in the actor). action_delay 0–3 and imu_delay 0–3 — [L60-L76](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L60-L76)
- Action delay = `randint(min=0, max=3)` steps into the action history — [L361-L376](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L361-L376)
- The IMU delay buffer only stores `gravity`, which is not in the actor observation — [L521-L530](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L521-L530)
- `accelerometer.at[0].set(accelerometer[0] + 1.3)` is a no-op in JAX; a community PR points out the mismatch with numpy inference — [L500-L502](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L500-L502); [Playground PR #24](https://github.com/apirrone/Open_Duck_Playground/pull/24)

**Pushes.** Enabled; interval U(5, 10) s; magnitude U(0.1, 1.0) m/s added to base xy velocity in a random direction — [L89-L93, L381-L400](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L381-L400)

**Resets** — [L212-L254](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L212-L254):
- xy position ±0.05 m, yaw U(−3.14, 3.14)
- joint positions multiplied by U(0.5, 1.5)
- base velocity U(−0.05, 0.05); this was ±0.5 before "less qvel reset" on 2025-03-24 ([5943d9e](https://github.com/apirrone/Open_Duck_Playground/commit/5943d9e))

**Commands** — [L94-L101, L671-L725](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L671-L725):
- vx U(−0.15, 0.15), vy U(−0.2, 0.2), ωz U(−1, 1)
- neck_pitch U(−0.34, 1.1), head_pitch U(−0.78, 0.78), head_yaw U(−1.5, 1.5), head_roll U(−0.5, 0.5)
- 10% chance of all zeros
- resampled when step > 500, i.e. every 10 s ([L456-L466](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L456-L466))

**Episode and termination.** episode_length 1000 ([L53](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L53)). Termination only if the upvector z < 0 (trunk tilted more than 90°) or NaN ([L483-L485](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L483-L485))

**Domain randomization** — [randomize.py L26-L95](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/randomize.py#L26-L95):
- geom friction of `FLOOR_GEOM_ID = 0` ~U(0.5, 1.0)
- frictionloss ×U(0.9, 1.1); armature ×U(1.0, 1.05)
- body_ipos of `TORSO_BODY_ID = 1` +U(−0.05, 0.05) m
- all body masses ×U(0.9, 1.1); body 1 mass +U(−0.1, 0.1) kg
- qpos0 +U(−0.03, 0.03) rad ("was −0.05 0.05")
- actuator kp (gainprm and biasprm) ×U(0.9, 1.1) ("was 0.8, 1.2")

The randomizer is passed to Brax PPO as `randomization_fn` — [common/runner.py L104-L112](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/runner.py#L104-L112)

**Terrain.**
- Flat scene: plane, friction 0.6, condim 3 — [scene_flat_terrain_backlash.xml L33-L38](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml#L33-L38)
- A rough hfield scene also exists (size 10 10 .01 0.1, friction 1.0) — [scene_rough_terrain_backlash.xml L22-L28](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/scene_rough_terrain_backlash.xml#L22-L28)
- The README's "Current win" is `--task flat_terrain_backlash --num_timesteps 300000000` — [PG README L94-L98](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/README.md#L94-L98). The runner defaults are num_timesteps 150M, env joystick, task flat_terrain — [runner.py L44-L46](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/runner.py#L44-L46)

**PPO.** `locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`, with num_timesteps overridden by the CLI — [common/runner.py L86-L101](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/runner.py#L86-L101). At mujoco_playground v0.0.3 (pyproject requires `playground>=0.0.3`, [pyproject.toml L15](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/pyproject.toml#L15)) — [locomotion_params.py v0.0.3 L27-L44, L106-L119](https://github.com/google-deepmind/mujoco_playground/blob/v0.0.3/mujoco_playground/config/locomotion_params.py#L106-L119):
- normalize_observations True, reward_scaling 1.0, action_repeat 1
- unroll_length 20, num_minibatches 32, num_updates_per_batch 4
- discounting 0.97, learning_rate 3e-4, max_grad_norm 1.0
- num_envs 8192, batch_size 256
- clipping_epsilon 0.2, entropy_cost 0.005
- num_evals 15, num_resets_per_eval 1 (10 in the 2026 main)
- policy (512, 256, 128) on "state", value (512, 256, 128) on "privileged_state"

### Inferences
- **Likely bug: `FLOOR_GEOM_ID = 0` and `TORSO_BODY_ID = 1` are hard-coded.** The scene includes the robot before the floor ([scene_flat_terrain_backlash.xml L2, L33-L37](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml#L2)). So geom 0 is probably the first robot (visual) geom, not the floor, and floor-friction DR probably does nothing. Body 1 is probably the `base` frame body, which has no inertial; the 0.698 kg trunk inertial is on its child `trunk_assembly` ([backlash.xml L59-L66](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml#L59-L66)). So the "torso CoM ±5 cm / mass ±0.1 kg" DR may act on a massless body. This is inferred from MJCF ordering and was not verified by loading the model.
- **Likely bug: joint-position noise indices are misaligned.** `qpos_noise_scale` has 14 entries, but the hip/knee/ankle indices are taken from the 10-entry `JOINTS_ORDER_NO_HEAD` list ([joystick.py L183-L200](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L183-L200)). As a result, head joints and right_hip_yaw get noise, while the right hip_roll, hip_pitch, knee and ankle get none.
- **The "IMU delay" never reaches the actor.** Only `gravity` is delayed, and gravity is not observed. Actions get 0–40 ms of delay.
- Even with these bugs, their recipe transferred to hardware ("pretty nice walk on the real robot"). The robustness likely comes mostly from the action delay, joint-velocity noise of 2.5 rad/s, kp/frictionloss DR, backlash, the 5.24 rad/s target rate limit, large pushes, and imitation.

### Gaps
- I could not confirm which mujoco_playground version the authors actually installed (only `>=0.0.3`); hyperparameters are those at v0.0.3–v0.0.5.
- No training curves or wall-clock times were found.

---

## Q5. What the team says about lateral sway, waddling, head motion, foot clearance and sim2real

### Takeaway
I found **no explicit discussion of lateral sway, waddle amplitude or head motion** in the READMEs, docs, issues or commit messages. Sim2real effort went into three areas:
- accurate actuator modelling: BAM identification, backlash, motor speed limits, slicer-accurate masses
- noise and delay
- imitation of placo references, plus small runtime knobs: phase speed-up, head Kp 8, an optional low-pass filter

Foot clearance appears only as user-reported hardware problems ("can't lift its feet", weak right-leg clearance with 7.4 V knees), with no published fix. The older pipeline (AWD/AMP on Isaac Gym, v1 duck) used a pure adversarial-imitation reward.

### Cited Findings
**sim2real doc.** It is marked "Not finalized yet". It says the cheap servos are "hard to model and not overly powerful". It stresses accurate motor modelling with BAM (damping, kp, frictionloss, armature, forcerange) and slicer-estimated masses. It says "We obtained good results by implementing the imitation reward described by Disney in their BDX paper" — [ODM docs/sim2real.md L1-L41](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/docs/sim2real.md#L1-L41)

**README timeline.** "07/02/2025: Big progress on sim2real" and "24/02/2025: Working hard on sim2real". Two released policies, BEST_WALK_ONNX(_2).onnx, run via `v2_rl_walk_mujoco.py` — [ODM README L26-L36](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/README.md#L26-L36)

**Commit milestones** on the path to the real-robot checkpoint:
- "motor velocity limit test", 2025-03-20 — [d8d15f6](https://github.com/apirrone/Open_Duck_Playground/commit/d8d15f6)
- "clipping actions" and "adding motor targets to obs", 2025-03-22 — [380c1ee](https://github.com/apirrone/Open_Duck_Playground/commit/380c1ee)
- "backlash in obs", 2025-03-06 — [cf45ff4](https://github.com/apirrone/Open_Duck_Playground/commit/cf45ff4)
- "more noisy imu", 2025-03-07 — [e5649d2](https://github.com/apirrone/Open_Duck_Playground/commit/e5649d2)
- "-6 trunk pitch in ref motion", 2025-03-07 — [907ad70](https://github.com/apirrone/Open_Duck_Playground/commit/907ad70)
- "less stand still cost" (−0.5 → −0.2), 2025-03-24 — [fd53d20](https://github.com/apirrone/Open_Duck_Playground/commit/fd53d20)
- "CHECKPOINT pretty nice walk on the real robot", 2025-03-24 — [097875d](https://github.com/apirrone/Open_Duck_Playground/commit/097875d)

**Foot-clearance reports.**
- Playground issue #5: "It looks like it can't lift its feet...". The maintainer only redirected to Discord — [Open_Duck_Playground #5](https://github.com/apirrone/Open_Duck_Playground/issues/5)
- Open_Duck_Mini issue #52: swapping the knee servos for 12 V 30 kg STS3215s "stopped the duck from falling over", but "the right leg isn't generating enough foot clearance". No answer — [Open_Duck_Mini #52](https://github.com/apirrone/Open_Duck_Mini/issues/52)
- The TODO list includes "12V motors? … retrain walk policy for these motors" — [Open_Duck_Mini #33](https://github.com/apirrone/Open_Duck_Mini/issues/33)

**Community smoothing PR.** It adds `action_acceleration` (e.g. action_rate −0.8 / action_acceleration −0.4 for "slower, more graceful motion"). It was closed; merge status not checked — [Playground PR #21](https://github.com/apirrone/Open_Duck_Playground/pull/21)

**Older pipeline (apirrone/AWD, Isaac Gym, "A mallard version of ASE", v1 "mini_bdx").** README — [AWD README](https://github.com/apirrone/AWD/blob/7ef60157171a41a5db1d9e37536f0bf5b0b0568e/README.md)
- Task reward is constant (`reward = torch.ones_like(...)`) — [duckling.py L745-L748](https://github.com/apirrone/AWD/blob/7ef60157171a41a5db1d9e37536f0bf5b0b0568e/awd/env/tasks/duckling.py#L745-L748)
- Training weights task_reward_w 0.0, disc_reward_w 0.5, enc_reward_w 0.5; lr 2e-5; horizon 32; MLP [1024, 1024, 512]; fixed σ (init −2.9) — [mini_bdx/train/awd_duckling.yaml L10-L114](https://github.com/apirrone/AWD/blob/7ef60157171a41a5db1d9e37536f0bf5b0b0568e/awd/data/cfg/mini_bdx/train/awd_duckling.yaml#L10-L114)
- Env: 4096 envs, episodeLength 300, "controlFrequencyInv: 2 # 30 Hz", numAMPObsSteps 10, termination height 0.15 — [mini_bdx/duckling_awd.yaml L2-L20](https://github.com/apirrone/AWD/blob/7ef60157171a41a5db1d9e37536f0bf5b0b0568e/awd/data/cfg/mini_bdx/duckling_awd.yaml#L2-L20)
- The Open_Duck_Mini README says "We are switching to Mujoco Playground" — [ODM README L54-L56](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/README.md#L54-L56)

### Inferences
- The team's evidence is qualitative ("pretty nice walk"). There is no quantitative sway or roll target to compare against; our 8.5–12 cm head-sway metric has no counterpart in their repos.
- Their reference (Q2) is the closest thing to a sway specification: level trunk, about 38 mm p2p lateral trunk translation, a 0.54 s cycle, and 33% double support.
- **What is worth copying for our roll problem**, in order of how much it changes:
  - (a) imitating per-joint leg reference angles with a level trunk, especially hip_roll at about 15° p2p, as a −15·Σe² penalty
  - (b) a ZMP-inward, high double-support placo gait as that reference
  - (c) letting the policy also actuate head joints
  - (d) the lateral dead-band in velocity tracking (0.1 m/s), so the velocity reward does not fight the natural lateral oscillation

### Gaps
- The Discord (the main support channel) is not mirrored on GitHub, and I found no blog or X posts with tuning notes. A web search surfaced only aggregator pages.
- I found no statements about how much the duck waddles, rolls or sways its head on hardware.

---

## Q6. Licenses

### Takeaway
Only Open_Duck_Mini has a repo-level license (Apache-2.0). Open_Duck_Playground, the reference motion generator and the Runtime have **no LICENSE file**; GitHub reports `license: null` for all three, so treat them as **reference-only**. Some Playground files carry Apache-2.0 headers because they were derived from DeepMind's mujoco_playground; those specific files can be argued to be Apache-2.0. The reward, reference-motion and runner files have no header. AWD is under the NVIDIA license (non-commercial). placo is MIT.

### Cited Findings
- Open_Duck_Mini: Apache License 2.0, per its LICENSE file and the GitHub API — [ODM LICENSE](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/LICENSE)
- Open_Duck_Playground, Open_Duck_reference_motion_generator and Open_Duck_Mini_Runtime: no LICENSE file in the tree; `api.github.com/repos/apirrone/<repo>` returns `"license": null` — [Playground repo](https://github.com/apirrone/Open_Duck_Playground); [generator repo](https://github.com/apirrone/Open_Duck_reference_motion_generator); [Runtime repo](https://github.com/apirrone/Open_Duck_Mini_Runtime)
- Playground files with "Copyright 2025 DeepMind Technologies Limited / Copyright 2025 Antoine Pirrone - Steve Nguyen … Licensed under the Apache License, Version 2.0" headers: joystick.py, standing.py, base.py, constants.py, randomize.py — [joystick.py L1-L15](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/joystick.py#L1-L15); [randomize.py L1-L15](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/common/randomize.py#L1-L15)
- Files with no header: custom_rewards.py, common/rewards.py, poly_reference_motion.py, runner.py, export_onnx.py, mujoco_infer*.py — [custom_rewards.py](https://github.com/apirrone/Open_Duck_Playground/blob/b9be205ac64488c23504ca42e5ec790337adeec3/playground/open_duck_mini_v2/custom_rewards.py)
- apirrone/AWD: "NVIDIA License". Use is limited to "non-commercially", defined as "for research or evaluation purposes only" — [AWD LICENSE.txt L1, L53-L56](https://github.com/apirrone/AWD/blob/7ef60157171a41a5db1d9e37536f0bf5b0b0568e/LICENSE.txt#L53-L56)
- placo: MIT-style ("Copyright (c) <2023-2099> Rhoban Team, Permission is hereby granted, free of charge…") — [placo LICENSE](https://github.com/Rhoban/placo/blob/b54affb692fb1fc80fe688ab50cb1c687d7c3ea6/LICENSE)

### Inferences
- **Safe path.** Re-implement the ideas (reward forms, gait parameters, DR ranges) in our own code rather than copying the no-license files verbatim. Generating our own references with placo (MIT) and our own parameters avoids any dependency on their pkl.

### Gaps
- No legal review. Whether the per-file Apache headers grant rights over the whole Playground repo is unclear.
