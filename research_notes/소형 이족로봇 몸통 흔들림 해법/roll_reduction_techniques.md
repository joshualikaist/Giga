# Reducing torso roll and head sway in a biped without ankle roll (Open Duck Mini v2): mechanisms, formulas, expected effects

Scope: lateral balance without ankle roll (hip strategy), how other small/no-ankle-roll bipeds walk, RL techniques, head/gaze stabilization with neck joints, and reward-design pitfalls.
Robot context (from the assignment, not re-verified): ~2 kg, leg ~0.18 m, CoM ~0.22 m, head 0.38 m above the feet and ~26 % of the mass, STS3215 servos (~4.4 rad/s no-load), gait period 0.6 s (0.3 s per step), step width ~18 cm, CoM lateral shift ~1.4 cm, torso roll range 24–30°/step, head lateral sway 8.5–12 cm.
Everything under "Inferences" that has a number is my own estimate (marked "est."), worked out with the formulas cited in "Cited Findings".

## 1. Lateral balance without ankle roll: how step width, cadence, double support, swing-foot clearance and hip hiking trade off against torso roll (formulas)

### Takeaway
In the LIP/H-LIP model, lateral CoM sway comes passively from the stance-foot offset and the step timing. It needs **no torso roll**, and it scales roughly with step width × (step time)². For this robot the H-LIP sway at 0.3 s steps and 18 cm width is about ±3 cm (est.), and the measured CoM shift is smaller than that. So the 24–30° roll is most likely doing something else: (a) hip hiking for swing-foot clearance (in humans, a stiff knee leads to more pelvic obliquity), and/or (b) a "flywheel" hip strategy that holds the CoM in instead of letting it sway. Humans walk with only about 3° of trunk roll range in the world frame.

### Cited Findings
**Reduced-order models (H-LIP / LIP / capture point)**
- H-LIP (Xiong & Ames): λ = √(g/z₀). Single support follows p̈ = λ²p (solution p = c₁e^{λt} + c₂e^{−λt}). Double support is modelled as constant velocity (p̈ = 0). Step-to-step dynamics are x_{k+1} = A x_k + B u_k with A = e^{A_SSP·T_SSP}[1 T_DSP; 0 1] and B = e^{A_SSP·T_SSP}[−1; 0]. Lateral walking uses a **period-2 orbit** with orbital slope σ₂ = λ·tanh(T_SSP·λ/2), states on v = ±σ₂p + d₂, and step widths u_{L/R} = 2p⁻ + T_DSP·v⁻. Stepping stabilization is u = u* + K(x − x*), with deadbeat gain K = [1, T_DSP + (1/λ)coth(T_SSP·λ)], which converges in two steps — [Xiong & Ames, H-LIP (ar5iv)](https://ar5iv.labs.arxiv.org/html/2101.09588)
- LIP plus flywheel: adding a flywheel (body rotational inertia) to the LIP lets the robot accelerate the CoM by changing centroidal angular momentum. The single instantaneous capture point then becomes a capture *region*, which is the formal version of the "hip/upper-body strategy" — [Pratt et al. (IHMC) patent US7949430, as summarized in search results](https://patents.google.com/patent/US7949430)
- Kuo 1999 (passive 3-D walker with roll): "the roll motion is unstable, resembling that of an inverted pendulum. The instability is due to mismatching of roll velocity with the ground contact conditions." Of several strategies, "quasi-static control of step width" was found "both simple and efficient". The model "rolls on curved feet and can rock side to side" — [Kuo, IJRR 1999](https://www.ias.informatik.tu-darmstadt.de/uploads/Research/LocomotionSeminar/KuoIJRR99.pdf)
- Minimum stride frequency for frontal-plane stability (SLIP-type frontal models): ω_s,min ≈ ω_n + ω_p, with ω_n = √(k/m) (leg stiffness) and ω_p = √(g/l). A minimum hip width w_min ≈ 4√(l_eq)/ω_n is needed. Narrower hips raise the minimum stride frequency. PD joint feedback cut the minimum frequency by ~85 % in the fixed-ankle model, but by <1 % in the fixed-hip model and ~15 % in the extended model — [Karunanayaka & Rezazadeh, arXiv 2510.22030](https://arxiv.org/html/2510.22030)

**Human biomechanics (step width, speed, trunk)**
- External lateral stabilization (springs at the waist) made subjects choose a **47 % narrower** step width, with 32 % less step-width variability and 5.7 % lower metabolic cost. At a prescribed zero step width it gave 33 % less variability and 9.2 % lower cost. The authors conclude "humans avoid narrow step widths because they are less stable". The step-to-step transition work "increases with the square of step width" (citing Donelan et al. 2001) — [Donelan, Shipman, Kram, Kuo, J Biomech 2004](https://www.sfu.ca/locomotionlab/assets/latstab2004.pdf)
- Narrowing step width (target 50 % of preferred; actual 60 %) significantly **reduced ML-CoM displacement and velocity**, which the authors explain as a "direct mechanical effect" of the narrower width. It also reduced the ML margin of stability — [Arvin et al., J Biomech 2016](https://research.vu.nl/ws/files/42590600/Arvin_JoB_2016.pdf)
- At very slow speeds, trunk-CoM and body-CoM lateral excursions grow while step width stays unchanged. Ankle eversion torque and hip abduction torque both rise (~7.09 N·m per m/s). The paper notes that "rotation of the trunk about the body CoM impacts the acceleration of the CoM", i.e. a trunk strategy is available when stepping is constrained — [Best & Wu, bioRxiv 2020](https://www.biorxiv.org/content/10.1101/2020.07.16.207092.full.pdf)
- ML CoM excursions increase as gait speed decreases (Orendurff et al., JRRD 2004, cited secondarily; the original was not read) — [Frontiers Sports Act. Living 2023 (secondary citation, search snippet only)](https://www.frontiersin.org/articles/10.3389/fspor.2023.1304141/full)
- Normal adults (n = 20) walking: trunk **obliquity (roll) range is 3.3° (SD 1.4) in the global frame** versus 13.0° (SD 4.5) relative to the pelvis. Global-frame obliquity was the smallest of all trunk ranges — [Chung et al., JNER 2010](https://jneuroengrehab.biomedcentral.com/track/pdf/10.1186/1743-0003-7-9)
- Swing clearance compensation: restricting knee flexion "significantly increased pelvic obliquity (2.79, p<0.01) compared to unrestricted walking (1.5)". The extracted text lost the units; they are probably degrees. The authors conclude "pelvic obliquity, not hip abduction, compensates for reduced knee flexion angle", and describe hip hiking as a more energy-efficient clearance strategy — [Akbas et al., bioRxiv 2019](https://www.biorxiv.org/content/10.1101/520684.full.pdf)

**Walk-engine practice (RoboCup)**
- Bit-Bots quintic walk: typical `double_support_ratio` is 0.05–0.3 (it must be non-zero). Typical `freq` is 0.5–1.5 steps/s. For `foot_rise`, "for hard floors, just a centimeter should be enough; for artificial grass its around 5 cm". `trunk_swing` sets the lateral trunk movement (1 = as far as footDistance; typical 0.1–1) — [Bit-Bots quintic walk docs](https://docs.bit-bots.de/package/bitbots_quintic_walk/latest/)

### Inferences
- **Closed form for the lateral sway (est., from the H-LIP equations above).** Symmetric period-2 lateral orbit, stance foot at ±w/2, CoP fixed at foot centre, T_SS single support, T_DS double support at constant velocity:
  - closest approach to the stance foot: p_m = (w/2) / [cosh(λT_SS/2) + (λT_DS/2)·sinh(λT_SS/2)]
  - lateral CoM amplitude about the midline: A = w/2 − p_m
  - peak lateral CoM speed: v = p_m·λ·sinh(λT_SS/2)
  - For small λT this is A ≈ (w/2)(λT_SS)²/8 + …, so **A scales with w·T²**.
  - With λ = √(9.81/0.22) = 6.68 rad/s:

  | step width w | step time T | DS ratio | A (±cm) | peak v_y (m/s) |
  |---|---|---|---|---|
  | 0.18 | 0.30 s | 0 / 0.2 | 3.2 / 3.1 | 0.46 / 0.35 |
  | 0.18 | 0.25 s | 0.2 | 2.3 | 0.32 |
  | 0.18 | 0.20 s | 0.2 | 1.6 | 0.28 |
  | 0.14 | 0.30 s | 0.2 | 2.4 | 0.27 |
  | 0.14 | 0.20 s | 0.2 | 1.3 | 0.21 |
  | 0.10 | 0.30 s | 0.2 | 1.7 | 0.20 |
  | 0.10 | 0.20 s | 0.2 | 0.9 | 0.15 |

  The Open Duck reference generator default (single support 0.17 s, `double_support_ratio` 0.18, feet 0.16 m; see Q2) gives T ≈ 0.20 s and A ≈ ±1.5 cm (est.).
- **The measured CoM sway (~1.4 cm) is smaller than what H-LIP predicts for the current gait (~±3.1 cm at 0.3 s and 18 cm).** If "1.4 cm" means amplitude, the robot keeps its CoM closer to the midline than a passive LIP would. Two likely explanations: (i) the effective CoP sits at the inner foot edge, so the effective width is less than 18 cm; (ii) the large torso roll works as a **flywheel (centroidal angular momentum about x)** that supplies part of the lateral force reversal. If (ii) is true, simply forbidding roll will force either more CoM sway (accept ~±3 cm), narrower steps, or faster steps. This is a hypothesis; a cheap test is to log centroidal angular momentum L_x versus the lateral CoP in simulation.
- **Hip hiking (est.).** Rolling the pelvis by φ about the stance hip raises the swing hip by s·sin φ (s = hip-joint spacing). With s ≈ 0.08–0.12 m and φ ≈ 12–15° (half of the measured 24–30° range), that is ~1.7–3.1 cm of free clearance. It is the same compensation humans use when knee flexion is limited.
- **Knee-based clearance runs into the servo speed limit (est.).** Assume a 0.09 + 0.09 m two-link leg at 0.16 m nominal length, with lift done in ~0.12 s (half of a 0.24 s swing):

  | clearance | extra knee flexion | avg knee speed | peak (sinusoidal profile) |
  |---|---|---|---|
  | 1 cm | 12.6° | 1.8 rad/s | 2.9 rad/s |
  | 1.5 cm | 18° | 2.6 rad/s | 4.1 rad/s |
  | 2 cm | 23° | 3.4 rad/s | 5.3 rad/s |
  | 3 cm | 33° | 4.8 rad/s | 7.5 rad/s |

  The STS3215 no-load speed is ~4.4 rad/s and lower under load. So knee-only clearance above ~1.5 cm at the current cadence is borderline, and the policy may be choosing roll-based hiking because it is "cheaper". Faster cadence would make this worse. Lower clearance (1 cm is enough on hard floors per Bit-Bots) makes a knee-only lift feasible.
- **Roll and step width are coupled through the geometry.** With no ankle roll the shank's frontal angle equals the foot's roll, so a flat stance foot forces the stance leg to be vertical in the frontal plane. Then the pelvis position over the foot can only change by rotating the trunk about the hip roll axis (torso roll) or by tipping the foot onto an edge. Rounded soles (Disney, Q2) and Kuo's curved feet make the "tip the foot" option smooth and passive.
- **Trade-off summary (est.).** Each lever, and its cost:
  - Narrower steps: sway scales with w, and hip-strategy demand falls; costs some margin of stability.
  - Shorter step time: sway scales with T²; but foot-lift speed demand rises roughly as 1/T.
  - More double support: slightly less sway and a smoother velocity reversal; leaves less swing time.
  - Lower swing clearance: removes the hip-hiking incentive.

### Gaps
- No source gave a closed-form "required torso roll vs. step frequency/width" for a hip-strategy biped. The table above is my own derivation from H-LIP and ignores the flywheel term.
- The original Orendurff 2004 numbers (ML CoM excursion vs speed) and Pratt 2006 capture-region equations were not read directly; only secondary summaries were found.
- The Akbas 2019 pelvic-obliquity values lost their units in PDF extraction (presumably degrees).

## 2. How small humanoids/bipeds without ankle roll walk; trunk sway used by walk engines; how they lift the swing foot

### Takeaway
Bipeds without an actuated ankle roll avoid large trunk roll in one of three ways:
- **Passive roll at the foot**: Disney's BD-X-style droid has rounded soles; BRUCE has line feet.
- **Explicitly regulated torso roll**: Cassie servos world-frame torso roll through the hips and optimizes gaits with a weight of 10,000 on torso roll.
- **Lateral balance by foot placement plus trunk translation, not roll**: the Rhoban IKWalk default has swingRollGain = 0 and a 2 cm lateral trunk swing.

The original Open Duck pipeline imitates a placo LIPM/ZMP reference whose trunk orientation is tracked, uses short steps (~0.2 s), and lifts the foot with sagittal joints.

### Cited Findings
**Disney bipedal character (BD-X-like)**
- 5 DoF per leg (hip abduction/adduction, hip flexion/extension, hip rotation, knee, ankle pitch) and a 4-DoF neck/head. It is "not equipped with ankle roll actuators. To allow for passive ankle roll, we rounded off the two foot soles". The soles are urethane-foam molded to damp impacts — [Grandia et al., Disney Research 2025, arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Reference gaits: the artist provides task-space gait samples at several speeds that are "procedurally combined". An MPC plans CoM/CoP and an inverse dynamics controller produces whole-body trajectories. "The horizontal torso trajectory ... will generally sway around the path frame to shift weight between the feet." Head commands are offsets (Δh_head, Δθ_head) "expressed in path frame coordinates" relative to a nominal head motion in the walk cycle. Performance: 0.7 m/s forward, 0.4 m/s lateral, 1.8 rad/s turning. Walking joint MAE is 0.123 rad — [Grandia et al., arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)

**Open Duck Mini v2 reference generator**
- `placo_defaults.json` values:
  - `double_support_ratio` 0.18, `single_support_duration` 0.17 s, `single_support_timesteps` 10
  - `walk_com_height` 0.2 m, `walk_foot_height` 0.04 m, `walk_foot_rise_ratio` 0.2, `walk_trunk_pitch` −3°
  - `feet_spacing` 0.16 m, `foot_length` 0.06 m
  - `walk_max_dx_forward` 0.08, `walk_max_dx_backward` 0.03, `walk_max_dy` 0.1, `walk_max_dtheta` 1.0
  - Head joints are fixed at 0 — [Open_Duck_reference_motion_generator, placo_defaults.json (v2)](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/main/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_defaults.json)
- The generator uses `placo.WalkTasks` (CoM/foot/trunk tasks) and masks the foot orientation tasks to the local "yz" axes, so foot roll is left unconstrained because there is no ankle roll. Upper-body/head joints are held by a soft joints task (weight 1.0). Joint velocity limits are 12 rad/s in the generator — [placo_walk_engine.py](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/main/open_duck_reference_motion_generator/placo_walk_engine.py)
- The Open Duck Playground RL policy imitates that reference. The imitation reward includes:
  - torso orientation exp(−20‖q−q_ref‖²) (w = 1.0)
  - base angular velocity xy exp(−2‖·‖²) (w = 0.5)
  - joint positions (w = 15, head/neck excluded) and foot contacts (w = 1)

  Top-level scales: tracking_lin_vel 2.5, tracking_ang_vel 6.0, action_rate −0.5, stand_still −0.2, alive 20, imitation 1.0. Control runs at ctrl_dt = 0.02 s with action_scale 0.25 — [Open_Duck_Playground custom_rewards.py](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/custom_rewards.py); [joystick.py](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/joystick.py)

**RoboCup open-loop walk engines**
- Rhoban IKWalk parameters:
  - `swingGain` = "amplitude of lateral oscillation of the trunk with respect to the feet (in meters)"
  - `swingPhase` = phase shift of that oscillation
  - `riseGain` = swing foot height
  - `supportPhaseRatio` = double-support fraction
  - `trunkRoll` = constant roll offset

  The engine also has a `swingRollGain` that adds trunk roll proportional to the swing spline — [IKWalk parameters.md](https://github.com/Rhoban/IKWalk/blob/master/Docs/parameters.md); [IKWalk.cpp](https://github.com/Rhoban/IKWalk/blob/master/IKWalk.cpp)
- IKWalk example values (leg 0.093 + 0.105 + 0.032 m, similar scale to Open Duck):
  - `freq` = 1.7 Hz full cycle (≈0.59 s period, almost the same as the 0.6 s here)
  - `swingGain` = 0.02 m, **`swingRollGain` = 0.0, `trunkRoll` = 0.0**
  - `riseGain` = 0.035 m, `supportPhaseRatio` = 0.0
  - `distFeetLateral` = 0.092 m, `footYOffset` = 0.025 m, `trunkPitch` = 0.15 rad

  So lateral balance comes from ±2 cm trunk *translation*, not roll — [IKWalk Example/example.cpp](https://github.com/Rhoban/IKWalk/blob/master/Example/example.cpp)
- Bit-Bots quintic walk (derived from IKWalk) parameterizes lateral sway as `trunk_swing` (fraction of foot distance, 0.1–1) plus `trunk_phase` (−0.1…0.1) — [Bit-Bots docs](https://docs.bit-bots.de/package/bitbots_quintic_walk/latest/)

**Other bipeds without ankle roll**
- Cassie (no ankle roll): the virtual constraints include **torso roll and pitch in world coordinates**, which "can be controlled through the hip motors" when the stance foot is on the ground. Gait-library optimization puts torso pitch/roll and hip roll/yaw into the cost "multiplied by a large weight, c = 10,000" to find "gaits with minimal roll and yaw movement". Lateral velocity is stabilized by foot placement — [Gong et al., Feedback control of a Cassie bipedal robot](https://grizzle.robotics.umich.edu/files/Feedback_Control_of_a_Cassie_Bipedal_Robot__Standing_and_Walking_Final.pdf)
- BRUCE (miniature humanoid): "each leg is composed of a spherical hip joint, a single DoF knee joint, and a single DoF ankle joint ... each foot is designed to have a line contact with the ground so that the actuation in the foot roll direction can be omitted" — [Liu et al., UCLA RoMeLa](https://www.seas.ucla.edu/~junjie/publication/Design%20and%20Control%20of%20a%20Miniature%20Bipedal%20Robot%20with%20Proprioceptive%20Actuation%20for%20Dynamic%20Behaviors.pdf). BRUCE is 70 cm tall, 4.8 kg, 16 DoF — [IEEE Spectrum (search snippet)](https://spectrum.ieee.org/video-friday-bruce)
- Mercury (6-DoF) and DRACO (10-DoF) have passive ankles and balance through a "time-to-velocity-reversal (TVR)" foot placement planner with whole-body locomotion control — [Kim et al., arXiv 1901.08100](https://arxiv.org/abs/1901.08100)
- Humanoid-Gym (a widely copied RL template for 5–6-DoF legs): the reference is a phase sinusoid applied **only to hip pitch (0.17 rad), knee (0.34 rad) and ankle pitch (0.17 rad)**, with stance-leg references at default. cycle_time is 0.64 s and target_feet_height 0.06 m. So the foot is lifted with sagittal joints only — [humanoid_env.py compute_ref_state](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py); [humanoid_config.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_config.py)

### Inferences
- **Rounded soles are the most direct hardware fix.** They give the robot the passive ankle roll it lacks, so the stance leg can lean over the foot while hip roll keeps the pelvis level. Disney chose this on a robot of the same kinematic class. A sole arc in the roll direction (est. radius of a few × foot width) plus foam is a low-cost experiment in both MuJoCo (capsule/ellipsoid foot geoms) and hardware.
- **The original Open Duck reference has a shorter step than this setup.** Its step is ≈0.20 s (0.17 s single support plus ~0.03 s double support), about 1.5× the cadence of the current fixed 0.6 s period, with 16 cm feet spacing and 4 cm foot height. Imitating a level-trunk reference (torso-quaternion tracking) is how that project kept roll down. Going back to (or blending in) a placo reference with torso-orientation imitation is a well-trodden path for this exact robot.
- The IKWalk defaults (same leg scale, ~same cycle period) show that a hand-tuned engine gets by with ±2 cm trunk translation and zero roll. On a robot with ankle roll that is easy; without ankle roll, translation needs the foot to tip, which again points to rounded soles.

### Gaps
- No primary documentation was found on ToddlerBot, Robotis, or kid-size RoboCup robots with genuinely 5-DoF (no ankle roll) legs, or on how much trunk roll those robots show. The kid-size engines found (IKWalk, Bit-Bots) are for 6-DoF legs.
- Measured torso-roll amplitudes for Disney's droid or Open Duck hardware were not reported in the sources read.
- Upkie (wheeled) and Bolt (point feet, 3-DoF legs) were not researched further.

## 3. RL-specific techniques: reference imitation with limited roll, periodic foot-height rewards, orientation penalties, commanded gait frequency, curricula, smoothing, critics

### Takeaway
The proven RL recipes for this robot class shape the gait directly rather than only penalizing posture:
- imitation of a level-trunk reference, tracked with high-weight joint/torso terms (Disney, Open Duck);
- sagittal-only periodic joint references plus foot-clearance rewards (Humanoid-Gym);
- bounded exp-kernel orientation rewards and a "keep hip roll near default" reward;
- gait frequency and swing height as commands (HugWBC, Walk These Ways);
- a penalty curriculum or constraints-as-terminations, so posture terms never make "stand still" the optimum.

### Cited Findings
**Imitation of reference motion**
- Disney's imitation rewards:
  - torso position exp(−200‖Δp_xy‖²) (w = 1.0)
  - leg joints −‖q_l − q̂_l‖² (w = 15)
  - **neck joints −‖q_n − q̂_n‖² (w = 100)**
  - leg action rate (w = 1.5), neck action rate (w = 5)
  - leg action acceleration (w = 0.45), neck action acceleration (w = 5)
  - contact match (w = 1)

  Actions get a first-order hold and then a 37.5 Hz low-pass filter (policy 50 Hz, actuators 600 Hz) — [Grandia et al., arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Open Duck imitation reward: see Q2. It tracks the reference torso quaternion with exp(−20‖Δq‖²) and the base angular velocity xy — [custom_rewards.py](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/custom_rewards.py)

**Orientation / posture / clearance shaping (Humanoid-Gym)**
- Orientation reward = [exp(−10·Σ|roll, pitch|) + exp(−20·‖g_xy‖)]/2, weight 1.0. It is a *bounded positive* reward, not an unbounded penalty — [humanoid_env.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py)
- default_joint_pos reward = exp(−100·clamp(‖hip yaw,roll dev‖_L + ‖…‖_R − 0.1, 0, 50)) − 0.01‖q − q_default‖, weight 0.5. This keeps **hip roll/yaw within ~0.1 rad of default** — [humanoid_env.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py)
- Other scales:
  - gait: joint_pos (reference) 1.6, feet_clearance 1.0, feet_contact_number 1.2, feet_air_time 1.0
  - feet_distance 0.2, knee_distance 0.2, base_height 0.2, base_acc 0.2
  - action_smoothness −0.002, dof_acc −1e-7
  - tracking_lin_vel 1.2, tracking_ang_vel 1.1, tracking_sigma 5

  [humanoid_config.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_config.py)
- Periodic reward composition (Cassie): phase-dependent costs on foot force norm (penalized in swing) and foot velocity norm (penalized in stance). A parametric family covers standing, walking, hopping, running and skipping, and transferred sim-to-real — [Siekmann et al., arXiv 2011.01387](https://arxiv.org/abs/2011.01387)

**Commanded gait parameters**
- HugWBC: one policy tracks 8 command dimensions (v_x, v_y, ω_z, **gait frequency, foot swing height**, body height, body pitch, waist yaw) and produces walking, jumping, standing and hopping — [HugWBC, arXiv 2502.03206](https://arxiv.org/html/2502.03206v3)
- Walk These Ways: a single policy encodes "a structured family of locomotion strategies" (Multiplicity of Behavior), with "variable footswing, posture, and speed" tunable at runtime — [Margolis & Agrawal, CoRL 2022](https://arxiv.org/abs/2212.03238)

**Smoothness**
- Lipschitz-Constrained Policies impose a gradient penalty on the policy. They "replace the need for smoothing rewards or low-pass filters" and were validated on several humanoids — [Chen et al., arXiv 2410.11825](https://arxiv.org/html/2410.11825v2)

**Constraints and curricula**
- CaT (constraints as terminations): a violation sets a termination probability δ = max_i p_i^max·clip(c_i⁺/c_i^max, 0, 1), where c_i^max is an EMA of the batch max violation. Hard constraints use p^max = 1. Soft constraints ramp p^max from 0.05 to 0.25. Style constraints on Solo-12 included **base orientation (0.1 rad)**, hip angles (0.2 rad), air time (0.25 s) and contact count. Minimal change to PPO — [Chane-Sane et al., arXiv 2403.18765](https://arxiv.org/html/2403.18765v1)
- Cost curriculum (ANYmal): "high penalty on [joint torque/velocity] results in a standing behavior ... such a behavior is already a good local minimum when there is high penalty associated with motion." Fix: multiply all cost terms (except the objective) by k_c, with k_c,0 = 0.3 and k_c ← k_c^0.997 per iteration. "k0 should be chosen to prevent the initial tendency to stand still" — [Hwangbo et al., Science Robotics 2019 (arXiv 1901.08652)](https://arxiv.org/pdf/1901.08652)
- Multi-critic PPO: separate critics per reward group; "the overall advantage is calculated as the sum of the normalized advantage estimates from each critic", which stops dominant terms from biasing value estimates — [Multi-critic Learning for Whole-body End-effector Twist Tracking, arXiv 2507.08656](https://arxiv.org/html/2507.08656v2). A similar dense/sparse reward-group multi-critic appears in [RobotKeyframing, arXiv 2407.11562 (search snippet)](https://arxiv.org/pdf/2407.11562)

### Inferences
#### Ranked candidate approaches for this robot (expected effects are estimates)
1. **Replace the "penalize head velocity" signal with a gait that does not need roll.**
   - Mechanism: a sagittal-only periodic reference or foot-height target (Humanoid-Gym style) with modest clearance (1–1.5 cm), plus a bounded hip-roll-near-default reward (exp kernel, 0.1 rad dead-band).
   - Why it should help: it removes the hip-hiking incentive (Q1) while keeping the knee within the STS3215 speed envelope.
   - Expected effect: roll range down to the range needed by H-LIP plus flywheel corrections, plausibly <10° (est.), so head sway ~3–4 cm (0.2 m lever × roll range + CoM sway).
2. **Imitation of a level-trunk placo reference** (the original Open Duck recipe).
   - Settings: torso-quaternion exp(−20‖Δq‖²) plus base ang-vel xy terms, at the robot's natural cadence (~0.2–0.25 s steps), with 0.14–0.16 m feet spacing.
   - Expected effect: H-LIP CoM sway ±1.3–2.4 cm (Q1 table). Torso roll is limited by how well the policy tracks the reference.
3. **Rounded soles / passive ankle roll** (Disney), in simulation first.
   - Lets the stance leg lean so the pelvis can stay level. It changes the contact model, so it needs a hardware change too.
4. **Make step frequency and step width commands or learnable outputs** (HugWBC/WTW-style), not fixed at 0.6 s / 18 cm.
   - H-LIP says sway ∝ w·T². Going from 0.3 s/18 cm to 0.25 s/14 cm cuts required sway by ~40 % (3.1 → 1.8 cm, est.), but it raises knee-speed demand (Q1). Prefer narrowing width first.
5. **Posture as constraint, not penalty.**
   - CaT termination on |roll| > θ_max (e.g. 0.1–0.15 rad, soft p^max 0.05→0.25), or a k_c curriculum on the posture/head terms.
   - Expected to avoid the "refuse backward/turn" local optimum, because the task reward is never outweighed early in training.
6. **Asymmetric critic / multi-critic for posture terms.** Normalizing each reward group's advantage separately stops a large posture penalty from dominating tracking. (Asymmetric actor-critic itself was not separately sourced here.)
7. **Smoothing.** A 30–40 Hz action low-pass (Disney) or LCP gradient penalty reduces jitter. It is not expected to change the 1.7 Hz roll oscillation itself.

- **Diagnostic before choosing.** Decompose the roll into (a) the hip-roll joint trajectories of the stance vs swing side, (b) swing-foot clearance (how much comes from pelvis roll vs knee flexion), and (c) centroidal L_x. If (b) dominates, use options 1 or 3. If (c) dominates, the robot is using the torso as a flywheel, and the fix is options 4 or 2 (accept more CoM sway or narrow the width).

### Gaps
- No paper found quantified how much a projected-gravity/orientation penalty reduces trunk roll on a no-ankle-roll biped, or documented its side effects as a function of weight.
- No source specifically on "reward normalization per command" for locomotion. The multi-critic advantage normalization is the closest documented mechanism.

## 4. Head/gaze stabilization with neck joints: what can a 4-DoF neck compensate (orientation vs. translation)?

### Takeaway
Robots and birds alike stabilize head **orientation** with neck rotations almost perfectly, but **translation** much less well:
- On iCub "the neck compensates the rotational component whilst the eyes ... counterbalance the translational part".
- Pigeons have rotation gain ≈1 but poor translation compensation at low frequency.

With Open Duck's neck (neck_pitch, head_pitch, head_yaw, head_roll), only the head_roll joint acts in the frontal plane, so lateral head translation cannot be cancelled while keeping the head level. Since head sway ≈ 0.2 m × torso roll range here, the fix has to come from reducing torso roll.

### Cited Findings
- iCub gaze stabilization (3-DoF neck + 3-DoF eyes): "the neck compensates the rotational component ω_FP, whilst the eyes have to counterbalance the translational part v_FP". Results:
  - 68.1 % average optical-flow reduction for self-generated torso motions; feedforward was 23.1 % better than feedback; adding neck DoF added 24.4 %.
  - 78 % reduction for external perturbations (neck adds 21.4 %).

  [Roncone et al., Gaze stabilization for humanoid robots (arXiv 1411.3525)](https://ar5iv.labs.arxiv.org/html/1411.3525)
- The head-stabilization literature notes that humans "tend to stabilize their head orientation while accomplishing a locomotor task". In anchored-gaze walking, head pitch relative to the trunk compensated trunk pitch rotation. This is a search-result summary of the Falotico/SABIAN line of work; the exact paper behind the quote was not confirmed — [Falotico et al. (HAL)](https://hal.archives-ouvertes.fr/hal-01142703)
- Pigeons on a motion platform:
  - Rotations: "all gains were close to one and the phases were close to zero", independent of frequency (0.25–2 Hz).
  - Translations: gains "close to zero" at low frequencies, ~1 at 2 Hz but with ~60° phase lag, ">1" at 4 Hz.
  - Vision improved translational but not rotational stabilization.

  [Theunissen & Troje, Front. Neurosci. 2017](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00551/full)
- SABIAN humanoid: IMU-driven neck controllers (an inverse-kinematics controller and a bio-inspired feedback-error-learning controller) stabilized **head pitch** during walking, alongside VOR/OKR models — [Falotico et al. (HAL summary)](https://hal.archives-ouvertes.fr/hal-01142703)
- Disney droid: head commands are offsets relative to a nominal head motion and are expressed in the *path frame* (not world or torso), with very high neck tracking weight (100) and neck action-rate/acceleration penalties (5) — [Grandia et al., arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Open Duck v2 neck joints: `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` (plus antennas). Open Duck Playground excludes neck/head joints from the imitation reward — [placo_defaults.json](https://github.com/apirrone/Open_Duck_reference_motion_generator/blob/main/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_defaults.json); [custom_rewards.py](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/custom_rewards.py)

### Inferences
- **Head sway here is almost entirely torso roll × lever arm (est.).** Lever ≈ 0.20 m (head 0.38 m above the feet, hip ~0.18 m). Peak-to-peak head sway ≈ 2L·sin(R/2):
  - R = 24° → 8.3 cm; R = 30° → 10.4 cm (matches the measured 8.5–12 cm)
  - R = 10° → 3.5 cm; R = 5° → 1.7 cm (plus the ~1.4 cm CoM shift)

  Getting head sway to ≤3–4 cm needs a torso roll range of roughly ≤6–10°.
- **What the neck can and cannot do (est., kinematics).**
  - The pitch joints (neck_pitch, head_pitch) have lateral axes and move the head only in the sagittal plane.
  - head_yaw moves a forward-offset head laterally by d·sin(ψ), but that rotates the gaze.
  - head_roll at the neck top: with torso roll φ, the roll joint itself translates H_n·sin φ. The head centre (r_h above the joint) adds r_h·sin(φ+θ).
  - Leveling the head (θ = −φ) removes only the r_h·sin φ part, a fraction r_h/(H_n + r_h) ≈ 15–30 % for r_h ≈ 3–6 cm and H_n + r_h ≈ 0.2 m.
  - Over-compensation (θ = −2φ) roughly doubles that, but visibly counter-tilts the head by φ.

  So a 1-DoF frontal-plane neck can fix head roll (gaze orientation) but cannot cancel lateral translation without tilting the head. This matches the iCub finding that neck joints handle rotation and translation needs other DoFs.
- **A VOR-like feedforward is still worth adding.** head_roll = −(torso roll from IMU) and neck/head pitch countering torso pitch, as Disney-style references or as a simple post-policy layer. It keeps the camera/eyes level and removes the visible "head tilt" component. Gaze-related image motion is orientation-dominated for distant targets, so this captures most of the visual benefit even though translation remains.
- **Why the world-frame head-velocity penalty conflicts with turning (est.).** A head offset d forward of the yaw pivot gets world lateral velocity d·ω_z during turns (5 cm × 1 rad/s = 5 cm/s). Backward walking also changes the sway pattern. Measuring head motion relative to a command-consistent frame (Disney's path frame), or penalizing deviation from a planned head trajectory, removes that conflict.
- The head is ~26 % of the mass and high up. Moving it laterally relative to the torso shifts the whole-body CoM by 0.26 × displacement, so a counter-sway of the head would itself perturb balance. Translation-based head "stabilization" is therefore also dynamically costly on this robot.

### Gaps
- No quantitative data found for chicken (as opposed to pigeon) translational head stabilization accuracy, or for neck lengths/DoF needed to cancel translation.
- SABIAN's quantitative reduction percentages were not retrieved (only the abstract-level summary).
- Open Duck v2's exact neck geometry (H_n, r_h, joint offsets) was not checked against the MJCF/URDF; the 15–30 % figure depends on it.

## 5. Pitfalls: why strong posture penalties make policies refuse motions, and established fixes

### Takeaway
A heavy penalty on motion or posture makes "don't move / refuse the hard command" a strong local optimum. Hwangbo et al. say this explicitly for ANYmal. The established fixes:
- a cost curriculum (k_c 0.3 → 1);
- constraints instead of penalties (CaT);
- bounded exp-kernel style rewards;
- per-group advantage normalization (multi-critic);
- potential-based shaping (policy-invariant);
- separate or command-conditioned policies (Disney uses separate perpetual and periodic policies);
- measuring style terms in a command-consistent frame.

### Cited Findings
- "Low penalty on joint torque and velocity results in unnatural motions whereas high penalty on them results in a standing behavior. The main reason for the standing behavior is that such a behavior is already a good local minimum when there is high penalty associated with motion." Fix: a multiplicative curriculum factor on all cost terms except the objective (k0 = 0.3, k_d = 0.997), so "the robot first learns how to achieve the objective and then how to respect various constraints" — [Hwangbo et al., arXiv 1901.08652](https://arxiv.org/pdf/1901.08652)
- CaT: "reward shaping remains a meticulous endeavor". Turning constraint violations into stochastic terminations gives a dense recovery signal, with excellent constraint adherence and minimal changes to PPO. Soft-constraint termination probability ramps 0.05 → 0.25 — [Chane-Sane et al., arXiv 2403.18765](https://arxiv.org/html/2403.18765v1)
- Potential-based shaping F(s,s′) = γΦ(s′) − Φ(s) is necessary and sufficient for policy invariance (Ng, Harada, Russell, ICML 1999) — [UT Austin reading page](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)
- Multi-critic advantage normalization per reward group keeps dominant reward terms from biasing value estimates — [arXiv 2507.08656](https://arxiv.org/html/2507.08656v2)
- Disney: "it is advisable to train at least one policy for perpetual motion and one to imitate a periodic motion". Standing and walking get separate policies, command interfaces and references — [Grandia et al., arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Bounded exp-kernel orientation reward (Humanoid-Gym), rather than an unbounded quadratic penalty — [humanoid_env.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py)

### Inferences
- The observed failure (the head-velocity penalty pushed sway to ~8.5 cm, then backward/turn commands were refused) fits the Hwangbo local-minimum mechanism. Backward and turning commands inherently create more world-frame lateral head motion (Q4), so the cheapest way to cut the penalty on those commands is to under-track them.
- Fixes ranked by how directly they address this failure (est.):
  1. Express the head/torso style term relative to a command-consistent reference (path-frame head trajectory, or deviation from a reference gait), so a correctly executed turn earns zero penalty.
  2. Make posture a CaT constraint with a threshold (|roll| ≤ ~0.1–0.15 rad) instead of a quadratic cost. A threshold leaves no gradient to trade against tracking below the limit.
  3. Use a k_c curriculum on style costs.
  4. Use a multiplicative form (tracking reward × style factor ∈ (0,1]), or per-group advantage normalization, so the penalty cannot exceed the tracking gain.
  5. Train separate policies or command-gated weights (the repo already has a head-sway gate per the git log) if a single policy keeps compromising.
- Potential-based shaping (Φ = −|roll| or −head offset) adds guidance without changing the optimal policy. By itself it does not *force* lower roll; it only speeds up learning toward whatever the base reward prefers. It is useful only if paired with a base reward or constraint that actually prefers low roll.

### Gaps
- No locomotion-specific study was found comparing these fixes head-to-head on a posture/head-stability objective; the ranking above is reasoning, not measured.
- No source found on "per-command reward normalization" as a named technique.
