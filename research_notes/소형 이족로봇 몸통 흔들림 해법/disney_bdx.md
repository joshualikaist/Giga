# Disney Research BDX-style robotic character control (BDX RSS 2024, Olaf 2025/26, gait tool, AMOR and related) and what transfers to Open Duck Mini v2

Primary sources read in full (PDF text extracted locally):
- BDX paper: Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer, "Design and Control of a Bipedal Robotic Character", RSS 2024; arXiv 2501.05204v1 (9 Jan 2025). Same numbers in the Disney-hosted PDF. — [arXiv](https://arxiv.org/abs/2501.05204), [Disney PDF](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf)
- Olaf paper: Müller, Knoop, Mylonopoulos, Serifi, Hopkins, Grandia, Bächer, "Olaf: Bringing an Animated Character to Life in the Physical World", arXiv 2512.16705v2 (2 Apr 2026), IEEE RA-L. — [arXiv](https://arxiv.org/abs/2512.16705)
- Gait tool paper: Hopkins, Wiedebach, Cesare, Bishop, Knoop, Bächer, "Interactive Design of Stylized Walking Gaits for Robotic Characters", ACM TOG 43(4) art. 137, 2024 (the procedural gait generator BDX cites as [14]). — [Disney PDF](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf); patent [US 12,293,445](https://patents.google.com/patent/US12293445)
- Related: AMOR (SIGGRAPH 2025, [arXiv 2505.23708](https://arxiv.org/abs/2505.23708)); RobotMDM (SIGGRAPH Asia 2024, [PDF](https://la.disneyresearch.com/wp-content/uploads/RobotMDM_red.pdf)); VMP (SCA 2024 / CGF, [PDF](https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf)); "Walk Like This" (arXiv 2610.18147, 13 Oct 2026, [arXiv](https://arxiv.org/abs/2610.18147)).
- Open-source: [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) (read at commit b9be205, 2025-08-05), [apirrone/Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) (commit 3d7bc6f, 2025-04-03), [apirrone/Open_Duck_Mini v2 docs](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md), [rimim/AWD](https://github.com/rimim/AWD) (commit 7ef6015, 2024-09-16). Code was read only, never run.

## Q1. Overall architecture: separate policies? How are animation references fed to the policy, and how are they parameterized by command?

### Takeaway
BDX uses several separate PPO policies: one "perpetual" standing policy, one "periodic" walking policy, and several "episodic" policies, one per animation clip. Each policy is conditioned on a low-dimensional command g_t and a phase φ_t. The reference motion is never an observation. It enters only through the imitation reward. The policy gets only the phase (as sin/cos harmonics) plus the command, and a reference generator turns (path frame, phase, command) into a full kinematic target. For walking, that generator is a procedural gait tool: artist-authored gait samples (forward/reverse/strafe/turn) are blended by command, made dynamically consistent with MPC plus inverse dynamics, and output a command-dependent phase rate (step frequency).

### Cited Findings
- **Three motion types and policies.** BDX defines *perpetual* motions (no start or end; balance plus continuous control inputs), *periodic* motions (a periodic phase signal is passed to the policy and cycles indefinitely) and *episodic* motions (fixed duration, monotonically increasing phase, then a forced transition). For the character they trained "one perpetual policy for standing while controlling the head and torso, one periodic policy for walking with separate head control, and several episodic policies that are each trained to imitate a single animation sequence." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- Policy form: π(a_t | s_t, φ_t, g_t), with reward r_t = r(s_t, a_t, s_{t+1}, φ_t, g_t). Each policy is a separate RL problem, and "we randomize the control inputs, g_t, over their full range during training", which lets policies be switched on the fly at runtime. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Reference (target) state.** Eq. 1: x_t = (p_t, θ_t, v_t, ω_t, q_t, q̇_t, c^L_t, c^R_t), i.e. torso global position, torso quaternion, torso linear/angular velocity, joint positions/velocities and left/right foot contact indicators. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Generator functions** map the path frame f_t (plus phase and command) to the target. Eq. 2–4: x_t = f^perp(f_t, g^perp_t); (x_t, φ̇_t) = f^peri(f_t, φ_t, g^peri_t); x_t = f^epis(f_t, φ_t). The periodic generator also outputs the phase rate φ̇_t, which "allows us to vary the stepping frequency during walking as a function of the commands." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Commands.**
  - Standing: g^perp = (Δh^head, Δθ^head, h^torso, θ^torso). Head height and orientation are offsets from a nominal configuration. Torso height and ZYX-Euler orientation are in path-frame coordinates.
  - Walking: g^peri = (Δh^head, Δθ^head, v^P, ω^P), where v^P is a 2D path-frame velocity and ω^P a yaw rate. For walking, "we assume the head commands to be offsets relative to a nominal head motion that the animator specifies as part of the periodic walk cycle."
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Path frame.** Each motion is stored in path coordinates and mapped to world coordinates through the path frame. While standing, the frame slowly converges to the centre of the two feet. While walking, it integrates the commanded path velocity. It is "projected to a maximum distance from the current torso state" to prevent drift. Fig. 4 caption: "The horizontal torso trajectory ... will generally sway around the path frame to shift weight between the feet." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **How references are generated.**
  - Standing references: inverse dynamics finds a pose that satisfies g^perp, with the remaining DoFs chosen so that "the center of pressure is in the middle of the support polygon."
  - Walking references: "an artist provides reference gaits at several walking speeds, defined as task space trajectories for torso and end-effectors. These gait samples are procedurally combined [14] into a new gait based on the commands." Then "a model predictive controller similar to [51] (Wieber 2006) is used to plan the desired center of mass and center of pressure", and these are tracked by "an inverse dynamics controller similar to [21] (Koolen/IHMC) to obtain whole-body trajectories."
  - Episodic clips are authored in Maya. The generators are densely pre-sampled and interpolated during RL.
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Policy observation.** Eq. 8: s_t = (p^P_t, θ^P_t, v^T_t, ω^T_t, q_t, q̇_t, a_{t−1}, a_{t−2}). This is torso xy position and orientation in the path frame, torso velocities in body frame, joints, and the two previous actions. Phase and command are added on top. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Phase features.** For walking the phase is replaced by its first two harmonics (sin(k·2πφ), cos(k·2πφ)), k ∈ {1,2}: "the second harmonic makes it easier for the policy to learn the head-bob which occurs at twice the gait frequency." Episodic policies use N = 50 Gaussian basis functions of φ. — [BDX arXiv, App. A](https://arxiv.org/abs/2501.05204)
- **Phase vs. full reference as input.** Feeding "current and future kinematic reference poses as the policy inputs in place of the phase signal ... converges to the same reward and produces a visually identical motion ... the phase signal and kinematic reference contain the same information." The phase is preferred because the robot does not need to store the references. — [BDX arXiv, Sec. VII-B](https://arxiv.org/abs/2501.05204)
- **Ablation of alternative formulations** (BDX Sec. VII-B):
  1. Tracking command velocities directly with the torso, with leg-joint/contact imitation weights at zero and no phase: "rapidly shuffles the feet". Foot clearance and slip penalties are "tedious to tune and the resulting gait remains far from natural".
  2. Phase plus contact reward but still tracking velocity commands directly: "follows the stepping pattern well. However, the effect of directly tracking the velocity commands remains visible, resulting in a stiff and upright torso motion."
  3. The full imitation formulation gives the natural gait.
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Gait tool details (the walking reference generator).**
  - Walking styles are authored as 9 gait samples at key velocities: Forward Fast, Forward Slow, Step In Place, Reverse Slow, Reverse Fast, Left/Right Strafe, Left/Right Turn.
  - The samples are blended by phase-space blending with f_soft(α) = −2α³+3α². Forward/reverse samples are blended first, then turn/strafe samples (in polar coordinates of normalized ẏ and θ̇), then the two results with α_turn-strafe = f_soft(ρ_n·(1 − η f_soft(|ẋ_n|))), η = 0.5 nominal.
  - The phase map places liftoff/touchdown events at φ = 0, 0.1, 0.5, 0.6, i.e. 10 % double support twice per cycle.
  - [Hopkins et al. TOG 2024](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **Step timing varies strongly with speed.** For the "Dino" character (20 actuators, 615 mm, 16.9 kg, 6-DoF reverse-knee legs, 4-DoF neck, 4-DoF tail) the two-step cycle is T_S = 0.95 s in place, 0.65 s at 0.2 m/s and 0.45 s at 0.4 m/s, "with the stepping frequency more than doubling as the character accelerates from 0 to 0.4 m/s." — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **Runtime transitions.**
  - Starting to walk: the phase is initialized from the commanded turning direction (left turn starts at a left step).
  - Stopping: the switch to the standing policy is delayed "until the start of the next double support phase".
  - Because policies receive the two previous actions and smoothness is rewarded, "the switch of policies is virtually undetectable".
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Walking envelope.** Max 0.7 m/s longitudinal, 0.4 m/s lateral, 1.8 rad/s turning. The robot weighs 15.4 kg and is 0.66 m tall. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Olaf keeps the architecture.** It has separate walking and standing policies and the same path frame. Standing command g = (q̂^neck, θ̂, p̂_z); walking command g = (q̂^neck, v̂^PF), so the head command is given directly as neck joint targets. Walking adds the gait phase φ_t. Walk cycles come from the same gait tool [38] "with heel-toe motion". A policy trained without heel-toe "appears more robotic." — [Olaf arXiv](https://arxiv.org/abs/2512.16705)

### Inferences
- Disney never asks the policy to trade off "low sway" against "follow backward/turn commands" through reward weights. The reverse, strafe and turn behaviour, and the torso motion that goes with each, are baked into the reference by command. The policy is rewarded for matching that reference's torso pose and velocity, not the raw command (the ablation shows direct command tracking is worse). This is the structural reason their weight table can stay fixed across all policies.
- Because the phase rate comes from the generator, step frequency is a function of the command (faster at speed). This is a lever a fixed-period phase clock cannot reach.

### Gaps
- BDX's own gait sample velocities, cycle times and stance widths are not published. Only the Dino, Groot and humanoid examples in the gait-tool paper give numbers.
- The exact interpolation density of the pre-sampled reference library is not given.

## Q2. Exact reward terms and weights as published

### Takeaway
BDX's reward is r = r_imitation + r_regularization + r_survival, with one table (Tab. I) used for all policies. Torso pose and velocity terms are exp kernels with weight ~1, while joint terms are negative quadratics with large weights (legs 15, neck 100). Survival is a large constant (20), and smoothness (action rate and action acceleration) is penalized. Olaf reuses the table with retuned weights and adds CBF-style thermal and joint-limit penalties, a foot–foot collision penalty and an impact-reduction term.

### Cited Findings
- **BDX Table I** (hat = reference quantity; ⊟ = orientation difference; per-step weights). Identical in the arXiv and Disney-hosted PDFs. — [BDX arXiv](https://arxiv.org/abs/2501.05204), [Disney PDF](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf)

| Group | Term | Formula | Weight |
|---|---|---|---|
| Imitation | Torso position xy | exp(−200.0·‖p_xy − p̂_xy‖²) | 1.0 |
| | Torso orientation | exp(−20.0·‖θ ⊟ θ̂‖²) | 1.0 |
| | Linear velocity xy | exp(−8.0·‖v_xy − v̂_xy‖²) | 1.0 |
| | Linear velocity z | exp(−8.0·(v_z − v̂_z)²) | 1.0 |
| | Angular velocity xy | exp(−2.0·‖ω_xy − ω̂_xy‖²) | 0.5 |
| | Angular velocity z | exp(−2.0·(ω_z − ω̂_z)²) | 0.5 |
| | Leg joint positions | −‖q_l − q̂_l‖² | 15.0 |
| | Neck joint positions | −‖q_n − q̂_n‖² | 100.0 |
| | Leg joint velocities | −‖q̇_l − q̇̂_l‖² | 1.0·10⁻³ |
| | Neck joint velocities | −‖q̇_n − q̇̂_n‖² | 1.0 |
| | Contact | Σ_{i∈{L,R}} 𝟙[c_i = ĉ_i] | 1.0 |
| Regularization | Joint torques | −‖τ‖² | 1.0·10⁻³ |
| | Joint accelerations | −‖q̈‖² | 2.5·10⁻⁶ |
| | Leg action rate | −‖a_l − a_{t−1,l}‖² | 1.5 |
| | Neck action rate | −‖a_n − a_{t−1,n}‖² | 5.0 |
| | Leg action acc. | −‖a_l − 2a_{t−1,l} + a_{t−2,l}‖² | 0.45 |
| | Neck action acc. | −‖a_n − 2a_{t−1,n} + a_{t−2,n}‖² | 5.0 |
| Survival | Survival | 1.0 | 20.0 |

- **Early termination** happens "when either head or torso are in contact with the ground and also if we detect a self-collision between the head and torso." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Per-clip modifications** for episodic clips: w̃(φ) = w₀ + 𝟙[φ_start < φ < φ_end]·w_extra. "Excited motion" gets extra angular-velocity weight during the torso shake. "Jump" gets an added torso-height reward, extra torso-orientation weight and extra contact weight ("Without this weight adjustment, the policy tends to cheat by keeping the toes on the ground"). — [BDX arXiv, App. A](https://arxiv.org/abs/2501.05204)
- **Olaf** keeps the same imitation formulas (‖·‖² explicit; ⊟ is "the SO(3) log-map orientation difference"). Reward = imitation + regularization + limits + impact reduction. — [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **Olaf Table III weights** (Standing/Walking where two values are given). — [Olaf arXiv](https://arxiv.org/abs/2512.16705)
  - Imitation: torso position xy 1.0/4.0; torso orientation 2.0/1.5; linear vel xy 1.5/2.5; linear vel z 1.0; angular vel z 1.5; leg joint pos 15.0; neck joint pos 40.0; leg joint vel 1.0·10⁻³; neck joint vel 0.5; contact 2.0/1.0; survival 20.0.
  - Regularization: neck action rate 5.0/10.0; leg action rate 2.0/5.0; leg action acc 0.5/1.0; neck action acc 15.0/10.0; joint torques 1.0·10⁻³; joint acc 2.5·10⁻⁶.
  - Limits and impact: neck temperature 2.0; joint limits 0.5/0.2; foot–foot collision 10.0; impact reduction 2.5·10⁻³.
- **Olaf limit and impact terms.**
  - Neck temperature: −‖min(−Ṫ_n + γ_T(T_max − T_n), 0)‖₁.
  - Joint limits: −‖min(q̇ + γ_q(q − (q_min + q_m)), 0)‖₁ and −‖min(−q̇ + γ_q((q_max − q_m) − q), 0)‖₁, with q_m = 0.1 rad and γ_q = 20.
  - Foot–foot: −𝟙[contact(L,R)].
  - Impact ("sound suppression"): −Σ_i min(Δv²_{i,z}, Δv²_max), saturated to avoid destabilizing the critic.
  - Early termination when head, torso, upper legs or arms touch the ground.
  - Thermal model Ṫ = −α(T − T_amb) + βτ², with α = 0.038, β = 0.377, T_amb = 43.94; T_max = 80 °C, T_clip = [70, 85] °C, γ_T = 0.312.
  - [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **Olaf results.** Impact reward cut mean sound level by 13.5 dB. Joint tracking MAE is 3.87° ± 2.40° standing and 4.02° ± 2.01° walking. — [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **BDX tracking MAE of joint positions (Tab. III).** Standing 0.035 rad; walking 0.123 rad; excited 0.029; happy dance 0.027; jump 0.043; tantrum 0.032. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **AMOR (Disney, SIGGRAPH 2025).** Instead of a fixed weighted sum, it trains one policy conditioned on a reward-weight vector w sampled from a simplex per episode, using multi-objective PPO. It has 7 objectives (upper-body joints, lower-body joints, foot joints, rigid-body poses, root pose, root velocities, smoothness) plus a survival bonus. Weights can be retuned after training, even on the real robot. A high-level policy can also choose weights over time. On a 20-DoF biped, tuning weights on hardware took about 1 day versus about 5 days to train, and raising the smoothness weight reduced real-robot jitter and the sim-to-real gap. — [AMOR arXiv](https://arxiv.org/abs/2505.23708)

### Inferences
These are my calculations from the published formulas.
- **Torso orientation kernel exp(−20·e²)**, where e is the rotation angle error:
  - 5° → 0.86; 10° → 0.54; 15° → 0.25; 25° → 0.022; 30° → 0.004.
  - A roll deviation of 24–30° from a near-upright reference therefore earns essentially zero from this term. The kernel is most informative for errors below about 15°.
- **Torso xy position kernel exp(−200·d²):** 2 cm → 0.92, 5 cm → 0.61, 10 cm → 0.14. This scale is tuned to a 0.66 m robot. For a smaller robot the coefficient would plausibly need to grow roughly with 1/L² (my assumption, not published).
- **Angular velocity xy kernel exp(−2·ω²):** 0.5 rad/s → 0.61, 1 rad/s → 0.14. A large periodic roll rate would be penalized here too.
- **Dominant shaping terms.** The joint-position terms are quadratic penalties (legs 15, neck 100), so with survival at 20 they dominate shaping. The torso exp terms (weight about 1 each) act as bounded bonuses.
- **Walking tracks the reference loosely.** Walking MAE (0.123 rad ≈ 7°) is roughly 3–4× the standing and episodic values. The walking policy deviates noticeably from the reference to balance, and Disney accepts that.

### Gaps
- BDX does not report how many joints are in each sum, so the per-joint error implied by the weights is not known.
- No ablation of individual weights is published for BDX.
- Olaf Table III lists no weight for "angular vel. xy" although Table I lists the term. It is unclear whether it was dropped or omitted from the table.

## Q3. Head/neck: does the policy stabilize the head in world frame? How are head commands combined with the gait? Show functions / animation layer?

### Takeaway
Nothing published indicates an explicit world-frame head-stabilization reward. The head is tracked in neck joint space (weight 100 in BDX, 40 in Olaf), and the neck joint targets come from the reference generator. Head pose commands (height/orientation offsets in BDX, neck joint targets in Olaf) are applied as offsets to the animator's nominal walk-cycle head motion, which includes a head-bob at twice the gait frequency. "Gaze holding" (body moves, head stays fixed) is implemented kinematically, in the animation engine and reference: the head is counter-rotated in the target configuration, not learned as a stabilization objective. Show functions (antennas, eyes, lamp, audio) are open-loop and outside the policy.

### Cited Findings
- The head commands Δh^head and Δθ^head are part of both standing and walking commands. In walking they are "offsets relative to a nominal head motion that the animator specifies as part of the periodic walk cycle ... the corresponding references for the neck-head assembly can be adapted to have the robot look in a particular direction during locomotion." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- At runtime the animation engine computes a target robot configuration c_t = (p^P, θ^P, q). For the head "we compare c_t to the nominal configuration of the robot and extract the relative head commands Δh^head_t and Δθ^head_t." While walking, "the lower body motion is determined entirely by the periodic policy and commanded path velocities ... we ignore the leg joint positions, which are not part of the policy inputs." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Animation engine layers.**
  1. Looped background animation (blinking, antenna motion).
  2. Triggered clips blended in with ν^bld = (1−β)ν^bg + βν^trig and c^bld = interp(c^bg, c^trig, α) (linear for positions and joints, slerp for body orientation), with ramp durations T_β = 0.1 s (show functions) and T_α = 0.35 s (body).
  3. Joystick mapping J^perp / J^peri adds offsets to head and torso pose. While walking, J^peri also outputs path velocities.
  - Show functions are modulated by speed ("As the robot reaches top speed, the antennas duck back and the eye radii narrow").
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Gaze/posture decoupling while standing.** "the left joystick modifies the body posture (torso yaw and pitch), while counter-rotating the head to maintain a fixed gaze. Meanwhile, the right joystick modifies the robot's gaze (head yaw and pitch), while introducing additive rotation of the torso when the neck reaches its kinematic limits." Fig. 6: "Posture control moves the torso without affecting the gaze." Button table: during standing, "Up pitches the torso forward while the head remains stationary"; L/R lateral stick "Roll the torso while the head remains stationary." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Neck hardware.** Neck joints are listed as neck yaw (NY), neck roll (NR), neck pitch (NP) and neck forward (NF). One lower-neck actuator is a quasi-direct drive (24 N·m, 30 rad/s, Unitree Go1-class per Tab. VI). The 3 head actuators are high-ratio Dynamixel XH540-V150 (4.8 N·m, 6.3 rad/s). Neck and head mass is 2.4 kg of 15.4 kg. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Head mechanics show up in the rewards.** Neck rewards are heavily weighted and smoothed (neck joint pos 100, neck joint vel 1.0, neck action rate 5.0, neck action acc 5.0), versus legs (15, 1e-3, 1.5, 0.45). Olaf says joint weights differ between neck and legs "to account for their substantially different reflected inertia." — [BDX arXiv](https://arxiv.org/abs/2501.05204); [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **Show functions** (antenna positions 2×1 rad, eye colours 2×3 RGB, eye radii 2×1 %, head lamp brightness) "do not affect the system dynamics and are controlled in an open-loop fashion." In Olaf, show functions (eyes, eyebrows, jaw, arms) are controlled with classical PD plus polynomial actuator mappings and a jaw feed-forward term. — [BDX arXiv](https://arxiv.org/abs/2501.05204); [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- In the gait tool, upper-body joints (including neck) are authored directly as joint-space function curves over the two-step cycle ("These include curves for each arm, neck, spine, and/or tail"). For the Dino, "most of the authoring time was spent animating the respective periodic joint trajectories ... The neck was animated to imitate a typical bird walk." — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)

### Inferences
- In BDX the head's world-frame motion during walking is approximately the reference torso motion composed with the reference neck trajectory, plus whatever torso tracking error the policy makes. No term rewards a world-frame head pose. Low head sway therefore depends on (a) small torso roll and sway in the reference, (b) accurate torso-orientation tracking, and (c) neck references that compensate.
- Whether BDX's walking neck references actively compensate torso sway is not stated. BDX says walk references are "task space trajectories for torso and end-effectors" converted by inverse dynamics, which would allow compensation if the head is treated as a task-space end-effector. But the gait-tool paper authors upper-body/neck motion in joint space, which would not compensate automatically.
- If a world-stabilized head is wanted for Open Duck, the closest Disney-consistent recipe is kinematic. Compute neck targets offline or in the generator so the head holds a path-frame pose given the reference torso motion, then track neck joints with a high weight. This would replace a world-frame head-velocity penalty that fights the legs.

### Gaps
- There is no statement of the BDX walk reference's head lateral displacement or of a measured head sway on hardware.
- The paper does not say whether the walking policy compensates torso roll with the neck beyond tracking neck joint references.
- RSS talk and video transcripts were not checked.

## Q4. Lateral balance without ankle roll: hip strategy, step width, trunk sway in the reference. How much torso roll does the reference have?

### Takeaway
BDX has no ankle roll. Disney uses rounded foam foot soles for passive ankle roll. Lateral balance is planned in the reference by an MPC that shifts the CoM between footholds (CoP kept inside the support polygon), while the authored pelvis orientation is upright by default (nominal z-axis vertical) plus optional artist roll curves. Step width ("stance width" attribute), step timing (short swings, high frequency at speed) and foot placement are the main lateral levers. The RL policy then deviates from reference footholds and timing when needed. No numeric torso-roll amplitude for BDX is published.

### Cited Findings
- **Feet.** "the robot has the ankle actuators directly placed at the two feet and is not equipped with ankle roll actuators. To allow for passive ankle roll, we rounded off the two foot soles. By molding them with urethane foam, foot impacts on hard ground are further dampened." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Leg DoFs** (Fig. 9): hip yaw (HY), hip roll (HR), hip pitch (HP), knee pitch (KP), ankle pitch (AP). Knees bend backwards. Nominal leg length is 0.28 m, extended 0.34 m. Mass split: torso 5.8 kg, neck+head 2.4 kg, each leg 3.6 kg. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Gait tool reference outputs** are r(t) = [z_pelvis, θ_pelvis, p_foot,l, p_foot,r, q_upper]. "Note that the reference does not specify a desired horizontal pelvis translation ... we rely on a model predictive controller to stabilize the CoM dynamics subject to the physical contact constraints." "automating the horizontal motion of the pelvis in this manner has little impact on the perceived style of a gait, and typically leads to more natural looking walking compared to manual curve editing." — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **Pelvis orientation.** "The nominal pelvis rotation ... is defined such that the pelvis z-axis remains vertical, while the horizontal axes are aligned with the current path frame rotation". Stylization adds "path frame-relative rotation ... prescribed by a set of Euler-ZYX angles that map to yaw, pitch, and roll" as cyclic spline curves. — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **Stance attributes.** "The stance splay, width, and x-bias attributes determine the mean angular, lateral, and forward offset between the left and right footholds relative to the walking path", allowing "narrow-stanced, wide-stanced, duck-footed, pigeon-toed, and asymmetric gaits". — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **CoM MPC.** Cart-table/LIP-style dynamics ẍ_com = (z̈_com + g)(x_com − x_cop)/z_com − τ_y/m (and the analogue for y). It is a sparse QP over CoP waypoints (two per support interval, constrained to the support polygon or the intersection with the next one) plus terminal CoM state. The cost pulls the final capture point to the last support centroid and CoP waypoints to polygon centroids, and penalizes CoP velocity. It runs at 125 Hz with a 2 s preview (~4 footholds at a 1 s cycle) and 0.02 s integration. A whole-body inverse-dynamics QP runs at 500 Hz. — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **Model-based gait feedback** (in the gait tool's own controller, not BDX RL):
  - Velocity adjustment: CoM error through a deadband and a 0.15 Hz low-pass filter.
  - Step adjustment: the foothold moves toward the CoM error, peaking around 0.1 m in a 25 N, 1 s push.
  - Time scaling κ_time = exp(k_p ε + k_d ε̇) speeds up or slows the reference. For the Dino, "time scaling is active during the nominal gait cycle, even before the push ... given the narrow foot width and fast CoM dynamics of the character."
  - [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **5-DoF leg, no ankle roll (Groot lower body, 10 actuators, line-contact feet).** "The line-contact foot and lack of ankle roll present certain animation challenges, including the inability to track long swing intervals or animate the full 6-DoF pose of the foot during swing (we choose here not to track foot roll)." — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- **BDX walking reference torso.** "The horizontal torso trajectory ... will generally sway around the path frame to shift weight between the feet" (Fig. 4). The RL policy "deviates from the reference trajectory and contact schedule to recover and maintain balance." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Ablation.** Replacing imitation with direct velocity-command tracking (even with phase and contact rewards) gave "a stiff and upright torso motion". The imitation reference therefore contains intentional, non-zero torso motion. — [BDX arXiv](https://arxiv.org/abs/2501.05204)

### Inferences
- **What "low sway" looks like in Disney's pipeline.** The torso translates laterally (CoM shift to the stance foot, set by step width and timing through the MPC) but keeps near-zero roll unless an animator adds a roll curve. A 24–30° per-step body roll means the robot uses torso rotation, not translation, as its weight-shift mechanism. Disney's reference explicitly asks for the opposite: translation via hip roll and leg kinematics, with torso orientation tracked at exp(−20e²).
- **Without ankle roll, lateral CoP cannot be modulated under a narrow or line foot.** Disney's levers are then foot placement (step width), shorter single support (higher step frequency at speed, e.g. Dino T_S 0.45 s at 0.4 m/s) and a CoM plan whose CoP stays inside the (small) support polygon. Their remark that long swing intervals cannot be tracked on 5-DoF line-foot legs suggests that a slow, long single-support gait is itself a source of large lateral excursion.
- **Rounded or soft soles** let the foot roll passively instead of the torso rolling. Whether that helps or hurts on a ~2 kg robot with flat feet is untested. It is a design lever Disney chose deliberately.

### Gaps
- No published numbers for BDX reference torso roll amplitude, lateral CoM sway amplitude, stance width or step frequencies.
- No measured hardware torso roll or head sway for BDX or Olaf.
- The MPC weight matrices W_T, W_c, W_v and gain values k_p, k_d are not given.

## Q5. Actuator modeling, low-level control rate, sim2real details and domain randomization (with relevance to small servos)

### Takeaway
BDX runs the policy at 50 Hz. Actions are PD setpoints that are linearly interpolated (first-order hold) to the 600 Hz actuator bus and low-pass filtered at 37.5 Hz. Setpoints are clipped to a maximum deviation from the measured joint position. The simulator (Isaac Gym) is augmented with a first-principles actuator model: PD, velocity-dependent torque limits, static/dynamic friction, backlash, encoder offset, velocity-proportional encoder noise and reflected inertia, all randomized per episode. Domain randomization adds observation noise, mass, friction, disturbance forces/torques on several bodies under a curriculum, and terrain for walking. The head uses Dynamixel servos with a different PD law (derivative on setpoint error). That law is the closest analogue to hobby servos like the STS3215.

### Cited Findings
- **Control pipeline.** "The policy outputs actions at a rate of 50 Hz, while our actuator communication operates at 600 Hz. To bridge this gap, we perform a first-order-hold ... followed by a low-pass filter with a cut-off frequency of 37.5 Hz." The low-level controller also implements the path frame and phase propagation, "identically implemented in the RL and runtime environments." Olaf uses the same 50 Hz → 600 Hz first-order hold plus 37.5 Hz LPF. — [BDX arXiv](https://arxiv.org/abs/2501.05204); [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **Action mapping.** The output is linearly transformed "such that 0 maps to the nominal joint positions of the robot and 1 to the expected range per joint." "joint position setpoints are clipped to a maximum deviation around the measured joint position ... chosen large enough such that the maximum actuator torque can still be produced." Inputs are normalized by expected range. — [BDX arXiv, App. A](https://arxiv.org/abs/2501.05204)
- **Actuator model** (App. B):
  - Quasi-direct-drive motors: τ_m = k_P(a − q̃) − k_D q̇, with q̃ = q + ε_q and encoder offset ε_q ~ U(−ε_q,max, ε_q,max) drawn per episode.
  - Head Dynamixels: τ_m = k_P(a − q̃) + k_D d/dt(a − q̃), i.e. the derivative acts on the numerically differentiated setpoint error.
  - Friction: τ_f = μ_s tanh(q̇/q̇_s) + μ_d q̇.
  - Joint torque: τ = clamp_[τ_min(q̇), τ_max(q̇)](τ_m) − τ_f. The limit is constant τ_max up to q̇_τmax, then ramps linearly to zero at q̇_max.
  - Measured position: q̂ = q̃ + 0.5·b·tanh(τ_m/τ_b) + N(0, σ_q²), with σ_q = σ_q,0 + σ_q,1|q̇| and backlash b ~ U(b_min, b_max) per episode.
  - Reflected inertia I_m is set as Isaac Gym armature and randomized by up to 20 %.
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Table VI parameters** (Unitree A1 / Unitree Go1 / Dynamixel XH540-V150):
  - k_P 15.0 / 10.0 / 5.0 N·m/rad; k_D 0.6 / 0.3 / 0.2 N·m·s/rad.
  - τ_max 34.0 / 23.7 / 4.8 N·m; q̇_τmax 7.4 / 10.6 / 0.2 rad/s; q̇_max 20.0 / 28.8 / 7.0 rad/s.
  - μ_s 0.45 / 0.15 / 0.05 N·m; μ_d 0.023 / 0.016 / 0.009 N·m·s/rad.
  - b_min 0.005 / 0.002 / 0.002 rad; b_max 0.015 / 0.005 / 0.005 rad.
  - ε_q,max 0.02 rad for all three.
  - σ_q,0 1.80e-4 / 1.89e-4 / 4.31e-4 rad; σ_q,1 3.61e-5 / 5.47e-5 / 2.43e-5 s.
  - I_m 0.011 / 0.0043 / 0.0058 kg·m².
  - [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Actuator assignment and saturation.** Hip roll, hip pitch and knee are the strongest (34 N·m peak, 20 rad/s). Hip yaw, ankle and lower neck are 24 N·m, 30 rad/s. The 3 head actuators are 4.8 N·m, 6.3 rad/s. In the jump, knee torque hits the velocity-dependent limits during push-off and the neck pitch actuator also saturates. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Domain randomization.** "We randomize the parameters of our actuator models within their experimentally observed range. We also add noise to the state ... randomize mass properties and frictional coefficients ... apply random disturbance forces and torques on the torso, head, hips, and feet ... During training our walking policy, we additionally randomize the terrain." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Disturbances (Table V).** Uniform on/off schedules per body, ramped in by a linear curriculum over the first 1500 iterations:
  - Hips and feet: force XY [0, 5] N, Z [0, 5] N; torque XY [0, 0.25] N·m, Z [0, 0.25] N·m; on [0.25, 2.0] s, off [1.0, 3.0] s.
  - Pelvis and head: same forces and torques; on [2.0, 10.0] s, off [1.0, 3.0] s.
  - Pelvis "short/large": force XY [90, 150] N, Z [0, 10] N; torque [0, 15] N·m; on 0.1 s, off [12, 15] s.
  - [BDX arXiv, App. A](https://arxiv.org/abs/2501.05204)
- **Training.** PPO with an adaptive learning rate. MLP of 3 × 512 ELU units, and a separate critic of the same size. The critic receives the noiseless simulation state and the friction parameter as privileged information. Hyperparameters: 100,000 iterations (~2 days on an RTX 4090; "1500 iterations or 30 min are enough to preview the nominal behavior"); batch 8192 envs × 24 steps; 4 mini-batches; 5 epochs; clip 0.2; entropy 0.0; γ 0.99; GAE λ 0.95; desired KL 0.01; max grad norm 1.0. — [BDX arXiv, App. A](https://arxiv.org/abs/2501.05204)
- **Runtime state estimation.** A standard estimator fuses IMU and actuator measurements (contact-aided invariant EKF, ref. [11]). Public deployments have run on the order of 10 h with up to 3 robots "without a single fall." — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- **Olaf training.** Isaac Sim, 8192 environments, 100k iterations (~2 days, RTX 4090). The critic additionally gets terrain height samples. Costume–leg interaction forces were "handled through domain-randomized disturbance forces." — [Olaf arXiv](https://arxiv.org/abs/2512.16705)
- **Related Disney robot.** The 20-DoF biped "LIME" (0.84 m, 16.2 kg) used in VMP, RobotMDM and AMOR has two 5-DoF legs with Unitree A1 actuators and Dynamixel XH540-V150-R in the neck and arms, with the same actuator-model family. — [VMP](https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf); [RobotMDM](https://la.disneyresearch.com/wp-content/uploads/RobotMDM_red.pdf)

### Inferences
- The BDX legs are quasi-direct-drive and high-bandwidth (k_P 10–15 N·m/rad on joints carrying a 15 kg robot). Open Duck's STS3215 are geared position servos. The Disney model for the *Dynamixel head servos* is the better template: low k_P, derivative on setpoint error, a very low q̇_τmax (torque starts dropping almost immediately with speed), backlash 0.002–0.005 rad, encoder offset ±0.02 rad. Open Duck instead uses BAM-identified STS3215 MuJoCo parameters (see Q6).
- The 37.5 Hz LPF after first-order hold, the clipping of setpoints around measured position, and the strong action-rate and action-acceleration penalties all suppress high-frequency corrections. This makes the gait smooth but limits fast roll correction. Their low sway comes from a feasible reference, not from aggressive feedback.

### Gaps
- No published randomization ranges for mass, friction or observation noise in BDX, and no terrain parameters.
- No per-joint "expected range" values for the action scaling.

## Q6. Open-source reimplementations of the BDX method and how they implement the reward and reference

### Takeaway
The main open reimplementation for our exact robot is Open Duck Playground (MuJoCo Playground / MJX + Brax PPO), with reference motions from Open_Duck_reference_motion_generator (a Placo ZMP/LIPM walk engine fitted to per-command polynomials). It copies Disney's imitation weights but in practice drops the torso orientation and torso position terms and excludes the neck from imitation. Imitation is switched off at zero command, and the imitation term sits next to separate velocity-tracking rewards. AWD (rimim) is an AMP/ASE-style adversarial reimplementation for "go_bdx"/"mini_bdx" in Isaac Gym, not Disney's tracking formulation. Our own repo (biped_sim) does not use the Open Duck imitation reward or a Placo reference.

### Cited Findings
- Open Duck docs: "We obtained good results by implementing the imitation reward described by Disney in their BDX paper ... To use this reward, we need reference motion ... generate such motion using a parametric walk engine ... polynomial_coefficients.pkl." Motors are modelled with BAM, using identified STS3215 parameters exported to MuJoCo (damping, kp, frictionloss, armature, forcerange). — [Open_Duck_Mini sim2real.md](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md)
- **`playground/open_duck_mini_v2/custom_rewards.py::reward_imitation`** — [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)
  - Weights copied from Disney: torso_pos 1.0, torso_orientation 1.0, lin_vel_xy 1.0, lin_vel_z 1.0, ang_vel_xy 0.5, ang_vel_z 0.5, joint_pos 15.0, joint_vel 1e-3, contact 1.0. Same exp coefficients (−8, −2) and quadratic joint terms.
  - The torso position term is commented out.
  - The torso orientation term is computed as exp(−20·Σ(q − q̂)²) on raw quaternion components ("real quaternion angle doesn't have the expected effect, switching back"), but it is **not added** to the sum (the line `# + torso_orientation_rew` is commented).
  - Neck, head and antenna joints are removed from joint position and velocity imitation.
  - The total is multiplied by (‖cmd[:3]‖ > 0.01), so there is no imitation reward for zero commands.
- **`playground/open_duck_mini_v2/joystick.py` config** — [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)
  - Reward scales: tracking_lin_vel 2.5, tracking_ang_vel 6.0 (tracking_sigma 0.01), torques −1e-3, action_rate −0.5, stand_still −0.2, alive 20.0, imitation 1.0. The summed reward is multiplied by dt and clipped at ≥ 0.
  - Timing: ctrl_dt 0.02 s, sim_dt 0.002 s, action_scale 0.25.
  - Motor speed limit: targets are clipped to ±5.24 rad/s × dt from the previous target.
  - Delays and noise: action delay 0–3 steps, IMU delay 0–3 steps. Noise: hip 0.03 rad, knee 0.05, ankle 0.08, joint vel 2.5 rad/s, gravity 0.1, gyro 0.1, accelerometer 0.05.
  - Pushes every 5–10 s with magnitude 0.1–1.0.
  - Commands: lin_vel_x ±0.15, lin_vel_y ±0.2, ang_vel_yaw ±1.0, plus head commands (neck_pitch [−0.34, 1.1], head_pitch ±0.78, head_yaw ±1.5, head_roll ±0.5), with a 10 % chance of an all-zero command. A line that would write the head commands straight into the head motor targets is commented out.
  - The observation includes gyro, accelerometer, command, joint angles and velocities, the last 3 actions, motor targets, contacts and imitation_phase (sin/cos).
- **Open Duck domain randomization** (`playground/common/randomize.py`): floor friction U(0.5, 1.0); frictionloss ×U(0.9, 1.1); armature ×U(1.0, 1.05); torso CoM offset U(−0.05, 0.05) m; link masses ×U(0.9, 1.1); torso mass +U(−0.1, 0.1) kg; qpos0 jitter ±0.03 rad; kp ×U(0.9, 1.1). STS3215 MJCF values: kp 13.37, forcerange ±3.23 N·m, damping 0.56, frictionloss 0.068, armature 0.027. — [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)
- **Reference generator.** It is "based on Placo". `auto_waddle.py --sweep` generates motions over command grids and `fit_poly.py` fits polynomials. — [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)
  - open_duck_mini_v2 defaults: double_support_ratio 0.18, single_support_duration 0.17 s, walk_com_height 0.2 m, walk_foot_height 0.04 m, walk_trunk_pitch −3°, feet_spacing 0.16 m, foot_zmp_target_y 0.0.
  - "medium" preset: double_support_ratio 0.5, single_support_duration 0.18 s, com height 0.205 m, trunk pitch −4°, foot_zmp_target_y −0.03 m, neck_pitch 20°, head_pitch −26°.
  - Sweep ranges: x −0.04..0.06, y −0.03..0.03, θ −0.3..0.3, with xy granularity 0.02 and θ granularity 0.07.
  - The README says the motions are used by a MuJoCo Playground RL project and by [rimim/AWD](https://github.com/rimim/AWD).
- **AWD.** "Adverserial Waddle Dynamics ... A mallard version of ASE". It provides AWD (ASE-like) and AMP training for go_bdx and mini_bdx in Isaac Gym from motion clips. — [rimim/AWD](https://github.com/rimim/AWD)
- **Our repo.** A grep of this repo (`/home/joshuali/giga`) found no `reward_imitation` or `polynomial_coefficients`. `biped_sim/envs/walk_mjx.py` has a `gait_phase` contact-schedule reward and an optional joint-space `reference_pose(t)`, a sinusoidal per-joint fwd/lift pattern scored by exp(−err/(n·0.15²)). Neither has a torso-orientation or torso-position imitation term. (Checked directly in local code; not a web source.)

### Inferences
- The Open Duck reimplementation tracks the reference's torso velocities and leg joints, but not its torso orientation or position, and it keeps separate command-velocity tracking rewards. That is closer to Disney's ablation #2 ("phase + contact + direct velocity tracking"), with imitation as a helper.
- Our setup (phase contact schedule plus sinusoidal joint reference plus command tracking) is likewise closer to Disney's ablation #1/#2 than to the full BDX formulation. Neither enforces the torso roll and CoM-translation pattern that a dynamically consistent reference would provide.
- Placo's walk engine is a ZMP/LIPM planner with a trunk orientation task. So an Open Duck reference likely has lateral CoM translation with near-upright trunk roll. This is inferred from Placo's design and the config fields (`walk_trunk_pitch`, no roll field), not verified by replaying the motions. Enabling the orientation and position terms against such a reference is the Disney-faithful way to penalize roll relative to a feasible target.

### Gaps
- I did not replay Open Duck reference motions, so the actual trunk roll and lateral CoM amplitude in `polynomial_coefficients.pkl` were not measured.
- It is unclear how Open Duck realizes head commands in the trained policy, given that the motor-target override is commented out.
- No open-source code reimplements Disney's MPC+ID gait-sample tool. Open Duck substitutes Placo.

## Q7. Synthesis: which Disney ideas map onto our problem (24–30° body roll per step, 8.5–12 cm head sway; weight tuning trades off against backward/turn commands)?

### Takeaway
Disney's published recipe does not rely on sway penalties at all. Low sway would come from (1) a dynamically consistent, command-conditioned reference (forward/reverse/strafe/turn samples, upright pelvis by default, CoM shift via an MPC over footholds, command-dependent step frequency), (2) imitation rewards on torso orientation, angular velocity and position relative to that reference, plus high-weight neck joint tracking, (3) heavy smoothness regularization and a large survival bonus, and (4) separate standing and walking policies. Their alternative for conflicting objectives is AMOR-style weight-conditioned policies. The items below are inferences for our setup, not Disney claims.

### Cited Findings
- Disney's walking reward penalizes deviation from a reference torso orientation (exp(−20‖θ⊟θ̂‖²), weight 1.0) and torso xy position in the path frame (exp(−200‖Δp_xy‖²), weight 1.0), not raw sway. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- Walk references cover forward, reverse, strafe and turning via 9 authored samples blended by command, and the generator outputs phase rate (step frequency) as a function of command. — [BDX arXiv](https://arxiv.org/abs/2501.05204); [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- Direct command-velocity tracking (even with phase and contact rewards) produced shuffling, or a "stiff and upright torso", versus the natural imitation gait. — [BDX arXiv](https://arxiv.org/abs/2501.05204)
- The pelvis CoM translation is planned by MPC from footholds and timing, while pelvis orientation defaults to upright. — [Hopkins et al.](https://la.disneyresearch.com/wp-content/uploads/Interactive_Design_of_Stylized_Walking_Gaits_for_Robotic_Characters-5.pdf)
- Weight-conditioned multi-objective PPO (AMOR) allows post-training and on-robot weight tuning instead of retraining for each trade-off. — [AMOR arXiv](https://arxiv.org/abs/2505.23708)

### Inferences
- **Root-cause hypothesis.** Penalizing head lateral velocity or rewarding CoM shift in isolation competes with command tracking, because nothing tells the policy how to move laterally for each command. Disney resolves this by making the desired lateral motion (CoM translation, near-zero roll, step width, step timing) part of a command-conditioned reference. Every command then has a feasible low-roll target, and the torso-orientation imitation term rewards it uniformly across forward, backward and turning.
- **Concrete transfers to try, in Disney order.**
  1. Generate a dynamically consistent reference (e.g. Placo LIPM like Open Duck, or a small MPC) per command including backward and turn, with explicit stance width, upright trunk and command-dependent cycle time. Feed only phase harmonics (k = 1, 2) and the phase rate.
  2. Add Disney's torso-orientation (SO(3) log map, not quaternion component difference), torso path-frame position and angular-velocity-xy imitation terms. Consider sharper kernels for a 2 kg robot.
  3. Track neck joints with a high weight against reference neck angles that are computed to keep the head steady in the path frame.
  4. Use Disney-style smoothness terms (leg action rate 1.5, action acc 0.45, neck 5.0/5.0) and survival 20.
  5. Consider separate standing and walking policies.
  6. If trade-offs remain, consider AMOR-style weight conditioning (sample w per episode, feed w to the policy) to sweep the sway versus command trade-off without retraining.
- **Hardware and design-level levers Disney used.** Rounded or soft soles for passive ankle roll. Shorter swing at higher speeds (Disney notes long swing intervals are hard to track without ankle roll).

### Gaps
- None of the Disney papers report torso-roll or head-sway metrics, so the gain these changes would bring on Open Duck is unquantified.
- BDX's leg actuators (34 N·m QDD on a 15 kg robot) have far higher torque-to-weight and bandwidth than STS3215 servos, so their feasibility margins may not transfer.
