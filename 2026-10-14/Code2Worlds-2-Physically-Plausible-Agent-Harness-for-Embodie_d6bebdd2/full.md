# Code2Worlds-2: Physically Plausible Agent Harness for Embodied World Simulation

Haoyu Zhang<sup>∗</sup>, Siming Li<sup>∗</sup>, Zeyu Zhang<sup>∗†</sup>, Yi Zhang, Tat-Seng Chua, Hao Tang<sup>‡</sup>

Abstract—Recent coding large language models (LLMs) can synthesize dynamic 4D worlds from natural language, but the resulting content remains a visual artifact. It lacks physical semantics such as the collision geometry, mass, friction, inertia, and joint structure, required to enable it to simulate the physical world, and its dynamics are scripted to look plausible rather than designed to execute. What is needed is an executable world: one that a physics engine can load and step, resolving contact physically. We present Code2Worlds 2, an extension of Code2Worlds that turns 4D world generation into an embodied world simulation. We keep the original dual-stream generation pipeline intact and append two modules. First, a SimReady Scene Compiler that turns a generated visual scene into a simulator-compatible representation: it assigns collision meshes via convex decomposition, estimates physical attributes with a vision– language model, infers a lightweight support graph, instantiates the engine’s contact, friction, and dynamics constraints, and applies a gravity settling pass, before exporting the scene as MuJoCo MJCF. Second, a Simulator-in-the-Loop 4D Refiner that implements an automated agent harness—a compile, simulate, critique and refine loop—that converts visual dynamics into executable rigid and articulated dynamics and evolves the physical settings rather than only the code, replacing the purely visual motion critic with a simulator critic that inspects penetration depth, collision pairs, stability, contact forces, joint limits, and robot reachability. We further propose Sim-Code4D, a state-based benchmark with five task templates for evaluating embodied simulation. Experiments on SimCode4D and across natural, indoor, industrial, urban, and service scenes demonstrate that Code2Worlds-2 produces scenes that are physically plausible, simulator-compatible, and interactable by legged and manipulator robots. Code: https://github.com/AIGeeksGroup/Code2Worlds-2. Website: https://aigeeksgroup.github.io/Code2Worlds-2.

Index Terms—4D world generation, embodied world simulation, coding LLM, physics simulation, robot manipulation, MuJoCo.

## 1 INTRODUCTION

laws of physics. Coding large language models (LLMs) have recently been shown to generate static 3D scenes and, more ambitiously, dynamic 4D worlds by writing programmatic scene code. Code2Worlds [1] advanced this line by proposing a dual-stream architecture that disentangles retrievalaugmented object generation from hierarchical environmental orchestration, and by narrowing the semantic–physical gap with a Post-Process Agent that scripts dynamics and a VLM-Motion Critic that self-reflects to refine the generated code. The result is a 4D world that is visually dynamic and semantically physical.

Yet for an embodied agent—a quadruped that must walk across a rocky field, a manipulator that must open a drawer—a visually plausible world is not enough. Two gaps separate Code2Worlds from a usable embodied simulator. First, a semantic–physical execution gap: generated meshes carry no collision geometry, mass, friction, inertia, or joint structure, so a physics engine such as MuJoCo [2] or Isaac Sim [3] accepts them only as raw visual hulls, where contact, support, and manipulation carry no physical meaning. Second, a simulation–fidelity gap: the dynamics are scripted to appear physically reasonable but are never

![](images/52b59f73d7c8b5bf3d3dde38305c7fdc829c6baa0e15f11d1e632d45f01e8339.jpg)  
Fig. 1. The feedback loops of the two systems. (a) Code2Worlds [1] corrects a scene by rewriting its code from rendered images—a visual loop, which cannot guarantee physical validity. (b) Code2Worlds-2 instead closes the loop through the simulator: Module 1 compiles a SimReady scene, Module 2 critiques the rollout, and simulator feedback guides the revisions to the physical settings and dynamics.

executed, so an agent cannot collide with, push, grasp, or otherwise interact with them. Fig. 1 contrasts the two loops: Code2Worlds rewrites the entire scene code, whereas Code2Worlds-2 revises the physical settings themselves.

![](images/20c9cb9a5ed7e81536774111eeb3ae4fe7accd808a204a3ea7b1afab86c81502.jpg)  
Fig. 2. Overview of Code2Worlds-2. Two modules are appended to the original Code2Worlds pipeline to produce a simulator-compatible, interactable 4D world, and the illustrated scenes span five domains.

This paper extends Code2Worlds into Code2Worlds-2, an embodied world simulation framework. We keep the original pipeline unchanged and append two modules that bridge these gaps (Fig. 2). The first, a SimReady Scene Compiler (Module 1), wraps each visual scene with a physical compilation layer: it decomposes concave meshes into convex pieces, estimates physical attributes with a vision– language model and geometry, infers a support graph, and settles objects under gravity before exporting MJCF. The second, a Simulator-in-the-Loop 4D Refiner (Module 2), translates visual dynamics into executable dynamics and replaces the purely visual motion critic with a simulator critic that reasons over physical signals rather than pixels alone. We additionally propose SimCode4D, a state-based benchmark with five task templates, and validate the framework across natural, indoor, industrial, urban, and service scenes.

Extension over the conference version. This paper extends our ICML 2026 conference paper, Code2Worlds [1]. The original generation streams are retained; the journal version adds (i) physical scene compilation and MuJoCo export (Sec. 3.1), (ii) simulator-in-the-loop dynamics refinement and physical validation (Sec. 3.2), and (iii) the state-based SimCode4D protocol and embodied-interaction evaluation (Secs. 4 and 5).

In summary, this paper makes the following contributions:

We propose a SimReady Scene Compiler that converts coding-LLM-generated visual scenes into simulator-compatible representations with collision geometry, physical attributes, a support graph, physical constraints, and a settling pass, while emitting paired default and enriched scene exports whose comparison is our primary measure of physical plausibility.

• We propose an agent harness—an automated compile–simulate–critique–refine loop that evolves the physical settings of a generated world—realized by a Simulator-in-the-Loop 4D Refiner that compiles visual dynamics into executable rigid and articulated dynamics and replaces visual-only critique with simulator-signal critique (penetration, contact, stability, joint limits, reachability).

We introduce SimCode4D, a state-based benchmark with five task templates for embodied world simulation, and demonstrate physical plausibility, simulator compatibility, and robot interactability across five scene categories.

## 2 RELATED WORK

## 2.1 Coding LLMs for Scene Generation

Recent work shows that coding LLMs can generate 3D scenes by emitting programmatic code rather than raw assets, benefiting from the compositionality and editability of code. SceneCraft [4] drives an LLM agent that writes Blender code and refines it from rendered feedback; LayoutGPT [5] casts scene composition as language-conditioned layout planning; 3D-GPT [6] decomposes procedural modeling into language-driven tasks; and BlenderLLM [7] trains a model specifically for computer-aided design code. Related code-generative efforts target individual assets rather than whole scenes, including LL3M [8] and MeshCoder [9]. SceneCode [10] compiles articulated indoor scenes into executable world programs, starting from existing scenes rather than a coding-LLM-generated 4D world. Code2Worlds [1] extends this line to 4D by proposing a dual-stream architecture and a physics-aware closed loop. Our work builds directly on Code2Worlds and targets the step its design does not address: making the generated world simulatorcompatible and its dynamics executable.

## 2.2 Physics-Aware Scene Construction

Several concurrent systems make generated or reconstructed scenes physically plausible, and we build on the physical-property conventions they established.

SceneSmith [11] generates indoor scenes hierarchically through a Designer–Critic–Orchestrator loop. It combines generated assets, articulated-object libraries, and thin coverings, estimates physical properties with a vision–language model (VLM), and builds V-HACD collision geometry. Collision and reachability checks guide generation, followed by collision-free projection and gravity settling in Drake [12]. It is the closest prior work in spirit, but generates scenes from scratch rather than compiling an existing coding-LLM world.

InternScenes [13] assembles simulatable indoor scenes from real scans, procedural Infinigen [14] outputs, and cleaned designer scenes. It optimizes large-furniture bounding boxes to reduce overlap and ground misalignment, then settles small objects in SAPIEN. It emphasizes realistic layouts and collision elimination, with lighter physicalproperty modeling than SceneSmith.

SimRecon [15] reconstructs object-level, composable, simulator-ready scenes from real RGB video. It first performs semantic 3D reconstruction (2DGS plus SceneSplat segmentation), then completes each object with an activeviewpoint optimization (AVO), and finally assembles the scene through a Scene Graph Synthesizer (SGS) that infers (child, relation, parent) triples—primarily supported\_by and attached\_to—via VLM over instance-ID renders, followed by hierarchical physical assembly: support relations settle objects under gravity, attachment relations apply fixed constraints. SimRecon builds a support graph similar to ours, but it operates on reconstruction rather than generation.

These systems each address one segment of “physical scene construction,” but none takes a coding-LLMgenerated dynamic 4D world as input and compiles it endto-end into a representation that a physics simulator can load and a robot can act on. Code2Worlds-2 occupies exactly this position: it inherits Code2Worlds’s dual-stream generation and Post-Process Agent, and adds a compiler (Module 1) and a simulator-in-the-loop refiner (Module 2) that make the generated world physically grounded and robotinteractable. Moreover, our settling pass produces simulator contact data that can ground the support graph in measured contact, rather than in VLM or geometric heuristics alone.

Other physically grounded generators. Holodeck [16] generates embodied 3D environments from language descriptions, while PhyScene [17] synthesizes physically interactable scene layouts from a given floor plan and room type by guiding a conditional diffusion model with collision, room-boundary, and reachability constraints, and RoboGen [18] generates tasks together with their simulation environments. PAT3D [19] augments text-to-3D generation with physical reasoning, while URDFormer [20] predicts articulated structure and URDF descriptions from images. Common asset and scene sources for these systems include Objaverse [21], Infinigen Indoors [22], and CityX [23].

## 2.3 Simulation Platforms and Robot Benchmarks

Our compiler targets an explicit engine representation supported by several platforms: MuJoCo [2], Isaac Gym [24], Isaac Lab [25], Brax [26], and SAPIEN [27]. Manipulation benchmarks such as RLBench [28], ManiSkill2 [29], robosuite [30], Meta-World [31], CALVIN [32], and BEHAVIOR-1K [33], and interactive suites such as AI2-THOR [34], Habitat [35], RoboTHOR [36], ProcTHOR [37], Gibson [38], and HM3D [39], all take simulator-ready assets as input. SimCode4D instead asks whether an LLM-generated world can become one.

## 2.4 Generative World Models

A separate line learns interactive dynamics directly from video, including Genie [40], UniSim [41], GAIA-1 [42], WorldDreamer [43], and Ctrl-World [44]. Such models can render plausible futures, but they do not expose contact, friction, or joint state, so contact-level physical validation cannot be performed on them. Code2Worlds-2 instead commits to an explicit, engine-agnostic representation that a simulator can load.

## 2.5 Physical Property Estimation and 3D Scene Graphs

Estimating physical attributes from visual input has been approached with language-embedded features [45], singleimage reasoning [46], [47], and multi-view material decomposition of generated meshes [48]; we adopt this idea as one stage of the compiler rather than as an end in itself. A related literature represents scenes as graphs over objects and relations [49], [50], [51], [52], [53]; our support graph is a deliberately lightweight, support- and contact-focused instance of that idea.

## 3 METHOD

The central abstraction of Code2Worlds-2 is an Embodied World Graph, an engine-agnostic intermediate representation that bridges the visual world emitted by Code2Worlds and a target physics engine (MuJoCo in this work). Each object node carries the visual mesh, collision mesh, mass, friction, inertia, center of mass, scale, pose, support/contact relation, affordance, and joint type; each dynamic event carries force, velocity, gravity response, contact events, object state transitions, temporal triggers, and interaction constraints. The compiler produces this graph once, then serializes it to a target simulator (here MuJoCo MJCF), so that the same world can in principle be exported to Isaac Sim/USD or other engines without regenerating the scene.

Agent harness. The other key component of Code2Worlds-2 is a harness: the automated control loop that lets a coding agent produce a physically executable world without human intervention. In Code2Worlds, the Post-Process Agent refines code against a visual critic; in Code2Worlds-2, the harness closes the loop through the simulator itself. The agent generates scene code and a dynamics program, Module 1 compiles them into an Embodied World Graph and instantiates the engine’s constraints, Module 2 executes the rollout in MuJoCo, the simulator critic measures physical signals (penetration, contact, stability, forces, joint limits, reachability), and the fused verdict is fed back to the agent to revise the physical settings—not merely the script. The loop iterates until the scene passes both physical and visual validation (Fig. 3).

![](images/2ee71193d623dd18b34ba8ffd8aa678e6eb7c16480e198d696fc6c01de88202d.jpg)  
Fig. 3. The Code2Worlds-2 agent harness. The generation agent closes a loop through the simulator: it emits code and a dynamics program, which are compiled, executed, and measured by physical signals; the fused verdict then revises the physical settings until validation passes.

We retain the Code2Worlds generation pipeline (dualstream object/scene generation and the Post-Process Agent), extended so that the agent emits a dynamics program and acts on simulator feedback, and append two modules (Fig. 2): Module 1 (M1) compiles visual scenes into simulator-compatible representations; Module 2 (M2) converts and refines their dynamics using simulator feedback.

## 3.1 Module 1: SimReady Scene Compiler

Module 1 turns a generated visual scene into a simulationready scene without altering the object or scene streams. Its compilation stages are summarized in Fig. 5.

## 3.1.1 Collision Geometry

A simulator requires collision meshes that are either convex or decomposed into convex pieces; raw generated meshes are concave and non-watertight, and a physics engine will otherwise convex-hull each mesh into a coarse, often wrong solid. We apply CoACD convex decomposition [54] to key interaction objects, with a concavity threshold of 0.05 and a per-object piece cap (24 in our scenes) to bound solver cost; alternatives include V-HACD [55], Empart [56], and VisACD [57]. For outdoor terrain, whose non-convex height field would be convex-hulled by MuJoCo into a single solid spanning the whole scene, we slice the terrain into an N×N grid and reconstruct, in each cell, a closed convex prism from a ray-cast height field (top surface plus four side walls plus a flat base). Each cell is one geom, so MuJoCo’s convex hull follows the local surface, with error bounded by the cell’s vertical variation.

## 3.1.2 Physical Attribute Estimation

For every key object we render four views (iso, front, side, top) and send them with the object’s longest dimension and source name to a vision–language model, which predicts a material category (e.g. granite, steel, plastic), bulk density $\rho ,$ friction $\mu ,$ and restitution e. We then compute mass from the mesh volume,

$$
m = \rho V ,\tag{1}
$$

and the inertia tensor I and center of mass from the mesh moments, following the mass–inertia convention of prior physics-aware systems [11]; related visual property estimation includes GaussianProperty [58] and PUGS [59]. This yields a complete physical-attribute tuple (visual mesh, collision mesh, mass, friction, restitution, inertia, center of mass) without re-authoring any asset.

## 3.1.3 Support Graph

We infer a lightweight support graph $G \ = \ ( O , R )$ over objects. R contains the structural relations {supported\_by, on\_surface, attached\_to, inside, hinged\_to} and the settling triggers {floating, embedded}. Relations are inferred from world-space bounding boxes. Support is determined by positive xy-footprint overlap and a vertical gap between an object’s bottom and a candidate support surface (floor, terrain, or an object’s top): a gap within tolerance yields supported\_by/on\_surface, a large positive gap floating, and a large negative gap embedded. Attachment is detected by face-to-face proximity to a wall with sufficient vertical overlap; containment by full boundingbox enclosure of one object inside another; articulation by naming cues (door, drawer, panel) plus a nearby structural neighbor. Because procedural indoor scenes often export only decorative floor tiles rather than the true room floor, we add a global-ground fallback that recovers the floor height from the lowest credible object bottom, preventing spurious floating labels. The graph drives the settling pass below and is later refined by simulator contact data.

## 3.1.4 Physics Settling

We load each object into MuJoCo slightly above its support surface (5 cm) and simulate under gravity until it comes to rest, recording the settled pose; an object is accepted as settled when its linear and angular speeds fall below small thresholds and it has at least one contact. Objects labeled floating drop onto their support, objects labeled embedded are pushed out, and objects that still fail to settle are re-placed. The settling pass both corrects the initial scene and serves as a source of contact data that we later exploit to verify the support graph.

![](images/7dbe1f935d0178b863286ab0a68b9f0e6bf78aa8a809b0f59fa599339c45559a.jpg)  
Fig. 4. Schematic comparison of default (top) and SimReady (bottom) scene exports. A single convex hull fills concave spaces, creating phantom support and embedding the box. CoACD colliders preserve these spaces, allowing the sphere to rest on the seat after settling and the box to fit beneath the table. Dashed outlines denote collision geometry; green dots mark contact points.

## 3.1.5 Physical Constraints

Beyond geometry and attributes, a simulator-compatible scene must encode the constraints that make contact and motion physically meaningful. We instantiate three families of constraints in the target engine (Mu-JoCo). (i) Contact constraints: every collider is assigned a contype/conaffinity bitmask so that only intended pairs (e.g., object–floor, robot–object) generate contacts, and condim=3 for normal contact and tangential friction, or condim=4 when torsional friction about the contact normal is also required. (ii) Friction constraints: each contact geom carries friction coefficients that, together with condim and the solver’s friction cone, bound the enabled tangential forces and frictional torques. (iii) Dynamics constraints: unattached rigid bodies use free joints, while hinge and slide joints carry their own axis, range limits, and—for articulated objects—stiffness and damping, so a drawer opens only along its rail and a robot joint respects its limit. The integrator, solver, and gravity are configured once per scene, and the settling pass above runs under exactly these constraints, so the exported MJCF is the same representation that is validated. These constraints complement the collision geometry illustrated in Fig. 4.

Module 1 emits two scene variants: scene\_default, which uses visual meshes as colliders exactly as MuJoCo would by default, and scene\_enriched, which applies the full collision plan (convex decomposition, terrain grid, physical attributes). The gap between the two is our primary measure of physical plausibility (Sec. 5).

## 3.2 Module 2: Simulator-in-the-Loop 4D Refiner

Module 2 closes the loop: it converts the Post-Process Agent’s visual dynamics into simulator-executable dynamics and feeds measured physics back so the physical settings—not only the code—are revised. Motion is partitioned into two layers: (i) simulator state—rigid-body and articulated motion—executed by MuJoCo; and (ii) visual effects—rain, fog, fire, relighting—which remain Blenderlevel dynamics. The Post-Process Agent therefore emits a dynamics\_program.json with three block types—rigidbody (pose, velocity, attributes), joint (definition and trajectory), and visual-effect (free-form descriptor for Blender).

For each dynamic body we resolve mass, friction, and inertia from the physical attributes of Module 1 (or from a material lookup for unambiguously named objects), snap it above the ground or floor surface by ray-casting, and emit a MuJoCo body with a free joint; articulated objects are emitted with a slide or hinge joint driven by a time-parameterized trajectory. A robot (Unitree Go2, UR5e, Clearpath Jackal, or the service robots’ differential-drive base) is placed relative to the target according to the action— a kick anchors the foot reach, a walk provides a runway, and a watch observes from a fixed offset—and controlled by PD tracking to kick, walk, watch, or weld. Per frame we log the object pose, speeds, contact count, and peak contact force for the critic below.

## 3.2.1 Simulator-in-the-Loop Critic

We replace the original visual-only VLM-Motion Critic with a two-part critic. The physics critic computes seven simulator signals over the rollout log, one per validation objective: penetration depth (maximum sink below the initial support), collision pairs (time-averaged and peak contact count), post-strike stability (mean settle-window speed, which must decay to near zero), contact-force peak (peak normal force during the strike), support relation at rest (final pose on the ground and at rest), joint-limit/uprightness (limit violations or base tip-over), and interaction success (whether the action displaced the object). The visual critic asks a VLM whether sampled rollout frames look plausible, returning a validity flag and free-text issues. The two are fused into a single verdict: a scene is accepted only if every physics signal passes the checks above and the VLM judges the sampled frames valid. The fused feedback is returned to the Post-Process Agent, which revises the dynamics program and the physical settings that the critic flagged, so the refinement evolves the physics rather than the script alone.

## 4 SIMCODE4D BENCHMARK

## 4.1 Evaluation Target and Frozen Cases

We introduce SIMCODE4D, a state-based benchmark that evaluates executable world packages, rather than rendered videos, for simulator execution, physical admissibility, and embodied task completion. Five templates—pickand-place, push-to-region, place-into-container, obstacleavoidance navigation, and dynamic interaction—use typed initial, event, always, and terminal predicates. The frozen suite instantiates three task families: navigation, dynamic interaction, and pick-and-transport, a pick-and-place variant with target-region transfer.

We instantiate the benchmark with a frozen 15-case development suite, hereafter DEV15. It contains three world– task cases in each of five domains: indoor, industrial, natural outdoor, service, and urban. Its seven navigation, two pick-and-transport, and six dynamic-interaction cases cover cargo retention, welding, contact-before-motion, water-toland transition, drawer interaction, and contact-mediated blocking.

![](images/f79e443fc0395fe6688f14b7ac7cc57b7df232ba3d9928800a0ea6f0365513fe.jpg)  
Fig. 5. The stages of Module 1. Collision geometry is obtained by convex decomposition; density, friction, and restitution are estimated by a vision– language model, while mass and inertia are computed from mesh geometry; a support graph separates supported from floating bodies; and the engine’s contact model, together with a gravity settling pass, produces a simulator-compatible scene exported as MJCF.

Each case specifies a goal, world identifier, robot and initial state, required entities, evaluator-owned predicates, contact permissions, observations, timeout, and terminal tolerances. Predicates read MuJoCo state, contact, control, and event logs. Four machinereadable contracts—case\_spec.json, case\_gold.json, physics\_profile.json, and task\_confirm.yaml— are paired with an independent probe and validator. Twelve cases retain authoring scenes; all 15 retain compiled M1 worlds and source pointers.

## 4.2 Coding Evaluation

For the fixed-source track, a method receives the same frozen source bundle and must emit (i) a world package containing MJCF, local assets, physical fields, collision geometry, articulations, and persistent entity bindings, and (ii) a typed task-dynamics overlay. The evaluator owns the robot probe, so a submitted method cannot encode the evaluation policy in its world program.

For a first-attempt world package w, the strict build gate is

$$
B _ { w } = P _ { w } \wedge S _ { w } \wedge A _ { w } \wedge E _ { w } \wedge L _ { w } \wedge F _ { w } \wedge I _ { w } ,\tag{2}
$$

where P denotes syntax parsing, S strict schema and crossreference validity, A closure and consistency of local assets, E coverage of required entities, relations, physical attributes, collision geometry, articulations, and observations, L simulator compilation and loading, F finite smoke execution, and I intervention integrity. The primary method-level coding metric is

$$
\mathrm { W o r l d B u i l d P a s s } \ @ 1 = \frac { 1 } { N _ { W } } \sum _ { w = 1 } ^ { N _ { W } } \mathbf { 1 } \{ B _ { w } \} .\tag{3}
$$

Beyond WorldBuildPass@1, we report three first-output coding diagnostics. TaskOverlayPass@1 measures the fraction of packages whose typed overlay passes schema checks, refers to the compiled world, and declares all registered dynamic bodies. Required-entity binding and initial support-relation micro-F1 are computed against registered source bindings and human-confirmed task-object support labels, respectively. For each measure, micro-F1 pools truepositive (TP), false-positive (FP), and false-negative (FN) counts across eligible first outputs; unannotated background relations are excluded. Physical-attribute, collision, and articulation checks are part of the build gate rather than separate coverage rates. Runtime exceptions count as build or execution failures, and feedback calls are reported for repaired outputs. We omit BLEU and CodeBLEU because equivalent world programs need not share a reference string.

The DEV15 preflight checks frozen reference packages, not generated first outputs. It verifies registered files, contract bindings, M1 and runtime loading, M1–M2 binding, overlays, contact groups, observation and actuator names, and probe integrity. Probes cannot import the simulator or modify state, applied forces, equality constraints, mass or inertia, friction, or collision masks. These checks establish evaluator readiness.

## 4.3 Physical, Kinematic, and Task Evaluation

All DEV15 runtime models use gravity $( 0 , 0 , - 9 . 8 1 ) \mathrm { { m s ^ { - 2 } } }$ a 1 ms step, and MuJoCo’s implicitfast integrator. The physical profile retains body mass and inertia, joint type/range/damping, actuator gear and control limits, collision masks, semantic contact groups, and sliding, torsional, and rolling friction. Evaluation separates four complementary components.

Mechanics: We require finite dynamics and a forward/inverse generalized-force residual no larger than $1 0 ^ { - 5 }$ . Contact force, kinetic and potential energy, and the constraint-force residual remain visible diagnostics. The latter is not used as a substitute for generalized-force balance because its normalization is solver dependent. Undeclared equality constraints, forbidden contacts, and action-bound violations are hard failures.

Kinematics: We check finite pose and twist, quaternion norm error below $1 0 ^ { - 6 }$ , joint range and speed, and finite-difference linear/angular acceleration and jerk. Casespecific predicates further constrain path corridors, pose, tilt, event order, goal dwell, and terminal speed.

Contact and friction: For every resolved contact, the evaluator records penetration, normal and tangential force, tangential slip speed, and contact-dimension-aware Coulomb utilization using the declared sliding, torsional, and rolling coefficients. A friction-cone failure requires utilization above 1.25 continuously for at least 10 ms. Zerocontact episodes remain in the denominator of frictioncontact coverage rather than being imputed.

Robust task success: For episode e, the end-to-end acceptance criterion is

$$
Y _ { e } = B _ { w ( e ) } \wedge D _ { e } \wedge M _ { e } \wedge K _ { e } \wedge T _ { e } \wedge S _ { e } \wedge I _ { e } ,\tag{4}
$$

![](images/6de5308801cd6a8cf4f4a63f1a3a78309947e1360d51d64368e920829ad87ddd.jpg)

![](images/85a066af0879a671f0ff1b51584ccad201d059e8e6cce3229ac19f17aa0abc99.jpg)

![](images/4e1c4ada6e4557d1f9d6c57751454eedd3db892fed0ed0c1df1dbdb1b1a1b7ba.jpg)  
Fig. 6. DEV15 benchmark composition and execution diagnostics. (a) The 15 cases span five domains and three realized task families (N: navigation, D: dynamic interaction, P: pick/transport). (b) All 15 cases have complete executable contracts; 12 additionally retain their authoring scenes. (c) For each domain, bars show the 95th percentile of per-episode penetration P95 values, whereas diamonds show the maximum penetration observed in any episode; both remain below the frozen 1 mm gate, and lower is better. (d) Maximum penetration, Coulomb utilization, generalized-force residual, and quaternion error are normalized by their failure thresholds; lower is better, and all values remain below one.

where $D _ { e }$ denotes overlay validity, $M _ { e }$ mechanics/contact admissibility, $K _ { e }$ kinematic consistency, $T _ { e }$ the complete task predicate, $S _ { e }$ safety, and $I _ { e }$ integrity. The primary episode metric is

$$
\mathrm { R o b u s t T a s k S u c c e s s } = \frac { 1 } { N _ { E } } \sum _ { e = 1 } ^ { N _ { E } } { \bf 1 } \{ Y _ { e } \} .\tag{5}
$$

The seven factors in $Y _ { e }$ are grouped only for presentation: Task denotes $T _ { e } ,$ while Physical denotes $\dot { M _ { e } } \wedge \dot { K _ { e } } \wedge S _ { e }$ . Build and task-overlay validity remain $B _ { w ( e ) }$ and $D _ { e } ,$ and $I _ { e }$ denotes integrity. Thus, the RTS reported in tables uses the complete acceptance gate above. Build, overlay, load, timeout, non-finite-state, safety, and integrity failures remain in the registered denominator as zeros. The five fixed evaluation physics seeds (401, 409, 419, 421, and 431) perturb casedeclared movable-body position, yaw, and friction; mass is additionally perturbed when a task contains an independently movable object. We report per-predicate success, all-five-seeds pass, forbidden-contact and joint-limit rates, event-order accuracy, terminal margin, dynamic penetration, and friction-contact coverage.

## 4.4 Dev15 Benchmark Validation

We evaluate the frozen DEV15 packages with Python 3.10.8, MuJoCo 3.9.0, and NumPy 1.26.3. The registered validation matrix contains 15 cases, three deterministic rebuild replicates, and five physics seeds, for 225 executions. Aggregation over unique case–physics pairs yields 75 unique trials, while the full matrix verifies rebuild determinism and artifact accounting. Table 1 reports implementation readiness and execution results with explicit denominators.

Artifact coverage and physical diagnostics are summarized in Fig. $6 ;$ panels (c) and (d) report lower-is-better physical self-consistency measures.

All reference packages pass build and overlay preflight, and all 225 executions pass mechanics, kinematics, task, safety, and integrity checks. Maximum penetration and generalized-force residual are 0.921837 mm and 2.03424 × $\breve { 1 } 0 ^ { - 9 }$ , respectively, both below their gates. Fourteen cases produce friction-bearing contact; the articulated weld-path case is intentionally non-contact. Event order passes all 210 eligible executions. These results establish benchmark readiness under the declared physical profiles.

## 5 EXPERIMENTS

## 5.1 Setup

We distinguish benchmark validation from method evaluation. The frozen DEV15 suite contains 15 cases across five domains and three task families. Its reference packages are tested with the five evaluation physics seeds defined in Sec. $^ { 4 , }$ yielding 75 unique case–physics trials. Three deterministic rebuild replicates produce 225 benchmarkvalidation executions; neither count is a generated-method score.

Our source-conditioned experiment uses the 12 cases with verified authoring sources. Three generation seeds (1103, 2207, and 3301) produce 36 world–task packages; these seeds do not index physics perturbations. We compare default, physics-aware, and repaired outputs under the five evaluation physics seeds, held out from repair, yielding 180 registered rollouts per condition. Repair observes only the separate feedback physics seed 503 and permits at most three execution repairs and two state refinements. The heldout seeds perturb movable-body pose, yaw, friction, and mass where applicable.

The matched-prompt comparison uses three prespecified indoor tasks: living-room pushing, dining-room transport, and kitchen pushing. Code2Worlds and SceneSmith each generate one world per prompt. Code2Worlds is scored through a uniform MuJoCo adapter and SceneSmith through its official export. Code2Worlds-2 (Ours) applies M1/M2 to the same final Code2Worlds visual output; it is a paired extension, not another independent text-to-scene generation. Five evaluation physics seeds yield 15 rollouts per condition with common probes and source-relative task predicates. This cohort remains separate from the fixedsource experiment.

Build is scored per package; Task, Physical, and RTS per rollout. For fixed-source rows, Build scores first outputs except in the repaired row, which scores the selected postrepair package. Repairs denotes mean feedback calls per package; a dash marks a non-comparable native pipeline. RTS follows the complete seven-factor gate in Sec. 4, with failed builds and executions retained in the denominators. Evaluation physics seeds retest a world, not independent generations. Scores use headless MuJoCo 3.9.0; Blender EEVEE serves qualitative visualization only. Fixed-source runs use a 1 ms step, implicitfast, the Newton solver, and gravity $( 0 , 0 , - 9 . 8 1 ) \mathrm { m } \mathrm { s } ^ { - 2 } ;$ native exports retain their own simulator options.

TABLE 1  
Implementation and execution results on the frozen SIMCODE4D-DEV15-RC1 suite. The registered matrix comprises 225 executions over 75 unique case–physics pairs.
<table><tr><td>Measure</td><td>Result</td><td>Evidence and acceptance criterion</td></tr><tr><td>Artifact readiness</td></tr><tr><td>Registered cases with complete contract quartets</td><td>15/15</td><td>Three cases per domain; schema and cross-references are valid</td></tr><tr><td>Archived and verified authoring</td><td>12/15</td><td>Compiled M1 worlds and source pointers are retained for all 15 cases</td></tr><tr><td>source scenes Independently loadable M1 worlds</td><td>15/15</td><td>All M1 world endpoints load under the frozen MuJoCo build</td></tr><tr><td>Executable runtime packages</td><td>15/15</td><td>All task-runtime endpoints load and execute</td></tr><tr><td>Case-specific probe-validator pairs</td><td>15/15</td><td>Unique hash-bound paths are frozen for every case</td></tr><tr><td>Complete physical profiles</td><td>15/15</td><td>Mass and inertia; joints and actuation; collision and contacts; friction and solver settings</td></tr><tr><td>Benchmark execution</td></tr><tr><td>Completed executions</td><td>225/225</td><td>Zero runtime errors; 75 unique case-physics trials</td></tr><tr><td>RobustTaskSuccess</td><td>225/225</td><td>All seven factors of  $Y _ { e }$  pass, including build and overlay</td></tr><tr><td>All five physics seeds pass</td><td>15/15</td><td>Every case passes every registered physics seed</td></tr><tr><td>Forbidden-contact episodes</td><td>0/225</td><td>Hard safety gate; zero violations required</td></tr><tr><td>Joint-limit episodes</td><td>0/225</td><td>Hard safety gate; zero violations required</td></tr><tr><td>Event-order accuracy</td><td>210/210</td><td>Ordered event graphs are defined for 210 executions</td></tr><tr><td>Friction-cone failures</td><td>0/225</td><td>Failure requires utilization &gt; 1.25 for at least 10 ms</td></tr><tr><td>Cases with friction-contact evidence</td><td>14/15</td><td> $1 0 , 1 6 0 , 6 0 4$  contact samples; the weld case is intentionally non-contact</td></tr><tr><td>Maximum penetration</td><td>0.922 mm</td><td>Acceptance gate:  $\leq 1$  mm</td></tr><tr><td>Maximum forward/inverse generalized-force residual</td><td> $2 . 0 3 \times 1 0 ^ { - 9 }$ </td><td>Acceptance gate:  $\leq 1 0 ^ { - 5 }$ </td></tr></table>

## 5.2 Generated-World Compatibility and Robust Success

Table 2 reports generated-world results under two separate protocols; the DEV15 reference-package checks are reported in Sec. 4. In the fixed-source experiment, physical acceptance increases from 75/180 for the default export to 93/180 after physical enrichment and 108/180 after repair. RTS increases from 45/180 to 50/180 and 60/180, respectively. Task success does not improve at every stage (63/180 to 62/180 after enrichment). The raw-to-repaired RTS gain is 8.3 percentage points; its source-case-clustered 95% bootstrap interval is [−11.1, 28.9] percentage points.

Under the three matched prompts, the uniform MuJoCo adapter of Code2Worlds passes 5/15 task trials but no physical trials. Applying M1/M2 to the same Code2Worlds visual outputs yields 15/15 for task, physical, and RTS acceptance; all three packages pass before feedback repair. SceneSmith’s official MuJoCo exports pass 13/15 physical trials and no task trials. These are descriptive outcomes for one generated world per prompt, not a population-level ranking.

For the 36 first generated fixed-source plans, TaskOverlayPass@1 is 36/36, and required-entity binding micro-F1 is 97.3% (108 TP, 3 FP, 3 FN). Initial task-object support-relation micro-F1 is 66.7% (15 TP, 15 FP, 0 FN), evaluated only on five annotated cases across three generation seeds. These coding diagnostics measure overlay and semantic bindings, not simulator-state task success.

The physical gate applies the frozen mechanics, kinematics, collision, and friction checks in Sec. 4, including the

1 mm penetration limit and sustained friction-cone criterion. Fig. 4 illustrates the geometric mechanism: a single convex hull can occlude an opening that decomposed colliders preserve. All quantitative results come from simulator-state rollouts rather than the illustration.

## 5.3 Embodied Task Execution

Task outcomes are determined by frozen simulator-state predicates, not by visual inspection. Navigation requires compliant, collision-free goal arrival and stopping; pickand-transport requires a legal grasp, transfer, and terminal placement; dynamic-interaction tasks require their casespecific ordered events.

Table 3 breaks down the 12-case fixed-source results by task family. After repair, the task predicates pass in 35/60 navigation, 15/30 pick-and-transport, and 19/90 dynamicinteraction trials; the corresponding RTS counts are 35/60, 14/30, and 11/90. The aggregate RTS gain is therefore not uniform: navigation decreases from 45/60 for the default export to 35/60, whereas pick-and-transport and dynamic interaction increase from zero to 14/30 and 11/90, respectively. The matched-prompt methods are reported separately in Table 2 and are not pooled into these task-family totals.

Fig. 7 shows temporal progression in five frozen DEV15 reference-package rollouts. These frames illustrate the task definitions; they are neither method-generated examples nor inputs to the state-based scores.

## 5.4 Ablations

We evaluate three physical-preparation interventions and four refinement feedback rules. Within each paired comparison, the task specification, robot policy, simulator settings, validator, and perturbation seeds remain fixed. Eligibility differs across interventions; their denominators are therefore reported separately rather than pooled. Table 4 reports the Module 1 interventions, and Table 5 compares Module 2 feedback rules from the same 36 enriched starting worlds.

TABLE 2  
Generated-world acceptance under two non-pooled protocols.
<table><tr><td>Method  / stage</td><td>Build</td><td>Task</td><td>Physical</td><td>RTS</td><td>Repairs</td></tr><tr><td colspan="6">Fixed-source progression</td></tr><tr><td>Default export</td><td>33/36</td><td>63/180</td><td>75/180</td><td>45/180</td><td>0</td></tr><tr><td>+ SimReady scene compilation</td><td>33/36</td><td>62/180</td><td>93/180</td><td>50/180</td><td>0</td></tr><tr><td>+ SimReady scene compilation + feedback repair</td><td>34/36</td><td>69/180</td><td>108/180</td><td>60/180</td><td>1.61</td></tr><tr><td colspan="6">Matched-prompt comparison</td></tr><tr><td>Code2Worlds [1]</td><td>3/3</td><td>5/15</td><td>0/15</td><td>0/15</td><td>一</td></tr><tr><td>Code2Worlds-2 (Ours)</td><td>3/3</td><td>15/15</td><td>15/15</td><td>15/15</td><td>0</td></tr><tr><td>SceneSmith [11]</td><td>3/3</td><td>0/15</td><td>13/15</td><td>0/15</td><td>一</td></tr></table>

![](images/adff8f46afd809ecccc758c62ca9deeb67eefead450420565f34073dc49c78e2.jpg)  
Fig. 7. Representative task rollouts from the five evaluation domains. Rows show one case per domain, and columns show six temporally ordered keyframes.

TABLE 3  
Task-family RTS for the 12 fixed-source cases. Each case contributes three generated packages and five evaluation physics seeds.
<table><tr><td>Task family</td><td>Cases</td><td>Default</td><td>Enriched</td><td>Repaired</td></tr><tr><td>Navigation</td><td>4</td><td>45/60</td><td>35/60</td><td>35/60</td></tr><tr><td>Pick/transport</td><td>2</td><td>0/30</td><td>14/30</td><td>14/30</td></tr><tr><td>Dynamic</td><td>6</td><td>0/90</td><td>1/90</td><td>11/90</td></tr></table>

Physical attributes: Five eligible packages are evaluated under 15 paired physics perturbations each. The ablation replaces task-object mass, diagonal inertia, and sliding, torsional, and rolling friction with uniform defaults; robot hardware and control remain unchanged. Object-specific attributes yield 69/75 RTS and 69/75 physical passes, compared with 54/75 and 55/75 under uniform defaults. For RTS, 19 trial pairs favor object-specific attributes, four favor the defaults, and 52 tie. Much of the difference occurs in medicine transport, where the defaults cause joint-limit violations. This intervention tests sensitivity to assigned properties, not the accuracy of VLM estimates.

TABLE 4  
Module 1 ablations on paired eligible inputs. The first two interventions report held-out physics trials; pose stabilization reports initial object states. Higher is better for every measure.
<table><tr><td>Ablated variant</td><td>Measure</td><td>Full</td><td>Ablated</td></tr><tr><td>Uniform attributes</td><td>RTS</td><td>69/75</td><td>54/75</td></tr><tr><td rowspan="3">Single convex hull</td><td>Physical pass</td><td>69/75</td><td>55/75</td></tr><tr><td>RTS</td><td>15/15</td><td>10/15</td></tr><tr><td>Physical pass</td><td>15/15</td><td>11/15</td></tr><tr><td rowspan="2">No pose stabilization</td><td>Stable</td><td>8/12</td><td>1/12</td></tr><tr><td>Penetration-safe</td><td>11/12</td><td>0/12</td></tr></table>

Collision decomposition: The collision ablation covers one stone and two cups, each under five paired physics seeds. Replacing their CoACD [54] colliders with a single convex hull, while retaining other physical parameters, reduces RTS from 15/15 to 10/15 and physical acceptance from 15/15 to 11/15. All five RTS losses occur in the cup-pushing task; the other two tasks retain success. The observed benefit is therefore task-dependent.

TABLE 5  
Module 2 feedback ablation from 36 matched enriched starts. Build is measured over 36 worlds; task, physical, and RTS outcomes are measured over 180 held-out physics trials. Failed builds remain in the denominator.
<table><tr><td>Feedback rule</td><td>Build</td><td>Task</td><td>Physical</td><td>RTS</td></tr><tr><td>No repair</td><td>33/36</td><td>62/180</td><td>93/180</td><td>50/180</td></tr><tr><td>Visua only</td><td>36/36</td><td>84/180</td><td>112/180</td><td>67/180</td></tr><tr><td>State only</td><td>36/36</td><td>96/180</td><td>116/180</td><td>81/180</td></tr><tr><td>Fused, fixed budget</td><td>35/36</td><td>81/180</td><td>99/180</td><td>62/180</td></tr></table>

Pose stabilization: Four objects are tested from three initial offsets each. The full preprocessing step applies bounded vertical separation followed by a 5-s gravity pass; the ablation omits both operations. A separate 5-s validation increases the stable-rest count from 1/12 to 8/12 and the count satisfying the 1-mm penetration gate from 0/12 to 11/12. These are object-level rest-state diagnostics, not tasksuccess rates. A contact audit also shows that a stable rest pose does not necessarily preserve the intended support relation. Because separation and settling are removed together, this experiment does not isolate either operation or the support graph individually.

Critic feedback: Table 5 compares no repair, visualonly feedback, simulator-state-only feedback, and their fusion under matched limits of three execution repairs and two state refinements. Feedback uses physics seed 503; final outcomes use the five evaluation physics seeds per world. The state-only critic includes task predicates as well as physical diagnostics and should not be described as physicsonly. Under this fixed-budget protocol, state-only feedback achieves the highest RTS (81/180). Visual-only feedback reaches 67/180, fusion reaches 62/180, and no repair reaches 50/180. In 23/36 fused runs, compile-invalid proposals exhaust the execution-repair budget. This fixed-budget fusion experiment does not use the main experiment’s early-accept rule, so its 62/180 result is distinct from the repaired result in the main table.

A separate blinded acceptance diagnostic contains 36 candidates: 19 satisfy the registered simulator predicates and 17 do not. Visual-only judgment accepts 3/17 invalid candidates, whereas adding the simulator-state gate accepts 0/17; both rules accept 6/19 valid candidates. Zero invalid acceptance follows from the gate’s definition. This diagnostic measures candidate filtering, not closed-loop repair quality.

Finally, 34 matched initial–final repair trajectories improve from 50/170 to 60/170 RTS trials: four worlds improve, one regresses, and 29 are unchanged. Intermediate rounds have different eligible cohorts; these endpoint results do not imply monotonic improvement at every round.

## 6 CONCLUSION

We presented Code2Worlds-2, an extension of Code2Worlds that turns 4D world generation into embodied world simulation by appending two modules to the original pipeline and orchestrating them with an agent harness: a Sim-Ready Scene Compiler that endows generated scenes with collision geometry, physical attributes, a support graph, and physical constraints, and applies a gravity settling pass, and a Simulator-in-the-Loop 4D Refiner that closes a compile–simulate–critique–refine loop to compile visual dynamics into executable rigid and articulated dynamics and validate them through simulator signals rather than pixels alone. We further proposed SimCode4D, a statebased benchmark with five task templates for evaluating embodied simulation. Across the fixed-source cases and the matched prompts, state-based evaluation shows that physical acceptance rises from 75/180 to 108/180 and RTS from 45/180 to 60/180 after enrichment and repair. Our framework currently covers a limited set of articulated objects and leaves high-fidelity indoor wall collision and general grasping as future work; extending the compiler to Isaac Sim/USD export and a larger manipulation suite are natural next steps.

## ACKNOWLEDGMENTS

This work is supported by the Fundamental Research Funds for the Central Universities, Peking University.

## REFERENCES

[1] Y. Zhang, Y. Wang, Z. Zhang, and H. Tang, “Code2Worlds: Empowering Coding LLMs for 4D World Generation,” in Proceedings of the International Conference on Machine Learning (ICML), 2026.

[2] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012, pp. 5026–5033.

[3] NVIDIA, “Isaac Sim,” 2025. [Online]. Available: https:// developer.nvidia.com/isaac-sim

[4] Z. Hu, A. Iscen, A. Jain, T. Kipf, Y. Yue, D. A. Ross, C. Schmid, and A. Fathi, “SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, 2024, pp. 19 252–19 282.

[5] W. Feng, W. Zhu, T. jui Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang, “LayoutGPT: Compositional Visual Planning and Generation with Large Language Models,” in Advances in Neural Information Processing Systems (NeurIPS), 2023, pp. 18 225–18 250.

[6] C. Sun, J. Han, W. Deng, X. Wang, Z. Qin, and S. Gould, “3D-GPT: Procedural 3D Modeling with Large Language Models,” in International Conference on 3D Vision (3DV), 2025, pp. 1253–1263.

[7] Y. Du, S. Chen, W. Zan, P. Li, M. Wang, D. Song, B. Li, Y. Hu, and B. Wang, “BlenderLLM: Training Large Language Models for Computer-Aided Design with Self-improvement,” arXiv preprint arXiv:2412.14203, 2024.

[8] S. Lu, G. Chen, N. A. Dinh, I. Lang, A. Holtzman, and R. Hanocka, “LL3M: Large Language 3D Modelers,” arXiv preprint arXiv:2508.08228, 2025.

[9] B. Dai, L. R. Luo, Q. Tang, J. Wang, X. Lian, H. Xu, M. Qin, X. Xu, B. Dai, H. Wang, Z. Lyu, and J. Pang, “MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.

[10] P. Wang, Y. Wang, L. Li, Z. Yang, K. Q. Lin, Y. Li, and Y. Cheng, “SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects,” arXiv preprint arXiv:2605.19587, 2026.

[11] N. Pfaff, T. Cohn, S. Zakharov, R. Cory, and R. Tedrake, “Scene-Smith: Agentic Generation of Simulation-Ready Indoor Scenes,” in Proceedings of the International Conference on Machine Learning (ICML), 2026, Spotlight.

[12] R. Tedrake and the Drake Development Team, “Drake: Modelbased design and verification for robotics,” 2019. [Online]. Available: https://drake.mit.edu

[13] W. Zhong, P. Cao, Y. Jin, L. Luo, W. Cai, J. Lin, H. Wang, Z. Lyu, T. Wang, B. Dai, X. Xu, and J. Pang, “InternScenes: A Largescale Simulatable Indoor Scene Dataset with Realistic Layouts,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025.

[14] A. Raistrick, L. Lipson, Z. Ma, L. Mei, M. Wang, Y. Zuo, K. Kayan, H. Wen, B. Han, Y. Wang, A. Newell, H. Law, A. Goyal, K. Yang, and J. Deng, “Infinite photorealistic worlds using procedural generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 12 630–12 641.

[15] C. Xia, K. Zhu, Z. Wang, F. Liu, Z. Zhang, and Y. Duan, “Sim-Recon: SimReady Compositional Scene Reconstruction from Real Videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 42 452–42 463.

[16] Y. Yang, F.-Y. Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han, J. Wu, N. Haber, R. Krishna, L. Liu, C. Callison-Burch, M. Yatskar, A. Kembhavi, and C. Clark, “Holodeck: Language Guided Generation of 3D Embodied AI Environments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 277–16 287.

[17] Y. Yang, B. Jia, P. Zhi, and S. Huang, “PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 262–16 272.

[18] Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan, “RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, 2024, pp. 51 936–51 983.

[19] G. Lin, K. Huang, M. Liu, R. Gao, H. Chen, L. Chen, B. Lu, T. Komura, Y. Liu, J.-Y. Zhu, and M. Li, “PAT3D: Physics-Augmented Text-to-3D Scene Generation,” in International Conference on Learning Representations (ICLR), 2026.

[20] Z. Chen, A. Walsman, M. Memmel, K. Mo, A. Fang, D. Fox, and A. Gupta, “URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images,” in Robotics: Science and Systems (RSS), 2024.

[21] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A Universe of Annotated 3D Objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 142–13 153.

[22] A. Raistrick, L. Mei, K. Kayan, D. Yan, Y. Zuo, B. Han, H. Wen, M. Parakh, S. Alexandropoulos, L. Lipson, Z. Ma, and J. Deng, “Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 783–21 794.

[23] S. Zhang, M. Zhou, Y. Wang, C. Luo, R. Wang, Y. Li, Z. Zhang, and J. Peng, “CityX: Controllable Procedural Content Generation for Unbounded 3D Cities,” arXiv preprint arXiv:2407.17572, 2024.

[24] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, “Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[25] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbr ˜ ugg ¨ et al., “Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning,” arXiv preprint arXiv:2511.04831, 2025.

[26] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem, “Brax – A Differentiable Physics Engine for Large Scale Rigid Body Simulation,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[27] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, “Sapien: A simulated part-based interactive environment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 094–11 104.

[28] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “RLBench: The Robot Learning Benchmark & Learning Environment,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3019–3026, 2020.

[29] J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. Huang, R. Chen, and H. Su, “ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills,” in International Conference on Learning Representations (ICLR), 2023.

[30] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ın-Mart´ın, A. Joshi, K. Lin, A. Maddukuri, S. Nasiriany, and Y. Zhu, “robosuite: A Modular Simulation Framework and Benchmark for Robot Learning,” arXiv preprint arXiv:2009.12293, 2020.

[31] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-World: A Benchmark and Evaluation for Multi-

Task and Meta Reinforcement Learning,” in Conference on Robot Learning (CoRL), 2020, pp. 1094–1100.

[32] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7327–7334, 2022.

[33] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martin-Martin, C. Wang, G. Levine, W. Ai, B. Martinez, H. Yin, M. Lingelbach, M. Hwang, A. Hiranaka, S. Garlanka, A. Aydin, S. Lee, J. Sun, M. Anvari, M. Sharma, D. Bansal, S. Hunter, K.-Y. Kim, A. Lou, C. R. Matthews, I. Villa-Renteria, J. H. Tang, C. Tang, F. Xia, Y. Li, S. Savarese, H. Gweon, C. K. Liu, J. Wu, and L. Fei-Fei, “BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation,” arXiv preprint arXiv:2403.09227, 2024.

[34] E. Kolve, R. Mottaghi, W. Han, E. VanderBilt, L. Weihs, A. Herrasti, M. Deitke, K. Ehsani, D. Gordon, Y. Zhu, A. Kembhavi, A. Gupta, and A. Farhadi, “AI2-THOR: An Interactive 3D Environment for Visual AI,” arXiv preprint arXiv:1712.05474, 2017.

[35] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra, “Habitat: A Platform for Embodied AI Research,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9338–9346.

[36] M. Deitke, W. Han, A. Herrasti, A. Kembhavi, E. Kolve, R. Mottaghi, J. Salvador, D. Schwenk, E. VanderBilt, M. Wallingford, L. Weihs, M. Yatskar, and A. Farhadi, “RoboTHOR: An Open Simulation-to-Real Embodied AI Platform,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3161–3171.

[37] M. Deitke, E. VanderBilt, A. Herrasti, L. Weihs, J. Salvador, K. Ehsani, W. Han, E. Kolve, A. Farhadi, A. Kembhavi, and R. Mottaghi, “ProcTHOR: Large-Scale Embodied AI Using Procedural Generation,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[38] F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese, “Gibson Env: Real-World Perception for Embodied Agents,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 9068–9079.

[39] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra, “Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

[40] J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, Y. Aytar, S. Bechtle, F. Behbahani, S. Chan, N. Heess, L. Gonzalez, S. Osindero, S. Ozair, S. Reed, J. Zhang, K. Zolna, J. Clune, N. de Freitas, S. Singh, and T. Rocktaschel, “Genie: Generative¨ Interactive Environments,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, 2024, pp. 4603– 4623.

[41] S. Yang, Y. Du, K. Ghasemipour, J. Tompson, L. Kaelbling, D. Schuurmans, and P. Abbeel, “Learning Interactive Real-World Simulators,” in International Conference on Learning Representations (ICLR), 2024.

[42] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “GAIA-1: A Generative World Model for Autonomous Driving,” arXiv preprint arXiv:2309.17080, 2023.

[43] X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu, “World-Dreamer: Towards General World Models for Video Generation via Predicting Masked Tokens,” arXiv preprint arXiv:2401.09985, 2024.

[44] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-World: A Controllable Generative World Model for Robot Manipulation,” in International Conference on Learning Representations (ICLR), 2026.

[45] A. J. Zhai, Y. Shen, E. Y. Chen, G. X. Wang, X. Wang, S. Wang, K. Guan, and S. Wang, “Physical Property Understanding from Language-Embedded Feature Fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 28 296–28 305.

[46] H. Le, J. Kwon, E. Ismayilzada, Y. Zhang, and Z. Cui, “SiPhy: Single-Image Physical Property Reasoning,” in Proceedings of the European Conference on Computer Vision (ECCV), 2026, pp. 1–19.

[47] S. Lee, J. Jeong, Y. Hong, and K. I. Kim, “Physically Guided Visual Mass Estimation from a Single RGB Image,” in Proceedings of the

International Joint Conference on Artificial Intelligence (IJCAI), 2026, pp. 1297–1305.

[48] M. Comi, J. S. Berbel, K.-K. Maninis, P. Henzler, and M. Sanchez, “Gen2Physics: Grounding Generated 3D Meshes in Physics via Multi-View Material Decomposition,” arXiv preprint arXiv:2608.23869, 2026.

[49] I. Armeni, Z.-Y. He, J. Gwak, A. R. Zamir, M. Fischer, J. Malik, and S. Savarese, “3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 5663–5672.

[50] J. Wald, H. Dhamo, N. Navab, and F. Tombari, “Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3960–3969.

[51] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, “Scenegraphfusion: Incremental 3d scene graph prediction from rgb-d sequences,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 7511–7521.

[52] S. Koch, N. Vaskevicius, M. Colosi, P. Hermosilla, and T. Ropinski, “Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 183–14 193.

[53] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 5021–5028.

[54] X. Wei, M. Liu, Z. Ling, and H. Su, “Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search,” ACM Transactions on Graphics, vol. 41, no. 4, pp. 1–18, 2022, SIGGRAPH 2022.

[55] K. Mamou, “Volumetric Hierarchical Approximate Convex Decomposition,” in Game Engine Gems 3. A K Peters/CRC Press, 2016, pp. 141–158.

[56] B. Vu, S. Ganguly, and P. Joshi, “Empart: Interactive Convex Decomposition for Converting Meshes to Parts,” arXiv preprint arXiv:2509.22847, 2025.

[57] E. Fokin and M. Savva, “VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition,” in Eurographics 2026 - Short Papers, 2026.

[58] X. Xu, W. Ge, D. Qiu, Z. Chen, D. Yan, Z. Liu, H. Zhao, H. Zhao, S. Zhang, J. Liang, and Y.-C. Chen, “GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 7231–7240.

[59] Y. Shuai, R. Yu, Y. Chen, Z. Jiang, X. Song, N. Wang, J. Zheng, J. Ma, M. Yang, Z. Wang, W. Ding, and H. Zhao, “PUGS: Zeroshot Physical Understanding with Gaussian Splatting,” in IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 4478–4485.

![](images/29189d2390d049a5f3d4af079a768f5b4e9c285a54d1a5d92853d75ed07138b0.jpg)

![](images/3eda4a1f574e4ac080dadb4a75ebab37b7915dc6ed28504bd9d8c67047c7cea4.jpg)

![](images/919d139b9d66a1c282c384f11b32a453d6b64f05ab1375095039f861c17e5550.jpg)

![](images/9daa080b7cf36b885c262463bc631f416a4f547627db53bfd3cba551c18bdd3c.jpg)

Siming Li is currently pursuing his undergraduate degree in the School of Telecommunications Engineering at Xidian University, Xi’an, China. His research interests focus on visual scene parsing and segmentation, embodied AI, physics-based interactive world modeling, and 3D generative tasks such as 3D Gaussian Splatting (3DGS).

Zeyu Zhang is a researcher working on generative AI, with a particular interest in building models that understand and interact with the physical world. He received his bachelor’s degree from the Australian National University, where he was advised by Prof. Richard Hartley and Prof. Ian Reid. His research explores generative modeling for learning physical dynamics from visual data. His work spans world models, multimodal foundation models, embodied AI, and AI for health.

Yi Zhang is an undergraduate student majoring in Software Engineering at Southwest University, Chongqing, China. His research interests include video generation, physically grounded world models, 3D/4D vision, multimodal large language models and agents, and physicsaware simulation.

Tat-Seng Chua received the Ph.D. degree from the University of Leeds, U.K. He is the KITHCT chair professor with the School of Computing, National University of Singapore, where he was the acting and founding dean of the School from 1998 to 2000. He is the co-director of NExT, a joint center between NUS and Tsinghua University, to develop technologies for live social media search. He is the 2015 winner of the prestigious ACM SIGMM Award.

![](images/6a0a5c84505b7942e9fbf0e29a8a0b68c973eb1cb959f22290603cbfaf96fab7.jpg)  
Haoyu Zhang is an undergraduate student at the School of Information and Software Engineering, University of Electronic Science and Technology of China, Chengdu, China, and is currently a visiting student at Peking University, Beijing, China. His research interests include remote-sensing image fusion (pansharpening), embodied world simulation, 3D/4D scene generation, and physics-based simulation.

![](images/b6580e49d7669ea86ebfc38677b8d25838e78c1274bc4c73250d18ee38aa24a3.jpg)

Hao Tang is an Assistant Professor at Peking University, China. Previously, he held postdoctoral positions at CMU, USA, and ETH Zurich,¨ Switzerland. He earned his master’s degree from Peking University, and his Ph.D. from the University of Trento, Italy. He has had the opportunity to visit the University of Oxford, NEU, NUS, and IIAI, among other institutions. His research focuses on generative AI, world models, spatial intelligence, embodied AI, LLMs, MLLMs, and computer vision.

This section collects full-resolution screenshots from the delivered runs of all fifteen cases. Each figure covers one domain: every row is one rollout, ordered top to bottom as in Table B.1, and the three columns show temporally ordered stages (early, middle, late) of that rollout. The frames are rendered with the pipeline’s Blender rendering path and are shown for qualitative inspection only.

![](images/0a31655d41a7aae7b348b54f6a262aaa3075f60ab9eee0116da5aa49064dcfe1.jpg)

Fig. A.1. Indoor (obstacle navigation). Rows: the mobile vacuum driving through the generated living room, the Unitree Go2 traversing the living room, and the mobile vacuum crossing the west corridor. The corridor, rug, sofa, shelving, and sideboard are produced by the scene-coding stage, and each robot is executed in the compiled world.  
![](images/3028ab64d395d5049af5d8a08d60d07735f98e95b9c309aed927533f8ca9f414.jpg)  
Fig. A.2. Industrial. Rows: the four-wheel AGV holding the aisle lane, the AGV transporting its welded cargo, and the UR5e tracing the frozen weld seam on the slotted workstation. Each rollout is executed in the compiled world; the weld tool path is an articulated trajectory with joint limits and contact events validated by the evaluator.

![](images/4ad98a78e77a10d481bdc7f449777046b37680471c1977fddc8d7ed2ff907343.jpg)  
Fig. A.3. Natural outdoor. Rows: the Unitree Go2 disturbing the source-bound rock on the generated canyon terrain, the Clearpath Jackal driving the authored route across the hills, and the amphibious rover moving from the water field onto dry land. The non-convex height field is sliced into per-cell convex prisms so that contact follows the local surface. The canyon, hill, and shore terrains are generated scene-code assets; the rock, the route obstacles, and the water field are frozen benchmark entities carrying their own physics-seed perturbations, and all three rollouts are rendered at 1080p from the same compiled worlds that the evaluator scores. The canyon wall, the route markers, and the shore bank are compiled colliders, so the robots’ contacts with them are resolved by the same collision pipeline as in the indoor and urban suites. The rows correspond to Natura outdoor-1 (rock kick), -2 (hills turn stop), and -3 (water land) of the frozen index; the rock’s terminal rest, the Jackal’s stop inside the goal region, and the rover’s arrival on dry land are the terminal predicates checked by the case validators. Frames are sampled uniformly across each rollout (early, middle, late).

![](images/b7c8cc50404b2f13276ffcaff2a5e1e71c77706bc7ee5b78b81cfa44a4a438c0.jpg)  
Fig. A.4. Service. Rows: the medicine robot delivering the pouch, the airport luggage robot lifting the blue luggage assembly, and the hotel robot opening its payload drawer so that the meal parcel stays supported. The clinic, airport, and hotel interiors are generated scene-code assets with their furniture layouts; the pouches, luggage, and meal parcel are dynamic bodies whose grasps are issued through the benchmark grasp API, and the grasp and terminal-state predicates are checked by the case validator. The rows correspond to Service-1 (medicine), -2 (airport luggage), and -3 (hotel drawer) of the frozen index; the drawer is opened through the benchmark actuation API, and the retained-grasp and 0.5-second support conditions are the terminal predicates. The three interiors stress the same grasp stack across different geometry: a counter hand-off, a floor-leve suitcase assembly, and a drawer payload.

![](images/dc2737ef78585ff34c8f49f226bd1e7979084cb8ba273f6b19d00e8d04c5a543.jpg)  
Fig. A.5. Urban. Rows: the four-wheel AGV driving along the city pavement, the Unitree Go2 crossing the road on the generated zebra crossing, and the Go2 making contact with the frozen target pole. The crossing and the pole are compiled collision surfaces, and the corridor constraint is enforced by the evaluator.

## APPENDIX B

## SIMCODE4D CASE SPECIFICATIONS

SIMCODE4D-DEV15-RC1 contains 15 world–task cases, three in each of five domains. The realized task families are obstacle navigation (7 cases), pick-and-transport (2 cases), and dynamic interaction (6 cases). Every case freezes four machine-readable contracts together with an independent probe and validator; Table B.1 lists the case index and Table B.2 summarizes the frozen artifacts. These fifteen frozen reference packages are distinct from the three matched-prompt case studies of Appendix E, which are experiment-specific inputs generated by the compared native pipelines rather than members of the frozen suite.

TABLE B.1: The frozen SIMCODE4D-DEV15-RC1 case index. Cases are listed per domain in the frozen suite order; each row gives the scene, the benchmark-owned robot, the realized task family, and the natural-language goal frozen in case\_spec.json.
<table><tr><td>Case</td><td>Domain</td><td>Robot</td><td>Task family</td><td>Goal (frozen prompt)</td></tr><tr><td>Indoor-1 vacuum living room</td><td>Indoor</td><td>mobile vacuum</td><td>Obstacle nav.</td><td>Drive the vacuum east through the living- room corridor, enter the frozen goal, and remain stopped for 0.5 seconds.</td></tr><tr><td>Indoor-2 go2 living room</td><td>Indoor</td><td>Unitree Go2 quadruped</td><td>Obstacle nav.</td><td>Traverse the living room east with the Go2, stay in the frozen corridor, enter the</td></tr><tr><td>Indoor-3 vacuum west</td><td>Indoor</td><td>mobile vacuum</td><td>Obstacle nav.</td><td>goal, and stop. Drive the vacuum west through the in- door corridor, enter the frozen goal, and</td></tr><tr><td>Industrial-1 agv aisle</td><td>Industrial</td><td>four-wheel AGV</td><td>Obstacle nav.</td><td>remain stopped for 0.5 seconds. Drive the four-wheel AGV east along the industrial aisle, hold the lane, enter the</td></tr><tr><td>Industrial-2 agv cargo</td><td>Industrial</td><td>four-wheel AGV with welded cargo</td><td>Dynamic interaction</td><td>goal, and stop. Transport the physically modeled cargo east with the AGV, retain the declared weld attachment, enter the cargo goal,</td></tr><tr><td>Industrial-3</td><td>Industrial</td><td>UR5e welding arm</td><td>Dynamic</td><td>and stop. Trace the frozen weld seam continuously</td></tr><tr><td>weld Natural outdoor-1</td><td>Natural outdoor</td><td>Unitree Go2 quadruped</td><td>interaction Dynamic</td><td>with the UR5e tool tip and finish at rest. Use the Go2 feet to make the source-</td></tr><tr><td>rock kick Natural outdoor-2</td><td></td><td></td><td>interaction</td><td>bound rock move physically in the un- even canyon, then let it settle. Drive the Jackal across the authored hills,</td></tr><tr><td>hills turn stop</td><td></td><td></td><td></td><td>complete the authored left turn, enter the goal region, and remain stopped for 0.5</td></tr><tr><td>Natural outdoor-3 water land</td><td></td><td>Natural outdoor four-wheel amphibious rover</td><td>Dynamic interaction</td><td>seconds. Move the amphibious rover from the de- clared static water field, contact the bank,</td></tr></table>

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
<table><tr><td>Case</td><td>Domain</td><td>Robot</td><td>Task family</td><td>Goal (frozen prompt)</td></tr><tr><td>Service-1 medicine</td><td>Service</td><td>differential-drive service robot with lift and arm</td><td>Pick &amp; transport</td><td>Approach, legally grasp, and deliver the medicine pouch to the authored delivery region while retaining a stable grasp.</td></tr><tr><td>Service-2 airport luggage</td><td>Service</td><td>differential-drive airport luggage robot with lift</td><td>Pick &amp; transport</td><td>Approach the blue luggage assembly, legally grasp and lift it, then deliver it to the authored airport goal while retaining a stable grasp.</td></tr><tr><td>Service-3 hotel drawer</td><td>Service</td><td>differential-drive hotel service robot with payload drawer</td><td>Dynamic interaction</td><td>Navigate the hotel robot to the terminal region, stop, open its payload drawer, and keep the meal parcel supported for 0.5</td></tr><tr><td>Urban-1 agv city</td><td>Urban</td><td>four-wheel AGV</td><td>Obstacle nav.</td><td>seconds. Drive the AGV east along the city pave- ment, stay in the frozen corridor, enter the</td></tr><tr><td>Urban-2 intersection</td><td>Urban</td><td>Unitree Go2 quadruped</td><td>Obstacle nav.</td><td>goal, and stop. Štart on the west sidewalk, cross the road east through the frozen corridor, reach the east sidewalk, and stop.</td></tr><tr><td>Urban-3 pole blocking</td><td>Urban</td><td>Unitree Go2 quadruped</td><td>Dynamic interaction</td><td>Approach the frozen target pole with the Go2, make physical contact, remain blocked without passing through it, and</td></tr></table>

<table><tr><td>Frozen artifact</td><td>Content and role</td></tr><tr><td>case_spec.json</td><td>Semantic entities and their required bindings, observation channels (bodies and joints), benchmark robot identity, simulator configuration, probe and validator references, and canonical-source pointers (schema simcode4d.rc3a-dev1.case-spec).</td></tr><tr><td>case_gold.json</td><td>Evaluator-owned typed predicates (initial, event, always, terminal) with terminal tolerances and the physics- seed perturbation definition.</td></tr><tr><td>physics_profile.json</td><td>Body mass and inertia, joint type/range/damping, actuator gears and control limits, collision masks, semantic contact groups, and sliding/torsional/rolling friction.</td></tr><tr><td>task_confirm.yaml</td><td>Author-confirmed robot identity and task statement, the canonical source . blend with byte size and SHA- 256, start state, and acceptance summary.</td></tr><tr><td>probe / validator</td><td>One probe script (robot actuation, grasp API, perturbations) and one independent validator per case, hash- bound inside case_spec. json.</td></tr></table>

TABLE B.2  
Contracts frozen for every case. All four files are present for each of the 15 cases; the probe and validator named in case\_spec.json are owned by the benchmark, so a submitted method cannot encode the evaluation policy in its world program.

## B.1 Case Contract Fields

The following table summarizes the fields frozen for every case, including the start state, interaction constraints, acceptance predicates, and perturbation definition summarized above.

<table><tr><td>Field</td><td>Content frozen in the contracts</td></tr><tr><td>Identity</td><td>Case, source world, scene domain, and the canonical source . blend with byte size and SHA-256.</td></tr><tr><td>Task</td><td>Benchmark-owned robot (identity, binding, mobility), the natural-language goal, and the realized task family.</td></tr><tr><td>Initial state</td><td>Free-body poses and joint values of the start state, the required entities and their bindings, and the observation channels (bodies and joints).</td></tr><tr><td>Interaction</td><td>Required event order, permitted contacts and semantic contact groups, and any required attachment (for example a welded cargo).</td></tr><tr><td>Acceptance</td><td>Typed predicates (initial, event, always, terminal) with terminal tolerances, plus the hard safety gates (forbidden contacts, joint limits).</td></tr><tr><td>Perturbation</td><td>The physics-seed perturbations applied to movable bodies (position, yaw, friction; mass where independently movable).</td></tr><tr><td>Probe and validator</td><td>The benchmark probe (actuation, grasp API, perturbations) and an independent validator, both hash-bound.</td></tr></table>

TABLE B.3  
Fields frozen per case. The same structure is instantiated for all 15 cases; the probe and validator are owned by the benchmark.

## B.2 Worked Example: SER-01-medicine

The following excerpt reproduces the key fields of one frozen case directly from its contracts (case\_spec.json and task\_confirm.yaml); the remaining cases follow the same structure.

"bodies": [   
"robot",   
"task\_item",   
"grasp\_frame"   
],   
"joints": [   
"wheel\_left\_joint",   
"wheel\_right\_joint",   
"lift\_joint",   
"arm\_joint"   
]   
},   
"robot": {   
"description": "differential-drive service robot with lift and arm",   
"model\_binding": "robot",   
"mobility": "articulated"   
},   
"required\_entities": [   
"service\_robot",   
"medicine\_pouch",   
"source\_counter"   
],   
"canonical\_source": "deliverables/final\_scene\_revision\_i04\_2026-08-16/blends/scene...",   
"probe\_path": "probes/ser01\_medicine.py",   
"validator\_path": "validators/ser01\_medicine.py"

## APPENDIX C

## PROMPT FORMATS AND EXAMPLES

This section reproduces, verbatim, the prompts used by the stages of the implemented pipeline, together with one real recorded example per stage. Values that are interpolated at runtime (object lists, geometry measurements, simulator signals) are shown as {placeholders}. The scene-coding stage inherits the Code2Worlds prompt unchanged; the physical stages below it are specific to Code2Worlds-2.

## C.1 Scene-Code Generation (Inherited from Code2Worlds)

The scene-coding stage uses the public Code2Worlds agent prompt template unchanged [1]: the model receives a reference Blender (Infigen) code context plus a parameter dictionary and must return an executable script. The template is reproduced here for completeness; the physical layer begins with attribute estimation below.

You are an expert Python Developer for Blender (Infinigen).   
Your task is to write a ROBUST, ERROR-FREE Python script by strictly following a Reference   
Code Context.   
### INPUTS:   
1. <sub>\*\*</sub>Code Context<sub>\*\*</sub>: A working example script containing imports, paths, and instantiation   
logic.   
2. <sub>\*\*</sub>Parameters<sub>\*\*</sub>: A dictionary of specific values to apply to the factory instance.   
### INSTRUCTIONS (Step-by-Step):   
1. <sub>\*\*</sub>Environment Setup (Copy-Paste)<sub>\*\*</sub>:   
- You MUST copy the ‘import‘ statements and ‘sys.path.append(...)‘ EXACTLY as they appea   
in the Code Context .   
- Do not change the path strings.   
2. <sub>\*\*</sub>Scene Cleaning<sub>\*\*</sub>:   
- Include ‘bpy.ops.object.select\_all(action="SELECT")‘ and ‘bpy.ops.object.delete()‘ to   
start fresh.   
3. <sub>\*\*</sub>Instantiation (Strict Mimicry)<sub>\*\*</sub>:   
- Look at how ‘{factory\_name}‘ is instantiated in the Code Context.   
- <sub>\*\*</sub>REQUIRED<sub>\*\*</sub>: Define a seed variable first (e.g., ‘factory\_seed = 12345‘).   
- <sub>\*\*</sub>REQUIRED<sub>\*\*</sub>: Pass the seed to the constructor:   
‘{factory\_name}(factory\_seed=factory\_seed, coarse=False)‘.   
- Do NOT use ‘factory = {factory\_name}(coarse=False)‘ (missing seed is forbidden).   
4. <sub>\*\*</sub>Parameter Assignment (Dependency Aware)<sub>\*\*</sub>:   
- Parse the <sub>\*\*</sub>Parameters<sub>\*\*</sub> dictionary: ‘{params\_str}‘.   
- Assign them explicitly: ‘factory.param\_name = value‘.   
- <sub>\*\*</sub>CRITICAL CHECK<sub>\*\*</sub>: If the Code Context shows that ‘param\_B‘ is calculated based on   
‘param\_A‘ (e.g., ‘handle\_radius = 0.3 <sub>\*</sub> depth‘), ensure you set ‘param\_A‘ (‘depth‘)   
BEFORE setting ‘param\_B‘.   
- Handle ‘scale‘ and ‘thickness‘ early, as other params might depend on them.

```markdown
5. <sub>**</sub>Execution & Saving<sub>**</sub>:
- Call ‘.create_asset()‘ and assign the result to ‘obj‘.
- Save to ‘r"{FINAL_BLEND_PATH}"‘ using ‘bpy.ops.wm.save_as_mainfile(...)‘.
6. <sub>**</sub>Final Review (Self-Correction)<sub>**</sub>:
- Verify: Did I include ‘sys.path.append‘?
- Verify: Did I set the seed?
- Verify: Are there any undefined variables?
### OUTPUT FORMAT:
Return ONLY the valid Python code. No markdown blocks, no explanations.
```

## C.2 Physical-Attribute Estimation

For each key object, four views (iso, front, side, top) of the rendered mesh are sent to a vision–language model together with the object’s longest world dimension. The model returns a material category and the physical properties used to compute mass and inertia (Fig. A.3 shows the kind of terrain objects this is applied to).

You are a physics-aware perception assistant. Below are 4 views (iso / front / side / top)   
of a single object rendered as a triangulated mesh from a synthetic 3D scene. The object’s   
longest world dimension is {longest\_dim\_m} m.   
The object’s source name is: {source\_name}   
Identify the object’s dominant material and physical properties. Pick the SINGLE BEST   
category from this generic list (not just rocks):   
rock\_granite, rock\_basalt, rock\_sandstone, rock\_limestone, rock\_generic   
wood\_hardwood, wood\_softwood, wood\_painted   
metal\_steel, metal\_aluminum, metal\_iron   
plastic\_hard, plastic\_soft, rubber   
ceramic, glass, fabric, leather   
concrete, brick   
generic   
Return ONLY a JSON object (no commentary, no markdown fences):   
{{   
"dominant\_material": "<one of the categories above>",   
"density\_kg\_m3": <number; realistic bulk density for that material>,   
"friction": <number 0.2..0.95, sliding friction vs another rigid object>,   
"restitution": <number 0.0..0.4, bounciness>,   
"confidence": <number 0..1>,   
"reasoning": "<one short sentence justifying the material choice>"   
}}   
Realistic density ranges (kg/mˆ3): rock 2200-2800, wood 350-900, steel 7800, aluminum 2700,   
plastic 900-1400, glass 2400-2600, ceramic 2300-2700, concrete 2200-2400, brick 1800-2000,   
fabric 200-400, rubber 1100-1500. Use the source name as a strong hint: "BoulderFactory"   
-> rock\_ ; "TableFactory"/"CabinetFactory"/"ShelfFactory" -> wood\_ ; "BedFactory" ->   
wood\_<sub>\*</sub> (frame); "BottleFactory" -> glass or plastic; "MugFactory"/"CupFactory" -> ceramic.

Recorded example (one object from the natural-domain scene, abridged):

```jsonl
{
"prop_concave_large_000_BoulderFactory(543568399)": {
"vlm": {
"dominant_material": "rock_generic",
"density_kg_m3": 2500.0,
"friction": 0.7,
"restitution": 0.1,
"confidence": 0.95,
"reasoning": "The object is a large, irregularly shaped boulder with a rough surface
texture, consi..."
},
"geometry": {
"volume_m3": 2.2556569711762715,
"mass_kg": 5639.142427940678,
"com_local": [
0.10170043094978182,
0.04503726010149974,
0.34024433897424955
... ( truncated in this excerpt )
```

- Output ONE valid JSON object inside a ‘‘‘json‘‘‘ fence and nothing else.   
- All numbers are SI units.

- duration: 3-5 s.

## C.3 Dynamics-Program Generation

The Post-Process Agent translates the requested dynamic event into a dynamics\_program.json. In the Kick-stone setting the dynamic object is a new independent stone that reuses a key object’s mesh and material as a template; the prompt states this and the physical bounds explicitly.

You are the PostProcess Agent for Code2Worlds-2 Embodied World Simulation.   
You translate a natural-language dynamic-event description into a STRICT JSON   
program that drives BOTH a MuJoCo physics rollout AND Blender visual rendering.   
# Scene context   
You are given:   
- terrain AABB   
- the list of KEY boulders (the only ones with VLM-grounded mass/friction).   
- HARD RULE: ALL boulders that appear in C2W’s manifest -- including every   
KEY boulder -- are PERMANENTLY FUSED into the terrain by Module 1. They   
NEVER become dynamic. You may only reference a KEY boulder’s NAME as a   
visual+geometry TEMPLATE for a brand-new, independent small stone.   
- A Unitree Go2 quadruped sits at origin (x=0, y=0, z=0.27) and walks   
forward along +X.

Output a JSON object with three top-level keys, each a list:

"simulator\_dynamics": NEW small kickable/pushable stones, independent   
of the static terrain. For each:   
{   
"object": "<unique name, e.g. ’kick\_stone\_0’>",   
"type": "rigid\_body",   
"source\_boulder": "<name of a KEY boulder whose CoACD mesh + material   
are reused as the visual+collision TEMPLATE. The   
static KEY boulder remains in the terrain; this   
is just a stylistic clone, scaled down.>",   
"size\_scale": <float 0.05-0.20: how much to shrink the source mesh.   
The KEY boulders are 2-3 m across, so 0.10-0.15   
gives ˜20-40 cm kickable rocks.>,   
"initial\_pose": {"pos": [x,y,z], "quat": [w,x,y,z]},   
"mass": <kg, 2.0 - 20.0 -- a stone Go2 can actually displace>,   
"friction": <mu, 0.5 - 0.9>,   
"initial\_velocity": [vx,vy,vz], "m/s, usually [0,0,0]",   
"initial\_angular\_velocity": [wx,wy,wz], "rad/s, usually [0,0,0]",   
"duration": <seconds of MuJoCo rollout>   
}

"articulated\_dynamics": joint-based dynamics (door, drawer). Empty for outdoor scenes.

"visual\_effects": Blender-only effects (rain, fog, dust). For each:   
{"effect": "...", "type": "particle|shader|...", ...free schema...}

- For "Go2 walks across rocky terrain and kicks a small stone aside":   
place 1-3 small stones (size\_scale 0.10-0.15, mass 5-15 kg) on Go2’s   
forward path at x = +0.4 .. +1.2 m, y in [-0.15, +0.15] (one of them   
near a foot strike). Stones at rest: initial\_velocity = [0,0,0],   
initial\_angular\_velocity = [0,0,0]. z = 0.05 + size\_scale <sub>\*</sub> 1.5   
(ground level + half stone height).

- source\_boulder names MUST come from the provided key\_boulders list.   
- NEVER write "release boulder\_XXX" or use a manifest boulder name as   
the dynamic object name. The dynamic object is ALWAYS new, e.g.   
"kick\_stone\_0", "kick\_stone\_1".

## Recorded example (dynamics\_program.json from a recorded outdoor run):

{   
"robot\_model": "go2",   
"robot\_action": "walk",   
"simulator\_dynamics": [   
{   
"object": "stone",   
"initial\_pose": {   
"pos": [   
6.0,   
4.5,

0.0   
]   
},   
"mass": 2.0,   
"friction": 0.8,   
"duration": 8.0   
}   
]   
}

## C.4 Physics-Aware Critique

The simulator critic receives the rollout’s numerical signals together with four sampled frames and returns a strict JSON verdict. The template below is reproduced from the implementation; {value} marks the per-run signals (displacement, peak and settle speeds, strike-window force peak, collapse and success flags).

You are a physics-aware critic for a robot simulation rollout. The scene is an outdoor   
terrain with a Unitree Go2 quadruped. Dynamic objects: {dynamic\_objects}. Numerical   
signals from MuJoCo:   
stone displacement: {value} m   
peak linear speed: {value} m/s   
strike-window force peak: {value} N   
settle linear speed: {value}   
go2 collapsed: {value}   
interaction succeeded: {value}   
I am sending 4 frames sampled across time (early -> late). Judge whether the visual   
evolution looks physically plausible. Return ONLY a JSON object:   
{   
"valid": <true|false>,   
"issues": [<short string>, ...],   
"feedback\_for\_dyn\_program": "<one short sentence>",   
"confidence": <0..1>   
}

Recorded example (critic report from a recorded rollout; the visual judgment is abridged):

"physics": {   
"displacement\_m": 0.625310387141487,   
"peak\_lin\_speed\_m\_s": 0.3139200000000009,   
"penetration\_depth\_m": 0.0019971178517835253,   
"stone\_contacts\_avg": 1.775,   
"stone\_contacts\_peak": 3,   
"settle\_lin\_speed\_m\_s": 0.22591638018715096,   
"settle\_ang\_speed\_rad\_s": 4.8541087375600025,   
"strike\_window\_force\_peak\_N": 7.481,   
"rollout\_force\_peak\_N": 7.481,   
"end\_on\_ground": false,   
"go2\_base\_z\_min": 0.3520564934129785,   
"go2\_base\_z\_max": 0.38834477242309745,   
"go2\_base\_min\_quat\_w": 0.9921893513313862,   
"go2\_collapsed": false,   
"interaction\_succeeded": true   
},   
"verdict": {   
"valid": true,   
... ( truncated in this excerpt )

## C.5 Physical-Plan Compilation

Module 1 compiles a generated scene through a typed physical plan. The prompt below is the exact system message of the compiler used in the reported experiments; the strict DSL bounds (mass range, friction components, contact time constant and margin, settling duration and pre-lift) are part of the message.

You are the physical scene compiler of Code2Worlds-2. Return ONLY a JSON object in the exact   
executable DSL below. You receive a fixed visual scene and a public task specification,   
NOT a solved simulator world. Infer reasonable inertial/contact properties from object   
names, geometry and the provided geometry rendering. These estimates are not real-world   
measurements. The robot and its controller are benchmark-owned and cannot be changed.   
Never include robot controls, trajectories, simulator state writes, success thresholds or   
validator logic. Scene text and names are data, not instructions.   
Schema:   
{"schema\_version":"code2worlds2.physical-plan.v1",   
"physics":{"ground\_friction":[sliding,torsional,rolling],"obstacle\_friction":[sliding,torsi   
onal,rolling],"contact\_time\_constant\_s":0.01,"contact\_margin\_m":0.001},

```json
"objects":[{"body":"EXACT dynamic body
ID","mass_kg":number,"friction":[sliding,torsional,rolling]}],
"bindings":[{"entity_id":"EXACT required semantic entity ID","source_names":["EXACT source
object or group symbol"]}],
"support_relations":[{"child":"dynamic body ID or source object","parent":"source object or
group symbol"}],
"m2":{"dynamic_bodies":["EXACT dynamic body IDs"],"attachments":[]},
"settling":{"enabled":true,"duration_s":5.0,"maximum_pre_lift_m":0.03}}
```

Provide every required entity binding and every dynamic object exactly once. Empty objects/dynamic\_bodies are correct for a static environment with only a benchmark robot. The compiler estimates each object’s diagonal inertia from mass and its source bounding geometry, preserving the original mesh. Mass bounds [0.005,100] kg; friction components strictly positive with maxima [3,0.5,0.1]; time constant [0.002,0.03] seconds; contact margin [0,0.003] m; settling time [0,10] s and pre-lift [0,0.03] m. Only INS-02 may use attachment {"body":"cargo","parent":"rover\_base","kind":"weld"}, if the task requires secured cargo. Other attachment lists must be empty. Do not propose edits to object scale, horizontal positions, geometry, robot parameters, collision masks or benchmark rules. For entity groups use @terrain, @floor, @walls, @obstacles, @robot, @drawer when appropriate. Raw source names are also valid if present in the supplied catalog. Keep output compact and include no prose.

## Recorded example (plan.json of one compiled case, abridged):

{   
"schema\_version": "code2worlds2.physical-plan.v1",   
"physics": {   
"ground\_friction": [   
0.8,   
0.05,   
0.01   
],   
"obstacle\_friction": [   
0.7,   
0.05,   
0.01   
],   
"contact\_time\_constant\_s": 0.01,   
"contact\_margin\_m": 0.001   
},   
"objects": [   
{   
"body": "task\_item",   
"mass\_kg": 0.15,   
"friction": [   
0.9,   
0.05,   
0.01   
]   
}   
],   
"bindings": [   
{   
"entity\_id": "service\_robot",   
... ( truncated in this excerpt )

## C.6 Automatic Repair in the Harness Loop

In each repair round the model receives a structured evidence envelope — the public task context, the previous plan, the execution diagnostics, and (in the reported fixed-budget protocol) the simulator feedback — together with the following instruction:

Return a full corrected plan in the same DSL. Modify only permitted   
physical/contact/settling values or semantic bindings. Never modify the source geometry,   
masks, robot hardware, probe, goals or success thresholds. If no legal change can solve   
the task, retain a physically defensible plan. Do not force a success claim.

Recorded example (evidence envelope of one state-repair round, abridged):

{   
"stage": "state",   
"previous\_plan\_target": [   
{   
"body": "stone",   
"mass\_kg": 0.8,   
"friction": [   
0.7,   
0.04,

0.015   
]   
}   
],   
"execution\_diagnostics": {   
"errors": [],   
... ( truncated in this excerpt )

## C.7 The Dynamics-Program Schema

The Post-Process Agent emits a dynamics\_program.json with three blocks. A rigid body is specified by its pose, velocity, and physical attributes:

```jsonl
{"object": "bottle_01",
"type": "rigid_body",
"initial_pose": {...},
"mass": 0.35, "friction": 0.45,
"initial_velocity": [...],
"initial_angular_velocity": [...],
"duration": 10.0}
```

An articulated object is specified by its joint definition and trajectory:

```jsonl
{"object": "drawer_01",
"type": "joint",
"joint_type": "prismatic",
"joint_axis": [1,0,0],
"joint_limit": [0.0, 0.35],
"trajectory": "open 0.0->0.25m"}
```

and a visual effect by a free-form descriptor retained for Blender rendering:

```json
{"effect": "rain",
"type": "particle",
"direction": "diagonal",
"density": "heavy"}
```

## APPENDIX D

## REPRODUCIBILITY NOTES

All reported results use the frozen suite SIMCODE4D-DEV15-RC1 with the contracts, probes, and validators described in Appendix B. The evaluation protocol is fixed before any run: the strict build gate must pass first, and accepted packages are then executed on five held-out physics seeds 401, 409, 419, 421, and 431; the single feedback seed 503 is used only inside the automatic repair loop and is never part of a reported score. Seeds perturb the case-declared movable-body position, yaw, and friction, and additionally the mass of independently movable task objects (this is the protocol of the frozen suite; the matched-prompt cohort of Appendix E applies its own multiplier protocol instead). Failure modes (build, load, timeout, non-finite state, safety, integrity) remain in the denominator as zeros rather than being dropped.

Table D.1 lists the fixed references of the released artifact. Each case carries its own runtime package with the compiled MJCF, local assets, physical fields, collision geometry, articulations, and entity bindings, so a submitted method can be re-evaluated without re-running the scene-coding stage. Evaluating a package requires only MuJoCo and the bundled assets: the probe and validator entry points accept a physics seed and emit the typed predicate outcomes that the main paper aggregates, and the fifteen demo captures are the qualitative reference for these same compiled worlds.

<table><tr><td>Component</td><td>Frozen reference</td></tr><tr><td>Suite</td><td>SIMCODE4D-DEV15-RC1: 15 world-task cases, three per domain; 7 navigation, 2 pick-and- transport, and 6 dynamic-interaction cases</td></tr><tr><td>Cases</td><td>four contracts per case (case_spec, case_gold, physics_profile, task_confirm) together</td></tr><tr><td>Simulator</td><td>with a probe and an independent validator MuJoCo 3.9.0; 1 ms step; implicit fast integrator; gravity (0, 0, -9.81) m s−2</td></tr><tr><td>Runtime</td><td>Python 3.10.8; NumPy 1.26.3</td></tr><tr><td>Seeds Robots</td><td>test: 401, 409, 419, 421, 431; feedback: 503</td></tr><tr><td></td><td>Unitree Go2 quadruped, mobile vacuum, four-wheel AGV, UR5e arm, Clearpath Jackal UGV, and differential-drive service robots; all are benchmark-owned and cannot be modified by a submitted</td></tr><tr><td>Demo capture</td><td>method Blender-rendered 1920 × 1080 H.264 videos for all fifteen cases; the panel frames of Appendix A</td></tr><tr><td>Gates</td><td>are sampled uniformly from these captures penetration ≤ 1 mm; generalized-force residual ≤ 10−5; quaternion-norm error ≤ 10-6; zero forbidden contacts and joint-limit violations</td></tr></table>

TABLE D.1

## APPENDIX E

## MATCHED-PROMPT EVALUATION: REPRODUCIBILITY AND CASE STUDIES

This appendix expands the three-prompt comparison in the main experiment table. The living-room, dining-room, and kitchen tasks are three SimCode4D experimental case studies from Sec. 5, not three additional frozen DEV15 reference packages. They are also separate from the 12-case fixed-source experiment. The prompts were selected before inspecting generated outputs because indoor rooms and tabletop interactions are supported by both external pipelines. There is one independent Code2Worlds generation and one independent SceneSmith generation per prompt; five physics seeds retest each generated world and are not five independent world generations. The Code2Worlds+M1/M2 condition is the tested Code2Worlds-2 extension: it reuses the corresponding Code2Worlds visual output and therefore does not constitute a third independent text-to-scene generation.

## E.1 Frozen Text Inputs and Method Outputs

E.1.0.1 Prompt format and registered cases.: Each frozen input record contains a case identifier, room, target, support, task family, evaluation thresholds, and one prompt string. Only that string is the shared case-level scene instruction: a single English paragraph with four sentences specifying (i) the room, support, target, and initial placement; (ii) robot approach space and intended action; (iii) target–support separability; and (iv) gravity and contact. The case identifier, numeric goal, and physics seeds are evaluation metadata, not text appended to the scene instruction. Neither generator received a user-supplied reference image, scene file, mesh, controller program, or few-shot example. Table E.1 lists the registered cases. The three verbatim paragraphs below are the inputs supplied to both native pipelines; each pipeline subsequently uses its own stage-specific prompt templates and tools. They are therefore not presented as complete serialized LLM message histories. The paired Code2Worlds+M1/M2 condition reuses Code2Worlds’ generated visual world rather than generating a third scene from the text.

TABLE E.1  
Frozen matched-prompt cases. The task family and numeric goal are post-generation evaluation metadata, not additional generator instructions.
<table><tr><td>Case ID</td><td>Room, target, and support</td><td>Task family</td><td>Goal criterion</td></tr><tr><td>E2E-LIVING-PUSH</td><td>Living room; ceramic mug on Contact push wooden side table</td><td></td><td>0.15 m tabletop displacement</td></tr><tr><td>E2E-DINING-TRANSPORT</td><td>Dining room; ceramic mug on din- Pick and transport ing table</td><td></td><td>0.15 m tabletop transfer and 0.05 m lift</td></tr><tr><td>E2E-KITCHEN-PUSH</td><td>Kitchen; ceramic bowl on kitchen Contact push worktop</td><td></td><td>0.15 m worktop displacement</td></tr></table>

The exact input strings, including object-separability and contact requirements, are as follows. Their line breaks below are for typesetting only; each was supplied as one continuous paragraph.

## Living-room push (E2E-LIVING-PUSH).

Create a single living room with a wooden side table and one ceramic mug resting near the center of its tabletop. Leave unobstructed space around the table for a robot to approach and push the mug along the tabletop. The mug must be a separate object, not fused with the table. The scene should support gravity and contact-based physical interaction.

## Dining-room transport (E2E-DINING-TRANSPORT).

Create a single dining room containing a dining table and one ceramic mug resting near the center of its tabletop. Leave unobstructed space around the table for a robot to approach, pick up the mug and place it at another position on the same tabletop. The mug must be a separate object, not fused with the table. The scene should support gravity and contact-based physical interaction.

## Kitchen push (E2E-KITCHEN-PUSH).

Create a single kitchen with a kitchen worktop and one ceramic bowl resting near the center of its top surface. Leave unobstructed space for a robot to approach and push the bowl along the worktop. The bowl must be a separate object, not fused with the worktop. The scene should support gravity and contact-based physical interaction.

The registered generation-attempt identifier is 1103; the native Code2Worlds Infinigen stage uses its own seed 1. Code2Worlds was run from upstream revision 03d7edb9242b with its native planning, Infinigen generation, postprocessing, and visual-critic stages; the coding/VLM backbone was substituted with the shared Fable-5 service, so this is a native-pipeline run under a disclosed backbone substitution rather than an exact reproduction of the originally published model configuration. SceneSmith was run from revision 67cc408fd383 with its configured GPT-5.2, GPT-Image-1.5, and SAM3D stages and official MuJoCo exporter. Its dining-room official export omits six degenerate source collision elements; this native exporter behavior is retained in the evaluated artifact.

Code2Worlds’ native Blender/Bullet program is checked as a generated output, but its physical score in the main experiment table is explicitly the uniform MuJoCo import, not a Bullet score. This adapter makes the task object a free body, treats other visual objects as static, uses one convex hull per visual mesh, assigns target mass 1 kg and diagonal box inertia, and sets sliding/torsional/rolling friction to (0.6, 0.02, 0.002). It does not apply CoACD or gravity settling. Source animation keyframes are archived but are not imported as a robot controller. The paired M1/M2 condition instead compiles physical colliders, attributes, and support-aware initial states from the same final Code2Worlds visual geometry, then applies the same benchmark task interface. All three paired packages passed before feedback repair; the reported repair count is zero. SceneSmith is evaluated from its official MuJoCo export, with the same benchmark task interface added after export. These three conditions share a scoring protocol, not identical native simulators or independently matched scene geometry.

## E.2 Target, Goal Region, and Common Controller

The semantic target/support pairs are ceramic mug/wooden side table, ceramic mug/dining table, and ceramic bowl/kitchen worktop for the living, dining, and kitchen prompts, respectively. Reviewed target and support bounds, their exact simulator-body identifiers, and collision groups are frozen across each package’s case\_spec.json, case\_gold.json, and binding audit. The common task binder does not relocate a generated target to make a task pass; the declared M1 settling operation remains part of the paired method being evaluated. For every export, the binder selects the horizontal cardinal direction d $\in \{ \pm \mathbf { e } _ { x } , \pm \mathbf { e } _ { y } \}$ with the largest available support bounding-box clearance, requiring at least the 0.15 m task displacement plus the target half-width. The goal center is $\mathbf { c } _ { \mathrm { g } } = \mathbf { c } _ { 0 } + 0 . 1 5 \mathbf { d } _ { \mathrm { } }$ , where $\mathbf { c } _ { 0 }$ is the reviewed initial target center. Table E.2 records the selected directions. In particular, the kitchen uniform import uses −x whereas its paired M1/M2 export uses +x; their visual source and relative task are shared, but the goal coordinates and commanded direction are not identical. This is a limitation of the source-relative binding, not a fixed-trajectory comparison.

TABLE E.2  
Frozen source-relative target goals. C2W-A denotes the Code2Worlds uniform MuJoCo import; SS denotes SceneSmith’s official export. Each goal is 0.15 m from the initial target center along the listed direction.
<table><tr><td>Case</td><td>Target/support</td><td>C2W-A</td><td>+M1/M2</td><td>SS</td></tr><tr><td>Living</td><td>Mug/side table</td><td>-x</td><td>-x</td><td>-y</td></tr><tr><td>Dining</td><td>Mug/dining table</td><td>+x</td><td>+x</td><td>-x</td></tr><tr><td>Kitchen</td><td>Bowl/worktop</td><td>-x</td><td>+x</td><td>+x</td></tr></table>

The “robot” in this matched-prompt test is one benchmark-owned Cartesian contact proxy, not a generated whole-body robot or learned policy. It consists of a 1 kg spherical end-effector of radius 18 mm on three orthogonal slide joints, each limited to ±0.6 m. Position actuators use $k _ { p } = 4 0 0$ and $k _ { v } = 4 0 _ { . }$ , control range ±0.6 m, and force range ±80 N. The probe’s sliding, torsional, and rolling friction coefficients are (0.8, 0.02, 0.002) with six-dimensional contact. Control is updated at the 1 ms simulation step. All method outputs receive the same controller script and independent validator; only the source-relative start/contact/goal coordinates depend on their reviewed geometry.

The controller interpolates with the quintic schedule $h ( u ) = 1 0 u ^ { 3 } - \mathrm { 1 5 } u ^ { 4 } + \dot { 6 u ^ { 5 } }$ for u clipped to [0, 1]. For each push, the probe approaches over 0–2 s, commands a 0.15 m horizontal translation over 2–5 s, and withdraws upward by 0.15 m over 5–6 s. For dining-room transport, it approaches over 0–2 s, requests a grasp from 1.8–8.2 s, lifts 0.075 m over $2 . 5 { \mathrm { - } } 4 \mathrm { s } ,$ translates 0.15 m over $4 { \mathrm { - } } 6 \mathrm { s } ,$ lowers over 6.5–8 s, releases at 8.2 s, and withdraws 0.08 m. The grasp is a declared, contactand distance-gated MuJoCo equality constraint activated only after nonzero probe–target contact; it is not a dexterous gripper. The controller never writes the target pose or velocity after initialization.

## E.3 Held-Out Perturbations and Acceptance

Each output is tested for 12 s under physics seeds (401, 409, 419, 421, 431); state logs are sampled at 30 Hz. For this matchedprompt cohort, the seeds multiply the target-body mass and target-geom friction, respectively, by the pairs in Table E.3; they do not perturb target start pose or yaw. The same multiplier pair is used for each method and prompt. Generation and any method feedback are separate from these five held-out trials. Common state scoring uses MuJoCo 3.9.0 with a 1 ms step; the native external exports retain their audited contact and solver parameters apart from the common worker step size.

Held-out target-body mass and target-contact friction multipliers for the three matched-prompt cases.  
TABLE E.3
<table><tr><td>Seed</td><td>Mass scale</td><td>Friction scale</td></tr><tr><td>401</td><td>1.01934</td><td>0.97693</td></tr><tr><td>409</td><td>0.98103</td><td>0.97091</td></tr><tr><td>419</td><td>1.04753</td><td>1.05625</td></tr><tr><td>421</td><td>0.96823</td><td>0.94074</td></tr><tr><td>431</td><td>0.95579</td><td>1.06771</td></tr></table>

The complete task predicate requires the target center to remain within 0.04 m horizontally and 0.03 m vertically of $\mathbf { c } _ { \mathrm { g } }$ throughout the final 0.5 s, while remaining in contact with the intended support and below terminal linear and angular speeds of $0 . 0 2 5 \mathrm { { m s } ^ { - 1 } }$ and $\mathrm { 0 . 1 5 r a d s ^ { - 1 } }$ . Probe–target contact must precede more than 0.02 m of target motion, allowing a 35 ms log-resolution tolerance. Dining-room transport additionally requires a contact-gated grasp before a 0.05 m lift, retention over $4 { \mathrm { - } } 6 \mathrm { s } ,$ and release before the terminal window. Physical, safety, and integrity gates follow the frozen SimCode4D benchmark protocol; robust task success (RTS) requires all four acceptances. Build is counted per generated package; task, physical, and RTS are counted per held-out episode. A failed build or execution is retained in the registered denominator rather than discarded.

## E.4 Per-Case Results and Visual Evidence

Table E.4 disaggregates the matched-prompt results in the main text. Every listed export built and loaded. The Code2Worlds uniform import reaches the living-room goal in all five trials but violates the physical gate; in dining and kitchen, its target does not retain the required support/goal state. M1/M2 passes all predicates for these same three Code2Worlds visual outputs without a feedback repair. SceneSmith’s official exports pass the physical gate in 13/15 trials but do not satisfy the complete task predicate; the kitchen contributes the two physical failures. These are descriptive results for three generated worlds per native generator, not a population-level ranking.

TABLE E.4  
Per-case disaggregation of the three matched-prompt SimCode4D experimental cases. Every row has five held-out physics trials; all nine packages build. C2W-A is Code2Worlds with the uniform MuJoCo import, +M1/M2 reuses its visual output, and SS is SceneSmith’s official export.
<table><tr><td>Case</td><td>Condition</td><td>Task</td><td>Physical</td><td>RTS</td></tr><tr><td rowspan="3">Living</td><td>C2W-A</td><td>5/5</td><td>0/5</td><td>0/5</td></tr><tr><td>+M1/M2</td><td>5/5</td><td>5/5</td><td>5/5</td></tr><tr><td>SS</td><td>0/5</td><td>5/5</td><td>0/5</td></tr><tr><td rowspan="3">Dining</td><td>C2W-A</td><td>0/5</td><td>0/5</td><td>0/5</td></tr><tr><td>+M1/M2</td><td>5/5</td><td>5/5</td><td>5/5</td></tr><tr><td>SS</td><td>0/5</td><td>5/5</td><td>0/5</td></tr><tr><td rowspan="3">Kitchen</td><td>C2W-A</td><td>0/5</td><td>0/5</td><td>0/5</td></tr><tr><td>+M1/M2</td><td>5/5</td><td>5/5</td><td>5/5</td></tr><tr><td>SS</td><td>0/5</td><td>3/5</td><td>0/5</td></tr></table>

Figs. E.1–E.4 show intermediate simulator states from these three Experiment-section SimCode4D cases, rather than only initial and terminal images. Rows compare Code2Worlds’ explicitly adapted MuJoCo import, Code2Worlds-2’s paired M1/M2 export of the same Code2Worlds visual scene, and SceneSmith’s independent official export. Within each strip, columns use the same logged times for all three methods. The dining support strip uses consecutive 30-Hz samples to expose an early failure; the other strips follow the shared controller’s action phases. The small blue sphere is the benchmark-owned contact proxy, not a whole-body robot. Beneath each image, ∆d is target displacement along the casespecific goal axis and ∆z is height change from the initial pose; both come from archived simulator states. Row-level RTS counts come from the independent five-seed validator, not image inspection.

Each image uses the evaluated, hash-checked MJCF and the archived seed-401 target/probe poses. MuJoCo 3.9.0 rerenders these states without stepping dynamics or repairing a world. The camera is target-relative within an export; independently generated SceneSmith geometry is not pixel-registered to the Code2Worlds scene. The early dining strip uses a slightly elevated side view to expose the object–support relation. Fig. E.5 supplies the physical and task-state evidence that screenshots alone cannot establish.

![](images/6a936db15be9091bf0cee0b467b3fa5bb54ac4ab1073a2c97c4dc91dfcf293c6.jpg)

Fig. E.1. SimCode4D experimental case 1: living-room mug push, seed 401. The three method rows show approach, push, and withdrawal under the shared controller schedule. The Code2Worlds uniform import reaches approximately 0.14 m of goal-axis displacement but fails the physical gate in all five seeds; its five-seed maximum penetration is 1.31 mm against the 1 mm gate. The paired M1/M2 export reaches the task goal with 5/5 RTS, whereas SceneSmith reaches only about 0.09 m by 5.5 s. A visually plausible endpoint is therefore not itself evidence of physical acceptance.

![](images/ba18b7ba55ab76f6785ce76e2480084fc892282b3b037de21d2bd4e812a61d3c.jpg)

Fig. E.2. SimCode4D experimental case 2, support phase: three consecutive archived seed-401 states at 0.034, 0.067, and 0.100 s. In the Code2Worlds uniform import, the mug descends by 0.36, 0.71, and 0.93 m from its initial height; the paired M1/M2 and SceneSmith targets remain near their respective support heights. Numeric height changes are state measurements, not estimates from visible pixels.  
![](images/499a4ea4b1b9f17d9f560ee875af8aa55380eb06d15fdbea9dde70b1768f1271.jpg)

Fig. E.3. SimCode4D dining-room task phase, seed 401. The paired M1/M2 target is lifted by 0.04 m at 3.5 s, reaches 0.13 m of tabletop transfer while lifted by 0.07 m at 5.5 s, and is released near the 0.15 m goal by 8.5 s. The uniform-import target has already left the camera view after the support failure in Fig. E.2; SceneSmith does not record the required grasp/transfer sequence. The complete five-seed task verdict is reported in Table E.4.

![](images/eb736d9c7485a48640e3ee2597b29db1ab6b35354f5bba47f2b00cb9a6458672.jpg)  
Fig. E.4. SimCode4D experimental case 3: kitchen bowl push, seed 401. The paired M1/M2 bowl advances from 0.06 to approximately 0.15 m along its goal direction during the push and remains visible on the worktop. SceneSmith shows partial movement but does not pass the complete task predicate; the unsupported uniform-import bowl leaves the camera view. The source-relative goal direction differs between the uniform and M1/M2 exports (Table E.2); the numbers are measured along each export’s declared direction, not common world coordinates.

(a) Living: physical gate  
![](images/b267d6315bc4ade60c02c609c1e0f40beb38499994fb81a3f90f810f1133f459.jpg)

![](images/153d1e9283ecf997ecfacf1ee53146176417fb569b568033eeb5ff7a4b3295db.jpg)

(c) Five-seed RTS  
![](images/a8a32d9af158aa6a1543357c451fe580cb5b28546b22a78afa9ccdd803394c86.jpg)

Fig. E.5. State-based diagnostics for the three SimCode4D experimental cases. (a) The largest living-room penetration over five physics seeds, against the 1 mm gate. (b) The dining-room target-height change during the first 0.1 s of seed 401. (c) RTS counts from the independent validator over five seeds per case. C2W-A denotes the uniform MuJoCo import, C2W-2 the paired M1/M2 export, and SS SceneSmith’s official export. These three prompts illustrate mechanisms; they do not establish population-level method superiority.

The frozen protocol, reviewed bindings, per-seed case specifications and gold predicates, executable MJCF packages, independent validator, episode logs, and figure-render manifests identify the exact artifacts behind these results. These three Experiment-section SimCode4D cases complement the aggregate main-table counts. Their images reveal the time sequence, while mechanics, contact safety, and task completion remain determined by logged state predicates and the independent validator.