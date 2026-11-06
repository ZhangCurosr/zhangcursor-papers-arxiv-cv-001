# Deformable Contact-Aware 3D Object Placement

Songhe Wang

David Miller

Charles A. Kamhoua

![](images/9db5b18c007f0c92bab40138cbb6d3bac033ae1d1f461638e6bf76972a233d7e.jpg)  
Figure 1: DCAP overview. The method first builds a release state from the prompt, scene, and object mesh. It gives the simulator a target, a scale, and an orientation. It also defines material roles and the particle representation. A rigid–deformable simulator then settles the object and returns the final state with diagnostics and a feasibility flag.

## Abstract

Current language-guided 3D object placement typically predicts a rigid 6-DoF pose, ignoring the physical reality that indoor placements are dictated by contact. Consequently, predicted poses often float, interpenetrate supports, or fail to deform soft surfaces. We propose Deformable Contact-Aware Placement(DCAP), a framework that reframes placement as a drop-to-equilibrium process. Rather than predicting a static pose, DCAP constructs a semantic release state that specifies the drop location, object scale, and deformable components, allowing a coupled rigiddeformable simulator to settle the object under gravity and friction. This explicitly separates semantic intent from physical realization. Evaluated on a benchmark of 2,063 prompt-asset-scene cases, DCAP significantly outperforms static baselines. On standard placements, it improves VLM Physics & Naturalness by 21.2 points and human preference by 26.5 points over the strongest raw rigid baseline, while reducing mean penetration depth by 72.3%. Compared to simulated rollout baselines, DCAP reduces residual speed by 72.1%. Ultimately, our results prove that physically accurate 3D placement requires modeling semantic intent and physical realization as strictly separate processes.

## 1 Introduction

Language-guided 3D object placement holds strong promise for embodied AI, synthetic data generation, and virtual scene design, because it lets users arrange environments through natural-language instructions (Savva et al., 2019; Kolve et al., 2017; Yang et al., 2024a; Li et al., 2025). In the physical world, however, placement is rarely a static event. It is a dynamic interaction. Ideally, a placement system should reflect the fact that every placement is fundamentally a drop-to-equilibrium process in which an object settles until gravity, frictional contact, and material response to reach balance (Todorov et al., 2012; Stomakhin et al., 2013; Jiang et al., 2015; Hu et al., 2018; Muller et al., 2007). Such physically grounded, contact-aware placements would provide better foundations for robotic manipulation training, interactive virtual environments, and realistic scene synthesis (Savva et al., 2019; Kolve et al., 2017; Li et al., 2025; Lin et al., 2026).

Despite impressive progress in 3D-language grounding and scene understanding (Chen et al., 2020; Hong et al., 2023; Zhu et al., 2024), current placement models are still primarily designed for rigid, semantic-only pose prediction. Models such as PlaceIt3D and FirePlace study the problem of inserting an asset into a 3D scene by directly predicting a final 6-DoF rigid pose (Abdelreheem et al., 2025; Huang et al., 2025a). Yet many indoor supports such as beds, cushions, and upholstered chairs are not rigid shelves. Their deformation changes the final height, contact patch, and visible shape of the placed object (Muller et al., 2007; Stomakhin et al., 2013; Hu et al., 2018). As a result, static predictions may be physically implausible: objects may hover above fabric, fail to press into soft supports, or interpenetrate the scene geometry, which leads to failure modes that are difficult to diagnose from semantic placement alone (Liu et al., 2021; Wu et al., 2024). These limitations make rigid-only placement a poor fit for interactive, physics-driven simulation.

To bridge the gap between semantic, non-deformable placement and physically faithful 3D realization, we argue that the settling process itself must be part of the task rather than a post hoc correction. We introduce Deformable Contact-Aware Placement (DCAP), a framework for physics-aware 3D object insertion, as illustrated in Figure 1. Given a reconstructed scene, a prompt, and an object mesh, DCAP does not directly guess the final pose. Instead, it first constructs a semantically grounded release state using language, vision, and geometry cues from modern 3D grounding and open-vocabulary perception pipelines (Chen et al., 2020; Kirillov et al., 2023; Cheng et al., 2022; Peng et al., 2023; Jatavallabhula et al., 2023; Yang et al., 2024b). This state identifies the support, sets the metric scale and coarse orientation, and assigns material and deformable roles. A coupled rigid–deformable simulator then lets the object settle naturally under gravity and frictional contact (Stomakhin et al., 2013; Jiang et al., 2015; Hu et al., 2018; Genesis, 2025a,b). This design separates what the user intended from what the physical world permits, and allows the system to return both a settled state and an interpretable feasibility flag when an instruction is physically impossible.

The separation is useful in three ways. First, compliant supports and deformable objects become natural cases rather than special cases. The final pose is produced by material response, not by a rigid correction. Second, failure becomes a valid output. If the support is too narrow, the object is too large, or the motion does not converge, the system can reject the placement instead of returning an implausible pose. Third, the method exposes physical diagnostics that are difficult to see in a few renderings. They measure penetration, contact, residual motion, and target drift. Such diagnostics complement human and VLM judgments, which are useful but can miss physical errors hidden in geometry or dynamics (Wu et al., 2024; Tam et al., 2026).

Contributions: 1) We formulate language-guided 3D object placement as release-state planning followed by drop-to-equilibrium settling; 2) We introduce DCAP, a modular pipeline that combines language/vision grounding with rigid–deformable simulation; 3) We build a benchmark of simulationready indoor scenes paired with placement prompts and rigid or deformable assets; 4) We evaluate placement along semantic and physical axes, with objective diagnostics, ablations, submodule checks, and failure analysis.

## 2 Related Work

Language-guided 3D placement, grounding, and layout. Language-guided placement connects a prompt to a support, a region, an orientation, and a set of spatial constraints. PlaceIt3D and FirePlace study this problem directly for asset insertion into real or reconstructed scenes (Abdelreheem et al.,

2025; Huang et al., 2025a). Broader 3D-language and layout methods use LLMs, VLMs, scene graphs, differentiable optimization, or embodied-scene priors to arrange objects in 3D (Hong et al., 2023; Zhu et al., 2024; Huang et al., 2025b; Feng et al., 2023; Yang et al., 2024a; Sun et al., 2025; Ran et al., 2025; Yang et al., 2025). These methods are valuable because they provide semantic intent and approximate geometry. DCAP uses the same kind of grounding signal, including replaceable open-vocabulary segmentation backends (Chen et al., 2020; Kirillov et al., 2023; Cheng et al., 2022; Peng et al., 2023; Jatavallabhula et al., 2023). The difference is where these signals enter the problem. In DCAP, they define a release state. The final pose is resolved by contact and compliance.

Scene resources and visual object insertion. Large indoor reconstructions and object repositories have enabled progress in scene understanding, embodied navigation, environment generation, and asset insertion (Dai et al., 2017; Chang et al., 2017; Ramakrishnan et al., 2021; Xia et al., 2018; Savva et al., 2019; Kolve et al., 2017; Deitke et al., 2023; Yang et al., 2024b). These resources are often designed for perception or rendering, not for deformable contact. Image-space and 3DGS insertion methods further improve visual plausibility through appearance matching, shadows, relighting, or multi-view consistency (Liu et al., 2021; Zhu et al., 2023; Li et al., 2025), complementary to DCAP. A visually plausible insertion can still be physically invalid, and a settled 3D state can later be rendered with stronger image-space tools. Our benchmark therefore focuses on the requirements of physical placement, including metric scale, usable support geometry, material labels, and contact diagnostics.

Physical simulation and diagnostic evaluation. Frictional contact and deformation are standard topics in MPM, PBD, and rigid-body simulation (Stomakhin et al., 2013; Jiang et al., 2015; Hu et al., 2018; Muller et al., 2007; Todorov et al., 2012; Makoviychuk et al., 2021; Genesis, 2025b,a; NVIDIA, 2025). MPM is a natural fit for compliant placement because particles represent soft material while a background grid transfers stress and contact. Rigid-body engines remain efficient for stiff components and mature collision handling. Recent physics-augmented scene generation also uses simulation to improve equilibrium and non-intersection, but it mainly targets rigid scene-level generation rather than deformable insertion into a fixed reconstruction (Lin et al., 2026). DCAP combines these views by simulating soft or contact-critical parts deformably while keeping stiff components rigid. For evaluation, we pair human and VLM judgments with direct measurements of contact, motion, and convergence (Wu et al., 2024; Tam et al., 2026).

## 3 Problem Setup

Given a reconstructed indoor scene mesh $S ,$ a natural-language placement instruction $p ,$ and an object mesh $M _ { O }$ , our goal is to produce a physically settled placement outcome

$$
y = \left( R ^ { \star } , \mathbf { t } ^ { \star } , \widetilde { M } _ { O } , \widetilde { S } , f , d \right) .
$$

Here $R ^ { \star } \in S O ( 3 )$ , the 3D rotation group, and $\mathbf { t } ^ { \star } \in \mathbb { R } ^ { 3 }$ are the final object rotation and translation in scene coordinates. The tildes denote the settled geometries after pose application and any simulated deformation. The flag $f \in \{ 0 , 1 \}$ indicates whether placement is feasible, and d stores diagnostic measurements such as penetration, support contact, target drift, residual motion, and convergence. The core modeling choice in DCAP is to separate semantic intent from physical realization. DCAP does not directly predict the final pose $( R ^ { \star } , { \bf t ^ { \star } } )$ . Instead, it constructs a semantic release state and lets a physics-based drop-to-equilibrium rollout determine the final settled outcome.

Release state. We define the pre-drop release state as

$$
z _ { 0 } = \left( b , s , S _ { b } , P _ { b } , R _ { 0 } , \mathbf { t } _ { 0 } , \pi , a , \theta , \mathcal { R } , \mathcal { D } , X _ { 0 } \right) .
$$

Here b is the intended support category, $S _ { b } \subseteq S$ is the selected support instance, and $P _ { b }$ is the set of semantic parts of that support. The scalar $s > 0$ is a category-level real-world scale prior for the object; it is not chosen to guarantee that the object fits on the selected support. The pair $( R _ { 0 } , \mathbf { t } _ { 0 } )$ is the initial release pose. The symbol π denotes the target support part, such as a cushion or shelf board, and a denotes the anchor within that part, such as center, left, right, front, back, top, head, or foot. The material map θ assigns physical parameters such as density, stiffness, and friction. The sets R and D contain the components treated as rigid and deformable, respectively, and $X _ { 0 }$ stores the initial Material Point Method (MPM) particles for deformable components. A release state is an intended initialization, not a guarantee of physical feasibility. Even if $z _ { \mathrm { 0 } }$ is semantically correct, the scaled object may be too large for the selected support, the support may not provide stable contact, or the rollout may fail to converge.

Drop-to-equilibrium settling. Given $z _ { 0 }$ , DCAP computes the final outcome through a rigid– deformable rollout

$$
y = \Gamma ( z _ { 0 } ; S , M _ { O } , \mathbf { g } ) ,
$$

where Γ denotes the simulator and diagnostic operator, and g is the gravity vector. The rollout applies the scale $s ,$ initializes the object at $( R _ { 0 } , \mathbf { t } _ { 0 } )$ , couples rigid bodies with deformable MPM particles, and simulates contact under gravity and friction. It terminates when the system reaches equilibrium or when the maximum simulation horizon is reached. Equilibrium is detected using velocity, center-of-mass drift, and bounding-box change thresholds over a fixed convergence window. Appendix F gives thresholds and simulator settings. We avoid treating rollout as post-processing correction: its output is the physical realization of the release state.

Feasibility and diagnostics. A placement is marked feasible only if three requirements are satisfied: the rollout converges, the object remains on the intended support $S _ { b } ,$ , and the diagnostic measurements in d stay within tolerance. The diagnostics include mean penetration depth, support contact ratio, target drift from the intended part-anchor pair $( \pi , a )$ , residual speed, and convergence status. Appendix L defines these diagnostics, and Appendix F lists the numerical tolerances. This definition separates different failure modes. A placement can be dynamically invalid if it does not settle, semantically off target if it settles on the wrong support or region, or infeasible if it violates any required semantic, contact, convergence, or tolerance condition. Static baselines do not run the rollout operator Γ, so dynamic quantities such as residual speed and convergence are undefined for them. They can still be evaluated using static geometric and semantic diagnostics, such as support correctness, penetration, contact, and target drift.

## 4 Method: DCAP

DCAP is modular. As shown in Figure 1, it separates semantic release-state construction from physical realization. DCAP maps the inputs to a release state through a planner:

$$
A _ { \psi } : ( S , p , M _ { O } ) \mapsto z _ { 0 } , \qquad y = \Gamma ( z _ { 0 } ; S , M _ { O } , \mathbf { g } ) .\tag{1}
$$

Here $A _ { \psi }$ denotes DCAP’s planner, and $\psi$ collects its fixed prompts, perception modules, and geometric rules. Upgrading any module, such as the LLM, VLM, segmenter, or simulator, improves one estimator inside $A _ { \psi }$ without changing the underlying task definition. Each stage estimates part of $z _ { 0 } ,$ , records uncertainty when needed, and passes a physically meaningful state to the simulator. Table 1 summarizes how modules instantiate the release-state, and Figure 2 shows particle-filling; Appendices B, D, and C give the prompt schemas, segmentation details, and material library.

Table 1: DCAP modules for release-state construction and physical rollout.
<table><tr><td>Component Module</td><td></td><td>Role</td></tr><tr><td> $b , s$ </td><td></td><td>Support grounding and scale prior Infer the intended support category and a physically mean- ingful object scale</td></tr><tr><td> $S _ { b } , P _ { b }$ </td><td>Segmentation and parts</td><td>Identify the support instance and decompose it into usable semantic parts</td></tr><tr><td> $R _ { 0 }$ </td><td></td><td>Three-candidate VLM orientation Select a coarse release orientation from prompt- conditioned rendered candidates</td></tr><tr><td> $\pi , a , \mathbf { t } _ { 0 }$ </td><td>Ray-cast anchoring</td><td>Convert the prompt into a release point above the target region</td></tr><tr><td> $\theta , \mathcal { R } , \mathcal { D }$ </td><td>Materials and split</td><td>Assign discrete material values and rigid/deformable roles</td></tr><tr><td> $X _ { 0 }$ </td><td>Thin-aware particle filling</td><td>Initialize a contact-ready deformable particle set</td></tr><tr><td>y</td><td>Drop-to-equilibrium simulation</td><td>Rollout output; not a component of z0</td></tr></table>

## 4.1 Semantic Grounding and Scene Decomposition

Because physics-based simulation is acutely sensitive to metric scale and boundary constraints, DCAP first establishes a formal mapping between semantic intent and the reconstructed scene. We leverage the common-sense reasoning of a large language model (LLM) to extract the intended support category b and a realistic longest-dimension prior from the instruction $p .$ By isotropically rescaling the object mesh to match this category-level size prior, we obtain metric collision geometry and an approximate volume scale for simulation. Mass and inertia are then approximated from the scaled geometry together with the material parameters in θ. Given the identified support category $b ,$ the system then locates the specific support instance $S _ { b }$ and decomposes it into semantic parts $P _ { b }$ (Figure 1, Steps 1–3). We follow a standard 2D-to-3D open-vocabulary pipeline to segment the scene into semantic parts (e.g., mattress vs. frame) (Kirillov et al., 2023; Cheng et al., 2022; Peng et al., 2023; Jatavallabhula et al., 2023). Beyond visually grounding the prompt, this decomposition establishes material properties to restrict expensive deformable simulation only to necessary regions.

![](images/5d6f43637bbdfa72c89e149703559724bfea691e189f6509edf7335c098fe49c.jpg)

![](images/da5ac60e76cef3ca0552cc4427d81da121430c2e7e17061d8d3194df64803ebe.jpg)  
Figure 2: Thin-aware particle filling. DCAP estimates local thickness and adapts particle spacing so thin compliant regions retain enough through-volume samples for contact and stress resolution.

## 4.2 Release Pose Initialization

With the object scaled and the scene segmented, the next step is to define the release pose $( R _ { 0 } , \mathbf { t } _ { 0 } )$ Natural language often provides only coarse orientation cues, such as a mug handle facing outward. To efficiently resolve this angular ambiguity, DCAP renders three prompt-conditioned orientation candidates created by 180-degree flips about canonical object axes and uses a vision-language model (VLM) to select the best semantic match (Figure 1, Step 4). This stage resolves only coarse semantic orientation, and the rollout handles small contact-induced pose adjustments rather than performing a continuous semantic orientation search. To determine the precise release translation $\mathbf { t } _ { 0 } ,$ , we utilize a geometric ray-cast anchoring mechanism. The system maps the prompt to a target part π and an anchor a and then casts a ray along the gravity direction to identify the corresponding physical point on the support surface (Figure 1, Step 5). To ensure numerical stability and provide sufficient headroom for gravitational settling, the object is initialized with a small positive vertical clearance $\delta > 0$ (specified in Appendix F) above the target surface. This guarantees that the settling process is driven entirely by physical mechanics rather than artificial forces from initial mesh interpenetration.

## 4.3 Material Assignment and Particle Discretization

To balance simulation fidelity with computational efficiency, DCAP must assign physical roles to the parsed geometric parts. Instead of learning continuous material parameters, we map each semantic part to a discrete material ID from a curated library, which provides the density, elastic modulus, and friction parameters collected in θ (Figure 1, Step 6). Compliant materials (foam, fabric, mattresses) are assigned to D and stiff materials (wood, metal, frames) are assigned to R. Simulating only compliant parts (e.g., cushions) deformably while keeping stiff frames rigid drastically reduces compute without sacrificing contact fidelity. Once compliant parts are identified, they must be discretized into MPM particles (Figure 1, Step 7). Uniform voxel filling often catastrophically under-samples thin indoor surfaces, such as blankets, rendering the geometry numerically fragile and incapable of resolving frictional contact. As illustrated in Figure 2, for a surface point x, DCAP estimates the local thickness $t ( \mathbf { x } )$ along the surface normal and adapts the particle spacing $h ( \mathbf { x } )$ . The constant $k _ { \operatorname* { m i n } } = 3$ is the required minimum number of particle layers through the local thickness. This targeted discretization, bounded by a global particle cap, ensures even thin compliant regions can realistically carry stress and deform naturally.

## 4.4 Drop-to-Equilibrium Simulation and Diagnostics

The final stage is where placement becomes physics. The object is released from $( R _ { 0 } , \mathbf { t } _ { 0 } )$ , allowing rigid bodies and MPM particles to interact through frictional contact until the strict convergence criteria from Section 3 are met or the maximum horizon is reached (Figure 1, Steps 8–10). The rollout outputs the final transform, the deformed scene geometry, and an array of physical diagnostics including residual motion, contact ratios, and a feasibility flag. Crucially, rather than collapsing uncertainty into an opaque score, DCAP propagates module-level confidence and diagnostic flags. Low-confidence or invalid module outputs trigger deterministic fallback rules before simulation such as scale clamping, canonical stable-orientation selection, default material assignment, and particle-budget coarsening. These fallback events are recorded as diagnostic flags; Appendix E lists the exact triggers and recovery rules. If the object leaves the target support, fails to converge, or violates tolerances, the system marks the case as infeasible and explicitly categorizes the error source (perception, geometry, materials, or physics).

## 5 Experiments and Analysis

## 5.1 Experimental Settings

A Simulation-Ready Benchmark. Unlike existing semantic-only placement benchmarks such as PlaceIt3D (Abdelreheem et al., 2025), evaluating DCAP requires physics-ready geometry. We construct a benchmark of 2,063 prompt–asset–scene cases drawn from 186 high-fidelity indoor reconstructions and 155 object assets (Dai et al., 2017; Chang et al., 2017; Ramakrishnan et al., 2021; Xia et al., 2018; Deitke et al., 2023; Yang et al., 2024b). To support rigorous contact queries and physical rollouts, we process scenes via TSDF fusion and screened-Poisson meshing to guarantee the watertight, metric surfaces required for physical simulation (Curless and Levoy, 1996; Kazhdan et al., 2006; Kazhdan and Hoppe, 2013). This pipeline preserves metric accuracy and provides watertight surfaces suitable for simulation. We evaluate all methods on two distinct splits. The standard split contains 1,923 cases and serves as the primary testbed for aggregate placement quality, human/VLM preference, and physical diagnostics. It is further diversified by contact regime, with 869 soft-contact, 550 mixed-contact, and 504 rigid-contact cases. The stress split contains 140 cases and is used specifically to evaluate robustness and feasibility handling. It includes 64 out-of-distribution feasible prompts and 76 annotated physically impossible prompts. Correct rejections on this split count only toward feasibility metrics, while any returned feasible placements receive the standard diagnostics. Appendix G documents the construction log and reporting denominators.

Baselines and Controlled Variants. To isolate the impact of our physics-aware formulation, we benchmark against both existing state-of-the-art models and internally controlled variants. To ensure a fair comparison, all methods receive the same prompt, object scale, scene mesh, and camera views. The raw static baselines are FirePlace, PlaceIt3D, LayoutGPT, and Holodeck (Huang et al., 2025a; Abdelreheem et al., 2025; Feng et al., 2023; Yang et al., 2024a). Because these models output purely static poses, dynamic metrics such as residual speed and convergence are undefined for them. We also evaluate controlled physics variants to untangle the value of the initial release state from the physical settling process. FirePlace + rigid and FirePlace + deformable use FirePlace’s static prediction as the initialization for rigid or deformable settling. Conversely, DCAP static (init+snap) tests the upstream language–vision–geometry planner as a static method by replacing the physics simulator with a naive gravity-axis snap. This geometric snap translates the object only along the gravity axis until it reaches the closest non-penetrating contact with the selected support. It does not simulate in a physics engine. Finally, DCAP rigid-only runs the full pipeline while treating all materials as rigid during rollout. Appendix I gives baseline adaptation details.

Evaluation Metrics. We assess placement success along two complementary dimensions, subjective visual plausibility and objective physical stability. Human evaluators and the VLM compare results along two axes. Right Place (RP) measures semantic accuracy by asking whether the object is on the correct support, in the intended region, and oriented consistently with the prompt. Physics & Naturalness (P/N) measures physical plausibility, so a high score requires stable support, realistic contact, material-consistent deformation, and absence of floating or deep interpenetration. Bradley– Terry–Luce estimates are derived from 2,520 preference trials with 42 retained participants, with 420 attention-check trials excluded from aggregation. Our objective physical diagnostics explicitly measure mesh penetration depth, not valid soft-support indentation, support contact ratio, residual motion, target adherence, and simulator convergence. Physical diagnostics are reported only for methods that run an explicit rollout. Appendices J, K, and L provide the full evaluation protocols.

## 5.2 Main Comparison and Controlled Variants

Table 2 gives the main comparison, with qualitative examples in Figure 4 and representative settling sequences in Figure 3. DCAP significantly outperforms existing static baselines across all placement and physical metrics. Compared with FirePlace, the strongest raw static baseline in our setting, DCAP raises the VLM Physics & Naturalness (P/N) score from 64.0 to 85.2 and the human P/N preference score from 50.1 to 76.6. Crucially, it reduces the mean penetration depth from 1.48 cm to 0.41 cm and increases valid support contact from 0.65 to 0.89. To understand where these gains come from, we isolate the two core components of our pipeline, the initial release state and the physical settling process. The controlled variants reveal that both precise initialization and dynamic physical rollout are strictly necessary. The first comparison shows that physical simulation cannot fix a bad initialization. Adding a settling process to the FirePlace baseline improves its performance, and deformable settling helps more than rigid settling. However, FirePlace + deformable still significantly underperforms full DCAP, which isolates the value of our release state. A physics simulator cannot correct an object that is dropped on the wrong part of a surface, oriented incorrectly, or scaled to an implausible size. The reverse comparison makes the complementary point. When we use DCAP’s strong initialization but replace the physics engine with a simple geometric snap (DCAP static), the result is highly competitive with raw static baselines and drastically reduces penetration. Yet its contact ratio and physical naturalness remain limited. A geometric snap can close a gap, but it cannot create a broad contact patch, simulate frictional resting, or deform a soft support. Together, these results explain why we evaluate semantic placement, Right Place (RP), and physical plausibility, Physics & Naturalness (P/N), as separate metrics. Correct spatial coordinates do not guarantee a correct physical state. A method can place a dumbbell at the center of a mattress and still fail if the mattress does not compress, or place a plush toy on the correct sofa cushion and still fail if it floats. Ultimately, these are not just different target coordinates; they are different physical states.

Table 2: Baseline comparison on the standard split (1,923 cases). Left: aggregate placement quality. Right: controlled ablations; VLM P/N gains report B − A for each A → B contrast, with “ref.” rows defining block references.
<table><tr><td colspan="7">Aggregate quality</td><td colspan="7">Controlled physics view</td></tr><tr><td>Method</td><td colspan="2">VLM Human Pen. P/N↑ P/N↑</td><td>cm↓</td><td>Contact</td><td>↑ cm/s↓</td><td>Spd. Conv.</td><td>Contrast ↑</td><td></td><td>Change</td><td>VLM P/N gain Pen. drop Final conv. Final time ↑</td><td>cm↑</td><td>↑</td><td>s↓</td></tr><tr><td>FirePlace raw</td><td>64.0</td><td>50.1 1.48</td><td></td><td>0.65</td><td>n.a.</td><td></td><td>FirePlace raw</td><td></td><td></td><td>ref.</td><td>ref.</td><td>n.a.</td><td>14.6</td></tr><tr><td>PlaceIt3D raw</td><td>60.6</td><td>n.a.</td><td>1.70</td><td>0.62</td><td></td><td>n.a.</td><td>n.a. n.a.</td><td>PlaceIt3D raw</td><td>static ref. static ref.</td><td>ref.</td><td>ref.</td><td>n.a.</td><td>n.a.</td></tr><tr><td>LayoutGPT raw</td><td>49.7</td><td>n.a.</td><td>2.31</td><td>0.47</td><td></td><td>n.a.</td><td>n.a.</td><td>FP raw→+rigid</td><td>rigid rollout</td><td>+2.7</td><td>0.38</td><td>0.61</td><td>38.2</td></tr><tr><td>Holodeck raw</td><td>54.2</td><td>n.a.</td><td>2.05</td><td>0.53</td><td></td><td>n.a.</td><td>n.a.</td><td>FP raw→+def.</td><td>deform. rollout</td><td>+8.7</td><td>0.73</td><td>0.71</td><td>86.9</td></tr><tr><td>FP + rigid</td><td>66.7</td><td>n.a.</td><td>1.10</td><td>0.70</td><td></td><td>6.8 0.61</td><td></td><td>DCAP static</td><td>geom. snap</td><td>ref.</td><td>ref.</td><td>n.a.</td><td>44.1</td></tr><tr><td>FP + def.</td><td>72.7</td><td>63.6</td><td>0.75</td><td>0.79</td><td></td><td>4.1 0.71</td><td></td><td>static→rigid</td><td>rigid rollout</td><td>+11.1</td><td>0.26</td><td>0.76</td><td>69.4</td></tr><tr><td>DCAP static (init+snap)</td><td>65.9</td><td>n.a.</td><td>0.86</td><td>0.58</td><td></td><td>n.a.</td><td>n.a. rigid→full</td><td></td><td>add deform.</td><td>+8.2</td><td>0.19</td><td>0.85</td><td>107.5</td></tr><tr><td>DCAP rigid</td><td>77.0</td><td></td><td>68.0 0.60</td><td>0.83</td><td></td><td>3.7 0.76</td><td>static→full</td><td></td><td>full rollout</td><td>+19.3</td><td>0.45</td><td>0.85</td><td>107.5</td></tr><tr><td>DCAP full</td><td>85.2</td><td></td><td>76.6 0.41</td><td>0.89</td><td></td><td>1.9 0.85</td><td></td><td>FP+def.→full</td><td>better init.</td><td>+12.5</td><td>0.34</td><td>0.85</td><td>107.5</td></tr></table>

![](images/e9dcff8c4dba13fe3fe0ebe9141ac1964d6d31130d5a24362759d0bfe943eb72.jpg)  
Figure 3: Representative settling rollouts. A backpack and a plush toy settle under gravity while preserving the intended support and scale.

![](images/c9b5eafaf532f14cec2c1a385593dfc1bf0b576629fcb319e10d8b51dfe5f751.jpg)  
Figure 4: Qualitative comparison. DCAP produces stable contact, deformation, and infeasibility handling, while rigid baselines often float, interpenetrate, or miss soft-support deformation.

## 5.3 Regime, Feasibility, and Human Validation

The strongest gains occur on soft supports like beds and sofas, where DCAP achieves an 88.2 P/N score compared to 60.4 for raw FirePlace and 74.7 for rigid-only DCAP (Tables 3 and 4). On these compliant surfaces, final height and contact are dictated by mass, stiffness, and friction rather than original mesh geometry alone. Rigid supports provide a natural control. Here, full DCAP and its rigid-only variant perform similarly, though the full pipeline remains robust due to strict convergence and feasibility checks. Beyond spatial accuracy, the stress split proves DCAP can reliably reject physically invalid prompts (0.83 F1) while accepting 52 of 64 feasible out-of-distribution cases. Human evaluations echo this physical robustness, showing a 26.5-point BTL preference over raw FirePlace in physical naturalness. Crucially, comparing full and rigid-only DCAP reveals an 8.6-point gap in naturalness (P/N) but only a 0.7-point gap in spatial accuracy (RP). This proves that deformable settling enhances realism without displacing the object from its semantic target. Ultimately, the strong alignment across human preference, VLM scoring, and objective diagnostics confirms that our gains are physically grounded rather than merely cosmetic.

Table 3: Human preference and objective diagnostics. Bradley–Terry–Luce (BTL) estimates from forced-choice evaluations strongly align with automated objective metrics. Asterisks indicate that the paired bootstrap confidence interval for the human BTL difference excludes zero after Holm correction; significance markers apply only to human forced-choice differences.
<table><tr><td colspan="5">Human forced-choice results</td><td colspan="5">Objective diagnostics</td></tr><tr><td>Comparison</td><td></td><td></td><td>Axis DCAP Baseline</td><td>Diff.</td><td>Method</td><td>Pen. cm↓</td><td>Contact Speed</td><td>↑cm/s↓</td><td>Conv. ↑</td></tr><tr><td>DCAP vs FP raw</td><td>RP</td><td>61.2</td><td>56.4</td><td>+4.8*</td><td>FirePlace raw</td><td>1.48</td><td>0.65</td><td>n.a.</td><td>n.a.</td></tr><tr><td>DCAP vs FP raw</td><td>P/N</td><td>76.6</td><td>50.1</td><td>+26.5*</td><td>FP+rigid</td><td>1.10</td><td>0.70</td><td>6.8</td><td>0.61</td></tr><tr><td>DCAP vs FP+def.</td><td>RP</td><td>61.2</td><td>55.6</td><td>+5.6*</td><td>FP+def.</td><td>0.75</td><td>0.79</td><td>4.1</td><td>0.71</td></tr><tr><td>DCAP vs FP+def.</td><td>P/N</td><td>76.6</td><td>63.6</td><td>+13.0*</td><td>DCAP static</td><td>0.86</td><td>0.58</td><td>n.a.</td><td>n.a.</td></tr><tr><td>DCAP vs DCAP rigid</td><td>RP</td><td>61.2</td><td>60.5</td><td>+0.7</td><td>DCAP rigid</td><td>0.60</td><td>0.83</td><td>3.7</td><td>0.76</td></tr><tr><td>DCAP vs DCAP rigid</td><td>P/N</td><td>76.6</td><td>68.0</td><td>+8.6*</td><td>DCAP full</td><td>0.41</td><td>0.89</td><td>1.9</td><td>0.85</td></tr></table>

Table 4: Performance across contact regimes and stress feasibility. Left: DCAP outperforms baselines on soft supports. Bold indicates best P/N score within each contact regime; convergence is reported after the slash. Right: robustness on extreme stress cases, measured by discrete rejection/acceptance counts rather than continuous diagnostics.
<table><tr><td colspan="6">Contact-regime placement quality</td><td colspan="4">Stress feasibility</td></tr><tr><td>Regime</td><td>Cases FP raw</td><td>P/N</td><td>FP+def. P/N</td><td>DCAP rigid P/N</td><td>DCAP full P/N / Conv.</td><td>Stress statistic</td><td>Infeas. Feas. Summary</td><td></td><td></td></tr><tr><td>Soft</td><td>869</td><td>60.4</td><td>70.4</td><td>74.7</td><td>88.2 / 0.89</td><td>Pred. reject</td><td>62</td><td>12</td><td>74</td></tr><tr><td>Mixed</td><td>550</td><td>64.8</td><td>74.0</td><td>78.0</td><td>84.5 / 0.84</td><td>Pred. feasible</td><td>14</td><td>52</td><td>66</td></tr><tr><td>Rigid</td><td>504</td><td>69.2</td><td>75.2</td><td>80.0</td><td>80.9 / 0.80</td><td>Reject P/R/F1</td><td>0.84</td><td>0.82</td><td>0.83</td></tr><tr><td>Standard avg.</td><td>1,923</td><td>64.0</td><td>72.7</td><td>77.0</td><td>85.2 / 0.85</td><td>OÖD true accept</td><td></td><td>0.81</td><td>52/64</td></tr></table>

## 5.4 Ablations, Runtime, and Failure Modes

Table 5 shows that robust physical simulation strictly depends on accurate upstream data. Degrading geometric fidelity by removing thin-aware filling drops the P/N score to 76.3 because coarse particles cannot support stable contact. Material precision is equally critical. Substituting generic defaults (78.0 P/N) or rigid supports (74.5 P/N) severely cripples the ability to model compliant deformation. Furthermore, physics engines cannot fix wrong semantic targets. Our fallback rules are therefore essential for mitigating inevitable perception errors and preventing hard failures from spiking (0.184 vs. 0.039). Given the 107.5-second average runtime bottlenecked by simulation, this performance profile points to a practical cascade that we can deploy fast rigid placement for stiff scenes and reserve expensive deformable settling for soft materials or uncertain contacts. Appendices E–O detail further submodule checks and rollouts.

Table 5: System ablations, runtime, and sensitivity. Left: Module-level ablations evaluated via VLM and objective metrics. Right: Pipeline runtime breakdown and hyperparameter sensitivity. Note: The left and right blocks are independent and not row-paired.
<table><tr><td colspan="5">Ablation and sensitivity results</td><td colspan="3">Runtime and sensitivity</td></tr><tr><td>Variant / setting</td><td>P/N</td><td>Pen.</td><td>Conv.</td><td>Hard fail</td><td>Stage / setting</td><td>Value</td><td>Note</td></tr><tr><td>DCAP full</td><td>84.1</td><td>0.42</td><td>0.84</td><td>0.039</td><td>Full pipeline</td><td>107.5 s</td><td>total mean</td></tr><tr><td>No thin-aware fill</td><td>76.3</td><td>0.58</td><td>0.75</td><td>0.094</td><td>Median pipeline</td><td>88.4 s median</td><td></td></tr><tr><td>No material-specific map</td><td>78.0</td><td>0.55</td><td>0.77</td><td>0.081</td><td>Simulation</td><td></td><td>54.2 s main bottleneck</td></tr><tr><td>Rigid support only</td><td>74.5</td><td>0.64</td><td>0.72</td><td>0.103</td><td>Segmentation + parts</td><td></td><td>18.6 s scene processing</td></tr><tr><td>Rigid object only</td><td>79.0</td><td>0.57</td><td>0.78</td><td>0.073</td><td>Thin-aware filling</td><td></td><td>12.8 s particle count</td></tr><tr><td>No target sub-part</td><td>76.8</td><td>0.63</td><td>0.74</td><td>0.105</td><td>Orientation VLM</td><td></td><td>8.0 s 3 candidates</td></tr><tr><td>No falīback rules</td><td>78.0</td><td>0.62</td><td>0.76</td><td>0.184</td><td>Target/overhead</td><td></td><td>2.7 s anchoring/retries</td></tr><tr><td>Coarser particles</td><td>79.8</td><td></td><td></td><td></td><td>Coarser particles</td><td>79.8 P/N</td><td>faster, less stable</td></tr><tr><td>Finer particles</td><td>84.8</td><td></td><td></td><td></td><td>Finer particles</td><td>84.8 P/N</td><td>156.7 s total</td></tr><tr><td>Shorter horizon</td><td>80.7</td><td></td><td></td><td></td><td>Shorter horizon</td><td>80.7 P/N</td><td>under-settled</td></tr><tr><td>Longer horizon</td><td>84.4</td><td></td><td></td><td></td><td>Longer horizon</td><td>84.4 P/N</td><td>132.4 s total</td></tr></table>

## 6 Discussion and Conclusion

While DCAP establishes 3D placement as a physical drop-to-equilibrium process, its robustness is inherently bounded by current simulation constraints. Our material library relies on simplified isotropic parameters, and placement fidelity is strictly bottlenecked by the quality of upstream geometry reconstruction. Furthermore, while humans and VLMs excel at assessing visual plausibility, they often overlook subtle physical violations that only objective metrics capture. Bridging these gaps will require future benchmarks that incorporate real-world physical validation for complex, deformable household materials. Despite these limitations, our coupled rigid-deformable pipeline significantly outperforms static baselines, demonstrating that realistic resting states cannot be recovered by 6-DoF pose estimation alone. Ultimately, this work highlights a fundamental truth for embodied scene synthesis. Assigned spatial coordinates are not a substitute for physical realization, and being in the right place is never the same as being physically placed.

## References

Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, and Guillermo Garcia-Hernando. PlaceIt3D: Languageguided object placement in real 3D scenes. In ICCV, 2025.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In 3DV, 2017.

Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. ScanRefer: 3D object localization in RGB-D scans using natural language. In ECCV, 2020.

Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In CVPR, 2022.

Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In SIGGRAPH, 1996.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In CVPR, 2017.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3D objects. In CVPR, 2023.

Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. LayoutGPT: Compositional visual planning and generation with large language models. In NeurIPS, 2023.

Genesis. Beyond rigid bodies: SPH, MPM, and PBD solvers in Genesis. Documentation, 2025a.

Genesis. Genesis: Physics platform for robotics and embodied AI. Documentation, 2025b.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D world into large language models. In NeurIPS, 2023.

Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics, 37(4), 2018.

Ian Huang, Yanan Bao, Karen Truong, Howard Zhou, Cordelia Schmid, Leonidas Guibas, and Alireza Fathi. FirePlace: Geometric refinements of LLM common sense reasoning for 3D object placement. In CVPR, 2025a.

Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. Reason3D: Searching and reasoning 3D segmentation via large language model. In 3DV, 2025b.

Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. ConceptFusion: Open-set multimodal 3D mapping. In Robotics: Science and Systems (RSS), 2023.

Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-in-cell method. ACM Transactions on Graphics, 34(4), 2015.

Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In Symposium on Geometry Processing, 2006.

Michael Kazhdan and Hugues Hoppe. Screened Poisson surface reconstruction. ACM Transactions on Graphics, 32(3), 2013.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In ICCV, 2023.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Kumar Gupta, and Ali Farhadi. AI2-THOR: An interactive 3D environment for visual AI. arXiv preprint arXiv:1712.05474, 2017.

Chenxi Li, Weijie Wang, Qiang Li, Nicu Sebe, Bruno Lepri, and Weizhi Nie. FreeInsert: Disentangled text-guided object insertion in 3D Gaussian scene without spatial priors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10915–10924, 2025.

Guying Lin, Kemeng Huang, Michael Liu, Ruihan Gao, Hanke Chen, Lyuhao Chen, Beijia Lu, Taku Komura, Yuan Liu, Jun-Yan Zhu, and Minchen Li. PAT3D: Physics-augmented text-to-3D scene generation. In ICLR, 2026.

Liu Liu, Zhenchen Liu, Bo Zhang, Jiangtong Li, Li Niu, Qingyang Liu, and Liqing Zhang. OPA: Object placement assessment dataset. arXiv preprint arXiv:2107.01889, 2021.

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac Gym: High performance GPU-based physics simulation for robot learning. In NeurIPS Datasets and Benchmarks, 2021.

Matthias Muller, Bruno Heidelberger, Marcus Hennix, and John Ratcliff. Position based dynamics. Journal ofVisual Communication and Image Representation, 18(2):109–118, 2007.

NVIDIA. PhysX SDK documentation, version 5. Documentation, 2025.

Songyou Peng, Kyle Genova, Chiyu “Max” Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser. OpenScene: 3D scene understanding with open vocabularies. In CVPR, 2023.

Santhosh K. Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X. Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-Matterport 3D dataset (HM3D): 1000 large-scale 3D environments for embodied AI. In NeurIPS Datasets and Benchmarks, 2021.

Xingjian Ran, Yixuan Li, Linning Xu, Mulin Yu, and Bo Dai. Direct numerical layout generation for 3D indoor scene synthesis via spatial reasoning. In NeurIPS, 2025.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A platform for embodied AI research. In ICCV, 2019.

Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew Selle. A material point method for snow simulation. ACM Transactions on Graphics, 32(4), 2013.

Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. LayoutVLM: Differentiable optimization of 3D layout via vision-language models. In CVPR, pages 29469–29478, 2025.

Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, and Manolis Savva. SceneEval: Evaluating semantic coherence in text-conditioned 3D indoor scene synthesis. In WACV, 2026.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5026–5033, 2012.

Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas J. Guibas, Dahua Lin, and Gordon Wetzstein. GPT-4V(ision) is a human-aligned evaluator for text-to-3D generation. In CVPR, 2024.

Fei Xia, Amir R. Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson Env: Real-world perception for embodied agents. In CVPR, 2018.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language guided generation of 3D embodied AI environments. In CVPR, 2024a.

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao, and Xihui Liu. SAMPart3D: Segment any part in 3D objects. arXiv preprint arXiv:2411.07184, 2024b.

Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, and Feng Zheng. OptiScene: LLM-driven indoor scene layout generation via scaled human-aligned data synthesis and multi-stage preference optimization. In NeurIPS, 2025.

Sijie Zhu, Zhe Lin, Scott Cohen, Jason Kuen, Zhifei Zhang, and Chen Chen. TopNet: Transformerbased object placement network for image compositing. In CVPR, 2023.

Chenming Zhu, Tai Wang, Wenwei Zhang, Kai Chen, and Xihui Liu. ScanReason: Empowering 3D visual grounding with reasoning capabilities. In ECCV, 2024.

## Appendix

## A Implementation and Reproducibility Details

Table 6 fixes the concrete implementation used for the reported experiments. The method is modular, but all numbers in the main paper use this configuration unless explicitly marked as an ablation. Cloud model calls are run with deterministic decoding when supported; invalid JSON responses are retried with the same prompt plus a parser error message before any geometric fallback is applied.

Table 6: Concrete implementation settings for the reported experiments.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>LLM parsing, target-part selection, material mapping</td><td>gpt-5.2; reasoning effort low; max output 512 tokens; structured JSON output; seed 0 where supported; two schema-repair retries before fallback</td></tr><tr><td>Orientation VLM inside DCAP</td><td>gpt-5.2; reasoning effort low; max output 256 tokens; three rendered candidates; no access to final evaluation scores</td></tr><tr><td>VLM evaluator Segmentation backend</td><td>gemini-3-pro-preview; temperature 0; top-p 1.0; max output 512 tokens; randomized method order; separate prompts from the orientation module OpenScene mesh features, Segment Anything ViT-H masks, and Mask2Former</td></tr><tr><td></td><td>panoptic labels fused from 48 RGB-D render views; fallback to geometric support extraction when confidence is low</td></tr><tr><td>Renderer</td><td>Blender 4.2.3 with Eevee Next rasterization; 1024× 1024 PNGs for VLM scoring and 768 ×768 panels for human trials; fixed focal length, exposure, and environment light across methods</td></tr><tr><td>Simulator</td><td>Genesis 0.2.1-compatible rigid–MPM backend with custom two-way frictional contact; Python 3.11, CUDA 12.4, single-precision GPU kernels</td></tr><tr><td>Runtime hardware</td><td>One NVIDIA RTX 4090 GPU, 24 GB VRAM; AMD Threadripper Pro 5975WX CPU; 256 GB RAM; Ubuntu 22.04; timings include rendering, segmentation, model-call latency, particle filling, simulation, and diagnostics, but exclude one-time mesh reconstruction</td></tr></table>

## B Prompts and Model Calls

This appendix records the prompt templates and constrained output schemas used by the LLM and VLM components. All LLM calls use the parser settings in Table 6. Confidence below 0.6 triggers a constrained retry or the fallback listed in Table 10.

## B.1 LLM: Intent Parsing and Scale Prior

System prompt. You are a careful assistant for a 3D placement system. Output only valid JSON.   
Use only allowed categories. If uncertain, set confidence below 0.6.

User prompt template.

Given a natural-language instruction for placing an object into an indoor scene, extract the support category, a realistic longest object dimension in meters, and a confidence value. The support category must come from ALLOWED\_BASES. Return JSON in the following schema: {base: ..., size\_m: ..., confidence: ...}. Instruction: <PROMPT>. Object category: <OBJ\_CAT>.

Table 7 lists the size ranges used as sanity checks for common object categories.

## B.2 LLM: Target Sub-Part and Anchor

User prompt template.

You are given a placement instruction and the available parts of the support object. Choose a target\_part from PARTS and an anchor from {center, left, right, head, foot, front, back, top}. Return only JSON in this schema: {target\_part: ..., anchor: ..., confidence: ...}. Instruction: <PROMPT>. Parts: <PARTS>.

Table 7: Size sanity ranges.
<table><tr><td>Object category</td><td>Min (m)</td><td>Max (m)</td></tr><tr><td>Ball</td><td>0.04</td><td>0.30</td></tr><tr><td>Pillow</td><td>0.30</td><td>0.80</td></tr><tr><td>Plush toy</td><td>0.12</td><td>0.65</td></tr><tr><td>Towel</td><td>0.35</td><td>1.60</td></tr><tr><td>Shirt / clothing</td><td>0.45</td><td>1.20</td></tr><tr><td>Dumbbell</td><td>0.20</td><td>0.55</td></tr><tr><td>Book</td><td>0.18</td><td>0.42</td></tr><tr><td>Other / unknown</td><td>0.08</td><td>1.00</td></tr></table>

If the selected target is invalid or confidence is low, the fallback chooses the largest upward-facing part compatible with the support category and uses the center anchor.

## B.3 VLM: Three-Candidate Orientation Selection

Model call. Orientation selection uses the upstream VLM in Table 6, not the VLM evaluator used for final scoring. Each candidate is rendered from the same oblique camera with the support held fixed, so the model only chooses among coarse object orientations.

## VLM prompt template.

You will see three images of the same object above the same support. The support is fixed; only the object orientation changes. Choose the orientation that best satisfies the instruction. Consider the up-facing side, the side facing the target region, and whether the object should lie flat or stand. Return only JSON: {best\_view: 1|2|3, confidence: ..., rationale: ...}. Instruction: <PROMPT>.

## B.4 LLM: Material Mapping

## Material prompt template.

Map each part label to one material from MATERIAL\_IDS. Use default\_unknown when uncertain and lower the confidence. Return only JSON: {part\_to\_material: {label: material\_id, ...}, confidence: ...}. Parts: <PARTS>. Materials: <MATERIAL\_IDS>.

## C Material Library

The material library uses a discrete set of stable simulator parameters. Table 8 lists the density, elastic modulus, Poisson ratio, and friction values used in the simulator. These are category-level approximations, not material-identification measurements. A fixed synonym dictionary is applied before the LLM material call: mattress, cushion, blanket, towel, clothing, and plush labels map to compliant IDs; wood, metal, glass, ceramic, and structural frame labels map to rigid IDs. The LLM resolves remaining ambiguous labels using the prompt in Appendix B. If confidence is below 0.6, DCAP uses default unknown, records a low-confidence material flag, and keeps the part rigid unless the support category is explicitly soft.

## D Segmentation and Part Decomposition

The segmentation report records the backend, view count, projection rule, confidence threshold, and synonym taxonomy. For each support category, we report base-instance success and part success. We also report fallback rate and the main failure source.

Table 8: Curated material library. Density is in kg/m<sup>3</sup> and Young’s modulus is in MPa.
<table><tr><td>Material ID</td><td>Density</td><td>Young&#x27;s modulus</td><td>Poisson ratio</td><td>Static friction</td><td>Dynamic friction</td></tr><tr><td>soft foam</td><td>35</td><td>0.08</td><td>0.30</td><td>0.70</td><td>0.55</td></tr><tr><td>firm foam</td><td>80</td><td>0.35</td><td>0.30</td><td>0.75</td><td>0.60</td></tr><tr><td>cotton fabric</td><td>250</td><td>0.02</td><td>0.35</td><td>0.65</td><td>0.50</td></tr><tr><td>wool fabric</td><td>180</td><td>0.015</td><td>0.35</td><td>0.70</td><td>0.55</td></tr><tr><td>leather</td><td>860</td><td>12</td><td>0.35</td><td>0.60</td><td>0.45</td></tr><tr><td>rubber</td><td>1100</td><td>5</td><td>0.49</td><td>0.95</td><td>0.80</td></tr><tr><td>wood</td><td>600</td><td>9000</td><td>0.30</td><td>0.55</td><td>0.40</td></tr><tr><td>plastic</td><td>950</td><td>1500</td><td>0.35</td><td>0.45</td><td>0.32</td></tr><tr><td>glass</td><td>2500</td><td>70000</td><td>0.23</td><td>0.35</td><td>0.25</td></tr><tr><td>metal</td><td>7800</td><td>200000</td><td>0.29</td><td>0.42</td><td>0.30</td></tr><tr><td>ceramic</td><td>2400</td><td>60000</td><td>0.22</td><td>0.40</td><td>0.28</td></tr><tr><td>paper / cardboard</td><td>700</td><td>400</td><td>0.30</td><td>0.55</td><td>0.42</td></tr><tr><td>plush fill</td><td>45</td><td>0.03</td><td>0.35</td><td>0.75</td><td>0.60</td></tr><tr><td>mattress composite</td><td>60</td><td>0.12</td><td>0.30</td><td>0.80</td><td>0.62</td></tr><tr><td>cushion composite</td><td>70</td><td>0.18</td><td>0.30</td><td>0.78</td><td>0.60</td></tr><tr><td>towel stack</td><td>220</td><td>0.04</td><td>0.35</td><td>0.82</td><td>0.65</td></tr><tr><td>folded clothing</td><td>180</td><td>0.05</td><td>0.35</td><td>0.76</td><td>0.58</td></tr><tr><td>default unknown</td><td>500</td><td>5</td><td>0.35</td><td>0.55</td><td>0.40</td></tr></table>

## D.1 Scene Segmentation

The reported implementation renders 48 RGB-D views per scene for semantic transfer, sampled from the 120 reconstruction views in Appendix G. We extract OpenScene features on the mesh, run Segment Anything ViT-H on the rendered RGB frames, and use Mask2Former panoptic labels as category priors. Image masks are projected to visible mesh triangles with z-buffer tests. Each triangle receives the visibility-weighted majority label, and each support instance receives a confidence equal to the mean projected-mask score multiplied by the fraction of visible supporting votes. The default confidence threshold is 0.55. Synonyms such as couch/sofa or bed/mattress are resolved through a fixed taxonomy. If the intended support is not found, DCAP retries with synonyms and then falls back to the largest upward-facing support surface.

## D.2 Part Decomposition

Given the base instance, a category-conditional vocabulary defines the support parts. If a learned part segmenter is unavailable, geometric heuristics propose candidate parts from planar clustering, connectivity, and local orientation. These candidates are then labeled with the constrained vocabulary. Base-instance success is counted when the selected support overlaps the verified support instance with triangle IoU at least 0.25 or the selected anchor lies inside the verified support mask. Part success is counted when the selected target part overlaps the verified part with triangle IoU at least 0.20, or when the final anchor is within 10 cm of the verified target region. Verification uses the benchmark support and part annotations fixed before running any method. Table 9 reports the resulting segmentation and decomposition rates.

## E Submodule Evaluation Details

This appendix reports independent checks used to attribute upstream errors. Raw rates measure first-pass outputs. Post-fallback rates include retries, geometric fallbacks, clamping, and confidence gating. Table 10 lists the fallback triggers used in all experiments, and Table 11 summarizes the module checks.

Table 9: Segmentation and decomposition reporting table.
<table><tr><td>Base category</td><td>Base found</td><td>Parts found</td><td>Fallback rate</td><td>Common failure</td></tr><tr><td>Bed</td><td>90.5%</td><td>83.7%</td><td>7.1%</td><td>Blanket/mattress boundary</td></tr><tr><td>Sofa</td><td>87.8%</td><td>79.4%</td><td>9.5%</td><td>Cushion vs. seat merging</td></tr><tr><td>Chair</td><td>85.2%</td><td>75.6%</td><td>11.0%</td><td>Thin legs and back occlusion</td></tr><tr><td>Shelf</td><td>88.6%</td><td>72.9%</td><td>8.3%</td><td>Small boards and clutter</td></tr><tr><td>Table / desk</td><td>92.1%</td><td>80.8%</td><td>5.6%</td><td>Merged tabletop objects</td></tr><tr><td>Floor / carpet</td><td>94.4%</td><td>84.0%</td><td>4.2%</td><td>Carpet/floor boundary</td></tr></table>

Table 10: Fallback triggers and recovery rules.
<table><tr><td>Module</td><td>Trigger</td><td>Recovery rule</td></tr><tr><td>LLM JSON output</td><td>Invalid schema or missing required key</td><td>Retry twice with parser error; if still invalid, use category prior and mark low confidence</td></tr><tr><td>Support grounding</td><td>Confidence &lt; 0.6 or no support mask above 0.55</td><td>Expand synonyms, rerun support query, then choose largest upward-facing compatible support if available</td></tr><tr><td>Scale prior</td><td>Predicted longest dimension outside category range</td><td>Clamp to Table 7; if clamping changes size  $\mathsf { b y } > 2 \times$  , mark hard upstream failure</td></tr><tr><td>Target part / anchor</td><td>Invalid part, unavailable anchor, or confidence &lt; 0.6</td><td>Choose largest upward-facing compatible part and center anchor</td></tr><tr><td>Orientation</td><td>VLM confidence &lt; 0.55 or tied top candidates</td><td>Use canonical stable orientation with lowest estimated center of mass</td></tr><tr><td>Material mapping</td><td>Confidence &lt; 0.6 or unknown label</td><td>Use dictionary match if possible; otherwise default unknown and record material warning</td></tr><tr><td>Particle filling</td><td>Particle cap exceeded or fewer than 3 layers in a thin region</td><td>Coarsen non-contact particles first; if thin contact still has fewer than 3 layers, flag discretization risk</td></tr><tr><td>Simulation</td><td>Horizon reached without convergence</td><td>Mark invalid final state; if the prompt is infeasible and the rejection criteria are satisfied, count as predicted rejection</td></tr></table>

## F Simulation Details and Convergence Criteria

Table 12 gives the simulator configuration and stopping thresholds. The 107.5 s runtime reported in the main paper is measured on the hardware in Table 6; it includes model-call latency and per-case processing, but excludes the one-time scene reconstruction and asset normalization steps.

## G Benchmark Construction Log

This appendix documents how the deformable placement benchmark was constructed. The goal is not only to collect meaningful prompts. Each case must also contain the information needed for contact evaluation. This includes metric scale and usable support geometry. It also includes a valid object mesh and enough semantics to assign parts, materials, and feasibility labels. All prompt–asset–scene triples are fixed before running any method. All baselines use the same triples and scales. They also use the same scene geometry, camera views, and prompts.

Table 11: Submodule evaluation. Raw rates measure the first-pass module output; post-fallback rates include recovery and confidence gating.
<table><tr><td>Submodule</td><td>Metric</td><td>Raw</td><td>Post-fallback</td></tr><tr><td>Base support selection (LLM)</td><td>Top-1 accuracy</td><td>91.8%</td><td>96.4%</td></tr><tr><td>Target sub-part selection (LLM)</td><td>Accuracy</td><td>84.6%</td><td>91.2%</td></tr><tr><td>Size prior sanity (LLM)</td><td>Within valid range</td><td>93.2%</td><td>98.1%</td></tr><tr><td>Material mapping (LLM)</td><td>Human- agreement / safe default</td><td>81.4%</td><td>94.7%</td></tr><tr><td>Orientation (VLM)</td><td>Accuracy / consistency</td><td>88.9%</td><td>92.1%</td></tr><tr><td>Segmentation (base instance)</td><td>Success rate</td><td>86.7%</td><td>93.8%</td></tr><tr><td>Part decomposition</td><td>Success rate</td><td>78.3%</td><td>88.5%</td></tr><tr><td>Triggered fallback recovery</td><td>Recovered failed upstream cases</td><td></td><td>62.5%</td></tr></table>

Table 12: Simulation configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Simulator / version</td><td>Genesis 0.2.1-compatible rigid–MPM backend with custom two-way contact kernels</td></tr><tr><td>Global timestep</td><td>1/240 s</td></tr><tr><td>Substeps</td><td>8</td></tr><tr><td>Gravity</td><td>9.81  $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$  along the negative vertical axis</td></tr><tr><td>Initial clearance δ</td><td>0.03 m</td></tr><tr><td>Rigid-deformable coupling</td><td>Two-way frictional contact with velocity projection</td></tr><tr><td>Friction model</td><td>Coulomb friction with material-specific static/dynamic coefficients</td></tr><tr><td>Elasticity model</td><td>Corotated linear elasticity for MPM soft parts</td></tr><tr><td>Particle spacing range</td><td>Adaptive 0.8–2.5 cm</td></tr><tr><td>Particle cap</td><td>250k particles per case</td></tr><tr><td>Max simulation horizon</td><td>4.0 s</td></tr><tr><td>Convergence window</td><td>0.5 s / 120 global steps</td></tr><tr><td>Rigid speed threshold</td><td>0.02 m/s</td></tr><tr><td>Angular speed threshold</td><td>0.05 rad/s</td></tr><tr><td>Particle speed threshold</td><td>0.03 m/s</td></tr><tr><td>COM drift threshold</td><td>0.01 m over the convergence window</td></tr><tr><td>AABB change threshold</td><td>0.5% of relative diagonal length</td></tr></table>

Table 13: Diagnostic tolerances used for feasibility decisions.
<table><tr><td>Diagnostic</td><td>Tolerance</td></tr><tr><td>Mean penetration depth</td><td> $\eta _ { \mathrm { p e n } } = 1 . 0 \ : \mathrm { c m }$ </td></tr><tr><td>Target drift</td><td> $\eta _ { \mathrm { d r i f t } } = 0 . 1 5 \ : \mathrm { o }$  f the support OBB diagonal</td></tr><tr><td>Support contact ratio</td><td> $\eta _ { \mathrm { c o n t a c t } } = 0 . 5 0$ </td></tr><tr><td>Residual speed</td><td> $\eta _ { \mathrm { r e s } } = 3 . 0$  cm/s, with component-specific convergence thresholds in Table 12</td></tr></table>

Source scenes and asset pool. We start from 214 high-fidelity indoor reconstructions drawn from existing indoor scene resources and embodied-environment datasets (Dai et al., 2017; Chang et al., 2017; Ramakrishnan et al., 2021; Xia et al., 2018; Savva et al., 2019; Kolve et al., 2017). Object assets come from diverse 3D repositories, including Objaverse-style assets and part-oriented assets when available (Deitke et al., 2023; Yang et al., 2024b). We keep meshes with identifiable category labels, plausible metric scale after normalization, and geometry suitable for rendering and collision queries. The final benchmark uses 186 simulation-ready scenes and 155 object assets.

Table 14: Runtime measurement protocol.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Reported statistic</td><td>Mean wall-clock time over the 1,923 standard placement cases LLM/VLM API latency, scene rendering, segmentation lookup, target</td></tr><tr><td>Included stages</td><td>anchoring, material mapping, particle filling, rollout, diagnostics, and final rendering</td></tr><tr><td>Excluded stages</td><td>One-time TSDF/Poisson reconstruction, dataset download, manual benchmark annotation, and cached asset normalization</td></tr><tr><td>Parallelism</td><td>One case per GPU during simulation; CPU preprocessing uses 16 workers; model calls are issued synchronously per case for deterministic logs</td></tr><tr><td>Timing source</td><td>Python wall-clock timers around each pipeline stage, with CUDA synchronization before and after simulation kernels</td></tr></table>

Scene conversion. Each source scene is converted to a simulation-ready mesh with a deterministic post-processing pipeline. We render 120 RGB-D views per scene on a hemispherical indoor camera path. The depth observations are fused into a volumetric representation, and a mesh is extracted with screened Poisson reconstruction at 2 cm nominal resolution (Curless and Levoy, 1996; Kazhdan et al., 2006; Kazhdan and Hoppe, 2013). We then remove small disconnected components, fill bounded holes, check scale consistency, and transfer semantic labels when available. The same mesh is used by all methods for rendering, surface projection, collision checks, and physical diagnostics. Table 15 summarizes the conversion and quality-control settings.

Quality-control criteria. A scene is accepted only if at least one target support category can be localized with enough geometry for contact evaluation. We reject scenes with ambiguous scale or damaged target regions. We also reject scenes with floaters near likely contacts, fragmented support geometry, or missing prompt supports. We also exclude scenes where semantic transfer prevents reliable support or part selection after synonym fallback. These filters are stricter than rendering-only filters because small geometric defects can dominate contact diagnostics.

Support categories and coverage. The benchmark covers common indoor supports across soft, mixed, and rigid contact. Beds and sofas emphasize compliant surfaces. Chairs and carpets create mixed contacts. Shelves, tables, and desks provide rigid controls. Table 16 summarizes the final composition.

Prompt construction. Prompts are written to exercise both semantic grounding and physical realization. The in-distribution set includes support-only prompts, sub-part prompts, orientationsensitive prompts, and material/contact-sensitive prompts. Examples include “place a plush toy on the sofa,” “place a dumbbell on the center of the mattress,” and “place a book on the top shelf.” Each prompt is paired with an intended support. When needed, it also stores a target part and an anchor token.

The stress split is constructed separately. Feasible OOD prompts are unusual relative to the standard distribution but physically satisfiable under the reconstructed geometry and material assumptions. Infeasible prompts are annotated as unsupported, oversized, unstable, or impossible under the modeled scene. These cases test whether a method can reject invalid instructions rather than forcing every prompt into a placement.

Annotation instructions for data labelers. Labelers received clean scene views, reconstructed mesh views, an asset preview, the object category, and the prompt. They annotated each case before any method was run. They first selected the intended support instance from the fixed support vocabulary. They then marked a target part and anchor when the prompt specified one. If the prompt was coarse, they chose the most natural upward-facing region of the intended support.

A feasible case had to satisfy three checks. The target support had to exist in the reconstructed mesh. The scaled object had to fit the target region. A stable resting state had to be plausible without hidden supports or impossible deformation. If any check failed, the case was labeled infeasible. The primary reason used one of six labels. The labels were unsupported target, oversized object, unstable support, missing geometry, ambiguous target, and contradictory instruction. Labelers never inspected outputs from DCAP or any baseline.

Table 15: Mesh conversion and quality-control log.
<table><tr><td>Item</td><td>Setting / outcome</td></tr><tr><td>Source scenes</td><td>214 candidate indoor reconstructions</td></tr><tr><td>Accepted scenes</td><td>186 simulation-ready scenes</td></tr><tr><td>Rendered views</td><td>120 RGB-D views per scene on a hemispherical indoor camera path</td></tr><tr><td>Mesh extraction</td><td>TSDF fusion followed by screened Poisson meshing at 2 cm nominal resolution</td></tr><tr><td>Hole repair</td><td>Fill bounded holes below 0.15 m diameter</td></tr><tr><td>Component filtering</td><td>Remove disconnected components below 0.5% of scene area</td></tr><tr><td>Scale calibration</td><td>Camera metadata plus furniture-size priors; bed and table dimensions used for sanity checks</td></tr><tr><td>Semantic transfer</td><td>Multi-view open-vocabulary masks projected to triangles with</td></tr><tr><td>Excluded scenes</td><td>visibility-weighted voting 28 scenes removed for non-watertight supports, ambiguous scale, severe floaters, or missing target supports</td></tr></table>

Table 16: Benchmark support-category split. Counts are prompt–asset–scene cases.
<table><tr><td>Support group</td><td>Cases</td><td>Scenes</td><td>Assets</td><td>Coverage focus</td></tr><tr><td>Bed / mattress</td><td>462</td><td>42</td><td>31</td><td>Soft supports; heavy rigid and plush objects</td></tr><tr><td>Sofa / cushion</td><td>407</td><td>37</td><td>29</td><td>Soft cushions, arms, backs, and seated supports</td></tr><tr><td>Chair / seat</td><td>310</td><td>31</td><td>23</td><td>Mixed rigid-soft seats and backs</td></tr><tr><td>Shelf / table / desk</td><td>504</td><td>42</td><td></td><td>34 Rigid supports, narrow</td></tr><tr><td>Floor / carpet</td><td>240</td><td>24</td><td></td><td>anchors, and tabletop controls 20 Mixed carpet and open-floor</td></tr><tr><td>OOD / infeasible</td><td>140</td><td>10</td><td>18</td><td>placement Unusual feasible prompts and</td></tr><tr><td>Total</td><td>2,063</td><td>186</td><td>155</td><td>annotated impossible cases Full benchmark</td></tr></table>

Contact regimes were assigned from the intended support and object material. Cases with mattresses, cushions, blankets, towels, clothing, or plush objects were marked soft when visible deformation was expected. Cases with both compliant and stiff contacts were marked mixed. Shelves, tables, desks, books, balls, and tabletop controls were marked rigid. Ambiguous cases were reviewed by a second labeler and resolved using the same mesh, prompt, and scale rules.

Table 17 gives the reporting splits used in the main experiments.

Asset preparation and scale. Each asset is assigned an object category, a canonical up direction when available, and a plausible real-world scale range. We normalize meshes to metric scale using category-level size priors and clamp the longest dimension. Assets are removed if their geometry is unusable for rendering or collision. For part-composed or deformable assets, semantic part labels are retained when available; otherwise, the asset receives a category-level material assignment. The same normalized scale is used for all methods.

Material and contact-regime labels. Each case receives a soft, mixed, or rigid contact-regime label. The label is determined from the intended support, object category, and material assignment. Soft cases are those where deformation should affect the visible resting state. Mixed cases include both stiff and compliant contact. Rigid cases involve supports where rigid settling is a reasonable approximation. The labels are used only for reporting; methods do not receive them as oracle inputs.

Table 17: Reporting splits and prompt-regime composition.
<table><tr><td>Split / regime</td><td>Cases</td><td>Use in evaluation</td></tr><tr><td>Standard placement split</td><td></td><td>1,923 Feasible in-distribution cases used for aggregate placement quality, VLM scoring, human preference,</td></tr><tr><td>Soft-contact regime</td><td>869</td><td>and objective physical diagnostics Cases dominated by compliant support or object be- havior, such as mattresses, cushions, blankets, towels,</td></tr><tr><td>Mixed-contact regime</td><td></td><td>clothing, and plush objects 550 Cases involving both rigid and compliant contact, such as chairs, carpeted floors, and partially soft sup-</td></tr><tr><td>Rigid-contact regime</td><td></td><td>ports 504 Rigid-support controls such as shelves, tables, desks, books, balls, and stable tabletop placement</td></tr><tr><td>Stress split</td><td></td><td>140 OOD or infeasible cases evaluated separately for fea- sibility behavior</td></tr><tr><td>Feasible OOD</td><td>64</td><td>Unusual but physically satisfiable prompts; returned feasible placements receive diagnostics</td></tr><tr><td>Annotated infeasible</td><td>76</td><td>Unsupported, oversized, unstable, or impossible prompts; correct rejections count only in feasibil- ity metrics</td></tr></table>

Case records. Each benchmark case stores the scene ID, object asset ID, prompt, intended support, target region when specified, anchor when specified, contact regime, feasibility label, and infeasibility reason. The same record stores the normalized object scale, the support instance used for evaluation, the canonical render views, and the diagnostic tolerances.

Evaluation denominators. We use two reporting denominators. The 1,923-case standard split is used for aggregate placement quality, human/VLM preference, and physical diagnostics. The 140-case stress split is evaluated separately. Returned feasible placements on stress cases receive the same diagnostics as standard placements. Correct rejections count only in feasibility metrics because they do not have final placement states.

Leakage control and baseline consistency. Benchmark construction is completed before running DCAP or any baseline. No method output is used to accept, reject, or relabel a case. All baselines receive the same prompt, object scale, scene mesh, and render views. Static baselines are evaluated for semantic placement, penetration, and support contact. Dynamic quantities are reported only for variants with an explicit physical rollout.

## H Rendering Protocol

All rendered comparisons use the same cameras, lighting, and object scale for every method on a case. The canonical VLM and human views are fixed before scoring: one prompt-facing overview, two oblique side views at ±35<sup>◦</sup>, one elevated front view at 45<sup>◦</sup>, and one support-level close-up centered on the target region. For qualitative figures, the same camera family is used, but crops are adjusted only to keep the object and support visible. Method names are removed from the images sent to the VLM evaluator and to human participants.

## I Baselines and Reimplementation Details

## I.1 FirePlace Reimplementation

The FirePlace-style baseline uses a language model to propose a support region. It samples candidate placements on upward-facing surfaces, filters candidates with excessive collision, and selects the best remaining candidate using a plausibility score. It uses the same object scale and scene assets as DCAP to isolate the effect of physical settling.

Table 18: Rendering settings for method visualization and evaluation.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Renderer</td><td>Blender 4.2.3, Eevee Next rasterization, ambient occlusion enabled</td></tr><tr><td>Resolution</td><td>1024× 1024 PNGs for VLM scoring; 768×768 panels for human trials; lossless PNG storage</td></tr><tr><td>Camera set</td><td>Five fixed canonical views per case: overview, left oblique, right oblique, elevated front, and target close-up</td></tr><tr><td>Lighting</td><td>Constant HDRI-style environment light plus weak area fill; no method-specific shadows or retouching</td></tr><tr><td>Normalization</td><td>Same object scale, material color, target support mesh, and camera intrinsics across all compared methods</td></tr><tr><td>Blinding</td><td>Human and VLM comparison panels randomize left/right order and omit method names</td></tr></table>

## I.2 LayoutGPT and Holodeck Adaptation

LayoutGPT and Holodeck outputs are mapped into the reconstructed scene coordinate frame by aligning axes and scale. If a method predicts abstract layout coordinates rather than mesh-surface anchors, the prediction is projected to the nearest valid support surface before rendering. We do not add deformable settling to these baselines unless it is explicitly reported as a hybrid variant. Table 19 summarizes the adaptation settings.

Table 19: Baseline adaptation settings.
<table><tr><td>Method</td><td>Main adaptation</td><td>Collision handling</td><td>Notes</td></tr><tr><td>FirePlace reimplementation</td><td>Same object scale as DCAP; upward-surface sampling and</td><td>Reject candidates with SDF penetration above 2 cm; no deformable settling</td><td>Strongest rigid placement baseline</td></tr><tr><td>LayoutGPT</td><td>plausibility scoring Abstract layout coordinates mapped to scene OBB anchors</td><td>Project to nearest valid support; shallow collision rejection</td><td>Good semantic intent, weak contact modeling</td></tr><tr><td>Holodeck</td><td>Scene-generation priors adapted to existing reconstructions</td><td>Nearest-support projection and AABB-overlap rejection</td><td>Strong indoor commonsense but sensitive to coordinate mismatch</td></tr></table>

## J Human Study Protocol

Participants compare two placements under a prompt. They also see the clean scene and the isolated object asset. The displayed axis is randomized between Right Place and Physics & Naturalness. Left/right order is randomized, and trials are balanced across method pairs. The interface uses the rendered panels described in Appendix H. Quality-control trials detect participants who ignore the prompt or consistently select physically impossible placements. Table 20 records the study counts and aggregation settings.

## K VLM Evaluation Protocol

The VLM evaluator is gemini-3-pro-preview, while the upstream orientation and JSON modules use gpt-5.2. This separation reduces direct evaluator–method coupling. The evaluator receives the prompt and the canonical renderings from Appendix H. It returns separate scores for Right Place and Physics & Naturalness, plus a short rationale. A separate pairwise prompt asks which of two results is better on the specified axis. Method order is randomized, method names are hidden, and the evaluator never receives diagnostic values such as penetration depth or residual speed.

Table 20: Human study reporting fields.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Number of participants</td><td>42</td></tr><tr><td>Total trials</td><td>2940</td></tr><tr><td>Preference trials</td><td>2520</td></tr><tr><td>Attention-check trials</td><td>420</td></tr><tr><td>Trials per participant</td><td>70, including 60 preference trials and 10 attention checks</td></tr><tr><td>Attention-check rate</td><td>14.3% of total trials (420 total checks; 10 per participant)</td></tr><tr><td>Exclusion rule</td><td>Exclude participants failing more than 2 of 10 attention checks or completing below one-third median time</td></tr><tr><td>Aggregation model</td><td>Bradley-Terry-Luce pairwise preference model</td></tr><tr><td>Bootstrap procedure</td><td>10,000 paired bootstrap resamples grouped by prompt</td></tr><tr><td>Significance test</td><td>Two-sided bootstrap confidence intervals with Holm correction</td></tr></table>

Rubric excerpt. Right Place score 10 means the object is on the correct support, in the specified sub-region, and oriented consistently with the prompt. Physics & Naturalness score 10 means the object appears stably supported, the contact is plausible, the deformation matches the material, and no obvious floating or deep penetration is visible.

## L Additional Quantitative Metrics

We compute diagnostics after final settling or after the baseline placement is rendered.

• Penetration depth and fraction. We sample object-surface points and compute signed distance to $\widetilde { S } _ { b }$ for rollout methods, or $S _ { b }$ for static baselines. We report the mean magnitude of negative distances in centimeters and the inside-support fraction; valid soft-support indentation is not counted as penetration.

• Residual speed. We record rigid linear speed, equivalent angular surface speed, and particle speed at termination. The reported value is the maximum of these speeds in centimeters per second.

• Target adherence. We project the final object center of mass onto the target support frame and measure its distance to the selected anchor, normalized by the base OBB diagonal.

• Support contact. We estimate the fraction of near-contact object samples whose nearest support normal is upward-compatible.

• Feasibility quality. For feasible and infeasible prompts, we report standard confusion-matrix rates.

## M Bootstrap Uncertainty

We compute uncertainty estimates with a paired bootstrap over the 1,923 standard-split prompt–asset– scene cases. Each bootstrap resample keeps method outputs paired within the same case. Table 21 reports 95% confidence intervals for the main Table 2 deltas.

## N Additional Qualitative Results and Failure Cases

Additional qualitative results include success grids and annotated failures. The failure taxonomy covers grounding, target-part, and orientation errors. It also covers material ambiguity, particle-fill failure, and geometry defects. Non-convergence and true infeasibility form the final group. Each failure includes the prompt, predicted support/part/material, diagnostic flags, and the mitigation used or proposed.

Table 21: 95% bootstrap confidence intervals for key standard-split deltas. Positive penetration and speed drops mean lower values for DCAP.
<table><tr><td>Metric</td><td>Contrast</td><td>Delta</td><td>95% CI</td></tr><tr><td>VLM P/N</td><td>DCAP full minus FirePlace raw</td><td>+21.2</td><td>[19.6, 22.8]</td></tr><tr><td>Human P/N</td><td>DCAP full minus FirePlace raw</td><td>+26.5</td><td>[22.1, 30.8]</td></tr><tr><td>Penetration depth</td><td>FirePlace raw minus DCAP full</td><td>+1.07 cm</td><td>[0.96, 1.19]</td></tr><tr><td>Support contact</td><td>DCAP full minus FirePlace raw</td><td>+0.24</td><td>[0.21, 0.27]</td></tr><tr><td>Residual speed</td><td>FP + rigid minus DCAP full</td><td>+4.9 cm/s</td><td>[4.2, 5.6]</td></tr><tr><td>VLM P/N</td><td>DCAP full minus FP + def.</td><td>+12.5</td><td>[10.9, 14.0]</td></tr><tr><td>VLM P/N</td><td>DCAP full minus DCAP rigid</td><td>+8.2</td><td>[6.8, 9.6]</td></tr><tr><td>Convergence</td><td>DCAP full minus DCAP rigid</td><td>+0.09</td><td>[0.06, 0.12]</td></tr></table>

## O Supplementary Videos Index

Table 22 lists the supplementary rollout videos.

Table 22: Supplementary video index.
<table><tr><td>Video ID</td><td>Scene / prompt</td><td>Compared methods</td><td>Notes</td></tr><tr><td>V6589283</td><td>Bed: dumbbell on the center of the mattress</td><td>DCAP / FirePlace</td><td>Indentation and stable heavy-object contact</td></tr><tr><td>V3459802</td><td>Sofa: plush toy on the left cushion</td><td>DCAP / FirePlace / Holodeck</td><td>Soft support with mild plush deformation</td></tr><tr><td>V2349234</td><td>Shelf: book on the top shelf</td><td>DCAP / LayoutGPT</td><td>Rigid-support control case</td></tr><tr><td>V9097692</td><td>Chair: towel draped over the seat and back</td><td>DCAP / Holodeck</td><td>Soft-on-rigid settling and folding behavior</td></tr><tr><td>V4204730</td><td>OOD: oversized ball on a narrow shelf</td><td>DCAP / all baselines</td><td>Infeasibility and drift diagnostics</td></tr></table>

## P Broader Impacts

DCAP targets physically grounded synthetic placement. Misuse or deployment errors can affect embodied systems if simulated contacts are treated as real-world guarantees. Sim-to-real gaps in material parameters, reconstruction quality, or friction can lead to unsafe robot training data or incorrect manipulation policies. The benchmark also inherits biases from indoor scene datasets and asset repositories. Scene content, room layouts, object categories, and household materials may overrepresent some environments and underrepresent others. Source reconstructions and assets should be used under their original licenses and privacy constraints. Finally, plausible but physically wrong placements can mislead downstream embodied agents. We therefore report feasibility flags and objective diagnostics, and we recommend using DCAP outputs as simulation evidence rather than real-world safety guarantees.