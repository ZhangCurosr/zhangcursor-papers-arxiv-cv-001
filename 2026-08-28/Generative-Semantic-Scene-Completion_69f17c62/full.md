# Generative Semantic Scene Completion

Shi Chen and Weifeng Ge

Abstract—Outdoor LiDAR semantic scene completion (SSC) recovers a dense semantic voxel grid from a scan observing 1% of the target volume, under class imbalance beyond 7,000×. We recast SSC as generative semantic scene completion (GSSC): a single discrete-diffusion formulation in three roles. First, paired sparse–dense scene synthesis (PS<sup>3</sup>) generates matched sparse LiDAR observations with their dense semantic completions, addressing the long tail at its source and yielding the PS<sup>3</sup>- SemanticKITTI corpus we train on alongside SemanticKITTI. Second, semantic-guided generative scene completion (SGSC) generates the scene from noise with multinomial discrete diffusion, conditioned on the sparse scan through a bird’s-eye-view semantic map and a sparse 3D feature stream. Third, the same framework instead refines an existing completion in one flow-matching step: structured source discrete diffusion (S<sup>2</sup>D<sup>2</sup>). S<sup>2</sup>D<sup>2</sup> improves the mIoU of SGSC’s own output and every external SSC base tested, without base retraining or test-time adaptation. On the strongest base, one step without test-time augmentation reaches 38.8% mIoU on the SemanticKITTI hidden test. To our knowledge that is the best causal, single-sweep, single-sample result on that leaderboard, +2.1 pp over the previous best published score under the same restriction. Four correction steps with eight-view test-time augmentation reach 39.2%, outside that restriction.

Index Terms—Semantic scene completion, discrete diffusion models, LiDAR point clouds, synthetic training data, bird’s-eye view perception, autonomous driving, 3D scene understanding

## I. INTRODUCTION

EMANTIC scene completion (SSC) is a key element of S 3D perception for autonomous driving. A vehicle cannot plan against the surfaces its sensor happens to see: it must reason about the space behind a parked car and the road that continues past the last returned point [2], [3]. Supervised networks learn to supply that volume because one benchmark labels it. SemanticKITTI [4] aggregates a whole sequence of scans into each target grid, so a voxel occluded in the current frame is still labelled if a later scan reached it. That construction also fixes the class histogram to whatever the recording vehicle drove past. Vegetation outnumbers motorcyclist by more than three orders of magnitude, and per-class accuracy follows. Our class frequency and the previous method’s published hidden test per-class IoU correlate strongly [1], so the classes a planner most needs are the ones the training set supplies least. The sensor compounds this. One sweep returns about one voxel in a hundred, and everything behind, inside and beyond what it struck has to be inferred, which is where rare classes, thin structures, ghost trails, boundary errors and occlusion all surface (Fig. 1).

Two remedies address the imbalance from inside that corpus. The first reweights the objective (class weights, or focal loss [5]), which changes how much each existing example counts, not how many exist. The second extracts more supervision from the same sequences (Section II). Neither adds a scene, so both work against a histogram that does not move.

Recognition and completion are not two stages for a person: we label what is visible while hallucinating what is not, the perceptual filling-in that Gestalt psychology named amodal completion [6], [7]. Feynman’s “What I cannot create, I do not understand” states what that asks of a model. A discriminative network predicts labels for what it sees; a generative one has to learn the structure of what could exist.

Generating 3D data is no longer the obstacle (Section II), but the two families that could supply one each leave a gap. An unconditional generator samples a complete labelled grid from noise, with no sparse observation to be completed from and the corpus histogram intact. A conditional one completes from noise, rebuilding the ground plane and the road each time even where a deterministic prediction already places them.

We propose to close both gaps with a single discrete-diffusion formulation in three roles, and call the framework generative semantic scene completion (GSSC). Paired sparse–dense scene synthesis (PS<sup>3</sup>) gives the generated scene an observation. Pyramid discrete diffusion synthesises the complete grid, a rare-class object bank pastes in the classes the generator seldom places, and a Velodyne HDL-64E ray-casting model resamples the result into the sparse scan that would have produced it. The amplification a class receives therefore follows the bank’s supply rather than the class’s rarity. Semantic-guided generative scene completion (SGSC) covers the case where no base network exists to start from, diffusing from pure categorical noise conditioned on bird’s-eye-view semantics and frozen LiDAR-segmentation features read off the sweep. Structured source discrete diffusion (S<sup>2</sup>D<sup>2</sup>) replaces that noise source with a frozen network’s own prediction and learns a deterministic transport from it to the ground truth, so nothing is rediscovered and one step suffices at deployment.

Alongside the framework we report the analyses an adopter would need. On the SemanticKITTI hidden test one sampling step with no test-time augmentation reaches 38.8% mIoU, +2.1 pp over SCPNet’s published 36.7%, indexed under a predicate that fixes what one prediction may use: one sweep, no future moments, no ensembling. That predicate excludes multi-sweep entries (SCPNet at four sweeps publishes 47.5%), test-time adaptation (TALoS reaches 37.9% by tuning the forward pass’s weights on other moments, future ones included), and a four-step, eight-view ensemble of our own that would otherwise read 39.2%. We refine three frozen bases spanning the field without retraining any, train on the synthetic pool, and test the frozen checkpoint zero-shot on two further datasets (Section IV).

![](images/4cb7e78be5c75e5c39f6a694c0fce746f412340ad3cf751d6f339cdb314778e9.jpg)

![](images/c7bf6c892be1134aa11993342e66d024673d83b5f9b1074ea17b5a52047327d9.jpg)  
Fig. 1. The challenge of semantic scene completion. (a) SSC infers dense structure and semantics from a sparse sweep. (b) Five failure modes of discriminative methods; the percentages are measured on the validation predictions of SCPNet [1]. (c) SemanticKITTI is severely long-tailed, vegetation 26.7% of labelled points against motorcyclist 0.0037%. (d) Our class frequency (log axis) and SCPNet’s published hidden-test IoU correlate at Pearson r=0.80.

In sum, our main contributions are the following:

• We recast semantic scene completion as generative modelling, driven by a single discrete-diffusion formulation that synthesises training data (PS<sup>3</sup>), completes scenes (SGSC), and refines existing completions $( \mathsf { S } ^ { 2 } \mathbf { D } ^ { 2 } )$

• Refining a frozen network’s prediction in a single step, with no retraining, posts, to our knowledge, the best causal, single-sweep, single-sample result on the SemanticKITTI hidden test.

• We give the exactness identity and the non-amplifying error bound behind one-step deployment—viable rather than provably sufficient—and state where each stops short of the shipped schedule.

• Our paired synthesis yields PS<sup>3</sup>-SemanticKITTI, a synthetic corpus of sparse–dense pairs with amplified rare classes, which improves completion on the voxel-grid-native base we test when trained alongside the real data (Section IV-D).

• We identify the gaps to be addressed next: thin structures barely move (Section IV-A), and no experiment here isolates occlusion.

Every name and metric we coin is also defined together on page 1 of the supplementary material.

## II. RELATED WORK

SSC predicts occupancy and semantics jointly across a volume the sensor barely observes. Song et al. [8] posed it indoors, from a single depth image. We group the outdoor work by what each method changes, and treat the generative approaches separately below.

Discrete diffusion. The variant this paper builds on corrupts categorical variables rather than continuous ones. D3PM [9] and the multinomial formulation of Hoogeboom et al. [10] replace Gaussian noise with a transition matrix over a label alphabet, so a voxel’s class is resampled rather than perturbed. A categorical process therefore fits a semantic scene where a continuous one must quantise at the end. An instance is fixed by its source and its kernel (Section III-A); conditioning the reverse process on an observation turns generation into completion.

Changing the network. One line treats completion as voxel labelling, differing mainly in representation: a trunk without 3D convolution [11], or a continuous field of local deep implicit functions in place of voxels [12]. Further variants fuse a bird’s-eye-view completion with a sparse-voxel one [13], merge a 3D segmentation branch into a bird’s-eye-view completion branch [14], couple completion to a segmentation backbone through point–voxel interaction [15], or separate semantics from geometry [16]. No representation changes what the sensor gives: about one voxel in a hundred, and unlabelled, so geometry must be inferred almost everywhere and semantics everywhere. Changing the supervision. A second line changes what the network learns from: SCPNet [1] distils a multi-frame teacher, TALoS [17] adapts at test time under line-of-sight constraints, and PaSCo [18] adds instance supervision. Each gain is written into the base network’s own weights.

Changing the data. A fourth line changes what the model trains on: CarlaSC [19] collects paired scans in a simulator. Pyramid discrete diffusion [20] generates outdoor scenes outright. Neither meets a long-tailed benchmark’s needs: CarlaSC’s scenes are CARLA’s [21], not SemanticKITTI’s, and an unconditional generator reproduces the distribution it was trained on, so it too seldom places the rare classes, and it supplies no matching sparse scan. Two nearer lines exist: discrete latent representations of LiDAR serve both completion and the generation of new scans [22], and pasting instances between scans is long-standing practice in LiDAR training [23], [24]. Our object bank belongs to that lineage, carried to scene scale and into the voxel label space.

In contrast to previous works, we do not seek a better representation, richer supervision written into a base network’s own weights, or an input built for dense image features. We learn a distribution over complete labelled scenes and read all three jobs off it, under two sources and three conditioning designs. Spanning several of those jobs from one generative model is not unique to us: Octree Diffusion [25] generates, completes and extends semantic scenes from one octree latent, conditioned on partial scans at inference without retraining. Ours is the third job, correcting a frozen base inside the label space it is scored in. The fourth axis we occupy from the other end: rather than collecting scenes we generate them, together with the observation that would have produced them, which is also LidarDM’s [26] architecture. $\bar { \mathrm { P S ^ { 3 } } }$ differs in purpose, supplying paired supervision with the rare classes amplified. Diffusion for 3D scenes. Object-level 3D diffusion is mature: Point-Voxel Diffusion [27], LION [28] and the point-cloud diffusion of Luo and Hu [29] generate object shapes. At scene scale, DiffuScene [30] synthesises indoor layouts, and LiDM [31] diffuses an outdoor scene in a latent space. In outdoor semantic scene completion, DiffSSC [32] and OccGen [33] run continuous Gaussian diffusion to produce a scene, leaving the label space; Lee et al. [34] stay in it and complete a scene from a sparse point cloud.

The from-noise setting is therefore not ours to claim: that work already reaches it inside the label space. What SGSC changes is the conditioning. Where they read the sparse cloud, we read the bird’s-eye-view semantics projected from it together with a sparse 3D feature stream. We report what each stream is worth under an oracle BEV rather than only in the regime we deploy.

Two gaps run through the outdoor-completion methods above: none supplies the matching sparse scan a completion pair needs, and none refines a scene at the resolution and label space it will be scored in. Refining an existing prediction is not itself new: SemCity [35] enhances completion outputs through a continuous triplane diffusion prior where we diffuse the discrete field itself, LiDiff [36] completes a single scan by diffusing points. ESSC-RM [37] appends a refinement module to camerabased backbones, and OccFiner [38] refines camera occupancy offboard from several frames, outside the predicate we index on. $\mathrm { P S } ^ { 3 }$ closes the first, driving the pyramid generator [20], pasting in the rare classes a generator seldom places, and resampling the result into the scan that would have observed it; $\mathrm { S } ^ { 2 } \bar { \mathrm { D } } ^ { 2 }$ closes the second by taking an existing prediction as its source.

Diffusion from a structured source. Diffusion is not solely generative. One use corrects a model’s output, as RefineCat-Diff [39] refines an initial-stage segmentation in medical imaging. Replacing the noise prior with a data-dependent endpoint is established: ResShift [40] diffuses along an image residual; bridge models such as I<sup>2</sup>SB [41], BBDM [42] and DDBM [43] transport between two given endpoints. PDE-Refiner [44] renoises a solver’s own earlier prediction, and Cold Diffusion [45] shows generation survives degradations that are not noise at all, the deterministic regime that $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ inherits. InDI [46] iterates directly from the degraded input rather than from noise, on continuous images.

In the discrete setting, SegRefiner [47] runs a diffusion whose terminal state is a base model’s coarse mask, and transports away from that mask stochastically. Discrete flow matching [48], [49] brings flow matching to categorical data, also with a stochastic kernel, and starts from a mask or uniform source distribution.

In this work we set the diffusion source to a frozen network’s prediction and read the corruption as a deterministic interpolant between the two endpoints. Training still draws categorically around that path, but the sampler runs on the path itself, so the reverse transports the source toward the ground truth instead of rebuilding the scene around it. Determinism is what makes the error bound telescope rather than compound. At deployment it collapses onto one residual plus a small schedule offset, which makes a single Euler step viable rather than provably sufficient. The operator trains across the full schedule and deploys at one, so no distillation stage stands between training and inference, and it leaves every base weight untouched, so the gain cannot be a retrained base in disguise. Whether the multi-step path buys the gain is a separate question: our control holds the loss fixed but still cannot separate the schedule from the variational term it carries, so we scope that claim to the path as a whole. What the operator learns is the correction, not the scene.

## III. METHODOLOGY

Semantic scene completion aims to predict a complete 3D semantic voxel grid $\bar { \mathbf { Y } } \in \{ 0 , 1 , \ldots , \bar { K } - 1 \} ^ { L \times W \times H }$ from sparse LiDAR observations $\bar { \mathbf { X } } \in \{ 0 , 1 \} ^ { L \times W \times \bar { H } }$ , where $K { = } 2 0$ counts 19 semantic classes plus one empty/unlabelled class and $( L , W , H ) = ( 2 5 6 , 2 5 6 , 3 2 )$ for SemanticKITTI [4]. Whereas prior methods regress a single deterministic completion, we learn the distribution of plausible complete scenes and generate completions by structured sampling. The GSSC framework instantiates one discrete-diffusion formulation at three stages spanning two sources—uniform noise and a frozen prediction— and three conditioning designs; SGSC and $\mathrm { S } ^ { 2 } \mathrm { D } ^ { \bar { 2 } }$ share the denoiser architecture but not its weights, and $\mathrm { P S } ^ { 3 }$ runs its own pyramid networks. The three follow in turn. $\mathrm { P S } ^ { 3 }$ (Section III-B) builds the paired data the other two train on; SGSC (Section III-C) completes a scene from noise under LiDARderived conditioning; and $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ (Section III-D) corrects a frozen base’s prediction in one deterministic step, whatever model produced it.

## A. Preliminaries

Everything in this subsection is inherited; Sections III-C and III-D state what we change. We embed a semantic grid onto the product simplex $\begin{array} { r } { \mathcal { M } = \prod , \Delta ^ { K - 1 } } \end{array}$ , sending a label k to the basis vector $\mathbf { e } _ { k } .$ , where $\hat { \Delta } ^ { K - 1 } : = \{ \mathbf { p } \in \mathbb { R } ^ { \bar { K } } : p _ { k } \geq$ $0 , \ \sum _ { k } p _ { k } = 1 \}$ . Multinomial diffusion [10] mixes each pervoxel label toward uniform categorical noise, with $\begin{array} { r } { { \bf u } = \frac { 1 } { K } { \bf 1 } } \end{array}$ the uniform simplex point, $\{ \beta _ { t } \bar  \} _ { t = 1 } ^ { T }$ a mixing schedule, and cumulative retention $\begin{array} { r } { \bar { \alpha } _ { t } : = \prod _ { r = 1 } ^ { t } ( 1 - \beta _ { r } ) } \end{array}$ . A single step keeps the label with probability $1 - \beta _ { t }$ and otherwise resamples it uniformly, $q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) = \operatorname { C a t } ( \mathbf { x } _ { t } \mid ( 1 - \beta _ { t } ) \mathbf { x } _ { t - 1 } + \beta _ { t } \mathbf { u } )$ , with closed-form marginal $q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \operatorname { C a t } ( \mathbf { x } _ { t } \mid { \bar { \alpha } } _ { t } \mathbf { x } _ { 0 } + ( 1 - { \bar { \alpha } } _ { t } ) \mathbf { u } )$ Since $\bar { \alpha } _ { 0 } = 1$ and $\bar { \alpha } _ { T } \approx 0$ , the terminal state $\mathbf { x } _ { T } \approx$ u is the inference prior.

Bayes’ rule along the chain gives the reverse posterior conditioned on the clean scene,

$$
q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) = { \frac { q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { 0 } ) } { q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) } } ,\tag{1}
$$

again categorical and, for $t > 1$ , strictly positive and available in closed form; at $t { = } 1$ it degenerates to the Dirac at $\mathbf { x } _ { \mathrm { 0 } }$ . Since $\mathbf { x } _ { \mathrm { 0 } }$ is unobserved at inference, the standard parameterisation [10] substitutes the denoiser’s simplex-valued estimate $\hat { \mathbf { x } } _ { 0 } : = \hat { p } _ { \theta } ( \cdot |$ $\mathbf { x } _ { t } , t , \mathbf { c } ) \in \Delta ^ { K - 1 }$

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { c } ) : = q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , { \hat { \mathbf { x } } } _ { 0 } ) .\tag{2}
$$

The timestep conditioning we use throughout, adaptive group normalisation [50], is likewise standard. Flow Matching [51], which Section III-D builds on, is likewise inherited: it fits a velocity field to the constant displacement of the straight interpolant ${ \bf z } _ { s } = ( 1 - s ) { \bf z } _ { 0 } + s { \bf z } _ { 1 }$ between a source and a target, minimising $\mathbb { E } \| \mathbf { v } _ { \theta } ( \mathbf { z } _ { s } , s ) - ( \mathbf { z } _ { 1 } - \mathbf { z } _ { 0 } ) \| ^ { 2 }$ . The minimiser is the marginal velocity carrying p<sub>0</sub> to $p _ { 1 }$

## B. Paired Sparse–Dense Scene Synthesis $( P S ^ { 3 } )$

A completion model learns only from paired examples (a sparse sweep aligned to the complete, per-voxel-labelled scene it should recover), and outdoor pairs are expensive and longtailed at once. $\mathrm { P S } ^ { 3 }$ manufactures them offline: it generates the dense scene $\widetilde { { \mathbf Y } } ,$ then derives the sparse observation $\widetilde { \mathbf { X } }$ that would have produced it (Fig. 2; Fig. 3 follows one pair through it). Nothing in the pool depends on the network being trained, so it is reusable augmentation.

a) Scene generation: The full-resolution grid is high dimensional and overwhelmingly empty, so we generate it coarse-to-fine with pyramid discrete diffusion [20]. Three levels of the multinomial diffusion of Section III-A are each conditioned on the trilinearly upsampled scene from the level below, so coarse geometry guides fine detail. The coarsest level, lacking a coarser prior, trains on masked complete scenes instead, under the context patterns of supplementary Appendix G.

b) Distribution filtering: Cascade sampling is fast but imperfect: some raw scenes are nearly empty, single-class, geometrically impossible, or distribution-shifted. Three inexpensive structural checks reject the first three: a plausible occupancy ratio, road plus one further structural class, and a gravityconsistent vertical layout. Each survivor is then screened on the Jensen–Shannon divergence between its class histogram and a corpus prior estimated on held-out real scenes, keeping scenes at $D _ { \mathrm { J S } } \leq \tau { = } 0 . 3 5$ and then the lowest- $D _ { \mathrm { J S } }$ half. The loose $\tau$ constrains class balance and leaves geometry to the structural checks.

c) Rare-class pasting: The generator reproduces the dominant layout faithfully but seldom places the safety-critical classes at all. We therefore graft real instances into each filtered scene, a 3D copy-paste from an offline bank of voxelised instances harvested by connected components over the real training labels. Per scene we visit the rare classes in a fixed order and raise each toward a minimum voxel count, under a paste budget shared across all of them, so the target is attempted rather than met. Every paste must fit the grid, overwrite no other class, and rest on a drivable surface rather than float.

d) Observation rendering: A generated scene enters training only alongside the sparse input a real sensor would return. Subsampling occupied voxels will not do: a rotating LiDAR samples through a fixed beam fan, dense near the sensor and ground, sparse far away and overhead. Our Hardware-Aware LiDAR Observation (HALO) operator marches the 64 non-uniformly spaced HDL-64E beams [52] from a fixed origin through Ye , so the sweep inherits a real scan’s ring structure and range falloff. Two choices serve the long tail: dual returns per beam, as in full-waveform practice [53], recover thin or occluded structures a first-hit tracer erases, and per-return range and angular jitter [54] injects real measurement error. An occupied voxel no sample lands in is marked invalid. An empty one never is. Algorithm 1 states it; sensor calibration, filter thresholds and paste parameters are in Appendix A.

## C. Semantic-Guided Generative Scene Completion (SGSC)

To complete a scene with no base prediction to start from, we introduce SGSC, which casts completion as conditional generation: a multinomial diffusion process carries per-voxel categorical distributions from a uniform noise prior to the complete semantic scene. What the chain reads is the whole of the design, a BEV semantic map and per-voxel segmentation features from a single sweep, so partial semantics reach it before it completes the geometry. No completion model enters the pipeline at any stage.

![](images/4131a80b3aca40b1686517f6da18f5f430a3f488858552f6cdf02a5b5bfdead8.jpg)  
Fig. 2. PS<sup>3</sup>: the offline data-augmentation pipeline. Three cascaded coarse-to-fine multinomial diffusions, screened by a Jensen–Shannon divergence filter; a rare-class object bank and the Hardware-Aware LiDAR Observation (HALO) HDL-64E ray-tracer then turn each synthetic scene into a (sparse, dense) pair. Calibration, thresholds and the pasting pseudocode are in Appendix A of the supplementary material.

![](images/6f15d83d1e9e3c2cf33d404744e362a77cdc279fd9d69677b0d706c2e273775d.jpg)  
Fig. 3. One $\mathbf { P S } ^ { 3 }$ pair. Pyramid diffusion generates a scene $( S _ { 1 }$ → S<sub>2</sub> → S<sub>3</sub>, a–c); the $D _ { \mathrm { { J S } } }$ -screened sample receives object-bank rare classes, giving $\widetilde { \mathbf Y }$ (d; pasted objects in colour), which HALO ray-traces into the sparse sweep $\widetilde { \textbf { X } } ( \mathbf { e } )$

Algorithm 1 HALO: Hardware-Aware LiDAR Observa  
tion. VOXEL(·) maps a metric point to its grid cell   
at edge $\Delta _ { v } { = } 0 . 2 \mathrm { m } .$ , or NULL outside the grid, so   
VOXEL $\scriptstyle ( \mathbf { o } ) = ( 0 , 1 2 8 , 1 0 )$ . A beam at angles $( \phi , \psi )$ has direc  
tion d(ϕ, ψ)=(cos ϕ cos ψ, cos ϕ sin ψ, sin ϕ). Beam angles,   
the march step $\delta t ,$ post-hit skip $\Delta t ,$ range limit $R _ { \mathrm { m a x } } ,$   
MAXRETURNS, SAMPLESPERRETURN and the noise scales   
are tabulated in Appendix A.   
Require: Complete label grid $\widetilde { \mathbf Y }$ with EMPTY=0; sensor origin o   
Ensure: Sparse mask $\widetilde { \mathbf { X } } ;$ invalid mask V (1= occupied, unobserved)   
1: Xe ← 0   
2: for all beams $( \phi , \psi )$ do   
3: t ← 0; r ← 0; v ← NULL   
4: while $t < R _ { \operatorname* { m a x } }$ and r < MAXRETURNS do   
5: $t \gets t + \delta t ; \quad v \gets \mathrm { V o X E L } ( \mathbf { o } + t \mathbf { d } ( \phi , \psi ) )$   
6: if v = NULL then break   
7: if Ye [v] = EMPTY or $v = v _ { \mathrm { p r e v } }$ then continue   
8: for k = 1 to SAMPLESPERRETURN do   
9: $\mathbf { \Phi } ( \epsilon _ { r } , \epsilon _ { \psi } , \epsilon _ { \phi } ) \sim \mathcal { N } \big ( \mathbf { 0 } , \mathrm { d i a g } ( \sigma _ { r } ^ { 2 } , \sigma _ { \psi } ^ { 2 } , \sigma _ { \phi } ^ { 2 } ) \big )$   
10: u ← VOXEL $\big ( \mathbf { o } + \big ( t { + } \epsilon _ { r } \big ) \mathbf { d } ( \phi { + } \epsilon _ { \phi } , \psi { + } \epsilon _ { \psi } ) \big )$   
11: if u ̸= NULL and $\widetilde { { \mathbf Y } } [ u ] \neq$ EMPTY then $\widetilde { \mathbf { X } } [ u ] \gets 1$   
12: end for   
13: v<sub>prev</sub> ← v; r ← r + 1; t ← t + ∆t ▷ clear the hit cell   
14: end while   
15: end for   
16: $\mathbf { V }  [ \widetilde { \mathbf { Y } } \neq ]$ EMPTY] $\wedge \neg \widetilde { \mathbf { X } }$ ▷ occupied, never sampled   
17: return $( \widetilde { \mathbf { X } } , \mathbf { V } )$

a) Denoiser and conditioning: The ground-truth grid is embedded on the product simplex M of Section III-A. A chain initialised at u generates unconditionally, so the observation must steer every reverse step. From a single sweep X we derive two complementary conditioning streams $\mathbf { c } = \{ \mathbf { B } , \mathbf { F } \}$ . A learned sparse-3D BEV head $h _ { \mathrm { b e v } }$ gives a per-cell label map $\mathbf { B } \in \{ 0 , \ldots , K - 1 \} ^ { L \times W }$ , replicated along $Z$ and embedded by the same convolution as the noisy iterate. A frozen LSK3DNet [55] backbone yields a per-voxel feature volume F (any voxel-aligned LiDAR-segmentation backbone can replace it), read by an auxiliary sparse residual encoder over the ${ \sim } 1 \%$ occupied voxels (architecture summary in Appendix G). $h _ { \mathrm { b e v } }$ is trained separately and its label maps are precomputed once over the pool, so it is frozen with respect to $f _ { \theta }$ and no gradient crosses the integer-valued B. The two streams reach a level differently. $\mathcal { E } _ { \ell } ( \mathbf { B } )$ embeds B once, as a K-channel one-hot volume through a $3 \times 3 \times 3$ convolution. It average-pools the volume by a factor of two, ℓ times, and projects it to level ${ \boldsymbol { \ell } } { \boldsymbol { \mathbf { \mathit { s } } } }$ width with a further $3 \times 3 \times 3$ convolution owned by that block. $\mathcal { E } _ { \ell } ( \mathbf { F } )$ is that same per-block projection applied to the sparse encoder’s own level-ℓ output, which needs no pooling. Both are injected additively,

$$
\mathbf { h } ^ { ( \ell ) } \gets \mathbf { h } ^ { ( \ell ) } + \mathcal { E } _ { \ell } ( \mathbf { B } ) + \mathcal { E } _ { \ell } ( \mathbf { F } ) ,\tag{3}
$$

not by FiLM’s [56] feature-wise affine modulation; the additive form keeps observation and iterate in one feature space. Neither stream is a completed scene: SGSC generates the volume from c rather than refining an existing prediction.

The reverse process is driven by a single denoiser $f _ { \theta }$ (Fig. 4): a four-level dense 3D U-Net (channels 32 → 256) with a transposed-convolution decoder and encoder skips, in the standard diffusion design [50], [57]. The timestep enters only through adaptive group normalisation (AdaGN) [50],

$$
\mathrm { A d a G N } ( \mathbf { h } , t ) = \mathbf { s } ( t ) \odot \mathrm { G N } ( \mathbf { h } ) + \mathbf { b } ( t ) , \qquad ( \mathbf { s } , \mathbf { b } ) = g \big ( \pmb { \tau } ( t ) \big ) ,\tag{4}
$$

with g a linear projection of the sinusoidal embedding $\tau ( t ) ;$ the observation instead enters additively (Eq. (3)). A final convolution gives per-voxel logits whose softmax is the ${ \bf x } _ { 0 ^ { - } }$

![](images/0260e30e7244201ceac296151a568319a9306d27db40f6477893c7d7ac494281.jpg)  
Fig. 4. The SGSC denoiser in its generation role. A four-level dense 3D U-Net (channels 32 → 256) generates the scene from the uniform prior ${ \bf { x } } _ { T } = { \bf { u } } ,$ conditioned on a BEV map B and per-voxel LSK3DNet features F from one sweep, injected additively at every level (Eq. (3)); the timestep enters only through adaptive group normalisation (AdaGN), and the bottleneck stacks two residual blocks (RB). The same architecture, retrained against a structured source, drives $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ →refinement (Fig. 5).

prediction $\hat { p } _ { \theta } ( \cdot \mid \mathbf { x } _ { t } , t , \mathbf { c } ) = \mathrm { s o f t m a x } \big ( f _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } ) \big )$ used by the reverse step. Per-level widths are in Appendix G. The denoiser totals ∼35M parameters.

b) Training objective: Training minimises the per-step KL between the analytic posterior (Eq. (1)) and the parametric reverse (Eq. (2)),

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { S G S C } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { t } \sim q ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) , \mathbf { c } } \big [ D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) \big \mid \big | } & { } \\ { p _ { \theta } \big ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { c } \big ) \big ) \big ] , } \end{array}\tag{5}
$$

with t uniform on $\{ 1 , \ldots , T \}$ ; summed over t this is the standard variational upper bound on − log p<sub>θ</sub>(x<sub>0</sub> | c) [10]; its parameter-free terminal term drops out, its t=1 term is cross-entropy against the one-hot target. We augment the KL with Lovasz-softmax [58] and a bottleneck-auxiliary surrogate´ (weights in Appendix G).

c) Full posterior sampling at inference: Sampling starts from the uniform prior $\tilde { \mathbf { x } } _ { T } = \mathbf { u } .$ , the tilde marking a sampler iterate, and runs the parametric reverse Eq. (2) from t=T to 1 under $f _ { \theta ^ { \prime } } { } .$ , the exponential-moving-average (EMA) copy of θ $( T { = } 1 0 0$ , EMA decay 0.9999). It draws each $\tilde { \mathbf { x } } _ { t - 1 } \sim p _ { \theta ^ { \prime } } ( \tilde { \mathbf { x } } _ { t - 1 } |$ $\tilde { \mathbf { x } } _ { t } , \mathbf { c } )$ and reads off $\hat { \mathbf Y } _ { v } = \arg \operatorname* { m a x } _ { k } \tilde { \mathbf { x } } _ { 0 } [ v , k ]$ from the t=0 draw (full loop in Appendix G). Prediction is stochastic and we fix the seed. Unlike the refinement regime below, it cannot be shortened: one-hot targets lie near the maximal simplex separation from u, so transport spans the whole trajectory and from-noise generation costs T denoiser evaluations per scene (few-step distillation is future work).

## D. Structured Source Discrete Diffusion $( S ^ { 2 } D ^ { 2 } )$

To improve a completion without retraining the model that produced it, we introduce $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ , which refines a frozen model’s prediction by flow matching on the per-voxel label simplex (Fig. 5). Standard flows start from a Gaussian or uniform prior and so discard whatever the base already recovered. We replace that prior with a structured source, the one-hot pervoxel argmax prediction $\mathbf { x } _ { \mathrm { s r c } }$ of a frozen base $g _ { \mathrm { b a s e } }$ (e.g., SCPNet [1]), keeping the one-hot ground truth $\mathbf { x } _ { \mathrm { 0 } }$ as the target. Both endpoints are then fixed and the path between them is straight, which is what admits a single deployment step and leaves the base untouched, its logits included.

a) Forward process from a structured source: We instantiate every ingredient on the product simplex $\begin{array} { r } { \mathcal { M } = \prod _ { v } \Delta ^ { K - 1 } } \end{array}$ of Section III-A. The source is the base’s decoded label map rather than its logits, which fixes what each uplift is measured against; any simplex-valued source would serve, and the confidence the argmax discards is an unevaluated extension. With both endpoints fixed we read the interpolant on the diffusion clock $\begin{array} { r } { \bar { \alpha } _ { t } \ = \ \prod _ { r = 1 } ^ { t } ( 1 - \beta _ { r } ) } \end{array}$ , the cumulative-retention schedule of Section III-A taken as a deterministic flow time: $\bar { \alpha } _ { 0 } = 1$ is the target end $( { \bf z } _ { 1 }  { \bf x } _ { 0 } ) , \bar { \alpha } _ { T } \approx 0$ the source end $( \mathbf { z } _ { \mathrm { 0 } }  \mathbf { x } _ { \mathrm { s r c } } )$ . The interpolant is z<sub>s</sub> at $s = \bar { \alpha } _ { t }$ (the implementation takes $\beta _ { t }$ linear from $1 0 ^ { - 4 }$ to 0.1 over T=100 steps, so $\bar { \alpha } _ { T } \approx 5 . 6 \times 1 0 ^ { - 3 } )$ :

![](images/776f9a4802252d2e43e0de450616d0c7e6620552ad5f251a1bb6eb2cdc0f3c43.jpg)

![](images/26e1b5e5cba3585db0d007367eb71524f06cf91a80a0b05824bead5f48f7ce8c.jpg)  
Fig. 5. S<sup>2</sup>D<sup>2</sup> refinement: the shared denoiser’s second role. The velocity field carries the base prediction $\mathbf { x } _ { \mathrm { s r c } }$ to ground truth x<sub>0</sub> in one Euler step on the per-voxel simplex (top); the identity and error bound rule out the amplification that would force the full chain (middle; Prop. 3, Thm. 1 and Cor. 1 of Appendix B); a motorcyclist scene, frozen base vs. base+S<sup>2</sup>D<sup>2</sup> at N=1, one denoiser evaluation (bottom).

$$
\mathbf { x } _ { t } \ = \ { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } \ + \ \left( 1 - { \bar { \alpha } _ { t } } \right) \mathbf { x } _ { \mathrm { s r c } } , \qquad t \in \{ 0 , 1 , \ldots , T \} .
$$

The forward kernel is therefore a Dirac on Eq. (6), not the stochastic Markov corruption of a D3PM [9]. Each $\mathbf { x } _ { t }$ is a convex combination of two simplex points, so the path stays in M with no mass reallocation and no transition matrix $\mathbf { Q } _ { t }$ (simplex invariance), and its terminal ${ \bf x } _ { T } \ \approx \ { \bf x } _ { \mathrm { s r c } }$ keeps it anchored to the base’s estimate.

(6)

Equation (6) is the mean path, and the two ends instantiate it differently: training feeds per-voxel one-hot draws $\check { \mathbf { x } } _ { t } \sim \operatorname { C a t } ( \mathbf { x } _ { t } )$ , whose mean is exactly $\mathbf { x } _ { t } .$ , while the sampler never draws and runs on the path points themselves. Keeping the training state categorical reuses the multinomial-diffusion posterior with the source in place of the uniform prior, and keeps every per-step term a finite categorical KL. Throughout, $\mathbf { x } _ { t }$ is the path point, xˇ<sub>t</sub> the categorical draw training takes from it, $\tilde { \mathbf { x } } _ { t }$ the sampler’s own iterate, and δ marks objects evaluated on the path point.

b) Source conditioning: The denoiser is conditioned on a pair of streams derived from the base prediction and raw sweep, $\mathbf { c } = \{ \mathbf { B } , \mathbf { L } \}$ . B is the base-derived BEV semantic map, the parameter-free topmost-non-empty-class projection of the base’s 3D prediction, and L a binary occupancy grid voxelised from the raw scan. $f _ { \theta }$ reads a state z (the path point here, the categorical draw in training, the sampler iterate at inference) and emits the clean-scene estimate $\hat { \mathbf { x } } _ { 0 } ( \mathbf { z } ) : = \mathrm { s o f t m a x } \big ( f _ { \theta } ( \mathbf { z } , t , \mathbf { c } ) / \tau _ { \mathrm { s m p } } \big )$ . The sampling temperature is $\tau _ { \mathrm { s m p } } { = } 1$ in every main-body result, immaterial at the deployed single step (Section IV-C). We drop arguments when clear. The BEV stream is left uncorrected: refining the map in 2D leaves the SSC output unchanged to within run-to-run variation (Section IV-C).

The occupancy grid is the only input whose regime differs between training and deployment. It is multi-frame-accumulated at training, following SCPNet’s multi-frame teacher [1], and single-frame at deployment. The shift is benign: retraining end to-end with a single-frame grid at both ends costs a fraction of a point (Section IV-C), so we keep the accumulated grid at training and deploy single-frame.

$\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ reuses the denoiser architecture of Section III-C, the four-level dense 3D U-Net of Fig. 4, but not its weights: $f _ { \theta }$ is trained from scratch against the structured source. The grid L enters through the auxiliary sparse encoder that processed F in Section III-C: the structured source already carries dense per-voxel semantics, so the conditioning need only reinject the scan’s geometric evidence. Both enter additively at every level as in Eq. (3), with $\mathcal { E } _ { \ell } ( \mathbf { L } )$ in place of $\mathcal { E } _ { \ell } ( \mathbf { F } )$ .

c) Prediction target and the per-step objective: The displacement to be estimated along this path is the constant $\mathbf { x } _ { \mathrm { 0 } } - \mathbf { x } _ { \mathrm { s r c } } ,$ so the velocity the network realises through $\hat { \mathbf { x } } _ { 0 }$ is

$$
{ \mathbf v } _ { \theta } ( { \mathbf x } _ { t } , t , { \mathbf c } ; { \mathbf x } _ { \mathrm { s r c } } ) = \hat { { \mathbf x } } _ { 0 } - { \mathbf x } _ { \mathrm { s r c } } .\tag{7}
$$

The estimate $\hat { \mathbf { x } } _ { 0 }$ is the network’s only prediction target, shared by training and sampling (also the SGSC denoiser’s $\mathbf { x } _ { 0 } .$ prediction) and free to lie in the simplex interior; velocities are read in flow time $s = \bar { \alpha } _ { t }$ . The source is a known constant of the sample. It enters the posterior and the plug-in reverse algebraically, and the denoiser through a learned convolution summed into the input features, alongside the state and BEV map B. It also shrinks the regressed displacement: $\mathbf { x } _ { \mathrm { s r c } } = \mathbf { u }$ gives the from-noise regime’s near-maximal one, closed only over many steps; the structured source leaves one supported only on the base’s erroneous voxels (small at high top-1 accuracy). Short and, by the straight interpolant, constant along the path, it needs one Euler step to traverse, up to the endpoint residual (telescoping error bound, Appendix B).

The objective is the discrete-diffusion variational one rather than a plain velocity regression, which under Eq. (7) would reduce to an $L _ { 2 }$ fit of $\hat { \mathbf { x } } _ { 0 }$ onto $\mathbf { x } _ { 0 } \mathbf { : }$ on the deterministic path the variational bound is not a bound but an exact conditional likelihood (Appendix B). Training tuples are formed offline from the frozen base’s argmax predictions on the training frames. Training instantiates the mean path stochastically through the draws $\check { \mathbf { x } } _ { t } \sim \mathrm { C a t } ( \mathbf { x } _ { t } )$ introduced above; the per-step target is the multinomial-diffusion posterior expression [10] with the source in place of the uniform prior,

$$
\begin{array} { r l } & { q ( \check { \mathbf { x } } _ { t - 1 } | \check { \mathbf { x } } _ { t } , \mathbf { x } _ { 0 } , \mathbf { x } _ { \mathrm { s r c } } ) \propto \left[ \alpha _ { t } \check { \mathbf { x } } _ { t } + \left( 1 - \alpha _ { t } \right) \mathbf { x } _ { \mathrm { s r c } } \right] \odot } \\ & { \qquad \left[ \bar { \alpha } _ { t - 1 } \mathbf { x } _ { 0 } + \left( 1 - \bar { \alpha } _ { t - 1 } \right) \mathbf { x } _ { \mathrm { s r c } } \right] , } \end{array}\tag{8}
$$

with $\alpha _ { t } = 1 - \beta _ { t }$ and ⊙ the elementwise product. Setting $\mathbf { x } _ { \mathrm { s r c } } =$ u recovers Eq. (1); otherwise Eq. (8) is a construction, not a Bayes identity. This is the form the released trainer evaluates.

The reverse reuses the endpoint estimate in the same plug-in form as Eq. (2), $p _ { \theta } ( \check { \mathbf { x } } _ { t - 1 } | \check { \mathbf { x } } _ { t } , \mathbf { x } _ { \mathrm { s r c } } , \mathbf { c } ) : = q ( \check { \mathbf { x } } _ { t - 1 } | \check { \mathbf { x } } _ { t } , \hat { \mathbf { x } } _ { 0 } , \mathbf { x } _ { \mathrm { s r c } } ) ;$ the two regimes share one reverse parameterisation and differ only in the noise target: uniform u there, the structured source $\mathbf { x } _ { \mathrm { s r c } }$ here. Although a one-hot source forfeits the uniform prior’s strict positivity, the KL stays finite $( \operatorname { s u p p } q \subseteq \operatorname { s u p p } p _ { \theta } ;$ support argument accompanying the training algorithm in Appendix G). At t=1 the second bracket collapses to $\mathbf { x } _ { \mathrm { 0 } }$ and we take the usual reconstruction cross-entropy against the label [10], [57].

With t uniform on $\{ 1 , \ldots , T \}$ and the expectation including the draw of $\check { \mathbf { x } } _ { t } ,$ the objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } = \mathbb { E } _ { t , \check { \mathbf { x } } _ { t } , ( \mathbf { x } _ { 0 } , \mathbf { x } _ { \mathrm { s r c } } , \mathbf { c } ) } \Bigm [ D _ { \mathrm { K L } } \bigm ( q ( \check { \mathbf { x } } _ { t - 1 } \mid \check { \mathbf { x } } _ { t } , \mathbf { x } _ { 0 } , \mathbf { x } _ { \mathrm { s r c } } ) } \\ { \bigm \lVert p _ { \theta } ( \check { \mathbf { x } } _ { t - 1 } \mid \check { \mathbf { x } } _ { t } , \mathbf { x } _ { \mathrm { s r c } } , \mathbf { c } ) \bigm ) \Bigm ] . } \end{array}\tag{9}
$$

d) How exact the objective is, and where it stops: On the deterministic mean path of Eq. (6), per-step matching admits an exact conditional-likelihood identity. Started from the sourceconditional prior $\delta _ { \mathbf { x } _ { \mathrm { s r c } } }$ , under a strictly decreasing, exactly terminating schedule and one-hot targets, the usual variational bound collapses to an equality. This holds because the Dirac kernel makes the Jensen step exact and the source-conditional prior matches the forward terminal. The identity transfers to the trained objective of Eq. (9) only approximately: that objective scores the one-hot draws $\check { \mathbf { x } } _ { t } .$ , not the mean-path points the identity governs. The two share a minimiser at $\hat { \mathbf { x } } _ { 0 } = \mathbf { x } _ { 0 }$ , so the transfer is exact in the realisable limit alone. Statement and proof are in Appendix B.

The shipped schedule does not terminate exactly, and the consequence is confined to deployment: with $\bar { \alpha } _ { T } ~ > ~ 0$ the forward terminal sits just short of $\mathbf { x } _ { \mathrm { s r c } } ,$ so we do not claim the deterministic-path bound there, and the effect is bounded by the $2 \bar { \alpha } _ { T }$ term below.

The deployed objective adds three task-specific terms to a per-voxel-weighted KL:

$$
\mathcal { L } = \tilde { \mathcal { L } } _ { \mathrm { K L } } + \lambda _ { \mathrm { L o v } } \big ( \mathcal { L } _ { \mathrm { L o v } } ^ { \mathrm { a l l } } + \mathcal { L } _ { \mathrm { L o v } } ^ { \mathrm { o c c } } \big ) + \lambda _ { a } \mathcal { L } _ { \mathrm { a u x } } ,\tag{10}
$$

where $\tilde { \mathcal { L } } _ { \mathrm { K L } }$ is Eq. (9) with per-voxel weights. The two Lovasz-softmax terms [58] share the weight´ $\lambda _ { \mathrm { L o v } }$ and apply the same IoU surrogate to $\hat { \mathbf { x } } _ { 0 }$ over different voxel populations. $\mathcal { L } _ { \mathrm { L o v } } ^ { \mathrm { a l l } }$ runs over the whole grid, so it prices predicted mass on empty space; $\mathcal { L } _ { \mathrm { L o v } } ^ { \mathrm { o c c } }$ runs over occupied voxels only, so it prices confusion among the classes. $\mathcal { L } _ { \mathrm { a u x } }$ is a cross-entropy of $\hat { \mathbf { x } } _ { 0 }$ against the clean labels, scaled by $( 1 - t / T ) + 1$ , and added alongside the per-step KL rather than substituted for it. The KL term is weighted by focal [5], effective-number class-balanced [59], and observation-aware terms (the last tripling the weight of voxels the sweep returns). These steer the signal onto the occupied voxels (∼5% of the grid, yet the ones scored by per-class IoU) and, within them, the rare classes. The coefficients $\lambda _ { \mathrm { L o v } } , \lambda _ { a }$ , the focal exponent, and the class and observation weights are tabulated in Appendix G, which also traces a training step end-to-end.

e) Inference by correction sampling: The sampler is initialised at the structured source, not at noise: at $t { = } T$ the training draw $\check { \mathbf { x } } _ { T }$ is an $O ( \bar { \alpha } _ { T } )$ perturbation of $\mathbf { x } _ { \mathrm { s r c } } ,$ identical wherever the base is right.

Every update then moves along one direction. The network estimates the path’s endpoint $\hat { \mathbf { x } } _ { 0 }$ at every t and Eq. (6) is straight, so the displacement left to travel is always $\hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { \mathrm { s r c } }$ up to a schedule factor, in place of the per-step slice a standard reverse sampler follows:

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { t - 1 } = \tilde { \mathbf { x } } _ { t } + \left( \bar { \alpha } _ { t - 1 } - \bar { \alpha } _ { t } \right) \cdot \left( \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { \mathrm { s r c } } \right) . } \end{array}\tag{11}
$$

In flow time $s = \bar { \alpha } _ { t }$ this is the explicit Euler step for the velocity of $\mathrm { E q . ~ } ( 7 )$ , and Cold Diffusion’s improved sampler [45], exact for $\mathbf { x } _ { \mathrm { 0 } }$ -affine degradations such as ours. Iterated from $\tilde { \mathbf { x } } _ { t _ { N } } =$ $\mathbf { x } _ { \mathrm { s r c } }$ over an N-step schedule $T = t _ { N } \ge \cdots \ge t _ { 0 } = 0$ (N counts denoiser evaluations), it carries the iterate toward $\mathbf { x } _ { 0 } ,$ , and the label is $\hat { \mathbf Y } _ { v } = \arg \operatorname* { m a x } _ { k } \tilde { \mathbf x } _ { t _ { 0 } } [ v , k ]$ (full loop in Appendix G).

$f )$ How the error behaves over the chain: Because the path is linear, summing Eq. (11) along the schedule telescopes, and the final iterate’s deviation from $\mathbf { x } _ { \mathrm { 0 } }$ is bounded by the largest per-step prediction residual $\varepsilon _ { i } : = |$ softmax $\left( f _ { \theta } ( \tilde { \mathbf { x } } _ { t _ { i } } , t _ { i } , \mathbf { c } ) \right) -$ $\mathbf { x } _ { 0 } \Vert _ { 1 }$ <sub>1</sub> plus a $2 \bar { \alpha } _ { T }$ offset from initialising at $\mathbf { x } _ { \mathrm { s r c } } \colon \| \tilde { \mathbf { x } } _ { t _ { 0 } } - \mathbf { x } _ { 0 } \| _ { 1 } \leq$ max<sub>i</sub> $\varepsilon _ { i } + 2 \bar { \alpha } _ { T }$ . The offset is the cost of the implementation’s nonzero terminal rather than of the method. Iterated correction therefore does not amplify per-step error, with no Lipschitz assumption, and deployment collapses the sum to one term. At N=1 the error is governed by the single residual at $t { = } T \colon$ on the discrete simplex the counterpart of InDI’s single-step regime [46]. There one step minimises distortion at the cost of perceptual quality—the trade a voxel-labelling metric wants. Appendix B proves the bound, gives the conditions under which it is attained, and locates the irreducible refiner error floor.

The bound is an $\ell _ { 1 }$ distance while mIoU scores the argmax. One line of algebra ties the two: a voxel decodes correctly once its endpoint estimate clears the margin $\begin{array} { r } { \frac { 1 } { 2 } + O ( \bar { \alpha } _ { T } ) } \end{array}$ which the shipped schedule puts at 0.503 (corollaries in Appendix B). The theory therefore reaches the reported metric, but only so far: it certifies the voxels that clear the margin and says nothing about the rest, where the empirical gap between bases lives; it does not predict an mIoU.

Setting $\begin{array} { r l r } { \mathbf { x } _ { \mathrm { s r c } } } & { { } = } & { \mathbf { u } } \end{array}$ recovers the from-noise chain of Section III-C, and Appendix B places Eq. (6) beside its neighbours (D3PM [9], DDPM/DDIM [57], [60], Rectified Flow [61], InDI [46] and Cold Diffusion [45]), marking which are literal instances.

## E. Why the Correction Works

Per-step matching supplies the usable signal. Where the base is right, ${ \bf x } _ { \mathrm { s r c } } = { \bf x } _ { 0 }$ pins the path and the term vanishes for any prediction; where it is wrong, Eq. (9) contrasts label against source rather than regressing the label alone. Replacing it with a timestep-uniform cross-entropy collapses training (Section IV-C).

The added terms tighten the objective rather than tilt it: all three score the same endpoint estimate and vanish as $\hat { \mathbf { x } } _ { 0 } \to \mathbf { x } _ { 0 }$ so $\hat { \mathbf { x } } _ { 0 } = \mathbf { x } _ { 0 }$ stays a minimiser of the composite. They carry the gradient wherever the KL term is flat because the base is already correct.

At N=1 the sampler starts at $\tilde { \mathbf { x } } _ { t _ { N } } = \mathbf { x } _ { \mathrm { s r c } } , \ s \mathbf { o } \ \varepsilon _ { 1 }$ is the residual of a single evaluation at an input the network was trained on, and the uniform-t schedule is a curriculum for it. The telescoping bound does not assert that one step suffices, since $\varepsilon _ { 1 }$ is a network error it does not bound. What it rules out is the amplification that would otherwise force many, and that one step is enough is empirical (Section IV-C).

## IV. EXPERIMENTS

Datasets: SemanticKITTI [4] is the standard outdoor SSC benchmark: 19,130 training frames (seq. 00–07, 09–10), 4,071 validation frames (seq. 08, stride-1 voxel release) and 3,901 hidden-test frames (seq. 11–21, subsampled). Its target is a 256×256×32 grid spanning 51.2×51.2×6.4 m at 0.2 m, labelled with 19 classes plus empty/unlabelled, and its distribution is severely skewed: empty voxels are ∼95% of ground truth and motorcyclist 0.0037% of labelled points (Fig. 1c). $\mathrm { P S } ^ { 3 }$ (Section III-B) adds 32,039 synthetic pairs to these 19,130 real frames, a 2.67× expansion to 51,169. Every part of it (the generator, the held-out scenes its distribution filter screens against, and the object bank) derives from this training split, and it augments SemanticKITTI training only. Two further datasets serve as zero-shot domains only, evaluated with the frozen SemanticKITTI checkpoint and no target labels or finetuning. They are SSCBench-KITTI360 [62] (val seq. 06, 1,812 frames, 16 shared classes) and SemanticPOSS [63] (val seq. 02, ≈500 frames, the 11-class TALoS Tab. 4 map, with ground truth reconstructed by us rather than official).

Metrics: Following [1], [64], the primary metric is mIoU over the 19 evaluation classes together with class-agnostic completion IoU (Comp.); precision and recall are quoted only where a diagnosis needs them. Every 3D number we report as a result comes from the official semantic-kitti-api, which excludes the invalid mask and so scores only voxels some sweep of the sequence reached. The never-observed volume, where a generative prior is most free to invent, lies outside every 3D number we report. Appendix E bounds how much of our predicted occupancy falls there. Every mIoU we report as a result is a deployable single-frame measurement. The 2D figures of Section IV-D come from a separate BEV evaluator. All ∆ values are computed from unrounded scores. Because mIoU weights every class alike and so hides the rare road users that safety depends on, we add three metrics that price the rare classes, two of them restricted to the vulnerable road user (VRU) classes: person, bicyclist and motorcyclist. Safety-Critical mIoU (SCmIoU) instead reweights all 19 by four hazard tiers of our own choosing. ISO 21448 [65] assigns severity to a hazardous event in an operational situation rather than to a semantic class, so neither the tiers nor their order follows from a standard. Across the refined bases the ordering they produce is mIoU’s at every weight vector we swept, so for those rows the SC-m column of Tables I and III is a robustness check rather than evidence separating methods; it is not ordering-free in general (Section IV-E). The tiers and the weight sweep are in Appendix E. VRU-IoU is the hyperparameter-free mean IoU over the three VRU classes, and it is the tail metric that does reorder methods. Detection-Window IoU (DW-IoU) discounts per-class IoU by how many frames a system can deliver inside a reaction window,

$$
\mathrm { D W - I o U } _ { c } = 1 - ( 1 - \mathrm { I o U } _ { c } ) ^ { R \cdot T _ { w } } ,\tag{12}
$$

with R each system’s deployed end-to-end rate and $T _ { w }$ the window, fixed at 1.0 s; DW-VRU-IoU averages it over the three VRU classes. Because consecutive single-frame predictions are temporally correlated, DW-IoU is an optimistic upper bound on the benefit of a faster refresh, not a detection probability. Training: All runs use NVIDIA H100 80 GB GPUs. SGSC (§III-C) and $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ (§III-D) share one ∼35M-parameter denoiser architecture and one recipe: AdamW at lr $1 0 ^ { - 4 }$ , effective batch 4 on $2 \times \mathrm { H 1 0 0 }$ $T { = } 1 0 0$ , and EMA weights. The headline frozen-base checkpoint is the best-EMA model at 40K steps (∼37 GPU-hours) deployed at $N { = } 1$ . The alt-base and SGSC runs train to 100K steps, and SGSC deploys under full posterior sampling (N=T=100) with conditioning $\mathbf { c } = \{ \mathbf { B } , \mathbf { F } \}$ , the LiDAR-only BEV map and per-voxel LSK3D [55] features.

TABLE I  
MAIN RESULTS ON SEMANTICKITTI. PUBLISHED BASELINES AT THEIR OWN test LEADERBOARD; OUR ROWS ON BOTH SPLITS. S3CNET, NOT OURS, TAKES THE VULNERABLE ROAD USER (VRU) COLUMN (SECTION IV-E).
<table><tr><td>Method</td><td>Eval</td><td>mIoU</td><td>Comp</td><td>SC-m</td><td>VRU</td></tr><tr><td>LMSCNet [11]</td><td>test</td><td>17.6</td><td>56.7</td><td>12.4</td><td>0.0</td></tr><tr><td>SSA-SC [14]</td><td>test</td><td>23.5</td><td>58.8</td><td>17.8</td><td>2.6</td></tr><tr><td>JS3C-Net [15]</td><td>test</td><td>23.8</td><td>56.6</td><td>18.7</td><td>4.5</td></tr><tr><td>DiffSSC [32]</td><td>test</td><td>27.4</td><td>63.4</td><td>n/a</td><td>n/a</td></tr><tr><td>S3CNet [13]</td><td>test</td><td>29.5</td><td>45.6</td><td>29.9</td><td>32.6</td></tr><tr><td>SCPNet [1] (published)</td><td>test</td><td>36.7</td><td>56.1</td><td>32.5</td><td>18.2</td></tr><tr><td>TALoS [17]</td><td>test</td><td>37.9</td><td>60.2</td><td>33.6</td><td>20.0</td></tr><tr><td> $\bullet \mathrm { S C P N e t } + \mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 } ~ ( N { = } 1 )$ </td><td>test</td><td>38.8</td><td>58.9</td><td>34.7</td><td>21.6</td></tr><tr><td> $\bullet w i t h D _ { 4 } e n s e m b l e ( N { = } 4 ) ^ { \ S }$ </td><td>test</td><td>39.2</td><td>59.0</td><td>35.2</td><td>22.3</td></tr><tr><td> $\mathrm { S C P N e t ~ [ 1 ] ~ a t ~ \# } \mathrm { f r a m e { = } } 4 ^ { \ddagger }$ </td><td>test</td><td>47.5</td><td>68.5</td><td>43.9</td><td>31.0</td></tr><tr><td>PaSCo [18]† (3 subnets)</td><td>val</td><td>30.1</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>● SGSC (from noise, T=100)</td><td>val</td><td>30.5</td><td>53.6</td><td>25.6</td><td>8.5</td></tr><tr><td> $\bullet \mathrm { s G S C } + \mathbf { S } ^ { 2 } \mathbf { D } ^ { 2 }$ </td><td>val</td><td>32.8</td><td>55.4</td><td>28.3</td><td>12.5</td></tr><tr><td>SCPNet [1] (released weights)#</td><td>val</td><td>37.55</td><td>50.24</td><td>n/a</td><td>n/a</td></tr><tr><td>+ SemCity [35]#</td><td>val</td><td>38.19</td><td>59.25</td><td>n/a</td><td>n/a</td></tr><tr><td>SCPNet* (our port, frozen)</td><td>val</td><td>36.17</td><td>49.9</td><td>32.5</td><td>14.7</td></tr><tr><td> $\bullet \mathrm { S C P N e t ^ { * } } + \mathbf { S } ^ { 2 } \mathbf { D } ^ { 2 } \ ( N { = } 1 )$ </td><td>val</td><td>38.54</td><td>52.7</td><td>35.2</td><td>19.6</td></tr></table>

SC-m: safety-critical mIoU. Bold = best per column within a split; • ours. <sup>∗</sup>our port; <sup>¶</sup>point-cloud output, voxelised; <sup>♯</sup>quoted from [35] Tab. 2 on SCPNet’s released weights, its base sits 1.4 pp above our port. Excluded from bolding: <sup>§</sup>eight-view $D _ { 4 }$ ensemble, <sup>†</sup>panoptic-mode evaluator, <sup>‡</sup>four sweeps, <sup>∥</sup>test-time adaptation.

The software stack, the seed, the noise schedule and the loss weights are in Appendix G.

Baselines and protocol: Published baselines are quoted at each method’s own test-leaderboard result, whereas our per-class and ablation evidence is measured on val seq. 08; Table I therefore carries an explicit Eval column and no row is read across splits. S3CNet [13] enters the table on an uneven profile (completion IoU 45.6 against 56.1–68.5 for every other test row at mIoU 29.5%). Yet among single-sweep rows it leads every VRU class and takes that column, a gap on the safety metric we introduce that we do not close (Section IV-E; its per class row in Appendix D). DiffSSC predicts a point cloud and submitted it voxelised, and publishes no per-class breakdown. TALoS adapts the same base we refine [17], so its test row and ours start from one source. All numbers use a single seed without error bars, matching SSC convention; what reproduces under it is the per-class delta over the frozen base rather than the absolute mIoU, for the reason given with Table II.

## A. Main Results

Table I places our rows against published SSC results on both splits. On the hidden test leaderboard, one N=1 correction step over a frozen SCPNet base reaches 38.8% mIoU: +2.1 pp over the published score of the base it refines. To our knowledge that is the strongest such test-leaderboard result to date. On val the same checkpoint reaches 38.54% against its base’s 36.17%. The nearest comparison is SemCity [35], which refines the same base with a continuous prior: it reports 38.19% from SCPNet’s released weights, which score 1.4 pp above our port, and it gains far more completion IoU than we do (+9.0 against +2.8). Our lead is 0.35 pp on mIoU from a weaker anchor, so we read the two as comparable rather than ranked. Run from noise instead of from a base, the same generative machinery (SGSC, $\mathbf { c } = \{ \mathbf { B } , \mathbf { F } \}$ , full posterior sampling) reaches 30.5% with no base at all, and 32.8% once its own output is corrected. We set that beside PaSCo [18] without differencing it: its evaluator is not protocol-matched to ours, and our predicate admits its single subnet (28.2%) rather than the three-subnet 30.1%. Among prior generative SSC, only DiffSSC [32] has an officialprotocol SemanticKITTI entry, 27.4% on the hidden test; Lee et al. [34], whose from-noise setting SGSC adopts, report only on synthetic CarlaSC, and OccGen [33]’s SemanticKITTI entry is monocular-camera SSC with the LiDAR stream removed. SGSC’s 32.8% is 3.3 pp short of the discriminative base, and its conditioning still reads a frozen segmentation backbone.

TABLE II  
PER-CLASS VAL IOU AGAINST THE FROZEN BASE (SEQ. 08, N=1). “RELEASED”: SHIPPED CHECKPOINT (38.54%); “RETRAIN”: FROM-SCRATCH (38.05%).
<table><tr><td>Class</td><td>Base IoU</td><td>Released ∆</td><td>Retrain ∆</td></tr><tr><td>car</td><td>50.3</td><td>+1.1</td><td>+1.0</td></tr><tr><td>bicycle</td><td>23.7</td><td>+0.6</td><td>-0.1</td></tr><tr><td>motorcycle</td><td>35.6</td><td>-0.1</td><td>+0.2</td></tr><tr><td>truck</td><td>54.7</td><td>+5.3</td><td>+5.4</td></tr><tr><td>other-vehicle</td><td>42.0</td><td>+2.5</td><td>+2.3</td></tr><tr><td>person</td><td>22.0</td><td>+1.2</td><td>+1.9</td></tr><tr><td>bicyclist</td><td>18.0</td><td>+5.3</td><td>+5.2</td></tr><tr><td>motorcyclist</td><td>4.1</td><td>+8.3</td><td>+0.3</td></tr><tr><td>road</td><td>70.0</td><td>+4.6</td><td>+4.3</td></tr><tr><td>parking</td><td>59.3</td><td>+2.4</td><td>+2.3</td></tr><tr><td>sidewalk</td><td>51.9</td><td>+2.1</td><td>+1.9</td></tr><tr><td>other-ground</td><td>13.6</td><td>+0.4</td><td>+0.1</td></tr><tr><td>building</td><td>33.3</td><td>+1.5</td><td>+0.7</td></tr><tr><td>fence</td><td>29.9</td><td>+0.3</td><td>+0.1</td></tr><tr><td>vegetation</td><td>38.5</td><td>+2.2</td><td>+1.9</td></tr><tr><td>trunk</td><td>32.0</td><td>+0.4</td><td>+0.2</td></tr><tr><td>terrain</td><td>51.9</td><td>+2.6</td><td>+2.0</td></tr><tr><td>pole</td><td>37.7</td><td>+0.1</td><td>+0.1</td></tr><tr><td>traffic-sign</td><td>18.8</td><td>+4.2</td><td>+5.8</td></tr><tr><td>Overall mIoU</td><td>36.17</td><td>+2.36</td><td>+1.9</td></tr><tr><td>VRU-IoU</td><td>14.7</td><td>+4.9</td><td>+2.5</td></tr></table>

That margin is indexed on the causal, single-sweep, singlesample predicate of § I, and is measured against a base we did not re-submit (test-margin caveats in Appendix G).

Table II breaks that val gain down by class, and it is not spread evenly. It concentrates where the frozen base is weakest and the missing geometry is still recoverable from context: bicyclist and truck (+5.3 each), road (+4.6), traffic-sign (+4.2), terrain (+2.6), other-vehicle (+2.5) and parking (+2.4). It is flat on thin structures that give a correction step almost nothing to close, at pole (+0.1), fence (+0.3), trunk and other-ground (+0.4); and on the static two-wheelers, bicycle (+0.6) and motorcycle (−0.1). Eighteen of nineteen classes improve. The exception, motorcycle, loses a tenth of a point. So on this base the operator lifts the tail without trading away what it already handles, and it does so class-selectively rather than by uniformly sharpening the grid. That pattern is specific to SCPNet; Section IV-B reports where it does not hold.

One entry in that column we do not claim. Motorcyclist’s +8.3 pp fails to reproduce twice: a from-scratch retrain recovers +0.3, and on the hidden test the class rises only 1.8 → 3.1 against SCPNet’s published row. We therefore read the validation figure as a seed-and-split artefact and print the retrain column beside it, where the other eighteen classes agree within 1.6 pp. The safety metrics inherit the same discount: VRU-IoU’s +4.9 becomes +2.5 on retrain, and the safety claim should be read against that.

![](images/c795cfaa71d68a967486de10f1e74e7e64b33c57e985650fd7a27c8c3fb9b8cd.jpg)  
Fig. 6. Qualitative comparison on val seq. 08. Three frozen sources (JS3C-Net, SCPNet, TALoS’s adaptation of SCPNet), our $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ on the SCPNet base, and ground truth; top bicyclist, bottom motorcyclist. Chips give per-sample rare-class IoU and TP/FP voxel counts (from the $N { = } 4 , + D _ { 4 } { \cdot } \mathrm { T } \mathrm { A }$ configuration).

Figure 6 shows what the correction does to a scene. All three frozen sources already recover the road surface and the building fac¸ades, and one correction step leaves that bulk alone: road, sidewalk, building and vegetation drift by under 1% IoU. The difference is at the tail. Where the base returns a fragmentary, under-filled rider, one step closes it into a coherent object with sharper boundaries, roughly doubling per-sample rare-class IoU with ≈6.6× fewer phantom voxels on the motorcyclist scene, while JS3C-Net collapses on both. The per-voxel figures in this paragraph are read off chips rendered from the $N { = } 4 , + D _ { 4 ^ { - } }$ TTA configuration, so they characterise the correction rather than the N=1 operating point the headline reports; this is the behaviour Eq. (11) predicts. Figure 7 shows the same single step on two ordinary frames, from a ∼1%-occupied sweep to dense semantics. Figure 10 extends this to six rare-class scenes; the per-sample voxel counts behind the chips are in Appendix D.

## B. $S ^ { 2 } D ^ { 2 }$ as a Base-Agnostic Refinement Operator

One operator, three frozen bases, one pass each. $\mathrm { S } ^ { 2 } \mathrm { D } ^ { 2 }$ is trained once per base on that base’s own decoded output and then run for a single N=1 correction step. No base is retrained or architecturally touched. It lifts every one: LMSCNet by +1.8, JS3C-Net by +1.6, SCPNet by +2.36 pp val mIoU (Table III). That is what base-agnostic means. The three span the field: a lightweight 2D–3D hybrid, a point-voxel segmentation-coupled network, and a distillation-trained state of the art. The denoiser and recipe are fixed across them, each base refined under its own best pool, step budget and LiDAR stream, so the controlled quantity is the per-base uplift rather than a cross-base absolute.

TABLE III  
ONE DENOISER ARCHITECTURE ON THREE FROZEN BASES, EACHTRAINED ON ITS OWN BASE’S OUTPUT AND REFINED BY A SINGLE N=1PASS. VAL SEQ. 08; ∆ IS THE MIOU UPLIFT OVER THE PAIRED BASE.
<table><tr><td>Configuration</td><td>mIoU</td><td>Comp</td><td>∆</td><td>SC-m</td><td>VRU</td></tr><tr><td>LMSCNetb</td><td>14.8</td><td>46.7</td><td></td><td>10.4</td><td>0.0</td></tr><tr><td>+S2D2</td><td>16.6</td><td>46.7</td><td> $\mathbf { + 1 . 8 }$ </td><td>12.1</td><td>0.3</td></tr><tr><td>JS3C-Net*b</td><td>22.7</td><td>53.1</td><td> $-$ </td><td>18.6</td><td>3.8</td></tr><tr><td>+S²D²†</td><td>24.3</td><td>50.9</td><td> $\mathbf { + 1 . 6 }$ </td><td>20.0</td><td>2.4</td></tr><tr><td>SCPNet*</td><td>36.17</td><td>49.9</td><td></td><td>32.5</td><td>14.7</td></tr><tr><td> $+ \mathbf { S } ^ { 2 } \mathbf { D } ^ { 2 }$ </td><td>38.54</td><td>52.7</td><td> $\mathbf { + 2 . 3 6 }$ </td><td>35.2</td><td>19.6</td></tr></table>

Bold = better within each base pair; the SCPNet pair repeats Table I as this table’s reference. <sup>∗</sup>re-implemented, <sup>♭</sup>re-scored, <sup>†</sup>official evaluator.

Two bases carry caveats worth stating before the numbers are used. The official SCPNet release depends on spconv 1.0, removed from PyPI. We run it under spconv 2.3 [66] after kernel-shape patches replicating v1’s shared-indice-key behaviour, loading pretrained.pth as-is to reach 36.17% val mIoU, 1.0 pp under the paper’s 37.2%.<sup>1</sup> We did not re-submit the port. The test row of Table I therefore quotes SCPNet’s published 36.7%, and val $\Delta$ is measured against 36.17% throughout (port details in Appendix G).

![](images/9374f8e131fb64aebb8ad94d0e6e30b7ad77a02b3d19fcdcbf81ce143e8b0cba.jpg)  
Fig. 7. Our completions on val seq. 08 (rows: frames 004025, 001850). From a ∼1%-occupied sweep (a) and the base-derived BEV (b), one correction step recovers dense semantics (c) close to ground truth (d).

JS3C-Net marks the boundary of the transfer. Under the protocol-matched official evaluator refinement takes it from 22.7 to 24.3%, but not uniformly. The gain holds on ten of nineteen classes while two of the three VRU classes lose ground: person 8.7 → 5.9 and bicyclist 2.6 → 1.2, motorcyclist 0.0 throughout, so VRU-IoU falls 3.8 → 2.4. It buys bulk geometry at the tail’s expense. The synthetic pool that helps voxel-grid bases also costs this one −3.8 pp under our internal evaluator, its point-voxel head treating the voxel-derived cloud as out of distribution. On both axes the operator’s benefit is scoped to voxel-grid-native bases rather than to LiDAR architectures in general. A second refinement round does not repair this or anything else: retrained from scratch on the SCPNet base’s own Round-1 prediction (38.73% val at $N { = } 4 { + } D _ { 4 }$ TTA), over the real sequences alone, it reaches 37.7% val mIoU, 1.1 pp below it, over-smoothing an already-refined source.

The precision/recall character of the correction is visible across bases. On the over-firing LMSCNet base the semantic gain comes from pruning false-positive occupancy, so completion IoU is unchanged at 46.7. On JS3C-Net the same trade costs 2.2 pp of completion IoU where it does not cancel. On SCPNet it lifts both (49.9 → 52.7). Two per-class regressions replicate across the weaker bases and are worth naming: building falls 34.7 → 28.6 on LMSCNet and 36.6 → 30.6 on JS3C-Net, the largest single-class loss on each, and LMSCNet also loses truck (1.9 → 0.4). On a weaker base the correction does not simply add. On the bulk structural classes it can subtract. An operating-point reading is not excluded by any of this: the base recovers only 53.1% of the occupied voxels, so a confidence-ranked densification of it could shift the same trade. Ruling that out needs the base’s per-voxel posteriors, which its released argmax predictions do not carry (Appendix E). Occupancy forensics and the full per-class breakdown for all three bases are in Appendix D.

## C. Ablations

Most rows of Table IV change one element of the recipe and rescore; each block states its own reference, because the blocks use different probes (provenance and full sweeps in Appendix C).

TABLE IV  
ABLATIONS. MOST ROWS REMOVE OR REPLACE A SINGLE ELEMENT OF THE HEADLINE RECIPE; WHERE A ROW DIFFERS IN MORE THAN ONE, THE TEXT SAYS SO. EACH BLOCK NAMES ITS OWN REFERENCE AND PROBE.
<table><tr><td>Configuration</td><td>mIoU</td><td>∆</td></tr><tr><td colspan="3">SGSC conditioning (from noise, oracle GT BEV, 100 frames):</td></tr><tr><td>no conditioning</td><td>1.29</td><td>-41.0</td></tr><tr><td>BEV stream B only</td><td>34.08</td><td>-8.2</td></tr><tr><td>B+LSK3D features F (ref.)</td><td>42.30</td><td></td></tr><tr><td colspan="3">S2D2 components (100-frame probe, ∆ vs. frozen base):</td></tr><tr><td>no BEV stream</td><td>n/a</td><td>+1.1</td></tr><tr><td>no synthetic pool</td><td>n/a</td><td>~ + 1.7</td></tr><tr><td>probe reference (BEV + 32K pool)</td><td>n/a</td><td>+2.5</td></tr><tr><td>+ 3D multi-scale stream</td><td>n/a</td><td>+2.5</td></tr><tr><td colspan="3">S2D2 loss (full val, ∆ vs. headline):</td></tr><tr><td>CE on 0, no KL term</td><td>10.3</td><td>-28.2</td></tr><tr><td>KL term alone</td><td>36.1</td><td>-2.5</td></tr><tr><td>KL + Lovász + focal (ref.)</td><td>38.54</td><td></td></tr><tr><td colspan="3">Training path (full val; reference per row, see notes):</td></tr><tr><td>fitted at the endpoint (supervised control)</td><td>27.0</td><td>-11.5</td></tr><tr><td>trained only at t=T</td><td>32.74</td><td>−5.4</td></tr><tr><td colspan="3">Deployment (full val, ∆ vs. headline N=1):</td></tr><tr><td>N=1, no TTA (ref.)</td><td>38.54</td><td></td></tr><tr><td>N=4 (interior peak)</td><td>38.65</td><td>+0.11</td></tr><tr><td>N=100 (full schedule)</td><td>38.2</td><td>-0.34</td></tr><tr><td>without EMA weights#</td><td>n/a</td><td>-3.5</td></tr></table>

<sup>♮</sup>released codebase at 40K, against the matched full-schedule arm’s 38.1%. <sup>♯</sup>100-frame probe at step 35K, N=100, not full val. n/a: the probe stores aggregate deltas only. ∼: probe precision; Appendix C gives the pool as a +0.8 marginal.

Conditioning carries the generator. Removing it entirely leaves SGSC barely off the floor at 1.29%; the BEV stream alone carries it to 34.08%, and the per-voxel feature stream adds a further +8.2 to an oracle ceiling of 42.30%. These are oracle-BEV ceilings, not deployable scores. The deployed SGSC of Table I uses its own BEV head and reaches 30.5%. Guidance quality matters as much as its presence: swapping the oracle BEV for a predicted one at fixed weights leaves bulk geometry intact while the two-wheeler classes collapse, as Fig. 8 shows side by side. Neither module manufactures semantics that the conditioning does not carry.

The BEV stream is the dominant component of the refiner, and the synthetic pool the second. Removing the base-derived BEV stream drops the uplift from +2.5 to +1.1 pp; removing the 32K synthetic pool drops it to ∼+1.7. Data- and model-side contributions are both positive and do not cancel. Two negatives sharpen the reading: adding a 3D multi-scale conditioning stream on top is a null result within noise, so we keep the BEV stream alone. Applied to TALoS’s already-adapted SCPNet the operator lifts least (+0.9), that base having already moved some of the mass the correction targets. Pool size does not scale naively (32K is the best measured point), and conditioning on a refined rather than base-derived BEV leaves quality inside headline noise, so we use the cheaper unrefined stream.

![](images/085662ee7a1669b114226f18dc982f4bde87f8294054ab45121b8cb2bd739f8a.jpg)  
Fig. 8. The generator inherits the quality of its guidance. Ground-truth scene (left) and two $2 5 6 ^ { 2 } \times$ 32 SGSC completions (val seq. 08, frame 6) from one development checkpoint (from-noise T=100, identical weights, only the BEV conditioning differs: oracle vs. predicted). A predicted BEV keeps the road surface but collapses the road users.

The KL posterior term is necessary and not sufficient. Replacing it with cross-entropy on the clean target collapses the model to 10.3%, 25.9 pp below its own input. Yet KL without the classbalanced surrogates improves nothing: it reaches 36.1%, the base’s 36.17% to within a tenth of a point, stable and inert. The +2.36 pp headline is therefore an interaction, with KL anchoring the per-timestep posterior while those surrogates (Lovasz–´ softmax [58] and focal weighting) supply the only gradient that pushes the model off its source. That is why the per-class gain of Table II concentrates on the classes those surrogates target.

What the multi-timestep training path buys. At N=1 the deployed operator is one forward pass whose source term contributes under 1% of the output $( \bar { \alpha } _ { T } \approx 5 . 6 \times 1 0 ^ { - 3 } )$ , so what does the formulation buy over training the same network to map $\mathbf { x } _ { \mathrm { s r c } }$ to the target in one supervised step? Fitting the endpoint directly is that arm: it floods rather than copies, never reaching the identity map inside its own hypothesis class, so it bounds the question loosely at best. Restricting t to $T$ asks something narrower. We ran it on the released codebase, matched in batch size, seed, data and every loss coefficient. At 40K steps it reaches 32.74% against a matched full-schedule arm’s 38.1%, falling 3.4 pp below the 36.17% it was handed and flooding the grid rather than copying it. It does not isolate the schedule, though. At t=T the state equals $\mathbf { x } _ { \mathrm { s r c } }$ on all but $\hat { \alpha } _ { T }$ of voxels, where both brackets of Eq. (8) collapse onto the source, so the KL yields no gradient; the auxiliary weighting and the 100 mixtures each pair is seen at move with it. All are consequences of restricting $t ,$ not choices we could have matched, so the arm prices the variational path as a whole and attributes the gap to none of its parts.

One step suffices at deployment. Sweeping the correction from 1 to 100 steps on full val, quality peaks at a shallow 38.65% at N=4. Both the 100-step run and the deployed N=1 sit within the assumed single-seed band of that peak (0.45 and 0.11 pp below it). Nothing in the sweep is resolvable, which is the point: one step buys a 100× speedup at no measurable cost, and needs neither the retraining nor the distillation that step reduction usually requires [68]–[70]. EMA weights are required (−3.5 pp without). The headline consumes one sweep at deployment, though its training accumulated several; a retrain matched to single-frame at both reaches 38.4%, 0.1 pp from the released checkpoint and inside the assumed 0.3–0.5 pp single-seed band (Appendix C).

## D. Transfer and Secondary Tasks

The gains survive outside the training distribution. Refining the frozen SemanticKITTI checkpoint zero-shot, with no target labels and no fine-tuning, lifts SSCBench-KITTI360 [62] from mIoU 5.8 to 6.2 and completion IoU 18.1 to 19.5, scored over the 16 shared classes rather than its own 18. The mIoU delta sits inside the assumed single-seed band, so here the claim rests on completion IoU. On SemanticPOSS [63], a different sensor, both clear the band: mIoU 1.0 → 6.5% and completion IoU 31.8 → 54.9%. Our SemanticPOSS ground truth is reconstructed rather than official, so these are not head-to-head with TALoS. TALoS reports 9.6% from a 7.6% SCPNet base under a SemanticPOSS completion ground truth it does not describe (protocol in Appendix E).

The synthetic pool that feeds this refiner reshapes the training distribution rather than merely enlarging it: $\mathrm { P S } ^ { 3 }$ raises rare-class voxel frequency by up to 3,907× (Fig. 9). Its measured benefit is real but bounded, and specific to the voxel-grid-native base we test. Under a retrain matched to single-frame input at both ends, only the pool’s peak location transfers, not its margin over real-only (Appendix C).

The same operator also refines 2D BEV perception (Fig. 11), though most of that headline is the projection rather than the operator. The pipeline reaches 36.1% BEV mIoU, of which 34.8 is a parameter-free projection of the frozen 3D base (the topmost non-empty class per column, already +7.8 pp over prior art), and $\bar { \mathbf { S } ^ { 2 } } \bar { \mathbf { D } ^ { 2 } }$ adds +1.3. Swapping the 3D denoiser for a 2D variant and the source for the base-derived BEV, the same residual-transport machinery applies unchanged, and the result is +9.1 pp over 2D S3CNet’s 27.0% (full comparison in Appendix F).

![](images/050a308beb6c37aa183b8c7d119f7b685e06934b1cf42328a7357826c344827d.jpg)

Fig. 9. PS<sup>3</sup> rebalances the long tail. Per-class voxel frequency (log scale, sorted ascending by real-only frequency), real-only vs. real+synth-32K. The annotated factors are the synthetic pool’s own frequency against real-only (4.6×–6,239×); the pooled bars gain 3.2×–3,907×.  
![](images/8bbb544b6c1a41b6ee55e747b0ccea943257a0c84ddead25c2a16fa5fa432f9d.jpg)  
Fig. 10. Six rare-class val scenes, one per class, in descending order of D -TTA-vs-base IoU gain (+40.1 to +20.1). Left: SCPNet base; right: S<sup>2</sup>D<sup>2</sup> (32K multi-frame + D TTA), sharing one camera so differences are the predictions, not the framing. These are the scenes where the correction helps most; the regime where it hurts, a weaker base whose source under-supplies rare classes, is Fig. 12.

## E. Where the Method Loses

We close on the opposite question: where a reader should expect this operator to fail. Four limits are measured rather than argued.

It cannot exceed its guidance. The generator inherits the ceiling of its conditioning, and the refiner inherits the ceiling of its source. Both were quantified above: predicted rather than oracle BEV collapses the two-wheeler classes while bulk geometry holds (Fig. 8), and one correction step on the JS3C-Net base costs two of three VRU classes. Within the evaluated volume the failure mode is not hallucination but erasure. The never-observed volume the official metric excludes speaks to neither. On val seq. 08 frame 001810 one step erases both pedestrians that the frozen JS3C-Net base had recovered, taking person IoU from 36.6 to 0.7%, a source-bounded erasure typical of that base. Restricted to frames where the base recovers the class at all, one step lowers person IoU on 74.7% of them and bicyclist IoU on 92.5%. On the SCPNet base the same two rates are 24.0% and 23.5%,

![](images/f8511554cbcd30392e3d8c322a4ccb4a09d626db821ddf404e76b793de139034.jpg)  
Fig. 11. Qualitative BEV semantic predictions on SemanticKITTI validation. Each panel is a different val seq-08 frame, not another view of one scene; within a panel, left to right: the sparse LiDAR input, the ground-truth BEV, and our predicted BEV. Colour key (this BEV renderer’s own palette, not the voxel key): grey: road; dark purple: sidewalk; magenta: parking; cyan: building; dark green: vegetation; light green: terrain.

![](images/7a6297be7831a2a643de46c987384ec6baad09394d9f22caa0ec72b070057fb6.jpg)  
(a) JS3C-Net base

![](images/1f42573eb59fa1dfdf36fb8ecbdb6257944c59ea925505347e28d36f39766551.jpg)  
(b) JS3C-Net + S<sup>2</sup>D<sup>2</sup>

![](images/58b562c34ae7f180866aee207e6dd51661d8dc38213f8123b84bcc7d07dbcf97.jpg)  
(c) ground truth

Fig. 12. Failure case: S<sup>2</sup>D<sup>2</sup> erases pedestrians on the JS3C-Net base (val seq. 08, frame 001810). Person and bicyclist voxels keep their semantic colour; the rest of the scene is muted to grey. Both pedestrians present in the ground truth (c) are recovered by the frozen base (a) and all but eliminated in the refined output (b), whose person IoU falls to 0.7%.

which is why we scope the transfer claim to voxel-grid-native sources (Fig. 12; both per-frame breakdowns in Appendix F).

The rarest class does not reproduce. Motorcyclist’s +8.3 pp on val falls to +0.3 on retrain and to +1.3 against the published test row. It is the single class the safety story would most like to claim, and we do not claim it.

We do not lead on safety. Among single-sweep rows S3CNet leads every VRU class, at VRU-IoU 32.6 against our 21.6, on an uneven profile whose completion IoU trails every other published test row. On the metric we introduce precisely because mIoU hides the tail, a 2021 method is ahead of us, and closing that gap is open work.

Latency is not yet repaid. The correction pass costs 107 ms, dropping the deployed base+refiner pipeline to 3.23 FPS endto-end on an idle H100, against the frozen base’s 4.95. Neither matches the sensor’s 10 Hz sweep cadence [71] (the refiner’s marginal 9.33 FPS is an incremental pass, not a deployable rate). Charged for that latency at $T _ { w } { = } 1 . 0 \ : \mathrm { s }$ , DW-VRU-IoU moves 50.7 → 49.9 on val and $5 4 . 9  4 9 . 7 $ on test: one-step refinement trails the frozen base inside a one-second window. That is a statement about the equal-weight three-class mean, whose crossing sits past $T _ { w } \approx 1 . 2 \mathrm { s }$ on val and ≈ 3.2 s on test [72]; taken singly, no class crosses at all (window sweep in Appendix C, extended limitations in Appendix H). On test, therefore, the accuracy it buys does not yet pay for the time it costs. The 100-step sampler is inadmissible under any budget, so the one-step reduction is what admits refinement into the latency envelope at all. A faster denoiser, not a better one, is what this result asks for next.

## V. CONCLUSION

We recast outdoor LiDAR semantic scene completion from a discriminative mapping into a generative process (GSSC) driven by a single discrete-diffusion formulation across data synthesis (PS<sup>3</sup>), base-free completion (SGSC), and baseconditioned refinement (S<sup>2</sup>D<sup>2</sup>). Conditioned on a frozen base, S<sup>2</sup>D<sup>2</sup> reaches 38.8% hidden-test mIoU at one step with no test-time augmentation, to our knowledge the best causal, single-sweep, single-sample leaderboard result to date. We also asked whether the operator is tied to the base it trained beside: it lifts three frozen bases spanning the field, retraining none and touching no architecture. The frozen checkpoint transfers zero-shot to SemanticPOSS and, on completion IoU, to SSCBench-KITTI360 (Section IV-D), and the same generative view yields PS<sup>3</sup>, a long-tail-rebalancing expansion of the SemanticKITTI training set. Generative modelling need not replace the discriminative pipeline to improve it: one diffusion formulation can supply its training data, complete a scene from noise, and correct what a deployed stack already produces.

Limitations. The refiner cannot exceed its source: on a weaker base it can erase a rare class rather than recover it, costing two of the three vulnerable-road-user classes on the JS3C-Net base (Section IV-E). Besides, its correction pass does not pay for its latency inside a one-second window. Our evidence for generality is two dense categorical tasks, 3D completion and BEV segmentation. Whether the correction reaches any output that is a per-element categorical simplex is untested rather than shown (Appendix H).

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (Grant 624B1006) and the Shanghai Science and Technology Committee (Grant 24511103900).

## REFERENCES

[1] Z. Xia, Y. Liu, X. Li, X. Zhu, Y. Ma, Y. Li, Y. Hou, and Y. Qiao, “SCPNet: Semantic scene completion on point cloud,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 17 642–17 651.

[2] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[3] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “VAD: Vectorized scene representation for efficient autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 8340–8350.

[4] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall, “SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 9297–9307.

[5] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[6] A. Michotte, G. Thines, and G. Crabb\` e,´ Les complements amodaux des´ structures perceptives, ser. Studia Psychologica. Louvain, Belgium: Publications Universitaires, 1964.

[7] G. Kanizsa, Organization in Vision: Essays on Gestalt Perception. New York: Praeger, 1979.

[8] S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser, “Semantic scene completion from a single depth image,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1746–1754.

[9] J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg, “Structured denoising diffusion models in discrete state-spaces,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 17 981–17 993.

[10] E. Hoogeboom, D. Nielsen, P. Jaini, P. Forre, and M. Welling, “Argmax´ flows and multinomial diffusion: Learning categorical distributions,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 12 454–12 465.

[11] L. Roldao, R. de Charette, and A. Verroust-Blondet, “LMSCNet:˜ Lightweight multiscale 3D semantic completion,” in Proceedings of the International Conference on 3D Vision (3DV), Fukuoka, Japan, 2020, pp. 111–119.

[12] C. B. Rist, D. Emmerichs, M. Enzweiler, and D. M. Gavrila, “Semantic scene completion using local deep implicit functions on LiDAR data,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 10, pp. 7205–7218, 2022.

[13] R. Cheng, C. Agia, Y. Ren, X. Li, and B. Liu, “S3CNet: A sparse semantic scene completion network for LiDAR point clouds,” in Conference on Robot Learning. PMLR, 2021, pp. 2148–2161.

[14] X. Yang, H. Zou, X. Kong, T. Huang, Y. Liu, W. Li, F. Wen, and H. Zhang, “Semantic segmentation-assisted scene completion for LiDAR point clouds,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 3555–3562.

[15] X. Yan, J. Gao, J. Li, R. Zhang, Z. Li, R. Huang, and S. Cui, “Sparse single sweep LiDAR point cloud segmentation via learning contextual shape priors from scene completion,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 4, 2021, pp. 3101–3109.

[16] J. Mei, Y. Yang, M. Wang, T. Huang, X. Yang, and Y. Liu, “SSC-RS: Elevate LiDAR semantic scene completion with representation separation and BEV fusion,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.

[17] H.-K. Jang, J. Kim, H. Kweon, and K.-J. Yoon, “TALoS: Enhancing semantic scene completion via test-time adaptation on the line of sight,” in Advances in Neural Information Processing Systems, 2024.

[18] A.-Q. Cao, A. Dai, and R. de Charette, “PaSCo: Urban 3D panoptic scene completion with uncertainty awareness,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[19] J. Wilson, J. Song, Y. Fu, A. Zhang, A. Capodieci, P. Jayakumar, K. Barton, and M. Ghaffari, “MotionSC: Data set and network for real-time semantic mapping in dynamic environments,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8439–8446, 2022.

[20] Y. Liu, X. Li, X. Li, L. Qi, C. Li, and M.-H. Yang, “Pyramid diffusion for fine 3D large scene generation,” in European Conference on Computer Vision. Springer, 2024, pp. 71–87.

[21] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An open urban driving simulator,” in Conference on Robot Learning (CoRL), 2017, pp. 1–16.

[22] Y. Xiong, W.-C. Ma, J. Wang, and R. Urtasun, “Learning compact representations for LiDAR completion and generation,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1074–1083.

[23] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely embedded convolutional detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.

[24] A. Xiao, J. Huang, D. Guan, K. Cui, S. Lu, and L. Shao, “PolarMix: A general data augmentation technique for LiDAR point clouds,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[25] X. Zhang, B. Crowe, and C. Heckman, “Octree diffusion for semantic scene generation and completion,” arXiv preprint arXiv:2509.16483, 2025.

[26] V. Zyrianov, H. Che, Z. Liu, and S. Wang, “LidarDM: Generative LiDAR simulation in a generated world,” in IEEE International Conference on Robotics and Automation (ICRA), 2025.

[27] L. Zhou, Y. Du, and J. Wu, “3D shape generation and completion through point-voxel diffusion,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5806–5815.

[28] X. Zeng, A. Vahdat, F. Williams, Z. Gojcic, O. Litany, S. Fidler, and K. Kreis, “LION: Latent point diffusion models for 3D shape generation,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 10 021–10 039.

[29] S. Luo and W. Hu, “Diffusion probabilistic models for 3D point cloud generation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2836–2844.

[30] J. Tang, Y. Nie, L. Markhasin, A. Dai, J. Thies, and M. Nießner, “DiffuScene: Denoising diffusion models for generative indoor scene synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[31] H. Ran, V. Guizilini, and Y. Wang, “Towards realistic scene generation with LiDAR diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[32] H. Cao and S. Behnke, “DiffSSC: Semantic LiDAR scan completion using denoising diffusion probabilistic models,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

[33] G. Wang, Z. Wang, P. Tang, J. Zheng, X. Ren, B. Feng, and C. Ma, “OccGen: Generative multi-modal 3D occupancy prediction for autonomous driving,” in European Conference on Computer Vision. Springer, 2024.

[34] J. Lee, W. Im, S. Lee, and S.-E. Yoon, “Diffusion probabilistic models for scene-scale 3D categorical data,” arXiv preprint arXiv:2301.00527, 2023.

[35] J. Lee, S. Lee, C. Jo, W. Im, J. Seon, and S.-E. Yoon, “SemCity: Semantic scene generation with triplane diffusion,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[36] L. Nunes, R. Marcuzzi, B. Mersch, J. Behley, and C. Stachniss, “Scaling diffusion models to real-world 3D LiDAR scene completion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 770–14 780.

[37] D. Zhang, J. Lu, H. Yang, L. Bao, and B. Song, “Enhancing 3D semantic scene completion with a refinement module,” arXiv preprint arXiv:2512.18363, 2025.

[38] H. Shi, S. Wang, J. Zhang, X. Yin, Z. Wang, Z. Zhao, G. Wang, J. Zhu, K. Yang, and K. Wang, “Offboard occupancy refinement with hybrid propagation for autonomous driving,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 10, pp. 16 001–16 016, 2025.

[39] F. Liu, H. Yu, J. Huang, X. Hua, H. Liu, Z. Ma, Z. Wang, Z. Du, and L. Sun, “RefineCatDiff: Toward high-quality medical image segmentation via a categorical diffusion refinement framework,” Advanced Intelligent Systems, vol. 7, no. 12, p. e202401125, 2025.

[40] Z. Yue, J. Wang, and C. C. Loy, “ResShift: Efficient diffusion model for image super-resolution by residual shifting,” in Advances in Neural Information Processing Systems (NeurIPS), 2023, arXiv:2307.12348.

[41] G.-H. Liu, A. Vahdat, D.-A. Huang, E. A. Theodorou, W. Nie, and A. Anandkumar, “I<sup>2</sup>SB: Image-to-image Schrodinger bridge,” in¨ International Conference on Machine Learning (ICML), 2023, arXiv:2302.05872.

[42] B. Li, K. Xue, B. Liu, and Y.-K. Lai, “BBDM: Image-to-image translation with Brownian bridge diffusion models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, arXiv:2205.07680.

[43] L. Zhou, A. Lou, S. Khanna, and S. Ermon, “Denoising diffusion bridge models,” in International Conference on Learning Representations (ICLR), 2024, arXiv:2309.16948.

[44] P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner, and J. Brandstetter, “PDE-Refiner: Achieving accurate long rollouts with neural PDE solvers,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[45] A. Bansal, E. Borgnia, H.-M. Chu, J. S. Li, H. Kazemi, F. Huang, M. Goldblum, J. Geiping, and T. Goldstein, “Cold diffusion: Inverting arbitrary image transforms without noise,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[46] M. Delbracio and P. Milanfar, “Inversion by direct iteration: An alternative to denoising diffusion for image restoration,” Transactions on Machine Learning Research (TMLR), 2023.

[47] M. Wang, H. Ding, J. H. Liew, J. Liu, Y. Zhao, and Y. Wei, “SegRefiner: Towards model-agnostic segmentation refinement with discrete diffusion process,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[48] I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Q. Chen, G. Synnaeve, Y. Adi, and Y. Lipman, “Discrete flow matching,” in Advances in Neural Information Processing Systems, 2024.

[49] A. Campbell, J. Yim, R. Barzilay, T. Rainforth, and T. Jaakkola, “Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design,” in Proceedings of the 41st International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 5453–5512.

[50] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[51] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations, 2023.

[52] HDL-64E S3: High Definition LiDAR Sensor — User’s Manual and Programming Guide, Velodyne LiDAR, Inc., 2013, document 63- HDL64ES3 Rev. G. The per-beam vertical angles used by our resampler are not from this manual but from the ROS Velodyne driver calibration file 64e\_s2.1-sztaki.yaml (https://github.com/ros-drivers/velodyne).

[53] L. Winiwarter, A. M. Esmor´ıs Pena, H. Weiser, K. Anders, J. Mart´ınez Sanchez, M. Searle, and B. H´ ofle, “Virtual laser scanning with¨ HELIOS++: A novel take on ray tracing-based simulation of topographic full-waveform 3D laser scanning,” Remote Sensing of Environment, vol. 269, p. 112772, 2022.

[54] S. Manivasagam, S. Wang, K. Wong, W. Zeng, M. Sazanovich, S. Tan, B. Yang, W.-C. Ma, and R. Urtasun, “LiDARsim: Realistic LiDAR simulation by leveraging the real world,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 167–11 176.

[55] T. Feng, W. Wang, F. Ma, and Y. Yang, “LSK3DNet: Towards effective and efficient 3D perception with large sparse kernels,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[56] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[57] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.

[58] M. Berman, A. R. Triki, and M. B. Blaschko, “The Lovasz-softmax loss:´ A tractable surrogate for the optimization of the intersection-over-union measure in neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4413–4421.

[59] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268–9277.

[60] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations (ICLR), 2021.

[61] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in International Conference on Learning Representations (ICLR), 2023.

[62] Y. Li, S. Li, X. Liu, M. Gong, K. Li, N. Chen, Z. Wang, Z. Li, T. Jiang, F. Yu, Y. Wang, H. Zhao, Z. Yu, and C. Feng, “SSCBench: A large-scale 3D semantic scene completion benchmark for autonomous driving,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

[63] Y. Pan, B. Gao, J. Mei, S. Geng, C. Li, and H. Zhao, “SemanticPOSS: A point cloud dataset with large quantity of dynamic instances,” in IEEE Intelligent Vehicles Symposium (IV), 2020.

[64] L. Roldao, R. de Charette, and A. Verroust-Blondet, “3D semantic scene˜ completion: A survey,” International Journal of Computer Vision, vol. 130, no. 8, pp. 1978–2005, 2022.

[65] International Organization for Standardization, “ISO 21448:2022 Road Vehicles — Safety of the Intended Functionality,” 2022.

[66] Spconv Contributors, “spconv: Spatially sparse convolution library,” https: //github.com/traveller59/spconv, 2022.

[67] T. Martyniuk, J. Seele, A. Boulch, G. Puy, R. Marlet, and R. de Charette, “Exploring easy boosts for LiDAR semantic scene completion,” in Proceedings of the IEEE International Conference on Image Processing (ICIP), 2026, pp. 1–6.

[68] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” in International Conference on Learning Representations (ICLR), 2022.

[69] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 211–32 252.

[70] S. Zhang, A. Zhao, L. Yang, Z. Li, C. Meng, H. Xu, T. Chen, A. Wei, P. P. Gu, and L. Sun, “Distilling diffusion models to efficient 3D LiDAR scene completion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[71] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The KITTI dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.

[72] AASHTO, A Policy on Geometric Design of Highways and Streets, 7th ed. American Association of State Highway and Transportation Officials, 2018.