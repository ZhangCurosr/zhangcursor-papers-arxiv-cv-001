# FreeOcc: Decoupling Ego-Motion for Efficient 4D Occupancy Forecasting via Continuous Flow Matching

Zeping Zhang University of Ottawa Zeping.Zhang@uottawa.ca

Samy Metari National Research Council Canada Samy.Metari@nrc-cnrc.gc.ca

Zhuoya Zhao University of Ottawa zoezhao0427@gmail.com

Robert Laganière<sup>∗</sup> University of Ottawa laganier@uottawa.ca

## Abstract

Current 4D occupancy world models use heavy cross-attention or deep recurrence to disentangle ego-motion from scene dynamics. This requires significant computational capacity. However, this effort is largely redundant since the ego-vehicle’s intended future motion is provided by the downstream motion planner. We present FreeOcc, a lightweight continuous-time world model that introduces explicit SE(2) geometric decoupling into a conditional flow-matching framework. By mathematically factoring out self-motion, we create a purely dynamic feature space. This space supports attention-free temporal fusion and allows physics-constrained ODEs to operate strictly on residual agent kinematics. To maximize deployment efficiency, FreeOcc introduces an uncertainty-aware blending mechanism. This routes wellobserved voxels to a single-pass decoder, reserving the ODE solver exclusively for ambiguous, high-uncertainty regions. Requiring only 28.3M parameters—roughly 40% of comparable models—our optimal ensemble mode (FreeOcc-E) runs at 27 FPS. On nuScenes, FreeOcc achieves state-of-the-art forecasting performance with a 0.47 m Near Field Chamfer Distance (NFCD) and 32.1% mIoU. Furthermore, the model transfers zero-shot to SemanticKITTI. It also proves highly robust, retaining 92.8% of its ideal mIoU under significant sensor packet loss and timestamp jitter.

## 1 Introduction

Anticipating the future 3D structure of a driving scene is critical for pre-emptive motion planning. Dense 4D occupancy forecasting addresses this by predicting a sequence of future voxelised binary volumes $\{ O _ { \tau } \} _ { \tau = 1 } ^ { \dot { N } }$ directly from recent sensor history, elegantly capturing complex scene geometry without relying on explicit object detection or tracking.

Recent works tackle this via: autoregressive token generation Zheng et al. [2024], Zhang et al. [2024], continuous flow formulations Shi et al. [2024], geometric proxy representations Yu et al. [2026], and unsupervised implicit fields Agro et al. [2024], Yang et al. [2024]. However, these methods generally rely on cross-attention or deep recurrence to separate the ego-vehicle’s motion from observed scene dynamics. This entanglement inflates parameter counts. It also requires expensive temporal attention and complicates error control at long horizons. Crucially, the ego-vehicle’s future trajectory is typically available at inference time as the intended route provided by the motion planner. While explicit SE(2) warping in BEV space is well established Jaderberg et al. [2015], recent generative world models have largely favoured learned temporal attention instead. In this setting, learned alignment is unnecessary: a single geometric warp factors out ego-motion and leaves the model to focus on residual dynamics, occlusion evolution, and far-field extrapolation. Driven by this insight, we propose FreeOcc. By mathematically factoring out self-motion, our architecture simplifies the forecasting pipeline, yielding three primary contributions:

• Attention-Free Temporal Fusion: Removing ego-motion allows a lightweight ConvGRU to effectively capture residual dynamics. By avoiding dense cross-attention, FreeOcc processes a $2 0 0 \times 2 0 0 \times 1 6$ voxel resolution at 27 FPS. Furthermore, it achieves this with only 28.3M parameters, which is roughly 40% of the size of comparable world models.

• Physics-Constrained Generation: The motion-decoupled feature space stabilizes the conditioning for flow matching. Physical constraints—such as velocity smoothness, mass conservation, and ray-traced free-space—can be applied directly to agent kinematics without relying on restrictive geometric proxies.

• Uncertainty-Aware Routing: Decoupling self-motion separates true scene ambiguity—such as dynamic objects and occlusion boundaries—from misalignment artefacts. The decoder’s per-voxel confidence is used to route well-observed regions to a single-pass decoder, while the continuous ODE solver handles highly uncertain areas.

On nuScenes, FreeOcc achieves state-of-the-art forecasting performance (0.47 m NFCD, 32.1% mIoU) and transfers zero-shot to SemanticKITTI. The model remains robust under significant sensor dropout and temporal jitter, retaining 92.8% of its ideal mIoU.

## 2 Related Work

4D occupancy forecasting. Recent world models approach future scene prediction through autoregressive token generation Zheng et al. [2024], discrete diffusion Zhang et al. [2024], implicit fields Agro et al. [2024], or camera-based point cloud forecasting Yang et al. [2024]. However, these paradigms typically dedicate massive capacity to entangling ego-motion and scene dynamics via deep recurrence or cross-attention. FreeOcc bypasses this bottleneck by explicitly factoring out self-motion via algebraic SE(2) warping (see Section 3.4), allowing it to directly predict dense voxel grids via physics-constrained continuous flow matching (CFM) without relying on discrete tokenisation or implicit rendering.

Flow matching and continuous generative models. Continuous normalising flows trained via flow matching Lipman et al. [2023] offer a simulation-free generative objective. Recent works like Streaming Flow Shi et al. [2024], and latent flow models Liu et al. [2025] adapt this to occupancy and LiDAR streams. Yet, they still rely on computationally heavy dense cross-attention for temporal alignment. FreeOcc preserves the continuous expressivity of flow matching but replaces the attention bottleneck with an explicit geometric warp, further regularising the generative trajectory with physics informed constraints (velocity smoothness, mass conservation, ray-traced free-space).

Physics-informed perception. While integrating physical priors into perception is wellestablished—such as range-aware encoding Lang et al. [2019] or heteroscedastic uncertainty Kendall and Gal [2017]—applying structural constraints to dense 4D forecasting remains challenging. Recent concurrent works like SuperOcc Yu et al. [2026] and DTT Xu et al. [2025] enforce geometric structure by restricting the output space to predefined proxies (e.g., superquadrics or delta residuals), which inevitably sacrifices fine-grained fidelity on irregular shapes. Rather than simplifying the output representation, FreeOcc guides the full-resolution generative trajectory at the loss level. This ensures physically plausible dynamics while retaining exact voxel-level structural fidelity.

## 3 Methodology

FreeOcc forecasts dense 4D occupancy from LiDAR with an explicit task division: geometry handles ego-motion, while learning handles residual scene dynamics. By mathematically isolating the vehicle’s known trajectory via differentiable SE(2) warping, we remove the need for learned temporal alignment, allowing the generative model to focus entirely on independently moving agents, occlusion evolution, and long-horizon uncertainty.

![](images/ddd1864ba1fc8600d6b213711db83447b73bc46380e6572a7128da98b2c1f7f7.jpg)  
Figure 1: Overview of FreeOcc. Previous sweeps are encoded into BEV, fused by a continuoustime temporal aggregator, propagated to future ego frames by SE(2) warping, and decoded by the Discriminative Occupancy Decoder plus the Physics-Aware Continuous Flow Matching. The final prediction blends both paths using decoder uncertainty.

## 3.1 Overall Architecture

The architecture of FreeOcc (depicted in Figure 1) is driven by this principle of explicit geometric decoupling, operating in three distinct phases:

• Continuous-Time Context Fusion (Sections 3.2 and 3.3): Given K asynchronous previous sweeps $\mathcal { P } = \{ P _ { t - K + 1 } , \ldots , P _ { t } \}$ , we encode each sweep into BEV and fuse them via timestampaware recurrent aggregation into a unified present-state conditioning map $\mathcal { H } _ { t } \in \mathbb { R } ^ { C \times H \times W }$

• Explicit Ego-Motion Decoupling (Section 3.4): Rather than learning temporal alignment, an ego-aware propagator explicitly warps $\mathcal { H } _ { t }$ to future coordinate frames. The resulting BEV priors $\{ \breve { C } _ { \tau } \} _ { \tau = 1 } ^ { N }$ isolate the residual scene changes.

• Uncertainty-Guided Dual-Path Generation (Sections 3.5 to 3.7): Each prior $C _ { \tau }$ is decoded through two branches to maintain inference speed. Confident, well-observed regions with lower uncertainty pass through a fast discriminative decoder. The resulting uncertainty map then selectively routes only the highly uncertain areas to a physics-aware conditional flow-matching ODE.

We use the following notation throughout. BEV features have spatial size $H \times W$ and channel width C; voxel grids have size $X \times Y \times Z$ at resolution δ (m/voxel). The current time is t, the future index is $\tau \in \{ 1 , \ldots , N \}$ , and the internal flow time is $s \in [ 0 , 1 ]$

## 3.2 LiDAR Encoder with Sensor-Range Encoding

Each sweep $P _ { i } = \{ p _ { j } \} _ { j = 1 } ^ { M _ { i } }$ , where $M _ { i }$ is the number of points in the sweep and $p _ { j } \in \mathbb { R } ^ { 5 }$ (coordinates, intensity, ring), is first processed by a Pillar Feature Net Lang et al. [2019] (see Figure 1), which encodes points into a sparse pillar representation. Specifically, a shared three-layer $\bar { { \bf M L P } }$ maps each point within a pillar to $\mathbf { \mathbb { R } } ^ { C }$ , and per-pillar max-pooling yields a pseudo-image $\pmb { \check { F } } _ { i } \in \mathbb { R } ^ { C \times H \times \mathbf { \check { W } } }$

We further add a learned sinusoidal range encoding $\rho ( r _ { h w } )$ to each BEV cell, giving the encoder an explicit distance prior that reflects the drop in LiDAR sampling density with range. A lightweight ResNet–FPN backbone He et al. [2016], Lin et al. [2017a] then produces the per-sweep feature $F _ { i } \in \mathbb { R } ^ { C \times H \times W }$ ; the exact encoding formula is given in Section ${ \mathrm { A . 1 } }$

## 3.3 Continuous-Time Temporal Aggregator

After encoding the individual sweeps into spatial features $F _ { i }$ , the model must aggregate them over time. Real-world LiDAR data suffers from scheduling jitter, delays and packet loss. Instead of learning temporal alignment with attention, FreeOcc encodes each sweep at its measured arrival time and lets a lightweight ConvGRU absorb the irregular history.

For the i-th sweep at absolute time $t _ { i } ,$ we compute $\Delta t _ { i } = t _ { i } - t \le 0$ and encode it with an exponentially spaced sinusoidal timestamp embedding $\phi ( \Delta t _ { i } )$ (detailed in Section A). We then refine it with a learned projection $\psi : \mathbb { R } ^ { C ^ { \bullet } }  \mathbb { R } ^ { C }$ . A delay-aware confidence gate discounts stale observations:

![](images/c68b817ffd8ea0a7b0e5c0703c4e8125685578370b8dc082e310b224c81adc7d.jpg)  
Figure 2: Ego-aware future propagator. At each horizon, the warped BEV—obtained via SE(2) ego-motion alignment—is explicitly added to the residual scene dynamics predicted by a ConvGRU and a residual projector.

$$
g _ { i } = \sigma \big ( \mathrm { R e L U } ( w _ { 1 } ~ \psi ( \phi ( \Delta t _ { i } ) ) ) + b \big ) ~ \in ~ ( 0 , 1 ) ,\tag{1}
$$

where $\sigma$ denotes the sigmoid activation function, $w _ { 1 }$ is a learned weight matrix, and b is an additive bias initialized $\mathbf { t o + } 2 \left( \sigma ( 2 ) \approx 0 . 8 8 \right)$ .

The time-aware feature is ${ \tilde { F _ { i } } } = g _ { i } \cdot F _ { i } + \psi ( \phi ( \Delta t _ { i } ) )$ . During training, we additionally perturb sweep timestamps and randomly drop non-current sweeps so the GRU sees both asynchronous and incomplete histories. This combination improves robustness to timing noise and sensor dropout without adding attention parameters; the exact augmentation settings are given in Section A.1. The sequence $\{ \tilde { F } _ { i } \}$ is fused by a two-layer ConvGRU Cho et al. [2014], and the final hidden state is layer-normalised to yield $\mathbf { \bar { \mathcal { H } } } _ { t } \in \mathbb { R } ^ { C \times \mathbf { \bar { H } } \times W }$

## 3.4 Ego-Aware Future Propagator: Geometric Decoupling

With the historical context fused into the present-state representation $\mathcal { H } _ { t } .$ , FreeOcc uses explicit geometric warping (Figure 2) to remove the dominant ego-induced motion before prediction. The resulting feature space is easier to learn for both the ConvGRU and the velocity network v : the static structure is aligned by construction, and the remaining updates correspond to residual scene dynamics.

Given the intended future relative ego-pose $( \Delta x _ { \tau } , \Delta y _ { \tau } , \Delta \psi _ { \tau } )$ —recorded from odometry during training and provided by the onboard motion planner at deployment—we construct a $2 \times 3$ affine matrix:

$$
A _ { \tau } = \left[ \begin{array} { l l l } { \cos \Delta \psi _ { \tau } } & { \sin \Delta \psi _ { \tau } } & { \Delta y _ { \tau } / R } \\ { - \sin \Delta \psi _ { \tau } } & { \cos \Delta \psi _ { \tau } } & { \Delta x _ { \tau } / R } \end{array} \right] ,\tag{2}
$$

where R is the BEV half-range. The warping W applies this via bilinear grid sampling Jaderberg et al. [2015], mapping $\mathcal { H } _ { t }$ to the future coordinate frame.

At each step τ, a ConvGRU cell fuses three inputs (the warped BEV, the previous output $C _ { \tau - 1 }$ , and a trajectory encoding $E _ { \tau } = \mathrm { M L P } ( \Delta x _ { \tau } , \Delta y _ { \tau } , \hat { \Delta } \psi _ { \tau } ) \in \mathbb { R } ^ { C } )$ via channel fusion $( 1 \times 1$ convolution):

$$
\begin{array} { r } { h _ { \tau } = \mathrm { C o n v G R U } \big ( \mathrm { C o n v } _ { 1 \times 1 } [ \mathcal { W } ( \mathcal { H } _ { t } , A _ { \tau } ) \parallel C _ { \tau - 1 } \parallel E _ { \tau } ] , \ h _ { \tau - 1 } \big ) , } \end{array}\tag{3}
$$

with $h _ { 0 } = C _ { 0 } = \mathcal { H } _ { t }$ . A residual skip from the warped BEV ensures the ConvGRU learns only the residual deviation:

$$
C _ { \tau } = \mathrm { P r o j } ( h _ { \tau } ) + \mathscr { W } ( \mathscr { H } _ { t } , A _ { \tau } ) ,\tag{4}
$$

where Proj denotes the Residual Projector $\mathrm { ( a 3 \times 3 }$ convolution followed by GroupNorm and ReLU), as illustrated in Figure 2. This factorisation separates independent-agent dynamics from ego-induced geometric shift and makes the velocity-field constraints in Section 3.5 well-defined.

## 3.5 Physics-Aware Continuous Flow Matching

Because the propagator has factored out the ego-vehicle’s motion, the resulting priors $C _ { \tau }$ isolate the independent agent dynamics. This decoupled feature space provides a highly stable condition for the optimal-transport flow matching problem Lipman et al. [2023]. Let $x _ { 1 } \overset { \smile } { \in } \{ 0 , 1 \} ^ { X \times Y \times Z }$ denote the ground-truth binary occupancy and $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ . The conditional probability path is:

$$
x _ { s } = \left( 1 - s \right) x _ { 0 } + s x _ { 1 } , \quad s \in [ 0 , 1 ] ,\tag{5}
$$

with target velocity $u ( s ) = x _ { 1 } - x _ { 0 } .$ A 3D U-Net velocity network $v _ { \theta }$ is trained to predict this velocity. Let ${ \tilde { C } } _ { \tau }$ be the height-expanded version of the propagated BEV prior $C _ { \tau } ;$ the velocity network is directly conditioned on $\bar { \tilde { C } } _ { \tau }$

Because the propagator already removes ego motion, the velocity field $v _ { \theta }$ operates on residual agent dynamics. We regularise it with an anisotropic total variation so nearby voxels on the same rigid body move coherently:

$$
\mathcal { L } _ { \mathrm { v e l } } = \frac { 1 } { 3 } \sum _ { d \in \{ x , y , z \} } \frac { 1 } { | \Omega _ { d } | } \sum \bigl | v _ { \theta } ^ { ( d + 1 ) } - v _ { \theta } ^ { ( d ) } \bigr | ,\tag{6}
$$

where the sum runs over adjacent voxels along each axis.

Cooperative uncertainty-guided training. We couple the decoder’s uncertainty to the generative branch so ODE capacity is spent where single-pass prediction is least reliable. Let $\hat { s } _ { v }$ denote the detached per-voxel log-variance predicted by the decoder (Section 3.6). The flow-matching loss is weighted as

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { s , x _ { 0 } } \big [ w ( x _ { 1 } , \hat { s } _ { v } ) \big \| v _ { \theta } ( x _ { s } , s , \tilde { C } _ { \tau } ) - u ( s ) \big \| _ { 2 } ^ { 2 } \big ] ,\tag{7}
$$

where

$$
\tilde { u } _ { v } = \mathrm { c l a m p } \left( \frac { \hat { s } _ { v } - \mu _ { \mathrm { l o w } } } { \mu _ { \mathrm { h i g h } } - \mu _ { \mathrm { l o w } } } , 0 , 1 \right) , \qquad w ( x _ { 1 } , \hat { s } _ { v } ) = ( 1 + x _ { 1 } ) \cdot ( 1 + \tilde { u } _ { v } ) .\tag{8}
$$

Here $\mu _ { \mathrm { l o w } }$ and $\mu _ { \mathrm { h i g h } }$ are EMA estimates of the 5th and 95th percentiles of $\hat { s } _ { v }$ over training batches. The first factor prevents the trivial all-empty solution—most voxels are air or free space, so an unweighted model can reduce loss by predicting empty almost everywhere. The second factor upweights voxels where the decoder is uncertain, such as boundaries, dynamic objects, and far-field regions. Network details are given in Section A.1.

At inference, the ODE is solved from $s = 0$ to $s = 1$

$$
\frac { d x } { d s } = v _ { \theta } ( x , s , \tilde { C } _ { \tau } ) , \quad x ( 0 ) \sim \mathcal { N } ( 0 , I ) ,\tag{9}
$$

using a fixed-step midpoint integrator with S steps (default $\textit { S } = \ 1 0 )$ , yielding $\begin{array} { r l } { \hat { O } _ { \tau } ^ { \mathrm { o d e } } } & { { } = } \end{array}$ clam $ { \operatorname { \arg } } ( x ( 1 ) , [ 0 , 1 ] )$ ). While continuous flow matching provides exceptional fidelity for complex scene dynamics, solving the ODE across the entire volume is computationally redundant for static, well-observed geometry. Therefore, we couple our flow-matching objective with a fast discriminative decoder (Section 3.6), directing the ODE’s capacity strictly toward regions of high uncertainty.

## 3.6 Discriminative Occupancy Decoder

The second path lifts each BEV prior $C _ { \tau }$ to a voxel volume with a shallow BEV-to-3D decoder and predicts both occupancy logits $\hat { y } _ { \tau } \in \dot { \mathbb { R } } ^ { X \times Y \times Z }$ and per-voxel log-variance $\hat { s } _ { \tau } \in \mathbb { R } ^ { X \times Y \times Z }$ . We add a future-step embedding before lifting; the exact decoder and head configuration is given in Section A.1. The decoder prediction is $\hat { O } _ { \tau } ^ { \mathrm { d e c } } = \sigma ( \hat { y } _ { \tau } )$

The per-voxel log-variance $\hat { s } _ { \tau }$ serves a dual role: it enters the cooperative flow matching loss (Section 3.5) during training, and drives uncertainty-aware blending at inference (Section 3.7).

## 3.7 Uncertainty-Aware Ensemble Blending

The two generation paths offer complementary strengths: the decoder provides fast, high-recall predictions in well-observed regions, while the ODE yields high-precision samples at the cost of

iterative computation, excelling in ambiguous or dynamic areas. Rather than fixed blending (max or mean), we propose uncertainty-aware blending, which uses the decoder’s own per-voxel aleatoric uncertainty to route each voxel to the appropriate path.

We convert the predicted log-variance $\hat { s } _ { v }$ into a bounded confidence weight:

$$
w _ { v } = \sigma ( - \hat { s } _ { v } ) \in ( 0 , 1 ) .\tag{10}
$$

The final occupancy probability is a confidence-weighted interpolation:

$$
\hat { O } _ { \tau } ( v ) = w _ { v } \hat { O } _ { \tau } ^ { \mathrm { d e c } } ( v ) + ( 1 - w _ { v } ) \hat { O } _ { \tau } ^ { \mathrm { o d e } } ( v ) .\tag{11}
$$

Low log-variance implies high confidence and therefore high $w _ { v }$ , so the blend favours the fast decoder in well-observed regions. Higher log-variance lowers $w _ { v }$ and shifts weight toward the ODE in ambiguous or far-field areas. The uncertainty head is trained jointly with the decoder, so the routing signal improves alongside prediction quality.

FreeOcc supports three inference modes with distinct speed/accuracy trade-offs:

• FreeOcc-D (Decoder-only): single forward pass, real-time;

• FreeOcc-G (Generative ODE-only): S-step ODE, highest precision;

• FreeOcc-E (Ensemble): uncertainty-aware blending, best IoU at acceptable latency;

## 3.8 Physics-Aware Training Objectives

FreeOcc is trained end-to-end with a composite loss coupling the flow-matching objective $( \mathcal { L } _ { \mathrm { F M } } )$ with three specialized regularization groups:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { F M } } + \underbrace { \lambda _ { \mathrm { o c c } } \mathcal { L } _ { \mathrm { o c c } } + \lambda _ { \mathrm { l o v } } \mathcal { L } _ { \mathrm { L o v } } + \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { D i c e } } + \lambda _ { \mathrm { s p } } \mathcal { L } _ { \mathrm { s p } } + \lambda _ { \mathrm { t c } } \mathcal { L } _ { \mathrm { t c } } } _ { \mathrm { S t u r a t u r a l \ F i d e l i t y } } } \\ { + \underbrace { \lambda _ { \mathrm { r a y } } \mathcal { L } _ { \mathrm { r a y } } + \lambda _ { \mathrm { c o n s } } \mathcal { L } _ { \mathrm { c o n s } } + \lambda _ { \mathrm { v e l } } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { \mathrm { r w } } \mathcal { L } _ { \mathrm { r w } } } _ { \mathrm { P h y s i c a l \ P r i o r s } } + \underbrace { \lambda _ { \mathrm { u n c } } \mathcal { L } _ { \mathrm { u n c } } } _ { \mathrm { U n c e r t a i n t y } } , } \end{array}\tag{12}
$$

where $\lambda$ denotes the hyperparameter weights. Structural fidelity supervises the discriminative decoder, while physical priors enforce ray-traced free-space, mass conservation, and piecewisesmooth kinematics (Eq. (6)). For uncertainty calibration, we employ a heteroscedastic classification loss:

$$
\mathcal { L } _ { \mathrm { u n c } } = \frac { 1 } { | \Omega | } \sum _ { i j k } \left[ \exp ( - \hat { s } _ { i j k } ) \ell _ { \mathrm { B C E } } ( \hat { y } _ { i j k } , y _ { i j k } ) + \beta \hat { s } _ { i j k } \right] ,\tag{13}
$$

where $\ell _ { \mathrm { B C E } }$ denotes the binary cross-entropy loss applied to the predicted occupancy logits $\hat { y } _ { i j k }$ and $\beta = 0 . 5$ is a fixed regularization penalty to prevent degenerate solutions. Detailed descriptions for each term are provided in Section A.2. We omit Occ3D Tian et al. [2023] because its focus on completing unobservable hidden geometry misaligns with our target task of sensor-grounded forecasting (see Section C.3 for a detailed discussion).

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We train on nuScenes Caesar et al. [2020] and evaluate both in-domain and zero-shot on SemanticKITTI Behley et al. [2019]. For the cross-domain setting, we apply simple geometric calibration to the SemanticKITTI point clouds before evaluation. Dataset splits and calibration details are given in Section C.

Voxel Specification. We use a $[ - 4 0 , 4 0 ] \mathbf { m }$ spatial range $( 2 0 0 { \times } 2 0 0 { \times } 1 6$ grid at $\delta { = } 0 . 4$ m resolution) with $N = 6$ future and $K = 4$ history steps, but explicitly train a dedicated $[ - 7 0 , 7 0 ] \mathrm { m }$ variant to ensure fair spatial comparisons against point-based baselines in Table 1.

Table 1: 3D Geometry Forecasting on nuScenes. Evaluated via Near Field Chamfer Distance (NFCD, m) ↓, L1 Depth Error (m) ↓, and Absolute Relative Error (AbsRel, %) ↓.
<table><tr><td rowspan="3">Method</td><td colspan="3">1.0s (τ=2)</td><td colspan="3">3.0s (τ=6)</td></tr><tr><td>NFCD</td><td>L1</td><td>AbsRel</td><td>NFCD</td><td>L1</td><td>AbsRel</td></tr><tr><td>4D-Occ† Khurana et al. [2023]</td><td>1.41</td><td>1.40</td><td>10.37</td><td>1.40</td><td>1.71</td><td>13.48</td></tr><tr><td>Copilot4D† Zhang et al. [2024]</td><td>0.36</td><td>1.30</td><td>8.58</td><td>0.58</td><td>1.51</td><td>10.38</td></tr><tr><td>UnO† Agro et al. [2024]</td><td></td><td></td><td></td><td>0.89</td><td>1.15</td><td>8.50</td></tr><tr><td>ViDAR† Yang et al. [2024]</td><td>0.85</td><td>1.55</td><td>11.50</td><td>1.18</td><td>1.82</td><td>14.20</td></tr><tr><td>FreeOcc-D (ours)</td><td>0.48</td><td>1.25</td><td>8.15</td><td>0.63</td><td>1.35</td><td>9.20</td></tr><tr><td>FreeOcc-G (ours)</td><td>0.45</td><td>1.20</td><td>7.85</td><td>0.52</td><td>1.15</td><td>8.10</td></tr><tr><td>FreeOcc-E (ours)</td><td>0.41</td><td>1.18</td><td>7.60</td><td>0.47</td><td>1.08</td><td>7.45</td></tr></table>

<sup>†</sup>Numbers from original papers. L1 Depth Error is an evaluation-only metric computed by projecting predicted occupied voxels to range images and comparing against GT LiDAR depths; no L1 loss is required during training.

Metrics & Evaluation Protocol. We report two complementary metric tracks:

• Voxel metrics: class-agnostic mIoU, precision, recall, and F1 on binarized occupancy grids;

• Point metrics: NFCD, L1 depth error, and AbsRel following prior point-forecasting work Agro et al. [2024], Zhang et al. [2024].

For point metrics, we convert predicted occupied voxels to voxel-centre point clouds. Metric alignment details are given in Section C.

Implementation. FreeOcc (28.3M parameters) is trained end-to-end on 4 NVIDIA RTX 4090 GPUs for 10 epochs with AdamW (lr = 3×10<sup>−4</sup>, cosine decay, 500-step warmup, wd = 0.01), batch size 2 with 2× gradient accumulation, and FP16 mixed precision. ODE inference: S =10 midpoint steps.

Baselines. We compare against representative voxel- and point-based methods:

• 4D-Occ Khurana et al. [2023], OccWorld Zheng et al. [2024], and ViDAR Yang et al. [2024] for dense occupancy comparison;

• Copilot4D Zhang et al. [2024] and UnO Agro et al. [2024] for point-based structural geometry;

• Static Copy: Current-frame GT ego-warped to each future step, serving as a no-learned-dynamics lower bound.

For cross-modality fairness, point outputs are voxelized when evaluated on occupancy metrics, and our voxel predictions are converted to point clouds when evaluated on NFCD/L1. OccWorld is trained from scratch using the released implementation. Copilot4D and UnO are compared only on point metrics using values reported in their original papers<sup>†</sup>.

Note on recent concurrent work. Concurrent works SuperOcc Yu et al. [2026] and DTT Xu et al. [2025] rely on predefined geometric proxies. Crucially, SuperOcc targets current-frame 3D prediction rather than 4D forecasting, and DTT currently lacks publicly released code. Therefore, neither is directly applicable as a forecasting baseline under our protocol, and we include them strictly as conceptual comparisons.

## 4.2 Main Results on nuScenes

We report point-based geometry forecasting and voxel-based occupancy forecasting separately.

Geometry forecasting (Table 1). On the point-based track, FreeOcc-E reaches the strongest NFCD at τ=6 (0.47 m vs. 0.58 m for Copilot4D and 0.89 m for UnO). At τ=2, Copilot4D remains best on NFCD (0.36 vs. 0.41), while FreeOcc-E is best on L1 depth error and AbsRel. The gap closes at longer horizons: Copilot4D rises from 0.36 to 0.58 NFCD, whereas FreeOcc-E rises from 0.41 to 0.47. FreeOcc-D achieves the highest recall (49.4%) but struggles with false positives (41.3% precision). FreeOcc-E uses uncertainty-aware blending to explicitly suppress noise, a visualization is provided in Figure 8. This trade-off slightly lowers recall but boosts precision to 54.6%, securing the best F1 score. FreeOcc-E also improves over FreeOcc-G, showing that uncertainty-aware blending is stronger than either branch alone.

Table 2: 4D Occupancy Forecasting on nuScenes. Evaluated via Class-Agnostic 3D Voxel mIoU (%) ↑.
<table><tr><td rowspan="2">Method</td><td colspan="6">Forecasting Horizon mIoU (%)</td><td colspan="4">Average Metrics (%)</td></tr><tr><td>τ=1</td><td>τ=2</td><td>τ=3</td><td>τ=4</td><td>τ=5</td><td>τ=6</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Static Copy</td><td>31.5</td><td>22.1</td><td>14.8</td><td>10.5</td><td>8.8</td><td>7.5</td><td>15.9</td><td>34.0</td><td>23.5</td><td>27.8</td></tr><tr><td>4D-Occ* Khurana et al. [2023]</td><td>35.8</td><td>27.5</td><td>22.4</td><td>19.0</td><td>16.5</td><td>14.2</td><td>22.6</td><td>39.1</td><td>31.8</td><td>35.1</td></tr><tr><td>OccWorld Zheng et al. [2024]</td><td>39.7</td><td>34.2</td><td>27.9</td><td>26.4</td><td>23.6</td><td>21.6</td><td>28.9</td><td>50.5</td><td>40.2</td><td>44.8</td></tr><tr><td>ViDAR* Yang et al. [2024]</td><td>30.5</td><td>24.2</td><td>20.5</td><td>17.8</td><td>15.9</td><td>14.3</td><td>20.5</td><td>39.5</td><td>29.8</td><td>34.1</td></tr><tr><td>FreeOcc-D (ours)</td><td>40.8</td><td>34.3</td><td>29.1</td><td>25.1</td><td>21.8</td><td>19.2</td><td>28.4</td><td>41.3</td><td>49.4</td><td>44.9</td></tr><tr><td>FreeOcc-G (ours)</td><td>41.9</td><td>35.2</td><td>30.2</td><td>26.9</td><td>24.5</td><td>23.1</td><td>30.3</td><td>52.5</td><td>42.5</td><td>47.4</td></tr><tr><td>FreeOcc-E (ours)</td><td>43.6</td><td>36.3</td><td>32.2</td><td>28.9</td><td>26.5</td><td>24.9</td><td>32.1</td><td>54.6</td><td>45.5</td><td>49.6</td></tr></table>

<sup>∗</sup>Indicates the models whose output point clouds we voxelize for evaluation.

Occupancy forecasting (Table 2). On the voxel-based track, FreeOcc-E achieves the highest average mIoU at 32.1%, improving over OccWorld by +3.2 points and over FreeOcc-G by +1.8 points. The ensemble is best at every horizon, reaching 24.9% mIoU at τ=6. FreeOcc-G consistently improves over FreeOcc-D, showing the benefit of continuous generation. The additional gain from FreeOcc-E indicates that uncertainty-aware blending balances the decoder’s recall with the ODE’s sharper long-horizon structure. Static Copy and 4D-Occ degrade much faster, and ViDAR remains limited by the point-to-voxel conversion used for dense occupancy evaluation.

## 4.3 Efficiency Comparison

Table 3 shows that FreeOcc keeps a strong accuracy–speed trade-off. FreeOcc-D runs at 45.4 FPS, FreeOcc-E sustains 27.0 FPS while improving accuracy over all baselines, and FreeOcc-G remains compact but slower because of iterative ODE integration. Detailed latency analysis under adverse conditions is deferred to Section H.

Table 3: Efficiency comparison on nuScenes. We evaluate the parameter count and inference speed of our three variants: D (Decoder-only), G (Generation-only), and E (Ensemble) against representative baselines. All measurements are conducted on a single NVIDIA RTX 4090.
<table><tr><td>Method</td><td>#Params</td><td>Latency (ms)</td><td>FPS</td></tr><tr><td>4D-Occ Khurana et al. [2023]</td><td>8.59M</td><td>73</td><td>13.7</td></tr><tr><td>OccWorld Zheng et al. [2024]</td><td>72.4M</td><td>56</td><td>17.8</td></tr><tr><td>ViDAR Yang et al. [2024]</td><td>69.1M</td><td>189</td><td>5.3</td></tr><tr><td>FreeOcc-D (ours)</td><td>20.7M</td><td>22</td><td>45.4</td></tr><tr><td>FreeOcc-G (ours)</td><td>7.5M</td><td>168</td><td>6.0</td></tr><tr><td>FreeOcc-E (ours)</td><td>28.3M</td><td>37</td><td>27.0</td></tr></table>

## 4.4 Zero-Shot Cross-Domain Generalisation

Table 4 evaluates cross-domain generalisation by deploying nuScenes-trained models directly on SemanticKITTI with no fine-tuning. Despite the substantial domain gap (different LiDAR beam count, sensor mounting height, and geographic driving context), FreeOcc-E remains clearly ahead of the baselines, outperforming OccWorld by +5.8 points at short term and +5.9 points at long term. The gap is larger at long horizons, where FreeOcc-E reaches 18.2% mIoU versus 12.3% for OccWorld and 8.9% for 4D-Occ. FreeOcc-E also improves over FreeOcc-G, again indicating that the ensemble is more robust than either branch alone under domain shift. Notably, while FreeOcc-D’s aggressive generation yields higher recall at long horizons (33.2% compared to 30.8% for FreeOcc-E), it suffers from reduced precision. Additional SemanticKITTI visualizations are provided in Section F.

Table 4: Zero-shot transfer from nuScenes to SemanticKITTI (no fine-tuning). We report metrics across short-term and long-term horizons to assess structural preservation under distribution shift.
<table><tr><td rowspan="2">Method</td><td colspan="3">Short-term (τ ≤ 3)</td><td colspan="3">Long-term (τ &gt; 3)</td></tr><tr><td>mIoU</td><td>Precision</td><td>Recall</td><td>mIoU</td><td>Precision</td><td>Recall</td></tr><tr><td>Static Copy</td><td>14.5</td><td>30.2</td><td>22.1</td><td>7.5</td><td>22.5</td><td>13.0</td></tr><tr><td>4D-Occ Khurana et al. [2023]</td><td>15.2</td><td>31.8</td><td>24.4</td><td>8.9</td><td>24.1</td><td>15.2</td></tr><tr><td>OccWorld Zheng et al. [2024]</td><td>18.5</td><td>38.1</td><td>30.1</td><td>12.3</td><td>31.5</td><td>22.1</td></tr><tr><td>ViDAR Yang et al. [2024]</td><td>6.5</td><td>12.4</td><td>8.3</td><td>2.3</td><td>1.5</td><td>2.2</td></tr><tr><td>FreeOcc-D (ours)</td><td>22.0</td><td>42.5</td><td>39.8</td><td>15.5</td><td>34.1</td><td>33.2</td></tr><tr><td>FreeOcc-G (ours)</td><td>23.5</td><td>44.0</td><td>36.5</td><td>17.2</td><td>38.5</td><td>29.8</td></tr><tr><td>FreeOcc-E (ours)</td><td>24.3</td><td>45.0</td><td>37.5</td><td>18.2</td><td>39.5</td><td>30.8</td></tr></table>

![](images/348986c3e12ef7f99af1f2e162be8012b5784355d5946d04d95f5bc86d6dc461.jpg)  
Figure 3: 4D occupancy forecasting on nuScenes(sample 1720) in BEV and 3D views across six horizons. FreeOcc-E is compared against OccWorld and ViDAR; more qualitative examples are given in Section D.

## 4.5 Robustness and Ablation Summary

Under timestamp jitter and 30% history dropout, FreeOcc-E maintains a 29.8% mIoU, compared with 19.0% for OccWorld, 14.2% for 4D-Occ, and 12.8% for ViDAR. In ablations, the largest single drop comes from removing ego-motion warping (32.1→25.8 mIoU), followed by removing flow matching (32.1%→28.4% mIoU). FreeOcc-E also remains stronger than FreeOcc-G, confirming that uncertainty-aware blending adds measurable gains beyond the ODE alone. Full quantitative results are reported in Section C.4.

## 4.6 Qualitative Results

Figure 3 shows the qualitative trend behind the tables. FreeOcc-E preserves cleaner occupancy boundaries and more stable long-horizon structure than OccWorld and ViDAR, especially in dynamic and far-field regions. Additional visualizations, uncertainty maps, and robustness examples can be found in Section D–Section G.

## 5 Conclusion

We introduce FreeOcc, a continuous-time world model for 4D occupancy forecasting. By explicitly decoupling ego-motion via SE(2) warping, we isolate residual scene dynamics. This formulation enables attention-free temporal fusion and physics-constrained continuous flow matching. To reduce inference latency, an uncertainty-aware routing mechanism processes well-observed geometry with a single-pass decoder, reserving ODE integration strictly for ambiguous regions. Operating with roughly 40% of the parameters of comparable models, FreeOcc achieves 32.1% mIoU on nuScenes at 27 FPS. Furthermore, the model demonstrates zero-shot transfer to SemanticKITTI and retains 92.8% of its mIoU under simulated sensor degradation. These results indicate that explicit geometric decoupling and physical priors provide an effective, robust foundation for autonomous driving perception.

Limitations and future directions are discussed in Section B.

## Acknowledgements

This work was partially funded by the National Research Council Canada (NRC) and the Natural Sciences and Engineering Research Council (NSERC) of Canada.

## References

Ben Agro, Quinlan Sykora, Sergio Casas, Thomas Gilles, and Raquel Urtasun. Uno: Unsupervised occupancy fields for perception and forecasting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14487–14496, 2024.

Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jürgen Gall. SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 9297–9307, 2019.

Maxim Berman, Amal Rannen Triki, and Matthew B Blaschko. The lovász-softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 4413–4421, 2018.

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

Kyunghyun Cho, Bart Van Merriënboer, Çaglar Gulçehre, Dzmitry Bahdanau, Fethi Bougares, Holger ˘ Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder–decoder for statistical machine translation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1724–1734, 2014.

Jiayuan Du, Yiming Zhao, Zhenglong Guo, Yong Pan, Wenbo Hou, Zhihui Hao, Kun Zhan, and Qijun Chen. Sparseworld-tc: Trajectory-conditioned sparse occupancy world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7425–7434, 2026.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu. Spatial transformer networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 28, 2015.

Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

Tarasha Khurana, Peiyun Hu, David Held, and Deva Ramanan. Point cloud forecasting as a proxy for 4d occupancy forecasting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1116–1124, 2023.

Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697–12705, 2019.

Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017a.

Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, 2017b.

Yaron Lipman, Ricky T.Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2023.

Tianran Liu, Shengwen Zhao, and Nicholas Rhinehart. Towards foundational lidar world models with efficient latent flow matching. arXiv preprint arXiv:2506.23434, 2025.

Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In Proceedings of the International Conference on 3D Vision (3DV), pages 565–571, 2016.

Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, pages 194–210. Springer, 2020.

Yining Shi, Kun Jiang, Ke Wang, Jiusi Li, Yunlong Wang, Mengmeng Yang, and Diange Yang. Streamingflow: Streaming occupancy forecasting with asynchronous multi-modal data streams via neural ordinary differential equation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14833–14842, 2024.

Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Haoran Xu, Peixi Peng, Guang Tan, Yiqian Chang, Yisen Zhao, and Yonghong Tian. Delta-triplane transformers as occupancy world models. arXiv preprint arXiv:2503.07338, 2025.

Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Zichen Yu, Quanli Liu, Wei Wang, Liyong Zhang, and Xiaoguang Zhao. Superocc: Toward cohesive temporal modeling for superquadric-based occupancy prediction. arXiv preprint arXiv:2601.15644, 2026.

Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Copilot4d: Learning unsupervised world models for autonomous driving via discrete diffusion. In The Twelfth International Conference on Learning Representations, 2024.

Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pages 55–72. Springer, 2024.

## A Additional Method Details

## A.1 Temporal augmentation and decoder details

Range encoding. For BEV cell $( h , w )$ with radial distance $r _ { h w } = \sqrt { x _ { h } ^ { 2 } + y _ { w } ^ { 2 } }$ , we use

$$
\rho _ { h w } = W _ { 2 } \mathrm { R e L U } \big ( W _ { 1 } [ \sin ( 2 ^ { 0 } r ) , \cos ( 2 ^ { 0 } r ) , \dots , \sin ( 2 ^ { B _ { r } - 1 } r ) , \cos ( 2 ^ { B _ { r } - 1 } r ) ] \big ) ,\tag{14}
$$

and add it to the pillar pseudo-image as $\bar { F } _ { i } \gets \bar { F } _ { i } + \rho .$

Timestamp jitter and sensor dropout. To train the temporal aggregator on irregular input streams, each history sweep $P _ { i } \left( i < t \right)$ is assigned a perturbed offset

$$
\Delta t _ { i } ^ { \prime } = \Delta t _ { i } + \eta _ { i } , \qquad \eta _ { i } \sim \mathcal { U } ( - \delta _ { t } , \delta _ { t } ) ,\tag{15}
$$

with $\delta _ { t } = 0 . 1 \mathrm { s } .$ The current frame is never perturbed. In parallel, every non-current sweep is dropped independently with probability $p _ { \mathrm { d r o p } } = 0 . 3$ , while forcing at least one earlier sweep to remain. Jitter trains continuous interpolation over residual timing noise, whereas dropout trains recovery from incomplete histories.

The timestamp embedding itself uses exponentially spaced sinusoidal frequencies:

$$
\phi ( \Delta t _ { i } ) = \big [ \mathrm { s i n } ( \omega _ { 1 } \Delta t _ { i } ) , ~ \mathrm { c o s } ( \omega _ { 1 } \Delta t _ { i } ) , ~ . ~ . ~ . , ~ \mathrm { s i n } ( \omega _ { C / 2 } \Delta t _ { i } ) , ~ \mathrm { c o s } ( \omega _ { C / 2 } \Delta t _ { i } ) \big ] ,\tag{16}
$$

where $\omega _ { j } = \exp ( - j \ln \Lambda / ( C / 2 ) )$

Decoder architecture. Each BEV prior $C _ { \tau }$ is refined by two $3 \times 3 ~ \mathrm { B E V }$ convolutions, lifted to $C \cdot Z$ channels by $\bf { a } \perp \times 1$ convolution, reshaped to $\mathbb { R } ^ { C \times H \times \check { W } \times Z }$ , augmented with a height positional encoding $z _ { \mathrm { p o s } } ,$ , and processed by two $3 \times 3 \times 3$ volume convolutions. A future-step embedding $e _ { \tau }$ is added before lifting. The occupancy head maps $C  C / 2  C / 4  1$ , while the uncertainty head predicts $\hat { s } _ { \tau }$ with a two-layer $\mathrm { M } \bar { \mathrm { L P } }$ whose bias is initialised $\mathrm { { [ 0 - 0 . 5 } }$

Velocity network details. The velocity network $v _ { \theta }$ is a 3D U-Net with channel widths $\{ c , 2 c , 4 c \}$ a bottleneck, and symmetric skip connections. Flow time s is injected with a sinusoidal embedding, and each future step τ has a learned embedding. For diversity, a fraction $\alpha = 0 . 2$ of initial Gaussian samples is replaced by slightly noised decoder outputs.

Uncertainty normalisation for cooperative flow matching. The normalisation bounds $\mu _ { \mathrm { l o w } }$ and $\mu _ { \mathrm { h i g h } }$ in Eq. (7) are maintained as exponential moving averages of the 5th and 95th percentiles of detached $\hat { s } _ { v }$ computed over training batches. This ensures that the uncertainty weight w remains stable while effectively spending generative capacity strictly in regions where the discriminative path is least reliable.

## A.2 Full loss definitions

Focal BCE $\mathcal { L } _ { \mathrm { o c c } }$ Lin et al. [2017b]. We use focal BCE with $\alpha = 0 . 7 5$ to upweight the minority occupied class and suppress easy free-space gradients.

Lovász hinge $\mathcal { L } _ { \mathrm { L o v } }$ Berman et al. [2018]. This term is a differentiable surrogate for IoU. Its weight is warmed up as

$$
\lambda _ { \mathrm { l o v } } ( n ) = \bar { \lambda } _ { \mathrm { l o v } } \cdot \mathrm { m i n } \bigg ( 1 , \mathrm { m a x } \bigg ( 0 , \frac { n - n _ { \mathrm { s t a r t } } } { n _ { \mathrm { r a m p } } } \bigg ) \bigg ) .\tag{17}
$$

Dice $\mathcal { L } _ { \mathrm { D i c e } }$ Milletari et al. [2016]. Dice complements BCE and Lovász with a smooth overlap objective.

Sparsity $\mathcal { L } _ { \mathrm { s p } }$ . We penalise deviation from a target occupied ratio $\bar { r } = 0 . 0 5$ using $| \bar { \sigma } ( \hat { y } ) - \bar { r } |$ , with a weight that grows linearly with horizon as $( 1 + \bar { 0 } . 5 \tau )$

Temporal consistency $\mathcal { L } _ { \mathrm { t c } }$ . We match predicted and ground-truth temporal derivatives:

$$
\mathcal { L } _ { \mathrm { t c } } = \frac { 1 } { N - 1 } \sum _ { \tau } \left. \left( \sigma ( \hat { y } _ { \tau + 1 } ) - \sigma ( \hat { y } _ { \tau } ) \right) - \left( y _ { \tau + 1 } - y _ { \tau } \right) \right. _ { 2 } ^ { 2 } .\tag{18}
$$

Ray-traced free-space $\mathcal { L } _ { \mathrm { r a y } }$ . LiDAR rays define observed free space before the first hit. With traversed voxel set ${ \mathcal F }$ , we use

$$
\mathcal { L } _ { \mathrm { r a y } } = \frac { 1 } { | \mathcal { F } | } \sum _ { ( i , j , k ) \in \mathcal { F } } \sigma ( \hat { y } _ { i j k } ) .\tag{19}
$$

Voxels behind the first surface hit remain unconstrained.

Conservation ${ \mathcal { L } } _ { \mathrm { c o n s } } .$ Under a closed-world approximation, total occupied mass should vary smoothly:

$$
\mathcal { L } _ { \mathrm { c o n s } } = \frac { 1 } { N - 1 } \sum _ { \tau } \bigl ( \Delta \hat { M } _ { \tau } - \Delta M _ { \tau } ^ { \mathrm { g t } } \bigr ) ^ { 2 } ,\tag{20}
$$

where $\Delta \hat { M } _ { \ i }$ <sub>τ</sub> is measured over the interior region Ω.

Range-weighted loss $\mathcal { L } _ { \mathrm { r w } }$ . We modulate supervision by sensor distance using

$$
w ( r ) = w _ { \mathrm { n e a r } } ( 1 - r / r _ { \mathrm { m a x } } ) + w _ { \mathrm { f a r } } r / r _ { \mathrm { m a x } } ,\tag{21}
$$

with $w _ { \mathrm { n e a r } } = 1 . 5$ and $w _ { \mathrm { f a r } } = 0 . 5$ .

Velocity smoothness $\mathcal { L } _ { \mathrm { v e l } }$ . Eq. (6) imposes piecewise-smooth motion on the ego-decoupled flow field.

Uncertainty calibration ${ \mathcal { L } } _ { \mathrm { u n c } } .$ . Following the heteroscedastic classification approach, we supervise the uncertainty head using:

$$
\mathcal { L } _ { \mathrm { u n c } } = \frac { 1 } { | \Omega | } \sum _ { i j k } \left[ \exp ( - \hat { s } _ { i j k } ) \ell _ { \mathrm { B C E } } ( \hat { y } _ { i j k } , y _ { i j k } ) + \beta \hat { s } _ { i j k } \right] ,\tag{22}
$$

where $\ell _ { \mathrm { B C E } }$ denotes the binary cross-entropy loss applied to the predicted occupancy logits $\hat { y } _ { i j k }$ . We fix $\beta = 0 . 5$ as a penalty to prevent the model from trivializing the objective by predicting infinite variance.

Hyperparameter configurations. For the composite objective defined in Eq. 12, the base scalar weights are empirically set as follows: $\lambda _ { \mathrm { o c c } } = 2 . 0 , \lambda _ { \mathrm { l o v } } = 1 . 0 , \lambda _ { \mathrm { d i c e } } = 1 . 0 , \bar { \lambda _ { \mathrm { s p } } } = 0 . 5 , \lambda _ { \mathrm { t c } } = 1 . 0$ $\lambda _ { \mathrm { { r a y } } } = 1 . 0 , \lambda _ { \mathrm { { c o n s } } } = 0 . 1 , \lambda _ { \mathrm { { v e l } } } = 0 . 5 , \lambda _ { \mathrm { { r w } } } = 1 . 0$ , and $\lambda _ { \mathrm { u n c } } = 1 . 0$ . For the range encoding (Eq. 14), the frequency band limit is $B _ { r } = 6$ , and for the timestamp embedding (Eq. 16), the base period $\Lambda = 1 0 0 0 0$ . The scheduling parameters for the Lovász hinge warmup are $n _ { \mathrm { s t a r t } } ~ = ~ 1 0 0 0$ and $n _ { \mathrm { r a m p } } = 2 0 0 0$ iterations.

## B Limitations and Future Directions

FreeOcc does not model semantic classes; extending the framework to class-conditional flow matching remains future work. The ODE branch still adds non-trivial inference cost, which motivates future work on adaptive solvers or distillation.

Furthermore, the explicit geometric decoupling relies on a planar SE(2) ego-motion warp. This assumes flat-world dynamics, which may introduce minor spatial misalignments under severe pitch, roll, or steep inclines. However, enforcing SE(2) decoupling rather than a full 3D volumetric SE(3) warp drastically reduces the computational overhead of grid sampling. In practice, our evaluations demonstrate that the residual ConvGRU effectively absorbs minor Z-axis translations and pitch deviations, confirming that SE(2) alignment strikes the optimal balance between geometric accuracy and real-time inference latency for urban driving scenarios. Furthermore, the model treats the motion planner’s trajectory as a known input; however, experiments show that a self-predicted trajectory extrapolated from the vehicle’s recent speed and yaw rate, using no future information, still achieves 31.2% mIoU (vs. 32.1% with the recorded route), indicating that the method does not critically depend on an accurate external plan. Sensitivity to trajectory noise is analysed in Section I. Mitigating vulnerability to local odometry or planning drift remains an area for future refinement.

Finally, the point-based track retains a quantisation floor induced by the voxel resolution $( \delta \mathrm { = } 0 . 4 \mathrm { m } )$ Since δ is a preprocessing choice rather than an architectural constraint, finer grids or a learned sub-voxel refinement head could reduce this error floor at additional compute cost.

## C Experimental Details

## C.1 Dataset and transfer details

We train on the standard nuScenes split (700/150/150 train/val/test) with 32-beam LiDAR, 20 Hz sweeps, and 2 Hz keyframes. For zero-shot transfer, we evaluate on SemanticKITTI. Before evaluation, SemanticKITTI point clouds are range-clipped to [−40, 40] m in the horizontal plane, shifted by 0.11 m along Z, and beam-downsampled to reduce the 64-to-32 line density mismatch.

## C.2 Metric Alignment with Baselines

In the point-based evaluation track (Table 1), we report Near Field Chamfer Distance (NFCD) to align with UnO Agro et al. [2024].

Copilot4D Zhang et al. [2024] does not use the term “NFCD”, but it reports Chamfer Distance within a fixed Region of Interest (ROI). This evaluation protocol is equivalent to the NFCD computation used by UnO. We therefore report Copilot4D’s ROI-restricted Chamfer Distance in the NFCD column.

## C.3 Discussion: Sensor-Grounded vs. Completion-Based Evaluation

We do not evaluate on Occ3D Tian et al. [2023] because it measures a different task. Occ3D targets occupancy completion: reconstructing the full 3D scene, including regions never observed by LiDAR, such as building interiors and heavily occluded surfaces. FreeOcc instead addresses sensor-grounded occupancyforecasting: predicting the 3D structure that the sensor will observe at future timesteps.

This distinction matters for planning. The relevant target is future sensor-observable geometry, not the interior of occluded structures. Accordingly, our supervision is derived from future LiDAR returns rather than dense completion labels. Training against completion labels would force the model to infer geometry behind walls and inside objects, which is underdetermined and unnecessary for the forecasting setting considered here.

The evaluation objectives also differ. Occ3D rewards filling unobserved regions, whereas our protocol measures binary occupied/free IoU against future scans and therefore rewards accurate prediction of observable structure. These objectives need not align: a model tuned for completion may hallucinate geometry that improves completion scores but is undesirable for downstream planning.

## C.4 Additional experimental results

## C.4.1 Robustness to sensor temporal imperfections

Table 5: Robustness to sensor temporal imperfections on nuScenes. The imperfect setting adds severe timestamp jitter and 30% history-frame dropout. All metrics are voxel-based.
<table><tr><td>Method</td><td>Condition mIoU ↑</td><td></td><td>Prec. ↑</td><td>Rec. ↑</td><td>F1 ↑</td></tr><tr><td rowspan="2">4D-Occ Khurana et al. [2023]</td><td>Ideal</td><td>22.6</td><td>39.1</td><td>31.8</td><td>35.1</td></tr><tr><td>Imperfect</td><td>14.2</td><td>30.5</td><td>21.0</td><td>24.9</td></tr><tr><td rowspan="2">Occ World Zheng et al. [2024]</td><td>Ideal</td><td>28.9</td><td>50.5</td><td>40.2</td><td>44.8</td></tr><tr><td>Imperfect</td><td>19.0</td><td>38.5</td><td>28.5</td><td>32.8</td></tr><tr><td rowspan="2">ViDAR Yang et al. [2024]</td><td>Ideal</td><td>20.5</td><td>39.5</td><td>29.8</td><td>34.1</td></tr><tr><td>Imperfect</td><td>12.8</td><td>28.2</td><td>19.5</td><td>23.1</td></tr><tr><td rowspan="2">FreeOcc-D (ours)</td><td>Ideal</td><td>28.4</td><td>41.3</td><td>49.4</td><td>44.9</td></tr><tr><td>Imperfect</td><td>26.0</td><td>37.5</td><td>46.5</td><td>41.3</td></tr><tr><td rowspan="2">FreeOcc-G (ours)</td><td>Ideal</td><td>30.3</td><td>52.5</td><td>42.5</td><td>47.4</td></tr><tr><td>Imperfect</td><td>28.1</td><td>49.8</td><td>39.0</td><td>43.7</td></tr><tr><td rowspan="2">FreeOcc-E (ours)</td><td>Ideal</td><td>32.1</td><td>54.6</td><td>45.5</td><td>49.6</td></tr><tr><td>Imperfect</td><td>29.8</td><td>51.5</td><td>42.0</td><td>46.3</td></tr></table>

FreeOcc retains the strongest performance under temporal noise. FreeOcc-E keeps 92.8% of its ideal mIoU (29.8/32.1), while OccWorld retains 65.7%, 4D-Occ 62.8%, and ViDAR 62.4%. This is

consistent with the timestamp-jitter and sensor-dropout training described in Section 3.3. Detailed results are shown in Table 6.

## C.4.2 Ablation studies

Table 6: Ablation study on nuScenes validation. Metrics are voxel-based. The results demonstrate that explicit geometric decoupling and physics-aware regularization are the primary drivers of forecasting quality.
<table><tr><td>Variant</td><td>mIoU ↑</td><td>Precision ↑</td><td>F1↑</td></tr><tr><td>FreeOcc-E (full)</td><td>32.1</td><td>54.6</td><td>49.6</td></tr><tr><td>Generation Path Ablations</td><td></td><td></td><td></td></tr><tr><td>w/o Flow Matching (FreeOcc-D)</td><td>28.4</td><td>41.3</td><td>44.9</td></tr><tr><td>w/o Decoder (FreeOcc-G)</td><td>30.3</td><td>52.5</td><td>47.4</td></tr><tr><td>Max blending</td><td>31.2</td><td>53.4</td><td>48.4</td></tr><tr><td>Mean blending</td><td>30.5</td><td>52.6</td><td>47.5</td></tr><tr><td>Architecture Ablations</td><td></td><td></td><td></td></tr><tr><td>w/o Ego-Motion Warping</td><td>25.8</td><td>46.2</td><td>40.5</td></tr><tr><td>w/o Residual Connection</td><td>30.2</td><td>52.1</td><td>47.2</td></tr><tr><td>w/o Delay-Aware Confidence Gate</td><td>31.1</td><td>53.3</td><td>48.3</td></tr><tr><td>w/o Sensor Dropout</td><td>31.4</td><td>53.8</td><td>48.8</td></tr><tr><td>w/o Timestamp Jitter  $( \delta _ { t } = 0 )$ </td><td>31.6</td><td>54.0</td><td>49.0</td></tr><tr><td>Fixed-stride temporal encoding</td><td>30.8</td><td>52.9</td><td>48.0</td></tr><tr><td>Physics-Aware Regularization Ablations</td><td></td><td></td><td></td></tr><tr><td>w/o Structural Fidelity (Group)</td><td>28.1</td><td>49.5</td><td>44.1</td></tr><tr><td>w/o Physical Priors (Group)</td><td>28.7</td><td>50.8</td><td>44.8</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { v e l } }$  (Smoothness)</td><td>30.4</td><td>52.3</td><td>47.4</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r a y } }$  (Free-space)</td><td>31.1</td><td>53.2</td><td>48.3</td></tr><tr><td>w/o Lcons (Mass conservation)</td><td>31.4</td><td>53.8</td><td>48.8</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { u n c } }$  (Calibration)</td><td>31.5</td><td>53.9</td><td>48.9</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r w } }$  (Range weighting)</td><td>31.6</td><td>54.1</td><td>49.0</td></tr><tr><td>Uncertainty Ablations</td><td></td><td></td><td></td></tr><tr><td>w/o cooperative unc.-guided FM</td><td>30.6</td><td>52.7</td><td>47.7</td></tr><tr><td>w/o uncertainty-aware blending</td><td>31.2</td><td>53.4</td><td>48.4</td></tr></table>

The ablations in Table 6 support our core design philosophy. Removing Ego-Motion Warping causes the single largest drop (32.1 → 25.8 mIoU), confirming that explicit geometric decoupling is fundamental to forecasting accuracy. Within our loss system, the Structural Fidelity group is essential for grounding the discriminative path, while the Physical Priors (specifically velocity smoothness $\mathcal { L } _ { \mathrm { v e l } } )$ ensure that the generative ODE produces kinematically plausible trajectories. Notably, removing either entire regularization group leads to a performance drop significantly larger than removing any individual term, proving that our physics-aware objectives act as a cohesive system rather than a collection of independent penalties.

## D Additional Qualitative Results

This appendix provides additional visualizations supporting the qualitative analysis in Section 4.6.  
![](images/596fa4173f0692e9263af31f2e608c42c4b1fd30d0a82a9fb19c542d9071ad84.jpg)  
Figure 4: BEV visualization of 4D occupancy forecasting (additional scene, sample 0970). Columns span τ=1 to τ=6, comparing FreeOcc against OccWorld and ViDAR on a dense urban scene with multiple interacting vehicles. FreeOcc-E preserves scene layout and object extent more faithfully over the full 3.0 s horizon, whereas OccWorld and ViDAR drift more noticeably after τ=3.

![](images/037fdc58f6615b79cc3093c94efdb52f20ccdee030329d92a946076c8db2ff6b.jpg)  
Figure 5: Temporal evolution of GT vs. FreeOcc-E (nuScenes sample 1730). Top row: groundtruth occupancy $\mathfrak { x } \tau { = } 2 ( 1 . 0 \mathrm { s } ) , \tau { = } 4 ( 2 . 0 \mathrm { s } ) .$ , and $\tau { = } 6 \left( 3 . 0 \mathrm { s } \right)$ , each shown as BEV (left) and 3D voxel rendering (right). Bottom row: corresponding FreeOcc-E predictions. Agreement is strongest at short horizons. $\mathbf { B y } \ \tau \mathrm { = } 6 .$ , the main discrepancies appear in the far field and around dynamic actors, while large static structures remain well preserved.

![](images/c0d617adce704fc07c45dad20b23db118b0cc454f39d4ab0fd64c988c781f10e.jpg)  
Figure 6: Long-horizon comparison at τ=6 (3.0 s, nuScenes sample 3380). Ground Truth, OccWorld, FreeOcc-D, and FreeOcc-E are shown in paired BEV and 3D views. At 3.0 s, OccWorld exhibits boundary blur and noise around moving objects. FreeOcc-D slightly over-predicts occupancy, whereas FreeOcc-E preserves sharper boundaries and cleaner free space.

![](images/a7ae4b18ffcb5f5d1e8037e0b765927de9c414044cee9880534a679d83054788.jpg)  
Figure 7: 3D voxel-level uncertainty across prediction horizons $( \mathbf { s c e n e - 0 2 7 4 } , \tau { = } 1 \ \mathbf { t o } \ \tau { = } 6 )$ . Each panel renders occupied voxels coloured by the decoder’s predicted aleatoric uncertainty $\hat { s } _ { v } ;$ cool tones indicate low uncertainty and warm tones indicate high. At short horizons, uncertainty is concentrated near dynamic actors and occupancy boundaries. As the horizon increases, uncertainty expands into the far field and intensifies around moving objects. By $\tau { = } 6 \left( 3 . 0 \mathrm { s } \right)$ , most remaining structures are assigned substantially higher uncertainty. This behaviour is consistent with the ensemble design in Section 3.7, which assigns more weight to the ODE path where decoder confidence weakens.

![](images/7ca655e727924091d40bdbb5d2c8f75ef0008104ab9e894c313b32ed3bf51fcf.jpg)  
Figure 8: Noise suppression analysis (nuScenes sample 1290). Paired BEV and 3D views highlighting hallucinated occupancy at long prediction horizons between FreeOcc-D and FreeOcc-E. FreeOcc-E clears vacated space more effectively while preserving plausible future object positions.

## E Uncertainty Visualisation

This appendix visualises the per-voxel aleatoric uncertainty $\hat { s } _ { v }$ predicted by the decoder head, complementing the quantitative ablation of the uncertainty-aware blending mechanism (Section 3.7 and Table 6).

BEV uncertainty evolution. Figure 9 shows BEV projections of predicted occupancy coloured by $\hat { s } _ { v }$ for scene-0220 across all six prediction horizons. Three trends are consistent across the sequence:

• Near-field confidence. $\mathrm { A t } \ \tau { = } 1 \ ( 0 . 5 \mathrm { s } )$ , uncertainty is concentrated on dynamic objects and occupancy boundaries, while large static structures remain comparatively confident.

• Horizon-dependent growth. As τ increases from 1 to 6 (0.5 s to 3.0 s), high-uncertainty regions expand and intensify. By τ=4, a turning vehicle near the centre of scene-0220 dominates the uncertainty map, reflecting the increasing ambiguity of its future trajectory.

• Far-field diffusion. Sparse peripheral voxels become uncertain earlier than dense near-field regions, consistent with the weaker LiDAR support available at long range.

These patterns are consistent with the ensemble rule in Section 3.7: short-horizon predictions rely primarily on the decoder, whereas long-horizon and dynamic regions receive greater weight from the ODE branch.

![](images/0c13d14f11a86103ae8d658ed8570c05ca0293024481f731e5d06e76a4792d05.jpg)  
Figure 9: BEV uncertainty evolution (nuScenes scene-0220, $\tau { = } 1$ to $\tau { = } 6 )$ . Each panel shows the BEV occupancy coloured by the predicted aleatoric uncertainty $\hat { s } _ { v }$ at horizon τ. Cool colours (cyan/green) indicate low uncertainty; warm colours (orange/red) indicate high uncertainty. Uncertainty increases with horizon and concentrates on dynamic objects and far-field voxels. Similar behaviour appears in Figure 7.

3D voxel-level analysis. Figure 7 renders the same uncertainty signal in 3D for scene-0274, which makes the vertical distribution easier to inspect. At short horizons, the predicted structure is vertically coherent and densely populated; as τ increases, dropout and insertion artefacts become more visible, especially in thin or weakly observed structures. The uncertainty map tracks this degradation closely: voxels that remain at $\tau { = } 6$ are predominantly assigned high uncertainty, indicating that decoder confidence deteriorates in step with forecast difficulty. This behaviour is precisely what the ensemble mechanism exploits when assigning more weight to the ODE path at longer horizons.

## F Cross-Domain Qualitative Results on SemanticKITTI

The following figures visualise zero-shot transfer results on SemanticKITTI sequences, using models trained exclusively on nuScenes with no fine-tuning.

![](images/0d3b1e9a02398f1c120d66f08f135908799f4766d6c98d9942bbb990a9977fd3.jpg)  
Figure 10: Zero-shot SemanticKITTI predictions (sample 2070). BEV and 3D occupancy predictions from a nuScenes-trained model evaluated on SemanticKITTI, comparing FreeOcc-E against OccWorld and ViDAR. Despite domain differences (64-beam LiDAR, different driving context, no fine-tuning), FreeOcc-E preserves road-level structure and major obstacles. The denser 64-beam input provides richer geometric detail than nuScenes’s 32-beam data, and FreeOcc adapts without retraining, an advantage of operating on raw 3D geometry rather than appearance-dependent features.

![](images/dcb62129b86fa35657c6b07afc6fd61b70e7b73feff4c7c410780e5c16b6ec77.jpg)  
Figure 11: Additional zero-shot SemanticKITTI scene (sample 1080) and noise suppression analysis. Top: 3D occupancy predictions at the maximum forecasting horizon $( \tau ~ = ~ 6 , 3 . 0 \mathrm { s } )$ comparing our model variants (FreeOcc-D, -G, -E) against Ground Truth. Bottom: BEV temporal evolution at τ = 3 and τ = 6. Road geometry, vegetation boundaries, and building walls are consistently captured, reinforcing the cross-domain robustness reported in Table 4.

## G Robustness Qualitative Results

These visualizations illustrate the effect of degraded sensor conditions (timestamp jitter + 30% frame dropout) on model predictions, corresponding to the quantitative analysis in Table 5.

![](images/7aa47067ef84c17c1afe52bf0a4991dad70067901d4c34a95585f5359b57488b.jpg)

Figure 12: Robustness on nuScenes (τ = 4, 2.0s, sample 2500). Occupancy predictions under ideal and imperfect sensor conditions. FreeOcc-E retains its predictions under severe temporal perturbations, while OccWorld and ViDAR break into fragments and hallucinated clusters when temporal input is degraded.  
![](images/b5dcbb8bbc7e5e566870b71e9a4a6004461db0b4b2a1aa603ffee7659fb89205.jpg)  
Figure 13: Cross-domain robustness on SemanticKITTI $( \tau = 4 , 2 . 0 \mathbf { s } ,$ sample 2090). Zero-shot predictions under ideal (top two rows) and imperfect (bottom two rows) sensor conditions. The cross-domain robustness gap compounds both the sensor-domain shift (nuScenes→SemanticKITTI) and the temporal perturbation, yet FreeOcc-E retains recognisable road and object geometry in both cases, while OccWorld and ViDAR suffer severe structural collapse.

![](images/7d4259ff39f0a0675fd97d1ffdcfd286e347b94b026b5a284325a490d655f5e5.jpg)  
Figure 14: Robustness on nuScenes at τ=6 (3.0 s, sample 1600). Long-horizon predictions under ideal (top) and imperfect (bottom) sensor conditions. At the maximum forecasting horizon, OccWorld and ViDAR dissolve into discontiguous fragments, whereas FreeOcc-E preserves the road layout and principal object silhouettes. The gap between ideal and imperfect rows is noticeably smaller for FreeOcc-E than for OccWorld or ViDAR, consistent with the quantitative retention rate in Table 5.

![](images/4a223bbddfda26fb78c3b53c497977d3fac2b5d50d6b24d3a42603a8cc95766f.jpg)  
Figure 15: Cross-domain robustness on SemanticKITTI at τ=6 (3.0 s, sample 2090). The most challenging evaluation setting: simultaneous cross-domain transfer, long-horizon prediction, and degraded sensor input. FreeOcc-E still recovers the overall road layout and major object boundaries, whereas OccWorld and ViDAR suffer near-complete structural collapse.

## H Latency Under Adverse Conditions

A practical question is whether latency remains stable on visually and geometrically difficult scenes, such as rain, reflections, and dense intersections. Here we evaluate runtime directly on a challenging nuScenes rain sequence and report the observed spread.

Evaluation protocol. Figure 16 shows a representative keyframe from nuScenes scene-0441 (rain, cross-intersection, lane change), one of the most demanding sequences in the dataset. We measure per-frame latency over all 40 keyframes and report the mean and worst-case runtime for FreeOcc-D and FreeOcc-E in Table 7. OccWorld is included as an ideal-condition reference from our benchmark run.

Observed latency stability. FreeOcc-E maintains a mean latency of 37.3 ms (26.8 FPS), with worst-case latency both at 39.1 ms. This spread is small and remains within roughly 2 ms of the dry-weather average reported in the main paper. FreeOcc-D is similarly stable at 22.1 ms (45.2 FPS). Even in this adverse sequence, FreeOcc-E remains faster than our OccWorld reference measured under ideal conditions (56 ms, 17.8 FPS).

Interpretation. These results suggest that, in our implementation, adverse weather and dense traffic do not materially bottleneck runtime. In FreeOcc-E, the fast discriminative decoder first processes the scene and generates the uncertainty map, which then selectively routes only the highly ambiguous regions to the continuous ODE solver. The stability shown in Table 7 indicates that even in challenging conditions like rain, the decoder remains confident enough to resolve the vast majority of the environment (e.g., road layout and free space). Consequently, the additional computational overhead from routing more voxels to the ODE solver is tightly bounded, resulting in a worst-case latency penalty of only about 2 ms. We report these empirical statistics to demonstrate that our uncertainty-aware routing is robust and avoids unbounded computational spikes in unpredictable environments.

![](images/b1a231d73df106dc9f3e088cdc203301ed4d4d1d6e2f890f8dfb8f439a8eec4d.jpg)

![](images/b65cfbf8aeeb2c1a2b69b5870d4dcb509763ad7720edf7dd0874624b5d817a5f.jpg)  
Figure 16: Rain scene overview (nuScenes scene-0441). Left: front camera image showing wet road, reflections, and rain droplets. Right: LiDAR BEV coloured by height; the intersection geometry and parked vehicles are clearly visible despite rain-induced noise in the point cloud.

Table 7: Inference latency on rain sequence (nuScenes scene-0441, 40 keyframes). Both FreeOcc variants maintain stable latency indistinguishable from dry-weather conditions. FreeOcc-E’s worstcase is still 1.4× faster than our OccWorld run under ideal conditions.
<table><tr><td>Method</td><td>Mean (ms)</td><td>Worst (ms)</td><td>FPS (mean)</td></tr><tr><td>OccWorld (ideal, ref.)</td><td>56.0</td><td></td><td>17.8</td></tr><tr><td>FreeOcc-D (rain)</td><td>22.1</td><td>23.8</td><td>45.2</td></tr><tr><td>FreeOcc-E (rain)</td><td>37.3</td><td>39.1</td><td>26.8</td></tr></table>

## I Trajectory Robustness Analysis

To verify that FreeOcc’s advantage does not arise from privileged access to the exact future route, we conduct two analyses.

Sensitivity to trajectory error. We inject Gaussian noise into the future ego-poses, reporting the accumulated L2 position error at 3 s.

Table 8: FreeOcc-E mIoU under injected future-trajectory error on nuScenes.
<table><tr><td>Traj. error @3 s (L2)</td><td>0.0m</td><td>0.3 m</td><td>0.5 m</td><td>1.0m</td><td>2.0m</td><td>4.0m</td></tr><tr><td>FreeOcc-E mIoU</td><td>32.1</td><td>31.8</td><td>31.5</td><td>31.1</td><td>30.2</td><td>28.6</td></tr></table>

FreeOcc-E degrades gracefully. Across the typical planner-error range (<2 m at 3 s), it stays at 30.2 mIoU or above, well above OccWorld’s 28.9.

Self-predicted trajectory. We replace the recorded route with a constant-velocity / constant-yawrate extrapolation from the recent history, using no future information.

Table 9: FreeOcc-E with a self-predicted trajectory (no future information).
<table><tr><td>Trajectory</td><td>mIoU</td><td>NFCD@3s</td></tr><tr><td>Recorded route (paper setting)</td><td>32.1</td><td>0.47</td></tr><tr><td>Self-predicted from history</td><td>31.2</td><td>0.51</td></tr><tr><td>OccWorld (no external traj.)</td><td>28.9</td><td></td></tr></table>

Even without access to the future route, FreeOcc-E reaches 31.2 mIoU, still above OccWorld.

## J Information-Matched Warp Baselines

To bound the contribution of the trajectory warp, we build two baselines on FreeOcc’s own temporally aggregated present state (not on OccWorld). Warp+Fuse warps this state to the future ego frame and refines it with a light fusion head. Warp+Fuse+FOV additionally masks the warped result to the future sensor field of view. Neither uses the flow-matching branch or the uncertainty ensemble.

Table 10: Information-matched warp baselines versus FreeOcc on nuScenes.
<table><tr><td>Method</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>NFCD@3s</td></tr><tr><td>Static Copy</td><td>15.9</td><td>34.0</td><td>23.5</td><td>27.8</td><td>1.11</td></tr><tr><td>Warp+Fuse</td><td>20.4</td><td>35.3</td><td>33.1</td><td>34.2</td><td>1.04</td></tr><tr><td>Warp+Fuse+FOV</td><td>20.8</td><td>36.5</td><td>33.4</td><td>34.9</td><td>0.98</td></tr><tr><td>FreeOcc-D</td><td>28.4</td><td>41.3</td><td>49.4</td><td>44.9</td><td>0.63</td></tr><tr><td>FreeOcc-E</td><td>32.1</td><td>54.6</td><td>45.5</td><td>49.6</td><td>0.47</td></tr></table>

Both warp baselines trail FreeOcc-E by at least 11.3 mIoU, attributing the gap to the flow-matching branch and the uncertainty ensemble rather than the trajectory warp.

## K Static vs. Dynamic Breakdown

We define a ground-truth occupied voxel as dynamic if it falls inside the BEV footprint of a nuScenes 3D box of a movable class (vehicle, human, cycle) moving faster than 0.5 m/s, and static otherwise. According to our statistics on the nuScenes dataset, dynamic voxels constitute only about 4% of occupied voxels.

FreeOcc and OccWorld are essentially tied on static voxels (37.5 vs 37.2), so the trajectory warp gives no meaningful static advantage. The entire FreeOcc gain is on dynamic voxels (+5.1), where the ego-warp offers no help.

Table 11: Static and dynamic mIoU on nuScenes.
<table><tr><td>Method</td><td>Static mIoU</td><td>Dynamic mIoU</td><td>Overall</td><td>Dyn. margin vs OccWorld</td></tr><tr><td>Warp+Fuse+FOV</td><td>26.8</td><td>4.2</td><td>20.8</td><td>-10.3</td></tr><tr><td>OccWorld</td><td>37.2</td><td>14.5</td><td>28.9</td><td></td></tr><tr><td>FreeOcc-E</td><td>37.5</td><td>19.6</td><td>32.1</td><td>+5.1</td></tr></table>

## L Comparison with SparseWorld-TC

SparseWorld-TC Du et al. [2026] is a camera-based method designed for Occ3D semantic completion. We run the released model and evaluate it at the {1, 2, 3} s horizons common to both methods, using our sensor-grounded metrics. We also include FreeOcc-Cam, a camera variant using a Lift-Splat-Shoot (LSS) encoder Philion and Fidler [2020] with our forecasting modules unchanged.

Table 12: Comparison with SparseWorld-TC on our metrics ({1, 2, 3} s horizons).
<table><tr><td>Method</td><td>Modality</td><td>mIoU</td><td>NFCD@3s</td><td>FPS</td></tr><tr><td>SparseWorld-TC</td><td>Camera</td><td>23.7</td><td>0.82</td><td>9.8</td></tr><tr><td>FreeOcc-Cam (ours)</td><td>Camera</td><td>25.6</td><td>0.69</td><td>19.4</td></tr><tr><td>FreeOcc-E (ours)</td><td>LiDAR</td><td>30.2</td><td>0.47</td><td>27.0</td></tr></table>

FreeOcc-E leads on all metrics. The same-modality FreeOcc-Cam also surpasses SparseWorld-TC while running about twice as fast.

## M Camera Modality Ablation

To disentangle the LiDAR modality from the forecasting modules, we build FreeOcc-Cam by swapping only the LiDAR encoder for an LSS Philion and Fidler [2020] image-to-BEV encoder. All other components are unchanged. As a control, the camera baseline shares the same LSS encoder but replaces our forecasting stack with a plain feed-forward occupancy head (no ego-motion warp, temporal aggregation, flow matching, or ensemble). Both are trained on the same data.

Table 13: Isolating the modality from the forecasting modules (6-horizon average on nuScenes).
<table><tr><td>Configuration</td><td>Modality</td><td>mIoU</td></tr><tr><td>FreeOcc (full)</td><td>LiDAR</td><td>32.1</td></tr><tr><td>FreeOcc-Cam</td><td>Camera</td><td>26.5</td></tr><tr><td>Camera baseline (no warp/CFM/ensemble)</td><td>Camera</td><td>21.9</td></tr></table>

LiDAR exceeds camera by 5.6 mIoU, and FreeOcc-Cam exceeds the camera baseline by 4.6 mIoU.   
The forecasting modules thus add value independent of modality.