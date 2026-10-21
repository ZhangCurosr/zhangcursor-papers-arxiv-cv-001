# Causal Spatiotemporal State Evolution for Monocular 4D Reconstruction

Qiaowei Miao<sup>∗</sup> Zhejiang University

Yijing Guo<sup>∗</sup> ShanghaiTech University

Lipei Zhang University of Cambridge

Qifeng Cheng NIO

Yuru Xiao Harbin Institute of Technology

Ziqian Ni NIO

Sheng Yang NIO

## Abstract

While feed-forward models excel in static 3D geometry estimation, extending them to dynamic 4D scenes requires modeling visual geometry as a causally evolving state. Current methods typically rely on unordered temporal aggregation, which lacks the directional structure of physical time. Furthermore, existing supervision often treats geometry and motion as disjoint objectives, failing to exploit their inherent physical coupling. Consequently, they lack a unified constraint to effectively supervise temporal state transitions, leading to inconsistent 4D scene representations. To address this, we propose a feed-forward framework for causal 4D visual geometry learning from monocular videos. At its core, Causal Spatiotemporal Mamba (CSM) augments temporal aggregation with directional state propagation. Through complementary Temporal-Spatial and Spatial-Temporal Scans, CSM continuously propagates geometry-aware states across frames to model spatiotemporal evolution. To align this causal state evolution with observable dynamics, we introduce Unified Motion Supervision (UMS). Rather than treating motion as an independent objective, UMS leverages geometry-aware predictions from the camera and depth heads to construct a unified forward motion field that is consistent with the underlying 3D structure. This geometry-grounded supervision explicitly constrains inter-frame state transitions, enabling stable states propagation across frames. Extensive evaluations on dynamic benchmarks demonstrate state-of-the-art performance across video depth, camera pose, and optical flow estimation. These results validate that the synergy between the directional state propagation of Causal Spatiotemporal Mamba and Unified Motion Supervision effectively captures continuous scene evolution, significantly reducing temporal drift and reinforcing spatial-temporal consistency in 4D representation learning.

## 1 Introduction

While feed-forward models have driven remarkable progress in static 3D geometry estimation [29, 35, 25, 31, 27, 28], extending this paradigm to dynamic 4D scenes [35, 39, 8, 24, 5] calls for modeling visual geometry as a temporally evolving state, rather than simply aggregating observations across frames. Unlike static reconstruction, which primarily relies on cross-view geometric consistency, 4D geometry learning must further account for temporally ordered state propagation and motionconsistent inter-frame constraints. However, existing feed-forward spatiotemporal models often rely on temporal aggregation mechanisms that weakly encode the directional structure of time, while geometry and motion are typically supervised through separate objectives. This separation limits the supervision of geometry-consistent state evolution and can lead to temporally unstable dynamic geometry estimates. These limitations motivate a causal spatiotemporal formulation in which geometry-aware states are propagated along the video order and guided by geometry-grounded motion constraints.

However, existing feed-forward 4D methods lack an explicit mechanism for modeling such causal temporal state evolution. The dominant paradigm extends static 3D backbones by applying attentionbased temporal aggregation to video tokens [23, 25]. Although global or masked attention [39, 8] can restrict the visible context, it primarily captures temporal interactions through pairwise token affinities rather than directional state propagation. Consequently, these methods control which tokens are accessed, but do not explicitly define how geometry-aware states evolve sequentially, limiting their ability to model dynamic 4D scenes as causally ordered state transitions.

Furthermore, existing frameworks lack a unified supervisory formulation that couples geometry and motion during temporal state evolution. Recent feed-forward 4D methods [35, 39, 8, 24, 5] typically introduce motion cues, such as optical flow, scene flow, or segmentation masks, as auxiliary output-level objectives. Yet in dynamic 4D scenes, inter-frame motion is not an independent signal, but a geometry-dependent manifestation of the evolving scene state. Treating motion supervision separately from geometry evolution therefore weakens its role as a structural constraint on temporal state transitions. This motivates unified motion supervision that grounds inter-frame motion in evolving 3D geometry, encouraging more coherent 4D geometry estimation.

To overcome these limitations, we formulate monocular 4D visual geometry learning as directionally ordered spatiotemporal state evolution. We instantiate this formulation with Causal Spatiotemporal Mamba (CSM), a feed-forward framework that moves beyond correlation-based temporal aggregation by introducing directional state propagation over geometry-aware representations. To capture complementary spatial and temporal dependencies, CSM introduces Temporal-Spatial Scan and Spatial-Temporal Scan, which propagate geometry-aware states along ordered video streams while respecting temporal causality. To align this propagation with observable dynamics, we further introduce Unified Motion Supervision (UMS). UMS constructs a unified forward motion field from geometry-aware predictions, coupling inter-frame motion with evolving 3D geometry and providing a structural constraint for learning temporally coherent state transitions. Our contributions are summarized as follows:

• We formulate monocular 4D visual geometry learning as directionally ordered spatiotemporal state evolution, in which geometry-aware states are modeled along the temporal order of video frames to capture dynamic geometry evolution.

• We propose Causal Spatiotemporal Mamba (CSM), a feed-forward state-space framework that introduces complementary Temporal-Spatial Scan and Spatial-Temporal Scan to propagate geometry-aware states across video frames while respecting temporal causality.

• We introduce Unified Motion Supervision (UMS), a geometry-grounded supervision mechanism that constructs a unified forward motion field from predicted geometry states, coupling inter-frame motion with evolving 3D geometry to guide temporally coherent state evolution.

Extensive experiments on multiple dynamic 4D reconstruction benchmarks demonstrate the effectiveness of our model, showing strong performance in dynamic geometry estimation.

## 2 Related Work

Feed-forward 4D Visual Geometry. Feed-forward reconstruction methods have shifted geometry recovery from per-scene optimization to direct prediction, showing strong generalization in static 3D reconstruction [29, 25, 31, 33]. Recent works extend this paradigm to dynamic 4D scenes through motion-aware geometry estimation [35], streaming temporal processing [39], motion-aware masking [8], spatiotemporal geometry prediction [24], and flow- or scene-flow-based supervision [5]. These methods introduce temporal context and motion cues into feed-forward reconstruction, but temporal reasoning is commonly realized through attention-based aggregation, streaming windows, or pairwise matching. Such designs model interactions among frames, yet do not explicitly define a directional update rule for propagating geometry-aware states along the video order. In addition, motion cues are often used as auxiliary flow, scene-flow, mask, or prior-based objectives, leaving their relation to continuous geometry evolution only implicitly modeled. Consequently, the coupling between inter-frame motion and geometry-consistent state transitions remains limited. In contrast, our framework formulates monocular 4D visual geometry as causal spatiotemporal state evolution, instantiated by Causal Spatiotemporal Mamba (CSM) and guided by Unified Motion Supervision (UMS). We summarize these differences in Table 1.

Table 1: Methodological comparison of feed-forward 3D/4D visual geometry representations. We compare spatiotemporal modeling paradigms and supervision mechanisms across feed-forward 4D methods. Our model combines temporally ordered state propagation with Unified Motion Supervision (UMS), which provides geometry-grounded supervision by coupling inter-frame motion with evolving 3D geometry, rather than treating motion cues as isolated auxiliary objectives.
<table><tr><td rowspan="2">Method</td><td colspan="2">Spatiotemporal Modeling</td><td colspan="3">Supervision</td></tr><tr><td>Global Correlation</td><td>Causality Order</td><td>Geometry</td><td>Motion</td><td>Unified Geo-Motion</td></tr><tr><td>VGGT [25]</td><td>V</td><td>x</td><td></td><td>x</td><td>x</td></tr><tr><td>CUT3R [26]</td><td>V</td><td>x</td><td></td><td>x</td><td>x</td></tr><tr><td>StreamVGGT [39]</td><td>V</td><td>x</td><td></td><td></td><td>x</td></tr><tr><td>MoRE [8]</td><td></td><td>x</td><td></td><td></td><td>x</td></tr><tr><td>PAGE-4D [37]</td><td></td><td>××</td><td></td><td>×VV&gt;</td><td>××</td></tr><tr><td>Flow3r [5]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>V</td><td>V</td><td>L</td><td>V</td><td>L</td></tr></table>

State Space Models for Visual Representation. Structured State Space Models (SSMs) provide an efficient alternative to attention for sequence modeling [10], and Mamba further enables inputdependent selective state updates [9, 6]. Recent works have applied Mamba-style modules to image and video representation learning [15, 14], as well as static 3D reconstruction and novel view synthesis [20]. However, feed-forward 4D visual geometry requires more than directly applying generic visual scans to video tokens: the model needs to preserve spatial geometric structure while propagating states along the temporal order of dynamic scenes. Existing image- or video-oriented scan patterns are typically designed for general visual representation learning, rather than geometry aware state evolution in dynamic 4D reconstruction. CSM addresses this gap with complementary Temporal-Spatial Scan and Spatial-Temporal Scan, which adapt state-space propagation to temporally ordered 4D geometry learning.

Motion Supervision for Dynamic Geometry. Cross-frame motion correspondences provide important supervision for learning dynamic geometry. Point tracking methods such as TAPIR [7] and CoTracker3 [12] offer reliable but sparse matches, while optical flow methods such as RAFT [22] and GMFlow [34] provide dense pixel-wise motion cues. Recent feed-forward 4D methods further exploit flow or scene flow for dynamic supervision [5]. However, such motion cues are often formulated as standalone output-level objectives, leaving their relation to the predicted scene geometry and temporal state evolution only implicitly modeled. In contrast, our framework introduces Unified Motion Supervision (UMS), which couples inter-frame motion with evolving 3D geometry and provides geometry-grounded supervision for temporally coherent state evolution.

## 3 Method

Problem Formulation. We formulate monocular dynamic 4D visual geometry learning as feedforward prediction over an ordered video stream. Let the input be a sequence of $T$ frames, $\mathcal { T } =$ $( \mathbf { I } _ { 1 } , \ldots , \mathbf { I } _ { T } )$ , where each frame $\mathbf { I } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ . The network $f _ { \theta }$ maps this ordered sequence to a tuple of temporally aligned camera, geometry, and motion outputs:

$$
f _ { \theta } ( \mathcal { T } ) = \left( ( \hat { \mathbf { T } } _ { 1 } , \ldots , \hat { \mathbf { T } } _ { T } ) , ( \hat { \mathbf { K } } _ { 1 } , \ldots , \hat { \mathbf { K } } _ { T } ) , ( \hat { \mathbf { X } } _ { 1 } , \ldots , \hat { \mathbf { X } } _ { T } ) , ( \hat { \mathbf { F } } _ { 1 } , \ldots , \hat { \mathbf { F } } _ { T - 1 } ) \right) .\tag{1}
$$

Here, $\hat { \mathbf { T } } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ denotes the camera extrinsic matrix, $\hat { \mathbf { K } } _ { t } \in \mathbb { R } ^ { 3 \times 3 }$ denotes the camera intrinsic matrix, $\hat { \mathbf { X } } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ denotes the pixel-aligned 3D point map for frame $\mathbf { I } _ { t } ,$ and $\hat { \mathbf { F } } _ { t } \in \mathbb { R } ^ { H \times W \times 2 }$ denotes the forward optical flow from I<sub>t</sub> to $\mathbf { I } _ { t + 1 }$ . Since monocular 4D visual geometry is defined over an ordered video sequence, the temporal order of frames specifies the direction along which scene states evolve.

Overall Architecture. Under this formulation, our framework realizes monocular 4D visual geometry learning through causal spatiotemporal state evolution, as shown in Figure 1. It consists of two main components: (1) Causal Spatiotemporal Mamba (CSM) propagates geometry-aware tokens with directional state updates, allowing temporally ordered representation evolution rather than relying solely on correlation-based temporal aggregation (Section 3.1); (2) Unified Motion Supervision (UMS) constructs a geometry-grounded unified forward motion field from camera, geometry, and residual motion predictions, coupling inter-frame motion with evolving 3D geometry to guide temporally coherent state evolution (Section 3.2). The resulting spatiotemporal representations parameterize the camera, geometry, and motion-related outputs defined in Equation (1).

![](images/e115521394c7d140be36e8ce411ace991ef0a37b11690fe7789d92e6ba5517e7.jpg)  
Figure 1: Overview of our causal spatiotemporal state evolution framework. Given an ordered monocular video, the model first extracts geometry-aware tokens with global context, and then applies Causal Spatiotemporal Mamba (CSM) to propagate these tokens through causal spatiotemporal state evolution. Complementary Temporal-Spatial Scan (TS Scan) and Spatial-Temporal Scan (ST Scan) update geometry-aware states across temporal and spatial dimensions while preserving the temporal order of scene evolution. The updated spatiotemporal representations parameterize camera extrinsics, intrinsics, pixel-aligned 3D point maps, and forward optical flow. Unified Motion Supervision (UMS) constructs a unified forward motion field with geometry-grounded supervision mechanism, enforcing geometry-aligned inter-frame state transitions.

## 3.1 Causal Spatiotemporal Mamba

Monocular 4D visual geometry benefits from both global geometric context and temporally ordered state evolution. While global attention is effective for long-range token aggregation, it does not explicitly propagate geometry-aware states along the video order. We introduce CSM to augment geometry-aware tokens with directional state-space propagation. Unlike generic visual Mamba designs for image or video recognition [15, 14], CSM adopts structured spatiotemporal scans tailored to 4D geometry prediction, preserving intra-frame spatial structure while modeling inter-frame temporal evolution, as shown in Figure 1 and Figure 2.

From Attention Aggregation to State-Space Propagation. For a flattened spatiotemporal token $\mathbf { g } _ { i }$ , attention-based aggregation retrieves information from a visible token set $ { \boldsymbol \nu } ( i )$

$$
\mathbf { y } _ { i } = \sum _ { j \in \mathscr { V } ( i ) } \alpha _ { i j } \mathbf { W } _ { v } \mathbf { g } _ { j } , \qquad \alpha _ { i j } = \mathrm { s o f t m a x } _ { j } \left( \frac { ( \mathbf { W } _ { q } \mathbf { g } _ { i } ) ^ { \top } ( \mathbf { W } _ { k } \mathbf { g } _ { j } ) } { \sqrt { d } } \right) ,\tag{2}
$$

where V(i) denotes the visible token set. Global attention sets $\mathcal { V } ( i )$ to all video tokens, while causal or grouped variants restrict it using temporal masks or local groups. In all cases, the update is computed as a weighted aggregation over pairwise token affinities. Such a mechanism specifies which tokens contribute to the output, but does not explicitly define a state transition process for geometry-aware representations along the temporal order of the video, as shown in Figure 2.

CSM therefore augments global correlation modeling with state-space propagation over serialized geometry-aware tokens. Given a scan order $^ { r , }$ , let $\mathbf { S } ^ { r } = \{ \mathbf { s } _ { i } ^ { r } \} _ { i = 1 } ^ { T N }$ denote the token sequence obtained by serializing the spatiotemporal token grid according to $r ,$ where $T$ is the number of frames and N is the number of spatial tokens per frame. Starting from $\mathbf { h } _ { 0 } ^ { r } = \mathbf { 0 }$ , the state-space update is

$$
\mathbf { h } _ { i } ^ { r } = \bar { \mathbf { A } } _ { i } ^ { r } \mathbf { h } _ { i - 1 } ^ { r } + \bar { \mathbf { B } } _ { i } ^ { r } \mathbf { s } _ { i } ^ { r } , \qquad \mathbf { y } _ { i } ^ { r } = \mathbf { C } _ { i } ^ { r } \mathbf { h } _ { i } ^ { r } .\tag{3}
$$

Attention Receptive Field  
![](images/ed70d0bd71903cb194dba57551617c7bcca9f4f7b6fa7110d3950919bf5fe81f.jpg)

![](images/71e48075e84d4aa8d63bac672137f2d0676b2fec443e59720f4b7a10b07b5c05.jpg)

![](images/c11c3c93288c3ab7e294e70250640b2cdf872ca7b80fac2adab1b434e88aa776.jpg)

![](images/35d6c0636086e5bbccee43000b4b06ba8c49ea50c4305daf9d670e0ac061f62a.jpg)  
Figure 2: Comparison of 4D modeling paradigms. Attention-based methods [25, 37] aggregate context through pairwise token affinities, while Image/Video Mamba [15, 14] applies generic bidirec tional scans for visual representation learning. In contrast, CSM introduces geometry-aware causal scans that propagate states along the temporal order of the video. The complementary Temporal-Spatial Scan and Spatial-Temporal Scan jointly model temporal evolution and spatial structure, enabling ordered spatiotemporal state propagation for monocular 4D visual geometry.

Unrolling Equation (3) yields

$$
\mathbf { y } _ { i } ^ { r } = \sum _ { j = 1 } ^ { i } \mathbf { K } _ { i , j } ^ { r } \mathbf { s } _ { j } ^ { r } , \qquad \mathbf { K } _ { i , j } ^ { r } = \mathbf { C } _ { i } ^ { r } \left( \prod _ { k = j + 1 } ^ { i } \bar { \mathbf { A } } _ { k } ^ { r } \right) \bar { \mathbf { B } } _ { j } ^ { r } ,\tag{4}
$$

Here, $\mathbf { K } _ { i , j } ^ { r }$ denotes the effective transition from position j to position i along scan order r, with the empty product defined as identity. Compared with attention-based aggregation, state-space propagation accumulates context through recursively composed state transitions over the serialized token sequence. This makes the directional dependency along the scan explicit and provides an ordered propagation mechanism for geometry-aware spatiotemporal representations.

Temporal-Spatial and Spatial-Temporal Scans. As shown in Figure 1, given geometry-aware tokens $\mathbf { G } \in \bar { \mathbb { R } ^ { B \times T \times N \times C } }$ , where $T$ is the number of frames and N is the number of spatial tokens per frame, we denote the n-th token in frame t as $\mathbf { g } _ { t , n }$ . CSM serializes the same token grid with two complementary scan orders. The Temporal-Spatial Scan (TS Scan) prioritizes temporal trajectories by scanning tokens with the same spatial index consecutively:

$$
\mathbf { S } ^ { \mathrm { T S S } } = \bigl [ \mathbf { g } _ { 1 , 1 } , \ldots , \mathbf { g } _ { T , 1 } , \ldots , \mathbf { g } _ { 1 , N } , \ldots , \mathbf { g } _ { T , N } \bigr ] .\tag{5}
$$

The Spatial-Temporal Scan (ST Scan) instead follows a frame-major order, aggregating spatial context within each frame before advancing in time:

$$
\mathbf { S } ^ { \mathrm { S T S } } = \bigl [ \mathbf { g } _ { 1 , 1 } , \ldots , \mathbf { g } _ { 1 , N } , \ldots , \mathbf { g } _ { T , 1 } , \ldots , \mathbf { g } _ { T , N } \bigr ] .\tag{6}
$$

Both sequences are processed by the state-space update in Equation (3), reshaped back to the original layout, and fused by

$$
\tilde { \mathbf { G } } = \Phi _ { \mathrm { f u s e } } \left( \mathbf { Y } ^ { \mathrm { T S S } } , \mathbf { Y } ^ { \mathrm { S T S } } \right) ,\tag{7}
$$

where $\mathbf { Y } ^ { \mathrm { T S S } }$ and $\mathbf { Y } ^ { \mathrm { S T S } }$ are the two scan outputs $\mathrm { { , \Phi _ { f u s e } \left( \cdot \right) } }$ denotes a linear projection layer and $\tilde { \mathbf { G } }$ is the output of CSM block. Together, this dual-scan design ensures that TS Scan models how regions evolve over time, while ST Scan constructs the geometric layout of each frame.

## 3.2 Unified Motion Supervision

While CSM propagates geometry-aware states along the temporal order, it does not by itself guarantee that adjacent-frame transitions remain consistent with the underlying 3D scene structure. To address this, we introduce Unified Motion Supervision (UMS), which enforces geometry-consistent state evolution by grounding inter-frame transitions in motion signals derived from predicted camera and depth. Specifically, UMS constructs a unified forward motion field $\hat { \mathbf { F } } _ { t } = \hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } } + \hat { \mathbf { F } } _ { t } ^ { \mathrm { d y n } }$ , where $\hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } }$ is induced by projective geometry using predicted depth, intrinsics, and relative pose, and $\hat { \mathbf { F } } _ { t } ^ { \mathrm { d y n } }$ captures complementary motion components not explained by the geometry-induced term. By supervising this composed motion field, UMS enforces that state transitions are both temporally ordered and geometrically valid, effectively anchoring spatiotemporal representation dynamics to a physically consistent 3D scene.

Composition of Forward Motion Field. For each adjacent frame pair $\left( \mathbf { I } _ { t } , \mathbf { I } _ { t + 1 } \right)$ , UMS first derives a geometry-induced motion field from the predicted geometry and camera motion. For a pixel $\mathbf { p } _ { t } \mathbf { \bar { \Pi } } = ( u , v , 1 ) ^ { \top }$ , we project its depth-backprojected 3D point to the next frame:

$$
\mathbf { p } _ { t + 1 } = \pi \Big ( \hat { \mathbf { K } } _ { t + 1 } ( \hat { \mathbf { R } } _ { t  t + 1 } \hat { \mathbf { D } } _ { t } ( u , v ) \hat { \mathbf { K } } _ { t } ^ { - 1 } \mathbf { p } _ { t } + \hat { \mathbf { t } } _ { t  t + 1 } ) \Big ) , \qquad \hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } } \big ( u , v \big ) = \mathbf { p } _ { t + 1 } - \mathbf { p } _ { t } .\tag{8}
$$

Here, $\pi ( \cdot )$ denotes perspective projection, and the camera parameters are obtained from the predicted intrinsics and relative pose. This $\hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } }$ explains the displacement induced by camera motion under the predicted scene geometry.

To capture motion not explained by geometry-induced term, UMS predicts an complementary dynamic components from both geometry-aware states and image features. Specifically, we concatenate the updated geometry state $\tilde { \mathbf { G } } _ { t }$ with the corresponding image feature $\mathbf { R } _ { t }$ to form $\mathbf Z _ { t } = \big [ \tilde { \mathbf G } _ { t } , \mathbf R _ { t } \big ]$ Using $\hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } }$ as a geometry-derived initialization, the fused feature of frame $t + 1$ is warped toward frame t, and encoded by a learnable predictor $\psi _ { \phi }$

$$
\begin{array} { r } { \hat { \mathbf { F } } _ { t } ^ { \mathrm { d y n } } = \psi _ { \phi } \left( \mathbf { Z } _ { t } , \mathcal { W } ( \mathbf { Z } _ { t + 1 } , \hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } } ) , \hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } } \right) , } \end{array}\tag{9}
$$

where W(⋅) denotes differentiable backward warping. The predictor $\psi _ { \phi }$ follows the design style of WAFT [32], but remove the DPT head and operates on our spatiotemporal representations. The final forward flow and UMS objective are:

$$
\hat { \mathbf { F } } _ { t } = \hat { \mathbf { F } } _ { t } ^ { \mathrm { g e o } } + \hat { \mathbf { F } } _ { t } ^ { \mathrm { d y n } } , \qquad \mathcal { L } _ { \mathrm { U M S } } = \sum _ { t = 1 } ^ { T - 1 } \rho \big ( \hat { \mathbf { F } } _ { t } - \mathbf { F } _ { t } ^ { * } \big ) ,\tag{10}
$$

where $\mathbf { F } _ { t } ^ { * }$ is the optical-flow target and $\rho ( \cdot )$ is a robust penalty. By supervising the composed motion field, UMS enforces consistency between geometry predictions and inter-frame state transitions, providing unified geometry-aware motion supervision for causal state evolution.

## 3.3 Optimization

The proposed framework is trained end-to-end with joint geometry and motion supervision. For geometry supervision, we adopt the standard geometric losses used in prior feed-forward reconstruction methods [31], supervising camera poses, depth maps, and pixel-aligned 3D point maps with scale alignment. In addition to geometric supervision, we further introduce UMS as an additional motion constraint. For motion supervision, we supervise the forward flow composed by UMS using dense optical-flow targets. The overall objective is

$$
\mathcal { L } = \lambda _ { \mathrm { c a m } } \mathcal { L } _ { \mathrm { c a m } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { p o i n t } } \mathcal { L } _ { \mathrm { p o i n t } } + \lambda _ { \mathrm { U M S } } \mathcal { L } _ { \mathrm { U M S } } ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { c a m } }$ supervises camera translation and rotation, ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ and ${ \mathcal { L } } _ { \mathrm { p o i n t } }$ supervise scale-aligned depth and 3D point predictions, and ${ \mathcal { L } } _ { \mathrm { U M S } }$ supervises the composed forward flow and visibility predictions.

## 4 Experiments

Implementation Details. Our framework builds upon a VGGT-style architecture [25]. We use a frozen DINOv2 ViT-L/14 backbone with register tokens to extract dense geometric features, and construct the 4D representation with a 36-layer decoder that alternates between causal Mamba sequence modeling and global spatiotemporal attention. During training, we randomly sample contiguous clips of 2 to 24 frames with dynamic temporal batching. Training is performed in two stages, from lowresolution temporal initialization to high-resolution fine-tuning. We use AdamW with a weight decay of 0.05 and a OneCycleLR schedule with a peak learning rate of $5 \times 1 0 ^ { - 5 }$ , including a 10% warm-up followed by cosine annealing. For evaluation, we follow the protocols and metrics of VGGT [25] and CUT3R [26]. We compare with VGGT [25], StreamVGGT [39], MoRe [8], PAGE-4D [37], and CUT3R [26] using their publicly released pretrained models. Additional architectural and training details are provided in the Appendix.

Training dataset. For spatiotemporal adaptation, we construct a unified training corpus from nine datasets: TarTanAir [30], Virtual KITTI2 [4], Spring [16], PointOdyssey [36], BEDLAM [2], DynamicReplica [13], GTA-SfM [11], Hypersim [19], and ARKitScenes [1]. These datasets cover diverse scene types and motion patterns, including autonomous driving, indoor and outdoor environments, articulated human motion, synthetic object dynamics, and handheld camera trajectories.

Table 2: Camera pose estimation on dynamic benchmarks. Best results are in bold, second-best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="3">Sintel</td><td colspan="3">Bonn</td></tr><tr><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>VGGT</td><td>0.214</td><td>0.079</td><td>0.643</td><td>0.021</td><td>0.023</td><td>0.937</td></tr><tr><td>StreamVGGT</td><td>0.216</td><td>0.106</td><td>0.779</td><td>0.045</td><td>0.010</td><td>0.662</td></tr><tr><td>MoRe</td><td>0.147</td><td>0.082</td><td>0.616</td><td>0.021</td><td>0.012</td><td>0.650</td></tr><tr><td>PAGE-4D</td><td>0.175</td><td>0.074</td><td>0.602</td><td>0.021</td><td>0.013</td><td>0.650</td></tr><tr><td>CUT3R</td><td>0.235</td><td>0.068</td><td>0.669</td><td>0.035</td><td>0.009</td><td>0.680</td></tr><tr><td>Ours</td><td>0.116</td><td>0.050</td><td>0.552</td><td>0.020</td><td>0.008</td><td>0.670</td></tr></table>

Table 3: Video depth estimation on dynamic benchmarks. Metric $\delta < 1 . 2 5$ is reported in %. Best results are in bold, and second-best results are underlined. “OOM” denotes out-of-memory under the same single-pass feed-forward inference setting.
<table><tr><td rowspan="2">Method</td><td colspan="2">Sintel</td><td colspan="2">Bonn</td><td colspan="2">KITTI</td></tr><tr><td>Abs Rel↓</td><td>δ &lt; 1.25 ↑</td><td>Abs Rel↓</td><td>δ &lt; 1.25 ↑</td><td>Abs Rel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td>VGGT</td><td>0.471</td><td>0.486</td><td>0.103</td><td>0.903</td><td>0.143</td><td>0.804</td></tr><tr><td>StreamVGGT</td><td>0.354</td><td>0.591</td><td>OOM</td><td>O0M</td><td>0.176</td><td>0.718</td></tr><tr><td>MoRE</td><td>0.340</td><td>0.592</td><td>0.057</td><td>0.967</td><td>0.141</td><td>0.878</td></tr><tr><td>PAGE-4D</td><td>0.299</td><td>0.735</td><td>0.087</td><td>0.907</td><td>0.147</td><td>0.894</td></tr><tr><td>CUT3R</td><td>0.534</td><td>0.558</td><td>0.075</td><td>0.943</td><td>0.118</td><td>0.881</td></tr><tr><td>Ours</td><td>0.243</td><td>0.735</td><td>0.059</td><td>0.972</td><td>0.086</td><td>0.937</td></tr></table>

Evaluation datasets. We evaluate our model across three geometry and motion perception tasks: camera pose estimation, video depth estimation, and optical flow estimation. Our evaluation suite includes Sintel [3], Bonn [18], and KITTI [17], covering synthetic and real-world scenes, indoor and outdoor environments, and a range of dynamic motion patterns. For video depth estimation, we report results on Sintel [3], Bonn [18], and KITTI [17]. For camera pose estimation, we use Sintel [3], and Bonn [18], as they contain challenging dynamic motions and viewpoint changes that stress-test temporal geometric consistency. Finally, we evaluate optical flow estimation on the Sintel [3] and KITTI [17] datasets following the standard evaluation protocol, where dense public ground truth is available.

## 4.1 Results

Monocular Video Depth Estimation We evaluate monocular video depth estimation using Absolute Relative Error (Abs Rel) and threshold accuracy δ < 1.25, where lower Abs Rel and higher accuracy indicate better depth prediction. For fair comparison across methods with different depth parameterizations, we apply a unified alignment protocol before evaluation. Specifically, disparity or affine-invariant relative depth predictions are aligned to ground-truth depth using median scaling and scale-and-shift alignment. Table 3 reports results on Sintel, Bonn, and KITTI. Our model performs consistently across synthetic, real-world, indoor, and outdoor scenes. The improvements suggest that the proposed spatiotemporal representation effectively exploits cross-frame geometric cues, improving temporal depth consistency while maintaining accurate per-frame geometry.

Camera Pose Estimation We evaluate camera pose estimation on Sintel, and Bonn using absolute trajectory error (ATE), translational relative pose error (RPE-T), and rotational relative pose error (RPE-R), where lower values indicate better pose recovery. Table 2 reports the results. Compared with prior feed-forward 4D methods, our model shows improved pose accuracy in challenging dynamic scenes, including large viewpoint changes on Sintel. These gains suggest that causal spatiotemporal state propagation helps maintain temporally coherent pose estimates, while UMS couples inter-frame motion with evolving 3D geometry to improve trajectory consistency.

Optical Flow Estimation Although our framework is primarily designed for geometry-centric 4D scene understanding, evaluating the derived optical flow provides an additional way to assess the geometry-motion coupling induced by UMS. Specifically, we evaluate optical flow estimation on the Sintel and KITTI splits using the standard End-Point Error (EPE) metric and compare against the state-of-the-art feed-forward optical flow method Flow3r [5]. This evaluation examines whether the proposed directional state propagation and unified motion supervision can produce dense motion correspondences that are consistent with dynamic scene geometry. As illustrated qualitatively in Figure 4 and reported quantitatively in Table 4, our model achieves strong optical flow performance.

![](images/7d2cc0d8073e1d238b514ce9cc4c352e722be12a6435914367137a675500f46b.jpg)  
Figure 3: Qualitative comparison of dynamic geometry reconstruction. Our method produces more accurate and temporally coherent geometry in dynamic regions, with sharper structures and fewer artifacts than existing feed-forward 4D methods.

![](images/0c161d6e26510370dace42e0f97919fac435db32cb4614f420b4801c1e05350f.jpg)  
Figure 4: Qualitative comparison of optical flow estimation. Benefiting from our unified motion supervision, our model yields highly accurate motion fields with sharp boundaries. Notably, in the middle columns where moving objects occupy a small proportion and share high visual similarity with the background, Flow3r struggles to effectively detect the foreground optical flow. In contrast, our approach successfully disentangles these challenging motions.

These results indicate that UMS helps align spatiotemporal state propagation with observable inter-frame motion. Rather than treating optical flow as an isolated output task, our framework derives motion fields within a geometry-aware formulation, providing complementary evidence that the learned spatiotemporal representations capture meaningful dynamic scene evolution. This geometry-motion coupling also supports the improvements observed in video depth and camera pose estimation.

Table 4: Comparison results of optical flow estimation.
<table><tr><td>Method Sintel EPE↓ KITTI EPE↓</td><td></td></tr><tr><td>Flow3r 45.146</td><td>38.649</td></tr><tr><td>Ours 16.108</td><td>20.161</td></tr></table>

Qualitative Results. As shown in Figure 4, the visualization results demonstrate that our method produces noticeably better dynamic geometry than existing feed-forward 4D methods. In challenging regions [21] involving object motion, large deformation, or camera-induced motion, competing methods often suffer from distorted object shapes, blurred structures, unstable depth boundaries, or temporally inconsistent geometry across adjacent frames. In contrast, our model better preserves object contours and maintains more coherent 3D structures over time, leading to sharper and more stable reconstructions of dynamic scenes. This improvement suggests that the proposed causal spatiotemporal state propagation helps maintain temporally ordered geometry evolution, while Unified Motion Supervision provides geometry-grounded motion guidance for dynamic regions. Together, these qualitative results further highlight the benefit of coupling directional state propagation with unified motion supervision for geometry-aware dynamic scene modeling.

## 4.2 Ablation Study

We ablate the key design choices on Sintel, as shown in Table 5. For a controlled comparison, all variants are trained for 100 epochs on the same training subset. We compare five configurations: VGGT-style global attention only, causal attention, Vision-Mamba-style modeling extended to video [38], our CSM, and the full model with UMS.

Effect of spatiotemporal modeling. As shown in Table 5 rows 1 – 4 compare different spatiotemporal modeling paradigms. VGGT-style Global attention only captures long-range correlations but does not explicitly encode temporal order. Causal attention improves temporal directionality through masking, but still relies on pairwise token aggregation rather than state propagation. Vision-Mamba-style modeling introduces state-space propagation, yet its generic scan pattern is not tailored to geometryaware video modeling. In contrast, CSM achieves better performance across depth and pose metrics, suggesting that the proposed Temporal-Spatial and Spatial-Temporal scans more effectively capture ordered temporal evolution while preserving spatial geometric structure.

Effect of Unified Motion Supervision. Adding UMS on top of CSM further improves both video depth and pose estimation. This indicates that directional state propagation alone benefits from additional geometry-grounded motion supervision. By coupling inter-frame motion with predicted geometry, UMS provides a structural constraint for learning more temporally coherent representations, leading to reduced trajectory error and improved depth consistency. Overall, the ablation confirms that CSM and UMS provide complementary benefits: CSM improves causal spatiotemporal representation learning, while UMS further strengthens geometry-motion consistency.

## 5 Conclusion

In this paper, we formulated monocular 4D visual geometry learning as causally ordered spatiotemporal state evolution, providing a structured perspective for extending feed-forward 3D reconstruction to dynamic scenes. We introduced Causal Spatiotemporal Mamba

Table 5: Ablation study on architectural components and training strategies. Evaluation is performed on Sintel. Best results are shown in bold.
<table><tr><td colspan="2">Components</td><td colspan="2">Video Depth</td><td colspan="3">Pose</td></tr><tr><td>Modeling</td><td>UMS</td><td>Abs Rel↓</td><td>δ &lt; 1.25 ↑</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>Only Global-Attn</td><td>x</td><td>0.481</td><td>0.500</td><td>0.321</td><td>0.213</td><td>0.814</td></tr><tr><td>Causal-Attn</td><td>x</td><td>0.451</td><td>0.592</td><td>0.311</td><td>0.191</td><td>0.637</td></tr><tr><td>Vision Mamba</td><td>x</td><td>0.309</td><td>0.564</td><td>0.329</td><td>0.177</td><td>0.732</td></tr><tr><td>CSM</td><td>x</td><td>0.253</td><td>0.631</td><td>0.124</td><td>0.078</td><td>0.559</td></tr><tr><td>CSM</td><td>V</td><td>0.243</td><td>0.689</td><td>0.116</td><td>0.050</td><td>0.552</td></tr></table>

(CSM), which augments geometry-aware representations with directional state propagation, enabling temporally ordered spatiotemporal modeling beyond purely correlation-based temporal aggregation. Furthermore, our Unified Motion Supervision (UMS) provides a geometry-grounded approach to motion supervision. By constructing a unified motion field from predicted camera, geometry, and residual motion cues, UMS couples inter-frame motion with evolving 3D structure and provides a structural constraint for temporally coherent state evolution. Extensive evaluations demonstrate the effectiveness of our approach, showing strong performance in dynamic geometry estimation and improved temporal consistency across diverse scenes.

Limitations and Future Work. While our framework improves dynamic scene representation, UMS relies on reasonably accurate camera and depth predictions to derive geometry-consistent motion signals. In scenarios involving severe motion blur, degenerate camera trajectories, large occlusions, or unreliable geometry estimates, the resulting motion supervision may become less effective. Future work could integrate learnable pose-refinement modules into the causal spatiotemporal latent space, enabling joint refinement of camera trajectories and geometry-aware states. We also plan to investigate the scalability of the proposed state-propagation paradigm on larger and more heterogeneous inthe-wild video datasets. More broadly, this work suggests that aligning sequence modeling with geometry-aware motion constraints is a promising direction for scalable and temporally consistent 4D visual representation learning.

## References

[1] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. In Adv. Neural Inform. Process. Syst., 2021.

[2] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8726–8737, 2023.

[3] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Eur. Conf. Comput. Vis., pages 611–625, 2012.

[4] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020.

[5] Zhongxiao Cong, Qitao Zhao, Minsik Jeon, and Shubham Tulsiani. Flow3r: Factored flow prediction for scalable visual geometry learning. In IEEE Conf. Comput. Vis. Pattern Recog., 2026.

[6] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Int. Conf. Mach. Learn., 2024.

[7] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Int. Conf. Comput. Vis., pages 10061–10072, 2023.

[8] Juntong Fang, Zequn Chen, Weiqi Zhang, Donglin Di, Xuancheng Zhang, Chengmin Yang, and Yu-Shen Liu. More: Motion-aware feed-forward 4d reconstruction transformer. In IEEE Conf. Comput. Vis. Pattern Recog., 2026.

[9] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Conference on Language Modeling, 2023.

[10] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In Int. Conf. Learn. Represent., 2022.

[11] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In IEEE Conf. Comput. Vis. Pattern Recog., pages 2821–2830, 2018.

[12] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labeling real videos. In Int. Conf. Comput. Vis., 2025.

[13] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In IEEE Conf. Comput. Vis. Pattern Recog., pages 13229–13239, 2023.

[14] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. In Eur. Conf. Comput. Vis., pages 237–255, 2024.

[15] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. In Adv. Neural Inform. Process. Syst., pages 103031–103063, 2024.

[16] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. Spring: A highresolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4981–4991, 2023.

[17] Moritz Menze and Andreas Geiger. Object scene flow for autonomous vehicles. In IEEE Conf. Comput. Vis. Pattern Recog., pages 3061–3070, 2015.

[18] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7855–7862, 2019.

[19] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Int. Conf. Comput. Vis., 2021.

[20] Qiuhong Shen, Zike Wu, Xuanyu Yi, Pan Zhou, Hanwang Zhang, Shuicheng Yan, and Xinchao Wang. Gamba: Marry gaussian splatting with mamba for single-view 3d reconstruction. IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[21] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 573–580, 2012.

[22] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Eur. Conf. Comput. Vis., pages 402–419, 2020.

[23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Adv. Neural Inform. Process. Syst., 2017.

[24] Haonan Wang, Hanyu Zhou, Haoyue Liu, and Luxin Yan. 4d-vggt: A general foundation model with spatiotemporal awareness for dynamic scene geometry estimation. arXiv preprint arXiv:2511.18416, 2025.

[25] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5294–5306, 2025.

[26] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10510–10522, 2025.

[27] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5261–5271, 2025.

[28] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. In Adv. Neural Inform. Process. Syst., 2025.

[29] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In IEEE Conf. Comput. Vis. Pattern Recog., pages 20697–20709, 2024.

[30] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916, 2020.

[31] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In Int. Conf. Learn. Represent., 2026.

[32] Yihan Wang and Jia Deng. Waft: Warping-alone field transforms for optical flow. In Int. Conf. Learn. Represent., 2026.

[33] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. In Adv. Neural Inform. Process. Syst., 2025.

[34] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8121–8130, 2022.

[35] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In Int. Conf. Learn. Represent., 2025.

[36] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Int. Conf. Comput. Vis., pages 19855–19865, 2023.

[37] Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, and Mengyu Wang. Page-4d: Disentangled pose and geometry estimation for vggt-4d perception. In Int. Conf. Learn. Represent., 2026.

[38] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024.

[39] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. In Int. Conf. Learn. Represent., 2026.