# DEQ-Stereo: Deep Equilibrium Video Stereo Matching via Adaptive Convergence

Kiljoon Han<sup>1</sup> , Kyumin Hwang<sup>2</sup> , Wonhyeok Choi<sup>1</sup> , Jeonghoon Kim<sup>1</sup> , and Sunghoon Im<sup>2,†</sup>

<sup>1</sup> Daegu Gyeongbuk Institute of Science and Technology, Daegu, South Korea

<sup>2</sup> Korea Advanced Institute of Science and Technology, Daejeon, South Korea

Abstract. Iterative refinement models have significantly improved stereo matching accuracy, but their rigid, fixed update schedules incur heavy computational costs and a strict trade-of between speed and accuracy. This challenge is magnified in video streams, where traditional online methods resort to reducing feature channels or relying on expensive temporal warping. To address this, we propose DEQ-Stereo, a novel video stereo matching framework based on the Deep Equilibrium (DEQ) paradigm. Instead of unrolling a fixed number of iterations, DEQ-Stereo reformulates the refinement as an implicit fixed-point problem, adaptively terminating once convergence is met. To ensure fast and stable intra-frame convergence, we introduce the Convolutional Simplified Gated Unit (ConvSGU), which eliminates unnecessary non-linearities by modeling the latent state as a gated, element-wise sum of inputs. Furthermore, to exploit cross-frame temporal coherence, we propose Temporal State Transfer (TST), a warm-start strategy that propagates the converged hidden state to the next frame without requiring explicit pose or warping priors. Extensive evaluations demonstrate that DEQ-Stereo delivers state-of-the-art performance with a superior accuracy-tolatency trade-of.

Keywords: Stereo Matching · Deep Equilibrium Models · Video Depth

## 1 Introduction

Stereo matching is a fundamental problem in computer vision, aimed at estimating depth by computing disparities between a pair of images. It underpins various 3D perception tasks, including autonomous driving [14,31], robotics [45], augmented reality [12], and 3D scene reconstruction [5]. Recent learning-based approaches, particularly those based on iterative refinement [24, 26, 40], have significantly improved accuracy. These methods refine the disparity map across multiple steps, progressively enlarging the receptive field and enabling robust predictions in challenging areas like textureless regions [24, 47] or thin structures [9, 37]. Their generalization ability and optimization-like behavior have made them widely adopted in both static and sequential stereo settings. However, their inference time increases linearly with the number of iterations, which can be a bottleneck in time-sensitive environments.

![](images/52e92f24e7b740001d55d781093915d5373e5024f0e69ea9403a6c1d1fb89c91.jpg)  
Fig. 1: Comparisons of Iterative Stereo Matching. Traditional methods (e.g., IGEV-Stereo) use a fixed number of iterations, resulting in a trade-of between accuracy and eficiency. In contrast, our method adaptively determines the number of iterations, achieving high accuracy and eficiency.

While stereo matching is often benchmarked on individual image pairs, many real-world applications require continuous stereo video processing. Applying iterative models independently to each frame leads to redundant computation and temporal inconsistencies across frames. Recent video stereo methods incorporate temporal cues to improve both runtime and coherence. These can be broadly categorized into ofline and online approaches. Ofline methods [19, 20] exploit full sequences for improved accuracy but are impractical for real-time use, as they rely on future frames. Online methods [25,42,46] are designed for real-time processing by using lightweight models and previous-frame information, but sufer from accumulated errors and require reliable pose and disparity priors.

To address the limitations of existing video stereo matching methods, we propose DEQ-Stereo, a novel framework based on the Deep Equilibrium (DEQ) model [3], designed for eficient and temporally consistent stereo matching in video sequences. Unlike conventional approaches that rely on a fixed number of iterative updates, DEQ-Stereo directly solves for the fixed point of the update module, eliminating the need for explicit unrolling and enabling adaptive inference based on a convergence criterion as shown in Fig. 1. This yields more efficient, stable, and scalable inference across frames. To support this equilibriumbased formulation, we propose two complementary strategies aimed at improving intra-frame convergence stability and enhancing robustness across frames.

First, we introduce the Convolutional Simplified Gated Unit (ConvSGU), designed to accelerate and stabilize per-frame convergence. Inspired by Recurrent Additive Networks (RAN) [23], ConvSGU simplifies the gating structure of ConvGRU by expressing the latent state as a gated, element-wise sum of inputs, thereby removing unnecessary non-linearities. This simplification facilitates faster and more stable convergence to equilibrium states within the DEQ formulation. Furthermore, to exploit temporal coherence in video, we propose Temporal State Transfer (TST), a mechanism that reuses the converged hidden state from the previous frame to initialize the current one. This warm-start strategy encourages consistency over time and improves convergence speed, particularly in scenes with low inter-frame motion. By integrating the DEQ paradigm with ConvSGU and TST, DEQ-Stereo produces temporally consistent, high-quality disparity estimates while significantly reducing computational cost. Extensive evaluation across multiple benchmarks including SceneFlow [29], TartanAir [36], KITTI2015 [31], and ETH3D SLAM [33], demonstrates that our method achieves the best trade-of between eficiency and accuracy.

Our main contributions can be summarized as follows:

1. We propose DEQ-Stereo, a novel stereo matching framework leveraging Deep Equilibrium models for adaptive and accurate iterative refinement.

2. To improve convergence from both intra-frame and inter-frame perspectives, we propose ConvSGU enhancing convergence stability within each frame, and TST accelerating convergence across video frames.

3. Our method achieves state-of-the-art performance across multiple online video stereo benchmarks, delivering the best trade-of between inference speed and accuracy.

## 2 Related Work

## 2.1 Per-frame Stereo Matching Networks

Early deep learning stereo methods centered on building a cost volume and regularizing it with computationally expensive 3D convolutions to directly regress disparity [7, 16, 21, 44]. While these methods laid the groundwork, their high computational cost motivated further refinements in cost aggregation strategies [27,39]. A significant shift occurred with the introduction of iterative refinement architectures, which adapted RAFT [34] from optical flow for stereo matching [26]. This design, while powerful, relies on a fixed number of iterations—a core limitation forcing a trade-of between accuracy and eficiency. Subsequent work has focused on improving performance within this fixed iteration, either by enhancing frameworks with hierarchical processing or better geometric representations [24, 40, 43], or by designing specialized recurrent modules to preserve high-frequency details [9, 37, 47]. However, despite these advances, the fundamental issue of fixed iteration remains.

## 2.2 Video Stereo Matching Networks

In video stereo matching, methods leverage temporal information for eficiency and consistency. While ofline approaches [18–20] yield high-quality, temporally smooth results, their non-causal nature makes them unsuitable for real-time applications. Online methods [12,25,42] are therefore more common, ranging from early RNN-based designs [48] to lightweight temporal modules and adaptations of iterative refinement architectures prominent in per-frame matching. Focusing on the latter, these iterative approaches operate with a low, fixed number of iterations, supplementing their limited refinement capabilities by warping the depth from the previous frame to provide a strong initial estimate. This warping strategy, however, necessitates auxiliary camera parameters, and its direct injection of prior estimates hinders the model’s ability to correct errors, leading to their accumulation over time. This specific vulnerability of fixed, low-iteration schemes directly motivates our adaptive approach.

## 2.3 Implicit Deep Learning

Unlike traditional fixed-sequence layers, Implicit Deep Learning networks—such as Deep Equilibrium Models (DEQ) [3] and Neural ODEs [8]—define outputs as fixed-point solutions of a parameterized process, enabling efectively infinite depth. By adaptively determining the iterations needed to reach equilibrium, DEQ overcomes the ineficiencies of fixed-iteration stereo matching, yielding significant computational savings without sacrificing refinement quality. This implicit paradigm has been successfully applied across diverse domains, including difusion models [6, 32], object detection [35], and graph networks [15]. In dense correspondence, DEQ-Flow [2] pairs the DEQ framework with ConvGRU for optical flow, reusing both the predicted flow and hidden state from the previous frame to accelerate convergence. While our TST shares this warm-start intuition, it difers crucially by propagating only the hidden state. Empirically, we find that this selective propagation reduces sensitivity to complex scene dynamics, ultimately improving temporal stability in video stereo matching.

## 3 Preliminaries

## 3.1 Iterative Stereo Matching

Iterative stereo matching methods [24, 26, 40] progressively refine disparity estimates. This process can be defined as a parameterized function $f _ { \theta }$ that updates hidden state $\mathbf { h } ^ { k }$ and disparity $\mathbf { d } ^ { k }$ at iteration k using the correlation volume C:

$$
[ \mathbf h ^ { k + 1 } , \mathbf d ^ { k + 1 } ] = f _ { \theta } ( \mathbf h ^ { k } , \mathbf d ^ { k } , \mathbf C ) .\tag{1}
$$

Internally, $f _ { \theta }$ typically employs a recurrent unit $( \mathrm { e . g . }$ , ConvGRU) represented as $\scriptstyle { \mathcal { R } } _ { \theta }$ , operating as follows:

$$
\begin{array} { r } { \mathbf { x } ^ { k } = \mathrm { M o t i o n E n c } ( \mathbf { d } ^ { k } , \mathrm { L o o k U p } ( \mathbf { C } , \mathbf { d } ^ { k } ) ) , \quad \mathbf { h } ^ { k + 1 } = \mathcal { R } _ { \theta } ( \mathbf { h } ^ { k } , \mathbf { x } ^ { k } ) , } \\ { \mathbf { d } ^ { k + 1 } = \mathbf { d } ^ { k } + \mathrm { D i s p H e a d } ( \mathbf { h } ^ { k + 1 } ) , \quad \quad } \end{array}\tag{2}
$$

where $\mathbf { x } ^ { k }$ is the motion feature extracted by MotionEnc $( \cdot , \cdot )$ consisting of two convolutional layers, $\mathrm { L o o k U p } ( \cdot , \cdot )$ samples local geometry from the correlation volume C guided by $\mathbf { d } ^ { k }$ as proposed in previous works [26,34], and d<sup>k+1</sup> $\mathbf { d } ^ { k + 1 }$ is updated by adding the residual disparity predicted by the disparity head DispHead(·).

Iterative methods excel by aggregating features across multiple recursive steps, efectively combining local matching cues with global context. A key limitation is their reliance on a fixed iteration count N, chosen empirically rather than based on explicit convergence criteria. This results in a trade-of between accuracy and eficiency: insuficient iterations impair disparity quality, while excessive iterations increase inference time. This balance becomes even more critical in video stereo settings, where real-time constraints and dynamic scene content demand adaptive iteration control to optimize both speed and accuracy.

## 3.2 Deep Equilibrium Paradigm

The DEQ [2, 3] reformulates the update operation to an implicit function and finds a joint equilibrium state $\mathbf { s } ^ { * } = [ \mathbf { h } ^ { * } , \mathbf { d } ^ { * } ]$ that satisfies:

$$
\mathbf { s } ^ { * } = [ \mathbf { h } ^ { * } , \mathbf { d } ^ { * } ] = f _ { \boldsymbol { \theta } } ( \mathbf { h } ^ { * } , \mathbf { d } ^ { * } , \mathbf { C } ) \quad \Leftrightarrow \quad g _ { \boldsymbol { \theta } } ( \mathbf { s } ^ { * } , \mathbf { C } ) \equiv f _ { \boldsymbol { \theta } } ( \mathbf { s } ^ { * } , \mathbf { C } ) - \mathbf { s } ^ { * } \to 0 ,\tag{3}
$$

where $\mathbf { s } ^ { * }$ is the joint state that no longer changes under further applications of an update function $f _ { \theta }$ , and $g _ { \theta }$ is termed the residual function. The core idea of DEQ is to find $\mathrm { a }$ joint equilibrium state $\mathbf { s } ^ { * }$ by finding the root of $g _ { \boldsymbol { \theta } } ( \mathbf { s } ^ { * } , \mathbf { C } ) = 0$ through a numerical solver, instead of performing a fixed number of explicit iterations. This numerical solver performs the forward pass by iteratively updating the joint state s. At each solver step $k ,$ the state $\mathbf { s } ^ { k }$ is updated to $\mathbf { s } ^ { k + 1 }$ as follows:

$$
\begin{array} { r } { \mathbf { s } ^ { k + 1 } = \mathbf { s } ^ { k } - \alpha \mathbf { J } _ { g _ { \theta } | \mathbf { s } ^ { k } } ^ { - 1 } g _ { \theta } ( \mathbf { s } ^ { k } , \mathbf { C } ) , \quad \mathbf { J } _ { g _ { \theta } | \mathbf { s } ^ { k } } ^ { - 1 } \approx \mathbf { B } _ { g _ { \theta } } ^ { k } , } \\ { \mathbf { B } _ { g _ { \theta } } ^ { k } = \mathbf { B } _ { g _ { \theta } } ^ { k - 1 } + \frac { \varDelta \mathbf { s } ^ { k } - \mathbf { B } _ { g _ { \theta } } ^ { k - 1 } \varDelta g _ { \theta } ^ { k } } { { \varDelta \mathbf { s } ^ { k } } ^ { \top } \mathbf { B } _ { g _ { \theta } } ^ { k - 1 } \varDelta g _ { \theta } ^ { k } } { \varDelta \mathbf { s } ^ { k } } ^ { \top } \mathbf { B } _ { g _ { \theta } } ^ { k - 1 } , } \end{array}\tag{4}
$$

where α denotes the step size, and $\mathbf { J } _ { g _ { \theta } \left| \mathbf { s } ^ { k } \right. } ^ { - 1 }$ is the inverse Jacobian of $g _ { \theta }$ with respect to the current state $\mathbf { s } ^ { k }$ , which is essential for determining the update direction towards the root of $g _ { \theta }$ . The update rule for $\mathbf { B } _ { g _ { \theta } } ^ { k } ~ ( \mathrm { E q . ~ ( 4 ) } )$ is Broyden’s method, a quasi-Newton approach that eficiently approximates the inverse Jacobian to determine the update direction. While this method provides the direction, the step size α is crucial for ensuring stable convergence. For this purpose, an Armijo backtracking line search [1] is employed to adaptively find a step size that guarantees a suficient decrease in the norm of the residual function, $| | g _ { \boldsymbol \theta } ( \mathbf { s } ^ { * } , \mathbf { C } ) | |$ ensuring robust progress towards the joint equilibrium state.

## 4 Method

## 4.1 Deep Equilibrium Stereo Matching

Building upon the principles of Deep Equilibrium (DEQ) outlined in Sec. $^ { 3 , }$ particularly their capacity for adaptive termination, we propose DEQ-Stereo as illustrated in Fig. 2. This approach replaces the standard recurrent refinement module in iterative stereo matching with a DEQ module to overcome the ineficiencies of fixed-iteration schemes. DEQ-stereo obtains the left and right images $\mathbf { I } _ { L }$ and ${ \mathbf { I } } _ { R }$ as input, designed to produce the disparity map d between two images. We first extract features $\mathbf { f } _ { L } , \mathbf { f } _ { R }$ of images using the shared feature encoder FeatureNet(·). In parallel, the left image is passed through a context network ContextNet(·) to generate an initial hidden state $\mathbf { h } ^ { 0 }$ as follows:

![](images/7cd598df7ab4945ea2799862c4832597c79cb72981b92a466f591f24937cffa6.jpg)  
Fig. 2: Overall pipeline. (a) Overview of DEQ-Stereo. From stereo pairs, we construct a correlation volume and regress an initial disparity, which is refined via an implicit, convergence-driven DEQ process. Unlike prior iterative methods, DEQ-Stereo supports adaptive termination and faster convergence, enabled by two key components: ConvSGU and Temporal State Transfer. We denote the frame index as $t ,$ the DEQ iteration step as $k ,$ and the converged state as ∗. (b) Standard ConvGRU structure. (c) Our proposed ConvSGU, a simplified variant of ConvGRU, which defines the residual function $g _ { \boldsymbol { \theta } } ( \cdot )$ for the DEQ solver.

$$
\mathbf { f } _ { L } = \mathrm { F e a t u r e N e t } ( \mathbf { I } _ { L } ) , \quad \mathbf { f } _ { R } = \mathrm { F e a t u r e N e t } ( \mathbf { I } _ { R } ) , \quad \mathbf { h } ^ { 0 } = \mathrm { C o n t e x t N e t } ( \mathbf { I } _ { L } ) .\tag{5}
$$

Following IGEV-Stereo [40], we then construct a correlation volume $\mathbf { C } \in \mathbb { R } ^ { ( N _ { g } + 1 ) \times D \times H \times W }$ composed of one All-Pairs Correlation $\mathbf { C } _ { A P C }$ and $N _ { g }$ Group-Wise Correlations $\mathbf { C } _ { G W C }$ . From the C<sub>GWC</sub>, we then regress an initial disparity map $\mathbf { d } ^ { 0 }$ . The volume and initial disparity regression are formulated as follows:

$$
\begin{array} { l } { { { \bf d } ^ { 0 } = \displaystyle \sum _ { d = 0 } ^ { D - 1 } d \times \mathrm { S o f t m a x } ( { \bf C } _ { G W C } ( d ) ) , ~ { \bf C } _ { G W C } = \mathrm { R e g } \bigl ( { \frac { N _ { g } } { N _ { c } } } \langle { \bf f } _ { L } ^ { g } ( x , y ) , { \bf f } _ { R } ^ { g } ( x - d , y ) \rangle \bigr ) , } } \\ { { { \bf C } = [ { \bf C } _ { A P C } , { \bf C } _ { G W C } ] , ~ { \bf C } _ { A P C } ( d , x , y ) = \langle { \bf f } _ { L } ( x , y ) , { \bf f } _ { R } ( x - d , y ) \rangle , } } \end{array}\tag{6}
$$

where $d$ is the disparity index, $g$ is the group index for $\mathbf { C } _ { G W C } , N _ { c }$ is the number of feature channels, $N _ { g }$ is the number of groups, Reg(·) denotes the regularization network, and $[ \cdot , \cdot ]$ denotes channel-wise concatenation.

With the initial hidden state $\mathbf { h } ^ { 0 }$ and disparity ${ \bf d } ^ { 0 }$ , we define the joint state as $\mathbf { s } ^ { k } = [ \mathbf { h } ^ { k } , \mathbf { d } ^ { k } ]$ . At each solver step $k \in \{ 0 , . . . , K _ { \mathrm { s t o p } } \}$ , the DEQ solver D iteratively updates the current joint state $\mathbf { s } ^ { k }$ using the correlation volume C and

![](images/df34c031740cca90380278fbae995faf9923b7aa0c13c2fa41740dc279ab89e6.jpg)  
Fig. 3: Illustration of TST. The output state from the previous frame is reused to initialize the current frame, allowing faster and more stable disparity refinement.

the residual function $g _ { \theta }$ , following the update rule defined in Eq. (4). The DEQ solver D iteratively updates the joint state, terminating after $K _ { \mathrm { s t o p } }$ steps when the following stopping criterion is satisfied:

$$
\begin{array} { r } { \| g _ { \theta } ( \mathbf { s } ^ { k } , \mathbf { C } ) \| < \epsilon \quad \mathrm { o r } \quad k \geq \mathrm { M a x I t e r } , } \end{array}\tag{7}
$$

where ϵ is a convergence threshold, and MaxIter is the iteration limit. The actual number of steps until convergence is recorded as $K _ { \mathrm { s t o p } }$

## 4.2 Convolutional Simplified Gated Unit

In DEQ-based networks, the convergence speed of the residual function critically afects inference eficiency. While ConvGRU provides expressive modeling through gating, its complex nonlinear update rules can impede convergence. To address this, we propose the Convolutional Simplified Gated Unit (ConvSGU), a streamlined variant of ConvGRU inspired by the Recurrent Additive Network (RAN) [23]. ConvSGU simplifies recurrent dynamics while retaining modeling capacity, defined as:

$$
\begin{array} { r } { \mathbf { z } = \sigma ( \operatorname { C o n v } ( [ \mathbf { h } ^ { k } , \mathbf { x } ^ { k } ] , W _ { i } ) + \mathbf { b } _ { i } ) , \quad \mathbf { f } = \sigma ( \operatorname { C o n v } ( [ \mathbf { h } ^ { k } , \mathbf { x } ^ { k } ] , W _ { f } ) + \mathbf { b } _ { f } ) , } \\ { \mathbf { c } = \operatorname { t a n h } ( \operatorname { C o n v } ( [ \mathbf { h } ^ { k } , \mathbf { x } ^ { k } ] , W _ { c } ) + \mathbf { b } _ { c } ) , \quad \mathbf { h } ^ { k + 1 } = ( \mathbf { z } \cdot \mathbf { c } + ( 1 - \mathbf { z } ) \cdot \mathbf { h } ^ { k } ) \cdot \mathbf { f } , } \end{array}\tag{8}
$$

where $\sigma$ is the activation function, and $\mathbf { b } _ { i } , \mathbf { b } _ { f } .$ , and $\mathbf { b } _ { c }$ are constant inputs generated from the left image. The variables z, f, and c denote the update gate, forget gate, and candidate cell state, respectively. ConvSGU modifies the standard ConvGRU by removing the reset gate—eliminating unstable multiplicative dynamics [3, 23]—and introducing a forget gate f as a dynamic damping factor. These changes smooth the residual function and enhance convergence stability in the DEQ framework.

## 4.3 Temporal State Transfer

For sequential inputs such as video data, consecutive frames exhibit strong temporal coherence. We introduce Temporal State Transfer (TST) to exploit this property, a mechanism designed exclusively to accelerate the inference of DEQ-Stereo in video settings. Inspired by Fixed-point Reuse in DEQ-Flow [2], TST selectively reuses the converged hidden state $\mathbf { h } _ { t - 1 } ^ { * }$ from the previous joint equilibrium state $\mathbf { s } _ { t - 1 } ^ { * } = [ \mathbf { h } _ { t - 1 } ^ { * } , \mathbf { d } _ { t - 1 } ^ { * } ]$ . The converged disparity $\mathbf { d } _ { t - 1 } ^ { * }$ , however, is not reused; instead, the initial disparity ${ \bf d } _ { t } ^ { 0 }$ is freshly regressed from the current correlation volume $\mathbf { C } _ { t }$ . By combining these, we construct the initial joint state for the current frame $t \in \{ 1 , . . . , T \}$ . Formally, the DEQ solver D for the sequence is defined as follows:

$$
\mathbf { s } _ { t } ^ { * } = \mathcal { D } ( \mathbf { s } _ { t } ^ { 0 } , g _ { \theta } , \mathbf { C } _ { t } ) = \mathcal { D } ( [ \mathbf { h } _ { t - 1 } ^ { * } , \mathbf { d } _ { t } ^ { 0 } ] , g _ { \theta } , \mathbf { C } _ { t } ) .\tag{9}
$$

The intuition is that the previous hidden state $\mathbf { h } _ { t - 1 } ^ { * }$ serves as a strong initia guess for frame $t ,$ especially when inter-frame changes are small. This approach significantly reduces the number of iterations needed per frame, resulting in much faster inference speed that is highly advantageous for real-time video stereo applications. By leveraging temporal continuity, TST also enhances temporal consistency across frames in most scenarios.

## 4.4 Loss Function

For training, gradients $\frac { \partial { \mathcal { L } } } { \partial \theta }$ are eficiently computed using the Implicit Function Theorem (IFT) [2, 3, 22].

$$
\frac { \partial \mathcal { L } } { \partial \theta } = \frac { \partial \mathcal { L } } { \partial \mathbf { s } ^ { * } } \frac { \mathrm { d } \mathbf { s } ^ { * } } { \mathrm { d } \theta } = - \frac { \partial \mathcal { L } } { \partial \mathbf { s } ^ { * } } ( \mathbf { J } _ { g _ { \theta } | \mathbf { s } ^ { * } } ^ { - 1 } ) \frac { \partial f _ { \theta } ( \mathbf { s } ^ { * } , \mathbf { C } ) } { \partial \theta } ,\tag{10}
$$

where $\frac { \partial \mathcal { L } } { \partial \mathbf { s } ^ { * } }$ represents the gradient of the loss with respect to the joint equilibrium state. A significant advantage of using IFT is that it enables backpropagation using only the converged state $\mathbf { s } ^ { * }$ , without needing to store intermediate solver iterates. However, since the DEQ module does not consist of discrete, explicitly unrolled layers, applying loss solely to the final converged state $\mathbf { s } ^ { * }$ might not suficiently supervise the model to ensure convergence to a desirable equilibrium.

To address this and better guide the learning process towards meaningful equilibrium, inspired by DEQ-Flow [2], we introduce an auxiliary loss term computed on randomly selected intermediate disparity estimates. Our loss term is defined as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { e q } } + \mathcal { L } _ { \mathrm { i n t } } = | | \mathbf { d } ^ { * } - \mathbf { d } _ { g t } | | _ { 2 } ^ { 2 } + \gamma | | \mathbf { d } ^ { [ i ] } - \mathbf { d } _ { g t } | | _ { 2 } ^ { 2 } ,\tag{11}
$$

where $\mathbf { d } _ { g t }$ is the ground truth disparity, $\mathbf { d } ^ { * }$ is the converged disparity, $\mathbf { d } ^ { [ i ] }$ is the disparity predicted from a randomly chosen intermediate hidden state during the solver’s iterations, and $\gamma < 1$ is a hyperparameter weighting the intermediate loss. This combined loss encourages both the final converged state to be accurate and the iterative refinement process itself to progress towards useful solutions.

Table 1: Quantitative results on TartanAir. The best and second-best are bolded and underlined.
<table><tr><td rowspan="2">Method</td><td colspan="3">OCC</td><td colspan="3">ALL</td></tr><tr><td>|EPE Bad 3.0 Bad 1.0|EPE Bad 3.0 Bad 1.0 (px) (%)</td><td></td><td>(%)</td><td>(px)</td><td>(%)</td><td>(%)</td></tr><tr><td>CODD</td><td>-</td><td>1</td><td>=</td><td>0.90</td><td>5.30</td><td>=</td></tr><tr><td>TemporalStereo</td><td>1.60</td><td>18.46</td><td>39.76</td><td>0.66</td><td>3.81</td><td>11.18</td></tr><tr><td>TC-Stereo</td><td>1.36</td><td>16.36</td><td>34.68</td><td>0.52</td><td>3.31</td><td>8.51</td></tr><tr><td>DEQ-Stereo (Ours)|0.77</td><td></td><td>4.48</td><td>11.20</td><td>|0.39</td><td>1.88</td><td>5.12</td></tr></table>

Table 2: Quantitative results on SceneFlow.
<table><tr><td>Method</td><td>EPE Bad 3.0 (px)</td><td>(%)</td></tr><tr><td>CODD TemporalStereo TC-Stereo</td><td>0.60 0.53 0.50</td><td>2.80 2.75 2.39</td></tr><tr><td colspan="3">DEQ-Stereo (Ours)|0.45 2.37</td></tr></table>

## 5 Experiments

## 5.1 Implementation Details

Training Setup. We implement DEQ-Stereo in PyTorch, training on eight NVIDIA TITAN RTX GPUs and evaluating on a single RTX 3090. We use AdamW [28] with gradients clipped to [−1, 1]. For pretraining, we separately train on SceneFlow [29] and the TartanAir TemporalStereo split [36,46] for 200K steps with a one-cycle learning rate schedule with a learning rate of $2 \times 1 0 ^ { - 4 }$ and a total batch size of 8. For KITTI fine-tuning, we initialize from the TartanAir model and fine-tune on KITTI raw [13] using LEAStereo [11] pseudo-labels, following TC-Stereo [42]. We train for 50K steps with a maximum learning rate of $1 \times 1 0 ^ { - 4 }$ . The TartanAir model is directly used for zero-shot evaluation on ETH3D SLAM [33]. All training is performed on a per-frame basis, while the video mode during inference is determined by whether TST is enabled.

DEQ Solver & Inference. The DEQ solver uses Broyden’s method with Armijo backtracking for stable convergence, with an Armijo constant of $1 \times 1 0 ^ { - 4 }$ initial step size of 0.5, and minimum step size of $1 \times 1 0 ^ { - 2 }$ . It terminates when $\operatorname* { m a x } ( | | g _ { \boldsymbol { \theta } } ( \mathbf { s } , \mathbf { C } ) | | )$ falls below $1 \times 1 0 ^ { - 3 }$ during training or 30 during inference, or after 16 iterations. Without TST, each frame is initialized from scratch; with TST, the converged state is propagated to the next frame to exploit temporal coherence. All other solver settings remain identical between the two modes.

## 5.2 Comparison with State-of-the-Art Methods

We evaluate our method against state-of-the-art methods on the TartanAir, SceneFlow, and KITTI2015 datasets. For a fair comparison, video-based baselines are restricted to online methods, excluding ofline ones that exploit future frames; representative per-frame methods are also reported for reference. We evaluate our method on the TartanAir using the train-test split provided by TemporalStereo to compare against other online video methods reported in Tab. 1. Our approach exhibits its most significant advantage in occluded regions, decisively outperforming online video methods that falter in these areas due to their reliance on unreliable temporal warping. This highlights the robustness of our method, which efectively handles ill-posed areas, even in the presence of misleading or missing temporal priors.

![](images/57ff10c2ae94106cb89684a11106184c519807ede5a7a2d0fbe596232e9ba37d.jpg)  
Fig. 4: Visualizations of disparity results on SceneFlow dataset. Compared to CODD, TC-Stereo (video-based) and IGEV-Stereo (per-frame), our method shows more temporally consistent disparity predictions (red box) and better preserves object boundaries in ill-posed regions where temporal cues are unavailable (black box).

Table 3: Quantitative results on KITTI2015. A method<sup>†</sup> requires camera parameters as auxiliary input. Note that the TST cannot be applied to per-frame methods.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="3">NOC</td><td colspan="3">ALL</td><td rowspan="2">Time (s)</td></tr><tr><td>D1-bg (%)</td><td>D1-fg (%)</td><td>D1-all (%)</td><td>D1-bg (%)</td><td>D1-fg (%)</td><td>D1-all (%)</td></tr><tr><td rowspan="5">Per frame</td><td>RAFT-Stereo</td><td>1.45</td><td>2.94</td><td>1.69</td><td>1.58</td><td>3.05</td><td>1.82</td><td>0.30</td></tr><tr><td>IGEV-Stereo</td><td>1.27</td><td>2.62</td><td>1.53</td><td>1.38</td><td>2.67</td><td>1.59</td><td>0.34</td></tr><tr><td>Selective-IGEV</td><td>1.22</td><td>2.55</td><td>1.44</td><td>1.33</td><td>2.61</td><td>1.55</td><td>0.45</td></tr><tr><td>MoCha-Stereo</td><td>1.24</td><td>2.42</td><td>1.44</td><td>1.36</td><td>2.43</td><td>1.53</td><td>0.50</td></tr><tr><td>DEQ-Stereo (Ours) w/o TST</td><td>1.20</td><td>2.47</td><td>1.41</td><td>1.31</td><td>2.49</td><td>1.51</td><td>0.16</td></tr><tr><td rowspan="3">Video</td><td rowspan="2">TemporalStereo† TC-Stereo†</td><td>1.52</td><td>2.58</td><td>1.70</td><td>1.61</td><td>2.78</td><td>1.81</td><td>0.05</td></tr><tr><td>1.21</td><td>2.24</td><td>1.38</td><td>1.29</td><td>2.33</td><td>1.46</td><td>0.11</td></tr><tr><td>DEQ-Stereo (Ours)</td><td>1.21</td><td>2.52</td><td>1.43</td><td>1.33</td><td>2.55</td><td>1.53</td><td>0.10</td></tr></table>

The results on SceneFlow in Tab. 2 demonstrate that our method achieves state-of-the-art performance among online video stereo approaches, attaining an EPE of 0.45 and outperforming TC-Stereo by 10%. Qualitative comparisons in Fig. 4 further highlight the advantages of our approach: it generates temporally consistent depth maps across video frames, whereas conventional methods such as TC-Stereo and IGEV-Stereo struggle to maintain consistency.

To evaluate performance under real-world conditions, we assess DEQ-Stereo on the KITTI2015 dataset, as shown in Tab. 3. Without employing additional modules for high-frequency detail preservation, our method achieves higher accuracy than most existing approaches [9,37,42] in background and overall metrics, while showing a moderate decline in the foreground region. In the per-frame mode, DEQ-Stereo maintains strong accuracy and operates 3.12× faster than MoCha-Stereo [9], demonstrating an improved balance between accuracy and eficiency. In the video mode, our method achieves accuracy comparable to TC-Stereo in background regions, with only minor degradation in foreground and overall metrics. This performance is obtained without auxiliary camera parameters or complex temporal warping, while running 10% faster, underscoring the efectiveness of our temporal modeling within the DEQ framework.

![](images/0cbbf2250ceb4ae85903be9b28d668d08984b07256b10f6a5ab864d0d31934b8.jpg)  
Fig. 5: Visualization of disparity results on ETH3D SLAM dataset. DEQ-Stereo produces accurate and stable disparities across diverse real-world scenes, demonstrating robustness to photometric variations and challenging scene structures.

Table 4: Generalization Performance on ETH3D SLAM.  
Table 5: Generalization Performance on Spring.
<table><tr><td>Method</td><td colspan="3">|EPE (px) Bad 3.0 (%) Bad 1.0 (%)</td></tr><tr><td>CODD</td><td>7.78</td><td>53.02</td><td>33.19</td></tr><tr><td>TC-Stereo</td><td>7.17</td><td>30.63</td><td>14.70</td></tr><tr><td>DEQ-Stereo (Ours)|</td><td>3.18</td><td>34.85</td><td>14.80</td></tr></table>

<table><tr><td>Method</td><td colspan="3">|EPE (px) Bad 3.0 (%) Bad 1.0 (%)</td></tr><tr><td rowspan="3">TemporalStereo CODD</td><td>31.36</td><td>31.40</td><td>38.20</td></tr><tr><td>6.82</td><td>8.18</td><td>14.70</td></tr><tr><td>9.91</td><td>5.45</td><td>11.92</td></tr><tr><td>DEQ-Stereo (Ours)</td><td>2.78</td><td>5.74</td><td>15.20</td></tr></table>

## 5.3 Zero-Shot Generalization to Real-World

We evaluate the real-world generalization of our method by testing a TartanAirpretrained model on the ETH3D SLAM dataset, as presented in Tab. 4 and Fig. 5. Compared to TC-Stereo, DEQ-Stereo achieves a substantially lower EPE, with a 2.25× improvement, indicating that large disparity errors are significantly reduced. While TC-Stereo reports slightly lower Bad 3.0 values in some frames, these metrics are sensitive to the distribution of small errors and do not fully capture the presence of large outliers. In contrast, the consistently low EPE of DEQ-Stereo demonstrates that it produces more reliable disparity estimates across all valid pixels, reducing the likelihood of severe mismatches near challenging regions such as object boundaries and textureless areas. This analysis confirms that our method not only generalizes well to unseen real-world scenes but also maintains stability and robustness in dificult stereo conditions.

## 5.4 Zero-Shot Generalization to Unseen Domains

We evaluate the generalization capability of models pretrained on SceneFlow [29] on the high-resolution (1920×1080) Spring [30] dataset (Tab. 5). While many online methods sacrifice generalization performance by reducing feature channels or limiting iterations to ensure real-time inference, our method maintains the same feature capacity as IGEV-Stereo [40] while utilizing the DEQ framework to eliminate redundant computation. Consequently, our model handles complex scene dynamics more efectively than alternative architectures. For instance, TemporalStereo [46] struggles on Spring because its 3D CNN-based aggregation fails to capture global context at high resolutions, whereas our recurrent-unit approach naturally expands the receptive field via iteration. As a result, our method achieves a 3.56× improvement in EPE over TC-Stereo [42] and a 2.45× improvement over the second-best method, CODD [25]. While Bad 3.0 and Bad 1.0 metrics remain competitive, this substantial EPE drop indicates that our approach significantly minimizes large disparity errors.

Table 6: Ablation study for DEQ-Stereo on TartanAir.
<table><tr><td>Method</td><td>DEQ ConvSGU TST |solver</td><td>(px)</td><td>|EPE Bad 3.0 Time (%)</td><td>(s)</td></tr><tr><td>Baseline</td><td></td><td></td><td>|0.40 1.95</td><td>0.253</td></tr><tr><td>w/o DEQ</td><td></td><td>√</td><td>0.45 2.22</td><td>0.246</td></tr><tr><td>A</td><td>√</td><td></td><td>0.41 2.04</td><td>0.287</td></tr><tr><td>B</td><td>√</td><td>√</td><td>0.49 2.39</td><td>0.192</td></tr><tr><td>C</td><td>√</td><td></td><td>0.38 1.87</td><td>0.190</td></tr><tr><td>Full Model|</td><td>√</td><td></td><td>0.39</td><td>1.88 0.125</td></tr></table>

Table 7: Comparison with foundation model on KITTI2015. A method<sup>†</sup> is finetuned on the KITTI Eigen split, while others are trained only on the SceneFlow dataset.
<table><tr><td rowspan="2">Method</td><td>Bad 1.0</td><td></td><td></td><td>[EPE| Time Params</td></tr><tr><td>|NOC OCC ALL|</td><td>|(px)|</td><td>(s)</td><td>(M)</td></tr><tr><td>StereoAnywhere</td><td>22.00 59.12 22.63|</td><td>0.97</td><td>1.619</td><td>346.75</td></tr><tr><td>MonSter</td><td>19.83 346.34 20.30</td><td>0.88</td><td>0.673</td><td>388.69</td></tr><tr><td>DEFOM-Stereo</td><td>23.82 57.05 24.39</td><td>1.06</td><td>0.530</td><td>382.62</td></tr><tr><td>DEQ-Stereo (Ours)</td><td>25.6652.78 26.15</td><td>1.33</td><td>0.216</td><td>12.60</td></tr><tr><td colspan="3">DEQ-Stereo† (Ours)| |2.57 12.88 2.73|</td><td>0.27|0.168</td><td>12.60</td></tr></table>

## 5.5 Ablation study for DEQ-Stereo

Deep Equilibrium Solver. Tab. 6 summarizes our ablation study conducted on the TartanAir dataset. A naive DEQ plug-in leads to degraded accuracy and increased runtime. As illustrated in Fig. 6, this failure arises from the ConvGRU’s unstable convergence, which struggles to reach a fixed point under the complex stereo dynamics. This behavior resembles the convergence issues reported in DEQ-Flow [2], underscoring the need for mechanisms that ensure faster and more stable equilibrium. Comparing our full model with a recurrent variant without the DEQ solver further supports this: removing the solver increases the EPE by 15.4% (0.39 → 0.45), while our full model achieves higher accuracy at nearly twice the speed. The recurrent variant sufers from initialization sensitivity due to reduced non-linearity and direct reuse of hidden states in TST. In contrast, our DEQ formulation treats inference as an optimization process that iteratively refines the hidden state toward a stable equilibrium, mitigating initialization issues and improving both accuracy and eficiency.

Convolutional Simplified Gated Unit. Replacing ConvGRU with our simplified ConvSGU results in noticeably diferent convergence behavior. As shown in Fig. 6, this stems from reduced non-linearity, which enables smoother dynamics and convergence to a deeper equilibrium state where the residual function’s norm is closer to zero. This deeper convergence is mathematically akin to performing more efective refinement iterations, explaining the higher accuracy and significantly faster inference reported in Tab. 6. Further analysis of gating mechanisms and their design choices is included in the supplementary material. Temporal State Transfer. Initializing the solver with the previous frame’s equilibrium state leads to a substantial acceleration in inference, reducing runtime by 34% in Tab. 6, while incurring only a minor drop in accuracy, around 0.5% in Bad 3.0, under challenging conditions. This trade-of makes it highly suitable for real-time video applications.

![](images/815b90344a951780375be89591f01a755869e695dab0d68fbee910f56a8f5afc.jpg)

![](images/6056e36cb2ec2bb445757bf74cf91e7e82e65884d5b372fb21283a3a5fb36ecd.jpg)  
Fig. 6: Convergence comparison across ablation variants. This graph shows the maximum norm of g<sub>θ</sub>(·) across DEQ solver iterations on TartanAir. Labels correspond to those in Tab. 6.  
Fig. 7: Comparison of Methods for Using Temporal Information. This compares the convergence of Fixedpoint Reuse from DEQ-Flow with our TST on TartanAir.

## 5.6 Comparison with Foundation Models

Recent works [4,10,17,38] have actively explored integrating the complementary strengths of stereo matching and monocular depth estimation by leveraging foundation depth estimation networks [41]. To fairly compare with these foundationbased approaches, we evaluate DEQ-Stereo on KITTI2015 dataset and analyze its practicality and eficiency. All models are trained on SceneFlow and evaluated on KITTI2015 in a zero-shot manner, except for DEQ-Stereo<sup>†</sup>, which is fine-tuned on KITTI Eigen split. As shown in Tab. 7, while the foundationbased methods demonstrate strong cross-domain generalization, they require substantially higher computation. In contrast, DEQ-Stereo attains competitive zero-shot accuracy while running 2.5–7.5× faster with 27–31× fewer parameters, highlighting its superior eficiency. Furthermore, when fine-tuned (DEQ-Stereo<sup>†</sup>), both accuracy and inference speed further improve, indicating that our adaptive equilibrium formulation efectively benefits from task-specific adaptation.

## 5.7 Comparative Analysis with DEQ-Flow

To validate that our TST better promotes convergence under the DEQ paradigm compared to the Fixed-point Reuse proposed in DEQ-Flow, we compare their convergence behaviors across iterations in Tab. 8 and Fig. 7. While Fixed-point Reuse provides strong temporal cues, propagating a final prediction with its inherent sharp discontinuities creates a rigid prior. This prior can become temporally misaligned in dynamic scenes, leading to unstable convergence. As shown in Tab. 8, this instability results in a notable degradation in performance compared to the case without temporal cues, indicating that Fixed-point Reuse can hinder convergence rather than accelerate it. In contrast, TST achieves more stable and faster convergence by propagating only the converged hidden state h<sup>∗</sup>. This enables the model to reach deeper fixed points more eficiently, maintaining consistent accuracy across iterations without performance degradation. By avoiding the reuse of potentially unreliable explicit predictions and instead leveraging an abstract geometric context as a warm-start, TST enhances both robustness and eficiency within the DEQ framework.

Table 8: Comparison of TST and Fixed-point Reuse on TartanAir.
<table><tr><td>Iterations</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>Adaptive</td></tr><tr><td>w/o Temporal Cue</td><td>0.64</td><td>0.53</td><td>0.47</td><td>0.42</td><td>0.39</td><td>0.39</td></tr><tr><td>Fixed-point Reuse</td><td>13.39</td><td>9.18</td><td>6.79</td><td>5.77</td><td>4.94</td><td>6.70</td></tr><tr><td>TST</td><td>0.59</td><td>0.52</td><td>0.47</td><td>0.42</td><td>0.40</td><td>0.39</td></tr></table>

## 6 Conclusion

We introduce DEQ-Stereo, an eficient iterative stereo matching framework rooted in the Deep Equilibrium (DEQ) paradigm. By replacing traditional fixed-iteration update loops with an implicit fixed-point formulation, our architecture adaptively balances accuracy and computational eficiency. Specifically, the proposed ConvSGU accelerates and stabilizes per-frame convergence by eliminating unnecessary non-linearities, while the TST mechanism enables efective cross-frame warm-starting. Together, these components allow DEQ-Stereo to achieve stateof-the-art or highly competitive performance across multiple challenging benchmarks, demonstrating strong robustness in occluded and dynamic regions without relying on explicit, computationally expensive temporal warping.

## Acknowledgments

This work was supported by Korea Research Institute for defense Technology planning and advancement through Defense Innovation Vanguard Enterprise Project, funded by Defense Acquisition Program Administration (R230206), the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2026-25480865), LG AI STAR Talent Development Program for Leading Large-Scale Generative AI Models in the Physical AI Domain (RS-2025-25442149) and the Advanced GPU Utilization Support Program funded by the Government of the Republic of Korea (Ministry of Science and ICT) (02-26-01-0057).

## References

1. Armijo, L.: Minimization of functions having lipschitz continuous first partial derivatives. Pacific Journal of mathematics 16(1), 1–3 (1966) 5

2. Bai, S., Geng, Z., Savani, Y., Kolter, J.Z.: Deep equilibrium optical flow estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 620–630 (2022) 4, 5, 8, 12

3. Bai, S., Kolter, J.Z., Koltun, V.: Deep equilibrium models. Advances in neural information processing systems 32 (2019) 2, 4, 5, 7, 8

4. Bartolomei, L., Tosi, F., Poggi, M., Mattoccia, S.: Stereo anywhere: Robust zeroshot deep stereo matching even where either stereo or mono fail. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1013–1027 (2025) 13

5. Cabon, Y., Stofl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1050–1060 (2025) 1

6. Cao, J., Shi, Y., Zhang, K., Zhang, Y., Timofte, R., Van Gool, L.: Deep equilibrium difusion restoration with parallel sampling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2824–2834 (2024) 4

7. Chang, J.R., Chen, Y.S.: Pyramid stereo matching network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5410–5418 (2018) 3

8. Chen, R.T., Rubanova, Y., Bettencourt, J., Duvenaud, D.K.: Neural ordinary differential equations. Advances in neural information processing systems 31 (2018) 4

9. Chen, Z., Long, W., Yao, H., Zhang, Y., Wang, B., Qin, Y., Wu, J.: Mochastereo: Motif channel attention network for stereo matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 27768– 27777 (2024) 1, 3, 10

10. Cheng, J., Liu, L., Xu, G., Wang, X., Zhang, Z., Deng, Y., Zang, J., Chen, Y., Cai, Z., Yang, X.: Monster: Marry monodepth to stereo unleashes power. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6273–6282. IEEE (2025) 13

11. Cheng, X., Zhong, Y., Harandi, M., Dai, Y., Chang, X., Li, H., Drummond, T., Ge, Z.: Hierarchical neural architecture search for deep stereo matching. Advances in neural information processing systems 33, 22158–22169 (2020) 9

12. Cheng, Z., Yang, J., Li, H.: Stereo matching in time: 100+ fps video stereo matching for extended reality. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 8719–8728 (2024) 1, 3

13. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The kitt dataset. International Journal of Robotics Research (IJRR) (2013) 9

14. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? the kitti vision benchmark suite. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 3354–3361. IEEE (2012) 1

15. Gu, F., Chang, H., Zhu, W., Sojoudi, S., El Ghaoui, L.: Implicit graph neural networks. Advances in neural information processing systems 33, 11984–11995 (2020) 4

16. Guo, X., Yang, K., Yang, W., Wang, X., Li, H.: Group-wise correlation stereo network. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3273–3282 (2019) 3

17. Jiang, H., Lou, Z., Ding, L., Xu, R., Tan, M., Jiang, W., Huang, R.: Defom-stereo: Depth foundation model based stereo matching. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21857–21867. IEEE (2025) 13

18. Jing, J., Luo, W., Mao, Y., Mikolajczyk, K.: Stereo any video: Temporally consistent stereo matching. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 20836–20846. IEEE (2025) 3

19. Jing, J., Mao, Y., Mikolajczyk, K.: Match-stereo-videos: Bidirectional alignment for consistent dynamic stereo matching. In: European Conference on Computer Vision. pp. 415–432. Springer (2024) 2, 3

20. Karaev, N., Rocco, I., Graham, B., Neverova, N., Vedaldi, A., Rupprecht, C.: Dynamicstereo: Consistent dynamic depth from stereo videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13229– 13239 (2023) 2, 3

21. Kendall, A., Martirosyan, H., Dasgupta, S., Henry, P., Kennedy, R., Bachrach, A., Bry, A.: End-to-end learning of geometry and context for deep stereo regression. In: Proceedings of the IEEE international conference on computer vision. pp. 66–75 (2017) 3

22. Krantz, S.G., Parks, H.R.: The implicit function theorem: history, theory, and applications. Springer Science & Business Media (2002) 8

23. Lee, K., Levy, O., Zettlemoyer, L.: Recurrent additive networks (2017), https: //arxiv.org/abs/1705.07393 2, 7

24. Li, J., Wang, P., Xiong, P., Cai, T., Yan, Z., Yang, L., Liu, J., Fan, H., Liu, S.: Practical stereo matching via cascaded recurrent network with adaptive correlation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16263–16272 (2022) 1, 3, 4

25. Li, Z., Ye, W., Wang, D., Creighton, F.X., Taylor, R.H., Venkatesh, G., Unberath, M.: Temporally consistent online depth estimation in dynamic scenes. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 3018–3027 (2023) 2, 3, 12

26. Lipson, L., Teed, Z., Deng, J.: Raft-stereo: Multilevel recurrent field transforms for stereo matching. In: 2021 International Conference on 3D Vision (3DV). pp. 218–227. IEEE (2021) 1, 3, 4

27. Liu, B., Yu, H., Long, Y.: Local similarity pattern and cost self-reassembling for deep stereo matching networks. In: Proceedings of the AAAI conference on artificial intelligence. vol. 36, pp. 1647–1655 (2022) 3

28. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017) 9

29. Mayer, N., Ilg, E., Hausser, P., Fischer, P., Cremers, D., Dosovitskiy, A., Brox, T.: A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4040–4048 (2016) 3, 9, 11

30. Mehl, L., Schmalfuss, J., Jahedi, A., Nalivayko, Y., Bruhn, A.: Spring: A highresolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4981–4991. IEEE (2023) 11

31. Menze, M., Geiger, A.: Object scene flow for autonomous vehicles. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3061–3070 (2015) 1, 3

32. Pokle, A., Geng, Z., Kolter, J.Z.: Deep equilibrium approaches to difusion models. Advances in Neural Information Processing Systems 35, 37975–37990 (2022) 4

33. Schops, T., Sattler, T., Pollefeys, M.: Bad slam: Bundle adjusted direct rgb-d slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 134–144 (2019) 3, 9

34. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419. Springer (2020) 3, 4

35. Wang, S., Teng, Y., Wang, L.: Deep equilibrium object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6296–6306 (2023) 4

36. Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., Scherer, S.: Tartanair: A dataset to push the limits of visual slam. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 4909–4916. IEEE (2020) 3, 9

37. Wang, X., Xu, G., Jia, H., Yang, X.: Selective-stereo: Adaptive frequency information selection for stereo matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19701–19710 (2024) 1, 3, 10

38. Wen, B., Trepte, M., Aribido, J., Kautz, J., Gallo, O., Birchfield, S.: Foundationstereo: Zero-shot stereo matching. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5249–5260. IEEE (2025) 13

39. Xu, G., Cheng, J., Guo, P., Yang, X.: Attention concatenation volume for accurate and eficient stereo matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12981–12990 (2022) 3

40. Xu, G., Wang, X., Ding, X., Yang, X.: Iterative geometry encoding volume for stereo matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21919–21928 (2023) 1, 3, 4, 6, 12

41. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. Advances in Neural Information Processing Systems 37, 21875–21911 (2024) 13

42. Zeng, J., Yao, C., Wu, Y., Jia, Y.: Temporally consistent stereo matching. In: European Conference on Computer Vision. pp. 341–359. Springer (2024) 2, 3, 9, 10, 12

43. Zeng, J., Yao, C., Yu, L., Wu, Y., Jia, Y.: Parameterized cost volume for stereo matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18347–18357 (2023) 3

44. Zhang, F., Prisacariu, V., Yang, R., Torr, P.H.: Ga-net: Guided aggregation net for end-to-end stereo matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 185–194 (2019) 3

45. Zhang, G., Lee, J.H., Lim, J., Suh, I.H.: Building a 3-d line-based map using stereo slam. IEEE Transactions on Robotics 31(6), 1364–1377 (2015) 1

46. Zhang, Y., Poggi, M., Mattoccia, S.: Temporalstereo: Eficient spatial-temporal stereo matching network. In: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 9528–9535. IEEE (2023) 2, 9, 12

47. Zhao, H., Zhou, H., Zhang, Y., Chen, J., Yang, Y., Zhao, Y.: High-frequency stereo matching network. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1327–1336 (2023) 1, 3

48. Zhong, Y., Li, H., Dai, Y.: Open-world stereo video matching with deep rnn. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 101– 116 (2018) 3