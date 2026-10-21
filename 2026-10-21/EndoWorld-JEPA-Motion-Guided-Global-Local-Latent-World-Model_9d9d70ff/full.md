# EndoWorld-JEPA: Motion-Guided Global–Local Latent World Modeling for Endoscopic Video Representation Learning

Jie Wang<sup>1</sup>, Shilong Yang<sup>2</sup>, Tao Zhou<sup>3</sup>, Juan Yu<sup>4</sup>, Rong Qu<sup>1</sup>, Jianfeng Ren<sup>1</sup>, Zhicheng Zhang<sup>5</sup>, and Linlin Shen<sup>1,2</sup>

<sup>1</sup>University of Nottingham <sup>2</sup>Shenzhen University <sup>3</sup>Tsinghua University <sup>4</sup>The First Affiliated Hospital of Shenzhen University <sup>5</sup>JancsiLab

## Abstract

Endoscopic video contains state transitions induced by camera displacement, deformable tissue, and instrument interaction, yet existing pre-training methods largely treat these observed dynamics implicitly. Inspired by action-conditioned joint-embedding predictive world models and by the observation that endoscopic optical flow contains both shared imageplane displacement and localized residual motion, we reformulate passive endoscopic video pre-training as causal motionguided latent-state prediction. We present EndoWorld-JEPA, to our knowledge the first latent world-model pre-training framework designed for general-purpose endoscopic video representation learning. In place of recorded actions, calibrated poses, or robot kinematics, the model conditions prediction of the next latent state on motion observed up to the current state. Its Optical Motion-guided module separates dominant displacement from residual motion and converts the resulting evidence into an ordered temporal motion condition and spatial local-motion weights. Complementary global and motionlocal predictors use these outputs: the temporal condition modulates both latent transitions, whereas the local weights define an auxiliary motion-local state that extends prediction from scene-level evolution to spatially selective dynamics. Patchwise motion weighting emphasizes localized change while retaining anatomical context, and past-motion-window dropout randomly withholds the complete motion condition to reduce dependence on unreliable motion. This global–local formulation learns predictable visual–motion dynamics in representation space without reconstructing stochastic appearance artifacts. Evaluation shows that a common pre-trained checkpoint achieves consistently strong transfer performance across four downstream benchmarks, supporting motion-guided global– local latent prediction as an effective pre-training paradigm for endoscopic video.

Keywords: Endoscopy, foundation model, joint-embedding predictive architecture, latent world model, optical flow, selfsupervised learning.

## 1 Introduction

Endoscopy is a dynamic imaging process in which the camera continuously changes viewpoint while soft tissue deforms, instruments interact with the scene, and illumination varies with camera–surface geometry. These factors can substantially alter appearance even when anatomy or lesion identity remains unchanged, and these dynamics are sufficiently pronounced that endoscope egomotion and deformable tissue motion are themselves explicit targets of estimation and reconstruction [1, 2]. Conversely, local temporal changes may reveal lesion boundaries, tissue deformation, or procedural events that are ambiguous from appearance alone. Because clinically relevant endoscopic tasks arise within dynamic examinations—even when annotations or evaluations are frame-wise—representation learning from endoscopic video should capture not only what is visible, but also how the observed state evolves.

Endoscopic foundation models have mainly focused on spatiotemporal invariance and temporal context. Endo-FM learns transferable representations from diverse endoscopic videos [3], and EndoFM-LV extends masked representation learning to minute-level sequences [4]. By emphasizing content that persists across time or can be recovered from surrounding observations, these approaches learn robust temporal representations. However, invariance-oriented objectives may attenuate informative change, whereas masked reconstruction models temporal transitions only indirectly; neither explicitly conditions prediction of the next latent state on preceding imageplane motion.

Latent world modeling provides a structured alternative in which observations are encoded as states and future states are predicted under an explicit transition condition. Jointembedding predictive architectures (JEPAs) perform this prediction in representation space rather than reconstructing pixels [5, 6]. This property is well matched to endoscopy: a useful representation should preserve anatomy, lesion identity, and motion-relevant structure without being required to synthesize reflections, smoke, fluid, blur, or illumination. LeWorldModel (LeWM) enables stable end-to-end next-latent-state prediction through a regularized objective [7]. Its conventional actionconditioned formulation, however, assumes robot commands, probe controls, or calibrated poses that are unavailable in most routine endoscopic recordings. Consequently, passive endoscopic recordings do not directly provide the transition variable assumed by standard state–action world models.

Passive video nevertheless offers observable motion evidence that can condition state transitions without physical controls. Dense optical flow requires only neighboring frames and captures both broad image displacement, predominantly associated with endoscope motion, and localized departures associated with tissue deformation or instruments. The former summarizes whole-view state evolution, whereas the latter identifies spatial regions whose dynamic evidence may be diluted by a global frame representation. Directly using raw flow magnitude as a single condition would entangle these roles and amplify errors caused by non-Lambertian surfaces, specularities, occlusion, or rapid motion. Dominant–residual decomposition allows the same motion evidence to support two complementary representations: an ordered temporal descriptor that jointly summarizes dominant displacement, flow dispersion, and residual statistics provides an observation-derived condition for latent-state transitions, while the residual-magnitude map serves as a spatial cue for locally dynamic content. Optical motion therefore supplies a transition condition and a local relevance cue, whereas JEPA learns their predictable consequences directly in representation space without reconstructing every photometric fluctuation.

![](images/ce55a21f527376cc11e6818f8497481f577aaa273c4cff8f79fde33a5ee1fc96.jpg)  
Figure 1: Comparison of three pre-training paradigms and their treatment of local dynamics. (A) Existing endoscopic video pre-training captures temporal context, but transitions remain implicit: invariance may attenuate change, masking models transitions indirectly, and observed motion does not condition prediction. (B) Medical world models rely on domain-specific data and transition conditions unavailable in passive endoscopy. (C) EndoWorld-JEPA couples an Optical Motion-guided (OM) block that derives motion guidance from unlabeled passive video with a causal global–local JEPA, requiring no recorded controls or calibrated poses. Bottom: a schematic comparison illustrates that existing video pre-training models localized dynamics only implicitly within global latent features, whereas the OM block produces an explicit local-motion map whose patch weighting emphasizes motion-relevant regions.

Motivated by this complementarity, we propose EndoWorld-JEPA, an endoscopy-oriented extension of LeWM for motionconditioned latent-state prediction from passive recordings under a causal temporal ordering. Here, causal refers to input availability: each prediction uses only the current visual state and motion observed before the target state. Its Optical Motionguided (OM) module converts observed flow into a temporal motion condition and a spatial local-motion magnitude map. The temporal condition parameterizes both global and motionlocal latent transitions, whereas the spatial map defines an auxiliary motion-local state, enabling complementary predictors to model scene-level evolution together with spatially selective change. Patch-wise mixed weighting retains anatomical context, and past-motion-window dropout reduces reliance on unreliable flow. We pre-train a single checkpoint on unlabeled endoscopic video and evaluate its transfer on four downstream tasks—video classification, polyp segmentation, polyp detection, and surgical phase recognition—using the corresponding PolypDiag, CVC-12k, KUMC, and Cholec80 benchmarks, which respectively assess appearance recognition, dense delineation, object localization, and procedural progression. We further define a dependency-aware cross-task ablation protocol that separates the individual and joint contributions of the proposed components. The main contributions are:

• We formulate passive endoscopic video pre-training as motion-guided latent world modeling under a causal temporal ordering, using motion observed up to the current state to predict the next latent state in representation space.

• We introduce an Optical Motion-guided module that decomposes dense flow into ordered temporal descriptors— combining dominant displacement, flow dispersion, and residual-motion statistics—that form an observationderived temporal condition, together with a residual localmotion magnitude map, providing transition conditioning and spatial guidance without recorded controls or calibrated poses.

• We introduce complementary global and motion-local JEPA objectives, together with motion–uniform mixed pooling in patch-wise motion weighting and past-motionwindow dropout, to preserve scene context, emphasize localized dynamics, and reduce sensitivity to flow uncertainty.

## 2 Related Work

## 2.1 Medical and Endoscopic Foundation Models

Deep learning has advanced medical image analysis across diagnosis, prognosis, and segmentation [8–10]. Building on these task-specific advances, medical foundation models use largescale pre-training to acquire representations that can be adapted across tasks and datasets. Modality-specific approaches further incorporate structural priors into their objectives. EchoFM combines consistent spatiotemporal masking with periodic contrastive learning for echocardiography [11]. CheXFound integrates global image features with multi-scale local radiographic features, and DCXFM combines text, image-level labels, and segmentation masks to support dense prediction [12, 13]. In endoscopy, Endo-FM learns spatiotemporal invariance from large video collections [3], whereas EndoFM-LV uses minutelevel sequences to model long-range examination context [4]. EndoDINO scales image-level self-distillation over curated frames sampled from endoscopic videos and transfers frozen encoders across recognition, segmentation, and disease-scoring tasks [14]. GastroNet-5M establishes large-scale multicenter domain-specific pre-training and evaluates its accuracy, data efficiency, and robustness across a broad range of gastrointestinal applications [15]. EndoMamba combines bidirectional spatial modeling, causal temporal state-space modeling, and hierarchical reconstruction–alignment supervision for efficient online video inference [16]. EndoWorld-JEPA instead conditions prediction of the next latent state on previously observed shortrange motion and uses the same motion evidence for spatially selective prediction.

## 2.2 JEPA and Medical World Modeling

I-JEPA learns image representations by predicting embeddings of masked regions, avoiding pixel-level reconstruction [5]. V-JEPA extends feature prediction to video, and V-JEPA 2 scales action-free video pre-training before post-training an actionconditioned predictor for robotic planning [6, 17]. LeWM formulates stable end-to-end next-latent-state prediction with a Gaussian latent regularizer [7]. Medical world models define states and transitions according to domain-specific structure. CheXWorld models local anatomy, global layout, and radiographic domain transformations [18]. Medical World Model predicts disease-state changes under treatment policies [19], while Cardiac Copilot learns probe-conditioned transitions between echocardiographic planes [20]. EndoWorld-JEPA focuses on one-step latent-state prediction for representation learning from passive endoscopy, deriving an interpretable transition condition from preceding image-plane motion rather than recorded control signals. Figure 1 contrasts this formulation with video-context endoscopic pre-training and with medical world models whose transitions are defined by modality- or task-specific signals.

## 2.3 Motion in Endoscopic Video

Endoscopic motion couples camera movement, tissue deformation, respiration, and instrument interaction. Optical flow has been used to estimate colonoscope egomotion from live video [1] and to support camera-pose estimation [21]. Videoderived geometric supervision also enables single-frame endoscopic depth estimation [22], while foundation-model-guided Gaussian splatting reconstructs deformable tissue from endoscopic sequences [2]. These methods use motion or geometry for navigation, depth estimation, or reconstruction. In contrast, EndoWorld-JEPA uses dense flow as weak supervision for latent representation learning by separating dominant image displacement from localized residual motion.

## 3 Method

The proposed framework comprises three stages (Fig. 2): OM feature construction, global–local latent pre-training, and downstream transfer. OM derives a temporal condition and a local-motion magnitude map; patch-wise motion weighting maps the latter to weights for local-state construction. A shared visual encoder forms the current global and motion-local states, and two conditioned predictors estimate their next states. Predictor inputs are restricted to the current visual state and preceding motion; the next sampled frame supplies only latent targets. Downstream models retain the visual and motion encoders, whereas the fixed spatial weighting pathway is pretraining-only.

## 3.1 World-Model Pre-Training Formulation

Given an unlabeled endoscopic sequence $\{ x _ { n } \} _ { n = 0 } ^ { T - 1 }$ of $T$ decoded observations, let n index their order and let t index sampled latent states. We deliberately use a shared observation horizon K for motion aggregation and latent-state prediction. The K incoming adjacent-observation transitions ending at $\boldsymbol { x } _ { s _ { t } }$ form the causal motion window, while the prediction target is $x _ { s _ { t } + K } ;$ equivalently, $s _ { t + 1 } = s _ { t } + K$ . Thus, motion observed over the preceding K transitions conditions prediction of the latent state $K$ observation steps ahead, aligning the temporal support of the observation-derived motion condition with the prediction interval. With $s _ { 0 }$ denoting the first sampled observation index, state t therefore corresponds to frame $\boldsymbol { x } _ { s _ { t } }$ , where $s _ { t } = s _ { 0 } + t K$ . During self-supervised optimization, the effective motion input is $\mathbf { A } _ { t } ^ { \mathrm { i n } } = \delta \mathbf { A } _ { t } .$ , where $\delta \sim$ Bernoulli(0.7) is sampled independently for each training pair; during evaluation and downstream adaptation, ${ \bf A } _ { t } ^ { \mathrm { i n } } = { \bf A } _ { t }$ . EndoWorld-JEPA predicts the next sampled latent state from the current global and motion-local states together with the effective causal motion input available at that state:

$$
( \hat { \mathbf { g } } _ { t + 1 } , \hat { \mathbf { l } } _ { t + 1 } ) = \mathcal { W } ( \mathbf { g } _ { t } , \mathbf { l } _ { t } , \mathbf { A } _ { t } ^ { \mathrm { i n } } ) ,\tag{1}
$$

where $\mathbf { g } _ { t }$ and ${ \mathbf l } _ { t }$ are the global and motion-local latent states of $x _ { s _ { t } } , \mathbf { A } _ { t }$ contains the K most recent standardized motion descriptors ending at $\boldsymbol { x } _ { s _ { t } }$ , and W is the latent world model. Hats denote predicted states. A zero value of δ withholds the complete standardized motion window while preserving the visual input. All predictor inputs are observable by $s _ { t } ; \boldsymbol { x } _ { s _ { t + 1 } }$ is used only as an embedding-space target.

This formulation separates visual latent-state encoding from conditional next-latent-state prediction in representation space. The motion history ${ \bf A } _ { t }$ is estimated from transitions preceding the current state and parameterizes the transition condition for passive recordings. The global and motion-local predictors are optimized directly in representation space.

The model contains a frame-wise visual encoder ${ \mathcal { E } } _ { \mathrm { V } }$ , the OM module with learned motion encoder ${ \mathcal { E } } _ { \mathrm { M } }$ , a global predictor $\mathcal { D } _ { \mathrm { G } }$ , and a motion-local predictor $\mathcal { D } _ { \mathrm { { L } } }$ . For each sampled state,

$$
\mathcal { E } _ { \mathrm { V } } ( x _ { s _ { t } } ) = [ \mathbf { v } _ { t } ^ { \mathrm { c l s } } , \mathbf { v } _ { t , 1 } ^ { \mathrm { p a t c h } } , \dots , \mathbf { v } _ { t , N } ^ { \mathrm { p a t c h } } ] ,\tag{2}
$$

![](images/6e358edd405152a9c9f30583794cdb7870a636490b8a3cd1592f0a3e566d1c68.jpg)

![](images/baee563a20077ac8857bf9b1dcd1ad6f9d47152e192a6ecb0d09443ec1a574c4.jpg)  
Figure 2: Overview of EndoWorld-JEPA, a causal motion-conditioned JEPA world model. Top: for the implemented $K = 5$ horizon, the OM block converts six consecutive observations ending at $x _ { s _ { t } }$ into five incoming optical flows, motion descriptors, a local-motion weight map, and a motion condition. Middle: the global–local JEPA block uses this five-transition motion history with the current state $\boldsymbol { x } _ { s _ { t } }$ to predict local and global latent targets at $x _ { s _ { t } + 5 }$ . Bottom: the pretrained encoder and OM block are transferred with task-specific decoders for four downstream tasks.

where $\mathbf { v } _ { t } ^ { \mathrm { c l s } }$ is the CLS token, $\mathbf { v } _ { t , i } ^ { \mathrm { p a t c h } }$ is patch token i, and N is the number of patch tokens. The global state is $\mathbf { g } _ { t } ~ =$ $P _ { \mathrm { G } } ^ { \mathrm { e n c } } ( \mathbf { v } _ { t } ^ { \mathrm { c l s } } )$ ), where $P _ { \mathrm { G } } ^ { \mathrm { e n c } }$ is the global projection MLP.

The shared encoder independently processes the current and target frames. Only the current embedding is passed to the predictors; the target embedding and target-side local-motion magnitude map $\mathbf { M } _ { t + 1 }$ define the supervision. The encoder- and predictor-side projection MLPs use per-sample LayerNorm to avoid cross-sample statistics and map their outputs into a common loss space. The visual encoder $\mathcal { E } _ { \mathrm { V } }$ is optimized end to end without a frozen or exponential-moving-average teacher. Downstream tasks retain ${ \mathcal { E } } _ { \mathrm { V } }$ and $\mathcal { E } _ { \mathrm { M } } ;$ the global predictor is reused only by the video-classification adapter, whereas the motion-local predictor is discarded.

## 3.2 Optical Motion-guided (OM) Module

The OM module transforms optical flow observed up to the current state into a temporal condition $\mathbf { c } _ { t }$ and a spatial local-motion magnitude map $\mathbf { M } _ { t }$ . Farneback estimation, dominant–residual decomposition, descriptor construction, map resampling, and patch-wise motion weighting are deterministic; only the temporal encoder ${ \mathcal { E } } _ { \mathrm { M } }$ is learned. Accordingly, the flow provides observation-derived weak supervision for image-plane dynamics.

For two consecutive RGB observations $x _ { n - 1 }$ and $x _ { n }$ with $n \geq 1$ , we apply field-of-view preprocessing, resize the motion input to a canonical $2 2 4 \times 2 2 4$ grid, and compute forward dense optical flow on this grid

$$
\mathbf { u } _ { n } ( \mathbf { q } ) = [ u _ { n } ^ { x } ( \mathbf { q } ) , u _ { n } ^ { y } ( \mathbf { q } ) ] ,\tag{3}
$$

Representative OM spatial responses  
![](images/eec4441e8e48973d49a4828ebb17ca2d82b9d3e865b5d809444fbde9e4b083ba.jpg)  
Figure 3: Representative OM responses to a whole-view shift (top), a local-region change (middle), and their combination (bottom). Columns show preceding and current frames, dense flow $\mathbf { u } _ { s _ { t } }$ , residual magnitude $R _ { s _ { t } }$ , local map $\mathbf { M } _ { t } ,$ relative weights $N w _ { t , i } .$ , and the weight overlay. Dense-flow hue follows the direction key, while intensity denotes displacement magnitude. Row annotations report $\| \mathbf { d } _ { s _ { t } } \| _ { 2 }$ and $P _ { 9 0 } ( R _ { s _ { t } } )$ . Residual magnitude and ${ { \bf { M } } _ { t } }$ share a within-row display range. Here, $N w _ { t , i } = 1$ denotes uniform pooling; weight maps and overlays share the clipped 0.3–7.0 scale. Display scaling does not affect OM computation.

where ${ \bf u } _ { n } ( { \bf q } ) \in \mathbb { R } ^ { 2 }$ maps pixel location q in $x _ { n - 1 }$ toward its correspondence in $x _ { n }$ . The components $u _ { n } ^ { x } ( \mathbf { q } )$ and $u _ { n } ^ { y } ( \mathbf { q } )$ measure per-transition horizontal and vertical displacement in pixels on the canonical motion grid, respectively. The index n associates $\mathbf { u } _ { n }$ with the incoming transition $x _ { n - 1 } \ \to \ x _ { n }$ Farneback estimation is deterministic, computationally efficient, and requires no additional training data [23].

Endoscope movement often induces a broad displacement shared by much of the field of view, whereas tissue deformation, instrument interaction, and independently moving content are more localized. We approximate this dominant displacement by taking the component-wise median over image pixels, which is less sensitive than the mean to sparse extreme responses, and define the residual component as

$$
\tilde { u } _ { n } ^ { x } = \mathrm { m e d i a n } _ { \bf q } u _ { n } ^ { x } ( { \bf q } ) ,\tag{4}
$$

$$
\tilde { u } _ { n } ^ { y } = \mathrm { m e d i a n } _ { \bf q } u _ { n } ^ { y } ( { \bf q } ) ,\tag{5}
$$

$$
\begin{array} { r } { \mathbf { d } _ { n } = [ \tilde { u } _ { n } ^ { x } , \tilde { u } _ { n } ^ { y } ] , } \end{array}\tag{6}
$$

$$
\mathbf { r } _ { n } ( \mathbf { q } ) = \mathbf { u } _ { n } ( \mathbf { q } ) - \mathbf { d } _ { n } ,\tag{7}
$$

$$
R _ { n } ( \mathbf { q } ) = \| \mathbf { r } _ { n } ( \mathbf { q } ) \| _ { 2 } .\tag{8}
$$

The component medians $\tilde { u } _ { n } ^ { x }$ and $\tilde { u } _ { n } ^ { y }$ form the dominant displacement $\mathbf { d } _ { n } ;$ subtracting ${ \bf d } _ { n }$ yields the residual-flow vector $\mathbf { r } _ { n } ( \mathbf { q } )$ and its Euclidean magnitude $R _ { n } ( \mathbf { q } )$ . The per-transition

raw motion descriptor is

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { a } _ { n } = [ \tilde { u } _ { n } ^ { x } , \tilde { u } _ { n } ^ { y } , \sigma ( u _ { n } ^ { x } ) , \sigma ( u _ { n } ^ { y } ) , } \\ & { \qquad \quad \mu ( R _ { n } ) , \sigma ( R _ { n } ) , P _ { 9 0 } ( R _ { n } ) , \rho _ { n } ] , } \end{array} } \end{array}\tag{9}
$$

where $\mu ( \cdot ) , \sigma ( \cdot )$ , and $P _ { 9 0 } ( \cdot )$ are the spatial mean, standard deviation, and 90th percentile over all pixels, respectively, and $\rho _ { n }$ is the fraction satisfying $R _ { n } ( \mathbf { q } ) > 0 . 5$ pixels on the canonical motion grid. The threshold identifies spatially salient residual displacement within an observed transition. The resulting $\mathbf { a } _ { n }$ is the raw 8-D descriptor in Fig. 2. With component-wise pretraining statistics $\pmb { \mu } _ { \mathrm { a } }$ and $\sigma _ { \mathrm { a } }$ computed on the same grid, its standardized form is $\mathbf { a } _ { n } ^ { \mathrm { s t d } } = ( \mathbf { a } _ { n } - { \pmb { \mu } } _ { \mathrm { a } } ) / { \pmb { \sigma } } _ { \mathrm { a } } ,$ , where the division is component-wise. At the first observation of each episode, the raw descriptor and residual map are set to zero. Missing prefix entries use the same raw-zero descriptor before standardization and therefore encode boundary padding rather than observed transitions.

The descriptor comprises two dominant-displacement, two flow-dispersion, and four residual-motion terms. For state t, the most recent K descriptors are concatenated, using the rawzero episode-prefix entries defined above:

$$
\mathbf { A } _ { t } = [ \mathbf { a } _ { s _ { t } - K + 1 } ^ { \mathrm { s t d } } ; \ldots ; \mathbf { a } _ { s _ { t } } ^ { \mathrm { s t d } } ] ,\tag{10}
$$

where semicolons denote ordered concatenation, giving a 40- dimensional window. Since $\mathbf { a } _ { n } ^ { \mathrm { s t d } }$ describes $x _ { n - 1 } \to x _ { n } ,$ an interior window spans K incoming observation transitions ending at $\boldsymbol { x } _ { s _ { t } }$ and contains no target-interval motion. Both A<sub>t</sub> and ${ { \bf { M } } _ { t } }$ are derived from apparent displacement on the canonical motion grid.

For motion-local state construction, the latest residualmagnitude field $R _ { s _ { t } } ( \mathbf { q } )$ is area-resampled to the ViT patch grid:

$$
\mathbf { M } _ { t } = \mathcal { R } _ { 1 6 \times 1 6 } ^ { \mathrm { a r e a } } \left( R _ { s _ { t } } ( \cdot ) \right) , \qquad m _ { t , i } = [ \mathbf { M } _ { t } ] _ { i } ,\tag{11}
$$

where $\mathcal { R } _ { 1 6 \times 1 6 } ^ { \mathrm { a r e a } }$ denotes area resampling and $m _ { t , i }$ is the entry at the i-th of N flattened patch locations. The temporal window ${ \bf A } _ { t }$ conditions latent-state prediction, whereas ${ { \bf { M } } _ { t } }$ is the patchgrid local-motion magnitude map that identifies current-state patches with local dynamic evidence. We apply patch-wise motion weighting to convert its entries $m _ { t , \ast }$ <sub>i</sub> into normalized patch weights:

$$
\bar { m } _ { t , i } = \log ( 1 + \operatorname* { m a x } ( m _ { t , i } , 0 ) ) ,\tag{12}
$$

$$
\gamma _ { t , i } = \frac { \bar { m } _ { t , i } - \mu ( \bar { \bf m } _ { t } ) } { \sigma ( \bar { \bf m } _ { t } ) + \epsilon } ,\tag{13}
$$

$$
w _ { t , i } = \alpha \frac { \exp ( \gamma _ { t , i } / \tau ) } { \sum _ { j } \exp ( \gamma _ { t , j } / \tau ) } + ( 1 - \alpha ) \frac { 1 } { N } .\tag{14}
$$

$\bar { m } _ { t , i }$ is the nonnegative log-compressed map value, and $\bar { \mathbf { m } } _ { t } =$ $[ \bar { m } _ { t , 1 } , \ldots , \bar { m } _ { t , N } ]$ collects all patch values. Their mean and standard deviation define the standardized patch score $\gamma _ { t , i } .$ The index j ranges over all N patches, τ is the softmax temperature, α is the motion-to-uniform mixture coefficient, and $\epsilon = 1 0 ^ { - 6 }$ ensures numerical stability. Arranged on the patch grid, $\{ w _ { t , i } \} _ { i = \cdot } ^ { N }$ forms the local-motion weight map. Mixed pooling within patch-wise motion weighting combines motionfocused softmax and uniform weights. The motion-local state is

$$
\mathbf { l } _ { t } = P _ { \mathrm { L } } ^ { \mathrm { e n c } } \left( \sum _ { i = 1 } ^ { N } w _ { t , i } \mathbf { v } _ { t , i } ^ { \mathrm { p a t c h } } \right) ,\tag{15}
$$

where $P _ { \mathrm { L } } ^ { \mathrm { e n c } }$ is the local-state projection MLP and $w _ { t , i }$ is the final normalized weight assigned to patch token $\mathbf { v } _ { t , i } ^ { \mathrm { p a t c h } }$ . The uniform component prevents the local state from concentrating entirely on isolated flow responses.

Patch-wise motion weighting limits isolated softmax peaks while retaining spatial selectivity. The temporal window ${ \bf A } _ { t }$ conditions both predictors, and the local-motion weight map $\{ w _ { t , i } \} _ { i = 1 } ^ { N }$ assigns greater weight to locally dynamic patches in the current frame. The independently computed target magnitude map $\mathbf { M } _ { t + 1 }$ is converted into $\{ w _ { t + 1 , i } \} _ { i = 1 } ^ { N }$ and contributes only to the local target. Both magnitude maps characterize image-plane motion and may respond to instruments, occlusion boundaries, specularities, and flow errors in addition to tissue motion. The lower bound $( 1 - \alpha ) / N$ on every patch weight preserves surrounding anatomical context.

Figure 3 illustrates OM responses to three representative image-plane motion patterns. Predominantly whole-view displacement produces a large median-flow magnitude and attenuated residuals after common-motion subtraction. Localized change produces a concentrated residual despite limited shared displacement, whereas mixed motion contains both responses.

In each case, M retains residual-motion magnitude, and patchwise motion weighting converts it into the local-motion weight map $\{ w _ { t , i } \} _ { i = \cdot } ^ { N }$ used to pool current-frame patch tokens. The overlays visualize this spatial emphasis while confirming retention of full-frame context.

## 3.3 Global–Local Latent State Prediction

EndoWorld-JEPA learns passive endoscopic video representations through causal motion-conditioned latent-state prediction. The global latent state $\mathbf { g } _ { t }$ represents the complete frame, whereas the motion-local latent state ${ \bf l } _ { t }$ pools patch tokens with the local-motion weights defined above.

Within the temporal branch of OM, the standardized pastmotion window ${ \bf A } _ { t }$ provides the observation-derived transition signal. The motion encoder maps its effective form to the encoded motion condition:

$$
\begin{array} { r } { \mathbf { c } _ { t } = \mathcal { E } _ { \mathrm { M } } ( \mathbf { A } _ { t } ^ { \mathrm { i n } } ) . } \end{array}\tag{16}
$$

The predictor receives only the current sampled state and its encoded causal motion condition. The global branch specializes the LeWM latent-state prediction formulation to

$$
\hat { \mathbf { g } } _ { t + 1 } = P _ { \mathrm { G } } ^ { \mathrm { o u t } } ( \mathcal { D } _ { \mathrm { G } } ( \mathbf { g } _ { t } , \mathbf { c } _ { t } ) ) .\tag{17}
$$

The motion-local branch first forms the fused state

$$
\mathbf { f } _ { t } = P _ { \mathrm { F } } \big ( \big [ \mathbf { g } _ { t } ; \mathbf { l } _ { t } \big ] \big ) .\tag{18}
$$

The fusion supplies scene context through $\mathbf { g } _ { t }$ before local prediction. The local branch then predicts

$$
\hat { \mathbf { l } } _ { t + 1 } = P _ { \mathrm { L } } ^ { \mathrm { o u t } } ( \mathcal { D } _ { \mathrm { L } } ( \mathbf { f } _ { t } , \mathbf { c } _ { t } ) ) .\tag{19}
$$

The operator $[ \cdot ; \cdot ]$ denotes concatenation, $P _ { \mathrm { F } }$ is the fusion MLP, and $P _ { \mathrm { G } } ^ { \mathrm { o u t } }$ and $P _ { \mathrm { L } } ^ { \mathrm { o u t } }$ are the corresponding predictor-output projections. The pairs $\left( \hat { \mathbf { g } } _ { t + 1 } , \mathbf { g } _ { t + 1 } \right)$ and $( \widehat { \mathbf { l } } _ { t + 1 } , \mathbf { l } _ { t + 1 } )$ define global and local supervision, respectively.

Each predictor processes one current-state token and returns one next-latent-state estimate. The encoded condition $\mathbf { c } _ { t }$ modulates every predictor block through adaptive LayerNorm with zero-initialized residual gates. The global and auxiliary local predictors have separate parameters; their dimensions and depths are specified in the implementation details.

## 3.4 Training Objective and Robustness

The past-motion-window dropout defined above retains the complete standardized motion window when $\delta \ = \ 1$ and replaces its effective input with zeros when $\delta = 0$ . This samplewise perturbation preserves descriptor ordering and forces the current visual state to remain predictive when the motion condition is withheld; it is confined to self-supervised optimization.

The global and local predictors are optimized jointly for the single target in each training pair. For batch size B and lossembedding dimension $d _ { z } ~ = ~ 7 6 8 ,$ let $b \in \{ 1 , \dots , B \}$ index training pairs and let the superscript (b) select the corresponding sample. Using $\| \cdot \| _ { 2 }$ for the Euclidean norm, the implemented mean-squared losses are

Table 1: Comparison across four downstream endoscopic tasks under the corresponding EndoFM-LV evaluation protocols. Values are mean ± standard deviation; best results are shown in bold.
<table><tr><td>Method</td><td>Video Classification F1 (%)</td><td>Polyp Segmentation Dice (%)</td><td>Polyp Detection F1 (%)</td><td>Surgical Phase Recognition Accuracy (%)</td></tr><tr><td>Scratch</td><td> $\overline { { 8 3 . 5 \pm 1 . 3 } }$ </td><td> $\overline { { 5 3 . 2 \pm 3 . 2 } }$ </td><td> $\overline { { 7 3 . 5 \pm 4 . 3 } }$ </td><td> $8 0 . 3 \pm 8 . 7$ </td></tr><tr><td>TimeSformer [24]</td><td> $8 4 . 2 \pm { 0 . 8 }$ </td><td> $5 6 . 3 \pm 1 . 5$ </td><td> $7 5 . 8 \pm 2 . 1$ </td><td> $8 1 . 2 \pm 8 . 9$ </td></tr><tr><td>CORP [25]</td><td> $8 7 . 1 \pm 0 . 6$ </td><td> $6 8 . 4 \pm 1 . 1$ </td><td> $7 8 . 2 \pm 1 . 4$ </td><td> $8 2 . 6 \pm 9 . 6$ </td></tr><tr><td>FAME [26]</td><td> $8 5 . 4 \pm 0 . 8$ </td><td> $6 7 . 2 \pm 1 . 3$ </td><td> $7 6 . 9 \pm 1 . 2$ </td><td> $8 1 . 9 \pm 9 . 2$ </td></tr><tr><td>ProViCo [27]</td><td> $8 6 . 9 \pm 0 . 5$ </td><td> $6 9 . 0 \pm 1 . 5$ </td><td> $7 8 . 6 \pm 1 . 7$ </td><td> $8 2 . 3 \pm 8 . 5$ </td></tr><tr><td>VCL [28]</td><td> $8 7 . 6 \pm 0 . 6$ </td><td> $6 9 . 1 \pm 1 . 2$ </td><td> $7 8 . 1 \pm 1 . 9$ </td><td> $8 2 . 7 \pm 9 . 4$ </td></tr><tr><td>ST-Adapter [29]</td><td> $8 4 . 8 \pm 0 . 7$ </td><td> $6 4 . 3 \pm 1 . 9$ </td><td> $7 4 . 9 \pm 2 . 9$ </td><td> $8 1 . 0 \pm 8 . 7$ </td></tr><tr><td>Endo-FM [3]</td><td> $9 0 . 7 \pm 0 . 4$ </td><td> $7 3 . 9 \pm 1 . 2$ </td><td> $8 4 . 1 \pm 1 . 3$ </td><td> $8 2 . 8 \pm 9 . 1$ </td></tr><tr><td>EndoSSL [30]</td><td> $9 1 . 3 \pm 0 . 6$ </td><td> $7 5 . 1 \pm 1 . 6$ </td><td> $8 3 . 2 \pm { 1 . 6 }$ </td><td> $8 3 . 0 \pm 8 . 0$ </td></tr><tr><td>EndoMamba [16]</td><td> $9 5 . 0 \pm 1 . 3$ </td><td> $8 5 . 4 \pm 0 . 2$ </td><td> $9 0 . 1 \pm 1 . 4$ </td><td> $8 4 . 3 \pm 7 . 3$ </td></tr><tr><td>EndoFM-LV [4]</td><td> $9 6 . 3 \pm 0 . 7$ </td><td> $8 3 . 2 \pm 0 . 3$ </td><td> $9 2 . 5 \pm 1 . 2$ </td><td> $8 6 . 1 \pm 7 . 1$ </td></tr><tr><td>EndoWorld-JEPA (ours)</td><td> ${ \bf 9 8 . 3 \pm 0 . 3 }$ </td><td> ${ \bf 8 7 . 2 \pm 0 . 6 }$ </td><td> ${ \bf 9 5 . 2 \pm 0 . 9 }$ </td><td> ${ \bf 8 9 . 7 \pm 6 . 3 }$ </td></tr></table>

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = \frac { 1 } { B d _ { z } } \sum _ { b = 1 } ^ { B } \left\| \hat { \mathbf { g } } _ { t + 1 } ^ { ( b ) } - \mathbf { g } _ { t + 1 } ^ { ( b ) } \right\| _ { 2 } ^ { 2 } ,\tag{20}
$$

$$
\mathcal { L } _ { \mathrm { l o c a l } } = \frac { 1 } { B d _ { z } } \sum _ { b = 1 } ^ { B } \left\| \hat { \mathbf { l } } _ { t + 1 } ^ { ( b ) } - \mathbf { l } _ { t + 1 } ^ { ( b ) } \right\| _ { 2 } ^ { 2 } ,\tag{21}
$$

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { g l o b a l } } + \lambda _ { \mathrm { L } } \mathcal { L } _ { \mathrm { l o c a l } } + \lambda _ { \mathrm { S I G } } \mathcal { L } _ { \mathrm { S I G R e g } } . } \end{array}\tag{22}
$$

In the total objective, $\mathcal { L } _ { \mathrm { S I G R e g } }$ is the distributional regularizer, and $\lambda _ { \mathrm { L } }$ and $\lambda _ { \mathrm { S I G } }$ weight the local prediction and SIGReg terms, respectively. SIGReg constrains the encoded global states g<sub>t</sub> and $\mathbf { g } _ { t + 1 }$ and prevents their collapse during end-to-end training. Prediction in representation space emphasizes predictable anatomical and dynamic structure without requiring exact reconstruction of reflections, smoke, blur, fluid, or illumination changes.

SIGReg projects the encoded current and target embeddings onto 1,024 randomly sampled unit directions and compares their empirical characteristic functions with that of an isotropic Gaussian at 17 knots. Applying this constraint to both encoded frames preserves dispersion without introducing negative pairs or a separate target network. The local term acts as an auxiliary spatial constraint derived from noisier residual flow, while the global objective remains the principal state-prediction signal.

## 4 Experiments and Results

## 4.1 Pre-training Data and Evaluation Protocols

Self-supervised pre-training data: We construct the unlabeled pre-training corpus from HyperKvasir Video [31], Kvasir-Capsule [32], and LDPolypVideo [33]. After excluding video segments that overlap the PolypDiag validation partition, the corpus contains 6.44 million frames from 587 videos. By frame count, this corpus is smaller than those used by EndoFM-LV (over 13 million) [4] and EndoMamba (11.30 million) [16].

Despite using fewer pre-training frames, the proposed model achieves stronger transfer performance across the four evaluated downstream tasks. Unlabeled segments overlapping the PolypDiag training partition are retained, and the complete filtered corpus is used for self-supervised optimization without an internal hold-out subset.

Downstream evaluation: We assess transfer across four task categories using the corresponding EndoFM-LV data partitions and primary metrics [4]. Visual and motion features are constructed causally at every sampled time: the frame at that time is encoded visually, OM uses at most K preceding transitions, missing prefix entries are zero padded, and no subsequent observation is used.

Video classification: The PolypDiag split contains 173 training and 80 validation videos. Videos are sampled at 2 fps, with 16– 128 retained frames per video, and performance is measured by video-level F1.

Polyp segmentation: The CVC-12k protocol contains 401 training and 171 evaluation frames, and performance is measured by Dice score.

Polyp detection: The processed KUMC train2019 and val2019 partitions contain 15,578 training and 4,254 validation frames, respectively, and performance is measured by frame-level F1. Surgical phase recognition: For Cholec80, videos 1–40 are used for training and 41–80 for testing, producing 85,984 training and 97,874 test windows. Ten-frame sequences are sampled at 1 fps and terminate at the supervised target; performance is measured by video-mean accuracy.

## 4.2 Implementation Details

Pre-training preprocessing: Visual inputs are resized to 224 × 224. After the applicable field-of-view preprocessing, the dedicated motion input is likewise resized to the canonical 224×224 grid before Farneback estimation. Farneback uses pyramid scale 0.5, three levels, a 15-pixel window, three iterations, polynomial neighborhood size 5, and standard deviation 1.2. Residual magnitudes are area-resampled to the patch grid, and motion descriptors are standardized using statistics computed over

Table 2: Dependency-aware component ablation and motion-design controls. Values are shown as mean ± standard deviation over three runs using seeds 0, 1, and 2. G: global prediction; TC: OM temporal conditioning; ML: motion-local prediction; MP: motion–uniform mixed pooling within patch-wise motion weighting; WD: past-motion-window dropout.
<table><tr><td>Variant</td><td></td><td></td><td></td><td>MP</td><td></td><td>Classification</td><td>Segmentation</td><td>Detection</td><td>Phase Recognition</td></tr><tr><td></td><td>G</td><td>TC</td><td>ML</td><td></td><td>WD</td><td>F1 (%)</td><td>Dice (%)</td><td>F1 (%)</td><td>Accuracy (%)</td></tr><tr><td>Global prediction baseline</td><td>√</td><td>一</td><td>一</td><td>一</td><td>一</td><td> $\overline { { 9 1 . 0 \pm 0 . 9 } }$ </td><td> $\overline { { 7 3 . 7 \pm 2 . 0 } }$ </td><td> $\overline { { 8 3 . 9 \pm 1 . 5 } }$ </td><td> $\overline { { 8 3 . 0 \pm 6 . 8 } }$ </td></tr><tr><td>+ temporal conditioning</td><td>√</td><td>√</td><td>一</td><td></td><td>一</td><td> $\overline { { 9 5 . 0 \pm 0 . 7 } }$ </td><td> $\overline { { 7 9 . 3 \pm 1 . 5 } }$ </td><td> $\overline { { 8 9 . 0 \pm 1 . 4 } }$ </td><td> $\overline { { 8 6 . 0 \pm 6 . 3 } }$ </td></tr><tr><td>+ temporal conditioning + dropout</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td> $9 5 . 7 \pm 0 . 3$ </td><td> $8 0 . 2 \pm 1 . 2$ </td><td> $8 9 . 8 \pm 1 . 2$ </td><td> $8 6 . 7 \pm 6 . 4$ </td></tr><tr><td>+ motion-local prediction</td><td>√</td><td>一</td><td>√</td><td>一</td><td>一</td><td> $\overline { { 9 3 . 4 \pm 0 . 6 } }$ </td><td> $8 1 . 1 \pm 1 . 3$ </td><td> $8 9 . 6 \pm 1 . 3$ </td><td> $\overline { { 8 5 . 0 \pm 6 . 7 } }$ </td></tr><tr><td>+ motion-local + mixed pooling</td><td>√</td><td>一</td><td>√</td><td>√</td><td></td><td> $9 4 . 2 \pm 0 . 5$ </td><td> $8 2 . 7 \pm 1 . 2$ </td><td> $9 0 . 9 \pm 1 . 1$ </td><td> $8 5 . 7 \pm 7 . 0$ </td></tr><tr><td>+ temporal conditioning + motion-local</td><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td><td> $\overline { { 9 7 . 0 \pm 0 . 6 } }$ </td><td> $\overline { { 8 4 . 9 \pm 1 . 2 } }$ </td><td> $\overline { { 9 3 . 4 \pm 1 . 2 } }$ </td><td> $\overline { { 8 8 . 1 \pm 7 . 1 } }$ </td></tr><tr><td>+ temporal conditioning + motion-local + mixed pooling</td><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td> $9 7 . 7 \pm 0 . 4$ </td><td> $8 6 . 1 \pm 0 . 9$ </td><td> $9 4 . 4 \pm 0 . 9$ </td><td> $8 8 . 9 \pm 6 . 3$ </td></tr><tr><td>+ temporal conditioning + motion-local + dropout</td><td>√</td><td>√</td><td>√</td><td>一</td><td>√</td><td> $9 7 . 5 \pm 0 . 3$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $9 4 . 1 \pm 1 . 0$ </td><td> $8 9 . 0 \pm 6 . 6$ </td></tr><tr><td>Full + shuffled motion condition</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td> $\overline { { 9 4 . 4 \pm 0 . 7 } }$ </td><td> $\overline { { 8 2 . 4 \pm 1 . 2 } }$ </td><td> $\overline { { 9 0 . 7 \pm 1 . 7 } }$ </td><td> $8 5 . 6 \pm 6 . 5$ </td></tr><tr><td>Dense-flow magnitude only</td><td>√</td><td>一</td><td>V</td><td>√</td><td>一</td><td> $9 2 . 8 \pm 0 . 7$ </td><td> $7 8 . 9 \pm 1 . 3$ </td><td> $8 7 . 8 \pm 1 . 6$ </td><td> $8 4 . 3 \pm 7 . 7$ </td></tr><tr><td>Full EndoWorld-JEPA</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td> $\overline { { 9 8 . 3 \pm 0 . 3 } }$ </td><td> $\overline { { 8 7 . 2 \pm 0 . 6 } }$ </td><td> $\overline { { 9 5 . 2 \pm 0 . 9 } }$ </td><td> $\overline { { 8 9 . 7 \pm 6 . 3 } }$ </td></tr></table>

MP requires ML; WD requires TC. Pre-training shuffle reassigns intact A<sub>t</sub> across episodes while $\overline { { \mathbf { M } _ { t } } }$ remains aligned. The dense-flow-only control omits TC and WD and defines M<sub>t</sub> by area-resampling $\| \mathbf { u } _ { s _ { t } } ( \cdot ) \| _ { 2 }$ to the patch grid. Metrics follow Table 1.

Common downstream adaptation: Visual inputs are resized to 224 × 224 for PolypDiag and CVC-12k and to $6 4 0 \times 6 4 0$ for KUMC. Cholec80 frames undergo fixed field-of-view cropping, 250 × 250 resizing, and 224 × 224 cropping. Regardless of the task-specific visual resolution, OM operates on the canonical 224 × 224 motion grid. Within each run, all tasks initialize from the common pre-training checkpoint and fuse

Pre-training architecture and optimization: The model uses a ViT-Base encoder with patch size 14 and hidden dimension 768. For each sampled state, the motion encoder ${ \mathcal { E } } _ { \mathrm { M } }$ receives the five ordered eight-dimensional descriptors as a 40- dimensional vector. $\mathrm { ~ A ~ 1 ~ } \times \mathrm { ~ 1 ~ }$ channel projection $( 4 0  1 0 )$ followed by a two-layer SiLU MLP $( 1 0  3 0 7 2  7 6 8 )$ produces a single condition token $\mathbf { c } _ { t } \in \mathbb { R } ^ { 1 \times 7 6 8 }$ . The global and local projection MLPs have a hidden dimension of 2,048 and use per-sample LayerNorm; the global and local conditional predictors contain six and two layers, respectively, with 12 attention heads and a 3,072-dimensional feed-forward layer. Transformer dropout is 0.2 and embedding dropout is 0.1. Following the temporal blocking setting of LeWM [7], the shared observation horizon is fixed at $K = 5$ . We retain $\lambda _ { \mathrm { S I G } } = 0 . 0 9$ , which lies near the optimum of LeWM’s reported sensitivity study within its stable 0.01–0.2 interval. Because the patch scores are standardized, $\tau = 1 . 0$ retains their unit scale. We fix $\alpha = 0 . 7 ,$ $\lambda _ { \mathrm { L } } = 0 . 0 5$ , and the past-motion-window dropout probability to 0.3 throughout all experiments; these choices allocate 30% uniform spatial mass, keep local prediction auxiliary, and retain motion conditioning in 70% of training pairs, respectively. Optimization runs for 200 epochs and uses AdamW with a learning rate of $3 \times 1 0 ^ { - 5 }$ , weight decay of $5 \times 1 0 ^ { - 3 }$ , batch size 64, bfloat16 precision, gradient clipping at 1.0, and a linearwarmup cosine schedule. The self-supervised objective is averaged across all corpus mini-batches within each epoch, and the epoch-end checkpoint with the lowest mean initializes all downstream tasks. Neither downstream annotations nor downstream validation metrics enter checkpoint selection.

the complete pre-training corpus on this canonical grid. Temporal adjacency follows decoded observation order.

$\mathbf { c } _ { t } ,$ but not $\mathbf { M } _ { t } ;$ the local-motion magnitude map, its derived weight map, and the motion-local predictor are pre-trainingonly. Across all four tasks, the transferred visual and motion encoders remain frozen, and only the task-specific adapters are optimized. The video-classification adapter additionally uses descriptor-derived frame weights and is the only one to reuse the frozen global predictor. The complete adaptation protocol is repeated three times using seeds 0, 1, and 2, and Table 1 reports the mean and standard deviation across these runs.

## 4.3 Video Classification

For video-level appearance recognition, descriptor-derived magnitudes weight sampled frame features, and an MLP combines the pooled visual state, mean encoded condition, temporal statistics, and one global-predictor feature computed from a checkpoint-compatible current-state input. On the PolypDiag benchmark, EndoWorld-JEPA achieves 98. $. 3 \pm 0 . 3 \% \mathrm { F 1 }$

## 4.4 Polyp Segmentation

Dense lesion delineation uses a hybrid decoder that combines ViT layers 3, 6, 9, and 12 with a shallow convolutional path. The condition $\mathbf { c } _ { t }$ is projected to biases at the four decoder scales, and the decoder is trained with a weighted binary-crossentropy–Dice loss. On the CVC-12k benchmark, the model achieves $8 7 . 2 \pm 0 . 6 \%$ Dice.

## 4.5 Polyp Detection

Lesion localization is performed by a compact feature pyramid applied to the target frame. The projected condition is added before heads that predict a center heatmap, offsets, and box dimensions using focal and smooth- ${ \boldsymbol { \mathbf { \ell } } } _ { \mathbf { \ell } } - { \boldsymbol { \ell } } _ { 1 }$ losses. The detector obtains $9 5 . 2 \pm 0 . 9 \% \mathrm { F } 1$ on the KUMC benchmark.

## 4.6 Surgical Phase Recognition

Procedural progression is modeled by adding the projected condition to spatial tokens before divided space–time decoding and concatenating it with decoded visual and elapsed-time features before a unidirectional LSTM. Causal smoothing uses no later frame. On the Cholec80 benchmark, the model achieves $8 9 . 7 \pm 6 . 3 \%$ video-mean accuracy.

Cross-task comparison: Baselines in Table 1 follow the corresponding EndoFM-LV evaluation protocols; statistics are taken from its unified evaluation [4], the two published EndoMamba entries [16], and matched-protocol EndoMamba runs on the remaining tasks. A common EndoWorld-JEPA checkpoint achieves the highest reported mean in every task category, exceeding the strongest previous values by 2.0, 1.8, 2.7, and 3.6 points for video classification, polyp segmentation, polyp detection, and surgical phase recognition, respectively. The consistent gains support the transferability of the learned visual– motion representation across task types.

## 4.7 Component Ablation Study

Table 2 reports independently pre-trained variants with global prediction retained throughout. Each variant is independently pre-trained and evaluated in three runs using seeds 0, 1, and 2. Across rows, the visual inputs, pre-training data, normalization, optimization budgets, checkpoint selection, and downstream adaptation protocol are identical; mean and standard deviation over the three runs are reported. Temporal conditioning and motion-local prediction are tested independently and jointly; mixed pooling requires the local branch, and pastmotion-window dropout requires temporal conditioning. In variants without TC, every active predictor receives $\mathbf { c } _ { t } \ = \ \mathbf { 0 } ;$ ML variants without MP set $\alpha = 1$ , retaining motion-only softmax pooling. The motion-design controls are separated from the component rows. The shuffled control retains all modules but reassigns intact windows across episodes, preserving motion statistics while removing state correspondence. The denseflow-only control removes descriptor-based temporal conditioning and area-resamples the aligned field $\| \mathbf { u } _ { s _ { t } } ( \cdot ) \| _ { 2 }$ to obtain $\mathbf { M } _ { t } ;$ consequently, TC and WD are inactive while G, ML, and MP are retained.

The global-prediction baseline performs at a comparable level to Endo-FM across the four tasks. Temporal conditioning adds 3.0–5.6 points. Motion-local prediction adds 2.0–7.4 points, with substantially larger gains of 7.4 and 5.7 points for segmentation and detection than for classification and phase recognition. Combining both components improves the baseline by 6.0, 11.2, 9.5, and 5.1 points for classification, segmentation, detection, and phase recognition, respectively. Adding mixed pooling or past-motion-window dropout to this combination contributes 0.7–1.2 and 0.5–0.9 points, respectively, while using both yields a further 1.3–2.3 points. During pre-training, shuffling affects only $\mathbf { A } _ { t } \mathbf { : }$ the current image and aligned ${ { \bf { M } } _ { t } }$ remain intact, so the predictor can attenuate the mismatch and remain close to the motion-local plus mixed-pooling variant. The consistent 3.9–4.8-point reduction under shuffled conditioning indicates that temporal guidance relies on correct state– motion correspondence rather than on motion-descriptor statistics alone. Under the matched ${ \bf G } { + } { \bf M } { \bf L } { + } { \bf M } { \bf P }$ setting, direct denseflow magnitude trails residual-motion weighting by 1.4, 3.8, 3.1, and 1.4 points, respectively. The larger reductions for segmentation and detection support dominant-motion removal before spatial guidance.

The global and local objectives impose complementary inductive biases. Global prediction preserves anatomy, viewpoint, and scene organization, whereas motion-local prediction emphasizes sparse changes that may be attenuated by global pooling. Consistent with this distinction, temporal conditioning improves all four tasks, and the local branch provides its largest gains for segmentation and detection. Median-flow subtraction establishes a robust image-plane reference, although residual responses can also arise from occlusion boundaries, specularities, instruments, or flow errors. Within patch-wise motion weighting, mixed pooling limits domination by a small number of high-response patches, and past-motion-window dropout maintains the predictive contribution of visual semantics when motion is unreliable. The fixed spatial weighting pathway is excluded from downstream heads but shapes the transferred visual encoder through the auxiliary pre-training objective. By comparison, the learned condition $\mathbf { c } _ { t }$ is fused at the representation level appropriate to each task, and the global predictor is reused only by the video-classification adapter. This transfer strategy preserves a common representation while allowing task-specific adaptation.

## 5 Discussion

Across four endoscopic tasks, motion-conditioned latent-state prediction provides a transferable pre-training signal for classification, segmentation, detection, and surgical phase recognition. Using a common checkpoint with task-specific heads, the results provide preliminary evidence that observation-indexed OM can transfer across datasets with heterogeneous temporal sampling. This pattern is consistent with cadence diversity acting as implicit temporal augmentation, while prospective evaluation across acquisition rates remains necessary.

The use of a common pre-trained checkpoint across four task-specific heads supports broad transferability, although final performance also depends on each downstream adapter. In contrast to objectives that enforce temporal invariance or reconstruct appearance details, the JEPA objective emphasizes content with predictable representation-level evolution. OM further reduces prediction ambiguity by conditioning the predictors on motion observed only up to the current state, thereby preserving causal feature construction in passive recordings. Optical flow and motion descriptors are precomputed offline during self-supervised pre-training. At downstream inference, OM can be updated causally with one Farneback estimate and lightweight descriptor encoding per new observation, without introducing an additional learned optical-flow network.

Several limitations remain. Farneback flow is efficient and deterministic, but degrades under rapid motion, large occlusion, weak texture, smoke, and severe illumination change. Component-wise median subtraction captures a dominant translation-like displacement but cannot fully factor rotation, zoom, depth-dependent parallax, or multiple independently moving regions. The fixed K-transition condition represents one observation scale; long-horizon stability, uncertainty calibration, counterfactual control, and planning remain outside this one-step formulation. Control-aware forecasting would additionally require physical control signals. The frame-wise visual encoder also provides a path to adaptation without temporal inputs when only isolated images are available; dedicated still-image validation remains future work. The external comparisons include different task-specific pipelines, so final performance reflects adapter design as well as pre-training quality. Although all our protocols use the same checkpoint and causal alignment, broader conclusions require prospective evaluation across institutions, devices, operators, and acquisition conditions.

Future work should explore confidence-aware flow, controlled temporal resampling, adaptive prediction horizons, and synchronized kinematics to test motion reliability and cadence robustness and to compare observation-derived conditions with physical controls.

## 6 Conclusion

We introduced EndoWorld-JEPA, a causal motion-conditioned JEPA world model for passive endoscopic video. OM converts recent dense flow into an observation-derived temporal motion condition and local-motion cue without recorded controls or calibrated poses. Dominant–residual decomposition and complementary global–local prediction preserve scene context while emphasizing localized dynamics; patch-wise mixed weighting and past-motion-window dropout improve robustness, and latent prediction avoids reconstructing stochastic appearance artifacts. A common pre-trained visual–motion checkpoint achieves the highest reported mean across four downstream task categories. Ablations link these gains to correct state–motion alignment and residual guidance, supporting causal motion-conditioned latent-state prediction as an effective and transferable pre-training principle for endoscopy.

## References

[1] J. Liu, K. R. Subramanian, and T. S. Yoo, “An optical flow approach to tracking colonoscopy video,” Computerized Medical Imaging and Graphics, vol. 37, no. 3, pp. 207–223, 2013.

[2] Y. Liu, C. Li, H. Liu, C. Yang, and Y. Yuan, “Foundation modelguided gaussian splatting for 4d reconstruction of deformable tissues,” IEEE Transactions on Medical Imaging, vol. 44, no. 6, pp. 2672–2682, 2025.

[3] Z. Wang, C. Liu, S. Zhang, and Q. Dou, “Foundation model for endoscopy video analysis via large-scale self-supervised pretrain,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, ser. Lecture Notes in Computer Science, vol. 14228. Springer Nature Switzerland, 2023, pp. 101– 111.

[4] Z. Wang, C. Liu, L. Zhu, T. Wang, S. Zhang, and Q. Dou, “Improving foundation model for endoscopy video analysis via representation learning on long sequences,” IEEE Journal of Biomedical and Health Informatics, vol. 29, no. 5, pp. 3526– 3536, 2025.

[5] M. Assran et al., “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15 619–15 629.

[6] A. Bardes et al., “Revisiting feature prediction for learning visual representations from video,” Transactions on Machine Learning Research, 2024. [Online]. Available: https://openreview.net/ forum?id=QaCCuDfBk2

[7] L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.

[8] J. Wang et al., “Development of deep learning-based narrowband imaging endocytoscopic classification for predicting colorectal lesions from a retrospective study,” Nature Communications, vol. 16, no. 1, 2025, art. no. 8351.

[9] Y. Jiang et al., “Biology-guided deep learning predicts prognosis and cancer immunotherapy response,” Nature Communications, vol. 14, no. 1, 2023, art. no. 5135.

[10] Y. Zhang et al., “Advancing brain tumor segmentation in smart healthcare via T1-generated virtual multimodal MRI fusion,” Applied Soft Computing, vol. 186, 2026, art. no. 114278.

[11] S. Kim et al., “EchoFM: Foundation model for generalizable echocardiogram analysis,” IEEE Transactions on Medical Imaging, vol. 44, no. 10, pp. 4049–4062, 2025.

[12] Z. Yang, X. Xu, J. Zhang, G. Wang, M. K. Kalra, and P. Yan, “Chest x-ray foundation model with global and local representations integration,” IEEE Transactions on Medical Imaging, vol. 44, no. 12, pp. 4787–4799, 2025.

[13] F. Wang and L. Yu, “Scaling chest x-ray foundation models from mixed supervisions for dense prediction,” IEEE Transactions on Medical Imaging, vol. 44, no. 12, pp. 5306–5318, 2025.

[14] P. Dermyer, A. Kalra, and M. Schwartz, “EndoDINO: A foundation model for GI endoscopy,” arXiv preprint arXiv:2501.05488, 2025.

[15] M. R. Jong et al., “GastroNet-5M: A multicenter dataset for developing foundation models in gastrointestinal endoscopy,” Gastroenterology, vol. 170, no. 1, pp. 174–187, 2026.

[16] Q. Tian et al., “EndoMamba: An efficient foundation model for endoscopic videos via hierarchical pre-training,” in Medical Image Computing and Computer Assisted Intervention – MIC-CAI 2025, ser. Lecture Notes in Computer Science, vol. 15968. Springer Nature Switzerland, 2025, pp. 224–234.

[17] M. Assran et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[18] Y. Yue, Y. Wang, C. Tao, P. Liu, S. Song, and G. Huang, “CheX-World: Exploring image world modeling for radiograph representation learning,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 20 778– 20 788.

[19] Y. Yang et al., “Medical world model,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 8319–8329.

[20] H. Jiang et al., “Cardiac copilot: Automatic probe guidance for echocardiography with world model,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, ser. Lecture Notes in Computer Science, vol. 15001. Springer Nature Switzerland, 2024, pp. 190–199.

[21] M. Hayoz et al., “Learning how to robustly estimate camera pose in endoscopic videos,” International Journal of Computer Assisted Radiology and Surgery, vol. 18, no. 7, pp. 1185–1192, 2023.

[22] X. Liu et al., “Dense depth estimation in monocular endoscopy with self-supervised learning methods,” IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1438–1447, 2020.

[23] G. Farnebäck, “Two-frame motion estimation based on polynomial expansion,” in Image Analysis – SCIA 2003, ser. Lecture Notes in Computer Science, vol. 2749. Springer, 2003, pp. 363–370.

[24] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 813–824.

[25] K. Hu, J. Shao, Y. Liu, B. Raj, M. Savvides, and Z. Shen, “Contrast and order representations for video self-supervised learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 7939–7949.

[26] S. Ding et al., “Motion-aware contrastive video representation learning via foreground-background merging,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 9716–9726.

[27] J. Park, J. Lee, I.-J. Kim, and K. Sohn, “Probabilistic representations for video contrastive learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 711–14 721.

[28] R. Qian, S. Ding, X. Liu, and D. Lin, “Static and dynamic concepts for self-supervised video representation learning,” in Computer Vision – ECCV 2022, ser. Lecture Notes in Computer Science, vol. 13686. Springer Nature Switzerland, 2022, pp. 145– 164.

[29] J. Pan, Z. Lin, X. Zhu, J. Shao, and H. Li, “ST-Adapter: Parameter-efficient image-to-video transfer learning,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 26 462–26 477.

[30] R. Hirsch et al., “Self-supervised learning for endoscopic video analysis,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, ser. Lecture Notes in Computer Science, vol. 14224. Springer Nature Switzerland, 2023, pp. 569–578.

[31] H. Borgli et al., “HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy,” Scientific Data, vol. 7, no. 1, 2020, art. no. 283.

[32] P. H. Smedsrud et al., “Kvasir-Capsule, a video capsule endoscopy dataset,” Scientific Data, vol. 8, no. 1, 2021, art. no. 142.

[33] Y. Ma, X. Chen, K. Cheng, Y. Li, and B. Sun, “LDPolypVideo benchmark: A large-scale colonoscopy video dataset of diverse polyps,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, ser. Lecture Notes in Computer Science, vol. 12905. Springer Nature Switzerland, 2021, pp. 387–396.