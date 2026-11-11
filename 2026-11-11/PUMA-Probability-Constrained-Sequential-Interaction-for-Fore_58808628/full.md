# PUMA: Probability-Constrained Sequential Interaction for Forensic Localization

Zhouli Yu

Mengyuan Zhou

## Abstract

Image manipulation localization must jointly exploit semantic structure and noise-sensitive forensic evidence. However, the efects of spatial correspondence, matching quality, and interaction order are often conflated. Existing semantic–forensic fusion methods provide complementary representations, but do not explicitly preserve quality-mismatch information during spatial correspondence or adaptively select the order in which information is propagated across branches. We propose PUMA (Probability-constrained Unbalanced Matching and Aggregation), which combines spatially constrained unbalanced optimal transport with position-wise convex selection between two sequential operators through a shared positional interface. Before normalization, PUMA retains signed transport-mass residuals, constructs row-stochastic cross-branch and withinnoise-branch operators, and updates noise features through a gated residual path.

PUMA achieves strong localization results on four test sets, with scores on COVERAGE and NIST16 exceeding the published results of the representative methods listed in our comparison. The experiments support the utility of local matching states and adaptive sequential interaction for crosssource localization and several post-processing conditions.

## 1 Introduction

With the rapid development of digital image editing and generative artificial intelligence, the barrier to image manipulation continues to fall. Images altered through splicing, object removal, or generative editing [34, 35] are widely circulated on social networks, in news media, and in legal-evidence settings [27], creating practical risks such as the spread of misinformation and the loss of evidentiary value. Passive forensic techniques that localize manipulation by analyzing an image’s content and statistical traces, without relying on an embedded watermark or an original reference image [24], are therefore important. Pixel-level image manipulation localization requires an algorithm to output a binary mask with the same dimensions as the input, labeling manipulated pixels individually [36]. Compared with image-level authenticity classification, this task provides finer-grained decisions, but it is more challenging because traces can be subtle and scenes are highly diverse [37].

Early passive-forensics research focused on individual types of manipulation traces and gradually evolved toward the joint use of multiple features [28, 7]. Noiseprint models characteristic camera-sensor noise to suppress scene-content interference and extract device-discriminative camera fingerprints, providing a stable microscopic cue for manipulation analysis [5]. CAT-Net jointly models the original RGB image and JPEG-compression artifacts, exploiting the complementarity between content structure and compression traces to improve localization [11, 1]. TruFor further combines RGB appearance with a learnable noise-sensitive fingerprint and predicts a reliability measure, improving robustness under complex post-processing [8]. More recently, Co-Transformers separately encode semantic and microscopic forensic evidence and coordinate them through cross-branch feature interaction. It is a representative high-performing paradigm among current two-branch manipulation-localization models [22].

Although two-branch architectures have substantially improved localization by exploiting complementary semantic and forensic features, current cross-branch interaction designs still have limitations. Many mechanisms use Softmax normalization to produce aggregation weights. This preserves only relative allocation among features and removes information about the overall degree of matching between the two feature sets. In forensic analysis, however, semantic–forensic mismatch itself can be a discriminative cue [29]; normalization discards this potentially valuable information. Conventional attention and balanced optimal transport also assume mass conservation and force all information to be transported. This assumption does not fit manipulated regions, where the two feature types may naturally disagree. Forced transfer of irrelevant information can instead impair forensic representations. In addition, existing methods commonly fix the execution order of within-branch aggregation and cross-branch transport, preventing adaptation to location-dependent diferences such as those between normal regions and manipulation boundaries.

To address these issues, we propose Probability-constrained Unbalanced Matching and Aggregation (PUMA). First, a spatially constrained unbalanced optimal transport module replaces conventional balanced matching within a local neighborhood. By relaxing strict mass conservation, it allows the amount of feature transport to vary with the degree of bilateral matching. It also explicitly extracts mass residuals before normalization, turning semantic–forensic mismatch into a usable discriminative cue and recovering information lost by conventional Softmax normalization. Building on this module, we design a probability-constrained sequential interaction block. It computes feature outputs for both “within-branch aggregation first” and “cross-branch transport first” propagation orders. Using mass residuals as control signals, a position-wise gate adaptively mixes the two outputs instead of imposing a manually fixed order. The interaction remains row-stochastic, and a separate residual-strength gate provides a fallback that prevents mismatched information from interfering with forensic features.

## 2 Related Work

## 2.1 Single-source forensic localization

Early image manipulation localization methods built discriminative representations around specific forensic traces, analyzing local image statistics and anomalous patterns to identify potential manipulated regions [28, 9]. Constrained convolution high lights traces left by manipulation by restricting the parameters of the first convolutional layer [49]. Noiseprint models charac teristic camera-sensor noise to extract device-discriminative fingerprints and reduce scene-content interference in forensic representations [5]. ManTra-Net localizes multiple types of manipulation by analyzing local anomalies [20, 23]. BusterNet targets copy-move manipulation and predicts both source and target regions [50]; DifForensics uses difusion priors to learn mesoscopic cues between macro- and microscopic evidence [48]. These methods provide useful evidence from diferent forensic cues, but their performance remains constrained by trace visibility and image post-processing.

## 2.2 Multi-source fusion and interaction

To exploit complementary evidence, later work moved from single-source representations toward multi-source evidence fusion [16, 25]. CAT-Net jointly models RGB information and JPEG-compression-domain features [11]; TruFor combines RGB appearance with noise-sensitive fingerprints and can predict localization reliability [8]. MVSS-Net combines noise and boundary evidence [3, 38]. ObjectFormer combines RGB and high-frequency features and uses object prototypes to model regional consistency [47]. Co-Transformers coordinates semantic and microscopic forensic information through dualbranch encoding and cross-branch interaction [22, 30]. IML-ViT investigates high-resolution, multi-scale features and edge supervision [45, 14], while Mesorch fuses Transformer and CNN representations in parallel [46]. These studies show that efective multi-source evidence use depends on an appropriate interaction mechanism. PUMA focuses more specifically on two central issues: local transport quality and the order of sequential propagation.

## 2.3 Attention and optimal-transport matching

Attention mechanisms aggregate information through normalized weights over positions or features [21, 40] and are widely used for relational modeling and information exchange [41]. Transformers use attention to establish long-range dependencies [17, 33]; non-local networks model long-range dependencies through pairwise similarities [19, 39]; and graph attention networks perform weighted aggregation over local neighborhoods [18, 31]. Unlike similarity-based weighted aggregation, optimal transport represents correspondences between two sets of features through a transport plan. Entropy regularization substantially improves the eficiency of optimal-transport solvers [6], while unbalanced optimal transport relaxes marginal constraints to handle unequal mass on the two sides [4]. Sinkformers explore Sinkhorn normalization in attention computation [15]. Forensic Similarity compares image regions using forensic traces and emphasizes that forensic correspondence is not equivalent to visual-appearance similarity [13].

Together, these studies ofer diferent approaches to multisource fusion through feature weighting, correspondence modeling, and multi-path interaction. Unlike methods that focus on general feature aggregation [42] or correspondence modeling, we investigate the relationship between local matching state and sequential interaction order. At each position, PUMA computes both within-branch-aggregation-first and cross-branch-transport-first outcomes, then uses local matching information to adapt their relative contributions. Spatially constrained unbalanced matching preserves local mass information, while probability-constrained sequential interaction adaptively selects the propagation order.

## 3 Method

This section describes PUMA. Given an input image, the model performs pixel-level manipulation localization. Following Co-Transformers, a two-branch backbone extracts semantic and noise-forensic features. PUMA is a plug-in interaction module inserted at the pre-fusion interface. On a shared spatial grid, it performs feature alignment, spatially constrained unbalanced matching, and adaptive sequential interaction. The two branch features are first projected into a common matching space. Unbalanced optimal transport then establishes cross-branch correspondences and extracts mass-residual cues. Finally, those residuals control the position-wise mixture of two propagation orders and the interaction strength, and the noise features are updated residually. The updated noise features and the original semantic features are passed to the baseline model’s existing fusion and localization head. The backbone and prediction pathway remain unchanged [22].

## 3.1 Local unbalanced matching and mass residuals

To address the loss of overall bilateral matching information caused by row normalization and the uninformative forensic transfer that can result from strict mass conservation, this module performs spatially constrained unbalanced optimaltransport matching. Unbalanced transport relaxes mass conservation within a local neighborhood, allowing transported feature mass to vary with bilateral match quality. Before normalization, the module explicitly extracts and retains the massresidual state. It outputs both relative aggregation weights for cross-branch message passing and match-quality cues for subsequent gated interaction.

At the pre-fusion interface, the RGB feature � and noise feature � have 512 and 768 channels, respectively, and both lie on a 32×32 grid. Point-wise matching cannot exploit neighboring responses, whereas global matching can introduce similar textures from distant locations. We therefore set the support radius to 2 and retain only the truncated RGB neighborhood around each noise position, limiting the number of candidates to at most 25. This permits small local ofsets while controlling erroneous correspondences and computational cost.

After layer normalization, bias-free linear projection, and unit-norm normalization, both branches yield 64-dimensional descriptors [32, 43]. Let $\bar { f } _ { i }$ and $\bar { r } _ { j }$ denote the descriptors at noise position � and RGB position �, respectively, and let S denote the local support. The matching cost is

![](images/333c6ef5ab5af4ef140df35ac43254399aeae76bd9d75e3b3fe985f71d9fc609.jpg)  
Figure 1: Overview of PUMA. Spatially constrained unbalanced matching preserves local mass states, which guide sequential interaction and residual updates of noise features.

$$
D _ { i j } = 1 - \bar { f _ { i } } ^ { \top } \bar { r } _ { j } + \frac { 0 . 1 \| \Delta _ { i j } \| _ { 2 } ^ { 2 } } { 2 ^ { 2 } } ,\tag{1}
$$

The first term measures feature diference using cosine distance, assigning lower cost to descriptors with similar directions. The second term penalizes spatial displacement, where $\Delta _ { i j }$ is the grid-coordinate diference and distance is normalized by the squared support radius. The coeficient 0.1 controls the spatial constraint. The local support excludes distant candidates, while the spatial penalty further favors nearby positions, preventing matching from being dominated by appearance similarity alone.

Let $a _ { i } \ = \ b _ { j } \ = \ 1 / 1 0 2 4$ be the reference masses at all positions on the two sides, and let the nonnegative matrix � represent transported mass between position pairs. Entries outside the support are fixed to zero. To let weakly corresponding positions transport less mass rather than enforcing fixed row and column totals, we use the following unbalanced optimal-transport objective:

$$
\begin{array} { r l } & { \underset { T \geq 0 } { \mathrm { m i n } } \ : \langle D , T \rangle + \varepsilon \mathrm { K L } ( T \| a b ^ { \top } ) } \\ & { ~ + \tau _ { r } \mathrm { K L } ( T \mathbf { 1 } \| a ) } \\ & { ~ + \tau _ { c } \mathrm { K L } ( T ^ { \top } \mathbf { 1 } \| b ) . } \end{array}\tag{2}
$$

The transport-cost term favors low-cost position pairs. The coupling regularizer uses $a b ^ { \top }$ as a reference and controls the concentration of the assignment. The final two terms softly constrain the row and column marginals using generalized KL divergence, allowing actual mass to deviate from the reference mass so that both matching cost and mass deviation determine the transport plan. We set $\varepsilon = 0 . 1$ and $\tau _ { r } = \tau _ { c } = 1$ , use 128 log-domain iterations, and retain gradients to learn the matching descriptors.

The row and column sums of �, denoted by $m _ { i }$ and $n _ { j } ,$ are the actual transported masses at noise and RGB positions. Their relative residuals are

$$
\rho _ { i } = \frac { m _ { i } } { a _ { i } } - 1 , \qquad \kappa _ { j } = \frac { n _ { j } } { b _ { j } } - 1 .\tag{3}
$$

Positive and negative residuals indicate excess and deficient mass, respectively. These states preserve total-mass diferences that row-wise normalization would remove, providing local matching evidence for subsequent gating.

The controller uses $\log ( m _ { i } / a _ { i } )$ and $\log ( n _ { j } / b _ { j } )$ , each centered by subtracting the mean over its grid. The logarithm describes relative change, and centering emphasizes positional diferences; the two states are aligned by their shared-grid indices. If a mass ratio is below $1 0 ^ { - 8 }$ or non-finite, its log ratio is set to zero before centering. This state describes deviation in matching mass and is not a manipulation probability.

After retaining the mass state, the coupling matrix is rownormalized into a local aggregation operator:

$$
A _ { i j } = { \frac { T _ { i j } } { m _ { i } } } .\tag{4}
$$

If $m _ { i } / a _ { i }$ is below $1 0 ^ { - 8 }$ or non-finite, that row uses the identity mapping to avoid division by near-zero mass. The operator � is derived from noise–RGB correspondences and is subsequently applied to noise value features indexed on the shared grid, allowing semantic relationships to guide the spatial organization of forensic information. The module thus outputs relative aggregation weights and pre-normalization mass states for message propagation and gate control, respectively.

## 3.2 Sequential propagation and gated updates

Existing cross-branch interaction often fixes the propagation order, which cannot accommodate location-dependent feature relationships at normal regions and manipulation boundaries. Moreover, coupling propagation order with update strength can introduce irrelevant information into forensic representations in mismatched regions. We address these issues with probability constrained sequential propagation and gated updating. Two operator-composition orders are constructed in parallel; mass residuals from unbalanced matching control a position-wise mixture of the two orders. A separate strength gate decouples order selection from update magnitude, and the noise feature is updated through a residual path while maintaining rowstochastic probability constraints.

On the shared indices and local support of �, the noise features are normalized and independently projected to form 64-dimensional queries and keys and a 128-dimensional value field �. Query–key dot products are divided by 64 and passed through Softmax over the valid neighborhood to obtain the noise self-aggregation operator �. The operator � describes within-forensic relationships, while � provides semantic guidance. Both act on the same value field and define two sequential messages:

$$
\begin{array} { c } { { Y _ { A N } = A ( N V ) , } } \\ { { } } \\ { { Y _ { N A } = N ( A V ) . } } \end{array}\tag{5}
$$

The first path supplements local forensic context before propagating messages according to cross-branch relationships; the second first applies semantic guidance and then reorganizes the messages according to forensic relationships. Both paths reuse the same �, �, and �, so their diference is concentrated in the composition order.

Two independent MLPs receive the two centered mass states and the 64-dimensional descriptors from both branches, for an input dimension of 130. Each MLP has a 64-dimensional GELU hidden layer and a Sigmoid output, producing an order gate $\alpha _ { i }$ and a strength gate $g _ { i }$ . The descriptors provide content evidence, while the mass states supply matching diferences that are unavailable after normalization. The order gate first constructs a position-wise mixed operator:

$$
\begin{array} { l } { P = \mathrm { d i a g } ( \alpha ) A N } \\ { \qquad + \left( I - \mathrm { d i a g } ( \alpha ) \right) N A . } \end{array}\tag{6}
$$

When the order gate approaches 1, it favors ��; when it approaches 0, it favors $N A$ . Intermediate values yield a continuous selection. To separate “which order to use” from “how much neighborhood information to introduce,” the strength gate interpolates between mixed propagation and retaining the original value:

$$
\begin{array} { l } { G = \operatorname { d i a g } ( g ) , } \\ { Q = I - G + G P , \qquad Z = Q V . } \end{array}\tag{7}
$$

A small strength gate preserves more of the original value, whereas a larger one introduces more neighborhood messages.

This prevents the order gate from also having to control update magnitude. Both gates are learned from localization supervision; neither mass-residual sign is assigned a fixed switch condition.

Because � and � are nonnegative and row-stochastic, their compositions �� and � � retain these properties. The order and strength gates lie in the unit interval, so the two convex combinations preserve nonnegative spatial weights whose rows sum to one. Thus, order adjustment does not rely on negative weights or amplify the total weight. This probability constraint applies to spatial mixing.

Finally, only the aggregated increment $Z - V$ is mapped back to the 768-dimensional noise space using a bias-free projection $W _ { o } .$ , followed by a residual update:

$$
F _ { \mathrm { o u t } } = F + W _ { o } ( Z - V ) .\tag{8}
$$

Writing back only the increment exploits neighborhood information while preserving the original forensic representation. As the strength gate approaches zero, the update approaches the identity mapping. The module updates only the noise feature; the RGB feature remains unchanged. They are then concatenated along the channel dimension and passed to the original prediction head.

## 3.3 Joint supervision loss

The proportion of manipulated pixels varies substantially, so optimizing only by pixel count can bias learning toward the background. Regional prediction also requires boundary structure from the semantic branch. Training therefore uses joint region and boundary Dice supervision: the region loss constrains overlap of the final mask, while the boundary loss constrains the structural representation of the RGB branch. The post-interaction forensic information and boundary cues thus jointly support localization.

The final localization output and boundary-branch output are first bilinearly upsampled to the input resolution and passed through Sigmoid to obtain probability maps $p$ and $p _ { e }$ . Their labels are the manipulation mask � and its derived boundary mask $y _ { e }$ . Both terms use the same Dice loss:

$$
\mathcal { L } _ { D } ( p , y ) = 1 - \frac { 2 \sum _ { u \in \Omega } p _ { u } y _ { u } + s } { \sum _ { u \in \Omega } p _ { u } ^ { 2 } + \sum _ { u \in \Omega } y _ { u } ^ { 2 } + s } .\tag{9}
$$

Here, Ω denotes all pixels in a training batch on the current device, and $p _ { u }$ and $y _ { u }$ are the predicted probability and binary label. The smoothing constant is $s = 1$ to stabilize computation in cases such as empty regions. The denominator uses the sum of squared predictions and labels; the loss balances regional recall and false-positive control through overall overlap.

Let $\mathcal { L } _ { \mathrm { p i x } } = \mathcal { L } _ { D } ( p , y )$ and $\mathcal { L } _ { \mathrm { e d g e } } = \mathcal { L } _ { D } ( p _ { e } , y _ { e } )$ denote the region and boundary losses, respectively. The total training objective is

$$
\begin{array} { r } { \mathcal { L } = 0 . 1 6 \mathcal { L } _ { \mathrm { p i x } } + 0 . 8 \mathcal { L } _ { \mathrm { e d g e } } . } \end{array}\tag{10}
$$

The two weights follow the original supervision setup. The region loss backpropagates through the fusion and prediction pathway to optimize matching descriptors, the order gate, and the strength gate. The boundary loss acts on the RGB boundary branch before interaction. The unbalanced-transport objective is used to solve internal correspondences, while the full network is trained end-to-end using the localization supervision above.

## 4 Experiments

## 4.1 Data, training, and evaluation

Training data follow the Protocol-CAT configuration reported by IMDL-BenCo (Ma et al., 2024b) [26]. This protocol builds a training set from manipulated images collected from multiple sources [2, 10] and uses source-balanced sampling to reduce training bias caused by diferences in source-set sizes.

PUMA and the control group use the same data split, initialization source, supervision, and efective batch size, with the random seed fixed to 42. Both configurations are trained for 150 epochs, and final results are computed from the epoch-150 checkpoint. We use AdamW with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay of 0.05. The learning rate is linearly warmed up for the first two epochs and then decayed using a cosine schedule, reaching $5 \times 1 0 ^ { - 7 }$ at the end of training. The efective batch size is 64, consisting of two GPUs, a per-GPU batch size of 16, and two gradient-accumulation steps.

At test time, pixel-level F1 uses a fixed threshold of 0.5. For each image, pixels with predicted probability greater than 0.5 are classified as manipulated, and positive-class F1 is computed as $2 T P / ( 2 T P + F P + F N )$ . If the denominator is zero, meaning both the ground truth and prediction are empty masks, that image’s F1 is set to zero. Each dataset’s metric is the arithmetic mean of per-image F1 scores, not a single F1 computed after aggregating pixels across the entire dataset. This protocol prevents true negatives from large background regions from dominating the result and is suitable for localization tasks in which manipulated regions occupy a small fraction of an image.

## 4.2 Comparison with existing methods

PUMA obtains higher pixel-level F1 scores than the corresponding published results listed in Table 1 on all four test sets: CASIA1.0, COVERAGE, Columbia, and NIST16. The gaps are more pronounced on COVERAGE and NIST16. This comparison suggests that local matching and forensic-context interaction are useful under the evaluation protocol adopted here.

Figure 2 presents representative examples from the four test sets to compare how methods handle the shape of manipulated regions and false positives in the background. Across several examples, PUMA recovers target boundaries more completely and reduces scattered responses outside the targets compared with the baselines. These observations are consistent with the cross-dataset quantitative results in Table 1, suggesting that local matching and sequential feature interaction help organize fine-grained forensic information.

Table 1: Published results of representative manipulation-localization methods and results for PUMA [22, 44].
<table><tr><td>Method</td><td>CASIA1.0 COVERAGE</td><td></td><td>Columbia</td><td>NIST16</td></tr><tr><td>MVSS-Net [3]</td><td>0.5486</td><td>0.5188</td><td>0.7322</td><td>0.2876</td></tr><tr><td>PSCC-Net [12]</td><td>0.5472</td><td>0.3791</td><td>0.8640</td><td>0.3681</td></tr><tr><td>IML-ViT [45]</td><td>0.7392</td><td>0.5364</td><td>0.9334</td><td>0.1100</td></tr><tr><td>CAT-Net [11]</td><td>0.8140</td><td>0.4272</td><td>0.9151</td><td>0.3787</td></tr><tr><td>TruFor [8]</td><td>0.8176</td><td>0.4573</td><td>0.8845</td><td>0.3480</td></tr><tr><td>Mesorch [46]</td><td>0.8398</td><td>0.5862</td><td>0.8903</td><td>0.3921</td></tr><tr><td>PUMA (ours)</td><td>0.8600</td><td>0.6791</td><td>0.9358</td><td>0.5543</td></tr></table>

## 4.3 Robustness to post-processing

The evaluation covers four test sets and four image perturbations: JPEG compression, Gaussian blur, Gaussian noise, and Gamma correction. Methods are evaluated on paired test images with the same perturbation types and strengths. Each test image is decoded once, and a shared perturbed version is generated using the specified transformation. Perturbations are applied to the original RGB image before each model’s own resizing and normalization pipeline. Gaussian noise is generated with a fixed seed determined by image identifier, perturbation type, and strength, ensuring reproducible paired noisy inputs across methods. All methods use the fixed threshold of 0.5 to compute per-image pixel-level F1, which is then averaged arithmetically across images. Unperturbed results serve as the reference.

Figure 3 shows that PUMA achieves higher per-image pixel-level F1 than PSCC-Net and MVSS-Net under most perturbation conditions across the four test sets. It also maintains relatively stable localization under multiple degradations in Columbia, under compression and noise changes in NIST16- SCI, and under noise and Gamma correction in CASIA1.0. Its advantage narrows under strong JPEG compression and larger blur kernels on COVERAGE and under strong blur on CASIA1.0. Overall, however, the results indicate that PUMA adapts well to a range of post-processing changes.

This behavior is consistent with the interaction mechanism. Unbalanced matching adjusts transport mass according to local match quality and preserves pre-normalization mass deviations as cues for subsequent gating. The order and strength gates use these cues to select a propagation order at each position and control the amount of neighborhood information written back. Feature interaction can therefore adapt to local forensic evidence. When compression, blur, noise, or intensity mapping changes local traces, the model can still use semantic guidance and forensic matching to select and propagate relevant information, reducing interference from mismatched evidence. This provides a mechanism-level explanation for PUMA’s advantage over the two comparison methods under most test conditions.

## 4.4 Interaction and performance analysis

Table 1 shows that PUMA exceeds the listed published baseline values on all four test sets, with more pronounced diferences

# Manipulated image Ground truth PSCC-Net IML-ViT MVSS-Net PUMA CASIA1.0 CASIA1.0 COVERAGE COVERAGE 6 Columbia Columbia I<sub>S</sub>T<sup>16</sup> I<sub>S</sub>T<sup>16</sup>

Figure 2: Qualitative comparison on four test sets. Each row shows the input image, ground-truth mask, and predictions from PSCC-Net, IML-ViT, MVSS-Net, and PUMA, in that order. Predictions are binarized using a fixed threshold of 0.5.

on COVERAGE and NIST16. Because these published results were not re-measured in this study using unified weights and data pipelines, the table serves primarily as a reference to existing benchmark values. Structurally, PUMA organizes noise-feature values through semantic correspondences rather than replacing forensic representations directly with RGB features, allowing semantic guidance and forensic evidence to jointly contribute to prediction.

Figure 4 shows Grad-CAM overlays for representative PUMA examples from the four test sets. Heatmaps are computed from the final backbone features, using the predicted response within the ground-truth manipulated region as the visualization target.

The high-response regions cover manipulated targets or their nearby boundaries in several examples, indicating that the deep features focus on spatial regions relevant to localization. Together with the binary-mask comparisons in Figure 2, these visualizations complement the quantitative and qualitative localization results from the perspective of response distribution and provide an intuitive indication that local forensic information contributes to prediction.

The central advantage of unbalanced matching is that it describes local correspondence separately from total matching mass. When overall candidate matching is weak, soft marginal constraints allow transported mass to deviate from the reference mass. The pre-normalization mass state communicates this deviation to the gating module, supplementing the matching information absent from relative aggregation weights. Consequently, the propagation process can use relative neighborhood relationships while adaptively adjusting interaction strength according to diferences in local matching state.

The sequential module further decouples propagation order from update strength. The two operator compositions organize local information diferently; the order gate adjusts their mixture position by position, while the strength gate controls how much neighborhood increment is written back. Convex combinations and the residual path preserve the contribution of the original feature. The stability of Columbia under compression and blur, and the stable compression response on NIST16, are consistent with the design goal of retaining the original representation while adjusting neighborhood propagation as evidence changes.

![](images/39f359b9bbc2bb74d235027b04553cf1f774d0e17fd7308a8a53fe7fec790c76.jpg)  
Figure 3: Localization robustness comparison among PUMA, PSCC-Net, and MVSS-Net across four test sets and four image perturbations. Rows denote perturbation types and columns denote test sets. The vertical axis is the arithmetic mean of per-image pixel-level F1 at a fixed threshold of 0.5; “None” denotes no perturbation.

## 5 Conclusion

PUMA addresses the interaction between semantic and forensic features in image manipulation localization by integrating spatially constrained unbalanced matching, preservation of mass states, and probability-constrained sequential propagation within a unified local interface. The matching module describes both neighborhood correspondences and deviations in transported mass. The sequential module then adjusts the two propagation orders and their update strength position by position. It enhances the noise representation through a residual update while preserving the row-stochastic property of spatial aggregation weights. The design retains the original two-branch backbone and prediction pathway and provides an efective way to organize forensic information using local matching states.

Across multiple public manipulation-localization benchmarks and common image post-processing perturbations, PUMA achieves competitive localization accuracy under several test conditions, with gains that vary by data distribution and perturbation type. Under clean-image and JPEG-compression conditions, the retrained control performs better on average across datasets, whereas PUMA shows a more pronounced advantage under strong smoothing degradations. The experiments indicate that local mass states preserved by unbalanced matching and the sequential gating mechanism can efectively adjust forensic-information propagation under specific image degradations. Diferent perturbations, however, place diferent demands on interaction strategies and require targeted evaluation and optimization.

![](images/30e45e8c4a0c4fd61912579e9f6becf7ddf1275e3404dd3ef676601e0fc6879b.jpg)  
Figure 4: Grad-CAM visualizations for PUMA on representative samples from CASIA1.0, COVERAGE, Columbia, and NIST16. Colors indicate the relative response strength of the final-layer features for manipulation-region prediction.

## References

[1] Edoardo Daniele Cannas, Sara Mandelli, Nataša Popović, Ayman Alkhateeb, Alessandro Gnutti, Paolo Bestagini, and Stefano Tubaro. Is jpeg ai going to change image forensics? arXiv preprint arXiv:2412.03261, 2024. URL: https: //arxiv.org/abs/2412.03261.

[2] Rui Chen, Bin Liu, Changtao Miao, Xinghao Wang, Yi Li, Tao Gong, Qi Chu, and Nenghai Yu. Training-free incontext forensic chain for image manipulation detection and localization. arXiv preprint arXiv:2510.10111, 2025. URL: https://arxiv.org/abs/2510.10111.

[3] Xinru Chen, Chengbo Dong, Jiaqi Ji, Juan Cao, and Xirong Li. Image manipulation detection by multi-view multi-scale supervision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14185–14193, 2021.

[4] Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard. Scaling algorithms for unbalanced optimal transport problems. Mathematics of Computation, 87(314):2563– 2609, 2018.

[5] Davide Cozzolino and Luisa Verdoliva. Noiseprint: A CNNbased camera model fingerprint. IEEE Transactions on Information Forensics and Security, 15:144–159, 2020.

[6] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, volume 26, 2013.

[7] David Fischinger and Martin Boyer. DF-Net: The digital forensics network for image forgery detection. arXiv preprint

arXiv:2503.22398, 2025. DOI: 10.5281/zenodo.8214996. URL: https://arxiv.org/abs/2503.22398.

[8] Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa Verdoliva. TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20606–20615, 2023.

[9] Xuefeng Hu, Zhihan Zhang, Zhenye Jiang, Syomantak Chaudhuri, Zhenheng Yang, and Ram Nevatia. SPAN: Spatial pyramid attention network for image manipulation localization. In European Conference on Computer Vision (ECCV), pages 312–328, 2020.

[10] Shan Jia, Mingzhen Huang, Zhou Zhou, Yan Ju, Jialing Cai, and Siwei Lyu. Autosplice: A text-prompt manipulated image dataset for media forensics. arXiv preprint arXiv:2304.06870, 2023. URL: https://arxiv.org/abs/2304.06870.

[11] Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee. CAT-Net: Compression artifact tracing network for detection and localization of image splicing. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 375–384, 2021.

[12] Xiaohong Liu, Yaojie Liu, Jun Chen, and Xiaoming Liu. PSCC-Net: Progressive spatio-channel correlation network for image manipulation detection and localization. IEEE Transactions on Circuits and Systems for Video Technology, 32(11):7505–7517, 2022.

[13] Owen Mayer and Matthew C. Stamm. Forensic similarity for digital images. arXiv preprint arXiv:1902.04684, 2019.

DOI: 10.1109/TIFS.2019.2924552. URL: https://arxiv. org/abs/1902.04684.

[14] Ju-Hyeon Nam, Dong-Hyun Moon, and Sang-Chul Lee. M2SFormer: Multi-spectral and multi-scale attention with edge-aware dificulty guidance for image forgery localization. arXiv preprint arXiv:2506.20922, 2025.

[15] Michael E. Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyré. Sinkformers: Transformers with doubly stochastic attention. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), volume 151 of PMLR, pages 3515–3530, 2022.

[16] Kostas Triaridis, Konstantinos Tsigos, and Vasileios Mezaris. MMFusion: Combining image forensic filters for visual manipulation detection and localization. arXiv preprint arXiv:2312.01790, 2023. URL: https://arxiv.org/abs/ 2312.01790.

[17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

[18] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations (ICLR), 2018.

[19] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7794–7803, 2018.

[20] Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan. ManTra-Net: Manipulation tracing network for detection and localization of image forgeries with anomalous features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9543–9552, 2019.

[21] Jianwei Yang, Chunyuan Li, Xiyang Dai, and Jianfeng Gao. Focal modulation networks. In Advances in Neural Information Processing Systems, volume 35, 2022.

[22] Jiwei Zhang, Wenbo Feng, Siwei Wang, Feifei Kou, Haoyang Yu, and Shaozhang Niu. Collaborative transformers with multilevel forensic attention for image manipulation localization. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026. DOI: 10.1609/aaai.v40i15.38250. URL: https://ojs. aaai.org/index.php/AAAI/article/view/38250.

[23] Keyang Zhang, Chenqi Kong, Hui Liu, Bo Ding, Xinghao Jiang, and Haoliang Li. Propose and rectify: A forensics-driven MLLM framework for image manipulation localization. arXiv preprint arXiv:2508.17976, 2025. URL: https://arxiv.org/abs/ 2508.17976.

[24] Xuanyu Zhang, Runyi Li, Jiwen Yu, Youmin Xu, Weiqi Li, and Jian Zhang. EditGuard: Versatile image watermarking for tamper localization and copyright protection. arXiv preprint arXiv:2312.08883, 2023. URL: https://arxiv.org/abs/ 2312.08883.

[25] Kaixiang Zhao, Tianrun Yu, Aoxu Zhang, Junhao Su, Porter Jenkins, and Amanda Hughes. FRAME: Forensic routing and adaptive multi-path evidence fusion for image manipulation detection. arXiv preprint arXiv:2605.12826, 2026. URL: https: //arxiv.org/abs/2605.12826.

[26] Xiaochen Ma, Xuekang Zhu, Lei Su, Bo Du, Zhuohang Jiang, Bingkui Tong, Zeyu Lei, Xinyu Yang, Chi-Man Pun, Jiancheng Lv, and Jizhe Zhou. IMDL-BenCo: A comprehensive benchmark and codebase for image manipulation detection & localization. Advances in Neural Information Processing Systems, 37, Datasets and Benchmarks Track, 2024.

[27] Cheng Chen, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Wen Wang, Yihao Meng, Hanlin Wang, Yixuan Li, Jiacheng Wei, Zhenshan Tan, Yanhong Zeng, Yujun Shen, Guosheng Lin, and Fayao Liu. DiVA: Enabling interactive digital life simulation via video models. arXiv preprint arXiv:2609.13830, 2026.

[28] Xinjue Hu, Zhenshan Tan, Chi Wang, Xiang Zhang, and Zhangjie Fu. High-resolution image steganalysis. IEEE Transactions on Dependable and Secure Computing, 2026.

[29] Zhenshan Tan, Chenhan Lu, Yuxiang Huang, Ziwen He, Xiang Zhang, Yuzhe Sha, Xianyi Chen, Tianrun Chen, and Zhangjie Fu. Semantic manipulation localization. arXiv preprint arXiv:2604.10132, 2026.

[30] Zhenshan Tan, Xiaodong Gu, and Qingrong Cheng. Co-saliency detection with two-stage co-attention mining and individual calibration. Engineering Applications of Artificial Intelligence, 2024.

[31] Zhenshan Tan, Cheng Chen, Keyu Wen, Yuzhuo Qin, and Xiaodong Gu. A unified two-stage group semantics propagation and contrastive learning network for co-saliency detection. In IEEE International Conference on Multimedia and Expo, 2022.

[32] Qingrong Cheng, Zhenshan Tan, Keyu Wen, Cheng Chen, and Xiaodong Gu. Semantic pre-alignment and ranking learning with unified framework for cross-modal retrieval. IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[33] Cheng Chen, Zhenshan Tan, Qingrong Cheng, Xin Jiang, Qun Liu, Yudong Zhu, and Xiaodong Gu. UTC: A unified transformer with inter-task contrastive learning for visual dialog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[34] Cheng Chen, Jiacheng Wei, Tianrun Chen, Chi Zhang, Xiaofeng Yang, Shangzhan Zhang, Bingchen Yang, Chuan-Sheng Foo, Guosheng Lin, Qixing Huang, and Fayao Liu. CADCrafter: Generating computer-aided design models from unconstrained images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[35] Cheng Chen, Xiaofeng Yang, Fan Yang, Chengzeng Feng, Zhoujie Fu, Chuan-Sheng Foo, Guosheng Lin, and Fayao Liu. Sculpt3D: Multi-view consistent text-to-3D generation with sparse 3D prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[36] Li Zhang, Zhenshan Tan, Guo Zhang, Wen Zhang, and Zhijiang Li. Learn more and learn usefully: Truncation compensation network for semantic segmentation of high-resolution remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 2024.

[37] Cheng Chen, Jerry Bai, Jiacheng Wei, Boyu Chen, Xiaoji Zheng, Fan Wu, Minghao Yang, Tianrun Chen, Ruibo Li, Xiaoyu Yue, Xiaoyang Guo, Yixiao Ge, Guosheng Lin, and Fayao Liu. AnyWorld: Factorized egocentric world models for cross-embodiment generalization. In Conference on Robot Learning (CoRL), 2026.

[38] Zhenshan Tan, Cheng Chen, Keyu Wen, Qingrong Cheng, and Zhangjie Fu. Bridging spatiotemporal feature gap for video salient object detection. Knowledge-Based Systems, 2024.

[39] Zhenshan Tan, Cheng Chen, and Xiaodong Gu. Triplet spatiotemporal aggregation network for video saliency detection. In IEEE International Conference on Multimedia and Expo (ICME), 2023.

[40] Yi Tan, Yanbin Hao, Xiangnan He, Yinwei Wei, and Xun Yang. Selective dependency aggregation for action classification. In Proceedings of the 29th ACM International Conference on Multimedia (ACM MM), 2021.

[41] Yi Tan, Zhaofan Qiu, Yanbin Hao, Ting Yao, and Tao Mei. Selective volume mixup for video action recognition. International Journal of Computer Vision, 2026.

[42] Yi Tan, Yanbin Hao, Hao Zhang, Shuo Wang, and Xiangnan He. Hierarchical hourglass convolutional network for eficient video classification. In Proceedings of the 30th ACM International Conference on Multimedia (ACM MM), 2022.

[43] Yi Tan, Beier Zhu, Sirui Chen, Cheng Chen, Xingyu Zhu, Yuan Zhou, Zhenshan Tan, Tianrun Chen, and Hanwang Zhang. RouteREPA: Routed representation alignment for just image transformers. OpenReview, 2026.

[44] ProgrameThinking. Co-Transformers: Oficial repository and updated benchmark results. GitHub. Accessed November 8, 2026. https://github.com/ProgrameThinking/ Co-Transformers.

[45] Xiaochen Ma, Bo Du, Zhuohang Jiang, Xia Du, Ahmed Y. Al Hammadi, and Jizhe Zhou. IML-ViT: Benchmarking image

manipulation localization by vision transformer. arXiv preprint arXiv:2307.14863, 2023.

[46] Xuekang Zhu, Xiaochen Ma, Lei Su, Zhuohang Jiang, Bo Du, Xiwen Wang, Zeyu Lei, Wentao Feng, Chi-Man Pun, and Ji-Zhe Zhou. Mesoscopic insights: Orchestrating multi-scale and hybrid architecture for image manipulation localization. Proceedings of the AAAI Conference on Artificial Intelligence, 39(10):11022–11030, 2025.

[47] Junke Wang, Zuxuan Wu, Jingjing Chen, Xintong Han, Abhinav Shrivastava, Ser-Nam Lim, and Yu-Gang Jiang. ObjectFormer for image manipulation detection and localization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2364–2373, 2022.

[48] Zeqin Yu, Jiangqun Ni, Yuzhen Lin, Haoyi Deng, and Bin Li. DifForensics: Leveraging difusion prior to image forgery detection and localization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12765–12774, 2024.

[49] Belhassen Bayar and Matthew C. Stamm. Constrained convolutional neural networks: A new approach towards general purpose image manipulation detection. IEEE Transactions on Information Forensics and Security, 13(11):2691–2706, 2018.

[50] Yue Wu, Wael Abd-Almageed, and Prem Natarajan. BusterNet: Detecting copy-move image forgery with source/target localization. In Computer Vision – ECCV 2018, pages 170–186. Springer, 2018.