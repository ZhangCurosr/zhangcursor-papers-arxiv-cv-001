# 2D Gum-Net: Unsupervised Dense Elastic Image Registration via Symmetric Correlation and Spectral Pooling

Aarya Bhave Google aaryabhave@google.com

Anushka Ajay Bhave Google nushiva@google.com

## Abstract

Dense non-rigid registration, the recovery of a pixel-wise deformation field that warps one image into geometric agreement with another, is fundamental across medical imaging, structural biology, and correspondence estimation, yet learningbased approaches face a severe bottleneck: ground-truth deformation fields are notoriously difficult to obtain for real data. We present 2D Gum-Net, an end-toend architecture that predicts dense elastic deformation fields fully unsupervised. We transfer the Geometric Unsupervised Matching Network (Gum-Net), originally developed for 3D subtomogram alignment in cryo-electron tomography, to 2D elastic registration: all three modules are redesigned in 2D, and the rigid transformation head is replaced by a Dense Spatial Transformer that regresses a sparse control-point grid, lifted to a full-resolution field Φ by bicubic interpolation. Downsampling uses DCT spectral pooling, implemented as frozen-basis matrix multiplications (≈50× GPU speedup), preserving the fine, quasi-periodic structure that max and average pooling destroy. We evaluate on contact-based fingerprints, a challenging testbed for this task: locally self-similar ridge patterns make correspondence maximally ambiguous, skin pressed against a sensor generates authentic large-displacement elastic distortion, and minutiae act as precise landmarks through which an independent matcher externally verifies the topological correctness of predicted fields. Trained purely on synthetic impressions and evaluated zero-shot on real benchmarks, our method improves structural correlation by up to 15.7% on FVC2004 and raises genuine match scores under an independent matcher while leaving imposter scores essentially unchanged. This suggests that the predicted deformations restore true structure without fabricating correspondences.

## 1 Introduction

Dense non-rigid registration is the problem of recovering a pixel-wise deformation field Φ such that a moving image $\bar { S _ { b } } ^ { \bar { ( ) } }$ , resampled through the field, comes into geometric agreement with a reference image $\bar { S _ { a } }$ . Formally, one seeks

$$
\Phi ^ { * } = \arg \operatorname* { m i n } _ { \Phi } ~ \mathcal { D } ( S _ { a } , ~ S _ { b } \circ \Phi ) + \lambda \mathcal { R } ( \Phi ) ,\tag{1}
$$

where D measures structural dissimilarity and R regularizes the field toward physically plausible, smooth deformations. This formulation underlies deformable medical image registration, dense correspondence estimation in natural images, and structural alignment in biology [1–4]. Classical methods solve Equation (1) by iterative per-pair optimization, which is accurate but slow and sensitive to initialization. Learning-based methods amortize this optimization into a feed-forward predictor of Φ [2, 5–7], but most require supervised ground-truth deformation fields, which for real data essentially never exist and must be simulated [7, 8]. Unsupervised dense registration, which train from image pairs alone using the similarity functional itself as the learning signal, removes this bottleneck [1]. A second, less discussed obstacle is architectural: correspondence information lives in fine image structure, yet max pooling retains only localized maximal responses and average pooling attenuates and blurs local structure [9–11], degrading the very signal a registration network must correlate.

![](images/ac3e5aa5d9d0127da999b5881d8fb06b0518c6fb9b259f75d25764102a142a13.jpg)  
Figure 1: 2D Gum-Net pipeline. The model is unsupervised and feed-forward. It inputs a geometrically distorted image $S _ { b }$ and a reference image $S _ { a }$ (here a fingerprint impression and its template) and outputs the un-warped image $S _ { \hat { b } }$ in geometric agreement with $S _ { a }$ . The parameters of this transformation are denoted by $( \Delta x , \Delta y )$ . The dashed line denotes that weights are shared between the two feature extractors.

In this work we transfer an unsupervised registration paradigm across both domain and dimensionality. The Geometric Unsupervised Matching Network (Gum-Net) of Zeng et al. [4] aligns 3D subtomograms in cryogenic electron tomography (cryo-ET) without any labels, reducing alignment error by 40–50% while accelerating the overall process 70–110×. Its mechanism, dense symmetric correlation of deep feature maps followed by transformation regression, is largely modality-agnostic [2, 4]. We hypothesize that the same mechanism, redesigned in two dimensions, solves general 2D elastic registration. Our adaptation (Figure 1) differs from the original in two fundamental ways: all three modules are rebuilt for 2D inputs, and, more consequentially, the rigid transformation head is replaced with a Dense Spatial Transformer, which regresses a sparse control-point grid of $2 G ^ { 2 }$ local displacements lifted to a full-resolution dense field Φ by bicubic interpolation. This changes the transformation class from rigid motions to a smooth, expressive family of elastic warps, keeping the field smooth by construction and the parameter count small.

We evaluate this framework on contact-based fingerprint alignment, and we argue that fingerprints are a deliberately demanding testbed for dense elastic registration rather than merely an application. First, ridge patterns are quasi-periodic and locally self-similar: sliding a patch by one ridge wavelength produces a near-perfect local match that is globally wrong, so local matching faces an extreme aperture problem and a registration network succeeds only by integrating long-range spatial context. Second, the deformations are authentic and free: elastic skin pressed and twisted against a rigid sensor produces genuine large-displacement, non-linear warps in every acquisition [12–14], on which rigid and affine models demonstrably fail. Third, the domain supplies an independent verification oracle. Registration evaluation is usually circular, since alignment quality is measured with the same class of similarity functional that was optimized. Fingerprint minutiae, by contrast, are precise point landmarks consumed by an external matching algorithm: a field that restores true structure raises genuine match scores while leaving imposter scores unchanged, whereas a field that hallucinates or folds structure is immediately penalized. Fourth, the correspondence signal is concentrated in a narrow fine-scale band (the ridge frequency), so fingerprints isolate the causal effect of information-preserving downsampling: pooling that destroys the ridge band destroys registration. Finally, procedural synthetic generators provide unlimited template–impression training pairs, leaving real benchmarks untouched for zero-shot transfer evaluation.

In summary, the primary contributions of this work are as follows:

• Cross-domain transfer of unsupervised registration: We transfer the Gum-Net paradigm from volumetric cryo-ET alignment to 2D elastic registration, redesigning all three modules and replacing rigid regression with a Dense Spatial Transformer that predicts a control-pointparameterized dense field Φ, trained fully unsupervised, circumventing the deformation groundtruth bottleneck.

• Information-preserving spectral downsampling: We implement DCT spectral pooling as frozen-basis matrix multiplications (≈10× GPU speedup over FFT-based extraction) and ablate it against max and average pooling to isolate its causal effect on registration quality.

• Externally validated evaluation: We evaluate predicted fields with structural correlation and through an independent minutiae matcher acting as a topological-correctness oracle, showing that alignment raises genuine similarity while leaving imposter similarity unchanged.

We deliberately evaluate depth-first on a single demanding domain; downstream minutiae-matching results serve as a probe of topological correctness and genuine-score separability, not as a benchmark of a complete recognition system.

## 2 Related work

## 2.1 Dense non-rigid registration and geometric matching

Learned dense correspondence is commonly built on feature correlation and cost-volume constructions that score every spatial location of one feature map against every location of the other, from which displacement fields are regressed [2, 3]. Spatial Transformer Networks [15] established the differentiable-resampling machinery for training end-to-end through a predicted warp; our Dense Spatial Transformer extends this from parametric transformations to a control-point-parameterized dense field. In medical imaging, VoxelMorph [1] showed dense registration networks can be trained unsupervised, with an image-similarity loss plus smoothness penalty replacing deformation ground truth; in effect, classical intensity-based registration amortized into a feed-forward predictor. In the fingerprint domain, dense registration and displacement-field regression have been studied classically and with deep networks [5–7, 16], but predominantly supervised, trained on simulated or annotated distortion fields. Fully unsupervised estimation of dense non-linear deformation fields remains comparatively unexplored outside the medical domain, and is the setting this work addresses.

## 2.2 Elastic registration in fingerprint biometrics

Our testbed has its own registration literature. Traditional fingerprint systems are minutiae-based: ridge endings and bifurcations are matched after estimating a geometric transformation between point sets [13], typically rigid or locally affine, via Hough-style voting [17] and RANSAC refinement with local structural descriptors [18, 19]. These approaches assume near-rigid geometry and depend on reliable minutiae extraction, which degrades under distortion; since real impressions exhibit non-linear stretching and torque, elastic deformation remains a primary source of false rejections [12]. Classical elastic registration addressed this directly: Bazen and Gerez fit thin-plate splines to minutiae correspondences [12], while Ross, Dass, and Jain estimate average deformation models [20] and non-linear warps from ridge curve correspondences [21]. Modern learned pipelines such as C2CL [8] and contrastive cross-domain approaches [22] retain explicit alignment modules (e.g., TPS-based rectification), but optimize them through pretrained identity networks or supervised metric learning.

Our method is best understood as the learned, amortized analogue of this classical elastic lineage: a control-point grid lifted by smooth interpolation is structurally a free-form deformation model, a close cousin of TPS. But where classical methods solve a per-pair optimization from explicitly extracted correspondences, inheriting the fragility of the extraction step, our network amortizes the inference into a single forward pass from raw pixels, learning the correspondence function itself without supervision.

## 2.3 Spectral representations and pooling

Frequency-domain techniques have long served the testbed domain for enhancement, translationinvariant matching, and compact descriptors [23–27], but for representation or rigid alignment rather than for learning non-rigid correspondence. Separately, spectral pooling (subsampling by truncating high-frequency transform coefficients) preserves substantially more spatial information per parameter than max or average pooling, with DFT [9], DCT [11], and Hartley [10] variants. This property is not domain-specific: it is a principled alternative to spatial subsampling whenever deformation must be estimated from fine structure, which is exactly the regime our testbed maximizes.

## 2.4 Unsupervised learning in the testbed domain

Unsupervised learning has been explored for fingerprints through RBMs, autoencoders, and GANbased enhancement [28–30], showing that structural priors can be learned without annotation, while deep matchers, whether CNN-based [31–34] or transformer-based [35, 36], learn identitydiscriminative features without explicitly modeling elastic deformation, which remains a major cause of matching errors. In contrast, we learn the deformation field itself, densely and without labels.

## 3 Method

Our model is shown in Figure 1. It takes two grayscale images: the reference (fixed) image $S _ { a }$ and the moving image $S _ { b }$ to be registered onto it. In our evaluation domain these are a fingerprint master template and an impression of the same finger, pairs that synthetic generators produce natively (Section 4). Weight-shared feature extractors produce feature maps $v _ { a }$ and $v _ { b }$ , which a Siamese matching module converts into two correlation maps: at spatial coordinate $( i , j ) , c _ { a b }$ holds the similarity between $v _ { a }$ at $( i , j )$ and all features of $v _ { b } .$ , with $c _ { b a }$ defined symmetrically. Both maps are processed by identical network branches, flattened, and concatenated to estimate the transformation parameters. The spatial transformation is realized by a dense deformation field $\Phi \in \mathbb { R } ^ { H \times W \times 2 } ;$ however, to ensure smoothness and reduce complexity, we do not regress this field directly. Instead it is parameterized by a sparse control-point grid $\mathbf { P } ^ { \prime } \in \mathbb { R } ^ { 2 \times G \times G }$ : for each control point $P _ { i } , i \in \{ 1 , . . . , G ^ { 2 } \}$ the network regresses a localized translation $\Delta { \bf p } _ { i } = ( \Delta x _ { i } , \Delta y _ { i } )$ , and the resulting $2 G ^ { 2 }$ parameters are upsampled via bicubic interpolation to the full-resolution field Φ. The objective is to find the field that maximizes structural overlap between $S _ { a }$ and the warped $S _ { \hat { b } }$ , completely unsupervised: a soft Dice alignment loss [37] plus a smoothness regularizer (Section 3.4) is back-propagated through the differentiable warp.

## 3.1 Feature extraction module

Popular feature-extraction backbones such as DenseNet [38], InceptionNet [39], and ResNet [40] rely on max or average pooling, whereas spectral subsampling preserves considerably more spatial information per parameter while allowing flexible output dimensionality [9]. Two-dimensional spectral pooling can be based on the DFT [9], DCT [11], or Hartley transform [10]; among these, the DCT has recurred in fingerprint pipelines [23, 24] and, relying only on real-valued cosine functions rather than complex exponentials, produces output in the real domain R.

We implement our own 2D spectral pooling layer based on the 2D Type-II DCT. Our implementation achieves a significant computational speed-up by decomposing the 2D transform into sequential 1D transforms and pre-computing the spectral transformation bases via an identity-matrix mapping. This reframes the transform as a pure General Matrix Multiply (GEMM) operation, achieving a 50× speedup on CUDA devices (details in Section C).

Rather than evaluating the standard separable Type-II DCT summation [41] (restated in Section C) directly, our implementation computes the full 2D transform as a single bilinear matrix multiplication:

$$
Y = W _ { H } X W _ { W } ^ { \top }\tag{2}
$$

Here $W _ { H } = { \mathcal { D } } _ { 1 D } ( I _ { H } )$ and ${ \cal W } _ { W } = { \cal D } _ { 1 D } ( { \cal I } _ { W } )$ , where $\mathcal { D } _ { 1 D }$ denotes the one-dimensional Type-II DCT operator, while $\dot { I _ { H } } \in \mathbb { R } ^ { H \times H }$ and $I _ { W } \in \mathbb { R } ^ { W \times W }$ denote identity matrices. The inverse of the 2D Type-II DCT is well defined as the 2D Type-III DCT [41]; thus the pooled and filtered representation in the frequency domain is transformed back to the spatial domain through the 2D Type-III DCT as the output of the layer.

The layer thus shifts dimensionality reduction into the frequency domain: the Type-II DCT concentrates the structurally significant low-frequency content into the top-left of the frequency matrix; downsampling is a simple crop to the top-left $h _ { f r e q } \times w _ { f r e q }$ block; and the Type-III DCT reconstructs the spatial feature map at a chosen $h _ { o u t } \times w _ { o u t }$ . We maintain $h _ { f r e q } \leq h _ { o u t }$ and $w _ { f r e q } \leq w _ { o u t }$ , since the converse would discard retained frequencies during the inverse transform and render the crop wasteful. In each convolutional block we use a cropping ratio $h _ { f r e q } / h _ { o u t } \approx w _ { f r e q } / w _ { o u t } \approx 0 . 8 5$ avoiding the aggressive reduction of $2 \times 2$ max or average pooling, which discards 75% of the spacial samples per stage. Given an input pair $S _ { a } , S _ { b } \in \mathbb { R } ^ { 1 \times 1 \breve { 9 } 2 \times 1 9 2 }$ , the feature extraction module derive feature maps $v _ { a } , v _ { b } \in \mathbb { R } ^ { 5 1 2 \times 1 4 \times 1 4 }$

![](images/c042fe6700a5cc8140837ddd72ec95031b16f4bbdf8f56d467ef8f8f33afc1e4.jpg)  
Figure 2: Reconstructions comparing max, average, and DCT spectral pooling on a fingerprint impression. DCT spectral pooling superiorly preserves fine spatial structure and supports arbitrary output dimensions.

## 3.2 Siamese matching module

While matching is often treated as an independent post-processing step [42–45, 3], the 2D correlation layer of [2] integrates matching information directly: acting as a normalized cross-correlation $\overset { \vartriangle } { \boldsymbol { G } ^ { \prime } } : \mathbb { R } ^ { \boldsymbol { C } \times \dot { \boldsymbol { H } } \times \boldsymbol { W } } \times \mathbb { R } ^ { \boldsymbol { C } \times \boldsymbol { H } \times \boldsymbol { W } }  \mathbb { R } ^ { ( \boldsymbol { H } \boldsymbol { W } ) \times \boldsymbol { H } \times \boldsymbol { W } }$ , it flattens one feature map to $\mathbb { R } ^ { C \times N } \left( N = H W \right)$ and computes, for each spatial feature of the other map, a dot product against all N flattened features. While effective, this breaks all spatial axes of one input while preserving the other. For precise alignment of fine, repeating structure, preserving the spatial hierarchy of both images is critical, since the reference and moving images differ fundamentally in their photometric properties. Inspired by the 3D matching approach in Gum-Net [4], we therefore propose a symmetric 2D Siamese correlation layer that computes two maps, $C _ { a b }$ and $C _ { b a }$ , processed in parallel.

Given the two outputs of the feature extraction module (Section 3.1), $v _ { a } \in \mathbb { R } ^ { 5 1 2 \times 1 4 \times 1 4 }$ (reference) and $v _ { b } \in \mathbb { R } ^ { 5 1 2 \times 1 4 \times 1 4 }$ (moving), we obtain $C _ { a b } , C _ { b a } \in \mathbb { R } ^ { 1 9 6 \times 1 4 \times 1 4 }$

The first correlation map, $C _ { a b }$ , preserves the spatial coordinates of $v _ { a }$ . Each element at channel k (where $k \in \{ 1 , . . . , N \}$ indexes a flattened spatial location of $v _ { b } )$ and spatial location $( h , w )$ of $v _ { a }$ is the normalized dot product, with $C _ { b a }$ defined symmetrically:

$$
( C _ { a b } ) _ { k , h , w } = \frac { \left. v _ { a _ { : , h , w } } , v _ { b _ { : k } } \right. } { \sqrt { \sum \left. v _ { a } , v _ { b } \right. } } , \qquad ( C _ { b a } ) _ { k , h , w } = \frac { \left. v _ { b _ { : , h , w } } , v _ { a _ { : k } } \right. } { \sqrt { \sum \left. v _ { b } , v _ { a } \right. } }\tag{3}
$$

The correlation maps $C _ { a b } , C _ { b a }$ are fed into a pseudo-Siamese network of contiguous convolutional layers. Each map is convolved separately, then the branch outputs are concatenated and flattened to predict the dense deformation field.

## 3.3 Unsupervised non-linear alignment module

Unlike the original Gum-Net [4], which regresses a rigid 3D transformation suitable for macromolecular structures, our architecture predicts a dense deformation field Φ. The non-linear alignment module translates the correlation representations from Section 3.2 into a pixel-wise displacement field in two stages: control-point regression and dense flow-field generation.

Control point regression: Each correlation branch is reduced by six successive 1024-channel valid convolutions to spatial size $2 \times 2$ and flattened to $\mathbb { R } ^ { 4 0 9 6 }$ ; the two branches are concatenated into a joint representation $v _ { c o r r } \in \mathbb { R } ^ { 8 1 9 2 }$ that captures the geometric consensus (full architecture in Section $\mathbf { A } ) . \ v _ { c o r r }$ is fed into a multi-layer perceptron with two hidden layers of 2000 ReLU units. The final regressor head projects an output vector d $\in \mathbb { R } ^ { 2 G ^ { 2 } }$ , where G is the resolution of the sparse control-point grid. We reshape d with the operator $\mathcal { R } : \mathbb { R } ^ { 2 G ^ { 2 } } \to \mathbb { R } ^ { 2 \times G \times G }$ , mapping the linear activations onto their spatial coordinates on the control-point grid to obtain $\Delta \mathbf { P } = \bar { ( \Delta x , \Delta y ) }$

Dense flow field generation: To generate a smooth, topology-preserving dense deformation field, we upsample the coarse control points to the input resolution $\dot { ( H , W ) }$ using bicubic interpolation:

$$
\Phi = { \mathrm { U p s a m p l e } } _ { b i c u b i c } ( \Delta \mathbf { P } )\tag{4}
$$

where $\Phi \in \mathbb { R } ^ { H \times W \times 2 }$ . The final aligned image $S _ { \hat { b } }$ is generated by spatially resampling the moving image $S _ { b }$ according to Φ:

$$
S _ { \hat { b } } = S _ { b } \circ \Phi\tag{5}
$$

This parameterization is, structurally, a free-form deformation model: any field in the network’s range is a bicubic lift of a coarse lattice, so smoothness is enforced architecturally, as high-frequency or folding fields are inexpressible, and is only refined, not created, by the regularizer below. It is the learned analogue of classical spline-based elastic registration [12, 21], with correspondence extraction and per-pair optimization amortized into one forward pass.

## 3.4 Training objective

To quantify pixel-wise alignment between the reference $S _ { a }$ and the warped image $S _ { \hat { b } } ,$ we use the soft Dice loss [37]. Let N be the number of pixels in the flattened images and k index a batch of size B:

$$
\mathcal { L } _ { \mathrm { d i c e } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \left( 1 - \frac { 2 \sum _ { i = 1 } ^ { N } S _ { \hat { b } , k } ^ { ( i ) } S _ { a , k } ^ { ( i ) } + \epsilon } { \sum _ { i = 1 } ^ { N } S _ { \hat { b } , k } ^ { ( i ) } + \sum _ { i = 1 } ^ { N } S _ { a , k } ^ { ( i ) } + \epsilon } \right)\tag{6}
$$

On near-binary ridge imagery, this is directly interpretable as the fraction of structure brought into coincidence. Let $V$ denote the predicted control-point grid; we apply a first-order smoothness penalty

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \mathbb { E } \left[ \left\| \nabla _ { x } V \right\| ^ { 2 } + \left\| \nabla _ { y } V \right\| ^ { 2 } \right]\tag{7}
$$

and train with the combined objective

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d i c e } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { s m o o t h } } .\tag{8}
$$

No deformation ground truth, correspondence annotation, or identity label is used at any stage.

## 4 Experiments

We train Gum-Net entirely on synthetic data and evaluate on synthetic as well as real-world data. Unlike [4], we perform no fine-tuning or adaptation on real-world datasets: the model is evaluated zero-shot on real benchmarks, so all real-data results directly measure synthetic-to-real transfer of the learned registration function.

## 4.1 Datasets

Real datasets. As real-world benchmarks providing organically generated elastic deformation, we evaluate on the FVC2004 benchmark [46] (110 fingers, 8 impressions each across 4 databases) and the Tsinghua Distorted Fingerprint Dataset (320 pairs of undistorted and distorted impressions) [47]. We use FVC2004 to quantify pre-matching structural alignment improvements via the Pearson Correlation Coefficient (PCC). Conversely, the Tsinghua dataset features severe real-world torque and complex skin distortion, and we use it exclusively to stress-test downstream minutiae matching behavior.

Simulated datasets. We use Anguli [48], a free and open-source synthetic fingerprint generator developed at the Indian Institute of Science, Bengaluru, to generate 73,500 fingerprints: 2,100 distinct impressions per noise level per class, across five signal-to-noise ratios and all fingerprint classes of the Henry classification [49]. The generator first produces a master template image, then casts impressions from that template (Figures 7 and 8 in the appendix show the templates and corpus diversity). Anguli is a procedural generator: it synthesizes orientation fields from mathematical models of singular points and grows ridge patterns by iterative filtering [48, 50]. It is not a learned generative model fitted to real fingerprints, so synthetic training involves no real biometric data at any stage; real datasets are used strictly for evaluation. Beyond scale, this yields unlimited reference–moving training pairs with correspondence guaranteed by construction.

![](images/37bf8a27093b91c62c984cf648d20894f28107b82af139a51ec89cbf6a91363f.jpg)  
Figure 3: Inference plot obtained from test data. This is an impression the model has never seen during training or validation.

Table 1: Downstream metrics, synthetic test set. Gum-Net improves genuine matching $( \mu _ { g e n } )$ and decidability $( d ^ { \prime } )$ while holding imposter scores $( \mu _ { i m p } )$ stable, outperforming the unaligned baseline and the AP/MP pooling variants.
<table><tr><td>Model</td><td> $\mu _ { g e n }$  ↑</td><td> $\mu _ { i m p } \downarrow$ </td><td> $d ^ { \prime } \uparrow$ </td><td> $\Delta \mu _ { g e n }$  ↑</td><td> $\Delta \mu _ { i m p } \downarrow$ </td><td> $\Delta d ^ { \prime } \uparrow$ </td></tr><tr><td>Baseline</td><td>153.25</td><td>10.06</td><td>4.0610</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GumNet-AP</td><td>152.65</td><td>10.30</td><td>3.7788</td><td>-0.60</td><td>0.24</td><td>-0.2823</td></tr><tr><td>GumNet-MP</td><td>123.75</td><td>9.39</td><td>3.4025</td><td>-29.49</td><td>-0.67</td><td>-0.6585</td></tr><tr><td>GumNet</td><td>170.39</td><td>10.25</td><td>4.8739</td><td>17.14</td><td>0.19</td><td>0.8128</td></tr></table>

## 4.2 Training

The models are implemented in PyTorch [51]. All inputs have size $1 9 2 ^ { 2 }$ , though the DCT spectral pooling layers make the input size arbitrary in principle. We train a single Gum-Net model on the entire synthetic dataset (seven fingerprint classes, five noise levels) with no labels at any level, following a curriculum [52]: training begins on the least noisy data and gradually introduces higher noise, 100 epochs per level with 14,700 examples each, concluding after 500 epochs. A control-point grid resolution $G \overset { \vartriangle } { = } 8$ is used throughout, with the objective $\mathcal { L } _ { \mathrm { t o t a l } }$ of Equation (8).

## 4.3 Evaluation

Evaluation of registration quality. To quantify the performance of the predicted dense deformation field, we measure pixel-level structural correspondence between the reference $S _ { a }$ and the rectified image $S _ { \hat { b } }$ using the Pearson Correlation Coefficient (PCC):

$$
\rho _ { S _ { a } , S _ { \hat { b } } } = \frac { \sum _ { i = 1 } ^ { N } ( S _ { a , i } - { \bar { S } } _ { a } ) ( S _ { { \hat { b } } , i } - { \bar { S } } _ { \hat { b } } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( S _ { a , i } - { \bar { S } } _ { a } ) ^ { 2 } \sum _ { i = 1 } ^ { N } ( S _ { { \hat { b } } , i } - { \bar { S } } _ { \hat { b } } ) ^ { 2 } } }\tag{9}
$$

where N is the total pixel count and $\bar { S } _ { a } , \bar { S } _ { \hat { b } }$ denote mean intensities. On near-binary ridge imagery, intensity correlation directly measures structural coincidence of ridge patterns [53, 54]; the improvement from $\rho ( S _ { a } , S _ { b } )$ to $\rho ( { \bar { S } } _ { a } , S _ { \hat { b } } )$ therefore evaluates unwarping capability independently of any downstream matching heuristic.

We compare against an ORB baseline [55], a feature-matching approach that strictly models linear (rigid/affine) transformations. As shown in Table 2, Gum-Net substantially outperforms ORB, achieving up to a 15.7% PCC increase on FVC2004 intra-class pairs with no real-data training. ORB in fact degrades baseline alignment on every dataset, which indicates that the deformation in these pairs lies outside the linear transformation class.

Topological validation via an independent minutiae matcher. A registration method can inflate similarity metrics by hallucinating structure; the real question is whether the predicted field restores true topology. Our testbed supplies an external oracle for exactly this question: minutiae are precise point landmarks consumed by matchers entirely independent of our training objective, so a topology restoring warp raises genuine match scores, a hallucinating warp is penalized, and, since no warp should manufacture similarity between different fingers, imposter scores should remain unchanged. Concretely, we evaluate with the NIST Bozorth3 [56] matcher: distorted images are unwarped via the predicted fields, minutiae are extracted from the aligned images, and matching scores form genuine and imposter distributions whose separability is summarized by the decidability index $d ^ { \prime } = \left| \mu _ { g e n } - \mu _ { i m p } \right| / \sqrt { { \textstyle \frac { 1 } { 2 } } ( \sigma _ { g e n } ^ { 2 } + \sigma _ { i m p } ^ { 2 } ) }$ , where $\mu$ and $\sigma ^ { 2 }$ denote the means and variances of the genuine and imposter score distributions.

Table 2: Pearson correlation coefficient increase in % on the FVC2004 datasets DB1–DB3 and our synthetic dataset.
<table><tr><td>Model</td><td>DB1</td><td>DB2</td><td>DB3</td><td>Synthetic (Ours)</td></tr><tr><td>No alignment</td><td>+0.0</td><td>+0.0</td><td>+0.0</td><td>+0.0</td></tr><tr><td>ORB alignment</td><td>-5.0</td><td>-4.6</td><td>-7.4</td><td>-3.0</td></tr><tr><td>Gum-Net AP</td><td>+8.7</td><td>+12.6</td><td>+10.7</td><td>+6.7</td></tr><tr><td>Gum-Net</td><td>+11.2</td><td>+15.7</td><td>+11.3</td><td>+10.5</td></tr></table>

Table 3: Downstream matching behavior on the real-world Tsinghua dataset.
<table><tr><td>Alignment Method</td><td> $\mu _ { g e n }$ </td><td> $\Delta \mu _ { g e n }$ </td><td> $d ^ { \prime }$ </td><td> $\Delta d ^ { \prime }$ </td></tr><tr><td>Baseline (No Alignment)</td><td>40.88</td><td></td><td>1.4111</td><td></td></tr><tr><td>Gum-Net 2D (Max Pooling)</td><td>42.90</td><td>+2.01</td><td>1.4227</td><td>+0.0116</td></tr><tr><td>Gum-Net 2D (Avg Pooling)</td><td>44.03</td><td>+3.15</td><td>1.4749</td><td>+0.0638</td></tr><tr><td>Gum-Net 2D (DCT)</td><td>43.97</td><td>+3.09</td><td>1.4775</td><td>+0.0664</td></tr></table>

As summarized in Tables 1 and 3, across both synthetic and real-world domains Gum-Net substantially increases genuine-pair scores $( \Delta \mu _ { g e n } \gg 0 )$ with negligible perturbation of imposter pairs $( \Delta \mu _ { i m p } \approx$ 0), outperforming the unaligned baseline and the AP/MP variants. This asymmetry is what one would expect if the warps recover real structure: alignment increases similarity only where genuine correspondence exists, indicating that the predicted fields do not fabricate matches. These are distribution-level summaries rather than operating characteristics; full ROC/DET behavior of a complete recognition pipeline is beyond our pre-matching scope and left to future work.

Two observations deserve comment. First, on synthetic data the max- and average-pooling variants do not merely underperform; they reduce decidability relative to no alignment at all: alignment quality, not alignment presence, drives the downstream benefit, and a poorly estimated warp is worse than none. Second, on Tsinghua the average-pooling variant attains a marginally higher genuine mean than DCT (44.03 vs. 43.97), while DCT retains the best decidability. We hypothesize that Tsinghua’s enhanced, binarized impressions reduce the fine-detail advantage spectral pooling provides on raw ridge imagery; the unambiguous synthetic ordering isolates the effect on unprocessed structure. Characterizing this preprocessing–pooling interaction remains open.

## 5 Conclusion

We introduced 2D Gum-Net, an end-to-end unsupervised framework for dense non-rigid 2D image registration, obtained by transferring the original 3D Gum-Net [4] from volumetric cryo-ET alignment to two dimensions and replacing its rigid transformation module with a Dense Spatial Transformer, so that a smooth dense deformation field is predicted without ground-truth displacement annotations while DCT spectral pooling and control-point parameterization preserve the fine structure on which correspondence depends. Trained purely on synthetic fingerprint data and evaluated zero-shot, the method improves structural correlation by 10.5% (synthetic) and up to 15.7% (FVC2004), and on Tsinghua raises genuine match scores while leaving imposter scores essentially unchanged, confirming deformation quality through an external matcher rather than the training objective alone. Evaluation is restricted to a single domain by design and reports distribution-level summaries rather than operating characteristics; future work includes extending to other imagery where fine quasiperiodic structure meets elastic deformation, operational metrics on larger datasets, and coupling with fixed-length embeddings [33] for retrieval at scale.

## References

[1] Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, and Adrian V. Dalca. VoxelMorph: A learning framework for deformable medical image registration. IEEE Transactions on Medical Imaging, 38(8):1788–1800, 2019.

[2] Ignacio Rocco, Relja Arandjelovic, and Josef Sivic. Convolutional neural network architecture for´ geometric matching, 2017. URL https://arxiv.org/abs/1703.05593.

[3] Tanner Schmidt, Richard Newcombe, and Dieter Fox. Self-supervised visual descriptor learning for dense correspondence. IEEE Robotics and Automation Letters, 2(2):420–427, 2016.

[4] Xiangrui Zeng and Min Xu. Gum-net: Unsupervised geometric matching for fast and accurate 3d subtomogram image alignment and averaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4073–4084, 2020.

[5] Xuanbin Si, Jianjiang Feng, Bo Yuan, and Jie Zhou. Dense registration of fingerprints. Pattern Recognition, 63:87–101, 2017.

[6] Zhe Cui, Jianjiang Feng, and Jie Zhou. Dense fingerprint registration via displacement regression network. In 2019 International Conference on Biometrics (ICB), pages 1–8. IEEE, 2019.

[7] Xiongjun Guan, Yongjie Duan, Jianjiang Feng, and Jie Zhou. Regression of dense distortion field from a single fingerprint image. IEEE Transactions on Information Forensics and Security, 18:4377–4390, 2023. ISSN 1556-6021. doi: 10.1109/tifs.2023.3296310. URL http://dx.doi.org/10.1109/TIFS.2023. 3296310.

[8] Steven A. Grosz, Joshua J. Engelsma, and Anil K. Jain. C2CL: contact to contactless fingerprint matching. CoRR, abs/2104.02811, 2021. URL https://arxiv.org/abs/2104.02811.

[9] Oren Rippel, Jasper Snoek, and Ryan P. Adams. Spectral representations for convolutional neural networks, 2015. URL https://arxiv.org/abs/1506.03767.

[10] Hao Zhang and Jianwei Ma. Hartley spectral pooling for deep learning. arXiv preprint arXiv:1810.04028, 2018.

[11] James S Smith and Bogdan M Wilamowski. Discrete cosine transform spectral pooling layers for convolutional neural networks. In International Conference on Artificial Intelligence and Soft Computing, pages 235–246. Springer, 2018.

[12] Asker M. Bazen and Sabih H. Gerez. Fingerprint matching by thin-plate spline modelling of elastic deformations. Pattern Recognition, 36(8):1859–1867, 2003. ISSN 0031-3203. doi: https://doi.org/10. 1016/S0031-3203(03)00036-0. URL https://www.sciencedirect.com/science/article/pii/ S0031320303000360.

[13] Davide Maltoni, Dario Maio, Anil K Jain, and Salil Prabhakar. Handbook offingerprint recognition. Springer, 2009.

[14] Qinghai Gao and Xiaowen Zhang. A study of distortion effects on fingerprint matching. Computer Science and Engineering, 2(3):37–42, 2012.

[15] Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu. Spatial transformer networks. Advances in Neural Information Processing Systems, 28, 2015.

[16] Xiongjun Guan, Jianjiang Feng, and Jie Zhou. Phase-aggregated dual-branch network for efficient fingerprint dense registration. IEEE Transactions on Information Forensics and Security, 19:5712–5724, 2024. ISSN 1556-6021. doi: 10.1109/tifs.2024.3403507. URL http://dx.doi.org/10.1109/TIFS. 2024.3403507.

[17] Nalini K. Ratha, Kalle Karu, Shaoyun Chen, and Anil K. Jain. A real-time matching system for large fingerprint databases. IEEE Trans. Pattern Anal. Mach. Intell., 18:799–813, 1996. URL https://api. semanticscholar.org/CorpusID:7198537.

[18] Jianjiang Feng. Combining minutiae descriptors for fingerprint matching. Pattern Recognit., 41:342–352, 2008. URL https://api.semanticscholar.org/CorpusID:874663.

[19] Raffaele Cappelli, Matteo Ferrara, and Davide Maltoni. Minutia cylinder-code: A new representation and matching technique for fingerprint recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32:2128–2141, 2010. URL https://api.semanticscholar.org/CorpusID:17406459.

[20] Arun Ross, Sarat Dass, and Anil K. Jain. A deformable model for fingerprint matching. Pattern Recognition, 38(1):95–103, 2005.

[21] Arun Ross, Sarat C. Dass, and Anil K. Jain. Fingerprint warping using ridge curve correspondences. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(1):19–30, 2006.

[22] Shubham Pandey, Bhavin Jawade, and Srirangaraj Setlur. Ridgeformer: Mutli-stage contrastive training for fine-grained cross-domain fingerprint recognition. In 2025 IEEE International Conference on Image Processing (ICIP), page 2480–2485. IEEE, September 2025. doi: 10.1109/icip55913.2025.11084677. URL http://dx.doi.org/10.1109/ICIP55913.2025.11084677.

[23] Suksan Jirachaweng and Vutipong Areekul. Fingerprint enhancement based on discrete cosine transform. In International Conference on Biometrics, pages 96–105. Springer, 2007.

[24] Muzhir Shaban Al-Ani and Wasan M Al-Aloosi. Biometrics fingerprint recognition using discrete cosine transform (dct). International Journal ofComputer Applications, 69(6), 2013.

[25] Koichi Ito, Hiroshi Nakajima, Koji Kobayashi, Takafumi Aoki, and Tatsuo Higuchi. A fingerprint matching algorithm using phase-only correlation. IEICE transactions on fundamentals of electronics, communications and computer sciences, 87(3):682–691, 2004.

[26] Zhengyu Ouyang, Jianjiang Feng, Fei Su, and Anni Cai. Fingerprint matching with rotation-descriptor texture features. In 18th International Conference on Pattern Recognition (ICPR’06), volume 4, pages 417–420. IEEE, 2006.

[27] Haiyun Xu and Raymond NJ Veldhuis. Spectral minutiae representations for fingerprint recognition. In 2010 Sixth International Conference on Intelligent Information Hiding and Multimedia Signal Processing, pages 341–345. IEEE, 2010.

[28] Mihir Sahasrabudhe and Anoop M Namboodiri. Learning fingerprint orientation fields using continuous restricted boltzmann machines. In 2013 2nd IAPR Asian Conference on Pattern Recognition, pages 351–355. IEEE, 2013.

[29] Dogus Karabulut, Pavlo Tertychnyi, Hasan Sait Arslan, Cagri Ozcinar, Kamal Nasrollahi, Joan Valls, Joan Vilaseca, Thomas Baltzer Moeslund, and Gholamreza Anbarjafari. Cycle-consistent generative adversarial neural networks based low quality fingerprint enhancement. Multimedia Tools and Applications, 79:18569 – 18589, 2020. URL https://api.semanticscholar.org/CorpusID:212420707.

[30] Yue-Jie Hou, Zai-Xin Xie, Jian-Hu, Yao-Shen, and Chi-Chun Zhou. An unsupervised deep-learning method for fingerprint classification: the ccae network and the hybrid clustering strategy, 2021. URL https://arxiv.org/abs/2109.05526.

[31] Yao Tang, Fei Gao, Jufu Feng, and Yuhang Liu. Fingernet: An unified deep network for fingerprint minutiae extraction, 2017. URL https://arxiv.org/abs/1709.02228.

[32] Dinh-Luan Nguyen, Kai Cao, and Anil K. Jain. Robust minutiae extractor: Integrating deep networks and fingerprint domain knowledge, 2017. URL https://arxiv.org/abs/1712.09401.

[33] Joshua J. Engelsma, Kai Cao, and Anil K. Jain. Learning a fixed-length fingerprint representation, 2019. URL https://arxiv.org/abs/1909.09901.

[34] Zhiyu Pan, Xiongjun Guan, Yongjie Duan, Jianjiang Feng, and Jie Zhou. Fixed-length dense fingerprint representation with alignment and robust enhancement, 2025. URL https://arxiv.org/abs/2505. 03597.

[35] Arnab Maity, Manasa, Pavan Kumar C, and Raghavendra Ramachandra. Latentprintformer: A hybrid cnn-transformer with spatial attention for latent fingerprint identification, 2025. URL https://arxiv. org/abs/2511.08119.

[36] Steven A. Grosz and Anil K. Jain. Afr-net: Attention-driven fingerprint recognition network, 2022. URL https://arxiv.org/abs/2211.13897.

[37] Zifu Wang, Teodora Popordanoska, Jeroen Bertels, Robin Lemmens, and Matthew B Blaschko. Dice semimetric losses: Optimizing the dice score with soft labels. In International conference on medical image computing and computer-assisted intervention, pages 475–485. Springer, 2023.

[38] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.

[39] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015.

[40] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[41] Nasir Ahmed, T. Natarajan, and Kamisetty R Rao. Discrete cosine transform. IEEE transactions on Computers, 100(1):90–93, 1974.

[42] Kwang Moo Yi, Eduard Trulls, Vincent Lepetit, and Pascal Fua. Lift: Learned invariant feature transform. In European conference on computer vision, pages 467–483. Springer, 2016.

[43] Iaroslav Melekhov, Juho Kannala, and Esa Rahtu. Image patch matching using convolutional descriptors with euclidean distance. In Asian Conference on Computer Vision, pages 638–653. Springer, 2016.

[44] Hyun Oh Song, Stefanie Jegelka, Vivek Rathod, and Kevin Murphy. Deep metric learning via facility location. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5382–5390, 2017.

[45] Xufeng Han, Thomas Leung, Yangqing Jia, Rahul Sukthankar, and Alexander C Berg. Matchnet: Unifying feature and metric learning for patch-based matching. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3279–3286, 2015.

[46] Dario Maio, Davide Maltoni, Raffaele Cappelli, Jim L Wayman, and Anil K Jain. Fvc2004: Third fingerprint verification competition. In International conference on biometric authentication, pages 1–7. Springer, 2004.

[47] Xuanbin Si, Jianjiang Feng, Jie Zhou, and Yuxuan Luo. Detection and rectification of distorted fingerprints. IEEE transactions on pattern analysis and machine intelligence, 37(3):555–568, 2015.

[48] Afzalul Haque Ansari. Generation and storage of large synthetic fingerprint database. ME Thesis, Jul, 2, 2011.

[49] Edward Richard Henry. Classification and uses offinger prints. HM Stationery office, 1928.

[50] Raffaele Cappelli, Dario Maio, Davide Maltoni, et al. Sfinge (synthetic fingerprint generator). International Workshop on Biometric Technologies, 2004.

[51] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[52] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.

[53] Sheng Lan, Zhenhua Guo, and Jane You. Pre-registration of translated/distorted fingerprints based on correlation and the orientation field. Information Sciences, 520:292–304, 2020. ISSN 0020-0255. doi: https://doi.org/10.1016/j.ins.2020.02.017. URL https://www.sciencedirect.com/science/ article/pii/S002002552030092X.

[54] William H Press. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.

[55] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: An efficient alternative to sift or surf. In 2011 International Conference on Computer Vision, pages 2564–2571, 2011. doi: 10.1109/ICCV. 2011.6126544.

[56] C. Watson, M. Garris, Elham Tabassi, C. Wilson, R. McCabe, Stanley Janet, and Kenneth Ko. User’s guide to nist biometric image software (nbis), nist, 2007. NIST, Gaithersburg, MD, USA, Tech. Rep, 1:2, 2007.

## A Detailed architecture of Gum-Net variants

To verify that DCT spectral pooling and filtering layers offer superior performance compared to max and average pooling, we built identical architectures differing only in the pooling operation, named Gum-Net MP and Gum-Net AP. Figure 4 shows the detailed architecture for Gum-Net with input size 192 × 192 for both $S _ { a }$ and $S _ { b } ;$ Figures 5 and 6 show the detailed architectures for Gum-Net MP and Gum-Net AP respectively. Gum-Net AP and Gum-Net MP change only the pooling within the feature extraction module (Section 3.1), keeping the rest of the architecture exactly the same.

![](images/4274168d2a1bcd54f1f4c2d073475692426609e9b11cf4e1d5485b2234f9fae7.jpg)  
Figure 4: Detailed architecture of Gum-Net.

![](images/0f5dc9498141787fd1ec4881e2d951bf823cd1bae4233ecd4f31e57a27bf434b.jpg)  
Figure 5: Detailed architecture of Gum-Net MP.

![](images/2f952d75774038bf67fa924ab6c8b30b5f20ae119a51800caea165729e8543c0.jpg)  
Figure 6: Detailed architecture of Gum-Net AP.

## B Dataset diversity and inference details

Figure 7 shows the master templates used by Anguli, and Figure 8 illustrates the diversity of our synthetic corpus across the seven fingerprint classes and five signal-to-noise ratios. Figure 9 shows inference plots across all finger types; the model has not seen these reference–moving pairs during training.

![](images/457f912a0a0df929c80c2d23e3afdb3bec39094546e10fabc948d5c6459d6404.jpg)  
Figure 7: Templates used by Anguli to generate the fingerprint impressions.

![](images/349e7b511c9feb064f01a2857bfc6867a06397f8ac97aaa80134c1b2282153b7.jpg)  
Figure 8: Fingerprint impressions across seven classifications and five signal-to-noise ratios.

![](images/b128bada864e6ae328f18030ac8edd49109c391e75da88db950ca52c0a6165a7.jpg)  
Figure 9: Inference plots across all finger types.

## C Details on the DCT optimization

The 2D DCT spectral pooling layer is implemented as a sequence of frozen linear transformations to exploit highly optimized General Matrix Multiply (GEMM) routines on CUDA devices. For a 1D signal of length N, the Type-II DCT basis matrix $\mathbf { \check { W } } \in \mathbb { R } ^ { N \times N }$ is precomputed by applying the DCT operator to the identity matrix ${ \mathbf { I } } _ { N }$ :

$$
\mathbf { W } = \mathrm { D C T - I I } ( \mathbf { I } _ { N } )\tag{10}
$$

such that the transform $\mathbf { y } = \mathbf { x } \mathbf { W } ^ { \top }$ is equivalent to a standard 1D DCT-II applied to the row vector x. This matrix is computed once at initialization and stored as a frozen (non-trainable) weight in a linear layer, reducing repeated transform computation to a single matrix multiplication at inference time.

The 2D Type-II DCT of a feature map $\mathbf { X } \in \mathbb { R } ^ { C \times H \times W }$ is defined as [41]

$$
Y _ { c , u , v } = \alpha _ { u } \alpha _ { v } \sum _ { x = 0 } ^ { H - 1 } \sum _ { y = 0 } ^ { W - 1 } X _ { c , x , y } \cos \left( { \frac { \pi u ( 2 x + 1 ) } { 2 H } } \right) \cos \left( { \frac { \pi v ( 2 y + 1 ) } { 2 W } } \right)\tag{11}
$$

with $\alpha _ { u } = \sqrt { 1 / H }$ for $u = 0$ and $\scriptstyle { \sqrt { 2 / H } }$ for $u > 0 , \forall u \in \{ 0 , . . . , H - 1 \}$ , and $\alpha _ { v }$ defined analogously. Its separability allows the transform to be decomposed into two sequential 1D operations:

$$
\mathbf { Y } = \mathbf { W } _ { H } \mathbf { X } \mathbf { W } _ { W } ^ { \top }\tag{12}
$$

where ${ \bf W } _ { H } = \mathrm { D C T - I I } ( { \bf I } _ { H } ) \in \mathbb { R } ^ { H \times H }$ and $\mathbf { W } _ { W } = { \mathrm { D C T - I I } } ( \mathbf { I } _ { W } ) \in \mathbb { R } ^ { W \times W }$ are the precomputed basis matrices for the height and width dimensions respectively. In practice, this is computed by first applying $\mathbf { W } _ { W }$ along the width axis, transposing, applying W<sub>H</sub> along the (now leading) spatial axis, and transposing back, matching Equation (2) in the main paper.

Following the forward 2D DCT, a binary low-pass mask $\mathbf { M } \in \{ 0 , 1 \} ^ { H \times W }$ is applied element-wise to the frequency representation:

$$
\tilde { \mathbf { Y } } = \mathbf { Y } \odot \mathbf { M } , \qquad \mathbf { M } _ { u , v } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } u < h _ { \mathrm { f r e q } } \mathrm { ~ a n d ~ } v < w _ { \mathrm { f r e q } } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{13}
$$

retaining only the top-left $h _ { \mathrm { f r e q } } \times w _ { \mathrm { f r e q } }$ block of low-frequency coefficients. The masked representation is then cropped to $h _ { \mathrm { o u t } } \times w _ { \mathrm { o u t } }$ and reconstructed via the inverse transform, where the 2D Type-III DCT (the inverse of Type-II) is applied separably using analogous precomputed basis matrices $\mathbf { W } _ { h _ { \mathrm { o u t } } } ^ { - 1 }$ and $\mathbf { W } _ { w _ { \mathrm { o u t } } } ^ { - 1 }$

$$
\hat { \mathbf { X } } = \mathbf { W } _ { h _ { \mathrm { o u t } } } ^ { - 1 } \tilde { \mathbf { Y } } _ { [ : h _ { \mathrm { o u t } } , : w _ { \mathrm { o u t } } ] } \left( \mathbf { W } _ { w _ { \mathrm { o u t } } } ^ { - 1 } \right) ^ { \top }\tag{14}
$$

The constraint $h _ { \mathrm { f r e q } } ~ \leq ~ h _ { \mathrm { o u t } }$ and $w _ { \mathrm { f r e q } } ~ \leq ~ w _ { \mathrm { o u t } }$ is enforced to ensure that no retained frequency coefficients are discarded during the inverse transform, which would render the masking step computationally redundant. This GEMM-based implementation achieves approximately a 50× speedup over a naive iterative DCT on CUDA devices, as matrix multiplication is among the most heavily optimized parallel primitives available on modern GPU hardware.

Figures 10 and 11 show our speedup and memory efficiency compared to a traditional implementation of the DCT in PyTorch, which uses the built-in FFT suite to extract DCT coefficients.

All code for this project is available at https://github.com/caffeinekeyboard/2DGumNet.

![](images/f6b975476121934a9997e3adae260b81f8587460d5a71c112cef928ce04487b6.jpg)  
Figure 10: Time comparison.

![](images/58d900e0ad84b4bb1c811c18f150756457e18ddc85e99949482378886b4bad03.jpg)  
Figure 11: Space comparison.