# Cross-Encoder Anchor Refinement for Zero-Shot Anomaly Detection

Kaisong Ji<sup>a,1</sup>, Ziyang Xu<sup>a,1</sup>, Chang Bian<sup>b</sup>, Yu Zhou<sup>a</sup>, Wenyu Liu<sup>a</sup>, Xinggang Wang<sup>a,∗</sup>

<sup>a</sup>Huazhong University of Science and Technology, Wuhan, China <sup>b</sup>New York University, New York, USA

## Abstract

Zero-shot anomaly detection aims to transfer anomaly discrimination from auxiliary training domains to unseen categories while localizing subtle defects. A key challenge is to construct discriminative references that combine contextual representations with complementary local evidence. We present Cross-Encoder Anchor Refinement (CEAR), which connects two visual encoding paths for the same image at the anchor stage, before normal and anomalous prototypes are constructed. An independent convolutional encoder pretrained for industrial product recognition builds an image-specific memory shared across selected vision transformer (ViT) layers. Spatial anchors read from ViT patch features query this memory through cross-attention and undergo residual refinement. Image-conditioned normal and anomalous queries then gate anchor aggregation to construct prototypes for anomaly scoring.<sup>c</sup> Across 13 industrial and medical datasets, CEAR achieves the best performance in 57 of 68 evaluations spanning image-level detection and pixel-level localization. Notably, CEAR surpasses all other state-of-the-art methods in 41 of 44 pixel-level com parisons. Across 11 localization datasets, mean gains in pixel-level maximum F1 score and average precision are 4.45 and 3.79 percentage points, respectively. This study shows how cross-encoder interaction at the anchor stage can integrate complementary visual representations for fine-grained anomaly localization across domains.

Keywords: Zero-shot anomaly detection, Anomaly localization, Spatial anchor refinement, Image-conditioned prototypes, Cross-domain generalization<sup>.</sup>

## 1. Introduction

v Industrial visual anomaly detection identifies defective products and localizes regions that deviate from normal appearance. Defects are rare and diverse, while changing product categories<sub>7</sub> make extensive annotation and category-specific training costly.9 We study zero-shot anomaly detection (ZSAD) with auxiliary<sup>3</sup> supervision, learning transferable anomaly discrimination for0 target categories absent from anomaly training. This setting couples cross-category recognition with precise localization,<sup>6</sup> requiring sensitivity to fine cracks, small notches, and lowcontrast regions within otherwise normal structures. MVTec<sup>v</sup> AD, Real-IAD, and MVTec AD 2 illustrate these challenges through diverse defects and demanding imaging conditions [1– 3].<sup>a</sup>

Vision–language models provide transferable semantic references for ZSAD. WinCLIP combines prompt ensembles with window-, patch-, and image-level features, whereas Anomaly-CLIP learns object-agnostic prompts for normality and abnormality [4, 5]. AA-CLIP, Bayes-PFL, and FAPrompt improve semantic separation, model prompt distributions, and adapt fine-grained abnormality prompts to individual images, respectively [6–8]. These advances also highlight the importance of matching semantic references to spatially faithful visual evidence. SCLIP and ClearCLIP improve dense CLIP predictions through attention reformulation and representation decomposition [9, 10]. For anomaly detection, FE-CLIP, DLVP-CLIP, and FB-CLIP exploit frequency cues, local visual prompts, and foreground–background separation [11–13]. Together, these studies motivate a closer examination of how local visual structure contributes to anomaly discrimination.

Recent work on pretrained visual representations highlights the importance of preserving spatial structure and recovering fine-grained detail for dense prediction [14, 15]. In anomaly detection, VisualAD constructs discriminative representations through visual tokens and spatial aggregation [16]. AnomalyVFM combines diverse anomaly synthesis with internal feature adaptation to improve transfer across datasets [17]. These studies motivate our central question: how can contextual representations and complementary local evidence jointly inform transferable anomaly references?

Spatial sampling afects the local detail available for constructing these references. With a fixed patch size, higher input resolution produces a denser patch grid and can preserve evidence of small defects. We therefore use MoonViT, a visual encoder supporting variable input resolutions and pretrained on image–text pairs [18]. Beyond spatial sampling, we focus on how local evidence can guide the construction of anomaly references at the output of this encoder. ViT-CoMer demonstrates that interactions between convolutional features and ViT representations can improve dense prediction [19]. For ZSAD, we therefore examine how features from an independent local encoder can inform discriminative references derived from ViT representations. Spatial anchors provide the interaction interface: they retrieve local evidence and undergo refinement before query-gated aggregation constructs normal and anomalous prototypes. Section 4.3 compares this mechanism with scorelevel late fusion and direct feature concatenation under controlled conditions. Figure 1 compares anomaly responses from MoonViT-only and CEAR on industrial and medical samples, illustrating spatial diferences associated with cross-encoder conditioning. In the examples shown, CEAR responds more selectively to annotated anomalies while reducing activations in some regions of normal texture and background.

![](images/a1e9eabd454a4aa765c2590e9e238ee3bc80562918a134be9b3a9e53ce1397be.jpg)  
Figure 1: Qualitative comparisons examine the contribution of cross-encoder conditioning to anomaly localization in industrial and medical images. CEAR concentrates responses more closely on annotated anomalies while reducing activations on some normal structures and background regions.

We therefore present Cross-Encoder Anchor Refinement (CEAR), which connects two separate visual encoding paths through spatial anchors. Here, cross-encoder refers to refining anchors derived from ViT features using local evidence encoded independently from the same image. A ConvNeXt V2 CNN pretrained for industrial product recognition supplies hierarchical features [20]. These features form a local memory shared across selected MoonViT layers. Spatial anchors read from each selected layer query this memory through crossattention and undergo residual refinement. The interaction operates on encoder outputs, before normal and anomalous prototypes are constructed. The anchors are shared by the normal and anomalous branches. Queries combining learned normal and anomalous seeds with image-level context then gate anchor aggregation to construct image-conditioned prototypes. The layerwise branches use independent refinement parameters and access the same memory; refined anchors are not passed between layers. Anomaly adaptation and inference use image features without text prompts or a language branch.

We evaluate CEAR on 13 industrial and medical datasets through cross-category and cross-dataset comparisons. Among the methods and protocols summarized in Table 1, CEAR achieves 57 best results across 68 dataset–metric comparisons. For pixel-level localization, CEAR scores higher than all other state-of-the-art methods in 41 of 44 comparisons. Across 11 localization datasets, mean gains in pixel-level maximum F1 score (F1-max) and average precision (AP) are 4.45 and 3.79 percentage points, respectively. Mean gains in area under the receiver operating characteristic curve (AUROC) and area under the per-region overlap curve (PRO) are 2.20 and 2.39 percentage points. Beyond these performance comparisons, ablations vary industrial pretraining, memory features, interaction stage, and multilayer supervision. Component ablations and visualizations examine how local conditioning influences predictions.

Our main contributions are as follows.

• We introduce CEAR for zero-shot anomaly detection, integrating contextual representations and complementary local evidence through cross-encoder interaction before normal and anomalous prototypes are constructed.

• We propose cross-encoder anchor refinement, allowing ViT-derived spatial anchors to retrieve evidence from a shared local memory built by an independent encoder. Layer-specific residual updates incorporate this evidence before query-gated construction of image-conditioned normal and anomalous prototypes.

• We evaluate CEAR across 13 industrial and medical datasets. Matched-backbone and matched-resolution comparisons, component ablations, and qualitative analyses examine transfer performance and the role of local conditioning.

## 2. Related Work

## 2.1. Vision–Language Referencesfor Anomaly Detection

Vision–language methods use normal and anomalous text representations as transferable discriminative references. Win-CLIP combines prompt ensembles with window-, patch-, and image-level matching, whereas AnomalyCLIP learns objectagnostic prompts [4, 5]. AdaCLIP combines static and imageconditioned prompts, while VCP-CLIP incorporates global and local visual context into text representations [21, 22]. FiLo couples fine-grained defect descriptions with dedicated localization modules [23]. AA-CLIP strengthens anomaly-aware visual– textual representations, while AdaptCLIP alternately adapts the visual and textual branches [6, 24].

Bayes-PFL, FAPrompt, and PromptMoE improve prompt flexibility through distribution modeling, sample-dependent abnormality prompts, and visually guided expert mixtures, respectively [7, 8, 25]. FE-CLIP, DLVP-CLIP, and FB-CLIP enhance local discrimination through frequency-aware adaptation, regional visual prompting, and foreground–background separation, respectively [11–13]. These developments connect semantic adaptation with spatially localized evidence, motivating consideration of where such evidence enters anomaly reference construction.

## 2.2. Reference Requirements and Transfer Settings

Anomaly detectors difer in their access to target-domain information during training and inference. MuSc and MuSc-V2 exploit relationships among unlabeled test samples through mutual scoring [26, 27]. RareCLIP estimates patch rarity using an online memory of previously observed images [28]. InC-TRL, UniVAD, and MetaUAS use normal reference images at inference [29–31]. Dinomaly and AnomalyMoE instead learn from target-category normal training data [32, 33]. Reference requirements and test-sample interactions are therefore distinct from whether a detector uses language.

For auxiliary-supervised transfer, AnomalyVFM combines anomaly synthesis with feature adaptation and confidenceweighted supervision [17]. CEAR learns from auxiliary anomaly data and processes unseen target categories one image at a time. Its conditioning evidence comes from the current input, without target-domain normal references or interactions among test samples. Zero-shot evaluation here concerns target categories excluded from anomaly adaptation, while auxiliary training uses image labels and pixel masks.

## 2.3. Dense Representations and Local Evidence

Strong pretrained ViTs support dense prediction, as demonstrated by EoMT [34], while MoonViT supports variableresolution visual encoding [18]. Register tokens and DVT address spatial artifacts in pretrained ViT features [14, 35]. DI-NOv3 preserves dense feature quality through Gram anchoring [36]. FeatUp and AnyUp recover spatial detail through feature upsampling [15, 37]. ViT-CoMer introduces multiscale CNN–ViT interactions within the backbone for dense prediction [19]. These approaches improve spatial representations through diferent operations, from feature organization and upsampling to interactions between encoding paths.

Real-IAD and MVTec AD 2 highlight fine-scale defects and challenging industrial imaging conditions [2, 3]. In CEAR, ConvNeXt V2 supplies hierarchical convolutional features [20], with product-recognition pretraining on Real-IAD providing the industrial representation source. These features serve as local conditioning evidence at the output of the frozen ViT, before anomaly prototypes are constructed.

## 2.4. Visual Prototypes and Conditional Refinement

VisualAD constructs layerwise discriminative references through token-gated aggregation of spatial anchors, with attention and gates recomputed for each image [16]. UniADet learns task- and layer-specific decision weights for classification and segmentation [38]. ReCAP conditions prototypes on globally pooled features from the same visual encoder through bounded residual updates [39]. Layer-specific discrimination and image conditioning therefore provide established foundations for visual anomaly detection.

CEAR investigates how independently encoded local evidence can inform prototype construction. ViT-derived anchors query a shared CNN memory through cross-attention and undergo residual refinement before normal and anomalous queries gate their aggregation into prototypes. Relative to VisualAD and ReCAP, the distinction concerns both the independent source of local evidence and its application to anchors before prototype construction. The interaction operates on outputs of the frozen backbones; layerwise branches share the local memory but retain independent refinement parameters.

## 3. Method

## 3.1. Problem Formulation and Framework Overview

CEAR refines spatial anchors to connect ViT representations with local evidence encoded independently from the same image by a CNN. The ViT supplies patch features and contextual representations, while fused and encoded CNN features form a local memory shared across layerwise branches. At each selected ViT layer, spatial evidence readout (SER) first constructs anchors from patch features. CNN-conditioned anchor refinement (CCAR) then retrieves information from the shared memory to update these anchors. Finally, query-gated prototype construction (QGPC) aggregates the refined anchors into normal and anomalous prototypes. This cross-encoder interaction operates on anchors and memory derived from encoder outputs; refined anchors are not propagated between layers. Figure 2 illustrates the overall data flow, the residual structure of anchor refinement, and the construction of the shared local memory.

We consider an auxiliary training set $\mathcal { D } _ { s } = \{ ( \mathbf { x } _ { i } , y _ { i } , \mathbf { m } _ { i } ) \} _ { i = 1 } ^ { n _ { s } } ,$ where image labels $y _ { i } \in \{ 0 , 1 \}$ and pixel masks $\mathbf { m } _ { i } \in \{ 0 , 1 \} ^ { H _ { 0 } \times \hat { W } _ { 0 } }$ indicate anomaly status at their respective granularities. After learning from the auxiliary domain, the model directly predicts an image anomaly score s(x) and a pixel-level anomaly map H(x) for unseen target categories. The CNN is pretrained using industrial product category supervision independent of both the source and target anomaly domains. Anomaly adaptation and inference operate entirely in the visual feature space, with each target image processed independently.

As shown in Figure 2, the two encoders process the same image to obtain ViT patch features and multiscale CNN features. Global statistics from the deepest selected ViT layer generate image-conditioned queries. Within each layerwise branch, SER, CCAR, and QGPC successively read spatial evidence, refine anchors, and construct prototypes. A parallel patch projection MLP maps patch features into a space where cosine similarities to the two prototypes yield layerwise anomaly responses. These responses are bilinearly upsampled and summed into an anomaly map, whose highest-response pixels are averaged to obtain the image score. The frozen components in the figure denote the two encoder backbones. The trainable components comprise the shared output layer normalization and the anomaly modeling modules.

## 3.2. Visual Representations and Shared Local Memory

Let I denote the set of selected ViT layers. Shared layer normalization converts their outputs into patch features:

![](images/9f76701f76a5c25dd621f5cd237a6fd731fd1d238c28ffa41e4644b5c305ce08.jpg)  
Figure 2: CEAR architecture and cross-encoder interaction. Top: frozen ViT and CNN backbones encode the same image. The industrially pretrained CNN constructs a shared local memory for refining ViT-derived spatial anchors before query-gated construction of normal and anomalous prototypes. Projected patch features are compared with these prototypes to produce layerwise anomaly responses for localization and image-level scoring. Middle: the residual CCAR block. Bottom: multiscale CNN memory construction. Layerwise refinement shares the memory but uses independent parameters, without passing refined anchors between layers.

$$
\mathbf { X } _ { \ell } = \mathbf { L } \mathbf { N } _ { o } ( E _ { \ell } ( \mathbf { x } ) ) \in \mathbb { R } ^ { N \times d } , \ell \in \mathcal { I } .\tag{1}
$$

Here, $E _ { \ell }$ denotes the encoding mapping at layer $\ell , N$ is the number of patches, and $d$ is the channel dimension. The selected layers retain the same spatial grid while providing representations at diferent depths. For clarity, we omit the batch dimension and training-time dropout. Tokens occupy rows of feature matrices, and linear transformations multiply weight matrices on the right.

To condition discrimination on the input image, we compute spatial means, maxima, and standard deviations from the deepest selected layer $\mathbf { X } _ { \ell ^ { * } \cdot } \mathbf { A }$ linear mapping converts these statistics into a global context vector u, which then conditions the normal and anomalous queries:

$$
\begin{array} { r l } & { \mathbf { u } = [ \mathrm { m e a n } ( \mathbf { X } _ { \ell ^ { * } } ) ; \mathrm { m a x } ( \mathbf { X } _ { \ell ^ { * } } ) ; } \\ & { ~ \mathrm { s t d } ( \mathbf { X } _ { \ell ^ { * } } ) ] \mathbf { W } _ { u } + \mathbf { b } _ { u } , } \\ & { ~ \mathbf { t } _ { c } = \mathbf { q } _ { c } + \mathbf { u } \mathbf { W } _ { c } + \mathbf { b } _ { c } , c \in \{ a , n \} . } \end{array}\tag{2}
$$

Here, [·; ·; ·] denotes channel concatenation, $\ell ^ { * } =$ max I identifies the deepest selected layer, and $c \in$ {a n} indexes the anomalous and normal classes, respectively. Each learnable query seed ${ \bf q } _ { c }$ is combined with image context to form the corresponding image-conditioned query $\mathbf { t } _ { c }$ . The three statistics capture average responses, salient activations, and spatial response variability. They allow the queries to combine a discriminative direction shared across images with information specific to the current input. These queries are generated at the output of the encoder.

The CNN branch in Figure 2 extracts multiscale features $\{ { \bf C } _ { s } \} _ { s \in { \cal S } }$ representing texture variations, edges, and component structure in the input image. Channel projection $\mathcal { P } _ { s }$ and spatial alignment $\mathcal { A } _ { s }$ map the features onto a common grid. Residual fusion then uses the intermediate scale b as its reference:

$$
\begin{array} { r l } & { \mathbf { F } _ { s } = \mathcal { A } _ { s } ( \mathcal { P } _ { s } ( \mathbf { C } _ { s } ) ) , s \in S , } \\ & { \mathbf { F } = \mathbf { F } _ { b } + \beta _ { f } \Phi _ { f } ( \mathbf { C o n c a t } _ { s \in S } ( \mathbf { F } _ { s } ) ) . } \end{array}\tag{3}
$$

All aligned features ${ \bf F } _ { s }$ have shape $h _ { m } \times w _ { m } \times d _ { c }$ . The convolutional fusion mapping $\Phi _ { f }$ supplies complementary features, whose contribution is controlled by $\beta _ { f }$ . The residual path preserves the intermediate-scale representation, while the fusion mapping learns complementary information from the other scales.

The “Flatten & project” block in Figure 2 flattens the fused features and matches their channel dimension to that of the ViT. The “Memory encoder” then uses self-attention to model relationships among local structures, producing the shared memory:

$$
\begin{array} { r l } & { \mathbf { T } = \mathrm { F l a t t e n } ( \mathbf { F } ) \mathbf { W } _ { m } , } \\ & { \mathbf { M } = \mathrm { L N } _ { 2 } ( \mathbf { T } + \mathbf { M } \mathrm { H S A } ( \mathrm { L N } _ { 1 } ( \mathbf { T } ) ) ) . } \end{array}\tag{4}
$$

The resulting memory has dimensions M $\epsilon \mathbb { R } ^ { N _ { m } \times d }$ , with $\begin{array} { r c l } { N _ { m } } & { = } & { h _ { m } w _ { m } } \end{array}$ , and MHSA denotes multihead self-attention. Computed from the current image, this memory is shared across all selected ViT layers. Each layer retrieves the information it needs through an independent conditional refinement module.

## 3.3. Cross-Encoder Anchor Refinement and Discriminative Prototypes

SER compresses the patch sequence into a set of spatial anchors. Each layer has learnable anchor seeds $\mathbf { A } _ { \ell } \in \mathbb { R } ^ { K \times d }$ , from which queries, keys, and values are constructed for spatial readout:

$$
\begin{array} { r l } & { \mathbf { Q } _ { \ell } = \mathrm { L N } _ { q } ( \mathbf { A } _ { \ell } \mathbf { W } _ { q , \ell } ) , } \\ & { \mathbf { K } _ { \ell } = \mathrm { L N } _ { k } ( ( \mathbf { X } _ { \ell } + \mathbf { P } _ { \ell } ) \mathbf { W } _ { k , \ell } ) , } \\ & { \mathbf { V } _ { \ell } = \mathbf { X } _ { \ell } \mathbf { W } _ { \nu , \ell } . } \end{array}\tag{5}
$$

The positional embedding $\mathbf { P } _ { \ell } \in \mathbb { R } ^ { N \times d }$ is added only to the key branch. Queries and keys are layer-normalized after projection. Spatial readout weights then aggregate the projected patch features into anchors:

$$
\begin{array} { r l } & { \mathbf { B } _ { \ell } = \operatorname { s o f t m a x } ( \mathbf { Q } _ { \ell } \mathbf { K } _ { \ell } ^ { \top } / \sqrt { d } ) , } \\ & { \mathbf { Z } _ { \ell } = \mathbf { B } _ { \ell } \mathbf { V } _ { \ell } . } \end{array}\tag{6}
$$

The softmax operation normalizes over patches, yielding spatial readout weights $\mathbf { B } _ { \ell } ~ \in ~ \mathbb { R } ^ { K \times N }$ and aggregated anchors $\mathbf { Z } _ { \ell } ~ \in ~ \mathbb { R } ^ { K \times d }$ These anchors provide intermediate evidence shared by the normal and anomalous branches. Class-specific discriminative prototypes are constructed subsequently through query-gated aggregation.

CCAR takes $\mathbf { Z } _ { \ell }$ as input and sequentially applies anchor self-attention, cross-attention to the CNN memory, and a feed forward update (Figure 2):

$$
\begin{array} { r l } & { \mathbf { Z } _ { \ell } ^ { ( 1 ) } = \mathbf { Z } _ { \ell } + \mathbf { M H S A } _ { \ell } ( \mathbf { L N } _ { s , \ell } ( \mathbf { Z } _ { \ell } ) ) , } \\ & { \mathbf { Z } _ { \ell } ^ { ( 2 ) } = \mathbf { Z } _ { \ell } ^ { ( 1 ) } + \mathbf { M H C A } _ { \ell } ( \mathbf { L N } _ { q , \ell } ( \mathbf { Z } _ { \ell } ^ { ( 1 ) } ) , \mathbf { M } , \mathbf { M } ) , } \\ & { \mathbf { Z } _ { \ell } ^ { \prime } = \mathbf { Z } _ { \ell } ^ { ( 2 ) } + \mathrm { F F N } _ { \ell } ( \mathbf { L N } _ { f , \ell } ( \mathbf { Z } _ { \ell } ^ { ( 2 ) } ) ) . } \end{array}\tag{7}
$$

MHCA denotes multihead cross-attention [40], with normalized anchors as queries and the shared memory M as keys and values. Self-attention models relationships among anchors, whereas cross-attention retrieves local features relevant to the current ViT evidence from the independently encoded memory. The feed-forward network then applies a nonlinear transformation to the updated anchors. All three sublayers use residual connections, and each selected layer has independent refinement parameters. The refined anchors $\mathbf { Z } _ { \ell } ^ { \prime }$ combine ViT evidence with CNN conditioning information and directly support prototype construction.

QGPC uses the normal and anomalous queries to select and aggregate evidence from the refined anchors:

$$
\begin{array} { r l } & { \mathbf { g } _ { \ell , c } = \sigma ( \mathbf { t } _ { c } \mathbf { W } _ { g , \ell } + \mathbf { b } _ { g , \ell } ) , } \\ & { \mathbf { p } _ { \ell , c } = \mathbf { t } _ { c } + \alpha _ { \ell } ( ( \mathbf { g } _ { \ell , c } \mathbf { Z } _ { \ell } ^ { \prime } ) \mathbf { W } _ { o , \ell } + \mathbf { b } _ { o , \ell } ) . } \end{array}\tag{8}
$$

Here, $\mathbf { g } _ { \ell , c } ~ \in ~ ( 0 , 1 ) ^ { 1 \times K }$ contains independent sigmoid gates, and $\alpha _ { \ell }$ is a learnable residual coeficient. The matrix product $\mathbf { g } _ { \ell , c } \mathbf { Z } _ { \ell } ^ { \prime }$ weights and aggregates the refined anchors. Together with the two queries, this aggregation yields image-conditioned normal and anomalous prototypes $\mathbf { p } _ { \ell , c }$ at each layer. The two class branches share parameters for spatial readout and conditional refinement; their distinction is expressed through the queries and gating weights.

CCAR introduces local evidence from the independent CNN into the ViT-derived anchors. QGPC then combines imageconditioned queries with gated anchor contributions to construct normal and anomalous prototypes. The two encoding paths therefore interact through intermediate anchors before prototype aggregation. The CNN serves as a source of local evidence in this design. Section 4.3.1 compares the CNN-only model, fusion controls, and full model to examine how standalone detection performance relates to its benefit as a conditioning source.

## 3.4. Patch Comparison and Anomaly Scoring

A patch projection MLP maps each layer’s patch features into a space compatible with its discriminative prototypes. It contains two linear layers that preserve the channel dimension, separated by a GELU activation. For the projected features $\mathbf { Y } _ { \ell } = \mathbf { M } \mathbf { L } \mathbf { P } _ { \ell } ( \mathbf { X } _ { \ell } )$ , the diference in cosine similarity to the anomalous and normal prototypes defines the patch anomaly response:

$$
\delta _ { \ell , i } = \cos ( \mathbf { Y } _ { \ell , i } , \mathbf { p } _ { \ell , a } ) - \cos ( \mathbf { Y } _ { \ell , i } , \mathbf { p } _ { \ell , n } ) .\tag{9}
$$

Here, i indexes patch positions, and cos(· ·) denotes cosine similarity. The response $\delta _ { \ell , i }$ increases as a patch becomes more similar to the anomalous prototype or less similar to the normal prototype.

We reshape each layer’s responses into a spatial grid and bilinearly upsample them. An unweighted sum forms the pixellevel anomaly map, and the image score averages a fraction of its highest-response pixels:

$$
\mathbf { H } ( \mathbf { x } ) = \sum _ { \ell \in J } \mathcal { U } ( \mathrm { R e s h a p e } ( \delta _ { \ell } ) ) ,\tag{10}
$$

$$
s ( \mathbf { x } ) = \frac { 1 } { k } \sum _ { i \in \mathcal { T } _ { k } ( \mathbf { H } ( \mathbf { x } ) ) } H _ { i } ( \mathbf { x } ) .\tag{11}
$$

Here, U upsamples the patch grid to $H _ { 0 } \ \times \ W _ { 0 }$ , and $\mathcal { T } _ { k } ( \mathbf { H } )$ returns the k highest-response locations, where $k \_ =$ max $\mathrm { ~ 1 , ~ } \lfloor \rho H _ { 0 } W _ { 0 } \rfloor )$ . The fraction $\rho$ determines how many pixels contribute to image-level discrimination. This aggregation retains the contribution of localized anomalies while reducing sensitivity to a single extreme response.

## 3.5. Joint Training Objective

The training objective combines task prediction, discriminative margins, and region completeness. We use binary crossentropy for image-level prediction and supervise the deeper branches $\mathcal { T } _ { s } \subset \mathcal { T }$ with focal and Dice segmentation losses [41, 42]:

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathrm { B C E W i t h L o g i t s } ( s ( \mathbf { x } ) , y ) ,\tag{12}
$$

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { s e g } } = \sum _ { \ell \in I _ { s } } [ \mathcal { L } _ { \mathrm { f o c a l } } ( \sigma ( \mathbf { h } _ { \ell } ) , \mathbf { m } ) } } \\ & { } & { + \sum _ { \mathrm { D i c e } } ( \sigma ( \mathbf { h } _ { \ell } ) , \mathbf { m } ) ] . } \end{array}\tag{13}
$$

Here, $\mathbf { h } _ { \ell } = \mathcal { U } ( \mathrm { R e s h a p e } ( \delta _ { \ell } ) ) , \sigma$ is the sigmoid function, and m is the ground-truth pixel mask. Segmentation losses are computed separately for each layer and summed over the selected deeper branches. A directional margin on the imageconditioned queries maintains separation between the normal and anomalous classes:

$$
\mathcal { L } _ { \mathrm { q r y } } = \mathbb { E } _ { \mathbf { x } } [ \operatorname* { m a x } ( 0 , \cos ( \mathbf { t } _ { a } , \mathbf { t } _ { n } ) + \tau ) ] .\tag{14}
$$

We use the positive-part notation $[ z ] _ { + } ~ = ~ \operatorname* { m a x } ( 0 , z )$ , and τ specifies the query margin. Applied to the queries $\mathbf { t } _ { a } , \mathbf { t } _ { n } .$ , this loss penalizes similarity above the prescribed boundary.

We additionally impose region-level margin supervision on the patch grid. Nearest-neighbor downsampling of the groundtruth mask defines anomalous locations $\Omega ^ { + }$ and normal locations $\Omega ^ { - }$ . For each image, the region loss is defined as

$$
\begin{array} { r l } & { \mathcal { R } _ { \ell } ( \eta ^ { + } , \eta ^ { - } , w , r ) } \\ & { \quad = \mathcal { H } _ { r } ( \{ [ \eta ^ { + } - \delta _ { \ell , i } ] _ { + } : i \in \Omega ^ { + } \} ) } \\ & { \quad \quad + w \mathcal { H } _ { r } ( \{ [ \eta ^ { - } + \delta _ { \ell , i } ] _ { + } : i \in \Omega ^ { - } \} ) . } \end{array}\tag{15}
$$

The margins $\eta ^ { + }$ and $\eta ^ { - }$ correspond to foreground and background, respectively, and w weights the background term. The operator $\mathcal { H } _ { r }$ averages the largest fraction r of values in its input set. Hard examples are selected independently for foreground and background, and each term is averaged over images containing a nonempty corresponding region. Images with an empty region are excluded from that term’s reduction. Diferent parameters are used for the deeper branches and the shallower branch set $\textstyle I _ { t } \colon$

$$
\mathcal { L } _ { \mathrm { s e m } } = \frac { 1 } { | \mathcal { T } _ { s } | } \sum _ { \ell \in \bar { \mathcal { I } } _ { s } } \mathcal { R } _ { \ell } ^ { \mathrm { b a t c h } } ( \eta _ { s } ^ { + } , \eta _ { s } ^ { - } , w _ { s } , r _ { s } ) ,\tag{16}
$$

$$
\mathcal { L } _ { \mathrm { t e x } } = \frac { 1 } { | \mathcal { T } _ { t } | } \sum _ { \ell \in \bar { \cal J } _ { t } } \mathcal { R } _ { \ell } ^ { \mathrm { b a t c h } } ( \eta _ { t } ^ { + } , \eta _ { t } ^ { - } , w _ { t } , r _ { t } ) .
$$

Here, $\mathcal { R } _ { \ell } ^ { \mathrm { b a t c h } }$ denotes the region loss after the image-wise reduction described above. The deeper-layer constraint promotes overall separation of normal and anomalous regions. The shallower-layer constraint emphasizes dificult local responses to reduce spurious activations caused by normal textures. This assignment distinguishes supervision roles; each layer may still contain both local and global information.

Finally, we use $\pi = \sigma ( \mathbf { H } )$ to compute soft true positives, false positives, and false negatives. A Tversky loss encourages region completeness in the fused anomaly map [43]:

$$
\mathcal { L } _ { \mathrm { m a p } } = \mathbb { E } _ { \mathbf { x } } [ 1 - \frac { \mathrm { T P } + \epsilon } { \mathrm { T P } + a _ { T } \mathrm { F P } + b _ { T } \mathrm { F N } + \epsilon } ] .\tag{17}
$$

The coeficients $a _ { T } , b _ { T }$ control the relative costs of false positives and false negatives, respectively, while ϵ provides numerical stability. The joint objective is

$$
\begin{array} { r l } & { \mathcal { L } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { q r y } } \mathcal { L } _ { \mathrm { q r y } } } \\ & { \quad + \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { \mathrm { t e x } } \mathcal { L } _ { \mathrm { t e x } } + \lambda _ { \mathrm { m a p } } \mathcal { L } _ { \mathrm { m a p } } . } \end{array}\tag{18}
$$

The coeficients λ weight the loss terms; their values and the training configuration are given in Section 4.1. Anomaly adaptation updates the queries, projections, memory encoding, and conditional refinement modules, together with the shared layer normalization at the ViT output. The ViT backbone and CNN encoder remain frozen throughout anomaly adaptation.

## 4. Experiments

We evaluate how CEAR generalizes across industrial categories and medical imaging domains, then examine the contribution of cross-encoder anchor refinement under controlled conditions. The ablations isolate input resolution, industrial pretraining, multiscale memory, conditional interactions, and supervision at diferent layers. Qualitative analyses complement these comparisons and help characterize the model’s range of applicability.

## 4.1. Experimental Setup

## 4.1.1. Datasets and Training Protocol

We evaluate anomaly detection and localization on 13 industrial and medical datasets. The industrial benchmarks are MVTec AD [1], VisA [44], BTAD [45], DTD-Synthetic [46], and MPDD [47]. All five support image-level detection and pixel-level localization. MVTec AD and VisA alternate as auxiliary and target domains in the two training directions. DTD-Synthetic denotes the texture benchmark with synthetic anomalies, rather than the original texture classification task.

Table 1: Comparison of zero-shot anomaly detection methods on 13 industrial and medical datasets. Image-level metrics are AUROC, F1-max and AP; pixel-level metrics are AUROC, F1-max, AP and PRO. Best and second-best reported values for each dataset and metric are shown in red and blue.
<table><tr><td colspan="4">Domain Metric</td><td>WinCLIP</td><td>APRIL-GAN</td><td>CLIP-AD</td><td>AnomalyCLIP</td><td>AdaCLIP</td><td>VisualAD (CLIP)</td><td>CEAR</td></tr><tr><td colspan="3"></td><td>ViT-L/14 @336px</td><td>ViT-L/14 @336px</td><td>ViT-L/14 @336px</td><td>ViT-L/14 @336px</td><td>ViT-L/14 @336px</td><td>ViT-L/14 @336px</td><td>MoonViT @672px</td></tr><tr><td></td><td></td><td>VisA</td><td>(75.6, 78.2, 78.8)</td><td>77.4, 78.6, 80.9)</td><td>(66.2, 74.3, 71.4)</td><td>(81.0, 80.3, 84.4)</td><td>(79.7, 79.6, 83.2)</td><td>(84.7, 82.5, 87.6)</td><td>(84.8, 82.7, 88.5)</td></tr><tr><td></td><td>Image-level</td><td>MVTec-AD</td><td>(90.4, 92.7, 95.6)</td><td>(86.1, 90.4, 93.6)</td><td>74.1, 86.3, 88.1) (</td><td>(91.6, 92.7, 96.2)</td><td>(92.0, 92.7, 96.4)</td><td>92.2, 93.2, 96.7)</td><td>(92.4, 92.7, 96.5)</td></tr><tr><td></td><td>(AUROC, F1-max, AP)</td><td>BTAD</td><td>(68.2, 67.8, 70.9)</td><td>(73.7, 68.7, 69.9)</td><td>66.7, 65.9, 67.3)</td><td>(88.7, 86.0, 90.6)</td><td>(90.0, 87.2, 91.5)</td><td>( 94.9, 93.9, 97.0)</td><td>94.9, 87.9, 92.6)</td></tr><tr><td></td><td></td><td>DTD</td><td>(95.1, 94.1, 97.7)</td><td>(85.5, 89.1, 94.0)</td><td>( (75.1, 86.3, 88.0)</td><td>( 93.7, 94.3, 97.4)</td><td>( 92.1, 92.4, 96.3)</td><td>( 97.5, 96.6, 99.1)</td><td>( 97.6, 97.4, 99.3)</td></tr><tr><td>Inautuial</td><td></td><td>MPDD</td><td>( 61.4, 77.5, 69.9)</td><td>(73.0, 76.0, 79.8)</td><td>( 74.5, 72.9, 77.9)</td><td>( 77.0, 75.4, 82.0)</td><td>(76.0, 82.5, 77.1)</td><td>( 71.5, 81.6, 79.5)</td><td>78.2, 85.2, 80.2)</td></tr><tr><td colspan="2"></td><td>VisA (73.2, 9.0,</td><td colspan="7">5.4,51.1) (93.8,32.6,26.2,86.5) (93.0,24.1,17.9,80.2) (95.4,27.6,20.7,86.4) (95.1,33.8,29.2,71.3) (95.8,34.6,28.4,91.0) )(</td></tr><tr><td></td><td>Pixel-level (AUROC,</td><td></td><td>MVTec-AD (82.3,24.8,18.2,62.0) (87.5,42.3,39.1,43.7) (77.9,26.3,21.1,55.7) (91.0,38.9,34.4,81.7) (88.5,43.9,41.0,47.6) (90.8,43.9,41.2,87.5)</td><td></td><td></td><td></td><td></td><td></td><td>(91.2,46.3,45.0,87.9)</td></tr><tr><td></td><td>F1-max, AP, PRO)</td><td>BTAD</td><td>(72.7,18.5,12.9,27.3) (91.3,40.1,37.7,21.0) (80.9,24.1,18.3,41.4) (93.0,47.1,41.5,71.0) (87.7,42.3,36.6,17.1) (91.1,49.8,43.1,80.4)</td><td></td><td></td><td></td><td></td><td></td><td>(96.7,49.3,44.0,84.9)</td></tr><tr><td></td><td></td><td>DTD</td><td>(79.5,16.1, 9.8,51.5) (94.9,60.4,61.0,33.8) (86.6,35.8,31.0,63.2) (97.5,55.8,52.5,87.9) (95.1,58.4,56.1,34.3) (98.1,64.3,65.5,94.8)</td><td></td><td></td><td></td><td></td><td></td><td>(98.7,66.6,70.0,95.5)</td></tr><tr><td></td><td rowspan="2"></td><td>MPDD</td><td>(71.2,15.4,22.6,48.9) (95.1,30.6,28.9,83.1) (95.1,26.3,28.4,41.5) (96.5,32.1,29.5,88.7) (96.1,34.9,32.9,82.0)</td><td></td><td></td><td></td><td></td><td></td><td>(97.1,36.0,33.3,91.4)(97.4,40.8,35.5,92.7)</td></tr><tr><td>Image-level (AUROC,</td><td>COVID-19</td><td>(64.4, 42.5, 42.9) ( 15.5,</td><td>33.7, 40.6) (62.5,</td><td>56.3, 43.2)</td><td>80.1, 53.6, 58.7)</td><td>(64.5, 49.2, 50.3)</td><td>(43.8, 68.3, 45.9)</td><td>64.8, 68.4, 64.4)</td></tr><tr><td colspan="2"></td><td>Br35H</td><td>(80.5, 74.1, 82.2) (93.1, 85.4, 92.9)</td><td></td><td>(96.0, 90.8, 95.5)</td><td>97.3, 92.4, 96.1)</td><td>( 95.7, 89.1, 95.7)</td><td>( 94.4, 87.9, 91.6)</td><td>96.2, 93.3, 93.2)</td></tr><tr><td></td><td>F1-max, AP)</td><td>Brain AD</td><td>(72.5, 90.7, 91.6)</td><td>( 58.8, 90.6, 87.8)</td><td>72.1, 91.3, 91.5)</td><td>( 69.0, 90.6, 90.1)</td><td>(80.0, 90.2, 94.1)</td><td>( 80.8, 91.8, 94.7)</td><td>83.6, 92.6, 95.7)</td></tr><tr><td>Medccal</td><td></td><td>ClinicDB</td><td></td><td>(70.3,27.3,19.4,32.5) (82.4,38.2,36.4,53.5) (76.5,32.1,24.6,52.8) (84.6,45.5,41.7,69.8) (84.3,44.4,43.3,63.6) (85.2,45.6,37.8,70.6)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Endo</td><td></td><td>(68.2,32.9,23.8,28.3) (82.7,46.9,47.2,48.3) (78.1,41.4,36.8,46.4) (86.5,53.7,50.7,67.1) (82.9,47.9,47.7,55.8) (84.9,52.0,46.2,66.3)</td><td></td><td></td><td></td><td></td><td>(88.9,52.2,44.9,76.1)</td></tr><tr><td></td><td>Pixel-level (AUROC,</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(91.1,62.2,60.8,76.7)</td></tr><tr><td></td><td>F1-max,</td><td>Kvasir</td><td></td><td>(69.7,35.9,27.8,24.5) (77.6,43.2,42.5,40.6) (73.6,38.9,31.1,30.6) (82.0,50.0,43.2,44.7) (80.3,47.7,31.0,27.5) (80.3,47.8,47.6,50.8)(88.5,60.1,56.2,52.2)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AP, PRO)</td><td>ISIC</td><td></td><td>(83.3,53.4,62.4,55.1) (85.8,59.9,69.8,13.7) (81.6,63.6,65.5,29.0) (88.4,62.3,74.4,78.1) (85.4,61.9,70.6, 5.3) (84.0,65.1,67.7,73.1)(89.4,72.1,75.7,80.0)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>TN3K</td><td></td><td>(67.2,30.0,32.5,39.8) (75.9,36.4,35.4,37.8) (66.5,32.8,31.7,35.9)(76.8,40.5,36.7,50.3) (77.2,41.3,36.5,17.0)(65.5,32.7,30.4,41.1)(79.1,42.0,36.9,51.4)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Brain AD</td><td></td><td>(87.6,21.7,13.3,59.7) (83.6,38.7,35.0,45.4) (94.1,42.9,40.5,74.2) (95.1,43.1,42.3,71.5) (95.2,40.6,37.0,36.4) (95.2,46.7,43.7,79.5)(96.8,53.5,53.0,81.9)</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Medical image-level detection is evaluated on COVID-19 Radiography, Br35H, and Brain AD. COVID-19 Radiography is abbreviated as COVID-19 in the tables [48, 49]. Br35H follows Hamada [50], and Brain AD is a two-dimensional evaluation set derived from BraTS [51]. Medical localization uses CVC-ClinicDB [52], Endo (the EndoTect segmentation benchmark [53]), and Kvasir (Kvasir-SEG [54]). It also includes ISIC (ISIC 2016 [55]), TN3K [56], and Brain AD. These names follow the established anomaly detection benchmarks. The original datasets identify the data sources, whereas their anomaly detection subsets define the evaluation protocols. We therefore do not use sample counts from the original three-dimensional segmentation tasks or other dataset versions to describe the evaluated subsets.

For evaluation on VisA, we train on MVTec AD as the auxiliary dataset; all other target datasets are evaluated using the model trained on VisA. Auxiliary training uses normal/anomalous image labels and pixel masks, with one shared model per training direction. Target categories are excluded from anomaly adaptation, and inference requires neither targetdomain normal reference images nor parameter updates. The CNN is separately pretrained for product-category recognition on Real-IAD, then frozen throughout anomaly adaptation and testing [2].

## 4.1.2. Evaluation Metrics

We measure image-level performance using AUROC, maximum F1 score (F1-max), and average precision (AP). Pixellevel evaluation additionally uses the area under the per-region overlap curve (PRO). For datasets with multiple categories, metrics are computed within each category and then macroaveraged. The controlled evaluation protocol uses three random seeds and summarizes the category-macro-averaged results by their mean and standard deviation. Comparisons are made separately for each metric.

## 4.1.3. Implementation Details

We use MoonViT-SO-400M as the visual backbone, drawing on its multilayer patch representations and resolution flexibility for anomaly modeling at the encoder output [18]. Inputs are resized to 672 × 672 pixels. A patch size of 14 yields a 48 × 48 grid with 1,152 channels. We extract features from layers {7, 14, 21, 27} of the 27-layer encoder. The shallowest branch receives supervision on dificult local regions; the other three form the deeper branch set for segmentation supervision. During anomaly adaptation, both encoder bodies remain frozen, while the shared final layer normalization of MoonViT and the anomaly modeling modules are updated. MoonViT inputs are normalized with a mean and standard deviation of 0.5; the CNN retains the normalization used during its pretraining.

The local encoder is ConvNeXt V2-Base [20]. We extract features at strides 4, 8, and 16, with 128, 256, and 512 channels, respectively. Each scale is projected to 256 channels using a 1 × 1 convolution, group normalization with 32 groups, and GELU. A 3 × 3 convolution with stride 2 downsamples the finest scale, while bilinear interpolation upsamples the coarsest scale to a common $8 4 \times 8 4 ~ \mathrm { g r i d } .$ The fusion mapping uses two $1 \times 1$ convolutions, group normalization, and GELU. Its final layer is zero-initialized, with residual scaling $\beta _ { f } = 0 . 1$ . A bias-free memory projection produces 7,056 tokens of dimension 1,152.

Each selected layer uses four spatial anchors, and both memory encoding and conditional refinement use eight attention heads. The refinement feed-forward network has a hidden width of 4,608. Dropout is disabled in memory self-attention and set to 0.1 in conditional refinement and patch projection. The prototype residual coeficient $\alpha _ { \ell }$ is initialized to 0.01. Refined anchors pass directly to query-gated aggregation; all three refinement sublayers retain their internal residual connections. No additional pre-normalization is applied to the encoded memory.

The query margin τ is set to 0.5. For the deep region loss, the foreground margin, background margin, background weight, and hard-example fraction are 0.5, 0.5, 0.25, and 1, respectively. The corresponding shallow-layer values are 0.5, 0.7, 1, and 0.25. The focal loss uses a focusing parameter of 2, and the Dice smoothing constant is 1. The Tversky loss uses $a _ { T } = 0 . 3 ,$ $b _ { T } = 0 . 7$ , and $\epsilon = 1 0 ^ { - 6 }$ . Loss weights for classification, segmentation, query separation, deep regions, shallow regions, and the fused map are 1, 1, 1, 0.5, 0.5, and 0.25, respectively.

Anomaly adaptation uses AdamW [57] for three epochs with a batch size of 4. The base learning rate is 10<sup>−4</sup>, with weight decay of 0.01. The shared final layer normalization uses a learning rate of $1 0 ^ { - 5 }$ and no weight decay. An epoch-wise cosine schedule reduces the learning rate to a minimum of 0.1 times its initial value. Gradient norms are clipped at 0.5. The image score is the mean of the highest 1% of anomaly-map responses. Pixel-level evaluation applies Gaussian smoothing with a standard deviation of 4; image scores are computed before smoothing.

Industrial pretraining fine-tunes a generically pretrained ConvNeXt V2-Base to recognize the 30 product categories in Real-IAD. Normal/anomalous labels and pixel masks do not enter this classification objective. Training uses 384 × 384 inputs for 10 epochs, with 16 images per step and gradients accumulated over four steps. The AdamW learning rate is scaled by the efective batch size to $1 . 5 6 2 5 \times 1 0 ^ { - 4 }$ . Weight decay is 0.05, and the layer-wise learning-rate decay factor is 0.6. Augmentations comprise RandAugment, random erasing, Mixup, and CutMix [58–61]. The random-erasing probability is 0.25; the Beta-distribution parameters for Mixup and CutMix are 0.8 and 1.0, respectively. Label smoothing and stochastic depth are both set to 0.1, with an exponential moving average decay of 0.9999.

## 4.2. Comparison with State-of-the-Art Methods

## 4.2.1. Industrial Anomaly Detection and Localization

Table 1 compares CEAR with WinCLIP [4], APRIL-GAN [62], and CLIP-AD [63]. It also includes Anomaly-CLIP [5], AdaCLIP [21], and VisualAD [16]. This comparison evaluates each complete system. Subsequent ablations on VisA use matched backbones to isolate input resolution, industrial pretraining, and the interaction mechanism. Across 68 dataset– metric comparisons on 13 datasets, CEAR achieves 57 best results: 56 outright and one tied. Figure 3 relates these aggregate results to localization behavior through anomaly maps from industrial and medical target domains.

CEAR shows its strongest advantage in pixel-level localization, exceeding every other method in 41 of 44 comparisons and achieving the highest AUROC on all 11 localization datasets. Relative to the strongest competing result for each dataset and metric, mean gains in pixel AUROC, F1-max, AP, and PRO are 2.20, 4.45, 3.79, and 2.39 percentage points. These averages weight all 11 datasets equally and include comparisons in which CEAR does not rank first. CEAR ranks first in 17 of the 20 industrial localization comparisons. Pixel AP reaches 45.0 on MVTec AD, exceeding the strongest baseline by 3.8 percentage points. On MPDD, pixel F1-max reaches 40.8, a gain of 4.8 points; DTD-Synthetic pixel AP reaches 70.0, a gain of 4.5 points. The improvements span diverse industrial appearances and defect types, with complementary gains in pixel ranking, regional detection, and precision–recall performance.

## 4.2.2. Transfer to Medical Imaging

CEAR achieves the highest scores in all 24 pixel-level comparisons across the six medical localization datasets. On Endo, pixel AP increases from the strongest baseline value of 50.7 to 60.8, while F1-max increases from 53.7 to 62.2. Pixel F1-max rises from 50.0 to 60.1 on Kvasir, and pixel AUROC rises from 77.2 to 79.1 on TN3K. Under the protocol in Section 4.1.1, all medical datasets use the same model trained on VisA, without target-domain parameter updates. The localization gains therefore extend beyond industrial categories to brain, thyroid, skin, and endoscopic images. These results support cross-domain localization with industrial auxiliary supervision; the advantage is most pronounced at the pixel level.

Figure 3 compares localization on eight industrial and med ical sample types represented in Table 1. For the illustrated grid, woven-texture, and metal-tube samples, CEAR retains localized responses at the annotated defects. Its high-response regions also show close spatial correspondence with the annotated lesions in brain, thyroid, skin, and polyp images. Other methods exhibit scattered activations, incomplete responses, or responses extending beyond annotated regions in these exam ples. The comparisons illustrate how CEAR organizes anomaly evidence across diferent appearances and imaging conditions. Table 1 quantifies aggregate performance, while Figure 3 shows the corresponding response locations and regional coverage. Heatmaps are normalized individually, so qualitative comparisons concern spatial distribution and correspondence with the ground truth.

## 4.3. Ablation Studies

We conduct ablations on VisA [44], following the MVTec AD-to-VisA protocol in Section 4.1.1. VisA target categories are excluded from anomaly adaptation. All configurations share the training split, optimization settings, and evaluation procedure, except for the factor under study. The default input resolution is $6 7 2 \times 6 7 2$ . Tables 2–7 report all seven image-level and pixel-level metrics used in Table 1.

![](images/1628a026739a0b0bbef6caae541e1f0efd87fa7ed2a04fa975c300c83d21470e.jpg)  
Figure 3: Qualitative comparisons assess anomaly localization across diverse industrial and medical target domains. Columns show grid, woven-texture, tube, capsule, brain, thyroid, skin, and polyp examples, with input images, seven methods’ anomaly maps, and ground-truth masks arranged vertically. Compared with several competing methods, CEAR shows closer spatial correspondence to annotated defects and lesions, with fewer fragmented responses or activations outside the annotated regions.

## 4.3.1. Backbone Resolution and the Contribution of Local Evidence

Table 2 compares controlled variants using the same Moon-ViT backbone at input resolutions of 448 × 448, 560 × 560, and $6 7 2 \times 6 7 2$ . MoonViT-only retains image-conditioned queries, spatial readout, patch projection, and the default supervision, while removing the CNN conditioning branch. The CNN-only controls include a fixed-feature response probe and a learned discriminative head trained with the same auxiliary anomaly supervision. Score-level fusion, feature concatenation, and anchor refinement before prototype construction use the same training split and optimization settings. These controls distinguish the efects of spatial resolution, additional local representations, and the stage at which the two encoding paths interact.

Table 2: Ablation of input resolution and local-evidence integration on VisA, comparing single-encoder models, score fusion, feature concatenation, and CEAR. Bold and underlined values indicate the best and second-best results for each metric, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Input</td><td colspan="4">Pixel-level</td><td colspan="3">Image-level</td></tr><tr><td>AUROC</td><td>F1-max</td><td>AP</td><td>PRO</td><td>AUROC</td><td>F1-max</td><td>AP</td></tr><tr><td>MoonViT-only</td><td>448</td><td>95.2</td><td>31.5</td><td>25.9</td><td>85.0</td><td>79.5</td><td>78.5</td><td>84.1</td></tr><tr><td>MoonViT-only</td><td>560</td><td>95.3</td><td></td><td>31.926.7</td><td>85.4</td><td>81.5</td><td>80.0</td><td>85.5</td></tr><tr><td>MoonViT-only</td><td>672</td><td>95.5</td><td>33.3</td><td>27.1</td><td>87.5</td><td>82.3</td><td>80.1</td><td>85.9</td></tr><tr><td>CNN-only (energy)</td><td>672</td><td>80.8</td><td>13.2</td><td>8.4</td><td>60.4</td><td>64.9</td><td>73.6</td><td>68.4</td></tr><tr><td>CNN-only (head)</td><td>672</td><td>90.1</td><td>24.6</td><td>18.4</td><td>85.0</td><td>83.2</td><td>81.2</td><td>86.1</td></tr><tr><td>Score fusion</td><td>672</td><td>95.2</td><td>32.9</td><td>25.3</td><td>87.0</td><td>80.2</td><td>79.7</td><td>84.8</td></tr><tr><td>Feature concat.</td><td>672</td><td>95.2</td><td>32.8</td><td>25.7</td><td>87.5</td><td>82.9</td><td>80.6</td><td>86.4</td></tr><tr><td>CEAR</td><td>448</td><td>95.4</td><td>31.9</td><td>26.0</td><td>86.9</td><td>82.2</td><td>80.9</td><td>85.9</td></tr><tr><td>CEAR</td><td>560</td><td>95.5</td><td>32.2</td><td>27.1</td><td>87.1</td><td>83.3</td><td>81.3</td><td>87.1</td></tr><tr><td>CEAR (full)</td><td>672</td><td>95.9</td><td>34.9</td><td>28.4</td><td>88.5</td><td>84.8</td><td>82.7</td><td>88.5</td></tr></table>

![](images/5e0cdbfacd252a61ad41ebcd23bbd4641c975e768ecc119f8bae7ad2e0ec7826.jpg)  
Figure 4: Qualitative ablations distinguish the efects of increased input resolution and cross-encoder conditioning on anomaly localization. At matched resolution, CEAR reduces responses on normal ring structures and covers more of the annotated skin lesion.

CEAR improves all four pixel-level metrics over MoonViTonly at each of the three input resolutions. $\mathrm { A t } 6 7 2 \times 6 7 2$ , gains in pixel AUROC, F1-max, AP, and PRO are 0.4, 1.6, 1.3, and 1.0 percentage points, respectively. Image AUROC, F1-max, and AP improve by 2.5, 2.6, and 2.6 points. The full model also outperforms score-level late fusion and direct feature concatenation. Thus, performance depends on how local evidence enters prototype construction, beyond the addition of a local encoding path. The learned CNN-only head is competitive on some image-level metrics, but its localization remains weaker than that of MoonViT-only. Using CNN features as conditioning memory for anchor refinement yields the best result on every metric in Table 2. This comparison supports the joint use of local evidence and contextual representations.

Figure 4 complements Table 2 by comparing MoonViT-only at resolutions of 448 and 672 with CEAR at 672 on the same images. For the MVTec AD metal-nut sample, increasing the MoonViT-only resolution leaves broad responses on the normal ring structure. At the same 672 resolution, CEAR concentrates high responses on the annotated defect to the right and suppresses some activations on normal structures. For the ISIC sample, CEAR covers more of the annotated lesion, whereas both single-backbone configurations concentrate on its center. These examples show distinct changes in response distribution from denser spatial sampling and conditional interaction. The controlled comparisons in Table 2 provide the quantitative evidence for the contribution of the interaction mechanism.

Table 3: Ablation of CNN pretraining. Real-IAD pretraining uses productcategory supervision; all CNN variants remain frozen during anomaly adaptation. Boldface and underlining follow the convention used in Table 2.
<table><tr><td>Configuration</td><td>Pixel-level (AUROC, F1-max, AP, PRO)</td><td>Image-level (AUROC, F1-max, AP)</td></tr><tr><td>Random init.</td><td>(95.4, 33.5, 27.5, 88.1)</td><td>(84.2, 82.0, 87.7)</td></tr><tr><td>Generic pretraining</td><td>(95.6, 34.0, 26.9, 87.8)</td><td>(84.0, 82.2, 87.5)</td></tr><tr><td>Real-IAD pretraining</td><td>(95.9, 34.9, 28.4, 88.5)</td><td>(84.8, 82.7, 88.5)</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 4: Ablation of multiscale local memory. C4, C8, and C16 denote CNN features at strides 4, 8, and 16, respectively. Boldface and underlining follow the convention used in Table 2.
<table><tr><td>Configuration</td><td>Pixel-level (AUROC, F1-max, AP, PRO)</td><td>Image-level (AUROC, F1-max, AP)</td></tr><tr><td>C4</td><td>(95.2, 33.7, 26.7, 87.9)</td><td>(83.8, 82.4, 87.3)</td></tr><tr><td>C8</td><td>(95.5, 34.5, 27.5, 88.2)</td><td>(84.4, 82.4, 87.8)</td></tr><tr><td>C16</td><td>(95.4, 34.3, 27.0, 87.3)</td><td>(84.0, 82.3, 87.6)</td></tr><tr><td> $\mathrm { C 4 } + \mathrm { C 8 }$ </td><td>(95.5, 34.1, 27.0, 87.4)</td><td>(84.1, 82.1, 87.4)</td></tr><tr><td> $\mathbf { C } 8 + \mathbf { C } 1 6$ </td><td>(95.5, 34.4, 27.2, 87.5)</td><td>(84.1, 81.9, 87.7)</td></tr><tr><td> $\mathbf { C 4 } + \mathbf { C 8 } + \mathbf { C 1 6 }$ </td><td>(95.9, 34.9, 28.4, 88.5)</td><td>(84.8, 82.7, 88.5)</td></tr></table>

## 4.3.2. Industrial Pretraining and Multiscale Memory

Table 3 varies the initialization of the frozen CNN: random initialization, generic image pretraining, or Real-IAD productrecognition pretraining. Memory construction and subsequent interactions remain unchanged. Table 4 compares single-scale features, pairs of scales, and the full feature pyramid. All variants are aligned to the same memory grid to control the token count.

Industrial pretraining improves every localization metric over both random initialization and generic pretraining. The source of learned local representations therefore afects the usefulness of the conditioning memory. The finest scale preserves detail relevant to small defects but can also respond to normal texture; the coarsest scale provides broader component context. Combining these receptive fields in the full pyramid yields the strongest localization performance in Table 4, consistent with complementary contributions to defect coverage and background suppression.

## 4.3.3. Refinement Architecture and Interaction Stage

Table 5 removes anchor self-attention, CNN cross-attention, and the feed-forward update individually, and examines parameter sharing across layers and memory compression. Table 6 compares direct use of the refinement output with fixed residual mixing, learned gated mixing, and direct interactions at the patch level.

Removing CNN cross-attention reduces localization performance, while anchor self-attention and the feed-forward update provide further improvements in evidence organization and transformation. Independent refinement parameters at each layer outperform full parameter sharing, supporting depthspecific mappings between ViT representations and local conditioning evidence. Table 6 assesses anchor refinement against direct patch interactions using all four localization metrics and three detection metrics. Fixed residual mixing and learned gated mixing further expose the trade-ofs between image-level detection and pixel-level localization. Direct use of the refined anchors yields the highest scores on six metrics, although additional memory pre-normalization gives a slightly higher PRO.

![](images/a685df83115968417dfc87d367c842bc3927c14f4f82c43332379217bff8876f.jpg)  
Figure 5: Qualitative comparisons for normal textures with small anomalies, industrial structural defects, medical anomaly localization, and local boundary difer ences. Each group contains two examples. Green and magenta contours denote the ground-truth and predicted boundaries respectively.

## 4.3.4. Supervision across Layers

Table 7 compares individual layers, the three deeper layers, and all four layers. It also removes the query margin, deep region margin, shallow hard-region constraint, and fused-map supervision in turn. Applying the full segmentation loss uniformly to every layer provides a further control for the role of diferentiated supervision.

Table 5: Ablation of memory encoding and anchor refinement. SA, CA, and FFN denote self-attention, cross-attention, and feed-forward network, respectively. Boldface and underlining follow the convention used in Table 2.
<table><tr><td>Configuration</td><td>Pixel-level (AUROC, F1-max, AP, PRO)</td><td>Image-level (AUROC, F1-max, AP)</td></tr><tr><td>Full refinement</td><td>(95.9, 34.9, 28.4, 88.5)</td><td>(84.8, 82.7, 88.5)</td></tr><tr><td>w/o anchor SA w/o CNN CA</td><td>(95.6, 34.0, 26.9, 87.9)</td><td>(84.2, 82.2, 87.6)</td></tr><tr><td>w/o FFN Shared refinement</td><td>(95.5, 33.9, 27.3, 87.5) (95.5, 34.1, 26.9, 88.3) (95.6, 34.5, 27.3, 88.3)</td><td>(83.7, 82.0, 87.7) (84.2, 82.3, 87.9)</td></tr></table>

Table 6: Ablation of refinement-output integration and interaction stage. Direct use passes refined anchors to prototype construction while retaining internal residual connections. Patch CA (C4) applies patch-level cross-attention using stride-4 CNN features. Boldface and underlining follow the convention used in Table 2.
<table><tr><td>Configuration</td><td>Pixel-level (AUROC, F1-max, AP, PRO)</td><td>Image-level (AUROC, F1-max, AP)</td></tr><tr><td>Direct use (default)</td><td>( 95.9, 34.9, 28.4, 88.5)</td><td>84.8, 82.7, 88.5)</td></tr><tr><td>Fixed mix (α = 0.1)</td><td>9 C 95.6, 33.9, 26.8, 88.2)</td><td>(84.0, 82.3, 87.5)</td></tr><tr><td>Gated mix</td><td>(95.7, 34.2, 27.1, 88.5)</td><td>(84.2, 82.3, 87.6)</td></tr><tr><td>Memory pre-norm</td><td>(95.6, 34.2, 27.1, 88.6)</td><td>( 84.3, 82.4, 87.7)</td></tr><tr><td>Patch CA (C4)</td><td>(95.3, 34.0, 27.0, 88.4)</td><td>( 83.3, 81.5, 87.1)</td></tr></table>

Table 7: Ablation of layer selection and supervision. “All layers” denotes layers 7, 14, 21, and 27; “All-layer $\mathcal { L } _ { \mathrm { s e g } } { } ^ { \mathrm { , , } }$ applies segmentation supervision to all four. $\mathcal { L } _ { \mathrm { q r y } } , \mathcal { L } _ { \mathrm { s e m } } , \mathcal { L } _ { \mathrm { t e x } } .$ , and ${ \mathcal { L } } _ { \mathrm { m a p } }$ denote query-separation, deep-region, shallow hardregion, and fused-map losses, respectively. Boldface and underlining follow the convention used in Table 2.
<table><tr><td></td><td>Pixel-level (AUROC, F1-max, AP, PRO)</td><td>Image-level (AUROC, F1-max, AP)</td></tr><tr><td>Configuration</td><td></td><td></td></tr><tr><td>All layers (default) Layer 7 only</td><td>(95.9, 34.9, 28.4, 88.5)</td><td>(84.8, 82.7, 88.5) (67.9, 73.8)</td></tr><tr><td>Layer 27 only</td><td>(92.0, 18.9, 13.5, 77.1)</td><td>76.1,</td></tr><tr><td>Layers 14/21/27</td><td>(93.7, 27.2, 19.3, 83.8)</td><td>(84.6, 81.8, 87.7)</td></tr><tr><td></td><td>(95.4, 34.1, 26.9, 88.3)</td><td>(84.5, 82.5, 87.9)</td></tr><tr><td>All-layer  $\mathcal { L } _ { \mathrm { s e g } }$ </td><td>(95.5, 34.6, 27.7, 88.2)</td><td>(84.0, 82.2, 87.5)</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { t e x } }$ </td><td>(95.5, 33.3, 25.4, 88.7)</td><td>(84.7, 82.4, 88.4)</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { { s e m } } }$ </td><td>(95.6, 34.8, 28.2, 88.4)</td><td>(84.6, 82.2, 87.7)</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { q r y } }$ </td><td>(95.6, 34.1, 26.9, 88.4)</td><td>(84.4, 82.3, 87.9)</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { m a p } }$ </td><td>(95.5, 33.6, 26.4, 88.5)</td><td>(84.5, 82.5, 88.2)</td></tr></table>

Using the deeper branches preserves overall discrimination, but their small-defect recall remains below that of the full model. The shallow-only branch is more susceptible to responses on normal texture. The shallow hard-region constraint helps suppress local false activations, the deep region margin promotes foreground–background separation, and fused-map supervision supports regional completeness. These contributions do not improve every metric uniformly: removing the shallow region loss slightly increases PRO while reducing the other localization metrics. Taken together, the results support combining complementary layer representations with supervision suited to their respective roles.

![](images/591481dfd9680fc7743304fa058bc4ba925b6402b72813e33e5a7a1feeec130e.jpg)  
Figure 6: Layerwise visualizations examine how responses from diferent feature depths contribute to CEAR’s fused anomaly map. Prediction columns show layers ${ \bar { 7 } } ,$ 14, 21, and 27 from the same trained checkpoint, followed by the pointwise sum of their original score maps.

## 4.4. Qualitative Analysis

Figure 5 examines localization in four groups of examples: normal textures with small anomalies, industrial structural defects, medical anomalies, and local boundary diferences. The comparisons illustrate where CEAR better distinguishes defects from normal appearance and more closely follows annotated regions than the other methods shown.

Figure 6 provides a complementary view of the layerwise behavior discussed in Section 4.3.4. It compares the branches at layers 7, 14, 21, and 27 with the sum of their scores, all from the same trained checkpoint. Two industrial and two medical examples were selected from existing inference results on domains excluded from anomaly adaptation. They illustrate spatial diferences among the individual responses and the fused maps.

## 5. Conclusion

We presented CEAR, which integrates contextual representations and complementary local evidence through cross-encoder anchor refinement before prototype construction. Spatial anchors read from each selected ViT layer retrieve evidence from a shared local memory built by an independent encoder of the same image. Residual refinement incorporates this evidence before normal and anomalous queries gate anchor aggregation into image-conditioned prototypes. Across the comparisons on 13 industrial and medical datasets, CEAR achieves 57 best results among 68 dataset–metric evaluations, including 41 outright wins in 44 pixel-level comparisons. Across 11 localization datasets, pixel F1-max and AP improve by averages of 4.45 and 3.79 percentage points over the strongest competing result for each dataset. These findings support anchor-stage cross-encoder interaction as a means of integrating complementary visual representations for fine-grained anomaly localization across domains. The shared memory introduces additional computation, motivating future work on memory eficiency and stability under domain shifts.

## References

[1] P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, Mvtec ad—a comprehensive real-world dataset for unsupervised anomaly detection, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2019, pp. 9584–9592.

[2] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, L. Ma, Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection, in: 2024 IEEE/CVF conference on computer vision and pattern recognition (CVPR), IEEE, 2024, pp. 22883–22892.

[3] L. Heckler-Kram, J.-H. Neudeck, U. Scheler, R. König, C. Steger, The mvtec ad 2 dataset: Advanced scenarios for unsupervised anomaly detection, International Journal of Computer Vision 134 (2026) 175.

[4] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, O. Dabeer, Winclip: Zero-/few-shot anomaly classification and segmentation, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2023, pp. 19606–19616.

[5] Q. Zhou, G. Pang, Y. Tian, S. He, J. Chen, AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection, in: The Twelfth International Conference on Learning Representations, 2024. URL: https: //openreview.net/forum?id=buC4E91xZE.

[6] W. Ma, X. Zhang, Q. Yao, F. Tang, C. Wu, Y. Li, R. Yan, Z. Jiang, S. K. Zhou, Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, pp. 4744–4754.

[7] Z. Qu, X. Tao, X. Gong, S. Qu, Q. Chen, Z. Zhang, X. Wang, G. Ding, Bayesian prompt flow learning for zero-shot anomaly detection, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, pp. 30398–30408.

[8] J. Zhu, Y.-S. Ong, C. Shen, G. Pang, Fine-grained abnormality prompt learning for zero-shot anomaly detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 22241– 22251.

[9] F. Wang, J. Mei, A. Yuille, Sclip: Rethinking selfattention for dense vision-language inference, in: European conference on computer vision, Springer, 2024, pp. 315–332.

[10] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, W. Zhang, Clearclip: Decomposing clip representations for dense vision-language inference, in: European Conference on Computer Vision, Springer, 2024, pp. 143–160.

[11] T. Gong, Q. Chu, B. Liu, W. Zhou, N. Yu, Fe-clip: Frequency enhanced clip model for zero-shot anomaly detection and segmentation, in: 2025 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, 2025, pp. 21220–21230.

[12] G. Zhang, L. Zhang, Dlvp-clip: Enhancing fine-grained zero-shot anomaly detection via dynamic local visual prompting, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 35524–35533.

[13] M. Hu, Y. Huo, M. Dou, J. Yin, P. Zhao, Y. Wang, C. Hu, B. Hu, Q. Wang, Fb-clip: Fine-grained zero-shot anomaly detection with foreground-background disentanglement, arXiv preprint arXiv:2603.19608 (2026).

[14] T. Darcet, M. Oquab, J. Mairal, P. Bojanowski, Vision transformers need registers, in: International conference on learning representations, volume 2024, 2024, pp. 2632–2652.

[15] S. Fu, M. Hamilton, L. E. Brandt, A. Feldmann, Z. Zhang, W. Freeman, Featup: A model-agnostic framework for features at any resolution, in: International Conference on Learning Representations, volume 2024, 2024, pp. 45324–45350.

[16] Y. Hou, P. Li, Z. Liu, Y. Wang, Y. Ruan, J. Qiu, K. Xu, Visualad: Language-free zero-shot anomaly detection via vision transformer, arXiv preprint arXiv:2603.07952 (2026).

[17] M. Fucka, V. Zavrtanik, D. Sko ˇ caj, Anomalyvfm– ˇ transforming vision foundation models into zero-shot anomaly detectors, arXiv preprint arXiv:2601.20524 (2026).

[18] K. Team, A. Du, B. Yin, B. Xing, B. Qu, B. Wang, C. Chen, C. Zhang, C. Du, C. Wei, et al., Kimi-vl technical report, arXiv preprint arXiv:2504.07491 (2025).

[19] C. Xia, X. Wang, F. Lv, X. Hao, Y. Shi, Vit-comer: Vision transformer with convolutional multi-scale feature interaction for dense predictions, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2024, pp. 5493–5502.

[20] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, S. Xie, Convnext v2: Co-designing and scaling convnets with masked autoencoders, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Ieee, 2023, pp. 16133–16142.

[21] Y. Cao, J. Zhang, L. Frittoli, Y. Cheng, W. Shen, G. Boracchi, Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection, in: European conference on computer vision, Springer, 2024, pp. 55–72.

[22] Z. Qu, X. Tao, M. Prasad, F. Shen, Z. Zhang, X. Gong, G. Ding, Vcp-clip: A visual context prompting model for zero-shot anomaly segmentation, in: European Conference on Computer Vision, Springer, 2024, pp. 301–317.

[23] Z. Gu, B. Zhu, G. Zhu, Y. Chen, H. Li, M. Tang, J. Wang, Filo: Zero-shot anomaly detection by fine-grained description and high-quality localization, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 2041–2049.

[24] B.-B. Gao, Y. Zhou, J. Yan, Y. Cai, W. Zhang, M. Wang, J. Liu, Y. Liu, L. Wang, C. Wang, Adaptclip: Adapting clip for universal visual anomaly detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2026, pp. 4095–4103.

[25] Y. Shao, L. Wang, C. Li, P. Chen, Q. Liu, Promptmoe: Generalizable zero-shot anomaly detection via visuallyguided prompt mixtures, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2026, pp. 8878–8886.

[26] X. Li, Z. Huang, F. Xue, Y. Zhou, Musc: Zero-shot industrial anomaly classification and segmentation with mutual scoring of the unlabeled images, in: International Conference on Learning Representations, volume 2024, 2024, pp. 2172–2197.

[27] X. Li, F. Xue, Y. Zhou, Musc-v2: Zero-shot multimodal industrial anomaly classification and segmentation with mutual scoring of unlabeled samples, IEEE Transactions on Pattern Analysis and Machine Intelligence (2026).

[28] J. He, M. Cao, S. Peng, Q. Xie, Rareclip: Rarity-aware online zero-shot industrial anomaly detection, in: 2025 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, 2025, pp. 24478–24487.

[29] J. Zhu, G. Pang, Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 17826–17836. URL: https://openaccess.thecv f.com/content/CVPR2024/html/Zhu\_Toward\_Gen eralist\_Anomaly\_Detection\_via\_In-context\_R esidual\_Learning\_with\_Few-shot\_CVPR\_2024\_pa per.html.

[30] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, J. Wang, Univad: A training-free unified model for few-shot visual anomaly detection, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, pp. 15194–15203.

[31] B.-B. Gao, Metauas: Universal anomaly segmentation with one-prompt meta-learning, Advances in Neural Information Processing Systems 37 (2024) 39812–39836.

[32] J. Guo, S. Lu, W. Zhang, F. Chen, H. Li, H. Liao, Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, pp. 20405–20415.

[33] Z. Gu, B. Zhu, G. Zhu, Y. Chen, W. Ge, M. Tang, J. Wang, Anomalymoe: Towards a language-free generalist model for unified visual anomaly detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2026, pp. 4348–4356.

[34] T. Kerssies, N. Cavagnero, A. Hermans, N. Norouzi, G. Averta, B. Leibe, G. Dubbelman, D. De Geus, Your vit is secretly an image segmentation model, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, pp. 25303–25313.

[35] J. Yang, K. Z. Luo, J. Li, C. Deng, L. Guibas, D. Krishnan, K. Q. Weinberger, Y. Tian, Y. Wang, Denoising vision transformers, in: European Conference on Computer Vision, Springer, 2024, pp. 453–469.

[36] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., Dinov3, arXiv preprint arXiv:2508.10104 (2025).

[37] T. Wimmer, P. Truong, M.-J. Rakotosaona, M. Oechsle, F. Tombari, B. Schiele, J. E. Lenssen, Anyup: Universal feature upsampling, in: International Conference on Learning Representations, volume 2026, 2026, pp. 140700–140720.

[38] B.-B. Gao, C. Wang, One language-free foundation model is enough for universal vision anomaly detection, arXiv preprint arXiv:2601.05552 (2026).

[39] Y. Wan, J. Cai, et al., Beyond static anchors: Bounded prototype conditioning for language-free medical anomaly detection, arXiv preprint arXiv:2608.00442 (2026).

[40] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).

[41] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[42] F. Milletari, N. Navab, S.-A. Ahmadi, V-net: Fully convolutional neural networks for volumetric medical image segmentation, in: 2016 fourth international conference on 3D vision (3DV), Ieee, 2016, pp. 565–571.

[43] S. S. M. Salehi, D. Erdogmus, A. Gholipour, Tversky loss function for image segmentation using 3d fully convolutional deep networks, in: International workshop on

machine learning in medical imaging, Springer, 2017, pp. 379–387.

[44] Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer, SPotthe-Diference self-supervised pre-training for anomaly detection and segmentation, in: Computer Vision – ECCV 2022, volume 13690 of Lecture Notes in Computer Science, Springer, Cham, 2022, pp. 392–408. URL: https: //doi.org/10.1007/978-3-031-20056-4\_23. doi:10.1007/978-3-031-20056-4\_23.

[45] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, G. L. Foresti, Vt-adl: A vision transformer network for image anomaly detection and localization, in: 2021 IEEE 30th international symposium on industrial electronics (ISIE), IEEE, 2021, pp. 01–06.

[46] T. Aota, L. T. T. Tong, T. Okatani, Zero-shot versus manyshot: Unsupervised texture anomaly detection, in: 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2023, pp. 5553–5561.

[47] S. Jezek, M. Jonak, R. Burget, P. Dvorak, M. Skotak, Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions, in: 2021 13th International congress on ultra modern telecommunications and control systems and workshops (ICUMT), IEEE, 2021, pp. 66–71.

[48] M. E. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al Emadi, et al., Can ai help in screening viral and covid-19 pneumonia?, Ieee Access 8 (2020) 132665– 132676.

[49] T. Rahman, A. Khandakar, Y. Qiblawey, A. Tahir, S. Kiranyaz, S. B. A. Kashem, M. T. Islam, S. Al Maadeed, S. M. Zughaier, M. S. Khan, et al., Exploring the effect of image enhancement techniques on covid-19 detection using chest x-ray images, Computers in biology and medicine 132 (2021) 104319.

[50] A. Hamada, Br35H: Brain tumor detection 2020, https: //www.kaggle.com/datasets/ahmedhamada0/bra in-tumor-detection, 2020. Online.

[51] U. Baid, S. Ghodasara, S. Mohan, M. Bilello, E. Calabrese, E. Colak, K. Farahani, J. Kalpathy-Cramer, F. C. Kitamura, S. Pati, et al., The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification, arXiv preprint arXiv:2107.02314 (2021).

[52] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, F. Vilariño, Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians, Computerized medical imaging and graphics 43 (2015) 99–111.

[53] S. A. Hicks, D. Jha, V. Thambawita, P. Halvorsen, H. L. Hammer, M. A. Riegler, The endotect 2020 challenge: evaluation and comparison of classification, segmentation and inference time for endoscopy, in: International Conference on Pattern Recognition, Springer, 2021, pp. 263– 274.

[54] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. De Lange, D. Johansen, H. D. Johansen, Kvasir-seg: A segmented polyp dataset, in: International conference on multimedia modeling, Springer, 2019, pp. 451–462.

[55] D. Gutman, N. C. Codella, E. Celebi, B. Helba, M. Marchetti, N. Mishra, A. Halpern, Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (isbi) 2016, hosted by the international skin imaging collaboration (isic), arXiv preprint arXiv:1605.01397 (2016).

[56] H. Gong, G. Chen, R. Wang, X. Xie, M. Mao, Y. Yu, F. Chen, G. Li, Multi-task learning for thyroid nodule segmentation with thyroid region prior, in: 2021 IEEE 18th international symposium on biomedical imaging (ISBI), IEEE, 2021, pp. 257–261.

[57] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, arXiv preprint arXiv:1711.05101 (2017).

[58] E. D. Cubuk, B. Zoph, J. Shlens, Q. Le, Randaugment: Practical automated data augmentation with a reduced search space, Advances in neural information processing systems 33 (2020) 18613–18624.

[59] Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang, Random erasing data augmentation, in: Proceedings of the AAAI conference on artificial intelligence, volume 34, 2020, pp. 13001–13008.

[60] H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz, mixup: Beyond empirical risk minimization, arXiv preprint arXiv:1710.09412 (2017).

[61] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, Y. Yoo, Cutmix: Regularization strategy to train strong classifiers with localizable features, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6023–6032.

[62] X. Chen, Y. Han, J. Zhang, April-gan: A zero-/few-shot anomaly classification and segmentation method for cvpr 2023 vand workshop challenge tracks 1&2: 1st place on zero-shot ad and 4th place on few-shot ad, arXiv preprint arXiv:2305.17382 (2023).

[63] X. Chen, J. Zhang, G. Tian, H. He, W. Zhang, Y. Wang, C. Wang, Y. Liu, Clip-ad: A language-guided staged dual-path model for zero-shot anomaly detection, in: International Joint Conference on Artificial Intelligence, Springer, 2024, pp. 17–33.