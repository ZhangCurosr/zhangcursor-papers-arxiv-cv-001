# Grounding Isn’t Knowing: Do VLMs Need Object Localization for Spatial Reasoning?

Xiwei Liu<sup>1</sup>, Yulong Li<sup>1</sup>, Xinlin Zhuang<sup>1,2</sup>, Xuhui Li<sup>1</sup>, Zhixiang Lu<sup>3</sup>, Haolin Yang<sup>1</sup>, Imran Razzak<sup>1</sup>, Yutong Xie<sup>1∗</sup>

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence <sup>2</sup>The Chinese University of Hong Kong, <sup>3</sup>University Of Liverpool

## Abstract

Vision-language models (VLMs) can answer spatial questions, yet the mechanisms connecting object grounding to spatial reasoning remain poorly understood. It is underexplored whether spatial reasoning internally requires precise objects localization, or can bypass explicit localization through global layout cues. In this work, we investigate two representative model families, LLaVA-1.5 and Qwen2.5-VL, using a suite of mechanistic interpretability tools, including token ablation, layer-wise probing, attention knockout, and causal mediation analysis. We find that spatial relation prediction follows a staged grounding-to-reasoning process in which object-aligned tokens establish coarse target-reference anchors, while precise bounding-box boundaries are not required. Positional information becomes decodable before relation decisions emerge, and a small set of attention heads mediates the causal efects of both localization and spatial reasoning. The two tasks share early grounding-related processing but ultimately rely on partially distinct specialized pathways. Through rigorous experiments, we provide a token-, layer-, and head-level account of how VLMs transform object grounding into spatial relations, showing that knowing where objects are is not equivalent to knowing how they relate.

## Introduction

Spatial reasoning is essential for interacting with and reasoning about the visual world. Modern vision-language models (VLMs) (Bai et al. 2025a; Comanici et al. 2025) can answer questions about relative object positions, including whether one object is to the left of, behind, or in front of another (Kamath, Hessel, and Chang 2023; Chen et al. 2025). Yet a correct answer does not establish that the model has localized the queried objects or derived the relation from their positions. The same prediction may arise from object-specific evidence, coarse scene layout, contextual regularities, or language priors (Li et al. 2025c; Schaumlöfel, Vilas, and Roig 2026). Benchmark accuracy therefore reveals what a model predicts, but provides limited insight into how object grounding is transformed into a spatial decision (Tong et al. 2024).

Unlike text, an image is inherently organized across two spatial dimensions. Most VLMs nevertheless inherit a Transformer pipeline that partitions an image into patches and serializes them as a one-dimensional token sequence (Dosovitskiy et al. 2020). Because sequence order does not explicitly preserve the image’s native two-dimensional topology, spatial structure must be recovered through positional encodings and interactions among visual tokens (Chu et al. 2024; Huang et al. 2025b). Fixed-resolution models commonly use learnable absolute position embeddings, whereas dynamic-resolution architectures can encode horizontal and vertical coordinates through two-dimensional rotary position embeddings (2D-RoPE) (Huang et al. 2025a). These mechanisms preserve where visual tokens occur, but do not by themselves establish which tokens belong to the queried objects or how those objects relate (Li et al. 2025c; Bousselham et al. 2024; Zhang et al. 2025b; Li et al. 2025b).

Recent studies have strengthened explicit visual grounding in VLMs through localized visual tokenization, spatial prompting, grounded visual reasoning, and specialized localization heads (Ma et al. 2024; Dorkenwald et al. 2024; Kang et al. 2025). Coordinate-based instruction tuning has also improved spatial-relation performance, suggesting a functional connection between localization supervision and spatial reasoning (Ranasinghe et al. 2024). However, these findings show that localization can be learned or exploited, not that a model must compute precise object locations before answering a relational query. Emerging evaluations further reveal that strong performance on conventional grounding benchmarks can coexist with severe failures on discriminative relational queries (Li et al. 2026). Whether localization is a prerequisite, a parallel capability, or a correlated outcome of spatial reasoning therefore remains underexplored.

These advances expose a conceptual ambiguity: precise localization, relational grounding, and spatial reasoning are often evaluated as if they were interchangeable. Precise localization requires recovering object boundaries, relational grounding requires binding the target and reference to spatially anchored representations, and spatial reasoning requires transforming those representations into a directed relation. Recent training frameworks and benchmarks increasingly separate object localization from broader spatial understanding (Li et al. 2025a; Jia et al. 2025), while preferencebased and token-based grounding methods optimize coordinate fidelity as a distinct objective (Qiu et al. 2025; Ren et al. 2026). This distinction motivates our central question: must a VLM precisely localize both objects before predicting their relation, or can it reach the correct decision from coarser object-grounded layout signals?

![](images/8a5d80fa7ce9b2473010fc4034315ec39512721bd2bc45dd47a93618fd65d5b6.jpg)

![](images/0c5561d7348884d431560e0c2c05b045c984186bd40dd32a7ee8ac14c59e0984.jpg)

![](images/627f37b12964e420deac09d22afd091074858e55512e87cb2c0f2c0de97852e6.jpg)

![](images/c58373d06acf3191f627c1c09eeec4dcea6b09cb1f53efb6f063aaf59a46d7c0.jpg)  
Figure 1: (a) and (b): The geometry of the 1D absolute position embedding in LLaVA-1.5-7B. The visualization is performed via t-SNE for dimensionality reduction. The labels are set to row IDs and column IDs, respectively. (c) and (d): The PCA results of the direction vectors for left of / right of / in front of / behind with and without position embedding in Qwen2.5-7B.

To resolve this question, we investigate LLaVA-1.5 and Qwen2.5-VL across tokens, layers, and attention heads, combining object-region interventions, layer-wise probing, attention knockout, causal mediation, and targeted head ablation (Neo et al. 2025; Bi et al. 2025). Together, these analyses reveal a consistent but nontrivial separation between precise localization and spatial reasoning in how object-grounded evidence is represented and routed. Our main findings are:

1 Precise localization is not required for spatial reasoning. Removing object-interior evidence sharply degrades coordinate recovery while largely preserving directional judgments, showing that accurate bounding boxes are not a necessary intermediate for relation prediction.

2 Spatial reasoning relies more on coarse object layout than on exact boundary geometry. Relation accuracy remains stable after removing fine boundary evidence but declines when interventions extend into the surrounding context, indicating that coarse object-centered layout provides a more relevant spatial signal than exact geometry.

3 Position precedes relation without forming a strict serial bottleneck. Object positions become decodable in intermediate layers, whereas relation answers emerge later and may remain correct despite localization failures.

4 Localization and spatial reasoning are mediated by sparsely overlapping attention circuits. Object-grounded evidence is carried by a small set of shared heads, while being largely transformed through task-specific pathways.

## Background

Sequential Visual Representation in VLMs. A visionlanguage model typically consists of a visual encoder $f _ { V } , \mathbf { a }$ modality connector $f _ { C } ,$ and a language model $f _ { L }$ . Given an image $\dot { \mathbf { X } } _ { V }$ , the visual encoder first partitions it into an $H \times W$ grid of patches, $\mathbf { P } = ( \mathbf { p } _ { 1 } , \hdots , \mathbf { \bar { p } } _ { N _ { V } } ) \in \mathbb { R } ^ { N _ { V } \times D _ { V } }$ , where $\bar { N } _ { V } = \bar { H } W$ and $D _ { V }$ is the dimension of the visual encoder. The grid is then flattened in raster order, such that the patch at row r and column c is assigned the sequence index $i = r W +$ c. The visual encoder and connector transform the patch sequence into language-compatible visual embeddings,

$$
\mathbf { V } = f _ { C } ( f _ { V } ( \mathbf { P } ) ) \in \mathbb { R } ^ { N _ { V } \times D _ { L } } .\tag{1}
$$

Given text embeddings $\mathbf { T } \in \mathbb { R } ^ { N _ { T } \times D _ { L } } , f _ { L }$ receives

$$
{ \bf H } ^ { ( 0 ) } = [ { \bf V } ; { \bf T } ] \in \mathbb { R } ^ { ( N _ { V } + N _ { T } ) \times D _ { L } } .\tag{2}
$$

Although rasterization allows images to be processed as token sequences, it does not explicitly preserve their original two-dimensional organization. Positional encoding is therefore required to make attention computations sensitive to spatial structure (Dosovitskiy et al. 2020; Liu et al. 2023).

Position Encoding. Position encoding in visual transformers can be broadly divided into absolute and relative. Given an input sequence $\mathbf { X } = ( \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { N } ) \in \mathbb { R } ^ { N \times D }$ , absolute position encoding assigns a vector $\mathbf { e } _ { i }$ to each sequence position before being passed into the Transformer blocks:

$$
\tilde { \mathbf { X } } = \mathbf { X } + \mathbf { E } = ( \mathbf { x } _ { 1 } + \mathbf { e } _ { 1 } , \ldots , \mathbf { x } _ { N } + \mathbf { e } _ { N } ) .\tag{3}
$$

The position vectors may be fixed or learnable. For a fixedresolution image, each sequence index corresponds to a persistent patch coordinate. Consequently, a one-dimensional position-embedding table can still encode the row and column structure of the original image grid. Relative position encoding instead introduces the displacement between tokens into the attention computation. Rotary position embedding applies a position-dependent rotation $f ( \cdot )$ to queries and keys (Su et al. 2024). For query $\mathbf { q } _ { m }$ and key $\mathbf { k } _ { n }$ at positions m and $n ,$ their inner product can be written as

$$
\left. f ( \mathbf { q } _ { m } , m ) , f ( \mathbf { k } _ { n } , n ) \right. = \mathrm { R e } \left[ \mathbf { q } _ { m } \mathbf { k } _ { n } ^ { * } e ^ { \mathrm { i } \left( m - n \right) \theta } \right] ,\tag{4}
$$

where $( \cdot ) ^ { * }$ denotes complex conjugation and $\theta$ is a preset angular frequency. The resulting attention score depends explicitly on the relative displacement $m - n$

For two-dimensional visual inputs, each token is assigned coordinates $\mathbf { s } _ { i } = ( x _ { i } , y _ { i } )$ . The query and key representations are divided into horizontal and vertical subspaces, giving

$$
\begin{array} { r l } & { \langle f ( \mathbf { q } _ { i } , x _ { i } , y _ { i } ) , f ( \mathbf { k } _ { j } , x _ { j } , y _ { j } ) \rangle } \\ & { = \mathrm { R e } \left[ \mathbf { q } _ { i } ^ { x } ( \mathbf { k } _ { j } ^ { x } ) ^ { * } e ^ { \mathrm { i } ( x _ { i } - x _ { j } ) \theta } + \mathbf { q } _ { i } ^ { y } ( \mathbf { k } _ { j } ^ { y } ) ^ { * } e ^ { \mathrm { i } ( y _ { i } - y _ { j } ) \theta } \right] . } \end{array}\tag{5}
$$

Thus, 2D RoPE represents spatial interactions through the relative displacement $\Delta \mathbf { s } _ { i j } = ( x _ { i } - x _ { j } , \ y _ { i } - y _ { j } )$ , allowing the same positional mechanism to ggeneralize consistently across varying image resolutions and sequence lengths.

In this work, we study LLaVA-1.5 and Qwen2.5-VL as representative instantiations of these two widely used mechanisms. LLaVA-1.5 uses learnable one-dimensional absolute position embeddings in its visual encoder (Liu et al. 2023), whereas Qwen2.5-VL employs 2D RoPE to support dynamic-resolution inputs (Bai et al. 2025b).

From Positional Geometry to Object Grounding. To verify how absolute and relative position encodings preserve spatial structure, we first apply t-SNE to reduce the dimensionality of the position embedding from LLaVA-1.5-7B to two dimensions. As visualized in Figure 1(a)(b), the geometric structure of the absolute position embedding exhibits distinct rows and columns. In the 2D-RoPE setting, we summarize an object o by mean-pooling its aligned visual tokens,

$$
\bar { \mathbf { h } } _ { o } = \frac { 1 } { | S _ { o } | } \sum _ { i \in \mathcal { S } _ { o } } \mathbf { h } _ { i } ,\tag{6}
$$

and characterize a target-reference pair using the direction vector ${ \bf d } _ { t , r } \ = \ \bar { \bf h } _ { t } \mathrm { ~ - ~ } \bar { \bf h } _ { r }$ . Figure 1(c)(d) demonstrates the collinearity and orthogonality of the direction vectors, while the geometry breaks down without 2D RoPE.

These observations show that VLM representations preserve recoverable spatial geometry, but encoding spatial structure does not explain how it supports relation reasoning. A model may first recover object boundaries and compare them, or infer the relation directly from coarse objectcentered cues. If localization is necessary, relation accuracy should decline when bounding-box localization fails. Otherwise, relation judgments may remain correct as long as coarse spatial grounding is preserved. We test this distinction through controlled interventions and trace the information across visual tokens, decoder layers, and attention heads.

## Preliminaries

## Dataset and Task

To ensure that our analyses isolate the object-level visual evidence used by VLMs for spatial perception, we construct a carefully curated dataset derived from the What’s Up benchmark (Kamath, Hessel, and Chang 2023).

Base Dataset and Filtering. We use controlled real-world images from What’s Up Subsets A and B. Subset A contains objects positioned on, under, left of, or right of a table, chair, or armchair. We remove samples labeled on or under and retain only the left-right relations. Subset B contains pairs of household objects arranged in front of, behind, left of, or right of one another, all of which are retained. We further annotate the target and reference objects with bounding boxes and segmentation masks. After filtering, the dataset contains 1,228 object annotations across 614 images, including 206 images from Subset A and 408 images from Subset B.

Object-Removed Control Set. Contextual cues can lead to hallucinated detections, where models predict the presence of an object solely from background context. To control for this efect, we construct an auxiliary object-removed variant of the dataset. For each image, the target object is removed and the missing region is inpainted using LaMa (Suvorov et al. 2022), which reconstructs background structure with high realism. We retain only those image pairs for which a model correctly identifies the object in the original image but fails to do so in the inpainted counterpart. This ensures that subsequent analyses rely on real object evidence rather than contextual correlations. Examples ofthe inpainted dataset are provided in Figure 2. More details are shown in Supplement.

![](images/92925a395bfd5f4ae65b96120a52cd18aae921dc0d3d5a4a6335e78e78f26493.jpg)  
Figure 2: Examples in the curated What’s Up dataset.

We evaluate models on two complementary visual tasks: localizing the target and reference objects and predicting their spatial relation. For each task, we design a diferent prompt for the same image. See detailed prompts in Supplement.

Localization. We prompt the model to predict target and reference bounding boxes separately. Each parsed prediction is compared with its ground-truth box using intersection over union (IoU). We report target and reference localization as the mean success rate at IoU thresholds of 0.5, 0.7, and 0.9.

Spatial Reasoning. We use the original four-way multiplechoice protocol from What’s Up, preserving its subsetspecific distractors. Accuracy is the proportion of predicted target-to-reference relations that match the ground-truth.

## Experiments

We examine localization and spatial reasoning at progressively finer scales. Token interventions test whether object evidence is necessary for relation prediction, layer-wise analyses trace when relevant information emerges and is used, and head-level mediation and ablation identify the attention components that route grounded evidence into outputs.

## Visual Information Ablation

We conduct an ablation study to investigate the contribution of visual input tokens to the performance of the VLMs.

Method. We ablate visual information at the LLM input, i.e., after the multimodal projection but before positional encodings and autoregressive processing. To remove imagespecific content while preserving domain-consistent embedding statistics, we replace the original visual token embeddings with a global average visual embedding computed once over the ImageNet (Deng et al. 2009) validation set.

(i) Object Tokens: We rasterize each segmentation mask onto the visual-token grid and select all tokens that overlap with it by at least one pixel. To probe for boundary sensitivity and context dependence, we shrink or dilate the mask by 1 or 2 token padding. This produces nested regions ranging from the object interior to its surrounding context.

<table><tr><td>Models:</td><td colspan="4">LLaVA-1.5 7B</td><td colspan="3">LLaVA-1.5 13B</td><td colspan="4">Qwen2.5-VL 7B</td></tr><tr><td>Strategy</td><td>Token (%)</td><td>Loc-T (%)</td><td>Loc-R (%)</td><td>Rel. (%)</td><td>Loc-T (%)</td><td>Loc-R (%)</td><td>Rel. (%)</td><td>Token (%)</td><td>Loc-T (%)</td><td>Loc-R (%)</td><td>Rel. (%)</td></tr><tr><td>Baseline</td><td>0</td><td>10.15</td><td>14.81</td><td>24.82</td><td>29.91</td><td>39.03</td><td>40.23</td><td>0</td><td>84.53</td><td>86.64</td><td>96.11</td></tr><tr><td>− 1 Padding</td><td>1.77</td><td>8.79 ↓1.36</td><td>14.33 ↓0.48</td><td>24.61 ↓0.21</td><td> $2 3 . 5 1 \downarrow 6 . 4 0$ </td><td>39.25 ↑0.22</td><td>40.07 ↓0.16</td><td>2.24</td><td>84.58 ↑0.05</td><td>86.75 ↑0.11</td><td>95.93 ↓0.18</td></tr><tr><td>Target</td><td>4.94</td><td>4.78 ↓5.37</td><td>14.71 ↓0.10</td><td>24.59 ↓0.23</td><td> $2 . 0 1 \downarrow 2 7 . 9 0$ </td><td>39.90 10.87</td><td>40.05 ↓0.18</td><td>4.23</td><td>19.60↓64.93</td><td>86.32 ↓0.32</td><td>94.79 ↓1.32</td></tr><tr><td>+ 1 Padding</td><td>9.52</td><td> $0 . 3 8 \downarrow 9 . 7 7$ </td><td> $1 7 . 5 9 \uparrow 2 . 7 8$ </td><td>24.57 ↓0.25</td><td> $0 . 1 2 \downarrow 2 9 . 7 9$ </td><td>36.91 ↓2.12</td><td>27.85 ↓12.38</td><td>6.71</td><td>2.55 ↓81.98</td><td>84.69 ↓1.95</td><td>89.74 ↓6.37</td></tr><tr><td>- 1 Padding</td><td>2.59</td><td>10.53 ↑0.38</td><td> $1 4 . 1 2 \downarrow 0 . 6 9$ </td><td>24.75 ↓0.07</td><td>29.64 ↓0.27</td><td> $3 1 . 3 8 \downarrow 7 . 6 5$ </td><td>39.90 ↓0.33</td><td>3.26</td><td>84.26↓0.27</td><td>88.60 ↑1.96</td><td>96.42 ↑0.31</td></tr><tr><td>Reference</td><td>7.64</td><td>10.58 ↑0.43</td><td>7.11 ↓7.70</td><td>24.59 ↓0.23</td><td>29.15 ↓0.76</td><td>2.33 ↓36.70</td><td>39.41 ↓0.82</td><td>6.46</td><td>84.31 ↓0.22</td><td>25.57 ↓61.07</td><td>95.77 ↓0.34</td></tr><tr><td>+ 1 Padding</td><td>14.33</td><td>10.91 ↑0.76</td><td>0.11 ↓14.70</td><td>24.42 ↓0.40</td><td>27.69 ↓2.22</td><td>0.24 ↓38.79</td><td>36.32 ↓3.91</td><td>10.16</td><td>83.28 ↓1.25</td><td>4.34 ↓82.30</td><td>92.83 ↓3.28</td></tr><tr><td rowspan="7">Random</td><td>1.77</td><td>10.23 ↑0.08</td><td>14.72 ↓0.09</td><td>24.88 ↑0.06</td><td>29.74 ↓0.17</td><td>39.43 ↑0.40</td><td>40.28 ↑0.05</td><td>2.24</td><td>84.46 ↓0.07</td><td>86.51 ↓0.13</td><td>95.97 ↓0.14</td></tr><tr><td>4.94</td><td>9.75 ↓0.40</td><td>16.85 ↑2.04</td><td>24.63 ↓0.19</td><td>29.38 ↓0.53</td><td>39.18 ↑0.15</td><td>39.92 ↓0.31</td><td>4.23</td><td>84.35 ↓0.18</td><td>86.47 ↓0.17</td><td>95.38 ↓0.73</td></tr><tr><td>9.52</td><td>9.38 ↓0.77</td><td>13.26 ↓1.55</td><td>24.51 ↓0.31</td><td>29.15 ↓0.76</td><td>38.96 ↓0.07</td><td>39.27 ↓0.96</td><td>6.71</td><td>83.29 ↓1.24</td><td>85.71 ↓0.93</td><td>94.51 ↓1.60</td></tr><tr><td>2.59</td><td>10.12 ↓0.03</td><td>15.11 ↑0.30</td><td>24.68 ↓0.14</td><td>29.69 ↓0.22</td><td>39.52 ↑0.49</td><td>39.96 ↓0.27</td><td>3.26</td><td>84.22 ↓0.31</td><td>86.34↓0.30</td><td>95.85 ↓0.26</td></tr><tr><td>7.64</td><td>10.37 ↑0.22</td><td>14.38 ↓0.43</td><td>24.77 ↓0.05</td><td>29.26 ↓0.65</td><td>38.94↓0.09</td><td>39.54 ↓0.69</td><td>6.46</td><td>83.96 ↓0.57</td><td>84.68 ↓1.96</td><td>95.03 ↓1.08</td></tr><tr><td>14.33</td><td>9.51 ↓0.64</td><td>14.89 ↑0.08</td><td>24.47 ↓0.35</td><td>28.88 ↓1.03</td><td>38.71 ↓0.32</td><td>39.06 ↓1.17</td><td>10.16</td><td>83.64 ↓0.89</td><td>83.88 ↓2.76</td><td>95.12 ↓0.99</td></tr></table>

Table 1: Performance under token ablation. Baseline removes no token. Target and reference ablations replace visual tokens selected from the corresponding object masks with a global average visual embedding; padding variants shrink or dilate the object-token mask by one token. Token (%) is the average percentage of image tokens replaced. Loc-T (Target) and Loc-R (Reference) report localization accuracy averaged over IoU@0.5/0.7/0.9, and Rel. reports spatial relation accuracy.

![](images/08b648824f1a4b0296fc2eba38d741d605ba2358726e1d11f06634777640a3ee.jpg)

![](images/26e2e56545946222b3804eeedf05116534a2a6b08ea7a130441989d3da318a34.jpg)

![](images/f445be6fd5361b39d8b0a1508ac3dfc77d725c0426c834d8706b52a434d5f62e.jpg)  
Figure 3: Layer-wise positional decoding and logit lens. (a) Joint two-dimensional position-probe accuracy across the LLM decoder layers of LLaVA-1.5-7B, LLaVA-1.5-13B, and Qwen2.5-VL-7B. (b) Axis-specific position-probe accuracy. (c) Spatialrelation answer accuracy obtained at each layer by applying a training-free logit lens to the final prompt-token representation. Error bars and shaded regions denote the standard error across samples.

(ii) Random Tokens: As a control, we replace an equal number of randomly selected visual tokens for each object-aligned intervention. Results are averaged over three seeds.

Result. Table 1 shows that object-aligned tokens selectively support localization. Target-token ablation reduces Loc-T by 5.37, 27.90, and 64.93 points for LLaVA-7B, LLaVA-13B, and Qwen-7B, respectively, while referencetoken ablation reduces Loc-R by 7.70, 36.70, and 61.07 points. Localization of the unablated object remains stable, and matched random removal causes substantially smaller changes. Yet relation accuracy drops by at most 1.32 points under the original masks despite large localization losses. Mask dilation increasingly impairs relation prediction, whereas erosion produces weaker localization efects. Thus, precise localization depends on object-internal evidence, while relation prediction remains robust until the intervention reaches the surrounding context.

## Position Decoding

Token ablation shows relation prediction can survive the loss of evidence required for localization. We next ask whether this dissociation is reflected in when positional and relation information becomes identifiable across decoder depth.

Method. To evaluate positional identifiability, we train separate linear classifiers at each decoder layer to predict the two-dimensional grid position of every image token. The horizontal (x) and vertical (y) coordinates are predicted independently, and a joint prediction is correct only when both coordinates are recovered. The classifiers are trained for 10 epochs on 50,000 ImageNet images and evaluated on the What’s Up images. To trace when relation-specific information becomes accessible to the model’s output, we further employ the logit lens, a training-free method for interpreting intermediate activations (nostalgebraist 2020). At each layer, the final prompt-token representation is passed through the model’s final normalization and language-model head. We then compare the logits assigned to the answer options.

Results. Figure 3 reveals that positional information increases rapidly, peaking around layer 14 for LLaVA-7B, layer 12 for LLaVA-13B, and layer 10 for Qwen-7B before gradually declining. The axis-specific probes show consistent trends, confirming that both horizontal and vertical coordinates remain recoverable across substantial portions of the decoder. In contrast, the logit lens reveals markedly diferent answer-formation dynamics. Qwen2.5-VL remains near chance in early layers but rises sharply after layer 19, reaching approximately 97% accuracy at the final layer. LLaVA-7B remains near chance throughout, whereas LLaVA-13B improves gradually to approximately 40%. These results show that positional decodability alone does not guarantee correct spatial reasoning. The temporal separation between position recovery and answer emergence is consistent with a staged process in which spatial layouts are represented before being transformed into relation-specific decisions.

![](images/78a9940e3a1c9f17525453239bf9a0d3330d861f612cb03439814038c879ce09.jpg)

![](images/d4ba1dcacc31f2c420647a6e7a8c8f7afd441be63bd34acda44c498232c0cab5.jpg)  
- Localization: Random / P0  Localization: Random / P1 0 Localization: Object / P0 -0 Localization: Object / P1Spatial: Random / P0 Spatial: Random / P1Spatial: Object / P0Spatial: Object /P1Baseline LocalizationBaseline Spatial  
Figure 4: Performance after attention knockout. We block attention to object-aligned visual tokens within grouped decoder layers and across all layers. Lines and bars report localization and spatial-relation accuracy, respectively, under the Padding=0 (P0) and Padding=1 (P1) settings. Object-token knockout is compared with a size-matched random-token control. Solid and dash-dotted horizontal lines indicate the localization and spatial-relation accuracy of the unmodified model.

## Attention Knockout

Layer-wise probes reveal when spatial information is accessible, but not whether later computations actually use it. We therefore investigate where the model extracts and processes object-grounded visual information.

Method. We apply attention knockout (Geva et al. 2023; Neo et al. 2025) to block attention from all post-image tokens to object-aligned visual tokens. This intervention prevents object information from being extracted within the selected layers. We group consecutive layers into groups of four and block all attention heads within each group. We also perform knockout across all layers as a global intervention. We measure the resulting changes in localization and spatial-relation accuracy using our curated What’s UP subset.

Results. The results are shown in Figure 4. For LLaVA-1.5- 13B, object-token knockout produces the largest localization decline in layers 16–19, while spatial-relation accuracy remains comparatively stable under most local interventions. Blocking attention across all layers reduces localization to nearly zero and lowers relation accuracy toward chance level. For Qwen2.5-VL-7B, localization is most sensitive to interventions in layers 16–23. Spatial-relation accuracy is also afected by several early and middle layer groups. Global object-token knockout again produces the strongest degradation in both tasks and is more disruptive than the matched random control. The similar trends under Padding = 0/1 indicate that these efects are not determined by a single choice of object-token boundaries. These results reveal only partial overlap between the layers supporting localization and spatial-relation prediction. Precise localization depends most strongly on specific intermediate layers, whereas relation prediction draws on information distributed across a broader processing range. Nevertheless, the global intervention shows that spatial reasoning still depends on information extracted from object-aligned tokens. This distinction motivates our subsequent head-level analysis of the pathways that transform object grounding into spatial decisions.

## Causal Mediation Analysis

Attention-Knockout presents evidence that the mechanisms underlying localization and spatial reasoning are concentrated within a narrow region of the language model’s processing pipeline. We therefore apply causal mediation analysis (CMA) (Yang et al. 2025; Wang et al. 2023; Meng et al. 2022) using activation patching to identify which attention heads causally contribute to solving the visual task.

Method. For each model, we curate up to 50 source-base pairs. The source image contains the target object, while the base image removes it using LaMa inpainting (Figure 2). For each attention head, we patch its source prompt-token activations into the corresponding base run. The counterfactual output measures whether the patched head restores information lost through object removal (Figure 5).

All runs are evaluated under teacher forcing using mean token-level negative log-likelihood (NLL). We score only numerical coordinate tokens for localization and the groundtruth answer token for a binary spatial-relation query. Let $L _ { \mathrm { s r c } } , \ L _ { \mathrm { b a s e } } ,$ and $L _ { \mathrm { p a t c h e d } }$ denote the corresponding NLL values. We define the Mediation Fraction as

$$
\mathrm { M F } = \frac { L _ { \mathrm { b a s e } } - L _ { \mathrm { p a t c h e d } } } { L _ { \mathrm { b a s e } } - L _ { \mathrm { s r c } } } .\tag{7}
$$

MF measures the fraction of the source-base gap restored by the patched head. $\mathrm { M F } \approx 1$ indicates that the patched head recovers most of the source-base performance gap; MF ≈ 0 indicates little contribution, while MF < 0 indicates misleading or interfering efects. To limit afirmative bias in the binary relation task, we retain only pairs where the source prediction is correct and object removal degrades the answer. This paired design ensures that NLL changes reflect the recovery of object-dependent relational evidence.

![](images/4054b34abcb8a86d3450086bd207a04d5d012439659be03826f0bc0c2ba8a561.jpg)  
Figure 5: Causal mediation via activation patching. We compare three model runs: (a) the source clean run, where the object is present and the model produces the correct answer; (b) the base counterfactual run, where the target object is removed and the model fails; and (c) the patched counterfactual run, where we transfer hidden activations from a selected attention head in the source run into the base run. Improvements in the patched prediction indicate that the transferred head carries task-relevant information. This procedure is repeated head-wise to quantify each head’s causal contribution.

![](images/716fccf8c1b17fe8bd17c527976927b09240a01c6b3c74c03193940f6b76a270.jpg)  
Head

![](images/80ed0d9876fd0bc48f189a2fde2c7d88e098c1fee51dfe1a10598eb38533b9d6.jpg)  
Head

![](images/8af607c4a467e202dd0473a8ac4d62e83752425d5c328135829c461a70020ad6.jpg)  
Head

![](images/2430bdf83d5cc9f46f622be4a0a78f3b2f5c1555d5ffc50be6f165eb23823d4c.jpg)  
Head

![](images/b2caa73c5cdbcd4276d5d0a507ac52ba6de2a9d867483ebd53fcfc71ee04b78d.jpg)  
Head

![](images/5af47300d9d853d2ea41a97254c40048cf843b625492e98870797d10f1a1f58e.jpg)  
Figure 6: Head-level causal mediation analysis. Mediation Fraction (MF) scores for every attention head across the decoder layers, shown separately for the localization and spatial reasoning tasks.

Results. Figure 6 reports head-level MF scores. Consistent with attention knockout, high-MF heads concentrate in the early–mid layers of LLaVA-1.5 and the mid-late layers of Qwen2.5-VL. In LLaVA-1.5-13B, both tasks show their strongest efects around layers 11-16, but rely on substantially diferent heads within this shared range. In Qwen2.5-VL-7B, localization efects are weaker and more dispersed, whereas relation mediation concentrates around layers 16-20. Most heads outside these regions have MF scores near zero.

Mediation is sparse and largely task-specific. Among the top 10 heads, localization and relation prediction share only 3 heads in LLaVA-1.5-13B and 2 heads each in LLaVA-1.5- 7B and Qwen2.5-VL-7B. The corresponding overlaps among the top 50 heads are only 3, 7, and 3. Thus, apparent overlap at the layer level masks substantial specialization at the head level. Negative MF values across all models further suggest that some isolated source activations are incompatible with the base-image computation and interfere with the correct prediction. Overall, localization and relation prediction reuse limited intermediate computation, but their dominant causal efects are mediated by largely distinct attention heads.

## Head Ablation Analysis

CMA reveals a small number of attention heads with high MF scores per task. To determine whether these heads are necessary rather than merely correlated with performance, we progressively remove them in a cumulative ablation study.

Method. We rank attention heads separately by their mean MF for localization and spatial relations. As controls, lowimportance heads have the smallest mean absolute MF, while negative heads have the lowest signed MF across both tasks. We progressively ablate a proportion of the total number of heads by setting their outputs to zero during the forward pass and evaluate localization and binary relation accuracy. Each pruning curve is summarized by normalized AUC (nAUC), where lower values indicate greater sensitivity.

Results. Figure 7 shows that removing task-important heads is substantially more damaging than removing lowimportance heads. Localization declines fastest under the localization ranking, whereas relation accuracy is most sensitive to relation-important heads, confirming that CMA identifies functionally necessary components. Relation accuracy also drops strongly when localization-important heads are removed, indicating that both tasks reuse objectgrounded intermediate representations despite limited head overlap. Ablating negative-MF heads produces gradual, model-dependent declines rather than consistent improvements. Negative mediation therefore reflects incompatibility

![](images/82de94416cb4f997e03c43b16be279b131204278451f0307668015a7f5f700f2.jpg)  
Loc. ImportantRel. Important  UnimportantLLaVA-7B LLaVA-13B Qwen-7B

Figure 7: Performance under cumulative ablation of attention heads. (a) Localization and (b) spatial-relation accuracy as increasing fractions of localization-important, relation-important, unimportant, and negative-MF heads are ablated.

during isolated activation patching, not necessarily a harmful role during normal inference. Together with attention knockout, these results support a staged mechanism in which shared object-grounded representations are transformed by largely specialized heads into precise locations or spatial decisions.

## Related Work

Spatial Reasoning in VLMs. Spatial reasoning has been evaluated from complementary perspectives. VSR examines a broad range of relations in natural images, while BLINK isolates perception-intensive abilities such as relative depth, visual correspondence, and multi-view reasoning (Fu et al. 2024). TopViewRS further separates object recognition, localization, and relational inference in controlled top-view environments (Li et al. 2024). Beyond evaluation, SpatialVLM introduces large-scale 3D supervision for qualitative and quantitative spatial reasoning (Chen et al. 2024), whereas SpatialRGPT incorporates region representations and depth cues to support grounded relational judgments (Cheng et al. 2024). These studies establish that spatial reasoning remains challenging and can benefit from explicit spatial supervision. However, neither benchmark performance nor training gains determine whether a model must recover precise object locations before predicting a relation.

Explicit Grounding in Multimodal Models. A parallel line of work equips multimodal models with explicit object grounding. Kosmos-2 (Peng et al. 2023) and Shikra (Chen et al. 2023b) generate coordinates directly within the language sequence, while Ferret combines discrete coordinates with continuous region features to support multi-granularity referring and grounding (You et al. 2024). GPT4RoI (Zhang et al. 2025a) and PVIT (Chen et al. 2023a) inject region-level features into visual instructions. More recent systems extend grounding from bounding boxes to pixel-level masks through reasoning segmentation, grounded conversation, and multiobject segmentation (Lai et al. 2024; Rasheed et al. 2024; Ren et al. 2024; Zhang et al. 2024; Li et al. 2025c; Schaumlöfel, Vilas, and Roig 2026). Collectively, these methods show that precise localization can be learned, elicited, or decoded from VLM representations, but not that such precision is causally required for spatial reasoning, since grounding supervision and output interfaces are introduced jointly.

Mechanistic Analysis of Multimodal Computation. Recent interpretability studies have begun to trace how visual evidence is transformed inside VLMs. Attention-lens analysis identifies distinct stages of visual enrichment and semantic refinement in intermediate layers (Jiang et al. 2025), while layer-wise information decomposition characterizes the transition from visual evidence to language-dominated predictions (Li et al. 2025c; Wu, Zhang, and Zhou 2026). Sparse feature analysis connects latent visual concepts to their downstream use in generated answers (Lim et al. 2026), and head-level interventions reveal that cross-modal retrieval can be concentrated in a small fraction ofattention heads (Sun et al. 2026). These findings suggest that visual computation is both depth-dependent and functionally sparse. Most existing analyses, however, only explain a single output behavior such as recognition, retrieval, or hallucination.

## Conclusion

Do VLMs really need object localization for spatial reasoning? Our results provide a qualified answer: VLMs require object grounding, but not precise localization. Spatial relations can survive severe localization failures, provided that coarse object-centered layout remains available. Thus, grounding is necessary evidence, but knowing exact object boundaries is not equivalent to knowing how objects relate. Whether this mechanism generalizes to crowded scenes, 3D relations, and temporal reasoning remains open.

D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025b. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Bi, J.; Guo, J.; Tang, Y.; Wen, L. B.; Liu, Z.; Wang, B.; and Xu, C. 2025. Unveiling visual perception in language models: An attention head analysis approach. In Proceedings of the Computer Vision and Pattern Recognition Conference, 4135–4144.

Bousselham, W.; Petersen, F.; Ferrari, V.; and Kuehne, H. 2024. Grounding everything: Emerging localization properties in vision-language transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3828–3837.

Chen, B.; Xu, Z.; Kirmani, S.; Ichter, B.; Sadigh, D.; Guibas, L.; and Xia, F. 2024. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14455–14465.

Chen, C.; Qin, R.; Luo, F.; Mi, X.; Li, P.; Sun, M.; and Liu, Y. 2023a. Position-enhanced visual instruction tuning for multimodal large language models. arXiv preprint arXiv:2308.13437.

Chen, K.; Zhang, Z.; Zeng, W.; Zhang, R.; Zhu, F.; and Zhao, R. 2023b. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195.

Chen, S.; Zhu, T.; Zhou, R.; Zhang, J.; Gao, S.; Niebles, J. C.; Geva, M.; He, J.; Wu, J.; and Li, M. 2025. Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas. arXiv preprint arXiv:2503.01773.

Cheng, A.-C.; Yin, H.; Fu, Y.; Guo, Q.; Yang, R.; Kautz, J.; Wang, X.; and Liu, S. 2024. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37: 135062–135093.

Chu, X.; Su, J.; Zhang, B.; and Shen, C. 2024. Visionllama: A unified llama backbone for vision tasks. In European Conference on Computer Vision, 1–18. Springer.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang,

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 248–255. Ieee.

Dorkenwald, M.; Barazani, N.; Snoek, C. G.; and Asano, Y. M. 2024. Pin: Positional insert unlocks object localisation abilities in vlms. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 13548–13558.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Fu, X.; Hu, Y.; Li, B.; Feng, Y.; Wang, H.; Lin, X.; Roth, D.; Smith, N. A.; Ma, W.-C.; and Krishna, R. 2024. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, 148–166. Springer.

Geva, M.; Bastings, J.; Filippova, K.; and Globerson, A. 2023. Dissecting Recall of Factual Associations in Auto-Regressive Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 12216–12235.

Huang, M.; Jiang, B.; Zheng, D.; Hu, H.; Han, K.; and Chen, X. 2025a. Positional Preservation Embedding for Multimodal Large Language Models. arXiv preprint arXiv:2510.22936.

Huang, R.; Ma, X.; Kong, R.; Yuan, Z.; and Zhang, P. 2025b. OMEGA: Optimized Multimodal Position Encoding Index Derivation with Global Adaptive Scaling for Vision-Language Models. arXiv preprint arXiv:2511.00821.

Jia, M.; Qi, Z.; Zhang, S.; Zhang, W.; Yu, X.; He, J.; Wang, H.; and Yi, L. 2025. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135.

Jiang, Z.; Chen, J.; Zhu, B.; Luo, T.; Shen, Y.; and Yang, X. 2025. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 25004–25014.

Kamath, A.; Hessel, J.; and Chang, K.-W. 2023. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 9161–9175.

Kang, S.; Kim, J.; Kim, J.; and Hwang, S. J. 2025. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, 9339–9350.

Lai, X.; Tian, Z.; Chen, Y.; Li, Y.; Yuan, Y.; Liu, S.; and Jia, J. 2024. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9579–9589.

Li, C.; Zhang, C.; Zhou, H.; Collier, N.; Korhonen, A.; and Vulić, I. 2024. Topviewrs: Vision-language models as topview spatial reasoners. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 1786–1807.

Li, H.; Li, D.; Wang, Z.; Yan, Y.; Wu, H.; Zhang, W.; Shen, Y.; Lu, W.; Xiao, J.; and Zhuang, Y. 2025a. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531.

Li, R.; Li, L.; Ren, S.; Tian, H.; Gu, S.; Li, S.; Yue, Z.; Wang, Y.; Ma, W.; Yang, Z.; et al. 2026. Groundingme: Exposing the visual grounding gap in mllms through multi-dimensional evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2412–2422.

Li, Y.; Wang, H.; Ding, X.; Wang, H.; and Li, X. 2025b. Token activation map to visually explain multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 48–58.

Li, Y.; Zhao, C.; Zang, Z.; Yuan, C.; and Wang, X. 2025c. Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models. arXiv preprint arXiv:2509.19191.

Lim, H.; Choi, J.; Kim, T.; Heo, B.; Choo, J.; and Han, D. 2026. VisualScratchpad: Inference-time Visual Concepts Analysis in Vision Language Models. arXiv preprint arXiv:2603.07335.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Advances in neural information processing systems, 36: 34892–34916.

Ma, C.; Jiang, Y.; Wu, J.; Yuan, Z.; and Qi, X. 2024. Groma: Localized visual tokenization for grounding multimodal large language models. In European Conference on Computer Vision, 417–435. Springer.

Meng, K.; Bau, D.; Andonian, A.; and Belinkov, Y. 2022. Locating and editing factual associations in GPT. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713871088.

Neo, C.; Ong, L.; Torr, P.; et al. 2025. Towards Interpreting Visual Information Processing in Vision-Language Models. In The Thirteenth International Conference on Learning Representations.

nostalgebraist. 2020. Interpreting GPT: The Logit Lens.

Peng, Z.; Wang, W.; Dong, L.; Hao, Y.; Huang, S.; Ma, S.; and Wei, F. 2023. Kosmos-2: Grounding Multimodal Large Language Models to the World. arXiv:2306.14824.

Qiu, H.; Gao, P.; Lu, L.; Zhang, X.; Shao, L.; and Lu, S. 2025. Spatial Preference Rewarding for MLLMs Spatial Understanding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 720–730.

Ranasinghe, K.; Shukla, S. N.; Poursaeed, O.; Ryoo, M. S.; and Lin, T.-Y. 2024. Learning to localize objects improves spatial reasoning in visual-llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12977–12987.

Rasheed, H.; Maaz, M.; Shaji, S.; Shaker, A.; Khan, S.; Cholakkal, H.; Anwer, R. M.; Xing, E.; Yang, M.-H.; and Khan, F. S. 2024. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13009–13018.

Ren, X.; Wang, Z.; Hou, L.; Tang, P.; Wang, G.; and Ma, C. 2026. Grounding everything in tokens for multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 41171–41181.

Ren, Z.; Huang, Z.; Wei, Y.; Zhao, Y.; Fu, D.; Feng, J.; and Jin, X. 2024. Pixellm: Pixel reasoning with large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26374–26383.

Schaumlöfel, T.; Vilas, M. G.; and Roig, G. 2026. Mechanisms of Object Localization in Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 31356–31365.

Su, J.; Ahmed, M.; Lu, Y.; Pan, S.; Bo, W.; and Liu, Y. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568: 127063.

Sun, R.; Qiu, Q.; Li, J.; Tang, Z.; Lou, Y.; and Zhang, M. 2026. Mechanistic Insights into Functional Sparsity in Multimodal LLMs via CoRe Heads. arXiv preprint arXiv:2606.05843.

Suvorov, R.; Logacheva, E.; Mashikhin, A.; Remizova, A.; Ashukha, A.; Silvestrov, A.; Kong, N.; Goka, H.; Park, K.; and Lempitsky, V. 2022. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2149–2159.

Tong, S.; Liu, Z.; Zhai, Y.; Ma, Y.; LeCun, Y.; and Xie, S. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9568– 9578.

Wang, K. R.; Variengien, A.; Conmy, A.; Shlegeris, B.; and Steinhardt, J. 2023. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. In The Eleventh International Conference on Learning Representations.

Wu, H.; Zhang, Y.; and Zhou, X. 2026. How Vision Becomes Language: A Layer-wise Information-Theoretic Analysis of Multimodal Reasoning. arXiv preprint arXiv:2602.15580.

Yang, Y.; Campbell, D.; Huang, K.; Wang, M.; Cohen, J.; and Webb, T. 2025. Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models. Proceedings ofMachine Learning Research, 267: 70515–70549.

You, H.; Zhang, H.; Gan, Z.; Du, X.; Zhang, B.; Wang, Z.; Cao, L.; Chang, S.-F.; and Yang, Y. 2024. Ferret: Refer and ground anything anywhere at any granularity. In International Conference on Learning Representations, volume 2024, 57153–57180.

Zhang, S.; Sun, P.; Chen, S.; Xiao, M.; Shao, W.; Zhang, W.; Liu, Y.; Chen, K.; and Luo, P. 2025a. GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest. arXiv:2307.03601.

Zhang, Y.; Ma, Z.; Gao, X.; Shakiah, S.; Gao, Q.; and Chai, J. 2024. Groundhog: Grounding large language models to holistic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 14227– 14238.

Zhang, Z.; Yadav, S.; Han, F.; and Shutova, E. 2025b. Crossmodal information flow in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 19781–19791.