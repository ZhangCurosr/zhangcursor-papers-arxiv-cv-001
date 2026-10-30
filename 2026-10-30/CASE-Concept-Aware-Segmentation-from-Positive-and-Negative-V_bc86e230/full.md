# CASE: Concept-Aware Segmentation from Positive and Negative Visual Exemplars

Yi-Jen Tsai<sup>1</sup>, Yen-Yu Lin<sup>1</sup>, and Chien-Yao Wang<sup>2⋆</sup>

<sup>1</sup> National Yang Ming Chiao Tung University, Taiwan tsai.cs14@nycu.edu.tw, lin@cs.nycu.edu.tw 2 Institute of Information Science, Academia Sinica, Taiwan kinyiu@iis.sinica.edu.tw

Abstract. Few-shot semantic segmentation typically identifies target objects based on positive support examples of a single semantic class. In practice, however, positive cues alone are often underspecified: for instance, “red cars” may be confused with cars of all colors, “vehicles” may include cars that should be excluded, and “material-defined objects” may span multiple categories. Fulfilling precise requests requires positive exemplars to provide inclusion evidence and negative exemplars to act as intent-defining constraints for exclusion and disambiguation. We formulate this setting as Specified Visual Prompt Segmentation (SVPS), where the target concept is specified by both positive and negative visual prompts. In this work, visual exemplars refer to supportimage regions specified by bounding-box prompts. To study SVPS, we introduce CASE (Concept-Aware Segmentation from Positive and Negative Visual Exemplars), a benchmark built on COCO and PACO. CASE encompasses six category- and attribute-level task families: exclusion, isolation, union, attribute exclusion, attribute isolation, and attribute union. We evaluate SAM 3 under this setting and find that, despite its powerful positive-only few-shot segmentation capability, frozen SAM 3 does not consistently treat negative visual prompts as exclusion evidence. To address this, we present Concept Prompt Adaptation (CPA), a conservative LoRA-based adaptation baseline that improves positive-negative prompt interpretation on CASE while remaining competitive under positive-only few-shot segmentation. Code is available at https://github.com/WongKinYiu/FSS-CASE.

Keywords: Specified visual prompt segmentation · Few-shot segmentation · Visual prompting · Negative exemplars · SAM 3

## 1 Introduction

Semantic segmentation [3, 15, 38] is a fundamental task in computer vision that aims to assign a semantic label to every pixel in an image. Although deep models have achieved remarkable performance on standard benchmarks, they rely on large-scale datasets with dense, pixel-level annotations for training. Beyond expensive annotation costs, most models are trained on a fixed set of categories, resulting in limited generalization to unseen objects in open-world scenarios.

![](images/595e68f1084cab1ba4ff8942b3b58a4e094f6908dab0a95ba70d5121a6642039.jpg)  
Fig. 1: Motivation and overview of CASE. Conventional negative prompts are commonly used as local refinement signals for an already specified target. In contrast, Specified Visual Prompt Segmentation (SVPS) treats positive and negative visual exemplars as intent-defining cues. The same query image can therefore induce diferent target under diferent prompts. CASE evaluates this setting through six task families, requiring a unified model to infer target intent from positive-negative exemplar.

Few-shot semantic segmentation (FSS) [26, 28] addresses this limitation by segmenting novel objects in a query image using only a small number of annotated support examples. Existing FSS methods commonly adopt episodic training to learn transferable matching mechanisms between support and query images, such as prototype matching [26, 28], pixel-level correspondence [17, 20], or cross-image attention [31, 35]. Despite their success on standard FSS benchmarks, these methods largely formulate the problem as category-level matching: Namely, given one or more positive support examples, the model is expected to segment all objects of the same semantic class in the query image.

However, this formulation is overly restrictive for practical visual prompting. In real-world scenarios, user specifications frequently require both inclusion and exclusion cues, rather than purely positive category matching. For instance, positive examples may overgeneralize from “red cars” to all cars, from “buses and trains” to all vehicles (thereby incorrectly including cars), or from “wooden chairs” to all chairs rather than wooden objects. These complex targets cannot be precisely encapsulated by positive examples alone: While positive examples define what to include, they do not explicitly specify what should be excluded. Consequently, a model needs to jointly interpret positive and negative visual prompts to infer the target mask from their relationship.

We define this task as Specified Visual Prompt Segmentation (SVPS), where the target concept is specified by both positive and negative visual exemplars. In this work, visual exemplars are support-image regions represented as bounding-box prompts. Positive exemplars provide visual evidence for inclusion, whereas negative exemplars act as intent-defining constraints that specify confusing categories, attributes, or instances to exclude. Unlike conventional FSS,

SVPS evaluates whether a model can infer a specified target concept from the interplay between positive and negative exemplars. As illustrated in Fig. 1, this changes the role of negative prompts from local mask refinement to conceptlevel intent specification: diferent positive-negative exemplar configurations can induce diferent target masks for the same query image.

To evaluate this task, we introduce CASE (Concept-Aware Segmentation from Positive and Negative Visual Exemplars), a benchmark constructed from COCO [13] and PACO [22] annotations. The CASE benchmark encompasses six task families spanning category-level and attribute-level concept specification: exclusion, isolation, and union. These tasks are designed to assess whether a segmentation model can leverage negative visual prompts as concept-level exclusion evidence, rather than simply appending them as additional prompts without modeling their relation to the positive visual evidence.

We further study Segment Anything Model 3 (SAM 3) [1] under the SVPS setting. SAM 3 is a strong vision foundation model for promptable concept segmentation, supporting segmentation from text descriptions, visual exemplars, or their combination. A recent shared-canvas formulation [27] has shown that support and query images can be placed in a single spatial canvas, enabling a fully frozen SAM 3 to perform few-shot segmentation without additional training or architectural modification. This makes SAM 3 a suitable testbed for studying whether promptable segmentation models can handle positive-negative visual prompts beyond positive-only matching. Unlike general promptable concept segmentation in SAM 3 [1], where diferent prompt types serve as alternative ways to refer to a concept, SVPS evaluates whether positive and negative exemplars can jointly specify a constrained target concept.

However, the same line of evidence also reveals a key limitation: directly adding negative visual prompts to the few-shot segmentation pipeline does not necessarily improve performance and may even suppress target objects [27]. This behavior contradicts the intended role of negative prompts as exclusion cues. It suggests that, although SAM 3 can efectively exploit positive visual prompts, it does not inherently parse negative prompts as precise exclusion constraints. In few-shot settings, negative prompts may instead compete with positive prompts and weaken or suppress predicted target masks.

To address this issue, we propose Concept Prompt Adaptation (CPA), a lightweight, LoRA-based adaptation strategy tailored for SAM 3 under positivenegative visual prompting. We use the term concept to highlight that CPA adapts the model to target concepts specified by positive-negative visual evidence. CPA updates only low-rank adaptation parameters while keeping the pretrained model frozen. Furthermore, we introduce positive-only preservation regularization that keeps the adapted model close to frozen SAM 3 when negative prompts are absent. This provides a conservative yet efective adaptation baseline that balances SVPS improvement on CASE while mitigating standard FSS degradation.

The primary contributions of this work are summarized as follows:

– We introduce SVPS, a new segmentation paradigm where positive and negative visual exemplars jointly specify a target concept.

– We present CASE, a benchmark for SVPS built upon COCO and PACO, covering six category- and attribute-level concept specification tasks.

– We propose CPA, a lightweight LoRA-based adaptation baseline equipped with positive-only preservation regularization, and evaluate the adaptationpreservation trade-of on both SVPS and positive-only FSS benchmarks.

## 2 Related Work

## 2.1 Few-Shot Semantic Segmentation

Few-shot semantic segmentation (FSS) [26,28] aims to segment novel categories in a query image given only a small number of annotated support examples. Early approaches commonly adopt episodic training to learn matching functions. A prominent line of work represents support evidence with prototypes or prototypeenhanced class summaries [7, 12, 26, 28]. Although prototype-based methods are eficient, compressing multiple support masks into a global vector discards finegrained spatial details crucial for precise localization.

To preserve richer support-query correspondences, subsequent methods explore pixel- or region-level matching through hypercorrelation [17], dense matching [20], or cross-attention [31, 35]. Despite efectiveness on FSS, these methods work under the restrictive assumption that the support set provides solely positive evidence for a single semantic class. Hence, the task is reduced to categorylevel matching, where the model merely transfers a positive support concept to the query image. In contrast, our work studies a more expressive setting where a target concept is jointly specified by positive and negative visual prompts.

## 2.2 Foundation Models for Promptable Segmentation

Vision foundation models such as CLIP [21] and DINO [2, 19] provide generalpurpose visual representations that have reshaped many recognition [37, 40, 41] and segmentation [16,23,30,39] tasks. Building on this trend, the Segment Anything Model (SAM) [11] demonstrates remarkable class-agnostic segmentation capabilities through spatial prompts such as points and boxes. Several works adapt SAM to FSS by generating visual reference prompts from support images [14, 25, 34]. While efective, these methods often introduce additional components, such as visual reference encoders [25], graph-based prompt refinement modules [34], or prompt-learning/adaptation modules [4, 9], to bridge the gap between support examples and query segmentation.

SAM 2 [24] extends promptable segmentation to videos through a memory mechanism for frame-to-frame mask propagation. Recent FSS methods based on SAM 2, such as FSSAM [32] and SANSA [5], adapt this memory mechanism by treating support images as memory references and query images as target frames. However, these memory-based formulations are primarily designed for temporal consistency and object tracking, making them suboptimal for semantic correspondences across temporally disjoint images.

SAM 3 [1] introduces Promptable Concept Segmentation (PCS), where an object concept can be specified by text prompts, positive or negative image exemplars, or their combinations. Native prompting operates on the image being segmented, whereas CASE requires cross-image transfer. A recent shared-canvas pipeline [27] further enables frozen SAM 3 to perform training-free FSS by placing support and query images into a single canvas, but directly incorporating negative prompts can degrade segmentation quality and suppress target objects. Rather than treating negative exemplars as interactive refinements to an existing concept prompt, SVPS studies whether positive and negative visual exemplars can jointly define complex target concepts through inclusion and exclusion roles.

## 2.3 Visual Prompting and Negative Prompts

Prompt design is crucial to foundation segmentation models. Existing works mainly improve positive guidance through visual reference prompts [25, 36], support-query feature matching [14, 34], learnable prompts or adaptation modules [4,9], and text- or image-guided prompting [6,16,29]. These mechanisms are used to localize a target object or transfer a positive support concept to a query image. Target concept ambiguity is commonly mitigated by strengthening positive evidence through refined localization [36], support-query matching [14, 34], or multimodal guidance [6, 16, 23, 29].

Negative prompts remain less explored in exemplar-based FSS. While negative points or boxes are commonly expected to suppress unwanted regions in user-interactive segmentation [11, 24], this assumption may not transfer directly to FSS, where negative exemplars define semantic exclusion constraints rather than merely correct local mask errors. Recent SAM 3-based FSS results [27] suggest that directly appending negative prompts reduces segmentation quality and suppresses target objects. This indicates that current promptable segmentation models fail to utilize negative visual prompts as exclusion evidence.

Beyond category-level segmentation, referring expression segmentation specifies targets through language descriptions involving attributes or context [6,10, 29, 33], while datasets such as PACO [22] evaluate attribute-level visual understanding. Although language can describe constraints, subtle visual distinctions are easier to specify through visual examples. Motivated by this gap, we define Specified Visual Prompt Segmentation (SVPS), where positive and negative visual prompts jointly specify a target concept rather than serving as auxiliary inputs. The proposed CASE (Concept-Aware Segmentation from Positive and Negative Visual Exemplars) benchmark systematically evaluates whether models can infer target concepts from positive-negative exemplar relationships, covering exclusion, isolation, union, and attribute-level disambiguation.

## 3 The CASE Benchmark

## 3.1 Benchmark Overview

We introduce CASE, a benchmark for SVPS. Traditional few-shot segmentation benchmarks evaluate whether a model can transfer a single positive support concept to a query image. In contrast, CASE requires models to infer target concepts by joint reasoning over both positive and negative visual exemplars. Positive exemplars define regions to include, whereas negative exemplars specify related categories, attributes, or instances that must be excluded.

Table 1: Task taxonomy of the CASE benchmark. Each task specifies a target concept through positive and negative visual exemplars.
<table><tr><td>Task</td><td>Target Concept</td><td>Positive Exemplars</td><td>Negative Exemplars</td><td>Challenge</td></tr><tr><td>SuperEx</td><td>Supercategory excluding one class, e.g., vehicles except cars.</td><td>Included classes from the supercategory, e.g., buses and trains.</td><td>The excluded class, e.g., cars.</td><td>Generalize to a broad group while enforcing explicit exclusion.</td></tr><tr><td>FineIso</td><td>One fine-grained class within a supercategory, e.g., dogs.</td><td>Target class instances, e.g., dogs.</td><td>Related distractor classes, e.g., cats or sheep.</td><td>Isolate a target class from visually close distractors.</td></tr><tr><td>Union</td><td>Classes sharing a common function, e.g., furniture for resting.</td><td>Functionally related classes, e.g., beds and sofas.</td><td>Related classes with different functions, e.g., chairs or tables.</td><td>Infer functional commonality across</td></tr><tr><td>AttrEx</td><td>A category excluding one attribute, e.g.,</td><td>Instances without the excluded attribute, e.g.,</td><td>Instances with the excluded attribute, e.g., red cars.</td><td>distinct objects. Apply visual negation by fine-grained</td></tr><tr><td>AttrIso</td><td>non-red cars. A category with one specified attribute, e.g.,</td><td>black or white cars. Target attribute instances, e.g., red cars.</td><td>Same category with other attributes, e.g., blue or white cars.</td><td>attribute differences. Separate attribute- specific targets from general object identity.</td></tr><tr><td>AttrUnion</td><td>red cars. An attribute shared across categories, e.g., wooden objects.</td><td>Diverse classes with the attribute, e.g., wooden</td><td>Similar classes without the attribute, e.g., plastic chairs or bowls.</td><td>Group objects by attribute while avoiding category or shape bias.</td></tr></table>

CASE is built on COCO [13] and PACO [22]. COCO provides category and supercategory annotations for category-level concept specification, while PACO ofers annotations for fine-grained attribute-level concept specification. Each episode provides support-image bounding boxes as visual exemplars and requires the model to predict the target mask in the query image.

## 3.2 Task Taxonomy

As summarized in Tab. 1, CASE encompasses six task families that evaluate different facets of positive-negative concept specification. The category-level tasks comprise supercategory exclusion (SuperEx), fine-grained isolation within a supercategory (FineIso), and functional union across related categories (Union). The attribute-level tasks include exclusion (AttrEx), isolation (AttrIso), and cross-category union (AttrUnion). Together, these tasks assess whether a model can move beyond naive positive matching and leverage negative exemplars as exclusion or boundary constraints for precise target concept specification.

## 3.3 Episode Construction

Following the standard FSS protocol, CASE is organized in an episodic format. Each episode consists of a support set with positive and negative bounding-box prompts, a query image, and a binary ground-truth query mask. The support prompts specify the target concept, and the query mask serves as the expected segmentation output in the query image for evaluation.

To construct an episode, we first instantiate a positive-negative concept definition using COCO or PACO annotations, such as a supercategory paired with an excluded class, a function-driven category group, or an attribute-defined object group. We then sample support images containing the required positive and negative exemplars. Query images are sampled independently to contain target objects and optionally include distractors. The target mask is generated from the underlying annotations according to the instantiated concept definition.

This design prevents the task from being solved by simply matching a positive support example. Because negative exemplars explicitly define what should be rejected or separated from the target, the model must infer the intended concept from the interplay between positive and negative prompts. Detailed per-task statistics and construction parameters are provided in the supplementary material, Sec. CASE Benchmark Construction.

## 4 Method

## 4.1 Task Formulation

Specified Visual Prompt Segmentation (SVPS) extends conventional FSS by defining the target concept through both positive and negative visual exemplars. Specifically, each episode contains a K-shot support set ${ \cal { S } } = \{ ( { \cal { I } } _ { i } ^ { s } , { \cal { B } } _ { i } ^ { + } , { \cal { B } } _ { i } ^ { - } ) \} _ { i = 1 } ^ { K }$ and a query image $I ^ { q }$ with the ground-truth mask $M ^ { q }$ . The positive bounding boxes $B _ { i } ^ { + }$ enclose visual evidence that must be included, whereas the negative boxes $B _ { i } ^ { - }$ indicate irrelevant distractors to be excluded. Given these inputs, the model $f _ { \theta }$ , parameterized by θ, predicts a binary mask $\hat { M } ^ { q }$ for the query image:

$$
\hat { M } ^ { q } = f _ { \theta } ( S , I ^ { q } ) .\tag{1}
$$

Conventional FSS can be viewed as a special case of SVPS where $B _ { i } ^ { - } = \varnothing$ for all support examples. Therefore, an SVPS model should leverage negative exemplars to suppress distractors when they are provided, while retaining competitive segmentation performance under the positive-only setting.

## 4.2 Model Overview

Fig. 2 illustrates our Concept Prompt Adaptation (CPA) pipeline. CPA adapts SAM 3 [1] to SVPS by augmenting the original SAM 3 detector with two trainable LoRA [8] modules while keeping the original SAM 3 weights frozen. We separate the pipeline into two parts: (i) the shared-canvas input construction from the support set S and query image $I ^ { q }$ , and (ii) the CPA-adapted SAM 3 forward path, where one LoRA module is attached to the transformer encoder and the other is attached to the transformer decoder.

Given S and I<sup>q</sup>, CPA first composes them into a shared canvas I<sup>c</sup> and transforms the positive and negative boxes into the canvas coordinate system [27]. The canvas image is processed by the SAM 3 image encoder, while the transformed prompt boxes are processed by the geometry encoder. The resulting features are passed through the SAM 3 transformer encoders and decoders, then through the prediction heads. The detector outputs mask logits $Z ^ { c }$ , boxes, and confidence scores $s ^ { c }$ on the composite canvas. We crop the query region from $Z ^ { c }$ to obtain query logits $Z ^ { q }$ , with corresponding confidence scores denoted as $s ^ { q }$ . The final query prediction $\hat { M } ^ { q }$ is derived from $Z ^ { q }$

![](images/a7d95e954028158bde6af46786a0617f3390ebab369f5c9210bfded84d404679.jpg)  
Fig. 2: Overview of Concept Prompt Adaptation (CPA). A K-shot support set $s$ and query image $I ^ { q }$ are composed into a shared canvas $I ^ { c }$ , with positive and negative support boxes used as visual prompts. CPA augments the original SAM 3 detector with two trainable LoRA modules while keeping the original SAM 3 weights frozen. The final query prediction is obtained from the cropped query outputs $Z ^ { q }$ and $s ^ { q }$ . The inset summarizes the two prompting conditions used during training; their formal objectives are defined in Sec. 4.5.

During training, CPA is optimized under two prompting conditions. In the positive-negative condition, the adapted model receives both $B ^ { + }$ and $B ^ { - }$ and is supervised by the CASE target mask $M ^ { q }$ . In the positive-only preservation condition, only $B ^ { + }$ is provided, and the adapted model is regularized to remain close to the original frozen SAM 3 prediction under the same positive-only input. The following subsections describe the shared-canvas formulation, LoRA placement in CPA, and the training objective, respectively.

## 4.3 Shared-Canvas Prompting

Following FSS-SAM3 [27], we adopt a shared-canvas formulation to process the support and query images within a unified spatial and attention space. The support set and query image are composed into a single image-like canvas:

$$
I ^ { c } = { \mathrm { C o m p o s e } } ( S , I ^ { q } ) ,\tag{2}
$$

and all support boxes $\{ B _ { i } ^ { + } , B _ { i } ^ { - } \} _ { i = 1 } ^ { K }$ are transformed into the coordinate system of $I ^ { c } .$ . The composite image and box prompts are then fed into SAM 3.

This formulation allows support and query regions to interact within the same attention space while reusing SAM 3’s native box-prompt interface, eliminating the need for an explicit module. The SAM 3 detector heads output composite mask logits $Z ^ { c }$ , boxes, and confidence scores $s ^ { c }$ . We crop the query region from the composite logits to obtain query logits $Z ^ { q } = \mathrm { C r o p } _ { q } ( Z ^ { c } )$ , and use the corresponding confidence scores $s ^ { q }$ to select or rank predictions. The binary query mask is obtained from the selected query logits as $\hat { M } ^ { q } = \mathbf { 1 } [ \sigma ( Z ^ { q } ) > \tau ]$ where $\sigma ( \cdot )$ denotes the sigmoid function and $\tau$ is the mask threshold. For CASE episodes, both positive and negative boxes are used. For positive-only evaluation and preservation regularization, only positive boxes are provided.

## 4.4 LoRA-Based Concept Prompt Adaptation

Frozen SAM 3 already provides strong positive-only few-shot segmentation behavior, but it does not interpret negative visual prompts as concept-level exclusion cues. To address this limitation, CPA adapts SAM 3 to positive-negative prompting while mitigating degradation of its original positive-only behavior. Let $f$ denote the original frozen SAM 3. CPA constructs an adapted model $f _ { \theta }$ by augmenting $f$ with two trainable LoRA modules: one attached to the transformer encoder and the other attached to the transformer decoder. All original SAM 3 parameters remain frozen, and θ contains only the parameters of these LoRA modules. For a pretrained attention projection matrix $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ each LoRA module replaces the efective projection by

$$
W ^ { \prime } = W + \varDelta W , \quad \varDelta W = \frac { \alpha } { r } B A ,\tag{3}
$$

where $A \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ and $B \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ are trainable matrices of rank $r \ll \operatorname* { m i n } ( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ and α is a scaling factor. The LoRA modules shown in Fig. 2 correspond to lowrank updates injected into the transformer encoder and decoder.

## 4.5 Training Objective

CPA is optimized with two objectives: an SVPS segmentation loss for positivenegative prompting and a preservation loss for maintaining positive-only behavior. Let $f _ { \theta }$ denote the adapted model and $f$ denote the original frozen SAM 3 model. Given positive and negative prompts, the adapted model predicts query logits $Z _ { \theta , \pm } ^ { q }$ . We supervise this prediction using the ground-truth query mask $M ^ { q } ;$

$$
\mathcal { L } _ { \mathrm { s v p s } } ^ { \pm } = \mathcal { L } _ { \mathrm { m a s k } } ( Z _ { \theta , \pm } ^ { q } , M ^ { q } ) , \mathrm { w h e r e } Z _ { \theta , \pm } ^ { q } = \mathrm { C r o p } _ { q } ( f _ { \theta } ( I ^ { c } , \mathcal { B } ^ { + } , \mathcal { B } ^ { - } ) )\tag{4}
$$

and $\mathcal { L } _ { \mathrm { m a s k } }$ combines sigmoid focal loss and Dice loss. The superscript ± indicates that the prediction is conditioned on both positive and negative prompts.

The preservation loss is computed under the positive-only prompting condition. For the same episode, the original frozen model $f$ produces reference query logits $Z _ { f , + } ^ { q }$ using only positive boxes. Under the same prompting condition, the adapted model $f _ { \theta }$ outputs query logits $Z _ { \theta , + } ^ { q }$ . We penalize the discrepancy between the two probability maps using the squared $L _ { 2 }$ norm:

$$
\mathcal { L } _ { \mathrm { p r e s } } ^ { + } = \left. \sigma ( Z _ { \theta , + } ^ { q } ) - \sigma ( Z _ { f , + } ^ { q } ) \right. _ { 2 } ^ { 2 } , \mathrm { ~ w h e r e ~ }\tag{5}
$$

$$
Z _ { \theta , + } ^ { q } = \mathrm { C r o p } _ { q } ( f _ { \theta } ( I ^ { c } , \mathcal { B } ^ { + } ) ) \mathrm { ~ a n d ~ } Z _ { f , + } ^ { q } = \mathrm { C r o p } _ { q } ( f ( I ^ { c } , \mathcal { B } ^ { + } ) ) ,\tag{6}
$$

and $\sigma ( \cdot )$ denotes the sigmoid function. The superscript + indicates that this term is computed under the positive-only prompting condition.

The final training objective is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s v p s } } ^ { \pm } + \lambda _ { \mathrm { p r e s } } \mathcal { L } _ { \mathrm { p r e s } } ^ { + } ,\tag{7}
$$

where $\lambda _ { \mathrm { p r e s } }$ controls the strength of positive-only preservation.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We evaluate CPA under two settings.

CASE. CASE evaluates specified visual prompt segmentation with positive and negative visual exemplars. It contains six task families constructed from COCO [13] and PACO [22]: SuperEx, FineIso, and Union for category-level concepts, and AttrEx, AttrIso, and AttrUnion for attribute-level concepts. Each episode provides positive and negative support boxes, a query image, and a binary target mask defined by a structured concept specification, such as category exclusion, fine-grained isolation, functional union, attribute isolation, attribute union, or attribute exclusion. The evaluation split is fixed at the episode level to ensure deterministic comparison across models.

Standard FSS benchmark. We evaluate the CASE-trained CPA checkpoints directly on all four COCO-20<sup>i</sup> folds [13, 18], without retraining or fold-specific tuning. COCO-20<sup>i</sup> is a standard few-shot segmentation benchmark derived from COCO, where the 80 object categories are divided into four folds with 20 categories per fold. We report results under both 1-shot and 5-shot settings. Since standard FSS uses only positive support annotations, each support mask is converted into its tight bounding box and used as a positive visual prompt.

Evaluation Metrics. We use mean Intersection-over-Union (mIoU) as the primary metric. For CASE, we report task-wise mIoU and the average mIoU across all tasks. We further aggregate results into category-level and attribute-level groups. For COCO-20<sup>i</sup>, we report fold-wise and mean mIoU under the standard positive-only 1-shot and 5-shot protocols.

Implementation Details. CPA is built on SAM 3 using the shared-canvas formulation of FSS-SAM3 [27], where support and query images are composed into a single 1008×1008 canvas. Positive and negative visual prompts are represented as support bounding boxes. We keep the pretrained SAM 3 weights frozen and insert LoRA modules into transformer encoder and decoder layers, resulting in 1.13M trainable parameters. CPA is trained with AdamW at a learning rate of $1 \times 1 0 ^ { - 4 }$ , with LoRA rank r = 16 and scaling factor α = 32. Standard FSS evaluation follows the positive-only prompting protocol, which is also used for the preservation regularization. Query masks are binarized at τ = 0.5. Additional details are provided in the supplementary material, Sec. Implementation Details.

Table 2: CASE benchmark results: task-wise and average mIoU across six tasks.
<table><tr><td>Method</td><td>Prompt</td><td>SuperEx</td><td>FineIso</td><td>Union</td><td>AttrEx</td><td>AttrIso</td><td>AttrUnion</td><td>Mean</td></tr><tr><td colspan="9">1-shot</td></tr><tr><td>GF-SAM [34]</td><td>Pos Only</td><td>24.87</td><td>35.61</td><td>33.25</td><td>16.74</td><td>17.44</td><td>10.26</td><td>23.03</td></tr><tr><td>FSSAM [32]</td><td>Pos Only</td><td>28.55</td><td>38.97</td><td>35.89</td><td>17.28</td><td>15.87</td><td>14.83</td><td>25.23</td></tr><tr><td>SANSA [5]</td><td>Pos Only</td><td>26.96</td><td>53.65</td><td>29.83</td><td>15.65</td><td>22.48</td><td>17.07</td><td>27.61</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos Only</td><td>28.80</td><td>56.23</td><td>43.43</td><td>20.05</td><td>19.83</td><td>15.32</td><td>30.61</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos+Neg</td><td>26.22</td><td>20.43</td><td>30.52</td><td>17.06</td><td>6.33</td><td>11.17</td><td>18.62</td></tr><tr><td>CPA (Ours)</td><td>Pos Only</td><td>27.04</td><td>46.64</td><td>38.11</td><td>20.29</td><td>24.46</td><td>14.36</td><td>28.48</td></tr><tr><td>CPA ~(Ours)</td><td>Pos+Neg</td><td>37.11</td><td>56.70</td><td>52.07</td><td>21.98</td><td>28.74</td><td>18.39</td><td>35.83</td></tr><tr><td colspan="9">5-shot</td></tr><tr><td>GF-SAM [34]</td><td>Pos Only</td><td>35.60</td><td>49.44</td><td>45.36</td><td>20.87</td><td>22.76</td><td>19.91</td><td>32.32</td></tr><tr><td>FSSAM [32]</td><td>Pos Only</td><td>37.11</td><td>55.13</td><td>44.51</td><td>17.76</td><td>17.87</td><td>20.61</td><td>32.17</td></tr><tr><td>SANSA [5]</td><td>Pos Only</td><td>34.37</td><td>61.55</td><td>36.89</td><td>18.45</td><td>25.24</td><td>23.87</td><td>33.40</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos Only</td><td>39.07</td><td>62.38</td><td>53.72</td><td>23.11</td><td>25.21</td><td>25.13</td><td>38.10</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos+Neg</td><td>39.76</td><td>50.83</td><td>47.50</td><td>20.74</td><td>21.48</td><td>24.55</td><td>34.14</td></tr><tr><td>CPA (Ours)</td><td>Pos Only</td><td>45.85</td><td>62.23</td><td>49.23</td><td>25.82</td><td>25.83</td><td>26.84</td><td>39.30</td></tr><tr><td>CPA (Ours)</td><td>Pos+Neg</td><td>49.39</td><td>66.00</td><td>56.05</td><td>26.08</td><td>27.64</td><td>27.71</td><td>42.15</td></tr></table>

Table 3: Aggregated CASE results by category-level and attribute-level concepts.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Prompt</td><td colspan="3">1-shot</td><td colspan="3">5-shot</td></tr><tr><td>Category</td><td>Attribute</td><td>Mean</td><td>Category</td><td>Attribute</td><td>Mean</td></tr><tr><td>GF-SAM [34]</td><td>Pos Only</td><td>31.24</td><td>14.81</td><td>23.03</td><td>43.47</td><td>21.18</td><td>32.32</td></tr><tr><td>FSSAM [32]</td><td>Pos Only</td><td>34.47</td><td>15.99</td><td>25.23</td><td>45.58</td><td>18.75</td><td>32.17</td></tr><tr><td>SANSA [5]</td><td>Pos Only</td><td>36.81</td><td>18.40</td><td>27.61</td><td>44.27</td><td>22.52</td><td>33.40</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos Only</td><td>42.82</td><td>18.40</td><td>30.61</td><td>51.72</td><td>24.48</td><td>38.10</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos+Neg</td><td>25.72</td><td>11.52</td><td>18.62</td><td>46.03</td><td>22.26</td><td>34.14</td></tr><tr><td>CPA (Ours)</td><td>Pos Only</td><td>37.26</td><td>19.70</td><td>28.48</td><td>52.44</td><td>26.16</td><td>39.30</td></tr><tr><td>CPA (Ours)</td><td>Pos+Neg</td><td>48.63</td><td>23.04</td><td>35.83</td><td>57.15</td><td>27.14</td><td>42.15</td></tr></table>

## 5.2 Main Results on CASE

Tab. 2 shows that CASE exposes limitations that are not visible in standard positive-only FSS. Positive-only methods remain competitive on tasks close to category matching, especially FineIso, but they cannot explicitly encode what should be rejected. This becomes clear on exclusion, union, and attribute-based tasks, where a target concept cannot be specified by positive similarity alone. Directly adding negative prompts to frozen SAM 3 is also inefective: FSS-SAM3 drops from 30.61 to 18.62 in 1-shot and from 38.10 to 34.14 in 5-shot. This suggests that the frozen model is not calibrated to treat negative boxes as conceptlevel exclusion cues; instead, they can act as unstable suppression signals.

By adapting SAM 3 to use negative exemplars relative to positive anchors, CPA improves over its positive-only variant from 28.48 to 35.83 in 1-shot and from 39.30 to 42.15 in 5-shot. The improvement is consistent across both categorylevel and attribute-level tasks, showing that lightweight adaptation enables the model to interpret positive and negative supports as a structured concept specification rather than as independent box prompts.

As shown in Tab. 3, foundation models favor category-level over attributelevel concepts due to object-centric priors. This highlights the value of CASE as a broader benchmark evaluating visual exclusions, compositional groupings, and attribute-defined targets. Comparing CPA with and without negative prompts shows that negative exemplars provide complementary support, improving the CASE mean from 28.48 to 35.83 in 1-shot and from 39.30 to 42.15 in 5-shot. The gains are more pronounced on category-level concepts, suggesting that negative prompts are especially useful when the target set can be refined by excluding related object categories or disambiguating broad concept groups. Attribute-level gains are smaller but consistent, indicating that material and color reasoning remains dificult. Detailed breakdowns are provided in the supplementary material, Sec. Additional Experimental and Qualitative Results.

Table 4: Comparison with state-of-the-art FSS methods on COCO-20<sup>i</sup>. CPA is evaluated under the standard positive-only protocol.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="5">1-shot</td><td colspan="5">5-shot</td></tr><tr><td>200</td><td>20¹</td><td> $2 0 ^ { 2 }$ </td><td> $2 0 ^ { 3 }$ </td><td>Mean</td><td> $2 0 ^ { 0 }$ </td><td>201</td><td> $2 0 ^ { 2 }$ </td><td>203</td><td>Mean</td></tr><tr><td>PFENet [26]</td><td>Classical FSS</td><td>36.5</td><td>38.6</td><td>34.5</td><td>33.8</td><td>35.8</td><td>36.5</td><td>43.3</td><td>37.8</td><td>38.4</td><td>39.0</td></tr><tr><td>HSNet [17]</td><td>Classical FSS</td><td>36.3</td><td>43.1</td><td>38.7</td><td>38.7</td><td>39.2</td><td>43.3</td><td>51.3</td><td>48.2</td><td>45.0</td><td>46.9</td></tr><tr><td>BAM [12]</td><td>Classical FSS</td><td>43.4</td><td>50.6</td><td>47.5</td><td>43.4</td><td>46.2</td><td>49.3</td><td>54.2</td><td>51.6</td><td>49.6</td><td>51.2</td></tr><tr><td>HDMNet [20]</td><td>Classical FSS</td><td>43.8</td><td>55.3</td><td>51.6</td><td>49.4</td><td>50.0</td><td>50.6</td><td>61.6</td><td>55.7</td><td>56.0</td><td>56.0</td></tr><tr><td>Matcher [14]</td><td>Foundation-based</td><td>52.7</td><td>53.5</td><td>52.6</td><td>52.1</td><td>52.7</td><td>60.1</td><td>62.7</td><td>60.9</td><td>59.2</td><td>60.7</td></tr><tr><td>GF-SAM [34]</td><td>Foundation-based</td><td>56.6</td><td>61.4</td><td>59.6</td><td>57.1</td><td>58.7</td><td>67.1</td><td>69.4</td><td>66.0</td><td>64.8</td><td>66.8</td></tr><tr><td>SANSA [5]</td><td>Foundation-based</td><td>58.9</td><td>62.6</td><td>61.5</td><td>58.0</td><td>60.2</td><td></td><td></td><td></td><td></td><td>67.3</td></tr><tr><td>FSSAM [32]</td><td>Foundation-based</td><td>59.9</td><td>65.6</td><td>62.1</td><td>61.6</td><td>62.3</td><td>68.6</td><td>74.0</td><td>64.5</td><td>69.9</td><td>69.3</td></tr><tr><td>FSS-SAM3 [27]</td><td>Foundation-based</td><td>64.2</td><td>66.8</td><td>67.7</td><td>67.5</td><td>66.6</td><td>68.6</td><td>72.5</td><td>72.5</td><td>73.6</td><td>71.8</td></tr><tr><td>CPA (Ours)</td><td>Ours</td><td>61.1</td><td>64.3</td><td>59.2</td><td>61.6</td><td>61.6</td><td>74.4</td><td>74.1</td><td>74.6</td><td>76.2</td><td>74.8</td></tr></table>

Table 5: Efect of positive-only preservation. CASE is evaluated with positivenegative prompts, while COCO-20<sup>i</sup> is evaluated under the standard FSS protocol.

<table><tr><td rowspan="2">Method</td><td colspan="3">CASE</td><td colspan="2">COCO-20i</td></tr><tr><td>Category</td><td>Attribute</td><td>Mean</td><td>1-shot</td><td>5-shot</td></tr><tr><td>FSS-SAM3 [27]</td><td>25.7</td><td>11.5</td><td>18.6</td><td>66.6</td><td>71.8</td></tr><tr><td>CPA w/o Preservation</td><td>50.6</td><td>25.2</td><td>37.9</td><td>50.8</td><td>66.8</td></tr><tr><td>CPA</td><td>48.6</td><td>23.0</td><td>35.8</td><td>61.6</td><td>74.8</td></tr></table>

## 5.3 Positive-Only FSS Evaluation

Although CPA is trained for SVPS, we evaluate its positive-only FSS behavior on COCO-20<sup>i</sup> under the standard protocol, where support masks are converted into tight bounding boxes. As shown in Tab. 4, CPA obtains 61.6 mIoU in 1- shot, which is lower than FSS-SAM3. However, CPA achieves the best 5-shot performance, 74.8 mIoU, among all methods. This suggests that the adapted model remains responsive to positive visual prompts and can efectively exploit additional support evidence when multiple exemplars are available.

## 5.4 Ablation and Sensitivity Analysis

Efect of preservation regularization. Tab. 5 shows the adaptation–preservation trade-of. Removing preservation gives the best CASE performance, but substan-

Table 6: Negative prompt sensitivity on CASE-NegStress. We report mIoU (%). The full per-task results are provided in the supplementary material.
<table><tr><td>Group</td><td>Method</td><td> $0 ~ \mathrm { N e g }$ </td><td> $1 ~ \mathrm { N e g }$ </td><td> $2 \ \mathrm { N e g }$ </td><td>All Neg</td></tr><tr><td>Category</td><td>FSS-SAM3 [27] CPA</td><td>37.28 49.29</td><td>29.82 59.66</td><td>21.86 60.27</td><td>12.53 59.19</td></tr><tr><td>Attribute</td><td>FSS-SAM3 [27] CPA</td><td>10.04 18.22</td><td>8.68 24.12</td><td>5.40 25.49</td><td>2.74 22.75</td></tr><tr><td>Overall</td><td>FSS-SAM3 [27] CPA</td><td>23.66 33.76</td><td>19.25 41.89</td><td>13.63 42.88</td><td>7.63 40.97</td></tr></table>

tially degrades COCO-20<sup>i</sup> positive-only FSS. Adding preservation slightly lowers CASE mIoU, yet recovers much of the positive-only FSS ability and yields strong 5-shot COCO performance. We therefore use the preservation-regularized CPA as the default, since it improves SVPS while maintaining the usability of SAM 3 under conventional positive prompts.

Negative prompt sensitivity. CASE-NegStress is a dedicated 1-shot subset that fixes the query image, target mask, positive prompts, and candidate negative pool while varying only the negative-prompt count; 0-Neg denotes Pos-only on this subset, not full CASE. As shown in Tab. 6, FSS-SAM3 degrades sharply as more negatives are added, dropping from 23.66 to 7.63 overall mIoU. In contrast, CPA improves from 33.76 to 40.97. This indicates that CPA learns to interpret negative prompts relative to positive anchors, while the original SAM 3 treats them as unstable suppression signals.

## 5.5 Qualitative Results and Limitations

Qualitative Results. Fig. 3 presents representative qualitative comparisons on CASE. SAM 3 with positive-only prompting often misses valid target regions or fails to respect the intended exclusion or attribute constraint. Naively adding negative prompts to SAM 3 remains unstable and may suppress valid target regions together with distractors. CPA with positive-only prompting remains responsive to positive prompts, but may retain distractor regions because the target concept remains underspecified without negative exemplars, especially for exclusion and attribute-based tasks. CPA with positive-negative prompting provides the most consistent predictions, producing masks that better match the concept jointly specified by positive and negative visual exemplars.

Limitations. Attribute-level tasks remain challenging when color or material cues are subtle or ambiguously annotated, and cross-category attribute union is particularly dificult because the model must group visually diverse objects by a shared property rather than by category or shape. CASE is also limited by the categories, masks, and attributes available in COCO and PACO. Additional qualitative results and failure cases are provided in the supplementary material, Secs. Additional Qualitative Results and Failure Cases and Limitations.

![](images/9be165ade129ddf7d8de4155f09044f37b57d1704c91a3b571dafc21f88f94da.jpg)  
Fig. 3: Qualitative comparison on CASE. Green/red boxes in the support image indicate positive/negative visual prompts. Yellow contours denote ground-truth query masks, and blue overlays denote model predictions. We compare SAM 3 (FSS-SAM3) and CPA under positive-only and positive-negative prompting.

## 6 Conclusion

We introduced Specified Visual Prompt Segmentation (SVPS), a new segmentation setting, where positive and negative visual exemplars jointly specify target concepts. To evaluate this setting, we constructed CASE (Concept-Aware Segmentation from Positive and Negative Visual Exemplars), a benchmark built upon COCO and PACO annotations with six category- and attributelevel task families. Our experiments show that while frozen SAM 3 exhibits strong few-shot segmentation capabilities with solely positive prompts, it struggles to reliably interpret negative visual exemplars as exclusion evidence. To address this limitation, we presented Concept Prompt Adaptation (CPA), a lightweight LoRA-based adaptation baseline to enhance positive-negative prompt interpretation. CPA significantly improves performance on CASE while keeping competitive performance under positive-only FSS setups. We believe that CASE can serve as a useful testbed for future research on positive-negative visual prompting, negative exemplars, and concept-level visual prompt segmentation.

Acknowledgements. This work was supported in part by the National Science and Technology Council (NSTC) under grants 114-2221-E-A49-038-MY3, 115-2221-E-A49-119-MY3, 115-2628-E-001-002 and 115-2634-F-A49-011, by IIS grant 3006-C4741, and by MediaTek.

## References

1. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. In: ICLR (2026)

2. Caron, M., Touvron, H., Misra, I., J’egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: ICCV. pp. 9650– 9660 (2021)

3. Chen, L.C., Zhu, Y., Papandreou, G., Schrof, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: ECCV. pp. 801–818 (2018)

4. Chen, T., Zhu, L., Deng, C., Cao, R., Wang, Y., Zhang, S., Li, Z., Sun, L., Zang, Y., Mao, P.: Sam-adapter: Adapting segment anything in underperformed scenes. In: IEEE/CVF International Conference on Computer Vision Workshops. pp. 3367– 3375 (2023)

5. Cuttano, C., Trivigno, G., Averta, G., Masone, C.: Sansa: Unleashing the hidden semantics in sam2 for few-shot segmentation. In: NeurIPS (2025)

6. Ding, H., Liu, C., Wang, S., Jiang, X.: Vision-language transformer and query generation for referring segmentation. In: CVPR. pp. 16321–16330 (2022)

7. Fan, Q., Pei, W., Tai, Y.W., Tang, C.K.: Self-support few-shot semantic segmentation. In: ECCV. pp. 701–719. Springer (2022)

8. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

9. Huang, J., Jiang, K., Zhang, J., Qiu, H., Lu, L., Lu, S., Xing, E.: Learning to prompt segment anything models. arXiv preprint arXiv:2401.04651 (2024)

10. Kazemzadeh, S., Ordonez, V., Matten, M., Berg, T.L.: Referitgame: Referring to objects in photographs of natural scenes. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 787–798 (2014)

11. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: ICCV. pp. 4015–4026 (2023)

12. Lang, C., Cheng, G., Tu, B., Han, J.: Learning what not to segment: A new perspective on few-shot segmentation. In: CVPR. pp. 8057–8067 (2022)

13. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll’ar, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV. pp. 740–755. Springer (2014)

14. Liu, Y., Zhu, M., Li, H., Chen, H., Wang, X., Shen, C.: Matcher: Segment anything with one shot using all-purpose feature matching. In: ICLR (2024)

15. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR. pp. 3431–3440 (2015)

16. Lüddecke, T., Ecker, A.: Image segmentation using text and image prompts. In: CVPR. pp. 7086–7096 (2022)

17. Min, J., Kang, D., Cho, M.: Hypercorrelation squeeze for few-shot segmentation. In: ICCV. pp. 6941–6952 (2021)

18. Nguyen, K., Todorovic, S.: Feature weighting and boosting for few-shot segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 6231–6240 (October 2019)

19. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

20. Peng, B., Tian, Z., Wu, X., Wang, C., Liu, S., Su, J., Jia, J.: Hierarchical dense correlation distillation for few-shot segmentation. In: CVPR. pp. 23641–23651 (2023)

21. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763. PMLR (2021)

22. Ramanathan, V., Kalia, A., Petrovic, V., Wen, Y., Zheng, B., Guo, B., Wang, R., Marquez, A., Kovvuri, R., Kadian, A., Mousavi, A., Song, Y., Dubey, A., Mahajan, D.: Paco: Parts and attributes of common objects. In: CVPR. pp. 7141–7151 (2023)

23. Rao, Y., Zhao, W., Chen, G., Tang, Y., Zhu, Z., Huang, G., Zhou, J., Lu, J.: Denseclip: Language-guided dense prediction with context-aware prompting. In: CVPR. pp. 18082–18091 (2022)

24. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., R"adle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. In: ICLR (2025)

25. Sun, Y., Chen, J., Zhang, S., Zhang, X., Chen, Q., Zhang, G., Ding, E., Wang, J., Li, Z.: Vrp-sam: Sam with visual reference prompt. In: CVPR. pp. 23565–23574 (2024)

26. Tian, Z., Zhao, H., Shu, M., Yang, Z., Li, R., Jia, J.: Prior guided feature enrichment network for few-shot segmentation. IEEE TPAMI 44(2), 1050–1065 (2020)

27. Tsai, Y.J., Lin, Y.Y., Wang, C.Y.: Few-shot semantic segmentation meets sam3. arXiv preprint arXiv:2604.05433 (2026)

28. Wang, K., Liew, J.H., Zou, Y., Zhou, D., Feng, J.: Few-shot image semantic segmentation with prototype alignment. In: ICCV. pp. 9197–9206 (2019)

29. Wang, Z., Lu, Y., Li, Q., Tao, X., Guo, Y., Gong, M., Liu, T.: Cris: Clip-driven referring image segmentation. In: CVPR. pp. 11686–11695 (2022)

30. Xu, J., De Mello, S., Liu, S., Byeon, W., Breuel, T., Kautz, J., Wang, X.: Groupvit: Semantic segmentation emerges from text supervision. In: CVPR. pp. 18134–18144 (2022)

31. Xu, Q., Zhao, W., Lin, G., Long, C.: Self-calibrated cross attention network for few-shot segmentation. In: ICCV. pp. 655–665 (2023)

32. Xu, Q., Zhu, L., Liu, X., Lin, G., Long, C., Li, Z., Zhao, R.: Unlocking the power of sam 2 for few-shot segmentation. In: ICML (2025)

33. Yu, L., Poirson, P., Yang, S., Berg, A.C., Berg, T.L.: Modeling context in referring expressions. In: ECCV. pp. 69–85. Springer (2016)

34. Zhang, A., Gao, G., Jiao, J., Liu, C., Wei, Y.: Bridge the points: Graph-based few-shot segment anything semantically. In: NeurIPS. vol. 37, pp. 33232–33261 (2024)

35. Zhang, J.W., Sun, Y., Yang, Y., Chen, W.: Feature-proxy transformer for few-shot segmentation. In: NeurIPS. vol. 35, pp. 6575–6588 (2022)

36. Zhang, R., Jiang, Z., Guo, Z., Yan, S., Pan, J., Ma, X., Dong, H., Gao, P., Li, H.: Personalize segment anything model with one shot. In: ICLR (2024)

37. Zhang, R., Zhang, W., Fang, R., Gao, P., Li, K., Dai, J., Qiao, Y., Li, H.: Tipadapter: Training-free adaption of clip for few-shot classification. In: ECCV. pp. 493–510. Springer (2022)

38. Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: CVPR. pp. 2881–2890 (2017)

39. Zhou, C., Loy, C.C., Dai, B.: Extract free dense labels from clip. In: ECCV. pp. 696–712. Springer (2022)

40. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for visionlanguage models. In: CVPR. pp. 16816–16825 (2022)

41. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. IJCV 130(9), 2337–2348 (2022)

# Supplementary Material of CASE: Concept-Aware Segmentation from Positive and Negative Visual Exemplars

Yi-Jen Tsai<sup>1</sup>, Yen-Yu Lin<sup>1</sup>, and Chien-Yao Wang<sup>2⋆</sup>

<sup>1</sup> National Yang Ming Chiao Tung University, Taiwan tsai.cs14@nycu.edu.tw, lin@cs.nycu.edu.tw <sup>2</sup> Institute of Information Science, Academia Sinica, Taiwan kinyiu@iis.sinica.edu.tw

## 1 CASE Benchmark Construction

This section provides additional details on the construction of CASE. CASE is designed to evaluate whether a segmentation model can infer a target concept from both positive and negative visual exemplars. Unlike standard few-shot semantic segmentation benchmarks, where the support set usually defines a single positive class, CASE episodes require the model to parse compositional intent relations such as inclusion, exclusion, fine-grained isolation, functional grouping, attribute selection, attribute union, and attribute exclusion.

## 1.1 Data Sources

CASE is built from two annotation sources. The category-level tasks, including SuperEx, FineIso, and Union, are constructed from MSCOCO instance segmentation annotations [13]. We use COCO train2017 to generate candidate training episodes and COCO val2017 to generate fixed evaluation episodes. The attributelevel tasks, including AttrIso, AttrUnion, and AttrEx, are constructed from PACO-LVIS annotations [22]. PACO provides object-level category annotations and attribute labels, which allow us to define visual concepts based on color and material properties.

For PACO-based tasks, we use object-level categories and filter out partlevel categories whose category names contain a part delimiter. This avoids constructing episodes in which the target concept is defined over object parts rather than complete object instances. We further restrict attribute concepts to visually grounded color and material attributes. The color set contains black, blue, brown, green, grey, orange, pink, purple, red, white, and yellow. The material set contains stone, wood, rattan, fabric, crochet, wool, leather, velvet, metal, paper, plastic, glass, and ceramic.

To specify, not every COCO/PACO image is used: only images satisfying the task-specific support and target constraints are eligible. Episode and mask construction is programmatic from the source annotations; only the high-level Union group definitions are manually specified.

Table S1: Data sources used for each CASE task family. Category-level tasks are constructed from COCO, while attribute-level tasks are constructed from PACO-LVIS.
<table><tr><td>Task family Source annotation</td><td></td><td>Split usage</td></tr><tr><td>SuperEx</td><td>COCO instance segmentation train2017</td><td></td><td>val2017</td></tr><tr><td>FineIso</td><td></td><td>COCO instance segmentation train2017</td><td>val2017</td></tr><tr><td>Union</td><td>COCO instance segmentation train2017</td><td></td><td>val2017</td></tr><tr><td>AttrIso</td><td>PACO-LVIS object attributes train / val</td><td></td><td></td></tr><tr><td>AttrUnion</td><td>PACO-LVIS object attributes train / val</td><td></td><td></td></tr><tr><td>AttrEx</td><td>PACO-LVIS object attributes train / val</td><td></td><td></td></tr></table>

## 1.2 Episode Format

Each CASE episode in the benchmark contains a target image, a frozen support set, a task label, an intent description, a frozen target mask, and frozen positive/negative visual prompts. The format is designed for reproducible evaluation: the support examples, prompt boxes, and target masks are fixed in advance, so diferent methods are evaluated on exactly the same episodes.

A support image contains positive and negative bounding-box prompts. Positive prompts indicate visual exemplars that should be included in the target concept, while negative prompts indicate exemplars that should be excluded or used as discriminative counterexamples. The target image contains a groundtruth mask encoded in COCO-style run-length encoding (RLE). This avoids requiring users to re-run the benchmark construction code.

A simplified released episode format is shown below.

{   
"episode\_id": "...",   
"task": "SuperEx",   
"intent": "All vehicle MINUS car",   
"target": {   
"image\_id": ...,   
"image\_path": "...",   
"gt\_foreground\_pixels": ...,   
"gt\_mask\_rle": {   
"size": [height, width],   
"counts": "..."   
},   
"pos\_bboxes": [...],   
"neg\_bboxes": [...]   
},   
"support": [   
{   
"image\_id": ...,   
"image\_path": "...",   
"prompts": {

Table S2: Detailed statistics of the CASE benchmark. We report the number of candidate training episodes, fixed evaluation episodes, concept definitions, and average prompts per episode for each task family. The overall average prompt counts are weighted by the number of candidate training episodes.
<table><tr><td>Task</td><td>Source</td><td>#Train Ep. #Eval Ep. #Concepts Avg. Pos</td><td></td><td></td><td></td></tr><tr><td>SuperEx</td><td>COCO</td><td>156,000</td><td>9,004</td><td>78</td><td>6.71 2.31</td></tr><tr><td>FineIso</td><td>COCO</td><td>140,727</td><td>8,662</td><td>79</td><td>2.43 6.17</td></tr><tr><td>Union</td><td>COCO</td><td>50,000</td><td>1,806</td><td>25</td><td>5.04 3.96</td></tr><tr><td>AttrEx</td><td>PACO</td><td>12,909</td><td>1,528</td><td>109</td><td>2.09 1.04</td></tr><tr><td>AttrIso</td><td>PACO</td><td>66,942</td><td>1,100</td><td>94</td><td>1.04 2.09</td></tr><tr><td>AttrUnion</td><td>PACO</td><td>34,812</td><td>1,116</td><td>21</td><td>2.15 1.64</td></tr><tr><td>Overall</td><td>COCO/PACO</td><td>461,390</td><td>23,216</td><td>406</td><td>3.94 3.56</td></tr></table>

```json
"pos_bboxes": [...],
"neg_bboxes": [...]
}
}
]
}
```

In the internal construction stage, each episode is first generated with a task-specific dynamic ground-truth configuration. This configuration specifies how the target mask should be produced from the underlying COCO or PACO annotations. For example, SuperEx stores a base supercategory and an excluded category, FineIso stores a target category, Union stores a fixed set of positive categories, AttrIso stores a target object category and a target attribute, AttrUnion stores a target attribute, and AttrEx stores a semantic object group and an excluded attribute. During release compilation, this dynamic configuration is converted into a frozen RLE target mask.

## 1.3 Benchmark Statistics

Tab. S2 provides detailed statistics for CASE. The training column reports candidate episodes used to sample training instances, while the evaluation column reports fixed episodes used for deterministic comparison. The number of concepts counts the distinct structured concept definitions within each task. The average prompt counts are computed across training episodes, and the overall average is weighted by the number of candidate training episodes per task.

## 1.4 Train/Test Splits

CASE follows the original train/validation split of its source annotations. COCObased tasks use train2017 for candidate training episodes and val2017 for evaluation episodes. PACO-based tasks follow the PACO train/val split. Thus, the training and evaluation episodes are image-disjoint by construction. The split is not concept-disjoint: the same structured concept definition may appear in both splits, while the underlying images remain disjoint.

Table S3: General construction parameters used in CASE. The overall average prompt counts reported in the main paper are weighted by the number of candidate training episodes.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Minimum target foreground area</td><td>500 pixels</td></tr><tr><td>Training support candidate pool size</td><td>300</td></tr><tr><td>Maximum training support pool size</td><td>100</td></tr><tr><td>Maximum target images per concept</td><td>2000</td></tr><tr><td>Training episode seed</td><td>42</td></tr><tr><td>Evaluation support sampling seed</td><td>42</td></tr><tr><td>Evaluation protocols</td><td>1-shot and 5-shot</td></tr><tr><td>Support nesting</td><td>1-shot ⊂ 5-shot</td></tr><tr><td>Maximum positive prompts per support image 3</td><td></td></tr><tr><td>Maximum negative prompts per support image 3</td><td></td></tr></table>

The evaluation split is frozen for reproducibility. For each validation episode, we first sample a 5-shot support set using a global random seed. The corresponding 1-shot episode is obtained by taking the first support image from the 5-shot set. Therefore, the 1-shot support set is nested within the 5-shot support set. This design ensures that the 1-shot and 5-shot protocols evaluate the same target episodes while difering only in the number of available support exemplars.

At training time, support images are dynamically sampled from a larger candidate support pool, which increases prompt diversity while keeping the underlying intent definition fixed. Unless otherwise specified, candidate training episodes are generated with a fixed random seed, and each concept is capped by a maximum number of target images to avoid severe imbalance.

## 1.5 Prompt Sampling Protocol

CASE uses bounding boxes as visual prompts. Each support image may contain multiple positive and negative boxes. Positive boxes are selected from instances that satisfy the intended concept, whereas negative boxes are selected from semantically or visually related distractors. Thus, negatives are task-related distractors rather than arbitrary classes. All prompts correspond to object-level instances; CASE does not cover stuf categories. This design forces the model to infer not only what should be segmented, but also what should be rejected.

To reduce annotation noise and duplicated prompts, candidate boxes are filtered by an IoU-based duplicate suppression rule. A newly selected box is discarded if it overlaps too strongly with an existing prompt box. We also avoid selecting positive boxes that collide with negative boxes, since such overlaps would make the support intent ambiguous.

Training and evaluation difer in strictness. For training, we adopt a targetrelaxed but support-strict protocol. A training target image is required to contain at least one positive foreground instance, but it is not always required to contain a negative distractor. This increases concept coverage and avoids over-restricting the training distribution. In contrast, the support set remains strict: support images are required to contain explicit positive and negative exemplars so that each training episode provides a clear contrastive intent.

Table S4: Task-specific construction rules in CASE. Pos. and Neg. indicate the positive and negative support prompts, respectively.
<table><tr><td>Task</td><td>Intent</td><td>Target mask</td><td>Pos. prompts</td><td>Neg. prompts</td></tr><tr><td>SuperEx</td><td>category.</td><td>All objects in a Union of all instances Instances from non- Instances from the COCO supercategory in the supercategory excluded categories excluded category. except one excluded minus instances of in the same supercat- the excluded cate- egory.</td><td></td><td></td></tr><tr><td>FineIso</td><td>category within a get category. COCO supercate-</td><td>gory. Only one fine-grained Instances of the tar- Instances of the tar- Instances from other get category.</td><td></td><td>categories in the same supercategory.</td></tr><tr><td>Union</td><td>gory. egory group, such as belonging to the pos- least two positive distractor categories public transportation itive categories of the categories when pos- associated with the or rest furniture.</td><td>A fixed high-level cat- Union of all instances Instances from at Instances from fixed group. sible.</td><td></td><td></td></tr><tr><td>AttrIso</td><td>specified color or ma- possess the target at- tribute.</td><td>Objects in a fixed Instances in the tar- Same-group instances Same-group object group with a get object group that with the target at- stances with different</td><td>group.</td><td>in-</td></tr><tr><td></td><td>terial attribute. tribute. regardless of cate-</td><td>AttrUnion All object instances All valid object-level Instances from multi- Instances with differ- with a specified color instances that possess ple categories sharing ent attributes from or material attribute, the target attribute. the target attribute. the same attribute</td><td></td><td></td></tr><tr><td>AttrEx</td><td>gory. ing a specified color instances with the ex- from the same at- or material attribute. cluded attribute.</td><td>Objects in a semantic Instances in the se- Instances with non- Instances with the ex- object group exclud- mantic group minus excluded attributes cluded attribute.</td><td></td><td>family.</td></tr></table>

For evaluation, we adopt a stricter discriminative protocol. Whenever the task definition involves exclusion or disambiguation, target images are required to contain both foreground objects satisfying the intent and related distractors. Support images are also required to contain positive and negative exemplars. As a result, the evaluation split directly tests whether a model can use negative visual prompts to suppress distractors rather than merely segmenting all objects that resemble the positive exemplars.

## 1.6 Dynamic Ground-truth Mask Generation

The target mask of each episode is generated from the source annotations according to its task-specific intent. For COCO-based tasks, masks are generated from instance segmentation annotations. For PACO-based tasks, masks are generated by filtering object annotations according to semantic object groups and visual attributes. The dynamic mask rules are summarized in Table S4. In exclusion tasks, the target mask is computed by subtracting the excluded instances from the base foreground. In union tasks, the target mask is the union of all instances that satisfy the positive concept. In attribute tasks, the target mask is obtained by selecting or excluding instances according to PACO attribute labels.

[SuperEx] Episode: SuperEx\_vehicle\_bus\_0504 Intent: All vehicle MINUS bus

![](images/178954e3b6d7470f58cd2c3e9108dc2c96ee4eff84b94fdc1e7945e9d6663d70.jpg)  
Fig. S1: Example SuperEx episode. The intent is to segment all vehicles except buses.

## 2 Task-specific Details

This section describes the six CASE task families in detail. Each task is designed to evaluate a diferent form of concept reasoning from positive and negative visual exemplars. In all task illustrations, green boxes on support images denote positive visual prompts and red boxes denote negative visual prompts. On target images, the green overlay denotes the ground-truth target mask. Red boxes on target images are shown only to indicate distractor instances for visualization and are not provided as model input during evaluation.

## 2.1 SuperEx: Supercategory Exclusion

SuperEx evaluates whether a model can infer a broad category-level concept while excluding a specified fine-grained category. Each concept is defined as

$$
y = \mathcal { C } _ { \mathrm { s u p e r } } \setminus \mathcal { C } _ { \mathrm { n e g } } ,
$$

where $\mathcal { C } _ { \mathrm { s u p e r } }$ is a COCO supercategory and $\mathcal { C } _ { \mathrm { n e g } }$ is one excluded category within that supercategory. For example, an episode may define the intent “all vehicles except cars.” Positive prompts are sampled from non-excluded categories in the same supercategory, such as buses and trains, while negative prompts are sampled from the excluded category, such as cars.

The target mask is generated by first taking the union of all instances in the base supercategory and then subtracting all instances belonging to the excluded category:

$$
M = M ( \mathcal { C } _ { \mathrm { s u p e r } } ) \setminus M ( \mathcal { C } _ { \mathrm { n e g } } ) .
$$

[Finelso] Episode: Finelso\_vehicle\_truck\_0703 Intent: ONLY truck (from vehicle)

![](images/1bd81634d87649385ffc03d77045908baba292f77e96d482bfb27a16b7122094.jpg)  
Fig. S2: Example FineIso episode. The intent is to isolate trucks from other vehicles.

This task tests whether a model can use negative exemplars to prevent overgeneralization. A model that relies only on positive prompts may infer the broad supercategory and incorrectly include the excluded category.

For evaluation, target images are required to contain both valid foreground instances and excluded-category distractors. Support images are required to include at least two positive categories and at least one negative exemplar. This makes SuperEx a direct test of category-level exclusion.

## 2.2 FineIso: Fine-grained Isolation

FineIso evaluates whether a model can isolate a single fine-grained category from related categories within the same supercategory. Each concept is defined as

$$
\begin{array} { r } { \mathcal { V } = \mathcal { C } _ { \mathrm { t a r g e t } } , } \end{array}
$$

where $\mathcal { C } _ { \mathrm { t a r g e t } }$ is one COCO category. For example, an episode may define the intent “only dogs” within the animal supercategory. Positive prompts are sampled from the target category, while negative prompts are sampled from other categories in the same supercategory, such as cats, sheep, horses, or cows.

The target mask is the union of all instances of the target category:

$$
M = M ( \mathcal { C } _ { \mathrm { t a r g e t } } ) .
$$

The negative prompts do not define a category to subtract from a broader foreground. Instead, they act as discriminative counterexamples that specify the intended granularity of the concept.

[Union] Episode: Union\_Electronic\_Displays\_0573 Intent: Electronic Displays  
![](images/a23aceb8fc4f4bf17f8436d7a5883a5c58801328a06c4c728e4352f91d71a44a.jpg)  
Fig. S3: Example Union episode. The intent is to segment electronic displays across related object categories (electronic devices).

For evaluation, target images are required to contain the target category and at least one same-supercategory distractor. Support images are required to contain at least one positive instance and multiple negative distractors. This makes FineIso particularly challenging for models that treat visually related categories as interchangeable.

## 2.3 Union: Function-driven Category Union

Union evaluates whether a model can infer a high-level concept that spans multiple COCO categories. Unlike SuperEx and FineIso, Union is not derived directly from COCO supercategories. Instead, we manually define a fixed set of high-level intent groups. Each group contains positive categories that share a functional or semantic relation and negative categories that serve as related distractors.

Each Union concept is defined as

$$
y = \bigcup _ { c \in \mathcal { C } _ { \mathrm { p o s } } } c ,
$$

where $\mathcal { C } _ { \mathrm { p o s } }$ is the manually defined positive category set for a given intent. The target mask is the union of all positive-category instances:

$$
M = \bigcup _ { c \in \mathcal { C } _ { \mathrm { p o s } } } M ( c ) .
$$

For example, the intent “public transportation” contains buses and trains as positive categories and cars and motorcycles as negative distractors. Similarly, “rest furniture” contains beds and couches as positives and chairs and dining tables as negatives.

Table S5: Representative examples of manually defined Union concepts.
<table><tr><td>Intent</td><td>Positive categories</td><td>Negative categories</td></tr><tr><td>Public Transportation</td><td>bus, train</td><td>car, motorcycle</td></tr><tr><td>Human Powered Mobility bicycle, skateboard</td><td></td><td>car, motorcycle</td></tr><tr><td>Rest Furniture</td><td>bed, couch</td><td>chair, dining table</td></tr><tr><td>Domestic Pets</td><td>dog, cat</td><td>cow, horse, zebra</td></tr><tr><td>Drink Containers</td><td>cup, wine glass, bottle</td><td>fork, knife</td></tr><tr><td>Electronic Displays</td><td>tv, laptop, cell phone</td><td>remote, keyboard, mouse, microwave</td></tr><tr><td>Cutlery</td><td>fork, knife, spoon</td><td>bowl, cup, wine glass, dining table</td></tr></table>

[Attrlso] Episode: Attrlso\_Container\_glass\_0179 Intent: ONLY glass Container  
![](images/d4c84662ecc2bb268a955e7b924dd2d0aba8926cd62c664a0fc7e92d89f93df1.jpg)  
Fig. S4: Example AttrIso episode. The intent is to segment glass containers while excluding containers with other attributes.

The negative prompts in Union are not subtracted from a base mask. Instead, they define the boundary of the high-level concept by showing related categories that should not be included. This setting tests whether a model can infer an abstract group from positive exemplars rather than simply segmenting all objects from a broad COCO supercategory. Table S5 lists representative Union concepts. The full implementation uses a fixed set of such groups across all experiments.

## 2.4 AttrIso: Attribute Isolation within an Object Group

AttrIso evaluates whether a model can isolate objects with a specified visual attribute within a fixed object group. Unlike FineIso, where the concept is defined only by object category, AttrIso defines the target by combining an object group and an attribute:

$$
\mathcal { Y } = \{ x \mid x \in \mathcal { G } , \ a _ { \mathrm { t a r g e t } } \in A ( x ) \} ,
$$

[AttrUnion] Episode: AttrUnion ceramic 0173 Intent: ALL ceramic objects  
![](images/b93340968a046a5aa22c4ed9d49515d4142e0cf596d9334cd64679f68959fb0b.jpg)  
Fig. S5: Example AttrUnion episode. The intent is to segment all ceramic objects.

where $\mathcal { G }$ is the target object group, A(x) is the set of PACO attributes assigned to instance x, and $a _ { \mathrm { t a r g e t } }$ is the target color or material attribute.

For example, an AttrIso episode may define the intent “glass containers.” Positive prompts contain container instances with the target attribute, such as glass containers. Negative prompts contain instances from the same object group with diferent attributes from the same attribute family, such as plastic containers. This design forces the model to isolate the target attribute while keeping the object group fixed.

The target mask is generated as

$$
M = \bigcup _ { x \in \mathcal { G } } \mathbf { 1 } [ a _ { \mathrm { t a r g e t } } \in A ( x ) ] M ( x ) .
$$

This mask includes only instances that match both the target category and the target attribute. Instances from the same category but with diferent attributes are excluded, and instances from other categories are not part of the target even if they share the same attribute.

For evaluation, target images are required to contain at least one positive instance and at least one same-category attribute distractor. Support images are required to contain at least one positive prompt and multiple negative prompts from the same object category. We additionally require the negative prompts to contain multiple distractor attributes whenever possible, so that the model cannot solve the episode by suppressing only one specific negative instance.

## 2.5 AttrUnion: Cross-category Attribute Union

AttrUnion evaluates whether a model can segment all objects that share a specified visual attribute across categories. Each concept is defined as

$$
\mathcal { V } = \{ x \mid a _ { \mathrm { t a r g e t } } \in A ( x ) \} ,
$$

where $a _ { \mathrm { t a r g e t } }$ is a target color or material attribute. Unlike AttrIso, AttrUnion does not restrict the target to a semantic group. The target mask includes all valid object-level instances with the target attribute:

$$
M = \bigcup _ { x } \mathbf { 1 } [ a _ { \mathrm { t a r g e t } } \in A ( x ) ] M ( x ) .
$$

For example, an AttrUnion episode may ask the model to segment all wooden objects or all red objects, regardless of their object categories. Positive prompts are sampled from multiple categories that share the target attribute. Negative prompts are sampled from objects that do not have the target attribute but have other attributes from the same attribute family. This prevents the model from solving the task by using object category alone.

For evaluation, we require the target image to contain the target attribute across at least two object categories whenever possible. Support images are also required to contain positive exemplars from multiple categories and negative exemplars with diferent attributes. This makes AttrUnion a direct test of categoryinvariant attribute reasoning.

## 2.6 AttrEx: Attribute Exclusion within a Semantic Group

AttrEx evaluates whether a model can segment objects in a semantic group while excluding a specified visual attribute. Each concept is defined as

$$
\mathcal { Y } = \{ x \mid x \in \mathcal { G } , \ a _ { \mathrm { e x c l u d e d } } \ \notin \ A ( x ) \} ,
$$

where $\mathcal { G }$ is a semantic object group and $a _ { \mathrm { e x c l u d e d } }$ is the excluded color or material attribute. For example, an episode may define the intent “non-red vehicles” or “non-wooden furniture.”

The target mask is generated by taking all valid instances in the semantic group and subtracting instances that possess the excluded attribute:

$$
M = M ( { \mathcal { G } } ) \setminus \bigcup _ { x \in { \mathcal { G } } } \mathbf { 1 } [ a _ { \mathrm { e x c l u d e d } } \in A ( x ) ] M ( x ) .
$$

Positive prompts contain instances from the same semantic group with nonexcluded attributes. Negative prompts contain instances with the excluded attribute. This construction requires the model to interpret negative prompts as an attribute-level exclusion rather than a category-level exclusion.

For training, target images are required to contain at least one valid nonexcluded foreground instance, while support images must contain both positive

[AttrEx] Episode: AttrEx Container rattan 0177 Intent: Non-rattan Container

![](images/5fbfa303c10047ceb0f746a810696ab646da343aee2ccf0ea72bc8cbbc160a03.jpg)  
Fig. S6: Example AttrEx episode. The intent is to segment non-rattan containers while excluding rattan containers.

and negative exemplars. For evaluation, target and support images are constructed more strictly to include explicit excluded-attribute distractors. We also require the positive prompts to provide diversity either across object categories or across non-excluded attributes. This prevents the task from collapsing into a single-instance matching problem.

## 2.7 CASE-NegStress

CASE-NegStress is a dedicated 1-shot diagnostic subset that fixes the query image, target mask, positive support, and candidate negative pool while varying only the number of negative prompts. The 0-Neg condition denotes positive-only prompting on this same NegStress subset rather than the full CASE. The purpose is not to define a new task family, but to stress-test whether the model’s predictions change appropriately when the negative evidence increases. A robust positive-negative visual prompt segmentation model should preserve regions supported by the positive exemplars while suppressing regions that match the negative exemplars. In contrast, a model that ignores negative prompts will produce similar predictions under the original CASE setting and the NegStress setting. Therefore, CASE-NegStress helps distinguish genuine negative-prompt reasoning from positive-only support matching.

## 3 Implementation Details

## 3.1 Shared-canvas Formulation

CPA follows the shared-canvas formulation used in the main paper. Given a K-shot support set ${ \cal { S } } = \{ ( I _ { i } ^ { s } , B _ { i } ^ { + } , B _ { i } ^ { - } ) \} _ { i = 1 } ^ { K }$ and a query image $I ^ { q } .$ , we compose them into a single RGB canvas $I ^ { c }$ of size $1 0 0 8 \times 1 0 0 8$ . The positive and negative support boxes are transformed from their original image coordinates into the coordinate system of $I ^ { c }$ and are used as SAM 3 box prompts. The model predicts composite canvas logits $Z ^ { c }$ , from which we crop the query region to obtain $Z ^ { q }$ for loss computation and evaluation.

For 1-shot episodes, we use a vertical support-query layout. The support image occupies the top half of the canvas, and the query image occupies the bottom half, where each region is represented as $[ x , y , w , h ]$

$$
\varOmega _ { \mathrm { { s u p } } } ^ { \mathrm { { 1 - s h o t } } } = [ 0 , 0 , 1 0 0 8 , 5 0 4 ] , \qquad \varOmega _ { \mathrm { { q r y } } } ^ { \mathrm { { 1 - s h o t } } } = [ 0 , 5 0 4 , 1 0 0 8 , 5 0 4 ] .
$$

For 5-shot episodes, we use an L-shaped layout. The first three support images are placed in the top row, the fourth and fifth support images are placed in the left column, and the query image is placed in the bottom-right region. With canvas size 1008, the support thickness is

$$
s = \operatorname { r o u n d } ( 0 . 3 3 \times 1 0 0 8 ) = 3 3 3 ,
$$

and the query region size is

$$
q = 1 0 0 8 - s = 6 7 5 .
$$

Therefore, the 5-shot query region is

$$
\Omega _ { \mathrm { q r y } } ^ { \mathrm { 5 - s h o t } } = [ 3 3 3 , 3 3 3 , 6 7 5 , 6 7 5 ] .
$$

The top support row consists of three regions of size $3 3 6 \times 3 3 3$ , and the left support column consists of two regions of sizes $3 3 3 \times 3 3 7$ and $3 3 3 \times 3 3 8 .$ , respectively. This layout keeps all five support examples visible while preserving a large square query region. Figure S7 illustrates both layouts.

All support boxes are converted into normalized canvas coordinates in $[ c _ { x } , c _ { y } , w , h ]$ format. Under positive-negative prompting, both $B ^ { + }$ and $B ^ { - }$ are provided to the model. Under positive-only prompting, only $B ^ { + }$ is retained.

## 3.2 LoRA Configuration

Let f denote the original frozen SAM 3 model and $f _ { \theta }$ denote the CPA-adapted model. We keep all pretrained SAM 3 parameters frozen and optimize only the LoRA parameters θ. LoRA adapters are inserted into transformer encoder and decoder layers. In particular, we apply LoRA to modules whose names end with q\_proj, k\_proj, v\_proj, out\_proj, linear1, or linear2.

Unless otherwise specified, we use LoRA rank $r = 1 6$ , scaling factor $\alpha = 3 2$ dropout 0.05, and no bias adaptation. This results in 1.13M trainable parameters. The original SAM 3 parameters remain frozen throughout training, so $f _ { \theta }$ difers from f only through the inserted LoRA updates.

![](images/87edb9e11ff74e48dc3dfb35a00b0ceb5c1a9e171b7a61e32b7ec54ae6301bd3.jpg)  
Fig. S7: Shared-canvas layouts used by CPA. Left: the 1-shot setting uses a vertical layout, where the support image occupies the top half of the 1008 × 1008 canvas and the query image occupies the bottom half. Right: the 5-shot setting uses an L-shaped layout, where three support images are placed in the top row, two support images are placed in the left column, and the query image occupies the bottom-right region.

## 3.3 Training Objective

CPA is optimized under two prompting conditions, following the notation in the main paper. In the positive-negative condition, the adapted model $f _ { \theta }$ receives both positive and negative boxes and predicts query logits $Z _ { \theta , \pm } ^ { q } \mathrm { : }$

$$
B ^ { + } , B ^ { - }  f _ { \theta }  Z _ { \theta , \pm } ^ { q } .
$$

The prediction is supervised by the CASE ground-truth query mask $M ^ { q }$

$$
\mathcal { L } _ { \mathrm { s v p s } } ^ { \pm } = \mathcal { L } _ { \mathrm { m a s k } } \left( Z _ { \theta , \pm } ^ { q } , M ^ { q } \right) .
$$

In the positive-only preservation condition, the adapted model $f _ { \theta }$ and the original frozen model $f$ are evaluated under the same positive-only input. The adapted model produces $Z _ { \theta , + } ^ { q }$ , while the frozen model produces $Z _ { f , + } ^ { q } ;$

$$
B ^ { + }  f _ { \theta }  Z _ { \theta , + } ^ { q } , ~ B ^ { + }  f  Z _ { f , + } ^ { q } .
$$

The preservation loss penalizes the discrepancy between their probability maps:

$$
\mathcal { L } _ { \mathrm { p r e s } } ^ { + } = \left. \sigma ( Z _ { \theta , + } ^ { q } ) - \sigma ( Z _ { f , + } ^ { q } ) \right. _ { 2 } ^ { 2 } ,
$$

where $\sigma ( \cdot )$ denotes the sigmoid function.

Table S6: CPA training and optimization settings.  
Component Setting   
Base model SAM 3   
Frozen model notation f   
Adapted model notation $f _ { \theta }$   
Input canvas size 1008 × 1008   
1-shot layout Vertical support-query split   
5-shot layout L-shaped support-query canvas   
Prompt type Support-image bounding boxes   
Positive-negative condition $B ^ { + }$ and B   
Positive-only condition B<sup>+</sup> only   
Trainable modules LoRA adapters   
Frozen modules All pretrained SAM 3 weights   
LoRA target modules q\_proj, k\_proj, v\_proj, out\_proj, linear1, linear2   
LoRA rank r = 16   
LoRA scaling factor α = 32   
LoRA dropout 0.05   
Trainable parameters 1.13M   
Optimizer AdamW   
Learning rate $1 \times 1 0 ^ { - 4 }$   
Weight decay $1 \times 1 0 ^ { - 4 }$   
Random seed 123   
Mixed precision bfloat16 autocast on CUDA   
Gradient clipping Max norm 1.0   
Preservation weight $\lambda _ { \mathrm { p r e s } } = 1 . 0$

The final training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s v p s } } ^ { \pm } + \lambda _ { \mathrm { p r e s } } \mathcal { L } _ { \mathrm { p r e s } } ^ { + } .
$$

We set $\lambda _ { \mathrm { p r e s } } = 1 . 0$ unless otherwise specified.

In implementation, $\mathcal { L } _ { \mathrm { m a s k } }$ combines sigmoid focal loss and Dice loss:

$$
\mathcal { L } _ { \mathrm { m a s k } } = 2 0 \cdot \mathcal { L } _ { \mathrm { f o c a l } } + \mathcal { L } _ { \mathrm { d i c e } } ,
$$

where the sigmoid focal loss uses $\alpha = 0 . 2 5$ and $\gamma = 2 . 0$ . When SAM 3 returns multiple mask candidates, we aggregate the candidate logits by taking their element-wise maximum before computing the losses.

## 3.4 Optimization Details

We train CPA on the mixed CASE training set. The dataloader shufles candidate training episodes, and support examples are sampled according to the requested shot setting. For 1-shot training, one support image is placed in the vertical layout. For 5-shot training, five support images are placed in the L-shaped layout.

We use AdamW with learning rate $1 \times 1 0 ^ { - 4 }$ and weight decay $1 \times 1 0 ^ { - 4 }$ . The default training configuration uses a batch size of 1, two dataloader workers, 10 epochs, a random seed of 123, and mixed precision with bfloat16 autocasting on CUDA. Gradients are clipped with a maximum norm of 1.0. LoRA checkpoints are saved after each epoch.

Table S7: Split-wise FSSAM results on CASE. FSSAM is evaluated using the publicly released split-specific checkpoints. The main paper reports the average over four splits.
<table><tr><td>Shot</td><td>Task</td><td>Split 0</td><td>Split 1</td><td>Split 2</td><td>Split 3</td><td>Avg.</td></tr><tr><td rowspan="6">1-shot</td><td>FineIso</td><td>40.75</td><td>37.74</td><td>38.25</td><td>39.15</td><td>38.97</td></tr><tr><td>SuperEx</td><td>27.88</td><td>28.80</td><td>29.19</td><td>28.35</td><td>28.55</td></tr><tr><td>Union</td><td>36.89</td><td>34.92</td><td>35.96</td><td>35.80</td><td>35.89</td></tr><tr><td>AttrIso</td><td>18.69</td><td>13.90</td><td>15.04</td><td>15.84</td><td>15.87</td></tr><tr><td>AttrUnion</td><td>15.04</td><td>15.25</td><td>13.90</td><td>15.13</td><td>14.83</td></tr><tr><td>AttrEx</td><td>17.25</td><td>17.16</td><td>17.31</td><td>17.40</td><td>17.28</td></tr><tr><td rowspan="6">5-shot</td><td>FineIso</td><td>54.84</td><td>54.52</td><td>55.72</td><td>55.45</td><td>55.13</td></tr><tr><td>SuperEx</td><td>35.36</td><td>38.74</td><td>37.07</td><td>37.26</td><td>37.11</td></tr><tr><td>Union</td><td>41.85</td><td>46.98</td><td>44.26</td><td>44.94</td><td>44.51</td></tr><tr><td>AttrIso</td><td>19.17</td><td>16.40</td><td>18.18</td><td>17.71</td><td>17.87</td></tr><tr><td>AttrUnion</td><td>19.36</td><td>21.52</td><td>20.09</td><td>21.49</td><td>20.61</td></tr><tr><td>AttrEx</td><td>18.10</td><td>17.46</td><td>17.57</td><td>17.90</td><td>17.76</td></tr></table>

## 3.5 Inference and Evaluation Protocol

At inference time, we use the same shared-canvas formulation as in training. Given $s$ and $I ^ { q } .$ , we compose the canvas $I ^ { c } { } _ { ; }$ , transform the support boxes into the canvas coordinate system, and apply $f _ { \theta }$ to obtain composite logits $Z ^ { c }$ . We crop the query region to obtain $Z ^ { q }$ and derive the binary query prediction as

$$
\hat { \cal M } ^ { q } = { \bf 1 } [ \sigma ( Z ^ { q } ) > \tau ] ,
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and τ is the mask threshold. We use $\tau = 0 . 5$ for all reported results. We then resize the query prediction to the original query resolution and compute mIoU against the ground-truth query mask $M ^ { q }$

For CASE, the fixed evaluation episode provides the support images and the positive/negative support boxes. We use the vertical layout for 1-shot evaluation and the L-shaped layout for 5-shot evaluation. For standard FSS benchmarks, we follow the conventional positive-only setting: support masks are converted into tight bounding boxes and used as positive visual prompts, and no negative prompts are provided. For COCO- ${ \bf \nabla } \cdot 2 0 ^ { i }$ , the CASE-trained CPA checkpoints are evaluated directly on all four folds without COCO-20<sup>i</sup> retraining or fold-specific tuning. This evaluation measures CPA’s positive-only FSS behavior after adaptation to CASE rather than assuming that the original behavior is fully preserved.

## 3.6 Baseline Adaptation

For SAM-based FSS baselines that do not natively support negative visual exemplars, we adapt the CASE inputs to their supported prompt formats. GF-SAM and FSSAM use positive support masks derived from the positive boxes, while SANSA uses its native positive-box prompt interface. Since these methods were designed for conventional positive-only FSS, their CASE results should be interpreted as positive-only reference baselines rather than fully specified SVPS.

Table S8: Full aggregated CASE results across concept dimensions. Category and Attribute correspond to the source concept type, while Exclusion, Isolation, and Union correspond to the compositional operation required by the task.
<table><tr><td>Method</td><td>Prompt</td><td>Category</td><td>Attribute</td><td>Exclusion</td><td>Isolation</td><td>Union</td><td>Mean</td></tr><tr><td colspan="8">1-shot</td></tr><tr><td>GF-SAM [34]</td><td>Pos Only</td><td>31.24</td><td>14.81</td><td>20.80</td><td>26.52</td><td>21.75</td><td>23.03</td></tr><tr><td>FSSAM [32]</td><td>Pos Only</td><td>34.47</td><td>15.99</td><td>22.92</td><td>27.42</td><td>25.36</td><td>25.23</td></tr><tr><td>SANSA [5]</td><td>Pos Only</td><td>36.81</td><td>18.40</td><td>21.30</td><td>38.07</td><td>23.45</td><td>27.61</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos Only</td><td>42.82</td><td>18.40</td><td>24.43</td><td>38.03</td><td>29.38</td><td>30.61</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos+Neg</td><td>25.72</td><td>11.52</td><td>21.64</td><td>13.38</td><td>20.85</td><td>18.62</td></tr><tr><td>CPA (Ours)</td><td>Pos Only</td><td>37.26</td><td>19.70</td><td>23.67</td><td>35.55</td><td>26.24</td><td>28.48</td></tr><tr><td>CPA (Ours)</td><td>Pos+Neg</td><td>48.63</td><td>23.04</td><td>29.55</td><td>42.72</td><td>35.23</td><td>35.83</td></tr><tr><td colspan="8">5-shot</td></tr><tr><td>GF-SAM [34]</td><td>Pos Only</td><td>43.47</td><td>21.18</td><td>28.24</td><td>36.10</td><td>32.64</td><td>32.32</td></tr><tr><td>FSSAM [32]</td><td>Pos Only</td><td>45.58</td><td>18.75</td><td>27.43</td><td>36.50</td><td>32.56</td><td>32.17</td></tr><tr><td>SANSA [5]</td><td>Pos Only</td><td>44.27</td><td>22.52</td><td>26.41</td><td>43.40</td><td>30.38</td><td>33.40</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos Only</td><td>51.72</td><td>24.48</td><td>31.09</td><td>43.80</td><td>39.43</td><td>38.10</td></tr><tr><td>FSS-SAM3 [27]</td><td>Pos+Neg</td><td>46.03</td><td>22.26</td><td>30.25</td><td>36.16</td><td>36.03</td><td>34.14</td></tr><tr><td>CPA (Ours)</td><td>Pos Only</td><td>52.44</td><td>26.16</td><td>35.84</td><td>44.03</td><td>38.04</td><td>39.30</td></tr><tr><td>CPA (Ours)</td><td>Pos+Neg</td><td>57.15</td><td>27.14</td><td>37.74</td><td>46.82</td><td>41.88</td><td>42.15</td></tr></table>

For the main-paper comparison, we use the publicly released checkpoints provided by the authors’ oficial repositories and do not fine-tune these baselines on CASE. A separate same-data SANSA control is reported below to isolate the efect of in-domain CASE exposure. For methods with a single released checkpoint, we report the result obtained from that checkpoint. For FSSAM, the oficial release provides split-specific checkpoints. We therefore evaluate FSSAM on each of the four splits and report the average across splits in the main paper. The reported FSSAM average is computed at the metric level over the four splitspecific evaluations, not by ensembling predictions from multiple checkpoints. The split-wise FSSAM results are shown in Table S7.

## 4 Additional Experimental and Qualitative Results

## 4.1 Full Concept-dimension Breakdown on CASE

Tab. S8 provides the full concept-dimension breakdown corresponding to the aggregated results in the main paper. Category and Attribute summarize the source type of the target concept. Exclusion, Isolation, and Union summarize the operation required to define the target set.

The results show that negative visual exemplars are useful beyond categorylevel matching. CPA improves performance on the exclusion and isolation dimensions, indicating that negative prompts help reject explicitly excluded categories or attributes and sharpen distinctions among related objects. Union-style concepts also benefit, suggesting that negative exemplars help constrain broad or compositional target groups. Attribute-level dimensions remain more challenging, consistent with the observation that color and material cues are often subtle and context-dependent.

Table S9: 1-shot CASE precision–recall analysis. All methods use the same evaluation episodes and $\tau = 0 . 5$
<table><tr><td>Model</td><td>Prompt</td><td>mIoU</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>FSS-SAM3</td><td>Pos</td><td>30.61</td><td>64.10</td><td>36.21</td><td>46.28</td></tr><tr><td>CPA (Ours)</td><td>Pos</td><td>28.48</td><td>35.12</td><td>58.02</td><td>43.75</td></tr><tr><td>FSS-SAM3</td><td>Pos+Neg</td><td>18.62</td><td>68.29</td><td>20.66</td><td>31.72</td></tr><tr><td>CPA (Ours)</td><td>Pos+Neg</td><td>35.83</td><td>63.32</td><td>55.72</td><td>59.28</td></tr></table>

Table S10: In-domain training control on CASE (mIoU).
<table><tr><td>Method</td><td>CASE FT</td><td>Prompt</td><td>1-shot</td><td>5-shot</td></tr><tr><td>SANSA</td><td>No</td><td>Pos</td><td>27.61</td><td>33.40</td></tr><tr><td>SANSA</td><td>Yes</td><td>Pos</td><td>30.01</td><td>36.27</td></tr><tr><td>CPA (Ours)</td><td>Yes</td><td>Pos+Neg</td><td>35.83</td><td>42.15</td></tr></table>

## 4.2 Additional Quantitative Analyses

Precision–recall behavior under positive and negative prompting. CPA is adapted to infer concepts jointly from positive and negative exemplars. Consequently, under positive-only inference it retains more candidate regions than FSS-SAM3, yielding higher recall but lower precision. When negative exemplars are provided, CPA recovers precision while retaining substantially higher recall, producing higher F1 and mIoU than FSS-SAM3 under Pos+Neg prompting.

Same-data training control. To separate the benefit of CASE training data from the ability to use negative exemplars, we fine-tune SANSA on the same CASE training set while retaining its native positive-only prompting interface, which does not support negative visual exemplars. In-domain training improves SANSA, but CPA remains +5.82/+5.88 mIoU higher in 1-/5-shot, showing that CASE exposure alone is insuficient to explain CPA’s gains.

SAM 3 text-prompt controls. SAM 3 supports text prompts, but exclusion intent is not reliably expressed by the tested text formulations. On SuperEx, the reviewer-inspired exclusion prompt “[supercategory] except [excluded]” obtains 5.79 mIoU as a text-only prompt. Combining text with visual prompts remains below the corresponding visual-only FSS-SAM3 results in the main paper (28.80/39.07 mIoU in 1-/5-shot).

Human validation of CASE. To quantify benchmark quality and the PACO ambiguity discussed in the failure analysis, two annotators independently reviewed 300 fixed 1-shot evaluation episodes (50 per task family) using four 1–5 criteria: intent quality, positive-exemplar consistency, negative-exemplar consistency, and target-mask consistency. The four mean scores are 4.22/4.38/4.54/4.10, with medians of 5 for all criteria. Across criteria, 84.7–86.7% of paired ratings are within one point, with linear-weighted $\kappa = 0 . 4 2 – 0 . 5 7$ . PACO-derived attribute episodes are more ambiguous than COCO-derived category episodes, especially for target-mask consistency (3.57 vs. 4.63), quantitatively supporting the limitation discussed below.

Table S11: SAM 3 text controls on SuperEx (mIoU). The two blocks are separate controls and are not row-wise paired.
<table><tr><td>Prompt Type</td><td>1-shot 5-shot</td></tr><tr><td>Text only</td><td>5.79</td></tr><tr><td>Text + Pos Visual</td><td>14.26 26.03</td></tr><tr><td>Text + Pos+Neg Visual 13.58</td><td>28.05</td></tr></table>

<table><tr><td colspan="2">Text Formulation</td><td>SuperEx</td></tr><tr><td> $\scriptstyle { \mathrm { ~ " ~ } } [ \operatorname { s u p e r c a t e g o r y } ] ^ { \mathrm { , } }$ </td><td></td><td>26.28</td></tr><tr><td></td><td>“[supercategory] except [excluded]&quot;</td><td>5.79</td></tr><tr><td>“[supercategory] not [excluded]&quot;</td><td></td><td>2.60</td></tr></table>

Table S12: Human validation of CASE. Agreement is computed over 300 paired episode ratings.
<table><tr><td>Criterion</td><td>Mean</td><td>Within ±1</td><td>Linear κ</td></tr><tr><td>Intent quality</td><td>4.22</td><td>84.7%</td><td>0.453</td></tr><tr><td>Positive consistency</td><td>4.38</td><td>85.3%</td><td>0.490</td></tr><tr><td>Negative consistency</td><td>4.54</td><td>86.0%</td><td>0.417</td></tr><tr><td>Target consistency</td><td>4.10</td><td>86.7%</td><td>0.570</td></tr></table>

## 4.3 Additional Qualitative Results

This section provides additional qualitative comparisons complementing the main paper. Unless otherwise noted, green boxes on support images denote positive visual prompts and red boxes denote negative visual prompts. On query images, yellow contours denote ground-truth masks and blue overlays denote model predictions. We compare the frozen SAM 3 model f and the CPA-adapted model $f _ { \theta }$ under both positive-only and positive-negative prompting. In the supportprompt column, the support regions follow the same shared-canvas layout used for inference, while the query region is grayed out.

Figs. S8 and S9 show representative 1-shot examples across category-level and attribute-level CASE tasks. The examples illustrate that positive-only prompting is often insuficient for specified concepts, since the model may segment visually related distractors together with the intended target or follow object-category priors instead of the specified attribute. Negative prompts are necessary to specify what should be rejected, but frozen SAM 3 does not consistently use them as concept-level constraints. CPA provides more stable predictions by adapting the model to interpret positive and negative visual exemplars jointly.

Fig. S10 provides 5-shot qualitative examples. Compared with the 1-shot setting, multiple support images provide broader visual evidence about the intended target concept. However, simply increasing the number of positive exemplars does not fully resolve exclusion, isolation, union, or attribute ambiguity.

Table S13: Summary of common failure modes in CASE.
<table><tr><td>Failure mode</td><td>Description</td></tr><tr><td>Attribute ambiguity</td><td>The model fails to distinguish vi- PACO attributes can be visu- sually subtle colors or materials, ally subtle, lighting-dependent, such as glossy plastic versus glass or partially visible. or dark brown versus black.</td></tr><tr><td>Over-exclusion</td><td>Negative exemplars suppress The negative prompt may share valid foreground regions to- shape, texture, or context with gether with distractors. the target concept.</td></tr><tr><td>Under-exclusion</td><td>Distractor instances matching The model relies on positive sim- the negative exemplars remain in ilarity and does not fully use neg- the prediction. ative evidence.</td></tr><tr><td></td><td>Category-attribute entanglement The model follows object cat- Foundation segmentation models egory more strongly than the retain object-centric priors from specified attribute. pretraining.</td></tr><tr><td>Small or occluded objects</td><td>The model misses small, par- Shared-canvas resizing reduces tially visible, or heavily occluded fine details, and support-query target instances. matching becomes less reliable.</td></tr><tr><td>Ambiguous grouping</td><td>The model inconsistently han- Some manually defined groups dles broad union concepts or se- reflect functional relations that mantic object groups. are not always visually apparent.</td></tr></table>

The positive-negative CPA predictions are generally more consistent with the target masks, indicating that the model benefits from both support diversity and explicit negative evidence.

## 5 Failure Cases and Limitations

Although CPA improves the use of positive and negative visual exemplars, several limitations remain. Fig. S11 and Tab. S13 summarize representative failure modes. Attribute-level tasks are particularly challenging. Color and material labels may depend on lighting, viewpoint, texture, and object state. Some materials are also visually confounded with object categories; for example, glass may be correlated with bottles or cups, while fabric may be correlated with furniture or clothing. As a result, the model may segment objects that match the category prior rather than the specified attribute.

Fig. S11 illustrates three representative limitations. First, attribute-level tasks remain vulnerable to category–attribute entanglement: the model may segment objects that match a strong object-category prior, such as containers, rather than strictly following the specified attribute constraint. Second, some PACO attributes are visually ambiguous, especially for color and material labels affected by lighting, texture, viewpoint, or annotation convention. Third, negative exemplars may still lead to under-exclusion when the excluded category remains visually salient in the query image.

Negative prompts can also introduce failure modes. When positive and negative exemplars are visually similar, the model may over-exclude valid target regions. Conversely, when negative exemplars are visually diverse or weakly related to the query distractors, the model may under-exclude and behave similarly to positive-only FSS. These errors indicate that positive-negative visual prompting requires not only recognizing individual support instances, but also inferring the intended relation between inclusion and exclusion evidence.

CASE itself is limited by the annotation spaces of COCO and PACO. COCO provides reliable category-level instance masks but does not contain fine-grained attributes. PACO provides attribute annotations, but the available color and material labels are incomplete and sometimes ambiguous. Union concepts are manually defined and capture only a subset of possible high-level semantic groupings. Extending CASE to richer attributes, open-vocabulary concepts, and more diverse real-world domains is an important direction for future work.

Finally, CPA is designed as a lightweight and conservative adaptation of SAM 3. The preservation regularization mitigates positive-only FSS degradation, but does not fully preserve the frozen model in the 1-shot setting; it also introduces an adaptation–preservation trade-of. Stronger adaptation may improve CASE performance further, but could degrade the original ability of SAM 3. Exploring stronger architectures or training objectives that improve SVPS without sacrificing positive-only FSS remains future work.

Support + Prompt

Query & GT

SAM 3 (pos-only)

SAM 3 (pos+neg)  
CPA (pos-only)  
CPA (pos+neg)  
![](images/dca99ab85d070d5bc6dc7326e984011dd1bd2fb4e80095d4e48956437ce974c7.jpg)  
Fig. S8: Additional 1-shot qualitative comparisons on category-level CASE tasks. The figure includes two examples from each category-level task family: SuperEx, FineIso, and Union. In the support-prompt column, the upper region shows the 1- shot support image with positive and negative box prompts, while the lower gray region indicates that the query image is omitted from the prompt visualization. Frozen SAM 3 with positive-only prompts often follows visual similarity without resolving the intended exclusion, isolation, or union constraint. Directly adding negative prompts to frozen SAM 3 is unreliable, either having little efect or suppressing valid target regions. CPA with positive-only prompts produces cleaner positive-prompt responses, while CPA with positive-negative prompts better matches the concept jointly specified by positive and negative exemplars.

![](images/182e569b4adcf24d280d4394031f013a2627be71718891cea8819f198eeb3d23.jpg)  
Fig. S9: Additional 1-shot qualitative comparisons on attribute-level CASE tasks. The figure includes two examples from each attribute-level task family: AttrEx, AttrIso, and AttrUnion. These examples highlight the dificulty of specifying targets by color or material cues rather than object category alone. Frozen SAM 3 tends to rely on category-level or appearance-level similarity and may fail to respect the intended attribute constraint. CPA improves the use of positive and negative attribute exemplars, although attribute-level concepts remain more challenging than categorylevel concepts due to subtle visual cues and context-dependent material appearance.

![](images/d36c7fa093102cd7c4b31e9fa29a3ff3a34481bcf6ae5c741286cfc5f4b43548.jpg)  
Fig. S10: Additional 5-shot qualitative examples on CASE. The figure includes one example from each of the six task families. The support-prompt column follows the L-shaped 5-shot shared-canvas layout, where the five support images occupy the top row and left column, and the lower-right gray region indicates that the query image is omitted from the prompt visualization. Multiple support exemplars provide more evidence about intra-concept variation, but the target concept can remain ambiguous without properly using negative exemplars. CPA uses the expanded support set more efectively under positive-negative prompting.

![](images/8ae5c5146d07f9dcd33a222cff26e9148d688a4ce17bb7872975d213e3fc8ad9.jpg)  
Fig. S11: Representative failure cases. Yellow contours denote ground-truth masks and blue overlays denote model predictions. Red circles highlight erroneous regions. Top: in attribute-isolation cases, the adapted model may follow object-category priors and expand from the specified attribute target to visually related containers. Middle: attribute annotations can be visually ambiguous; for example, the distinction between brown and non-brown containers may depend on lighting, texture, and annotation convention. Bottom: negative exemplars do not always fully suppress distractors, leading to under-exclusion when excluded categories remain visually salient in the query image.