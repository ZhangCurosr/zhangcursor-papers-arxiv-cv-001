# A Unified Evaluation Framework of Physical Adversarial Robustness in the Edge VLM Era

Eric Liang<sup>1</sup> Alan Luo<sup>2,3∗</sup> Kaiwen Yuan<sup>4</sup>

<sup>1</sup>Saratoga High School

<sup>2</sup>Google

<sup>3</sup>University of Illinois Urbana-Champaign <sup>4</sup>Mecka.ai

{eric.jinchen.liang@gmail.com, alanluo3@illinois.edu, kaiwen@mecka.ai}

## Abstract

Recent advancements in Vision-Language Models (VLMs) enable end-to-end visual grounding via language prompts and generative decoding, demonstrating largely improved zero-shot and open-vocabulary detection capabilities. As a result, VLMs are rapidly being deployed at the edge in safety-critical real-world systems. However, the physical adversarial robustness of edge VLMs has not been systematically studied, and the fundamental differences in output with modern object detectors create methodological challenges for fair evaluation. To compare the two families directly, we evaluate every selected representative model on one outcome: whether an adversarial patch worn on a person evades detection of the model. We report this by introducing a unified metric, conditional attack success rate (cASR), counted only on images the model still localizes under a matched gray occluder, and partition every failure into omission (no box), substitution (a box on the patch), and displacement (a box on neither). We benchmark the physical adversarial vulnerability of compact grounding VLMs against a range of object detectors, and find that the VLMs can be as vulnerable as the detectors. We disclose that failure mode is largely associated with the output format of a model rather than its architecture family. Of the three, only omission is visible to a downstream system; displacement and substitution return confident, well-formed boxes, making them potentially more dangerous failures. Further, we find that, under distribution shift, VLMs that primarily failed by omission instead shift to failing more by displacement/substitution. Thus, a grounding VLM can inherit the detector’s vulnerability while showing the possibility to adopt more dangerous failure modes. Our code and data are available at https: //github.com/ericliang1/physical-adversarial-robustness.

## 1 Introduction

Vision-language models (VLMs) are transitioning from describing a scene to localizing within one. Given a natural-language reference to an object, a VLM returns a bounding box. In embodied and autonomous settings [5, 29], a planner then conditions on that box directly [9], with no confidence score to threshold and no signal of uncertainty, so a correct box and an incorrect one arrive in the same form. Recent grounding VLMs are increasingly built for real-world deployment, with compact sub-4B variants suited to embedded accelerators [43, 30, 39, 4, 20]. Existing edge inference pipelines, however, were built around object detectors, and whether a grounding VLM is equally robust in their place cannot be settled on clean images, where both families localize a person at high accuracy.

We therefore induce failures by training a physical adversarial patch against each model, a printed texture designed to suppress detection when worn on a person in the real world—an established threat to detectors that has not been tested against grounding VLMs on a common scale [33, 6]. The suppression objective of each adversarial patch drives down its corresponding model’s native output, making adversarial robustness directly comparable across architectures and exposing the inherent vulnerability of each model’s spatial grounding.

We compare the two families across eleven models (Table 1) on the one predicate they share, whether a box lands on the target. We measure failure by introducing conditional attack success rate (cASR), which scores a patch only on images the model localizes under a matched gray occluder, and partition grounding failure into three modes—omission (no box), substitution (a box on the patch), and displacement (a box on neither); Figure 1 illustrates all three. We apply partitions in an occlusioncontrolled comparison of the two families across held-out and out-of-distribution images.

![](images/44fca46e58a2908caefc7cd61f1017265b21828159c851caecc8c4d64d38fdaf.jpg)  
Figure 1: The three failure modes: omission, displacement, and substitution.

Two findings follow. First, grounding VLMs can be as vulnerable as the detectors under adversaria patches in deployment settings. Substituting one for the other does not necessarily reduce a system’s failure rate under a white-box attack. Second, the two families do not fail in the same way. Whether an attacked model omits or displaces is instead largely associated with its output format rather than its architecture family, while architecture also plays a role in the failure partition.

## 2 Related work

## 2.1 Object detection and grounding VLMs

Traditional object detectors [12, 27] have centered on convolutional neural networks (CNNs) for their superior capability for edge-device deployment due to low inference latency and excellent energy efficiency. The recent success of the transformer [34] has inspired vision transformer (ViT) based vision tasks [10, 14] that rethink the architecture in a natural language processing schema. Shortly after, transformer-based object detectors [7] came to dominate leaderboards [40, 44], demonstrating accuracy and generalization on large datasets. On the other hand, transformers still suffer from a lack of inductive biases and weaker performance on small and low-data problems [23, 10]. Therefore, both CNN and transformer models are commonly used for edge deployment. To make detectors more general purpose, open-vocabulary capabilities have been integrated into CNN [8] and transformer architectures [21], allowing detections to be driven by natural language prompts.

More recently, general purpose large language models have trended towards multimodality and have thus also emerged into visual grounding capabilities [43, 30, 39, 32, 4]. These grounding models not only yield bounding box coordinates of objects but also query objects in an open-vocabulary manner. Even though most open-weight multimodal large language models (MLLMs) and VLMs are very heavy, with billions, and in the largest cases trillions, of parameters [32], there are also a handful of compact VLM variants that are promising for deployment in lightweight settings [39, 20]. As a result, it may be a matter of time before edge devices are armed with VLMs either in place or in combination with detectors. Understanding the security of models in these two different detection schemas is therefore critical.

## 2.2 Adversarial attacks on object detection

The first adversarial attacks on modern object detectors date back to 2017, focusing on imperceptible perturbations applied to an image [38]. Later studies focused on transferability and generalization [33, 6] of adversarial attacks. These works utilized adversarial patches – textures capable of existing in diverse scenes in the real world, revealing a model’s vulnerabilities to specific localized patterns. The same idea has been further extended from the RGB domain to other (and mixed) modalities [2, 1]. Underneath the variety, the threat model is consistent: the attacker optimizes pixels in texture space against a model’s native output.

However, language models admit no such formulation. Gradients do not act on discrete tokens, so attacks there are found by searching over candidate strings, with gradients through the token embeddings guiding the search [35, 45]. The form of an attack on a language interface is dramatically different: it can be typed, injected or retrieved.

VLMs sit at the intersection of both methods and are exposed to both vulnerabilities. Early work showed that CLIP-style encoders can classify objects by text in the image rather than by the object itself [13], while later attacks introduced traditional imperceptible perturbation-based adversaries that transfer across models and prompts [42, 24]. A few recent works have begun to target grounded localization directly through training-time backdoors, though they are digital rather than physical [26, 17]. On the other hand, vision-language-action models, which use VLM backbones to act in robotics, have been demonstrated to fail by physical attacks in their visual input [36, 22]; however, these attacks target the model’s actions rather than its localization.

Despite these advances, understanding the physical vulnerabilities of a model’s localization requires evaluating the way traditional detector attacks do [33]. Existing physical-attack evaluations cover detectors alone [18]; none of these works addresses real-time VLMs small enough to be deployed on edge devices, which our work aims to address.

## 3 Methodology

## 3.1 Models and datasets

For models, representative traditional closed-set object detectors [16, 15, 25], open-vocabulary detectors [8, 41, 21], and edge VLMs [43, 30, 39, 4, 20] whose parameter counts are no larger than 4B are selected for evaluation.

We conduct our training on three datasets for people detection—COCO [19], PASCAL VOC [11] and AdvT-Shirt-1K [37]—and each contains ground-truth labels for every human in each scene. COCO and PASCAL VOC contain diverse person images in various scenes while AdvT-Shirt contains people with adversarial patches worn on shirts. However, we treat AdvT-Shirt the same as COCO and PASCAL VOC, as its garments were optimized against older detectors and have no effect on the models we test, as seen in the clean column of Table 1. We manually filter each dataset into subsets that contain only images with exactly one person. This ensures that every model is evaluated on a matched single-target detection task, removing ambiguity from VLM prompts, as VLMs generally output a single bounding box. Across all models and in-distribution datasets we use an 80/20 training/validation split, giving 1,324 training and 331 held-out validation images. We additionally evaluate out-of-distribution on 999 filtered single-person images from the Waymo Open Dataset [31], which represent street-level pedestrian views captured from a moving vehicle. Exact sources, filter counts, and split construction are given in Appendix B. For our experiments, all images are resized to the native input resolution of the vision model: all VLMs use 448 × 448 while all detectors use $6 4 0 \times 6 4 0$

## 3.2 Adversarial patch training

We train on 1,324 images—622 COCO, 539 PASCAL VOC, 163 AdvT-Shirt—each containing exactly one person. For each image, a single adversarial patch is applied to the center of the person. The patch is a square texture with an original resolution of 160 × 160. When placed, it is resized relative to the person ground-truth label according to $s _ { i } = a \cdot \operatorname* { m i n } ( w _ { i } , h _ { i } )$ , where $s _ { i }$ is the side length of the placed patch in image $i , w _ { i }$ and $h _ { i }$ denote the width and height of the target bounding box, and a denotes the scaling factor, with $a = 0 . 5$ for our main experiment. This scale allows the patch to cover the center of the person while leaving important person features such as heads and limbs fully visible, representing the size of a feasibly wearable garment.

![](images/9e2b5d629dc6eca8310771be34bb9d29566c57ff1ae93232b530ccc88f999bdd.jpg)  
Figure 2: Patch training pipeline.

![](images/275325ff23cc047becf7064845cb5319ba04f6343ace7fcc1fa6097703763a20.jpg)  
Figure 3: Trained adversarial patches for each model. Patches trained on detectors contain high frequency patterns while patches trained on VLMs contain features that resemble text.

We use white-box optimization to independently train an image-agnostic patch for each model, as summarized in Figure 2. Each attack is optimized to directly suppress the output of its respective target model. For detectors, patches are trained to suppress the final detection of the person. For VLMs, patches are trained to suppress the bounding box tokens, reducing the model’s likelihood of emitting the spatial information of the patched individual. The resulting patches are shown in Figure 3.

Optimization is performed across 500 epochs with a learning rate of 0.01 for all models, allowing all patches to converge and ensuring a fair comparison between them. To make the patches physically robust and effective across different scenes, we train with Expectation Over Transformation (EOT) [3], applying random rotations drawn uniformly from [−20<sup>◦</sup>, 20<sup>◦</sup>] and multiplicative brightness jitter drawn uniformly from [0.9, 1.1] $( \mathrm { i . e . , \pm 1 0 \% } )$ . The patch scale is applied when the patch is placed on the target and is not part of the EOT distribution. All patches are applied digitally; “physical” represents the threat model rather than the evaluation medium.

The patch is trained using the loss function

$$
L = L _ { s } + \alpha L _ { n p s } + \beta L _ { t v } ,\tag{1}
$$

where $L _ { s }$ suppresses the output, $L _ { n p s }$ enforces printability [28], $L _ { t v }$ enforces smoothness, and α and $\beta$ are two hyperparameters to balance the scale. We set $\alpha = 0 . 0 1$ and $\beta = 0 . 0 1$ , keeping the

printability and smoothness terms small enough to constrain the patch to be physically realizable without trading away attack strength. Adapted from Thys et al. [33], the non-printability and smoothness terms are

$$
L _ { n p s } = \frac { 1 } { | P | } \sum _ { p \in P } \operatorname* { m i n } _ { c \in C } { ( p - c ) ^ { 2 } } ,\tag{2}
$$

$$
L _ { t v } = \frac { 1 } { N } \sum _ { i , j } \left( | p _ { i , j } - p _ { i + 1 , j } | + | p _ { i , j } - p _ { i , j + 1 } | \right) ,\tag{3}
$$

where $p _ { i , j }$ denotes a patch pixel, $p \mathrm { ~ a ~ }$ pixel of the patch $P ,$ c a color in the set of printable colors $C ,$ and $| P |$ the number of patch pixels; in ${ \cal L } _ { t v } ,$ each absolute difference is taken per color channel (red, green, blue) and $N$ is the total number of differences. For each model, $L _ { t v }$ and $L _ { n p s }$ remain the same, and are used to constrain the patch to be physically realizable, while $L _ { s }$ performs the same suppression task on the final differentiable prediction scores before the model’s final output decision, which is unique to each model’s architecture and native output structure. Concretely, $L _ { s }$ takes one of three forms,

$$
L _ { s } = \left\{ \begin{array} { l l } { \operatorname* { m a x } _ { k } \ z _ { k } } & { \mathrm { c l o s e d - s e t \ d e t e c t o r } , } \\ { \operatorname* { m a x } _ { k } \ \sin ( r _ { k } , t ) } & { \mathrm { o p e n \mathrm { - } v o c a b u l a r y \ d e t e c t o r } , } \\ { \log p _ { \theta } ( b \mid q ) } & { \mathrm { v i s i o n \mathrm { - } l a n g u a g e \ m o d e l } , } \end{array} \right.\tag{4}
$$

where $z _ { k }$ is the person confidence of proposal $k ;$ sim $( r _ { k } , t )$ is the similarity between region k and the “person” text embedding $t ;$ and $p _ { \theta } ( b \mid q )$ is the likelihood of decoding the target’s box tokens b under prompt q. Minimizing $L _ { s }$ suppresses the person confidence, the region–text similarity, or the grounding likelihood, respectively.

## 4 Evaluation framework

We create a unified evaluation framework that aligns adversarial attack evaluation across heterogeneous model architectures. The framework ensures that the effects of adversarial attacks can be directly compared by framing success as the ability of a patched target individual to evade detection.

## 4.1 Configuration

The 331 validation images comprise 155 COCO, 135 PASCAL VOC, and 41 AdvT-Shirt; the 999 Waymo images are evaluated separately as the out-of-distribution set. Unless stated otherwise, the following default conditions hold: the overlap threshold is $\tau = 0 . 5$ , the standard detection criterion [11]; the patch is placed at the scale it was trained at; and detectors are read at their canonical confidence threshold.

## 4.2 Output alignment

To align with detectors, we prompt each VLM to produce one bounding box per image in its native grounding format, then normalize all outputs to a standard xyxy format. Any output that is malformed or fails to correspond to a valid bounding box is treated as a detection failure, enforcing evaluation based on spatial correctness. Each model is scored on one box $\hat { b } _ { i } \colon$ : for a detector, it is the highest-confidence person detection clearing γ; for a VLM, it is the single box it emits.

Since traditional object detectors produce many detections per target instance ranked by confidence scores, while vision-language models produce a single detection, a detector is read at a chosen confidence threshold whereas a VLM is read at its one emission. For VLMs, we decode greedily and do not consider the temperature, which is standard for single-target detection tasks. For detectors, we evaluate at the confidence thresholds at which their family is conventionally operated, ensuring that all models are evaluated in deployment-relevant settings. Closed-set detectors are read at the confidence threshold $\gamma = 0 . 5$ . Open-vocabulary detectors are read at a lower confidence threshold, at $\gamma = 0 . 3$

## 4.3 Occlusion control and conditional ASR

In evaluation, a gray digital patch can cause a failure by occluding part of the target, causing a detection to fail. As a result, standard attack success rate (ASR) can overestimate the adversarial effect of a patch, while subtraction-based flooring methods may underestimate the adversarial effect when the gray patch itself causes a high failure rate. To separate true adversarial success from occlusion and baseline model error, we evaluate each adversarial patch only on the images for which the corresponding model can successfully detect the person under the gray-patched condition. We define this metric as conditional ASR (cASR). Let G be the set of images the model localizes correctly (Intersection over Union, Io ${ \mathrm { J } } \geq \tau .$ , against the ground-truth box $b _ { i } )$ under the gray occluder, and let $f _ { i } = 1$ when IoU $\begin{array} { r } { \mathrm { J } ( \hat { b } _ { i } , b _ { i } ) < \tau . } \end{array}$ , where $\hat { b } _ { i }$ is the box predicted on the adversarially patched image. Then

$$
\mathrm { c A S R } = \frac { 1 } { \left| G \right| } \sum _ { i \in G } f _ { i } ,\tag{5}
$$

the fraction of gray survivors that the adversarial patch breaks.

The floor’s magnitude corresponds to localization tightness: tight boxes lose overlap once the target center is covered, whereas loose ones retain enough to clear τ. The models most affected by the control are therefore the most precise rather than the weakest—and a benchmark reporting raw attack success would penalize them for that precision.

## 4.4 Failure partition

Conditional attack success quantifies failure frequency but not failure mode. Omission is tracked when no box of a proper format forms. Substitution is tracked when a box is formed and closely aligns with the patch, measured by $\mathrm { I o U } ( \hat { b } _ { i } , F _ { i } ) \ge 0 . 5$ , where $F _ { i }$ is the patch footprint. Displacement is tracked when a box is formed but lands on neither the target nor the patch, falling below τ against $b _ { i }$ and below 0.5 against $F _ { i }$ . The three are exclusive by construction and exhaustive by the last case, so they sum to cASR.

## 5 Results

Baseline localization. Table 1 reports localization with no patch applied, where a model counts as correct if it emits a box overlapping the ground truth at IoU ≥ 0.5. The criterion is identical for every model. In-distribution accuracy spans 83.0% to 100.0%. On the out-of-distribution Waymo set accuracy drops for every model, reflecting the harder detection task of street-level pedestrian views from a moving vehicle. Most models remain well above 65%, while MiniCPM-V 4.6 falls to 51.1% and InternVL3-2B to 27.6% on the same images. By construction, cASR counts a failure only on images the model localizes correctly under the matched gray occluder (Eq. 5), so no failure below is attributable to baseline localization. This conditioning means each model’s Waymo cASR is read over its own, smaller gray-control set. For InternVL3-2B and MiniCPM-V 4.6, which localize driving pedestrians poorly, this set is as small as 305 and 366 of 999 images, respectively.

## 5.1 VLMs can be as vulnerable as detectors

Table 2 reports cASR (Eq. 5), computed over the gray-control set, so occlusion alone contributes near zero. Grounding VLMs rank among the most affected models in the benchmark: InternVL3-2B and PaliGemma 2-3B fail on 57.7% and 55.2% of the gray-control set, higher than two of the six detectors we test. AdvT-Shirt gives the cleanest comparison, as every model localizes at 95% or above with no patch applied. With the adversarial patch applied, InternVL3-2B and PaliGemma 2-3B fail on 100% and 97.6% of their gray-control sets, comparable to the worst detectors (Table 1), YOLO11s and YOLO26s, which both reach 100%. Substituting a VLM for a detector therefore does not necessarily lower the failure rate.

The most robust models also come from both families. MiniCPM-V 4.6 is the most robust overall, failing on only 9.2% of the gray-control set. YOLO-World, though a detector, sits at 18.2%, more robust than three of the five VLMs. The VLMs span 9.2% to 57.7%, while the detectors span 18.2% to 78.3%. The two ranges largely overlap, with the variation within each family exceeding the difference between them. Robustness is therefore largely a property of the individual model rather than its family.

Table 1: Clean detection accuracy and conditional attack success (cASR). Bold marks the lowest cASR within each family.
<table><tr><td colspan="2"></td><td colspan="6">In-distribution (trained)</td><td colspan="3">Out-of-distribution</td></tr><tr><td colspan="2"></td><td colspan="2">COCO</td><td colspan="2">VOC</td><td colspan="2">AdvT-Shirt</td><td colspan="2">Waymo</td></tr><tr><td>Model</td><td>Params</td><td>Clean</td><td>cASR</td><td>Clean</td><td>cASR</td><td>Clean</td><td>cASR</td><td>Clean</td><td>cASR</td></tr><tr><td colspan="10">Closed-set detectors</td></tr><tr><td>YOLO11s</td><td>9.5M</td><td>96.1</td><td>80.3</td><td>91.9</td><td>66.7</td><td>100.0</td><td>100.0</td><td>74.5</td><td>86.1</td></tr><tr><td>YOLO26s</td><td>9.5M</td><td>95.5</td><td>66.1</td><td>92.6</td><td>50.5</td><td>100.0</td><td>100.0</td><td>75.5</td><td>38.7</td></tr><tr><td>RT-DETRv2</td><td>20.2M</td><td>94.2</td><td>63.0</td><td>92.6</td><td>53.8</td><td>100.0</td><td>75.6</td><td>65.8</td><td>80.4</td></tr><tr><td colspan="10">Open-vocabulary detectors</td></tr><tr><td>YOLO-World</td><td>76.2M</td><td>96.1</td><td>25.6</td><td>97.8</td><td>15.0</td><td>100.0</td><td>2.5</td><td>75.2</td><td>25.3</td></tr><tr><td>GDINO-Tiny</td><td>172.3M</td><td>97.4</td><td>64.2</td><td>97.8</td><td>56.1</td><td>100.0</td><td>85.4</td><td>89.8</td><td>62.1</td></tr><tr><td>GDINO-Base</td><td>232.8M</td><td>100.0</td><td>60.0</td><td>97.8</td><td>47.7</td><td>100.0</td><td>58.5</td><td>90.7</td><td>40.2</td></tr><tr><td colspan="10">Vision-language models</td></tr><tr><td>MiniCPM-V 4.6</td><td>1.3 B</td><td>94.8</td><td>10.1</td><td>83.0</td><td>10.3</td><td>95.1</td><td>2.5</td><td>51.1</td><td>9.6</td></tr><tr><td>InternVL3-2B</td><td>2.1 B</td><td>98.7</td><td>59.2</td><td>91.1</td><td>41.5</td><td>97.6</td><td>100.0</td><td>27.6</td><td>35.4</td></tr><tr><td>Qwen3-VL-2B</td><td>2.1 B</td><td>99.4</td><td>22.2</td><td>94.8</td><td>22.6</td><td>100.0</td><td>7.5</td><td>90.3</td><td>39.7</td></tr><tr><td>PaliGemma 2-3B</td><td>3.0 B</td><td>97.4</td><td>59.3</td><td>94.1</td><td>37.6</td><td>100.0</td><td>97.6</td><td>86.7</td><td>15.4</td></tr><tr><td>LFM2.5-VL-3B</td><td>3.1 B</td><td>96.1</td><td>16.4</td><td>91.1</td><td>11.4</td><td>100.0</td><td>14.6</td><td>87.8</td><td>5.8</td></tr></table>

Table 2: Failure partition into omission (O), substitution (S), and displacement (D).
<table><tr><td rowspan="2">Model</td><td rowspan="2">Grammar</td><td colspan="4">In-distribution</td><td colspan="4">Waymo (out-of-distribution)</td></tr><tr><td>cASR</td><td>0</td><td>S</td><td>D</td><td>cASR</td><td>0</td><td>S</td><td>D</td></tr><tr><td>Closed-set detectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8"></td></tr><tr><td>YOL011s</td><td>score-gated</td><td>78.3</td><td>72.8</td><td>0.0</td><td>5.5</td><td>86.1</td><td>81.5</td><td>0.0</td><td>4.6</td></tr><tr><td>YOLO26s</td><td>score-gated</td><td>65.3</td><td>62.2</td><td>0.0</td><td>3.1</td><td>38.7</td><td>31.0</td><td>0.0</td><td>7.6</td></tr><tr><td>RT-DETRv2</td><td>score-gated</td><td>61.9</td><td>53.5</td><td>0.0</td><td>8.4</td><td>80.4</td><td>61.2</td><td>0.5</td><td>18.7</td></tr><tr><td colspan="2">Open-vocabulary detectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YOLO-World</td><td>text-match</td><td>18.2</td><td>15.6</td><td>0.7</td><td>1.8</td><td>25.3</td><td>22.4</td><td>0.3</td><td>2.6</td></tr><tr><td>GDINO-Tiny</td><td>text-match</td><td>63.6</td><td>49.4</td><td>3.4</td><td>10.8</td><td>62.1</td><td>8.8</td><td>16.6</td><td>36.7</td></tr><tr><td>GDINO-Base</td><td>text-match</td><td>54.9</td><td>29.9</td><td>10.7</td><td>14.3</td><td>40.2</td><td>1.4</td><td>18.0</td><td>20.9</td></tr><tr><td colspan="2">Vision-language models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MiniCPM-V 4.6</td><td>free-text</td><td>9.2</td><td>0.7</td><td>0.3</td><td>8.1</td><td>9.6</td><td>0.0</td><td>0.3</td><td>9.3</td></tr><tr><td>InternVL3-2B</td><td>gated</td><td>57.7</td><td>56.7</td><td>0.0</td><td>1.0</td><td>35.4</td><td>4.6</td><td>1.3</td><td>29.5</td></tr><tr><td>Qwen3-VL-2B</td><td>free-text</td><td>20.1</td><td>0.0</td><td>1.5</td><td>18.7</td><td>39.7</td><td>0.0</td><td>0.9</td><td>38.8</td></tr><tr><td>PaliGemma 2-3B</td><td>gated</td><td>55.2</td><td>53.4</td><td>0.0</td><td>1.9</td><td>15.4</td><td>8.9</td><td>0.1</td><td>6.3</td></tr><tr><td>LFM2.5-VL-3B</td><td>free-text</td><td>14.3</td><td>0.0</td><td>5.2</td><td>9.1</td><td>5.8</td><td>0.0</td><td>1.1</td><td>4.7</td></tr></table>

The difference between the most and least affected models therefore reflects what the attack can suppress. The attack drives some quantity down (a detector’s confidence, a VLM’s token likelihood), which removes the box only when a single token gates the box’s existence (Eq. 4). Producing no box guarantees a failure, whereas perturbing coordinates only constitutes a failure if the box moves far enough to fall below τ. InternVL3-2B and PaliGemma 2-3B open their output with such a gating token, so driving down its likelihood produces an omission; they are the two most affected VLMs we test. A free-text model has no such token. It writes coordinates as ordinary text that the decoder produces whether or not the target is visible, so the attack cannot delete the box, only mislocalize it. The free-text VLMs (LFM2.5-VL-3B, MiniCPM-V 4.6, and Qwen3-VL-2B) are also the three least affected as measured by cASR.

## 5.2 Failure modes

Table 2 partitions each model’s failures into omission, substitution, and displacement (Section 4.4).

Omission and displacement. The two VLM grammars occupy opposite extremes of the failure mode partition. The gated VLMs, PaliGemma 2-3B and InternVL3-2B, fail almost entirely by omission: 96.7% and 98.3% of their failures produce no box, a higher share than any detector we test. The score-gated detectors behave the same way, with the YOLO models failing by omission on 93% to 95% of failures. At the other end, the free-text VLMs Qwen3-VL-2B and LFM2.5-VL-3B record exactly zero omissions at every threshold we sweep, and MiniCPM-V 4.6 fewer than 1%. What these models produce instead is displacement, a box that parses correctly but lands on neither the person nor the patch. Notably, Qwen3-VL-2B and LFM2.5-VL-3B share a grammar yet fail at different rates and split differently between displacement and substitution (displacement accounts for 18.7% of Qwen3-VL-2B’s 20.1% and 9.1% of LFM2.5-VL-3B’s 14.3%). Therefore, under a shared suppression objective that lowers the likelihood of the target box without targeting an empty response, output format is associated with whether a box is emitted, but not how often or where it lands. Alternative prompts can still shift this balance, as MiniCPM-V 4.6 omits on 7–9% of attacked images under alternative phrasings (Table 3).

Because grammar fixes emission but not placement, a stricter overlap criterion should affect only the models that emit a box. Raising the IoU threshold (τ ) from 0.5 to 0.9 (Figure 5 in Appendix A), the free-text models’ failure rates climb sharply while the gated models’ barely move. Qwen3-VL-2B’s cASR reaches 95.6%—the largest aggregated in-distribution cASR any model reaches in the benchmark—while its omission holds at exactly zero throughout, the entire increase being fully attributable to boxes it still emits. PaliGemma 2-3B and InternVL3-2B only move from 55.2% to 68.6% and 57.7% to 62.8%, remaining omission-dominated at every point. Thus raising τ does not affect omissions: with no box emitted, they fail at every threshold regardless. It instead converts displaced boxes into failures as they stop clearing the threshold.

Substitution. A box that lands on the patch itself rather than the person similarly occurs in models whose emission is not gated on the target’s presence. Every closed-set detector and both gated VLMs (PaliGemma 2-3B, InternVL3-2B) register exactly zero. Every other model registers a nonzero rate: the region-text matchers (GDINO-Base 10.7%, GDINO-Tiny 3.4%, YOLO-World 0.7%) and the free-text VLMs (LFM2.5-VL-3B 5.2%, Qwen3-VL-2B 1.5%, MiniCPM-V 4.6 0.3%). Substitution cuts across architectures and arises in the models where neither a confidence threshold nor a gating token withholds the box.

In deployment, substitution and displacement carry worse consequences than omission because both emit a box. A downstream system can detect an omission, since a model that returns nothing signals its own failure. Substitution and displacement, on the other hand, deliver a confident, well-formed box at an incorrect location, which may be indistinguishable from a correct detection. Substitution is arguably the more severe of the two. It inverts the attack, supplying a stronger signal that the model reports as the person rather than suppressing the person’s own, so the system’s attention is anchored to an object the attacker carries rather than to empty space. Its box also takes the patch’s size, no larger than the person’s at every scale we evaluate, so any spatial feature a system infers from the bounding box (such as distance) is systematically distorted. A single success rate from prior work collapses these distinctions, directly motivating our proposed failure partition.

## 5.3 Under distribution shift

The Waymo set holds every model to the same task on a domain none of the patches were trained against, approximating the distribution shift of deployment. cASR moves in both directions, so the distribution shift is not a uniform increase in difficulty. What changes is the failure mode. Indistribution, InternVL3-2B, GDINO-Base, and GDINO-Tiny fail predominantly by omission. On the Waymo set, their omission rate drops sharply (InternVL3-2B from 56.7% to 4.6%, GDINO-Base from 29.9% to 1.4%, and GDINO-Tiny from 49.4% to 8.8%) while their overall failure rates fall much less in comparison. Their remaining failures are instead registering as displacement or substitution, the modes a downstream system cannot detect. InternVL3-2B’s displacement rises from 1.0% to 29.5%, GDINO-Tiny’s substitution from 3.4% to 16.6% and its displacement from 10.8% to 36.7%, and GDINO-Base’s substitution from 10.7% to 18.0%. Counterintuitively, InternVL3-2B’s cASR decreases (57.7% to 35.4%), yet most of its remaining failures are now undetectable.

The closed-set detectors exhibit this shift far less: YOLO11s, YOLO26s, and RT-DETRv2 all remain omission-dominated on Waymo. In contrast, the open-vocabulary detectors and grounding VLMs do lose detectability. The one exception is YOLO-World, the most robust among them, which stays omission-dominated throughout.

Further, gated VLMs lose a key safety property under the distribution shift: in-distribution, InternVL3- 2B and PaliGemma 2-3B fail by omission on 98.3% and 96.7% of their failures. On Waymo, PaliGemma 2-3B’s omission falls from 53.4% to 8.9% and InternVL3-2B’s nearly vanishes. Among the grounding VLMs, these two models are the only ones that predominantly fail visibly by omission, yet even that largely disappears under the distribution shift—highlighting that grounding VLMs can be as vulnerable as, or more vulnerable than, detectors in deployment settings.

## 6 Limitations

Our adversarial attacks are digitally simulated, and are not evaluated in physical settings. The patches are optimized to be printable and pose-robust, which is what “physical” refers to throughout; no printed patch or camera capture was evaluated, so we do not measure physical-world transfer. In addition, we only utilize white-box attack methods, which assume full access to the model and do not reflect real-world deployment scenarios. Future work should focus on evaluating the adversarial effect of physically printed patches and test black-box methods. Furthermore, we test only a suppression objective, and future work should evaluate persistence of the failure-mode partitions under different attack objectives.

## 7 Conclusion

Under one attack framework, one occlusion control, and a single shared predicate, edge grounding VLMs can be as vulnerable as the classic detectors, yet they do not fail in the same way: the form of each failure is largely associated with the model’s output format rather than its architecture family. This difference carries the deployment consequence. A patch trained to suppress the model’s detection of the person produces no box in some models and a well-formed, incorrectly placed box in others. Only the first is visible to a downstream system, while the second may be indistinguishable from a correct detection. Any grounding model bound for deployment should therefore be evaluated on two axes: how often it fails under physical attack, and whether those failures are visible in its output. The first requires an occlusion-controlled measurement, which our benchmark provides. Therefore, as evidenced by our controlled comparison, the shift from detectors to grounding VLMs does not reduce a system’s vulnerability to physical attack. Under the distribution shift of real-world deployment, it may render the system even more vulnerable—failing in modes that are invisible to the downstream systems relying on it.

## Acknowledgments and Disclosure of Funding

We thank our reviewers for helpful discussions. This work received no external funding.

## References

[1] Mazen Abdelfattah, Kaiwen Yuan, Z Jane Wang, and Rabab Ward. Adversarial attacks on camera-lidar models for 3d car detection. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2189–2194. IEEE, 2021.

[2] Mazen Abdelfattah, Kaiwen Yuan, Z Jane Wang, and Rabab Ward. Towards universal physical attacks on cascaded camera-lidar 3d object detection models. In 2021 IEEE international conference on image processing (ICIP), pages 3592–3596. IEEE, 2021.

[3] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In International conference on machine learning, pages 284–293. PMLR, 2018.

[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[6] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017.

[7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

[8] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16901–16911. IEEE, 2024.

[9] Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, and Marie-Francine Moens. Talk2Car: Taking control of your self-driving car. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2088–2098. Association for Computational Linguistics, 2019.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[11] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. International Journal of Computer Vision, 88(2):303–338, 2010. doi: 10.1007/s11263-009-0275-4.

[12] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.

[13] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, 2021. doi: 10.23915/distill.00030.

[14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 15979–15988. IEEE, 2022.

[15] Glenn Jocher and Jing Qiu. Ultralytics YOLO26: Unified real-time end-to-end vision models. arXiv preprint arXiv:2606.03748, 2026.

[16] Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024.

[17] Junxian Li, Beining Xu, Simin Chen, Jiatong Li, Jingdi Lei, Haodong Zhao, and Di Zhang. Iag: Input-aware backdoor attack on vlm-based visual grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27872–27883, 2026.

[18] Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, and Shaohui Mei. Padetbench: Towards benchmarking physical attacks against object detection. arXiv preprint arXiv:2408.09181, 2024.

[19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In Computer Vision – ECCV 2014, volume 8693 of Lecture Notes in Computer Science, pages 740–755. Springer, 2014. doi: 10.1007/978-3-319-10602-1\_48.

[20] Liquid AI. LFM2.5-VL-3B. Model card, Hugging Face, 2026. https://huggingface.co/ LiquidAI/LFM2.5-VL-3B.

[21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[22] Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C Kot, and Xudong Jiang. When robots obey the patch: Universal transferable patch attacks on visionlanguage-action models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22867–22878, 2026.

[23] Alan Luo and Kaiwen Yuan. Simple self-organizing map with vision transformers. IEEE Signal Processing Letters, 33:331–335, 2026. doi: 10.1109/LSP.2025.3643388.

[24] Haochen Luo, Jindong Gu, Fengyuan Liu, and Philip Torr. An image is worth 1000 lies: Transferability of adversarial images across prompts on vision-language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=nc5GgFAvtk.

[25] Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, and Yi Liu. RT-DETRv2: Improved baseline with bag-of-freebies for real-time detection transformer. arXiv preprint arXiv:2407.17140, 2024.

[26] Ankita Raj and Chetan Arora. Backdoor attacks on open vocabulary object detectors via multi-modal prompt tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 37849–37857, 2026.

[27] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016.

[28] Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K. Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In Proceedings ofthe 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1528–1540. ACM, 2016. doi: 10.1145/2976749.2978392.

[29] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. CLIPort: What and where pathways for robotic manipulation. In Proceedings ofthe 5th Conference on Robot Learning (CoRL), volume 164 of Proceedings ofMachine Learning Research, pages 894–906. PMLR, 2021.

[30] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

[31] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2443–2451. IEEE, 2020.

[32] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[33] Simen Thys, Wiebe Van Ranst, and Toon Goedemé. Fooling automated surveillance cameras: adversarial patches to attack person detection. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 49–55. IEEE, 2019.

[34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[35] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. Universal adversarial triggers for attacking and analyzing nlp. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2153–2162, 2019.

[36] Taowen Wang, Cheng Han, James Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, and Ruixiang Tang. Exploring the adversarial vulnerabilities of visionlanguage-action models in robotics. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6948–6958. IEEE, 2025.

[37] Zhixiang Wang, Xiaosen Wang, Bo Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, and Yu-Gang Jiang. Diffpatch: Generating customizable adversarial patches using diffusion models. arXiv preprint arXiv:2412.01440, 2024.

[38] Cihang Xie, Jianyu Wang, Zhishuai Zhang, Yuyin Zhou, Lingxi Xie, and Alan Yuille. Adversarial examples for semantic segmentation and object detection. In 2017 IEEE international conference on computer vision (ICCV), pages 1378–1387. IEEE, 2017.

[39] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Ranchi Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11704–11715, 2026.

[40] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.

[41] Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, and Haian Huang. An open and comprehensive pipeline for unified object grounding and detection. arXiv preprint arXiv:2401.02361, 2024.

[42] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, and Min Lin. On evaluating adversarial robustness of large vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[43] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[44] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6725–6735. IEEE, 2023.

[45] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A Ablation studies

We report four ablations, each varying a single parameter of the attack while holding the rest of the pipeline constant: patch scale, overlap threshold, VLM prompt, and detector confidence threshold. All four are evaluated on the 331 validation images at the canonical settings, conditioned on the gray-control set.

## A.1 Patch size ablation

Figure 4 sweeps the patch scale $a \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ . cASR generally rises with scale, though the increase is not monotonic. The ordering of vulnerability is broadly preserved across the sweep, meaning the adversarial patches are robust to scale and reflect vulnerability rather than the choice of that parameter.

![](images/cd81dcd669ec4db9b5a64a4403592a27b000621c6cd33f37b07abe695b6f9f5a.jpg)  
Figure 4: Conditional attack success versus patch scale.

## A.2 IoU threshold ablation

Figure 5 sweeps the overlap threshold τ from 0.5 to 0.9. The size of the increase separates models by failure mode rather than family. Omission-dominated models show comparatively little change, because a box that is never emitted fails at every threshold. Models that always emit a box accumulate failures as the requirement tightens, with Qwen3-VL-2B rising from 20.1% to 95.6%.

![](images/7017ed62850badc89b0f76b8a0c42b024ff7929a96dcf9643c97a7c96b5cf0b9.jpg)  
Figure 5: Conditional attack success versus IoU threshold.

## A.3 VLM prompt ablation

Table 3 shows prompt sensitivity, re-evaluating each VLM patch under its training prompt and three paraphrases with the patch itself held fixed. The failure rate varies by prompt, so a single prompt should not be read as a model’s robustness. However, the attacks mostly still generalize across various

different prompts, emphasizing deployment vulnerability. The failure partition remains stable: the gated grammars remain omission-dominated under all four prompts, and Qwen3-VL-2B records exactly zero omissions under every prompt we test.

Table 3: Conditional attack success and failure partition across prompts.
<table><tr><td>Prompt</td><td>cASR</td><td>Omission</td><td>Substitution</td><td>Displacement</td></tr><tr><td>PaliGemma 2-3B (gated)</td><td></td><td></td><td></td><td></td></tr><tr><td>detect all people (TRAINED)</td><td>55.2</td><td>53.4</td><td>0.0</td><td>1.9</td></tr><tr><td>detect all pedestrians</td><td>57.0</td><td>55.0</td><td>0.0</td><td>2.0</td></tr><tr><td>detect all humans</td><td>43.4</td><td>39.7</td><td>0.0</td><td>3.7</td></tr><tr><td>detect all persons</td><td>39.2</td><td>38.0</td><td>0.0</td><td>1.2</td></tr><tr><td>InternVL3-2B (gated) Please provide the bounding box</td><td>57.7</td><td>56.7</td><td>0.0</td><td>1.0</td></tr><tr><td>coordinate of the region this sentence describes:</td><td></td><td></td><td></td><td></td></tr><tr><td>&lt;ref&gt;person&lt;/ref&gt; (TRAINED) Please detect the person and return the bounding box.</td><td>29.0</td><td>27.4</td><td>0.0</td><td>1.7</td></tr><tr><td>Where is the person? Give the bounding</td><td>45.9</td><td>39.3</td><td>0.0</td><td>6.7</td></tr><tr><td>box coordinates. Locate the person and provide the bounding box coordinates.</td><td>28.0</td><td>25.2</td><td>0.0</td><td>2.8</td></tr><tr><td>MiniCPM-V 4.6 (free-text) Locate the person. Output ONLY the bounding box [x1, y1, x2, y2] in 0–1000</td><td>9.2</td><td>0.7</td><td>0.3</td><td>8.1</td></tr><tr><td>normalized coordinates, nothing else. (TRAINED) Detect the person and output the bounding</td><td>19.7</td><td>7.5</td><td></td><td></td></tr><tr><td>box [x1, y1, x2, y2] in 0–1000 coordinates. Identify the person and give the bounding</td><td>42.7</td><td>9.3</td><td>0.4</td><td>11.8</td></tr><tr><td>box [x1, y1, x2, y2] in 0–1000 coordinates. Find the person and output the bounding</td><td>30.1</td><td>7.3</td><td>0.0</td><td>33.3</td></tr><tr><td>box [x1, y1, x2, y2] in 0–1000 coordinates. Qwen3-VL-2B (free-text)</td><td></td><td></td><td>1.2</td><td>21.6</td></tr><tr><td>Locate every person in the image and output the bounding box coordinates as JSON in the form {“bbox_2d&quot;: [x1, y1, x2,</td><td>20.1</td><td>0.0</td><td>1.5</td><td>18.7</td></tr><tr><td>y2], &quot;label&quot;: &quot;person&quot;}. (TRAINED) Locate the person and return {“bbox_2d&quot;:</td><td>17.2</td><td></td><td></td><td></td></tr><tr><td>[x1, y1, x2, y2], &quot;label&quot;: “&quot;person&quot;}. Where is the person? Output {“bbox_2d&quot;:</td><td>17.1</td><td>0.0 0.0</td><td>1.1</td><td>16.1</td></tr><tr><td>[x1, y1, x2, y2], &quot;label&quot;: “&quot;person&quot;}. Detect the person and output the bounding</td><td>17.9</td><td>0.0</td><td>3.5 1.5</td><td>13.6 16.3</td></tr><tr><td>box as JSON {“bbox_2d&quot;: [x1, y1, x2, y2], &quot;label&quot;: “person&quot;}. LFM2.5-VL-3B (free-text)</td><td></td><td></td><td></td><td></td></tr><tr><td>Please provide the bounding box coordinate of the region this sentence</td><td>14.3</td><td>0.0</td><td>5.2</td><td>9.1</td></tr><tr><td>describes: the person. (TRAINED) Locate the person in the image and output</td><td>1.6</td><td>0.0</td><td>0.0</td><td>1.6</td></tr><tr><td>the bounding box. Detect the person and output the bounding</td><td>2.3</td><td>0.3</td><td>0.0</td><td>2.0</td></tr><tr><td>box [x1, y1, x2, y2].</td><td>1.3</td><td>0.0</td><td>0.0</td><td></td></tr><tr><td>Where is the person? Give the bounding box [x1, y1, x2, y2].</td><td></td><td></td><td></td><td>1.3</td></tr></table>

## A.4 Detector confidence threshold ablation

Table 4 varies the confidence threshold γ at which each detector is read. As the confidence requirement increases, attack success increases for every model, suggesting that tasks that require more precision become more vulnerable.

Table 4: Conditional attack success versus detector confidence threshold.
<table><tr><td colspan="4">Detector cASR (%)</td></tr><tr><td colspan="4">Closed-set detectors</td></tr><tr><td>γ</td><td>0.1</td><td>0.25</td><td>0.5 0.7</td></tr><tr><td>YOLO11s</td><td>67.2</td><td>73.1 78.3</td><td>83.8</td></tr><tr><td>YOLO26s</td><td>40.7</td><td>56.7 65.3</td><td>67.9</td></tr><tr><td>RT-DETRv2</td><td>34.7 49.8</td><td>61.9</td><td>68.6</td></tr><tr><td colspan="4">Open-vocabulary detectors</td></tr><tr><td>γ</td><td>0.1</td><td>0.2 0.3</td><td>0.4</td></tr><tr><td>YOLO-World</td><td>11.1</td><td>16.8 18.2</td><td>23.1</td></tr><tr><td>GDINO-Tiny</td><td>4.5</td><td>29.3 63.6</td><td>81.2</td></tr><tr><td>GDINO-Base</td><td>3.6 22.5</td><td>54.9</td><td>77.4</td></tr></table>

## B Dataset construction

Table 5 reports the filtering counts for each source dataset. From COCO we only take val2017, to prevent data leakage as some models are trained on train2017; from PASCAL VOC, the val sets of 2007 and 2012; from AdvT-Shirt, its 1-person folder; and from Waymo, the training split of v2 perception. Each dataset was first filtered to one-person images using its ground-truth bounding box labels. We then manually removed those containing extra unannotated people, and dropped Waymo’s night-time images to ensure proper visibility. Figure 6 shows example images from the final version of each dataset.

Table 5: Per-dataset filtering counts.
<table><tr><td>Source dataset</td><td>Original</td><td>One-person pool</td><td>Kept</td><td>Kept (%)</td></tr><tr><td>COCO val2017</td><td>5,000</td><td>925</td><td>777</td><td>15.5</td></tr><tr><td>PASCAL VOC 2007+2012 val</td><td>8,333</td><td>987</td><td>674</td><td>8.1</td></tr><tr><td>AdvT-Shirt-1K, 1-person subset</td><td>208</td><td>208</td><td>204</td><td>98.1</td></tr><tr><td>Waymo v2 perception, training split</td><td>4,662</td><td>1,014</td><td>999</td><td>21.4</td></tr></table>

![](images/74cf0fe1c0f5e1382b8129325f3618c271af10c41b7c7117b879360fc6a34533.jpg)  
Figure 6: Example images from each source dataset.