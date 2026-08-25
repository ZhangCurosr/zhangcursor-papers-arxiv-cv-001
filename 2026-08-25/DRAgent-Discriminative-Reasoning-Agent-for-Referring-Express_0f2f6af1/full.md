# DRAgent: Discriminative Reasoning Agent for Referring Expression Segmentation

Yujie Qi<sup>1</sup> and Luyan Zhang<sup>2,∗</sup>

Abstract— Referring Expression Segmentation (RES) aims to generate a pixel-level mask for the object specified by a language expression. Recent methods based on multimodal large language models (MLLMs) often rely on one-pass coordinate prediction for visual localization, which serializes continuous spatial locations as discrete text tokens and may lead to localization bias and alignment errors. To address these issues, we propose DRAgent, an MLLM-driven discriminative reasoning (DR) framework for RES. Instead of requiring the MLLM to generate localization coordinates, DRAgent first constructs a detector-generated candidate space and then uses the MLLM as a visual-semantic target discriminator. Specifically, the MLLM performs reliable target selection among potential distractors through a two-stage DR mechanism, which first screens highrecall candidates and then performs instance-wise verification. The selected target box is subsequently used as a spatial prompt for a foundation segmentation model to produce the final pixel-level mask. Furthermore, we construct a self-consistencyfiltered reasoning-chain data pipeline for LoRA-based finetuning, providing more reliable supervision for enhancing the MLLM’s discriminative reasoning capability. Experiments demonstrate that DRAgent achieves competitive performance on RefCOCO, RefCOCO+, and RefCOCOg.

## I. INTRODUCTION

Referring Expression Segmentation (RES) aims to generate a pixel-level mask for the object specified by a natural language expression [1], [2]. In autonomous systems such as industrial or assistive robotics, it serves as a critical bridge between perception and control, where precise identification is the prerequisite for subsequent task execution.

Recent methods increasingly combine multimodal large language models (MLLMs) [3], [4], [5] with foundation segmentation models such as SAM [6]. In these methods, MLLMs generate visual localization as spatial prompts for mask generation, exploiting the high-level reasoning and semantic understanding capabilities of MLLMs.

However, these methods still face two major limitations. First, when MLLMs generate spatial prompts [3], [4], [5] for the referred target in an autoregressive manner, coordinates are serialized into discrete text tokens. Such a representation fails to preserve the continuity of coordinates and their underlying spatial relationships, instead treating them as independent symbols, thereby increasing the risk of coarse or biased localization. Moreover, in complex scenes with densely distributed objects or highly similar appearances, one-pass methods that rely on MLLMs to localize the referent directly are prone to confusion with distractors. Such localization bias and alignment errors directly propagate incorrect spatial prompts to the downstream segmentation task, thereby degrading segmentation performance.

![](images/d8801a50adc200b6c1e3cf5c79b08828d981c574e97e780b9cada0992ef0d3bf.jpg)  
Fig. 1. Generative visual localization by MLLMs is prone to localization errors and may propagate erroneous prompts to the downstream segmentation module. In contrast, discriminative reasoning mitigates this issue by selecting among detector-generated candidates.

Second, some methods introduce reasoning chains, such as Chain-of-Thought (CoT) [7] and multi-step iterative trajectories [4], and use them to supervise the training of MLLMs, aiming to enhance their interpretability and reasoning capability in visual localization. However, these methods rely on high-quality reasoning-chain data, while valid reasoning paths for visual-semantic alignment are not unique [8]. Existing approaches often lack an effective mechanism for constructing and validating reasoning chains used for model training. Current methods typically obtain reasoning chains either through manual annotation or generation by powerful external MLLMs. The former is costly and difficult to scale, whereas the latter tends to produce logically inconsistent reasoning, which may cause the model to learn unstable or erroneous reasoning patterns.

Regarding these issues, we argue that a key limitation of many recent MLLM-based RES methods lies in treating visual localization as a generative coordinate prediction problem. RES can be better addressed by introducing a discriminative target selector, where the MLLM selects the referred object from a candidate space through screening and instance-wise verification, rather than generating localization coordinates in a one-pass manner. Moreover, reasoning-chain data used for training should not be accepted solely based on its surface plausibility. Instead, closed-loop consistency should be introduced as a reliability signal.

From this perspective, we propose DRAgent, an MLLMdriven discriminative target selection framework for RES. Rather than asking the MLLM to generate localization coordinates directly, DRAgent uses it as a visual-semantic discriminator over a detector-generated candidate space. An open-vocabulary detector [9] first constructs a high-recall candidate set, which serves as the decision space for the MLLM. The MLLM then performs two-stage discriminative reasoning (DR), consisting of candidate screening and instance-wise visual verification, to identify the candidate that best matches the referring expression. The selected bounding box is finally fed into SAM [6] as a spatial prompt to generate the pixel-level mask. To further strengthen the MLLM’s DR capability, we construct a self-consistencyfiltered reasoning-chain data pipeline and use the resulting data for LoRA-based [10] supervised fine-tuning (SFT).

In summary, our contributions are three-fold.

• We propose DRAgent, a RES framework that reformulates visual localization from coordinate generation into candidate-space DR, where the MLLM serves as a target discriminator over detector-generated candidates.

• We introduce a two-stage DR mechanism and a self consistency-filtered reasoning-chain data pipeline, improving MLLM-driven target discrimination from the inference and training perspectives, respectively.

• Extensive experiments demonstrate that DRAgent achieves competitive performance on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks.

## II. RELATED WORK

## A. MLLMs for RES

Early RES studies explored task-specific architectures for aligning referring expressions with target regions. LAVT [1] injects linguistic features into the visual backbone through a language-aware vision transformer. ReLA [2] strengthens relation-aware cross-modal modeling, while PolyFormer [11] formulates referring image segmentation as sequential polygon generation. However, they still struggle with fine-grained expression-region alignment in complex scenes.

To address these limitations, recent studies incorporate MLLMs into end-to-end RES frameworks to enhance openvocabulary semantic understanding and cross-modal mod eling capabilities. For example, LISA [12] connects an MLLM with SAM through the special <SEG> token, whose hidden-state embedding serves as a prompt to guide mask generation. GSVA [13] and OpenWorldSAM [14] employ an MLLM as a multimodal feature encoder to provide semantically fused representations for the mask decoder. These methods have substantially improved cross-modal alignment. However, most of them primarily treat MLLMs as feature encoders [15], [16] and do not fully exploit their strong reasoning capabilities.

## B. MLLMs as Visual Localization Generators in RES

Some recent works have begun to use MLLMs as visual localization generators, producing bounding boxes, point prompts, or combinations thereof to guide external segmentation models for target mask generation. For instance, SAM-Veteran [5] uses multi-round reasoning with an MLLM to generate and update point- and box-based coordinate prompts, which drive external segmentation tools to progressively refine the mask. In SegAgent [4], the MLLM reads the current mask state together with the textual description and iteratively generates positive and negative point coordinates to simulate human collaboration with interactive segmentation tools, thereby refining the segmentation result step by step. Although such methods better exploit the explicit reasoning ability of MLLMs, generative visual localization for the referent [3], [17] remains prone to localization bias and alignment errors, resulting in unreliable spatial prompts.

In contrast to visual localization generator methods, DRAgent reformulates the role of the MLLM as a target discriminator. Candidate proposals are first produced by an open-vocabulary detector, while the MLLM is responsible only for discriminative selection over the candidate space. Specifically, DRAgent introduces a two-stage DR mechanism that combines candidate screening with instance-wise visual verification, enabling the MLLM to judge visual-semantic consistency rather than generate coordinates directly. This design decouples candidate generation from target discrimination and shifts the core MLLM function from spatial prompt prediction to reliable target verification.

## III. METHOD

DRAgent decomposes the RES task into three stages: candidate object detection, discriminative target selection, and pixel-level segmentation. Within this pipeline, the MLLM performs robust target selection through a two-stage DR mechanism, and its discriminative capability is further enhanced through fine-tuning with self-consistency-filtered reasoning-chain data.

## A. Overview of DRAgent

We formulate the task pipeline of DRAgent as a multistage composite function. Given an input image I and a text query Q, the goal is to generate a pixel-level mask M. Candidate Object Detection. This stage constructs a highrecall discrete candidate space that serves as the decision space for subsequent discriminative target selection. Specifically, we first use an LLM to extract the target category from the text query Q, denoted as $ { \mathcal { L } } { \mathrm { ~  ~ { ~ \mathcal ~ { ~ L ~ } ~ } ~ } } = \mathrm { L L M } (  { \mathcal { Q } } )$ . We then feed it into an open-vocabulary detector, instantiated as Grounding DINO [9] in this work, to obtain a candidate set $\mathcal { O } = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { n } \}$ , where each candidate $o _ { i }$ is represented by a bounding box $[ x _ { 1 } ^ { i } , y _ { 1 } ^ { i } , x _ { 2 } ^ { i } , y _ { 2 } ^ { i } ]$

Discriminative Target Selection. This module uses the MLLM as a target discriminator to select the candidate $o _ { t }$ from O that best matches Q. The selection is performed through the two-stage DR mechanism detailed in Section III-B, rather than through direct coordinate generation.

Pixel-Level Segmentation. After selecting the target $o _ { t }$ referred to by the text query Q, a pixel-level segmentation model, instantiated as SAM [6] in this work, is applied to generate the corresponding mask M.

Overall Pipeline Formulation. Taken together, the entire task can be represented as the following composite mapping:

$$
\mathcal { M } = f _ { \mathrm { s e g } } \big ( \mathcal { T } , \Phi _ { \mathrm { s e l e c t } } ( \mathcal { T } , \mathcal { Q } , g _ { \mathrm { d e t } } ( \mathcal { T } , \mathrm { L L M } ( \mathcal { Q } ) ) ) \big ) ,\tag{1}
$$

![](images/9bcf4879e0655bc65e3456751566e804193a874f172a3cc7fd12cf88c6e5748a.jpg)  
Fig. 2. (a) DRAgent divides RES into three stages: candidate object detection, discriminative target selection, and pixel-level segmentation. (b) Two-stage DR mechanism for the discriminative target selection module. (c) Reasoning-chain data pipeline based on self-consistency filtering for SFT.

where $g _ { \mathrm { d e t } }$ denotes the open-vocabulary detector, $\Phi _ { \mathrm { s e l e c t } }$ denotes the core discriminative target selection module, and $f _ { \mathrm { s e g } }$ maps the final decision back to the pixel space.

## B. Two-Stage Target Selection

Although the detector provides a candidate space for the MLLM, directly asking the MLLM to select the target from all candidates in one pass is prone to confusion with distractors in complex scenes. To make the MLLM’s decision process more stable, we formulate target selection as a cascaded two-stage DR mechanism: the MLLM first performs candidate screening to reduce the search space, and then conducts instance-wise verification to identify the referred target through finer visual-semantic comparison.

Candidate-Level Preliminary Screening. To reduce the search space while preserving candidate recall, we first visualize detected candidates on I with indexed Set-of-Mark (SoM) [18] bounding boxes. Based on this candidate-marked image, the MLLM selects the top-K candidates from O that best match Q. Because dense object layouts can impair the MLLM’s judgment, we further supplement these top-K candidates with candidates whose Intersection over Union (IoU) with the selected boxes exceeds a threshold γ, yielding the second-stage candidate set $\mathcal { O } _ { \mathrm { t o p } }$

Discriminative Verification via Verification Score. To further reduce ambiguity among retained candidates, the second stage performs instance-wise visual verification. Each candidate $o _ { i } ~ \in ~ \mathcal { O } \mathrm { t o p }$ is rendered with a SoM bounding box, denoted as ISoM<sup>i</sup>. We design a task-specific instruction prompt, $\mathrm { P r o m p t _ { D R } } ( { \mathcal { Q } } )$ , to elicit a structured visual response:

“Look at the red bounding box. Does it tightly enclose the object described as ‘{Q}’? Answer yes or no.”

We perform candidate verification through vision-based discriminative reasoning, leveraging the visual question answering capability of the MLLM to determine from $\mathcal { T } _ { \mathrm { S o M } } ^ { i }$ whether the candidate corresponds to the queried referent. This alleviates the mismatch between textual coordinate decoding and visual-spatial grounding by constructing an alignment score directly from visual evidence.

Concretely, let V denote the output vocabulary of the MLLM. Given the multimodal input $\begin{array} { r l } { \mathcal { X } _ { i } } & { { } = } \end{array}$ $\{ \mathcal { T } _ { \mathrm { S o M } } ^ { i } , \mathrm { P r o m p t } _ { \mathrm { D R } } ( \mathcal { Q } ) \}$ , conventional generative decoding makes a direct decision based on the output:

$$
\hat { v } _ { i } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } } \mathbb { P } _ { \theta } ( y _ { 1 } = v \mid \mathcal { X } _ { i } )\tag{2}
$$

where $y _ { 1 }$ denotes the first generated answer token. This strategy is easily affected by an ambiguous probability distribution over answer tokens. To obtain a more stable and discriminative verification signal, we define the verification score of each candidate based on the relative strength between the positive and negative answer token sets:

$$
S _ { i } = \frac { \exp ( z _ { i , \mathcal { V } _ { \mathrm { p o s } } } ) } { \exp ( z _ { i , \mathcal { V } _ { \mathrm { p o s } } } ) + \exp ( z _ { i , \mathcal { V } _ { \mathrm { n e g } } } ) } .\tag{3}
$$

where $\mathcal { V } _ { \mathrm { p o s } }$ and $\mathcal { V } _ { \mathrm { n e g } }$ contain the token IDs corresponding to positive answer variants $( ^ { 6 6 } \mathrm { Y e s } ^ { 3 9 } , ^ { 6 6 } \mathrm { y e s } ^ { 3 9 }$ , and “YES”) and negative answer variants (“No”, “no”, and “NO”), respectively. The terms $z _ { i ,  { \mathcal { V } _ { \mathrm { p o s } } } }$ and $z _ { i , \boldsymbol { \nu } _ { \mathrm { n e g } } }$ denote the average logits over the corresponding token sets. For example,

$$
z _ { i , \mathscr { V } _ { \mathrm { p o s } } } = \frac { 1 } { | \mathscr { V } _ { \mathrm { p o s } } | } \sum _ { v \in \mathscr { V } _ { \mathrm { p o s } } } \mathrm { L o g i t } _ { \theta } ( v \mid \mathscr { X } _ { i } ) .\tag{4}
$$

The term $z _ { i , \boldsymbol { \nu } _ { \mathrm { n e g } } }$ is defined analogously. The logits are extracted at the first decoding step under a constrained yes/no answer format. The final target $o _ { t }$ is selected as the candidate with the highest verification score:

$$
t = \arg \operatorname* { m a x } _ { i : o _ { i } \in \mathcal { O } _ { \mathrm { t o p } } } S _ { i } .\tag{5}
$$

Here, $S _ { i }$ is used as a relative ranking score rather than as a calibrated probability. It provides a signal for comparing the visual-semantic matching of different candidates.

## C. Data Pipeline and Training Strategy

Reasoning-Chain Data Pipeline. To obtain reliable reasoning-chain supervision for model training, we construct a self-consistency-filtered data pipeline that filters out reasoning chains that are merely linguistically plausible but not grounded in correct visual target discrimination.

a) Stage 1. Self-Consistency Generation: We first render the target region, which may correspond to either the referred target or a hard negative distractor, with an SoM prompt to obtain $\mathcal { T } _ { \mathrm { S o M } }$ , and then feed $( \mathcal { T } _ { \mathrm { S o M } } , \mathcal { Q } )$ into an MLLM and constrain its output to a structured response format using a task instruction prompt $\mathrm { P r o m p t } _ { \mathrm { C o T } } ( \mathcal { Q } )$

“Look at the image. Does it accurately match {Q}? Out-

put JSON strictly as {"reason": "...", "description":

${ \bf { \hat { \Pi } } } _ { \cdots } \cdot \cdot ,$ "answer": "Yes/No"}.”

The MLLM outputs a structured response containing a reasoning chain $\mathcal { R } _ { 1 }$ which represents the reasoning process with respect to the query Q, a description $\mathcal { D } _ { 1 }$ which describes the visual content of the boxed candidate region, and an answer $\mathcal { A } _ { 1 }$ , which represents the decision result.

b) Stage 2. Self-Verification: To avoid reasoning chains that are self-consistent only at the linguistic level but inconsistent with the visual evidence, we design a closed-loop verification [19] mechanism based on description reconstruction. Specifically, we use the $\mathcal { D } _ { 1 }$ generated in Stage 1 to construct a new verification query $\mathrm { P r o m p t } _ { \mathrm { C o T } } ( \mathcal { D } _ { 1 } )$ , which is then fed back into the MLLM for closed-loop verification. The model again produces a structured response containing the reasoning chain $\mathcal { R } _ { 2 }$ and answer $A _ { 2 } . \mathrm { ~ A ~ }$ reasoning chain is retained only if the first-stage answer $\mathcal { A } _ { 1 }$ matches the ground-truth answer, and the second-stage answer $\boldsymbol { A } _ { 2 }$ confirms the reconstructed description with “Yes”. Otherwise, the sample is discarded from the training set.

Training Strategy. To enhance the MLLM’s capability for visual-semantic discrimination while keeping the training cost manageable, we perform LoRA-based SFT [10] using the filtered reasoning-chain data.

## IV. EXPERIMENTS

## A. Experimental Settings

Implementation Details. We implement DRAgent using Grounding DINO [9] as the candidate object detector,

Qwen3-VL-8B-Instruct [20] as the MLLM-based target discriminator, and SAM [6] as the pixel-level segmentation model. Qwen3-VL-8B-Instruct is loaded with 4-bit quantization for efficient fine-tuning. LoRA [10] is applied to both the visual and language layers, with both the rank and α set to 16. The maximum sequence length is set to 2048, the learning rate to $5 \times 1 0 ^ { - 6 } .$ , and the weight decay to 0.01. Each training sample consists of a candidate-marked image and a text prompt, and the model outputs a short reasoning process together with the final verification answer.

Benchmarks and Datasets. We conduct experiments on three standard RES benchmarks: RefCOCO, RefCOCO+, and RefCOCOg [21], [22]. For training, we use only 6K randomly sampled instances from the RefCOCO training set to construct the reasoning-chain data.

Evaluation Metrics. Following prior work on text-guided segmentation [1], [12], [13], [23], we adopt cumulative Intersection over Union (cIoU) as the evaluation metric, defined as the ratio between the accumulated intersection and the accumulated union over the entire dataset.

Baselines. We compare DRAgent against three representative categories of methods on the RES benchmarks to provide a representative basis for comparison:

1) Non-MLLM-based, which align visual and textual features through convolutional networks, Transformers, or related architectures, and directly predict the mask.

2) Implicit MLLM-based, which use MLLMs for crossmodal fusion and perform mask prediction in an endto-end manner.

3) MLLMs as explicit visual localization generators, which use MLLMs to generate spatial prompts for foundation segmentation models to produce masks.

## B. Main Results

As shown in Table I, DRAgent achieves competitive overall results on RefCOCO, RefCOCO+, and RefCOCOg. Notably, it obtains the best performance on RefCOCOg, reaching 76.78% on val(U) and 77.54% on test(U), outperforming both implicit MLLM-based methods and explicit visual localization generator approaches. On RefCOCO and RefCOCO+, DRAgent does not consistently surpass the strongest methods, but remains competitive with recent explicit localization-generator methods.

These results support the effectiveness of using the MLLM as a discriminative target selector rather than a coordinate generator. Compared with methods that rely on implicit multimodal fusion, DRAgent explicitly exposes candidate objects to the MLLM and requires target-level discrimination. Compared with visual localization generator methods, DRAgent avoids directly decoding coordinates or points as textual outputs and instead performs target selection within a detector-generated candidate space. This design is particularly beneficial for RefCOCOg, where longer and more compositional expressions require fine-grained visualsemantic comparison among candidate objects.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val(U)</td><td>test(U)</td></tr><tr><td colspan="10">Non-MLLM-based</td></tr><tr><td>LAVT [1]</td><td>CVPR&#x27;22</td><td>72.7</td><td>75.8</td><td>68.8</td><td>62.1</td><td>68.4</td><td>55.1</td><td>61.2</td><td>62.1</td></tr><tr><td>SEEM [24]</td><td>NeurIPS&#x27;23</td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>65.7</td><td></td></tr><tr><td>ReLA [2]</td><td>CVPR&#x27;23</td><td>73.8</td><td>76.5</td><td>70.2</td><td>66.0</td><td>71.0</td><td>57.7</td><td>65.0</td><td>66.0</td></tr><tr><td>PolyFormer [11]</td><td>CVPR&#x27;23</td><td>75.96</td><td>78.29</td><td>73.25</td><td>69.33</td><td>74.56</td><td>61.87</td><td>69.20</td><td>70.19</td></tr><tr><td>CoHD [25]</td><td>ICCV&#x27;25</td><td>78.11</td><td>80.39</td><td>75.20</td><td>72.03</td><td>76.37</td><td>65.45</td><td>70.83</td><td>72.11</td></tr><tr><td colspan="10">Implicit MLLM-based</td></tr><tr><td>LISA [12]</td><td>CVPR&#x27;24</td><td>74.9</td><td>79.1</td><td>72.3</td><td>65.1</td><td>70.8</td><td>58.1</td><td>67.9</td><td>70.6</td></tr><tr><td>GSVA [13]</td><td>CVPR&#x27;24</td><td>79.2</td><td>81.7</td><td>77.1</td><td>70.3</td><td>73.8</td><td>63.6</td><td>75.7</td><td>77.0</td></tr><tr><td>GLaMM [15]</td><td>CVPR&#x27;24</td><td>79.5</td><td>83.2</td><td>76.9</td><td>72.6</td><td>78.7</td><td>64.6</td><td>74.2</td><td>74.9</td></tr><tr><td>PixelLM [16]</td><td>CVPR&#x27;24</td><td>73.0</td><td>76.5</td><td>68.2</td><td>66.3</td><td>71.7</td><td>58.3</td><td>69.3</td><td>70.5</td></tr><tr><td>PSALM [23]</td><td>ECCV’24</td><td>83.6</td><td>84.7</td><td>81.6</td><td>72.9</td><td>75.5</td><td>70.1</td><td>73.8</td><td>74.4</td></tr><tr><td>OpenWorldSAM [14]</td><td>NeurIPS’25</td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.0</td><td></td></tr><tr><td>POPEN [26]</td><td>CVPR&#x27;25</td><td>79.3</td><td>82.0</td><td>74.1</td><td>73.1</td><td>77.0</td><td>65.1</td><td>75.4</td><td>75.6</td></tr><tr><td>Dr.Seg [27]</td><td>CVPR&#x27;26</td><td></td><td>80.2</td><td>一</td><td></td><td>76.8</td><td></td><td></td><td>74.2</td></tr><tr><td colspan="10"></td></tr><tr><td>MLLMs as Explicit Visual Localization Generators</td><td></td><td></td><td>82.7</td><td>74.7</td><td>74.6</td><td>80.0</td><td>67.2</td><td>75.5</td><td>76.4</td></tr><tr><td>SAM4MLLM [3] SAM-R1 [28]</td><td>ECCV&#x27;24 NeurIPS&#x27;25</td><td>79.8</td><td>79.2</td><td></td><td></td><td>74.7</td><td></td><td></td><td>73.1</td></tr><tr><td>SegAgent [4]</td><td>CVPR&#x27;25</td><td>79.69</td><td>81.35</td><td>76.57</td><td>72.49</td><td>75.80</td><td>66.89</td><td>75.11</td><td>75.20</td></tr><tr><td>DPAD [17]</td><td>CVPR&#x27;26</td><td></td><td>79.3</td><td></td><td></td><td>74.7</td><td>一</td><td></td><td>72.6</td></tr><tr><td>SAM-Veteran [5]</td><td>ICLR&#x27;26</td><td>一</td><td>80.8</td><td></td><td>一</td><td>76.6</td><td></td><td></td><td>73.4</td></tr><tr><td>DRAgent</td><td>this work</td><td>77.33</td><td>80.73</td><td>1 76.34</td><td>73.14</td><td>78.79</td><td>一 64.29</td><td>一 76.78</td><td>77.54</td></tr><tr><td colspan="3">TABLE II ABLATION STUDY ON CORE COMPONENTS.</td><td colspan="4">TABLE III PARAMETER ANALYSIS OF K.</td><td colspan="3">TABLE IV PARAMETER ANALYSIS OF γ.</td></tr><tr><td colspan="3">Setting</td><td colspan="4"></td><td colspan="3">RefCOCOg</td></tr><tr><td></td><td>val(U)</td><td>Avg. test(U)</td><td>K</td><td>RefCOCOg val(U)</td><td>test(U)</td><td>Avg.</td><td></td><td>test(U)</td><td>Avg.</td></tr><tr><td>Baseline +2Stage-DR</td><td>66.50</td><td>67.17 66.84</td><td>2</td><td>73.76</td><td>75.95</td><td>74.86</td><td>40%</td><td>74.55</td><td>76.96</td></tr><tr><td></td><td>74.13</td><td>76.70 75.42</td><td>345</td><td>74.54</td><td>77.83</td><td>76.19</td><td>50%</td><td>76.36</td><td>77.47</td><td>75.76 76.92</td></tr><tr><td>+2Stage-DR+SFT w/o CoT</td><td>72.35</td><td>72.66 72.51</td><td></td><td>76.78</td><td>77.54</td><td>77.16</td><td>60%</td><td>76.78</td><td>77.54</td><td>77.16</td></tr><tr><td>+2Stage-DR+SFT w/ Raw CoT</td><td>74.26</td><td>74.59</td><td></td><td>76.69</td><td>77.32</td><td>77.01</td><td>70%</td><td>77.03</td><td>77.25</td><td>77.14</td></tr><tr><td>DRAgent (Full)</td><td>76.78</td><td>74.91 77.54 77.16</td><td>6</td><td>75.87</td><td>77.24</td><td>76.56</td><td>80%</td><td>76.20</td><td>76.97</td><td>76.59</td></tr></table>

TABLE I  
COMPARISON WITH EXISTING STATE-OF-THE-ART METHODS ON RES BENCHMARKS. THE EVALUATION METRIC IS CIOU (%). THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.

## C. Ablation Study

We conduct a systematic ablation study on the RefCOCOg dataset, with the detailed settings and results reported in Table II. The baseline uses Grounding DINO [9] to generate candidates, a pretrained MLLM to select the target in a single stage, and SAM [6] to generate the final mask. We progressively incorporate the two-stage DR mechanism, SFT under different supervision settings, and filtered reasoningchain data, in order to evaluate the contributions of each component as well as their joint effect.

The baseline achieves an average cIoU of 66.84% on RefCOCOg, indicating that a single-stage pretrained MLLM selector is not sufficient to reliably identify the referred target from detector-generated candidates. After introducing the two-stage DR mechanism, the average cIoU increases to 75.42%. This improvement can be attributed to the design of narrowing the candidate space through screening and then conducting instance-wise verification, which reduces the interference from irrelevant candidates while allowing the MLLM to focus on fine-grained visual-semantic matching. The fine-tuning variants show the influence of supervision design. Applying SFT without CoT supervision obtains an average cIoU of 72.51%, which is lower than the two-stage

DR variant. This suggests that final-answer supervision alone provides limited information about how the MLLM should compare candidates and make discriminative decisions. With unfiltered raw CoT data, the average cIoU reaches 74.59%, indicating that reasoning-chain supervision can provide intermediate guidance for learning target discrimination. However, raw CoT data may still contain visually inconsistent or weakly grounded reasoning, which limits its effectiveness. The full DRAgent achieves the best average cIoU of 77.16% with self-consistency-filtered reasoning-chain data, suggesting that consistency-based filtering provides more reliable supervision by retaining reasoning chains whose decisions are aligned with visual evidence. Overall, the two-stage DR mechanism improves the inference-time selection process, while self-consistency-filtered reasoning-chain data improves the training signal for discriminative target selection.

## D. Analysis

During discriminative target selection, the number of initially selected candidates K controls the size of the candidate subset. As shown in Table III, the average cIoU increases as K grows from 2 to 4, suggesting that retaining more candidates improves target recall and provides a richer decision space for subsequent verification. When K is further increased, the average cIoU declines slightly, suggesting that an excessively large candidate set introduces more semantically irrelevant or visually confusable candidates, thereby increasing the difficulty of subsequent discriminative verification. Overall, K = 4 achieves the best performance.

![](images/9e39fd4781daf457a8c44fb2e6c3f6b92dfdef5fc7d8f5963fae15cc098e739f.jpg)  
Fig. 3. Visualization of qualitative results. Each row shows one RES example, with referring text, input image, baseline result, our DRAgent result, and ground truth (GT). Examples cover diverse challenging scenarios, and DRAgent achieves more accurate segmentation consistent with GT.

We also evaluate the overlap compensation threshold γ, which determines whether candidates overlapping with the initially selected boxes are retained. As shown in Table IV, increasing γ from 40% to 60% improves the average cIoU, indicating that moderate overlap compensation helps preserve relevant neighboring candidates. However, further in creasing γ reduces performance, as an overly strict overlap threshold reduces the number of retained relevant candidates. Overall, $\gamma \ = \ 6 0 \%$ provides the best trade-off between candidate recall and noise suppression.

## V. CONCLUSION

We present DRAgent, a MLLM-driven RES framework. DRAgent reformulates the role of the MLLM as a discriminative target selector over detected candidates. Through the two-stage DR mechanism, the MLLM first performs candidate screening and then conducts instance-wise verification to identify the referred target, which is subsequently converted into a mask by the segmentation model. We further introduce a self-consistency-filtered reasoningchain data pipeline for LoRA-based fine-tuning, enhancing the MLLM’s DR capability with more reliable supervision. Experiments on RefCOCO, RefCOCO+, and RefCOCOg demonstrate the effectiveness of the proposed framework.

## REFERENCES

[1] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H. S. Torr, “LAVT: Language-aware vision transformer for referring image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18134–18144.

[2] C. Liu, H. Ding, and X. Jiang, “GRES: Generalized referring expression segmentation,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 23592–23601.

[3] Y.-C. Chen, W.-H. Li, C. Sun, Y.-C. F. Wang, and C.-S. Chen, “SAM4MLLM: Enhance multi-modal large language model for referring expression segmentation,” in Computer Vision – ECCV 2024, 2024, pp. 323–340.

[4] M. Zhu, Y. Tian, H. Chen, C. Zhou, Q. Guo, Y. Liu, M. Yang, and C. Shen, “SegAgent: Exploring pixel understanding capabilities in MLLMs by imitating human annotator trajectories,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 3686–3696.

[5] T. Du, H. Li, Z. Fan, J. Zhang, P. Pan, and Y. Zhang, “SAM-Veteran: An MLLMbased human-like SAM agent for reasoning segmentation,” in Proceedings of the Fourteenth International Conference on Learning Representations, 2026.

[6] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick, “Segment any-´ thing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.

[7] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu, and H. Li, “Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 8612–8642.

[8] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou, “Self-consistency improves chain of thought reasoning in language models,” in Proceedings of the 11th International Conference on Learning Representations, 2023.

[9] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in Computer Vision – ECCV 2024, 2024, pp. 38–55.

[10] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proceedings of the 10th International Conference on Learning Representations, 2022.

[11] J. Liu, H. Ding, Z. Cai, Y. Zhang, R. K. Satzoda, V. Mahadevan, and R. Manmatha, “PolyFormer: Referring image segmentation as sequential polygon generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18653–18663.

[12] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “LISA: Reasoning segmentation via large language model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9579–9589.

[13] Z. Xia, D. Han, Y. Han, X. Pan, S. Song, and G. Huang, “GSVA: Generalized segmentation via multimodal large language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3858–3869.

[14] S. Xiao, R. Kabra, Y. Li, D. Lee, J. Carreira, and P. Panda, “OpenWorldSAM: Extending SAM2 for universal image segmentation with language prompts,” in Advances in Neural Information Processing Systems, 2025.

[15] H. Rasheed, M. Maaz, S. S. Mullappilly, A. Shaker, S. Khan, H. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, “GLaMM: Pixel grounding large multimodal model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13009–13018.

[16] Z. Ren, Z. Huang, Y. Wei, Y. Zhao, D. Fu, J. Feng, and X. Jin, “PixelLM: Pixel reasoning with large multimodal model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26374– 26383.

[17] T. Yang, Q. Zhou, Y. Li, and Q. Wang, “Discriminative perception via anchored description for reasoning segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, to appear. arXiv:2603.04002.

[18] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4V,” arXiv preprint arXiv:2310.11441, 2023.

[19] Y. Weng, M. Zhu, F. Xia, B. Li, S. He, S. Liu, B. Sun, K. Liu, and J. Zhao, “Large language models are better reasoners with self-verification,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 2550–2575.

[20] S. Bai et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[21] J. Mao, J. Huang, A. Toshev, O. Camburu, A. L. Yuille, and K. Murphy, “Generation and comprehension of unambiguous object descriptions,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 11–20.

[22] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in Computer Vision – ECCV 2016, 2016, pp. 69–85.

[23] Z. Zhang, Y. Ma, E. Zhang, and X. Bai, “PSALM: Pixelwise segmentation with large multi-modal model,” in Computer Vision – ECCV 2024, 2024, pp. 74–91.

[24] X. Zou, J. Yang, H. Zhang, F. Li, L. Li, J. Wang, L. Wang, J. Gao, and Y. J. Lee, “Segment everything everywhere all at once,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 19769–19782.

[25] Z. Luo, Y. Wu, T. Cheng, Y. Liu, Y. Xiao, H. Wang, X.-P. Zhang, and Y. Yang, “CoHD: A counting-aware hierarchical decoding framework for generalized referring expression segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 22685–22694.

[26] L. Zhu, T. Chen, Q. Xu, X. Liu, D. Ji, H. Wu, D. W. Soh, and J. Liu, “POPEN: Preference-based optimization and ensemble for LVLM-based reasoning segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 30231–30240.

[27] H. Sun, T. Wang, C. Tang, L. Yuan, and J. Lv, “Dr.Seg: Revisiting GRPO training for visual large language models through perception-oriented design,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, to appear. arXiv:2603.00152.

[28] J. Huang, Z. Xu, J. Zhou, T. Liu, Y. Xiao, M. Ou, B. Ji, X. Li, and K. Yuan, “SAM-R1: Leveraging SAM for reward feedback in multimodal segmentation via reinforcement learning,” in Advances in Neural Information Processing Systems, 2025.