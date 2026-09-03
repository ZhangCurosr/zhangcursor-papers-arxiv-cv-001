# Blending Concepts: Benchmarking Visual Metaphor Generation in Text-to-Image Models

Chuer Chen<sup>1</sup> Zichen Wang<sup>1</sup> Yi He<sup>1</sup> Zhengxi Yu<sup>1</sup> Nan Cao<sup>1,2∗</sup> <sup>1</sup>Tongji University <sup>2</sup>Shanghai Innovation Institute

## Abstract

Text-to-image (T2I) models have achieved remarkable success at faithfully rendering specified objects and attributes, yet their ability to produce visual metaphors—images that convey abstract ideas by combining elements from two distinct domains—remains largely unexamined. To bridge this gap, we introduce VMetaphor-Bench, the first benchmark for evaluating visual metaphor generation in T2I models. It comprises 1,500 visual metaphors curated from real-world creative imagery, organized into three levels and ten categories, with each sample paired with two prompts of differing specificity. For evaluation, we develop a hybrid framework within an MLLM-as-judge paradigm, combining a multiplechoice question (MCQ) based protocol of 9,594 questions across four levels of metaphorical fidelity with a dimension-based scoring protocol along three perceptual dimensions. Extensive evaluation of 11 representative T2I models reveals that even the strongest proprietary models struggle with compositional structuring and cross-domain mapping, key aspects of metaphorical expression, highlighting visual metaphor generation as an important frontier for future T2I research.

## 1 Introduction

Recent years have witnessed remarkable progress in text-to-image (T2I) generation, driven by advances across three main architectural directions: diffusion models [20, 35], autoregressive models [40, 43], and unified multimodal architectures [8, 9, 48]. Powered by large-scale training, a wide range of T2I systems, from leading proprietary models such as GPT-Image-1 [30] and Gemini 3.1 Flash Image [16] to open-source counterparts [27, 47], now produce photorealistic images that faithfully render specified objects, attributes, and spatial layouts, achieving strong performance on basic semantic prompt following. However, their ability to express abstract concepts and convey meaning beyond what is literally depicted remains far less explored.

A particularly important form of such abstract expression is the visual metaphor, where one concept is understood through another via visual analogy [14]. Drawing on conceptual blending theory [13], a visual metaphor combines elements from two distinct domains — a source that lends its properties, and a target that receives them — to produce a single image carrying meaning richer than either domain alone. Generating such images is far from straightforward: the model must (i) blend the two given domains coherently using a specific compositional strategy (e.g., fusion or juxtaposition), (ii) establish meaningful element-level mappings between the two, and (iii) ensure that the resulting image conveys the intended meaning to viewers. As shown in Figure 1, even state-of-the-art T2I models frequently struggle with these tasks, producing images that miss one of the domains, adopt the wrong structural form, blend the two awkwardly, or fail to convey the intended meaning.

Yet this challenge has been largely overlooked by existing T2I benchmarks, which remain dominated by literal prompt following. Mainstream benchmarks such as GenEval [15] and T2I-CompBench [22] evaluate object, attribute, and basic spatial fidelity from short prompts, while more recent reasoning aware benchmarks [29, 37, 55] extend toward factual or physical inference, leaving the cross-domain conceptual blending required by visual metaphors unexamined. On the metaphor side, existing efforts have focused predominantly on the understanding direction [2, 26, 52]; the only generation evaluation we are aware of, conducted as a small probe within MetaCLUE [2], is limited to 300 samples assessed via FID, CLIP similarity, and a small user study — none of which can systematically diagnose where and how T2I models fail at metaphorical expression. Together, these limitations leave a clear gap: no existing benchmark systematically evaluates how well T2I models generate visual metaphors across diverse themes and metaphorical structures.

![](images/0cc816629d363c45ac39f8b56a45589c31eef60ff8edf23b67d03125ce51e70e.jpg)  
Figure 1: Common failure modes in visual metaphor generation from GPT-Image-1.5 [30], Seedream 5.0 Lite [4], and FLUX.2-dev [28].

To address this gap, we introduce VMetaphor-Bench, a new benchmark designed for evaluating visual metaphor generation in T2I models. It comprises 1,500 visual metaphors curated from realworld creative imagery, organized into three levels and ten categories, as illustrated in Figure 2. Each sample is annotated with its source and target domains, structure type, metaphorical meaning, element-level mappings, and paired with two prompts of differing specificity: a conceptual prompt that conveys only the metaphorical idea, and a descriptive prompt that details its visual realization. For evaluation, we develop a hybrid framework consisting of an MCQ-based protocol — 9,594 questions targeting four levels of metaphorical fidelity (domain presence, structure type, meaning, and element mapping) — and a dimension-based scoring protocol rating each image on metaphoric efficacy, metaphor logic, and perceptual harmony. Both protocols are realized within an MLLM-as-judge paradigm, enabling fine-grained, interpretable diagnosis of T2I models’ strengths and weaknesses in visual metaphor generation.

Using VMetaphor-Bench, we conduct a comprehensive evaluation of 11 representative T2I models, spanning leading proprietary systems, open-source diffusion models, and unified multimodal architectures. Our results reveal that proprietary models such as GPT Image 1.5 and Nano Banana 2 achieve substantially better performance than open-source counterparts. However, even the strongest models struggle with compositional structuring and cross-domain mapping—key aspects of metaphorical expression—highlighting visual metaphor generation as an important frontier for future T2I research.

![](images/037c8f0cd717248e0dd7bd5f3a3edaad67c4fb4e292096cf5629152667986873.jpg)  
Figure 2: Overview of VMetaphor-Bench. Left: hierarchical taxonomy with three levels and ten thematic categories. Middle: representative examples spanning the three levels and the three structure types of visual metaphors. Right: performance of 11 evaluated T2I models on MCQ accuracy and dimension scoring under conceptual prompts.

In summary, our contributions are threefold:

• We introduce VMetaphor-Bench, the first benchmark (to the best of our knowledge) dedicated to evaluating visual metaphor generation in T2I models, comprising 1,500 high-quality, carefully curated samples organized into a hierarchical taxonomy of three levels and ten categories.

• We design a hybrid evaluation framework with an MCQ-based protocol of 9,594 questions across four levels of metaphorical fidelity and a dimension-based protocol along three perceptual dimensions, enabling fine-grained, interpretable assessment.

• We conduct a comprehensive evaluation of 11 representative T2I models, providing insights into their capabilities in visual metaphor generation and highlighting key directions for future research.

## 2 Related Works

## 2.1 Text-to-Image Generation Models

Text-to-image (T2I) generation has advanced rapidly along three main architectural directions. Diffusion models [10, 20, 35] remain the dominant paradigm, with recent progress driven by scaling transformer-based denoisers [12, 27, 32]. Autoregressive models [38, 40, 43] cast image generation as sequential token prediction, offering fine-grained compositional control through explicit modeling of inter-token dependencies. More recently, unified multimodal models [8, 9, 48, 49] integrate visual understanding and generation within a single architecture, leveraging multimodal reasoning to better interpret complex prompts. State-of-the-art systems such as GPT-Image-1 [30] and Gemini 3.1 Flash Image [16] now produce images of remarkable fidelity and aesthetic quality for everyday scenes. However, generating images that convey abstract semantic relationships, such as visual metaphors, remains a significant challenge across all three paradigms.

## 2.2 Text-to-Image Generation Benchmarks

Early evaluation of text-to-image generation relied on distribution-level metrics such as FID [19] and feature-based similarity scores like CLIPScore [18], which capture perceptual quality but fail to assess fine-grained semantic correctness. The recent rise of multimodal large language models (MLLMs) [23, 41] has enabled a paradigm shift toward MLLM-as-judge evaluation [6], where a vision-language model directly examines the generated image against the prompt, providing a more flexible and semantically grounded assessment framework. Under this paradigm, a line of benchmarks targets foundational semantic alignment — verifying whether generated images faithfully depict the specified objects [24], attributes [51], and spatial relationships [45, 46]. Representative works include GenEval [15], T2I-CompBench [22], and TIFA [21], which decompose prompts into atomic verification questions to systematically evaluate compositional fidelity. More recently, benchmark have shifted toward higher-order reasoning capabilities, probing whether T2I models can incorporate world knowledge [29, 53] and reasoning [7, 37, 55] into their generations, reflecting a broader trend from evaluating object-level fidelity toward scene-level semantic understanding.

## 2.3 Visual Metaphor

Visual metaphor communicates abstract ideas by mapping one conceptual domain onto another through visual imagery [11, 14, 44]. On the understanding side, MetaCLUE [2] introduces a comprehensive benchmark for metaphor classification, localization, interpretation, along with a small probe on generation; ImageMet [26] benchmarks VLMs on metaphor captioning and VQA, and MetaphorStar [52] applies visual reinforcement learning to improve metaphor reasoning. On the generation side, several recent methods leverage text-to-image models as the visual backbone: Creative Blends [39] fuses disparate visual concepts via commonsense knowledge, The Mind’s Eye [25] proposes a source-target-meaning decomposition with self-evaluation rewards, and Beyond Pixels [50] introduces schema-driven reasoning for metaphor transfer. However, evaluation of these methods has been limited—typically conducted on small-scale ad hoc test sets via FID, CLIP similarity, or user studies, none of which can diagnose where and how T2I models fail at metaphorical expression. While benchmarks exist for metaphor understanding, no counterpart has been established for systematically evaluating the generation quality of visual metaphors.

## 3 VMetaphor-Bench

We present VMetaphor-Bench, the first benchmark, to the best of our knowledge, for evaluating visual metaphor generation in text-to-image models, featuring multi-dimensional assessments of cross-domain conceptual blending. We describe the dataset construction process in Section 3.1 and detail the evaluation framework in Section 3.2. An overview of the complete pipeline is illustrated in Figure 3.

![](images/f2d7e771880037269547e07e4187039a13f45d6f2a0a122d7493cf59caf44674.jpg)  
Figure 3: Overview of the VMetaphor-Bench pipeline. Left: Dataset Construction — images are collected and annotated through a two-stage pipeline, producing structured annotations and two granularity levels of prompts. Right: Evaluation Framework — generated images are assessed via MCQ-based evaluation (L1–L4 levels) and dimension-based scoring (Efficacy, Logic, Harmony).

## 3.1 Dataset Construction

Image Collection. To construct a benchmark grounded in real-world visual metaphors, we collect images from Pinterest [1], a platform rich in creative visual content such as advertisements, editorial illustrations, and conceptual art. We query the platform using over ten keywords spanning both general and domain-specific terms, e.g. visual metaphor, creative ads. This process yields approximately 5,000 candidate images. Two authors then review the entire collection, filtering out duplicates, images that do not contain a clear visual metaphor, and low-quality or ambiguous samples, resulting in approximately 1,600 images for subsequent annotation.

Data Annotation. After collecting images, we employ Gemini 3.1 Pro [17] as the annotation backbone for its strong visual reasoning capabilities. Since visual metaphors involve multi-layered semantics [5], generating accurate annotations and high-quality prompts simultaneously in a single pass is challenging. We therefore adopt a two-stage annotation pipeline. In the first stage, the model is presented with each image and tasked with extracting structured annotations:

• Target domain: the subject being described — what the image is ultimately “about”.

• Source domain: the concept used to characterize the target— it lends a property to the target through visual analogy.

• Structure type: how the two domains are visually combined. Following Phillips et al. [33], we categorize it as replacement (one domain is absent and inferred from context),fusion (both domains are merged into a single object), or juxtaposition (both domains appear as separate objects in the same scene).

• Meaning: the overall message conveyed through the interaction of the two domains.

• Element mappings: fine-grained correspondences specifying which source element maps to which target element.

In the second stage, the model receives both the image and the extracted annotations, and generates two prompts at different levels of granularity: (1) The descriptive prompt provides a detailed depiction of the visual content and how the metaphor is physically rendered in the image; (2) The conceptual prompt specifies the target and source domains, implicitly conveys the structure type, and states the intended meaning, without describing specific visual details or element-level mappings. Additionally, a style suffix describing the visual style of the original image (e.g., photorealistic, illustration) is extracted and appended to both prompts during generation to ensure stylistic consistency. This two-stage design allows the structured annotations to guide prompt generation, ensuring consistency between the structured annotations and the textual descriptions.

Three annotators, after training on visual metaphor theory and annotation guidelines, perform rigorous manual inspection on all images, annotations, and prompts, and cross-validate each other’s decisions. Images where the metaphor is ambiguous or unrecognizable are removed. Inaccurate annotations (e.g., misidentified target or source domains) and imprecise prompts are manually corrected to align with the visual content. This verification process yields our final set of 1,500 high-quality images, each accompanied by structured annotations and two granularity levels of prompts.

## 3.2 Evaluation Framework

Evaluating visual metaphor generation requires assessing both metaphorical fidelity — whether the generated image faithfully captures the intended cross-domain mapping — and expressive quality — how effectively and aesthetically the metaphor is conveyed. To this end, we adopt a hybrid evaluation framework with MLLM-as-judge [54], comprising two complementary approaches: (1) MCQ-based evaluation that measures the correctness of metaphorical elements through multiple-choice questions, and (2) dimension-based scoring that rates the overall quality of the generated metaphor on a 1-5 Likert scale. An overview of the evaluation framework is shown in Figure 3 (Right).

MCQ-based Evaluation. To objectively assess the correctness of generated images at each level of the metaphorical structure, we automatically construct a set of multiple-choice questions (MCQs) for all 1,500 samples in the dataset. Specifically, we leverage Gemini 3.1 Pro to generate MCQs based on each sample’s structured annotations along with its descriptive prompt. The generated MCQs assess four hierarchical levels of metaphorical fidelity:

• L1 – Domain Presence: Whether the target and source domains are visually recognizable in the image (one question per domain).

• L2 – Structure Type: Whether the visual combination strategy matches the ground-truth structure type (replacement, fusion, or juxtaposition).

• L3 – Meaning: Whether the image conveys the intended metaphorical meaning.

• L4 – Element Mapping: Whether each source element correctly corresponds to its intended target element (one question per mapping pair; scores are averaged).

This process yields a total of 9,594 MCQs across all samples, which are manually verified by the same annotators to ensure correctness. Each question consists of four options: one ground-truth answer derived from the annotation, two plausible but incorrect distractors, and a "cannot be determined" option. All option orders are randomized to eliminate positional bias. For each image, we compute per-level accuracy: L1 averages over the two domain questions, L2 is a single binary score, L3 is a single binary score, and L4 averages over all mapping questions. The overall MCQ score is the mean across all four levels.

Dimension-based Scoring. While MCQ-based evaluation measures the correctness of metaphorical elements, it does not capture the overall perceptual quality of the generated image. We therefore complement it with dimension-based scoring, where an MLLM evaluator is presented with the generated image along with its structured annotations and asked to rate the image on a 1–5 Likert scale across three independent dimensions:

• Metaphoric Efficacy: How clearly the image communicates the intended metaphorical meaning to a viewer without any textual explanation.

• Metaphor Logic: Whether the cross-domain mapping between source and target is conceptually coherent and logically sound.

• Perceptual Harmony: How naturally and coherently the two domains are visually presented, considering blending, composition, and overall visual cohesion.

The overall dimension score is the average across all three dimensions. Together with the MCQ-based evaluation, this hybrid framework provides a comprehensive assessment of both the metaphorical fidelity and expressive quality of generated visual metaphors. Full evaluation prompts are provided in Appendix A.7.

Table 1: Evaluation results on VMetaphor-Bench with conceptual prompts.
<table><tr><td rowspan="2">Model</td><td colspan="5">MCQ Accuracy (%)</td><td colspan="4">Dimension Score (1–5)</td></tr><tr><td>Domain</td><td>Structure</td><td>Meaning</td><td>Mapping</td><td>Overall</td><td>Efficacy</td><td>Logic</td><td>Harmony</td><td>Overall</td></tr><tr><td>Closed-source T2I Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream 5.0 Lite</td><td>90.6</td><td>65.6</td><td>96.0</td><td>66.5</td><td>78.5</td><td>3.78</td><td>3.74</td><td>4.08</td><td>3.86</td></tr><tr><td>Nano Banana 2</td><td>93.0</td><td>76.3</td><td>98.5</td><td>75.9</td><td>84.8</td><td>4.05</td><td>4.08</td><td>4.21</td><td>4.11</td></tr><tr><td>GPT Image 1.5</td><td>93.6</td><td>78.1</td><td>97.6</td><td>76.5</td><td>85.4</td><td>4.27</td><td>4.23</td><td>4.38</td><td>4.30</td></tr><tr><td>Open-source T2I Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLUX.1-dev</td><td>82.3</td><td>54.9</td><td>85.6</td><td>55.2</td><td>68.4</td><td>2.90</td><td>2.87</td><td>3.82</td><td>3.19</td></tr><tr><td>SD 3.5 Large</td><td>83.0</td><td>60.1</td><td>85.7</td><td>58.9</td><td>70.8</td><td>2.99</td><td>3.01</td><td>3.75</td><td>3.25</td></tr><tr><td>Z-Image</td><td>84.3</td><td>59.6</td><td>89.5</td><td>60.8</td><td>72.4</td><td>2.93</td><td>3.03</td><td>3.67</td><td>3.21</td></tr><tr><td>Qwen-Image</td><td>87.3</td><td>66.1</td><td>90.5</td><td>61.7</td><td>74.9</td><td>3.39</td><td>3.31</td><td>3.84</td><td>3.52</td></tr><tr><td>FLUX.2-dev</td><td>88.6</td><td>63.7</td><td>92.1</td><td>63.9</td><td>76.0</td><td>3.48</td><td>3.46</td><td>3.93</td><td>3.62</td></tr><tr><td>Open-source Unified MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniGen2</td><td>74.8</td><td>48.1</td><td>77.1</td><td>49.1</td><td>61.4</td><td>2.53</td><td>2.44</td><td>3.45</td><td>2.74</td></tr><tr><td>Bagel</td><td>79.8</td><td>54.8</td><td>84.2</td><td>54.2</td><td>67.0</td><td>2.69</td><td>2.72</td><td>3.56</td><td>2.99</td></tr><tr><td>Janus-Pro</td><td>83.4</td><td>59.1</td><td>89.1</td><td>63.2</td><td>73.0</td><td>2.90</td><td>3.02</td><td>3.43</td><td>3.12</td></tr></table>

## 4 Experiments

In this section, we evaluate representative T2I models on VMetaphor-Bench and analyze their behavior under different prompt granularities. We further validate the reliability of our MLLMas-judge framework through cross-judge consistency and human alignment studies. Additional experimental details and results are provided in the Appendix.

## 4.1 Setup

Models. We evaluate 11 representative text-to-image models spanning three categories: (1) Closedsource T2I Models. GPT Image 1.5 [30], Nano Banana 2 (Gemini 3.1 Flash Image) [16], and Seedream 5.0 Lite [4]. These proprietary models represent the current state of the art in text-toimage generation; (2) Open-source T2I Models. Stable Diffusion 3.5 Large [36], FLUX.1-dev [27], FLUX.2-dev [28], Z-Image [42], and Qwen-Image [47]. These models represent the leading opensource text-to-image models; (3) Open-source Unified MLLMs. OmniGen2 [48], Bagel [9], and Janus-Pro [8]. These models employ a unified architecture for both image generation and multimodal understanding. All models are evaluated using their default generation configurations.

MLLM Evaluator. For both MCQ-based evaluation and dimension-based scoring, we adopt an opensource MLLM, Qwen3.5-27B [34], as the default judge model for its strong visual reasoning capability and full reproducibility. We additionally report results with GPT-5.4 [31] in the Appendix A.5 to validate cross-judge consistency.

## 4.2 Main Results on VMetaphor-Bench

We report the main evaluation results with conceptual prompts in Table 1. All metrics are assigned by Qwen3.5-27B as the default judge. Based on the results across 11 models, we highlight several key findings regarding model capabilities and core challenges in visual metaphor generation.

Closed-source models lead by a substantial margin. GPT Image 1.5 achieves the best overall performance with 85.4% MCQ accuracy and a dimension score of 4.30, followed by Nano Banana 2 (84.8%, 4.11). In contrast, the best-performing open-source T2I model, FLUX.2-dev, trails by 9.4 percentage points (76.0%, 3.62). While FLUX.2-dev narrows the gap with the weakest closed-source model Seedream 5.0 Lite (78.5%), a clear performance tier remains, suggesting that visual metaphor generation—which demands both semantic comprehension and creative compositional ability—still substantially benefits from the larger-scale training and architectural advantages of proprietary systems. Unified MLLMs show mixed performance: while Janus-Pro (73.0%) is competitive with mid-tier open-source T2I models, OmniGen2 (61.4%) and Bagel (67.0%) rank at the bottom despite their multimodal understanding capabilities.

![](images/f0fb8eff59652be78acae6bed96aaf1d1bf7d4faf4af61e881af6567717338c5.jpg)  
Figure 4: Qualitative comparison of generated images across seven representative models on four visual metaphors. Each row corresponds to one metaphor, with the conceptual prompt shown on the left. The four metaphors cover three structure types: fusion (rows 1–2), replacement (row 3), and juxtaposition (row 4). Blue and purple text highlight the target and source domains, respectively.

Structure and Mapping are the primary bottleneck. All models exhibit a consistent difficulty hierarchy across MCQ levels: Domain and Meaning accuracies are consistently the two highest among the four levels for every model, indicating that models can generate images containing the correct conceptual elements and conveying the intended metaphorical meaning. However, performance drops sharply on Structure (48.1–78.1%) and Mapping (49.1–76.5%), revealing that the core challenge lies in compositional structuring—models can depict the right elements but struggle to compose them into coherent metaphorical structures. This pattern is particularly evident in Seedream 5.0 Lite, which achieves 90.6% on Domain and 96.0% on Meaning yet drops to 65.6% on Structure and 66.5% on Mapping. Notably, a deeper analysis of structure type errors reveals that replacement is far more challenging thanfusion or juxtaposition, a pattern that mirrors the cognitive complexity hierarchy in visual rhetoric theory [33] (see Appendix A.3).

Visually harmonious but metaphorically shallow. Across all models, Perceptual Harmony is consistently the highest-scoring dimension (3.43–4.38), while Metaphoric Efficacy and Metaphor Logic generally trail behind. This suggests that although the intended meaning can be inferred (as reflected by high meaning accuracy), the metaphor is not conveyed with sufficient clarity and immediacy through visual structure—models produce aesthetically pleasing images without powerful metaphorical expression. The gap between Harmony and Efficacy widens for weaker models: OmniGen2 scores 3.45 on Harmony but only 2.53 on Efficacy (gap = 0.92), whereas GPT Image 1.5 maintains a gap of only 0.11 (4.38 vs. 4.27).

## 4.3 Qualitative Analysis

To complement the quantitative results above, Figure 4 presents representative outputs from seven models on four visual metaphors covering different structure types (fusion, replacement, and juxtaposition). Several patterns emerge that are consistent with our quantitative findings. Closed-source models such as GPT Image 1.5 and Nano Banana 2 generally produce coherent compositions that integrate the two domains in meaningful ways. In contrast, weaker open-source models often struggle to compose the two domains correctly. Mid-tier models such as FLUX.2-dev and Z-Image fall in between, occasionally capturing the metaphorical concept but producing visually disharmonious compositions or failing on structurally demanding cases.

Table 2: Performance comparison between conceptual and descriptive prompts on representative models. “Conc.” and “Desc.” denote the conceptual and descriptive prompt settings, respectively. ∆ denotes the change when switching from conceptual to descriptive prompts.
<table><tr><td rowspan="2">Model</td><td colspan="3">MCQ Overall (%)</td><td colspan="3">Dimension Overall</td></tr><tr><td>Conc.</td><td>Desc.</td><td>∆</td><td>Conc.</td><td>Desc.</td><td>∆</td></tr><tr><td>GPT Image 1.5</td><td>85.4</td><td>88.5</td><td>+3.1</td><td>4.30</td><td>4.24</td><td>-0.06</td></tr><tr><td>Nano Banana 2</td><td>84.8</td><td>89.2</td><td>+4.4</td><td>4.11</td><td>4.18</td><td>+0.07</td></tr><tr><td>FLUX.2-dev</td><td>76.0</td><td>87.3</td><td>+11.3</td><td>3.62</td><td>4.09</td><td>+0.47</td></tr><tr><td>Z-Image</td><td>72.4</td><td>85.9</td><td>+13.5</td><td>3.21</td><td>3.90</td><td>+0.69</td></tr><tr><td>OmniGen2</td><td>61.4</td><td>78.0</td><td>+16.6</td><td>2.74</td><td>3.35</td><td>+0.61</td></tr></table>

A closer look at typical failure cases reveals three recurring issues that map directly to our quantitative findings. The most common is incorrect structural composition: models depict both domains but fail to combine them in the structural form implied by the prompt. For example, in row 4, where the prompt suggests a juxtaposition of a cigarette and a handgun-shaped shadow, SD 3.5 Large, Z-Image, and Janus-Pro instead fuse the two into a single physical object—echoing the sharp performance drop on L2 Structure across all models. A second issue is poor visual integration, where the two domains are placed together but blend awkwardly, such as a coffee cup stacked on visibly distinct pillows without any unified form (row 2). Although the resulting images can still appear visually polished, the awkward integration weakens the metaphor’s expressive force, which is consistent with our finding that Metaphoric Efficacy lags behind Perceptual Harmony across most models. In the most severe cases, models omit one of the domains entirely: Z-Image, for instance, fills the claw machine with ordinary toy balls rather than business professionals (row 3), making the metaphor impossible to recover. Together, these failure modes underscore the value of a multi-level evaluation framework for diagnosing the specific weaknesses of current T2I models.

## 4.4 Effect of Prompt Granularity

In addition to the conceptual prompts reported in our main experiments, we also generate and evaluate images using the descriptive prompts paired with each sample. Full results under descriptive prompts are provided in the Appendix A.4. Here we analyze the differences between the two prompt settings on a representative subset of five models spanning the three model categories, summarized in Table 2.

We first observe that all models benefit from descriptive prompts on MCQ accuracy, and most models also see improvements on dimension scores, indicating that explicit visual specification reduces the difficulty of metaphor generation. However, the magnitude of improvement varies dramatically across models. GPT Image 1.5 gains only +3.1 percentage points on MCQ overall and shows essentially no change on dimension score (−0.06), whereas Z-Image improves by +16.6 points and +0.69, respectively. A clear inverse relationship emerges between a model’s baseline capability and the gain it receives: stronger models exhibit smaller gains, while weaker models benefit substantially more.

This pattern suggests that capable models such as GPT Image 1.5 and Nano Banana 2 already possess the ability to reason about how a metaphorical idea should be visually composed without explicit specification. In contrast, weaker models lack this inferential capacity and rely heavily on the explicit visual guidance provided by descriptive prompts. This aligns with our earlier observation that visual metaphor generation requires more than literal rendering: it demands a degree of conceptual reasoning, on which current open-source models still fall short.

Beyond the per-model gains, descriptive prompts also narrow the capability gap between models. Under conceptual prompts, the dimension score gap between GPT Image 1.5 (4.30) and Z-Image (3.21) is 1.09; under descriptive prompts, this gap narrows to 0.34 (4.24 vs. 3.90). For practitioners working with open-source models, providing fine-grained visual descriptions is therefore an effective strategy to compensate for the limited metaphorical reasoning capability.

## 4.5 Reliability of the Evaluator

To validate the reliability of using Qwen3.5-27B as our default evaluator, we conduct two studies: a cross-judge consistency study assessing robustness to the choice of evaluator, and a human alignment study measuring agreement with human judgment.

Table 3: Cross-judge comparison. Model rankings by MCQ Overall and Dimension Overall under two judges.
<table><tr><td></td><td colspan="2">MCQ</td><td colspan="2">Dimension</td></tr><tr><td>Model</td><td>Qwen3.5</td><td>GPT-5.4</td><td>Qwen3.5</td><td>GPT-5.4</td></tr><tr><td>GPT Image 1.5</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Nano Banana 2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Seedream 5.0 Lite</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>FLUX.2-dev</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Qwen-Image</td><td>5</td><td>5</td><td>5</td><td>5</td></tr></table>

Table 4: Alignment between MLLM evaluators (Qwen3.5-27B and GPT-5.4) and human annotators on 200 sampled images. (a) MCQ alignment by balanced accuracy (%); (b) dimension alignment by mean absolute error (MAE) on a 1–5 scale.  
(a) MCQ alignment (Bal.Acc, %)  
(b) Dimension alignment (MAE)
<table><tr><td>Judge</td><td>Domain</td><td>Structure</td><td>Meaning</td><td>Mapping</td><td>Overall</td></tr><tr><td>Qwen3.5-27B</td><td>87.61</td><td>71.25</td><td>83.97</td><td>76.45</td><td>80.24</td></tr><tr><td>GPT-5.4</td><td>88.79</td><td>73.89</td><td>84.50</td><td>76.51</td><td>81.16</td></tr></table>

<table><tr><td>Judge</td><td>Efficacy</td><td>Logic</td><td>Harmony</td></tr><tr><td>Qwen3.5-27B</td><td>0.65</td><td>0.71</td><td>0.64</td></tr><tr><td>GPT-5.4</td><td>0.71</td><td>0.89</td><td>0.75</td></tr></table>

Cross-Judge Consistency. To test whether our conclusions depend on the specific choice of evaluator, we re-run the full evaluation pipeline with GPT-5.4 [31] as an alternative judge and compare model rankings and scores against those produced by Qwen3.5-27B. As shown in Table 3, the top-5 models receive identical rankings from both judges across both metrics. Over all 11 models, Spearman’s ρ reaches 0.991 for MCQ and 0.936 for dimension scores, confirming that our findings are not dependent on the choice of evaluator.

Human Alignment Study. We further evaluate how well the two MLLM judges align with human annotators. We randomly sample 100 images each from FLUX.2-dev and Qwen-Image, yielding 200 images in total. Five human annotators, trained on visual metaphor theory and our annotation guidelines, independently evaluate each image using the same protocol applied to the MLLM evaluator. We measure alignment using balanced accuracy [3] for MCQ evaluation and Mean Absolute Error (MAE) for dimension-based scoring, following established practice in recent benchmark studies [46, 55]. Results are reported in Table 4.

The MCQ alignment is consistently high across all four levels, with both Qwen3.5-27B and GPT-5.4 achieving over 80% balanced accuracy overall. Alignment is highest on the more concrete L1 (Domain) and L3 (Meaning) levels (above 83%), and lower on L2 (Structure) and L4 (Mapping), which require finer-grained judgments about how source and target are visually combined. For dimension-based scoring, Qwen3.5-27B yields a low MAE across all three dimensions (below 0.8 on a 1–5 scale), indicating that its ratings are close to those of human annotators. GPT-5.4 shows slightly larger errors (up to 0.89 on Logic) but remains within a reasonable range (< 1.0). The fact that an open-source evaluator achieves alignment comparable to or better than a leading proprietary model further supports the choice of Qwen3.5-27B as our default judge.

## 5 Conclusion

In this paper, we introduce VMetaphor-Bench, the first benchmark for evaluating visual metaphor generation in text-to-image models. Built upon 1,500 carefully curated visual metaphors with structured annotations, our benchmark provides a hierarchical taxonomy of three levels and ten categories and a hybrid evaluation framework that combines MCQ-based fidelity assessment with dimensionbased scoring. Extensive evaluation of 11 representative T2I models reveals that proprietary models substantially outperform open-source counterparts, yet even the strongest models continue to struggle with compositional structuring and cross-domain mapping—key aspects of metaphorical expression. We hope VMetaphor-Bench serves as a foundation for future research on bridging the gap between literal rendering and creative, meaning-rich visual generation.

## References

[1] Pinterest. https://www.pinterest.com.

[2] Arjun R Akula, Brendan Driscoll, Pradyumna Narayana, Soravit Changpinyo, Zhiwei Jia, Suyash Damle, Garima Pruthi, Sugato Basu, Leonidas Guibas, William T Freeman, et al. Metaclue: Towards comprehensive visual metaphors research. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 23201–23211, 2023.

[3] Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno Stephan, and Joachim M Buhmann. The balanced accuracy and its posterior distribution. In 2010 20th international conference on pattern recognition, pages 3121–3124. IEEE, 2010.

[4] ByteDance Seed. Seedream 5.0 lite. https://seed.bytedance.com/en/seedream5\_0\_ lite, 2026.

[5] Silvia Cappa, Anna Sofia Lippolis, and Stefano Zoia. Meanings are like onions: A layered approach to metaphor processing. arXiv preprint arXiv:2507.10354, 2025.

[6] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-asa-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning, 2024.

[7] Kaijie Chen, Zihao Lin, Zhiyang Xu, Ying Shen, Yuguang Yao, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. R2i-bench: Benchmarking reasoning-driven text-to-image generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12606–12641, 2025.

[8] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

[9] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[11] Francesca Ervas. Metaphor, ignorance and the sentiment of (ir) rationality. Synthese, 198(7):6789–6813, 2021.

[12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[13] Gilles Fauconnier and Mark Turner. Conceptual blending, form and meaning. Recherches en communication, 19:57–86, 2003.

[14] Charles Forceville et al. Metaphor in pictures and multimodal representations. The Cambridge handbook of metaphor and thought, pages 462–482, 2008.

[15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

[16] Google. Gemini 3.1 Flash Image Preview. https://ai.google.dev/gemini-api/docs/ models/gemini-3.1-flash-image-preview, 2026.

[17] Google DeepMind. Gemini 3.1 pro. https://deepmind.google/models/gemini/pro/, 2026.

[18] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.

[19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[21] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20406–20417, 2023.

[22] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.

[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[24] Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

[25] Girish A Koushik, Fatemeh Nazarieh, Katherine Birch, Shenbin Qian, and Diptesh Kanojia. The mind’s eye: A multi-faceted reward framework for guiding visual metaphor generation. arXiv preprint arXiv:2508.18569, 2025.

[26] Manishit Kundu, Sumit Shekhar, and Pushpak Bhattacharyya. Looking beyond the pixels: Eval uating visual metaphor understanding in vlms. Findings ofthe Associationfor Computational Linguistics: EMNLP, pages 23137–23158, 2025.

[27] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[28] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[29] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

[30] OpenAI. GPT-Image-1.5. https://developers.openai.com/api/docs/models/ gpt-image-1.5, 2025.

[31] OpenAI. Introducing gpt-5.4. https://openai.com/zh-Hans-CN/index/ introducing-gpt-5-4/, 2025.

[32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[33] Barbara J Phillips and Edward F McQuarrie. Beyond visual metaphor: A new typology of visual rhetoric in advertising. Marketing theory, 4(1-2):113–136, 2004.

[34] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[36] Stability AI. Stable diffusion 3.5 large. https://huggingface.co/stabilityai/ stable-diffusion-3.5-large, 2024.

[37] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025.

[38] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

[39] Zhida Sun, Zhenyao Zhang, Yue Zhang, Min Lu, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. Creative blends of visual concepts. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–17, 2025.

[40] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

[41] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[42] Z-Image Team. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

[43] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

[44] Margot Van Mulken, Andreu Van Hooft, and Ulrike Nederstigt. Finding the tipping point: Visual metaphor and conceptual complexity in advertising. Journal of Advertising, 43(4):333–343, 2014.

[45] Zehan Wang, Jiayang Xu, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Genspace: Benchmarking spatially-aware image generation. arXiv preprint arXiv:2505.24870, 2025.

[46] Zengbin Wang, Xuecai Hu, Yong Wang, Feng Xiong, Man Zhang, and Xiangxiang Chu. Everything in its place: Benchmarking spatial intelligence of text-to-image models. arXiv preprint arXiv:2601.20354, 2026.

[47] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.

[48] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

[49] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

[50] Yu Xu, Yuxin Zhang, Juan Cao, Lin Gao, Chunyu Wang, Oliver Deussen, Tong-Yee Lee, and Fan Tang. Beyond pixels: Visual metaphor transfer via schema-driven agentic reasoning. arXiv preprint arXiv:2602.01335, 2026.

[51] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025.

[52] Chenhao Zhang, Yazhe Niu, and Hongsheng Li. Metaphorstar: Image metaphor understanding and reasoning with end-to-end visual reinforcement learning. arXiv preprint arXiv:2602.10575, 2026.

[53] Daoan Zhang, Che Jiang, Ruoshi Xu, Biaoxiang Chen, Zijian Jin, Yutian Lu, Jianguo Zhang, Liang Yong, Jiebo Luo, and Shengda Luo. Worldgenbench: A world-knowledge-integrated benchmark for reasoning-driven text-to-image generation. arXiv preprint arXiv:2505.01490, 2025.

[54] Zicheng Zhang, Junying Wang, Farong Wen, Yijin Guo, Xiangyu Zhao, Xinyu Fang, Shengyuan Ding, Ziheng Jia, Jiahao Xiao, Ye Shen, et al. Large multimodal models evaluation: a survey. Science China Information Sciences, 68(12):221301, 2025.

[55] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

## A Appendix

## A.1 Benchmark Statistics

Table 5 summarizes the dataset statistics. Conceptual prompts are concise (mean 25.6 words), while descriptive prompts provide richer detail (mean 65.0 words); appending the style suffix roughly doubles the length in both cases. We organize the 1,500 images into 10 thematic categories across three levels: Individual (495 images) covers topics related to personal cognition, emotion, health, and growth; External (499 images) encompasses societal, environmental, technological, and interpersonal themes; and Creative (506 images) includes artistic designs and commercial advertisements. The distribution is approximately balanced across levels, with the largest single category being Creative Design (21.07%).

Table 5: Left: Prompt length statistics (word count). Right: Distribution of thematic categories.
<table><tr><td></td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>Conceptual Prompt</td><td>25.6</td><td>10</td><td>56</td></tr><tr><td>+ Style Suffix</td><td>68.4</td><td>29</td><td>133</td></tr><tr><td>Descriptive Prompt</td><td>65.0</td><td>20</td><td>142</td></tr><tr><td>+ Style Suffix</td><td>107.7</td><td>38</td><td>226</td></tr></table>

<table><tr><td>Level</td><td>Category</td><td>Count</td><td>Prop.(%)</td></tr><tr><td rowspan="5">Individual (495)</td><td>Cognitive Processes</td><td>198</td><td>13.20</td></tr><tr><td>Mental States</td><td>161</td><td>10.73</td></tr><tr><td>Healthy Living</td><td>72</td><td>4.80</td></tr><tr><td>Personal Development</td><td>64</td><td>4.27</td></tr><tr><td>Social Issues</td><td>243</td><td>16.20</td></tr><tr><td rowspan="3">External (499)</td><td>Natural Environment</td><td>119</td><td>7.93</td></tr><tr><td>Technological Life</td><td>88</td><td>5.87</td></tr><tr><td>Social Interaction</td><td>49</td><td>3.27</td></tr><tr><td>Creative (506)</td><td>Creative Design Advertising</td><td>316 190</td><td>21.07 12.67</td></tr></table>

## A.2 Experimental Setup Details

We conduct all experiments on a single server equipped with 4 × NVIDIA H200 GPUs. Detailed information about each evaluated model and MLLM evaluators is provided in Table 6.

## A.3 Per-Structure Analysis

To better understand where models struggle on the L2 Structure level, we examine the structure type predicted by the MCQ evaluation against the ground-truth structure type annotated in our dataset. Each L2 MCQ question asks the judge to identify the structure type of the generated image among fusion, replacement, juxtaposition, and cannot be determined, allowing us to construct a confusion matrix between ground-truth and predicted structure types. Aggregated results across all 11 evaluated models are shown in Figure 5.

The confusion matrix reveals a striking asymmetry across structure types. Models achieve reasonable accuracy onfusion (67.9%) and juxtaposition (69.2%), where the two domains are either merged into a single object or placed alongside each other. In contrast, replacement accuracy drops dramatically to only 18.9%. When prompted with replacement metaphors, models tend to either fuse the two domains together (37.8%) or place them as separate objects (36.6%), rather than substituting one for the other.

Table 6: Detailed information about the evaluated T2I models and MLLM evaluators.
<table><tr><td>Model</td><td>Year</td><td>Resolution</td><td>Source</td><td>Version / HF Checkpoint</td></tr><tr><td colspan="5">Closed-source T2I Models</td></tr><tr><td>Seedream 5.0 Lite 2026</td><td></td><td>2K</td><td>API</td><td> $\mathtt { d o u b a o - s e e d r e a m - 5 - 0 - 2 6 0 1 2 8 }$ </td></tr><tr><td>Nano Banana 2</td><td>2026</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>API</td><td>gemini-3.1-flash-image-preview</td></tr><tr><td>GPT Image 1.5</td><td>2025</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>API</td><td> $\mathbf { g } \mathbf { p } \mathbf { t } - \mathbf { i } \mathbf { m } \mathbf { a } \mathbf { g } \mathbf { e } - 1 . 5 \mathbf { - } 2 0 2 5 \mathbf { - } 1 2 \mathbf { - } 1 6$ </td></tr><tr><td colspan="5">Open-source T2I Models</td></tr><tr><td>FLUX.1-dev</td><td>2024</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>black-forest-labs/FLUX.1-dev</td></tr><tr><td>SD 3.5 Large</td><td>2024</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>stabilityai/stable-diffusion-3.5-large</td></tr><tr><td>Z-Image</td><td>2025</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>Tongyi-MAI/Z-Image-Turbo</td></tr><tr><td>Qwen-Image</td><td>2025</td><td> $1 3 2 8 \times 1 3 2 8$ </td><td>Checkpoint</td><td>Qwen/Qwen-Image-2512</td></tr><tr><td>FLUX.2-dev</td><td>2025</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>black-forest-labs/FLUX.2-dev</td></tr><tr><td colspan="5">Open-source Unified MLLMs</td></tr><tr><td>OmniGen2</td><td>2025</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>OmniGen2/OmniGen2</td></tr><tr><td>Bagel</td><td>2025</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td>Checkpoint</td><td>ByteDance-Seed/BAGEL-7B-MoT</td></tr><tr><td>Janus-Pro</td><td>2025</td><td> $3 8 4 \times 3 8 4$ </td><td>Checkpoint</td><td>deepseek-ai/Janus-Pro-7B</td></tr><tr><td colspan="5">MLLM Evaluators</td></tr><tr><td>Qwen3.5-27B</td><td>2026</td><td></td><td></td><td>Checkpoint Qwen/Qwen3.5-27B</td></tr><tr><td>GPT-5.4</td><td>2026</td><td></td><td>API</td><td>gpt-5.4-2026-03-05</td></tr></table>

Interestingly, this difficulty hierarchy mirrors the cognitive complexity ordering proposed in visual rhetoric theory [33], where replacement is identified as the most demanding structure type for human viewers because the missing element must be inferred from its absence. In contrast, fusion requires disentangling two co-present elements, while juxtaposition simply pairs them side by side. Our results extend this ordering from comprehension to generation: T2I models exhibit the same complexity gradient when producing visual metaphors. This suggests that the cognitive demands of replacement, which require preserving context while substituting a localized element, pose a fundamental challenge for current models. This finding points to compositional precision, the ability to perform localized, structure-aware modifications, as a key direction for advancing visual metaphor generation.

![](images/ad9d161430dd50de47a63fe33c97ec947ffb540afad0e2c074f60e3d3ef5bda7.jpg)  
Figure 5: Confusion matrix of predicted vs. ground-truth structure types, aggregated across all models.

## A.4 Full Results with Descriptive Prompts

Table 7 reports the full evaluation results under descriptive prompts, complementing the conceptual prompt results in Table 1. Most models show substantial improvements under descriptive prompts, as the detailed visual descriptions provide more explicit guidance for generation. A notable exception is GPT Image 1.5, whose dimension score remains essentially unchanged (−0.06), suggesting that the strongest proprietary models can already infer appropriate visual realizations from abstract conceptual prompts and do not require further specification. Despite these magnitude differences, the overall model ranking remains largely consistent across the two prompt types.

Table 7: Evaluation results on VMetaphor-Bench with descriptive prompts.
<table><tr><td rowspan="2">Model</td><td colspan="5">MCQ Accuracy (%)</td><td colspan="4">Dimension Score (1–5)</td></tr><tr><td>Domain</td><td>Structure</td><td>Meaning</td><td>Mapping</td><td>Overall</td><td>Efficacy</td><td>Logic</td><td>Harmony</td><td>Overall</td></tr><tr><td colspan="10">Closed-source T2I Models</td></tr><tr><td>Seedream 5.0 Lite</td><td>89.3</td><td>69.2</td><td>96.5</td><td>87.4</td><td>86.6</td><td>4.02</td><td>4.14</td><td>4.27</td><td>4.15</td></tr><tr><td>GPT Image 1.5</td><td>89.6</td><td>72.9</td><td>96.6</td><td>90.6</td><td>88.5</td><td>4.12</td><td>4.24</td><td>4.34</td><td>4.24</td></tr><tr><td>Nano Banana 2</td><td>90.8</td><td>73.6</td><td>97.1</td><td>91.2</td><td>89.2</td><td>4.06</td><td>4.21</td><td>4.28</td><td>4.18</td></tr><tr><td colspan="10">Open-source T2I Models</td></tr><tr><td>SD 3.5 Large</td><td>83.0</td><td>62.3</td><td>93.4</td><td>76.0</td><td>78.8</td><td>3.36</td><td>3.45</td><td>3.82</td><td>3.54</td></tr><tr><td>FLUX.1-dev</td><td>83.8</td><td>64.1</td><td>93.5</td><td>79.6</td><td>80.7</td><td>3.45</td><td>3.55</td><td>3.97</td><td>3.66</td></tr><tr><td>Z-Image</td><td>87.9</td><td>67.7</td><td>96.4</td><td>87.5</td><td>85.9</td><td>3.76</td><td>3.90</td><td>4.04</td><td>3.90</td></tr><tr><td>Qwen-Image</td><td>88.6</td><td>70.9</td><td>95.9</td><td>88.5</td><td>87.0</td><td>3.98</td><td>4.03</td><td>4.08</td><td>4.03</td></tr><tr><td>FLUX.2-dev</td><td>89.4</td><td>70.3</td><td>96.4</td><td>88.9</td><td>87.3</td><td>4.00</td><td>4.09</td><td>4.18</td><td>4.09</td></tr><tr><td colspan="10">Open-source Unified MLLMs</td></tr><tr><td>OmniGen2</td><td>82.6</td><td>59.9</td><td>93.2</td><td>75.3</td><td>78.0</td><td>3.07</td><td>3.26</td><td>3.71</td><td>3.35</td></tr><tr><td>Bagel</td><td>83.2</td><td>62.7</td><td>92.6</td><td>76.8</td><td>79.1</td><td>3.28</td><td>3.36</td><td>3.77</td><td>3.47</td></tr><tr><td>Janus-Pro</td><td>84.0</td><td>65.9</td><td>94.9</td><td>79.1</td><td>81.0</td><td>3.31</td><td>3.52</td><td>3.62</td><td>3.48</td></tr></table>

## A.5 Cross-Judge Results with GPT-5.4

Table 8 reports the full evaluation results using GPT-5.4 as the judge, complementing the Qwen3.5- 27B results in Table 1. Across all 11 models, GPT-5.4 produces highly consistent rankings with Qwen3.5-27B, with Spearman rank correlations of 0.991 for MCQ and 0.936 for dimension scoring. The two judges differ slightly in absolute scores, with GPT-5.4 tending to assign somewhat more lenient dimension scores. However, the relative performance ordering of models is preserved, confirming that the conclusions reported in our main experiments do not depend on the specific choice of evaluator.

Table 8: Evaluation results on VMetaphor-Bench with conceptual prompts, judged by GPT-5.4.
<table><tr><td rowspan="2">Model</td><td colspan="5">MCQ Accuracy (%)</td><td colspan="4">Dimension Score (1–5)</td></tr><tr><td>Domain</td><td>Structure</td><td>Meaning</td><td>Mapping</td><td>Overall</td><td>Efficacy</td><td>Logic</td><td>Harmony</td><td>Overall</td></tr><tr><td>Closed-source T2I Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream 5.0 Lite</td><td>93.3</td><td>66.7</td><td>96.3</td><td>73.0</td><td>82.0</td><td>3.72</td><td>3.85</td><td>3.94</td><td>3.84</td></tr><tr><td>Nano Banana 2</td><td>94.3</td><td>75.6</td><td>98.3</td><td>76.9</td><td>85.5</td><td>4.08</td><td>4.29</td><td>4.14</td><td>4.17</td></tr><tr><td>GPT Image 1.5</td><td>94.4</td><td>77.2</td><td>97.6</td><td>77.2</td><td>85.8</td><td>4.13</td><td>4.37</td><td>4.33</td><td>4.27</td></tr><tr><td>Open-source T2I Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLUX.1-dev</td><td>91.3</td><td>63.8</td><td>91.9</td><td>62.8</td><td>76.4</td><td>2.95</td><td>3.07</td><td>3.70</td><td>3.24</td></tr><tr><td>SD 3.5 Large</td><td>91.5</td><td>65.7</td><td>91.4</td><td>64.6</td><td>77.4</td><td>2.97</td><td>3.14</td><td>3.68</td><td>3.26</td></tr><tr><td>Z-Image</td><td>93.1</td><td>66.5</td><td>94.8</td><td>65.6</td><td>78.9</td><td>3.05</td><td>3.25</td><td>3.56</td><td>3.28</td></tr><tr><td>Qwen-Image</td><td>91.9</td><td>70.3</td><td>93.9</td><td>67.3</td><td>79.6</td><td>3.32</td><td>3.45</td><td>3.73</td><td>3.50</td></tr><tr><td>FLUX.2-dev</td><td>94.0</td><td>69.3</td><td>94.7</td><td>69.0</td><td>80.9</td><td>3.45</td><td>3.62</td><td>3.84</td><td>3.64</td></tr><tr><td colspan="2">Open-source Unified MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniGen2</td><td>90.2</td><td>62.0</td><td>90.9</td><td>57.6</td><td>73.7</td><td>2.51</td><td>2.65</td><td>3.42</td><td>2.86</td></tr><tr><td>Bagel</td><td>91.7</td><td>64.7</td><td>92.6</td><td>62.1</td><td>76.6</td><td>2.78</td><td>2.92</td><td>3.44</td><td>3.05</td></tr><tr><td>Janus-Pro</td><td>92.3</td><td>67.1</td><td>93.9</td><td>67.6</td><td>79.4</td><td>3.06</td><td>3.34</td><td>3.49</td><td>3.30</td></tr></table>

## A.6 Benchmark Construction Prompts.

Figure 6 and Figure 7 show the prompts for the two-stage annotation pipeline: structured annotation extraction (Stage 1) and descriptive/conceptual prompt generation (Stage 2). Figure 8 shows the prompt for MCQ distractor generation.

![](images/a53393160d892c82e389fafd53c8e023e68533ed1f6d19c78c372a064af8e617.jpg)  
Figure 6: Prompt for structured annotation generation (Stage 1).

![](images/74bdaacc3da82989a7908ed745dc9a5dd6de4cdf5163f8a5fe60d8949b0a8b5f.jpg)  
Figure 7: Prompt for descriptive and conceptual prompt generation (Stage 2).

![](images/b928d87a38f3f18a72ec235329da3e71e7f1d9242e48e62535479434200ad5a8.jpg)  
Figure 8: Prompt for MCQ distractor generation.

## A.7 Evaluation Prompts

Figure 9 and Figure 10 show the prompts for MCQ evaluation, which is split into two calls: Q1– Q4 (target, source, structure, and meaning) are answered together, while Q5 (element mapping) is answered separately to prevent domain leakage from its answer options. Specifically, Q5 (element mapping) options may reveal the source domain (e.g., a question stem mentioning ’spherical bomb casing’ would expose ’bomb’ as the source), which would compromise the independence of L1 and L4 evaluations. Figure 11 shows the prompt for dimension scoring with 1–5 Likert-scale rubrics.

![](images/dcb57fdea6ed7c1cdd1b809cf6974ea336003cadf7b96004a9d608f61377e244.jpg)  
Figure 9: Prompt for MCQ evaluation (Q1–Q4: target domain, source domain, structure type, and meaning).

![](images/d7fefccbe90e57a8fcdb7a62030af26fe04d7e562c2cc9a5bc247edd1b5301cc.jpg)  
Figure 10: Prompt for MCQ evaluation (Q5: Element mapping).

![](images/bbbc373ca70f190f59e6df65864fbaf504efd8262fd98edca6294797433265ec.jpg)  
Figure 11: Prompt for dimension score evaluation.

## A.8 Interface for Human Annotators

We provide screenshots of the annotation interfaces used in our study. Figure 12 shows the interface for manual verification during data curation (Section 3.1), where annotators review and edit metadata fields such as domains, structure type, element mappings, and meaning. Figures 13 and 14 show the two pages of the interface used in the human alignment study (Section 4.5): one for answering MCQ questions alongside the generated image, and another for rating dimension scores on a 1–5 scale.

![](images/1f174fee4623f7e9df29e30d0c3cbb68def13d2aae4f8b381372df4291ad6a2d.jpg)  
Figure 12: Manual verification interface for data curation

![](images/233c043afba78d1cd1b8aad5c31729e237b50ad75e567cd81a8798385dd53ca7.jpg)  
Figure 13: Human evaluation interface for the alignment study: MCQ answering

![](images/ed96e99ee408d0dcbf62c8ba1bde3594ba16e98bc0fe9f36efe07799b5737eff.jpg)  
Figure 14: Human evaluation interface for the alignment study: dimension rating

## A.9 Samples

We present representative samples in Figure 15. For each category, we select one visual metaphor and display the generated images from all evaluated models under the conceptual prompt setting. These examples illustrate the diversity of metaphorical themes in our benchmark and highlight the varying capabilities of different models in generating visual metaphors.

![](images/b3ea94bc6c4f77574ef33cf64633507c8ad44ba8b2f91628ece370d335b74b2f.jpg)  
Figure 15: Generated visual metaphors across ten categories from VMetaphor-Bench.

## A.10 Limitations

We acknowledge several limitations. First, the distribution of samples across thematic categories is uneven, reflecting the natural prevalence of visual metaphors in different topics in real-world creative imagery. Second, our evaluation relies on an MLLM-as-judge framework, which, despite strong alignment with human annotators, inherits the fundamental limitations of automatic visual reasoning on the most subjective judgments. Finally, our prompts and source imagery are predominantly English and from Western creative traditions, leaving multilingual prompts and cross-cultural metaphors as future directions.

## A.11 Broader Impacts

This work advances the evaluation of visual metaphor generation, a challenging aspect of text-toimage models that requires coherent cross-domain reasoning and expressive visual composition. By providing a structured benchmark with fine-grained evaluation criteria, it can support the development of more capable and interpretable generative models. Such progress may benefit real-world applications including creative design, education, and advertising, where visual metaphors are widely used to communicate messages effectively and engage audiences.

We also acknowledge potential negative implications. Improved visual metaphor generation could be misused to produce persuasive misleading imagery (e.g., manipulative advertising or political propaganda) that exploits the strong emotional and rhetorical effects of visual analogies. We encourage future work to consider safeguards against such misuse, and our benchmark itself is intended for evaluation rather than direct deployment.

## A.12 Data Licensing and Release

To respect the rights of original content creators on Pinterest, we do not redistribute the source images. Our public release consists of: (1) image URLs pointing to the original Pinterest pages, (2) all structured annotations (target/source domains, structure types, meaning, element mappings), (3) descriptive and conceptual prompts, and (4) the 9,594 multiple-choice questions. Users can retrieve images via the URLs for research purposes following Pinterest’s terms of service. The annotations and prompts are released under CC-BY-4.0.