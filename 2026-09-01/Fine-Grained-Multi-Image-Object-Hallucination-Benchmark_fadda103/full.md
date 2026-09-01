# Fine-Grained Multi-Image Object Hallucination Benchmark

Joonki Min<sup>1∗</sup>, Chaeyun Kim<sup>1,2∗</sup>, Hyungwook Choi<sup>1</sup>, Yejin Kim<sup>1</sup>, Kihyun Kim<sup>1</sup>

Yohan Jo<sup>1†</sup>, Joonseok Lee<sup>1†</sup>

<sup>1</sup>Seoul National University <sup>2</sup>AIM Intelligence

{minc69,golddohyun,chooi221,a2000yejin,ki5477,yohan.jo,joonseok}@snu.ac.kr

## Abstract

Multimodal Large Language Models (MLLMs) are increasingly deployed in multi-image scenarios requiring complex reasoning across visual contexts. However, current MLLMs remain fundamentally limited by object hallucination—generating plausible yet factually inconsistent descriptions about objects. Existing benchmarks, designed primarily for single-image settings or providing only high level multi-image assessments, cannot systematically diagnose how visual complexity and reasoning demands trigger hallucination. To address this gap, we introduce MIOH, afine-grained multi-image object hallucination benchmark that systematically evaluates object hallucination across four foundational tasks (existence, counting, attribute, position) through three multi-image reasoning patterns (comprehensive, comparative, selective) under three controlled adversarial pressures (visual context scale, perceptual difficulty, contextual bias). Through evaluation of 29 models, we reveal that even state-of-the-art systems like GPT-5 and Gemini-2.5-Pro exhibit distinct failure patterns across different reasoning patterns and tasks. Our evaluation reveals that hallucination stems not merely from perceptual failures but from integration-stage limitations when maintaining object representations across multiple images. MIOH provides a controlledframeworkfor analyzing multi-image object hallucination and serves as a critical evaluation tool for developing more reliable multimodal AI systems.

## 1. Introduction

With recent advances, Multimodal Large Language Models (MLLMs) are capable of reasoning over multiple images simultaneously. This capability requires models not only to recognize content in individual images but also to integrate and synthesize information across diverse visual inputs. Despite these advancements, however, current MLLMs remain fundamentally limited by object hallucination, where models generate plausible yet factually inconsistent descriptions about objects in the queried images.

Multi-image scenarios amplify object hallucination along two critical dimensions. First, perceptual failure : even within individual images, models struggle with visually challenging objects (small size, occlusion, contextual ambiguity) or contextually misleading scenes where co-occurring objects create false expectations. Second, information integration : multi-image contexts demand integrating and synthesizing information across images, where models must aggregate information across all images (comprehensive reasoning), identify differences between images (comparative reasoning), or retrieve specific images matching given criteria (selective reasoning). Failures in either dimension create compounding pathways for hallucination. Systematic evaluation requires isolating both: which visual conditions trigger perceptual errors, and which integration demands are most vulnerable to failure.

Despite the importance of understanding object hallucination in multi-image contexts, existing evaluation frameworks do not adequately address these two dimensions. Most object hallucination benchmarks [18, 28, 43, 47, 51] are designed for single-image scenarios with binary questions, focusing narrowly on existence and counting. This design cannot reveal how visual difficulty factors (perceptual challenges, contextual biases) systematically induce hallucination, nor can it assess multi-image reasoning patterns or compositional capabilities (attributes, spatial relations) that require fine-grained object understanding. While general multi-image benchmarks [11, 16, 26, 33, 37, 50] evaluate overall reasoning, they lack the controlled manipulation of visual factors needed to diagnose object hallucination vulnerabilities. Recent work such as MIHBench [27] explores multi-image hallucination but examines only image quantity as an adversarial factor in ablation studies, without systematically varying visual difficulty or distinguishing between reasoning patterns. Consequently, existing benchmarks cannot pinpoint which specific visual conditions or reasoning demands trigger hallucination, nor how

these factors interact.

To narrow these gaps, we introduce the Fine-Grained Multi-Image Object Hallucination (MIOH) benchmark, designed to systematically evaluate both dimensions of multiimage object hallucination. MIOH systematically integrates four object-centric tasks (existence, counting, attribute, position) with three reasoning patterns (comprehensive, comparative, selective), enabling compositional evaluation from basic detection to fine-grained property binding. Furthermore, MIOH introduces three controlled adversarial pressures—visual context scale (number of images), perceptual difficulty (small/occluded objects), and contextual bias (misleading co-occurrence priors)—that systematically vary visual complexity while measuring performance across reasoning patterns. This design enables diagnostic analysis by separately measuring performance under different visual conditions (perceptual difficulty, contextual bias, image scale) and across different reasoning patterns (comprehensive, comparative, selective).

Our contributions are summarized as follows:

• We introduce MIOH, the first benchmark to systematically assess object hallucination in multi-image contexts with fine-grained diagnostic capabilities across visual complexity and reasoning demands.

• We define three multi-image reasoning patterns (comprehensive, comparative, selective) and instantiate them across four foundational object-centric tasks (existence, counting, attribute, position), enabling diagnosis of which reasoning capability fails under hallucination pressures.

• We design three controllable adversarial pressures (visual context scale, perceptual difficulty, contextual bias) that systematically exacerbate hallucination, allowing fine-grained analysis of when and why MLLMs fail in multi-image scenarios.

## 2. Related Work

Multimodal Large Language Models (MLLMs). Following the success of LLMs, MLLMs have rapidly evolved through visual instruction tuning, utilized by LLaVA [30] and extended by InstructBLIP [8] and MiniGPT-4 [62]. Early MLLMs face challenges in cross-image reasoning due to limitations in visual token processing and inter-image semantic modeling [2, 4, 8, 21, 25]. Recent work has enabled multi-image understanding [16, 20, 24, 36, 57]; Mantis [16], LLaVA-NeXT-Interleave [24], and Idefics3 [20] leverage large-scale image-text data at training, and Qwen2.5- VL [3], InternVL3.5 [54], and Gemini-2.5-Pro [6] demonstrate powerful cross-image reasoning capabilities.

Object Hallucination in MLLMs. Object hallucination, defined as MLLMs generating plausible but inaccurate object descriptions inconsistent with visual inputs, remains a critical challenge [7, 43]. Systematic analysis has pinpointed causes across the MLLM pipeline: data-related issues such as statistical biases in training data [28], limitations of vision encoder in fine-grained semantics [48], insufficient modality alignment [31], and inherited LLM biases such as weak context attention [53]. Recent studies reveal additional visual vulnerabilities, e.g., perception of small or occluded objects [58], contextual bias due to object co-occurrence patterns [28] and semantic similarities [23]. MLLMs are also linguistically susceptible to sycophantic alignment with user beliefs and context hijacking from misleading narratives [61]. Mitigation strategies, e.g., data augmentation [44], preference optimization [59], and inference-time interventions [14, 60], have primarily targeted single image scenarios. Multi-image contexts amplify hallucination challenges, requiring models not only to recognize objects accurately, but to track them and maintain contextual consistency across distinct images [56]. This increased complexity requires a new benchmark tailored to assess object hallucination on multiple images.

Benchmarks for MLLMs. Early MLLM benchmarks focus on single-image scenarios across various tasks including visual question answering, reasoning, and compositional understanding [10, 12, 22, 29, 32]. Recently, several benchmarks [11, 16, 26, 33, 37, 50] assess general reasoning capabilities across multiple images, though none of them specifically target object hallucination.

In parallel, object hallucination has been addressed through specialized benchmarks, including discriminative approaches using binary questions [13, 28] and generative approaches directly assessing free-form descriptions [18, 43, 47, 51]. They typically focus on existence and simple counting tasks, with limited coverage of attributes or spatial relations. Also, they predominantly employ simple binary questions or captioning tasks, confined to singleimage settings. Recently, MIHBench[27] has also explored hallucination in multi-image settings. However, it focuses mainly on binary existence questions and varies only the number of images, without modeling broader visual difficulty factors or diverse object-centric tasks. Our benchmark is complementary to this direction, offering a more fine-grained diagnostic space across multiple tasks, reasoning patterns, and adversarial conditions.

## 3. Overall Design of our MIOH Benchmark

We introduce the Multi-Image Object Hallucination (MIOH) benchmark, designed to systematically evaluate object hallucination in MLLMs under multi-image contexts. MIOH is structured around two complementary dimensions: (1) multi-image reasoning patterns that probe different integration capabilities (Sec. 3.1), and (2) adversarial pressures that systematically challenge perceptual and contextual robustness (Sec. 3.3). By crossing four fundamental object-centric tasks with three reasoning patterns under varying adversarial conditions, MIOH enables finegrained diagnosis of when and why MLLMs hallucinate.

![](images/3001810ade59036ccdeea947208bd174e0570d012e67cf4eafb451f5117a87c8.jpg)  
Figure 1. Overview of our MIOH Benchmark. MIOH evaluates object hallucination in multi-image contexts across four core tasks: existence, counting, attribute, and position. Each task includes three question types (comprehensive, comparative, and selective) designed to probe different aspects of multi-image reasoning capabilities

## 3.1. Multi-Image Reasoning Patterns

Motivation. Multi-image understanding requires fundamentally different reasoning capabilities compared to single-image settings. While single-image benchmarks test whether models can recognize objects in isolation, multiimage scenarios demand that models synthesize, compare, and retrieve information across multiple visual contexts. We identify three core reasoning patterns that capture these distinct capabilities, each placing unique demands on the model’s information integration mechanisms.

Comprehensive Reasoning. requires models to aggregate information across all images to form a holistic judgment. Questions in this category ask about collective properties spanning the entire image set, such as “Does any image contain a zebra?” or “What is the total count of cars across all images?” This pattern tests the model’s ability to maintain a unified representation of information distributed across multiple scenes—a capability essential for summarization and global understanding tasks. Failures in comprehensive reasoning indicate difficulties in information aggregation or

token limitations.

Comparative Reasoning. requires models to identify differences between specific images. Questions such as “Which image contains more cars?” demand precise crossimage attention and the ability to maintain separate representations for different scenes while comparing them. This is critical for change detection, progress monitoring, and contrastive analysis. Failures suggest weaknesses in maintaining distinct object representations across contexts.

Selective Reasoning. requires models to retrieve a specific image that matches given criteria from a set of candidates. Questions like “In which image are exactly three zebras present?” test the ability to perform targeted retrieval while filtering out irrelevant information. This is essential for search and localization tasks in multi-image contexts. Failures indicate poor image-level indexing or inability to isolate relevant visual evidence from distractors.

Diagnostic Value. These three patterns are not merely different question formats—they probe distinct failure modes. Uniform failure across all patterns suggests a fundamental bottleneck in object recognition (perceptual failure). Conversely, failure in selective reasoning while succeeding in comprehensive reasoning points to a specific integration failure in retrieval and localization. This systematic evaluation allows us to diagnose which integration capability breaks down under specific adversarial conditions.

## 3.2. Object-Centric Tasks

We apply these three reasoning patterns to four fundamental object-centric tasks that represent core visual understanding capabilities: Existence (verifying object presence/absence), Counting (enumerating objects), Attribute (binding visual properties to objects, e.g., “red car”), and Position (binding spatial relationships, e.g., “dog next to a cat”). While Existence and Counting have been the primary focus of object hallucination benchmarks [5, 18, 28, 35, 43, 52], all four tasks form the foundation for assessing object-centric capabilities of MLLMs [10, 17, 32, 41, 47, 49, 50].

From Tasks to Question Types. By crossing four tasks with three reasoning patterns, we create a comprehensive evaluation space. However, not all combinations are equally meaningful or distinct. For example, in the Counting task, comprehensive reasoning (“total count across all images”) naturally leads to multiple variants depending on whether we ask for exact counts, comparisons of counts, or presence of specific quantities. Through careful design, we developed 26 distinct question types that systematically cover the task and reasoning space while ensuring each type probes a unique aspect of multi-image understanding. Tab. 1 presents the complete taxonomy with representative templates for each type. Complete examples are in Fig. 1 and Sec. C.

## 3.3. Adversarial Pressures

To systematically probe model vulnerabilities, we introduce three adversarial factors that create challenging but realistic evaluation conditions. These factors target distinct failure mechanisms: visual context scale tests integration capacity as the number of images increases, perceptual difficulty challenges feature extraction from small or occluded objects, and contextual bias probes susceptibility to misleading co-occurrence priors.

Visual Context Scale (Number of Images). Inspired by findings that MLLMs struggle to identify information across large image sets (the “Visual Haystack” problem [56]), we systematically vary the Number of input Images (NI) among {2, 4, 8, 10} for the same underlying question. This tests the model’s integration capacity—as the visual context expands with more images, models must maintain accurate object representations across an increasing number of visual scenes.

Perceptual Difficulty (Hard Positive). Small or partially occluded objects are inherently harder to detect [34, 55, 58]. We curate Hard Positive (HP) examples using two complementary approaches: (a) Rule-basedfiltering selects images where target objects have small bounding boxes or high occlusion ratios; (b) CLIP-based semantic filtering identifies cases where CLIP similarity between the image and text prompt “A photo of [object]” is abnormally low, indicating perceptual ambiguity. These examples test whether models can extract accurate features under perceptually challenging conditions—a necessary prerequisite for any downstream reasoning.

Table 1. MIOH Question Type Taxonomy. We design 26 question types by crossing 4 object-centric tasks with 3 reasoning patterns. Each type is illustrated with a template showing its structure.
<table><tr><td>Type</td><td>Template</td></tr><tr><td colspan="2">Existence</td></tr><tr><td>Comprehensive Which object appears in all images? Selective Comparative</td><td>Comprehensive Is there at least one {object } in any of these images? Comprehensive Is a {object} present in all of these images? Comprehensive Which object appears in at least one image? In which image does a {object} appear? Which of the two images contains a {object}?</td></tr><tr><td colspan="2">Which object is in Image 1 but not Image 2? Attribute</td></tr><tr><td>Comprehensive Is a “{attribute} {object }&quot; present in all images? Comprehensive Which attribute-object pair appears in one image? Comprehensive Which attribute-object pair appears in all images? Selective Comparative</td><td>Comprehensive Is a “{attribute} {object}&quot; present in any image? In which image is “{attribute} {object}&quot; found? Which of the two images “{attribute} {object}&quot; present?</td></tr><tr><td colspan="2">Which attribute-object pair is in Image 1 but not Image 2? Counting</td></tr><tr><td>Comprehensive In how many images is a “{object}&quot; present? Selective</td><td>Comprehensive What is the total number of “{object }&quot; across images? Comprehensive Which object appears {count} times across images? In which image are {count} “{object}&quot; found?</td></tr><tr><td colspan="2">In which image does the “{object }&quot; appear most/least? Position</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>Comprehensive Is a {subject} {relation} a {anchor} present in any image?</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Comprehensive Is a {subject} {relation } a {anchor} present in all images?</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Comprehensive Which object with spatial relationship appears in one image?</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Comprehensive Which object with spatial relationship appears in all images?</td></tr><tr><td>Selective</td><td></td></tr><tr><td></td><td>In which image is a {subject} {relation} a {anchor}?</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Selective</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>Where is “{subject} {relation } {anchor}&quot; present?</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Comparative</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>Which object with spatial relationship is in Image 1 but not Image 2?</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Contextual Bias (Hard Negative). Strong contextual priors can mislead models into hallucinating objects that are contextually plausible but visually absent [9, 23, 28]. For example, a kitchen scene may activate strong priors for “frying pan,” leading to false positives. We construct Hard Negative (HN) examples by: (a) estimating co-occurrence probabilities $P ( \mathrm { o b j e c t } _ { A } | \mathrm { o b j e c t } _ { B } )$ from COCO training data and selecting images containing highprobability co-occurring objects but missing the target; (b) applying CLIP-based semantic confusion to find images with high visual-text similarity despite object absence. These examples test whether models rely on contextual shortcuts rather than grounded visual evidence.

## 3.4. Implementation details

Dataset Selection. To ensure annotation quality, we use three complementary datasets: COCO-ReM [46] (existence, counting) addresses COCO’s incomplete masks and missing instances via systematic re-annotation; PACO [42] (attributes) provides standardized attribute labels across object categories; SVG [39] (spatial relationships) offers complete scene-level annotations, unlike Visual Genome [19] which averages only 1.5 relationships per subject.

![](images/cc8011f9c2862dcc2110cc9e30d8d06ce36b381cbb10cee51590bd04b71ec8ca.jpg)  
Figure 2. Representative examples from MIOH. (Top) Existence-Comprehensive questions across Easy, Hard Negative, and Hard Positive scenarios. (Bottom) Counting-Selective questions with varying difficulty levels. Additional examples are provided in Sec. C.

Benchmark Statistics. Three independent annotators validated all questions. The final benchmark comprises 3,484 questions across 11,732 images, balanced across tasks, reasoning patterns, and difficulty levels. Full construction details, filtering criteria, and statistics are in Sec. A.

## 4. Benchmark Results and Discussion

We conduct a comprehensive comparative study using our MIOH over the state-of-the-art MLLMs, including GPT-5 [45] and Gemini-2.5-Pro [6]. Among open-source models, we choose the LLaVA [24] series, the Qwen [3] series, InternVL [54], Phi-4-multimodal [1], MiniCPM-V [57], Ovis-2.5 [36], and Mantis-8B [16]. For reproducibility, the decoding temperature is set to 0 for all experiments. All experiments were conducted on four NVIDIA A6000 GPUs.

## 4.1. Overall Performance

We first present overall performance results demonstrating the extent of object hallucination challenges across different model categories and tasks. Tab. 2 reports the comprehensive evaluation results over 29 models, revealing that multiimage object hallucination remains as significant challenge, with an overall average accuracy of only 36.1%. Not surprisingly, a clear performance gap is measured between proprietary and open-source models. The leading models, Gemini-2.5 Pro (64.4%) and GPT-5 (63.1%), set the stateof-the-art, but are still far from perfect. Top-performing open-source models, such as Qwen2-VL-7B (49.1%) and

MiniCPM-V-2.6 (48.5%), demonstrate strong capabilities but lag behind the frontier models.

Beyond the overall gap, performance is shaped by three key factors—reasoning pattern, task type, and adversarial pressure—each revealing distinct failure modes, which we analyze in the following subsections.

## 4.2. Reasoning Patterns

Our analysis reveals that MLLM performance is governed not only by the task but also, fundamentally, by the required reasoning pattern. As shown in Fig. 3, performance varies significantly across Comprehensive, Comparison, and Selection patterns, demonstrating that existing object hallucination benchmarks, constrained to a few question styles, cannot properly assess object hallucination across diverse multi-image settings.

Overall, Comprehensive emerges as the most manageable reasoning pattern, with basic Existence questions yielding the highest performance. Comparing this densely high-performing row against the rest of the heatmap reveals that models are heavily biased towards simple, presenceverifying queries. Evaluating models solely within this narrow scope inevitably overestimates their true robustness.

On average, Selection(34.2%) emerges as the most challenging reasoning pattern, underperforming both Comprehensive and Comparison. This difficulty becomes especially pronounced when paired with compositional tasks such as Attribute(26.7%), indicating that models often recognize that an object-attribute pair exists within the image set but fail to identify which specific image contains it—a distinct form of grounding failure in multi-image settings.

<table><tr><td rowspan="2">Model</td><td colspan="5">Existence</td><td colspan="5">Counting</td></tr><tr><td>Easy</td><td>HN</td><td>HP</td><td>NI</td><td>Avg</td><td>Easy</td><td>HN</td><td>HP</td><td>NI</td><td>Avg</td></tr><tr><td>Overall GPT-5</td><td>62.4 91.4</td><td>51.9 88.6</td><td>51.5 78.4</td><td>30.0 5..</td><td>49.4 78.4</td><td>24.3 55.0</td><td>29.5 54.3</td><td>27.0 51.5</td><td>20.5 35.7</td><td>25.4 49.1</td></tr><tr><td>Gemini-2.5 Pro Owen2-VL-2B Qwen2.5-VL-3B</td><td>83.0 58.3 80.7</td><td>81.5 53.5 9555</td><td>80.7 45.9 55.5</td><td>32.7 44.7</td><td>75.4 47.6 60.8</td><td>64.4 21.7 23.3</td><td>66.4 23.3 26.7</td><td>59.4 18.6 19.6</td><td>40.0 21.4 17.1</td><td>57.5 21.</td></tr><tr><td>Owen2-VL-7B Qwen2.5-VL-7B LLaVA-v1.6 (Mistral-7B) LLaVA-Interleave (Qwen-0.5B) LLaVA-Interleave (Qwen-7B) LLaVA-Interleave (Qwen-7B-DPO) LLaVA-OneVision (Qwen2-0.5B-SI) LLaVA-OneVision (Qwen2-0.5B-OV)</td><td>87.3 84.0 33.0 36.3 46.7 67.7 27.7 57.3</td><td>76.0 47.0 31.5 44.0 44.0 11.5 25</td><td>76.4 73.6 36.8 28.6 65.5 55.9 37.3 33.6</td><td>30.7 28.0 18.7 - 30.7 31.3 40.0 22.0</td><td>67.5 65.4 38.9 28.8 46.7 49.7 29.1 34.1</td><td>28.9 33.3 20.6 22.8 20.6 21.1 13.9</td><td>36.2 39.7 27.6 28.4 25.0 25.9 21.6</td><td>21.6 17.5 18.6 24.7 28.9 30.9 18.6</td><td>17.1 21.4 14.3 - 24.3 27.1 12.9</td><td>26.0 28.0 22.2 22.6 24.7 26.3 16.7</td></tr><tr><td>LLaVA-OneVision (Qwen2-7B-SI) LLaVA-OneVision (Qwen2-7B-OV) LLaVA-OneVision (Qwen2-7B-OV-Chat) InternVL3.5-1B InternVL3.5-2B InternVL3.5-4B InternVL3.5-8B Pretrained InternVL3.5-8B Instruct</td><td>68.3 83.3 82.7 44.0 51.7 59.0 72.0 76.0</td><td>78.0 77.0 41.0 37.5 71.0 62.0 59.5</td><td>40.9 57.7 43.6 30.5 36.8 60.0 595.5.5</td><td>36.0 24.7 26.7 42.0 26.0 22.7 30.0 30.7</td><td>47.9 60.9 57.5 39.4 38.0 53.2 55.9 60.4</td><td>16.7 183 20.0 21.7 15.0 18.9 22.8 25.6</td><td>26.7 24.1 28.4 30.2 21.6 28.4 27.6 26.7</td><td>16.5 25.8 29.9 27.8 19.6 16.5 34.0 27.8 29.9</td><td>18.6 25.7 18.6 21.4 21.4 17.1 17.1 14.3</td><td>19.6 23.2 23.8 24.9 21.1 19.3 24.4 22.9</td></tr><tr><td>InternVL3.5-8B MPO InternVL3.5-8B Mantis-8B (CLIP-Llama3) Mantis-8B (SIGLIP-Llama3) MiniCPM-Llama3-V-2.5 MiniCPM-V-2.6 Ovis2.5-2B Ovis2.5-9B</td><td>76.7 34.0 45.7 47.0 35.0 86.3 72.0 78.0</td><td>60.0 55.5 57.0 45.0 16.5 75.5 23.5 23.0</td><td>76.4 34.1 36.8 42.7 25.5 73.6 47.3 51.8</td><td>18.0 20.0 18.7 24.7 4.7 38.7 34.0 28.7</td><td>57.8 35.9 39.5 39.8 208.8. 44.2 45.4</td><td>26.1 30.0 22.8 20.6 4.4 21.1 23.9 28.9</td><td>29.3 29.3 37.1 25.9 25.0 5.2 35.3 25.9</td><td>28.9 32.0 21.6 33.0 8.2 38.1 18.6</td><td>8.6 10.0 11.4 30.0 22.9 5.7 18.6 25.7</td><td>23.3 23.6 27.6 25.1 25.4 5.9 28.3 23.5</td></tr><tr><td>Phi-4-multimodal Model</td><td colspan="4">45.3 38.0 33.6 Attribute</td><td colspan="4">25. 1755 35.1 25.0 Position</td><td>25.7 28.6</td><td>32.7 24.0</td></tr><tr><td>Overall GPT-5</td><td>Easy 37.5</td><td>HN 32.9</td><td>HP NI 31.3 26.9</td><td>Avg 32.4</td><td>Easy 46.9</td><td>HN 35.2</td><td>HP 45.2</td><td>NI 22.2</td><td>Avg 37.3</td><td>Overall Avg 36.1</td></tr><tr><td>Gemini-2.5 Pro Qwen2-VL-2B Qwen2.5-VL-3B</td><td>65.8 62.9 42.3</td><td>57.4 </td><td>57.0 45.. 64.3 50.0 25.0</td><td>57.8 37.9</td><td>8.5. 46.2</td><td>70.3 38.4</td><td>75.5 74.4</td><td>34.1 37.6 22.0</td><td>67.1 66.6 45.3</td><td>63.4 38.0</td></tr><tr><td>Qwen2-VL-7B Qwen2.5-VL-7B LLaVA-v1.6 (Mistral-7B) LLaVA-Interleave (Qwen-0.5B) LLaVA-Interleave (Qwen-7B) LLaVA-Interleave (Owen-7B-DPO)</td><td>58.8 52.7 30.4 2121 28.5 22.3</td><td>2O</td><td>47.1 26.7 44.3 28.7 38.1 24.3 17.5 31.4 1 29.0 15.8 28.6 13.3</td><td>40.0 3999 25.4 30.5 24.9</td><td>0 3 29.2 15.0 31.2</td><td>50.4 56.8 52.4 25.2 127.</td><td>72.1 7055 31.2 40.9 32.1</td><td>22.0 21 21.7 34.5 21.4</td><td>54.6 61.1 52.6 28.5 27.0 27.4</td><td>44.3 49.1 46.5 28.8 27.2 30.9 32.2</td></tr><tr><td>LLaVA-OneVision (Qwen2-0.5B-SÍ) LLaVA-OneVision (Qwen2-0.5B-OV) LLaVA-OneVision (Qwen2-7B-SI) LLaVA-OneVision (Qwen2-7B-OV) LLaVA-OneVision (Qwen2-7B-OV-Chat) InternVL3.5-1B InternVL3.5-2B InternVL3.5-4B InternVL3.5-8B Pretrained InternVL3.5-8B Instruct</td><td>23.1 24.2 30.4 30.8 28.8 40.4 42.3 40.8 38.1</td><td>17.9 17.1 2 24.2 24.2 37.9 3735</td><td>27.1 12.555 28.1 273 28.3 25 26.7 27.1 17.5 15.2 35.0 34.8 35.8 19.5 38.3 18.1 39.2</td><td>24.8 20.0 22.7 24.5 26.4 26.2 24.4 32.1 38.0 34.0</td><td>37.1 18.3 1872 47.9 48.3 57.1 60.0 31.7 37.5</td><td>26.4 15.6 21.2 30.4 24.8 25.6 37.2 43.6 18.8 32.0</td><td>30.2 34.9 33.5 42.7 43.3 25.1 13.5 57.2 37.2</td><td>18.5 16.8 22.9 2318 32.7 18.4 19.2 28.1</td><td>28.1 21.4 24.2 34.4 34.4 34.5 38.0 33.9 31.7 33.7</td><td>2 2 32.5 36.4 35.8 30.7 30.8 36.8 36.6</td></tr><tr><td>InternVL3.5-8B MPO InternVL3.5-8B Mantis-8B (CLIP-Llama3) Mantis-8B (SIGLIP-Llama3) MiniCPM-Llama3-V-2.5 MiniCPM-V-2.6 Ovis2.5-2B Ovis2.5-9B Phi-4-multimodal</td><td>37.7 34.6 24.6 26.2 542 39.6 34.6</td><td>242 232 27.9 26.7</td><td>14.8 39.2 28.6 37.5 24.8 20.0 26.2 20.0 28.6 明 33.8 34.3 33.8 26.7 21.0 35.0</td><td>32.5 33.6 32.0 23.5 25.4 24.9 38.2 31.3 32.0 29.3</td><td>3.. 55.8 35. 21.8 44.6 79.6 35.0 325</td><td>33.2 34.0 36.4 25. 44.8 62.0 20.8 24.4 38.4</td><td>37.7 20.0 35.8 31.6 28.8 53.5 69.3 40.0 39.5</td><td>18.85 35.0 20.6 19.5 15.0 2310 37.6 22.7</td><td>272 40.8 25.8 25.6 39.5 59.0 31.7 39.4 35.8</td><td>36.9 35.5 34.1 28.5 29.0 22.7 48.5 32.7 37.3</td></tr></table>

Table 2. MLLM performance on MIOH benchmark. HN: Hard Negatives, HP: Hard Positives, NI: Number of Images.

![](images/59fb5991d26bd2c0467c272940fefc9b1cc99df6cff15ee779ca94ba6e3f82d1.jpg)  
Figure 3. Performance heatmap across multi-image reasoning patterns (Comprehensive, Comparison, Selection) and object-centric tasks (Existence, Counting, Attribute, Position).

## 4.3. Task-Specific Vulnerability

We conduct detailed analyses to identify specific failure patterns and characterize how different adversarial pressures affect model behavior in multi-image scenarios. As reported in Tab. 2, the overall task-level accuracy reveals a clear hierarchy of difficulty. Existence (49.4%) emerges as the most manageable task for most models, suggesting a basic level of object recognition. In stark contrast, counting (25.4%) is a critical failure point across the board, highlighting a fundamental weakness in quantitative reasoning under multi-image contexts. Attribute (32.4%) and position (37.3%) tasks reveal significant difficulty as well, underscoring the challenges of compositional understanding.

Existence. While Existence is the highest-scoring task, model performance is brittle. The accuracy on easy samples (62.4%) consistently drops on perceptually challenging objects (HP, 51.5%) or on contextually misleading scenes (HN, 51.9%). This indicates that models’ understanding of object presence is heavily reliant on clear, unambiguous visual cues and can be easily disrupted, leading to false negative hallucinations.

Counting. The average accuracy of 25.4% confirms that counting is a core deficiency. Our sub-task analysis reveals this is not limited to one failure mode; models struggle with both aggregating counts across images and locating an image with a specific number of objects. This points to a fundamental inability in quantitative grounding, especially on multiple scenes.

Attribute and Position. These tasks require binding objects to their properties or spatial relations, a challenge of compositional reasoning. The low accuracy in the attribute task (32.4%) suggests that MLLMs may recognize an object and an attribute separately but fail to confirm their visual cooccurrence (e.g., seeing a ‘car’ and ‘red’ but hallucinating a ‘red car’). Similarly, the position task scores slightly higher (37.3%), but degrades under Hard Negative pressure, indicating that models often tend to overlook complex spatial relationships beyond mere existence.

## 4.4. Impact of Adversarial Pressures

As reported in Tab. 2, introducing adversarial constraints consistently degrades MLLM performance across all tasks relative to their easy baselines.

Perceptual and Contextual Pressures (HP & HN). Performance on Existence, Attribute, and Position consistently drops under both HP and HN conditions. For instance, Attribute accuracy falls from 37.5% to 31.3% (HP), and Position from 46.9% to 35.2% (HN), confirming model vulnerabilities to misleading visual and contextual cues. In contrast, Counting exhibits only a marginal decline. However, this reflects a floor effect rather than genuine robustness; with baseline counting accuracy already near random chance (25.4%), added complexities leave little room for further degradation.

Cognitive Load from Multiple Images (NI). While HP and HN induce moderate decays, increasing the number of input images to 8 (NI) triggers a far more severe collapse. Under this heightened cognitive load, even the fundamental Existence task plummets to 30.0%—a relative drop of exceeding 50% from the easy baseline. Although top-tier models like GPT-5 and Gemini 2.5 Pro manage this expanded context better than average, their substantial performance drops confirm that scaling multi-image reasoning remains a critical open challenge.

![](images/2805c128d590eb81d7a1a8ae5e96b2ecddf3d8b25b0beb22546e94044a4aa6ce.jpg)  
Figure 4. Accuracy on easy questions and robustness under adversarial conditions in VLMs. Circle size indicates model size, where Gemini and GPT size is based on estimation.

## 4.5. Accuracy-Robustness Trade-off

As shown in Fig. 4, our analysis uncovers a fundamental trade-off between achieving high accuracy on straightforward tasks and maintaining robust performance under adversarial pressures. In other words, a strong baseline performance does not guarantee robustness against object hallucination. Open-source models with higher baseline accuracy often exhibit greater vulnerability to adversarial conditions. The correlation analysis reveals a moderate positive relationship between model size and performance on easy questions, but virtually no correlation between size and robustness, suggesting that simply scaling model parameters does not inherently improve resilience to object hallucination. The top-performing open-source models on easy questions—Qwen2-VL-7B and Qwen2.5-VL-7B—experience substantial robustness drops of -27.2% and -21.8%, respectively, indicating that high capability models might be more susceptible to the adversarial pressures we designed. This finding suggests that the ability to handle straightforward multi-image tasks does not guarantee robustness against object hallucination.

## 4.6. Multi-Image Context as a Hallucination Amplifier: An Ablation Study

To isolate the impact of multi-image processing on object hallucination, we conduct a controlled ablation study focusing on the Existence task. Specifically, we compare two evaluation approaches for identical visual content: comprehensive questions that require synthesizing information across all images simultaneously (“Is there an OBJECT in any of these IMAGEs?”) vs. decomposed questions that mirror the traditional single-image setting by asking each image separately (“Is there a OBJECT in IMAGE?”) and combining the answers to determine overall presence. This design isolates whether multi-image contexts introduce systematic errors beyond simple accumulation of individua image processing mistakes.

![](images/11213a97eb750139a2fe9c793bf71e4e2ed7d180419f80cf7adb115cec5a7a6f.jpg)  
Figure 5. Multi-image processing consistently degrades object hallucination across all models. Comparison of accuracy between multi-image comprehensive (red bars) and single-image decomposed (blue bars) evaluation on Existence task.

The results in Fig. 5 reveal that single-image processing substantially outperforms simultaneous multi-image analysis, consistently across all models and scales. This indicates that multi-image contexts significantly amplify object hallucination beyond what error accumulation alone would predict, pointing to cross-image integration as a primary source of failure. The consistency of this penalty suggests that current MLLM training paradigms fail to adequately address object hallucination in multi-image reasoning.

## 5. Conclusion

We introduce MIOH, a benchmark that systematically evaluates object hallucination in multi-image contexts across four object-centric tasks (Existence, Counting, Attribute, Position) with three reasoning patterns (Comprehensive, Comparative, Selective) under three adversarial pressures (Visual Context Scale, Perceptual Difficulty, Contextual Bias). Through evaluation of 29 models, we demonstrate that current MLLMs—including GPT-5 and Gemini-2.5- Pro—exhibit fundamental limitations in multi-image integration. Our fine-grained diagnostic framework reveals that hallucination stems not merely from perceptual failures but from integration-stage breakdowns when maintaining object representations across multiple images. MIOH provides a foundation for targeted improvements in multiimage understanding and serves as a critical evaluation tool for developing more reliable multimodal AI systems.

## Acknowledgments

This work was also supported by the SOFT Foundry Institute at SNU, Samsung Electronics, Youlchon Foundation, National Research Foundation of Korea (NRF) grants (RS-2021-NR05515, RS-2024-00336576, RS-2023-0022663, RS-2025-25399604, RS-2024-00333484), and the Institute for Information & Communication Technology Planning & Evaluation (IITP) grants (RS-2022-II220264, RS-2024- 00353131) funded by the Korean government.

## References

[1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv:2503.01743, 2025. 5

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 2

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv:2502.13923, 2025. 2, 5

[4] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In CVPR, 2024. 2

[5] Xuweiyi Chen, Ziqiao Ma, Xuejun Zhang, Sihan Xu, Shengyi Qian, Jianing Yang, David Fouhey, and Joyce Chai. Multi-object hallucination in vision language models. In NeurIPS, 2024. 4

[6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv:2507.06261, 2025. 2, 5

[7] Wenliang Dai, Zihan Liu, Ziwei Ji, Dan Su, and Pascale Fung. Plausible may not be faithful: Probing object hallucination in vision-language pre-training. arXiv:2210.07688, 2022. 2

[8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. InstructBLIP: Towards general-purpose visionlanguage models with instruction tuning. In NeurIPS, 2023. 2

[9] Shounak Datta and Dhanasekar Sundararaman. Evaluating hallucination in large vision-language models based on context-aware object similarities. arXiv:2501.15046, 2025. 4

[10] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evalu-

ation benchmark for multimodal large language models. arXiv:2306.13394, 2023. 2, 4

[11] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024. 1, 2

[12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Ba tra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017. 2

[13] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. HallusionBench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In CVPR, 2024. 2

[14] Yixiao He, Haifeng Sun, Pengfei Ren, Jingyu Wang, Huazheng Wang, Qi Qi, Zirui Zhuang, and Jing Wang. Evaluating and mitigating object hallucination in large visionlanguage models: Can they still see removed objects? In NAACL, 2025. 2

[15] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. i

[16] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. MANTIS: Interleaved multiimage instruction tuning. arXiv:2405.01483, 2024. 1, 2, 5

[17] Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. Faithscore: Fine-grained evaluations of hallucinations in large vision-language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5042– 5063, 2024. 4

[18] Prannay Kaul, Zhizhong Li, Hao Yang, Yonatan Dukler, Ashwin Swaminathan, CJ Taylor, and Stefano Soatto. Throne: An object-based hallucination benchmark for the free-form generations of large vision-language models. In CVPR, 2024. 1, 2, 4

[19] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 5, i

[20] Hugo Laurenc¸on, Andres Marafioti, Victor Sanh, and´ Leo Tronchon. Building and better understanding´ vision-language models: insights and future directions. arXiv:2408.12637, 2024. 2

[21] Hyun Lee, Hyemin Jeong, Yejin Kim, Hyungwook Choi, Hyunsoo Cho, Sookyung Kim, and Joonseok Lee. A more word-like image tokenization for mllms. In CVPR, 2026. 2

[22] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv:2307.16125, 2023. 2

[23] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evalu ating vision-language models on natural adversarial samples. In NeurIPS, 2024. 2, 4

[24] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. LLaVA-NeXTinterleave: Tackling multi-image, video, and 3D in large multimodal models. arXiv:2407.07895, 2024. 2, 5

[25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 2

[26] Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Fine-tuning multimodal LLMs to follow zero-shot demonstrative instructions. arXiv:2308.04152, 2023. 1, 2

[27] Jiale Li, Mingrui Wu, Zixiang Jin, Hao Chen, Jiayi Ji, Xiaoshuai Sun, Liujuan Cao, and Rongrong Ji. MIHBench: Benchmarking and mitigating multi-image hallucinations in multimodal large language models. In ACM MM, 2025. 1, 2

[28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv:2305.10355, 2023. 1, 2, 4

[29] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023. 2

[30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2

[31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 2

[32] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? In ECCV, 2024. 2, 4

[33] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. Advances in Neural Information Processing Systems, 37: 8698–8733, 2024. 1, 2

[34] Zhaochen Liu, Kaiwen Gao, Shuyi Liang, Bin Xiao, Limeng Qiao, Lin Ma, and Tingting Jiang. Beyond the visible: Benchmarking occlusion perception in multimodal large language models. arXiv:2508.04059, 2025. 4

[35] Holy Lovenia, Wenliang Dai, Samuel Cahyawijaya, Ziwei Ji, and Pascale Fung. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. arXiv:2310.05338, 2023. 4

[36] Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, et al. Ovis2.5 technical report. arXiv:2508.11737, 2025. 2, 5

[37] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. arXiv:2408.02718, 2024. 1, 2

[38] Yannic Neuhaus and Matthias Hein. RePOPE: Impact of annotation errors on the POPE benchmark. arXiv:2504.15707, 2025. i

[39] Jae Sung Park, Zixian Ma, Linjie Li, Chenhao Zheng, Cheng-Yu Hsieh, Ximing Lu, Khyathi Chandu, Quan Kong, Norimasa Kobori, Ali Farhadi, et al. Synthetic visual genome. In CVPR, 2025. 5, i

[40] Genevieve Patterson and James Hays. COCO attributes: Attributes for people, animals, and objects. In ECCV, 2016. i

[41] Han Qiu, Jiaxing Huang, Peng Gao, Qin Qi, Xiaoqin Zhang, Ling Shao, and Shijian Lu. LongHalQA: Long-context hal lucination evaluation for multimodal large language models. arXiv:2410.09962, 2024. 4

[42] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Mar quez, Rama Kovvuri, Abhishek Kadian, et al. PACO: Parts and attributes of common objects. In CVPR, 2023. 4, i

[43] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. arXiv:1809.02156, 2018. 1, 2, 4

[44] Pritam Sarkar, Sayna Ebrahimi, Ali Etemad, Ahmad Beirami, Sercan O Arık, and Tomas Pfister. Mitigating ob-<sup>¨</sup> ject hallucination in mllms via data-augmented phrase-level alignment. arXiv:2405.18654, 2024. 2

[45] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 5

[46] Shweta Singh, Aayan Yadav, Jitesh Jain, Humphrey Shi, Justin Johnson, and Karan Desai. Benchmarking object de tectors with COCO: A new path forward. In ECCV, 2024. 4, i

[47] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv:2309.14525, 2023. 1, 2, 4

[48] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, 2024. 2

[49] Andres Villa, Juan L´ eon, Alvaro Soto, and Bernard Ghanem.´ Behind the magic, merlim: Multi-modal evaluation bench mark for large image-language models. In CVPR, 2025. 4

[50] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv:2406.09411, 2024. 1, 2, 4

[51] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, et al. AMBER: An LLM-free multidimensional benchmark for mllms hallucination evaluation. arXiv:2311.07397, 2023. 1, 2

[52] Lei Wang, Jiabang He, Shenshen Li, Ning Liu, and Ee-Peng Lim. Mitigating fine-grained hallucination by fine-tuning large vision-language models with caption rewrites. In In ternational Conference on Multimedia Modeling, 2024. 4

[53] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. CogVLM: Visual expert for pretrained language models. In NeurIPS, 2024. 2

[54] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv:2508.18265, 2025. 2, 5

[55] Wenbo Wei, Jun Wang, and Abhir Bhalerao. Coco-olac: A benchmark for occluded panoptic segmentation and image understanding. In ICASSP, 2025. 4

[56] Tsung-Han Wu, Giscard Biamby, Jerome Quenum, Ritwik Gupta, Joseph E Gonzalez, Trevor Darrell, and David M Chan. Visual haystacks: A vision-centric needle-in-ahaystack benchmark. arXiv:2407.13766, 2024. 2, 4

[57] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. MiniCPM-V: A GPT-4V level mllm on your phone. arXiv:2408.01800, 2024. 2, 5

[58] Jiarui Zhang, Jinyi Hu, Mahyar Khayatkhoei, Filip Ilievski, and Maosong Sun. Exploring perceptual limitation of multimodal large language models. arXiv:2402.07384, 2024. 2, 4

[59] Mengxi Zhang, Wenhao Wu, Yu Lu, Yuxin Song, Kang Rong, Huanjin Yao, Jianbo Zhao, Fanglong Liu, Haocheng Feng, Jingdong Wang, et al. Automated multi-level preference for mllms. NeurIPS, 2024. 2

[60] Linxi Zhao, Yihe Deng, Weitong Zhang, and Quanquan Gu. Mitigating object hallucination in large vision-language models via image-grounded guidance. In ICML, 2025. 2

[61] Yunpu Zhao, Rui Zhang, Junbin Xiao, Changxin Ke, Ruibo Hou, Yifan Hao, Qi Guo, and Yunji Chen. Towards analyzing and mitigating sycophancy in large vision-language models. arXiv:2408.11261, 2024. 2

[62] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing visionlanguage understanding with advanced large language models. arXiv:2304.10592, 2023. 2

# Fine-Grained Multi-Image Object Hallucination Benchmark

Supplementary Material

## A. Detailed Benchmark Construction

In this section, we provide comprehensive details on our benchmark construction process, including: the curation and selection criteria for our datasets (Sec. A.1), the metadata construction methodology (Sec. A.2), the automated question generation framework (Sec. A.3), and the adversarial benchmark design strategy (Sec. A.4).

## A.1. Dataset Curation

## A.1.1. COCO-ReM

Annotation Quality Issues in Original COCO. The original COCO dataset annotations contain significant gaps that make them unsuitable for reliable object hallucination evaluation. These issues include incomplete object masks, missing instances, and inaccurate bounding boxes that would introduce systematic errors in our rule-based question generation framework.

COCO-ReM Improvements. COCO-ReM (Refined Masks) [46] addresses these limitations through a comprehensive re-annotation process: (1) Mask boundary refinement using the Segment Anything Model (SAM) to improve precision, (2) Missing instance detection using advanced detection models to identify previously unlabeled objects, (3) Label correction through systematic review and human validation, and (4) Enhanced object masks and bounding boxes providing more complete scene coverage.

Impact on Benchmark Quality. As demonstrated in Re-POPE [38], high-reliability annotations significantly impact ground truth accuracy, making this a crucial consideration for benchmark design. The enhanced annotation quality in COCO-ReM ensures our existence and counting questions have reliable ground truth labels, substantially reducing false negatives that could arise from missed objects in original COCO annotations.

Object Count Limitations. During validation, we observed that even COCO-ReM’s accuracy degrades when object counts exceed certain thresholds. Specifically, images containing more than 10 objects showed decreased annotation reliability. To maintain benchmark integrity, we implemented a conservative approach by limiting counting questions to images with 5 or fewer objects, ensuring high reliability through validated annotations while preserving sufficient complexity for meaningful MLLM evaluation.

## A.1.2. PACO

Limitations of Existing Attribute Datasets. While various datasets address object attributes, they suffer from systematic limitations: (1) Original COCO annotations lack standardized attribute labeling across object categories, (2) COCO Attributes [40] provides standardized annotation but suffers from limited diversity in both object categories and attribute types, and (3) Insufficient coverage for comprehensive benchmark construction requiring comparison across diverse objects and attributes.

PACO’s Comprehensive Approach. PACO (Parts and Attributes of Common Objects) [42] provides a superior solution through: (1) Broader category coverage spanning a more diverse range of object types, (2) Systematic attribute annotation ensuring consistency across identical objects, (3) Detailed annotation process that identifies constituent object parts and labels their diverse attributes, and (4) Largescale structured dataset resulting in comprehensive finegrained object understanding capabilities.

Advantages for Question Generation. PACO’s structured approach offers several key benefits: systematic attribute labeling with sufficient scale and diversity to support robust question generation, extensive object-attribute combinations enabling comprehensive evaluation across diverse visual scenarios, standardized annotation framework ensuring consistent evaluation criteria across different object categories, and high-quality ground truth reducing ambiguity in attribute-based question validation.

## A.1.3. SVG

Limitations of Existing Spatial Relation Datasets. Existing datasets for spatial relationship evaluation suffer from critical annotation gaps: Visual Genome [19] and GQA [15] provide relation data but have incomplete spatial relationship coverage, missing relationships in ground truth annotations that exist visually but are not labeled, and annotation inconsistencies that reduce reliability for systematic evaluation.

SVG’s Multifaceted Approach. SVG (Synthetic Visual Genome) [39] addresses these limitations through comprehensive methodology: object detection integration for accurate entity identification, scene graph enhancement to capture missing relationships, region descriptions providing contextual relationship validation, depth information enabling more accurate spatial reasoning, region masks for precise relationship localization, VQA-based verification for non-spatial relationships to ensure annotation quality, and systematic filtering to reduce incorrect relationship annotations.

Key Advantages for Spatial Evaluation. SVG provides several critical improvements: (1) Richer spatial relation coverage per subject compared to existing datasets, enabling more comprehensive spatial reasoning evaluation, (2) Comprehensive filtering that systematically reduces incorrect relationships, improving ground truth reliability, (3) Region mask-based verification enabling more reliable relationship identification through visual evidence, and (4) High relation density minimizing the critical impact of missing positional relationships on question accuracy. These enhancements make SVG particularly well-suited for generating position-based questions that can reliably assess MLLM spatial reasoning capabilities in multi-image contexts, where accurate relationship identification becomes even more challenging due to increased visual complexity.

## A.2. Metadata Construction

Having established our data sources, we now detail the metadata construction process that enables efficient question generation.

Hierarchical Organization Structure. Our metadata follows a systematic three-level organization: (1) Task-specific property categorization where objects are categorized by relevant attributes, relations, or counts, (2) Difficulty level classification with Easy/Hard Negative/Hard Positive assignments based on visual and semantic complexity, and (3) Image identifier mapping where specific image IDs are linked to categorized objects for efficient retrieval.

Rule-Based Filtering Criteria. We implement several filtering mechanisms: minimum bounding box size requirements to ensure object visibility, occlusion level thresholds based on mask overlap calculations, and image resolution considerations for consistent object detectability across different image qualities. For difficulty classification, we define easy positives/negatives as clear, unambiguous cases with high visibility and minimal contextual confusion, hard positives as present objects with small size, high occlusion, or minimal contextual cues, and hard negatives as absent objects in contexts with high co-occurrence bias or semantic similarity.

CLIP-Based Semantic Similarity Implementation. Our similarity score calculation involves text prompt generation using standardized formats (“A photo of [object]”,“[attribute] [object]”), image encoding through CLIP visual encoder, cosine similarity computation between text and image embeddings, and threshold determination through empirical validation on representative samples. This metadata system enables rapid question synthesis while maintaining quality through automated filtering based on rule-based criteria, semantic validation using CLIP similarity scores, systematic difficulty categorization across different visual reasoning scenarios, and efficient question generation through pre-computed metadata lookup.

## A.3. Question Generation

Using the prepared positive and negative samples per task, we generate multi-image questions that instantiate the three types of questions (comprehensive, comparative, and selective) across multi-image contexts for each of the four core tasks. As all ingredients are ready, this step can be easily automated by rule-based templates; e.g., for Counting, we construct a question by randomly selecting N IMAGEs containing a particular OBJECT, and the sum of COUNT per each example is the right answer. Other tasks follow a similar procedure to generate the question and correct answer. All questions are constructed in multiple-choice (MCQ) format by incorporating incorrect answers. We also include the “None of the above” options to more precisely assess the model’s understanding.

## A.4. Adversarial Example Construction

Building upon the adversarial pressures described in Sec. 3.3, we detail the implementation procedures for constructing challenging evaluation examples.

## A.4.1. Hard Positive Example

We employ two complementary filtering approaches to identify perceptually difficult positive examples: Rulebased filtering selects (IMAGE, OBJECT) pairs where the target object exhibits challenging visual characteristics. We filter based on bounding box dimensions relative to image size, segmentation mask area indicating occlusion levels, and spatial positioning within the image frame. This approach captures objects that are inherently difficult to perceive due to size or visibility constraints. CLIP-based semantic filtering identifies cases where visual-semantic alignment is weak. We compute similarity scores between image embeddings and text prompts formatted as “A photo of OBJECT” using CLIP. Examples with unexpectedly low similarity scores despite object presence indicate perceptual ambiguity—such as unusual viewpoints, partial visibility, or atypical visual contexts that challenge standard recognition patterns.

## A.4.2. Hard Negative Example

We construct misleading negative examples through two strategies: Co-occurrence-based selection leverages statistical patterns from COCO training data. We compute pairwise co-occurrence probabilities $P ( \mathrm { o b j e c t } _ { A } | \mathrm { o b j e c t } _ { B } )$ across all object categories. For a target object, we select images containing frequently co-occurring objects while the target itself is absent, creating scenarios where strong contextual priors may mislead models into false positive predictions. CLIP-based semantic confusion identifies images that exhibit high visual-semantic similarity with target object prompts despite the object’s absence. We compute CLIP similarity between images and “A photo of OBJECT” prompts for absent objects, selecting cases with high similarity scores. These examples represent strong false association triggers where visual context strongly suggests object presence without actual visual evidence.

![](images/3359bfbc72356bfc09aa5883742654d4f731f8a0d83b13b4531d6cdfb92a7b0c.jpg)  
Figure I. Aggregated Performance Heatmap across Task, Reasoning Pattern, and Adversarial Pressure Dimensions.

## A.4.3. Difficulty Level Integration.

During question generation, we systematically control difficulty by varying the proportion of challenging examples. For a question requiring N images, we can include anywhere from 0 to N hard positive or hard negative examples, with difficulty increasing as the proportion of challenging examples approaches N.

## A.4.4. Quality Assurance and Validation

To ensure benchmark reliability, we conducted comprehensive manual validation and systematic sampling to construct a balanced evaluation set. Our automated generation pipeline initially produced over 26,000 questions across all task types and reasoning patterns.

Each question underwent independent review by three annotators with expertise in computer vision and visual reasoning tasks. Annotators assessed: (1) ground truth correctness based on visual evidence, (2) question clarity and lack of ambiguity, (3) validity and distinctiveness of multiple choice options, and (4) consistency with source dataset annotations. Disagreements were resolved through majority voting, with persistent ambiguities leading to question removal.

Despite carefully selecting well-annotated datasets for construction, the validation process revealed that certain task types were still particularly prone to systemic issues. Position questions frequently exhibited ambiguous or underspecified spatial relationships, often stemming from incomplete or inconsistent relationship annotations in the source datasets. Similarly, Counting questions continued to suffer from missed instances or miscounted objects, indicating that even high-quality datasets contain non-trivial annotation noise for fine-grained quantitative reasoning. After validation and filtering, approximately 20,000 questions remained as the verified question pool.

To ensure fair and comprehensive evaluation across all dimensions, we applied stratified sampling to construct the final benchmark with balanced distribution across three key axes: Task Types, Reasoning Patterns, and Adversarial Pressures. This sampling procedure resulted in the final benchmark comprising 3,484 questions across 11,732 images, ensuring both high-quality ground truth through rigorous validation and fair evaluation coverage across all benchmark dimensions.

## B. Detailed Performance Analysis

In this section, we present a comprehensive performance analysis of various multimodal large language models (MLLMs). Before delving into the complex interactions between different factors, we first examine the aggregated performance of models across three primary dimensions: Task, Reasoning Pattern, and Adversarial Pressure. Fig. I illustrates the performance overview. This visualization allows for a direct comparison of how different models handle specific types of challenges independently.

## B.1. Variation Analysis

Fig. II presents marginal performance distributions across each dimension, illustrating their capacity to distinguish model capabilities. Among task types, Position tasks demonstrate the highest variance (σ = 0.163), suggesting that models exhibit notably different levels of spatial understanding ability. In contrast, Counting tasks show the lowest variance $( \sigma = 0 . 0 9 6 )$ , indicating that this task type presents consistent difficulty across most models. Similarly, among adversarial pressure conditions, increasing Number of Images(NI) exhibits the lowest variance $( \sigma ~ = ~ 0 . 0 7 9 )$ , comparable to Counting tasks, suggesting that extreme context pressure leads to uniformly poor performance across models. For reasoning patterns, Selection demonstrates the highest variance $( \sigma = 0 . 1 5 3 )$ , suggesting that the ability to accurately identify a specific image among multiple candidates varies considerably across different models.

![](images/40fa22e07f83a337aa4a5d567e271abb45f14aaee1215d9fdb723b9984c48a1d.jpg)

![](images/f35810e3fa3f1791a6dfe4b588a496214a4ffe011e607f4a47d6075f05014d5a.jpg)

![](images/397e04f47bdc578b75c6e7b4e4f6eb67946ef6ab2c8d749983cc03d6fdd2a41e.jpg)  
Figure II. Performance mean and variance. (a) Tasks, (b) Reasoning patterns, and (c) Adversarial Pressures.

(a)  
![](images/55dbaab6ad70f088498e991b6189aaadd09edc0878f4aeaf0f8e708f37468902.jpg)

(b)  
![](images/2faaaa5afbbd71e2127b9d0b8fd9d9d0013219ec61656490ada4ad6fcbfd0601.jpg)  
Figure III. Cross-dimensional degradation analysis. Performance degradation from Easy baseline across (a) Task types and (b) Reasoning dimensions under adversarial pressure.

## B.2. Cross-dimensional Interaction Analysis

Fig. III presents degradation heatmaps showing performance drops from the Easy baseline across different taskpressure and reasoning-pressure combinations, averaged over all models. Increasing Number of Images(NI) causes catastrophic degradation across all dimensions, with Comparison reasoning and Existence tasks most severely affected. Combined(HP+HN) pressure shows additive difficulty, causing larger drops than either pressure alone. Comprehensive reasoning demonstrates superior robustness compared to Comparison and Selection, suggesting holistic reasoning strategies better withstand adversarial pressure.

Counting tasks show minimal degradation not due to robustness but floor effects, as their Easy baseline is already low. These patterns reveal that adversarial robustness is highly dimension-dependent.

## B.3. Performance Gap Between Open-Source and Commercial Models

We compare the capabilities of leading commercial models and top-performing open-source models in Fig. IV. The performance gap between these two categories is most pronounced in Counting and Comparison tasks. These capabilities represent the areas where commercial models demonstrate the largest advantages over open-source alternatives, indicating differences in multi-image reasoning capacity.

## C. Benchmark Examples

## C.1. Qualitative Examples from MIOH benchmark

Figs. VI and VII provide qualitative examples from the MIOH benchmark, illustrating how questions are formulated across our four core tasks and various visual adversarial pressures. Each example is designed to test a specific aspect of an MLLM’s object-centric capabilities and robustness against perceptual challenges.

![](images/45f43c80dbd6220f28b1d1a369bad79e34552c81e1b2ded00a6ca1eedacc4e04.jpg)  
Figure IV. Performance of commercial and top open-source models.

Existence Tasks. (Fig. VI, top section) assess the model’s ability to verify the presence of objects and pinpoint their location within the image set. The Selective questions showcased here require the model to identify the specific image containing the target object among candidates. Examples progress from straightforward cases (Easy: clearly visible “donut”) to perceptually challenging scenarios. The hard positive example requires detecting a “bench” in a rainy scene, where the object is situated near the greenery and partially obscured by a person with umbrella, making it difficult to spot. The Hard Negative example tests robustness against contextual bias, such as identifying a “keyboard” in a set of images containing office-like environments or other electronics (e.g., game consoles) that visually resemble the target context but lack the specific object.

Counting Tasks. (Fig. VI, bottom section) evaluate quantitative reasoning capabilities through Comprehensive questions that require aggregating counts across all provided images. The Easy examples involve basic enumeration, such as counting the occurrences of clearly visible “elephant”s across four frames. The Hard Positive scenario challenges the model to sum the total number of “sandwiches” across images where objects are heavily occluded, cut into pieces, or cluttered on plates, testing the ability to handle dense visual information. The Hard Negative example asks for the count of “trains,” where models must distinguish the actual object from contextually similar background elements like tracks or station platforms in the nontarget images.

Position Tasks.(Fig. VII, left section) present the most complex spatial relationship challenges. We use Selective questions to ask the model to identify the image where a specific relation holds. Examples include Easy scenarios (“dog next to cat”), Hard Negative cases (“chair positioning relative to dog”), and Hard Positive examples (“person next to umbrella”) that test compositional scene understanding beyond simple object detection.

Attribute Tasks. (Fig. VII, right section) assess detailed compositional understanding by requiring models to bind visual properties with objects using Comparative questions. The tasks range from detecting visually distinct attributes (Easy: “red scarf”) to more subtle distinctions. The Hard Negative scenario involves identifying a “dark yellow mug” in a cluttered indoor environment where lighting and other objects may create confusion. The Hard Positive example tests the model’s ability to consistently recognize a “white bench” across two different scenes despite variations in perspective and background.

Each example demonstrates the three question types designed for multifaceted evaluation: comprehensive (collective understanding across images), comparative (identifying differences between images), and selective (retrieving specific images matching descriptions). The progression of difficulty incorporates both visual factors (scale, occlusion, contextual bias) as detailed in Sec. 3.3.

## C.2. Failure Case Examples of Commercial Models

To better understand the limitations of commercial MLLMs, we analyze specific failure cases of GPT-5 and Gemini-2.5-Pro on the MIOH benchmark. Despite their strong baseline performance, these models still exhibit hallucination patterns under visual adversarial pressures. Fig. V illustrates representative error cases where models fail to ground visual evidence correctly within multi-image contexts.

For example, in the Attribute-Comprehensive task, models exhibit perceptual blindness or aggregation failures when objects are visually ambiguous or occluded; for instance, both models fail to confirm the presence of “dark gray bowls” in all target images, demonstrating a breakdown in comprehensive verification. The example in Counting-Comparative task reveals weaknesses in fine-grained classification, where models confuse “wine glasses” with perceptually similar objects like water goblets or tumblers, leading to incorrect frequency comparisons. Most critically, the Position-Selective task exposes severe object and relationship hallucination under the “None of the above” option. When asked for a “cow behind a dog,” models are forced into making a selection, resulting in hallucinated object identities (e.g., misidentifying a zebra as a cow in Image 4) or ignoring spatial constraints (e.g., selecting a dog-cow pair with incorrect positioning in Image 3), rather than correctly abstaining.

![](images/5be05535e7b9af544966db6880b012e163706f185ada63aca939a42f93cd3e6a.jpg)  
Figure V. Failure cases of top commercial models (GPT-5, Gemini-2.5-pro). Blue text indicates the Ground Truth (GT). The icons indicate the incorrect choices made by each model. These examples highlight how perceptual difficulty lead to reasoning failures.

![](images/5a7e2b4262912fe8375fae4cbee5cd9c45b20384bce40705c0f276a530a8beac.jpg)

## Existence - Selective

![](images/8f6d5cd8e9ffcb3e577ca9e9021e1dd2b984fc1bcacf9141d291ec2862ebb53a.jpg)

![](images/5dc8bd267c2d5ee94b6de5e799173385512a1b03d4d05aa23348de288cec1696.jpg)

![](images/e886998fe95cf171fde73f000efbf27ac6cdaa8bb62d80f25e1e186539c31f70.jpg)

![](images/6e1fbca16deb53cb4f0a92fcaad0374c37d5202582f61efc35f03d39f475ae77.jpg)

![](images/4128bc396ee59cd0ffd5edf823ca8e48860e93697065176d69d3e4193a8d55f5.jpg)

![](images/884b33143a6d02cbfb93995065c086d2be9f766365ba3560239a6003904a181c.jpg)

![](images/4da2d609bfed8d1601b301c27cb2374bbdf995a2e749585719e7b28affee8ada.jpg)

![](images/db69240e31572dbf55079547a6badbcc7755b1485f257eb5803643bfe808cb16.jpg)

## Counting - Comprehensive

![](images/e2795aac2e26f15ae7da53ec70df0ec27a33ff45696a0cb9851e76a75be46f81.jpg)

![](images/7a6bfad68d9a2bc169c7bf62ee5fe508562ee8770e1a6a9b03817586c2325db0.jpg)

![](images/4081bf97e0593e92fae4ec226718e272f8c3b29f4f4710d846bfbea1d3877cbc.jpg)  
In which image does a ‘donut’ present?  
In which image does a ‘donut’ present?

## Hard Negative

![](images/ae9b944f78088d9e4629fb7fab57314b971c101abc6ccf8c803cf4b7ec99bc72.jpg)  
A) Image 1 B) Image 2 C) Image 3 D) Image 4 E) None of the above  
A) Image 1 B) Image 2 C) Image 3 D) Image 4 E) None of the above  
Which of the following objects appears a total of 1 time across all four images?

## EASY

A) hot dog B) toaster

C) cup D) broccoli E) None of the above

![](images/202d2fe9fe3acd5d0d96196bd8de5ff5b0992b09f37d32987b52e26da2802a3e.jpg)

![](images/0e9109dea8835faad53f0a1132eecc2bc9de2aca76eadbcbed67ab3df3228dff.jpg)

![](images/e9c74b6768f3c38e1fe366e05ebbc961b6a3dbcab374800010a95cb64656e373.jpg)

![](images/98d7dbc3406b058a5892770d42e0132afc2bfd3fa6e57aed6c2e553c467eadc6.jpg)

![](images/1b98e41a72e9d46fa2ff10d08579ff78dbb3b7beaa9605c9066c41c97c7e189b.jpg)

![](images/975ce3a4324889bc53006a5489b7b852c7ef1fc09c421900715f0e9042a61156.jpg)

![](images/0c16649d54b8c359ee167a626aee2a5c78ee3413a07102ae7b06d600c17949c6.jpg)

![](images/24698554022d3fb6e1825e968468dea987a24b4bf435e4030ced166189176392.jpg)  
In which image does a ‘keyboard’ present?

![](images/c979dab2650e75d56218e3c0641683fae080738f193ed6310a9030bc60a6deb0.jpg)

![](images/fb1e908d9b1c604443a34dac7259cbfbe45d68d726e01ced81a4172312aa4a6b.jpg)

![](images/374b904e5af323c80a6462098c6797c385befddd312fae50f1411071c95eefea.jpg)

![](images/94e0617c904c8c7161c09370b0706aa47ce14b4d1f08ce0042f5a00a5722d8c0.jpg)

![](images/021ea438701e0c3d80599ed350d1a1eac44be7eb4f7ab7db1bbafb0fdc37a660.jpg)

![](images/cd8a053e6c0ad9b0db5f8c21bbe0211a2bff2436053696d371b39bc03cf3c0eb.jpg)

## Hard Positive

![](images/76685e2ea3f58a58dc7ebdb4dadf48cadee0004b4cb6df949166de3af15824e8.jpg)  
In which image does a ‘bench’ present?  
A) 1 Image B) 2 Images C) 3 Images D) 0 Image E) I don’t know

A) 2 Images B) 3 Images C) 0 Image

![](images/625fb737345e3f25d75571c048de9be38df09d9eaf042f80868e2559d0637c5d.jpg)

D) 4 ImagesE) I don’t know

![](images/b9d8ffce0f4ce541f3b819f06f7074089e5ab54a41d84d8f2d81efb78b03aa89.jpg)  
A) Image 1 B) Image 2

![](images/54fdd9ad12498274f511f96ccb08de739789519de766af82ed6577cfda77343e.jpg)

## EASY

## Hard Negative

In how many of these four images is a(n) ‘elephant’ present?

![](images/05783ec2a35a21a076018afc9cc81a4045cd14ff4ed7aed0305133206d8001a1.jpg)  
C) Image 3 D) Image 4 E) None of the above

Figure VI. Benchmark Examples 1. Existence and Counting Task  
![](images/aee95178d755e1c78b5fdf2a00adde6f80b3d2c359d16c0a7786c22f45b6604c.jpg)

## Hard Negative

A) Image 1   
B) Image 2   
C) Image 3

In how many of these four images is a ‘train’ present?

D) Image 4 E) None of the above

## Hard Positive

What is the total number of ‘sandwiches’ across all four images?

C) 10 D) 8 E) None of the above

![](images/0fe57ed5cc2feb78f16e1acc979d617a0bc8b43d5ddb6d8d4149ae1e181e2a52.jpg)  
Which of the following is present in Image 1 but not in Image 2?  
Where is a chair that is right of a dog?  
Where is a person that is next to an umbrella?  
A) Image 1 B) Image 2 C) Image 3 D) Image 4 E) None of the above

## Position - Selective

![](images/3800bf5c284a618f3de88ecbcbf9c6aa30f55743ad53359dae35caeceb0a350f.jpg)

![](images/46b80fc256e72f1a44c94433469326117d043921fe5b1c6e087651069c4b5d72.jpg)

![](images/0da92e55593847007afac5491f11c53092962509d94e3e5cddf32e334307391f.jpg)

![](images/892f50ec425dcc5ddfcc2f129d244b9ae2081daecce4401fa177e5c0069b06eb.jpg)  
In which of these images can you find a dog that is next to a cat?

![](images/dd227b8440f407b4cce1cdf88108518631a643d83b7cc92ce3ebb5cd0e338ac3.jpg)

## Hard Negative

B) Neither C) Image 1 D) Image 2

## Attribute - Comparative

![](images/def36a6b8c3e07d1646a31ff4670938e83c41aaa5b9348562f88749d4e6005f8.jpg)

![](images/9f2d1c046ea309af47f0917904248532b57de761c010fbf8180384c2199ee566.jpg)  
Which of the following is present in Image 1 but not in Image 2?

![](images/cd6d5f196b27eeb2e39c755a769dde8b234cbfe5f1346ebef5af8da4d3a26e0a.jpg)

![](images/a038c93c3fcbd29947d8d4f9756dec7359ffbf77339595b8428396e3a7008f70.jpg)  
Which of the two images a ‘white bench’ is/are present?

![](images/3e37f944103625ebca2ee714c1ccaa4430050201455a00ea4795873e5d78f90d.jpg)  
A) light yellow hat B) rattan hat C) red scarf D) light green box E) None of the above

![](images/e978fa4bde60dc4a5ad0c1f4faea4d2b46ece191e61350ce8837fb6baa2522cf.jpg)

A) dark yellow mug B) striped microwave oven C) dark purple car D) light brown mouse (computer equipment) E) None of the above

![](images/780eb601c41d37fb8cb70aeea3d99b6aa8c6b957bc6a00769e546d51804f39ff.jpg)

![](images/d082e61dfead38148f70885301ae5899ddf2c78fb900129522cc1181b896e47f.jpg)

![](images/bf1ed21169d88c6c6de6a55259e2e3808c7ccf204878d5cfa5ad125b2fc400ef.jpg)  
Figure VII. Benchmark Examples 2. Position and Attribute Task

## Hard Positive

A) BothB) NeitherC) Image 1

A) BothB) NeitherC) Image 1D) Image 2