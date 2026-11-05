# Beyond Scene Shortcuts: Action-Centric Video-Text Alignment for Fine-Grained Action Understanding

Zhiliang Zhu<sup>1</sup>, Jiazheng Huang<sup>2</sup>, Dawei Zhao<sup>1,∗</sup>

<sup>1</sup>Harbin Engineering University, China <sup>2</sup>South China Normal University, China

## Abstract

Fine-grained action understanding requires vision-language models to ground what an agent does, how it acts, and why it acts in spatio-temporal evidence. However, richer supervision can backfire: fine-tuned models may generate detailed action descriptions grounded in static scenes and salient objects rather than motion, a failure we call Environment-Grounded Action Hallucination (EGAH). To mitigate EGAH, we propose Staged Attention Rectification and Reinforcement (STARR), an action-centric video– text alignment framework that breaks scene–action shortcuts through Action-Environment Dissociation and reinforces action-focused grounding with an Attention Consistency Reward within Group Relative Policy Optimization. We further construct FineActionR&R, combining temporally localized captions with hierarchical and temporal-causal reasoning data. Built on Qwen2.5-VL-7B, FineActionVLM-R1 improves average MotionBench accuracy by 3.01 percentage points over its base model and achieves 91.50% on HAU-MCQ-Bench. Qualitative comparisons further indicate more faithful finegrained action descriptions with fewer environment-induced hallucinations.

## Introduction

Understanding how and why an agent acts—not merely what it does—requires vision-language models (VLMs) to ground action language in subtle spatio-temporal evidence. Finegrained action understanding therefore depends on motion transitions, body dynamics, interaction order, and objectstate changes rather than recognizable scenes and objects alone. For example, saying that “the robot moves the plate” misses that it first pauses and then gently slides the plate left to maintain a safe distance from a nearby cup. Despite strong scene and object recognition, recent VLMs still struggle to associate such fine-grained action language with the corresponding dynamic evidence (Team 2025; Choi et al. 2025; Salah et al. 2024; Wu et al. 2025).

A natural response is to fine-tune VLMs with richer action captions and reasoning annotations. Yet we observe that this supervision can backfire: fine-tuned models produce longer and more specific descriptions while introducing actions unsupported by the video, especially in out-of-domain (OOD) examples. As illustrated in Figure 1, the environment and salient objects may be described correctly even when the central action is not, suggesting that the model learns where an action is likely to occur rather than identifying the motion that actually occurs. We call this failure Environment-Grounded Action Hallucination (EGAH), a specific form of multimodal hallucination (Zhou et al. 2023; Chen et al. 2024b; Yang et al. 2025b) in which action language is grounded in static environmental cues rather than temporal dynamics.

We attribute EGAH to two coupled biases. First, static scenes and salient objects are visually dominant and persist across frames, making them easier to align with text than short, subtle, or multi-stage movements. A model can therefore reduce its training loss through scene–action correlations instead of learning precise temporal localization. Second, supervised fine-tuning strengthens linguistic co-occurrences between environments and actions, allowing weak visual grounding to expand into a fluent but unsupported narrative. Together, visual shortcuts suggest a plausible action, while linguistic priors elaborate it into a detailed description.

To shift video–text grounding from scene shortcuts to genuine action evidence, we propose Staged Attention Rectification and Reinforcement (STARR), an action-centric video–text alignment framework. Its first component, Action-Environment Dissociation, inserts a temporally annotated action clip into unrelated background sequences while retaining the action caption as the generation target. Because the surrounding environment no longer provides a reliable signal for predicting the caption, the model is encouraged to locate the action-containing segment and ground its description in temporal evidence. Rather than simply adding more supervision, this construction changes which visual evidence is useful for predicting action language.

Even after supervised fine-tuning, action-focused grounding may weaken as autoregressive generation becomes increasingly influenced by linguistic priors. STARR therefore introduces an Attention Consistency Reward within Group Relative Policy Optimization (GRPO) (Shao et al. 2024). The reward promotes strong and balanced attention over actionrelevant visual tokens, reinforcing action-centered grounding during generation. The two components are complementary: Action-Environment Dissociation reduces the predictive value of environmental shortcuts, while the Attention Consistency Reward reinforces the use of action-relevant visual evidence.

To connect fine-grained grounding with higher-level action reasoning, we further construct FineActionR&R, a dataset suite combining temporally localized action captions, hierarchical action-reasoning questions, and temporalcausal multiple-choice questions. Existing video benchmarks largely emphasize action recognition or classification, with limited support for studying how an action unfolds, why a particular strategy is adopted, and how multiple events are temporally or causally related (Hong et al. 2025; Tu et al. 2025). FineActionR&R provides complementary supervision for recognizing what occurs, describing how it occurs, and reasoning about why it occurs.

![](images/743a65d555346eb6fe0e86e2c6900e4be3d4b9eebb65e5089fd57f6de7dc782e.jpg)  
(a)  
(b)  
Figure 1: Illustration of Environment-Grounded Action Hallucination. (a) VLM often generate accurate environment descriptions but incorrect action ones. During SFT, the model develops a strong tendency to anchor action predictions on easily perceived static backgrounds rather than true spatio-temporal evidence, leading to systematically incorrect action descriptions even when its environment descriptions remain accurate. (b) When environmental information is explicitly described in fine-grained action description tasks, it can easily trigger hallucinated actions unrelated to the actual video content.

FineActionVLM-R1 is trained on FineActionR&R using the STARR framework, with Qwen2.5-VL-7B as the base model (Team 2025). The resulting model improves average MotionBench accuracy by 3.01 percentage points over its base model and achieves 91.50% accuracy on HAU-MCQ-Bench. On the latter benchmark, it outperforms substantially larger models, including Qwen2.5-VL-72B and Qwen3-VL-32B, with training exposure difering across models; matched-SFT controls are reported in Appendix D.4. Qualitative and out-of-domain analyses further show that FineActionVLM-R1 produces more faithful fine-grained descriptions with fewer environment-induced hallucinations. These results support our central finding: efective finegrained action understanding requires not only richer supervision, but also explicit control over the visual evidence to which action language is aligned.

Our main contributions are summarized as follows:

• We identify and characterize Environment-Grounded Action Hallucination, a fine-grained action grounding failure in which action language relies on static environmental cues rather than motion evidence. We further analyze the roles of visual shortcuts and linguistic priors in producing this behavior.

• We propose Staged Attention Rectification and Reinforcement, which combines Action-Environment Dissociation during supervised fine-tuning with an Attention Consistency Reward during GRPO to promote action-centered

video–text grounding.

• We construct FineActionR&R, integrating temporally localized action captions with hierarchical and temporalcausal reasoning supervision. Experiments show gains in action recognition and reasoning, while qualitative comparisons indicate more faithful fine-grained captioning.

## Related Work

## Action-aware VLMs

Recent work on human action understanding has improved recognition robustness, temporal modeling, and finegrained analysis. MASH-VLM (Choi et al. 2025) mitigates action–scene hallucination by explicitly disentangling spatiotemporal representations. FTP (Salah et al. 2024) uses VLMaided training to combine the generalization of VLMs with the specialization ofVideo Transformers. For finer action distinctions, HieroAction (Wu et al. 2025) decomposes complex actions into hierarchically organized sub-movements, while FinePseudo (Rizve, Shah, and Dave 2024) exploits temporal alignability to improve pseudo-label quality in semisupervised recognition. Complementing these model-based advances, MotionBench (Hong et al. 2025) and FAVOR-Bench (Tu et al. 2025) provide dedicated evaluations of finegrained video motion understanding.

However, this strong body of work remains focused on action recognition—identifying what movement occurred. A key challenge in action understanding lies in inferring the actor’s intent and causal reasoning behind fine- grained actions. We study a supervision-induced generation failure in which rich action descriptions become grounded in static environmental cues. STARR addresses this misalignment at the data and optimization levels, while FineActionR&R extends supervision from recognizing what occurs to describing how actions unfold and reasoning about their temporal and causal structure.

## Mitigating Hallucinations in VLMs

Methods for mitigating VLM hallucinations broadly fall into training-free inference-time interventions and training-based optimization (Chen et al. 2024b; Zhou et al. 2023; Yang et al. 2025b). Training-free methods modify multimodal decoding without updating model parameters. PAINT (Arif et al. 2025) amplifies attention to informative visual tokens, IKOD (Yang et al. 2025a) compresses text tokens through KV-cache merging to increase the relative contribution of visual information, and M3ID (Favero et al. 2024) contrasts multimodal logits with a text-only prior. Training-based approaches instead strengthen visual grounding through model optimization. Qwen-LA (Chu et al. 2025) teaches the model to re-attend to visual tokens during reasoning, while Hallucination-Induced Optimization (Lyu et al. 2024) trains the model to recognize and move away from its self-generated hallucinations.

Most general-purpose hallucination mitigation methods are developed primarily for image-based settings. Finegrained video action hallucination introduces an additional grounding challenge: the relevant evidence may be brief and distributed across frames, while the surrounding scene can be recognized correctly even when the generated action is unsupported. EGAH captures this specific mismatch between static scene evidence and temporal action language. STARR accordingly combines Action-Environment Dissociation, which reduces the predictive value of environmental shortcuts during supervised fine-tuning, with an Attention Consistency Reward that reinforces action-relevant visual grounding during generation.

## Datasets

We construct FineActionR&R (Fine-grained Action Recognition & Reasoning), a dataset suite with four components for fine-grained action grounding and reasoning. Additional construction and annotation details are provided in Appendix A.

Fine-Action-Caption and Fine-Action-Segment. Using videos sampled from established action datasets, Fine-Action-Caption provides dense human-written descriptions that decompose complex actions into ordered sub-events and capture body dynamics, object interactions, and relevant context. Fine-Action-Segment further provides the start and end timestamps of each annotated action.

Hierarchical Action Understanding QA (HAU-QA). HAU-QA is a large-scale question–answer dataset generated with advanced language models such as Gemini 2.5 Pro. Each QA pair includes a rationale covering micro-level physical interactions, macro-level task logic and causality, and agent–environment interaction.

Hierarchical Action Understanding MCQ (HAU-MCQ). HAU-MCQ converts HAU-QA into multiple-choice questions with plausible but visually inconsistent distractors, providing discriminative supervision for GRPO-based action reasoning.

Temporal Action Sequencing and Causality MCQ (TASC-MCQ). TASC-MCQ provides structured video questions about action concurrency, ordering, duration, and causality to improve temporal action recognition and reasoning.

## Method

## Overview

To address the problem of Environment-Grounded Action Hallucination in fine-grained action understanding, we propose a two-stage training framework named Staged Attention Rectification and Reinforcement. This framework is designed to systematically decouple the model’s dependence on static environments and reinforce its focus on core dynamic content.

Stage 1: Attention Rectification. This stage employs SFT on a specially crafted synthetic dataset. In this dataset, core action video clips are randomly spliced into irrelevant background videos. This forces the model to perform implicit temporal localization to identify the action, thereby learning to ignore environmental distractors and fundamentally rectifying its attention bias.

Stage 2: Attention Consistency Reinforcement. This stage is based on GRPO and introduces a novel Attention Consistency Reward. This reward function incentivizes the model to continuously and stably focus its attention on key action-relevant visual tokens throughout the entire text generation process. This solidifies the corrections from the first stage and further suppresses the negative efects of attention drift and linguistic priors.

## Attention Rectification via Action-Environment Dissociation.

To address the model’s attention bias towards static environmental cues and the resulting EGAH problem, we propose a SFT stage named “Attention Rectification”. The core objective of this stage is to compel the model to shift its attention from the static background to the crucial dynamic action content. To achieve this, we devise a data synthesis strategy that eliminates the model’s reliance on environmental shortcuts by constructing training samples where the visual environment and the target description are deliberately mismatched.

Specifically, our method involves constructing a synthetic dataset, Action-Environment Dissociation Dataset (AED) $\mathcal { D } _ { s y n }$ . We first prepare two data sources: a collection of action clips with temporal annotations and corresponding textual descriptions, denoted as $\mathcal { D } _ { a c t } = \{ ( V _ { a c t } , \bar { C _ { a c t } } ) \}$ , and a collection of video clips containing only dynamic backgrounds with no significant actions, denoted as $\mathcal { D } _ { b g } = \{ \bar { V _ { b g } } \}$ . For each action clip $V _ { a c t }$ in $\mathcal { D } _ { a c t }$ , we randomly sample n background clips $\{ V _ { b g , i } \} _ { i = 1 } ^ { n }$ from $\mathcal { D } _ { b g }$ . The action clip $V _ { a c t }$ is then inserted at a random temporal position among the sequence of these n background clips, which are then concatenated to form a synthetic video $V _ { s y n }$ . This process yields a new training sample $( V _ { s y n } , C _ { a c t } )$ , where the synthetic video $V _ { s y n }$ serves as the model input and the original action description $C _ { a c t }$ serves as the generation target. During training, for a target caption $C _ { a c t }$ composed of tokens $y _ { 1 } , \ldots , y _ { T }$ , the optimization objective for the model M (with parameters θ) is to minimize the standard autoregressive negative log-likelihood loss (i.e., the next-token prediction loss):

![](images/fa4afa4c82afca978c41643d27a07a2c501e717be1ec8963e3d92dce623623d2.jpg)  
(a) Attention Rectification via Action-Environment Dissociation.  
(b) Attention Consistency Reinforcement.  
Figure 2: Staged Attention Rectification and Reinforcement method for fine-grained action understanding. (a) Attention Recti fication: SFT is used to train on distractor videos that combine an action clip with irrelevant backgrounds, forcing the model to focus on true action regions. (b) Attention Consistency Reinforcement: GRPO optimizes a reward encouraging stable attention on key action-relevant visual tokens throughout generation.

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = \mathbb { E } _ { ( V _ { \mathrm { s y n } } , C _ { \mathrm { a c t } } ) \sim \mathcal { D } _ { \mathrm { s y n } } } \bigg [ - \sum _ { t = 1 } ^ { T } \log P ( y _ { t } \mid V _ { \mathrm { s y n } } ,\tag{1}
$$

Under this training paradigm, the multiple, randomly concatenated background clips $\left\{ V _ { b g , i } \right\}$ act as strong distractors. Since their visual content is entirely unrelated to the target caption $C _ { a c t } ,$ , any predictive strategy relying on global or static scene features is rendered inefective. To minimize the loss, the model is forced to perform implicit temporal localization: accurately identifying and attending to the actioncontaining clip $\dot { V _ { a c t } }$ within the full video $\breve { V _ { s y n } } ,$ while suppressing interference from the irrelevant background clips. This construction reduces correlations between the target action and surrounding context, encouraging action-focused alignment while retaining the scene and object cues within the action clip.

## Attention Consistency Reinforcement

Following attention rectification in the SFT stage, Attention Consistency Reinforcement uses GRPO with an Attention Consistency Reward (ACR). This reward encourages strong and balanced caption-averaged attention to referenceselected visual evidence. Its variance term operates across visual tokens after averaging over generated text positions; it does not directly measure stability across decoding steps. Token-selection validation and component ablations are provided in Appendix D.

To compute the ACR, we must first define a set of actionrelevant visual tokens. This is achieved by leveraging the ground-truth fine-grained action caption associated with each video. To generate a stable attention baseline, we use the model’s policy from immediately before the current optimization step as a frozen reference model. We then compute the attention from the entire caption to the full set of visual tokens, $F _ { V } = f _ { 1 } , \ldots , f _ { N }$ . For each visual token, its cumulative attention score is calculated by summing the attention it receives from all textual tokens in the caption. Based on these scores, all visual tokens are ranked. The top 25% of this ranking are designated as the action-relevant visual tokens, $F _ { \mathrm { a c t } } \subset F _ { V }$ . These tokens constitute the critical visual evidence the model is rewarded for focusing on.

To capture the model’s attention behavior, we define the attention matrix $A \ \in \ \mathbb { R } ^ { | F _ { \mathrm { a c t } } | \times T }$ , where each element $A _ { i j }$ denotes the attention score that the j-th generated text token $y _ { j }$ allocates to the i-th action-relevant visual token $f _ { i } \in F _ { \mathrm { a c t } }$ For each such token, we compute its average attention value across the T generated text tokens:

$$
a _ { i } = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } A _ { i j } .\tag{2}
$$

The set $\{ a _ { i } | f _ { i } \in F _ { \mathrm { a c t } } \}$ thus characterizes the model’s attention distribution within the action-relevant visual space.

Maximizing Mean Attention $( R _ { \mathbf { m e a n } } ) { : }$ This component encourages strong overall attention to action-relevant regions. However, relying solely on an absolute measure can be insuficient when diferent generated samples yield numerically close mean attention scores. To create a more robust reward signal that diferentiates well even in such cases, we formulate $R _ { \mathrm { m e a n } }$ as a composite of two sub-rewards: an absolute reward $R _ { \mathrm { a b s } }$ and a relative reward $R _ { \mathrm { r e l } }$ . The absolute reward $R _ { \mathrm { a b s } }$ measures the raw attention strength against a predefined threshold $\tau _ { m } .$ , rewarding the model for achieving a high baseline of attention. It is defined as:

$$
R _ { \mathrm { a b s } } = \operatorname* { m i n } \left( \frac { 1 } { \tau _ { m } | F _ { \mathrm { a c t } } | } \sum _ { f _ { i } \in F _ { \mathrm { a c t } } } a _ { i } , 1 . 0 \right) .\tag{3}
$$

The relative reward $R _ { \mathrm { r e l } }$ amplifies the diferences between samples within the same generation batch. Let $\begin{array} { r l } { \bar { a } _ { k } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { \left| F _ { \mathrm { a c t } } \right| } \sum _ { f _ { i } \in F _ { \mathrm { a c t } } } a _ { i } ^ { ( k ) } } \end{array}$ be the mean attention for the k-th sample in a batch B. The relative reward for this sample is calculated by applying min-max normalization across the batch, which highlights its performance relative to its peers:

$$
R _ { \mathrm { r e l } } ^ { ( k ) } = \frac { \bar { a } _ { k } - \operatorname* { m i n } _ { j \in { \cal B } } ( \bar { a } _ { j } ) } { \operatorname* { m a x } _ { j \in { \cal B } } ( \bar { a } _ { j } ) - \operatorname* { m i n } _ { j \in { \cal B } } ( \bar { a } _ { j } ) + \epsilon } ,\tag{4}
$$

where ϵ is a small constant to ensure numerical stability. The final mean attention reward, $R _ { \mathrm { m e a n } } ,$ is a weighted sum of these two components, allowing us to simultaneously encourage high absolute attention and reward the best-performing samples within a group:

$$
R _ { \mathrm { m e a n } } = \alpha _ { \mathrm { a b s } } R _ { \mathrm { a b s } } + \alpha _ { \mathrm { r e l } } R _ { \mathrm { r e l } } ,\tag{5}
$$

where $\alpha _ { \mathrm { a b s } }$ and $\alpha _ { \mathrm { r e l } }$ are hyperparameters balancing the two objectives.

Minimizing Attention Variance $( R _ { \mathbf { v a r } } ) { : }$ This term promotes balanced attention across all action-relevant regions. It rewards lower variance in $\left\{ { a } _ { i } \right\}$ , ensuring that attention is not concentrated on a specific sub-region. The reward attains its maximum (1.0) when attention is uniformly distributed and decreases linearly as the variance approaches a threshold $\tau _ { v } \colon$

$$
R _ { \mathrm { v a r } } = \operatorname* { m a x } \left( 0 , 1 - { \frac { \mathrm { V a r } _ { f _ { i } \in F _ { \mathrm { a c t } } } ( a _ { i } ) } { \tau _ { v } } } \right) .\tag{6}
$$

Finally, the total ACR combines these two components:

$$
R _ { \mathrm { A C R } } = \lambda _ { \mathrm { m e a n } } R _ { \mathrm { m e a n } } + \lambda _ { \mathrm { v a r } } R _ { \mathrm { v a r } } ,\tag{7}
$$

where $\lambda _ { \mathrm { { m e a n } } }$ and $\lambda _ { \mathrm { v a r } }$ are weighting coeficients that contro the trade-of between overall attention strength and spatial uniformity.

Intuitively, $R _ { \mathrm { m e a n } }$ encourages attention strength over the reference-selected tokens, while $R _ { \mathrm { v a r } }$ encourages balanced coverage of those tokens after averaging over generated positions. Integrating $R _ { \mathrm { A C R } }$ into GRPO directly optimizes this attention objective. Its contribution to caption faithfulness is evaluated through the controlled ablations in Appendix D.2.

## Training Recipe

Our training methodology employs a three-stage paradigm to progressively impart the model with action understanding and description capabilities. This staged approach systematically addresses the challenges of attention bias and linguistic overfitting. The entire training process begins with Qwen2.5- VL-7B-Instruct(Team 2025) as the base model.

Stage 1: Temporal Reasoning Pre-training The initial stage aims to build a foundational capacity for temporal reasoning and causality analysis. Starting with base model, we utilize the TASC-MCQ within the GRPO. This process encourages the model to generate intermediate reasoning steps before providing a final answer, efectively stimulating its ability to reason about the temporal dynamics presented in a video, rather than relying on static priors. The outcome of this stage is an initial model checkpoint, FineActionVLM-R1-Zero, which demonstrates a nascent ability for videogrounded thinking.

Stage 2: Attention Rectification and Multi-task Joint Fine-tuning This stage is designed to tackle the core issue of EGAH while simultaneously enhancing the model’s fine-grained action recognition and understanding abilities. We perform a multi-task SFT on the base Qwen2.5-VL-7B-Instruct model, integrating several key datasets: (1) The synthetic AED dataset is used to rectify the model’s attention bias, as described above. (2) The Fine-Action-Caption dataset and a subset of the HAU-QA dataset are incorporated to strengthen the model’s ability to recognize and comprehend nuanced human actions. (3) A curated dataset for videogrounded reasoning is also included. This data is created by distilling reasoning chains from the FineActionVLM-R1- Zero model on the TASC-MCQ, followed by LLM-based filtering to remove samples reliant on common-sense rather than video observation. Through this comprehensive SFT process, we obtain the FineActionVLM-R1-Dev1 model. This model shows significant improvement in mitigating EGAH and is capable of generating higher-quality samples for both in-domain and out-of-domain data.

Stage 3: Attention Consistency Reinforcement and Comprehensive Capability Enhancement Starting from the FineActionVLM-R1-Dev1 checkpoint, this final stage aims to stabilize generation quality and fully activate its advanced reasoning capabilities using a GRPO-based reinforcement learning paradigm. We introduce the novel ACR with the Fine-Action-Caption dataset. This reward mechanism reinforces the strength and balance of caption-averaged attention over reference-selected visual tokens. Grounded in the principle that action recognition and understanding are mutually reinforcing, we continue to incorporate the TASC-MCQ and HAU-MCQ datasets. Training on these discriminative multiple-choice tasks further refines the model’s ability to discern subtle temporal and causal relationships. This final stage yields our ultimate model, FineActionVLM-R1.

## Experiment

In this section, we conduct comprehensive experiments to evaluate both the efectiveness of our FineActionR&R dataset and the proposed STARR method. Specifically, we first demonstrate that training on FineActionR&R significantly improves the model’s capability in fine-grained action recognition and understanding, supported by results on MotionBench and HAU-MCQ-Bench. These findings verify the dataset’s strong contribution to enhancing temporal perception and hierarchical reasoning. Furthermore, to validate the efectiveness of our STARR method, we perform caption quality assessments, showing that it substantially alleviates EGAH and enables the generation of temporally consistent, visually grounded fine-grained descriptions. Additional qualitative analyses are presented in Appendix C; human evaluation, component ablations, and quantitative controls are reported in Appendix D.

## Benchmarks

Fine-grained Action Recognition. We use Motion-Bench(Hong et al. 2025) to evaluate our model’s fine-grained action perception. The benchmark assesses a model’s perception of subtle, transient dynamics across six core tasks: Motion Recognition, Location-related Motion, Action Order, Repetition Count, Motion-related Objects, and Camera Motion.

<table><tr><td># Model</td><td>Organization Frames</td><td></td><td>LLM Params</td><td colspan="7">Dev Avg (%) MR (%) LM (%) CM (%) MO (%) AO (%) RC (%)</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>Alibaba</td><td>2fps</td><td>30B (A3B)</td><td>57.59</td><td>61.03</td><td>61.90</td><td>55.84</td><td>75.07</td><td>45.09</td><td>26.75</td></tr><tr><td>2 Qwen3-VL-8B-Instruct</td><td>Alibaba</td><td>2fps</td><td>8B</td><td>55.55</td><td>59.07</td><td>57.88</td><td>55.84</td><td>72.46</td><td>44.70</td><td>24.00</td></tr><tr><td>3 Kimi-VL-2506</td><td>Kimi</td><td>2fps</td><td>16B(A3B)</td><td>54.6</td><td>58.7</td><td>55.78</td><td>45.64</td><td>73.48</td><td>42.08</td><td>22.52</td></tr><tr><td>4 Qwen2.5-VL-72B-Instruct</td><td>Alibaba</td><td>1fps</td><td>72B</td><td>57.00</td><td>58.00</td><td>61.00</td><td>63.00</td><td>72.00</td><td>47.00</td><td>31.00</td></tr><tr><td>5 gemini-2.5-flash</td><td>Google</td><td>2fps</td><td></td><td>56.12</td><td>59.25</td><td>59.97</td><td>59.9</td><td>72.55</td><td>44.4</td><td>29.14</td></tr><tr><td>6 GLM-4.5V</td><td>Zhipu AI</td><td>2fps</td><td>106B(A12B)</td><td>61.42</td><td>65.08</td><td>65.16</td><td>58.72</td><td>77.38</td><td>48.34</td><td>40.37</td></tr><tr><td>7 GLM-4V-Plus-0111</td><td>Zhipu AI</td><td>2fps</td><td>106B(A12B)</td><td>62.8</td><td>64.1</td><td>67</td><td>67.4</td><td>73.5</td><td>46.7</td><td>42.8</td></tr><tr><td>8 GLM-4.1V-Thinking</td><td>Zhipu AI</td><td>2fps</td><td>9B</td><td>58.68</td><td>62.48</td><td>62.98</td><td>53.85</td><td>76.18</td><td>46.23</td><td>33.51</td></tr><tr><td>9 gpt-5-chat</td><td>OpenAI</td><td>2fps</td><td></td><td>58.81</td><td>61.37</td><td>66.67</td><td>58.61</td><td>73.79</td><td>49.38</td><td>26.74</td></tr><tr><td>10 Qwen2.5-VL-7B-Instruct (Base)</td><td>Alibaba</td><td>2fps</td><td>7B</td><td>54.83</td><td>58.25</td><td>57.69</td><td>49.09</td><td>72.90</td><td>38.92</td><td>33.25</td></tr><tr><td>11 FineActionVLM-R1 (ours)</td><td></td><td>2fps</td><td>7B</td><td>57.84</td><td>62.92</td><td>60.81</td><td>51.94</td><td>73.77</td><td>43.74</td><td>31.5</td></tr></table>

Table 1: Performance comparison of diferent models on MotionBench, which evaluates fine-grained action perception across six core tasks: Motion Recognition (MR), Location-related Motion (LM), Camera Motion (CM), Motion-related Objects (MO), Action Order (AO), and Repetition Count (RC). Dev Avg is overall accuracy across questions, weighted by task size (Appendix D.1).
<table><tr><td>Model</td><td>Avg.</td><td>Seen Videos +</td><td>Unseen Videos + Accuracy(%) Unseen Questions.(%) Unseen Questions.(%)</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>63.75</td><td>60.95</td><td>66.95</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>67.2</td><td>65.92</td><td>68.66</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>78.75</td><td>76.12</td><td>81.77</td></tr><tr><td>Qwen3-VL-32B-Thinking</td><td>75.43</td><td>72.64</td><td>78.63</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>72.11</td><td>71.39</td><td>72.93</td></tr><tr><td>Qwen3-VL-30b-A3B-Thinking</td><td>71.31</td><td>72.93</td><td>69.9</td></tr><tr><td>GLM-4.5V</td><td>73.97</td><td>73.88</td><td>74.07</td></tr><tr><td>GLM-4V-plus-0111</td><td>65.07</td><td>66.67</td><td>63.25</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>45.29</td><td>45.27</td><td>45.3</td></tr><tr><td>Qwen2.5-VL-72B-Instruct</td><td>64.01</td><td>59.2</td><td>69.52</td></tr><tr><td>FineActionVLM-R1(ours)</td><td>91.50</td><td>92.54</td><td>90.31</td></tr></table>

Table 2: Performance comparison of diferent models on HAU-MCQ-bench. All models were evaluated at a frame rate of 2 fps.

Fine-grained Action Understanding. Current public video understanding benchmarks primarily focus on action recognition and classification, while generally lacking assessments for the deep underlying reasons (e.g., intent, motivation) behind fine-grained actions. To fill this gap, we have constructed a new evaluation benchmark, named HAU-MCQ-bench, specifically designed to measure a model’s reasoning capabilities in this area. This benchmark is derived from our self-constructed HAU-MCQ dataset. We extracted 753 MCQs from it to serve as the test set (HAU-MCQ-Bench), with the remaining portion forming the training set (HAU-MCQ-Train). To evaluate the model’s generalization ability, we have carefully partitioned HAU-MCQ-Bench into two distinct evaluation settings:

(1) Seen Videos, Unseen Questions: This split, accounting for 50% of the benchmark, involves questions whose corresponding videos have appeared in the HAU-MCQ-Train. The model’s comprehension of these videos may be enhanced by its exposure to other related questions during training.

(2) Unseen Videos, Unseen Questions: The other half (50%) consists of questions whose corresponding videos are entirely new and do not appear in the HAU-MCQ-Train.

## MotionBench

As shown in Table 1, our model, FineActionVLM-R1, achieves an average accuracy of 57.84% on MotionBench, marking an improvement of 3.01 percentage points over its base model, Qwen2.5-VL-7B-Instruct (54.83%). This result indicates that our proposed STARR training paradigm not only efectively mitigates hallucinations but also substantially enhances the model’s fundamental perception of underlying dynamic information. Notably, despite its 7B parameter scale, our model outperforms larger or comparable models such as Qwen3-VL-8B-Instruct (55.55%) and Kimi-VL (54.6%), and demonstrates competitive performance against leading closed-source models like gpt-5-chat (58.81%).

A detailed analysis of the sub-task scores reveals that the most significant gains are in MR and AO, the two tasks most relevant to core dynamic understanding. Our model achieves scores of 62.92% (+4.67%) and 43.74% (+4.82%) in these areas, respectively. This strongly validates the success of our method in decoupling actions from their environment and reinforcing attention on temporal sequences. Although there is a slight decrease in the RC task (-1.75%), the overall performance boost confirms the efectiveness and eficiency of the STARR paradigm in enhancing fine-grained action perception.

![](images/6f7db95d142cb947ba7fce1299ab4789f75017399ba9870cb32f7660f319c2c4.jpg)  
(b) Out of domain example 1.  
Figure 3: Caption Quality Assessment

## Caption Quality Assessment

We compare the base model, vanilla SFT, and FineActionVLM-R1 under identical prompts on in-domain and OOD videos (Figure 3). Base produces brief, coarse descriptions, while vanilla SFT adds detail but may infer unsupported actions from scene cues, such as opening a drawer and retrieving a bowl. R1 better captures action transitions and procedural detail in these examples. Further examples appear in Appendix C.1. Direct human evaluation and controlled ablations in Appendix D.2 quantify caption faithfulness and distinguish the contributions of AED and ACR.

## HAU-MCQ-Bench

We use our custom HAU-MCQ-Bench to evaluate the model’s reasoning ability regarding the underlying causes of actions. As shown in Table 2, our FineActionVLM-R1 achieves an average accuracy of 91.50% on this benchmark, which is higher than all compared baselines. Compared with its base model, Qwen2.5-VL-7B-Instruct (45.29%), it attains an absolute improvement of 46.21%. It also performs better than larger-scale models, such as Qwen3-VL-32B-Instruct (78.75%) and Qwen2.5-VL-72B-Instruct (64.01%), indicating that our training strategy efectively enhances finegrained action understanding without relying on model scale.

These quantitative results strongly indicate that our training paradigm has successfully endowed and activated the model’s ability to think about and understand fine-grained actions. The consistently high performance in both the “Seen Videos, Unseen Questions” (92.54%) and the more challenging “Unseen Videos, Unseen Questions” (90.31%) settings further demonstrates the generalizability of this acquired ability.

## Conclusions and Limitations

This work identifies Environment-Grounded Action Hallucination (EGAH) in fine-grained action understanding, where action language is spuriously grounded in static environmental cues rather than motion evidence. To mitigate EGAH, we propose STARR, a two-stage framework that redirects attention toward dynamic action cues and reinforces actioncentered grounding. We further construct FineActionR&R to support fine-grained action recognition and hierarchical reasoning. Together, these components improve action recognition, reasoning, and caption faithfulness. In the supplemental human evaluation, Full reduces AH from 28.76% to 19.71% and raises action recall from 70.23% to 75.55% relative to matched SFT. The ablations show an ACR benefit without ES, while sampling controls and additional benchmarks support the gains across the evaluated settings (Appendix D).

However, our method and model possess several limitations that highlight important avenues for future research.

Implicit Environmental Bias in Visual Tokens. Attention Rectification reduces explicit environment–action correlations but cannot eliminate implicit visual biases. Even without environmental text, salient static-object tokens may dominate attention after projection into the language space, triggering object-related action hallucinations.

Limited Generalizability to Out-of-Domain Data. Attention Consistency Reinforcement remains less robust on OOD videos, where difuse attention may overlook critical action segments and yield incomplete or incoherent descriptions.

Dependence on Data Diversity and Scale. Performance remains dependent on training-data diversity and scale. Novel actions or scenes may lead to coarse descriptions or plausible but unsupported details driven by linguistic priors.

## References

Arif, K. H. I.; Dip, S. A.; Hussain, K.; Zhang, L.; and Thomas, C. 2025. PAINT: Paying Attention to INformed Tokens to Mitigate Hallucination in Large Vision-Language Model. arXiv preprint arXiv:2501.12206.

Chen, L.-H.; Lu, S.; Zeng, A.; Zhang, H.; Wang, B.; Zhang, R.; and Zhang, L. 2024a. MotionLLM: Understanding Human Behaviors from Human Motions and Videos. arXiv preprint arXiv:2405.20340.

Chen, Z.; Zhao, Z.; Luo, H.; Yao, H.; Li, B.; and Zhou, J. 2024b. Halc: Object hallucination reduction via adaptive focal-contrast decoding. arXiv preprint arXiv:2403.00425.

Choi, J.; Lee, S.; Kim, J.; Bae, K.; Lee, S.; and Lee, G. 2025. MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations. arXiv:2503.15871.

Chu, X.; Chen, X.; Wang, G.; Tan, Z.; Huang, K.; Lv, W.; Mo, T.; and Li, W. 2025. Qwen Look Again: Guiding Vision-Language Reasoning Models to Re-attention Visual Information. arXiv preprint arXiv:2505.23558.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Epstein, D.; Chen, B.; and Vondrick, C. 2019. Oops! Predicting Unintentional Action in Video. arXiv preprint arXiv:1911.11206.

Favero, A.; Zancato, L.; Trager, M.; Choudhary, S.; Perera, P.; Achille, A.; Swaminathan, A.; and Soatto, S. 2024. Multimodal hallucination control by visual information grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14303–14312.

Goyal, R.; Ebrahimi Kahou, S.; Michalski, V.; Materzynska, J.; Westphal, S.; Kim, H.; Haenel, V.; Fruend, I.; Yianilos, P.; Mueller-Freitag, M.; et al. 2017. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, 5842–5850.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Hong, W.; Cheng, Y.; Yang, Z.; Wang, W.; Wang, L.; Gu, X.; Huang, S.; Dong, Y.; and Tang, J. 2025. MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models. arXiv:2501.02955.

Jia, B.; Lei, T.; Zhu, S.-C.; and Huang, S. 2022. EgoTaskQA: Understanding Human Tasks in Egocentric Videos. In The 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks.

Liu, Y.; Wang, L.; Wang, Y.; Ma, X.; and Qiao, Y. 2022. Fineaction: A fine-grained video dataset for temporal action localization. IEEE transactions on image processing, 31: 6937–6950.

Lyu, X.; Chen, B.; Gao, L.; Shen, H.; and Song, J. 2024. Alleviating hallucinations in large vision-language models through hallucination-induced optimization. Advances in Neural Information Processing Systems, 37: 122811– 122832.

Rizve, M. N.; Shah, M.; and Dave, I. R. 2024. FinePseudo: Improving Pseudo-Labelling through Temporal-Alignablity for Semi-Supervised Fine-Grained Action Recognition. arXiv:2409.01448.

Salah, A. A.; Lu, H.; Poppe, R.; and Jian, H. 2024. Enhancing Video Transformers for Action Understanding with VLMaided Training. arXiv:2403.16128.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Soomro, K.; Zamir, A. R.; and Shah, M. 2012. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402.

Team, Q. 2025. Qwen2.5-VL.

Tu, C.; Zhang, L.; Chen, P.; Ye, P.; Zeng, X.; Cheng, W.; Yu, G.; and Chen, T. 2025. Favor-bench: A comprehensive benchmark for fine-grained video motion understanding. arXiv preprint arXiv:2503.14935.

Wu, J.; Fan, Z.; Jin, Y.; Li, Z.; Diao, Y.; Zhang, X.; Song, Z.; Li, Z.; and Gu, X. 2025. HieroAction: Hierarchically Guided VLM for Fine-Grained Action Analysis. arXiv:2508.16942.

Yang, J.; Cui, C.; Zhou, Y.; Chen, Y.; Xia, P.; Wei, Y.; Yu, T.; Huang, Y.; and Wang, L. 2025a. Ikod: Mitigating visual attention degradation in large vision-language models. arXiv preprint arXiv:2508.03469.

Yang, Z.; Luo, X.; Han, D.; Xu, Y.; and Li, D. 2025b. Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key. In Proceedings of the Computer Vision and Pattern Recognition Conference, 10610–10620.

Zhou, Y.; Cui, C.; Yoon, J.; Zhang, L.; Deng, Z.; Finn, C.; Bansal, M.; and Yao, H. 2023. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754.

## A FineActionR&R Dataset Statistics

To comprehensively evaluate both fine-grained human action understanding and temporal-causal reasoning, we construct the FineActionR&R dataset suite. It consists of two complementary components:

• (1) Fine-grained Human Action — focusing on detailed action perception, segmentation, and basic reasoning.

• (2) Temporal Action Sequencing and Causality Multiple-Choice Question (TASC-MCQ) — targeting higher-level temporal and causal reasoning over complex action sequences.

Together, these two parts establish a diverse and scalable benchmark for evaluating models’ multi-level understanding of dynamic human actions.

## A.1 Fine-grained Human Action

To build the fine-grained human action dataset suite, which includes Fine-Action-Caption, Fine-Action-Segment, HAU-QA, and HAU-MCQ, we carefully sampled video clips from a wide range of mainstream public video datasets. The selected sources encompass videos from FineAction (Liu et al. 2022) (162 videos), UCF101 (Soomro, Zamir, and Shah 2012) (255 videos), Movid (Chen et al. 2024a) (175 videos), Egotaskqa (Jia et al. 2022) (133 videos), Oops (Epstein, Chen, and Vondrick 2019) (76 videos), and Something-Something V2 (Goyal et al. 2017), forming a foundational pool of 857 unique videos.

This diverse mixture ensures a balanced coverage across action scales and viewpoints: UCF101 contributes standard third-person actions; Egotaskqa focuses on egocentric, task-oriented interactions; Something-Something V2 emphasizes fine-grained, object-centric manipulations; while FineAction, Movid, and Oops add complex, long-horizon, and unexpected behaviors.

Building on this video base, we derived the following datasets with distinct annotation methodologies:

• Fine-Action-Segment: fully manually annotated temporal segments;

• HAU-QA and HAU-MCQ: automatically generated with Gemini 2.5 Pro, followed by manual spot-checking for correctness and quality.

Figure 4 illustrates the dataset composition and annotation statistics, while Figure 5 presents a qualitative annotation example.

![](images/2869415bb337b8383f5681b38cbf7ecd4f7d293c1f2cb793c5e25b6eedb21b1e.jpg)

![](images/7376096af309d20548fa68109cc26f01555285fb1593ee290aeaa191f20854bd.jpg)

![](images/5f06ea6c2bbda0a09d1f67b895ff24dc905ed6b33f7fd553bfab46e3f18c3c1a.jpg)  
(a) Video source distribution across (b) Distributions of video six datasets. duration and caption length.

![](images/f7493291ecc210ad7276737a68c91b3c7a69014cacc19d7c406874e0d732561f.jpg)  
(c) Sample counts of each Fine-grained Human Action subset.

Figure 4: Statistics of the Fine-grained Human Action dataset. (a) Video source distribution across six public datasets, showing balanced diversity in action types and visual domains. (b) Distributions of video durations and caption word counts, indicating that clips are generally short and accompanied by concise textual descriptions. (c) Counts of data samples across four derived subsets—Fine-Action-Caption, Fine-Action-Segment, HAU-QA, and HAU-MCQ— demonstrating the overall scale of the dataset suite.

![](images/aef40570c0bcf02488dc915c17e136741fb18981a1bab44f478661d044c87f67.jpg)  
Figure 5: An example annotation from the Fine-grained Human Action dataset.

## A.2 Temporal Action Sequencing and Causality Multiple-Choice Question Dataset

The second component of FineActionR&R is the Temporal Action Sequencing and Causality Multiple-Choice Question (TASC-MCQ) dataset, designed to evaluate a model’s reasoning capability over temporal and causal relationships in video sequences. It was constructed using a combination of clips from the Fine-Action-Caption and FAVOR-Bench (Tu et al. 2025) datasets, totaling 1,948 videos.

Using Gemini-2.5-Pro(Comanici et al. 2025), we generated multiple-choice questions covering five aspects of temporal-causal reasoning, and each question features a unique correct answer and semantically-relevant distractors that serve as hard negatives, driving the model to learn finegrained visual discrimination.:

• Concurrent Actions: Identifying simultaneous actions performed by one or more subjects;

• Local Sequence: Determining an action immediately preceding or following a key event;

• Action Duration & Transition: Assessing durations or transitions between actions;

• Global Ordering: Arranging three or more non-adjacent actions in the correct order;

• Causality: Identifying an action that directly triggers a later outcome.

To ensure data quality, the generated questions underwent a two-stage refinement process. First, we employed DeepSeek-R1 (Guo et al. 2025) to filter out linguisticallyanswerable questions, ensuring genuine visuo-temporal reasoning requirements. Then, manual spot-checking was performed to verify correctness. The final dataset contains 6,134 high-quality MCQs focused on fine-grained temporal reasoning. Figure 6 summarizes the distribution of diferent question types.

## B Training Detail

## B.1 Hyperparameters

In this SFT phase, the model is trained for two epochs using a constant learning rate of $1 \times 1 0 ^ { - 6 }$ . For video data preprocessing, we sample frames at a rate of 2 FPS. To handle inputs of varying resolutions uniformly, we cap the maximum number of pixels per video frame at 100,352.

In GRPO stage largely inherits the training configuration from the SFT phase. Specifically, the learning rate is maintained at $1 \times \mathrm { { 1 0 ^ { - 6 } } }$ , and the video processing parameters, including the sampling rate (2 FPS) and the maximum pixel count (100,352), remain identical. During the sampling step of the GRPO process, we generate four candidate samples for each input prompt using a sampling temperature of 1.0. A core component of this stage is a custom Attention Consistency Reward function. This reward is formulated as a weighted sum of a mean consistency term and a variance consistency term, with their respective coeficients set to $\lambda _ { \mathrm { m e a n } } = 0 . 7$ and $\lambda _ { \mathrm { v a r } } = 0 . 3$ . Furthermore, the reward calculation balances an absolute component and a relative component, governed by the coeficients $\alpha _ { \mathrm { a b s } } = 0 . 7$ and $\alpha _ { \mathrm { r e l } } = 0 . 3$ respectively.

![](images/7f9c50bff7e100023e210c6779ad07b114c405b21c61a2cac01a327c9bc0d633.jpg)

(a) Distribution of question types in TASC-MCQ.  
![](images/241def7cf139ae2091d3dd73866cf7f90b3a4b238666711b2d207354d68e8f62.jpg)  
(b) Sample counts for each question type.

Figure 6: Statistics of the TASC-MCQ dataset. (a) Distribution of the five question types, including Concurrent Actions, Local Sequence, Action Duration & Transition, Global Ordering, and Causality. (b) Counts of samples within each question type, showing a balanced coverage across diverse temporal and causal reasoning categories.

## B.2 GRPO Reward Function

In the GRPO training stage, we designed a composite reward function to synergistically optimize the model’s generation behavior across multiple dimensions, particularly for fine-grained action understanding and description tasks. This function is composed of the following core components, each targeting one or more specific objectives to collectively guide the model toward the desired convergence.

1. Attention Consistency Reward: ACR is applied to the fine-grained action captioning task. It rewards strong mean attention and balanced coverage across referenceselected visual tokens after averaging over generated text positions. It does not directly measure attention stability across decoding steps. Appendix D.5 validates the token selection and clarifies this distinction.

2. Accuracy Reward: This reward ensures the faithfulness and correctness of the generated content. Its form is contingent on the task type:

• For Multiple-Choice Questions (HAU-MCQ, TASC-MCQ): We provide a fixed positive reward if the model’s selected answer option (e.g., “A”, “B”) matches the ground truth, and zero reward otherwise. This provides direct binary feedback on correctness.

• For the action captioning task (Fine-Action-Caption): We calculate the ROUGE score between the generated caption and the ground-truth caption, using the score as the reward value. This serves to maintain the factual accuracy of the generated content and prevents the model from deviating excessively from the visual facts while optimizing for other rewards like ACR.

![](images/af01cd2e2d145577f175d68a4bf0b0cece7435078e5efd074f9f64a6b9951ff9.jpg)  
Figure 7: Temporal Action Sequencing and Causality MCQ.

3. Formatting Reward: We employ a reward based on regular expressions to enforce specific output formats required by diferent tasks. This is crucial for guiding the model to perform structured reasoning and provide structured answers.

• For Multiple-Choice Questions: The output is required to strictly adhere to the <think>...</think> <answer>...</answer> structure. This format compels the model to first generate a Chain-of-Thought (CoT) before providing the final answer, thereby explicitly stimulating its reasoning process.

• For the action captioning task: The output format is expected to be <answer>...</answer>.

4. Length Reward: This reward encourages the model to generate text of an appropriate length, avoiding outputs that are either too long or too short. Its implementation also varies by task:

• For Multiple-Choice Questions: We reward the length of the Chain-of-Thought within the <think> block for falling within a predefined reasonable range. This encourages suficient but not overly verbose reasoning.

• For the action captioning task: We reward the length of the text within the <answer> block. This design is critical as it directly counteracts a potential side efect of optimizing ACR alone. We observed that without this reward, ACR would incentivize the model to generate shorter captions (as shorter text makes it easier to maintain high attention consistency). However, this leads to coarser-grained descriptions that omit fine-grained dynamic details. The length reward thus ensures that the model can generate content-rich, detailed descriptions while maintaining a focused attention.

5. Environmental Information Suppression Reward: To consolidate the gains from the SFT stage in mitigating EGAH, we introduce this reward. We observed that the inherent “attention sink” problem in Vision-Language Models can cause some visual tokens identified by ACR as “action-relevant” to actually correspond to static objects in the background. Without suppression, the model might re-introduce descriptions of the environment during the RL stage, causing the EGAH problem to resurface. To address this, we employ a powerful external Large Language Model as a judge to evaluate the content generated within the <answer> block. A positive reward is given if the judge determines that the content is free of descriptions of irrelevant environmental information. This measure penalizes descriptions of the extraneous environment and reinforces the model’s focus on the dynamic action itself.

## C Caption Quality Analysis

## C.1 Qualitative Caption Comparisons

In the in-domain evaluation on videos in Figure 8a and Figure 8b, the Qwen2.5VL base model provided generally coarse descriptions of the video content. For video in Figure 8a, the model was able to identify the overall motion pattern of the shot-put activity but failed to clearly disentangle and articulate the fine-grained motion details. In video in Figure 8b, the model only produced a global description of the athlete’s movement in the stadium, accompanied by partial hallucinations that extended beyond the visual evidence.

The vanilla SFT model generated longer and more content-rich text compared with Qwen2.5VL, providing additional contextual details at the scene level. However, it did not show a substantial improvement in fine-grained action understanding. In video in Figure 8a, the model tended to hallucinate due to the presence of background objects such as a horizontal bar, incorrectly associating the scene with a long jump activity. Similarly, in video in Figure 8b, it generated several speculative descriptions about the athlete’s movement based on perceptual inference rather than visible evidence.

The FineActionVLM-R1 model demonstrated a more structured understanding of the actions in video in Figure 8a. It accurately decomposed the athlete’s movements in the shot-put sequence—from bending down for preparation to releasing the shot—providing a higher level of detail compared with the human-provided annotations. Although the motions of other participants were not mentioned, the main actor’s actions were described with considerable granularity. In video in Figure 8b, FineActionVLM-R1 consistently focused on the primary athlete and precisely delineated the bowler’s entire motion trajectory from the run-up to the ball release.

For the out-of-domain evaluation on videos in Figure 8c and Figure 8d, the Qwen2.5VL base model introduced substantial hallucinations in video in Figure 8c. The initial hallucinations arose from misinterpreting contextual elements such as soy sauce bottles on the kitchen counter, while the later ones appeared as imaginative extensions of the observed actions. In video in Figure 8d, no hallucinations were detected, but the model presented only coarse descriptions of the actors’ activities, omitting significant motion details.

![](images/5a606c0ba4b0b195737b8cc4bf987a3693f997fc9734eefd9aafbc262dab7a6d.jpg)  
(c) Out-of-domain sample 2.  
(d) Out-of-domain sample 3.  
Figure 8: Caption Quality Supplementary Assessment.

The vanilla SFT model exhibited similar behavior on video in Figure 8c, with comparable levels of hallucination and fine-grained description as Qwen2.5VL, though it incorporated slightly more environmental context. On video in Figure 8d, the model provided a more complete narrative of the sauce-pouring process but still included minor hallucinatory elements.

In contrast, the FineActionVLM-R1 model generated hallucination-free descriptions for video in Figure 8c. While its fine-grained depiction was less detailed than that observed on in-domain videos, it clearly conveyed the overall procedural flow of the action. For video in Figure 8d, FineActionVLM-R1 demonstrated a precise understanding of both spatial and temporal dimensions, comprehensively describing the process of mixing sauces with clear and structured articulation of the corresponding actions.

## C.2 OOD Environment-Description Analysis

To validate the eficacy of our proposed “Attention Rectification” strategy, we conducted an in-depth comparison between our model trained with this method (FineActionVLM-R1- Dev1) and a baseline model trained using only Vanilla SFT. Under identical conditions (temperature = 1, fps = 4), both models were evaluated on 11 diferent OOD videos. For each video, 25 responses were generated, resulting in a total of 275 samples per model. Using the DeepSeek-R1 evaluator, we analyzed the proportion of responses containing environmentally descriptive content that was irrelevant to human actions. The results show that 23.63% of the Vanilla SFT model’s responses contained such irrelevant environmental descriptions, whereas this proportion dropped to 15.63% for FineActionVLM-R1-Dev1, an absolute reduction of 8.00 percentage points. This metric measures irrelevant environmental descriptions, rather than unsupported action claims, and is therefore indirect evidence. The 275 responses per model are repeated generations from 11 videos, not 275 independent videos. Direct human-rated action hallucination and action recall are reported separately in Appendix D.2.

![](images/d5b4e26172db85a20cef1905477321b4eed6dd42899d2ed81dd5d43b7722c723.jpg)  
Figure 9: FineActionVLM-R1-Dev1 vs. Vanilla SFT

When processing OOD videos, FineActionVLM-R1- Dev1 demonstrates significant advantages. As illustrated by the qualitative examples in Figures 9, the descriptions generated by FineActionVLM-R1-Dev1 are consistently superior in overall quality to those from the baseline. Specifically, it is capable of capturing and articulating more fine-grained dynamic action details, its generated text is richer and more comprehensive, and the phenomenon of hallucination— generating content not present in the video—is efectively mitigated.

However, we also acknowledge that FineActionVLM-R1- Dev1 is not without its limitations. It still produces hallucinatory content in certain instances, and its generation quality can exhibit a degree of instability. We posit that these limitations may stem from two primary factors: the inherent capability ceiling of the foundation model used, and a lack of suficient diversity in the training data, which constrains the model’s generalization ability on certain video types. A noteworthy observation is that for videos that are either exceptionally simple or dificult, the performance gap between the two models is less pronounced. This further corroborates our hypothesis that the model’s ultimate performance is constrained by both the training strategy and the fundamental dimensions of the base model and data.

Despite these challenges, the model trained with Attention Rectification provides a superior policy starting point for our subsequent optimization stages. Although its generated candidate descriptions may have occasional flaws, their overall quality is higher, laying a more solid foundation for our subsequent GRPO stage. This implies that GRPO can learn and optimize within a higher-quality and more promising candidate space, which is expected to lead to a higher performance ceiling. This finding also highlights future avenues for improvement: namely, that adopting a more powerful foundation model or further expanding the diversity of the training data could significantly enhance the model’s overall performance.

## D Ablation Studies and Extended Evaluation

Complementing the caption analyses in Appendix C, we evaluate human-rated action hallucination, isolate component contributions, and examine sampling sensitivity, task adaptation, token selection, and generalization. These experiments distinguish caption faithfulness from multiple-choice accuracy and examine the contributions of AED, standard GRPO, ACR, and Environmental Information Suppression (ES). Unless otherwise specified, Base denotes Qwen2.5- VL-7B-Instruct and Full or R1 denotes FineActionVLM-R1.

## D.1 Evaluation Protocol and Data Organization

Human evaluation. The human evaluation covers 400 videos and involves five annotators. Each video is independently assessed by two annotators drawn from this pool, providing overlapping paired assessments across the annotation team. Model identities are blinded during evaluation. The reported inter-annotator agreement score is 0.82. Action hallucination (AH) is the percentage of captions containing actions unsupported by the video; action recall measures coverage of the ground-truth actions. We report both metrics because reducing unsupported claims is useful only if relevant action content is retained. AH directly evaluates action errors, while attributing a particular error to environmental cues requires additional controlled evidence.

Video sampling and source review. The dataset sampling policy uses a 1:1 ratio of first-person to third-person videos. Source-video deduplication involves 15 reviewers, with review within groups followed by cross-checking across groups to prevent duplicate original videos in the source pool. This source-review procedure is separate from the five-person human-evaluation protocol above. Multiple captions or questions can be derived from a single source video; source deduplication therefore does not imply that every derived annotation uses a diferent video.

HAU split scope. HAU-MCQ-Bench contains 753 test questions. The seen-video subset contains new questions about videos seen in HAU-MCQ-Train; the unseen-video subset contains both videos and questions unseen relative to HAU-MCQ-Train. Throughout this evaluation, “unseen” is defined relative to HAU-MCQ-Train.

Training and comparison structure. The three-stage recipe in the main-text Training Recipe subsection contains an auxiliary reasoning-data stage and the two STARR optimization stages. First, GRPO on TASC-MCQ produces FineActionVLM-R1-Zero, which supplies distilled reasoning chains. Second, multi-task SFT starts from the base model, using fine-action captions, HAU-QA, distilled reasoning data, and AED construction where enabled, to obtain R1-Dev1. Third, GRPO starts from R1-Dev1 and uses captioning and multiple-choice tasks to obtain R1. Thus, R1- Zero supplies supervision for SFT; it is not the initialization checkpoint for that SFT stage.

Matched SFT is the supervised adaptation control. The component comparisons use matched supervision, with AED changing the construction of the SFT video input and ACR and ES varied within the GRPO stage. Base-to-SFT comparisons include both exposure to the task data and supervised optimization. Subsequent comparisons measure conditional component efects; they do not separately ablate every constituent dataset. Training hyperparameters and the shared task rewards are given in Appendix B.

<table><tr><td>Training</td><td>MB↑</td><td>AH↓</td><td>Recall↑</td></tr><tr><td>Base</td><td>54.83</td><td>29.78</td><td>66.41</td></tr><tr><td>Matched SFT</td><td>55.27</td><td>28.76</td><td>70.23</td></tr><tr><td>SFT+AED</td><td>55.83</td><td>25.24</td><td>71.91</td></tr><tr><td>SFT+G</td><td>55.69</td><td>26.60</td><td>71.53</td></tr><tr><td>SFT+AED+G</td><td>56.21</td><td>23.53</td><td>72.74</td></tr><tr><td>SFT+AED+G+A</td><td>56.68</td><td>21.30</td><td>74.28</td></tr><tr><td>SFT+AED+G+E</td><td>56.60</td><td>22.32</td><td>73.60</td></tr><tr><td>Full (+A+E)</td><td>57.84</td><td>19.71</td><td>75.55</td></tr></table>

Table 3: Component ablations at 2 FPS. All values are percentages. MB is MotionBench Dev accuracy; AH and recall are human-evaluation metrics. G=GRPO, A=ACR, E=ES. Full adds both A and E to SFT+AED+G.

MotionBench aggregation. The Dev Avg column in Table 1 is the overall accuracy pooled across questions. If task k contains $n _ { k }$ questions and has percentage accuracy $a _ { k } ,$ the reported value is

$$
{ \mathrm { A v g } } = { \frac { \sum _ { k = 1 } ^ { 6 } n _ { k } a _ { k } } { \sum _ { k = 1 } ^ { 6 } n _ { k } } } .\tag{8}
$$

It is therefore weighted by task size, rather than the unweighted mean of the six displayed task accuracies. This explains why directly averaging those columns does not reproduce the reported 54.83% for Base or 57.84% for R1. All diferences below are expressed in percentage points (pp), unless explicitly stated otherwise.

## D.2 Component Ablations and Human-Rated Faithfulness

Table 3 combines MotionBench accuracy with human-rated AH and action recall. We use G for standard GRPO, A for ACR, and E for ES. The A-only and E-only reward additions share the SFT+AED+G control; Full includes both ACR and ES.

Data intervention and standard GRPO. Adding AED to matched SFT lowers AH from 28.76% to 25.24% (3.52 pp) and raises recall from 70.23% to 71.91% (1.68 pp). Standard GRPO without AED reduces AH to 26.60%. Combining AED with GRPO reduces it further to 23.53%, a 3.07 pp reduction relative to SFT+G. The corresponding MotionBench accuracy increases from 55.69% to 56.21%. These comparisons support a contribution from AED beyond supervised adaptation or standard GRPO alone.

ACR independently ofES. On top ofSFT+AED+G, ACR lowers AH from 23.53% to 21.30% and raises recall from 72.74% to 74.28%. The AH diference is −2.23 pp with a paired 95% confidence interval of [−3.68, −0.78] pp. Because neither configuration includes ES, this improvement cannot be attributed solely to the environmental-suppression reward. Adding ES alone yields 22.32% AH and 73.60% recall; adding both rewards gives the best observed values, 19.71% and 75.55%.

<table><tr><td>Comparison (treatment — control)</td><td>∆AH</td><td>95% CI</td></tr><tr><td>SFT+AED – matched SFT</td><td>-3.52</td><td>[−5.31, −1.73]</td></tr><tr><td>Add ACR to SFT+AED+G, without</td><td>-2.23</td><td>[-3.68, -0.78]</td></tr><tr><td>ES R1 – Base at 4 FPS</td><td></td><td>-11.11 [-13.42, −8.80]</td></tr></table>

Table 4: Reported paired 95% confidence intervals for AH diferences, in percentage points. Negative diferences favor the treatment.
<table><tr><td rowspan="2">FPS</td><td colspan="2">MotionBench↑</td><td colspan="2">AH↓</td></tr><tr><td>Base</td><td>R1</td><td>Base</td><td>R1</td></tr><tr><td>1</td><td>53.43</td><td>56.29</td><td>32.02</td><td>22.16</td></tr><tr><td>2</td><td>54.83</td><td>57.84</td><td>29.78</td><td>19.71</td></tr><tr><td>4</td><td>55.82</td><td>59.13</td><td>28.71</td><td>17.60</td></tr></table>

Table 5: Frame-sampling controls. All values are percentages. The advantage of R1 persists at every evaluated sampling rate.

Full-system comparison. Relative to matched SFT, Full improves MotionBench by 2.57 pp, reduces AH by 9.05 pp, and improves recall by 5.32 pp. The joint decrease in AH and increase in recall supports more faithful descriptions with greater action coverage, rather than a gain obtained simply by omitting actions. Table 4 collects the paired intervals reported for the targeted comparisons. These intervals concern AH diferences, not MotionBench accuracy or variation across training seeds.

## D.3 Frame-Sampling Controls

Sparse sampling can omit short actions. We therefore compare Base and R1 at 1, 2, and 4 FPS, using identical prompts, decoding settings, and visual-token limits across the comparison. Table 5 reports both MotionBench accuracy and AH.

Denser sampling benefits both models. Increasing Base from 2 to 4 FPS improves MotionBench by 0.99 pp and lowers AH by 1.07 pp. At the same 4 FPS rate, however, R1 remains 3.31 pp higher on MotionBench and 11.11 pp lower in AH than Base, with the AH confidence interval shown in Table 4. The gains therefore persist under denser sampling; increasing frame rate alone does not explain the observed model diference under these evaluation settings.

## D.4 HAU-MCQ: Task Adaptation and Visual Input

Table 6 compares Base, matched SFT, R1, and R1 without video input on HAU-MCQ-Bench. Video-input evaluations use 2 FPS. The benchmark contains 753 questions in total, with “unseen” defined as in Appendix D.1.

Matched SFT improves overall accuracy by 40.57 pp over Base, accounting for much of the 46.21 pp Base-to-R1 gain.

<table><tr><td>Model/input</td><td>Overall↑</td><td>Unseen↑</td></tr><tr><td>Base</td><td>45.29</td><td>45.30</td></tr><tr><td>Matched SFT</td><td>85.86</td><td>82.83</td></tr><tr><td>R1</td><td>91.50</td><td>90.31</td></tr><tr><td>R1, no video</td><td>37.05</td><td>33.36</td></tr></table>

Table 6: HAU-MCQ accuracy (%). Unseen denotes unseen videos and questions relative to HAU-MCQ-Train.

<table><tr><td>Selection</td><td></td><td>Temporal Spatiotemporal</td></tr><tr><td>Matched random</td><td>39.71</td><td>20.46</td></tr><tr><td>Reference top 25%</td><td>64.54</td><td>44.06</td></tr></table>

Table 7: Overlap with annotated action evidence (%). Reference selection improves temporal overlap by 24.83 pp and spatiotemporal overlap by 23.60 pp.

R1 improves over matched SFT by a further 5.64 pp overall and 7.48 pp on unseen videos. This separates the substantial efect of task adaptation from the subsequent improvement of the full training pipeline. Consequently, the headline comparison with larger models should be interpreted in light of diferences in training exposure, rather than as a controlled comparison of model scale alone.

Removing video reduces R1 accuracy by 54.45 pp overall and 56.95 pp on the unseen subset. This provides evidence that visual input materially contributes to the result. It does not imply that all linguistic or question-format cues have been eliminated. HAU-MCQ and MotionBench assess diferent capabilities and involve diferent training exposure, so their gains need not have the same magnitude.

## D.5 Validation of Reference-Selected Visual Tokens

ACR selects the top 25% of visual tokens according to the reference model’s attention to the ground-truth caption. We compare these selections with per-video random selections matched in token count, using annotated action evidence, including the temporal intervals in Fine-Action-Segment. Table 7 reports temporal and spatiotemporal overlap.

Reference-selected tokens align more closely with annotated evidence than the matched random control. This validates the selection empirically, while allowing for residual reference-model bias. Overlap establishes association with annotated evidence; it does not establish that the selected tokens causally determine action predictions.

Interpretation ofACR. The attention value $a _ { i }$ in the maintext Attention Consistency Reinforcement formulation is averaged over generated text positions before $R _ { \mathrm { v a r } }$ is computed across selected visual tokens. The variance term therefore encourages balanced caption-averaged coverage of those tokens. Two attention trajectories with identical averages receive the same variance reward even if their behavior across decoding steps difers. ACR promotes strong and balanced attention to the selected evidence; it does not directly measure or guarantee generation-time stability.

<table><tr><td>Training</td><td>Correct pairs/100</td><td>AH (%)</td></tr><tr><td>Matched SFT</td><td>54</td><td>28.76</td></tr><tr><td>SFT+AED</td><td>71</td><td>25.24</td></tr></table>

Table 8: Same-scene action discrimination and separate human-rated AH. AH is not computed from the 100-pair task.

<table><tr><td>Model (7B)</td><td colspan="2">Video-MME accuracy (%)↑</td></tr><tr><td>Base</td><td></td><td>56.00</td></tr><tr><td>Matched SFT</td><td></td><td>56.45</td></tr><tr><td>Full</td><td></td><td>57.77</td></tr></table>

Table 9: General video-understanding performance. Full improves over matched SFT by 1.32 pp.

## D.6 Same-Scene, Diferent-Action Discrimination

AED changes the clips surrounding an action segment while retaining the original scene and object cues within that segment. To examine action discrimination when the scene is shared, we additionally evaluate 100 same-scene/diferentaction video pairs. Table 8 reports correct-pair counts alongside AH from the separate human-caption evaluation.

AED increases correct-pair counts from 54 to 71. Separately, its AH reduction relative to matched SFT is 3.52 pp, with a paired 95% confidence interval of [−5.31, −1.73] pp. Together, these results support improved action discrimination and caption faithfulness. The intervention targets correlations with surrounding context; it does not remove all scene or object information inside the action clip or establish complete scene–action disentanglement.

## D.7 General Video Understanding and Model Scale

We evaluate general video understanding on Video-MME and test the training framework at the 3B scale within the Qwen2.5-VL family. Tables 9 and 10 summarize the results.

On Video-MME, Full gains 1.77 pp over Base and 1.32 pp over matched SFT, extending the observed improvement beyond the fine-grained benchmarks. At 3B scale, full training improves MotionBench by 1.91 pp and reduces AH by 7.33 pp. These results support transfer across the evaluated 3B and 7B scales within Qwen2.5-VL. They do not yet establish transfer across distinct backbone architectures.

## D.8 Discussion of Caption Faithfulness

Appendix C.2 evaluates 11 OOD videos at 4 FPS, with 25 generated responses per video and model, yielding 275 responses per model. Its DeepSeek-R1-based score measures irrelevant environmental descriptions: 23.63% for Vanilla SFT and 15.63% for R1-Dev1, an absolute decrease of 8.00 pp. This score is distinct from human-rated AH. Repeated generations from 11 videos do not constitute 275 independent videos, and the environmental-description score is indirect evidence rather than a measurement of unsupported actions. The human evaluation supplies the direct captionfaithfulness measurement, while recall checks action coverage.

<table><tr><td>Qwen2.5-VL-3B</td><td>MotionBench↑</td><td>AH↓</td></tr><tr><td>Base</td><td>51.49</td><td>31.22</td></tr><tr><td>Full training</td><td>53.40</td><td>23.89</td></tr></table>

Table 10: Transfer to the 3B backbone. All values are percentages.

Taken together, these experiments show that AED and ACR provide conditional improvements beyond the matched SFT and standard GRPO controls, including an ACR benefit without ES. These gains coexist with higher action recall, persist under denser frame sampling, and extend to Video-MME and a smaller model within the same family. Token overlap and same-scene discrimination provide complementary evidence of improved use of action-related information. The supported conclusion is improved action faithfulness and discrimination under the evaluated settings, with the scope of each mechanism grounded in its actual objective and controls.