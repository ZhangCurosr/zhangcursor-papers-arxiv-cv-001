# EVAThink: Evidence Sufficiency-Guided Training for Adaptive Thinking in Video Understanding

Changti Wu<sup>1,2</sup>, Ruizhe Fan<sup>3</sup>, Bin Wang<sup>3</sup>, Lei Zhang<sup>1\*</sup>, Yuxiang Zhang<sup>3\*</sup>

<sup>1</sup>East China Normal University <sup>2</sup>Zhongguancun Academy <sup>3</sup>Huawei Technologies Co., Ltd

## Abstract

Under a fixed frame budget, video understanding with vision-language models (VLMs) depends on whether the sampled frames provide sufficient evidence for the question. Sufficient frame evidence supports direct answering, whereas less informative observations may benefit from reasoning that connects the available cues. We introduce EVAThink, which enhances video understanding through evidence sufficiency-guided training for adaptive thinking. First, Recoverable Evidence Gap Mining constructs evidence-dense and evidence-sparse video views that share the same question and answer under an identical frame budget. It retains pairs where sparse views yield lower direct-answer accuracy than dense views but benefit from explicit reasoning, providing supervisionfor responsemode selection. Second, Evidence-Guided Two-Stage Reinforcement Learning uses stage-specific rewards to guide training with evidence-based mode preferences and then refine response efficiency, while first-token advantage regularization mitigates mode collapse. Trained on only 6.7K curated samples, EVAThink consistently improves upon the base model across video understanding benchmarks under both 32- and 64-frame budgets, while outperforming strong baselines on average. The code, model, and datasets are available at EVAThink.

## 1. Introduction

Vision-language models (VLMs) have made substantial progress in video understanding, including video captioning, temporal grounding, and video question answering [23, 30, 37, 38, 40, 52]. In practice, these models often operate under a limited visual budget, accessing a sampled sequence of frames rather than the complete video. Their answers therefore depend on the question-relevant information preserved in those frames. We refer to this information as frame evidence, which includes visual cues within individual frames and temporal relationships across frames.

![](images/3379481f328cc11612407ed67c976f624f370a62e5b62b3fb259d015ca702ea6.jpg)  
Figure 1. Motivation for evidence sufficiency-guided training for adaptive thinking. For the same question and frame budget, sufficient frame evidence supports direct answering, while thinking can help make better use of limited frame evidence.

With a fixed budget of uniformly sampled frames, some sampled sequences provide sufficient evidence for a reliable direct answer, while others offer only sparse coverage of question-relevant content, particularly when that content occupies a small fraction of the video. We term the extent to which frame evidence supports reliable direct answering by the model evidence sufficiency, an important consideration when allocating reasoning computation.

Explicit reasoning can help models interpret frame evidence by connecting observations across frames. However, when the evidence already supports a reliable direct answer, additional reasoning may yield little benefit while increasing generation cost and introducing unsupported inferences. When useful cues are present but require further analysis, reasoning may instead improve the answer. This motivates using evidence sufficiency to guide adaptive-thinking training, encouraging direct answering when frame evidence is sufficient and reasoning when a direct answer is less reliable but further analysis may help.

Recent work on adaptive reasoning for video understanding selects response modes using initial-answer confidence [25] or objectives that balance answer quality, including correctness and grounding, against generation cost [19, 31]. These approaches provide useful signals for mode selection given a visual input, but supervision across different evidence conditions remains less explored. Such supervision could use observed differences in direct-answer accuracy and reasoning gains to guide adaptive-thinking training.

Learning such a response-mode policy also poses challenges in training stability. Prior studies report that adaptive reasoning training can prematurely favor one mode, limiting exploration of the other and potentially leading to mode collapse, in which the model consistently answers directly or always reasons [7, 25, 32]. Existing methods can also require substantial training data, e.g., 83K samples for VideoAuto-R1 [25] and 140K for Video-FLAIR [19]. Some further rely on a supervised fine-tuning (SFT) cold start, adding the burden of preparing and annotating response demonstrations [19, 31].

To address these challenges, we introduce EVAThink, which learns adaptive thinking from differences in frame evidence sufficiency. During data curation, our Recoverable Evidence Gap Mining constructs evidence-contrastive pairs of dense and sparse video views that share the same question and answer under a fixed frame budget. We retain pairs with higher direct-answer accuracy on the dense view, lower direct-answer accuracy on the sparse view, and a sufficient accuracy gain from explicit reasoning on the sparse view. These pairs capture a recoverable evidence gap, linking differences in frame evidence sufficiency to the benefit of reasoning. Building on these pairs, we develop Evidence-Guided Two-Stage Reinforcement Learning, which requires no SFT cold start. Using stage-specific rewards, the first stage guides response-mode training with evidence-based mode preferences from high-confidence pairs while rewarding answer correctness; the second removes viewdependent mode preferences and adds a small bonus for correct direct answers to improve efficiency. First-token advantage regularization mitigates mode collapse.

With only 6.7K curated samples, EVAThink consistently improves over the base model across six video understanding benchmarks and outperforms a range of strong baselines overall. Its adaptive thinking policy outperforms direct-only and think-only mode by 2.3 and 3.6 percentage points, respectively, while reducing average output length by approximately 62% compared with think-only inference. Our main contributions are as follows:

• We propose Recoverable Evidence Gap Mining to construct evidence-contrastive pairs comprising evidencedense and evidence-sparse views of the same questionanswer pair. These pairs link a decline in direct-answer accuracy on the sparse view to a gain from explicit reasoning, providing targeted supervision for evidenceguided adaptive thinking.

• We develop Evidence-Guided Two-Stage Reinforcement Learning with stage-specific rewards that first guide training with evidence-based mode preferences and then refine response efficiency. The framework incorporates firsttoken advantage regularization to mitigate mode collapse.

• Using only 6.7K samples for training, EVAThink consistently improves over the base model across video understanding benchmarks and outperforms strong baselines overall under controlled visual budgets. Analysis shows that adaptive thinking achieves higher accuracy than both direct-only and think-only modes while substantially reducing output length relative to think-only mode.

## 2. Related Work

## 2.1. Video Understanding

Vision-language models (VLMs) have advanced video understanding through instruction tuning and temporal modeling [1, 2, 30, 40, 48, 50, 52]. Building on these capabilities, recent studies explore explicit reasoning and reinforcement learning (RL) for video tasks. Video-R1 [8] introduces temporal-aware rewards, while VideoRFT [34] combines reasoning supervision with RL. Temporal-RLT [20] and Video-RTS [35] investigate data-efficient RL, while LongVILA-R1 [4] extends RL to long videos. VITAL [46], LOVE-R1 [11], and VideoChat-R1.5 [43] incorporate tool use or iterative perception into reasoning. Complementary approaches enhance video understanding through dedicated frame-selection modules [13, 14], explicit evidencegrounding mechanisms [16], or agentic pipelines to guide further observation and retrieval [36, 42]. These designs incur additional inference-time overhead for selection, grounding, or retrieval, while leaving native adaptive thinking less explored. EVAThink uses RL with paired evidence conditions to train the VLM itself to choose when to reason, without external tools or additional retrieval at inference.

## 2.2. Adaptive Thinking

Adaptive thinking adjusts reasoning computation to balance answer quality and efficiency. For language models, existing approaches include response-mode selection [7, 47], multi-stage reinforcement learning [32], accuracy-aware length rewards [49], and confidence-guided reflection [33]. For video understanding, VideoAuto-R1 [25] uses initialanswer confidence to determine whether further reasoning is needed. AdaThinkV [31] estimates the utility of explicit reasoning by balancing accuracy gains against generation cost, while Video-FLAIR [19] compares reasoning modes based on correctness, grounding, and cost. While these methods adapt reasoning to individual inputs, explicit supervision across paired evidence conditions for the same question remains less explored. EVAThink constructs paired video views that share the same question and answer but differ in the availability of question-relevant evidence under a fixed frame budget. It uses differences in directanswer accuracy and reasoning gains across these views to guide adaptive-thinking training.

![](images/25adb15c9f1845624a4d526213797f4a0b0f24d2d065bb6a0fa9cf94646774f7.jpg)  
Figure 2. Overview of EVAThink. Recoverable Evidence Gap Mining constructs the evidence-contrastive pairs through video cropping or concatenation, retaining pairs with a direct-answer accuracy gap and a reasoning gain on the sparse view. Evidence-Guided Two-Stage Reinforcement Learning first trains response-mode selection using preferences derived from evidence-contrastive pairs, then refines response efficiency on the remaining pairs, using stage-specific rewards and first-token advantage regularization.

## 3. EVAThink

We propose EVAThink, which enhances video understanding through evidence sufficiency-guided training for adaptive thinking. As illustrated in Fig. 2, EVAThink comprises data curation and policy training, where Recoverable Evidence Gap Mining constructs evidence-contrastive pairs for Evidence-Guided Two-Stage Reinforcement Learning. Stage 1 trains response-mode selection using preferences derived from evidence-contrastive pairs, while Stage 2 refines response efficiency. Both stages use stage-specific rewards and first-token advantage regularization.

![](images/783b90aea7e15f4c4826bc63122be4dcda6b0ddf5f3dd1139d73bbf2a57e4731.jpg)  
Figure 3. Data pool.

Data Collection. We first construct a candidate pool of 118.2K multiple-choice video-question samples from eight existing datasets, covering diverse video understanding tasks. As shown in Fig. 3, we organize the pool into three groups based on their primary focus: general video understanding (55.3K, 46.8%), including LLaVA-Video-178K [50], Molmo2-VideoCapQA [6], and VideoMarathon [24]; reasoning-oriented video understanding (49.1K, 41.5%), including LongVILA [3], SR-91K [27], and Holmes-train [5]; and perception-oriented video understanding (13.8K, 11.7%), including NeXT-QA [41] and PerceptionTest [28]. This pool provides the candidate samples for subsequent Recoverable Evidence Gap Mining. See Appendix A for more details.

## 3.1. Recoverable Evidence Gap Mining

Here, we construct evidence-contrastive pairs comprising an evidence-dense view and an evidence-sparse view that share the same question and answer. The two views differ in the fraction of video duration occupied by the evidence segment relevant to the question: this fraction is larger in the dense view and smaller in the sparse view. Under a fixed budget of uniformly sampled frames, the dense view is therefore expected to provide greater coverage of questionrelevant evidence. We retain pairs for which direct-answer accuracy decreases from the dense to the sparse view, while explicit reasoning improves accuracy on the sparse view. These pairs provide supervision for adaptive-thinking training across different evidence conditions.

Starting from the candidate data pool, we construct such pairs in three steps (see Appendix B for more details).

1) Candidate pair construction. Throughout data curation, the target model receives 32 frames uniformly sampled from each video. For each evaluated view and response mode, we estimate answer accuracy using eight rollouts.

We first estimate direct-answer accuracy, $A _ { \mathrm { d i r } }$ , using eight rollouts per sample. We discard samples answered correctly in all eight rollouts, as they are likely too easy to provide useful signals for subsequent reinforcement learning (RL). For samples with $A _ { \mathrm { d i r } } < 0 . 5$ , we use the original video as the sparse view and construct the dense view by cropping around an evidence segment annotated by a stronger model given the video, question, and answer. We extend the crop to include 2–3 sampled frames before and after the annotated segment, using their timestamps in the original video’s uniformly sampled 32-frame sequence. For the remaining samples, with $0 . 5 \le A _ { \mathrm { d i r } } < 1$ , we use the original video as the dense view and construct the sparse view by concatenating another video from the same data source, placing it randomly before or after the original video. For questions involving temporal intervals, we restrict cropping and concatenation to the portion after the annotated segment to preserve the original timestamps.

2) Evidence-gap filtering. For each candidate pair, we evaluate the newly constructed view using eight directanswer rollouts and retain pairs satisfying

$$
\Delta _ { \mathrm { e v i } } = A _ { \mathrm { d i r } } ^ { \mathrm { d e n s e } } - A _ { \mathrm { d i r } } ^ { \mathrm { s p a r s e } } > 0 . 1 2 5 ,\tag{1}
$$

where $A _ { \mathrm { d i r } } ^ { \mathrm { d e n s e } }$ and $A _ { \mathrm { d i r } } ^ { \mathrm { s p a r s e } }$ denote the direct-answer accuracies on the dense and sparse views, respectively. We use this accuracy gap as an empirical proxy for the difference in how reliably the two views support direct answering.

3) Recoverability filtering. For each pair retained after evidence-gap filtering, we prompt the model to use explicit reasoning on the sparse view and retain pairs satisfying

$$
\Delta _ { \mathrm { t h i n k } } = A _ { \mathrm { t h i n k } } ^ { \mathrm { s p a r s e } } - A _ { \mathrm { d i r } } ^ { \mathrm { s p a r s e } } > 0 . 1 2 5 ,\tag{2}
$$

where $A _ { \mathrm { t h i n k } } ^ { \mathrm { s p a r s e } }$ denotes the reasoning accuracy on the sparse view. These pairs exhibit a recoverable evidence gap: direct answering is less accurate on the sparse view than on the dense view, but explicit reasoning improves accuracy on the sparse view using the same sampled frames. Here, recoverability refers to improved answer accuracy, not the recovery of unobserved visual information. The resulting evidence-contrastive pairs are used for subsequent training.

## 3.2. Evidence-Guided Two-Stage Reinforcement Learning

To explicitly anchor the two response modes, we specify distinct response formats for two response modes during training. Specifically, we append a training prompt to each question that instructs the model to answer directly when the visual evidence clearly supports an answer, and to use explicit reasoning when further analysis is needed. The prompt requires each response to follow one of two formats, with distinct openings identifying the selected mode:

$$
\begin{array} { r l } & { \mathrm { D i r e c t ~ a n s w e r : } < \mathsf { a n s w e r } > [ \mathsf { A n s w e r } ] < / \mathsf { a n s w e r } > } \\ & { \mathrm { T h i n k : } \quad \qquad \mathsf { A l } \mathtt { r i g h t } , [ \mathsf { R e a s o n i n g } ] } \\ & { \qquad < \mathsf { a n s w e r } > [ \mathsf { A n s w e r } ] < / \mathsf { a n s w e r } > } \end{array}
$$

This prompt elicits both response modes during initial rollouts, enabling RL training without an SFT cold start, avoiding the need to curate supervised response demonstrations. The distinct openings also support first-token advantage regularization, as described below. The full training prompt is provided in Table 11 in the Appendix B.4.

Building on the Group Relative Policy Optimization (GRPO) [12, 29], we develop a two-stage RL framework that first guides training with evidence-based mode preferences and then refines response efficiency. Our training incorporates two key designs: (i) stage-specific rewards that guide mode selection in Stage 1 and refine responseefficiency in Stage 2; (ii) a first-token advantage regularization mechanism to mitigate entropy collapse during training and discourage premature concentration on a single response mode; To maintain balanced exposure to paired evidence conditions, we additionally use paired sampling, including the dense and sparse views of each selected pair in the same training batch. Besides, following DAPO [45], we omit the KL-divergence penalty and adopt token-level mean loss aggregation throughout both training stages.

In both training stages, we uniformly sample 32 frames from each video, consistent with data curation. We next describe the reward design and first-token advantage regularization for each stage.

Stage 1 RL. In this stage, we aim to use high-confidence evidence-contrastive pairs to guide response-mode training. From the evidence-contrastive pairs obtained through Recoverable Evidence Gap Mining, we select high-confidence pairs satisfying $A _ { \mathrm { d i r } } ^ { \mathrm { d e n s e } } \geq 0 . 7 5$ and $\Delta _ { \mathrm { t h i n k } } \geq 0 . 5$ . These pairs motivate training preferences for direct answering on dense views and explicit reasoning on sparse views, based on the evaluated rollout accuracies.

Reward design. Let $x = ( v , q )$ denote a video-question input, z ∈ {dense, sparse} its view type, and $o _ { i }$ a sampled response with mode $m _ { i } \in \{ \mathrm { d i r e c t , t h i n k } \}$ . The preferred response mode is defined as

$$
m ^ { \star } ( z ) = \left\{ { \begin{array} { l l } { \mathrm { d i r e c t } , } & { z = \mathrm { d e n s e } , } \\ { \mathrm { t h i n k } , } & { z = \mathrm { s p a r s e } . } \end{array} } \right.\tag{3}
$$

The answer reward jointly encourages answer correctness and alignment with the preferred mode:

$$
R _ { i } ^ { \mathrm { a n s } } = \left\{ \begin{array} { l l } { + 2 , } & { \mathrm { c o r r e c t ~ a n s w e r } , m _ { i } = m ^ { \star } ( z ) , } \\ { + 1 , } & { \mathrm { c o r r e c t ~ a n s w e r } , m _ { i } \neq m ^ { \star } ( z ) , } \\ { 0 , } & { \mathrm { i n c o r r e c t ~ a n s w e r } , m _ { i } = m ^ { \star } ( z ) , } \\ { - 1 , } & { \mathrm { i n c o r r e c t ~ a n s w e r } , m _ { i } \neq m ^ { \star } ( z ) . } \end{array} \right.\tag{4}
$$

The format reward penalizes violations of the response formats specified above:

$$
R _ { i } ^ { \mathrm { f m t } } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { v a l i d ~ r e s p o n s e ~ f o r m a t } } , } \\ { - 1 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

The total Stage 1 reward is

$$
R _ { i } ^ { ( 1 ) } = R _ { i } ^ { \mathrm { a n s } } + R _ { i } ^ { \mathrm { f m t } } .\tag{6}
$$

First-token advantage regularization. For responses following the prescribed formats, the distinct first tokens (< and Alright for Qwen3-VL) identify the response mode. We regularize the first-token advantage to mitigate entropy collapse and premature concentration on a single mode. Let $A _ { i }$ denote the GRPO group-relative advantage of response $o _ { i } ,$ and let $p _ { i } = \pi _ { \mathrm { o l d } } ( o _ { i , 1 } \mid x )$ be the probability of its first generated token. The regularized token-level advantage is

$$
\widetilde { A } _ { i , t } = A _ { i } - \mathbb { I } [ t = 1 ] \beta \operatorname* { m a x } ( \log p _ { i } - \log c _ { i } , 0 ) ,\tag{7}
$$

where $c _ { i }$ is a probability threshold and $\beta$ controls the regularization strength. This adjustment reduces the first-token advantage when $p _ { i }$ exceeds $c _ { i } .$ , leaving all subsequent token advantages unchanged. In Stage 1, we set $c _ { i } = 0 . 8 5$ when $m _ { i } = m ^ { \star } ( z )$ and 0.50 otherwise, applying stronger regularization to confident selections of the non-preferred mode. We use $\beta = 5$ in both stages.

For each input $x ,$ , we sample G responses $\{ o _ { i } \} _ { i = 1 } ^ { G }$ from $\pi _ { \mathrm { o l d } }$ and optimize the clipped surrogate objective with token-level mean aggregation and no KL penalty:

$$
\begin{array} { r l } & { \mathcal { I } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \mathrm { o l d } } ( \cdot | \boldsymbol { x } ) } \Bigg [ \frac { 1 } { \sum _ { i = 1 } ^ { G } | o _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } } \\ & { \quad \quad \quad \operatorname* { m i n } \Big ( \rho _ { i , t } ( \theta ) \widetilde { A } _ { i , t } , \mathrm { c l i p } \left( \rho _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \right) \widetilde { A } _ { i , t } \Big ) \Bigg ] , } \end{array}\tag{8}
$$

where D is the training set for the current stage, ε is the clipping parameter, and $\begin{array} { r } { \rho _ { i , t } ( { \boldsymbol { \theta } } ) = \frac { \pi _ { \boldsymbol { \theta } } \left( o _ { i , t } | \boldsymbol { x } , o _ { i , < t } \right) } { \pi _ { \mathrm { o l d } } \left( o _ { i , t } | \boldsymbol { x } , o _ { i , < t } \right) } } \end{array}$

Stage 2 RL. This stage aims to refine response efficiency while maintaining answer accuracy. Starting from the Stage 1 policy, we continue training on the remaining evidence-contrastive pairs after excluding the high confidence pairs used in Stage 1.

Reward design. As Stage 1 updates the policy, the viewbased mode preferences inferred during data curation may become less reliable for the remaining, lower-confidence pairs. We therefore remove the view-dependent mode preference used in Stage 1 and introduce a small bonus for correct direct answers:

$$
R _ { i } ^ { \mathrm { a n s } } = \left\{ \begin{array} { l l } { + 1 . 1 , } & { \mathrm { c o r r e c t a n s w e r } , m _ { i } = \mathrm { d i r e c t } , } \\ { + 1 , } & { \mathrm { c o r r e c t a n s w e r } , m _ { i } = \mathrm { t h i n k } , } \\ { 0 , } & { \mathrm { i n c o r r e c t a n s w e r } . } \end{array} \right.\tag{9}
$$

Building on evidence sufficiency-guided data curation and Stage 1 training, this reward further refines response-mode selection through answer correctness, with a small bonus

favoring correct direct answers. Using the same format reward as in Stage 1, the total Stage 2 reward is

$$
R _ { i } ^ { ( 2 ) } = R _ { i } ^ { \mathrm { a n s } } + R _ { i } ^ { \mathrm { f m t } } .\tag{10}
$$

First-token advantage regularization. We retain the regularization in Eq. (7), using a shared threshold $c _ { i } = 0 . 8 5$ for both response modes and keeping $\beta = 5$ . This removes the mode-dependent thresholds used in Stage 1 while continuing to discourage premature concentration on either mode. We optimize the policy using the same objective in Eq. (8).

## 4. Experiments

## 4.1. Experiment Setup

Training Setup. After the Recoverable Evidence Gap Mining, we obtain 1.76K high-confidence contrastive pairs (i.e., 3.5K samples) and 4.17K remaining contrastive pairs (i.e., 8.3K samples). EVAThink is initialized from Qwen3-VL-8B-Instruct and trained on 32 Ascend 910B NPUs with a batch size of 32. Stage 1 uses the high-confidence pairs for 110 steps, corresponding to approximately one epoch. Stage 2 continues training on the remaining pairs for 100 steps, using 3.2K samples. In total, the two stages use 6.7K training samples (See Appendix C for more details).

Baselines. We compare EVAThink with the Qwen2.5- VL-7B-Instruct [2] and Qwen3-VL-8B-Instruct [1], as well as representative video reasoning methods: Temporal-RLT [20], Video-RTS [35], VITAL [46], LongVILA-R1 [4], LOVR-R1 [11], VideoChat-R1.5 [43], Video-R1 [8], VideoRFT [34], VideoAuto-R1 [25], AdaThinkV [31], and Video-FLAIR [19]. We distinguish results reported under the respective original evaluation protocols from those obtained under a controlled visual budget. For the latter, we evaluate available baseline checkpoints using uniformly sampled 32 or 64 frames, with a fixed per-frame pixel budget of $2 5 6 \times 2 8 \times 2 8$ . VideoAuto-R1 supports adaptive thinking and serves as a strong baseline evaluated under these same visual budgets. For adaptive-thinking methods Ada-ThinkV and Video-FLAIR, we report published results under their original evaluation protocols because their checkpoints are unavailable. Notably, AdaThinkV allows up to 256 frames, compared with our fixed budgets of 32 and 64. In addition, we compare different response modes and ablate individual components to evaluate adaptive thinking and component contributions. During evaluation, we remove EVAThink’s training-specific prompt.

Benchmarks. We evaluate video understanding on six benchmarks: Video-MME [9], MVBench [21], LongVideoBench [39], MMVU (multiple-choice split) [51], Video-MMMU [15], and VSI-Bench [44]. These benchmarks cover general and long-video understanding, as well as video perception and reasoning. We report accuracy on each benchmark and the unweighted average across all six, computing the average only when all results are available. To assess zero-shot transfer to image understanding, we further evaluate on MMBench (MMBench-EN and MMBench-CN) [26], MME (MME-P and MME-C) [10], AI2D [18], and POPE (accuracy) [22] without additional image-task training.

<table><tr><td>Model</td><td>Train. Size</td><td>Video-MME</td><td>MVBench</td><td>LongVideoBench</td><td>MMVU(mc)</td><td>Video-MMMU</td><td>VSI-Bench</td><td>Avg.</td></tr><tr><td colspan="9">Results reported under the respective evaluation protocols (different visual budgets and evaluation settings)</td></tr><tr><td>Temporal-RLT</td><td>32K</td><td>57.6</td><td>68.1</td><td></td><td>65.0</td><td></td><td></td><td></td></tr><tr><td>Video-RTS</td><td>6K</td><td>63.0</td><td>-</td><td>56.6</td><td>66.4</td><td>52.7</td><td></td><td></td></tr><tr><td>VITAL</td><td>182K</td><td>64.1</td><td>一</td><td></td><td>68.7</td><td>54.2</td><td>41.8</td><td></td></tr><tr><td>LongVILA-R1</td><td>206K</td><td>65.1</td><td>67.6</td><td>58.0</td><td></td><td></td><td></td><td></td></tr><tr><td>LOVE-R1</td><td>192.5K</td><td>66.2</td><td>66.6</td><td>60.1</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoChat-R1.5</td><td>80K</td><td>65.2</td><td>70.6</td><td>61.4</td><td></td><td>49.6</td><td></td><td></td></tr><tr><td>AdaThinkV†</td><td>40.7K</td><td>72.1</td><td>72.5</td><td>68.1</td><td>71.4</td><td>65.3</td><td></td><td></td></tr><tr><td>Video-FLAIR</td><td>140K</td><td>s</td><td>-</td><td></td><td></td><td>60.5</td><td>58.2</td><td></td></tr><tr><td colspan="9">32 frames, fixed 256×28×28 pixels per frame</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td></td><td>55.4</td><td>60.5</td><td>58.2</td><td>63.5</td><td>49.4</td><td>33.3</td><td>53.4</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td></td><td>62.6</td><td>65.3</td><td>58.4</td><td>71.4</td><td>63.8</td><td>55.6</td><td>62.9</td></tr><tr><td>Video-R1</td><td>169K</td><td>59.3</td><td>63.9</td><td></td><td>63.8</td><td>52.3</td><td>35.8</td><td></td></tr><tr><td>VideoRFT</td><td>110K</td><td>59.8</td><td>62.1</td><td></td><td>68.5</td><td>51.1</td><td>36.8</td><td></td></tr><tr><td>VideoAuto-R1-Qwen2.5</td><td>83K</td><td>62.1</td><td>70.0</td><td>52.1</td><td>66.4</td><td>55.3</td><td>36.6</td><td>57.1</td></tr><tr><td>VideoAuto-R1-Qwen3</td><td>83K</td><td>63.4</td><td>68.5</td><td>53.0</td><td>69.9</td><td>63.6</td><td>52.1</td><td>61.8</td></tr><tr><td>EVAThink</td><td>6.7K</td><td>62.8</td><td>65.6</td><td>58.5</td><td>71.7</td><td>64.9</td><td>57.2</td><td>63.5</td></tr><tr><td>(think ratio)</td><td></td><td>(44.8%)</td><td>(7.2%)</td><td>(56.8%)</td><td>(76.6%)</td><td>(91.9%)</td><td>(0.0%)</td><td></td></tr><tr><td colspan="9">64 frames, fixed 256×28×28 pixels per frame</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td></td><td>62.6</td><td>67.5</td><td>60.1</td><td>64.5</td><td>55.1</td><td>38.6</td><td>58.1</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td></td><td>64.9</td><td>65.8</td><td>60.4</td><td>70.4</td><td>65.6</td><td>58.3</td><td>64.2</td></tr><tr><td>VideoAuto-R1-Qwen2.5</td><td>83K</td><td>64.1</td><td>69.8</td><td>53.3</td><td>67.0</td><td>55.1</td><td>39.2</td><td>58.1</td></tr><tr><td>VideoAuto-R1-Qwen3</td><td>83K</td><td>66.0</td><td>69.0</td><td>54.6</td><td>69.6</td><td>64.3</td><td>54.6</td><td>63.0</td></tr><tr><td>EVAThink</td><td>6.7K</td><td>66.3</td><td>66.5</td><td>60.7</td><td>72.8</td><td>67.4</td><td>59.9</td><td>65.6</td></tr><tr><td>(think ratio)</td><td></td><td>(34.4%)</td><td>(5.1%)</td><td>(46.1%)</td><td>(59.4%)</td><td>(85.7%)</td><td>(0.0%)</td><td></td></tr></table>

Table 1. Performance across video understanding benchmarks. Bold and underlined values indicate the best and second-best scores within each fixed-budget group, respectively. Avg. is the unweighted mean over all six benchmarks and is reported only for complete results. <sup>†</sup>AdaThinkV allows up to 256 input frames.

<table><tr><td>Model</td><td>Think Mode</td><td>Avg.</td><td>Tok.</td></tr><tr><td>EVAThink</td><td>Direct</td><td>63.3</td><td>8</td></tr><tr><td>EVAThink</td><td>Think Only</td><td>62.0</td><td>560</td></tr><tr><td>EVAThink</td><td>Auto Think</td><td>65.6</td><td>213</td></tr></table>

Table 2. Effect of adaptive thinking under the 64-frame setting. Tok. denotes the average number of generated tokens per response.
<table><tr><td rowspan="2">Model</td><td colspan="2">Think Ratio</td><td rowspan="2">Overall Acc. (%)</td></tr><tr><td>Dense</td><td>Sparse</td></tr><tr><td>EVAThink</td><td>22%</td><td>83%</td><td>65.3</td></tr></table>

Table 3. Evaluation on held-out dense/sparse pairs sharing the same question and answer, evaluated under the 64-frame setting.

## 4.2. Main Results

Table 1 shows performance on six video understanding benchmarks, including results under the original evaluation protocols and controlled comparisons with fixed budgets of 32 and 64 frames. Under both fixed-budget settings, EVAThink achieves the highest average accuracy among the evaluated baselines with results on all six benchmarks. With 32 frames, EVAThink reaches 63.5%, outperforming Qwen3-VL-8B-Instruct and VideoAuto-R1-Qwen3 by 0.6 and 1.7 percentage points, respectively. With 64 frames, its average accuracy increases to 65.6%, extending these margins to 1.4 and 2.6 percentage points. These improvements are achieved using only 6.7K training samples, compared with 83K for VideoAuto-R1, demonstrating effective training with a small trainset. Beyond the average gains, EVA-Think consistently improves upon base model Qwen3-VL-8B-Instruct on every benchmark under both frame budgets, demonstrating benefits across video understanding tasks.

Increasing the frame budget from 32 to 64 also reveals a consistent shift in response-mode selection. Increasing the frame budget from 32 to 64 improves EVAThink’s accuracy on all six benchmarks while reducing its thinking ratio on all five benchmarks where explicit reasoning is used. This pattern is consistent with the motivation for evidence sufficiency-guided training: a larger frame budget may provide more question-relevant evidence and make direct answering more effective. Meanwhile, the thinking ratio remains highest on Video-MMMU, consistent with its demands for concept understanding and knowledge application. Conversely, the lowest thinking ratio on VSI-Bench suggests a preference for direct spatial judgments, aligning with prior findings [17, 19, 44] that explicit verbal reasoning can hinder visual-spatial performance.

<table><tr><td>Model</td><td>MMBench-En</td><td>MMBench-Cn</td><td>MME-P</td><td>MME-C</td><td>AI2D</td><td>POPE</td><td>Rel.</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>83.7</td><td>82.2</td><td>1671.9</td><td>546.4</td><td>81.9</td><td>88.2</td><td>100%</td></tr><tr><td>EVAThink</td><td>85.1</td><td>82.9</td><td>1723.1</td><td>562.5</td><td>82.5</td><td>88.8</td><td>101.7%</td></tr></table>

Table 4. More results across image understanding tasks. “Rel.” denotes the average relative performance across benchmarks, computed as the mean percentage normalized by the base model (Qwen3-VL-8B-Instruct) set to 100%.

## 4.3. Extended Results

Effect of adaptive thinking. We compare direct-answer, think-only, and default adaptive inference using the same EVAThink checkpoint under the 64-frame setting. As shown in Table 2, adaptive thinking achieves 65.6% average accuracy, exceeding the two fixed modes by 2.3 and 3.6 percentage points, respectively, while reducing output length by 62% relative to think-only inference.

We further evaluate response-mode selection on held-out dense/sparse pairs with matched questions and answers under the same 64-frame setting. We randomly sample 200 crop-constructed pairs from the evidence-contrastive pairs, excluding any pair whose source video was used in training. Each pair shares the same question and answer, with the original video as the sparse view and an evidence-centered crop as the dense view, evaluated under the same frame budget and resolution. EVAThink’s thinking ratio increases from 22% on dense views to 83% on sparse views, while its overall accuracy reaches 65.3%. These paired results suggest that EVAThink’s mode selection is sensitive to evidence conditions, consistent with its training motivation.

More results across image understanding tasks. To assess generalization beyond video understanding, we evaluate EVAThink on four image benchmarks without addi tional image-task training. As shown in Table 4, beyond its consistent gains on video benchmarks, EVAThink also outperforms Qwen3-VL-8B-Instruct on every reported imageunderstanding metric, achieving an average relative performance of 101.7%. These results suggest that the benefits of EVAThink also transfer to image understanding.

## 4.4. Ablation Study

We ablate individual components of EVAThink under the 64-frame setting. Table 5 reports the resulting performance. Effect of data curation. We evaluate the effect of data curation by removing recoverability filtering alone and then removing both evidence-gap and recoverability filtering. To accommodate the changes in the available training pairs, we adjust stage-wise sample selection accordingly: removing the $\Delta _ { \mathrm { t h i n k } } \geq 0 . 5$ requirement for Stage 1 in the first variant, and randomly selecting samples for both stages from the candidate pairs in the second. As shown in Table 5, removing recoverability filtering reduces average accuracy from 65.6% to 65.0% while increasing average output length from 213 to 247 tokens, suggesting that identifying reasoning gains helps provide more effective supervision for adaptive thinking. Removing both filters further reduces average accuracy to 63.5%. Although this variant produces slightly shorter responses (198 tokens), its lower accuracy indicates a less favorable accuracy-efficiency trade-off. Together, these results support selecting evidence-contrastive pairs that exhibit both a direct-answer accuracy gap and a reasoning gain on the sparse view.

![](images/d45f3cccdfeca1a4b7d42b436919f8d58f3245a128164b2c73b109ffc40ba9ec.jpg)

![](images/2c42ff39dec9621e4b5c90dbe2da393795d47bdc13e282000efbd2f94463ee9f.jpg)  
Figure 4. Effect of first-token advantage regularization on policy entropy during Stage 1 (left) and Stage 2 (right). Without regularization, entropy rapidly collapses in both stages, accompanied by a shift to direct answering for all rollouts (response-mode collapse).

Effect of reward design. We separately remove the Stage 1 mode-preference reward and the Stage 2 directanswer bonus, using correctness-only answer rewards of +2/0 in Stage 1 and +1/0 in Stage 2 for correct/incorrect answers, respectively, while keeping all other objective components and training settings unchanged. As shown in Table 5, removing the Stage 1 mode-preference reward reduces average accuracy from 65.6% to 61.2% and increases average response length from 213 to 392 tokens, supporting its contribution to accurate and efficient adaptive answering. Removing the Stage 2 direct-answer bonus yields comparable accuracy (65.3% vs. 65.6%) but substantially longer responses (465 vs. 213 tokens), indicating that the bonus improves response efficiency with little change in accuracy.

Effect of first-token advantage regularization. Figure 4 illustrates the effect of first-token advantage regularization on training dynamics. Without first-token advantage regularization, average accuracy decreases to 64.6%, with the largest decline on Video-MMMU. Its effect is also evident in the training dynamics: without regularization, policy entropy rapidly approaches zero in both stages, accompanied by mode collapse in which all rollouts shift to direct answering. In contrast, the regularized policy maintains nonzero entropy throughout training. Together, these observations suggest that the regularizer helps prevent premature policy concentration and preserves exploration, contributing to better final performance.

<table><tr><td>Method</td><td>Video-MME</td><td>MVBench</td><td>LongVideoBench</td><td>MMVU(mc)</td><td>Video-MMMU</td><td>VSI-Bench</td><td>Avg.</td><td>Tok.</td></tr><tr><td>w/o Recoverability filtering</td><td>65.9</td><td>65.6</td><td>60.2</td><td>72.3</td><td>67.2</td><td>58.5</td><td>65.0</td><td>247</td></tr><tr><td>w/o Evidence-gap &amp; Recoverability filtering</td><td>64.8</td><td>65.1</td><td>59.5</td><td>69.8</td><td>63.3</td><td>58.2</td><td>63.5</td><td>198</td></tr><tr><td>w/o Stage 1 mode preference reward</td><td>63.2</td><td>62.7</td><td>56.4</td><td>68.2</td><td>59.7</td><td>56.7</td><td>61.2</td><td>392</td></tr><tr><td>w/o Stage 2 direct-answer bonus reward</td><td>66.0</td><td>65.9</td><td>60.5</td><td>72.6</td><td>68.1</td><td>58.6</td><td>65.3</td><td>465</td></tr><tr><td>w/o First-token advantage regularization</td><td>66.1</td><td>65.2</td><td>60.2</td><td>72.2</td><td>63.8</td><td>60.1</td><td>64.6</td><td>52</td></tr><tr><td>w/o Stage1 training</td><td>66.2</td><td>66.8</td><td>60.0</td><td>71.5</td><td>63.2</td><td>58.9</td><td>64.4</td><td>80</td></tr><tr><td>w/o Stage2 training</td><td>65.4</td><td>65.2</td><td>61.3</td><td>73.4</td><td>66.7</td><td>60.3</td><td>65.4</td><td>331</td></tr><tr><td>EVAThink</td><td>66.3</td><td>66.5</td><td>60.7</td><td>72.8</td><td>67.4</td><td>59.9</td><td>65.6</td><td>213</td></tr></table>

Table 5. Ablation study under the 64-frame setting. RL optimization settings are held fixed; data-curation changes are described in the text.

![](images/9b5b84b046759c1ae486428be8e253a1ffd3ca12d1b0e4a328b9dcb063059d17.jpg)  
Figure 5. Case studies of adaptive thinking: EVAThink avoids unnecessary reasoning (left), prevents reasoning-induced errors through direct answering (middle), and resolves errors through evidence-grounded reasoning (right).

Effect of Stage 1 training. Omitting Stage 1 reduces average accuracy to 64.4%, with particularly notable drops on Video-MMMU and MMVU of 4.2 and 1.3 points, respectively. The average output length also decreases to 80 tokens. The combination of shorter responses and lower accuracy suggests that the model may invoke reasoning less effectively without Stage 1. These results support the role of Stage 1 in establishing an adaptive response-mode policy before Stage 2 refines response efficiency.

Effect of Stage 2 training. Omitting Stage 2 yields an average accuracy of 65.4%, slightly below the full model, while increasing the average output length from 213 to 331 tokens. Although Stage 2 does not improve every benchmark, it raises overall accuracy and reduces output length by approximately 36% relative to the Stage 1-only model. These results suggest that Stage 2 refines the learned policy, improving the overall balance between answer accuracy and reasoning length.

## 4.5. Case Study

Figure 5 presents examples of EVAThink’s adaptive thinking. On the left, EVAThink answers correctly without explicit reasoning, using 6 tokens compared with 276 in thinkonly mode. In the middle, it correctly identifies the shoe color through direct answering, whereas think-only mode answers incorrectly. On the right, it uses visual time cues to correctly infer a duration that direct-answer mode gets wrong. These cases illustrate how adaptive thinking can reduce unnecessary computation, avoid reasoning-induced errors, and improve answers through evidence integration.

## 5. Conclusion

We presented EVAThink, which uses evidence sufficiency to guide adaptive-thinking training for video understanding. Recoverable Evidence Gap Mining constructs evidencecontrastive pairs to supervise mode selection, while twostage reinforcement learning and first-token advantage regularization support effective and efficient adaptive thinking. Using only 6.7K training samples, EVAThink improves video understanding under fixed visual budgets. Compared with always reasoning, it achieves higher average accuracy with substantially shorter responses, supporting evidence sufficiency-guided training for adaptive thinking.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2, 5

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 2, 5

[3] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In International Conference on Learning Representations, pages 18227–18246, 2025. 3

[4] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. Advances in Neural Information Processing Systems, 38: 172842–172870, 2026. 2, 5

[5] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? In European Conference on Computer Vision, pages 539–556. Springer, 2026. 3

[6] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28652–28668, 2026. 3

[7] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: Llm learns when to think. Advances in neural information processing systems, 38:151268–151295, 2026. 2

[8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. Advances in Neural Information Processing Systems, 38:99114–99137, 2026. 2, 5

[9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108– 24118. IEEE, 2025. 5

[10] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026. 6

[11] Shenghao Fu, Qize Yang, Yuan-Ming Li, Xihan Wei, Xiaohua Xie, and Wei-Shi Zheng. Love-r1: Advancing long

video understanding with an adaptive zoom-in mechanism via multi-step reasoning. arXiv preprint arXiv:2509.24786, 2025. 2, 5

[12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 4

[13] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26181–26191. IEEE, 2025. 2

[14] Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, et al. M-llm based video frame selection for efficient video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13702–13712. IEEE, 2025. 2

[15] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Xiang Yue, Bo Li, Yuanhan Zhang, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multidisciplinary professional videos. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 27798–27828, 2026. 5

[16] Yanxiang Huang, Guohua Gao, and Zhaoyang Wei. Video evidence to reasoning efficient video understanding via explicit evidence grounding. arXiv preprint arXiv:2601.07761, 2026. 2

[17] Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N Balasubramanian, and Tanuja Ganu. Chain-of-thought degrades visual spatial reasoning capabilities of multimodal llms. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 862–876, 2026. 7

[18] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016. 6

[19] Yogesh Kulkarni and Pooyan Fazli. Video-flair: Not whether to reason, but how. arXiv preprint arXiv:2608.26495, 2026. 1, 2, 5, 7

[20] Hongyu Li, Songhao Han, Yue Liao, Junfeng Luo, Jialin Gao, Shuicheng Yan, and Si Liu. Reinforcement learning tuning for videollms: Reward design and data efficiency. arXiv preprint arXiv:2506.01908, 2025. 2, 5

[21] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 22195– 22206, 2024. 5

[22] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 conference

on empirical methods in natural language processing, pages 292–305, 2023. 6

[23] Shijie Lian, Bin Yu, Xiaopeng Lin, Changti Wu, Hang Yuan, Xiaolin Hu, Zhaolong Shen, Yuzhuo Miao, Haishan Liu, Yuxuan Tian, et al. Physbrain 1.0 technical report. arXiv preprint arXiv:2605.15298, 2026. 1

[24] Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Xiaodong Yu, Hao Chen, Jiebo Luo, Zicheng Liu, et al. Unleashing hour-scale video training for long video-language understanding. Advances in Neural Information Processing Systems, 38:17523–17552, 2026. 3

[25] Shuming Liu, Mingchen Zhuge, Changsheng Zhao, Jun Chen, Lemeng Wu, Zechun Liu, Chenchen Zhu, Zhipeng Cai, Chong Zhou, Haozhe Liu, et al. Videoauto-r1: Video auto reasoning via thinking once, answering twice. arXiv preprint arXiv:2601.05175, 2026. 1, 2, 5

[26] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 6

[27] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 3

[28] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 3

[29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 4

[30] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systemsfor Video Technology, 36(2):1355–1376, 2025. 1, 2

[31] Jingqi Tian, Haoji Zhang, Lin Chen, Hongbo Jin, Haonan Xu, Tianrui Zhu, Xingming Shui, Shilin Ma, Wenjing Yang, and Yansong Tang. Adathinkv: Adaptive thinking for tokenefficient video reasoning. arXiv preprint arXiv:2608.01980, 2026. 1, 2, 5

[32] Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage rl. Advances in Neural Information Processing Systems, 38:15181–15207, 2026. 2

[33] Xu Wan, Wei Wang, Wenyue Xu, Wotao Yin, Jie Song, and Mingyang Sun. Adapthink: Adaptive thinking preferences for reasoning language model. arXiv preprint arXiv:2506.18237, 2025. 2

[34] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms

via reinforced fine-tuning. Advances in neural information processing systems, 38:4350–4376, 2026. 2, 5

[35] Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, and Mohit Bansal. Video-rts: Rethinking reinforcement learning and test-time scaling for efficient and enhanced video reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 28114–28128, 2025. 2, 5

[36] Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caim ing Xiong, Silvio Savarese, Mohit Bansal, Michael S Ryoo, and Juan Carlos Niebles. Active video perception: Iterative evidence seeking for agentic long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 9088–9099, 2026. 2

[37] Changti Wu, Shijie Lian, Zihao Liu, Lei Zhang, Laurence Tianruo Yang, and Kai Chen. Dynasolidgeo: A dynamic benchmark for genuine spatial mathematical reasoning of vlms in solid geometry. arXiv preprint arXiv:2510.22340, 2025. 1

[38] Changti Wu, Jiahuai Mao, Yuzhuo Miao, Shijie Lian, Bin Yu, Xiaopeng Lin, Cong Huang, Lei Zhang, and Kai Chen. Scalselect: Scalable training-free multimodal data selec tion for efficient visual instruction tuning. arXiv preprint arXiv:2602.11636, 2026. 1

[39] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Informa tion Processing Systems, 37:28828–28857, 2024. 5

[40] Jianlong Wu, Wei Liu, Ye Liu, Meng Liu, Liqiang Nie, Zhouchen Lin, and Chang Wen Chen. A survey on video temporal grounding with multimodal large language model. IEEE Transactions on Pattern Analysis and Machine Intelli gence, 2025. 1, 2

[41] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9772– 9781. IEEE, 2021. 3

[42] Caijun Yan, Yang Zhou, Meixing Shi, Haoran Sun, Yichen Li, Yuxiang Cai, and Yankai Jiang. Reveal: A rubric-guided agent for explicit evidence sufficiency verificationin longvideo question answering. arXiv preprint arXiv:2608.08612, 2026. 2

[43] Ziang Yan, Yinan He, Xinhao Li, Zhengrong Yue, Xiangyu Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. Advances in Neural Information Processing Systems, 38:119152–119184, 2026. 2, 5

[44] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10632–10643. IEEE, 2025. 5, 7

[45] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong

Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026. 4

[46] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal toolaugmented reinforcement learning for long video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32903–32914, 2026. 2, 5

[47] Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3716–3730, 2025. 2

[48] Kecheng Zhang, Zongxin Yang, Mingfei Han, Haihong Hao, Yunzhi Zhuge, Changlin Li, Zhihui Li, Xiaojun Chang, et al. Progressive online video understanding with evidencealigned timing and transparent decisions. In International Conference on Learning Representations, pages 5092–5121, 2026. 2

[49] Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, and Xunliang Cai. When to continue thinking: Adaptive thinking mode switching for efficient reasoning. arXiv preprint arXiv:2505.15400, 2025. 2

[50] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 2, 3

[51] Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, et al. Mmvu: Measuring expert-level multidiscipline video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8475–8489. IEEE, 2025. 5

[52] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, et al. Apollo: An exploration of video understanding in large multimodal models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18891–18901. IEEE, 2025. 1, 2

## A. More Details of Data Pool

We construct the data pool of approximately 118.2K samples from eight video understanding datasets. Table 6 summarizes the data sources, selection criteria, and corresponding sample counts. Specifically, we use the academic subset of LLaVA-Video-178K and randomly sample subsets from Molmo2-VideoCapQA and VideoMarathon. For SR-91K, we retain the multiple-choice subset. We include all samples from LongVILA, Holmes-train, NeXT-QA, and PerceptionTest. These sources form the candidate pool for subsequent contrastive-pair construction and filtering.

<table><tr><td>Data source</td><td>Selection criterion</td><td># Selected samples</td></tr><tr><td>LLaVA-Video-178K</td><td>Academic subset</td><td>28.9K</td></tr><tr><td>Molmo2-VideoCapQA</td><td>Random subset</td><td>21.9K</td></tr><tr><td>VideoMarathon</td><td>Random subset</td><td>4.5K</td></tr><tr><td>LongVILA</td><td>All</td><td>29.1K</td></tr><tr><td>SR-91K</td><td>Multiple-choice subset</td><td>18.4K</td></tr><tr><td>Holmes-train</td><td>All</td><td>1.6K</td></tr><tr><td>NeXT-QA</td><td>All</td><td>7.5K</td></tr><tr><td>PerceptionTest</td><td>All</td><td>6.3K</td></tr></table>

Table 6. Composition of the data pool.

## B. Implementation Details of EVAThink

## B.1. Prompts for Rollouts

Tables 7 and 8 present the prompts used for direct-answer and thinking rollouts during data curation. The former requests a direct answer, while the latter instructs the model to reason over cross-frame clues before answering, without introducing unsupported details. Both prompts require the final answer to be enclosed in <answer></answer> tags, containing only the option letter for multiple-choice questions.

USER PROMPT FOR DIRECTANSWER   
You are given a video. Please answer the question directly.   
Question: [Question]   
[Options]   
Your final answer must be enclosed in <answer></answer>.   
- For a multiple-choice question, put only the option letter inside   
<answer></answer>.   
Examples: <answer>A</answer>  
Table 7. User prompt for direct answer in the data curation.

## B.2. Details of Evidence Segment Annotation

We use a stronger model to annotate evidence segments (the prompt is shown in the Table 9). Given 32 uniformly sampled frames in chronological order, together with the question, candidate options, and ground-truth answer, the annotator identifies the minimal continuous frame interval containing the supporting evidence. When the required clues occur at separate moments, the interval covers all of them. Annotations use inclusive frame indices and include a localization status and confidence level. Ambiguous or absent evidence is explicitly marked as partial or not localized, respectively, and the corresponding samples are discarded.

![](images/ff762aa91174cdeca356408ff89c612bff31716a24840e292a7246e8859d3030.jpg)  
Table 8. User prompt for thinking in the data curation.

Annotation accuracy. The annotations serve as candidate evidence regions for mining, so precise temporal boundaries are not required. Sample acceptance is subsequently determined by two filtering stages based on rollout accuracy, which verify whether each candidate exhibits the behavior required by our mining criteria. These checks reduce sensitivity to localization errors: an annotation alone is insufficient for a sample to be retained. Thus, the retained samples satisfy our operational criteria for recoverable evidence gaps, without requiring frame-exact annotations.

USER PROMPT   
You are a video evidence localizer.   
You will be given a video constructed from exactly 32 frames uniformly sampled from the original video and then arranged in chronological order.   
The frame ids are defined by their order in this 32-frame video:   
- The first sampled frame is Frame 0.   
- The last sampled frame is Frame 31.   
Your task is not to answer the question.   
Your task is to locate the sampled frame range that contains the visual evidence needed to support the given ground-truth answer   
Question:   
[Question]   
Options:   
[Options]   
Ground-truth answer:   
[Answer]   
Rules:   
1. Output the minimal sampled frame range that visually supports the ground-truth answer.   
2. If the required evidence appears in multiple separated moments, output the smallest continuous range covering all required evidence.   
3. Use frame ids (starting from 0), not estimated timestamps.   
4. Do not include irrelevant context unless it is needed to answer the question.   
5. If the evidence is not visible in the provided frames, output status = "not localized"; if the evidence is ambiguous or only partially visible   
output status = "partial".   
6. Both start frame and end frame are included in the selected range.   
Return only valid JSON in the following format:   
{   
"status": "localized | partial | not localized",   
"start frame": int,   
"end frame": int,   
"confidence": "high | medium | low"   
}  
Table 9. Prompt for video evidence segment annotation, which instructs the model to identify the minimal sampled frame range supporting the ground-truth answer.

Annotation cost. Although annotation uses a stronger model, it is applied only to a small subset of the original data pool. After removing samples for which all preceding rollouts are correct, only the remaining samples identified as sparse-view cases require evidence annotation. Empirically, these account for less than 22.7% of the original pool. Each annotation call receives only 32 sampled frames and requests a compact localization output, keeping the overall annotation budget controlled.

## B.3. Details of Resulting Evidence-Contrastive Pairs

Table 10 summarizes the evidence-contrastive pairs retained from each data source for the two training stages. The resulting training sets contain 1,766 pairs for Stage 1 and 4,179 pairs for Stage 2, spanning eight video datasets. Both stages include pairs from all eight sources, with SR-91K contributing the largest share in each stage.

<table><tr><td>Data source</td><td>Stage 1 (pairs)</td><td>Stage 2 (pairs)</td></tr><tr><td>LLaVA-Video-178K</td><td>317</td><td>628</td></tr><tr><td>Molmo2-VideoCapQA</td><td>325</td><td>529</td></tr><tr><td>VideoMarathon</td><td>51</td><td>169</td></tr><tr><td>LongVILA</td><td>329</td><td>720</td></tr><tr><td>SR-91K</td><td>514</td><td>1682</td></tr><tr><td>Holmes-train</td><td>28</td><td>67</td></tr><tr><td>NeXT-QA</td><td>75</td><td>136</td></tr><tr><td>PerceptionTest</td><td>127</td><td>248</td></tr><tr><td>Total</td><td>1766</td><td>4179</td></tr></table>

Table 10. Composition of the evidence-contrastive training pairs for the two stages.

## B.4. Training Prompt

Table 11 presents the training prompt, which instructs the model to select between direct answering and evidencegrounded reasoning based on the available visual evidence. Direct responses begin with <answer>, while reasoning responses begin with Alright and briefly connect observed clues before answering. Both modes place the fina answer within <answer></answer> tags. The distinct openings allow the selected response mode to be identified directly from the output.

![](images/5e5c865fe8acf5fcd22f3fa3b7da42c0f45cc44c12a051bf6d4b93c33469b76d.jpg)

Table 11. Training prompt for response mode selection, which instructs the model to adaptively select between direct and thinking modes.
<table><tr><td>Model</td><td>Think Mode</td><td>Video-MME</td><td>MVBench</td><td>LongVideoBench</td><td>MMVU(mc)</td><td>Video-MMMU</td><td>VSI-Bench</td><td>Avg.</td><td>Tok.</td></tr><tr><td>EVAThink</td><td>Direct</td><td>66.2</td><td>66.4</td><td>59.8</td><td>70.6</td><td>56.8</td><td>59.9</td><td>63.3</td><td>8</td></tr><tr><td>EVAThink</td><td>Think Only</td><td>60.8</td><td>64.2</td><td>57.4</td><td>71.2</td><td>67.1</td><td>51.4</td><td>62.0</td><td>560</td></tr><tr><td>EVAThink</td><td>Auto Think</td><td>66.3</td><td>66.5</td><td>60.7</td><td>72.8</td><td>67.4</td><td>59.9</td><td>65.6</td><td>213</td></tr><tr><td>(think ratio)</td><td></td><td>(34.4%)</td><td>(5.1%)</td><td>(46.1%)</td><td>(59.4%)</td><td>(85.7%)</td><td>(0.0%)</td><td></td><td></td></tr></table>

Table 12. Effect of adaptive thinking under the 64-frame setting. Tok. denotes the average number of generated tokens per response.

## C. Details of Experiments

## C.1. Training Hyperparameters

Table 13 summarizes the training hyperparameters. We use AdamW with a constant learning rate of $1 0 ^ { - 6 }$ and a batch size of 32. For each prompt, we sample eight rollouts at a temperature of 1.0 and perform one policy update per step. Gradient clipping is set to 1.0, and the maximum prompt and response lengths are 20,480 and 1,024 tokens, respectively.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Rollouts per prompt</td><td>8</td></tr><tr><td>Policy updates per step</td><td>1</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1e-6 (constant)</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Rollout temperature</td><td>1.0</td></tr><tr><td>Max prompt length</td><td>20480</td></tr><tr><td>Max response length</td><td>1024</td></tr></table>

Table 13. Training Hyperparameters.

## C.2. Evaluation Settings

All evaluations use greedy decoding with the temperature set to 0. The maximum response length is set to 2048 tokens.

## C.3. More Results of Effect of Adaptive Thinking

Table 12 provides per-benchmark results for direct-answer, think-only, and adaptive inference using the same EVA-Think checkpoint under the 64-frame setting. Adaptive inference achieves 65.6% average accuracy with 213 output tokens per response, compared with 63.3% and 8 tokens for direct answering and 62.0% and 560 tokens for think-only inference. It matches or exceeds both fixed modes on every benchmark.

## D. Case Study

Figures 6 and 7 present examples of EVAThink in thinking mode, while Figures 8 and 9 present examples in direct-answer mode. These cases illustrate the two response modes, showing how the model connects visual clues through explicit reasoning or provides an answer directly.

![](images/393aa1a9ea4263c7b95d9c74428f7404490db5ad3d2f61d1d6ccaf705e075305.jpg)

What is the order of skincare products used before makeup in the video? A. Face serum, toner, eye cream, face cream.   
B. Toner, face serum, eye cream, face cream.   
C. Face serum, toner, face cream, eye cream.   
D. Toner, face serum, face cream, eye cream.

EVAThink: The video shows the following skincare products being used in this order:\n\n1. \*\*Toner\*\* – “PREP SKIN WITH TONIQUE CONFORT.” (0:05)\n2. \*\*Face serum\*\* – “APPLY GÉNIFIQUE SERUM...” (0:13)\n3. \*\*Face cream\*\* – “MOISTURIZE WITH RÉNERGIE LIFT MULTI-ACTION FACE CREAM...” (0:29)\n4. \*\*Eye cream\*\* – “APPLY ADVANCED GÉNIFIQUE EYE CREAM...” (0:37)\n\nThis matches option \*\*D. Toner, face serum, face cream, eye cream.\*\*\n\n Correct Answer: \*\*D. Toner, face serum, face cream, eye cream.\*\*

![](images/7fefd3da1d9c205f0e6f3ff039206b4761a7a9c75ea25f160a64d8f38a466227.jpg)

Which was the duration of the male protagonist's morning work in the video? A. Three hours.   
B. Two hours and forty-five minutes.   
C. Four hours.   
D. Three hours and forty-five minutes.

EVAThink: The video shows the male protagonist starting his work at \*\*9:15\*\* and continuing until \*\*12:00\*\*, as indicated by the on-screen text “9:15 - work, work, work” and “12:00 - время идти на обед” (time to go for lunch). \n\nCalculating the duration:\n- From 9:15 to 12:00 is \*\*2 hours and 45 minutes\*\*.\n\nTherefore, the correct answer is: \n\*\*B. Two hours and forty-five minutes.\*\*

![](images/c58bd1cffad4e132f3200a8dd0ecd8030248e1af3b642b3723e5def093fb5e6d.jpg)

EVAThink: The video presents several luxury foods with their prices:\n\n- \*\*Doughnuts\*\*: \$100 (at 0:49)\n-\*\*Caviar\*\*: \$350 (at 2:20)\n- \*\*Japanese honeydew melon\*\*: \$17,000 (at 6:28)\n- \*\*Taco\*\*: \$25,000 (at 6:45)\n\nAmong these, the \*\*doughnuts\*\* are the cheapest at \*\*\$100\*\*.\n\n### Correct Answer: \*\*D. Doughnuts\*\*

Figure 6. Case studies of EVAThink: think mode (1).

![](images/e7afe95a81371f36a8b96d0ad59fd44fc0ed24f4345bf4f680471beb7fc79f9a.jpg)  
Figure 7. Case studies of EVAThink: think mode (2).

![](images/758a492128b639a3f148a8d022ea40eafd2fad9f78161d11f9c6e7288ff4e42b.jpg)  
Figure 8. Case studies of EVAThink: direct mode (1).

![](images/bdf999fa8e4503606051801375e79d7b24fce83b5ea624d101624dbe9a36dfdc.jpg)  
Figure 9. Case studies of EVAThink: direct mode (2).