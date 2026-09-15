# Learning from Reliable Negatives: Confidence-Anchored Test-Time Adaptation for GUI Grounding

Yizhou Liu<sup>∗1</sup>, Fei Tang<sup>∗1</sup>, Yuchen Yan<sup>1</sup>, Zhengxi Lu<sup>1</sup>, Songqin Nong<sup>2</sup>, Tao Jiang<sup>2</sup>, Wenhao Xu<sup>2</sup>, Wenqi Zhang<sup>1</sup>, Weiming Lu<sup>1</sup>, Jun Xiao<sup>1</sup>, and Yongliang Shen<sup>†1</sup>

<sup>1</sup> Zhejiang University <sup>2</sup> Ant Group {22551294,syl}@zju.edu.cn

Abstract. Graphical User Interface (GUI) grounding is essential for autonomous agents to map natural language instructions to precise screen coordinates. However, existing supervised fine-tuning and reinforcement learning methods are constrained by the high cost of annotation, creating a scalability bottleneck. In this paper, we introduce a label-free test-time training paradigm driven by two key insights: (1) confidence patterns in coordinate tokens are a better indicator than full-sequence confidence, and (2) in sparse GUI coordinate spaces, negative samples ofer more reliable learning signals than potentially noisy positive ones. We first propose Confidence-Anchored Learning (CAL), which utilizes coordinatetoken confidence to filter pseudo-labels and assign distance-based binary rewards. Building on this, we develop Confidence-Anchored Negative Learning (CANL), which exclusively optimizes the model using negative samples to bypass the risks of incorrect positive samples. Experimental results demonstrate that CANL-7B achieves 92.1% on ScreenSpot-V2. On more challenging ScreenSpot-Pro, CANL-7B reaches 33.8%, an 8.9% absolute improvement over the base model. Our findings establish coordinate-token confidence as a powerful alternative to manual annotations for scalable GUI agent development.

Keywords: GUI Grounding · Test-Time Training

## 1 Introduction

GUI agents have emerged as a critical technology for automating human-computer interaction, enabling natural language commands to be translated into precise interface actions. While early rule-based systems required extensive manual engineering and lacked adaptability [42,59], the advent of multimodal large language models (MLLMs) [2–4, 29, 56, 60] has opened new possibilities for flexible GUI understanding and interaction [11,13,17,38,40,48,49,51,55,58]. At the heart of these agents lies GUI grounding, the task of mapping natural language instructions to specific interface elements through coordinate prediction [7, 22, 46].

Current approaches to GUI grounding employ two main training paradigms, as illustrated in Fig. 1: (1) Supervised Fine-Tuning (SFT) [7, 12, 15, 22, 30, 34, 39, 45, 46, 50], which directly learns from ground-truth bounding box annotations, and (2) Reinforcement Learning with Verifiable Rewards (RLVR) [6,18,23–27,41,54,57,61], which requires ground-truth labels to compute accurate rewards for policy optimization. However, both paradigms face a fundamental challenge: the heavy reliance on annotated data. Obtaining pixel-level bounding box annotations is prohibitively expensive and time-consuming, requiring manual labeling of precise coordinates for each UI element. This dependency on labeled data severely limits the scalability and practical deployment of GUI agents across diverse applications and domains.

A natural question arises: can we enhance GUI grounding capabilities without explicit supervision? Test-time reinforcement learning (TTRL) [63] ofers a promising direction, having demonstrated success in mathematical reasoning tasks through label-free adaptation. However, applying TTRL to GUI grounding presents unique challenges. The region consistency supervision based on predicted bounding boxes introduces large supervision errors, leading to marginal performance gains [9]. Unlike mathematical problems where correctness can be verified through symbolic computation, GUI tasks require spatial reasoning over visual elements where ground truth is unavailable during inference.

![](images/0646182512ed5dad8ca6f37f94fb510c8d760946aa46fdd9d38413f078c65a02.jpg)  
Fig. 1: Training paradigms for GUI grounding and validation of coordinate-token confidence. Left: Comparison of training methods: (a) SFT and (b) RLVR require ground truth labels, while (c) our $\mathrm { C A L / C A N I }$ methods leverage this confidence signal for label-free training. Right: Distribution of coordinate-token confidence on ScreenSpot-V2 shows clear separation between correct and incorrect predictions, with correct predictions exhibiting significantly higher confidence values.

To address this challenge, we propose leveraging the model’s own confidence signals as a source of supervision. Our key insight is that while model’s most confident predictions may not always be correct, its coordinate-level token probabilities provide valuable signals for distinguishing likely correct from incorrect predictions. As illustrated in Fig. 1, we empirically observe that the confidence distributions of correct and incorrect predictions form distinct, well-separated clusters when focusing on coordinate tokens. This observation motivates our development of coordinate-token confidence, a metric that focuses specifically on the probability values of tokens representing spatial coordinates rather than the entire response sequence.

Building on this foundation, we introduce Confidence-Anchored Learning (CAL), which generates multiple candidate predictions, selects the most confident one as a pseudo-label based on coordinate-token confidence, and performs reinforcement learning using binary rewards derived from spatial distances to this pseudo-label. As shown in Fig. 1(c), our approach eliminates the need for ground truth labels by using these pseudo-labels to assign rewards directly, enabling truly label-free training. However, pseudo-labels inherently contain errors that could negatively impact learning. This leads to a critical observation: in the sparse coordinate space of GUI grounding, negative samples (predictions far from the pseudo-label) are overwhelmingly likely to be incorrect, while positive samples near potentially misplaced pseudo-labels may be unreliable. This asymmetry motivates Confidence-Anchored Negative Learning (CANL), which modifies the advantage computation to zero out potentially unreliable positive samples while preserving negative learning signals. By focusing solely on reliably incorrect predictions, CANL transforms the challenge of pseudo-label uncertainty into an opportunity for robust learning.

Evaluation across four benchmarks validates our approach. Without any annotations, CANL-7B achieves 92.1% on ScreenSpot-V2, bringing a 4% performance improvement. CANL-7B reaches 33.8% on ScreenSpot-Pro, an 8.9% absolute improvement over the base model, exceeding GUI-R1-7B which requires ground truth rewards. On challenging benchmarks ScreenSpot-Pro and UI-Vision, CANL consistently outperforms CAL by 0.6-1.5%, confirming that negative samples provide more reliable supervision when targets are small and sparse. Analysis reveals coordinate-token confidence outperforms eight alternative pseudo-labeling strategies by 2.1-11.9%, while CANL maintains stable performance across predefined distance thresholds compared to CAL’s variance, demonstrating superior robustness for practical deployment. Furthermore, Experiments on AndroidWorld demonstrate that the grounding performance gains delivered by our methods can be efectively applied to GUI navigation tasks.

Our contributions can be summarized as follows:

– We propose CAL, a label-free training approach for GUI grounding that introduces coordinate-token confidence to identify optimal predictions among multiple candidates and uses these as pseudo-labels for reinforcement learning without manual annotations.

We further develop CANL, which exclusively uses negative samples during reinforcement learning, demonstrating that learning from reliably incorrect predictions can be more efective than using potentially inaccurate positive samples in label-free settings.

– Extensive experiments demonstrate that our label-free approaches achieve competitive performance across GUI Grounding benchmarks.

## 2 Related Work

## 2.1 GUI Grounding

GUI grounding bridges natural language instructions with GUI elements, enabling agents to understand and interact with software environments through grounded multimodal reasoning [7, 35, 37]. Given a screenshot and a natural language command, the task requires identifying the corresponding UI element and returning its location as either a bounding box or a point coordinate. Early eforts in enhancing GUI grounding capabilities primarily relied on supervised fine-tuning (SFT) [7, 12, 15, 46, 50] using large-scale annotated datasets. Building on the success of GRPO and DeepSeek-R1 [14, 32], recent work has shifted toward RLVR, where models leverage interaction signals and feedback to further enhance grounding performance [18, 23, 25, 27, 36, 52, 54, 61]. However, both SFT and RLVR paradigms fundamentally depend on labeled data, where manual annotation is prohibitively expensive and automated labeling processes often introduce errors, creating a major bottleneck for scaling GUI agents to diverse applications and domains. GUI-RCPO [9] adopts unsupervised training with predicted bounding boxes, yet erroneous supervision yields marginal gains.

## 2.2 Negative Learning

Negative learning (NL) reframes supervised learning by training models on what to avoid rather than what to produce. It operates on the principle that model’s least confident predictions serve as reliable negative signals, even when its most confident predictions may be incorrect [19]. This approach has proven efective for tasks with noisy labels and in few-shot settings where robust training signals are scarce [44]. Recently, NL has been applied to large language models for mathematical reasoning, using either an increased ratio of negative samples [5, 43] or negative samples exclusively [62]. Despite its success in text-based domains, the application of NL to multimodal models remains unexplored. Our work explore whether learning solely from negative examples can be an efective strategy for complex vision-language tasks such as GUI grounding.

## 3 Method

We propose a label-free training paradigm for GUI grounding. Our approach consists of two key components: (1) coordinate-token confidence-based pseudolabel generation that selects the most reliable prediction from multiple samples, and (2) distance-based reinforcement learning that assigns rewards without ground truth. We present two variants: Confidence-Anchored Learning (CAL) that performs policy optimization using both positive and negative rewards, and Confidence-Anchored Negative Learning (CANL) that exclusively learns from negative samples to mitigate pseudo-label errors in the GUI coordinate space.

![](images/baa4a9b2d7fef41ffb79a37dfef252c54f42261f80edb2272d0b6ac7a69cce92.jpg)  
Fig. 2: The CANL pipeline. Given an instruction and screenshot, the model generates multiple predictions with associated coordinate-token confidence scores, selects the highest-confidence prediction as a pseudo-label, constructs a pseudo-area using the distance threshold τ, and assigns binary rewards. During policy optimization, CANL leverages only negative samples (gray) while zeroing advantages for potentially unreliable positive samples (orange), as illustrated in the bottom panel showing the transformation from standard to negative-only reinforcement learning.

## 3.1 Confidence-Based Pseudo-Label Generation

The core challenge in label-free GUI grounding is identifying reliable training signals from the model’s own predictions. Standard confidence estimation averages probabilities across all tokens:

$$
c ( y \mid x ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } P ( y _ { i } \mid y _ { 1 } , \ldots , y _ { i - 1 } , x )\tag{1}
$$

However, this approach fails in GUI tasks as high-probability formatting and descriptive tokens mask the uncertainty inherent in critical coordinate values. Our analysis reveals that coordinate tokens—the numerical values specifying pixel locations—carry the primary uncertainty signal, with 71.5% having probabilities below 0.6 while 78.1% of non-coordinate tokens exceed 0.9 on ScreenSpot-V2. We therefore introduce coordinate-token confidence, which focuses exclusively on these informative tokens:

$$
c _ { \mathrm { c o o r d } } ( y \mid x _ { i m g } , x _ { i n s } ) = { \frac { 1 } { k } } \sum _ { i \in \mathrm { c o o r d } } P ( y _ { i } \mid y _ { 1 } , \dots , y _ { i - 1 } , x _ { i m g } , x _ { i n s } )\tag{2}
$$

where k represents the number of coordinate tokens. As shown in Fig. 1 right, this metric efectively distinguishes correct from incorrect predictions. Given an image–instruction pair $( x _ { i m g } , x _ { i n s } )$ , we generate multiple predictions through

parallel sampling and select the one with highest coordinate-token confidence as our pseudo-label:

$$
\hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } c _ { \mathrm { c o o r d } } ( y \mid x _ { i m g } , x _ { i n s } ) , \quad \hat { p } = \mathrm { e x t r a c t } ( \hat { y } )\tag{3}
$$

where yˆ represents the optimal response selected by coordinate-token confidence, and $\mathcal { V }$ denotes the collection consisting of generated responses. The function extract(y) is defined to obtain a point from the response y. pˆ indicates the optimal predicted point, i.e. the pseudo-label. This confidence-based selection provides a principled basis for self-supervised learning without requiring ground truth annotations.

## 3.2 Confidence-Anchored Reinforcement Learning

To enable reinforcement learning without ground truth annotations, we design a distance-based reward mechanism using our pseudo-labels. For the ith sampled prediction $( x _ { i } , y _ { i } )$ , given an image with height H and width W, we normalize its coordinates as $( x _ { i } / W , y _ { i } / H )$ , and compute its Euclidean distance to the pseudolabel and assign binary rewards:

$$
R _ { i } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \mathrm { d i s t } ( p _ { i } , { \hat { p } } ) \leq \tau } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{4}
$$

This formulation exploits the sparse nature of GUI coordinate space where predictions close to our high-confidence pseudo-label are likely correct, while distant predictions are almost certainly wrong. The distance threshold τ defines the boundary between potentially correct and incorrect predictions. We then apply Group Relative Policy Optimization (GRPO) [32], which estimates advantages through standardization across multiple samples:

$$
A _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { N } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { N } ) }\tag{5}
$$

where N denotes the sampling number. The policy optimization follows the standard GRPO objective with clipped probability ratios and KL regularization:

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q , o _ { i } } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left[ \operatorname* { m i n } \left( r _ { i } ( \theta ) A _ { i } , \operatorname { c l i p } ( r _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \right) - \beta \mathbb { D } _ { \mathrm { K L } } [ \pi _ { \theta } | \pi _ { \mathrm { r e f } } ] \right] .\tag{6}
$$

where the probability ratio $r _ { i } ( \theta )$ prevents excessive policy updates, $\beta$ controls the strength of regularization, and $G$ denotes the group size. By treating the model’s most confident prediction as a learning target and using spatial proximity to define rewards, CAL transforms GUI grounding into a self-supervised reinforcement learning problem requiring no external annotations.

## 3.3 Confidence-Anchored Negative Reinforcement Learning

While CAL provides a path to label-free training, pseudo-labels inevitably contain errors that can mislead learning, particularly when incorrect predictions are mistakenly rewarded as positive examples. However, the sparse nature of GUI coordinate space ofers an opportunity: negative samples are overwhelmingly reliable since predictions far from any reasonable target are almost certainly incorrect. CANL exploits this asymmetry by modifying the advantage computation to use only negative samples:

$$
A _ { i } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } R _ { i } = 1 } \\ { \frac { - \operatorname* { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { N } ) } { \operatorname* { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { N } ) } , } & { \mathrm { i f ~ } R _ { i } = 0 } \end{array} \right.\tag{7}
$$

This approach zeros out advantages for potentially unreliable positive samples while preserving the learning signal from negative samples. The model thus learns exclusively from what to avoid rather than what to produce, which proves particularly efective in high-resolution GUI tasks where the vast coordinate space makes distant predictions almost certainly incorrect. As illustrated in Fig. 2, CANL maintains the same pseudo-label generation and reward assignment pipeline as CAL but selectively updates the model using only the most reliable learning signals. This strategy efectively leverages label uncertainty as a catalyst for robust negative learning, yielding superior performance on complex datasets where pseudo-label quality is often vulnerable to noise.

## 4 Experiments

## 4.1 Experiments Setup

Datasets and Benchmarks. We evaluate on four benchmarks: ScreenSpot [7] and ScreenSpot-V2 [46], which have low-resolution interfaces and larger targets, and more challenging benchmark ScreenSpot-Pro [21] and UI-Vision [28]. For UI-Vision, we only use the Element Grounding subset for training and evaluation, without Layout Grounding and Action Prediction subset. Predictions are correct if they fall within ground truth bounding boxes.

Implementation Details. We use Qwen-2.5-VL [4] (3B/7B) as the base model within the VLM-R1 framework [33]. Following [63], we adopt a test-time training setting where the model is optimized and evaluated on each benchmark independently, obviating the need for a separate training set. All trainings are conducted for 1 epoch, with learning rate 1e-6. To enhance diversity during sampling, we set temperature $T = 1 . 0$ , top\_k = 50, top\_ $p = 1 . 0$ . KL penalty β is set to 0.04. Distance threshold τ is set to 0.05. We apply Flash Attention2 [8] for training. Following [63], to reduce computational costs while improve the accuracy of pseudo-labels, we apply downsampling strategy during training. Specifically, we sample 16 responses to construct pseudo labels. For CAL, we randomly downsample 8 responses to compute the advantages. For CANL, we compute the advantages using 16 responses and then select the 8 negative responses farthest from the pseudo-label for policy optimization.

Table 1: Performance comparison on ScreenSpot-V1 and V2. "-" indicates missing values due to unavailable results, unreleased checkpoints, and code. For label-free methods, the optimal and the suboptimal results are bolded and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">GUI Labels</td><td colspan="2">v1 Mobile</td><td colspan="2">v1 Desktop</td><td colspan="2">v1 Web</td><td rowspan="2">v1 Avg.</td><td rowspan="2">v2 Avg.</td></tr><tr><td>Text</td><td>Icon</td><td>Text</td><td>Icon</td><td>Text</td><td>Icon</td></tr><tr><td colspan="8">Proprietary Models</td><td></td></tr><tr><td>GPT-40 [29]</td><td>=</td><td>30.5</td><td>23.2</td><td>20.6</td><td>19.4</td><td>11.1</td><td>7.8</td><td>18.8</td><td>20.1</td></tr><tr><td>Claude Computer Use [1]</td><td>=</td><td>=</td><td></td><td></td><td></td><td>=</td><td>=</td><td>83.0</td><td></td></tr><tr><td colspan="8"></td></tr><tr><td>General Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-2.5-VL-3B [4] Qwen-2.5-VL-7B [4]</td><td>0 0</td><td>93.8 91.9</td><td>68.1 80.8</td><td>91.2 88.1</td><td>55.0 75.7</td><td>81.7 90.0</td><td>64.6 77.7</td><td>77.6 84.9</td><td>82.1 88.1</td></tr><tr><td colspan="8"></td></tr><tr><td>GUI-specific Models (Label-required) CogAgent-18B [15]</td><td>222M</td><td>67.0</td><td>24.0</td><td>74.2</td><td>20.0</td><td>70.4</td><td>28.6</td><td>47.4</td><td></td></tr><tr><td>SeeClick-9.6B [7]</td><td>1M</td><td>78.0</td><td>52.0</td><td>72.2</td><td>30.0</td><td>55.7</td><td>32.5</td><td>53.4</td><td>55.1</td></tr><tr><td>UGround-7B [12]</td><td>10M</td><td>82.8</td><td>60.3</td><td>82.5</td><td>63.6</td><td>80.4</td><td>70.4</td><td>73.3</td><td>76.3</td></tr><tr><td>OS-Atlas-7B [46]</td><td>13M</td><td>93.0</td><td>72.9</td><td>91.8</td><td>62.9</td><td>90.9</td><td>74.3</td><td>82.5</td><td></td></tr><tr><td>ShowUI-2B [22]</td><td>256K</td><td>92.3</td><td>75.5</td><td>76.3</td><td>61.1</td><td>81.7</td><td>63.6</td><td>75.1</td><td>77.3</td></tr><tr><td>Aguvis-72B [50]</td><td>1M</td><td>94.5</td><td>85.2</td><td>95.4</td><td>77.9</td><td>91.3</td><td>85.9</td><td>89.2</td><td></td></tr><tr><td>UI-TARS-7B [30]</td><td>18.4M</td><td>94.5</td><td>85.2</td><td>95.9</td><td>85.7</td><td>90.0</td><td>83.5</td><td>89.5</td><td>91.6</td></tr><tr><td>UI-TARS-72B [30]</td><td>18.4M</td><td>94.9</td><td>82.5</td><td>89.7</td><td>88.6</td><td>88.7</td><td>85.0</td><td>88.4</td><td>90.3</td></tr><tr><td>GUI-Actor-7B [45]</td><td>9.6M</td><td>94.9</td><td>82.1</td><td>91.8</td><td>80.0</td><td>91.3</td><td>85.4</td><td>88.3</td><td>92.1</td></tr><tr><td>Jedi-7B [47]</td><td>4M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>91.7</td></tr><tr><td>UI-R1-3B [25]</td><td>136</td><td>95.6</td><td>84.7</td><td>90.2</td><td>59.3</td><td>85.2</td><td>73.3</td><td>83.3</td><td>85.4</td></tr><tr><td>GUI-R1-7B [27]</td><td>3K</td><td></td><td></td><td>91.8</td><td>73.6</td><td>91.3</td><td>75.7</td><td></td><td></td></tr><tr><td>InfiGUI-R1-3B [23]</td><td>32K</td><td>97.1</td><td>81.2</td><td>94.3</td><td>77.1</td><td>91.7</td><td>77.6</td><td>87.5</td><td></td></tr><tr><td>SE-GUI-7B [54]</td><td>3K</td><td></td><td></td><td></td><td></td><td></td><td></td><td>88.2</td><td>90.3</td></tr><tr><td>GuirlVG-7B [18]</td><td>5.2K</td><td>96.0</td><td>84.7</td><td>92.8</td><td>80.0</td><td>92.6</td><td>85.9</td><td>88.7</td><td>91.9</td></tr><tr><td colspan="8">GUI-specific Models (Label-free)</td><td></td></tr><tr><td>GUI-RCPO-7B [9]</td><td>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>86.6</td><td>88.9</td></tr><tr><td colspan="8">Ours</td><td></td></tr><tr><td>CAL-3B</td><td>0</td><td>96.7</td><td>78.6</td><td>95.4</td><td>67.9</td><td>87.8</td><td>72.8</td><td>84.6</td><td>88.9</td></tr><tr><td>CANL-3B</td><td>0</td><td>96.0</td><td>79.0</td><td>96.4</td><td>66.4</td><td>87.0</td><td>73.8</td><td>84.5</td><td>88.5</td></tr><tr><td>CAL-7B</td><td>0</td><td>97.1</td><td>87.3</td><td>86.1</td><td>80.7</td><td>91.7</td><td>83.0</td><td>88.6</td><td>92.1</td></tr><tr><td>CANL-7B</td><td>0</td><td>96.7</td><td>87.3</td><td>88.6</td><td>82.1</td><td>91.3</td><td>84.0</td><td>89.2</td><td>92.1</td></tr></table>

Baselines. We compare against three categories of methods: (1) Label-required models, including SFT paradigms (e.g., SeeClick [7], ShowUI [22], and UI-TARS [30]) and RL approaches (e.g., UI-R1 [25], GUI-R1 [27]); (2) Label-free model: GUI-RCPO [9], which leverages region consistency for training guidance; (3) Proprietary models such as GPT-4o [29] and Claude Computer Use [1]. Both our method and GUI-RCPO are trained on test sets, enabling a fair performance comparison. By contrast, comparisons with label-required models mainly highlight our data eficiency.

## 4.2 Main Results

Our label-free methods achieve competitive performance without any annotations. Table 1 demonstrates substantial improvements over vanilla base models on ScreenSpot and ScreenSpot-V2, with absolute gains ranging from 4.0% to 7%. On ScreenSpot-V2, CANL-7B achieves 92.1%, improving 4.0% over the base Qwen-2.5-VL-7B, outperforming GUI-RCPO-7B (88.9%), proving the reward signal of our methods is more reliable. Meanwhile, CAL-7B matches GUI-Actor-7B (92.1%) which uses 9.6M labels. At the 3B scale, CANL improves the base model from 77.6% to 84.5% (+6.9%), approaching InfiGUI-R1-3B (87.5%) which requires 32K labels, validating that coordinate-token confidence provides suficient supervision.

Table 2: Performance comparison on ScreenSpot-Pro. For label-free methods, the optimal and the suboptimal results are bolded and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">GUI Labels</td><td colspan="2">CAD</td><td colspan="2">Dev</td><td colspan="2">Creative</td><td colspan="2">Scientific</td><td colspan="2">Office</td><td colspan="2">OS</td><td rowspan="2">Avg.</td></tr><tr><td>Text</td><td>Icon</td><td>Text</td><td>Icon</td><td>Text Icon</td><td></td><td>Text</td><td>Icon</td><td>Text Icon</td><td>Text</td><td>Icon</td><td></td></tr><tr><td colspan="10">Proprietary Models</td><td colspan="7"></td></tr><tr><td>GPT-4o [29]</td><td>=</td><td>2.0</td><td>0.0</td><td>1.3</td><td>0.0</td><td>1.0</td><td>0.0</td><td>2.1</td><td>0.0</td><td>1.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.8</td></tr><tr><td>Claude Computer Use [1]</td><td></td><td>14.5</td><td>3.7</td><td>22.0</td><td>3.9</td><td>25.9</td><td>3.4</td><td>33.9</td><td>15.8</td><td>30.1</td><td>16.3</td><td>11.0</td><td>4.5</td><td>17.1</td></tr><tr><td colspan="10">General Models</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Qwen-2.5-VL-3B [4]</td><td>0</td><td>9.1</td><td>7.3</td><td>22.1</td><td>1.4</td><td>26.8</td><td>2.1</td><td>38.2</td><td>7.3</td><td>33.9</td><td>15.1</td><td>10.3</td><td>1.1</td><td>16.1</td></tr><tr><td>Qwen-2.5-VL-7B [4]</td><td>0</td><td>13.7</td><td>7.8</td><td>44.2</td><td>6.9</td><td>28.8</td><td>10.5</td><td>48.6</td><td>5.5</td><td>46.9</td><td>15.1</td><td>31.8</td><td>12.4</td><td>24.9</td></tr><tr><td colspan="10">GUI-specific Models (Label-required)</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>SeeClick-9.6B [7]</td><td>1M</td><td>2.5</td><td>0.0</td><td>0.6</td><td>0.0</td><td>1.0</td><td>0.0</td><td>3.5</td><td>0.0</td><td>1.1</td><td>0.0</td><td>2.8</td><td>0.0</td><td>1.1</td></tr><tr><td>CogAgent-18B [15]</td><td>222M</td><td>7.1</td><td>3.1</td><td>14.9</td><td>0.7</td><td>9.6</td><td>0.0</td><td>22.2</td><td>1.8</td><td>13.0</td><td>0.0</td><td>5.6</td><td>0.0</td><td>7.7</td></tr><tr><td>Aria-UI [53]</td><td>17.6M</td><td>7.6</td><td>1.6</td><td>16.2</td><td>0.0</td><td>23.7</td><td>2.1</td><td>27.1</td><td>6.4</td><td>20.3</td><td>1.9</td><td>4.7</td><td>0.0</td><td>11.3</td></tr><tr><td>OS-Atlas-7B [46]</td><td>13M</td><td>12.2</td><td>4.7</td><td>33.1</td><td>1.4</td><td>28.8</td><td>2.8</td><td>37.5</td><td>7.3</td><td>33.9</td><td>5.7</td><td>27.1</td><td>4.5</td><td>18.9</td></tr><tr><td>ShowUI-2B [22]</td><td>256K</td><td>2.5</td><td>0.0</td><td>16.9</td><td>1.4</td><td>9.1</td><td>0.0</td><td>13.2</td><td>7.3</td><td>15.3</td><td>7.5</td><td>10.3</td><td>2.2</td><td>7.7</td></tr><tr><td>UGround-7B [12]</td><td>10M</td><td>14.2</td><td>1.6</td><td>26.6</td><td>2.1</td><td>27.3</td><td>2.8</td><td>31.9</td><td>2.7</td><td>31.6</td><td>11.3</td><td>17.8</td><td>0.0</td><td>16.5</td></tr><tr><td>ZonUI-3B [16]</td><td>24K</td><td>31.9</td><td>15.6</td><td>24.6</td><td>6.2</td><td>40.9</td><td>7.6</td><td>54.8</td><td>18.1</td><td>57.0</td><td>26.4</td><td>19.6</td><td>7.8</td><td>28.7</td></tr><tr><td>UI-R1-3B [25]</td><td>136</td><td>11.2</td><td>6.3</td><td>22.7</td><td>4.1</td><td>27.3</td><td>3.5</td><td>42.4</td><td>11.8</td><td>32.2</td><td>11.3</td><td>13.1</td><td>4.5</td><td>17.8</td></tr><tr><td>GUI-R1-7B [27]</td><td>3K</td><td>23.9</td><td>6.3</td><td>49.4</td><td>4.8</td><td>38.9</td><td>8.4</td><td>55.6</td><td>11.8</td><td>58.7</td><td>26.4</td><td>42.1</td><td>16.9</td><td>31.0</td></tr><tr><td>UI-Ins-32B [6]</td><td>316K</td><td>51.8</td><td>29.7</td><td>83.1</td><td>26.9</td><td>69.7</td><td>18.9</td><td>83.3</td><td>34.5</td><td>88.7</td><td>50.9</td><td>70.1</td><td>34.8</td><td>57.0</td></tr></table>

<table><tr><td colspan="14"></td></tr><tr><td>GUI-specific Models (Label-free) GUI-RCPO-7B [9]</td><td>0</td><td>=</td><td>1</td><td>=</td><td>=</td><td>=</td><td>=</td><td>=</td><td>1</td><td>=</td><td>=</td><td>=</td><td>=</td><td>25.9</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CAL-3B</td><td>0</td><td>35.0</td><td>9.4</td><td>44.2</td><td>3.4</td><td>46.5</td><td>4.9</td><td>60.4</td><td>18.1</td><td>53.1</td><td>20.8</td><td>38.3</td><td>7.9</td><td>32.1</td></tr><tr><td>CANL-3B</td><td>0</td><td>39.1</td><td>7.8</td><td>42.9</td><td>4.8</td><td>46.0</td><td>3.5</td><td>58.3</td><td>20.9</td><td>55.9</td><td>22.6</td><td>38.3</td><td>7.9</td><td>32.7</td></tr><tr><td>CAL-7B</td><td>0</td><td>25.4</td><td>11.0</td><td>51.3</td><td>6.9</td><td>38.9</td><td>9.8</td><td>60.4</td><td>15.5</td><td>64.7</td><td>19.2</td><td>37.4</td><td>19.2</td><td>32.7</td></tr><tr><td>CANL-7B</td><td>0</td><td>29.4</td><td>10.9</td><td>53.2</td><td>9.0</td><td>37.4</td><td>8.4</td><td>63.2</td><td>14.5</td><td>62.1</td><td>26.4</td><td>40.2</td><td>15.7</td><td>33.8</td></tr></table>

![](images/fe0ceda597826e049cea688c7fab7ba404e6a5d457dab09558796a305e801d1e.jpg)

![](images/057a461ff98ecc70067d3cd614e7f91571efa8cfc9544e0aca8514b6dace8538.jpg)  
Fig. 3: Performance of diferent pseudo-label construction methods. Left: Comparison of pseudo-label construction methods with sampling number N=12. Right: Pseudolabel quality scales with sampling budget across four strategies.

Negative learning dominates on challenging high-resolution benchmarks. As shown in Tab. 2 and Tab. 3, the performance gap between CAL and CANL reveals a critical pattern: as dificulty increases, exclusive negative learning becomes increasingly advantageous. On ScreenSpot-Pro, CANL-7B achieves 33.8%, improving 1.1% over CAL-7B, 8.9% over the base model, and GUI-RCPO-7B (25.9%) by a notable margin. This advantage extends to UI-Vision where CANL-7B reaches 20.1%, surpassing CAL by 1.5%. The systematic superiority of CANL confirms that on high-resolution and challenging tasks, negative samples provide more reliable learning signals than potentially incorrect positive samples derived from pseudo-labels.

Table 3: Performance comparison on UI-Vision. For label-free methods, the optimal and the suboptimal results are bolded and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">GUI Labels</td><td colspan="6">Grouped by Category</td><td colspan="2">Grouped by Setting</td><td></td><td rowspan="2">Overall</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Edu. Browser Dev. Prod. Creative Entert. Basic Func.</td><td></td><td>Spatial</td></tr><tr><td colspan="10">Proprietary Models</td><td></td><td></td><td></td></tr><tr><td>GPT-4o [29]</td><td>1</td><td>1.5</td><td>0.0</td><td>2.2</td><td>1.1</td><td>0.8</td><td>4.2</td><td>1.6</td><td>1.5</td><td>1.0</td><td>1.4</td></tr><tr><td>Claude Computer Use [1]</td><td>1</td><td>6.1</td><td>9.8</td><td>8.0</td><td>9.4</td><td>7.7</td><td>8.3</td><td>9.5</td><td>7.7</td><td>7.6</td><td>8.3</td></tr><tr><td>General Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-2.5-VL-3B [4]</td><td>0</td><td>7.6</td><td>22.4</td><td></td><td>15.4 12.8</td><td>6.6</td><td>33.9</td><td>18.6</td><td>13.8</td><td>4.3</td><td>12.0</td></tr><tr><td>Qwen-2.5-VL-7B [4]</td><td>0</td><td>11.1</td><td>37.1</td><td>18.1</td><td>15.4</td><td>9.6</td><td>29.7</td><td>20.0</td><td>18.6</td><td>7.1</td><td>15.0</td></tr><tr><td colspan="10">GUI-specific Models (Label-required)</td><td></td><td></td></tr><tr><td>SeeClick-9.6B [7]</td><td>1M</td><td>4.2</td><td>13.3</td><td>7.3</td><td>4.3</td><td>4.0</td><td>11.0</td><td>9.4</td><td>4.7</td><td>2.1</td><td>5.4</td></tr><tr><td>ShowUI-2B [22]</td><td>256K</td><td>3.7</td><td>13.3</td><td>7.5</td><td>6.5</td><td>2.5</td><td>15.6</td><td>8.1</td><td>7.7</td><td>2.1</td><td>5.9</td></tr><tr><td>CogAgent-9B [15]</td><td>222M</td><td>8.7</td><td>11.2</td><td>8.6</td><td>10.3</td><td>5.6</td><td>15.6</td><td>12.0</td><td>12.2</td><td>2.6</td><td>8.9</td></tr><tr><td>OSAtlas-7B [46]</td><td>13M</td><td>8.7</td><td>16.8</td><td>10.3</td><td>9.2</td><td>5.6</td><td>16.2</td><td>12.2</td><td>11.2</td><td>3.7</td><td>9.0</td></tr><tr><td>AriaUI [53]</td><td>17.6M</td><td>9.0</td><td>18.9</td><td>11.2</td><td>10.4</td><td>6.5</td><td>19.3</td><td>12.2</td><td>14.0</td><td>4.0</td><td>10.1</td></tr><tr><td>UGround-v1-7B [12]</td><td></td><td>10.4</td><td>28.7</td><td>17.5</td><td>12.2</td><td>8.6</td><td>18.2</td><td>15.4</td><td>17.1</td><td>6.3</td><td>12.9</td></tr><tr><td>Aguvis-7B [50] UI-TARS-7B [30]</td><td>1M 18.4M</td><td>13.1 14.2</td><td>30.8 35.0</td><td>17.1 19.7</td><td>12.1 18.3</td><td>9.6 11.1</td><td>24.0 38.5</td><td>17.8 20.1</td><td>18.3 24.3</td><td>5.1 8.4</td><td>13.7 17.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Ours</td><td></td><td></td><td></td></tr><tr><td>CAL-3B</td><td>0</td><td>15.1</td><td>32.9</td><td>20.6</td><td>18.1</td><td>10.4</td><td>37.0</td><td>24.9</td><td>22.1</td><td>5.7</td><td>17.2</td></tr><tr><td>CANL-3B</td><td>0</td><td>16.0</td><td>33.6</td><td>21.6</td><td>19.4</td><td>12.0</td><td>39.1</td><td>26.4</td><td>23.8</td><td>6.5</td><td>18.5</td></tr><tr><td>CAL-7B</td><td>0</td><td>17.6</td><td>42.7</td><td>22.0</td><td>18.9</td><td>10.9</td><td>41.1</td><td>25.6</td><td>22.9</td><td>8.4</td><td>18.6</td></tr><tr><td>CANL-7B</td><td>0</td><td>17.3</td><td>38.5</td><td>24.1</td><td>20.9</td><td>12.4</td><td>44.8</td><td>27.5</td><td>25.1</td><td>8.8</td><td>20.1</td></tr></table>

![](images/be986c26159363ce1feae6025f8d8e6686b9363f29f1351b5f70dbf6daae53da.jpg)

![](images/f151f6404c25353b4486845758cb589653b5b0fb4355fab0f7421f58134741ef.jpg)  
Fig. 4: Left: Reward accuracy on ScreenSpot-V2. Right: Reward accuracy on ScreenSpot-Pro.

## 4.3 In-depth Analysis

Coordinate-token confidence outperforms alternative pseudo-labeling strategies. Figure 3 evaluates diferent test-time scaling methods to generate pseudo-label on ScreenSpot-V2. Detailed descriptions of these methods are provided in Appendix A.5. Our proposed coordinate-token confidence (C-Conf) achieves 83.9% accuracy with N = 12 samples, substantially exceeding alternatives. The 5.3% gain over standard confidence validates our core insight that coordinate tokens carry the primary uncertainty signal in GUI grounding, as formatting and descriptive tokens exhibit consistently high probabilities regardless of prediction quality. Moreover, this superiority persists across varying sampling budgets: C-Conf maintains its advantage from minimal sampling $( N = 2 { : }$ 76.7% vs 74.8% for standard confidence) to larger ensembles (N = 12: 83.9% vs 78.6%). Notably, while spatial methods like Center collapse at higher sampling rates due to averaging efects, C-Conf shows monotonic improvement, confirming that confidence-based selection scales efectively with computational budget.

Reward reliability validates the asymmetric quality of positive and negative samples. Figure 4 quantifies the actual correctness of our binary reward assignments by comparing against ground truth labels. On ScreenSpot-V2 with larger bounding boxes, CANL’s reward accuracy falls slightly below CAL’s because some correct predictions lying outside threshold τ are misclassified as negative. However, CANL maintains near-perfect negative sample reliability (>0.95) while CAL’s mixed positive-negative accuracy hovers around 0.5. This empirical validation confirms our core hypothesis that high-resolution GUI tasks naturally provide accurate negative signals even without labels, as the vast coordinate space makes random distant predictions almost certainly incorrect.

Distance threshold sensitivity reveals fundamental diferences between positive and negative learning. Table 4 examines how the threshold τ afects both methods across datasets. On ScreenSpot-V2, CAL exhibits strong sensitivity with performance degrading from 88.9% at $\tau = 0 . 0 5$ to 86.9% at $\tau = 0 . 1$ , reflecting its reliance on accurate positive sample identification. Conversely, CANL maintains remarkable stability at 88.5%±1.0% across all thresholds. This pattern intensifies on ScreenSpot-Pro where CAL’s performance drops precipitously from 33.2% to 30.1% as τ increases, while CANL remain stable. This invariance stems from a key property of sparse coordinate spaces: predictions far from any reasonable target remain unambiguously incorrect as distance threshold increases. While positive samples require careful boundary definition to avoid including incorrect predictions, negative samples beyond any plausible threshold provide

Table 4: Distance threshold τ sensitivity analysis.
<table><tr><td>T</td><td>SS-V2</td><td>SS-Pro</td></tr><tr><td>CAL</td><td></td><td></td></tr><tr><td>0.01</td><td>87.7</td><td>33.3</td></tr><tr><td>0.03 0.05</td><td>88.4</td><td>33.2</td></tr><tr><td>0.07</td><td>88.9</td><td>33.2 30.1</td></tr><tr><td>0.1</td><td>88.9 86.9</td><td>30.7</td></tr><tr><td></td><td></td><td></td></tr><tr><td>CANL</td><td></td><td></td></tr><tr><td>0.01</td><td>87.5</td><td>30.2</td></tr><tr><td>0.03</td><td>87.8</td><td>30.8</td></tr><tr><td>0.05</td><td>88.5</td><td>33.7</td></tr><tr><td>0.07</td><td>88.4</td><td>32.7</td></tr><tr><td>0.1</td><td>88.5</td><td>33.5</td></tr></table>

consistently reliable learning signals, making CANL robust to hyperparameter.

Training dynamics reveal distinct convergence patterns across dificulty levels. Figure 5 tracks the evolution of positive ratio, defined as the fraction of samples within distance threshold τ from the pseudo-label. On ScreenSpot-V2, both methods exhibit monotonic increases, indicating progressive prediction concentration around confident regions. ScreenSpot-Pro presents markedly different dynamics: slower convergence with higher variance, plateauing around 0.8 rather than continuing upward. This divergence reflects the inherently greater challenge of high-resolution grounding where precise localization becomes critical. Notably, CANL consistently maintains higher positive ratios than CAL across both settings, suggesting that learning exclusively from negative samples paradoxically helps models develop more concentrated predictions, possibly by more efectively pruning incorrect hypotheses from the prediction space.

![](images/f190eea8e3323300f0a67aa0f3b49684744ec621755a167364b9717cae2be682.jpg)

![](images/99b8345c6cc1dace5e35fa9a0c5cded3676f8c1e732424270453f15e4d073ef1.jpg)

Fig. 5: Left: Positive Ratio on ScreenSpot-V2. Right: Positive Ratio on ScreenSpot-Pro.  
![](images/bed6b7340edacfd1796348f762528a0365beed5591b270e2d9772dd286bae90a.jpg)

![](images/3f1954a41e6e71186cd9c8680209e4b9418981b042fe60594df7e2d6fa6a8f5e.jpg)  
Fig. 6: Left: Confidence of pseudo-label on ScreenSpot-V2. Right: Distance between the pseudo-label and the center of ground truth bounding box on ScreenSpot-V2.

Pseudo-label characteristics reveal both successes and inherent limitations. Figure 6 provides deeper insights into pseudo-label behavior. The left panel shows coordinate-token confidence increases for both correct and incorrect pseudo-labels during training, with correct predictions maintaining a consistent advantage. This convergence suggests models become increasingly confident regardless of accuracy, potentially limiting further improvements without external supervision. The right panel reveals a striking bimodal distribution: correct pseudo-labels cluster within 0.02 normalized distance of ground truth centers, while incorrect ones remain at 0.40 distance throughout training. This binary pattern validates our distance-based reward design and explains why negative samples are overwhelmingly reliable in sparse coordinate spaces. The persistence of distant incorrect pseudo-labels indicates certain challenging cases remain beyond the model’s capability without ground truth, representing an inherent limitation of label-free learning.

![](images/cc58e225ff70eb8ef023a98e3a99f98b974796a652737181e18be7c5dbfcc2ac.jpg)  
Fig. 7: Cropped images of diferent cases. Top-Left: Geometric mismatch. Top-Right: Good alignment. Bottom: Misplaced pseudo-label.

Reward estimation cases demonstrate the advantages of negative learning. To assess the reliability of our distance-based reward assignment, we analyze three common scenarios during pseudo-label generation, shown in Fig. 7: Geometric mismatch: The pseudo-label is within the target bounding box, but the ground truth’s elongated shape does not match the region generated by the pseudo-labels. Good alignment: The pseudo-label and reward region align well with the target, resulting in accurate reward estimation. Misplaced pseudo-label: The pseudo-label is far from the target, causing all samples in the reward region to be incorrectly rewarded, though negative samples remain correctly classified. These cases highlight the varying reliability of positive samples under diferent pseudo-label conditions and explain why CANL’s focus on negative samples yields more robust learning signals.

Cross-dataset experiments demonstrate the generalization capability. We train Qwen-2.5-VL-3B for one epoch on a source dataset and evaluate its transferability, as shown in Fig. 8; ScreenSpot-V1 is excluded due to documented annotation errors [46]. Training on UI-Vision yields gains of 5.3% on ScreenSpot-V2 (87.4% vs. 82.1%) and 15.6% on ScreenSpot-Pro (31.7% vs. 16.1%), while training on ScreenSpot-V2 improves UI-Vision by 4.6%. These results confirm that our method avoids overfitting, achieving robust performance on unseen distributions. Notably, on more challenging benchmarks, CANL consistently outpaces CAL, highlighting its superior robustness in complex scenarios.

![](images/3376e145dcb93918b11eb68b992b523bc59408fc23ff58823a0c98df292809b2.jpg)

![](images/71c7db85f44e71d8cb2e4ac1272f9ae1eb8881dadbeb7d274965b17f340d3767.jpg)

![](images/d241831cb56b4087feb518538515594101c18a5c9026ab3fc24af0b219b9de6b.jpg)  
Fig. 8: Cross-dataset generalization performance. Accuracy of our methods across different training sets when evaluated on ScreenSpot-V2, ScreenSpot-Pro, and UI-Vision.

Training on public datasets further validates generalization. we extend our training to GroundCUA [10], a publicly available benchmark. Specifically, we randomly sample 5k instances for a single epoch of training, with results summarized in Tab. 5. Our method exhibits robust generalization capability; even when trained on external public data, it yields substantial performance gains across evaluation benchmarks. This underscores that our label-free paradigm efectively captures the underlying spatial logic of GUI grounding rather than merely memorizing dataset-specific patterns.

Table 5: Evaluation of our methods trained on publicly available datasets.
<table><tr><td>Method</td><td colspan="3">ScreenSpot ScreenSpot-V2 ScreenSpot-Pro UI-Vision</td></tr><tr><td>Qwen-2.5-VL-3B</td><td>77.6</td><td>82.1</td><td>16.1 12.0</td></tr><tr><td>w/CAL</td><td>84.6</td><td>87.3</td><td>32.6 16.7</td></tr><tr><td>w/CANL</td><td>83.5</td><td>86.7</td><td>33.4 16.7</td></tr></table>

Online end-to-end evaluation demonstrates the efectiveness of our method in real-world scenarios. We evaluate our approach on AndroidWorld [31], a realistic platform comprising 116 tasks. To isolate the impact on GUI grounding, we adopt the SeeAct-V framework [12], which decouples planning from precise coordinate prediction. Specifically, we evaluate the model previously trained on ScreenSpot-V2 [46] using our methods. As shown in Tab. 6, our approaches yield a 4.3–5.2% performance gain. CAL slightly outperforms CANL, since the simple training dataset produces more precise pseudolabels. These results indicate that the enhanced grounding precision from our methods directly mitigates execution failures in multi-step sequences, demonstrating the practical eficacy of our methods for real-world GUI agents.

Table 6: Success rate on AndroidWorld.
<table><tr><td>Planner</td><td>Grounding</td><td>SR</td></tr><tr><td rowspan="3">GPT-4o</td><td>Qwen-2.5-VL-3B</td><td>325.0</td></tr><tr><td>w/CAL</td><td>30.2</td></tr><tr><td>w/CANL</td><td>29.3</td></tr></table>

## 5 Conclusion and Future Improvement

This work introduces a label-free paradigm that successfully decouples GUI grounding from expensive manual labeling. By identifying coordinate-token confidence as a reliable proxy for accuracy, we proposed CAL and CANL to enable autonomous optimization. Our results—particularly the success of CANL—show that in sparse GUI environments, negative reinforcement can be more efective than error-prone positive pseudo-labels. However, an inherent limitation remains: while negative samples are highly reliable in challenging scenarios, the absence of accurately positive reward signal—compared to supervised training—may constrain the further scaling of model capabilities. Future research could focus on integrating more accurate positive feedback to achieve a more synergistic balance between avoiding errors and reinforcing optimal grounding behaviors.

## Acknowledgement

This work was supported by New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123100), National Natural Science Foundation of China (No. 62506332), "Pioneer" and "Leading Goose" R&D Program of Zhejiang (NO. 2026C02A1223), and CCF-Ant-Research Fund.

## Ethics Statement

This work focuses on advancing label-free reinforcement learning methods for GUI grounding tasks. Our research does not involve the collection or annotation of human subject data, nor does it utilize personally identifiable information. The datasets employed in this study are publicly available benchmarks, ensuring that no additional privacy or ethical risks are introduced. Potential misuse of our method, such as deploying GUI agents in malicious automation scenarios, should be carefully considered by practitioners. We encourage responsible application of our approach within research and development contexts that align with ethical guidelines and benefit broader society.

## Reproducibility Statement

To ensure reproducibility, we provide detailed descriptions of training configurations, hyperparameters, and evaluation protocols in the main paper and supplementary materials. All experiments are conducted on publicly available datasets. we report results averaged over multiple runs to account for variability. These steps are intended to facilitate faithful reproduction and fair comparison of our results.

## A Appendix

## A.1 Evaluation Details

Following [63], we independently apply our methods on each benchmark to implement test-time reinforcement learning. On each dataset, we use all available original inputs (image and instruction) on each benchmark for label-free training. This section provides an overview of the benchmarks.

– ScreenSpot [7] is a widely used benchmark for GUI grounding, which contains 1272 instructions across mobile, desktop and web domains.

– ScreenSpot-V2 [46] is a enhanced version of ScreenSpot with error correction and re-annotation. It contains 1272 instructions across mobile, desktop and web domains.

– ScreenSpot-Pro [21] is designed to rigorously evaluate the grounding capabilities of MLLMs in high-resolution professional settings. It contains 1581 samples, spanning 23 applications across five industries and three operating systems.

UI-Vision [28] is a comprehensive, license-permissive benchmark for ofline, fine-grained evaluation of computer use agents in real-world desktop environments. It provides three fine-to-coarse grained tasks: (1) element grounding; (2) layout grounding; and (3) action prediction. Since our work focuses on the model’s grounding capability for target elements, we use only the element grounding component during both training and evaluation. It introduce three grounding subtasks—basic, functional, and spatial—to assess diferent aspects of GUI understanding beyond simple textual queries. These three categories contain 1772, 1772, and 1935 instructions, respectively, totaling 5749 samples.

## A.2 Sparse Rewards vs. Dense Rewards Under Label-Free Setting

In our proposed methods, the pseudo label is represented as a single point, making reward assignment based on point-to-point distance a natural choice, which corresponds to a dense reward scheme. In addition, all points can be dilated into a region according to the distance threshold τ, and the elliptical IoU between each region and the pseudo-label region can be computed as a dense reward. Dense rewards provide fine-grained supervisory signals, thereby facilitating more efective model learning. As shown in Fig. 9, training with continuous rewards (Distance) accelerates convergence. However, as training progresses, its accuracy drops noticeably below that of binary rewards, as demonstrated in Tab. 7. Although the IoU-based reward scheme slows down the model’s convergence, the resulting performance gains are noticeably smaller compared to binary reward. We contend that under noisy labels, finer reward granularity increases susceptibility to error. Specifically, continuous fine-grained rewards encourage the model to align outputs closely with the pseudo label. Even when the pseudo label lies within the target bounding box, it rarely coincides with the true center of the bounding box, thereby introducing bias. More critically, when the pseudo label falls outside the bounding box, continuous rewards exacerbate the issue by further misleading the model.

Table 7: Accuracy of diferent reward type during training on ScreenSpot-V2.
<table><tr><td>Reward Type</td><td colspan="3">50 Step 100 Step 159 Step</td></tr><tr><td>Binary Reward</td><td>87.6</td><td>87.9</td><td>88.9</td></tr><tr><td>Continuous Reward (Distance)</td><td>87.7</td><td>86.3</td><td>85.2</td></tr><tr><td>Continuous Reward (IoU)</td><td>87.1</td><td>87.4</td><td>87.7</td></tr></table>

![](images/84708d077e951d106d8de2ccc02d1d60c48e2076c56b67b0cfe302012864a9f5.jpg)

![](images/48555827498e65f726ad7c644d68919f82dfb8aec6358c02d554f1f25c2fd903.jpg)  
Fig. 9: Left: Reward of binary reward and continuous reward during training on ScreenSpot-V2. Right: Reward std of binary reward and continuous reward during training on ScreenSpot-V2.

## A.3 Pure Positive Reinforcement Learning

We hypothesize that negative samples provide more reliable supervision than positive ones in label-free settings. To validate this, we conduct an ablation study using Confidence-Anchored Positive Learning (CAPL), which reinforces pseudo-labeled positive samples exclusively. As shown in Fig. 10, CAPL exhibits faster reward convergence; however, Table 8 reveals that accuracy gains remain marginal and eventually deteriorate as training progresses. This discrepancy suggests that rapid reward convergence merely reflects the model’s tendency to collapse its predictions toward erroneous pseudo-labels, leading to severe overfitting to incorrect targets. Given the inherent noise in pseudo-labels, over-reliance on positive reinforcement misguides the optimization and triggers performance degradation. In contrast, while CANL converges more gradually, it significantly outperforms CAPL, confirming that negative signals ofer more robust supervision when ground truth is absent. Notably, on the relatively simpler

ScreenSpot-V2, positive samples retain a degree of reliability. Consequently, integrating both signals—as implemented in CAL—can further boost performance, striking a balance between cautious avoidance and constructive reinforcement.

Table 8: Accuracy of diferent sample selection mechanisms during training on ScreenSpot-V2.
<table><tr><td colspan="3">Method 50 Step 100 Step 159 Step</td></tr><tr><td>CAPL</td><td>86.3</td><td>87.2 86.7</td></tr><tr><td>CAL</td><td>87.6</td><td>87.9 88.9</td></tr><tr><td>CANL</td><td>88.4</td><td>88.0 88.5</td></tr></table>

![](images/995fddbacb18833f79cec0c0b45ee5c1836347c3c4fd5c1c69eacf9d027d3c04.jpg)

![](images/d94e52db5876eb608bbe6fbd25fe45b02b6f3a1ad7ab3b11be4a9726c6fed996.jpg)  
Fig. 10: Left: Reward of CAL, CANL and pure positive learning during training on ScreenSpot-V2. Right: Reward std of CAL, CANL and pure positive learning during training on ScreenSpot-V2.

## A.4 Reinforcement learning vs. Supervised Fine-Tuning.

We generate pseudo-labels for Supervised Fine-Tuning (SFT) using the same sampling parameters as RL and apply identical training settings. Results, shown in Tab. 9, reveal that SFT underperforms compared to RL, suggesting that RL better unlocks the model’s potential in pseudo-label settings. Unlike SFT’s fixed targets, GRPO uses relative advantage to optimize the policy, which mitigates the impact of noisy pseudo-labels, leading to better performance with imperfect data.

## A.5 Comparison of Diferent Pseudo-Label Construction Method

Table 10 and Table 11 shows the performance of diferent pseudo-label construction methods with diferent temperature and rollout number. Coordinate-token confidence, Confidence represents selecting by coordinate-token confidence and averaged all-token confidence respectively. Coordinate-token Entropy and Entropy indicates selecting by coordinate-token entropy and averaged all-token entropy respectively. Random means selecting a predicted point randomly. Majority Voting selects as the answer the point that has the largest number of neighboring points within a predefined distance threshold. Following [20], we applied Kde, Center and Medoid. Kde selects the point with highest estimated density in the plane. Center is simply getting the mean of all predicted coordinates. Medoid selects an actual predicted point that minimizes the sum of distances to all other predictions. As shown in Tab. 10, with the increase of temperature, the prediction points generated by the model become more inaccurate, leading to a decline in the performance of all methods. As the number of rollouts increases, the pseudo-labels constructed by each method become increasingly accurate, as shown in Tab. 11. Compared with other methods, our proposed approach achieves a substantial lead in performance under any temperature and number of rollouts, which demonstrated its efectiveness and robustness.

Table 9: Accuracy of RL and SFT on ScreenSpot-V2.
<table><tr><td>Method</td><td>Accuracy</td></tr><tr><td> $\mathrm { Q w e n { - } 2 . 5 { - } V L { - } 3 B }$ </td><td>82.1</td></tr><tr><td> $\mathrm { w / S F T }$ </td><td>84.6</td></tr><tr><td> $\mathrm { w / C A P L }$ </td><td>87.2</td></tr><tr><td> $\mathrm { w / C A L }$ </td><td>88.9</td></tr><tr><td> $\mathrm { w / C A N L }$ </td><td>88.5</td></tr></table>

Table 10: Performance of diferent pseudo-label construction methods with diferent temperature. Best and second-best results are shown in Bold and underline, respec tively.
<table><tr><td>Method</td><td colspan="4"> $T = 0 . 6 \ T = 0 . 7 \ T = 0 . 8 \ T = 0 . 9 \ T = 1 . 0$ </td></tr><tr><td>Coordinate-Token Confidence</td><td>83.3</td><td>83.2</td><td>82.4</td><td>82.6</td><td>80.7</td></tr><tr><td>Confidence</td><td>82.1</td><td>80.3</td><td>77.7</td><td>74.2</td><td>71.7</td></tr><tr><td>Coordinate-Token Entropy</td><td>81.6</td><td>81.4</td><td>80.9</td><td>78.2</td><td>77.4</td></tr><tr><td>Entropy</td><td>82.1</td><td>81.6</td><td>80.3</td><td>77.7</td><td>76.6</td></tr><tr><td>KDE</td><td>82.2</td><td>82.1</td><td>79.7</td><td>80.0</td><td>78.2</td></tr><tr><td>Center</td><td>74.5</td><td>70.8</td><td>68.2</td><td>62.0</td><td>55.0</td></tr><tr><td>Majority Voting</td><td>81.6</td><td>80.5</td><td>80.0</td><td>80.3</td><td>79.6</td></tr><tr><td>Medoid</td><td>82.5</td><td>81.8</td><td>80.0</td><td>79.3</td><td>78.0</td></tr><tr><td>Random</td><td>79.0</td><td>75.5</td><td>73.3</td><td>69.7</td><td>67.0</td></tr></table>

## A.6 Comparison of Diferent Coordinate-Token Confidence Calculation

Table 11: Performance of diferent pseudo-label construction methods with diferent rollout number N. Best and second-best results are shown in Bold and underline, respectively.
<table><tr><td>Method</td><td colspan="5">N = 2 N = 4 N = 6 N = 8 N = 10 N = 12</td></tr><tr><td>Coordinate-Token Confidence</td><td>76.7</td><td>81.5</td><td>82.8</td><td>83.2</td><td>83.3</td><td>83.9</td></tr><tr><td>Confidence</td><td>74.8</td><td>78.4</td><td>80.0</td><td>80.3</td><td>78.9</td><td>78.6</td></tr><tr><td>Coordinate-Token Entropy</td><td>75.7</td><td>80.1</td><td>80.7</td><td>81.4</td><td>81.4</td><td>81.4</td></tr><tr><td>Entropy</td><td>75.6</td><td>79.9</td><td>80.5</td><td>81.6</td><td>82.0</td><td>81.4</td></tr><tr><td>KDE</td><td>75.6</td><td>80.1</td><td>81.4</td><td>82.1</td><td>81.3</td><td>81.1</td></tr><tr><td>Center</td><td>73.3</td><td>74.2</td><td>72.9</td><td>70.8</td><td>71.3</td><td>72.0</td></tr><tr><td>Majority Voting</td><td>72.7</td><td>79.0</td><td>80.7</td><td>80.5</td><td>81.5</td><td>80.9</td></tr><tr><td>Medoid</td><td>72.7</td><td>80.6</td><td>81.5</td><td>81.8</td><td>82.5</td><td>81.8</td></tr><tr><td>Random</td><td>73.5</td><td>74.5</td><td>76.5</td><td>75.5</td><td>76.4</td><td>75.9</td></tr></table>

Prior work has proposed multiple approaches for computing confidence scores. Given the probability values of the generated coordinate tokens, we evaluate three confidence estimation methods: (1) Mean, which is the approach adopted in our training; (2) Product, which uses the N-th root of product of all coordinatetoken probabilities as the confidence; (3) Max, which selects the highest probability among all coordinate-token probabilities; and (4) Min, which selects the lowest probability among all coordinate-token probabilities. We conduct ex-

Table 12: Accuracy of different confidence calculation methods on ScreenSpot-V2.
<table><tr><td>Method</td><td>Accuracy</td></tr><tr><td>Mean</td><td>80.7</td></tr><tr><td>Product</td><td>80.4</td></tr><tr><td>Max</td><td>76.4</td></tr><tr><td>Min</td><td>78.1</td></tr></table>

periments with temperature T = 1.0 and rollout number N = 8, and the results are presented in Tab. 12. Experimental results indicate that directly computing the mean probability yields the most efective confidence measure. It strikes the best balance between penalizing uncertainty, providing the most reliable signal for pseudo-label selection.

## A.7 Downsampling Strategy

During training, we sample 16 responses to construct pseudo-labels, followed by downsampling 8 to optimization, which is similar to [63]. It is a critical mechanism for overcoming the fundamental challenge of no ground-truth data in our label-free setting. Sampling 16 responses (rather than 8) is crucial for two reasons: it generates a suficient variety of predictions, which is essential for discovering high-quality pseudo-labels and efective negative samples. Furthermore, downsampling 8 responses for optimization improves computational eficiency (in contrast to optimizing with all 16 responses) and enhances the negative signal quality for CANL. These 8 samples are not random, but are specifically selected as the farthest from the chosen pseudo-label. By restricting the policy update to these most distant samples, the negative reward signals are derived from the highest quality "incorrect" predictions, making the penalty mechanism highly efective for correcting errors.

## A.8 Training Hyperparameters

Table 13: Training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>T</td><td>0.05</td></tr><tr><td> $\beta$ </td><td>0.04</td></tr><tr><td> $T$ </td><td>1.0</td></tr><tr><td> $t o p \_ k$ </td><td>50</td></tr><tr><td> $t o p \_ p$ </td><td>1.0</td></tr><tr><td>learning_rate</td><td>1e-6</td></tr><tr><td>bf16</td><td>true</td></tr><tr><td> $\operatorname { t o r c h \_ d t y p e }$ </td><td>bfloat16</td></tr><tr><td>data seed</td><td>42</td></tr><tr><td>gradient_checkpointing</td><td>true</td></tr><tr><td>attn implementation</td><td>flash attention 2</td></tr><tr><td>num_train_epochs</td><td>1</td></tr><tr><td> $\mathrm { \ m a x \_ p i x e l s }$ </td><td>12845056</td></tr></table>

We provide detailed hyperparameter configurations to ensure reproducibility of our label-free training approach. Table 13 presents the core training parameters used across all experiments. The distance threshold τ=0.05 was selected based on preliminary experiments to balance between capturing genuine positive samples and maintaining negative sample reliability. We employ a relatively conservative learning rate of 1e-6 with KL penalty β=0.04 to ensure stable policy updates during reinforcement learning. Sampling parameters $( T = 1 . 0 , t o p _ { - } k = 5 0 , t o p _ { - } p = 1 . 0 )$ are configured to maximize response diversity, which proves critical for generating varied candidates for pseudo-label selection. To accommodate diferent dataset characteristics, we adjust gradient accumulation steps: 8 for ScreenSpot-V2, 4 for ScreenSpot-Pro, and 16 for UI-Vision, ensuring consistent efective batch sizes despite varying computational demands. All experiments utilize bfloat16 mixed precision training with gradient checkpointing and Flash Attention 2 for memory eficiency, enabling training on consumer-grade GPUs. The prompts are tailored to model capacity, with 3B models using a simpler single-point format while 7B models support multi-element detection, though both maintain the same coordinate-based output structure essential for our confidence computation.

## 3B Model Prompt

point to the instruction: {Question}, output its coordinates in JSON format {{"point\_2d": [x, y], "label": "object name/description"}}.

## 7B Model Prompt

Locate the UI element(s) for {Question}, output the coordinates using JSON format: [{{"point\_2d": [x, y]}}, ...]

## References

1. Anthropic: Claude computer use. Available at: https://www.anthropic.com/news/developing-computer-use (2024), accessed 27 June 2026

2. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 (2023)

3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

4. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

5. Chen, H., Zheng, K., Zhang, Q., Cui, G., Cui, Y., Ye, H., Lin, T.Y., Liu, M.Y., Zhu, J., Wang, H.: Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116 (2025)

6. Chen, L., Zhou, H., Cai, C., Zhang, J., Tong, P., Kong, Q., Zhang, X., Liu, C., Liu, Y., Wang, W., et al.: Ui-ins: Enhancing gui grounding with multi-perspective instruction-as-reasoning. arXiv preprint arXiv:2510.20286 (2025)

7. Cheng, K., Sun, Q., Chu, Y., Xu, F., Li, Y., Zhang, J., Wu, Z.: Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935 (2024)

8. Dao, T.: Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 (2023)

9. Du, Y., Yan, Y., Tang, F., Lu, Z., Zong, C., Lu, W., Jiang, S., Shen, Y.: Test-time reinforcement learning for gui grounding via region consistency (2025)

10. Feizi, A., Nayak, S., Jian, X., Lin, K.Q., Li, K., Awal, R., Lù, X.H., Obando-Ceron, J., Rodriguez, J.A., Chapados, N., et al.: Grounding computer use agents on human demonstrations. arXiv preprint arXiv:2511.07332 (2025)

11. Gao, L., Zhang, L., Gao, P., Liu, W., Luan, J., Xu, M.: Gui-shift: Enhancing vlmbased gui agents through self-supervised reinforcement learning. arXiv preprint arXiv:2505.12493 (2025)

12. Gou, B., Wang, R., Zheng, B., Xie, Y., Chang, C., Shu, Y., Sun, H., Su, Y.: Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243 (2024)

13. Gu, Z., Zeng, Z., Xu, Z., Zhou, X., Shen, S., Liu, Y., Zhou, B., Meng, C., Xia, T., Chen, W., et al.: Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833 (2025)

14. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)

15. Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., et al.: Cogagent: A visual language model for gui agents. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14281–14290 (2024)

16. Hsieh, Z., Wei, T.J., Yang, S.: Zonui-3b: A lightweight vision-language model for cross-resolution gui grounding. arXiv e-prints pp. arXiv–2506 (2025)

17. Hu, X., Xiong, T., Yi, B., Wei, Z., Xiao, R., Chen, Y., Ye, J., Tao, M., Zhou, X., Zhao, Z., et al.: Os agents: A survey on mllm-based agents for computer, phone and browser use. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 7436–7465 (2025)

18. Kang, W., Lei, B., Liu, G., Ding, C., Yan, Y.: Guirlvg: Incentivize gui visual grounding via empirical exploration on reinforcement learning. arXiv preprint arXiv:2508.04389 (2025)

19. Kim, Y., Yim, J., Yun, J., Kim, J.: Nlnl: Negative learning for noisy labels. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 101–110 (2019)

20. Lee, H., Kim, J., Kim, B., Tack, J., Jo, C., Lee, J., Park, C., In, S., Shin, J., Yoo, K.M.: Reguide: Data eficient gui grounding via spatial reasoning and search. arXiv preprint arXiv:2505.15259 (2025)

21. Li, K., Meng, Z., Lin, H., Luo, Z., Tian, Y., Ma, J., Huang, Z., Chua, T.S.: Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981 (2025)

22. Lin, K.Q., Li, L., Gao, D., Yang, Z., Wu, S., Bai, Z., Lei, S.W., Wang, L., Shou, M.Z.: Showui: One vision-language-action model for gui visual agent. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19498–19508 (2025)

23. Liu, Y., Li, P., Xie, C., Hu, X., Han, X., Zhang, S., Yang, H., Wu, F.: Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239 (2025)

24. Liu, Y., Liu, Z., Zhu, S., Li, P., Xie, C., Wang, J., Hu, X., Han, X., Yuan, J., Wang, X., et al.: Infigui-g1: Advancing gui grounding with adaptive exploration policy optimization. arXiv preprint arXiv:2508.05731 (2025)

25. Lu, Z., Chai, Y., Guo, Y., Yin, X., Liu, L., Wang, H., Xiao, H., Ren, S., Xiong, G., Li, H.: Ui-r1: Enhancing eficient action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620 (2025)

26. Lu, Z., Ye, J., Tang, F., Shen, Y., Xu, H., Zheng, Z., Lu, W., Yan, M., Huang, F., Xiao, J., et al.: Ui-s1: Advancing gui automation via semi-online reinforcement learning. arXiv preprint arXiv:2509.11543 (2025)

27. Luo, R., Wang, L., He, W., Xia, X.: Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458 (2025)

28. Nayak, S., Jian, X., Lin, K.Q., Rodriguez, J.A., Kalsi, M., Awal, R., Chapados, N., Özsu, M.T., Agrawal, A., Vazquez, D., et al.: Ui-vision: A desktop-centric gui benchmark for visual perception and interaction. arXiv preprint arXiv:2503.15661 (2025)

29. OpenAI: Introducing gpt-4o. Available at: https://openai.com/index/hello-gpt-4o (2024), accessed 27 June 2026

30. Qin, Y., Ye, Y., Fang, J., Wang, H., Liang, S., Tian, S., Zhang, J., Li, J., Li, Y., Huang, S., et al.: Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326 (2025)

31. Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., et al.: Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573 (2024)

32. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

33. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al.: Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025)

34. Sun, Q., Cheng, K., Ding, Z., Jin, C., Wang, Y., Xu, F., Wu, Z., Jia, C., Chen, L., Liu, Z., Kao, B., Li, G., He, J., Qiao, Y., Wu, Z.: Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2025)

35. Tang, F., Chen, B., Lu, Z., Chen, T., Nong, S., Jiang, T., Xu, W., Lu, W., Xiao, J., Zhuang, Y., et al.: Ui-zoomer: Uncertainty-driven adaptive zoom-in for gui grounding. arXiv preprint arXiv:2604.14113 (2026)

36. Tang, F., Gu, Z., Lu, Z., Liu, X., Shen, S., Meng, C., Wang, W., Zhang, W., Shen, Y., Lu, W., et al.: Gui-g <sup>2</sup>: Gaussian reward modeling for gui grounding. arXiv preprint arXiv:2507.15846 (2025)

37. Tang, F., Gu, Z., Lu, Z., Zhang, S., Zeng, Z., Shen, S., Meng, C., Yan, Y., Zhang, W., Shen, Y., et al.: Gui-sage: Enhancing gui automation with self-explanatory learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13007–13016 (2026)

38. Tang, F., Lu, Z., Zhang, B., Lu, W., Xiao, J., Zhuang, Y., Shen, Y.: Clawgui: A unified framework for training, evaluating, and deploying gui agents. arXiv preprint arXiv:2604.11784 (2026)

39. Tang, F., Shen, Y., Zhang, H., Chen, S., Hou, G., Zhang, W., Zhang, W., Song, K., Lu, W., Zhuang, Y.: Think twice, click once: Enhancing gui grounding via fast and slow systems. arXiv preprint arXiv:2503.06470 (2025)

40. Tang, F., Xu, H., Zhang, H., Chen, S., Wu, X., Shen, Y., Zhang, W., Hou, G., Tan, Z., Yan, Y., et al.: A survey on (m) llm-based gui agents. arXiv preprint arXiv:2504.13865 (2025)

41. Tang, J., Xia, Y., Wu, Y.F., Hu, Y., Chen, Y., Chen, Q.G., Xu, X., Wu, X., Lu, H., Ma, Y., et al.: Lpo: Towards accurate gui agent interaction via location preference optimization. arXiv preprint arXiv:2506.09373 (2025)

42. Wang, J., Xu, H., Ye, J., Yan, M., Shen, W., Zhang, J., Huang, F., Sang, J.: Mobileagent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158 (2024)

43. Wang, R., Li, H., Han, X., Zhang, Y., Baldwin, T.: Learning from failure: Integrating negative examples when fine-tuning large language models as agents. arXiv preprint arXiv:2402.11651 (2024)

44. Wei, X.S., Xu, H.Y., Zhang, F., Peng, Y., Zhou, W.: An embarrassingly simple approach to semi-supervised few-shot learning. Advances in Neural Information Processing Systems 35, 14489–14500 (2022)

45. Wu, Q., Cheng, K., Yang, R., Zhang, C., Yang, J., Jiang, H., Mu, J., Peng, B., Qiao, B., Tan, R., et al.: Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143 (2025)

46. Wu, Z., Wu, Z., Xu, F., Wang, Y., Sun, Q., Jia, C., Cheng, K., Ding, Z., Chen, L., Liang, P.P., et al.: Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218 (2024)

47. Xie, T., Deng, J., Li, X., Yang, J., Wu, H., Chen, J., Hu, W., Wang, X., Xu, Y., Wang, Z., et al.: Scaling computer-use grounding via user interface decomposition and synthesis. arXiv preprint arXiv:2505.13227 (2025)

48. Xu, H., Zhang, X., Liu, H., Wang, J., Zhu, Z., Zhou, S., Hu, X., Gao, F., Cao, J., Wang, Z., et al.: Mobile-agent-v3. 5: Multi-platform fundamental gui agents. arXiv preprint arXiv:2602.16855 (2026)

49. Xu, Y., Liu, X., Liu, X., Fu, J., Zhang, H., Jing, B., Zhang, S., Wang, Y., Zhao, W., Dong, Y.: Mobilerl: Online agentic reinforcement learning for mobile gui agents. arXiv preprint arXiv:2509.18119 (2025)

50. Xu, Y., Wang, Z., Wang, J., Lu, D., Xie, T., Saha, A., Sahoo, D., Yu, T., Xiong, C.: Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454 (2024)

51. Yan, H., Wang, J., Huang, X., Shen, Y., Meng, Z., Fan, Z., Tan, K., Gao, J., Shi, L., Yang, M., et al.: Step-gui technical report. arXiv preprint arXiv:2512.15431 (2025)

52. Yang, Y., Li, D., Dai, Y., Yang, Y., Luo, Z., Zhao, Z., Hu, Z., Huang, J., Saha, A., Chen, Z., et al.: Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791 (2025)

53. Yang, Y., Wang, Y., Li, D., Luo, Z., Chen, B., Huang, C., Li, J.: Aria-ui: Visual grounding for gui instructions. arXiv preprint arXiv:2412.16256 (2024)

54. Yuan, X., Zhang, J., Li, K., Cai, Z., Yao, L., Chen, J., Wang, E., Hou, Q., Chen, J., Jiang, P.T., et al.: Enhancing visual grounding for gui agents via self-evolutionary reinforcement learning. arXiv preprint arXiv:2505.12370 (2025)

55. Zhang, M., Xu, Z., Zhu, J., Dai, Q., Qiu, K., Yang, Y., Luo, C., Chen, T., Wagle, J., Franklin, T., et al.: Phi-ground tech report: Advancing perception in gui grounding. arXiv preprint arXiv:2507.23779 (2025)

56. Zhang, Y., Ni, B., Chen, X.S., Zhang, H.R., Rao, Y., Peng, H., Lu, Q., Hu, H., Guo, M.H., Hu, S.M.: Bee: A high-quality corpus and full-stack suite to unlock advanced fully open mllms. arXiv preprint arXiv:2510.13795 (2025)

57. Zhao, Z., Liu, Y., Liu, Y., Wang, H., Tian, L., Zhou, X., You, Y., Yu, Z., Yu, Y., Zhou, J.: Points-gui-g: Gui-grounding journey. arXiv preprint arXiv:2602.06391 (2026)

58. Zhou, H., Zhang, X., Tong, P., Zhang, J., Chen, L., Kong, Q., Cai, C., Liu, C., Wang, Y., Zhou, J., Hoi, S.: Mai-ui technical report: Real-world centric foundation gui agents (2025)

59. Zhou, S., Xu, F.F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Bisk, Y., Fried, D., Alon, U., et al.: Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854 (2023)

60. Zhou, Y., Liu, L., Gou, C.: Learning from observer gaze:zero-shot attention prediction oriented by human-object interaction recognition. In: CVPR (2024)

61. Zhou, Y., Dai, S., Wang, S., Zhou, K., Jia, Q., Xu, J.: Gui-g1: Understanding r1-zero-like training for visual grounding in gui agents. arXiv preprint arXiv:2505.15810 (2025)

62. Zhu, X., Xia, M., Wei, Z., Chen, W.L., Chen, D., Meng, Y.: The surprising efectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347 (2025)

63. Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., et al.: Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084 (2025)