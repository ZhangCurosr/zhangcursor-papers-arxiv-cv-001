# CAUSAL-EVC: BREAKING EMOTIONAL SPURIOUS CAUSALITY VIA SPATIOTEMPORAL GROUNDING AND COUNTERFACTUAL INTERVENTION

Cheng Ye<sup>1</sup>, Weidong Chen<sup>1</sup>, Peipei Song<sup>1</sup>, Zhendong Mao<sup>1</sup> <sup>1</sup>University of Science and Technology of China, Hefei chenweidong@ustc.edu.cn

## ABSTRACT

Emotional Video Captioning aims to generate factually accurate and emotionally empathetic descriptions. While recent methods have recognized the importance of visual causes to guide emotion perception and caption generation, they fundamentally rely on simple attention matching, which inevitably suffers from causal redundancy and spurious correlations in co-occurrence bias (e.g., misclassifying “sadness” as “joy” on a sunny beach), leading to severe shortcut learning from confusing backgrounds. Furthermore, existing evaluations fail to verify whether models have genuinely mastered causal reasoning or merely exploited background confounders. To address these limitations, we first construct EVC-CauseGround, a comprehensive benchmark with dense spatio-temporal causal annotations. Crucially, it introduces a carefully selected Causal-Faithfulness Subset to explicitly quantify genuine emotion-cause attribution. Second, we propose Causal-EVC, an emotion-grounding captioning framework, which introduces a Motion-guided Causal Spatiotemporal Localization module to precisely decouple causal triggers from background confounders. Besides, we introduce an Interpretable Sparse Emotion Routing module. By synthesizing counterfactual representations and formulating a novel counterfactual contrastive objective, we enforce the model to anchor its emotion predictions strictly on authentic causal triggers instead of confusing background. Extensive experiments show that Causal-EVC not only achieves the best performance on semantic metrics but also exhibits significant advantages in the causal-faithfulness subset, which demonstrates that our model could mine emotional cues from genuine visual causes and mitigate co-occurrence bias for interpretable multimodal emotion understanding.<sup>1</sup>

## 1 INTRODUCTION

Emotional Video Captioning (EVC) aims to generate factually accurate and emotionally empathetic descriptions for videos (El Assadi et al., 2026; Deng et al., 2026; Zhang et al., 2026). Unlike standard video captioning (Ye et al., 2025a; Li et al., 2026b; Xu et al., 2026; Chen et al., 2026), EVC represents a fundamentally more complex endeavor. It requires models to not only cross-modally align factual visual content with natural language, but also accurately perceive implicit affective cues and seamlessly synthesize them into human-like narratives.

Despite recent progress, mainstream EVC methods (Wang et al.; Song et al., 2022; Xu et al., 2024) predominantly map global video representations to predefined emotion spaces via attention patterns. However, this paradigm overlooks the intrinsic nature of visual emotions, which are typically triggered by fine-grained motivational causes rather than global backgrounds. While pioneering efforts like MM-ECPE (Ye et al., 2025b) attempted to explicitly extract emotion-cause pairs, we argue that current cause attribution methods still suffer from two fundamental limitations: 1) Causal Redundancy under Consistent Backgrounds: As shown in Fig. 1(a), when the contextual cofounder semantically aligns with the true emotion, the background inherently leaks the emotion and generates the causal redundancy. Because MM-ECPE merely maximizes positive mutual information, models effortlessly exploit these environmental confounders to achieve high prediction accuracy while ignoring the authentic causal triggers, failing to penalize this shortcut learning and losing the genuine emotional reasoning capabilities. 2) Spurious Correlations under Inconsistent Backgrounds: Conversely, when the background conflicts with the true foreground trigger (e.g., a child crying on a sunny beach), traditional patterns are heavily hacked by dataset-level co-occurrence bias (e.g., “Sunny Beach → Joy”). Lacking causal intervention mechanisms, the model is easily misled by the dominant confounder and produces severe emotional hallucinations. Meanwhile, existing benchmarks only evaluate the quality of generated captions and the accuracy of the predicted emotions, which completely fails to quantify the true causal reasoning ability and prevent making lucky statistical guesses based on environmental leakage.

![](images/b16f8c592c41f3327a3c114349a019d55ceea0ce957ee69434caa4ef032df850.jpg)

![](images/ffe777b1b8431321b4cf869f4ce21a4c80d23ecaf79f5666471160f3fcb6e46e.jpg)  
Figure 1: Motivation. Traditional methods suffer from (a) Causal Redundancy and (b) Spurious Correlations, where models exploit shortcut learning and emotional hallucinations due to background confounders. (c) Paradigm Comparison between Causal-EVC and previous methods demonstrate the superiority for genuine emotion-cause reasoning of Causal-EVC.

To fundamentally resolve these dilemmas, we tackle the EVC challenge from both benchmark and model perspectives: 1) We construct EVC-CauseGround, the first benchmark dedicated to fine-grained emotion causality. We achieve this via two major enhancements: We first introduce dense spatiotemporal causal annotations, defining precise temporal spans and bounding-box tubes for visual causes. Crucially, we collect a pure Causal-Faithfulness Subset, where background leakage is strictly eliminated. Based on it, we design a novel metric ST-IoU to rigorously quantify genuine emotion attribution. 2) We propose Causal-EVC, a counterfactual causal grounding framework. Technically, we design a Motion-Guided Causal

i) Caption Generation  
![](images/351970ba64ed1a14a6479ffae9f4fd00faa20a4751fa1d982bb948fdfbcab1d5.jpg)

ii) Causal Reasoning  
![](images/ab899596ec8c4e9ae9f8c532c30a7a6df59d14cee7275f5ac515d5e9e5279e79.jpg)  
Figure 2: Performance Comparison on the caption generation and causal reasoning.

Spatiotemporal Localization module to guide spatial and temporal attention, generating an accurate causal mask. Guided by it, we physically sever the causal path by synthesizing counterfactual background representations and propose an Interpretable Sparse Emotion Routing module to deduce class-specific emotion probabilities. By enforcing a Counterfactual Contrastive Objective, we explicitly penalize background exploitation, ensuring emotions are reasoned strictly from foreground triggers. As shown in Fig. 2, our model achieves consistent improvements in both caption generation and causal reasoning. In short, our main contributions are summarized as follows:

• Benchmark: We construct EVC-CauseGround, a pioneering benchmark with dense spatiotemporal annotations to evaluate the causal attribution capabilities. Besides, we design a convincing metric ST-IoU to explicitly quantify the genuine emotion-cause reasoning ability.

• Framework: We propose Causal-EVC, a novel framework with motion-guided cause localization and counterfactual emotion deduction. Besides, we design several objective functions to constrain cause extraction and emotion mining, which breaks spurious correlations and compels the model to reason emotions from true triggers.

• Performance: Extensive experiments demonstrate the superiority of Causal-EVC especially in cause grounding, i.e., +17.3%/11.8% improvements on mIoU and emotion accuracy than zero-shot

![](images/e10aead5851029b7a71e6865e9c9a05aed1918af911da1afb24e2b36ecec1c44.jpg)  
Figure 3: The construction pipeline of the EVC-CauseGround benchmark, which is consist of (A) Data Source Collection, (B) Spatiotemporal Visual Cause Annotation and (C) Human-led Sample Check and Partition, respectively.

Qwen-2.5-VL-7B, which validates that our approach learns genuine emotional causal reasoning capability for better emotional descriptions.

## 2 EVC-CAUSEGROUND: A VISUAL-CAUSE GROUNDING EMOTIONAL DESCRIPTIVE BENCHMARK

In this paper, we construct EVC-CauseGround, a novel benchmark that extends conventional emotion descriptions with a fine-grained emotional cause annotations. Unlike traditional datasets that provide only holistic labels or coarse utterance-level annotations (Wang et al., 2022; Ma et al., 2024; Zhang et al., 2020), EVC-CauseGround is the first benchmark to unify fine-grained spatiotemporal causal tubes and a rigorously curated counterfactual subset<sup>2</sup>, establishing a comprehensive benchmark for authentic emotional cause reasoning.

Data Source Collection. We meticulously curate diverse, in-the-wild videos from EmVidCap(Wang et al.) and MAFW(Liu et al., 2022) datasets. Specifically, we discard videos shorter than 5 seconds, as too-short durations make fine-grained causal localization trivial. We finally collect 1,523 videos from EmVidCap and 477 videos from MAFW (averaging ∼100s), covering unconstrained scenarios from daily interactions to natural disasters. They are split into 1,500 training and 500 evaluation samples. All selected videos are natively equipped with ground-truth emotion labels and captions, serving as reliable anchors for our subsequent causal annotation.

Spatiotemporal Visual Cause Annotation. To avoid cost-prohibitive manual frame-by-frame drawing, we propose an efficient MLLM-led, tracker-assisted pipeline. We first uniformly sample T = 30 frames from each video and concatenate them into a $6 \times 5$ grid image with sequence identifiers (e.g., F00 to F29). We prompt an advanced MLLM (e.g., GPT-5.6 (Singh et al., 2025)) with this grid and the ground-truth emotion to deduce the causal temporal window $\left[ { t _ { s t a r t } } , { t _ { e n d } } \right]$ . To prevent lazy behaviors of MLLMs that include all frames, we constraint that the causal span must not exceed 30% of the total frames. For spatial annotation, we first compute inter-frame optical flow within $[ t _ { s t a r t } , t _ { e n d } ]$ to identify the peak motion frame, which is the climax of the causal action. By feeding only this high-resolution frame into the MLLM, we obtain a highly confident static bounding box, devoid of multi-frame interference. Utilizing the generated anchor box as an initial prompt, we leverage the zero-shot tracker SAM-3 (Carion et al., 2026) to bidirectionally track the causal object across the temporal span, yielding continuous and dense spatio-temporal causal tubes.

Human-led Sample Check and Partition. While the automated pipeline accelerates annotation, human verification is essential to eliminate hallucinations. Firstly, human annotators are required to check for redundant or missed frames at the boundaries of the temporal spans and spatial regions. Besides, since MLLMs fail to address the Contextual Leakage issue, we synthesize a counterfactual video for each sample by blacking out the verified causal tube. Then, annotators, blinded to the original labels, are asked to guess the emotion based solely on the masked background. If annotators can still correctly predict the emotion, the sample is flagged as Causal-Redundant (i.e., background confounders are too dominant). Conversely, the video is flagged as Causal-Faithfulness. Finally, we collect a Causal-Faithfulness Subset containing 176 videos, which empowers us to evaluate whether a model performs genuine causal reasoning or merely exploits contextual cofounders.

![](images/bd43534cfe308832fbeb2ff40c6a96c6dda2fb6d4216a672c0f80fc920b039f2.jpg)  
Figure 4: Overview of Causal-EVC. Given the video and emotion dictionary, we perform (A) Motion-Guided Causal Spatiotemporal Localization to extract visual cause masks, and (B) Interpretable Sparse Emotion Routing to mine visual emotion cues for cause-grounding captioning.

Evaluation Metrics. To effectively evaluate the performance of cause grounding, we design a cascaded metric Spatio-Temporal Grounding Accuracy (ST-IoU). Specifically, we first compute the 1D temporal IoU between the predicted $[ t _ { s t a r t } ^ { p r e d } , t _ { e n d } ^ { p r e d } ]$ and the ground-truth span. If (tIoU = 0), the overall score is 0. If $t I o U > 0 .$ , we calculate the 2D spatial IoU $( s I o U )$ for each selected frame $t _ { i } \in T _ { i n t e r e s t }$ . The final ST-IoU is formulated as:

$$
\mathrm { S T - I o U } = \mathrm { t I o U } \times \left( \frac { 1 } { | T _ { \mathrm { i n t e r s e c t } } | } \sum _ { } _ { t _ { i } \in T _ { \mathrm { i n t e r s e c t } } } \mathrm { s I o U } _ { i } \right) ,\tag{1}
$$

we report the mIoU, R@0.3, and R@0.5 to robustly measure causal localization performance.

## 3 CAUSAL-EVC: A CAUSE-GROUNDING VIDEO CAPTIONING FRAMEWORK

In this section, we describe the Causal-EVC framework for the cause-grounding video captioning task shown in Fig. 4. Given the input video and emotion category, we first encode them to obtain the semantic features. Following the setting of previous works (Song et al., 2023; Ye et al., 2024), we firstly down-sample the video to obtain a frame sequence $\{ f _ { 1 } , f _ { 2 } . . . , f _ { T } \}$ , where $T$ denotes the number of frames. Subsequently, we leverage the pre-trained vision encoder of Qwen-2.5-VL (Bai et al., 2025) to extract visual features:

$$
\mathcal { V } = \mathrm { V i s E n c o d e r } ( [ \mathrm { f } _ { 1 } , \mathrm { f } _ { 2 } , \dots , \mathrm { f } _ { \mathrm { T } } ] ) \in \mathbb { R } ^ { \mathrm { T } \times \mathrm { P } \times \mathrm { D } } ,\tag{2}
$$

where VideoEncoder denotes the visual feature projector and D is the feature dimension. $P$ is the number of patches since each frame is divided into $P$ patches. For the emotion category, we leverage a pre-defined psychology emotional vocabulary dictionary $\Omega = \{ w _ { i } \} _ { i = 1 } ^ { N _ { w } }$ , where $w _ { i }$ denotes the i-th emotion word, such as “happy”, and $N _ { w }$ denotes the number of emotion words. Then, we leverage the pre-trained Transformer-based emotion embedding, $i . e .$ , GloVe (Pennington et al., 2014), to encode these emotion words to $\mathcal { E } \in \mathbb { R } ^ { N _ { w } \times D }$ . The whole process can be formalized as:

$$
\mathcal { E } = \mathrm { E m o E n c o d e r } ( \Omega ) \in \mathbb { R } ^ { N _ { w } \times D } .\tag{3}
$$

These visual and emotional features contain rich semantic information and help to mine accurate emotional causes from video contents.

## 3.1 MOTION-GUIDED CAUSAL SPATIOTEMPORAL LOCALIZATION

Motion Prior Generation. In video streams, genuine emotional triggers are rarely isolated static objects or a specific token. Instead, they are inherently embedded in the dynamic action states and state transitions. Therefore, we propose to leverage action semantics to serve as a kinematic prior, explicitly collaborating causal localization across both temporal and spatial dimensions. To capture motion prior, we propose a pixel-level motion correlation calculation module, which quantifies motion information through calculating the similarity of all pixel positions between adjacent frames.

For the j-th patch of the t-th frame $v _ { t } ^ { j }$ , the motion information is quantified as $\begin{array} { r } { s _ { t } ^ { j } = 1 - \frac { v _ { t - 1 } ^ { j } \cdot v _ { t } ^ { j } } { | | v _ { t - 1 } ^ { j } | | | v _ { t } ^ { j } | | } , } \end{array}$ where $s _ { t } ^ { j } \in [ 0 , 2 ]$ denotes the degree of motion change. The lower the similarity of corresponding patches between adjacent frames, the greater the degree of motion change. Meanwhile, in order to eliminate biases caused by some unexpected disturbances $( i . e .$ ,camera shake or perspective change), we calculate the global motion baseline through the average of all pixels. The motion information comprehensively considers the global motion baseline and its own motion mask matrix. Finally, we leverage the motion information to generate motion priors:

$$
\overline { { s } } = \frac { 1 } { T \times P } \sum _ { i , j } s _ { t } ^ { j } , m _ { t } ^ { j } = \mathrm { R e L U } ( ( 1 - \lambda _ { m } ) s _ { t } ^ { j } + \lambda _ { m } \overline { { s } } ) ,\tag{4}
$$

where $\lambda _ { m }$ is a hyper-parameter to control the weight of the global motion baseline. By aggregating all $m _ { t } ^ { j }$ , we obtain an accurate motion prior matrix $M \in \mathbb { R } ^ { T \times P }$ . We decouple the motion prior into temporal and spatial dimensions to achieve coordinated guidance:

$$
M _ { t e m p } = \mathrm { S o f t m a x } ( \{ \sum _ { j = 1 } ^ { P } m _ { t } ^ { j } \} T _ { t = 1 } ) \in \mathbb { R } ^ { T } , ( M _ { s p a } ) _ { t } ^ { j } = \frac { m _ { t } ^ { j } } { \operatorname* { m a x } _ { k \in \{ 1 . . P \} } m _ { t } ^ { k } + \epsilon } ,\tag{5}
$$

$M _ { t e m p }$ indicates the peaks of the action and $M _ { s p a }$ represents moving regions in each frame. These constitute a powerful prior that guides two mutually synergistic perspectives on causal exploration.

Temporal Causal Aggregation. To achieve temporal causal aggregation, we introduce a motionenergy bias into the cross-attention mechanism to remove the impact of irrelevant temporal frames:

$$
Q _ { \tau } = W _ { q } \mathcal { E } , \quad K _ { \tau } = W _ { k } \mathcal { V } , \quad V _ { \tau } = W _ { v } \mathcal { V } ,\tag{6}
$$

$$
A _ { \tau } = \mathrm { S o f t m a x } \left( \frac { Q _ { \tau } K _ { \tau } ^ { T } } { \sqrt { d } } + \alpha \cdot \mathrm { D i a g } ( \log M _ { t e m p } ) \right) , \mathcal { F } _ { t e m p } = A _ { \tau } V _ { \tau } \in \mathbb { R } ^ { T \times P \times D } ,\tag{7}
$$

where $\alpha$ is the scaling parameter and Diag transforms the 1-D curve into a diagonal matrix. The bias term significantly amplifies the weight of frames representing the peak of the movement, thereby focusing more on the emotional climax associated with intense action.

Spatial Causal Localization. Subsequently, we further consider key regions of the emotional causes. We calculate cross-attention along different geometric axes to capture entity boundaries:

$$
Q _ { s } ^ { g } = W _ { s , q } ^ { g } \mathcal { E } , \quad K _ { s } ^ { g } = W _ { s , k } ^ { g } \mathcal { F } _ { t e m p } , \quad V _ { s } ^ { g } = W _ { s , v } ^ { g } \mathcal { F } _ { t e m p } ,\tag{8}
$$

$$
A _ { s } ^ { g } = \mathrm { S o f t m a x } ( Q _ { s } ^ { g } K _ { s } ^ { g T } ) , { \mathcal { O } } _ { s } = A _ { t } ^ { y } ( A _ { t } ^ { x } V _ { s } ^ { g } ) ^ { T } \in \mathbb { R } ^ { T \times P \times D } ,\tag{9}
$$

where $g \in \{ x , y \}$ represents the horizontal and vertical axes. $\mathcal { O } _ { s }$ highlights all salient emotionrelated regions in the scene. Finally, we utilize $M _ { s p a }$ to modulate $\mathcal { O } _ { s }$ and obtain causal features and their corresponding masks:

$$
\mathcal { F } _ { c } = \mathcal { O } _ { s } \odot M _ { s p a } \in \mathbb { R } ^ { T \times P \times D } , M _ { c } = \sigma ( \mathrm { C o n v 3 D } ( \mathcal { F } _ { c } ) ) \in [ 0 , 1 ] ^ { T \times P \times 1 } ,\tag{10}
$$

where σ is the Sigmoid activation function. To precisely ground the spatiotemporal labels, we design two MLP heads and objective functions to achieve powerful supervision. Instead of predicting frame-wise and pixel-wise probabilities, we regress the normalized temporal indexes and spatial coordinates. First, we aggregate $\mathcal { F } _ { c }$ to obtain a spatial visual vector $f _ { s } ^ { \mathrm { ~ \tiny ~ \cdot ~ } } \in \mathbb { R } ^ { T \times D }$ and a compact visual vector $f _ { g } \in \mathbb { R } ^ { D }$

$$
f _ { s } = \mathrm { A v g P o o l } _ { P } ( \mathcal { F } _ { c } ) , f _ { g } = \mathrm { M a x P o o l } _ { T } \mathrm { ( C o n v 1 D } ( f _ { s } ) ) ,\tag{11}
$$

Then, temporal and spatial heads are used to predict the temporal indexes and spatial coordinates:

$$
\big ( [ \hat { t } _ { s t a r t } , \hat { t } _ { e n d } ] , [ \hat { x } _ { m i n } ^ { ( t ) } , \hat { y } _ { m i n } ^ { ( t ) } , \hat { x } _ { m a x } ^ { ( t ) } , \hat { y } _ { m a x } ^ { ( t ) } ] \big ) = \big ( \sigma ( \mathbf { M } \mathbf { L } \mathbf { P } _ { t e m p } ( f _ { g } ) \big ) , \sigma ( \mathbf { M } \mathbf { L } \mathbf { P } _ { s p a } ( f _ { s , t } ) ) \big ) ,\tag{12}
$$

where $\sigma$ is the Sigmoid activation function and MLP is the multi-layer projection.

Causal Grounding Objective. During training, given the ground-truth temporal span $[ t _ { s t a r t } ^ { g t } , t _ { e n d } ^ { g t } ]$ we apply the SmoothL1 loss (Huber, 1992) to robustly supervise the boundary regression:

$$
\mathcal { L } _ { t e m p } = \mathrm { S m o o t h L 1 } ( \hat { p } _ { s t a r t } , p _ { s t a r t } ^ { g t } ) + \mathrm { S m o o t h L 1 } ( \hat { p } _ { e n d } , p _ { e n d } ^ { g t } ) ,\tag{13}
$$

For the supervision of spatial regions, denoting the predicted and ground-truth bounding box as $B _ { t } ^ { p r e d }$ and $B _ { t } ^ { g t }$ , we employ a combination of SmoothL1 loss and Generalized IoU loss (Hamid et al., 2019), which effectively penalizes scale mismatches and prevents gradient vanishing when there is no overlap between boxes:

$$
\mathcal { L } _ { G I o U } ( B _ { t } ^ { p r e d } , B _ { t } ^ { g t } ) = 1 - \left( \mathrm { I o U } ( B _ { t } ^ { p r e d } , B _ { t } ^ { g t } ) - \frac { \mathrm { A r e a } ( C _ { t } ) - \mathrm { A r e a } ( B _ { t } ^ { p r e d } \cup B _ { t } ^ { g t } ) } { \mathrm { A r e a } ( C _ { t } ) } \right) ,\tag{14}
$$

$$
\mathcal { L } _ { s p a } = \frac { 1 } { \sum _ { t = 1 } ^ { T } \mathbb { I } _ { t } ^ { g t } } \sum _ { t = 1 } ^ { T } \mathbb { I } _ { t } ^ { g t } \cdot \Big [ \lambda _ { L 1 } \mathcal { L } _ { L 1 } \big ( B _ { t } ^ { p r e d } , B _ { t } ^ { g t } \big ) + \lambda _ { g i o u } \mathcal { L } _ { G I o U } \big ( B _ { t } ^ { p r e d } , B _ { t } ^ { g t } \big ) \Big ] ,\tag{15}
$$

where $C _ { t }$ is the minimum enclosing box containing both $B _ { t } ^ { p r e d }$ and $B _ { t } ^ { g t } , \lambda _ { L 1 }$ and $\lambda _ { g i o u }$ are balancing weights. $\mathbb { I } ^ { g t } \in \{ 0 , 1 \} ^ { T }$ is a temporal indicator mask to ensure only calculating the spatial loss on frames where the genuine causal action occurs.

## 3.2 INTERPRETABLE SPARSE EMOTION ROUTING

Emotion Prototype Augmentation. Unlike objective elements, the semantic information of emotion words is characterized by complexity and subjectivity. Relying on a unified text encoder for emotional mining is insufficient. Therefore, we introduce the VAD lexicon (Mohammad, 2018) $R = \{ ( v _ { i } , a _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N _ { w } } \in \mathbb { R } ^ { N _ { w } \times 3 }$ , where each entry is normalized to [−1, 1] (and out-of-lexicon words are set to 0), to obtain an interpretable emotion-augmented representation. Given the emotion vocabulary $\mathcal { E }$ and the VAD lexicon, we stack these triplets to obtain the emotion prototype features:

$$
\begin{array} { r } { \hat { \mathcal { E } } = \mathrm { L a y e r N o r m } ( \mathcal { E } \oplus W _ { e } R ) \in \mathbb { R } ^ { N _ { w } \times D } , } \end{array}\tag{16}
$$

The $\hat { \mathcal { E } }$ contains both textual semantic and psychological attributes, enhancing the interpretability and facilitating the extraction of emotional semantics from visual information.

Top-K Sparse Emotion Routing. Affective semantics are characterized by significant vagueness and ambiguity. Traditional methods are highly susceptible to interference from irrelevant emotional information. Therefore, we design a sparse emotion routing mechanism. First, instead of using redundant visual features, we employ the cause mask $M _ { c }$ to filter and obtain visual causal features $\mathcal { C } = V \odot M _ { c } ,$ , which are then used to query emotion prototypes. Subsequently, to prevent the traditional attention mechanism from capturing ambiguous emotional semantics, we enforce matching only the top-k emotion categories. Specifically, we first compute the similarity matrix $A ^ { e }$ between $\mathcal { C }$ and ${ \hat { \mathcal { E } } } .$ Subsequently, for each visual token, we retain only the top-K values and set the rest to $- \infty$ , thereby obtaining the sparse attention matrix $\hat { A } _ { e }$ to route the emotion extraction:

$$
\hat { A } _ { i , j } ^ { e } = \left\{ \begin{array} { l l } { \frac { \exp ( A _ { i , j } ^ { e } ) } { \sum _ { m \in \mathrm { I o p } \cdot \mathrm { K } ( A _ { i } ^ { e } ) } \exp ( A _ { i , m } ) } , } & { \mathrm { i f ~ } j \in \mathrm { T o p } \cdot \mathrm { K } ( A _ { i } ^ { e } ) } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. , \mathcal { H } _ { t r u e } = \hat { A } ^ { e } ( \hat { \mathcal { E } } W _ { h } ) \in \mathbb { R } ^ { T \times D } ,\tag{17}
$$

$\mathcal { H } _ { t r u e }$ denotes the visual hidden states related to the emotions. Finally, we leverage a set of learnable category query vectors $\mathbf { q } _ { e m o } \in \mathbb { R } ^ { N _ { w } \times D }$ to mine emotions:

$$
\begin{array} { r } { \hat { \mathcal { E } } _ { t r u e } = \mathbf { M } \mathbf { H } \mathbf { C } \mathbf { A } ( \mathbf { Q } \mathbf { u } \mathrm { e r y } = \mathbf { q } _ { e m o } , \mathbf { K } \mathbf { e y } = \mathcal { H } _ { t r u e } , \mathbf { V } \mathbf { a } \mathbf { l } \mathbf { u } \mathrm { e } = \mathcal { H } _ { t r u e } ) \in \mathbb { R } ^ { N _ { w } \times D } , } \end{array}\tag{18}
$$

besides, to achieve the accurate supervision for emotion mining, we introduce a counterfactual causal alignment module. Specifically, we replace the visual causal features C in the aforementioned process with counterfactual visual features $\hat { \mathcal { C } } = V \odot ( 1 - M _ { c } )$ to obtain counterfactual emotion features $\hat { \mathcal { E } } _ { f a k e }$ . Then, we calculate the cosine similarity between the ground-truth and the predicted emotion features to obtain the emotion logits:

$$
z _ { t r u e } = \mathrm { C o s } ( \mathcal { E } _ { g t } , \hat { \mathcal { E } } _ { t r u e } ) \in \mathbb { R } ^ { N _ { w } } , z _ { f a k e } = \mathrm { C o s } ( \mathcal { E } _ { g t } , \hat { \mathcal { E } } _ { f a k e } ) \in \mathbb { R } ^ { N _ { w } } ,\tag{19}
$$

Table 1: Performance for emotional video captioning task on two benchmarks. The best results are highlighted in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Emotion</td><td colspan="5"></td><td colspan="2">Hybrid</td></tr><tr><td> $\operatorname { A c c } _ { s w } \uparrow$ </td><td> $\operatorname { A c c } _ { c } \uparrow$ </td><td>BLEU-1↑</td><td>BLEU-4↑</td><td>METEOR↑</td><td>ROUGE↑</td><td>CIDEr↑</td><td>BFS↑</td><td>CFS↑</td></tr><tr><td colspan="10">EVC-MSVD</td></tr><tr><td>SA-LSTM (2018)</td><td>68.8</td><td>67.2</td><td>80.7</td><td>45.5</td><td>33.0</td><td>68.2</td><td>72.1</td><td>59.0</td><td>71.3</td></tr><tr><td>FT ()</td><td>69.4</td><td>67.1</td><td>77.2</td><td>36.3</td><td>29.0</td><td>63.4</td><td>62.5</td><td>52.5</td><td>63.7</td></tr><tr><td>SGN (2021)</td><td>73.9</td><td>73.1</td><td>77.5</td><td>41.1</td><td>30.6</td><td>63.6</td><td>71.0</td><td>56.4</td><td>71.5</td></tr><tr><td>CANet (2022)</td><td>78.7</td><td>76.8</td><td>78.5</td><td>41.8</td><td>30.8</td><td>65.7</td><td>74.4</td><td>57.9</td><td>75.1</td></tr><tr><td>VEIN (2024)</td><td>82.7</td><td>82.1</td><td>82.0</td><td>45.9</td><td>33.0</td><td>69.0</td><td>79.6</td><td>62.4</td><td>80.2</td></tr><tr><td>EPAN (2023)</td><td>84.1</td><td>82.8</td><td>82.5</td><td>46.2</td><td>34.4</td><td>69.8</td><td>80.6</td><td>63.1</td><td>81.1</td></tr><tr><td>DCGN (2024)</td><td>86.5</td><td>85.7</td><td>84.5</td><td>48.7</td><td>35.7</td><td>71.0</td><td>85.2</td><td>65.7</td><td>86.6</td></tr><tr><td>GLM-EER (2026)</td><td>88.2</td><td>86.6</td><td>92.2</td><td>68.3</td><td>43.8</td><td>80.1</td><td>153.9</td><td>78.8</td><td>140.6</td></tr><tr><td>MM-ECPE (2025b)</td><td>90.4</td><td>89.1</td><td>96.9</td><td>71.4</td><td>45.5</td><td>83.1</td><td>168.3</td><td>81.8</td><td>152.6</td></tr><tr><td>SAGML (2026)</td><td>93.5</td><td>92.3</td><td>93.6</td><td>53.8</td><td>40.8</td><td>77.2</td><td>96.1</td><td>71.5</td><td>95.5</td></tr><tr><td>Causal-EVC(Ours)</td><td>94.2</td><td>92.9</td><td>97.4</td><td>71.2</td><td>46.2</td><td>84.6</td><td>169.8</td><td>82.7</td><td>154.6</td></tr><tr><td colspan="10">EVC-VE</td></tr><tr><td>SA-LSTM (2018)</td><td>48.6</td><td>47.1</td><td>71.0</td><td>22.5</td><td>19.6</td><td>40.7</td><td>30.2</td><td>38.9</td><td>33.7</td></tr><tr><td>CANet (2022)</td><td>41.9</td><td>39.7</td><td>66.9</td><td>19.3</td><td>18.2</td><td>37.9</td><td>23.3</td><td>33.9</td><td>26.8</td></tr><tr><td>VEIN (2024)</td><td>57.4</td><td>56.8</td><td>71.6</td><td>26.3</td><td>20.9</td><td>41.7</td><td>33.4</td><td>43.0</td><td>39.2</td></tr><tr><td>EPAN (2023)</td><td>63.8</td><td>62.3</td><td>73.6</td><td>27.0</td><td>21.2</td><td>42.3</td><td>34.7</td><td>45.0</td><td>40.4</td></tr><tr><td>DCGN (2024)</td><td>71.0</td><td>69.4</td><td>74.5</td><td>28.1</td><td>23.4</td><td>47.7</td><td>41.5</td><td>47.3</td><td>46.9</td></tr><tr><td>HEART (2026)</td><td>55.8</td><td>54.2</td><td>71.5</td><td>22.7</td><td>19.7</td><td>41.2</td><td>28.8</td><td>40.3</td><td>34.0</td></tr><tr><td>MM-ECPE (2025b)</td><td>73.4</td><td>72.3</td><td>76.8</td><td>28.9</td><td>24.7</td><td>49.5</td><td>65.2</td><td>49.2</td><td>66.7</td></tr><tr><td>SAGML (2026)</td><td>80.2</td><td>79.6</td><td>82.7</td><td>31.8</td><td>26.9</td><td>52.7</td><td>71.8</td><td>53.8</td><td>73.4</td></tr><tr><td>Causal-EVC(Ours)</td><td>82.8</td><td>81.5</td><td>85.1</td><td>32.4</td><td>27.7</td><td>54.6</td><td>74.0</td><td>55.3</td><td>75.6</td></tr></table>

where $\mathcal { E } _ { g t }$ denotes the prototype embedding of the ground-truth emotion labels. Then, we employ an NCE loss (Fu et al., 2024; Radford et al., 2021; Liu et al., 2026; Li et al., 2026a) on $\left( z _ { t r u e } , z _ { f a k e } \right)$ pair to guide the model to learn accurate emotion features from genuine causal features, rather than relying on shortcut learning from spurious visual backgrounds.

$$
\mathcal { L } _ { c f } = - \log \frac { \exp ( z _ { t r u e } ^ { ( y ) } / \tau _ { c f } ) } { \exp ( z _ { t r u e } ^ { ( y ) } / \tau _ { c f } ) + \exp ( z _ { f a k e } ^ { ( y ) } / \tau _ { c f } ) } ,\tag{20}
$$

where $\tau _ { c f }$ is the temperature parameter. After obtaining sufficient semantic information including video, cause, and emotion features, we concatenate them to generate a unified multi-modal semantic representation $M = \lvert \mathcal { V } ; \mathcal { C } ; \mathcal { E } _ { t r u e } \rvert \in \mathbb { R } ^ { ( T \times 2 P + N _ { w } ) \times D }$ . Finally, we send them into the decoder with Qwen-2.5-VL model (Bai et al., 2025) to generate the emotional captions.

## 4 RESULTS AND DISCUSSION

## 4.1 MAIN RESULTS

As shown in Table 1, Causal-EVC consistently achieves the best performance across almost all metrics, i.e., improving +4.2% with $A c c _ { s w }$ than MM-ECPE and +15.7% with BFS than SAGML on EVC-MSVD. More notably, on the larger EVC-VE benchmark, Causal-EVC also outperforms SAGML, i.e., +2.4%/+3.0% improvements with $A c c _ { c } / \mathrm { B L E U } { - 4 }$ These prove that Causal-EVC effectively

Table 2: Results on EVC-CauseGround.
<table><tr><td>Methods</td><td> $\overline { { m I o U } }$ </td><td>R@0.3</td><td>R@0.5</td><td> $\overline { { \mathrm { A c c } _ { \mathrm { s w } } } }$ </td></tr><tr><td>MM-ECPE (2025b)</td><td>18.6</td><td>28.7</td><td>12.6</td><td>53.1</td></tr><tr><td>Momentor (2024) VTimeLLM (2024)</td><td>21.5 23.7</td><td>35.8 42.1</td><td>20.4 22.0</td><td>58.5 61.4</td></tr><tr><td>VideoChat-T (2025) Qwen2.5-VL-7B (2025)</td><td>37.6 39.8</td><td>61.9</td><td>40.8</td><td>78.4 80.8</td></tr><tr><td>Gemini-2.5-Flash (2025)</td><td>30.7</td><td>62.5 50.9</td><td>41.3 27.1</td><td>67.0</td></tr><tr><td>GPT-4o (2024)</td><td>35.1</td><td>55.5</td><td>33.2</td><td>73.9</td></tr><tr><td>Ours</td><td>46.7</td><td>70.4</td><td>49.2</td><td>90.3</td></tr></table>

achieves better emotional description generation. Besides, Table 2 reports the cause-grounding performance on EVC-CauseGround. Causal-EVC demonstrates substantial dominance across all metrics. Compared to specialized visual grounding models, our model leads +67.2%/13.7% than VTimeLLM (Huang et al., 2024)/VideoChat-T (Zeng et al., 2025) on R@0.3. While these models excel at objective localization, they are prone to hallucinations when it comes to emotional visual attribution due to their lack of specific emotion understanding components. Besides, compared to general MLLMs, our model leads +52.1%/33.0% than Gemini-2.5-Flash (Comanici et al., 2025)/GPT-4o (Hurst et al., 2024) on mIoU. Despite the powerful comprehension capabilities of these models, the dual requirements of emotional understanding and visual grounding exacerbate the frequency of hallucinations, which demonstrates that Causal-EVC improves the authentic and robust emotion-cause reasoning and alleviates the co-occurrence bias.

Table 3: The ablation study for dense MCSL.
<table><tr><td>Motion</td><td>Components Temporal</td><td>Spatial</td><td> $\operatorname { A c c } _ { s w }$ </td><td>BLEU-4</td><td>CIDEr</td><td>CFS</td></tr><tr><td rowspan="4">×</td><td>×</td><td>×</td><td>92.9</td><td>67.8</td><td>154.4</td><td>142.0</td></tr><tr><td>√</td><td>×</td><td>89.4</td><td>68.6</td><td>156.9</td><td>143.2</td></tr><tr><td>×</td><td>√</td><td>91.2</td><td>69.7</td><td>158.5</td><td>144.9</td></tr><tr><td>√</td><td>√</td><td>90.8</td><td>70.3</td><td>160.2</td><td>146.2</td></tr><tr><td rowspan="4">√</td><td>×</td><td>×</td><td>90.5</td><td>67.2</td><td>152.8</td><td>140.2</td></tr><tr><td>√</td><td>×</td><td>93.4</td><td>70.1</td><td>160.6</td><td>147.0</td></tr><tr><td>×</td><td>√</td><td>93.7</td><td>70.6</td><td>163.9</td><td>149.7</td></tr><tr><td>√</td><td>√</td><td>94.2</td><td>71.2</td><td>169.8</td><td>154.6</td></tr></table>

Table 4: The ablation study for proposed losses.
<table><tr><td colspan="3">Components</td><td rowspan="2"> $\operatorname { A c c } _ { s w }$ </td><td rowspan="2">BLEU-4</td><td rowspan="2">CIDEr</td><td rowspan="2">CFS</td></tr><tr><td> $\mathcal { L } _ { c f }$ </td><td>Ltemp</td><td> $\mathcal { L } _ { s p a }$ </td></tr><tr><td rowspan="4">×</td><td>×</td><td>×</td><td>87.3</td><td>66.7</td><td>153.2</td><td>139.9</td></tr><tr><td>√</td><td>×</td><td>87.8</td><td>68.5</td><td>157.4</td><td>143.4</td></tr><tr><td>×</td><td>√</td><td>87.4</td><td>68.9</td><td>158.3</td><td>144.0</td></tr><tr><td>√</td><td>√</td><td>88.4</td><td>70.1</td><td>164.0</td><td>148.8</td></tr><tr><td rowspan="4">√</td><td>×</td><td>×</td><td>90.8</td><td>67.2</td><td>155.9</td><td>142.8</td></tr><tr><td>√</td><td>×</td><td>92.1</td><td>69.0</td><td>159.3</td><td>145.8</td></tr><tr><td>×</td><td>√</td><td>91.6</td><td>69.4</td><td>160.2</td><td>146.4</td></tr><tr><td>√</td><td>√</td><td>94.2</td><td>71.2</td><td>169.8</td><td>154.6</td></tr></table>

## 4.2 ABLATION STUDIES

Discussion on proposed modules. As shown in Table 5, we make an ablation study on proposed modules. We observe that each module could improve its performance independently. Specifically, MCSL effectively localizes the cause cues, which helps to generate more accurate captions for crucial emotion-related events,

Table 5: The ablation study for proposed modules.
<table><tr><td colspan="2">Components</td><td rowspan="2"> $\operatorname { A c c } _ { s w }$ </td><td rowspan="2">BLEU-4</td><td rowspan="2">CIDEr</td><td rowspan="2">CFS</td></tr><tr><td>MCSL</td><td>ISER</td></tr><tr><td>×</td><td>X</td><td>82.1</td><td>65.3</td><td>150.9</td><td>137.0</td></tr><tr><td>√</td><td>×</td><td>85.7</td><td>70.2</td><td>165.4</td><td>149.4</td></tr><tr><td>×</td><td>√</td><td>92.9</td><td>67.8</td><td>154.4</td><td>142.0</td></tr><tr><td>√</td><td>√</td><td>94.2</td><td>71.2</td><td>169.8</td><td>154.6</td></tr></table>

resulting in obvious improvements on semantic metrics, i.e., +7.5%/+9.1% improvements on BLEU-4/CFS. Besides, ISER effectively utilizes causal masks and counterfactual interventions to achieve accurate emotion cue extraction, resulting in obvious improvements in emotion accuracy, i.e., +13.2% improvements on $\operatorname { A c c } _ { s w }$ . Finally, we combine two modules to achieve the best performance, i.e., +14.7%/+12.8% improvements on $\operatorname { A c c } _ { s w } / \mathbf { C F S }$

Discussion on MCSL. We further discuss the dense components in MCSL module in Table 3. We first observe that solely relying on motion dynamics to generate the causal mask leads to a performance degradation. We analyze that pixel-wise differences lack semantic discriminability and are easily corrupted by environmental disturbances, such as swaying leaves. However, when motion information is used as a kinematic prior to guide both spatial and temporal localization, the model achieves a substantial performance surge. This result strongly validates that raw motion dynamics are inherently suited as guiding priors rather than standalone causes, while the genuine causal triggers fundamentally reside in the structural semantic entities executing these dynamic actions.

Discussion on Top-K Routing. As shown in Table 6, we discuss the performance of different Top-K routing. First, we observe that an excessively tight constraint $( \mathrm { i . e . , } K = 1 )$ leads to a performance collapse. This degradation occurs because forcing a single-prototype selection severely undermines the model’s fault tolerance and fails to accommodate the emotional

Table 6: Results of different Top-K routing.
<table><tr><td>Top-K</td><td> $\operatorname { A c c } _ { s w }$ </td><td>BLEU-4</td><td>CIDEr</td><td>CFS</td></tr><tr><td>1</td><td>49.6</td><td>51.5</td><td>115.9</td><td>102.5</td></tr><tr><td>5</td><td>88.6</td><td>67.4</td><td>149.8</td><td>137.4</td></tr><tr><td>10</td><td>94.2</td><td>71.2</td><td>169.8</td><td>154.6</td></tr><tr><td>20</td><td>92.9</td><td>70.1</td><td>161.8</td><td>147.8</td></tr><tr><td>50</td><td>92.2</td><td>68.9</td><td>156.3</td><td>143.3</td></tr></table>

diversity nature. Conversely, scaling K to an overly large value (e.g., $K = 5 0 )$ causes trivial performance. An unconstrained routing introduces redundant and irrelevant emotion prototypes, which compromises the sparsity bottleneck and dilutes the authentic causal emotion features with background semantic noise. Overall, we set $K = 1 0$ as the default configuration, which strikes the optimal trade-off between affective expressiveness and noise suppression.

Discussion on objective functions. Besides, we conduct an ablation study to discuss the effectiveness of three specialized objective functions $\mathcal { L } _ { t e m p } , \mathcal { L } _ { s p a }$ , and $\mathcal { L } _ { c f }$ . As shown in Table 4, applying $\mathcal { L } _ { c f }$ brings significant improvements on emotion accuracy. It isolates spurious background information by counterfactual contrastive learning, ensuring that emotion predictions stem from genuine causal triggers. Besides, applying $\mathcal { L } _ { t e m p }$ and $\mathcal { L } _ { s p a }$ brings remarkable improvements on semantic metrics. They leverage ground-truth spatiotemporal tube annotations to supervise cause grounding, ensuring the accuracy of key cause-related words in emotional captions. Finally, we employ these objective functions in combination to achieve end-to-end training, ensuring the accuracy in both emotional description and causal localization.

## 4.3 QUALITATIVE RESULTS

![](images/b4533caffcbe5768ed6c465e63a052d192367965d1d2c30477d61b158e25bc1d.jpg)  
Figure 5: Comparison of caption generation between MM-ECPE and our Causal-EVC.

Case Study of Causal Localization. We first make a comparison of caption generation between MM-ECPE and Causal-EVC shown in Fig. 5. MM-ECPE generates incorrect emotions ‘annoyed’ and ‘expected’, due to co-occurrence bias for ‘throwing stones’ and insufficient localization for ‘sicken bread’. Besides, we display the effect of cause localization in Fig. 6. We first observe from the Temporal Energy Curve that the baseline is easily misled by the early preparation of the ice cream, peaking

![](images/84b22d208bac8561c3f97b4afcd7dc2fe7dc29ed0f518cad2c469a62f218c4e0.jpg)  
Figure 6: Case study for cause localization.

prematurely. Instead, our curve surges precisely when the girl experiences a sudden sensory discomfort. Consequently, our Causal-EVC predicts a tighter temporal span that aligns with GT labels than baselines. More crucially, from the spatial heatmaps, we observe that our model successfully eliminates the spurious correlation. Tricked by the prior bias “ice-cream→happy“, the baseline focuses predominantly on the ice cream region, mistakenly inferring a “Happy” emotion while entirely overlooking the child’s adverse reaction. In contrast, Causal-EVC penalizes the confounder and concentrates on the disgust expressions of the girl, inferring a “Disgust“ emotion successfully.

## t-SNE Visualization of Cause Rep-

resentation. We visualize the t-SNE distributions of C across emotion spaces alongside C<sup>ˆ</sup>. As shown in Fig. 7, we first observe that compared MM-ECPE, our Causal-EVC achieves better semantic clustering. This stems from the MCSL module, which synergistically leverages motion priors to steer cause localization under strict coordinate supervision.

![](images/922d71f2ae33cb7f9c72c7063c4df7ca8424960b51b0203906f4f31e6d01bb22.jpg)  
Figure 7: t-SNE visualization of cause representation.

More crucially, In MM-ECPE, C<sup>ˆ</sup> are systematically entangled within the respective emotion clusters. This confirms our findings that conventional in-batch contrastive learning on (C, E) suffers from shortcut learning, mistakenly encoding background contexts as emotional evidence. Instead, under our Causal-EVC, C<sup>ˆ</sup> are uniformly distributed across the entire latent space. This validates that our $\mathcal { L } _ { c f }$ on (C, C<sup>ˆ</sup>) pairs successfully severs the spurious learning from environmental contexts, strictly compelling to deduce affective semantics exclusively from authentic cause triggers.

## 5 CONCLUSION

In this paper, we propose Causal-EVC, a cause-grounding captioning framework. To address causal redundancy and spurious correlations, we establish EVC-CauseGround, a pioneering benchmark featuring dense spatio-temporal causal tubes, a curated Causal-Faithfulness Subset, and a convinced metric. Technically, Causal-EVC introduces a Motion-guided Causal Localization module with threshold-free heads to decouple foreground causal triggers into a 3D mask. Furthermore, we de sign an Interpretable Sparse Emotion Routing module to deduce class-specific emotion probabilities via sparse prototype routing, enforcing a Counterfactual Contrastive Objective on synthesized background representations to strictly penalize shortcut learning. Extensive experiments demonstrate that Causal-EVC achieves the best performance in both text generation and causal faithfulness.

## AI USE STATEMENT

In this work, we used generative AI tools for polishing the language of the manuscript, writing and debugging portions of the experiment code, and some steps in constructing the dataset. We have not used generative AI tools for the motivation or the production of the reported results and figures. These are carried out directly by the authors. All AI-assisted code and data was tested and verified by running the experiments reported in this paper, and the resulting claims were checked against the run outputs by the authors. We take responsibility for the final content of this work, including the data, text, claims, and artifacts produced with the aid of generative AI.

## REFERENCES

Jiaming An, Zixiang Ding, Ke Li, and Rui Xia. Global-view and speaker-aware emotion cause extraction in conversations. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:3814–3823, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings ofthe acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pp. 65–72, 2005.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International Conference on Learning Representations, volume 2026, pp. 138846–138923, 2026.

David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.

Weidong Chen, Cheng Ye, Peipei Song, Lei Zhang, Yongdong Zhang, and Zhendong Mao. Subjective-objective emotion correlated generation network for subjective video captioning. IEEE Transactions on Image Processing, 2026.

Xinhao Chen, Chong Yang, Changzhi Sun, Man Lan, and Aimin Zhou. From coarse to fine: A distillation method for fine-grained emotion-causal span pair extraction in conversation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 17790–17798, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yicheng Deng, Hideaki Hayashi, and Hajime Nagahara. From pixels to semantics: Unified facial action representation learning for micro-expression analysis. In International Conference on Learning Representations, volume 2026, pp. 105640–105659, 2026.

Adnan El Assadi, Isaac Chung, Roman Solomatin, Niklas Muennighoff, and Kenneth Enevoldsen. Hume: Measuring the human-model performance gap in text embedding tasks. In International Conference on Learning Representations, volume 2026, pp. 71873–71904, 2026.

Zheren Fu, Lei Zhang, Hou Xia, and Zhendong Mao. Linguistic-aware patch slimming framework for fine-grained cross-modal alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26307–26316, 2024.

Junyu Gao, Mengyuan Chen, and Changsheng Xu. Vectorized evidential learning for weaklysupervised temporal action localization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(12):15949 – 15963, 2023.

Rezatofighi Hamid, Tsoi Nathan, Gwak JunYoung, Sadeghian Amir, Reid Ian, and Savarese Silvio. Generalized intersection over union: a metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 658–666, 2019.

Tingting Han, Yuxuan Gong, Sicheng Zhao, Min Tan, Zhou Yu, and Hongxun Yao. Heart: Emotionally grounded video captioning via hierarchical emotion-aligned representation. IEEE Trans actions on Affective Computing, 2026.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https: //arxiv.org/abs/2106.09685.

Guimin Hu, Yi Zhao, and Guangming Lu. Emotion prediction oriented method with multiple supervisions for emotion-cause pair extraction. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:1141–1152, 2023.

Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14271–14280. IEEE, 2024.

Xiaoyu Huang, Weidong Chen, Bo Hu, and Zhendong Mao. Graph mixture of experts and memoryaugmented routers for multivariate time series anomaly detection. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pp. 17476–17484, 2025.

Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pp. 492–518. Springer, 1992.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Yu-Gang Jiang, Baohan Xu, and Xiangyang Xue. Predicting emotions in user-generated videos. In Proceedings ofthe AAAI conference on artificial intelligence, volume 28, 2014.

Yuda Jin, Weidong Chen, Yuanhe Tian, Yan Song, Chenggang Yan, and Zhendong Mao. Improving radiology report generation with d 2-net: When diffusion meets discriminator. In ICASSP 2024- 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2215–2219. IEEE, 2024.

Xincheng Ju, Dong Zhang, Junhui Li, Shoushan Li, and Guodong Zhou. Enhanced generative framework with llms for multimodal emotion-cause pair extraction in conversations. IEEE Transactions on Multimedia, 2025.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Bobo Li, Hao Fei, Fei Li, Tat-seng Chua, and Donghong Ji. Multimodal emotion-cause pair extraction with holistic interaction and label constraint. ACM Transactions on Multimedia Computing, Communications and Applications, 2024.

Hengyou Li, Xinyan Liu, Guorong Li, Shuhui Wang, Laiyun Qing, and Qingming Huang. Boost tracking by natural language with prompt-guided grounding. IEEE Transactions on Intelligent Transportation Systems, 26(1):1088–1100, 2025. doi: 10.1109/TITS.2024.3492263.

Mingyuan Li, Tong Jia, Hao Wang, Bowen Ma, Shiyi Guo, Da Cai, and Dongyue Chen. Cspcl: Category semantic prior contrastive learning for deformable detr-based prohibited item detectors. Advances in Neural Information Processing Systems, 38:141729–141750, 2026a.

Shihao Li, Yuanxing Zhang, Jiangtao Wu, Zhide Lei, Yiwen He, Runzhe Wen, Chenxi Liao, Chengkang Jiang, An Ping, Shuo Gao, et al. If-vidcap: Can video caption models follow instructions? In International Conference on Learning Representations, volume 2026, pp. 23157–23201, 2026b.

Wei Li, Yang Li, Vlad Pandelea, Mengshi Ge, Luyao Zhu, and Erik Cambria. Ecpec: Emotion-cause pair extraction in conversations. IEEE Transactions on Affective Computing, 14(3):1754–1765, 2022.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.

Zefeng Lin, Weidong Chen, Yan Song, and Yongdong Zhang. Prompting few-shot multi-hop question generation via comprehending type-aware semantics. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 3730–3740, 2024.

Chang Liu, Yuanhe Tian, Weidong Chen, Yan Song, and Yongdong Zhang. Bootstrapping large language models for radiology report generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 18635–18643, 2024.

Xiaohao Liu, Xiaobo Xia, See-Kiong Ng, and Tat-Seng Chua. Continual multimodal contrastive learning. Advances in Neural Information Processing Systems, 38:48414–48455, 2026.

Yuanyuan Liu, Wei Dai, Chuanxu Feng, Wenbin Wang, Guanghao Yin, Jiabei Zeng, and Shiguang Shan. Mafw: A large-scale, multi-modal, compound affective database for dynamic facial expression recognition in the wild. In Proceedings of the 30th ACM international conference on multimedia, pp. 24–32, 2022.

Chong Ma, Shengbo Chen, Pengjie Tang, Hong Rao, and Hanli Wang. Glm-eer: Global-local memory and emotion evaluation refinement for emotional video description. Expert Systems with Applications, pp. 132042, 2026.

Heqing Ma, Jianfei Yu, Fanfan Wang, Hanyu Cao, and Rui Xia. From extraction to generation: multimodal emotion-cause pair generation in conversations. IEEE Transactions on Affective Computing, 2024.

Saif Mohammad. Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 english words. In Proceedings of the 56th annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 174–184, 2018.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532–1543, 2014.

Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Hobin Ryu, Sunghun Kang, Haeyong Kang, and Chang D Yoo. Semantic grouping network for video captioning. In proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 2514–2522, 2021.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Peipei Song, Dan Guo, Jun Cheng, and Meng Wang. Contextual attention network for emotional video captioning. IEEE Transactions on Multimedia, 2022.

Peipei Song, Dan Guo, Xun Yang, Shengeng Tang, Erkun Yang, and Meng Wang. Emotion-prior awareness network for emotional video captioning. In Proceedings ofthe 31st ACM International Conference on Multimedia, pp. 589–600, 2023.

Peipei Song, Dan Guo, Xun Yang, Shengeng Tang, and Meng Wang. Emotional video captioning with vision-based emotion interpretation network. IEEE Transactions on Image Processing, 2024.

Peipei Song, Long Zhang, Long Lan, Weidong Chen, Dan Guo, Xun Yang, and Meng Wang. Towards efficient partially relevant video retrieval with active moment discovering. IEEE Transac tions on Multimedia, 2025.

Shengeng Tang, Dan Guo, Richang Hong, and Meng Wang. Graph-based multimodal sequential embedding for sign language translation. IEEE Transactions on Multimedia, 24:4433–4445, 2021.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566–4575, 2015.

Bairui Wang, Lin Ma, Wei Zhang, and Wei Liu. Reconstruction network for video captioning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7622–7631, 2018.

Botao Wang, Keke Tang, and Peican Zhu. Enhancing emotion-cause pair extraction in conversations via center event detection and reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 10773–10783, 2024a.

Fanfan Wang, Zixiang Ding, Rui Xia, Zhaoyu Li, and Jianfei Yu. Multimodal emotion-cause pair extraction in conversations. IEEE Transactions on Affective Computing, 14(3):1832–1844, 2022.

Fanfan Wang, Heqing Ma, Xiangqing Shen, Jianfei Yu, and Rui Xia. Observe before generate: Emotion-cause aware video caption for multimodal emotion cause generation in conversations. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 5820–5828, 2024b.

Hanli Wang, Pengjie Tang, Qinyu Li, and Meng Cheng. Emotion expression with fact transfer for video description. IEEE Transactions on Multimedia.

Junbo Wang, Liangyu Fu, Yuke Li, Xuecheng Wu, and Zhiyong Wang. Adaptive emotional video captioning via affective heterogeneous graph reasoning and multi-task joint learning. arXiv preprint arXiv:2607.29045, 2026.

Yu Wang, Yuanyuan Liu, Shunping Zhou, Yuxuan Huang, Chang Tang, Wujie Zhou, and Zhe Chen. Emotion-oriented cross-modal prompting and alignment for human-centric emotional video captioning. IEEE Transactions on Multimedia, 2025.

Rui Xia and Zixiang Ding. Emotion-cause pair extraction: A new task to emotion analysis in texts. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1003–1012, 2019.

Ning Xu, Yifei Gao, Ting-Ting Zhang, Hongshuo Tian, and An-An Liu. Cross-modal coherenceenhanced feedback prompting for news captioning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 9369–9377, 2024.

Yifan Xu, Xinhao Li, Yichun Yang, Desen Meng, Rui Huang, and Limin Wang. Carebench: A finegrained benchmark for video captioning and retrieval. In International Conference on Learning Representations, volume 2026, pp. 51124–51150, 2026.

Cheng Ye, Weidong Chen, Jingyu Li, Lei Zhang, and Zhendong Mao. Dual-path collaborative generation network for emotional video captioning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 496–505, 2024.

Cheng Ye, Weidong Chen, Bo Hu, Lei Zhang, Yongdong Zhang, and Zhendong Mao. Improving video summarization by exploring the coherence between corresponding captions. IEEE Transactions on Image Processing, 2025a.

Cheng Ye, Weidong Chen, Peipei Song, Xinyan Liu, Lei Zhang, and Zhendong Mao. Multi-round mutual emotion-cause pair extraction for emotion-attributed video captioning. In Proceedings of the 33rd ACM International Conference on Multimedia, 2025b.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.

Keunwoo Yu, Zheyuan Zhang, Fengyuan Hu, Shane Storks, and Joyce Chai. Eliciting in-context learning in vision-language models for videos through curated data distributional properties. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 20416–20431, 2024.

Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. In International Conference on Learning Representations, volume 2025, pp. 38057–38081, 2025.

Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.

Fan Zhang, Zebang Cheng, Chong Deng, Haoxuan Li, Zheng Lian, Qian Chen, Huadai Liu, Wen Wang, YiFan Zhang, Renrui Zhang, et al. Mme-emotion: A holistic evaluation benchmark for emotional intelligence in multimodal large language models. In International Conference on Learning Representations, volume 2026, pp. 48767–48807, 2026.

Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10665–10674. IEEE, 2020.

Peican Zhu, Botao Wang, Keke Tang, Haifeng Zhang, Xiaodong Cui, and Zhen Wang. A knowledgeguided graph attention network for emotion-cause pair extraction. Knowledge-Based Systems, 286:111342, 2024.

## A RELATED WORK

## A.1 EMOTION-CAUSE PAIR EXTRACTION

Emotion cause analysis has received increasing attention due to its potential to promote emotional exploration. Emotion-cause pair extraction(ECPE) is one of the most representative tasks (Tang et al., 2021; Li et al., 2022; Wang et al., 2024a;a; An et al., 2023; Song et al., 2025; Lin et al., 2024; Li et al., 2025). Xia and Ding (Xia & Ding, 2019) firstly propose this task to extract emotions and their causes in documents. To this day, recent research has focused more on analyzing emotioncause pairs in diverse textual scenarios such as articles, stories, microblogs, or conversations. Zhu et al. (Zhu et al., 2024) propose a novel knowledge-guided graph attention network, which designs two guiding principles: inter-clause dependency graph and inter-pair possibility graph, to aggregate features between clause pairs for emotion-clause and cause-clause pair extraction. Chen et al. (Chen et al., 2024) propose to utilize the causal discourse knowledge in a knowledge distillation way, which designs a teacher model to learn to predict causal connective words between utterances and then guide a student model in identifying both the fine-grained emotion labels and causal spans. Hu et al. (Hu et al., 2023) analyze that emotion extraction is more crucial to the ECPE task than cause extraction and propose an approach oriented toward emotion prediction, which aims to fully exploit the potential of emotion prediction to enhance ECPE. Furthermore, they design a synchronization mechanism to share their optimizations in the training process.

Besides, some researchers have extended the ECPE task to the multimodal field (Li et al., 2024; Jin et al., 2024; Lin et al., 2024; Jin et al., 2024; Liu et al., 2024; Wang et al., 2022; Ma et al., 2024; Ju et al., 2025; Huang et al., 2025; Jin et al., 2024; ?; Gao et al., 2023). Wang et al. (Wang et al., 2024b) design a two-stage framework, which first generates emotion-cause aware video captions and then facilitates the generation of emotion causes. With the help of corresponding captions, the model improves the accuracy of pair extractions. However, all existing multimodal ECPE still perform on the text modality while treating image and audio modalities as auxiliary information. In contrast, our work focuses on the cross-modal scenario (visual causes + textual emotions).

## A.2 EMOTIONAL VIDEO CAPTIONING

Emotional video captioning is proposed to solve the problem of boring and soulless captions generated by traditional video captioning model, which first mines emotional clues hidden in videos and collaborates visual contents to generate vivid captions. Wang et al. (Wang et al.) firstly propose this task and release a new dataset. They design two independent prediction networks for factual and emotional captioning separately and finally generate emotional descriptions by weighted average scores. Song et al. (Song et al., 2022) propose a contextual attention network to recognize and describe the fact and emotion in the video by semantic-rich context learning. Song et al. (Song et al., 2023) propose a novel tree-structured emotion learning module to achieve explicit emotion perception. Song et al. (Song et al., 2024) incorporate visual context, textual context, and visualtextual relevance into an aggregated multimodal contextual vector to enhance video captioning. Ye et al. (Ye et al., 2024) propose a dual-path collaborative generation network to dynamically perceive emotions at different time steps and design an emotion-adaptive decoder to solve the problem of overemphasis on the role of emotional guidance. Wang et al. (Wang et al., 2025) introduce two learnable prompting strategies: visual emotion and textual emotion prompting, to learn emotional cue representations and further design two levels of objective functions: the ER-sentence level and the AU-word level alignment losses, to facilitate the interaction and alignment.

All aforementioned works directly perceive the emotional cues from visual contents and ignore that emotional cues have intrinsic motivational causes reflected in the video content. To the best of ou knowledge, we are the first work to notice the visual causes of emotions, improving the accuracy and interpretability of emotion perception and effectively alleviating the mutual interference of multiple factors in emotional caption generation.

## B THEORETICAL ANALYSIS: FROM THE INFORMATION THEORY PERSPECTIVE

To provide a rigorous theoretical foundation for our Counterfactual Contrastive Objective $( \mathcal { L } _ { c f } ) _ { }$ we analyze the fundamental flaws of existing positive-alignment methods (e.g., standard InfoNCE) from an information-theoretic perspective and demonstrate why our interventional approach strictly guarantees causal faithfulness.

The Flaw of Standard InfoNCE. Let $V , Y$ and C denote the random variables for the video, emotion, and true foreground cause, respectively. Traditional EVC models optimize the representation by applying InfoNCE loss between the video representation and the emotion embedding. Mathematically, minimizing the standard InfoNCE loss is equivalent to maximizing the Mutual Information (MI) between the video and the emotion:

$$
\operatorname* { m i n } \mathcal { L } _ { I n f o N C E } \iff \operatorname* { m a x } I ( V ; Y )\tag{21}
$$

However, according to our structural causal model, the video is composed of true causes $C$ and the background confounders U. By the mutual information chain rule, $\bar { I ( V ; Y ) }$ can be decomposed as:

$$
I ( V ; Y ) = I ( C , U ; Y ) = I ( C ; Y ) + I ( U ; Y \vert C )\tag{22}
$$

In the datasets, the confounder U (e.g., a ”party” scene) frequently co-occurs with the emotion $Y ~ ( ^ { \mathrm { { ' h a p p y } ^ { \mathrm { { ' } } } } } )$ independently of true causes $C \ { \mathrm { ( e . g . } }$ ”opening gifts”). Thus, the conditional mutual information $I ( U ; Y | C ) > 0$ . When a model blindly maximizes $I ( V ; Y )$ , it inevitably maximizes the spurious correlation $I ( U ; Y | C )$ This explains why existing models suffer from Contextual Leakage. They learn to predict emotions by maximizing the information from the easily accessible background confounders U rather than highly dynamic and harder-to-capture true causes $C .$

The Superiority of Counterfactual Intervention. To achieve authentic causal reasoning, our goal is to exclusively maximize $I ( C ; Y )$ while strictly minimizing the spurious leakage $I ( \bar { U } ; Y )$ . Our framework accomplishes this through the explicitly generated causal mask $\mathcal { M } _ { c }$ . We decouple the video into the foreground cause representation $c \overset { \cdot } { = } \mathcal { V } \odot \mathcal { M } _ { c }$ and the counterfactual background confounder $U = \mathcal { V } \odot ( 1 - \mathcal { M } _ { c } )$

Our proposed Counterfactual Contrastive Objective $( \mathcal { L } _ { c f } )$ is formulated as:

$$
\mathcal { L } _ { c f } = - \mathbb { E } \left[ \log \frac { \exp ( \sin ( \mathcal { C } , \mathcal { E } ) / \tau ) } { \exp ( \sin ( \mathcal { C } , \mathcal { E } ) / \tau ) + \exp ( \sin ( U , \mathcal { E } ) / \tau ) } \right]\tag{23}
$$

According to the variational bounds of mutual information (e.g., InfoNCE as an MI estimator), optimizing $\mathcal { L } _ { c f }$ intrinsically acts on the ratio of conditional densities. By treating the true cause $\mathcal { C }$ as the positive sample and the decoupled confounder $U$ as the strict hard negative sample within the same video context, our objective function tightly bounds the difference in mutual information:

$$
\begin{array} { r } { \operatorname* { m i n } \mathcal { L } _ { c f } \implies \operatorname* { m a x } \Big ( I ( \mathcal { C } ; \mathcal { E } ) - I ( U ; \mathcal { E } ) \Big ) } \end{array}\tag{24}
$$

This formulation yields a profound theoretical guarantee: our model is explicitly penalized if the background confounder $U$ contains any predictive information about the emotion $\varepsilon .$ To minimize the loss, the optimization process forces the causal mask $\mathcal { M } _ { c }$ to absorb all emotion-relevant semantic into ${ \mathcal { C } } ,$ driving the spurious mutual information $I ( U ; { \mathcal { E } } )$ towards zero.

Therefore, compared to traditional positive alignment that inadvertently absorbs environmental biases, our counterfactual intervention rigorously severs the backdoor path $V  U  Y$ . It ensures that the emotion prediction is authentically anchored to the interventional distribution $P ( Y | d o ( C ) )$ , thereby achieving superior causal faithfulness.

## C MORE DETAILS OF EVC-CAUSEGROUND

## C.1 COMPARISON WITH EXISTING BENCHMARK

In this paper, we construct EVC-CauseGround, a novel benchmark that extends conventional emotion descriptions with a comprehensive suite of fine-grained cause annotations. Table A.1 presents a systematic comparison between existing emotion/causal reasoning datasets and our proposed EVC CauseGround. Traditional emotion classification and captioning datasets exclusively provide holistic emotion labels or descriptions, fundamentally lacking the annotation of causal triggers, which makes it impossible to assess the true causal reasoning ability. Conversely, while some Emotion-Cause Pair Extraction datasets attempt to address causality, they not only omit comprehensive emotion captions required for generation evaluation, but their causal annotations are also restricted to coarse-grained utterance levels, entirely neglecting fine-grained visual cause localization. Therefore, EVC-CauseGround stands as the first benchmark to unify emotion labels, rich emotional captions, dense spatio-temporal visual cause annotations, and a meticulously curated counterfactual subset. It establishes a solid foundation for comprehensively evaluating the dual capabilities of EVC models: high-quality caption generation and authentic emotional cause reasoning.

Table A.1: Comparison of EVC-CauseGround with existing datasets across different domains. “V”, “A”, and “T” stand for video, audio, and text modalities, respectively. Our dataset is the first to simultaneously provide fine-grained spatio-temporal causal grounding and a rigorously diagnosed counterfactual subset for real-world emotional videos
<table><tr><td>Dataset</td><td>Modality</td><td>Emotion Labels</td><td>Emotion Captions</td><td>Temporal Grounding</td><td>Spatial (Dense Tubes)</td><td>Counterfactual Subset</td></tr><tr><td colspan="7">Emotion Classification &amp; Captioning</td></tr><tr><td>V-Emo (2014)</td><td>V, A</td><td>√</td><td>×</td><td>X</td><td>X</td><td>X</td></tr><tr><td>EmVidCap (0)</td><td>V,T</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td colspan="7">Emotion-Cause Pair Extraction (ECPE)</td></tr><tr><td>ECF (2022)</td><td>V, A, T</td><td>√</td><td>X</td><td>√(Utterance-level)</td><td>X</td><td>X</td></tr><tr><td>MECESD (2024)</td><td>V, A, T</td><td>√</td><td>√</td><td>√(Utterance-level)</td><td>X</td><td>X</td></tr><tr><td colspan="7">Spatio-Temporal Grounding &amp; Causal Reasoning</td></tr><tr><td>VidSTG (2020)</td><td>V,T</td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>CLEVRER (2019)</td><td>V, T</td><td>X</td><td>√</td><td>√</td><td>√</td><td>√(Synthetic)</td></tr><tr><td>EVC-CauseGround (Ours)</td><td>V, T</td><td>√</td><td>√</td><td>√(Frame-level)</td><td>√</td><td>√(Real-world)</td></tr></table>

## C.2 DETAILS OF DATA SOURCES

The foundation of a robust causal reasoning benchmark relies heavily on the quality and diversity of its video sources. To effectively evaluate the ability to decouple genuine visual emotion triggers from spurious correlations, the selected videos must exhibit highly diverse, open-world scenes (in-thewild) and sufficient temporal durations. This structural complexity is crucial for providing abundant environmental contexts, acting as potential environmental confounders, against which the model must learn to pinpoint the true causal regions. Driven by these principles, we meticulously curate raw videos from two original datasets.

First, we source a substantial portion of our data from the widely-used EmVidCap (Wang et al.) benchmark. The original EmVidCap consists of two subsets: EmVidCap-S and EmVidCap-L. However, we explicitly discard the videos from EmVidCap-S. The videos in EmVidCap-S are exceedingly short (typically under 5 seconds) and feature monotonic scenes. In such cases, the causal action spans nearly the entire video duration, rendering fine-grained spatiotemporal cause annotation virtually meaningless. Therefore, we exclusively select videos from EmVidCap-L. EmVidCap-L comprises videos primarily collected from user-generated content (UGC) platforms, offering a much broader spectrum of daily events and dynamic human interactions. It contains 1523 videos with an average duration of approximately 100 seconds, providing an ideal temporal canvas for fine-grained causal grounding. Second, to further scale up the benchmark and amplify the diversity, we incorporate a carefully selected subset of videos from MAFW (Liu et al., 2022). MAFW is a large-scale, in-the-wild emotion dataset sourced from various internet platforms (e.g., YouTube and movies) and features an extensive array of unconstrained environments, ranging from extreme sports and natural disasters to bustling streets and crowded parties. The original MAFW dataset contains 10,045 video clips. We filter and select 477 videos from MAFW that exhibit prominent foreground dynamic actions accompanied by complex background scenes. By amalgamating these two sources, we have successfully collected a total of 2k high-quality emotional videos, which are split into 1500 videos for training and 500 videos for evaluation. Crucially, all selected videos are natively equipped with emotion labels and descriptive emotional captions provided by their original datasets. This inherent semantic richness serves as a reliable ground-truth anchor for our subsequent pipeline to deduce and annotate the precise spatiotemporal causes without suffering from emotion hallucination.

## C.3 SPATIOTEMPORAL VISUAL CAUSE ANNOTATION

Unlike previous datasets that rely on exhaustive, cost-prohibitive manual drawing, we propose a novel MLLM-led, Tracker-assisted Coarse-to-Fine Pipeline to efficiently generate high-fidelity spatio-temporal causal tubes. This pipeline explicitly circumvents the inherent temporal hallucinations and spatial coordinate drifting typical of current MLLMs when processing long-form videos. The annotation process involves three meticulously designed stages:

Temporal Span Grounding via Sequence Grid Prompting. Current Video-LLMs struggle with precise temporal localization due to context-length constraints and memory bottlenecks when processing native video streams. To bypass this, we map the temporal dimension into a unified spatial layout. Specifically, we uniformly sample $T = 3 0$ frames from each video, implicitly aligning with the input space of most mainstream EVC models. We then concatenate these frames into a highly structured $6 \times 5$ grid image, with each frame explicitly watermarked with a sequence identifier $( \mathrm { e . g . }$ F00 to F29). We prompt an advanced MLLM (e.g., GPT-5.5 (Singh et al., 2025)) with this grid, alongside the ground-truth emotion label and caption, to reversely deduce the precise temporal window $\mathbf { \bar { [ } } t _ { s t a r t } , t _ { e n d } ]$ that directly triggers the emotion. Crucially, to prevent the MLLM from exhibiting ”lazy behaviors” (i.e., trivially selecting the entire video sequence as the cause), we enforce a strict length-constraining prompt. We mandate that the predicted causal span must not exceed 30% of the total frames (e.g., max 9 frames). This forces the model to pinpoint the absolute climax or peak moment of the causal action rather than the redundant pre- or post-action context.

Spatial Grounding via Peak Motion Anchoring. Directly prompting an MLLM to generate bounding boxes across multiple consecutive frames frequently leads to severe spatial hallucinations and bounding box jittering. To ensure pixel-perfect spatial accuracy, we decouple temporal and spatial grounding. Within the localized temporal span $[ t _ { s t a r t } , t _ { e n d } ]$ , we employ the optical flow mechanism to calculate inter-frame pixel displacement. This allows us to automatically identify the peak motion frame, the exact moment where the dynamic causal action is most pronounced. We then feed only this single, high-resolution peak frame into the MLLM, prompting it to generate a highly confident, static bounding box surrounding the causal agent or action. By reducing the cognitive load to a single image, the MLLM achieves remarkable spatial precision devoid of multi-frame interference.

Dense Causal Tube Generation via Zero-shot Tracking. A single bounding box is insufficient for dynamic causal reasoning. To extrapolate the spatial annotation across the entire temporal event, we leverage the state-of-the-art zero-shot visual tracker, SAM-3 (Carion et al., 2026). Utilizing the MLLM-generated bounding box on the peak frame as an initial prompt, SAM-3 bidirectionally tracks the causal object forward to $t _ { e n d } \mathrm { a n d }$ backward to $t _ { s t a r t }$ This seamless integration of MLLM reasoning and traditional visual tracking yields continuous, smooth, and dense spatiotemporal causal tubes, achieving human-level annotation quality at a fraction of the cost.

## D EXPERIMENTAL SETTINGS

Datasets. We experiment on two public EVC benchmarks, $e . g .$ , EVC-MSVD (Chen & Dolan, 2011) and EVC-VE (Jiang et al., 2014). EVC-MSVD contains 240/134 videos and 8,169/4,611 sentences for training/testing by additionally embedding emotion words into the caption sentences of the traditional video captioning dataset MSVD (Chen & Dolan, 2011). EVC-VE is built based on a traditional emotional video prediction dataset VideoEmotion-8 (Jiang et al., 2014) and is divided into 1,141/382 videos and 19,398/6,527 sentences for training/testing, respectively.

Evaluation Metrics. To evaluate the accuracy of emotion in generated sentences effectively, following previous work (Wang et al.), we consider the emotion word accuracy $\operatorname { A c c } _ { \operatorname { s w } }$ (Wang et al.) and emotion sentence accuracy Acc<sub>c</sub> (Wang et al.). Additionally, to measure the semantic matching degree between the generated sentence and the ground-truth label, we use the standard metrics, $e . g .$ , BLEU (Papineni et al., 2002), METEOR (Banerjee & Lavie, 2005), ROUGE (Lin, 2004), and CIDEr (Vedantam et al., 2015). Moreover, there are two hybrid metrics BFS (Wang et al.) and CFS (Wang et al.) that combine the emotion evaluation with BLEU and CIDEr metrics, respectively.

Implementation Details. For each video, we sample $T = 3 0$ frames and resize them to 224 × 224 with central cropping for pixel values. Following (Wang et al.) and (Song et al., 2023), we build an overall vocabulary of size 32,128 that contains all words in the corpus and set the number of emotion words to $N _ { w } = 1 7 9$ . The embedding dimensions are constructed to $d _ { E } = 3 0 0$ and $D = 1 4 0 8$ . In training, we adopt Qwen-2.5-VL-7B (Raffel et al., 2020) as the language decoder and integrate a LoRA adapter (Hu et al., 2021) with $r ~ = ~ 1 6$ and $\alpha = 3 2$ to align the visual and emotional latent space. We adopt the Adam optimizer (Kingma & Ba, 2014) with a learning rate of 7e-4, and the batch size is set to 4. We set the global motion parameter $\lambda _ { m } .$ the penalty parameter θ, the objective function weights $\lambda _ { e } , \lambda _ { c l s } , \lambda _ { t e m p } , \lambda _ { s p a }$ , and $\lambda _ { c f }$ to 0.2, 0.1, 0.8, 0.3, 0.2, 0.2, and 0.1. The maximum length of captions is set to 15 and the size of beam search is set to 4. All experiments are implemented on 4 NVIDIA A800 GPUs.

## E BASELINES

We consider several state-of-the-art methods to make comparisons and divide them into two categories: (i) Traditional Captioning Methods and (ii) Emotional Captioning Methods.

## (i) Traditional Captioning Methods

1) SA-LSTM (CVPR18’) (Wang et al., 2018) proposes a reconstruction network to leverage both the forward (video to sentence) and backward (sentence to video) flows. The forward flow produces the sentence description based on the encoded video semantic features and the backward flow reproduces the video features based on the hidden state sequence generated by the decoder.

2) CANet (TMM22’) (Song et al., 2022) proposes a contextual attention network to recognize and describe the fact and emotion by semantic-rich context learning, which first extracts visual and textual features from both video and previously generated words and then applies the attention mechanism to capture informative contexts for captioning.

3) VideoBLIP (EMNLP24’) (Yu et al., 2024) proposes a training paradigm that induces in-context learning over video and text by capturing key properties of pre-training data found by prior work to be crucial for improving the ability of LLMs to generate video descriptions.

## (ii) Emotional Captioning Methods

1) FT (TMM21’) (Wang et al.) introduces a visual emotion analyzer to detect implicit emotional cues and a dual-stream network that integrates emotional and factual semantics for caption generation by a weighted sum operation.

2) EPAN (MM23’) (Song et al., 2023) introduces a tree-structured emotion repository to enable hierarchical emotion perception and an emotion-prior awareness network to achieve the explicit and fine-grained emotion perception by an emotion masking mechanism and then decode the emotional caption by exploiting the multimodal semantic cues.

3) VEIN (TIP24’) (Song et al., 2024) designs a vision-based emotion interpretation network, which first models the emotion distribution over an open psychological vocabulary and then incorporates visual context, textual context, and visual-textual relevance into an aggregated multimodal contextual vector to enhance video captioning.

4) DCGN (MM24’) (Ye et al., 2024) propose a framework to leverage fact-emotion collaborative learning to address the challenge of dynamic emotion changes during caption generation.

5) MM-ECPE (MM25’) (Ye et al., 2025b) focus on the importance of emotional causes in emotional exploration and propose a multi-round mutual learning network to jointly extract emotion-cause pairs for LLM-based caption generation.

6) HEART (TAFFC26’) (Han et al., 2026) introduces a hierarchical semantic extraction module to decompose visual content into entity-, action-, and event-level representations, and a temporal pyramid module to capture temporal dependencies for temporally coherent captioning.

7) GLM-EER (ESWA26’) (Ma et al., 2026) designs a global-local memory module to learn abstract emotional representation across time dimensions and a Emotion Evaluation Refinement module to enhance emotion learning by dynamically adjusting cross-entropy loss weights.

8) SAGML (Arxiv26’) (Wang et al., 2026) proposes an adaptive EVC framework via affective heterogeneous graph and multi-task language modeling, which constructs a soft affective heterogeneous graph to allow visually supported lexical emotions to remain recoverable.

## F HUMAN EVALUATION

Due to the highly subjective nature of the EVC task, it is a crucial criterion for evaluation whether the generated descriptions conform to human preferences. Therefore, we add human evaluation to fully measure the quality of emotional descriptions. Inspired by the work (Wang et al.), we designs four metrics, including (1) emotion accuracy: assess the accuracy of emotional expression, (2) relevance: assess whether the generated description is relevant to the video content, (3) coherence: assess the coherence and readability of the description, and (4) usability: how useful would the description be for a person (especially a blind person) to understand what is happening in the video. The score of each metric ranges from 1 to 10. Subsequently, we invite 20 participants with excellent English skills to score a subset of 50 video-caption pairs on these four metrics and calculate the average score on each metric. We make a comparison with two types of models on human evaluation: the state-of-the-art models on EVC and several large vision-language models (LVLMs).

Comparison with SOTA methods. As shown in Table A.2, compared with the SOTA methods for EVC, our model is significantly ahead in all four metrics. Our proposed MCSL captures finegrained cause localization, thereby generating more accurate and diverse vocabulary. Furthermore, our proposed ISER utilizes VAD constraints to enhance the interpretability of emotions, making the generated descriptions more easily empathized with and understood by humans. These show that our model does not simply imitate the reference sentences, but learns the representation of emotional and factual semantics.

Comparison with LVLMs. As shown in Table A.2, despite the outstanding performance of LVLMs, our model still outperforms them in human evaluations. Due to their powerful ability in objective understanding and generation, LVLMs could produce a wide variety of descriptions. However, limited by insufficient emotion understanding and the influence of visual hallucinations, they sometimes generate factual and emotional contents irrelevant to the video, leading to lower human ratings. Conversely, our model filters out emotion-irrelevant visual information and visual-irrelevant emotional information through fine-grained visual and emotional refinement, thereby generating emotional descriptions that are more closely aligned with the video content.

Table A.2: The comparison with SOTA methods on human evaluations.
<table><tr><td>Metric</td><td>Accuracy</td><td>Relevance</td><td>Coherence</td><td>Usability</td></tr><tr><td>VEIN (2024)</td><td>5.47</td><td>5.18</td><td>6.27</td><td>5.04</td></tr><tr><td>EPAN (2023)</td><td>6.24</td><td>5.97</td><td>7.13</td><td>5.76</td></tr><tr><td>DCGN (2024)</td><td>6.88</td><td>6.43</td><td>7.68</td><td>6.32</td></tr><tr><td>VideoBLIP (2024)</td><td>5.91</td><td>5.54</td><td>7.20</td><td>4.56</td></tr><tr><td>MM-ECPE (2025b)</td><td>6.40</td><td>6.75</td><td>8.04</td><td>7.36</td></tr><tr><td>GPT-4o (2024)</td><td>6.98</td><td>6.04</td><td>7.24</td><td>6.62</td></tr><tr><td>VideoLLaMA3 (2025)</td><td>7.22</td><td>6.20</td><td>7.72</td><td>6.84</td></tr><tr><td>CausalEVC(Ours)</td><td>7.52</td><td>6.85</td><td>8.16</td><td>7.58</td></tr></table>