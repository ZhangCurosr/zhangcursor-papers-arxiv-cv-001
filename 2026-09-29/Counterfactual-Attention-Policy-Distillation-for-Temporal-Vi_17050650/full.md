# Counterfactual Attention Policy Distillation for Temporal Video Grounding

Shaobo Ju<sup>1</sup>, Haiyang Yu<sup>2</sup>, Xuecheng Wu<sup>3</sup>, Qiong Wu<sup>1</sup>, Jiacong Wang<sup>4</sup>, Fan Shi<sup>2</sup>,

Jun Peng<sup>1</sup>, Yiyi Zhou<sup>1∗</sup>

<sup>1</sup> Key Laboratory of Multimedia Trusted Perception and Eficient Computing,

Ministry of Education of China, Xiamen University, 361005, P.R. China.

<sup>2</sup> Fudan University.

<sup>3</sup> MMLab, The Chinese University of Hong Kong.

<sup>4</sup> University of the Chinese Academy of Sciences. jushaobo@stu.xmu.edu.cn

## Abstract

Temporal video grounding is a key capability of advanced Multimodal Large Language Models (MLLMs) for the thorough understanding of video events, which is however often limited by repeated actions and visually similar contexts in long videos. In this paper, we study this issue from the perspective of On-policy distillation (OPD) and propose a new training regime for MLLMs termed Counterfactual Attention Policy Distillation (CAPD). In particular, OPD is a viable solution for MLLMs via providing dense teacher supervision on student-generated trajectories. But its next-token based teacher-student distillation is hard to identify the specific video segments supporting each predicted timestamp, which is critical for temporal grounding. In this case, CAPD measures how masking each temporal group changes the teacher’s output distribution. The resulting counterfactual influence calibrates the teacher’s attention and weights token-level distillation, allowing the student to learn the temporal evidence that affects boundary prediction. To validate CAPD, we trained it on Qwen3-VL-8B-Instruct using only 2,500 samples for one epoch, and evaluated it on the TimeLens and multiple general video benchmarks. Experimental results show that CAPD improves average recall by 12.0% relative to GRPO on Time-Lens while preserving general video understanding, achieving comparable accuracy to the base model.

## Introduction

Temporal video grounding (TVG) aims to localize the video interval described by a language query. It is a fundamental capability for video retrieval, editing, and evidence-based question answering. Early studies formulate TVG as proposal ranking, span prediction, or set prediction (Gao et al. 2017; Zhang et al. 2020b; Mun, Cho, and Han 2020; Zhang et al. 2020a; Lei, Berg, and Bansal 2021; Li et al. 2022). Recent methods improve query-conditioned video representations, develop unified temporal localization frameworks, and leverage large-scale video-language pretraining (Moon et al. 2023; Lin et al. 2023; Bao et al. 2025; Yang et al. 2025). Despite this progress, accurately identifying precise event boundaries remains dificult when a short target event is surrounded by repeated actions, visually similar temporal context, or irrelevant but visually salient content.

Multimodal large language models (MLLMs) (Bai et al. 2025a,b; Ren et al. 2024; Huang et al. 2024) formulate temporal video grounding as a text generation task, which directly generate the start and end timestamps from a video and a query without requiring a specially designed localization module (Qian et al. 2024; Wang et al. 2024). Timestampaware encoding and instruction tuning improve temporal perception (Ren et al. 2024; Huang et al. 2024; Qian et al. 2024; Wang et al. 2024; Yang et al. 2025), while recent post-training methods further strengthen grounding through verifiable rewards or dense teacher supervision (Wang et al. 2025b; Zhang et al. 2026b; Li et al. 2026b). In particular, Li et al. adopts On-Policy-Distillation (OPD) (Li et al. 2026b) to let the student generate its own on-policy trajectory and queries a stronger teacher on the same token prefixes. This setup provides the student with intensive monitoring signals, avoiding reliance solely on teacher-generated trajectories.

Query: Identify the time range when a golden line is added to the nail.  
![](images/9f6ad6a74d48c85f321dbf0fccca6bcc1ae3576f8cfa8a0c42add3a2391b8ae8.jpg)  
Figure 1: Comparison of OPD, RAL-AD, and CAPD. OPD only transfers output distributions, RAL-AD distills teacher attention, and CAPD uses counterfactual influence to calibrate attention and weight output-level distillation.

However, OPD only tells the student what to predict by matching the teacher’s next-token distribution, but it doesn’t provide guidance on which temporal regions should support each timestamp token. Reinforced Attention Learning (RAL) (Li et al. 2026a) alleviates this gap by treating the teacher’s internal attention distribution as an evidencerouting policy and transferring it to the student through onpolicy attention distillation (Li et al. 2026a). RAL provides an input-side learning signal that guides the student where to look in the video. However, where the teacher attends is not necessarily what its prediction depends on. As shown in Figure 1, masking a highly attended segment may leave the predicted timestamps unchanged, indicating that raw attention does not reliably identify decision-relevant evidence.

To address this limitation, we propose Counterfactual Attention Policy Distillation (CAPD). As introduced above, OPD asks the student to match the teacher’s output distribution, while RAL also asks it to match where the teacher attends. In particular, CAPD goes one step further by checking whether each attended video segment actually afects the teacher’s prediction. Specifically, for each answer token, CAPD sums the teacher’s attention within several contiguous temporal groups. It then masks one group at a time, runs the frozen teacher again under the same student-generated context, and then measures the change in its next-token distribution. A larger change indicates that the masked group contributes more to the prediction, so CAPD will increase the target attention on groups that cause larger output changes and reduces it on groups that cause little or no change. The student is trained to match both this corrected attention distribution and the teacher’s output distribution, using the same on-policy rollouts as OPD. With these careful designs, CAPD can transfer decision-relevant temporal evidence while preserving OPD’s dense output-level supervision, thereby improving temporal localization.

We validate CAPD with Qwen3-VL-8B using only 2,500 training examples for one epoch. Across three TimeLenscorrected benchmarks, CAPD achieves a 12.0% improvement in average recall over GRPO. Importantly, these improvements in temporal understanding did not come at the expense of undermining general video understanding capabilities. CAPD remains comparable to the base model on several general video benchmarks, indicating that it largely retains the model’s general capabilities after training.

Overall, our contributions are three-fold:

• We identify the key shortcoming ofexisting OPD methods for temporal video grounding, i.e., OPD supervises outputs, while attention distillation shows where the teacher attends, not which regions determine its prediction.

• We propose a novel training paradigm for MLLMs termed Counterfactual Attention Policy Distillation (CAPD), which masks contiguous temporal groups to measure their efects on teacher predictions. The resulting influence calibrates teacher attention and reweights output-level distillation without altering the architecture or rollout.

• Under the same training settings, CAPD consistently outperforms OPD and RAL on TimeLens while largely preserving general video understanding.

## Related Work

## Temporal Video Grounding

Temporal video grounding has evolved from specialized localization models to generative video MLLMs (Wu et al. 2026). Early approaches rank candidate segments, model pairwise moment relations, predict boundary spans, or perform set prediction (Gao et al. 2017; Zhang et al. 2020b,a; Lei, Berg, and Bansal 2021). With the development of video MLLMs, TimeChat improves timestamp-aware longvideo understanding through temporal instruction tuning, while VTimeLLM strengthens boundary perception using boundary-aware instruction data (Ren et al. 2024; Huang et al. 2024). Recent methods further emphasize temporal reasoning. Time-R1 applies reinforcement learning, VTime-CoT introduces training-free visual temporal reasoning, and TAR constrains intermediate reasoning with progressively refined timestamp anchors (Wang et al. 2025b; Zhang et al. 2025; Guo et al. 2026). TimeLens improves annotation quality and post-training for more reliable grounding, whereas Video-OPD provides dense teacher distributions on studentgenerated responses (Zhang et al. 2026b; Li et al. 2026b). Counterfactual learning has also been explored for temporal grounding: Zhai et al. synthesize altered video-query samples to mitigate moment bias, while Xu et al. construct counterfactual sequences for weakly supervised contrastive learning (Zhai et al. 2022; Xu, Xu, and Miao 2025; Wang, Chen, and Shen 2025). Unlike these approaches, CAPD uses temporal-group interventions to measure a frozen teacher’s token-level output sensitivity and transfers this decisionrelevant evidence through on-policy attention distillation.

## On-Policy Distillation

Recent surveys place on-policy distillation (OPD) within distribution-based knowledge distillation for large language models (Yang et al. 2024; Fang et al. 2026; Song and Zheng 2026). Unlike of-policy distillation, which trains the student on fixed reference or teacher-generated prefixes, OPD uses trajectories generated by the current student and asks the teacher to supervise the states that the student actually visits, thereby reducing the gap between training and inference. MiniLLM (Gu et al. 2024) applies reverse KL to student-generated sequences to reduce exposure mismatch. GKD (Agarwal et al. 2024) generalizes OPD by mixing student-generated and fixed trajectories and supporting different divergence objectives. DistiLLM (Ko et al. 2024) combines skew KL with adaptive reuse of of-policy data to improve training eficiency and stability. Veto (Jang et al. 2026) modifies the teacher target to reduce harmful updates on lowconfidence tokens. Prefix OPD (Zhang et al. 2026a) supervises only useful reasoning prefixes and stops rollouts early to reduce training cost. TIP (Xu et al. 2026) estimates token importance from student uncertainty and teacher–student disagreement. Self-Distilled Reasoner (Zhao et al. 2026) uses the same model as a teacher with access to extra information and as a student with only the original input, avoiding the need for a separate teacher model. VOLD (Bousselham, Kuehne, and Schmid 2026) transfers reasoning from a textonly language model to a vision–language model through reinforcement learning and OPD. Video-OPD (Li et al. 2026b) applies OPD to temporal video grounding. RAL (Li et al. 2026a) treats attention distributions as latent policies and distills them on student-generated trajectories. Although attention can be an efective distillation target (Li et al. 2024), it may not faithfully show which inputs determine the model prediction (Jain and Wallace 2019; Wiegrefe and Pinter 2019). CAPD therefore measures the counterfactual influence of temporal groups and uses it to calibrate teacher attention before transferring it to the student.

![](images/babe1a1bd9c461575b821666ec7092a9811d877f0255df4da1074ed20ec857d2.jpg)  
Figure 2: Illustration of the CAPD framework. Given the temporally grouped video tokens and query, the student generates an on-policy trajectory and the frozen teacher provides next-token supervision. CAPD masks each temporal group (a), estimates its counterfactual influence from the resulting teacher-output changes (b), and uses this influence to calibrate teacher attention and weight output-level distillation (c). The final objective combines output-level OPD with calibrated attention-policy learning.

## Method

## Overview

In this paper, we propose CAPD, a post-training method for temporal video grounding, whose overall framework is illustrated in Figure 2. CAPD aims to transfer not only the teacher’s output distribution but also the temporal evidence that directly afects its prediction. Standard OPD (Li et al. 2026b) aligns the student with the teacher’s next-token distributions, providing dense supervision for timestamp generation but no explicit guidance about which temporal regions should be used for prediction. To address this gap, RAL (Li et al. 2026a) treats attention distributions as policies, revealing where the teacher attends. However, a highly attended region may have little efect on the prediction. CAPD addresses this limitation by using counterfactual interventions to identify decision-relevant temporal evidence.

For each video-query pair, the student first generates an onpolicy response with predicted timestamps, and the frozen teacher evaluates the same student-generated prefixes to provide next-token distributions and output-to-input attention. The former supervises the output, while the latter indicates where the teacher attends.

CAPD consists of five steps. We first perform on-policy distillation and construct a temporal-group attention policy by aggregating attention over temporal groups, non-visual prompt tokens, and previous response tokens. We then estimate counterfactual group importance by masking each temporal group and measuring the resulting change in the teacher’s output distribution. We use this importance for counterfactual attention calibration toward influential groups and counterfactual token weighting for outputs that depend more strongly on visual evidence. Finally, we combine the weighted OPD loss with the calibrated attention loss.

## On-Policy Distillation

Given a video V and a text query q, we denote the trainable student by $\pi _ { \mathrm { { s } } } ( \cdot ; \theta )$ and the frozen teacher by $\pi _ { \mathrm { t } } ( \cdot ; \phi )$ . The response sampled from the student is

$$
y = ( y _ { 1 } , \ldots , y _ { L } ) \sim \pi _ { \mathrm { s } } ( \cdot \mid V , q ) ,\tag{1}
$$

where L is the number of response tokens included in distillation and i indexes an output position. At position i, both models receive the same student-generated prefix $y _ { < i }$ . For either model $m \in \{ \mathrm { s } , \mathrm { t } \}$ , its next-token distribution is

$$
p _ { m , i } = \pi _ { m } ( \cdot \mid V , q , y _ { < i } ) .\tag{2}
$$

We define the output-level discrepancy at this position as

$$
d _ { i } = \mathrm { K L } ( p _ { \mathrm { s } , i } \parallel p _ { \mathrm { t } , i } ) .\tag{3}
$$

OPD averages this discrepancy over the sampled response:

$$
\mathcal { L } _ { \mathrm { O P D } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } d _ { i } .\tag{4}
$$

This objective tells the student which output distribution to match, but it does not tell the student which part of the video supports each output token.

## Temporal-Group Attention Policy

First, we divide the video into G temporal groups, assigning all visual tokens corresponding to the same frame to the same group. At the final Transformer layer, let $A _ { m , i , j }$ denote model m’s self-attention probability from output position i to input position $j ,$ after applying the same positional encoding and masks as in the forward and averaging over attention heads. Let $\mathcal { T } _ { g }$ contain the indices of all visual tokens in temporal group g. The attention mass assigned to that group is

$$
a _ { m , i , g } = \sum _ { j \in \mathcal { T } _ { g } } A _ { m , i , j } .\tag{5}
$$

In addition to the G temporal groups, we separately calculate attention to non-visual prompt tokens and previously generated response tokens. The resulting attention policy is

$$
\begin{array} { r } { \mathbf { a } _ { m , i } = [ a _ { m , i , 1 } , \dots , a _ { m , i , G } , a _ { m , i , \mathrm { p r } } , a _ { m , i , \mathrm { p f } } ] \in \Delta ^ { G + 2 } . } \end{array}\tag{6}
$$

These two language-token groups preserve how much the token depends on the prompt and its response prefix. Otherwise, the visual attention would be incorrectly renormalized to one. In our RAL-AD baseline, attention is directly used for distillation, achieved by minimizing the Jensen–Shannon divergence between $\mathbf { a } _ { \mathrm { s } , \mathrm { i } }$ <sub>i</sub> and $\mathbf { a } _ { \mathrm { t } , i }$ together with the OPD loss. It transfers where the teacher looks, but it does not establish whether a visual group changes the teacher’s prediction.

## Counterfactual Group Importance

To measure the efect of temporal group g, we construct a counterfactual video $V ^ { ( - g ) }$ . For every visual token in the group, its feature is replaced by the mean feature at the same spatial location over the remaining temporal positions. This removes the group’s temporal content while preserving the input shape, spatial layout, and positional indices. The frozen teacher then evaluates $V ^ { ( - g ) }$ using the same query and the same student prefix $y _ { < i }$ as in the full-video forward pass. We denote the resulting next-token distribution by $p _ { \mathrm { t } , i } ^ { ( - g ) }$ . The importance of group g to output position i is the change in the teacher’s next-token distribution after this removal:

$$
c _ { i , g } = \mathrm { J S D } \Big ( p _ { \mathrm { t } , i } , p _ { \mathrm { t } , i } ^ { ( - g ) } \Big ) .\tag{7}
$$

Thus, $a _ { \mathrm { t } , i , g }$ and $c _ { i , g }$ answer diferent questions. The former measures how much attention the teacher routes to group g, whereas the latter measures whether that group afects the teacher’s output. We summarize the total visual dependence of token i as

$$
u _ { i } = \sum _ { g = 1 } ^ { G } c _ { i , g } .\tag{8}
$$

Since the absolute scale of $u _ { i }$ varies across responses, we use a bounded gate

$$
\rho _ { i } = \frac { u _ { i } } { u _ { i } + s } ,\tag{9}
$$

where s is the median of the positive $u _ { i }$ values within the current response. If no token has positive importance, we set every $\rho _ { i }$ to zero. This gate prevents numerically small counterfactual changes from causing a large adjustment.

## Counterfactual Attention Calibration

For each output position $i ,$ we standardize the $G$ group scores $c _ { i , 1 : G } ,$ , obtaining $\widetilde { \mathbf { c } } _ { i }$ . We then modify the teacher’s visual attention logits according to counterfactual importance:

$$
z _ { i , b } = \left\{ \begin{array} { l l } { \log ( a _ { \mathrm { t } , i , b } + \epsilon ) + \rho _ { i } \widetilde { c } _ { i , b } , } & { b \in \{ 1 , . . . , G \} , } \\ { \log ( a _ { \mathrm { t } , i , b } + \epsilon ) , } & { b \in \{ \mathrm { p r } , \mathrm { p f } \} , } \end{array} \right.\tag{10}
$$

where ϵ is a small constant for numerical stability. The calibrated teacher policy is

$$
\widehat { \mathbf { a } } _ { \mathrm { t } , i } = \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ) .\tag{11}
$$

A temporal group receives more target mass when its removal changes the teacher’s prediction more than the other groups for the same token, and less target mass when its efect is below average. When the teacher shows little total visual dependence, $\rho _ { i }$ keeps the calibrated policy close to the original attention policy.

We train the student to match this target using

$$
\mathcal { L } _ { \mathrm { a t t } } = \frac { \sum _ { i = 1 } ^ { L } \gamma _ { i } \operatorname { J S D } ( \mathbf { a } _ { \mathrm { s } , i } , \widehat { \mathbf { a } } _ { \mathrm { t } , i } ) } { \sum _ { i = 1 } ^ { L } \gamma _ { i } + \epsilon } ,\tag{12}
$$

where $\gamma _ { i } = \rho _ { i } u _ { i }$ assigns more attention supervision to tokens with obvious visual dependence.

## Counterfactual Token Weighting

The same counterfactual signal is also used to emphasize output tokens whose predictions depend more strongly on the video. We first normalize each $u _ { i }$ by the mean visual dependence within the response:

$$
{ \overline { { u } } } _ { i } = { \frac { u _ { i } } { L ^ { - 1 } \sum _ { j = 1 } ^ { L } u _ { j } + \epsilon } } .\tag{13}
$$

We convert this value into a capped preliminary weight:

$$
\widetilde { w } _ { i } = \operatorname* { m i n } ( 1 + \overline { { u } } _ { i } , w _ { \operatorname* { m a x } } ) .\tag{14}
$$

Finally, we divide all preliminary weights by their response mean so that the average weight remains one:

$$
w _ { i } = \frac { \widetilde { w } _ { i } } { L ^ { - 1 } \sum _ { j = 1 } ^ { L } \widetilde { w } _ { j } } .\tag{15}
$$

The weighted output-level loss is

$$
\mathcal { L } _ { \mathrm { { O P D } } } ^ { \mathrm { { w } } } = \frac { \sum _ { i = 1 } ^ { L } w _ { i } d _ { i } } { \sum _ { i = 1 } ^ { L } w _ { i } } .\tag{16}
$$

CAPD combines output-distribution matching with calibrated attention-policy matching:

$$
\mathcal { L } _ { \mathrm { C A P D } } = \mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { w } } + \lambda _ { \mathrm { a t t } } \mathcal { L } _ { \mathrm { a t t } } .\tag{17}
$$

Table 1: Comparison of CAPD with proprietary models, open-source models, and post-training frameworks, following Video-OPD (Li et al. 2026b). Bold and underline indicate the best and second-best results among the post-training methods.
<table><tr><td rowspan="2">Method</td><td colspan="3">Charades-TimeLens</td><td colspan="3">ActivityNet-TimeLens</td><td colspan="3">QVHighlights-TimeLens</td><td rowspan="2">Avg.</td></tr><tr><td>R@0.3 R@0.5 R@0.7</td><td></td><td></td><td>R@0.3 R@0.5 R@0.7</td><td></td><td></td><td>R@0.3 R@0.5</td><td></td><td>R@0.7</td></tr><tr><td colspan="10">Proprietary Models</td><td></td></tr><tr><td>GPT-40</td><td>60.6</td><td>44.5</td><td>23.5</td><td>55.2</td><td>41.4</td><td>25.8</td><td>69.0</td><td>54.8</td><td>38.5</td><td>45.9</td></tr><tr><td>GPT-5</td><td>59.3</td><td>42.0</td><td>22.0</td><td>57.4</td><td>44.9</td><td>30.4</td><td>72.4</td><td>60.4</td><td>46.4</td><td>48.4</td></tr><tr><td>Gemini-2.0-Flash</td><td>66.4</td><td>53.5</td><td>27.1</td><td>62.9</td><td>54.0</td><td>37.7</td><td>76.2</td><td>66.4</td><td>48.3</td><td>54.7</td></tr><tr><td>Gemini-2.5-Flash</td><td>68.7</td><td>56.1</td><td>30.6</td><td>66.8</td><td>57.5</td><td>41.3</td><td>78.2</td><td>69.4</td><td>55.0</td><td>58.2</td></tr><tr><td>Gemini-2.5-Pro</td><td>74.1</td><td>61.1</td><td>34.0</td><td>72.3</td><td>64.2</td><td>47.1</td><td>84.1</td><td>75.9</td><td>61.1</td><td>63.8</td></tr><tr><td colspan="10">Open-Source Models</td><td></td></tr><tr><td>VideoChat-Flash-7B (Li et al. 2026c)</td><td>60.2</td><td>37.9</td><td>17.8</td><td>35.5</td><td>21.8</td><td>10.5</td><td>45.2</td><td>30.6</td><td>16.7</td><td>30.7</td></tr><tr><td>VideoChat-R1-7B (Li et al. 2025)</td><td>51.9</td><td>30.8</td><td>11.7</td><td>35.0</td><td>23.9</td><td>11.3</td><td>29.3</td><td>19.1</td><td>9.4</td><td>24.7</td></tr><tr><td>Time-R1-7B (Wang et al. 2025b)</td><td>57.9</td><td>32.0</td><td>16.9</td><td>44.8</td><td>31.0</td><td>19.0</td><td>65.8</td><td>51.5</td><td>36.1</td><td>39.4</td></tr><tr><td>TVG-R1-7B (Chen et al. 2025)</td><td>44.5</td><td>23.6</td><td>12.4</td><td>46.7</td><td>31.0</td><td>18.6</td><td>55.8</td><td>41.2</td><td>28.0</td><td>33.5</td></tr><tr><td>VideoChat-R1.5-7B (Yan et al. 2025)</td><td>46.4</td><td>24.0</td><td>10.4</td><td>40.6</td><td>25.3</td><td>16.4</td><td>62.2</td><td>44.5</td><td>28.3</td><td>33.1</td></tr><tr><td>MiMo-VL-7B (Xiaomi LLM-Core Team et al. 2025)</td><td>57.9</td><td>42.6</td><td>20.5</td><td>49.3</td><td>38.7</td><td>22.4</td><td>57.1</td><td>42.6</td><td>28.4</td><td>39.9</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025a)</td><td>58.1</td><td>35.1</td><td>18.2</td><td>47.2</td><td>32.5</td><td>20.2</td><td>55.0</td><td>41.7</td><td>29.3</td><td>37.5</td></tr><tr><td>Qwen3-VL-8B-Instruct (Bai et al. 2025b)</td><td>61.7</td><td>41.5</td><td>23.1</td><td>41.2</td><td>30.7</td><td>20.0</td><td>46.6</td><td>38.2</td><td>29.5</td><td>36.9</td></tr><tr><td colspan="10">Post-Training Frameworks (Based on Qwen3-VL-8B-Instruct)</td></tr><tr><td>GRPO (Shao et al. 2024)</td><td>72.7</td><td>44.4</td><td>27.6</td><td>58.6</td><td>42.7</td><td>32.1</td><td>69.8</td><td>53.0</td><td>41.5</td><td>49.2</td></tr><tr><td>Video-OPD (Li et al. 2026b)</td><td>73.1</td><td>45.8</td><td>32.4</td><td>60.5</td><td>45.6</td><td>35.8</td><td>73.8</td><td>60.3</td><td>50.4</td><td>53.1</td></tr><tr><td>Vanilla OPD (Song and Zheng 2026)</td><td>75.5</td><td>48.5</td><td>30.7</td><td>62.6</td><td>46.6</td><td>36.0</td><td>71.8</td><td>57.2</td><td>46.1</td><td>52.8</td></tr><tr><td>RAL-AD (Li et al. 2026a)</td><td>75.6</td><td>48.6</td><td>31.3</td><td>62.9</td><td>46.6</td><td>36.3</td><td>72.6</td><td>58.1</td><td>47.1</td><td>53.3</td></tr><tr><td>CAPD (Ours)</td><td>76.1</td><td>49.8</td><td>32.0</td><td>64.9</td><td>48.8</td><td>38.3</td><td>74.4</td><td>60.9</td><td>50.7</td><td>55.1</td></tr></table>

## Experiments

## Implementation Details

We initialize the student from Qwen3-VL-8B-Instruct and use the GRPO-post-trained Qwen3-VL-32B model as the frozen teacher. All methods are trained for one epoch on the 2,500 examples provided by Video-OPD (Li et al. 2026b). Videos are sampled at 2 FPS under an 8,192 visual token budget. Training uses full-parameter optimization with a learning rate of $1 \times \mathrm { { 1 0 ^ { - 6 } } }$ , and DeepSpeed ZeRO-3 on 8 × H20 GPUs.

For CAPD, we use G = 8 temporal groups and extract student and teacher policies from the last language-model layer. We use fused AdamW with a learning rate of $\mathrm { { 1 \times 1 0 ^ { - 6 } } }$ , set the random seed to 42, the attention-loss coeficient $\lambda _ { \mathrm { a t t } } = 0 . 2 5 .$ and the maximum token weight $w _ { \mathrm { m a x } } = 5 . \mathrm { R A L - A D }$ uses the on-policy attention-distillation variant introduced by RAL, which shares the same training configuration, directly distills teacher attention. Vanilla OPD disables both attention distillation and counterfactual token weighting. All experiments use the same training examples, configuration, and prompts.

## Benchmarks and Metrics

We evaluate temporal grounding on Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens, which use the corrected temporal annotations introduced by Time-Lens (Zhang et al. 2026b). The evaluation sets contain 3,363, 4,500, and 1,541 queries. For a predicted interval P and ground-truth interval G, temporal IoU is $| P \cap G | / | P \cup G |$ We report Recall at IoU thresholds 0.3, 0.5, and 0.7.

We additionally evaluate Video-MME (Fu et al. 2025), LongVideoBench (Wu et al. 2024), and LVBench (Wang et al. 2025a) to measure whether temporal post-training preserves general video understanding. Video-MME is a benchmark spanning short, medium and long videos. LongVideoBench contains 3,763 videos in four duration ranges from 8 seconds to 60 minutes. LVBench contains 103 videos, with an average video length of 68 minutes.

## Quantitative Analysis

Performance Comparison with Other Methods We first compare CAPD with existing video MLLMs and temporalgrounding post-training methods on three TimeLenscorrected benchmarks (Zhang et al. 2026b) in Table 1. The results of Video-OPD (Li et al. 2026b) are included as reference, while Vanilla OPD, RAL-AD, and CAPD are trained for one epoch and evaluated under the same settings. From these results, we can first observe that post-training substantially improves the temporal grounding ability of video MLLMs. However, the improvements of existing methods vary across datasets and IoU thresholds, indicating that retrieving a relevant segment does not necessarily produce accurate temporal boundaries. Compared with Vanilla OPD, RAL-AD achieves better performance by aligning the raw teacher attention. More importantly, CAPD reaches an average of 55.1, improving over GRPO by 5.9 points (12.0% relative), and achieves the best average performance among the compared post-training methods. It also improves most recall metrics, especially under the stricter IoU threshold. These results support the efectiveness of counterfactual calibration for learning decision-relevant temporal evidence. Across the three benchmarks, the gains are not concentrated in a single dataset. The larger improvements at stricter IoU thresholds indicate that CAPD not only retrieves an overlapping moment, but also refines the temporal extent of the prediction.

Table 2: Comparison of Qwen3-VL-8B-Instruct and its posttrained variants on general video understanding benchmarks.
<table><tr><td>Method</td><td>Video-MME</td><td>LongVideoBench</td><td>LVBench</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>70.8</td><td>64.0</td><td>53.0</td></tr><tr><td>+ Vanilla OPD</td><td>71.1</td><td>64.0</td><td>52.6</td></tr><tr><td>+ RAL-AD</td><td>71.2</td><td>63.8</td><td>52.4</td></tr><tr><td>+ CAPD</td><td>71.1</td><td>64.0</td><td>52.5</td></tr></table>

![](images/be0d5f811afa94021a220ed28e8b150d33148e279bde7a18704695bb1dc4f67a.jpg)  
Figure 3: Analysis of evidence coverage, preference gap, group precision, and selection stability under diferent numbers of temporal groups.

This suggests that counterfactual calibration helps the model focus on the evidence needed to identify the queried event and determine its boundaries.

We evaluate the general video understanding capability of CAPD, Vanilla OPD (Song and Zheng 2026), RAL-AD (Li et al. 2026a), and the base model on Video-MME (Fu et al. 2025), LongVideoBench (Wu et al. 2024), and LVBench (Wang et al. 2025a), as reported in Table 2. These benchmarks cover a broad range of general video understanding tasks, providing a complementary evaluation to the temporal grounding benchmarks above. CAPD maintains comparable performance to Qwen3-VL-8B (Bai et al. 2025b) and the baselines across all three benchmarks, indicating that the counterfactual distillation objective does not interfere with the model’s general video understanding capability. This result suggests that CAPD’s training signal is sufficiently targeted to temporal grounding, leaving the broader video comprehension abilities of the base model intact.

Training Eficiency Under the same settings, CAPD requires 10.2 hours of training compared with 6.4 hours for OPD, corresponding to a 1.60× training cost. The additional computation mainly comes from the teacher forward passes used for counterfactual temporal analysis. Nevertheless, the peak memory on GPU increases by only 5.21 GiB, allowing CAPD to remain trainable under the same setting as OPD.

Ablation Study In Table 3, we ablate the two components of CAPD, i.e., counterfactually calibrated attention supervision and counterfactual token weighting, with OPD as the reference. Both components improve over OPD, confirming that counterfactual influence provides useful supervision for both evidence selection and output generation. Token-Only produces a larger gain than Attention-Only, indicating that weighting decision-sensitive output tokens provides a stronger training signal. Attention-Only also improves performance, showing that aligning the student with the calibrated temporal evidence policy is beneficial, though insuficient alone. Combining both components achieves the best performance across all benchmarks, confirming their complementary roles, where token weighting strengthens supervision on decision-sensitive outputs while calibrated attention improves the temporal evidence used to generate them.

Table 3: Component ablation on TimeLens. C-TL, A-TL, and QV-TL are the means of R@0.3, R@0.5, and R@0.7 on each benchmark; Avg. averages all recalls.
<table><tr><td>Method</td><td>C-TL</td><td>A-TL</td><td>QV-TL</td><td>Avg.</td></tr><tr><td>OPD</td><td>51.6</td><td>48.4</td><td>58.4</td><td>52.8</td></tr><tr><td>Attention-Only</td><td>51.8</td><td>48.6</td><td>59.3</td><td>53.3</td></tr><tr><td>Token-Only</td><td>52.0</td><td>49.4</td><td>60.3</td><td>53.9</td></tr><tr><td>CAPD</td><td>52.6</td><td>50.7</td><td>62.0</td><td>55.1</td></tr></table>

Table 4: Parameter ablation on TimeLens. C-TL, A-TL, and QV-TL are the mean recalls on each benchmark; Avg. averages all nine recall metrics.
<table><tr><td>Parameter</td><td>Value</td><td>C-TL A-TL</td><td>QV-TL</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Temporal Group (G)</td><td>2 4 8 16</td><td>52.2 49.8 51.9 49.5 52.6 50.7 52.0 49.5</td><td>60.7 60.5 62.0 60.4</td><td>54.2 54.0 55.1 53.9</td></tr><tr><td>Attention Weight  $\left( \lambda _ { \mathrm { a t t } } \right)$ </td><td>0 0.1 0.25 0.5 0.75 1</td><td>52.0 49.4 52.4 49.6 52.6 50.7 52.0 49.6 52.1 49.3 51.9 49.0</td><td>60.3 60.9 62.0 60.4 59.7 59.5</td><td>53.9 54.3 55.1 54.0 53.8 53.5</td></tr></table>

In Table 4, we study the efects of the number of temporal groups G and the attention-loss coeficient $\lambda _ { \mathrm { a t t } }$ . The analysis in Figure 3 reveals a clear trade-of in temporal grouping. Increasing G improves group precision by isolating finer temporal regions, but reduces evidence coverage, preference gap, and selection stability because a complete event may be divided across multiple groups. Coarser grouping better preserves complete and stable event evidence, although each group contains more irrelevant context. This trade-of is reflected in the downstream results, where performance does not improve monotonically with finer grouping and G = 8 achieves the best overall result. Meanwhile, performance first improves and then declines as $\lambda _ { \mathrm { a t t } }$ increases. A moderate attention weight provides useful evidence supervision while preserving the contribution of output-level distillation, whereas an overly large weight can disturb this balance. We therefore use G = 8 and $\bar { \lambda } _ { \mathrm { a t t } } = 0 . 2 5$ , which provide the best balance between precise evidence selection, complete event coverage, and output-level supervision.

![](images/c5b4020d0f4b06edf0dbe1d694c922fec3368a25e7a4bdfe50b0b3b20733e3e3.jpg)  
Figure 4: Visualized results of CAPD. The top panel shows the temporal boundary predictions of diferent methods on a long form video query, alongside the ground-truth interval, showcasing CAPD’s ability to produce compact and accurate temporal boundaries. The GREEN bar denotes the ground truth, the GRAY segments are OPD’s predictions, the BLUE line represents RAL-AD, and the PURPLE bar indicates CAPD’s output. The bottom-left and bottom-right panels contrast raw teacher attention with counterfactual (CF) influence across temporal groups G0–G7 for two additional queries, demonstrating that while raw attention concentrates on late or visually salient groups, CF influence more precisely identifies the relevant temporal segments.

Overall, the component and parameter studies confirm that counterfactual token weighting and calibrated attention provide complementary output- and evidence-level supervision.

## Qualitative Analysis

To better understand how counterfactual calibration improves temporal grounding, we visualize both the predicted temporal boundaries and the corresponding temporal supervision signals in Figure 4. In the upper panel, OPD mainly captures the visually salient middle of the target event, while RAL-AD recovers its onset but still terminates the prediction before the action is complete. The continuous video frames show that the queried action remains ongoing after both predictions have ended. In contrast, CAPD recovers the decision-relevant event tail and predicts a more complete interval that closely matches the ground-truth boundary. This comparison shows that attention-policy transfer improves evidence localization, while counterfactual calibration further corrects the boundary decision.

The lower panel compares the supervision signals. RAL-AD distills raw teacher attention, which can assign substantial mass to visually salient but irrelevant content. In the left example, attention focuses on the final scene, whereas counterfactual influence highlights the groups containing the queried action. A similar pattern appears in the right example, where attention peaks on an irrelevant group while counterfactual influence concentrates on the groups covering the actual tool-use action. These examples show that counterfactual influence better identifies temporal evidence that afects the teacher’s prediction and therefore provides a more reliable signal for calibrating attention supervision.

Together, the two panels connect temporal evidence selection with boundary generation. Raw attention shows where the teacher attends, whereas counterfactual influence identifies the temporal groups whose removal changes its output. CAPD preserves the teacher’s attention structure while increasing the target mass of influential groups, helping the student retain the evidence needed to predict complete event boundaries. This calibration more directly connects the selected temporal evidence to boundary generation. The resulting improvement in start and end timestamps is consistent with the larger gains at stricter IoU thresholds in Table 1, supporting the CAPD design.

## Conclusion

In this paper, we introduce CAPD, an on-policy distillation method for temporal grounding in video MLLMs. CAPD estimates the counterfactual influence of temporal groups to calibrate teacher attention and output-level distillation. Experiments on TimeLens benchmarks demonstrate that CAPD improves temporal localization over OPD and RAL-AD while preserving general video understanding ability. These results show that CAPD provides more efective supervision for temporal grounding by integrating temporal evidence into on-policy distillation. Despite its promising results, CAPD uses fixed temporal groups, which may not fully capture events with diverse durations and temporal structures. We will explore event-adaptive grouping with finer partitions for short actions and coarser partitions for extended events. CAPD requires multiple teacher forward passes, increasing post-training cost over OPD. Since the teacher is frozen, batched inference can reduce this overhead.

## References

Agarwal, R.; Vieillard, N.; Zhou, Y.; Stanczyk, P.; Ramos, S.; Geist, M.; and Bachem, O. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In The Twelfth International Conference on Learning Representations.

Bai, S.; et al. 2025a. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Bai, S.; et al. 2025b. Qwen3-VL Technical Report. arXiv:2511.21631.

Bao, P.; Kong, C.; Yang, S.; Shao, Z.; Jiang, X.; Ng, B. P.; Er, M. H.; and Kot, A. 2025. Vid-Group: Temporal Video Grounding Pretraining from Unlabeled Videos in the Wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20541–20550.

Bousselham, W.; Kuehne, H.; and Schmid, C. 2026. VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26209–26218.

Chen, R.; Luo, T.; Fan, Z.; Zou, H.; Feng, Z.; Xie, G.; Zhang, H.; Wang, Z.; Liu, Z.; and Zhang, H. 2025. Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, 983–992. Association for Computational Linguistics.

Fang, L.; Yu, X.; Cai, J.; Chen, Y.; Wu, S.; Liu, Z.; Yang, Z.; Lu, H.; Gong, X.; Liu, Y.; Ma, T.; Ruan, W.; Abbasi, A.; Zhang, J.; Wang, T.; Latif, E.; You, W.; Jiang, H.; Liu, W.; Zhang, W.; Kolouri, S.; Zhai, X.; Zhu, D.; Zhong, W.; Liu, T.; and Ma, P. 2026. Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends, Challenges, and Future Directions. arXiv:2504.14772.

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; Chen, P.; Li, Y.; Lin, S.; Zhao, S.; Li, K.; Xu, T.; Zheng, X.; Chen, E.; Shan, C.; He, R.; and Sun, X. 2025. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24108–24118.

Gao, J.; Sun, C.; Yang, Z.; and Nevatia, R. 2017. TALL: Temporal Activity Localization via Language Query. In Proceedings of the IEEE International Conference on Computer Vision, 5267–5275.

Gu, Y.; Dong, L.; Wei, F.; and Huang, M. 2024. MiniLLM: Knowledge Distillation of Large Language Models. In The Twelfth International Conference on Learning Representations.

Guo, C.; Mo, X.; Nie, Y.; Ma, F.; Xu, X.; and Long, C. 2026. TAR: Temporal Anchor-Constrained Reasoning for Video Temporal Grounding. In European Conference on Computer Vision.

Huang, B.; Wang, X.; Chen, H.; Song, Z.; and Zhu, W. 2024. VTimeLLM: Empower LLM to Grasp Video Moments. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14271–14280.

Jain, S.; and Wallace, B. C. 2019. Attention is not Explanation. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, 3543–3556.

Jang, I.; Yeom, J.; Yeo, J.; Lim, H.; and Kim, T. 2026. Stable On-Policy Distillation through Adaptive Target Reformulation. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, 42217–42227. Association for Computational Linguistics.

Ko, J.; Kim, S.; Chen, T.; and Yun, S.-Y. 2024. DistiLLM: Towards Streamlined Distillation for Large Language Models. In Proceedings of the 41st International Conference on Machine Learning.

Lei, J.; Berg, T. L.; and Bansal, M. 2021. Detecting Moments and Highlights in Videos via Natural Language Queries. In Advances in Neural Information Processing Systems, volume 34, 11846–11858.

Li, A. C.; Tian, Y.; Chen, B.; Pathak, D.; and Chen, X. 2024. On the Surprising Efectiveness of Attention Transfer for Vision Transformers. In Advances in Neural Information Processing Systems.

Li, B.; Ni, J.; Qu, C.; Miao, I.; Yang, L.; Fu, X.; Chen, M.; and Cheng, D. Z. 2026a. Reinforced Attention Learning. arXiv:2602.04884.

Li, J.; Xie, J.; Qian, L.; Zhu, L.; Tang, S.; Wu, F.; Yang, Y.; Zhuang, Y.; and Wang, X. E. 2022. Compositional Temporal Grounding with Structured Variational Cross-Graph Correspondence Learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3032–3041.

Li, J.; Yin, H.; Xu, H.; Xu, B.; Tan, W.; He, Z.; Ju, J.; Luo, Z.; and Luan, J. 2026b. Video-OPD: Eficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR.

Li, X.; Wang, Y.; Yu, J.; Zeng, X.; Zhu, Y.; Huang, H.; Gao, J.; Li, K.; He, Y.; Wang, C.; Qiao, Y.; Wang, Y.; and Wang, L. 2026c. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. In The Fourteenth International Conference on Learning Representations.

Li, X.; Yan, Z.; Meng, D.; Dong, L.; Zeng, X.; He, Y.; Wang, Y.; Qiao, Y.; Wang, Y.; and Wang, L. 2025. VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning. arXiv:2504.06958.

Lin, K. Q.; Zhang, P.; Chen, J.; Pramanick, S.; Gao, D.; Wang, A. J.; Yan, R.; and Shou, M. Z. 2023. UniVTG: Towards Unified Video-Language Temporal Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2794–2804.

Moon, W.; Hyun, S.; Park, S.; Park, D.; and Heo, J.-P. 2023. Query-Dependent Video Representation for Moment Retrieval and Highlight Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23023–23033.

Mun, J.; Cho, M.; and Han, B. 2020. Local-Global Video-Text Interactions for Temporal Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10810–10819.

Qian, L.; Li, J.; Wu, Y.; Ye, Y.; Fei, H.; Chua, T.-S.; Zhuang, Y.; and Tang, S. 2024. Momentor: Advancing Video Large Language Model with Fine-Grained Temporal Reasoning. In Proceedings ofthe 41st International Conference on Machine Learning, 41340–41356.

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14313–14323.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Song, M.; and Zheng, M. 2026. A Survey of On-Policy Distillation for Large Language Models. arXiv:2604.00626.

Wang, Q.; Chen, S.; and Shen, Y. 2025. CausalVTG: Towards Robust Video Temporal Grounding via Causal Inference. In Advances in Neural Information Processing Systems.

Wang, W.; He, Z.; Hong, W.; Cheng, Y.; Zhang, X.; Qi, J.; Ding, M.; Gu, X.; Huang, S.; Xu, B.; Dong, Y.; and Tang, J. 2025a. LVBench: An Extreme Long Video Understanding Benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 22958–22967.

Wang, Y.; Meng, X.; Liang, J.; Wang, Y.; Liu, Q.; and Zhao, D. 2024. HawkEye: Training Video-Text LLMs for Grounding Text in Videos. arXiv:2403.10228.

Wang, Y.; Wang, Z.; Xu, B.; Du, Y.; Lin, K.; Xiao, Z.; Yue, Z.; Ju, J.; Zhang, L.; Yang, D.; Fang, X.; He, Z.; Luo, Z.; Wang, W.; Lin, J.; Luan, J.; and Jin, Q. 2025b. Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding. In Advances in Neural Information Processing Systems.

Wiegrefe, S.; and Pinter, Y. 2019. Attention is not not Explanation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 11–20.

Wu, H.; Li, D.; Chen, B.; and Li, J. 2024. LongVideoBench: A Benchmark for Long-Context Interleaved Video-Language Understanding. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, volume 37.

Wu, J.; Liu, W.; Liu, Y.; Liu, M.; Nie, L.; Lin, Z.; and Chen, C. W. 2026. A Survey on Video Temporal Grounding with Multimodal Large Language Model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(2): 1521–1541.

Xiaomi LLM-Core Team; et al. 2025. MiMo-VL Technical Report. arXiv:2506.03569.

Xu, Y.; Sang, H.; Zhou, Z.; He, R.; Wang, Z.; and Geramifard, A. 2026. TIP: Token Importance in On-Policy Distillation. arXiv:2604.14084.

Xu, Y.; Xu, W.; and Miao, Z. 2025. Counterfactual Contrastive Learning for Weakly Supervised Temporal Sentence Grounding. Neurocomputing, 624: 129508.

Yan, Z.; He, Y.; Li, X.; Yue, Z.; Zeng, X.; Wang, Y.; Qiao, Y.; Wang, L.; and Wang, Y. 2025. VideoChat-R1.5: Visual Test-Time Scaling to Reinforce Multimodal Reasoning by Iterative Perception. In Advances in Neural Information Processing Systems, volume 38.

Yang, C.; Lu, W.; Zhu, Y.; Wang, Y.; Chen, Q.; Gao, C.; Yan, B.; and Chen, Y. 2024. Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application. arXiv:2407.01885.

Yang, Z.; Yu, Y.; Zhao, Y.; Lu, S.; and Bai, S. 2025. Time-Expert: An Expert-Guided Video LLM for Video Temporal Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 24286–24296.

Zhai, M.; Li, C.; Jing, C.; and Wu, Y. 2022. Synthesizing Counterfactual Samples for Overcoming Moment Biases in Temporal Video Grounding. In Pattern Recognition and Computer Vision, 436–448. Springer.

Zhang, D.; Yang, Z.; Janghorbani, S.; Han, J.; Ressler II, A.; Qian, Q.; Lyng, G. D.; Batra, S. S.; and Tillman, R. E. 2026a. Fast and Efective On-Policy Distillation from Reasoning Prefixes. In Findings of the Association for Computational Linguistics: ACL 2026, 25553–25569. Association for Computational Linguistics.

Zhang, H.; Sun, A.; Jing, W.; and Zhou, J. T. 2020a. Spanbased Localizing Network for Natural Language Video Localization. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, 6543–6554.

Zhang, J.; Guo, Y.; Potamias, R. A.; Deng, J.; Xu, H.; and Ma, C. 2025. VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 24203–24213.

Zhang, J.; Wang, T.; Ge, Y.; Ge, Y.; Li, X.; and Wang, L. 2026b. TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10419–10429.

Zhang, S.; Peng, H.; Fu, J.; and Luo, J. 2020b. Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 12870–12877.

Zhao, S.; Xie, Z.; Liu, M.; Huang, J.; Pang, G.; Chen, F.; and Grover, A. 2026. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. In Proceedings of the 43rd International Conference on Machine Learning.