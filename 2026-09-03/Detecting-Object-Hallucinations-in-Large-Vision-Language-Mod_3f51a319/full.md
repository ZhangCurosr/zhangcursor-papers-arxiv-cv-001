# Detecting Object Hallucinations in Large Vision-Language Models via Cross-Modal Attention Drifts and Mask-Based Verification

Xuanbing Wen<sup>1</sup>, Boxu Chen<sup>1</sup>, Le Yang<sup>1</sup>, Jiakai Wang<sup>2</sup>, Zhengyu Zhao<sup>1</sup>, Chenhao Lin<sup>1</sup>, Chao Shen<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University, Xi’an 710049, China

<sup>2</sup>Zhongguancun Laboratory, Beijing 100194, China

22009200389@stu.xidian.edu.cn, chenboxu@stu.xjtu.edu.cn, yangle15@xjtu.edu.cn, jk\_buaa\_scse@buaa.edu.cn, zhengyu.zhao@xjtu.edu.cn, linchenhao@xjtu.edu.cn, chaoshen@mail.xjtu.edu.cn

## Abstract

Despite recent advances in large vision-language models (LVLMs), object hallucination remains a major barrier to their reliable deployment. Existing detection methods often characterize visual grounding using attention from individual layers, leaving its evolution across layers underexplored. We propose CADMP, a lightweight object hallucination detection framework that combines adjacent-layer cross-modal attention drift with prediction sensitivity to targeted visual masking. During decoding, CADMP quantifies distributional changes between consecutive cross-modal attention maps to capture abrupt transitions in visual grounding. It then selects the transition with the largest drift, locates the corresponding visually relevant regions, and measures the change in prediction probability after masking these regions. These two signals provide complementary evidence: attention drift characterizes the stability of internal visual grounding, while probability variation verifies whether a prediction truly depends on the identified visual evidence. A lightweight detector integrates both signals to identify hallucinated predictions. Experiments on multiple benchmarks and representative open-source LVLMs demonstrate that CADMP achieves consistently competitive detection performance. Ablation studies further confirm the complementary contributions of adjacent-layer drift modeling and mask-based grounding verification.

## Introduction

Large Vision-Language Models (LVLMs) have made substantial progress in multimodal understanding and generation (Li et al. 2023a; Zhu et al. 2023), enabling applications in areas such as autonomous driving (Wei et al. 2024) and robotics (Kim et al. 2024). Nevertheless, their reliability is still limited by object hallucination (OH), where a model mentions objects that are absent from or unsupported by the input image (Bai et al. 2024). Such errors compromise the factual consistency of model output and may pose serious risks in safety-critical applications.

Existing hallucination assessment techniques obtain evidence from either external resources or internal model signals. External evaluation frameworks use reference annotations, auxiliary evaluators, or additional reasoning pipelines to assess image-text consistency (Fu et al. 2023; Wang et al.

![](images/9b0770de2b81f8503904e88e5219c4f239ceb136109e74c0ed30b1b701548b02.jpg)  
Figure 1: Overview of CADMP, detecting OHs by modeling cross-modal attention drifts and verifying visual grounding through drifted attention-guided masks.

2023; Jing et al. 2024). Although efective for ofline evaluation, they may require additional annotations, models, or computational resources. Instead, internal detection methods characterize predictions using signals available within the LVLM, including predictive uncertainty (Malinin and Gales 2021), hidden representations (Park and Li 2025; Jiang et al. 2025b), and cross-modal attention (Zhang et al. 2025). Among these signals, cross-modal attention is particularly relevant to visual grounding because it characterizes the interactions between output and visual tokens.

Prior attention-based studies have used cross-modal attention patterns for hallucination detection (Zhang et al. 2025), highlighted the importance of intermediate layers in visual processing (Jiang et al. 2025b), and modeled interlayer changes in aggregated image-token attention for hallucination mitigation (Xu et al. 2025). However, changes in the full attention distribution on visual tokens between consecutive layers remain underexplored for hallucination detection. Our analysis shows that grounded and hallucinated predictions exhibit distinct layer-wise drift profiles, motivating us to treat adjacent-layer attention drift as a diagnostic signal. Based on this observation, we propose CADMP, which combines attention drift modeling with mask-based probability verification. As illustrated in Figure 1, CADMP computes the KL divergence between consecutive cross-modal attention distributions to obtain a layer-wise drift profile. It then uses the attention map immediately before the maximumdrift transition to identify and mask relevant visual regions, and measures the resulting change in the probability of the same token. The drift profile captures the evolution of visual grounding, while the probability change estimates the prediction’s dependence on visual evidence.

CADMP feeds the layer-wise drift profile, the prediction probabilities before and after masking, and their diference into a small MLP detector, while keeping the target LVLM frozen and training only the detector. Experiments on three representative open-source LVLMs across multiple hallucination benchmarks demonstrate strong and consistent detection performance. The main contributions of this work are summarized as follows:

• We reveal distinct adjacent-layer attention drift patterns between grounded and hallucinated predictions, establishing attention drift as an efective detection signal.

• We propose CADMP, which integrates layer-wise attention drift with mask-based probability verification using a lightweight detector without retraining the LVLM.

• Experiments on three LVLMs and four benchmarks show that CADMP achieves the best average ACC–AUROC in eight of nine primary settings, with ablations validating both components.

## Related Work

Large Vision-Language Models (LVLMs) Large Vision-Language Models (LVLMs) integrate computer vision and natural language processing to enable multimodal understanding and generation (Liu et al. 2023; Bai et al. 2025a; Wang et al. 2024b; Lu et al. 2024). An LVLM typically consists of three components: a vision encoder, a language model, and a connector. The vision encoder, such as ViT, extracts visual features from images and represents them as dense embeddings. The language model, usually a pretrained large language model such as LLaMA (Touvron et al. 2023), Qwen (Bai et al. 2023), or Vicuna, is responsible for language understanding and generation. The connector (Li et al. 2023b; Liu et al. 2023, 2024b) bridges the visual and textual modalities by transforming visual features into representations that can be processed by the language model. Through the collaboration of these components, LVLMs achieve strong multimodal perception and reasoning capabilities.

Hallucination in LVLMs Hallucination in LVLMs refers to cases where the generated output is inconsistent with visual input or factual reality (Liu et al. 2024c). Such errors may arise from several factors, including conflicting training data (Delétang et al. 2023), overly strong language priors (Liu et al. 2023), and the loss or weakening of visual information during processing (Favero et al. 2024). Existing studies generally categorize hallucinations into three types: object hallucination (Li et al. 2023c), attribute hallucination, and relational hallucination (Jing et al. 2024). Among them, object hallucination is the most directly related to our setting, as it occurs when the model mentions objects that do not appear in the image. By contrast, attribute hallucination involves incorrect descriptions of an object’s properties, while relational hallucination concerns errors in the spatial or semantic relationships between objects.

Detecting Hallucination in LVLMs Hallucination detection in LVLMs has attracted increasing attention in recent years (Wang et al. 2024a; Augustin, Neuhaus, and Hein 2025; Arteaga, Schön, and Pielawski 2024). Although many existing studies focus on hallucination mitigation (Hu et al. 2023; You et al. 2023), accurately detecting hallucinations is equally important for improving the reliability of LVLMs. Existing detection methods mainly fall into three categories. First, some methods estimate the uncertainty of model outputs and use it as a signal for hallucination detection (Zhou et al. 2024). Second, representation-based methods such as GLSim (Park and Li 2025) and IC (Jiang et al. 2025a) employ Logits Lens-style techniques to project hidden representations and measure their similarity to target objects for hallucination detection. Third, attention-based methods such as DHCP (Zhang et al. 2025) detect hallucinations by modeling cross-modal attention patterns in internal layers. Different from these methods, our work focuses on cross-modal attention drifts during decoding and examines whether the model genuinely grounds its predictions in visual evidence, thereby enabling more precise hallucination detection.

## Method

In this section, we introduce CADMP, an object hallucination detection framework that keeps the target LVLM unchanged. As shown in Figure 2, CADMP computes KL divergences between consecutive cross-modal attention distributions to capture layer-wise grounding drift. It then uses the attention map preceding the maximum-drift transition to mask relevant visual regions and measures the resulting probability change. Grounded predictions are expected to be more sensitive to removing visual evidence, whereas hallucinated predictions typically show smaller changes. These complementary signals are combined for hallucination detection.

## Adjacent-Layer Cross-Modal Attention Drifts

We first investigate cross-modal attention drifts to characterize how visual grounding evolves across decoding layers. Based on previous findings that cross-modal attention patterns are important for hallucination detection (Zhang et al. 2025), we further evaluate if hallucinated predictions are more likely to exhibit abrupt changes in their attention distributions over visual tokens, and grounded predictions tend to maintain more consistent cross-layer attention patterns.

Given an image-text input pair, we focus on the crossmodal attention from the currently generated token to the visual tokens. Let N denote the number of visual tokens (e.g., N = 576 for LLaVA-1.5-7B), H the number of attention heads, and L the number of decoder layers (e.g., $L = 3 2$ for LLaVA-1.5-7B). For each layer i, we average the crossmodal attention weights over all heads and obtain an attention vector on the visual tokens:

![](images/ab5db81287ab4723de4299847aeaaf70d5c6c30094a4e0307699699117a7b131.jpg)  
Figure 2: Overview of the proposed framework. (a) Extract KL divergence: Compute adjacent-layer KL divergences and the original target-token probability. (b) Visual Masking: Use cross-modal attention to identify key image regions and generate masked images. (c) Hallucination Detector: Feed the extracted features into a MLP model for hallucination detection.

$$
A _ { i } = \left[ a _ { i } ^ { ( 1 ) } , \ldots , a _ { i } ^ { ( N ) } \right] , a _ { i } ^ { ( m ) } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { A t t n } _ { i , h } ( y , v _ { m } ) .
$$

$\mathrm { A t t n } _ { i , h } ( y , v _ { m } )$ denotes the attention weight assigned by the query position whose logits predict $y$ to the m-th visual token $v _ { m }$ at layer i and head $\bar { h . }$ By collecting the attention vectors from all layers, we obtain a layer-wise attention sequence:

$$
\mathcal { A } = \{ A _ { 1 } , A _ { 2 } , . . . , A _ { L } \} .
$$

After normalization, $A _ { i }$ can be viewed as a discrete probability distribution over the visual-token index space. Let $p _ { i } ( x )$ denote the normalized attention probability assigned to token index x at layer $i ,$ and let $p _ { i - 1 } ( x )$ denote the corresponding distribution at layer $i - 1$ . To quantify the crosslayer attention drifts, we measure the discrepancy using the Kullback–Leibler (KL) divergence:

$$
K _ { i } = D _ { \mathrm { K L } } ( A _ { i } \| A _ { i - 1 } ) = \sum _ { x } p _ { i } ( x ) \log { \frac { p _ { i } ( x ) } { p _ { i - 1 } ( x ) } } .
$$

Here, $K _ { i }$ measures how much the attention distribution in layer i deviates from that in layer i − 1 over the visualtoken space. A larger $K _ { i }$ indicates a stronger redistribution of attention across visual tokens between adjacent layers, while a smaller value suggests that the cross-modal attention remains relatively stable.

By computing the KL divergence across all consecutive layer pairs, we obtain (L − 1) dimensional drift features,

$F _ { \mathrm { d r i f t } } = [ K _ { 2 } , K _ { 3 } , \dots , K _ { L } ]$ . From analytical experiments in Figure 3, we see whether layer-wise KL divergence provides an informative signal for hallucination detection, we analyze the drift features extracted from Qwen2.5-VL-7B on 5,000 images from MSCOCO dataset. The blue and red curves show the average attention drift for grounded and hallucinated objects. Hallucinated objects show stronger attention drift in middle layers, particularly around layers 8-16, suggesting that adjacent-layer attention drift can provide a useful signal of potential instability in visual grounding. Therefore, We use cross-modal attention drifts to capture layerwise variations in the model’s internal visual grounding and treat them as diagnostic signals for hallucination detection.

![](images/cd9b64d085ba0cfb9fde40c0d7eb2a6980677631ad89135c967f2763ced9d658.jpg)  
Figure 3: Average cross-modal attention drift across layers for grounded and hallucinated samples.

## Drift-guided Adaptive Masking

The preceding analysis establishes adjacent-layer attention drift as an informative signal for hallucination detection. However, drift alone characterizes how visual attention is redistributed across layers, but does not reveal whether a prediction depends on the attended visual content. Since the drift is defined over visual-token attention distributions, the transition with the largest drift provides a principled cue for targeted visual perturbation. We use the attention map immediately before this transition, which represents the model’s visual allocation before its strongest reorganization, to locate and mask highly attended regions. We then measure the probability change of the same output token. A larger decrease suggests a stronger dependence on the removed visual content, whereas a smaller change is more consistent with the reliance on language priors. Thus, drift-guided masking complements internal attention dynamics with a counterfactual measure of visual dependency.

Given an input image v, a text prompt t, a generated prefix $Y _ { < t }$ , and a target token $y ,$ we first compute its original prediction probability:

$$
P _ { \mathrm { o r i g } } = P ( y \mid v , t , Y _ { < t } ) .
$$

Let $K _ { i }$ denote the attention drift between layers $i - 1$ and i, as defined in the preceding section, we select the layer before the maximum-drift transition as the masking layer:

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i \in \{ 2 , . . . , L \} } K _ { i } , \qquad l _ { \mathrm { m a s k } } = i ^ { * } - 1 .
$$

The attention map at $l _ { \mathrm { m a s k } }$ is then used to identify the visual regions most relevant to the current prediction.

After selecting $l _ { \mathrm { m a s k } }$ , we use its attention map to identify the visual regions most relevant to the prediction of the target token. Let $A ( i , j )$ denote the attention value at spatial location $( i , j )$ . We then generate a binary mask by thresholding the attention map:

$$
M ( i , j ) = \left\{ \begin{array} { l l } { 1 , } & { A ( i , j ) \geq \mathrm { P e r c e n t i l e } ( A , 1 0 0 - \tau ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

In our experiments, we set $\tau = 3 0$ , so we mask pixels at or above the 70th percentile of $A ( i , j )$ . We replace the selected regions with a neutral gray value of 128 while keeping the remaining regions unchanged, thereby obtaining the masked image. Detailed masking procedures are provided in the $\mathsf { A p - }$ pendix.

The masked image, together with the original prompt t, is then fed back into the model, and the prediction probability of the same answer token is recomputed as

$$
P _ { \mathrm { { m a s k e d } } } = P ( y \mid v _ { \mathrm { { m a s k e d } } } , t , Y _ { < t } ) .
$$

Importantly, we compare the probability of the same token under the original and masked visual conditions, which provides a direct measure of confidence change after removing the relevant visual evidence. Finally, we define the probability variation as

$$
\Delta P = P _ { \mathrm { o r i g } } - P _ { \mathrm { m a s k e d } } ,
$$

## Hallucination Detection

Leveraging the diferences in cross-modal attention drifts and probability variations between grounded and hallucinated samples, we design a detector to verify the factuality of the answer produced by the model. We train a lightweight MLP detector by feeding it the extracted KL-divergence profile together with the prediction probabilities before and after masking. For example, LLaVA-1.5-7B contains 32 layers, producing 31 adjacent-layer KL-divergence values. We then put the 31-dimensional feature vector, together with $P _ { \mathrm { o r i g } } ,$ $P _ { \mathrm { { m a s k e d } } }$ and $\Delta P$ , into the MLP detector. The detector outputs a hallucination score for each object prediction, which is converted into a probability through a sigmoid function.

To address the class imbalance between hallucinated and grounded samples, we employ weighted random sampling during training, assigning each sample a weight inversely proportional to the size of its class.

## Experiments

Datasets To evaluate the efectiveness and generalizability of the proposed method for hallucination detection in LVLMs, we conduct experiments on four datasets. Specifically, we use POPE (Li et al. 2023c) and Pascal VOC (Everingham et al. 2010) to evaluate object-existence hallucination detection. In addition, we construct a COCO-Caption setting based on MSCOCO (Lin et al. 2014), where LVLMgenerated captions are annotated using the CHAIR metric (Rohrbach et al. 2018) to evaluate descriptive object hallucination detection. We also conduct experiments on AM-BER (Wang et al. 2023), which covers object, attribute, and relation hallucinations. More details about the datasets are provided in the supplementary materials.

Evaluation Metrics To evaluate detection performance, we report two metrics: (1) the area under the receiver operating characteristic curve (AUROC), (2) classification accuracy (ACC). These metrics are commonly used in OH detection and provide complementary views of model performance (Park and Li 2025; Hoang-Xuan et al. 2026).

Implementation. We evaluate our method on three representative LVLMs: LLaVA-1.5-7B (Liu et al. 2023, 2024b), Qwen2.5-VL-7B (Bai et al. 2025b), and InternVL-2.5- 4B (Chen et al. 2024). Also, we compare the proposed CADMP with recent OH detection baselines, including Internal Confidence (IC) (Jiang et al. 2025a), Global-Local Similarity (GLSim) (Park and Li 2025), Negative Log-likelihood (NLL) (Zhou et al. 2024), a token probability-based uncertainty method denoted as Entropy (Malinin and Gales 2021), summed Visual Attention Ratio (SVAR) (Jiang et al. 2025b), and the cross-modal attention pattern-based method DHCP (Zhang et al. 2025). For fair comparison, all methods are evaluated on the same training and test splits under the same experimental protocol. More details are provided in the supplementary materials.

## Experimental results

Results on OH Detection As shown in Table 1, CADMP achieves the best result in 16 of the 18 model–dataset–metric combinations, demonstrating consistent efectiveness across architectures and evaluation settings. The gains are particularly pronounced on COCO-Caption, where CADMP improves AUROC over the strongest baseline by 5.6, 8.4, and 4.6 percentage points for LLaVA-1.5-7B, Qwen2.5-VL-7B, and

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">COCO-Cap.</td><td colspan="2">POPE</td><td colspan="2">VOC</td></tr><tr><td>ACC</td><td>AUROC</td><td>ACC</td><td>AUROC</td><td>ACC</td><td>AUROC</td></tr><tr><td rowspan="7">LLaVA-1.5-7B</td><td>IC (ICLR’25)</td><td> $7 7 . 7 _ { \pm 0 . 5 }$ </td><td> $7 6 . 5 { \scriptstyle \pm 0 . 6 }$ </td><td> $7 3 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $7 3 . 7 _ { \pm 0 . 3 }$ </td><td> $8 0 . 2 _ { \pm 0 . 3 }$ </td><td> $7 9 . 9 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>GLSim (NeurIPS’25)</td><td> $7 4 . 7 _ { \pm 0 . 4 }$ </td><td> $5 2 . 9 { \scriptstyle \pm 1 . 5 }$ </td><td> $7 1 . 8 { \scriptstyle \pm 0 . 7 }$ </td><td> $6 2 . 7 _ { \pm 0 . 5 }$ </td><td> $7 8 . 7 _ { \pm 0 . 6 }$ </td><td> $6 3 . 4 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>NLL (ICLR&#x27;24)</td><td> $7 6 . 0 { \scriptstyle \pm 0 . 5 }$ </td><td> $7 1 . 5 { \scriptstyle \pm 0 . 4 }$ </td><td> $7 8 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 9 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> $9 0 . 6 _ { \pm 0 . 3 }$ </td><td> $7 6 . 7 _ { \pm 0 . 2 }$ </td></tr><tr><td>Entropy (ICLR’21)</td><td> $7 5 . 3 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 2 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $8 0 . 6 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 1 . 1 { \scriptstyle \pm 0 . 4 }$ </td><td> $9 2 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 5 . 8 { \scriptstyle \pm 0 . 1 }$ </td></tr><tr><td>SVAR (CVPR’25)</td><td> $7 5 . 4 { \scriptstyle \pm 1 . 0 }$ </td><td> $8 0 . 9 { \scriptstyle \pm 0 . 7 }$ </td><td> $6 2 . 5 { \scriptstyle \pm 0 . 7 }$ </td><td> $7 0 . 2 { \scriptstyle \pm 0 . 8 }$ </td><td> $7 2 . 2 { \scriptstyle \pm 0 . 6 }$ </td><td> $8 5 . 9 { \scriptstyle \pm 0 . 5 }$ </td></tr><tr><td>DHCP (MM&#x27;25)</td><td> $7 6 . 4 { \scriptstyle \pm 0 . 7 }$ </td><td> $8 1 . 8 { \scriptstyle \pm 0 . 6 }$ </td><td> $7 0 . 6 { \scriptstyle \pm 0 . 2 }$ </td><td> $6 0 . 7 _ { \pm 1 . 2 }$ </td><td> $9 1 . 0 { \scriptstyle \pm 0 . 2 }$ </td><td> $9 2 . 2 { \scriptstyle \pm 0 . 3 }$ </td></tr><tr><td>CADMP (Ours)</td><td> $\mathbf { 8 2 . 0 { \scriptstyle \pm 0 . 2 } }$ </td><td> $\mathbf { 8 7 . 4 _ { \pm 0 . 2 } }$ </td><td> $\mathbf { 8 4 . 9 \pm 1 . 1 }$ </td><td> $\mathbf { 8 1 . 4 } _ { \pm 0 . 2 }$ </td><td> $\mathbf { 9 4 . 5 \pm 0 . 2 }$ </td><td> $\mathbf { 9 4 . 9 { \scriptstyle \pm 0 . 2 } }$ </td></tr><tr><td rowspan="7">Qwen2.5-VL-7B</td><td>IC (ICLR’25)</td><td> $8 6 . 7 _ { \pm 0 . 5 }$ </td><td> $6 6 . 1 _ { \pm 2 . 9 }$ </td><td> $5 5 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $5 3 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $7 5 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $7 2 . 2 { \scriptstyle \pm 0 . 5 }$ </td></tr><tr><td>GLSim (NeurIPS’25)</td><td> $8 6 . 7 _ { \pm 1 . 3 }$ </td><td> $6 8 . 3 { \scriptstyle \pm 0 . 8 }$ </td><td> $6 0 . 4 { \scriptstyle \pm 0 . 6 }$ </td><td> $6 3 . 8 { \scriptstyle \pm 0 . 5 }$ </td><td> $7 0 . 8 { \scriptstyle \pm 0 . 3 }$ </td><td> $7 6 . 8 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>NLL (ICLR&#x27;24)</td><td> $8 6 . 9 { \scriptstyle \pm 0 . 7 }$ </td><td> $6 3 . 6 { \scriptstyle \pm 2 . 7 }$ </td><td> $8 4 . 2 _ { \pm 0 . 4 }$ </td><td> $8 0 . 6 _ { \pm 0 . 2 }$ </td><td> $8 8 . 6 { \scriptstyle \pm 0 . 8 }$ </td><td> $8 6 . 9 { \scriptstyle \pm 0 . 1 }$ </td></tr><tr><td>Entropy (ICLR’21)</td><td> $8 6 . 7 _ { \pm 0 . 8 }$ </td><td> $6 5 . 4 { \scriptstyle \pm 3 . 4 }$ </td><td> $8 4 . 6 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 0 . 8 { \scriptstyle \pm 0 . 3 }$ </td><td> $8 9 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $9 0 . 9 { \scriptstyle \pm 0 . 5 }$ </td></tr><tr><td>SVAR (CVPR’25)</td><td> $8 6 . 9 { \scriptstyle \pm 0 . 5 }$ </td><td> $7 4 . 2 { \scriptstyle \pm 0 . 4 }$ </td><td> $8 4 . 5 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 0 . 1 { \scriptstyle \pm 0 . 4 }$ </td><td> $8 7 . 1 _ { \pm 0 . 5 }$ </td><td> $6 7 . 0 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>DHCP (MM&#x27;25)</td><td> $8 6 . 6 { \scriptstyle \pm 0 . 6 }$ </td><td> $7 2 . 0 { \scriptstyle \pm 2 . 2 }$ </td><td> $\mathbf { 8 6 . 8 \pm 0 . 3 }$ </td><td> $7 4 . 7 _ { \pm 0 . 6 }$ </td><td> $9 4 . 0 { \scriptstyle \pm 0 . 1 }$ </td><td> $9 1 . 2 _ { \pm 0 . 4 }$ </td></tr><tr><td>CADMP (Ours)</td><td> ${ \bf 8 7 . 6 _ { \pm 0 . 7 } }$ </td><td> ${ \bf 8 2 . 6 _ { \pm 1 . 0 } }$ </td><td> $8 5 . 8 { \scriptstyle \pm 0 . 8 }$ </td><td> $\mathbf { 8 2 . 5 { \pm 0 . 5 } }$ </td><td> ${ \bf 9 6 . 5 { \scriptstyle \pm 0 . 1 } }$ </td><td> ${ \bf 9 2 . 1 { \bf _ { \pm 0 . 6 } } }$ </td></tr><tr><td rowspan="7">InternVL-2.5-4B</td><td>IC (ICLR’25)</td><td> $8 7 . 1 _ { \pm 0 . 3 }$ </td><td> $7 2 . 7 _ { \pm 0 . 9 }$ </td><td> $7 8 . 5 { \scriptstyle \pm 0 . 3 }$ </td><td> $6 1 . 6 { \scriptstyle \pm 0 . 3 }$ </td><td> $7 7 . 7 _ { \pm 0 . 8 }$ </td><td> $7 1 . 3 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td>GLSim (NeurIPS’25)</td><td> $8 7 . 2 _ { \pm 0 . 3 }$ </td><td> $6 1 . 5 { \scriptstyle \pm 2 . 5 }$ </td><td> $7 0 . 7 { \scriptstyle \pm 0 . 8 }$ </td><td> $6 0 . 6 _ { \pm 1 . 1 }$ </td><td> $7 2 . 6 { \scriptstyle \pm 0 . 6 }$ </td><td> $6 1 . 1 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>NLL (ICLR’24)</td><td> $8 7 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td> $6 1 . 8 { \scriptstyle \pm 1 . 4 }$ </td><td> $8 6 . 0 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 2 . 5 { \scriptstyle \pm 0 . 2 }$ </td><td> $9 5 . 1 _ { \pm 0 . 4 }$ </td><td> $8 9 . 0 { \scriptstyle \pm 0 . 1 }$ </td></tr><tr><td>Entropy (ICLR&#x27;21)</td><td> $8 7 . 3 { \scriptstyle \pm 0 . 3 }$ </td><td> $6 2 . 4 { \scriptstyle \pm 1 . 3 }$ </td><td> $8 6 . 9 { \scriptstyle \pm 0 . 7 }$ </td><td> $\mathbf { 8 3 . 2 \pm 0 . 4 }$ </td><td> $9 2 . 8 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 9 . 9 { \scriptstyle \pm 0 . 9 }$ </td></tr><tr><td>SVAR (CVPR’25)</td><td> $8 7 . 4 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 2 . 5 { \scriptstyle \pm 1 . 8 }$ </td><td> $7 0 . 1 { \scriptstyle \pm 1 . 0 }$ </td><td> $6 2 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> $8 0 . 6 _ { \pm 0 . 1 }$ </td><td> $7 2 . 2 { \scriptstyle \pm 1 . 4 }$ </td></tr><tr><td>DHCP (MM&#x27;25)</td><td> $8 7 . 2 _ { \pm 0 . 3 }$ </td><td> $7 8 . 8 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 2 . 5 { \scriptstyle \pm 0 . 5 }$ </td><td> $7 2 . 2 { \scriptstyle \pm 0 . 5 }$ </td><td> $6 7 . 6 { \scriptstyle \pm 0 . 2 }$ </td><td> $7 8 . 8 { \scriptstyle \pm 1 . 2 }$ </td></tr><tr><td>CADMP (Ours)</td><td> $\mathbf { 8 9 . 0 _ { \pm 0 . 4 } }$ </td><td> ${ \bf 8 7 . 1 _ { \pm 1 . 1 } }$ </td><td> $\mathbf { 8 8 . 2 \pm 1 . 1 }$ </td><td> $8 1 . 1 { \scriptstyle \pm 1 . 5 }$ </td><td> $\mathbf { 9 7 . 5 { \scriptstyle \pm 0 . 1 } }$ </td><td> $\mathbf { 9 2 . 8 { \scriptstyle \pm 0 . 4 } }$ </td></tr></table>

Table 1: Comparison of object hallucination detection performance in terms of accuracy (ACC) and area under the ROC curve (AUROC) across three models and three datasets. Higher values indicate better performance.

InternVL-2.5-4B, respectively. On Pascal VOC, CADMP ranks first in both ACC and AUROC across all three models, reaching up to 97.5% ACC and 94.9% AUROC. It also remains competitive on POPE, although its ACC on Qwen2.5- VL-7B and AUROC on InternVL-2.5-4B are slightly below the corresponding best baselines. Overall, these results indicate that combining adjacent-layer attention drift with maskbased probability verification provides robust and complementary signals for object hallucination detection.

<table><tr><td rowspan="2">Method</td><td colspan="2">Object</td><td colspan="2">Attribute</td><td colspan="2">Relation</td></tr><tr><td>ACC</td><td>AUROC</td><td>ACC</td><td>AUROC</td><td>ACC</td><td>AUROC</td></tr><tr><td>Entropy</td><td>92.89</td><td>86.92</td><td>82.16</td><td>78.99</td><td>58.59</td><td>55.38</td></tr><tr><td>DHCP</td><td>93.51</td><td>81.67</td><td>74.19</td><td>70.96</td><td>53.91</td><td>54.89</td></tr><tr><td>GLSim</td><td>87.54</td><td>86.76</td><td>71.35</td><td>67.52</td><td>56.25</td><td>60.19</td></tr><tr><td>CADMP</td><td>93.69</td><td>87.11</td><td>82.50</td><td>80.75</td><td>72.40</td><td>73.05</td></tr></table>

Table 2: Hallucination detection performance on the AM-BER benchmark using Qwen2.5-VL-7B.

Results on AMBER To further evaluate whether CADMP generalizes beyond OH detection, we conducted experiments on the AMBER discriminative dataset, which covers three types of hallucinations: object, attribute, and relation hallucinations. As shown in Table 2, CADMP consistently outperforms representative baselines across all three hallucination types on Qwen2.5-VL-7B. These results suggest that the proposed attention-drift modeling and adaptive mask-based grounding verification are not limited to object hallucination, but can also provide useful detection signals for more fine-grained hallucination types.

over, compared to the forward based detection method in (Liu et al. 2024a), our method still achieves an accuracy of 88.2% with faster inference speed, demonstrating the efectiveness of CADMP.

Eficiency Analysis To evaluate the practical overhead of CADMP, we performed an eficiency analysis covering latency and ACC (Figure 4). Compared to GLSim (Park and Li 2025), our method without additional masked-image forward pass can achieve higher accuracy with lower latency. More-

Cross-dataset Generalization Analysis. To further evaluate whether CADMP captures intrinsic hallucination cues rather than dataset-specific patterns, we train the detector on a source dataset and then evaluate it on diferent target datasets without any additional adaptation (Source → Target). As shown in Table 3, our method consistently outperforms DHCP across all target datasets and all hallucination types. For example, CADMP obtains AUROC scores of 73.2% on AMBER-object, 76.7% on AMBER-Attribute and 66.4% on AMBER-relation, demonstrating its ability to generalize across diverse types of hallucinations. It also maintains favorable performance on Pascal VOC. These results suggest that CADMP captures more robust and transferable hallucination signals, rather than relying primarily on dataset-specific characteristics.

(a) Efficiency Analysis  
![](images/6704d94566e8d29300e92f23210703091809600644b965f7c82f5160299f9181.jpg)

![](images/d428f02e21309b075871e951bc82f939eb2713bfed6501dd3852659730b0905e.jpg)

(b) Masking Layer Selection  
![](images/9624fd23d958c7e5c6f92b8f4b443b264de7e62e6c1efad2e59afe16e5d0005f.jpg)

(c) Masking Region Selection  
![](images/625c3ca548d58f6584dfc461577a2e730255fd2564eacbe5eadb7a4d404cfb4c.jpg)

Figure 4: (a) Eficiency analysis on InternVL-2.5-4B using the POPE dataset; (b) Diferent masking-layer selection strategies; (c) Diferent masking-region selection strategies.
<table><tr><td>Model</td><td>Setting</td><td>DHCP</td><td>CADMP</td></tr><tr><td rowspan="2">LLaVA-1.5-7B</td><td>POPE → VOC</td><td>77.8</td><td>86.7</td></tr><tr><td>VOC → POPE</td><td>65.9</td><td>74.8</td></tr><tr><td rowspan="2">InternVL-2.5-4B</td><td>POPE → VOC</td><td>78.9</td><td>82.6</td></tr><tr><td>VOC → POPE</td><td>71.8</td><td>75.5</td></tr><tr><td rowspan="5">Qwen2.5-VL-7B</td><td>VOC → POPE</td><td>71.2</td><td>74.4</td></tr><tr><td>POPE → VOC</td><td>73.1</td><td>87.6</td></tr><tr><td>POPE → Am-Obj</td><td>70.2</td><td>73.2</td></tr><tr><td>POPE → Am-Attr</td><td>72.5</td><td>76.7</td></tr><tr><td>POPE → Am-Rel</td><td>50.4</td><td>66.4</td></tr></table>

Table 3: Cross-dataset AUROC results comparison between DHCP and our method across three LVLMs.

## Ablation Studies

We further conducted a series of ablation studies for the proposed CADMP detection method to evaluate the contribution of each component.

<table><tr><td>Metric</td><td>LLaVA-1.5</td><td>Qwen2.5-VL</td><td>InternVL-4B</td><td>Avg.</td></tr><tr><td>Cosine</td><td>86.6</td><td>80.8</td><td>86.5</td><td>84.6</td></tr><tr><td>Euclidean</td><td>86.4</td><td>80.6</td><td>86.9</td><td>84.6</td></tr><tr><td>Pearson</td><td>86.7</td><td>80.5</td><td>86.7</td><td>84.6</td></tr><tr><td>KL</td><td>87.5</td><td>81.5</td><td>87.9</td><td>85.6</td></tr></table>

Table 4: Comparison ofAUROC across diferent cross-modal attention drift methods on the MSCOCO dataset.

Selection of cross-layer evaluation metrics As shown in Table 4, we compare several methods for measuring cross-modal attention drift, including Cosine Similarity, Euclidean Distance, Pearson Correlation, and Kullback–Leibler (KL) Divergence, across diferent models on the MSCOCO dataset. Among them, KL divergence achieves the best overall performance, with an average AUROC of 85.6%, outperforming the other methods. This result is consistent with our formulation of cross-modal attention drift as redistribution over visual tokens and suggests that KL divergence is better suited to capture abnormal grounding transitions.

Layer Selection for Cross-modal Attention Masking To validate whether the layer immediately preceding the largest attention drift can better capture useful grounded information, we conduct an ablation study using four methods: the previous layer of the largest drift appearance, the subsequent layer, a fixed middle layer and the final layer. As shown in Figure 4 (b), experiments on LLaVA-1.5 and Qwen2.5-VL show that selecting the previous layer achieves the best performance. These results suggest that the attention distribution immediately before the most abrupt cross-layer redistribution may provide a somewhat more informative signal for grounding verification.

Masking Region Selection We compare four region selection methods under the same 30% masking ratio and identical replacement method. In addition to the top-attention masking, we also conduct experiments on random masking, the least-attention masking and center masking. As shown in Figure 4 (c), top-attention masking achieves the highest AU-ROC and ACC among the compared region-selection methods. These results indicate that detector’s performance gains arise from removing the visual evidence attended to by the model, rather than from arbitrary image perturbations.

Efectiveness of diferent components in CADMP. As shown in Figure 2, the proposed CADMP is built based on Cross-Modal Attention Drifts (CMAD) and Mask-based Verification (MV). Table 5 reports the hallucination detection performance of each component in our method on COCO. The results show that both attention drift and probability features provide informative signals for hallucination detection, while their combination yields the best overall performance. Specifically, on LLaVA-1.5-7B, the attention drift feature can achieve an AUROC of 85.7%, while the probability signal obtains an AUROC of 72.8%. By combining these complementary signals, CADMP can improve the AUROC to 87.4% and achieve the best detection performance. The same trend is also observed on Qwen2.5-VL-7B and InternVL-2.5-4B.

![](images/4fec5d4919d324379968f55a081926a2015fbf2f3e7dece1ba408e2cb412c4b4.jpg)

Figure 5: CADMP visualization of grounded (top) and hallucination (bottom) cases.
<table><tr><td>Model</td><td>CMAD</td><td>MV</td><td>AUC (%)</td><td>ACC (%)</td></tr><tr><td rowspan="3">LLaVA-1.5-7B</td><td>√</td><td></td><td>85.7</td><td>80.0</td></tr><tr><td></td><td>√</td><td>72.8</td><td>76.2</td></tr><tr><td>√</td><td>√</td><td>87.4</td><td>82.1</td></tr><tr><td rowspan="3">Qwen2.5-VL-7B</td><td>√</td><td></td><td>80.3</td><td>87.3</td></tr><tr><td></td><td>√</td><td>66.1</td><td>86.8</td></tr><tr><td>√</td><td>√</td><td>82.6</td><td>87.6</td></tr><tr><td rowspan="3">InternVL-2.5-4B</td><td>√</td><td></td><td>86.3</td><td>88.0</td></tr><tr><td></td><td>√</td><td>57.8</td><td>88.0</td></tr><tr><td>√</td><td>√</td><td>87.1</td><td>89.1</td></tr></table>

Table 5: Ablation study of CMAD and MV modules for CADMP with diferent LVLMs.

Ablation Study of KL Drift Formulations. To investigate which cross-layer KL formulation is most suitable for hallucination detection, we conducted an ablation study on the Pascal VOC dataset using four drift variants over the crossmodal attention sequence $S A = \{ A _ { 1 } , A _ { 2 } , \ldots , A _ { L } \}$ , where $A _ { i }$ denotes the attention distribution at layer i. Specifically, Adjacent-layer computes $K L ( A _ { i } \| A _ { i - 1 } )$ to capture local layer-to-layer changes, First-anchor computes $\bar { K L } ( A _ { i } \| A _ { 1 } )$ to measure deviations from the initial grounding state, Last-anchor computes $K L ( A _ { i } \| A _ { L } )$ to reflect discrepancies from the final attention state, and First-last computes $K L ( A _ { 1 } \| A _ { L } )$ as a global summary of cross-layer change. As shown in Table 6, experiments on both LLaVA-1.5-7B and Qwen2.5-VL-7B show that Adjacent-layer consistently achieves the best performance among the four variants. This suggests that hallucination is better characterized by adjacent cross-layer attention drift than by deviations from a fixed anchor or a single global discrepancy, highlighting the importance of modeling fine-grained grounding transitions.

Visualization Analysis. We present qualitative visualizations in Figure 5. As shown, hallucinated cases exhibit substantially stronger cross-modal attention drifts in the critical intermediate layers and smaller probability drops after masking relevant visual regions, indicating weaker visual grounding. In contrast, grounded cases show relatively milder attention changes and greater sensitivity to visual perturbation. These results further support the efectiveness of our design in distinguishing hallucinated and grounded cases.

<table><tr><td>Model</td><td>Method</td><td>AUROC (%)</td><td>ACC (%)</td></tr><tr><td rowspan="4">LLaVA-1.5-7B</td><td>Adjacent-layer</td><td>94.9</td><td>94.5</td></tr><tr><td>Last-anchor</td><td>93.9</td><td>93.5</td></tr><tr><td>First-anchor</td><td>93.2</td><td>93.3</td></tr><tr><td>First-last</td><td>88.6</td><td>92.9</td></tr><tr><td rowspan="4">Qwen2.5-VL-7B</td><td>Adjacent-layer</td><td>92.1</td><td>96.5</td></tr><tr><td>Last-anchor</td><td>90.4</td><td>94.6</td></tr><tr><td>First-anchor</td><td>86.3</td><td>93.9</td></tr><tr><td>First-last</td><td>84.1</td><td>90.1</td></tr></table>

Table 6: Ablation study of diferent KL drift formulations.

## Conclusion

In this paper, we proposed CADMP, a hallucination detection framework for large vision-language models. CADMP detects object hallucinations by measuring distribution shifts between layers and further verifying the visual grounding of model predictions through mask-based probability variation analysis. By jointly capturing internal attention drifts and perturbation-based visual dependence, our method provides a more fine-grained and complementary empirical signal for hallucination detection. Extensive experiments on three representative LVLMs, including LLaVA-1.5-7B, Qwen2.5-VL-7B, and InternVL-2.5-4B, demonstrate the efectiveness and generalizability of the proposed method across diferent hallucination detection settings.

## References

Arteaga, G. Y.; Schön, T. B.; and Pielawski, N. 2024. Hallucination detection in llms: Fast and memory-eficient finetuned models. arXiv preprint arXiv:2409.02976.

Augustin, M.; Neuhaus, Y.; and Hein, M. 2025. Dash: Detection and assessment of systematic hallucinations of vlms. In Int. Conf. Comput. Vis., 22748–22759.

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025b. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Bai, Z.; Wang, P.; Xiao, T.; He, T.; Han, Z.; Zhang, Z.; and Shou, M. Z. 2024. Hallucination of Multimodal Large Language Models: A Survey. arXiv preprint arXiv:2404.18930.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE Conf. Comput. Vis. Pattern Recog., 24185–24198.

Delétang, G.; Ruoss, A.; Duquenne, P.-A.; Catt, E.; Genewein, T.; Mattern, C.; Grau-Moya, J.; Wenliang, L. K.; Aitchison, M.; Orseau, L.; et al. 2023. Language modeling is compression. arXiv preprint arXiv:2309.10668.

Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.; and Zisserman, A. 2010. The pascal visual object classes (voc) challenge. Int. J. Comput. Vis., 88(2): 303–338.

Favero, A.; Zancato, L.; Trager, M.; Choudhary, S.; Perera, P.; Achille, A.; Swaminathan, A.; and Soatto, S. 2024. Multimodal hallucination control by visual information grounding. In IEEE Conf. Comput. Vis. Pattern Recog., 14303–14312.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; et al. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.

Hoang-Xuan, N.; Vu, M.; Thai, M. T.; and Bhattarai, M. 2026. PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models. In IEEE Conf. Comput. Vis. Pattern Recog.

Hu, H.; Zhang, J.; Zhao, M.; and Sun, Z. 2023. Ciem: Contrastive instruction evaluation method for better instruction tuning. arXiv preprint arXiv:2309.02301.

Jiang, N.; Kachinthaya, A.; Petryk, S.; and Gandelsman, Y. 2025a. Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations. In Int. Conf. Learn. Represent.

Jiang, Z.; Chen, J.; Zhu, B.; Luo, T.; Shen, Y.; and Yang, X. 2025b. Devils in middle layers of large vision-language

models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 25004–25014.

Jing, L.; Li, R.; Chen, Y.; and Du, X. 2024. Faithscore: Fine-grained evaluations of hallucinations in large visionlanguage models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, 5042–5063.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; et al. 2024. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246.

Li, C.; Wong, C.; Zhang, S.; Usuyama, N.; Liu, H.; Yang, J.; Naumann, T.; Poon, H.; and Gao, J. 2023a. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Adv. Neural Inform. Process. Syst., 36: 28541– 28564.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023b. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Int. Conf. Machine Learn., 19730–19742. PMLR.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.-R. 2023c. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 conference on empirical methods in natural languageprocessing, 292–305.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft coco: Common objects in context. In Eur. Conf. Comput. Vis., 740–755. Springer.

Liu, F.; Lin, K.; Li, L.; Wang, J.; Yacoob, Y.; and Wang, L. 2024a. Mitigating hallucination in large multi-modal models via robust instruction tuning. Int. Conf. Learn. Represent.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024b. Improved baselines with visual instruction tuning. In IEEE Conf. Comput. Vis. Pattern Recog., 26296–26306.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Adv. Neural Inform. Process. Syst., 36: 34892–34916.

Liu, H.; Xue, W.; Chen, Y.; Chen, D.; Zhao, X.; Wang, K.; Hou, L.; Li, R.; and Peng, W. 2024c. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Lu, H.; Liu, W.; Zhang, B.; Wang, B.; Dong, K.; Liu, B.; Sun, J.; Ren, T.; Li, Z.; Yang, H.; et al. 2024. Deepseekvl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525.

Malinin, A.; and Gales, M. 2021. Uncertainty Estimation in Autoregressive Structured Prediction. In Int. Conf. Learn. Represent.

Park, S.; and Li, S. 2025. GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Rohrbach, A.; Hendricks, L. A.; Burns, K.; Darrell, T.; and Saenko, K. 2018. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 4035–4045.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971.

Wang, C.; Chen, X.; Zhang, N.; Tian, B.; Xu, H.; Deng, S.; and Chen, H. 2024a. Mllm can see? dynamic correction decoding for hallucination mitigation. arXiv preprint arXiv:2410.11779.

Wang, J.; Wang, Y.; Xu, G.; Zhang, J.; Gu, Y.; Jia, H.; Wang, J.; Xu, H.; Yan, M.; Zhang, J.; et al. 2023. Amber: An llmfree multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397.

Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; et al. 2024b. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Wei, Y.; Wang, Z.; Lu, Y.; Xu, C.; Liu, C.; Zhao, H.; Chen, S.; and Wang, Y. 2024. Editable scene simulation for autonomous driving via collaborative llm-agents. In IEEE Conf. Comput. Vis. Pattern Recog., 15077–15087.

Xu, X.; Chen, H.; Lyu, M.; Zhao, S.; Xiong, Y.; Lin, Z.; Han, J.; and Ding, G. 2025. Mitigating Hallucinations in Multimodal Large Language Models via Image Token Attention-Guided Decoding.

You, H.; Zhang, H.; Gan, Z.; Du, X.; Zhang, B.; Wang, Z.; Cao, L.; Chang, S.-F.; and Yang, Y. 2023. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704.

Zhang, Y.; Xie, R.; Sun, X.; Huang, Y.; Chen, J.; Kang, Z.; Wang, D.; and Wang, Y. 2025. Dhcp: Detecting hallucinations by cross-modal attention pattern in large visionlanguage models. In Proceedings of the 33rd ACM International Conference on Multimedia, 3555–3564.

Zhou, Y.; Cui, C.; Yoon, J.; Zhang, L.; Deng, Z.; Finn, C.; Bansal, M.; and Yao, H. 2024. Analyzing and Mitigating Object Hallucination in Large Vision-Language Models. In Int. Conf. Learn. Represent.

Zhu, D.; Chen, J.; Shen, X.; Li, X.; and Elhoseiny, M. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## Further Ablation Studies

## Efect of Masking Ratios

We investigate the efect of the masking ratio τ on the MSCOCO dataset using LLaVA-1.5-7B by varying τ from 10% to 50%. As shown in Table 7, CADMP achieves the best performance at a masking ratio of 30%, obtaining an AUROC of 87.97% and an ACC of 82.88%. These results suggest that a small masking ratio may fail to cover the key object regions, whereas a large ratio may introduce substantial visual perturbations. A masking ratio of 30% achieves a favorable balance between removing critical visual evidence and preserving the overall image context.

<table><tr><td>Masking Ratio τ</td><td>AUROC (%)</td><td>ACC (%)</td></tr><tr><td>10%</td><td>87.17</td><td>82.32</td></tr><tr><td>20%</td><td>87.27</td><td>81.84</td></tr><tr><td>30%</td><td>87.97</td><td>82.88</td></tr><tr><td>40%</td><td>87.26</td><td>82.44</td></tr><tr><td>50%</td><td>87.31</td><td>81.57</td></tr></table>

Table 7: Efect of diferent masking ratios on the MSCOCO dataset using LLaVA-1.5-7B.

## Efect of Mask Filling Values

To study the choice of fill color for masked image regions, we conduct an experiment using LLaVA-1.5-7B on the MSCOCO dataset. As shown in Table 8, filling the masked regions with the default gray value of 128 achieves the best performance, with an AUROC of 87.97% and an ACC of 82.88%. Compared with black, white and the ImageNet mean RGB value (123, 116, 103), gray filling achieves the best performance. Therefore, we adopt gray filling with a value of 128, as it efectively masks the selected visual evidence without introducing excessively dark or bright perturbations.

<table><tr><td>Filling Value</td><td>AUROC (%)</td><td>ACC (%)</td></tr><tr><td>Gray</td><td>87.97</td><td>82.88</td></tr><tr><td>Black</td><td>87.23</td><td>82.02</td></tr><tr><td>White</td><td>87.38</td><td>82.02</td></tr><tr><td>Mean</td><td>87.31</td><td>81.72</td></tr></table>

Table 8: Efect of diferent mask filling values on the MSCOCO dataset using LLaVA-1.5-7B.

## Multi-token Objects

To compute the KL divergence sequence and the probability variation for a target object token, we select the first token of each multi-token object. To evaluate this design choice, we conduct an ablation study comparing three strategies: using the first token, using the last token and taking the average across all tokens. We conduct experiments on MSCOCO dataset across three LVLMs in Table 9. The results show that the first-token strategy is most efective, likely because the first token often captures the core semantic meaning of the object.

<table><tr><td colspan="4">Strategy LLaVA-1.5-7B Qwen2.5-VL-7B InternVL2.5-4B</td></tr><tr><td>First</td><td>87.97</td><td>82.60</td><td>88.47</td></tr><tr><td>Last</td><td>86.34</td><td>81.46</td><td>88.17</td></tr><tr><td>Average</td><td>87.65</td><td>81.97</td><td>88.16</td></tr></table>

Table 9: AUROC (%) of diferent token selection strategies on the MSCOCO dataset.

<table><tr><td>Variant</td><td>AUROC (%)</td><td>ACC (%)</td></tr><tr><td>Full KL sequence</td><td>87.97</td><td>82.88</td></tr><tr><td>Statistical summary</td><td>80.58</td><td>79.01</td></tr><tr><td>Peak centered (5)</td><td>79.05</td><td>78.23</td></tr><tr><td>Mean KL</td><td>78.20</td><td>78.62</td></tr><tr><td>Peak centered (3)</td><td>76.50</td><td>77.94</td></tr><tr><td>Maximum KL</td><td>72.68</td><td>76.27</td></tr></table>

Table 10: Ablation of KL-drift feature representations on MSCOCO using LLaVA-1.5-7B.

## Ablation of Drift Feature Representations

To examine whether the complete layer-wise drift sequence is necessary for hallucination detection, we compare six KL feature representations. Let $K _ { i } = D _ { \mathrm { K L } } ( A _ { i } \Vert A _ { i - 1 } ^ { \cdot } )$ denote the attention drift between two adjacent decoder layers, and let $k ^ { * } = \arg \operatorname* { m a x } _ { i } K _ { i }$ denote the position of the largest drift. The Full KL sequence uses the complete sequence $[ K _ { 2 } , \dots , K _ { L } ]$ . Maximum KL and Mean KL compress the sequence into its maximum and average values, respectively, while Statistical summary represents it using the mean, maximum, and standard deviation. For the Peak-centered local window variants, we select three or five consecutive KL values centered around k<sup>∗</sup>, capturing the local drift pattern around the strongest transition.

As shown in Table 10, the full KL sequence achieves the best performance, reaching an AUROC of 87.97% and an ACC of 82.88%. Compressing the sequence into statistical summaries decreases the AUROC by 7.39 percentage points, while using only the maximum KL value leads to a substantial reduction of 15.29 points. These results indicate that hallucination information is distributed across the layer-wise drift sequence rather than being confined to the largest drift or its immediate neighboring layers.

## Distribution of Peak Attention-Drift Layers

Figure 6 shows the distribution of layers where the maximum adjacent-layer attention drift occurs on Qwen2.5-VL-7B. For hallucinated samples, the maximum attention drift most frequently occurs at Layers 15, 10, and 13, suggesting that abrupt cross-modal attention redistribution tends to emerge in the intermediate layers. This observation further supports the efectiveness of CADMP in capturing hallucinationrelated signals from layer-wise attention dynamics.

![](images/c732c47cdb21cd64d4a79d8daacb5b625f974d2816bd09ccdf259dccb8f974b4.jpg)  
Figure 6: Distribution of the layers exhibiting the maximum adjacent-layer cross-modal attention drift for grounded and hallucinated samples on Qwen2.5-VL-7B.

## Experiment Datasets

## COCO-Caption

To evaluate hallucination detection in open-ended generation, we construct the COCO-Caption setting from the COCO-val2014 split of MSCOCO (Lin et al. 2014). We select 5,000 images from COCO-val2014 and independently generate captions using each evaluated LVLM. Caption generation is performed with greedy decoding, and the maximum number of newly generated tokens is set to 128. We apply the CHAIR annotation protocol (Rohrbach et al. 2018) to extract object mentions from each generated caption and determine whether each mentioned object is supported by the corresponding image. For multi-token objects, CADMP uses only the first subtoken for probability, attention, and masking.

Because one generated caption may contain multiple object mentions, directly splitting object-level instances could place objects extracted from the same image in diferent subsets and introduce information leakage. We therefore perform an image-level split using a ratio of 70%/10%/20% for training, validation, and testing, respectively.

## POPE

POPE (Li et al. 2023c) is designed to evaluate objectexistence hallucinations in Large Vision-Language Models. It formulates the evaluation as a binary question-answering task by asking whether a particular object is present in an input image. The benchmark is constructed from images in MSCOCO (Lin et al. 2014) and contains three subsets, namely Random, Popular, and Adversarial. The Random subset uniformly samples objects that do not appear in the image, the Popular subset selects frequently occurring object categories, and the Adversarial subset selects absent objects that commonly co-occur with the objects present in the image.

For implementation, we use 1,500 images. For each image, three positive questions and three negative questions are constructed, where positive questions query objects present in the image and negative questions query absent objects. This results in a total of 9,000 image-question pairs.

## Pascal VOC

Pascal VOC is a widely used benchmark for object detection and recognition in computer vision (Everingham et al. 2010). It provides image-level and object-level annotations for 20 common object categories, covering animals, vehicles, household objects, and people. Owing to its diverse visual scenes and reliable category annotations, Pascal VOC ofers a suitable testbed for evaluating whether an LVLM can correctly determine the presence or absence of specific objects.

In our experiments, we use images from the VOC2012 split to construct an additional evaluation set for object hallucination detection. Specifically, for each image, we create one positive sample by querying an object category that is present according to the ground-truth annotations, together with three negative samples based on categories absent from the image. This results in a total of 9,000 samples in the Pascal VOC-based evaluation set.

## AMBER

AMBER (Wang et al. 2023) is a benchmark designed to evaluate hallucinations in multimodal large language models without relying on an external language model as the evaluator. It contains 1,004 images and provides both generative and discriminative evaluation settings. While the generative setting assesses hallucinations in free-form model responses, the discriminative setting uses targeted questions to examine whether a model can accurately distinguish visually supported information from unsupported claims.

The discriminative component covers three representative hallucination types: object existence, attribute, and relation hallucinations. Object-existence hallucination occurs when a model claims that an object is present even though it does not appear in the image. Attribute hallucination refers to assigning an incorrect property, such as color, number, size, or state, to an existing object. Relation hallucination occurs when the model incorrectly describes the spatial or semantic relationship between objects. By jointly evaluating these three dimensions, AMBER provides a broader and more finegrained assessment than benchmarks focusing only on object presence.

## Baselines

We compare CADMP with six representative object hallucination detection baselines.

## Negative Log-Likelihood (NLL)

NLL (Zhou et al. 2024) uses the generation probability of the target object token as the hallucination score, where j denotes the positional index of object o:

$$
s _ { \mathrm { N L L } } ( o ) = - \log P ( o \mid \mathbf { v } , \mathbf { t } , Y _ { < j } ) .
$$

For a multi-token object, we use its first token. A larger NLL score indicates a higher likelihood of hallucination.

## Entropy

Entropy (Malinin and Gales 2021) measures predictive uncertainty from the vocabulary distribution at the target object-

token position:

$$
s _ { \mathrm { E n t } } ( o ) = - \sum _ { w \in \mathcal { V } } P ( w ) \log P ( w ) ,
$$

where V denotes the vocabulary. Higher entropy indicates greater uncertainty in the object prediction.

## IC

IC (Jiang et al. 2025a) applies the Visual Logit Lens to project the hidden representation of each visual token at every decoder layer into the vocabulary space:

$$
\mathrm { V L L } _ { l } ( v _ { i } ) = \mathbf { h } _ { l } ( v _ { i } ) \mathbf { W } _ { U } ,
$$

where $\mathbf { h } _ { l } ( v _ { i } )$ denotes the hidden representation of visual token $v _ { i }$ at layer l, and $\mathbf { W } _ { U }$ is the language-model unembedding matrix. The IC score is defined as the maximum probability assigned to the target object token across all decoder layers and visual positions:

$$
s _ { \mathrm { I C } } ( o ) = \operatorname* { m a x } _ { l , i } \mathrm { S o f t m a x } \left( \mathrm { V L L } _ { l } ( v _ { i } ) \right) _ { o } .
$$

A higher IC score indicates stronger internal visual support for the presence of the target object.

## SVAR

SVAR (Jiang et al. 2025b) measures the visual attention received by a generated object token. For an object token $^ { O , }$ the Visual Attention Ratio at layer ℓ and attention head h is defined as

$$
\mathrm { V A R } _ { \ell , h } ( o ) = \sum _ { i = 1 } ^ { N } A _ { \ell , h } ( o , v _ { i } ) .
$$

where $A ^ { ( \ell , h ) } ( o , v _ { i } )$ denotes the attention weight from the object token o to the i-th visual token $v _ { i } ,$ , and N is the number of visual tokens. SVAR then averages the VAR scores across all attention heads and sums them over Layers 5–18:

$$
s _ { \mathrm { S V A R } } ( o ) = \frac { 1 } { H } \sum _ { \ell = 5 } ^ { 1 8 } \sum _ { h = 1 } ^ { H } { \mathrm { V A R } ^ { ( \ell , h ) } ( o ) } ,
$$

where H denotes the number of attention heads. A higher SVAR score indicates that the generated object token assigns more attention to visual information.

## GLSim

GLSim (Park and Li 2025) is an object hallucination detection method that jointly evaluates global scene-level consistency and local visual grounding. It measures whether a predicted object is semantically compatible with the overall image context and whether relevant visual regions provide suficient evidence for its presence. For evaluation on POPE, we extract the queried object from each question and use its token representation to compute the GLSim score.

## DHCP

DHCP (Zhang et al. 2025) detects object hallucinations by modeling cross-modal attention patterns within LVLMs. It uses the attention assigned by generated tokens to visual tokens across decoder layers as internal evidence to distinguish visually grounded predictions from hallucinated ones.

## Baselines Details

For experiments on LLaVA-1.5-7B using coco-caption dataset, we implement all baselines using the following configurations. For SVAR, we aggregate the attention assigned by the generated object token to all visual tokens over Layers 5-18 and normalize the resulting score by the number of attention heads. For GLSim, we set the image layer and text layer to 32 and 31, respectively, select the top 32 visual tokens for local similarity computation, and combine the global and local similarity scores using weights of 0.6 and 0.4; the visual sequence contains 576 tokens arranged in a 24×24 grid. DHCP extracts head-averaged attention from the generated object token to the visual tokens at each layer, pads each layer to 576 visual positions, and flattens the resulting features. Its MLP classifier consists of a 128-dimensional hidden layer followed by a two-class output layer and is trained using Adam with a learning rate of $1 \times \mathrm { { 1 0 ^ { - 3 } } }$ , a batch size of 256, and 30 epochs. We use cross-entropy loss and a Weighted Random Sampler with inverse class-frequency weights.

## Experiment Settings

## Models

We evaluate CADMP on three representative opensource LVLMs, including LLaVA-1.5-7B, Qwen2.5-VL-7B-Instruct, and InternVL2.5-4B, which difer in model architecture, visual-language alignment strategy, and parameter scale. All experiments are conducted using Python 3.10 and PyTorch 2.6.0 on a single NVIDIA A6000 GPU with 48GB of memory.

## Mask Construction

After identifying the layer transition with the maximum attention drift, we select its preceding layer as the suspicious layer. We extract the cross-modal attention from the target output token to all visual tokens and reshape it into an H×W attention map according to the spatial layout of the visual feature map. The resulting attention map is then upsampled to the original image resolution using bilinear interpolation, such that each pixel is assigned a continuous attention weight. Next, a threshold is determined according to a predefined percentile of the upsampled attention values. Pixels whose attention weights are greater than or equal to this threshold are replaced with a neutral gray value of 128, while the remaining pixels are left unchanged. The resulting masked image is subsequently fed into the LVLM for a second forward pass to measure the change in the prediction probability of the target token.

## Implementation Details

For the COCO-Caption experiments, LLaVA-1.5-7B, Qwen2.5-VL-7B, and InternVL2.5-4B use the same MLP detector configuration. The input consists of the layer-wise KL-divergence drift sequence and three probability-based features: the target-token probability under the original image, the probability under the masked image, and their difference. All features are standardized using StandardScaler before being passed to a two-hidden-layer MLP. The two hidden layers contain 128 and 64 neurons, respectively, and each is followed by Batch Normalization, ReLU activation, and Dropout with a rate of 0.3. The detector is trained using AdamW with a learning rate of $5 \times 1 0 ^ { - 4 }$ , a batch size of 128, and a weight decay of $\mathrm { \bar { 1 } \times 1 0 ^ { - 4 } }$ . Training is performed for at most 150 epochs, with early stopping using a patience of 20 epochs to select the best checkpoint on the validation set. The data are divided into training, validation, and test sets using a ratio of 70%/10%/20%.

For AUROC computation, hallucinated samples are treated as the positive class. For threshold-based evaluation, predictions are obtained from the probability of the grounded class, with the decision threshold selected on the validation set by maximizing the F1 score. All experiments are conducted using three random seeds: 42, 7777, and 37, and the final results are reported as the mean and standard deviation across the three runs.

## Visualization of CADMP

To qualitatively examine the efectiveness of CADMP, we present visualization examples from both the POPE and COCO-Caption datasets in Figure 7.

![](images/80d3f893bad908801b749c7519d936b13e29d3c3e81c29046868c11978f10f96.jpg)  
Figure 7: Visualization of CADMP for detecting hallucinations on the POPE and COCO-Caption datasets