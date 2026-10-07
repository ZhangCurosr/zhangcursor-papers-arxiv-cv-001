# Dynamic Alignment and Calibration for Multimodal Learning

Jinghao Xu<sup>1</sup> nolle98xu@gmail.com

Zhenhua Guo<sup>2</sup> 6cszguo@gmail.com

<sup>2</sup>Xiaofeng Zhu<sup>3</sup> seanzhuxf@gmail.com

Xiaoshuang Shi<sup>1</sup> <sup>c</sup>xsshi2013@gmail.com <sup>1</sup> School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu,China

<sup>2</sup> Tianyijiaotong Technology Ltd.,China

<sup>3</sup> School of Computer Science and Technology, Hainan University, Haikou, China

## Abstract

Dynamic multimodal learning aims to learn robust representations by adaptively modeling information discrepancies across modalities. However, existing methods still suffer from two limitations: (i) static cross-modal alignment strategies usually impose uniform constraints on all samples while overlooking sample-wise variations, potentially leading to unreasonable over-alignment; and (ii) confidence- or uncertainty-aware fusion methods often fail to adequately account for feature magnitude and confidence differences across modalities. For modality pairs with significant feature magnitude differences or small confidence gaps, it might be unreliable to strictly align fusion weights according to confidence. To address these issues, we propose an Alignment- and Calibrationdriven Multimodal Learning framework (ACML). Specifically, ACML incorporates a dynamic cross-modal triplet alignment module, which enforces strong semantic consistency for high-confidence positive pairs while encouraging diverse representation learning between high- and low-confidence positive pairs according to their confidence gaps. Additionally, ACML introduces a difference-aware attention calibration strategy that adaptively adjusts attention regularization based on feature magnitude and confidence differences across modalities, thereby mitigating biases caused by unreasonable fusion constraints. Extensive experiments on multiple multimodal benchmark datasets demonstrate that ACML consistently achieves superior performance and robustness over recent state-of-the-art methods.

## 1 Introduction

Multimodal learning has achieved significant success in various fields, such as medical diagnosis[10, 27, 36, 47], emotion recognition[9, 14, 19], and scene understanding[13, 30, 33, 46], by integrating complementary information from diverse modalities to enhance representation power and generalization performance. As the pivotal mechanisms for achieving these advantages, alignment and fusion constitute two core sub-problems in multimodal learning. However, in real-world open environments, data quality and reliability across different modalities often fluctuate dynamically due to device heterogeneity, environmental noise, and occlusions[42]. This necessitates multimodal modeling that not only captures cross-modal consistency but also adapts to sample-level variations. Unfortunately, existing methods fall short of adequately modeling such dynamics in both the alignment and fusion stages.

![](images/1e951c55d0e18495829be06b7156dc346c08db11b60bc8888a67718387ca799b.jpg)  
Figure 1: Overview of the proposed framework ACML. In L , △ and □ denote different categories, colors represent modalities, and color intensity reflects confidence.

During the alignment stage, most methods still rely on uniform static constraints, applying the same alignment intensity to all samples[17]. This paradigm overlooks the significant disparities in discriminability across different samples[29]: while strong alignment helps enhance cross-modal consistency for high-confidence samples that are easy to classify, applying the same intensity to samples with semantic ambiguity or high noise levels may be counterproductive. Forcibly compressing the distance between all positive sample pairs inherently diminishes inter-modal diversity, thereby suppressing the expression of complementary information[10, 23]. Ultimately, this leads to feature space degradation and undermines the representational advantages that multimodal models are intended to possess.

During the fusion stage, recent studies[3, 12, 41] introduce confidence- or uncertaintyaware dynamic weighting mechanisms to adaptively calibrate modality contributions. However, these methods often overlook feature magnitude differences and confidence differences across modalities. When the feature magnitude difference between modalities is large, their actual contributions to the fused representation are often already significantly differentiated. In such cases, imposing additional strong constraints may lead to unreasonable contribution reallocation and consequently interfere with effective multimodal fusion. On the other hand, when the confidence difference between modalities is small, excessively enforcing confidence-based modulation of modality contributions may restrict potential synergistic modeling across modalities.

To address the aforementioned issues, we propose an Alignment- and Calibration-driven Multimodal Learning framework (ACML), as illustrated in Figure 1. ACML consists of two core components: a dynamic cross-modal triplet alignment (DCTA) module and a differenceaware attention calibration (DAC) module. Specifically, modality-specific features are first extracted using dedicated encoders, followed by an independent classification head with multiple stochastic forward passes for sample-level confidence estimation. The features are then projected through independent layers for dimensional alignment and jointly optimized under the proposed dynamic triplet alignment mechanism. This mechanism enhances interclass discriminability while enforcing cross-modal semantic consistency for high-confidence positive pairs. For low-confidence samples, the alignment constraint is relaxed and an orthogonality encouragement strategy is introduced to preserve potential complementary information and mitigate noise propagation caused by forced alignment. Furthermore, in the DAC module, we impose consistency regularization on the weighted attention distributions across modalities for the same sample, while adaptively adjusting the regularization strength according to feature magnitude differences and confidence differences among modalities. This design enables more reasonable sample-aware fusion while avoiding unnecessary overregularization on high-magnitude modalities. Extensive experimental results demonstrate that ACML achieves superior effectiveness and robustness under both standard settings and challenging scenarios involving noisy and missing modalities.

1. We propose ACML, an alignment- and calibration-driven multimodal learning framework that achieves robust and adaptive multimodal representation learning through dynamic cross-modal alignment and difference-aware attention calibration.

2. We design DCTA to enhance semantic consistency for high-confidence positive pairs while promoting diversity between high- and low-confidence pairs, thereby preserving complementary information and mitigating noise propagation.

3. We propose DAC, which dynamically adjusts the strength of attention regularization according to feature magnitude differences and confidence differences, thereby avoiding unreasonable over-regularization on attention weights.

## 2 Related Work

## 2.1 Multimodal Alignment

Multimodal alignment aims to establish semantic correspondences across different modalities (e.g., text and images) by mapping their representations into a unified shared embedding space. According to the underlying alignment mechanisms, existing approaches can be broadly categorized into explicit and implicit alignment strategies[17]. Explicit methods [18, 35, 39] typically rely on similarity matrices to directly measure cross-modal associations, whereas implicit methods[15, 16, 21] learn joint representations in a latent space through translation or prediction-based objectives. Along the methodological evolution, contrastive learning has emerged as a dominant paradigm for multimodal alignment[18]. Its core principle is to pull positive sample pairs closer while pushing negative pairs apart, thereby enforcing discriminative cross-modal consistency in the embedding space [6, 39]. Our method follows the explicit alignment paradigm but addresses the limited consideration of modality dynamics in prior work by introducing a dynamic-aware mechanism. Moreover, we extend conventional contrastive objectives by designing a triplet alignment loss, which aims to strike a better balance between promoting cross-modal consistency and preserving complementary information.

## 2.2 Multimodal Fusion

Multimodal fusion aims to integrate complementary information from different modalities to improve performance on downstream tasks. Existing approaches are typically categorized into early, intermediate, or late fusion based on the fusion stage [2]. Early fusion methods [28, 40] combine multiple modalities directly at the data level, but they often struggle with high-dimensional or heterogeneous data. Intermediate fusion approaches [5] focus on learning joint feature representations across modalities, while late fusion methods [3, 41] emphasize semantic-level integration of individual modality predictions.

Recent studies have increasingly focused on dynamic multimodal fusion [22, 37, 42, 44] to address the observation that the relative importance of each modality may vary across samples or over time. Unlike static fusion strategies that assume fixed modality weights regardless of input conditions, dynamic fusion approaches leverage attention mechanisms or uncertainty modeling to adaptively modulate the contribution of each modality. This adaptability enables them to better handle challenging scenarios, such as degraded data quality or cross-domain variations, thereby improving both robustness and generalization. However, existing studies typically impose homogeneous constraints on fusion weights while overlooking the differences in feature norms and confidence across modalities. This oversight may lead to unreasonable constraints on modality pairs with abnormal norm ratios or highly consistent confidence, thereby suppressing the expression of complementary cross-modal information. In contrast, we introduce a pairwise constraint loss that adaptively adjusts the constraint strength for different modality pairs based on both norm and confidence differences.

## 3 Method

## 3.1 Feature Extraction and Confidence Estimation

Given a mini-batch of N multimodal samples, each of which has M modality-specific inputs and a corresponding class label. The inputs are represented as $\{ \mathbf { X } ^ { m } \} _ { m = 1 } ^ { M }$ , where $\mathbf { X } ^ { m } =$ $\{ \mathbf { x } _ { i } ^ { m } \} _ { i = 1 } ^ { N }$ denotes the m-th modality of N samples. The associated label set is $\mathbf { Y } = \{ y _ { 1 } , y _ { 2 } , \ldots , y _ { N } \}$ with $y _ { i } \in \{ 1 , \ldots , C \}$

For the m-th modality, the input $\mathbf { X } ^ { m }$ is first passed through a modality-specific encoder $f _ { \mathrm { e n c } } ^ { m } ( \cdot )$ to extract high-level feature representations:

$$
\mathbf { Z } ^ { m } = f _ { \mathrm { e n c } } ^ { m } \left( \mathbf { X } ^ { m } \right) ,\tag{1}
$$

where $\mathbf { Z } ^ { m } = [ \mathbf { z } _ { 1 } ^ { m } , \mathbf { z } _ { 2 } ^ { m } , \ldots , \mathbf { z } _ { N } ^ { m } ] ^ { \top } \in \mathbb { R } ^ { N \times d _ { m } }$ is the feature matrix, $\pmb { z } _ { i } ^ { m } \in \mathbb { R } ^ { d _ { m } }$ represents the feature vector of the i-th sample.

To obtain robust and stable confidence estimates during training, we employ a stochastic inference scheme. Specifically, we leverage the classifier $f _ { \mathrm { c l s } } ^ { m } ( \cdot )$ to construct two parallel computational paths. In the primary training path, the feature $\mathbf { z } _ { i } ^ { m }$ is directly fed into the classifier without activating dropout. This path is supervised by the standard cross-entropy loss $\mathcal { L } _ { \mathrm { c l s } } ^ { m }$ , which is to ensure that the model learns fundamental discriminative representations:

$$
\mathcal { L } _ { \mathrm { c l s } } ^ { m } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \left( \left[ s ( f _ { \mathrm { c l s } } ^ { m } ( \mathbf { z } _ { i } ^ { m } ) ) \right] _ { y _ { i } } \right) ,\tag{2}
$$

where $s ( \cdot )$ denotes the softmax activation function, and $[ \cdot ] _ { y _ { i } }$ extracts the probability corresponding to the ground-truth index. Meanwhile, to mitigate the prevalent over-confidence issue associated with a single deterministic forward pass and to enhance the quality of confidence estimation, we introduce a probabilistic auxiliary path. Under this path, we apply a dropout operator $\delta ( \cdot )$ to the same feature vector $\mathbf { z } _ { i } ^ { m }$ and execute K stochastic forward passes. In this work, we set the dropout probability to 0.1 and the number of stochastic forward passes K to 10. For each pass $k \in \{ 1 , \ldots , K \}$ , the generated stochastic prediction vector is formulated as:

$$
\mathbf { p } _ { i , k } ^ { m } = s ( f _ { \mathrm { c l s } } ^ { m } ( \delta ( \mathbf { z } _ { i } ^ { m } ) ) ) .\tag{3}
$$

The final confidence score $c _ { i } ^ { m }$ for the i-th sample is defined as the empirical mean of the ground-truth probabilities across these K stochastic predictions:

$$
c _ { i } ^ { m } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left[ \mathbf { p } _ { i , k } ^ { m } \right] _ { y _ { i } } .\tag{4}
$$

After obtaining the confidence vector $\mathbf { c } ^ { m } = [ c _ { 1 } ^ { m } , c _ { 2 } ^ { m } , \ldots , c _ { N } ^ { m } ] ^ { \top } \in \mathbb { R } ^ { N }$ , we detach it from the computational graph to serve as dynamic guidance for subsequent modules.

## 3.2 Dynamic Cross-modal Triplet Alignment

Cross-modal semantic alignment is a fundamental component of multimodal learning [17]. However, due to the varying discriminative capabilities across modalities and the need to preserve complementary information, enforcing uniform semantic consistency constraints on all sample pairs is often suboptimal. To address this issue, we propose a Dynamic Crossmodal Alignment loss, denoted as $\mathcal { L } _ { \mathrm { D C T A } }$ . The proposed loss dynamically adjusts alignment strength according to sample confidence and explicitly preserves feature separability based on confidence discrepancy, thereby alleviating noisy semantic aggregation and promoting feature diversity.

For the feature representation $\mathbf { Z } ^ { m }$ of the m-th modality, we first employ a modalityspecific projection layer $f _ { \mathrm { p r o j } } ^ { m } ( \cdot )$ , implemented as an MLP, to map the features into a shared multimodal embedding space:

$$
\mathbf { H } ^ { m } = f _ { \mathrm { p r o j } } ^ { m } ( \mathbf { Z } ^ { m } ) = \{ \mathbf { h } _ { 1 } ^ { m } , \mathbf { h } _ { 2 } ^ { m } , \dots , \mathbf { h } _ { N } ^ { m } \} ,\tag{5}
$$

where $\mathbf { h } _ { i } ^ { m } \in \mathbb { R } ^ { d }$ denotes the embedded representation of the i-th sample in the m-th modality. Within the embedding space, we compute the cosine similarity matrix $\mathbf { S } ^ { m , n } \in \mathbb { R } ^ { N \times N }$ for each modality pair $( m , n )$ , where each element $s _ { i j } ^ { m , n }$ is defined as:

$$
s _ { i j } ^ { m , n } = \frac { \mathbf h _ { i } ^ { m ^ { \top } } \mathbf h _ { j } ^ { n } } { \| \mathbf h _ { i } ^ { m } \| \cdot \| \mathbf h _ { j } ^ { n } \| } .\tag{6}
$$

To dynamically regulate the alignment strength according to paired-sample reliability, we define a joint confidence weight:

$$
w _ { i j } ^ { m , n } = \operatorname* { m i n } ( c _ { i } ^ { m } , c _ { j } ^ { n } ) .\tag{7}
$$

In this way, high-confidence positive pairs are encouraged to achieve stronger semantic consistency, while unreliable pairs are softly relaxed to alleviate noisy semantic aggregation.

The alignment term is formulated as:

$$
\mathrm { A l i g n } _ { i } ^ { m , n } = \sum _ { j = 1 } ^ { N } \mathbb { I } ( y _ { i } = y _ { j } ) w _ { i j } ^ { m , n } \exp \left( \frac { s _ { i j } ^ { m , n } } { \tau } \right) ,\tag{8}
$$

where I(·) denotes the indicator function and τ is a temperature parameter.

Although the confidence-weighted alignment effectively alleviates noisy aggregation caused by unreliable samples, it does not explicitly preserve the intrinsic representation diversity within the same class. To this end, we further introduce a confidence-gap-aware decorrelation constraint to encourage more diverse feature representations among positive pairs. Specifically, we define a confidence-gap weight as:

$$
\tilde { w } _ { i j } ^ { m , n } = | c _ { i } ^ { m } - c _ { j } ^ { n } | .\tag{9}
$$

This weight adaptively modulates the strength of the decorrelation constraint according to the confidence discrepancy between positive pairs. Based on this weighting scheme, we further construct an orthogonality-inspired decorrelation objective using the absolute feature similarity. Positive pairs with larger confidence gaps are subjected to stronger decorrelation, encouraging them to maintain lower feature correlation. As a result, the learned representations remain semantically consistent while preserving sufficient diversity, thereby retaining potential complementary information. The decorrelation term is defined as:

$$
\mathrm { D e c o r } _ { i } ^ { m , n } = \sum _ { j = 1 } ^ { N } \mathbb { I } ( y _ { i } = y _ { j } ) \tilde { w } _ { i j } ^ { m , n } \exp \left( \frac { | s _ { i j } ^ { m , n } | } { \tau } \right) .\tag{10}
$$

For negative pairs $( y _ { i } \neq y _ { j } )$ , we further introduce a repulsion term to separate irrelevant category representations in the embedding space:

$$
{ \mathrm { R e p e l } } _ { i } ^ { m , n } = \sum _ { j = 1 } ^ { N } \mathbb { I } ( y _ { i } \neq y _ { j } ) \exp \left( { \frac { s _ { i j } ^ { m , n } } { \tau } } \right) .\tag{11}
$$

Finally, we integrate the alignment, decorrelation, and repulsion terms into a unified optimization objective. The batch-wise loss between modality m and modality n is:

$$
\mathcal { L } _ { \mathrm { { D C T A } } } ^ { m , n } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \mathrm { A l i g n } _ { i } ^ { m , n } } { \mathrm { A l i g n } _ { i } ^ { m , n } + \mathrm { D e c o r } _ { i } ^ { m , n } + \mathrm { R e p e l } _ { i } ^ { m , n } } .\tag{12}
$$

The final global objective is obtained by summing all bidirectional modality-pair losses:

$$
\mathcal { L } _ { \mathrm { D C T A } } = \sum _ { 1 \leq m < n \leq M } \left( \mathcal { L } _ { \mathrm { D C T A } } ^ { m , n } + \mathcal { L } _ { \mathrm { D C T A } } ^ { n , m } \right) .\tag{13}
$$

## 3.3 Difference-aware Attention Calibration

In dynamic multimodal fusion, modality contributions are typically modulated by predictive confidence or uncertainty [3, 11, 41]. Existing methods typically adopt a unified global attention calibration strategy without adequately considering sample-wise modality differences. However, when the feature magnitude difference between modalities is large, their contributions to the fused representation are often already significantly differentiated, making additional strong calibration potentially unnecessary and even harmful to effective multimodal fusion. Meanwhile, when the confidence difference between modalities is small, excessive calibration may suppress potential synergistic information across modalities. To address these issues, we propose a difference-aware attention calibration constraint, termed $\mathcal { L } _ { \mathrm { D A C } }$ , which adaptively emphasizes modality pairs with distinguishable confidence levels and comparable feature magnitudes, thereby enabling more adaptive and effective multimodal fusion.

Specifically, for the i-th sample, we employ a modality-specific attention layer $f _ { \mathrm { a t t n } } ^ { m }$ , implemented using a modality-specific MLP followed by a Sigmoid function, to compute the attention score for each modality. The original features are then reweighted by the corresponding attention scores to obtain the attention-weighted features $\tilde { \mathbf { h } } _ { i } ^ { m }$ , as follows:

$$
\alpha _ { i } ^ { m } , \tilde { \mathbf { h } } _ { i } ^ { m } = f _ { \mathrm { a t t n } } ^ { m } ( \mathbf { h } _ { i } ^ { m } ) .\tag{14}
$$

To calibrate the relative attention allocation at the sample level, we define the pairwise attention ratio and confidence ratio for each modality pair $( m , n )$ as:

$$
{ \mathrm { A t t n R a t i o } } _ { i } ^ { m , n } = { \frac { \alpha _ { i } ^ { m } } { \alpha _ { i } ^ { m } + \alpha _ { i } ^ { n } + \varepsilon } } , \quad { \mathrm { C o n f R a t i o } } _ { i } ^ { m , n } = { \frac { c _ { i } ^ { m } } { c _ { i } ^ { m } + c _ { i } ^ { n } + \varepsilon } } ,\tag{15}
$$

where $\mathbf { A t t n R a t i o } _ { i } ^ { m , n }$ measures the relative attention allocation between modalities, while ConfRatio<sup>m,n</sup> represents their relative predictive confidence. Here, ε is a small constant introduced for numerical stability.

Furthermore, to selectively enforce reliable calibration signals, we define a differenceaware pairwise weight as:

$$
\hat { w } _ { i } ^ { m , n } = \gamma _ { i } ^ { m , n } \cdot \left( \left| c _ { i } ^ { m } - c _ { i } ^ { n } \right| + 1 \right) ,\tag{16}
$$

where $\begin{array} { r } { \gamma _ { i } ^ { m , n } = \frac { \operatorname* { m i n } ( \| \mathbf { h } _ { i } ^ { m } \| , \| \mathbf { h } _ { i } ^ { n } \| ) } { \operatorname* { m a x } ( \| \mathbf { h } _ { i } ^ { m } \| , \| \mathbf { h } _ { i } ^ { n } \| ) + \varepsilon } } \end{array}$ measures the similarity between modality magnitudes. When the magnitude difference between modalities is large, the corresponding calibration constraint is adaptively relaxed to avoid unreasonable contribution reallocation. In contrast, modality pairs with comparable feature magnitudes receive relatively larger calibration weights. Meanwhile, the confidence difference term $\left( \left| c _ { i } ^ { m } - c _ { i } ^ { n } \right| + 1 \right)$ preserves a basic alignment objective while further emphasizing modality pairs with more distinguishable reliability levels. Therefore, $\hat { w } _ { i } ^ { m , n }$ jointly models feature magnitude differences and confidence differences, enabling more adaptive and reliable attention calibration.

Finally, the difference-aware attention calibration constraint between modality m and modality n is formulated as:

$$
\mathcal { L } _ { \mathrm { D A C } } ^ { m , n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { w } _ { i } ^ { m , n } \left( { \mathrm { A t t n R a t i o } _ { i } ^ { m , n } } - { \mathrm { C o n f R a t i o } _ { i } ^ { m , n } } \right) ^ { 2 } .\tag{17}
$$

The global difference-aware attention calibration constraint is obtained by summing over all modality pairs:

$$
\mathcal { L } _ { \mathrm { D A C } } = \sum _ { 1 \leq m < n \leq M } \mathcal { L } _ { \mathrm { D A C } } ^ { m , n } .\tag{18}
$$

## 3.4 Fusion and Optimization Objective

After cross-modal alignment and attention calibration, we adopt a mean fusion strategy to aggregate all reweighted modality representations and perform final classification through the classification head $f _ { \mathrm { c l s } } ( \cdot )$

$$
\boldsymbol { \mathsf { h } } _ { i } ^ { \mathrm { f u s e } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \widetilde { \boldsymbol { \mathrm { h } } } _ { i } ^ { m } , \quad \mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \left( \left[ s \left( f _ { \mathrm { c l s } } ( \boldsymbol { \mathrm { h } } _ { i } ^ { \mathrm { f u s e } } ) \right) \right] _ { y _ { i } } \right) .\tag{19}
$$

Finally, the overall optimization objective of ACML is formulated as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C L S } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { D C T A } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { D A C } } , } \end{array}\tag{20}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { C L S } } = \mathcal { L } _ { \mathrm { c l s } } + \sum _ { m = 1 } ^ { M } \mathcal { L } _ { \mathrm { c l s } } ^ { m } } \end{array}$ denotes the summation of the fused classification loss and all unimodal classification losses, while $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are balancing hyperparameters for L<sub>DCTA</sub> and $\mathcal { L } _ { \mathrm { D A C } }$ , respectively. For clarity, we summarize the whole training procedure of ACML in Algorithm 1.

Algorithm 1 ACML   
Input: Training data D, the number of training iterations $T _ { \mathbf { \delta } }$   
Output: Well trained ACML.   
1: for iteration ∈ [1, T] do   
2: Randomly select a batch of samples $\{ \{ \mathbf { X } ^ { m } \} _ { m = 1 } ^ { M } , \mathbf { Y } \}$ from $\mathcal { D } ;$   
3: # Obtain modality-specific representation.   
4: $\{ { \bf Z } ^ { m } \} _ { m = 1 } ^ { M }  \{ f _ { \mathrm { e n c } } ^ { m } ( \mathbf { \hat { X } } ^ { m } ) \} _ { m = 1 } ^ { M }$   
5: # Get the confidence score for each modal sample.   
6: $\{ \mathbf { c } ^ { m } \} _ { m = 1 } ^ { M }  \mathrm { E q s . } ~ ( 2 ) – ( 4 )$   
7: # Map multimodal features to the same dimension.   
8: $\{ { \bf H } ^ { m } \} _ { m = 1 } ^ { M }  \{ f _ { \mathrm { p r o j } } ^ { m } ( { \bf Z } ^ { m } ) \} _ { m = 1 } ^ { M }$   
9: # Perform dynamic cross-modal triplet alignment.   
10: $\mathcal { L } _ { \mathrm { D C T A } } $ Eqs. (6)-(13)   
11: # Compute attention weights and reweight features.   
12: $\alpha ^ { m } , \tilde { \mathbf { H } } ^ { \bar { m } } \gets f _ { \mathrm { a t t n } } ^ { m } ( \mathbf { H } ^ { m } )$   
13: # Perform difference-aware attention calibration.   
14: L ← Eqs. (15)-(18)   
15: # Perform multimodal fusion and final classification.   
16: $\mathbf { H } ^ { \mathrm { f u s e } } , \mathcal { L } _ { \mathrm { c l s } } \gets \mathrm { E q } .$ . (19)   
17: # Integrate multiple constraints into the final optimization objective.   
18: $\mathcal { L }  \mathrm { E q . }$ (20)   
19: Back-propagate $\mathcal { L }$ to update model parameters;   
20: end for

## 4 Experiments and Analysis

In this section, we evaluate the effectiveness of the proposed ACML by addressing the following six Research Questions (RQs):

• RQ1 (Effectiveness & Robustness): How does ACML compare with SOTA baselines in terms of overall performance and robustness against noise?

• RQ2 (Ablation Study): What are the individual contributions of the key components within ACML to its overall performance?

• RQ3 (Visualization Analysis): How does $\mathcal { L } _ { \mathrm { D C T A } }$ influence the cross-modal feature distribution in the embedding space?

• RQ4 (Hyper-parameter Analysis): How do the critical hyper-parameters influence the stability of ACML?

• RQ5 (Applicability to Missing Modalities): Can ACML effectively enhance prompttuning frameworks as a plug-and-play module in challenging missing-modality scenarios?

• RQ6 (Scalability to Multiple Modalities): Does ACML maintain its effectiveness and scalability when extended to complex scenarios with more than two modalities $( M > 2 ) ?$

## 4.1 RQ1 (Effectiveness & Robustness)

Setting. We evaluate ACML on two representative multimodal tasks: (1) Scene Recognition on the NYU Depth V2[25] and SUN RGB-D [26] datasets, and (2) Image–Text Classification on the UPMC FOOD101 [32] dataset. Detailed dataset descriptions are provided in the supplementary material. All models are implemented in PyTorch and optimized using Adam on an NVIDIA A100 GPU. For the scene recognition task, we adopt ResNet-18 as the backbone network, with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a batch size of 16, and an embedding dimension d of 512. For the image–text classification task, ResNet-152 and BERT are used to extract image and text features, respectively, with a learning rate of $5 \times 1 0 ^ { - 5 }$ , an effective batch size of 640 achieved via gradient accumulation (i.e., batch size 32 with 20 accumulation steps), and an embedding dimension d of 2048. Following [20], we simulate real-world multimodal uncertainty by injecting Gaussian noise into images and blank noise into text. Scene recognition experiments are repeated over 10 random seeds, while image–text classification experiments are conducted with 5 random seeds. We report both the average accuracy (with standard deviation) and the worst-case accuracy. All models are trained for up to 100 epochs, with early stopping applied to prevent overfitting.

Analysis. As shown in Table 1, we compare ACML with three categories of methods: traditional fusion approaches (Concat and LateFusion), state-of-the-art dynamic learning methods (TMC[12], QMF[41], and PDF[3]), and modality-imbalance learning methods (LFM[38] and OPM[34]). The results show that ACML consistently achieves the best performance across all datasets under both clean and noisy conditions. Specifically, under the clean setting (ε = 0), ACML yields relative gains of 1.82%, 1.64%, and 0.41% in average accuracy over the strongest baseline on NYU Depth V2, SUN RGB-D, and FOOD101, respectively. As the noise level increases, the advantage of ACML becomes more pronounced on NYU Depth V2 and SUN RGB-D. For example, under severe corruption (ε = 10), the relative gains in average accuracy further increase to 6.00% and 5.36%, respectively. This trend suggests that ACML not only improves predictive performance under clean conditions but also exhibits stronger robustness against modality corruption.

## 4.2 RQ2 (Ablation Study)

Setting. We conduct ablation studies on the NYU Depth V2 dataset to systematically analyze the contributions of the dynamic cross-modal triplet alignment loss $( \mathcal { L } _ { \mathrm { D C T A } } )$ and the difference-aware attention constraint $( { \mathcal { L } } _ { \mathrm { D A C } } )$ . Furthermore, based on the final ACML framework, we further investigate the effectiveness of different optimization objectives and constraint formulations. Specifically, we replace $\mathcal { L } _ { \mathrm { D C T A } }$ with the standard supervised contrastive loss $( \mathcal { L } _ { \mathrm { S C } } )$ and the confidence-constrained supervised contrastive loss $( \mathcal { L } _ { \mathrm { C S C } } )$ , respectively. Meanwhile, we replace the proposed ${ \mathcal { L } } _ { \mathrm { D A C } }$ with a standard unweighted MSE constraint $( \mathcal { L } _ { \mathrm { M S E } } )$ to analyze the impact of difference-aware weighting on the final performance. We report the average classification accuracy with standard deviation and the worst-case classification accuracy under different noise conditions.

Table 1: Classification comparison when 50% of the modalities are corrupted with Gaussian noise, i.e., zero mean with variance of ε. We highlight the optimal results using bold text and indicate the suboptimal results with underlining for clarity.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">ε = 0.0</td><td colspan="2">ε = 5.0</td><td colspan="2">ε = 10.0</td></tr><tr><td>Avg±Std</td><td>Worst</td><td>Avg±Std</td><td>Worst</td><td>Avg±Std</td><td>Worst</td></tr><tr><td rowspan="13">NYU Depth V2</td><td>Depth</td><td>61.97±0.99</td><td>60.40</td><td>49.65±3.22</td><td>43.27</td><td>42.42±2.42</td><td>36.85</td></tr><tr><td>RGB</td><td>62.57±0.82</td><td>61.32</td><td>52.56±2.28</td><td>47.55</td><td>46.16±3.10</td><td>38.84</td></tr><tr><td>LateFusion</td><td>69.22±0.98</td><td>67.43</td><td>56.35±2.84</td><td>51.22</td><td>48.65±3.40</td><td>42.36</td></tr><tr><td>Concat</td><td>68.70±0.67</td><td>67.89</td><td>55.76±2.23</td><td>49.54</td><td>48.17±3.36</td><td>40.21</td></tr><tr><td>TMC</td><td>71.42±0.84</td><td>70.03</td><td>61.44±2.19</td><td>56.58</td><td>53.36±3.36</td><td>45.87</td></tr><tr><td>QMF</td><td>68.55±1.29</td><td>66.82</td><td>56.60±3.12</td><td>50.31</td><td>49.04±3.69</td><td>41.44</td></tr><tr><td>PDF</td><td>68.46±0.69</td><td>66.67</td><td>56.67±2.69</td><td>48.47</td><td>48.61±3.56</td><td>38.23</td></tr><tr><td>LFM</td><td>69.27±0.62</td><td>68.20</td><td>59.17±2.15</td><td>53.98</td><td>51.35±2.70</td><td>44.95</td></tr><tr><td>OPM</td><td>71.35±0.93</td><td>69.88</td><td>63.35±1.41</td><td>60.40</td><td>54.82±2.37</td><td>49.54</td></tr><tr><td>ACML</td><td>72.72±0.81</td><td>71.56</td><td>65.43±1.53</td><td>60.86</td><td>58.11±2.13</td><td>49.69</td></tr><tr><td rowspan="8">SUN RGB-D</td><td>Depth</td><td>52.58±0.67</td><td>51.43</td><td>39.05±1.06</td><td>37.15</td><td>34.66±1.61</td><td>31.08</td></tr><tr><td>RGB</td><td>57.46±0.39</td><td>56.84</td><td>48.05±0.88</td><td>45.25</td><td>42.45±1.47</td><td>38.74</td></tr><tr><td>LateFusion</td><td>59.90±0.45</td><td>59.24</td><td>47.78±2.03</td><td>43.01</td><td>41.27±2.82</td><td>35.57</td></tr><tr><td>Concat</td><td>59.52±0.50</td><td>58.58</td><td>48.54±1.63</td><td>45.12</td><td>42.74±1.98</td><td>38.96</td></tr><tr><td>TMC</td><td>61.02±0.37</td><td>60.34</td><td>50.64±1.69</td><td>47.18</td><td>43.89±3.22</td><td>37.54</td></tr><tr><td>QMF</td><td>61.11±0.26</td><td>60.59</td><td>51.44±1.06</td><td>49.13</td><td>46.45±1.61</td><td>42.71</td></tr><tr><td>PDF</td><td>60.89±0.39</td><td>60.23</td><td>50.24±1.43</td><td>46.71</td><td>44.92±2.21</td><td>40.44</td></tr><tr><td>LFM</td><td>61.98±0.25</td><td>61.64</td><td>52.84±0.96</td><td>50.55</td><td>47.36±1.72</td><td>42.58</td></tr><tr><td>OPM</td><td>62.23±0.26</td><td>61.79</td><td>51.72±1.08</td><td>49.60</td><td>43.62±1.90</td><td>40.59</td></tr><tr><td>Img</td><td>ACML</td><td>63.25±0.25 62.76</td><td>56.50±0.87</td><td>54.15</td><td>49.90±1.15</td><td>47.48</td></tr><tr><td rowspan="8">FOOD 101</td><td></td><td>64.78±0.17</td><td>64.54</td><td>34.55±0.58</td><td>33.66</td><td>32.99±0.28</td><td>32.54</td></tr><tr><td>Text</td><td>86.49±0.05</td><td>86.44</td><td>67.35±0.18</td><td>67.04</td><td>43.84±0.31</td><td>43.39</td></tr><tr><td>LateFusion</td><td>90.71±0.09</td><td>90.61</td><td>70.07±2.86</td><td>64.56</td><td>59.51±0.87</td><td>57.81</td></tr><tr><td>Concat</td><td>88.35±0.57</td><td>87.68</td><td>63.28±2.35</td><td>59.45</td><td>51.18±3.71</td><td>46.34</td></tr><tr><td>TMC</td><td>90.26±0.17</td><td>89.94</td><td>73.69±0.82</td><td>72.15</td><td>61.51±0.21</td><td>61.01</td></tr><tr><td>QMF</td><td>92.88±0.11</td><td>92.70</td><td>76.39±0.62</td><td>75.30</td><td>61.86±0.41</td><td>61.74</td></tr><tr><td>PDF</td><td>92.98±0.17</td><td>92.71</td><td>75.34±0.55</td><td>74.29</td><td>62.77±0.23</td><td>62.45</td></tr><tr><td>LFM</td><td>92.80±0.05</td><td>92.74</td><td>76.15±0.35</td><td>75.51</td><td>62.57±0.34</td><td>61.88</td></tr><tr><td>OPM</td><td>92.60±0.22</td><td>92.21</td><td>75.17±0.51</td><td>74.37</td><td>60.97±0.78</td><td>59.66</td></tr><tr><td></td><td>ACML</td><td>93.36±0.10</td><td>93.23</td><td>76.64±0.49</td><td>75.69</td><td>63.15±0.24</td><td>62.76</td></tr></table>

Analysis. As shown in Table 2, introducing either $\mathcal { L } _ { \mathrm { D C T A } }$ or $\mathcal { L } _ { \mathrm { D A C } }$ consistently improves the performance compared with the original baseline model under different noise conditions, demonstrating their effectiveness in enhancing cross-modal representation learning. Moreover, the complete ACML framework achieves the best performance in terms of both average classification accuracy and worst-case classification accuracy, further validating the complementarity between $\mathcal { L } _ { \mathrm { D C T A } }$ and $\mathcal { L } _ { \mathrm { D A C } }$ . Furthermore, replacing $\mathcal { L } _ { \mathrm { D C T A } }$ with either $\mathcal { L } _ { \mathrm { S C } }$ or $\mathcal { L } _ { \mathrm { C S C } }$ leads to inferior performance, demonstrating the superiority of the proposed dynamic cross-modal triplet alignment strategy in regularizing cross-modal representation distributions. Similarly, replacing $\mathcal { L } _ { \mathrm { D A C } }$ with the standard unweighted $\mathcal { L } _ { \mathrm { M S E } }$ constraint also degrades the performance, validating the effectiveness of the proposed difference-aware weighting mechanism.

Table 2: Ablation studies on NYU Depth V2. We report both the average performance and the worst-case performance under different noise levels.
<table><tr><td rowspan="2">Configuration</td><td colspan="2"> $\pmb { \varepsilon } = \mathbf { 0 . 0 }$ </td><td colspan="2"> $\varepsilon = 5 . 0$ </td><td colspan="2"> $\pmb { \varepsilon = 1 0 . 0 }$ </td></tr><tr><td>Avg±Std</td><td>Worst</td><td> $\mathrm { A v g \pm S t d }$ </td><td>Worst</td><td>Avg±Std</td><td>Worst</td></tr><tr><td>Baseline</td><td> $7 0 . 8 7 { \scriptstyle \pm 0 . 7 9 }$ </td><td>69.27</td><td> $6 2 . 6 1 { \scriptstyle \pm 1 . 5 3 }$ </td><td>58.87</td><td> $5 5 . 0 2 { \scriptstyle \pm 3 . 3 8 }$ </td><td>46.79</td></tr><tr><td> $\mathrm { \mathbf { B } a s e l i n e } + \mathcal { L } _ { \mathrm { D C T A } }$ </td><td> $7 2 . 3 7 { \scriptstyle \pm 0 . 8 0 }$ </td><td>70.80</td><td> $6 3 . 8 2 { \scriptstyle \pm 1 . 9 7 }$ </td><td>58.56</td><td> $5 5 . 7 0 { \scriptstyle \pm 2 . 9 1 }$ </td><td>48.62</td></tr><tr><td> $\mathrm { \mathbf { B a s e l i n e } } + \mathcal { L } _ { \mathrm { D A C } }$ </td><td> $7 1 . 6 7 { \scriptstyle \pm 0 . 8 2 }$ </td><td>70.32</td><td> $6 3 . 6 6 { \scriptstyle \pm 2 . 5 8 }$ </td><td>58.42</td><td> $5 3 . 9 3 { \scriptstyle \pm 4 . 3 3 }$ </td><td>47.59</td></tr><tr><td>ACML</td><td> $7 2 . 7 2 { \scriptstyle \pm 0 . 7 1 }$ </td><td>71.56</td><td> ${ \bf 6 5 . 4 3 { \scriptstyle \pm 1 . 6 3 } }$ </td><td>60.86</td><td> ${ \bf 5 8 . 1 1 } { \bf \pm } 2 . 6 9 $ </td><td>49.69</td></tr><tr><td> $\mathrm { A C M L } \left( \mathcal { L } _ { \mathrm { D C T A } } \right) \mathcal { L } _ { \mathrm { S C } } )$ </td><td> $7 2 . 4 8 { \scriptstyle \pm 0 . 6 9 }$ </td><td>71.10</td><td> $6 5 . 0 6 { \scriptstyle \pm 1 . 7 4 }$ </td><td>60.86</td><td> $5 7 . 0 9 { \scriptstyle \pm 2 . 7 5 }$ </td><td>51.07</td></tr><tr><td> $\mathrm { A C M L } ( \mathcal { L } _ { \mathrm { D C T A } } \to \mathcal { L } _ { \mathrm { C S C } } )$ </td><td> $7 2 . 5 5 { \scriptstyle \pm 0 . 7 6 }$ </td><td>71.41</td><td> $6 4 . 6 7 { \scriptstyle \pm 1 . 5 5 }$ </td><td>60.09</td><td> $5 7 . 4 3 { \scriptstyle \pm 2 . 3 9 }$ </td><td>51.38</td></tr><tr><td>ACML  $( \mathcal { L } _ { \mathrm { D A C } } \to \mathcal { L } _ { \mathrm { M S E } } )$ </td><td> $7 2 . 4 9 { \scriptstyle \pm 0 . 7 1 }$ </td><td>71.10</td><td> $6 4 . 7 8 { \scriptstyle \pm 2 . 5 7 }$ </td><td>58.72</td><td> $5 7 . 4 1 { \scriptstyle \pm 3 . 1 5 }$ </td><td>49.69</td></tr></table>

![](images/cb265e0e98800837bee5a1dc3765a0328194f38c5f4df57c79b1cb26df899ea1.jpg)  
Figure 2: 2D t-SNE visualization of multimodal features. $\triangle$ indicates the image modality, and □ denotes the text modality. Each color corresponds to a distinct class, with darker shades indicating higher confidence.

## 4.3 RQ3 (Visualization Analysis)

Setting. To verify whether $\mathcal { L } _ { \mathrm { D C T A } }$ effectively reshapes the representation space and improves cross-modal semantic alignment, we perform t-SNE visualizations on the FOOD101 dataset to qualitatively examine the feature distributions before and after applying $\mathcal { L } _ { \mathrm { D C T A } }$

Analysis. As shown in Figure 2, the t-SNE visualizations further reveal that, after introducing $\mathcal { L } _ { \mathrm { D C T A } }$ , high-confidence samples form more compact clusters with improved crossmodal consistency. In contrast, low-confidence samples (indicated by lighter colors) remain relatively scattered, indicating that the proposed loss effectively promotes cross-modal alignment.

![](images/94a1043cb9963b037427b16a7f4aa6e7d1c8d59992dda662706293217aec0052.jpg)  
Figure 3: Performance sensitivity to hyperparameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ across two datasets.

Table 3: Accuracy comparison on FOOD101 validation and test sets under different missing rates (baselines vs. +ACML)
<table><tr><td rowspan="2">Missing rate η</td><td colspan="2">Train/Test</td><td colspan="4">Validation Set</td><td colspan="4">Testing Set</td></tr><tr><td>Image</td><td>Text</td><td>DCP</td><td>(+ACML)</td><td>SyP</td><td>(+ACML)</td><td>DCP</td><td>(+ACML)</td><td>SyP</td><td>(+ACML)</td></tr><tr><td rowspan="3">50%</td><td>100%</td><td>50%</td><td>82.33</td><td>84.72</td><td>82.64</td><td>83.68</td><td>82.11</td><td>84.42</td><td>81.93</td><td>83.85</td></tr><tr><td>50%</td><td>100%</td><td>89.23</td><td>89.44</td><td>89.07</td><td>89.16</td><td>89.12</td><td>89.44</td><td>89.18</td><td>89.39</td></tr><tr><td>75%</td><td>75%</td><td>85.25</td><td>85.86</td><td>85.20</td><td>86.28</td><td>85.24</td><td>86.03</td><td>86.17</td><td>86.50</td></tr><tr><td rowspan="3">70%</td><td>100%</td><td>30%</td><td>79.18</td><td>80.61</td><td>79.39</td><td>80.21</td><td>78.87</td><td>80.39</td><td>79.28</td><td>80.27</td></tr><tr><td>30%</td><td>100%</td><td>87.53</td><td>88.11</td><td>87.62</td><td>87.87</td><td>87.32</td><td>87.77</td><td>87.27</td><td>87.75</td></tr><tr><td>65%</td><td>65%</td><td>82.38</td><td>83.24</td><td>82.53</td><td>83.55</td><td>81.87</td><td>83.11</td><td>82.25</td><td>83.24</td></tr><tr><td rowspan="3">90%</td><td>100%</td><td>10%</td><td>75.54</td><td>77.21</td><td>76.07</td><td>77.05</td><td>75.26</td><td>77.54</td><td>76.30</td><td>77.47</td></tr><tr><td>10%</td><td>100%</td><td>86.26</td><td>86.36</td><td>86.50</td><td>86.59</td><td>85.78</td><td>86.00</td><td>86.19</td><td>86.55</td></tr><tr><td>55%</td><td>55%</td><td>80.39</td><td>80.77</td><td>80.64</td><td>81.48</td><td>79.87</td><td>79.93</td><td>79.31</td><td>80.59</td></tr></table>

## 4.4 RQ4 (Hyper-parameter Analysis)

Setting. We evaluate the sensitivity of two key hyper-parameters, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ , on the NYU Depth V2 and FOOD101 datasets, which control the constraint strengths of $\mathcal { L } _ { \mathrm { D C T A } }$ and $\mathcal { L } _ { \mathrm { D A C } }$ , respectively. The search spaces for both hyper-parameters are set to {0.1,0.5,1,5}. We report the mean classification accuracy under three settings, including the clean setting, $\varepsilon = 5$ , and ε = 10 noisy settings, where the final result is obtained by averaging the accuracies across the three settings.

Analysis. As illustrated in Figure 3, excessively small or large values of either $\lambda _ { 1 }$ or $\lambda _ { 2 }$ lead to noticeable performance degradation. In contrast, when the hyper-parameters are within a reasonable range (e.g., 0.5 and 1), the proposed method achieves more stable and robust performance. Similar trends can be observed for both hyper-parameters, and the phenomenon becomes more evident on the larger-scale FOOD101 dataset. These results demonstrate that the proposed method maintains good stability and robustness within a reasonable hyper-parameter range.

## 4.5 RQ5 (Applicability to Missing Modalities)

Setting. To evaluate the adaptability of ACML under missing-modality conditions, we integrate it as a plug-and-play module into two prompt-tuning-based approaches, DCP[24] and SyP[43]. Experiments are conducted on the FOOD101 dataset. Following prior works[24,

XUETAL.: DYNAMIC ALIGNMENTAND CALIBRATIONFOR MULTIMODAL LEARNING13  
Table 4: Classification results (%) on the BRCA and ROSMAP datasets.
<table><tr><td rowspan="2">Method</td><td colspan="3">BRCA</td><td colspan="3">ROSMAP</td></tr><tr><td>Accuracy</td><td>Weighted  $\mathrm { F } _ { 1 }$ </td><td>Macro  $\mathrm { F } _ { 1 }$ </td><td>Accuracy</td><td> $\mathrm { F _ { 1 } }$ </td><td>AUC</td></tr><tr><td>MOGONet</td><td> $8 2 . 9 { \pm } 1 . 8 $ </td><td> $8 2 . 5 { \scriptstyle \pm 1 . 7 }$ </td><td> $7 7 . 4 { \pm } 1 . 7$ </td><td> $8 1 . 5 { \scriptstyle \pm 2 . 3 }$ </td><td> $8 2 . 1 { \pm } 1 . 2 $ </td><td> $8 7 . 4 { \pm } 1 . 2 $ </td></tr><tr><td>MMDyn</td><td>87.7±0.3</td><td> $8 8 . 0 { \pm } 0 . 5 $ </td><td> $8 4 . 5 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 4 . 2 { \pm } 1 . 3 $ </td><td> $8 4 . 6 { \scriptstyle \pm 0 . 7 }$ </td><td>91.2±0.7</td></tr><tr><td>MLCLNet</td><td>86.4±1.6</td><td> $8 7 . 8 { \pm } 1 . 7 $ </td><td> $8 2 . 6 { \pm } 1 . 8$ </td><td> $8 4 . 4 { \pm } 1 . 5 $ </td><td> $8 5 . 2 { \pm } 1 . 5 $ </td><td>89.3±1.1</td></tr><tr><td>DMIB</td><td>86.0±1.6</td><td> $8 6 . 0 { \scriptstyle \pm 0 . 8 }$ </td><td> $8 1 . 6 { \pm } 0 . 9$ </td><td> $8 4 . 9 { \pm } 1 . 8 $ </td><td> $8 5 . 3 { \pm } 1 . 7 $ </td><td>91.6±0.7</td></tr><tr><td>GCFANet</td><td>88.6±1.5</td><td> $8 8 . 9 { \pm } 1 . 6 $ </td><td> $8 5 . 3 { \pm } 1 . 6 $ </td><td> $8 6 . 3 { \pm } 1 . 4 $ </td><td> $8 8 . 3 { \pm } 1 . 6 $ </td><td>91.5±1.2</td></tr><tr><td>HCMAF</td><td> $\underline { { 8 9 . 9 \pm 0 . 2 } }$ </td><td> $9 0 . 1 { \pm } 0 . 2 $ </td><td> $8 6 . 4 { \scriptstyle \pm 0 . 6 }$ </td><td> $8 8 . 0 { \pm } 0 . 7 \ $ </td><td> $\mathbf { 8 7 . 9 2 1 . 2 }$ </td><td>91.9±0.7</td></tr><tr><td>ACML</td><td>90.7±0.2</td><td> $\mathbf { 9 1 . 1 { \pm } 0 . 2 }$ </td><td> $\mathbf { 8 8 . 1 } { \pm 0 . 2 }$ </td><td> $\mathbf { 8 8 . 2 \pm 0 . 8 }$ </td><td> $8 7 . 6 { \pm } 0 . 8 $ </td><td>92.4±0.7</td></tr></table>

43], we consider three missing ratios: 50%, 70%, and 90%. For each ratio, we further evaluate three missing patterns: text missing, image missing, and both modalities missing, where each modality is removed at a ratio of 50%. All methods keep the backbone encoders frozen and only update the prompts together with our module. Performance is reported on both the validation and test sets.

Analysis. As reported in Table 3, ACML consistently improves the performance of both DCP and SyP across all missing-modality ratios and missing patterns on both validation and test sets. Notably, the gains remain stable even under severe modality absence (e.g., 90% missing ratio), demonstrating the strong robustness of ACML to incomplete multimodal inputs. These results suggest that ACML can effectively enhance representation learning when modality information is partially unavailable and can be seamlessly integrated into different prompt-tuning frameworks as a general and effective enhancement module.

## 4.6 RQ6 (Generalization and Scalability)

Setting. To evaluate the adaptability of ACML in broader multi-modal scenarios, we conduct classification experiments on two medical multi-omics benchmark datasets: BRCA[31] and ROSMAP[1, 4]. These tasks involve three sequential omics modalities (M = 3) and particularly emphasize population-level relationship modeling. Specifically, the BRCA dataset contains 875 samples spanning five subtypes, while the ROSMAP dataset includes 351 samples for binary Alzheimer’s disease diagnosis. For the network architecture, each modality employs a lightweight encoder—consisting of a single linear layer followed by a ReLU activation—to project input features into a 2048-dimensional space $( d _ { m } = 2 0 4 8 )$ , with an MLP used for feature dimension alignment (d = 2048). The model is trained for 500 epochs using the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ . For evaluation, the BRCA dataset uses accuracy, macro $_ \mathrm { F _ { 1 } - s c o r e }$ , and weighted $_ \mathrm { F _ { 1 } - s c o r e }$ , while the ROSMAP dataset adopts accuracy, $\mathrm { F _ { 1 } - s c o r e . }$ , and AUC to comprehensively assess diagnostic performance. All results are reported as the mean over 20 random seeds.

Analysis. We compare ACML against six SOTA multimodal methods that focus on sequence-level modeling: MOGONet [31], MMDyn [12], MLCLNet [44], DMIB [7], GCFANet [45], and HCMAF [8]. As shown in Table 4, ACML achieves significant and consistent performance gains across all tasks. These results demonstrate that our method maintains strong generalization capability across different data types and more diverse multimodal scenarios.

## 5 Conclusion

This paper proposes an Alignment- and Calibration-driven Multimodal Learning framework (ACML) to address the limitations of existing methods in static alignment, which overlooks sample-wise differences and fusion strategies lacking difference-aware modeling. Specifically, ACML consists of two key modules: (i) a dynamic cross-modal triplet alignment (DCTA) module, which adaptively adjusts alignment strength according to sample confidence, enforcing cross-modal semantic consistency for high-confidence samples while preserving potential complementary information in low-confidence samples through an orthogonality encouragement strategy; and (ii) a difference-aware attention calibration (DAC) module, which adaptively adjusts the attention regularization strength according to feature magnitude differences and confidence differences across modalities, enabling more reasonable sample-aware fusion while avoiding unnecessary over-regularization. Experimental results demonstrate that ACML consistently achieves superior performance and robustness across multiple multimodal benchmark datasets, validating the effectiveness of the proposed modules. In future work, we plan to explore the integration of ACML with large-scale pre-trained foundation models, extending its applicability to complex real-world scenarios such as medical multimodal analysis, as well as multimodal understanding and generation tasks.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. 62276052), and the Medico Engineering Cooperation Funds from University of Electronic Science and Technology of China and West China Hospital (No. ZYGX2022YGRH009).

## References

[1] David A. Bennett, Julie A. Schneider, Zoe Arvanitakis, and Robert S. Wilson. Overview and findings from the religious orders study. Current Alzheimer Research, 9 (6), 2012.

[2] Pradeep K Atrey, M Anwar Hossain, Abdulmotaleb El Saddik, and Mohan S Kankanhalli. Multimodal fusion for multimedia analysis: a survey. Multimedia Systems, 16 (6), 2010.

[3] Bing Cao, Yinan Xia, Yi Ding, Changqing Zhang, and Qinghua Hu. Predictive dynamic fusion. In ICML, 2024.

[4] Philip L De Jager, Yiyi Ma, Cristin McCabe, et al. A multi-omic atlas of the human frontal cortex for aging and alzheimer’s disease research. Scientific Data, 5(1), 2018.

[5] Saisai Ding, Juncheng Li, Jun Wang, Shihui Ying, and Jun Shi. Multimodal coattention fusion network with online data augmentation for cancer subtype classification. IEEE Transactions on Medical Imaging, 43(11), 2024.

[6] Benoit Dufumier, Javiera Castillo Navarro, Devis Tuia, and Jean-Philippe Thiran. What to align in multimodal contrastive learning? In ICLR, 2025.

[7] Yingying Fang, Shuang Wu, Sheng Zhang, et al. Dynamic multimodal information bottleneck for multimodality classification. In WACV, 2024.

[8] Yanglan Gan, Hangkai Zhao, Kaili Wang, Cairong Yan, and Guobing Zou. Hcmaf: Hierarchical feature aggregation and cross-modal attention fusion framework for multiomics patient classification. IEEE Journal ofBiomedical and Health Informatics, 2025.

[9] AV Geetha, T Mala, D Priyanka, and E Uma. Multimodal emotion recognition with deep learning: advancements, challenges, and future directions. Information Fusion, 105, 2024.

[10] Kangfu Han, Dan Hu, Fenqiang Zhao, et al. Incomplete multi-modal disentanglement learning with application to alzheimer’s disease diagnosis. IEEE Transactions on Medical Imaging, 2025.

[11] Zongbo Han, Fan Yang, Junzhou Huang, Changqing Zhang, and Jianhua Yao. Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In CVPR, 2022.

[12] Zongbo Han, Changqing Zhang, Huazhu Fu, and Joey Tianyi Zhou. Trusted multi-view classification with dynamic evidential fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[13] Lingdong Kong, Xiang Xu, Jiawei Ren, et al. Multi-modal data-efficient 3d scene understanding for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[14] Deng Li, Bohao Xing, Xin Liu, et al. Deemo: De-identity multimodal emotion recognition and reasoning. In ACM MM, 2025.

[15] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In ICML, 2022.

[16] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

[17] Songtao Li and Hao Tang. Multimodal alignment and fusion: A survey: S. li, h. tang. International Journal ofComputer Vision, 134(3), 2026.

[18] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. NeurIPS, 35, 2022.

[19] Wei Liu, Jie-Lin Qiu, Wei-Long Zheng, and Bao-Liang Lu. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition. IEEE Transactions on Cognitive and Developmental Systems, 14(2), 2021.

[20] Huan Ma, Zongbo Han, Changqing Zhang, et al. Trustworthy multimodal regression with mixture of normal-inverse gamma distributions. NeurIPS, 34, 2021.

[21] Qianxia Ma, Ming Zhang, Yan Tang, and Zhen Huang. Att-sinkhorn: Multimodal alignment with sinkhorn-based deep attention architecture. In ICAC, 2023.

[22] Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, and Chen Sun. Attention bottlenecks for multimodal fusion. NeurIPS, 34, 2021.

[23] Jiahui Qu, Yuanbo Yang, Wenqian Dong, and Yufei Yang. Lds2ae: Local diffusion shared-specific autoencoder for multimodal remote sensing image classification with arbitrary missing modalities. In AAAI, volume 38, 2024.

[24] Tongkai Shi, Wei Feng, Fanhua Shang, Liang Wan, et al. Deep correlated prompting for visual recognition with missing modalities. NeurIPS, 37, 2024.

[25] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.

[26] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In CVPR, 2015.

[27] Xiaorui Su, Pengwei Hu, Dongxu Li, et al. Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning. Nature Biomedical Engineering, 2025.

[28] Xueling Suo, Mengyao Chen, Li Chen, et al. Automatic identification of parkinsonism using clinical multi-contrast brain mri: a large self-supervised vision foundation model strategy. EBioMedicine, 116, 2025.

[29] Hu Wang, Yuanhong Chen, Congbo Ma, Jodie Avery, Louise Hull, and Gustavo Carneiro. Multi-modal learning with missing modality via shared-specific feature modelling. In CVPR, 2023.

[30] Qun Wang, Feng Zhu, Ge Wu, et al. Object-level and scene-level feature aggregation with clip for scene recognition. Information Fusion, 120, 2025.

[31] Tongxin Wang, Wei Shao, Zhi Huang, et al. Mogonet integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification. Nature Communications, 12(1), 2021.

[32] Xin Wang, Devinder Kumar, Nicolas Thome, Matthieu Cord, and Frederic Precioso. Recipe recognition with large multimodal food dataset. In ICMEW, 2015.

[33] Zheng Wang, Zhenwei Gao, Yang Yang, et al. Geometric matching for cross-modal retrieval. IEEE Transactions on Neural Networks and Learning Systems, 36(3), 2024.

[34] Yake Wei, Di Hu, Henghui Du, and Ji-Rong Wen. On-the-fly modulation for balanced multimodal learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(1), 2025.

[35] Jie Xu, Huayi Tang, Yazhou Ren, et al. Multi-level feature learning for contrastive multi-view clustering. In CVPR, 2022.

[36] Chonghua Xue, Sahana S Kowshik, Diala Lteif, et al. Ai-based differential diagnosis of dementia etiologies on multimodal data. Nature Medicine, 30(10), 2024.

[37] Zihui Xue and Radu Marculescu. Dynamic multimodal fusion. In CVPR, 2023.

[38] Yang Yang, Fengqiang Wan, Qing-Yuan Jiang, and Yi Xu. Facilitating multimodal classification via dynamically learning modality gap. NeurIPS, 37, 2024.

[39] Xin Yuan, Zhe Lin, Jason Kuen, Jianming Zhang, Yilin Wang, Michael Maire, Ajinkya Kale, and Baldo Faieta. Multimodal contrastive training for visual representation learning. In CVPR, 2021.

[40] Ying Zeng, Wenjun Yan, Sijie Mai, and Haifeng Hu. Disentanglement translation network for multimodal sentiment analysis. Information Fusion, 102, 2024.

[41] Qingyang Zhang, Haitao Wu, Changqing Zhang, et al. Provable dynamic fusion for low-quality multimodal data. In ICML, 2023.

[42] Qingyang Zhang, Yake Wei, Zongbo Han, et al. Multimodal fusion on low-quality data: A comprehensive survey. Information Fusion, 135, 2026.

[43] Zhihui Zhang, Luanyuan Dai, Qika Lin, Yunfeng Diao, Guangyin Jin, Yufei Guo, Jing Zhang, and Xiaoshuai Hao. Synergistic prompting for robust visual recognition with missing modalities. In ICCV, 2025.

[44] Xiao Zheng, Chang Tang, Zhiguo Wan, Chengyu Hu, and Wei Zhang. Multi-level confidence learning for trustworthy multimodal classification. In AAAI, volume 37, 2023.

[45] Xiao Zheng, Minhui Wang, Kai Huang, and En Zhu. Global and cross-modal feature aggregation for multi-omics data classification and application on drug response prediction. Information Fusion, 102, 2024.

[46] Wujie Zhou, Shaohua Dong, Jingsheng Lei, and Lu Yu. Mtanet: Multitask-aware network with hierarchical multimodal fusion for rgb-t urban scene understanding. IEEE Transactions on Intelligent Vehicles, 8(1), 2022.

[47] Xiaoguang Zhu, Lianlong Sun, Yang Liu, et al. Causal debiasing medical multimodal representation learning with missing modalities. arXiv preprint arXiv:2509.05615, 2025.