# Diferent Changes Require Diferent Reasoning: Change-Type-Specialized Experts for Robust Change Captioning

Jiyoung Park<sup>\*</sup> , InJae Oh<sup>\*</sup> , and Jung Uk Kim<sup>†</sup>

Kyung Hee University, Yong-in, South Korea {jy0117, seanoh, ju.kim}@khu.ac.kr

Abstract. Change captioning is the task of generating natural language descriptions that explain the changes between a pair of images. Although diferent change types (e.g., color shifts, object additions) exhibit distinct visual cues and require specialized reasoning processes, existing methods often overlook these distinctions. To address this limitation, we propose Multi-Expert Diagnosis for Image Change (MEDIC), a novel framework that introduces change-type awareness by explicitly modeling change categories. We build our MEDIC as a memory network to dynamically retrieve type-relevant visual patterns conditioned on the input. This design allows each expert to flexibly capture diverse variations within each change type and focus on the most informative cues for its designated change type. By routing inputs through type-specialized experts and learning dedicated representations for each change category, MEDIC generates more precise and type-aware change descriptions. Extensive experiments demonstrate that proposed MEDIC consistently outperforms across diverse and challenging datasets. The code is available at https://github.com/VisualAIKHU/MEDIC.

Keywords: Change Captioning · Memory Network · Mixture-of-Experts

## A Introduction

In many vision-based applications, understanding how a scene evolves over time is essential. Change captioning [16, 38] addresses this need by generating natural language descriptions that explain the visual diferences between two images taken at diferent times, enabling intuitive communication of what has changed and how it appears. Such capability is particularly valuable in real-world scenarios such as surveillance [12, 16] and medical diagnostics [29], where precise and reliable interpretation of visual changes is critical.

Research in change captioning has progressed through several stages: early pixel- or feature-level diferencing with simple encoder-decoder captioning [16,38], improved alignment and robustness via cross-view matching, distractor suppression, and structure-aware modeling [47–49], and more recent generative and reasoning-centric approaches that incorporate contrastive or difusion objectives, adversarial hard negatives, MLLM/VLM-based reasoning, and textual compositional reasoning [3, 6, 30, 39, 59].

![](images/62d81db75ea59ee21ea744bf84de06ebe944a11a4609276131acebf631c4c145.jpg)  
Fig. S.1: Conceptual comparison between (a) MEDIC <sub>Fig.</sub> <sub>S.2:</sub> <sub>CIDEr</sub> <sub>comparisons</sub> and (b) existing methods using a color-change example. <sub>across change types.</sub>

Despite these advances, most prior methods still assume that diverse visual changes can be addressed through a single reasoning process. Color change depends on subtle pixel-level intensity variations [33], and object addition requires background separation and object-level reasoning [34]. Nevertheless, existing methods are constrained to process color variations, object additions, and spatial movements through a single pipeline even though each type relies on diferent visual evidence. As shown in Fig. S.1 (b), when these heterogeneous cues are processed within a single unified pipeline, the model often misinterprets the nature of the change-type (e.g., describing a red cylinder that became green (color change-type) as ‘the green cylinder appeared’ (add change-type) rather than ‘the red cylinder that is behind the yellow metal cube became green’). They overlook the fact that diferent change types rely on distinct visual cues.

To address this issue, we introduce MEDIC (Multi-Expert Diagnosis for Image Change), a new framework that models change captioning as a type-aware process. As shown in Fig. S.1 (a), MEDIC employs a two-stage routing mechanism inspired by mixture-of-experts (MoE) [26]: a change router first identifies whether a meaningful change is present, and a type router then assigns the input to a specialized expert dedicated to the corresponding change type. We design each expert as a memory network that retrieves diverse type-relevant visual patterns, enabling adaptive and discriminative reasoning beyond conventional feed-forward experts [26]. To support type-aware specialization, we introduce three training losses—routing loss, expert consistency loss, and expert disentangle loss—that collectively promote correct expert assignment, enforce type-specific representation learning, and maintain clear separation among expert representations. As a result, MEDIC consistently outperforms existing state-of-the-art, code-released methods such as DIRL [47] across all change types and datasets (see Fig. S.2), showing the efectiveness of explicitly modeling change types in change captioning.

The main contributions of our work are as follows:

– We introduce MEDIC, a novel type-aware change captioning framework that addresses the limitations that does not consider change types. By using specialized experts for each change category, MEDIC prevents reasoning confusion and enables type-aware interpretation.

– We design specialized memory-based experts that dynamically retrieve inputdependent patterns within each change type, resulting in consistent performance gains across all datasets and change types.

– We develop three training losses—routing loss, expert consistency loss, and expert disentangle loss—to learn type-specific representations while encouraging clear separation across change categories.

## B Related Works

## B.1 Change Captioning

Change captioning generates descriptions of semantic diferences between image pairs. Early methods used aligned pairs [16] or feature subtraction [38], but struggled with distractors such as viewpoint shifts. Recent methods improve robustness through relation-aware diference modeling and distractor suppression: SRDRL [52] and R<sup>3</sup>Net [51] learn semantic or relation-embedded diferences, SCORER [49] and DIRL [47] use contrastive and relational learning, SMART [48], NCT [46], and I3N [57] exploit syntactic, neighborhood, or cross-view cues, and MURAT [56] and CHEERS [31] further improve multi-grained aggregation and change-entity-guided disentanglement.

Despite these advances, most methods still use a unified, type-agnostic pipeline. They do not explicitly adapt visual diference reasoning to the change type MEDIC addresses this limitation with type-specific experts.

## B.2 Mixture of Experts

The Mixture of Experts (MoE) framework routes inputs to specialized subnetworks, or ‘experts,’ each handling a subset of the input space [15]. Sparsely-Gated MoE [42] enabled scalable training by activating only a few experts per input. This design was later adopted in large-scale models such as GShard [25] and Switch Transformers [8] to boost capacity without proportional compute.

Beyond language modeling, MoE has been efective in vision/multimodal learning. V-MoE [41] introduced sparsely activated experts into Vision Transformers, and later works extended MoE to structured domains like scene decomposi tion [58] and visual tracking [4], validating its strength in handling diverse inputs and enabling specialization. Unlike conventional MoE designs that rely on shared representations between experts, MEDIC specializes in distinct change categories.

## B.3 Memory Network

Memory networks were introduced to equip models that support content-based retrieval. Notable examples include Dynamic Memory Networks [23] and Key-Value Memory Networks [36], which improve reasoning by explicitly modeling memory access mechanisms. In computer vision, such networks have been applied to various tasks, including object tracking [60], predictive representation learning [9], and video question answering [7]. Key-value memory, in particular, enables eficient and interpretable retrieval through query-key matching [24, 35], and recent works have extended this design by incorporating multi-modal cues or spatial priors for fine-grained alignment [19–22, 54].

![](images/813fcb38522b95dd346f68b41a5c215fd3f9a8169b5fd862f8f70973f9ac62c7.jpg)  
Fig. S.3: Overview of our Multi-Expert Diagnosis for Image Change (MEDIC) with a color change example. MEDIC consists of (1) a two-stage router, which first determines the presence of change and then assigns input to specialized experts based on the predicted change type, and (2) change-type experts, each designed to capture finegrained, type-specific visual cues via dynamic memory-based retrieval. L denotes concatenation.

Motivated by these advances, we design each expert as a key-value memory. It enables type-specific reasoning by dynamically attending to relevant features, supporting more adaptive and semantically grounded change reasoning.

## C Methodology

## C.1 Overall Architecture

As shown in Fig. S.3, we propose MEDIC to generate accurate and type-aware change descriptions by explicitly modeling change-type diversity. We describe how MEDIC can be integrated into existing change captioning pipelines that use conventional visual diference encoders and transformer decoders. Given a pair of before image $I ^ { b }$ and after image $I ^ { a }$ , visual features $f ^ { b }$ and $f ^ { a }$ are first extracted using the same image encoder, ResNet-101 [10]. These features are passed to a visual diference encoder that highlights potential change cues between the two images, referred to as visual diference features $\hat { f } ^ { b }$ and ${ \hat { f } } ^ { a }$ . They are concatenated to form paired feature $F ^ { p } { = } [ \hat { f } ^ { b } ; \hat { f } ^ { a } ] \ \in \ \mathbb { R } ^ { h \times w \times 2 d } \ ( h , \ w$ , and d indicate height, width, and feature dimension, respectively). Then it is fed into the MEDIC.

In the MEDIC, two-stage routers guide the interaction with a set of T changetype experts. A change router first detects whether a change has occurred. If the change is detected, then a type router estimates the distribution over the $( T { - } 1 )$ actual change types $( e . g .$ , color, texture, add) and softly routes the paired feature $F ^ { p }$ to the change-type experts based on the estimated type distribution. The set of expert outputs are denoted as ${ { F } ^ { m } } { = } \{ { { F } ^ { m _ { t } } } \} _ { t = 1 } ^ { T - 1 }$ where each $F ^ { m _ { t } } \in \mathbb { R } ^ { h \times w \times 2 d }$ These outputs are then aggregated via a weighted summation according to the predicted type distribution to form a MEDIC feature $\hat { F } ^ { m } \in \mathbb { R } ^ { h \times w \times 2 d }$ . In contrast, if no-change is detected, only the output of the no-change expert (T-th expert) is used as $\hat { F } ^ { m }$ . Then, $\hat { F } ^ { m }$ and $F ^ { p }$ are concatenated and fed into a transformer decoder to generate the change caption.

## C.2 Two-stage Router

The two-stage router in MEDIC routes the paired feature $F ^ { p }$ based on a two-step decision process: (1) the change router determines whether a change is present, and (2) if so, the type router estimates the distribution over change types.

(1) Change Router. To determine whether a change is present, the change router applies a classifier to the input $F ^ { p }$ . For the i-th sample, it encodes pixellevel logits $\mathbf { z } _ { \mathrm { c h g } } ^ { i } \in \mathbb { R } ^ { h \times w \times 2 }$ , then aggregated via global average pooling (GAP) and passed through a softmax to obtain the change probability vector $p ^ { i } \in \mathbb { R } ^ { 2 }$

The change router is trained using cross-entropy loss:

$$
\mathcal { L } _ { c h g } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log p _ { y ^ { i } } ^ { i } ,\tag{S.1}
$$

where B is the batch size, $y ^ { i } \in \{ 0 , 1 \}$ denotes the ground-truth label for the i-th input (1: change, 0: no-change), and $p _ { y } ^ { i }$ <sub>i</sub> is the predicted probability in $p ^ { i }$

(2) Type Router. For inputs where a change is detected, the type router takes $F ^ { p }$ as input and outputs pixel-level logits $\mathbf { z } _ { \mathrm { t y p e } } ~ \in ~ \mathbb { R } ^ { h \times w \times ( T - \bar { 1 } ) }$ . These logits are pooled with GAP and passed through softmax to yield the change-type probability vector $p _ { \mathrm { t y p e } } ^ { i } \in \bar { \mathbb { R } } ^ { T - 1 }$ . Here, (T−1) denotes the number of change types, excluding the no-change category. The type router is also trained using the cross-entropy loss:

$$
\mathcal { L } _ { t y p e } = - \frac { 1 } { B _ { \mathrm { c h g } } } \sum _ { i = 1 } ^ { B _ { \mathrm { c h g } } } \log p _ { \mathrm { t y p e } , y ^ { i } } ^ { i } ,\tag{S.2}
$$

where $B _ { \mathrm { c h g } }$ is the number of samples with actual changes, $y ^ { i } \in \{ 1 , 2 , \dotsc , T - 1 \}$ is the ground-truth change-type label for the i-th sample, and $p _ { \mathrm { t y p e } , y ^ { i } } ^ { i }$ is the predicted probability for the correct change type.

![](images/4e367cb0cd044e699ab2d7990af2c4959ffa8657847669a11462efb7fd81e10b.jpg)  
Fig. S.4: Detailed architecture of the change-type memory-expert. Paired feature $F ^ { p }$ is split into N tokens, each independently processed through a memory network (illustrated here with the n-th token). Each token retrieves type-specific cues via addressing, and the outputs are reassembled into the spatial feature ${ \mathbf { } } { \mathbf { } } { \mathbf { } } F ^ { m _ { t } }$

The final routing loss $\mathcal { L } _ { r o u t e r }$ is defined as:

$$
\mathcal { L } _ { r o u t e r } = \mathcal { L } _ { c h g } + \mathcal { L } _ { t y p e } ,\tag{S.3}
$$

The $\mathcal { L } _ { r o u t e r }$ encourages the router to adaptively select and apply suitable reasoning according to the change type.

## C.3 Memory Network for Change-type Experts

We design expert, specialized for a specific change type $t \in \{ 1 , . . . , T \}$ , as a key-value memory network to capture distinct visual cues. As shown in Fig. S.4, the memory of the t-th expert is defined as $M ^ { t } = \{ M ^ { k _ { t } } , M ^ { v _ { t } } \}$ , where $M ^ { k _ { t } } = \{ m _ { \ell } ^ { k _ { t } } \} _ { \ell = 1 } ^ { L }$ and $M ^ { v _ { t } } = \{ m _ { \ell } ^ { v _ { t } } \} _ { \ell = 1 } ^ { L }$ denote the key and value memory, each with L slots of dimension $\mathbb { R } ^ { 2 d }$

To embed diverse grid-level representations of each change type, we divide the paired feature $F ^ { p }$ into N=hw tokens $\{ f _ { n } ^ { p } \} _ { n = 1 } ^ { N }$ where each $f _ { n } ^ { p } \in \mathbb { R } ^ { 2 d }$ . For each token, we compute cosine similarity with every key memory slot to identify relevant entries for retrieval, denoted as:

$$
d ( f _ { n } ^ { p } , m _ { \ell } ^ { k _ { t } } ) = \frac { f _ { n } ^ { p } \cdot m _ { \ell } ^ { k _ { t } } } { \| f _ { n } ^ { p } \| \| m _ { \ell } ^ { k _ { t } } \| } , \quad \mathrm { f o r } \ \ell = 1 , . . . , L .\tag{S.4}
$$

To retrieve information from the value memory, we convert these similarity scores into normalized read weights over memory slots, which we refer to as the addressing vector $\alpha _ { n } ^ { t } { = } \{ \alpha _ { n , \ell } ^ { t } \} _ { \ell { = } 1 } ^ { L } \in \mathbb { R } ^ { L }$ . Specifically, it is computed by a softmax with temperature τ:

$$
\alpha _ { n , \ell } ^ { t } = \frac { \exp \Big ( d ( f _ { n } ^ { p } , m _ { \ell } ^ { k _ { t } } ) / \tau \Big ) } { \sum _ { s = 1 } ^ { L } \exp \Big ( d ( f _ { n } ^ { p } , m _ { s } ^ { k _ { t } } ) / \tau \Big ) } .\tag{S.5}
$$

Using this addressing vector, the output token $f _ { n } ^ { m _ { t } }$ is retrieved by a weighted sum over the value memory:

$$
f _ { n } ^ { m _ { t } } = \sum _ { \ell = 1 } ^ { L } \alpha _ { n , \ell } ^ { t } \cdot m _ { \ell } ^ { v _ { t } } .\tag{S.6}
$$

![](images/b347b69354da4747b3478bbfe3ee76553de22ed0d80ad87961a2150467253a09.jpg)  
Fig. S.5: Overview of the expert consistency loss for (1) symmetric and (2) asymmetric change types, guiding each expert to capture its change-type behavior. For each input pair, paired and reversed features are passed through the expert of the ground-truth change type. Symmetric types $( e . g .$ , ‘color’) generate type-invariant representations under reversal, whereas asymmetric types $\left( e . g . , \mathrm { \langle d r o p ^ { \prime } \right) }$ yield type-variant representations.

Applying this retrieval process to all N tokens yields output tokens $\{ f _ { n } ^ { m _ { t } } \} _ { n = 1 } ^ { N }$ which are reshaped to the original spatial layout to obtain the type feature $F ^ { m _ { t } } \in \mathbb { R } ^ { h \times w \times 2 d }$ for the t-th expert. The memory-based design allows each expert to dynamically retrieve input-adaptive patterns for its change type, enhancing generalization across diverse scenes.

Finally, the ouput of the MEDIC feature $\hat { F } ^ { m } \in \mathbb { R } ^ { h \times w \times 2 d }$ is obtained by aggregating the expert outputs based on the routing decisions made by the change and type routers. The aggregation is defined as follows:

$$
\hat { F } ^ { m } = \left\{ \begin{array} { l l } { \sum _ { t = 1 } ^ { T - 1 } p _ { \mathrm { t y p e } , t } \cdot F ^ { m _ { t } } , } & { \mathrm { i f ~ a ~ c h a n g e ~ i s ~ d e t e c t e d , } } \\ { F ^ { m _ { T } } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{S.7}
$$

where $F ^ { m _ { T } }$ denotes the no-change expert output. This soft aggregation adaptively integrates expert outputs based on the predicted change type distribution.

## C.4 Expert Consistency Loss

To encourage each change-type expert to learn more type-specific representations, we introduce an expert consistency loss that exploits the intrinsic structural property of change types under input reversal.

As shown in Fig. S.5 (1), a ‘color’ change from red to green remains a ‘color change when reversed to green to red. Since the type label is preserved under swapping, we define such types as symmetric. In contrast, as shown in Fig. S.5 (2), a ‘drop’ becomes an ‘add’ when the input order is reversed. Because the type label changes under reversal, we define such cases as asymmetric.

Based on this, we construct two paired features: (1) original $F ^ { p } { = } [ \hat { f } ^ { b } ; \hat { f } ^ { a } ]$ and (2) reversed $F ^ { r } { = } [ \hat { f } ^ { a } ; \hat { f } ^ { b } ]$ , where the ‘before’ and ‘after’ features are swapped. Both are passed through the same type expert corresponding to the ground-truth change type, producing $F ^ { m } , F ^ { m r } \in \mathbb { R } ^ { h \times w \times 2 d }$ . We normalize both along the feature dimension and compute cosine similarity between corresponding features. The expert consistency loss is defined based on change-type symmetry as:

$$
\mathcal { L } _ { c o n } \mathrm { = } \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { B _ { s y m } } \sum _ { i = 1 } ^ { B _ { s y m } } \left( 1 - \cos ( F _ { i } ^ { m } , F _ { i } ^ { m r } ) \right) \mathrm { , ~ } } & { \mathrm { i f ~ } s y m m e t r y , } \\ { \displaystyle \frac { 1 } { B _ { a s y m } } \sum _ { i = 1 } ^ { B _ { a s y m } } \left( 1 + \cos ( F _ { i } ^ { m } , F _ { i } ^ { m r } ) \right) \mathrm { , ~ } } & { \mathrm { i f ~ } a s y m m e t r y , } \end{array} \right.\tag{S.8}
$$

where $B _ { s y m }$ and $B _ { a s y m }$ denote the number of samples for symmetric and asymmetric types, respectively.

Therefore, the loss enforces directional alignment between $F ^ { m }$ and $F ^ { m r }$ for symmetric types, while encouraging directional opposition for asymmetric types. By leveraging this type-specific orientation signal, each expert learns to encode the intrinsic characteristics that define its corresponding change type.

## C.5 Expert Disentangle Loss

We propose an expert disentangle loss to explicitly separate the representation spaces of diferent change-type experts and the no-change expert. While the expert consistency loss guides each expert to capture type-intrinsic characteristics under input reversal, it does not prevent representations from diferent experts from becoming overly similar, particularly for visually correlated change types. The disentangle loss addresses this by promoting inter-type separability and isolating no-change features from change features. It consists of (1) an intra-group term that increases separation among change-type experts and (2) an inter-group term that separates change-type experts from the no-change expert.

For each expert $t ,$ we compute a summary representation $F _ { a v g } ^ { m _ { t } }$ by applying GAP over its output feature $F ^ { m _ { t } }$ . Let $\tau$ denote the set of change types, and $T$ the no-change expert. The cosine similarity between any pair of expert representations is defined as $d _ { t , t ^ { \prime } } = \cos ( F _ { a v q } ^ { m _ { t } } , F _ { a v g } ^ { m _ { t ^ { \prime } } } )$ . The intra-group loss is defined as follows:

$$
\mathcal { L } _ { i n t r a } = \frac { 2 } { | \mathcal { T } | ( | \mathcal { T } | - 1 ) } \sum _ { \stackrel { t , t ^ { \prime } \in \mathcal { T } } { t < t ^ { \prime } } } \left[ \operatorname* { m a x } \left( 0 , d _ { t , t ^ { \prime } } - \delta _ { i n t r a } \right) \right] ^ { 2 } .\tag{S.9}
$$

In addition, the inter-group loss is defined as:

$$
\mathcal { L } _ { i n t e r } = \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } } \left[ \operatorname* { m a x } \left( 0 , d _ { t , T } - \delta _ { i n t e r } \right) \right] ^ { 2 } ,\tag{S.10}
$$

where $\delta _ { i n t r a }$ and $\delta _ { i n t e r }$ are margin hyperparameters for intra- and inter-group separation, respectively. The total expert disentangle loss is computed as follows:

$$
\mathcal { L } _ { d i s } = \mathcal { L } _ { i n t r a } + \mathcal { L } _ { i n t e r } .\tag{S.11}
$$

Together, these terms explicitly prevent inter-type representation overlap, ensuring that each expert occupies a distinct region in the embedding space while maintaining clear separation from the no-change expert. A detailed figure is included in the supplementary material.

## C.6 Overall Objective

The final training objective combines the captioning loss with all auxiliary terms:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c a p } + \lambda _ { 1 } \mathcal { L } _ { r o u t e r } + \lambda _ { 2 } \mathcal { L } _ { c o n } + \lambda _ { 3 } \mathcal { L } _ { d i s } ,\tag{S.12}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ are hyperparameters. $\mathcal { L } _ { c a p }$ denotes the captioning loss used to supervise the transformer decoder.

## D Experiments

## D.1 Datasets and Evaluation Metrics

Datasets. We conduct experiments on four datasets: CLEVR-DC [18], CLEVR-Change [38], Spot-the-Dif [16], and Image Editing Request [44]. CLEVR-DC includes 48,000 image pairs, featuring more drastic scene diferences due to strong distractors. We use the oficial split: 85% for training, 5% for validation, and 10% for testing. CLEVR-Change contains 79,606 image pairs and 493,735 associated captions. We use the oficial split of 67,660/3,976/7,970 for train/val/test. Spotthe-Dif consists of 13,192 surveillance-based image pairs with illumination changes and uses a standard 8:1:1 split. Image Editing Request (IER) dataset comprises 3,939 image pairs and 5,695 editing instructions. We adopt the oficial split: 3,061 pairs for training, 383 for validation, and 495 for testing.

CLEVR-DC and CLEVR-Change follow a single change setting with six predefined change types (color, texture, add, drop, move, no-change). Spot-the-Dif supports both single- and multi-change settings and contains four types (add, drop, move, no-change). Image Editing Request also follows a single change setting with seven types (add, drop, replace, background, illumination, style, resize). The change-type labels for Spot-the-Dif and IER dataset are constructed using an GPT-4o [14] (see supplementary material for details).

Evaluation Metrics. We evaluate the caption quality using five metrics: BLEU-4 (B) [37], METEOR (M) [2], ROUGE-L (R) [32], CIDEr (C) [53], SPICE (S) [1].

## D.2 Implementation Details

We apply MEDIC to representative recent change captioning models with publicly available code: SCORER [49], SMART [48], and DIRL [47]. We use $L { = } 1 0 0$ memory slots. The number of experts is determined by the change-type taxonomy of each dataset. Specifically, we allocate one expert to each change type, including the no-change type. Thus, we use T=6 for CLEVR-DC and CLEVR-Change, $T { = } 4$ for Spot-the-Dif, and $T = 7$ for Image-Editing-Request. We set $\tau { = } 0 . 1 , \delta _ { i n t r a } { = } 0 . 7 .$ and $\delta _ { i n t e r } { = } 0 . 2$ . The weights are $\lambda _ { 1 } { = } 0 . 1 , \lambda _ { 2 } { = } 0 . 0 1$ , and $\lambda _ { 3 } \mathrm { = } 0 . 1$ . All experiments use a single NVIDIA RTX 4090 GPU.

Table S.1: Comparison of change captioning performance on CLEVR-DC (left), CLEVR-Change (center), and Spot-the-Dif (right) datasets under single-change setting. Note that, our MEDIC is compared with three state-of-the-art baselines (SCORER [49], SMART [48], and DIRL [47]) with publicly available source code.
<table><tr><td rowspan="2">Method</td><td colspan="5">CLEVR-DC Dataset</td><td colspan="5">CLEVR-Change Dataset</td><td colspan="5">Spot-the-Diff Dataset</td></tr><tr><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td></tr><tr><td>DUDA (ICCV&#x27;19) [38]</td><td>40.3</td><td>27.1</td><td></td><td>56.7</td><td>16.1</td><td>47.3</td><td>33.9</td><td></td><td>112.3</td><td>24.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DUDA+ (CVPR&#x27;2i) [11]</td><td></td><td></td><td></td><td></td><td></td><td>51.2</td><td>37.7</td><td>70.5</td><td>115.4</td><td>31.1</td><td>8.1</td><td>12.5</td><td></td><td>34.5</td><td></td></tr><tr><td>M-VAM (ECCV&#x27;20) [43]</td><td></td><td></td><td></td><td></td><td></td><td>50.3</td><td>37.0</td><td>69.7</td><td>114.9</td><td>30.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VACC (ICCV’21) [17]</td><td>45.0</td><td>29.3</td><td></td><td>71.7</td><td>17.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MCCFormers-D (ICCV&#x27;21) [40]</td><td></td><td></td><td></td><td></td><td></td><td>52.4</td><td>38.3</td><td></td><td>121.6</td><td>26.8</td><td>10.0</td><td>12.4</td><td></td><td>43.1</td><td>18.3</td></tr><tr><td>MCCFormers-S (ICCV&#x27;21) [40]</td><td></td><td></td><td></td><td></td><td></td><td>57.4</td><td>41.2</td><td></td><td>125.5</td><td>32.4</td><td></td><td>12.3</td><td></td><td>41.6</td><td>16.3</td></tr><tr><td>PCL w/o Pretrain (AAAÍ&#x27;22) [55]</td><td></td><td></td><td></td><td></td><td></td><td>32.7</td><td>27.7</td><td>57.2</td><td>89.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NCT (TMM&#x27;23) [46]</td><td>47.5</td><td>32.5</td><td>65.1</td><td>76.9</td><td>15.6</td><td>55.1</td><td>40.2</td><td>73.8</td><td>124.1</td><td>32.9</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VARD-Trans (TIP’23) [45]</td><td>48.3</td><td>32.4</td><td></td><td>77.6</td><td>15.4</td><td>55.4</td><td>40.1</td><td>73.8</td><td>126.4</td><td>32.6</td><td></td><td>12.5</td><td>29.3</td><td>30.3</td><td>17.3</td></tr><tr><td>I3N-TD (TMM&#x27;23) [57]</td><td></td><td></td><td></td><td></td><td></td><td>55.8</td><td>40.6</td><td>73.9</td><td>125.6</td><td>32.8</td><td></td><td>13.0</td><td>31.5</td><td>42.7</td><td>18.6</td></tr><tr><td>RDD+ACR (AAAI&#x27;25) [30]</td><td></td><td></td><td></td><td></td><td></td><td>56.1</td><td>41.3</td><td>75.0</td><td>128.1</td><td>33.5</td><td>9.2</td><td>13.9</td><td>31.0</td><td>43.6</td><td></td></tr><tr><td>DECIDER (ÀAAI&#x27;25) [59]</td><td></td><td></td><td></td><td></td><td></td><td>56.4</td><td>39.7</td><td>75.3</td><td>131.3</td><td></td><td>10.7</td><td>14.2</td><td>41.6</td><td>39.9</td><td></td></tr><tr><td>MCT-CCDiff (TIP&#x27;25) [13]</td><td></td><td></td><td></td><td></td><td></td><td>57.5</td><td>40.6</td><td>75.6</td><td>131.7</td><td></td><td>10.8</td><td>14.5</td><td>35.5</td><td>41.7</td><td></td></tr><tr><td>SCORER (ICCV&#x27;23) [49]</td><td>49.4</td><td>33.4</td><td>66.1</td><td>83.7</td><td>16.2</td><td>56.3</td><td>41.2</td><td>74.5</td><td>126.8</td><td>33.3</td><td>10.2</td><td>12.2</td><td>1</td><td>38.9</td><td>18.4</td></tr><tr><td>SCORER + MEDIC (Ours)</td><td>56.0</td><td>35.9</td><td>70.1</td><td>97.4</td><td>19.0</td><td>57.6</td><td>41.8</td><td>75.5</td><td>130.7</td><td>33.7</td><td>10.2</td><td>12.4</td><td>32.9</td><td>39.2</td><td>18.4</td></tr><tr><td>SMART (TPAMI&#x27;24) [48]</td><td>48.3</td><td>30.7</td><td>65.2</td><td>81.2</td><td>16.0</td><td>56.1</td><td>40.8</td><td>74.2</td><td>127.0</td><td>33.4</td><td></td><td>13.5</td><td>31.6</td><td>39.4</td><td>19.0</td></tr><tr><td>SMART + MEDIC (Ours)</td><td>57.1</td><td>35.3</td><td>70.9</td><td>98.9</td><td>19.8</td><td>56.4</td><td>42.5</td><td>75.9</td><td>128.1</td><td>34.6</td><td>9.1</td><td>14.2</td><td>32.6</td><td>43.1</td><td>22.1</td></tr><tr><td>DIRL (ECCV&#x27;24) [47]</td><td>51.4</td><td>32.3</td><td>66.3</td><td>84.1</td><td>16.8</td><td>56.2</td><td>41.0</td><td>73.8</td><td>126.0</td><td>33.1</td><td>10.3</td><td>13.8</td><td>32.8</td><td>40.9</td><td>19.9</td></tr><tr><td>DIRL + MEDIĆ (Ours)</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td><td>57.5</td><td>41.3</td><td>74.6</td><td>129.2</td><td>33.5</td><td>11.1</td><td>14.6</td><td>33.7</td><td>45.5</td><td>23.0</td></tr></table>

## D.3 Comparison under Single-Change Setting

Table S.1 shows the performance comparison across three datasets, where MEDIC is applied to three baselines (SCORER [49], SMART [48], DIRL [47]). As SMART and DIRL lack CLEVR-DC and CLEVR-Change results respectively, and no method reports type-wise performance, we reproduced all missing results using publicly available oficial source code. Paired t-tests on all datasets confirm that the improvements of MEDIC over baselines are statistically significant (p < 0.05).

Results on the CLEVR-DC Dataset. CLEVR-DC presents a particularly challenging setting, where severe viewpoint shifts act as strong distractors within the scene. As shown in Table S.1 (left), despite these challenges, applying MEDIC consistently yielded superior performance across all metrics. It demonstrates that explicitly separating change types during training is crucial for accurately capturing changes, especially in the presence of distractors.

Results on the CLEVR-Change Dataset. We also evaluate MEDIC on CLEVR-Change, which shares the same synthetic domain as CLEVR-DC but involves fewer distractors and more controlled scene variations. As shown in Table S.1 (center), MEDIC consistently improved performance across all evaluation metrics. Even in less cluttered environments, explicitly modeling change types contributes to better scene understanding and caption generation.

Results on the Spot-the-Dif Dataset. In Table S.1 (right), we evaluate our method on Spot-the-Dif, a low-resolution real-world dataset with subtle illumination changes that make localization challenging. Despite these challenges, MEDIC consistently improves across all evaluation metrics, demonstrating strong

Table S.2: CIDEr performance comparison by change type across datasets. Types: ‘Color’ (Clr), ‘Texture’ (Tex), ‘Add’ (Add), ‘Drop’ (Drp), ‘Move’ (Mv), and ‘No-Change’ (NC).  
Table S.3: Comparison on the Spot-the-Dif under multi-change setting, following [50].
<table><tr><td colspan="7">CLEVR-DC Dataset</td></tr><tr><td>Method</td><td>Clr</td><td>Tex</td><td>Add</td><td>Drp</td><td>Mv</td><td>NC</td></tr><tr><td>DIRL [47]</td><td>108.6</td><td>76.0</td><td>76.5</td><td>81.5</td><td>38.8</td><td>41.6</td></tr><tr><td>DIRL + MEDIC</td><td>118.8</td><td>88.2</td><td>86.1</td><td>88.3</td><td>54.4</td><td>82.3</td></tr><tr><td colspan="7">CLEVR-Change Dataset</td></tr><tr><td>Method</td><td>Clr</td><td>Tex</td><td>Add</td><td>Drp</td><td>Mv</td><td>NC</td></tr><tr><td>DIRL [47]</td><td>149.8</td><td>137.5</td><td>126.9</td><td>137.5</td><td>87.6</td><td>113.8</td></tr><tr><td>DIRL + MEDIC</td><td>151.7</td><td>142.6</td><td>138.6</td><td>141.2</td><td>96.3</td><td>116.2</td></tr><tr><td colspan="7">Spot-the-Diff Dataset</td></tr><tr><td>Method</td><td>Add</td><td>Drp</td><td>Mv</td><td>NC</td><td></td><td></td></tr><tr><td>DIRL [47]</td><td>38.4</td><td>38.6</td><td>41.2</td><td>14.3</td><td></td><td></td></tr><tr><td>DIRL + MEDIC</td><td>47.2</td><td>47.6</td><td>48.4</td><td>17.4</td><td></td><td></td></tr></table>

<table><tr><td>Method</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td></tr><tr><td>SCORER [49]</td><td>5.1</td><td>9.3</td><td>23.0</td><td>20.9</td><td>11.9</td></tr><tr><td>SCORER + MEDIC</td><td>6.1</td><td>9.3</td><td>25.7</td><td>25.9</td><td>16.4</td></tr><tr><td>SMART [48]</td><td>3.7</td><td>8.7</td><td>23.1</td><td>25.3</td><td>15.5</td></tr><tr><td>SMART + MEDIC</td><td>4.5</td><td>8.9</td><td>25.7</td><td>31.6</td><td>17.4</td></tr><tr><td>DIRL [47]</td><td>4.6</td><td>9.5</td><td>23.8</td><td>21.5</td><td>14.8</td></tr><tr><td>DIRL + MEDIC</td><td>6.8</td><td>11.2</td><td>26.7</td><td>31.2</td><td>20.1</td></tr></table>

Table S.4: Expert architecture comparison of MEDIC on CLEVR-DC (DIRL baseline).
<table><tr><td>Architecture</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td></tr><tr><td>MLP (2 layer)</td><td>56.1</td><td>35.1</td><td>70.1</td><td>96.1</td><td>19.3</td></tr><tr><td>Memory Network</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td></tr></table>

robustness and generalization to complex and real-world scenarios.

Type-wise Performance Comparison. We present a detailed comparison across change types in Table S.2, which contrasts the baseline DIRL with MEDICintegrated model. The table includes six change types for CLEVR-DC and CLEVR-Change (‘color’, ‘texture’, ‘add’, ‘drop’, ‘move’, ‘no-change’), and four for Spot-the-Dif (‘add’, ‘drop’, ‘move’, ‘no-change’). MEDIC outperforms DIRL across all datasets and change types. Type-specific experts improve the ability of the model to capture diverse change patterns and achieve more accurate results.

## D.4 Comparison under Multi-Change Setting

To verify the scalability of MEDIC to multi-change scenarios, we further conducted experiments on the Spot-the-Dif dataset [16], the only publicly available dataset applicable to multi-change settings. While most previous works [30, 59] used the Spot-the-Dif dataset in a single-change setting, CARD [50] configured it for a multi-change setting and conducted experiments accordingly, which we followed. Using this protocol, we applied MEDIC to SCORER [49], SMART [48], and DIRL [47], reproducing the latter two for fair comparison. For this setting, we computed the consistency loss for each annotated type and took the mean across them, while using Binary Cross-Entropy for multi-label routing supervision. We generated multi-change type labels using a prompt detailed in the supplementary material. Table S.3 shows that MEDIC clearly outperforms all baselines, indicating that the soft gating mechanism efectively combines knowledge from multiple type experts and generalizes robustly beyond the single-change assumption.

## D.5 Ablation Studies

To better understand the contribution of each component in our method, we conduct ablation studies on the CLEVR-DC, with MEDIC applied to DIRL [47].

Efect of the Memory-based Expert. To examine the impact of expert architecture within the same type-specific framework, we compare two expert designs: a standard MLP-based expert [26] and our memory expert. Table S.4 shows that ours outperforms the MLP-based expert across all metrics, verifying the advantages of input-adaptive retrieval in capturing type-specific cues.

![](images/921e52f0a04509f6c9c6771712b3f92810da70f205c809586e7b0a2c1cc82441.jpg)  
Fig. S.6: t-SNE visualization of learned representations by change type.

Table S.5: Comparison between typeagnostic MoE and our type-specific expert.
<table><tr><td>Method</td><td># Experts</td><td>CIDEr</td></tr><tr><td>Standard MoE</td><td>6</td><td>94.2</td></tr><tr><td>(type-agnostic)</td><td>12</td><td>90.7</td></tr><tr><td></td><td>24</td><td>93.6</td></tr><tr><td>DIRL+MEDIC (type-specific)</td><td>16</td><td>99.8</td></tr></table>

Table S.6: Efect of proposed losses on the CLEVR-DC dataset.
<table><tr><td>Settings</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td></tr><tr><td>DIRL + MEDIC</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td></tr><tr><td> $\bar { \mathrm { \mathbf { w } } } / \bar { \mathrm { o } } \bar { \mathcal { L } } _ { d i s }$ </td><td>57.7</td><td>35.2</td><td>70.6</td><td>98.2</td><td>19.3</td></tr><tr><td> $\mathrm { ~ w ~ } / \mathrm { ~ o ~ } \mathcal { L } _ { c o n }$ </td><td>57.0</td><td>35.0</td><td>70.5</td><td>99.1</td><td>19.7</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ \mathcal { L } _ { d i s } , \mathcal { L } _ { c o n }$ </td><td>54.8</td><td>34.9</td><td>69.9</td><td>96.7</td><td>18.4</td></tr><tr><td> $\mathbf { w } / o \mathcal { L } _ { r o u t e r } , \mathcal { L } _ { d i s } , \mathcal { L } _ { c o n }$ </td><td>53.2</td><td>33.0</td><td>69.9</td><td>87.7</td><td>18.5</td></tr></table>

Table S.7: Ablation on the number of memory slots per expert on CLEVR-DC dataset.
<table><tr><td># Slot</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td><td>|# Params(M)</td><td>Time(ms/sample)</td></tr><tr><td></td><td>51.4</td><td>32.3</td><td>66.3</td><td>84.1</td><td>16.8</td><td>13.90</td><td>26.90</td></tr><tr><td> $\overline { { \cdot } } \overline { { 5 } } 0 ^ { - } \overline { { } }$ </td><td>56.6</td><td>35.2</td><td>70.5</td><td>97.8</td><td>19.0</td><td>24.50</td><td>30.11</td></tr><tr><td>100</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td><td>25.12</td><td>30.16</td></tr><tr><td>200</td><td>57.8</td><td>35.1</td><td>70.3</td><td>98.8</td><td>18.2</td><td>26.35</td><td>30.21</td></tr><tr><td>400</td><td>56.9</td><td>35.0</td><td>70.9</td><td>97.7</td><td>18.5</td><td>28.80</td><td>30.22</td></tr></table>

t-SNE Visualization of Type-aware Experts. To see whether each expert captures type-specific semantics, we visualize the expert outputs using t-SNE. As shown in Fig. S.6, the baseline DIRL shows entangled clusters with unclear boundaries across change types. In contrast, MEDIC generates well-separated clusters, which shows that each expert learns more discriminative and typespecific representations and highlights the efectiveness of type-aware modeling.

Efectiveness of the Type-Specific Expert Design. To validate the efectiveness of our type-specific expert design, we compare MEDIC with a standard type-agnostic MoE. As shown in Table S.5, our design consistently achieves higher performance across diferent expert configurations. This result indicates that a type-specific expert design is more efective than a generic MoE design. We further compare MEDIC with a simple type-aware baseline trained with an auxiliary change-type classification loss in the supplementary material, showing that the gains achieved by MEDIC are not merely due to access to type labels but come from explicit type-specialized expert reasoning.

Efect of the Proposed Losses. We analyze the impact of the expert consistency loss $\mathcal { L } _ { c o n }$ , expert disentangle loss $\mathcal { L } _ { d i s }$ , and routing loss $\mathcal { L } _ { r o u t e r }$ (Table S.6). Removing $\mathcal { L } _ { d i s } \ \mathrm { o r } \ \mathcal { L } _ { c o n }$ reduces performance, while each individually still improves results, indicating their complementary role of expert specialization. Removing $\mathcal { L } _ { r o u t e r }$ forces implicit expert selection and prevents explicit routing training, and leads to further performance drops. Combining all three losses achieves the best performance and demonstrates their synergy in learning type-specialized experts.

![](images/4f6518b94db6d55ca6576a37f66bba45d7d2dc9a88bec09bb10b8a200700c3ad.jpg)

![](images/6648c2673cd5bd752699aa58157af18f5231f97a1f57b4820fdc505a84a40e3b.jpg)

![](images/35bbc6874508d3a0b5e32f4e499393aa5e55aee0be9c66a23a788efeedbc91a9.jpg)  
Fig. S.7: Visualization of two image pairs with the same change type (color), and their slot activation profiles. Green circles indicate commonly activated slots across inputs, capturing type-specific patterns, while blue circles highlight input-specific activations.

Efect of the Memory Slot Size and Overhead. We analyze how the number of memory slots per expert afects the results and computational cost. As shown in Table S.7, increasing the number of slots improves performance up to 100, achieving the best results with only a marginal cost increase. Although performance slightly drops beyond this point, it still surpasses the non-memory baseline.

## D.6 Discussion

Dynamic Slot Activation. To verify that our memory-based experts adapt to input content, we visualize the addressing vectors of two color-type samples in Fig. S.7. Both consistently attend to specific slots (green circles), indicating anchors for type-specific information, while distinct slots (blue circles) reflect input-specific responses. This shows that our memory architecture captures both shared patterns and adaptive behavior within each change type.

Efect of the Hyper-parameters. As shown in Table S.8, MEDIC remains stable across a wide range of hyper-parameters $\lambda _ { 1 } , \lambda _ { 2 } ,$ and $\lambda _ { 3 }$ values, with all configurations achieving strong results and only minor variation.

Generalization to Object- and Global-Level Changes. To see that MEDIC can extend beyond object-level changes, we evaluate it on the Image Editing Request (IER) dataset [44] that contains more diverse change scenarios. While most existing datasets primarily focus on object-level changes, IER includes three object-level types (‘add’, ‘drop’, ‘replace’) and four global-level types (‘background’, ‘illumination’, ‘style’, ‘resize’). As shown in Table S.9, when using DIRL as the baseline, MEDIC achieves consistent improvements across all change types. It demonstrates robustness across both object-level and global changes.

Table S.8: Evaluation results under diferent settings of the hyper-parameters on the CLEVR-DC dataset.  
Table S.9: Results on IER dataset.
<table><tr><td>Method</td><td>B</td><td>M</td><td>R</td><td>C</td></tr><tr><td>DIRL [47]</td><td>10.9</td><td>15.0</td><td>41.0</td><td>34.1</td></tr><tr><td>DIRL+MEDIC</td><td>11.2</td><td>15.6</td><td>42.5</td><td>37.2</td></tr></table>

<table><tr><td>Target</td><td>λ</td><td>B</td><td>M</td><td>R</td><td>C</td><td>S</td></tr><tr><td rowspan="3"> $\lambda _ { 1 }$ </td><td>0.1</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td></tr><tr><td>0.01</td><td>58.1</td><td>35.5</td><td>71.1</td><td>99.2</td><td>19.3</td></tr><tr><td>0.001</td><td>56.5</td><td>35.6</td><td>70.7</td><td>98.6</td><td>18.6</td></tr><tr><td rowspan="3"> $\lambda _ { 2 }$ </td><td>0.1</td><td>58.2</td><td>35.5</td><td>71.1</td><td>99.3</td><td>19.5</td></tr><tr><td>0.01</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td></tr><tr><td>0.001</td><td>57.4</td><td>35.3</td><td>70.8</td><td>99.6</td><td>19.5</td></tr><tr><td rowspan="3"> $\lambda _ { 3 }$ </td><td>0.1</td><td>58.5</td><td>35.6</td><td>71.0</td><td>99.8</td><td>20.0</td></tr><tr><td>0.01</td><td>57.7</td><td>35.6</td><td>70.9</td><td>98.9</td><td>19.9</td></tr><tr><td>0.001</td><td>57.4</td><td>35.3</td><td>70.6</td><td>98.2</td><td>20.0</td></tr></table>

Table S.10: Results on LVLM-based works.
<table><tr><td>Method</td><td>B</td><td>M</td><td>C</td></tr><tr><td>BLIP2IDC [6]</td><td>49.3</td><td>33.0</td><td>88.5</td></tr><tr><td>BLIP2IDC+MEDIC</td><td>57.3</td><td>33.8</td><td>106.4</td></tr></table>

Compatibility with LVLM-Based Methods. To further verify compatibility with recent LVLM-based approaches, we apply MEDIC to BLIP2IDC [6] and evaluate it on CLEVR-DC. As shown in Table S.10, incorporating MEDIC consistently improves performance across all metrics, demonstrating that our method can be efectively integrated with LVLM-based models. Additional comparisons with zero-shot and fine-tuned LVLM-based change captioning methods are provided in the supplementary material.

Label Reliability & Routing Robustness. For datasets without type annotations, we derive change-type labels from ground-truth captions using GPT-4o. To validate this supervision, applying the same labeling protocol to CLEVR-Change and CLEVR-DC, where ground-truth type labels are available, yields 91.2% and 93.8% accuracy, respectively. Moreover, even when 10/20/40% of CLEVR-DC training type labels are randomly corrupted, MEDIC achieves 95.4/94.3/94.0 CIDEr, still outperforming DIRL (84.1), showing robustness to imperfect type supervision. In addition, stage-1/2 routing accuracies show 92%/81%, respectively, and qualitative routing-error analysis shows that our soft expert aggregation can compensate for some top-1 routing mistakes. Detailed prompts, label validation, noisy-label analysis, and routing-error examples are provided in the supplementary material.

Generalization to In-Domain Held-Out Types As MEDIC is built on a modular design with dedicated experts for each change type, it can be extended when additional supervised change types become available. To examine this property, we conduct a leave-one-type-out evaluation on CLEVR-DC in an in-domain held-out setting: one existing change type, such as ‘color’, is excluded during training and evaluated afterward. When ‘color’ is held out, CIDEr on that type drops from 118.8 to 18.4. When this held-out type is later introduced, MEDIC can add a new expert and update the expanded router while freezing the existing experts. With only 0.2M additional parameters (0.82% of the 25.12M total), CIDEr on the held-out type recovers from 18.4 to 113.4 while largely preserving performance on previously learned types. Details are provided in the supplementary material.

![](images/b981073e046b7e88b1fc7792bf13b90b9988d2e467525e3f2ffdc4a22ba415f7.jpg)  
Fig. S.8: Qualitative comparisons across datasets. Each example shows ground-truth (GT), baseline (DIRL), and the proposed method (DIRL+MEDIC).

Limitations. While MEDIC can flexibly extend to unseen datasets by using LLMs to extract new change-type labels, it still relies on the availability of such type annotations. Future work will explore self-supervised approaches that can infer change types without explicit labels.

## D.7 Qualitative Results

Figure S.8 presents qualitative examples on CLEVR-Change, CLEVR-DC, Spotthe-Dif, and Image-Editing-Request. MEDIC generally produces more accurate and type-consistent captions than DIRL by explicitly modeling change types. We additionally provide representative failure cases and their analysis in the supplementary material to highlight the remaining limitations of MEDIC under subtle local changes, crowded multi-change scenes, and ambiguous global edits.

## E Conclusion

We introduce MEDIC, a novel framework that explicitly models change types using memory-based experts specialized for the change type. These experts dynamically retrieve type-relevant visual patterns to enable input-adaptive and type-aware reasoning for accurate change descriptions. Extensive experiments show that explicit change-type modeling leads to consistent performance across diverse and challenging scenarios.

## Acknowledgements

This work was partly supported by IITP-ITRC grant funded by the Korea government (MSIT)(IITP-2026-RS-2023-00258649, 30%) and partly supported by IITP grant funded by the Korea government (MSIT)(IITP-2023-RS-2023- 00266615: Convergence Security Core Talent Training Business Support Program (20%), IITP-2022-II220078: Explainable Logical Reasoning for Medical Knowledge Generation (25%), No. RS-2024-00509257: Global AI Frontier Lab (25%)).

## References

1. Anderson, P., Fernando, B., Johnson, M., Gould, S.: Spice: Semantic propositional image caption evaluation. In: ECCV (2016)

2. Banerjee, S., Lavie, A.: Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In: ACL Workshop (2005)

3. Black, A., Shi, J., Fan, Y., Bui, T., Collomosse, J.: Vixen: Visual text comparison network for image diference captioning. In: AAAI (2024)

4. Cai, W., Liu, Q., Wang, Y.: Spmtrack: Spatio-temporal parameter-eficient finetuning with mixture of experts for scalable visual tracking. In: CVPR (2025)

5. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning (2023)

6. Evennou, G., Chafin, A., Chappelier, V., Kijak, E.: Reframing image diference captioning with blip2idc and synthetic augmentation. In: WACV (2025)

7. Fan, C., Zhang, X., Zhang, S., Wang, W., Zhang, C., Huang, H.: Heterogeneous memory enhanced multimodal attention model for video question answering. In: CVPR (2019)

8. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. JMLR (2022)

9. Han, T., Xie, W., Zisserman, A.: Memory-augmented dense predictive coding for video representation learning. In: ECCV (2020)

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)

11. Hosseinzadeh, M., Wang, Y.: Image change captioning by learning from an auxiliary task. In: CVPR (2021)

12. Hoxha, G., Chouaf, S., Melgani, F., Smara, Y.: Change captioning: A new paradigm for multitemporal remote sensing image analysis. TGRS (2022)

13. Hu, J., Zhong, G., Yuan, J., Pan, W., Wang, X.: Mct-ccdif: Context-aware contrastive difusion model with mediator-bridging cross-modal transformer for image change captioning. TIP (2025)

14. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)

15. Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E.: Adaptive mixtures of local experts. Neural computation (1991)

16. Jhamtani, H., Berg-Kirkpatrick, T.: Learning to describe diferences between pairs of similar images. In: EMNLP (2018)

17. Kim, H., Kim, J., Lee, H., Park, H., Kim, G.: Agnostic change captioning with cycle consistency. In: ICCV (2021)

18. Kim, H., Kim, J., Lee, H., Park, H., Kim, G.: Viewpoint-agnostic change captioning with cycle consistency. In: ICCV (2021)

19. Kim, J.U., Kim, H.I., Ro, Y.M.: Stereoscopic vision recalling memory for monocular 3d object detection. TIP (2023)

20. Kim, J.U., Park, S., Ro, Y.M.: Robust small-scale pedestrian detection with cued recall via memory learning. In: ICCV (2021)

21. Kim, J.U., Park, S., Ro, Y.M.: Towards versatile pedestrian detector with multisensory-matching and multispectral recalling memory. In: AAAI (2022)

22. Kim, J.U., Ro, Y.M.: Enabling visual object detection with object sounds via visual modality recalling memory. TNNLS (2023)

23. Kumar, A., Irsoy, O., Ondruska, P., Iyyer, M., Bradbury, J., Gulrajani, I., Zhong, V., Paulus, R., Socher, R.: Ask me anything: Dynamic memory networks for natural language processing. In: ICML (2016)

24. Lee, S., Kim, H.G., Choi, D.H., Kim, H.I., Ro, Y.M.: Video prediction recalling long-term motion context via memory alignment learning. In: ICCV (2021)

25. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.: Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020)

26. Li, J., Wang, X., Zhu, S., Kuo, C.W., Xu, L., Chen, F., Jain, J., Shi, H., Wen, L.: Cumo: Scaling multimodal llm with co-upcycled mixture-of-experts (2024)

27. Li, J., Pan, K., Ge, Z., Gao, M., Ji, W., Zhang, W., Chua, T.S., Tang, S., Zhang, H., Zhuang, Y.: Fine-tuning multimodal llms to follow zero-shot demonstrative instructions. In: ICLR (2024)

28. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In: ICML (2023)

29. Li, M., Lin, B., Chen, Z., Lin, H., Liang, X., Chang, X.: Dynamic graph enhanced contrastive learning for chest x-ray report generation. In: CVPR (2023)

30. Li, R., Li, L., Zhang, J., Zhao, Q., Wang, H., Yan, C.: Region-aware diference distilling with attribute-guided contrastive regularization for change captioning. In: AAAI (2025)

31. Li, Y., Tu, Y., Li, L., Su, L., Huang, Q.: Change entity-guided heterogeneous representation disentangling for change captioning. In: ACL (2025)

32. Lin, C.Y.: Rouge: A package for automatic evaluation of summaries. In: ACL (2004)

33. Liu, C., Chen, K., Qi, Z., Liu, Z., Zhang, H., Zou, Z., Shi, Z.: Pixel-level change detection pseudo-label learning for remote sensing change captioning. In: IGARSS. IEEE (2024)

34. Liu, Y., Wang, R., Shan, S., Chen, X.: Structure inference net: Object detection using scene-level context and instance-level relationships. In: CVPR (2018)

35. Marchetti, F., Becattini, F., Seidenari, L., Bimbo, A.D.: Mantra: Memory augmented networks for multiple trajectory prediction. In: CVPR (2020)

36. Miller, A., Fisch, A., Dodge, J., Karimi, A.H., Bordes, A., Weston, J.: Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126 (2016)

37. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic evaluation of machine translation. In: ACL (2002)

38. Park, D.H., Darrell, T., Rohrbach, A.: Robust change captioning. In: CVPR (2019)

39. Park, K.R., Park, J., Kim, S.T., Lee, H.J., Kim, J.U.: Leveraging textual compositional reasoning for robust change captioning. In: AAAI (2026)

40. Qiu, Y., Yamamoto, S., Nakashima, K., Suzuki, R., Iwata, K., Kataoka, H., Satoh, Y.: Describing and localizing multiple changes with transformers. In: ICCV (2021)

41. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts (2021)

42. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

43. Shi, X., Yang, X., Gu, J., Joty, S., Cai, J.: Finding it at another side: A viewpointadapted matching encoder for change captioning. In: ECCV (2020)

44. Tan, H., Dernoncourt, F., Lin, Z., Bui, T., Bansal, M.: Expressing visual relationships via language. In: ACL (2019)

45. Tu, Y., Li, L., Su, L., Du, J., Lu, K., Huang, Q.: Adaptive representation disentanglement network for change captioning. TIP (2023)

46. Tu, Y., Li, L., Su, L., Lu, K., Huang, Q.: Neighborhood contrastive transformer for change captioning. TMM (2023)

47. Tu, Y., Li, L., Su, L., Yan, C., Huang, Q.: Distractors-immune representation learning with cross-modal contrastive regularization for change captioning. In: ECCV (2024)

48. Tu, Y., Li, L., Su, L., Zha, Z.J., Huang, Q.: Smart: Syntax-calibrated multi-aspect relation transformer for change captioning. TPAMI (2024)

49. Tu, Y., Li, L., Su, L., Zha, Z.J., Yan, C., Huang, Q.: Self-supervised cross-view representation reconstruction for change captioning. In: ICCV (2023)

50. Tu, Y., Li, L., Su, L., Zha, Z.J., Yan, C., Huang, Q.: Context-aware diference distilling for multi-change captioning. In: ACL (2024)

51. Tu, Y., Li, L., Yan, C., Gao, S., Yu, Z.: R^3net: relation-embedded representation reconstruction network for change captioning. In: EMNLP (2021)

52. Tu, Y., Yao, T., Li, L., Lou, J., Gao, S., Yu, Z., Yan, C.: Semantic relation-aware diference representation learning for change captioning. In: ACL (2021)

53. Vedantam, R., Lawrence Zitnick, C., Parikh, D.: Cider: Consensus-based image description evaluation. In: CVPR (2015)

54. Weng, W., Wu, M., Jiang, H., Kong, W., Kong, X., Xia, F.: Pattern-matching dynamic memory network for dual-mode trafic prediction. T-ITS (2025)

55. Yao, L., Wang, W., Jin, Q.: Image diference captioning with pre-training and contrastive learning. In: AAAI (2022)

56. Yue, S., Tu, Y., Li, L., Gao, S., Yu, Z.: Multi-grained representation aggregating transformer with gating cycle for change captioning. ACM MM (2024)

57. Yue, S., Tu, Y., Li, L., Yang, Y., Gao, S., Yu, Z.: I3n: Intra-and inter-representation interaction network for change captioning. TMM (2023)

58. Zhenxing, M., Xu, D.: Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In: ICLR (2022)

59. Zhong, G., Hu, J., Chen, J., Yuan, J., Pan, W.: Decider: Diference-aware contrastive difusion model with adversarial perturbations for image change captioning. In: AAAI (2025)

60. Zhu, L., Yang, Y.: Inflated episodic memory with region self-attention for long-tailed visual recognition. In: CVPR (2020)

# Diferent Changes Require Diferent Reasoning: Change-Type-Specialized Experts for Robust Change Captioning – Supplementary Material –

Jiyoung Park<sup>\*</sup> , InJae Oh<sup>\*</sup> , and Jung Uk Kim<sup>†</sup>

Kyung Hee University, Yong-in, South Korea {jy0117, seanoh, ju.kim}@khu.ac.kr

## Overview

In this supplementary material, we present additional analyses and results that complement the main paper. Specifically, we provide:

## Table of Contents

A. Explanation of Expert Disentangle Loss . . 20   
B. Prompting and Validation of LLM-Generated Type Labels . . . . . . 21   
C. Robustness to Noisy Type Labels . . . 22   
D. Analysis of the Relationship Between Router Predictions and Gener  
ated Captions . . . . . . . . 23   
E. Type-Aware Baseline Analysis . . .24   
F. Leave-One-Type-Out Evaluation for In-Domain Held-Out Change   
Types . . . . .24   
G. Taxonomy-Mismatch Stress Test . . 25   
H. Impact of Routing Accuracy on Caption Quality . . 26   
I. Ablation of Routing Strategy . . . . 26   
J. Additional Evaluation Beyond Caption Similarity Metrics . . . . 27   
K. Comparison with LVLM-Based Methods . 28   
L. Qualitative Results . . 28   
M. Failure Cases . . 29   
N. Dynamic Slot Activation Analysis . . 29

This document ofers in-depth qualitative and quantitative analyses to support the efectiveness of our method. The included figures, ablations, and visualizations provide further understanding of how change-type awareness and memory-based experts contribute to improved change captioning.

![](images/94e0cc9ee1ad307a18cac7beb73b2feb320ea057584e0817948ae0fee552ea82.jpg)  
Fig. S.1: Conceptual illustration of the expert disentangle loss. Within the change group (top-right), diferent type-specific experts are encouraged to stay apart from one another. Across groups (bottom-right), each change expert is pushed away from the no-change expert, enforcing pairwise inter-group separation between all change/no-change expert pairs to ensure semantic separation.

## A Explanation of Expert Disentangle Loss

As introduced in the “Expert Disentangle Loss” section of the main paper (Section 3.5), this section provides a detailed explanation of how the loss operates, along with a conceptual visualization for better understanding. The expert disentangle loss aims to organize the representation space by enforcing separation among change-type experts and distinguishing them from the no-change expert. It is composed of two components: (1) an intra-group loss that enforces disentanglement among change-type experts, and (2) an inter-group loss that separates all change-type experts from the no-change expert.

## Feature Construction.

– As illustrated in Figure S.1, for each input image pair, visual diference features from the before and after images, denoted as $\hat { \ b { f } } ^ { b }$ and ${ \hat { f } } ^ { a }$ , respectively. These are concatenated to form the paired feature $F ^ { p } = [ \hat { f } ^ { b } ; \hat { f } ^ { a } ]$ , which serves as the input to all experts.

– The paired feature is routed to the corresponding expert based on the groundtruth change type. Each expert produces a type-specific output feature $F ^ { m _ { t } }$ 2 which captures visual evidence relevant to its assigned change type (e.g., color, texture).

– Global average pooling (GAP) is applied to each output of the expert to obtain a summary representation $F _ { a v g } ^ { m _ { t } } \in \mathbb { R } ^ { 2 d }$ , which is used to compute similarity among expert outputs.

Loss Mechanism.

– Intra-group Loss (Figure S.1 (top-right)): To ensure distinct representations among change-type experts, the intra-group loss penalizes high similarity between diferent change-type experts. For every pair of change-type experts $( t , t ^ { \prime } )$ , the cosine similarity $d _ { t , t ^ { \prime } } = \cos ( F _ { a v g } ^ { m _ { t } } , F _ { a v g } ^ { m _ { t ^ { \prime } } } )$ is computed. The loss term is defined as:

$$
\mathcal { L } _ { \mathrm { i n t r a } } = \frac { 2 } { | T | ( | T | - 1 ) } \sum _ { \stackrel { t , t ^ { \prime } \in T } { t < t ^ { \prime } } } \left[ \operatorname* { m a x } \left( 0 , d _ { t , t ^ { \prime } } - \delta _ { \mathrm { i n t r a } } \right) \right] ^ { 2 }
$$

where $\tau$ is the set of change types, and $\delta _ { \mathrm { i n t r a } }$ is a margin that softly enforces a minimum dissimilarity among experts.

– Inter-group Loss (Figure S.1 (bottom-right)): The inter-group loss separates the change-type experts from the no-change expert. For each change-type expert $t \in \tau$ , the similarity with the no-change expert $T$ is computed as $d _ { t , T } = \cos ( F _ { a v g } ^ { m _ { t } } , F _ { a v g } ^ { m _ { T } } )$ , and the loss is defined as:

$$
\mathcal { L } _ { \mathrm { i n t e r } } = \frac { 1 } { | T | } \sum _ { t \in \mathcal { T } } \left[ \operatorname* { m a x } \left( 0 , d _ { t , T } - \delta _ { \mathrm { i n t e r } } \right) \right] ^ { 2 }
$$

where $\delta _ { \mathrm { { i n t e r } } }$ is a margin that enforces separation between changed and unchanged representations.

– Total Disentangle Loss: The overall loss is the sum of the two terms:

$$
{ \mathcal { L } } _ { \mathrm { d i s } } = { \mathcal { L } } _ { \mathrm { i n t r a } } + { \mathcal { L } } _ { \mathrm { i n t e r } }
$$

Interpretation. Figure S.1 shows a conceptual illustration of the expert disentangle loss. Type-specific experts are encouraged to diverge within the change group while being separated from the no-change expert, resulting in a well-structured and disentangled representation space.

## B Prompting and Validation of LLM-Generated Type Labels

In this section, we present our approach for generating change-type labels for the Spot-the-Dif [16] and Image-Editing-Request [44] datasets using $\mathrm { G P T - 4 o }$ [14], including the detailed prompting strategy and an analysis of label quality, as mentioned in the Dataset and Discussion section of the main paper. Since both datasets lack ground-truth type labels, we address this limitation by generating them through large language model prompting.

We utilize the ground-truth captions provided in the dataset, which describe the semantic diferences between the two images. Since these captions inherently contain information about what has changed, they serve as a strong signal for inferring change types. We prompt GPT-4o with the caption and ask it to predict the corresponding change type. The full prompts for the single-change and multichange scenarios are shown in Table S.8, Table S.9 and Table S.10, respectively. We adopt in-context learning to guide the predictions of model.

Table S.1: Validation of type labels extracted using the Spotthe-Dif approach on CLEVR-Change and CLEVR-DC, both with ground-truth types.  
Table S.2: Results on noisy type labels. We randomly corrupt a portion of training type labels on CLEVR-DC while keeping image pairs and caption supervision unchanged.
<table><tr><td>Dataset</td><td>Accuracy(%)</td></tr><tr><td>CLEVR-Change</td><td>91.2</td></tr><tr><td>CLEVR-DC</td><td>93.8</td></tr></table>

<table><tr><td>Method</td><td>Type Label Noise</td><td>CIDEr</td></tr><tr><td rowspan="4">DIRL+MEDIC</td><td>0%</td><td>99.8</td></tr><tr><td>10%</td><td>95.4 (-4.4)</td></tr><tr><td>20%</td><td>94.3 (-5.5)</td></tr><tr><td>40%</td><td>94.0 (-5.8)</td></tr><tr><td>DIRL</td><td>N/A</td><td>84.1</td></tr></table>

In single-change scenarios, to evaluate the quality of the LLM-generated type labels, we apply the same prompting strategy to two benchmark datasets—CLEVR Change [38] and CLEVR-DC [18]—which contain ground-truth type labels. By comparing the LLM-predicted labels to the ground-truth type labels, we assess the accuracy of LLM. As shown in Table S.1, the accuracy is high on both datasets, demonstrating that our method produces high-quality type labels even in scenarios where ground-truth annotations are unavailable, such as real-world datasets. In the multi-change setting, we do not conduct a similar evaluation because Spot-the-Dif [16] is the only publicly available multi-change dataset.

While the above validation confirms that the LLM-generated labels are generally reliable, some label noise may still exist in datasets without groundtruth type annotations. We therefore further analyze the robustness of MEDIC to corrupted type supervision in the following section.

## C Robustness to Noisy Type Labels

LLM-generated type labels on Spot-the-Dif and Image-Editing-Request may contain potential noise, since these datasets do not provide ground-truth changetype annotations. To examine whether MEDIC is overly dependent on perfectly clean type supervision, we conduct a noisy-label stress test on CLEVR-DC, where ground-truth type labels are available.

Specifically, we randomly corrupt 10%, 20%, and 40% of the training type labels by replacing each selected label with an incorrect label from the changetype taxonomy. The image pairs and caption supervision are kept unchanged, so only the type-level supervision used for routing and expert specialization is corrupted.

As shown in Table S.2, MEDIC achieves 95.4, 94.3, and 94.0 CIDEr under 10%, 20%, and 40% type-label noise, respectively. Although performance decreases as the noise ratio increases, MEDIC consistently remains above the DIRL baseline, which achieves 84.1 CIDEr. Even under 40% noisy type labels, MEDIC still preserves a substantial improvement over DIRL.

![](images/b9ba5673ad3ee1e6a2eafafc1d3305d40d3fcae5a5fcd18af06df70c243204d1.jpg)  
Fig. S.2: Visualization of MEDIC under imperfect routing.

These results indicate that MEDIC does not rely solely on discrete type labels. Instead, its expert routing and caption generation are jointly guided by image-pair features, caption supervision, and soft expert aggregation.

## D Analysis of the Relationship Between Router Predictions and Generated Captions

To better understand behavior of MEDIC under imperfect routing, we conduct a qualitative analysis on CLEVR-Change, CLEVR-DC, and Spot-the-Dif (Figure S.2 (a), Figure S.4), focusing on challenging cases where the router’s top-1 predicted change type difers from the ground-truth (GT). Despite this mismatch, the generated captions often correctly describe both the changed object and the modification itself, indicating that a non-GT top-1 prediction does not necessarily lead to captioning failure.

This robustness stems from the soft expert aggregation mechanism: even when the GT type is not ranked first, it typically receives the second-largest probability with a value close to the top-1 score. As MEDIC aggregates all typespecific experts through weighted summation, the GT expert still contributes substantially, allowing the model to integrate information from multiple plausible change types. The bar plots in Figure S.4 illustrate this behavior, where the GT probability remains non-negligible even when it is not the highest.

Such routing ambiguities arise for diferent reasons across datasets. In CLEVR-Change and CLEVR-DC, subtle modifications (e.g., small objects, partial occlusions, or minor displacements) can blur the distinction between visually related types, causing probability mass to spread across competing categories. In Spotthe-Dif, a single image pair may contain multiple change types, while the router predicts only one label, making the top-1 prediction reflect a dominant pattern rather than all underlying changes. Nevertheless, soft aggregation enables MEDIC to maintain reliable caption generation in many of these cases.

Table S.3: Comparison with a simple type-aware baseline on CLEVR-DC using DIRL as the backbone. “Aux. Type Loss” denotes adding an auxiliary change-type classification loss to DIRL without using type-specific experts.
<table><tr><td>Method</td><td>Type Labels</td><td>Type-Specific Experts</td><td>CIDEr</td></tr><tr><td>DIRL</td><td>x</td><td>x</td><td>84.1</td></tr><tr><td>DIRL + Aux. Type Loss</td><td>√</td><td>x</td><td>87.2</td></tr><tr><td> $\mathrm { D I R L + M E D I C }$ </td><td>√</td><td>√</td><td>99.8</td></tr></table>

In addition to the mitigation cases discussed above, Figure S.2 (b), (c) also present representative failure cases. In Figure S.2 (b), the router predicts the correct type but the caption is inaccurate, showing that correct routing alone does not guarantee accurate description. In Figure S.2 (c), both routing and captioning fail, including no-change confusion. These failures typically occur under extreme viewpoint variations or substantial scene reconfiguration, where unreli able object alignment between the before and after images leads to ambiguous correspondences that adversely afect the final caption.

## E Type-Aware Baseline Analysis

To verify that the improvement of MEDIC does not simply come from access to change-type labels, we compare MEDIC with a simpler type-aware baseline. Specifically, we train DIRL on CLEVR-DC with an auxiliary change-type classification loss while keeping the captioning architecture type-agnostic.

As shown in Table S.3, adding type supervision alone improves DIRL from 84.1 to 87.2 CIDEr. However, this gain remains much smaller than that of DIRL+MEDIC, which reaches 99.8 CIDEr. This result confirms that the performance improvement of MEDIC is not merely due to using change-type labels during training. Instead, the explicit type-specialized expert architecture, together with routing-based expert selection, provides a stronger mechanism for learning change-type-specific reasoning.

## F Leave-One-Type-Out Evaluation for In-Domain Held-Out Change Types

To examine how MEDIC handles in-domain change types that are not observed during training and how it can later incorporate them with additional supervision, we conduct a leave-one-type-out evaluation on CLEVR-DC.

Specifically, CLEVR-DC provides a predefined change-type taxonomy, and we exclude one existing type, such as ‘Color’ or ‘Add’, during training. At test time, samples from the excluded type are evaluated without having trained a dedicated expert for that type.

As shown in Table S.4, when ‘Color’ is excluded during training, CIDEr on the held-out type drops from 118.8 to 18.4. A similar drop is observed when ‘Add’ is held out, where the score decreases from 86.1 to 13.1. This behavior is expected because no dedicated expert has been optimized for the excluded type. Nevertheless, MEDIC does not completely fail: it can still generate captions by softly routing held-out samples to the most relevant previously learned experts.

Table S.4: Leave-one-type-out results on CLEVR-DC in an in-domain held-out type setting. FT denotes incremental fine-tuning after adding a new expert for the held-out type while preserving the existing experts.
<table><tr><td>Method</td><td># Train Types</td><td>Unseen</td><td>Total</td><td>Clr</td><td>Tex</td><td>Add</td><td>Drp</td><td>Mv</td><td>NC</td></tr><tr><td>DIRL</td><td>6</td><td></td><td>84.1</td><td>108.6</td><td>76.0</td><td>76.5</td><td>81.5</td><td>38.8</td><td>41.6</td></tr><tr><td>DIRL+MEDIC</td><td>6</td><td></td><td>99.8</td><td>118.8</td><td>88.2</td><td>86.1</td><td>88.3</td><td>54.4</td><td>82.3</td></tr><tr><td>(1) DIRL+MEDIC(B)</td><td>5</td><td>Color</td><td>76.1</td><td>18.4</td><td>85.7</td><td>87.0</td><td>84.7</td><td>39.6</td><td>56.0</td></tr><tr><td>(2) (B)+FT Color</td><td>6</td><td></td><td>92.0</td><td>113.4</td><td>85.1</td><td>80.5</td><td>85.1</td><td>48.0</td><td>70.1</td></tr><tr><td>(1) DIRL+MEDIC(B)</td><td>5</td><td>Add</td><td>74.5</td><td>117.5</td><td>85.9</td><td>13.1</td><td>83.3</td><td>39.4</td><td>54.3</td></tr><tr><td>(2) (B)+FT Add</td><td>6</td><td></td><td>93.7</td><td>114.9</td><td>85.6</td><td>86.6</td><td>82.8</td><td>53.2</td><td>63.6</td></tr></table>

We then simulate an incremental update scenario where supervision for the held-out type becomes available after the initial training stage. In this setting, we freeze the existing experts, add a newly initialized expert for the held-out type, and update the expanded router together with the new expert. This requires only 0.2M additional parameters, corresponding to 0.82% of the 25.12M total parameters. After this lightweight update, performance on the held-out type substantially recovers: ‘Color’ improves from 18.4 to 113.4 and ‘Add’ improves from 13.1 to 86.6, approaching the fully supervised setting trained with all six types. (118.8 and 86.1)

These results support the modular extensibility of MEDIC. MEDIC substantially recovers performance on the held-out type while avoiding catastrophic degradation on previously learned types. This demonstrates that newly supervised change categories can be incorporated through lightweight expert-level fine-tuning rather than full model retraining. Overall, our leave-one-type-out evaluation shows two complementary strengths: (1) partial robustness to in-domain held-out types through soft expert routing, and (2) eficient extensibility via modular expert addition.

## G Taxonomy-Mismatch Stress Test

To evaluate robustness under taxonomy mismatch, we conduct a stress test in which the predefined change-type taxonomy is intentionally perturbed. Specifically, the Color type is removed and its samples are reassigned to other categories using an LLM, introducing ambiguous and potentially inconsistent type labels during training. Importantly, this setting does not remove color-change instances themselves. The underlying image pairs and their caption ground-truths still contain color modifications, while only the type labels provided to the model are mismatched.

As shown in Table S.5, this label inconsistency reduces routing accuracy because the router is trained under ambiguous supervision. However, captioning performance degrades only moderately, and the model continues to describe many color changes correctly. This outcome is reasonable since MEDIC still observes color-change examples during training; although their type labels are reassigned, the model can learn from the visual evidence in the input images rather than relying solely on discrete type annotations.

Table S.5: Taxonomy-mismatch stress test.
<table><tr><td>Method</td><td>Reclassified</td><td>Total</td><td>Color</td><td>Routing Acc.</td></tr><tr><td>DIRL</td><td></td><td>84.1</td><td>108.6</td><td></td></tr><tr><td>DIRL+MEDIC</td><td></td><td>99.8</td><td>118.8</td><td>75.6</td></tr><tr><td></td><td>Color</td><td>87.5</td><td>115.7</td><td>60.4</td></tr></table>

![](images/27f013ae0a89c750a7f1c902707b379b27229f3d84b31feceb25960af53c934e.jpg)  
Fig. S.3: Efect of Routing Accuracy.

Overall, these results indicate that MEDIC remains functional under taxonomylevel ambiguity. Even when type supervision is imperfect, the expert modules retrieve and integrate visual cues directly from the images, enabling stable change captioning despite mismatched change-type definitions.

## H Impact of Routing Accuracy on Caption Quality

To investigate how routing accuracy afects caption quality, we simulate diferent routing accuracies by adding Gaussian noise $\sigma \in \{ 0 , 0 . 5 , 1 , 5 , 1 0 , 2 0 \}$ . As shown in Figure S.3, when routing noise is small $( \sigma \leq 1 )$ , caption performance remains nearly unchanged, indicating that the model is robust to minor routing perturbations. However, as the noise level increases (indicating reduced routing accuracy), the CIDEr score decreases steadily, dropping significantly when the routing decisions become highly unreliable. These results reveal a clear relationship between routing reliability and caption generation quality. Accurate routing enables the model to leverage the appropriate type-specific experts, which in turn leads to more precise change descriptions.

## I Ablation of Routing Strategy

To validate the efectiveness of our routing design, we compare the proposed two-stage routing strategy with a simplified single-stage alternative on CLEVR-DC. In the proposed two-stage routing, the model first determines whether a change exists using a change router (Stage-1). If a change is detected, a type router (Stage-2) then predicts the specific change type. This hierarchical structure separates change detection from fine-grained type classification. In contrast, the single-stage variant performs routing in a single step over all six categories, including the no-change class. That is, one router directly predicts among all change types and the no-change label without an explicit decomposition. Under this setting, the single-stage routing achieves a CIDEr score of 90.4, whereas the proposed two-stage routing reaches 99.8. This substantial performance gap confirms that explicitly decomposing change detection and type classification leads to more efective routing and improved caption quality.

Table S.6: Additional change-aware evaluation on CLEVR-DC using DIRL as the backbone. Chg/NC Acc. measures whether the generated caption correctly distinguishes changed and unchanged cases. Type Recog. measures whether the described change type matches the ground-truth type. Content Align. is measured by CLIPScore.
<table><tr><td>Method</td><td>Chg/NC Acc.</td><td>Type Recog.</td><td>Content Align.</td></tr><tr><td>DIRL</td><td>83.5</td><td>77.8</td><td>26.5</td></tr><tr><td>DIRL+MEDIC</td><td>92.4</td><td>84.7</td><td>27.2</td></tr></table>

## J Additional Evaluation Beyond Caption Similarity Metrics

Standard captioning metrics such as BLEU-4 [37], METEOR [2], ROUGE-L [32], CIDEr [53], and SPICE [1] evaluate similarity between generated captions and reference captions, but they do not directly isolate whether a model correctly recognizes the existence of change, the type of change, or the detailed visual content. To complement these metrics, we conduct additional change-aware evaluations on CLEVR-DC using DIRL as the backbone.

We consider three complementary measures. First, change/no-change accuracy (Chg/NC Acc.) checks whether the generated caption correctly distinguishes changed cases from no-change cases. Second, change-type recognition (Type Recog.) measures whether the generated caption describes the same change type as the ground-truth label. For these two measures, we use a rule-based parser based on no-change expressions and type-specific keywords that commonly appear in CLEVR-DC captions. Third, content alignment (Content Align.) is measured using CLIPScore, which evaluates image-caption alignment and provides a complementary signal for detailed content correctness.

As shown in Table S.6, DIRL+MEDIC outperforms DIRL across all three evaluations. In particular, MEDIC improves change/no-change accuracy from 83.5 to 92.4 and change-type recognition from 77.8 to 84.7, indicating that typespecialized experts help the model better identify whether a change occurred and what type of change occurred. MEDIC also improves CLIPScore from 26.5 to 27.2, suggesting improved alignment between the generated caption and visual content. These results support that MEDIC improves not only caption similarity metrics but also change-aware recognition behavior.

Table S.7: Comparison with LVLM-based methods on Spot-the-Dif. Zero-shot LVLM results are reported from prior work [27]. BLIP2IDC is a fine-tuned LVLM-based model.
<table><tr><td>Method</td><td>Setting</td><td>METEOR</td><td>CIDEr</td></tr><tr><td>BLIP-2 [28]</td><td>Zero-shot</td><td></td><td>17.5</td></tr><tr><td>InstructBLIP [5]</td><td>Zero-shot</td><td></td><td>19.7</td></tr><tr><td>VPG-C-LLaMA2-7B [27]</td><td>Zero-shot</td><td></td><td>21.0</td></tr><tr><td>VPG-C-Vicuna-7B [27]</td><td>Zero-shot</td><td></td><td>20.0</td></tr><tr><td>VPG-C-Vicuna-13B [27]</td><td>Zero-shot</td><td></td><td>21.6</td></tr><tr><td>BLIP2IDC [6]</td><td>Fine-tuned LVLM</td><td>13.5</td><td>51.4</td></tr><tr><td>DIRL+MEDIC</td><td>Task-trained lightweight model</td><td>14.6</td><td>45.5</td></tr></table>

## K Comparison with LVLM-Based Methods

Recent LVLM-based approaches have shown strong potential for image diference captioning by leveraging large-scale vision-language pretraining. To better position MEDIC with respect to these methods, we compare it with both zero-shot LVLMs and a fine-tuned LVLM-based change captioning model.

Table S.7 summarizes results on Spot-the-Dif. Zero-shot LVLMs such as BLIP-2 [28], InstructBLIP [5], and VPG-C [27] variants achieve CIDEr scores between 17.5 and 21.6. These results indicate that directly applying generalpurpose LVLMs in a zero-shot manner remains challenging for change captioning, where the model must precisely compare two similar images and describe only the visual diference.

When task-specific fine-tuning is introduced, the LVLM-based BLIP2IDC [6] achieves a much higher CIDEr score of 51.4, showing the importance of adapting LVLMs to the change captioning task. Compared with this fine-tuned LVLM-based model, DIRL+MEDIC obtains a lower CIDEr score but a higher METEOR score. This suggests that MEDIC remains competitive with fine-tuned LVLM-based approaches while using a lightweight task-trained backbone and a type-aware reasoning module, without relying on large-scale LVLM pretraining.

## L Qualitative Results

As mentioned in the main paper (Section 4.7) “Qualitative Results”, We provide additional visualizations for each dataset to further illustrate the efects of change-type awareness. We present qualitative examples in Figure S.5, S.6, S.7 to demonstrate how the proposed MEDIC, through type-aware modeling, more accurately identifies and localizes changes, leading to more detailed and precise descriptions of the changed regions. Across all cases, MEDIC enhances ability of the model to generate type-aware and semantically accurate descriptions, particularly in challenging scenarios where previous methods often fail to capture the correct change type.

## M Failure Cases

While MEDIC generally improves change captioning performance across datasets, it still exhibits several characteristic failure modes. Figure S.8 presents representative failure cases from CLEVR-Change, Spot-the-Dif, and Image-Editing-Request.

In CLEVR-Change, MEDIC may fail when the visual change is subtle and localized, such as a small object movement under limited appearance variation. In such cases, the changed object can be dificult to align reliably between the before and after images, leading the model to incorrectly predict that the scene remains unchanged.

In Spot-the-Dif, failure often arises in crowded scenes containing multiple entities and potentially multiple simultaneous changes. Under these conditions, the model may generate a partially correct but incomplete caption that describes salient objects in the after image without accurately capturing the underlying change relation, such as movement or disappearance.

In Image-Editing-Request, failures are frequently associated with global editing operations whose semantic boundaries are ambiguous, such as contrast adjustment, color enhancement, or style-related appearance changes. In these cases, MEDIC may produce an edit description that is semantically related to the target instruction but does not precisely match the intended operation.

Overall, these examples suggest that the main remaining challenges for MEDIC include subtle local changes, complex multi-object scenes, and ambiguous global edits. Addressing these limitations more efectively remains an important direction for future work.

## N Dynamic Slot Activation Analysis

To complement the analysis presented in the main paper (Section 4.6) “Dynamic Slot Activation”, we provide additional visualizations across various change types to further validate the dynamic behavior of our memory-based experts. Figures S.9 and S.10 show memory activation patterns for two samples per change type from the CLEVR-DC dataset [18].

Although the samples belong to the same change type, we observe notable diferences in the activated memory slots across examples. In particular, we observe two distinct types of slot activations: type-specific slots (green circles) and input-specific slots (blue circles). Type-specific slots are consistently activated across diferent input samples that share the same change type, indicating that they capture generalizable patterns associated with that type. In contrast, input-specific slots are selectively activated based on the unique visual context of each input. These input-dependent activations allow the expert to encode fine-grained, input-specific details.

As a result, these findings highlight that the memory-based experts do not rely on static mappings. Instead, they dynamically retrieve relevant information by selecting diferent memory slots in an input-conditioned manner, demonstrating both flexibility and semantic sensitivity in slot utilization. It also provides further evidence that memory-based retrieval enables our experts to dynamically specialize without hard-coded priors or slot assignments.

![](images/11517a60d272831d73addeb355d9a8599b3c5c514f1e95c68eb4c9158ddf25b8.jpg)  
(a) CLEVR-Change dataset.

![](images/dd948e7622b4024a5bd970a3c529f35bd6bff6c6edbf91d19b218b2dc5e9cf15.jpg)  
(b) CLEVR-DC dataset.

![](images/a1805dddf192b9218382a02645101b9116aa99ab5a73db970ceaee4bee2a4186.jpg)  
(c) Spot-the-Dif dataset.  
Fig. S.4: Qualitative analysis of cases where the router’s top-1 predicted change type difers from the ground-truth type. Across CLEVR-Change, CLEVR-DC, and Spotthe-Dif, our model (DIRL+MEDIC) still produces accurate change captions in these challenging cases. This illustrates how the weighted expert aggregation efectively mitigates the impact of non-GT top-1 predictions by integrating complementary cues from all change-type experts.

<table><tr><td>Prompt: You are given a JSON object where each key is an image ID, and each value is</td></tr><tr><td>associated with that ID. Your task is to classify each image into exactly one of the following six change</td></tr><tr><td>types: &quot;color&quot;, &quot;texture&quot;, &quot;add&quot;, &quot;drop&quot;, &quot;move&quot;, &quot;no change&quot;. Return ONLY the type name (e.g., &quot;move&quot;, &quot;add&quot;, etc.) for the given set of</td></tr><tr><td>captions.</td></tr><tr><td>[Classification Rules &amp; Examples] (color) - Only color changed.</td></tr><tr><td>- Examples: &quot;the other large red object that is the same shape as the brown metal object became blue&quot;, &quot;the other ball that is the same size as the cylinder became yellow&quot;, &quot;the other small purple object the same shape as the tiny shiny object became green&quot;</td></tr><tr><td>(texture) - Material changes (rubber, shiny, matte, metallic). - Examples: &quot;the other large cylinder the same color as the large matte cylinder turned rubber&quot;, &quot;the other big object the same shape as the tiny yellow rubber</td></tr><tr><td>thing became shiny&quot;, &quot;the other red sphere the same material as the tiny cyan object became matte&quot; (add) - Something new appears in the after image.</td></tr><tr><td>- Examples: &quot;the object behind the green shiny cylinder and right of the large block in the after image has been added&quot;, &quot;the large purple thing has been added&quot;, &quot;the other gray object that is the same size as the cyan object has been newly placed&quot;</td></tr><tr><td>(drop) - Something from the before image is missing in the after image.</td></tr><tr><td></td></tr><tr><td>- Examples: &quot;the red cylinder is missing&quot;,&quot;the other sphere that is the same</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>size as the blue cube is missing&quot;, &quot;the brown shiny ball is no longer there&quot;</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>(move)</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- Object&#x27;s position changed.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- Examples: &quot;the small brown object is in a different location&quot;, &quot;the other shiny</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>ball that is the same size as the blue object is in a different location&quot;, &quot;the other</td></tr></table>

Table S.8: Prompt used for LLM-based generation of change-type labels from ground truth captions for single-change scenarios in the Spot-the-Dif dataset.

Now, given the following GT Captions for a single image, respond with one and only one of the following types:

Table S.9: Prompt used for LLM-based generation of change-type labels from ground truth captions for single-change scenarios in the Image-Editing-Request dataset.

## Prompt:

You are given a JSON object where each key is an image ID, and each value is a list of ground truth (GT) captions describing the change between a pair of images.

Your task is to read the caption and determine the main type of visual change described in it.

Focus on understanding the semantic change operation expressed in the caption.

Examples of possible change operations include introducing a new element, removing something, replacing one object with another, modifying the background scene, adjusting lighting properties, applying visual efects, or changing the spatial framing of the image.

## Instructions:

1. Carefully read the caption.

2. Infer the primary visual editing operation it describes.

3. Categorize the change at a coarse semantic level, focusing only on the main type of change.

4. Do not create overly specific or fine-grained categories. The goal is to capture the general type of change, not detailed subtypes.

5. If multiple edits are mentioned, choose the most dominant or central change.

6. Output only the change category label.

Do not output explanations.

Table S.10: Prompt used for LLM-based generation of change-type labels from groundtruth captions in multi-change scenarios.  
Prompt:   
You are given a JSON object where each key is an image ID, and each value is   
a list of ground truth (GT) captions describing changes between a pair of images.   
Your task is to classify each image into one or more of the following change   
types: “color", “texture", “add", “drop", “move", “no change".   
Return ALL applicable types for the given set of captions.   
Output MUST be a JSON dictionary with:   
- key: image ID   
- value: a list of ALL correct change types (multi-label allowed)   
- order of types does not matter   
- do NOT generate any types that are not supported   
- do NOT output any explanations   
- do NOT invent diferences not mentioned   
If a caption states multiple diferent changes, include ALL relevant types.   
[Examples]   
Captions for “000235.png": [“the person in the blue shirt is no longer present.   
the dark grey SUV is gone in the photo on the right"]   
Output: {   
“000235.png": [“drop"]   
}   
Captions for “000103.png": [ “the blue car has moved. a person appeared in the   
second picture" ]   
Output: {   
“000103.png": [“move", “add"]   
}   
Captions for “000214.png": [ “people moved from corner crossing street. white   
car on left side. trafic light red instead of green" ]   
Output: {   
“000214.png": [“move", “add", “color"]   
}   
Now, based on the following input JSON, return only the final JSON result:

![](images/03998c14bce5677cec3f564f1033d53ff5904c77874b6ecf35cf95eadfbf3c3c.jpg)  
(a) CLEVR-Change dataset.

![](images/ff104a2c5ff8ee82f30cb6f662fedc19963a586577850b67909e7f67d5b89c04.jpg)  
(b) CLEVR-DC dataset.  
Fig. S.5: Qualitative examples comparing GT, DIRL, and DIRL+MEDIC (Ours) across datasets (Part 1). Our model generates more accurate and typeconsistent captions by explicitly modeling change types.

![](images/add6326dd0ecd55b90caf752938d9a55a356b72b915aa51d50d6acf313e933d4.jpg)  
(c) Spot-the-Dif dataset.

Fig. S.6: Qualitative examples comparing GT, DIRL, and DIRL+MEDIC (Ours) across datasets (Part 2). Our model generates more accurate and type consistent captions by explicitly modeling change types.  
![](images/ea481fe4d6756224fb442d95d557e87a85145226706c184e8a785adc87a05e01.jpg)  
(d) Image Editing Request dataset.  
Fig. S.7: Qualitative examples comparing GT, DIRL, and DIRL+MEDIC (Ours) across datasets (Part 3). Our model generates more accurate and typeconsistent captions by explicitly modeling change types.

![](images/e78fe8863441ad03e65db4e4b1bb4e203f5cd71224fbb1f534c77e29b526d4d3.jpg)  
Fig. S.8: Representative failure cases of MEDIC on CLEVR-Change, Spot-the-Dif, and Image-Editing-Request. From left to right, the examples illustrate failures under subtle local changes, crowded multi-object scenes, and ambiguous global edit instructions.

(a) Add Change (Example 1)  
![](images/707feffc3e63d0fd9f4edeb3bc5aa07492bff005c3e4481cb930ecf59f1a74a0.jpg)

![](images/1f41708b3201e009c04b6791db8a459e7aacc69f4cf194b7f7e231ede492ad2a.jpg)

![](images/3e701da4897b132e8090ac14da0e342eee6f34e0de19546a83cf55f681f3a3af.jpg)

![](images/8fe9a19200cba74a54dbc11d7627a7cb57fea0215299d72cebf6cf749ca60fe8.jpg)

![](images/8fa50d7dd2db0a99419f0278b2cb8447e759d221492216f497ab339f9602870f.jpg)

![](images/33b603c6d61b22ae6b5606ce12a572124b45a3e1f8a7b3c3fba161df81f5d7bf.jpg)

![](images/2de48be434605be153d8f8917702fd572d4ec0417ae20a0cc65377845ec107fc.jpg)

![](images/d01c75eee4c4d89964ccefbe7b0c586b47f072b394c52fea911c56ef53369687.jpg)

![](images/563900e1db5bae880ffe04703fa48cfed05476b54629f2007bc48745991b21ec.jpg)

![](images/bea112c094a71bcec79f74f4782fd49287a1dc25a7415c2251bfb400a26c0142.jpg)

![](images/1b19b82359138b525d9519b2a8c3f68474c7d4430fd57046d6294823ef74ea2d.jpg)

![](images/46c315559096f87afaeae6556c719144fb34bf09521522a7c47cffcbca2b3e2e.jpg)  
Fig. S.9: Dynamic slot activation examples on CLEVR-DC dataset using DIRL+MEDIC (Ours) (Part 1). For each sample, we show the input image pair (top) and the addressing vector from the selected expert (bottom). Green-circled peaks represent type-specific memory slots that consistently appear across diferent samples of the same change type, while blue-circled peaks reflect input-specific slots adapted to the particular input.

![](images/abcbf2184a69945c414a9c20a39acf20a2afa83bcbf570cee33a8495b23ae69d.jpg)

![](images/3f55de70e67847fc1a76f445aeb0025426902cd50c7ceb25767b8a739f18930d.jpg)

![](images/e584cffd6c9dce045296b259bf5ee50e625208987c2eb5e2bbd794bc25e7218b.jpg)  
Drop Expert (Addressing Vector)  
Drop Expert (Addressing Vector)

![](images/5539f9832a8c928f9a3acb05868c332e93a058194f772aef942ddd4e657e1d51.jpg)

![](images/236047773a97178b4703ddb2ad297c8bf824e0491ee07c48c8ad263d19181cf4.jpg)

![](images/a24a946493f55e98245d6e311b43075bd9742395ff746790ad2b9dd5b95aa540.jpg)

![](images/4bca2ea2eabec6c58d2d28d922e02be5884ccef0d808a4a2a9f246d74e880683.jpg)  
Move Expert (Addressing Vector)

Move Expert (Addressing Vector)  
![](images/1b2889765fd90c68085b6d6bf7756d30e42bfdd0e3e9cdc92e7d94137b8b38dc.jpg)  
(a) Move Change (Example 1)

![](images/7a614f09c6497f9840f77d9d36365bee765416a07ed436601bcd7bce966854ac.jpg)

![](images/a0614690efc4f04c1a76aab85c970e47ef9035a142b69e49460adb8cbae8fc88.jpg)  
(b) Move Change (Example 2)

![](images/a84b3c2b1bfcc5f7f12f9ef598f8a36b4cfa392bc1537a445883a6e839d0fed9.jpg)  
(a) No Change (Example 1)

![](images/81f010cb10a190ea2180c1b1ccc8346875015da637919246762794b692d26302.jpg)

![](images/c40f38c1d0180a9927f50705bb22d15089e326f3bce1e1456e25c2c2872e6927.jpg)

No-Change Expert (Addressing Vector)  
![](images/b29c85c52ce041586f78b03ae466a013558e4f76a61f3c6c1bb58052593bcb3c.jpg)  
(b) No Change (Example 2)  
Fig. S.10: Dynamic slot activation examples on CLEVR-DC dataset using DIRL+MEDIC (Ours) (Part 2). For each sample, we show the input image pair (top) and the addressing vector from the selected expert (bottom). Green-circled peaks represent type-specific memory slots that consistently appear across diferent samples of the same change type, while blue-circled peaks reflect input-specific slots adapted to the particular input.