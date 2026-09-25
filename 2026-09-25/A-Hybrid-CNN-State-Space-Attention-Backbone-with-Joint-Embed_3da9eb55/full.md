# A Hybrid CNN–State-Space–Attention Backbone with Joint-Embedding Predictive Pretraining for 12-Lead ECG Classification

Yakoub Bazi<sup>a,∗</sup>, Sarah Aljuhani<sup>a</sup>, Mohamad M. Al Rahhal<sup>b</sup>, Mansour Zuair<sup>a</sup>, Naif Alajlan<sup>a</sup>

<sup>a</sup>Computer Engineering Department, College of Computer and Information Sciences, King Saud University, Riyadh, 11543, Saudi Arabia

<sup>b</sup>Applied Computer Science Department, College of Applied Computer Science, King Saud University, Riyadh, 11543, Saudi Arabia

## Abstract

Automatic 12-lead electrocardiogram (ECG) classification requires representations that jointly capture local waveform morphology, long-range temporal dynamics, and cross-lead dependencies, yet integrating these properties within a single eficient architecture remains challenging. This paper introduces a hybrid CNN–SSM–Attention backbone for 12-lead ECG classification. A convolutional stem performs early waveform tokenization and temporal reduction, mixed statespace and depthwise convolutional blocks model temporal dynamics and local morphology, and a late self-attention stage enables global token interaction at reduced resolution. To improve transfer from unlabeled data, we further develop an ECG-oriented Joint-Embedding Predictive Pretraining (JEPA) framework. Unlike ViT-based JEPA methods that mask patch tokens before the encoder, the proposed method samples span masks at the latent temporal resolution and projects them back to the waveform domain, then predicts clean latent targets from a momentum encoder without waveform reconstruction. Experiments on CPSC2018, Chapman-Shaoxing, and PTB-XL, with pretraining on approximately 350K unlabeled CODE-15 recordings, show that the proposed backbone provides strong supervised baselines under a compact parameter budget. JEPA pretraining further improves transfer, particularly in reduced-label settings and under both full fine-tuning and LoRA-based adaptation. Code: https://github.com/yakoubbazi/ Hybrid\_ECG\_Jepa.

Keywords: Electrocardiogram (ECG), 12-lead ECG, convolutional neural networks, state-space models, Mamba, self-attention, self-supervised learning, Joint-Embedding Predictive Pretraining (JEPA)

## 1. Introduction

Electrocardiography (ECG) is one of the most widely used non-invasive tools for cardiac assessment. A standard 12-lead ECG records cardiac electrical activity from complementary anatomical views, making it possible to assess rhythm disorders, conduction abnormalities, ischemic changes, and morphological deviations. Automated 12-lead ECG interpretation, however, remains challenging because clinically relevant patterns appear at multiple temporal and spatial scales. Some diagnostic cues are highly local, such as QRS morphology and ST–T changes, whereas others depend on longer temporal context and consistency across leads [1, 2].

Deep learning has substantially advanced automated ECG classification, but designing an efective backbone for 12-lead signals remains non-trivial. Convolutional neural networks provide strong local inductive biases for waveform morphology and remain competitive ECG baselines. Transformers, on the other hand, enable explicit global interaction among temporal tokens, but their quadratic complexity can be ineficient for long ECG sequences. More recently, selective state-space models (SSMs), including Mamba-style architectures, have provided an eficient alternative for long-context sequence modeling with linear-time complexity [3]. These developments suggest that local morphology extraction, eficient temporal sequence modeling, and global token interaction are complementary rather than mutually exclusive. Nevertheless, scalable hybrid CNN–SSM–Attention designs remain underexplored for 12-lead ECG classification.

A second challenge is the dependence on labeled ECG data. Large collections of raw ECG recordings are increasingly available, but expert annotations remain costly, time-consuming, and often dataset-specific. Self-supervised learning (SSL) has therefore become an important direction for ECG representation learning. Existing methods include contrastive learning, crossview objectives, masked reconstruction, and large-scale ECG foundation models [4, 5, 6, 7, 8, 9, 10]. Among these directions, joint-embedding predictive architectures (JEPA) are attractive because they avoid direct raw-signal reconstruction and instead learn by predicting latent targets from corrupted inputs. Recent ECG studies indicate that JEPA-style pretraining can learn transferable representations for downstream classification [11, 12].

In this work, we propose a scalable hybrid CNN–SSM–Attention backbone for 12-lead ECG classification. The architecture follows a staged design. A convolutional stem first extracts local waveform morphology and performs early temporal reduction. The intermediate stages use mixed sequence-morphology blocks that combine selective state-space mixing with lightweight depth-wise convolutional processing, allowing the model to capture long-range temporal dependencies while preserving local waveform structure. A final self-attention stage is then applied after temporal compression, enabling explicit global token interaction at a reduced computational cost. This design assigns each modeling mechanism to a suitable part of the network hierarchy: convolution for early local structure, SSM-based mixing for eficient temporal modeling, and attention for high-level global integration.

We further study JEPA-style self-supervised pretraining for the proposed backbone. Unlike ViT-based JEPA formulations, where masking is applied directly to patch tokens before the encoder, our method uses latent-aligned waveform corruption. Specifically, temporal masks are sampled at the latent temporal resolution of the backbone, which is reduced by a factor of four relative to the input waveform. The sampled mask is then expanded back to the raw waveform domain, where the selected ECG regions are replaced by learned lead-wise masking values with small additive noise. The online encoder processes this corrupted ECG, while a momentum target encoder processes the clean signal and provides latent prediction targets. Thus, the method performs masked latent prediction without reconstructing the raw ECG waveform and without inserting explicit ViT-style mask tokens into the encoder.

We evaluate the proposed framework on three public 12-lead ECG benchmarks: CPSC2018, Chapman, and PTB-XL. The experiments include supervised training from random initialization, full fine-tuning from self-supervised initialization, LoRA-based parameter-eficient adaptation, and reduced-label transfer settings. This evaluation is intended to assess both the intrinsic capacity of the proposed backbone and the transfer behavior induced by JEPA-style pretraining.

The main contributions of this work are summarized as follows:

1. We propose a scalable hybrid CNN–SSM–Attention backbone for 12-lead ECG classification, integrating convolutional morphology extraction, eficient state-space sequence modeling, and late self-attention within a unified architecture.

2. We introduce an ECG-specific JEPA pretraining strategy with latent-aligned waveform corruption, where masks are sampled at latent resolution and applied in the waveform domain for masked latent prediction.

3. We conduct comprehensive experiments on CPSC2018, Chapman, and PTB-XL, covering supervised training from scratch, full fine-tuning, LoRA-based adaptation, and reducedlabel transfer.

The remainder of this paper is organized as follows. Section 2 reviews related work on ECG backbones, state-space sequence modeling, and self-supervised ECG representation learning. Section 3 presents the proposed hybrid ECG backbone. Section 4 describes the JEPA-style self-supervised pretraining framework. Section 5 reports the experimental setup and results.

Section 6 concludes the paper.

## 2. Related Work

This section reviews prior work related to the proposed framework. We focus on two directions: backbone design for 12-lead ECG classification and self-supervised ECG representation learning.

## 2.1. Backbones for 12-Lead ECG Classification

Deep learning has become a dominant paradigm for automated 12-lead ECG classification. Early and widely used methods are largely based on one-dimensional convolutional neural networks, which are well matched to the local structure of ECG waveforms. Many diagnostic patterns, including QRS morphology, ST–T abnormalities, and beat-level deformation, are expressed through localized temporal changes. As a result, convolutional models remain strong baselines on public ECG benchmarks such as PTB-XL and CPSC2018 [1, 2]. However, ECG interpretation also requires broader temporal reasoning and cross-lead integration, since rhythm abnormalities, conduction patterns, and multilead morphological consistency may depend on information distributed across time and leads.

To address this limitation, recent studies have moved toward hybrid architectures that combine local waveform processing with mechanisms for longer-range or multiscale interaction. DAMS-Net introduces dual attention and multiscale feature fusion to capture complementary local and global representations [13]. MSGFormer and MCTNet further explore transformer-style and multiscale token interaction for 12-lead ECG analysis [14, 15]. Other convolution-attention hybrids similarly combine front-end morphology extraction with attention-based aggregation or temporal reasoning modules [16, 17]. These methods suggest that efective ECG backbones should not rely on a single modeling principle, but should instead combine local, temporal, and global representations in a structured manner.

State-space models (SSMs) have recently emerged as an eficient alternative for long-sequence modeling. In a standard discrete formulation, an SSM updates a latent state according to

$$
\mathbf { s } _ { t + 1 } = \mathbf { A } \mathbf { s } _ { t } + \mathbf { B } \mathbf { x } _ { t } , \qquad \mathbf { y } _ { t } = \mathbf { C } \mathbf { s } _ { t } + \mathbf { D } \mathbf { x } _ { t } ,\tag{1}
$$

where $\mathbf { x } _ { t }$ is the input at time step $t , \mathbf { s } _ { t }$ is the latent state, and $\mathbf { y } _ { t }$ is the output. Selective SSMs, particularly Mamba, make the state update input-dependent and provide linear-time sequence modeling, making them attractive for long ECG recordings [3]. This eficiency is important because clinically relevant ECG dependencies may span multiple beats, while full self-attention can be computationally expensive when applied at high temporal resolution.

Several recent works have adapted Mamba-style architectures to ECG analysis. ECGMamba uses bidirectional Mamba blocks with residual and feed-forward refinement for ECG classification [18]. ECG-Mamba follows a related direction by building a 12-lead classifier around a bidirectional SSM backbone and task-specific augmentation [19]. In these approaches, Mamba typically acts as the main temporal modeling component or as a substitute for transformer-style sequence processing.

Diferent from these works, our goal is not to replace attention with SSMs or to use Mamba as a standalone ECG backbone. Instead, we design a staged hybrid architecture motivated by a distinct modeling consideration. The convolutional stem performs early morphology extraction and temporal reduction. The intermediate stages combine selective state-space mixing with depth-wise convolutional processing to jointly model long-range temporal structure and local waveform patterns. The final stage applies self-attention only after temporal compression, enabling explicit global token interaction with reduced computational cost. This design provides a unified CNN–SSM–Attention hierarchy tailored to 12-lead ECG signals.

## 2.2. Self-Supervised ECG Representation Learning

A major challenge in ECG learning is the limited availability of high-quality diagnostic labels, which has motivated a broad range of SSL methods. Early ECG SSL approaches were dominated by contrastive learning. Representative examples include CLOCS, which constructs contrastive objectives across temporal, spatial, and patient-level views [4], and later multilead ECG SSL methods that exploit ECG-specific augmentations or cross-view consistency [5, 6, 7, 20, 21, 22]. A parallel line of work explores masked reconstruction or masked autoencoding, where the model reconstructs either raw waveforms or latent features from corrupted inputs [8, 10, 23].

More recently, JEPA-style latent predictive learning has emerged as an alternative to both contrastive learning and raw-signal reconstruction. Instead of enforcing invariance across augmented views or reconstructing waveform samples directly, JEPA predicts target representations in latent space using a momentum target encoder [24]. This formulation is particularly appealing for ECG because the downstream objective is not faithful recovery of every low-level fluctuation, but the learning of transferable latent structure that captures waveform morphology, rhythm patterns, and inter-lead relationships.

Recent ECG studies have started to explore this direction explicitly. ECG-JEPA adapts JEPA to 12-lead ECGs using masked latent prediction and a transformer-based encoder, and further introduces a cross-pattern attention design tailored to multilead ECG structure [11]. In another work, the authors also study JEPA-based self-supervised pretraining for ECG classification and report that latent predictive learning can outperform both invariance-based and generative pretraining strategies on PTB-XL benchmarks [12]. These results indicate that JEPA is a promising self-supervised paradigm for ECG representation learning.

Our pretraining strategy follows this JEPA direction but is paired with a diferent backbone design. Unlike prior ECG-JEPA work built primarily on transformer-style encoders, the present work couples JEPA with the proposed hybrid CNN–SSM–Attention backbone. Pretraining is based on masked latent prediction with a momentum target encoder and ECG-specific structured corruption, without waveform reconstruction or contrastive pairing. In this way, JEPA serves as a representation-learning strategy that complements the proposed backbone rather than defining the main architectural contribution.

## 2.3. Relation to the proposed method

The above literature suggests two main trends. First, 12-lead ECG backbones increasingly benefit from combining local waveform modeling with mechanisms that capture broader temporal and inter-lead context. Second, self-supervised ECG learning is moving from generic contrastive objectives toward masked and latent predictive formulations that better match the structure of multilead signals.

The proposed method combines these two directions in a unified framework. Its main contribution is a scalable hybrid CNN–SSM–Attention backbone in which convolution, state-space mixing, and self-attention are assigned distinct roles across the network hierarchy. JEPA-style pretraining is then used as a complementary self-supervised strategy for learning transferable latent representations on top of this backbone. In this sense, the proposed method lies at the intersection of recent hybrid ECG backbone design, emerging SSM-based ECG sequence modeling, and latent predictive SSL for multilead ECGs.

## 3. Proposed Scalable Hybrid ECG Backbone

## 3.1. Overview

We propose a scalable hybrid backbone for 12-lead ECG classification. The backbone is organized as a three-stage hierarchy that progressively transforms the raw multi-lead signal into a compact token representation. The overall design is guided by a simple idea: local waveform morphology should be captured early, longer temporal dependencies should be modeled eficiently as the sequence becomes shorter, and self-attention should be introduced only in the final stage, where global interaction is more afordable.

Let the input ECG segment be denoted by

$$
\mathbf { X } \in \mathbb { R } ^ { C \times T } ,\tag{2}
$$

where $C = 1 2$ is the number of leads and $T$ is the temporal length. In our setting, each ECG is represented as a 10-second recording sampled at 100 Hz, giving $T = 1 0 0 0$ time samples. The backbone maps X to a latent token sequence

$$
\mathbf { H } = g _ { \theta } ( \mathbf { X } ) \in \mathbb { R } ^ { T ^ { \prime } \times D } ,\tag{3}
$$

where $T ^ { \prime }$ is the final token length and D is the output feature dimension. In our implementation, the temporal resolution is reduced by a factor of four, yielding a final sequence length of approximately $T / 4$

As shown in Fig. 1, the network begins with a convolutional stem that performs early lead fusion and initial temporal downsampling. Stage 1 and Stage 2 then apply hybrid mixer blocks that combine eficient temporal modeling with local convolutional refinement. Stage 3 operates on the compressed sequence and uses self-attention to capture global relationships across tokens. Finally, the resulting token sequence is aggregated by attention pooling and passed to a classifier to produce the diagnostic prediction.

## 3.2. Convolutional Stem

The convolutional stem performs early lead fusion and temporal downsampling with a 1-D convolution of kernel size $^ { 7 , }$ stride 2, and padding 3, followed by group normalization and a SiLU nonlinearity. The kernel size of $7$ provides a moderately broad temporal receptive field for early waveform extraction while keeping the stem lightweight:

$$
\mathbf { H } _ { 1 } = \mathrm { S i L U } \big ( \mathrm { G N } ( \mathrm { C o n v 1 D } ( \mathbf { X } ) ) \big ) \in \mathbb { R } ^ { T _ { 1 } \times C _ { 1 } } ,\tag{4}
$$

where $T _ { 1 } = T / 2$ denotes the reduced temporal length and $C _ { 1 }$ denotes the stem feature dimension. This stage jointly integrates cross-lead information and reduces the temporal resolution, producing a compact representation for the subsequent backbone stages.

## 3.3. Stage 1: Hybrid mixer-based sequence modeling

Stage 1 takes the stem output $\mathbf { H } _ { 1 }$ as input and applies $N _ { 1 }$ repeated blocks, each consisting of a hybrid mixer followed by an MLP. As shown in Fig. 2, the hybrid mixer combines an SSM branch for temporal sequence modeling with a local gated depthwise-convolution branch for morphology-aware refinement. The rationale is to couple eficient long-range temporal mixing with explicit local waveform modeling within the same block. In particular, the SSM branch captures broader temporal dependencies, while the local branch is intended to preserve shortrange morphological structure. This design is inspired by the MambaVision mixer proposed for visual backbones [25], where a non-SSM convolutional path is introduced to complement the sequential branch and enrich the resulting representation, but is adapted here to 1-D multi-lead ECG signals. After fusion of the two branches, an MLP further refines the representation.

![](images/662bc07e209405355deaa186414cb79605f1306f9ad13847c4ab4d2115bba9dd.jpg)  
Figure 1: Overview of the proposed hybrid CNN–SSM–Attention backbone for 12-lead ECG classification. The architecture combines early convolutional tokenization, hybrid mixer stages, late self-attention, and attention pooling for final classification.

![](images/44d24b18d6ef047b687178f050b72ab9a758ede2fb923abe38b51018fbc3d160.jpg)  
Figure 2: Architecture of a Stage 1 block. The block operates at resolution $T _ { 1 } \times C _ { 1 }$ and combines an SSM branch for temporal sequence modeling with a local gated depthwise-convolution branch for morphology-aware refinement, followed by residual fusion and an MLP sublayer.

The computation within a single Stage 1 block can be described as follows. Given an input sequence $\mathbf { H } _ { 1 } \in \mathbb { R } ^ { T _ { 1 } \times C _ { 1 } }$ , the block first applies RMS normalization:

$$
\begin{array} { r } { \widetilde { \bf H } _ { 1 } = \mathrm { R M S N o r m } ( { \bf H } _ { 1 } ) . } \end{array}\tag{5}
$$

The first branch then applies a state-space sequence modeling module:

$$
\begin{array} { r } { \mathbf { A } = \mathrm { S S M } ( \widetilde { \mathbf { H } } _ { 1 } ) , } \end{array}\tag{6}
$$

which serves as an eficient temporal mixing operator for capturing longer-range dependencies in the ECG sequence [3].

The second branch applies depthwise convolution followed by gated channel mixing:

$$
{ \bf U } , { \bf V } = \mathrm { s p l i t } \Big ( { \cal W } _ { \mathrm { i n } } \mathrm { D W C o n v } ( \widetilde { \bf H } _ { 1 } ) \Big ) ,\tag{7}
$$

$$
\mathbf { B } = W _ { \mathrm { o u t } } \Big ( \mathrm { S i L U } ( \mathbf { U } ) \odot \mathbf { V } \Big ) ,\tag{8}
$$

where DWConv(·) denotes depthwise 1-D convolution, ⊙ denotes element-wise multiplication, and $W _ { \mathrm { i n } }$ and $W _ { \mathrm { o u t } }$ are learned input and output projection matrices of the local branch. The depthwise convolution provides lightweight local temporal filtering, while the gating operation adaptively modulates the resulting features so that informative short-range waveform patterns can be emphasized before fusion.

The outputs of the two branches are then concatenated, linearly fused, and added through a residual connection:

$$
\mathbf { H } _ { 1 } ^ { \prime } = \mathbf { H } _ { 1 } + \operatorname { D r o p o u t } \Bigl ( W _ { \mathrm { f u s e } } [ \mathbf { A } ; \mathbf { B } ] \Bigr ) ,\tag{9}
$$

thereby retaining complementary information from the SSM and local convolutional paths within a unified token representation.

The mixer output ${ \bf { H } } _ { 1 } ^ { \prime }$ is then passed to the MLP, which applies two successive linear projections separated by a SiLU nonlinearity:

$$
\begin{array} { r l } & { \mathrm { M L P } ( \mathbf { H } _ { 1 } ^ { \prime } ) = \mathbf { H } _ { 1 } ^ { \prime } + \mathrm { D r o p o u t } \Big ( W _ { \mathrm { m l p , 2 } } \mathrm { S i L U } ( \mathbf { Q } ) \Big ) , } \\ & { \quad \quad \quad \quad \mathbf { Q } = W _ { \mathrm { m l p , 1 } } \mathrm { R M S N o r m } ( \mathbf { H } _ { 1 } ^ { \prime } ) . } \end{array}\tag{10}
$$

The MLP further refines the fused representation through channel mixing by first expanding the feature dimension with $W _ { \mathrm { m l p } , 1 }$ and then projecting it back to the original dimension with $W _ { \mathrm { m l p } , 2 }$ . Repeating this block $N _ { 1 }$ times yields the Stage 1 output, denoted by $\mathbf { Z } _ { 1 } \in \mathbb { R } ^ { T _ { 1 } \times C _ { 1 } }$ , while preserving the feature resolution.

## 3.4. Stage 2: Hybrid mixer-based sequence modeling at reduced resolution

Stage 2 begins with a convolutional transition applied to the Stage 1 output $\mathbf { Z } _ { 1 }$ , which further reduces the temporal resolution and expands the feature dimension:

$$
\mathbf { H } _ { 2 } = \operatorname { C o n v } 1 \mathrm { D } ( \mathbf { Z } _ { 1 } ) \in \mathbb { R } ^ { T _ { 2 } \times C _ { 2 } } ,\tag{11}
$$

where the convolution uses kernel size 5, stride 2, and padding 2, yielding $T _ { 2 } = T / 4$ , while $C _ { 2 }$ denotes the expanded feature dimension. The resulting sequence is then processed by the same hybrid mixer-based blocks used in Stage 1. Accordingly, Stage 2 retains the same combination of SSM-based temporal mixing and local gated depthwise-convolutional refinement, but applies it to a shorter and higher-dimensional representation. This allows the model to capture longer-range temporal structure more eficiently while preserving sensitivity to local waveform morphology. Repeating these blocks yields the Stage 2 output, denoted by $\mathbf { Z } _ { 2 } \in \mathbb { R } ^ { T _ { 2 } \times C _ { 2 } }$

## 3.5. Stage 3: Global sequence modeling with self-attention

Stage 3 begins with a convolutional transition applied to the Stage 2 output $\mathbf { Z } _ { 2 }$ , which projects the representation to a higher-dimensional space while preserving the temporal resolution:

$$
\mathbf { H } _ { 3 } = \mathrm { C o n v } 1 \mathrm { D } ( \mathbf { Z } _ { 2 } ) \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } } ,\tag{12}
$$

where the convolution uses kernel size 5, stride 1, and padding 2, yielding $T _ { 3 } = T _ { 2 }$ , while $C _ { 3 }$ denotes the expanded feature dimension. Since the temporal resolution has already been reduced before Stage 3, no further downsampling is applied at this point; instead, the transition preserves the compressed token sequence while projecting it to a higher-dimensional space prior to global self-attention. Unlike the preceding stages, Stage 3 adopts a pure Transformer-style design composed of repeated self-attention and MLP blocks, as illustrated in Fig. 3. Operating at the reduced temporal resolution makes self-attention computationally afordable while enabling explicit global interaction across tokens. In this way, Stage 3 complements the earlier SSM-based stages by emphasizing long-range dependencies and global contextual integration at the highest representation level.

The computation within a single Stage 3 block can be described as follows. Given an input sequence $\mathbf { H } _ { 3 } \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } }$ , the self-attention sublayer is defined as

$$
\mathbf { H } _ { 3 } ^ { \prime } = \mathbf { H } _ { 3 } + \operatorname { D r o p o u t } \Bigl ( \mathrm { M H S A } \bigl ( \mathrm { R M S N o r m } ( \mathbf { H } _ { 3 } ) \bigr ) \Bigr ) ,\tag{13}
$$

where MHSA denotes multi-head self-attention. The resulting representation is then refined by an MLP sublayer:

$$
\begin{array} { r l } & { \mathrm { M L P } ( \mathbf { H } _ { 3 } ^ { \prime } ) = \mathbf { H } _ { 3 } ^ { \prime } + \mathrm { D r o p o u t } \Big ( W _ { \mathrm { m l p , 2 } } \mathrm { S i L U } ( \mathbf { Q } ) \Big ) , } \\ & { \quad \quad \quad \quad \mathbf { Q } = W _ { \mathrm { m l p , 1 } } \mathrm { R M S N o r m } ( \mathbf { H } _ { 3 } ^ { \prime } ) . } \end{array}\tag{14}
$$

Thus, each Stage 3 block follows the standard pre-normalized Transformer pattern of selfattention followed by an MLP, with a residual connection around each sublayer. Repeating this block $N _ { 3 }$ times yields the final encoded representation, denoted by $\mathbf { Z } _ { 3 } \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } }$

## 3.6. Classification head

To obtain a compact sequence-level representation, the final encoded sequence produced by Stage 3, denoted by $\mathbf { Z } _ { 3 } = \{ \mathbf { z } _ { t } \} _ { t = 1 } ^ { T _ { 3 } } \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } }$ , is aggregated using attention pooling. This choice allows the model to assign larger weights to diagnostically informative temporal regions, rather than treating all tokens equally as in average pooling. The attention weight assigned to each token is computed as

$$
\alpha _ { t } = \frac { \exp \bigl ( \mathbf { w } ^ { \top } \operatorname { t a n h } ( \mathbf { W } _ { p } \mathbf { z } _ { t } ) \bigr ) } { \sum _ { t ^ { \prime } = 1 } ^ { T _ { 3 } } \exp \bigl ( \mathbf { w } ^ { \top } \operatorname { t a n h } ( \mathbf { W } _ { p } \mathbf { z } _ { t ^ { \prime } } ) \bigr ) } ,\tag{15}
$$

![](images/c086cab26c7aa602339272bddd86c60b3ab1805f0623d3f5f2672d982981b93b.jpg)  
Figure 3: Architecture of a Stage 3 block. Operating at resolution $T _ { 3 } \times C _ { 3 }$ , each block follows a Transformer-style design composed of a multi-head self-attention sublayer and an MLP sublayer, with a residual connection around each component.

Table 1: Scalable backbone configurations used in this work. $C _ { 1 } , \ C _ { 2 }$ , and $C _ { 3 }$ are the channel dimensions of Stages 1–3, and $N _ { 1 } , N _ { 2 }$ , and $N _ { 3 }$ are the numbers of blocks in each stage. Heads is the number of attention heads in the Stage 3 self-attention blocks, and Params is the total number of parameters.
<table><tr><td>Variant</td><td> $C _ { 1 }$ </td><td> $C _ { 2 }$ </td><td> $C _ { 3 }$ </td><td> $N _ { 1 }$ </td><td> $N _ { 2 }$ </td><td> $N _ { 3 }$ </td><td>Heads</td><td>Params</td></tr><tr><td>Micro</td><td>32</td><td>64</td><td>64</td><td>1</td><td>1</td><td>1</td><td>2</td><td>0.21M</td></tr><tr><td>Mini</td><td>48</td><td>96</td><td>96</td><td>1</td><td>2</td><td>2</td><td>2</td><td>0.78M</td></tr><tr><td>Tiny</td><td>64</td><td>128</td><td>256</td><td>2</td><td>2</td><td>2</td><td>4</td><td>3.11M</td></tr><tr><td>Small</td><td>64</td><td>128</td><td>256</td><td>2</td><td>2</td><td>3</td><td>4</td><td>4.17M</td></tr></table>

where $\mathbf { W } _ { p }$ and w are learnable parameters of the attention-pooling module. The pooled representation is then given by

$$
\bar { \mathbf { z } } = \sum _ { t = 1 } ^ { T _ { 3 } } \alpha _ { t } \mathbf { z } _ { t } .\tag{16}
$$

The aggregated feature is finally passed to a linear classifier:

$$
\hat { \mathbf { p } } = \mathrm { s o f t m a x } ( \mathbf { W } _ { c } \bar { \mathbf { z } } + \mathbf { b } _ { c } ) ,\tag{17}
$$

where $\mathbf { W } _ { c }$ and ${ \bf b } _ { c }$ denote the classifier weight matrix and bias vector, respectively.

## 3.7. Scalable model variants

We implement the backbone in four sizes: Micro, Mini, Tiny, and Small (Table 1). All variants share the same three-stage design and difer only in channel width, stage depth, and number of attention heads. Scaling capacity through width and depth, while keeping the architecture fixed, is common practice in hierarchical backbones and lets us study the efect of model size without changing the underlying structure.

This scalability serves two purposes. It lets the same design fit diferent computational budgets, from lightweight to more expressive models, and it enables a controlled study of how representation quality changes with capacity under a fixed CNN–SSM–Attention formulation.

The models remain compact overall, ranging from 0.21M parameters for Micro to 4.17M for Small, which is lightweight relative to typical transformer-based ECG backbones. Unless stated otherwise, we use the Small variant for the main pretraining and downstream experiments.

## 4. Self-Supervised Pretraining

We pretrain the proposed backbone on unlabeled 12-lead ECG recordings using a jointembedding predictive objective inspired by JEPA [24]. The model is not trained to reconstruct the raw ECG waveform. Instead, it predicts the latent representation of masked temporal regions from a corrupted multilead input. This design encourages the encoder to learn waveform representations that preserve local morphology, temporal context, and cross-lead structure in feature space.

As illustrated in Fig. 4, the pretraining framework consists of an online encoder $f _ { \theta }$ , a momentum target encoder $f _ { \xi }$ , and a predictor $g _ { \phi }$ . Given a clean ECG segment $\mathbf { X } \in \mathbb { R } ^ { 1 2 \times T }$ , a corrupted view $\tilde { \mathbf { X } }$ is generated before the online encoder. The clean signal X is processed by the momentum target encoder, while the corrupted signal $\tilde { \mathbf { X } }$ is processed by the online encoder. The target encoder is updated as an exponential moving average of the online encoder:

$$
\xi  m \xi + ( 1 - m ) \theta ,\tag{18}
$$

where $\theta$ and $\xi$ denote the online and target encoder parameters, respectively, and $m$ is the momentum coeficient.

Both encoders produce final-stage temporal representations. The target encoder maps the clean ECG to

$$
\mathbf { Z } _ { 3 } = f _ { \boldsymbol { \xi } } ( \mathbf { X } ) = \{ \mathbf { z } _ { t } \} _ { t = 1 } ^ { T _ { 3 } } \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } } ,
$$

where $T _ { 3 }$ is the final temporal token length and $C _ { 3 }$ is the feature dimension. In parallel, the online encoder maps the corrupted ECG to

$$
\begin{array} { r } { \tilde { \mathbf { Z } } _ { 3 } = f _ { \theta } ( \tilde { \mathbf { X } } ) = \{ \tilde { \mathbf { z } } _ { t } \} _ { t = 1 } ^ { T _ { 3 } } \in \mathbb { R } ^ { T _ { 3 } \times C _ { 3 } } . } \end{array}
$$

The prediction target is defined at the final Stage 3 representation level, which is the same representation later used by the downstream attention-pooling classifier.

The predictor $g _ { \phi }$ maps the online representation $\tilde { \mathbf { Z } } _ { 3 }$ to the target latent space. The loss is computed only over the masked temporal token set M:

$$
\mathcal { L } _ { \mathrm { J E P A } } = \frac { 1 } { | \mathcal { M } | } \sum _ { t \in \mathcal { M } } \mathrm { S m o o t h L 1 } \left( g _ { \phi } ( \tilde { \mathbf { Z } } _ { 3 } ) _ { t } , \mathrm { s g } ( \mathbf { z } _ { t } ) \right) ,\tag{19}
$$

![](images/6d08de04b2f608d5b96f712e35818730afe33079460fc60d4f58a74a8a372768.jpg)  
Figure 4: Overview of the proposed ECG-JEPA pretraining scheme. A clean 12-lead ECG segment is processed by the momentum target encoder $f _ { \xi }$ to produce the final Stage 3 target representation $\mathbf { Z } _ { 3 }$ . A corrupted view of the same signal is processed by the online encoder $f _ { \theta } ,$ followed by the predictor $g _ { \phi } .$ , to predict masked final-stage latent targets. The loss is computed only over masked temporal positions.

where $g _ { \phi } ( \tilde { \mathbf { Z } } _ { 3 } )$ is the predicted latent vector at masked position $t , \mathbf { z } _ { t }$ is the corresponding target vector from the momentum encoder, and $\operatorname { s g } ( \cdot )$ denotes stop-gradient. No waveform reconstruction loss or auxiliary representation loss is used.

The corruption process is defined by two operations. First, random temporal span masks are sampled at the final latent temporal resolution with an overall masking ratio of 75%. Each selected token corresponds to a temporal region in the input signal; therefore, the sampled mask is expanded back to the waveform domain and the corresponding raw ECG regions are corrupted before entering the online encoder. This difers from ViT-style masked modeling, where explicit mask tokens are inserted into the token sequence. In our case, the encoder receives a corrupted waveform rather than learnable mask-token embeddings. Second, lead dropout is applied with probability 0.25 by selecting one or two complete leads and replacing them with noise. This simulates missing or unreliable channels and encourages the model to exploit cross-lead redundancy.

Overall, the proposed pretraining scheme learns by predicting clean final-stage latent targets from corrupted multilead ECG inputs. By aligning the predictor output with $\mathbf { Z } _ { 3 }$ , the objective directly optimizes the representation level used for downstream classification.

## 4.1. LoRA-Based Adaptation

In addition to full fine-tuning, we evaluate parameter-eficient adaptation using LoRA. For a pretrained linear layer with frozen weight $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , LoRA introduces a trainable low-rank

update:

$$
\mathbf { W } _ { \mathrm { L o R A } } = \mathbf { W } _ { 0 } + { \frac { \alpha } { r } } \mathbf { B } \mathbf { A } ,\tag{20}
$$

where $\mathbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ and $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ are trainable matrices, r is the adapter rank, and α is the scaling factor. In our backbone, LoRA is applied to selected linear projections inside the encoder, including attention and MLP-related linear layers. During LoRA adaptation, the original pretrained weights $\mathbf { W } _ { 0 }$ remain frozen, while only the low-rank adapter parameters are updated. The attention-pooling layer and classification head remain fully trainable.

This setting provides a parameter-eficient way to adapt the SSL-pretrained ECG encoder to downstream datasets. It reduces the number of trainable encoder parameters, helps preserve the pretrained representation, and can be useful when labeled ECG data are limited. In our experiments, LoRA uses rank $r = 3 2$ , scaling factor $\alpha = 1 2 8$ , and dropout 0.05.

## 5. Experiments

This section presents the experimental evaluation of the proposed framework. We first describe the experimental setup, including datasets, preprocessing, implementation details, and evaluation protocol. We then evaluate the backbone variants under supervised training from scratch, followed by JEPA-style transfer under full-label and reduced-label settings. Finally, we compare the proposed method with existing 12-lead ECG classification approaches.

## 5.1. Datasets and preprocessing

We use four public 12-lead ECG datasets in this study. CODE-15% is used for self-supervised pretraining, whereas CPSC2018, Chapman-Shaoxing, and PTB-XL are used for downstream evaluation. CODE-15% contains 345,779 ECG records from 233,770 patients and is used only as an unlabeled waveform source. CPSC2018 contains 6,877 recordings sampled at 500 Hz with durations between 6 and 60 s. Chapman-Shaoxing contains 10,646 10-second recordings sampled at 500 Hz. PTB-XL contains 21,837 10-second recordings from 18,885 subjects and is provided at both 100 Hz and 500 Hz.

For downstream evaluation, CPSC2018 is formulated as a 9-class single-label task with Normal/SNR (normal sinus rhythm), AF (atrial fibrillation), IAVB (first-degree atrioventricular block), LBBB (left bundle branch block), RBBB (right bundle branch block), PAC (premature atrial contraction), PVC (premature ventricular contraction), STD (ST-segment depression), and STE (ST-segment elevation). Chapman–Shaoxing is evaluated under a 4-class rhythm setting with AFIB (atrial fibrillation), GSVT (general supraventricular tachycardia), SB (sinus bradycardia), and SR (sinus rhythm). PTB-XL follows the diagnostic superclass protocol with

Table 2: Downstream dataset statistics after single-label filtering. Majority/minority classes and Max/Min ratios are computed from the training split.
<table><tr><td>Dataset</td><td>Classes</td><td>Train</td><td>Val</td><td>Test</td><td>Majority</td><td>Minority</td><td>Max/Min</td></tr><tr><td>CPSC2018</td><td>9</td><td>5,121</td><td>640</td><td>640</td><td>RBBB (1,227)</td><td>LBBB (143)</td><td>8.58</td></tr><tr><td>Chapman-Shaoxing</td><td>4</td><td>6,871</td><td>859</td><td>859</td><td>SB (3,085)</td><td>GSVT (573)</td><td>5.38</td></tr><tr><td>PTB-XL</td><td>5</td><td></td><td></td><td></td><td>12,957 1,637 1,650 NORM (7,243)</td><td>HYP (415)</td><td>17.45</td></tr></table>

NORM (normal ECG), MI (myocardial infarction), STTC (ST/T change), CD (conduction disturbance), and HYP (hypertrophy).

For data splitting, CPSC2018 follows a fold-based protocol with folds 1–8 for training, fold 9 for validation, and fold 10 for testing. Chapman-Shaoxing uses a stratified 80/10/10 train/validation/test split. PTB-XL preserves the oficial patient-wise split, using folds 1–8 for training, fold 9 for validation, and fold 10 for testing. The resulting split statistics after singlelabel filtering are summarized in Table 2. The train sets remain imbalanced, with maximum-tominimum class ratios of 8.58, 5.38, and 17.45 for CPSC2018, Chapman-Shaoxing, and PTB-XL, respectively.

All datasets are converted to a unified input format. Signals are reordered to the canonical lead arrangement {I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6}. Recordings are resampled to 100 Hz and represented as fixed-length 10-second segments, yielding $1 2 \times 1 0 0 0$ waveform tensors. Longer recordings are center-cropped, whereas shorter recordings are resized to the target length using linear interpolation. During training, a zero-phase fourth-order Butterworth band-pass filter is applied before per-lead normalization.

## 5.2. Implementation details

All models are implemented in PyTorch and trained with fixed-size 12×1000 ECG inputs. For self-supervised pretraining, the backbone is trained on 345,779 unlabeled CODE-15% recordings. We use AdamW with learning rate $7 . 4 \times 1 0 ^ { - 5 }$ , weight decay 0.05, and $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 5 )$ Training is performed for 50K optimization steps with batch size 16 and gradient accumulation over 8 steps, resulting in an efective batch size of 128. The learning rate is linearly warmed up for 4K steps and then decayed with a cosine schedule using a minimum learning-rate ratio of 0.01. Gradients are clipped to a maximum norm of 1.0. The momentum target encoder is updated by exponential moving average, with the decay coeficient scheduled from 0.998 to 0.9999.

The self-supervised objective is masked latent prediction with a Smooth- $\cdot \ell _ { 1 }$ loss $( \beta = 1 . 0 )$

The predictor is a transformer-style module with four blocks, four attention heads, dropout 0.1, and hidden dimension $d _ { p } = \operatorname* { m a x } ( 1 . 5 C _ { 3 } , 1 2 8 )$ , where $C _ { 3 }$ denotes the final encoder dimension. For the Small backbone, $C _ { 3 } = 2 5 6$ and $d _ { p } = 3 8 4$ . The online branch receives the corrupted ECG signal, whereas the momentum branch processes the clean signal and provides stop-gradient latent targets.

The masking policy is fixed during pretraining. Random temporal spans are sampled at the latent temporal resolution using a masking ratio of 75% and span lengths of 120–400 ms. The sampled masks are expanded back to the waveform domain, where the corresponding ECG regions are replaced by learned lead-wise masking values with small additive noise. Lead dropout is applied with probability 0.25 by replacing one or two complete leads with noise.

For downstream classification, an attention-pooling layer and a lightweight classification head are attached to the encoder. Models are trained for up to 50 epochs using AdamW, weight decay 0.05, cosine learning-rate decay, and a 5% warm-up ratio. We use batch size 16 with gradient accumulation over 2 steps. The checkpoint with the highest validation macro-F1 is selected for testing, and early stopping is applied. Class imbalance is handled using class-reweighted cross-entropy with inverse-square-root frequency weighting.

We consider three training modes. In scratch training, all parameters are optimized from random initialization. In full SSL fine-tuning, the pretrained encoder, attention-pooling layer, and classification head are updated jointly. In LoRA-based SSL adaptation, the pretrained encoder is frozen except for low-rank adapters inserted into selected linear layers, while the attention-pooling layer and classifier remain trainable. LoRA uses rank $r = 3 2$ , scaling factor $\alpha = 1 2 8$ , and dropout 0.05. The learning rates are $1 \times 1 0 ^ { - 4 }$ for scratch training, $3 \times 1 0 ^ { - 5 }$ for full SSL fine-tuning, and $1 \times 1 0 ^ { - 4 }$ for LoRA-based SSL adaptation. All experiments are conducted on a single NVIDIA RTX A6000 GPU with 48 GB of memory.

## 5.3. Evaluation protocol

We report AUC, accuracy, macro-F1, and Cohen’s κ. Macro-F1 is treated as the primary metric because the downstream datasets contain imbalanced diagnostic or rhythm categories. AUC, accuracy, and κ are reported as complementary measures of class separability, overall correctness, and agreement beyond chance.

For each downstream run, the checkpoint with the highest validation macro-F1 is selected for final test evaluation. In the reduced-label setting, we sample stratified subsets of the labeled training set using 1%, 5%, 10%, and 20% of the available training data. For each fraction, the same subset manifest is shared across scratch, SSL-Full, and SSL-LoRA to ensure matched comparisons. Each reduced-label experiment is repeated three times with diferent sampling seeds, and results are reported as mean ± standard deviation.

Table 3: Scratch performance of the backbone variants on CPSC2018, Chapman-Shaoxing, and PTB-XL. All values are reported in percentage (%).
<table><tr><td rowspan="2">Variant</td><td colspan="4">CPSC2018</td><td colspan="4">Chapman-Shaoxing</td><td colspan="4">PTB-XL</td></tr><tr><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td></tr><tr><td>Micro</td><td>94.66</td><td>76.88</td><td>71.58</td><td>72.90</td><td>98.60</td><td>93.60</td><td>91.71</td><td>90.64</td><td>90.02</td><td>76.97</td><td>62.13</td><td>62.99</td></tr><tr><td>Mini</td><td>94.68</td><td>80.16</td><td>75.25</td><td>76.74</td><td>98.52</td><td>94.30</td><td>92.38</td><td>91.67</td><td>89.87</td><td>76.24</td><td>62.80</td><td>61.53</td></tr><tr><td>Tiny</td><td>94.95</td><td>78.91</td><td>73.23</td><td>75.25</td><td>98.10</td><td>94.06</td><td>90.70</td><td>91.26</td><td>89.14</td><td>74.42</td><td>61.06</td><td>58.57</td></tr><tr><td>Small</td><td>95.41</td><td>80.16</td><td>74.56</td><td>76.76</td><td>98.60</td><td>94.88</td><td>92.70</td><td>92.47</td><td>88.91</td><td>77.52</td><td>64.75</td><td>64.13</td></tr></table>

## 5.4. Supervised Baselines from Scratch

We first train the backbone family from random initialization, without any self-supervised pretraining. The goal is to see how model size afects performance before we add self-supervised learning. We test the four proposed variants, Micro, Mini, Tiny, and Small, which grow in size from 0.21M to 4.17M parameters and difer in channel width and the number of blocks.

Table 3 shows the results for the four variants. Small performs best overall. It gives the highest accuracy, macro-F1, and κ on both Chapman-Shaoxing and PTB-XL, and the highest AUC and κ on CPSC2018. The main exception is Mini, which gives a slightly higher macro-F1 than Small on CPSC2018 (75.25 vs. 74.56). Micro gives the highest AUC on PTB-XL.

The efect of model size is small and not always in the same direction. Larger models tend to give better results on Chapman-Shaoxing and PTB-XL, but the smaller models are still strong. For example, Micro (0.21M) matches or beats Small (4.17M) on PTB-XL AUC, and Mini (0.78M) is close to Small on all three datasets. Because many of the gaps between variants are small, we treat these diferences as a general trend rather than a strict ranking. In practice, this is a useful property, since it shows that the model already works well even with very few parameters.

Based on these results, we use the Small backbone for the self-supervised experiments. It gives the best or joint-best macro-F1 and κ on two of the three datasets and stays competitive on CPSC2018, so it is a reliable choice for studying JEPA-style pretraining. Mini is a good lightweight option when a smaller model is needed.

![](images/d5754ee87980bf34129040fb50f8174bcab355518d9b4368d028b04bd1b21763.jpg)  
(a) 25% temporal masking.

![](images/371594facbbc632f6c2dee246041bb581025b82218e9bf5378e0cdb9fdf9d372.jpg)  
(b) 75% temporal masking.  
Figure 5: Representative ECG waveform corruption under diferent temporal masking ratios. Temporal masks are sampled at the latent token resolution and projected back to the waveform domain before the online encoder. Lower masking preserves more visible waveform morphology, whereas higher masking imposes a stronger contextual prediction task. Lead dropout is applied as an additional channel-level corruption.

## 5.5. Self-Supervised Pretraining and Downstream Transfer

We next evaluate whether JEPA-style pretraining improves downstream ECG classification. The Small backbone is first pretrained on unlabeled CODE-15% recordings and then adapted to CPSC2018, Chapman-Shaoxing, and PTB-XL. The purpose of this experiment is to compare the same architecture under three settings: supervised training from scratch, LoRA-based adaptation from the SSL-pretrained encoder, and full fine-tuning from the SSL-pretrained encoder.

Figure 5 shows representative corruption examples under two temporal masking ratios. Random temporal span masks are sampled at the latent temporal resolution and expanded back to the waveform domain, where the corresponding ECG regions are corrupted before entering the online encoder. The 25% setting preserves more visible waveform morphology, whereas the 75% setting removes a larger portion of the temporal context and therefore imposes a harder latent prediction task. Lead dropout is also applied as an additional channel-level corruption. In the main transfer experiments, the temporal masking ratio is fixed at 75%, while the lower masking ratios are evaluated separately in the masking-sensitivity analysis. The target branch receives the corresponding clean ECG and provides latent targets through the momentum encoder; thus, the model predicts latent representations from partially corrupted ECG signals rather than reconstructing raw waveforms.

Table 4 reports the full-label transfer results. The results show that SSL pretraining provides clear gains on CPSC2018 and Chapman-Shaoxing. On CPSC2018, SSL-Full achieves the best performance across all metrics, indicating that the pretrained encoder remains useful even when full labeled data are available. On Chapman-Shaoxing, SSL-LoRA gives the strongest results, suggesting that parameter-eficient adaptation can preserve and exploit the pretrained representation efectively.

Table 4: Transfer results of the Small backbone on CPSC2018, Chapman-Shaoxing, and PTB-XL. SSL-pretrained models are adapted using LoRA and full fine-tuning. All values are reported in percentage (%).
<table><tr><td rowspan="2">Mode</td><td colspan="4">CPSC2018</td><td colspan="4">Chapman-Shaoxing</td><td colspan="4">PTB-XL</td></tr><tr><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td></tr><tr><td>Scratch</td><td>95.41</td><td>80.16</td><td>74.56</td><td>76.76</td><td>98.60</td><td>94.88</td><td>92.70</td><td>92.47</td><td>88.91</td><td>77.52</td><td>64.75</td><td>64.13</td></tr><tr><td>SSL-LoRA</td><td>95.39</td><td>82.03</td><td>77.58</td><td>78.93</td><td>99.15</td><td>97.09</td><td>94.92</td><td>95.71</td><td>90.21</td><td>77.03</td><td>61.56</td><td>63.05</td></tr><tr><td>SSL-Full</td><td>96.67</td><td></td><td>82.97 78.58</td><td>80.03</td><td>98.14</td><td>96.27</td><td>94.10</td><td>94.55</td><td>89.92</td><td>77.82</td><td>61.95</td><td>63.08</td></tr></table>

On PTB-XL, the efect is more metric-dependent. SSL-LoRA gives the highest AUC and SSL-Full gives the highest accuracy, whereas scratch training remains better in macro-F1 and κ. This indicates that SSL initialization improves ranking-oriented separability but does not uniformly improve the final class-balanced decision metrics under full supervision. Given the stronger class imbalance in PTB-XL, this behavior suggests that SSL pretraining is beneficial but its impact depends on the downstream label distribution and adaptation mode.

## 5.6. Low-Label Transfer Analysis

We further evaluate downstream transfer under limited supervision. For each dataset, we sample stratified subsets containing 1%, 5%, 10%, and 20% of the labeled training data. The same subset manifest is used across scratch, SSL-LoRA, and SSL-Full to ensure matched comparisons. Each experiment is repeated over three sampling seeds, and results are reported as mean ± standard deviation.

Tables 5–7 report the reduced-label transfer results. In general, SSL initialization provides the largest gains when the amount of labeled data is small. The efect is most pronounced on CPSC2018 and Chapman-Shaoxing, where SSL-based adaptation consistently improves over scratch across all training fractions. On CPSC2018, full fine-tuning gives the strongest overall performance at most fractions, while LoRA remains competitive and gives the highest AUC at selected fractions. On Chapman-Shaoxing, LoRA is particularly efective at the lowest label fraction, whereas full fine-tuning becomes slightly stronger as more labels are available.

The trend on PTB-XL is more moderate. SSL-based adaptation improves most metrics at 1%, 5%, and 10%, indicating that the pretrained encoder provides a useful representation prior under limited supervision. At 20%, the diferences become smaller and metric-dependent, with scratch remaining competitive for κ while SSL-based models retain advantages in AUC, accuracy, or macro-F1. This behavior is consistent with the stronger class imbalance of PTB-XL and suggests that the benefit of SSL initialization depends on both the label budget and the downstream label distribution.

Table 5: Low-label transfer results on CPSC2018 using diferent fractions of the labeled training data. SSLpretrained models are adapted using LoRA and full fine-tuning. Results are reported as mean ± standard deviation over three runs, in percentage (%).
<table><tr><td>Train Fraction</td><td>Mode</td><td>AUC</td><td>Acc</td><td>Macro-F1</td><td>κ</td></tr><tr><td rowspan="3">1%</td><td>Scratch</td><td> $7 1 . 6 2 \pm 1 . 2 4$ </td><td> $3 6 . 9 8 \pm 2 . 1 3$ </td><td> $2 8 . 1 1 \pm 2 . 9 0$ </td><td> $2 6 . 3 5 \pm 2 . 6 8$ </td></tr><tr><td>SSL-LoRA</td><td> $8 1 . 3 7 \pm 1 . 1 4$ </td><td> $5 1 . 1 5 \pm 3 . 5 6$ </td><td> $3 9 . 5 8 \pm 6 . 3 1$ </td><td> $4 2 . 4 3 \pm 4 . 1 7$ </td></tr><tr><td>SSL-Full</td><td> $\mathbf { 8 2 . 9 2 \pm 1 . 0 0 }$ </td><td> ${ \bf 5 4 . 5 3 \pm 3 . 1 7 }$ </td><td> ${ \bf 4 5 . 7 0 \pm 4 . 3 3 }$ </td><td> ${ \bf 4 6 . 6 0 \pm 3 . 7 4 }$ </td></tr><tr><td rowspan="3">5%</td><td>Scratch</td><td> $8 4 . 4 1 \pm 0 . 5 2$ </td><td> $5 7 . 0 3 \pm 1 . 8 4$ </td><td> $4 9 . 5 0 \pm 2 . 7 3$ </td><td> $4 9 . 8 4 \pm 2 . 1 9$ </td></tr><tr><td>SSL-LoRA</td><td> ${ \bf 9 2 . 0 6 \pm 0 . 2 7 }$ </td><td> $\mathbf { 7 2 . 4 5 \ : \pm { \ : 0 . 7 2 } }$ </td><td> ${ \bf 6 5 . 3 0 \pm 2 . 4 0 }$ </td><td> ${ \bf 6 7 . 7 1 \pm 1 . 0 0 }$ </td></tr><tr><td>SSL-Full</td><td> $9 1 . 6 5 \pm 0 . 5 0$ </td><td> $7 1 . 8 2 \pm 0 . 8 9$ </td><td> $6 5 . 1 2 \pm 2 . 0 8$ </td><td> $6 6 . 9 9 \pm 1 . 1 5$ </td></tr><tr><td rowspan="3">10%</td><td>Scratch</td><td> $8 8 . 7 4 \pm 0 . 3 0$ </td><td> $6 5 . 5 7 \pm 1 . 4 1$ </td><td> $5 8 . 4 4 \pm 2 . 5 2$ </td><td> $5 9 . 7 3 \pm 1 . 6 4$ </td></tr><tr><td>SSL-LoRA</td><td> $9 2 . 9 6 \pm 0 . 0 8$ </td><td> $7 4 . 9 0 \pm 0 . 1 8$ </td><td> $6 8 . 0 0 \pm 0 . 8 4$ </td><td> $7 0 . 5 8 \pm 0 . 1 3$ </td></tr><tr><td>SSL-Full</td><td> $\mathbf { 9 3 . 5 6 \ : \pm { \ : 0 . 4 0 } }$ </td><td> ${ \bf 7 5 . 7 3 \pm 0 . 3 3 }$ </td><td> ${ \bf 7 0 . 5 3 \pm 0 . 9 2 }$ </td><td> $\mathbf { 7 1 . 6 2 \pm 0 . 3 3 }$ </td></tr><tr><td rowspan="3">20%</td><td>Scratch</td><td> $9 1 . 6 6 \pm 0 . 3 2$ </td><td> $7 0 . 9 9 \pm 0 . 4 5$ </td><td> $6 4 . 6 7 \pm 0 . 3 3$ </td><td> $6 5 . 9 9 \pm 0 . 4 7$ </td></tr><tr><td>SSL-LoRA</td><td> $\mathbf { 9 4 . 7 3 \ : \pm { \ : 0 . 4 0 } }$ </td><td> $7 8 . 9 1 \pm 0 . 0 0$ </td><td> $7 3 . 9 4 \pm 0 . 7 7$ </td><td> $7 5 . 3 0 \pm 0 . 0 2$ </td></tr><tr><td>SSL-Full</td><td> $9 4 . 5 8 \pm 0 . 3 2$ </td><td> ${ \bf 7 9 . 2 2 \pm 1 . 6 5 }$ </td><td> ${ \bf 7 4 . 7 2 \pm 1 . 5 8 }$ </td><td> ${ \bf 7 5 . 6 8 \pm 1 . 9 0 }$ </td></tr></table>

Overall, these results show that the JEPA-style pretraining is most beneficial in low-label regimes. The gains are especially clear when only a small portion of the training set is available, supporting the role of the pretrained encoder as a transferable initialization for data-eficient ECG classification.

Table 6: Low-label transfer results on Chapman-Shaoxing using diferent fractions of the labeled training data. SSL-pretrained models are adapted using LoRA and full fine-tuning. Results are reported as mean ± standard deviation over three runs, in percentage (%).
<table><tr><td>Train Fraction</td><td>Mode</td><td>AUC</td><td>Acc</td><td>Macro-F1</td><td>κ</td></tr><tr><td rowspan="3">1%</td><td>Scratch</td><td> $8 5 . 0 5 \pm 1 . 5 6$ </td><td> $6 6 . 6 7 \pm 1 . 3 4$ </td><td> $5 9 . 2 0 \pm 1 . 8 6$ </td><td> $5 0 . 0 3 \pm 2 . 0 4$ </td></tr><tr><td>SSL-LoRA</td><td> ${ \bf 9 8 . 2 7 \pm 0 . 1 7 }$ </td><td> $\mathbf { 9 0 . 8 4 \ : \pm { \ : 0 . 8 7 } }$ </td><td> ${ \bf 8 6 . 9 9 \pm 1 . 7 2 }$ </td><td> ${ \bf 8 6 . 4 9 \pm 1 . 3 3 }$ </td></tr><tr><td>SSL-Full</td><td> $9 8 . 1 5 \pm 0 . 2 9$ </td><td> $9 0 . 6 5 \pm 1 . 0 5$ </td><td> $8 6 . 6 1 \pm 1 . 4 0$ </td><td> $8 6 . 2 2 \pm 1 . 5 7$ </td></tr><tr><td rowspan="3">5%</td><td>Scratch</td><td> $9 2 . 5 1 \pm 0 . 9 5$ </td><td> $8 0 . 7 5 \pm 3 . 5 4$ </td><td> $7 6 . 0 9 \pm 4 . 0 2$ </td><td> $7 1 . 4 2 \pm 5 . 7 8$ </td></tr><tr><td>SSL-LoRA</td><td> $9 8 . 3 0 \pm 0 . 2 2$ </td><td> $9 1 . 7 3 \pm 0 . 5 1$ </td><td> $8 8 . 4 1 \pm 0 . 6 9$ </td><td> $8 7 . 9 2 \pm 0 . 7 5$ </td></tr><tr><td>SSL-Full</td><td> ${ \bf 9 8 . 3 7 \pm 0 . 1 8 }$ </td><td> ${ \bf 9 1 . 9 3 \pm 0 . 6 4 }$ </td><td> ${ \bf 8 8 . 7 8 \pm 0 . 9 6 }$ </td><td> ${ \bf 8 8 . 2 4 \pm 0 . 9 5 }$ </td></tr><tr><td rowspan="3">10%</td><td>Scratch</td><td> $9 5 . 3 4 \pm 0 . 0 4$ </td><td> $8 6 . 8 1 \pm 0 . 2 7$ </td><td> $8 2 . 4 3 \pm 1 . 3 6$ </td><td> $8 0 . 3 9 \pm 0 . 4 2$ </td></tr><tr><td>SSL-LoRA</td><td> ${ \bf 9 8 . 3 8 \pm 0 . 6 0 }$ </td><td> $9 2 . 2 8 \pm 1 . 5 7$ </td><td> $8 9 . 4 8 \pm 1 . 6 2$ </td><td> $8 8 . 7 7 \pm 2 . 2 3$ </td></tr><tr><td>SSL-Full</td><td> $9 7 . 9 8 \pm 0 . 3 3$ </td><td> ${ \bf 9 3 . 1 7 \pm 0 . 3 4 }$ </td><td> ${ \bf 9 0 . 0 0 \pm 0 . 5 3 }$ </td><td> ${ \bf 8 9 . 9 7 \pm 0 . 4 7 }$ </td></tr><tr><td rowspan="3">20%</td><td>Scratch</td><td> $9 7 . 0 5 \pm 0 . 5 4$ </td><td> $9 0 . 2 6 \pm 1 . 1 7$ </td><td> $8 7 . 1 3 \pm 1 . 0 6$ </td><td> $8 5 . 6 9 \pm 1 . 7 2$ </td></tr><tr><td>SSL-LoRA</td><td> ${ \bf 9 8 . 1 5 \pm 0 . 7 4 }$ </td><td> $9 3 . 3 6 \pm 1 . 1 5$ </td><td> $9 0 . 3 7 \pm 1 . 4 0$ </td><td> $9 0 . 2 8 \pm 1 . 6 8$ </td></tr><tr><td>SSL-Full</td><td> $9 8 . 0 7 \pm 0 . 4 3$ </td><td> $\mathbf { 9 3 . 7 5 \ : \pm { \ : 0 . 2 9 } }$ </td><td> $\mathbf { 9 0 . 7 8 \ : \pm { \ : 0 . 6 0 } }$ </td><td> $\mathbf { 9 0 . 8 5 \ : \pm { \ : 0 . 4 1 } }$ </td></tr></table>

Table 7: Low-label transfer results on PTB-XL using diferent fractions of the labeled training data. SSLpretrained models are adapted using LoRA and full fine-tuning. Results are reported as mean ± standard deviation over three runs, in percentage (%).
<table><tr><td>Train Fraction</td><td>Mode</td><td>AUC</td><td>Acc</td><td>Macro-F1</td><td>K</td></tr><tr><td rowspan="3">1%</td><td>Scratch</td><td> $7 4 . 5 1 \pm 0 . 8 1$ </td><td> $6 2 . 5 3 \pm 0 . 7 3$ </td><td> $4 3 . 0 4 \pm 1 . 2 7$ </td><td> $3 9 . 7 7 \pm 0 . 6 7$ </td></tr><tr><td>SSL-LoRA</td><td> $7 5 . 8 8 \pm 0 . 8 0$ </td><td> $6 4 . 8 3 \pm 2 . 4 3$ </td><td> $4 3 . 3 6 \pm 1 . 1 5$ </td><td> $4 1 . 4 0 \pm 2 . 4 1$ </td></tr><tr><td>SSL-Full</td><td> ${ \bf 7 6 . 5 1 \pm 1 . 5 1 }$ </td><td> ${ \bf 6 5 . 2 7 \pm 2 . 4 3 }$ </td><td> ${ \bf 4 5 . 4 5 \pm 0 . 6 5 }$ </td><td> ${ \bf 4 3 . 2 9 \pm 1 . 8 9 }$ </td></tr><tr><td rowspan="3">5%</td><td>Scratch</td><td> $8 1 . 6 5 \pm 1 . 1 9$ </td><td> $7 0 . 0 8 \pm 1 . 9 1$ </td><td> $5 0 . 9 6 \pm 0 . 8 3$ </td><td> $5 0 . 7 4 \pm 2 . 4 9$ </td></tr><tr><td>SSL-LoRA</td><td> $8 2 . 5 1 \pm 0 . 5 7$ </td><td> ${ \bf 7 0 . 7 9 \pm 1 . 3 6 }$ </td><td> $5 2 . 1 1 \pm 1 . 4 6$ </td><td> $5 1 . 7 8 \pm 1 . 3 8$ </td></tr><tr><td>SSL-Full</td><td> ${ \bf 8 2 . 7 8 \pm 1 . 3 2 }$ </td><td> $7 0 . 4 6 \pm 1 . 3 6$ </td><td> ${ \bf 5 2 . 6 4 \pm 1 . 5 2 }$ </td><td> ${ \bf 5 2 . 0 6 \pm 1 . 0 9 }$ </td></tr><tr><td rowspan="3">10%</td><td>Scratch</td><td> $8 3 . 0 8 \pm 1 . 1 1$ </td><td> $7 0 . 3 4 \pm 1 . 8 2$ </td><td> $5 5 . 3 6 \pm 1 . 1 8$ </td><td> $5 3 . 2 8 \pm 2 . 4 8$ </td></tr><tr><td>SSL-LoRA</td><td> $8 4 . 1 1 \pm 1 . 3 4$ </td><td> $7 1 . 4 9 \pm 2 . 1 9$ </td><td> $5 3 . 8 9 \pm 1 . 4 1$ </td><td> ${ \bf 5 4 . 0 3 \pm 1 . 7 2 }$ </td></tr><tr><td>SSL-Full</td><td> ${ \bf 8 4 . 4 9 \pm 1 . 3 8 }$ </td><td> $\mathbf { 7 1 . 9 0 \pm 1 . 3 4 }$ </td><td> ${ \bf 5 5 . 4 3 \pm 1 . 1 8 }$ </td><td> $5 3 . 9 8 \pm 1 . 0 2$ </td></tr><tr><td rowspan="3">20%</td><td>Scratch</td><td> $8 5 . 8 6 \pm 1 . 0 0$ </td><td> $7 3 . 7 4 \pm 1 . 0 5$ </td><td> $5 7 . 4 3 \pm 0 . 4 7$ </td><td> ${ \bf 5 7 . 6 0 \pm 1 . 7 3 }$ </td></tr><tr><td>SSL-LoRA</td><td> $8 6 . 4 1 \pm 1 . 3 6$ </td><td> $7 3 . 0 5 \pm 1 . 8 3$ </td><td> ${ \bf 5 7 . 7 9 \pm 0 . 8 9 }$ </td><td> $5 6 . 5 4 \pm 2 . 0 3$ </td></tr><tr><td>SSL-Full</td><td> $\mathbf { 8 6 . 9 0 \ : \pm { \ : 1 . 0 2 } }$ </td><td> $\mathbf { 7 3 . 7 8 \ : \pm { \ : 0 . 4 3 } }$ </td><td> $5 7 . 1 5 \pm 0 . 8 3$ </td><td> $5 7 . 0 9 \pm 0 . 2 7$ </td></tr></table>

## 5.7. Sensitivity to Masking Ratio

We further examine the efect of the temporal masking ratio used during JEPA-style pretraining. In the main experiments, the masking ratio is fixed at 75%, which defines the default pretraining configuration used for the reported SSL transfer results. To assess the influence of corruption strength, we additionally pretrain the Small backbone using lower masking ratios of 25% and 50%. The downstream evaluation is conducted using full fine-tuning on CPSC2018, Chapman-Shaoxing, and PTB-XL, while keeping the backbone, pretraining data, downstream splits, optimizer, and adaptation protocol unchanged. To isolate the efect of the masking ratio, this analysis uses full fine-tuning for all masking configurations; LoRA-based adaptation is evaluated separately under the default 75% masking configuration in Table 4.

Table 8 reports the masking-ratio sensitivity results. The efect of the masking ratio is not monotonic across datasets or metrics. On CPSC2018, the 50% masking ratio gives the strongest decision-level performance, achieving 83.13% accuracy, 79.19% macro-F1, and 80.26% κ. The default 75% setting remains slightly higher in AUC, but the 50% setting provides better classbalanced and agreement-based performance. This suggests that an intermediate corruption level can provide a favorable balance between preserving discriminative waveform morphology and enforcing contextual latent prediction.

On Chapman-Shaoxing, the three masking ratios produce relatively close results. The 50% setting gives the highest macro-F1, reaching 94.29%, while the default 75% setting gives the highest accuracy and κ. The 25% setting is also competitive, but remains slightly below the 50% and 75% configurations in the main decision metrics. These results indicate that the proposed JEPA-style objective is stable on this rhythm-classification dataset, with only modest variation across masking ratios.

On PTB-XL, the trend difers from CPSC2018 and Chapman-Shaoxing. The 50% masking ratio obtains the highest macro-F1, whereas the 75% setting gives the best AUC and accuracy, and the 25% setting gives the highest κ. This metric-dependent behavior is consistent with the stronger class imbalance and diagnostic heterogeneity of PTB-XL. In particular, lower or intermediate masking ratios may better preserve morphology-sensitive diagnostic cues, while stronger masking can improve overall separability and accuracy.

Overall, the sensitivity analysis shows that the masking ratio influences the balance between morphology preservation and contextual prediction dificulty. The 75% configuration remains a strong and consistent default, especially in terms of AUC and overall accuracy, and is therefore retained for the main transfer experiments. At the same time, the 50% results show that an intermediate corruption level can improve macro-F1 on all three datasets, suggesting that masking strength is an important design factor for ECG latent predictive pretraining.

Table 8: Sensitivity of JEPA-style pretraining to the temporal masking ratio using the Small backbone and full fine-tuning. The 75% setting corresponds to the default configuration used in the main experiments. All values are reported in percentage (%).
<table><tr><td rowspan="2">Mask Ratio</td><td colspan="4">CPSC2018</td><td colspan="4">Chapman-Shaoxing</td><td colspan="4">PTB-XL</td></tr><tr><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td></tr><tr><td>25%</td><td>94.32</td><td>76.88</td><td>71.13</td><td>72.90</td><td>97.61</td><td>96.16</td><td>93.69</td><td>94.37</td><td>89.50</td><td>77.64</td><td>63.25</td><td>63.58</td></tr><tr><td>50%</td><td>96.30</td><td>83.13</td><td>79.19</td><td>80.26</td><td>98.57</td><td>96.16</td><td>94.29</td><td>94.35</td><td>89.42</td><td>76.73</td><td>63.69</td><td>62.73</td></tr><tr><td>75% (default)</td><td>96.67</td><td>82.97</td><td>78.58</td><td>80.03</td><td>98.14</td><td>96.27</td><td>94.10</td><td>94.55</td><td>89.92</td><td>77.82</td><td>61.95</td><td>63.08</td></tr></table>

## 5.8. Qualitative Interpretability Analysis

To complement the quantitative results, we perform a simple qualitative analysis of the Small backbone. The aim is not to explain individual decisions clinically, but to check whether the model responds to meaningful parts of the ECG waveform. We use two attribution methods. Grad-CAM++ [26] is applied to the Stage 2 representation, the last temporally localized feature map before the global self-attention stage, giving a class-specific view of the time regions that contribute to the prediction. Attention rollout [27] is then computed over the Stage 3 selfattention blocks to show how information is combined in the final stage. Since the two maps capture diferent aspects, class-specific attribution and attention-based information flow, we show them together for each example.

Figure 6 shows representative correct and incorrect predictions. In the correct PTB-XL NORM example (Fig. 6a), both maps mainly highlight regions around the repeated beats, in particular near the QRS complexes. In the NORM→MI case (Fig. 6b), the highlighted regions shift toward parts of the signal involving the ST segment and T-wave. This does not confirm that the prediction is clinically correct, but it suggests that the error is linked to waveform regions plausibly related to the confused class. A similar pattern appears in the CPSC2018 Normal→IAVB case (Fig. 6c), where the maps concentrate around beat-level structures rather than unrelated parts of the signal.

Overall, these examples suggest that the model tends to respond to structured regions of the ECG and that its errors are often linked to less regular or ambiguous segments. We treat this as qualitative support only, complementing the classification metrics rather than providing a clinical interpretation.

![](images/4f2b9bcfee41c64dfdb1063bbb19c753718a57e877e0e4287583b956455310fb.jpg)  
(a) Correct NORM prediction on PTB-XL.

![](images/7841ca56d6bf3948e4274e7316b14ff89d5a23b6c97436dd6750751f67b0f23f.jpg)  
(b) NORM misclassified as MI on PTB-XL.

![](images/2d86bf9fb1bcb32281282badffd9a32374f06387debbd2619deb44a96661a9c2.jpg)  
(c) Normal misclassified as IAVB on CPSC2018.  
Figure 6: Qualitative interpretation of the Small backbone. Panels (a) and (b) show correct and incorrect predictions on $\mathrm { P T B - X L } ,$ respectively, while panel (c) shows a misclassification on CPSC2018. In each panel, the upper strip presents the Grad-CAM++ map [26] computed from the Stage 2 representation, and the lower strip presents the attention-rollout map [27] aggregated from the Stage 3 self-attention blocks.

## 5.9. Comparison with State-of-the-Art Methods

We compare the proposed hybrid CNN–SSM–Attention model with recent 12-lead ECG classification methods. The comparison is organized into three parts. First, we evaluate the reduced-label setting on CPSC2018 and PTB-XL using only 20% of the labeled training data, which is particularly useful for assessing the transfer value of self-supervised pretraining. Second, we report full-label results on CPSC2018 and PTB-XL under the fixed-split protocol used in the main experiments. Third, we present a separate comparison on Chapman-Shaoxing. Throughout, we also include 10-fold cross-validation results for our variants as an additional robustness check. These 10-fold results are not intended to replace the fixed-split comparison with prior work, but to verify that the observed trends are not tied to a single data split.

Table 9 reports the comparison at 20% labeled training data. Two main observations can be made. First, even without self-supervised initialization, the proposed backbone provides a strong supervised baseline. The random-initialized variant improves over the random-initialized baseline of Ma et al. on both datasets and remains competitive with several self-supervised baselines. This supports the efectiveness of the proposed architecture itself. Second, JEPAstyle pretraining provides clear gains in the low-label regime, especially on CPSC2018. On this dataset, SSL-Full achieves the best accuracy, macro-F1, and κ, while SSL-LoRA obtains the best AUC. These results show that the pretrained representation transfers efectively when labeled supervision is limited.

On PTB-XL, the improvement from pretraining is more moderate. The proposed variants remain competitive with the matched baselines, but the gains are metric-dependent. SSL-LoRA gives the highest macro-F1 among the compared methods, whereas the random-initialized backbone gives the highest κ. This suggests that, for PTB-XL, the proposed backbone already provides a strong supervised representation, while JEPA-style pretraining adds a smaller and

Table 9: Comparison at 20% labeled training data on CPSC2018 and PTB-XL. All values are reported in percentage (%). Bold values indicate the best result within each dataset and metric.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Setting</td><td colspan="4">CPSC2018</td><td colspan="4">PTB-XL</td></tr><tr><td>AUC</td><td>Acc</td><td>F1</td><td>K</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td></tr><tr><td>Ma et al. (2026) Random Init.</td><td></td><td>86.95</td><td>68.03</td><td>58.21</td><td>61.53</td><td>85.76</td><td>72.37</td><td>54.31</td><td>53.50</td></tr><tr><td>Ma et al. (2026)</td><td>MoCo</td><td>80.30</td><td>51.04 </td><td>41.50</td><td>41.75</td><td>79.70</td><td>65.93</td><td>45.21</td><td>43.25</td></tr><tr><td>Ma et al. (2026)</td><td>NNCLR</td><td>86.86</td><td>64.36 </td><td>54.98</td><td>58.00</td><td>85.76</td><td>74.39</td><td>55.75</td><td>56.38</td></tr><tr><td>Ma et al. (2026)</td><td>SimCLR</td><td>89.52</td><td>68.84</td><td>60.82</td><td>63.32</td><td>86.64</td><td>73.97 </td><td>56.07</td><td>56.13</td></tr><tr><td>Ma et al. (2026)</td><td>DCCLR</td><td>88.40</td><td>67.39</td><td>59.57</td><td>61.61</td><td>87.21</td><td>74.24</td><td>56.10</td><td>56.67</td></tr><tr><td>Ma et al. (2026)</td><td>SimSiam-GL</td><td>88.41</td><td>69.06</td><td>60.63</td><td>63.53</td><td>87.44</td><td>74.04</td><td>56.99</td><td>56.06</td></tr><tr><td>Ours</td><td>Random Init.</td><td>91.66</td><td>70.99</td><td>64.67</td><td>65.99</td><td>85.86</td><td>73.74</td><td>57.43</td><td>57.60</td></tr><tr><td>Ours</td><td>SSL-LoRA</td><td>94.73</td><td>78.91</td><td>73.94</td><td>75.30</td><td>86.41</td><td>73.05</td><td>57.79</td><td>56.54</td></tr><tr><td>Ours</td><td>SSL-Full</td><td>94.58</td><td>79.22</td><td>74.72</td><td>75.68</td><td>86.90</td><td>73.78</td><td>57.15</td><td>57.09</td></tr></table>

less uniform benefit than on CPSC2018.

Table 10 reports the full-label comparison. Under the fixed-split protocol, the proposed method achieves the strongest overall performance on CPSC2018. SSL-Full obtains the best AUC, accuracy, macro-F1, and κ, showing that JEPA-style initialization remains useful even when the full labeled training set is available. The random-initialized backbone is also strong, which further supports the contribution of the proposed hybrid architecture.

The 10-fold cross-validation results provide additional evidence for the stability of these trends on CPSC2018. SSL-LoRA gives the highest accuracy, macro-F1, and κ, while SSL-Full gives the highest AUC. This indicates that both full fine-tuning and parameter-eficient adaptation can benefit from the pretrained representation, and that the advantage is not limited to one fixed split.

On PTB-XL, the comparison is more nuanced. Prior task-specific self-supervised methods achieve the highest fixed-split AUC and accuracy, while our random-initialized backbone achieves the strongest macro-F1 and κ among the fixed-split results reported in the table. In the 10- fold setting, SSL-LoRA gives the highest AUC and accuracy, and a marginally higher κ, while the random-initialized model gives the highest macro-F1. These results suggest that, on PTB-XL, JEPA-style pretraining can improve ranking-oriented or overall metrics, but it does not consistently improve class-balanced decision metrics after adaptation. This behavior is likely related to the stronger class imbalance of PTB-XL.

Table 10: Full-label results and comparison with recent ECG classification methods on CPSC2018 and PTB-XL. All values are reported in percentage (%).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Setting</td><td colspan="4">CPSC2018</td><td colspan="4">PTB-XL</td></tr><tr><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td><td>AUC</td><td>Acc</td><td>F1</td><td>κ</td></tr><tr><td>Liu et al. (2023)</td><td>Frozen</td><td>92.01</td><td>68.07</td><td>63.22</td><td></td><td>86.76</td><td>76.00</td><td>57.27</td><td></td></tr><tr><td>Shi et al. (2024)</td><td>SSL-Full</td><td>95.38</td><td>74.40</td><td></td><td></td><td>91.23</td><td>78.87</td><td>一</td><td></td></tr><tr><td>Liu et al. (2025)</td><td>SSL-Full</td><td>93.98</td><td></td><td>68.82</td><td></td><td>90.46</td><td></td><td>61.69</td><td></td></tr><tr><td>Ma et al. (2026)</td><td>Proposed</td><td>92.53</td><td>77.47</td><td>69.99</td><td></td><td>90.23</td><td>76.86</td><td>63.11</td><td></td></tr><tr><td>Ours</td><td>Random Init.</td><td>95.41</td><td>80.16</td><td>74.56</td><td>76.76</td><td>88.91</td><td>77.52</td><td>64.75</td><td>64.13</td></tr><tr><td>Ours</td><td>SSL-LoRA</td><td>95.39</td><td>82.03</td><td>77.58</td><td>78.93</td><td>90.21</td><td>77.03</td><td>61.56</td><td>63.05</td></tr><tr><td>Ours</td><td>SSL-Full</td><td>96.67</td><td>82.97</td><td>78.58</td><td>80.03</td><td>89.92</td><td>77.82</td><td>61.95</td><td>63.08</td></tr><tr><td>Ours</td><td>Random Init. / 10-CV</td><td>95.04±1.08</td><td>80.72±1.90</td><td>76.59±2.27</td><td>77.42±2.25</td><td>91.50±0.93</td><td>79.33±1.15</td><td>68.19±0.76</td><td>67.20±1.53</td></tr><tr><td>Ours</td><td>SSL-LoRA / 10-CV</td><td>96.08±1.00</td><td>82.81±1.73</td><td>78.96±2.12</td><td>79.87±2.02</td><td>91.69±0.54</td><td>79.36±0.98</td><td>67.55±1.42</td><td>67.29±1.35</td></tr><tr><td>Ours</td><td>SSL-Full / 10-CV</td><td>96.24±0.70</td><td>81.21±1.87</td><td>77.01±2.11</td><td>77.99±2.19</td><td>90.78±0.37</td><td>78.52±0.85</td><td>64.69±1.49</td><td>65.23±1.26</td></tr></table>

Table 11: Comparison on Chapman-Shaoxing under the four-class rhythm.
<table><tr><td>Method</td><td>Acc</td><td>Macro-F1</td></tr><tr><td>Zheng et al. [28] (XGBoost + demographics + handcrafted features)</td><td>97.00</td><td>96.50</td></tr><tr><td>Ours (Random Init.)</td><td>94.57±0.59</td><td>93.86±0.72</td></tr><tr><td>Ours (SSL-Full)</td><td>96.34±0.87</td><td>95.24±0.85</td></tr><tr><td>Ours (SSL-LoRA)</td><td>96.15±0.71</td><td>95.63±0.80</td></tr></table>

Overall, the results show that the proposed backbone is competitive with recent ECG classification methods across diferent supervision levels and evaluation protocols. JEPA-style pretraining provides the clearest benefit in the reduced-label setting and on CPSC2018, while its efect on PTB-XL is more dependent on the metric and adaptation strategy. This suggests that the proposed architecture can serve as a strong supervised baseline and as an efective platform for further ECG-specific self-supervised learning.

For Chapman-Shaoxing, we report a separate comparison under the four-class rhythm setting of AFIB, GSVT, SB, and SR [28]. We use 10-fold cross-validation and report the mean and standard deviation across folds. As a reference, we include the validation baseline reported by Zheng et al. [28], which uses an XGBoost classifier with age, gender, and 230 handcrafted ECG features. In contrast, our model operates directly on the raw 12-lead ECG waveform without demographic variables or engineered clinical descriptors.

Table 11 shows that the proposed model is competitive on Chapman-Shaoxing when trained directly from raw multilead ECG signals. JEPA-style pretraining improves over random initialization for both adaptation modes. Among our variants, SSL-Full achieves the highest accuracy, while SSL-LoRA achieves the highest macro-F1. Although the feature-based baseline of Zheng et al. [28] remains stronger overall, the gap is relatively small considering that their method uses demographic information and handcrafted ECG descriptors, whereas our method uses only raw ECG waveforms. These results indicate that the proposed backbone learns useful rhythm representations and benefits from self-supervised initialization on this benchmark.

## 6. Conclusion

In this work, we presented a scalable hybrid CNN–SSM–Attention framework for 12-lead ECG classification, combining convolutional tokenization for local waveform morphology, statespace mixing for eficient temporal modeling, and late self-attention for global token interaction after temporal reduction. We further introduced an ECG-oriented JEPA-style pretraining strategy that predicts latent targets from a momentum encoder instead of reconstructing raw waveforms, applying ECG-specific corruption by projecting latent masks back to the waveform domain. Experiments on CPSC2018, Chapman-Shaoxing, and PTB-XL show that the backbone is a strong supervised baseline even from random initialization and at a compact parameter budget, and that JEPA-style pretraining further improves transfer, most clearly under reducedlabel supervision, while LoRA-based adaptation reaches similar gains with far fewer trainable parameters.

Beyond these results, the framework opens several directions for future work. First, it can be pretrained on larger and more diverse ECG datasets to improve generalization. Second, the masking strategy can be refined using lead-aware or rhythm-aware corruption. Third, the same backbone can be extended to other ECG tasks, such as rhythm analysis, abnormality localization, and clinical risk prediction. Together, these directions suggest that eficient sequence modeling combined with ECG-specific predictive pretraining is a promising path toward more transferable 12-lead ECG analysis.

## Data availability

The downstream datasets used in this study are publicly available. The CODE-15%, PTB-XL, and CPSC2018 datasets can be accessed from their respective public repositories. Code and trained models will be made available upon reasonable request.

## Funding

This work was supported by the Ongoing Research Funding Program King Saud University, Riyadh, Saudi Arabia (ORF-2026-1606).

## References

[1] P. Wagner, N. Strodthof, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, T. Schaefter, PTB-XL, a large publicly available electrocardiography dataset, Scientific Data 7 (1) (2020) 154.

[2] E. A. Perez Alday, A. Gu, A. J. Shah, C. Robichaux, A. I. Wong, C. Liu, F. Liu, A. Bahrami Rad, A. Elola, S. Seyedi, Q. Li, A. Sharma, G. D. Cliford, M. A. Reyna, Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology challenge 2020, Physiological Measurement 41 (12) (2020) 124003.

[3] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, arXiv preprint arXiv:2312.00752 (2023).

[4] D. Kiyasseh, T. Zhu, D. A. Clifton, CLOCS: Contrastive learning of cardiac signals across space, time, and patients, in: Proceedings of the International Conference on Machine Learning (ICML), 2021, pp. 5606–5615.

[5] T. Mehari, N. Strodthof, Self-supervised representation learning from 12-lead ECG data, Computers in Biology and Medicine 141 (2022) 105114.

[6] W. Huang, et al., A joint cross-dimensional contrastive learning framework for 12-lead ECGs and its heterogeneous deployment on SoC, Computers in Biology and Medicine 152 (2023) 106390.

[7] A. Ran, H. Liu, Joint spatio-temporal features constrained self-supervised electrocardiogram representation learning, Biomedical Engineering Letters 14 (2) (2024) 209–220.

[8] S. Sawano, et al., Applying masked autoencoder-based self-supervised learning for highcapability vision transformers of electrocardiographies, PLOS ONE 19 (8) (2024) e0307978.

[9] K. Ma, T. Zhang, H. Zhang, W. Huang, Self-supervised contrastive learning achieves 12-lead ECG classification, Biomedical Signal Processing and Control 112 (2026) 108420.

[10] K. McKeen, S. Masood, A. Toma, B. Rubin, B. Wang, ECG-FM: an open electrocardiogram foundation model, JAMIA Open 8 (5) (2025) ooaf122.

[11] S. Kim, Learning general representation of 12-lead electrocardiogram with a jointembedding predictive architecture, arXiv preprint arXiv:2410.08559 (2024).

[12] K. Weimann, T. O. F. Conrad, Self-supervised pre-training with joint-embedding predictive architecture boosts ECG classification performance, Computers in Biology and Medicine 196 (2025) 110809.

[13] R. Zhou, J. Yao, Q. Hong, Y. Zheng, L. Zheng, DAMS-Net: Dual attention and multi-scale information fusion network for 12-lead ECG classification, Methods 220 (2023) 134–141.

[14] C. Ji, L. Wang, J. Qin, L. Liu, Y. Han, Z. Wang, MSGFormer: A multi-scale grid transformer network for 12-lead ECG arrhythmia detection, Biomedical Signal Processing and Control 87 (2024) 105499.

[15] P. Li, et al., 12-lead ECG signal classification for detecting ECG arrhythmia via an information bottleneck-based multi-scale network, Information Sciences 662 (2024) 120239.

[16] X. Bai, X. Dong, Y. Li, R. Liu, et al., A hybrid deep learning network for automatic diagnosis of cardiac arrhythmia based on 12-lead ECG, Scientific Reports 14 (2024) 24441.

[17] W. Zhang, et al., MSFT: A multi-scale feature-based transformer model for arrhythmia classification, Biomedical Signal Processing and Control 95 (2024) 106968.

[18] Y. Qiang, X. Dong, X. Liu, Y. Yang, Y. Fang, J. Dou, ECGMamba: Towards eficient ECG classification with BiSSM, arXiv preprint arXiv:2406.10098 (2024).

[19] H. Jiang, H. Mutahira, S. Wei, M. S. Muhammad, ECG-Mamba: Cardiac abnormality classification with non-uniform-mix augmentation on 12-lead ECGs, IEEE Journal of Translational Engineering in Health and Medicine 13 (2025) 461–470.

[20] W. Liu, Z. Li, H. Zhang, S. Chang, H. Wang, J. He, Q. Huang, Dense lead contrast for selfsupervised representation learning of multi-lead electrocardiograms, Information Sciences 634 (2023) 189–205.

[21] W. Liu, H. Zhang, S. Chang, H. Wang, J. He, Q. Huang, Learning representations for multi-lead electrocardiograms from morphology-rhythm contrast, IEEE Transactions on Instrumentation and Measurement 73 (2024) 2509615.

[22] W. Liu, S. Pan, S. Chang, Q. Huang, N. Jiang, Self-supervised learning for electrocardiogram classification using lead correlation and decorrelation, Applied Soft Computing 172 (2025) 112871.

[23] J. Shi, W. Liu, H. Zhang, Z. Li, S. Chang, H. Wang, J. He, Q. Huang, Universal 12-lead ECG representation for signal denoising and cardiovascular disease detection by fusing generative and contrastive learning, Biomedical Signal Processing and Control 94 (2024) 106253.

[24] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, N. Ballas, Self-supervised learning from images with a joint-embedding predictive architecture, arXiv preprint arXiv:2301.08243 (2023).

[25] A. Hatamizadeh, J. Kautz, MambaVision: A hybrid mamba-transformer vision backbone, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 25261–25270.

[26] A. Chattopadhay, A. Sarkar, P. Howlader, V. N. Balasubramanian, Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks, in: 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, 2018, pp. 839–847. doi:10.1109/WACV.2018.00097.

[27] S. Abnar, W. Zuidema, Quantifying attention flow in transformers, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 4190–4197. doi:10.18653/v1/2020.acl-main.385.

[28] J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, C. Rakovski, A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients, Scientific Data 7 (1) (2020) 48. doi:10.1038/s41597-020-0386-x.