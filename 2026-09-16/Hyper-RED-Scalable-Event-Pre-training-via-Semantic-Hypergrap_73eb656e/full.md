# Hyper-RED: Scalable Event Pre-training via Semantic Hypergraph Distillation

Meisen Wang<sup>1</sup>, Zhiqiang Tian<sup>1,2</sup>, Wei Bao<sup>3,4</sup>, Chengjie Wang<sup>5</sup>, Shaoyi Du<sup>2</sup>, Siqi Li<sup>3,4</sup>,

<sup>1</sup>School of Software Engineering, Xi’an Jiaotong University, Xi’an 710049, China <sup>2</sup>National Key Laboratory of Human-Machine Hybrid Augmented Intelligence, National Engineering Research Center for Visual Information and Applications, and Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an 710049, China   
<sup>3</sup>BNRist, THUIBCS, BLBCI, School of Software, Tsinghua University, Beijing 100084, China <sup>4</sup>Yangtze Delta Region Institute, Tsinghua University, Jiaxing 314006, China   
<sup>5</sup>College of Grassland Science, Inner Mongolia Agricultural University, Hohhot 010018, China 3125158005@stu.xjtu.edu.cn, denghao293@stu.xjtu.edu.cn zhiqiangtian@xjtu.edu.cn, dushaoyi@xjtu.edu.cn, nmgcjwang3@imau.edu.cn {baoweivvv,lisiqi19971013,kevin.gaoy}@gmail.com

## Abstract

Event cameras have shown great potential for robust visual perception, yet scaling event representation learning remains challenging due to the scarcity of large-scale annotated event data. Pretrained image models provide scalable semantic supervision, but existing image-to-event methods rely on rigid pixel-wise or token-wise alignment that overlooks modality discrepancies in texture, density, and appearance, potentially causing semantic collapse and limiting transferability. To ad dress this issue, we propose Hyper-RED, a simple, painless, and scalable image-to-event pretraining framework that transfers high-order semantic structures from images to events. Hyper-RED uses hypergraphs to model and align high-order semantic associations among multiple image and event tokens, enabling cross-modal knowledge transfer while accommodating modality-specific diferences rather than enforcing rigid one-to-one correspondence. Specifically, given a paired event–image sample, Hyper-RED leverages DINOv3 to extract spatial token representations and constructs image, event, and cross-modal semantic hypergraphs, where each hyperedge connects multiple semantically correlated tokens. We further introduce a hypergraph relational distillation loss that imposes complementary intra- and cross-modal constraints, enabling the event encoder to inherit image-derived semantic organization while preserving local relational consistency and eventspecific characteristics. Experiments on three tasks across five event datasets demonstrate consistent scaling from ViT-S to ViT-L and state-of-the-art performance (Fig. 1). The code is available at: https://github.com/meisenwang/Hyper--RED.

## 1 Introduction

Event cameras asynchronously record per-pixel brightness changes with high temporal resolution, high dynamic range, low latency, and low power consumption, making them well suited to fast motion and extreme illumination (Gallego et al. 2020). These advantages have motivated event-based representation learning for a wide range of visual perception tasks, including recognition (Yang et al. 2025), semantic segmentation (Kong et al. 2024; Jing et al. 2024), depth estimation (Zhu et al. 2023; Bartolomei et al. 2025), and object detection (Peng et al. 2024; Chen et al. 2025). However, learning transferable event representations at scale remains challenging because event streams are sparse, asynchronous, and lack rich texture cues, while large-scale annotated event datasets remain far less abundant than image datasets. In contrast to event-based vision, the image domain has benefited from powerful pretrained models and visual foundation models (Caron et al. 2021; Oquab et al. 2023; Siméoni et al. 2025; Radford et al. 2021; He et al. 2022), which provide strong semantic representations across diverse downstream tasks. Transferring these semantic priors to event encoders therefore provides a scalable alternative to annotation-intensive event pre-training.

![](images/80a148f14334b5574c4f6b7baf679f760976f4763c60207e9525d6a86f5846b0.jpg)  
Figure 1: Overall comparison across ten metrics on various datasets. With the same backbone model, our method demonstrates superior and consistent performance across all tasks.

![](images/e28940b3718ea879319e3edf6ecf4227bfbf86773c6b1ac70ce2788457a595af.jpg)  
Figure 2: Representation analysis under rigid image–event alignment. We train an event encoder through rigid alignment with a frozen image encoder, and compare their similarity heatmaps on paired inputs. PCA and t-SNE further visualize the feature distributions of the two encoders.

Recent event pretraining methods have alleviated the scarcity of event annotations by leveraging pretrained image models for event representation learning (Yang, Pan, and Liu 2023, 2024; Liang et al. 2025; Wu et al. 2025; Cao et al. 2026). Although these methods have achieved promising performance, existing distillation paradigms primarily rely on embedding-level, pixel-wise, or token-wise correspondence between paired event and image inputs. Such a design implicitly assumes that spatially corresponding features should encode consistent semantics across modalities. However, this assumption becomes fragile under the inherent image–event modality gap, where image representations are strongly shaped by dense appearance cues, while event representations are derived from sparse motion- and contrastsensitive responses. As shown in Fig. 2, compared with the compact similarity patterns of the image encoder, those of the rigidly aligned event encoder become noticeably more diffuse, accompanied by reduced class separability. This observation reveals that rigid alignment fails to preserve semantic organization across the modality gap, posing a key bottleneck to scalable image-to-event pre-training. ScaleEvent (Chen et al. 2026) recognizes this limitation and alleviates rigid point-wise matching through graph-based structural alignment. However, its pairwise formulation decomposes the multi-token associations underlying high-order semantics into isolated relations, potentially causing semantic information loss during cross-modal transfer. Motivated by this observation, we rethink image-to-event distillation from the perspective of high-order semantic relation alignment, aiming to transfer modality-invariant semantic structures beyond isolated spatial correspondence and pairwise graph relations.

To address this limitation, we propose Hyper-RED, a simple, painless and scalable image-to-event pretraining distillation paradigm that transfers high-order semantic structures across modalities. By exploiting the multi-node relational capacity of hypergraphs, Hyper-RED captures shared semantics and aligns high-order semantic associations among multiple image and event tokens, thereby narrowing the crossmodal representation gap without enforcing rigid one-to-one correspondence. Given paired event–image samples, Hyper-RED extracts spatial token representations using a frozen DI-NOv3 image teacher and a trainable event encoder, and constructs image, event, and cross-modal semantic hypergraphs. Each hyperedge groups semantically correlated tokens to represent region-level dependencies beyond isolated point-wise correspondence. Based on these hypergraphs, we introduce a hypergraph relational distillation loss with complementary intra-modal and cross-modal constraints, allowing the event encoder to inherit image-derived semantic organization while preserving event-specific characteristics. Experiments on three downstream tasks across five datasets show that Hyper-RED scales consistently from ViT-S to ViT-L and achieves state-of-the-art overall performance in object recognition, semantic segmentation, and depth estimation.

Our contributions are summarized as follows:

• We propose Hyper-RED, a scalable image-to-event pretraining framework that distills high-order semantic structures from a frozen visual foundation model through semantic hypergraphs, moving beyond point-wise matching and pairwise relations.

• We revisit semantic collapse arising from image–event mismatch under rigid alignment and introduce a hypergraph relational distillation loss that aligns hyperedge memberships and prototypes through intra- and crossmodal constraints, thereby enabling reliable semantic transfer while preserving event-specific characteristics.

• We demonstrate state-of-the-art overall performance on five benchmarks across three tasks, with consistent scaling from ViT-S to ViT-L and strong transferability under linear probing, few-shot fine-tuning, and full supervision.

## 2 Related Work

## 2.1 Event-Based Representation Pre-training

Event-Only Self-Supervision. The sparsity and asynchronous nature of event streams, together with the scarcity of large-scale annotations, have motivated self-supervised pre-training directly on unlabeled event data. Masked Event Modeling (Klenk et al. 2024) learns transferable representations by reconstructing masked event inputs, whereas TESPEC (Mohammadi, Wu, and Gilitschenski 2025) extends this paradigm to long event sequences and recurrent encoders. By deriving supervisory signals entirely from event streams, these methods eliminate the need for paired event– image data or manual annotations during pre-training.

Cross-Modal Knowledge Transfer. Another line of research transfers semantic priors from large-scale pretrained image models to event encoders. An early representative study distilled multi-level image features into event networks for recognition and optical flow (Deng et al. 2021). ECDP (Yang, Pan, and Liu 2023) and ECDDP (Yang, Pan, and Liu 2024) subsequently transferred image knowledge to event encoders for generic representation learning and dense prediction, respectively. Subsequent studies extended this paradigm through prompt fusion (Liang et al. 2025), crossmodal masked modeling (Wu et al. 2025), and generative pre-training (Cao et al. 2026). The Cross-Modal Event Encoder (Jeong et al. 2026) further transfers image–text knowledge to event streams while retaining text-aligned zero-shot capabilities. ScaleEvent (Chen et al. 2026) advances beyond direct feature matching by introducing graph-based alignment of pairwise token relations.

Remark. Event-only self-supervision eliminates the need for paired event–image data, but its efectiveness depends heavily on the scale and diversity ofevent data and on carefully designed pretext tasks. Cross-modal knowledge transfer instead allows event encoders to inherit rich semantic priors from visual foundation models, reducing the burden of discovering high-level semantics solely from sparse event streams. Nevertheless, existing image-to-event distillation methods primarily rely on point-wise feature imitation, task-specific proxy supervision, or pairwise relational constraints. Such objectives are sensitive to the image–event modality gap and provide insuficient supervision for preserving region-level semantic organization. Scalable image-to-event pre-training therefore requires robust, task-agnostic alignment beyond isolated point-wise and pairwise relations.

## 2.2 Hypergraph Representation Learning

Hypergraphs generalize ordinary graphs by allowing a single hyperedge to connect multiple nodes, thereby representing group-wise dependencies beyond pairwise edges (Gao et al. 2022, 2020). HGNN (Feng et al. 2019) introduced hypergraph convolution to encode complex data correlations during representation learning. DHGNN (Jiang et al. 2019) subsequently learned and dynamically updated hypergraph structures from evolving features, while UniGNN (Huang and Yang 2021) formulated a unified message-passing framework for graph and hypergraph neural networks. Beyond architectural design, hypergraphs have also been incorporated into learning objectives. In visual metric learning, HIST (Lim et al. 2022) employs semantic tuplets to capture multilatera sample-to-class relations that cannot be represented by independent pairwise constraints. Collectively, these studies establish hypergraphs as an efective mechanism for modeling high-order relations among multiple entities.

Remark. Lessons from existing studies demonstrate that hypergraphs are efective at modeling high-order dependencies among multiple entities. However, they have primarily been used as prediction backbones or task-specific learning objectives within a single modality. Consequently, their potential as structured knowledge carriers between heterogeneous visual modalities remains underexplored. This gap motivates us to investigate semantic hypergraphs as structured carriers for image-to-event relational distillation.

## 3 Preliminary

Synchronized Event–Image Data. Let E denote an event representation constructed from an event stream and I its spatially and temporally aligned image. Although they describe the same scene, the two modalities exhibit substantially different sensing characteristics: images contain dense appearance and texture cues, whereas event data capture sparse brightness changes caused by motion or illumination variation. This modality gap makes isolated one-to-one token correspondence insuficient for transferring the semantic organization learned by an image foundation model.

Cross-Modal Distillation. Given an aligned pair (E, I), a trainable event encoder $F _ { \theta _ { e } }$ and a frozen image teacher $G _ { \theta _ { i } }$ extract spatial token features

$$
{ \bf K } = F _ { \boldsymbol \theta _ { e } } ( { \bf E } ) , \qquad { \bf Q } = G _ { \boldsymbol \theta _ { i } } ( { \bf I } ) ,\tag{1}
$$

where ${ \bf K } , { \bf Q } \in \mathbb { R } ^ { N \times D }$ contain N spatial tokens of dimension D. The two encoders share the same spatial token layout, enabling the teacher to provide aligned supervision. Conventional distillation directly aligns spatially corresponding tokens in K and Q. Beyond this one-to-one alignment, Hyper-RED transfers the semantic organization among multiple tokens, allowing the event encoder to inherit region-level structure from the teacher.

Hypergraph Representation. A hypergraph $\begin{array} { r l } { \mathcal { G } } & { { } = } \end{array}$ $( \bar { \mathcal { V } } , \bar { \mathcal { E } } _ { h } , \mathbf { \bar { H } } )$ comprises a node set V, a hyperedge set $\mathcal { E } _ { h } .$ , and an incidence matrix H. Unlike an ordinary graph edge that relates two nodes, a hyperedge jointly connects multiple nodes and captures group-wise dependencies. In our formulation, each node corresponds to a spatial token, each hyperedge represents a semantic group, and the incidence weights quantify the relative contributions of its connected tokens.

## 4 Methodology

Our goal is to learn fine-grained event representations by transferring high-order semantic structures from a frozen pretrained DINOv3 (Siméoni et al. 2025) image teacher using synchronized event–image data without manual annotations. To this end, we propose Hyper-RED, a teacher-guided semantic hypergraph distillation framework. As illustrated in Fig. 3, Hyper-RED organizes semantically correlated tokens into teacher-defined hyperedges and transfers their groupwise organization to the event encoder. It introduces two complementary objectives: a high-order intra-modal structure loss that aligns image and event token organization, and a high-order cross-modal structure loss that establishes correspondence between image semantic anchors and event-token groups. Because the construction operates on normalized token afinities, the formulation is agnostic to feature dimension and applies uniformly to ViT-S/B/L encoders without scale-specific modifications.

## 4.1 Semantic Hypergraph Construction

Teacher-Guided Semantic Grouping. We first $\ell _ { 2 ^ { - } }$ normalize each token in Q and K along the feature dimension and reuse the same notation for the normalized matrices. We then construct image–image, event–event, and image–event afinity matrices as

$$
\mathbf { S } ^ { \mathrm { I } } = [ \mathbf { Q Q } ^ { \top } ] _ { + } , \quad \mathbf { S } ^ { \mathrm { E } } = [ \mathbf { K K } ^ { \top } ] _ { + } , \quad \mathbf { S } ^ { \mathrm { X } } = [ \mathbf { Q K } ^ { \top } ] _ { + } ,\tag{2}
$$

where $[ \cdot ] _ { + }$ clips negative similarities to zero. The teacherderived matrix $\mathbf { \dot { S } } ^ { \mathrm { I } }$ provides a stable structural reference, while $\mathbf { S } ^ { \mathrm { E } }$ and $\mathbf { S } ^ { \mathrm { X } }$ characterize the corresponding event–event and image–event relations, respectively. In particular, the $a _ { e } \mathrm { - t h }$ row of $\mathbf { S } ^ { \mathrm { X } }$ measures the associations between image anchor ${ \bf q } _ { a , \epsilon }$ and all event tokens, allowing the teacher anchor to directly query the event representation.

![](images/81bd6c05f806413f6fbcef094c4de4d1bda4f240a1f4dea68e205d96a395f1b2.jpg)  
Figure 3: Overview of Hyper-RED. Hyper-RED mitigates semantic collapse from rigid one-to-one image–event alignment by constructing teacher-guided semantic hypergraphs and distilling high-order intra- and cross-modal relations, yielding transferable, modality-invariant event representations across encoder scales.

We rank image tokens by their total afinities to all other tokens and select the M highest-scoring ones as semantic anchors, whose indices form $\mathcal { A } = \{ a _ { e } \} _ { e = 1 } ^ { M }$ . For each anchor $^ { a _ { e } , }$ its k most correlated image token indices form the member set

$$
\mathcal { V } _ { e } = \mathrm { T o p K I n d i c e s } _ { k } \left( \left\{ S _ { a _ { e } j } ^ { \mathrm { I } } \right\} _ { j = 1 } ^ { N } \right) .\tag{3}
$$

Each hyperedge thus groups an anchor with multiple tokens from a coherent teacher-defined semantic region. Because the two encoders share the same spatial token layout, the teacher-selected anchor and member indices define a common topology across all branches. This design ensures that the losses compare branch-specific relation strengths within identical semantic groups rather than mismatched neighborhoods.

Soft Hyperedge Representation. Although the three branches share a topology, their member contributions depend on branch-specific afinities. For branch $r \in \{ \mathrm { I } , \mathrm { E } , \mathrm { X } \}$ the soft membership of token i in hyperedge e is

$$
H _ { i e } ^ { r } = \frac { \exp ( S _ { a _ { e } i } ^ { r } / \tau ) } { \sum _ { j \in \mathcal { V } _ { e } } \exp ( S _ { a _ { e } j } ^ { r } / \tau ) } , \qquad i \in \mathcal { V } _ { e } ,\tag{4}
$$

where $\tau \ = \ 0 . 0 7$ controls the distribution concentration, $H _ { i e } ^ { r } = 0 \mathrm { f o r } i \notin \mathcal { V } _ { e }$ , and $\mathbf { H } ^ { r } \in \mathbb { R } ^ { N \times M }$ . The matrices H<sup>I</sup>, $\mathbf { H } ^ { \mathrm { E } }$ , and ${ \bf H } ^ { \mathrm { X } }$ encode teacher grouping, event-token organization, and cross-modal association, respectively.

Membership distributions characterize the internal organization of a hyperedge but do not explicitly summarize its semantic content. We therefore aggregate the member tokens into a normalized prototype:

$$
\mathbf { p } _ { e } ^ { r } = \mathrm { N o r m } \left( \sum _ { i \in \mathcal { V } _ { e } } H _ { i e } ^ { r } \mathbf { u } _ { i } ^ { r } \right) ,\tag{5}
$$

where ${ \bf u } _ { i } ^ { \mathrm { I } } = { \bf q } _ { i }$ and $\mathbf { u } _ { i } ^ { \mathrm { E } } = \mathbf { u } _ { i } ^ { \mathrm { X } } = \mathbf { k } _ { i }$ . Here, $\mathbf { q } _ { i }$ and $\mathbf { k } _ { i }$ are normalized image and event tokens, respectively, and Norm denotes $\ell _ { 2 }$ normalization. The membership distribution and prototype therefore encode the internal composition and aggregated semantics of each hyperedge, respectively.

## 4.2 High-Order Structure Distillation

The image teacher defines the topology and target organization of the semantic hypergraph. We transfer this knowledge by aligning the teacher hypergraph with the event–event and image–event hypergraphs. All divergences are averaged over the M constructed hyperedges, and teacher-derived memberships and prototypes are detached during optimization.

High-Order Intra-Modal Structure Loss. The event–event branch models the internal semantic geometry of event tokens. We align its membership distribution $\mathbf { \dot { H } } ^ { \mathrm { E } }$ with the teacher distribution $\mathbf { H } ^ { \mathrm { I } }$ using KL divergence and align their hyperedge prototypes using cosine distance:

$$
\mathcal { L } _ { \mathrm { H O I } } = \mathrm { K L } ( \mathbf { H } ^ { \mathrm { I } } \| \mathbf { H } ^ { \mathrm { E } } ) + \left[ 1 - \cos ( \mathbf { P } ^ { \mathrm { I } } , \mathbf { P } ^ { \mathrm { E } } ) \right] ,\tag{6}
$$

where $\mathbf { P } ^ { r } = [ \mathbf { p } _ { 1 } ^ { r } , \hdots , \mathbf { p } _ { M } ^ { r } ] ^ { \top } \in \mathbb { R } ^ { M \times D }$ stacks the M prototypes. Here and below, the cosine term averages similarities between corresponding prototypes over all hyperedges. The membership term transfers the relative roles of tokens within each semantic group, while the prototype term preserves its group-level representation. It is termed intra-modal because the two hypergraphs are independently constructed from image–image and event–event relations before structural alignment.

High-Order Cross-Modal Structure Loss. Internal eventtoken consistency alone does not ensure alignment with the image semantic space. We therefore use the image–event branch derived from cross-modal afinities and impose

$$
\mathcal { L } _ { \mathrm { H O C } } = \mathrm { K L } ( \mathbf { H } ^ { \mathrm { I } } | | \mathbf { H } ^ { \mathrm { X } } ) + \left[ 1 - \cos ( \mathbf { P } ^ { \mathrm { I } } , \mathbf { P } ^ { \mathrm { X } } ) \right] .\tag{7}
$$

This loss transfers the teacher-defined semantic association between each image anchor and its event-token group. Consequently, ${ \cal { L } } _ { \mathrm { { H O I } } }$ preserves the organization among event tokens, whereas ${ \mathcal { L } } _ { \mathrm { H O C } }$ anchors that organization to the teacher.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">Backbone</td><td colspan="4">MVSEC-Depth</td><td colspan="4">DSEC-Depth</td></tr><tr><td>δ1 ↑ δ2 ↑  $\delta _ { 3 } \uparrow$ </td><td>AbsRel↓ RMSE↓</td><td></td><td> $\mathrm { R M S E } _ { \log } \downarrow$  δ1↑</td><td> $\delta _ { 2 } \uparrow$   $\delta _ { 3 } \uparrow$ </td><td>AbsRel↓ RMSE↓</td><td></td><td> $\mathrm { R M S E } _ { \log  } \downarrow$ </td></tr><tr><td colspan="10">RGB Initialization†</td></tr><tr><td>E2Depth</td><td>3DV&#x27;20</td><td></td><td>ResNet-18 0.432 0.717 0.868</td><td>0.420</td><td>7.268</td><td>0.455</td><td>0.409 0.719 0.891</td><td>0.395</td><td>13.258</td><td>0.412</td></tr><tr><td>EReFormer</td><td>TCSVT’24 Swin-T</td><td></td><td>0.391 0.652 0.810</td><td>0.551</td><td>8.373</td><td>0.523</td><td>0.5240.824 0.945</td><td>0.297</td><td>11.608</td><td>0.334</td></tr><tr><td colspan="10"></td></tr><tr><td>Event Pretraining† ECDP</td><td></td><td></td><td>0.476 0.772 0.863</td><td>0.496</td><td>7.680</td><td>0.506</td><td>0.528 0.818 0.938</td><td>0.324</td><td>11.473</td><td>0.376</td></tr><tr><td>ECDDP</td><td>ICCV’23 ECCV&#x27;24</td><td>ViT-S/16 ViT-S/16</td><td>0.513 0.762 0.871</td><td>0.428</td><td>6.957</td><td>0.469</td><td>0.545 0.857 0.959</td><td>0.263</td><td>9.477</td><td>0.294</td></tr><tr><td>DepthAnyEvent-R ICCV&#x27;25</td><td></td><td>ViT-S/16</td><td>0.489 0.751 0.878</td><td>0.365</td><td>6.465</td><td>0.483</td><td>0.691 0.930 0.981</td><td>0.191</td><td>8.880</td><td>0.266</td></tr><tr><td>ScaleEvent</td><td>CVPR&#x27;26</td><td>ViT-L/16</td><td>0.625 0.8340.934</td><td>0.268</td><td>5.554</td><td>0.343</td><td>0.896 0.983 0.997</td><td>0.101</td><td>3.694</td><td>0.144</td></tr><tr><td>Ours</td><td></td><td>ViT-S/16</td><td>0.776 0.940 0.982</td><td>0.165</td><td>3.596</td><td>0.214</td><td>0.838 0.965 0.992</td><td>0.123</td><td>4.513</td><td>0.167</td></tr><tr><td>Ours</td><td></td><td>ViT-B/16</td><td>0.8040.943 0.982</td><td>0.155</td><td>3.449</td><td>0.205</td><td>0.864 0.974 0.995</td><td>0.111</td><td>4.193</td><td>0.158</td></tr><tr><td>Ours</td><td></td><td>ViT-L/16</td><td>0.807 0.947 0.983</td><td>0.152</td><td>3.383</td><td>0.202</td><td>0.904 0.985 0.998</td><td>0.096</td><td>3.624</td><td>0.137</td></tr></table>

Table 1: Comparison of monocular depth estimation performance on the MVSEC-Depth and DSEC-Depth. The best and second best results are shown in bold and underline, respectively. <sup>†</sup> denotes fully supervised training.

Their combination provides complementary high-order supervision beyond isolated token-wise correspondence and pairwise relations.

Overall Objective. The complete objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \ell _ { 1 } } + \mathcal { L } _ { \mathrm { P R D } } + \lambda _ { \mathrm { H } } \left( \mathcal { L } _ { \mathrm { H O I } } + \mathcal { L } _ { \mathrm { H O C } } \right) , } \end{array}\tag{8}
$$

where $\lambda _ { \mathrm { H } } = 2 .$ , and $\mathcal { L } _ { \mathrm { P R D } }$ denotes the pairwise relational distillation loss (Chen et al. 2026).

## 5 Experiments

## 5.1 Experimental Setup

Pre-training Datasets. To learn transferable event representations, we construct a large-scale pre-training corpus containing both real and simulated event data. The real-world subset comprises DSEC (Gehrig et al. 2021), DDD17 (Li et al. 2019), MVSEC (Zhu et al. 2018), CoeSot (Tang et al. 2025), VisEvent (Wang et al. 2023), FEVD (Kim, Cho, and Yoon 2024), SEE-600K (Lu et al. 2025), and HighREV (Sun et al. 2023). The simulated subset is generated with v2e from Cityscapes (Cordts et al. 2016), KITTI (Geiger et al. 2013), DAVIS 2017 (Pont-Tuset et al. 2017), DECD (Rebecq et al. 2019), and GoPro (Nah et al. 2019). Following Ev-DTAD (Wang et al. 2026), we use HTA as the unified event representation and resize both modalities to 640 × 480.

Feature Encoders. We evaluate DINOv3 (Siméoni et al. 2025) ViT-S/B/L encoders with 16 × 16 patches. At each scale, the image teacher and event encoder share the same architecture and pretrained initialization; the teacher is frozen, while the event encoder is optimized.

Benchmark Setup. Following ScaleEvent (Chen et al. 2026), we adopt the same task-specific decoders, training pipelines, and evaluation protocols for full supervision, linear probing (LP), and few-shot fine-tuning. LP freezes the encoder and optimizes only the decoder, while few-shot evaluation uses 1%, 5%, 10%, or 20% of the annotations selected by fixed-interval sampling.

Implementation Details. We optimize the event encoder with AdamW using a learning rate o $\mathrm { 5 \times 1 0 ^ { - 6 } }$ , weight decay of $1 0 ^ { - 4 }$ , gradient clipping at 0.1, and exponential decay of

<table><tr><td>Method</td><td>Venue</td><td>Backbone</td><td colspan="2">N-Caltech101</td></tr><tr><td></td><td></td><td></td><td>acc1↑</td><td>acc5↑</td></tr><tr><td colspan="5">Training from scratch</td></tr><tr><td>ViT</td><td>ICLR&#x27;21</td><td>ViT-S/16</td><td>55.63</td><td></td></tr><tr><td colspan="5">RGB Initialization</td></tr><tr><td>BEiT</td><td>ICLR&#x27;22</td><td>ViT-B/16</td><td>53.10</td><td></td></tr><tr><td>MAE</td><td>CVPR&#x27;22</td><td>ViT-B/16</td><td>67.68</td><td></td></tr><tr><td>MoCo-v3</td><td>ICCV’21</td><td>ViT-S/16</td><td>76.59</td><td></td></tr><tr><td>DINOv2</td><td>TMLR&#x27;24</td><td>ViT-S/16</td><td>91.94</td><td>98.12</td></tr><tr><td colspan="5">Event Representation Learning 十</td></tr><tr><td>EventPillars</td><td>AAAI&#x27;25</td><td>ResNet-34</td><td>85.30</td><td></td></tr><tr><td>EVA-L</td><td>ICLR&#x27;26</td><td>EVA-L</td><td>86.30</td><td>一</td></tr><tr><td>OmniEvent</td><td>AAAI&#x27;26</td><td>ResNet-34</td><td>90.20</td><td>一</td></tr><tr><td colspan="5">Event Pretraining X</td></tr><tr><td>ECDP</td><td>ICCV’23</td><td>ViT-S/16</td><td>87.66</td><td></td></tr><tr><td>EventBind</td><td>ECCV’24</td><td>ViT-B/16</td><td>94.08</td><td></td></tr><tr><td>GEP</td><td>CVPR&#x27;26</td><td>ViT-B/16</td><td>96.47</td><td>99.56</td></tr><tr><td>Ours</td><td></td><td>ViT-S/16</td><td>94.81</td><td>98.83</td></tr><tr><td>Ours</td><td>一</td><td>ViT-B/16</td><td>97.62</td><td>99.68</td></tr><tr><td>Ours</td><td>一</td><td>ViT-L/16</td><td>98.25</td><td>99.89</td></tr></table>

Table 2: Object recognition results on N-Caltech101. Top-1 (acc1) and top-5 (acc5) accuracies are reported.

0.9. We construct 128 hyperedges, each containing an anchor and its top-32 related tokens. Training runs for 15 epochs on 4 NVIDIA RTX 3090 GPUs with 100,000 pairs per epoch. Additional details are provided in the Appendix.

## 5.2 Depth Estimation

Settings. We employ DAv2 (Yang et al. 2024) as the downstream depth decoder and initialize it with the released pre-trained weights. Following DepthAnyEvent (Bartolomei et al. 2025), we report absolute relative error (AbsRel), root mean squared error (RMSE), logarithmic RMSE $\mathrm { ( R M S E _ { l o g } ) }$ , and threshold accuracies $\delta _ { 1 } , \delta _ { 2 }$ , and $\delta _ { 3 } ,$ corresponding to 1.25, 1.25<sup>2</sup>, and $1 . 2 5 ^ { 3 }$ , respectively. Lower error and higher accuracy indicate better performance.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">MVSEC-Depth</td><td colspan="6">DSEC-Depth</td></tr><tr><td>LP</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>Full</td><td>LP</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>Full</td></tr><tr><td>DepthAnyEvent-R</td><td>ViT-S/16</td><td>7.473</td><td>7.542</td><td>7.261</td><td>6.794</td><td>6.637</td><td>6.465</td><td>10.584</td><td>10.347</td><td>9.898</td><td>9.534</td><td>9.065</td><td>8.880</td></tr><tr><td>ScaleEvent</td><td>ViT-S/16</td><td>6.756</td><td>6.930</td><td>6.712</td><td>6.477</td><td>6.352</td><td>6.145</td><td>4.861</td><td>4.983</td><td>4.751</td><td>4.728</td><td>4.694</td><td>4.564</td></tr><tr><td>Ours</td><td>ViT-S/16</td><td>4.020</td><td>4.223</td><td>3.945</td><td>3.727</td><td>3.651</td><td>3.596</td><td>4.860</td><td>4.923</td><td>4.726</td><td>4.684</td><td>4.632</td><td>4.513</td></tr></table>

Table 3: Monocular depth estimation results on MVSEC-Depth and DSEC-Depth under linear probing (LP), few-shot finetuning, and full supervision (Full). Performance is measured by RMSE, where lower values are better.

Results. Table 1 reports the monocular depth estimation results. The ViT-L/16 variant achieves the strongest overall performance on both benchmarks. On MVSEC-Depth, it improves over ScaleEvent by 0.182, 0.113, and 0.049 in $\delta _ { 1 } , \delta _ { 2 } ,$ and $\delta _ { 3 } ,$ respectively, while reducing AbsRel, RMSE, and $\mathrm { R M S E _ { l o g } }$ by 0.116, 2.171, and 0.141. These gains across both accuracy and error metrics indicate that the learned representations encode efective geometric information. On DSEC-Depth, the same model increases $\delta _ { 1 }$ from 0.896 to 0.904 and reduces AbsRel from 0.101 to 0.096 and RMSE from 3.694 to 3.624.

## 5.3 Object Recognition

Settings. Following GEP (Cao et al. 2026), we employ a multilayer perceptron (MLP) as the downstream classification head and report top-1 accuracy (acc1) and top-5 accuracy (acc5) as the evaluation metrics.

Results. Table 2 reports recognition results on N-Caltech101. With ViT-S/16, Hyper-RED achieves 94.81% top-1 accuracy, exceeding the matched-scale DINOv2 (Oquab et al. 2023) and ECDP (Yang, Pan, and Liu 2023) baselines by 2.87 and 7.15 percentage points, respectively. With ViT-B/16, it reaches 97.62% top-1 and 99.68% top-5 accuracy, surpassing GEP (Cao et al. 2026) by 1.15 and 0.12 percentage points under the same backbone scale. Increasing the encoder to ViT-L/16 raises the two scores to 98.25% and 99.89%, establishing state-of-the-art performance.

## 5.4 Semantic Segmentation

Settings. We employ EoMT (Kerssies et al. 2025) as the downstream semantic segmentation decoder and initialize it with the released pre-trained weights. For evaluation, following OpenESS (Kong et al. 2024), we report mean intersection over union (mIoU) and accuracy (Acc).

Results. Table 4 presents the semantic segmentation results. On DDD17-Seg, our ViT-L/16 model achieves 67.16% mIoU, improving over ScaleEvent by 2.08 percentage points. On DSEC-Semantic, it obtains the highest Acc of 93.86% and the second-best mIoU of 69.04%, only 0.61 percentage points below ScaleEvent. Under the same ViT-L/16 backbone, Hyper-RED improves Acc over ScaleEvent by 0.46 and 0.76 percentage points on DDD17-Seg and DSEC-Semantic, respectively. Figure 4 further presents qualitative results on DSEC, showing accurate predictions for both semantic segmentation and monocular depth estimation. Additional qualitative results and visual comparisons with prior methods are provided in the Appendix.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">Backbone</td><td colspan="2">DDD17</td><td colspan="2">DSEC</td></tr><tr><td></td><td>Acc↑ mIoU↑ Acc↑ mIoU↑</td><td></td><td></td></tr><tr><td colspan="8">RGB Initialization†</td></tr><tr><td>Ev-SegNet</td><td>CVPRW&#x27;19 Xception</td><td></td><td></td><td></td><td></td><td>89.76 54.81 88.61 51.76</td></tr><tr><td>E2VID</td><td>TPAMI&#x27;19</td><td>ResNet-18</td><td></td><td></td><td></td><td>85.84 48.47 80.0644.08</td></tr><tr><td>PVT-FPN</td><td>ICCV’21</td><td>ResNet-34</td><td>94.28</td><td>53.89</td><td></td><td></td></tr><tr><td>MaskCLIP</td><td>ECCV’22</td><td>ViT-B/16</td><td>90.50</td><td>61.27</td><td></td><td>89.81 55.01</td></tr><tr><td>ESS</td><td>ECCV’22</td><td>E2VID</td><td>88.43</td><td>53.09</td><td></td><td>84.1745.38</td></tr><tr><td>ESS-Sup</td><td>ECCV’22</td><td>E2VID</td><td>91.0861.37</td><td></td><td></td><td>89.37 53.29</td></tr><tr><td>HMNet</td><td>CVPR&#x27;23</td><td>HMNet-L1</td><td></td><td></td><td></td><td>89.8055.00</td></tr><tr><td>EvSegformer TIP&#x27;23</td><td></td><td>MiT-B1</td><td>94.7254.41</td><td></td><td></td><td></td></tr><tr><td>FC-CLIP</td><td>NeurIPS&#x27;23 CNeXt-L</td><td></td><td>90.6862.01</td><td></td><td></td><td>89.97 55.67</td></tr><tr><td>DINOv2</td><td>TMLR&#x27;24</td><td>ViT-S/16</td><td></td><td>53.85</td><td></td><td>52.17</td></tr><tr><td>HALSIE</td><td>WACV&#x27;24</td><td>SNN-ANN 92.50</td><td></td><td>60.66</td><td>89.01</td><td>52.43</td></tr><tr><td>ESEG</td><td>AAAI&#x27;25</td><td>MiT-B1</td><td>90.68</td><td>59.97</td><td>91.47</td><td>57.55</td></tr><tr><td>KWYAF</td><td>AAAI&#x27;25</td><td>MiT-B0</td><td>91.32</td><td>62.41</td><td>90.87</td><td>57.75</td></tr><tr><td colspan="7">Event Pretraining†</td></tr><tr><td>ECDP</td><td>ICCV&#x27;23</td><td>ResNet-50</td><td></td><td>59.15</td><td></td><td>59.16</td></tr><tr><td>ECDDP</td><td>ECCV’24</td><td>ViT-S/16</td><td></td><td>55.73</td><td></td><td>56.38</td></tr><tr><td>ECDDP</td><td>ECCV’24</td><td>Swin-T/7</td><td></td><td>62.56</td><td></td><td>61.25</td></tr><tr><td>OpenESS</td><td>CVPR&#x27;24</td><td>ResNet-50</td><td></td><td>57.01</td><td></td><td>55.01</td></tr><tr><td>OpenESS</td><td>CVPR&#x27;24</td><td>E2VID</td><td>91.05</td><td>63.00</td><td>90.21</td><td>57.21</td></tr><tr><td>STP</td><td>ICCV’25</td><td>ResNet-50</td><td></td><td>62.13</td><td></td><td>61.29</td></tr><tr><td>STP</td><td>ICCV’25</td><td>Swin-T/7</td><td></td><td>63.29</td><td></td><td>62.05</td></tr><tr><td>GEP</td><td>CVPR&#x27;26</td><td>ViT-B/16</td><td></td><td>61.90</td><td></td><td>67.37</td></tr><tr><td>ScaleEvent</td><td>CVPR&#x27;26</td><td>ViT-L/16</td><td>92.62</td><td>65.08</td><td>93.10</td><td>69.65</td></tr><tr><td>Ours</td><td></td><td>ViT-S/16</td><td>91.58</td><td>62.55</td><td>91.83</td><td>61.43</td></tr><tr><td>Ours</td><td></td><td>ViT-B/16</td><td>92.54</td><td>65.58</td><td>93.14</td><td>65.35</td></tr><tr><td>Ours</td><td></td><td>ViT-L/16</td><td>93.08</td><td>67.16</td><td>93.86</td><td>69.04</td></tr></table>

Table 4: Semantic segmentation results on DDD17-Seg and DSEC-Semantic, with all metrics reported in percentage (%).

## 5.5 Scalability Analysis

We assess Hyper-RED along two practical dimensions: model-capacity scaling and annotation-eficient transfer.

Model-Capacity Scaling. Using the same formulation and 16×16 patches, Hyper-RED improves monotonically across ViT-S/B/L on N-Caltech101 top-1 accuracy (94.81% → 97.62% → 98.25%), DDD17-Seg mIoU (62.55% → 65.58% → 67.16%), and MVSEC-Depth RMSE (3.596 → 3.449 → 3.383). These consistent gains across recognition, dense segmentation, and geometric prediction show that increased capacity translates into task-general improvements rather than benefiting a specific benchmark. Hyper-RED therefore scales to larger encoders without scale-specific modifications to its distillation formulation.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">DDD17-Seg</td><td colspan="6">DSEC-Semantic</td></tr><tr><td>LP</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>Full</td><td>LP</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>Full</td></tr><tr><td>MaskCLIP</td><td>ViT-B/16</td><td>31.91</td><td>53.91</td><td>56.27</td><td>59.32</td><td>59.97</td><td>61.27</td><td>33.08</td><td>33.89</td><td>37.03</td><td>38.83</td><td>42.40</td><td>55.01</td></tr><tr><td>FC-CLIP</td><td>ConvNeXt-L</td><td>54.07</td><td>56.38</td><td>58.50</td><td>60.05</td><td>60.85</td><td>62.01</td><td>43.00</td><td>39.12</td><td>43.71</td><td>44.09</td><td>47.77</td><td>55.67</td></tr><tr><td>OpenESS</td><td>E2VID</td><td>55.61</td><td>57.58</td><td>59.07</td><td>61.03</td><td>61.78</td><td>63.00</td><td>44.26</td><td>41.41</td><td>44.97</td><td>46.25</td><td>48.28</td><td>57.21</td></tr><tr><td>ScaleEvent</td><td>ViT-B/16</td><td>57.87</td><td>57.23</td><td>59.54</td><td>61.45</td><td>62.06</td><td>62.81</td><td>58.42</td><td>54.37</td><td>62.82</td><td>63.88</td><td>64.15</td><td>64.93</td></tr><tr><td>Ours</td><td>ViT-B/16</td><td>62.94</td><td>60.86</td><td>63.51</td><td>64.04</td><td>64.68</td><td>65.58</td><td>61.94</td><td>56.16</td><td>62.17</td><td>63.65</td><td>64.95</td><td>65.35</td></tr></table>

Table 5: Semantic segmentation results on DDD17-Seg and DSEC-Semantic under linear probing (LP), few-shot fine-tuning, and full supervision (Full). All mIoU scores are reported in percentage (%).
<table><tr><td rowspan="2">Exp.</td><td rowspan="2">Distill</td><td rowspan="2">PRD</td><td rowspan="2">HOI</td><td rowspan="2">HOC</td><td colspan="2">MVSEC-Depth</td><td colspan="2">DSEC-Depth</td><td colspan="2">DDD17-Seg</td><td colspan="2">DSEC-Semantic</td></tr><tr><td>δ1 ↑</td><td>RMSE↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>RMSE↓</td><td>Acc. ↑</td><td>mIoU↑</td><td>Acc. ↑</td><td>mIoU↑</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td></td><td>0.593</td><td>6.635</td><td>0.846</td><td>4.424</td><td>91.39</td><td>59.60</td><td>91.94</td><td>64.31</td></tr><tr><td>(b)</td><td>√</td><td></td><td></td><td></td><td>0.609</td><td>6.114</td><td>0.875</td><td>4.063</td><td>92.16</td><td>62.41</td><td>92.74</td><td>66.17</td></tr><tr><td>(c)</td><td>√</td><td>√</td><td></td><td></td><td>0.739</td><td>4.238</td><td>0.893</td><td>3.713</td><td>92.48</td><td>65.82</td><td>93.66</td><td>68.73</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td>√</td><td></td><td>0.792</td><td>3.470</td><td>0.896</td><td>3.727</td><td>92.72</td><td>66.24</td><td>93.84</td><td>68.79</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td></td><td>√</td><td>0.799</td><td>3.448</td><td>0.893</td><td>3.734</td><td>92.61</td><td>67.01</td><td>93.81</td><td>68.68</td></tr><tr><td>(f)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.807</td><td>3.383</td><td>0.904</td><td>3.624</td><td>93.08</td><td>67.16</td><td>93.86</td><td>69.04</td></tr></table>

Table 6: Component ablation using ViT-L/16 under full supervision. Distill, PRD, HOI, and HOC denote feature distillation, pairwise relational distillation, and the proposed high-order intra- and cross-modal losses, respectively. Exp. (a) uses imagedomain initialization, and Exp. (f) represents the full model.

Transferability and Annotation Eficiency. Tables 3 and 5 show that Hyper-RED outperforms ScaleEvent across all protocols on MVSEC-Depth and DDD17-Seg, reducing RMSE by 2.549–2.767 and improving mIoU by 2.59–5.07 percentage points, respectively. It also achieves the lowest DSEC-Depth RMSE throughout and ranks first under LP, 1%, 5%, 10%, 20%, and full-supervision settings, while remaining competitive otherwise. The strong LP and low-shot results indicate that the pretrained encoder already captures transferable semantic and geometric structures before extensive taskspecific adaptation. Overall, Hyper-RED consistently benefits from larger encoders while remaining efective across diverse tasks and annotation budgets.

![](images/741394efbd6043925ce089306036c490ec74375ae9bfe6b649d322b4dd4171ef.jpg)  
Figure 4: Qualitative results on semantic segmentation (top) and monocular depth estimation (bottom).

## 5.6 Ablation Studies

Table 6 evaluates each component. Feature distillation improves over image-domain initialization, while PRD further establishes a stronger pairwise-relational baseline. Building on Exp. (c), ${ \mathcal { L } } _ { \mathrm { H O I } }$ and ${ \mathcal { L } } _ { \mathrm { H O C } }$ yield task-dependent gains, reflecting their complementary roles: the former preserves event-token organization, whereas the latter anchors event groups to image semantics. Combining both in Exp. (f) achieves the best results across all metrics. Relative to Exp. (c), it improves MVSEC-Depth δ by 0.068 and reduces RMSE by 0.855, while increasing mIoU by 1.34 and 0.31 points on DDD17-Seg and DSEC-Semantic, respectively. These results validate the efectiveness and complementarity of the proposed high-order objectives. Additional ablation results and further analyses are provided in the Appendix.

## 6 Conclusion

In this work, we presented Hyper-RED, a scalable imageto-event pre-training framework that addresses semantic collapse caused by rigid point-wise alignment across the image– event modality gap. Instead of forcing isolated token correspondences, Hyper-RED organizes semantically correlated tokens into teacher-guided hypergraphs and transfers their region-level organization through complementary intra- and cross-modal high-order relational distillation. By aligning both hyperedge memberships and prototypes, it preserves relational consistency and group-level semantics while retaining event-specific characteristics. Built on real and simulated pre-training data, Hyper-RED scales consistently from ViT-S to ViT-L without scale-specific modifications. Experiments on three downstream tasks across five event-based datasets demonstrate state-of-the-art overall performance and strong transferability under linear probing, few-shot fine-tuning, and full supervision. These results establish semantic hypergraphs as structured knowledge carriers for scalable and annotation-eficient event representation learning.

## References

Bartolomei, L.; Mannocci, E.; Tosi, F.; Poggi, M.; and Mattoccia, S. 2025. Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19669–19678.

Cao, J.; Xing, J.; Messikommer, N.; and Scaramuzza, D. 2026. Generative Event Pretraining with Foundation Model Alignment. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3189–3199.

Caron, M.; Touvron, H.; Misra, I.; Jégou, H.; Mairal, J.; Bojanowski, P.; and Joulin, A. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 9650–9660.

Chen, N.; Xiao, C.; Dai, Y.; He, S.; Li, M.; and An, W. 2025. Event-based tiny object detection: A benchmark dataset and baseline. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7209–7218.

Chen, Z.; Hou, J.; Zhu, Z.; Wu, J.; and Shi, G. 2026. Scaling Dense Event-Stream Pretraining from Visual Foundation Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8011–8022.

Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3213–3223.

Deng, Y.; Chen, H.; Chen, H.; and Li, Y. 2021. Learning from images: A distillation learning framework for event cameras. IEEE Transactions on Image Processing, 30: 4919–4931.

Feng, Y.; You, H.; Zhang, Z.; Ji, R.; and Gao, Y. 2019. Hypergraph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, 3558–3565.

Gallego, G.; Delbrück, T.; Orchard, G.; Bartolozzi, C.; Taba, B.; Censi, A.; Leutenegger, S.; Davison, A. J.; Conradt, J.; Daniilidis, K.; et al. 2020. Event-based vision: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(1): 154–180.

Gao, Y.; Feng, Y.; Ji, S.; and Ji, R. 2022. HGNN+: General hypergraph neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3): 3181–3199.

Gao, Y.; Zhang, Z.; Lin, H.; Zhao, X.; Du, S.; and Zou, C. 2020. Hypergraph learning: Methods and practices. IEEE transactions on pattern analysis and machine intelligence, 44(5): 2548–2566.

Gehrig, M.; Aarents, W.; Gehrig, D.; and Scaramuzza, D. 2021. Dsec: A stereo event camera dataset for driving scenarios. IEEE Robotics and Automation Letters, 6(3): 4947– 4954.

Geiger, A.; Lenz, P.; Stiller, C.; and Urtasun, R. 2013. Vision meets robotics: The kitti dataset. The international journal ofrobotics research, 32(11): 1231–1237.

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. 2022. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 16000–16009.

Huang, J.; and Yang, J. 2021. Unignn: a unified framework for graph and hypergraph neural networks. arXiv preprint arXiv:2105.00956.

Jeong, S.; Chen, H.; Yun, S.; Cho, S.; Huang, W.; Liu, X.; and Imani, M. 2026. Cross-Modal Event Encoder: Bridging Image-Text Knowledge to Event Streams. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 3213–3222.

Jiang, J.; Wei, Y.; Feng, Y.; Cao, J.; and Gao, Y. 2019. Dynamic hypergraph neural networks. In Ijcai, 2635–2641.

Jing, L.; Ding, Y.; Gao, Y.; Wang, Z.; Yan, X.; Wang, D.; Schaefer, G.; Fang, H.; Zhao, B.; and Li, X. 2024. Hpl-ess: Hybrid pseudo-labeling for unsupervised event-based semantic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 23128– 23137.

Kerssies, T.; Cavagnero, N.; Hermans, A.; Norouzi, N.; Averta, G.; Leibe, B.; Dubbelman, G.; and De Geus, D. 2025. Your vit is secretly an image segmentation model. In Proceedings of the computer vision and pattern recognition conference, 25303–25313.

Kim, T.; Cho, H.; and Yoon, K.-J. 2024. Frequency-aware event-based video deblurring for real-world motion blur. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24966–24976.

Klenk, S.; Bonello, D.; Koestler, L.; Araslanov, N.; and Cremers, D. 2024. Masked event modeling: Self-supervised pretraining for event cameras. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 2378–2388.

Kong, L.; Liu, Y.; Ng, L. X.; Cottereau, B. R.; and Ooi, W. T. 2024. Openess: Event-based semantic scene understanding with open vocabularies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15686–15698.

Li, J.; Dong, S.; Yu, Z.; Tian, Y.; and Huang, T. 2019. Eventbased vision enhanced: A joint detection framework in autonomous driving. In 2019 ieee international conference on multimedia and expo (icme), 1396–1401. IEEE.

Liang, Q.; Li, Q.; Liu, S.; Cao, X.; Lu, J.; Yang, F.; Zhang, W.; Huang, K.; and Tian, Y. 2025. Eficient Event Camera Data Pretraining with Adaptive Prompt Fusion. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision, 8656–8667.

Lim, J.; Yun, S.; Park, S.; and Choi, J. Y. 2022. Hypergraphinduced semantic tuplet loss for deep metric learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 212–222.

Lu, Y.; Xu, X.; Lu, H.; Qian, Y.; Li, P.; Yao, H.; Yang, B.; Li, J.; Cai, Q.; Guo, W.; et al. 2025. SEE: See Everything Every Time—Adaptive Brightness Adjustment for Broad Light Range Images via Events. arXiv:2502.21120.

Mohammadi, M.; Wu, Z.; and Gilitschenski, I. 2025. TESPEC: Temporally-Enhanced Self-Supervised Pretraining for Event Cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7782–7793.

Nah, S.; Baik, S.; Hong, S.; Moon, G.; Son, S.; Timofte, R.; and Mu Lee, K. 2019. Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 0–0.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Peng, Y.; Li, H.; Zhang, Y.; Sun, X.; and Wu, F. 2024. Scene adaptive sparse transformer for event-based object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 16794–16804.

Pont-Tuset, J.; Perazzi, F.; Caelles, S.; Arbeláez, P.; Sorkine-Hornung, A.; and Van Gool, L. 2017. The 2017 DAVIS Challenge on Video Object Segmentation. arXiv:1704.00675.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Rebecq, H.; Ranftl, R.; Koltun, V.; and Scaramuzza, D. 2019. High speed and high dynamic range video with an event camera. IEEE transactions on pattern analysis and machine intelligence, 43(6): 1964–1980.

Siméoni, O.; Vo, H. V.; Seitzer, M.; Baldassarre, F.; Oquab, M.; Jose, C.; Khalidov, V.; Szafraniec, M.; Yi, S.; Ramamonjisoa, M.; et al. 2025. Dinov3. arXiv preprint arXiv:2508.10104.

Sun, L.; Sakaridis, C.; Liang, J.; Sun, P.; Cao, J.; Zhang, K.; Jiang, Q.; Wang, K.; and Van Gool, L. 2023. Event-based frame interpolation with ad-hoc deblurring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18043–18052.

Tang, C.; Wang, X.; Huang, J.; Jiang, B.; Zhu, L.; Chen, S.; Zhang, J.; Wang, Y.; and Tian, Y. 2025. Revisiting colorevent based tracking: A unified network, dataset, and metric. Pattern Recognition, 112718.

Wang, M.; Deng, H.; Bao, W.; Ma, Y.; Wang, C.; Tian, Z.; Du, S.; and Li, S. 2026. Rethinking Event-Based Object Detection through Representation-Level Temporal Aggregation and Model-Level Hypergraph Reasoning. arXiv:2605.08825.

Wang, X.; Li, J.; Zhu, L.; Zhang, Z.; Chen, Z.; Li, X.; Wang, Y.; Tian, Y.; and Wu, F. 2023. Visevent: Reliable object tracking via collaboration of frame and event flows. IEEE transactions on cybernetics, 54(3): 1997–2010.

Wu, W.; Wang, X.; Li, C.; Jiang, B.; Tang, J.; Luo, B.; and Liu, Q. 2025. CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework. In Proceedings of the 33rd ACM International Conference on Multimedia, 2159– 2168.

Yang, L.; Kang, B.; Huang, Z.; Zhao, Z.; Xu, X.; Feng, J.; and Zhao, H. 2024. Depth anything v2. Advances in Neural Information Processing Systems, 37: 21875–21911.

Yang, Y.; Pan, L.; Li, D.; and Liu, L. 2025. EZSR: Eventbased zero-shot recognition. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 4628–4638.

Yang, Y.; Pan, L.; and Liu, L. 2023. Event camera data pre-training. In Proceedings ofthe IEEE/CVF international conference on computer vision, 10699–10709.

Yang, Y.; Pan, L.; and Liu, L. 2024. Event camera data dense pre-training. In European Conference on Computer Vision, 292–310. Springer.

Zhu, A. Z.; Thakur, D.; Özaslan, T.; Pfrommer, B.; Kumar, V.; and Daniilidis, K. 2018. The multivehicle stereo event camera dataset: An event camera dataset for 3D perception. IEEE Robotics and Automation Letters, 3(3): 2032–2039.

Zhu, J.; Liu, L.; Jiang, B.; Wen, F.; Zhang, H.; Li, W.; and Liu, Y. 2023. Self-supervised event-based monocular depth estimation using cross-modal consistency. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7704–7710. IEEE.