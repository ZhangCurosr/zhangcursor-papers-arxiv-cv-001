# 3R-Adapter: Retrieval, Rewiring, and Refinement for Efficient Adaptation of 3D Reconstruction Model

Zixiang Ai<sup>1</sup> Zhenyu Cui<sup>2</sup> Xu Zou<sup>3</sup> Jiahuan Zhou<sup>1,†</sup>

<sup>1</sup>Peking University <sup>2</sup>Tsinghua University <sup>3</sup>Huazhong University of Science and Technology

## Abstract

Feed-forward 3D reconstruction models have demonstrated strong generalization under large-scale pre-training, yet they struggle in long-tail scenarios such as low-light reconstruction, joint human-scene reconstruction, and test-time adaptation. We observe relational attention structure collapse, manifested as the failure of cross-view correspondences, as a primary cause of these limitations. While adaptation offers a practical way to bridge this gap, existing parameter-efficient fine-tuning (PEFT) approaches mainly perform channel-wise feature adjustment and lack mechanisms to explicitly adapt relational structures, leaving correspondence failures unresolved. In this work, we formulate adaptation of 3D reconstruction models as a problem of correspondence recovery under domain shift and propose 3R-Adapter, a PEFT paradigm for long-tail 3D reconstruction scenarios. Our method consists of three components: Deformable Retrieval for task-adaptive relation candidate retrieval, Relational Rewiring for reconstructing cross-view relational structures, and Consistency Refinement for injecting the rewired relations into stable predictions. Extensive experiments on low-light reconstruction, joint human-scene reconstruction, and test-time adaptation demonstrate that our approach significantly improves robustness over existing PEFT methods while maintaining efficient lightweight adaptation. Our code is released at https://github.com/PKU-OV3-LAB/3R-Adapter.

## 1 Introduction

Recent feed-forward 3D reconstruction models [1, 2, 3, 4, 5] have demonstrated strong generalization ability under large-scale pre-training. These models leverage large-scale data [6, 7, 8, 9, 10, 11, 12] to implicitly learn geometric structures through attention-based representations, enabling the estimation of camera poses, intrinsics, and dense geometry from uncalibrated image collections. Despite these advances, current models exhibit clear limitations under domain shift. In scenarios such as low-light environments [13], joint human-scene reconstruction [14, 15, 16], and test-time adaptation [17, 18], data distributions deviate significantly from pre-training conditions and reliable 3D supervision is often unavailable, as depicted in Fig. 1. As a result, pre-trained models frequently fail to generalize, leading to unstable geometric predictions.

We observe that this failure is closely related to the instability of cross-view relational modeling, as shown in Fig. 2. Specifically, attention patterns that are well-structured in in-domain settings become diffuse and less informative under domain shift, making it difficult to establish reliable cross-view correspondences. This suggests that the core challenge lies not in insufficient feature representation, but in the inability to preserve and adapt relational structures across views.

Addressing this issue requires effective adaptation mechanisms that can restore stable cross-view relations. Parameter-efficient fine-tuning (PEFT) [19, 20, 21, 22, 23] provides a practical solution by enabling lightweight adaptation of large pre-trained models. However, existing PEFT methods, such as LoRA [24] and Adapter [25], primarily perform channel-wise feature modulation and lack mechanisms to explicitly adapt relational structures. Consequently, they rely on implicit adjustments of attention to capture cross-view dependencies, offering limited control over relational routing and often failing to recover stable correspondences under domain shift, demonstrated in Fig. 1.

![](images/d0d77960e3e1057458e41878f2114a5d7ed93a274e94cf8de19533a0b2eb0615.jpg)  
Figure 1: The data scaling bottleneck in 3D pre-training and our relational adaptation paradigm. (a) While foundation models thrive on massive data, they struggle to generalize to long-tail 3D domains due to the high cost of dense 3D ground truth. (b) Conventional PEFT methods (e.g., LoRA, Adapter) rely on implicit, channel-wise feature adjustment, which struggle to model the correspondence relationship in data-scarce scenarios. In contrast, our 3R-Adapter shifts the paradigm to explicit relational adaptation via Deformable Retrieval, Relational Rewiring, and Consistency Refinement, effectively establishing robust cross-view correspondences without destroying pretrained priors.

In this work, we argue that effective adaptation for feed-forward 3D reconstruction requires relational structure recovery. We therefore reformulate model adaptation as a problem of relational adaptation rather than feature reparameterization. Under this perspective, PEFT is not merely a tool for adjusting feature distributions, but a mechanism for introducing task-level structural constraints that explicitly guide cross-view correspondence modeling.

Based on this formulation, we propose 3R-Adapter, a relation-centric PEFT framework for longtail adaptation. The proposed method consists of three components. Deformable Retrieval, which performs task-adaptive relational querying by selecting appropriate anchor tokens and retrieving candidate cross-view features via deformable sampling. Relational Rewiring, which reconstructs cross-view relational structures by reorganizing retrieved candidates into consistent associations. Consistency Refinement, which injects reconstructed relations into the prediction space to enforce stable and consistent geometric outputs. Together, these modules enable efficient and explicit adaptation of relational structures while preserving the generalization capability of the pre-trained backbone.

In summary, our contributions are threefold:

• We identify relational structure collapse as a fundamental failure mode of feed-forward 3D reconstruction models under domain shift, and show that effective adaptation requires restoring cross-view relational structures rather than merely adjusting features.

• We propose 3R-Adapter, a relation-centric PEFT framework that enables explicit adaptation of cross-view associations through retrieval, rewiring, and refinement. To our knowledge, this is the first work that explores parameter-efficient adaptation specifically designed for feed-forward 3D reconstruction models.

• Extensive experiments demonstrate that, with minimal parameter overhead, 3R-Adapter significantly improves robustness in long-tail scenarios, including joint human-scene reconstruction, low-light conditions, and test-time adaptation.

![](images/3211152ca519240faffbe25516efe3240854c9e6824f2fc887d7f13f8412c88e.jpg)  
Figure 2: Relational attention structure collapse in long-tail scenarios. (a) In data-rich daytime scenes, the pre-trained 3D reconstruction model produces concentrated cross-view attention, allowing a selected query token to attend to consistent physical counterparts across reference views. (b) For selected low-light queries, the attention distribution becomes diffuse and fragmented, indicating degraded relational routing and unreliable correspondence modeling. This observation suggests that long-tail adaptation requires explicitly restoring cross-view relational structures, motivating the design of 3R-Adapter.

## 2 Observation and Insight

Feed-forward 3D reconstruction models rely on attention-based relational structures to establish crossview correspondences across input images [26]. While such structures are effectively learned from data-rich pre-training scenarios, their behavior in data-sparse long-tail scenarios remains less reliable. To better understand this failure mode, we visualize cross-view attention patterns of pre-trained reconstruction models in standard daytime scenes and challenging low-light scenes.

As shown in Fig. 2, in standard data-rich scenarios, the attention distribution of a selected query token is concentrated around consistent physical counterparts across reference views. This indicates that the pre-trained model can form stable relational routes when the input distribution is close to the pre-training regime. In contrast, for selected queries under extreme low-light conditions, the attention distribution becomes diffuse and fragmented, making it difficult to maintain reliable cross-view correspondences. We refer to the degradation of geometrically consistent cross-view associations as relational attention structure collapse. The quantitative correspondence analysis in Sec. 5.6 measures this geometric degradation and its recovery after adaptation.

This observation suggests that the degradation of feed-forward 3D reconstruction models in long-tail scenarios is not merely caused by weakened local feature representations. A key bottleneck lies in the instability of cross-view relational structures that support correspondence modeling. Therefore, effective adaptation should not only adjust channel-wise feature distributions, but also explicitly restore and reorganize cross-view relations. This motivates our 3R-Adapter, which retrieves taskrelevant relational candidates, rewires them into structured cross-view associations, and refines predictions through consistency-preserving residual updates.

## 3 Related Work

## 3.1 3D Reconstruction Model

Traditional 3D reconstruction methods like structure-from-motion (SfM) [27] and multi-view stereo (MVS) [28] rely on establishing cross-view correspondences through handcrafted or learned local features, followed by geometrically grounded optimization with strong 3D inductive priors. These pipelines remain sensitive to weak texture, limited overlap, and viewpoint changes, and are computationally costly.

Feed-forward 3D reconstruction models [1, 29, 30, 2, 3, 5, 4] provide a promising alternative by learning geometric reasoning directly from data. Pioneering work DUSt3R [1] reformulates reconstruction as dense pointmap prediction from uncalibrated image pairs. CUT3R [2] introduces a persistent-state recurrent formulation, allowing the model to continuously update a shared 3D state from streaming observations and produce metric-scale pointmaps online. VGGT [3] further scales feed-forward reconstruction to hundreds of views, directly predicting camera parameters, depth maps, point maps, and point tracks with a unified transformer architecture. More recent approaches such as $\pi ^ { 3 }$ [4] remove the reliance on a fixed reference view by adopting permutation-equivariant visual geometry learning, while Depth Anything 3 [5] investigates a minimal formulation based on a single transformer and depth-ray representation for joint any-view depth and pose estimation. These approaches demonstrate that correspondence and geometry can be implicitly learned at scale, marking a shift from explicit optimization to data-driven geometric reasoning.

Despite this progress, existing feed-forward models are primarily optimized for in-distribution performance and exhibit limited robustness under domain shift. In scenarios such as low-light conditions, the learned cross-view correspondences often become unreliable. This highlights the need for effective adaptation mechanisms that can recover stable cross-view relations under distribution shifts.

## 3.2 Parameter-efficient Fine-tuning

With the increasing scale of models, full fine-tuning has become computationally prohibitive for downstream adaptation. To address this challenge, parameter-efficient fine-tuning (PEFT) methods [19, 24, 25] have been widely explored in both vision and language domains.

Existing PEFT approaches can be broadly categorized into three families. Prompt-based methods, such as visual prompt tuning [19], introduce a small set of learnable tokens to steer pretrained models without modifying backbone parameters. Adapter-based methods [25, 31] insert lightweight bottleneck modules into intermediate layers to refine latent feature representations. Reparameterization methods, including LoRA [24] and its variants such as DoRA [32], approximate weight updates via low-rank decompositions, enabling efficient adaptation with minimal trainable parameters. Recent works have extended these paradigms to feed-forward 3D reconstruction. For example, Test3R [17] and Online3R [18] employ prompt-based adaptation with self-supervised objectives for test-time adaptation on single scenes, while Gen3R [33] introduces lightweight adapters to bridge representation gaps across models. Meanwhile, methods such as VIST3A [34] and VGGRPO [35] adopt LoRA-style reparameterization to enable efficient alignment and adaptation of large geometry models. These works demonstrate that PEFT techniques provide a practical solution for adapting feed-forward reconstruction models under limited supervision.

However, despite their effectiveness, existing methods primarily operate through feature-level modulation and rely on implicit updates of attention to capture cross-view dependencies. In multi-view 3D reconstruction, where geometric reasoning critically depends on structured cross-view relations, such implicit adaptation provides limited control over relational routing and often fails to maintain stable correspondences under domain shift. This limitation motivates our work, where we explicitly model and adapt cross-view relational structures through a relation-centric PEFT framework.

## 4 Method

In this section, we detail our method for efficiently adapting the pre-trained 3D reconstruction models, improving the correspondence capturing, thus enhancing the reconstruction quality. To explicitly guide cross-view correspondence modeling beyond feature shift in original pre-trained models, we introduce 3R-Adapter with three modules: Deformable Retrieval, Relational Rewiring, and Consistency Refinement, demonstrated in Fig. 3. Given a pre-trained model, we freeze the parameters and only update our newly inserted modules.

## 4.1 Preliminary: Feed-forward 3D Reconstruction

Given a set of input images $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { N }$ , feed-forward 3D reconstruction models first encode them into patch-level feature tokens. We denote the flattened token set across all views as ${ \mathbf X } = \{ { \mathbf x } _ { k } \} _ { k = 1 } ^ { M } \in$ $\mathbb { R } ^ { M \times C }$ , where $M = N \times H \times W$ is the total number of tokens. The core mechanism of these models is to establish cross-view geometric correspondences by learning a relational function $\scriptstyle { \mathcal { R } } _ { \theta }$ , yielding contextualized representations:

$$
\mathbf { Z } = { \mathcal { R } } _ { \theta } ( \mathbf { X } ) .\tag{1}
$$

![](images/9a7b2a6811a0852ef44b0fdaeae0096debf38620b449485ba420233ce4919592.jpg)  
Figure 3: The overall pipeline of the proposed 3R-Adapter. To enable efficient adaptation for complex long-tail scenes, our 3R-Adapter modulates the network via three cohesive modules: Deformable Retrieval extracts view-specific anchors and estimates spatial offsets alongside routing weights; Relational Rewiring establishes explicit geometric correspondences across views through deformable spatial sampling; and Consistency Refinement integrates the relational features back into the main stream via a learned gating mechanism, effectively enforcing cross-view consistency.

Recent vision foundation models typically instantiate $\scriptstyle { \mathcal { R } } _ { \theta }$ through two representative architectural paradigms:

Global Attention-based Models. Architectures such as VGGT [3] and $\pi ^ { 3 }$ [4] capture cross-view interactions via global attention. For a given token $\mathbf { x } _ { i } ,$ its relational output $\mathbf { z } _ { i }$ is computed by attending to all context tokens:

$$
\mathbf { z } _ { i } = \sum _ { j } \mathrm { S o f t m a x } ( \mathbf { Q } _ { i } \mathbf { K } _ { j } ^ { \top } ) \mathbf { V } _ { j } ,\tag{2}
$$

where $j \in \{ 1 , \dots , M \}$

Recurrent State-based Models. Alternatively, models such as CUT3R [2] maintain a shared latent state to aggregate multi-view information sequentially:

$$
\mathbf { S } _ { t } = g _ { \theta } ( \mathbf { S } _ { t - 1 } , \mathbf { X } _ { t } ) ,\tag{3}
$$

where $\mathbf { S } _ { t }$ encodes accumulated cross-view relations up to time step t. Despite their architectural differences, both paradigms rely on large-scale data to implicitly learn the relational structure $\scriptstyle { \mathcal { R } } _ { \theta }$ Under domain shift in long-tail scenarios, this learned structure often becomes unstable, leading to degraded correspondence estimation. This limitation motivates the need for explicit relational adaptation.

## 4.2 Deformable Retrieval

A straightforward way to adapt this relation function is to recompute dense interactions among all tokens across all views. However, for N views with feature resolution $H \times W$ , dense all-to-all matching introduces quadratic interactions and can also amplify noisy correlations under domain shift. To this end, we design a lightweight Deformable Retrieval module to query task-relevant relational candidates.

Firstly, we restrict the relational search space by introducing a structured sliding window over the view dimension. For a target view t, tokens from $\mathbf { X } ^ { t } = \{ \breve { \mathbf { x } } _ { u } ^ { t } \} _ { u = 1 } ^ { H W } \in \mathbb { R } ^ { H W \times C }$ serve as self-view query tokens, while tokens from neighboring views $\mathbf { X } ^ { v }$ are used as reference tokens. Specifically, we restrict reference views by a sliding window over the view dimension:

$$
\mathcal { N } ( t ) = \{ v \mid | v - t | \leq w \} ,\tag{4}
$$

where w is the window radius. This reduces the number of view-level interactions from $O ( N ^ { 2 } )$ to $O ( N w )$ and provides a coarse locality prior over the relational graph.

For each self-view query token $\mathbf { x } _ { u } ^ { t }$ with spatial position $\mathbf { p } _ { u } .$ , we first project it into a low-dimensional latent space:

$$
\widetilde { \mathbf { x } } _ { u } ^ { t } = \phi _ { s e l f } ( \mathbf { x } _ { u } ^ { t } ) \in \mathbb { R } ^ { r } ,\tag{5}
$$

where $r \ll C .$ . From this compact query representation, we predict K deformable offsets and their corresponding sampling weights:

$$
\{ \Delta \hat { \mathbf { p } } _ { u } ^ { k } \} _ { k = 1 } ^ { K } = \psi _ { \Delta } ( \tilde { \mathbf { x } } _ { u } ^ { t } ) , \quad \{ \mathbf { w } _ { u } ^ { k } \} _ { k = 1 } ^ { K } = \psi _ { w } ( \tilde { \mathbf { x } } _ { u } ^ { t } ) .\tag{6}
$$

In parallel, reference tokens are projected into the same low-rank space:

$$
\begin{array} { r } { \tilde { \mathbf { X } } ^ { v } = \phi _ { r e f } ( \mathbf { X } ^ { v } ) \in \mathbb { R } ^ { H W \times r } , \quad v \in \mathcal { N } ( t ) . } \end{array}\tag{7}
$$

Thus, Deformable Retrieval defines a sparse set of candidate relation locations $\{ \mathbf { p } _ { u } ^ { k } \}$ and weights $\{ \mathbf { w } _ { u } ^ { k } \}$ for each self-view token, while keeping both query and reference features in a lightweight latent space.

Compared with dense attention, Deformable Retrieval introduces only a small number of trainable parameters and provides a structured search space for relational adaptation, making it well-suited for parameter-efficient fine-tuning without disrupting the pre-trained backbone. The sliding window constrains the view-level relational graph, while the deformable offsets provide fine-grained spatial flexibility within selected reference views. The resulting sparse candidates are then materialized and aggregated in the following Relational Rewiring stage.

## 4.3 Relational Rewiring

Given the offsets and weights predicted by Deformable Retrieval, Relational Rewiring explicitly constructs sparse cross-view relations through deformable sampling and weighted aggregation. For each self-view query token $\mathbf { x } _ { u } ^ { t }$ with spatial position $\mathbf { p } _ { u }$ , we first obtain K candidate sampling locations:

$$
\mathbf { p } _ { u } ^ { k } = \mathbf { p } _ { u } + \Delta \mathbf { p } _ { u } ^ { k } ,\tag{8}
$$

where $\Delta \mathbf { p } _ { u } ^ { k }$ is the k-th predicted offset.

With the predicted raw offsets $\{ \Delta \hat { \mathbf { p } } _ { u } ^ { k } \} _ { k = 1 } ^ { K }$ in Equation 6, we constrain the search range by applying a bounded transformation:

$$
\Delta \mathbf { p } _ { u } ^ { k } = \boldsymbol { \rho } \cdot \operatorname { t a n h } ( \Delta \hat { \mathbf { p } } _ { u } ^ { k } ) ,\tag{9}
$$

where $\rho$ is a radius hyper-parameter controlling the maximum deformable search range. The candidate sampling location is then computed as:

$$
\mathbf { p } _ { u } ^ { k } = \mathbf { p } _ { u } + \Delta \mathbf { p } _ { u } ^ { k } .\tag{10}
$$

After that, we reshape the projected reference feature map $\tilde { \mathbf { X } } ^ { v } \in \mathbb { R } ^ { H W \times r }$ into a spatial feature grid $\tilde { \mathbf { X } } ^ { v } \in \mathbb { R } ^ { H \times W \times r }$ and view it as a continuous feature field via bilinear interpolation, where $v \in \mathcal { N } ( t )$ The candidate relational feature is retrieved by:

$$
\mathbf { f } _ { u } ^ { v , k } = B ( \tilde { \mathbf { X } } ^ { v } , \mathbf { p } _ { u } ^ { k } ) ,\tag{11}
$$

where $B ( \cdot , \cdot )$ denotes bilinear sampling on the reference feature grid. The sampled candidates are then aggregated into a rewired relational representation:

$$
\mathbf { r } _ { u } ^ { t } = \sum _ { v \in \mathcal { N } ( t ) } \sum _ { k = 1 } ^ { K } \mathbf { w } _ { u } ^ { k } \mathbf { f } _ { u } ^ { v , k } ,\tag{12}
$$

where $a _ { u } ^ { k }$ denotes the learned importance weight for the k-th candidate and $\mathbf { r } _ { u } ^ { t } \in \mathbb { R } ^ { r }$ is the relationenhanced representation for query token $\mathbf { x } _ { u } ^ { t }$

This process converts the offsets and weights predicted in Deformable Retrieval into explicit crossview associations. Instead of relying on unconstrained all-to-all attention, Relational Rewiring constructs a sparse and task-adaptive relational graph, enabling more stable correspondence modeling under domain shift.

## 4.4 Consistency Refinement

After Relational Rewiring, each self-view query token $\mathbf { x } _ { u } ^ { t }$ is associated with a rewired relational representation $\mathbf { r } _ { u } ^ { t } \in \mathbb { R } ^ { r }$ . Rather than directly modifying the pre-trained backbone features, we inject this relational information through a lightweight residual branch, which preserves the original representation while enabling task-specific adaptation.

Specifically, we fuse the self-view latent token $\tilde { \mathbf { x } } _ { u } ^ { t }$ and the rewired relational representation $\mathbf { r } _ { u } ^ { t }$ in the adapter space:

$$
\mathbf { h } _ { u } ^ { t } = \left( 1 - \alpha \right) \cdot \tilde { \mathbf { x } } _ { u } ^ { t } + \alpha \cdot \mathbf { r } _ { u } ^ { t } ,\tag{13}
$$

where α controls the strength of relational blending. The fused representation is then projected back to the original feature dimension:

$$
\Delta \mathbf { x } _ { u } ^ { t } = \phi _ { \mathrm { u p } } ( \mathbf { h } _ { u } ^ { t } ) \in \mathbb { R } ^ { C } .\tag{14}
$$

Finally, the adapter output is added to the frozen backbone feature through a residual connection:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { u } ^ { t } = \mathbf { x } _ { u } ^ { t } + \gamma \Delta \mathbf { x } _ { u } ^ { t } , } \end{array}\tag{15}
$$

where $\gamma$ denotes the scaling factor. This residual formulation constrains the adaptation to be a lightweight correction to the pre-trained representation, reducing the risk of disrupting the original geometric priors while allowing the rewired cross-view relations to refine downstream predictions.

## 5 Experiments

## 5.1 Adaptation Tasks

We evaluate 3R-Adapter on three representative adaptation settings: low-light reconstruction, joint human-scene reconstruction, and test-time adaptation. These tasks cover diverse forms of domain shift and long-tail scenarios where feed-forward 3D reconstruction models struggle to maintain stable cross-view relations.

Low-light reconstruction. Under low-light conditions, severe image degradation and distribution shifts from pre-training data lead to unreliable feature extraction. Consequently, cross-view correspondences become unstable, making accurate geometric reconstruction challenging.

Joint human-scene reconstruction. This task requires reconstructing both parametric human geometry (e.g., SMPL [36]) and background scene structure within a unified framework. The presence of non-rigid motion and cross-entity interactions introduces significant complexity, making it difficult for pre-trained models to recover consistent geometry and fine-grained human meshes.

Test-time adaptation. In this setting, models must adapt to a single scene without additional supervision, relying on self-supervised signals or internal consistency at test time. This scenario is particularly sensitive to errors in cross-view relations, as no external supervision is available to correct them.

Overall, these tasks share a common challenge: maintaining stable cross-view correspondences under domain shift, making them suitable benchmarks for evaluating relational adaptation.

## 5.2 Comparing Methods

Our 3R-Adapter is evaluated across multiple feed-forward 3D reconstruction backbones, including DUSt3R [1], CUT3R [2], and $\pi ^ { 3 }$ [4], to demonstrate its generality and backbone-agnostic design.

We compare against three categories of methods. First, we include recent feed-forward reconstruction models across all tasks. Second, we consider task-specific approaches tailored to individual settings. For low-light reconstruction, we compare with Dark3R [13], which leverages contrastive learning between low-light and normal images. For joint human-scene reconstruction, we include Josh3R [16], Human3R [14], and UniSH [15], which incorporate specialized designs for modeling human-scene interactions. For test-time adaptation, we compare with Test3R [17] and Online3R [18], which refine predictions via self-supervised objectives on single scenes. Finally, we compare with representative parameter-efficient fine-tuning (PEFT) methods from NLP [32, 37], 2D vision [31, 38], and 3D vision [21]. These baselines share the same training protocols and evaluation settings as ours, enabling a fair comparison of adaptation efficiency and effectiveness.

Table 1: Comparison on the Oxford Day&Night [39] test dataset for camera pose estimation and multi-view scene reconstruction.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publication</td><td rowspan="2">Params.</td><td colspan="3">Camera Pose Estimation</td><td colspan="4">Scene Reconstruction</td></tr><tr><td>ATE↓</td><td>RPE-T ↓</td><td>RPE-R↓</td><td>Accuracy ↓</td><td>Complete ↓</td><td>Chamfer ↓</td><td>Normal ↑</td></tr><tr><td>DUSt3R [1]</td><td>CVPR 24</td><td>0.57B</td><td>0.937</td><td>0.729</td><td>10.351</td><td>0.741</td><td>0.598</td><td>1.339</td><td>0.526</td></tr><tr><td>CUT3R [2]</td><td>CVPR 25</td><td>0.79B</td><td>0.784</td><td>0.542</td><td>9.924</td><td>0.562</td><td>0.639</td><td>1.201</td><td>0.534</td></tr><tr><td>VGGT [3]</td><td>CVPR 25</td><td>1.19B</td><td>0.401</td><td>0.324</td><td>4.236</td><td>0.356</td><td>0.585</td><td>0.941</td><td>0.522</td></tr><tr><td>Dark3R [13]</td><td>CVPR 26</td><td>0.69B</td><td>0.387</td><td>0.285</td><td>4.019</td><td>0.473</td><td>0.424</td><td>0.897</td><td>0.532</td></tr><tr><td>π³ [4]</td><td>ICLR 26</td><td>0.96B</td><td>0.345</td><td>0.261</td><td>3.657</td><td>0.190</td><td>0.746</td><td>0.936</td><td>0.518</td></tr><tr><td colspan="10">Parameter-efficient Fine-tuning on CUT3R [2]</td></tr><tr><td>DoRA [32]</td><td>ICML 24</td><td>5.6M</td><td>0.565</td><td>0.342</td><td>6.784</td><td>0.439</td><td>0.504</td><td>0.943</td><td>0.541</td></tr><tr><td>HydraLoRA [37]</td><td>NeurIPS 24</td><td>10.0M</td><td>0.612</td><td>0.457</td><td>7.719</td><td>0.514</td><td>0.578</td><td>1.092</td><td>0.536</td></tr><tr><td>GAPrompt [21]</td><td>ICML 25</td><td>5.2M</td><td>0.577</td><td>0.410</td><td>6.694</td><td>0.467</td><td>0.484</td><td>0.951</td><td>0.532</td></tr><tr><td>NightAdapter [31]</td><td>ICCV 25</td><td>9.5M</td><td>0.489</td><td>0.310</td><td>5.331</td><td>0.375</td><td>0.416</td><td>0.791</td><td>0.544</td></tr><tr><td>T-LoRA [38]</td><td>AAAI 26</td><td>7.0M</td><td>0.516</td><td>0.368</td><td>5.362</td><td>0.350</td><td>0.489</td><td>0.839</td><td>0.538</td></tr><tr><td>3R-Adapter</td><td>This Paper</td><td>2.8M</td><td>0.416</td><td>0.253</td><td>4.781</td><td>0.254</td><td>0.327</td><td>0.581</td><td>0.550</td></tr><tr><td colspan="10">Parameter-efficient Fine-tuning on π³ [4]</td></tr><tr><td>DoRA [32]</td><td>ICML 24</td><td>6.4M</td><td>0.376</td><td>0.245</td><td>3.459</td><td>0.198</td><td>0.543</td><td>0.741</td><td>0.524</td></tr><tr><td>HydraLoRA [37]</td><td>NeurIPS 24</td><td>12.0M</td><td>0.312</td><td>0.260</td><td>3.359</td><td>0.178</td><td>0.624</td><td>0.802</td><td>0.526</td></tr><tr><td>GAPrompt [21]</td><td>ICML 25</td><td>6.2M</td><td>0.396</td><td>0.291</td><td>3.852</td><td>0.215</td><td>0.589</td><td>0.804</td><td>0.518</td></tr><tr><td>NightAdapter [31]</td><td>ICCV 25</td><td>10.3M</td><td>0.297</td><td>0.225</td><td>3.341</td><td>0.182</td><td>0.560</td><td>0.742</td><td>0.531</td></tr><tr><td>T-LoRA [38]</td><td>AAAI 26</td><td>7.9M</td><td>0.326</td><td>0.269</td><td>3.340</td><td>0.192</td><td>0.613</td><td>0.805</td><td>0.528</td></tr><tr><td>3R-Adapter</td><td>This Paper</td><td>3.2M</td><td>0.258</td><td>0.199</td><td>3.199</td><td>0.152</td><td>0.481</td><td>0.633</td><td>0.538</td></tr></table>

Table 2: Scene reconstruction under test-time adaptation (TTA) on 7-Scenes [40] and NRGBD [41] datasets. Mean metrics are reported. Additional PEFT baselines are included for comparison.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publication</td><td rowspan="2">Params.</td><td colspan="4">7-Scenes [40]</td><td colspan="4">NRGBD [41]</td></tr><tr><td>Accuracy ↓</td><td>Complete ↓</td><td>Chamfer ↓</td><td>Normal ↑</td><td>Accuracy ↓</td><td>Complete ↓</td><td>Chamfer ↓</td><td>Normal ↑</td></tr><tr><td>DUSt3R [1]</td><td>CVPR 24</td><td>0.57B</td><td>0.146</td><td>0.181</td><td>0.327</td><td>0.736</td><td>0.144</td><td>0.154</td><td>0.299</td><td>0.871</td></tr><tr><td>MASt3R [29]</td><td>CVPR 24</td><td>0.63B</td><td>0.189</td><td>0.211</td><td>0.400</td><td>0.687</td><td>0.085</td><td>0.063</td><td>0.148</td><td>0.794</td></tr><tr><td>CUT3R [2]</td><td>CVPR 25</td><td>0.79B</td><td>0.126</td><td>0.154</td><td>0.280</td><td>0.727</td><td>0.099</td><td>0.076</td><td>0.175</td><td>0.837</td></tr><tr><td>Test3R [17] Online3R [18]</td><td>NeurIPS 25</td><td>0.8M</td><td>0.105</td><td>0.136</td><td>0.241</td><td>0.746</td><td>0.083</td><td>0.079</td><td>0.162</td><td>0.870</td></tr><tr><td></td><td>CVPR 26</td><td>1.2M</td><td>0.112</td><td>0.145</td><td>0.257</td><td>0.733</td><td>0.084</td><td>0.081</td><td>0.165</td><td>0.829</td></tr><tr><td colspan="9">Parameter-Efficient Fine-Tuning on DUSt3R Backbone</td><td></td><td></td></tr><tr><td>HydraLoRA [37]</td><td>NeurIPS 24</td><td>9.3M</td><td>0.114</td><td>0.148</td><td>0.262</td><td>0.730</td><td>0.095</td><td>0.183</td><td>0.318</td><td>0.869</td></tr><tr><td>GAPrompt [21]</td><td>ICML 25</td><td>4.6M</td><td>0.106</td><td>0.136</td><td>0.242</td><td>0.731</td><td>0.101</td><td>0.179</td><td>0.308</td><td>0.861</td></tr><tr><td>NightAdapter [31]</td><td>ICCV 25</td><td>8.1M</td><td>0.120</td><td>0.147</td><td>0.267</td><td>0.729</td><td>0.093</td><td>0.142</td><td>0.235</td><td>0.865</td></tr><tr><td>T-LoRA [38]</td><td>AAAI 26</td><td>6.4M</td><td>0.102</td><td>0.135</td><td>0.237</td><td>0.733</td><td>0.089</td><td>0.131</td><td>0.220</td><td>0.872</td></tr><tr><td>3R-Adapter</td><td>This Paper</td><td>2.5M</td><td>0.094</td><td>0.122</td><td>0.216</td><td>0.738</td><td>0.082</td><td>0.123</td><td>0.205</td><td>0.879</td></tr></table>

## 5.3 Analysis on Low-light Reconstruction

Table 1 reports results on the Oxford Day&Night [39] dataset. Low-light scenes severely degrade visual features and make cross-view matching unreliable, providing a challenging testbed for evaluating whether adaptation can recover stable relational structures.

3R-Adapter achieves the strongest PEFT performance on both CUT3R and $\pi ^ { 3 }$ backbones with the fewest trainable parameters. On CUT3R, it reduces Chamfer distance by about 27% compared with the strongest PEFT baseline, while using only 2.8M trainable parameters. On the stronger $\pi ^ { 3 }$ backbone, 3R-Adapter also consistently improves pose estimation and reconstruction quality, reducing Chamfer distance by about 32% over the original backbone.

These gains indicate that the improvement does not come from increasing adaptation capacity, but from the proposed relation-centric design. By retrieving sparse relational candidates and rewiring them into structured cross-view associations, 3R-Adapter more effectively restores correspondence under low-light degradation than channel-wise feature adjustment.

## 5.4 Analysis on Test-time Adaptation

Test-time adaptation evaluates whether a model can refine reconstruction for each test scene without relying on additional supervised training data. This setting is particularly sensitive to unstable crossview relations, since correspondence errors must be corrected using only scene-level self-supervised signals.

As shown in Table 2, 3R-Adapter achieves the best PEFT performance on both 7-Scenes and NRGBD. It reduces Chamfer distance by about 9% over the strongest PEFT baseline on 7-Scenes and by about 7% on NRGBD, while using only 2.5M trainable parameters. The consistent improvement across two datasets suggests that our method does not overfit to a single scene distribution.

Table 3: Comparison on the BEDLAM [42] validation dataset for joint human-scene reconstruction.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publication</td><td rowspan="2">Params.</td><td colspan="3">Scene Reconstruction</td><td colspan="4">Human Mesh Reconstruction</td></tr><tr><td>Accuracy ↓</td><td>Complete ↓</td><td>Chamfer ↓</td><td>PA-PVE ↓</td><td>Metric-PVE ↓</td><td>PA-MPJPE ↓</td><td>Metric-MPJPE ↓</td></tr><tr><td>DUST3R [1]</td><td>CVPR 24</td><td>0.57B</td><td>0.237</td><td>0.159</td><td>0.396</td><td></td><td></td><td></td><td></td></tr><tr><td>CUT3R [2]</td><td>CVPR 25</td><td>0.79B</td><td>0.185</td><td>0.096</td><td>0.281</td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [3]</td><td>CVPR 25</td><td>1.19B</td><td>0.092</td><td>0.044</td><td>0.136</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Joint Human-Scene Reconstruction</td></tr><tr><td>JOSH3R [16]</td><td>ICLR 26</td><td>0.66B</td><td>0.231</td><td>0.107</td><td>0.338</td><td>46.2</td><td>245.7</td><td>41.5</td><td>239.4</td></tr><tr><td>Human3R [14]</td><td>ICLR 26</td><td>1.12B</td><td>0.157</td><td>0.059</td><td>0.216</td><td>43.8</td><td>222.1</td><td>37.2</td><td>220.9</td></tr><tr><td>UniSH [15]</td><td>CVPR 26</td><td>0.85B</td><td>0.124</td><td>0.056</td><td>0.180</td><td>45.3</td><td>231.0</td><td>39.6</td><td>228.4</td></tr><tr><td colspan="10">Parameter-efficient Fine-tuning on CUT3R [2]</td></tr><tr><td>DoRA [32]</td><td>ICML 24</td><td>42.5M</td><td>0.184</td><td>0.086</td><td>0.270</td><td>46.1</td><td>245.1</td><td>39.1</td><td>236.9</td></tr><tr><td>HydraLoRA [37]</td><td>NeurIPS 24</td><td>48.0M</td><td>0.201</td><td>0.124</td><td>0.325</td><td>47.8</td><td>258.8</td><td>40.5</td><td>251.4</td></tr><tr><td>NightAdapter [31]</td><td>ICCV 25</td><td>46.7M</td><td>0.167</td><td>0.077</td><td>0.244</td><td>45.2</td><td>231.0</td><td>38.2</td><td>222.0</td></tr><tr><td>T-LoRA [38]</td><td>AAAI 26</td><td>40.3M</td><td>0.159</td><td>0.068</td><td>0.227</td><td>45.5</td><td>227.2</td><td>37.9</td><td>225.4</td></tr><tr><td>3R-Adapter</td><td>This Paper</td><td>36.1M</td><td>0.135</td><td>0.052</td><td>0.187</td><td>44.4</td><td>218.5</td><td>37.6</td><td>217.2</td></tr></table>

![](images/1f97a2789e1bfe071c9b2b2e50394051caadb4e8a5e7748b2a58d15fbaf65a70.jpg)  
Figure 4: Qualitative results on joint human-scene reconstruction. 3R-Adapter produces more consistent human mesh recovery and cleaner scene geometry under complex human-scene interactions.

Compared with specialized TTA methods such as Test3R [17] and Online3R [18], 3R-Adapter remains competitive while following a lightweight relation-centric adaptation design. Instead of relying solely on self-supervised refinement at test time, our method explicitly retrieves and rewires reliable cross-view relations, providing a more structured way to improve scene-level consistency.

These results further support our central claim that effective adaptation for feed-forward 3D reconstruction should target relational structures, rather than only modulating channel-wise features.

## 5.5 Analysis on Joint Human-Scene Reconstruction

Joint human-scene reconstruction requires the model to go beyond standard point-level correspondence and capture structured associations between human proxy tokens, body regions, and surrounding scene context. This makes it a particularly suitable benchmark for evaluating whether relational adaptation can transfer a generic feed-forward reconstruction backbone to more structured downstream tasks.

As shown in Table 3, 3R-Adapter achieves the strongest overall performance among PEFT baselines under the same CUT3R backbone, while using the fewest trainable parameters. This indicates that the improvement does not simply come from larger adaptation capacity, but from explicitly adapting relational structures through retrieval, rewiring, and refinement. Compared with channel-wise PEFT methods, our design is better aligned with the human-scene setting, where reconstruction depends on structured cross-entity relations rather than only local feature adjustment.

3R-Adapter also remains competitive with task-specific joint reconstruction methods, despite operating as a lightweight adaptation module for a generic reconstruction backbone. This suggests that relation-centric PEFT can effectively extend pre-trained 3D reconstruction models to structured downstream tasks without requiring a fully specialized architecture. The qualitative results in Fig. 4 further support this observation: our method produces more coherent human meshes and more consistent scene reconstruction under challenging human-scene interactions, indicating that the adapted relational structures benefit both human and scene modeling.

## 5.6 Quantitative Cross-view Correspondence Matching Accuracy

We further evaluate the cross-view correspondence matching accuracy of attended or retrieved reference locations with respect to geometry-consistent ground-truth correspondences. Using π<sup>3</sup>, we analyze 204 four-view inputs from 51 Oxford Day&Night [39] sequences, with correspondences obtained by 3D projection using ground-truth depth and camera geometry. Each $5 1 8 \times 5 1 8$ image contains 1,369 patch tokens on a $3 7 \times 3 7$ grid, using 14 × 14-pixel patches.

Table 4: Cross-view correspondence diagnostics on $O x f o r d D a y \& N i g h t$ with $\pi ^ { 3 }$ . (a) Frozen Match@1 pooled over blocks 17/19/21/23. (b) Coverage of 3R-Adapter’s deformable retrieval offsets, pooled over day/night and evaluated within one layer, across all eight adapter layers, or across valid reference views. (c) Paired nighttime Match@1 with a fixed $3 \times 3$ tolerance.
<table><tr><td colspan="3">(a) Frozen Match@1</td><td colspan="4">(b) Deformable Offset Coverage</td><td colspan="3">(c) Night Match@1</td></tr><tr><td>Window size</td><td>Day</td><td>Night</td><td>Window size</td><td>Single layer</td><td>All layers</td><td>All references</td><td>Block</td><td>Frozen</td><td>Adapted</td></tr><tr><td>1</td><td>7.56</td><td>3.69</td><td>1</td><td>3.71</td><td>24.03</td><td>31.27</td><td>19</td><td>2.86</td><td>3.85</td></tr><tr><td>3</td><td>18.49</td><td>11.14</td><td>3</td><td>28.72</td><td>82.00</td><td>86.19</td><td>21</td><td>14.38</td><td>19.21</td></tr><tr><td>5</td><td>24.38</td><td>16.64</td><td>5</td><td>62.44</td><td>91.55</td><td>93.50</td><td>23</td><td>26.47</td><td>33.19</td></tr><tr><td>7</td><td>29.60</td><td>22.22</td><td>7</td><td>83.26</td><td>94.74</td><td>96.05</td><td>25</td><td>36.02</td><td>44.41</td></tr></table>

Match@1 measures the fraction of valid query/reference pairs whose highest-attention token lies within a $w \times w$ patch window around the projected correspondence. Window size w denotes the geometric tolerance width in patch tokens; $w = 1$ requires an exact token match. These tolerances therefore retain local geometric constraints. All reported metrics are percentages and adapted Match@1 and deformable offset coverage are averaged over three training seeds.

Table 4(a) shows lower frozen Match@1 at night across all window sizes. $\mathrm { \ A t \ } w = 3$ , it falls from 18.49% by day to 11.14% at night, confirming weaker cross-view matching.

Deformable offset coverage measures whether any reference location sampled by 3R-Adapter’s learned offsets falls within the same window. Table 4(b) uses four candidates from one layer and 32 across eight layers per query/reference pair, then up to 64 across valid reference views per query. At $w = 3 ,$ coverage is 28.72%, 82.00%, and 86.19%, respectively. Pooling candidates thus provides broader geometric support for rewiring; these coverage rates reflect different budgets and counting units and do not evaluate learned ranking accuracy.

Table 4(c) further shows higher nighttime Match@1 after adaptation in all four reported blocks, using $w = 3$ . Block 21 improves from 14.38% to 19.21%, and block 25 from 36.02% to 44.41%. Block 31 changes from 52.94% to 52.16%, indicating that recovery is layer-dependent. These results provide direct evidence that 3R-Adapter recovers geometric associations at intermediate decoder layers, consistent with our relation-centric design.

## 6 Conclusion

We present 3R-Adapter, a relation-centric parameter-efficient adaptation framework for feed-forward 3D reconstruction models. Motivated by the observation that pre-trained models exhibit relational attention structure collapse in data-sparse long-tail scenarios, we reformulate adaptation as the recovery of cross-view relational structures rather than channel-wise feature adjustment. Our method retrieves task-relevant relational candidates, rewires them into structured cross-view associations, and refines predictions through consistency-preserving residual updates. Experiments on low-light reconstruction, joint human-scene reconstruction, and test-time adaptation demonstrate that 3R-Adapter improves robustness over existing PEFT methods with minimal trainable parameters. These results suggest that explicitly adapting relational structures is a promising direction for efficient and reliable adaptation of 3D reconstruction foundation models.

## Acknowledgments and Disclosure of Funding

This work is supported by the National Natural Science Foundation of China (62376011) and the National Key R&D Program of China (2024YFA1410000).

## References

[1] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20697–20709, 2024.

[2] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025.

[3] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[4] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025.

[5] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[6] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[8] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22378–22389, 2024.

[9] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2041–2050, 2018.

[10] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021.

[11] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In CVPR, 2023.

[12] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020.

[13] Andrew Y Guo, Anagh Malik, SaiKiran Tedla, Yutong Dai, Yiqian Qin, Zach Salehe, Benjamin Attal, Sotiris Nousias, Kiriakos N Kutulakos, and David B Lindell. Dark3r: Learning structure from motion in the dark. arXiv preprint arXiv:2603.05330, 2026.

[14] Yue Chen, Xingyu Chen, Yuxuan Xue, Anpei Chen, Yuliang Xiu, and Gerard Pons-Moll. Human3r: Everyone everywhere all at once. arXiv preprint arXiv:2510.06219, 2025.

[15] Mengfei Li, Peng Li, Zheng Zhang, Jiahao Lu, Chengfeng Zhao, Wei Xue, Qifeng Liu, Sida Peng, Wenxiao Zhang, Wenhan Luo, et al. Unish: Unifying scene and human reconstruction in a feed-forward pass. arXiv preprint arXiv:2601.01222, 2026.

[16] Zhizheng Liu, Joe Lin, Wayne Wu, and Bolei Zhou. Joint optimization for 4d human-scene reconstruction in the wild. arXiv preprint arXiv:2501.02158, 2025.

[17] Yuheng Yuan, Qiuhong Shen, Shizun Wang, Xingyi Yang, and Xinchao Wang. Test3r: Learning to reconstruct 3d at test time, 2025.

[18] Shunkai Zhou, Zike Yan, Fei Xue, Dong Wu, Yuchen Deng, and Hongbin Zha. Online3r: Online learning for consistent sequential reconstruction based on geometry foundation model. arXiv preprint arXiv:2604.09480, 2026.

[19] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pages 709–727. Springer, 2022.

[20] Zhengxiang Shi and Aldo Lipani. Dept: Decomposed prompt tuning for parameter-efficient fine-tuning. arXiv preprint arXiv:2309.05173, 2023.

[21] Zixiang Ai, Zichen Liu, Yuanhang Lei, Zhenyu Cui, Xu Zou, and Jiahuan Zhou. GAPrompt: Geometry-aware point cloud prompt for 3D vision model. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 796–808. PMLR, 13–19 Jul 2025.

[22] Zixiang Ai, Zichen Liu, and Jiahuan Zhou. Vision graph prompting via semantic low-rank decomposition. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 809–821. PMLR, 13–19 Jul 2025.

[23] Zixiang Ai, Zhenyu Cui, Yufei Guo, Wenwen Qiang, Lei Chen, Jiwen Lu, and Jiahuan Zhou. GAPrompt++: Multi-granular geometry-aware point cloud prompt for 3D vision model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[25] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019.

[26] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, and Jérôme Revaud. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. Advances in Neural Information Processing Systems, 35:3502–3516, 2022.

[27] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.

[28] Michael Goesele, Brian Curless, and Steven M Seitz. Multi-view stereo revisited. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 2, pages 2402–2409. IEEE, 2006.

[29] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European conference on computer vision, pages 71–91. Springer, 2024.

[30] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In 2025 International Conference on 3D Vision (3DV), pages 78–89. IEEE, 2025.

[31] Qi Bi, Jingjun Yi, Huimin Huang, Hao Zheng, Haolan Zhan, Yawen Huang, Yuexiang Li, Xian Wu, and Yefeng Zheng. Nightadapter: Learning a frequency adapter for generalizable night-time scene segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23838–23849, 2025.

[32] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning, 2024.

[33] Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, and Yiyi Liao. Gen3r: 3d scene generation meets feed-forward reconstruction. arXiv preprint arXiv:2601.04090, 2026.

[34] Hyojun Go, Dominik Narnhofer, Goutam Bhat, Prune Truong, Federico Tombari, and Konrad Schindler. Vist3a: Text-to-3d by stitching a multi-view reconstruction network to a video generator. arXiv preprint arXiv:2510.13454, 2025.

[35] Zhaochong An, Orest Kupyn, Théo Uscidda, Andrea Colaco, Karan Ahuja, Serge Belongie, Mar Gonzalez-Franco, and Marta Tintore Gazulla. Vggrpo: Towards world-consistent video generation with 4d latent reward. arXiv preprint arXiv:2603.26599, 2026.

[36] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, October 2015.

[37] Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, and Chengzhong Xu. Hydralora: An asymmetric lora architecture for efficient fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[38] Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, and Konstantin Sobolev. T-lora: Single image diffusion model customization without overfitting. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 9051–9059, 2026.

[39] Zirui Wang, Wenjing Bian, Xinghui Li, Yifu Tao, Jianeng Wang, Maurice Fallon, and Victor Adrian Prisacariu. Seeing in the dark: Benchmarking egocentric 3d vision with the oxford day-and-night dataset. arXiv preprint arXiv:2506.04224, 2025.

[40] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2930–2937, 2013.

[41] Dejan Azinovic, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies.´ Neural rgb-d surface reconstruction. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6290–6301, 2022.

[42] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8726–8737, 2023.

[43] Fabien Baradel, Matthieu Armando, Salma Galaaoui, Romain Brégier, Philippe Weinzaepfel, Grégory Rogez, and Thomas Lucas. Multi-hmr: Multi-person whole-body human mesh recovery in a single shot. In European Conference on Computer Vision, pages 202–218. Springer, 2024.

## A Technical appendices and supplementary material

## A.1 Analysis and Discussion

Dense attention offers a flexible way to learn cross-view relations during large-scale pre-training, but it also introduces an unconstrained all-to-all relational search space. In data-sparse long-tail scenarios, directly adapting such dense relations is inefficient and may amplify noisy correspondences. Our Deformable Retrieval can be interpreted as a structured sparse approximation to dense cross-view attention. The sliding window mask provides a weak view-locality prior, while learnable offsets retain spatial flexibility for task-adaptive correspondence search. This design reduces the difficulty of relational discovery without imposing hard geometric constraints.

This perspective also distinguishes 3R-Adapter from conventional PEFT methods. Rather than using trainable parameters only for channel-wise feature adjustment, our method uses them to adapt how cross-view relations are retrieved, rewired, and injected into predictions. Thus, the parameter efficiency of 3R-Adapter comes not from a larger adaptation capacity, but from placing lightweight trainable modules at the relational bottleneck of feed-forward 3D reconstruction.

## A.2 Limitation

Although 3R-Adapter effectively alleviates the degradation of feed-forward reconstruction models in data-sparse long-tail scenarios, it still relies on the representational and geometric priors learned by the pre-trained backbone. Our method adapts and reorganizes existing cross-view relational structures rather than learning reconstruction ability from scratch. Thus, when the backbone fails to extract meaningful visual features or the input distribution lies far outside the pre-training regime, relational retrieval and rewiring may remain limited. In addition, the current sliding-window design improves efficiency but may restrict long-range relation modeling in extremely long sequences or scenes with large viewpoint changes. Future work may explore adaptive view selection and hierarchical relational memory for more scalable adaptation.

## A.3 Details of Datasets and Metrics

Due to the scarcity of high-quality annotations in long-tail scenarios, we evaluate our method on representative benchmarks that reflect diverse forms of domain shift. We report task-specific metrics for camera pose estimation, scene reconstruction, and human mesh reconstruction. Unless otherwise specified, camera trajectory and scene reconstruction errors are measured in meters, while human reconstruction errors are measured in millimeters.

Low-light reconstruction. We use the Oxford Day&Night [39] dataset, which provides calibrated camera poses and semi-dense point clouds captured under varying illumination conditions, including day, dusk, and night. This dataset enables systematic evaluation of reconstruction robustness under severe appearance degradation. For camera pose estimation, we report Absolute Trajectory Error (ATE), Relative Pose Error in translation (RPE-T), and Relative Pose Error in rotation (RPE-R). ATE measures the global trajectory discrepancy after alignment, RPE-T measures relative translation drift between frames, and RPE-R measures relative rotational error. ATE and RPE-T are reported in meters, while RPE-R is reported in degrees.

For scene reconstruction, we report Accuracy, Completeness, Chamfer Distance, and Normal Consistency. Accuracy measures the average distance from predicted points to the ground-truth point cloud, while Completeness measures the average distance from ground-truth points to the prediction. Chamfer Distance is computed as the sum of the two directional distances:

$$
\mathrm { C D } ( \mathcal { P } , \mathcal { G } ) = \frac { 1 } { | \mathcal { P } | } \sum _ { \mathbf { p } \in \mathcal { P } } \operatorname* { m i n } _ { \mathbf { g } \in \mathcal { G } } \| \mathbf { p } - \mathbf { g } \| _ { 2 } + \frac { 1 } { | \mathcal { G } | } \sum _ { \mathbf { g } \in \mathcal { G } } \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { P } } \| \mathbf { g } - \mathbf { p } \| _ { 2 } ,\tag{16}
$$

where $\mathcal { P }$ and $\mathcal { G }$ denote the predicted and ground-truth point sets. Accuracy, Completeness, and Chamfer Distance are reported in meters. Normal Consistency measures the cosine similarity between predicted and ground-truth surface normals, where higher values indicate better local geometric consistency.

Joint human-scene reconstruction. We adopt the BEDLAM [42] dataset, which provides both 3D scene geometry and human SMPL parameters as ground truth. The dataset is synthetically generated across diverse indoor and outdoor environments, allowing controlled evaluation of human-scene interaction.

For scene reconstruction, we use the same Accuracy, Completeness, Chamfer Distance, and Normal Consistency metrics as described above. For human mesh reconstruction, we report PA-PVE, Metric-PVE, PA-MPJPE, and Metric-MPJPE. PVE measures the mean per-vertex error between predicted and ground-truth human meshes, while MPJPE measures the mean per-joint position error. The prefix PA indicates evaluation after Procrustes alignment, which removes global similarity transformation differences. In contrast, Metric denotes evaluation in the original metric space without alignment, and therefore reflects absolute human localization and the spatial consistency between the reconstructed human body and the scene. All human reconstruction metrics are reported in millimeters.

Test-time adaptation. Following the Test3R/CUT3R sparse-view protocol, we sample every 200 frames on 7-Scenes [40] and every 500 frames on NRGBD [41] to simulate test-time adaptation scenarios. Each sequence therefore contains approximately 3–5 views on 7-Scenes and 2–4 views on NRGBD, requiring the model to reconstruct geometry under self-supervised conditions without additional training data. We evaluate scene reconstruction using Accuracy, Completeness, Chamfe Distance, and Normal Consistency, with distance-based metrics reported in meters.

Cross-view attention diagnostics. GT-valid query/reference pairs are determined by projection and reprojection with dataset depth and camera geometry, and shared by all models. $H _ { \mathrm { n \Omega } } =$ $\begin{array} { r } { - \sum _ { j } p _ { j } \log p _ { j } / \log N } \end{array}$ uses the conditional reference distribution over N patch tokens. The frozen backbone is evaluated once; adapted statistics use three released checkpoints (seeds 1111/1112/1113). All confidence intervals use 10,000 whole-sequence bootstrap replicates stratified by site and day/night, preserving input, layer, and model pairing. They describe sequence sampling within the five observed sites, conditional on the three checkpoints, rather than training-seed population uncertainty. Passive measurement leaves model outputs unchanged.

Table 5: Ablation of components in 3R-Adapter on the Oxford Day&Night [39] test dataset for low-light reconstruction.
<table><tr><td>Deformable Retrieval</td><td>Relational Rewiring</td><td>Consistency Refinement</td><td>Accuracy ↓</td><td>Complete ↓</td><td>Chamfer ↓</td><td>Normal ↑</td></tr><tr><td></td><td></td><td></td><td>0.562</td><td>0.639</td><td>1.201</td><td>0.534</td></tr><tr><td>√</td><td></td><td></td><td>0.540</td><td>0.612</td><td>1.152</td><td>0.534</td></tr><tr><td>√</td><td>√</td><td></td><td>0.397</td><td>0.445</td><td>0.842</td><td>0.542</td></tr><tr><td></td><td>1</td><td>√</td><td>0.483</td><td>0.529</td><td>1.012</td><td>0.536</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.254</td><td>0.327</td><td>0.581</td><td>0.550</td></tr></table>

## A.4 Detailed Ablation Study

Ablation on main components. Table 5 studies the contribution of each component in 3R-Adapter on the Oxford Day&Night dataset using CUT3R as the backbone. The vanilla CUT3R baseline obtains a Chamfer Distance of 1.201, indicating substantial degradation under low-light conditions.

Adding Deformable Retrieval alone slightly improves the reconstruction results, reducing Chamfer distance from 1.201 to 1.152. This suggests that sparse candidate querying can reduce noisy dense interactions, but retrieving candidate locations alone is insufficient to recover stable cross-view structure. When Relational Rewiring is further introduced, performance improves significantly, with Chamfer distance reduced to 0.842 and Accuracy improved from 0.540 to 0.397. This confirms that the key benefit comes from explicitly reorganizing retrieved candidates into structured cross-view relations.

Consistency Refinement also provides clear gains when used alone, improving Chamfer distance from 1.201 to 1.012. This shows that lightweight residual correction can stabilize predictions even without explicit retrieval and rewiring. However, the full 3R-Adapter achieves the best results across all metrics, reducing Chamfer distance to 0.581 and improving Normal consistency to 0.550. These results demonstrate the complementary roles of the three components: Deformable Retrieval identifies reliable candidates, Relational Rewiring reconstructs cross-view associations, and Consistency Refinement safely injects the adapted relations into the prediction stream.

Ablation on Hyperparameters. We further analyze the sensitivity of 3R-Adapter to four key hyperparameters: the number of sampling locations K, the search radius ρ, the relational fusion factor α, and the residual scaling factor γ. We report Chamfer Distance on the Oxford Day&Night dataset, where lower values indicate better reconstruction quality.

![](images/dcd8a20cfee9e7f73d1f55903a66c964295d8e746a02690733fe65f14167d63d.jpg)  
Figure 5: Ablation on the hyperparameters of 3R-Adapter on the Oxford Day&Night [39] test dataset for scene reconstruction.

As shown in Fig. 5, increasing the number of sampling locations from $K = 1$ to $K = 4$ consistently improves performance, suggesting that multiple candidates provide richer relational evidence for cross-view reconstruction. However, using more than four samples brings marginal gains and may introduce redundant or noisy candidates. The search radius $\rho$ exhibits a similar trend. A moderate radius improves local relation discovery, with the best performance achieved at $\rho = 0 . 2 0$ , while an overly large radius tends to include irrelevant distant features.

The relational fusion factor α controls the contribution of rewired cross-view relations. The model performs best at $\alpha = 0 . 1 0$ , while larger values degrade performance, indicating that excessive relational injection can disturb the pre-trained feature distribution. The residual scaling factor γ shows a clear balance between adaptation strength and prior preservation, with the best result obtained a $\mathrm { t } \gamma = 0 . 3 0$

Overall, 3R-Adapter achieves the best configuration with K = 4, ρ = 0.20, $\alpha = 0 . 1 0$ , and $\gamma = 0 . 3 0$ which is used as the default setting in our experiments.

## A.5 Implementation Details

We build 3R-Adapter on top of pretrained feed-forward 3D reconstruction backbones, including DUSt3R [1], CUT3R [2], and $\pi ^ { \bar { 3 } } \ [ 4 ]$ . Across all experiments, the pretrained backbone is frozen, and only the lightweight adapter parameters are updated. This setting allows the model to preserve geometric priors learned from large-scale pre-training while adapting to data-sparse long-tail scenarios with minimal trainable parameters. We optimize the adapter modules using AdamW. The hyperparameters of 3R-Adapter are set according to the ablation study: the number of deformable sampling points is $K = 4 ,$ , the search radius is $\rho = 0 . 2 0$ , the relational fusion factor is $\alpha = 0 . 1 0$ and the residual scaling factor is $\gamma = 0 . 3 0$ . All experiments are conducted on 8 NVIDIA RTX 5090 GPUs. For low-light reconstruction and joint human-scene reconstruction, we train the adapters on the corresponding training splits and report results on the held-out evaluation sets. For test-time adaptation, the adapter is optimized independently on each test scene using self-supervised reconstruction loss [17], without accessing additional training data or ground-truth geometry.

## A.6 Training Objectives

We optimize 3R-Adapter using the standard objective of each adaptation task, while keeping the pre-trained backbone frozen. This ensures that the comparison focuses on the adaptation mechanism rather than additional supervision.

For low-light reconstruction, we use geometric reconstruction losses on predicted point maps or depth representations, together with pose supervision when camera trajectories are available:

$$
\mathcal { L } _ { \mathrm { l o w } } = \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } .\tag{17}
$$

For joint human-scene reconstruction, we jointly supervise scene geometry and human body reconstruction:

$$
{ \mathcal { L } } _ { \mathrm { h s } } = { \mathcal { L } } _ { \mathrm { s c e n e } } + \lambda _ { \mathrm { m e s h } } { \mathcal { L } } _ { \mathrm { m e s h } } + \lambda _ { \mathrm { j o i n t } } { \mathcal { L } } _ { \mathrm { j o i n t } } + \lambda _ { \mathrm { s m p l } } { \mathcal { L } } _ { \mathrm { s m p l } } .\tag{18}
$$

Following Human3R [14], we incorporate human prior tokens from Multi-HMR [43] and fuse them with reconstruction tokens to enhance human-aware relational modeling. These tokens provide body-structure cues that help the model associate human proxy representations with body regions, while 3R-Adapter adapts the cross-view relational structures of the reconstruction backbone.

For test-time adaptation, we follow the self-supervised objective of Test3R [17]. Given sparse input views, the model predicts pointmaps under different reference-view choices. Since these predictions correspond to the same underlying scene geometry, we enforce consistency among pointmaps generated from different references:

$$
\mathcal { L } _ { \mathrm { t t a } } = \sum _ { r _ { 1 } \neq r _ { 2 } } \left. \mathbf { P } ^ { ( r _ { 1 } ) } - \mathbf { P } ^ { ( r _ { 2 } ) } \right. _ { 1 } ,\tag{19}
$$

where $\mathbf { P } ^ { ( r ) }$ denotes the predicted pointmap when view $r$ is selected as the reference view. This self-supervised loss enables scene-level adaptation without ground-truth geometry.

## A.7 Broader Impact

3R-Adapter aims to improve the adaptability and robustness of feed-forward 3D reconstruction models in data-sparse scenarios, such as low-light environments, joint human-scene reconstruction, and test-time adaptation. By reducing the number of trainable parameters required for downstream adaptation, our method may lower the computational and data costs of deploying 3D reconstruction systems in practical settings, including robotics, augmented reality, digital content creation, and embodied intelligence.

At the same time, improved 3D reconstruction capabilities may raise privacy and safety concerns when applied to real-world environments or human-centered scenes. In particular, reconstructing human geometry or private spaces from visual inputs should be conducted with appropriate consent, data protection, and usage constraints. Our work focuses on technical improvements for efficient adaptation and does not introduce new data collection procedures. We encourage future applications of such systems to follow responsible data governance practices and to carefully consider potential misuse in surveillance or unauthorized 3D reconstruction.