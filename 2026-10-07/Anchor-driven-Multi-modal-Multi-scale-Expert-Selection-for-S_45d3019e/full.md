# Anchor-driven Multi-modal Multi-scale Expert Selection for Survival Prediction

Tao Zhou, Senior Member, IEEE, Ying Hu, Huazhu Fu, Senior Member, IEEE, Yi Zhou Senior Member, IEEE, Xiao-Jun Wu, Haibin Ling, Fellow, IEEE,

Abstract— The integrative analysis of histopathological Whole-Slide Images (WSIs) and transcriptomic profiles holds significant promise for cancer survival prediction. However, existing methods typically project multi-modal features directly into a shared latent space without explicit alignment, leading to the entanglement of mismatched morphological cues and molecular signals. Furthermore, current fusion strategies often treat the extreme spatial heterogeneity of WSIs uniformly, lacking mechanisms to adaptively prioritize clinically relevant tissue scales for individual patients. To address these limitations, we propose an Anchor-driven Multi-modal Multi-scale Expert Selection (AM<sup>2</sup>ES) framework for survival prediction. Specifically, we present an Anchor-driven Multi-modal Fusion (AMF) module, which introduces learnable semantic anchors as cross-modal mediators to bridge the semantic gap by enforcing a structurally regularized alignment between transcriptomic features and multi-scale pathology representations. Built upon this aligned semantic space, we further design a Hierarchical Mixture-of-Experts (H-MoE) selection module to decouple the hierarchical prognostic selection process. Mimicking the pathologist’s diagnostic workflow, H-MoE performs (i) Intra-scale Expert Filtering to discriminatively identify salient tumor regions within each magnification, and (ii) Inter-scale Hierarchy Routing to dynamically weight and select the most informative resolution levels. Extensive experiments on multiple TCGA cancer cohorts demonstrate that our AM<sup>2</sup>ES achieves state-of-the-art performance while offering fine-grained interpretability by visualizing how specific molecular pathways drive the expert routing decisions across tissue scales. The code will be released at https://github.com/taozh2017/AM2ES .

Index Terms—Survival Prediction, Multi-scale feature learning, Multi-modal Fusion, Mixture-of-experts, Intra-scale expert selection, Inter-scale hierarchy routing

## 1 INTRODUCTION

A <sup>CCURATE</sup> <sup>survival</sup> <sup>prediction</sup> <sup>is</sup> <sup>a</sup> <sup>cornerstone</sup> <sup>of</sup> <sup>pre-</sup>cision oncology. To tackle the challenges posed by the cision oncology. To tackle the challenges posed by the heterogeneity of cancer prognosis, researchers have drawn inspiration from clinical diagnostic workflows and focused on multimodal learning [1]. In clinical practice, pathologists typically integrate fine-grained spatial context from Whole Slide Images (WSIs) with molecular profiles to determine risk stratification. Translating this workflow into computational models, researchers have introduced various WSI-Omics fusion frameworks [2], [3], [4], [5], [6], enabling the extraction of complementary phenotypic and molecular determinants of patient outcomes from high-dimensional biomedical data.

Despite these significant advancements, existing WSI-Omics frameworks face intertwined challenges in both cross-modal alignment and spatial hierarchy modeling (as overviewed in Fig. 1). First, bridging the severe semantic gap between pixel-level morphology and abstract gene expression vectors remains a highly underconstrained prob-

![](images/9f1b01a67d3e8f5770a4c33f7cf3aa6d445c971621fe4c4758502862b2a5b1c0.jpg)  
. <sup>.</sup>Figure 1: Comparison of multimodal fusion paradigms in WSI-Omics survival analysis: (a) Early Fusion, which concatenates multi-scale pathological features prior to cross-modal interaction; (b) Hierarchical Fusion, which sequentially integrates multi-scale features layer by layer in a fixed cascading manner; and (c) Our proposed AM<sup>2</sup>ES framework, which explicitly decouples semantic alignment via dynamic anchors from hierarchical scale selection via H-MoE.

lem. Yet, current models typically rely on standard global cross-attention to fuse these disparate modalities in a single step [7], [8]. This “flat” interaction inevitably entangles two distinct objectives: semantic alignment and prognostic selection. By forcing gigapixel WSIs and 1D molecular signatures into a shared space simultaneously, the network is compelled to learn what to align (matching tissue patches to biological pathways) and what to select (identifying features that drive survival) all at once. Without explicit decoupling, this entanglement frequently leads to modality overfitting and feature collapse. Consequently, attention maps often highlight spurious background noise rather than meaningful biomarkers, severely compromising clinical interpretability.

Additionally, existing methods struggle to adaptively capture the clinical hierarchy of pathological features. In diagnostic pathology, survival-relevant patterns manifest at distinct anatomical scales. For instance, macro-tissue architecture and tumor margins are evaluated at low magnification $( e . g . , ~ 5 \times )$ , while micro-cellular atypia and mitotic figures require high magnification $( e . g . , \ 2 0 \times )$ . However, existing fusion architectures struggle to adaptively capture this clinical hierarchy, generally falling into two paradigms (Fig. 1(a-b)): early fusion and hierarchical Fusion. Early fusion methods (Fig. 1(a)) typically extract multi-scale WSI features using conventional flattening strategies [9], [10] and concatenate them into a unified sequence before interaction. This strategy treats distinct biological structures as equivalent visual tokens, inevitably leading to scale confusion, noise propagation, and the dilution of scalespecific prognostic signals. Conversely, Hierarchical fusion methods (Fig. 1(b)) adopt a sequential pipeline, integrating different magnifications layer by layer in a fixed cascading manner [11], [12], [13]. By assuming a fixed order of feature importance across all patients, these models fail to adaptively prioritize the specific resolutions that are most relevant to prognosis for individual heterogeneous cases.

To address the dual challenges of entangled alignment and architectural rigidity, an effective multimodal framework must explicitly decouple these processes. In this paper, we propose AM<sup>2</sup>ES, a novel Anchor-driven Multimodal Multi-scale Expert Selection framework for cancer survival prediction using multi-modal data. As illustrated in Fig. 1(c), our model effectively bridges the gap between WSIs and transcriptomic profiles while selecting informative features to enhance prediction performance. Specifically, to resolve the modality gap, we present an Anchordriven Multi-modal Fusion (AMF) module. Rather than relying on static prototypes, our anchors are dynamically conditioned on global patient contexts, serving as personalized semantic mediators. Through structural regularization, these anchors explicitly align transcriptomic and pathological representations, ensuring that the extracted visual features are biologically grounded in the corresponding molecular profiles. To address the scale hierarchy, we develop a Hierarchical Mixture-of-Experts (H-MoE) selection strategy. This strategy operates in two stages: (1) an Intra-scale Filtering stage, which employs scoring experts to rigorously isolate salient multi-modal representations and filter out background noise within each distinct resolution; and (2) an Inter-scale Fusion stage, which leverages rank-based alignment and feature-level MoE to dynamically and efficiently recalibrate and synthesize evidence across different spatial scales. This joint design enables the model to adaptively weigh multi-scale evidence tailored to each patient’s unique pathology for accurate survival prediction.

In summary, our contributions are four-fold:

• We propose $\mathrm { A M ^ { 2 } E S } ,$ a novel framework that explicitly decouples cross-modal semantic alignment from hierarchical prognostic selection, addressing the entanglement and flatness issues prevalent in existing multimodal survival models.

We present a dynamic anchor-driven multi-modal fusion module where learnable and patientcontextualized anchors serve as semantic bridges. By enforcing orthogonality and sparsity constraints, this module ensures that pathological features are diverse, stable, and biologically grounded in their corresponding transcriptomic profiles.

We propose a two-stage H-MoE mechanism for expert selection. It integrates intra-scale gating for noise filtration and inter-scale dynamic recalibration for adaptive resolution fusion, achieving a balance between multi-scale expressiveness and stable routing.

Extensive experiments on multiple TCGA cohorts demonstrate that our AM<sup>2</sup>ES achieves state-of-theart performance. Crucially, our framework offers fine-grained interpretability by visually tracing both the anchor-driven semantic shifts and the patientspecific scale preferences.

## 2 RELATED WORK

## 2.1 Multimodal Learning for Cancer Prognosis

Integrating whole-slide histopathology with transcriptomic data has become a pivotal direction for enhancing cancer prognostication. Early multimodal approaches [14], [15] predominantly fused WSI-derived features with genomic signatures within a unified embedding space via simple concatenation, aiming to capture complementary cues. Mobadersany et al. [16] pioneered the joint modeling of histology and gene expression using convolutional networks. Subsequent works, such as Pathomic Fusion [17], incorporated tensorbased gating to model modality interactions. More recently, Transformer-based frameworks have dominated the field. Approaches like MCAT [2] and MOTCat [18] utilize coattention mechanisms to align visual patches with genomic embeddings. While effective, these methods typically intertwine representation learning and prognostic fusion within a single “flat” interaction step. They lack explicit semantic mediators, forcing the network to implicitly learn crossmodal correspondence while simultaneously performing risk prediction. This often results in modality overfitting and feature entanglement, directly motivating the explicitly decoupled anchor-driven multi-modal fusion paradigm in our framework.

## 2.2 Multi-scale Representation in Computational Pathology

Histopathology images exhibit substantial heterogeneity across scales, where biological signals are distributed hierarchically: global tissue phenotypes (e.g., tumor-stroma ratio) are visible at low magnification (5×), while finegrained cellular atypia requires high magnification (20×). Consequently, multi-scale learning has become essential for comprehensive WSI analysis [19], [20], [21]. Early approaches extended Multiple Instance Learning (MIL) to handle multi-resolution inputs. For instance, Ilse et al. [22] proposed attention-based pooling to weigh instances, while DS-MIL [4] introduced a dual-stream architecture that utilizes low-magnification scores to guide the selection of high-magnification patches. To better capture spatial context, Graph Convolutional Networks (GCNs) such as Patch-GCN [23] have been employed to model the topological relationships between multi-scale tissue regions.

![](images/80e2061d02184853f9ebfb89046bd8ab83aa76b8da26815d8b6299b6f3fb851a.jpg)  
Figure 2: Overview of the proposed AM<sup>2</sup>ES framework for multi-modal survival prediction. The pipeline integrates multi-scale WSIs and transcriptomic profiles through two synergistic stages: (1) Dynamic Anchor-Driven Alignment: Patient-specific anchors are generated conditioned on global genomic and morphological contexts. These anchors serve as semantic mediators to align multi-modal features, where structural regularization (orthogonality and sparsity constraints) is enforced to capture diverse biological patterns. (2) Hierarchical Mixture-of-Experts (H-MoE): A two-stage mechanism handles tissue heterogeneity. The Intra-scale Expert Selection module evaluates and filters morphological signals within each resolution (i.e., 5×, 10×, 20×) to remove redundancy. Subsequently, the Inter-scale MoE Fusion module employs rank-based alignment and feature-level gating to dynamically recalibrate and aggregate prognostic evidence across scales for final survival estimation.

More recently, Transformer-based architectures have advanced WSI analysis by modeling long-range dependencies. TransMIL [24] employs pyramid-like token aggregation, whereas HIPT [25] uses a hierarchical vision transformer to capture features from cellular to regional levels. Promptbased methods, such as MSCPT [26], further leverage learnable prompts to adaptively capture multi-scale context for tasks such as WSI classification. Despite these advances, most existing methods combine multi-scale features through fixed strategies, such as early or hierarchical fusion. They lack explicit mechanisms to identify the prognostically informative scale for each individual patient.

## 2.3 Anchor-driven Prototypes and Multi-modal Fusion

Prototype- and anchor-based learning has emerged as an effective paradigm for structured representation learning [27], [28]. In computer vision, DETR [29] introduced learned query embeddings as semantic extractors. This concept has been extended to general multi-modal learning by recent frameworks [30], which utilized anchors to construct an optimal unified representation space for heterogeneous modalities. However, existing anchor-based approaches in survival analysis typically treat anchors as fixed parameters.

While methods like DPU [31] have introduced dynamic prototype updating to handle out-of-distribution shifts, such dynamic mechanisms have rarely been adapted for patient-specific cross-modal fusion in survival tasks. Most current models seldom adapt anchors based on the joint transcriptomic–histological context. In contrast, the Anchordriven Multi-modal Fusion (AMF) module in our AM<sup>2</sup>ES introduces dynamic, patient-contextualized anchors. These anchors act as personalized semantic bridges, conditioned on global RNA profiles, to explicitly probe and align morphological features via structural regularization.

## 2.4 Mixture-of-Experts Selection Mechanisms

Mixture-of-Experts (MoE) [32] has re-emerged as a powerful paradigm for conditional computation, where sparse gates dynamically activate specialized expert units. While modern variants like Switch Transformers [33] and V-MoE [34] have demonstrated scalability in large-scale language and vision models, their adoption in computational pathology remains nascent. In medical imaging, MoE mechanisms have primarily been explored to handle domain variability and stain heterogeneity [35], [36]. However, these applications generally adopt single-stage structures without explicit semantic disentanglement. More critically, existing WSI–omics survival models rarely combine MoE gating with explicit multimodal fusion strategies. Most frameworks fuse modalities early or late in a shared latent space, lacking mechanisms to perform scale-aware or survival-aware expert selection.

## 3 PROPOSED METHOD

## 3.1 Problem Definition

We consider a survival prediction task on a cohort of $N$ patients, denoted as $\boldsymbol { \mathcal { D } } ^ { \setminus } = \ \{ ( X _ { i } , t _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N } .$ . For the i-th patient, the multi-modal input is defined as $\dot { X } _ { i } = \{ X _ { i } ^ { p } , X _ { i } ^ { g } \}$ where $X _ { i } ^ { p }$ represents a multi-scale set of pathological wholeslide images (WSIs), and $X _ { i } ^ { g }$ denotes the genomic profile. Let $T _ { i }$ be the continuous random variable representing the true time-to-event (survival time) for the i-th patient. The survival outcome is composed of the observed time $t _ { i } \in \mathbb { R } ^ { + }$ and the censorship status $c _ { i } \in \{ 0 , 1 \}$ . In our task, $c _ { i } = 1$ denotes a right-censored observation, while $c _ { i } = 0$ indicates an uncensored event (death).

To effectively model the survival distribution, we adopt the discrete-time survival paradigm. We partition the continuous time horizon into R non-overlapping intervals: $\left[ q _ { 0 } , q _ { 1 } \right) , \left[ q _ { 1 } , q _ { 2 } \right) , \ldots , \left[ q _ { R - 1 } , q _ { R } \right)$ . The bin boundaries $\{ q _ { j } \} _ { j = 1 } ^ { R }$ are determined by the R-quantiles of survival times from the uncensored subset, with $q _ { 0 } = \mathrm { m i n } _ { i } ( t _ { i } ) - \epsilon$ and $q _ { R } =$ max<sub>i</sub> $( t _ { i } ) + \epsilon$ to cover the entire cohort. The goal is to estimate the discrete hazard probability $p _ { i , j } \colon$

$$
\begin{array} { r } { p _ { i , j } = \operatorname* { P r } ( T _ { i } \in [ q _ { j - 1 } , q _ { j } ) \mid T _ { i } \geq q _ { j - 1 } ) , } \end{array}\tag{1}
$$

which represents the conditional probability of the event occurring in the j-th interval, given that the patient survived until the start of that interval. The survival function at the k-th time interval is subsequently derived as follows:

$$
S _ { i } ( k ) = \prod _ { j = 1 } ^ { k } ( 1 - p _ { i , j } ) , \quad \mathrm { w i t h ~ } S _ { i } ( 0 ) = 1 .\tag{2}
$$

## 3.2 Overview of Architecture

In this work, we propose AM<sup>2</sup>ES for survival prediction from multi-scale WSI and transcriptomic data (Fig. 2). The framework comprises three components: multi-modal feature encoding, Anchor-driven Multi-modal Fusion (AMF), and Hierarchical Mixture-of-Experts (H-MoE) selection. First, independent encoders extract multi-scale WSI features $( 5 \times , 1 0 \times , \mathsf { \hat { 2 0 } } \times )$ and RNA embeddings, while a Transformerbased clustering module summarizes WSI features into semantic prototypes. Next, AMF employs learnable anchors to align image prototypes and genomic embeddings in a shared representation space. Finally, H-MoE selects informative features within each scale and adaptively integrates them across scales for survival prediction.

## 3.3 Multi-scale Histopathology and Genomic Feature Encoding

We employ separate encoding streams to derive multi-scale morphological representations from WSIs and biologically grouped descriptors from genomic profiles. The obtained image features and genomic embeddings provide structured inputs for the subsequent anchor-driven fusion module. Moreover, rather than prematurely blending representations across resolutions through heuristic pooling, our framework hierarchically integrates multiscale features.

## 3.3.1 Multi-scale WSI Representation Learning

For patient $i ,$ the WSI at scale $s ~ \in ~ \{ 5 \times , 1 0 \times , 2 0 \times \}$ is represented as a bag of patches $W _ { i } ^ { s } ~ = ~ \{ w _ { i , n } ^ { s } \} _ { n = 1 } ^ { M _ { i } ^ { s } }$ . Each patch is encoded by a shared visual encoder $E _ { w } ( \cdot )$ and projected into a D-dimensional latent space:

$$
\mathbf { v } _ { i , n } ^ { s } = \phi \big ( E _ { w } ( \mathbf { w } _ { i , n } ^ { s } ) \big ) \in \mathbb { R } ^ { D } ,\tag{3}
$$

where $\phi ( \cdot )$ denotes the projection head, implemented as a fully connected layer with ReLU activation.

Then, we employ a Transformer-based clustering module to aggregate the dense patch embeddings into scalespecific $\breve { P ^ { s } }$ representative semantic prototypes. Specifically, we initialize a set of learnable prototype queries ${ \bf Q } ^ { s } { \bf \Xi } \in \Xi$ $\mathbb { R } ^ { P ^ { s } \times D }$ , which serve as learnable cluster centers in the embedding space. Unlike traditional clustering algorithms that rely on static distance metrics, our module utilizes the cross-attention mechanism to perform task-driven softassignment. Given the input dense patch embeddings $\mathcal { V } _ { i } ^ { s } \in$ $\mathbb { R } ^ { M _ { i } ^ { s } \times D }$ (serving as both keys and values), the semantic prototypes $\mathcal { H } _ { i } ^ { s }$ are computed via standard Multi-Head Cross-Attention (MHCA) followed by a Feed-Forward Network (FFN):

$$
\mathcal { H } _ { i } ^ { s } = \mathrm { F F N } ( \mathrm { M H C A } ( \mathbf { Q } ^ { s } , \mathcal { V } _ { i } ^ { s } , \mathcal { V } _ { i } ^ { s } ) ) \in \mathbb { R } ^ { P ^ { s } \times D } ,\tag{4}
$$

where the cross-attention mechanism implicitly calculates the normalized contributions of all constituent patches to each specific prototype. By optimizing the model end-toend, $\hat { \mathbf { Q } } ^ { s }$ automatically converges to representative centers that capture distinct morphological patterns without requiring heuristic clustering steps. This attention-driven softassignment dynamically groups morphologically similar patches into the patient-level WSI representation $\mathcal { H } _ { i } ^ { s } \ =$ $\left\{ { \bf h } _ { i , n } ^ { s } \right\} _ { n = 1 } ^ { P ^ { s } }$ . These multi-scale embeddings efficiently preserve coarse-to-fine structural information while strictly controlling computational complexity for downstream fusion.

## 3.3.2 Biologically-grouped Genomic Encoding

We partition the genomic profile of the i-th patient into $M _ { g }$ biologically defined groups, i.e., ${ \bf X } _ { i } ^ { g } \ = \ \{ \dot { \bf x } _ { i , 1 } ^ { g } , \ldots , { \bf x } _ { i , M _ { g } } ^ { g } \} .$ where each subset $\mathbf { x } _ { i , j } ^ { g } ~ \in ~ \mathbb { R } ^ { d _ { j } }$ contains $d _ { j }$ genes associated with a specific biological process. To accommodate these differences in dimensionality and statistical properties, we employ category-specific encoders. Then, each genomic subset is projected into a shared latent dimension D via a lightweight network $f _ { j } ( \cdot )$ as follows:

$$
\mathbf { g } _ { i , j } = f _ { j } ( \mathbf { x } _ { i , j } ^ { g } ) \in \mathbb { R } ^ { D } .\tag{5}
$$

The resulting genomic bag is defined as $\begin{array} { r l } { \mathbf { G } _ { i } } & { { } = } \end{array}$ $\bigl \{ \mathbf { g } _ { i , 1 } , \ldots , \mathbf { g } _ { i , M _ { g } } \bigr \} ^ { \cup } \in \mathbb { R } ^ { M _ { g } \times D }$ , which preserves pathway-level organization for anchor-driven fusion.

## 3.4 Anchor-driven Multi-modal Fusion

In multi-modal survival prediction, histology and transcriptomics provide complementary yet distinct biological information. As illustrated in Fig. 3(a), existing methods [18] typically rely on co-attention and direct concatenation to fuse cross-modal data, which often leads to an entangled latent space where patients across different risk levels are (b) Our anchor-driven fusion method

![](images/7d055b88f8ef11bb142dfb0a647b9141457d7085f59218d00daa35c3d811ff8c.jpg)  
Figure 3: Comparison of multi-modal fusion paradigms in WSI-Omics survival analysis. (a) Existing methods [18] leverage co-attention and direct concatenation to fuse cross-modal data. (b) Our anchor-driven fusion strategy acts as a semantic bridge, explicitly aligning cross-modal features via learnable anchors to construct a highly discriminative and structured representation space.

not well separated. To address this limitation, we introduce learnable, scale-specific cross-modal anchors. Inspired by prototype learning, these anchors act as semantic bridges that align hierarchical tissue representations with geneexpression embeddings at each biological scale. Unlike static queries, the anchors are dynamic and patient-conditioned. Through bidirectional attention, they establish structured correspondences between high-level phenotypic patterns and underlying molecular pathway activities. As illustrated by the t-SNE visualization in Fig. 3(b), anchor-driven integration yields a more discriminative latent space and clearly separates distinct survival risk profiles.

## 3.4.1 Dynamic Anchor Generation

For each scale $s ,$ we maintain a set of K learnable reference anchors $A _ { r } ^ { s } \in \mathbb { R } ^ { K \times D }$ . Patient-specific anchors are obtained by conditioning these reference anchors on the global ${ \mathrm { g e } } \mathrm { - }$ nomic and scale-specific morphological contexts. To adapt these anchors to individual patients, we compute global context vectors from the encoded morphological features $\mathcal { H } _ { i } ^ { s }$ and $\mathbf { G } _ { i }$ (detailed in Sec. 3.3.1 and 3.3.2). More concretely, the global genomic context $\mathbf { c } _ { i } ^ { ( g ) }$ and the scale-specific morphological context ${ \bf c } _ { i } ^ { ( w , s ) }$ are derived via average pooling as follows:

$$
\mathbf { c } _ { i } ^ { ( g ) } = \frac { 1 } { M _ { g } } \sum _ { m = 1 } ^ { M _ { g } } \mathbf { g } _ { i , m } , \mathbf { c } _ { i } ^ { ( w , s ) } = \frac { 1 } { P ^ { s } } \sum _ { n = 1 } ^ { P ^ { s } } \mathbf { h } _ { i , n } ^ { s } .\tag{6}
$$

We then generate patient-specific dynamic anchors $\widetilde { \mathbf { A } } _ { i } ^ { s }$ by injecting these global signals into the reference anchors:

$$
\widetilde { \mathbf { A } } _ { i } ^ { s } = \mathbf { A } _ { r } ^ { s } + f _ { \mathrm { c o n d } } ^ { s } [ \mathbf { c } _ { i } ^ { ( g ) } \parallel \mathbf { c } _ { i } ^ { ( w , s ) } ] ,\tag{7}
$$

where $\big [ \cdot \big | \big | \big . \big ]$ denotes concatenation, and $f _ { \mathrm { c o n d } } ^ { s } : \mathbb { R } ^ { 2 D } $ $\mathbb { R } ^ { K \times D }$ is a dynamic calibration network. This mechanism ensures that the anchors $\widetilde { \mathbf { A } } _ { i } ^ { s }$ are tailored to the specific tumor microenvironment of the i-th patient at resolution s.

## 3.4.2 Dual-stream Anchor Attention

Using the patient-conditioned anchors $\widetilde { \mathbf { A } } _ { i } ^ { s }$ , we perform structured multi-modal alignment in parallel for each scale. Each anchor acts as a dual query to retrieve evidence from both histology and genomics. For a given scale s and its k-th anchor $\widetilde { \mathbf { a } } _ { i , k } ^ { s } ,$ the aggregated morphological evidence $\mathbf { z } _ { i , k } ^ { s , \mathrm { W S I } } \in \mathbb { R } ^ { D }$ is computed by attending over the morphological prototypes $\mathcal { H } _ { i } ^ { \bar { s } }$ as follows:

$$
\mathbf { z } _ { i , k } ^ { s , \mathsf { W S I } } = \sum _ { n = 1 } ^ { P ^ { s } } \mathrm { s o f t m a x } \left( \frac { ( \mathbf { W } _ { w } ^ { Q } \widetilde { \mathbf { a } } _ { i , k } ^ { s } ) ^ { \top } ( \mathbf { W } _ { w } ^ { K } \mathbf { h } _ { i , n } ^ { s } ) } { \sqrt { D } } \right) ( \mathbf { W } _ { w } ^ { V } \mathbf { h } _ { i , n } ^ { s } ) ,\tag{8}
$$

where $\mathbf { W } _ { w } ^ { Q } , \mathbf { W } _ { w } ^ { K } , \mathbf { W } _ { w } ^ { V } \in \mathbb { R } ^ { D \times D }$ are learnable projection matrices. The softmax is applied over all prototypes n, ensuring selective attention to scale-specific morphology patterns.

Similarly, the same anchor queries the genomic bag $\mathbf { G } _ { i }$ to retrieve molecular evidence $\mathbf { z } _ { i , k } ^ { s , \mathrm { G E N } } \in \mathbb { R } ^ { D }$ via a structurally symmetric operation:

$$
\mathbf { z } _ { i , k } ^ { s , \mathrm { G E N } } = \sum _ { m = 1 } ^ { M _ { g } } \mathrm { s o f t m a x } \left( \frac { ( \mathbf { W } _ { g } ^ { Q } \widetilde { \mathbf { a } } _ { i , k } ^ { s } ) ^ { \top } ( \mathbf { W } _ { g } ^ { K } \mathbf { { \mathbf { g } } } _ { i , m } ) } { \sqrt { D } } \right) ( \mathbf { W } _ { g } ^ { V } \mathbf { { \mathbf { g } } } _ { i , m } ) ,\tag{9}
$$

where $\mathbf { W } _ { g } ^ { Q } , \mathbf { W } _ { g } ^ { K } , \mathbf { W } _ { g } ^ { V } \in \mathbb { R } ^ { D \times D }$ are learnable genomic projection matrices.

To synthesize the retrieved cross-modal evidence while preserving the intrinsic semantics of the anchors, we employ a residual fusion mechanism:

$$
\mathbf { u } _ { i , k } ^ { s } = \mathrm { L N } \left( \sigma ( \mathbf { W } _ { f } [ \mathbf { z } _ { i , k } ^ { s , \mathrm { W S I } } \mid \mid \mathbf { z } _ { i , k } ^ { s , \mathrm { G E N } } ] + \mathbf { b } _ { f } ) + \widetilde { \mathbf { a } } _ { i , k } ^ { s } \right) ,\tag{10}
$$

where $\mathbf { W } _ { f } \in \mathbb { R } ^ { D \times 2 D }$ and $\mathbf { b } _ { f } \in \mathbb { R } ^ { D }$ are the fusion layer parameters, $\sigma ( \cdot )$ denotes the ReLU activation, and LN(·) represents Layer Normalization. This process yields a collection of aligned cross-modal features $\dot { \mathbb { U } } _ { i } = \{ \dot { \mathcal { U } } _ { i } ^ { 5 \times } , \mathcal { U } _ { i } ^ { 1 0 \times } , \mathcal { U } _ { i } ^ { 2 0 \times } \} .$ where each scale-specific set $\mathcal { U } _ { i } ^ { s } ~ = ~ \{ \mathbf { u } _ { i , k } ^ { s } \} _ { k = 1 } ^ { K }$ contains K refined multi-modal candidates. The distinct feature sets in $\mathbb { U } _ { i }$ are preserved to allow for dynamic and context-aware selection both within and across scales in the subsequent process.

## 3.4.3 Regularization for Stable Anchor Fusion

We regularize the patient-specific anchors to encourage diversity and sparsity. For each scale $s ,$ an orthogonality term promotes distinct anchor directions, while an $\ell _ { 1 }$ penalty encourages compact anchor activations. Let $\tilde { \mathbf { A } } _ { i } ^ { s } \in \overline { { \mathbb { R } } } ^ { K \times \check { D } }$ denote the matrix of patient-specific anchors at scale $s ,$ where each row a <sup>s</sup> $\widetilde { \mathbf { a } } _ { i , k } ^ { s }$ represents a distinct latent expert. We first enforce an orthogonality constraint to encourage diversity among the $K$ anchors. By penalizing the correlation between different anchors, we ensure they span distinct semantic directions in the latent space by

$$
\mathcal { L } _ { \mathrm { o r t h } } ^ { s } = \left. \widehat { \mathbf { A } } _ { i } ^ { s } ( \widehat { \mathbf { A } } _ { i } ^ { s } ) ^ { \top } - \mathbf { I } _ { K } \right. _ { F } ^ { 2 } ,\tag{11}
$$

where $\widehat { \mathbf { A } } _ { i } ^ { s }$ denotes the row-normalized anchor matrix, with its k-th row computed as $\widehat { \mathbf { a } } _ { i , k } ^ { s } = \widetilde { \mathbf { a } } _ { i , k } ^ { s } / \Vert \widetilde { \mathbf { a } } _ { i , k } ^ { s } \Vert _ { 2 }$ , and $\| \cdot \| _ { F }$ represents the Frobenius norm.

To improve interpretability and mitigate overfitting, we apply an ℓ<sub>1</sub>-norm sparsity constraint. This encourages each

anchor to activate only a compact subset of feature dimensions, which can be expressed by

$$
\mathcal { L } _ { \mathrm { s p a r s e } } ^ { s } = \frac { 1 } { K D } \sum _ { k = 1 } ^ { K } \| \widetilde { \mathbf { a } } _ { i , k } ^ { s } \| _ { 1 } .\tag{12}
$$

The total anchor regularization loss is aggregated across all three magnification levels, which can be formulated as follows:

$$
\mathcal { L } _ { \mathrm { a n c h o r } } = \sum _ { s \in \{ 5 \times , 1 0 \times , 2 0 \times \} } \big ( \mathcal { L } _ { \mathrm { o r t h } } ^ { s } + \beta \mathcal { L } _ { \mathrm { s p a r s e } } ^ { s } \big ) ,\tag{13}
$$

where $\beta$ is a hyperparameter that balances the contribution of the sparsity penalty.

## 3.5 Hierarchical Mixture-of-Experts Strategy

Given the aligned cross-modal features $\{ U _ { i } ^ { s } \} _ { s } ,$ H-MoE performs two-stage selection and fusion. Specifically, the intrascale expert selection sparsely routes and filters multimodal representations within each resolution, while the inter-scale MoE fusion dynamically recalibrates and integrates evidence across resolutions. This design adaptively prioritizes the clinically relevant magnifications for each patient.

## 3.5.1 Intra-scale Experts Scoring and Selection

We assess each descriptor’s prognostic relevance via an intra-scale scoring MoE. Given the complexity of tumor heterogeneity, a single linear evaluator is insufficient; we instead employ E lightweight MLP experts $\{ f _ { e } \} _ { e = 1 } ^ { E } ,$ each specialized in distinct risk patterns. A sparse top $- k _ { e }$ routing mechanism dynamically assigns each descriptor $\mathbf { u } _ { i , k } ^ { s }$ to the most competent experts. The final score $r _ { i , k } ^ { s }$ is obtained via a routed ensemble as follows:

$$
r _ { i , k } ^ { s } = \sum _ { e \in \mathscr { T } _ { i , k } ^ { s } } \pi _ { i , k , e } ^ { s } \cdot f _ { e } ( \mathbf { u } _ { i , k } ^ { s } ) ,\tag{14}
$$

where $\mathcal { T } _ { i , k } ^ { s }$ denotes the set of indices for the $k _ { e }$ experts with the highest routing scores, and $f _ { e } ( \cdot )$ maps the descriptor to a scalar confidence score. The gating coefficient $\pi _ { i , k , e } ^ { s } ,$ representing the routing probability of the e-th expert for the current descriptor, is derived via a routing network as follows:

$$
\pi _ { i , k } ^ { s } = \mathrm { s o f t m a x } \left( \mathrm { T o p } { - } k _ { e } ( \mathbf { W } _ { \mathrm { g a t e } } ^ { s } \mathbf { u } _ { i , k } ^ { s } ) \right) \in \mathbb { R } ^ { E } ,\tag{15}
$$

where $\mathbf { W } _ { \mathrm { g a t e } } ^ { s } \in \mathbb { R } ^ { E \times D }$ is the routing weight matrix. The Top- $\cdot k _ { e } ( \cdot )$ operation retains the $k _ { e }$ largest routing logits and masks the remaining logits.

Based on these comprehensive risk scores, we employ a hard selection strategy to isolate the most salient features. We sort the descriptors in descending order of $r _ { i , k } ^ { s }$ and retain the top- $. K ^ { \prime }$ ranked candidates:

$$
\begin{array} { r } { \mathcal { S } _ { i } ^ { s } = \left\{ \mathbf { u } _ { i , ( 1 ) } ^ { s } , \mathbf { u } _ { i , ( 2 ) } ^ { s } , \ldots , \mathbf { u } _ { i , ( K ^ { \prime } ) } ^ { s } \right\} , } \end{array}\tag{16}
$$

where the parenthesized subscript $( j )$ denotes the index of the descriptor with the j-th highest score. The selected set $S _ { i } ^ { s }$ is passed to the inter-scale fusion stage.

## 3.5.2 Inter-Scale Mixture-of-Experts Fusion

To synthesize the multi-scale evidence, we adopt a Rankbased alignment strategy. We posit that descriptors ranking at the same importance level across different scales share a latent semantic correlation. We construct a multi-scale feature vector $\mathbf { x } _ { i , j }$ by concatenating the j-th ranked descriptors from all scales:

$$
\begin{array} { r } { \mathbf { x } _ { i , j } = \left[ \mathbf { u } _ { i , ( j ) } ^ { 5 \times } \parallel \mathbf { u } _ { i , ( j ) } ^ { 1 0 \times } \parallel \mathbf { u } _ { i , ( j ) } ^ { 2 0 \times } \right] \in \mathbb { R } ^ { 3 D } , \quad j = 1 , \ldots , K ^ { \prime } . } \end{array}\tag{17}
$$

To mitigate optimization difficulties from direct linear projection of high-dimensional heterogeneous features, we introduce a feature-level MoE for dynamic channel-wise recalibration. It generates a context-aware mask $\mathbf { m } _ { i , j }$ for each multi-scale vector $\mathbf { x } _ { i , j }$ as a weighted sum of M fusion experts as follows:

$$
\mathbf { m } _ { i , j } = \sum _ { m = 1 } ^ { M } \boldsymbol { \eta } _ { i , j } ^ { m } \cdot \sigma \left( g _ { m } ( \mathbf { x } _ { i , j } ) \right) \in \mathbb { R } ^ { 3 D } ,\tag{18}
$$

where $g _ { m } ( \cdot ) \ : \ \mathbb { R } ^ { 3 D } \  \ \mathbb { R } ^ { 3 D }$ represents the m-th maskgeneration expert, $\sigma ( \cdot )$ is the sigmoid activation, and the routing coefficient vector $\begin{array} { r c l } { \eta _ { i , j } } & { = } & { [ \eta _ { i , j } ^ { 1 } , \dots , \eta _ { i , j } ^ { M } ] ^ { \top } } \end{array}$ is dynamically derived via a standard softmax router: $\begin{array} { r l } { \eta _ { i , j } } & { { } = } \end{array}$ softmax $\mathbf { \widetilde { \Gamma } } ( \mathrm { R o u t e r } ( \mathbf { x } _ { i , j } ) ) \ \in \ \mathbb { R } ^ { M }$ . The final fused representation $\mathbf { z } _ { i , j }$ is obtained by applying the generated mask to the input features and projecting them back to the latent space:

$$
\mathbf { z } _ { i , j } = \mathbf { W } _ { \mathrm { r e d } } \left( \mathbf { x } _ { i , j } \odot \mathbf { m } _ { i , j } \right) \in \mathbb { R } ^ { D } ,\tag{19}
$$

where $\odot$ denotes element-wise multiplication and $\mathbf { W _ { \mathrm { r e d } } } \in$ $\mathbb { R } ^ { D \times 3 D }$ is a learnable reduction matrix. The patient-level holistic representation is then derived by averaging these fused tokens: $\begin{array} { r } { { \bf h } _ { i } = \frac { 1 } { K ^ { \prime } } \sum _ { j = 1 } ^ { K ^ { \prime } } { \bf z } _ { i , j } } \end{array}$

## 3.6 Total Objective Function

After obtaining the patient-level holistic representation h<sub>i</sub> from the H-MoE module, we feed it into the survival prediction head. The vector of interval-wise hazard probabilities $\mathbf { p } _ { i } = [ p _ { i , 1 } , \ldots , p _ { i , R } ] ^ { \top }$ is predicted via a fully connected layer with sigmoid activation:

$$
\mathbf { p } _ { i } = \sigma ( \mathbf { W } _ { \mathrm { r i s k } } \mathbf { h } _ { i } + \mathbf { b } ) ,\tag{20}
$$

where $\mathbf { W } _ { \mathrm { r i s k } } \in \mathbb { R } ^ { R \times D }$ and $\mathbf { b } \in \mathbb { R } ^ { R }$ are learnable parameters. The training objective combines the survival loss with anchor regularization. For patient i, with observed time $t _ { i }$ assigned to interval $y _ { i }$ and censoring indicator $c _ { i } ,$ , the survival loss is defined as follows:

$$
\begin{array} { c l } { \displaystyle { \mathcal { L } _ { \mathrm { s u r v } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigg [ ( 1 - c _ { i } ) \log \big ( p _ { i , y _ { i } } S _ { i } ( y _ { i } - 1 ) \big ) } } \\ { \displaystyle { + c _ { i } \log \big ( S _ { i } ( y _ { i } ) \big ) \bigg ] , } } \end{array}\tag{21}
$$

where $S _ { i } ( \cdot )$ is derived from Eq. (2), and N denotes the number of patients.

The final objective function is a weighted combination of the survival loss and the anchor regularization terms. Therefore, the total loss function can be expressed by

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s u r v } } + \lambda \mathcal { L } _ { \mathrm { a n c h o r } } ,\tag{22}
$$

where λ is a hyperparameter balancing the prediction task and the regularization constraints.

## 4 EXPERIMENTS AND RESULTS

## 4.1 Experimental Setup

## 4.1.1 Datasets

To evaluate the performance of our proposed framework, we conducted extensive experiments on five publicly available cancer datasets from The Cancer Genome Atlas (TCGA), covering both survival outcomes and paired modalities: diagnostic Whole Slide Images (WSIs) and genomic profiles. The datasets include Bladder Urothelial Carcinoma (BLCA), Breast Invasive Carcinoma (BRCA), Uterine Corpus Endometrial Carcinoma (UCEC), Glioblastoma $\&$ Lower Grade Glioma (GBMLGG), and Lung Adenocarcinoma (LUAD). The sample size N is detailed in Table 1. For genomic data preprocessing, we filter genes based on variability and missingness. Following the protocols in [7], [8], [37], we organize genomic features into $M _ { g } = 6$ functional categories based on biological pathways: (1) Tumor Suppression, (2) Oncogenesis, (3) Protein Kinases, (4) Cellular Differentiation, (5) Transcription, and (6) Cytokines and Growth.

## 4.1.2 Experimental Settings

For WSI preprocessing, we employ the OTSU’s thresholding algorithm to create tissue masks, effectively filtering out background and artifacts. Non-overlapping patches of size $2 2 4 \times 2 2 4$ are extracted from the segmented tissue regions at three distinct magnification levels: 5×, 10×, and $2 0 \times ,$ , to capture hierarchical morphological details. Each patch is encoded using the pretrained UNI model [38] to extract 1024-dimensional embeddings, utilizing the feature extraction pipeline from CLAM [39], and we set $D = 2 5 6$ in the latent space. Following our proposed method, we aggregate these dense patch embeddings into scale-specific semantic prototypes using a Transformer-based clustering module. For genomic data, we employ Self-Normalizing Neural Networks (SNNs) [40] to instantiate the categoryspecific encoders $f _ { j } ( \cdot )$ described in Sec. 3.3.2, following the setting of [8].

## 4.1.3 Implementation Details and Evaluation Metric

All experiments are conducted in a PyTorch 2.6.0 environment with CUDA 12.4 and an NVIDIA GeForce RTX 4090 GPU with 24GB memory. During model training, we adopt the Adam optimizer with an initial learning rate of $2 \times 1 0 ^ { \dot { - } 4 }$ and a weight decay of $1 \times 1 0 ^ { - 5 }$ . All experiments are trained for 20 epochs. For the WSI representation learning, the number of scale-specific semantic prototypes $P ^ { s }$ is set to 64, 128, and 256 for the $5 \times , 1 0 \times ,$ , and 20× magnifications, respectively. For the proposed AM<sup>2</sup>ES architecture, the number of scale-specific dynamic anchors is set to $K = 1 6$ . In the H-MoE module, we configure $E = 4$ intra-scale scoring experts with a sparse top- $\boldsymbol { \cdot } \boldsymbol { k } _ { e } = 2$ gating strategy, and retain the top $K ^ { \prime } = 4$ candidates per scale. This is followed by $M = 4$ inter-scale experts for feature recalibration. Additionally, we empirically set $\lambda = 0 . 0 5$ and $\beta = 0 . 1$ . For a fair evaluation, we standardized the configurations of all comparison methods in our experiments.

Moreover, we employ a 5-fold cross-validation strategy with patient-level separation. This ensures that all WSI patches belonging to the same patient are exclusively assigned to either the training or testing set, strictly preventing data leakage. To evaluate survival prediction performance, we employ the Concordance Index (C-index), a standard metric that quantifies the model’s ability to correctly rank patient risk scores relative to their observed survival outcomes.

## 4.2 Comparison with State-of-the-art Methods

## 4.2.1 Comparison Methods

To demonstrate the effectiveness and robustness of our proposed framework, we conduct a comprehensive comparison against a wide range of state-of-the-art (SOTA) methods. As reported in Table 1, these baselines are rigorously categorized based on input modality (Unimodal vs. Multimodal) and visual granularity (Single-scale vs. Multi-scale). Unimodal Methods: For genomic data only, we utilize MLP [41], SNN [40], and transformer-based SNNTrans [42] as benchmarks. For WSI data only, we compare the proposed model with methods across different scales. For single-scale settings (20×), we include standard MIL approaches: ABMIL [43], CLAM [39], TransMIL [44], and WiKG [45]. To assess multi-scale capabilities $( 5 \times , 1 0 \times , 2 0 \times )$ we compare against DSMIL [10], CSMIL [46], and the hierarchical transformer HIPT [47]. Multimodal Methods: We further compare our model against SOTA multimodal fusion frameworks. The first group operates on single-scale WSIs, including PORPOISE [48], MCAT [7], MoCAT [8], and Survpath [49]. The second group involves multi-scale fusion methods, specifically M3IF [50], HiMT [12], and UMSA [13], which serve as the most direct competitors to our hierarchical approach. For a fair comparison, all WSI-based methods utilize the same UNI-encoded features and identical data splits.

## 4.2.2 Result Comparison

The quantitative results of survival prediction across five TCGA cohorts are summarized in Table 1. As observed, our model achieves the best overall performance, demonstrating superior generalization and robustness compared to existing SOTA methods.

Comparison with Unimodal Methods: In Table 1, we compare the proposed model against gene-based baselines (MLP, SNN, SNNTrans) and several pathology-based approaches. Among genomic methods, the transformer-based SNNTrans achieves the highest C-index of 0.667, outperforming MLP (0.657) and SNN (0.659). This suggests that attention mechanisms are effective at capturing complex intra-genomic interactions. In the pathological domain, the transformer-based TransMIL leads with a C-index of 0.660, surpassing ABMIL (0.652) and CLAM (0.650). Although multi-scale unimodal methods such as HIPT and DSMIL aim to incorporate hierarchical information, their performance (e.g., HIPT on BRCA: 0.538) does not consistently exceed that of single-scale transformer-based approaches. This suboptimal result may stem from the lack of effective cross-modal guidance to filter out irrelevant features across different magnifications.

Comparison with Multimodal Methods: Compared to unimodal approaches, multimodal methods generally demonstrate superior performance in survival prediction. Methods that emphasize modality alignment, such as Mo-CAT and MCAT, achieve C-index scores of 0.700 and 0.694, respectively. Survpath further improves upon this by leveraging topological interactions, reaching a C-index of 0.709. However, these approaches are limited by their reliance on single-scale inputs (20×), which fail to capture broader contextual tissue structures. Among multi-scale fusion methods, HiMT and UMSA exhibit competitive performance but achieve only average C-indices of 0.626 and 0.625, respectively, likely due to their relatively simplistic fusion or alignment strategies. Notably, our method attains a superior average C-index of 0.735, substantially outperforming the second-best method, Survpath (0.709). Specifically on the

Table 1: Quantitative comparison of C-index (mean ± std) across five TCGA cohorts. Baselines are categorized by input modalities (g: genomic, h: histology) and WSI scales. The best results are in bold, and the second-best are underlined.
<table><tr><td>Methods</td><td></td><td>Modal Scale</td><td>BLCA (N=373)</td><td>BRCA (N=955)</td><td>GBMLGG (N=550)</td><td>LUAD (N=452)</td><td>UCEC (N=480)</td><td>Average</td></tr><tr><td>MLP [41]</td><td></td><td>-</td><td></td><td> $0 . 6 1 3 \pm 0 . 0 1 9$ </td><td> $0 . 5 8 7 \pm 0 . 0 3 3$ </td><td> $0 . 8 0 9 \pm 0 . 0 2 9$ </td><td> $0 . 6 1 7 \pm 0 . 0 2 6$ </td><td> $0 . 6 5 7 \pm 0 . 0 3 6$ </td><td>0.657</td></tr><tr><td></td><td>SNN [40]</td><td>g. g.</td><td></td><td> $0 . 6 1 9 \pm 0 . 0 2 3$ </td><td> $0 . 5 9 6 \pm 0 . 0 2 7$ </td><td> $0 . 8 0 5 \pm 0 . 0 3 0$ </td><td> $0 . 6 2 5 \pm 0 . 0 1 9$ </td><td> $0 . 6 5 1 \pm 0 . 0 1 8$ </td><td>0.659</td></tr><tr><td>SNNTrans [42]</td><td></td><td>g.</td><td>一</td><td> $0 . 6 2 7 \pm 0 . 0 1 9$ </td><td> $0 . 6 1 8 \pm 0 . 0 1 8$ </td><td> $0 . 8 1 6 \pm 0 . 0 3 7$ </td><td> $0 . 6 3 1 \pm 0 . 0 2 3$ </td><td> $0 . 6 4 1 \pm 0 . 0 2 6$ </td><td>0.667</td></tr><tr><td>ABMIL [43]</td><td></td><td>h.</td><td>20×</td><td> $0 . 6 2 2 \pm 0 . 0 5 1$ </td><td> $0 . 6 1 4 \pm 0 . 0 3 7$ </td><td> $0 . 7 8 6 \pm 0 . 0 2 8$ </td><td> $0 . 5 9 6 \pm 0 . 0 6 7$ </td><td> $0 . 6 4 9 \pm 0 . 0 2 8$ </td><td>0.652</td></tr><tr><td>CLAM [39]</td><td></td><td>h.</td><td>20×</td><td> $0 . 6 1 3 \pm 0 . 0 5 8$ </td><td> $0 . 6 0 7 \pm 0 . 0 1 6$ </td><td> $0 . 7 9 2 \pm 0 . 0 3 2$ </td><td> $0 . 5 9 0 \pm 0 . 0 7 6$ </td><td> $0 . 6 5 0 \pm 0 . 0 6 6$ </td><td>0.650</td></tr><tr><td>TransMIL [44]</td><td></td><td>h.</td><td>20×</td><td> $0 . 6 2 5 \pm 0 . 0 5 0$ </td><td> $0 . 6 1 4 \pm 0 . 0 3 7$ </td><td> $0 . 8 0 3 \pm 0 . 0 3 7$ </td><td> $0 . 5 9 3 \pm 0 . 0 6 3$ </td><td> $0 . 6 6 5 \pm 0 . 0 1 9$ </td><td>0.660</td></tr><tr><td>WiKG [45]</td><td></td><td>h.</td><td>20×</td><td></td><td> $0 . 5 0 7 \pm 0 . 0 0 1$ </td><td></td><td> $0 . 5 9 2 \pm 0 . 0 0 8$ </td><td> $0 . 5 6 2 \pm 0 . 0 0 1$ </td><td>0.553</td></tr><tr><td>DSMIL [10]</td><td></td><td>h.</td><td>5×20×</td><td></td><td> $0 . 5 4 7 \pm 0 . 0 5 5$ </td><td></td><td> $0 . 5 8 8 \pm 0 . 0 8 3$ </td><td> $0 . 6 1 3 \pm 0 . 0 3 9$ </td><td>0.582</td></tr><tr><td>CSMIL [46]</td><td></td><td>h.</td><td> $5 \times 1 0 \times 2 0 \times$ </td><td></td><td> $0 . 5 4 1 \pm 0 . 0 2 9$ </td><td></td><td> $0 . 5 5 0 \pm 0 . 0 6 9$ </td><td> $0 . 5 2 6 \pm 0 . 0 8 2$ </td><td>0.539</td></tr><tr><td>HIPT [47]</td><td></td><td>h.</td><td> $5 \times 1 0 \times 2 0 \times$ </td><td></td><td> $0 . 5 3 8 \pm 0 . 0 8 7$ </td><td></td><td> $0 . 5 5 8 \pm 0 . 0 3 1$ </td><td> $0 . 6 0 2 \pm 0 . 0 4 2$ </td><td>0.566</td></tr><tr><td>PORPOISE [48]</td><td></td><td>h.+g.</td><td>20×</td><td> $0 . 6 6 1 \pm 0 . 0 3 4$ </td><td> $0 . 6 4 4 \pm 0 . 0 1 0$ </td><td> $0 . 8 2 9 \pm 0 . 0 3 3$ </td><td> $0 . 6 4 6 \pm 0 . 0 2 5$ </td><td> $0 . 6 6 0 \pm 0 . 1 0 6$ </td><td>0.688</td></tr><tr><td>MCAT [7]</td><td></td><td>h.+g.</td><td>20×</td><td> $0 . 6 6 0 \pm 0 . 0 4 6$ </td><td> $0 . 6 5 2 \pm 0 . 0 1 9$ </td><td> $0 . 8 3 3 \pm 0 . 0 2 9$ </td><td> $0 . 6 4 2 \pm 0 . 0 5 4$ </td><td> $0 . 6 8 6 \pm 0 . 0 4 2$ </td><td>0.694</td></tr><tr><td>MoCAT [8]</td><td></td><td>h.+g.</td><td>20×</td><td> $0 . 6 6 7 \pm 0 . 0 3 8$ </td><td> $0 . 6 5 9 \pm 0 . 0 3 4$ </td><td> $0 . 8 3 5 \pm 0 . 0 3 7$ </td><td> $0 . 6 5 6 \pm 0 . 0 2 1$ </td><td> $0 . 6 8 3 \pm 0 . 0 6 0$ </td><td>0.700</td></tr><tr><td>Survpath [49]</td><td></td><td>h.+g.</td><td>20×</td><td> $0 . 6 6 3 \pm 0 . 0 2 6$ </td><td> $\underline { { 0 . 6 6 1 \pm 0 . 0 2 4 } }$ </td><td> $\underline { { 0 . 8 3 9 \pm 0 . 0 3 9 } }$ </td><td> $\underline { { 0 . 6 7 1 \pm 0 . 0 3 6 } }$ </td><td> $\underline { { 0 . 7 0 9 \pm 0 . 0 3 9 } }$ </td><td>0.709</td></tr><tr><td>M3IF [50]</td><td></td><td> $\mathrm { h . + g . }$ </td><td> $5 \times 1 0 \times 2 0 \times$ </td><td>一</td><td> $0 . 5 2 8 \pm 0 . 0 6 0$ </td><td></td><td> $0 . 6 1 4 \pm 0 . 0 4 5$ </td><td> $0 . 6 1 5 \pm 0 . 0 6 0$ </td><td>0.585</td></tr><tr><td>HiMT [12]</td><td></td><td> $\mathrm { h . + g . }$ </td><td> $5 \times 1 0 \times 2 0 \times$ </td><td> $0 . 6 6 0 \pm 0 . 0 2 1$ </td><td> $0 . 4 9 0 \pm 0 . 0 6 6$ </td><td> $0 . 8 2 3 \pm 0 . 0 1 9$ </td><td> $0 . 5 7 7 \pm 0 . 0 1 8$ </td><td> $0 . 5 8 3 \pm 0 . 0 8 9$ </td><td>0.626</td></tr><tr><td>UMSA [13]</td><td></td><td> $\mathrm { h . + g . }$ </td><td> $5 \times 1 0 \times 2 0 \times$ </td><td></td><td> $0 . 5 7 6 \pm 0 . 0 5 0$ </td><td></td><td> $0 . 6 5 2 \pm 0 . 0 8 3$ </td><td> $0 . 6 4 9 \pm 0 . 0 4 6$ </td><td>0.625</td></tr><tr><td></td><td>Ours</td><td> $\mathrm { h . + g . }$ </td><td> $5 \times 1 0 \times 2 0 \times$ </td><td> $\mathbf { 0 . 6 8 5 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td>0.849 ± 0.019</td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.735</td></tr></table>

![](images/be41133ff280be2276e47c519203ccf3ac6941c567fc892a80904b873a297ba6.jpg)

![](images/5c7bdafec66f4a125922d51a25f41fcff70cfe9555a1f48562631a156cd1ef69.jpg)

![](images/d857804b7c48c39a4ceaa15e9f7c112810c8c1738bf410883218c9154f30192c.jpg)

![](images/d34f85b433aa05d62509a8807e22f4eac63197b9a990823948e658d5a4797bd0.jpg)

![](images/ece83c2c900072290dca92549db63107f249947d304f7fcbd97e04d602816a2d.jpg)

![](images/61250f1f527b789937fe98df03a3f7930032ccb0744ac85ba6388b2b638515fe.jpg)

![](images/d19c2112e06d4c8590af05efce34f1434b3e54afbd041f03860750973eeaa7f8.jpg)

![](images/ba1d43649ed8eb4dc574d31f18bab05ef35fe86119ab4b99a97b8e2c97cecfcd.jpg)

![](images/0ee5bdf659dba626355dc60ecb2abbec37fd6ab34750525f39c6e2f69d0113f5.jpg)

![](images/a37ac48e789f194b6e6cccba3fc6131177091e97a7b27fc72e0a56f83d3625a7.jpg)

![](images/063d93b95dd530843fcf45e74fc99d7a74ee7df23734524fc5f265c5217840d4.jpg)

![](images/c940408a891ab12d1a23e72f2977bbaf8e934ae6a06f4a4bc24d387924630298.jpg)

![](images/0799c49ec986ca2dd67efee04c43ad72bba01349f5104890abc21cb9f5f28c2e.jpg)

![](images/f9f012e7b24a02f44db778bb4b722ed3f298f7ab04b0955c7408c9c1e5142d05.jpg)

![](images/9b812c6e3d807d7eace305b38ef73c12a5f7c6982c46aabc60133b12eb1f8950.jpg)  
Figure 4: Kaplan-Meier survival analysis across five cancer cohorts (BLCA, BRCA, GBMLGG, LUAD, and UCEC), comparing our proposed AM<sup>2</sup>ES framework against Survpath and HiMT. Patients are stratified into high-risk (red) and low-risk (blue) groups based on the median predicted risk score. The shaded regions represent 95% confidence intervals.

Table 2: Comparison of external validation performance (Cindex) on two independent CPTAC cohorts.
<table><tr><td></td><td>Method |CPTAC-UCEC [51]</td><td>CPTAC-LUAD [52]</td></tr><tr><td>Survpath [49]</td><td> $0 . 5 0 3 4 \pm 0 . 0 1 1 8$ </td><td> $0 . 5 6 7 0 \pm 0 . 0 3 4 1$ </td></tr><tr><td>HiMT [12]</td><td> $0 . 5 4 7 0 \pm 0 . 0 7 4 7$ </td><td> $0 . 5 2 6 9 \pm 0 . 0 5 3 8$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 5 6 5 1 \pm 0 . 0 6 9 0 }$ </td><td> $\mathbf { 0 . 5 8 4 3 \pm 0 . 0 4 2 8 }$ </td></tr></table>

UCEC dataset, our AMF-MES model achieves an impressive score of 0.762, representing a considerable margin over Survpath’s 0.709. This consistent performance gain underscores the critical role of our anchor-guided routing mechanism, which adaptively integrates biological pathway information with hierarchical visual cues.

External Validation. We further evaluate our model on two independent CPTAC cohorts: Uterine Corpus Endometrial Carcinoma (CPTAC-UCEC [51]) and Lung Adenocarcinoma (CPTAC-LUAD [52]). To ensure a fair cross-cohort evaluation, we apply data harmonization and qualitycontrol procedures consistent with the TCGA training setting. Only patients with complete multi-omics profiles and valid survival data were included; cases with zero survival time were excluded. The final cohorts comprised 34 CPTAC-UCEC and 53 CPTAC-LUAD patients. For external validation, models trained on the corresponding TCGA cohorts are directly applied to the CPTAC cohorts without fine-tuning or retraining. We compare our method with two representative approaches, i.e., SurvPath [49] and HiMT [12]. As shown in Table 2, our model achieves the best performance on both cohorts. Although absolute C-index values were lower than in the internal TCGA evaluations, this is expected in zeroshot cross-cohort validation owing to batch effects, cohort differences, and limited sample sizes. Nevertheless, our method consistently outperformed competing approaches on both CPTAC cohorts, supporting its robustness and generalizability.

## 4.2.3 Kaplan-Meier Survival Analysis

To further assess the clinical value of our model in risk stratification, we perform Kaplan-Meier (KM) survival analysis on five TCGA cancer cohorts. Patients in each cohort are stratified into high-risk and low-risk groups based on the median predicted risk score derived from 5-fold crossvalidation.

As illustrated in Fig. 4, the Kaplan-Meier curves reveal statistically significant risk stratification across all cancer types. Visually, the high-risk (red) and low-risk (blue) curves diverge clearly shortly after the initial time point. In aggressive cancer subtypes such as BLCA and GBMLGG, the highrisk group exhibits a steep decline in survival probability within the first 20–40 months, whereas the low-risk group maintains a relatively stable survival rate. Furthermore, Fig. 4 provides a comparison between our model and two strong baselines, i.e., Survpath and HiMT. As the baseline methods struggle to maintain clear separation in cohorts like BRCA and UCEC, frequently exhibiting overlapping confidence intervals (shaded regions) in the later stages, while our framework achieves a consistently wider and more stable margin between the risk groups. This clear separation, accompanied by minimal overlap in the 95% confidence intervals, further underscores the model’s robustness. These results confirm that our model can effectively disentangle risk profiles at an early stage, highlighting its superior clinical utility in guiding timely therapeutic interventions compared to existing methods.

## 4.3 Ablation Study

To systematically evaluate the contribution of each component in our proposed framework and verify the robustness of our design choices, we conduct extensive ablation experiments on the BRCA, LUAD, and UCEC datasets. The ablative results are summarized in Table 3.

## 4.3.1 Effectiveness of Each Component

(a) Impact of Input Modalities. We first investigate the necessity of multimodal integration. As shown in Table 3 (IDs 1-3), the uni-modal baselines yield suboptimal performance, with both the genomic-only (MLP) and WSI-only (ABMIL) models achieving an identical average C-index of 0.620. Simply concatenating features from both modalities (ID 3) brings a noticeable improvement (+3.0%), reaching 0.650. This confirms that genomic profiles and pathological images contain complementary prognostic signals. However, the performance of simple concatenation is still limited compared to our full model (0.714), highlighting the importance of our advanced fusion module.

(b) Impact of Multi-scale Hierarchy. We further assess the contribution of the hierarchical processing of WSIs. In Exp. (4)-(6) in Table 3, we evaluate different scale combinations. The single-scale model (20×) achieves an average C-index of 0.683. Incorporating the 5× scale (Dual Scale, ID 5) slightly improves the performance to 0.687. The best performance is achieved by utilizing the full hierarchy (5×, 10×, 20×) in Exp. (6), reaching a C-index of 0.708. This result validates our hypothesis that a holistic survival analysis requires multi-granular visual information, combining global tissue architecture with local cellular details.

(c) Impact of Multi-modal Fusion Strategy. A key contribution of the proposed framework is the dynamic anchor-driven fusion strategy. To verify its superiority, we compare it against two established fusion strategies under the same multi-scale setting: Cross-Attention (widely used in TransMIL-style models) and Bilinear Pooling. As reported in experiments (7) and (8) in Table 3, our method (0.714) outperforms Cross-Attention (0.689) and Bilinear Pooling (0.682) by margins of 2.5% and 3.2%, respectively. We attribute this improvement to the design of patient-specific anchors, which act as dynamic prototypes that align highdimensional WSI features with genomic embeddings more effectively than generic attention maps, thereby reducing semantic misalignment.

(d) Impact of MoE Routing. Finally, we analyze the effectiveness of our hierarchical expert design, dissecting it into two stages: intra-scale selection and inter-scale fusion. Stage 1: Intra-scale Selection. We compare the Top-K instance selection strategy against Global Average Pooling (GAP) and Attention Pooling. As shown in entries (9)-(10) in Table 3, GAP (No Selection) and Attention Pooling (Soft Selection) yield average C-indices of 0.686 and 0.691, respectively. Our Top-K strategy (embedded in ID 12) achieves the highest performance (0.714), demonstrating that filtering out non-informative patches explicitly is more effective for prognosis than soft weighting or global averaging. Stage 2: Inter-scale Fusion. We further verify the necessity of the MoE module for fusing features across scales. Replacing the MoE dynamic gating with a static Linear Concatenation (shown in entry (11)) results in a performance drop to 0.687. This demonstrates that the “soft” routing mechanism in MoE allows the model to adaptively weight contributions from different magnifications based on sample difficulty, rather than relying on a fixed combination.

Table 3: Comprehensive ablation study of $\mathbf { A M ^ { 2 } E S } .$ The study validates: (a) input modalities, (b) multi-scale hierarchy, (c) anchor fusion mechanism, (d) hierarchical expert mechanism, and (e) different inter-scale alignment strategies.
<table><tr><td>ID</td><td>Model Variants</td><td>BRCA</td><td>LUAD</td><td>UCEC</td><td> $\underline { { \overline { { \mathbf { A v g . } } } } }$ </td></tr><tr><td colspan="6">(a) Impact of Input Modalities</td></tr><tr><td>1</td><td> $\mathrm { g . \ ( M L P ) }$ </td><td> $0 . 5 8 7 \pm 0 . 0 3 3$ </td><td> $0 . 6 1 7 \pm 0 . 0 2 6$ </td><td> $0 . 6 5 7 \pm 0 . 0 3 6$ </td><td>0.620</td></tr><tr><td>2</td><td>h. (ABMIL)</td><td> $0 . 6 1 4 \pm 0 . 0 3 7$ </td><td> $0 . 5 9 6 \pm 0 . 0 6 7$ </td><td> $0 . 6 4 9 \pm 0 . 0 2 8$ </td><td>0.620</td></tr><tr><td>3</td><td> $\mathrm { h . + g . \ ( C o n c a t ) }$ </td><td> $0 . 6 3 0 \pm 0 . 0 2 1$ </td><td> $0 . 6 3 2 \pm 0 . 0 4 7$ </td><td> $0 . 6 8 9 \pm 0 . 0 3 2$ </td><td>0.650</td></tr><tr><td colspan="6">(b) Impact of Multi-scale Hierarchy</td></tr><tr><td>4</td><td> ${ \mathrm { S i n g l e } } \ S { \mathrm { c a l e } } \ ( 2 0 \times )$ </td><td> $0 . 6 6 3 \pm 0 . 0 2 4$ </td><td> $0 . 6 4 4 \pm 0 . 0 3 9$ </td><td> $0 . 7 4 3 \pm 0 . 0 3 6$ </td><td>0.683</td></tr><tr><td>5</td><td> $\operatorname { D u a l } \operatorname { S c a l e } \left( 5 \times 2 0 \times \right)$ </td><td> $0 . 6 6 1 \pm 0 . 0 3 7$ </td><td> $0 . 6 4 8 \pm 0 . 0 3 1$ </td><td> $0 . 7 5 2 \pm 0 . 0 4 2$ </td><td>0.687</td></tr><tr><td>6</td><td> $\mathrm { F u l l ~ H i e r a r c h y } \left( 5 \times 1 0 \times 2 0 \times \right)$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr><tr><td colspan="6">(c) Impact of Anchor Fusion Mechanism</td></tr><tr><td>7</td><td>w/ Cross-Attention (TransMIL-style)</td><td> $0 . 6 6 3 \pm 0 . 0 1 3$ </td><td> $0 . 6 5 7 \pm 0 . 0 3 5$ </td><td> $0 . 7 4 6 \pm 0 . 0 3 3$ </td><td>0.689</td></tr><tr><td>8</td><td>w/ Bilinear Pooling</td><td> $0 . 6 5 0 \pm 0 . 0 3 5$ </td><td> $0 . 6 5 0 \pm 0 . 0 3 6$ </td><td> $0 . 7 4 5 \pm 0 . 0 3 0$ </td><td>0.682</td></tr><tr><td colspan="6">(d) Impact of Hierarchical Expert Mechanism</td></tr><tr><td></td><td>Stage 1: Intra-scale Selection Strategy</td><td></td><td></td><td></td><td></td></tr><tr><td>9 10</td><td>w/ Global Average Pooling (No Selection)</td><td> $0 . 6 5 4 \pm 0 . 0 3 1$ </td><td> $0 . 6 5 3 \pm 0 . 0 3 2$ </td><td> $0 . 7 5 1 \pm 0 . 0 2 5$ </td><td>0.686</td></tr><tr><td></td><td>w/ Attention Pooling (Soft Selection)</td><td> $0 . 6 6 7 \pm 0 . 0 5 4$ </td><td> $0 . 6 5 7 \pm 0 . 0 3 0$ </td><td> $0 . 7 4 8 \pm 0 . 0 2 0$ </td><td>0.691</td></tr><tr><td colspan="6">Stage 2: Inter-scale Fusion Strategy</td></tr><tr><td>11</td><td>w/ Linear Concatenation (No MoE)</td><td> $0 . 6 7 3 \pm 0 . 0 2 6$ </td><td> $0 . 6 5 6 \pm 0 . 0 1 8$ </td><td> $0 . 7 3 3 \pm 0 . 5 1$ </td><td>0.687</td></tr><tr><td>12</td><td>AMF-MES (Top-K Selection + MoE Routing)</td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr><tr><td colspan="6">(e) Impact of Different Inter-scale Alignment Strategies</td></tr><tr><td>13</td><td>No Alignment (Global Pooling)</td><td> $0 . 6 5 1 \pm 0 . 0 3 2$ </td><td> $0 . 6 8 0 \pm 0 . 0 2 8$ </td><td> $0 . 7 3 4 \pm 0 . 0 2 9$ </td><td>0.688</td></tr><tr><td>14</td><td>Original Order (w/o Sorting)</td><td> $0 . 6 5 7 \pm 0 . 0 3 5$ </td><td> $0 . 6 7 3 \pm 0 . 0 2 9$ </td><td> $0 . 7 5 0 \pm 0 . 0 3 1$ </td><td>0.693</td></tr><tr><td>15</td><td>Random Alignment</td><td> $0 . 6 7 0 \pm 0 . 0 3 3$ </td><td> $0 . 6 5 2 \pm 0 . 0 2 8$ </td><td> $0 . 7 4 1 \pm 0 . 0 3 0$ </td><td>0.688</td></tr><tr><td>16</td><td>Rank-based Alignment (Ours)</td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr></table>

fled before fusion, intentionally disrupting the rank-based risk hierarchy while preserving the feature distribution. As shown in Table 3, both Original Order and Random Alignment consistently reduce performance across cohorts, supporting meaningful semantic correspondence among same-ranked descriptors across scales. Our rank-based alignment captures these cross-scale correlations without additional learnable parameters or computational overhead.

Table 4: Sensitivity analysis of anchor number K across LUAD, UCEC, and BRCA datasets.
<table><tr><td rowspan="2">K</td><td colspan="3">Dataset (C-index)</td><td rowspan="2">Avg.</td></tr><tr><td>BRCA</td><td>LUAD</td><td>UCEC</td></tr><tr><td>6</td><td> $0 . 6 4 7 \pm 0 . 0 2 3$ </td><td> $0 . 6 6 0 \pm 0 . 0 2 6$ </td><td> $0 . 7 3 4 \pm 0 . 0 4 3$ </td><td>0.680</td></tr><tr><td>8</td><td> $0 . 6 6 9 \pm 0 . 0 1 4$ </td><td> $0 . 6 6 0 \pm 0 . 0 2 0$ </td><td> $0 . 7 4 0 \pm 0 . 0 1 6$ </td><td>0.690</td></tr><tr><td>16</td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr><tr><td>32</td><td> $0 . 6 7 4 \pm 0 . 0 1 4$ </td><td> $0 . 6 7 8 \pm 0 . 0 2 9$ </td><td> $0 . 7 4 8 \pm 0 . 0 3 6$ </td><td>0.700</td></tr></table>

(e) Impact of Different Inter-scale Alignment Strategies. To validate our inter-scale alignment strategy, we compare it with three alternatives: 1) No Alignment (Global Pooling): Each scale is independently compressed using global average pooling, thereby discarding fine-grained correspondence among cross-scale descriptors. 2) Original Order (w/o Sorting): The top-K<sup>′</sup> highest-scoring descriptors are retained but preserve their original index order, i.e., index j corresponds to its original unsorted identifier k. This setting directly isolates the contribution of the proposed descending risk-based sorting mechanism. 3) Random Alignment: Descriptor indices at one scale are randomly shuf-

Table 5: Sensitivity analysis of the retained top instances number $K ^ { \prime }$ across BRCA, LUAD, and UCEC datasets.
<table><tr><td rowspan="2"> $K ^ { \prime }$ </td><td colspan="3">Dataset (C-index)</td><td rowspan="2">Avg.</td></tr><tr><td>BRCA</td><td>LUAD</td><td>UCEC</td></tr><tr><td>2</td><td> $0 . 6 5 9 \pm 0 . 0 3 6$ </td><td> $0 . 6 5 3 \pm 0 . 0 2 7$ </td><td> $0 . 7 5 6 \pm 0 . 0 1 2$ </td><td>0.689</td></tr><tr><td>4</td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr><tr><td>8</td><td> $0 . 6 6 9 \pm 0 . 0 2 0$ </td><td> $0 . 6 5 3 \pm 0 . 0 3 1$ </td><td> $0 . 7 4 7 \pm 0 . 0 2 4$ </td><td>0.690</td></tr><tr><td>16</td><td> $0 . 6 7 1 \pm 0 . 0 1 8$ </td><td> $0 . 6 6 6 \pm 0 . 0 3 1$ </td><td> $0 . 7 4 4 \pm 0 . 0 2 3$ </td><td>0.694</td></tr></table>

Table 6: Ablation on the anchor regularization loss $( \mathcal { L } _ { \mathrm { a n c h o r } } )$
<table><tr><td rowspan="2">Setting</td><td colspan="3">Dataset</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>BRCA</td><td>LUAD</td><td>UCEC</td></tr><tr><td rowspan="2"> $\mathbf { w } / \mathbf { o } ~ \mathcal { L } _ { \mathrm { a n c h o r } }$  w/  $\mathcal { L } _ { \mathrm { a n c h o r } }$ </td><td> $0 . 6 7 8 \pm 0 . 0 2 0$ </td><td> $0 . 6 7 5 \pm 0 . 0 2 8$ </td><td> $0 . 7 4 6 \pm 0 . 0 3 3$ </td><td>0.700</td></tr><tr><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 0 . 6 8 7 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 0 . 7 6 2 \pm 0 . 0 2 8 }$ </td><td>0.714</td></tr></table>

## 4.3.2 Hyperparameter Sensitivity

To determine the optimal settings for our model, we investigate the effects of key hyperparameters within our model and structural constraints: the number of dynamic anchors (K) in the prototype bridge, the number of retained instances per scale $K ^ { \prime }$ , and the necessity of anchor regularization.

![](images/32ed6b85ba10dd4431c3267e0c274914eac9f7a2d4743d1e853d4fa5527170c2.jpg)  
Figure 5: Visualization of multimodal interpretability for glioma survival prediction. This compares a high-risk GBM patient (Left. TCGA-06. $0 1 \breve { 4 1 } .$ survival 10.28 months) and a low-risk ODG patient (Right, TCGA-S9-A7R1, survival 169.71 months). For each patient, the visualization comprises: (1) The Whole Slide Image (WSI) overlaid with spatial attention heatmaps, highlighting regions contributing most to the prediction; (2) Representative pathological patches extracted from these high-attention regions across three distinct magnification scales (as labeled: 5×, 10×, and 20×), capturing both macroscopic tumor architecture and microscopic cellular details; and (3) The top 25 genomic features ranked by importance using Integrated Gradients, where red and blue bars indicate positive and negative contributions to the risk prediction, respectively.

Impact of Anchor Quantity (K). The anchors act as a semantic bottleneck to align high-dimensional pixel data with gene expression. We tested $\mathrm { ~ \bar { ~ } { ~ K ~ } ~ } \in \{ 6 , 8 , 1 6 , 3 \bar { 2 } \}$ . Results in Table 4 indicate that performance improves significantly from $K = 6$ to $K = 1 6 .$ . As observed, a smaller K fails to capture the full spectrum of histological subtypes $( e . g .$ , endometrioid vs. serous). However, increasing $\dot { K }$ to 32 yields diminishing returns. Therefore, $K = 1 6$ provides a sufficient semantic basis for the cross-modal attention bridge while preventing overfitting to patient-specific noise.

Impact of Retained Instances (K<sup>′</sup>). This parameter controls the number of most informative patches (instances) selected from each visual scale during the intra-scale selection stage. These selected instances are subsequently fed into the cross-modal fusion module. We evaluate the model with $K ^ { \prime } \in \{ 2 , 4 , 8 , 1 6 \}$ to determine the optimal information bottleneck. As presented in Table 5, the performance initially improves and peaks at $K ^ { \prime } = 4$ . When $\bar { K } ^ { \prime } = 2 ,$ , the selection is too restrictive, potentially discarding valuable morphological context $( \mathrm { e . g . }$ , tumor margins, immune infiltrates) that is crucial for accurate prognosis. At $K ^ { \prime } = 4 ,$ , the model achieves the optimal trade-off, successfully capturing the most distinctive pathological patterns without overwhelming the fusion router. However, as $K ^ { \prime }$ increases to $8$ or 16, we observe a slight degradation in performance. This indicates that retaining an excessive number of instances inevitably introduces noisy or uninformative patches (such as background, healthy tissue, or necrosis), which dilutes the attention mechanism and misguides the subsequent genomic alignment. Consequently, we set $K ^ { \prime } = 4$ as the default setting to ensure a pure and highly discriminative visual representation.

Impact of Anchor Regularization. To prevent the learnable anchors from suffering mode collapse or overfitting to noise, we introduce a composite regularization objective $\mathcal { L } _ { \mathrm { a n c h o r } }$ (detailed in Sec. 3.4.3). This objective enforces both orthogonality and sparsity to ensure that the anchors capture diverse and compact biological concepts. To validate the overall necessity of this structural constraint, we evaluate the model by completely removing the $\mathcal { L } _ { \mathrm { a n c h o r } }$ during training. As shown in Table 6, omitting the regularization leads to a noticeable performance drop, with the average Cindex decreasing from 0.714 to 0.700. This confirms that explicitly constraining the anchor space is crucial for learning distinctive representations, ultimately preventing semantic collapse during multi-modal fusion.

## 4.4 Interpretability and Case Studies

To better understand the decision-making process of the proposed multimodal framework, we conduct interpretability analyses for both imaging and genomic modalities. Specifically, spatial attention weights are visualized on whole-slide images (WSIs) to highlight the regions most influential to survival prediction, while feature importance for genomic inputs is quantified using Integrated Gradients (IG). Fig. 5 presents a comparative case study of two glioma patients with markedly different clinical outcomes: a highrisk glioblastoma (GBM) patient and a low-risk oligodendroglioma (ODG) patient.

## 4.4.1 Morphological Attention Analysis

As shown in Fig. 6, the spatial attention maps reveal distinct morphological patterns across multiple magnification levels that are consistent with known histopathological characteristics of different glioma subtypes. For the GBM patient (TCGA-06-0141, left panel), the attention heatmaps are primarily concentrated within tumor-dense regions of the WSI. Representative patches extracted from these regions illustrate disrupted tissue architecture and increased cellular density at lower magnifications (5× and 10×). At higher magnification (20×), the highlighted regions exhibit morphological features suggestive of high-grade glioma pathology, including pronounced nuclear atypia and heterogeneous cellular morphology. These observations are consistent with the aggressive histological phenotype commonly observed in glioblastoma. In contrast, for the ODG patient (TCGA-S9-A7R1, right panel), the attention maps display a more distributed spatial pattern across tumor regions. Across magnification scales, the selected patches consistently exhibit morphological characteristics typical of oligodendroglial tumors. At higher magnification, the model focuses on regions containing relatively uniform round nuclei with subtle perinuclear halos, corresponding to the well-known “fried-egg” appearance associated with oligodendroglioma. Such morphological patterns are generally associated with more favorable clinical outcomes compared with high-grade gliomas.

![](images/162fa2505b08ccb7b9668a9a403cac5132d94d3f2f0d054e385a8e3cea995f2c.jpg)  
Figure 6: Global feature importance analysis across five diverse TCGA pan-cancer datasets. The horizontal bar charts display the top-ranked genomic features (comprising both RNA-seq expression and Copy Number Variations) for each respective cohort, quantified by their overall importance to the model’s survival predictions. The identification of completely distinct, disease-specific biomarker signatures across different cohorts demonstrates the model’s robust capacity to capture highly generalized, biologically relevant molecular pathways across varied oncology domains.

Overall, these multi-scale visualizations suggest that the proposed model captures clinically meaningful histopathological patterns that align with known subtype-specific glioma morphology. Importantly, these patterns emerge without requiring pixel-level annotations, demonstrating the capability of the model to implicitly learn prognostically relevant morphological representations from weakly supervised WSI data.

## 4.4.2 Importance of Genomic Features

To further interpret the molecular component of the model, we apply Integrated Gradients (IG) to quantify the contribution of transcriptomic features to the predicted survival risk. The horizontal bar charts in Fig. 6 display the top 25 genes ranked by attribution scores. Positive attribution values (red bars) indicate features that increase the predicted hazard risk (poorer prognosis), whereas negative attribution values (blue bars) correspond to features that reduce the predicted risk. As observed in the high-risk GBM patient, genes such as SPP1 (Secreted Phosphoprotein 1, also known as osteopontin) emerge as strong positive contributors to the predicted risk score. This observation is consistent with previous studies [53] reporting that SPP1 is frequently overexpressed in gliomas and is associated with tumor progression, angiogenesis, and unfavorable survival outcomes. In contrast, for the low-risk ODG patient, the model assigns negative attribution values to immune-related genes such as PRF1 (Perforin 1). PRF1 is a key effector molecule for cytotoxic T cells and natural killer (NK) cells, and its negative risk attribution may reflect transcriptional programs associated with enhanced anti-tumor immune responses. Overall, these attribution patterns suggest that the model captures biologically meaningful molecular signals consistent with known glioma biology.

## 4.4.3 Multimodal Synergy

The joint analysis of pathological and genomic explanations demonstrates that our model leverages truly complementary, mutually reinforcing information. In the high-risk GBM case, the visual encoder pinpoints WSI regions of aggressive tissue remodeling and dense cellularity across multiple scales. Concurrently, the genomic branch highlights the exact molecular underpinnings driving these morphological phenotypes—such as the massive upregulation of SPP1, a gene notorious for mediating tumor invasiveness and macrophage recruitment. Conversely, in the ODG case, WSI regions displaying lower-grade oligodendroglial morphology are evaluated alongside a genome-level protective immune signature (e.g., PRF1). This synergy, where the visual modality provides the spatial manifestation of tumor burden while the genomic modality uncovers the functional drivers, enables a significantly more robust and deeply interpretable survival prediction compared to unimodal approaches.

## 4.4.4 Importance ofGlobal Pan-Cancer Features

To further investigate the cohort-level biological signals captured by the model, we analyze the importance of global genomic features across five TCGA datasets (as shown in Fig. 6). For each cohort, we aggregate attribution scores to identify the molecular features—spanning both RNA-seq expression and copy number variations (CNVs)—that most strongly influence the predicted survival hazard.

Consistency with Local Explanations (GBMLGG): Consistent with the patient-level case studies presented in Sec. 4.4, the global analysis of the GBMLGG cohort identifies SPP1 as the top-ranked prognostic feature across the entire dataset. This finding aligns with previous studies reporting that SPP1 is strongly associated with glioma progression, tumor invasion, and poor clinical outcomes. Other highly ranked features, such as NAMPT and $C D K 4 \_ c n v ,$ further reflect key molecular processes underlying glioma biology, including metabolic reprogramming and cell-cycle dysregulation.

Multimodal Molecular Signals: The presence of both transcriptomic features $( e . g . \mathrm { , }$ , SPP1, NAMPT) and genomic alterations $( e . g . , C D K 4 \_ c n v )$ among the top-ranked predictors illustrates the model’s ability to integrate heterogeneous molecular modalities when estimating survival risk. This multimodal feature prioritization suggests that the model captures complementary biological signals spanning both gene expression dynamics and genomic structural alterations.

Cancer-Type–Specific Molecular Patterns: Importantly, the dominant molecular features vary substantially across different cancer cohorts. For example, the LUAD cohorts highlight regulators involved in signaling and transcriptional control (e.g., TRIM26 and MAP2K2), while the UCEC cohort prioritizes genes related to epigenetic regulation and cellular signaling pathways $( e . g . ,$ ASXL1 and EXT1). These observations suggest that the model dynamically adapts its feature attribution to the biological context of each cancer type, rather than relying on a fixed set of global biomarkers.

Overall, the analysis demonstrates that the proposed framework captures biologically meaningful molecular signals across diverse cancer types while maintaining the flexibility to emphasize disease-specific prognostic drivers.

## 4.5 Discussion of Computational Complexity

Our model incorporates multi-scale processing and hierarchical routing, which could otherwise incur substantial computational costs. Direct self-attention over tens of thousands of unaligned patches from three magnifications $( 5 \times , 1 0 \times , 2 0 \times )$ has prohibitive quadratic complexity of $O ( M _ { i } ^ { 2 } )$ , where $M _ { i } = \mathbf { \hat { \mu } } \mathbf { \hat { \Sigma } } \mathbf { \hat { \Sigma } } M _ { i } ^ { s }$ denotes the total number of patches for patient i. To address this issue, we employ a two-stage sequence reduction strategy (Section 3.3.1). First, a Transformer-based soft-clustering module uses a fixed number of learnable prototype queries $( \mathbf { Q } ^ { s } \in \mathbb { R } ^ { P ^ { s } \times D } )$ , to aggregate $M _ { i } ^ { s }$ patch features into $\bar { P } ^ { s }$ semantic prototypes at each scale via a multi-head cross-attention (MHCA). Since $P ^ { s } \ll M _ { i } ^ { s }$ the aggregation complexity is reduced from $O ( ( M _ { i } ^ { s } ) ^ { 2 } )$ to $O ( P ^ { s } \cdot \dot { M } _ { i } ^ { s } )$ , which scales linearly with the number of patches. The Anchor Bridge then maps these condensed multi-scale prototypes to a compact set of global anchors. This early sequence reduction enables scalable processing of ultra-large WSIs while retaining coarse-tofine structural information. For hierarchical routing, we replace dense fusion with sparse, multi-level MoE routing. Adaptive Top-K gating activates only relevant experts at each level, facilitating rich cross-modal and multi-scale interactions while maintaining bounded inference costs. To evaluate computational efficiency, we benchmark our model against three representative methods, including MoCAT [8], HiMT [12], and UMSA [13], on a single NVIDIA GeForce RTX 4090 GPU with 24 GB of memory. As reported in Table 7, our model contains 12.66M parameters and achieves an inference time of 9.5 ms per WSI, offering a favorable trade-off between computational efficiency and predictive performance.

Table 7: Comparison of computational complexity and efficiency.
<table><tr><td>Method</td><td>(M)</td><td>(G)</td><td>(MB)</td><td>Params. FLOPs Peak Memory Inference Time (ms/WSI)</td></tr><tr><td>MoCAT [8]</td><td>3.90</td><td>1.83</td><td>123.93</td><td>60.3</td></tr><tr><td>HiMT [12]</td><td>5.48</td><td>2.41</td><td>156.39</td><td>7.7</td></tr><tr><td>UMSA [13]</td><td>31.81</td><td>40.76</td><td>553.75</td><td>72.7</td></tr><tr><td>Ours</td><td>12.66</td><td>4.99</td><td>190.16</td><td>9.5</td></tr></table>

## 5 CONCLUSION

In this paper, we propose $\mathrm { A M ^ { 2 } E S } ,$ , a novel framework for multi-modal cancer survival prediction. AM<sup>2</sup>ES employs dynamic semantic anchors to mediate the integration of transcriptomic profiles and multi-scale pathological features, thereby facilitating cross-modal alignment. It further incorporates a H-MoE module that decouples phenotypic analysis across tissue scales. Specifically, H-MoE first selects informative features within each scale and then adaptively integrates evidence across scales for each patient. Experiments on multiple cancer cohorts show that AM<sup>2</sup>ES consistently outperforms state-of-the-art methods while providing fine-grained interpretability of its decision process. Ablation studies further confirm the contribution of each key component.

## REFERENCES

[1] J. Lipkova, R. J. Chen, B. Chen, M. Y. Lu, M. Barbieri, D. Shao, A. J. Vaidya, C. Chen, L. Zhuang, D. F. Williamson et al., “Artificial intelligence for multimodal data integration in oncology,” Cancer cell, vol. 40, no. 10, pp. 1095–1110, 2022.

[2] R. J. Chen, M. Y. Lu, W.-H. Weng, T. Y. Chen, D. F. Williamson, T. Manz, M. Shady, and F. Mahmood, “Multimodal co-attention transformer for survival prediction in gigapixel whole slide images,” in Proc. IEEE Conf. Comput. Vis., 2021, pp. 4015–4025.

[3] X. Han, H. Zhou, Z. Tian, S. Du, and Y. Gao, “Inter-intra hypergraph computation for survival prediction on whole slide images,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 6006–6021, 2025.

[4] B. Li, Y. Li, and K. W. Eliceiri, “Dual-stream multiple instance learning network for whole slide image classification with selfsupervised contrastive learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 14 318–14 328.

[5] D. Di, C. Zou, Y. Feng, H. Zhou, R. Ji, Q. Dai, and Y. Gao, “Generating hypergraph-based high-order representations of whole-slide histopathological images for survival prediction,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 5, pp. 5800–5815, 2022.

[6] R. Yan, X. Zhang, Z. Jiang, B. Wang, X. Bian, F. Ren, and S. K. Zhou, “Pathway-aware multimodal transformer (PAMT): Integrating pathological image and gene expression for interpretable cancer survival analysis,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 896–913, 2025.

[7] R. J. Chen, M. Y. Lu, W.-H. Weng, T. Y. Chen, D. F. Williamson, T. Manz, M. Shady, and F. Mahmood, “Multimodal co-attention transformer for survival prediction in gigapixel whole slide images,” in Proc. IEEE Conf. Comput. Vis., 2021, pp. 4015–4025.

[8] Y. Xu and H. Chen, “Multimodal optimal transport-based coattention transformer with global structure consistency for survival prediction,” in Proc. IEEE Conf. Comput. Vis., 2023, pp. 21 241– 21 251.

[9] N. Hashimoto, D. Fukushima, R. Koga, Y. Takagi, K. Ko, K. Kohno, M. Nakaguro, S. Nakamura, H. Hontani, and I. Takeuchi, “Multiscale domain-adversarial multiple-instance cnn for cancer subtype classification with unannotated histopathological images,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020, pp. 3852–3861.

[10] B. Li, Y. Li, and K. W. Eliceiri, “Dual-stream multiple instance learning network for whole slide image classification with selfsupervised contrastive learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 14 318–14 328.

[11] P. Pati, G. Jaume, A. Foncubierta-Rodriguez, F. Feroce et al., “Hierarchical graph representations in digital pathology,” Medical Image Analysis, vol. 75, p. 102264, 2022.

[12] C. Li, X. Zhu, J. Yao, and J. Huang, “Hierarchical transformer for survival prediction using multimodality whole slide images and genomics,” in Proc. IEEE Conf. Pattern Recognit. IEEE, 2022, pp. 4256–4262.

[13] S. Jiang, L. Cai, Z. Gan, Y. Wang, G. Tang, and Y. Zhang, “Uncertainty-aware survival analysis with dirichlet distribution for multi-scale pathology and genomics,” IEEE Trans. Med. Imaging, vol. 45, no. 2, pp. 554–568, 2025.

[14] N. Nikolaou, D. Salazar, H. RaviPrakash, M. Gonc¸alves, R. Mulla, N. Burlutskiy, N. Markuzon, and E. Jacob, “A machine learning approach for multimodal data fusion for survival prediction in cancer patients,” NPJ Precision Oncology, vol. 9, no. 1, p. 128, 2025.

[15] Z. Fan, Z. Jiang, H. Liang, and C. Han, “Pancancer survival prediction using a deep learning architecture with multimodal representation and integration,” Bioinformatics Advances, vol. 3, no. 1, p. vbad006, 2023.

[16] P. Mobadersany, S. Yousefi, M. Amgad, D. A. Gutman, J. S. Barnholtz-Sloan, J. E. Velazquez Vega, D. J. Brat, and L. A. Cooper,´ “Predicting cancer outcomes from histology and genomics using convolutional networks,” Proceedings of the National Academy of Sciences, vol. 115, no. 13, pp. E2970–E2979, 2018.

[17] R. J. Chen, M. Y. Lu, J. Wang, D. F. Williamson, S. J. Rodig, N. I. Lindeman, and F. Mahmood, “Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis,” IEEE Trans. Med. Imaging, vol. 41, no. 4, pp. 757–770, 2020.

[18] Y. Xu and H. Chen, “Multimodal optimal transport-based coattention transformer with global structure consistency for survival prediction,” in Proc. IEEE Conf. Comput. Vis., 2023, pp. 21 241– 21 251.

[19] H. Paverd, K. Zormpas-Petridis, H. Clayton, S. Burge, and M. Crispin-Ortuzar, “Radiology and multi-scale data integration for precision oncology,” NPJ Precision Oncology, vol. 8, no. 1, p. 158, 2024.

[20] Z. Wang, J. Li, Z. Pan, W. Li, A. Sisk, H. Ye, W. Speier, and C. W. Arnold, “Hierarchical graph pathomic network for progression free survival prediction,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assisted Intervent. Springer, 2021, pp. 227–237.

[21] J. Wu, X. Ke, X. Jiang, H. Wu, Y. Kong, and L. Shao, “Leveraging tumor heterogeneity: Heterogeneous graph representation learning for cancer survival prediction in whole slide images,” Proc. Adv. Neural Inf. Process. Syst., vol. 37, pp. 64 312–64 337, 2024.

[22] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in Proc. Int. Conf. Mach. Learn. PMLR, 2018, pp. 2127–2136.

[23] R. J. Chen, M. Y. Lu, M. Shaban, C. Chen, T. Y. Chen, D. F. Williamson, and F. Mahmood, “Whole slide images are 2d point clouds: Context-aware survival prediction using patch-based graph convolutional networks,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assisted Intervent. Springer, 2021, pp. 339–349.

[24] Z. Shao, H. Bian, Y. Chen, Y. Wang, J. Zhang, X. Ji et al., “Transmil: Transformer based correlated multiple instance learning for whole slide image classification,” Proc. Adv. Neural Inf. Process. Syst., vol. 34, pp. 2136–2147, 2021.

[25] R. J. Chen, C. Chen, Y. Li, T. Y. Chen, A. D. Trister, R. G. Krishnan, and F. Mahmood, “Scaling vision transformers to gigapixel images via hierarchical self-supervised learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 16 144–16 155.

[26] M. Han, L. Qu, D. Yang, X. Zhang, X. Wang, and L. Zhang, “Mscpt: Few-shot whole slide image classification with multi-scale and context-focused prompt tuning,” IEEE Trans. Med. Imaging, vol. 44, no. 9, pp. 756–3769, 2025.

[27] M. Ning, D. Lu, Y. Xie, D. Chen, D. Wei, Y. Zheng, Y. Tian, S. Yan, and L. Yuan, “MADAv2: Advanced multi-anchor based active domain adaptation segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 13 553–13 566, 2023.

[28] Q. Ming, Z. Zhou, L. Miao, H. Zhang, and L. Li, “Dynamic anchor learning for arbitrary-oriented object detection,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 3, 2021, pp. 2355–2363.

[29] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Proc. Eur. Conf. Comput. Vis. Springer, 2020, pp. 213–229.

[30] M. Jeong, Z. M. Kim, M. Namgung, D. Kang, Y.-Y. Chiang, and A. Hero, “Anchors aweigh! sail for optimal unified multi-modal representations,” arXiv preprint arXiv:2410.02086, 2024.

[31] S. Li, H. Gong, H. Dong, T. Yang, Z. Tu, and Y. Zhao, “Dpu: Dynamic prototype updating for multimodal out-of-distribution detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 10 193–10 202.

[32] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.

[33] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal ofMachine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[34] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling vision with sparse mixture of experts,” Proc. Adv. Neural Inf. Process. Syst., vol. 34, pp. 8583–8595, 2021.

[35] H. Li, Y. Chen, Y. Chen, R. Yu, W. Yang, L. Wang, B. Ding, and Y. Han, “Generalizable whole slide image classification with finegrained visual-semantic interaction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 11 398–11 407.

[36] S. Jiang, T. Zheng, Y. Zhang, Y. Jin, L. Yuan, and Z. Liu, “Medmoe: Mixture of domain-specific experts for lightweight medical vision-language models,” arXiv preprint arXiv:2404.10237, 2024.

[37] A. Liberzon, C. Birger, H. Thorvaldsdottir, M. Ghandi, J. P.´ Mesirov, and P. Tamayo, “The molecular signatures database hallmark gene set collection,” Cell Systems, vol. 1, no. 6, pp. 417–425, 2015.

[38] R. J. Chen, T. Ding, M. Y. Lu, D. F. Williamson, G. Jaume, A. H. Song, B. Chen, A. Zhang, D. Shao, M. Shaban et al., “Towards a general-purpose foundation model for computational pathology,” Nature Medicine, vol. 30, no. 3, pp. 850–862, 2024.

[39] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood, “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nature biomedical engineering, vol. 5, no. 6, pp. 555–570, 2021.

[40] G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, “Selfnormalizing neural networks,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30. Curran Associates, Inc., 2017.

[41] S. Haykin, Neural networks: a comprehensive foundation. Prentice hall PTR, 1994.

[42] F. Zhou and H. Chen, “Cross-modal translation and alignment for survival analysis,” in Proc. IEEE Conf. Comput. Vis., 2023, pp. 21 485–21 494.

[43] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in Proceedings of the 35th Proc. Int. Conf. Mach. Learn., vol. 80, 2018, pp. 2127–2136.

[44] Z. Shao, H. Bian, Y. Chen, Y. Wang, J. Zhang, X. Ji, and y. zhang, “TransMIL: Transformer based correlated multiple instance learning for whole slide image classification,” in Proc. Adv. Neural Inf. Process. Syst., M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 2136–2147.

[45] J. Li, Y. Chen, H. Chu, Q. Sun, T. Guan, A. Han, and Y. He, “Dynamic graph representation with knowledge-aware attention for histopathology whole slide image analysis,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 11 323–11 332.

[46] R. Deng, C. Cui, L. W. Remedios, S. Bao, R. M. Womick, S. Chiron, J. Li, J. T. Roland, K. S. Lau, Q. Liu et al., “Cross-scale multiinstance learning for pathological image diagnosis,” Medical Image Analysis, vol. 94, p. 103124, 2024.

[47] R. J. Chen, C. Chen, Y. Li, T. Y. Chen, A. D. Trister, R. G. Krishnan, and F. Mahmood, “Scaling vision transformers to gigapixel images via hierarchical self-supervised learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 16 144–16 155.

[48] R. J. Chen, M. Y. Lu, D. F. Williamson, T. Y. Chen, J. Lipkova, Z. Noor, M. Shaban, M. Shady, M. Williams, B. Joo et al., “Pancancer integrative histology-genomic analysis via multimodal deep learning,” Cancer Cell, vol. 40, no. 8, pp. 865–878, 2022.

[49] G. Jaume, A. Vaidya, R. J. Chen, D. F. Williamson, P. P. Liang, and F. Mahmood, “Modeling dense multimodal interactions between biological pathways and histology for survival prediction,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 11 579– 11 590.

[50] H. Li, F. Yang, X. Xing, Y. Zhao, J. Zhang, Y. Liu, M. Han, J. Huang, L. Wang, and J. Yao, “Multi-modal multi-instance learning using weakly correlated histopathological images and tabular clinical information,” in Proc. Int. Conf. Med. Image Comput. Comput.- Assisted Intervent. Springer, 2021, pp. 529–539.

[51] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC), “The clinical proteomic tumor analysis consortium uterine corpus endometrial carcinoma collection (CPTAC-UCEC)(version 13) [dataset],” 2019. [Online]. Available: https://doi.org/10.7937/k9/tcia.2018.3r3juisw

[52] ——, “The clinical proteomic tumor analysis consortium lung adenocarcinoma collection (cptac-luad)(version 13) [dataset],” 2018, dataset. [Online]. Available: https://doi.org/10.7937/k9/ tcia.2018.pat12tbs

[53] K. Wang, J. Wan, R. Zheng, Y. Xiao, F. Lv, H. Ge, G. Yang, and Y. Cheng, “Spp1 as a prognostic and immunotherapeutic biomarker in gliomas and other cancer types: a pan-cancer study,” Journal of Inflammation Research, pp. 2247–2265, 2025.