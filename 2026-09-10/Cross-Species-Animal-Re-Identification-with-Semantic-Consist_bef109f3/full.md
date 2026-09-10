# Cross-Species Animal Re-Identification with Semantic Consistency Learning

Shuoyi Chen<sup>\*</sup>, Yuejia Li<sup>\*</sup>, and Mang Ye<sup>†</sup>

School of Computer Science, Wuhan University, Wuhan, China {chenshuoyi,liyuejia,yemang}@whu.edu.cn

Abstract. Generalizable animal Re-Identification (ReID) aims to recognize individual animals across species with diverse morphologies and ecological contexts. Unlike person ReID, where diferent domains share similar body structures, animal species often exhibit drastically diferent anatomical structures and visual patterns, making it dificult to establish shared visual correspondences. As a result, representations learned across species tend to form fragmented embedding spaces, which severely limits cross-species generalization. To address this challenge, we propose Semantic Consistency Learning (SCL), a framework designed to learn representations that remain stable across appearance variations while preserving semantic structures shared across species. SCL consists of two complementary components. Foreground–Background Decoupled Spectral Normalization (FDSNorm) stabilizes feature statistics by suppressing environment-induced style variations in a region-aware manner, while Cross-species Neighborhood Modeling (CNM) captures transferable relational structures across species through dynamic feature neighborhoods. Extensive experiments on 11 public animal ReID datasets demonstrate that SCL consistently outperforms state-of-the-art methods under multiple cross-species evaluation protocols and generalizes efectively to previously unseen species and ecological domains. Code is available at https://github.com/Kemalau/ECCV-26-SCL.

Keywords: Animal Re-Identification · Domain Generalization

## 1 Introduction

Animal re-identification (ReID) aims to recognize the same animal individual across diferent times, viewpoints, and environmental conditions [20,48]. It plays a critical role in ecological monitoring, wildlife conservation, and long-term population analysis. While remarkable progress has been achieved in person and vehicle ReID [9, 10, 31], these tasks are typically studied in relatively structured environments where objects share consistent geometric layouts and appearance statistics. In contrast, animal ReID operates in open-world scenarios that involve diverse species, complex habitats, and highly varying visual characteristics. Animals from diferent species exhibit substantial variations in body morphology, texture patterns, and environmental context. These factors introduce severe distribution shifts that go far beyond the viewpoint or illumination changes commonly addressed in human-centered ReID tasks. As a result, developing models that can generalize across species remains a fundamental challenge.

![](images/c042f991c3371d61aeb96de4db18ece296cab7fc020f3c1e5cfeb22a64874b51.jpg)  
Fig. 1: Multi-level domain discrepancies in animal ReID, including individual, breed, and species level variations beyond viewpoint and pose changes.

Most existing animal ReID studies focus on a single-species setting, where models are trained and evaluated using data from one specific species [22,26,34, 40]. Under this formulation, models can learn discriminative representations tailored to the appearance statistics of that species and achieve strong performance on curated benchmarks. However, such a paradigm limits model reusability in real-world applications. Each new species often requires collecting additional data, annotating identities, and retraining models. To address this limitation, recent work has begun to explore multi-species animal ReID, which aims to train a unified model using data from multiple species and generalize to unseen species or new ecological domains. Some approaches attempt to improve robustness by constructing larger and more diverse datasets [7, 18, 20], while others focus on extracting richer visual cues such as local discriminative patterns [33] or high-frequency information [25]. Although these methods improve recognition accuracy on known species, they mainly enhance instance-level discrimination within species and do not explicitly model how representations should be shared across species. Consequently, the learned embeddings often fail to generalize to unseen species or ecological domains.

In the person ReID community, domain generalization (DG) has been widely studied to address distribution shifts between source and target domains [19,23, 54]. Existing DG approaches typically aim to learn domain-invariant representations in order to improve cross-domain robustness. Representative techniques include feature disentanglement [29, 57], style normalization [21, 38], and metalearning [12, 37]. These methods are efective in mitigating variations caused by viewpoint, pose, and illumination changes. However, they implicitly rely on the assumption that objects across domains share consistent semantic structures. For example, human bodies exhibit stable part layouts and similar geometric configurations across domains. This assumption rarely holds in animal ReID.

As illustrated in Figure 1, animals present substantial discrepancies at the individual, breed, and species levels. Cross-species variations in body morphology and semantic structure are significantly larger than those encountered in person ReID. Therefore, existing DG methods struggle to learn transferable representations for multi-species animal ReID.

These limitations indicate that efective cross-species animal ReID requires representations that remain robust to appearance variations while preserving semantic structures shared across species. To this end, we propose a Semantic Consistency Learning (SCL) framework. The key idea is to stabilize appearance statistics while maintaining structural information in the learned representations. Specifically, to mitigate representation instability caused by environmental and species variations, we introduce Foreground–Background Decoupled Spectral Normalization (FDSNorm), a frequency-domain normalization mechanism that decouples foreground and background regions for adaptive spectral modulation. Unlike existing normalization strategies that suppress style variations globally in spatial or spectral spaces, our approach explicitly accounts for semantic diferences across regions. We preserve the original phase information while adaptively modulating amplitude spectra in diferent semantic regions, enabling structure-preserving style control under cross-species domain shifts.

However, stabilizing feature statistics alone does not explicitly model crossspecies relationships in the embedding space. To address this limitation, we introduce Cross-species Neighborhood Modeling (CNM), which captures relational structures across species through dynamic neighborhood construction. Specifically, CNM discovers mutual neighbors to form both intra-species and interspecies neighborhoods, enabling the model to learn from relational topology rather than relying solely on appearance similarity. This mechanism encourages the learning of structural regularities shared across species while preserving discriminative capability within each species. In summary, our main contributions are as follows:

We introduce Foreground–Background Decoupled Spectral Normalization (FDSNorm), a frequency-domain normalization mechanism that decouples foreground and background regions for adaptive spectral modulation. By preserving phase information while modulating amplitude spectra, FDSNorm suppresses appearance variations and stabilizes feature representations under cross-species domain shifts.

We propose Cross-species Neighborhood Modeling (CNM), which explicitly captures relational structures across species through mutual neighbor discovery. CNM dynamically constructs intra-species and inter-species neighborhoods, enabling the model to learn shared semantic regularities across species while preserving discriminative capability within each species.

– We introduce two complementary evaluation protocols for cross-species animal ReID to enable a comprehensive evaluation of generalization to unseen species. Extensive experiments on 11 public datasets demonstrate consistent improvements over competitive state-of-the-art methods.

## 2 Related Work

Object ReID. Re-Identification (ReID) has made substantial progress, with a large body of work focused on person and vehicle identification [8, 47, 51, 56]. This has produced many powerful methods, from strong convolutional baselines [32, 44] to more recent transformer-based [15] and vision-language [27] architectures. While these general-purpose frameworks are versatile, their standard implementation requires training a separate, species-specific model for each animal category when adapted for animal ReID. Concurrently, a distinct line of research has emerged that focuses specifically on the challenges of animal ReID. These methods are tailored to specific intra-species challenges, such as identifying livestock by coat patterns [3], re-identifying tigers by their unique stripe patterns [26], using high-frequency supervision for fine-grained details [25], or addressing pose variation with 3D models [55]. These domain-generalizable ReID methods rely on consistent body structures and shared semantic correspondences across domains. In cross-species animal ReID, drastic diferences in anatomy and visual patterns break this assumption, limiting their ability to learn transferable representations.

Domain Generalized ReID. DG ReID has a substantial literature, particularly in person ReID [19, 35], aiming to improve generalization under changes in scene, viewpoint, and illumination. Various approaches have been explored to achieve this. One line of work focuses on normalization-based methods [12, 21], which suppress camera/style statistics to preserve identity cues. Other strategies include employing Mixture-of-Experts (MoE) [13, 46] to structure domain variability. These models typically require predefined experts for known domains, making them ill-suited for unseen species. Methods using memory banks [28,43] stabilize matching across domains, but their eficacy diminishes in animal ReID due to high inter-species similarity and vast intra-species diversity, which can pollute the memory bank. Meta-learning frameworks [4, 54] have also been proposed to simulate test-time shifts. Their limitation lies in the simulation: the meta-tasks often simulate variations in viewpoint or illumination, failing to prepare the model for the drastic object-type shift encountered when generalizing to a new animal species. More recently, data-driven routes like BAU [11] emphasize augmentations, and CLIP-based methods [52, 53] leverage vision-language priors. While powerful, standard CLIP priors often capture species-level semantics rather than fine-grained individual identity, requiring significant adaptation. Training-free approaches such as Pose2ID [50] leverage pose priors at test time, but their reliance on structured human pose limits applicability to animal ReID.

Multi-species ReID. More recently, large-scale animal ReID foundation models such as MegaDescriptor [7] and MiewID [39] have been proposed, leveraging massive community-curated datasets to train unified embedding networks. UniReID [20] specifically adapted CLIP-based architectures to tackle domain generalized animal ReID and contributed the large-scale Wildlife71 dataset. Community benchmarks such as AnimalCLEF [2] further promote evaluation of individual animal recognition at scale. However, these studies mainly focus on constructing large-scale datasets and adopt conventional instance-level metric learning objectives commonly used in ReID. They do not explicitly design mechanisms for generalization to completely unseen species.

![](images/6241982a738faefe3067f7b94fcfa40ea0233770edd346f8667adbb03c09c722.jpg)  
Fig. 2: Overview of the proposed Semantic Consistency Learning framework. The upper part illustrates the Foreground–Background Decoupled Spectral Normalization, while the lower part presents Cross-species Neighborhood Modeling.

## 3 Method

## 3.1 Overview

The objective of multi-species animal ReID is to learn a unified representation that preserves individual-level discriminability while generalizing across species with diverse morphologies and ecological environments. However, jointly learning representations from heterogeneous species introduces substantial distributional discrepancies. Diferences in texture patterns, body structures, and environmental contexts lead to unstable feature statistics and hinder the formation of a coherent embedding space. As a result, representations learned from diferent species tend to cluster around species-specific appearance statistics rather than capturing transferable semantic structures.

Furthermore, many existing ReID approaches rely on alignment-based learning strategies that exploit shared visual cues or explicit correspondences across samples. While efective in person ReID or single-species settings, such assumptions rarely hold across species with drastically diferent anatomies and visual characteristics. Consequently, these methods struggle to establish consistent cross-species representations and often produce fragmented embedding spaces with limited generalization to unseen species. Domain generalization methods developed for person ReID attempt to mitigate distribution shifts by learning domain-invariant representations through techniques such as feature disentanglement, style normalization, or meta-learning [11, 38]. However, these approaches implicitly assume comparable semantic structures across domains. In the multispecies setting, where anatomical structures and visual semantics difer substantially, this assumption becomes invalid, limiting their ability to capture transferable representations across species.

To address these challenges, we propose the Semantic Consistency Learning (SCL) framework, which promotes stable and transferable feature learning through two complementary components. As illustrated in Figure 2, (1) Foreground–Background Decoupled Spectral Normalization (FDSNorm) introduces a region-aware frequency-domain normalization mechanism that suppresses environment induced style variations while preserving structure-sensitive semantics. (2) Cross-species Neighborhood Modeling (CNM) captures relational regularities within and across species by constructing dynamic feature neighborhoods, enabling the model to align transferable semantics while maintaining intra-species discriminative structure.

## 3.2 Foreground–Background Decoupled Spectral Normalization

In multi-species generalized ReID, heterogeneous textures, morphologies, and environmental conditions introduce substantial species-dependent biases, leading to unstable feature distributions. Recent studies show that such variations are closely related to the spectral characteristics of visual representations [24,30]. Specifically, the amplitude spectrum mainly captures style-related factors such as illumination and background statistics, whereas the phase spectrum preserves semantic structure. Consequently, normalization strategies that suppress feature statistics uniformly may inadvertently distort phase-dependent semantics. To address this limitation, we introduce Foreground–Background Decoupled Spectral Normalization (FDSNorm), a frequency-domain normalization mechanism tailored for multi-species ReID. Unlike existing normalization methods that suppress style variations globally in spatial or spectral spaces, FDSNorm explicitly accounts for semantic diferences across regions. By preserving the original phase information and adaptively modulating amplitude spectra in foreground and background regions, the proposed mechanism achieves structure-preserving style control under cross-species domain shifts.

Frequency-Domain Feature Normalization. Given an input image $\mathbf { I } \in$ $\mathbb { R } ^ { H _ { 0 } \times W _ { 0 } \times C _ { 0 } }$ , the Vision Transformer [14] divides it into N non-overlapping patches of size $P \times P ,$ , each projected into a C-dimensional embedding space. After positional encoding and class-token concatenation, the resulting sequence is processed through L Transformer layers. Let $\mathbf { F } ^ { ( l ) } \mathbf { \Sigma } \in \mathbb { R } ^ { B \times C \times H \times W }$ denote the reshaped token feature map at the l-th layer, where B is the batch size and $( H , W )$ represent the spatial grid reconstructed from tokens.

To adaptively suppress style-induced domain bias while preserving semantic consistency, we employ a learnable frequency-domain normalization strategy. First, we obtain a style-normalized version of the feature map via spatial nor-

malization:

$$
\tilde { \mathbf { F } } ^ { ( l ) } = \mathrm { N o r m } ( \mathbf { F } ^ { ( l ) } ) ,\tag{1}
$$

where Norm(·) denotes instance normalization applied across spatial dimensions. We then perform Discrete Fourier Transform (DFT) on both the original and normalized features:

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { o r g } } ^ { ( l ) } ( u , v ) = \displaystyle \sum _ { x = 0 } ^ { H - 1 } \sum _ { y = 0 } ^ { W - 1 } \mathbf { F } ^ { ( l ) } ( x , y ) e ^ { - j 2 \pi \left( \frac { u x } { H } + \frac { v y } { W } \right) } , } \\ { \mathcal { F } _ { \mathrm { n o r m } } ^ { ( l ) } ( u , v ) = \displaystyle \sum _ { x = 0 } ^ { H - 1 } \sum _ { y = 0 } ^ { W - 1 } \tilde { \mathbf { F } } ^ { ( l ) } ( x , y ) e ^ { - j 2 \pi \left( \frac { u x } { H } + \frac { v y } { W } \right) } , } \end{array}\tag{2}
$$

where $( x , y )$ and $( u , v )$ denote the spatial and frequency coordinates, respectively, and $j = \sqrt { - 1 }$ . Each spectral representation is decomposed into amplitude and phase components:

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { o r g } } ^ { ( l ) } ( u , v ) = A _ { \mathrm { o r g } } ^ { ( l ) } ( u , v ) e ^ { j \varPhi _ { \mathrm { o r g } } ^ { ( l ) } ( u , v ) } , } \\ & { \mathcal { F } _ { \mathrm { n o r m } } ^ { ( l ) } ( u , v ) = A _ { \mathrm { n o r m } } ^ { ( l ) } ( u , v ) e ^ { j \varPhi _ { \mathrm { n o r m } } ^ { ( l ) } ( u , v ) } , } \end{array}\tag{3}
$$

where $A ^ { ( l ) } ( u , v ) = | \mathcal { F } ^ { ( l ) } ( u , v ) |$ captures the magnitude spectrum and $\begin{array} { r } { \varPhi ^ { ( l ) } ( u , v ) = } \end{array}$ $\angle \mathcal { F } ^ { ( l ) } ( u , v )$ encodes structural information. Following [24], we treat the amplitude spectrum as a style carrier reflecting environmental variations $( e . g$ ., illumination and background), while the phase spectrum represents semantic structure that should remain invariant across domains.

Given the CLS token $\mathbf { c } ^ { ( l ) }$ and patch tokens $\{ \mathbf { t } _ { i } ^ { ( l ) } \} _ { i = 1 } ^ { N }$ at layer $l ,$ we build a soft foreground mask from CLS-to-patch cosine similarity:

$$
s _ { i } = \left. \frac { \mathbf { t } _ { i } ^ { ( l ) } } { \lVert \mathbf { t } _ { i } ^ { ( l ) } \rVert } , \frac { \mathbf { c } ^ { ( l ) } } { \lVert \mathbf { c } ^ { ( l ) } \rVert } \right. , \quad \tilde { s } _ { i } = \frac { s _ { i } - \mathrm { m i n } ( \mathbf { s } ) } { \mathrm { m a x } ( \mathbf { s } ) - \mathrm { m i n } ( \mathbf { s } ) + \epsilon } ,\tag{4}
$$

$$
\tau = \mathrm { Q u a n t i l e } _ { 1 - r } ( \tilde { \mathbf { s } } ) , \quad m _ { i } = \sigma \bigg ( \frac { \tilde { s } _ { i } - \tau } { T _ { m } } \bigg ) ,\tag{5}
$$

where $r$ is the foreground ratio and $T _ { m }$ is the mask temperature. Reshaping $\{ m _ { i } \}$ gives $\mathbf { M } ^ { ( l ) } \in [ 0 , 1 ] ^ { 1 \times H \times W }$ . We keep the CLS token unchanged to preserve the global semantic representation, and apply spectral normalization only to patch tokens.

We then apply spatial split before frequency mixing:

$$
\mathbf { F } _ { \mathrm { f g , o r g } } ^ { ( l ) } = \mathbf { M } ^ { ( l ) } \odot \mathbf { F } ^ { ( l ) } , \quad \mathbf { F } _ { \mathrm { f g , n o r m } } ^ { ( l ) } = \mathbf { M } ^ { ( l ) } \odot \tilde { \mathbf { F } } ^ { ( l ) } ,\tag{6}
$$

$$
\mathbf { F } _ { \mathrm { b g , o r g } } ^ { ( l ) } = ( 1 - \mathbf { M } ^ { ( l ) } ) \odot \mathbf { F } ^ { ( l ) } , \quad \mathbf { F } _ { \mathrm { b g , n o r m } } ^ { ( l ) } = ( 1 - \mathbf { M } ^ { ( l ) } ) \odot \tilde { \mathbf { F } } ^ { ( l ) } .\tag{7}
$$

After spatial decoupling, we independently perform DFT on the foreground and background branches to obtain branch-wise amplitude and phase spectra:

$$
\mathcal { F } _ { \mathrm { f g } , * } ^ { ( l ) } = A _ { \mathrm { f g } , * } ^ { ( l ) } e ^ { j \Phi _ { \mathrm { f g } , * } ^ { ( l ) } } , \quad \mathcal { F } _ { \mathrm { b g } , * } ^ { ( l ) } = A _ { \mathrm { b g } , * } ^ { ( l ) } e ^ { j \Phi _ { \mathrm { b g } , * } ^ { ( l ) } } ,\tag{8}
$$

where $\ast \in \{ \mathrm { o r g } , \mathrm { n o r m } \}$ denotes original and normalized branches. Foreground and background use independent mixing strengths:

$$
\alpha _ { \mathrm { f g } } = [ \mathrm { s o f t m a x } ( \lambda _ { \mathrm { f g } } / T _ { s } ) ] _ { 0 } , \quad \alpha _ { \mathrm { b g } } = [ \mathrm { s o f t m a x } ( \lambda _ { \mathrm { b g } } / T _ { s } ) ] _ { 0 } ,\tag{9}
$$

where $\lambda _ { \mathrm { f g } } , \lambda _ { \mathrm { b g } } \in \mathbb { R } ^ { 2 }$ are learnable two-dimensional parameter vectors initialized to zeros, $T _ { s }$ is the temperature parameter in softmax, and $[ \cdot ] _ { 0 }$ selects the first element as the normalized mixing weight.

$$
\hat { A } _ { \mathrm { f g } } ^ { ( l ) } = \alpha _ { \mathrm { f g } } A _ { \mathrm { f g , n o r m } } ^ { ( l ) } + ( 1 - \alpha _ { \mathrm { f g } } ) A _ { \mathrm { f g , o r g } } ^ { ( l ) } , \ \hat { A } _ { \mathrm { b g } } ^ { ( l ) } = \alpha _ { \mathrm { b g } } A _ { \mathrm { b g , n o r m } } ^ { ( l ) } + ( 1 - \alpha _ { \mathrm { b g } } ) A _ { \mathrm { b g , o r g } } ^ { ( l ) } .\tag{10}
$$

In this work, we use the unconstrained variant, i.e., no explicit ordering constraint is imposed between $\alpha _ { \mathrm { f g } }$ and $\alpha _ { \mathrm { b g } } .$ , allowing the network to adaptively discover when background regions require stronger style suppression. Reconstruction preserves branch-wise original phase:

$$
\hat { \mathbf { F } } _ { \mathrm { f g } } ^ { ( l ) } = \mathcal { F } ^ { - 1 } \Big ( \hat { A } _ { \mathrm { f g } } ^ { ( l ) } e ^ { j \phi _ { \mathrm { f g , o r g } } ^ { ( l ) } } \Big ) , \hat { \mathbf { F } } _ { \mathrm { b g } } ^ { ( l ) } = \mathcal { F } ^ { - 1 } \Big ( \hat { A } _ { \mathrm { b g } } ^ { ( l ) } e ^ { j \phi _ { \mathrm { b g , o r g } } ^ { ( l ) } } \Big ) ,\tag{11}
$$

$$
\hat { \mathbf { F } } ^ { ( l ) } = \hat { \mathbf { F } } _ { \mathrm { f g } } ^ { ( l ) } + \hat { \mathbf { F } } _ { \mathrm { b g } } ^ { ( l ) } .\tag{12}
$$

Through empirical analysis, the FDSNorm module is inserted at multiple Transformer depths $( l \in \{ 0 , 4 , 8 \} )$ ), corresponding to the patch-embedding output and intermediate blocks, thereby forming a progressive de-stylization pipeline. Specifically, shallow layers mainly clean pixel-level domain shifts (e.g., illumination and color temperature); middle layers align structure-level shifts $( \mathrm { e . g . }$ , parts and shape); and deeper layers compensate for residual shifts leaked by residual connections. In parallel, as the CLS token passes through more attention layers, its semantic awareness becomes stronger. This allows the foreground mask to be refined from coarse to fine, yielding a coordinated progression between destylization strength and mask quality. In addition, the foreground ratio $r$ and branch-wise mixing strengths $( \alpha _ { \mathrm { f g } }$ and $\alpha _ { \mathrm { b g } } )$ are dynamically learnable parameters, while the mask temperature $T _ { m }$ remains fixed.

Temporal Semantic Distillation. While normalization suppresses stylerelated instability within the backbone, temporal inconsistency may still arise from noisy or domain-biased updates during optimization. To stabilize the evolution of semantic representations, we employ a teacher–student framework in which the teacher network maintains an exponential moving average (EMA) of the student parameters:

$$
\pmb \theta _ { t } \gets \mu \pmb \theta _ { t } + ( 1 - \mu ) \pmb \theta _ { s } ,\tag{13}
$$

where $\theta _ { t }$ and $\theta _ { s }$ denote the teacher and student parameters, respectively, and $\mu \in ( 0 , 1 )$ is a momentum coeficient. The teacher network provides temporally smoothed features that serve as stable semantic anchors for subsequent neighborhood consistency learning, efectively distilling long-term structural knowledge into the student without introducing additional supervision.

## 3.3 Cross-species Neighborhood Modeling

In multi-species ReID, each species forms a visually coherent cluster within the feature space, yet these clusters remain topologically isolated due to the absence of shared semantic anchors. This leads to semantic fragmentation: features are discriminative within species but unaligned across them, hindering transfer to unseen species. Conventional ReID metric learning objectives such as triplet loss [16] rely on instance-level correspondences and fail to exploit the latent relational regularities shared across species. To bridge this gap, we propose Crossspecies Neighborhood Modeling, which constructs dynamic relational structures using a teacher-maintained memory to jointly enforce intra-species consistency and cross-species semantic connectivity.

Dynamic Memory Construction. Let $\mathbf { f } _ { t }$ and $\mathbf { z } _ { s }$ denote the global features from the teacher and student networks, respectively. We maintain a feature memory queue $\mathcal { M } = \{ ( \mathbf { f } _ { i } , s p _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { M } | }$ , where each entry consists of a normalized teacher feature $\mathbf { f } _ { i } .$ , its species label $s p _ { i }$ , and identity label $y _ { i }$ . After each iteration, newly computed teacher features are enqueued, while the oldest entries are dequeued to maintain a fixed capacity, ensuring $\mathcal { M }$ captures long-term semantic structure across species. For each student feature $\mathbf { z } _ { s }$ with species label $s p _ { ; }$ , the cosine similarity to all memory entries is computed as:

$$
\mathrm { s i m } ( \mathbf { z } _ { s } , \mathbf { f } _ { i } ) = \frac { \mathbf { z } _ { s } ^ { \top } \mathbf { f } _ { i } } { \left\| \mathbf { z } _ { s } \right\| \left\| \mathbf { f } _ { i } \right\| } .\tag{14}
$$

Mutual Neighborhood Search. Let $\mathcal { P } = \{ \mathbf { f } _ { t } \} \cup \mathcal { M }$ denote the union of the current-batch teacher features and the memory queue. Each anchor $\mathbf { z } _ { s }$ retrieves two types of neighborhoods from $\mathcal { P } \mathrm { : }$ an intra-species neighborhood $\mathcal { N } _ { i n t r a } ( { \bf z } _ { s } )$ consisting of top- $K _ { 1 }$ nearest neighbors whose species label matches the anchor, i.e., $s p _ { i } = s p _ { ; }$ , and a cross-species neighborhood $\mathcal { N } _ { c r o s s } ( \mathbf { z } _ { s } )$ of top- $K _ { 1 }$ nearest neighbors where $s p _ { i } \neq s p$ . To suppress incidental correlations in the cross-species neighborhood, we adopt a reciprocal nearest-neighbor rule: for an anchor A and a candidate neighbor B retrieved from the top- $K _ { 1 }$ cross-species list of $A ,$ B is retained only if $A$ also appears in the top- $K _ { 2 }$ cross-species nearest list of B within $\mathcal { P }$ . Here, $K _ { 1 }$ determines the size of the candidate neighborhood for retrieval, whereas $K _ { 2 } \ ( K _ { 2 } \leq K _ { 1 } )$ controls the stringency of the reciprocal verification: a larger $K _ { 1 }$ broadens the candidate pool, while a smaller $K _ { 2 }$ retains only strongly mutual cross-species pairs. This reciprocal filtering is applied solely to the cross-species neighborhood. This bidirectional filtering yields more stable semantic neighborhoods that reflect intrinsic relational similarity across species. The neighborhood centers are defined as:

$$
\mathbf { c } _ { i n t r a } = \frac { 1 } { | \mathcal { N } _ { i n t r a } | } \sum _ { \mathbf { f } _ { i } \in \mathcal { N } _ { i n t r a } } \mathbf { f } _ { i } , \quad \mathbf { c } _ { c r o s s } = \frac { 1 } { | \mathcal { N } _ { c r o s s } | } \sum _ { \mathbf { f } _ { i } \in \mathcal { N } _ { c r o s s } } \mathbf { f } _ { i } .\tag{15}
$$

Loss Formulation. Cross-species neighborhood modeling jointly optimizes two complementary objectives: intra-species compactness and cross-species relational consistency. 1) Intra-species Compactness. For each sample, we align

the student feature with its intra-species neighborhood center to enforce intraspecies compactness, where the center is computed over same-species neighbors regardless of identity:

$$
\mathcal { L } _ { \mathrm { \mathrm { i n t r a } } } = 1 - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sin ( \mathbf { z } _ { i } , \mathbf { c } _ { \mathrm { i n t r a } } ^ { ( i ) } ) ,\tag{16}
$$

where B denotes the batch size and sim $( \cdot , \cdot )$ denotes cosine similarity. 2) Crossspecies Relational Constraint. To bridge gaps across species, we introduce a margin-based relational constraint:

$$
\mathcal { L } _ { \mathrm { c r o s s } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } \left( 0 , m - \sin ( \mathbf { z } _ { i } , \mathbf { c } _ { \mathrm { c r o s s } } ^ { ( i ) } ) \right) ,\tag{17}
$$

where $m$ is a margin. Although cross-species centers provide transferable semantic cues, overly strong attraction toward them may pull features of diferent identities too close, impairing identity discrimination. The hinge thus acts as a bounded attraction: each feature is pulled toward its cross-species relational center only until the similarity reaches the target margin $m _ { ; }$ , after which no gradient is applied. This injects a controlled level of cross-species connectivity $( \mathrm { i . e . } ,$ sim $\geq m )$ , while the small margin and the saturation of the hinge prevent features from being driven into full alignment, thereby preserving identity-level discrimination. This formulation softly aligns inter-species manifolds without collapsing structural diversity. The overall CNM objective is:

$$
\mathcal { L } _ { C N M } = \lambda _ { i n t r a } \mathcal { L } _ { i n t r a } + \lambda _ { c r o s s } \mathcal { L } _ { c r o s s } .\tag{18}
$$

In summary, our learning objective is the total loss ${ \mathcal { L } } ,$ , formulated as a weighted sum of the identification loss, the triplet loss, and the CNM loss.

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } + \lambda _ { \mathrm { t r i } } \mathcal { L } _ { \mathrm { t r i } } + \mathcal { L } _ { C N M } . } \end{array}\tag{19}
$$

Memory Update. After each iteration, normalized teacher features with their species and identity labels are added to the memory:

$$
\mathcal { M } \gets \mathrm { e n q u e u e } ( \mathrm { N o r m } ( \mathbf { f } _ { t } ) , s p _ { t } , y _ { t } ) , \quad \mathrm { d e q u e u e ~ o l d e s t . }\tag{20}
$$

This online update maintains a temporally smoothed and semantically consistent teacher space, providing robust relational guidance for the student network.

## 4 Evaluation Protocol

## 4.1 Datasets and Evaluation Protocols

Datasets and Splits. To comprehensively evaluate cross-species generalization, we conduct experiments on 11 publicly available animal ReID datasets covering diverse habitats and species morphologies. This diverse dataset collection provides a challenging evaluation setting with substantial variations in visual appearance and environmental conditions. (1) Wildlife71 Dataset [20]. This large-scale benchmark comprises 71 species. Following its oficial split protocol, we use the predefined 67 seen species for training. (2) PetFace Dataset [42]. This dataset contains facial images from 13 domestic animal species, and we test the model separately on each species. (3) Nine public datasets. We further evaluate on iPanda-50 [45], ELPephants [22], SealID [34], GZGC (zebra and girafe domains) [41], WhaleSharkID [17], ATRW [26], HyenaID2022 [5], LeopardID2022 [6], and SeaTurtleID2022 [1]. As shown in Tab. 2, we design two complementary evaluation protocols to simulate diferent cross-species generalization scenarios. Both protocols enforce a fully open-set setting in which identities and species in the test set are disjoint from those used during training, ensuring that the evaluation reflects genuine cross-species generalization rather than dataset-specific overlap. Since existing animal ReID datasets adopt inconsistent split strategies or lack oficial training/testing partitions, we use the full datasets under both protocols to maintain consistent evaluation conditions. The only exception is PetFace, for which we follow the oficial split. This unified data usage ensures fair and reproducible comparisons across diferent methods.

Table 1: Dataset Statistics  
Table 2: Evaluation Protocols
<table><tr><td>Dataset</td><td>#Image</td><td>#ID</td><td>Species</td><td>Protocol</td><td>Training Data</td><td>Testing Data</td></tr><tr><td>PetFace [42]</td><td>115,708</td><td>55,686</td><td>13</td><td rowspan="5">1</td><td rowspan="5">Wildlife71</td><td>PetFace, iPanda-50 ELPephants, SealID,</td></tr><tr><td>Wildlife71 [20]</td><td>108,096</td><td>1,924</td><td>67</td><td></td></tr><tr><td>iPanda-50 [45]</td><td>6,874</td><td>50</td><td>1</td><td>SeaTurtleID2022, ATRW,</td></tr><tr><td>ELPephants [22]</td><td>2,078</td><td>276</td><td>1</td><td>HyenaID2022, LeopardID2022,</td></tr><tr><td>SealID [34]</td><td>2,080</td><td>61</td><td>1</td><td>GZGC, WhaleSharkID,</td></tr><tr><td>GZGC [41]</td><td>4,948</td><td>1,762</td><td>2</td><td rowspan="5"></td><td>iPanda-50+ELPephants</td><td rowspan="5">PetFace,</td></tr><tr><td>ATRW [26] HyenaID2022 [5]</td><td>2,950 3,129</td><td>135 256</td><td>1</td><td>+LeopardID2022+GZGC</td></tr><tr><td></td><td>6,806</td><td></td><td>1</td><td>+ATRW+HyenaID2022</td></tr><tr><td>LeopardID2022 [6] SeaTurtleID2022 [1]</td><td>8,729</td><td>430 438</td><td>1 1</td><td>+SealID+SeaTurtleID2022</td></tr><tr><td>WhaleSharkID [17]</td><td>7,693</td><td>543</td><td>1</td><td>+WhaleSharkID</td></tr></table>

Evaluation Metrics. We adopt Cumulative Matching Characteristics (CMC) at Rank-1 and mean Average Precision (mAP) as standard metrics. Since most animal ReID datasets lack explicit camera annotations, we include all valid gallery matches in evaluation rather than only cross-camera ones.

## 5 Experiments

## 5.1 Implementation Details

All experiments were conducted on four NVIDIA 4090 GPUs using $\mathrm { P y }$ Torch. We employ a ViT [14] pre-trained on ImageNet-1K as the backbone; unless a method has specific architectural constraints, all baselines share the same backbone for fair comparison. Input images are resized to 256 × 256 with patch size $1 6 \times 1 6$ Training augmentations include random horizontal flipping (50%) and 10-pixel padding. For CNM hyperparameters, we set $K _ { 1 } { = } 8 , K _ { 2 } { = } 3 .$ , margin = 0.2, and memory size = 4096. The model is trained for 60 epochs with SGD (initial lr

Table 3: Protocol-1 results on 10 unseen domains. The symbol † denotes results obtained from re-implemented versions of the corresponding methods.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">ELPephants [22]</td><td colspan="2">SealID [34]</td><td colspan="2">GZGC (zebra) [41]</td><td colspan="2">ATRW [26]</td><td colspan="2">GZGC (giraffe) [41]</td></tr><tr><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td></tr><tr><td>Base [15]</td><td>ICCV 2021</td><td>31.9</td><td>7.9</td><td>79.4</td><td>24.8</td><td>12.6</td><td>7.1</td><td>96.2</td><td>58.9</td><td>21.0</td><td>25.3</td></tr><tr><td>TransReID [15]</td><td>ICCV 2021</td><td>32.3</td><td>8.1</td><td>78.6</td><td>23.3</td><td>12.2</td><td>6.9</td><td>96.5</td><td>59.1</td><td>20.0</td><td>25.6</td></tr><tr><td>META [46]</td><td>ECCV 2022</td><td>26.4</td><td>5.8</td><td>79.0</td><td>20.2</td><td>8.0</td><td>3.2</td><td>95.7</td><td>51.6</td><td>13.5</td><td>9.3</td></tr><tr><td>CLIP [27]</td><td>AAAI 2023</td><td>28.9</td><td>6.8</td><td>77.6</td><td>20.8</td><td>11.9</td><td>6.9</td><td>95.6</td><td>58.1</td><td>22.9</td><td>25.7</td></tr><tr><td>PartAware [36]</td><td>ICCV 2023</td><td>32.0</td><td>7.9</td><td>79.7</td><td>24.9</td><td>12.4</td><td>7.1</td><td>96.2</td><td>58.9</td><td>20.6</td><td>25.4</td></tr><tr><td>UniReID† [20]</td><td>NeurIPS 2023</td><td>25.2</td><td>6.1</td><td>79.4</td><td>23.6</td><td>12.0</td><td>6.8</td><td>96.1</td><td>55.9</td><td>24.7</td><td>26.0</td></tr><tr><td>BAU [11]</td><td>NeurIPS 2024</td><td>13.2</td><td>3.7</td><td>80.4</td><td>30.8</td><td>10.1</td><td>5.1</td><td>90.6</td><td>49.9</td><td>20.6</td><td>22.2</td></tr><tr><td>AdaFreq [25]</td><td>ECCV 2024</td><td>33.3</td><td>8.4</td><td>78.6</td><td>22.8</td><td>12.1</td><td>6.8</td><td>96.4</td><td>58.0</td><td>21.6</td><td>26.0</td></tr><tr><td>ReNorm [38]</td><td>ECCV 2024</td><td>20.7</td><td>5.0</td><td>78.6</td><td>23.6</td><td>10.6</td><td>5.5</td><td>96.3</td><td>56.1</td><td>23.5</td><td>24.1</td></tr><tr><td>Megadescriptor† [7]</td><td>WACV 2024</td><td>32.0</td><td>8.2</td><td>76.5</td><td>22.1</td><td>12.9</td><td>7.0</td><td>95.9</td><td>57.4</td><td>23.5</td><td>26.8</td></tr><tr><td>MiewID† [39]</td><td>CoRR 2024</td><td>16.2</td><td>4.3</td><td>72.9</td><td>19.9</td><td>9.1</td><td>4.7</td><td>94.6</td><td>51.2</td><td>17.9</td><td>20.8</td></tr><tr><td>CLIP-FGDI [53]</td><td>TIFS 2025</td><td>16.2</td><td>4.3</td><td>72.4</td><td>22.2</td><td>10.6</td><td>5.6</td><td>88.5</td><td>45.8</td><td>21.3</td><td>24.5</td></tr><tr><td>ARBase† [18]</td><td>ICCV 2025</td><td>25.4</td><td>5.5</td><td>77.9</td><td>21.4</td><td>10.0</td><td>3.9</td><td>96.1</td><td>53.8</td><td>12.7</td><td>9.1</td></tr><tr><td rowspan="2">Ours Method</td><td rowspan="2"></td><td colspan="2">34.4 9.0</td><td colspan="2">81.5 25.6</td><td colspan="2">13.1 7.8</td><td colspan="2">98.0 60.3</td><td colspan="2">22.9 27.3</td></tr><tr><td colspan="2">iPanda-50 [45]</td><td colspan="2">HyenaID2022 [5]</td><td colspan="2">LeopardID2022 [6]</td><td colspan="2">SeaTurtleID2022 [1]</td><td colspan="2">WhaleSharkID [17]</td></tr><tr><td></td><td>Venue</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td></tr><tr><td>Base [15]</td><td>ICCV 2021</td><td>91.8</td><td>13.1</td><td>60.4</td><td>22.6</td><td>76.0</td><td>17.5</td><td>46.7</td><td>7.3</td><td>37.4</td><td>6.9</td></tr><tr><td>TransReID [15]</td><td>ICCV 2021</td><td>91.8</td><td>13.1</td><td>62.9</td><td>23.7</td><td>78.0</td><td>18.4</td><td>51.2</td><td>8.1</td><td>42.2</td><td>7.8</td></tr><tr><td>META [46]</td><td>ECCV 2022</td><td>87.9 90.2</td><td>11.4 12.9</td><td>50.8</td><td>15.5</td><td>64.0</td><td>12.0</td><td>39.3</td><td>5.0</td><td>35.1 38.8</td><td>5.4</td></tr><tr><td>CLIP [27]</td><td>AAAI 2023</td><td>91.7</td><td></td><td>59.3 61.8</td><td>21.3</td><td>74.7</td><td>16.5</td><td>45.9</td><td>6.9 8.0</td><td>40.2</td><td>7.0</td></tr><tr><td>PartAware [36]</td><td>ICCV 2023</td><td>88.8</td><td>13.1 12.9</td><td>59.2</td><td>23.0 21.2</td><td>77.6</td><td>18.3</td><td>51.1</td><td>6.9</td><td>30.9</td><td>7.3</td></tr><tr><td>UniReID† [20]</td><td>NeurIPS 2023</td><td>81.9</td><td>10.2</td><td>49.8</td><td></td><td>74.6</td><td>16.4</td><td>45.8</td><td>7.3</td><td>22.4</td><td>5.2</td></tr><tr><td>BAU [11]</td><td>NeurIPS 2024</td><td>90.6</td><td>12.8</td><td>59.9</td><td>15.0 23.0</td><td>65.6 76.5</td><td>11.9 17.4</td><td>51.2 53.1</td><td>8.4</td><td>42.5</td><td>3.6 8.0</td></tr><tr><td>AdaFreq [25] ReNorm [38]</td><td>ECCV 2024 ECCV 2024</td><td>82.1</td><td>9.9</td><td>59.9</td><td>21.4</td><td>72.5</td><td>13.6</td><td>60.4</td><td>9.3</td><td>32.9</td><td></td></tr><tr><td></td><td></td><td></td><td>13.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.4</td></tr><tr><td>Megadescriptor† [7]</td><td>WACV 2024</td><td>89.6</td><td></td><td>61.3</td><td>23.0</td><td>77.1</td><td>18.0</td><td>45.3</td><td>6.9</td><td>39.1</td><td>7.1</td></tr><tr><td>MiewID† [39]</td><td>CoRR 2024</td><td>74.2</td><td>8.9</td><td>53.0</td><td>17.1</td><td>66.3</td><td>12.0</td><td>42.0</td><td>5.5</td><td>27.2</td><td>4.5</td></tr><tr><td>CLIP-FGDI [53] ARBase† [18]</td><td>TIFS 2025</td><td>59.9 88.2</td><td>6.9</td><td>47.8</td><td>15.2</td><td>65.8</td><td>13.1</td><td>20.2</td><td>3.0</td><td>25.9</td><td>4.9</td></tr><tr><td></td><td>ICCV 2025</td><td></td><td>11.7</td><td>51.1</td><td>15.6</td><td>64.3</td><td>12.1</td><td>39.6</td><td>5.1</td><td>34.4</td><td>5.8</td></tr><tr><td>Ours</td><td></td><td>92.5</td><td>13.8</td><td>64.0</td><td>24.8</td><td>78.9</td><td>19.4</td><td>58.8</td><td>10.0</td><td>45.4</td><td>8.9</td></tr></table>

0.004, cosine decay) and a total batch size of 128 (8 identities × 4 images per GPU × 4 GPUs). Both CNM and FDSNorm use a 3-epoch warm-up. At test time, only original features are used for distance computation.

## 5.2 Comparison with SOTA Methods

Table 3 and Fig. 3 report Protocol-1 results; Table 4 reports Protocol-2 results on Wildlife71. We compare four categories of methods: (1) General ReID (e.g., TransReID, CLIP-ReID). These methods employ strong architectures and often achieve competitive results, yet they remain species-specific and do not capture fine-grained cues that transfer across species. (2) Animal ReID (e.g., UniReID, AdaFreq, ARBase). These methods enhance within-species discrimination but lack cross-species structural alignment, leading to negative transfer on unseen domains. Foundation models such as MegaDescriptor [7] and MiewID [39] benefit from large-scale data but still optimize instance-level metrics without explicit cross-species alignment. (3) DG person ReID (e.g., ReNorm, META). These methods improve statistical invariance and handle style variations, but do not address the semantic shifts induced by species changes and therefore underperform on unknown species. (4) DG animal ReID (e.g., UniReID). UniReID relies on dataset-specific textual descriptions; when applied to a face-centric dataset such as PetFace, its generic whole-body description becomes mismatched, leading to a substantial performance drop. Overall, existing methods may excel on

![](images/139803d2695030d264e6059a07c50c8436d3599f2789530c4815ab5f20ff1407.jpg)  
Fig. 3: Protocol-1 results on PetFace (13 species). Circles: mAP; triangles: Rank-1.

Table 4: Protocol-2 results on Wildlife71.
<table><tr><td>Method</td><td>mAP</td><td>mINP</td><td>Rank1</td></tr><tr><td>Base [15]</td><td>90.1</td><td>75.1</td><td>96.6</td></tr><tr><td>TransReID [15]</td><td>91.6</td><td>72.6</td><td>96.5</td></tr><tr><td>META [46]</td><td>83.7</td><td>51.0</td><td>96.5</td></tr><tr><td>CLIP [27]</td><td>86.4</td><td>54.1</td><td>96.7</td></tr><tr><td>PartAware [36]</td><td>90.1</td><td>70.3</td><td>96.8</td></tr><tr><td>UniReID† [20]</td><td>84.4</td><td>53.6</td><td>96.6</td></tr><tr><td>BAU [11]</td><td>86.7</td><td>56.4</td><td>96.2</td></tr><tr><td>AdaFreq [25]</td><td>91.2</td><td>79.2</td><td>97.3</td></tr><tr><td>ReNorm [38]</td><td>71.9</td><td>25.3</td><td>95.6</td></tr><tr><td>Megadescriptor† [7]</td><td>87.3</td><td>60.6</td><td>97.3</td></tr><tr><td>MiewID† [39]</td><td>82.5</td><td>43.3</td><td>96.1</td></tr><tr><td>CLIP-FĠDI [53]</td><td>73.6</td><td>27.7</td><td>97.4</td></tr><tr><td>ARBase† [18]</td><td>86.4</td><td>59.4</td><td>96.8</td></tr><tr><td>Ours</td><td>93.8</td><td>78.5</td><td>97.6</td></tr></table>

Table 5: Ablation on CNM memory.
<table><tr><td>Setting</td><td>Same-ID (%)</td><td>mAP (%)</td><td>Rank1 (%)</td></tr><tr><td>k-NN</td><td>65.12</td><td>18.36</td><td>53.53</td></tr><tr><td>CNM w/o Mem.</td><td>78.24</td><td>19.33</td><td>56.01</td></tr><tr><td>Ours (CNM)</td><td>89.36</td><td>20.69</td><td>58.75</td></tr></table>

Table 6: Layer replacement and EMA ablation results.

<table><tr><td>Setting</td><td>|mAP (%)</td><td>Rank1 (%)</td></tr><tr><td> $l \in \{ 0 , 4 , 8 \}$ </td><td>20.69</td><td>58.75</td></tr><tr><td> $l \in \{ 2 , 4 , 6 \}$ </td><td>19.77</td><td>56.58</td></tr><tr><td> $l \in \{ 0 , \ldots , 1 1 \}$ </td><td>19.37</td><td>54.53</td></tr><tr><td> $l \in \{ 0 , 4 , 8 \} \ \mathrm { \vec { w } / o \ E M A }$ </td><td>20.14</td><td>57.53</td></tr></table>

selected species yet degrade significantly when the species changes. In contrast, our approach learns a species-agnostic model without any target-domain adaptation and achieves leading performance across diverse unseen species.

## 5.3 Ablation Experiments

Efectiveness of Each Component. Ablation results in Table 7 confirm the complementary contributions of FDSNorm and CNM to cross-species generalization. FDSNorm stabilizes style statistics to yield domain-robust features, while CNM promotes intra-species cohesion and cross-species semantic connectivity. Table 5 further shows that CNM with memory significantly improves nearestneighbor identity purity over vanilla k-NN; crucially, removing the memory queue leads to a noticeable drop in retrieval purity, proving it essential for providing stable, long-term relational anchors. Table 6 reveals that sparse layer placement $( l \in \{ 0 , 4 , 8 \} )$ is optimal for FDSNorm—applying it too densely or across all layers over-suppresses structural semantics. Removing EMA further destabilizes feature evolution, confirming the necessity of temporal smoothing.

Foreground-Only Analysis. We use MVANet [49] to segment foreground regions and construct foreground-only inputs. Table 8 further evaluates robustness when background cues are largely removed. All methods degrade under foreground-only input, indicating that context still contributes to matching. Nevertheless, our method achieves the strongest absolute foreground performance and the smallest mAP drop, showing that SCL relies more on transferable identity structure than on scene-specific shortcuts.

![](images/088b3061ad68658105b910aa5b311aa7cb2782fc772ff03cb6ea7b71ce3b1aee.jpg)

Table 7: Ablation study of diferent components on ten wildlife datasets under Protocol-1. Intra: Intra-species consistency; Cross: Cross-species consistency.
<table><tr><td rowspan="2"></td><td rowspan="2">ID FDSNorm Intra Cross</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">ELPephants [22]</td><td colspan="2">SealID [34]</td><td colspan="2">Wildlife71 [20]</td><td colspan="2">ATRW [26]</td><td colspan="2">GZGC (giraffe) [41]</td></tr><tr><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>Rank1</td><td>mAP</td><td>|Rank1</td><td>mAP</td><td>|Rank1</td><td>mAP</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td>31.9</td><td>7.9</td><td>79.4</td><td>24.8</td><td>96.6</td><td>90.1</td><td>96.2</td><td>58.9</td><td>21.0</td><td>25.3</td></tr><tr><td>(b)</td><td>√</td><td></td><td></td><td>32.8</td><td>8.4</td><td>80.9</td><td>25.3</td><td>97.0</td><td>91.4</td><td>97.1</td><td>59.3</td><td>22.4</td><td>26.8</td></tr><tr><td>(c)</td><td></td><td>√</td><td></td><td>33.1</td><td>8.5</td><td>80.8</td><td>24.8</td><td>97.2</td><td>92.3</td><td>97.7</td><td>59.6</td><td>22.2</td><td>26.0</td></tr><tr><td>(d)</td><td></td><td>√</td><td>√</td><td>33.3</td><td>8.6 9.0</td><td>81.2</td><td>25.0</td><td>97.4</td><td>93.0</td><td>97.6</td><td>60.0</td><td>22.5</td><td>27.0</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>√</td><td colspan="2">34.2</td><td>81.3</td><td>25.6</td><td>97.6</td><td>93.8</td><td>97.8</td><td>60.3</td><td>22.7</td><td>27.3</td></tr><tr><td></td><td colspan="2"></td><td colspan="2"></td><td colspan="2">PetFace [42]</td><td colspan="2">| HyenaID2022 [5] | LeopardID2022 [6] |</td><td colspan="2"></td><td colspan="2">| SeaTurtleID2022 [1] |</td><td colspan="2">WhaleSharkID [17]</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td>41.9</td><td>39.8</td><td>60.4</td><td>22.6</td><td>76.0</td><td>17.5</td><td>46.7</td><td>7.3</td><td>37.4</td><td>6.9</td></tr><tr><td>(b)</td><td>√</td><td></td><td></td><td>43.6</td><td>42.8</td><td>62.6</td><td>24.1</td><td>77.8</td><td>18.7</td><td>55.5</td><td>9.2</td><td>44.0</td><td>8.4</td></tr><tr><td>(c)</td><td></td><td>√</td><td></td><td>44.5</td><td>44.8</td><td>63.2</td><td>23.9</td><td>78.2</td><td>18.6</td><td>56.7</td><td>8.9</td><td>44.3</td><td>8.5</td></tr><tr><td>(d)</td><td></td><td>√</td><td>√</td><td>45.3</td><td>45.9</td><td>63.5</td><td>24.2</td><td>78.5</td><td>18.9</td><td>57.4</td><td>9.3</td><td>44.7</td><td>8.7</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>√</td><td>46.2</td><td>46.9</td><td>63.8</td><td>24.8</td><td>78.7</td><td>19.4</td><td>58.6</td><td>10.0</td><td>45.2</td><td>8.9</td></tr></table>

Table 8: Original vs. foreground-only input under Protocol-2.
<table><tr><td rowspan="2">Model</td><td colspan="3">Wildlife71 [20]</td><td colspan="3">Wildlife71 (Foreground) [20] |</td><td colspan="3">Drop</td></tr><tr><td>mAP</td><td>mINP</td><td>Rank1</td><td>mAP</td><td>mINP</td><td>Rank1</td><td>mAP</td><td>mINP</td><td>Rank1</td></tr><tr><td>Base [15]</td><td>90.1</td><td>75.1</td><td>96.6</td><td>71.2</td><td>28.6</td><td>95.8</td><td>18.9</td><td>46.5</td><td>0.8</td></tr><tr><td>Megadescriptor [7]</td><td>87.3</td><td>60.6</td><td>97.3</td><td>69.8</td><td>28.7</td><td>95.2</td><td>17.5</td><td>31.9</td><td>2.1</td></tr><tr><td>SCL (Ours)</td><td>93.8</td><td>78.5</td><td>97.6</td><td>79.0</td><td>45.8</td><td>96.1</td><td>14.8</td><td>32.7</td><td>1.5</td></tr></table>

Visualization Analysis. Figure 4 shows eight subfigures: Compared with Base, SCL consistently shifts attention toward semantically meaningful animal regions (e.g., torso contours, texture-rich parts, and limbs) and reduces difuse activation on irrelevant background areas. This trend is stable across elephant, zebra, seal, sea turtle, whale shark, hyena, nyala and tiger, indicating stronger structurefocused consistency under large appearance and habitat variations.

![](images/a2df46207553697d11af3004a2493b09e9b4489f5b3c95a0455b904eec4f1877.jpg)  
(a) Nyala

![](images/de5d2156ada11bc255cc7006091fc1a1ac0480b7e47d79902e36f1cb9cae78ef.jpg)

![](images/b76fd95c86a1f070c0459f8fcc9ce8b27a05404aaa263c4519f78ee7e2845b0b.jpg)  
(b) Hyena  
(c) Elephant

![](images/a8b3fc63aa5e3e4e57070acfb8e5f4100b10736bef0e35e906283d5153a75f83.jpg)  
(d) Zebra

![](images/9b8d264244b66a269273517dac1c09d914c869b27f3ca52629e18e96f57e4736.jpg)  
(e) Seal

![](images/4334dbd352ad25a6dfe34e532c0d858979a29a4bc21f479543f6388d451c6106.jpg)  
(f) SeaTurtle

![](images/3fb5c7d748d05171f4577fbc12098f3daec1a69887d7ff1671b95312dab11178.jpg)  
(g) Whale Shark

![](images/c1e60b444ec013013c01de7d194e9c0a70f8014983d3bfcb39974cadd1128cfa.jpg)  
(h) Tiger  
Fig. 4: Last layer activation-map comparison between Base (left) and SCL (right).

Feature Distribution Analysis. Figure 5 shows what CNM learns in the embedding space. For this experiment, Base denotes the plain ViT baseline. Both the baseline and our model are trained on Wildlife71, and we randomly sample

![](images/258de7a0af2b8038e592a946b2f4d1dc2358b55c8b7c9235a599479e39795c5c.jpg)  
Fig. 5: Embedding space visualization of samples from diverse species.

Table 9: Ratio of inter-ID distance to intra-ID distances on Protocol-1. Higher is better.
<table><tr><td>Dataset</td><td>SCL</td><td>Mega</td><td>MiewID</td></tr><tr><td>iPanda-50 [45]</td><td>1.265</td><td>1.145</td><td>1.062</td></tr><tr><td>ELPephants [22]</td><td>1.220</td><td>1.122</td><td>1.049</td></tr><tr><td>SealID [34]</td><td>1.379</td><td>1.263</td><td>1.134</td></tr><tr><td>GZGC-Żebra [41]</td><td>1.242</td><td>1.106</td><td>1.048</td></tr><tr><td>ATRW [26]</td><td>2.816</td><td>2.106</td><td>1.269</td></tr><tr><td>GZGC-Giraffe [41]</td><td>1.838</td><td>1.726</td><td>1.326</td></tr><tr><td>HyenaID2022 [5]</td><td>1.741</td><td>1.412</td><td>1.126</td></tr><tr><td>LeopardID2022 [6]</td><td>1.456</td><td>1.272</td><td>1.091</td></tr><tr><td>SeaTurtleID2022 [1]</td><td>1.311</td><td>1.234</td><td>1.108</td></tr><tr><td>WhaleSharkID [17]]</td><td>1.196</td><td>1.064</td><td>1.039</td></tr></table>

100 instances for each of the 22 species for visualization. The Base representation exhibits clear species-wise separation, where samples from diferent species form isolated clusters. In contrast, CNM produces a more mixed embedding distribution across species while achieving better ReID performance, suggesting that it reduces species-specific clustering tendencies and learns more efective cross-species representations.

To further examine representation quality under unseen domains, we report the ratio of inter-ID distance to intra-ID distance in Table 9. A higher ratio indicates a better clustering structure, namely tighter intra-identity compactness together with clearer inter-identity separation. Compared with strong animal foundation models such as Megadescriptor and MiewID, SCL consistently achieves higher ratios on all ten unseen datasets. This result supports the same conclusion as our qualitative visualizations: our method improves cluster compactness while preserving clearer boundaries between identities, thereby mitigating species-isolated feature fragmentation.

## 6 Conclusion

This paper introduced the Semantic Consistency Learning framework for crossspecies generalization in animal ReID. Through Foreground–Background Decoupled Spectral Normalization, SCL suppresses feature instability caused by environmental variations while aggregating long-range semantic cues shared across species. Cross-species Neighborhood Modeling further shifts the objective from instance-level discrimination to relational semantic understanding, enabling the model to capture structural regularities that generalize across species and yield a unified, transferable embedding space. Promising future directions include extending SCL to fully open-set ReID, integrating large-scale ecological foundation models, and enabling adaptive deployment across diverse platforms such as UAVs and camera traps. We hope this work encourages the community to move beyond species-specific ReID and toward general, cross-species visual understanding that can support large-scale biodiversity monitoring in real-world ecosystems.

Acknowledgments. This work was partially supported by the National Natural Science Foundation of China under Grant T2541022.

## References

1. Adam, L., Čermák, V., Papafitsoros, K., Picek, L.: Seaturtleid2022: A long-span dataset for reliable sea turtle re-identification. In: WACV. pp. 7146–7156 (2024)

2. Adam, L., Papafitsoros, K., Kovář, R., Čermák, V., Picek, L.: Overview of AnimalCLEF 2025: Recognizing individual animals in images. In: Working Notes of CLEF (2025)

3. Andrew, W., Hannuna, S., Campbell, N., Burghardt, T.: Friesian: A novel dataset and a two-stage deep learning framework for cattle re-identification. In: ICIP. pp. 3103–3107 (2021)

4. Bai, Y., Jiao, J., Ce, W., Liu, J., Lou, Y., Feng, X., Duan, L.Y.: Person30k: A dual-meta generalization network for person re-identification. In: CVPR (2021)

5. Botswana Predator Conservation Trust: Panthera pardus csv custom export (2022), https://lila.science/datasets/hyena-id-2022, retrieved from African Carnivore Wildbook. Dataset export dated 2022-04-28. Accessed: June 29, 2026

6. Botswana Predator Conservation Trust: Panthera pardus csv custom export (2022), https : / / lila . science / datasets / leopard - id - 2022, retrieved from African Carnivore Wildbook. Dataset export dated 2022-04-28. Accessed: June 29, 2026

7. Čermák, V., Picek, L., Adam, L., Papafitsoros, K.: Wildlifedatasets: An opensource toolkit for animal re-identification. In: WACV. pp. 5953–5963 (2024)

8. Chen, S., Wu, Y., Ye, M.: Object-generalized re-identification: A step towards universal instance perception. In: CVPR. pp. 18481–18491 (2026)

9. Chen, S., Ye, M., Du, B.: Rotation invariant transformer for recognizing object in uavs. In: ACM MM. pp. 2565–2574 (2022)

10. Chen, Y.C., Zhu, X., Zheng, W.S., Lai, J.H.: Person re-identification by camera correlation aware feature augmentation. IEEE TPAMI 40(2), 392–408 (2017)

11. Cho, Y., Kim, J., Kim, W.J., Jung, J., eui Yoon, S.: Generalizable person reidentification via balancing alignment and uniformity. In: NeurIPS (2024)

12. Choi, S., Kim, T., Jeong, M., Park, H., Kim, C.: Meta batch-instance normalization for generalizable person re-identification. In: CVPR. pp. 3425–3435 (2021)

13. Dai, Y., Li, X., Liu, J., Tong, Z., Duan, L.Y.: Generalizable person re-identification with relevance-aware mixture of experts. In: CVPR. pp. 16145–16154 (2021)

14. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2020)

15. He, S., Luo, H., Wang, P., Wang, F., Li, H., Jiang, W.: Transreid: Transformerbased object re-identification. In: ICCV. pp. 15013–15022 (2021)

16. Hermans, A., Beyer, L., Leibe, B.: In defense of the triplet loss for person reidentification. arXiv preprint arXiv:1703.07737 (2017)

17. Holmberg, J., Norman, B., Arzoumanian, Z.: Estimating population size, structure, and residency time for whale sharks rhincodon typus through collaborative photoidentification. Endangered Species Research 7(1), 39–53 (2009)

18. Hou, S., Huang, P., Wang, Z., Liu, Y., Li, Z., Zhang, M., Huang, Y.: Openanimals: Revisiting person re-identification for animals towards better generalization. ICCV (2024)

19. Jiang, Y., Cheng, X., Yu, H., Liu, X., Chen, H., Zhao, G.: Domain shifting: A generalized solution for heterogeneous cross-modality person re-identification. In: ECCV. pp. 289–306 (2024)

20. Jiao, B., Liu, L., Gao, L., Wu, R., Lin, G., Wang, P., Zhang, Y.: Toward reidentifying any animal. NeurIPS 36, 40042–40053 (2023)

21. Jin, X., Lan, C., Zeng, W., Chen, Z., Zhang, L.: Style normalization and restitution for generalizable person re-identification. In: CVPR. pp. 3143–3152 (2020)

22. Korschens, M., Denzler, J.: Elpephants: A fine-grained dataset for elephant reidentification. In: ICCVW. pp. 0–0 (2019)

23. Lee, H., Park, J., Oh, J., Eom, C.: Domain generalization for person reidentification: A survey towards domain-agnostic person matching. Neurocomputing p. 130763 (2025)

24. Lee, S., Bae, J., Kim, H.Y.: Decompose, adjust, compose: Efective normalization by playing with frequency for domain generalization. In: CVPR. pp. 11776–11785 (2023)

25. Li, C., Chen, S., Ye, M.: Adaptive high-frequency transformer for diverse wildlife re-identification. In: ECCV. pp. 296–313. Springer (2024)

26. Li, S., Li, J.W., Wu, C., Zheng, W.S.: ATRW: A benchmark for amur tiger reidentification in the wild. In: ACM MM. pp. 1297–1305 (2021)

27. Li, S., Sun, L., Li, Q.: Clip-reid: exploiting vision-language model for image reidentification without concrete text labels. In: AAAI. vol. 37, pp. 1405–1413 (2023)

28. Liao, S., Shao, L.: Interpretable and generalizable person re-identification with query-adaptive convolution and temporal lifting. In: ECCV. pp. 456–474. Springer (2020)

29. Lin, C., Yuan, Z., Zhao, S., Sun, P., Wang, C., Cai, J.: Domain-invariant disentangled network for generalizable object detection. In: ICCV. pp. 8771–8780 (2021)

30. Lin, S., Zhang, Z., Huang, Z., Lu, Y., Lan, C., Chu, P., You, Q., Wang, J., Liu, Z., Parulkar, A., et al.: Deep frequency filtering for domain generalization. In: CVPR. pp. 11797–11807 (2023)

31. Lou, Y., Bai, Y., Liu, J., Wang, S., Duan, L.: Veri-wild: A large dataset and a new method for vehicle re-identification in the wild. In: CVPR. pp. 3235–3243 (2019)

32. Luo, H., Gu, Y., Liao, X., Lai, S., Jiang, W.: Bag of tricks and a strong baseline for deep person re-identification. In: CVPRW. pp. 122–130 (2019)

33. Nepovinnykh, E., Chelak, I., Eerola, T., Immonen, V., Kälviäinen, H., Kholiavchenko, M., Stewart, C.V.: Species-agnostic patterned animal re-identification by aggregating deep local features. IJCV 132(9), 4003–4018 (2024)

34. Nepovinnykh, E., Eerola, T., Biard, V., Mutka, P., Niemi, M., Kunnasranta, M., Kälviäinen, H.: Sealid: Saimaa ringed seal re-identification dataset. Sensors 22(19), 7602 (2022)

35. Nguyen, V.D., Mirza, S., Zakeri, A., Gupta, A., Khaldi, K., Aloui, R., Mantini, P., Shah, S.K., Merchant, F.: Tackling domain shifts in person re-identification: A survey and analysis. In: CVPRW. pp. 4149–4159 (2024)

36. Ni, H., Li, Y., Gao, L., Shen, H.T., Song, J.: Part-aware transformer for generalizable person re-identification. In: ICCV. pp. 11280–11289 (2023)

37. Ni, H., Song, J., Luo, X., Zheng, F., Li, W., Shen, H.T.: Meta distribution alignment for generalizable person re-identification. In: CVPR. pp. 2487–2496 (2022)

38. Nie, R., Ding, J., Zhou, X., Li, X.: Rethinking normalization layers for domain generalizable person re-identification. In: ECCV. pp. 267–284. Springer (2024)

39. Otarashvili, L., Subramanian, T., Holmberg, J., Levenson, J.J., Stewart, C.V.: Multispecies animal re-id using a large community-curated dataset. CoRR (2024)

40. Papafitsoros, K., Adam, L., Čermák, V., Picek, L.: Seaturtleid: A novel long-span dataset highlighting the importance of timestamps in wildlife re-identification. arXiv preprint arXiv:2211.10307 (2022)

41. Parham, J., Crall, J., Stewart, C., Berger-Wolf, T., Rubenstein, D.I.: Animal population censusing at scale with citizen science and photographic identification. In: AAAI (2017)

42. Shinoda, R., Shiohara, K.: Petface: A large-scale dataset and benchmark for animal identification. In: ECCV. pp. 19–36. Springer (2025)

43. Song, J., Yang, Y., Li, Y.Z., Hospedales, T.M.: Generalizable person reidentification by domain-invariant mapping network. In: CVPR. pp. 718–727 (2019)

44. Wang, G., Yuan, Y., Chen, X., Li, J., Zhou, X.: Learning discriminative features with multiple granularities for person re-identification. In: ACM MM. pp. 274–282 (2018)

45. Wang, L., Ding, R., Zhai, Y., Zhang, Q., Tang, W., Zheng, N., Hua, G.: Giant panda identification. IEEE TIP 30, 2837–2849 (2021)

46. Xu, B., Liang, J., He, L., Sun, Z.: Mimic embedding via adaptive aggregation: Learning generalizable person re-identification. In: ECCV. pp. 372–388. Springer (2022)

47. Yang, Z., Wu, D., Wu, C., Lin, Z., Gu, J., Wang, W.: A pedestrian is worth one prompt: Towards language guidance person re-identification. In: CVPR. pp. 17343–17353 (2024)

48. Ye, M., Chen, S., Li, C., Zheng, W.S., Crandall, D., Du, B.: Transformer for object re-identification: A survey. arXiv preprint arXiv:2401.06960 (2024)

49. Yu, Q., Zhao, X., Pang, Y., Zhang, L., Lu, H.: Multi-view aggregation network for dichotomous image segmentation. In: CVPR. pp. 3921–3930 (2024)

50. Yuan, C., Zhang, G., Ma, C., Zhang, T., Niu, G.: From poses to identity: Trainingfree person re-identification via feature centralization. In: CVPR (2025)

51. Zhang, Q., Wang, L., Patel, V.M., Xie, X., Lai, J.: View-decoupled transformer for person re-identification under aerial-ground camera network. In: CVPR. pp. 22000–22009 (2024)

52. Zhao, H., Qi, L., Geng, X.: Clip-dfgs: A hard sample mining method for clip in generalizable person re-identification. ACM T MULTIM COMP 21(1), 1–20 (2024)

53. Zhao, H., Qi, L., Geng, X.: Cilp-fgdi: Exploiting vision-language model for generalizable person re-identification. IEEE TIFS (2025)

54. Zhao, Y., Zhang, J., et al.: Learning to generalize unseen domains via memorybased multi-source meta-learning for person re-identification. In: CVPR (2021)

55. Zheng, Z., Zheng, Z., Zheng, W.S., Tao, D.: Deep smal-based 3d reconstruction for animal re-identification. In: ACM MM. pp. 4680–4688 (2021)

56. Zhu, H., Budhwant, P., Zheng, Z., Nevatia, R.: Seas: Shape-aligned supervision for person re-identification. In: CVPR. pp. 164–174 (2024)

57. Zou, Y., Yang, X., Yu, Z., Kumar, B.V., Kautz, J.: Joint disentangling and adaptation for cross-domain person re-identification. In: ECCV. pp. 87–104. Springer (2020)