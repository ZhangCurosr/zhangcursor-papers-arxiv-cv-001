# DenseTwins: Dense Modality Decoupling for Multimodal Self-Supervision

Carlo Sgaravatti<sup>1,\*</sup> Daniele Kopyshevskiy<sup>1,2,\*</sup> Davide Secco<sup>3</sup> Giacomo Boracchi<sup>1</sup> <sup>1</sup>Politecnico di Milano <sup>2</sup>Sorbonne University <sup>3</sup>Universita Bocconi\`

## Abstract

Multimodal perception is central to many computer vision applications, including robotics and autonomous driving, where different sensing modalities provide complementary cues. Self-supervised learning offers a scalable solution for exploiting large amounts of unlabeled multimodal data. However, existing methods either neglect modality-specific information by enforcing feature-space alignment between modalities or learn only global representations without preserving spatial structure. We propose DenseTwins, a multimodal self-supervisedframework that learns dense representations by explicitly decoupling the dense feature space into shared and modality-specific components. The proposed approach combines dense cross-modal correspondences with redundancy-reduction objectives, enabling the model to capture both shared semantics and modalityspecific cues within a structured dense embedding space. DenseTwins jointly optimizes an intra-modal objective across augmented views and a cross-modal objective that aligns sharedfeatures while decorrelating modality-specific information. We evaluate DenseTwins across multiple multimodal benchmarks and sensing setups, achieving stateof-the-art or competitive performance across multimodal self-supervised baselines. Code available at: https: //github.com/CarloSgaravatti/DenseTwins.

## 1. Introduction

Many real-world perception systems rely on multiple sensing modalities, such as RGB cameras, depth sensors, Li-DAR, event cameras, or thermal imagery [7]. These modalities provide complementary information about the same scene, and their joint exploitation is crucial in perception applications such as autonomous driving, robotics, and video understanding [5, 27]. Unfortunately, collecting large-scale annotated multimodal datasets is expensive and often impractical. Self-supervised learning (SSL) provides a natural alternative, enabling learning latent representations directly from unlabeled data [41].

SSL has been widely studied in single-modal settings [6, 10, 37]. A common approach to extend SSL to multiple modalities is to project inputs from different modalities into a shared embedding space and align their representations. The most prominent example is CLIP [26], which demonstrates that contrastive learning can effectively align images and text in the latent space. Alignment can also be achieved through redundancy-reduction objectives [6, 37], which avoid representation collapse by decorrelating feature dimensions. However, in multimodal perception, different sensors provide only partially overlapping information: while some semantic content is shared across modalities, other cues are inherently modality-specific (e.g., texture in RGB, discontinuities in depth images, thermal signatures, or event dynamics). Therefore, an effective multimodal representation should preserve both the shared and the modality-specific information [33].

This motivates modality decoupling, where the embedding space is explicitly separated into shared and modalityspecific components. To the best of our knowledge, De-CUR [33] is the only multimodal SSL method that explicitly performs decoupling. However, it operates on global representations of the input modalities, discarding spatial structure and limiting the applicability to downstream tasks such as semantic segmentation and object detection, which are particularly relevant in multimodal perception. On the other hand, dense approaches that learn pixel-level representations have been studied mainly in the single-modal setting, e.g., DenseCL [32].

In this work, we propose DenseTwins (Figure 1), the first SSL framework explicitly performing decoupling in a dense latent space, preserving the spatial structure while retaining both shared and modality-specific information. Two modality-specific encoders extract dense feature maps from the input modalities, which are then projected into a structured embedding space. Part of the embedding space captures information shared across modalities (the shared subspace), while the remaining dimensions preserve modalityspecific cues (the modality-specific subspace). This decomposition allows the model to learn spatially consistent multimodal features while retaining complementary information from each sensing modality. Our experiments demonstrate that resulting representations are particularly well-suited for dense perception tasks, while remaining general-purpose representations transferable to global downstream tasks.

![](images/d1253542f449893f9bee98d3ee990c70e015aa8f162abb2802eebe78fa3f3779.jpg)  
Figure 1. The structure of the dense embeddings of DenseTwins. Two separate encoders $\mathcal { F } _ { 1 } , \mathcal { F } _ { 2 }$ process the two input modalities $\mathbf { X } _ { 1 } , \mathbf { X } _ { 2 }$ separately, producing two feature maps $\mathbf { F } _ { 1 } , \mathbf { F } _ { 2 }$ . Finally, two projection heads project $\mathbf { F } _ { 1 } , \mathbf { F } _ { 2 }$ into a common latent space $\mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ,$ each divided into modality-shared $\mathbf { S } _ { 1 } , \mathbf { S } _ { 2 }$ and modality-specific $\mathbf { U } _ { 1 } , \mathbf { U } _ { 2 }$ components.

DenseTwins builds on three established principles: redundancy reduction from Barlow Twins [37], hence its name, modality decoupling [33], and dense correspondence learning [32], combining them within a unified multimodal SSL framework. Distinctive design choices that are key for DenseTwins include: (i) designing a structured dense embedding space that separates shared and modality-specific components spatially, (ii) establishing patch-level crossmodal correspondences within the shared subspace, and (iii) combining dense intra-modal and cross-modal training losses coherently. Specifically, we first establish dense correspondences across modalities to identify semantically related spatial locations. We then align the shared information while decorrelating the modality-specific content. This design enables the model to learn spatial multimodal representations that capture both shared semantics and modalityspecific cues. Since correspondences are estimated directly in feature space, DenseTwins only requires paired observations of the same scene, and can be applied to arbitrary spatially organized modality pairs without modality-specific design choices.

Contributions. Our main contributions are as follows:

• We propose DenseTwins, a multimodal self-supervised framework applicable across heterogeneous spatially organized modalities that jointly learns dense representations that explicitly decouple different modalities.

• We introduce a cross-modal training objective aligning the shared components of semantically corresponding features while preserving modality-specific information in the embeddings.

We evaluate DenseTwins across multiple multimodal datasets and sensing setups (RGB–DEM, RGB–Depth, RGB–LiDAR, RGB–Event, SAR–Optical, RGB–Thermal). Experimental results show that DenseTwins consistently improves overall performance across 2D and 3D semantic segmentation, object detection, and multilabel classification downstream tasks.

## 2. Related Work

SSL methods can be broadly categorized into two groups: (i) Global SSL methods, which learn global representations of the inputs, and (ii) Dense SSL methods, which learn representations at the pixel or patch level. Orthogonally, in multimodal learning, Modality Decoupling approaches explicitly separate the information shared across modalities from the information that is specific to each modality.

## 2.1. Global self-supervised representation learning.

Among global SSL methods, contrastive approaches such as SimCLR [10] maximize agreement between pairs of augmented views of the same input (positive pairs) while pushing apart the negative samples given by the other samples in the mini-batch (negatives). MoCo [19] introduced a momentum encoder and a queue-based dictionary that allows defining negative samples also outside the mini-batch to scale contrastive learning without increasing the batch size. MoCoV2 [20] adopts stronger augmentations and custom projection heads applied to the global representations. Redundancy reduction approaches such as Barlow Twins [37] and VICReg [6] learn representations by enforcing invariance of the feature space to different augmentations. While these frameworks are designed for processing single-modal inputs, they can be extended to multimodal settings. Similarly, CLIP [26] demonstrated the effectiveness of largescale cross-modal contrastive pretraining, aligning global representations across modalities. However, all these alignment strategies focus on modality-shared features and may suppress useful modality-specific information. Moreover, they do not explicitly model spatial structure, limiting their effectiveness for downstream dense prediction tasks.

## 2.2. Dense self-supervised representation learning.

DenseCL [32] introduced dense contrastive learning by establishing correspondences between spatial locations across different augmented views of the same input image, by maximizing their cosine similarity in the feature space. DenseCL++ [22] improves correspondences and training stability, further enhancing performance on dense downstream tasks. Differently, PixelPro [35] computes the normalized distances between features of the pixels of two augmented views and defines positive and negative pairs using a predefined threshold. Recent self-distillation methods such as DINO [9], iBOT [39], and DINOv2 [25] learn representations by matching the outputs of a teacher and a student network under different image augmentations, leveraging either a local-to-global view matching objective [9, 25], or a masked online tokenization objective [25, 39], yielding spatially structured dense representations with remarkable emergent semantic properties. Predictive learning frameworks such as I-JEPA [2] and LeJEPA [4] instead learn dense representations by predicting latent representations of masked image regions, with LeJEPA further providing a provable, heuristic-free training objective. These approaches are designed for single-modal settings, and their extension to arbitrary multimodal input pairs is non-trivial. In contrast, DenseTwins establishes dense correspondences across modalities and applies modality decoupling at the patch level, jointly matching and separating shared and modality-specific content within dense representations.

## 2.3. Modality decoupling.

Modality decoupling in SSL has been studied by De-CUR [33], which explicitly decouples common and modality-specific components using a redundancyreduction objective analogous to Barlow Twins. However, DeCUR [33] operates on global feature vectors and does not preserve spatial structure, limiting its applicability to dense prediction tasks. Other approaches apply modality decoupling for sentiment analysis and person re-identification [16, 17, 23], as well as across audio, video and text [1, 31]; however, these methods are either supervised or designed for non-visual, non-dense tasks.

## 3. Problem Formulation

We consider a multimodal self-supervised learning setting, where each training sample consists of observations of the same scene acquired through different sensing modalities. Formally, we are given an unlabeled dataset

$$
\boldsymbol { \mathcal { D } } = \{ ( \mathbf { X } _ { 1 } ^ { ( i ) } , \mathbf { X } _ { 2 } ^ { ( i ) } ) \} _ { i = 1 } ^ { N } ,
$$

where $\mathbf { X } _ { m } ^ { ( i ) }$ denotes the observation of the i-th scene captured by modality $m \in \{ 1 , 2 \}$ . The two modalities may correspond to heterogeneous sensors (e.g., RGB, depth, thermal, LiDAR, or event-camera acquisitions) and are not required to be perfectly spatially aligned, even though we require synchronized acquisition time.

Our goal, as illustrated in Figure 1, is to learn modalityspecific encoders $\mathcal { F } _ { 1 } , \mathcal { F } _ { 2 }$ , where each encoder $\mathcal { F } _ { m } : \mathbf { X } _ { m } \to$ ${ \bf { F } } _ { m }$ maps an input observation ${ \bf { X } } _ { m }$ to a dense latent representation ${ \bf { F } } _ { m }$ containing $T _ { m }$ feature vectors $\mathbf { f } _ { m , j } \in \mathbb { R } ^ { K }$ We consider any encoder $\mathcal { F } _ { m }$ producing spatially organized representations; thus each vector $\mathbf { f } _ { m , j } \in \mathbb { R } ^ { K }$ corresponds to a localized input region, e.g., a CNN feature-map element (ResNet) or a token of a vision Transformer (ViT). Nonspatial modalities, such as text, are therefore outside the scope of our formulation. Our primary objective is to train $\mathcal { F } _ { 1 }$ and $\mathcal { F } _ { 2 }$ to learn representations suitable for addressing downstream dense prediction tasks, such as semantic segmentation. The challenge is to learn such representations in a fully self-supervised manner, without relying on manual annotations, and from different sensing modalities that often capture complementary information.

## 4. DenseTwins

Figure 2 illustrates the DenseTwins architecture, which consists of two separate branches, one per modality. For each modality input $\mathbf { X } _ { m } .$ , two independently augmented views $( \mathbf { X } _ { m } ^ { ' } , \bar { \mathbf { X } } _ { m } ^ { ' \prime } )$ of $\mathbf { X } _ { m }$ are processed by the same encoder $\mathcal { F } _ { m }$ extracting feature maps $( \mathbf { F } _ { m } ^ { ' } , \mathbf { F } _ { m } ^ { ' \prime } )$ . These are fed to a projection layer $\mathcal { P } _ { m }$ that is composed of: (i) a global projection head $\mathcal { P } _ { m } ^ { g }$ and (ii) a dense projection head ${ \mathcal { P } } _ { m } ^ { d } ,$ both projecting into a $K _ { p }$ -dimensional feature space. The global projection head extracts the global embeddings from $\bigl ( \mathbf { F } _ { m } ^ { ' } , \bar { \mathbf { F } } _ { m } ^ { ' \prime } \bigr )$ by global average pooling, and projects them to $K _ { p }$ -dimensional vectors ${ \bf y } _ { m } ^ { \mathrm { ~ ~ } } , { \bf y } _ { m } ^ { \mathrm { ~ ~ } } .$ The dense projection head preserves spatial structure, necessary for downstream dense prediction tasks, by computing the dense feature maps $( \mathbf { Z } _ { m } ^ { ' } , \mathbf { Z } _ { m } ^ { ' \prime } )$ . Following common practice in SSL, the projection heads operate in a high-dimensional embedding space. This design has been shown to improve representation learning, particularly for redundancy-reduction objectives such as Barlow Twins [37].

Inspired by DeCUR [33], we partition the embedding space into a shared subspace (first $K _ { s }$ components) capturing cross-modal semantics, and a modality-specific subspace (last $K _ { u }$ components) preserving unique cues (Figure 2), extending this decomposition, applied only to global representations in DeCUR, to dense representations. We establish spatial correspondences across modalities by measuring similarity in the shared subspace, while enforcing decorrelation of the modality-specific components.

Downstream inference. Only $\mathcal { F } _ { 1 } , \mathcal { F } _ { 2 }$ are transferred to downstream tasks: $\mathcal { P } _ { 1 } , \mathcal { P } _ { 2 }$ are discarded after pretraining. The separation between $\mathcal { F } _ { 1 }$ and $\mathcal { F } _ { 2 }$ makes DenseTwins agnostic to the fusion strategy; either simple feature concatenation or task-specific architectures can be employed.

![](images/f915c5c2613ecc31ad60f5bbe31c6cfe782eeefe25bed26225d8657afa181263.jpg)  
Figure 2. DenseTwins architecture. For each modality $m \in \{ 1 , 2 \}$ , two augmented views are processed by a modality-specific backbone $\mathcal { F } _ { m }$ , followed by a projection head $\mathcal { P } _ { m }$ that produces both global and dense embeddings. The embedding space is explicitly partitioned into a shared subspace (red) and a modality-specific subspace (blue). Training is based on dense correspondences between feature vectors within each modality and across modalities computed using cosine similarities. The shared components are aligned through redundancy reduction and invariance objectives, while modality-specific components are decorrelated across modalities to preserve complementary information. Intra-modal learning, instead, applies a redundancy-reduction objective, encouraging invariance between augmented views. The final objective combines intra-modal and cross-modal losses applied to both global and dense representations.

Cross-correlation primitive. All training objectives in DenseTwins share a common building block. Given a set of embedding pairs $\{ ( \mathbf { a } _ { b } , \mathbf { b } _ { b } ) \} _ { b = 1 } ^ { B }$ with $\mathbf { a } _ { b } , \mathbf { b } _ { b } \in \mathbb { R } ^ { K _ { p } }$ , we compute a cross-correlation matrix $\Sigma \in \mathbb { R } ^ { K _ { p } \times K _ { p } }$

$$
\Sigma _ { p q } = \frac { \sum _ { b } a _ { b , p } b _ { b , q } } { \sqrt { \sum _ { b } a _ { b , p } ^ { 2 } } \sqrt { \sum _ { b } b _ { b , q } ^ { 2 } } } ,\tag{1}
$$

DenseTwins uses four such matrices: two intra-modal matrices $\bar { I } ^ { ( m ) }$ and $I ^ { ( m ) }$ , computed from global and dense embeddings of the same modality, and two cross-modal matrices $\breve { \bar { C } }$ and $C .$ , computed from embeddings of different modalities. Table 1 summarises the correlation matrices of DenseTwins, all having size $K _ { p } \times K _ { p }$ . For global matrices $( \bar { I } ^ { ( m ) } , \bar { C } )$ , B equals the batch size; for dense matrices $( I ^ { ( m ) } , C )$ , B equals the batch size multiplied by the number of correspondences.

Since each projected embedding is decomposed into a shared component of dimension $K _ { s }$ and a modality-specific component of dimension $K _ { u }$ (with $K _ { s } + K _ { u } = K _ { p }$ , illustrated in Figure 1), the cross-modal matrices $\bar { C }$ and $C$ admit the following block partition:

$$
\bar { C } = \left[ \bar { C } _ { \bar { C } ^ { u s } } ^ { s s } \bar { C } _ { \bar { C } ^ { u u } } ^ { s u } \right] , \qquad C = \left[ C _ { C ^ { u s } } ^ { s s } C ^ { s u } \right] ,\tag{2}
$$

<table><tr><td>Matrix</td><td>Input pairs</td><td>Section</td></tr><tr><td> $\bar { I } ^ { ( m ) }$ </td><td>global intra-modal embeddings</td><td>Sec.4.1</td></tr><tr><td> $I ^ { ( m ) }$ </td><td>dense intra-modal correspondences</td><td>Sec.4.1</td></tr><tr><td> $\bar { C }$ </td><td>global cross-modal embeddings</td><td>Sec. 4.2</td></tr><tr><td>C</td><td>dense cross-modal correspondences</td><td>Sec. 4.2</td></tr></table>

Table 1. Cross-correlation matrices in DenseTwins.

where $( \cdot ) ^ { s s } ~ \in ~ \mathbb { R } ^ { K _ { s } \times K _ { s } }$ captures correlations within the shared subspace, and $( \cdot ) ^ { u u } \in \overline { { \mathbb { R } ^ { K _ { u } \times K _ { u } } } }$ within the modalityspecific subspace.

The overall training objective is computed as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 1 ) } + \mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 2 ) } + \mathcal { L } _ { \mathrm { c r o s s } } ,\tag{3}
$$

where $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 1 ) }$ and $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 2 ) }$ assess consistency between augmented views of the same modality (Sec. 4.1), and $\mathcal { L } _ { \mathrm { c r o s s } }$ aligns the shared subspace while decorrelating the modality-specific ones (Sec. 4.2).

## 4.1. Intra-Modal Loss

As in SimCLR [10], two augmented views of the same modality are generated, and we replicate this process for each modality. Intra-modal learning enforces consistency between different views of the same image, learning invariant representations that capture semantic content.

Intra-Modal Loss. To achieve invariance in the augmented views, the intra-modal loss is a convex combination of both global and dense redundancy reduction losses [37]:

$$
\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( m ) } = ( 1 - \alpha ) \mathcal { L } _ { g } ^ { ( m ) } + \alpha \mathcal { L } _ { d } ^ { ( m ) } ,\tag{4}
$$

where $\alpha \in [ 0 , 1 ]$ balances $\mathcal { L } _ { g } ^ { ( m ) }$ and $\mathcal { L } _ { d } ^ { ( m ) }$ . This loss is computed for each modality m $\in \{ 1 , 2 \}$ independently, resulting in two intra-modal losses, $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 1 ) }$ and $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 2 ) }$

Global redundancy-reduction. The global redundancy reduction loss is applied to the global embeddings within the same modality. Given a batch of paired global embeddings $( \mathbf { y } ^ { \prime } , \mathbf { y } ^ { \prime \prime } )$ of the two augmented views from the same modality, we compute $\bar { I } ^ { ( m ) }$ as in (1). The loss is then defined as

$$
\mathcal { L } _ { g } ^ { ( m ) } = \sum _ { p } ( 1 - \bar { I } _ { p p } ^ { ( m ) } ) ^ { 2 } + \lambda \sum _ { p \neq q } ( \bar { I } _ { p q } ^ { ( m ) } ) ^ { 2 } .\tag{5}
$$

By minimizing $\mathcal { L } _ { g } ^ { ( m ) }$ , we enforce: (i) Invariance, by pushing diagonal entries toward one to ensure both views have similar embeddings, and (ii) Redundancy reduction, by forcing the off-diagonal entries towards zero.

Dense token alignment and redundancy reduction. Let $\mathbf { F ^ { \prime } }$ and $\mathbf { F } ^ { \prime \prime }$ be spatial features returned by the backbone $\mathcal { F }$ from the two augmented views of the same modality:

$$
\mathbf { F } ^ { \prime } = \{ \mathbf { f } _ { i } ^ { \prime } \} _ { i = 1 } ^ { T } , \quad \mathbf { F } ^ { \prime \prime } = \{ \mathbf { f } _ { j } ^ { \prime \prime } \} _ { j = 1 } ^ { T } , \quad \mathbf { f } _ { i } ^ { \prime } , \mathbf { f } _ { j } ^ { \prime \prime } \in \mathbb { R } ^ { K } .
$$

Similarly, let $\mathbf { Z } ^ { \prime }$ and $\mathbf { Z } ^ { \prime \prime }$ be the output of the dense projection head for the two augmented views:

$$
\mathbf { Z } ^ { \prime } = \{ \mathbf { z } _ { i } ^ { \prime } \} _ { i = 1 } ^ { T } , \quad \mathbf { Z } ^ { \prime \prime } = \{ \mathbf { z } _ { j } ^ { \prime \prime } \} _ { j = 1 } ^ { T } , \quad \mathbf { z } _ { i } ^ { \prime } , \mathbf { z } _ { j } ^ { \prime \prime } \in \mathbb { R } ^ { K _ { p } } .
$$

To extend redundancy reduction to dense representation, we establish correspondences between every single spatial feature $ { \mathbf { z } } _ { i } ^ { \prime } \in  { \mathbf { Z } } ^ { \prime }$ and one vector of $\mathbf { Z } ^ { \prime \prime }$ . This association needs to preserve semantics: we require the two corresponding tokens to refer to semantically similar parts of the input (e.g. regions referring to the same type of object in RGB images). However, augmentations alter the spatial layouts, and tokens at the same spatial index might not correspond to the same semantic region.

To address this spatial misalignment, we establish correspondences in the backbone representation space by computing the pairwise cosine similarity between embeddings from the two augmented views:

$$
\Delta _ { i j } = \frac { \mathbf { f } _ { i } ^ { \prime \top } \mathbf { f } _ { j } ^ { \prime \prime } } { \Vert \mathbf { f } _ { i } ^ { \prime } \Vert \Vert \mathbf { f } _ { j } ^ { \prime \prime } \Vert } .\tag{6}
$$

We define a set of matches approximating semantic correspondences across views. In practice, each token $\mathbf { f } _ { i } ^ { \prime }$ in the first view is matched with the most similar token in the second view:

$$
j _ { i } ^ { * } = \arg \operatorname* { m a x } _ { j } \Delta _ { i j } .\tag{7}
$$

We then retrieve the corresponding projected embeddings $( \mathbf { z } _ { i } ^ { \prime } , \mathbf { z } _ { j _ { i } ^ { * } } ^ { \prime \prime } )$ and compute an intra-modality cross-correlation matrix $I ^ { ( m ) } \in \mathbb { R } ^ { K _ { p } \times K _ { \tau } }$ over matched dense token pairs as in (1). The redundancy-reduction loss in (5), which we denote by $\mathcal { L } _ { d } ^ { ( m ) }$ , is applied to $I ^ { ( m ) }$

## 4.2. Cross-Modal Loss

Cross-modal learning leverages the relationship between representations extracted from different modalities. The objective is twofold: (i) align information that is shared across modalities, and (ii) prevent modality-specific information from collapsing into the shared representation.

Let $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathbb { R } ^ { K _ { p } }$ denote the global projected embeddings of the two modalities, and $\mathbf { \bar { Z } } _ { 1 } \ = \ \{ \mathbf { z } _ { 1 , i } \} _ { i = 1 } ^ { T _ { 1 } } , \ \mathbf { Z } _ { 2 } \ =$ $\{ \mathbf { z } _ { 2 , j } \} _ { j = } ^ { T _ { 2 } }$ the dense projected embeddings.

Embedding space decomposition. The projector embedding space is partitioned into a shared subspace of dimension $K _ { s }$ and a modality-specific subspace of dimension $K _ { u }$ with $K _ { p } = K _ { s } + K _ { u }$ , as illustrated in Figures 1 and 2:

$$
\mathbf { z } = \left[ \mathbf { \bar { s } } \right] , \quad \mathbf { s } \in \mathbb { R } ^ { K _ { s } } , \quad \mathbf { u } \in \mathbb { R } ^ { K _ { u } } .
$$

This decomposition is applied to both global $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 }$ and dense $\mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 }$ embeddings.

Final cross-modal objective. The cross-modal loss function combines global and dense components, both enforcing modality decoupling and decorrelation on global and dense embeddings:

$$
\mathcal { L } _ { \mathrm { c r o s s } } = ( 1 - \alpha ) \mathcal { L } _ { \mathrm { c r o s s } } ^ { g } + \alpha \mathcal { L } _ { \mathrm { c r o s s } } ^ { d } .\tag{8}
$$

Global cross-modal loss. As in DeCUR [33], the crosscorrelation matrix $\bar { C }$ is computed from global embeddings $\left( \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \right)$ via (1), and partitioned into $\bar { \bar { C } } ^ { s s }$ and $\bar { C } ^ { u u }$ as in (2). The goal is to align the first $K _ { s }$ components of $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 }$ and make the last $K _ { u }$ components uncorrelated to promote learning modality-specific features. Thus, we apply an invariance and redundancy reduction loss as in (5) to $\bar { C } ^ { s s }$ , and enforce full decorrelation on $\bar { C } ^ { u u }$

$$
\mathcal { L } _ { s s } ^ { g } = { \sum _ { p = 1 } ^ { K _ { s } } } ( 1 - \bar { C } _ { p p } ^ { s s } ) ^ { 2 } + \lambda { \sum _ { p \neq q } ^ { K _ { s } } } ( \bar { C } _ { p q } ^ { s s } ) ^ { 2 } ,\tag{9}
$$

$$
\mathcal { L } _ { u u } ^ { g } = { \sum _ { p = 1 } ^ { K _ { u } } } ( \bar { C } _ { p p } ^ { u u } ) ^ { 2 } + \lambda { \sum _ { p \neq q } ^ { K _ { u } } } ( \bar { C } _ { p q } ^ { u u } ) ^ { 2 } .\tag{10}
$$

The global cross-modal loss is finally defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c r o s s } } ^ { g } = \mathcal { L } _ { s s } ^ { g } + \mathcal { L } _ { u u } ^ { g } . } \end{array}\tag{11}
$$

Dense cross-modal decoupling. The goal for cross-modal dense representation decoupling is similar to the global case: training the shared subspace of the embedding to be invariant across modalities and the modality-specific subspace to be uncorrelated. Similarly to the intra-modal case, we first establish cross-modal correspondences between tokens of the two modalities. However, as the embeddings are split into modality-shared and specific, we use only the shared subspace to define correspondences since this is the subspace where cross-modal alignment is expected.

For each feature vector $\mathbf { z } _ { 1 , }$ <sub>i</sub>, we extract its common component ${ \mathbf { s } } _ { 1 , i }$ and compute the cosine similarity $\Delta _ { i j } ^ { \mathrm { c r o s s } }$ with common components $\mathbf { s } _ { 2 , j }$ from the second modality as in (6). Then we select, for each spatial location i in the first modality, the most similar location $j _ { i } ^ { * }$ in the second modality as in (7). Using these matched token pairs $\left( \mathbf { z } _ { 1 , i } , \mathbf { z } _ { 2 , j _ { i } ^ { * } } \right)$ we compute a cross-correlation matrix $C \in \mathbb { R } ^ { K _ { p } \times K _ { p } }$ over the full projected embeddings (both shared and modalityspecific components), exactly as in the global case, and partition it into blocks $C ^ { s s }$ and $C ^ { u u }$ , as in (2).

The same combination of invariance, redundancy reduction and decorrelation used at the global level in (11) is then applied to $C ^ { s s }$ and $C ^ { u u }$ , yielding the following dense crossmodal loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c r o s s } } ^ { d } = \mathcal { L } _ { s s } ^ { d } + \mathcal { L } _ { u u } ^ { d } . } \end{array}\tag{12}
$$

As correspondences are noisy early in training, we apply a linear warmup of N epochs to the weight α of $\mathcal { L } _ { \mathrm { c r o s s } } ^ { d } .$

## 5. Experiments

Datasets. We evaluate DenseTwins across six multimodal sensing pairs, covering semantic segmentation, object detection, and multilabel classification: RGB– Depth (SUN-RGBD [28]), RGB–Event (EventScape [13], DSEC [14]), RGB–LiDAR (Semantic-KITTI [8]), RGB– DEM (GeoNRW [3]), SAR–Optical (BigEarthNet [29]), and RGB–Thermal (KAIST [21]). Preprocessing and pretraining details for each dataset are provided in the supplementary material (Section A.1).

Competitors. We adapt SimCLR [10], Barlow Twins [37] and VICReg [6] to the multimodal setting, applying the two modalities in place of the two augmented views normally required by these methods. We also consider CLIP [26] and DeCUR [33], which are natively multimodal. All these methods operate on global representations; thus, we take the last dense feature map before pooling. Finally, we compare with single-modal methods: DINO [9], iBOT [39], I-JEPA [2] and LeJEPA [4], which we train independently on each modality. Further details are provided in the supplementary material (Section A.3).

## 5.1. Evaluation Protocols

We test under two complementary evaluation protocols (additional details are available in Section A.2).

Table 2. SUN-RGBD and EventScape (CMX fine-tuning).
<table><tr><td rowspan="2">Pretrain</td><td colspan="2">SUN-RGBD mIoU</td><td>EventScape mIoU</td></tr><tr><td>10% labels</td><td>100% labels</td><td>100% labels</td></tr><tr><td>None</td><td>35.7</td><td>49.7</td><td>61.9</td></tr><tr><td>SimCLR</td><td>37.6</td><td>50.8</td><td>63.2</td></tr><tr><td>CLIP</td><td>37.6</td><td>50.6</td><td>63.5</td></tr><tr><td>BarlowTwins</td><td>37.8</td><td>51.0</td><td>63.0</td></tr><tr><td>VICReg</td><td>37.7</td><td>51.0</td><td>62.4</td></tr><tr><td>DeCUR</td><td>36.9</td><td>50.6</td><td>62.3</td></tr><tr><td>DenseTwins</td><td>38.2</td><td>51.5</td><td>63.3</td></tr></table>

Frozen-backbone probing. We assess the quality of the learned representations by freezing both pretrained encoders and training only a downstream predictor. Since all methods learn one encoder per modality, their feature maps are concatenated before the task-specific head. For semantic segmentation, we evaluate increasingly expressive predictors: k-nearest-neighbours (kNN), linear probing (LP), and a lightweight FCN. kNN and LP operate directly at the feature-map resolution, with predictions subsequently upsampled to the input resolution. For object detection, we train a lightweight FCOS [30] head on the frozen features, while multilabel classification is evaluated through linear probing. All competing methods use the same backbone to ensure a controlled comparison: unless otherwise specified, we use ViT-S as iBOT and I-JEPA are mainly based on Transformer-based architectures.

Fine-tuning. Following DeCUR [33], we also evaluate how effectively SSL provides an initialization for downstream training when all transferred parameters are optimized jointly for the downstream task. As in DeCUR, for 2D semantic segmentation, we adopt CMX [38] with MiT-B2 [34] as backbone, and FCN [24] with ResNet50 [18]. For SemanticKITTI, we fine-tune PMF [40], whose RGB branch uses ResNet34 and transfer only the pretrained RGB backbone, as PMF’s early RGB-LiDAR fusion strategy is not compatible with an independent LiDAR backbone.

Metrics. We report the mean Intersection over Union (mIoU) for semantic segmentation, Average Precision at 50% of IoU (AP50) for object detection, and mean Average Precision (mAP) for multilabel classification.

## 5.2. 2D Semantic Segmentation

We evaluate 2D semantic segmentation on SUN-RGBD [28] and EventScape [13]. On SUN-RGBD, we consider both full supervision and a 10% low-label setting, together with frozen-backbone probing on the reduced 13-class label set [12]. EventScape is evaluated under full-supervision fine-tuning.

DenseTwins achieves the best fine-tuned mIoU on SUN-

Table 3. SUN-RGBD frozen probing (NYU-13 class label set).
<table><tr><td>Pretrain</td><td>kNN</td><td>LP</td><td>FCN</td></tr><tr><td>SimCLR</td><td>22.4</td><td>25.1</td><td>30.4</td></tr><tr><td>CLIP</td><td>20.3</td><td>22.3</td><td>29.9</td></tr><tr><td>Barlow Twins</td><td>18.5</td><td>20.2</td><td>27.0</td></tr><tr><td>VICReg DeCUR</td><td>18.6 22.0</td><td>22.7 23.7</td><td>28.0 35.8</td></tr><tr><td>DINO</td><td>25.6</td><td></td><td>37.6</td></tr><tr><td>iBOT</td><td>24.9</td><td>31.0 30.8</td><td>36.3</td></tr><tr><td>I-JEPA</td><td>27.4</td><td>31.1</td><td>35.2</td></tr><tr><td>LeJEPA</td><td>18.8</td><td>23.9</td><td>32.7</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>DenseTwins</td><td>29.1</td><td>33.7</td><td>40.6</td></tr></table>

RGBD at both label budgets (Table 2). Despite also decoupling modalities, DeCUR underperforms most baselines, suggesting that global-only decoupling is less effective for dense tasks. On EventScape, DenseTwins reaches 63.3 mIoU, above all redundancy-reduction baselines; only CLIP (63.5) scores slightly higher. Qualitative CMX segmentations are in Section E. Under frozen-backbone probing on SUN-RGBD (Table 3), DenseTwins outperforms every baseline by a wide margin, demonstrating the quality of the learned representations independently of downstream fine-tuning. We further analyze single-modal encoders in isolation in Section D.2, showing competitive results even when only one modality is available at test time.

## 5.3. Object Detection for Autonomous Driving

Frozen-backbones evaluation for object detection on KAIST [21] and DSEC [14] shows that DenseTwins achieves the best overall AP50 on both benchmarks under the same backbone (Table 4). We additionally report the results of DenseTwins using Swin-T and ConvNeXtV2-T to demonstrate further improvements. These results highlight that the benefits of dense modality decoupling extend beyond semantic segmentation and generalize across backbone families. Notably, global-representation methods substantially lag behind DenseTwins and the dense singlemodal methods on both benchmarks, suggesting that object detection benefits more from dense, patch-level pretraining than from a single global representation.

## 5.4. Remote Sensing

We evaluate DenseTwins on GeoNRW [3] (RGB–DEM segmentation), under both fine-tuning and frozen-probing protocols, and BigEarthNet [29] (SAR–Optical multilabel classification). As shown in Table 5, DenseTwins gives the best low-label GeoNRW mIoU and the best frozen-probing results across all three protocols, while remaining competitive with all baselines at full supervision. On BigEarthNet,

Table 4. Object detection AP50 on KAIST and DSEC. Baselines use ViT-S/16; DenseTwins also uses Swin-T and ConvNeXtV2-T.
<table><tr><td></td><td></td><td colspan="2">KAIST AP50</td><td colspan="3">DSEC AP50</td></tr><tr><td>Pretrain</td><td>Backbone</td><td>All</td><td>Reasonable</td><td>All</td><td>Car</td><td>Pedestrian</td></tr><tr><td>DenseTwins</td><td>Swin-T</td><td>33.3</td><td>35.3</td><td>24.3</td><td>51.1</td><td>11.9</td></tr><tr><td>DenseTwins</td><td>ConvNeXtV2-T</td><td>31.2</td><td>32.9</td><td>21.3</td><td>47.1</td><td>9.7</td></tr><tr><td>DenseTwins</td><td>ViT-S/16</td><td>26.8</td><td>29.8</td><td>20.6</td><td>46.0</td><td>9.3</td></tr><tr><td>SimCLR</td><td>ViT-S/16</td><td>16.8</td><td>19.2</td><td>8.4</td><td>36.7</td><td>6.8</td></tr><tr><td>CLIP</td><td>ViT-S/16</td><td>16.5</td><td>18.8</td><td>8.2</td><td>31.0</td><td>5.1</td></tr><tr><td>Barlow Twins</td><td>ViT-S/16</td><td>12.0</td><td>14.8</td><td>7.8</td><td>30.0</td><td>4.9</td></tr><tr><td>VICReg</td><td>ViT-S/16</td><td>13.1</td><td>15.1</td><td>8.5</td><td>34.6</td><td>5.2</td></tr><tr><td>DeCUR</td><td>ViT-S/16</td><td>12.0</td><td>13.8</td><td>7.9</td><td>31.6</td><td>4.7</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>25.8</td><td>30.6</td><td>18.4</td><td>44.6</td><td>12.0</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>24.8</td><td>29.2</td><td>16.2</td><td>43.5</td><td>9.2</td></tr><tr><td>I-JEPA</td><td>ViT-S/16</td><td>25.8</td><td>29.2</td><td>14.9</td><td>37.9</td><td>7.8</td></tr><tr><td>LeJEPA</td><td>ViT-S/16</td><td>19.8</td><td>22.4</td><td>14.1</td><td>38.1</td><td>7.0</td></tr></table>

Table 5. GeoNRW (fine-tuned FCN and frozen probing) and BigEarthNet (frozen-probing mAP).
<table><tr><td rowspan="2">Pretrain</td><td colspan="2">GeoNRW (fine-tuned FCN)</td><td colspan="3">GeoNRW (frozen probe)</td><td>BigEarthNet</td></tr><tr><td>1% mIoU</td><td>100% mIoU</td><td>kNN</td><td>LP</td><td>FCN</td><td>mAP</td></tr><tr><td>Random</td><td>26.2</td><td>35.4</td><td>16.1</td><td>19.5</td><td>30.1</td><td>0.601</td></tr><tr><td>SimCLR</td><td>40.6</td><td>55.8</td><td>22.5</td><td>27.9</td><td>42.9</td><td>0.809</td></tr><tr><td>CLIP</td><td>40.8</td><td>55.2</td><td>23.8</td><td>28.9</td><td>42.6</td><td>0.819</td></tr><tr><td>BarlowTwins</td><td>40.8</td><td>52.1</td><td>19.2</td><td>23.7</td><td>41.8</td><td>0.797</td></tr><tr><td>VICReg</td><td>41.0</td><td>56.3</td><td>24.5</td><td>27.9</td><td>43.9</td><td>0.803</td></tr><tr><td>DeCUR</td><td>41.5</td><td>54.8</td><td>30.6</td><td>34.7</td><td>45.5</td><td>0.831</td></tr><tr><td>DenseTwins</td><td>41.7</td><td>56.0</td><td>36.6</td><td>40.5</td><td>46.1</td><td>0.834</td></tr></table>

Table 6. SemanticKITTI: PMF (fine-tuned) and MinkUNet+ViT (frozen probing, FOV/Full mIoU).
<table><tr><td rowspan="2">Pretrain</td><td rowspan="2">PMF</td><td colspan="2">MinkUNet+ViT</td></tr><tr><td>mIoU FOV mIoU</td><td>Full mIoU</td></tr><tr><td>Random</td><td>63.9</td><td>31.1</td><td>31.8</td></tr><tr><td>DeCUR</td><td>65.0</td><td>44.8</td><td>44.4</td></tr><tr><td>DenseTwins</td><td>65.2</td><td>47.1</td><td>44.1</td></tr></table>

DenseTwins achieves the best classification performance, outperforming CLIP, SimCLR and DeCUR, confirming that the benefits extend to image-level recognition as well.

## 5.5. RGB-LiDAR 3D Semantic Segmentation

We evaluate the RGB–LiDAR setting on Semantic-KITTI [8], both by fine-tuning PMF [40] and by using frozen backbones. For the latter, we use MinkUNet [11] as 3D LiDAR backbone, which we fuse with ViT RGB features as in PointFusion [36]. We report in Table 6, for frozen-backbone probing, the mIoU both in the region of the 3D scan inside the camera field of view (FOV mIoU) and in the full 360° scan (mIoU). DenseTwins leads in FOV mIoU, where RGB features are available for every point, while DeCUR is marginally better in mIoU, where most points fall outside the camera field of view and therefore lack RGB features, suggesting that DenseTwins’ frozen representations are particularly effective where both modalities are actually available. DenseTwins improves on DeCUR also when fine-tuning.

Table 7. Effect of the global and dense projectors on SUN-RGBD.
<table><tr><td>Global Proj. Dense Proj.</td><td></td><td>mIoU</td><td>LP mIoU</td></tr><tr><td>x</td><td>x</td><td>50.7</td><td>28.3</td></tr><tr><td>x</td><td>√</td><td>50.9</td><td>29.1</td></tr><tr><td>√</td><td>x</td><td>51.1</td><td>32.0</td></tr><tr><td>√</td><td>√</td><td>51.5</td><td>33.7</td></tr></table>

Table 8. Effect of the loss components on SUN-RGBD.
<table><tr><td>Cm</td><td>Lm</td><td> $\mathcal { L } _ { \mathrm { c r o s s } } ^ { g }$ </td><td></td><td>mIoU LP mIoU</td></tr><tr><td>√</td><td>x</td><td>√</td><td>x</td><td>50.6</td></tr><tr><td>x</td><td>√</td><td>x</td><td>√</td><td>51.3</td></tr><tr><td>x</td><td>√</td><td>√</td><td>√</td><td>25.3 51.4 31.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>51.5</td></tr></table>

![](images/489b3c0d34417aeafda5ad5bda575540daae56832ab22f926b0ed8554dbc2081.jpg)  
Figure 3. Effect of the percentage of shared features on SUN-RGBD and GeoNRW.

![](images/9e4c62728f7ede246b89dae8f41018af6a07291a053e86cfabd2de29d22cb27c.jpg)  
Figure 4. Effect of the dimensionality of $K _ { p } ,$ on SUN-RGBD and GeoNRW.

## 5.6. Ablation Studies

We perform a series of ablation studies to analyze the impact of the main design choices of our framework, including the dimensionality of the projection heads, the percentage of modality-shared features, and the role of the global and dense objectives, using SUN-RGBD [28] and, for the first two ablations, also GeoNRW [3].

Projector dimensionality. We first analyze the effect of the projector dimensionality $K _ { p }$ . As shown in Figure 4, increasing $K _ { p }$ generally improves performance, suggesting that larger projection spaces allow the model to better capture multimodal correspondences.

Decoupling percentage. We then investigate the ratio $K _ { s } / K _ { p }$ of modality-shared components. As shown in Figure 3, the optimum varies across modality pairs: $K _ { s }$ ≈ 75%, $K _ { p }$ for RGB-Depth and $K _ { s } \approx 6 2 \% , K _ { p }$ for GeoNRW. Too few shared components limit cross-modal alignment, whereas sharing all features degrades performance by preventing the model from preserving modality-specific information. Automatically adapting this split to each modality pair is left for future work.

Effect of the projector heads. We further study the contribution of the global and dense projection heads. As shown in Table 7, both components positively contribute to performance, under both the fine-tuned CMX mIoU and the frozen linear-probe (LP) mIoU on SUN-RGBD. The global projector contributes more individually (+0.4 mIoU) than the dense one (+0.2 mIoU), yet their combination yields a larger joint gain (+0.8 mIoU), confirming complementary roles; the same trend holds under linear probing, with an even larger gap (+5.4 LP mIoU). This benefit comes at a moderate computational cost, quantified in Section D.1.

Effect of the loss components. Finally, we analyze the impact of the different training loss terms. As reported in Table 8, both the global and dense objectives contribute to the final performance, with the dense losses $( \mathcal { L } _ { d } ^ { ( m ) }$ and $\mathcal { L } _ { \mathrm { c r o s s } } ^ { d } )$ providing the largest gain. In particular, training with dense objectives only already outperforms the global-only configuration by +0.7 mIoU (51.3 vs. 50.6), confirming that dense supervision is the core contribution of DenseTwins; adding the global objectives further improves performance under both fine-tuning and linear probing, showing that the two scales of supervision are complementary. We further verify in Section D.3 that the dense correspondences underlying $\mathcal { L } _ { d } ^ { ( m ) }$ and $\mathcal { L } _ { \mathrm { c r o s s } } ^ { d }$ remain robust under moderate spatial misalignment between modalities.

Beyond the quantitative ablations presented in this section, we provide additional analyses of the learned shared and modality-specific subspaces in the supplementary material to verify that DenseTwins successfully disentangles shared and modality-specific information (Section C).

## 5.7. Discussion

A consistent trend across the experiments is that contrastive methods such as SimCLR and CLIP remain among the strongest baselines, often outperforming redundancyreduction approaches such as Barlow Twins, VICReg, and even DeCUR. This suggests that contrastive learning may be worth exploring for aligning the shared subspace in place of redundancy reduction, and indicates that dense correspondences play an important role that these methods lack.

DenseTwins nonetheless achieves the best overall results across modality pairs, application domains, and evaluation protocols, from fine-tuning to frozen-backbone probing and object detection, indicating that dense modality decoupling captures information complementary to existing self-supervised objectives, rather than replacing them.

## 6. Conclusion

In this paper, we presented DenseTwins, a multimodal selfsupervised framework for learning dense representations while explicitly separating modality-shared and modalityspecific information. DenseTwins extends redundancyreduction and modality-decoupling principles from global embeddings to dense token-level representations by establishing dense correspondences and combining intra-modal and cross-modal objectives. Through experiments across multiple sensing setups, we showed that DenseTwins consistently improves over existing self-supervised baselines on downstream tasks. These results demonstrate the effectiveness of dense modality decoupling for learning transferable multimodal representations.

## Acknowledgements

This paper is supported by FAIR (NextGenerationEU program, PNRR-PE-AI scheme, M4C2, Investment 1.3, Line on Artificial Intelligence) and by PEPR IA on grant SHARP ANR-23-PEIA-0008. We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer Karolina hosted by IT4Innovations, Czech Republic through an EuroHPC Development Access call. Part of this work was granted access to the JeanZay HPC/AI resources of IDRIS under the allocation AD011017661.

## References

[1] Jean-Baptiste Alayrac, Adria Recasens, Rosalia Schneider, Relja Arandjelovic, Jason Ramapuram, Jeffrey De Fauw, Lu-´ cas Smaira, Sander Dieleman, and Andrew Zisserman. Selfsupervised multimodal versatile networks. Advances in neural information processing systems, 33:25–37, 2020. 3

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023. 3, 6, 1

[3] Gerald Baier, Antonin Deschemps, Michael Schmitt, and Naoto Yokoya. Geonrw, 2020. 6, 7, 8, 1

[4] Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025. 3, 6

[5] Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippeˇ Morency. Multimodal machine learning: A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2):423–443, 2019. 1

[6] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for selfsupervised learning. arXiv preprint arXiv:2105.04906, 2021. 1, 2, 6, 3

[7] Khaled Bayoudh. A survey of multimodal hybrid deep learning for computer vision: Architectures, applications, trends, and challenges. Information Fusion, 105:102217, 2024. 1

[8] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall. SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences. In Proc. of the IEEE/CVF International Conf. on Computer Vision (ICCV), 2019. 6, 7, 1

[9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 3, 6

[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning (ICML), 2020. 1, 2, 4, 6, 3

[11] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural

networks. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 3070–3079. IEEE, 2019. 7, 2

[12] Camille Couprie, Clement Farabet, Laurent Najman, and´ Yann LeCun. Indoor semantic segmentation using depth in formation. arXiv preprint arXiv:1301.3572, 2013. 6

[13] Daniel Gehrig, Michelle Ruegg, Mathias Gehrig, Javier¨ Hidalgo-Carrio, and Davide Scaramuzza. Combining events and frames using recurrent asynchronous multimodal networks for monocular depth prediction. IEEE Robotic and Automation Letters. (RA-L), 2021. 6, 1

[14] Mathias Gehrig, Willem Aarents, Daniel Gehrig, and Davide Scaramuzza. Dsec: A stereo event camera dataset for driving scenarios. IEEE Robotics and Automation Letters, 6(3): 4947–4954, 2021. 6, 7, 1

[15] Saurabh Gupta, Ross Girshick, Pablo Arbelaez, and Jiten-´ dra Malik. Learning rich features from rgb-d images for object detection and segmentation. In European conference on computer vision, pages 345–360. Springer, 2014. 1

[16] Xin Hao, Sanyuan Zhao, Mang Ye, and Jianbing Shen. Cross-modality person re-identification via modality confusion and center aggregation. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 16383–16392. IEEE, 2021. 3

[17] Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. Misa: Modality-invariant and -specific representations for multimodal sentiment analysis. Proceedings of the 28th ACM International Conference on Multimedia, 2020. 3

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778. IEEE, 2016. 6, 2

[19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2020. 2

[20] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Moco v2: Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 2

[21] Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon. Multispectral pedestrian detection: Benchmark dataset and baselines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1037–1045. IEEE, 2015. 6, 7, 1

[22] Berk Iskender, Zhenlin Xu, Simon Kornblith, En-Hung Chu, and Maryam Khademi. Improving dense contrastive learning with dense negative pairs. arXiv preprint arXiv:2210.05063, 2022. 3

[23] Songning Lai, Jiakang Li, Guinan Guo, Xifeng Hu, Yulong Li, Yuan Tan, Zichen Song, Yutong Liu, Zhaoxia Ren, Zhi Liu, Chun Wang, and Danmin Miao. Shared and private information learning in multimodal sentiment analysis with deep modal alignment and self-supervised multi-task learn ing. 2024 International Joint Conference on Neural Net works (IJCNN), pages 1–8, 2023. 3

[24] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings ofthe IEEE conference on computer vision andpattern recognition, pages 3431–3440, 2015. 6, 3

[25] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 2, 6, 3

[27] Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M Savaresi, Luca Magri, and Giacomo Boracchi. Lcf3d: A robust and real-time late-cascade fusion framework for 3d object detection in autonomous driving. Pattern Recognition, page 113046, 2026. 1

[28] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 567–576, 2015. 6, 8, 1

[29] Gencer Sumbul, Marcela Charfuelan, Begum Demir, and¨ Volker Markl. Bigearthnet: A large-scale benchmark archive for remote sensing image understanding. arXiv preprint arXiv:1902.06148, 2019. 6, 7, 1

[30] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In 2019 IEEE/CVF international conference on computer vision (ICCV), pages 9626–9635. Ieee, 2019. 6, 2

[31] Chenyu Wang, Sharut Gupta, Xinyi Zhang, Sana Tonekaboni, Stefanie Jegelka, Tommi Jaakkola, and Caroline Uhler. An information criterion for controlled disentanglement of multimodal data. arXiv preprint arXiv:2410.23996, 2024. 3

[32] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1, 2, 3

[33] Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham, Chenying Liu, Zhitong Xiong, and Xiao Xiang Zhu. Decoupling common and unique representations for multimodal self-supervised learning (decur). In Computer Vision – ECCV 2024, Lecture Notes in Computer Science, vol. 15087. Springer, 2024. 1, 2, 3, 5, 6

[34] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077–12090, 2021. 6, 1

[35] Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16684–16693, 2021. 3

[36] Danfei Xu, Dragomir Anguelov, and Ashesh Jain. Pointfusion: Deep sensor fusion for 3d bounding box estimation. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 244–253. IEEE, 2018. 7, 2

[37] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 12310–12320, 2021. 1, 2, 3, 5, 6

[38] Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruip ing Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers. IEEE Transactions on intelligent transportation systems, 24(12): 14679–14694, 2023. 6, 1, 2

[39] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021. 3, 6

[40] Zhuangwei Zhuang, Rong Li, Kui Jia, Qicheng Wang, Yuanqing Li, and Mingkui Tan. Perception-aware multi-sensor fusion for 3d lidar semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16280–16290, 2021. 6, 7, 3

[41] Yongshuo Zong, Oisin Mac Aodha, and Timothy M Hospedales. Self-supervised multimodal learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(7):5299–5318, 2024. 1

# DenseTwins: Dense Modality Decoupling for Multimodal Self-Supervision

Supplementary Material

In this document, we provide additional details about the implementation of DenseTwins for the multimodal datasets (SUN-RGBD, KAIST, DSEC, EventScape, GeoNRW, SemanticKITTI and BigEarthNet) used for testing. Additionally, we show the pseudocode of the training procedure of DenseTwins, an analysis of the learned shared/specific decomposition, an analysis of the computational overhead introduced by the projection heads and the dense matching mechanism, and some visualizations of the downstream results of CMX [38] on SUN-RGBD and EventScape.

## A. Implementation Details

## A.1. Pretraining

We pretrain multimodal backbones using DenseTwins on seven multimodal datasets covering different sensing modalities: RGB–Depth (SUN-RGBD [28]), RGB– Event (EventScape [13], DSEC [14]), RGB–Thermal (KAIST [21]), RGB–DEM (GeoNRW [3]), RGB–LiDAR (Semantic-KITTI [8]), and SAR–Optical (BigEarthNet [29]). All pretraining experiments are run on 8×A100 GPUs; unless otherwise noted, batch sizes are reported per GPU. For each dataset, two independently augmented views are generated per modality following standard selfsupervised learning practices. Augmentations are applied independently to each modality to generate two views per input pair.

SUN-RGBD (RGB–Depth). SUN-RGBD consists of indoor RGB images paired with depth maps captured using different RGB-D sensors. Following common practice in RGB-D scene understanding, the depth maps are preprocessed into the HHA representation [15], which encodes horizontal disparity, height above ground, and angle with gravity. Following DeCUR [33], for the fine-tuning protocol both RGB and HHA images are resized using random resized cropping to a spatial resolution of 224 × 224 pixels, using the same augmentations as DeCUR: random resized cropping, horizontal flipping, color jittering and Gaussian blur for RGB images, and geometric augmentations only (random cropping and horizontal flipping) for HHA images. For this protocol we adopt MiT-B2 encoders following the SegFormer architecture [34], with a projector composed of two convolutional layers with kernel size 1. Pretraining is performed using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 5 }$ , batch size 96, for 200 epochs. For the frozenbackbone probing protocol, we instead pretrain a ViT-S/14 backbone (14-pixel patch size) at the original image resolu tion (640 × 480), using the same augmentations, with batch

size 64 for 200 epochs.

EventScape (RGB–Event). EventScape provides synchronized RGB frames and event streams captured in driving scenarios. Event streams are converted into voxel grid representations by discretizing the temporal dimension into three bins. The resulting voxel grids are stored as threechannel images, allowing them to be processed similarly to RGB inputs. Both RGB images and event voxel grids are resized using random resized cropping to $2 2 4 \times 2 2 4$ pixels. RGB images are augmented using random resized cropping, horizontal flipping, color jittering and Gaussian blur, while event voxel grids undergo geometric augmentations such as random cropping and horizontal flipping. Pretraining is performed using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 5 }$ , batch size 96, for 50 epochs.

KAIST (RGB–Thermal). KAIST is a multimodal pedestrian-detection benchmark for autonomous driving, providing RGB images paired with thermal images and object-detection annotations for pedestrians. Thermal images are single-channel (grayscale); we repeat the channel three times to obtain a 3-channel input. We pretrain ViT-S/16, Swin-T, and ConvNeXtV2-T backbones at the original image resolution (640 × 480), using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 4 }$ , batch size 64, for 100 epochs. For these backbones, the projector is a 3-layer vision transformer, following the predictor design of I-JEPA [2].

DSEC (RGB–Event). DSEC is a multimodal objectdetection benchmark for autonomous driving, providing RGB images paired with event streams. As in EventScape, events are discretized into a voxel grid, here using 5 temporal bins, yielding a 5-channel input at the original image resolution (640 × 480); this changes, e.g., the patch embedding of the ViT backbone to accept 5-channel inputs. Pretraining otherwise follows the same protocol as KAIST: ViT-S/16, Swin-T, and ConvNeXtV2-T backbones, learning rate $1 \times 1 0 ^ { - 4 }$ , batch size 64, for 100 epochs.

GeoNRW (RGB–DEM). GeoNRW is a multimodal remote sensing dataset composed of aerial RGB orthophotos and digital elevation models (DEM). Since the origina scenes are very large, we generate training samples by extracting tiles of size 250 × 250 pixels from the raw images. DEM maps are normalized to the [0, 1] range and resized to match the RGB spatial resolution. Both modalities are resized using random resized cropping to 224 × 224 pixels during training. Augmentations include random cropping and horizontal flipping for both modalities, while color jittering is applied only to RGB images. For this fine-tuning protocol we adopt ResNet50 encoders [18]. Pretraining is performed using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 5 }$ , batch size 96, for 50 epochs. For the frozenbackbone probing protocol, we instead pretrain a ViT-S/16 backbone for 50 epochs, using the same preprocessing and image resolution.

SemanticKITTI (RGB–LiDAR). SemanticKITTI provides outdoor driving sequences containing LiDAR point clouds and synchronized camera images. Since camera images are available only for the front field of view, the Li-DAR point clouds are spatially cropped to match the camera field of view before being processed by the model. The cropped point clouds are then converted into dense representations used by the LiDAR encoder. RGB images are resized using random resized cropping to $2 2 4 \times 2 2 4$ pixels and augmented with horizontal flipping and color jittering. LiDAR representations undergo geometric augmentations consistent with the projection operation. For this fine-tuning protocol we adopt a ResNet34 encoder for the camera branch and a VoxelNet-style encoder for the Li-DAR branch. Pretraining is performed using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 6 }$ , batch size 32, for 200 epochs. For the frozen-backbone probing protocol, we instead pretrain a MinkUNet [11] encoder for the LiDAR branch and a ViT backbone for the camera branch, using the AdamW optimizer with learning rate $1 \times 1 0 ^ { - 5 }$ , batch size 32, for 200 epochs.

BigEarthNet (SAR–Optical). For the BigEarthNet benchmark, we follow the multimodal setting introduced in DeCUR and use Sentinel-1 SAR and Sentinel-2 optical imagery as the two modalities. Sentinel-1 images contain two channels (VV and VH polarization), while Sentinel-2 images contain 13 multispectral bands. Following DeCUR, all images are resampled to a common spatial resolution of 10 m per pixel. Optical channels are normalized using band-wise statistics, while SAR images are clipped to remove extreme outliers and subsequently normalized. We use a ResNet50 backbone for both modalities. Optimization follows the same protocol adopted for GeoNRW, using AdamW with learning rate $1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 4 }$ , cosine learning-rate decay, and batch size 96 for 100 epochs.

Hyperparameters Across all datasets, the encoders are followed by projection heads that map the features into a $K _ { p }$ -dimensional embedding space used during selfsupervised training. Following common practice in redundancy-reduction-based self-supervised learning methods, we use a high-dimensional embedding space with $K _ { p } = 8 1 9 2$ . The embedding space is divided into shared and modality-specific subspaces, with $K _ { s }$ shared dimensions and $K _ { u }$ modality-specific dimensions $( K _ { p } = K _ { s }$ + $K _ { u } )$ . The projection heads are used only during pretraining and are discarded when transferring the encoders to downstream tasks.

## A.2. Downstream Evaluation Protocols

To evaluate the quality of the learned representations, the pretrained encoders are transferred to downstream tasks under the two protocols introduced in Section 5.1: fine-tuning and frozen-backbone probing.

Frozen-Backbone Probing. For semantic segmentation (SUN-RGBD, GeoNRW), we evaluate the frozen backbones with three protocols: k-nearest-neighbor classification with k = 10 neighbors retrieved among the patch tokens of the training set; linear probing, training a linear classifier on the patch tokens for 20 epochs; and a lightweight FCN head, trained for 30 epochs on the feature maps of the last three ViT stages. For object detection (KAIST, DSEC), we attach a FCOS [30] head to the last three feature maps of the frozen backbone, combined through a feature pyramid network, and train it for 20 epochs. For SemanticKITTI, we train the downstream model on the full 360° scan, fusing the frozen MinkUNet and ViT backbones by projecting each 3D point onto the image plane and concatenating the corresponding RGB feature to the point feature, as in Point-Fusion [36]. Points falling outside the camera field of view receive a zero feature vector. The fused features are then processed by the MinkUNet decoder, which serves as the segmentation head and is trained for 20 epochs while both backbones remain frozen. We report mIoU both within the camera field of view and over the entire scan.

Low-Label Setting. For the low-label fine-tuning experiments (SUN-RGBD, 10% labels; GeoNRW, 1% labels), the labelled subset of the training set is drawn once via simple random sampling, using a fixed random seed shared across all compared methods, so that every method is fine-tuned on the exact same subset; validation and test sets are left unchanged. The frozen-backbone probing protocol, in contrast, always uses the full training set.

SUN-RGBD (Fine-Tuning). For SUN-RGBD we adopt the CMX [38] architecture for RGB-D semantic segmentation. The pretrained RGB and depth encoders are used to initialize the corresponding branches of the model. Training follows the same configuration as the original CMX paper and runs for 200 epochs.

EventScape (Fine-Tuning). For EventScape we also use the CMX architecture to fuse RGB frames and event representations. The pretrained encoders are used to initialize the modality-specific branches and the model is trained for 50 epochs using the same training configuration as the original CMX implementation.

GeoNRW (Fine-Tuning). For GeoNRW we evaluate the learned representations using a lightweight fully convolutional segmentation model [24]. The feature maps extracted from the two modality-specific encoders are concatenated and passed to a segmentation head composed of upsampling layers followed by a linear classification layer. Specifically, we use feature maps from the last three stages of the ResNet backbone and apply three different upsampling factors before combining them to produce the final prediction map. The segmentation model is trained for 30 epochs.

SemanticKITTI (Fine-Tuning). For SemanticKITTI, we adopt the PMF [40] architecture for LiDAR-camera semantic segmentation. We follow exactly the same training protocol as the original PMF implementation and train the model for 50 epochs. To ensure compatibility with the architecture, we load the pretrained weights only for the camera branch of the model while keeping the LiDAR backbone unchanged, as it requires RGB features to be fused in an early stage of the LiDAR backbone, which is not compatible with our pretraining scheme.

BigEarthNet. For BigEarthNet, we evaluate the learned representations on multilabel land-use classification. Following a standard linear probing protocol, the pretrained encoder is kept frozen, and a linear classification head is trained on top of the extracted features. The classification head consists of a single fully-connected layer mapping the backbone features to the target classes. Performance is evaluated using mean Average Precision (mAP).

## A.3. Baseline Methods

We compare the proposed method with contrastive, redundancy-reduction-based, and dense single-modal selfsupervised learning baselines, including SimCLR [10], VI-CReg [6], Barlow Twins [37], CLIP [26], DeCUR [33], DINO [9], iBOT [39], I-JEPA [2], and LeJEPA [4]. All baselines are trained using the same backbones, data augmentations, and optimization settings whenever applicable.

SimCLR, VICReg, and Barlow Twins are originally designed for single-modal representation learning. In our multimodal setting, we adapt these methods by treating the two modalities as the two views used during training. In practice, to train all three methods, we use one augmented view per modality and apply the loss of SimCLR, VICReg, or Barlow Twins to the embeddings extracted from the two modalities instead of two different augmented views of the same modality, allowing the methods to learn cross-modal representations.

For CLIP, we use the implementation provided in the De-CUR codebase to ensure consistent training settings across baselines.

For DeCUR, we use the official implementation released by the authors, adapting only the dataset loading and preprocessing pipelines to match the datasets used in our experiments.

For DINO, iBOT, I-JEPA, and LeJEPA, we use the official code released by the respective authors, adopting the same ViT backbone as DenseTwins but each method’s own official hyperparameters for augmentations and image sizes, except for the number of training epochs and batch sizes, which we set to match our own pretraining schedule. Each method is trained independently on each modality, producing two single-modal backbones that are combined via late fusion for downstream evaluation, as described in Section 5.1.

## B. Algorithm Pseudocode

In this section we summarize the training procedure of DenseTwins. Algorithm 4 describes the overall training loop, while Algorithms 1 and 2 define the two loss functions used to optimize the model.

Given a pair of observations from two modalities, two independently augmented views are generated for each modality. The augmented samples are processed by two modality-specific encoders followed by projection heads that produce both global embeddings and dense tokenlevel representations. The training objective combines four components: intra-modal global losses, intra-modal dense losses, cross-modal global losses and cross-modal dense losses.

Algorithm 1 Barlow Twins Loss   
Require: Cross-correlation matrix C, coefficient λ   
1: function BARLOWTWINSLOSS(C, λ)   
2: l<sub>on-diag</sub> = P<sub>p</sub>(1 − C<sub>pp</sub>)<sup>2</sup>   
3: l<sub>off-diag</sub> = P <sub≯</sub> (C<sub>pq</sub>)<sup>2</sup>   
return l<sub>on-diag</sub> + λ · l<sub>off-diag</sub>   
5: end function

## C. Analysis of the Learned Decomposition

While downstream performance provides the primary evaluation of pretrained representations, it does not directly verify whether the proposed decomposition successfully separates shared and modality-specific information. To better understand the behaviour of DenseTwins, we analyze the learned embedding space through three complementary perspectives: correlation structure, cross-modal retrieval, and feature visualization.

Algorithm 2 Decorrelation Loss   
Require: Cross-correlation matrix C, coefficient λ   
1: function DECORRELATIONLOSS(C, λ)   
2: $\begin{array} { r } { l _ { \mathrm { o n - d i a g } } = \sum _ { p } ( C _ { p p } ) ^ { 2 } } \end{array}$   
3: $\begin{array} { r } { l _ { \mathrm { o f f - d i a g } } = \sum _ { p \neq q } ( C _ { p q } ) ^ { 2 } } \end{array}$   
4: return $l _ { \mathrm { o n - d i a g } } \mp \lambda \cdot l _ { \mathrm { o f f } } .$ -diag   
5: end function   
Algorithm 3 GatherDense   
Require: Feature maps $\mathbf { Z } _ { a } \in \mathbb { R } ^ { B \times K \times T _ { a } } , \mathbf { Z } _ { b } \in \mathbb { R } ^ { B \times K \times T _ { b } } ,$   
Similarity $\Delta \in \mathbb { R } ^ { \mathbf { \bar { B } } \times T _ { a } \times T _ { b } }$   
1: function $\mathrm { G A T H E R D E N S E } ( \mathbf { Z } _ { a } , \mathbf { Z } _ { b } , \Delta )$   
2: I = arg max<sub>j</sub> $\Delta _ { i j }$ \triangleright find best spatial   
correspondence   
3: $\mathbf { Z } _ { b }  \mathbf { G } \mathrm { A T H E R } ( \mathbf { Z } _ { b } , I )$ \triangleright reorder spatial features   
4: $\mathbf { Z } _ { a } \gets \mathrm { R E S H A P E } ( \mathbf { Z } _ { a } , ( B \cdot T _ { a } , K ) )$   
5: $\mathbf { Z } _ { b }  \mathrm { R E S H A P E } ( \mathbf { Z } _ { b } , ( B \cdot T _ { a } , K ) )$   
6: return $\mathbf { Z } _ { a } , \mathbf { Z } _ { b }$   
7: end function

## C.1. Correlation Structure

DenseTwins explicitly encourages shared dimensions to capture information that is consistent across modalities while suppressing correlations between modality-specific components. To verify that this behaviour emerges in practice, we analyze the global cross-correlation matrix G computed between modalities after pretraining.

In particular, we compare the diagonal values of the shared correlation block $G ^ { s s }$ and the modality-specific block $G ^ { u u }$ We expect dimensions belonging to the shared subspace to exhibit strong cross-modal correlations, whereas dimensions belonging to the modality-specific subspace should remain largely decorrelated.

The observed behaviour is consistent with the objective. The average diagonal value of $G ^ { s s }$ is 0.382, while the average diagonal value of $G ^ { u u }$ is only 0.038, indicating that most cross-modal information is concentrated in the shared subspace and that modality-specific dimensions remain largely independent across modalities.

## C.2. Cross-Modal Retrieval

We further evaluate whether the shared subspace captures semantically meaningful cross-modal information through a patch-level retrieval experiment.

Given a query RGB patch embedding, we retrieve the most similar Depth patch embedding using either the shared or the modality-specific subspace. A retrieval is considered correct if the retrieved patch corresponds to the same scene and spatial location.

Algorithm 4 DenseTwins Training Procedure   
Require: Modality encoders $\mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } ,$ , projectors $\mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ,$   
dataset $\mathcal { D } ,$ batch size N   
1: for $( \mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } )$ in D do   
\triangleright Feature extraction & Spatial correspondences   
2: $( \mathbf { X } _ { 1 } ^ { \prime } , \mathbf { X } _ { 1 } ^ { \prime \prime } ) = \mathrm { A U G M E N T } ( \mathbf { X } _ { 1 } )$   
3: $( \mathbf { X } _ { 2 } ^ { \prime } , \mathbf { X } _ { 2 } ^ { \prime \prime } ) = \mathrm { A U G M E N T } \big ( \mathbf { X } _ { 2 } \big )$   
4: $\mathbf { F } _ { 1 } ^ { \prime } , \mathbf { F } _ { 1 } ^ { \prime \prime } = \mathscr { F } _ { 1 } ( \mathbf { X } _ { 1 } ^ { \prime } ) , \mathscr { F } _ { 1 } ( \mathbf { X } _ { 1 } ^ { \prime \prime } )$   
5: $\mathbf { F } _ { 2 } ^ { \prime } , \mathbf { F } _ { 2 } ^ { \prime \prime } = \mathscr { F } _ { 2 } ( \mathbf { X } _ { 2 } ^ { \prime } ) , \mathscr { F } _ { 2 } ( \mathbf { X } _ { 2 } ^ { \prime \prime } )$   
6: $( \bar { { \bf y } _ { 1 } ^ { \prime } } , \bar { { \bf z } _ { 1 } ^ { \prime } } ) , ( { \bf y } _ { 1 } ^ { \prime \prime } , \bar { { \bf z } _ { 1 } ^ { \prime \prime } } ) = \mathcal { P } _ { 1 } ( \bar { { \bf F } } _ { 1 } ^ { \prime } ) , \mathcal { P } _ { 1 } ( \bar { { \bf F } } _ { 1 } ^ { \prime \prime } )$   
7: $( \mathbf { y } _ { 2 } ^ { \prime } , \mathbf { Z } _ { 2 } ^ { \prime } ) , ( \mathbf { y } _ { 2 } ^ { \prime \prime } , \mathbf { Z } _ { 2 } ^ { \prime \prime } ) = \mathcal { P } _ { 2 } ( \mathbf { F } _ { 2 } ^ { \prime } ) , \mathcal { P } _ { 2 } ( \mathbf { F } _ { 2 } ^ { \prime \prime } )$   
8: $\Delta _ { 1 } = \mathrm { C O S I N E S I M I L A R I T Y } ( \mathbf { F } _ { 1 } ^ { \prime } , \mathbf { F } _ { 1 } ^ { \prime \prime } )$   
9: $\Delta _ { 2 } = \mathrm { C O S I N E S I M I L A R I T Y } \left( \mathbf { F } _ { 2 } ^ { \prime } , \mathbf { F } _ { 2 } ^ { \prime \prime } \right)$   
10: $\Delta _ { 1 2 } = \mathrm { C o s I N E S I M I L A R I T Y } ( \bar { \bf Z } _ { 1 } ^ { \prime } [ : \bar { K } _ { c } ] , { \bf Z } _ { 2 } ^ { \prime } [ : K _ { c } ] )$   
\triangleright Intra-modal global loss   
11: $\begin{array} { r } { C _ { 1 } ^ { g } = \frac { 1 } { N } { \mathbf B } { \mathbf N } _ { g } ( \mathbf { y } _ { 1 } ^ { \prime } ) { \mathbf \Sigma } _ { - } ^ { \top } { \mathbf B } { \mathbf N } _ { g } ( \mathbf { y } _ { 1 } ^ { \prime \prime } ) } \end{array}$   
12: $\begin{array} { r } { C _ { 2 } ^ { g } = \frac { 1 } { N } { \mathbf B } { \mathbf N } _ { g } ( \mathbf y _ { 2 } ^ { \prime } ) ^ { \top } { \mathbf B } { \mathbf N } _ { g } ( \mathbf y _ { 2 } ^ { \prime \prime } ) } \end{array}$   
13: $\mathcal { L } _ { g } ^ { ( 1 ) } = \mathbf { B A R L O W T W I N S L O s s } ( C _ { 1 } ^ { g } , \lambda )$   
14: $\mathcal { L } _ { g } ^ { ( 2 ) } = \mathtt { B A R L O W T W I N S L O s s } ( C _ { 2 } ^ { g } , \lambda )$   
\triangleright Intra-modal dense loss   
15: $( \mathbf { Z } _ { 1 a } , \mathbf { Z } _ { 1 b } ) = \mathbf { G a T H E R D E N S E } ( \mathbf { Z } _ { 1 } ^ { \prime } , \mathbf { Z } _ { 1 } ^ { \prime \prime } , \boldsymbol { \Delta } _ { 1 } )$   
16: $( \mathbf { Z } _ { 2 a } , \mathbf { Z } _ { 2 b } ) = \mathbf { G } \mathbf { A } \mathrm { T H E R } \mathbf { D E N S E } ( \mathbf { Z } _ { 2 } ^ { \prime } , \mathbf { Z } _ { 2 } ^ { \prime \prime } , \boldsymbol { \Delta } _ { 2 } )$   
17: $\begin{array} { r } { C _ { 1 } ^ { d } = \frac { 1 } { T _ { 1 } N } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 1 a } ) ^ { \top } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 1 b } ) } \end{array}$   
18: $\begin{array} { r } { C _ { 2 } ^ { d } = \frac { \hat { 1 } } { T _ { 2 } N } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 2 a } ) ^ { \top } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 2 b } ) } \end{array}$   
19: $\mathcal { L } _ { d } ^ { ( 1 ) }$ = BARLOWTWINSLOSS(C<sup>d</sup>, λ)   
20: $\mathcal { L } _ { d } ^ { ( 2 ) } = \mathbf { B A R L O W T W I N S L O s s } ( C _ { 2 } ^ { d } , \lambda )$   
21: $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 1 ) } = ( 1 - \alpha ) \mathcal { L } _ { g } ^ { ( 1 ) } + \alpha \mathcal { L } _ { d } ^ { ( 1 ) }$   
22: $\mathcal { L } _ { \mathrm { i n t r a } } ^ { ( 2 ) } = ( 1 - \alpha ) \mathcal { L } _ { g } ^ { ( 2 ) } + \alpha \mathcal { L } _ { d } ^ { ( 2 ) }$   
\triangleright Cross-modal losses   
23: $\begin{array} { r } { G = \frac { 1 } { N } \mathbf { B } \mathbf { N } _ { g } ( \mathbf { y } _ { 1 } ^ { \prime } ) ^ { \top } \mathbf { B } \mathbf { N } _ { g } ( \mathbf { y } _ { 2 } ^ { \prime } ) } \end{array}$   
24: $G ^ { s s } , \ddot { G } ^ { u u } = G [ : K _ { c } , : \check { K } _ { c } ] , \dot { G } [ K _ { c } : , K _ { c } : ]$   
25: $L _ { s s } ^ { g } = \mathbf { B A R L O W T W I N S L O S S } ( G ^ { s s } , \lambda )$   
26: ${ \cal L } _ { u u } ^ { g } = \mathrm { D E C O R R E L A T I O N L O S S } ( G ^ { u u } , \lambda )$   
27: $( \mathbf { Z } _ { 1 c } , \mathbf { Z } _ { 2 c } ) = \mathbf { G } \mathbf { A } \mathbf { T } \mathbf { H } \mathbf { E } \mathbf { R } \mathbf { D } \mathbf { E } \mathbf { N } \mathbf { S } \mathbf { E } ( \mathbf { Z } _ { 1 } ^ { \prime } , \mathbf { Z } _ { 2 } ^ { \prime } , \boldsymbol { \Delta } _ { 1 2 } )$   
28: $\begin{array} { r } { D = \frac { 1 } { T _ { 1 } N } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 1 c } ) ^ { \top } \mathbf { B } \mathbf { N } _ { d } ( \mathbf { Z } _ { 2 c } ) } \end{array}$   
29: $D ^ { s s } , \dot { D } ^ { \dot { u } \dot { u } } = D [ : K _ { c } , : K _ { c } ] , D [ K _ { c } : , K _ { c } : ]$   
30: $\mathcal { L } _ { s s } ^ { d } = \mathbf { B A R L O W T W I N S L O s s } ( D ^ { s s } , \lambda )$   
31: $\mathcal { L } _ { u u } ^ { d } = \mathrm { D E C O R R E L A T I O N L O s s } ( D ^ { u u } , \lambda )$   
32: $\mathcal { L } _ { \mathrm { c r o s s } } = ( 1 - \alpha ) ( \mathcal { L } _ { s s } ^ { g } + \mathcal { L } _ { u u } ^ { g } ) + \alpha ( \mathcal { L } _ { s s } ^ { d } + \mathcal { L } _ { u u } ^ { d } )$   
\triangleright Total loss   
33: $\mathcal { L } = ( \mathcal { L } _ { 1 } + \mathcal { L } _ { 2 } + \mathcal { L } _ { \mathrm { c r o s s } } ) / 3$   
34: BACKPROPAGATION(L)   
35: end for

Results are reported in Table 9. Retrieval using the shared subspace substantially outperforms retrieval using the modality-specific subspace, achieving Recall@1 of 0.62 and Recall@5 of 0.85. In contrast, retrieval based on the modality-specific dimensions performs close to chance level. These results provide direct evidence that semantic correspondences between modalities are encoded primarily in the shared subspace.

Table 9. Cross-modal patch retrieval using different subspaces.
<table><tr><td>Subspace</td><td>Recall@1</td><td>Recall@5</td></tr><tr><td>Shared</td><td>0.62</td><td>0.85</td></tr><tr><td>Specific</td><td>0.06</td><td>0.13</td></tr></table>

## C.3. Feature Visualization

Finally, Figure 5 presents t-SNE projections of the global embeddings extracted from the shared and modalityspecific subspaces.

Embeddings from the shared subspace exhibit substantially stronger cross-modal mixing, indicating that samples from different modalities are projected to a common semantic space. In contrast, embeddings from the modalityspecific subspace remain clearly separated according to their sensing modality, suggesting that modality-dependent information is preserved in these dimensions.

Together, the correlation analysis, retrieval experiment, and feature visualization consistently support the intended behaviour of DenseTwins and provide evidence that the learned embedding space successfully disentangles shared and modality-specific information. Combined with the consistent downstream improvements reported in Section 5, these results suggest that dense modality decoupling is not only achieved in practice, but also contributes to learning more effective multimodal representations for dense prediction tasks.

![](images/f2f15a4008ba5216ad2dd190f2a6c8f6d5fe61374d76af725419ac0a3b199e29.jpg)  
Figure 5. t-SNE visualization of global embeddings extracted from the shared and modality-specific subspaces. The shared subspace exhibits stronger cross-modal alignment, whereas the modality-specific subspace remains separated according to modality.

## D. Additional Analysis

## D.1. Computational Overhead

We analyze the computational and memory overhead introduced by the projection heads and the dense matching mechanism during pretraining, to complement the ablation studies presented in Section 5 with a direct measurement of their training cost.

Table 10. Per-step time breakdown of a DenseTwins training step (single A100 GPU) at two per-GPU batch sizes.
<table><tr><td>Stage</td><td colspan="2">ms % forward % step</td></tr><tr><td>Batch size 8/GPU</td><td></td><td></td></tr><tr><td>Backbones forward</td><td>69.9</td><td>56.8 17.7</td></tr><tr><td>Projection heads</td><td>9.3 7.6</td><td>2.4</td></tr><tr><td>Dense Correspondences</td><td>3.6</td><td>2.9 0.9</td></tr><tr><td>BT correlation and loss</td><td>41.0</td><td>33.4 10.4</td></tr><tr><td>Total forward</td><td>123.0</td><td>100.0 31.2</td></tr><tr><td>Backward step Total</td><td>270.7 394.0</td><td>68.8 一 100.0</td></tr><tr><td>Batch size 24/GPU</td><td></td><td></td></tr><tr><td>Backbones forward</td><td>200.4</td><td>65.3</td></tr><tr><td>Projection heads</td><td>21.9</td><td>21.2 2.3</td></tr><tr><td>Dense Correspondences</td><td>8.9</td><td>7.1 2.9 0.9</td></tr><tr><td>BT correlation and loss</td><td></td><td>24.9</td></tr><tr><td>Total forward</td><td>76.5</td><td>8.1 32.5</td></tr><tr><td></td><td>307.0</td><td>100.0</td></tr><tr><td>Backward step</td><td>639.0</td><td>67.6 一</td></tr><tr><td>Total</td><td>946.0</td><td>一 100.0</td></tr></table>

We profile a full training step of DenseTwins on a single NVIDIA A100 GPU, decomposing the forward and backward computation into its main stages for two per-GPU batch sizes (8 and 24). Table 10 reports, for each stage, its wall-clock cost in milliseconds together with its share of the total forward pass and of the full training step (forward, backward and optimizer update).

The backbones’ forward pass, executed four times per step (one pass per augmented view and per modality), together with the backward step, dominates the training step at both batch sizes. In contrast, the components introduced by DenseTwins on top of a standard multimodal pretraining pipeline add only a modest overhead to the forward pass: the projection heads account for 7.1–7.6% of the forward pass (2.3–2.4% of the full step, not counting the backward pass through the projectors themselves), while the dense correspondences step, which establishes the spatial matching used by the dense objectives, accounts for only 2.9% of the forward pass (0.9% of the full step).

Considering the full training step, including the backward pass through the projection heads, the projection heads increase training time by +25% and peak GPU memory by +26% with respect to a version of the pipeline without them. Table 11 quantifies the practical impact of this overhead on a representative large-scale pretraining run on DSEC using 8×A100 GPUs.

We stress that this overhead is incurred only during pretraining: the projection heads are always discarded before transferring the encoders to downstream tasks (Section A), so downstream fine-tuning/probing cost and inference cost of the encoders are unaffected by it. Removing the projection heads, as in the w/o projection heads configuration of Table 11, means that shared/specific decoupling and dense correspondence matching are instead computed directly on the backbone feature space rather than on a dedicated projected embedding. As shown in the ablation studies in Section 5 (Tables 7 and 8), using dedicated projection heads is nonetheless beneficial for the performance of DenseTwins, motivating their use despite the added training-time cost.

Table 11. Effect of the projection heads on DSEC pretraining using 8×A100 GPUs. At a fixed memory budget, removing them allows a larger per-GPU batch size and reduces training time.
<table><tr><td>Configuration</td><td>Batch size/GPU</td><td>Training time</td></tr><tr><td>w/o projection heads</td><td>28</td><td>5.1 h</td></tr><tr><td>w/ projection heads</td><td>22</td><td>6.4h</td></tr></table>

## D.2. Single Modal Results

DenseTwins is a multimodal method: both modalities are required during pretraining, as the shared/specific decomposition is learned jointly across the modality pair. However, DenseTwins learns two separate modality-specific encoders rather than a single fused backbone, which leaves room for flexible deployment solutions, including deploying a single-modal model built from one of the two pretrained encoders when only one modality is available at test time. To assess how competitive the resulting single-modal representations are, we take the RGB and Depth encoders jointly pretrained by DenseTwins on SUN-RGBD and evaluate each of them individually, and compare them against dedicated single-modal SSL baselines (DINO, iBOT, I-JEPA, LeJEPA) trained separately on each modality with the same backbone. We also report the multimodal setting, where the two modality-specific encoders are combined via late fusion (feature concatenation), for both DenseTwins and the baselines. All representations are evaluated with linear probing (LP) and a lightweight FCN head on frozen features, following the protocol of Section 5.

Table 12 reports the results. When only a single modality is available, the DenseTwins encoder pretrained for that modality remains competitive with dedicated single-modal baselines: it is the best-performing method on Depth LP, and consistently outperforms DINO, iBOT and LeJEPA on RGB (both LP and FCN), although I-JEPA attains higher RGB scores and DINO attains a slightly higher Depth FCN score. Notably, in the multimodal (Concat) setting, DenseTwins outperforms all baselines, including I-JEPA, on both LP and FCN, despite I-JEPA’s RGB-only encoder being stronger than the corresponding DenseTwins encoder. This suggests that the DenseTwins representation captures modality-specific information more effectively than singlemodal baselines once combined across modalities, even where its individual single-modal encoders are not the strongest in isolation.

Table 12. Linear probing (LP) and FCN performance (mIoU) on SUN-RGBD when only RGB or only Depth is available at test time, compared to the multimodal (Concat) setting obtained via late fusion.
<table><tr><td></td><td>RGB</td><td>Depth</td><td>Concat</td><td></td></tr><tr><td>Method</td><td>LP FCN</td><td>LP FCN</td><td>LP</td><td>FCN</td></tr><tr><td>DenseTwins</td><td>17.2 26.7</td><td>27.8</td><td>32.5</td><td>33.7 40.6</td></tr><tr><td>I-JEPA</td><td>19.6 27.5</td><td>27.2 32.0</td><td>31.1</td><td>35.2</td></tr><tr><td>DINO</td><td>15.0 22.8</td><td>26.4</td><td>33.9 31.0</td><td>37.6</td></tr><tr><td>iBOT</td><td>12.7 19.7</td><td>24.8</td><td>31.6 30.8</td><td>36.3</td></tr><tr><td>LeJEPA</td><td>9.5 19.1</td><td>17.6</td><td>30.0 23.9</td><td>32.7</td></tr></table>

## D.3. Dense Matching Robustness

The retrieval experiment in Section C (Table 9) measures cross-modal correspondence quality under perfectly aligned modalities. To assess robustness to imperfect spatial alignment between modalities, we repeat the same retrieval protocol after applying synthetic misalignment to one modality: a random translation of one patch (16 pixels) or two patches (32 pixels), before performing retrieval.

Table 13 reports the results. Recall@1 and Recall@5 remain close to the aligned setting under both shifts (0.60– 0.61 and 0.82–0.84, respectively, vs. 0.62/0.85 without misalignment), showing that the dense correspondence mechanism degrades only marginally under moderate spatial misalignment between modalities.

Table 13. Cross-modal patch retrieval recall on SUN-RGBD under synthetic misalignment (translation applied to one modality) using the shared subspace.
<table><tr><td>Misalignment</td><td>Recall@1</td><td>Recall@5</td></tr><tr><td>None (aligned)</td><td>0.62</td><td>0.85</td></tr><tr><td>Shift 16px (1 patch)</td><td>0.60</td><td>0.84</td></tr><tr><td>Shift 32px (2 patches)</td><td>0.61</td><td>0.82</td></tr></table>

This robustness follows directly from a design choice of DenseTwins: cross-modal correspondences are established through a nearest-neighbour search in feature space over all spatial positions, rather than by assuming a fixed positional correspondence between the two modalities. It is precisely this relaxation, matching by feature similarity rather than by position, that makes the mechanism robust to the kind of spatial misalignment a strictly positional matching scheme would be sensitive to.

## E. Qualitative Downstream Results

We present qualitative results of the downstream semantic segmentation models on two representative multimodal datasets: SUN-RGBD and EventScape. For each example, we visualize the input modalities, the ground-truth segmentation, and the predictions obtained using backbones pretrained with DeCUR and with the proposed DenseTwins framework.

SUN-RGBD. Figure 6 shows qualitative segmentation results on indoor RGB-D scenes. While both methods correctly segment the main structures of the scene, DenseTwins generally produces cleaner object boundaries and more consistent predictions across objects with similar appearance. In particular, DenseTwins better preserves thin structures and object contours (e.g., chairs and table legs) and reduces spurious predictions around object boundaries. These results suggest that learning dense multimodal representations improves the spatial consistency of the learned features.

EventScape. Figure 7 shows qualitative results on driving scenes combining RGB images and event representations. For visualization purposes, the event modality is displayed using a frame-based representation where positive and negative events are accumulated into separate channels. However, the model is trained using a voxel-based event representation as input, as described in Section A. In these scenarios, DenseTwins better captures distant objects and thin structures such as poles and fences, which are often missed or fragmented by the baseline, improving the recognition of small objects in the scene. These results suggest that dense multimodal alignment enables the model to better exploit complementary cues from RGB and event data, particularly for dynamic and fine-grained structures.

![](images/b070c7fed51ebad2fe5c60f0bc25d7bbdca8e106bc9a90224d8da6621f6ed050.jpg)  
Figure 6. Qualitative Results on SUN-RGBD. From left to right we show the RGB image, the depth modality encoded as HHA, the ground-truth semantic segmentation, and the predictions obtained using DeCUR and DenseTwins pretraining. While both methods cor rectly segment the main scene structures, DenseTwins produces sharper object boundaries and more consistent predictions.

![](images/0cd4c0e5026830fdc5a45696bc1ea67820c7fdaad2e779417ef54befa75e771f.jpg)  
Figure 7. Qualitative Results on Eventscape. From left to right, we show the RGB frame, the event representation, the ground-truth segmentation, and the predictions obtained using DeCUR and DenseTwins. Highlighted regions show typical failure cases of DeCUR (red boxes), such as missed distant objects or fragmented thin structures, while DenseTwins provides more complete predictions and improved detection of small or distant regions (green boxes).