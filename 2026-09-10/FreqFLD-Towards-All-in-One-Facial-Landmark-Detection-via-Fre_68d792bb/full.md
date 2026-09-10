# FreqFLD: Towards All-in-One Facial Landmark Detection via Frequency Modulation

Shun Ren<sup>a</sup>, Kaijie Jin<sup>a</sup>, Shengkai Hu<sup>b,∗</sup>, Beihang Song<sup>c,∗</sup>, Hang Sun<sup>a</sup>, Wenwen Min<sup>d</sup>, Youfa Liu<sup>e</sup>, Jun Wan<sup>b,f</sup>

<sup>a</sup>College of Computer and Information Technology, China Three Gorges University, Yichang 443002, China

<sup>b</sup>School of Information Engineering, Zhongnan University of Economics and Law, Wuhan 430073, China

<sup>c</sup>National Institute of Natural Hazards, Ministry of Emergency Management of China, Beijing 100085, China

<sup>d</sup>School of Information Science and Engineering, Yunnan University, Kunming, Yunnan 650091, China

<sup>e</sup>School of Computer Science, Wuhan University, Wuhan 430072, China

<sup>f</sup>School of Computer Science and Engineering, Nanyang Technological University, Singapore 639798, Singapore

## Abstract

Recent progress in deep learning has significantly advanced facial landmark detection. However, most existing methods process features in a spatial-domain manner under a dataset-specific training paradigm, which overlooks the fact that facial landmark detection is inherently geometry-driven and sensitive to frequency variations, thereby limiting cross-dataset generalization under complex scenarios and hindering the development of a facial landmark detection model. To address this issue, we propose FreqFLD, a frequency-modulated framework towards All-in-One facial landmark detection. Specifically, FreqFLD introduces a Frequency Modulation Module (FreqMoM) to explicitly induce the frequency prior by decoupling and modulating low- and high-frequency components, which is then injected into subsequent feature modeling to enable balanced modeling of global facial structure and local landmark details. Furthermore,

FreqFLD employs a Frequency-Modulated Mixture-of-Experts (FreqMoE), with expert selection adaptively conditioned on frequency-modulated priors, enabling flexible modeling of heterogeneous facial landmark patterns under diverse and challenging scenarios. To regularize frequency-consistent modeling under the All-in-One paradigm, we further introduce a Frequency-Consistent Routing (FreqCR) loss, which constrains the routing and assignment of frequency-aware experts to promote balanced expert utilization across diverse facial scenarios, thereby enabling stable expert specialization and achieving robust facial landmark detection. Extensive experiments demonstrate that the proposed FreqFLD achieves comparable performance on popular datasets. The code is available at: https://github.com/jkj1059657014/FreqFLD.

Keywords: Facial Landmark Detection, Heatmap regression, All-in-One, Frequency Learning, Multi-Dataset Training

## 1. INTRODUCTION

Facial landmark detection (FLD), also known as face alignment, is a fundamental task in computer vision that aims to localize semantically consistent keypoints (e.g., eyes, nose tip, mouth corners, etc.) for facial geometry and structure. It serves as a critical prerequisite for a wide range of downstream applications, including face recognition [1, 2], expression recognition [3, 4], 3D face reconstruction [5, 6] and virtual avatar generation [7, 8].

With the advancement of deep learning, most existing FLD methods follow a dataset-specific training paradigm [9, 10, 11, 12], where models are independently optimized for individual benchmarks. While such a paradigm has achieved impressive performance on single datasets, it often sufers from limited generalization across datasets and incurs considerable computational and maintenance costs due to repeated training. A comparison across various FLD benchmarks indicates that, despite variations in the number of annotated landmarks (68 for 300W, 19 for AFLW, 29 for COFW, and 98 for WFLW), all datasets describe the same fundamental facial geometry. In particular, overlapping semantic landmarks across diferent datasets span a shared geometric subspace, which is repeatedly learned by dataset-specific models. This structural redundancy suggests that a unified representation can be learned across heterogeneous datasets, thereby motivating a unified architecture for FLD. Moreover, to improve cross-domain generalization, more recently, inspired by advances in low-level image restoration [13, 14, 15, 16, 17], and fine-grained visual understanding [18, 19, 20], the All-in-One paradigm has been explored to handle multiple degradations within a single framework. However, extending these paradigms to FLD inevitably exacerbates feature conflicts induced by large variations in facial pose, expression, and occlusion.

![](images/6159667be7749d904c2aca6d9204b4039478e31a3ec8748d69214f11b86c0fc3.jpg)  
Figure 1: The proposed FreqFLD facilitates unified learning across multiple facial landmark detection datasets with heterogeneous annotation schemes and demonstrates comparable performance.

More recently, mixture-of-experts (MoE) architectures [21, 22, 23, 24] have been explored within the All-in-One paradigm as an efective mechanism to alleviate feature conflicts by employing multiple specialized experts to model heterogeneous data distributions through adaptive routing. However, extending existing MoE frameworks to FLD presents non-trivial challenges. On the one hand, traditional individual experts are primarily based on spatial-domain feature modeling, which limits their ability to capture the intrinsic geometric structure of faces. On the other hand, unconstrained expert routing leads to unstable specialization, resulting in expert imbalance or inconsistent assignment of facial patterns across experts, which further compromises training stability and generalization performance. These limitations are further exacerbated under multi-dataset joint training in the All-in-One paradigm.

Motivated by these observations, we propose FreqFLD, a frequency-modulated All-in-One framework for robust FLD. FreqFLD incorporates a FreqMoM that explicitly decomposes input features into complementary low- and high-frequency components. By separately modeling low-frequency structural information and high-frequency landmark-sensitive details, FreqMoM constructs frequencydisentangled priors. Furthermore, we introduce a FreqMoE, in which expert routing is adaptively conditioned on frequency-modulated priors. By guiding expert selection with frequency-aware structural cues, FreqMoE promotes consistent expert specialization across datasets, enabling diferent experts to capture heterogeneous facial patterns while preserving shared geometric structure, thereby mitigating feature conflicts induced by diverse facial conditions. To further regularize frequency-consistent expert specialization under the All-in-One paradigm, we propose a FreqCR loss. By constraining expert assignment to remain aligned with frequency-aware structural cues, FreqCR stabilizes expert routing dynamics and prevents expert imbalance during training, leading to more stable optimization and improved cross-dataset generalization. By jointly integrating FreqMoM, FreqMoE, and FreqCR, FreqFLD achieves coherent frequency-aware representation learning and stable expert specialization under heterogeneous facial statistics, leading to comparable FLD performance (Fig. 1). The main contributions of this work are summarized as follows:

(1) We propose FreqFLD, a frequency-modulated All-in-One framework for FLD, which alleviates cross-dataset feature conflicts by integrating frequencyaware representation learning with expert-based modeling. The proposed FreqFLD achieves competitive FLD performance compared with dataset-specific training methods.

(2) We introduce the Frequency Modulation Module (FreqMoM) and the Frequency-Modulated Mixture-of-Experts (FreqMoE), two complementary components that respectively enable frequency-disentangled feature modeling and frequency-conditioned expert specialization, facilitating robust facial structure modeling under diverse facial conditions.

(3) We propose a Frequency-Consistent Routing (FreqCR) loss to regularize expert routing, promoting stable and frequency-consistent specialization under the All-in-One paradigm.

## 2. RELATED WORK

FLD can be traced back to the end of the 19th century. Early approaches relied on handcrafted models, such as Active Shape Models (ASM) [25, 26], Constrained Local Models (CLM) [27, 28], and random forest–based methods [29, 30], which exhibit limited robustness under unconstrained conditions. With the advent of deep learning, FLD has shifted toward deep learning-based methods [31, 32, 33, 34], which can be broadly categorized into coordinate regression and heatmap regression paradigms.

## 2.1. Coordinate Regression Methods

Coordinate regression methods formulate facial landmark detection as a direct coordinate prediction problem, where landmark locations are learned by minimizing regression losses between predicted and ground-truth coordinates. This formulation enables seamless integration into end-to-end learning frameworks and avoids the intermediate heatmap representation. Early coordinate regression approaches [35] primarily improve robustness through data-driven modeling strategies that leverage dataset-level variations for generalization. In particular, modeling inter- and intra-dataset variations has been shown to facilitate crossdataset face alignment. To alleviate localization sensitivity, loss re-weighting strategies [36] have been introduced to emphasize small and medium-range errors, thereby enhancing robustness under challenging conditions. Subsequent studies further enhance coordinate regression by explicitly modeling structural dependencies among facial landmarks. Coarse-to-fine frameworks [37] incorporating landmark-guided self-attention capture global contextual relationships and improve spatial consistency. In addition, graph-based relational modeling [38] has been explored to encode inter-landmark dependencies explicitly, enabling more structured reasoning over facial geometry. More recently, sparse and adaptive interaction mechanisms [39] have been introduced to eficiently model landmark relationships while reducing computational overhead. Despite these advances, coordinate regression methods remain sensitive to variations in facial geometry and spatial configurations, which limits their robustness in scenarios involving large pose changes, occlusions, or complex expressions.

## 2.2. Heatmap Regression Methods

In contrast to coordinate regression, heatmap regression methods formulate facial landmark detection as a dense spatial prediction problem, where each landmark is represented by a probability heatmap. This formulation provides richer spatial supervision and has therefore become the dominant paradigm, ofering improved robustness to pose variations and partial occlusions. Early heatmap-based approaches primarily enhance structural modeling through stacked convolutional architectures. By integrating multi-stage refinement with shape normalization mechanisms, stacked hourglass-style networks [40] efectively reduce shape variance and improve localization accuracy. Hierarchical ensemble strategies [41] further exploit multi-level representations to implicitly capture inter-landmark relationships and structural consistency. Subsequent studies strengthen robustness by explicitly modeling facial structure and occlusion patterns. Part–whole representations [42] have been introduced to handle severe occlusions by decomposing facial geometry into structured components. In addition, masked representation learning combined with correspondence refinement [43] has been explored to improve landmark localization under missing or corrupted regions. More recently, transformer-based architectures [44] have been adopted to capture long-range spatial dependencies and global contextual interactions across landmarks, enabling more holistic modeling of facial geometry. Despite their strong performance, heatmap regression methods largely operate in the spatial domain and rely on dataset-specific supervision, which makes them sensitive to domain shifts and limits their generalization across datasets.

![](images/6ac3b3b3c3783b40246f9b8443601939d122ec09a60c6fa4fb9e222f6fdfeeac.jpg)  
300W

![](images/963554c4373a017fcc4c6ce330063a9328913edac448f100f3908d0ab13f5330.jpg)  
COFW

![](images/3910e3532b78437d251b70a762312a1549795177b16717cb3a70f31b7ce7ed27.jpg)  
AFLW

![](images/43dd4aece0d51de3de7850bcf7350f08c920c285a29db406a2d1c0049c03ca47.jpg)  
WFLW  
Figure 2: Proposed unified landmark index. A total of 124 unified landmarks are constructed by merging the original landmark annotations from four commonly used facial landmark datasets.

Overall, existing FLD methods mainly focus on dataset-specific optimization and spatial-domain modeling, which limits their generalization across diverse datasets and scenarios. Motivated by these limitations, we propose a unified framework that incorporates FreqMoM and FreqMoE with adaptive expert routing, further regularized by the FreqCR loss, for robust FLD across diverse datasets under the All-in-One paradigm.

## 3. Method

In this section, we first introduce the unified landmark index in Section 3.1, followed by the overall pipeline in Section 3.2. The core components, including the FreqMoM and the FreqMoE, are described in Section 3.3 and Section 3.4, respectively. Finally, the FreqCR loss is introduced in Section 3.5.

## 3.1. Unified Landmark Index

Diferent FLD datasets contain diferent numbers of landmarks, and some landmarks in diferent datasets correspond to the same semantic information. Hence, to construct a universal landmark detection paradigm, we first define a universal landmark version that aims to contain all landmarks from the popular facial landmark datasets including 300W, WFLW, COFW, and AFLW. Each landmark in the universal landmark version has a specific index for identification. As a result, we establish a set of 124 universal landmarks with well-defined semantics and a unified indexing system for consistent referencing, as shown in Fig.2. With the proposed universal landmark definition, FreqFLD can leverage shared landmark annotations across diverse datasets, thereby improving the overall detection accuracy.

![](images/81a55628d6bbf1b80f52f51fa4f0574124415276a721298c93a76ba1ec9f6e5f.jpg)  
Figure 3: The proposed FreqFLD first preprocesses the input image and extracts hierarchical representations through a multi-scale encoder-decoder architecture composed of TSAB and SSAB blocks. At the bottleneck stage, features are decomposed by FreqMoM into highand low-frequency components, which are further refined via HFPB and LFPB to generate frequency-aware prompts. These prompts guide the decoder through the FreqMoE, enabling dynamic expert activation according to sample complexity. The refined output is subsequently processed by the final prediction head to obtain the unified landmark index.

## 3.2. Overall Pipeline

We propose FreqFLD (Fig. 3), a frequency-modulated All-in-One framework for robust FLD. Given an input facial image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , FreqFLD first applies a 3×3 convolutional embedding layer to obtain shallow features $\mathbf { F } \in \mathbb { R } ^ { B \times C ^ { \prime } \times H \times W }$ The embedded features are then processed by a multi-stage encoder–decoder architecture with skip connections for hierarchical feature fusion. To expose geometry-sensitive frequency cues, features $\mathbf { F } _ { s } \in \mathbb { R } ^ { \frac { H } { 8 } \times \frac { W } { 8 } \times 8 C }$ at the intermediate stage are processed by the proposed FreqMoM, which decomposes them into complementary low-frequency structural components and high-frequency detail components and produces frequency-modulated priors. These priors are subsequently leveraged in the decoder to guide the FreqMoE, where expert routing is adaptively conditioned on the frequency-aware representations to model heterogeneous facial patterns. After the refinement block aggregation, the features are progressively upsampled to produce dense landmark-aware feature maps $\begin{array} { r } { \hat { \mathbf { F } } \in \mathbb { R } ^ { B \times C \times \frac { H } { 4 } \times \frac { W } { 4 } } } \end{array}$ . Finally, a lightweight prediction head generates high-resolution landmark heatmaps $\hat { \mathbf { Y } } \in \mathbb { R } ^ { B \times 1 2 4 \times \frac { H } { 2 } \times \frac { W } { 2 } }$ , where 124 denotes the number of unified facial landmarks. A FreqCR loss is further introduced to regularize expert assignment during training under the All-in-One paradigm.

## 3.3. Frequency Modulation Module (FreqMoM)

FLD inherently involves heterogeneous frequency characteristics, with lowfrequency components modeling global facial geometry and high-frequency components emphasizing fine-grained landmark details. However, existing FLD methods predominantly operate in the spatial domain, without explicitly modeling frequency heterogeneity across regions. Under the All-in-One setting, this spatialdomain bias amplifies cross-dataset conflicts, weakening fine-grained landmark representations. To address this issue, we propose FreqMoM, which explicitly decomposes intermediate features into complementary low- and high-frequency components and adaptively modulates them to construct frequency-aware features.

Unified Frequency Decoupling Module (UFDM). Given an input shallow feature map $\mathbf { F } _ { s } ,$ FreqMoM first applies global average pooling (GAP) followed by a $1 \times 1$ convolution to extract a compact global descriptor $\mathbf { F } _ { s } ^ { \prime }$ . To generate an input-adaptive low-pass filter, $\mathbf { F } _ { s } ^ { \prime }$ is further processed by a lightweight gating branch. Specifically, the gating branch predicts a sigmoid-normalized modulation map, which is multiplied with $\mathbf { F } _ { s } ^ { \prime }$ to obtain the gated representation $\hat { \mathbf { F } } _ { s }$ . The gated representation is then normalized by batch normalization and a softmax operation to produce the adaptive low-pass filter $\mathbf { K } _ { \mathrm { l } }$ . This process is formulated as:

$$
\mathbf { F } _ { s } ^ { \prime } = \operatorname { C o n v } _ { 1 \times 1 } ( \operatorname { G A P } ( \mathbf { F } _ { s } ) ) ,\tag{1}
$$

$$
\hat { \mathbf { F } } _ { s } = \mathrm { S i g m o i d } \left( \mathcal { G } ( \mathbf { F } _ { s } ^ { \prime } ) \right) \odot \mathbf { F } _ { s } ^ { \prime } ,\tag{2}
$$

$$
\mathbf { K } _ { 1 } = \mathrm { S o f t m a x } \left( \mathcal { B } \mathcal { N } ( \hat { \mathbf { F } } _ { s } ) \right) ,\tag{3}
$$

where $\mathcal { G } ( \cdot )$ is implemented by a $1 \times 1$ convolution, ⊙ denotes element-wise multiplication, $B \mathcal { N } ( \cdot )$ denotes batch normalization, and $\mathbf { K } _ { 1 }$ denotes the learned adaptive low-pass filter, which is then applied to the input feature ${ \bf F } _ { s }$ through a convolution operation to obtain the low-frequency features. The filtering operation is formulated as:

$$
\mathbf { F } _ { 1 } = \mathbf { K } _ { 1 } * \mathbf { F } _ { s } ,\tag{4}
$$

where ∗ denotes the convolution operation and $\mathbf { F } _ { 1 }$ denotes the low-frequency features. Furthermore, the high-frequency features are obtained through an element-wise subtraction operation, which can be defined as:

$$
\mathbf { F } _ { \mathrm { h } } = \mathbf { F } _ { s } - \mathbf { F } _ { \mathrm { l } } ,\tag{5}
$$

where $\mathbf { F } _ { \mathrm { h } }$ is complementary to $\mathbf { F } _ { 1 } .$ , with low-frequency components capturing global facial geometry and high-frequency components emphasizing fine-grained structural details. Together, they serve as efective priors that guide subsequent feature modeling toward geometrically informative regions within a unified design.

Unified Frequency Modulation Module (UFMM). Although low- and high-frequency components are explicitly decoupled, they are not independent in terms of semantic roles. Both components originate from the same facial geometry and jointly reflect the underlying structural priors of facial landmarks. Motivated by this observation, we propose the UFMM, which modulates low- and high-frequency features in an explicit manner under the All-in-One paradigm.

Given the input feature ${ \bf F } _ { s }$ , UFMM incorporates the high-frequency prior $\mathbf { F } _ { \mathrm { h } }$ and the low-frequency prior $\mathbf { F } _ { 1 }$ into the High-Frequency Prompt Block (HFPB) (Fig. 4, top) and the Low-Frequency Prompt Block (LFPB) (Fig. 4, bottom), respectively. These processes can be defined as:

$$
\begin{array} { r } { \mathbf { Z } _ { \mathrm { h } } = H ( \mathbf { F } _ { s } \mid \mathbf { F } _ { \mathrm { h } } ) , } \\ { \mathbf { Z } _ { \mathrm { l } } = L ( \mathbf { F } _ { s } \mid \mathbf { F } _ { \mathrm { l } } ) , } \end{array}\tag{6}
$$

![](images/99cd4e9a339a2e7505f773693f96453cd9f968a631289317eeab4698db8f49c2.jpg)  
Figure 4: The top and bottom panels illustrate the High-Frequency Prompt Block (HFPB) and Low-Frequency Prompt Block (LFPB), respectively.

where (· | ·) denotes that each branch is explicitly modulated by its corresponding frequency prior. Finally, the refined features $\mathbf { Z } _ { \mathrm { h } }$ and $\mathbf { Z } _ { \mathrm { l } }$ are processed by a concatenation operation along the channel dimension to construct the refined feature X<sub>FP</sub>. Thus, UFMM enables the two branches to model complementary frequency components in a decoupled manner, while both remain governed by the same underlying facial structural prior.

(1) High-frequency Prompt Block (HFPB). Given the input feature ${ \bf F } _ { s }$ and the high-frequency prior $\mathbf { F } _ { \mathrm { h } }$ , HFPB aims to selectively enhance finegrained structural details through frequency-aware prompting and local attention. Specifically, the high-frequency prior $\mathbf { F } _ { \mathrm { h } }$ is first processed by a lightweight depthwise convolution to aggregate local responses, and is subsequently modulated by a GELU-based gating function to suppress noisy activations while preserving informative high-frequency cues, achieved via element-wise multiplication with the high-frequency prior $\mathbf { F } _ { \mathrm { h } }$ . The gated feature is then further modulated by a learnable high-frequency prompt $\mathbf { P } _ { \mathrm { h } } ,$ yielding a frequency-enhanced prompt representation. These processes can be formulated as:

$$
\mathbf { F } _ { \mathrm { h } } ^ { \mathrm { p } } = \left( \mathbf { F } _ { \mathrm { h } } \odot \sigma ( \mathrm { D C o n v } ( \mathbf { F } _ { \mathrm { h } } ) ) \right) \odot \mathbf { P } _ { \mathrm { h } } ,\tag{7}
$$

where $\mathbf { F } _ { \mathrm { h } } ^ { \mathrm { p } }$ denotes the high-frequency prompt feature, serving as a frequencyenhanced representation that highlights geometrically informative local details.

To enhance fine-grained structural modeling, $\mathbf { F _ { \mathrm { h } } ^ { \mathrm { p } } }$ is refined via a crossattention [45] mechanism guided by the input feature $\mathbf { F } _ { s }$ . This process can be defined as:

$$
\mathbf { F } _ { \mathrm { h } } ^ { \mathrm { o u t } } = \mathrm { M H C A } ( \mathbf { F } _ { s } , \mathbf { F } _ { \mathrm { h } } ^ { \mathrm { p } } , \mathbf { F } _ { \mathrm { h } } ^ { \mathrm { p } } ) ,\tag{8}
$$

where ${ \bf F } _ { s }$ denotes the query, $\mathbf { F } _ { \mathrm { h } } ^ { \mathrm { p } }$ denotes the key and value embeddings. MHCA denotes multi-head cross-attention. The resulting feature $\mathbf { F } _ { \mathrm { h } } ^ { \mathrm { o u t } }$ represents the refined high-frequency response, in which fine-grained structural details are selectively emphasized under the guidance of the input feature.

(2) Low-frequency Prompt Block (LFPB). LFPB focuses on modeling global facial geometry and large-scale structural cues. Given the input feature ${ \bf F } _ { s }$ and the low-frequency prior $\mathbf { F } _ { 1 }$ , the low-frequency prior is first aligned to the spatial resolution of ${ \bf F } _ { s }$ and transformed into the frequency domain via a fast Fourier transform (FFT). Then, a lightweight gating operation is applied to regulate the propagation of informative low-frequency responses, followed by modulation with a learnable low-frequency prompt ${ \bf P } _ { 1 }$ . The modulated features are subsequently restored to the spatial domain through an inverse FFT, yielding the low-frequency prompt representation:

$$
\mathbf { F } _ { 1 } ^ { \mathrm { p } } = { \mathcal { F } } ^ { - 1 } { \big ( } { \big ( } { \mathcal { F } } ( \mathbf { F } _ { 1 } ) \odot \sigma { \big ( } { \mathrm { C o n v } } _ { 1 \times 1 } \left( { \mathcal { F } } ( \mathbf { F } _ { 1 } ) \right) { \big ) } { \big ) } \odot \mathbf { P } _ { 1 } { \big ) } ,\tag{9}
$$

where $\mathcal F ( \cdot )$ and $\mathcal { F } ^ { - 1 } ( \cdot )$ denote the FFT and inverse FFT, respectively.

To incorporate structural guidance from the input feature, the low-frequency prompt $\mathbf { F } _ { \mathrm { l } } ^ { \mathrm { p } }$ is further refined via a cross-attention mechanism. The refinement process is formulated as:

$$
\mathbf { F } _ { 1 } ^ { \mathrm { o u t } } = \mathrm { M H C A } ( \mathbf { F } _ { s } , \mathbf { F } _ { 1 } ^ { \mathrm { p } } , \mathbf { F } _ { 1 } ^ { \mathrm { p } } ) ,\tag{10}
$$

![](images/9f90fab65a8c168601786ef6d5667056f6352dafc1d25c95c8a2352f3bb9802e.jpg)  
Figure 5: The proposed FreqMoE. FreqMoE consists of a Shared Structural Expert Modeling (ShSEM) path and a Frequency-Adaptive Expert Modeling $( { \mathrm { F r e q A E M } } )$ path. ShSEM captures unified facial structural representations to preserve common geometric priors, while FreqAEM performs frequency-aware image-level routing and activates pose-complexity-aware experts for modeling sample-specific facial variations. The two paths are finally fused by cross-attention to produce the refined output feature.

where ${ \bf F } _ { s }$ denotes the query, $\mathbf { F } _ { \mathrm { l } } ^ { \mathrm { p } }$ denotes the key and value embeddings and ${ \bf F } _ { 1 } ^ { \mathrm { o u t } }$ denotes the refined low-frequency feature that captures coherent global facial structure.

## 3.4. Frequency Mixture-of-Experts (FreqMoE)

FLD is inherently challenged by large variations in facial pose and structural complexity across samples, especially under the All-in-One training paradigm. However, conventional MoE architectures fail to adequately account for such sample-wise heterogeneity, as they typically adopt uniformly configured experts and static capacity allocation. To address this limitation, we propose FreqMoE (Fig. 5), a pose-complexity-aware mixture framework for All-in-One FLD. Freq MoE integrates a unified shared structural expert modeling (ShSEM) path for preserving common facial priors and a frequency-adaptive expert modeling (FreqAEM) path for handling sample-specific pose and frequency variations. Their complementarity enables FreqMoE to unify stable structural prior preservation with complexity-aware frequency adaptation.

## 3.4.1. Unified Structural Modeling

FreqMoE is designed as a unified dual-path modeling framework that integrates ShSEM and FreqAEM. The overall process is expressed as:

$$
\begin{array} { r l } & { \mathbf { z } _ { s } = \mathrm { S h S E M } ( \mathbf { x } ) , } \\ & { \mathbf { z } _ { a } = \mathrm { F r e q A E M } ( \mathbf { x } , \mathcal { X } _ { \mathrm { F P } } , \mathbf { z } _ { s } ) , } \\ & { \mathbf { y } = \mathrm { M H C A } ( \mathbf { z } _ { a } , \mathbf { z } _ { s } ) + \mathbf { x } , } \end{array}\tag{11}
$$

where ShSEM(·) denotes the shared structural modeling and FreqAEM(·) denotes the frequency-adaptive expert modeling. This process allows FreqMoE to jointly preserve common facial structural priors and model sample-specific pose-frequency variations.

## 3.4.2. Shared Structural Expert Modeling (ShSEM)

To decouple shared structural representation from sample-specific adaptive modeling, given the input feature x, we first project it through a $1 \times 1$ convolution and then apply an MHSA mechanism to capture unified facial structural representations. This process can be formulated as:

$$
\begin{array} { r l } & { \mathbf { x } _ { s } = \phi _ { s } ( \mathbf { x } ) , } \\ & { } \\ & { \mathbf { z } _ { s } = \mathrm { M H S A } ( \mathbf { x } _ { s } ) , } \end{array}\tag{12}
$$

where $\phi _ { s } ( \cdot )$ denotes the $1 \times 1$ convolutional projection, $\mathbf { x } _ { s }$ is the projected shared feature, and $\mathbf { z } _ { s }$ denotes the unified structural representation. Through MHSA, $\mathbf { z } _ { s }$ aggregates long-range dependencies among facial components, allowing the ShSEM to encode global facial geometry and consistent structural priors. Therefore, $\mathbf { z } _ { s }$ provides a structure-aware reference for subsequent fusion with the frequency-adaptive expert output.

## 3.4.3. Frequency-Adaptive Expert Modeling (FreqAEM)

To capture sample-specific pose and frequency variations, we introduce the refined frequency-aware feature $\mathcal { X } _ { \mathrm { F P } }$ into the adaptive expert modeling. This enables frequency-aware facial geometry modulation by selecting suitable experts in the frequency domain, complementing the unified structural representation $\mathbf { z } _ { s } .$

FreqAEM consists of image-level expert routing and a pose-complexity-aware expert group. The routing mechanism predicts sample-wise expert probabilities from the image-level representation and the refined frequency-aware feature $\chi _ { \mathrm { F P } } .$ enabling adaptive expert allocation according to pose complexity and frequency characteristics. The expert group introduces progressively enlarged frequency receptive fields through diferent patch sizes, while all experts share the same frequency-modulated architecture, providing multi-scale structural modeling for All-in-One FLD. The overall process is formulated as:

$$
\mathbf { z } _ { a } = \sum _ { e = 1 } ^ { \mathrm { N } } \mathcal { R } _ { e } \left( \mathbf { x } , \mathcal { X } _ { \mathrm { F P } } \right) \mathbf { E } _ { e } \left( \mathbf { x } , \mathbf { z } _ { s } \right) ,\tag{13}
$$

where N denotes the number of experts, $\mathcal { R } _ { e } ( { \bf x } , \mathcal { X } _ { \mathrm { F P } } )$ is the frequency-aware image-level expert routing weight assigned to the e-th expert, and ${ \bf E } _ { e } \left( { \bf x } , { \bf z } _ { s } \right)$ denotes the output of the e-th frequency-aware expert.

(1) Frequency-aware Image-level Expert Routing. In FLD, achieving scale-invariant tokenization is particularly challenging, as facial structures must remain consistent across varying resolutions and facial scales. Consequently, conventional token-level routing strategies commonly adopted in prior MoE frameworks [46, 47, 48] may lead to unstable expert assignment and fragmented structural reasoning. To address this issue, we adopt an image-level routing strategy, where each input image is routed as a whole to a single expert, thereby preserving global facial structure and scale consistency.

Given an input feature x and the refined frequency-aware feature $\mathcal { X } _ { \mathrm { F P } }$ FreqMoE applies a lightweight frequency-conditioned routing function $\mathcal { R } ( \cdot )$ to estimate the association between the input representation and each expert. Specifically, the input feature x is first aggregated into an image-level descriptor by GAP, reshaped into a vector, and then mapped to image-level routing logits through a fully connected layer. Meanwhile, $\mathcal { X } _ { \mathrm { F P } }$ is fed into a frequency-aware gate to produce frequency-conditioned routing logits. The two logits are added to produce the final routing logits:

$$
\mathbf { r } = \mathrm { F C } _ { x } \left( \mathrm { G A P } ( \mathbf { x } ) \right) + \mathrm { F C } _ { f } \left( \mathcal { X } _ { \mathrm { F P } } \right) ,\tag{14}
$$

where r denotes the routing logits, $\mathrm { F C } _ { x } ( \cdot )$ maps the GAP-based image-level descriptor to expert logits, and $\mathrm { F C } _ { f } ( \cdot )$ maps the refined frequency-aware feature $\mathcal { X } _ { \mathrm { F P } }$ to frequency-conditioned expert logits.

To encourage exploration and mitigate early routing collapse, FreqMoE injects independent Gaussian noise $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ into the routing logits. The noisy logits are then normalized by the Softmax operation and sparsified by the Top-k function:

$$
\mathcal { R } ( { \bf x } , \mathcal { X } _ { \mathrm { F P } } ) = \mathrm { T o p } _ { k } \left( \mathrm { S o f t m a x } \left( { \bf r } + \epsilon \right) \right) .\tag{15}
$$

(2) Expert Design. Facial pose variations introduce scale-dependent structural complexity, requiring both local landmark details and global facial geometry. Therefore, FreqAEM constructs a pose-complexity-aware expert group with progressively enlarged receptive fields, where experts share the same frequency-modulated structure but use diferent patch sizes (e.g., smaller patches focus on local landmark details, while larger patches capture broader facial geometry). For each expert, given the feature x and $\mathbf { z } _ { s } .$ , we first project them into an expert-specific embedding space:

$$
\begin{array} { r } { \mathbf { Z } _ { f } = \phi _ { x } ( \mathbf { x } ) , } \\ { \mathbf { P } _ { s } = \phi _ { s } ( \mathbf { z } _ { s } ) , } \end{array}\tag{16}
$$

where $\mathbf { Z } _ { f }$ denotes the projected input representation, ${ \bf P } _ { s }$ denotes the projected shared representation, and $\phi _ { x } ( \cdot )$ and $\phi _ { s } ( \cdot )$ are two independent 1×1 convolutional projections. Then, to capture frequency-structured features, each frequency expert transforms the feature $\mathbf { Z } _ { f }$ into query, key, and value representations:

$$
\begin{array} { r } { \mathbf { Q } = \mathrm { D W C o n v } _ { 3 \times 3 } \left( \mathrm { C o n v } _ { 1 \times 1 } ( \mathbf { Z } _ { f } ) \right) , } \\ { \mathbf { \Phi } } \\ { \mathbf { [ K , V ] } = \mathrm { D W C o n v } _ { 7 \times 7 } \left( \mathrm { C o n v } _ { 1 \times 1 } ( \mathbf { Z } _ { f } ) \right) . } \end{array}\tag{17}
$$

Subsequently, $\mathbf { Q }$ and K are partitioned into local patches and transformed into the frequency domain. Their frequency-domain cross-interaction is computed as:

$$
\mathbf { O } _ { f } = \mathcal { P } _ { p _ { i } } ^ { - 1 } \left[ \mathcal { F } ^ { - 1 } \left( \mathcal { F } \left( \mathcal { P } _ { p _ { i } } ( \mathbf { Q } ) \right) \odot \mathcal { F } \left( \mathcal { P } _ { p _ { i } } ( \mathbf { K } ) \right) \right) \right] ,\tag{18}
$$

where $\mathcal { P } _ { p _ { i } } ( \cdot )$ denotes patch partition with patch size $p _ { i } \in \{ 2 ^ { i + 2 } \} _ { i = 0 } ^ { N - 1 }$ for the i-th frequency expert, and $\mathcal { P } _ { p _ { i } } ^ { - 1 } ( \cdot )$ denotes the inverse patch rearrangement, $\mathcal F ( \cdot )$ and $\mathcal { F } ^ { - 1 } ( \cdot )$ denote FFT and inverse FFT, respectively, and ⊙ denotes element-wise multiplication in the frequency domain. After obtaining the frequency-domain interaction response $\mathbf { O } _ { f }$ , we further normalize it and modulate it with the value representation V to inject spatial content information:

$$
\mathbf { Z } _ { f } ^ { \prime } = \phi _ { o } \left( \operatorname { L N } ( \mathbf { O } _ { f } ) \odot \mathbf { V } \right) ,\tag{19}
$$

where $\phi _ { o } ( \cdot )$ denotes the output $1 \times 1$ convolution, and $\mathbf { Z } _ { f } ^ { \prime }$ is the frequencyenhanced representation, which is then reweighted by the shared modulation signal ${ \bf P } _ { s }$ and combined with the input through a residual connection. This process is defined as:

$$
\begin{array} { r } { \tilde { \mathbf { Z } } _ { f } = \rho \left( \mathbf { Z } _ { f } ^ { \prime } \odot \sigma ( \mathbf { P } _ { s } ) \right) + \mathbf { x } , } \end{array}\tag{20}
$$

where $\sigma ( \cdot )$ denotes the SiLU activation, $\rho ( \cdot )$ denotes the output projection, and ⊙ denotes element-wise multiplication.

The above operation integrates frequency-refined cues with expert-specific representations, enabling fine-grained FLD under the All-in-One paradigm across diverse facial variations and imaging conditions.

## 3.5. Frequency-Consistent Routing (FreqCR) Loss

In MoE architectures, experts with heterogeneous computational capacities are susceptible to imbalanced utilization, routing instability, and degenerate expert collapse. These issues are further exacerbated under the All-in-One paradigm, where a single model is required to generalize across diverse facial conditions and varying levels of structural and frequency complexity, leading to pronounced disparities in expert activation and routing behavior. To address these issues, we introduce a FreqCR loss, which explicitly regularizes expert utilization to be both balanced across experts and consistent with their computational capacities. The FreqCR loss is defined as:

$$
\mathcal { L } _ { \mathrm { F r e q C R } } = \frac { 1 } { 2 } \mathcal { L } _ { \mathrm { i m p } } + \frac { 1 } { 2 } \mathcal { L } _ { \mathrm { l o a d } } ,\tag{21}
$$

where ${ \mathcal { L } } _ { \mathrm { i m p } }$ enforces balanced expert importance under a complexity-aware bias, and $\mathcal { L } _ { \mathrm { l o a d } }$ regularizes the efective expert assignment frequency during routing.

(1) Complexity-aware importance loss. To incorporate expert computational capacity into routing, each expert is associated with a pre-computed complexity prior $\theta ,$ where each element $\theta _ { e } \in \pmb \theta$ denotes the number of learnable parameters of the e-th expert and is fixed during training. A normalized complexity bias is then defined as $\beta = \theta / \theta _ { \mathrm { m a x } }$ . For a mini-batch $B ,$ the complexity-aware importance loss is formulated as:

$$
\mathcal { L } _ { \mathrm { i m p } } = \mathrm { C O V } \left( \left( \sum _ { x \in \mathcal { B } } \mathrm { S o f t m a x } ( \mathcal { R } ( x ) ) \right) \odot \beta \right) ^ { 2 } ,\tag{22}
$$

where Softmax $\mathbf { \alpha } ( { \mathcal { R } } ( x ) )$ yields the normalized importance weights over experts, $\beta \in \mathbb { R } ^ { n }$ is the normalized complexity bias derived from the expert parameter counts, ⊙ denotes element-wise multiplication, and COV(·) [47] denotes the coeficient of variation computed across experts.

(2) Load balancing loss. To mitigate routing imbalance, we estimate the expected expert load using a probabilistic approximation under noisy Top-k routing. Given routing logits $\mathcal { R } ( x ) \in \mathbb { R } ^ { n }$ , we inject Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and obtain noisy logits, which can be formulated as:

$$
\tilde { \mathcal { R } } ( x ) = \mathcal { R } ( x ) + \epsilon .\tag{23}
$$

Let $\tau ( x )$ denote the selection threshold defined as the k-th largest value in $\tilde { \mathcal { R } } ( x )$ . The expected expert load over a mini-batch B and the corresponding load balancing loss are defined as:

$$
\mathcal { L } _ { \mathrm { l o a d } } = \mathrm { C O V } \Bigg ( \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } \Big ( \mathbf { 1 } - \Phi \Big ( \frac { \tau ( x ) \mathbf { 1 } - \mathcal { R } ( x ) } { \sigma } \Big ) \Big ) \Bigg ) ^ { 2 } ,\tag{24}
$$

where $\mathbf { 1 } \in \mathbb { R } ^ { n }$ is an all-ones vector used for threshold broadcasting, $\sigma$ denotes the per-expert noise standard deviation, Φ(·) is the standard normal cumulative

distribution function (CDF) used to approximate expert selection probability under noisy Top-k routing, and COV(·) measures the coeficient of variation across experts.

Finally, the overall training objective is defined by integrating a MSE loss [49] into the optimization, which is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { M S E } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { F r e q C R } } , } \end{array}\tag{25}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ control the corresponding weights of the individual loss terms, respectively.

## 4. EXPERIMENTS

We evaluate the proposed method on four widely adopted FLD benchmarks, including 300W [50], COFW [51], WFLW [52], and AFLW [53]. Comparisons are conducted against state-of-the-art FLD approaches, where ◦ and ⋄ denote heatmap regression and coordinate regression methods, respectively. We further present comprehensive ablation studies and self-evaluation analyses to systematically examine the efectiveness of the proposed FreqFLD framework.

## 4.1. Datasets and Implementation Details

300W (68 landmarks) [50]. The 300W dataset is a widely adopted benchmark for facial landmark detection. It comprises 3,148 training images and 689 testing images, with each face annotated using 68 landmarks. The test set is further divided into two subsets: a Common subset and a Challenging subset. The Common subset is composed of 224 images drawn from the LFPW [54] test set and 330 images from the HELEN [55] test set, while the Challenging subset contains 135 images from the IBUG [50] dataset, which exhibit significant pose variations and occlusions.

WFLW (98 landmarks) [52]. WFLW consists of 7,500 training images and 2,500 testing images, each annotated with 98 facial landmarks. The test set is further categorized into multiple subsets according to diverse facial variations, including pose, expression, illumination, and occlusion. This dataset provides a comprehensive benchmark for evaluating the robustness of FLD methods.

Table 1: Comparison of state-of-the-art methods on the 300W dataset, with NME normalized by inter-ocular distance. (% omitted)
<table><tr><td>Method</td><td>Common</td><td>Challenging</td><td>Full</td></tr><tr><td>。HGs (ECCV16) [40]</td><td>3.72</td><td>7.23</td><td>4.41</td></tr><tr><td>。MDM (CVPR16) [56]</td><td>4.36</td><td>7.56</td><td>4.99</td></tr><tr><td>。FAN (ICCV17) [57]</td><td>3.08</td><td>5.52</td><td>3.56</td></tr><tr><td>。LAB (CVPR18) [52]</td><td>2.98</td><td>5.19</td><td>3.49</td></tr><tr><td>。Wing (CVPR18) [36]</td><td>2.93</td><td>5.23</td><td>3.38</td></tr><tr><td>。ODN (CVPR19) [58]</td><td>3.56</td><td>6.67</td><td>4.17</td></tr><tr><td>◇AWing (ICCV19) [59]</td><td>2.72</td><td>4.52</td><td>3.07</td></tr><tr><td>◇LUVLi (CVPR20) [60]</td><td>2.76</td><td>5.16</td><td>3.23</td></tr><tr><td>SAAT (ICCV21) [61]</td><td>2.82</td><td>5.03</td><td>3.25</td></tr><tr><td>SDFL (TIP21) [38]</td><td>2.88</td><td>4.93</td><td>3.28</td></tr><tr><td>SLPT (CVPR22) [39]</td><td>2.75</td><td>4.90</td><td>3.17</td></tr><tr><td>GlomFace (CVPR22) [42]</td><td>2.72</td><td>4.79</td><td>3.13</td></tr><tr><td>PicassoNet (TNNLS23) [62]</td><td>3.03</td><td>5.81</td><td>3.58</td></tr><tr><td>GFL (CVPR24) [63]</td><td>2.79</td><td>4.91</td><td>3.20</td></tr><tr><td> FreqFLD (ours)</td><td>2.72</td><td>4.52</td><td>3.08</td></tr></table>

COFW (29 landmarks) [51]. COFW is a benchmark dataset designed for evaluating facial landmark detection in the presence of heavy occlusions. The dataset comprises 1,345 facial images annotated with 29 landmarks. Among them, 845 images are used for training and the remaining 500 images are used for testing.

AFLW (19 landmarks) [53]. The AFLW dataset contains 24,386 face images with large pose variations. Following standard evaluation protocols, we use 19 facial landmarks for evaluation.

Evaluation Metrics. FreqFLD adopts the Normalized Mean Error (NME) as the primary evaluation metric for FLD. Following standard practice, diferent normalization factors are used for diferent datasets: the inter-ocular distance is adopted for 300W and WFLW, the face bounding box size is used for AFLW, and the inter-pupil distance is employed for COFW. In addition, we also report the Failure Rate (FR) for the COFW dataset.

Table 2: Comparison of state-of-the-art methods on the COFW dataset, with NME normalized by inter-pupil distance (% omitted).
<table><tr><td>Method</td><td> ${ \mathrm { N M E } } _ { \mathrm { i p } }$ </td><td>FR</td></tr><tr><td>。Wing (CVPR18) [36]</td><td>5.44</td><td>3.75</td></tr><tr><td>。DCFE (ECCV18) [64]</td><td>5.27</td><td>0.35</td></tr><tr><td>。AWing (ICCV19) [59]</td><td>4.94</td><td>0.99</td></tr><tr><td>。ODN (CVPR19) [58]</td><td>5.30</td><td>一</td></tr><tr><td>。MHHN (TIP20) [65]</td><td>4.95</td><td>1.78</td></tr><tr><td>ADNet (ICCV21) [66]</td><td>4.68</td><td>0.59</td></tr><tr><td>MMDN (TNNLS22) [31]</td><td>5.01</td><td>1.78</td></tr><tr><td>◇SLPT (CVPR22) [39]</td><td>4.79</td><td>1.18</td></tr><tr><td>DSLPT-R50 (TPAMI23) [67]</td><td>4.81</td><td>1.18</td></tr><tr><td>CIT-v2 (IJCV24) [68]</td><td>5.81</td><td>3.55</td></tr><tr><td> FreqFLD (ours)</td><td>4.80</td><td>0.39</td></tr></table>

Implementation Details. All input images are resized to $2 5 6 \times 2 5 6 \times 3$ in our experiments. The loss weights $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are set to 1 and $1 \times 1 0 ^ { - 4 }$ , respectively. During training, data augmentation is applied to enhance model robustness, including random rotations within $\pm 3 0 ^ { \circ }$ and horizontal flipping with a probability of 0.5. The proposed FreqFLD is implemented in $\mathrm { P y }$ Torch and trained on an NVIDIA RTX 4090 GPU for 100,000 iterations with a batch size of 8. We employ the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$

All-in-One Training Paradigm. Beyond the conventional paradigm, we further extend our evaluation to an All-in-One training setting, where multiple datasets are jointly used to train a single unified model. In this setting, the training data from 300W, AFLW, COFW, and WFLW are combined, and each dataset is expanded to contain an equal number (20,000) of training samples, resulting in a balanced training set of 80,000 images in total. The unified model is then separately evaluated on each test set following the standard protocol of the corresponding dataset.

Table 3: Comparison of state-of-the-art methods on the WFLW dataset, with NME normalized by inter-ocular distance (% omitted).
<table><tr><td>Method</td><td colspan="6">Pose Expression Illumination Make-Up Occlusion Blur Testset</td></tr><tr><td></td><td>Subset</td><td>Subset</td><td>Subset</td><td>Subset</td><td>Subset</td><td>Subset</td></tr><tr><td>。 ESR [69]</td><td>11.13</td><td>25.88</td><td>11.47</td><td>10.49 11.05</td><td>13.75</td><td>12.20</td></tr><tr><td>。SDM [70]</td><td>10.29</td><td>24.10</td><td>11.45</td><td>9.32 9.38</td><td>13.03</td><td>11.28</td></tr><tr><td>。CFSS [71]</td><td>9.07</td><td>21.36</td><td>10.09</td><td>8.30 8.74</td><td>11.76</td><td>9.96</td></tr><tr><td>。LAB (CVPR18) [52]</td><td>5.27</td><td>10.24</td><td>5.51</td><td>5.23 5.15</td><td>6.79</td><td>6.32</td></tr><tr><td>。Wing (CVPR18) [36]</td><td>5.11</td><td>8.75</td><td>5.36</td><td>4.93 5.41</td><td>6.37</td><td>5.81</td></tr><tr><td>。DeCaFA (ICCV19) [72]</td><td>4.62</td><td>8.11</td><td>4.65</td><td>4.41 4.63</td><td>5.74</td><td>5.38</td></tr><tr><td>HRNet (TPAMI20) [73]</td><td>4.60</td><td>7.86</td><td>4.78</td><td>4.57 4.26</td><td>5.42</td><td>5.36</td></tr><tr><td>MHHN (TIP20) [65]</td><td>4.77</td><td>9.31</td><td>4.79</td><td>4.72</td><td>4.59 6.17</td><td>5.82</td></tr><tr><td>MMDN (TNNLS21) [31]</td><td>4.87</td><td>7.71</td><td>4.79</td><td>4.61</td><td>4.72 6.17</td><td>5.72</td></tr><tr><td>GlomFace (CVPR22) [42]</td><td>4.81</td><td>8.71</td><td>-</td><td>-</td><td>- 5.14</td><td>一</td></tr><tr><td>EfficientFan (TNNLS23) [74]</td><td>4.54</td><td>8.20</td><td>4.87</td><td>4.39</td><td>4.54 5.42</td><td>5.04</td></tr><tr><td>PicassoNet (TNNLS23) [62]</td><td>4.82</td><td>8.61</td><td>5.14</td><td>4.73</td><td>4.68 5.91</td><td>5.56</td></tr><tr><td>o FreqFLD (ours)</td><td>4.50</td><td>7.54</td><td>4.59</td><td>4.55</td><td>4.46 5.53</td><td>5.15</td></tr></table>

## 4.2. Quantitative analysis

In this section, we quantitatively evaluate the proposed FreqFLD on diverse and challenging scenarios under the All-in-One training paradigm.

Evaluations under Normal Circumstances. Under normal circumstances, we conduct comparative evaluations on the 300W and AFLW benchmarks, which mainly contain favorable facial images. FreqFLD achieves an NME of 2.72 on the 300W Common subset and 3.08 on the 300W Full set (Table 1), achieving comparable FLD performance across heatmap-based and coordinateregression-based methods. Moreover, on the WFLW benchmark, FreqFLD achieves competitive performance across multiple subsets (Table 3). These results can be attributed to the proposed FreqMoM, which explicitly disentangles and enhances frequency-aware facial representations, the FreqMoE that adaptively models heterogeneous landmark patterns, and the FreqCR loss that regularizes expert assignment and stabilizes frequency-consistent learning.

Evaluation of Robustness against Occlusion. To evaluate robustness under occluded scenarios, we conduct experiments on the COFW dataset, the 300W challenging subset, and the WFLW Occlusion subset. On the COFW benchmark (Table 2), FreqFLD achieves an $\mathrm { N M E _ { i p } }$ of 4.80 with a failure rate of 0.39, outperforming most recent FLD methods [59, 58, 67, 68]. Moreover, FreqFLD attains an $\mathrm { N M E _ { i o } }$ of 5.53 (Table 3), demonstrating consistent improvements over existing approaches [52, 36, 72, 31, 79]. Similar performance gains are observed on the 300W challenging subset, where FreqFLD maintains stable accuracy and competitive robustness compared with prior methods [38, 39, 42, 62, 63, 79]. These results are mainly attributed to the proposed FreqMoE. By adaptively activating frequency-aware experts, FreqMoE preserves complementary lowfrequency structural cues and high-frequency landmark-sensitive details, enabling stable FLD performance.

Table 4: Comparison of state-of-the-art methods on the AFLW dataset, with NME normalized by face size. (% omitted)
<table><tr><td>Method</td><td> $\mathrm { N M E } _ { \mathrm { b o x } }$ </td></tr><tr><td>。DAC-CSR (CVPR17) [75]</td><td>2.27</td></tr><tr><td>。SAN (CVPR18) [76]</td><td>1.91</td></tr><tr><td>。LAB (CVPR18) [52]</td><td>1.85</td></tr><tr><td>。LLL (ICCV19) [77]</td><td>1.97</td></tr><tr><td>。LUVLi (CVPR20) [60]</td><td>1.39</td></tr><tr><td>。HRNet (TPAMI20) [73]</td><td>1.57</td></tr><tr><td>。MHHN (TIP20) [65]</td><td>1.38</td></tr><tr><td>PIPNet (IJCV21) [78]</td><td>1.42</td></tr><tr><td>PicassoNet (TNNLS23) [62]</td><td>1.59</td></tr><tr><td>Protoformer (TMM26) [23]</td><td>1.47</td></tr><tr><td>o FreqFLD (ours)</td><td>1.69</td></tr></table>

Evaluation of Robustness against Large Poses . Faces with large pose variations or extreme expressions introduce complex geometric distortions. To assess the performance of FreqFLD under these challenging conditions, we conduct evaluations on WFLW-Pose subset, the AFLW full set, and the 300W challenge subset. As shown in Table 1, FreqFLD achieves an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.52 on the 300W challenging subset, outperforming several SOTA methods [61, 38, 39, 42, 62, 63, 79]. Moreover, FreqFLD achieves comparable NME on the AFLW full set (Table 4) and superior performance on the pose and expression subsets of WFLW (Table 3) compared with existing approaches [36, 72, 73, 74, 62], further confirming its robustness to extreme facial variations. These experimental results indicate that the performance gains of FreqFLD arise from two complementary aspects. By explicitly modeling facial structural cues, FreqFLD exhibits enhanced robustness to large pose and expression variations. Meanwhile, the proposed FreqMoE estimates the pose complexity of input samples and adaptively routes them to experts with appropriate capacity, thereby contributing to robust facial landmark detection performance.

Evaluation of Robustness against Blur. Faces captured under low-light conditions or afected by blur sufer from substantial loss of visual details, which poses considerable challenges for accurate FLD. To evaluate the efectiveness of our proposed FreqFLD, we conduct experiments on the 300W challenging subset and WFLW-blur subset. As reported in Table 1, FreqFLD achieves an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.52 on the 300W challenging subset, outperforming competing approaches. Furthermore, as shown in Table 3, FreqFLD attains $\mathrm { N M E } _ { \mathrm { i o } }$ scores of 4.55 and 5.15 on the illumination and blur subsets of WFLW, respectively, surpassing several representative methods [36, 73, 31, 62]. These results demonstrate that FreqFLD consistently maintains high FLD accuracy even under poor image quality. This robustness can be largely attributed to the proposed FreqMoM, which efectively preserves and enhances facial structural cues, enabling reliable landmark prediction despite severe blur and illumination degradation.

## 4.3. Ablation Study

The ablation studies are conducted to analyze the impact of FreqMoE, FreqMoM and FreqCR loss, together with the efect of multi-dataset joint training. The detailed results are presented as follows.

Influence of FreqMoM, FreqMoE and FreqCR loss. As shown in Table 5, we conduct an ablation study by progressively incorporating FreqMoE, FreqMoM, and FreqCR into the baseline Trans on the 300W challenging subset. The baseline Trans achieves an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.71. By introducing FreqMoE, the NME is reduced to 4.60. Similarly, adding FreqMoM yields an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.62, improving the baseline by 0.09. When FreqMoE and FreqMoM are jointly employed and further equipped with the FreqCR loss, the full model achieves the best performance with an $\mathrm { N M E _ { i o } }$ of 4.52. This result surpasses Trans + FreqMoE and Trans + FreqMoM by margins of 0.08 and 0.10, respectively, and leads to an overall improvement of 0.19 compared with the baseline.

Table 5: Influence of FreqMoE, FreqMoM and FreqCR on the 300W challenging subset.
<table><tr><td>Method</td><td>TB</td><td>FreqMoE</td><td>FreqMoM</td><td>FreqCR</td><td>NMEio</td></tr><tr><td>Trans (baseline)</td><td>√</td><td></td><td></td><td></td><td>4.71</td></tr><tr><td>Trans + FreqMoE</td><td>√</td><td>√</td><td></td><td></td><td>4.60</td></tr><tr><td>Trans + FreqMoM</td><td>√</td><td></td><td>√</td><td></td><td>4.62</td></tr><tr><td>Trans + FreqMoE + FreqMoM + FreqCR</td><td>√</td><td>√</td><td>√</td><td>√</td><td>4.52</td></tr></table>

Table 6: Influence of multi-dataset training on the 300W challenging subset.
<table><tr><td>300W</td><td>AFLW</td><td>WFLW</td><td>COFW</td><td> $\mathrm { N M E } _ { \mathrm { i o } }$ </td></tr><tr><td>√</td><td></td><td></td><td></td><td>4.97</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>4.75</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>4.64</td></tr><tr><td>√</td><td>√</td><td>✓</td><td>√</td><td>4.52</td></tr></table>

These results can be attributed to: 1) The introduced FreqMoE efectively strengthens the model’s capability to adapt to faces of difering complexities. 2) By integrating high- and low-frequency facial cues, the FreqMoM produces refined feature representations, thereby improving the model’s adaptability to diverse facial geometries. By integrating Trans, FreqMoM and FreqMoE, the model achieves high-precision FLD across diferent datasets.

Influence of Multi-dataset Training. We evaluate the efect of multi-dataset training by conducting experiments with diferent combinations of datasets. Using 300W as the baseline, AFLW, WFLW, and COFW are gradually incorporated into the training set. As shown in Table 6, this progressive inclusion leads to performance improvements of 0.22, 0.33, and 0.45, respectively, indicating that leveraging multiple datasets consistently enhances model performance.

Table 7: Leave-one-dataset-out cross-dataset generalization evaluation. (% omitted)
<table><tr><td>Setting</td><td>Training Datasets</td><td>Test Dataset NME</td><td></td></tr><tr><td>LODO-300W</td><td> $\mathrm { W F L W + C O F W + A F L W }$ </td><td>300W</td><td>4.74</td></tr><tr><td>LODO-WFLW</td><td> $\mathrm { 3 0 0 W + C O F W + A F L W }$ </td><td>WFLW</td><td>4.85</td></tr><tr><td>LODO-COFW</td><td> $\mathrm { 3 0 0 W + W F L W + A F L W }$ </td><td>COFW</td><td>5.13</td></tr><tr><td>LODO-AFLW</td><td> $\mathrm { 3 0 0 W + W F L W + C O F W }$ </td><td>AFLW</td><td>1.88</td></tr></table>

Table 8: Influence of diferent numbers of experts and TopK values on the 300W challenging subset (% omitted).
<table><tr><td>Number of experts</td><td>TopK</td><td> $\mathrm { N M E } _ { \mathrm { i o } }$ </td></tr><tr><td>16</td><td>1</td><td>4.95</td></tr><tr><td>12</td><td>1</td><td>4.82</td></tr><tr><td>8</td><td>1</td><td>4.65</td></tr><tr><td>6</td><td>3</td><td>4.75</td></tr><tr><td>4</td><td>2</td><td>4.60</td></tr><tr><td>2</td><td>1</td><td>4.87</td></tr><tr><td>4</td><td>1</td><td>4.52</td></tr></table>

Leave-One-Dataset-Out Cross-Dataset Evaluation. To verify FreqFLD learns generalizable representations across datasets, we conduct a leave-onedataset-out evaluation. In each setting, one dataset is excluded from training and used only for testing, while the remaining three datasets are used for model optimization. As shown in Tab. 7, FreqFLD maintains stable performance across all held-out datasets, achieving $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.74 on 300W and 4.85 on WFLW, $\mathrm { N M E _ { i p } }$ of 5.13 on COFW, and $\mathrm { N M E _ { b o x } }$ of 1.88 on AFLW. These results indicate that FreqFLD can efectively transfer learned frequency-aware facial priors to unseen annotation distributions and facial appearance statistics, demonstrating the cross-dataset generalization ability of FreqFLD.

## 4.4. Self Evaluation

The self evaluation provides comprehensive analyses of the design choices and practical behavior of the proposed FreqFLD. We examine the efects of the number of complexity experts and Top-K routing, followed by evaluations of the FreqMoE and FreqMoM modules, and report time and memory eficiency.

Table 9: The efect of diferent frequency prompts on the 300W challenging subset (% omitted).
<table><tr><td>Method</td><td> $\mathrm { N M E } _ { \mathrm { i o } }$ </td></tr><tr><td> $\mathrm { T r a n s \ ( b a s e l i n e ) + F r e q M o E }$ </td><td>4.60</td></tr><tr><td> $\mathrm { T r a n s } + \mathrm { F r e q M o E } + \mathrm { H F P B }$ </td><td>4.56</td></tr><tr><td> $\mathrm { T r a n s } + \mathrm { F r e q M o E } + \mathrm { L F P B }$ </td><td>4.58</td></tr><tr><td> $\mathrm { T r a n s + F r e q M o E + F r e q M o M \ ( H F P B + L F P B ) }$ </td><td>4.52</td></tr></table>

![](images/26d78f0dc22c7bff96a16fa3ceb3d7df51dc86036859fb4489ea221b2c61ae20.jpg)  
Figure 6: Visualization of the complexity experts across diferent facial landmark datasets.

Evaluation of diferent numbers of complexity experts and Top-K values. We investigated the efect of varying the number of activated experts. As shown in Table 8, the model achieves the best performance among the evaluated expert configurations with 4 experts at TopK=1. Increasing the number of experts or activating multiple experts per sample does not further improve accuracy and even leads to performance degradation. This suggests that excessive experts introduce redundancy and insuficient expert specialization, while Top-1 selection encourages clear complexity-aware expert assignment, which is more suitable for FLD.

Evaluation on FreqMoE. We analyze the dataset-level routing behavior of the proposed FreqMoE by measuring expert utilization induced by the routing function. As shown in Fig. 6, distinct expert utilization patterns emerge across diferent datasets. Samples from relatively less challenging datasets (e.g., 300W) are predominantly routed to lightweight experts, whereas more complex datasets such as WFLW and COFW exhibit higher activation frequencies of highercapacity experts. These results indicate that FreqMoE adaptively allocates model capacity according to dataset-specific facial complexity. In addition, compared with the variant without FreqMoE (Fig. 7(b)), the model integrating FreqMoE (Fig. 7(c)) achieves more accurate and stable FLD performance, particularly under challenging conditions.

![](images/c65075b651a9319a14237cbc6987c91d40757bb86d3c9b51b24fd6427d466940.jpg)

Figure 7: (a) The original input images, (b) the prediction results of the FreqFLD without the FreqMoE and (c) the prediction results with the FreqMoE integrated. It can be seen that by integrating FreqMoE, the model can learn more efective facial structural information, thereby improving performance.  
![](images/af5a741f35d8dbfc738fcae339a37800b5427cb9f04e2829a175bad9002b93ca.jpg)  
Figure 8: Comparison of t-SNE visualizations of features between the baseline and FreqMoM on latent layer. The results demonstrate that the proposed FreqMoM exhibits clear inter-dataset separation while remaining compact within each dataset.

Evaluation on FreqMoM. We evaluate FreqMoM on the 300W challenging subset. Starting from the Trans+FreqMoE baseline, we separately introduce the HFPB and LFPB. As shown in Table 9, HFPB reduces $\mathrm { N M E } _ { \mathrm { i o } }$ from 4.60 to 4.56, indicating improved modeling of fine-grained details, while LFPB achieves an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.58, reflecting its efectiveness in capturing global facial structure. When both branches are jointly integrated via FreqMoM, the model achieves the best performance with an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.52, highlighting the complementary roles of high- and low-frequency cues. In addition, Fig. 8 presents the t-SNE visualizations of feature representations learned by the baseline and FreqMoM. The baseline features in Fig. 8(a) are highly mixed across diferent datasets, indicating limited discrimination under heterogeneous data distributions. By contrast, FreqMoM produces more compact intra-dataset clusters and clearer inter-dataset separation, as shown in Fig. 8(b). This demonstrates that frequency modulation improves the discriminability of learned representations and helps alleviate feature conflicts in the $_ { \mathrm { A l l - i n - O n e } }$ FLD setting. These results can be attributed to the fact that high-frequency features primarily enhance sensitivity to fine-grained landmark details, while low-frequency features contribute to more stable modeling of global facial geometry, and their combination enables more geometry-sensitive FLD under the All-in-One paradigm.

## 5. CONCLUSION

In the All-in-One FLD task, learning a unified model that generalizes across heterogeneous datasets remains challenging. To address this issue, we present FreqFLD, an All-in-One framework that incorporates FreqMoM and FreqMoE to alleviate feature conflicts arising from heterogeneous facial data. Specifically, FreqMoM introduces explicit frequency modulation to disentangle low- and high-frequency facial cues, achieving balanced modeling of global structure and local details across heterogeneous datasets. Moreover, FreqMoE explicitly models the complexity of facial samples and adaptively routes them to experts with appropriate representational capacity, enabling robust handling of large variations in pose, expression, and appearance across datasets. In addition, the FreqCR loss is introduced to stabilize expert assignment and mitigate expert imbalance, thereby promoting consistent and robust expert specialization under diverse facial scenarios. By jointly leveraging FreqMoM, FreqMoE and FreqCR, the proposed FreqFLD alleviates gradient conflicts in multi-dataset training, achieving comparable performance across multiple face alignment benchmarks. In the future, we will explore more lightweight frequency-aware expert designs to further reduce computational overhead while preserving the robustness of All-in-One FLD.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant NOs. 62476172, 62571555, 62576192 and 62576257), and the Natural Science Foundation of Hubei Province (2024AFB992).

## References

[1] M. Matsugu, K. Mori, Y. Mitari, Y. Kaneda, Subject independent facial expression recognition with robust face detection using a convolutional neural network, Neural networks 16 (5-6) (2003) 555–559.

[2] S.-H. Yoo, S.-K. Oh, W. Pedrycz, Optimized face recognition algorithm using radial basis function neural networks and its practical applications, Neural Networks 69 (2015) 111–125.

[3] S. V. Ioannou, A. T. Raouzaiou, V. A. Tzouvaras, T. P. Mailis, K. C. Karpouzis, S. D. Kollias, Emotion recognition through facial expression analysis based on a neurofuzzy network, Neural Networks 18 (4) (2005) 423–435.

[4] H. Tang, L. Chai, Facial micro-expression recognition using stochastic graph convolutional network and dual transferred learning, Neural Networks 178 (2024) 106421.

[5] Y. Feng, F. Wu, X. Shao, Y. Wang, X. Zhou, Joint 3d face reconstruction and dense alignment with position map regression network, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 534–551.

[6] S. Basak, P. Corcoran, R. McDonnell, M. Schukat, 3d face-model reconstruction from a single image: A feature aggregation approach using hierarchical transformer with weak supervision, Neural Networks 156 (2022) 108–122.

[7] Y. Xu, Z. Yang, Y. Yang, Seeavatar: Photorealistic text-to-3d avatar generation with constrained geometry and appearance, arXiv preprint arXiv:2312.08889 (2023).

[8] X. Chu, Y. Li, A. Zeng, T. Yang, L. Lin, Y. Liu, T. Harada, Gpavatar: Generalizable and precise head avatar from image (s), arXiv preprint arXiv:2401.10215 (2024).

[9] M. Kowalski, J. Naruniec, T. Trzcinski, Deep alignment network: A convolutional neural network for robust face alignment, in: CVPR Workshops, 2017, pp. 88–97. doi:10.1109/CVPRW.2017.18.

[10] J. Wan, J. Li, Z. Lai, B. Du, L. Zhang, Robust face alignment by cascaded regression and de-occlusion, Neural networks : the oficial journal of the International Neural Network Society 123 (2019) 261–272.

[11] J. Wan, J. Liu, J. Zhou, Z. Lai, L. Shen, H. Sun, P. Xiong, W. Min, Precise facial landmark detection by reference heatmap transformer, IEEE Transactions on Image Processing 32 (2023) 1966–1977.

[12] J. Wan, H. Xi, Y. Yao, H. Sun, Z. Lai, J. Zhou, Interpretable facial landmark detection by multi-expert collaborative uncertainty-aware deep networks, Neural networks : the oficial journal of the International Neural Network Society 194 (2025) 108195.

[13] J. Ma, S. Hu, X. Zhang, J. Wan, J. Huang, L. Zhang, S. Khan, Evoir: Towards all-in-one image restoration via evolutionary frequency modulation, arXiv preprint arXiv:2512.05104 (2025).

[14] S. Hu, J. Ma, X. Zhang, Y. Jing, L. Zhang, J. Wan, Clusir: Towards cluster-guided all-in-one image restoration, arXiv preprint arXiv:2512.10948 (2025).

[15] X. Chen, H. Li, J. Dong, J. Pan, X. Li, X. He, N. Chen, S. Li, F. Liu, H. Lv, et al., Lovif 2026 challenge on real-world all-in-one image restoration: Methods and results, arXiv preprint arXiv:2604.19445 (2026).

[16] S. Hu, J. Shao, J. Ma, X. Zhang, K. Wu, Q. Zhu, B. Song, J. Wan, Spikerestormer: Towards energy-eficient all-in-one image restoration via unified event reasoning, arXiv preprint arXiv:2608.02290 (2026).

[17] X. Zhang, H. Zhang, G. Wang, Q. Zhang, L. Zhang, Clearair: A humanvisual-perception-inspired all-in-one image restoration, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2026, pp. 12861–12869.

[18] J. Wan, M. Gan, L. Zhang, J. Zhou, J. Liu, B. Du, C. P. Chen, Fine-grained image captioning by ranking difusion transformer, IEEE Transactions on Image Processing 34 (2025) 8332–8344.

[19] X. Zhang, J. Ma, G. Wang, Q. Zhang, H. Zhang, L. Zhang, Perceive-ir: Learning to perceive degradation better for all-in-one image restoration, IEEE Transactions on Image Processing 35 (2026) 2018–2033.

[20] W. Min, Z. Shi, J. Zhang, J. Wan, C. Wang, Multimodal contrastive learning for spatial gene expression prediction using histology images, Briefings in Bioinformatics 25 (6) (2024) bbae551.

[21] O. Tuzel, T. K. Marks, S. Tambe, Robust face alignment using a mixture of invariant experts, in: European Conference on Computer Vision, Springer, 2016, pp. 825–841.

[22] E. Arnaud, A. Dapogny, K. Bailly, Tree-gated deep mixture-of-experts for pose-robust face alignment, IEEE Transactions on Biometrics, Behavior, and Identity Science 2 (2) (2019) 122–132.

[23] S. Hu, H. Qi, J. Wan, J. Huang, L. Zhang, H. Sun, D. Tao, Proto-former: Unified facial landmark detection by prototype transformer, IEEE Transactions on Multimedia (2026).

[24] X. Zhang, H. Zhang, G. Wang, Q. Zhang, L. Zhang, B. Du, Uniuir: Considering underwater image restoration as an all-in-one learner, IEEE Transactions on Image Processing 34 (2025) 6963–6977.

[25] S. Yan, C. Liu, S. Z. Li, H. Zhang, H.-Y. Shum, Q. Cheng, Face alignment using texture-constrained active shape models, Image and Vision Computing 21 (1) (2003) 69–75.

[26] Z. Zheng, J. Jiong, D. Chunjiang, X. Liu, J. Yang, Facial feature localization based on an improved active shape model, Information Sciences 178 (9) (2008) 2215–2223.

[27] D. Cristinacce, T. F. Cootes, et al., Feature detection and tracking with constrained local models., in: Bmvc, Vol. 1, Edinburgh, 2006, p. 3.

[28] D. Cristinacce, T. Cootes, Automatic feature localisation with constrained local models, Pattern Recognition 41 (10) (2008) 3054–3067.

[29] V. Kazemi, J. Sullivan, One millisecond face alignment with an ensemble of regression trees, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 1867–1874.

[30] Z.-H. Feng, P. Huber, J. Kittler, W. Christmas, X.-J. Wu, Random cascadedregression copse for robust facial landmark detection, IEEE Signal Processing Letters 22 (1) (2014) 76–80.

[31] J. Wan, Z. Lai, J. Li, J. Zhou, C. Gao, Robust facial landmark detection by multiorder multiconstraint deep networks, IEEE Transactions on Neural Networks and Learning Systems 33 (5) (2021) 2181–2194.

[32] J. Wan, H. Liu, Y. Wu, Z. Lai, W. Min, J. Liu, Precise facial landmark detection by dynamic semantic aggregation transformer, Pattern Recognition 156 (2024) 110827.

[33] J. Wan, X. Xiong, N. Chen, Z. Lai, J. Zhou, W. Min, Fgtbt: Frequencyguided task-balancing transformer for unified facial landmark detection, Information Sciences (2026) 123130.

[34] J. Wan, Y. Yao, J. Huang, X. Ding, L. Zhang, Y. Gao, D. Tao, Universal facial landmark detection by landmark-clustering relation-reasoning transformer, International Journal of Computer Vision 134 (6) (2026) 310.

[35] W. Wu, S. Yang, Leveraging intra and inter-dataset variations for robust face alignment, in: Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2017, pp. 150–159.

[36] Z.-H. Feng, J. Kittler, M. Awais, P. Huber, X.-J. Wu, Wing loss for robust facial landmark localisation with convolutional neural networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2235–2245.

[37] P. Gao, K. Lu, J. Xue, L. Shao, J. Lyu, A coarse-to-fine facial landmark detection method based on self-attention mechanism, IEEE Transactions on Multimedia 23 (2020) 926–938.

[38] C. Lin, B. Zhu, Q. Wang, R. Liao, C. Qian, J. Lu, J. Zhou, Structurecoherent deep feature learning for robust face alignment, IEEE Transactions on Image Processing 30 (2021) 5313–5326.

[39] J. Xia, W. Qu, W. Huang, J. Zhang, X. Wang, M. Xu, Sparse local patch transformer for robust face alignment and landmarks inherent relation learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4052–4061.

[40] J. Yang, Q. Liu, K. Zhang, Stacked hourglass network for robust facial landmark localisation, in: Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2017, pp. 79–87.

[41] X. Zou, S. Zhong, L. Yan, X. Zhao, J. Zhou, Y. Wu, Learning robust facial landmark detection via hierarchical structured ensemble, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 141–150.

[42] C. Zhu, X. Wan, S. Xie, X. Li, Y. Gu, Occlusion-robust face alignment using a viewpoint-invariant hierarchical network architecture, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11112–11121.

[43] K. Yin, V. Rao, R. Jiang, X. Liu, P. Aarabi, D. B. Lindell, Sce-mae: Selective correspondence enhancement with masked autoencoder for self-supervised landmark estimation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1313–1322.

[44] Z. Dang, J. Li, L. Liu, Cascaded dual vision transformer for accurate facial landmark detection, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2025, pp. 5884–5894.

[45] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).

[46] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean, The sparsely-gated mixture-of-experts layer, Outrageously large neural networks 2 (2017).

[47] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, N. Houlsby, Scaling vision with sparse mixture of experts, Advances in Neural Information Processing Systems 34 (2021) 8583–8595.

[48] J. Puigcerver, C. Riquelme, B. Mustafa, N. Houlsby, From sparse to soft mixtures of experts, arXiv preprint arXiv:2308.00951 (2023).

[49] H. Park, D. Kim, A complementary regression network for accurate face alignment, Image and Vision Computing 95 (2020) 103883.

[50] C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic, 300 faces in-the-wild challenge: Database and results, Image and vision computing 47 (2016) 3–18.

[51] X. P. Burgos-Artizzu, P. Perona, P. Doll´ar, Robust face landmark estimation under occlusion, in: Proceedings of the IEEE international conference on computer vision, 2013, pp. 1513–1520.

[52] W. Wu, C. Qian, S. Yang, Q. Wang, Y. Cai, Q. Zhou, Look at boundary: A boundary-aware face alignment algorithm, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2129–2138.

[53] S. Zhu, C. Li, C.-C. Loy, X. Tang, Unconstrained face alignment via cascaded compositional learning, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 3409–3417.

[54] P. N. Belhumeur, D. W. Jacobs, D. J. Kriegman, N. Kumar, Localizing parts of faces using a consensus of exemplars, IEEE transactions on pattern analysis and machine intelligence 35 (12) (2013) 2930–2940.

[55] V. Le, J. Brandt, Z. Lin, L. Bourdev, T. S. Huang, Interactive facial feature localization, in: European conference on computer vision, Springer, 2012, pp. 679–692.

[56] G. Trigeorgis, P. Snape, M. A. Nicolaou, E. Antonakos, S. Zafeiriou, Mnemonic descent method: A recurrent process applied for end-to-end face alignment, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4177–4187.

[57] A. Bulat, G. Tzimiropoulos, How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks), in: Proceedings of the IEEE international conference on computer vision, 2017, pp. 1021–1030.

[58] M. Zhu, D. Shi, M. Zheng, M. Sadiq, Robust facial landmark detection via occlusion-adaptive deep networks, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 3486–3496.

[59] X. Wang, L. Bo, L. Fuxin, Adaptive wing loss for robust face alignment via heatmap regression, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6971–6981.

[60] A. Kumar, T. K. Marks, W. Mou, Y. Wang, M. Jones, A. Cherian, T. Koike-Akino, X. Liu, C. Feng, Luvli face alignment: Estimating landmarks’ location, uncertainty, and visibility likelihood, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8236–8246.

[61] C. Zhu, X. Li, J. Li, S. Dai, Improving robustness of facial landmark detection by defending against adversarial attacks, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11751– 11760.

[62] T. Wen, Z. Ding, Y. Yao, Y. Wang, X. Qian, Picassonet: Searching adaptive architecture for eficient facial landmark localization, IEEE Transactions on Neural Networks and Learning Systems 34 (12) (2022) 10516–10527.

[63] J. Liang, H. Liu, H. Xu, D. Luo, Generalizable face landmarking guided by conditional face warping, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2425–2435.

[64] R. Valle, J. M. Buenaposada, A. Valdes, L. Baumela, A deeply-initialized coarse-to-fine ensemble of regression trees for face alignment, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 585– 601.

[65] J. Wan, Z. Lai, J. Liu, J. Zhou, C. Gao, Robust face alignment by multi-order high-precision hourglass network, IEEE Transactions on Image Processing 30 (2020) 121–133.

[66] Y. Huang, H. Yang, C. Li, J. Kim, F. Wei, Adnet: Leveraging errorbias towards normal direction in face alignment, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3080– 3090.

[67] J. Xia, M. Xu, H. Zhang, J. Zhang, W. Huang, H. Cao, S. Wen, Robust face alignment via inherent relation learning and uncertainty estimation, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (8) (2023) 10358–10375.

[68] Y. Li, G. Tan, C. Gou, Cascaded iterative transformer for jointly predicting facial landmark, occlusion probability and head pose, International Journal of Computer Vision 132 (4) (2024) 1242–1257.

[69] X. Cao, Y. Wei, F. Wen, J. Sun, Face alignment by explicit shape regression, uS Patent App. 13/728,584 (Jul. 3 2014).

[70] X. Xiong, F. De la Torre, Supervised descent method and its applications to face alignment, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2013, pp. 532–539.

[71] S. Zhu, C. Li, C. Change Loy, X. Tang, Face alignment by coarse-to-fine shape searching, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 4998–5006.

[72] A. Dapogny, K. Bailly, M. Cord, Decafa: Deep convolutional cascade for face alignment in the wild, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6893–6901.

[73] J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Wang, et al., Deep high-resolution representation learning for visual recognition, IEEE transactions on pattern analysis and machine intelligence 43 (10) (2020) 3349–3364.

[74] P. Gao, K. Lu, J. Xue, J. Lyu, L. Shao, A facial landmark detection method based on deep knowledge transfer, IEEE Transactions on Neural Networks and Learning Systems 34 (3) (2021) 1342–1353.

[75] Z.-H. Feng, J. Kittler, W. Christmas, P. Huber, X.-J. Wu, Dynamic attention-controlled cascaded shape regression exploiting training data

augmentation and fuzzy-set sample weighting, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2481–2490.

[76] X. Dong, Y. Yan, W. Ouyang, Y. Yang, Style aggregated network for facial landmark detection, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 379–388.

[77] J. P. Robinson, Y. Li, N. Zhang, Y. Fu, S. Tulyakov, Laplace landmark localization, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 10103–10112.

[78] H. Jin, S. Liao, L. Shao, Pixel-in-pixel net: Towards eficient facial landmark detection in the wild, International Journal of Computer Vision 129 (12) (2021) 3174–3194.

[79] J. Yu, J. Bae, H. Y. Kim, Helpnet: Facial structure-aware landmark coordinate regression with heatmap-guided local patch embedding, Neurocomputing 650 (2025) 130860.