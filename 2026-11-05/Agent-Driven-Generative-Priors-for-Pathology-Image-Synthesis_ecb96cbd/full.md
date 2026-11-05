# Agent-Driven Generative Priors for Pathology Image Synthesis and Segmentation

Sicheng Yang, Rui Tang, Haipeng Zhou, Hongqiu Wang, Zhaohu Xing, Jianda Ma, Lie Dai, Guang Yang, and Lei Zhu

Abstract— Deep learning methods for pathology are constrained by the limited availability and high cost of largescale pixel-level annotations. Synthetic data can alleviate this limitation, but existing pipelines often lack reliable quality control and remain weakly connected to downstream analysis. To address these issues, we propose a three-stage framework that integrates automated data screening, generative representation learning, and efficient adaptation to downstream tasks. First, we develop a pathology agent system to assess image quality and filter samples, resulting in TCGA-200K, a dataset containing 217,030 pathology patches selected from the public TCGA repository. Second, we train a generative model on the screened data. In addition to image synthesis, its backbone learns structural and semantic representations that can be transferred to downstream tasks. Third, we adapt the pretrained generative backbone to semantic segmentation using a lightweight task-specific head. We further introduce PathSeg-380K, a semantic segmentation dataset with expert annotations for approximately 380,000 cells across 12 cell types. Experiments on multiple pathology datasets show that the proposed framework achieves state-of-the-art segmentation performance. Our code is publicly available at https://github.com/script-Yang/Agent-GenPath.

Index Terms— Agent systems, pathology image synthesis, image segmentation

## I. INTRODUCTION

Istopathological examination is a cornerstone of clinical diagnosis and disease assessment [1]. Advances   
in deep learning have further expanded the capabilities of   
computational pathology, supporting applications ranging from   
nuclei segmentation [2] and tumor detection [3] to tissue   
classification [4]. However, these models often rely on large  
scale expert annotations [5]–[7]. Obtaining such annotations   
is costly and time-consuming, as accurate labeling requires

![](images/ca031895653e7c1196bfdf91961f5937ad2d51e2ecea5f1ae38a2fda6109912a.jpg)  
(a) Datasets for Generative Prior

![](images/a05ce7f1d4f90a8a097d3c491f6738747b8b059c2af695834173cd551ad3b031.jpg)  
(b) Datasets for Downstream task  
Fig. 1. Overview of the datasets and data distributions. (a) Source composition of TCGA-200K, utilized for pre-training the generative prior. (b) Class distributions for downstream segmentation tasks, comparing the publicly available PanNuke dataset with our large-scale, annotated PathSeg-380K dataset, containing ∼ 380,000 annotated cells.

pathologists to manually identify and delineate complex cellular structures [8]. As a result, sufficiently annotated datasets remain scarce in many clinical settings [9], increasing the risk of overfitting and limiting model generalization.

Synthetic data generation has emerged as a promising strategy to augment scarce labeled data [10]. Generative models, including generative adversarial networks (GANs) [11] and diffusion models [12], have demonstrated strong capability in synthesizing realistic pathology images. Typically, these generated images are combined with the existing datasets to augment the training pool, thereby improving the performance of downstream tasks such as segmentation [13]. However, it still faces two major limitations. (1) Existing pipelines lack effective data quality control. In the pathology area, images frequently suffer from staining artifacts, blurred structures, and poor cellular visibility [14], [15]. For example, within a single whole slide image (WSI), many regions are often uninformative for diagnosis. Training generative models on such low-quality data without quality assessment leads to unreliable synthetic images with inaccurate pathological patterns [16]. (2) The generation process and downstream tasks are disconnected from each other. Existing methods treat generative models merely as black-box data engines to sample new images. Therefore, they fail to leverage the rich semantic information learned during the generation process [17]. Our experiments reveal that the generative model backbone itself can serve as a powerful feature extractor for downstream tasks, providing rich prior information.

![](images/74400118d9f9f82b0918fe71591a067208c36b207b227105b6f86d76280a4d87.jpg)  
Fig. 2. Overview of the proposed unified three-stage framework. In Stage I, an agent-driven screening pipeline evaluates pathology patches from multiple quality dimensions and selects high-quality samples to construct the TCGA-200K dataset. In Stage II, the curated data are used to learn pathology-specific generative priors through DiT blocks. In Stage III, the learned generative priors are transferred to downstream segmentation on PanNuke and PathSeg-380K, where the pretrained backbone provides multi-level pathology representations for accurate pathology segmentation.

To address these limitations, we propose a novel threestage framework that unifies automated data quality control, the learning of high-quality generative priors, and transfer learning for downstream tasks. First, we introduce an agentbased screening mechanism for dataset refinement. We design an intelligent pathology agent system to automatically evaluate and score the quality of pathology images across multiple clinical dimensions. Only high-scoring images are retained to construct a curated training set for learning the generative prior. Second, we train a generative model on this refined dataset. Crucially, this model is not deployed merely to synthesize new data; instead, its backbone naturally extracts rich structural and semantic information during the generative training process, allowing it to serve as a powerful foundational encoder. Third, we redefine downstream analysis as a transfer learning paradigm. By simply appending a lightweight task-specific head (e.g., for semantic segmentation) to the pretrained generative backbone, our framework leverages these learned priors with minimal training overhead. Our contributions are summarized as follows:

• A Unified Three-Stage Framework. We identify that current pathology analysis pipelines suffer from inadequate data quality control and a disconnect between generation and downstream tasks. To address this, we propose a novel framework that unifies agent-guided quality screening, robust generative prior learning, and efficient downstream transfer learning.

• Two Novel High-Quality Datasets. We construct and introduce two large-scale datasets to the community: TCGA-200K, comprising 217,030 premium pathology patches curated from the open-source TCGA repository using our agent-based screening mechanism. We also use PathSeg-380K for semantic segmentation. It contains annotations provided by two clinical pathologists for approximately 380,000 cells across 12 cell types.

• State-of-the-Art Performance. Extensive experiments show that our framework generates high-quality pathology images. When the pretrained generative backbone is transferred to semantic segmentation, the resulting model also outperforms existing segmentation methods across both public and private datasets, indicating that the learned generative prior benefits both image synthesis and downstream pathology analysis.

## A. Synthetic Data for Pathology

Acquiring massive and fully annotated pathology datasets is frequently constrained by privacy concerns and high costs [18]. To address this data scarcity, researchers have actively explored synthetic data generation [19]. While models such as Generative Adversarial Networks (GANs) and diffusion models have achieved notable progress in producing visually realistic images [20], [21], a critical limitation persists. Current approaches largely neglect the rigorous filtering of data sources [22], [23]. Consequently, incorporating training images of poor quality inevitably introduces erroneous information into the generative models. As a result, the synthesized images may fail to preserve key pathological structures, reduc ing their usefulness for downstream medical tasks.

## B. Agents in Pathology

The reasoning capabilities of AI agents have opened new frontiers in medical image analysis [24]. These agents have shown remarkable potential in automating complex diagnostic workflows, including pathological image interpretation, report generation, and interactive clinical decision-making [25]. By simulating the reasoning processes of human experts, agents can analyze intricate tissue morphologies and cellular patterns. Despite their growing success in diagnostic tasks, the ability of agents to guide pathological image generation remains largely unexplored [26]. Specifically, they can act as virtual experts, providing feedback on image quality and structural validity, which offers a promising avenue for enhancing the reliability of synthetic medical data. Motivated by this gap, we introduce a novel framework that leverages agentic reasoning to screen and guide pathological image synthesis.

![](images/0c18ec41ab0b1a54ec603d2452fb3ab0db23cf6d65206a9354b3e2e40884e939.jpg)  
Fig. 3. Stage I: Agent-driven high-quality pathology image selection. Given an input pathology patch $x _ { i } , A _ { P }$ initializes the memory state $^ { m _ { i } , }$ $\mathbf { \nabla } A _ { R }$ evaluates cell presence, stain quality, and histologic clarity, and $\pmb { A } _ { V }$ performs closed-loop verification to produce the final quality scores.

## C. Pathology Image Segmentation

Semantic segmentation is widely used in computational pathology to delineate cellular structures and tissue regions [27]–[30]. Some methods train segmentation networks from scratch on task-specific datasets, but limited pixel-level annotations often lead to overfitting [31]. Others adopt backbones pretrained on natural images, such as ResNet [32] and DINO [33]–[36], whose representations do not fully capture pathology-specific morphology [37]. Pathology foundation models provide stronger domain-specific features [38], [39], yet they are designed as encoders and cannot support image generation. We instead pretrain a generative model on agentcurated pathology images and transfer its multi-level features to semantic segmentation, allowing the same generative prior to serve both image synthesis and downstream analysis.

## II. METHODOLOGY

## A. Overview

As illustrated in Figure 2, we propose a unified three-stage framework that tightly integrates automated data quality control, generative prior learning, and downstream segmentation. First, we introduce an agent-driven screening mechanism to evaluate and filter raw pathology patches, yielding a highly curated, large-scale dataset (TCGA-200K). Second, we leverage this refined dataset to train a Diffusion Transformer (DiT) via Flow Matching, enabling the backbone to learn robust structural and semantic generative priors rather than serving merely as a data synthesizer. Finally, we reformulate downstream semantic segmentation as a transfer learning paradigm, where a lightweight multi-level decoder is appended to the pretrained DiT backbone to map the extracted multi-scale features to precise pixel-level predictions. The effectiveness and generalization ability of this unified pipeline are comprehensively validated on both the public PanNuke dataset and our newly proposed PathSeg-380K dataset.

## B. Stage I: Agent-Driven Pathology Data Curation

Current generative modeling in pathology is constrained by two major challenges: the limited scale of existing datasets, which often contain only a few thousand images, and the lack of rigorous quality control. As a result, low-quality samples, such as patches with sparse cellular content or out-of-focus blur, can substantially degrade model performance. To address this, we propose an agent-driven and scalable quality curation pipeline with a semi-supervised design. Specifically, we first construct a large-scale pathology dataset, denoted as $\mathcal { D } ,$ from multiple public TCGA cohorts [40], and partition it into a small labeled pilot set $\mathcal { D } _ { P }$ and a large unlabeled pool $\mathcal { D } _ { U }$ . A Multi-dimensional Assessment Pathology Expert Agent is then employed to annotate $\mathcal { D } _ { P }$ across three key quality dimensions, providing supervision for training a lightweight regressor. The trained regressor is subsequently used to score the much larger unlabeled pool $\mathcal { D } _ { U }$ , enabling scalable quality estimation in a semi-supervised manner. Finally, a multi-dimensional quantile filtering strategy is applied to extract a high-fidelity subset $\mathcal { D } ^ { * }$ resulting in TCGA-200K, a curated dataset of over 200,000 high-quality pathology patches for generative modeling.

![](images/032d4a3a25571ee2d551e26ecbd12320fec428f72e52e1a67f5e5848ee118310.jpg)  
Fig. 4. Score distributions before and after the integration of our pathology expert agent across three quality dimensions. The distributions shift toward higher scores after agent-driven filtering, indicating an improvement in overall data quality.

1) Large-Scale Pathology Dataset Construction: As illustrated in Fig. 1 (a), we source a diverse set of Whole Slide Images (WSIs) across five cancer types from TCGA: Breast Invasive Carcinoma (BRCA), Cervical Squamous Cell Carcinoma (CESC), Colon Adenocarcinoma (COAD), Kidney Renal Clear Cell Carcinoma (KIRC), and Liver Hepatocellular Carcinoma (LIHC). We employ the CLAM toolkit [41] to process these WSIs into a comprehensive dataset D containing approximately 14 million 256 × 256 patches. While this preprocessing effectively removes the majority of the white background regions, the retained patches inevitably exhibit localized artifacts (e.g., tissue folding, out-of-focus blurring, and uneven staining) that can destabilize training dynamics. To facilitate our agent-driven quality curation, we randomly partition the full dataset D into two disjoint subsets: a small pilot set $\mathcal { D } _ { P }$ and a massive unannotated target pool $\mathcal { D } _ { U }$

2) PathologyExpertAgentforQualitySupervision: To obtain reliable quality pseudo-labels without exhaustive manual annotation, we design a Multi-dimensional Assessment Pathology Expert Agent. The agent follows a structured Perception–

Layer11  
![](images/ebe983b5ed90d97a805234a7d85bd90188893cad0b9583c16ee20dcf466d0033.jpg)  
Fig. 5. Comparison of synthetic (left) and real (right) pathology images. The generated samples faithfully reproduce complex tissue morphology, cellular structures, and spatial organization, showing strong visual consistency with real clinical images.

![](images/7a8e00d13bf087507a02f77625a11daae24562b3b439a6cb120ce6515d16abd4.jpg)  
Fig. 6. Stage II: Learning pathology generative priors via Flow Matching. Given an image patch and timestep t, the DiT generator G is trained to predict the target flow field.

Reasoning–Verification workflow to evaluate the visual quality and pathological content of each patch in the pilot set $\mathcal { D } _ { P }$

As shown in Fig. 3, for each image $x _ { i } \in \mathcal { D } _ { P }$ , the agent’s perception module $A _ { P }$ first conducts a global inspection to identify tissue types and prominent artifacts (e.g., folding or blurring). This preliminary step initializes a contextual memory state $m _ { i } = \mathcal { A } _ { P } ( x _ { i } )$ that guides subsequent decisionmaking. Conditioned on $m _ { i }$ , the reasoning module $\boldsymbol { \mathcal { A } } _ { R }$ then autonomously analyzes three pathologically critical dimensions: cell presence (C), stain quality (S), and histologic clarity (H). Instead of directly outputting ratings, the agent is mandated to explicitly generate a diagnostic rationale $r _ { i } ^ { d }$ before taking a preliminary scoring action $\tilde { s } _ { i } ^ { d } \in \{ 1 , 2 , 3 , 4 , 5 \}$ for each dimension $d \in \{ C , S , H \}$

$$
( r _ { i } ^ { d } , \tilde { s } _ { i } ^ { d } ) = \mathcal { A } _ { R } ( x _ { i } , m _ { i } , d ) .\tag{1}
$$

To ensure the robustness of the supervision, the agent enters a self-verification stage via the verification module $\mathbf { \mathcal { A } } _ { V }$ to examine internal consistency across dimensions. For example, high cellularity with low clarity may indicate a poorly focused region, prompting the agent to revisit its initial assessment. The verification module then jointly considers the image, preliminary rationales, and actions to resolve inconsistencies and produce the final quality score vector:

Image

$$
\mathbf { s } _ { i } = \mathcal { A } _ { V } \left( x _ { i } , \{ ( r _ { i } ^ { d } , \tilde { s } _ { i } ^ { d } ) \} _ { d \in \{ C , S , H \} } \right) ,\tag{2}
$$

Layer2  
Layer5  
Layer8  
![](images/cb257bb2bc90520c628560dacca20ff594861e5424d1eddc6dae8d246bcc7485.jpg)  
Fig. 7. Visualizing the refinement of feature activations in the DiT generative prior. Moving from shallow to deep layers, the model sequentially learns to disentangle semantic structures at various scales, such as individual cells, glandular tissues, and the surrounding background, which provides a crucial foundation for robust semantic segmentation.

where $\mathbf { s } _ { i } = [ s _ { i } ^ { C } , s _ { i } ^ { S } , s _ { i } ^ { H } ] ^ { \top }$ . This closed-loop mechanism reinforces reliability, yielding an expert-supervised pilot set $\mathcal { D } _ { P } =$ $\{ ( x _ { i } , { \bf s } _ { i } ) \} _ { i = 1 } ^ { N }$ , where N is the number of sampled images.

3) Learning the Quality Regressor: Applying our agent directly to the massive pool $\mathcal { D } _ { U }$ incurs prohibitive computational overhead. To overcome this, we adopt a semi-supervised paradigm, training a lightweight convolutional neural network (i.e., ResNet-50 [32]) as a quality regressor, denoted as $f _ { \theta } ,$ to approximate the expert agent.

Using the labeled pilot set $\mathcal { D } _ { P }$ , the regressor maps a raw input image $x _ { i }$ to continuous predicted scores $\hat { \mathbf { s } } _ { i } = f _ { \theta } ( x _ { i } )$ The model is optimized to minimize the discrepancy between its predictions and the agent’s scores using the Mean Squared Error (MSE) objective:

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. f _ { \theta } ( x _ { i } ) - \mathbf { s } _ { i } \right. _ { 2 } ^ { 2 } .\tag{3}
$$

Once optimized, $f _ { \theta }$ can be deployed for real-time quality estimation across the entire target pool $\mathcal { D } _ { U }$ , preserving expertlevel accuracy while bypassing the Pathology Expert Agent’s inference latency.

![](images/7af8c991030786bf0a9bb3e1e1ea88380e88997c03127f8171fb521a25c2d0db.jpg)  
Fig. 8. Stage III: Transferring generative priors to downstream segmentation. Multi-level features are extracted from the pathology-pretrained DiT backbone, transformed and fused into a unified representation, and then fed into the segmentation head.

4) Scoring and Quantile-based Selection: We deploy the trained regressor $f _ { \theta }$ to evaluate all images in the unannotated pool $\mathcal { D } _ { U }$ . For each $x _ { j } \in { \mathcal { D } } _ { U }$ , we obtain the predicted multidimensional scores:

$$
\begin{array} { r } { \hat { \bf s } _ { j } = [ \hat { s } _ { j } ^ { C } , \hat { s } _ { j } ^ { S } , \hat { s } _ { j } ^ { H } ] ^ { \top } . } \end{array}\tag{4}
$$

To establish a unified ranking criterion, we compute the empirical score distributions for each quality dimension across the entire target pool $\mathcal { D } _ { U }$ . Rather than relying on rigid absolute thresholds, which can be sensitive to domain shifts, we implement a quantile-based selection strategy. Specifically, let $\tau _ { d }$ denote the score threshold corresponding to the top k% of the distribution for dimension $d \in \{ C , S , H \}$ . A patch is selected if and only if its predicted scores all satisfy this top k% criterion across all three critical dimensions. Formally, the filtered subset $\mathcal { D } ^ { * }$ is defined as:

$$
\mathcal { D } ^ { * } = \left\{ x _ { j } \in \mathcal { D } _ { U } \vert \hat { s } _ { j } ^ { C } \geq \tau _ { C } \wedge \hat { s } _ { j } ^ { S } \geq \tau _ { S } \wedge \hat { s } _ { j } ^ { H } \geq \tau _ { H } \right\} .\tag{5}
$$

This strict filtering ensures that no single severe defect (e.g., severe out-of-focus blur, despite adequate cellularity) compromises the overall utility of the image. Following this process, we distill the massive unannotated pool down to a highly refined subset comprising 217,030 premium-quality patches, which we designate as TCGA-200K. Quantitative analysis reveals a substantial enhancement in data quality: compared to the unfiltered pool $\mathcal { D } _ { U }$ , as shown in Fig. 4, the average predicted scores in TCGA-200K improved by approximately 48.7% (3.065 → 4.558), 22.4% (3.631 → 4.446), and 21.4% (3.607 → 4.379) across the cell presence, histologic clarity, and stain quality dimensions, respectively.

## C. Stage II: Pathology Generative Priors Pretraining

Using the high-quality TCGA-200K dataset curated in Stage I, Stage II focuses on learning a strong pathology generative prior. We adopt a Diffusion Transformer (DiT) architecture [49] optimized via a flow matching objective [50]. As shown in Fig. 6, let $x _ { 0 }$ denote the image from TCGA-200K, and $x _ { 1 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ denote the Gaussian noise. We define an interpolation path to formulate the intermediate state $x _ { t }$ at timestep $t \in [ 0 , 1 ]$

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } .\tag{6}
$$

The corresponding target vector field driving this transformation is defined as $v = x _ { 1 } – x _ { 0 }$ . The DiT network, parameterized by ϕ, is optimized to predict this vector field $\hat { v } _ { \phi } ( x _ { t } , t )$ by minimizing the Mean Squared Error (MSE) loss:

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { x _ { 0 } , x _ { 1 } , t } [ \| \hat { v } _ { \phi } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \| _ { 2 } ^ { 2 } ] .\tag{7}
$$

Throughout the network, the timestep t is embedded and injected into the Transformer blocks via Adaptive Layer Normalization (AdaLN [49]).

Once trained, the model generates a pathology image by sampling $\begin{array} { r } { x _ { 1 } \ \sim \ { \mathcal { N } } ( 0 , \mathbf { I } ) } \end{array}$ and solving the learned ordinary differential equation backward from t = 1 to t = 0:

$$
\frac { \mathrm { d } x _ { t } } { \mathrm { d } t } = \hat { v } _ { \phi } ( x _ { t } , t ) , \qquad \widetilde { x } _ { 0 } = x _ { 1 } - \int _ { 0 } ^ { 1 } \hat { v } _ { \phi } ( x _ { t } , t ) \mathrm { d } t .\tag{8}
$$

By training on the patches selected in Stage I, the model learns the distribution of informative pathology regions. As shown in Fig. 5, the generated samples closely resemble real pathology images in both cellular morphology and tissue organization, a result further supported by the quantitative and expert evaluations presented in later experiments.

## D. Stage III: Generative Features for Segmentation

In Stage III, we transfer the generative representations learned by the DiT to pathology semantic segmentation. Related studies have shown that intermediate features from pretrained diffusion models can support dense prediction [51], [52]. Different from these methods, our backbone is pretrained on the agent-curated pathology dataset and therefore captures pathology-specific cellular morphology and organization.

As shown in Fig. 8, given an input image x, we first construct an intermediate state using the same interpolation path as in Stage II:

$$
x _ { t } = ( 1 - t ) x + t \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{9}
$$

and feed $( x _ { t } , t )$ into the pretrained DiT $v _ { \phi } .$ . The timestep t determines the corruption level of the input and consequently affects the balance between local appearance and high-level semantic information in the extracted features. We treat t as a hyperparameter and study its effect in Sec. III-F.2.

Inspired by the multi-level feature extraction strategy of DPT [53], we extract representations from DiT blocks 2, 5, 8, and 11, which span different depths of the backbone. Since the DiT operates on flattened token sequences, these representations are rearranged into 2D feature maps, denoted as $\{ F _ { k } \} _ { k = 1 } ^ { 4 }$ . A lightweight multi-level decoder then projects them into hierarchical spatial resolutions. Specifically, each $F _ { k }$ is processed by a scale-specific projection $\phi _ { k } ( \cdot )$ consisting of channel reduction and strided convolutions. The projected features are resized to a common spatial resolution by bilinear interpolation R(·) and refined by a convolutional block $\mathcal { C } _ { k } ( \cdot ) \colon$

$$
\widetilde F _ { k } = \mathcal C _ { k } \left( \mathrm { R } \left( \phi _ { k } ( F _ { k } ) \right) \right) , \qquad k = 1 , \dots , 4 .\tag{10}
$$

TABLE I  
QUANTITATIVE COMPARISON ON PANNUKE AND PATHSEG-380K. <sup>\*</sup> DENOTES A STATISTICALLY SIGNIFICANT IMPROVEMENT OVER THE CORRESPONDING SECOND-BEST RESULT $\mathbf { \left( \right)} p < \mathbf { 0 . 0 5 } $
<table><tr><td rowspan="2">Method</td><td colspan="4">PanNuke</td><td colspan="4">PathSeg-380K</td></tr><tr><td>mIOU (%)↑ mDICE (%)↑ p-Acc (%)↑</td><td></td><td></td><td>mASD↓</td><td>|mIOU (%)↑ mDICE (%)↑ p-Acc (%)↑</td><td></td><td></td><td>mASD↓</td></tr><tr><td colspan="9">Task-specific Segmentation Methods</td></tr><tr><td>U-Net [42]</td><td> $4 8 . 0 9 _ { \pm 0 . 2 5 }$ </td><td> $6 1 . 4 4 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $9 1 . 5 6 _ { \pm 0 . 1 4 }$ </td><td> $2 9 . 7 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $4 8 . 6 2 _ { \pm 0 . 9 4 }$ </td><td> $6 0 . 8 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $9 2 . 6 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $6 . 4 0 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>TransUNet [43]</td><td> $4 6 . 6 3 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $5 8 . 2 3 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $9 1 . 8 8 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $2 3 . 0 5 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $4 6 . 6 8 _ { \pm 0 . 9 6 }$ </td><td> $5 9 . 2 7 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $9 2 . 2 9 _ { \pm 0 . 2 1 }$ </td><td> $1 0 . 3 3 { \scriptstyle \pm 0 . 1 9 }$ </td></tr><tr><td>HoVer-Net [44]</td><td> $5 1 . 3 8 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $6 4 . 8 3 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $9 1 . 7 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 2 . 4 2 _ { \pm 0 . 9 3 }$ </td><td> $5 2 . 4 1 _ { \pm 0 . 7 5 }$ </td><td> $6 5 . 7 3 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $9 2 . 5 6 _ { \pm 0 . 1 6 }$ </td><td> $7 . 8 9 _ { \pm 0 . 1 3 }$ </td></tr><tr><td>CPP-Net [45]</td><td> $4 7 . 5 8 _ { \pm 0 . 1 7 }$ </td><td> $6 0 . 8 2 _ { \pm 0 . 2 1 }$ </td><td> $9 1 . 5 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $2 9 . 4 5 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $5 0 . 1 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $6 2 . 2 1 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $9 2 . 7 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $7 . 5 8 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>CellViT [46] Un-SAM [47]</td><td> $5 1 . 5 4 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $6 5 . 1 3 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $9 2 . 1 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $2 6 . 6 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $4 7 . 2 2 _ { \pm 0 . 8 3 }$ </td><td> $6 1 . 2 6 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $9 1 . 2 3 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 . 5 1 { \scriptstyle \pm 0 . 3 3 }$ </td></tr><tr><td></td><td> $5 2 . 3 1 _ { \pm 0 . 6 8 }$ </td><td> $6 5 . 8 8 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $9 2 . 2 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $2 6 . 4 7 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 2 . 8 5 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $6 6 . 3 3 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $9 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $5 . 7 8 _ { \pm 0 . 2 0 }$ </td></tr><tr><td colspan="9">Pretrained Representation-based  $M e t h o d s$ </td></tr><tr><td>SegDINO [34]</td><td> $5 1 . 6 2 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $6 5 . 1 7 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $9 1 . 6 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $\mathbf { 2 1 . 7 5 { \scriptstyle \pm 0 . 6 1 } }$ </td><td> $4 9 . 4 3 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $6 1 . 9 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 2 . 7 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 2 . 8 0 { \scriptstyle \pm 0 . 1 8 }$ </td></tr><tr><td>MAE [48]</td><td> $5 0 . 5 4 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $6 3 . 8 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $9 1 . 4 2 _ { \pm 0 . 1 5 }$ </td><td> $2 5 . 3 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 0 . 2 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $6 3 . 3 4 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $9 1 . 3 6 _ { \pm 0 . 1 5 }$ </td><td> $7 . 9 5 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td>CONCH [38]</td><td> $5 3 . 3 9 _ { \pm 0 . 2 6 }$ </td><td> $6 6 . 7 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $9 1 . 8 9 _ { \pm 0 . 2 0 }$ </td><td> $2 2 . 4 5 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $5 3 . 7 4 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $6 6 . 3 2 _ { \pm 0 . 7 2 }$ </td><td> $9 1 . 7 7 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 . 7 2 _ { \pm 0 . 1 6 }$ </td></tr><tr><td>UNI [39]</td><td> $5 3 . 7 7 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $6 6 . 7 0 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $9 2 . 2 5 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 2 . 2 6 _ { \pm 0 . 4 8 }$ </td><td> $5 4 . 5 9 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $6 5 . 1 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $9 2 . 4 9 _ { \pm 0 . 1 2 }$ </td><td> $5 . 3 7 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Ours</td><td> $\mathbf { 5 5 . 4 6 } _ { \pm 0 . 3 4 } ^ { \ast }$ </td><td> ${ \bf 6 8 . 8 1 _ { \pm 0 . 3 9 } ^ { \ast } }$ </td><td> $\mathbf { 9 2 . 7 7 _ { \pm 0 . 1 4 } ^ { \ast } }$ </td><td> $2 2 . 9 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td> ${ \bf 5 5 . 4 5 { \scriptstyle \pm 0 . 7 1 } }$ </td><td> $\mathbf { 6 7 . 8 1 } _ { \pm 0 . 5 2 }$ </td><td> $\mathbf { 9 3 . 4 4 } _ { \pm 0 . 1 6 } ^ { \ast }$ </td><td> ${ \bf 4 . 8 1 _ { \pm 0 . 1 1 } ^ { * } }$ </td></tr></table>

The aligned features are concatenated and passed to the segmentation head ${ \mathcal { H } } _ { \mathrm { s e g } }$ to obtain the pixel-wise prediction:

$$
M = \mathcal { H } _ { \mathrm { s e g } } \left( \operatorname { C o n c a t } \left[ \widetilde { F } _ { 1 } , \widetilde { F } _ { 2 } , \widetilde { F } _ { 3 } , \widetilde { F } _ { 4 } \right] \right) .\tag{11}
$$

The segmentation network is optimized using a combination of cross-entropy and Dice losses [54]:

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { C E } } + \mathcal { L } _ { \mathrm { D i c e } } .\tag{12}
$$

As shown in Fig. 7, the multi-level DiT features capture both local cellular structures and broader tissue context. The segmentation decoder therefore focuses on integrating these representations and mapping them to task-specific labels, rather than relearning pathology morphology from limited annotations. This allows the learned generative prior to be effectively transferred to semantic segmentation.

## III. EXPERIMENTS

## A. Datasets

model takes $2 5 6 \times 2 5 6$ patches as input, featuring a patch size of 8, an embedding dimension of 768, 12 Transformer blocks, and 12 attention heads. We employ the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . The model is trained for 200,000 iterations with a total batch size of 16. In Stage III, the segmentation framework is initialized with the DiT backbone pretrained in Stage II. We optimize the network using AdamW, explicitly applying a learning rate of $1 \times 1 0 ^ { - 5 }$ to the backbone and $3 \times 1 0 ^ { - 4 }$ to the newly initialized segmentation head. To ensure fair evaluation, we employ a 3-fold cross-validation strategy, training the model for 20 epochs per fold using the loss function defined in Eq. 12. All experiments are conducted in parallel across four NVIDIA A6000 GPUs.

As shown in Fig. 1(b), we use two pathology segmentation datasets for downstream evaluation: the public PanNuke dataset [55] and our proposed PathSeg-380K dataset. Pan-Nuke contains 5 nucleus categories: Inflammatory (32,276), Connective (50,585), Neoplastic (77,403), Non-neoplastic epithelial (26,572), and Dead (2,908). PathSeg-380K is our inhouse pathology segmentation dataset, collected with ethical approval, covering 12 categories: Mast (3,253), DC (9,087), pDC (15,550), B (44,993), T (42,944), Endothelial (49,498), Plasma (82,850), Mono/Mac (59,319), Fibroblast (69,868), MSC (114), Cycling (1,436), and NA (2,720). Together, they cover diverse semantic categories and morphological patterns for evaluating segmentation performance.

## B. Implementation Details

In Stage II, we train the pathology generative prior using a DiT backbone with the objective defined in Equation 7. The

## C. Reliability of Agent-based Quality Assessment

We implement the pathology expert agent using GPT-4.1 mini (“gpt-4.1-mini-2025-04-14”) and evaluate its scoring reliability on an independent set of 600 pathology patches, which are excluded from the pilot set used to train the quality regressor. Two pathologists independently assess each patch in terms of cell presence, stain quality, and histologic clarity. Samples with a score difference of two or more points are jointly reviewed to obtain the final consensus score. We use the pathologist consensus as the reference and report Spearman’s correlation [56] for each quality dimension. Mean absolute error (MAE) and quadratic weighted Cohen’s κ [57] are averaged across the three dimensions.

As shown in Table II, the two pathologists show consistent assessments, with Spearman’s correlations above 0.81 and an average weighted κ of 0.827. The agent reaches correlations of 0.782, 0.811, and 0.829 for cell presence, stain quality, and histologic clarity, respectively. Its average MAE is 0.326, while the weighted κ against the pathologist consensus is 0.814, close to the agreement between the two pathologists. The quality regressor shows a small decrease against the consensus, with an MAE of 0.378 and a weighted κ of 0.789. Meanwhile, its correlations with the agent are all above 0.90, with an MAE of 0.170 and a weighted κ of 0.912. These results indicate that the regressor closely reproduces the agent scores while retaining strong agreement with pathologist assessment, supporting its use for large-scale patch screening.

Input  
GT  
Ours  
HoverNet  
CellViT  
Un-SAM  
SegDINO  
![](images/cdb04a43b323c15856ca5820076a65413756a3a27010eab0eb6b13a8ca02fe8a.jpg)  
Fig. 9. Visual comparison of segmentation results across pathology tissue types. Green boxes highlight challenging regions where our framework better matches the ground truth (GT), particularly under cluttered backgrounds and varying cell scales.

TABLE II  
RELIABILITY OF PATHOLOGY QUALITY ASSESSMENT AGAINST PATHOLOGIST CONSENSUS ON AN INDEPENDENT TEST SET.
<table><tr><td rowspan="2">Evaluator</td><td colspan="3">Spearman&#x27;s  $\rho \uparrow$ </td><td rowspan="2">MAE↓ κ↑ Avg.</td><td rowspan="2">Avg.</td></tr><tr><td>Cell</td><td>Stain</td><td>Clarity</td></tr><tr><td>Pathologist 1 vs. Pathologist 2</td><td>0.815</td><td>0.835</td><td>0.816</td><td>0.306</td><td>0.827</td></tr><tr><td>Agent vs. Consensus</td><td>0.782</td><td>0.811</td><td>0.829</td><td>0.326</td><td>0.814</td></tr><tr><td>Regressor vs. Consensus</td><td>0.759</td><td>0.797</td><td>0.796</td><td>0.378</td><td>0.789</td></tr><tr><td>Regressor vs. Agent</td><td>0.909</td><td>0.914</td><td>0.913</td><td>0.170</td><td>0.912</td></tr></table>

## D. Effect of Agent-based Data Curation

We evaluate generation quality using FID [58], KID [59], and Mean Opinion Score (MOS), where lower FID and KID and higher MOS indicate better quality. MOS is independently assessed by two pathologists on a five-point scale, with “1” and “5” indicating the lowest and highest quality, respectively, and their average score is reported. For each curation setting, we train the generative model on the corresponding retained subset, generate 10,000 pathology images, and compute FID and KID against real images. Downstream utility is evaluated by transferring the learned backbone to PanNuke under the same segmentation protocol, with mIoU as the metric. In Table III, From Scratch denotes the full segmentation architecture with a randomly initialized DiT backbone and the same multi-level decoder, while w/o Agent (100%) denotes generative pretraining on the complete unfiltered training set. For the remaining settings, an image is retained only when its predicted scores for cell presence, stain quality, and histologic clarity simultaneously fall within the top 75%, 50%, 25%, 10%, or 5% of their respective distributions. Using the complete unfiltered dataset yields an FID of 38.75, a KID of 0.056, and an MOS of 3.12. Agent-based curation generally improves generation quality as the selection criterion becomes stricter. In particular, requiring samples to rank within the top 10% in each quality dimension reduces FID and KID to 23.96 and 0.021, respectively, while achieving the highest MOS of 4.43. Further tightening the threshold to the top 5% improves FID and KID to 23.11 and 0.019, but slightly decreases MOS to 4.41. These results indicate that training with higher-ranked samples improves distributional fidelity and pathological plausibility.

TABLE III  
EFFECT OF AGENT-BASED DATA CURATION ON GENERATION QUALITY AND DOWNSTREAM SEGMENTATION PERFORMANCE ON PANNUKE.
<table><tr><td>Curation Setting</td><td>FID ↓</td><td>KID ↓</td><td>MOS ↑</td><td>mIoU (%) ↑</td></tr><tr><td>From Scratch</td><td></td><td></td><td></td><td>47.19</td></tr><tr><td>w/o Agent (100%)</td><td>38.75</td><td>0.056</td><td>3.12</td><td>53.41</td></tr><tr><td>w/ Agent (Top 75%)</td><td>33.61</td><td>0.032</td><td>4.06</td><td>54.42</td></tr><tr><td>w/ Agent (Top 50%)</td><td>30.08</td><td>0.028</td><td>3.87</td><td>54.46</td></tr><tr><td>w/ Agent (Top 25%)</td><td>24.94</td><td>0.024</td><td>4.20</td><td>55.12</td></tr><tr><td>w/ Agent (Top 10%)</td><td>23.96</td><td>0.021</td><td>4.43</td><td>55.46</td></tr><tr><td>w/ Agent (Top 5%)</td><td>23.11</td><td>0.019</td><td>4.41</td><td>55.03</td></tr></table>

The downstream results show a similar trend. Training the segmentation model from scratch achieves an mIoU of 47.19%, while transferring the generative prior trained without agent-based curation increases it to 53.41%. Using the prior trained on samples ranked within the top 10% in each quality dimension further improves mIoU to 55.46%, corresponding to gains of 8.27 and 2.05 percentage points over the two settings, respectively. Although the top 5% setting achieves the best FID and KID, its mIoU decreases to 55.03%, suggesting that overly strict filtering may reduce the diversity needed for transferable representation learning. We therefore adopt the top 10% setting in subsequent experiments.

TABLE IV  
ABLATION STUDY OF THE PROPOSED COMPONENTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Gen. Prior</td><td rowspan="2">Agent</td><td rowspan="2">Seg. Head</td><td colspan="2">PanNuke</td><td colspan="2">PathSeg-380K</td></tr><tr><td>mIoU (%)↑</td><td>mASD↓</td><td>mIoU (%)↑</td><td>mASD↓</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>44.58</td><td>30.93</td><td>44.23</td><td>8.86</td></tr><tr><td>M1</td><td>√</td><td rowspan="4">√</td><td></td><td>52.17</td><td>25.86</td><td>52.08</td><td>6.58</td></tr><tr><td>M2</td><td>√</td><td></td><td>53.85</td><td>24.42</td><td>53.72</td><td>5.37</td></tr><tr><td>M3</td><td>√</td><td></td><td>53.41</td><td>24.01</td><td>53.19</td><td>5.51</td></tr><tr><td>Ours (Full)</td><td>√</td><td></td><td>55.46</td><td>22.91</td><td>55.45</td><td>4.81</td></tr></table>

## E. Comparison on Downstream Segmentation

1) Quantitative Comparisons: We compare our framework with two groups of methods. The first group consists of task-specific segmentation models, including CNN-based U-Net [42], HoVer-Net [44], and CPP-Net [45]; Transformerbased TransUNet [43] and CellViT [46]; and SAM-based Un-SAM [47]. The second group evaluates pretrained encoders coupled with the same segmentation decoder as our framework, including SegDINO [33], [34] and MAE [48] pretrained on natural images, as well as CONCH [38] and UNI [39] pretrained on pathology images.

As shown in Table I, our framework achieves the best overlap-based segmentation performance on PanNuke, with an mIoU of 55.46%, an mDice of 68.81%, and a p-Acc of 92.77%. Compared with the corresponding second-best results, it improves mIoU, mDice, and p-Acc by 1.69, 2.08, and 0.52 percentage points, respectively. On PathSeg-380K, our framework achieves the best results across all four metrics, reaching 55.45% mIoU, 67.81% mDice, 93.44% p-Acc, and 4.81 mASD. It surpasses the corresponding second-best results by 0.86, 1.48, and 0.70 percentage points in mIoU, mDice, and p-Acc, respectively, while reducing mASD from 5.37 to 4.81. Improvements marked with an asterisk in Table I are statistically significant compared with the corresponding second-best method under a paired two-sided t-test (p < 0.05). These results demonstrate that the transferred generative representations improve segmentation accuracy across both datasets, particularly in terms of region overlap.

2) Visual Comparisons: As shown in Fig. 9, our method produces segmentation results that better match the ground truth. In challenging regions containing small or densely clustered nuclei, it reduces missed predictions and boundary errors while providing more complete masks. These observations are consistent with the quantitative results in Table I.

## F. Ablation Study

1) Module Ablation: To assess the contribution of each component, we conduct the ablation study shown in Table IV. The Baseline uses a randomly initialized DiT with an MLP head. Introducing the pretrained generative prior in M1 improves mIoU by 7.59 and 7.85 percentage points on PanNuke and PathSeg-380K, respectively. Starting from M1, M2 and M3 separately introduce Agent Screening and the multi-level decoder. Agent Screening improves mIoU by 1.68 and 1.64 points, while the multi-level decoder yields gains of 1.24 and 1.11 points and reduces mASD from 25.86 to 24.01 on PanNuke and from 6.58 to 5.51 on PathSeg-380K. Combining all components achieves the best performance on both datasets, reaching mIoUs of 55.46% and 55.45% and mASDs of 22.91 and 4.81, respectively. These results demonstrate that agentbased data curation and multi-level feature fusion provide complementary benefits to the generative prior.

![](images/0d736a08994278f6dd6ab32f8587f7c8f1ad9ba0ef6edb952907302056285d14.jpg)  
Fig. 10. Effect of hyperparameter t on mIoU, mDice, and mASD. Yellow stars highlight the best-performing settings for each metric.

2) Hyperparameter Ablation: The timestep t controls the semantic level of features extracted from the DiT backbone. Although the model is trained over the full flow trajectory t ∈ [0, 1], downstream segmentation uses a fixed t without performing iterative denoising. Through AdaLN [49], smaller values emphasize local textures and cellular details, whereas larger values favor more abstract tissue semantics. We therefore investigate the optimal balance between spatial detail and semantic abstraction. As shown in Fig. 10, increasing t from 0.1 to 0.3 improves the overlap-based metrics, with t=0.3 achieving the best mIoU of 55.46% and mDice of 68.81%, together with a competitive mASD of 22.91. Although t=0.4 yields a slightly lower mASD, its mIoU and mDice are inferior to those obtained at t=0.3. At larger timesteps, the overall segmentation performance generally declines, with mIoU decreasing to 53.58% at t=0.9, likely because excessive corruption weakens fine-grained spatial information. We therefore set t=0.3 for all experiments as the best overall trade-off between region overlap and boundary accuracy.

## IV. CONCLUSION

In this work, we presented an agent-driven framework that integrates pathology data curation, generative prior learning, and downstream semantic segmentation. The pathology expert agent enables scalable quality screening and the construction of the curated TCGA-200K dataset. Using these high-quality samples, the DiT backbone learns pathology-specific representations that capture both cellular morphology and tissue-level semantics, which are effectively transferred to a lightweight segmentation head. Strong results on PanNuke and PathSeg-380K demonstrate the value of generative pretraining for both pathology image synthesis and downstream analysis, while future work will extend the framework to other clinically relevant tasks, including classification and survival analysis. We will also investigate more efficient agent-based curation strategies to reduce screening costs while preserving data quality across institutions and staining protocols.

[1] S. Yang et al. Lcm-net: Llm-driven cross-modality moe feature fusion network for cancer survival analysis. IEEE Transactions on Medical Imaging, 45(5):2505–2517, 2026.

[2] R. Verma et al. Monusac2020: A multi-organ nuclei segmentation and classification challenge. IEEE Transactions on Medical Imaging, 40(12):3413–3423, 2021.

[3] B. Ehteshami Bejnordi et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama, 318(22):2199–2210, 2017.

[4] N. Coudray et al. Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning. Nature medicine, 24(10):1559–1567, 2018.

[5] P. Puttapirat et al. Openhi-an open source framework for annotating histopathological image. In 2018 IEEE international conference on bioinformatics and biomedicine (BIBM), pages 1076–1082. IEEE, 2018.

[6] R. Wang, S. Yang, Q. Li, and D. Zhong. Cytogan: Unpaired staining transfer by structure preservation for cytopathology image analysis. Computers in Biology and Medicine, 180:108942, 2024.

[7] H. Zhou, S. Yang, S. Yang, J. Qin, L. Chen, and L. Zhu. Coc: Chainof-cancer based on cross-modal autoregressive traction for survival prediction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 85–94. Springer, 2025.

[8] A. J. Schaumberg et al. Large-scale annotation of histopathology images from social media. BioRxiv, 1:396663, 2018.

[9] A. Sylolypavan, D. Sleeman, H. Wu, and M. Sim. The impact of inconsistent human annotations on ai driven clinical decision making. NPJ Digital Medicine, 6(1):26, 2023.

[10] X. Yi, E. Walia, and P. Babyn. Generative adversarial network in medical imaging: A review. Medical image analysis, 58:101552, 2019.

[11] I. Goodfellow et al. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

[12] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[13] X. Yu et al. Diffusion-based data augmentation for nuclei image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 592–602. Springer, 2023.

[14] R. Harb, T. Pock, and H. Muller. Diffusion-based generation of¨ histopathological whole slide images at a gigapixel scale. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5131–5140, 2024.

[15] S. Yang, Z. Xing, H. Zhou, and L. Zhu. K-stain: Keypointdriven correspondence for h&e-to-ihc virtual staining. arXiv preprint arXiv:2511.06709, 2025.

[16] V. Brodsky et al. Generative artificial intelligence in anatomic pathology. Archives of pathology & laboratory medicine, 149(4):298–318, 2025.

[17] Y. Zhang et al. Datasetgan: Efficient labeled data factory with minimal human effort. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10140–10150. IEEE, 2021.

[18] Z. Xiao et al. Artificial intelligence in pathology: advancing large models for scalable applications. Annual Review of Biomedical Data Science, 8(1):149–171, 2025.

[19] S. Yellapragada et al. Pixcell: A generative foundation model for digital histopathology images. ArXiv, pages arXiv–2506, 2025.

[20] M. Ali, M. Ali, M. Hussain, and D. Koundal. Generative adversarial networks (gans) for medical image processing: recent advancements. Archives of Computational Methods in Engineering, 32(2):1185–1198, 2025.

[21] L. Yang et al. Diffusion models: A comprehensive survey of methods and applications. ACM computing surveys, 56(4):1–39, 2023.

[22] Y. Zhang et al. To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now. In European Conference on Computer Vision, pages 385–403. Springer, 2024.

[23] Z. Weng et al. Grandqc: A comprehensive solution to quality control problem in digital pathology. Nature Communications, 15(1):10685, 2024.

[24] Y. Sun et al. Cpathagent: An agent-based foundation model for interpretable high-resolution pathology image analysis mimicking pathologists’ diagnostic logic. Advances in Neural Information Processing Systems, 2025.

[25] J. Li et al. Agent hospital: A simulacrum of hospital with evolvable medical agents. arXiv preprint arXiv:2405.02957, 2024.

[26] F. Ghezloo et al. Pathfinder: A multi-modal multi-agent system for medical diagnostic decision-making applied to histopathology. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23431–23441, 2025.

[27] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE Transactions on Medical Imaging, 39(6):1856– 1867, 2019.

[28] J. Zhang et al. Sam-path: A segment anything model for semantic segmentation in digital pathology. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 161–170. Springer, 2023.

[29] S. K. Danisetty, A. Graikos, S. Yellapragada, and D. Samaras. Pathsegdiff: Pathology segmentation using diffusion model representations. In Deep Generative Models, pages 141–150, Cham, 2026. Springer Nature Switzerland.

[30] W. Chen et al. Beyond vit tokens: Masked-diffusion pretrained convolutional pathology foundation model for cell-level dense prediction. arXiv preprint arXiv:2605.08276, 2026.

[31] D. Komura et al. Restaining-based annotation for cancer histology segmentation to overcome annotation-related limitations among pathologists. Patterns, 4(2), 2023.

[32] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[33] M. Caron et al. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[34] S. Yang, H. Wang, Z. Xing, S. Chen, and L. Zhu. Segdino: An efficient design for medical and natural image segmentation with dino-v3. arXiv preprint arXiv:2509.00833, 2025.

[35] S. Yang et al. Segdino: introducing multi-scale structure into dino for efficient medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 514–524. Springer, 2026.

[36] S. Yang, Z. Xing, and L. Zhu. Vq-seg: Vector-quantized token perturbation for semi-supervised medical image segmentation. Advances in Neural Information Processing Systems, 2026.

[37] T. Kataria, B. Knudsen, and S. Elhabian. To pretrain or not to pretrain? a case study of domain-specific pretraining for semantic segmentation in histopathology. In Workshop on Medical Image Learning with Limited and Noisy Data, pages 246–256. Springer, 2023.

[38] M. Y. Lu et al. A visual-language foundation model for computational pathology. Nature medicine, 30(3):863–874, 2024.

[39] R. J. Chen et al. Towards a general-purpose foundation model for computational pathology. Nature medicine, 30(3):850–862, 2024.

[40] R. L. Grossman et al. Toward a shared vision for cancer genomic data. New England Journal of Medicine, 375(12):1109–1112, 2016.

[41] M. Y. Lu, D. F. K. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6):555–570, 2021.

[42] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

[43] J. Chen et al. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.

[44] S. Graham et al. Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical image analysis, 58:101563, 2019.

[45] S. Chen, C. Ding, M. Liu, J. Cheng, and D. Tao. Cpp-net: Context-aware polygon proposal network for nucleus segmentation. IEEE Transactions on Image Processing, 32:980–994, 2023.

[46] F. Horst et al. Cellvit: Vision transformers for precise cell segmentation¨ and classification. Medical image analysis, 94:103143, 2024.

[47] Z. Chen, Q. Xu, X. Liu, and Y. Yuan. Un-sam: Domain-adaptive self-prompt segmentation for universal nuclei images. Medical Image Analysis, 103:103607, 2025.

[48] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick. Masked´ autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[49] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[50] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[51] D. Baranchuk, I. Rubachev, A. Voynov, V. Khrulkov, and A. Babenko. Label-efficient semantic segmentation with diffusion models. In International Conference on Learning Representations, 2022.

[52] J. Wu et al. Medsegdiff: Medical image segmentation with diffusion probabilistic model. In Medical imaging with deep learning, pages 1623–1639. PMLR, 2024.

[53] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

[54] F. Milletari, N. Navab, and S.-A. Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[55] J. Gamper, N. Alemi Koohbanani, K. Benet, A. Khuram, and N. Rajpoot. Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In European congress on digital pathology, pages 11–19. Springer, 2019.

[56] C. Spearman. The proof and measurement of association between two things. 1961.

[57] J. Cohen. Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. Psychological bulletin, 70(4):213, 1968.

[58] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[59] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton. Demystifying´ mmd gans. In International conference on learning representations, 2018.