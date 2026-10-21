# CAPS: Clinically-Verifiable Anatomy-Prioritized Synthesis for Missing MRI Imputation in Brain Tumor Segmentation

Cecilia Diana-Albelda<sup>a</sup>,<sup>†</sup>,<sup>∗</sup>, Arthur Longuefosse<sup>b</sup>,<sup>†</sup>, Mario Rubén Paz Campos<sup>c</sup>, Álvaro García-Martín<sup>a</sup>, Jesus Bescos<sup>a</sup>

<sup>†</sup> Equal contribution. <sup>∗</sup> Corresponding author: cecilia.diana.albelda@gmail.com

<sup>a</sup> Universidad Autónoma de Madrid, Department of Electronic Technology and Communications, Madrid, Spain

<sup>b</sup> RIKEN Center for Integrative Medical Sciences, Medical Data Deep Learning Team, Tokyo, Japan

<sup>c</sup> Hospital Universitario La Princesa, Diagnostic Radiology Department, Madrid, Spain

## Abstract

Multi-parametric MRI is the gold standard for neuro-oncology, yet acquiring complete 4-modality protocols may be unfeasible in real-world clinical practice. When handling missing modalities, existing automated systems force a critical compromise: robust segmenters operate as uninterpretable "black boxes" that bypass image generation, while standard synthesis models often produce blurry reconstructions lacking diagnostic reliability. To address this issue, we propose CAPS, a Clinically-Verifiable Anatomy-Prioritized Synthesis framework that reconciles computational precision with clinical transparency. Our multi-modal generative architecture is driven by a hybrid objective function (L<sub>1</sub> + L<sub>AFP</sub> + L<sub>DualDice</sub>) that explicitly enforces the preservation of deep anatomical features and strict topological boundaries, rather than relying solely on pixel-wise optimization. Evaluated comprehensively on the BraTS Adult Glioma dataset, the framework achieves superior synthesis fidelity across intensity-based metrics, and demonstrates robust downstream segmentation performance (82 38% mean Dice), successfully restoring verifiable visual context. Furthermore, the framework exhibits exceptional zero-shot cross-domain generalization, maintaining high structural fidelity across completely unseen Meningioma, Pediatric, and Sub-Saharan African cohorts without domain-specific adaptation. Finally, a blinded radiological assessment confirms that our anatomy-aware approach significantly enhances both perceived image quality and clinical utility, providing clinicians with the reliable visual evidence required to trust downstream automated workflows.

Keywords: Image Synthesis, Missing Modalities, MRI, Brain Tumors, Generalization

## 1. Introduction

Gliomas are among the most prevalent primary Central Nervous System (CNS) tumors Weller et al. (2015), with high-grade gliomas being particularly aggressive due to their highly infiltrative growth and low survival rates Guo et al. (2023). The gold standard for the diagnosis, treatment planning, and post-operative monitoring of these tumors is multi-parametric Magnetic Resonance Imaging (mp-MRI). A standard neurooncological protocol typically comprises four scan modalities: Native T1-weighted (T1n), Post-contrast T1-weighted (T1c), T2-weighted (T2w), and T2 Fluid-Attenuated Inversion Recovery (T2f) Diana-Albelda et al. (2025a). Each modality provides unique and complementary tissue contrasts that are critical for accurate tumor characterization. For instance, the active, proliferating tumor core and neoangiogenesis are primarily visible in the gadolinium-enhanced T1c sequence, while the surrounding peritumoral edema and non-enhancing infiltrative regions are best delineated using T2f. However, acquiring the complete 4-modality protocol in real-world clinical settings is sometimes unfeasible, as scans may be incomplete due to time constraints, patient motion artifacts, financial limitations, or contraindications to gadolinium-based contrast agents Li et al. (2024). The absence of even a single modality introduces a twofold clinical and algorithmic challenge. Clinically, it deprives neuroradiologists of vital anatomical information, potentially leading to suboptimal surgical planning and, ultimately, negatively impacting patient survival and recurrence rates. Algorithmically, state-of-the-art automated segmentation systems, which are increasingly integrated to assist clinical workflows, are typically designed to process the full set of four input channels, thereby experiencing severe performance degradation or failure when a modality is missing or zero-padded Sharma and Hamarneh (2019).

To mitigate this problem, two main strategies have emerged in the literature. The first one focuses on designing robust segmentation frameworks that can predict tumor boundaries despite incomplete inputs Zhang et al. (2022); Pipoli et al. (2025). While these systems achieve remarkable accuracy, they present a significant challenge in clinical practice: they output a tumor mask without restoring the missing anatomical image, leaving the clinician without the visual evidence needed to corroborate the automated delineation. The second strategy relies on image synthesis to impute the missing modality Diana-Albelda et al. (2025b). However, traditional synthesis methods primarily optimize standard pixel-wise reconstruction losses, such as L1, which frequently results in blurry outputs that miss important structural details Longuefosse et al. (2025), thereby restricting their practical utility for the clinicians.

Addressing these limitations, we propose CAPS, a novel Clinically-Verifiable Anatomy-Prioritized Synthesis framework that bridges the gap between computational robustness and clinical reliability. Instead of optimizing solely for pixel-level similarity, our approach explicitly maximizes the preservation of anatomical structures and tumor topology. By restoring the missing modality with high structural fidelity, our method provides the clinician with the necessary visual data to plan surgeries and verify AI-generated segmentations, while simultaneously supporting robust downstream automated segmentation, as illustrated in Figure 1.

The main contributions of this paper are summarized as follows:

• Empirical Analysis of the Accuracy-Interpretability Trade-of. We compare robust segmentation frameworks against synthesis-based pipelines and conclude that, while robust segmenters may achieve marginally higher numerical accuracy on missing modalities, high-fidelity image synthesis is an indispensable step to provide the visual context that neuroradiologists require for diagnostic verification.

• Anatomy-Prioritized Synthesis. Unlike traditional approaches that rely solely on pixel-wise optimization, we design a generative architecture driven by a hybrid loss function $( \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P } +$ $\mathcal { L } _ { D u a l D i c e } )$ that explicitly enforces the preservation of fine anatomical structures, tumor boundaries, and overall integrity. Building upon the AFP loss originally proposed for MR-to-CT translation Longuefosse et al. (2025), we adapt it to the multimodal MRI synthesis setting and complement it with a Dual Dice loss that jointly penalizes regionlevel and boundary-level deviations. Central to our design is a frozen, modality-specialized segmentation network that serves a dual supervisory role: as a multi-scale anatomical feature extractor and as a topological boundary enforcer, directly coupling the generative objective with downstream diagnostic consistency.

![](images/fb1f8016fcc0b28676fae876cd6d57951b2c5b061c3054d147deb386a1a0aab0.jpg)  
Figure 1: Overview of CAPS. Given three available MRI modalities (X: T1n, T2w, T2f), the generator G synthesizes the missing sequence (Y<sup>ˆ</sup> , T1c in this example). The synthesized volume is then concatenated with the available modalities and fed into the clinical segmenter S , which produces the predicted tumor mask (Z<sup>ˆ</sup>) for the three nested BraTS regions (WT, TC, ENH). Moreover, the synthesized volume is provided to a radiologist for blinded clinical validation, ensuring that the framework delivers both automated segmentation and verifiable visual context.

• Blinded Radiological Evaluation. Moving beyond standard pixel-level metrics (e.g., PSNR, NMSE) that may not fully reflect clinical utility, we incorporate a blinded clinical validation. We conclude that our CAPS approach yields synthetic images with improved diagnostic quality and structural reliability compared to the standard synthesis baseline. To facilitate reproducibility and encourage clinical validation in future synthesis research, we publicly release the evaluation tool used for this assessment.

• Cross-Domain Generalization. We successfully evaluate the synthesis capability of our method on completely unseen Meningioma, Pediatric Glioma, and Sub-Saharan Glioma datasets, efectively over coming the domain shift without requiring any domain-specific adaptation.

## 2. Related Work

Handling missing mp-MRI modalities in neurooncology generally falls into two distinct paradigms: robust segmentation architectures that adapt to incomplete inputs Zhang et al. (2024), and medical image synthesis models that generate the absent data Dar et al. (2019).

## 2.1. Robust Segmentation with Missing Modalities

Recent literature has extensively explored adaptive segmentation networks designed to be resilient against incomplete input configurations. Several studies have focused on learning modality-agnostic networks and shared latent representations that remain stable even when specific sequences are omitted Dorent et al. (2019). For instance, early foundational approaches like HeMIS Havaei et al. (2016) achieved this by computing statistical moments across available feature maps. This concept has evolved into sophisticated featurelevel fusion strategies employing attention mechanisms, transformer-based architectures like mmFormer Zhang et al. (2022), and state-space models such as IM-Fuse Pipoli et al. (2025). Additionally, knowledge distillation techniques have been widely proposed to transfer structural information from a full-modality “teacher” network to a missing-modality “student” network Hu et al. (2020).

Despite achieving remarkably high computational accuracy, typically evaluated through standard metrics like the Dice Similarity Coeficient (DSC) and Hausdorf Distance (HD95), these robust segmenters face a critical usability limitation. Because they do not generate the missing scan and directly predict tissue boundaries, they fail to provide the structural reference that radiologists rely on. This lack of diagnostic transparency hinders the adoption of pure segmentation approaches in clinical workflows where visual verification is mandatory.

## 2.2. Medical Image Synthesis

A parallel line of research focuses on cross-modality medical image synthesis to impute the missing sequences. Early generative methods heavily relied on Generative Adversarial Networks (GANs) Goodfellow et al. (2014); Tavse et al. (2022); Al Khalil et al. (2024), such as Pix2Pix Isola et al. (2017) and CycleGAN Zhu et al. (2017), which primarily performed one-to-one image translations. More recently, the field has advanced towards highly adaptable architectures capable of handling multi-modal inputs, shifting from basic 1- to-1 mappings to complex many-to-one synthesis models Meng et al. (2024). This evolution is highlighted by advanced Difusion Models Kebaili et al. (2025b) like AMM-Dif Kebaili et al. (2025a), which adaptively process variable sets of inputs, and State Space Models like I2I-Mamba Atli et al. (2024), which capture long-range contextual features across multi-contrast MRI.

However, a fundamental limitation persists: most of these models are trained using generic pixel intensitybased objectives like $L _ { 1 }$ or Mean Squared Error Nie et al. (2018). Because these losses penalize errors on a purely pixel basis, they lack structural awareness and tend to over-smooth the generated images, washing out critical high-frequency details such as intricate tumor borders Isola et al. (2017). Furthermore, their performance is often solely measured via Image Quality Metrics (e.g., PSNR, SSIM), leaving their actual utility for downstream automated clinical tasks unverified.

## 2.3. Structural Constraints and Joint Architectures

To mitigate the smoothing efects of intensity-based losses, recent studies have introduced structural constraints and joint synthesis-segmentation frameworks. For example, methods like EaGAN Yu et al. (2019) explicitly integrate edge-aware perceptual constraints to preserve high-frequency structures. Similarly, 3D Wavelet Difusion Models (WDM) Ferreira et al. (2024) strive to maintain spatial resolution while generating missing modalities. Moreover, unified frameworks have been proposed to jointly optimize synthesis and segmentation, acknowledging that image generation must ultimately be useful for tumor delineation Dorent et al. (2025). However, while feature-based objectives like perceptual loss efectively balance pixel accuracy and realism in natural scenes, they often lack the anatomical rigor required for 3D medical volumes, where structural integrity must be prioritized over stylistic consistency Longuefosse et al. (2025).

Despite these significant advances, existing joint frameworks often struggle to preserve fine tissue interfaces when critical modalities are missing. Our work addresses this exact issue. By integrating a targeted Anatomical Feature-Prioritized (AFP) loss and a Dual Dice loss into a multi-modal 3-to-1 generative architecture, our framework successfully reconciles algorithmic precision with clinical interpretability. This approach successfully delivers highly verifiable restorations for radiologists while maintaining competitive performance in subsequent segmentation tasks.

## 3. Method

This section details the proposed Clinically-Verifiable Anatomy-Prioritized Synthesis framework (CAPS), designed to impute missing MRI modalities while preserving high-fidelity structural details. First, the mathematical problem formulation and the overall two-stage pipeline are established. Subsequently, the network architecture and the hybrid objective function are introduced. Finally, the specific implementation and training configurations required to ensure robust convergence and downstream clinical utility are outlined.

## 3.1. Problem Formulation and Pipeline

The core objective of our framework is to synthesize a missing MRI scan from a subset of available complementary modalities, ensuring that the generated image retains both structural fidelity and diagnostic utility.

Let $\mathcal { M } = \{ T 1 n , T 1 c , T 2 w , T 2 f \}$ denote the complete set of four standard neuro-oncological MRI scan modalities acquired for a given patient. In real-world clinical scenarios, it is common for a specific modality to be missing, corrupted, or otherwise unavailable Li et al. (2024). We denote the target missing modality as $t \in \mathcal { M }$ and its corresponding ground-truth 3D volume as ${ \mathit { Y } } ,$ and formalize the problem as a multi-modal synthesis task (specifically, a 3-to-1 mapping). The set of available input modalities is therefore defined as $X = \mathcal { M } \backslash \{ t \}$ Our goal is to train a generative mapping function $G ,$ parametrized by weights θ, that takes the available modalities X as a multi-channel input and synthesizes a high-fidelity target estimation of the missing sequence, denoted as ${ \hat { Y } } \colon$

$$
\hat { Y } = G _ { \theta } ( X ) ,\tag{1}
$$

where the synthesized volume $\hat { Y }$ is explicitly optimized to approximate the underlying anatomical information on Y.

To achieve this, we propose a Clinically-Verifiable Anatomy-Prioritized Synthesis framework (CAPS) designed to operate in two interconnected phases (see Figure 1):

1. Image Synthesis Phase: The core generative network takes the available 3D volumes X and synthesizes $\hat { Y }$ according to Figure 2 scheme. Unlike traditional synthesis pipelines that optimize purely for intensity matching, our generator G is driven by a hybrid objective function. This function integrates a pre-trained feature extractor to explicitly align the latent anatomical representations of $\hat { Y }$ with those of Y through $S _ { a u x } .$ ensuring the preservation of intricate tumor morphologies.

2. Downstream Segmentation Phase: To validate the clinical utility of the synthesized modality, $\hat { Y }$ is subsequently combined with the available modalities X to assemble the full imputed protocol, defined as $M _ { s y n t h } = \chi \cup \{ \hat { Y } \}$ . This complete set is then fed into an independent segmentation network S to predict the final tumor boundaries.

This two-stage pipeline guarantees that the synthesis process is not evaluated in isolation, but rather through its ability to provide actionable and interpretable visual data that supports robust downstream clinical tasks.

It is important to note that, since each MRI sequence encodes fundamentally diferent tissue contrasts, both the generator G and the auxiliary extractor $S _ { a u x }$ are independently trained for each specific missing modality scenario $t \in M ,$ yielding four specialized pairs $( G _ { t } ,$ $S _ { a u x , t } )$ . In contrast, the final clinical segmenter S is trained as a single unified model capable of processing any imputed 4-modality protocol, regardless of which sequence was synthesized. This design ensures that each synthesis pathway is maximally specialized for its target modality, while the downstream segmenter remains a general-purpose clinical tool.

## 3.2. CAPS Overview

The proposed method is organized into three distinct components designed to prioritize anatomical fidelity: a generative trainable network $G ,$ an auxiliary feature extractor $S _ { a u x } .$ , and a final clinical segmenter S . This tripartite design ensures that the synthesis process is guided by deep anatomical priors while remaining robust for downstream diagnostic tasks.

![](images/e40752481afa850115edbec5f5d4840f09c75e5ba02829a89f5f87991d7966f2.jpg)  
Figure 2: Overview of the Synthesis Framework. A trainable generator G synthesizes the missing modality $( { \hat { Y } } ,$ T1c in this example) from the available ones (X: T1n, T2w, T2f) through a hybrid objective function. First, a pixel-level $\mathcal { L } _ { 1 }$ loss enforces global intensity consistency between the synthesized volume $\hat { Y }$ and the real one Y. Then, both are processed by a frozen pre-trained auxiliary segmenter $( S _ { a u x } ) ,$ which acts as an expert feature extractor to compute the Anatomical Feature-Prioritized $( \mathcal { L } _ { A F P } )$ loss across multi-scale latent representations. Finally, the predicted mask $( \hat { Z } _ { a u x } )$ and the real one $( Z )$ are used to calculate the topological Dual Dice loss, thereby ensuring the preservation of both deep structural invariants and precise tumor boundaries.

## 3.2.1. Generator (G)

The generator consists of a 3D Residual U-Net (nnResU-Net) Isensee et al. (2024) architecture configured with six encoding and decoding stages.

By incorporating residual connections within each convolutional block, the network facilitates an eficient gradient flow that prevents the vanishing gradient problem typical of deep 3D models Isensee et al. (2021), thereby accelerating convergence and enhancing the stability of the generative process. The encoder is designed to process a multi-channel stacked input X, composed of the three available MRI modalities, while the decoder is responsible for synthesizing the single-channel volume $\hat { \hat { Y } }$ corresponding to the missing sequence. To prevent the emergence of chequerboard artifacts, the decoder utilizes a trilinear interpolation strategy followed by standard convolutions Longuefosse et al. (2026), ensuring smooth spatial transitions in the reconstructed anatomy.

## 3.2.2. Auxiliary Feature Extractor $( S _ { a u x } )$

To drive the anatomy-prioritized optimization, we utilize an auxiliary network, $S _ { a u x } ,$ which acts as a frozen expert during the training of G. This network is pretrained using only one single real modality as input to ensure that its latent embeddings are highly specialized in the specific anatomical features of each sequence. During the generative phase, $\boldsymbol { S } _ { a u x }$ remains strictly frozen and serves as a multi-scale feature extractor. Specifically, to compute the Anatomical Feature-Prioritized $( \mathcal { L } _ { A F P } )$ loss Longuefosse et al. (2025), feature maps are extracted from every level of both the encoder and the decoder, forcing the generator to preserve structural invariants across multiple levels of abstraction. Furthermore, this network provides the topological guidance required for the $\mathcal { L } _ { D u a l D i c e }$ loss, aligning the synthesized lesion area with the expected clinical morphology.

Architecturally, $S _ { a u x }$ shares the same nnU-Net Res-Enc M configuration and training objective as the final clinical segmenter S , with the critical distinction that its input layer is configured to accept a single modality channel rather than the full four-channel protocol.

## 3.2.3. Final Clinical Segmenter (S )

The final component of our framework is the clinical segmenter S , which utilizes the nnU-Net ResEnc M architecture Isensee et al. (2024) to achieve an optimal balance between segmentation performance and computational eficiency. This network is optimized using the standard nnU-Net objective function, combining Dice loss and Cross-Entropy loss in equal proportion. Unlike the auxiliary extractor, this network is trained to process the full complement of four input modalities. To enhance its resilience to synthetic imputation, S is trained under a specialized augmentation scheme where there is a 50% probability that any given modality is replaced by its synthetic counterpart generated by $G .$ During the inference phase, this network acts as the definitive evaluator, processing the imputed 4-modality protocol to produce the final tumor masks. This configuration ensures that the segmenter is not only capable of interpreting synthetic data but also of integrating it seamlessly with real modalities to maintain high diagnostic accuracy.

## 3.3. Loss Formulation

The total objective function used to optimize the generator G is formulated as a weighted sum of three distinct components: 1) a global intensity loss $( \mathcal { L } _ { 1 } ) ; 2 )$ an anatomical feature-level loss $( \mathcal { L } _ { A F P } ) ;$ and 3) a topological segmentation loss $( \mathcal { L } _ { D u a l D i c e } )$

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P } + \mathcal { L } _ { D u a l D i c e } ,\tag{2}
$$

## 3.3.1. Voxel-Level Reconstruction Loss $( \mathcal { L } _ { 1 } )$

To ensure global intensity consistency and minimize the fundamental voxel-wise error between the synthesized modality $\hat { Y }$ and the ground truth one $Y ,$ we employ the standard $\mathcal { L } _ { 1 }$ distance. Unlike Mean Squared Error $( L _ { 2 } )$ , the $L _ { 1 }$ norm is less prone to excessive blurring, making it a suitable baseline for contrast restoration Isola et al. (2017). It is computed as:

$$
\mathcal { L } _ { 1 } = \frac { 1 } { V } \sum _ { \nu = 1 } ^ { V } \left| \hat { Y } _ { \nu } - Y _ { \nu } \right| ,\tag{3}
$$

where $V$ is the total number of voxels in the 3D volume, and $\hat { Y } _ { \nu }$ and $Y _ { \nu }$ represent the intensity values of the v-th voxel in the predicted and target sequences, respectively.

## 3.3.2. Anatomical Feature-Prioritized Loss $( \mathcal { L } _ { A F P } )$

While $\mathcal { L } _ { 1 }$ ensures global contrast, it lacks the spatial context required to preserve high-frequency structures. To explicitly promote the preservation of fine anatomical details, we employ the Anatomical Feature-Prioritized (AFP) loss, originally proposed for crossmodality translation in Longuefosse et al. (2025) and here adapted to the multi-modal MRI synthesis setting. Instead of computing diferences purely in the image space, $\mathcal { L } _ { A F P }$ calculates the $L _ { 1 }$ distance in a highly specialized latent space.

By leveraging the pre-trained and frozen segmentation network $S _ { a u x }$ as a feature extractor, we map both $\hat { Y }$ and Y into their respective deep embeddings. The loss is defined as:

$$
\mathcal { L } _ { A F P } = \frac { 1 } { N } \sum _ { l = 1 } ^ { N } \left. \phi _ { l } ( \hat { Y } ) - \phi _ { l } ( Y ) \right. _ { 1 } ,\tag{4}
$$

where N denotes the total number of extracted layers from $S _ { a u x } ,$ and $\phi _ { l } ( \cdot )$ represents the intermediate feature activation maps obtained at the l-th layer.

## 3.3.3. Dual Dice Loss $( \mathcal { L } _ { D u a l D i c e } )$

To enforce strict anatomical alignment specifically at the tumor regions, we propose a Dual Dice loss that evaluates both full-region overlap and boundary precision.

Since G outputs continuous intensity images, we map these volumes into a topological space by passing them through the frozen auxiliary segmentation network $S _ { a u x } .$ Let $\hat { Z } _ { a u x } = S _ { a u x } ( \hat { Y } )$ represent the tumor mask predicted from the synthesized image, the loss is then computed directly against the manual clinical ground truth mask, denoted as $Z .$ This formulation forces the generator to synthesize textures and intensities that are interpreted by the segmentation model in the same manner as the true anatomical pathology, thereby maximizing downstream diagnostic consistency.

To capture fine-grained morphological details, we define the boundary contours $\partial \hat { Z } _ { a u x }$ and ∂Z through a differentiable morphological gradient $\Delta _ { k }$ . For any binary mask B, the boundary is extracted as $\partial B = \Delta _ { k } ( B ) =$ $\delta _ { k } ( B ) - \epsilon _ { k } ( B )$ , where $\delta _ { k }$ and $\epsilon _ { k }$ denote dilation and erosion operators with default kernel size $k \ = \ 3$ , implemented via 3D max-pooling. The proposed topological penalty is then formulated as:

$$
\mathcal { L } _ { D u a l D i c e } = \left( 1 - \frac { 2 | \hat { Z } _ { a u x } \cap Z | } { | \hat { Z } _ { a u x } | + | Z | } \right) + \left( 1 - \frac { 2 | \partial \hat { Z } _ { a u x } \cap \partial Z | } { | \partial \hat { Z } _ { a u x } | + | \partial Z | } \right) ,\tag{5}
$$

where the first term ensures global topological agreement of the lesion, and the second term heavily penalizes morphological deviations strictly at the borders. This dual formulation prevents the generated tumor from artificially expanding or shrinking relative to the actual anatomy.

## 3.4. Implementation and Training Details

CAPS is implemented in PyTorch and integrated into the adapted nnU-Netv2 architecture Isensee et al. (2021); Longuefosse et al. (2024).

Preprocessing and Data Augmentation. All MRI volumes were independently normalized using standard Z-score normalization (zero mean, unit variance) per modality. During training, we employed a robust online data augmentation pipeline to prevent overfitting, which included spatial transformations (random rotations and scaling) and intensity transformations (Gaussian noise, Gaussian blur, brightness/contrast adjustments, and gamma correction) Isensee et al. (2021). Notably, spatial mirroring (flips) was explicitly disabled to preserve the inherent left-right anatomical asymmetries of the brain.

Training Configuration. The generative model was trained using 3D patches of size 96 × 160 × 160 voxels and a batch size of 2. The network parameters were optimized using Stochastic Gradient Descent (SGD) with Nesterov momentum Shi (2024) (0 99) and weight decay $( 3 \times 1 0 ^ { - 5 } )$ . The initial learning rate was set to 10<sup>−2</sup> and progressively decayed according to a polynomial schedule function over the training process.

To ensure training stability, especially given the multi-objective nature of our hybrid loss function, we applied gradient clipping. The model was trained for 1000 epochs, with each epoch consisting of 250 iterations. All models were trained on a single NVIDIA A40 GPU (48 GB VRAM).

Inference Phase. During the evaluation phase, the inference was performed using a sliding-window approach with a 50% patch overlap Isensee et al. (2021). To reconstruct the final synthetic modality Y<sup>ˆ</sup> , we employ a mean aggregation strategy across overlapping patches.

## 4. Experimental Setup

This section outlines the experimental configuration utilized to validate CAPS. Initially, the selected neurooncological datasets and their respective data partitioning strategies are detailed for both the primary evaluation and the cross-domain inference. Finally, the quantitative evaluation metrics employed to assess both image reconstruction quality and downstream segmentation accuracy are defined.

## 4.1. Datasets and Data Partitioning

To comprehensively train and evaluate our proposal, we utilized the globally recognized Brain Tumor Segmentation (BraTS) datasets Menze et al. (2014). All multi-parametric MRI sequences (T1n, T1c, T2w, and

T2f) across these datasets share a standardized preprocessing pipeline: they are co-registered to the same anatomical template, skull-stripped, and resampled to an isotropic resolution of $1 \times 1 \times 1$ mm.

Primary Training and Evaluation (Adult Glioma). The core development of our method was conducted using the BraTS 2023 Adult Glioma datasetBaid et al. (2021) (conceptually identical to the BraTS 2021 and 2022 cohorts), whose oficial training set comprises 1251 patient cases. We performed an 80% − 20% split on this subset: the 80% partition was exclusively used to train the generative synthesis network G, while the remaining 20% was strictly reserved for training both the auxiliary feature extractor $S _ { a u x }$ and the final clinical segmenter S , ensuring complete data separation from the generator.

In order to ensure a completely unbiased evaluation, all our quantitative results, ablation studies, and Stateof-the-Art comparisons were extracted directly from the oficial, unseen BraTS Adult Glioma Validation set, which comprises unseen 219 patient cases. To guarantee absolute fairness and reproducibility, the downstream segmentation metrics on this validation set were computed and retrieved through the oficial Synapse automated evaluation platform.

Cross-Domain Generalization Datasets. To evaluate the zero-shot robustness of our framework under significant domain shifts, we conducted additional inference on three distinct, unseen neuro-oncological cohorts from the BraTS challenge:

• BraTS Meningioma LaBella et al. (2023): 1000 cases featuring structurally distinct, typically benign extra-axial tumors.

• BraTS Pediatric Glioma Kazerooni et al. (2024): 99 cases, presenting unique developmental brain anatomies and difering tumor morphologies.

• BraTS Sub-Saharan Africa (SSA) Glioma Adewole et al. (2023): 60 cases acquired from diverse, low-resource clinical settings, introducing significant variations in scanner hardware and image quality.

## 4.2. Evaluation Metrics

Image Synthesis Metrics. To assess the quality of the imputed MRI modalities, we utilized three standard Image Quality Metrics (IQM): Normalized Mean Squared Error (NMSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Measure (SSIM).

Given the arbitrary and highly variable intensity ranges inherent to MRI sequences, computing these metrics directly on raw voxel values can yield inconsistent results. Therefore, prior to evaluation, both the predicted volume $\hat { Y }$ and the target volume Y were robustly normalized to a standard [0 1] range using the 1st and 99th percentiles of the target volume intensity distribution Isensee et al. (2021). This step efectively clips hyper-intense artifacts and establishes a reliable dynamic range. Using the normalized volumes, the metrics are computed as follows:

$$
N M S E ( Y , \hat { Y } ) = \frac { \| Y - \hat { Y } \| _ { 2 } ^ { 2 } } { \| Y \| _ { 2 } ^ { 2 } }\tag{6}
$$

$$
P S N R ( Y , { \hat { Y } } ) = 1 0 \cdot \log _ { 1 0 } \left( { \frac { M A X _ { Y } ^ { 2 } } { M S E ( Y , { \hat { Y } } ) } } \right)\tag{7}
$$

$$
S S I M ( Y , \hat { Y } ) = \frac { ( 2 \mu _ { Y } \mu _ { \hat { Y } } + c _ { 1 } ) ( 2 \sigma _ { Y \hat { Y } } + c _ { 2 } ) } { ( \mu _ { Y } ^ { 2 } + \mu _ { \hat { Y } } ^ { 2 } + c _ { 1 } ) ( \sigma _ { Y } ^ { 2 } + \sigma _ { \hat { Y } } ^ { 2 } + c _ { 2 } ) }\tag{8}
$$

where $\| \cdot \| _ { 2 } ^ { 2 }$ denotes the squared $L _ { 2 }$ norm, $M A X _ { Y } = 1$ is the maximum possible pixel value after normalization, $\mu$ and $\sigma$ denote the mean and standard deviation of the images, and $c _ { 1 } , c _ { 2 }$ are standard stabilization constants. NMSE provides a robust measure of absolute intensity deviation, PSNR evaluates the global reconstruction quality relative to the peak signal, and SSIM assesses perceived textural and structural realism.

Segmentation Metrics. To validate the clinical interpretability and functional utility of the synthesized images, we evaluated the performance of the segmentation networks using two complementary metrics. First, spatial overlap is measured by the standard Dice Similarity Coeficient (DSC):

$$
D S C ( \hat { Z } , Z ) = \frac { 2 | \hat { Z } \cap Z | } { | \hat { Z } | + | Z | }\tag{9}
$$

where $\hat { Z }$ and Z represent the predicted and ground truth multi-class tumor masks, respectively.

Second, to rigorously assess boundary precision, we employed the 95th percentile Hausdorf Distance (HD95). Unlike the maximum Hausdorf Distance, which is highly sensitive to single-voxel outliers Taha and Hanbury (2015), HD95 measures the spatial distance between the boundaries of the predicted mask $( \partial \hat { Z } )$ and those of the true mask (∂Z), disregarding the worst 5% of matching errors:

$$
H D 9 5 ( \partial \hat { Z } , \partial Z ) = \operatorname* { m a x } \left( d _ { 9 5 } ( \partial \hat { Z } , \partial Z ) , d _ { 9 5 } ( \partial Z , \partial \hat { Z } ) \right)\tag{10}
$$

where $d _ { 9 5 } ( A , B )$ represents the 95th percentile of the minimum Euclidean distances from all points in contour A to contour B.

## 5. Results

Evaluation of the proposed framework encompasses a rigorous analysis of both synthesis fidelity and clinical utility. Initial experiments establish the performance degradation caused by missing modalities, providing a baseline for the subsequent ablation studies on architecture and loss formulation. Additionally, the results include a comprehensive comparison with state-of-theart models, alongside a zero-shot generalization analysis and a blinded radiological validation to confirm the diagnostic reliability of the synthesized modality.

## 5.1. Impact ofMissing Modalities

Before evaluating the results of CAPS, it is crucial to quantify the clinical and computational impact that the absence of MRI modalities has on automated segmentation systems, detailed in Table 1.

The reference model, trained and evaluated on the full protocol of four real modalities, establishes the upper performance bound with a mean Dice coeficient of 86 65% and a Hausdorf distance (HD95) of 12 07mm. On the other hand, the zero-imputation strategy reveals critical network dependencies on specific sequences, identifying the absence of the T1c sequence as the most critical scenario, where the Dice coeficient for the enhancing tumor (ENH) drastically collapses to 14 02%. Similarly, the missing T2f case disproportionately affects the whole tumor (WT), with its Dice score dropping to 47 66% due to the loss of contrast in the peritumoral edema regions.

Although robust approaches, such as modality dropout training, partially mitigate these declines, these systems operate under a “black-box” paradigm that delivers an isolated segmentation mask without restoring the underlying anatomical volume. This lack of visual evidence prevents neuroradiologists from corroborating the AI-generated segmentation, thus justifying the pressing need for our synthesis framework, which is aimed at preserving structural fidelity alongside diagnostic interpretability.

## 5.2. Ablation Study: Architecture and Loss

To empirically justify the design choices of our synthesis and segmentation framework, we conducted a comprehensive ablation study where all configurations were evaluated under the missing T1c scenario, as it represents the most critical challenge from an anatomical perspective, according to Table 1. The architectural and loss decisions derived from this scenario were subsequently applied identically to the remaining three missing modality cases, with final results reported across all scenarios in Table 5.

Architectural Optimization of G. As detailed in Table 2, we established a baseline using the standard nnU-Net architecture, which yielded a mean Dice coeficient of 70 86% and a HD95 of 22 55 mm. The integration of 3D residual blocks (nnResU-Net) facilitated a more robust training optimization, raising the Dice score to 71 99%. This performance gain is characteristic of residual architectures, which help mitigate gradient degradation in deep 3D networks while preserving high-frequency spatial features. Finally, replacing standard transposed convolutions with a trilinear interpolation decoder efectively mitigated spatial chequerboard artifacts, stabilizing synthesis quality and achieving an optimal NMSE, PSNR and mean dice.

Hybrid Loss Formulation through $S _ { a u x } .$ The quantitative results presented in Table 3 and the visual comparisons in Figure 3 underscore the critical importance of our hybrid loss formulation. As observed in Table 3, optimization relying exclusively on the voxellevel absolute error $( \mathcal { L } _ { 1 } )$ yields the poorest segmentation performance, which suggests that purely intensitybased metrics do not prioritize the preservation of essential anatomical structures, treating all voxels with equal weight regardless of their diagnostic relevance.

The introduction of Dice-based components into the objective function leads to a measurable improvement in the mean Dice score of the synthesized images. However, the most robust validation results are achieved only when combining all three terms: $\mathcal { L } _ { 1 }$ , the Anatomical Feature-Prioritized loss $( \mathcal { L } _ { A F P } )$ , and the $\mathcal { L } _ { D u a l D i c e } .$ which ensures a synergy between global intensity consistency, deep latent feature alignment, and topological boundary integrity.

Qualitatively, Figure 3 demonstrates that $\scriptstyle { \mathcal { L } } _ { 1 } - { \mathrm { d r i v e n } }$ reconstructions sufer from significant blurring, particularly at the tumor margins. While intermediate solutions, such as $\mathcal { L } _ { 1 } + A F P ,$ , succeed in producing welldefined boundaries that closely resemble the ground truth, our proposed total loss configuration $( \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P } +$ $\mathcal { L } _ { D u a l D i c e } )$ consistently provides the superior balance. Specifically for T1c sequences, where the identification of active tumor zones and neoangiogenesis is clinically vital, our framework successfully preserves these intricate anatomical details while simultaneously generating sharp, non-blurry volumes that mirror the target distribution.

Table 1: Quantitative Comparison of Baseline Strategies for Missing Modalities. Segmentation performance (Dice and HD95) evaluated across diferent missing MRI scenarios. The Reference model establishes the upper performance bound, having been trained and evaluated on the complete 4-modality protocol. For the missing scenarios, both Zero Imputation and Synthesis Baseline evaluate this exact same reference model by replacing the absent modality with zeros or a standard $\mathcal { L } _ { 1 }$ -synthesized volume, respectively. In contrast, Trained with Dropout evaluates an independent baseline model specifically trained with modality dropout augmentation to handle incomplete inputs. The best performing baseline strategy for each specific missing scenario is highlighted in bold.
<table><tr><td rowspan="2">Strategy</td><td colspan="4">Dice (↑)</td><td colspan="4">HD95 (↓)</td></tr><tr><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td></tr><tr><td>Reference (All Real)</td><td>81.47</td><td>86.27</td><td>92.22</td><td>86.65</td><td>21.56</td><td>9.67</td><td>4.98</td><td>12.07</td></tr><tr><td>Missing T1n</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero Imputation</td><td>79.60</td><td>83.55</td><td>91.59</td><td>84.91</td><td>22.37</td><td>12.84</td><td>5.71</td><td>13.64</td></tr><tr><td>Trained with Dropout</td><td>79.04</td><td>86.77</td><td>92.40</td><td>86.07</td><td>30.42</td><td>8.31</td><td>4.46</td><td>14.40</td></tr><tr><td>Synthesis Baseline</td><td>79.89</td><td>84.97</td><td>90.87</td><td>85.24</td><td>24.48</td><td>12.60</td><td>8.06</td><td>15.05</td></tr><tr><td>Missing T1c</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero Imputation</td><td>14.02</td><td>43.09</td><td>89.46</td><td>48.86</td><td>70.49</td><td>31.66</td><td>6.61</td><td>36.25</td></tr><tr><td>Trained with Dropout</td><td>58.81</td><td>77.38</td><td>91.95</td><td>76.05</td><td>35.59</td><td>12.56</td><td>4.60</td><td>17.58</td></tr><tr><td>Synthesis Baseline</td><td>52.27</td><td>70.27</td><td>90.03</td><td>70.86</td><td>41.44</td><td>18.83</td><td>7.37</td><td>22.55</td></tr><tr><td>Missing T2w</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero Imputation</td><td>77.26</td><td>79.81</td><td>80.45</td><td>79.17</td><td>27.77</td><td>18.26</td><td>16.34</td><td>20.29</td></tr><tr><td>Trained with Dropout</td><td>79.89</td><td>87.07</td><td>92.26</td><td>86.41</td><td>28.46</td><td>6.43</td><td>4.56</td><td>13.15</td></tr><tr><td>Synthesis Baseline</td><td>78.64</td><td>84.93</td><td>90.68</td><td>84.75</td><td>32.56</td><td>15.33</td><td>14.29</td><td>20.73</td></tr><tr><td>Missing T2f</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero Imputation</td><td>52.48</td><td>58.60</td><td>47.66</td><td>52.91</td><td>55.99</td><td>44.27</td><td>45.31</td><td>48.52</td></tr><tr><td>Trained with Dropout</td><td>79.42</td><td>86.47</td><td>89.91</td><td>85.27</td><td>28.58</td><td>8.08</td><td>5.76</td><td>14.14</td></tr><tr><td>Synthesis Baseline</td><td>75.13</td><td>83.28</td><td>89.09</td><td>82.50</td><td>36.44</td><td>13.68</td><td>8.61</td><td>19.58</td></tr><tr><td>Average (1 MRI missing)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero Imputation</td><td>55.84</td><td>66.26</td><td>69.03</td><td>63.71</td><td>44.16</td><td>26.76</td><td>18.49</td><td>29.80</td></tr><tr><td>Trained with Dropout</td><td>74.29</td><td>84.42</td><td>91.63</td><td>83.45</td><td>30.76</td><td>8.85</td><td>4.85</td><td>14.82</td></tr><tr><td>Synthesis Baseline</td><td>71.48</td><td>80.86</td><td>90.17</td><td>80.84</td><td>33.73</td><td>15.11</td><td>9.58</td><td>19.48</td></tr></table>

Table 2: Generator Architecture (G) Ablation Study. Quantitative performance comparison of diferent generative network architectures for the synthesis task in the missing T1c scenario. The study evaluates the impact of progressively upgrading the baseline nnU-Net by integrating 3D residual blocks (nnResU-Net) and replacing standard transposed convolutions with a trilinear interpolation decoder. Performance is assessed using both downstream clinical segmentation metrics (Dice and HD95) and standard Image Quality Metrics, IQM (NMSE and PSNR). The best results among the evaluated generative architectures are highlighted in bold.
<table><tr><td></td><td colspan="4">Dice (↑)</td><td colspan="4">HD95 (↓)</td><td colspan="2">IQM</td></tr><tr><td>Network</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>NMSE (↓)</td><td>PSNR (↑)</td></tr><tr><td>Reference (All Real)</td><td>81.47</td><td>86.27</td><td>92.22</td><td>86.65</td><td>21.56</td><td>9.67</td><td>4.98</td><td>12.07</td><td>=</td><td>=</td></tr><tr><td>nnU-net</td><td>52.27</td><td>70.27</td><td>90.03</td><td>70.86</td><td>41.44</td><td>18.83</td><td>7.37</td><td>22.55</td><td>0.0261</td><td>27.60</td></tr><tr><td>nnResU-net</td><td>54.47</td><td>71.50</td><td>90.01</td><td>71.99</td><td>38.44</td><td>19.22</td><td>7.47</td><td>21.71</td><td>0.0256</td><td>27.72</td></tr><tr><td>nnResUnet + trilinear decoder</td><td>54.54</td><td>71.46</td><td>90.26</td><td>72.09</td><td>41.13</td><td>17.42</td><td>7.47</td><td>22.01</td><td>0.0241</td><td>28.05</td></tr></table>

Segmentation Strategy on S . Having optimized the image generator G, we evaluated the most efective strategy for integrating these synthetic volumes into the segmentation network S . We compared training under varying missing modality ratios (25%, 50%, 75%) using Random Dropout, where the T1c modality is zeropadded with the given probability, against a generative augmentation approach, where the missing T1c is replaced by our model’s synthetic imputation. All configurations were ultimately evaluated in the scenario where the 3 real MRI modalities are available and the T1c modality must be synthesized.

The underlying hypothesis of the Random Dropout strategy is that, by forcing the network to encounter zero-padded T1c channels during training, it will develop fewer dependencies on this specific modality, theoretically improving robustness when it is missing at inference. However, our results demonstrate that this approach is suboptimal, likely due to the inherent domain shift between the training conditions (zero-padded inputs) and the testing conditions (synthesized inputs).

Instead, we observed that explicitly feeding the segmentation network with synthetically imputed volumes during training consistently yields superior performance. Specifically, training the segmenter from scratch with a 50% probability of replacing the real T1c with its synthetic counterpart during training provided the best overall results, particularly in improving the mean Dice coeficient (75.80%).

Overall Performance Across All Modalities. Having established the optimal generator architecture G, hybrid loss formulation through $S _ { a u x } .$ , and downstream segmentation strategy on S for the missing T1c scenario, we extended the evaluation of our final framework to all other missing modality cases, presented in Table 5.

Table 3: Objective Loss Function Ablation Study. Quantitative evaluation of diferent loss configurations for the optimized generative network G (nnResU-Net with trilinear decoder) in the missing T1c scenario. The baseline pixel-wise $\mathcal { L } _ { 1 }$ loss is progressively augmented with the Anatomical Feature-Prioritized (L ) loss and topological Dice constraints (Dice targeting global region overlap, and Dice focusing on boundary precision). The proposed hybrid objective $( \mathcal { L } _ { 1 } + \bar { \mathcal { L } } _ { A F P } ) .$ + Dice ), obtained with the combined use of Generator G and pretrained segmenter $S _ { a u x }$ , achieves the best overall downstream segmentation performance, obtaining the highest mean Dice score and lowest mean HD95.
<table><tr><td></td><td colspan="4">Dice (↑)</td><td colspan="4">HD95 (↓)</td><td colspan="2">IQM</td></tr><tr><td>Loss</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>NMSE (↓)</td><td>PSNR (↑)</td></tr><tr><td>Reference (All Real)</td><td>81.47</td><td>86.27</td><td>92.22</td><td>86.65</td><td>21.56</td><td>9.67</td><td>4.98</td><td>12.07</td><td></td><td>1</td></tr><tr><td>L1</td><td>54.54</td><td>71.46</td><td>90.26</td><td>72.09</td><td>41.13</td><td>17.42</td><td>7.47</td><td>22.01</td><td>0.0241</td><td>28.05</td></tr><tr><td> $\mathrm { L } 1 + D i c e _ { 1 }$ </td><td>54.83</td><td>73.70</td><td>90.97</td><td>73.17</td><td>39.74</td><td>15.32</td><td>7.01</td><td>20.69</td><td>0.0286</td><td>26.95</td></tr><tr><td> $\mathrm { L } 1 + D i c e _ { 2 }$ </td><td>54.93</td><td>76.99</td><td>91.17</td><td>74.36</td><td>38.21</td><td>13.07</td><td>5.94</td><td>19.07</td><td>0.0290</td><td>26.88</td></tr><tr><td> $\mathrm { L 1 + A F P }$ </td><td>55.33</td><td>73.46</td><td>91.46</td><td>74.42</td><td>37.59</td><td>14.53</td><td>6.26</td><td>19.46</td><td>0.0240</td><td>27.83</td></tr><tr><td> $\mathrm { L 1 } + \mathrm { A F P } + D i c e _ { 1 }$ </td><td>56.10</td><td>75.41</td><td>90.09</td><td>73.87</td><td>38.56</td><td>15.66</td><td>7.47</td><td>20.56</td><td>0.0245</td><td>27.63</td></tr><tr><td> $\mathrm { L 1 } + \mathrm { A F P } + D i c e _ { 2 }$ </td><td>54.80</td><td>76.21</td><td>90.50</td><td>73.84</td><td>45.14</td><td>12.17</td><td>6.64</td><td>21.32</td><td>0.0254</td><td>27.49</td></tr><tr><td> $\mathrm { L 1 } + \mathrm { A F P } + D i c e _ { 1 , 2 }$ </td><td>57.19</td><td>76.18</td><td>90.10</td><td>74.49</td><td>37.22</td><td>12.45</td><td>7.08</td><td>18.92</td><td>0.0259</td><td>27.32</td></tr></table>

![](images/f4aaffee4b88556757eadb301911d09dee0bd44d2482b663cf8143f2f167df52.jpg)  
Figure 3: Visual Comparison of Loss Configurations. Generated T1c sequences under varying loss configurations for the missing T1c scenario. The purely pixel-wise L optimization produces over-smoothed lesions lacking structural definition. While intermediate configurations $( \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P }$ and variations with partial Dice constraints) yield incremental improvements in boundary demarcation, the proposed hybrid loss (L + L + $\mathcal { L } _ { D u a l D i c e } )$ achieves the optimal qualitative balance. It successfully recovers both the sharp anatomical margins and the critical intra-tumoral heterogeneity and the regions of neoangiogenesis, closely mirroring the real T1c target.

The results confirm that the benefits of our methodology are not isolated to T1c, but generalize exceptionally well to any missing sequence. For instance, in the missing T1n and T2w scenarios, the framework achieves mean Dice scores of 85 90% and 85 12%, respectively, alongside high PSNR values (28 99 and 26 18). Even in the missing T2f scenario, which heavily impacts edema delineation, the model successfully restores the peritumoral contrast, yielding a mean Dice of 82 69%. Overall, our unified framework achieves a highly robust global average Dice of 82 38% and an HD95 of 15 96 mm across all missing modality scenarios, solidifying its reliability as a comprehensive imputation tool for incomplete clinical protocols.

## 5.3. Comparison with State-of-the-Art

To comprehensively validate our Synthesis Framework, we compared its performance against recent State-of-the-Art approaches in both image reconstruction quality and downstream segmentation utility.

Image Reconstruction Performance. Table 6 presents a quantitative comparison of synthesis quality. A persistent challenge in this domain is the lack of standardized evaluation protocols on the BraTS dataset; existing literature diverges significantly in terms of target missing modalities, data partitioning (e.g., using the oficial validation set versus reserving an internal 20% held-out split from the training data), and input availability (1- to-1 vs. 3-to-1 synthesis) Li et al. (2024). Furthermore, the limited availability of public source code for several baselines precludes perfectly matched retraining. To ensure the most rigorous and fair evaluation possible under these constraints, we carefully aligned our testing with the specific conditions reported by each baseline.

In the oficial validation split, our volumetric approach significantly outperforms existing 1-to-1 methods (e.g., Pix2Pix Isola et al. (2017), CycleGAN Zhu et al. (2017), EaGAN Yu et al. (2019), ALDM Kim and Park (2024)), achieving an SSIM of 93 97%. While this performance gap naturally underscores the substantial advantage of utilizing the full multi-modal context (3-to-1 mapping) inherently supported by our architecture, it also highlights the capacity of our model to effectively integrate complementary information for highfidelity imputation. To further evaluate our architectural robustness and isolate the efect of input configuration, we compared our method against recent 3-to-1 models, such as AMM-Dif Kebaili et al. (2025a), under their reported evaluation protocol (a 20% heldout split). Evaluated on this corresponding fraction of unseen data, our method maintains a highly competitive SSIM of 93 88%. Even though difusion-based models like AMM-Dif Kebaili et al. (2025a) achieve a marginally higher PSNR, our approach excels in SSIM, demonstrating remarkable structural consistency.

Table 4: Segmenter (S ) Training Strategy Ablation. Quantitative evaluation of diferent training paradigms for the final clinical segmenter under the missing T1c scenario. Using the optimized generator configuration, we explore the impact of varying the probability ratio (25%, 50%, and 75%) of encountering a missing modality during training. The standard Random Dropout (zero-padding) baseline is compared against our proposed synthetic data augmentation, evaluated both by training the segmenter from scratch and by fine-tuning (FT) a pre-trained model. The results demonstrate that training the segmenter from scratch with a 50% probability of synthetic imputation achieves the optimal balance, yielding the highest mean Dice score and validating the utility of generative augmentation for downstream robustness.
<table><tr><td rowspan="2">Training Strategy</td><td colspan="4">Dice (↑)</td><td colspan="4">HD95 (↓)</td></tr><tr><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td></tr><tr><td>Reference (All Real)</td><td>81.47</td><td>86.27</td><td>92.22</td><td>86.65</td><td>21.56</td><td>9.67</td><td>4.98</td><td>12.07</td></tr><tr><td colspan="9">25% Missing Modality Ratio</td></tr><tr><td>Random Dropout</td><td>57.39</td><td>77.10</td><td>90.89</td><td>75.13</td><td>40.19</td><td>13.47</td><td>6.24</td><td>19.97</td></tr><tr><td>Synthetic (from scratch)</td><td>56.98</td><td>77.55</td><td>92.05</td><td>75.53</td><td>35.98</td><td>10.85</td><td>4.44</td><td>17.09</td></tr><tr><td>Synthetic (FT)</td><td>56.76</td><td>76.40</td><td>90.13</td><td>74.43</td><td>37.56</td><td>14.89</td><td>11.44</td><td>21.30</td></tr><tr><td colspan="9">50% Missing Modality Ratio</td></tr><tr><td>Random Dropout</td><td>56.73</td><td>76.86</td><td>90.93</td><td>74.84</td><td>38.69</td><td>13.58</td><td>7.52</td><td>19.93</td></tr><tr><td>Synthetic (from scratch)</td><td>57.95</td><td>77.59</td><td>91.86</td><td>75.80</td><td>36.10</td><td>10.90</td><td>4.63</td><td>17.21</td></tr><tr><td>Synthetic (FT)</td><td>55.10</td><td>75.96</td><td>90.33</td><td>73.80</td><td>39.88</td><td>15.88</td><td>8.70</td><td>21.49</td></tr><tr><td colspan="9">75% Missing Modality Ratio</td></tr><tr><td>Random Dropout</td><td>57.67</td><td>77.18</td><td>90.36</td><td>75.07</td><td>38.81</td><td>14.92</td><td>11.68</td><td>21.80</td></tr><tr><td>Synthetic (from scratch)</td><td>58.25</td><td>76.88</td><td>90.98</td><td>75.37</td><td>37.09</td><td>12.18</td><td>5.48</td><td>18.25</td></tr><tr><td>Synthetic (FT)</td><td>55.59</td><td>75.90</td><td>90.39</td><td>73.96</td><td>40.47</td><td>14.43</td><td>7.67</td><td>20.86</td></tr></table>

Table 5: Overall Performance Across All Missing Modalities. Evaluation of the fully optimized CAPS for every single missing MRI scenario (T1n, T1c, T2w, and T2f). The table reports both clinical segmentation accuracy (Dice and HD95) and Image Quality Metrics, IQM (NMSE and PSNR). Utilizing the optimal generative architecture $G ,$ hybrid loss formulation through the use of $S _ { a u x } ,$ and synthetic data augmentation strategy on S established in the previous ablation studies, the framework demonstrates robust imputation fidelity across all sequences, achieving a global average Dice score of 82 38%.
<table><tr><td rowspan="2"></td><td colspan="4">Dice (↑)</td><td colspan="4">HD95 (↓)</td><td colspan="2">IQM</td></tr><tr><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>ENH</td><td>TC</td><td>WT</td><td>Mean</td><td>NMSE (↓)</td><td>PSNR (↑)</td></tr><tr><td>Missing T1n</td><td>80.58</td><td>85.87</td><td>91.24</td><td>85.90</td><td>22.56</td><td>11.07</td><td>6.02</td><td>13.22</td><td>0.0160</td><td>28.99</td></tr><tr><td>Missing T1c</td><td>57.95</td><td>77.59</td><td>91.86</td><td>75.80</td><td>36.10</td><td>10.90</td><td>4.63</td><td>17.21</td><td>0.0259</td><td>27.32</td></tr><tr><td>Missing T2w</td><td>79.22</td><td>85.46</td><td>90.69</td><td>85.12</td><td>22.99</td><td>11.59</td><td>10.91</td><td>15.16</td><td>0.0530</td><td>26.18</td></tr><tr><td>Missing T2f</td><td>73.58</td><td>83.93</td><td>90.57</td><td>82.69</td><td>36.93</td><td>10.09</td><td>7.68</td><td>18.23</td><td>0.0453</td><td>25.34</td></tr><tr><td>Average</td><td>72.83</td><td>83.21</td><td>91.09</td><td>82.38</td><td>29.65</td><td>10.91</td><td>7.31</td><td>15.96</td><td>0.0351</td><td>26.96</td></tr></table>

Downstream Segmentation Performance. Beyond pure synthesis quality, the ultimate goal of our framework is to provide data that is functionally useful for clinical workflows. Table 7 details the segmentation accuracy of our synthesis-based pipeline against two distinct paradigms: robust segmentation frameworks and other synthesis-based approaches.

As expected, pure robust segmentation architectures (such as mmFormer Zhang et al. (2022) and IM-Fuse Pipoli et al. (2025)) establish the upper numerical bounds, achieving mean Dice scores of 86 93% and 87 34%, respectively. Because these models bypass the complex task of image generation and directly optimize boundary predictions from incomplete data, their purely computational metrics are higher. However, as previously discussed, this comes at the severe cost of clinical interpretability, outputting isolated masks without the underlying anatomical context.

Table 6: State-of-the-Art Comparison of Image Reconstruction Quality. Quantitative synthesis evaluation against existing baselines using standard Image Quality Metrics (IQM), reported as the mean across all evaluated missing modality scenarios within each group. To ensure a fair comparison, methods are grouped according to their specific evaluated missing scenarios and reported data partitions (Oficial Validation set vs. a 20% held-out split). The results demonstrate that CAPS, our proposed 3-to-1 framework, substantially outperforms traditional 1-to-1 methods and achieves highly competitive structural fidelity (highest SSIM) compared to recent multi-modal difusion approaches.
<table><tr><td rowspan="2"></td><td rowspan="2">Eval. Split</td><td colspan="3">IQM</td><td colspan="2">Synthesis Type</td></tr><tr><td>NMSE↓</td><td>PSNR ↑</td><td>SSIM % ↑</td><td>1-to-1</td><td>3-to-1</td></tr><tr><td colspan="7">Evaluated Scenarios: Missing T2w, T2f</td></tr><tr><td>Pix2Pix Isola et al. (2017)</td><td>Official Val.</td><td>0.105</td><td>24.89</td><td>87.05</td><td>×</td><td></td></tr><tr><td>CycleGAN Zhu et al. (2017)</td><td>Official Val.</td><td>0.145</td><td>23.62</td><td>82.85</td><td>X</td><td></td></tr><tr><td>EaGAN Yu et al. (2019)</td><td>Official Val.</td><td>0.103</td><td>24.73</td><td>86.65</td><td>X</td><td></td></tr><tr><td>ALDM Kim and Park (2024)</td><td>Official Val.</td><td>0.089</td><td>25.45</td><td>88.55</td><td>X</td><td></td></tr><tr><td>CAPS</td><td>Official Val.</td><td>0.049</td><td>25.76</td><td>93.97</td><td></td><td>X</td></tr><tr><td colspan="7">Evaluated on All Missing Modality Scenarios (T1n, T1c, T2w, T2f)</td></tr><tr><td>Pix2Pix Isola et al. (2017)</td><td>20% Held-out Split</td><td></td><td>29.70</td><td>75.83</td><td></td><td>X</td></tr><tr><td>AMM-Diff Kebaili et al. (2025a)</td><td>20% Held-out Split</td><td></td><td>31.80</td><td>93.50</td><td></td><td>X</td></tr><tr><td>CAPS</td><td>20% Held-out Split</td><td>0.036</td><td>27.12</td><td>93.88</td><td></td><td>X</td></tr></table>

Evaluating our framework within the interpretable synthesis-based paradigm presents additional challenges, as the majority of existing literature focuses exclusively on reconstruction metrics and does not report downstream segmentation performance. When compared to the available synthesis-based joint models that do provide these metrics, such as the recent work by Ferreira et al. Ferreira et al. (2024), our approach demonstrates a substantial improvement, raising the mean Dice score from 74 67% to 82 38%.

Notably, while pure segmentation frameworks achieve approximately a 5% higher mean Dice score, they operate as “black boxes”, forcing clinicians to blindly trust the automated boundaries without the underlying visual context Zhu et al. (2026). CAPS deliberately trades this purely computational advantage for a critical gain in clinical interpretability. By explicitly enforcing structural and topological consistency via our hybrid loss, we deliver highly verifiable anatomical restorations alongside robust tumor delineations. This approach ensures that neuroradiologists have the necessary visual evidence to validate the predictions of our method.

## 5.4. Cross-Domain Generalization

A critical limitation of many medical deep learning frameworks is their susceptibility to domain shift. Models trained on a specific cohort often experience severe performance degradation when exposed to Out-of-

Table 7: State-of-the-Art Comparison of Segmentation. Evaluation of tumor segmentation accuracy (Dice scores), reported as the average across all four missing modality scenarios (T1n, T1c, T2w, and T2f). Methods are categorized into pure robust segmentation models and interpretable 3-to-1 synthesis-based pipelines. Our framework outperforms existing synthesis-based approaches and remains competitive with leading purely computational segmenters.

<table><tr><td>Method</td><td>ENH↑</td><td>TC↑</td><td>WT↑</td><td>Mean ↑</td></tr><tr><td>Robust Segmentation:</td><td></td><td></td><td></td><td></td></tr><tr><td>Random Dropout</td><td>74.29</td><td>84.42</td><td>91.63</td><td>83.45</td></tr><tr><td>U-HVED Dorent et al. (2019)</td><td>70.88</td><td>80.75</td><td>88.63</td><td>80.08</td></tr><tr><td>mmFormer Zhang et al. (2022)</td><td>80.23</td><td>88.38</td><td>92.20</td><td>86.93</td></tr><tr><td>Sfusion Liu et al. (2023)</td><td>77.95</td><td>86.60</td><td>91.05</td><td>85.20</td></tr><tr><td>IM-Fuse Pipoli et al. (2025)</td><td>80.73</td><td>88.95</td><td>92.35</td><td>87.34</td></tr><tr><td>Synthesis-based:</td><td></td><td></td><td></td><td></td></tr><tr><td>Ferreira et al. (2024)</td><td>68.00</td><td>73.00</td><td>83.00</td><td>74.67</td></tr><tr><td>CAPS</td><td>72.83</td><td>83.21</td><td>91.09</td><td>82.38</td></tr></table>

Distribution (OoD) data Diana-Albelda et al. (2026). To validate the robustness of our framework, we evaluated its zero-shot inference capabilities on three completely unseen datasets from the BraTS challenge: Meningioma LaBella et al. (2023), Pediatric Glioma Kazerooni et al. (2024), and Sub-Saharan African Glioma Adewole et al. (2023).

Each of these datasets introduces unique morphological and technical challenges. Meningiomas are tumors that typically tend to compress rather than infiltrate surrounding brain tissue, presenting a fundamentally different topology than adult gliomas Fathi and Roelcke (2013). Pediatric Gliomas involve developing brains with distinct myelination patterns and tumor presentations Dufau (2022). Finally, the Sub-Saharan African Glioma cohort introduces significant technical variability, featuring scans acquired from diverse hardware and protocols typical of low-resource clinical settings Boonzaier et al. (2025).

Table 8: Zero-Shot Cross-Domain Generalization. Quantitative synthesis evaluation (NMSE and PSNR) on three completely unseen external datasets: Meningioma, Pediatric Glioma, and Sub-Saharan African Glioma. CAPS maintains consistently high image quality across all missing modality scenarios. This demonstrates robust out-of-the-box generalization to morphological and technical domain shifts, entirely without the need for domain-specific adaptation.
<table><tr><td rowspan="2">Missing Modality</td><td colspan="2">Meningioma</td><td colspan="2">Pediatric Glioma</td><td colspan="2">Sub-Saharan Glioma</td></tr><tr><td>NMSE (↓)</td><td>PSNR (↑)</td><td>NMSE (↓)</td><td>PSNR (↑)</td><td>NMSE (↓)</td><td>PSNR (↑)</td></tr><tr><td>4MRI Real</td><td>0.0</td><td>∞</td><td>0.0</td><td>∞</td><td>0.0</td><td>∞</td></tr><tr><td>Missing T1n</td><td>0.0248</td><td>27.72</td><td>0.0216</td><td>28.05</td><td>0.0215</td><td>27.56</td></tr><tr><td>Missing T1c</td><td>0.0505</td><td>25.41</td><td>0.0330</td><td>25.93</td><td>0.0272</td><td>25.91</td></tr><tr><td>Missing T2w</td><td>0.0534</td><td>26.21</td><td>0.0648</td><td>25.20</td><td>0.0507</td><td>25.67</td></tr><tr><td>Missing T2f</td><td>0.0360</td><td>25.63</td><td>0.0266</td><td>26.46</td><td>0.0279</td><td>26.55</td></tr><tr><td>Average</td><td>0.0412</td><td>26.24</td><td>0.0365</td><td>26.41</td><td>0.0318</td><td>26.42</td></tr></table>

As reported in Table 8, the framework demonstrates robust generalization despite these substantial domain shifts. Notably, the average synthesis quality on the three unseen cohorts (PSNR ranging from 26.24 to 26.42; NMSE from 0.0318 to 0.0412) remains closely aligned with the in-distribution Adult Glioma performance reported in Table 5 (PSNR = 26.96; NMSE =

0.0351). Furthermore, the narrow spread across the three external datasets indicates that the fidelity of our method is not contingent on a specific tumor morphology or acquisition protocol. Downstream segmentation metrics are intentionally omitted for these cross-domain cohorts for two reasons. First, the BraTS Meningioma dataset employs a fundamentally diferent tumor annotation protocol, rendering segmentation predictions from our Adult Glioma-trained segmenter S inapplicable. Second, for the Pediatric and Sub-Saharan cohorts, reporting segmentation accuracy would conflate two independent sources of performance variation: synthesis fidelity and segmenter generalization, thereby preventing a clean assessment of the generator’s cross-domain robustness, which is the specific focus of this evaluation.

![](images/c6798d478bf57951731dc9e6ca2d398688a12cf553471eb855aa3e623108f922.jpg)  
Figure 4: Zero-Shot Cross-Domain Generalization. Qualitative comparison of real and synthesized MRI modalities (T1n, T1c, T2w, T2f) across the in-distribution training cohort (Adult Glioma) and three completely unseen Out-of-Distribution datasets: Meningioma, Pediatric Glioma, and Sub-Saharan African Glioma. CAPS consistently reconstructs the missing sequences with high structural fidelity across all scenarios, thereby reproducing domain-specific morphological variations and demonstrating robust generalization capabilities without requiring any domain-specific adaptation.

Qualitatively, Figure 4 corroborates these numerical findings by presenting visual comparisons of real and synthesized volumes across all four modality scenarios (T1n, T1c, T2w, and T2f). The inference is shown for both the in-distribution Adult Glioma cohort and the three unseen OoD domains. As observed, the generated images closely mirror the real anatomical targets. Despite the unique anatomical and technical presentations of each dataset, the framework successfully preserves the overall brain structures and intricate tumor topologies without noticeable degradation.

## 5.5. Clinical Validation

While Image Quality Metrics (IQMs) are standard benchmarks in the synthesis literature, they do not inherently guarantee that a generated image is diagnostically useful or structurally trustworthy for clinicians. To assess the practical clinical value of the synthesized images, we conducted a blinded radiological evaluation.

A senior radiology resident evaluated the generated T1c volumes for the complete Adult Glioma validation set (219 patients). The evaluator was presented with volumes synthesized using both the baseline pixelwise objective $( \mathcal { L } _ { 1 } )$ and our proposed hybrid objective $( \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P } + \mathcal { L } _ { D u a l D i c e } )$ . The assessment was conducted in a blinded manner, without revealing which model generated which image, using the evaluation tool illustrated in Figure 5. For each patient, the assignment of synthesis methods to Model A and Model B was randomly shufled to prevent positional bias. The radiologist scored each volume on a 1-5 scale across two distinct axes: Image Quality (encompassing contrast, noise, and the presence of artifacts) and Clinical Utility.

As illustrated in Figure 6, the $\mathcal { L } _ { 1 }$ baseline Image Quality scores were concentrated between 3 and 4 $( \mu =$ $3 . 6 4 \pm 0 . 9 0 )$ , while the proposed hybrid loss shifted

![](images/94dabe448b81fa8f40a8172ac949be10e4fdeeb24208e751e158b391fff2818d.jpg)  
Figure 5: Blinded Radiological Evaluation Interface. Web-based tool for clinical validation of synthesized T1c volumes, organized into six panels: (1) patient list for case navigation; (2) slice selector with Real/Synth overlay toggle and windowing controls; (3) available input modalities (T1n, T2w, T2f); (4) side-by-side comparison of the real T1c against two anonymized synthetic volumes (Model A and Model B), whose assignment is randomly shufled per patient to prevent positional bias; (5) independent scoring of Image Quality and Clinical Value on a 1–5 scale for each model; and (6) session management for exporting and resuming evaluations. Tool available at https://github.com/Phyrise/NIfTI-to-HTML-Evaluator.

![](images/e534c1097209752203d2c83263e9ba6e14cdfe9e8a984d5662793a84a41229d3.jpg)

![](images/bdbc7881140247ee17e86deac2325bcb88d31b195aab7ec3d912d81f167ae563.jpg)  
Figure 6: Radiologist Evaluation Scores. Distribution of blinded radiologist scores comparing the standard L<sub>1</sub> baseline (blue) against our proposed synthesis generation (green). To facilitate a direct visual comparison, the data is presented as follows: the upper half displays the score distribution for Image Quality, while the lower half displays the scores for Clinical Utility. As observed, the proposed hybrid loss decisively shifts the distribution towards the maximum scores (4-5) in both evaluation dimensions, demonstrating its superior visual fidelity and diagnostic reliability.

the distribution significantly toward scores of 4 and 5 $( \mu = 4 . 4 0 \pm 0 . 6 8 )$ . A consistent trend was observed for Clinical Utility, where the proposed method achieved a score of $3 . 5 1 \pm 1 . 0 7$ , outperforming the baseline’s $3 . 1 2 \pm 1 . 1 6$

To rigorously quantify the significance of these improvements, we conducted paired Wilcoxon signedrank tests across all 219 evaluated patients. The proposed hybrid loss yielded a statistically significant improvement over the $\mathcal { L } _ { 1 }$ baseline in both evaluation axes: Image Quality $( W = 1 0 8 0 , p = 6 . 5 0 \times 1 0 ^ { - 2 4 } , r = 0 . 6 8 )$ and Clinical Utility $( W ~ = ~ 1 8 3 7 , ~ p ~ = ~ 7 . 9 3 \times 1 0 ^ { - 1 0 }$ $r = 0 . 4 2 )$ , with significance maintained after Bonferroni correction for multiple comparisons $( \alpha _ { \mathrm { a d j } } = 0 . 0 2 5 )$ . The efect sizes indicate a large improvement in perceived image quality and a medium improvement in clinical utility. Beyond the higher mean scores, the proposed loss also exhibits a lower standard deviation across both axes, reflecting enhanced robustness across diverse clinical presentations. These results confirm that our anatomy-aware synthesis provides reliable visual evidence to strengthen clinical confidence and support surgical planning.

Limitations & Future Work. Despite these promising results, the current method presents certain limitations, particularly its decoupled two-stage structure and its reliance on a specific pre-trained segmentation network. To address these constraints, future research could focus on the development of a unified, end-to-end architecture that leverages anatomy-prioritized constraints within a single joint framework, potentially further closing the accuracy-interpretability gap. Additionally, other promising avenues for investigation include adapting the framework to accommodate dynamic input configurations (e.g., 1-to-1 or 2-to-1 synthesis) and extending this anatomy-prioritized paradigm to other complex oncological domains, such as multi-parametric prostate Armato III et al. (2018) or liver MRI. Finally, although the clinical validation confirms the diagnostic utility of the synthesized volumes, the assessment was performed by a single radiologist, so future studies may incorporate multiple experts to robustly assess inter-rater reliability.

## 6. Conclusion

In this work, we presented a Clinically-Verifiable Anatomy-Prioritized Synthesis framework (CAPS) to address the critical challenge of missing MRI modalities in neuro-oncological imaging Chrysochoou et al. (2026). Traditional approaches to this problem typically force a compromise between computational accuracy and diagnostic transparency: robust segmenters predict tumor boundaries while leaving clinicians blind to the underlying anatomy Zhang et al. (2022); Pipoli et al. (2025), whereas standard synthesis models reconstruct images that lack the structural fidelity required for clinical trust Longuefosse et al. (2025). By driving our 3D generative architecture with a hybrid objective function $( \mathcal { L } _ { 1 } + \mathcal { L } _ { A F P } + \mathcal { L } _ { D u a l D i c e } )$ , we successfully bridged this gap, enforcing the preservation of deep anatomical features and strict topological boundaries.

Our comprehensive evaluation on the BraTS Adult Glioma dataset Baid et al. (2021) demonstrated that our framework significantly outperforms existing Stateof-the-Art synthesis methods with a highly competitive downstream segmentation mean Dice score of

82 38%. Beyond standard computational metrics, our approach exhibited exceptional zero-shot generalization capabilities across completely unseen and morphologically distinct domains, including Meningioma LaBella et al. (2023), Pediatric Kazerooni et al. (2024), and Sub-Saharan African Adewole et al. (2023) cohorts. Most importantly, a blinded radiological evaluation confirmed that our anatomy-aware synthesis substantially improves both perceived image quality and clinical utility, providing human experts with the verifiable visual context needed to trust downstream automated segmentation.

Funding Statements. This work has been partially supported by the Agencia Estatal de Investigación of Spain under Grant PID2025-171806OB-I00 and by the Regional Government of Madrid of Spain under Grant TEC-2024/COM-322, and by the FPI-UAM grant.

Data Availability The BraTS 2023 Adult Glioma, Meningioma, Pediatric Glioma, and Sub-Saharan African Glioma datasets used in this study are publicly available through the Synapse platform (https: //www.synapse.org/Synapse:syn51156910).

Code Availability The generative framework was developed by adapting the publicly available nnU-Net translation codebase (https://github.com/ Phyrise/nnUNet\_translation), which provides the modified nnU-Net architecture for MR-to-CT synthesis including the AFP loss implementation. The final clinical segmenter S was built upon the official nnU-Net repository (https://github.com/ MIC-DKFZ/nnUNet). The blinded radiological evaluation tool is publicly available at https://github. com/Phyrise/NIfTI-to-HTML-Evaluator

CRediT Authorship Contribution Statement Cecilia Diana-Albelda: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Writing - Original Draft, Writing - Review & Editing, Visualization. Arthur Longuefosse: Conceptualization, Methodology, Software, Validation, Investigation, Writing - Review & Editing, Visualization. Mario Rubén Paz Campos: Investigation, Validation (clinical radiological evaluation). Álvaro García-Martín: Conceptualization, Supervision, Writing - Review & Editing, Funding Acquisition. Jesus Bescos: Conceptualization, Supervision, Writing - Review & Editing, Funding Acquisition.

Declaration of Competing Interests The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Adewole, M., Rudie, J.D., Gbdamosi, A., Toyobo, O., Raymond, C., Zhang, D., Omidiji, O., Akinola, R., Suwaid, M.A., Emegoakor, A., et al., 2023. The brain tumor segmentation (brats) challenge 2023: Glioma segmentation in sub-saharan africa patient population (brats-africa). ArXiv , arXiv–2305.

Al Khalil, Y., Ayaz, A., Lorenz, C., Weese, J., Pluim, J., Breeuwer, M., 2024. Multi-modal brain tumor segmentation via conditional synthesis with fourier domain adaptation. Computerized Medical Imaging and Graphics 112, 102332.

Armato III, S.G., Huisman, H., Drukker, K., Hadjiiski, L., Kirby, J.S., Petrick, N., Redmond, G., Giger, M.L., Cha, K., Mamonov, A., et al., 2018. Prostatex challenges for computerized classification of prostate lesions from multiparametric magnetic resonance images. Journal of Medical Imaging 5, 044501–044501.

Atli, O.F., Kabas, B., Arslan, F., Demirtas, A.C., Yurt, M., Dalmaz, O., Cukur, T., 2024. I2i-mamba: Multimodal medical image synthesis via selective state space modeling. arXiv preprint arXiv:2405.14022 .

Baid, U., Ghodasara, S., Mohan, S., Bilello, M., Calabrese, E., Colak, E., Farahani, K., Kalpathy-Cramer, J., Kitamura, F.C., Pati, S., et al., 2021. The rsnaasnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv preprint arXiv:2107.02314 .

Boonzaier, W.P., Moosa, F., Lebang, K., Jabaar, H., Iorumbur, A., Zhang, D., Raymond, C., 2025. Domain adaptation for adult glioma segmentation in sub-saharan africa: An ensemble of nnu-net v2 and mednext, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 248–261.

Chrysochoou, D., Gandhi, D.B., Adib, S., Familiar, A.M., Vunnava, B., Varshochi, S., Khalili, N., Khalili, N., Ware, J.B., Tu, W., et al., 2026. Aipowered segmentation and prognosis with missing mri in pediatric brain tumors. npj Precision Oncology .

Dar, S.U., Yurt, M., Karacan, L., Erdem, A., Erdem, E., Cukur, T., 2019. Image synthesis in multi-contrast mri with conditional generative adversarial networks. IEEE transactions on medical imaging 38, 2375– 2388.

Diana-Albelda, C., Alcover-Couso, R., García-Martín, Á., Bescos, J., Escudero-Viñolo, M., 2026. Gbt-sam: A parameter-eficient depth-aware model for generalizable brain tumor segmentation on mp-mri. International Journal of Imaging Systems and Technology 36, e70389.

Diana-Albelda, C., García-Martín, Á., Bescos, J., 2025a. A review on deep learning methods for glioma segmentation, limitations, and future perspectives. Journal of Imaging 11, 269.

Diana-Albelda, C., Longuefosse, A., García-Martín, Á., Bescos, J., 2025b. Multi-modal brain mri synthesis with nnu-net: Exploring segmentation performance and cross-modality relationships, in: International Workshop on Simulation and Synthesis in Medical Imaging, Springer. pp. 52–62.

Dorent, R., Haouchine, N., Golby, A., Frisken, S., Kapur, T., Wells, W., 2025. Unified cross-modal medical image synthesis with hierarchical mixture of productof-experts. IEEE Transactions on Pattern Analysis and Machine Intelligence .

Dorent, R., Joutard, S., Modat, M., Ourselin, S., Vercauteren, T., 2019. Hetero-modal variational encoder-decoder for joint modality completion and segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 74–82.

Dufau, H., 2022. White matter tracts and difuse lowergrade gliomas: the pivotal role of myelin plasticity in the tumor pathogenesis, infiltration patterns, functional consequences and therapeutic management. Frontiers in Oncology 12, 855587.

Fathi, A.R., Roelcke, U., 2013. Meningioma. Current neurology and neuroscience reports 13, 337.

Ferreira, A., Luijten, G., Puladi, B., Kleesiek, J., Alves, V., Egger, J., 2024. Brain tumour removing and missing modality generation using 3d wdm. arXiv preprint arXiv:2411.04630 .

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2014. Generative adversarial nets. Advances in neural information processing systems 27.

Guo, X., Shi, Y., Liu, D., Li, Y., Chen, W., Wang, Y., Wang, Y., Xing, H., Xia, Y., Li, J., et al., 2023. Clinical updates on gliomas and implications of the 5th edition of the who classification of central nervous system tumors. Frontiers in Oncology 13, 1131642.

Havaei, M., Guizard, N., Chapados, N., Bengio, Y., 2016. Hemis: Hetero-modal image segmentation, in: International conference on medical image computing and computer-assisted intervention, Springer. pp. 469–477.

Hu, M., Maillard, M., Zhang, Y., Ciceri, T., La Barbera, G., Bloch, I., Gori, P., 2020. Knowledge distillation from multi-modal to mono-modal segmentation networks, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 772–781.

Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H., 2021. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18, 203–211.

Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., Jaeger, P.F., 2024. nnu-net revisited: A call for rigorous validation in 3d medical image segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 488–498.

Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A., 2017. Imageto-image translation with conditional adversarial networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125– 1134.

Kazerooni, A.F., Khalili, N., Liu, X., Haldar, D., Jiang, Z., Anwar, S.M., Albrecht, J., Adewole, M., Anazodo, U., Anderson, H., et al., 2024. The brain tumor segmentation (brats) challenge 2023: focus on pediatrics (cbtn-connect-dipgr-asnr-miccai brats-peds). ArXiv , arXiv–2305.

Kebaili, A., Lapuyade-Lahorgue, J., Vera, P., Ruan, S., 2025a. Amm-dif: Adaptive multi-modality difusion network for missing modality imputation, in: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), IEEE. pp. 1–4.

Kebaili, A., Lapuyade-Lahorgue, J., Vera, P., Ruan, S., 2025b. Multi-modal mri synthesis with conditional latent difusion models for data augmentation in tumor segmentation. Computerized Medical Imaging and Graphics 123, 102532.

Kim, J., Park, H., 2024. Adaptive latent difusion model for 3d medical image to image translation: Multimodal magnetic resonance imaging study, in: Proceedings of the IEEE/CVF Winter conference on applications of computer Vision, pp. 7604–7613.

LaBella, D., Adewole, M., Alonso-Basanta, M., Altes, T., Anwar, S.M., Baid, U., Bergquist, T., Bhalerao, R., Chen, S., Chung, V., et al., 2023. The asnrmiccai brain tumor segmentation (brats) challenge 2023: Intracranial meningioma. arXiv preprint arXiv:2305.07642 .

Li, H.B., Conte, G.M., Hu, Q., Anwar, S.M., Kofler, F., Ezhov, I., van Leemput, K., Piraud, M., Diaz, M., Cole, B., et al., 2024. The brain tumor segmentation (brats) challenge 2023: Brain mr image synthesis for tumor segmentation (brasyn). ArXiv , arXiv–2305.

Liu, Z., Wei, J., Li, R., Zhou, J., 2023. Sfusion: Selfattention based n-to-one multimodal fusion block, in: International conference on medical image computing and computer-assisted intervention, Springer. pp. 159–169.

Longuefosse, A., Gonzalez, J.S., Baldacci, F., Seita, J., 2026. From intensity to anatomy: Optimizing nnunet for mr-to-ct translation on synthrad2025, in: 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI), IEEE. pp. 1–4.

Longuefosse, A., Denis de Senneville, B., Dournes, G., Benlala, I., Baldacci, F., Desbarats, P., 2025. Anatomical feature-prioritized loss for enhanced mr to ct translation. Physics in Medicine & Biology 70, 145012.

Longuefosse, A., et al., 2024. Adapted nnu-net: A robust baseline for cross-modality synthesis and medical image inpainting, in: Simulation and Synthesis in Medical Imaging: 8th International Workshop, SASHIMI 2023, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 8, 2023, Proceedings, Springer. pp. 24–33.

Meng, X., Sun, K., Xu, J., He, X., Shen, D., 2024. Multi-modal modality-masked difusion network for brain mri synthesis with random modality missing. IEEE Transactions on Medical Imaging 43, 2587– 2598.

Menze, B.H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., Burren, Y., Porz, N., Slotboom, J., Wiest, R., et al., 2014. The multimodal brain tumor image segmentation benchmark (brats).

IEEE transactions on medical imaging 34, 1993– 2024.

Nie, D., Trullo, R., Lian, J., Wang, L., Petitjean, C., Ruan, S., Wang, Q., Shen, D., 2018. Medical image synthesis with deep convolutional adversarial networks. IEEE Transactions on Biomedical Engineering 65, 2720–2730.

Pipoli, V., Saporita, A., Marchesini, K., Grana, C., Ficarra, E., Bolelli, F., 2025. Im-fuse: A mamba-based fusion block for brain tumor segmentation with incomplete modalities, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 225–235.

Sharma, A., Hamarneh, G., 2019. Missing mri pulse sequence synthesis using multi-modal generative adversarial network. IEEE transactions on medical imaging 39, 1170–1183.

Shi, B., 2024. On the hyperparameters in stochastic gradient descent with momentum. Journal of Machine Learning Research 25, 1–40.

Taha, A.A., Hanbury, A., 2015. Metrics for evaluating 3d medical image segmentation: analysis, selection, and tool. BMC medical imaging 15, 29.

Tavse, S., Varadarajan, V., Bachute, M., Gite, S., Kotecha, K., 2022. A systematic literature review on applications of gan-synthesized images for brain mri. Future Internet 14, 351.

Weller, M., Wick, W., Aldape, K., Brada, M., Berger, M., Pfister, S.M., Nishikawa, R., Rosenthal, M., Wen, P.Y., Stupp, R., et al., 2015. Glioma. Nature reviews Disease primers 1, 1–18.

Yu, B., Zhou, L., Wang, L., Shi, Y., Fripp, J., Bourgeat, P., 2019. Ea-gans: edge-aware generative adversarial networks for cross-modality mr image synthesis. IEEE transactions on medical imaging 38, 1750– 1762.

Zhang, Y., He, N., Yang, J., Li, Y., Wei, D., Huang, Y., Zhang, Y., He, Z., Zheng, Y., 2022. mmformer: Multimodal medical transformer for incomplete multimodal learning of brain tumor segmentation, in: International conference on medical image computing and computer-assisted intervention, Springer. pp. 107–117.

Zhang, Y., Peng, C., Wang, Q., Song, D., Li, K., Zhou, S.K., 2024. Unified multi-modal image synthesis for

missing modality imputation. IEEE Transactions on Medical Imaging 44, 4–18.

Zhu, J.Y., Park, T., Isola, P., Efros, A.A., 2017. Unpaired image-to-image translation using cycleconsistent adversarial networks, in: Proceedings of the IEEE international conference on computer vision, pp. 2223–2232.

Zhu, S., Chen, Y., Chen, W., Jiang, S., Zhou, G., Wang, Y., Qin, F., Wang, C., Tian, Q., 2026. No modality left behind: Adapting to missing modalities via knowledge distillation for brain tumor segmentation. Medical Image Analysis , 104108.