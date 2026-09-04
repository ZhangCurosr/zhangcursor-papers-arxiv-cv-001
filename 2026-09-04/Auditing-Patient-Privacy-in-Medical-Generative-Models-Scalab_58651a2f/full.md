# Auditing Patient Privacy in Medical Generative Models: Scalable Memorization Detection with DeepSSIM++

Antonio Scardace<sup>a,∗</sup>, Francesco Guarnera<sup>a</sup>, Sebastiano Battiato<sup>a</sup> and Daniele Ravì<sup>b</sup>

<sup>a</sup>University of Catania, Department of Mathematics and Computer Science, Catania, Italy

<sup>b</sup>University of Messina, MIFT Department, Messina, Italy

## A R T I C L E I N F O

Keywords: Medical Imaging Artificial Intelligence Generative Models Memorization Privacy Preserving

## A BS T RA C T

While deep generative models ofer new opportunities for medical image synthesis and data sharing, their ability to memorize and reproduce training samples raises serious concerns about patient confidentiality. Detecting such memorization at scale remains challenging: traditional pixel-based metrics are sensitive to generation artifacts, whereas generic embedding-based metrics often lack the anatomical sensitivity required for medical data. To address this challenge, we introduce DeepSSIM++, a self-supervised similarity metric for scalable memorization auditing in medical generative models. By leveraging multi-scale feature aggregation and anatomy-preserving augmentations, DeepSSIM++ learns an embedding space where cosine similarity approximates the Structural Similarity Index (SSIM), eliminating the need for exact pixel-level registration. Compared with state-of-the-art baselines, DeepSSIM++ achieves an average Macro F1 improvement of 33 percentage points under ideal alignment and 46 percentage points under realistic spatial and intensity perturbations. Furthermore, it accelerates large-scale similarity computation by several orders of magnitude compared with analytical SSIM. By combining anatomical sensitivity and computational eficiency, DeepSSIM++ provides an open-source tool for scalable memorization auditing in medical generative AI. Code and data are publicly available at: https://github.com/brAIn-science/DeepSSIM.

## 1. Introduction

Generative models are increasingly adopted in medical imaging, enabling applications such as image reconstruction (Ahishakiye et al., 2021), modality translation (Sargood et al., 2026), synthetic data generation for downstream clinical tasks such as disease progression modeling (Puglisi et al., 2025), and data augmentation (Seo et al., 2025). Among the most studied families of image generative models are Variational Autoencoders (Kingma and Welling, 2014), Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), Difusion Models (Ho et al., 2020; Rombach et al., 2022), and Flow Matching models (Lipman et al., 2023). However, despite their remarkable success, a fundamental question remains: do these models truly learn the underlying data distribution, or do they simply memorize their training data? Ideally, a generative model should synthesize novel images that preserve the statistical properties of the training distribution without reproducing specific examples. When this property is violated and generated samples closely replicate training instances, the model is considered to exhibit memorization rather than generalization. The boundary between these two behaviors is governed by multiple interacting factors, including model architecture, training data characteristics, and optimization dynamics (Bonnaire et al., 2026).

This distinction becomes particularly important in privacy sensitive applications, where synthetic data is often proposed as a privacy-preserving alternative to sharing realworld data. This assumption relies on the expectation that generative models capture the statistical properties of the training distribution without retaining information specific to individual training instances. For example, Jahanian et al. (2022) have suggested that difusion models may improve privacy by generating synthetic samples rather than directly exposing real data. However, if a model memorizes and reproduces training examples, synthetic generation no longer guarantees privacy and may instead introduce risks of information leakage. The consequences are particularly severe in healthcare applications, where clinical data often contain fine-grained information that may act as subjectspecific identifiers. Structural brain MRI scans, for example, feature highly detailed morphological patterns, such as cortical gyri or lesion boundaries, that may uniquely identify individual patients (Chauvin et al., 2020). In this context, memorization compromises patient confidentiality, reduces trust in AI systems, and slows the clinical adoption of otherwise promising generative AI technologies.

Reliable memorization assessment is therefore essential for comparing models and providing measurable evidence of privacy and reliability. Defining memorization, however, remains challenging because similarity can be evaluated at diferent levels. While natural images may exhibit semantic similarity despite substantial pixel-level diferences, medical images require stricter criteria based on identity-preserving anatomical structures. In our work, we consider and extend the memorization definition introduced by Dar et al. (2023), which was specifically designed for the medical imaging domain. Here, a generated sample is considered memorized if it closely reproduces a training instance up to minor afine transformations. Since our objective is to assess the risk of patient re-identification, we generalize this notion beyond afine transformations to include realistic artifacts and appearance distortions, such as intensity variations and noise. These perturbations may alter the visual appearance of a sample while preserving the subject-specific anatomical information that enables re-identification. Current approaches face significant challenges in identifying this type of memorization: they either rely on generic visual embeddings that overlook subtle anatomical similarities or require computationally expensive image comparisons that limit their scalability.

To address these limitations, we propose DeepSSIM++, a self-supervised metric designed to detect and quantify memorization in medical generative models. Existing pixelbased metrics often fail under minor spatial misalignment, and generic visual embeddings overlook subtle anatomical correspondence. Our approach is developed to bridge these gaps by learning a transformation-invariant structural feature space while enabling scalable memorization detection with low computational cost. We focus our empirical evaluation on 2D structural brain MRI, a domain representing a particularly critical scenario for patient confidentiality, as prior work has shown that the high individual specificity of anatomical morphology enables patient re-identification with 100% accuracy (Chauvin et al., 2020).

More specifically, we extend our previous study (Scardace et al., 2026) along four key directions:

• We introduce a refined annotation protocol for the memorization benchmark, including revised labels and rigorous inter-rater agreement analysis.

• We enhance DeepSSIM through several architectural and optimization improvements, including multi-scale feature aggregation and Layer-wise Learning Rate Decay (LLRD) (Bu et al., 2024), leading to improved similarity estimation accuracy.

• We provide a rigorous validation of DeepSSIM++ through extensive ablation studies, explainability analysis using LayerCAM (Jiang et al., 2021).

• We demonstrate that DeepSSIM++ consistently outperforms existing memorization metrics and achieves orders-of-magnitude faster similarity computation compared with analytical SSIM, enabling scalable memorization auditing on large synthetic datasets.

The remainder of this paper is organized as follows: Section 2 reviews related work on memorization and similarity metrics in generative modeling. Section 3 describes the proposed framework, the memorization taxonomy, and the construction of our benchmark dataset. Section 4 presents the experimental setup together with a comprehensive quantitative, qualitative, and computational evaluation of the proposed approach. Finally, Section 5 discusses the practical implications and limitations of our findings.

## 2. Related Work

Recent studies suggest that memorization is not an inherent property of a specific generative architecture. Instead, it emerges from the interplay of multiple factors, including the composition of the training data, the model architecture, and the optimization process.

## 2.1. Memorization in Generative Models

Early GAN literature linked memorization to mode collapse, where generators produce a restricted subset of samples rather than learning the full target distribution (Arora et al., 2018; Feng et al., 2021). This reduced diversity can give a misleading impression of generalization while mask ing training-set memorization and identity leakage. These concerns become even more significant for difusion models (Yoon et al., 2023; Bonnaire et al., 2026), whose training objective difers fundamentally from that of GANs. Unlike GANs, which learn a generator that maps noise to data through adversarial training, difusion models are trained using denoising score matching (DSM) (Vincent, 2011), which explicitly learns to reconstruct corrupted training instances. Recent theoretical analyses (Gu et al., 2025) have shown that, with high model capacity, the optimal DSM solution can converge toward the empirical data distribution rather than the underlying population distribution, leading to significantly higher rates of training-data memorization. Empirical studies further support this concern, demonstrating that difusion models can expose more training data than GANs (Somepalli et al., 2023; Carlini et al., 2023).

## 2.2. Memorization in Medical Generative Models

Within the medical domain, empirical investigations indicate that Latent Difusion Models (LDMs) (Rombach et al., 2022) are highly susceptible to training set reproduction, memorizing up to 59% of their training instances under specific conditions (Dar et al., 2023). More recently, largescale multi-modal evaluations across diverse 2D and 3D imaging cohorts revealed that over 37% of patient records could be identified as memorized, with nearly 70% of generated samples corresponding to direct patient copies. Crucially, these findings demonstrate that memorization is not exclusive to DSM-based approaches, as alternative generative models were also found to reproduce training instances. However, their lower synthesis quality reduced the perceptual recognizability of these copies, suggesting that high generative fidelity plays a critical role in whether memorized content can be perceptually and radiologically identified. This finding was supported not only by automatic similarity measures but also by expert radiological assessment (Dar et al., 2026).

2.3. What are the Major Causes of Memorization? Training Data Properties. The statistical properties of the training dataset are a primary determinant of memorization behavior. Empirical evidence (Feng et al., 2021; Bonnaire et al., 2026; Gu et al., 2025; Dar et al., 2026) indicates that dataset size and diversity strongly influence memorization, as models trained on smaller or less variable cohorts are more likely to replicate training instances. Similarly, the presence of duplicated samples has been identified as a major trigger for memorization, since repeated exposure increases the tendency of generative models to reproduce training examples rather than generalize (Somepalli et al., 2023; Carlini et al., 2023). Yoon et al. (2023) demonstrated that lower-resolution data and structurally simpler distributions are more susceptible to memorization. Additionally, they show that severe class imbalance further exacerbates this behavior, making under-represented minority classes more likely to be memorized than majority classes.

Model Design and Capacity. The architecture and capacity of a generative model also influence its tendency to memorize training samples. In general, high-capacity models with a large number of parameters exhibit greater memorization potential, whereas under-parameterized models are more constrained to capture broader patterns in the data distribution (Yoon et al., 2023; Dar et al., 2026). More detailed analyses (Bonnaire et al., 2026; Gu et al., 2025) have shown that increasing layer width is associated with increased memorization capacity. In contrast, increasing network depth exhibits a non-monotonic relationship: while adding layers initially enhances a model’s ability to memorize, excessive depth may hinder optimization and reduce memorization rates. Beyond overall scale and depth, specific architectural components can also influence memorization behavior. Recent studies (Gu et al., 2025; Dar et al., 2026) have identified skip connections and conditioning mechanisms to significantly influence memorization. In particular, conditional difusion models exhibit higher memorization rates than unconditional models, and even random or uninformative conditioning signals can increase the tendency to replicate training instances.

Training and Optimization Strategies. The optimization process also strongly influences when and how memorization occurs. Bonnaire et al. (2026); Gu et al. (2025); Dar et al. (2026) have shown that models often undergo distinct learning phases: an early training stage characterized by high-quality generation, followed by a later stage in which memorization gradually emerges. Consequently, early stopping can act as an implicit regularization mechanism, whereas longer training schedules generally tend to increase memorization rates. Training hyperparameters can further modulate memorization behavior. In Gu et al. (2025), larger batch sizes have been associated with increased memorization, whereas stronger regularization through weight decay can reduce memorization behavior. In medical generative models, training duration has also been identified as a key factor influencing memorization (Dar et al., 2026).

## 2.4. Privacy Risks and Mitigation Strategies

Memorization remains challenging to detect because models can generate visually convincing samples while still leaking training information, either by reproducing training examples directly or by preserving sensitive patterns learned during training (Carlini et al., 2023; Yoon et al., 2023).

Such leakage may remain hidden during standard inference and only emerge through dedicated attacks. These include membership inference attacks, which aim to determine whether a specific sample was part of the training set (Shokri et al., 2017; Carlini et al., 2022); model inversion approaches, which attempt to reconstruct sensitive information from model outputs or internal representations (Fredrikson et al., 2015; Zhang et al., 2020); and data extraction attacks, which have demonstrated that memorized training examples can be recovered directly from generative models at scale (Carlini et al., 2019, 2021; Balle et al., 2021; Nasr et al., 2025; Chen et al., 2026).

These findings have motivated the development ofprivacypreserving training strategies aimed at reducing the amount of sensitive information retained by generative models. In particular, Diferential Privacy (Abadi et al., 2016; Ziller et al., 2021) approaches aim to limit the amount of retained information by introducing controlled perturbations during optimization, whereas Federated Learning (McMahan et al., 2017; Sheller et al., 2020) approaches reduce the need for centralized data sharing by keeping patient data distributed across multiple institutions. However, evaluating whether such approaches efectively reduce memorization without destroying clinical utility remains challenging, making reliable, transformation-invariant quantification metrics essential for auditing privacy–utility trade-ofs.

## 2.5. How Can Memorization Be Quantified?

Recent literature has proposed diferent memorization metrics that typically follow a two-step strategy: first, a similarity function is used to measure correspondence between generated and training images; second, similarity scores are converted into memorization decisions through predefined thresholds or ranking criteria.

## 2.5.1. Embedding-Based Metrics

Several approaches have been proposed to quantify memorization by combining image similarity measures with specific classification strategies. In Somepalli et al. (2023), memorization is assessed by embedding both real and synthetic images using the pre-trained Self-Supervised for image Copy Detection (SSCD) model (Pizzi et al., 2022), and computing cosine similarities between the resulting embeddings. The resulting similarity scores are then converted into a memorization estimate by considering the 95th percentile of the similarity distribution as an indicator of the most similar generated-training pairs. In Chen et al. (2024), the authors note that this strategy may underestimate memorization in heavy-tailed similarity distributions. More recently, Chen et al. (2026) introduced two complementary metrics: the Average Memorization Score (AMS), which estimates the overall prevalence of memorization by measuring the average proportion of generated images that match training instances across predefined similarity thresholds, and the Unique Memorization Score (UMS), which quantifies the diversity of memorized samples by measuring how many distinct training examples are recovered from the generated set. Their framework computes similarity using normalized $\ell _ { 2 }$ distance for low-resolution datasets, following Carlini et al. (2023), and SSCD embeddings for high-resolution datasets, using two fixed thresholds to assign image pairs to predefined similarity categories. However, these approaches were primarily developed for natural images and rely on generic similarity measures, which may not capture the finegrained anatomical structures required to assess memorization in medical imaging. In particular, SSCD is pre-trained on large-scale natural image datasets and is not explicitly designed to identify subtle domain-specific structures.

A related line of research approaches memorization as a duplication detection problem. For instance, SemDeDup (Abbas et al., 2023) leverages embeddings from pretrained foundation models, such as BioMedCLIP (Zhang et al., 2024), by first clustering samples in the embedding space and then identifying semantically duplicate pairs within each cluster based on a similarity threshold. Although originally proposed to improve data eficiency by removing redundant training samples, semantic deduplication frameworks can also be adapted to estimate memorization by identifying generated images that closely match training examples. However, general biomedical foundation models are typically optimized for high-level semantic tasks rather than fine-grained structural correspondence, making them less suitable for detecting strict anatomical duplication where preservation of individual structures is required.

## 2.5.2. Domain-Specific Similarity Learning

As an alternative, Dar et al. (2026) proposes training a feature extractor using contrastive learning directly on the same training set as the generative model under study. This approach avoids relying on external pre-trained representations, which may limit the detection of medically relevant memorization patterns. However, even domain-specific approaches based on learned representations may face challenges when detecting strict anatomical duplication.

## 2.5.3. Pixel-Based Metrics

Pixel-based metrics provide stronger structural correspondence but remain sensitive to spatial misalignment, causing direct pixel-space comparisons to fail under afine transformations (Zhang et al., 2018). This motivates the need for robust metrics capable of combining structural sensitivity with transformation invariance.

## 2.6. Similarity Metrics in Medical Imaging

Unlike natural image analysis, duplicate detection in medical images requires the precise preservation of anatomical structures, with near pixel-wise correspondence between images. To capture these clinically relevant details, the Structural Similarity Index (SSIM) (Wang et al., 2004) is one of the standard similarity measures.

Breger et al. (2025) examined SSIM’s behavior by comparing original brain MRI scans with lower-quality reconstructions. Their results revealed that SSIM scores remained relatively stable even as image quality visibly deteriorated.

While this insensitivity to degradation is a limitation for Image Quality Assessment tasks, it is highly advantageous for duplicate detection, enabling the identification of memorized anatomical features despite the generation artifacts. However, SSIM sufers from two critical bottlenecks: it assumes strict spatial alignment, which is often violated by generation artifacts, and its exhaustive pairwise comparisons become computationally prohibitive at scale.

Recent evidence further highlights the importance of robustness to image transformations for accurate memorization detection in realistic generation settings. Dar et al. (2026) showed that embedding-based pipelines can identify memorized samples even when generated images undergo transformations such as rotations, flips, and contrast variations. While embedding-based methods provide the scalability required for large-scale evaluation and robustness to such perturbations, they often lack the structural sensitivity needed to capture subtle anatomical correspondences, motivating the development of specialized hybrid approaches.

These observations motivate DeepSSIM++, which combines the structural sensitivity required for medical image memorization analysis with the eficiency, scalability, and transformation robustness of learned representations, enabling large-scale evaluation of generative models.

## 2.7. Legal and Regulatory Implications

Beyond technical challenges, memorization in generative models raises significant legal, regulatory, and ethical concerns. Recent debates on copyright infringement in AIgenerated images have questioned whether generative models unlawfully reproduce protected material (Gillotte, 2020; Butterick and Joseph Saveri Law Firm, 2023). Similarly, privacy concerns have emerged regarding the unauthorized sharing of clinical data with proprietary models (Agarwal et al., 2024). These cases show that memorization extends beyond an academic issue, raising fundamental questions of ownership, consent, and legal compliance. The key concern is not only whether models generate novel content, but whether their outputs or training procedures reproduce protected information in ways that create liability.

In Europe, these concerns are increasingly reflected in regulatory frameworks such as the General Data Protection Regulation (GDPR) (European Union, 2016) and the EU AI Act (European Union, 2024), which establish requirements for transparency, risk management, and the trustworthy deployment of AI systems. In contrast, the United States regulatory framework, including the Health Insurance Portability and Accountability Act (HIPAA) (U.S. Department of Health and Human Services, 2003), primarily focuses on the protection of individually identifiable health information. While the EU AI Act explicitly addresses emerging challenges introduced by AI technologies, HIPAA provides limited guidance on the specific risks associated with memorization in generative AI systems.

![](images/461b471b828a902652ed8a2f1ba6296cc2081dc41e182a9c430047895c3538fd.jpg)  
Figure 1: Training Pipeline: The figure illustrates the DeepSSIM++ self-supervised training process, which is divided as follows: (i) Inputs: real and synthetic MRI scans are first preprocessed. (ii) Augmentation: random anatomical-preserving augmentations are applied before feature extraction to improve robustness to image variability. (iii) Embeddings: images are mapped into a lowerdimensional embedding space using a feature extractor $f _ { \theta } .$ . (iv) Model Optimization: the training loss minimizes the discrepancy between the embedding cosine similarity and the corresponding SSIM target, optimizing the feature extractor parameters �.

## 3. Methods

In this section, we first introduce the datasets used to train and evaluate our framework. We then present the two main components of the proposed DeepSSIM++ pipeline for memorization detection. The first component, illustrated in Figure 1, is a self-supervised model that learns a representation space where the cosine similarity between two image embeddings approximates the SSIM of the corresponding image pairs in the pixel domain. During training, SSIM targets are computed from spatially aligned image pairs to guide the learning of their structural similarity. The second component is a thresholding strategy to classify image pairs in this embedding space as Unrelated, Near-Duplicates, or Exact Duplicates based on their similarity scores.

## 3.1. Datasets

We construct three datasets, each serving a specific role in our framework. The first dataset, denoted by , consists of real brain MRI scans and is used to train a standard LDM used to generate synthetic images under memorization-promoting conditions. The second dataset, denoted by , comprises the synthetic MRI slices generated by this LDM, which potentially contain memorized instances. The third dataset, denoted by , consists of real– synthetic image pairs with varying levels of similarity drawn from  and . This dataset forms the core focus of our analysis and is used to train and evaluate DeepSSIM++.

We note that the specific generative architecture is not the subject of our investigation. In fact, the LDM considered here is utilized exclusively as a practical case study to demonstrate our primary contribution: a robust and scalable methodology for detecting memorization. Because our detection pipeline operates entirely on the generated images rather than internal model weights, our work provides a general framework suitable for evaluating any other medical generative architecture.

## 3.1.1. Real MRI Dataset ()

This dataset comprises 2,583 cross-sectional brain MRI scans from two public datasets: IXI (1,122 scans) and CoRR (1,461 scans). It includes 2,024 T1-weighted (T1w) and 559 T2-weighted (T2w) acquisitions. The data are randomly split into a training set $D _ { \mathrm { t r a i n } }$ (85%) to train the LDM, and a validation set (15%) used for early stopping.

MRI Preprocessing. All MRI scans in this dataset are processed using a standardized pipeline consisting of bias-field correction (Tustison et al., 2021), skull stripping (Isensee et al., 2019), afine registration to MNI space (Tustison et al., 2021), and intensity normalization (Shinohara et al., 2014). From each preprocessed 3D volume, we extract the central axial slice, resized to 112 × 136 pixels.

## 3.1.2. Synthetic Dataset ()

The second dataset consists of synthetic MRI slices generated by the LDM. To increase the likelihood of generating memorized instances, we follow Dar et al. (2026), who showed that using a synthetic set substantially larger than the real training set facilitates the creation of memorized samples. Specifically, for each real image present in $D _ { \mathrm { t r a i n } } ,$ we generate 30 synthetic samples, resulting in a synthetic dataset  containing 65,850 images. Additionally, to further increase training-data memorization, we incorporate three primary risk factors identified in Gu et al. (2025): (i) controlled data duplication by randomly injecting 1–15 copies of each training image, (ii) an extended training schedule, and (iii) image-specific text conditioning, where each sample is associated with a unique textual description. Each prompt follows the template below:

“middle axial slicefrom a <MRI SEQUENCE> brain MRI (image ID <MRI\_UID>) acquired from a <AGE>-year-old <BIOLOGICAL SEX>, with subject ID <SID>,from the <DATASET NAME> dataset.”

![](images/c68adb27a4625cb611d88e20a6c421ff007ce9cf8c2a1da695c0443c8f9c91c8.jpg)

![](images/3d024626db21620a78f69fce42b50baac5483b9780391d794cd9aa6ec15fc4ac.jpg)

![](images/a6c52aae365d4891164d8c809ba0c8dd91ab49e860a78db6525548a74a0ffb1b.jpg)  
Figure 2: Distribution of Ground-Truth SSIM Scores. The figure illustrates the distribution of image pairs across SSIM bins (width 0.1) for the three dataset splits. Panels (A) and (B) show the training and validation sets, $\mathcal { P } _ { \mathrm { t r a i n } }$ and $\mathcal { P } _ { \mathrm { v a l i d } }$ , respectively, both exhibiting approximately balanced distributions obtained by stratified random sampling. Panel (C) presents the manually annotated test set $\mathcal { P } _ { \mathrm { t e s t } }$ , which is unbalanced due to the hybrid sampling strategy.

The resulting text embeddings are extracted using Pub-MedBERT (Gu et al., 2021) and incorporated into the LDM through a cross-attention mechanism.

## 3.1.3. Real–Synthetic Pair Dataset ()

The third dataset, denoted by , consists of real–synthetic image pairs. Specifically, each real image $I _ { a } \in \mathcal { D } _ { \operatorname { t r a i n } }$ is paired with every synthetic image $I _ { b } \in \ S$ , resulting in $| D _ { \mathrm { t r a i n } } | \times | S | ~ = ~ 2 , 1 9 5 \times 6 5 , 8 5 0 ~ = ~ 1 4 4 , 5 4 0$ , 750 pairs. Formally, this dataset is then defined as:

$$
\mathcal { P } = \{ ( I _ { a } , I _ { b } ) \ : | \ : I _ { a } \in \mathcal { D } _ { \mathrm { t r a i n } } , \ : I _ { b } \in \mathcal { S } \} .\tag{1}
$$

The dataset  is used to obtain three disjoint subsets: $\mathcal { P } _ { \mathrm { t r a i n } }$ , used to train our feature extractor $f _ { \theta } ; \mathcal { P } _ { \mathrm { v a l i d } } .$ , used for model selection of $f _ { \theta }$ during the ablation study; and $\mathcal { P } _ { \mathrm { t e s t } } .$ used for the final evaluation of the proposed model.

To prevent distributional biases during model training, which could skew DeepSSIM++ toward over-represented regions of the similarity spectrum, $\mathcal { P } _ { \mathrm { t r a i n } }$ and $\mathcal { P } _ { \mathrm { v a l i d } }$ are obtained through stratified random sampling based on the reference SSIM scores, ensuring a balanced distribution of image pairs across the entire similarity range, as illustrated in Figure 2 (A) and (B). Additionally, we ensure that no image pair is assigned to more than one subset.

For the test set $\mathcal { P } _ { \mathrm { t e s t } }$ , applying the same SSIM-based sampling strategy would introduce an evaluation bias. Because the network is explicitly trained to predict SSIM values, using SSIM to curate the test set would evaluate the model on the same metric it was optimized to learn, po tentially leading to overly optimistic performance estimates. To avoid this, we adopt a hybrid candidate selection strategy. For each real image, four image pairs are constructed: two using the synthetic samples with the highest FSIM scores (Zhang et al., 2011) and two using randomly selected synthetic samples. The FSIM-based selection, which relies on principles fundamentally diferent from SSIM, enriches the test set with highly similar cases, while random sampling preserves coverage of the natural similarity distribution. The resulting SSIM distribution of the test set obtained by this strategy is illustrated in Figure 2 (C).

The resulting test set contains 8,780 image pairs, independently annotated by three expert observers according to the similarity definitions proposed in Section 3.3.1. Final labels are determined via majority voting, with no unresolved ambiguities across all annotated pairs. To assess annotation consistency, we measure the agreement among observers using Fleiss’ �, a statistical measure that evaluates how consistently observers assign the same labels while accounting for agreement that may occur by chance.

## 3.2. Proposed Self-Supervised Similarity Metric

We begin by computing SSIM-based supervision targets, which are used to train our model. For each pair $( I _ { a } , I _ { b } )$ where $I _ { a }$ is a real image and $I _ { b }$ is a synthetic image, we co-register $I _ { b }$ to $I _ { a }$ using a rigid transformation �. The aligned images are then normalized using z-score brightness normalization before computing the reference SSIM score:

$$
s _ { a b } = \mathrm { S S I M } ( I _ { a } , \phi ( I _ { b } ) ) .\tag{2}
$$

Brightness normalization reduces the influence of nonanatomical intensity variations, ensuring that the target similarity primarily reflects structural correspondence. The coregistration step is used only during training to compute reliable SSIM targets. Because SSIM is highly sensitive to small spatial misalignments, aligning each image pair ensures that the supervision signal reflects structural similarity rather than positional diferences. At inference time, no explicit co-registration is required, as the learned representation is designed to be robust to these variations.

To do so, we optimize a feature extractor $f _ { \theta }$ that maps images into a representation space where the learned similarity score $s _ { \theta } ( I _ { a } , I _ { b } )$ approximates the corresponding SSIM target $s _ { a b }$ . Each image in the pair is then independently transformed by a randomly sampled transformation before feature extraction:

$$
\tilde { I } _ { a } = T _ { a } ( I _ { a } ) , \quad \tilde { I } _ { b } = T _ { b } ( I _ { b } ) , \quad T _ { a } , T _ { b } \sim \mathcal { T } ,\tag{3}
$$

where  denotes the set of training transformations. All transformations are designed to preserve anatomical identity while exposing the model to realistic spatial and appearance variations that may be encountered during inference. Consequently, the learned representation assigns high similarity scores to image pairs depicting the same anatomical content despite spatial misalignment or minor appearance distortions. The complete set of transformations used in our approach is described in Section 3.2.1. The transformed images are then mapped into the representation space:

$$
\begin{array} { r } { \mathbf { v } _ { a } = f _ { \theta } ( \tilde { I } _ { a } ) , \quad \mathbf { v } _ { b } = f _ { \theta } ( \tilde { I } _ { b } ) . } \end{array}\tag{4}
$$

The feature extractor $f _ { \theta }$ is optimized by minimizing the Mean Squared Error (MSE) between the learned similarity score $s _ { \theta }$ and the SSIM supervision target over a batch of � image pairs:

$$
\mathcal { L } ( \theta ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left( s _ { \theta } ( I _ { a } ^ { ( i ) } , I _ { b } ^ { ( i ) } ) - s _ { a _ { i } b _ { i } } \right) ^ { 2 } .\tag{5}
$$

## 3.2.1. Training Augmentations

To improve the robustness of DeepSSIM++ for memorization detection, we apply a set of anatomy-preserving augmentations during training. These transformations generate perturbed views while maintaining the underlying anatomical identity of each image. Formally, let $\mathcal { T } = \{ T ^ { ( 1 ) } , \dots , T ^ { ( n ) } \}$ denote the set of augmentation operators, each applied independently with probability $\boldsymbol { p } ^ { ( i ) }$

The augmentation strategy comprises two categories of transformations: intensity-based transformations, including bias-field augmentation, Gaussian noise, and contrast adjustment, which simulate appearance variations without modifying anatomical structures; and geometry-based transformations, including rotations, translations, zooming, and horizontal and vertical flips, which improve robustness to spatial perturbations and imperfect alignment. All transformations are designed to preserve anatomical correspondence between image pairs. Compared with the original DeepSSIM Scardace et al. (2026), the proposed augmentation pipeline extends the set of transformations by introducing additional perturbations, including zooming, translations, bias-field augmentation, and Gaussian noise.

## 3.2.2. Feature Extractor Architecture

The network $f _ { \theta } ,$ illustrated in Figure 3, combines a convolutional backbone with a multi-scale pooling and fusion head. We employ a pre-trained ConvNeXt-T backbone to extract feature maps from four progressively deeper stages of the network, denoted as Stages A, B, C, and D. As defined in the original architecture (Liu et al., 2022), these stages correspond to the main sequential blocks of the model, each operating at a progressively lower spatial resolution and higher feature dimensionality. Collectively, they capture hierarchical representations ranging from finegrained anatomical details to high-level structural information (Zeiler and Fergus, 2014; Wang et al., 2021).

Unlike Scardace et al. (2026), which relies solely on the deepest representation, DeepSSIM++ aggregates feature maps from all four stages. Specifically, we propose that each stage output is independently pooled through a dedicated Generalized Mean (GeM) layer (Radenović et al., 2019) with a separately learned pooling exponent $p .$ The resulting descriptors are concatenated and projected into the final embedding space through a two-layer MultiLayer Perceptron (MLP) comprising Layer Normalization, two linear layers separated by a GeLU activation, and dropout.

![](images/bb7e7aac607d2a8618624b13d7178d16d605624e10e5a27924bbbd075a243dd4.jpg)  
Figure 3: Feature Extractor Architecture: The figure illustrates the proposed feature extractor $f _ { \theta } ,$ which combines a ConvNeXt backbone with multi-scale GeM pooling and an MLP-based fusion head. Feature maps are extracted from four progressively deeper stages (Stage A, B, C, and D), capturing increasingly abstract anatomical information. The stage outputs are independently pooled, concatenated, and projected into the final embedding space.

To account for the diferent adaptation requirements of the pre-trained backbone and the randomly initialized fusion head, we employ LLRD (Bu et al., 2024). This strategy assigns progressively smaller learning rates to earlier layers, preserving pre-trained representations while enabling the fusion head to adapt more rapidly. We treat the entire feature extractor as a sequence of � trainable layers, and define the learning rate of layer � as:

$$
\begin{array} { r } { \eta _ { j } = \eta _ { 0 } \times \gamma ^ { L - j } . } \end{array}\tag{6}
$$

where $\eta _ { 0 }$ is the base learning rate, � is the total number of trainable layers, and $\gamma \in ( 0 , 1 ]$ is the decay factor.

## 3.3. Proposed Memorization Metric

For each pair $( I _ { a } , I _ { b } )$ , where $I _ { a }$ is a real image and $I _ { b }$ is a synthetic image, we compute their structural similarity using the learned similarity function $s _ { \theta } .$ . The resulting similarity score is converted into a semantic relationship between the two images using the thresholding function �:

$$
\psi ( I _ { a } , I _ { b } ) = \left\{ \begin{array} { l l } { \mathrm { U n r e l a t e d } } & { \mathrm { i f ~ } s _ { \theta } ( I _ { a } , I _ { b } ) \leq \alpha } \\ { \mathrm { N e a r \mathrm { - } D u p l i c a t e } } & { \mathrm { i f ~ } \alpha < s _ { \theta } ( I _ { a } , I _ { b } ) < \beta , } \\ { \mathrm { E x a c t } \mathrm { D u p l i c a t e } } & { \mathrm { i f ~ } s _ { \theta } ( I _ { a } , I _ { b } ) \geq \beta } \end{array} \right.\tag{7}
$$

where � and � denote the decision thresholds and are hyperparameters of the proposed pipeline.

## 3.3.1. Definition ofMemorization Categories

The three mutually exclusive categories defined by � describe diferent levels of anatomical correspondence between a training image and a generated sample.

Exact Duplicate. A pair $( I _ { a } , I _ { b } )$ is classified as an Exact Duplicate if and only if every anatomical structure present in $I _ { a }$ has a corresponding structure in $I _ { b } ,$ and no structural deviation is observed between corresponding structures. Only anatomy-independent artifacts that do not alter the position, shape, size, or boundary of any individual anatomical structure are tolerated, such as uniform blur.

Near-Duplicate. A pair $( I _ { a } , I _ { b } )$ is classified as a Near-Duplicate when the overall anatomical configuration of $I _ { b }$ is still recognizable as corresponding to $I _ { a } ,$ but one or more localized deviations are present. These deviations are confined to specific anatomical regions and may involve boundary distortions, shape alterations, or tissue composition inconsistencies, without compromising the identifiability of the underlying anatomical correspondence.

Unrelated. A pair $( I _ { a } , I _ { b } )$ is classified as Unrelated when the anatomical structure of $I _ { b }$ is not consistent with that of $I _ { a } ,$ and no meaningful anatomical correspondence can be established between the two images.

## 3.4. Implementation Details

The framework is implemented in PyTorch (Paszke et al., 2019), leveraging MONAI (Cardoso et al., 2022) for anatomy-preserving data augmentations. The feature extractor $f _ { \theta }$ is based on a ConvNeXt-T architecture (Liu et al., 2022) pre-trained on ImageNet. The original classification layer is replaced with an identity mapping, enabling direct extraction of image embeddings rather than image classification. To preserve low-level spatial features while adapting the model to the neuroimaging domain, all parameters in Stage A (including the stem and the initial convolutional layers) are frozen, whereas the remaining backbone stages and the projection head are jointly fine-tuned.

The network is trained for 65 epochs, empirically determined via validation convergence, using the AdamW optimizer (Loshchilov and Hutter, 2019) with a weight decay of $1 \times 1 0 ^ { - 3 }$ and a batch size of 64. We employ a base learning rate $\eta _ { 0 } = 1 \times 1 0 ^ { - 3 }$ for the projection head, which is then scaled across the trainable backbone layers using LLRD with a decay factor $\gamma = 0 . 3$ . The multi-scale GeM pooling layers are initialized with $p = 3 . 0$ and $\epsilon = 1 0 ^ { - 6 }$ , and the projection head uses a dropout probability of 0.1 to produce final embeddings of dimension � = 256.

The thresholds of the decision function � are selected via grid search over [0, 1] with a step size of 0.01, such that $\alpha < \beta .$ , resulting in � = 0.76 and $\beta = 0 . 9 2$

The LDM used as a case study is trained following the protocol proposed by Pinaya et al. (2022), adapted to generate 2D axial brain MRI slices.

All model training procedures and experiments were performed on a workstation equipped with an Intel(R) Core(TM) i7-14700 CPU, 32 GB of RAM, and a single NVIDIA RTX 4070 Ti SUPER GPU (16 GB of VRAM).

## 4. Experiments and Results

In this section, we conduct an extensive evaluation of DeepSSIM++ through five experiments: (i) an ablation study to identify the optimal configuration and quantify the impact of each design choice; (ii) a quantitative comparison of DeepSSIM++ for memorization detection against stateof-the-art baselines; (iii) a qualitative and explainability analysis to investigate the learned similarity representations; (iv) a sensitivity analysis of the proposed classification thresholds; and (v) a computational eficiency analysis comparing DeepSSIM++ with SSIM.

All experiments were performed using fixed random seeds to ensure reproducibility across data splitting, network initialization, and optimization.

## 4.1. Ablation Study

To establish the optimal DeepSSIM++ configuration, we conducted an ablation study covering architectural choices, training strategies, optimization schemes, and hyperparameter tuning. All experiments were performed on the validation set $\mathcal { P } _ { \mathrm { v a l i d } }$ , and performance was measured in terms of Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between the predicted similarity scores from DeepSSIM++ and the reference SSIM values for each pair.

We adopt a sequential ablation strategy, evaluating one design choice at a time while retaining the best-performing configuration at each step. The base configuration follows the original DeepSSIM design described in Scardace et al. (2026), with a ConvNeXt-B backbone, 256-dimensional output, partial fine-tuning of the pre-trained backbone, and MSE as the training objective.

Augmentations. We first evaluate the impact of the enriched augmentation strategy defined in Section 3.2.1. As shown in Table 1, expanding the set of transformations improves over the base configuration, reducing MAE by 4.64% and RMSE by 5.30%. This suggests that exposing the network to a wider range of realistic spatial and intensity perturbations improves robustness to such variations when learning the similarity metric.

Architecture. Next, we compare the single-scale baseline— which extracts features exclusively from the deepest network stage—with the proposed multi-scale architecture. As shown in Table 1, the multi-scale variant achieves a reduction in MAE of 20.13% and in RMSE of 17.41%. These results suggest that aggregating hierarchical spatial representations at diferent granularities consistently improves the approximation of structural similarity.

<table><tr><td>Experiment Setting</td><td>MAE RMSE</td></tr><tr><td colspan="2">(A) Ablation on the Augmentations</td></tr><tr><td>Base Augmentations</td><td> $0 . 0 4 7 4 \pm 0 . 0 4 6 1$   $0 . 0 6 6 1 \pm 0 . 0 1 2 9$ </td></tr><tr><td>Enriched Augmentations</td><td> $0 . 0 4 5 2 \pm 0 . 0 4 3 3$   $0 . 0 6 2 6 \pm 0 . 0 1 0 3$ </td></tr><tr><td colspan="2">(B) Ablation on the Architecture</td></tr><tr><td>Single-Scale</td><td> $0 . 0 4 5 2 \pm 0 . 0 4 3 3$   $0 . 0 6 2 6 \pm 0 . 0 1 0 3$ </td></tr><tr><td colspan="2">Multi-Scale  $0 . 0 3 6 1 \pm 0 . 0 3 6 9$   $0 . 0 5 1 7 \pm 0 . 0 0 8 2$ </td></tr><tr><td colspan="2">(C) Ablation on the Training Strategy</td></tr><tr><td>From Scratch</td><td> $0 . 0 6 0 7 \pm 0 . 0 4 9 7$   $0 . 0 7 8 4 \pm 0 . 0 1 0 7$ </td></tr><tr><td>Full Fine-Tuning Partial Fine-Tuning</td><td> $0 . 0 5 2 1 \pm 0 . 0 4 8 1$   $0 . 0 7 0 9 \pm 0 . 0 1 2 5$   $0 . 0 5 1 7 \pm 0 . 0 0 8 2$ </td></tr><tr><td>Head-Only Training</td><td> $0 . 0 3 6 1 \pm 0 . 0 3 6 9$   $0 . 0 4 3 9 \pm 0 . 0 4 4 6$   $0 . 0 6 2 6 \pm 0 . 0 0 9 7$ </td></tr><tr><td colspan="2"></td></tr><tr><td> $0 . 0 3 6 1 \pm 0 . 0 3 6 9$ </td><td>(D) Ablation on the Optimization Scheme  $0 . 0 5 1 7 \pm 0 . 0 0 8 2$ </td></tr><tr><td colspan="2">Uniform LLRD  $0 . 0 3 2 6 \pm 0 . 0 3 8 3$ </td></tr><tr><td>(E) Ablation on the Loss Function</td><td> $0 . 0 5 0 3 \pm 0 . 0 0 9 0$ </td></tr><tr><td colspan="2">MSE  $0 . 0 3 2 6 \pm 0 . 0 3 8 3$ </td></tr><tr><td>MAE Soft Contrastive Loss</td><td> $0 . 0 5 0 3 \pm 0 . 0 0 9 0$   $0 . 0 3 3 5 \pm 0 . 0 4 3 5$   $0 . 0 5 4 9 \pm 0 . 0 1 2 0$ </td></tr><tr><td colspan="2"> $0 . 0 4 9 7 \pm 0 . 0 3 9 5$   $0 . 0 6 3 5 \pm 0 . 0 1 0 0$  (F) Ablation on the Embedding Dimension</td></tr><tr><td>64</td><td> $0 . 0 3 2 6 \pm 0 . 0 3 8 6$   $0 . 0 5 0 5 \pm 0 . 0 0 9 3$ </td></tr><tr><td>128</td><td> $0 . 0 3 2 6 \pm 0 . 0 3 9 2$   $0 . 0 5 0 4 \pm 0 . 0 0 9 2$ </td></tr><tr><td>256</td><td> $0 . 0 3 2 5 \pm 0 . 0 3 8 3$   $0 . 0 5 0 3 \pm 0 . 0 0 9 0$ </td></tr><tr><td>512</td><td> $0 . 0 3 3 8 \pm 0 . 0 3 8 1$   $0 . 0 5 0 9 \pm 0 . 0 0 8 9$ </td></tr><tr><td colspan="2">(G) Ablation on the Backbone Architecture ResNet50</td></tr><tr><td>DenseNet161</td><td> $0 . 0 4 5 8 \pm 0 . 0 4 4 0$   $0 . 0 6 3 5 \pm 0 . 0 1 0 8$   $0 . 0 6 0 3 \pm 0 . 0 5 4 7$   $0 . 0 8 1 4 \pm 0 . 0 1 4 4$ </td></tr><tr><td>ConvNeXt-T</td><td> $0 . 0 3 2 5 \pm 0 . 0 3 7 0$   $0 . 0 4 9 2 \pm 0 . 0 0 8 1$ </td></tr><tr><td colspan="2">(H) Ablation on Hyperparameters</td></tr><tr><td>Default  $0 . 0 3 2 5 \pm 0 . 0 3 7 0$ </td><td> $0 . 0 4 9 2 \pm 0 . 0 0 8 1$ </td></tr><tr><td>Optimized</td><td> $0 . 0 2 6 9 \pm 0 . 0 3 1 0$   $0 . 0 4 1 0 \pm 0 . 0 0 5 6$ </td></tr></table>

Results of the sequential ablation study on the validation set. Performance is reported in terms of MAE (± SD) and RMSE (± SD), evaluating the model’s ability to accurately predict SSIM values. At each step, the best-performing configuration identified in the previous evaluation is retained for subsequent experiments (highlighted in bold with a gray background).

Training Strategy. We evaluate four training strategies to investigate how diferent fine-tuning schemes afect the adaptation of the backbone to the similarity prediction task: (i) training the network from scratch; (ii) head-only training, where the backbone is frozen and only the prediction head is trained; (iii) full fine-tuning, updating the entire backbone; and (iv) partial fine-tuning, selectively adapting a subset of pre-trained layers. The results in Table 1 reveal a clear performance ordering that highlights the trade-of between preserving pre-trained visual knowledge and adapting to the target domain. Partial fine-tuning achieves the best performance, reducing MAE by 17.77% and RMSE by 17.41% compared with the second-best configuration (head-only training). Full fine-tuning produces a performance degradation, increasing MAE by 44.32% relative to partial finetuning, suggesting that updating the entire backbone leads to excessive adaptation and disrupts the pre-trained representations. Finally, training the network from scratch yields the worst performance across all configurations, increasing MAE by 68.14% relative to partial fine-tuning, further underscoring the critical importance of transfer learning from pre-training on natural images. Overall, these results indicate that freezing too much of the backbone limits adaptation, whereas updating too much of it disrupts transferable representations; consequently, selectively adapting a subset of pre-trained layers provides the optimal balance.

Optimization Scheme. We evaluate the impact of the optimization strategy by comparing LLRD with a uniform learning rate strategy. In Table 1, we can see that LLRD consistently achieves better performance by assigning progressively smaller learning rates to earlier backbone layers and larger learning rates to the prediction head, enabling a more controlled adaptation of the pre-trained representations. Compared with the uniform optimization strategy, LLRD reduces MAE by 9.70% and RMSE by 2.71%, confirming the benefit of layer-wise adaptation for this task.

Loss Function. We evaluate MSE, MAE, and Soft Contrastive Loss as supervision objectives for similarity learning. In Table 1, MSE achieves the best performance, reducing MAE by 2.69% and RMSE by 8.38% compared with the second-best loss function (MAE). This result suggests that the quadratic penalty provides a more efective optimization objective for this regression problem, as it places greater emphasis on larger prediction errors. Conversely, the contrastive loss underperforms in this direct regression setting, likely because its objective is not specifically designed to regress continuous similarity scores.

Embedding Dimension. We evaluate four embedding dimensions corresponding to the backbone output dimensionality: 64, 128, 256, and 512. As shown in Table 1, an embedding dimension of 256 achieves the best overall performance. However, the average relative deviation across all dimensions is 1.54% for MAE and 0.60% for RMSE, suggesting that DeepSSIM++ is robust to this hyperparameter.

Backbone Architecture. We evaluate diferent convolutional backbones to assess their impact on similarity prediction, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), and ConvNeXt (Liu et al., 2022) variants with diferent capacities. As shown in Table 1, ConvNeXt-T achieves the best overall performance, outperforming both the ResNet and DenseNet families as well as the larger ConvNeXt variants. Compared with the best-performing

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Unrelated</td><td colspan="2">Near-Duplicate</td><td colspan="2">Exact Duplicate</td><td rowspan="2">Macro F1</td><td rowspan="2">TPR05%FPR</td><td rowspan="2">Silhouette</td></tr><tr><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td><td>Precision Recall</td><td></td></tr><tr><td rowspan="7">Reied Pptairs (retc ct</td><td>SemDeDup</td><td>71.82%</td><td>99.66%</td><td>61.79%</td><td>2.57%</td><td>0.00%</td><td>0.00%</td><td>29.48%</td><td>24.93%</td><td>-0.04</td></tr><tr><td>UMS (t2)</td><td>95.64%</td><td>96.94%</td><td>86.69%</td><td>75.79%</td><td>56.59%</td><td>83.63%</td><td>81.55%</td><td>90.64%</td><td>0.23</td></tr><tr><td>UMS (SSCD)</td><td>94.56%</td><td>90.08%</td><td>64.63%</td><td>67.99%</td><td>33.50%</td><td>49.61%</td><td>66.18%</td><td>54.02%</td><td>0.08</td></tr><tr><td>Dar et al.</td><td>71.49%</td><td>99.39%</td><td>N/A</td><td>N/A</td><td>17.50%</td><td>1.81%</td><td>28.87%</td><td>23.11%</td><td>0.13</td></tr><tr><td>SSIM</td><td>99.66%</td><td>99.66%</td><td>95.24%</td><td>98.31%</td><td>94.62%</td><td>77.66%</td><td>93.91%</td><td>100.00%</td><td>0.46</td></tr><tr><td>DeepSSIM</td><td>95.17%</td><td>94.62%</td><td>75.91%</td><td>52.57%</td><td>32.53%</td><td>91.16%</td><td>68.33%</td><td>71.94%</td><td>0.31</td></tr><tr><td>DeepSSIM++ (Proposed)</td><td>96.28%</td><td>97.61%</td><td>87.06%</td><td>84.86%</td><td>74.79%</td><td>68.58%</td><td>84.81%</td><td>94.29%</td><td>0.39</td></tr><tr><td rowspan="7">(nnmymml Pertd pPpuairs</td><td>SemDeDup</td><td>74.13%</td><td>94.48%</td><td>47.40%</td><td>17.89%</td><td>0.00%</td><td>0.00%</td><td>36.35%</td><td>65.89%</td><td>-0.06</td></tr><tr><td>UMS (t2)</td><td>71.95%</td><td>99.79%</td><td>76.04%</td><td>3.41%</td><td>66.66%</td><td>1.55%</td><td>31.06%</td><td>10.12%</td><td>0.00</td></tr><tr><td>UMS (SSCD)</td><td>76.32%</td><td>98.49%</td><td>69.03%</td><td>21.77%</td><td>39.39%</td><td>3.37%</td><td>41.77%</td><td>27.53%</td><td>0.02</td></tr><tr><td>Dar et al.</td><td>71.42%</td><td>98.72%</td><td>N/A</td><td>N/A</td><td>19.23%</td><td>1.29%</td><td>28.58%</td><td>14.54%</td><td>0.10</td></tr><tr><td>SSIM</td><td>71.78%</td><td>99.98%</td><td>83.22%</td><td>2.61%</td><td>0.00%</td><td>0.00%</td><td>29.54%</td><td>11.68%</td><td>0.06</td></tr><tr><td>DeepSSIM</td><td>95.61%</td><td>92.63%</td><td>69.97%</td><td>55.42%</td><td>31.31%</td><td>83.37%</td><td>67.16%</td><td>59.48%</td><td>0.27</td></tr><tr><td>DeepSSIM++ (Proposed)</td><td>96.35%</td><td>96.69%</td><td>83.58%</td><td>84.82%</td><td>69.88%</td><td>58.45%</td><td>81.39%</td><td>91.17%</td><td>0.36</td></tr></table>

Quantitative performance evaluation on the annotated test set using precision and recall for each class, macro F1 score, TPR@5%FPR, and Silhouette score. Results are reported for both perfectly registered pairs and randomly perturbed pairs. The best results for each metric are highlighted with a gray background and bold font, whereas the second-best results are highlighted with a gray background only.

ResNet50 configuration, it reduces MAE by 29.04% and RMSE by 22.52%. Similarly, relative to the best DenseNet161 configuration, it reduces MAE by 46.10% and RMSE by 39.56%. Notably, although all previous experiments were conducted using ConvNeXt-B, the smaller ConvNeXt-T achieves comparable performance with fewer parameters (28M vs 88M) and slightly better results. These findings indicate that ConvNeXt-T is particularly well suited for this task and that increasing model capacity alone does not necessarily improve performance.

Hyperparameter Analysis. Finally, we evaluate the influence of the remaining hyperparameters, namely the dropout probability in the range [0.1, 0.5] and the weight decay in the range $[ 1 \times 1 0 ^ { - 2 } , 1 \times 1 0 ^ { - 4 } ]$ . As shown in Table 1, optimizing these parameters for the selected architecture and training strategy further improves performance, reducing MAE by 17.23% and RMSE by 16.67%.

## 4.2. Annotation Reliability

Before evaluating the memorization detection methods, we assessed the reliability of our manually annotated test set $\mathcal { P } _ { \mathrm { t e s t } }$ . The obtained Fleiss’ � score of 0.99 indicates almost perfect agreement among the three expert observers, confirming that the annotation protocol produces highly reproducible labels. The few disagreements observed were strictly confined to neighboring categories, suggesting uncertainty only in anatomically borderline cases rather than inconsistencies in the labeling criteria.

## 4.3. Quantitative Comparison against Baselines

We perform a quantitative evaluation comparing our approach against baseline methods by assigning each image pair to one of three predefined similarity categories:

Unrelated, Near-Duplicate, or Exact Duplicate. These experiments are conducted on the annotated dataset $\mathcal { P } _ { \mathrm { t e s t } }$ described in Section 3.1.3. We report precision, recall, macro F1 score, and True Positive Rate at 5% False Positive Rate (TPR@5%FPR) for each evaluated method. Since TPR@5%FPR is a binary metric, we adopt a one-vs-rest evaluation scheme, with Exact Duplicate as the positive class. Additionally, we compute the Silhouette score to evaluate class separability in the learned feature space, providing a threshold-independent assessment of representation quality. All experiments are conducted under two settings: (A) an ideal setting with perfectly registered image pairs, and (B) a realistic challenging setting where image pairs undergo spatial and intensity perturbations. In the latter, images are subjected to random, anatomy-preserving transformations, including horizontal and vertical flips, subtle rotations, and contrast variations. Statistical significance is assessed using McNemar’s test on paired classification outcomes, with all improvements of DeepSSIM++ over competing approaches reaching significance $( p < 0 . 0 5 )$ .

## 4.3.1. Baselines

The first baseline considered is the standard reference similarity measure SSIM. Although SSIM is not designed for memorization detection, its evaluation provides insight into the behavior of this traditional pixel-based metric under the two considered settings and serves as an approximate upper-bound reference in the ideal, alignment-dependent scenario rather than a direct competing baseline.

The second baseline is SemDeDup (Abbas et al., 2023), an approach originally designed for the semantic deduplication of large-scale training datasets, which we adapt to our evaluation protocol. Specifically, we map the “perceptual duplicate” category to our Exact Duplicate class and the “semantic duplicate” category to our Near-Duplicate class. To better capture biomedical image representations, we employ BioMedCLIP (Zhang et al., 2024) as the underlying foundation model instead of the standard CLIP (Radford et al., 2021) architecture originally used by the authors.

![](images/da222e4003177f539e88b92bf9e3b17dc87080aceef90faf66610287348b422c.jpg)  
Figure 4: Similarity Score Distributions Across Methods. This figure illustrates the distribution of similarity scores predicted by each method under perfect registration, stratified by ground-truth category: Unrelated, Near-Duplicate, Exact Duplicate. Dashed vertical lines indicate the decision threshold(s) used by each method.

The next baseline is UMS (Chen et al., 2026), which categorizes image pairs into four similarity classes: diferent, low, medium, and high. UMS provides two variants based on diferent similarity measures: the normalized $\ell _ { 2 }$ distance between embeddings for low-resolution inputs and SSCD embeddings for high-resolution inputs. To align UMS with our three-class setting, we merge the low and medium categories into a unified Near-Duplicate class, while mapping high similarity to Exact Duplicate and diferent to Unrelated.

We also include the contrastive approach proposed by Dar et al. (2026), which applies a single decision threshold based on the 95th percentile of the training–validation similarity distribution. As a result, this baseline performs binary classification between Exact Duplicate and Unrelated pairs, without explicitly identifying Near-Duplicate cases. Using the oficial implementation, we retrained this method on our dataset following the authors’ training protocol as closely as possible. Our final baseline is the original DeepSSIM model presented in Scardace et al. (2026).

To ensure a fair comparison across methods, we standardize the threshold selection procedure for approaches that rely on predefined decision thresholds, namely UMS and the original DeepSSIM. For each of these methods, we perform an independent grid search on the validation set to identify the optimal thresholds. SemDeDup and the contrastive method of Dar et al. (2026) are instead evaluated using their original decision procedures, as their classification mechanisms are integral to the methods themselves. Results obtained using the original thresholds reported for UMS and DeepSSIM are additionally provided in Appendix A to quantify the efect of threshold adaptation.

## 4.3.2. Performance on Registered Pairs

As shown in Table 2 (A), under ideal registration conditions, DeepSSIM++ achieves a Silhouette score of 0.39, outperforming all external memorization detection baselines and closely approaching the SSIM reference (0.46). Compared with the strongest external competitor, UMS $( \ell _ { 2 } ) .$ , DeepSSIM++ achieves a substantial improvement in Silhouette score (0.39 vs 0.23), indicating a more discriminative similarity space. This improved class separation is reflected in the classification results: DeepSSIM++ achieves an overall macro F1 score of 84.81%, outperforming the strongest external baseline, UMS $( \ell _ { 2 } ) _ { : }$ , which reaches 81.55%. Likewise, DeepSSIM++ achieves a TPR@5%FPR of 94.29%, compared with 90.64% for UMS $( \ell _ { 2 } )$ . The nearperfect performance of SSIM in this setting is expected, as images are perfectly aligned, confirming that pixel-wise similarity remains highly efective under ideal registration conditions. Relative to the original DeepSSIM formulation (Scardace et al., 2026), DeepSSIM++ substantially improves macro F1 from 68.33% to 84.81% and Silhouette score from 0.31 to 0.39. Although DeepSSIM++ exhibits lower Exact Duplicate recall than the baseline (68.58% vs 91.16%), it substantially improves precision (74.79% vs 32.53%). By adopting a more conservative decision boundary, the model reclassifies borderline cases as Near-Duplicate, yielding markedly higher recall in this category (84.86% vs 52.57%). This trade-of is well suited to auditing, as it reduces false positives while still surfacing suspicious samples for manual review.

## 4.3.3. Performance on Perturbed Pairs

As shown in Table 2 (B), DeepSSIM++ demonstrates strong robustness to spatial and contrast variations, achieving the highest overall Silhouette score (0.36), substantially exceeding the SSIM reference (0.06) by a factor of 6×. This result highlights the sensitivity of direct image-space similarity measures, such as SSIM and $\ell _ { 2 }$ distance, to even small spatial perturbations, and the advantage of learned similarity representations that are less dependent on exact pixel alignment. DeepSSIM++ achieves an overall macro F1 score of 81.39%, outperforming the strongest external memorization detection baseline, UMS (SSCD), which reaches 41.77%, corresponding to a relative improvement of 95%. Likewise, DeepSSIM++ achieves a TPR@5%FPR of 91.17%, compared with 65.89% for SemDeDup, corresponding to a relative improvement of 38%. Interestingly, SemDeDup’s macro F1 increases under perturbations, rising from 29.48% to 36.35%, primarily due to higher Near-Duplicate recall. We hypothesize that this improvement does not reflect genuine robustness to structural changes. Instead, it may result from a distributional shift in which geometric and intensity perturbations move borderline embeddings across the decision threshold, causing previously missed samples to be classified as Near-Duplicate. Notably, UMS $( \ell _ { 2 } )$ , which represents one of the strongest baselines under ideal registration, substantially degrades under perturbations, reaching a Silhouette score of 0.00, a macro F1 score of 31.06%, and a TPR@5%FPR of only 10.12%. This degradation further highlights the limitations of direct image-space similarity measures under realistic spatial misalignment. In contrast, the original DeepSSIM formulation (Scardace et al., 2026) remains substantially more robust than most competing approaches, while DeepSSIM++ further improves upon it, increasing macro F1 from 67.16% to 81.39% and Silhouette score from 0.27 to 0.36.

DeepSSIM++  
![](images/15b94dc5227254e869a650ee859ad5281742a4e836a1b5a1137142c9fb3de42d.jpg)  
Figure 5: Qualitative Explainability Analysis via LayerCAM. This figure illustrates six representative examples, comprising three Exact Duplicate examples (top row) and three Near-Duplicate examples (bottom row). For each pair, we show the real image, the corresponding synthetic image, and the associated LayerCAM activation map, which highlights the anatomical regions that most strongly drive the similarity score prediction and not the regions of agreement or discrepancy between the two images. Across both similarity categories, the network consistently attends to anatomically relevant structures, including the lateral ventricles, periventricular white matter, and cortical gyri, with negligible activation over background regions.

These results demonstrate that the proposed refinements improve the ability of DeepSSIM++ to learn alignmenttolerant similarity representations suitable for realistic, unregistered medical imaging scenarios.

![](images/d607465cf4ff3eb422f78faadbd3a4bc00f72d2cde28faf2460d4af55e3fdbbe.jpg)  
Figure 6: Sensitivity of the Proposed Thresholds. This figure illustrates the variations of the macro F1 score as a function of Gaussian noise applied to thresholds � and $\beta .$ The �- and �-axes represent increasing noise levels, corresponding to $\sigma _ { \beta }$ and $\sigma _ { \alpha }$ , respectively.

## 4.4. Qualitative Analysis

Figure 4 illustrates the similarity score distributions produced by each method across diferent classes. The resulting distributions indicate that baseline methods provide more limited separation between similarity categories, whereas DeepSSIM++ produces better-separated score distributions, consistent with the Silhouette scores reported in Table 2.

## 4.4.1. Explainability Analysis

To investigate whether DeepSSIM++ relies on meaningful visual regions when estimating image similarity, we perform an interpretability analysis using LayerCAM (Jiang et al., 2021). In particular, given a real–synthetic image pair, LayerCAM is obtained by backpropagating the predicted similarity score $s _ { \theta } .$ . The resulting multi-scale activation maps are then aggregated and min–max normalized to [0, 1] for visualization. The activation map highlights regions considered by the network when producing the similarity score.

Figure 5 shows representative explanations for six image pairs, covering both Exact Duplicate (top row) and Near-Duplicate (bottom row) cases. Across both categories, DeepSSIM++ consistently attends to anatomically relevant structures, most notably the lateral ventricles, periventric ular white matter, and cortical gyri, with negligible activation over background regions. This pattern indicates that the network relies on anatomically meaningful evidence to estimate similarity regardless of the underlying degree of correspondence between the paired images. These results support the use of DeepSSIM++ as an interpretable auditing tool rather than a black-box model.

## 4.5. Threshold Sensitivity Analysis

To evaluate the robustness of the proposed thresholds � and $\beta$ in the DeepSSIM++ pipeline, we analyze their sensitivity to Gaussian perturbations using the annotated brain MRI test set $\mathcal { P } _ { \mathrm { t e s t } }$ . Formally, let $\alpha = 0 . 7 6$ and $\beta = 0 . 9 2$ be the optimal threshold values. We define the perturbed thresholds ̃� and $\tilde { \beta }$ by injecting independent additive noise:

$$
\tilde { \alpha } = \alpha + \epsilon _ { \alpha } , \quad \tilde { \beta } = \beta + \epsilon _ { \beta }\tag{8}
$$

where $\epsilon _ { \alpha } ~ \sim ~ \mathcal { N } ( 0 , \sigma _ { \alpha } ^ { 2 } )$ and $\epsilon _ { \beta } ~ \sim ~ \mathcal { N } ( 0 , \sigma _ { \beta } ^ { 2 } )$ are zeromean Gaussian variables parameterized by their respective standard deviations $\sigma _ { \alpha }$ and $\sigma _ { \beta } .$ . Perturbations violating $\tilde { \alpha } < \tilde { \beta }$ or ̃�, $\tilde { \beta } \in [ 0 , 1 ]$ are discarded.

We evaluate the impact of increasing noise levels, with $\sigma _ { \alpha }$ and $\sigma _ { \beta }$ ranging from 0.00 to 0.05 in increments of 0.01, by measuring the corresponding variation in macro F1 score. $\mathbf { A } s$ shown in Figure 6, the F1 score decreases gradually as perturbation magnitude increases, from 0.85 in the noisefree setting $( \sigma _ { \alpha } = \sigma _ { \beta } = 0 )$ to 0.71 under the largest joint perturbation $( \sigma _ { \alpha } = { \overset { . } { \sigma } } _ { \beta } = 0 . 0 5 )$ , with no abrupt collapse observed anywhere in the grid. Notably, even under the largest perturbations, the performance consistently remains above all embedding-based baselines reported in Table 2.

## 4.6. Computational Eficiency

Beyond its superior predictive performance, DeepSSIM++ ofers a substantial computational advantage over traditional SSIM. By encoding images into compact vector embeddings, it enables the use of highly optimized nearestneighbor search frameworks, such as FAISS (Johnson et al., 2019), for large-scale retrieval. Furthermore, cosine similarities between entire sets of real and synthetic images can be computed eficiently through a single matrix multiplication of their $\ell _ { 2 }$ -normalized embedding matrices, fully leveraging GPU-accelerated linear algebra routines.

In particular, Figure 7 compares the execution time of SSIM and DeepSSIM++ as the number of image pairs increases. For SSIM, we report three implementations: the standard single-process version, a multiprocessing implementation, and a GPU-accelerated implementation. At small scales $( 1 \times 1 0 ^ { 5 }$ comparisons), all methods complete within seconds, and runtime diferences remain limited. As the number of comparisons increases, however, the gap progressively widens, reflecting the fundamentally diferent computational complexity of pixel-wise similarity evaluation and embedding-based similarity computation. At the largest evaluated scale $( \sim 2 \times 1 0 ^ { 8 }$ image pairs), the standard singleprocess implementation of SSIM requires nearly 48 hours, while multiprocessing reduces the runtime to approximately 7 hours. Even the GPU implementation requires about 90 minutes to complete all pairwise comparisons. In contrast, DeepSSIM++ completes the entire pipeline, including embedding extraction and similarity computation, in approximately 5 seconds, or in less than 1 second when embeddings are pre-computed. Compared with the GPU implementation of SSIM, this corresponds to approximately three orders of magnitude faster, while the speedup exceeds four orders of magnitude relative to the single-process implementation. Since embeddings are computed only once, subsequent similarity queries require only a matrix multiplication in the embedding space. This makes DeepSSIM++ readily scalable to datasets containing hundreds of millions of image comparisons, as demonstrated in this work.

![](images/d5e9277e91d2d3d82f4053e926de20db240bc51ac7a50a3242eedd4cf064410d.jpg)  
Figure 7: Computational Runtime Comparison Between SSIM and DeepSSIM++. This figure illustrates the elapsed time required to compute an increasing number of pairwise comparisons, shown on the �-axis. The top panel reports the runtime of three SSIM implementations: single-process (light blue), multiprocessing (dark blue, dashed), and GPUaccelerated (light green, dotted), measured in hours. The bottom panel reports DeepSSIM++ runtime under two configurations: full computation, including embedding extraction (light red), and pre-computed embeddings, requiring only similarity computation (dark red, dashed), measured in seconds.

## 5. Discussion and Conclusion

In this work, we introduced DeepSSIM++, a selfsupervised similarity metric for detecting memorization in medical generative models. Our results demonstrate that DeepSSIM++ achieves performance comparable to SSIM under ideal registration conditions, while substantially outperforming SSIM and all evaluated learning-based baselines when image pairs undergo realistic spatial and intensity perturbations. This robustness is particularly relevant for practical memorization auditing, where perfect pixel-level alignment between real and synthetic images cannot generally be assumed. Beyond detection performance, DeepSSIM++ provides a computational advantage of several orders of magnitude over pixel-space SSIM at scale, enabling memorization analysis on datasets involving hundreds of millions of image pairs, for which exhaustive pixel-wise evaluation becomes impractical. The explainability analysis further indicates that DeepSSIM++ predictions are grounded in anatomically meaningful structures rather than spurious visual patterns, supporting its use as an interpretable auditing tool rather than a black-box classifier.

Despite these strengths, we acknowledge that the decision thresholds � and � are calibrated on our brain MRI dataset, as the mapping from continuous similarity scores to memorization categories is data-distribution dependent. Adaptive thresholding strategies based on unsupervised density estimation, such as Gaussian Mixture Models or multilevel Otsu thresholding, provide potential alternatives but rely on the assumption that memorization categories correspond to statistically separable modes in the score distribution. This assumption cannot be guaranteed, as in a continuous similarity space, semantic categories may overlap, collapse into a single density mode, or exhibit multiple modes within the same category depending on dataset composition. Model selection criteria such as the Bayesian Information Criterion can estimate the number of statistical components but cannot determine their correspondence with meaningful memorization classes. Consequently, while the similarity metric is expected to generalize across domains, the decision thresholds may require recalibration to adapt to the specific score distribution of any new dataset.

Beyond memorization detection, the efectiveness and scalability of DeepSSIM++ enable additional practical applications in privacy-preserving generation. For example, before the public release of synthetic datasets, generated samples could be automatically screened to identify and remove those classified as near- or exact-duplicates, reducing the risk of unintended training data disclosure. More interestingly, DeepSSIM++ could be integrated directly into the generation pipeline: embeddings of training images could be pre-computed, and during inference, each newly generated sample could be mapped into the embedding space and compared against the training set to identify its nearest neighbors. Samples exhibiting near- or exact-duplicate similarity could then be discarded. Lastly, our approach could be used by experts in medical imaging to evaluate how much their models memorize training data.

Future research will aim to integrate DeepSSIM++ directly into generative model training, moving beyond post-hoc detection toward generation procedures that explicitly control similarity to the training set. In this setting, DeepSSIM++ could serve as a regularization term to penalize the reproduction of training samples, mitigating memorization while preserving data utility and ensuring the generation of diverse and clinically meaningful outputs.

## A. Results Using Original Threshold Values

To complement the standardized evaluation in the main text, we evaluate UMS and the original DeepSSIM using the decision thresholds reported in their respective publications. Unlike the main-text results, where thresholds were selected via grid search on the validation set for fair comparison, these results assess the out-of-the-box applicability of the original methods to the categories defined in Section 3.3.1. As shown in Table 3, these thresholds substantially degrade performance: UMS $( \ell _ { 2 }$ and SSCD) achieves macro F1 below 3.3%, while the original DeepSSIM drops from 68.33% to 41.32% on registered pairs and from 67.16% to 40.48% under perturbations, due to overly aggressive Exact Duplicate classification. Importantly, the advantage of DeepSSIM++ is not attributable to threshold calibration alone. Metrics independent of the decision threshold favor DeepSSIM++, with higher Silhouette score (0.39 vs 0.23 for UMS $\ell _ { 2 } )$ and TPR@5%FPR (94.29% vs 90.65%), confirming a more discriminative similarity space.

## CRediT authorship contribution statement

Antonio Scardace: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Visualization. Francesco Guarnera: Writing - Review & Editing, Supervision. Sebastiano Battiato: Writing - Review & Editing, Supervision. Daniele Ravì: Conceptualization, Writing - Review & Editing, Project Administration, Supervision.

## References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., et al., 2016. Deep learning with diferential privacy, in: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, p. 308–318. URL: https://doi.org/10.1145/2976749. 2978318, doi:10.1145/2976749.2978318.

Abbas, A., Tirumala, K., Simig, D., Ganguli, S., Morcos, A.S., 2023. Semdedup: Data-eficient learning at web-scale through semantic deduplication. URL: https://arxiv.org/abs/2303.09540, arXiv:2303.09540.

Agarwal, S., Wood, D., Carpenter, R., Wei, Y., Modat, M., Booth, T.C., 2024. Letter to the editor: what are the legal and ethical considerations of submitting radiology reports to chatgpt? Clinical Radiology 79, e979– e981. doi:10.1016/j.crad.2024.03.017.

Ahishakiye, E., Van Gijzen, M.B., Tumwiine, J., Wario, R., Obungoloch, J., 2021. A survey on deep learning in medical image reconstruction. Intelligent Medicine 1, 118–127. doi:https://doi.org/10.1016/j.imed. 2021.03.003.

Arora, S., Risteski, A., Zhang, Y., 2018. Do GANs learn the distribution? some theory and empirics, in: International Conference on Learning Representations. URL: https://openreview.net/forum?id=BJehNfW0-.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Unrelated</td><td colspan="2">Near-Duplicate</td><td colspan="2">Exact Duplicate</td><td rowspan="2">Macro F1</td><td rowspan="2">TPR@5%FPR</td><td rowspan="2">Silhouette</td></tr><tr><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td><td>Precision Recall</td><td></td></tr><tr><td rowspan="8">Reied Pptdairs (D(ret ct</td><td>SemDeDup</td><td>71.82%</td><td>99.66%</td><td>61.79%</td><td>2.57%</td><td>0.00%</td><td>0.00%</td><td>29.48%</td><td>24.93%</td><td>-0.04</td></tr><tr><td>UMS (t2)</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>4.39%</td><td>100.00%</td><td>2.80%</td><td>90.64%</td><td>0.23</td></tr><tr><td>UMS (SSCD)</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>4.60%</td><td>100.00%</td><td>2.93%</td><td>54.02%</td><td>0.08</td></tr><tr><td>Dar et al.</td><td>71.49%</td><td>99.39%</td><td>N/A</td><td>N/A</td><td>17.50%</td><td>1.81%</td><td>28.87%</td><td>23.11%</td><td>0.13</td></tr><tr><td>SSIM</td><td>99.66%</td><td>99.66%</td><td>95.24%</td><td>98.31%</td><td>94.62%</td><td>77.66%</td><td>93.91%</td><td>100.00%</td><td>0.46</td></tr><tr><td>DeepSSIM</td><td>99.93%</td><td>52.48%</td><td>18.30%</td><td>29.96%</td><td>19.38%</td><td>100.00%</td><td>41.32%</td><td>71.94%</td><td>0.31</td></tr><tr><td>DeepSSIM++(Proposed)</td><td>96.28%</td><td>97.61%</td><td>87.06%</td><td>84.86%</td><td>74.79%</td><td>68.58%</td><td>84.81%</td><td>94.29%</td><td>0.39</td></tr><tr><td>SemDeDup</td><td>74.13%</td><td>94.48%</td><td>47.40%</td><td>17.89%</td><td>0.00%</td><td>0.00%</td><td>36.35%</td><td>65.89%</td><td>-0.06</td></tr><tr><td rowspan="6">(nnomm ml Pertd Puupdirs</td><td>UMS (t2)</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>4.39%</td><td>100.00%</td><td>2.80%</td><td>10.12%</td><td>0.00</td></tr><tr><td>UMS (SSCD)</td><td>100.00%</td><td>0.15%</td><td>0.10%</td><td>0.04%</td><td>4.93%</td><td>100.00%</td><td>3.26%</td><td>27.53%</td><td>0.02</td></tr><tr><td>Dar et al.</td><td>71.42%</td><td>98.72%</td><td>N/A</td><td>N/A</td><td>19.23%</td><td>1.29%</td><td>28.58%</td><td>14.54%</td><td>0.10</td></tr><tr><td>SSIM</td><td>71.78%</td><td>99.98%</td><td>83.22%</td><td>2.61%</td><td>0.00%</td><td>0.00%</td><td>29.54%</td><td>11.68%</td><td>0.06</td></tr><tr><td>DeepSSIM</td><td>99.93%</td><td>50.49%</td><td>17.80%</td><td>29.95%</td><td>19.08%</td><td>100.00%</td><td>40.48%</td><td>59.48%</td><td>0.27</td></tr><tr><td>DeepSSIM++ (Proposed)</td><td>96.35%</td><td>96.69%</td><td>83.58%</td><td>84.82%</td><td>69.88%</td><td>58.45%</td><td>81.39%</td><td>91.17%</td><td>0.36</td></tr></table>

## Table 3

Quantitative performance evaluation on the annotated test set using precision and recall for each class, macro F1 score, TPR@5%FPR, and Silhouette score. Results are reported for both perfectly registered pairs and randomly perturbed pairs, using the original decision thresholds reported for UMS and DeepSSIM. The best results for each metric are highlighted with a gray background and bold font, whereas the second-best results are highlighted with a gray background only.

Balle, B., Cherubin, G., Hayes, J., 2021. Reconstructing training data with informed adversaries, in: NeurIPS 2021 Workshop Privacy in Machine Learning. URL: https://openreview.net/forum?id=Yi2DZTbnBl4.

Bonnaire, T., Urfin, R., Biroli, G., Mezard, M., 2026. Why difusion models don’t memorize: The role of implicit dynamical regularization in training, in: The Thirty-ninth Annual Conference on Neural Information Processing Systems. URL: https://openreview.net/forum?id= BSZqpqgqM0.

Breger, A., Karner, C., Selby, I., Gröhl, J., Dittmer, S., Lilley, E., et al., 2025. A study on the adequacy of common iqa measures for medical images, in: Proceedings of 2024 International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2024), Springer Nature Singapore. pp. 451–462.

Bu, C., Liu, Y., Huang, M., Shao, J., Ji, S., Luo, W., et al., 2024. Layer-wise learning rate optimization for task-dependent fine-tuning of pre-trained models: An evolutionary approach. ACM Trans. Evol. Learn. Optim. 4. doi:10.1145/3689827.

Butterick, M., Joseph Saveri Law Firm, 2023. Stable difusion litigation. URL: https://imagegeneratorlitigation.com/.

Cardoso, M.J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., et al., 2022. Monai: An open-source framework for deep learning in healthcare. URL: https://arxiv.org/abs/2211.02701, arXiv:2211.02701.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., Tramèr, F., 2022. Membership inference attacks from first principles, in: 2022 IEEE Symposium on Security and Privacy (SP), pp. 1897–1914. doi:10.1109/ SP46214.2022.9833649.

Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., et al., 2023. Extracting training data from difusion models, in: 32nd USENIX Security Symposium (USENIX Security 23), USENIX Association. pp. 5253–5270. URL: https://www.usenix.org/conference/ usenixsecurity23/presentation/carlini.

Carlini, N., Liu, C., Erlingsson, U., Kos, J., Song, D., 2019. The secret sharer: evaluating and testing unintended memorization in neural networks, in: Proceedings of the 28th USENIX Conference on Security Symposium, USENIX Association. p. 267–284.

Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., et al., 2021. Extracting training data from large language models, in: 30th USENIX Security Symposium (USENIX Security 21), USENIX Association. pp. 2633–2650. URL: https://www.usenix.org/conference/ usenixsecurity21/presentation/carlini-extracting.

Chauvin, L., Kumar, K., Wachinger, C., Vangel, M., de Guise, J., Desrosiers, C., et al., 2020. Neuroimage signature from salient keypoints is highly specific to individuals and shared by close relatives. NeuroImage 204, 116208. doi:https://doi.org/10.1016/j.neuroimage. 2019.116208.

Chen, C., Liu, D., Xu, C., 2024. Towards memorization-free difusion models, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8425–8434. doi:10.1109/CVPR52733.2024. 00805.

Chen, Y., Wang, S., Zou, D., Ma, X., 2026. Side: Surrogate conditional data extraction from difusion models. Proceedings of the AAAI Conference on Artificial Intelligence 40, 128–136. doi:10.1609/aaai.v40i1.36972.

Dar, S.U.H., Ghanaat, A., Kahmann, J., Ayx, I., Papavassiliu, T., Schoenberg, S.O., et al., 2023. Investigating data memorization in 3d latent difusion models for medical image synthesis, in: Deep Generative Models: Third MICCAI Workshop, DGM4MICCAI 2023, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 8, 2023, Proceedings, p. 56–65. doi:10.1007/978-3-031-53767-7\_6.

Dar, S.U.H., Seyfarth, M., Ayx, I., Papavassiliu, T., Schoenberg, S.O., Siepmann, R.M., et al., 2026. Unconditional latent difusion models memorize patient imaging data. Nature Biomedical Engineering 10, 458–472.

European Union, 2016. Regulation (eu) 2016/679 of the european parliament and of the council of 27 april 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing directive 95/46/ec (general data protection regulation). URL: https://eur-lex.europa.eu/eli/reg/2016/ 679/oj.

European Union, 2024. Regulation (eu) 2024/1689 of the european parliament and ofthe council of13june 2024 laying down harmonised rules on artificial intelligence and amending regulations (ec) no 300/2008, (eu) no 167/2013, (eu) no 168/2013, (eu) 2018/858, (eu) 2018/1139 and (eu) 2019/2144 and directives 2014/90/eu, (eu) 2016/797 and (eu) 2020/1828 (artificial intelligence act). URL: https://eur-lex.europa.eu/eli/reg/ 2024/1689/oj/eng.

Feng, Q., Guo, C., Benitez-Quiroz, F., Martinez, A., 2021. When do gans replicate? on the choice of dataset size, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6681–6690. doi:10.1109/ ICCV48922.2021.00663.

Fredrikson, M., Jha, S., Ristenpart, T., 2015. Model inversion attacks that exploit confidence information and basic countermeasures, in:

Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, Association for Computing Machinery. p. 1322–1333. doi:10.1145/2810103.2813677.

Gillotte, J., 2020. Copyright infringement in ai-generated artworks. UC Davis Law Review 53, 2655–2691. URL: https://ssrn.com/abstract= 3657423.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., et al., 2014. Generative adversarial nets, in: Advances in Neural Information Processing Systems (NeurIPS).

Gu, X., Du, C., Pang, T., Li, C., Lin, M., Wang, Y., 2025. On memorization in difusion models. Transactions on Machine Learning Research URL: https://openreview.net/forum?id=D3DBqvSDbj.

Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., et al., 2021. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH) 3, 1–23.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. doi:10.1109/CVPR.2016.90.

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851.

Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q., 2017. Densely connected convolutional networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2261– 2269. doi:10.1109/CVPR.2017.243.

Isensee, F., Schell, M., Pflueger, I., Brugnara, G., Bonekamp, D., Neuberger, U., et al., 2019. Automated brain extraction of multisequence mri using artificial neural networks. Human Brain Mapping 40, 4952–4964. doi:https://doi.org/10.1002/hbm.24750.

Jahanian, A., Puig, X., Tian, Y., Isola, P., 2022. Generative models as a data source for multiview representation learning, in: International Conference on Learning Representations. URL: https://openreview. net/forum?id=qhAeZjs7dCL.

Jiang, P.T., Zhang, C.B., Hou, Q., Wei, Y., 2021. Layercam: Exploring hierarchical class activation maps for localization. IEEE Transactions on Image Processing 30, 5875–5888. doi:10.1109/TIP.2021.3089943.

Johnson, J., Douze, M., Jégou, H., 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 535–547.

Kingma, D.P., Welling, M., 2014. Auto-encoding variational bayes, in: 2nd International Conference on Learning Representations, ICLR 2014, Banf, AB, Canada, April 14-16, 2014, Conference Track Proceedings.

Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M., 2023. Flow matching for generative modeling, in: The Eleventh International Conference on Learning Representations. URL: https://openreview. net/forum?id=PqvMRDCJT9t.

Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A convnet for the 2020s, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11966–11976. doi:10.1109/ CVPR52688.2022.01167.

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations. URL: https: //openreview.net/forum?id=Bkg6RiCqY7.

McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.y., 2017. Communication-Eficient Learning of Deep Networks from Decentralized Data, in: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pp. 1273–1282. URL: https:// proceedings.mlr.press/v54/mcmahan17a.html.

Nasr, M., Rando, J., Carlini, N., Hayase, J., Jagielski, M., Cooper, A.F., et al., 2025. Scalable extraction of training data from aligned, production language models, in: The Thirteenth International Conference on Learning Representations. URL: https://openreview.net/forum?id= vjel3nWP2a.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al., 2019. Pytorch: An imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems. URL: https://proceedings.neurips.cc/paper\_files/paper/2019/ file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

Pinaya, W.H., Tudosiu, P.D., Daflon, J., Da Costa, P.F., Fernandez, V., Nachev, P., et al., 2022. Brain imaging generation with latent difusion models, in: MICCAI Workshop on Deep Generative Models, Springer. pp. 117–126.

Pizzi, E., Roy, S.D., Ravindra, S.N., Goyal, P., Douze, M., 2022. A selfsupervised descriptor for image copy detection, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14512–14522. doi:10.1109/CVPR52688.2022.01413.

Puglisi, L., Alexander, D.C., Ravì, D., 2025. Brain latent progression: Individual-based spatiotemporal disease progression on 3d brain mris via latent difusion. Medical Image Analysis 106, 103734. doi:https: //doi.org/10.1016/j.media.2025.103734.

Radenović, F., Tolias, G., Chum, O., 2019. Fine-tuning cnn image retrieval with no human annotation. IEEE Transactions on Pattern Analysis and Machine Intelligence 41, 1655–1668. doi:10.1109/TPAMI.2018.2846566.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al., 2021. Learning transferable visual models from natural language supervision, in: International conference on machine learning, PmLR. pp. 8748–8763.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10674–10685. doi:10.1109/CVPR52688.2022.01042.

Sargood, A., Puglisi, L., Cole, J., Oxtoby, N., Ravì, D., 2026. Cocolit: Controlnet-conditioned latent image translation for mri to amyloid pet synthesis, in: Proceedings of the AAAI Conference on Artificial Intelligence. URL: https://ojs.aaai.org/index.php/AAAI/article/view/37831.

Scardace, A., Puglisi, L., Guarnera, F., Battiato, S., Ravi, D., 2026. A Novel Metric for Detecting Memorization in Generative Models for Brain MRI Synthesis , in: 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE Computer Society. pp. 3868–3877. doi:10.1109/WACV61042.2026.00377.

Seo, S., Lee, I.K., Kim, H.W., Min, J., Jung, C.H., 2025. Difusion-Based User-Guided Data Augmentation for Coronary Stenosis Detection , in: proceedings of Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, Springer Nature Switzerland. pp. 149 – 169.

Sheller, M.J., Edwards, B., Reina, G.A., Martin, J., Pati, S., Kotrotsou, A., et al., 2020. Federated learning in medicine: facilitating multiinstitutional collaborations without sharing patient data. Scientific Reports 10, 12598. doi:10.1038/s41598-020-69250-1.

Shinohara, R.T., Sweeney, E.M., Goldsmith, J., Shiee, N., Mateen, F.J., Calabresi, P.A., et al., 2014. Statistical normalization techniques for magnetic resonance imaging. NeuroImage: Clinical 6, 9–19. doi:https: //doi.org/10.1016/j.nicl.2014.08.008.

Shokri, R., Stronati, M., Song, C., Shmatikov, V., 2017. Membership inference attacks against machine learning models, in: 2017 IEEE Symposium on Security and Privacy (SP), pp. 3–18. doi:10.1109/SP.2017.41.

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., Goldstein, T., 2023. Difusion art or digital forgery? investigating data replication in difusion models, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6048–6058. doi:10.1109/CVPR52729. 2023.00586.

Tustison, N.J., Cook, P.A., Holbrook, A.J., Johnson, H.J., Muschelli, J., Devenyi, G.A., et al., 2021. The ANTsX ecosystem for quantitative biological and medical imaging. Scientific Reports 11, 9068. doi:10. 1038/s41598-021-87564-6.

U.S. Department of Health and Human Services, 2003. Standards for privacy of individually identifiable health information (hipaa privacy rule). 45 CFR Parts 160 and 164.

Vincent, P., 2011. A connection between score matching and denoising autoencoders. Neural Computation 23, 1661–1674. doi:10.1162/NECO\_ a\_00142.

Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., et al., 2021. Deep high-resolution representation learning for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 3349– 3364. doi:10.1109/TPAMI.2020.2983686.

Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E., 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 600–612. doi:10.1109/TIP.2003.819861.

Yoon, T., Choi, J.Y., Kwon, S., Ryu, E.K., 2023. Difusion probabilistic models generalize when they fail to memorize, in: ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling. URL: https://openreview.net/forum?id=shciCbSk9h.

Zeiler, M.D., Fergus, R., 2014. Visualizing and understanding convolutional networks, in: Computer Vision – ECCV 2014, Springer International Publishing, Cham. pp. 818–833.

Zhang, L., Zhang, L., Mou, X., Zhang, D., 2011. Fsim: A feature similarity index for image quality assessment. IEEE Transactions on Image Processing 20, 2378–2386. doi:10.1109/TIP.2011.2109730.

Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O., 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric , in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE Computer Society. pp. 586–595. doi:10. 1109/CVPR.2018.00068.

Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., et al., 2024. A multimodal biomedical foundation model trained from fifteen million image–text pairs. NEJM AI 2. doi:10.1056/AIoa2400640.

Zhang, Y., Jia, R., Pei, H., Wang, W., Li, B., Song, D., 2020. The secret revealer: Generative model-inversion attacks against deep neural networks, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 250–258. doi:10.1109/CVPR42600.2020.00033.

Ziller, A., Usynin, D., Braren, R., Makowski, M., Rueckert, D., Kaissis, G., 2021. Medical imaging deep learning with diferential privacy. Scientific Reports 11, 13524. doi:10.1038/s41598-021-93030-0.