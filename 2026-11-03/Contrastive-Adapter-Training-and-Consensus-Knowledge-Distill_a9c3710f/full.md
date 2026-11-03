# Contrastive Adapter Training and Consensus Knowledge Distillation for Multi-Source-Free Domain Adaptation in Skin Cancer Diagnosis

Ali Abedi<sup>a</sup>, Q. M. Jonathan Wu<sup>a</sup>, Ning Zhang<sup>a</sup>, Farhad Pourpanah<sup>b</sup>

<sup>a</sup>University of Windsor, Department of Electrical and Computer Engineering, <sup>b</sup>Queen’s University, Department of Electrical and Computer Engineering,

## Abstract

Skin cancer diagnosis, particularly the diferentiation of melanoma from benign nevi, is a vital yet challenging task due to the visual similarity between lesions. Although deep learning models such as convolutional neural networks (CNNs) and vision transformers (ViTs) have demonstrated promising performance, their efectiveness often deteriorates when applied to data from heterogeneous clinical sources. While conventional domain adaptation methods address domain shift, they require access to source data during adaptation, which is often infeasible due to privacy regulations. Multi-sourcefree unsupervised domain adaptation (MSFDA) addresses this limitation by leveraging multiple labeled source domains to generalize to an unlabeled target domain without requiring access to source data, making it suitable for privacy-sensitive medical settings. However, existing MSFDA methods rely on full backbone fine-tuning, leading to catastrophic forgetting and overfitting on small clinical datasets, and address domain shift at the aggregation stage without establishing a shared domain-invariant feature space. Furthermore, their reliance on hard pseudo-labels or confidence-weighted aggregation introduces noisy supervision signals under domain shift. To address these limitations, we propose CAT-CKD, consisting of two components: (1) contrastive adapter training (CAT), which trains lightweight ConvPass adapters within a frozen ViT backbone using supervised contrastive learning (SCL) to establish a shared domain-invariant feature space before source-specific model training, and (2) consensus knowledge distillation (CKD), which aggregates logits from multiple source models into a consensus supervisory signal and adapts a student model on unlabeled target data using KL divergence. Experiments on five publicly available skin lesion datasets show that CAT-

CKD achieves an average AUROC of 86.1%, outperforming existing MSFDA methods while requiring only 4.3M trainable parameters. The code for this paper is available at https://github.com/A-Abedi/CAT\_CKD.

Keywords: Skin cancer diagnosis, Supervised contrastive learning, Unsupervised domain adaptation, Knowledge distillation

## 1. Introduction

Melanoma is one of the most aggressive and potentially fatal forms of skin cancer, responsible for a large number of skin cancer-related deaths worldwide [1, 2]. Early and accurate diagnosis of melanoma is crucial, as it is highly treatable in its initial stages but can rapidly metastasize if left undetected [3]. Diferentiating melanoma from benign skin lesions such as nevi is a fundamental challenge in dermatology due to their similar visual appearance [4, 5]. This motivates the development of computer-aided diagnostic systems to assist clinicians in improving diagnostic accuracy.

In recent years, deep learning techniques, such as convolutional neural networks (CNNs) [6, 7, 8] and vision transformers (ViTs) [9], have shown promising results in medical image analysis, including skin cancer diagnosis [4, 2]. However, the limited availability of annotated data drives researchers to rely on existing labeled datasets collected from multiple clinical centers, as annotation requires scarce domain expertise and is prohibitively timeconsuming [10]. Nevertheless, a major obstacle remains: the domain shift problem. Variations in imaging devices, acquisition protocols, and patient demographics across centers introduce discrepancies between source and target domains, which substantially degrade the generalizability of deep learning models when deployed in previously unseen clinical data [11, 12, 13, 14]. Compounding this challenge, privacy regulations and institutional policies [15] typically prevent source data from being revisited once a model is deployed to a new clinical environment [16].

Multi-source-free unsupervised domain adaptation (MSFDA) addresses domain shift by leveraging labeled data from multiple source domains to generalize to an unlabeled target domain without requiring access to source data during adaptation. This setup aligns better with real-world clinical settings, where privacy regulations often prevent access to source datasets during deployment [17, 18]. While promising, existing MSFDA methods suffer from two key limitations. First, they rely on full backbone fine-tuning during adaptation, which causes catastrophic forgetting of pretrained representations and overfitting on the small, imbalanced datasets typical of clinical settings. Second, they attempt to reconcile domain shift at the aggregation or adaptation stage [19], without first establishing a shared domain-invariant backbone, which if available, could be frozen during source-specific model training, reducing both overfitting and computational overhead. Furthermore, their reliance on hard pseudo-labels that directly propagate teacher errors [20], or confidence-weighted aggregation that amplifies errors from overconfident teachers [21, 22], introduces noisy supervision signals that are unreliable under the domain shift conditions inherent to multi-center clinical data.

To address the challenge of noisy supervision under domain shift, knowledge distillation (KD) [23] can facilitate the transfer of information from well-trained source models to a target model, which enables efective learning without direct access to the source data [24, 25]. Using multiple teacher models allows KD to mitigate individual source biases and provide more robust supervision signals for the student model. The success of this process depends heavily on the quality of the teacher models, which must be carefully fine-tuned to ensure they provide accurate and consistent guidance.

Addressing the absence of a shared domain-invariant backbone requires a training stage that leverages all available labeled source data jointly with an objective that explicitly encourages same-class samples from diferent domains to be close together in the feature space. However, directly fine-tuning large pretrained ViTs for this purpose is computationally expensive and prone to overfitting and catastrophic forgetting, particularly when labeled data are scarce and imbalanced across clinical centers. Parameter-eficient fine-tuning (PEFT) methods [26], including adapters [27, 28] and low-rank adaptation (LoRA) [29, 30], ofer a practical solution by updating only a small subset of parameters while keeping the backbone frozen, thereby preserving pretrained knowledge and reducing the risk of overfitting. This makes PEFT particularly well-suited for building a shared domain-invariant backbone in the MSFDA setting, where labeled data are limited and privacy constraints prevent repeated access to source data during deployment.

This paper introduces CAT-CKD, an MSFDA framework for skin cancer diagnosis that directly addresses the limitations of prior methods through two complementary components. First, we introduce Contrastive Adapter Training (CAT), a PEFT strategy that inserts convolutional bypass adapters (ConvPass) [28] into a frozen ViT backbone and trains them using supervised contrastive learning (SCL) [31], which encourages intra-class compactness and inter-class separation across domains regardless of their origin, to establish a shared domain-invariant feature space before source-specific models are built. Second, we design a Consensus Knowledge Distillation (CKD) strategy, in which the logits from multiple trained source models are aggregated into a consensus supervision signal, which we show reduces logit-level variance compared to any single teacher, and used to adapt a student model on the unlabeled target domain via Kullback–Leibler (KL) divergence, avoiding the unreliable confidence estimates that plague existing aggregation strategies under domain shift. Because adaptation relies solely on trained source models, the proposed framework avoids direct access to source-domain images during deployment. Extensive experiments demonstrate that CAT-CKD provides a practical and parameter-eficient solution for source-free adaptation in skin cancer diagnosis under data-sharing constraints.

The main contributions of this work include:

• We propose CAT-CKD, an MSFDA framework for skin cancer diagnosis that explicitly establishes a shared domain-invariant backbone before source-specific models are built, directly addressing the representationlevel and supervision-level limitations of existing methods.

• We introduce CAT, which trains ConvPass adapters using supervised contrastive learning across source domains in the MSFDA setting, producing a shared domain-invariant feature space while avoiding catastrophic forgetting and overfitting.

• We design CKD, which aggregates logits from multiple source models into a theoretically grounded consensus supervisory signal and adapts a student model on unlabeled target data using a KL-divergence ob jective, providing more stable supervision than existing aggregation strategies.

• We conduct extensive experiments, ablation studies, and eficiency analyses on five public skin lesion datasets, demonstrating competitive performance with substantially fewer trainable parameters than existing methods.

The remainder of this paper is organized as follows. Section 2 reviews related work in skin cancer diagnosis, domain adaptation, parameter-eficient fine tuning, and knowledge distillation. Section 3 presents the problem formulation and details the proposed CAT-CKD framework. Section 4 describes the experimental setup and discusses the results. Finally, Section 5 concludes the paper and outlines directions for future work.

## 2. Related Work

## 2.1. Skin Cancer Diagnosis

Recent research has increasingly leveraged deep learning methods to distinguish melanoma from benign nevi. CNNs are the most widely used techniques, typically fine-tuned from ImageNet-pretrained architectures such as DenseNet [8], which uses dense connections between layers to encourage feature reuse, ResNet [6], which introduces residual connections to enable training of very deep networks, and MobileNet [7], which employs depthwise separable convolutions for lightweight and eficient inference. In [4], a pre-trained CNN model is fine-tuned on over 120,000 dermoscopic images to classify lesions as either melanoma or benign, using a softmax-generated malignancy probability threshold of 0.5. The study in [2] fine-tunes four pretrained CNN architectures, including DenseNet121, ResNet50V2, NAS-NetMobile, and MobileNetV2, by applying extensive data augmentation to a dataset of 8,825 images. However, CNNs primarily focus on local features due to their limited receptive fields, which restricts their ability to capture long-range dependencies and global context [32].

To overcome the limitations of CNNs, recent studies have explored ViTs, which employ self-attention mechanisms [33] to model relationships between image regions. This design enables ViTs to integrate information across both local and long-range spatial dependencies. Their success across a wide range of vision tasks has generated growing interest in applying ViTs to skin cancer diagnosis. RETFound [5] fine-tunes a ViT model to classify ultrawidefield color and autofluorescence fundoscopy images without the need for augmentation or handcrafted features. In [34], architectures such as ConvNeXt, ViT-Base-16, and Swin V2 leverage self-attention and patch-based designs to enhance diagnostic performance. Additionally, GAN-MBViT [35] employs a dual-branch training strategy to boost feature learning by combining StyleGAN2-ADA for generating melanoma images with a ViT enhanced by BatchFormer for classification. Finally, ScATNet [36] applies a multi-scale ViT to histopathology slides by using self-attention across image patches at diferent magnifications, enabling the efective fusion of features from various tissue scales. Despite these advances, existing skin cancer diagnosis methods typically do not address the domain shift that arises when models are deployed across heterogeneous clinical sources.

## 2.2. Unsupervised Domain Adaptation

UDA aims to transfer knowledge from a labeled source domain to an unlabeled target domain with a diferent data distribution. DANN [37] learns domain-invariant and task-discriminative features by introducing a domain classifier connected through a gradient reversal layer. It reverses gradients during backpropagation to encourage indistinguishability between source and target domains. The method jointly trains the label predictor and domain classifier in an adversarial manner. ADDA [38] first learns a discriminative feature mapping from source data and then trains a separate target encoder to map target data into the same space through adversarial training against a domain discriminator. It uses unshared weights and a GAN-based loss to align source and target representations without requiring image generation or weight sharing. Beyond single-source UDA, many tasks require adaptation from multiple sources. MDAN [39] derives new generalization bounds for multiple source domain adaptation that motivate jointly learning domaininvariant, task-discriminative features. Two variants are proposed: a hard version optimizing a worst-case bound and a soft version using log-sum-exp weighting to adaptively combine gradients from all source domains.

While multi-source unsupervised domain adaptation requires access to source data, multi-source-free variants rely solely on trained source models and are therefore better suited to real-world scenarios. DECISION [20] addresses MSFDA by leveraging only pre-trained source models and unlabeled target data to construct a target model through adaptive ensembling. It jointly optimizes model-specific feature extractors and combines weights through information maximization and pseudo-labeling. This design ensures that the target model performs at least as well as the best individual source model. EAAF [22] introduces two uncertainty measures: Evidential Prediction Uncertainty (EPU), which estimates how well each source model fits a given target sample to guide instance-level aggregation, and Evidential Adjacency-Consistent Uncertainty (EAU), which measures semantic consistency within local neighborhoods to enable reliable semantic propagation, all within a dual-branch Multinomial-Dirichlet network. Bi-ATEN [21] learns instance-specific and domain-consistent ensemble weights through a bi-level attention mechanism. It requires only minimal tuning of the source bottlenecks. While these methods demonstrate promising adaptation performance, they either require source data during adaptation or address domain shift only at the aggregation stage.

## 2.3. Parameter-Eficient Fine Tuning

PEFT methods aim to adapt large pretrained models to downstream tasks using a small number of trainable parameters. Early PEFT strategies for language models have inspired several adaptation techniques for vision models. Adapters [40] insert lightweight bottleneck modules between transformer layers, LoRA [29] decomposes weight updates into low-rank matrices to reduce trainable parameters, and Prefix-Tuning [41] prepends trainable tokens to the input sequence to guide model behavior, all while keeping the backbone frozen. Vision-specific PEFT variants have also shown competitive or superior performance to full fine-tuning, especially in low-data regimes. VPT [42] prepends learnable prompt tokens to the input of each transformer layer to adapt the model to downstream tasks. AdaptFormer [43] replaces the MLP layers in ViT with lightweight adapter modules for scalable visual recognition. NOAH [44] searches for optimal combinations of PEFT modules through neural architecture search to maximize adaptation performance. While methods like Med-SA [45] apply PEFT principles to interactive segmentation in medical imaging, they rely on user prompts and are not designed for multi-source-free domain adaptation. Most PEFT approaches also remain language-oriented, inheriting architectural biases suitable for NLP, such as sequence handling and weak spatial priors. Convpass [28] addresses this issue by proposing a vision-oriented PEFT method. It introduces convolutional bypasses in parallel with the multi-head self-attention (MHSA) and MLP blocks within ViT layers. These modules restore spatial locality by applying convolutional transformations to the flattened patch embeddings to bridge the gap between CNNs and transformers.

## 2.4. Knowledge Distillation

KD has become a widely adopted approach for compressing deep models by transferring knowledge from a large, well-performing teacher model to a smaller student model. Hinton et al. [23] introduced KD as a technique to transfer the generalization ability of a large or ensemble model (the teacher) to a smaller model (the student) by training the student on soft targets.

KDCL [46] is an online KD framework where multiple student networks collaboratively learn by sharing dynamically generated ensemble soft targets. This approach enhances generalization and does not rely on a pre-trained teacher. Attention transfer [47] is a KD method that improves student CNNs by forcing them to mimic the teacher’s spatial attention maps. While most early works used a single teacher, recent methods leverage multiple teachers to provide a broader supervision signal to the student. For example, AE-KD [48] formulates ensemble KD as a multi-objective optimization in gradient space. It adaptively weights teacher contributions based on their agreement to guide the student more efectively. MMKD [49] introduces a meta-weight network that adaptively combines multiple teachers’ logits and feature representations at the instance level using second-order optimization to ensure that the student receives diverse yet compatible knowledge tailored to its learning capacity. FEED [50] distills knowledge from multiple teachers at the feature-map level by introducing separate non-linear transformation layers for each teacher. This design allows the student network to align and absorb ensemble feature representations either in parallel or through sequential training.

## 3. Problem Formulation and Proposed Solution

## 3.1. Problem Formulation

Consider M source models, each possessing a dataset derived from a unique data distribution and distinct source domain $\mathcal { D } _ { i } ^ { S } ~ = ~ \{ x _ { i j } , y _ { i j } \} _ { j = 1 } ^ { N _ { i } } ;$ where $x _ { i j } \in \mathcal { X } _ { i }$ and $y _ { i j } \in \mathcal { D } _ { i }$ represent the $j ^ { t h }$ input and its corresponding label for the $i ^ { t h }$ source domain. These datasets are distributionally diverse, with each dataset sampled from a distinct joint distribution $P _ { i } ( X , Y )$ , such that $P _ { i } \neq P _ { k }$ for $i \neq k$ $N _ { i }$ denotes the number of samples of the $i ^ { t h }$ source domain $( i = 1 , 2 , \dots , M )$

Our objective is to train a model that performs well on an unlabeled target domain $\mathcal { D } ^ { \check { T } } = \{ x _ { j } ^ { T } \} _ { j = 1 } ^ { N _ { T } }$ , where $N _ { T }$ denotes the number of target samples, by leveraging a set of source models trained on labeled source domains. Each source model $f _ { i }$ is trained on its respective labeled dataset by minimizing the following objective:

$$
\operatorname* { m i n } \mathcal { L } _ { i } ( f _ { i } ) = \frac { 1 } { N _ { i } } \sum _ { j = 1 } ^ { N _ { i } } \ell ( f _ { i } ( x _ { i j } ) , y _ { i j } ) ,\tag{1}
$$

![](images/2bd5743091cb7ffcacd9efc15b168a74fe95f2e4c3069cc38e9091e35d120abb.jpg)  
Figure 1: Overview of the proposed CAT-CKD framework. (a) CAT: ConvPass adapters are trained on pooled source-domain data using SCL, while the ViT backbone remains frozen to form a shared domain-invariant feature space. Domain and class imbalance are addressed via weighted sampling. (b) Source model training: Each domain independently trains a bottleneck and classification head on the frozen backbone with binary CE loss. (c) CKD: Source model logits on unlabeled target samples are averaged to form a consensus signal, which supervises the student model through KL divergence. Samples with the same border color belong to the same source domain.<sup>1</sup>

where ℓ(·) denotes the binary cross-entropy loss. Once trained, these source models are used in the adaptation phase to guide the learning of a domainadapted model without requiring access to the original source data.

## 3.2. Model Overview

Figure 1 provides an overview of the proposed CAT-CKD framework for skin-cancer diagnosis. The framework operates on data from multiple source domains that may exhibit domain shift due to variations in imaging protocols, patient demographics, and annotation practices.

CAT-CKD consists of two stages. To obtain class-discriminative and domain-invariant representations, CAT (Section 3.3) trains a ViT-based feature extractor $\theta ^ { S }$ using labeled data from all source domains, optimizing only the adapter parameters $\theta _ { \omega ^ { \prime } } ^ { S }$ with SCL [31] while keeping the ViT [9] backbone parameters $\theta _ { \omega } ^ { S }$ frozen. Domain- and class-level weighting is applied to reduce the impact of imbalance. The resulting backbone is then frozen and shared across the M source models; each source model attaches a domain-specific parameter set $\theta ^ { i } = \{ \theta _ { \psi } ^ { i } , \theta _ { \phi } ^ { i } \}$ , where $\theta _ { \psi } ^ { i }$ and $\theta _ { \phi } ^ { i }$ denote the bottleneck and classification head, respectively, and is trained separately using only its local labeled data.

In the second stage, the trained source models act as an ensemble of teachers for an unlabeled target domain. CKD (Section 3.4) aggregates their predictions on target samples by averaging to form a consensus supervisory signal. A student model is then trained by aligning its outputs with this consensus via KL divergence [23] by distilling knowledge from all sources into a single target-adapted model. Since adaptation relies only on trained source models, the framework avoids direct access to source-domain data during deployment, aligning with data-sharing constraints in medical applications.

## 3.3. Contrastive Adapter Training

CAT is a PEFT approach that incorporates adapters into the ViT architecture to mitigate the catastrophic forgetting challenge. These adapters are convolutional bypass modules [28] added in parallel to the existing MHSA and MLP blocks which allows eficient adaptation while preserving the original backbone structure. The modified transformation process in ViT with adapters is formulated as:

$$
X ^ { \prime } \qquad \qquad X \quad + \quad \mathrm { M H S A / M L P ( L N ( } X \backslash ) \quad + \quad \mathrm { A d a p t e r ( L N ( } X \backslash ) ) ,\tag{2}
$$

where LN(·) is the layer normalization function, MHSA/MLP(·) represents the self-attention and feed-forward layers in ViT, and Adapter(·) denotes the Convpass adapter module [28]. By training only the adapter parameters while keeping the ViT backbone frozen, we reduce the number of trainable parameters and minimize the risk of overfitting. This makes the model more suitable for domain adaptation in medical imaging.

The goal of CAT is to learn domain-invariant feature representations so that samples from the same class, even when originating from diferent domains, are close together in the feature space, while samples from diferent classes are kept far apart. To achieve this, we train the adapters using $\mathrm { { S C L , } }$ which encourages intra-class compactness and inter-class separation. The loss function is defined as:

$$
\mathcal { L } _ { \mathrm { s c l } } = \sum _ { i \in \mathcal { I } } \frac { - 1 } { | \mathcal { P } ( i ) | } \sum _ { p \in \mathcal { P } ( i ) } \log \frac { \exp ( z _ { i } \cdot z _ { p } / \tau ) } { \sum _ { a \in \mathcal { T } \backslash \{ i \} } \exp ( z _ { i } \cdot z _ { a } / \tau ) } ,\tag{3}
$$

where $\mathcal { T }$ represents the index set of samples in a mini-batch, $z _ { i } = f ( x _ { i } ; \theta ^ { S } )$ denotes the feature embedding of sample i extracted by the backbone parameterized by $\theta ^ { S }$ , and $\mathcal { P } ( i )$ corresponds to the set of positive samples that share the same class as i. The denominator includes all samples except i to normalize the contrastive similarity scores. The temperature parameter τ controls the sharpness of similarity scores to adjust the impact of diferent distances in the representation space. The similarity between embeddings is computed using a dot product to ensure that positive pairs are pulled closer together while negative pairs are pushed further apart.

Given that SCL forms positive pairs within each mini-batch, severe class and domain imbalance can bias positive pair construction toward majority groups, thereby reducing the contribution of minority classes or underrepresented domains. To mitigate this issue during CAT training, we assign each training sample a weight based on its domain and class frequency, and use these weights to construct mini-batches through weighted random sampling.

Let the pooled source dataset used in CAT contain N samples from M source domains and C classes. For each domain $i \in \{ 1 , \ldots , M \}$ , let $N _ { i }$ denote the number of samples in domain $i ,$ and let $n _ { i j }$ denote the number of samples of class $j \in \{ 1 , \ldots , C \}$ within domain i. We compute the domain weight and the class weight (within each domain) as inverse-frequency terms:

$$
w _ { d } ^ { i } = \frac { N } { N _ { i } } ,\tag{4}
$$

$$
w _ { c } ^ { i j } = \frac { N _ { i } } { n _ { i j } } .\tag{5}
$$

For a sample that belongs to domain i and class $j ,$ the final sampling weight is defined as:

$$
w ^ { i j } = w _ { d } ^ { i } w _ { c } ^ { i j } .\tag{6}
$$

Substituting the definitions above yields:

$$
w ^ { i j } = \frac { N } { N _ { i } } \cdot \frac { N _ { i } } { n _ { i j } } = \frac { N } { n _ { i j } } ,\tag{7}
$$

which shows that the efective weight is proportional to the inverse frequency of the corresponding (domain $i ,$ class $j )$ combination. These weights are then used to construct mini-batches via weighted random sampling, so that samples from rare domain–class pairs are selected more frequently during CAT training. The training process for CAT is shown in Algorithm 1.

Algorithm 1 Contrastive Adapter Training (CAT)   
Require: Source datasets $\{ \mathcal { D } _ { i } ^ { S } \} _ { i = 1 } ^ { M }$ , ViT backbone parameters $\theta _ { \omega } ^ { S }$ , tempera  
ture $\tau$   
Ensure: Frozen feature extractor $\theta _ { \textrm { \tiny a } } ^ { S } = \{ \theta _ { \omega } ^ { S } , \theta _ { \omega ^ { \prime } } ^ { S } \}$ with trained adapters   
1: Initialize $\theta _ { \omega ^ { \prime } } ^ { S }$ (adapters); freeze $\theta _ { \omega } ^ { S }$   
2: Compute sampling weights $w ^ { i j }$ from the pooled source set $\cup _ { i = 1 } ^ { M } \mathcal { D } _ { i } ^ { S }$   
(Eq. (6))   
3: for each mini-batch $B = \{ ( x _ { j } , y _ { j } , d _ { j } ) \}$ sampled from $\cup _ { i = 1 } ^ { M } \mathcal { D } _ { i } ^ { S }$ using $w ^ { i j }$   
do   
4: Extract embeddings $z _ { j } = f ( x _ { j } ; \theta ^ { S } )$ with frozen $\theta _ { \omega } ^ { S }$ and trainable $\theta _ { \omega ^ { \prime } } ^ { S }$   
5: for each anchor $j \in B$ do   
6: Construct the positive set ${ \mathcal { P } } ( j ) = \{ k \in B \setminus \{ j \} \mid y _ { k } = y _ { j } \}$   
7: end for   
8: Compute the supervised contrastive loss $\mathcal { L } _ { \mathrm { s c l } }$ using Eq. 3 (temperature   
$\tau )$   
9: Update $\theta _ { \omega ^ { \prime } } ^ { S }$ by backpropagating $\nabla _ { \theta _ { \omega ^ { \prime } } ^ { S } } \mathcal { L } _ { \mathrm { s c l } }$   
10: end for   
11: return $\theta ^ { S } = \{ \theta _ { \omega } ^ { S } , \theta _ { \omega ^ { \prime } } ^ { S } \}$

After training the backbone with CAT, the source models are trained independently. To promote eficient learning and reduce overfitting, the trained backbone is kept frozen. For each source domain i, we attach a domain-specific bottleneck with parameters $\theta _ { \psi } ^ { i }$ and a classification head with parameters $\theta _ { \phi } ^ { i }$ on top of the frozen backbone, and optimize these layers using the binary cross-entropy loss, as follows:

$$
\mathcal { L } _ { B C E } = - \frac { 1 } { N _ { i } } \sum _ { i = 1 } ^ { N _ { i } } \left[ y _ { i j } \log \hat { y } _ { i j } + ( 1 - y _ { i j } ) \log ( 1 - \hat { y } _ { i j } ) \right] ,\tag{8}
$$

where $\hat { y } _ { i j }$ denotes the predicted probability for the positive class. Once the source models are trained using their respective data, direct access to source data is no longer available due to privacy constraints.

## 3.4. Consensus Knowledge Distillation

CKD adapts a student model to the unlabeled target domain by treating the trained source models as a set of frozen teachers that provide soft supervisory signals. Rather than relying on a single teacher, CKD aggregates the predictions of all M teachers to obtain a consensus target, which helps reduce the influence of any individual source-specific bias and provides a more stable supervision signal under domain shift. Moreover, using soft labels preserves inter-class similarity information and provides smoother gradients than hard pseudo-labels. This also mitigates hard-label bias when the consensus prediction is uncertain or incorrect (e.g., when teachers disagree), since the student is encouraged to match the full probability distribution rather than commit to a potentially erroneous one-hot target.

Given an unlabeled target sample $x _ { j } ^ { T }$ , each teacher $f _ { i }$ produces a logit vector $a _ { j } ^ { i } = f _ { i } ( x _ { j } ^ { T } )$ . We form a consensus logit representation by averaging teacher logits,

$$
\bar { a } _ { j } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } a _ { j } ^ { i } ,\tag{9}
$$

which provides a single soft supervisory signal that reflects agreement across teachers for sample $j$ . The student model $f _ { s } ( \cdot )$ is then optimized to align its predictive distribution with this consensus by minimizing the KL divergence

$$
\mathcal { L } _ { \mathrm { C K D } } = \frac { 1 } { N _ { T } } \sum _ { j = 1 } ^ { N _ { T } } D _ { \mathrm { K L } } \big ( \sigma ( \bar { a } _ { j } ) \ | | \ \sigma ( f _ { s } ( x _ { j } ^ { T } ) ) \big ) ,\tag{10}
$$

where $\sigma ( \cdot )$ denotes the softmax function, $N _ { T }$ is the number of target samples, and $D _ { \mathrm { K L } } ( \cdot \parallel \cdot )$ is defined as:

$$
D _ { \mathrm { K L } } ( P \parallel Q ) = \sum _ { c = 1 } ^ { C } P _ { c } \log \frac { P _ { c } } { Q _ { c } } ,\tag{11}
$$

where $P _ { c } = \sigma ( \bar { a } _ { j } ) .$ and $Q _ { c } = \sigma ( f _ { s } ( x _ { j } ^ { T } ) )$ denote the class-c probabilities. By training against the consensus soft target $\sigma ( \bar { a } _ { j } )$ , the student distills shared knowledge across multiple sources while avoiding dependence on a single teacher, thereby improving robustness to heterogeneous source domains. Since adaptation relies only on teacher outputs for target samples and does not require access to source-domain images, CKD aligns with source-data-free deployment constraints in medical applications. The CKD procedure is summarized in Algorithm 2.

Algorithm 2 Consensus Knowledge Distillation (CKD)   
Require: Trained source models $\left\{ f _ { 1 } , f _ { 2 } , . . . , f _ { M } \right\}$ , target data $\mathcal { D } ^ { T }$   
Ensure: Adapted student model $f _ { s }$   
1: Randomly initialize student model $f _ { s }$   
2: for each mini-batch $\{ x _ { j } ^ { T } \}$ from $\mathcal { D } ^ { T }$ do   
3: for each $x _ { j } ^ { T }$ in the mini-batch do   
4: for $i = \mathrm { { \dot { 1 } } }$ to M do   
5: Compute teacher logits: $a _ { j } ^ { i } = f _ { i } ( x _ { j } ^ { T } )$   
6: end for   
7: Compute consensus logits ${ \bar { a } _ { j } }$ using Eq. 9   
8: end for   
9: Compute ${ \mathcal { L } } _ { \mathrm { C K D } }$ using Eq. 10 and update $f _ { s }$ by backpropagation   
10: end for   
11: return Trained student model $f _ { s }$

## 3.5. Theoretical Justification

To show why consensus distillation can provide a stable supervisory signal under domain shift, a brief analysis is provided under standard assumptions.

Assumption 1. For an input image x and its corresponding class label y, let $\ell ( x ) \in \mathbb { R } ^ { C }$ denote a latent logit representation such that $\sigma ( \ell ( x ) ) = P ( y \mid x )$ Each teacher $f _ { i }$ produces a logit vector that can be shown as:

$$
a _ { j } ^ { i } = \ell ( x _ { j } ^ { T } ) + \epsilon _ { i } ( x _ { j } ^ { T } ) , \quad \forall i \in \{ 1 , \ldots , M \} ,\tag{12}
$$

where $\epsilon _ { i } ( \cdot )$ captures the teacher-specific logit error on the target domain.

Lemma 1. If the teacher logit errors have bounded second moment and are not perfectly correlated, then the consensus logit ${ \bar { a } _ { j } }$ reduces the variance of the aggregated logit error.

Proof. Using Assumption 1 and the definition of the consensus logit in Eq. 9, it follows that:

$$
\bar { a } _ { j } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } a _ { j } ^ { i } = \ell ( \boldsymbol { x } _ { j } ^ { T } ) + \bar { \epsilon } ( \boldsymbol { x } _ { j } ^ { T } ) ,\tag{13}
$$

where

$$
\bar { \epsilon } ( x _ { j } ^ { T } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \epsilon _ { i } ( x _ { j } ^ { T } ) .\tag{14}
$$

The mean-squared magnitude of the aggregated error satisfies:

$$
\mathbb { E } \left[ \| \bar { \epsilon } ( x _ { j } ^ { T } ) \| _ { 2 } ^ { 2 } \right] = \frac { 1 } { M ^ { 2 } } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { M } \mathbb { E } \left[ \langle \epsilon _ { i } ( x _ { j } ^ { T } ) , \epsilon _ { k } ( x _ { j } ^ { T } ) \rangle \right] .\tag{15}
$$

When the cross-correlations $\mathbb { E } [ \langle \epsilon _ { i } , \epsilon _ { k } \rangle ]$ for $i \neq$ k are limited $( \mathrm { i . e . }$ , errors are not perfectly aligned), the above quantity decreases as M increases, indicating that ${ \bar { a } _ { j } }$ is a more stable logit-level supervisory signal than $a _ { j } ^ { i }$ from any single teacher. □

Lemma 2. For a fixed consensus logit ${ \bar { a } _ { j } }$ , minimizing $D _ { K L } ( \sigma ( \bar { a } _ { j } ) \parallel q _ { j } )$ with respect to the student predictive distribution $q _ { j }$ yields the unique optimum $q _ { j } = \sigma ( \bar { a } _ { j } )$

Proof. For any fixed $\bar { a } _ { j } , D _ { \mathrm { K L } } ( \sigma ( \bar { a } _ { j } ) \parallel q _ { j } )$ is minimized when $q _ { j }$ matches the target distribution $\sigma ( \bar { a } _ { j } )$ □

Combining Lemma 1 and Lemma 2, training the student by minimizing $D _ { \mathrm { K L } } ( \sigma ( \bar { a } _ { j } ) \parallel \sigma ( f _ { s } ( x _ { j } ^ { T } ) ) )$ encourages the student to match a consensus target whose underlying logit error can be less variable than individual teacher logits when teachers exhibit complementary errors. This provides intuition for why CKD can improve robustness to heterogeneous source domains and domain shift.

## 4. Experiments and Results

## 4.1. Datasets

In this study, we utilized five publicly available datasets for skin lesion analysis: ISIC 2018 [51, 52], ISIC 2020 [53], PAD-UFES-20 [54], Derm7pt-Derm [55], and Derm7pt-Clinic [55]. Images of melanoma and nevus were selected from each dataset because these two classes are the most challenging to diferentiate and are the most clinically significant for skin-lesion diagnosis. In addition, all of the datasets used in this study contain samples from these two classes, ensuring consistency across domains.

The Derm7pt dataset is partitioned into two subsets which are Derm7pt-Derm (dermoscopic images) and Derm7pt-Clinic (clinical images)[56]. This split is necessary because the subsets come from diferent imaging modalities. Consequently, they exhibit distinct data distributions that influence the domain-adaptation process. The number of melanoma and nevus samples in each dataset are summarized in Table 1. As it can be seen, the datasets exhibit substantial class and sample-size imbalance, closely resembling deployment scenarios in clinical practice.

Table 1: Summary of the number of samples in skin lesion datasets used in the study.
<table><tr><td>Dataset</td><td>Melanoma</td><td>Nevus</td></tr><tr><td>ISIC 2018 [51, 52]</td><td>1113</td><td>6705</td></tr><tr><td>ISIC 2020 [53]</td><td>584</td><td>5193</td></tr><tr><td>PAD-UFES-20 [54]</td><td>52</td><td>244</td></tr><tr><td>Derm7pt-Derm [55]</td><td>90</td><td>256</td></tr><tr><td>Derm7pt-Clinic [55]</td><td>90</td><td>256</td></tr></table>

The selected datasets reflect real-world variability in imaging modalities, acquisition settings, and clinical sources, that introduces natural domain shifts essential for evaluating MSFDA methods. Their independent collection aligns with source-free constraints, where source data cannot be shared. To further quantify inter-dataset domain shift, Figure 2 reports the pairwise Fréchet Distance between feature distributions. This highlights the significant visual and statistical domain shift between datasets used in this study.

## 4.2. Implementation Details

For all experiments, we use a batch size of 32 across all datasets to maintain consistency. ViT-B/16 pre-trained on ImageNet is used as the backbone across all experiments. The source models are trained for 20 epochs to ensure suficient learning from the data, while the student model is trained for 15 epochs using the proposed CKD framework. The learning rate is set to

![](images/125de1081a502bfbd78ef8a8c93df90346dc37799915c30fa1b65ea6654a9600.jpg)  
Figure 2: Pairwise Fréchet Distance between feature distributions of the five skin lesion datasets, where higher values indicate greater domain shift. The substantial distances observed across dataset pairs confirm the presence of significant inter-domain variability. This result motivates the use of domain adaptation.

$3 \times 1 0 ^ { - 2 }$ , with a warmup phase of 5% of the total training steps to stabilize the optimization process during the initial epochs. SGD is the optimizer, configured with a momentum of 0.9 to enhance convergence by maintaining directional consistency during updates. τ in Eq. 3 is set to 0.2 based on the details provided in Section 4.6.5. The bottleneck of our method contains four fully connected layers comprising 2048, 1024, 512, and 256 nodes, respectively. The head consists of a single fully connected layer. We treat each dataset as an independent domain and employ a leave-one-domain-out protocol, where, in each experimental setting, one domain is selected as the unlabeled target and all other domains are used as labeled sources.

Data Augmentation. To enhance the robustness and generalization capability of our model, we apply a series of data augmentation techniques during the training process (see Figure 3). These augmentations simulate diverse imaging conditions, which are common in real-world scenarios, thereby enabling the model to better handle variability in the data. Specifically, we use resizing to ensure uniform input dimensions, random horizontal and vertical flips to introduce variability in orientation, and random rotations within a range of 0 to 90 degrees to account for angular variations in imaging setups. These transformations are implemented as part of a pre-processing pipeline.

![](images/5f8d252b4172def6befb09deff5c5e7b60c752ecbffb4e16aa466da850e03c48.jpg)  
Figure 3: Demonstration of data augmentations applied to a skin lesion images in our proposed method.

## 4.3. Baselines

To ensure a comprehensive evaluation, we select several well-established state-of-the-art models as baselines. Specifically, we consider both singlesource and multi-source UDA methods, as well as multi-source-free UDA methods. First, we include DANN [37] and ADDA [38], two widely recognized UDA models. Since these models are designed for single-source domain adaptation, we combine all source datasets into a single dataset and use a separate dataset as the target domain to align with their training paradigm.

This setup allows for a fair comparison of their efectiveness in a multi-source setting. Next, we consider MDAN [39], a well-known multi-source domain adaptation method that explicitly handles multiple source domains. For comparison with our proposed MSFDA approach, we select DECISION [20], Bi-ATEN [21], and EAAF [22], all of which are specifically designed for scenarios where source data is unavailable during adaptation. Since our method operates under similar privacy-preserving constraints, these baselines provide an appropriate benchmark for evaluating the efectiveness of our proposed framework. All models are evaluated on the same datasets and data distributions to ensure a fair comparison.

## 4.4. Results

Evaluation. We evaluate all models using the Area Under the Receiver Operating Characteristic Curve (AUROC), a threshold-independent metric suited for imbalanced datasets like skin cancer diagnosis. AUROC measures the model’s ability to rank positive samples higher than negative ones by computing the area under the ROC curve, defined as:

$$
\mathrm { A U R O C } = \int _ { 0 } ^ { 1 } \mathrm { T P R } ( t ) d \mathrm { F P R } ( t ) ,\tag{16}
$$

where True Positive Rate (TPR) and False Positive Rate (FPR) are:

$$
\mathrm { T P R } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } , \quad \mathrm { F P R } = \frac { \mathrm { F P } } { \mathrm { F P } + \mathrm { T N } } .\tag{17}
$$

Unlike accuracy, AUROC remains robust to class imbalance. This makes the AUROC ideal for medical imaging tasks.

Performance. Table 2 presents the AUROC scores for CAT-CKD and the competing baselines. To further illustrate the severity of domain shift across datasets, we include a Source Only baseline, in which all source domains are pooled and a single model is trained using standard cross-entropy without any domain adaptation. The results demonstrate that CAT-CKD achieves strong performance across datasets in the MSFDA setting for skin cancer diagnosis. Importantly, the framework operates without access to source data during adaptation, thereby preserving data privacy. Despite this constraint, the method maintains robust generalization across domains, supporting its suitability for real-world clinical deployment. Notably, these results are obtained with a substantially reduced number of trainable parameters.

Table 2: AUROC comparison of domain adaptation models on skin lesion datasets. Best results are shown in bold.
<table><tr><td>Model</td><td>MS</td><td>SF</td><td>ISIC 2018</td><td>ISIC 2020</td><td> PAD-UFES-20</td><td>Derm7pt-Derm</td><td>Derm7pt-Clinic</td><td> $\mathbf { A v g . }$ </td></tr><tr><td>Source Only</td><td>X</td><td>X</td><td> $7 8 . 3 \pm 4 . 8$ </td><td> $5 4 . 1 \pm 6 . 5$ </td><td> $4 5 . 8 \pm 7 . 4$ </td><td> $5 0 . 8 \pm 5 . 1$ </td><td> $4 8 . 6 \pm 2 . 3$ </td><td> $5 5 . 5 \pm 5 . 2$ </td></tr><tr><td>ADDA [38]</td><td>X</td><td>X</td><td> $9 2 . 1 \pm 0 . 8$ </td><td> $8 3 . 0 \pm 0 . 4$ </td><td> $7 2 . 5 \pm 0 . 7$ </td><td> $7 7 . 2 \pm 0 . 3$ </td><td> $7 2 . 2 \pm 0 . 6$ </td><td> $7 9 . 4 \pm 0 . 5$ </td></tr><tr><td>DANN [37]</td><td>×</td><td>X</td><td> $8 8 . 7 \pm 0 . 6 $ </td><td> $8 1 . 7 \pm 0 . 6$ </td><td> $7 9 . 9 \pm 0 . 7$ </td><td> $8 0 . 2 \pm 0 . 4$ </td><td> $7 8 . 3 \pm 0 . 5$ </td><td> $8 1 . 8 \pm 0 . 6$ </td></tr><tr><td>MDAN [39]</td><td>7</td><td>X</td><td> $8 4 . 9 \pm 0 . 7$ </td><td> $7 5 . 1 \pm 0 . 6$ </td><td> $7 9 . 8 \pm 0 . 9$ </td><td> $8 3 . 7 \pm 0 . 5$ </td><td> $7 7 . 1 \pm 0 . 7$ </td><td> $8 0 . 1 \pm 0 . 7$ </td></tr><tr><td>DECISION [20]</td><td>vv</td><td>√</td><td> $7 6 . 3 \pm 1 . 5$ </td><td> $5 8 . 7 \pm 0 . 7$ </td><td> $7 7 . 8 \pm 1 . 1$ </td><td> $7 9 . 1 \pm 0 . 6$ </td><td> $5 6 . 2 \pm 0 . 8$ </td><td> $6 9 . 6 \pm 0 . 9$ </td></tr><tr><td>EAAF [22]</td><td></td><td>√</td><td> $8 0 . 8 \pm 0 . 7$ </td><td> $7 2 . 5 \pm 0 . 6$ </td><td> $7 8 . 5 \pm 0 . 5$ </td><td> $8 2 . 4 \pm 0 . 7$ </td><td> $7 9 . 7 \pm 0 . 4$ </td><td> $7 8 . 8 \pm 0 . 6$ </td></tr><tr><td>Bi-ATEN [21]</td><td>vv</td><td>√</td><td> $8 1 . 2 \pm 1 . 2$ </td><td> $8 6 . 5 \pm 0 . 8$ </td><td> $8 3 . 5 \pm 0 . 7$ </td><td> $8 6 . 5 \pm 0 . 5$ </td><td> $8 5 . 7 \pm 1 . 0$ </td><td> $8 4 . 3 \pm 0 . 4$ </td></tr><tr><td>CAT-CKD (Ours)</td><td></td><td>√</td><td> $9 0 . 8 \pm 0 . 2$ </td><td> $9 0 . 5 \pm 0 . 5$ </td><td> $7 5 . 8 \pm 0 . 6$ </td><td> $8 9 . 7 \pm 0 . 1$ </td><td> $8 4 . 0 \pm 0 . 5$ </td><td> ${ \bf 8 6 . 1 \pm 0 . 4 }$ </td></tr></table>

To further illustrate the efectiveness of CAT-CKD, Figure 4 presents t-SNE visualizations of the target-domain feature embeddings before and after adaptation. Following adaptation, the features exhibit markedly improved inter-class separability and intra-class compactness. The resulting clusters are more coherent and better aligned with class boundaries, suggesting that CAT-CKD successfully learns a discriminative and domain-consistent feature representation that enhances generalization on the target domain.

![](images/271755e56e67c96a73dfebeb51044d2fb1929ec4073f31631d82b2b84e1857e4.jpg)  
Figure 4: t-SNE visualization of target-domain features before and after adaptation with CAT-CKD.

Eficiency Analysis. Table 3 presents a comparison of trainable parameters between our proposed model and baselines. Existing methods, including EAAF [22] and DECISION [20], involve training over 96M parameters, while others like DANN [37] and ADDA [38] still require tens of millions of parameters to be updated. In contrast, our parameter-eficient design significantly reduces the number of trainable parameters by integrating lightweight adapters into a frozen ViT backbone. The full ViT model contains over 90 million parameters, but during the backbone training phase using CAT, only the adapter modules are updated—resulting in just 0.3 million trainable parameters. In the subsequent source model training and target adaptation stages, only the bottleneck and classification head are trained, with the total number of trainable parameters after CAT approximately 4.3 million. This represents a substantial reduction compared to fully fine-tuned baselines. The reduction in trainable parameters not only improves computational and memory eficiency, but also helps mitigate overfitting.

Table 3: Comparison of the trainable parameters between our proposed model and baseline models is presented.
<table><tr><td>Model Configuration</td><td># Parameters</td></tr><tr><td>EAAF [22]</td><td>98.2M</td></tr><tr><td>DECISION [20]</td><td>96.1M</td></tr><tr><td>DANN [37]</td><td>53.0M</td></tr><tr><td>ADDA [38]</td><td>26.0M</td></tr><tr><td>MDAN [39]</td><td>11.1M</td></tr><tr><td>Bi-ATEN [21]</td><td>10.6M</td></tr><tr><td>CAT-CKD (Full model)</td><td>4.3M</td></tr></table>

Discussion on Results. The experimental results demonstrate the practical efectiveness of the proposed CAT-CKD framework in the context of MSFDA for skin cancer diagnosis. By adopting a parameter-eficient tuning strategy, the ViT backbone is adapted using lightweight adapters, enabling eficient optimization. This results in a robust and domain-invariant feature extractor capable of strong cross-domain generalization. Notably, the framework maintains competitive performance across diverse datasets without requiring access to source data during target adaptation, which is particularly important for privacy-sensitive medical applications. In addition, the significant reduction in trainable parameters improves computational eficiency and enhances suitability for deployment in resource-constrained clinical environments.

PAD-UFES-20 presents a uniquely challenging adaptation scenario, as confirmed by the pairwise Fréchet Distance matrix (Figure 2), where it exhibits the largest domain distances from all other datasets. The consistent performance drop across methods indicates that the dificulty arises from the severity of the domain gap rather than from CAT-CKD. Despite this, CAT-CKD achieves the highest average AUROC of 86.1% and outperforms all competing methods.

Nevertheless, adaptation to target domains with large domain gaps remains challenging, especially when imaging modalities or acquisition conditions difer substantially. Moreover, while the framework extends to multiclass or multi-label settings with minimal modifications, constructing crossdomain datasets with consistent label definitions and realistic domain shifts remains a practical challenge.

## 4.5. Ablation Study

In this section, we conduct an ablation study to evaluate the efectiveness of each component of the CAT-CKD framework. As such, we remove these components individually to assess their contributions. We first remove CAT and use the backbone in a frozen manner without the CAT-based finetuning step. We then evaluated fine-tuning the entire backbone instead of applying the CAT-based parameter-eficient adaptation. This full fine-tuning strategy resulted in inferior performance, due to overfitting on limited data and catastrophic forgetting of pretrained representations. To eliminate CKD, we supervise the student model using logits from a randomly selected single source model instead of leveraging the ensemble. For removing KL divergence, we substitute it with a standard cross-entropy loss. As shown in Table 4, removing each component lead to notable performance degradation. These results confirm that all components in our proposed method are essential to the framework’s success, as removing either results in a substantial drop in AUROC, thereby validating the proposed design.

Table 4: Ablation study evaluating the impact of key components in our framework. We report AUROC scores across five target datasets and the average. Best results are shown in bold.
<table><tr><td>Model</td><td>ISIC 2018</td><td>ISIC 2020</td><td>PAD-UFES-20</td><td>Derm7pt-Derm</td><td>Derm7pt-Clinic</td><td>Avg.</td></tr><tr><td>Ours</td><td>90.8</td><td>90.5</td><td>75.8</td><td>89.7</td><td>84.0</td><td>86.1</td></tr><tr><td>Ours w/o CAT</td><td>87.5</td><td>90.1</td><td>72.3</td><td>78.6</td><td>78.0</td><td>81.3</td></tr><tr><td>Ours w/o CAT, w/ Full backbone finetuning</td><td>78.6</td><td>90.1</td><td>64.5</td><td>65.9</td><td>75.4</td><td>74.9</td></tr><tr><td>Ours w/o CKD</td><td>86.0</td><td>59.5</td><td>53.4</td><td>73.2</td><td>74.2</td><td>69.3</td></tr><tr><td>Ours w/o KL Divergence</td><td>85.7</td><td>61.2</td><td>53.3</td><td>72.1</td><td>73.5</td><td>69.2</td></tr></table>

## 4.6. Sensitivity Analysis

In this section, we perform a sensitivity analysis to evaluate how hyperparameter settings and architectural components influence our model’s performance.

## 4.6.1. Efect of CAT on Attention Maps

Figure 5 shows attention maps generated by the backbone, with and without CAT. Without CAT, attention is broadly dispersed across irrelevant background regions with little to no focus on the lesion area, indicating poor feature localization. Introducing CAT noticeably concentrates attention toward lesion regions, demonstrating that contrastive adapter training encourages the model to attend to more semantically relevant areas. While perfect lesion localization is not expected from ViT attention maps due to their global patch-relationship nature, the key observation is that CAT shifts the attention distribution toward more focused and semantically meaningful patterns, which indicates improved domain-invariant feature learning.

![](images/3b42afdba26f7e0d46c28888feb6d78f6a5cca447fe8d60b26ccb0503f34d0a8.jpg)  
Figure 5: Comparison of attention maps generated by the ViT backbone with and without CAT.

## 4.6.2. Soft-label vs. Hard-label Knowledge Distillation

To evaluate the impact of supervision type during distillation, we compare the proposed soft-label CKD with a hard-label variant in which the consensus prediction is converted into one-hot pseudo-labels and the student is trained using cross-entropy loss. As shown in Table 5, soft-label CKD achieves a higher average AUROC and delivers more robust performance on challenging targets such as PAD-UFES-20. This behavior arises because hard-label distillation propagates teacher errors directly when the consensus prediction is incorrect and enforces an overconfident, potentially noisy target. In contrast, soft-label distillation trains the student using the full predictive distribution, which preserves uncertainty and inter-class similarity information from multiple teachers.

Table 5: Comparison between soft-label and hard-label knowledge distillation.
<table><tr><td>Model</td><td>ISIC 2018</td><td>ISIC 2020</td><td>PAD-UFES-20</td><td>Derm7pt-Derm</td><td>Derm7pt-Clinic</td><td>Avg.</td></tr><tr><td>Hard-label CKD</td><td>87.4</td><td>86.9</td><td>71.3</td><td>87.7</td><td>83.7</td><td>83.4</td></tr><tr><td>Soft-label CKD (Ours)</td><td>90.8</td><td>90.5</td><td>75.8</td><td>89.7</td><td>84.0</td><td>86.1</td></tr></table>

## 4.6.3. Convpass vs. Other Adapter Models

To assess the efect of adapter architecture on the proposed method, we compare the ConvPass adapter with two widely used alternatives. The Houlsby adapter [40] inserts bottleneck modules after both the attention and feed-forward sublayers within each transformer block. In contrast, the Pfeifer adapter [57] adopts a more lightweight design by placing a single adapter after the feed-forward layer only. Both variants employ a linear down-projection, non-linearity, and up-projection structure, but difer in their placement and overall expressiveness (see Table 6).

Table 6: Comparison between diferent adapter models.
<table><tr><td>Model</td><td>ISIC 2018</td><td>ISIC 2020</td><td>PAD-UFES-20</td><td>Derm7pt-Derm</td><td>Derm7pt-Clinic</td><td>Avg.</td></tr><tr><td>CAT(w/ Houlsby [40])-CKD</td><td>90.7</td><td>73.7</td><td>68.0</td><td>91.2</td><td>84.6</td><td>81.6</td></tr><tr><td>CAT(w/ Pfeiffer [57])-CKD</td><td>89.7</td><td>77.9</td><td>74.1</td><td>91.7</td><td>81.0</td><td>82.8</td></tr><tr><td>CAT(w/ Convpass [28])-CKD (Ours)</td><td>90.8</td><td>90.5</td><td>75.8</td><td>89.7</td><td>84.0</td><td>86.1</td></tr></table>

## 4.6.4. Efect of Diferent Consensus Label Generation Strategies

We evaluate diferent strategies for generating consensus pseudo-labels from multiple teacher models during distillation. Under domain shift, variations in teacher confidence and reliability make the aggregation strategy a critical component of the distillation process. As shown in Table 7, in addition to uniform averaging (used in our method), we consider three confidenceaware alternatives: confidence-weighted averaging [58] (based on maximum softmax score), margin-based weighting [59] (using the top-1 vs. top-2 probability gap), and entropy-weighted averaging [60] (favoring low-entropy predictions). While these strategies aim to prioritize confident teachers, they rely on potentially miscalibrated confidence estimates. This can amplify errors from overconfident but incorrect teachers. Uniform averaging, by contrast, avoids such bias and provides a stable, balanced supervision signal across domains.

Table 7: Comparison between diferent consensus label generation strategies.
<table><tr><td>Model</td><td>ISIC 2018</td><td>ISIC 2020</td><td>PAD-UFES-20</td><td>Derm7pt-Derm</td><td>Derm7pt-Clinic</td><td>Avg.</td></tr><tr><td>Confidence-weighted averaging [58]</td><td>91.2</td><td>75.7</td><td>70.8</td><td>90.7</td><td>86.2</td><td>82.9</td></tr><tr><td>Margin-based weighting [59]</td><td>90.8</td><td>75.0</td><td>73.1</td><td>90.7</td><td>85.3</td><td>83.0</td></tr><tr><td>Entropy-weighted averaging [60]</td><td>89.1</td><td>84.4</td><td>72.2</td><td>89.6</td><td>84.1</td><td>83.9</td></tr><tr><td>CKD (Ours)</td><td>90.8</td><td>90.5</td><td>75.8</td><td>89.7</td><td>84.0</td><td>86.1</td></tr></table>

## 4.6.5. Efect of τ in Supervised Contrastive Learning

The temperature parameter τ in SCL modulates the concentration of the similarity distribution and thus determines the strength of separation between positive and negative pairs. Lower values yield sharper distributions that emphasize hard negatives, whereas higher values produce smoother distributions with reduced contrastive force. We tested three common values: 0.07, 0.2, and 0.5. As shown in Figure 6, $\tau = 0 . 2$ delivers the best performance by generating the most discriminative representations. In contrast, $\tau = 0 . 0 7$ and 0.5 both diminish class separation, underscoring the importance of tuning τ for efective domain-invariant feature learning.

![](images/a707afc6ba2577f5364f5fb25733da15e017b94fe248d87ef364c415b84a749c.jpg)  
Figure 6: AUROC performance comparison for diferent temperature values.

## 5. Conclusion

In this work, we proposed CAT-CKD, a novel MSFDA framework for skin cancer diagnosis that preserves patient privacy by eliminating the need for source-domain images during adaptation. First, CAT instills domaininvariant, class-discriminative representations into a ViT backbone through SCL and lightweight adapters to reduce overfitting while retaining pretrained knowledge. Next, CKD transfers ensemble knowledge from frozen source models to a student network on unlabeled target data via a KL-divergence objective. Extensive experiments on five public dermoscopic datasets demonstrate that our framework consistently outperforms both traditional UDA and state-of-the-art multi-source-free methods. Moreover, the adapter-based design slashes the number of trainable parameters and cuts computational overhead, enabling practical deployment in resource-constrained clinical settings.

Future work will focus on developing a more eficient source-domain training phase to reduce complexity. We will also investigate the robustness of the framework under diferent attack settings and privacy constraints. In addition, extending the approach to federated multi-source domain adaptation will be explored to further enhance its applicability in distributed and privacy-sensitive environments.

## Acknowledgment

This research is partially funded by the NSERC CREATE TrustCAV program and the NSERC Discovery Grant program, and in part by the NSERC Canada Research Chair (CRC) Program.

## References

[1] A. K. Tiwari, M. K. Mishra, A. R. Panda, B. Panda, Survey on computer-aided automated melanoma detection, Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization 11 (7) (2024) 2300257.

[2] M. Kreouzi, N. Theodorakis, G. Feretzakis, E. Paxinou, A. Sakagianni, D. Kalles, A. Anastasiou, V. S. Verykios, M. Nikolaou, Deep Learning for Melanoma Detection: A Deep Learning Approach to Diferentiating Malignant Melanoma from Benign Melanocytic Nevi, Cancers 17 (1) (2024) 28.

[3] N. Melarkode, K. Srinivasan, S. M. Qaisar, P. Plawiak, AI-Powered Diagnosis of Skin Cancer: A Contemporary Review, Open Challenges and Future Research Directions, Cancers 15 (4) (2023) 1183.

[4] C. Fink, A. Blum, T. Buhl, C. Mitteldorf, R. Hofmann-Wellenhof, T. Deinlein, W. Stolz, L. Trennheuser, C. Cussigh, D. Deltgen, J. Winkler, F. Toberer, A. Enk, A. Rosenberger, H. Haenssle, Diagnostic performance of a deep learning convolutional neural network in the diferentiation of combined naevi and melanomas, Journal of the European Academy of Dermatology and Venereology 34 (6) (2020) 1355–1361.

[5] M. Jackson, H. Kalirai, R. N. Hussain, H. Heimann, Y. Zheng, S. E. Coupland, Diferentiating Choroidal Melanomas and Nevi Using a Self-Supervised Deep Learning Model Applied to Clinical Fundoscopy Images, Ophthalmology Science 5 (2) (2025) 100647.

[6] K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, in: IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[7] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, H. Adam, MobileNets: Eficient Convolu tional Neural Networks for Mobile Vision Applications (Apr. 2017).

[8] G. Huang, Z. Liu, L. v. d. Maaten, K. Q. Weinberger, Densely Connected Convolutional Networks (Jan. 2018).

[9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, in: International Conference on Learning Representations, 2021.

[10] Y. Zhang, Y. Wei, Q. Wu, P. Zhao, S. Niu, J. Huang, M. Tan, Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis, IEEE Transactions on Image Processing 29 (2020) 7834–7844.

[11] M. Wang, W. Deng, Deep visual domain adaptation: A survey, Neurocomputing 312 (2018) 135–153.

[12] I. Redko, E. Morvant, A. Habrard, M. Sebban, Y. Bennani, Advances in domain adaptation theory, Computer engineering, ISTE Press Ltd ; Elsevier Ltd, London, UK : Kidlington, Oxford, UK, 2019.

[13] Z. Han, H. Sun, Y. Yin, Learning Transferable Parameters for Unsupervised Domain Adaptation, IEEE Transactions on Image Processing 31 (2022) 6424–6439.

[14] Z. Cai, Y. Huang, T. Zhang, Y. Zheng, D. Yue, Multi-Source Domain Adaptation by Causal-Guided Adaptive Multimodal Difusion Networks, International Journal of Computer Vision 133 (7) (2025) 4623–4645.

[15] B. Jiang, Y. Kuang, H. H. Song, Privacy-Preserving Services for Internet of Medical Things: Architecture, Techniques, and Challenges, IEEE Transactions on Services Computing 19 (2) (2026) 1688–1707.

[16] Y. Zhang, D. Cai, Z. Khan, P. Chen, A. Ammar, Cluster-Fair-FL for Multi-Label Consumer Health Privacy-Preserving Image Analysis, IEEE Transactions on Consumer Electronics (2026).

[17] J. Dong, Z. Fang, A. Liu, G. Sun, T. Liu, Confident Anchor-Induced Multi-Source Free Domain Adaptation, in: Advances in Neural Information Processing Systems, Vol. 34, Curran Associates, Inc., 2021, pp. 2848–2860.

[18] Z. Cai, J. Liu, Y. Huang, C. Hu, X. Wu, X.-Y. Jing, L. Shao, Make source-free transfer great: Dynamic uncertainty-driven confident fusion for multi-source-free domain adaptation, Information Fusion 127 (2026) 103859.

[19] Y. Zhang, J. Jiao, H. Li, T. Zhang, Z. Zhao, H. Wang, More will be better: Multi-source-free aggregation adaptation with confidence calibration for fault diagnosis, Advanced Engineering Informatics 68 (2025) 103770.

[20] S. M. Ahmed, D. S. Raychaudhuri, S. Paul, S. Oymak, A. K. Roy-Chowdhury, Unsupervised Multi-Source Domain Adaptation Without Access to Source Data, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10103–10112.

[21] X. Li, J. Li, F. Li, L. Zhu, K. Lu, Agile Multi-Source-Free Domain Adaptation, Proceedings of the AAAI Conference on Artificial Intelligence 38 (12) (2024) 13673–13681, number: 12.

[22] J. Pei, A. Men, Y. Liu, X. Zhuang, Q. Chen, Evidential Multi-Source-Free Unsupervised Domain Adaptation, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (8) (2024) 5288–5305.

[23] G. Hinton, O. Vinyals, J. Dean, Distilling the Knowledge in a Neural Network, arXiv:1503.02531 [cs, stat] (Mar. 2015).

[24] L. T. Nguyen-Meidine, E. Granger, M. Kiran, J. Dolz, L.-A. Blais-Morin, Joint Progressive Knowledge Distillation and Unsupervised Domain Adaptation, in: 2020 International Joint Conference on Neural Networks (IJCNN), 2020, pp. 1–8.

[25] L. T. Nguyen-Meidine, A. Belal, M. Kiran, J. Dolz, L.-A. Blais-Morin, E. Granger, Knowledge distillation methods for eficient unsupervised adaptation across multiple domains, Image and Vision Computing 108 (2021) 104096.

[26] Z. Han, C. Gao, J. Liu, J. Zhang, S. Q. Zhang, Parameter-Eficient Fine-Tuning for Large Models: A Comprehensive Survey, arXiv:2403.14608 [cs] (Sep. 2024).

[27] T. Yang, Y. Zhu, Y. Xie, A. Zhang, C. Chen, M. Li, AIM: Adapting Image Models for Eficient Video Understanding, in: International Con ference on Learning Representations, 2023.

[28] S. Jie, Z.-H. Deng, S. Chen, Z. Jin, Convolutional bypasses are better vision transformer adapters, in: ECAI 2024, IOS Press, 2024, pp. 202– 209.

[29] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-Rank Adaptation of Large Language Models (Jun. 2021).

[30] Z. Yuan, J. Zhang, S. Shan, FullLoRA-AT: Eficiently Boosting the Robustness of Pretrained Vision Transformers, arXiv:2401.01752 [cs] (Jan. 2024).

[31] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, D. Krishnan, Supervised Contrastive Learning, arXiv:2004.11362 [cs] (Mar. 2021).

[32] J. Maurício, I. Domingues, J. Bernardino, Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review, Applied Sciences 13 (9) (2023) 5521.

[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is All you Need, in: Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017.

[34] S. Aksoy, P. Demircioglu, I. Bogrekci, Enhancing Melanoma Diagnosis with Advanced Deep Learning Models Focusing on Vision Transformer, Swin Transformer, and ConvNeXt, Dermatopathology 11 (3) (2024) 239–252.

[35] R. Wang, X. Chen, X. Wang, H. Wang, C. Qian, L. Yao, K. Zhang, A novel approach for melanoma detection utilizing GAN synthesis and vision transformer, Computers in Biology and Medicine 176 (2024) 108572.

[36] W. Wu, S. Mehta, S. Nofallah, S. Knezevich, C. J. May, O. H. Chang, J. G. Elmore, L. G. Shapiro, Scale-Aware Transformers for Diagnosing Melanocytic Lesions, IEEE Access 9 (2021) 163526–163541.

[37] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, V. Lempitsky, Domain-Adversarial Training of Neural Networks, in: G. Csurka (Ed.), Domain Adaptation in Computer Vision Applications, Springer International Publishing, Cham, 2017, pp. 189–209.

[38] E. Tzeng, J. Hofman, K. Saenko, T. Darrell, Adversarial Discriminative Domain Adaptation, 2017, pp. 7167–7176.

[39] H. Zhao, S. Zhang, G. Wu, J. M. F. Moura, J. P. Costeira, G. J. Gordon, Adversarial Multiple Source Domain Adaptation, in: S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 31, Curran Associates, Inc., 2018.

[40] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. D. Laroussilhe, A. Gesmundo, M. Attariyan, S. Gelly, Parameter-Eficient Transfer Learning for NLP, in: Proceedings of the 36th International Conference on Machine Learning, PMLR, 2019, pp. 2790–2799.

[41] X. L. Li, P. Liang, Prefix-Tuning: Optimizing Continuous Prompts for Generation, in: C. Zong, F. Xia, W. Li, R. Navigli (Eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Association for Computational Linguistics, Online, 2021, pp. 4582–4597.

[42] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, S.-N. Lim, Visual Prompt Tuning, in: S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, T. Hassner (Eds.), Computer Vision – ECCV 2022, Springer Nature Switzerland, Cham, 2022, pp. 709–727.

[43] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, P. Luo, Adapt-Former: Adapting Vision Transformers for Scalable Visual Recognition, Advances in Neural Information Processing Systems 35 (2022) 16664– 16678.

[44] Y. Zhang, K. Zhou, Z. Liu, Neural Prompt Search, IEEE Transactions on Pattern Analysis and Machine Intelligence (2024) 1–14Conference Name: IEEE Transactions on Pattern Analysis and Machine Intelligence.

[45] J. Wu, Z. Wang, M. Hong, W. Ji, H. Fu, Y. Xu, M. Xu, Y. Jin, Medical SAM adapter: Adapting segment anything model for medical image segmentation, Medical Image Analysis 102 (2025) 103547.

[46] Q. Guo, X. Wang, Y. Wu, Z. Yu, D. Liang, X. Hu, P. Luo, Online Knowledge Distillation via Collaborative Learning, 2020, pp. 11020– 11029.

[47] S. Zagoruyko, N. Komodakis, Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, arXiv:1612.03928 [cs] (Feb. 2017).

[48] S. Du, S. You, X. Li, J. Wu, F. Wang, C. Qian, C. Zhang, Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space, in: Advances in Neural Information Processing Systems, Vol. 33, Curran Associates, Inc., 2020, pp. 12345–12355.

[49] H. Zhang, D. Chen, C. Wang, Adaptive Multi-Teacher Knowledge Distillation with Meta-Learning, in: 2023 IEEE International Conference on Multimedia and Expo (ICME), 2023, pp. 1943–1948.

[50] S. Park, N. Kwak, FEED: Feature-level Ensemble for Knowledge Distillation, arXiv:1909.10754 [cs] (Sep. 2019).

[51] N. C. F. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, A. Halpern, Skin lesion analysis toward melanoma detection, in: 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), 2018, pp. 168–172.

[52] P. Tschandl, C. Rosendahl, H. Kittler, The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions, Scientific Data 5 (2018) 180161.

[53] V. Rotemberg, N. Kurtansky, B. Betz-Stablein, L. Cafery, E. Chousakos, N. Codella, M. Combalia, S. Dusza, P. Guitera, D. Gutman, A. Halpern, B. Helba, H. Kittler, K. Kose, S. Langer, K. Lioprys, J. Malvehy, S. Musthaq, J. Nanda, O. Reiter, G. Shih, A. Stratigos, P. Tschandl, J. Weber, H. P. Soyer, A patient-centric dataset of images and metadata for identifying melanomas using clinical context, Scientific Data 8 (1) (2021) 34.

[54] A. G. C. Pacheco, G. R. Lima, A. S. Salomão, B. Krohling, I. P. Biral, G. G. de Angelo, F. C. R. Alves Jr, J. G. M. Esgario, A. C. Simora, P. B. C. Castro, F. B. Rodrigues, P. H. L. Frasson, R. A. Krohling, H. Knidel, M. C. S. Santos, R. B. do Espírito Santo, T. L. S. G. Macedo, T. R. P. Canuto, L. F. S. de Barros, PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones, Data in Brief 32 (2020) 106221.

[55] J. Kawahara, S. Daneshvar, G. Argenziano, G. Hamarneh, Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets, IEEE Journal of Biomedical and Health Informatics 23 (2) (2019) 538–546, conference Name: IEEE Journal of Biomedical and Health Informatics.

[56] J. Wang, Y. Zhang, Z. Ding, J. Hamm, Achieving Reliable and Fair Skin Lesion Diagnosis via Unsupervised Domain Adaptation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2024, pp. 5157–5166.

[57] J. Pfeifer, A. Kamath, A. Rücklé, K. Cho, I. Gurevych, AdapterFusion: Non-Destructive Task Composition for Transfer Learning, in: P. Merlo, J. Tiedemann, R. Tsarfaty (Eds.), Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, Association for Computational Linguistics, Online, 2021, pp. 487–503.

[58] Y. Chen, X. Zhu, S. Gong, Semi-Supervised Deep Learning with Memory, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[59] T. Sosea, C. Caragea, MarginMatch: Improving semi-supervised learning with pseudo-margins, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 15773–15782.

[60] M. Long, Z. CAO, J. Wang, M. I. Jordan, Conditional Adversarial Domain Adaptation, in: Advances in Neural Information Processing Systems, Vol. 31, 2018.