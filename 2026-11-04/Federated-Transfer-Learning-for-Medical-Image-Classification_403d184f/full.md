# Federated Transfer Learning for Medical Image Classification under Mixed Skew

Presented at 4th IEEE International Conference on Federated Learning Technologies and Applications (FLTA 2026)

Alexandre G. Leclercq<sup>∗†‡</sup>, Sebastien Bougleux´ <sup>∗</sup>, Aurelien Corroyer-Dulmont´ <sup>†‡</sup>, Romain Herault´ <sup>∗</sup>

<sup>∗</sup>Universite Caen Normandie, ENSICAEN, CNRS, Normandie Univ, GREYC UMR 6072 ´ , F-14000 Caen, France

<sup>†</sup>Universite de Caen Normandie, CNRS, Normandie Univ, ISTCT UMR6030, CYCERON ´ , F-14000 Caen, France

<sup>‡</sup>Artificial Intelligence Department, Centre Franc¸ois Baclesse, F-14000 Caen, France

Email: {alexandre.leclercq, sebastien.bougleux, romain.herault}@unicaen.fr

a.corroyer-dulmont@baclesse.unicancer.fr

Abstract—Deep learning in medical imaging is constrained by data scarcity and by privacy regulations that make centralizing patient data across institutions impractical. Federated Learning (FL) addresses these constraints by keeping data local, but its distributed nature exposes training to strong statistical heterogeneity. Multicentric medical data is particularly affected, exhibiting a mixed skew that combines feature, label, concept, and quantity shifts across centers. Personalized FL mitigates heterogeneity by tailoring a model to each client, yet most methods assume training from scratch, which is unrealistic given the high dimensionality of medical features relative to the limited data available. We propose the Personalized Aligner Federated Pretrained (PAFP) framework, a personalized FL approach built on a pre-trained backbone. PAFP inserts client-specific Feature Alignment modules before the fine-tuned blocks to realign hidden feature distributions toward a server-aggregated target. On the FedISIC benchmark, PAFP attains the highest unweighted balanced accuracy while reducing inter-client performance variance roughly six-fold relative to the strongest baseline, yielding accuracy and cross-client equity jointly. Our code is publicly available at https://github.com/Alexandre-Leclercq/PAFP.

Index Terms—Medical Imaging Classification, Cross-Silo, Mixed-skew, pFL

## I. INTRODUCTION

Machine learning and deep learning models are attracting increasing interest in the medical imaging field to address various tasks such as tumor segmentation, disease prediction, anomaly detection, and modality to modality translation [1]– [3]. However, one of the biggest problems that the community is facing when it comes to training a medical imaging deep learning model is the lack of data. This data scarcity is due to several factors [2], [3], such as the need for specialized expertise to annotate training data, the necessity of patient consent, and strict privacy regulations and legal barriers (GDPR in Europe, HIPAA in the USA, etc.) that protect patient data but make the process of sharing across institutions complicated, if not impossible.

Federated Learning (FL) is a machine learning paradigm that addresses these privacy regulations and legal barriers [4], [5]. Instead of aggregating data from multiple clients (e.g., medical centers), as a traditional centralized learning approach would, FL keeps the data localized at every client and transmits the models to a server, whose goal is to gather and aggregate each client’s model. In particular, FL in medical settings is typically framed as a cross-silo scenario [6]. This implies a small number of clients, each holding a large dataset and sufficient computational power, and acting as a trustworthy party. This stands in contrast to cross-device FL, which assumes the opposite. However, the distributed nature of FL introduces new challenges, particularly the issue of data heterogeneity [5], [7], [8], where the data distribution is highly variable across clients (i.e., non-IID data). This problem creates diverging optimization directions during local training, which leads to a consensus issue commonly known as client drift. Medical imaging is particularly exposed to such phenomena, with feature skew, label skew, concept skew, and quantity skew naturally occurring across institutional datasets. The objective of traditional FL is to learn a single global model that generalizes well across all client datasets. Such an objec tive can be very hard to achieve under severe heterogeneity. To overcome this limitation, Personalized Federated Learning (pFL) shifts the paradigm [9]. In contrast to traditional FL, pFL does not attempt to learn a one-size-fits-all global model; instead, it aims to adapt and tailor models to the specific data distribution of each individual client. In this context, the global model essentially serves as a shared baseline to facilitate knowledge transfer toward the local clients. However, most of the approaches proposed in the literature assume training from scratch. These methods are not realistically suitable for medical imaging applications, given that the feature space is highly dimensional while the amount of available training data is severely limited. Thus, it is necessary to leverage pretrained models [10]–[13] in this context. In this work, we focus on the medical image classification task and validate our approach on the FedISIC2019 dataset from the FLamby benchmark [14], a real-world, naturally partitioned federated skin-lesion classification problem that exhibits strong interclient heterogeneity. Our contributions are summarized as follows:

• We formalize the pFL problem in the context of medical imaging, a formulation whose scope extends beyond this domain

• We analyze the data shift across clients.

• We propose the Personalized Aligner Federated Pretrained (PAFP) framework, a pFL approach based on a pre-trained model, using injected alignment modules before the finetuned block, which is client specific.

## II. RELATED WORKS

Traditional FL algorithms, such as FedAvg [4], optimize a single global model under the implicit assumption of independent and identically distributed (IID) data across clients. Under realistic non-IID settings, statistical heterogeneity causes the global model to converge toward a sub-optimal compromise that may perform poorly on individual client distributions [7]. To mitigate this client drift, methods like FedProx [15], SCAFFOLD [16], and MOON [17] introduce proximal terms, control variates, or contrastive losses, respectively. However, these optimization-based strategies still fundamentally aim to produce a single shared model, limiting their efficacy on highly skewed local data.

Personalized Federated Learning (pFL) shifts the objective to a family of models tailored to local distributions. Local adaptation refines a global model either through simple fine-tuning or meta-learning initializations (e.g., Per-FedAvg [18]) for rapid convergence. Regularization-based methods like Ditto [19] jointly optimize global and local models to balance consensus and personalization. Architecture-based methods introduce structural decoupling: FedBN [20], FedPer [21], and FedRep [22] retain specific local layers while sharing the feature extractor. Feature-decoupling methods (DBE [23], FDSE [24]) extract client-invariant global features alongside localized representations. Finally, to specifically handle label skew, methods like FedLC [25] calibrate logits based on class probabilities to prevent overfitting to minority classes.

Mixed Skew and Decoupled Personalization. In practical medical scenarios, clients simultaneously face domain shift (sensor variability / feature skew) and class imbalance (pathology prevalence / label skew). This mixed skew defeats traditional pFL methods designed for a single skew type, prompting the recent shift toward decoupling visual and statistical variability treatments. For instance, FED-ROD [26] introduces a two-predictor framework on top of a shared, actively trained feature extractor. It optimizes a generic head using a classbalanced loss while learning a personalized add-on head via empirical risk minimization. Similarly, Fed- ${ \bf - C O } _ { 2 }$ [27] relies on mutual knowledge distillation between a global online model (using personalized BN layers) and a client-specific offline model. However, because these architectures require training the entire backbone from scratch or maintaining computeintensive dual networks, they remain heavy and prone to overfitting or destabilization under severe data scarcity.

Federated Transfer Learning (FTL) & Parameter-Efficient Fine-Tuning (FedPEFT). To address data scarcity in domains like medical imaging, FTL [12] leverages pre-trained models to initialize the federated network. However, naive fine-tuning across clients under severe non-IID conditions often causes catastrophic forgetting of the pre-trained prior or fails to properly personalize features [13]. To achieve stable personalization without the overhead of full-network training, recent paradigms shift toward FedPEFT. Pioneered in NLP [28], [29], FedPEFT inserts lightweight adapters into a strictly frozen backbone. For example, AdaFL [30] freezes over 99% of parameters to align biased representations with negligible overhead. This paradigm has transitioned to Computer Vision [31], [32] and medical imaging (e.g., FedFMS [33], which federates the Segment Anything Model). Beyond simple insertion, recent works like PERADA [34] regularize personalized adapters with a distilled global adapter, effectively balancing local personalization with global generalization capabilities. To address memory constraints on heterogeneous edge devices, frameworks like Fed-HeLLo [35] propose Heterogeneous LoRA Allocation (HLA) to dynamically distribute trainable layers. Finally, for black-box foundation models, ZOOPFL [36] employs Zeroth-Order Optimization (ZOO) and input surgery to bypass internal backpropagation, while post-hoc methods like CCVR [37] debias the global classifier via virtual Gaussian representations to mitigate label skew.

Robustness in Federated PEFT. While standard FedPEFT methods focus on efficiency, they remain highly susceptible to noisy data and aggregation inconsistencies. Specifically, Low-Rank Adaptation (LoRA) components can inadvertently amplify bias in the presence of label noise [32]. To combat this, RFedLR [38] proposes Sensitivity-aware Robust Tuning (SRT), which selectively updates noise-sensitive parameters based on gradient discrepancies between clean and noisy proxies. Furthermore, to address the sub-optimality of naive LoRA averaging, RFedLR introduces Adaptive Federated LoRA Aggregation (AFLA), which dynamically weights client updates based on parameter importance (measured via Fisher Information and noise sensitivity), rather than relying on simple parameter averaging.

Positioning our approach. While existing FedPEFT methods successfully reduce parameter overhead, they largely focus on generic personalization or global-local regularization (e.g., PERADA leverages knowledge distillation to reconcile globallocal objectives). Frameworks such as RFedLR have highlighted the vulnerability of standard adapter-based methods to label noise. However, these robust frameworks are primarily reactive, focusing on filtering or mitigating the impact of noisy inputs, rather than addressing the structural mixed skew inherent in medical imaging. In contrast, PAFP adopts a proactive stance against distribution shift by positioning internal alignment modules to standardize features before the fine-tuned blocks.

## III. PROBLEM FORMULATION

Let K be the number of clients, where each client k holds a private dataset $\mathcal { D } _ { k } : = \{ x _ { k , i } , y _ { k , i } \} _ { i = 1 } ^ { n _ { k } }$ of size $n _ { k }$ . The total number of samples is denoted by $\begin{array} { r } { \bar { N } : = \sum _ { k = 1 } ^ { K } n _ { k } } \end{array}$ . Multicentric medical imaging is inherently exposed to multi-level data heterogeneity. Let $\mathbb { P } _ { k } ( \cdot , \cdot )$ and $\mathbb { P } _ { k } ( \cdot | \cdot )$ denote the joint and conditional data distributions for client k, respectively. We consider the following forms of statistical skew, which naturally arise in medical datasets:

![](images/2c41b38664555e5bb35eb3ac318ff8674a12649b27911f9f4a75eb001bd29e4d.jpg)  
Fig. 1. Overview of the Personalized Aligner Federated Pretrained (PAFP) framework

• Feature Skew (covariate shift): $\mathbb { P } _ { i } ( X ) ~ \neq ~ \mathbb { P } _ { j } ( X )$ while $\mathbb { P } _ { i } ( Y | X ) = \mathbb { P } _ { j } ( Y | X )$ , for all $i \neq j$ . Variations in acquisition devices, practitioners, and clinical protocols result in inconsistent feature space distributions across medical centers.

• Label Skew (prior probability shift): $\mathbb { P } _ { i } ( Y ) \neq \mathbb { P } _ { j } ( Y )$ while $\mathbb { P } _ { i } ( X | Y ) = \mathbb { P } _ { j } ( X | Y )$ , for all $i \neq j .$ Geographical disparities and center-specific specializations lead to variations in label distributions across clients.

• Concept Drift: $\mathbb { P } _ { i } ( Y | X ) ~ \neq ~ \mathbb { P } _ { j } ( Y | X )$ while $\mathbb { P } _ { i } ( X ) ~ =$ $\mathbb { P } _ { j } ( X )$ , for all $i \neq j$ . Inter-observer variability and differing clinical guidelines cause the relationship between features and labels to vary across centers.

In addition to these distributional shifts, medical imaging datasets exhibit quantity skew $( n _ { i } \neq n _ { j }$ for clients $i \neq j )$ . Unlike the previous types, quantity skew is distribution-agnostic and can occur even when $\mathbb { P } _ { i } ( X , Y ) = \mathbb { P } _ { j } ( X , Y )$ : it simply reflects imbalances in patient volume or collection duration. In practice, it frequently co-occurs with distributional shifts, further compounding the overall mixed skew challenge.

To formally address these heterogeneities, we adopt a decoupled pFL paradigm. We formulate the optimization objective to jointly learn a set of shared global representations alongside client-specific parameters:

$$
\operatorname* { m i n } _ { w , \{ v _ { k } \} _ { k = 1 } ^ { K } } \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { N } F _ { k } ( w , v _ { k } ) ,\tag{1}
$$

$$
\mathrm { w i t h ~ } F _ { k } ( w , v _ { k } ) : = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { k } } \big [ \mathcal { L } ( g ( x ; w , v _ { k } ) , y ) \big ] ,\tag{2}
$$

where L denotes the loss function, and $g$ is a neural network with global weights w and local personalized weights $v _ { k }$ for each client k.

Furthermore, medical imaging datasets typically feature a limited number of samples relative to their high-dimensional feature spaces (e.g., high-resolution volumetric or multi-modal scans). Training a model from scratch in this regime is unrealistic, as it would require vastly more data than is available locally, or even globally. This sample-to-dimension deficit restricts the space of admissible global models w to those leveraging pre-trained representations. In the following section, we present a method designed to jointly address the mixed skews under the pFL objective and the pre-trained backbone constraint defined above.

## IV. METHOD

This section describes the proposed PAFP framework for the classification task. We consider a neural network $g : = h \circ f ,$ where f is a pre-trained backbone that maps from the input space X to a representation space Z, and h is a classifier head that maps from Z to the label space Y.

The full training pipeline is summarized by Fig. 1.

## A. Feature Alignment modules

Feature skew induces distributional divergence not only in the input space X, but also across the intermediate representation spaces $H ^ { ( l ) }$ inside the feature extractor. Specifically, for a pre-trained model where L blocks are fine-tuned, the hidden representation at each fine-tuned layer l, denoted $H ^ { ( l ) }$ , exhibits a divergence in its feature distribution. Left unaddressed, this divergence propagates through the fine-tuning process, causing model divergence and ultimately leading to a collapse in consensus between fine-tuned client layers.

To mitigate this, we introduce Feature Alignment (FA) modules, each inserted before a corresponding fine-tuned block. Each FA module consists of a Group Normalization layer, a position-wise affine transformation across channels implemented as $\textbf { a } 1 \times 1$ convolution, and a SiLU activation, operating jointly as a learned transformation of the local feature distribution.

The parameters of the FA modules are client-specific and belong to $v _ { k }$ . Their role is to realign each client’s local

Fig. 2. Personalized Aligner Federated Pretrained Algorithm.

representations toward a shared target distribution obtained by aggregating statistics across clients, so that every finetuned block receives a representation $H ^ { ( l ) }$ whose feature distribution is consistent across clients. This consistency limits the divergence of local updates and improves agreement on the shared weights w.

Our FA modules are inspired by FDSE [24], which similarly introduces learnable modules to address distributional shift across clients. Our approach differs in two key aspects. First, we update the global target statistics using an Exponential Moving Average across communication rounds, rather than a direct average, in order to stabilize convergence and reduce sensitivity to round-to-round fluctuations in client participation. Second, since the FA modules are injected directly into a pre-trained backbone, we initialize the 1 × 1 convolution using a Dirac delta (identity) initialization [39], setting its kernel to an identity mapping with zero bias, while the Group Normalization affine parameters are initialized to unit scale and zero shift. This ensures that, at the start of fine-tuning, the FA modules do not perturb the pre-trained representations, thereby avoiding the training instability that would otherwise arise from inserting randomly initialized layers into an alreadyconverged backbone.

a) Local distribution: To guide the FA modules toward a shared target distribution, we introduce an alignment loss that acts as a regularization between the local statistics at layer l and the estimated global distribution. Let $\mu _ { k } ^ { ( l ) } \ \in \dag \mathbf { H }$ $\mathbf { \nabla } \overline { { \mathbb { R } } } ^ { \bullet _ { C } ( l ) } \times H ^ { ( l ) } \times W ^ { ( l ) }$ and $\sigma _ { k } ^ { 2 ( l ) } \in \mathbb { R } ^ { C ^ { ( l ) } \times H ^ { ( l ) } \times W ^ { ( l ) } }$ denote the spatial mean and variance, computed over the batch dimension, of the output of the l-th FA module for client k. We estimate the local running statistics over a complete epoch using Exponential Moving Average (EMA) with momentum $\beta _ { 1 }$ , such that:

$$
\begin{array} { r l } & { \hat { \mu } _ { k } ^ { ( l ) }  ( 1 - \beta _ { 1 } ) \hat { \mu } _ { k } ^ { ( l ) } + \beta _ { 1 } \mu _ { k } ^ { ( l ) } , } \\ & { \hat { \sigma } _ { k } ^ { 2 ( l ) }  ( 1 - \beta _ { 1 } ) \hat { \sigma } _ { k } ^ { 2 ( l ) } + \beta _ { 1 } \sigma _ { k } ^ { 2 ( l ) } . } \end{array}\tag{3}
$$

b) Global target distribution: At the end of each communication round t, each client transmits its local statistics $\{ \hat { \mu } _ { k } ^ { ( l ) } , \hat { \sigma } _ { k } ^ { 2 ( l ) } \} _ { l = 1 } ^ { L }$ to the server alongside its locally fine-tuned global weights $( w _ { k } ^ { t } )$ . The server estimates the global target distribution by aggregating the local statistics received from all clients:

$$
\mu _ { g } ^ { ( l ) } : = \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { N } \hat { \mu } _ { k } ^ { ( l ) } , \quad \sigma _ { g } ^ { 2 ( l ) } : = \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { N } \hat { \sigma } _ { k } ^ { 2 ( l ) } .\tag{4}
$$

To ensure stability across communication rounds, the global target statistics are smoothed via EMA with momentum $\beta _ { 2 } \colon$

$$
\begin{array} { r l } & { \hat { \mu } _ { g } ^ { ( l ) }  ( 1 - \beta _ { 2 } ) \hat { \mu } _ { g } ^ { ( l ) } + \beta _ { 2 } \mu _ { g } ^ { ( l ) } , } \\ & { \hat { \sigma } _ { g } ^ { 2 ( l ) }  ( 1 - \beta _ { 2 } ) \hat { \sigma } _ { g } ^ { 2 ( l ) } + \beta _ { 2 } \sigma _ { g } ^ { 2 ( l ) } . } \end{array}\tag{5}
$$

c) Alignment loss: Each client computes the following alignment loss to guide its FA modules to transform local

Input: global model w, local epochs $E ,$ rounds T   
1: for each round $t = 0 , 1 \ldots , T$ do   
2: Server sends $\{ w ; ( \hat { \mu } _ { g } , \hat { \sigma } _ { g } ^ { 2 } ) \}$ to each client k   
3: for each client $k \in [ K ]$ in parallel do   
4: for each epochs do   
5: for each batch b do   
6: $\hat { \mu } _ { k , b } \gets ( 1 - \beta _ { 1 } ) \cdot \hat { \mu } _ { k , b - 1 } + \beta _ { 1 } \cdot \mu _ { k , b }$   
7: if alternate case then   
8: if local round then   
9: Compute loss $\mathcal { L } _ { k } \gets \mathcal { L } _ { \mathrm { t a s k } , k } + \lambda \mathcal { L } _ { \mathrm { a l i g n } , k }$   
10: $v _ { k } ^ { t } \gets \mathrm { S G D } ( \mathcal { L } _ { k } )$ for $E$ epochs   
11: $\{ ( \hat { \mu } _ { k } , \hat { \sigma } _ { k } ^ { 2 } ) \}$ send to server   
12: if global round then   
13: Compute loss $\mathcal { L } _ { k } \gets \mathcal { L } _ { \mathrm { t a s k } , k }$   
14: $w _ { k } ^ { t } \gets \mathrm { S G D } ( \mathcal { L } _ { k } )$ for E epochs   
15: $\{ \ddot { w } _ { k } ^ { t } , ( \hat { \mu } _ { k } , \hat { \sigma } _ { k } ^ { 2 } ) \}$ send to server   
16: if non-alternate case then   
17: Compute loss $\mathcal { L } _ { k } \gets \mathcal { L } _ { \mathrm { t a s k } , k } + \lambda \mathcal { L } _ { \mathrm { a l i g n } , k }$   
18: $\begin{array} { r } { w ^ { t + 1 }  \bar { \sum _ { k = 1 } ^ { K } { \frac { n _ { k } } { N } } w _ { k } ^ { t } } } \end{array}$   
19: $\begin{array} { r } { ( \mu _ { \mathrm { g , r o u n d } } , \sigma _ { \mathrm { g , r o u n d } } ^ { 2 } )  \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { N } ( \hat { \mu } _ { k } , \hat { \sigma } _ { k } ^ { 2 } ) } \end{array}$   
20: $( \hat { \mu } _ { \mathrm { g } } , \hat { \sigma } _ { \mathrm { g } } ^ { 2 } ) \stackrel { \sim } { = } ( 1 - \beta _ { 2 } ) \cdot ( \hat { \mu } _ { \mathrm { g } } , \hat { \sigma } _ { \mathrm { g } } ^ { 2 } ) + \beta _ { 2 } \cdot ( \mu _ { \mathrm { g , r o u n d } } , \sigma _ { \mathrm { g , r o u n d } } ^ { 2 } )$

feature distributions toward the estimated global target:

$$
\begin{array} { r } { \displaystyle \mathcal { L } _ { \mathrm { a l i g n } , k } : = \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { a l i g n } , k } ^ { ( l ) } , } \\ { \mathrm { w i t h } ~ \mathcal { L } _ { \mathrm { a l i g n } , k } ^ { ( l ) } : = \mathbf { M S E } \Big ( \hat { \mu } _ { k } ^ { ( l ) } , \hat { \mu } _ { g } ^ { ( l ) } \Big ) + \mathbf { M S E } \Big ( \hat { \sigma } _ { k } ^ { 2 ( l ) } , \hat { \sigma } _ { g } ^ { 2 ( l ) } \Big ) } \end{array}\tag{6}
$$

(7)

The total local objective for client k combines the task loss $\mathcal { L } _ { \mathrm { t a s k } , k }$ with the alignment loss as a regularization term:

$$
\mathcal L _ { k } ( w , v _ { k } ) : = \mathcal L _ { \mathrm { t a s k } , k } ( w , v _ { k } ) + \lambda \cdot \mathcal L _ { \mathrm { a l i g n } , k } ( v _ { k } ) ,\tag{8}
$$

where $\lambda > 0$ controls the regularization strength.

## B. Optimization Strategy

We introduce two variants of our method. Standard PAFP uses a single optimizer to update both the local parameters v<sub>k</sub> and shared parameters w based on the joint loss. Alternatively, PAFP-alt employs two optimizers: one updates w using only the task loss, and another updates $v _ { k }$ using both the task and alignment losses. This variant alternates between optimizing shared and local parameters, with a hyper-parameter r dictating that local updates occur every r rounds. The full training procedure, including client-side local updates and server-side aggregation, is detailed in Fig. 2.

## C. Logits correction

Label skew induces client-specific imbalances in the label distribution $\mathbb { P } _ { k } ( Y )$ , which biases the classifier head h toward the label distribution observed locally during fine-tuning. Left unaddressed, this bias causes the model’s output logits to systematically favor over-represented classes at each client, degrading performance on under-represented classes.

TABLE I  
FEDISIC CLIENT LABEL DISTRIBUTION
<table><tr><td rowspan="2">Client</td><td colspan="8">Class</td></tr><tr><td> $\overline { { C _ { 0 } } }$ </td><td> $\overline { { C _ { 1 } } }$ </td><td> $\overline { { C _ { 2 } } }$ </td><td> $\overline { { C _ { 3 } } }$ </td><td> $\overline { { C _ { 4 } } }$ </td><td> $\overline { { C _ { 5 } } }$ </td><td> $\overline { { C _ { 6 } } }$ </td><td> $\overline { { C _ { 7 } } }$ </td></tr><tr><td>1 (n=12413)</td><td>22.6%</td><td>34.0%</td><td>22.7%</td><td>6.1%</td><td>9.3%</td><td>1.0%</td><td>0.9%</td><td>3.3%</td></tr><tr><td>2 (n=3954)</td><td>0.6%</td><td>94.1%</td><td>0.1%</td><td>0.00%</td><td>3.0%</td><td>0.7%</td><td>1.4%</td><td>0.00%</td></tr><tr><td>3 (n=3359)</td><td>19.8%</td><td>54.6%</td><td>6.5%</td><td>0.7%</td><td>14.1%</td><td>1.5%</td><td>2.4%</td><td>0.4%</td></tr><tr><td>4 (n=2258)</td><td>15.3%</td><td>35.9%</td><td>12.3%</td><td>4.8%</td><td>22.5%</td><td>1.1%</td><td>0.2%</td><td>8.0%</td></tr><tr><td>5 (n=814)</td><td>25.3%</td><td>53.3%</td><td>0.0%</td><td>0.0%</td><td>21.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>6 (n=437)</td><td>16.5%</td><td>78.6%</td><td>1.4%</td><td>0.00%</td><td>1.7%</td><td>1.1%</td><td>0.6%</td><td>0.00%</td></tr></table>

To mitigate this, in the experiments, we tested a Logit Corrector (LC), a lightweight, client-specific adaptation layer inserted immediately after the classifier head $h .$ The Logit Corrector applies an elementwise affine transformation to the output logit vector, allowing each client to shift and scale its predictions to better match its local label distribution. Formally, let $\hat { y } _ { k } = h ( z ; w ^ { h } ) \in \mathbb R ^ { C }$ denote the raw logit vector produced by the classifier head for client $k ,$ where C is the number of classes. The corrected logit vector is given by:

$$
\hat { y } _ { k } ^ { \prime } : = a _ { k } \odot \hat { y } _ { k } + b _ { k }\tag{9}
$$

where $a _ { k } \in \mathbb { R } ^ { C }$ and $b _ { k } \in \mathbb { R } ^ { C }$ are client-specific scaling and shifting vectors, respectively, and ⊙ denotes the Hadamard product.

As with the FA modules, the parameters $\{ a _ { k } , b _ { k } \}$ of the Logit Corrector are client-specific and constitute part of $v _ { k }$

## V. EXPERIMENTAL SETUP

a) Dataset: We evaluate our method on FedISIC, a federated skin lesion classification dataset from the FLamby benchmark [14], derived from the ISIC 2019/2020 challenges. FedISIC partitions dermoscopic images across K clients according to their original acquisition center, providing a naturally occurring, non-simulated instance of multicentric data heterogeneity.

b) Baseline model and training: Following the FedISIC baseline, we use an EfficientNet backbone pre-trained on ImageNet as the feature extractor $f ,$ followed by a linear classification head h. As the task loss $\mathcal { L } _ { \mathrm { t a s k } , k } ,$ , we reuse the baseline focal loss [40] with class weights $\alpha _ { k } \in \mathbb { R } ^ { C }$ computed from the local label distribution of each client k, and a fixed focusing parameter $\gamma = 2 . 0$ . For the feature alignment modules, we set the regularization strength to $\lambda = 1 0$ , and the momentum parameters to $\beta _ { 1 } = 0 . 8$ for the local modules and $\beta _ { 2 } = 0 . 8$ for the server side.

All clients are trained using an SGD optimizer with a learning rate of $1 0 ^ { - 2 }$ and a weight decay of $1 0 ^ { - 5 }$ . Training is conducted for up to 1500 communication rounds—though most methods converge well before this limit—with each client performing 1 local epoch per round using a batch size of 64.

c) Characterizing dataset skew: We first characterize the heterogeneity present in FedISIC prior to training. Table I reports the per-client label distribution, expressed as the percentage of samples belonging to each class, confirming the presence of label skew across clients. Fig. 3 depicts the label distribution of each client and Fig. 4 highlight quantity skew.

![](images/08dc6ab584d708e8794bb93122b85f212eb82e7464ac0252f9f59e35631c3bc3.jpg)

Fig. 3. Per-client data distributions across the six clients.  
![](images/7977bfc9148110575b13fd3fcc57941b204f60882ad2623c806b7e11eccee908.jpg)  
Fig. 4. Client size distribution.

![](images/243df33fe25f2f57a74edbb026724ca1418068aa4142416b04a2ba2c5cc30703.jpg)  
Fig. 5. FedISIC interclient MMD

To quantify feature skew, we compute the pairwise Maximum Mean Discrepancy (MMD) between clients’ latent representations $Z ,$ extracted from the pre-trained backbone f prior to any fine-tuning. We use a Multiple RBF Kernel with σ ranging from a scaling of $[ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , \ 1 , 1 0 ^ { 1 } , 1 0 ^ { 2 } , 1 0 ^ { 3 } ]$ apply to a base gamma obtained using the median heuristic. Table 5 reports the resulting pairwise MMD matrix, quantifying the initial feature-space shift between clients before federated training begins. As commonly observed in the literature, empirical pairwise MMD estimates on latent representations may violate the triangle inequality due to finite-sample estimation noise, yet they reliably capture the relative statistical discrepancies across clients.

TABLE II  
BALANCED ACCURACY ON FEDISIC (MEAN±STD, BEST FEDERATED IN BOLD, SECOND BEST FEDERATED IN ITALIC)
<table><tr><td></td><td>client 1  $\scriptstyle ( \mathrm { N } = 1 2 4 1 3 )$ </td><td> $\overline { { \mathrm { c l i e n t } 2 } }$   $\scriptstyle ( \mathbf { N } = 3 9 5 4 )$ </td><td>client 3  $\scriptstyle ( \mathbf { N } = 3 3 5 9 )$ </td><td></td><td>client 4 (N=2258)</td><td>client 5 (N=814)</td><td>client 6 (N=437)</td><td> $\overline { { \mathrm { { w e i g h t ~ a v g } } } }$ </td><td></td><td>unweighted avg</td><td>inter-client variance</td></tr><tr><td>isolated</td><td> $\overline { { 8 0 . 1 6 \pm 1 . 0 1 } }$ </td><td> $7 5 . 4 5 \pm 5 . 3 0$ </td><td> $\overline { { 7 0 . 6 9 \pm 0 . 8 9 } }$ </td><td></td><td> $\overline { { 7 0 . 2 0 \pm 4 . 5 9 } }$ </td><td> $\overline { { 6 4 . 1 7 \pm 1 . 8 5 } }$ </td><td> $\overline { { 3 5 . 6 7 \pm 2 . 9 9 } }$ </td><td> $7 5 . 6 2 \pm 1 . 6 9$ </td><td></td><td> $6 6 . 0 6 \pm 1 . 8 8$ </td><td> $2 . 1 6 \pm 0 . 4 6$ </td></tr><tr><td>FedAVG ft</td><td> $\overline { { { 7 9 . 9 9 \pm 0 . 9 5 } } }$ </td><td> $\overline { { 5 5 . 9 4 \pm 6 . 8 6 } }$ </td><td> $\overline { { 6 7 . 1 0 \pm 5 . 6 3 } }$ </td><td></td><td> $\overline { { 5 7 . 4 7 \pm 3 . 8 4 } }$ </td><td> $2 3 . 5 6 \pm 1 . 0 8$ </td><td> $\overline { { 5 5 . 5 7 \pm 5 . 4 6 } }$ </td><td> $\overline { { 6 9 . 3 9 \pm 1 . 7 0 } }$ </td><td></td><td> $\overline { { 5 6 . 6 0 \pm 1 . 9 2 } }$ </td><td> $\overline { { 3 . 0 9 \pm 0 . 2 8 } }$ </td></tr><tr><td>DBE ft</td><td> $7 9 . 8 6 \pm 0 . 7 9$ </td><td> $6 0 . 2 6 \pm 6 . 9 4$ </td><td> ${ \bf 6 8 . 9 7 \pm 7 . 3 7 }$ </td><td></td><td> $5 9 . 2 9 \pm 6 . 0 9$ </td><td> $3 5 . 6 8 \pm 5 . 4 9$ </td><td> ${ \bf 6 6 . 8 5 \pm 3 . 4 2 }$ </td><td> $7 1 . 1 5 \pm 1 . 6 9$ </td><td></td><td> $6 1 . 8 2 \pm 1 . 8 7$ </td><td> $2 . 0 9 \pm 0 . 5 9$ </td></tr><tr><td>DITTO ft</td><td> $7 8 . 9 7 \pm 1 . 2 5$ </td><td> $5 1 . 4 0 \pm 4 . 7 0$ </td><td> $6 5 . 0 5 \pm 1 0 . 3 6$ </td><td></td><td> $6 1 . 2 1 \pm 3 . 4 0$ </td><td> $3 0 . 8 4 \pm 4 . 5 2$ </td><td> $4 5 . 5 5 \pm 6 . 6 5$ </td><td> $6 8 . 2 1 \pm 2 . 2 6$ </td><td></td><td> $5 5 . 5 0 \pm 2 . 9 6$ </td><td> $2 . 1 2 \pm 0 . 5 6$ </td></tr><tr><td>PAFP</td><td> $\overline { { 7 6 . 4 9 \pm 0 . 7 3 } }$ </td><td> $\overline { { 7 6 . 5 2 \pm 5 . 4 9 } }$ </td><td> $\overline { { 6 4 . 9 2 \pm 0 . 8 9 } }$ </td><td></td><td> $\overline { { 6 6 . 2 4 \pm 1 . 3 I } }$ </td><td> $\overline { { 6 4 . 8 3 \pm 0 . 7 4 } }$ </td><td> $5 3 . 2 3 \pm 1 4 . 4 4$ </td><td> $\overline { { 7 2 . 9 7 \pm 1 . 3 4 } }$ </td><td></td><td> $\overline { { 6 7 . 0 4 \pm 2 . 5 I } }$ </td><td> $\overline { { 0 . 9 7 \pm 0 . 7 6 } }$ </td></tr><tr><td> $\mathrm { P A F P - a l t ~ } ( \mathrm { r } = 1 0 )$ </td><td> $7 6 . 5 3 \pm 1 . 5 1$ </td><td> $7 1 . 8 8 \pm 5 . 9 9$ </td><td> $6 4 . 3 8 \pm 4 . 4 8$ </td><td></td><td> $6 6 . 1 9 \pm 1 . 1 2$ </td><td> $6 3 . 6 2 \pm 1 . 4 6$ </td><td> $6 2 . 5 6 \pm 2 . 3 0$ </td><td> $7 2 . 2 6 \pm { I . 4 2 }$ </td><td></td><td> ${ \pm 7 . 5 3 \pm 1 . 3 6 }$ </td><td> ${ \bf 0 . 3 5 \pm 0 . 1 3 }$ </td></tr><tr><td>ft 2B</td><td> $7 8 . 4 8 \pm 0 . 6 9$ </td><td> $\overline { { 6 9 . 0 2 \pm 2 . 4 2 } }$ </td><td> $6 3 . 0 8 \pm 5 . 1 2$ </td><td></td><td> $6 5 . 9 4 \pm 1 . 1 0$ </td><td> $\overline { { 5 9 . 7 2 \pm 2 . 4 8 } }$ </td><td> $3 7 . 5 3 \pm 5 . 0 0$ </td><td> $\overline { { 7 1 . 9 9 \pm 0 . 6 1 } }$ </td><td></td><td> $6 2 . 2 9 \pm 0 . 4 2$ </td><td> $\overline { { 1 . 6 7 \pm 0 . 4 2 } }$ </td></tr><tr><td> $\mathrm { f t } 2 \mathrm { B } + \mathrm { L C }$ </td><td> $7 7 . 9 8 \pm 0 . 9 4$ </td><td> $6 7 . 5 3 \pm 7 . 1 8$ </td><td> $6 4 . 5 5 \pm 4 . 7 1$ </td><td></td><td> $6 5 . 0 3 \pm 2 . 1 2$ </td><td> $5 7 . 7 9 \pm 1 . 2 0$ </td><td> $4 5 . 0 7 \pm 1 2 . 7 7$ </td><td> $7 1 . 6 7 \pm 1 . 5 9$ </td><td></td><td> $6 2 . 9 9 \pm 1 . 9 2$ </td><td> $1 . 3 7 \pm 0 . 6 3$ </td></tr><tr><td> $\mathrm { P A F P + L C }$ </td><td> $7 6 . 4 0 \pm 0 . 4 7$ </td><td> $7 3 . 2 5 \pm 7 . 4 5$ </td><td> $6 5 . 4 2 \pm 0 . 9 4$ </td><td></td><td> ${ \bf 6 6 . 3 1 \pm 0 . 5 7 }$ </td><td> $6 3 . 0 5 \pm 1 . 9 1 $ </td><td> $4 0 . 0 9 \pm 1 1 . 2 5$ </td><td> $7 2 . 1 4 \pm 1 . 4 5$ </td><td></td><td> $6 4 . 0 9 \pm 2 . 1 3$ </td><td> $1 . 6 3 \pm 0 . 7 7$ </td></tr><tr><td> $\mathrm { P A F P - a l t } \ ( \mathrm { r } = 5 0 )$ </td><td> $7 6 . 4 6 \pm 0 . 8 5$ </td><td> $7 1 . 8 2 \pm 9 . 7 0$ </td><td> $6 4 . 4 5 \pm 4 . 0 4$ </td><td></td><td> $6 3 . 9 3 \pm 1 . 3 9$ </td><td> ${ \bf 6 4 . 9 2 \pm 1 . 1 2 }$ </td><td> $3 6 . 7 2 \pm 3 . 6 7$ </td><td> $7 1 . 5 6 \pm 2 . 1 9$ </td><td></td><td> $6 3 . 0 5 \pm 1 . 8 0$ </td><td> $1 . 4 1 \pm 0 . 3 6$ </td></tr><tr><td> $\mathrm { P A F P - a l t } \ ( \mathrm { r } = 2 )$ </td><td> $7 6 . 7 5 \pm 0 . 2 6$ </td><td> $6 9 . 9 6 \pm 6 . 0 0$ </td><td> $6 5 . 6 0 \pm 0 . 7 3$ </td><td></td><td> $6 5 . 4 1 \pm 2 . 0 8$ </td><td> $6 2 . 6 0 \pm 1 . 7 9$ </td><td> $5 1 . 7 2 \pm 1 3 . 4 5$ </td><td> $7 1 . 9 1 \pm 1 . 1 2$ </td><td></td><td> $6 5 . 3 4 \pm 2 . 0 3$ </td><td> $0 . 9 I \pm 0 . 5 7$ </td></tr></table>

on its last two blocks, and $^ { \bullet \bullet } \mathrm { L C } ^ { \bullet \bullet }$ indicates the incorporation of the Logit Corrector.

## VI. RESULTS

a) Evaluation protocol: For each experiment, we report the mean and standard deviation of the balanced accuracy over 5 independent runs. We evaluate all baselines from a pFL perspective, even those that were not originally designed as pFL methods. Concretely, for every method we report the performance of each client’s local fine-tuned model at the end of its local training, rather than that of the aggregated global model as is conventional for traditional FL methods. This ensures all methods are assessed on the same footing, namely their per-client personalized performance. We report per-client balanced accuracy, as well as two aggregate measures: the weighted average, using data proportion $n _ { k } / N ,$ , and the unweighted average, using equal client weighting $1 / K$ to assess overall system performance from both a populationlevel and a client-level perspective, respectively. In the context of pFL where the goal is to obtain the best local model for each client, an unweighted average metric provides a more faithful assessment of the system, as it prevents large clients from masking the poor performance of smaller ones. To evaluate whether a given method reduces performance disparity across clients, we further report the inter-client variance of balanced accuracy, computed within each run and then averaged (with standard deviation) across the 5 runs.

Overall, we observe that the PAFP variants achieve the best overall performance under both evaluation schemes. The alternate variant (PAFP-alt) obtains the highest unweighted average balanced accuracy $( 6 7 . 5 3 \pm 1 . 3 6 \% )$ and the second-highest weighted average $( 7 2 . 2 6 \pm 1 . 4 2 \% )$ . The highest weighted average is attained by non-alternate PAFP $( 7 2 . 9 7 \pm 1 . 3 4 \% )$ The best per-client results for clients 2, 4, and 5 are likewise attained by PAFP configurations (PAFP, PAFP + LC, and PAFPalt $( r = 5 0 )$ , reaching $7 6 . 5 2 \pm 5 . 4 9 \%$ $6 6 . 3 1 \pm 0 . 5 7 \%$ , and $6 4 . 9 2 \pm 1 . 1 2 \%$ , respectively). Furthermore, PAFP-alt $( r = 1 0 )$ reduces the inter-client variance of balanced accuracy to 0.35 (compared to the lowest baseline variance of 2.09). Thus, the configuration that is most accurate under the client-level unweighted average is also the most equitable across clients, rather than trading one for the other.

Results from the isolated (non-FL) setting show that clients holding large amounts of data benefit little from federation. This is most visible for client 1 $( 8 0 . 1 6 \pm 1 . 0 1 \%$ isolated vs. 79.99±0.95% FedAvg-ft), whose performance is already close to its local-only ceiling.

Experimental results are presented in Table II. The isolated case corresponds to a non-FL scenario where each client trains a model solely on its own local data. These isolated results are included solely as a baseline reference, with our analysis focused primarily on the federated training outcomes. We benchmark the FL baselines FedAvg, DBE, and Ditto in a fine-tuned version (ft), since we start from a pre-trained model, which departs from the train-from-scratch assumption these methods were designed for. We deliberately exclude Parameter-Efficient Fine-Tuning (PEFT) methods from our evaluation. While they share similarities with our local adaptation objective, our primary focus is isolating and addressing data skew rather than optimizing parameter efficiency.

Finally, by comparing each configuration utilizing an LC module $( { \mathrm { P A F P } } + { \mathrm { L C } }$ and $\mathrm { f t } { - } 2 \mathrm { B } + \mathrm { L C } )$ to its counterpart without LC (PAFP and ft-2B), we note that adding the Logit Corrector does not improve performance.

We present PAFP and PAFP-alt $( r ~ = ~ 1 0 )$ as the bestperforming configurations, and ft-2B, ft-2B + LC, PAFP + LC, PAFP-alt $( r = 5 0 )$ , and PAFP-alt (r = 2) as exploratory experiments investigating various configuration variations. Here, $\mathrm { ^ { 6 6 } f t { - } } 2 \mathrm { B } ^ { \bullet }$ denotes the EfficientNet model fine-tuned exclusively

## VII. DISCUSSION

a) Interpreting the main result: The strongest configuration couples the FA modules with an alternating optimization schedule, and the gap between this variant and the singleoptimizer one is informative. Because PAFP fine-tunes a backbone pre-trained on natural images, all clients start from a common, already-structured latent space. Before inter-client heterogeneity becomes the dominant obstacle, every client must first adapt the shared weights w from the naturalimage domain to the medical-imaging domain. Only once w has specialized to medical imaging does the residual divergence between client distributions become the problem the FA modules are meant to solve. We hypothesize that alternating between rounds that update the shared parameters and rounds that update the local parameters allows the system to interleave these two processes: global domain adaptation

## REFERENCES

(natural to medical) and local adaptation to each client’s distribution. In the single-optimizer variant, by contrast, the alignment gradient flows into w from the first round, pushing the shared backbone to satisfy a client-specific alignment objective before it has adapted to the medical domain. This conflates the two adaptation problems and perturbs the shared representation, which we believe explains its lower and less stable performance. That the alternating variant simultaneously attains the best unweighted average and the lowest inter-client variance supports this reading: shielding w from the alignment gradient yields both a better shared representation and more equitable personalization.

b) Limitations: Several limitations temper these conclusions. First, our evaluation covers a single dataset and imaging modality (FedISIC dermoscopy). While the problem formulation and the PAFP architecture make no domainspecific assumptions, the generality of the approach beyond dermoscopic classification remains argued rather than empirically demonstrated. Second, although we formalize four types of skew, only feature skew and label skew receive dedicated mechanisms. Concept drift and quantity skew are handled only implicitly: quantity skew in particular is reflected in our unweighted evaluation and in the data-proportional aggregation, but PAFP contains no component that explicitly targets it. Third, the FA modules introduce a modest communication overhead, since each client transmits per-layer feature statistics alongside its weights at every round. This corresponds to 56,672 statistics, equivalent to an increase of 453 kB of communication per client. Compared to the 3,181,612 shared parameters, this represents a mere 1.7% increase in communication volume and can thus be considered negligible. Finally, because the backbone model was pretrained on natural images rather than medical imaging, the observed performance gains may be confounded by the domain shift. Consequently, further experiments are required to isolate whether the proposed method effectively mitigates client heterogeneity independent of this initial domain gap.

c) Perspectives: Three directions follow naturally from this work. The most immediate is to extend PAFP to Magnetic Resonance Imaging (MRI) , a non-quantitative modality whose signal intensities are not standardized across scanners, field strengths, and acquisition protocols, and which therefore exhibits particularly strong and structured inter-client feature skew. The second direction is to leverage a medical imaging foundation model to mitigate the domain gap that the backbone must bridge during early training, thereby isolating data skew as the primary bottleneck and more rigorously demonstrating PAFP’s efficacy. The third direction is to address quantity skew explicitly within the mixed-skew setting, rather than leaving it to the aggregation and evaluation scheme, so that data-scarce clients are supported by a dedicated mechanism as feature and label skew already are. Together, these extensions would move the framework from a single-modality proof of concept toward the general treatment of mixed skew that motivates it.

[1] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. van der Laak, B. van Ginneken, and C. I. Sanchez,´ “A survey on deep learning in medical image analysis,” Medical Image Analysis, vol. 42, pp. 60–88, 2017.

[2] S. K. Zhou, H. Greenspan, C. Davatzikos, J. S. Duncan, B. Van Ginneken, A. Madabhushi, J. L. Prince, D. Rueckert, and R. M. Summers, “A review of deep learning in medical imaging: Imaging traits, technology trends, case studies with progress highlights, and future promises,” Proc. of the IEEE, vol. 109, no. 5, pp. 820–838, 2021.

[3] M. Moor, O. Banerjee, Z. S. H. Abad, H. M. Krumholz, J. Leskovec, E. J. Topol, and P. Rajpurkar, “Foundation models for generalist medical artificial intelligence,” Nature, vol. 616, no. 7956, pp. 259–265, 2023.

[4] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in 20th International Conference on Artificial Intelligence and Statistics, ser. PMLR, vol. 54, 2017, pp. 1273–1282.

[5] P. Kairouz, H. B. McMahan et al., “Advances and open problems in federated learning,” Foundations and Trends in Machine Learning, vol. 14, no. 1–2, 2021.

[6] N. Rieke, J. Hancox, W. Li, F. Milletar\`ı, H. R. Roth, S. Albarqouni, S. Bakas, M. N. Galtier, B. A. Landman, K. Maier-Hein, S. Ourselin, M. Sheller, R. M. Summers, A. Trask, D. Xu, M. Baust, and M. J. Cardoso, “The future of digital health with federated learning,” npj Digital Medicine, vol. 3, no. 1, p. 119, 2020.

[7] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, “Federated learning with non-iid data,” arXiv:1806.00582, 2018.

[8] Z. Lu, H. Pan, Y. Dai, X. Si, and Y. Zhang, “Federated learning with non-iid data: A survey,” IEEE Internet ofThings Journal, vol. 11, no. 11, pp. 19 188–19 209, 2024.

[9] A. Z. Tan, H. Yu, L. Cui, and Q. Yang, “Towards personalized federated learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 12, pp. 9587–9603, 2023.

[10] N. Tajbakhsh, J. Y. Shin, S. R. Gurudu, R. T. Hurst, C. B. Kendall, M. B. Gotway, and J. Liang, “Convolutional neural networks for medical image analysis: Full training or fine tuning?” IEEE Transactions on Medical Imaging, vol. 35, no. 5, pp. 1299–1312, 2016.

[11] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, “Transfusion: Understanding transfer learning for medical imaging,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[12] Y. Chen, X. Qin, J. Wang, C. Yu, and W. Gao, “Fedhealth: A federated transfer learning framework for wearable healthcare,” IEEE Intelligent Systems, vol. 35, no. 4, pp. 83–93, 2020.

[13] J. Nguyen, J. Wang, K. Malik, M. Sanjabi, and M. Rabbat, “Where to begin? on the impact of pre-training and initialization in federated learning,” in 12th International Conference on Learning Representations, 2023.

[14] J. Ogier du Terrail, S.-S. Ayed, E. Cyffers, F. Grimberg, C. He et al., “Flamby: Datasets and benchmarks for cross-silo federated learning in realistic healthcare settings,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 5315–5334.

[15] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proceedings of Machine Learning and Systems, vol. 2, 2020, pp. 429–450.

[16] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “SCAFFOLD: Stochastic controlled averaging for federated learning,” in 37th International Conference on Machine Learning, ser. PMLR, vol. 119, 2020, pp. 5132–5143.

[17] Q. Li, B. He, and D. Song, “Model-contrastive federated learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 10 713–10 722.

[18] A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 3557–3568.

[19] T. Li, S. Hu, A. Beirami, and V. Smith, “Ditto: Fair and robust federated learning through personalization,” in 38th International Conference on Machine Learning, ser. PMLR, vol. 139, 2021, pp. 6357–6368.

[20] X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, “FedBN: Federated learning on non-IID features via local batch normalization,” in International Conference on Learning Representations, 2021.

[21] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, “Federated learning with personalization layers,” arXiv:1912.00818, 2018.

[22] L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, “Exploiting shared representations for personalized federated learning,” in 38th International Conference on Machine Learning, ser. PMLR, vol. 139, 2021, pp. 2089–2099.

[23] J. Zhang, Y. Hua, J. Cao, H. Wang, T. Song, Z. XUE, R. Ma, and H. Guan, “Eliminating domain bias for federated learning in representation space,” in 37th Conference on Neural Information Processing Systems, 2023.

[24] Z. Wang, Z. Wang, Z. Wang, X. Fan, and C. Wang, “Federated learning with domain shift eraser,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 4978–4987.

[25] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, and C. Wu, “Federated learning with label distribution skew via logits calibration,” in 39th International Conference on Machine Learning, ser. PMLR, vol. 162, 2022, pp. 26 311–26 329.

[26] H.-Y. Chen and W.-L. Chao, “On bridging generic and personalized federated learning for image classification,” in International Conference on Learning Representations, 2022.

[27] Z. Cai, Y. Shi, W. Huang, and J. Wang, “Fed-CO2: Cooperation of online and offline models for severe data heterogeneity in federated learning,” in 37th Conference on Neural Information Processing Systems, 2023.

[28] H. Zhao, W. Du, F. Li, P. Li, and G. Liu, “Fedprompt: Communicationefficient and privacy-preserving prompt tuning in federated learning,” in International Conference on Acoustics, Speech and Signal Processing, 2023, pp. 1–5.

[29] Z. Zhang, Y. Yang, Y. Dai, Q. Wang, Y. Yu, L. Qu, and Z. Xu, “FedPETuning: When federated learning meets the parameter-efficient tuning methods of pre-trained language models,” in Findings of the Association for Computational Linguistics, 2023, pp. 9963–9977.

[30] D. Cai, Y. Wu, S. Wang, F. X. Lin, and M. Xu, “Efficient federated learning for modern nlp,” in 29th Annual International Conference on Mobile Computing and Networking. ACM, 2023.

[31] T. Guo, S. Guo, J. Wang, X. Tang, and W. Xu, “Promptfl: Let federated participants cooperatively learn prompts instead of models – federated learning in age of foundation model,” IEEE Transactions on Mobile Computing, vol. 23, no. 5, pp. 5179–5194, 2024.

[32] F.-E. Yang, C.-Y. Wang, and Y.-C. F. Wang, “Efficient model personalization in federated learning via client-specific prompt generation,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 159–19 168.

[33] Y. Liu, G. Luo, and Y. Zhu, “ FedFMS: Exploring Federated Foundation Models for Medical Image Segmentation ,” in Medical Image Computing and Computer Assisted Intervention, vol. LNCS 15008. Springer Nature Switzerland, 2024.

[34] C. Xie, D.-A. Huang, W. Chu, D. Xu, C. Xiao, B. Li, and A. Anandkumar, “Perada: Parameter-efficient federated learning personalization with generalization guarantees,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 838–23 848.

[35] Z. Zhang, P. Liu, J. Xu, and R. Hu, “Fed-hello: Efficient federated foundation model fine-tuning with heterogeneous lora allocation,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 10, pp. 17 556–17 569, 2025.

[36] wanglu, H. Yu, J. Wang, D. Teney, H. Wang, Y. Zhu, Y. Chen, Q. Yang, X. Xie, and X. Ji, “ZOOPFL: Exploring black-box foundation models for personalized federated learning,” in International Workshop on Federated Foundation Models, 2024.

[37] M. Luo, F. Chen, D. Hu, Y. Zhang, J. Liang, and J. Feng, “No fear of heterogeneity: Classifier calibration for federated learning with non-iid data,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 5972–5984.

[38] X. Fang and M. Ye, “Towards robust parameter-efficient fine-tuning for federated learning,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 141 777–141 800.

[39] S. Zagoruyko and N. Komodakis, “Diracnets: Training very deep neural networks without skip-connections,” arXiv preprint arXiv:1706.00388, 2017.

[40] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in IEEE International Conference on Computer Vision, 2017, pp. 2980–2988.