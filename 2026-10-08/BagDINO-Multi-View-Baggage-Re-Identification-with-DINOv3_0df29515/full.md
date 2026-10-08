# BagDINO: Multi-View Baggage Re-Identification with DINOv3

Vita Santa Barletta   
University of Bari Aldo moro   
Department of Computer Science   
Bari, Italia   
vita.barletta@uniba.it

Massimiliano Morga SER&Practices Spin-off of the University of Bari Aldo Moro Bari, Italia m.morga@serandp.com

Danilo Caivano University of Bari Aldo Moro, SER&Practices Department of Computer Science Bari, Italia danilo.caivano@uniba.it

Rebecca Margiotta University of Bari Aldo Moro Department of Computer Science Bari, Italia r.margiotta5@studenti.uniba.it

Davide Pio Posa University of Bari Aldo Moro, SER&Practices Department of Computer Science Bari, Italia d.posa@studenti.uniba.it

Abstract—Mishandled checked baggage remains a recurrent issue in airport operations, and current recovery workflows still largely rely on tag-based tracking, which does not directly support visual identification when tag evidence is missing or unavailable. This paper investigates baggage re-identification as an instance-level retrieval problem in a multi-camera setting, leveraging DINOv3 foundation-model representations to match a query image against a gallery of registered baggage images. A Torchreid-style BNNeck re-identification head is placed on top of a DINOv3 backbone, and parameter-efficient adaptation is performed via LoRA. Experiments are conducted on the MVB benchmark using a progressive study that compares a fully frozen backbone against LoRA and fine-tuning strategies. Results indicate that parameter-efficient adaptation of foundation-model features provides an effective and stable approach for multi-view baggage re-identification under limited training data.

Index Terms—re-identification, baggage, DINO, computer vision

## I. INTRODUCTION

Despite continuous improvements in airport operations, mishandled checked baggage remains a persistent operational and customer experience issue. Recent industry reporting estimates a global mishandling rate of 6.9 bags per 1,000 passengers in 2023 (down from 7.6 in 2022), with delayed bags accounting for the majority of cases (77%) and transfer-related issues representing a large fraction of delays (46%) [1]. Beyond passenger frustration, mishandling also translates into nonnegligible operational costs: IATA (International Air Transport Association) estimates that reuniting passengers and baggage costs roughly \$100 per bag [2].

In current airport workflows, baggage identification and tracking are predominantly tied to tag-based processes, where tracking evidence must be recorded at specific checkpoints. To standardize this process and reduce mishandling, IATA Resolution 753 requires tracking at four core points: Acceptance, Load, Transfer, and Arrival [2]. While such mechanisms provide essential traceability, they do not directly address scenarios in which a bag must be recognized from images (e.g.,

CCTV or operational cameras), or matched across cameras when tag evidence is missing, corrupted, or unavailable at the time of inspection.

At the same time, the increasing adoption of self-service and automated check-in/bag-drop solutions creates an opportunity for visual registration under controlled conditions. In this setting, the standard identification step (tag assignment and record creation) can be complemented by capturing a small set of multi-view images of each bag in a stable environment (consistent viewpoint constraints, lighting, and background), thus producing a clean visual “signature” at registration time. Later, when the bag is observed in less constrained areas of the airport, the system can perform instance-level retrieval by matching a query image against a gallery of registered multiview images to retrieve the corresponding bag identity (and hence its associated record).

Baggage ReID is challenging because many baggage items share nearly identical global appearance due to industrial manufacturing, and the discriminative evidence is frequently confined to fine-grained instance-specific cues such as stickers, surface wear, or small damages. These cues may further be affected by viewpoint changes, occlusions, motion blur, illumination differences, and background clutter. This paper investigates baggage re-identification on the MVB benchmark using foundation-model visual representations, leveraging self supervised transformer features for cross-view matching [3].

Therefore, considering the need to improve current solutions, the paper proposes BagDINO as a foundation-modelbased approach to baggage re-identification and makes the following three main contributions:

The development of a baggage re-identification pipeline (Figure 1) built on top of DINOv3 ViT-L/16 representations [4] and a Torchreid-style BNNeck head [5], framing the task as instance-level retrieval in a multi-camera setting.

• A LoRA-based parameter-efficient adaptation configuration [6] is evaluated by injecting low-rank updates into the backbone projections qkv, proj, fc1, and fc2, while keeping the original backbone weights frozen.

• A progressive experimental study on MVB [3] compares a fully frozen backbone, LoRA adaptation, and partial/full fine-tuning under a shared training recipe, reporting performance in terms of mAP and CMC at Rank-1/5/10; across the tested configurations, results range from 0.57 mAP and 0.60 Rank-1 in the frozen setting to 0.88 mAP and 0.87 Rank-1 with LoRA (Rank-5/10: 0.96/0.98).

The paper is structured as follows: Section II reviews the main research directions in baggage re-identification and related instance-retrieval approaches. Section III presents the proposed pipeline, including the DINOv3 backbone, the LoRA-based parameter-efficient adaptation strategy, the reidentification head, and the training objective. Section IV describes the experimental setup on the MVB benchmark and reports a comparative evaluation of different adaptation strategies together with an analysis of retrieval errors. Finally, Section V concludes the paper and discusses possible directions for future research.

## II. RELATED WORK

Baggage re-identification has recently gained momentum as a practical instance-retrieval problem in airport logistics, largely driven by the introduction of the Multi-View Baggage (MVB) benchmark, a large-scale dataset with 4,519 identities and 22,660 images acquired in real multi-camera airport environments, which highlights domain shift between acquisition points and strong variations due to viewpoint, illumination, occlusions, blur, and background clutter [3].Compared to person re-identification, baggage re-identification is inherently more challenging due to the high visual similarity across different baggage identities and the significant appearance variations within the same identity [7].

In fact, early and “classical” pipelines on MVB typically adapt person/object ReID recipes with CNN backbones (e.g., ResNet-50) and metric learning (ID loss + triplet variants), and progressively enhance the baseline by injecting context and fine-grained cues. For example, Mazzeo et al. [8] propose an attention/multi-granularity strategy that combines global descriptors with local feature learning (e.g., masking/erasingstyle mechanisms to force attention to discriminative regions) and global-context modeling, obtaining consistent gains over strong baselines by improving robustness to viewpoint changes. Another influential direction is to explicitly model the multi-view nature of the problem at the loss/design level: QuadNet introduces a multi-view quadruplet loss to enforce cross-view compactness for the same identity while separating hard negatives both within and across views, and further strengthens discrimination with view-aware local features (e.g., CAM-guided regions), robustness-oriented augmentations (e.g., blur), and auxiliary supervision such as material prediction, reporting substantial improvements over conventional ReID baselines on MVB [9].

More recent works focus on improving cross-domain and cross-camera robustness and the final ranking stage; for instance, methods that combine domain-robust feature learning (e.g., normalization and attention modules), hard-sample mining, and camera-aware re-ranking (as in CDR-CARNet) show that carefully addressing domain shift and exploiting camera structure can markedly boost mAP and Rank-1 on MVB [10].

The Region-to-Global Vision Transformer (RGViT), the best reported SOTA in our survey, aggregates local evidence into regional tokens and explicitly models global spatial relations between regions, optimizing the embedding with a quadruplet objective designed for near-duplicate instances; in MVB it reports 88.7% Rank-1 and 89.6% mAP, suggesting that structured local-to-global modeling is particularly effective when many identities share highly similar global appearance [11].

Although recent methods have significantly improved performance on the MVB benchmark, most existing solutions still rely on adaptations of person re-identification pipelines originally designed for human appearance modeling. However, baggage objects present different visual characteristics compared to human subjects, including weaker structural priors, highly similar object categories, and frequent occlusions or deformations during handling. Consequently, approaches developed for person ReID may not fully exploit the discriminative cues specific to baggage identification.

This observation highlights the need for more specialized feature learning strategies capable of capturing both global structural patterns and subtle local differences between visually similar baggage items.

## III. PROPOSED METHOD

The proposed approach, BagDINO, aims to improve robustness to viewpoint variations, background clutter, and visual similarity across baggage identities, ultimately leading to more reliable instance matching in real-world airport scenarios. Consequently, the proposed pipeline for baggage ReID is organized around three main components:

• Backbone: DINOv3 foundation model with parameterefficient fine-tuning via Low-Rank Adaptation;

• ReID Head: takes the backbone patch token and produces the final embeddings;

• Composite Loss: a combination of a Triplet loss and a Cross-Entropy loss used during training.

## A. Backbone and LoRA Fine-Tuning

As visual backbone we adopt DINOv3 [4], a Vision Transformer (ViT) pre-trained with self-supervised objectives on a large and curated image corpus. DINOv3 produces rich, semantically meaningful patch-level representations that have shown strong transferability across visual tasks without taskspecific architectural changes.

Given the relatively small scale of the MVB training split and the high computational cost of full fine-tuning a large ViT, we adopt Low-Rank Adaptation (LoRA) [6] as a parameterefficient fine-tuning strategy. LoRA injects trainable low-rank decomposition matrices into selected linear projections of each Transformer block, while keeping the original pre-trained weights frozen.

![](images/29cba93b477533bae3bb50345cc5dc56474072bf283ba52c5b2bcc70ba7f7fd1.jpg)  
Fig. 1: Overview of the proposed baggage re-identification pipeline. A DINOv3 ViT backbone provides a global image representation (from the [CLS] token) and is adapted via LoRA modules injected into selected transformer projections (e.g., qkv, proj, fc1, fc2) while keeping the pre-trained weights frozen. The resulting embedding is processed by a Torchreidstyle BNNeck head and optimized with a composite objective that combines identity classification (cross-entropy with labe smoothing) and batch-hard triplet loss. At inference time, the linear classifier is discarded and retrieval is performed using the normalized embeddings (e.g., cosine similarity) to rank gallery items for a given query.

Formally, for a pre-trained weight matrix $\mathbf { W } _ { 0 } ~ \in ~ \mathbb { R } ^ { d \times k }$ LoRA parameterizes the update as:

$$
\mathbf { W } = \mathbf { W } _ { 0 } + \Delta \mathbf { W } = \mathbf { W } _ { 0 } + \frac { \alpha } { r } \mathbf { B } \mathbf { A } ,
$$

where $\mathbf { W } _ { 0 }$ is kept frozen during training, $\textbf { A } \in \mathbb { R } ^ { r \times k }$ and $\mathbf { B } \in \mathbb { R } ^ { d \times r }$ are the trainable low-rank matrices, $r \ll \operatorname* { m i n } ( d , k )$ is the rank hyperparameter controlling the number of trainable parameters, and α is a scalar that governs the magnitude of the adaptation [6]. At initialization, B = 0 and A is randomly drawn from a Gaussian distribution, ensuring ∆W = 0 at the start of fine-tuning [6].

In this implementation, LoRA adapters are applied to four linear projections within each Transformer block: the combined Query-Key-Value projection $( \mathrm { { q k v } ) }$ , the attention output projection (proj), and the two fully-connected layers of the feed-forward network (fc1, fc2). This choice ensures that both the attention mechanism and the token-wise non-linear transformations are adapted to the baggage domain, while the pre-trained weights of the backbone remain frozen.

## B. Re-Identification Head

The backbone produces a sequence of patch tokens, so the [CLS] token representation is extracted as the global image descriptor, as it aggregates information from all spatial positions through the self-attention mechanism. This embedding $\textbf { f } \in \mathbb { R } ^ { D }$ is then passed through the re-identification head, which consists of:

1) Batch Normalization Neck (BNNeck) [5]: a batch normalization layer applied after the backbone output and before the classification layer. Following common ReID practice, this design separates the feature representation used for metric learning from the one used for identity classification, which is known to support stable optimization and effective retrieval performance [5].

2) Linear Classifier: a fully-connected layer with $N _ { \mathrm { i d } }$ outputs (one per training identity), used only during training to compute the identity loss; at inference time this layer is discarded and retrieval is performed directly via cosine similarity (Figure 2).

At inference time, given a query image and a gallery of images, the gallery items are ranked by decreasing cosine similarity:

$$
s ( \mathbf { x } _ { q } , \mathbf { x } _ { i } ) = \frac { \mathbf { f } _ { q } \cdot \mathbf { f } _ { i } } { \| \mathbf { f } _ { q } \| \| \mathbf { f } _ { i } \| }
$$

![](images/2486c6dc63888e2cb57771706e11278dfc8a22b96af87e5791370b3f9fc81d27.jpg)  
Fig. 2: Inference-stage retrieval. Query and gallery images are embedded by the backbone+BNNeck, and gallery instances are ranked by similarity in the embedding space to produce the final match list.

## C. Loss Function

Training is supervised by a combination of an identity classification loss and a metric learning loss, following the

standard approach in ReID pipelines [5].

a) Identity Loss: Identity classification loss is a standard cross-entropy loss with label smoothing [12] applied to post-BN features <sup>ˆ</sup>f:

$$
\mathcal { L } _ { \mathrm { I D } } = - \sum _ { c = 1 } ^ { N _ { \mathrm { i d } } } \tilde { y } _ { c } \log \hat { p } _ { c } ,
$$

where $\hat { p } _ { c }$ is the predicted probability for class c and $\tilde { y } _ { c } =$ $\left( 1 - \varepsilon \right) y _ { c } + \varepsilon / { N _ { \mathrm { i d } } }$ is the smoothed label, with ε the smoothing factor. Label smoothing regularizes the classifier by preventing overconfident predictions on training identities.

b) Triplet Loss: The metric learning component is a batch-hard triplet loss [13], which, for each anchor sample in a mini-batch, selects the hardest positive (same identity, maximum distance) and the hardest negative (different identity, minimum distance):

$$
\mathcal { L } _ { \mathrm { T r i } } = \sum _ { a \in \mathcal { B } } \left[ \operatorname* { m a x } _ { p : y _ { p } = y _ { a } } d ( \mathbf { f } _ { a } , \mathbf { f } _ { p } ) - \operatorname* { m i n } _ { n : y _ { n } \neq y _ { a } } d ( \mathbf { f } _ { a } , \mathbf { f } _ { n } ) + m \right] _ { + } ,
$$

where $d ( \cdot , \cdot )$ denotes Euclidean distance, m is the margin, and $[ \cdot ] _ { + } = \operatorname* { m a x } ( 0 , \cdot )$

c) Total Loss: The overall training objective is the weighted sum of the two terms:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { I D } } + \lambda \mathcal { L } _ { \mathrm { T i } } , } \end{array}
$$

where λ controls the relative contribution of the metric learning term with respect to the identity classification term.

## IV. EXPERIMENTS

## A. MVB Dataset

All experiments are conducted on the Multi-View Baggage (MVB) benchmark, a large-scale dataset designed for baggage re-identification in real airport environments [3]. MVB contains 4,519 baggage identities and 22,660 annotated baggage images, and provides additional annotations such as surface material labels (e.g. soft, hard) and segmentation masks in the original release. Importantly, MVB is not limited to standard suitcases (e.g. rigid trolleys), but also includes a broader range of baggage items such as soft bags, duffels and transported parcels, better reflecting the diversity of objects that may be checked in and subsequently matched across cameras. Images are collected with a specially designed multi-view camera system and span two acquisition points with substantially different imaging condition, commonly described as BHS/conveyorbelt views and checkpoint views. This introduces a realistic domain gap across acquisition points (background, lighting, viewpoint/pose), and the dataset also exhibits strong intraclass variation due to occlusions, motion blur and background clutter. As a result, MVB does not represent a fully controlled capture setup, even though each acquisition point can be interpreted as an operational checkpoint that is compatible with the visual registration setting discussed in industry workflows. All experiments follow the official train/query/gallery splits and the evaluation protocol defined in the original paper. Performance is reported using standard retrieval metrics, namely mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) at Rank-1, Rank-5, and Rank-10.

![](images/bf00167157368ac3903195cd74fb09275b90f8dcf11e64cdafe0642b15cb97bc.jpg)  
Fig. 3: Example MVB query-gallery pairs illustrating the heterogeneity of baggage items and the variability induced by different acquisition conditions.

## B. Experimental Setup

All experiments follow a Torchreid-style training recipe with a BNNeck head and a composite objective that combines identity classification (cross-entropy) and batch-hard triplet loss (margin $m \ : = \ : 0 . 3 )$ with equal weights. We resize all images to 256 × 256 and apply standard ImageNet normalization; data augmentation includes random horizontal flipping $( p \ : = \ : 0 . 5 )$ and random cropping with padding (10 pixels). Training uses PK sampling with batch size 32 and $K = 4$ instances per identity (i.e., $P = 8$ identities per mini-batch) for 80 epochs, with the random seed fixed to 42. Optimization is performed with AdamW (base learning rate $3 . 2 5 \times 1 0 ^ { - 4 }$ weight decay $5 \times 1 0 ^ { - 4 } )$ and a cosine learning-rate schedule; we use separate optimizer parameter groups with a 10× smaller learning rate for trainable LoRA parameters in the backbone than for the BNNeck head. We apply gradient clipping with max norm 5.0 and optionally enable early stopping (patience 15, minimum ∆ of 0.001). For LoRA-based adaptation, lowrank updates (rank $1 6 , \alpha = 3 2 )$ are injected into the backbone projections qkv, proj, fc1, and fc2.

## C. Progressive Study

A progressive experimental analysis is conducted to quantify the impact of parameter-efficient adaptation and to compare it with partial and full fine-tuning strategies, while keeping the same ReID head and loss formulation across all runs.

a) Frozen backbone: To evaluate the out-of-the-box transferability of DINOv3 features to the baggage ReID task, the first experiment uses a fully frozen DINOv3 ViT-L/16 backbone and trains only the ReID head. This baseline achieves 0.57 mAP, with Rank-1/5/10 equal to 0.60/0.80/0.87, outperforming the original baseline reported in the MVB paper [3]. These results confirm that foundation-model representations provide a strong starting point, although domain-specific adaptation remains necessary to better cope with the crossview variability of MVB.

b) LoRA adaptation: Low-Rank Adaptation (LoRA) is then introduced to specialize the backbone to the operational baggage domain. LoRA modules are applied to qkv, proj, fc1, and fc2 layers (rank 16, alpha 32). This configuration yields a substantial performance improvement, reaching 0.88 mAP with Rank-1/5/10 of 0.87/0.96/0.98. The gain indicates that parameter-efficient updates are sufficient to adapt highcapacity ViT representations to fine-grained baggage cues, while mitigating the overfitting risk typically associated with full fine-tuning on limited training data.

c) Partial fine-tuning: Although full fine-tuning is not the primary strategy due to the limited size of the training split, a partial fine-tuning setting is evaluated for completeness. In this configuration, 75% of the backbone remains frozen, while the remaining layers are updated using a conservative learning rate (on the order of 10<sup>−6</sup>) and stronger weight decay. The model achieves 0.83 mAP with Rank-1/5/10 of 0.83/0.95/0.97. While competitive, this setup underperforms LoRA, suggesting that even partial fine-tuning may require more data to effectively adapt the backbone without degrading its generalization properties.

d) Full fine-tuning: Finally, all backbone parameters are unfrozen and optimized under the same conservative regime (low learning rate and aggressive weight decay). This configuration achieves 0.85 mAP, with Rank-1/5/10 equal to 0.86/0.95/0.96. Overall, the fine-tuning experiments confirm the initial hypothesis: given the restricted training size and the high visual similarity among many MVB instances, LoRA provides a more stable and effective adaptation mechanism than updating the entire backbone.

## D. Error Analysis

Beyond standard retrieval metrics, the purity of the top-k list is inspected, i.e., the fraction of the first k retrieved items that match the query identity. Under the adopted test protocol, each identity is represented by only a small number of gallery images (typically 1-3), which makes purity@5 inherently conservative: even when all available true matches are retrieved at the top, the remaining positions in the top-5 list must be filled with other identities and are therefore counted as impurities. In an operational setting, visual registration would typically store a larger multi-view template per item (e.g., 6+ images), increasing the number of available true matches and yielding a purer shortlist of candidates at fixed k. Consistently with the high Rank-1 performance of the best configuration (0.87), failures are relatively infrequent; however, when they occur, they often correspond to visually near-duplicate baggage items in the gallery (similar shape, material, and color patterns), leading to incorrect matches with high similarity scores. Figure 4 reports representative high-similarity failure cases, where the top-ranked gallery image is incorrect despite being very close to the query in the embedding space.

![](images/21a4b438f4656f9780ddc8e9d7e5f2ac66453a5d7eb0928000cb46e0bed5552f.jpg)  
Fig. 4: High-similarity failure cases on MVB. Three examples are reported, each shown as (i) the query, (ii) the corresponding true match in the gallery (positive), and (iii) the top-ranked incorrect retrieval returned with high confidence (similarity ≳ 78%). These errors typically occur when different baggage identities exhibit near-duplicate visual cues (e.g., similar shape, material, and color patterns), yielding very close neighbors in the embedding space.

## E. Results Summary

Table I summarizes the adaptation strategies evaluated on MVB. A frozen BagDINO backbone already provides a strong baseline (0.57 mAP, 0.60 Rank-1), while LoRA-based adaptation yields the best overall trade-off among the tested settings (0.88 mAP, 0.87 Rank-1). Partial and full fine-tuning remain competitive but do not match the LoRA configuration under the same training recipe.

Table II compares BagDINO + LoRA with representative methods from the literature on the same benchmark. The proposed approach is competitive with prior work and achieves results close to the strongest published methods, while remaining based on a parameter-efficient adaptation scheme.

More generally, the results should also be interpreted in light of the different design choices adopted by the compared approaches. In particular, while some methods, such as RGViT [11], introduce additional architectural modules to improve feature modeling, BagDINO relies on a parameter-efficient LoRA configuration, where only a limited set of low-rank updates is optimized while the DINOv3 backbone remains frozen. This suggests that the observed differences across methods are not only related to benchmark scores but also reflect different trade-offs between architectural specialization and lightweight adaptation.

<table><tr><td>Method</td><td>mAP</td><td>Rank-1</td></tr><tr><td>BagDINO (Frozen)</td><td>0.57</td><td>0.60</td></tr><tr><td>BagDINO + LoRA</td><td>0.88</td><td>0.87</td></tr><tr><td>BagDINO FT (25%)</td><td>0.83</td><td>0.83</td></tr><tr><td>BagDINO FT (100%)</td><td>0.83</td><td>0.83</td></tr></table>

TABLE I: Results on the MVB benchmark in terms of mAP and Rank-1 summarizing the evaluated DINOv3-based configurations under different adaptation strategies.

<table><tr><td>Architecture</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Huangbin et al. [14]</td><td>0.83</td><td>0.85</td></tr><tr><td>Yang et al. [9]</td><td>0.86</td><td>0.88</td></tr><tr><td>Liu et al. [10]</td><td>0.87</td><td>0.86</td></tr><tr><td>Zhao et al. [15]</td><td>0.91</td><td>0.87</td></tr><tr><td>Mazzeo et al. [8]</td><td>0.63</td><td>0.69</td></tr><tr><td>Zhiwei et al. [11]</td><td>0.90</td><td>0.89</td></tr><tr><td>Zhang et al. [3]</td><td>0.50</td><td>0.50</td></tr><tr><td> $\mathrm { \Delta B a g D I N O + L o R A }$ </td><td>0.88</td><td>0.87</td></tr></table>

TABLE II: Results on the MVB benchmark in terms of mAP and Rank-1 comparing BagDINO best approach with the representative methods from literature.

a) Cross-Domain Evaluation: To assess whether the proposed approach can transfer beyond the baggage domain, we additionally evaluate it on Market-1501 [16]. Market-1501 is a widely used person re-identification benchmark composed of pedestrian images acquired from multiple cameras, and it is commonly adopted to evaluate cross-camera retrieval performance. This experiment is not intended as an in-domain comparison with MVB, but rather as an out-of-domain validation to assess whether the proposed adaptation strategy remains effective in a different visual context.

Table III reports the comparison with baggage ReID methods that were also benchmarked on Market-1501, where BagDINO + LoRA achieves the highest mAP and Rank-1 among the considered approaches.

<table><tr><td>Method</td><td>mAP</td><td>Rank-1</td></tr><tr><td>BagDINO + LoRA</td><td>93.63</td><td>96.32</td></tr><tr><td>Yang et al. [9]</td><td>88.60</td><td>95.40</td></tr><tr><td>Mazzeo et al. [8]</td><td>80.92</td><td>85.79</td></tr></table>

TABLE III: Cross-domain evaluation on Market-1501 to assess the transferability of the proposed approach beyond the baggage domain.

## V. CONCLUSIONS

This work investigated baggage re-identification as an instance-level retrieval problem in multi-camera airport environments, leveraging foundation-model representations to match query images against a gallery of registered baggage views. A DINOv3 ViT backbone combined with a Torchreidstyle BNNeck head was evaluated under different adaptation strategies, including a fully frozen backbone, parameterefficient LoRA adaptation, and partial/full fine-tuning. On the MVB benchmark, LoRA-based adaptation provided the strongest overall results among the tested configurations, reaching 0.88 mAP and 0.87 Rank-1, while maintaining a stable training regime under limited training data.

The qualitative inspection of retrieval lists highlighted two practical aspects of the benchmark setting: the limited number of gallery images per identity in the test split makes purity at fixed top-k inherently conservative, and the remaining failures are typically associated with visually near-duplicate baggage items that produce high-similarity confusions. These observations suggest that future evaluations aligned with operational deployment could benefit from larger multi-view templates per item at registration time, as well as from additional mechanisms for handling near-duplicate instances (e.g., stronger fine-grained cues, re-ranking, or auxiliary supervision) without compromising retrieval efficiency.

## ACKNOWLEDGMENT

This work was partially supported by the following project: “ThermoNET” - Codice CUP B25H25002090007. TITOLO II CAPO 2 DEL REGOLAMENTO GENERALE “Aiuti ai programmi integrati promossi da MEDIE IMPRESE ai sensi dell’articolo 26 del Regolamento”; Patto territoriale ”Sistema universitario pugliese” – CUP F61B23000370006.

## REFERENCES

[1] SITA, “Sita baggage it insights 2024,” 2024.

[2] International Air Transport Association (IATA), Baggage Tracking – IATA Resolution 753 Implementation Guide. International Air Transport Association, issue 4.0 ed., Nov. 2023.

[3] Z. Zhang, D. Li, J. Wu, Y. Sun, and L. Zhang, “Mvb: A large-scale dataset for baggage re-identification and merged siamese networks,” in Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pp. 84–96, Springer, 2019.

[4] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jegou, P. Labatut, and P. Bojanowski, “DINOv3,” 2025.´

[5] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang, “Bag of tricks and a strong baseline for deep person re-identification,” 2019.

[6] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” 2021.

[7] R. Chen, H. Zhang, C. Li, and Y. Wang, “Lsdnn: Local-salient deep neural network for baggage re-identification with material discerning,” in 2020 Chinese Automation Congress (CAC), pp. 6344–6349, 2020.

[8] P. L. Mazzeo, C. Libetta, P. Spagnolo, and C. Distante, “A siamese neural network for non-invasive baggage re-identification,” Journal of Imaging, vol. 6, no. 11, p. 126, 2020.

[9] H. Yang, X. Chu, L. Zhang, Y. Sun, D. Li, and S. J. Maybank, “Quadnet: Quadruplet loss for multi-view learning in baggage re-identification,” Pattern Recognition, vol. 126, p. 108546, 2022.

[10] Y. Liu, H. Zhang, X. Yang, S. Zhao, and J. Zhang, “Cdr-carnet: Baggage re-identification based on cross-domain robust features and cameraaware re-ranking,” Computers & Graphics, p. 104377, 2025.

[11] Z. Xing, S. Zhu, T. Zhang, and Q. Luo, “Region to global vision transformer for baggage re-identification,” in 2023 42nd Chinese Control Conference (CCC), pp. 7433–7439, IEEE, 2023.

[12] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818– 2826, 2016.

[13] A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for person re-identification,” 2017.

[14] H. Wu, Z. Luo, D. Cao, D. Lin, S. Su, and S. Li, “Attention and multigranied feature learning for baggage re-identification,” in CCF Conference on Computer Supported Cooperative Work and Social Computing, pp. 460–472, Springer, 2021.

[15] Q. Zhao, H. Ma, R. Lu, Y. Chen, and D. Li, “Mvad-net: Learning viewaware and domain-invariant representation for baggage re-identification,” in Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pp. 142–153, Springer, 2021.

[16] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, “Scalable person re-identification: A benchmark,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2015.