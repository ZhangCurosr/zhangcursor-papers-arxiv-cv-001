# Compensating for Scarce Historical Images in Cross-Domain Cultural Heritage Retrieval Using Synthetic Aging

Marcin Iwanowski<sup>a,b,∗</sup>, Adam Mazgaj<sup>c</sup>, Ferdynand Górski<sup>c</sup>, Sabina Szymoniak<sup>d</sup>

<sup>a</sup>Inst.ofEngineering and Technology, Faculty ofPhysics, Astronomy and Informatics, Nicolaus Copernicus University, ul.Grudzi ˛adzka 5, 87-100 Toru´n, POLAND <sup>b</sup>Institute ofControl and Industrial Electronics, Warsaw University ofTechnology, ul.Koszykowa 75, 00-662 Warszawa, POLAND cClemens, ul Pawlikowskiego 10/2. 31-127 Krakow, POLAND Clemens, ul.Pawlikowskiego 10/2, 31-127 Krakow, POLAND

<sup>d</sup>Department of Computer Science, Czestochowa University of Technology, ul.Dabrowskiego 69, 42-201 Czestochowa, POLAND

## Abstract

Cultural heritage collections often contain contemporary and historical visual records of the same physical object, but linking<sup>2</sup> them is dificult because genuine historical images are scarce and difer from contemporary photographs. This study investigates<sup>0</sup> whether synthetically aged contemporary images can replace or complement missing historical training observations in bidirectional instance-level retrieval.<sup>p</sup>

A dataset of 1,077 identities, consisting of real old and contemporary images. Synthetic old-domain images were generated with a degradation-oriented aging pipeline. An EficientNetV2-M model trained with batch-hard triplet loss was evaluated across three dataset partitions with pairwise disjoint test identities and three training seeds.

Complete synthetic replacement reduced the bidirectional mean R@1 from 92.15% to 87.94% (paired efect: −4 21 percentage points), while increasing the synthetic pool from one to three variants produced no improvement. Under severe scarcity, syntheticV completion was more efective: at 10% genuine old-domain coverage, mean R@1 improved by 12.30 and 13.99 points relative to exposure-matched and full-budget real-only controls; at 30%, the corresponding gains were 4.67 and 5.47 points. The benefit. diminished as genuine coverage increased and was negligible at 90%. A prespecified 2-percentage-point acceptable-loss criterion was first satisfied at 70% genuine coverage. Adding synthetic observations at complete genuine coverage increased the mean R@1<sup>[</sup> to 94.05%.

Synthetic aging is therefore more efective for completing missing cross-domain supervision than for replacing genuine historical data. Broader identity coverage appears to be the primary benefit under severe scarcity, while a weaker augmentation efect may6 also be present.6

Keywords: cultural heritage, cross-domain image retrieval, synthetic aging, data scarcity, deep metric learning, historical images

## 1. Introduction

Cultural heritage collections often contain multiple visual records of the same physical object created at diferent times and under diferent conditions. A painting, sculpture, icon, or museum artifact may be represented by a contemporary photograph, an archival image, a scanned catalog entry, or a printed<sup>r</sup> reproduction. Such records are frequently dispersed across institutions, while diferences in acquisition technology, digitization, color reproduction, resolution, framing, viewpoint, and physical degradation hinder direct visual correspondence. Automatically linking historical and contemporary records could support collection integration, catalog enrichment, duplicate detection, provenance research, and the reconstruction of incomplete documentation.

This task can be formulated as a bidirectional cross-domain instance-level image retrieval. Given an image from one domain, the objective is to retrieve another representation of the same physical object from the opposite domain. Unlike generic similarity search, the target is not a semantically related object, but the same individual instance. Deep representations have become central to instance-level retrieval [1, 2], while cultural heritage benchmarks demonstrate substantial appearance variation among images of the same artwork or museum object [3, 4].

Cross-domain instance retrieval requires robustness to changes in viewpoint, scale, illumination, sharpness, color, background, framing, and image quality while preserving de tails that distinguish visually similar objects. Related general ization problems have been observed in cultural heritage image analysis [5] and are closely connected to domain adaptation and domain generalization [6, 7]. Deep metric learning provides a natural framework for this setting by constructing an embedding space in which images of the same object are close, and images of diferent objects are separated. Triplet-based objectives and hard-example mining have been widely used for instance recognition and retrieval [8, 9, 10]. Their efectiveness, however, depends on representative positive pairs. Ideally, each training identity should therefore be observed in both the historical and contemporary domains.

This requirement is dificult to satisfy in cultural heritage collections. Contemporary photographs can often be acquired relatively easily, whereas historical records are scarce, unevenly distributed, poorly digitized, or dificult to locate. Some objects have extensive archival documentation, while others have only one historical image or none. Constructing a complete paired training set may consequently be infeasible even when contemporary imagery is abundant. The resulting imbalance is particularly problematic for metric-learning approaches, because identities without observations from both domains cannot provide genuine cross-domain positive pairs.

Synthetic aging ofers a possible means of reducing this dependence. A contemporary photograph can be transformed into an identity-preserving old-domain representation using effects associated with historical photographs, catalog reproductions, scans, and degraded archival records. Related degradations have been studied in old photo restoration [11, 12]; here, they are deliberately introduced rather than removed to generate training observations. Synthetic aging can therefore be viewed not only as conventional data augmentation [13], but also as a mechanism for completing an incompletely observed visual domain.

Its usefulness is not self-evident. A synthetically aged image retains the viewpoint, composition, and object geometry of its contemporary source and may therefore reproduce only part of the genuine historical-to-contemporary domain gap. Genuine historical records may contain structural and acquisition diferences that cannot be reproduced by degradation-oriented transformations alone. Nevertheless, even imperfect synthetic observations may provide useful cross-domain supervision for identities lacking genuine historical counterparts.

The mechanism behind such an improvement is also ambiguous. Synthetic completion increases the number of identities represented in both domains, but it may simultaneously alter training exposure and introduce additional appearance variability. Consequently, an observed gain may arise from broader cross-domain identity coverage, from repeated exposure to a reduced genuine subset, or from a more general augmentation efect of synthetic old-domain observations. These mechanisms require separate controls because they lead to diferent practical conclusions about how synthetic data should be used when historical records are scarce.

Although synthetic imagery is widely used to enlarge training datasets, its role as a targeted substitute for missing identity–domain observations in cultural heritage retrieval remains underexplored. In particular, it is unclear how much genuine historical identity coverage is required before synthetic completion ceases to provide a meaningful benefit, whether multiple stochastic realizations of the same aging procedure improve synthetic-only training, and whether synthetic observations remain useful when genuine historical coverage is already complete. Addressing these questions require a controlled evaluation that varies genuine historical coverage while separating identity-coverage efects from training exposure and general augmentation.

## 2. Research aim

The aim of this study is to determine how synthetic aging can compensate for missing genuine historical observations in cross-domain cultural heritage retrieval and to identify the mechanism underlying any resulting improvement. We evaluate a degradation-based synthetic aging pipeline under complete replacement, partial synthetic completion, and fullcoverage real-plus-synthetic training.

The study addresses four research questions:

RQ1 To what extent can synthetically aged images replace genuine historical images in cross-domain retrieval training?

RQ2 Does increasing the number of independently generated synthetic variants improve retrieval when genuine historical observations are absent?

RQ3 How does retrieval performance change as genuine olddomain identity coverage varies and missing observations are replaced synthetically?

RQ4 Are the benefits of synthetic data explained primarily by broader cross-domain identity coverage, by repeated exposure to a reduced genuine subset, or by a more general augmentation efect when genuine coverage is already complete?

We hypothesize that synthetic aging cannot fully reproduce genuine historical variability, but can complement scarce genuine data by restoring missing cross-domain supervision, with its benefit diminishing as genuine historical coverage increases.

## 3. Materials and methods

## 3.1. Dataset and cross-domain retrieval scenario

The experiments use a cross-domain dataset of cultural heritage objects assembled from the Louvre Collections and the Lost Art Database [14, 15]. Contemporary images define the new domain, whereas original historical records and their corrected grayscale or sepia versions define the old domain. The corrected versions preserve the original acquisition, viewpoint, and object representation and are therefore treated as genuine rather than synthetic observations.

Only identities represented in both domains are included. The final dataset contains 1077 object identities. Splitting is performed at the identity level, so all images belonging to a physical object are assigned to the same subset within a given dataset partition. Three partitions are evaluated. Each contains 808 training identities, 108 validation identities, and 161 test identities, corresponding approximately to a 75%/10%/15% train–validation–test division.

The three partitions were constructed so that their test identity sets are pairwise disjoint. Consequently, the evaluation covers 483 distinct test identities across the three partitions. Training and validation identities are allowed to occur in more than one partition, provided that they do not overlap with the test set of the same partition. The partitions should therefore be regarded as three experimental realizations with distinct test identities rather than as statistically independent datasets.

Object-category information was used during partition construction to reduce diferences in category composition between subsets. Test and validation identities were selected using category-stratified allocation, while preserving the required subset sizes and the pairwise disjointness of the test sets. Category-specific performance is not used as a primary outcome of the study.

Synthetic images are generated exclusively from contemporary images in the training subset of each partition. Validation and test sets contain only genuine old- and new-domain records. Reported retrieval performance therefore measures generalization across the genuine old–new domain gap and is never evaluated using synthetic queries or gallery images.

In the scarcity experiments, genuine old-domain identity coverage denotes the proportion of training identities for which genuine old-domain observations are retained. In the real– synthetic completion conditions, identities without genuine old-domain observations receive synthetic old-domain counterparts, so that cross-domain training pairs remain available for the full training identity set. The construction of the corresponding real-only controls is described in Section 3.4.

## 3.2. Synthetic aging of contemporary images

Synthetic old-domain images are generated ofline from contemporary training photographs using a degradation-oriented aging procedure. The objective is not to reconstruct the authentic historical appearance of a photograph, but to create identitypreserving old-domain observations that reproduce selected visual characteristics of historical photographs, printed reproductions, and digitized archival material.

For a contemporary image $x _ { \mathrm { n e w } }$ with identity $y ,$ the synthetic representation is

$$
\tilde { x } _ { \mathrm { o l d } } = T _ { \mathrm { a g e } } \left( x _ { \mathrm { n e w } } ; \phi \right) , \qquad y \left( \tilde { x } _ { \mathrm { o l d } } \right) = y \left( x _ { \mathrm { n e w } } \right) ,\tag{1}
$$

where $\phi$ contains independently sampled transformation parameters. The fixed transformation sequence is

$$
\begin{array} { r l } & { T _ { \mathrm { a g e } } = T _ { \mathrm { s e p i a } } \circ T _ { \mathrm { l e a k } } \circ T _ { \mathrm { p a p e r } } \circ T _ { \mathrm { p e r s p } } \circ T _ { \gamma } } \\ & { \qquad \circ T _ { \mathrm { b l u r } } \circ T _ { \mathrm { s c r a t c h } } \circ T _ { \mathrm { v i g } } \circ T _ { \mathrm { n o i s e } } \circ T _ { \mathrm { f r a m e } } . } \end{array}\tag{2}
$$

The procedure introduces replicated borders, Gaussian and impulse noise, vignetting, scratches and dust artifacts, blur, tonal and perspective changes, paper texture, light artifacts, and sepia-to-grayscale conversion. It is inspired by degradations considered in old-photo restoration [11, 12], but deliberately introduces rather than removes such efects. Figure 1 compares a contemporary image, its genuine old-domain counterpart, and a synthetically aged variant.

Three synthetic variants, denoted OLDIFY1, OLDIFY2, and OLDIFY3, are generated using the same transformation pipeline and parameter ranges. They difer only in the independently sampled transformation parameters. Synthetic generation is performed separately for the training subset of each dataset partition. Consequently, when the same contemporary source image occurs in the training sets of more than one partition, it may receive a diferent synthetic realization in each partition.

![](images/4164023a2689c55b1a7a61aa1a3b4b16f199b93fa26c5cc3233a6db0016265ff.jpg)  
Figure 1: Aleksander Gierymski, Jewess with Oranges (Zydowka z pomaranczami), 1880–1881, National Museum in Warsaw. (a) Historical pre-´ 1939 photograph; (b) contemporary digital reproduction; (c) synthetically aged version of (b), generated by the authors. The painting is in the public domain. Contemporary digital reproduction: National Museum in Warsaw/Wikimedia Commons, public domain.

Generation is deterministic for a fixed partition, source image, and OLDIFY variant, allowing the complete synthetic dataset to be reproduced. Within a given partition, the pregenerated synthetic images remain fixed across all training seeds and experimental conditions that use the corresponding OLDIFY variant. This preserves paired comparisons between conditions while allowing the three dataset partitions to represent diferent realizations of both data composition and synthetic degradation.

Synthetic images are generated only for training data and are never introduced into validation or test subsets. No online augmentation is applied during model training; input images are only resized and normalized using ImageNet statistics. Full transformation details, processing order, and parameter ranges are provided in Supplementary Methods S1 and Table S1.

## 3.3. Retrieval model

The retrieval model uses an EficientNetV2-M backbone pretrained on ImageNet [16, 17]. Its classification layer is replaced by a linear projection producing a 1280-dimensional, $\ell _ { 2 }$ -normalized embedding:

$$
f _ { \theta } ( x ) = \frac { W g _ { \theta } ( x ) + b } { \vert \vert W g _ { \theta } ( x ) + b \vert \vert _ { 2 } } ,\tag{3}
$$

where $g _ { \theta } ( x )$ denotes the backbone representation. The projection is initialized with $W = I$ and $b = 0$ , so the initial embedding corresponds to the normalized pretrained representation.

Training uses batch-hard triplet loss [8, 9, 10]. For embeddings $z _ { i } = f _ { \theta } ( x _ { i } )$ , let $d _ { i j } = \lVert z _ { i } - z _ { j } \rVert _ { 2 }$ . The hardest positive and negative distances for anchor i are

$$
d _ { i } ^ { + } = \operatorname* { m a x } _ { \stackrel { j \neq i } { y _ { j } = y _ { i } } } d _ { i j } , \qquad d _ { i } ^ { - } = \operatorname* { m i n } _ { y _ { j } \neq y _ { i } } d _ { i j } .\tag{4}
$$

The loss is

$$
\mathcal { L } _ { \mathrm { t r i } } = \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { i \in \mathcal { B } } \operatorname* { m a x } \left( 0 , d _ { i } ^ { + } - d _ { i } ^ { - } + m \right) ,\tag{5}
$$

Table 1: Main model and optimization parameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Backbone</td><td>EfficientNetV2-M, ImageNet pretrained</td></tr><tr><td>Input resolution</td><td>384× 384</td></tr><tr><td>Embedding dimension</td><td>1280</td></tr><tr><td>Projection layer Embedding normalization</td><td>Linear, identity initialization  $\ell _ { 2 }$ </td></tr><tr><td>Unfrozen backbone blocks</td><td>2 final blocks</td></tr><tr><td>Backbone BatchNorm</td><td>Frozen</td></tr><tr><td>Loss function</td><td>Batch-hard triplet loss</td></tr><tr><td>Triplet margin</td><td>0.3</td></tr><tr><td>Identities per batch</td><td>8</td></tr><tr><td>Images per identity</td><td>1 old and 1 new</td></tr><tr><td>Batch size</td><td></td></tr><tr><td>Maximum epochs</td><td>16</td></tr><tr><td></td><td>30</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Backbone learning rate</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Projection learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine annealing</td></tr><tr><td>Early-stopping criterion</td><td>Validation mean R@1</td></tr><tr><td>Early-stopping patience</td><td>7 epochs</td></tr><tr><td>No. of training seeds</td><td> $^ 3$ </td></tr><tr><td>Online augmentation</td><td>None</td></tr></table>

where B denotes the minibatch and $m = 0 . 3$

Each batch contains eight identities and 16 images. For every selected identity, one old-domain and one new-domain image are sampled randomly from the corresponding image pools. Consequently, each identity contributes exactly one cross-domain positive pair per batch. When multiple olddomain variants are available, they enlarge the sampling pool but do not increase the number of images contributed by that identity in a single optimization step.

Input images are resized to 384 × 384 pixels and normalized using ImageNet statistics. No online data augmentation is applied during training.

The final two backbone blocks and the projection layer are trainable, while all remaining backbone parameters are frozen. Backbone batch-normalization layers remain in evaluation mode and their parameters are not updated. AdamW is used with learning rates of $1 0 ^ { - 5 }$ for the unfrozen backbone and $1 0 ^ { - 3 }$ for the projection layer, weight decay $1 0 ^ { - 4 }$ , and cosine annealing.

Training lasts for at most 30 epochs. Checkpoints are selected using bidirectional validation mean R@1, with an earlystopping patience of seven epochs and a minimum improvement of $1 0 ^ { - 4 }$ . All conditions are trained with seeds 99, 100, and 101. Model architecture, optimization parameters, sampling structure, and stopping criteria are kept fixed across experimental conditions. The number of batches per epoch is determined from the available identity set according to the experimental protocol described in Section 3.4.

## 3.4. Experimental design

Three experiments evaluate synthetic old-domain observations in complementary settings: complete replacement and full-coverage augmentation, synthetic completion under controlled scarcity of genuine historical observations, and real-only controls designed to separate the efect of broader identity coverage from increased training exposure.

All experiments use the same identity-based batch structure described in Section 3.3. To avoid defining an epoch by an arbitrary fixed number of batches, the training budget is normalized with respect to the number of identities available to the sampler. Let N denote the number of training identities, $P \ = \ 8$ the number of identities sampled per batch, and B the number of batches per epoch. Assuming independent batch draws, the probability that a particular identity is sampled at least once during an epoch is

$$
C ( N , B ) = 1 - \left( 1 - \frac { P } { N } \right) ^ { B } .\tag{6}
$$

The full training schedule is defined by a target identity coverage of $C _ { 0 } = 0 . 9 9$ , giving

$$
B _ { \mathrm { f u l l } } = \left\lceil \frac { \log ( 1 - C _ { 0 } ) } { \log \left( 1 - P / N \right) } \right\rceil .\tag{7}
$$

For $N ~ = ~ 8 0 8$ training identities, this yields $B _ { \mathrm { f u l l } } ~ = ~ 4 6 3$ batches per epoch. This schedule is used whenever the full identity set is retained. The model architecture, optimizer, batch composition, early stopping, and all other training settings remain unchanged across conditions.

## 3.4.1. Experiment 1: complete replacement and full-coverage augmentation

Experiment 1 addresses RQ1 and RQ2. The Real+Synth condition additionally contributes to RQ4 by evaluating a general augmentation efect under complete genuine historical coverage. All four conditions use the full set of 808 training identities and $B _ { \mathrm { f u l l } } = 4 6 3$ batches per epoch. They difer only in the source of the old-domain observation sampled for each identity.

The Real condition (R100\_S0) uses genuine old-domain images together with contemporary images. In Synth-1 (R0\_S100), genuine old-domain images are completely replaced by OLDIFY1 variants. In Synth-3, the old-domain im age is sampled from the combined pool of OLDIFY1, OLD-IFY2, and OLDIFY3 variants. Because each selected identity still contributes only one old-domain image per batch, Synth-1 and Synth-3 difer in synthetic variability rather than in the number of optimization steps or images sampled per identity.

A fourth condition, Real+Synth, retains genuine old-domain observations for every training identity and additionally makes OLDIFY1 observations available. For each selected identity, the old-domain source is first chosen with equal probability from the genuine and synthetic pools, $P ( { \mathrm { r e a l } } ) = P ( { \mathrm { s y n t h e t i c } } ) =$ 0 5, after which an image is sampled from the selected source. This source-balanced sampling prevents diferences in pool cardinality from determining the relative frequency of genuine and synthetic observations. Comparison with Real tests whether synthetic aging provides useful augmentation even when no historical observations are missing.

## 3.4.2. Experiment 2: controlled scarcity and synthetic completion

Experiment 2 addresses RQ3 by varying the proportion of training identities for which genuine old-domain observations are retained. Nine intermediate genuine-coverage levels are evaluated,

$$
r \in \{ 1 0 , 2 0 , 3 0 , 4 0 , 5 0 , 6 0 , 7 0 , 8 0 , 9 0 \} \% .
$$

The corresponding mixed condition is denoted $\mathsf { R } r \_ \mathsf { S } ( 1 0 0 - r )$ For each partition, the number of genuine identities is

$$
N _ { \mathrm { r e a l } } = \mathrm { r o u n d } \Big ( \frac { r } { 1 0 0 } N \Big ) ,\tag{8}
$$

and all remaining training identities receive OLDIFY1 olddomain observations. Each identity is assigned exclusively to one old-domain source: all available genuine old-domain observations are used for identities in the genuine subset, whereas identities outside this subset use their OLDIFY1 counterparts. Contemporary images remain available for all 808 identities. Thus, every mixed condition preserves the full cross-domain training identity set.

The genuine subsets are nested,

$$
R 1 0 \subset R 2 0 \subset \cdots \subset R 9 0 \subset R 1 0 0 ,
$$

and are constructed separately for each dataset partition. The ordering is category-stratified so that successive prefixes approximately preserve the category composition of the full training set. The resulting source assignment is fixed across all three training seeds within a partition. Consequently, diferences between model seeds do not alter which identities receive genuine or synthetic old-domain observations.

All mixed conditions retain $N = 8 0 8$ training identities and use $B _ { \mathrm { f u l l } } = 4 6 3$ batches per epoch. The endpoints of the coverage curve are reused from Experiment 1: R0\_S100 corresponds to Synth-1, whereas R100\_S0 corresponds to Real. The complete coverage series therefore spans genuine old-domain identity coverage from 0% to 100% in 10-percentage-point increments.

## 3.4.3. Experiment 3: real-only controls for identity coverage and training exposure

Experiment 3 addresses RQ4 using real-only controls at

$$
r \in \{ 1 0 , 3 0 , 5 0 , 7 0 , 9 0 \} \% .
$$

At each coverage level, the corresponding mixed condition and both real-only controls use exactly the same genuine identity subset. Unlike the mixed condition, however, the real-only controls remove all remaining identities entirely from training, including their contemporary images. Hence, the A and B variants contain only $N _ { \mathrm { r e a l } }$ identities represented by genuine oldand new-domain observations.

The A variants (Rr\_A) use an exposure-matched schedule obtained by scaling the full-set budget according to the number of retained identities:

Table 2: Summary of the experimental training conditions. N = 808 denotes the full training identity set, $N _ { \mathrm { r e a l } } = \mathrm { r o u n d } ( r N / 1 0 0 ) .$ , and $B _ { \mathrm { f u l l } } = 4 6 3 .$
<table><tr><td>Exp.</td><td>Condition</td><td>Genuine coverage</td><td>Training IDs</td><td>Batches/epoch</td></tr><tr><td>1</td><td>REAL (R100_SO)</td><td>100%</td><td>N</td><td> $\overline { { B _ { \mathrm { f u l l } } } }$ </td></tr><tr><td>1</td><td>SYNTH-1 (RO_S100)</td><td>0%</td><td>N</td><td> $B _ { \mathrm { f u l l } }$ </td></tr><tr><td>1</td><td>SYNTH-3</td><td>0%</td><td>N</td><td> $B _ { \mathrm { f u l l } }$ </td></tr><tr><td>1</td><td>REAL+SYNTH</td><td>100% genuine + synthetic</td><td>N</td><td> $B _ { \mathrm { f u l l } }$ </td></tr><tr><td>2</td><td> $\mathtt { R } r \_ { \mathbf { S } } ( 1 0 0 - r )$ </td><td> $r = 1 0 , 2 0 , \ldots , 9 0 \%$ </td><td>N</td><td> $B _ { \mathrm { f u l l } }$ </td></tr><tr><td>3</td><td>Rr_A</td><td>r = 10, 30, 50, 70,90%</td><td> $N _ { \mathrm { r e a l } }$ </td><td> $B _ { A }$ </td></tr><tr><td>3</td><td>Rr_B</td><td> $r = 1 0 , 3 0 , 5 0 , 7 0 , 9 0 \%$ </td><td> $N _ { \mathrm { r e a l } }$ </td><td> $B _ { \mathrm { f u l l } }$ </td></tr></table>

$$
B _ { A } = \mathrm { r o u n d } \left( B _ { \mathrm { f u l l } } \frac { N _ { \mathrm { r e a l } } } { N } \right) .\tag{9}
$$

This preserves, to rounding accuracy, the expected number of selections of each retained identity per epoch relative to the full-set schedule. For 10%, 30%, 50%, 70%, and 90% genuine coverage, the resulting schedules contain 46, 139, 232, 324, and 417 batches per epoch, respectively.

The B variants (Rr\_B) use the same reduced genuine identity sets but retain the full $B _ { \mathrm { f u l l } } = 4 6 3$ -batch schedule. They therefore match the corresponding mixed conditions in the number of optimization steps and image selections per epoch while repeatedly sampling a smaller set of genuine identities.

Three paired comparisons are used to distinguish the evaluated efects. Comparing a mixed condition with its A con trol measures the benefit of synthetic completion relative to real-only training with matched expected per-identity exposure. Comparing B with A estimates the efect of increasing repeated exposure to the same reduced genuine subset. Finally, comparing the mixed condition with B holds the number of batches and per-epoch image selections constant and tests whether retaining broader cross-domain identity coverage provides an advantage over repeated sampling of the reduced genuine subset.

Across the three dataset partitions and three training seeds, the protocol required 207 separate model training runs: 36 for Experiment 1 and 171 for Experiments 2–3. The R0\_S100 and R100\_S0 endpoints of the scarcity series reuse the corresponding Experiment 1 models and were not retrained.

## 3.5. Retrieval evaluation

All models are evaluated using the same bidirectional instance-level retrieval protocol [2]. For each query, similarity is computed against every test image from the opposite domain. Because the embeddings are $\ell _ { 2 } .$ -normalized, similarity is defined as

$$
s ( q , g ) = f _ { \boldsymbol { \theta } } ( q ) ^ { \top } f _ { \boldsymbol { \theta } } ( g ) ,\tag{10}
$$

which is equivalent to cosine similarity.

In old-to-new retrieval, genuine old-domain images are used as queries and contemporary images form the gallery. In newto-old retrieval, contemporary images are used as queries and genuine old-domain images form the gallery. Only genuine observations are used for evaluation; synthetic images are never included as queries or gallery items.

Because an identity may be represented by multiple gallery images, retrieval at rank k is considered successful when at least one image of the correct identity occurs among the first k retrieved items:

$$
{ \mathrm R @ 6 } = \frac { 1 } { | Q | } \sum _ { q \in Q } \mathbf { 1 } \left[ \exists g \in \mathrm { T o p } _ { k } ( q ) : y _ { g } = y _ { q } \right] ,\tag{11}
$$

where Q denotes the query set. Thus, R@k is query-weighted: every query contributes equally within a retrieval direction, irrespective of how many images are available for its identity.

Recall is reported for $k \in \{ 1 , 5 , 1 0 \}$ . The primary evaluation metric is bidirectional mean R@1,

$$
\mathrm { m e a n } \mathrm { \mathrm { R } } \ @ 1 = \frac { \mathrm { R } @ 1 _ { \mathrm { o l d }  \mathrm { n e w } } + \mathrm { R } @ 1 _ { \mathrm { n e w }  \mathrm { o l d } } } { 2 } .\tag{12}
$$

This metric assigns equal weight to the two retrieval directions and is also used for checkpoint selection on the validation subset. R@5 and R@10 provide complementary measures of whether the correct identity remains within a short candidate list suitable for expert inspection.

Within each dataset partition, all compared models are evaluated using exactly the same genuine queries and galleries. This permits paired comparisons between experimental conditions at the query level while ensuring that diferences in retrieval performance cannot result from changes in the evaluation set.

## 3.6. Statistical analysis

Each experimental condition is evaluated in three dataset partitions and with three training seeds, yielding nine split–seed results. For descriptive reporting, retrieval performance is summarized as the mean and sample standard deviation across these nine runs. Because the three seeds within a partition share the same test identities, and because training and validation identities may overlap between partitions, the nine results are not treated as independent observations for confidence-interval estimation.

Comparisons between experimental conditions are paired within partition and training seed. Since all models compared within a partition are evaluated using identical queries and galleries, paired efects can be computed directly at the query level. For a metric M, the efect of variant B relative to reference A is defined as

$$
\Delta M = M _ { B } - M _ { A } .\tag{13}
$$

Confidence intervals for paired diferences in the primary metric, bidirectional mean R@1, are estimated using 10,000 paired bootstrap resamples at the test-identity level. Within each dataset partition, test identities are sampled with replacement. For identities represented by more than one query in a retrieval direction, all queries belonging to the selected identity are retained together. Training seeds are independently resampled with replacement within the same partition.

For every bootstrap replicate, query-weighted R@1 diferences are calculated separately for old-to-new and new-to-old retrieval, averaged across the two directions, and then averaged across the three dataset partitions with equal partition weight.

The partitions themselves are not resampled. The 95% confidence interval is given by the 2.5th and 97.5th percentiles of the bootstrap distribution.

This procedure preserves the query-weighted definition of R@1 from Section 3.5 while using object identity as the resampling unit. The great majority of test identities are represented by one old-domain and one new-domain image; for the small number of identities represented by multiple images, retaining all associated queries together prevents pseudo-replication. Resampling training seeds additionally accounts for variability due to model training.

For Experiment 1, the principal paired comparisons are

$$
\mathrm { S Y N T H - 1 - R e A L } , \qquad \mathrm { S Y N T H - 3 - R e A L } , \qquad \mathrm { S Y N T H - 3 - S Y N T H - 1 } ,
$$

together with

$$
\mathrm { R E A L + S Y N T H - R E A L } ,
$$

which evaluates whether synthetic old-domain observations provide an augmentation benefit when genuine historical coverage is already complete.

For the scarcity controls in Experiment 3, three paired effects are evaluated at genuine old-domain coverage levels $r \in$ {10 30 50 70 90}%:

$$
\Delta _ { \mathrm { c o m p l e t i o n } , A } = M _ { \mathrm { R } r _ { - } \mathrm { S } ( 1 0 0 - r ) } - M _ { \mathrm { R } r _ { - } \mathrm { A } } ,\tag{14}
$$

$$
\Delta _ { \mathrm { e x p o s u r e } } = M _ { \mathrm { R } r \_ { \mathrm { B } } } - M _ { \mathrm { R } r \_ { \mathrm { A } } , }\tag{15}
$$

$$
\Delta _ { \mathrm { c o m p l e t i o n } , B } = M _ { \mathrm { R } r _ { - } \mathrm { S } ( 1 0 0 - r ) } - M _ { \mathrm { R } r _ { - } \mathrm { B } } .\tag{16}
$$

The first comparison measures the benefit of synthetic completion relative to real-only training with matched expected peridentity exposure. The second measures the efect of increasing repeated exposure to the same reduced genuine identity subset. The third compares mixed and real-only training under the same full per-epoch optimization budget.

Mixed completion conditions are additionally compared with the complete genuine-data reference R100\_S0. Before inspection of the results from the revised experimental protocol, an acceptable-loss margin of

$$
\delta = 0 . 0 2\tag{17}
$$

was specified, corresponding to a maximum reduction of two percentage points in bidirectional mean R@1. For a mixed condition,

$$
\Delta _ { r } = \mathrm { m e a n } \mathrm { R @ 1 } _ { \mathrm { R } r _ { - } \mathrm { S ( 1 0 0 - } r ) } - \mathrm { m e a n } \mathrm { R @ 1 } _ { \mathrm { R 1 0 0 \_ S 0 } } ,\tag{18}
$$

the descriptive acceptable-loss criterion is satisfied when

$$
\Delta _ { r , \mathrm { l o w } } > - \delta ,\tag{19}
$$

where $\Delta _ { r , \mathrm { l o w } }$ is the lower bound of the 95% bootstrap confidence interval. This analysis is used to identify the lowest evaluated genuine-coverage level whose uncertainty interval remains within the prespecified two-percentage-point margin. It is not interpreted as a formal non-inferiority test.

Table 3: Retrieval performance in Experiment 1, reported as mean ± sample standard deviation over three dataset partitions and three training seeds.
<table><tr><td>Variant</td><td>Old→New R@1</td><td>New→Old R@1</td><td>Mean R@1</td></tr><tr><td>REAL</td><td> $9 2 . 2 1 \pm 3 . 4 4 \%$ </td><td> $\overline { { 9 2 . 1 0 \pm 3 . 7 3 \% } }$ </td><td> $9 2 . 1 5 \pm 3 . 5 2 \%$ </td></tr><tr><td>SYNTH-1</td><td> $8 8 . 1 9 \pm 2 . 5 0 \%$ </td><td> $8 7 . 6 9 \pm 3 . 1 2 \%$ </td><td> $8 7 . 9 4 \pm 2 . 6 8 \%$ </td></tr><tr><td>SYNTH-3</td><td> $8 8 . 0 6 \pm 3 . 4 5 \%$ </td><td> $8 7 . 7 6 \pm 3 . 2 9 \%$ </td><td> $8 7 . 9 1 \pm 3 . 2 9 \%$ </td></tr><tr><td>REAL+SYNTH</td><td> $9 4 . 1 9 \pm 2 . 5 2 \%$ </td><td> $9 3 . 9 1 \pm 3 . 0 8 \%$ </td><td> $9 4 . 0 5 \pm 2 . 7 0 \%$ </td></tr></table>

The three dataset partitions have pairwise disjoint test identities but originate from the same assembled collection and may share training or validation identities. Confidence intervals therefore quantify uncertainty associated with the evaluated test identities and training seeds rather than population-level uncertainty across independent datasets. Partition-level results are additionally examined to assess whether the principal efects are consistent across diferent test-set compositions.

## 4. Results

## 4.1. Complete replacement and full-coverage augmentation

Table 3 summarizes Experiment 1, including complete synthetic replacement, increased synthetic variability, and synthetic augmentation under complete genuine old-domain coverage.

Complete replacement of genuine old-domain observations with OLDIFY1 reduced bidirectional mean R@1 from 92.15% for Real to 87.94% for Synth-1. The paired efect was −4 21 percentage points (pp), with a 95% bootstrap confidence interval (CI) of [−6 19 −2 23] pp. The reduction was observed in both retrieval directions, showing that synthetic aging did not reproduce the full retrieval value of genuine historical observations.

Increasing the synthetic sampling pool from one to three independently generated OLDIFY variants produced essentially no change in rank-1 retrieval. Synth-3 achieved 87.91% mean R@1, and its paired diference relative to Synth-1 was −0 03 pp (95% CI: [−2 39 +2 43] pp). Relative to Real, the diference for Synth-3 was −4 24 pp (95% CI: [−6 93 −1 71] pp). Increasing the number of stochastic realizations of the same degradation pipeline therefore did not reduce the gap between synthetic and genuine old-domain training observations.

Synthetic-only training nevertheless retained strong retrieval performance beyond rank 1. Synth-1 achieved a bidirectional mean R@10 of 98.43%, compared with 99.23% for Real. Thus, complete synthetic replacement usually placed the correct identity within a short candidate list even though it was less efective at ranking the correct identity first.

The Real+Synth condition produced the highest mean R@1, 94.05%, compared with 92.15% for Real. The paired diference was +1 90 pp, with a 95% CI of [−0 10 +4 09] pp. The interval slightly crossed zero, so the result does not provide clear evidence of an augmentation benefit under complete genuine old-domain coverage. Nevertheless, the positive point estimate in both retrieval directions suggests that synthetic olddomain observations may provide additional useful appearance variability even when genuine historical observations are available for every training identity.

## 4.2. Efect of genuine old-domain coverage and synthetic completion

Figure 2 summarizes retrieval performance under progressively increasing genuine old-domain identity coverage. Panel (a) shows the complete synthetic-completion series from 0% to 100% genuine coverage in 10-percentage-point increments, together with the real-only A and B controls evaluated at 10%, 30%, 50%, 70%, and 90%. Panel (b) shows the corresponding paired efects; these comparisons are analyzed in detail in Section 4.3. Bidirectional mean R@1 values for all scarcity and real-only control conditions are provided in Supplementary Table S2.

Across the mixed real–synthetic series, retrieval performance generally increased as genuine old-domain coverage increased, although the relationship was not strictly monotonic. Complete synthetic replacement (R0\_S100) achieved 87.94% mean R@1. Introducing 10% genuine coverage increased mean R@1 to 89.44%, while the 30% and 50% conditions reached 89.92% and 91.06%, respectively. Performance approached the complete-genuine-data reference at higher coverage levels: R70\_S30 achieved 92.18%, R80\_S20 92.41%, and R90\_S10 91.96%, compared with 92.15% for R100\_S0. Thus, the mixed conditions from 70% to 90% genuine coverage remained within approximately 0.3 percentage points of the complete-genuinedata mean.

The advantage of retaining the full training identity set was most pronounced when genuine historical observations were highly scarce. At 10% genuine coverage, R10\_S90 achieved 89.44% mean R@1, whereas the corresponding real-only A and B controls achieved 77.14% and 75.45%, respectively. At 30% coverage, the mixed condition achieved 89.92%, compared with 85.26% for R30\_A and 84.45% for R30\_B. The separation became smaller as genuine coverage increased. At 50%, the respective values were 91.06%, 89.41%, and 88.79%, and at 70% they were 92.18%, 90.69%, and 89.89%. By 90% genuine coverage, the three conditions converged, with mean R@1 values of 91.96% for the mixed condition, 92.18% for A, and 92.04% for B.

The overall pattern therefore indicates a diminishing performance advantage of synthetic completion as genuine olddomain coverage increases. The largest separation from the real-only controls occurs under severe historical-data scarcity, whereas little diference remains when genuine observations are available for nearly all training identities. The magnitude and uncertainty of these paired efects are examined next.

## 4.3. Paired efects and acceptable-loss analysis

Figure 2(b) summarizes the paired efects of synthetic completion and increased repeated exposure. Numerical efect estimates and 95% bootstrap confidence intervals are reported in Table 4.

![](images/a9588b1dd9f09c2b25816f8f6ecf1faf1aaa805e6fe460784ddc0021c0a36fee.jpg)

![](images/fffd54d7f3617f7456d6fa73ee593d19bc20b2bdd04ca56b7892cdf198d6bcf2.jpg)  
Figure 2: Efect of genuine old-domain identity coverage on bidirectional mean R@1. (a) Retrieval performance for the real–synthetic completion series and the real-only A and B controls. Mixed conditions retain all training identities and synthetically complete those without genuine old-domain observations. A controls use the reduced genuine identity set with matched expected per-identity exposure, whereas B controls use the same reduced identity set with the full per-epoch optimization budget. Error bars in panel (a) represent sample standard deviations across three dataset partitions and three training seeds. (b) Paired efects of synthetic completion and increased repeated exposure. Error bars in panel (b) represent 95% confidence intervals obtained using the paired identity-level bootstrap described in Section 3.6.

Table 4: Paired diferences in bidirectional mean R@1 for the scarcity controls. Positive values favor the first variant. Confidence intervals were estimated using the paired identity-level bootstrap with seed resampling described in Section 3.6.
<table><tr><td>Comparison type</td><td>Coverage</td><td>Mean difference</td><td>95% CI</td></tr><tr><td rowspan="5">Completion vs. A</td><td>10%</td><td>+12.30 pp</td><td>[+9.27, +15.40] pp</td></tr><tr><td>30%</td><td>+4.67 pp</td><td>[+2.11, +7.28] pp</td></tr><tr><td>50%</td><td>+1.65 pp</td><td>[−0.06, +3.45] pp</td></tr><tr><td>70%</td><td>+1.48 pp</td><td>[−0.29, +3.29] pp</td></tr><tr><td>90%</td><td>-0.22 pp</td><td>[-1.59, +1.19] pp</td></tr><tr><td rowspan="5">Additional exposure</td><td>10%</td><td>-1.68 pp</td><td>[-3.88, +0.47] pp</td></tr><tr><td>30%</td><td>-0.81 pp</td><td>[-3.29, +1.66] pp</td></tr><tr><td>50%</td><td>-0.62 pp</td><td>[-3.02, +1.67] pp</td></tr><tr><td>70%</td><td>-0.80 pp</td><td>[-2.51, +0.87] pp</td></tr><tr><td>90%</td><td>-0.14 pp</td><td>[-1.57, +1.32] pp</td></tr><tr><td rowspan="5">Completion vs. B</td><td>10%</td><td>+13.99 pp</td><td>[+11.23, +16.82] pp</td></tr><tr><td>30%</td><td>+5.47 pp</td><td>[+3.45, +7.60] pp</td></tr><tr><td>50%</td><td>+2.27 pp</td><td>[+0.06, +4.49] pp</td></tr><tr><td>70%</td><td>+2.29 pp</td><td>[+0.37, +4.36] pp</td></tr><tr><td>90%</td><td>-0.08 pp</td><td>[-1.60, +1.41] pp</td></tr></table>

Synthetic completion provided its largest advantage under severe historical-data scarcity. At 10% genuine olddomain coverage, the mixed condition exceeded the exposurematched A control by 12.30 pp (95% CI: [+9 27 +15 40] pp) and the full-budget B control by 13.99 pp (95% CI: [+11 23 +16 82] pp). At 30% coverage, the corresponding gains remained substantial, at 4.67 pp and 5.47 pp, and both confidence intervals remained above zero.

The completion advantage decreased as genuine coverage increased. At 50%, the mixed condition exceeded A by 1.65 pp, with the confidence interval narrowly including zero, whereas its 2.27 pp advantage over B had a lower confidence bound just above zero. A similar pattern was observed at 70% coverage: the mixed–A efect was +1 48 pp with a confidence interval including zero, while the mixed–B efect was +2 29 pp (95% CI: [+0 37 +4 36] pp). At 90%, the diferences between the mixed and real-only variants were close to zero.

In contrast, increasing the training budget for the reduced real-only identity sets did not reproduce the completion efect.

All B–A point estimates were small and negative, ranging from −1 68 to −0 14 pp, and every confidence interval included zero. Thus, the large gains observed at low genuine coverage cannot be explained by additional optimization steps or repeated exposure to the same reduced genuine identity subset.

The mixed completion series was also compared with the complete genuine-data reference R100\_S0 using the prespecified two-percentage-point acceptable-loss margin. At 50% genuine coverage, the paired efect relative to R100\_S0 was −1 09 pp, but the lower confidence bound reached −2 68 pp. At 60%, the efect was −0 56 pp with a 95% CI of [−2 19 +1 15] pp. At 70%, the efect was +0 03 pp and the 95% CI was [−1 78 +1 88] pp. Consequently, 70% genuine old-domain identity coverage was the lowest evaluated level whose 95% bootstrap confidence interval remained within the prespecified 2-pp acceptable-loss margin relative to complete genuine coverage. The 80% and 90% conditions also remained within this margin.

The complete acceptable-loss curve is reported in Supplementary Figure S2. This analysis is descriptive and is not interpreted as a formal non-inferiority test.

## 4.4. Variability across dataset partitions

Absolute retrieval performance varied across the three dataset partitions, reflecting diferences in test-set composition. Nevertheless, the principal scarcity-related pattern was reproduced across the pairwise disjoint test sets.

The strongest consistency was observed at low genuine historical coverage. For the mixed condition relative to the exposure-matched A control, the split-level gains at 10% coverage were +10 09, +14 63, and +12 19 pp for Splits 1–3, respectively. At 30% coverage, the corresponding gains were +6 29, +4 36, and +3 35 pp. Thus, synthetic completion provided a clear advantage in every partition when genuine old-domain observations were highly scarce.

The efect became smaller at higher coverage levels. At 50%, the mixed–A diferences were +0 87, +1 30, and +2 79 pp, and at 70% they were +0 76, +2 06, and +1 63 pp. By 90% coverage, the completion efect was small and its direction was no longer consistent across partitions.

These results show that dataset composition afects both absolute retrieval dificulty and the magnitude of the syntheticcompletion efect. However, the decreasing benefit of synthetic completion as genuine historical coverage increases is reproduced across all three test partitions. Detailed partition-level results and the corresponding split-specific completion curves are provided in Supplementary Section S2.

## 5. Discussion

The results distinguish three roles of synthetic old-domain observations: replacement of genuine historical data, completion of missing cross-domain identity observations, and augmentation when genuine historical coverage is already complete. These roles produce diferent efects. Complete synthetic replacement remained inferior to genuine historical training data, whereas synthetic completion provided large gains when genuine old-domain observations were scarce. Increasing repeated exposure to the same reduced genuine identity subsets did not reproduce these gains. At complete genuine coverage, however, adding synthetic observations produced a smaller positive efect whose confidence interval narrowly included zero. Taken together, the results indicate that broader cross-domain training identity coverage is a primary contributor to the benefit of synthetic data under severe scarcity, while a weaker general augmentation efect may also be present.

## 5.1. Complete replacement and synthetic diversity

Experiment 1 addresses RQ1 by showing that synthetically aged contemporary images are not equivalent to genuine historical observations. Complete replacement with OLDIFY1 reduced bidirectional mean R@1 by 4.21 percentage points, from 92.15% to 87.94%, with a 95% bootstrap confidence interval of [−6 19 −2 23] pp. The corresponding reduction was observed in both retrieval directions.

This gap is consistent with the limitations of a degradationoriented transformation. OLDIFY modifies noise, blur, scratches, tone, perspective, paper appearance, light artifacts, and other low-level characteristics, but the resulting image is still derived from the contemporary source. It therefore largely preserves its viewpoint, composition, object geometry, and scene structure. Genuine historical records may additionally difer in acquisition viewpoint, crop, framing, background, reproduction process, physical condition, and digitization. Such diferences cannot be reproduced fully by applying appearance degradations to a contemporary photograph.

Synthetic-only training nevertheless remained informative. Although its rank-1 performance was lower than that obtained with genuine historical observations, Synth-1 achieved a bidirectional mean R@10 of 98.43%, compared with 99.23% for Real. Synthetic aging therefore often placed the correct object within a short candidate list even when it did not provide suficient cross-domain variability to rank the correct identity first. This distinction is relevant for cultural heritage workflows in which retrieval is used to generate candidates for subsequent expert verification rather than to make fully automatic identity decisions.

Experiment 1 also addresses RQ2. Increasing the old-domain sampling pool from one to three independently generated OLD-IFY variants changed mean R@1 by only −0 03 pp, with a 95% confidence interval of [−2 39 +2 43] pp. Thus, additional stochastic realizations of the same degradation pipeline did not reduce the gap to genuine historical data. Increasing synthetic multiplicity alone appears less important than introducing forms of variability not present in the current transformation family.

Future improvements to synthetic replacement may therefore require methods that alter more structural properties of the observation, including viewpoint, framing, background, acquisition style, or visible object condition, rather than generating additional realizations of similar low-level degradations.

## 5.2. Synthetic completion under historical-data scarcity

Experiments 2 and 3 address RQ3 and the main identitycoverage and training-exposure components of RQ4. The mixed completion series showed an overall increase in retrieval performance as genuine old-domain coverage increased, although the relationship was not strictly monotonic. The benefit of synthetic completion was largest when genuine historical observations were scarce and diminished substantially as genuine coverage approached completeness.

The strongest efects occurred at 10% genuine coverage. The mixed condition exceeded the exposure-matched A control by 12.30 pp (95% CI: [+9 27 +15 40] pp) and the full-budget B control by 13.99 pp (95% CI: [+11 23 +16 82] pp). At 30% coverage, the corresponding gains were still 4.67 and 5.47 pp, with both confidence intervals remaining above zero. Synthetic completion therefore provided a substantial advantage when most identities would otherwise be absent from cross-domain training.

The efect became progressively smaller at higher genuine coverage. At 50% and 70%, the mixed conditions exceeded the A controls by 1.65 and 1.48 pp, respectively, but both confidence intervals included zero. Relative to the B controls, the corresponding efects were 2.27 and 2.29 pp, with lower confidence bounds just above zero. At 90% coverage, the mixed, A, and B conditions converged and the paired efects were close to zero. The results therefore support a diminishing-return interpretation: synthetic completion is most valuable when genuine historical observations are severely limited and contributes progressively less as the missing portion of the old domain becomes small.

The B controls help distinguish this completion efect from additional optimization. At every evaluated coverage level, the B–A point estimate was small and negative, and every confidence interval included zero. Increasing the number of opti mization steps and repeatedly sampling the same reduced genuine identity set therefore did not reproduce the gain obtained by retaining the full training identity set through synthetic completion. The result should not be interpreted as evidence that repeated exposure is detrimental; rather, there is no indication that additional exposure alone explains the completion advantage.

The acceptable-loss analysis provides a complementary practical perspective. At 50% and 60% genuine coverage, the point estimates were already close to the complete-genuine-data reference, but the lower confidence bounds extended beyond the prespecified two-percentage-point margin. At 70% coverage, the paired diference relative to R100\_S0 was +0 03 pp with a 95% CI of [−1 78 +1 88] pp. Thus, 70% was the lowest evaluated genuine coverage level whose hierarchical confidence interval remained within the prespecified acceptable-loss margin. This value should be interpreted as an empirical result for the present dataset and protocol rather than as a universal requirement for cultural heritage retrieval.

## 5.3. Identity coverage and augmentation efects

The control experiments support broader training identity coverage as an important mechanism behind synthetic completion, particularly under severe scarcity. Mixed conditions retain all training identities and provide an old–new pair for each of them, whereas the A and B controls remove identities without genuine historical observations entirely. At 10% coverage, for example, the mixed model is trained across the full identity set, while the real-only controls retain only approximately one tenth of those identities. The large mixed–A and mixed–B efects at this coverage level are therefore consistent with the value of preserving a broad set of cross-domain training identities rather than repeatedly optimizing on a much smaller genuine subset.

This interpretation is also supported by the rapid reduction of the completion efect as genuine coverage increases. When most identities already contain genuine old-domain observations, synthetic completion adds progressively fewer otherwise missing identities to the cross-domain training set. By 90% genuine coverage, little diference remains between mixed and real-only training.

The results do not, however, support attributing all benefits of synthetic data exclusively to identity completion. In the Real+Synth condition, all identities already had genuine old-domain observations, yet adding OLDIFY1 increased mean R@1 from 92.15% to 94.05%. The paired efect was +1 90 pp, with a 95% CI of [−0 10 +4 09] pp. Because the interval narrowly included zero, this result does not establish a clear augmentation benefit, but it provides a signal that synthetic observations may also contribute useful appearance variability independently of missing-identity completion.

The relative importance of these mechanisms appears to depend on the degree of scarcity. At 10% and 30% genuine coverage, the completion efects of approximately 5–14 pp are much larger than the full-coverage Real+Synth efect, indicating that preservation of broader training identity coverage is the dominant explanation in this regime. At 50%–70% coverage, the completion efects are of roughly the same order as the Real+Synth efect. Both identity completion and more general augmentation may therefore contribute when genuine historical coverage is already moderate.

## 5.4. Dataset composition and retrieval implications

Absolute retrieval performance and the magnitude of the completion efect varied across dataset partitions, indicating that the value of historical observations depends not only on their number but also on the identities represented in the training and test sets. The decreasing completion efect with increasing genuine coverage was, however, reproduced across all three pairwise disjoint test partitions. At 10% and 30% genuine coverage, the mixed condition outperformed its exposure-matched A control in every partition, whereas the diferences became much smaller at 50% and 70% and efectively disappeared at 90%.

This variability suggests that genuine historical observations have nonuniform informational value. Records involving large viewpoint changes, unusual degradation, strong cropping, uncommon reproduction processes, or visually ambiguous objects may provide more useful cross-domain supervision than records difering mainly in low-level appearance. Consequently, subsets containing the same number of historical identities need not be equally informative.

From a collection-management perspective, the results suggest a hybrid strategy. Genuine historical observations remain the most valuable source of cross-domain supervision, but synthetic completion can reduce the performance cost of incomplete archival coverage. Rather than requiring genuine historical material for every training identity, an informative genuine subset could potentially be combined with synthetic counterparts for identities lacking archival records. The present study does not determine how such a genuine subset should be selected, but retrieval dificulty, domain discrepancy, visual diversity, or expected information gain are plausible criteria for future investigation.

The high R@5 and R@10 values further distinguish expertassisted from fully automatic retrieval. The strongest efects of historical-data scarcity occur at rank 1, whereas the correct identity generally remains within a short candidate list at greater retrieval depths. Synthetic training may therefore be particularly useful in workflows where an automated system narrows a large collection to a small set of candidates that can subsequently be inspected by a curator or other domain expert.

More broadly, synthetic data can be viewed as a mechanism for completing an incompletely observed domain rather than solely as a means of enlarging an already represented image distribution. This perspective may apply to other instance-level cross-domain retrieval tasks in which one observation domain is widely available but its counterpart exists only for a subset of identities.

## 5.5. Limitations

Several limitations constrain the generalization and causal interpretation of these findings. First, the experiments use one dataset assembled from two cultural heritage sources. Although the three test sets are pairwise identity-disjoint and together contain 483 distinct test identities, all observations originate from the same assembled collection, and training or validation identities may occur in more than one partition. The observed relationship between genuine historical coverage and synthetic completion should therefore be validated on independent collections with diferent object types, acquisition processes, and degrees of domain shift.

Second, the study evaluates one retrieval backbone, one metric-learning objective, and one degradation-based synthetic aging pipeline. Diferent architectures, loss functions, sampling strategies, or synthetic-generation methods may respond diferently to the same scarcity regime. In particular, OLD-IFY mainly modifies image appearance while preserving much of the viewpoint, framing, composition, and geometry of its source. Synthetic methods capable of introducing realistic structural or acquisition changes may reduce the remaining gap between synthetic and genuine historical observations.

Third, the A and B controls do not isolate cross-domain pairing from overall identity diversity completely. Identities without genuine old-domain observations are removed entirely from these controls, including their contemporary images. The mixed variants therefore difer from A and B not only in the availability of old-domain counterparts, but also in the number of unique identities retained in training. The results are consequently consistent with broader cross-domain training identity coverage being a primary contributor, but the present design cannot determine how much of the gain arises specifically from restored old–new pairing and how much arises from retaining a larger and more diverse identity set.

Fourth, the A variants match expected per-identity exposure only up to rounding and do not reproduce complete optimization trajectories. Early stopping and stochastic sampling can result in diferent total numbers of updates across trained models. The B variants provide a complementary full-budget control, but necessarily increase repeated sampling of the reduced genuine subsets. Together, the controls show that additional exposure alone does not reproduce the large low-coverage completion gains, but they cannot remove every possible interaction between identity-set size and optimization.

Fifth, scarcity is simulated by selecting nested genuine identity subsets rather than by reproducing the historical missingness process of a particular museum or archive. Although the subsets are constructed using category-stratified ordering, real collections may lack historical documentation systematically rather than randomly. Objects with certain ages, materials, acquisition histories, or documentation practices may be more likely to have archival images. The evaluated coverage percentages should therefore not be interpreted as universal thresholds applicable to all collections.

Finally, confidence intervals are estimated using paired bootstrap resampling at the test-identity level together with resampling of training seeds. Identity-level resampling preserves the dependence among queries belonging to the same object in the small number of cases with multiple images, while the three evaluated partitions retain equal weight. The resulting confidence intervals quantify uncertainty over the evaluated test identities and training seeds; they should not be interpreted as population-level uncertainty across independent cultural heritage datasets. Similarly, the prespecified two-percentage-point acceptable-loss criterion provides a practical descriptive reference rather than a formal non-inferiority test.

## 6. Conclusions

Synthetic aging cannot fully replace genuine historical training data in cross-domain cultural heritage retrieval. Complete replacement reduced bidirectional mean R@1 from 92.15% to 87.94%, corresponding to a paired efect of −4 21 percentage points. Increasing the synthetic sampling pool from one to three independently generated variants did not improve retrieval, indicating that additional stochastic realizations of the same degradation pipeline do not reproduce the variability contained in genuine historical observations.

Synthetic completion was substantially more efective when genuine historical observations were scarce. At 10% genuine old-domain identity coverage, retaining the full training identity set through synthetic completion improved mean R@1 by 12.30 pp relative to the exposure-matched real-only control and by 13.99 pp relative to the real-only control using the full optimization budget. At 30% coverage, the corresponding gains were 4.67 and 5.47 pp. The completion advantage decreased as genuine historical coverage increased and was efectively absent at 90%.

Increasing repeated exposure to the same reduced genuine identity sets did not reproduce these gains. The results are therefore consistent with broader cross-domain training identity coverage being a primary contributor to the value of synthetic completion, particularly under severe scarcity. The present controls do not, however, separate this efect completely from the benefit of retaining a larger and more diverse set of training identities.

Using a prespecified acceptable-loss margin of two percentage points, 70% genuine old-domain identity coverage was the lowest evaluated level whose 95% bootstrap confidence interval remained within the margin relative to complete genuine coverage. This value should be interpreted as specific to the present dataset and experimental protocol rather than as a universal requirement for cultural heritage retrieval.

Synthetic data may also provide a weaker augmentation effect when genuine historical coverage is already complete. Adding OLDIFY1 observations to the complete genuine training set increased mean R@1 from 92.15% to 94.05%, although the paired 95% confidence interval ([−0 10 +4 09] pp) narrowly included zero. Genuine and synthetic observations should therefore be regarded as complementary rather than interchangeable: genuine images provide authentic historical variability, while synthetic observations can extend crossdomain supervision where archival records are missing and may additionally contribute useful appearance variability.

For cultural heritage applications, the practical implication is that synthetic data are most valuable when used to fill specific informational gaps rather than simply to increase image count. Training-set construction should therefore consider which identities and domains are missing, not only the total number of available images. This perspective can support artwork matching, collection integration, catalog enrichment, provenance research, and other retrieval tasks involving heterogeneous historical documentation.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The source images used in this study originate from the publicly accessible Louvre Collections and Lost Art Database. Access to, reuse of, and redistribution of these images remain subject to the terms and copyright restrictions of the respective source institutions. The dataset partitions, experimental configurations, aggregated results, and source code used to generate the synthetic images and evaluate the retrieval models are available from the corresponding author upon reasonable request.

## Funding

This work was supported by the Polish Agency for Enterprise Development (PARP) under Grant No. FENG.01.01- IP.02-3717/23, co-funded by the European Union through the European Funds for a Modern Economy 2021–2027 (FENG), SMART Path.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used OpenAI ChatGPT to support data analysis, language editing, restructuring, and improvement of the clarity and readability of the manuscript. After using this tool, the authors reviewed, verified, and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] S. R. Dubey, A decade survey of content based image retrieval using deep learning, arXiv preprint arXiv:2012.00641 (2020). arXiv:2012.00641.

[2] F. Radenovic, A. Iscen, G. Tolias, Y. Avrithis, O. Chum,´ Revisiting oxford and paris: Large-scale image retrieval benchmarking, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 5706–5715. arXiv:1803.11285.

[3] N.-A. Ypsilantis, N. Garcia, G. Han, S. Ibrahimi, N. van Noord, G. Tolias, The Met dataset: Instance-level recognition for artworks, in: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, Vol. 1, 2021, pp. 1–12.

[4] P. Koniusz, Y. Tas, H. Zhang, M. Harandi, F. Porikli, R. Zhang, Museum exhibit identification challenge for the supervised domain adaptation and beyond, in: Computer Vision – ECCV 2018, Vol. 11220 of Lecture Notes in Computer Science, Springer, 2018, pp. 815–833. doi: 10.1007/978-3-030-01270-0\_48.

[5] W. Peaslee, L. Wrapson, C.-B. Schönlieb, Domain generalization and punch mark classification, Journal of Cultural Heritage 72 (2025) 226–236. doi:10.1016/j. culher.2025.02.001.

[6] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, V. Lempitsky, Domain-adversarial training of neural networks, Journal of Machine Learning Research 17 (59) (2016) 1–35. arXiv:1505.07818.

[7] K. Zhou, Z. Liu, Y. Qiao, T. Xiang, C. C. Loy, Domain generalization: A survey, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (4) (2023) 4396–4415. arXiv:2103.02503, doi:10.1109/ TPAMI.2022.3195549.

[8] E. Hofer, N. Ailon, Deep metric learning using triplet network, in: Similarity-Based Pattern Recognition, Vol. 9370 of Lecture Notes in Computer Science, Springer, 2015, pp. 84–92. arXiv:1412.6622, doi:10.1007/ 978-3-319-24261-3\_7.

[9] F. Schrof, D. Kalenichenko, J. Philbin, FaceNet: A unified embedding for face recognition and clustering, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815– 823. arXiv:1503.03832, doi:10.1109/CVPR.2015. 7298682.

[10] A. Hermans, L. Beyer, B. Leibe, In defense of the triplet loss for person re-identification, arXiv preprint arXiv:1703.07737 (2017). arXiv:1703.07737.

[11] Z. Wan, B. Zhang, D. Chen, P. Zhang, D. Chen, J. Liao, F. Wen, Bringing old photos back to life, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. arXiv:2004.09484.

[12] Z. Wan, B. Zhang, D. Chen, P. Zhang, D. Chen, F. Wen, J. Liao, Old photo restoration via deep latent space translation, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (2) (2023) 2071–2087. doi:10. 1109/TPAMI.2022.3163183.

[13] C. Shorten, T. M. Khoshgoftaar, A survey on image data augmentation for deep learning, Journal of Big Data 6 (1) (2019) 60. doi:10.1186/s40537-019-0197-0.

[14] Musée du Louvre, Louvre Collections, https:// collections.louvre.fr/en/, accessed: 2026-07-14 (2026).

[15] German Lost Art Foundation, Lost Art Database, https: //www.lostart.de/, accessed: 2026-07-14 (2026).

[16] M. Tan, Q. V. Le, EficientNetV2: Smaller models and faster training, in: Proceedings of the 38th International Conference on Machine Learning (ICML), Vol. 139 of Proceedings of Machine Learning Research, 2021, pp. 10096–10106. arXiv:2104.00298.

[17] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, ImageNet: A large-scale hierarchical image database, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255. doi:10.1109/CVPR.2009.5206848.

Supplementary Material

## S1. Synthetic aging procedure

The OLDIFY images were generated ofline from contemporary images in the training subset. Images whose larger spatial dimension exceeded 1500 pixels were proportionally downscaled to a maximum dimension of 1500 pixels; smaller images retained their original resolution.

For each generated image, all transformation parameters were sampled independently from the ranges in Table S1. Continuous parameters followed ${ \mathcal { U } } ( a , b )$ , whereas integer parameters followed a discrete uniform distribution including both endpoints. All processing stages were enabled, although an efect could become negligible when its sampled intensity was zero or close to zero.

OLDIFY1, OLDIFY2, and OLDIFY3 were independent realizations of the same stochastic procedure. They used identical transformation definitions, ranges, and ordering, difering only in the sampled parameter values.

Table S1: Parameter ranges used for synthetic aging. Parameters were sampled independently for each generated image.
<table><tr><td>Parameter</td><td>Sampling range</td><td>Interpretation</td></tr><tr><td>frame_thickness</td><td>Uint(4, 15) px</td><td>Replicated border</td></tr><tr><td>gaussian_mean</td><td>U(0.00,0.05)</td><td>Gaussian-noise mean</td></tr><tr><td>gaussian_var</td><td>U(0.001,0.010)</td><td>Gaussian-noise variance</td></tr><tr><td>salt_pepper_amount</td><td>U(0.001,0.005)</td><td>Impulse-noise fraction per polarity</td></tr><tr><td>vignette_strength</td><td>U(0.55,0.95)</td><td>Vignetting strength</td></tr><tr><td>scratch_intensity</td><td>Uint(0,4)</td><td>Scratch and dust intensity</td></tr><tr><td>blur_sigma</td><td>U(1.0,3.0)</td><td>Gaussian-blur standard deviation</td></tr><tr><td>gamma</td><td>U(0.85,1.20)</td><td>Gamma correction</td></tr><tr><td>perspective_top_shift</td><td>U(0.00,0.06)</td><td>Relative top-edge inset</td></tr><tr><td>perspective_bottom_shift</td><td>U(0.00,0.06)</td><td>Relative bottom-edge inset</td></tr><tr><td>perspective_left_shift</td><td>U(0.00,0.15)</td><td>Relative left-edge inset</td></tr><tr><td>perspective_right_shift</td><td>U(0.00,0.15)</td><td>Relative right-edge inset</td></tr><tr><td>paper_strength</td><td>U(0.08,0.25)</td><td>Paper-texture modulation</td></tr><tr><td>leak_intensity</td><td>U(0.00,0.25)</td><td>Light-leak intensity</td></tr><tr><td>leak_angle_deg</td><td>U(0, 360) deg</td><td>Light-leak origin</td></tr><tr><td>sepia_strength</td><td>U(0.90, 1.00)</td><td>Sepia-to-grayscale interpolation</td></tr></table>

## S1.1. Processing sequence

The transformations were applied in the following fixed order:

1. Replicated border. A border of 4–15 pixels was added by replicating the outermost image pixels.

2. Gaussian and impulse noise. Additive Gaussian noise was applied in the normalized intensity range [0 1], after which approximately $p _ { \mathrm { s p } } H W$ pixels were independently replaced with white values and the same number with black values. Intensities were clipped to the valid range.

3. Vignetting. Each channel was multiplied by $V ( x , y ) = 1 -$ $s _ { \mathrm { v } } r ( x , y ) ^ { 2 }$ , where $r ( x , y )$ is the normalized distance from the image center. The mask was clipped to [0 1].

4. Scratches and dust. For positive intensity values, randomly positioned lines of 20–80 pixels in length and one or two pixels in thickness were generated and blurred. Sparse dust artifacts were added with density proportional to the sampled intensity. This stage was inactive when the intensity equaled zero.

5. Blur and gamma correction. Gaussian blur was followed by the transformation $I ^ { \prime } \ = \ I ^ { 1 / \gamma }$ in normalized intensity space.

6. Perspective transformation. The four image corners were moved toward the interior according to the independently sampled top, bottom, left, and right ofsets. The resulting quadrilateral was projected onto a rectangular image. Only its internal region was retained, so the output could be smaller than the input.

7. Paper texture. Uniform random noise was smoothed with a Gaussian filter of $\sigma = 3$ , normalized to [0 85 1 15], and applied as $I ^ { \prime } = I T ^ { s _ { \mathrm { p a p e r } } }$

8. Light leak. A warm artifact was generated from an image corner determined by the sampled angle. Its intensity decreased quadratically with distance from the origin. Relative BGR channel contributions were 0 40, 0 65, and 1 00.

9. Sepia-to-grayscale conversion. A standard sepia transform was applied using

$$
{ \left[ \begin{array} { l } { R _ { \mathrm { s } } } \\ { G _ { \mathrm { s } } } \\ { B _ { \mathrm { s } } } \end{array} \right] } = { \left[ \begin{array} { l l l } { 0 . 3 9 3 } & { 0 . 7 6 9 } & { 0 . 1 8 9 } \\ { 0 . 3 4 9 } & { 0 . 6 8 6 } & { 0 . 1 6 8 } \\ { 0 . 2 7 2 } & { 0 . 5 3 4 } & { 0 . 1 3 1 } \end{array} \right] } { \left[ \begin{array} { l } { R } \\ { G } \\ { B } \end{array} \right] } .
$$

The result was interpolated with its grayscale version using $\alpha = 2 s _ { \mathrm { s e p i a } } - 1$ and $I _ { \mathrm { o u t } } = ( 1 - \alpha ) I _ { \mathrm { s e p i a } } + \alpha I _ { \mathrm { g r a y } }$ . Because $s _ { \mathrm { s e p i a } } ~ \in ~ [ 0 . 9 0 , 1 . 0 0 ]$ , the generated images ranged from strongly desaturated sepia to fully grayscale.

OLDIFY1 was used in the controlled-scarcity and syntheticcompletion experiments. The union of OLDIFY1, OLD-IFY2, and OLDIFY3 was used only in the Synth-3 completereplacement condition.

## S2. Partition-level retrieval results

Partition-level results were examined to assess whether the relationship between genuine old-domain coverage and synthetic completion was reproduced across diferent test-set compositions. Each partition contains 161 test identities, and the three test identity sets are pairwise disjoint. Within each partition, the reported values are averaged over the three training seeds.

![](images/325f2c94d4dd1a5fc7b53083eb2dcc9084d5cf8245a0f910206ba6008f05cb11.jpg)  
Figure S1: Bidirectional mean R@1 for the real–synthetic completion series, reported separately for the three dataset partitions and averaged over three training seeds. Each curve spans genuine old-domain identity coverage from 0% to 100%; identities without genuine old-domain observations are completed using OLDIFY1 images.

As shown in Figure S1, absolute retrieval performance differed across partitions, indicating diferences in test-set difficulty. Nevertheless, the general relationship between genuine historical coverage and retrieval performance was similar: synthetic-completion performance improved substantially from the fully synthetic endpoint and tended to flatten at higher genuine-coverage levels. The curves were not strictly monotonic within individual partitions.

The benefit of synthetic completion relative to the exposurematched real-only A controls was particularly consistent under severe scarcity. At 10% genuine coverage, the mixed–A diferences were +10 09, +14 63, and +12 19 percentage points (pp) for Splits 1, 2, and 3, respectively. At 30%, the corresponding diferences were +6 29, +4 36, and +3 35 pp. Thus, the large completion benefit observed at low genuine coverage was reproduced in all three pairwise disjoint test partitions.

At intermediate coverage, the magnitude of the efect became smaller but remained positive in each partition. At 50% genuine coverage, the mixed–A diferences were +0<sub>.</sub>87, +1<sub>.</sub>30, and +2 79 pp, while at 70% they were +0 76, +2 06, and +1 63 pp for Splits 1–3, respectively. By 90% genuine coverage, the differences were small and their direction was no longer consistent across partitions.

These results indicate that dataset composition afects both absolute retrieval dificulty and efect magnitude, but the main scarcity-related pattern is stable: synthetic completion provides its largest and most consistent benefit when genuine historical observations cover only a small fraction of the training identities, and this benefit diminishes as genuine coverage approaches completeness.

## S3. Detailed and secondary retrieval results

This section provides numerical results complementary to the primary analyses reported in the main text. It includes the complete genuine-coverage series and real-only controls, the prespecified acceptable-loss comparison with complete genuine old-domain coverage, and retrieval performance at larger rank depths.

## S3.1. Complete scarcity-series results

Table S2 reports bidirectional mean R@1 for the complete real–synthetic completion series and the real-only A and B controls. Values are averages over three dataset partitions and three training seeds. The mixed series retains all 808 training identities and replaces missing genuine old-domain observations with OLDIFY1 images. The A and B controls retain only identities with genuine old-domain observations.

Table S2: Bidirectional mean R@1 for the complete controlled-scarcity series and real-only controls, averaged over three dataset partitions and three training seeds. Mixed conditions use OLDIFY1 observations for identities without genuine old-domain images. A controls use matched expected per-identity exposure, whereas B controls use the full per-epoch optimization budget.
<table><tr><td>Condition</td><td>Genuine OLD coverage</td><td>Mean R@1</td></tr><tr><td>RO_S100</td><td>0%</td><td>87.94%</td></tr><tr><td>R10_S90</td><td>10%</td><td>89.44%</td></tr><tr><td>R10_A</td><td>10%</td><td>77.14%</td></tr><tr><td>R10_B</td><td>10%</td><td>75.45%</td></tr><tr><td>R20_S80</td><td>20%</td><td>88.92%</td></tr><tr><td>R30_S70</td><td>30%</td><td>89.92%</td></tr><tr><td>R30_A</td><td>30%</td><td>85.26%</td></tr><tr><td>R30_B</td><td>30%</td><td>84.45%</td></tr><tr><td>R40_S60</td><td>40%</td><td>89.57%</td></tr><tr><td>R50_S50</td><td>50%</td><td>91.06%</td></tr><tr><td>R50_A</td><td>50%</td><td>89.41%</td></tr><tr><td>R50_B</td><td>50%</td><td>88.79%</td></tr><tr><td>R60_S40</td><td>60%</td><td>91.59%</td></tr><tr><td>R70_S30</td><td>70%</td><td>92.18%</td></tr><tr><td>R70_A</td><td>70%</td><td>90.69%</td></tr><tr><td>R70_B</td><td>70%</td><td>89.89%</td></tr><tr><td>R80_S20</td><td>80%</td><td>92.41%</td></tr><tr><td>R90_S10</td><td>90%</td><td>91.96%</td></tr><tr><td>R90_A</td><td>90%</td><td>92.18%</td></tr><tr><td>R90_B</td><td>90%</td><td>92.04%</td></tr><tr><td>R100_S0</td><td>100%</td><td>92.15%</td></tr></table>

The mixed completion series shows an overall increase in retrieval performance with increasing genuine old-domain coverage, but the relationship is not strictly monotonic. The largest improvements relative to the real-only controls occur at low genuine coverage. Diferences become progressively smaller at intermediate coverage and are negligible at 90%, consistent with the paired analyses in the main text. Small discrepancies between diferences calculated from the rounded values in Table S2 and the reported paired efects arise because statistical calculations use unrounded query-level results.

## S3.2. Performance relative to complete genuine coverage

The mixed completion conditions were compared directly with the complete genuine-data reference R100\_S0. Figure S2 shows the paired diference in bidirectional mean R@1 and the prespecified acceptable-loss boundary of −2 percentage points.

For completeness, the numerical paired efects are reported in Table S3. A condition meets the descriptive acceptable-loss criterion when the lower bound of its 95% confidence interval is greater than −2 pp.

The point estimates approached the complete-genuine-data reference before the uncertainty intervals satisfied the prespecified margin. At 50% and 60% genuine coverage, the mean diferences were only −1 09 and −0 56 pp, respectively, but their lower confidence bounds extended below −2 pp. At 70%, the paired diference was +0<sub>.</sub>03 pp with a 95% CI of [−1 78 +1 88] pp. Thus, 70% was the lowest evaluated genuine-coverage level satisfying the descriptive 2-pp acceptable-loss criterion. The 80% and 90% conditions also satisfied the criterion. As stated in the main text, this analysis is not interpreted as a formal non-inferiority test.

Performance loss relative to complete genuine OLD coverage  
![](images/fc7a5cac976867e8fd9c4fadba4adc2b39218670beb67e7b57db5fbba7726328.jpg)  
Figure S2: Paired diference in bidirectional mean R@1 between each real–synthetic completion condition and the complete genuine-data reference R100\_S0. Error bars represent 95% confidence intervals obtained using the paired identity-level bootstrap described in Section 3.6.

Table S3: Paired diferences in bidirectional mean R@1 between the real– synthetic completion series and the complete genuine-data reference R100\_S0. Confidence intervals were estimated using the paired identity-level bootstrap with seed resampling described in Section 3.6.
<table><tr><td>Genuine OLD coverage</td><td>Difference vs. R100_S0</td><td>95%CI</td><td>2-pp criterion</td></tr><tr><td>0%</td><td>-4.21 pp</td><td>[-6.19, −2.23] pp</td><td>No</td></tr><tr><td>10%</td><td>-2.71 pp</td><td>[-4.98, −0.49] pp</td><td>No</td></tr><tr><td>20%</td><td>-3.23 pp</td><td>[-5.18,-1.33] pp</td><td>No</td></tr><tr><td>30%</td><td>-2.23 pp</td><td>[-4.18, -0.30] pp</td><td>No</td></tr><tr><td>40%</td><td>-2.58 pp</td><td>[-4.63, -0.60] pp</td><td>No</td></tr><tr><td>50%</td><td>-1.09 pp</td><td>[-2.68, +0.55] pp</td><td>No</td></tr><tr><td>60%</td><td>-0.56 pp</td><td>[-2.19, +1.15] pp</td><td>No</td></tr><tr><td>70%</td><td>+0.03 pp</td><td>[−1.78, +1.88] pp</td><td>Yes</td></tr><tr><td>80%</td><td>+0.26 pp</td><td>[-1.59, +2.12] pp</td><td>Yes</td></tr><tr><td>90%</td><td>-0.19 pp</td><td>[-1.88, +1.52] pp</td><td>Yes</td></tr></table>

## S3.3. Rank-dependent retrieval performance

Figure S3 compares bidirectional mean recall at ranks 1, 5, and 10 across the complete real–synthetic completion series.

![](images/ebb89ba4af35f604207ba99072aac4160d216c6d159da5a13b06f8a50fce502e.jpg)  
Figure S3: Rank-dependent retrieval performance of the real–synthetic completion series. Curves show bidirectional mean R@1, R@5, and R@10 as a function of genuine old-domain identity coverage, averaged over three dataset partitions and three training seeds.

The efect of historical-data scarcity was substantially larger at rank 1 than at greater retrieval depths. Even complete synthetic replacement achieved 98.43% bidirectional mean R@10, while the complete genuine-data reference reached 99.23%. Across the mixed series, R@10 remained close to 99%, whereas the corresponding R@1 values varied more strongly with genuine old-domain coverage.

These results indicate that synthetic aging often preserves enough identity information to place the correct object within a small set of highly ranked candidates, even when it does not reproduce the genuine historical domain suficiently well to rank the correct identity first. The distinction supports the potential use of synthetic completion in expert-assisted retrieval workflows, where the automated system provides a short candidate list for subsequent manual verification.