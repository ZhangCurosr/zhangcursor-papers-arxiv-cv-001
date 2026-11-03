Preprint. Accepted for publication at the British Machine Vision Conference (BMVC) 2026.

# Circumventing Magnitude Collapse via Modular Two-Stage Open-Set Recognition

Laurin van den Bergh1,2   
bergh@ifi.uzh.ch   
Halil Bisgin³   
bisgin@umich.edu   
Andres Palechor1,†   
andrespalechor11@gmail.com

Mike Suter1,† mi.suter@protonmail.com Manuel Günther1,2,† siebenkopf@googlemail.com

1 Department of Informatics University of Zurich Zurich, Switzerland

² Digital Society Initiative University of Zurich Zurich, Switzerland

3 College of Innovation & Technology University of Michigan-Flint Michigan, USA

## Abstract

Open-set recognition (OSR) requires neural networks to classify known classes while rejecting unknown samples, which is critical for real-world deployment. So far, OSR research studied and developed representation learning and postprocessing methods independently and their interactions remain unexplored. In this paper, we reconsider OSR as a modular two-stage pipeline and present the first systematic study of interactions between representation learning and postprocessing methods across dataset scales and usage of known unknown data. First, we discover a failure mode we term magnitude collapse, where representation learning methods that utilize known unknown data can suffer performance degradation at large scale and irreversibly destroy discriminative information, despite excelling at small scale. Second, we study the modular interactions and reveal when they can be leveraged for modular performance gains via two-stage processing and where interaction effects amplify performance degradation due to magnitude collapse. Third, we show how these findings can be used to achieve state-of-the-art performance with a simple K +1 class classifier baseline and magnitude-aware postprocessing while circumventing magnitude collapse. Finally, our results demonstrate that small-scale evaluations with known unknown data are not predictive of large-scale performance, invalidating current best practices in OSR research. Code available at https://github.com/TwoDigitsOneNumber/Two-Stage-OSR.

## 1 Introduction

The rapid advancement of deep learning methods for image recognition increasingly promotes their real-world adoption, which requires them to adequately detect and handle unknown inputs for the reliability and safety of these systems [, , ]. This task is typically studied under the two closely-related problem formulations: Open-set Recognition (OSR) and Out-of-Distribution (OOD) detection. Both aim to improve the robustness of classifiers by detecting test-time distributional shifts.

![](images/432837ab21e30e924bf70f152e451fa6cde1e2792173fc909d80bec2419ba684.jpg)

![](images/634de9e0a2ab2b4d011c8fd04bcee644c992fcddcc24fc5a79f706143441d2da.jpg)  
Figure 1: OPEN-SET RECOGNITION FRAMEWORK. The modular two-stage OSR framework separates representation learning and postprocessing, and reveals additional performance gains, by leveraging interaction effects between the two. Undesirable interaction effects exist for magnitude-manipulating representation learning (red) combined with magnitude-aware postprocessors (blue), while beneficial effects are observed for K +1 class classifiers and magnitude-aware postprocessors. Small-scale benchmarks are not predictive of large-scale performance and do not exhibit similar behavior due to limited similarity between known and known unknown classes.

While the categorization of OSR methods into representation learning and postprocessing methods is commonly understood, current OSR methods are studied in isolation or compared as standalone methods, neglecting their differences and modular nature and potential for improvements through combinations. Since the interaction effects between representation learning and postprocessors have neither been explored nor formalized for OSR methods, we ask: can representation learning methods enhance or shape feature representations to amplify downstream postprocessors performance? Or vice versa: is the optimal choice of postprocessor dependent on the representation learning training objective?

In both representation learning and postprocessing methods, the feature magnitude has been identified as a crucial factor for performance with unknown data observing smaller feature magnitudes [, , 3, , 2]. For instance, Wang et al. [B4] highlight that magnitude-aware (MA) OOD postprocessing generally outperforms alternatives. This raises the question: do MA postprocessors synergize with representation learning methods that purposefully manipulate feature magnitudes during training? We identify a sub-category of representation learning methods, which we term magnitude-manipulating (MM), that utilize known unknown data to separate known and known unknown classes during training based on feature magnitude. However, MM methods tend to be sensitive to known unknown data distribution at large scale, with performance falling below baselines, despite their success on small-scale datasets [, ]. This raises concerns about their real-world applicability, and we seek to understand the underlying causes of this performance degradation at scale.

Our analysis reveals the mechanism behind the performance degradation of MM methods as the interplay of magnitude-manipulation and high similarity between known unknown and known classes, a scenario not found on small-scale benchmarks. This causes a magnitude collapse in similar known classes and creates an undesirable linear dependency between feature magnitudes and class-wise accuracy, leading to systematically imbalanced class-wise detection performance. We show that magnitude collapse can be avoided by using an additional output node (a K + 1 class) for known unknown data, a simple and effective baseline that consistently outperforms other methods across scales and does not require hyperparameter tuning. This performance degradation of MM methods is further amplified by MA postprocessing, which otherwise experiences desirable interaction effects when combined with non-MM representation learning methods and outperforms other postprocessors. Moreover, our experiments suggest that representation learning without known unknown data and postprocessors are highly independent components of OSR systems with clear separation of responsibilities, enabling modular additive performance gains.

Overall, our study advances the understanding of OSR systems and provides actionable guidelines for researchers and practitioners. First, leverage the modularity of OSR systems by augmenting non-MM representation learning methods with MA postprocessors to achieve additional performance gains almost for free. Second, when training or fine-tuning a classifier with known unknown data that is visually similar to known classes, avoid MM representation learning methods. Instead, use a K + 1 class classifier as baseline to mitigate magnitude collapse and leverage positive interaction effects. Finally, validate OSR methods on large-scale benchmarks before deployment, as small-scale evaluations are not predictive of large-scale performance.

In summary, our contributions are as follows:

• For the first time in OSR literature, we explore the modularity and interaction effects of representation learning and postprocessing methods through two-stage OSR, revealing where modular performance gains can be achieved and where to avoid negative synergies.

• We discover the magnitude collapse mechanism behind performance degradation at scale and how it impacts interaction effects.

• We demonstrate how interaction effects and known unknown data can be leveraged at scale to achieve state-of-the-art performance regardless of known unknown data distribution with the simple K +1 class baseline and two-stage processing of OSR techniques.

• Our experiments highlight that small-scale evaluations are not predictive of largescale performance when using known unknown data.

## 2 Related work

OSR and Relation to OOD Detection. OSR is formalized as the task of accurately classifying samples from known classes while rejecting samples from semantically unknown classes [9], thereby detecting test-time semantic shift [2]. OOD detection is a broader task [B] that aims to detect general distribution shift, which can include semantic or covariate shifts [, , ] and is often posed as a binary classification problem [, , , , ]. As a result, OSR and OOD detection differ in their evaluation protocols [B, ]: OSR partitions a single dataset into known and unknown classes to mitigate covariate shifts [3, ], while OOD detection typically uses different datasets for in-distribution (ID) and OOD classes []. Despite these differences, it has been indicated that methods that perform well on one task tend to perform well on the other [, , 7].

Known Unknown Data in OSR. Known unknown samples serve as a proxy for unknown classes during training and are distinct from known or ID classes [, B]. Known unknown samples are also referred to as outlier images [], natural adversarial examples [], and negative samples []. Real known unknown data is used for OOD detection [, ] and OSR [, ], dating back to the earliest approaches [B]. While a large attention in OSR research is paid to artificially generate known unknown samples [, , , 3, in this study we exclude generative methods and instead use real known unknown data. The standard small-scale benchmarks MNIST, CIFAR, SVHN, and TinyImageNet partition all classes into known and unknown [3], therefore do not allow any known unknown classes. The large-scale Semantic Shift Benchmark (SSB) [2] uses the entire ImageNet-1K dataset as known classes and selects unknowns from a set of disjoint classes from ImageNet-21K-P.

Differences and Similarities to Prior Art. Wang et al. [B] recently acknowledged the distinction between representation learning and postprocessors and the potential for combined approaches in the context of disentangling OSR and OOD methods and benchmarks without analyzing component interactions. However, they focus primarily on OOD detection methods, with 10 out of 12 methods being postprocessing, leaving modern OSR methods and their interaction effects unexplored.

## 3 Modular two-stage framework for OSR

We disentangle OSR methods into two modular sequential components: representation learning f (classifier training) and postprocessing g (of pre-computed representations). Within this two-stage framework, every OSR system can be viewed as a combination of one representation learning method f and one postprocessor g, denoted as $g \circ f , e . g .$ , our baseline $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ combines Cross-Entropy (CE) training with Maximum SoftMax Probability (MSP) postprocessing. Figure 2 provides an overview of the two-stage framework. We summarize key characteristics relevant to this study of representation learning and postprocessing methods in Table 1. Appendix A.1 discusses how methods from Table 1 can be formalized in this framework.

Representation learning methods train or fine-tune a classifier f and extract the representations $\mathcal { R } _ { n } = ( \varphi _ { n } , \mathbf { z } _ { n } , \mathbf { y } _ { n } ) = f ( \mathbf { x } _ { n } )$ for sample ${ \bf x } _ { n }$ , where $\varphi _ { n }$ are discriminative deep features, $\mathbf { z } _ { n }$ are the logits, and ${ \bf y } _ { n }$ the probability distribution over the known classes. They methods can modify the training process in various ways, typically by adapting the loss function [, , , ], involving data augmentation, such as mixup [, ]or generative methods [, , , , , , ], or combinations thereof [B9]. They are trained on a dataset $\mathcal { K } _ { \mathrm { t r a i n } } \cup \mathcal { A } _ { \mathrm { t r a i n } }$ , where for input ${ \bf x } _ { n } , ~ { \boldsymbol { \mathcal { K } } } = \{ ( { \bf x } _ { n } , \tau _ { n } ) ~ | ~ \tau _ { n } \in \mathcal { V } \}$ is the set of known samples with known class labels $\mathcal { V } = \{ 1 , . . . , K \}$ , and $\mathcal { A } = \{ ( \mathbf { x } _ { n } , \tau _ { n } ) \mid \tau _ { n } \not \in \mathcal { V } \}$ is the set of known unknown samples. Evaluation uses $\boldsymbol { \mathcal { K } } _ { \mathrm { t e s t } } \cup \boldsymbol { \mathcal { U } } _ { \mathrm { t e s t } }$ , where $\mathcal { U } = \{ ( \mathbf { x } _ { n } , \tau _ { n } ) \mid \tau _ { n } \not \in \mathcal { V } \}$ denotes unknown samples. Note that while known unknown and unknown samples do not require a specific target label, they are required to not share the label space with known classes  [9],

![](images/3876950c5cc21858ddf85125a9552f70acf247de9f6011169dc74fe04662ecde.jpg)  
Figure 2: TWO-STAGE PROCESSING FRAMEWORK FOR OSR. An image is presented to the backbone network $F _ { \mathrm { : } }$ , which extracts deep features $\varphi$ that are then processed with a linear layer W to logits z, and further with SoftMax to probabilities y. Solid lines indicate (potentially) learnable connections, while dashed lines highlight non-learnable connections. The postprocessor takes the deep features, logits, or probabilities as input and outputs a class prediction $k ^ { * }$ and a OOD score $\gamma ^ { * }$

Postprocessors operate post-hoc on representations $\mathcal { R } _ { n }$ to add open-set capabilities to a pre-trained closed-set classifier,1 making them a cheap alternative to expensive representation learning training. However, postprocessors cannot undo any damage to the feature representation learned by the pre-trained network, e.g., when deep feature distributions $\varphi$ from known and unknown classes overlap, no postprocessor is able to separate those samples. Postprocessors can involve training a secondary classifier [, ], employing a statistical model [0], modifying the inputs [], or simply returning elements of $\mathcal { R } _ { n }$ [, 3]. We formalize postprocessors as follows: for test sample $\mathbf { x } _ { n } ^ { * }$ with $\mathcal { R } _ { n } ^ { * }$ , we require a postprocessor to produce two outputs $\mathcal { P } _ { n } ^ { * } = ( k _ { n } ^ { * } , \gamma _ { n } ^ { * } ) = g ( \mathcal { R } _ { n } ^ { * } )$ , where $k _ { n } ^ { * } \in \mathcal { V }$ is the prediction of a known class label and $\gamma _ { n } ^ { * }$ is an OOD score, where high $\gamma _ { n } ^ { * }$ scores indicate known classes. In an operational setting with a threshold $\theta .$ , the OSR decision function can be defined as:

$$
G _ { \mathrm { O S R } } ( \mathcal { R } _ { n } ^ { * } ; \theta ) = \left\{ \begin{array} { l l } { k _ { n } ^ { * } } & { \mathrm { i f ~ } \gamma _ { n } ^ { * } \geq \theta } \\ { \mathrm { u n k n o w n } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{1}
$$

Attributing performance contributions. To isolate and evaluate the performance contributions of individual components, we propose to measure the effect of

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Known unknown</td><td rowspan=1 colspan=1>MM</td><td rowspan=1 colspan=1>Output K + 1</td></tr><tr><td rowspan=1 colspan=1>Cross-Entopy (CE)</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ARPL []</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>K + 1 class [4]</td><td rowspan=1 colspan=1>real</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>Objectosphere (OS) []</td><td rowspan=1 colspan=1>real</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Outlier Exposure (OE) 口</td><td rowspan=1 colspan=1>real</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1></td></tr></table>

(a) Representation Learning
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Trainable</td><td rowspan=1 colspan=1>MA</td><td rowspan=1 colspan=1>Inputs</td></tr><tr><td rowspan=1 colspan=1>Maximum Softmax (MSP) []</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>y</td></tr><tr><td rowspan=1 colspan=1>MaxLogits/MLS [3, ]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Z</td></tr><tr><td rowspan=1 colspan=1>OpenMax [0]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>PostMax []</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>4, z</td></tr><tr><td rowspan=1 colspan=1>GHOST [5]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>4, z</td></tr></table>

(b) Postprocessing

Table 1: KEY CHARACTERISTICS. We list key characteristics of (a) representation learning f and (b) postprocessing g methods for OSR. Representation learning methods highlight which types of known unknown data they use, whether they are Magnitude-Manipulating (MM), and whether they use an additional output node K +1 for the unknown class. For postprocessors, we list whether they require training, are Magnitude-Aware (MA), and which types of inputs they operate on.

representation learning or postprocessing by computing contribution deltas of the respective components, for a given metric M. We compute the representation learning contribution delta to the CE baseline as a function of the postprocessor. Similarly, we separate improvements of postprocessors by computing the postprocessor contribution delta to the MSP baseline. We define the representation learning and postprocessor contribution deltas as follows:

$$
\Delta _ { f } ( f _ { \mathrm { i } } \mid g ) : = \mathcal { M } ( g \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g \circ f _ { \mathrm { C E } } )\tag{2}
$$

$$
\Delta _ { g } ( g _ { \mathrm { i } } \mid f ) : = \mathcal { M } ( g _ { \mathrm { i } } \circ f ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f )\tag{3}
$$

This allows us to decompose the performance improvement of any two-stage OSR system $g _ { \mathrm { j } } \circ f _ { \mathrm { i } }$ over the $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ baseline into representation learning and postprocessing contributions as follows:

$$
\mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) = \Delta _ { f } ( f _ { \mathrm { i } } \mid g _ { \mathrm { M S P } } ) + \Delta _ { g } ( g _ { \mathrm { j } } \mid f _ { \mathrm { i } } )\tag{4}
$$

$$
= \Delta _ { f } ( f _ { \mathrm { { i } } } \mid g _ { \mathrm { { j } } } ) + \Delta _ { g } ( g _ { \mathrm { { j } } } \mid f _ { \mathrm { { C E } } } )\tag{5}
$$

## 4 Study Design and Experimental Setup

We choose five representation learning approaches, based on whether the method requires known unknown data and whether it (explicitly or implicitly) manipulates the feature magnitude. As such we selected the following methods: Cross-Entropy (CE), ARPL [], K + 1 class [], Outlier Exposure (OE) [], and ObjectoSphere (OS) []. We also choose five postprocessors, based on whether they take the feature magnitude into account, i. e., whether they are magnitude-aware. We select MSP [], MaxLogits/MLS [, ], OpenMax [0], PostMax [] and GHOST [].

Datasets. We conduct small-scale experiments on the standard OSR CIFAR+N benchmarks with $N \in \{ 1 0 , 5 0 \}$ [23]. These protocols randomly sample 4 known classes from CIFAR-10 and N unknown classes from CIFAR-100 [].2 To allow training with known unknown samples, we randomly sample N known unknown classes from the remaining classes in CIFAR-100. Large-scale experiments are conducted on the ImageNet protocols $P _ { 1 } , P _ { 2 } .$ and $P _ { 3 }$ [] based on the ILSVRC 2012 dataset []. These protocols offer increasing levels of difficulty from $P _ { 1 }$ to $P _ { 3 }$ by increasing semantic similarity between known, known unknown, and unknown classes based on the WordNet hierarchy []. While $P _ { 1 }$ poses an easy open-set task with low similarity between known and known unknown classes, $P _ { 2 }$ and $P _ { 3 }$ pose increasingly difficult open-set tasks with high similarity between known and known unknown classes.

Evaluation Metrics. To evaluate the binary unknown rejection and the closedset performance in isolation, we use the Area Under the Receiver Operating Characteristics (AUROC) curve and closed-set accuracy, respectively. To evaluate OSR performance we use Correct Classification Rate (CCR) [] and False Positive Rate (FPR) and their single-valued derivations: Area Under the Open-Set Classification Rate (AUOSCR) curve [2], which provides a threshold-agnostic measure, and the Operational Open-set Accuracy (OOSA) [], which equally weights known and unknown samples and measures performance at an operational threshold. For detailed descriptions of all metrics, please refer to Appendix A.6. Given knowns K, unknowns $\mathcal { U } .$ predictions $\mathcal { P } _ { n } ^ { * } = ( k _ { n } ^ { * } , \gamma _ { n } ^ { * } )$ , and threshold $\theta$ we compute CCR and FPR following Rabinowitz et al. [] to allow for arbitrary OOD scores $\gamma ^ { * }$

$$
\mathrm { C C R } ( \theta ) = \frac { \bigl | \{ ( \mathbf { x } _ { n } , \tau _ { n } ) \in K \land k _ { n } ^ { * } = \tau _ { n } \land \gamma _ { n } ^ { * } \geq \theta \} \bigr | } { \bigl | \mathcal { K } \bigr | }\tag{6}
$$

$$
\mathrm { F P R } ( \theta ) = { \frac { \left| \left\{ ( \mathbf { x } _ { n } , \tau _ { n } ) \in \mathcal { U } \wedge \gamma _ { n } ^ { * } \geq \theta \right\} \right| } { | \mathcal { U } | } }\tag{7}
$$

To evaluate our methods we use multiple metrics to capture different aspects of performance. AUOSCR and OOSA lack the interpretability of closed-set accuracy and AUROC, and therefore should not be solely relied upon. On the other hand, the latter do not measure OSR performance holistically []. For example, closedset accuracy is incapable of measuring interaction effects because all postprocessors except OpenMax exclusively rely on the logit order from upstream representation learning methods to produce predictions $k ^ { * }$ and ignore $\gamma ^ { * }$ , resulting in identical accuracy scores (Figure 13d in the Supplemental). AUROC measures the ability to separate known and unknown samples across all thresholds via $\gamma ^ { * }$ and ignores $k ^ { * }$ , thereby directly measuring the discriminative ability of the postprocessor and indirectly of the representation learning method. AUOSCR and OOSA however jointly measure the quality of $k ^ { * }$ and $\gamma ^ { * }$ for any given threshold $\theta$ via CCR(θ) and FPR(θ). As such we rely primarily on AUOSCR and OOSA to answer our research questions but draw complementary insights from closed-set accuracy and AUROC.

Training Details. We train all networks from scratch, both to prevent leakage from pre-training on unknown classes and to isolate representation learning from fine-tuning effects. All networks are trained using SGD with momentum of 0.9 and an initial learning rate of 0.1 with cosine annealing schedule []. For large-scale experiments we train ResNet50 for 120 epochs with batch size of 32 and weight decay of 1e-4. The small-scale experiments are trained with the CNN architecture from Neal et al. [3] and Chen et al. [] for 100 epochs with batch size of 128 and weight decay of 5e-4. We perform early stopping according to validation confidence []. For CE and ARPL, we compute validation confidence on known classes only since including known unknowns yielded unreliable results. Where possible, we rely on recommended hyperparameters for each representation learning and postprocessing method. Others are optimized on the validation set via grid search. Optimal hyperparameters and their ranges are reported in Appendix A.5.

![](images/dae3364d9ec3e14f78003335877a7d2f63944b680c9140efd9e429f184957231.jpg)  
Figure 3: OSR PERFORMANCE. We show OSR performance for gg ° ff combinations across datasets in AUOSCR (top) and OOSA (bottom). The color of each heatmap is normalized independently and centered at the gMspo f cE baseline, where blue shows an increase and red a decrease. Single-stage methods are surrounded by a black border. Results for CIFAR+N are averaged over 5 trials.

## 5 Results and Discussion

## 5.1 Representation Learning with Known Unknown Data

The first set of experiments aims at answering if representation learning with known unknown data can improve OSR performance on large-scale datasets, despite recent studies that suggest otherwise [B]. We compare the performance of representation learning methods with known unknown data (K + 1 class, OE, and OS) to methods that only utilize known classes (CE and ARPL) across datasets. Here, we ignore interaction effects and consider MSP postprocessing or aggregate results over all postprocessors. The AUOSCR and OOSA for every g o f combination are shown in Figure 3, other metrics are reported in Tables 3 and 4 in the Supplemental.

Small-Scale Training Excels on Known Unknown Data. On CIFAR+N, all representation learning methods utilizing known unknown data systematically outperform those that do not by up to 5.9 percentage points in AUOSCR with MSP. State-of-the-art ARPL achieves consistent but negligible improvements over CE for any given postprocessor. With known classes being held constant between CIFAR+10 and CIFAR+50, we can see that additional known unknown data consistently improves AUOSCR by over 2 percentage points, even when evaluated on more unknown classes. For all representation learning methods, the variations across postprocessors are comparably small, suggesting that representation learning contributes more toward overall performance than postprocessors on small-scale benchmarks, likely due to the limited diversity between known classes and the large distributional gap between known and unknown classes that can easily be exploited by representation learning methods with known unknown data. Only MaxLogits provides substantial gains, up to 2.2 percentage points of AUOSCR over MSP for CE.

![](images/1f73e27dba5bfb6c141fff7b48e2e1d90b8fa403777ea402d18b0ba88f2d82cb.jpg)  
Figure 4: AGGREGATED RESULTS. The representation learning performance contribution $\Delta _ { f } ( f \mid g )$ in AUOSCR is shown as distributions over postprocessors and across protocols. Methods that use known unknown data are marked with blue background. CIFAR+N results are averaged over 5 trials.

Performance Degradation at Large Scale. On large-scale ImageNet protocols, this outperformance from using known unknown data vanishes, even underperforming the $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ baseline on $P _ { 3 }$ for most postprocessors, in line with Wang et al. [B]. Figure 4 depicts the representation learning contribution deltas $\Delta _ { f } ( f _ { \mathrm { i } } | g _ { \mathrm { j } } )$ for each representation learning method i across datasets as distribution over all postprocessors $j .$ The representation learning contribution delta for MM methods OE and OS degrades and turns negative with increasing similarity of known and known unknown classes on $P _ { 2 }$ and $P _ { 3 }$ , destroying performance across most postprocessors. Wang et al. [B] attribute their findings of poor OE performance to low correlation between known unknown and unknown classes or high correlation between known and unknown classes. However, the K + 1 class classifier does not experience this performance degradation with identical data, demonstrating that it cannot be explained by the data distributions alone but by the interplay between data distributions and training objective. Strong OOD detection performance with MSP (see AUROC in Figure 13c) and high correlation between AUOSCR and closed-set accuracy (cf. Figure 13d in Supplemental) suggest that the performance degradation can partially be explained by a loss of discriminative information for known classes

![](images/534cffd49f2932896479faf814295e129a6f93ab45cca968a27b7a461f1422ac.jpg)  
Figure 5: FEATURE MAGNITUDES. We plot feature magnitude distributions of known, known unknown, and unknown classes on protocols $P _ { 1 } , \ P _ { 3 }$ , and EMNIST. OE and OS experience feature magnitude collapse on $P _ { 3 }$ and EMNIST, pulling their feature magnitudes towards zero. For EMNIST, we show distributions for known classes 1 and 9 (black and grey) that are highly similar to known unknown classes, and other known classes (blue) separately.

Magnitude Collapse.To understand why MM methods degrade in accuracy, AU-OSCR, and OOSA, we analyze the feature magnitude distributions of known, known unknown, and unknown samples (Figure 5). With highly similar known unknown samples, MM methods inadvertently draw features of known classes towards the origin of the feature space, resulting in magnitude collapse and effectively overlapping them with known unknown, unknown, and other known samples. We analyze the relationship between feature magnitude and class-wise CCR [] at the operational threshold, $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ , see Appendix A.6, and perform linear regression3 against the class-wise average feature magnitude for each known class c (Figure 15 in the Supplemental). On CIFAR+N, only MM methods exhibit a statistically significant positive correlation between average feature magnitude and $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ , with large effect sizes of $R ^ { 2 }$ up to 80%. On the easily separable $P _ { 1 }$ , most models exhibit significant positive correlations, but often with meaningless effect sizes below 10%. However, with increasing similarity in known and known unknown samples on $P _ { 2 }$ and $P _ { 3 }$ , MM methods learn much stronger and statistically significant relationships, resulting in practically significant effect sizes of $R ^ { 2 }$ up to 38% on $P _ { 3 }$ (Figure 6). This shows how magnitude-manipulation increases the dependency between feature magnitude and CCR by systematically reducing performance on a few classes in a trade-off to maintain overall high binary ID-vs-OOD separation via MSP (see AUROC Figure 13c) and increasing the minimal class-wise CCR.

In contrast, K + 1 class counteracts magnitude collapse by learning sufficiently large feature magnitudes to achieve high SoftMax probabilities for the additional output node during training. Since $K + 1$ class is trained via CE loss, it forces probabilities corresponding to any output node (known or known unknown) to be close to 1 for samples of the respective class. Formally, we want the trained network to be able to achieve $\mathbf { y } _ { k } > 1 - \epsilon$ for any sample and output class $c \in \{ 1 , \ldots , K \}$ , for some small $\epsilon > 0$ . This holds if (but not only if) the difference between the maximal two logits surpasses a lower bound l, which is equivalent to (see Appendix A.3):

![](images/dd5225b7434f35bffb6f07ad76d84b68676cc99306515bf0f45044d6982738e0.jpg)  
Figure 6: CCR VS. MAGNITUDE. We exhibit a linear relationship between classwise CCR, $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ , at the operational threshold and class-wise average feature magnitude for known classes on $P _ { 3 }$ . We perform regression for each representation learning method independently and over all postprocessors.

$$
\| \varphi \| _ { 2 } > \frac { l } { \delta _ { \mathbf { z } _ { c } / \| \varphi \| _ { 2 } } } = \frac { \log ( \frac { 1 - \epsilon } { \epsilon } ) + \log ( K - 1 ) } { \| \mathbf { W } _ { c } \| _ { 2 } \cos ( \alpha _ { c } ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c } \{ \| \mathbf { W } _ { c ^ { \prime } } \| _ { 2 } \cos ( \alpha _ { c ^ { \prime } } ) \} } \Longrightarrow \mathbf { y } _ { c } > 1 - \epsilon ,\tag{8}
$$

where $\delta _ { \mathbf { z } _ { c } \parallel \varphi \parallel _ { 2 } }$ is the difference logits divided by their feature magnitude. Note that l is constant for any dataset, loss and €, and $\delta _ { \mathbf { z } _ { c } / \parallel \varphi \parallel _ { 2 } }$ is generally small because weight magnitudes are minimized through weight decay and the cosines are bounded between -1 and 1. This provides upward pressure on feature magnitudes during training and requires deep feature magnitudes to exceed a lower bound for any sample given a trained classifier. In other words, learning sufficiently large feature magnitudes is a sufficient condition to achieve high confidence for some class given trained classifier weights W. This training incentive prevents magnitude collapse in K + 1 class and ensures that K +1 class learns feature magnitude distributions similar to CE and ARPL. Surprisingly, despite training on known unknown data, K + 1 class learns lower feature magnitudes for known unknown and unknown classes while maintaining sufficiently large feature magnitudes to achieve high confidences for known and known unknown classes. The difference in feature magnitudes between known, known unknown, and unknown classes is analogous to CE but slightly more pronounced on $P _ { 3 }$ , see Figure 12. This provides theoretical support for combinations of $K + 1$ class with MA postprocessors, because high level magnitude distribution statistics are similar to CE, for which those postprocessors were designed.

Qualitative Analysis on Class Similarity. We replicate and qualitatively investigate the feature collapse for MM methods at small scale by curating a hard

![](images/1e529efdd5ad6cdd677b9a06ca7e03446965b76d154164bb73299ff54743edf7.jpg)  
Figure 7: INTERACTION EFFECTS. We indicate interaction effects of representation learning and postprocessing components as correlation between representation learning contribution $\Delta _ { f } ( f \mid g )$ and postprocessor contribution $\Delta _ { g } ( g \mid f )$ for AUOSCR (top row) and OOSA (bottom row).

EMNIST benchmark that includes known unknown classes which are highly similar to known class 1 and often indistinguishable to class 9 without context (see Appendix A.4 for details). Experiments are identical to CIFAR+N but networks are trained for 50 epochs. Observable from Figure 5 (right), for MM methods, the feature representations of digits 1 and 9 are pulled towards the origin. We compute $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ per representation learning method for classes 1, 9, and rest (all other known classes). For OS and OE combined, these classes experience $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ in the range [58.1%, 88.7%] (digit 1) and [93.6%, 97.2%] (digit 9), while other digits achieve over 96.7% CCR, showing dramatically imbalanced detection performance on known classes. See Table 5 in the Supplemental for all representation learning methods. Non-MM methods have $\mathrm { C C R } _ { k } ( \theta ^ { \star } )$ evenly distributed between 83.8% and 98% over all postprocessors. This demonstrates how high similarity between known and known unknown classes, combined with magnitude-manipulation, irreparably damages feature representations and ruins classification of visually similar known classes.

## 5.2 Interaction Effects

The second set of experiments aims to answer whether the optimal postprocessor should be informed by the representation learning method, and whether magnitudemanipulating representation learning can enhance the performance of magnitudeaware postprocessors. We analyze the relationship and interactions between representation learning and postprocessor contributions by plotting the representation learning contribution delta $\Delta _ { f } ( f \mid g )$ against the postprocessing contribution delta $\Delta _ { g } ( g \mid f )$ for each combination $g \circ f$ in Figure 7.

Independent Contributions Without Known Unknown Data. Across all evaluation metrics and protocols, all experiments reveal that postprocessor contributions are almost perfectly independent from representation learning contributions, when trained without known unknown data, $e . g .$ , the postprocessor contribution of GHOST is independent of the used representation learning method: $\Delta _ { g } ( g _ { \mathrm { G H O S T } } \mid$ $f _ { \mathrm { C E } } ) \approx \Delta _ { g } ( g _ { \mathrm { G H O S T } } \mid f _ { \mathrm { A R P L } } ) = 1 . 5$ . This suggests that OSR system components have separate responsibilities when trained on known classes only, with representation learning addressing the ID classification and postprocessors addressing the open-set capabilities.1 This allows to freely combine representation learning and postprocessor, with magnitude-aware postprocessors consistently favored, supporting Wang et al. [B], to achieve performance gains without risking undesirable interactions.

Interaction Effects With Known Unknown Data. Representation learning methods that train with known unknown data only show interaction effects with high similarity of known and known unknown classes. Interaction effects are characterized by a correlation between $\Delta _ { f } ( f \mid g )$ and $\Delta _ { g } ( g \mid f )$ in Figure 7. On ImageNet benchmarks, representation learning methods with known unknown data show interaction effects, with particularly MA postprocessors strongly amplifying positive and negative gains from representation learning. While magnitude-aware postprocessors consistently outperform others for non-MM methods, they can amplify performance degradation for MM methods due to their sensitivity to the magnitude collapse. In particular, PostMax, and to a lesser extent GHOST, demonstrate both the highest gains for non-MM methods (8.3 percentage points or +11.7% for K +1 class), as well as the most severe performance degradation for MM methods (-1.2 percentage points or -1.5% for OS) on $P _ { 3 }$ . These interaction effects are even clearer when evaluated via OOSA, which equally weights known and unknown test samples.

## 5.3 Robustness to Backbone Architecture

To ensure that our findings are not specific to the ResNet architecture, we replicate the experiments on $P _ { 2 }$ using a Swin-B transformer []. Protocol $P _ { 2 }$ balances difficulty and computational cost and provides a good trade-off between the two [], allowing us to evaluate the generalizability of our findings to a transformer-based architecture. Training is conducted with identical hyperparameters to the ResNet-50 experiments, except for a reduced learning rate of $2 e - 4$ to account for a lower batch size of 90. We trained 300 epochs with the AdamW optimizer and weight decay of 0.05 to ensure convergence.

Overall we observe similar relative performance as with ResNet-50 (Figure 8). Particularly, the magnitude-manipulating methods OE and OS experience performance degradation to the $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ baseline when using MSP postprocessing, and combined with magnitude-aware postprocessors they experience further degradation or mixed postprocessing contributions when evaluated via OOSA. For AUOSCR and AUROC, the performance degradation is less severe. As expected, K +1 class systematically outperforms all other combinations when combined with magnitude-aware postprocessors across all metrics. The absolute performance of all methods is lower compared to ResNet-50 backbones, likely due to the increased data requirement of a transformer coupled with a larger model and limited data set sizes due to the protocols containing only a fraction of the ImageNet dataset classes. Interestingly, the training of the reciprocal points (ARPL) proved to be unstable under the AdamW optimizer, while the backbone training remained stable, resulting in significantly lower accuracy and consequently lower AUOSCR, OOSA, and AUROC. While future work is needed to to investigate better training methods for ARPL with transformers, we observe that magnitude-aware postprocessors nontheless can recover most of the performance gap to CE for binary ID-vs-OOD separation, as shown by the AUROC results, highlighting the practical importance and utility of a two-stage approach to OSR by combining representation learning with postprocessing.

![](images/7e2afd7a18b84d4fa24ff0519e87506004b447e4b85dc59fdd01e8f25aee5eb0.jpg)  
Figure 8: SwIN BACKBONE. We show OSR performance on P2 with a Swin-B backbone, showing accuracy, AUROC, and AUOSCR metrics. Similar interaction effects between representation learning and postprocessing methods are observed as with ResNet-50 backbones.

## 6 Conclusion

In this study, we adopt a two-stage framework for modularly combining representation learning and postprocessing methods for Open-Set Recognition (OSR). We show that representation learning without known unknown data leads to independent OSR components, that can be freely combined to achieve additive performance gains, whereas representation learning with known unknown data can experience interaction effects with postprocessors that can improve or degrade OSR performance. We explain this performance degradation and the key role of feature magnitude in interaction effects via the magnitude collapse mechanism, revealing several insights. First, the similarity between known unknown and known classes is a key factor for performance degradation at scale, a scenario that does not occur on small-scale benchmarks. Second, magnitude collapse creates an undesirable linear dependency between feature magnitude and class-wise detection performance on the in-distribution and OSR task, leading to systematically imbalanced detection across known classes. Furthermore, we demonstrate how magnitude collapse can be circumvented by the simple yet effective baseline K + 1 class and that, contrary to recent findings, training with known unknown data can improve OSR performance at any scale and regardless of known unknown data selection. Our findings invalidate current best practices in OSR, demonstrating that small-scale evaluations with known unknown data do not translate to large-scale performance. Generally, small-scale experiments are primarily driven by representation learning via inclusion of known unknown data and favor the simpler postprocessors like MaxLogits, while large-scale experiments benefit from combining representation learning and postprocessing, mainly through joining K + 1 class with MA postprocessors like PostMax or GHOST. As such, representation learning methods considered state-of-the-art based on CIFAR benchmarks, such as Outlier Exposure, can degrade significantly below baselines at scale.

## Acknowledgements

We thank the University of Zurich and the Digital Society Initiative for financing this research through a research scholarship. Furthermore, we thank the University of Zurich for providing the computational resources.

## References

[1] Abhijit Bendale and Terrance E. Boult. Towards open set deep networks. In Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016.

[2] Guangyao Chen, Limeng Qiao, Yemin Shi, Peixi Peng, Jia Li, Tiejun Huang, Shiliang Pu, and Yonghong Tian. Learning open set network with discriminative reciprocal points. In European Conference on Computer Vision (ECCV). Springer, 2020.

[3] Guangyao Chen, Peixi Peng, Xiangqian Wang, and Yonghong Tian. Adversarial reciprocal points learning for open set recognition. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 44(11), 2021.

[4] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. EM-NIST: Extending MNIST to handwritten letters. In International Joint Conference on Neural Networks (IJCNN), 2017.

[5] Steve Cruz, Ryan Rabinowitz, Manuel Günther, and Terrance E. Boult. Operational open-set recognition and postmax refinement. In European Conference on Computer Vision (ECCV), 2024.

[6] Akshay Raj Dhamija, Manuel Günther, and Terrance E. Boult. Reducing network agnostophobia. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[7] Akshay Raj Dhamija, Manuel Günther, Jonathan Ventura, and Terrance E. Boult. The overlooked elephant of object detection: Open set. In Winter Conference on Applications of Computer Vision (WACV), 2020.

[8] Andrija Djurisic, Nebojsa Bozanic, Arjun Ashok, and Rosanne Liu. Extremely Simple Activation Shaping for Out-of-Distribution Detection. In The Eleventh International Conference on Learning Representations, 2023.

[9] Zongyuan Ge, Sergey Demyanov, and Rahil Garnavi. Generative OpenMax for multi-class open set classification. In British Machine Vision Conference (BMVC), 2017.

[10] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations (ICLR), 2017.

[11] Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. In International Conference on Learning Representations (ICLR), 2019.

[12] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[13] Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joseph Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling outof-distribution detection for real-world settings. In International Conference on Machine Learning (ICML). PMLR, 2022.

[14] Hongzhi Huang, Yu Wang, Qinghua Hu, and Ming-Ming Cheng. Class-specific semantic reconstruction for open set recognition. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 45(4), 2023.

[15] Rui Huang, Andrew Geng, and Yixuan Li. On the importance of gradients for detecting distributional shifts in the wild. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[16] Shu Kong and Deva Ramanan. Opengan: Open-set recognition via open data generation. In International Conference on Computer Vision (ICCV), 2021.

[17] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[18] Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In International Conference on Learning Representations (ICLR), 2017.

[19] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-ofdistribution detection. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[20] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In International Conference on Computer Vision (ICCV), 2021.

[21] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2017.

[22] George A Miller. WordNet: An electronic lexical database. MIT press, 1998.

[23] Lawrence Neal, Matthew Olson, Xiaoli Fern, Weng-Keen Wong, and Fuxin Li. Open set learning with counterfactual images. In European Conference on Computer Vision (ECCV), 2018.

[24] Andres Palechor, Annesha Bhoumik, and Manuel Günther. Large-scale open-set classification protocols for imagenet. In Winter Conference on Applications of Computer Vision (WACV), 2023.

[25] Ryan Rabinowitz, Steve Cruz, Manuel Günther, and Terrance E. Boult. GHOST: Gaussian hypothesis open-set technique. In AAAI Conference on Artificial Intelligence, 2025.

[26] Jie Ren, Peter J Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan. Likelihood ratios for outof-distribution detection. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

[27] Ethan M. Rudd, Lalit P. Jain, Walter J. Scheirer, and Terrance E. Boult. The extreme value machine. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017.

[28] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision (IJCV), 115(3), 2015.

[29] Walter J. Scheirer, Anderson de Rezende Rocha, Archana Sapkota, and Terrance E. Boult. Towards open set recognition. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 35(7), 2013.

[30] Walter J. Scheirer, Lalit P. Jain, and Terrance E. Boult. Probability models for open set recognition. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 36(11), 2014.

[31] Yiyou Sun, Chuan Guo, and Yixuan Li. ReAct: Out-of-distribution detection with rectified activations. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[32] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zissermann. Open-set recognition: A good closed-set classifier is all you need? In International Conference on Learning Representations (ICLR), 2022.

[33] Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, and Yoshua Bengio. Manifold mixup: Better representations by interpolating hidden states. In International Conference on Machine Learning (ICML). PMLR, 2019.

[34] Hongjun Wang, Sagar Vaze, and Kai Han. Dissecting out-of-distribution detection and open-set recognition: A critical analysis of methods and benchmarks. International Journal of Computer Vision (IJCV), 133(3), 2025.

[35] Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, and Qingming Huang. Openauc: Towards auc-oriented open-set recognition. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[36] Samuel Wilson, Tobias Fischer, Feras Dayoub, Dimitry Miller, and Niko Sünderhauf. SAFE: Sensitivity-aware features for out-of-distribution object detection. In International Conference on Computer Vision (ICCV), 2023.

[37] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized outof-distribution detection: A survey. International Journal of Computer Vision (IJCV), 132(12), 2024.

[38] Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations (ICLR), 2018.

[39] Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Learning placeholders for openset recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

## A Supplemental

## A.1 Method Selection and Background

An overall view of the two-stage processing pipeline, including the nomenclature as used in this section, is given in Figure 2. Below, we provide details for the representation learning and postprocessors that we investigate in this work as well as a discussion on method selection to answer our research questions.

## A.1.1 Representation Learning Methods

Methods Selection. Our model selection for representation learning aims to cover a representative set of methods that reflect the current state of the art in OSR as well as OOD detection while simultaneously adhering to a set of constraints given by our rigorous experimental setup to ensure that we can adequately test our research questions and hypotheses. We selected representation learning methods that satisfy the following criteria:

1. We require methods that utilize no or only natural known unknown samples as this is at the core of our analysis.

2. Our modularization of OSR methods requires classification-based (discriminative) representation learning methods since most postprocessors rely on output logits or probabilities to perform classification.

3. representation learning methods must be able to be trained from scratch on the ImageNet protocols without pre-training in order to prevent information leakage from training on classes that our protocols declare as unknown.

Following, we discuss each method and its formalization in the two-stage framework in more detail. For a systematic overview including method characteristics, see Table 1a

CE. Our baseline [] training-based approach is the categorical Cross-Entropy (CE) loss trained only on samples from K known classes. For an input sample $( \mathbf { x } _ { n } , \tau _ { n } ) \in \mathcal { K }$ and an arbitrary backbone network we obtain the deep features $\varphi _ { n } \in \mathbb { R } ^ { D }$ for some deep feature dimension D. These features are then passed through a fullyconnected logit layer $\mathbf { W } \in \mathbb { R } ^ { C \times D }$ with $C = K$ output logits $\mathbf { z } _ { n } = \mathbf { W } \varphi _ { n } \in \mathbf { \bar { \mathbb { R } } } ^ { C } .$ 4 The logits are then turned into probabilities $\mathbf { y } _ { n } \in \mathbb { R } ^ { K }$ through SoftMax activation:

$$
y _ { n , c } = \frac { e ^ { z _ { n , c } } } { \displaystyle \sum _ { c ^ { \prime } = 1 } ^ { C } e ^ { z _ { n , c ^ { \prime } } } } .\tag{9}
$$

Based on these, the CE loss is computed as:

$$
\mathcal { T } _ { \mathrm { C E } } = - \mathbb { E } _ { ( \mathbf { x } _ { n } , \tau _ { n } ) \in \mathcal { K } } \log y _ { n , \tau _ { n } } .\tag{10}
$$

OE. We include Outlier Exposure (OE) [] as state-of-the-art representation learning method from the OOD detection literature that utilize known unknown data [B]. OE adds a regularization term to (10) that maximizes the entropy for known unknown samples by computing the CE loss between the uniform distribution and the SoftMax confidences of the network:

$$
\mathcal { I } _ { \mathrm { O E } } = \mathcal { I } _ { \mathrm { C E } } - \lambda _ { \mathrm { O E } } \mathbb { E } _ { ( \mathbf { x } _ { n } , \tau _ { n } ) \in \mathcal { A } } \ \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \log y _ { n , c }\tag{11}
$$

OE essentially is equivalent to the Entropic Open-Set (EOS) loss $\mathcal { I } _ { \mathrm { E O S } }$ proposed by [], with the only exception that OE provides a more intuitive way to weight the impact of known unknown samples to the training (with $\lambda _ { \mathrm { O E } } = 0 . 5$ for computer vision tasks), whereas EOS does so via class weighting (set to 1). From [] we know that EOS and, by extension, OE implicitly manipulate feature magnitudes, by encouraging the network to learn small feature magnitudes for known unknown samples and large magnitudes for known samples.

OS. We employ the ObjectoSphere (OS) loss [] as an extension to the EOS loss which explicitly manipulates feature magnitudes. It learns vanishing vectors for known unknown samples and large magnitudes for known samples by using the following regularization term combined with the EOS loss:

$$
\mathcal { I } _ { \mathrm { O S } } = \mathcal { I } _ { \mathrm { E O S } } + \lambda _ { \mathrm { O S } } \left\{ \begin{array} { l l } { \operatorname* { m a x } \hfill \left( 0 , \xi - \| \varphi _ { n } \| _ { 2 } ^ { 2 } \right) } & { \mathrm { i f ~ } \mathbf { x } _ { n } \in K } \\ { \| \varphi _ { n } \| _ { 2 } ^ { 2 } } & { \mathrm { i f ~ } \mathbf { x } _ { n } \in \mathcal { A } } \end{array} \right.\tag{12}
$$

where the hyperparameter $\xi$ is the lower bound for the feature magnitude of known samples.

$K + 1$ class. We use a $K + 1$ class classifier as representation learning baseline for OSR and OOD methods that utilize known unknown data. $K + 1$ class uses the known data K as in CE, and known unknown data A to train an additional output node $z _ { n , K + 1 }$ , so that we have a total of $C = K + 1$ outputs, creating a default class for all known unknown and unknown samples. It is trained with the standard CE loss (10) with label $\tau = K + 1$ for known unknown samples. While this is a common approach in object detection models [], which collect a lot of background samples, it is only rarely applied directly to OSR problems [, ] or as component of more complex architectures such as PROSER or G-OpenMax [, 9]. Depending on the context, a $K + 1$ class classifier is known as Background Class [, , ] or Dummy Classifier [9]. Note that for computing OOD scores $\gamma _ { n } ^ { * }$ (see A.1.2) for the $K + 1$ class classifier, we follow Ren et al. [ and ignore the unknown class $\left( y _ { n , K + 1 } \mathrm { o r } z _ { n , K + 1 } \right)$ if it exists. A low probability for the unknown class $y _ { n , K + 1 }$ does not indicate a high probability for any of the known classes. On the other hand, due to Softmax requiring probabilities sum up to 1, a large probability $y _ { n , K + 1 }$ enforces low probabilities for all known classes. Therefore, $y _ { n , K + 1 }$ does not add any new information.

ARPL. Finally, we include the Adversarial Reciprocal Point Learning (ARPL) loss [B], which currently is the state of the art for $\mathrm { O S R ^ { 5 } }$ and trained solely on the known data $\kappa .$ Unlike most training-based methods, ARPL does not learn a prototype for a class $c ,$ but a reciprocal point $\mathbf { p } _ { c } \in \mathbb { R } ^ { D }$ in deep feature space, i. e., a point that represents everything but class c. The ARPL objective maximizes the distance between the reciprocal point and the features of samples from the respective class by computing logits $\mathbf { z } _ { n }$ via distance measures between deep features $\varphi _ { n }$ and the reciprocal points $\mathbf { p } _ { c } \mathbf { : }$

$$
z _ { n , c } = \frac { 1 } { D } \| \varphi _ { n } - { \bf p } _ { c } \| _ { 2 } ^ { 2 } - \varphi _ { n } ^ { \mathrm { T } } { \bf p } _ { c } ,\tag{13}
$$

which are used via SoftMax (9) in the CE loss (10). To constrain open space, the distance between deep features and reciprocal points is bound by a learnable constant $\rho$ via the regularization term with weight $\lambda _ { \mathrm { A R P L } } = 0 . 1$ ..

$$
\mathcal { I } _ { \mathrm { A R P L } } = \mathcal { I } _ { \mathrm { C E } } + \lambda _ { \mathrm { A R P L } } \mathbb { E } _ { ( \mathbf { x } _ { n } , \tau _ { n } ) \in K } \operatorname* { m a x } \left( 0 , \frac { 1 } { D } \| \varphi _ { n } - \mathbf { p } _ { \tau _ { n } } \| _ { 2 } ^ { 2 } - \rho \right)\tag{14}
$$

## A.1.2 Postprocessing Methods

Method Selection. We selected postprocessors to cover a split between representative magnitude-aware and magnitude-unaware methods. Furthermore, we aim to cover both trainable and non-trainable methods to include methods that can specifically adjust to differently learned representations. Following, we discuss each method and its formalization in the two-stage framework in more detail. For a systematic overview including method characteristics, see Table 1b.

MSP. The Maximum Softmax Probability (MSP) [] is the de-facto default postprocessor for OSR and OOD detection and serves as our baseline. Class predictions and OOD scores are computed from probabilities for known classes as:

$$
k _ { n } ^ { * } = \mathop { \mathrm { a r g m a x } } _ { 1 \leq c \leq K } y _ { n , c } \qquad \mathrm { a n d } \qquad \gamma _ { n } ^ { * } = y _ { n , k _ { n } ^ { * } } .\tag{15}
$$

MSP is not magnitude-aware since SoftMax (9) normalizes the logits and only considers their relative differences.

MaxLogits. MaxLogits [] or MLS [] exploit the magnitude of the logits $\mathbf { z } _ { n }$ which contains useful information for OSR and OOD detection that is lost during softmax. MLS is magnitude-aware since logit magnitude is linked to feature magnitude [B].4 Class predictions and OOD scores are computed from known logits as:

$$
k _ { n } ^ { * } = \mathop { \mathrm { a r g m a x } } _ { 1 \leq c \leq K } z _ { n , c } \qquad \mathrm { a n d } \qquad \gamma _ { n } ^ { * } = z _ { n , k _ { n } ^ { * } } .\tag{16}
$$

OpenMax [] uses deep features $\varphi _ { n }$ , referred to as Activation Vectors (AVs), to statistically model probabilities for an additional output node for the unknown class. During training, for each known class $^ { c , }$ the Mean Activation Vector (MAV) $\mu _ { c }$ is computed by averaging the deep features $\varphi _ { n }$ extracted from all correctly classified training samples. Then, the cosine distances of the MAV $\mu _ { c }$ to all the AVs $\varphi _ { n , c }$ of the same class are computed. Here, we make use of a twist implemented in the VAST package of the original authors:6 instead of using the original distances to model the distribution, we multiply the cosine distances by a factor $\kappa ,$ which allows modeling more compact class representations:

$$
d _ { n , c } = \kappa ( 1 - \cos ( \varphi _ { n } , \mu _ { c } ) )\tag{17}
$$

Then, a per-class Weibull distribution $\Psi _ { c }$ is fitted to the top λ largest cosine distances $d _ { n , c } .$ For features $\varphi _ { n }$ of a test sample, the class-wise Weibull distributions estimate a logit $\hat { z } _ { n , K + 1 }$ for the unknown class, as well as modifying the logits for the top α classes, giving newly estimated logits $\hat { z } _ { n , c }$ . From these, the output $\hat { y } _ { n , c }$ is computed via softmax (9) and the output $\mathcal { P } _ { n }$ is computed as:

$$
k _ { n } ^ { * } = \mathop { \mathrm { a r g m a x } } _ { 1 \leq c \leq K } \hat { z } _ { n , c } \qquad \mathrm { a n d } \qquad \gamma _ { n } ^ { * } = \hat { y } _ { n , k _ { n } ^ { * } } .\tag{18}
$$

OpenMax is not magnitude-aware since the cosine distance ignores the feature magnitude.

PostMax. Postnormalization of Maxima (PostMax) [] uses Extreme Value Theory (EVT) by applying a Generalized Pareto Distribution (GPD) to maximum logits post-normalized by the feature magnitude. Based on their findings that unknown samples have larger feature magnitude than known samples on large-scale data, they normalize logits by dividing them by their deep feature magnitude to further increase the separation between known and unknown samples:

$$
\hat { \mathbf { z } } _ { n } = \frac { \mathbf { z } _ { n } } { \| \varphi _ { n } \| _ { 2 } + 1 } .\tag{19}
$$

This makes it an explicitly magnitude-aware method. We modify the original implementation to shift the magnitudes by 1 to avoid issues with magnitudes $\| \varphi \| _ { 2 } < 1$ which reverse the desired effect of normalization.7 This occurs consistently for features from magnitude-manipulating representation learning methods, but not for others. The class-agnostic GPD $\Psi _ { \mu , \sigma , \xi }$ is fitted only on the maximum normalized logits of correctly classified known training samples, which is then used to compute a probability of the sample being known. Following [], the scores $\mathcal { P } _ { n }$ are computed as:

$$
k _ { n } ^ { * } = \arg \operatorname* { m a x } _ { 1 \leq c \leq K } \hat { z } _ { n , c } \qquad \mathrm { a n d } \qquad \gamma _ { n } ^ { * } = \Psi _ { \mu , \sigma , \xi } \big ( \hat { z } _ { n , k _ { n } ^ { * } } \big ) .\tag{20}
$$

GHOST. The Gaussian Hypothesis Open-Set Technique (GHOST) [] models each class in deep feature space as a multivariate Gaussian distribution with the intuition that features $\varphi$ from known and unknown samples deviate by feature magnitude even if the angular direction overlaps, making it magnitude-aware. During training, GHOST fits a Gaussian distribution $( \mu _ { c } , \sigma _ { c } )$ for each known class c based on the deep features $\varphi _ { n }$ of correctly classified training samples. During inference, GHOST first computes a class prediction from the logits:

$$
k _ { n } ^ { * } = \underset { 1 \leq c \leq K } { \mathrm { a r g m a x } } z _ { n , c } ,\tag{21}
$$

as well as z-score $s _ { n }$ for each sample ${ \bf x } _ { n }$ and corresponding Gaussian $( \mu _ { k _ { n } ^ { * } } , \sigma _ { k _ { n } ^ { * } } )$ . This is then used to compute the score $\gamma _ { n } ^ { * }$ by dividing the original logit:

$$
s _ { n } = \sum _ { d = 1 } ^ { D } \frac { \lvert \varphi _ { n , d } - \mu _ { k _ { n } ^ { * } , d } \rvert } { \sigma _ { k _ { n } ^ { * } , d } } , \qquad \gamma _ { n } ^ { * } = \frac { z _ { n , k _ { n } ^ { * } } } { s _ { n } } .\tag{22}
$$

## A.2 Derivation of Performance Decomposition

Here we provide the derivation for the performance decomposition of the performance increase of a $g _ { \mathrm { j } } \circ f _ { \mathrm { i } }$ over $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ into the sum of the individual performance gains of the representation learning method and postprocessor. For the decomposition as in Equation (4) we have: We start with the performance of $g _ { \mathrm { j } } \circ f _ { \mathrm { i } }$ and add and subtract the baseline $g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } }$ and the intermediate $g _ { \mathrm { M S P } } \circ f _ { \mathrm { i } }$

$$
\begin{array} { r l } & { \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) = \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) + \underbrace { \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) } _ { = 0 } } \\ & { \qquad + \underbrace { \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { i } } ) } _ { = 0 } } \end{array}\tag{23}
$$

(24)

By rearanging and using definitions of the contribution deltas $\Delta _ { f } ( f \mid g )$ and $\Delta _ { g } ( g \mid f )$ (Equations (2) and (3)) we have:

$$
\begin{array} { r l } { } & { = \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) + \underbrace { \left( \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) \right) } _ { = \Delta _ { f } ( f _ { \mathrm { i } } | g _ { \mathrm { M S P } } ) } } \\ { } & { + \underbrace { \left( \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { i } } ) \right) } _ { = \Delta _ { g } ( g _ { \mathrm { j } } | f _ { \mathrm { i } } ) } } \end{array}\tag{25}
$$

(26)

By rearranging we have:

$$
\Longleftrightarrow \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) = \Delta _ { f } ( f _ { \mathrm { i } } \mid g _ { \mathrm { M S P } } ) + \Delta _ { g } ( g _ { \mathrm { j } } \mid f _ { \mathrm { i } } )\tag{27}
$$

Analogously for the decomposition as in Equation (5) we have: We start with the performance of $g _ { \mathrm { j } } \circ f _ { \mathrm { i } }$ and add and subtract the baseline gMsp  fcE and the intermediate $g _ { \mathrm { j } } \circ f _ { \mathrm { C E } }$

$$
{ \mathcal { M } } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) = { \mathcal { M } } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) + \underbrace { { \mathcal { M } } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) - { \mathcal { M } } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) } _ { = 0 }\tag{28}
$$

$$
+ \underbrace { \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { C E } } ) - \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { C E } } ) } _ { = 0 }\tag{29}
$$

By rearanging and using definitions of the contribution deltas $\Delta _ { f } ( f \mid g )$ and $\Delta _ { g } ( g \mid f )$ (Equations (2) and (3)) we have:

$$
\begin{array} { r l } & { = \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) + \underbrace { ( \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { C E } } ) ) } _ { = \Delta _ { f } ( f _ { \mathrm { i } } | g _ { \mathrm { j } } ) } } \\ & { + \underbrace { ( \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { C E } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) ) } _ { = \Delta _ { g } ( g _ { \mathrm { j } } | f _ { \mathrm { C E } } ) } } \\ & { \iff \mathcal { M } ( g _ { \mathrm { j } } \circ f _ { \mathrm { i } } ) - \mathcal { M } ( g _ { \mathrm { M S P } } \circ f _ { \mathrm { C E } } ) = \Delta _ { f } ( f _ { \mathrm { i } } | g _ { \mathrm { j } } ) + \Delta _ { g } ( g _ { \mathrm { j } } | f _ { \mathrm { C E } } ) } \end{array}\tag{30}
$$

(31)

(32)

## A.3 Derivation for Feature Magnitude Incentive

Here, we provide the derivation for the training incentive of CE with $C = K$ (or $C = K + 1 )$ to learn “sufficiently large" feature magnitudes:

$$
\| \varphi \| _ { 2 } > \frac { l } { \delta _ { \mathbf { z } _ { c } / \| \varphi \| _ { 2 } } } = \frac { \log ( \frac { 1 - \epsilon } { \epsilon } ) + \log ( C - 1 ) } { \| \mathbf { W } _ { c } \| _ { 2 } \cos ( \alpha _ { c } ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c } \{ \| \mathbf { W } _ { c ^ { \prime } } \| _ { 2 } \cos ( \alpha _ { c ^ { \prime } } ) \} } \Longrightarrow \mathbf { y } _ { c } > 1 - \epsilon ,
$$

For some small $\epsilon > 0$ and class $c \in \{ 1 , \ldots , C \}$ , we want the trained network to be able to achieve the maximum probability over all classes $\mathbf { y } _ { c } > 1 - \epsilon$ for any sample. Note that we have $\mathbf { y } _ { c } = \operatorname* { m a x } _ { c ^ { \prime } \in \{ 1 , \dots , C \} } \mathbf { y } _ { c ^ { \prime } }$

$$
\begin{array} { r l } { y _ { c } > 1 - \epsilon \Longleftrightarrow } & { \frac { e ^ { 2 \epsilon } } { \sum e ^ { \epsilon } e ^ { \epsilon } } = \frac { e ^ { 2 \epsilon } } { e ^ { \epsilon } + \sum _ { \gamma ^ { \prime } \neq c } e ^ { \epsilon } e ^ { \epsilon } } > 1 - \epsilon } \\ & { \Longleftrightarrow e ^ { \epsilon } < ( 1 - \epsilon ) ( e ^ { \epsilon } + \sum _ { \gamma } e ^ { \epsilon } ) } \\ & { \Longleftrightarrow e ^ { \epsilon } > e ^ { \epsilon } < \sum _ { \epsilon } \sum _ { \ell ^ { \prime } \neq c } e ^ { \epsilon } } \\ & { \Longleftrightarrow e ^ { \epsilon } > e ^ { \epsilon } < \sum _ { \ell ^ { \prime } \neq c } e ^ { \epsilon } e ^ { \epsilon } - \epsilon \sum _ { \ell ^ { \prime } \neq c } e ^ { \epsilon } } \\ & { \Longleftrightarrow e ^ { \epsilon } > \frac { 1 - \epsilon } { \epsilon } \sum _ { \ell ^ { \prime } \neq c } e ^ { \epsilon } , } \\ & { \Longleftrightarrow \pi _ { c } > \underbrace { \log \left( \frac { 1 - \epsilon } { \epsilon } \right) } _ { = \pi ^ { 3 } } + \log \left( \sum _ { \ell ^ { \prime } \neq c } e ^ { \epsilon } \right) } \end{array}
$$

By binding the LogSumExp log $\left( \Sigma _ { c ^ { \prime } \neq c } e ^ { \mathbf { z } _ { c ^ { \prime } } } \right)$ above, we get a sufficient condition $\delta _ { \mathbf { z } _ { c } } > l$ such that $\delta _ { \mathbf { z } _ { c } } > l \Longrightarrow \mathbf { y } _ { c } > \dot { 1 } - \epsilon$ . We háve the upper bound of the LogSumExp function:

$$
\log \left( \sum _ { c ^ { \prime } \neq c } e ^ { \mathbf { z } _ { c ^ { \prime } } } \right) \leq \log ( C - 1 ) + \operatorname* { m a x } _ { c \neq c ^ { \prime } } { } \mathbf { z } _ { c ^ { \prime } }
$$

$$
\mathbf { z } _ { c } > \beta + \operatorname* { m a x } _ { c \neq c ^ { \prime } } \mathbf { z } _ { c ^ { \prime } } + \log ( C - 1 ) \Longrightarrow \mathbf { z } _ { c } > \beta + \log \left( \sum _ { c ^ { \prime } \neq c } e ^ { \mathbf { z } _ { c ^ { \prime } } } \right)
$$

From this we get:

$$
\mathbf { z } _ { c } > \beta + \operatorname* { m a x } _ { c \neq c ^ { \prime } } \mathbf { z } _ { c ^ { \prime } } + \log ( C - 1 ) \iff \underbrace { \mathbf { z } _ { c } - \operatorname* { m a x } _ { c \neq c ^ { \prime } } \mathbf { z } _ { c ^ { \prime } } } _ { = : \delta _ { \mathbf { z } _ { c } } } > \beta + \log ( C - 1 ) = : b
$$

This provides a lower bound on the difference between the maximal two logits. Using footnote 4 we have

$$
\begin{array} { r l } & { \iff \| \varphi \| _ { 2 } \| \mathbf { W } _ { c } \| _ { 2 } \cos ( \alpha _ { \varphi , \mathbf { W } _ { c } } ) - \underset { c ^ { \prime } \neq c } { \operatorname* { m a x } } \{ \| \varphi \| _ { 2 } \| \mathbf { W } _ { c ^ { \prime } } \| _ { 2 } \cos ( \alpha _ { \varphi , \mathbf { W } _ { c ^ { \prime } } } ) \} > l } \\ & { \iff \| \varphi \| _ { 2 } \underbrace { ( \| \mathbf { W } _ { c } \| _ { 2 } \cos ( \alpha _ { \varphi , \mathbf { W } _ { c } } ) - \underset { c ^ { \prime } \neq c } { \operatorname* { m a x } } \| \mathbf { W } _ { c ^ { \prime } } \| _ { 2 } \cos ( \alpha _ { \varphi , \mathbf { W } _ { c ^ { \prime } } } ) \} ) } _ { = : \delta _ { \mathbf { z } _ { c ^ { \prime } } \| \varphi \| _ { 2 } } } > l } \\ & { \iff \| \varphi \| _ { 2 } > \frac { l } { \delta _ { \mathbf { z } _ { c ^ { \prime } } \| \varphi \| _ { 2 } } } } \end{array}
$$

where $\delta _ { \mathbf { z } _ { c } / \parallel \varphi \parallel _ { 2 } } > 0$

## A.4EMNIST Benchmark

We perform qualitative small-scale experiments on the EMNIST protocol [] to replicate settings of high similarity between known and known unknown classes at small scale. This benchmark contains a wide spectrum of clearly attributable visual similarities between known and known unknown classes because the samples do not contain any image background. It consists of EMNIST MNIST split as knowns, the first half of EMNIST letters (a-m) as known unknown data, and the second half (n-z) as unknowns.8 While most digits (known) do not contain any visually similar letters, digits 1 and 9 contain known unknown classes $( ^ { 6 } \mathrm { i } ^ { \dag } , \ ^ { 6 } \mathrm { l } ^ { \dag }$ and “g”, “q" respectively) that are visually very similar or indistinguishable without context depending on handwriting and capitalization.

To visualize how magnitude collapse and known class classification interacts, we repeat the EMNIST experiments with a 2D bottleneck deep feature layer, following Dhamija et al. []. These can illustrate how different representation learning methods structure the feature space and how this affects postprocessors, however, the low dimensionality limits not only the representational capacity but also changes the geometry of the feature space and the behavior, so extrapolation to high dimensional feature spaces is limited. Experiments are otherwise identical to those in Section 4.

## A.5 Hyperparameter Optimization

For most methods, we rely on the hyperparameters as provided by the according papers. Few methods however do not provide any, namely: λ and ξ for ObjectoSphere and λ, κ, and α for OpenMax, where we perform hyperparameter optimization on the validation set using grid search, based on the maximum AUOSCR. For OpenMax the parameter optimization is performed for each upstream representation learning method separately to ensure optimal settings, since the feature representations differ significantly. The considered hyperparameter ranges are as follows:

![](images/af70bed01dc68b51dfa6fdfeabca7ad5603222db77414e60431a2962a93e5440.jpg)

![](images/67861af290595b17f9548edbd6c79b5f12cfd470f244e1904fd242d29767ffd3.jpg)

![](images/facc953192847a42b3ab0f9149e0d1161c6de584afbce977ff447adefddede75.jpg)

![](images/d13b93ece9a08903dad26062304dc21bb1af64eb96a9415743c269d598303785.jpg)

![](images/fbd4bb0fa7a596f9a8d69a67029eee34032e7b9f94956b373a41ba0c579f6a74.jpg)  
0 1 2 3 4 5 6 7 8 9 Unknown Known Unknown

Figure 9: FEATURE VISUALIZATION. We provide 2D feature visualizations of EM-NIST experiments with different representation learning methods.
<table><tr><td></td><td></td><td colspan="2">CIFAR+N</td><td colspan="3">ImageNet</td></tr><tr><td>Method</td><td>Parameter</td><td>CIFAR+10</td><td>CIFAR+50</td><td> $P _ { 1 }$ </td><td> $P _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>fOS</td><td>λos</td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 2 } / 1 0 ^ { - 2 } / 1 0 ^ { - 2 } / 1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 2 } / 1 0 ^ { - 2 } / 1 0 ^ { - 3 } / 1 0 ^ { - 2 }$ </td><td>10−3</td><td>10⁻3</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>ξ</td><td>10/10/10/10/10</td><td>20/10/10/10/10</td><td>30</td><td>20</td><td>10</td></tr><tr><td>gOpenMax  fCE</td><td>α</td><td>3/3/3/1/3</td><td>3/2/3/3/3</td><td>5</td><td>5</td><td>2</td></tr><tr><td></td><td>κ</td><td>1.5/1.5/2.3/2/1.7</td><td>1.7/1.5/2.3/1.5/2.3</td><td>2.3</td><td>2.3</td><td>2</td></tr><tr><td></td><td>λ</td><td>100/100/250/10/100</td><td>250/100/500/100/100</td><td>750</td><td>750</td><td>100</td></tr><tr><td>gOpenMax  fARPL</td><td>α</td><td>3/3/1/3/3</td><td>3/2/3/3/3</td><td>5</td><td>3</td><td>2</td></tr><tr><td></td><td>κ</td><td>1.5/2.3/2/1.5/1.7</td><td>2/1.5/1.5/1.5/1.5</td><td>2.3</td><td>2.3</td><td>2.3</td></tr><tr><td></td><td>λ</td><td>100/250/10/100/100</td><td>100/100/100/100/100</td><td>750</td><td>750</td><td>10</td></tr><tr><td>gOpenMax  f K + 1 class</td><td>α</td><td>1/1/1/1/1</td><td>1/1/1/1/1</td><td>5</td><td>3</td><td>5</td></tr><tr><td></td><td>κ</td><td>1.5/2/2.3/1.7/2</td><td>1.7/2/1.5/1.5/2.3</td><td>1.7</td><td>2.3 750</td><td>2</td></tr><tr><td></td><td>λ</td><td>100/250/100/100/100</td><td>250/500/250/250/250</td><td>750</td><td></td><td>100</td></tr><tr><td>gOpenMax  fOE</td><td>α</td><td>1/2/1/3/3</td><td>3/3/1/2/1</td><td>3</td><td>2</td><td>2</td></tr><tr><td></td><td>κ</td><td>1.5/1.7/2/2/2</td><td>2.3/1.5/1.5/2/1.5</td><td>1.7</td><td>1.5</td><td>2.3</td></tr><tr><td></td><td>λ</td><td>10/100/10/100/250</td><td>100/100/10/100/10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>gOpenMax o fos</td><td>α</td><td>1/1/3/2/2</td><td>3/2/3/3/2</td><td>10</td><td>5 2.3</td><td>10</td></tr><tr><td></td><td>κ</td><td>1.5/2.3/1.5/1.5/2.3</td><td>2.3/1.7/2/2.3/2</td><td>2.3</td><td></td><td>2</td></tr><tr><td></td><td>λ</td><td>10/10/10/10/10</td><td>100/10/10/100/10</td><td>10</td><td>10</td><td>10</td></tr></table>

Table 2: HYPERPARAMETERs. This table include the optimized hyperparameter values for each method and protocol. CIFAR+N results are reported for each trial separately.

$$
\bullet \mathrm { O S } \colon \lambda \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \} , \xi \in \{ 1 0 , 2 0 , 3 0 \}
$$

• OpenMax: α ∈ {1, 2,3, 5, 10}, κ ∈ {1.5, 1.7,2, 2.3}, λ ∈ {10, 50, 100, 250, 500, 750, 1000}

The best hyperparameter values for each method and protocol are summarized in Table 2.

## A.6 Performance Metrics

CCR@FPR computes the CCR at a specific FPRs $\zeta \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 0 ^ { 0 } \}$ and provides insights into the classification performance at various tolerances for errors caused by missed rejections. It is therefore highly relevant for practical applications where a fixed threshold θ is required, which is typically selected based on a certain FPR. It is computed as:

$$
\mathrm { C C R @ F P R } = \left\{ \begin{array} { l l } { \mathrm { C C R } ( \theta _ { \zeta } ) } & { \mathrm { i f ~ } \theta _ { \zeta } \mathrm { ~ e x i s t s } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right. ,\tag{33}
$$

where $\theta _ { \zeta } = \mathrm { F P R } ^ { - 1 } ( \zeta )$ is the threshold that yields $\mathrm { F P R } = \zeta$ . CCR@FPR for $\zeta = 1 0 ^ { 0 }$ resembles closed-set accuracy.

AUOSCR. The OSCR curve [] simultaneously evaluates classification of known samples via CCR as well as the rejection of unknown samples via FPR over all possible thresholds. It is computed by varying the threshold θ from the smallest to the largest possible score value, and plotting the CCR over the FPR, $i . e . ,$ computing CCR@FPR at all thresholds. The Area Under the OSCR curve (AUOSCR) is computed by integrating the OSCR curve from $\zeta = 0 \ \mathrm { t o } \ \zeta = 1$ . Since the OSCR curve is a monotonically increasing function, the AUOSCR is maximized at and bounded by the closed-set accuracy.

OOSA. The Operational Open-set Accuracy (OOSA) [] evaluates the open-set performance at a fixed operational threshold $\theta ^ { \star }$ , determined on the validation set, and provides insights into the performance of a method in a real-world setting. It is defined as a trade-off between the CCR and the Unknown Rejection Rate (URR), $\mathrm { U R R } ( \theta ) = 1 - \mathrm { F P R } ( \theta )$

$$
\mathrm { O O S A } = \alpha _ { \mathrm { C C R } } \mathrm { C C R } ( \theta ^ { \star } ) + ( 1 - \alpha _ { \mathrm { C C R } } ) \mathrm { U R R } ( \theta ^ { \star } )\tag{34}
$$

where $\theta ^ { \star }$ is the operational threshold that maximizes this equation on the validation set. We follow Cruz et al. [] and set $\begin{array} { r } { \alpha _ { \mathrm { C C R } } = \frac { \vert \mathcal { K } _ { \mathrm { t e s t } } \vert } { \vert \mathcal { K } _ { \mathrm { t e s t } } \vert + \vert \mathcal { U } _ { \mathrm { t e s t } } \vert } } \end{array}$ to equally weight known and unknown test samples.

AUROC. In order to evaluate the OOD detection capabilities independently of the ID classification task, we use the Area Under the Receiver Operating Characteristics (AUROC) curve [, , , , , , ]. AUROC concerns how well known and unknown classes can be distinguished by computing FPR (6), as well as the True Positive Rate (TPR):

$$
\mathrm { T P R } ( \theta ) = \frac { \left| \left\{ ( \mathbf { x } _ { n } , \tau _ { n } ) \in K \wedge \gamma _ { n } ^ { * } \geq \theta \right\} \right| } { N _ { K } }\tag{35}
$$

The ROC is computed by varying θ, and the area under that curve is determined.

## A.7 Experiments on Additional Postprocessing Methods

To assess the generalizability of our findings, we conducted additional experiments on four widely used postprocessing methods that were not included in the main paper. Particularly, we include Energy [], ODIN [], ReAct+Energy [], and ASH-S [ε]. The AUOSCR heatmaps for combinations on CIFAR+N and ImageNet are shown in Figure 13 and Figure 11, respectively. For Energy and ODIN we observe nearly identical effects as previous postprocessors, while ReAct+Energy and ASH-S struggled with high-similarity setting $P _ { 3 }$ (shifting the color-map and slightly hiding the effect). On the smaller CIFAR+N dataset, we observe that ReAct+Energy and ASH-S can exhibit high variation in performance, dragging down the average performance significantly, while other poseprocessors are highly stable.

![](images/ba9cbc5e29406ad8da5c6d8c31da0d068195dff9075715aed2e09d8fa4cf2db1.jpg)  
Figure 10: CIFAR+N OSR PERFORMANCE. We show OSR performance for gg ° f f combinations across datasets in AUOSCR. The color of each heatmap is normalized independently and centered at the gMspf cE baseline, where blue shows an increase and red a decrease. Single-stage methods are surrounded by a black border. Results depict the average performance ± standard deviation over 5 trials.

![](images/94ed06100044dd2fba6ed26da5d58b9e4752c1d613652505c1f043eb9260fd86.jpg)  
Figure 11: IMAGENET OSR PERFORMANCE. We show OSR performance for gg ° f f combinations across datasets in AUOSCR. The color of each heatmap is normalized independently and centered at the gMspf cE baseline, where blue shows an increase and red a decrease. Single-stage methods are surrounded by a black border.

## A.8 Additional Figures and Tables

The main paper contained only a subset of evaluation metrics and visualizatrions.   
Here we provide remaining figures and tables containing the exact results.

$$
4 3 . 0 \pm 1 3 . 2
$$

$$
4 0 . 5 \pm 1 3 . 3
$$

$$
6 3 . 8 \pm 5 . 1
$$

$$
4 4 . 1 \pm 1 2 . 3
$$

$$
8 4 . 5 \pm 2 . 4
$$

$$
6 7 . 8 \pm 3 . 0
$$

$$
g \gamma . 0 \pm \ : 0 . 7
$$

$$
8 6 . 1 \pm 1 . 9
$$

$$
9 2 . 2 \pm 1 . 1
$$

$$
9 3 . 7 \pm 0 . 8
$$

$$
8 7 . 8 \pm 1 . 1
$$

$$
g \gamma . 0 \pm \ : 0 . 7
$$

$$
9 2 . 9 \pm 0 . 9
$$

$$
\it 4 6 . 5 \pm \ : 9 . 2
$$

$$
\uparrow 2 . 5 \pm 2 . 7
$$

$$
9 4 . 5 \pm 0 . 5
$$

$$
8 8 . 4 \pm 1 . 0
$$

$$
8 9 . 8 \pm \ : 1 . 7
$$

$$
g \gamma . 0 \pm \ : 0 . 7
$$

$$
3 3 . 6 \pm 1 5 . 3
$$

$$
3 3 . 6 \pm 1 5 . 3
$$

$$
g _ { \not 4 \cdot \not 4 } \pm \ : 0 . 9
$$

$$
6 4 . 6 \pm 4 . 3
$$

$$
g 6 . 6 \pm \ : 0 . 5
$$

$$
g 0 . 5 \pm \ : 1 . 0
$$

$$
8 5 . 3 \pm 1 . 6
$$

$$
g \gamma . 0 \pm \ : 0 . 7
$$

$$
9 2 . 8 \pm 0 . 7
$$

$$
9 4 . 7 \pm 0 . 3
$$

$$
8 8 . 3 \pm 1 . 0
$$

$$
4 1 . 0 \pm 1 1 . 8
$$

$$
4 2 . 8 \pm 9 . 6
$$

$$
6 5 . 3 \pm 1 . 7
$$

$$
8 4 . 6 \pm 2 . 1
$$

$$
g 7 . 1 \pm \ : 0 . 7
$$

$$
9 2 . 4 \pm 1 . 1
$$

$$
9 3 . 9 \pm 0 . 8
$$

$$
4 4 . 4 \pm 9 . 4
$$

$$
4 4 . 6 \pm 9 . 2
$$

$$
6 8 . 2 \pm 1 . 5
$$

$$
8 7 . 9 \pm 1 . 1
$$

$$
8 6 . 4 \pm 1 . 6
$$

$$
g 7 . 1 \pm \ : 0 . 7
$$

$$
9 3 . 0 \pm 0 . 9
$$

$$
9 4 . 7 \pm 0 . 5
$$

$$
8 8 . 6 \pm 0 . 9
$$

$$
\gamma _ { 3 . 2 } \pm \ : 2 . 9
$$

$$
g 0 . 2 \pm \ : 1 . 2
$$

$$
g 7 . 1 \pm \ : 0 . 7
$$

$$
g _ { 4 } . 5 \pm \ : 0 . 8
$$

$$
3 6 . 3 \pm 1 5 . 0
$$

$$
3 6 . 3 \pm 1 5 . 0
$$

$$
6 4 . 3 \pm 2 . 4
$$

$$
8 5 . 5 \pm 1 . 4
$$

$$
g 7 . 1 \pm \ : 0 . 7
$$

$$
9 3 . 0 \pm 0 . 8
$$

$$
3 . 6 \pm 4 . 3
$$

$$
9 4 . 9 \pm 0 . 4
$$

$$
3 . 6 \pm 4 . 3
$$

$$
8 8 . 7 \pm 0 . 7
$$

$$
8 8 . 1 \pm 1 . 4
$$

$$
9 5 . 2 \pm 0 . 4
$$

$$
9 0 . 2 \pm 1 . 0
$$

$$
8 0 . 7 \pm 7 . 6
$$

$$
9 2 . 6 \pm 2 . 5
$$

$$
9 2 . 2 \pm 2 . 5
$$

$$
6 1 . 8 \pm 1 2 . 9
$$

$$
8 0 . 8 \pm 4 . 0
$$

$$
9 1 . 9 \pm 1 . 2
$$

$$
6 3 . 1 \pm 2 2 . 8
$$

$$
8 4 . 2 \pm 4 . 4
$$

$$
8 1 . 3 \pm 7 . 8
$$

$$
9 3 . 5 \pm 1 . 9
$$

$$
g 3 . 7 \pm \ : 0 . 7
$$

$$
3 4 . 8 \pm 3 1 . 4
$$

$$
8 1 . 1 \pm 7 . 7
$$

$$
9 4 . 1 \pm 1 . 3
$$

$$
9 3 . 0 \pm 1 . 8
$$

$$
9 7 . 3 \pm 0 . 7
$$

$$
9 6 . 0 \pm 0 . 5
$$

$$
9 3 . 6 \pm 1 . 1
$$

$$
5 8 . 2 \pm 1 8 . 2
$$

$$
8 1 . 5 \pm 5 . 7
$$

$$
9 3 . 2 \pm 1 . 8
$$

$$
g \gamma . 2 \pm \ : 0 . 7
$$

$$
5 8 . 8 \pm \ : 1 9 . 5
$$

$$
g 5 . 7 \pm \ : 0 . 6
$$

$$
9 8 . 0 \pm 0 . 9
$$

$$
5 8 . 8 \pm \ : 1 9 . 5
$$

$$
9 2 . 7 \pm 1 . 5
$$

$$
8 2 . 4 \pm \ : 5 . 8
$$

$$
5 2 . 3 \pm 1 9 . 6
$$

$$
g _ { 3 . 3 } \pm . 1 . 5
$$

$$
5 2 . 3 \pm 1 9 . 6
$$

$$
9 5 . 6 \pm 0 . 6
$$

$$
g 8 . 2 \pm \ : 0 . 9
$$

$$
8 1 . 3 \pm 6 . 2
$$

$$
9 3 . 2 \pm 1 . 5
$$

$$
9 2 . 6 \pm 1 . 6
$$

$$
9 7 . 1 \pm 0 . 8
$$

$$
9 5 . 6 \pm 0 . 6
$$

$$
9 8 . 1 \pm 0 . 9
$$

$$
3 6 . 0 \pm 3 2 . 5
$$

$$
8 1 . 7 \pm 6 . 4
$$

$$
g 2 . 8 \pm \ : 1 . 5
$$

$$
9 5 . 6 \pm 0 . 6
$$

$$
9 8 . 1 \pm 1 . 0
$$

$$
9 2 . 7 \pm 1 . 6
$$

$$
6 5 . 5 \pm \ : 1 4 . 8 \ : 6 5 . 5 \pm \ : 1 4 . 8
$$

$$
8 2 . 1 \pm 6 . 5
$$

$$
9 8 . 0 \pm 1 . 1
$$

$$
g 2 . 7 \pm \ : 1 . 9
$$

$$
8 1 . 9 \pm 6 . 9
$$

$$
9 8 . 1 \pm 1 . 1
$$

$$
g 8 . 3 \pm \ : { 1 . 0 }
$$

$$
g 2 . 7 \pm \ : 1 . 8
$$

$$
4 4 . 0 \pm 1 8 . 2
$$

$$
4 4 . 0 \pm 1 8 . 2
$$

$$
7 2 . 1 \pm 1 0 . 2
$$

$$
4 2 . 0 \pm 3 3 . 9
$$

$$
9 8 . 1 \pm 1 . 2 
$$

$$
9 2 . 6 \pm 2 . 0
$$

$$
2 2 . 9 \pm 0 . 0
$$

$$
3 1 . 9 \pm 3 . 7
$$

$$
5 4 . 2 \pm 3 . 9
$$

$$
\it { 2 4 . 7 \pm { 7 . 2 } }
$$

$$
9 1 . 6 \pm 0 . 6
$$

$$
8 4 . 4 \pm 0 . 7
$$

$$
5 8 . 4 \pm 3 . 4
$$

$$
2 0 . 4 \pm 6 . 3
$$

$$
3 8 . 0 \pm \ : 6 . 3
$$

$$
g _ { 4 } . 5 \pm \ : 0 . 4
$$

$$
1 1 . 4 \pm 1 . 4
$$

$$
8 6 . 0 \pm \ : 0 . 7
$$

$$
7 9 . 5 \pm 1 . 2
$$

$$
5 1 . 8 \pm 5 . 1
$$

$$
0 . 3 \pm 0 . 2
$$

$$
8 0 . 8 \pm 0 . 7
$$

$$
9 1 . 8 \pm 0 . 4
$$

$$
8 5 . 5 \pm 0 . 6
$$

$$
{ 3 0 . 0 \pm \ : 0 . 0 }
$$

$$
3 8 . 3 \pm 3 . 3
$$

$$
5 5 . 3 \pm 5 . 3
$$

$$
8 0 . 3 \pm 1 . 6
$$

$$
9 1 . 8 \pm 0 . 6
$$

$$
3 6 . 1 \pm 8 . 2
$$

$$
8 5 . 0 \pm 0 . 9
$$

$$
5 8 . 5 \pm 5 . 0
$$

$$
8 2 . 1 \pm 1 . 3
$$

$$
2 3 . 5 \pm 1 2 . 8
$$

$$
3 6 . 6 \pm 6 . 6
$$

$$
6 2 . 1 \pm 4 . 7
$$

$$
8 5 . 0 \pm \ : 1 . 0
$$

$$
1 0 . 3 \pm 5 . 0
$$

$$
g _ { 4 } . 7 \pm \ : 0 . 4
$$

$$
2 3 . 1 \pm 7 . 2
$$

$$
5 1 . 8 \pm 4 . 0
$$

$$
8 0 . 1 \pm 1 . 3
$$

$$
8 6 . 4 \pm \ : 0 . 8
$$

$$
9 0 . 9 \pm 0 . 6
$$

$$
9 2 . 5 \pm 0 . 4
$$

$$
0 . 1 \pm 0 . 2
$$

$$
1 . 2 \pm 0 . 9
$$

$$
8 4 . 9 \pm 0 . 7
$$

$$
1 2 . 0 \pm 5 . 6
$$

$$
8 1 . 3 \pm 1 . 4
$$

$$
8 6 . 0 \pm 0 . 6
$$

$$
\mathit { 4 8 . 9 \pm 1 2 . 6 }
$$

$$
6 5 . 4 \pm 4 . 4
$$

$$
8 3 . 8 \pm \ : 2 . 9
$$

$$
9 8 . 4 \pm \ : 0 . 5
$$

$$
g _ { 0 } { } . 6 \pm \ : 1 . 0
$$

$$
6 0 . 8 \pm 6 . 4
$$

$$
7 9 . 3 \pm 2 . 9
$$

$$
9 1 . 8 \pm 1 . 0
$$

$$
6 0 . 2 \pm 6 . 9
$$

$$
7 9 . 8 \pm 2 . 3
$$

$$
9 2 . 3 \pm 0 . 6
$$

$$
4 1 . 9 \pm 1 7 . 6
$$

$$
6 0 . 5 \pm 4 . 2
$$

$$
8 1 . 8 \pm 3 . 0
$$

$$
9 6 . 2 \pm 0 . 4
$$

$$
9 8 . 1 \pm 0 . 5
$$

$$
9 3 . 1 \pm 0 . 8
$$

$$
3 6 . 1 \pm 1 4 . 8
$$

$$
8 0 . 8 \pm 2 . 7
$$

$$
9 6 . 0 \pm 0 . 5
$$

$$
5 0 . 1 \pm 1 4 . 4
$$

$$
6 6 . 7 \pm 7 . 6
$$

$$
8 3 . 5 \pm 3 . 4
$$

$$
9 8 . 2 \pm 0 . 6
$$

$$
9 3 . 4 \pm 1 . 0
$$

$$
4 9 . 6 \pm 1 3 . 1
$$

$$
6 7 . 8 \pm 6 . 7
$$

$$
8 3 . 5 \pm 3 . 5
$$

$$
9 4 . 1 \pm 1 . 0
$$

$$
g \gamma . 7 \pm \ : 0 . 5
$$

$$
g 6 . 4 \pm 0 . 3
$$

$$
9 8 . 2 \pm 0 . 5
$$

$$
g _ { \mathrm { * } } ^ { \mathrm { * } } \beta . 6 \pm \ : 0 . 9
$$

$$
6 7 . 1 \pm 7 . 7
$$

$$
8 3 . 7 \pm \ : 3 . 7
$$

$$
g _ { 4 } . 2 \pm \ : 1 . 0
$$

$$
9 7 . 6 \pm 0 . 4
$$

$$
9 6 . 2 \pm 0 . 3
$$

$$
4 6 . 8 \pm 1 7 . 4
$$

$$
g _ { 0 } { } . 6 \pm \ : 1 . 0
$$

$$
6 7 . 0 \pm 6 . 8
$$

$$
8 3 . 2 \pm 3 . 5
$$

$$
9 4 . 0 \pm 1 . 1
$$

$$
9 7 . 6 \pm 0 . 4
$$

$$
9 6 . 2 \pm 0 . 3
$$

$$
9 8 . 2 \pm 0 . 6
$$

$$
9 3 . 4 \pm 0 . 9
$$

$$
7 . 7 \pm 5 . 2
$$

$$
4 6 . 8 \pm 2 1 . 1
$$

$$
8 3 . 4 \pm 3 . 6
$$

$$
9 4 . 1 \pm 1 . 0
$$

$$
9 7 . 6 \pm 0 . 4
$$

$$
9 6 . 2 \pm 0 . 3
$$

$$
9 8 . 2 \pm 0 . 6
$$

$$
g _ { \mathrm { * } } 3 . 6 \pm \ : 0 . 9
$$

$$
\begin{array} { l } { 5 1 . 2 \pm 1 1 . 8 } \\ { 4 8 . 0 \pm 1 1 . 8 } \end{array}
$$

$$
7 0 . 2 \pm 8 . 2
$$

$$
8 5 . 1 \pm 2 . 5
$$

$$
9 4 . 0 \pm 1 . 1
$$

$$
g \gamma _ { \cdot } \phi \pm \ : \theta . 5
$$

$$
g 6 . 2 \pm \ : 0 . 4
$$

$$
8 5 . 1 \pm 2 . 6
$$

$$
9 4 . 0 \pm 1 . 1
$$

$$
g 8 . 3 \pm \ : 0 . 5
$$

$$
g \gamma _ { \cdot } \phi \pm \ : \theta . 5
$$

$$
g 6 . 2 \pm \ : 0 . 4
$$

$$
9 3 . 9 \pm 0 . 8
$$

$$
g 8 . 3 \pm \ : 0 . 5
$$

$$
9 3 . 8 \pm 0 . 8
$$

$$
{ \bf 5 } 1 . 6 \pm \ : { \bf 1 } 1 . 9
$$

$$
\ 7 0 . 4 \pm \ 8 . 4
$$

$$
3 5 . 9 \pm 1 9 . 7
$$

$$
5 3 . 5 \pm 8 . 1
$$

$$
7 8 . 2 \pm 4 . 2
$$

$$
9 2 . 7 \pm 1 . 4
$$

$$
g \gamma _ { \cdot } \phi \pm \ : \theta . 5
$$

$$
9 5 . 6 \pm 0 . 6
$$

$$
8 . 2 \pm 9 . 9
$$

$$
g _ { 4 } , 1 \pm \ : 1 . 1
$$

$$
8 5 . 5 \pm \ : 3 . 0
$$

$$
5 6 . 8 \pm 2 8 . 2
$$

$$
9 7 . 5 \pm 0 . 6
$$

$$
g \gamma _ { \cdot } \phi \pm \ : \theta . 5
$$

$$
g 6 . 2 \pm \ : 0 . 4
$$

$$
9 2 . 5 \pm 0 . 7
$$

$$
g 8 . 3 \pm \ : 0 . 6
$$

$$
g _ { 4 } . 0 \pm \ : 0 . 8
$$

Table 3: SMALL-SCALE EVALUATION. This table includes a performance overview of all representation learning and postprocessing methods on the small-scale protocols. Metrics include CCR@ζ for $\zeta \in \bar { 1 0 ^ { - 2 } } , 1 0 ^ { - 1 } , 1 \bar { 0 ^ { - 0 } } , 1 0 ^ { 1 } , 1 0 ^ { 2 }$ , AUOSCR, AUROC, and OOSA. All scores are reported in percent and CCR@100 is the closed-set accuracy. Performance metrics are reported as mean ± standard deviation over 5 randomized trials. The best performing combination for each protocol and metric w.r.t mean score is highlighted in bold. The best performing postprocessor for each representation learning method and metric w.r.t mean score is highlighted in italic.

<table><tr><td colspan="2">Dataset f CE</td><td colspan="8">CCR@0.01 CCR@0.1 CCR@1 CCR@10 CCR@100 AUOSCR AUROC OOSA</td></tr><tr><td>P1</td><td>g MSP OpenMax MaxLogits</td><td>7.8 13.2 6.1</td><td>22.7 34.4 31.3</td><td>53.4 65.0 64.1</td><td>71.8 76.6 76.6</td><td>77.4 77.0</td><td>75.6 76.5</td><td>95.3 98.7</td><td>83.7 86.2</td></tr><tr><td rowspan="8">ARPL</td><td rowspan="8">PostMax GHOST OpenMax</td><td></td><td>33.0 44.9</td><td>65.9 71.4</td><td>75.9 77.0</td><td>77.4 77.4 77.4</td><td>76.8 76.7 77.1</td><td>98.5 98.0 99.1</td><td>86.3 86.6 87.5</td></tr><tr><td>MSP</td><td>18.3 10.6 8.0</td><td>27.2 44.6</td><td>54.8 72.5 66.1</td><td>78.2 77.3 78.2</td><td>76.4 76.8</td><td></td><td>95.6 98.8 98.7</td></tr><tr><td>MaxLogits PostMax GHOST</td><td>27.8 20.6 21.1</td><td>35.1 41.5 48.9</td><td>77.0 66.3 77.4 65.6 76.8 72.3 77.9</td><td>78.2 78.2</td><td>77.7 77.5 77.9 77.5</td><td></td><td>98.1 99.2 97.9</td></tr><tr><td>MSP OpenMax</td><td>11.6 16.2 17.1</td><td>32.2 41.5 39.4</td><td>62.2 77.4 65.5 77.8 72.6 78.1</td><td>78.4 78.3 78.4 78.4</td><td>77.7 78.1 77.7</td><td>98.7 99.3 98.4 99.5</td><td></td></tr><tr><td>MaxLogits PostMax GHOST</td><td>14.5 19.7</td><td>35.9 51.3</td><td>67.3 77.0 74.8 78.1 62.7 77.6</td><td>78.4 78.1 78.0</td><td>78.2 77.4 77.3</td><td>98.2 98.2</td><td>86.9 89.0 84.3 83.7</td></tr><tr><td>MSP OpenMax MaxLogits PostMax</td><td>5.6 5.6 16.1 8.8</td><td>27.9 28.2 36.7 27.2</td><td>62.4 77.4 73.1 77.7 63.3 77.1</td><td>78.1 78.1 78.1</td><td>77.8 77.4 77.8</td><td>99.0 98.2 99.1</td><td>84.3 84.2 86.4</td></tr><tr><td>GHOST MSP OpenMax</td><td>12.1 18.0 22.5</td><td>48.0 34.2 33.4</td><td>77.8 77.8 77.6</td><td>78.3 78.0 78.3</td><td>77.7 77.4 78.0</td><td>98.6 98.6 99.3 99.1</td><td>83.4 84.3 86.4 87.3</td></tr><tr><td>MaxLogits PostMax GHOST MSP</td><td>16.4 21.4 19.5</td><td>46.0 41.4 54.5</td><td>78.0 78.0 78.0 50.5 51.5</td><td>78.3 78.3 74.0 73.7</td><td>77.9 78.0 64.9 66.3</td><td>99.4 80.1 83.4</td><td>83.9 76.4 77.1</td></tr><tr><td>ARPL</td><td>OpenMax MaxLogits PostMax GHOST MSP</td><td>2.5 7.6 6.0 8.1</td><td>3.6 8.4 7.4 10.3</td><td>24.5 25.0 58.9 30.5 59.9 30.3 61.3 24.4 49.9</td><td>74.0 74.0 74.0 75.4</td><td>68.5 68.8 69.1 66.1</td><td>87.8 88.2 88.9 80.7</td><td></td><td>78.8 79.4 80.0 76.3</td></tr><tr><td>K+1</td><td>OpenMax MaxLogits PostMax GHOST</td><td>2.6 3.1 5.7 4.5</td><td>4.1 3.7 13.6 13.3</td><td>29.5 26.9 34.5 34.5</td><td>52.2 73.8 58.5 59.8 75.4 60.1 75.4</td><td>75.4</td><td>83.6 87.5 88.3 88.9</td><td></td><td>77.2 78.8 80.2 80.5</td></tr><tr><td></td><td>MSP OpenMax MaxLogits PostMax</td><td>3.5 2.3 8.3</td><td>5.9 4.3 11.0 10.6</td><td>25.8 33.6 33.0 34.5 40.1</td><td>55.8 59.1 62.5 63.4 64.4</td><td>76.9 76.4 76.9 76.9 76.9</td><td></td><td>84.5 87.2 89.2 89.4 90.3</td><td>65.2 78.8 79.7 79.2 79.0</td></tr><tr><td>OE</td><td>GHOST MSP OpenMax MaxLogits PostMax</td><td>10.1 2.5</td><td>5.0 3.3 2.7</td><td>20.9 21.5 23.1 24.5</td><td>54.8 55.1 55.7 57.8</td><td>74.3 74.5 74.3 74.3</td><td></td><td>84.5 84.6 85.1 85.9</td><td>71.6 72.2 71.0 73.3</td></tr><tr><td>OS</td><td>GHOST MSP OpenMax</td><td>2.3 4.7 3.1</td><td>8.3 10.3 4.4 4.7</td><td>27.3 25.1 25.7</td><td>57.9 56.5 57.1 58.1</td><td>74.3 74.9 74.9 74.9</td><td></td><td>85.9 85.1 85.3</td><td>71.5 71.2 71.1</td></tr><tr><td rowspan="8">P3 CE</td><td>MaxLogits PostMax GHOST</td><td>4.2 6.3 5.8</td><td>6.5 9.3</td><td>26.1 27.9</td><td>58.6 58.5</td><td>74.9 74.9</td><td>85.9 85.8 86.0</td><td></td><td>71.2 73.9 70.2</td></tr><tr><td>MSP</td><td>7.5</td><td>10.0</td><td>32.3 21.4 20.9</td><td>64.9 64.6</td><td>85.7 85.3 85.7</td><td>78.1 77.9</td><td>86.4 86.4</td><td>78.2 78.0</td></tr><tr><td></td><td>OpenMax MaxLogits</td><td>3.2 1.9</td><td>14.6 17.7</td><td>62.6 66.3 65.9</td><td>85.7 85.7</td><td>77.9 78.9 78.8</td><td>87.7 88.7 88.6</td><td>78.1 79.1</td></tr><tr><td>ARPL</td><td>PostMax GHOST</td><td>2.3</td><td>20.1 21.5</td><td>64.5</td><td>86.5 86.3</td><td>78.4 78.3</td><td>86.0</td><td>78.7 78.0</td></tr><tr><td></td><td></td><td>6.8 6.8</td><td>21.7 15.8</td><td>64.4 62.0</td><td>86.5</td><td>78.4</td><td>86.0 87.6</td><td>78.0 77.9</td></tr><tr><td></td><td>OpenMax MaxLogits</td><td>0.5</td><td>2.5 17.4</td><td>65.4</td><td>86.5 86.5</td><td>79.3 79.2</td><td>88.6</td><td>79.0 78.6</td></tr><tr><td></td><td>PostMax</td><td>1.4</td><td>4.8 3.4</td><td>65.0</td><td>85.6</td><td>77.8</td><td>88.5</td><td>70.7</td></tr><tr><td>K+1</td><td>GHOST</td><td>2.3</td><td>3.7</td><td>65.6 65.6</td><td>85.6</td><td>78.5 78.7</td><td>86.5 87.0</td><td></td></tr><tr><td></td><td>MSP OpenMax</td><td></td><td></td><td>24.5 22.5</td><td>85.6</td><td></td><td>88.5</td><td></td></tr><tr><td></td><td>MaxLogits</td><td>1.0</td><td>3.2</td><td>19.1 23.5 69.1</td><td>66.0 85.6</td><td>79.7 79.6</td><td></td><td>89.5</td></tr><tr><td></td><td>PostMax</td><td>2.1</td><td>4.2</td><td>25.4</td><td>68.8 85.6</td><td></td><td></td><td>89.4</td></tr><tr><td></td><td>GHOST</td><td>0.5</td><td>4.7</td><td></td><td></td><td>84.6</td><td></td><td>87.5</td></tr><tr><td>OE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MSP</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.2 86.8</td></tr><tr><td></td><td>OpenMax</td><td></td><td>2.7</td><td>12.7</td><td>61.5</td><td>84.6</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>17.7</td><td>65.0</td><td>84.6</td><td>77.0</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GHOST</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.5</td></tr><tr><td>PostMax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MaxLogits</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>21.8</td><td>66.9</td><td></td><td>77.7</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>84.7</td><td>77.8</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>66.7</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>21.7</td><td></td><td>84.6</td><td>77.1</td><td></td></tr><tr><td></td><td></td><td></td><td></td></table>

Table 4: LARGE-SCALE EVALUATION. This table repeats the performance overview of Table 3 on large-scale protocols. — indicates that at no threshold $\theta _ { \zeta }$ for corresponding $F P R \zeta$ could be achieved.

![](images/aebcd8ea3d0bff6f5d282f985431ec833f60deb302db24baf68585fd4e9d6118.jpg)  
Figure 12: MAGNITUDES. Feature magnitude distributions of known, known unknown, and unknown classes on all protocols and for all representation learning methods, complete version of Figure 5. OE and OS experience feature magnitude collapse on $P _ { 3 }$ and EMNIST, pulling their feature magnitudes towards zero. For EMNIST, we show distributions for known classes 1 and 9 (black and grey) that are highly similar to known unknown classes, and other known classes (blue) separately.

<table><tr><td>Model</td><td>Class 1 min max</td><td>min</td><td>Class 9 max</td><td>Knowns Except 1 and 9 min max</td></tr><tr><td>CE</td><td>94.1 97.6</td><td>89.5</td><td>91.9</td><td>86.7 97.9</td></tr><tr><td>ARPL</td><td>86.7 96.9</td><td>89.5</td><td>92.2</td><td>83.8 98.0</td></tr><tr><td>K+1</td><td>67.0 94.8</td><td>92.4</td><td>95.6</td><td>95.2 99.5</td></tr><tr><td>OE</td><td>65.7 86.6</td><td>94.2</td><td>97.2</td><td>96.7 99.8</td></tr><tr><td>OS</td><td>58.1 88.7</td><td>93.6</td><td>95.3</td><td>97.7 99.6</td></tr></table>

Table 5: CLAss-W1SE CCR. EMNIST ranges of class-wise CCR are given at the operational threshold, $\mathrm { C C R } _ { c } ( \theta ^ { \star } )$ , for classes 1, 9, and rest (all other known classes) per representation learning method.

![](images/153931aad0ea1c8d5203847b4a89670f84b7d76d70cdf559acac08c687cf75e1.jpg)  
(d) Accuracy  
Figure 13: ABSOLUTE PERFORMANCE. These heatmaps show the absolute values for evaluations of (a) AUOSCR, (b) OOSA, (c) AUROC, and (d) Accuracy. Each heatmap is normalized independently and centered around CE+MSP, where blue shows an increase and red a decrease. Baseline representation learning and postprocessing methods are surrounded by a black border. Results for CIFAR+N are averaged over 5 trials.

![](images/d0cfc2fc5c7d9ec5841e29edbe6dfb48b051b8e96e3217b53c6d645b7b7b158f.jpg)

![](images/a0946d2be2331671b550648c7194c89b2240875aa884594203b7be4c9e221277.jpg)

![](images/14350f978f1b335ece60ccb990b3c83b47aa0734fa3f50d1a7bcd610fa90355c.jpg)

![](images/cb43095f71cdf4263fce7d86d9e57624d28764f9b98925bbb1a0fcaffc539ff4.jpg)

![](images/9a15a82e348065d7aba94127cecd49ae399f4cbc22d38e96e306b2fd5f761449.jpg)

![](images/a0debd86a978b6218c48c47523f87e9487be636296fcf1197e1551cc28d819e0.jpg)

![](images/9b117945918482f1a939e866488e212c206de0e15e1f138a5248200dd2fa0b15.jpg)

(a) AUOSCR  
![](images/853e79c0f9d2ea3c690914ff77623cd83d6ebfe6de881492bf7a5def684129f5.jpg)

![](images/1f9a3d55f70967833df238025152fec9ccce1df06dd7d47c1da6c872b9a6a827.jpg)  
(b) OOSA

![](images/332369a5242ffc3aa7968a47b5c65a80669750d84c5f608f6d906264bde0e8c0.jpg)

![](images/c41b252a41901ed77448f94041621541f20d45f345182410f729d56610e70b79.jpg)

![](images/b4c5a5c05cec8464f0d182579af2dbdf9874dfd6f8e615a6156e4ba46be40f8e.jpg)

![](images/d1ca320f1597a7f3a6e026f01943f2eecf360ef139eea7061e91fbb83e226cda.jpg)

![](images/3f163d6cfe166538488dc9c9a8a2916761c22d03cdad14093604fc44dd2079d0.jpg)  
(c) AUROC

![](images/6d89aa3eb921d01230f6abdfa22d288609d6abd4a61612139b30470b13638f8d.jpg)

![](images/dbb35a87182de937c2573c5a6c1a11961811c74383f7efa4f57e8bcf8c151388.jpg)

![](images/1bbb859a1cb75b01ed5f90101d7fffc6c465aa3623a6eb434a7e56922d8738b1.jpg)

![](images/840a3185dd5dc363459220b1900975ce844bc915f41c7e713c4d1e3c3e4f09cd.jpg)

![](images/169c5c3c666a118cb628be6c992e0a434a52129c781b4f33b0db1d5d6448efc0.jpg)  
(d) Accuracy

![](images/4c5fe46c96626bc76716d1ce0d953aee643e1ed28519e964a535e49d56c00d83.jpg)  
Figure 14: INTERACTION EFFECTS. This figure shows the interaction effects of representation learning and postprocessing components as correlation between representation learning performance contribution $\Delta _ { f } ( f \mid g )$ and postprocessor contribution $\Delta _ { g } ( g \mid f )$ in terms of (a) AUOSCR, (b) OOSA, (c) AUROC, and (d) Accuracy. Results for CIFAR+N are averaged over 5 trials.

![](images/1d3ff0e8808d69673f197eca88a0b8fd5c0b45c214b19cb3381dee6879df57f0.jpg)

(a) CIFAR+10  
![](images/51c445117539536684f7cb3c3ba5fc31d2483f4bdb61a61a630063fa9663927a.jpg)

(b) CIFAR+50  
![](images/14b5c20a46c8b46bff107deab02c36724dec7d29a3a4f8c06bc5fb1c67a7404c.jpg)

(c) $P _ { 1 }$  
![](images/688620886bf0dfaff0ab6b1df687ec44394dae46fcb6fa0f4709825e0936b1e9.jpg)

(d) $P _ { 2 }$  
![](images/2911c756c53cb25f2ab647dbb1ad4b06353ec45c2d0e26dd3b3ebf9690b6c0e3.jpg)  
(e) $P _ { 3 }$  
Figure 15: REGRESSION ANALYSIS. Linear regressions of class-wise CCR at the operational threshold, $\mathrm { C C R } _ { c } ( \theta ^ { * } )$ , against class-wise average feature magnitude for known classes. Regression is performed for each representation learning method independently and over all postprocessors. For $C I F A R { + } N$ results are reported for the first trial only.