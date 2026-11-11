# MEOW: Medical Open-World Interactive Segmentation

Philip Matthias Winter<sup>1\*</sup>, Maria Wimmer<sup>1\*</sup>, Bernhard Schafl¨ <sup>1</sup>, David Major<sup>1</sup>, Dimitrios Lenis<sup>1</sup>, Astrid Berg<sup>1</sup>, Theresa Neubauer<sup>1</sup>, Ariharasudhan Muthusami<sup>1</sup>, Johannes Novotny<sup>1</sup>, Georg Zotti<sup>1</sup>, Katja Buhler¨ <sup>1</sup>

Abstract—In open-world recognition, machine learning models not only make predictions on unknown structures (”generalized open-set”), but also continually learn and improve from those predictions over time. Interactive segmentation, despite its access to user feedback, has so far remained confined to the open-set setting. In this work we introduce MEOW, the first framework that turns non-interactive segmentation backbones into interactive open-world learners without updating learnable model parameters. MEOW combines parameter-free transductive reasoning with memory consolidation to accumulate knowledge and thereby improve future performance. Focusing on medical data, we show that MEOW consistently improves its own predictions on classes never seen during training, and transfers zero-shot to unfamiliar domains, including micro-CT data and other imaging modalities.

Index Terms—open-world learning, interactive segmentation, memory network, transduction, k-nearest neighbors, retrieval, fast learning, message passing

## I. INTRODUCTION

Recent advances in lifelong learning have fundamentally shifted how deep learning models are deployed in dynamic environments where data distributions and task definitions continuously evolve [1], [2]. When a static model trained on a closed-world distribution is deployed in the wild, it inevitably encounters out-of-distribution data. Processing these unseen distributions without adaptation degrades predictive performance [3]. In critical vision-tasks [4], e.g. medical image analysis [5] and autonomous driving [6], this may lead to fatal outcomes. Consequently, enabling models to continuously adapt during their operational lifetime is crucial for maintaining high reliability and performance.

However, model adaptation introduces various challenges concerning plasticity-stability trade-offs and catastrophic forgetting [7], as well as model complexity and prohibitive computational costs [8]. Another conceptual problem arises from the nature of the incoming data stream. Standard lifelong learning frameworks often assume a well-defined stream where novel tasks or classes are explicitly delineated and labeled by human experts. In practical settings, however, models must be able to handle completely novel, unknown classes and domains that emerge dynamically.

To succeed in such a dynamic environment, open-world (OW) learning has emerged as a highly relevant and pressing challenge. In OW-learning, a model must be able to operate in unknown domains, make predictions for unknown classes, and also actively learn and improve over time. This is usually achieved by leveraging knowledge directly from its own predictions on these unknowns.

![](images/7e6762a8c84caff614dc3a7241c29764d6fe418ef53f0403bf175766001bcc9d.jpg)  
Fig. 1. We propose MEOW, a framework for open-world interactive segmentation. a) Flexible properties for Interactive Segmentation. b) Segmenting an unknown structure and learning from own prediction. c) Improving segmentation performance by leveraging previously learned structures.

In clinical environments, closed-world systems with fixed label spaces fail due to rare pathologies and domain shifts. They often yield hazardous, overconfident predictions when unknown structures are forced into predefined categories [9]. OW-learning addresses this by incorporating these unknowns into a model’s taxonomy. Moreover, it mitigates severe data scarcity caused by costly expert annotations, licensing constraints, and strict privacy regulations, as exhaustive labeling of every possible class is no longer required. By dynamically adapting to anatomical variations and unseen distributions, OW frameworks help bridge the gap toward safe, robust clinical deployment.

OW-learning combines multiple learning paradigms [10], including incremental, continual, transfer, out-of-distribution, few-shot, and open-set (OS) learning. Furthermore, desired properties encompass mechanisms for fast learning and unlearning at scale. Beyond these, OW systems must unify instance classification for existing classes with novelty detection and label management for new classes. While OS systems are limited to identifying or rejecting unknown classes, OW systems incrementally learn from their own predictions during deployment. Previously unknown classes become known once processed, thereby enhancing future performance. Beyond traditional tasks like classification or regression, OW recognition also extends to generative modeling, where models must adapt to generate samples from previously unseen distributions.

We focus on the 3D vision-task of open-world interactive segmentation (OW-IS), which has not been studied yet. In interactive segmentation (IS) [11], [12], a machine learning (ML) model takes user inputs like clicks or brushes to segment parts of an image. Self-training and pseudo-labeling schemes for OW systems often rely on gradient-based parameter updates, which are prone to confirming a model’s systematic biases, degrade representation stability, and require significant computational overhead. This motivates the need for an adaptation mechanism that can exploit a model’s own predictions to learn new classes and domains reliably, rapidly, and safely.

In this work, we present the first system capable of OW-IS, thereby addressing the core challenges encountered by an OW-learning system (see Fig. 1). On top of the desired OW-properties mentioned above, our approach offers various favorable properties in terms of IS- and OW aspects, including flexible user interactions and a modular architecture. Most importantly, our method requires no gradient-based learning [13] to transform non-interactive, frozen-parameter models into flexible IS models while simultaneously enabling efficient OW-learning capabilities. This is achieved by building upon our previously proposed method PARMESAN [14], which is a memory-based approach for solving dense prediction tasks such as semantic segmentation and depth estimation. It was specifically designed as a continual learning method, exhibiting stable knowledge retention without catastrophic forgetting.

## Our main contributions are as follows:

Interactive Segmentation: We introduce MEOW, a modular 3D IS system that is capable of handling sparse annotations (e.g., from clicks or brushes) in a flexible manner. In contrast to methods that are trained end-to-end using a fixed interaction strategy during training, our approach takes non-interactive off-the-shelf segmentation models and integrates them into our IS framework. Our system is invariant to diverse annotation styles, ensuring robust performance across different users. Apart from clicking and brushing interactions, our method further supports interactive hyperparameter tuning via sliders. Open-World Learning: We propose a solution to OW-IS, i.e., an IS system learning from its own predictions of novel structures. Notably, this also includes the selective removal of previously learned samples and domains. To the best of our knowledge, this is the first operational OW-IS framework.

General Improvements: To enable efficient OW-IS for 3D data, we introduce modifications to PARMESAN, including improved feature compression, better memory search kernels, and more efficient test-time augmentation. We also improve its message passing approach for post-processing of raw predictions by implementing hierarchical message passing, thereby enhancing overall segmentation quality and robustness. Finally, we propose novel evaluation metrics that are useful for IS.

Our approach uses publicly available, pre-trained models without requiring any additional training of learnable parameters, making it flexible, modular, updatable, and fast-learning by design. Our work can be fully reproduced with consumer-grade hardware, thereby underlining our energy-efficient use of computational resources.

## II. RELATED WORK

## A. Interactive Segmentation

Semantic segmentation has recently been transformed by foundation models like U-Net SuPreM [15], TotalSegmentator [16], and MemSAM [17]. Being trained on large datasets containing highly diverse classes, these models excel at segmenting basic structures like organs and bones. Interactive segmentation (IS) [18] is typically addressed via end-to-end training in a simulated environment with a virtual user [19], [20]. These virtual users mimic actions of human users in an automated manner, which is sufficient for model training and evaluation. Some IS frameworks incorporate multimodal inputs, including text prompts [21] or diverse prompt mixtures [22]. Click- and scribble annotations serve as weak labels for guiding IS model predictions [23], [24]. Click-based annotations have also been used for region of interest selection in transfer function design [25]. Annotation platforms [26] can be leveraged to annotate large-scale datasets for IS [27], in turn leading to coherent and high-quality ML models. Powerful IS systems like SAM-Med2D [28], MedSAM [29], nnInteractive [30], VISTA3D [31], and others [32]–[36] have been developed for the medical domain. Recently, memory-attention modules [37] have been introduced to improve IS performance by storing past user inputs and segmentation states to enable temporal context integration.

Comparison to MEOW: In contrast to monolithic IS models that are trained end-to-end using a fixed interaction strategy, MEOW is modular and thus flexible. Processing is handled by combining a frozen foundation model [38] as feature extractor with a transduction module, supporting diverse user interactions and different annotation styles. MEOW makes predictions by combining nearest neighbor retrieval [39]–[41] and correspondence matching [42]–[45] with transductive reasoning [46], [47]. Hierarchical message passing [48]–[51] is applied as a post-processing step to ensure spatial consistency. We provide qualitative comparisons of our framework w.r.t. different baselines and semantic segmentation methods in Table I.

## B. Open-Set and Open-World Recognition

Closed-set recognition involves processing known structures that a model has been trained on. Open-set (OS) recognition involves identifying or rejecting unknown structures [52], [53]. Generalized OS recognition involves not only identifying or rejecting unknowns, but also being able to make predictions for them [54], [55]. Open-world (OW) recognition [10], [56], [57] goes even further by incorporating unknown structures into the knowledge base during deployment. This requires robust novelty detection [58] and is closely related to continual learning [2], [59], where models accumulate knowledge over time. Although test-time adaptation [60] also aims at improving performance on unknown structures during deployment [61], it does not integrate the continual learning aspect of OW recognition. This paradigm has gained traction in classification [62], [63], object detection [64], [65], semantic segmentation [66], [67], instance segmentation [68], and panoptic segmentation [69]. However, interactive segmentation has not been studied in this context. Medical IS foundation models [31], [70] address OS segmentation of novel classes via interactive branches and learned context- and prompt encoders. While some methods improve models on known classes [71] or specific queries during deployment [72], [73], they often fail under data distribution shifts [74]–[76]. Moreover, they often suffer from data-hungry and inefficient test-time fine-tuning. This creates high computational overhead that is prohibitive for clinical applications, where data-scarce environments demand low-latency, deterministic responses [77].

TABLE I  
QUALITATIVE COMPARISON OF MEOW TO BASELINES AND SEMANTIC SEGMENTATION METHODS.
<table><tr><td>Method</td><td>modular / flexible</td><td>interaction- OS agnostic</td><td></td><td>OW</td><td>real-time learning</td></tr><tr><td>U-Net SuPreM [15]</td><td>X</td><td></td><td>X</td><td>X</td><td>X</td></tr><tr><td>TotalSegmentator [16]</td><td>x</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>PARMESAN [14]</td><td>√</td><td></td><td>x</td><td>x</td><td>√</td></tr><tr><td>SAM-Med3D [35]</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>VISTA3D [31]</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>nnInteractive [30]</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>MEOW</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Comparison to MEOW: As a memory-based, transductive reasoning framework, our approach is able to perform OW-IS via decoupling processing from knowledge [78], [79]. The latter is offloaded to a memory module containing labeled data, effectively making it a memory-augmented architecture [80]–[82]. In contrast to monolithic models that rely on learned decoders [83], this architectural choice aligns with recent findings that pre-training on procedural data [84] is effective for decoupling perception from task-specific knowledge. The overall approach results in a flexible, composable, and scalable design which is suited for OW-IS. Learning capabilities include fast learning, transfer learning, few-shot learning, continual learning, and unlearning.

## III. METHOD

We begin with a problem formulation for open-world interactive segmentation (OW-IS) in III-A, followed by a brief overview of our solution from a user’s perspective in III-B. In III-C and III-D, we provide details about our approach to IS. In III-E, we describe open-set and open-world capabilities of our approach. Lastly, we propose some general improvements to the original PARMESAN method in III-F. A list of symbols and abbreviations is given in Appendix V.

## A. Problem Formulation

Interactive Segmentation (IS): In 3D interactive segmentation, a user U actively guides an interactive segmentation model S to segment a structure of interest (SoI). The overall objective of IS is maximizing segmentation quality w.r.t. an SoI while at the same time minimizing the required user guidance (e.g., number and complexity of interactions). Moreover, an IS system must operate at low latency to enable meaningful interactions for U.

Open-World Interactive Segmentation (OW-IS): In OW-IS, S is capable of segmenting an SoI while also learning from own predictions. Segmentation not only includes known classes (closed-set IS), but extends to unknown classes and domains (open-set IS). The overall objective aligns with the objective for IS, but also requires learning from classes and domains that are unknown to S. In addition to low-latency interactions, an OW-IS system should preferably learn (and unlearn) at low latency in order to adapt fast.

Grounded in mixed-initiative principles [85], effective collaboration between U and S requires leveraging automated predictions when confidence is high while relying on user guidance when confidence is low. Designing a system to balance both aspects is critical for enabling efficient interaction.

We consider structures of interest (SoIs) to be flexible target regions that do not need to correspond to predefined semantic classes. They may contain holes and can consist of multiple disconnected or nested parts. The goal of U is to segment these SoIs by interacting with S. Considering an SoI with its ground truth binary segmentation mask (i.e., ground truth label), any additional information such as its class, semantic description, input domain, etc. are considered metadata. Depending on S’s knowledge about an SoI and its metadata, four different class categories can be defined in the OW setting [52]:

• KKCs (”known-known”): Classes with unambiguous labels and semantic / attribute metadata.

• KUCs (”known-unknown”): Classes explicitly labeled as unknown (e.g., background or any unknown structure).

• UKCs (”unknown-known”): Classes with no samples at training, but available semantic / attribute metadata.

• UUCs (”unknown-unknown”): Classes with neither data at training, nor metadata available.

## B. Overview of MEOW for OW-IS

When presented with a volumetric image (e.g., a CT or MRI scan), a user U first selects an SoI by clicking on it. A local region of fixed resolution (e.g., $1 2 8 \times 1 2 8 \times 1 2 8$ voxels) is subsequently cropped around the selection point, yielding a query image $x _ { q }$ that is centered on a portion of the SoI.

The query is then processed to make a first prediction using the initial selection point as a weak positive label (see Fig. 3). After this initialization phase, the alternating interaction phase starts, where U guides the model to modify and improve its prediction. At every interaction step, U provides guidance and S makes a prediction, which is then visually displayed to U for further refinement. Guidance is provided by placing additional (positive or negative) clicks and brushes, or by modifying interactive hyperparameters via sliders. U may end interaction when a certain segmentation quality is reached.

The final result may then be learned depending on the use case. If another, similar query is encountered in the future, previously learned samples can be leveraged to improve OW-IS objectives. Learning with MEOW is fully reversible and previously learned samples can easily be unlearned if they are no longer required.

## C. Elements of MEOW

We define an IS system as the combination of an interactive segmentation model $s$ and a user $\mathcal { U }$ (see Fig. 2). Given a query image $x _ { q }$ along with some user guidance, $s$ makes segmentation predictions for an SoI within $x _ { q }$ (see Fig. 6).

For data, we assume regular Cartesian voxel grids (i.e., uniform structured rectilinear grids) on uniformly discretized Euclidean domains. This naturally accommodates diverse spatiotemporal data modalities, including volumetric medical images (CT, MRI), videos, and Eulerian simulation grids in numerical modeling. Let $\Omega ^ { l } \ = \ ( N _ { 1 } ^ { l } \times N _ { 2 } ^ { l } . . . N _ { d } ^ { l } )$ denote the d-dimensional spatial domain of a representation at resolution-level $l = 1 , \ldots , L .$ , with $\Omega ^ { 1 }$ having the highest resolution and $\Omega ^ { L }$ having the lowest resolution. Let $x \in$ $X \subseteq \mathbb { R } ^ { \Omega ^ { 1 } }$ denote a model input. For example, 3D CT data has $\Omega ^ { 1 } = ( H ^ { 1 } \times W ^ { 1 } \times D ^ { 1 } )$ , where H, W, and $D$ indicate grid-resolutions in height, width, and depth.

User U: For guiding predictions, U relies on a cognitive mental model of an SoI’s outline, which we formalize as the ground truth binary segmentation mask $z _ { g t } ~ \in ~ \{ 0 , 1 \} ^ { \Omega ^ { 1 } }$ . U might also have access to an SoI’s metadata like semanticand attribute information, which may be used by S (details in Section III-E). In the context of this work, U is typically a medical expert with domain expertise (e.g., a radiologist). At inference, $\mathcal { U }$ and S interact in an alternating manner to segment an SoI. Model predictions $y \in [ 0 , 1 ] ^ { \breve { \Omega } ^ { 1 } }$ can be thresholded into a pseudo-label binary mask $z _ { p } = \mathbb { I } ( y \geq \tau )$ with $z _ { p } ~ \in ~ \{ 0 , 1 \} ^ { \Omega ^ { 1 } }$ and $\tau \in \mathbb { R }$ being the classification threshold. These masks can then be learned via memory consolidation to improve future predictions. While $\chi _ { \mathrm { s } }$ role is providing weak labels and hyperparameter changes for guidance (see Fig. 3), S provides corresponding segmentation predictions y.

For adapting to specific use cases and diverse annotation preferences of different users, we introduce several components to increase overall flexibility of our method. These components can be controlled independently and dynamically (e.g., via sliders) as interactive hyperparameters $\iota ,$ allowing U fine-grained control over segmentation behavior. Depending on the technical expertise of U, they may prefer to bypass fine-grained control via $\mathbf { \delta } _ { \mathbf { \delta } } ^ { \mathbf { \delta } _ { t , \mathrm { ~ \tiny ~ } } }$ falling back to default values typically obtained through automated hyperparameter optimization (see Appendix IV for details).

Segmentation Model $s { : }$ In our framework, the interactive segmentation model $s$ consists of three modules: a frozen feature extractor ${ \mathcal { F } } ,$ a memory module M for learning, and a parameter-free transduction module $\mathcal { G }$ for making predictions (see Fig. 1 and Fig. 2). We attach $\mathcal { G } _ { \mathrm { ~ \scriptsize ~ t o ~ } } \mathcal { F }$ to form a functional IS system. The system is further enhanced with a memory module $\mathcal { M }$ to enable OW-learning capabilities. We now introduce these modules in more detail and describe how they operate.

Let $z ^ { \pm } \in \{ + 1 , - 1 , \emptyset \} ^ { \Omega ^ { 1 } }$ denote a weak user label, representing the combination of positive foreground (+1), negative background $( - 1 )$ , or no label (∅) w.r.t. an SoI at image resolution $\Omega ^ { 1 }$ (i.e., a trinary mask). We view $z ^ { \pm }$ as a special type of sparse labels where some voxels are left unlabeled. Weak labels may be composed of any number of clicks and brushes in any spatial direction, including off-axis directions. The combination of all clicks and brushes is $z ^ { \pm }$ Brushing can be done in both 2D (via circles, lassos, etc.) or 3D (via spheres, cubes, etc.). Unlabeled voxels are being ignored for processing. Weak labels are combined with a set of interactive hyperparameters ι, which control different aspects of how predictions are calculated by $\mathcal { G }$ (details in Appendix IV). Together, they form a unified user guidance $u = \{ z ^ { \pm } , \iota \}$ During the alternating interaction phase, u is partially or fully modified by U in-between calls to ${ \mathcal { G } } .$

Feature Extractor $\mathcal { F } \colon \mathrm { ~ A ~ }$ pre-trained feature extractor $\mathcal { F }$ with frozen weights θ extracts a multi-scale feature pyramid $( h ) ^ { l } = { \mathcal { F } } ( x )$ , yielding hidden representations $h ^ { l } \in \overline { { \mathbb { R } } } ^ { C _ { l } \times \Omega ^ { l } }$ at each resolution-level $l = 1 , \ldots , L$ , where $C _ { l }$ denotes the number of feature channels. The number of levels L depends on $\Omega ^ { 1 }$ , scaling down by a factor of two at each level until reaching the lowest resolution $\Omega ^ { L } \leq ( 2 \times 2 \times 2 )$ . In case of incomplete feature pyramids, missing levels are interpolated or extrapolated from existing representations. Because $\mathcal { F } ( x )$ is independent of user guidance $u ,$ feature extraction is computed only once during the initialization phase (details in Section III-D).

Our requirement for $\mathcal { F }$ is that it is capable of providing feature pyramids $( h ) ^ { l }$ which is generally true for hourglass-shaped neural network architectures like the U-Net family [86]–[88]. However, we do not assume that $\mathcal { F }$ is pre-trained specifically for the task of IS. Instead, we can utilize models trained for other tasks such as semantic segmentation, instance segmentation, panoptic segmentation, or regression. For some use cases, $\mathcal { F }$ may even be completely untrained.

Memory Module M: A memory module $\mathcal { M }$ stores $j = 1 , \ldots , m$ labeled samples $\begin{array} { c c l } { { \mathcal { M } } } & { { = } } & { { \{ \vphantom { \sum _ { i } } ( x , z , c , \frac { } { } \in h ) ^ { l } \} _ { i } \} } } \end{array}$ where $c _ { j } \in \{ 0 , \ldots , K \}$ denotes the semantic class of the corresponding SoI. We place no restrictions on the class definition, allowing c to group any anatomical, pathological, operational, or custom semantic concept. The process of preparing and writing samples to $\mathcal { M }$ is referred to as learning through memory consolidation. Consolidation strategies may involve quantization and compression in latent- and spatial dimensions for efficient storage. Our strategy involves local novelty search to find and discard voxels in $( h ) ^ { l }$ that are redundant w.r.t. other voxels [14]. Alternatively, one might discard voxels simply by random sampling. Making $0 < L _ { s p } \leq L$ levels sparse reduces the total number of voxels $p = \sqrt { \mathrm { I } _ { d } } \Omega ^ { l }$ of level l by a factor of $2 ^ { d \cdot \operatorname* { m a x } ( L _ { s p } - l + 1 , 0 ) }$ . This results in resolutions $\dot { \Omega } ^ { l }$ being identical across all sparse levels. We further apply compression of $( h ) ^ { l }$ in the channel dimension (more details in Section III-F).

We distinguish between a long-term memory module $\mathcal { M } ,$ which accumulates and stores episodic memories, and a working memory $M ,$ which is used more flexibly during a forward pass for making predictions. Depending on the processing step of the inference pipeline, M may contain the full M, a subset of M, or the current query itself (details in Section III-D and Section III-E). Fast learning and unlearning of individual samples is achieved simply through write and delete operations on M.

Transduction Module G: Given a query feature pyramid $( h ) _ { q } ^ { l }$ along with user guidance u and some working memory $M .$ , a transduction module G makes segmentation predictions $y = \mathcal { G } ( \mathcal { F } ( x )$ , u; M) (see Fig. 2 and Fig. 4). $\mathcal { G }$ essentially replaces learned decoders in a parameter-free manner. The process of leveraging stored samples for making predictions is referred to as transductive reasoning. To obtain a prediction $y ,$ transductive reasoning is applied w.r.t. user guidance u (for interpreting weak labels $z ^ { \pm } )$ , M (for making memory a prediction $y _ { \mathcal { M } } )$ , as well as $x _ { q }$ itself (for post-processing raw predictions $y _ { r } )$ .

G performs approximate nearest neighbor (NN) search [89] for non-parametric context-retrieval and reasoning (also see Appendix I). Operating over multi-scale feature pyramids, this mechanism enables highly scalable and parallelized correspondence matching suited for various vision tasks [14]. For every query voxel, NN search is performed against two kinds of memory contents: unpaired contents from stored samples and paired contents from the query itself. This yields semantic affinities that are used in subsequent reasoning steps. Comparable to retrieval-augmented generation in language models [82], our approach first retrieves relevant information via similarity-based NN search, followed by reasoning over them to obtain predictions. Post-processing via hierarchical message passing (HMP) yields smooth and self-consistent segmentations (details in Section III-D).

While G is primarily responsible for segmentation predictions from both user guidance and memory, it also comprises functions relevant for learning, novelty detection, and instance classification (details in Section III-E).

## D. Interactive Segmentation Pipeline

For IS using MEOW, inference is divided into an initialization phase and an alternating interaction phase. An additional learning step is present in an open-world scenario (see Section III-E). The initialization phase is executed once before alternating interactions start. This phase covers all computations that are independent from user guidance, including:

1) Embedding: A query $x _ { q }$ is embedded into $( h ) _ { q } ^ { l } = \mathcal { F } ( x _ { q } )$ with respective resolutions Ω<sup>l</sup>.

2) Feature Compression Weights: Linear projection weights for compressing $( h ) ^ { l }$ in the channel dimension are calculated via linear autoencoders (details in Section III-F).

3) Initialize Hierarchical Message Passing (HMP): A query-specific HMP graph is calculated by finding NNs between voxels of $( h ) _ { q } ^ { l }$ for all $l ~ = ~ 1 , \ldots , L _ { m p } ~ \leq ~ L$ levels, utilizing MEOW’s NN search capabilities (see Appendix I, Algorithm 1).

4) Memory- and Initial Prediction: A memory prediction $y _ { \mathcal { M } }$ may be made by utilizing stored memory samples (details in Section III-E). An automated weak positive label $z ^ { + }$ is placed at the center of $x _ { q }$ (i.e., where U first

![](images/315f5c85db4e7d34b9cd0dadcb5c92b12eef5516ca661d6109b2bd9483951aeb.jpg)

Fig. 2. Data flow diagram of our OW-IS approach MEOW: A memory module M accumulates knowledge over time and is used in combination with a frozen feature extractor $\bar { \mathcal { F } }$ and a parameter-free transduction module $\mathcal { G }$ for making predictions. A user U interacts with the system by providing guidance to improve results. Final results may be learned via memory consolidation. Learned samples can instantly be leveraged to improve performance for future predictions.  
![](images/2241e685ae5f1188025507be01a3a3b1232ef603fa679e58e0880f8612ae1c6b.jpg)  
Fig. 3. After the initialization phase, the alternating interaction phase starts by presenting several visual components to U: the query image $x _ { q }$ with an overlay of the initial prediction y (in blue), as well as the weak label $z ^ { + }$ at the center (in green). A crosshair indicates the current cursor position on the canvas. U can navigate through the volume via the mouse wheel or via keyboard buttons. On a control panel, interactive hyperparameters ι can be adjusted via sliders. A memory module suited for the task at hand can be selected. After every user action (i.e., clicks, brushes, or hyperparameter changes), G is called to update the prediction.

clicked into the SoI), resulting in a base prediction y<sub>b</sub>.   
It is merged with $y _ { \mathcal { M } }$ to make an initial prediction y.

5) Setup Interaction: The alternating interaction phase with U begins through a graphical interface, where a query image $x _ { q } .$ , weak labels $z ^ { \pm }$ , the initial prediction y, and interactive hyperparameters are visualized (see Fig. 3).

The alternating interaction phase covers all computations that depend on user guidance, i.e., correcting and refining model predictions. The dynamics of this phase are driven by alternating between user actions and calls to ${ \mathcal { G } } .$ . Based on visual overlays of model predictions $y ,$ U interacts with the system by placing weak labels $z ^ { \pm }$ (i.e., clicking and brushing)

![](images/e09df62c3e18273104f43f4e8f7fa5712e87cad5bc4ced752ea271ae2c67616e.jpg)

and adjusting interactive hyperparameters ι. Weak labels $z ^ { \pm }$ denote the cumulative aggregation of all positive and negative clicks and brushes across all interaction steps. Clicking and brushing tools can also be used to erase previously drawn labels.

We now describe how $\mathcal { G }$ handles weak user labels $z ^ { \pm }$ , i.e., labels without complete, voxel-wise annotations. Given the feature pyramid $( h ) _ { q } ^ { l }$ and user guidance u, predictions y are made in three processing steps (also see Fig. 4):

1) interactive base prediction $y _ { b } .$

2) optional merging of memory prediction $y \mathcal { M }$

3) and hierarchical message passing.

Predictions y are calculated from scratch with every call of ${ \mathcal { G } } .$ In processing step #1 (Appendix I, Algorithm 1), an interactive base prediction $y _ { b }$ is made by $\mathcal { G }$ to incorporate u (Fig. 4 b). We begin by filling a working memory M with the query itself, along with weak labels $z ^ { \pm }$ . To indicate labeled regions within $( h ) _ { q } ^ { l } .$ , a pyramid binary mask $( \mu ) _ { q } ^ { l }$ is constructed by resizing $| z ^ { \dot { \pm } } |$ via linear interpolation to resolutions $\Omega ^ { l }$ , followed by non-zero thresholding. Fast approximate, hierarchical k-NN search is then performed for every voxel in $x _ { q } ,$ restricted to the sparsely labeled regions $( \mu ) _ { q } ^ { l }$ . Starting from the global representation $h ^ { L }$ , levels are traversed downward through the feature pyramids, thereby retrieving both globally and locally similar stored voxels in M (Fig. 4 c). Since resolutions $\Omega ^ { l }$ increase $2 ^ { d } .$ -fold from level l to l − 1, k is reduced accordingly such that computational complexity remains quasi-constant throughout the hierarchical search process [14].

For any level l, voxel-wise cosine similarities between query features and memory features are

$$
s _ { c o s } ^ { l } ( h _ { q } ^ { l } , h _ { j } ^ { l } ) = \frac { h _ { q } ^ { l } \cdot h _ { j } ^ { l } } { \vert \vert h _ { q } ^ { l } \vert \vert \vert h _ { j } ^ { l } \vert \vert } \in [ - 1 , 1 ] ^ { \Omega ^ { l } } ,\tag{1}
$$

Fig. 4. Interactive segmentation with MEOW. The three processing steps of a call to G include an interactive base prediction (red), optional merging of a memory prediction (blue), and hierarchical message passing (green). a) A feature pyramid $( h ) _ { q } ^ { l }$ is calculated once by applying a feature extractor $\mathcal { F }$ to a query input $x _ { q } .$ b) Weak user labels $z ^ { \pm }$ for model guidance (positive: blue, negative: red) determine labeled parts $( \mu ) _ { q } ^ { l }$ of $( h ) _ { q } ^ { l } .$ . c) Approximate, hierarchical k-NN w.r.t. $( \mu ) _ { q } ^ { l }$ illustrated for 4 exemplary query voxels (in different colors), followed by label retrieval and reduction to obtain a base prediction $y _ { b } .$ d) Combining $_ { y _ { b } }$ with an optional memory prediction $_ { y , M }$ yields a raw prediction $y _ { r } .$ e) Hierarchical message passing for smooth and consistent segmentations.

with feature-similarities being

$$
s _ { f } ^ { l } ( h _ { q } ^ { l } , h _ { j } ^ { l } ) = \frac { 1 } { 2 } \left( 1 + s _ { c o s } ^ { l } ( h _ { q } ^ { l } , h _ { j } ^ { l } ) \right) \in [ 0 , 1 ] ^ { \Omega ^ { l } } .\tag{2}
$$

To avoid ambiguous matches originating from symmetries and repeated patterns [90], we employ a combination of both feature-based and proximity-based scoring for memory search as a geometric prior. For any NN of a query voxel in level l, accumulating similarities over levels yields

$$
s ^ { l } = \prod _ { i = l } ^ { L } ( \omega s _ { f } ^ { i } + ( 1 - \omega ) s _ { o } ^ { i } ) \in [ 0 , 1 ] ^ { \Omega ^ { l } } ,\tag{3}
$$

where $s _ { o }$ are proximity-similarities calculated using Euclidean distances between query positions and memory positions on the voxel grid. The product over levels acts as a momentum component that integrates the search trajectory into $s ^ { l } .$ The hyperparameter $\omega \in [ 0 , 1 ]$ controls the contribution of $s _ { f }$ and $s _ { o }$ to the final similarity score $s ^ { l }$ of level l.

NN search progresses from global, abstract representations to local, primitive representations to find semantically meaningful matches. Traversing through levels increases spatial resolutions, necessitating a reciprocal reduction in k (controlled by $k _ { s t a r t }$ and reduction factor ϕ) to maintain near-constant hardware overhead [14].

After finding k NNs for every query voxel, we retrieve respective final similarity scores $s _ { j } ^ { 1 }$ with $j = 1 , \dotsc , k$ along with the corresponding label values of $z ^ { \pm }$ . A base prediction $y _ { b }$ is computed via weighted reduction over the top-k NNs. When $( \mu ) _ { q } ^ { 1 }$ are extremely sparse (e.g., by clicking only a handful of voxels), distant NNs that are dissimilar w.r.t. other NNs can introduce noise for naive weighted averaging. To mitigate this, we adopt a continuous Hopfield layer [81] without learnable parameters, where a hyperparameter $\beta _ { n n }$ influences how strongly reduction weights $w _ { j }$ are committed towards higher similarity scores.

The processing step #2 is optional depending on the availability of a memory prediction $y _ { \mathcal { M } }$ . Given a working memory M containing labeled samples (see eq. (6)), a dense prediction $y _ { \mathcal { M } }$ is made by finding and retrieving top-k NNs from M for every query voxel. Retrieved labels are reduced via $y _ { \mathcal { M } } = \sum _ { k }$ s $\mathrm { \ r f } \mathrm { m a x } _ { k } ( s _ { j } ) { \cdot } z _ { j }$ . Here, $s _ { j }$ denotes the similarity score for the j-th retrieved NN voxel. We integrate $y _ { \mathcal { M } }$ into the main forward pass by fusing it with the base prediction $y _ { b }$ (Fig. 4 d, also see Appendix I, Algorithm 2). The same fusion approach applies to other scenarios, e.g., when continuing a previous IS session or when fusing predictions from multiple different methods. In our framework, the calculation of $y _ { \mathcal { M } }$ is part of the initialization phase preceding user interaction. A memory prediction is therefore only computed once and remains fixed during user interaction.

In processing step #3, raw predictions $y _ { r }$ are refined via hierarchical message passing (HMP, also see Appendix I, Algorithm 3). Recall that $y _ { r }$ is composed of independent predictions for every query voxel, thereby neglecting spatial priors like local smoothness and boundary awareness. By capturing intra-query correlations only indirectly through feature embeddings, $y _ { r }$ is prone to metrication artifacts [91] (e.g., blockiness) and imprecise segment boundaries. These limitations are resolved by exploiting intra-query correlations via HMP throughout $( h ) _ { q } ^ { \dot { l } }$ for levels $l ~ = ~ 1 , \ldots , L _ { m p } ~ \leq ~ L$ (Fig. 4 e). The additional use of $L _ { m p } > 1$ low-res representations leads to smoother and more consistent predictions w.r.t. $L _ { m p } = 1 ( \mathrm { i . e }$ PARMESAN’s original approach), as well as being capable of correcting large metrication artifacts and perturbations. HMP employs a coarse-to-fine refinement strategy w.r.t. spatial resolutions, where higher levels apply global modifications and lower levels apply local modifications. As in processing step #1, feature pyramid levels are traversed downward from $l = L _ { m p }$ to l = 1.

In our inference pipeline, a static HMP graph is calculated during the initialization phase leveraging MEOW’s k-NN search capabilities. Given $( h ) _ { q } ^ { l }$ both as query and working memory content, we find κ NNs for every query voxel within the query itself. Consequently, the top-1 NN is the respective query voxel itself. Search is dense where $\forall \mu _ { i } \in ( \mu ) _ { q } ^ { l } , \mu _ { i } = 1$ (i.e., all voxels labeled). For calculating HMP graph edges $( e ) _ { j } ^ { l }$ , NN indices $( i ) _ { j } ^ { l }$ with similarity scores $( s ) _ { j } ^ { l }$ are returned instead of label-retrieval and reduction (also see Appendix I, Algorithm 1).

For every HMP level, node features y and weak labels $z ^ { \pm }$ are first interpolated to the respective resolution Ω<sup>l</sup>. Then, $\eta _ { m p }$ message passing steps are taken while keeping weakly labeled nodes fixed. Update strength $\lambda _ { m p }$ origins from graph neural network theory [92] and can be controlled interactively by $\mathcal { U } .$

Through message passing, the over-smoothing problem [93] tends to make node features more similar. This can lead to vanishing segments, especially when SoIs are small. We mitigate this effect by combining artificial sharpening with another Hopfield layer, as well as applying mass-conservation in terms of foreground voxels. Sharpening strength around the classification threshold τ is controlled by $\sigma _ { m p }$ . A continuous Hopfield layer $( e ) _ { j } ^ { l } = \operatorname { s o f t m a x } _ { \kappa } ( \beta _ { m p } \cdot ( s ) _ { j } ^ { l } )$ without learnable parameters is controlled by a hyperparameter $\beta _ { m p } .$ , which determines how strongly graph edges $( e ) _ { i } ^ { l }$ are committed towards more similar NNs. Larger values of $\beta _ { m p }$ effectively make the HMP graph more sparse.

Retaining certain quantities in the form of conservation laws is beneficial for solving problems related to dynamical systems [94]–[98]. To further mitigate over-smoothing, we employ mass-conservation in terms of voxel count by adjusting $\tau$ such that the number of foreground voxels remains constant w.r.t. raw prediction $y _ { r }$ . HMP dynamics can therefore only re-distribute mass without growing or shrinking the resulting segment.

## E. Open-World Learning with MEOW

We now focus on introducing MEOW as an open-world interactive segmentation (OW-IS) system, incrementally learning from its own predictions during deployment under user guidance. Building upon its flexible learning and inference framework, MEOW is uniquely suited for OW-IS. By extending the base architecture $( { \mathcal { F } } \cup { \mathcal { G } } )$ with a memory module $\mathcal { M } ,$ our system adapts and learns from novel structures via memory consolidation (see Fig. 2). Through this setup, MEOW supports multiple learning paradigms: incremental, continual, transfer, out-of-distribution, few-shot, and OW learning. Furthermore, our model allows fast learning and unlearning of individual samples at scale, as well as performing novelty detection (ND) and instance classification (IC).

Known vs. Unknown: Given the four OW class categories KKCs, KUCs, UKCs, and UUCs as defined in Section III-A, we further refine these categories based on their status relative to the backbone $\mathcal { F }$ (via training), the memory $\mathcal { M }$ (via consolidation), or both, yielding twelve distinct combinations. Since we keep learnable parameters $\theta$ frozen, the ”known” status w.r.t. $\mathcal { F }$ is governed by the scope of the pre-training data. Note that replacing or fine-tuning $\mathcal { F }$ necessitates re-embedding existing memory samples to maintain consistency across feature pyramids.

For the task of $\operatorname { I S } , z _ { g t }$ are ground truth binary segmentation masks, whereas ground truth class indices $c _ { g t }$ and associated semantic information are treated as metadata. Our framework does not strictly require $\mathcal { U }$ to provide $c _ { g t }$ due to its ND and IC capabilities. Consequently, our method supports all twelve OW category combinations, as well as flexibly modifying and merging them throughout a model’s operational lifecycle.

Learning with MEOW: In our framework, an OW-learning system is initialized with an empty memory, where any incoming query is novel to M (i.e., an $\mathrm { U K C } _ { \mathcal { M } }$ or $\mathrm { U U C } _ { \mathcal { M } } )$ The query gets assigned a class index $c \geq 0 .$ , e.g., by U providing the respective $c _ { g t }$ . While $c > 0$ refer to known classes $( \mathrm { i . e . , K K C } _ { \mathcal { M } } ) , c = 0$ is regarded as the unknown class $( \mathrm { i . e . , K U C } _ { \mathcal { M } } )$

An open-set (OS) segmentation result y is obtained via IS (see Section III-D and green pipeline in Fig. 2). U can then judge segmentation quality and decide if the query and its prediction shall be learned (i.e., become a memory sample). A sample is learned by first creating a pseudo-label binary mask $z _ { p } = \mathbb { I } ( y \geq \tau )$ from the final prediction. The feature pyramid $( h ) _ { q } ^ { l }$ along with $x _ { q } , z _ { p } ,$ , and c are then stored in $\mathcal { M }$ . It can accommodate storing arbitrary sets of ground truth labels $z _ { g t }$ and pseudo-labels $z _ { p } ,$ allowing hybrid approaches such as initializing M with some ground truth labeled samples while learning pseudo-labeled samples during deployment.

Novelty Detection: Assuming a non-empty M, we apply ND and IC for predicting a class index c and deciding on whether a segmentation prediction $y _ { \mathcal { M } }$ from memory can be made. During the initialization phase, we calculate accumulated similarities $s _ { f * , j }$ of small pyramid center-patches $( h ) _ { q * } ^ { l } \subset ( h ) _ { q } ^ { l }$ and $( h ) _ { j * } ^ { l } \in ( \bar { h } ) _ { j } ^ { l }$ between a query patch and $j = 1 , \dots ,$ m memory patches:

$$
s _ { f * , j } = \prod _ { i = 1 } ^ { L } s _ { f * } ^ { i } ( h _ { q * } ^ { i } , h _ { j * } ^ { i } ) .\tag{4}
$$

Let $\mathcal { N } _ { \sf k }$ be the set of indices of the top-k most similar memory patches according to $\{ s _ { f * , j } \}$ . The average similarity of the query to these neighbors is

$$
s _ { f * } ^ { q } = \frac { 1 } { \mathsf { k } } \sum _ { j \in \mathcal { N } _ { \mathsf { k } } } s _ { f * , j } .\tag{5}
$$

To get a robust estimate for typical intra-memory similarities $s _ { f * } ^ { M }$ , we average all pairwise similarities between memory patches in $\mathcal { N } _ { \sf k }$ . A query is considered novel if M is empty or if $s _ { f * } ^ { q } < \zeta \cdot s _ { f * } ^ { M }$ with the novelty threshold $\zeta \in [ 0 , 1 ]$ being a hyperparameter. If a query is novel, no memory prediction $y _ { \mathcal { M } }$ is made and the query is considered unknown (i.e., associated with $c = 0 )$

Instance Classification: If a query is not novel, we perform IC by similarity-weighted voting over the class labels of the top-k NN patches to obtain a class prediction $c \geq 0$ . This task corresponds to a multi-class IC problem with a variable number of classes. If $c > 0$ , the query is a ${ \mathrm { K K C } } _ { { \mathcal { M } } }$ and being associated with the respective known class. If $c = 0$ , the query is a ${ \mathrm { K U C } } _ { { \mathcal { M } } }$

Given our focus on medical OW-IS, U typically has extensive domain expertise, including prior knowledge (i.e., semantic metadata) of the SoIs to be segmented (e.g., a specific pathology). In this case, U may verify the correctness of c or correct it by providing $c _ { g t }$ . When $c _ { g t }$ is provided upfront, ND and IC are skipped entirely. In scenarios where U can’t provide $c _ { g t }$ (i.e., UUCs), the predicted c is used to continue.

Memory Prediction: A working memory M is created by selecting z and $( h ) ^ { l }$ of all samples from M that either belong to the class c or are among the top-k NNs obtained from IC:

$$
M = \left\{ \left( z , ( h ) ^ { l } \right) _ { i } \in \mathcal { M } _ { * } \middle | \left( \left( c _ { i } = c \right) \wedge ( c > 0 ) \right) \vee i \in \mathcal { N } _ { \mathsf { k } } \right\}\tag{6}
$$

where $\mathcal { M } _ { * } = \big \{ \big ( z , ( h ) ^ { l } \big ) _ { i } \big | \big ( x , z , c , ( h ) ^ { l } \big ) _ { i } \in \mathcal { M } \big \}$ . If M is empty, no memory prediction $y _ { \mathcal { M } }$ can be made. If M is non-empty, a dense prediction is made and integrated by fusing it with the base prediction $y _ { b }$ as described in Section III-D. Note that $y _ { \mathcal { M } }$ can be calculated fully automatically without a user, thereby enabling method variants for OS and OW semantic or panoptic segmentation. While M holds a single sample (namely the query itself) for making a base prediction y<sub>b</sub> in processing step #1, M may hold many samples for making a memory prediction $y _ { \mathcal { M } }$

Finally, U provides a binary decision (accept or reject) on whether to store the resulting segmentation $y$ in $\mathcal { M }$ thereby establishing an open-world learning loop that enhances predictive performance for subsequent queries.

## F. Additional Improvements to PARMESAN

Similar to MEOW, PARMESAN [14] is a transductive reasoning framework extending non-parametric classification [99] and kernel-based detection [100], [101] to dense prediction tasks (e.g., semantic segmentation or depth estimation) for continual learning. PARMESAN makes non-interactive predictions using densely labeled, unpaired memory samples.

Starting from the original method, we introduce the following general improvements toward high-quality, fast, and efficient OW-IS for 3D data. These improvements are integral components of MEOW and are applied in all configurations and experiments throughout this work:

Extension to 3D: We extend the method to 3D volumetric data such as CT and MRI scans. This includes using 3D feature pyramids from respective feature extractors, as well as updating G to efficiently operate on 3D grids.

Feature Compression: Storing raw feature pyramids $( h ) ^ { l }$ requires substantial hardware memory resources, especially in 3D. To keep requirements low, PARMESAN uses simple channel trimming to reduce the number of channels. In contrast, MEOW uses linear projections for fast and efficient compression of $( h ) ^ { l }$ in the channel dimension. Given one or more samples, optimal projection weights for every level l are obtained through a linear autoencoder [102] and singular value decomposition [103], [104]. Compression weights are re-calculated for every incoming query and remain fixed once $\mathcal { M }$ is being used. If no data is available, projection weights may be randomly sampled instead [105].

Kernels: For k-NN search between query pyramid voxels and memory contents, we discard the originally proposed box-shaped search kernel in favor of two new kernels, the cross-kernel and the flexi-kernel, which reduce metrication artifacts in predictions while at the same time retaining the same hardware efficiency as the box-shaped kernel (details in Appendix I).

Test-Time Augmentation: To increase robustness of extracted features, we extend a feature-ensembling approach for image classification [106] to our OW-IS framework by aligning and averaging $( h ) ^ { l }$ obtained from several different augmentations. In this work, augmentations are rotations of $x _ { q }$ that are restricted to orthogonal increments along each axis $( { \mathrm { i . e . , ~ } } \left\{ 0 ^ { \circ } , 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \right\} )$ . Rotated variants are embedded, aligned, and averaged with the original $( h ) ^ { l }$ to obtain a single augmented feature pyramid. To ensure consistency, a fixed augmentation scheme is applied to both queries and memory samples. Notably, combining feature pyramids is much faster than our original TTA approach combining predictions [14], as a call to $\mathcal { G }$ now has constant time complexity $\mathcal { O } ( 1 )$ instead of linear complexity $\mathcal O ( a )$ w.r.t. the number of augmentations a. This is especially relevant for the alternating interaction phase, where low-latency is required when interacting with $\mathcal { U } .$

Exploration vs. Exploitation in k-NN Search: Let us consider a fixed level l with the total number of query voxels $p = \prod _ { d } \Omega ^ { l }$ . With PARMESAN, k-NN search between p queries and n stored memory patterns is performed independently for all queries using cosine similarity. This corresponds to greedy search with the goal to find the best match for every query voxel. However, greedy search typically leads to the hubness problem [107], where, in high-dimensional k-NN search a small subset of patterns (”hubs”) frequently appear as NNs to an unusually large number of query voxels. Moreover, there is no ground truth 1:1 mapping between queries and stored patterns in the case of unpaired correspondence matching. Dual-softmax matching [43], [44] is therefore not applicable in our case.

Instead, we propose regularized cosine matching using a single softmax. By combining soft mutual nearest neighbor filtering [108] with parameter-free modern Hopfield retrieval [81] and the regularization term of the dual-softmax matcher [43], we introduce the following regularizer to resolve hubness and enable an exploration vs. exploitation trade-off for unpaired correspondence matching:

$$
s _ { r e g } : = \mathrm { s o f t m a x } _ { p } \left( \left( \log p \right) ^ { \beta _ { r e g } } \cdot s _ { c o s } ( h _ { q } , h _ { j } ) \right) .\tag{7}
$$

This regularizer re-distributes matching weights over all $p$ queries (also see Appendix I, Algorithm 1). The commitment hyperparameter $\beta _ { r e g }$ determines how exclusive a stored memory pattern is to an individual query voxel w.r.t. all other query voxels. Since k-NN search is carried out hierarchically and locally across memory feature pyramids, the commitment factor $( \log p ) ^ { \beta _ { r e g } }$ concentrates the matching process on spatially local regions of query feature maps.

## IV. EXPERIMENTS

In this section we present our experiments split across three domains: closed-set, open-set, and open-world.

Data: We consider different datasets in our experiments, covering a diverse set of anatomical structures and modalities, as well as non-medical 3D data. Importantly, our dataset choices allow addressing the four categories of known and unknown classes in the OW setting as defined in Section III-E. For performance analysis on known classes (KKCs), we use the organ classes liver, kidney, pancreas, and spleen from the TotalSegmentator [16], MSD [109], and KiTS23 [110] datasets, respectively. For performance analysis on unknown classes and domains (KUCs, UKCs and UUCs), we use the Mouse $\mu \mathrm { C T }$ [111] and InsectAnatomy [112] $\mu \mathrm { C T }$ datasets, as well as the MSD BRATS [109] MRI brain tumor dataset. We provide detailed information about used datasets and splits in Appendix II.

Virtual User: For automated evaluation, we employ a virtual user $\mathcal { U } _ { v }$ whose role is mimicking typical behavior of a human user $\boldsymbol { \mathcal { U } } _ { h }$ (see Appendix III for details). By default, $\mathcal { U } _ { v }$ has access to ground truth class indices $c _ { g t }$ and segmentation masks $z _ { g t }$ for simulating clicks and adjusting interactive hyperparameters $\pmb { \iota } .$ For fair comparisons to baseline methods, we let $\mathcal { U } _ { v }$ only perform clicks without any brushing.

Metrics: We use the Dice-Sørensen coefficient [113], [114] (DSC) for measuring segmentation quality. Interaction with a virtual user $\mathcal { U } _ { v }$ stops when $D S C _ { m i n } = 0 . 9 5$ is reached within a maximum of $n _ { m a x } = 8$ clicks for segmenting any SoI. We define $D S C ^ { * }$ as the segmentation quality after $n$ clicks $( \mathrm { e . g . }$ $D S C ^ { 8 }$ for $n = 8 )$ . The early success rate $( E R ,$ , higher is better) is the proportion of SoIs that require $< n$ clicks before interaction stops, i.e., reaching minimum performance before n clicks. The number of clicks metric $( N o C ^ { * }$ [115]) measures how many clicks were needed to reach a certain performance (e.g., $N o C ^ { 9 0 }$ for $\ D S C \ = \ 0 . 9 )$ . In addition to $N o C ^ { * }$ , we measure the success rate (SR, higher is better), defined as the fraction of successfully segmented SoIs (i.e., those who reached $D S C = * )$ relative to the total number of evaluated SoIs. Since $N o C ^ { * }$ is exclusively computed on successfully segmented SoIs, reporting it without $S R$ can lead to biased or misleading interpretations. To address known limitations of $N o C ^ { * }$ and $D S C ^ { * }$ [116], [117], we propose two novel effort-aware metrics – rate-penalized $N o C _ { r } ^ { * }$ and rate-corrected $D S C _ { r } ^ { * }$

$$
N o C _ { r } ^ { * } = S R \cdot N o C ^ { * } + ( 1 - S R ) \cdot n _ { m a x } \ ,\tag{8}
$$

$$
D S C _ { r } ^ { * } = E R \cdot D S C _ { f i n a l } + ( 1 - E R ) \cdot D S C ^ { * } \ .\tag{9}
$$

$N o C _ { r } ^ { * }$ reflects the effort that was spent on fails $( \mathrm { i . e . }$ , not reaching the respective performance within $n _ { m a x }$ clicks), while $D S C _ { r } ^ { * }$ reflects the contribution from early successes. As stated in Section III-A, our overall objective is maximizing segmentation quality w.r.t. an SoI while minimizing required user guidance. To account for this, we use the cumulative mean performance after consecutive user actions:

$$
D S C _ { c r } ^ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } D S C _ { r } ^ { i } \ .\tag{10}
$$

This metric aligns with recent views that area under the curve metrics are much more relevant for the task of IS [116].

Baselines: As the first to propose OW-IS, respective baseline methods are absent. We are therefore limited to comparing our method against existing OS-IS baselines, as well as performing extensive ablation studies in the OW case. For consistent comparisons with baselines, we integrate nnInteractive [30], SAM-Med3D [35], and VISTA3D [31] into our framework and apply their default hyperparameters. Although these baseline methods were trained on multiple domains, they lack explicit strategies for continually learning new domains and classes without encountering the catastrophic forgetting problem. Given that nnInteractive was trained on 30.7×, 53.6×, and 5.6× more data relative to feature extractors used for MEOW (see Table II), we consider nnInteractive as an empirical upper bound on segmentation performance.

MEOW Setup: To demonstrate that our method is backbone-agnostic and compatible with various hourglass-style architectures, we evaluate it using three distinct feature extractors: U-Net SuPreM $( \mathcal { F } _ { 1 } )$ [15], TotalSegmentator $( \mathcal { F } _ { 2 } )$ [16], and the automatic-branch sub-module of VISTA3D $( \mathcal { F } _ { 3 } )$ [31]. Note that unlike our baseline methods, neither feature extractor is pre-trained explicitly for IS, but for multi-class semantic segmentation. Note that feature extractors requiring weak user labels as inputs $( \mathrm { e . g . }$ , nnInteractive) are incompatible with MEOW, as they would necessitate repeated feature extraction after every user interaction. For test-time augmentation (TTA), we augment both queries and memory samples by averaging the original feature map $( h ) ^ { l }$ in addition to two $9 0 °$ rotated variants (rotated around the depth and width axes, respectively). In the presence of $\mathcal { M } ,$ we store $\begin{array} { l } { m } \end{array} = \begin{array} { l } { 6 4 } \end{array}$ uncompressed (i.e., $L _ { s p } = 0 )$ memory samples by default. For 3D data, we set the k-NN search reduction factor $\phi = 0 . 2 5$ in order to be scalable to high resolutions while retaining quasi-constant hardware memory consumption [14]. Unless stated otherwise, we apply all improvements discussed in Section III-F.

TABLE II  
MODEL SIZE AND TRAINING DATA EFFICIENCY OF DIFFERENT METHODS. FOR MEOW FEATURE EXTRACTORS ARE FROZEN (MARKED WITH \*).
<table><tr><td>Method</td><td>Learnable Params</td><td>Training Volumes</td><td>Training Masks</td><td>Memory Samples</td></tr><tr><td>SAM-Med3D</td><td>100.51 M</td><td>22 K</td><td>143 K</td><td>=</td></tr><tr><td>VISTA3D</td><td>217.96M</td><td>11.5 K</td><td> ${ > } 1 . 3 4 \mathrm { ~ M ~ }$ </td><td></td></tr><tr><td>nnInteractive</td><td>102.36 M</td><td>~61.3 K</td><td> ${ \sim } 6 8 1 \mathrm { ~ K ~ }$ </td><td></td></tr><tr><td>MEOW  $( \mathcal { F } _ { 1 } \& \mathcal { M } )$ </td><td>*19.07 M</td><td>2.1 K</td><td> $\sim 5 2 \ \bf K$ </td><td> $\leq 1 2 8$ </td></tr><tr><td>MEOW  $( \mathcal { F } _ { 2 } \& \mathcal { M } )$ </td><td> $^ { * } 3 1 . 2 1 \mathrm { ~ M ~ }$ </td><td>1.2 K</td><td> ${ < 1 4 1 \mathrm { ~ K ~ } }$ </td><td> $\leq 1 2 8$ </td></tr><tr><td>MEOW  $( \mathcal { F } _ { 3 } \& \mathcal { M } )$ </td><td> $^ { * } 2 1 7 . 2 6 \mathrm { ~ M ~ }$ </td><td>11.5 K</td><td> ${ > } 1 . 3 4 \mathrm { ~ M ~ }$ </td><td> $\leq 1 2 8$ </td></tr></table>

Hyperparameters: To find an optimal set of interactive and non-interactive hyperparameters, we leverage Bayesian optimization [118]. We use organ KKCs (i.e., liver, pancreas, spleen, and kidney) for evaluating our objective function $f ( \mathbf { x } )$ on a small, independent training split. We maximize $D S C _ { c r } ^ { 8 }$ until convergence by optimizing the parameter set $\mathbf { x } =$ $\{ \tau , \omega , \gamma , \xi ^ { - } , \beta _ { n n } , \beta _ { r e g } , \alpha _ { f u s e } , \sigma _ { m p } , \lambda _ { m p } , L _ { m p } , \beta _ { m p } , \eta _ { m p } ,$ TTA} (see Appendix IV for details).

Technical Setup: All methods including baselines are implemented in a unified PyTorch [119] framework [120]–[124] and evaluated using individual NVIDIA GPUs (models RTX 3090, A40, or A100). Every experimental setup satisfies a memory constraint of $\leq ~ 2 4$ GB VRAM and all timing measurements are benchmarked on a single workstation featuring an RTX 3090, reflecting a consumer-grade hardware setup.

## A. Closed-Set Interactive Segmentation

General Improvements: Improving upon PARMESAN, we introduce several architectural modifications that increase overall performance. To evaluate individual contributions, we conduct extensive ablation studies on the MSD+KiTS23 organ CT data and summarize our results in Table III. Transitioning from original settings to our proposed settings leads to substantial performance gains. The largest performance gains are obtained from feature compression and new kernels. Replacing simple channel trimming with a linear autoencoder preserves representative feature information (A1, A9, A17). Replacing the $2 ^ { d }$ box-shaped kernel with our improved kernels allows for better correspondence matching (A2, A10, A18). While individual adjustments to similarity weighting, balancing, and regularized matching yield mixed results, synergistic effects (A7, A15, A23) suggest that the interplay between all components is crucial for robust IS.

To demonstrate the effectiveness of hierarchical message passing (HMP), we conduct ablation studies with degraded base predictions $y _ { b }$ preceding HMP using $\mathcal { F } _ { 1 }$ without M. We use perturbed base predictions to isolate the contribution of HMP, test robustness under realistic conditions, and highlight the benefit of hierarchical vs. single-level MP. Our results are presented in Table IV and Fig. 5. While the original PARMESAN approach used $L _ { m p } ~ = ~ 1$ to correct minor artifacts, increasing $L _ { m p }$ allows to correct strong perturbations of base predictions. This iterative correction is consistent with the behavior of message-passing graph neural networks, which can be viewed as performing a form of local fixed-point inference on the voxel graph [47], [125].

TABLE III  
RESULTS FOR ABLATION STUDIES, DEMONSTRATING VARIOUS IMPROVEMENTS WHEN APPLYING MODIFICATIONS TO PARMESAN.
<table><tr><td>ID</td><td>Method</td><td>Ablation</td><td> $\overline { { \mathbf { O r i g i n a l } \to \mathbf { O u r s } } }$ </td><td> $\mathbf { \overline { { D S C _ { c r } ^ { s } \uparrow ( E R \uparrow ) } } }$  一</td></tr><tr><td>A0</td><td> $\overline { { \mathcal { F } _ { 1 } \& \mathcal { M } } }$ </td><td>PARMESAN</td><td>all original</td><td> $\overline { { 0 . 8 0 5 \ ( 0 . 1 0 6 ) } }$ </td></tr><tr><td>A1</td><td> $\mathcal { F } _ { 1 } \& \mathcal { M }$ </td><td>compression</td><td> $\mathrm { t r i m }  \mathrm { l i n e a r ~ A E }$ </td><td>0.851 (0.234)</td></tr><tr><td>A2</td><td> $\bar { \mathcal { F } _ { 1 } } \& \mathcal { M }$ </td><td>kernel</td><td> $\mathbf { b o x }  c r o s s \mathbf { + } f l e x i$ </td><td>0.859 (0.170)</td></tr><tr><td>A3</td><td> $\mathcal { F } _ { 1 } \& \mathcal { M }$ </td><td>similarity</td><td> $\omega = 1  0 . 9 0$ </td><td>0.814 (0.096)</td></tr><tr><td>A4</td><td>F1&amp;M</td><td>balancing</td><td> $\gamma = 0  - 0 . 9 9$ </td><td>0.808 (0.106)</td></tr><tr><td>A5</td><td> $\mathcal { F } _ { 1 } \& \mathcal { M }$ </td><td>matcher</td><td> $\mathrm { c o s }  \mathrm { r e g . } \mathrm { c o s }$ </td><td>0.813 (0.125)</td></tr><tr><td>A6</td><td> $\mathcal { F } _ { 1 } \& \mathcal { M }$ </td><td>HMP</td><td> $L _ { m p } = 1  2$ </td><td>0.823 (0.115)</td></tr><tr><td>A7</td><td> $\mathcal { F } _ { 1 } \& \mathcal { M }$ </td><td>MEOW</td><td>all ours</td><td>0.892 (0.541)</td></tr><tr><td>A8</td><td> $\overline { { \mathcal { F } _ { 2 } \& \mathcal { M } } }$ </td><td>PARMESAN</td><td>all original</td><td>0.660 (0.0)</td></tr><tr><td>A9</td><td> $\mathcal { F } _ { 2 } \& M$ </td><td>compression</td><td> $\mathrm { t r i m }  \mathrm { l i n e a r ~ A E }$ </td><td>0.784 (0.106)</td></tr><tr><td>A10</td><td> $\mathcal { F } _ { 2 } \& M$ </td><td>kernel</td><td> $\mathbf { b o x }  c r o s s \mathbf { + } f l e x i$ </td><td>0.752 (0.019)</td></tr><tr><td>A11</td><td> $\mathcal { F } _ { 2 } \& M$ </td><td>similarity</td><td> $\omega = 1  0 . 5 0$ </td><td>0.713 (0.0)</td></tr><tr><td>A12</td><td> $\mathcal { F } _ { 2 } \& \mathcal { M }$ </td><td>balancing</td><td> $\gamma = 0  - 0 . 9 9$ </td><td>0.658 (0.0)</td></tr><tr><td>A13</td><td> $\mathcal { F } _ { 2 } \& \mathcal { M }$ </td><td>matcher</td><td> $\mathrm { c o s }  \mathrm { r e g . } \mathrm { c o s }$ </td><td>0.653 (0.0)</td></tr><tr><td>A14</td><td> $\mathcal { F } _ { 2 } \& M$ </td><td>HMP</td><td> $L _ { m p } = 1  2$ </td><td>0.685 (0.0)</td></tr><tr><td>A15</td><td> $\mathcal { F } _ { 2 } \& M$ </td><td>MEOW</td><td> $\mathrm { \ a l l \ o u r s }$ </td><td>0.836 (0.199)</td></tr><tr><td>A16</td><td> $\mp _ { 3 } \& \mathcal { M }$ </td><td>PARMESAN</td><td> $\overline { { \mathrm { \bf { a l l } \ o r i g i n a l } } }$ </td><td>0.662 (0.010)</td></tr><tr><td>A17</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>compression</td><td> $\mathrm { t r i m }  \mathrm { l i n e a r ~ A E }$ </td><td>0.737 (0.019)</td></tr><tr><td>A18</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>kernel</td><td> $\mathbf { b o x }  c r o s s \mathbf { + } f l e x i$ </td><td>0.723 (0.019)</td></tr><tr><td>A19</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>similarity</td><td> $\omega = 1  0 . 6 5$ </td><td>0.697 (0.0)</td></tr><tr><td>A20</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>balancing</td><td> $\gamma = 0  - 0 . 5 5$ </td><td>0.653 (0.010)</td></tr><tr><td>A21</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>matcher</td><td> $\mathrm { c o s }  \mathrm { r e g . } \mathrm { c o s }$ </td><td>0.678 (0.010)</td></tr><tr><td>A22</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>HMP</td><td> $L _ { m p } = 1  2$ </td><td>0.696 (0.010)</td></tr><tr><td>A23</td><td> $\mathcal { F } _ { 3 } \& \mathcal { M }$ </td><td>MEOW</td><td>all ours</td><td>0.789 (0.048)</td></tr></table>

TABLE IV

RESULTS FOR ABLATION STUDIES (TOP BLOCK: $D S C _ { c r } ^ { 8 }$ BOTTOM BLOCK: $D S C _ { r } ^ { 8 } \mathrm { \cdot }$ ), DEMONSTRATING THE EFFECTIVENESS OF HMP UNDER STRONG (MASS-CONSERVING) PERTURBATIONS OF 3D BASE PREDICTIONS y<sub>b</sub> PRECEDING HMP.
<table><tr><td> $\scriptstyle \mathbf { L _ { m p } }$ </td><td>Unperturbed</td><td>Blur</td><td>Noise</td><td>Translate</td><td>Rotate</td><td>Shear</td></tr><tr><td>0</td><td>0.814</td><td>0.655</td><td>0.403</td><td>0.531</td><td>0.364</td><td>0.492</td></tr><tr><td>1</td><td>0.830</td><td>0.712</td><td>0.828</td><td>0.578</td><td>0.394</td><td>0.563</td></tr><tr><td>2</td><td>0.840</td><td>0.731</td><td>0.829</td><td>0.644</td><td>0.448</td><td>0.639</td></tr><tr><td>3</td><td>0.840</td><td>0.746</td><td>0.823</td><td>0.710</td><td>0.521</td><td>0.691</td></tr><tr><td>0</td><td>0.833</td><td>0.707</td><td>0.487</td><td>0.554</td><td>0.361</td><td>0.499</td></tr><tr><td>1</td><td>0.856</td><td>0.703</td><td>0.851</td><td>0.612</td><td>0.390</td><td>0.571</td></tr><tr><td>2</td><td>0.868</td><td>0.712</td><td>0.855</td><td>0.686</td><td>0.444</td><td>0.647</td></tr><tr><td>3</td><td>0.872</td><td>0.732</td><td>0.851</td><td>0.749</td><td>0.515</td><td>0.704</td></tr></table>

![](images/982abfa181dbfed9d9e65382626ee78c97f849c7a873d51d2002032dafaab76e.jpg)  
Fig. 5. Qualitative results demonstrating the effectiveness of hierarchical message passing (HMP) under strong (mass-conserving) perturbations of 3D base predictions y<sub>b</sub> preceding HMP. Ground truth is indicated in contours.

TABLE V  
CLOSED-SET IS PERFORMANCE RESULTS FOR KKCS ON MEDICAL DATASETS (TOP: TOTSEG ORGAN, BOTTOM: MSD+KITS23) FOR ORGAN CLASSES LIVER, KIDNEY, PANCREAS, AND SPLEEN.
<table><tr><td>ID</td><td>Method</td><td> $\mathbf { \overline { { N o C } } } _ { \mathbf { r } } ^ { \mathbf { 9 0 } }$ </td><td>↓  $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { \mathbf { r } } ^ { \mathbf { s } } } }$  ↑</td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { \mathbf { c r } } ^ { \mathbf { T } } } }$  ↑</td><td> $\overline { { { \bf { D } } { \bf { S } } { \bf { C } } _ { \bf { c r } } ^ { 8 } \mathrm { \Lambda } } } $ </td></tr><tr><td>B0.0</td><td>nnInteractive</td><td>2.518</td><td>0.940</td><td>0.854</td><td>0.920</td></tr><tr><td>B0.1</td><td>VISTA3D</td><td>5.296</td><td>0.844</td><td>0.702</td><td>0.797</td></tr><tr><td>B0.2</td><td>SAM-Med3D</td><td>4.100</td><td>0.847</td><td>0.805</td><td>0.842</td></tr><tr><td>B0.3</td><td>MEOW  $( \mathcal { F } _ { 1 } \& \mathcal { M } )$ </td><td>4.184</td><td>0.858</td><td>0.792</td><td>0.831</td></tr><tr><td>B0.4</td><td>MEOW  $( \mathcal { F } _ { 1 } )$ </td><td>5.321</td><td>0.795</td><td>0.637</td><td>0.740</td></tr><tr><td>B0.5</td><td>MEOW  $( \mathcal { F } _ { 2 } \& \mathcal { M } )$ </td><td>4.529</td><td>0.838</td><td>0.767</td><td>0.817</td></tr><tr><td>B0.6</td><td>MEOW  $( \mathcal { F } _ { 2 } )$ </td><td>5.870</td><td>0.779</td><td>0.593</td><td>0.733</td></tr><tr><td>B0.7</td><td>MEOW  $( \mathcal { F } _ { 3 } \& \mathcal { M } )$ </td><td>6.322</td><td>0.786</td><td>0.676</td><td>0.751</td></tr><tr><td>B0.8</td><td>MEOW  $( \mathcal { F } _ { 3 } )$ </td><td>7.640</td><td>0.692</td><td>0.469</td><td>0.628</td></tr><tr><td>B1.0</td><td>nnInteractive</td><td>3.106</td><td>0.918</td><td>0.783</td><td>0.886</td></tr><tr><td>B1.1</td><td>VISTA3D</td><td>5.436</td><td>0.834</td><td>0.747</td><td>0.811</td></tr><tr><td>B1.2</td><td>SAM-Med3D</td><td>3.585</td><td>0.856</td><td>0.772</td><td>0.847</td></tr><tr><td>B1.3</td><td>MEOW  $( \mathcal { F } _ { 1 } \& \mathcal { M } )$ </td><td>3.316</td><td>0.903</td><td>0.868</td><td>0.892</td></tr><tr><td>B1.4</td><td>MEOW  $( \mathcal { F } _ { 1 } )$ </td><td>3.928</td><td>0.873</td><td>0.781</td><td>0.841</td></tr><tr><td>B1.5</td><td>MEOW  $( \mathcal { F } _ { 2 } \& \mathcal { M } )$ </td><td>4.021</td><td>0.857</td><td>0.787</td><td>0.836</td></tr><tr><td>B1.6</td><td>MEOW  $( \mathcal { F } _ { 2 } )$ </td><td>5.272</td><td>0.811</td><td>0.634</td><td>0.762</td></tr><tr><td>B1.7</td><td>MEOW  $( \mathcal { F } _ { 3 } \& \mathcal { M } )$ </td><td>6.028</td><td>0.816</td><td>0.738</td><td>0.790</td></tr><tr><td>B1.8</td><td>MEOW  $( \mathcal { F } _ { 3 } )$ </td><td>7.0</td><td>0.736</td><td>0.502</td><td>0.673</td></tr></table>

Interactive Segmentation: We evaluate closed-set performance of all methods on organ CT classes (KKCs) and report results in Table V. We achieve state of the art performance on the MSD+KiTS23 organ segmentation task (B1.3). MEOW otherwise performs similar to SAM-Med3D and VISTA3D baselines with feature extractors pre-trained on substantially less data (see Table II). Using M increases performance in all setups while reducing the required $N o C .$ . This is due to memory predictions providing high-quality segmentations before alternating user interaction, meaning that U only needs to provide corrective guidance during the interaction phase (also see $D S C _ { c r } ^ { 1 }$ column in Table V). MEOW is robust w.r.t. the choice of F, with $\mathcal { F } _ { 1 }$ (U-Net SuPreM) yielding best overall results. MEOW is almost on par with VISTA3D while not using its decoder (B0.1, B0.7, B1.1, B1.7).

nnInteractive achieves the best overall performance across datasets. However, our framework offers distinct structural advantages for real-world deployment. First, MEOW is an OW-IS method, whereas nnInteractive is designed as an OS-IS method without dedicated OW-learning strategies. Second, our system is inherently more flexible, featuring a modular design that allows for the seamless replacement of backbones and memory modules without re-training. Third, our approach is designed for fast learning and stable knowledge retention. Moreover, nnInteractive was trained on a vastly larger training corpus relative to $\mathcal { F } _ { 1 } , ~ \mathcal { F } _ { 2 }$ , and $\mathcal { F } _ { 3 }$ (see Table II), underlining MEOW’s data-efficiency. These backbones were further optimized for multi-class segmentation rather than IS. This introduces an additional transfer learning task that we successfully resolve, thereby enabling the system’s superior adaptability and OW-learning capabilities.

## B. Open-Set Interactive Segmentation

We now investigate open-set (OS) capabilities of our method, i.e., segmenting unknown classes in the context of UKCs and UUCs. We select Mouse $\mu \mathrm { C T }$ [111] and InsectAnatomy [112] datasets for quantitative evaluation, since none of our baselines, nor feature extractors were pre-trained on them. Although nnInteractive was trained on different micro-CT data, we still consider our selected sets as unknown to this baseline. Performance results are summarized in Table VI. Note that for MEOW, M is not incorporated in this setting, since the pure OS scenario provides no samples to store. The working memory M is therefore filled exclusively with the query itself for interpreting user guidance.

TABLE VI  
OPEN-SET IS PERFORMANCE RESULTS FOR UKCS FOR UNKNOWN CT DATASETS (TOP: MOUSE $\mu { \mathrm { C T } } ,$ BOTTOM: INSECTANATOMY). WE USE ALL ORGAN CLASSES FOR MOUSE $\mu \mathrm { C T }$ AND THE SINGLE BRAIN CLASS FOR INSECTANATOMY. NOTE THAT WE DO NOT USE M FOR MEOW.
<table><tr><td>ID</td><td>Method</td><td> $\mathbf { \widetilde { N o C } } _ { \mathbf { r } } ^ { \mathbf { 9 0 } }$  →</td><td> $\mathbf { \overline { { D S C } } } _ { \mathbf { r } } ^ { 8 }$  ←</td><td> $\mathbf { \overline { { D S C _ { c r } ^ { 1 } } } } \uparrow$ </td><td> $\overline { { { \bf D S C } _ { \mathrm { c r } } ^ { 8 } \uparrow } }$ </td></tr><tr><td>B2.0</td><td>nnInteractive</td><td>4.880</td><td>0.894</td><td>0.650</td><td>0.826</td></tr><tr><td>B2.1</td><td>VISTA3D</td><td>7.990</td><td>0.537</td><td>0.166</td><td>0.422</td></tr><tr><td>B2.2</td><td>SAM-Med3D</td><td>7.698</td><td>0.673</td><td>0.485</td><td>0.651</td></tr><tr><td>B2.3</td><td>MEOW  $( \mathcal { F } _ { 1 } )$ </td><td>7.760</td><td>0.747</td><td>0.495</td><td>0.668</td></tr><tr><td>B2.4</td><td>MEOW  $( \mathcal { F } _ { 2 } )$ </td><td>7.990</td><td>0.694</td><td>0.340</td><td>0.583</td></tr><tr><td>B2.5</td><td>MEOW  $( \mathcal { F } _ { 3 } )$ </td><td>7.990</td><td>0.710</td><td>0.474</td><td>0.622</td></tr><tr><td>B3.0</td><td>nnInteractive</td><td>4.280</td><td>0.940</td><td>0.629</td><td>0.859</td></tr><tr><td>B3.1</td><td>VISTA3D</td><td>7.880</td><td>0.664</td><td>0.209</td><td>0.503</td></tr><tr><td>B3.2</td><td>SAM-Med3D</td><td>8.0</td><td>0.708</td><td>0.522</td><td>0.688</td></tr><tr><td>B3.3</td><td>MEOW (F1)</td><td>7.800</td><td>0.790</td><td>0.537</td><td>0.715</td></tr><tr><td>B3.4</td><td>MEOW (F2)</td><td>7.960</td><td>0.747</td><td>0.553</td><td>0.684</td></tr><tr><td>B3.5</td><td>MEOW  $( \mathcal { F } _ { 3 } )$ </td><td>8.0</td><td>0.719</td><td>0.478</td><td>0.671</td></tr></table>

In the OS scenario, our framework systematically outperforms the VISTA3D and SAM-Med3D baselines. Notably, using MEOW with a VISTA3D sub-module for feature extraction $( \mathcal { F } _ { 3 } )$ yields a substantial performance gain over the full VISTA3D method (B2.1, B2.5, B3.1, B3.5). These results demonstrate that the learned interactive branch in VISTA3D can be effectively replaced by our parameter-free transduction approach, simultaneously reducing learnable parameters, enabling dynamic hyperparameters, and enhancing OS-IS performance.

nnInteractive performs best among all evaluated methods for the reasons discussed above. Crucially, MEOW is not being leveraged to its full architectural potential in this specific comparison, partially accounting for this performance gap and identifying the integration of M as a critical mechanism.

We perform qualitative tests of MEOW w.r.t. user guidance and visualize results in Fig. 6. We demonstrate IS on KKCs and UKCs using diverse user actions. Moreover, we demonstrate IS under increasingly extreme domain shifts and visualize qualitative results in Fig. 7.

MEOW is capable of instance classification by leveraging stored samples with known metadata (i.e., KKCs / UKCs). Given 4 organ classes liver, kidney, pancreas, and spleen, we find instance classification accuracies of 0.909 (B0.3), 0.966 (B0.5), 0.923 (B0.7), 0.993 (B1.3), 0.999 (B1.5), and 0.986 (B1.7) (see Table V).

Real-Time Capabilities: IS methods should be designed to interact with U in real-time. We therefore report inference speed $t _ { i n f }$ and learning speed $t _ { l e a r n }$ (if applicable) in Table VII. We consider any method with $t _ { i n f } ~ < ~ 1$ seconds as real-time IS and any method with $t _ { l e a r n } ~ < ~ 1$ seconds as real-time learning. MEOW meets both of these conditions. Notably, our approach can learn a sample as fast as $t _ { l e a r n } = 7 . 6 1 \pm 0 . 1 8$ milliseconds through memory consolidation, substantially outperforming standard fine-tuning strategies that typically require several minutes to hours [77].

![](images/e9c4961456766e63f3fedf7f80218233cda908ea5edcbf95bc4b981377042d68.jpg)  
Fig. 6. Qualitative demonstrations of user interactions (blue: model prediction, green/red: weak positive/negative labels $z ^ { \pm }$ , contours: ground truth). With MEOW, clicks and brushes can be mixed flexibly across slices and axes.

TABLE VII  
MEAN AND STANDARD DEVIATION OF INFERENCE SPEED $t _ { i n f }$ (PER MODEL PREDICTION) AND LEARNING SPEED $t _ { l e a r n }$ (PER SAMPLE) AT A RESOLUTION OF 128<sup>3</sup> VOXELS USING AN RTX 3090 GPU. MEOW IS REAL-TIME CAPABLE W.R.T. INFERENCE AND LEARNING USING FROZEN FEATURE EXTRACTORS.
<table><tr><td>ID</td><td>Method</td><td>tinf [ms]</td><td> $\overline { { \bf { t } _ { 1 e a r n } } }$  [ms]</td></tr><tr><td>B0.0</td><td>nnInteractive</td><td> $\overline { { 2 8 9 . 7 5 0 \pm 4 . 6 1 0 } }$ </td><td>1</td></tr><tr><td>B0.1</td><td>VISTA3D</td><td> $2 6 7 . 4 3 0 \pm 6 . 6 4 7$ </td><td>一</td></tr><tr><td>B0.2</td><td>SAM-Med3D</td><td> $\mathbf { 8 . 9 8 0 \pm 0 . 0 9 9 }$ </td><td></td></tr><tr><td>B0.3</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 1 } \& \mathcal { M } )$ </td><td> $2 7 9 . 4 7 0 \pm 1 2 . 4 8 8$ </td><td> $9 . 6 2 \pm 0 . 1 1$ </td></tr><tr><td>B0.4</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 1 } )$ </td><td> $5 1 9 . 3 6 0 \pm 1 5 . 6 1 3$ </td><td></td></tr><tr><td>B0.5</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 2 } \& \mathcal { M } )$ </td><td> $5 0 1 . 3 1 5 \pm 1 3 . 0 1 8$ </td><td> ${ \bf 7 . 6 1 \pm 0 . 1 8 }$ </td></tr><tr><td>B0.6</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 2 } )$ </td><td> $5 1 7 . 2 0 5 \pm 1 . 8 4 6$ </td><td></td></tr><tr><td>B0.7</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 3 } \& \mathcal { M } )$ </td><td> $5 9 4 . 4 5 0 \pm 0 . 4 3 8$ </td><td> $5 7 . 7 5 \pm 0 . 7 1$ </td></tr><tr><td>B0.8</td><td> $\mathbf { M E O W } ( \mathcal { F } _ { 3 } )$ </td><td> $6 2 5 . 9 3 5 \pm 1 . 0 1 1$ </td><td>一</td></tr></table>

## C. Open-World Interactive Segmentation

We now investigate open-world (OW) capabilities of our method, i.e., continually learning and improving by leveraging model predictions from unknown classes. Recall that OW recognition requires having open-set (OS) capabilities, i.e., being able to make predictions for unknown classes in the first place. OW-learning is achieved by first predicting pseudo-labels $z _ { p }$ for incoming queries, followed by memory consolidation in M. As MEOW represents the first method to address the OW-IS problem, comparisons to baseline methods are limited to the OS-IS case, which translates to using MEOW without M $( m = 0 )$

During evaluation, we turn off any learning and evaluate on a fixed validation split. During training in an OW scenario, query images along with their final segmentation results are being learned one at a time. Since MEOW applies transductive reasoning from M, learned samples are instantly leveraged for processing any subsequent query. We report mean and standard deviation of performance measurements over two independent training folds (more details in Appendix II). Unless stated otherwise, our default setup for OW-IS experiments is $L _ { s p } = 1$ (sparse), $m \ = \ 1 2 8 , \ z _ { p }$ in M, treating samples as UKCs (specifically U having access to class indices), and using pre-trained F.

![](images/d28638eba8380ced17131c57583c7ffa640b75e1e4b59897ec1c132475ba3c95.jpg)  
Fig. 7. Qualitative demonstrations of open-domain IS capabilities of MEOW (predictions in colors). We use U-Net SuPreM pre-trained on human abdomen CT data $( \mathrm { i . e . , ~ } \mathcal { F } _ { 1 }$ without M) to segment various unknown objects under extreme domain shifts, as well as different data modalities [109], [126]–[128]. Individual results were obtained in ≈ 1 minute of human user interaction. For the brain tumor example, prediction and ground truth are visualized in blue and yellow, respectively $( \dot { D } S C ^ { 4 } = 0 . 8 2 \dot { 3 } )$ .

We present OW-IS experiment results in Fig. 8 and Table VIII. OW-learning generally improves segmentation performance for all feature extractors F. Increasing memory size m consistently improves results, though the benefit of OW-learning takes effect already at very small $m .$ . For example, we observe relative performance gains of up to $D S C _ { c r } ^ { 1 } ~ = ~ 0 . 1 1 8$ and $D S C _ { c r } ^ { 8 } ~ = ~ 0 . 1 3 3$ with as few as $m = 8$ stored samples (C1.4, C2.3), underlining MEOW’s data-efficiency in terms of OW-learning. Because performance usually converges later on, earlier clicks typically benefit more from OW-learning than later clicks. Since memory states are fully reversible, identical results would be obtained in the case of unlearning, i.e., removing samples from M.

![](images/b80c370df42336408311f0d09f10de1be76042f6bb2fd55a4a6ba7c913d4e781.jpg)  
Fig. 8. Demonstration of OW-learning for IS via memory consolidation. MEOW is continually trained by filling M with self-generated pseudo-labels, whic are then used for making future predictions via transductive reasoning. OW-learning typically leads to increased performance (black arrows), even with small memory modules.

MEOW outperforms VISTA3D by a large margin in all OW-IS experiments, even without M (C0.1, C0.3-5, C1.1, C1.3-5). MEOW also outperforms SAM-Med3D already without M on Mouse $\mu \mathrm { C T }$ (C0.2, C0.3) and InsectAnatomy (C1.2, C1.3). MEOW generally underperforms nnInteractive, except for $D S C _ { c r } ^ { 1 }$ on InsectAnatomy when using $\mathcal { F } _ { 1 }$ or $\mathcal { F } _ { 2 }$ with as few as $m = 8$ memory samples (C1.0, C1.3, C1.4). This result indicates that MEOW is highly efficient in terms of required user guidance.

Ablation Studies: We further investigate OW-IS capabilities of MEOW through comprehensive ablation studies (see Table IX) in order to address aspects regarding:

• using $z _ { g t }$ instead of $z _ { p }$ in M to study the performance gain from ground truth labels,

• treating samples as UUCs without any class-information (i.e., relying on novelty detection),

• using untrained backbones ${ \mathcal { F } } ^ { r }$ with random weights to investigate effectiveness of large-scale pre-training.

Using ground truth labels $z _ { g t }$ instead of self-generated pseudo-labels $z _ { p }$ tends to slightly increase performance due to degraded pseudo-label quality (C\*.6-8). Segmentation performance also remains stable when no metadata regarding an SoI is available (C\*.12-14).

We demonstrate that MEOW is highly feature-robust. Notably, when used in combination with randomized, untrained $\begin{array} { r l } { \mathcal { F } ^ { r } } & { { } ( \mathbf { C } ^ { * } . 9 – 1 1 ) } \end{array}$ our proposed framework still outperforms VISTA3D across datasets and feature extractors, while SAM-Med3D is being outperformed in one setup (C1.2 vs. C1.9). Through our hierarchical correspondence matching approach, both our similarity-based IS and transductive reasoning via M successfully exploits structural priors that are present in image data. We argue that MEOW can therefore even be applied in data-scarce domains without necessitating any feature extractor pre-training. This finding is supported by related work on random projections [129]–[131], reservoir computing [132], distributed representations [133], [134], and associative memory [81], [135].

TABLE VIII  
OW-IS RESULTS FOR OW-LEARNING CAPABILITIES ON MOUSE $\mu \mathrm { C T }$ DATA (TOP) AND INSECTANATOMY DATA (BOTTOM) WITH DIFFERENT NUMBER m OF PSEUDO-LABELED MEMORY SAMPLES (UKCS).
<table><tr><td>ID</td><td>Method</td><td>m (#SoIs)</td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { \mathbf { c r } } ^ { \mathbf { T } } \mathrm { ~ \hat { \Pi } } } }$ </td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { \mathbf { c r } } ^ { \mathbf { s } } \mathrm { ~ \uparrow ~ } } }$ </td></tr><tr><td>C0.0</td><td>nnInteractive</td><td></td><td> $\mathbf { 0 . 6 6 7 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { \overline { { 0 . 8 3 1 \pm 0 . 0 } } }$ </td></tr><tr><td>C0.1 C0.2</td><td>VISTA3D</td><td></td><td> $0 . 1 6 7 \pm 0 . 0 0 3$ </td><td> $0 . 4 2 5 \pm 0 . 0 0 6$ </td></tr><tr><td>C0.3</td><td>SAM-Med3D</td><td></td><td> $0 . 4 8 8 \pm 0 . 0$ </td><td> $0 . 6 5 0 \pm 0 . 0 0 1$ </td></tr><tr><td></td><td> $\mathrm { M E O W } \left( \mathcal { F } _ { 1 } \& \mathcal { M } \right)$ </td><td>0</td><td> $0 . 5 0 7 \pm 0 . 0 0 1$ </td><td> $0 . 6 7 2 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="6">C0.4</td><td rowspan="6"> $\mathrm { M E O W } \ ( \mathcal { F } _ { 2 } \& M )$ </td><td> $ 3 2$ </td><td> $0 . 5 4 2 \pm 0 . 0 1 0$ </td><td> $0 . 6 5 3 \pm 0 . 0 0 9$ </td></tr><tr><td>→ 64</td><td> $0 . 6 0 0 \pm 0 . 0 0 7$ </td><td>0.681 ± 0.005</td></tr><tr><td> $ 1 2 8$ </td><td> $0 . 6 2 5 \pm 0 . 0 0 6$ </td><td>0.691 ± 0.004</td></tr><tr><td>0</td><td>0.336 ± 0.012</td><td>0.576 ± 0.0</td></tr><tr><td>→ 32</td><td>0.451 ± 0.011</td><td>0.528 ± 0.004</td></tr><tr><td>→ 64</td><td>0.500 ± 0.007</td><td>0.563 ± 0.007</td></tr><tr><td rowspan="5">C0.5</td><td> $\mathbf { M E O W } \ ( \mathcal { F } _ { 3 } \& M )$  0</td><td> $ 1 2 8$ </td><td>0.515 ± 0.005</td><td>0.574 ± 0.001</td></tr><tr><td></td><td></td><td> $0 . 4 6 8 \pm 0 . 0 0 3$ </td><td> $0 . 6 1 3 \pm 0 . 0 0 2$ </td></tr><tr><td>→ 32 → 64</td><td></td><td> $0 . 3 6 1 \pm 0 . 0 0 4$ </td><td> $0 . 5 5 9 \pm 0 . 0 0 3$ </td></tr><tr><td>→ 128</td><td></td><td> $0 . 4 1 1 \pm 0 . 0 0 2$   $0 . 4 4 0 \pm 0 . 0 0 3$ </td><td> $0 . 5 8 8 \pm 0 . 0 0 1$ </td></tr><tr><td></td><td></td><td></td><td> $0 . 6 0 8 \pm 0 . 0 0 7$ </td></tr><tr><td>C1.1</td><td>nnInteractive VISTA3D</td><td></td><td> $\overline { { 0 . 5 6 6 \pm 0 . 0 } }$   $0 . 2 0 3 \pm 0 . 0$ </td><td> $\mathbf { \overline { { 0 . 8 5 1 \pm 0 . 0 } } }$ </td></tr><tr><td>C1.2</td><td>SAM-Med3D</td><td></td><td> $0 . 5 2 2 \pm 0 . 0$ </td><td> $0 . 4 9 6 \pm 0 . 0$ </td></tr><tr><td>C1.3</td><td>MEOW (F1&amp;M)</td><td></td><td></td><td>0.682 ± 0.0</td></tr><tr><td></td><td></td><td>0</td><td> $0 . 5 3 6 \pm 0 . 0 0 8$ </td><td> $0 . 7 1 7 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="8">C1.4</td><td></td><td>→ 8</td><td> $0 . 6 1 6 \pm 0 . 0 1 4$ </td><td> $0 . 7 2 8 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>→ 16</td><td> $0 . 6 4 3 \pm 0 . 0 1 2$ </td><td>0.740 ± 0.009</td></tr><tr><td>→ 32</td><td></td><td> $0 . 6 4 9 \pm 0 . 0 1 3$ </td><td>0.744 ± 0.002</td></tr><tr><td> $\mathrm { M E O W } \ ( \mathcal { F } _ { 2 } \& M )$  0</td><td></td><td> $0 . 5 2 1 \pm 0 . 0 1 4$ </td><td>0.683 ± 0.004</td></tr><tr><td>→ 8</td><td></td><td> $0 . 6 3 9 \pm 0 . 0 1 6$ </td><td>0.703 ± 0.010</td></tr><tr><td>→ 16</td><td></td><td> $0 . 6 5 0 \pm 0 . 0 0 9$ </td><td> $0 . 6 9 5 \pm 0 . 0 1 6$ </td></tr><tr><td>→ 32</td><td></td><td> $\mathbf { 0 . 6 7 5 \pm 0 . 0 1 0 }$ </td><td> $0 . 7 1 2 \pm 0 . 0 1 2$ </td></tr><tr><td> $\mathbf { M E O W } \ ( \mathcal { F } _ { 3 } \& M )$  0</td><td></td><td> $0 . 4 8 9 \pm 0 . 0 0 1$ </td><td> $0 . 6 7 3 \pm 0 . 0 0 5$ </td></tr><tr><td rowspan="4">C1.5</td><td></td><td>→ 8</td><td> $0 . 4 8 8 \pm 0 . 1 4 5$ </td><td> $0 . 6 2 0 \pm 0 . 0 9 7$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>→ 16</td><td></td><td> $0 . 5 1 5 \pm 0 . 1 2 8$ </td><td> $0 . 6 3 2 \pm 0 . 0 9 2$ </td></tr><tr><td>→ 32</td><td></td><td> $0 . 5 1 2 \pm 0 . 0 9 3$ </td><td> $0 . 6 2 8 \pm 0 . 0 7 6$ </td></tr></table>

Open-Domain Learning: OW recognition is not limited to learning new classes in a fixed domain, but also extends to learning within entirely new domains. We therefore demonstrate cross-modality generalization of MEOW and its OW-learning capabilities using brain MRI data for tumor segmentation (using MSD BRATS [109]). We summarize our results in Table X. nnInteractive and SAM-Med3D are excluded for this comparison since they were trained on brain MRI data.

TABLE IX  
OW-IS RESULTS FOR ABLATION STUDIES ON MOUSE $\mu \mathrm { C T }$ DATA (TOP) AND INSECTANATOMY DATA (BOTTOM). WE ANALYZE USING GROUND TRUTH LABELS, USING UNTRAINED FEATURE EXTRACTORS (MARKED WITH r), AND USING NO CLASS METADATA.
<table><tr><td>ID</td><td>Ablation</td><td> $\mathbf { D s C } _ { \mathbf { c r } } ^ { \mathbf { 1 } } \uparrow$ </td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { \mathrm { c r } } ^ { 8 } \uparrow } }$ </td></tr><tr><td>C0.3</td><td></td><td> $\mathbf { \overline { { 0 . 6 2 5 \pm 0 . 0 0 6 } } }$ </td><td> $\mathbf { \overline { { 0 . 6 9 1 \pm 0 . 0 0 4 } } }$ </td></tr><tr><td>C0.4</td><td></td><td> $0 . 5 1 5 \pm 0 . 0 0 5$ </td><td> $0 . 5 7 4 \pm 0 . 0 0 1$ </td></tr><tr><td>C0.5</td><td></td><td> $0 . 4 4 0 \pm 0 . 0 0 3$ </td><td> $0 . 6 0 8 \pm 0 . 0 0 7$ </td></tr><tr><td>C0.6</td><td> $\overline { { z _ { p }  z _ { g t } \mathrm { ~ i n ~ } \mathcal { M } ( \mathcal { F } _ { 1 } ) } }$ </td><td> $\mathbf { \overline { { 0 . 6 8 2 \pm 0 . 0 2 0 } } }$ </td><td> $\mathbf { \overline { { 0 . 7 5 0 \pm 0 . 0 0 9 } } }$ </td></tr><tr><td>C0.7</td><td> $z _ { p }  z _ { g t }$  in  $\mathcal { M } \left( \mathcal { F } _ { 2 } \right)$ </td><td> $0 . 6 7 1 \pm 0 . 0 0 9$ </td><td> $0 . 7 1 8 \pm 0 . 0 0 5$ </td></tr><tr><td>C0.8</td><td> $z _ { p }  z _ { g t } ^ { - } \mathrm { i n } \mathcal { M } ( \mathcal { F } _ { 3 } )$ </td><td> $0 . 5 5 0 \pm 0 . 0 0 3$ </td><td> $0 . 7 0 2 \pm 0 . 0 0 1$ </td></tr><tr><td>C0.9</td><td> $\overline { { \mathcal { F } _ { 1 } ^ { r } \mathrm { ( u n t r a i n e d ) } } }$ </td><td> $\mathbf { 0 . 4 7 3 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { \overline { { 0 . 5 7 9 \pm 0 . 0 0 1 } } }$ </td></tr><tr><td>C0.10</td><td> $\mathcal { F } _ { 2 } ^ { \hat { r } } \ \mathrm { ( u n t r a i n e d ) }$ </td><td> $0 . 3 9 2 \pm 0 . 0 0 1$ </td><td> $0 . 5 3 0 \pm 0 . 0 0 6$ </td></tr><tr><td>C0.11</td><td> $\mathcal { F } _ { 3 } ^ { r } ~ \mathrm { ( u n t r a i n e d ) }$ </td><td> $0 . 4 5 9 \pm 0 . 0 0 8$ </td><td> $0 . 5 3 0 \pm 0 . 0 0 6$ </td></tr><tr><td>C0.12</td><td> $\overline { { \mathrm { U K C s } \to \mathrm { U U C s } \left( \mathcal { F } _ { 1 } \right) } }$ </td><td> $\mathbf { 0 . 5 9 5 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { \overline { { 0 . 6 7 6 \pm 0 . 0 } } }$ </td></tr><tr><td>C0.13</td><td> $\mathrm { U K C s }  \mathrm { U U C s } \ ( \mathcal { F } _ { 2 } )$ </td><td> $0 . 4 7 1 \pm 0 . 0 0 7$ </td><td> $0 . 5 6 7 \pm 0 . 0 0 4$ </td></tr><tr><td>C0.14</td><td> $\mathrm { U K C s }  \mathrm { U U C s } \ ( \mathcal { F } _ { 3 } )$ </td><td> $0 . 4 6 4 \pm 0 . 0 1 1$ </td><td> $0 . 5 8 7 \pm 0 . 0 0 3$ </td></tr><tr><td>C1.3</td><td></td><td> $\overline { { 0 . 6 4 9 \pm 0 . 0 1 3 } }$ </td><td> $\mathbf { \overline { { 0 . 7 4 4 \pm 0 . 0 0 2 } } }$  1</td></tr><tr><td>C1.4</td><td></td><td> $\mathbf { 0 . 6 7 5 \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>C1.5</td><td>-</td><td> $0 . 5 1 2 \pm 0 . 0 9 3$ </td><td> $0 . 7 1 2 \pm 0 . 0 1 2$ </td></tr><tr><td>C1.6</td><td>=</td><td> $\overline { { 0 . 6 5 1 \pm 0 . 0 1 0 } }$ </td><td> $0 . 6 2 8 \pm 0 . 0 7 6$ </td></tr><tr><td>C1.7</td><td> $\overline { { z _ { p }  z _ { g t } \mathrm { ~ i n ~ } \mathcal { M } ( \mathcal { F } _ { 1 } ) } }$ </td><td></td><td> $\mathbf { 0 . 7 4 9 \pm 0 . 0 0 7 }$ </td></tr><tr><td></td><td> $z _ { p }  z _ { g t }$  i  $\textrm { n } \mathcal { M } \left( \mathcal { F } _ { 2 } \right)$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 0 0 3 }$ </td><td> $0 . 7 2 9 \pm 0 . 0 0 2$ </td></tr><tr><td>C1.8</td><td> $z _ { p }  z _ { g t }$  in  $\mathcal { M } \left( \mathcal { F } _ { 3 } \right)$ </td><td> $0 . 5 6 8 \pm 0 . 0 1 0$ </td><td> $0 . 6 7 5 \pm 0 . 0 0 3$ </td></tr><tr><td>C1.9</td><td>(untrained)</td><td> $\mathbf { \overline { { 0 . 6 1 2 \pm 0 . 0 0 2 } } }$ </td><td> $\mathbf { \overline { { 0 . 7 2 1 \pm 0 . 0 0 8 } } }$ </td></tr><tr><td>C1.10</td><td>(untrained)</td><td> $0 . 4 4 8 \pm 0 . 0 3 5$ </td><td> $0 . 5 4 6 \pm 0 . 0 2 9$ </td></tr><tr><td>C1.11</td><td>FF F (untrained)</td><td> $0 . 5 5 2 \pm 0 . 0 1 1$ </td><td> $0 . 6 5 5 \pm 0 . 0 1 8$ </td></tr><tr><td>C1.12</td><td> $\overrightarrow { \mathrm { U K C s }  \mathrm { U U C s } ( \mathcal { F } _ { 1 } ) }$ </td><td> $\overline { { 0 . 6 4 2 \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 0 . 7 4 7 \pm 0 . 0 0 1 }$ </td></tr><tr><td>C1.13</td><td> $\mathrm { U K C s }  \mathrm { U U C s } \ ( \mathcal { F } _ { 2 } )$ </td><td> $\mathbf { 0 . 6 7 4 \pm 0 . 0 0 4 }$ </td><td> $0 . 7 2 8 \pm 0 . 0 0 4$ </td></tr><tr><td>C1.14</td><td> $\mathrm { U K C s }  \mathrm { U U C s } \ ( \mathcal { F } _ { 3 } )$ </td><td> $0 . 5 0 2 \pm 0 . 0 7 3$ </td><td> $0 . 6 5 3 \pm 0 . 0 3 9$ </td></tr></table>

TABLE X

OW-IS RESULTS WHEN LEARNING WITHIN UNKNOWN MRI DOMAIN FOR BRAIN TUMORS. ALL F WERE PRE-TRAINED ON HUMAN CT DATA.
<table><tr><td>ID</td><td>Method</td><td>m (#SoIs)</td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { c r } ^ { 1 } \mathrm { ~ \uparrow ~ } } }$ </td><td> $\overline { { \mathbf { D } \mathbf { S } \mathbf { C } _ { c r } ^ { 8 } } }$  ←</td></tr><tr><td>C2.0 C2.1</td><td>VISTA3D</td><td>-</td><td> $\overline { { 0 . 2 9 1 \pm 0 . 0 } }$ </td><td>0.653 ± 0.0</td></tr><tr><td rowspan="8"></td><td> $\mathrm { M E O W } \left( \mathcal { F } _ { 1 } \& \mathcal { M } \right)$ </td><td>0</td><td> $0 . 5 5 5 \pm 0 . 0 0 6$ </td><td>0.650 ± 0.004</td></tr><tr><td>→ 8</td><td></td><td> $0 . 6 3 5 \pm 0 . 0 1 3$ </td><td>0.715 ± 0.009</td></tr><tr><td>→ 16</td><td></td><td> $0 . 6 4 2 \pm 0 . 0 1 4$ </td><td>0.717 ± 0.007</td></tr><tr><td></td><td>→ 32</td><td> $0 . 6 4 0 \pm 0 . 0 0 1$ </td><td>0.718 ± 0.007</td></tr><tr><td>→ 64</td><td></td><td> $0 . 6 4 3 \pm 0 . 0 0 3$ </td><td>0.719 ± 0.006</td></tr><tr><td>→ 128</td><td></td><td> $0 . 6 4 8 \pm 0 . 0 1 0$ </td><td>0.721 ± 0.009</td></tr><tr><td> $( \mathcal { F } _ { 2 } \& \mathcal { M } )$  0</td><td></td><td> $0 . 3 9 9 \pm 0 . 0 0 5$ </td><td>0.551 ± 0.005</td></tr><tr><td>→ 8</td><td></td><td> $0 . 5 9 3 \pm 0 . 0 1 5$ </td><td>0.616 ± 0.020</td></tr><tr><td rowspan="8">C2.3 MEOW</td><td></td><td>→ 16</td><td> $0 . 5 9 1 \pm 0 . 0 1 2$ </td><td>0.612 ± 0.022</td></tr><tr><td>→ 32</td><td></td><td> $0 . 5 9 1 \pm 0 . 0 0 2$ </td><td>0.606 ± 0.016</td></tr><tr><td>→ 64</td><td></td><td> $0 . 5 9 2 \pm 0 . 0 0 9$ </td><td>0.613 ± 0.013</td></tr><tr><td> $( \mathcal { F } _ { 3 } \& \mathcal { M } )$ </td><td>→ 128</td><td> $0 . 5 9 1 \pm 0 . 0 1 1$ </td><td>0.613 ± 0.013</td></tr><tr><td>0</td><td></td><td> $0 . 6 7 4 \pm 0 . 0 0 1$ </td><td>0.650 ± 0.003</td></tr><tr><td>→ 8</td><td></td><td> $0 . 7 2 5 \pm 0 . 0 2 2$ </td><td> $0 . 7 8 3 \pm 0 . 0 0 3$ </td></tr><tr><td>→ 16</td><td></td><td> $0 . 7 2 6 \pm 0 . 0 1 9$ </td><td> $0 . 7 8 1 \pm 0 . 0 0 1$ </td></tr><tr><td>→ 32</td><td></td><td> $0 . 7 1 5 \pm 0 . 0 1 1$ </td><td> $0 . 7 7 9 \pm 0 . 0 0 2$ </td></tr><tr><td colspan="2"></td><td>→ 64 → 128</td><td> $0 . 7 2 3 \pm 0 . 0 0 1$   $\mathbf { 0 . 7 2 9 \pm 0 . 0 0 7 }$ </td><td> $0 . 7 8 1 \pm 0 . 0 0 4$   $\mathbf { 0 . 7 8 7 \pm 0 . 0 0 1 }$ </td></tr></table>

Our approach outperforms VISTA3D by a large margin when using $\mathcal { F } _ { 1 }$ (C2.1) or $\mathcal { F } _ { 3 }$ (C2.3), and when using $\mathcal { F } _ { 2 }$ with $n < 4$ clicks (C2.2). Although $\mathcal { F } _ { 1 }$ has not been trained on either CT or MRI brain data, using it in combination with MEOW still outperforms VISTA3D, which has been trained on brain CT data. Highest performance is obtained when combining our method with VISTA3D’s sub-module $( \mathcal { F } _ { 3 } )$ while at the same time requiring less learnable parameters. For all feature extractors, we achieve much higher performance than the original VISTA3D baseline already for $n = 1$ clicks and $m = 0$ memory samples. In agreement with our results for in-domain OW-learning, cross-domain OW-learning via memory consolidation further increases performance. Segmentation quality remains relatively stable with larger memory sizes, indicating a saturation effect (C2.3). In combination with qualitative demonstrations in Fig. 7, these results suggest that MEOW generalizes across domains for OS-IS and OW-IS.

To summarize, we demonstrate that our approach flexibly adapts to various visual contexts without finetuning of learnable parameters. MEOW succeeds in closed-set, open-set, and open-world settings while at the same time being modular, data-efficient, fast-learning, and feature-robust.

## V. DISCUSSION

Our framework addresses the core challenges of OW-learning as formulated in Section III-E. By leveraging memory consolidation within M, the system facilitates seamless transition from zero-shot IS (via base prediction y<sub>b</sub>) to few-shot and many-shot scenarios (via memory prediction $y _ { \mathcal { M } } )$ as knowledge is accumulated during deployment. Notably, our approach inherently supports incremental and continual learning paradigms, including out-of-distribution detection and data-efficient adaptation while keeping backbone parameters θ frozen. This enables scalable learning and unlearning without catastrophic forgetting [14], [136]–[138] by modulating memory states rather than performing computationally expensive gradient-based learning.

While performance in fully automated continual semanticor panoptic segmentation frameworks tends to degrade over time due to error propagation [139], integrating a human user U in the loop provides a natural mechanism for pseudo-label quality verification. High-quality results may be learned while sub-optimal results may be discarded, ensuring purity of stored data for lifelong OW-learning. Through memory contents and different options for user guidance, our method further accounts for user-specific annotation styles, thereby enabling flexible personalization capabilities.

Moreover, the MEOW architecture provides task-level flexibility. By reconfiguring the functional modules ${ \mathcal { F } } , { \mathcal { G } } ,$ , and $\mathcal { M } ,$ the system supports a broad range of downstream tasks, including instance classification, novelty detection, class labeling with arbitrary number of classes, semantic segmentation, IS, as well as HMP. Notably, this multi-task capability is achieved while maintaining a constant number of learnable parameters regardless of task complexity.

Recently proposed tasks for continual medical image segmentation include domain-, class-, and organ-level scenarios [140]. Notably, our method supports the domain scenario by consolidating images from different clinical centers while retaining the flexibility to adapt to specific domains by employing domain-specific memory modules. The class scenario is supported simply by swapping stored memory labels. The organ scenario is a combination of the other two, which is also supported due to our flexible architecture.

Deploying ML models in sensitive domains like medicine necessitates additional considerations regarding robustness, privacy, interpretability, and regulatory standards. Our parameter-free architecture offers several advantages in this regard:

Regulatory compliance: Our memory-focused architecture is compliant with mandates such as Europe’s GDPR right to be forgotten [141, Article 17], which allows individuals to request that their personal data be removed under certain conditions. With our method, removal becomes as simple as deleting an entry from M. This contrasts other monolithic methods, which often require complicated and inefficient unlearning or re-training schemes to remove the influence of specific data from neural network weights.

Modularity and Maintenance: The decoupled nature of ${ \mathcal { F } } ,$ G, and M allows flexible updates. Feature extractors can be swapped with better backbones or to address licensing changes while retaining the accumulated knowledge base. Memory modules can be swapped to specialize on specific classes or to meet user-specific preferences.

Flexible Usage of Memory: M may hold a fixed dataset with ground truth labels $z _ { g t }$ of known classes to improve IS objectives. In the presence of $\mathcal { M } ,$ any interaction with U may be completely skipped to end up with a fully-automatic semantic- or instance segmentation system.

Stability: Our system seamlessly integrates any number of new classes. As M treats known classes independently for making predictions, the addition of new classes does not impact the segmentation performance on existing classes for KKCs and UKCs. Under the assumption of unconstrained memory, this translates to stable knowledge retention without catastrophic forgetting.

Transparency and Usage: By relying on parameter-free transduction with G and M for making predictions, our method is transparent and intuitive to use. Model predictions are governed by a set of interactive hyperparameters, each providing explicit control over a distinct conceptual aspect. This enables nuanced and predictable modifications to resulting segmentations.

## VI. CONCLUSION

We propose MEOW, a modular open-world interactive segmentation (OW-IS) system combining a frozen feature extractor F, a transduction module G, and a memory module M to learn from its own predictions of unknown structures. Our approach enables flexible learning and unlearning paradigms through memory consolidation. The system takes weak user labels and interactive hyperparameters as guidance to segment any structure of interest. Predictions are made by leveraging hierarchical correspondence matching and message passing with label retrieval and transductive reasoning.

Takeaways: We demonstrate that non-interactive, non-OW backbones can be made interactive and OW-capable without any additional training of neural network parameters. By filling a memory module with self-labeled samples at test time, our system improves both in terms of segmentation performance and required user guidance. Notably, we show that our method is highly data-efficient and even works with completely untrained feature extractors. Due to our memory-based architecture, learning and unlearning is done in a matter of milliseconds, establishing a new standard for fast and stable learning at scale.

Limitations: Storing feature pyramids in M and applying k-NN search within memory contents generally requires more computational resources (higher hardware memory and slightly higher inference times) compared to fully end-to-end neural network based approaches. Moreover, because MEOW requires feature extractors with specific hourglass-shaped architectures, it is currently unable to leverage transformer-based architectures [142] for open-world learning.

Future Work: Future directions include studying the effectiveness of self-supervised and unsupervised feature extractors. MEOW can also be used to analyze sequential data such as longitudinal CT studies or videos. To enhance personalization, memory modules may be used to create user-specific profiles based on individual labeling preferences. Finally, to address privacy concerns inherent in handling sensitive data, privacy-preserving computation may be investigated.

Ultimately, this work provides a scalable foundation for deploying lifelong learning interactive segmentation systems capable of continuous self-improvement in dynamic real-world settings.

## REFERENCES

[1] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.

[2] L. Wang, X. Zhang, H. Su, and J. Zhu, “A comprehensive survey of continual learning: Theory, method and application,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5362–5383, 2024.

[3] C. Liu, K. Tang, Y. Qin, and Q. Lei, “Bridging distribution shift and ai safety: Conceptual and methodological synergies,” 2026.

[4] B. Zhao, J. Wang, W. Ma, A. Jesslen, S. Yang, S. Ya, O. Zendel, C. Theobalt, A. Yuille, and A. Kortylewski, “Ood-cv-v2 : An extended benchmark for robustness to out-of-distribution shifts of individual nuisances in natural images,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, p. 11104, Dec. 2024.

[5] M. Gutbrod, D. Rauber, D. W. Nunes, and C. Palm, “Openmibood: Open medical imaging benchmarks for out-of-distribution detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), June 2025, pp. 25 874–25 886.

[6] Y. Shoeb, A. Nowzad, and H. Gottschalk, “ Out-of-Distribution Segmentation in Autonomous Driving: Problems and State of the Art ,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). Los Alamitos, CA, USA: IEEE Computer Society, 06 2025, pp. 4310–4320.

[7] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 7, pp. 3366–3385, 2022.

[8] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” 2022.

[9] D. Karimi and A. Gholipour, “Improving calibration and out-of-distribution detection in deep models for medical image segmentation,” IEEE Transactions on Artificial Intelligence, vol. 4, no. 2, pp. 383–397, 2023.

[10] A. Bendale and T. Boult, “Towards open world recognition,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1893–1902.

[11] Y. Boykov and M.-P. Jolly, “Interactive graph cuts for optimal boundary & region segmentation of objects in n-d images,” in Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, vol. 1, 2001, pp. 105–112 vol.1.

[12] N. Xu, B. Price, S. Cohen, J. Yang, and T. S. Huang, “Deep interactive object selection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 373–381.

[13] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” Nature, vol. 323, no. 6088, pp. 533–536, 10 1986.

[14] P. M. Winter, M. Wimmer, A. Berg, D. Major, D. Lenis, T. Neubauer, G. R. De Paolis, J. Novotny, S. Ulonska, and K. Buhler, “PARMESAN:¨ Parameter-free memory search and transduction for dense prediction tasks,” in Pattern Recognition (DAGM GCPR 2024), D. Cremers, Z. Lahner, M. Moeller, M. Nießner, B. Ommer, and R. Triebel, Eds.¨ Cham: Springer Nature Switzerland, 2025, pp. 3–20.

[15] W. Li, A. Yuille, and Z. Zhou, “How well do supervised models transfer to 3d image segmentation?” in The Twelfth International Conference on Learning Representations, 2024.

[16] J. Wasserthal et al., “Totalsegmentator: Robust segmentation of 104 anatomic structures in ct images,” Radiology: Artificial Intelligence, vol. 5, no. 5, p. e230024, 2023.

[17] X. Deng, H. Wu, R. Zeng, and J. Qin, “Memsam: Taming segment anything model for echocardiography video segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 9622–9631.

[18] Z. Marinov, P. F. Jager, J. Egger, J. Kleesiek, and R. Stiefelhagen,¨ “Deep interactive segmentation of medical images: A systematic review and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, p. 10998–11018, Dec. 2024.

[19] M. Amrehn, S. Gaube, M. Unberath, F. Schebesch, T. Horz, M. Strumia, S. Steidl, M. Kowarschik, and A. Maier, “Ui-net: Interactive artificial neural networks for iterative image segmentation based on a user model,” arXiv preprint arXiv:1709.03450, 2017.

[20] F. Du, J. Yuan, Z. Wang, and F. Wang, “Efficient mask correction for click-based interactive image segmentation,” in CVPR, 2023.

[21] T. Zhao et al., “A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities,” Nature Methods, vol. 22, no. 1, pp. 166–176, jan 2025.

[22] Q. Liu, J. Cho, M. Bansal, and M. Niethammer, “Rethinking interactive image segmentation with low latency high quality and diverse prompts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3773–3782.

[23] H. E. Wong, M. Rakic, J. Guttag, and A. V. Dalca, “Scribbleprompt: Fast and flexible interactive segmentation for any biomedical image,” European Conference on Computer Vision (ECCV), 2024.

[24] D. Lin, J. Dai, J. Jia, K. He, and J. Sun, “ ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation ,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2016, pp. 3159–3167.

[25] D. Engel, L. Sick, and T. Ropinski, “Leveraging self-supervised vision transformers for segmentation-based transfer function design,” IEEE Transactions on Visualization and Computer Graphics, vol. 31, no. 8, pp. 4357–4368, 2025.

[26] A. Diaz-Pinto et al., “Monai label: A framework for ai-assisted interactive labeling of 3d medical images,” Medical Image Analysis, 2024.

[27] J. Cheng et al., “Interactive medical image segmentation: A benchmark dataset and baseline,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, Canada, June 2025, p. to appear.

[28] J. Cheng, J. Ye, Z. Deng, J. Chen et al., “Sam-med2d,” 2023.

[29] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, no. 1, p. 654, 2024.

[30] F. Isensee et al., “nninteractive: Redefining 3d promptable segmentation,” 2025.

[31] Y. He et al., “Vista3d: A unified segmentation foundation model for 3d medical imaging,” 2024.

[32] Y. Du, F. Bai, T. Huang, and B. Zhao, “Segvol: Universal and interactive volumetric medical image segmentation,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 110 746–110 783.

[33] Y. Gao, D. Liu, Z. Li, Y. Li, D. Chen, M. Zhou, and D. N. Metaxas, “Show and segment: Universal medical image segmentation via in-context learning,” 2025.

[34] M. Rokuss, Y. Kirchhoff, S. Akbal, B. Kovacs, S. Roy, C. Ulrich, T. Wald, L. T. Rotkopf, H.-P. Schlemmer, and K. Maier-Hein, “Lesionlocator: Zero-shot universal tumor segmentation and tracking in 3d whole-body imaging,” 2025.

[35] H. Wang et al., “Sam-med3d: A vision foundation model for general-purpose segmentation on volumetric medical images,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 10, pp. 17 599–17 612, 2025.

[36] J. Ma, Z. Yang, S. Kim, B. Chen, M. Baharoon, A. Fallahpour, R. Asakereh, H. Lyu, and B. Wang, “Medsam2: Segment anything in 3d medical images and videos,” arXiv preprint arXiv:2504.03600, 2025.

[37] M. Orbes-Arteaga, O. Lucena, S. Ourselin, and M. J. Cardoso, “Mais: Memory-attention for interactive segmentation,” 2025.

[38] R. Bommasani et al., “On the opportunities and risks of foundation models,” CoRR, vol. abs/2108.07258, 2021.

[39] H. Jegou, M. Douze, and C. Schmid, “Product quantization for nearest´ neighbor search,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 1, pp. 117–128, 2011.

[40] J. Johnson, M. Douze, and H. Jegou, “Billion-scale similarity search´ with gpus,” IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021.

[41] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazare,´ M. Lomeli, L. Hosseini, and H. Jegou, “The faiss library,” 2024.´

[42] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, pp. 91–110, 2004.

[43] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detector-free local feature matching with transformers,” CVPR, 2021.

[44] J. Edstedt, Q. Sun, G. Bokman, M. Wadenb ¨ ack, and M. Felsberg,¨ “RoMa: Robust Dense Feature Matching,” in IEEE Conference on Computer Vision and Pattern Recognition, 2024.

[45] O. Mariotti, O. Mac Aodha, and H. Bilen, “Improving semantic correspondence with viewpoint-guided spherical maps,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[46] V. N. Vapnik, Statistical Learning Theory. Wiley, 1998.

[47] D. Zhou, O. Bousquet, T. N. Lal, J. Weston, and B. Scholkopf,¨ “Learning with local and global consistency,” in Proceedings of the 17th International Conference on Neural Information Processing Systems, ser. NIPS’03. Cambridge, MA, USA: MIT Press, 2003, p. 321–328.

[48] J. Pearl, “Reverend bayes on inference engines: A distributed hierarchical approach,” in Proceedings of the Second AAAI Conference on Artificial Intelligence (AAAI-82). AAAI Press, 1982, pp. 133–136.

[49] B. Appleton and H. Talbot, “Globally minimal surfaces by continuous maximal flows,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 28, pp. 106–118, 2006.

[50] B. J. Frey and D. Dueck, “Clustering by passing messages between data points,” Science, vol. 315, no. 5814, pp. 972–976, 2007.

[51] I. E. Givoni, C. Chung, and B. J. Frey, “Hierarchical affinity propagation,” in Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence, ser. UAI’11. Arlington, Virginia, USA: AUAI Press, 2011, p. 238–246.

[52] C. Geng, S.-J. Huang, and S. Chen, “Recent advances in open set recognition: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 10, pp. 3614–3631, 2021.

[53] I. Nunes, C. Laranjeira, H. Oliveira, and J. A. dos Santos, “A systematic review on open-set segmentation,” Computers & Graphics, vol. 115, pp. 296–308, 2023.

[54] J. Hong, W. Li, J. Han, J. Zheng, P. Fang, M. Harandi, and L. Petersson, “GOSS: towards generalized open-set semantic segmentation,” The Visual Computer, vol. 40, no. 4, pp. 2391–2404, Apr 2024.

[55] Y. Liu, Y. Yin, C. Jing, M. Zhu, H. Chen, Y. Xi, B. Feng, H. Wang, S. Li, and C. Shen, “Unified open-world segmentation with multi-modal prompts,” in ICCV, 2025.

[56] W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult, “Toward open set recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 7, pp. 1757–1772, 2013.

[57] S. Cruz, K. Doctor, C. Funk, and W. Scheirer, “Open issues in open world learning,” AI Magazine, vol. 46, no. 2, p. e70001, 2025.

[58] G. Kim, C. Xiao, T. Konishi, Z. Ke, and B. Liu, “Open-world continual learning: Unifying novelty detection and continual learning,” Artificial Intelligence, vol. 338, p. 104237, 2025.

[59] C. Gonzalez, A. Ranem, D. Pinto Dos Santos, A. Othman, and´ A. Mukhopadhyay, “Lifelong nnu-net: a framework for standardized medical continual learning,” Scientific Reports, vol. 13, no. 1, p. 9381, 06 2023.

[60] Z. Xiao and C. G. M. Snoek, “Beyond model adaptation at test time: A survey,” 2024.

[61] R. Schon, J. Lorenz, K. Ludwig, and R. Lienhart, “Adapting the¨ segment anything model during usage in novel situations,” 2024.

[62] Y. Li, X. Yang, H. Wang, X. Wang, and T. Li, “Learning to prompt knowledge transfer for open-world continual learning,” in Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications

of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, ser. AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024.

[63] Y. Li, G. Lai, X. Yang, Y. Li, M. Bonsangue, and T. Li, “Exploring open-world continual learning with knowns-unknowns knowledge transfer,” 2025.

[64] K. J. Joseph, S. Khan, F. S. Khan, and V. N. Balasubramanian, “Towards open world object detection,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 5826–5836.

[65] B. Dong, Z. Huang, G. Yang, L. Zhang, and W. Zuo, “Mr-gdino: Efficient open-world continual object detection,” 2024.

[66] M. Sodano, F. Magistri, L. Nunes, J. Behley, and C. Stachniss, “Open-World Semantic Segmentation Including Class Similarity,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[67] D. Li, T. Wu, B. Lin, Z. Chen, Y. Zhang, Y. Li, M.-M. Cheng, and X. Li, “WOW-seg: A word-free open world segmentation model,” in The Fourteenth International Conference on Learning Representations, 2026.

[68] T. Kalluri, W. Wang, H. Wang, M. Chandraker, L. Torresani, and D. Tran, “Open-world instance segmentation: Top-down learning with bottom-up supervision,” arxiv, 2023.

[69] M. Sodano, F. Magistri, J. Behley, and C. Stachniss, “Open-world panoptic segmentation,” 2024.

[70] S. Zhang et al., “A generalist foundation model and database for open-world medical image segmentation,” Nature Biomedical Engineering, sep 2025.

[71] T. Huang, T. Zhou, W. Xie, S. Wang, Q. Dou, and Y. Zhang, “On-the-fly improving segment anything for medical image segmentation using auxiliary online learning,” IEEE Transactions on Medical Imaging, pp. 1–1, 2025.

[72] G. Wang et al., “Interactive medical image segmentation using deep learning with image-specific fine tuning,” IEEE Transactions on Medical Imaging, vol. 37, no. 7, p. 1562–1573, Jul. 2018.

[73] K. Chen, X. Luo, T. Qin, J. Liu, H. Liu, V. H. F. Lee, H. Yan, and H. Li, “Test-time adaptation for foundation medical segmentation model without parametric updates,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 20 075–20 084.

[74] Z. Zhou, L.-Z. Guo, L.-H. Jia, D. Zhang, and Y.-F. Li, “ODS: Test-time adaptation in the presence of open-world data shift,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 42 574–42 588.

[75] T. H. Rafi, A. Agarwal, H. L. Patel, and D.-K. Chae, “ Learning from Unknown for Open-Set Test-Time Adaptation ,” in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). Los Alamitos, CA, USA: IEEE Computer Society, Mar. 2026, pp. 2993–3004.

[76] H. Dong, E. Chatzi, and O. Fink, “Towards robust multimodal open-set test-time adaptation via adaptive entropy-aware optimization,” in The Thirteenth International Conference on Learning Representations, 2025.

[77] K. Phuntsho, Abdullah, K. Lee, I. Lee, and E. Ahn, “Adaptation of foundation models for medical image analysis: Strategies, challenges, and future directions,” 2025.

[78] J. Weston, S. Chopra, and A. Bordes, “Memory networks,” 2015, publisher Copyright: © 2015 International Conference on Learning Representations, ICLR. All rights reserved.; 3rd International Conference on Learning Representations, ICLR 2015 ; Conference date: 07-05-2015 Through 09-05-2015.

[79] A. Graves et al., “Hybrid computing using a neural network with dynamic external memory,” Nature, vol. 538, no. 7626, pp. 471–476, Oct. 2016.

[80] D. Krotov and J. J. Hopfield, “Dense associative memory for pattern recognition,” in Proceedings of the 30th International Conference on Neural Information Processing Systems, ser. NIPS’16. Red Hook, NY, USA: Curran Associates Inc., 2016, p. 1180–1188.

[81] H. Ramsauer et al., “Hopfield networks is all you need,” in 9th International Conference on Learning Representations (ICLR), 2021.

[82] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 9459–9474.

[83] X. He, Y. Hu, Z. Zhou, M. Jarraya, and F. Liu, “Few-shot adaptation of training-free foundation model for 3d medical image segmentation,” 2025.

[84] A. Rodriguez-Munoz, M. Baradad, P. Isola, and A. Torralba, “Separating knowledge and perception with procedural data,” in Forty-second International Conference on Machine Learning, 2025.

[85] E. Horvitz, “Principles of mixed-initiative user interfaces,” in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, ser. CHI ’99. New York, NY, USA: Association for Computing Machinery, 1999, p. 159–166.

[86] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.

[87] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6230–6239.

[88] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, feb 2021.

[89] P. Indyk and R. Motwani, “Approximate nearest neighbors: towards removing the curse of dimensionality,” in Proceedings of the Thirtieth Annual ACM Symposium on Theory of Computing, ser. STOC ’98. New York, NY, USA: Association for Computing Machinery, 1998, p. 604–613.

[90] O. Mariotti, O. Mac Aodha, and H. Bilen, “Improving semantic correspondence with viewpoint-guided spherical maps,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[91] Boykov and Kolmogorov, “Computing geodesics and minimal surfaces via graph cuts,” in Proceedings Ninth IEEE International Conference on Computer Vision, 2003, pp. 26–33 vol.1.

[92] Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel, “Gated graph sequence neural networks,” in International Conference on Learning Representations (ICLR), 2016.

[93] T. K. Rusch, M. M. Bronstein, and S. Mishra, “A survey on oversmoothing in graph neural networks,” 2023.

[94] S. Greydanus, M. Dzamba, and J. Yosinski, “Hamiltonian neural networks,” in Advances in Neural Information Processing Systems, vol. 32. NeurIPS, 2019, pp. 15 379–15 389.

[95] Z. Chen, J. Zhang, M. Arjovsky, and L. Bottou, “Symplectic recurrent neural networks,” in International Conference on Learning Representations, 2020.

[96] P.-J. Hoedt, F. Kratzert, D. Klotz, C. Halmich, M. Holzleitner, G. S. Nearing, S. Hochreiter, and G. Klambauer, “Mc-lstm: Mass-conserving lstm,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 4275–4286.

[97] E. Rodriguez, M. Raissi, and P. Baque, “Physics-informed neural networks with hard constraints for inverse design,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139. PMLR, 2021, pp. 8967–8977.

[98] M. Cranmer, S. Greydanus, D. W. Hogg, and K. Cranmer, “Lagrangian neural networks,” in International Conference on Learning Representations, 2021.

[99] K. Nakata, Y. Ng, D. Miyashita, A. Maki, Y.-C. Lin, and J. Deguchi, “Revisiting a kNN-based image classification system with high-capacity storage,” in ECCV, S. Avidan, G. Brostow, M. Cisse,´ G. M. Farinella, and T. Hassner, Eds., 2022, pp. 457–474.

[100] B. Scholkopf, A. Smola, and K.-R. M¨ uller, “Nonlinear component¨ analysis as a kernel eigenvalue problem,” Neural Computation, vol. 10, no. 5, pp. 1299–1319, 07 1998.

[101] K. Grauman and T. Darrell, “The pyramid match kernel: Discriminative classification with sets of image features,” in Proceedings of the Tenth IEEE International Conference on Computer Vision - Volume 2, ser. ICCV ’05. USA: IEEE Computer Society, 2005, p. 1458–1465.

[102] H. Bourlard and Y. Kamp, “Auto-association by multilayer perceptrons and singular value decomposition,” Biological Cybernetics, vol. 59, no. 4, pp. 291–294, sep 1988.

[103] K. G. Joreskog, “Some contributions to maximum likelihood factor¨ analysis,” Psychometrika, vol. 32, no. 4, pp. 443–482, dec 1967.

[104] B. S. Everitt, An Introduction to Latent Variable Models. London: Chapman and Hall, 1984, chapter 3: Principal Component Analysis.

[105] E. Bingham and H. Mannila, “Random projection in dimensionality reduction: applications to image and text data,” in Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge

Discovery and Data Mining, ser. KDD ’01. New York, NY, USA: Association for Computing Machinery, 2001, p. 245–250.

[106] A. Ashukha, A. Atanov, and D. P. Vetrov, “Mean embeddings with test-time data augmentation for ensembling of representations,” CoRR, vol. abs/2106.08038, 2021.

[107] M. Radovanovic, A. Nanopoulos, and M. Ivanovi ´ c, “Hubs in space:´ Popular nearest neighbors in high-dimensional data,” Journal of Machine Learning Research, vol. 11, no. 86, pp. 2487–2531, 2010.

[108] I. Rocco, M. Cimpoi, R. Arandjelovic, A. Torii, T. Pajdla, and´ J. Sivic, “Neighbourhood consensus networks,” in Advances in Neural Information Processing Systems, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018.

[109] M. Antonelli et al., “The medical segmentation decathlon,” Nature Communications, vol. 13, no. 1, p. 4128, 07 2022.

[110] N. Heller et al., “The kits21 challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase ct,” 2023.

[111] S. Rosenhain, Z. A. Magnuska, G. G. Yamoah, W. A. Rawashdeh, F. Kiessling, and F. Gremse, “A preclinical micro-computed tomography database including 3d whole body organ segmentations,” Scientific Data, vol. 5, no. 1, p. 180294, 12 2018.

[112] E. Toulkeridou, C. Gutierrez, D. Baum et al., “Automated segmentation of insect anatomy from micro-CT images using deep learning,” 2023, [Dataset].

[113] L. R. Dice, “Measures of the amount of ecologic association between species,” Ecology, vol. 26, no. 3, pp. 297–302, 1945.

[114] T. Sørensen, “A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on danish commons,” Biologiske Skrifter / Kongelige Danske Videnskabernes Selskab, vol. 5, no. 4, pp. 1–34, 1948.

[115] W.-D. Jang and C.-S. Kim, “Interactive image segmentation via backpropagating refinement scheme,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5292–5301.

[116] M. Forte, B. Price, S. Cohen, N. Xu, and F. Pitie, “Getting to 99´

[117] A. Moskalenko, V. Shakhuro, A. Vorontsova, A. Konushin, A. Antonov, A. Krapukhin, D. Shepelev, and K. Soshin, “Tetris: Towards exploring the robustness of interactive segmentation,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 5, p. 4287–4295, Mar. 2024.

[118] B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas, “Taking the human out of the loop: A review of bayesian optimization,” Proceedings of the IEEE, vol. 104, no. 1, pp. 148–175, 2016.

[119] A. Paszke et al., PyTorch: an imperative style, high-performance deep learning library. Red Hook, NY, USA: Curran Associates Inc., 2019.

[120] F. Perez-Garc ´ ´ıa, R. Sparks, and S. Ourselin, “TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning,” Computer Methods and Programs in Biomedicine, p. 106236, 2021.

[121] J. D. Hunter, “Matplotlib: A 2d graphics environment,” Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007.

[122] C. R. Harris et al., “Array programming with NumPy,” Nature, vol. 585, no. 7825, pp. 357–362, Sep. 2020.

[123] C. J. Markiewicz et al., “nipy/nibabel: 5.4.0,” Mar. 2026.

[124] W. Silversmith, “cc3d: Connected components on multilabel 3D & 2D images.” Sep. 2021.

[125] F. D. Giovanni, J. Rowbottom, B. P. Chamberlain, T. Markovich, and M. M. Bronstein, “Graph neural networks as gradient flows,” 2022.

[126] P. Klacansky, “Open scivis datasets,” December 2017, http://klacansky.com/open-scivis-datasets/.

[127] F. Y. Zhou et al., “Universal consensus 3d segmentation of cells from 2d segmented stacks,” Nature Methods, vol. 22, no. 11, pp. 2386–2399, Nov. 2025.

[128] P. Helber, B. Bischke, A. Dengel, and D. Borth, “Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification,” Jul. 2018.

[129] M. S. Charikar, “Similarity estimation techniques from rounding algorithms,” in Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, ser. STOC ’02. New York, NY, USA: Association for Computing Machinery, 2002, p. 380–388.

[130] A. Rahimi and B. Recht, “Random features for large-scale kernel machines,” in Advances in Neural Information Processing Systems, J. Platt, D. Koller, Y. Singer, and S. Roweis, Eds., vol. 20. Curran Associates, Inc., 2007.

[131] S. Attaiki and M. Ovsjanikov, “Ncp: Neural correspondence prior for effective unsupervised shape matching,” 2023.

[132] G. Tanaka, T. Yamane, J. B. Heroux, R. Nakane, N. Kanazawa,´ S. Takeda, H. Numata, D. Nakano, and A. Hirose, “Recent advances in physical reservoir computing: A review,” Neural Networks, vol. 115, pp. 100–123, 2019.

[133] T. A. Plate, Holographic Reduced Representation: Distributed Representation for Cognitive Structures. Center for the Study of Language and Information, 2003.

[134] P. Kanerva, “Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors,” Cognitive Computation, vol. 1, no. 2, pp. 139–159, 2009.

[135] J. J. Hopfield, “Neural networks and physical systems with emergent collective computational abilities.” Proceedings of the National Academy of Sciences, vol. 79, no. 8, pp. 2554–2558, 1982.

[136] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Advances in Research and Theory, ser. Psychology of Learning and Motivation, G. H. Bower, Ed., 1989, vol. 24, pp. 109–165.

[137] R. Ratcliff, “Connectionist models of recognition memory: constraints imposed by learning and forgetting functions.” Psychological review, vol. 97 2, pp. 285–308, 1990.

[138] R. M. French, “Catastrophic forgetting in connectionist networks,” Trends in Cognitive Sciences, vol. 3, no. 4, pp. 128–135, 1999.

[139] B. Kim, J. Yu, and S. J. Hwang, “Eclipse: Efficient continual learning in panoptic segmentation with visual prompt tuning,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 3346–3356.

[140] B. Wang, H. Zhou, Y. Gao, and X. Zhuang, “Beyond forgetting in continual medical image segmentation: A comprehensive benchmark study,” 2026.

[141] European Union, “Regulation (eu) 2016/679 of the european parliament and of the council of 27 april 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing directive 95/46/ec (general data protection regulation),” Official Journal of the European Union, L 119, 4 May 2016, pp. 1–88, 2016, article 17.

[142] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[143] G. Louppe, M. Kumar, and H. Nahrstaedt, “Scikit-Optimize: Sequential model-based optimization in Python,” 2016–2020, version 0.10.2.

[144] B. Matern,´ Spatial Variation. New York: Springer, 1960.

# Appendix MEOW: Medical Open-World Interactive Segmentation

## I. METHOD

We visualize our newly introduced memory search kernels (called cross-kernel and flexi-kernel) in Fig. 9, replacing the original box-kernel of PARMESAN. In contrast to the former, where a common kernel region is shared across all query nodes that are related to a specific parent, locations and orientations of our new kernels adapt depending on query node positions. These adaptations can be regarded as location-informed biases, assuming that positional offsets in queries correlate to corresponding offsets in memory contents. The new kernels reduce metrication artifacts while requiring less (for cross) or equal (for flexi) hardware memory w.r.t. the box-kernel for 3D data (see ablation study results A0 vs. A2, A8 vs. A10, and A16 vs. A18 in the main text). Further, the flexi-kernel improves search capacity by adapting to labeled regions in the image, effectively reducing overhead computations from unlabeled regions.

We provide pseudo-codes for calculating base predictions (Algorithm 1), fusing memory prediction (Algorithm 2), and hierarchical message passing (Algorithm 3).

## II. DATA

## A. Datasets

Following, we briefly describe all datasets that are used in this work:

MSD [109]: The Medical Segmentation Decathlon combines 10 different medical segmentation datasets. We use whole-organ CT annotations for liver, pancreas, and spleen as KKCs. When using TotalSegmentator for feature extraction, we use MSD as an independent evaluation set.

KiTS23 [110]: We use whole-organ annotations for left and right kidneys from the Kidney Tumor Segmentation challenge as KKCs. When using TotalSegmentator for feature extraction, we use KiTS23 as an independent evaluation set.

Mouse $\mu \mathbf { C T }$ [111]: We use organ annotations from 140 native and 85 contrast-enhanced, whole-body mouse $\mu \mathrm { C T }$ scans. Notably, the acquired resolution significantly exceeds that of typical human CT data, achieving a voxel spacing of 0.14 mm. W.r.t. human CT data, the Mouse $\mu \mathrm { C T }$ exhibit a strong domain shift both in terms of spatial scale and content. In our OW experiments, mouse organ classes are considered UKCs both for our method and baseline methods. Classes for the native set include heart and lung, while classes for the contrast-enhanced set include spleen, lung, heart, kidneys, bladder, stomach, liver, and intestine.

InsectAnatomy [112]: We use ant brain annotations from 143 (118 training, 25 testing) $\mu \mathrm { C T }$ scans from different species. Again, the acquired resolution significantly exceeds that of typical human CT data, achieving isotropic voxel spacings between $1 - 1 0 \mu m$ . In our pipeline, data is min-max normalized and resampled to $7 . 5 \mu m$ for all methods.

![](images/e5417efb65d4207ac18b463ae2cd86e6a896e3a2e6c21bb5c69b9f6afba149e1.jpg)  
Fig. 9. a) Query patch containing $2 ^ { d }$ query nodes (in 2D for simplicity) that are used for similarity search. Green nodes indicate query nodes of interest. b) Original box-kernel with $2 ^ { d }$ nodes, where a common kernel region is shared across all query nodes in the query patch. c) Newly introduced cross-kernel with $1 + 2 \cdot d$ nodes, used for dense labels $( \mathrm { e . g . }$ , message passing). Kernel positions (with anchor nodes in blue) adapt locally depending on relative query node positions in the patch. d) Newly introduced flexi-kernel, used for sparse labels (e.g., weak user-labels). Kernel orientations adapt locally depending on query node positions. From the total of $\cdot _ { 2 ^ { d } }$ +d nodes within the kernel region, $2 ^ { d }$ nodes are being kept while preferring labeled nodes as well as prioritizing nodes within the reference frame.

TotSeg Organ [16]: We compile a subset of TotalSegmentator data comprising liver, pancreas, spleen, kidney left, and kidney right for ablation studies. When using U-Net SuPreM for feature extraction, we use TotSeg Organ as an independent evaluation set.

MSD BRATS [109]: We use MSD BRATS MR brain segmentation data to investigate cross-domain generalization for OW-IS.

## B. Data Preparation

We distinguish between 3 data split types: train hyper (for hyperparameter optimization), train memory (for model training via memory consolidation), and validation. For MSD+KiTS23, we employ an $8 0 : 2 0$ split and partition the training split into a 10% train hyper and a 70% train memory sub-split. TotSeg Organ and InsectAnatomy datasets have official training and validation splits. We partition the TotSeg Organ training split into a 6.25% train hyper and a 93.75% train memory sub-split. For Mouse $\mu \mathrm { C T }$ and MSD BRATS, we apply a 70 : 30 training vs. validation split. Note that Mouse $\mu { \mathrm { C T } } ,$ , InsectAnatomy, and MSD BRATS do not have train hyper splits because we treat them as unknown in our open-set and open-world experiments. Thus, we do not perform any hyperparameter optimization on them.

```latex
Algorithm 1 Inference Step #1: base prediction
Require hyperparameters: $( \Omega ) ^ { l } , k _ { s t a r t } , \phi$
Require query pyramid: $( \dot { h } ) _ { q } ^ { l ^ { \prime } } = \mathcal { F } ( \boldsymbol { x } _ { q } )$
Require user guidance: $u = \{ z ^ { \pm } , \iota \} ; \quad \iota = \{ \beta _ { r e g } , \omega , \gamma , \xi ^ { - } , \beta _ { n n } \}$
Require working memory: $M = \{ z _ { j } , ( h ) _ { j } ^ { l } \}$
initialize:
$m  \operatorname { l e n } ( M )$
$k \gets k _ { : }$ start
$k _ { u s e }  \bar { k } _ { s t }$ tart
$i d _ { k , p a r } \gets p ^ { n } \times [ 1 , 2 , 3 , . . . , m ]$ parent index
$s _ { k , p , a r } \gets \bar { p ^ { n } } \times [ \mathrm { i } , 1 , 1 , . . . , 1 ]$ parent similarity
$( \ddot { \mu } ) _ { j } ^ { l } \gets \mathrm { c a l \bar { c } \_ m a s \bar { k } } ( z ^ { \pm } , ( \Omega ) ^ { l } )$ labeled regions
hierarchical memory search:
for $l = L , 1$ do
id<sup>l</sup> ← get children $( i d _ { k , p a r } , \mu _ { j } ^ { l } ,$ , kernel) $[ p ^ { l } \times k _ { u s e } \cdot n _ { c h } ]$
o<sup>l</sup> ← init meshgrid(Ω<sup>l</sup>)
$s _ { f , c o s } ^ { \check { l } } \gets \mathrm { c o s i n e \_ s i m } ( h _ { q } ^ { l } , h _ { j } ^ { l } [ i d ^ { l } ] )$ $[ p ^ { l } \times k _ { u s e } \cdot n _ { c h } ]$
$\begin{array} { r } { s _ { f , m a t c h } ^ { l ^ { \prime } ^ { \cdots } }  \mathrm { s o f t m a x } _ { p ^ { l } } \overline { { ( l o g ( p ^ { l } ) ^ { \beta _ { r e g } } \cdot s _ { f , c o s } ^ { l } ) } } } \end{array}$ regularized matcher
$s _ { \circ , m a t c h } ^ { \bar { l } }  1 - \mathrm { L 2 } \dot { \mathrm { d i s t } } ( o _ { q } ^ { l } , o _ { j } ^ { l } [ i d ^ { l } ] ) / \mathrm { L 2 d i s t \_ m a x }$ $[ p ^ { l } \times k _ { u s e } \cdot n _ { c h } ]$
$s _ { f } ^ { l }  0 . 5 \cdot ( 1 + s _ { f , c o s } ^ { l } )$
$s _ { c } ^ { l } \gets s _ { { \scriptscriptstyle \alpha } } ^ { l }$ ,match
$s _ { m a t c h } ^ { \breve { l } }  \omega \cdot s _ { f , m a t c h } ^ { l } + ( 1 - \omega ) \cdot s _ { o , } ^ { l }$ match
$s ^ { l }  \omega \cdot s _ { f } ^ { l } + ( 1 - \omega ) \cdot s _ { o } ^ { l }$ feature-proximity balancing
$s _ { { m a t c h } } ^ { l } [ \right. \dot { \mu _ { j } ^ { l } } ] \left. 0$ search in labeled regions
$s _ { m a t c h } ^ { \iota }  s _ { m a t c h } ^ { \iota } \cdot s _ { k , p a r } \mathrm { ~ \qquad ~ } ,$ update, $[ p ^ { l } \times k _ { u s e } \cdot n _ { c h } ]$
$\Gamma _ { , } ^ { '   \gamma \cdot l / ( n - 1 ) } + 0 . 5 \cdot ( 1 - \gamma )$
$s _ { . } ^ { l } \gets \dot { \Gamma } _ { . } \dot { \cdot } s ^ { l } + ( 1 - \Gamma )$ local-global balancing
$s ^ { l }  s ^ { l } \cdot s _ { k }$ ,par update, $[ p ^ { l } \times k _ { u s e } \cdot n _ { c h } ]$
$i d ^ { l } , s _ { m a t c h } ^ { l } \gets$ duplicate cleanup $( i d ^ { l } , s _ { m a t c h } ^ { l } )$
$\mathbf { i f } \ l > \ ^ { \prime \prime }$ 1 then
$k \gets \lfloor \phi \cdot k \rfloor$
end if
$k _ { u s e } \gets \operatorname* { m i n } ( k , \sum \mu _ { j } ^ { l } )$ upper bound
$i d _ { k e e p } \gets \mathrm { g e t \_ t o p k } ( s _ { m \underline { { a } } t c h } ^ { \mathbb { Z } } , k _ { u s e } )$ $[ p ^ { l } \times k _ { u s e } ] _ { . }$
$i d _ { k , p a r } \gets i d ^ { l } [ i d _ { k e e p } ]$ update, $[ p _ { \cdot } ^ { l - 1 } \times k _ { u s e } ]$
$s _ { k , p a r } \gets s ^ { l } [ i \dot { d } _ { k e e p } ] ^ { \ast }$ update, $\stackrel { \sim } { [ } p ^ { l - 1 } \times k _ { u s e } ]$
if init HMP graph then
$i _ { j _ { \ast } } ^ { l }  i d _ { k , p a r } ^ { \ast }$
$s _ { j } ^ { l } \gets s _ { k , p a r }$
if l = 1 then
return $( \bar { i } ) _ { j } ^ { l } , ~ ( s ) _ { j } ^ { l }$ HMP graph
end if
end if
end for
label-retrieval & reduction:
id<sub>z</sub> ← get children(id<sub>k,par</sub>)
$y _ { j }  z ^ { \exists } [ { \bar { i } } d _ { z } ]$
$\tilde { s _ { j } } \gets s _ { k , p a r } [ \dot { i } d _ { z } ]$
$\bar { w _ { j } }  p ^ { \mathrm { 1 } ^ { \circ } } \times [ 0 . ]$ init weights
$i \dot { d } ^ { + } \gets y _ { j } = \dot { + } \dot { 1 }$ positives
$i d ^ { - }  y _ { j } = - 1$ negatives
$w _ { j } [ i d ^ { + } ] \stackrel { \cdot } {  } 1$
$w _ { j } [ i d ^ { - } ]  \xi$ negative click strength
$w _ { \mathcal { I } } ^ { - }  \operatorname { s o f t m a x } _ { k } ( \beta _ { n n } \cdot w _ { j } \cdot s _ { j } )$ commitment
$\begin{array} { r } { y _ { b } ^ { \pm }  \sum _ { j = 1 } ^ { k } w _ { j } \cdot s _ { j } \cdot y _ { j } } \end{array}$
$y _ { b }  0 . 5 \cdot ( 1 + y _ { b } ^ { \pm } )$
return y<sub>b</sub> base prediction ∈ $\{ y \mid 0 \leq y \leq 1 \}$
```

We apply a folding strategy for all open-world experiments to study the impact of stored memory samples on performance. For Mouse $\mu { \mathrm { C T } } ,$ InsectAnatomy, and MSD BRATS, we partition the training split into 2 independent, non-overlapping folds. We do not apply folding of the validation split. This setup allows for memory sizes of $m ~ \leq ~ 1 2 8$ SoIs for any training fold.

Medical data (MSD, KiTS23, TotSeg Organ, and MSD BRATS) is canonicalized and resampled to a spacing of 1.5 mm. We use a crop size of $\Omega ^ { 1 } \ = \ ( 1 2 8 \times 1 2 8 \times$ 128) voxels for all experiments. Window-levels for U-Net SuPreM and VISTA3D are [−175, 250] and [−60, 140], respectively. For TotalSegmentator (organs model), we follow the official pre-processing scheme according to available meta-information [16]. Data is z-normalized for SAM-Med3D and not normalized for nnInteractive in analogy to their official documentation.

```latex
Algorithm 2 Inference Step #2: fusing memory prediction
Require $\tau , y _ { b } , \alpha _ { f u s e } ,$ optional: y<sub>M</sub>
if ∃ y<sub>M</sub> then
$\tau ^ { \ddag } \gets 2 \cdot \tau - 1$
$y _ { b } ^ { \pm } \gets 2 \cdot y _ { b } - 1$
$y _ { M } ^ { \pm } \gets 2 \cdot y _ { M } - 1$
$y _ { M } ^ { \pm }  \mathrm { C D F \_ m a t c h i n g } ( y _ { M } ^ { \pm } , y _ { b } ^ { \pm } , \alpha _ { f u s e } )$
$i \dot { d } ^ { + } \gets y _ { b . } ^ { \pm } > \tau ^ { \pm }$ mask of positives
$\begin{array} { r } { i d ^ { - }  y _ { b } ^ { \mp } < - \tau ^ { \pm } } \end{array}$ mask of negatives
$y _ { r _ { i } } ^ { \pm }  y _ { M } ^ { \pm }$ init (incl. neutral)
$y _ { r _ { \cdot } } ^ { \pm } [ i d ^ { + } ]  \operatorname* { m a x } ( y _ { b } ^ { \pm } [ i d ^ { + } ] , y _ { M } ^ { \pm } [ i d ^ { + } ] )$
$y _ { r } ^ { \pm } [ i d ^ { - } ] \gets \operatorname* { m i n } ( y _ { b _ { \cdot } } ^ { \pm } [ i d ^ { - } ] , y _ { M } ^ { \pm } [ i d ^ { - } ] )$
$y _ { r }  0 . 5 \cdot ( 1 + y _ { r } ^ { \mp } )$
else
$y _ { r } \gets y _ { b }$
end if
return y<sub>r</sub> raw prediction $\in \{ y \mid 0 \leq y \leq 1 \}$
```

```latex
Algorithm 3 Inference Step #3: hierarchical message passing
Require $y _ { r } , \tau , \sigma _ { m p } , n _ { m p } , \eta _ { m p } , \lambda _ { m p } , \beta _ { m p } , ( \Omega ) ^ { l } , ( s _ { j } ) ^ { l } , ( i _ { j } ) ^ { l }$
ν ← voxel coun $( y _ { r } , \tau )$ segment size
$y \gets \mathrm { c l a m p } ( \sigma _ { m p } \cdot ( y _ { r } - \tau ) + \tau , \mathrm { m i n } = 0 ,$ max = 1) sharpen
for $l = L _ { m p } , 1 $ do
$y \gets \mathrm { i n t e r p o l a t e } ( y , \Omega ^ { l } )$
z ← interpolate $( z ^ { \pm } , \dot { \Omega } ^ { l } )$
$i \dot { d } ^ { + } \gets z _ { l } \dot { > } 0 . 5 $
$i d ^ { - }  z _ { l } < - 0 . 5$
$e _ { j } ^ { l } \gets \mathrm { s o f t m a x } _ { \kappa } ( \bar { \beta } _ { m p } \cdot s _ { j } ^ { l } )$ graph edges
for $\operatorname { s t e p } = 1 , \eta _ { m p }$ do
$y [ i d ^ { + } ] \gets 1$
$\bar { y } [ i d ^ { - } \bar { ] } \gets 0$
$\begin{array} { r } { y _ { m s g } \dot { \bf \sigma }  \sum _ { \kappa } e _ { j } ^ { l } \cdot { \bf \boldsymbol { y } } [ i _ { j } ^ { l } ] } \end{array}$ aggregated message
$y  \lambda _ { m p } \cdot y _ { m s g } + ( \bar { 1 } - \lambda _ { m p } ) \cdot y$ update
end for
end for
τ ← sort(y)[ν] conserve size
return y final prediction ∈ $\vdots \{ y \mid 0 \le y \le 1 \}$
```

## III. VIRTUAL USER

For automated evaluation, we employ a virtual user $\mathcal { U } _ { v }$ that simulates the typical behavior of a human user $\mathcal { U } _ { h }$ . For a given dataset and IS setup, $\mathcal { U } _ { v }$ simulates clicks under fixed hyperparameter settings. The initial click is always placed at the volume center where the SoI resides. Afterwards, $\mathcal { U } _ { v }$ executes the correction click strategy described in Algorithm 4, where clicks are placed within a RoI depending on over- or under-segmentation. Given that $\boldsymbol { \mathcal { U } } _ { h }$ rarely clicks at the edge of a ground-truth mask $z _ { g t }$ unless necessary, $\mathcal { U } _ { v }$ similarly prefers placing clicks far away from the edge. For efficient selection of click positions, we sample 32 random candidates from the RoI and select the one farthest away from the RoI’s edge. Given a single-pixel click, we extend it to a 7-voxel cross-shaped click region for MEOW.

Algorithm 4 click-policy of virtual user $\mathcal { U } _ { v }$   
inputs: $y , z _ { g t } , z ^ { \pm } , \tau$   
$\dot { F N }  y < \tau \cup z _ { g t } = 1 \cup z _ { . } ^ { \pm } \neq 1$ false negatives   
$F P \gets \bar { y } \geq \tau \cup z _ { g t } = 0 \cup z ^ { \pm } \neq - 1$ false positives   
if $\sum F P > \sum F { \bar { N } }$ then   
click ← ”negative”   
RoI $ F \bar { P }$   
else   
click ← ”positive”   
$R o I  \dot { F } N$   
end if   
$p _ { j } \gets$ random position $( R o I )$ for $j = 1 , . . . , 3 2$   
d<sub>j</sub> ← calc distance $( p _ { j } ,$ edge(RoI))   
i = argmax $( d _ { j } ) _ { - }$   
return click, $\dot { p _ { j } } [ i ]$

## IV. HYPERPARAMETERS

For adapting to specific use cases and diverse annotation preferences of different users, we introduce several components to increase overall flexibility of our method. These components can be controlled independently and dynamically (e.g., via sliders) as interactive hyperparameters $\bullet ,$ allowing U fine-grained control over segmentation behavior (also see Algorithms 1-3):

• Classification Threshold: A threshold $\tau \in \ [ 0 , 1 ]$ is set to discriminate between foreground and background. High values are more conservative and lead to smaller segments.

• Commitment – Exploration: $\beta _ { r e g } \in [ - 1 , 0 ]$ influences the extent of exploration during correspondence matching by making stored patterns more exclusive. A $\beta _ { r e g }$ close to zero means that stored patterns are more exclusive, in turn leading to more exploration.

• Similarity Score: The significance of spatial proximity compared to feature-similarity (both w.r.t. $z ^ { \pm } )$ is adjusted with $\omega \in [ 0 , 1 ]$ . High values prioritize feature-similarity over proximity.

• Balancing: The balancing parameter $\gamma \in ( - 1 , 1 )$ controls how much emphasis is put on local, high-res features versus global, low-res features. Low values prioritize local features while high values prioritize global features.

• Negative Labels: $\xi ^ { - } \in [ 0 , 1 ]$ determines the overall strength of negative labels obtained by negative clicks or brushes, with $\xi ^ { - } = 1$ having equal contribution of $z ^ { + }$ and $z ^ { - }$

• Commitment – Sparse Labels: $\beta _ { n n } \in [ 0 . 1 , 1 0 0 0 ]$ influences the extent of more similar NNs contributing to base predictions $y _ { b }$ . While large $\beta _ { n n }$ focuses on the most similar NN, small $\beta _ { n n }$ assigns equal importance to all NNs.

• Fusion Strategy: $\alpha _ { f u s e } ~ \in ~ [ 0 , 1 ]$ determines how much to blend from the source distribution $y _ { M } ^ { \pm }$ towards the reference distribution $y _ { b } ^ { \pm }$ before fusing them into $y _ { r }$ (see Algorithm 2).

• HMP Sharpening: $\sigma _ { m p } \in [ 1 , 2 5 ]$ determines sharpening strength for mitigating over-smoothing during HMP.

• Commitment – HMP: $\beta _ { m p } ~ \in ~ [ 0 . 1 , 1 0 ]$ influences the extent of more similar NNs contributing to messages. While large $\beta _ { m p }$ increase graph sparsity, small $\beta _ { m p }$ distribute equal weight to all NNs.

• HMP Strength: $\lambda _ { m p } \in [ 0 , 1 ]$ is the update strength for all HMP levels, with $\lambda _ { m p } = 0$ fully disabling HMP.

Hyperparameter Optimization: To find an optimal set of interactive and non-interactive hyperparameters, we leverage Bayesian optimization [118], a probabilistic approach to maximize an objective function $f ( \mathbf { x } )$ over a bounded parameter space $\mathcal { X } .$ In our case, $f ( \mathbf { x } )$ maps a set of hyperparameters $\textbf { x } \in \ \mathcal { X }$ to the resulting IS performance over an evaluation split. Optimization is performed using Scikit-Optimize [143] (version 0.10.2) with default settings. We use 10 initial, random evaluation steps, followed by optimization until convergence. The Matern kernel´ [144] is chosen to model the smoothness of $f ( \mathbf { x } )$ . We optimize the following parameters within respective intervals: $\tau ~ \in ~ [ 0 . 7 , 0 . 9 5 ]$ $\omega ~ \in ~ [ 0 . 5 , 0 . 9 ] , ~ \gamma ~ \in ~ [ - 0 . 9 9 , 0 ] , ~ \xi ^ { - } ~ \in $ [0.025, 0.5], $l o g ( \beta _ { n n } ) \in [ - 1 . , 3 . ]$ 2 $\beta _ { r e g } \in [ - 1 , 0 ]$ , α<sub>fuse</sub> ∈ [0.75, 1], σ<sub>mp</sub> ∈ [1, 25], $\begin{array} { r l r } { \lambda _ { m p } } & { { } \in } & { [ 0 , 1 ] } \end{array}$ $L _ { m p } \in$ [1, 3], $l o g ( \beta _ { m p } ) \in [ - 1 . , 1 . ]$ $\eta _ { m p } \in [ 6 , 1 2 ]$ TTA $\in [ { \pmb x } , \pmb { \zeta } ]$ Optimal hyperparameters for all feature extractors $\mathcal { F }$ are given in Table XI.

<table><tr><td>ID</td><td>Data</td><td>F</td><td>Mem</td><td>T</td><td>ω</td><td>γ</td><td>ξ</td><td>log(βnn)</td><td>βreg</td><td>αfuse</td><td>σmp</td><td>λmp</td><td>Lmp</td><td></td><td>log(βmp)</td><td>ηmp TTA</td></tr><tr><td>H0</td><td>TotSeg Organ</td><td>F1</td><td>√</td><td>0.873</td><td>0.900</td><td>-0.850</td><td>0.120</td><td>-1.000</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.957</td><td>2</td><td>0.462</td><td>12</td><td>x</td></tr><tr><td>H1</td><td>TotSeg Organ</td><td>F1</td><td>x</td><td>0.821</td><td>0.900</td><td>-0.990</td><td>0.500</td><td>3.000</td><td></td><td></td><td>22.583</td><td>1.000</td><td>2</td><td>-1.000</td><td>12</td><td>x</td></tr><tr><td>H2</td><td>MSD+KiTS23</td><td>F1</td><td>√</td><td>0.863</td><td>0.900</td><td>-0.990</td><td>0.462</td><td>1.308</td><td>0.000</td><td>0.000</td><td>25.000</td><td>1.000</td><td>2</td><td>0.805</td><td>12</td><td>x</td></tr><tr><td>H3</td><td>MSD+KiTS23</td><td>F1</td><td>x</td><td>0.812</td><td>0.900</td><td>-0.990</td><td>0.025</td><td>2.516</td><td></td><td></td><td>21.581</td><td>1.000</td><td>3</td><td>1.000</td><td>12</td><td>x</td></tr><tr><td>H4</td><td>TotSeg Organ</td><td>F2</td><td>√</td><td>0.842</td><td>0.900</td><td>-0.179</td><td>0.177</td><td>0.113</td><td>-1.000</td><td>0.000</td><td>1.000</td><td>0.718</td><td>2</td><td>1.000</td><td>6</td><td>x</td></tr><tr><td>H5</td><td>TotSeg Organ</td><td>F2</td><td>x</td><td>0.818</td><td>0.900</td><td>-0.990</td><td>0.089</td><td>1.516</td><td></td><td></td><td>25.000</td><td>0.441</td><td>3</td><td>0.310</td><td>6</td><td>x</td></tr><tr><td>H6</td><td>MSD+KiTS23</td><td>F2</td><td>√</td><td>0.860</td><td>0.500</td><td>-0.990</td><td>0.025</td><td>-0.851</td><td>-0.709</td><td>0.000</td><td>25.000</td><td>1.000</td><td>2</td><td>1.000</td><td>6</td><td>x</td></tr><tr><td>H7</td><td>MSD+KiTS23</td><td>F2</td><td>x</td><td>0.813</td><td>0.900</td><td>-0.990</td><td>0.500</td><td>3.000</td><td></td><td></td><td>25.000</td><td>0.732</td><td>3</td><td>1.000</td><td>6</td><td>x</td></tr><tr><td>H8</td><td>TotSeg Organ</td><td>F3</td><td>√</td><td>0.827</td><td>0.500</td><td>-0.990</td><td>0.025</td><td>-0.524</td><td>-0.349</td><td>0.000</td><td>6.218</td><td>0.427</td><td>2</td><td>-0.381</td><td>12</td><td>√</td></tr><tr><td>H9</td><td>TotSeg Organ</td><td>F3</td><td>x</td><td>0.786</td><td>0.682</td><td>-0.990</td><td>0.435</td><td>2.395</td><td></td><td></td><td>25.000</td><td>0.799</td><td>3</td><td>0.125</td><td>6</td><td>√</td></tr><tr><td>H10</td><td>MSD+KiTS23</td><td>F3</td><td>√</td><td>0.841</td><td>0.651</td><td>-0.549</td><td>0.025</td><td>2.587</td><td>-0.434</td><td>0.022</td><td>1.000</td><td>0.723</td><td>2</td><td>0.275</td><td>12</td><td>√</td></tr><tr><td>H11</td><td>MSD+KiTS23</td><td>F3</td><td>x</td><td>0.775</td><td>0.707</td><td>-0.990</td><td>0.497</td><td>3.000</td><td></td><td></td><td>25.000</td><td>0.800</td><td>3</td><td>-0.197</td><td>10</td><td>√</td></tr></table>

V. LIST OF SYMBOLS AND ABBREVIATIONS

Notation Description

$\alpha _ { f u s e }$ strength of CDF matching   
$\beta _ { m p }$ Hopfield commitment (message passing)   
$\beta _ { n n }$ Hopfield commitment (k-NN search)   
$\beta _ { r e g }$ Hopfield commitment (matcher)   
$c$ class index   
$\mathrm { C T }$ Computed Tomography   
$D S C$ Dice-Sørensen coefficient   
$( e ) ^ { l }$ message passing edge weights   
$\eta _ { m p }$ number of message passing steps   
$\mathcal { F }$ feature extractor (neural network)   
γ balancing parameter (global vs. local)   
$\mathcal { G }$ transduction module   
$\mathrm { H M P }$ hierarchical message passing   
$( h ) ^ { l }$ multi-scale feature pyramid   
$i d$ voxel indices (pseudo-code)   
$( i ) ^ { l }$ voxel indices (message passing connectivity)   
$\dot { \Gamma }$ instance classification   
$\mathrm { I S }$ interactive segmentation   
$\iota$ interactive hyperparameters   
$k$ number of nearest neighbor voxels   
κ number of nearest neighbor HMP nodes   
k number of nearest neighbor patches   
KKC known known class   
KUC known unknown class   
$L$ number of feature pyramid levels   
$L _ { m p }$ number of message passing levels   
$L _ { s p }$ number of sparse levels   
$\lambda _ { m p }$ message passing strength   
$m$ number of memory samples   
$M$ working memory   
$\mathcal { M }$ memory module   
MR magnetic resonance   
$( \mu ) ^ { l }$ labeled regions of feature pyramid   
n number of clicks   
ND novelty detection   
$N N$ nearest neighbor   
$N o C$ number of clicks metric   
Ω spatial domain (spatial resolution)   
ω balancing parameter (feature vs. position)   
OS open set   
OW open world   
p number of voxels   
RoI region of interest   
S interactive segmentation model   
$\sigma _ { m p }$ message passing sharpening   
SoI segment of interest   
τ classification threshold   
θ learnable parameters of F   
t time   
TTA test-time augmentation   
u user guidance   
$\mathcal { U }$ user   
UKC unknown known class   
UUC unknown unknown class   
w weight   
x model input   
$\xi ^ { - }$ negative click strength   
y prediction   
$z ^ { \pm }$ weak user labels (mask)   
$z _ { g t }$ ground truth labels (mask)   
$z _ { p }$ pseudo-labels (mask)   
$\zeta$ novelty threshold