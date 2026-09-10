# Hyperbolic Geometry for Open-World Object Detection in Remote Sensing Imagery

Wuzhou Li, Jiawei Zhou, Shenghang Wang, and Xiang Li

Abstract—Open-world object detection (OWOD) extends closed-set detection by requiring models to identify unknown objects and incrementally learn them once annotations become available. In remote sensing imagery, object categories often exhibit latent hierarchical relationships that may be inadequately represented in the Euclidean spaces commonly adopted by existing methods, limiting unknown-object recall and incremental-learning performance. To address this issue, we investigate hyperbolic geometry for OWOD in remote sensing imagery and propose HyRS-OWOD. To improve unknown object recall, we design a two-step unknown-object discovery mechanism: a Decoupled Objectness Learning (DOL) module that disentangles foreground perception from semantic information to separate foreground proposals from background regions, followed by a Hyperbolic Uncertainty Learning (HUL) component that leverages the radius of hyperbolic embeddings as an uncertainty-aware cue for known–unknown discrimination. For incremental learning, we develop a Hyperbolic Metric Learning (HML) strategy that enhances inter-class separability, facilitating the incorporation of novel categories while mitigating catastrophic forgetting. Experiments on three remote sensing benchmarks demonstrate consistent improvements in unknown recall and incremental learning over state-of-the-art OWOD methods.

Index Terms—Open-world object detection, remote sensing imagery, hyperbolic geometry, incremental learning.

## I. INTRODUCTION

O <sup>BJECT</sup> <sup>detection</sup> <sup>is</sup> <sup>a</sup> <sup>fundamental</sup> <sup>computer</sup> sensing applications, including urban planning, environmental monitoring, and disaster management [1, 2]. To operate reliably in dynamic real-world scenarios, object detectors must be able to handle unknown objects that continuously emerge and may contain valuable information. However, conventional closedset detectors are typically trained on a predefined set of categories and are expected to recognize only objects belonging to these known (labeled) classes, whereas unknown (unlabeled) objects are often misclassified as similar known categories or ignored as background, limiting the applicability of such detectors in open-world environments. In response, open-world object detection (OWOD) has emerged as a promising paradigm to bridge this gap [3, 4]. As illustrated in Fig. 1, OWOD requires a detector not only to recognize known objects but also to identify previously unseen objects as unknown and incrementally incorporate newly annotated categories across successive learning stages. Recently, OWOD has attracted growing attention in the remote sensing field [5, 6], as exemplified by the study of Tan et al. [7], which extend the unknown-aware region proposal network to handle oriented unknown objects in remote sensing imagery. Hu et al. [8] further leverage the Segment

![](images/bdc781e494021502f3ce2a5cce613db7719019f488536a2c84789aefc09e5da2.jpg)  
Fig. 1. Comparison between closed-set and open-world object detection in remote sensing imagery. Closed-set detectors recognize objects from only a predefined set of known categories, whereas OWOD additionally identifies unseen objects as unknown and incrementally learns them as novel classes once annotations become available.

Anything Model (SAM) to discover potential unknown objects and employ a geometric-mean-based thresholding function to reduce the influence of noisy labels in the raw SAM outputs.

Despite these advances, OWOD in remote sensing imagery remains challenging due to two major issues: the low recall of unknown objects and catastrophic forgetting of previously learned classes during incremental learning. The first issue stems primarily from the lack of supervision, as unknown objects, unlike known classes, are unlabeled during training and can therefore be easily confused with background regions. Furthermore, in remote sensing imagery, unknown objects may correspond to fine-grained extensions or semantic subcategories of known classes, making them prone to confusion with similar known objects, which further suppresses unknown-object recall. The second issue is mainly rooted in high inter-class similarity: the continuous introduction of novel classes shifts the input data distribution, leading to catastrophic forgetting of previously learned classes during adaptation [9]. This effect is especially pronounced when novel classes are visually similar to base classes or correspond to fine-grained variants, as their feature distributions may significantly overlap with those of previously learned classes, thereby exacerbating the problem.

To tackle these challenges, our key insight is to embed the underlying hierarchical structure of remote sensing categories into hyperbolic space. Although Euclidean space has long been the de facto manifold for visual representation learning, its flat geometry cannot faithfully preserve hierarchical or tree-like relationships without considerable distortion. Remote sensing categories, however, often exhibit such hierarchical structure, where unknown objects may be semantically related to known categories at different levels of granularity. For example, a new type of destroyer can be regarded as a type of warship, while warship itself is a subcategory of ship. Embedding such data in Euclidean space may distort the underlying semantic relationships, leading to suboptimal performance. In contrast, hyperbolic space, a Riemannian manifold endowed with constant negative curvature, offers a more suitable geometry for modeling the hierarchical structure of remote sensing data. More importantly, its geometric properties are relevant to the two principal challenges of OWOD. On one hand, the hyperbolic radius of embeddings can naturally serve as a measure of uncertainty for discovering potential unknown objects [10, 11]. On the other hand, its exponential volume growth and hierarchical structure provide a suitable foundation for metric learning, helping maintain discriminative relationships among previously learned and novel classes during incremental learning, thereby mitigating catastrophic forgetting.

Motivated by these observations, we introduce hyperbolic geometry into OWOD for remote sensing imagery. To improve the recall of unknown objects, we design a two-step unknown-object discovery mechanism comprising foreground–background separation and known–unknown discrimination. In the first step, we propose Decoupled Objectness Learning (DOL), which explicitly models class-agnostic objectness to distinguish foreground proposals from background regions. Although objectness is intended to be classagnostic, supervision solely from known classes may introduce a confounding effect that suppresses potential unknown objects [12, 13, 14]. To alleviate this issue, DOL introduces a decoupling loss that reduces the statistical correlation between objectness scores and class predictions. In the second step, termed Hyperbolic Uncertainty Learning (HUL), we employ an uncertainty-based heuristic to discriminate unknown objects from known classes. Intuitively, among foreground objects, unknowns can be regarded as outliers with high predictive uncertainty. Inspired by prior studies that leverage uncertainty estimation for outof-distribution classification [11] and segmentation tasks [15], HUL uses the hyperbolic radius of a learned embedding, measured by its distance to the origin, as a geometry-aware uncertainty cue: proposals with smaller radii tend to correspond to more abstract and ambiguous concepts. As such, foreground proposals with high objectness scores but small hyperbolic radii are more likely to be identified as unknown objects.

To address catastrophic forgetting, our key idea is to learn stable and discriminative representations. Metric learning has been an effective method for learning an embedding space or distance metric in which samples from the same class are drawn closer together, while samples from different classes are pushed farther apart [16, 17]. Since remote sensing categories typically exhibit a hierarchical structure, we further perform metric learning in hyperbolic space. Compared with its Euclidean counterpart, hyperbolic space exhibits exponential volume growth, which naturally accommodates hierarchical trees with low distortion [10]. When combined with metric learning, this geometry promotes more compact intra-class representations and larger inter-class margins. Accordingly, we propose Hyperbolic Metric Learning (HML), which combines a class-balanced proposal buffer with distance-based hard-negative weighting. The proposal buffer retains representative embeddings of previously learned classes, while the weighting mechanism emphasizes visually confusing proposals from different classes. Consequently, HML reduces interference between previously learned and novel classes, thereby mitigating catastrophic forgetting during incremental learning.

To summarize, our main contributions are as follows:

1) We propose HyRS-OWOD, a novel open-world object detection (OWOD) framework for remote sensing imagery that, to the best of our knowledge, is the first to investigate the effectiveness of hyperbolic geometry for OWOD in the remote sensing domain. Our method enables the detection of known classes, discovery of unknown objects, and incremental learning of novel classes in a unified manner.

2) We design a two-step unknown-object discovery mechanism comprising Decoupled Objectness Learning (DOL), which models class-agnostic objectness with a decoupling loss to separate foreground from background, and Hyperbolic Uncertainty Learning (HUL), which leverages hyperbolic uncertainty to identify unknown objects from known categories, thereby improving unknown-object recall.

3) We propose Hyperbolic Metric Learning (HML) for incremental object detection. By combining a class-balanced proposal buffer with distancebased hard-negative weighting, HML promotes intra-class compactness and inter-class separation while reducing interference between previously learned and novel classes.

4) Comprehensive experiments on three remote sensing object detection benchmarks demonstrate the effectiveness and superiority of our proposed HyRS-OWOD over existing approaches.

## II. RELATED WORK

## A. Object Detection

Deep learning has substantially advanced object detection through representative architectures such as the YOLO family [18, 19, 20, 21], two-stage detectors represented by Faster R-CNN [22, 23, 24], and transformer-based detectors such as DETR [25, 26]. In remote sensing, these architectures have been adapted to address domain-specific challenges, including small objects [27], complex backgrounds [2], and arbitrarily oriented objects [28]. Nevertheless, most existing methods are developed under a static, closed-set assumption, restricting their detection to a fixed set of predefined classes. To relax this assumption, open-set detection aims to identify objects outside the known label space [5], whereas open-vocabulary detection exploits external semantic knowledge to recognize a broader range of categories [29]. These developments highlight the unique challenges of open-world perception and motivate the need for a more continual and comprehensive approach.

## B. Open-World Object Detection

OWOD combines the detection of unknown objects with their incremental learning as novel classes once annotations are progressively provided. Joseph et al. [3] first formulated this task and proposed ORE, a Faster R-CNN-based framework that employs an unknown-aware pseudo-labeling scheme and an energy-based classifier to distinguish unknown objects from known classes. Following this seminal work, several studies have adopted different heuristic assumptions for discovering potential unknown objects. OW-DETR [30], for example, employs attention-driven pseudo-labeling to mine potential unknown samples. Wang et al. [13] utilize random proposal generation to mitigate the confounding effect and explore more potential proposals of unknown objects. CAT [31] decouples localization and classification with a cascade decoder and proposes a self-adaptive pseudo-labeling scheme for generating robust pseudo-labels. Sun et al. [12] further address the interference between objectness and class information via orthogonalization. Zohar et al. [32] directly learn a probabilistic objectness head from only known classes, eliminating the need for pseudo-labels. Another line of work introduces foundation-model and vision-language models [33, 34, 35, 36, 37] into OWOD for external semantic knowledge, leveraging vision-language alignment, textual descriptions, and attribute-level cues to enhance generalization to unseen categories, partially bridging OWOD with open-vocabulary object detection.

Compared with generic-domain OWOD, research on OWOD in remote sensing remains relatively limited. OPODet [7] primarily improves detection of unknown rotated objects in remote sensing images. Saini et al. [38] employ multimodal large language models to assign semantic labels to unknown candidate regions generated by closed-set detectors, enabling the automatic discovery and naming of unknown classes. However, these methods focus mainly on unknownobject discovery without subsequent incremental learning, thus falling short of the full OWOD requirements. More recently, Hu et al. [8] proposed a SAM-guided OWOD framework with a label-mapping alignment mechanism to filter noisy candidate proposals. Despite this progress, existing remote sensing OWOD methods still largely overlook the latent hierarchical structure among remote sensing categories. Motivated by these observations, we explore hyperbolic geometry for OWOD in remote sensing images, aiming to better model the semantic relationships between unknown objects and known classes.

## C. Hyperbolic Geometry

Hyperbolic geometry has attracted increasing attention in representation learning because of its ability to represent hierarchical and tree-like structures with low distortion [39, 40]. Hyperbolic representations have been applied to natural language processing [41], knowledge graph modeling [42], and graph neural networks [43]. More recently, they have been extended to various computer vision tasks, including continual learning [44, 17], few-shot learning [45], image classification [46], semantic segmentation [47, 48], and object detection [49]. These studies suggest that hyperbolic embeddings can effectively capture latent semantic similarities and hierarchical relationships in visual data, providing a complementary alternative to conventional Euclidean representations. Moreover, another key property lies in the Poincare model of´ hyperbolic spaces, where the embedding radius—i.e., the distance to the origin—can serve as a measure of model uncertainty [41, 39]. This property has been exploited in tasks such as visual anomaly recognition [11] and image segmentation [47].

Hyperbolic geometry has also recently been introduced into OWOD. Kong et al. [50] develop a hyperbolic learning framework that models hierarchical relationships between visual and caption embeddings, thereby reducing the influence of hallucinated synthetic captions. Doan et al. [51] propose Hyp-OW, which combines hyperbolic contrastive learning for object representation, a superclass regularizer for modeling class-level hierarchies, and adaptive relabeling for retrieving unknown objects based on hyperbolic distance. These studies demonstrate the potential of hyperbolic geometry for modeling category hierarchies and facilitating unknown-object discovery. However, they are primarily designed for natural-scene imagery, and the application of hyperbolic geometry to remote sensing OWOD remains underexplored.

## D. Hyperbolic Learning in Remote Sensing

Remote sensing imagery often contains semantically related object categories with implicit hierarchical structures, which naturally motivates the use of hyperbolic spaces. Recent studies have introduced hyperbolic geometry into remote sensing tasks, including image classification [52], semantic segmentation [53, 54], and change detection [55]. In this work, we introduce hyperbolic geometry into remote sensing OWOD to better capture the latent hierarchical relationships between known and unknown classes and formulate an uncertainty-aware unknown object discovery strategy based on the hyperbolic radius. Moreover, we further perform metric learning in hyperbolic space to preserve intra-class compactness and interclass separability during incremental learning, thereby advancing OWOD for remote sensing imagery.

## III. PRELIMINARIES

In this section, we begin by introducing the fundamentals of hyperbolic space, followed by a formal definition of the Poincare ball model. We then describe´ several commonly used operations within this model.

## A. Hyperbolic Space

Hyperbolic space is defined as a complete, simply connected Riemannian manifold with constant negative sectional curvature. Among its various isometric models, the Poincare ball model has conformal prop-´ erties, which enable effective modeling of hierarchical data structures and facilitate efficient optimization.

## B. Poincare Ball Model´

An n-dimensional Poincare ball model with constant´ negative curvature $- c \left( c > 0 \right)$ is defined as a Riemannian manifold $\mathbb { M } _ { c } ^ { n } = ( B _ { c } ^ { n } , g ^ { \underline { { \hat { \mathbb { B } } } } } )$ , where $B _ { c } ^ { n }$ denotes an open ball embedded in the Euclidean space $\mathbb { R } ^ { n }$

$$
B _ { c } ^ { n } = \left\{ \mathbf { x } \in \mathbb { R } ^ { n } : c \| \mathbf { x } \| ^ { 2 } < 1 \right\} ,\tag{1}
$$

with $\| \cdot \|$ the Euclidean norm. The radius of the ball is $1 / { \sqrt { c } } .$ The Riemannian metric tensor of the Poincare´ ball model is conformal to the Euclidean metric and is given by

$$
\begin{array} { r } { g _ { \mathbf { x } } ^ { \mathbb { B } } = \left( \lambda _ { \mathbf { x } } ^ { c } \right) ^ { 2 } g ^ { E } , } \end{array}\tag{2}
$$

where $g ^ { E }$ denotes the Euclidean metric tensor and $\lambda _ { \mathbf { x } } ^ { c }$ is the conformal factor:

$$
\lambda _ { \mathbf { x } } ^ { c } = \frac { 2 } { 1 - c \| \mathbf { x } \| ^ { 2 } } .\tag{3}
$$

As $\| \mathbf { x } \|$ approaches the boundary, the conformal factor grows without bound, enabling the model to capture hierarchical structures with exponentially expanding capacity near the boundary.

## C. Mobius addition¨

We adopt the formalism of Mobius gyrovector ¨ spaces, which generalize several Euclidean operations to hyperbolic geometry and enables algebraic computations within the Poincare ball model. For´ $\mathbf { x } , \mathbf { y } \in B _ { c } ^ { n }$ the Mobius addition¨ $\oplus _ { c }$ is defined as:

$$
\mathbf { x } \oplus _ { c } \mathbf { y } = { \frac { \left( 1 + 2 c { \bigl \langle } \mathbf { x } , \mathbf { y } { \bigr \rangle } + c { \bigl \| } \mathbf { y } { \bigr \| } ^ { 2 } \right) \mathbf { x } + \left( 1 - c { \bigl \| } \mathbf { x } { \bigr \| } ^ { 2 } \right) \mathbf { y } } { 1 + 2 c { \bigl \langle } \mathbf { x } , \mathbf { y } { \bigr \rangle } + c ^ { 2 } \| \mathbf { x } \| ^ { 2 } \| \mathbf { y } \| ^ { 2 } } } .\tag{4}
$$

## D. Exponential Map

For $\mathbf { x } \in B _ { c } ^ { n }$ and $\mathbf { v } \in T _ { \mathbf { x } } B _ { c } ^ { n }$ , the exponential map is defined as:

$$
\exp _ { \mathbf { x } } ^ { c } ( \mathbf { v } ) = \mathbf { x } \oplus _ { c } \left( \operatorname { t a n h } \left( \sqrt { c } \frac { \lambda _ { \mathbf { x } } ^ { c } \| \mathbf { v } \| } { 2 } \right) \frac { \mathbf { v } } { \sqrt { c } \| \mathbf { v } \| } \right)\tag{5}
$$

## E. Distance and Hyperbolic Radius

For $\mathbf { x } , \mathbf { y } \in B _ { c } ^ { n }$ , the hyperbolic distance is given by

$$
d _ { c } ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { \sqrt { c } } \operatorname { a r c c o s h } \left( 1 + \frac { 2 c \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } { ( 1 - c \| \mathbf { x } \| ^ { 2 } ) ( 1 - c \| \mathbf { y } \| ^ { 2 } ) } \right) .
$$

The hyperbolic radius of a point x is defined as its Poincare distance to the ball origin:´

$$
r _ { c } ( { \bf x } ) \triangleq d _ { c } ( { \bf x } , { \bf 0 } ) = \frac { 2 } { \sqrt { c } } \mathrm { a r c t a n h } \left( \sqrt { c } \| { \bf x } \| \right) .\tag{7}
$$

The hyperbolic radius of an object proposal serves as a measure of model uncertainty: higher confidence corresponds to embeddings lying farther away from the origin in hyperbolic space, whereas lower confidence leads to embeddings located closer to the origin [10, 15].

TABLE I  
RELATIVE δ-HYPERBOLICITY VALUES OF EMBEDDINGS EXTRACTED BY DIFFERENT FEATURE EXTRACTORS.
<table><tr><td></td><td>NWPU VHR-10</td><td>DIOR</td><td>DOTA</td></tr><tr><td>VGG16</td><td>0.2306</td><td>0.2296</td><td>0.2525</td></tr><tr><td>GoogLeNet</td><td>0.2196</td><td>0.2415</td><td>0.2372</td></tr><tr><td>ResNet50</td><td>0.2544</td><td>0.2751</td><td>0.2910</td></tr></table>

## F. δ-Hyperbolicity

Following the analysis in [10], we compute the Gromov δ-hyperbolicity of each remote sensing dataset to measure its structural properties. This evaluation is performed by first computing the Gromov product for points $x , y , z \in { \mathcal { X } }$ , which is defined as:

$$
( y , z ) _ { x } = \frac { 1 } { 2 } \left( d ( x , y ) + d ( x , z ) - d ( y , z ) \right) ,\tag{8}
$$

where $( \mathcal { X } , d )$ denotes an arbitrary metric space. For a set of points, we construct the matrix M of pairwise Gromov products. The δ-hyperbolicity is then computed as:

$$
\delta = \operatorname* { m a x } _ { i , j } \left[ ( M \otimes M ) _ { i j } - M _ { i j } \right] ,\tag{9}
$$

where $\otimes$ denotes the min-max matrix product defined as:

$$
( A \otimes B ) _ { i j } = \operatorname* { m a x } _ { k } \operatorname* { m i n } \{ A _ { i k } , B _ { k j } \} .\tag{10}
$$

with $i , j ,$ k denoting the indices of matrices A and $B .$ We further compute the relative hyperbolicity to ensure scale invariance. Its value ranges from 0 to 1, where values closer to 0 indicate stronger hierarchical structure in the data, while values closer to 1 suggest a weaker hierarchy.

$$
\delta _ { \mathrm { r e l } } = \frac { \delta } { \operatorname* { m a x } _ { i , j } M _ { i j } } .\tag{11}
$$

As shown in [10], $\delta _ { \mathrm { r e l } }$ can be used to estimate the manifold curvature of the Poincare ball model:´

$$
c ( \mathcal { X } ) = \left( \frac { 0 . 1 4 4 } { \delta _ { \mathrm { r e l } } } \right) ^ { 2 } .\tag{12}
$$

We follow the above procedure to evaluate $\delta _ { \mathrm { r e l } }$ on remote sensing detection datasets, including NWPU VHR-10, DIOR, and DOTA. For each dataset, we extract region-level embeddings from object proposals using different detection models. We randomly sample 450 embeddings per run and the final results are reported in Table I.

## IV. METHODOLOGY

## A. Problem Formulation

Let $\mathcal { K } ^ { t } = \{ 1 , 2 , \ldots , C \}$ denote the annotated known classes, and ${ \mathcal { U } } ^ { t } = \{ C + 1 , . . . \}$ denote the unbounded set of unknown classes of interest that may appear during inference. Given a time step t, the known object classes ${ \boldsymbol { \mathcal { K } } } ^ { t }$ are labeled in the dataset ${ \mathcal { D } } ^ { t } = \{ { \mathcal { T } } ^ { t } , { \mathcal { L } } ^ { t } \}$ which consists of N input images $\mathcal { T } ^ { t } = \{ I _ { 1 } , \ldots , I _ { N } \}$ and their corresponding label sets $\mathcal { L } ^ { t } = \{ L _ { 1 } , . . . , L _ { N } \}$ Each ${ \cal L } _ { i } = \{ l _ { i 1 } , . . . , l _ { i P _ { i } } \}$ contains the annotations of $P _ { i }$ object instances in image $I _ { i } .$ Each instance annotation is represented as $l _ { i j } = [ c _ { i j } , x _ { i j } , y _ { i j } , w _ { i j } , h _ { i j } ] ,$ where $c _ { i j }$ denotes its class label, $( x _ { i j } , y _ { i j } )$ denotes the center coordinates of its bounding box, and $w _ { i j }$ and $h _ { i j }$ denote its width and height, respectively.

In OWOD, the trained model $\mathcal { M } ^ { t }$ is required to detect objects from known classes ${ \boldsymbol { \mathcal { K } } } ^ { t }$ and identify unknown objects belonging to $\mathcal { U } ^ { t }$ , labeling them as ”unknown”. The model then sends the discovered unknown objects to an oracle, which annotates the new classes of interest. Together with their corresponding training examples, these annotations update $\mathcal { D } ^ { t }$ to $\mathcal { D } ^ { t + 1 }$ and ${ \boldsymbol { \mathcal { K } } } ^ { t }$ to $\mathcal { K } ^ { t + 1 } = \{ 1 , 2 , \ldots , C , \ldots , C + n \}$ The model incorporates these n novel classes into the set of known classes and incrementally updates itself to $\mathcal { M } ^ { t + 1 }$ without retraining from scratch on the entire dataset $\mathcal { D } ^ { t + 1 }$ while preserving previously learned knowledge. This cycle continues over the model’s lifespan.

## B. Overall Architecture

The overall architecture of the proposed HyRS-OWOD framework is illustrated in Fig. 2. HyRS-OWOD is built upon RandBox [13] and introduces three key components: (1) Decoupled Objectness Learning (DOL) (Sec. IV-C), which separates foreground from background; (2) Hyperbolic Uncertainty Learning (HUL) (Sec. IV-D), which distinguishes unknown objects from known classes; and (3) Hyperbolic Metric Learning (HML) (Sec. IV-E), which supports learning novel classes while mitigating catastrophic forgetting.

Given an input image, the model first extracts feature maps and then randomly generates a set of candidate proposal boxes, which are fed into their exclusive dynamic head [56] to generate object features. The bounding box regression branch refines the proposal locations based on these object features. In parallel, the object features are projected into hyperbolic space to obtain hyperbolic embeddings. The hyperbolic embeddings are then fed into a known-class classification branch and a class-agnostic objectness branch. The objectness branch is implemented as a lightweight multilayer perceptron (MLP) that outputs an objectness score for each proposal. The hyperbolic embeddings are further used for uncertainty estimation and metric learning in the following modules.

During training, we follow RandBox to employ the dynamic matcher [57] to assign proposal boxes to ground-truth instances. The matched proposals are used for known class detection, while the remaining unmatched proposals with top objectness are selected as initial candidate unknowns and used to optimize

![](images/1725db8459556bdada51518ad179eee72dfee599a7f48354c7069e391b1b4e3d.jpg)  
Fig. 2. Overview of our proposed HyRS-OWOD framework. The backbone and FPN extract multi-scale features from an input image, which are processed together with randomly generated proposals by the dynamic head to obtain proposal features f<sub>i</sub>. These features are used for bounding-box regression and mapped into the Poincare ball via the exponential map´ $\mathrm { e x p } _ { \mathbf { 0 } } ^ { c } ( \cdot )$ . For unknown-object prediction, DOL estimates class-agnostic objectness, while HUL measures uncertainty using the hyperbolic radius; proposals with high objectness and high hyperbolic uncertainty are identified as unknown. In parallel, the classifier predicts known object categories. During incremental training, HML is additionally applied to learn discriminative representations and mitigate catastrophic forgetting.

HUL. At inference, unknown objects are directly identified based on their class-agnostic objectness and hyperbolic uncertainty. During incremental learning, we retain 50 exemplars from each previously learned class following the exemplar-replay protocols adopted in [3, 12, 8].

## C. Decoupled Objectness Learning

Objectness should indicate whether a proposal corresponds to a foreground object, irrespective of its semantic category. To train the objectness branch, we adopt a binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { o b j } } = - \sum _ { b } \left[ y _ { b } ^ { \mathrm { f g } } \log ( o _ { b } ) + ( 1 - y _ { b } ^ { \mathrm { f g } } ) \log ( 1 - o _ { b } ) \right] ,\tag{13}
$$

where $o _ { b } ~ \in ~ [ 0 , 1 ]$ and $y _ { b } ^ { \mathrm { { f g } } } \in \{ 0 , 1 \}$ are the predicted objectness score and the ground-truth foreground/background label for proposal b, respectively.

Because the objectness branch is supervised using only known-class annotations, its predictions may become correlated with known-class confidence. Consequently, proposals containing unknown objects may receive low objectness scores and be incorrectly suppressed as background. To reduce this semantic bias, we introduce a decoupling loss that penalizes the linear correlation between the objectness scores and known-class predictions. Inspired by [12], we measure their linear correlation using the squared correlation coefficient:

$$
\mathcal { L } _ { \mathrm { d e c } } = \sum _ { c = 1 } ^ { C } \frac { \left( \mathrm { c o v } ( p _ { b , c } , o _ { b } ) \right) ^ { 2 } } { \mathrm { v a r } ( p _ { b , c } ) \mathrm { v a r } ( o _ { b } ) } ,\tag{14}
$$

where covariance and variance are computed over all proposals within a mini-batch; C denotes the number of known classes. By reducing the statistical dependence between objectness and known-class predictions, DOL encourages the objectness branch to capture class-agnostic foreground cues rather than category-specific semantic information.

## D. Hyperbolic Uncertainty Learning for Unknown Discrimination

We estimate the uncertainty of each proposal by its radial position in the Poincare ball. Given an input´ image I, let $\mathcal { P } _ { I } = \{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { N _ { I } }$ denote the set of proposal embeddings, where $\mathbf { f } _ { i } \in \mathbb { R } ^ { D }$ . Each proposal embedding is mapped into a d-dimensional Poincare ball´ $\mathbb { B } _ { c } ^ { d } = \{ \mathbf { z } \in \bar { \mathbb { R } } ^ { d } : c \| \mathbf { z } \| _ { 2 } ^ { 2 } < 1 \}$ with constant negative curvature $- c$ by a lightweight hyperbolic projection head $g _ { \phi } : \mathbb { R } ^ { D }  \mathbb { R } ^ { d }$ , followed by a projection onto the ball:

$$
\mathbf { z } _ { i } = \Pi _ { \mathbb { B } _ { c } ^ { d } } \left( g _ { \phi } ( \mathbf { f } _ { i } ) \right) = \operatorname { t a n h } \left( \sqrt { c } \left. g _ { \phi } ( \mathbf { f } _ { i } ) \right. _ { 2 } \right) \frac { g _ { \phi } ( \mathbf { f } _ { i } ) } { \sqrt { c } \left. g _ { \phi } ( \mathbf { f } _ { i } ) \right. _ { 2 } } ,
$$

where $\mathbf { z } _ { i } \in \mathbb { B } _ { c } ^ { d }$ denotes the hyperbolic proposal embedding, and the zero vector case is defined by continuity.

we compute the hyperbolic radius of proposal i as its distance to the origin [10, 15]:

$$
\rho _ { i } = d _ { c } ( \mathbf { z } _ { i } , \mathbf { 0 } ) = \frac { 2 } { \sqrt { c } } \operatorname { a r c t a n h } \left( \sqrt { c } \| \mathbf { z } _ { i } \| _ { 2 } \right) .\tag{16}
$$

Since the scale of the hyperbolic radius may vary across images, we normalize the radius within each image:

$$
\tilde { \rho } _ { i } = \frac { \rho _ { i } } { \operatorname* { m a x } _ { 1 \leq j \leq N _ { I } } \rho _ { j } + \epsilon } ,\tag{17}
$$

where ϵ is a small constant to avoid division by zero. The hyperbolic uncertainty score is then defined as

$$
u _ { i } = 1 - \tilde { \rho } _ { i } ,\tag{18}
$$

where a larger $u _ { i }$ indicates higher uncertainty. In this way, proposals close to the origin obtain larger uncertainty scores, whereas confident known proposals near the boundary receive smaller uncertainty scores.

(a) Unknown candidate identification  
![](images/1f13070aee0b4f3e0d2358a1417284f5e99e3f7d75e7f3d3cc205437eb57fb50.jpg)  
(b) Unknown regularization with ${ \mathcal { L } } _ { \mathrm { H U L } }$ (c) Learned unknown-discriminative space  
Fig. 3. Illustration of the proposed HUL strategy for known–unknown discrimination. Different colors represent proposals from different known classes, gray points denote candidate unknown proposals. (a) Candidate unknown proposals are selected for uncertainty-aware learning. (b) The HUL loss ${ \mathcal { L } } _ { \mathrm { H U L } }$ encourages candidate unknown proposals to lie in the high-uncertainty region near the origin. (c) After training, uncertain unknown proposals tend to lie closer to the origin than confident known proposals. The yellow dashed circle represents the normalized radius threshold $\tilde { \rho } = \tau _ { \rho }$ used for known–unknown discrimination.

To explicitly encourage unknown proposals to stay in the high-uncertainty region of the Poincare ball,´ we introduce a smooth hyperbolic unknown loss to regularize candidate unknown proposals:

$$
\mathcal { L } _ { \mathrm { H U L } } = \frac { 1 } { \vert \mathcal { U } \vert } \sum _ { i \in \mathcal { U } } \frac { 1 } { \beta } \log \left( 1 + \exp \left( \beta ( \tilde { \rho } _ { i } - \tau _ { \rho } ) \right) \right) ,\tag{19}
$$

where $\mathcal { U }$ denotes the set of candidate unknown proposals, $\tau _ { \rho }$ is the radius threshold, and $\beta$ controls the smoothness of the margin. When $\beta$ becomes large, Eq. (19) approaches the hard hinge loss max $( 0 , \tilde { \rho } _ { i } -$ $\tau _ { \rho } )$ . The overall HUL process is illustrated in Fig. 3.

E. Hyperbolic Metric Learning for Incremental Learning

In this subsection, we introduce HML to mitigate catastrophic forgetting in incremental learning. In particular, HML assigns larger weights to hard-negative proposal pairs that belong to different classes but are located close to each other.

At incremental step $t ,$ let ${ \mathcal { C } } ^ { t }$ denote the set of all classes learned up to and including step t. We maintain a class-balanced proposal buffer Q, which stores hyperbolic proposal embeddings from previously learned classes. Given the foreground proposal embeddings in the current mini-batch $B ,$ we construct the metric learning set as $\mathcal { A } = \mathcal { B } \cup \mathcal { Q }$ . For each anchor proposal embedding $\mathbf { z } _ { i } \in { \mathcal { A } }$ with class label $y _ { i } \in \mathcal { C } ^ { t }$ , we define its positive and negative sets as

$$
\{ \begin{array} { l l } { \mathcal { P } ( i ) = \{ p \in \mathcal { A } \} \{ i \} : y _ { p } = y _ { i } \} , } \\ { \mathcal { N } ( i ) = \{ n \in \mathcal { A } : y _ { n } \neq y _ { i } \} . } \end{array}\tag{20}
$$

Only anchors with non-empty positive sets are used for metric learning, and the corresponding anchor index set is denoted as $s .$

In incremental learning, the most informative negative samples are typically those that lie close to the anchor in the embedding space but belong to different classes. To exploit this, we assign larger weights to hard negatives according to their Poincare distances to´ the anchor:

$$
\alpha _ { i , n } ^ { - } = \frac { \exp { ( - d _ { c } ( \mathbf { z } _ { i } , \mathbf { z } _ { n } ) / \tau _ { \mathrm { h } } ) } } { \sum _ { m \in \mathcal { N } ( i ) } \exp { ( - d _ { c } ( \mathbf { z } _ { i } , \mathbf { z } _ { m } ) / \tau _ { \mathrm { h } } ) } } , \quad n \in \mathcal { N } ( i ) ,\tag{21}
$$

where $d _ { c } ( \cdot , \cdot )$ denotes the Poincare distance and´ $\tau _ { \mathrm { h } }$ is a temperature parameter that controls the hardness distribution. A smaller distance yields a larger weight, thereby directing the model to focus on more confusing inter-class proposal pairs. We define the positive term and the hard-negative term as

$$
\left\{ \begin{array} { l l } { \displaystyle A _ { i } ^ { + } = \sum _ { p \in \mathcal { P } ( i ) } \exp \left( - d _ { c } ( \mathbf { z } _ { i } , \mathbf { z } _ { p } ) / \tau _ { \mathrm { m } } \right) , } \\ { \displaystyle A _ { i } ^ { - } = \sum _ { n \in \mathcal { N } ( i ) } \alpha _ { i , n } ^ { - } \exp \left( - d _ { c } ( \mathbf { z } _ { i } , \mathbf { z } _ { n } ) / \tau _ { \mathrm { m } } \right) . } \end{array} \right.\tag{22}
$$

where $\tau _ { \mathrm { m } }$ is the metric learning temperature. HML loss is then written as

$$
\mathcal { L } _ { \mathrm { H M L } } = - \frac { 1 } { \left| S \right| } \sum _ { i \in S } \log \frac { A _ { i } ^ { + } } { A _ { i } ^ { + } + A _ { i } ^ { - } } .\tag{23}
$$

This objective pulls same-class proposal embeddings closer together in the Poincare ball while pushing hard´ negatives from different classes apart. By emphasizing the most confusing inter-class pairs, the model learns more discriminative class boundaries during incremental learning.

The proposal buffer Q is updated after each minibatch by storing foreground proposal embeddings along with their corresponding labels. To prevent class imbalance, we retain at most $K _ { \mathrm { m a x } }$ embeddings per class and update the buffer in a first-in-first-out manner.

## F. Overall Training Objective

The detection loss for known objects consists of a classification loss $\mathcal { L } _ { c l s }$ and a bounding box regression loss $\mathcal { L } _ { r e g } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { k n } } = \mathcal { L } _ { c l s } + \mathcal { L } _ { r e g } ,\tag{24}
$$

where $\mathcal { L } _ { c l s }$ is the focal loss [58] for class imbalance and the $\mathcal { L } _ { r e g }$ is smooth $L _ { 1 }$ loss [59].

The overall training objective is therefore

$$
{ \mathcal { L } } _ { \mathrm { b a s e } } = { \mathcal { L } } _ { \mathrm { k n } } + \lambda _ { \mathrm { o b j } } { \mathcal { L } } _ { \mathrm { o b j } } + \lambda _ { \mathrm { d e c } } { \mathcal { L } } _ { \mathrm { d e c } } + \lambda _ { \mathrm { H U L } } { \mathcal { L } } _ { \mathrm { H U L } } ,\tag{25}
$$

where $\lambda _ { \mathrm { o b j } } , \lambda _ { \mathrm { d e c } }$ , and $\lambda _ { \mathrm { H U I } }$ are the corresponding loss weights.

In the incremental learning stage, we further introduce the weighted HML loss ${ \mathcal { L } } _ { \mathrm { H M L } }$ . The overall incremental training objective is defined as

$$
\mathcal { L } _ { \mathrm { i n c } } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { H M L } } \mathcal { L } _ { \mathrm { H M L } } ,\tag{26}
$$

where $\lambda _ { \mathrm { H M L } }$ controls the contribution of the weighted HML objective.

## V. EXPERIMENTS

In this section, we first describe the datasets, evaluation metrics, and implementation details. We then present quantitative and qualitative comparisons against baseline and state-of-the-art approaches. Finally, we conduct ablation studies to validate the effectiveness of the proposed components.

## A. Datasets

1) NWPU VHR-10: This dataset consists of 650 positive images, categorized into ten geospatial object classes for object detection, including airplane (AP), vehicle (VH), ship (SH), harbor (HB), baseball diamond (BD), basketball court (BC), tennis court (TC), ground track field (GTF), storage tank (ST), and bridge (BR). The spatial resolution of the images ranges from 0.5 to 2 m.

2) DIOR: This dataset comprises 23,463 aerial images, containing 192,472 object instances across 20 categories: airplane (AP), windmill (WM), airport (AT), train station (TS), baseball field (BF), tennis court (TC), bridge (BR), storage tank (ST), chimney (CM), ship (SH), dam (DM), expressway service area (EA), stadium (SD), expressway toll station (ES), harbor (HB), golf course (GF), vehicle (VH), ground track field (GTF), overpass (OP), and basketball court (BC). The spatial resolution ranges from 0.5 to 30 m, and each image has a fixed size of 800 × 800 pixels.

3) DOTA-v1.5: This dataset comprises 2,806 largescale aerial images, with dimensions ranging from 800 × 800 to 4, $0 0 0 \times 4 , 0 0 0$ pixels. It contains 402,089 annotated instances across 16 categories: basketball court, baseball diamond, bridge, container crane, ground track field, harbor, helicopter, large vehicle, plane, roundabout, small vehicle, ship, storage tank, soccer ball field, swimming pool, and tennis court.

4) Dataset Splitsfor OWOD: For a fair comparison, we adopt the same known–unknown class partitions as in [8], including the 16+4, 10+10, and 4+16 splits on DIOR, the 8+8 split on DOTA, and the 8+2 split on NWPU VHR-10. In addition to these benchmark settings, we establish a progressive OWOD protocol on NWPU VHR-10 to further evaluate the model under sequential category discovery, as summarized in Table III. In Task 1, seven categories (AP, BC, BD, BR, GTF, HB, and SH) are used as the known classes, whereas ST, TC, and VH are regarded as unknown classes. ST, TC, and VH are then sequentially introduced as newly annotated classes in Tasks 2, 3, and 4, respectively. This protocol evaluates the model’s ability to discover unknown objects, incrementally learn novel categories, and retain knowledge of previously learned classes. Furthermore, to systematically evaluate the model’s performance, we conduct additional experiments on DIOR under the 15+5 and 19+1 settings following the protocols commonly adopted in generic-domain OWOD methods [3, 32, 12, 13]. For clarity, the novel classes are highlighted with a gray background in Table IV.

## B. Evaluation Metrics

We follow the evaluation protocol established in [3, 32, 12]. For known classes, we report mean average precision (mAP) to evaluate the overall detection performance. In the incremental learning setting, mAP is reported separately for known classes and novel classes. For unknown classes, we adopt unknown recall (U-Recall) as the primary metric for evaluating the model’s ability to retrieve unknown objects. We further report wilderness impact (WI) [60], which measures the influence of unknown objects on the precision of known class detection, and absolute open-set error (A-OSE) [61], which counts the number of unknown objects misclassified as known classes.

## C. Implementation Details

We build our model upon RandBox [13], using ResNet-50 [62] with a Feature Pyramid Network (FPN) [63] as the backbone. All experiments are conducted on a single NVIDIA GeForce RTX 5090 GPU with a batch size of 8. The model is optimized using AdamW with a base learning rate of $3 \times 1 0 ^ { - 5 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . A warmup schedule is applied over the first 33 iterations. The learning rate starts at $3 \times 1 0 ^ { - 7 }$ , corresponding to a warmup factor of 0.01, and linearly increases to the base learning rate of $3 \times 1 0 ^ { - 5 }$ . The base training stage is conducted for 30 epochs on NWPU VHR-10, DIOR, and DOTA. Each incremental learning task is also trained for 30 epochs.

## D. Quantitative Results

Open-world object detection. We compare our method with other state-of-the-art methods on DIOR,

TABLE II  
OPEN-WORLD OBJECT DETECTION PERFORMANCE ON DIOR (TOP), DOTA (MIDDLE), AND NWPU VHR-10 (BOTTOM) UNDER VARIOUS SETTINGS. THE BEST RESULTS WITHIN EACH SETTING ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Task IDs (→)</td><td colspan="2">Task 1 (Base Stage)</td><td colspan="3">Task 2 (Incremental Stage)</td></tr><tr><td>U-Recall (↑)</td><td>mAP (↑)</td><td>Previously known</td><td>mAP (↑) Currently known</td><td>Both</td></tr><tr><td colspan="6">DIOR (16 + 4 Setting)</td></tr><tr><td></td><td>14.18</td><td>50.62</td><td>46.30</td><td>67.11</td><td>50.46</td></tr><tr><td>ORE [3]</td><td>18.53</td><td>51.58</td><td>49.71</td><td>67.46</td><td>53.26</td></tr><tr><td>OW-DETR [30]</td><td>23.37</td><td>53.90</td><td>51.61</td><td>71.01</td><td>55.49</td></tr><tr><td>PROB [32]</td><td>27.59</td><td>55.88</td><td>52.96</td><td>70.90</td><td>56.55</td></tr><tr><td>KTCN [64] SGROD [65]</td><td>35.82</td><td>55.68</td><td>52.22</td><td>71.22</td><td>56.02</td></tr><tr><td>Hu et al. [8]</td><td>42.93</td><td>57.69</td><td>54.23</td><td>75.47</td><td>58.48</td></tr><tr><td>Base model</td><td>39.07</td><td>69.15</td><td>59.51</td><td>63.74</td><td>60.36</td></tr><tr><td>Ours</td><td>52.93</td><td>71.26</td><td>68.59</td><td>72.57</td><td>69.39</td></tr><tr><td colspan="6"></td></tr><tr><td>DIOR (10 + 10 Setting)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ORE [3]</td><td>21.41 28.01</td><td>54.62 56.58</td><td>49.61 49.54</td><td>53.11 55.20</td><td>51.36</td></tr><tr><td>OW-DETR [30]</td><td>27.77</td><td>57.90</td><td>54.21</td><td>55.01</td><td>52.37 54.61</td></tr><tr><td>PROB [32]</td><td>33.10</td><td>59.22</td><td>56.12</td><td>58.12</td><td>57.12</td></tr><tr><td>KTCN [64]</td><td>39.25</td><td>59.17</td><td>55.02</td><td>61.11</td><td>58.07</td></tr><tr><td>SGROD [65] Hu et al. [8]</td><td>46.33</td><td>62.51</td><td>57.09</td><td>63.21</td><td>60.15</td></tr><tr><td>Base model</td><td>34.56</td><td>71.13</td><td>58.31</td><td>60.69</td><td>59.50</td></tr><tr><td>Ours</td><td>53.72</td><td>73.49</td><td>61.14</td><td>65.97</td><td>63.56</td></tr><tr><td colspan="6">DIOR (4 + 16 Setting)</td></tr><tr><td>ORE [3]</td><td>16.77</td><td>58.90</td><td>59.04</td><td>53.66</td><td>54.74</td></tr><tr><td>OW-DETR [30]</td><td>21.08</td><td>64.62</td><td>62.44</td><td>54.60</td><td>56.17</td></tr><tr><td>PROB [32]</td><td>19.57</td><td>67.10</td><td>61.91</td><td>51.17</td><td>53.32</td></tr><tr><td>KTCN [64]</td><td>23.67</td><td>65.34</td><td>59.76</td><td>57.89</td><td>58.26</td></tr><tr><td>SGROD [65]</td><td>31.77</td><td>69.18</td><td>63.17</td><td>56.92</td><td>58.17</td></tr><tr><td>Hu et al. [8]</td><td>41.61</td><td>68.79</td><td>67.39</td><td>59.26</td><td>60.89</td></tr><tr><td>Base model</td><td>23.21</td><td>73.10</td><td>63.50</td><td>58.83</td><td>59.76</td></tr><tr><td>Ours</td><td>40.11</td><td>79.48</td><td>70.84</td><td>60.98</td><td>62.96</td></tr><tr><td colspan="6">DOTA (8 + 8 Setting)</td></tr><tr><td>ORE [3]</td><td>15.01</td><td>41.92</td><td>39.72</td><td>56.11</td><td>47.92</td></tr><tr><td>OW-DETR [30]</td><td>13.79</td><td>43.55</td><td>39.23</td><td>57.01</td><td>48.12</td></tr><tr><td>PROB [32]</td><td>19.05</td><td>43.09</td><td>40.06</td><td>59.01</td><td>49.54</td></tr><tr><td>KTCN [64]</td><td>25.02</td><td>46.80</td><td>44.99</td><td>58.60</td><td>51.80</td></tr><tr><td>SGROD [65]</td><td>31.90</td><td>45.11</td><td>43.69</td><td>59.96</td><td>51.82</td></tr><tr><td>Hu et al. [8]</td><td>36.44</td><td>47.99</td><td>45.84</td><td>61.24</td><td>53.54</td></tr><tr><td>Base model</td><td>28.35</td><td>45.53</td><td>41.63</td><td>49.99</td><td>45.81</td></tr><tr><td>Ours</td><td>36.85</td><td>49.36</td><td>47.72</td><td>58.47</td><td>53.10</td></tr><tr><td colspan="6">NWPU VHR-10 (8 + 2 Setting)</td></tr><tr><td>ORE [3]</td><td>27.18</td><td>83.19</td><td>81.30</td><td>81.40</td><td>81.32</td></tr><tr><td>OW-DETR [30]</td><td>26.52</td><td>83.58</td><td>82.15</td><td>78.70</td><td>81.46</td></tr><tr><td>PROB [32]</td><td>38.49</td><td>84.16</td><td>84.21</td><td>78.01</td><td>82.97</td></tr><tr><td>KTCN [64]</td><td>36.74</td><td>83.78</td><td>86.50</td><td>80.90</td><td>85.38</td></tr><tr><td>SGROD [65]</td><td>46.82</td><td>82.68</td><td>83.72</td><td>81.02</td><td>83.18</td></tr><tr><td>Hu et al. [8]</td><td>55.90</td><td>86.03</td><td>86.23</td><td>85.73</td><td>86.13</td></tr><tr><td>Base model</td><td>45.18</td><td>91.94</td><td>91.47</td><td>76.85</td><td>88.54</td></tr><tr><td>Ours</td><td>58.94</td><td>89.06</td><td>91.57</td><td>84.86</td><td>90.23</td></tr></table>

DOTA, and NWPU VHR-10 datasets under different incremental settings and summarize the results in Table II. Following the experimental protocol of Hu et al. [8], we report U-Recall and mAP after base training, together with the mAP of previously known, currently known, and overall performance after incremental learning. For unknown object detection, our method achieves the best performance in four of the five settings, surpassing the previous best results by 10.00, 7.39, 0.41, and 3.04 percentage points in DIOR 16+4, DIOR 10+10, DOTA 8+8, and NWPU VHR-10 8+2, respectively. In terms of known object detection, our method outperforms the competing methods in most settings, achieving the highest mAP after base training in four of the five settings. After incremental learning, our method achieves the best performance on previously known classes across all five settings, with improvements of 14.36, 4.05, 3.45, 1.88, and 5.07 percentage points, respectively. Moreover, our method obtains the highest overall mAP in four of the five settings, outperforming the previous best results by 10.91, 3.41, 2.07, and 4.10 percentage points in DIOR 16+4, DIOR 10+10, DIOR 4+16, and NWPU VHR-10 8+2, respectively, further demonstrating the superiority of our method in maintaining previously learned knowledge while achieving strong overall detection performance after incremental learning.

TABLE III  
EVALUATION OF UNKNOWN OBJECT CONFUSION ON NWPU VHR-10.
<table><tr><td>Task IDs (→)</td><td colspan="4">Task 1</td><td colspan="4">Task 2</td><td colspan="4">Task 3</td><td>Task 4</td></tr><tr><td>Method</td><td>mAP</td><td>U-Recall (1)</td><td>WI (4)</td><td>A-OSE (↓)</td><td>mAP</td><td>U-Recall (↑)</td><td>WI (↓)</td><td>A-OSE (↓)</td><td>mAP</td><td>U-Recall (↑)</td><td>WI (4)</td><td>A-OSE (4)</td><td>mAP (1)</td></tr><tr><td>ORE [3]</td><td>88.97</td><td>27.33</td><td>0.000000</td><td>2</td><td>89.74</td><td>30.18</td><td>0.004090</td><td>3</td><td>87.40</td><td>34.23</td><td>0.006612</td><td>15</td><td>87.22</td></tr><tr><td>OW-DETR [30]</td><td>85.95</td><td>24.07</td><td>0.000000</td><td>9</td><td>85.39</td><td>25.76</td><td>0.000129</td><td>8</td><td>85.60</td><td>27.34</td><td>0.001691</td><td>7</td><td>82.98</td></tr><tr><td>PROB [32]</td><td>86.60</td><td>36.21</td><td>0.000000</td><td>0</td><td>82.49</td><td>31.33</td><td>0.000000</td><td>0</td><td>78.24</td><td>29.68</td><td>0.000000</td><td>0</td><td>84.88</td></tr><tr><td>KTCN [64]</td><td>82.62</td><td>33.34</td><td>0.000234</td><td>12</td><td>78.27</td><td>37.94</td><td>0.000311</td><td>15</td><td>77.16</td><td>36.06</td><td>0.000707</td><td>3</td><td>82.48</td></tr><tr><td>SGROD [65]</td><td>80.48</td><td>35.25</td><td>0.000000</td><td>0</td><td>84.24</td><td>40.46</td><td>0.000000</td><td>0</td><td>81.33</td><td>39.35</td><td>0.000000</td><td>0</td><td>83.90</td></tr><tr><td>Base model</td><td>90.48</td><td>38.39</td><td>0.000000</td><td>1</td><td>83.76</td><td>45.54</td><td>0.000000</td><td>13</td><td>84.56</td><td>51.97</td><td>0.000000</td><td>2</td><td>84.04</td></tr><tr><td>Ours</td><td>93.74</td><td>55.28</td><td>0.000578</td><td>2</td><td>94.46</td><td>60.57</td><td>0.000214</td><td>2</td><td>93.63</td><td>63.01</td><td>0.000000</td><td>0</td><td>92.39</td></tr></table>

To further evaluate the confusion between unknown and known objects, we conduct additional experiments on NWPU VHR-10 using WI and A-OSE as evaluation metrics, with the results reported in Table III. Our method consistently achieves the highest U-Recall across all three open-world stages, outperforming the best competing methods by 19.07, 20.11, and 23.66 percentage points in Tasks 1–3, respectively. Meanwhile, the WI remains at a very low level, with values of only 0.000578 and 0.000214 in Tasks 1 and 2, and further decreases to zero in Task 3. A similar trend can be observed for A-OSE, where only two unknown objects are misclassified as known classes in Tasks 1 and 2, while no such confusion occurs in Task 3. These results show that the substantial gains in unknown-object recall are achieved while maintaining low absolute WI and A-OSE values.

Incremental object detection. To further evaluate the class-incremental detection performance of our method, we compare it with existing OWOD approaches following the evaluation protocols adopted in [3, 30, 32]. Table IV reports the results on DIOR under two class-incremental settings. Although the RandBox baseline already achieves competitive performance, our method further improves the overall mAP by 1.7 and 4.9 percentage points in the two settings, respectively. These improvements demonstrate the effectiveness of the proposed method in learning novel classes while maintaining the performance of previously learned classes.

## E. Qualitative Results

Fig. 4 presents qualitative comparisons between our method and the RandBox baseline on NWPU VHR-10. Compared with the baseline, our method discovers more unknown objects while maintaining accurate detection of known classes. In the left example, our method identifies more tennis courts as unknown while correctly detecting the known basketball court and baseball diamonds, demonstrating improved unknownobject recall without compromising known-class detection. In the middle example, both vehicles are correctly identified as unknown with higher confidence scores, while the false-positive prediction produced by the baseline is suppressed. Similarly, in the right example, our method identifies individual tennis courts as unknown with more accurate localization while preserving reliable predictions for the known baseball diamonds and basketball courts. These qualitative observations are in line with the improvements in U-Recall, WI, and A-OSE reported in Table III, showing that our method improves unknown-object discovery while reducing the misclassification of unknown objects as known classes. Similar improvements are observed on DIOR, as illustrated in Fig. 5. Our method detects more unknown vehicles and windmills that are missed by the baseline while maintaining reliable detection of known objects and suppressing falsepositive unknown predictions. These results further demonstrate the effectiveness of our method across different remote sensing datasets and object categories.

## F. Ablation Study

In this subsection, we conduct ablation studies to analyze the contribution of each proposed component to the performance of HyRS-OWOD. All ablation experiments are conducted on NWPU VHR-10 under the 8+2 known–unknown class split described in Sec. V-A4.

Effectiveness of DOL and HUL. We first evaluate the contributions of DOL and HUL to unknown object discovery. As shown in the upper part of Table V, introducing DOL alone increases U-Recall from 45.2% to 48.1%, indicating that decoupling class-agnostic objectness from semantic classification helps retrieve potential unknown objects. Incorporating HUL alone produces a larger improvement in U-Recall, from airplanebaseball diamondbasketball courtbridgevehicle

![](images/5046d406c27c383e538b2ad094bfce0f5ee01f5392e6703da594c10426afbaea.jpg)  
Fig. 4. Qualitative comparison on the NWPU VHR-10 dataset. The first row shows the ground-truth annotations (green), while the second and third rows show the predictions of the baseline and our method, respectively. Orange and magenta boxes denote known and unknown objects, respectively.

![](images/a3c42029f5917da03e9b9f59323fc2781eca570a41b2a9f0105ad901d0cbf3db.jpg)  
Fig. 5. Qualitative comparison on the DIOR dataset. The first row shows the ground-truth annotations (green), while the second and third rows show the predictions of the baseline and our method, respectively. Orange and magenta boxes denote known and unknown objects, respectively.

TABLE IV  
INCREMENTAL OBJECT DETECTION RESULTS ON DIOR.
<table><tr><td>15 + 5 setting</td><td>AP</td><td>AT</td><td>BF</td><td>BC</td><td>BR</td><td>CM</td><td>DM</td><td>EA</td><td>ES</td><td>GF</td><td>GTF</td><td>HB OP</td><td>SH</td><td>SD</td><td>ST</td><td>TC</td><td>TS</td><td></td><td>VH</td><td>WM mAP</td></tr><tr><td>ORE [3]</td><td>55.5</td><td>64.7</td><td>64.6</td><td>83.7</td><td>24.8</td><td>75.8</td><td>42.6</td><td>53.1</td><td>43.6</td><td>71.9</td><td>64.4</td><td>46.9 45.5</td><td>35.0</td><td>53.5</td><td>43.7</td><td>77.9</td><td>35.9</td><td>52.0</td><td>57.4</td><td>54.6</td></tr><tr><td>OW-DETR [30]</td><td>71.6</td><td>59.6</td><td>53.1</td><td>56.4</td><td>26.8</td><td>71.6</td><td>46.1</td><td>44.0</td><td>38.7</td><td>73.3 41.9</td><td>36.6</td><td>39.6</td><td>32.1</td><td>58.3</td><td>53.7</td><td>78.4</td><td>47.4</td><td>41.3</td><td>74.1</td><td>52.2</td></tr><tr><td>PROB [32]</td><td>81.7</td><td>63.2</td><td>83.2</td><td>61.8</td><td>28.0</td><td>81.4</td><td>48.1</td><td>45.4</td><td>49.5</td><td>66.1 56.9</td><td>37.0</td><td>42.6</td><td>28.0</td><td>71.1</td><td>63.0</td><td>74.4</td><td>38.9</td><td>43.0</td><td>47.4</td><td>55.5</td></tr><tr><td>KTCN [64]</td><td>72.5</td><td>56.4</td><td>70.2</td><td>80.9</td><td>27.2</td><td>76.1</td><td>45.3</td><td>51.8</td><td>50.6</td><td>70.5 66.1</td><td>44.8</td><td>45.8</td><td>70.7</td><td>50.7</td><td>60.3</td><td>82.3</td><td>25.3</td><td>46.8</td><td>68.8</td><td>58.1</td></tr><tr><td>SGROD [65]</td><td>85.9</td><td>64.9</td><td>85.6</td><td>55.4</td><td>14.1</td><td>87.3</td><td>51.4</td><td>33.1</td><td>39.0</td><td>70.6 59.3</td><td>3.7</td><td>47.7</td><td>30.9</td><td>66.4</td><td>76.1</td><td>85.5</td><td>46.1</td><td>55.2</td><td>67.4</td><td>56.3</td></tr><tr><td>Base model</td><td>89.5</td><td>61.9</td><td>83.7</td><td>67.3</td><td>27.0</td><td>86.2</td><td>47.2</td><td>45.8</td><td>51.3</td><td>65.7 58.5</td><td>44.4</td><td>47.7</td><td>11.2</td><td>74.6</td><td>79.2</td><td>87.4</td><td>42.9</td><td>70.3</td><td>67.8</td><td>60.5</td></tr><tr><td>Ours</td><td>89.5</td><td>68.9</td><td>88.4</td><td>71.0</td><td>29.3</td><td>86.4</td><td>50.4</td><td>58.1</td><td>53.9</td><td>72.1 57.3</td><td>40.9</td><td>44.3</td><td>22.1</td><td>74.1</td><td>76.7</td><td>86.6</td><td>37.1</td><td>66.4</td><td>70.4</td><td>62.2</td></tr><tr><td>19 + 1 setting</td><td>AP</td><td>AT</td><td>BF</td><td>BC</td><td>BR</td><td>CM</td><td>DM</td><td>EA</td><td>ES</td><td>GF</td><td>GTF HB</td><td>OP</td><td>SH</td><td>SD</td><td>ST</td><td>TC</td><td>TS</td><td>VH</td><td>WM</td><td>mAP</td></tr><tr><td>ORE [3]</td><td>55.1</td><td>63.4</td><td>67.2</td><td>86.2</td><td>29.0</td><td>79.2</td><td>41.8</td><td>53.1</td><td>46.9</td><td>68.4 66.5</td><td>49.3</td><td>50.6</td><td>67.3</td><td>68.9</td><td>55.5</td><td>80.0</td><td>45.1</td><td>41.1</td><td>59.8</td><td>60.7</td></tr><tr><td>OW-DETR [30]</td><td>64.0</td><td>60.5</td><td>65.9</td><td>78.9</td><td>31.2</td><td>73.9</td><td>41.8</td><td>50.6</td><td>44.3</td><td>67.2 48.9</td><td>28.3</td><td>48.4</td><td>68.7</td><td>29.4</td><td>49.8</td><td>76.4</td><td>44.2</td><td>38.4</td><td>73.8</td><td>54.2</td></tr><tr><td>PROB [32]</td><td>83.2</td><td>55.2</td><td>86.3</td><td>70.5</td><td>28.2</td><td>81.3</td><td>34.8</td><td>43.6</td><td>52.7</td><td>51.6 55.9</td><td>12.7</td><td>41.3</td><td>52.1</td><td>64.9</td><td>72.2</td><td>82.9</td><td>40.1</td><td>49.1</td><td>45.0</td><td>55.2</td></tr><tr><td>KTCN [64]</td><td>77.7</td><td>57.5</td><td>70.8</td><td>85.5</td><td>28.5</td><td>75.0</td><td>45.6</td><td>52.9</td><td>52.9</td><td>67.9 66.2</td><td>47.5</td><td>49.6</td><td>70.2</td><td>62.1</td><td>60.1</td><td>80.9</td><td>33.3</td><td>39.1</td><td>68.0</td><td>59.6</td></tr><tr><td>SGROD [65]</td><td>89.4</td><td>51.0</td><td>90.4</td><td>74.6</td><td>22.6</td><td>87.9</td><td>38.7</td><td>59.0</td><td>53.2</td><td>61.3 56.8</td><td>13.2</td><td>48.0</td><td>29.8</td><td>71.3</td><td>75.3</td><td>84.2</td><td>42.0</td><td>51.6</td><td>63.0</td><td>58.2</td></tr><tr><td>Base model</td><td>79.2</td><td>66.6</td><td>87.6</td><td>78.0</td><td>35.7</td><td>88.2</td><td>51.0</td><td>62.7</td><td>55.5</td><td>76.1 56.8</td><td>55.4</td><td>55.0</td><td>11.5</td><td>86.1</td><td>75.3</td><td>85.8</td><td>47.4</td><td>45.5</td><td>76.6</td><td>63.8</td></tr><tr><td>Ours</td><td>90.5</td><td>75.4</td><td>92.5</td><td>79.8</td><td>39.0</td><td>84.4</td><td>50.6</td><td>65.8</td><td>56.8</td><td>78.5</td><td>68.7 46.7</td><td>57.8</td><td>68.1</td><td>77.4</td><td>78.5</td><td>88.5</td><td>39.1</td><td>58.8</td><td>76.9</td><td>68.7</td></tr></table>

TABLE V  
ABLATION STUDY OF THE PROPOSED COMPONENTS ON NWPU VHR-10 UNDER THE 8+2 SETTING.
<table><tr><td>Task 2</td><td>DOL HUL</td><td>U-Recall K-mAP (↑)</td><td>(↑)</td><td>WI (↓)</td><td>A-OSE (↓)</td></tr><tr><td rowspan="2">Base model</td><td>一</td><td>45.2</td><td>91.9</td><td>0.0002</td><td>8</td></tr><tr><td>V X V</td><td>48.1</td><td>90.2</td><td>0.0001</td><td>10</td></tr><tr><td></td><td>X V</td><td>51.2</td><td>89</td><td>0</td><td>0</td></tr><tr><td>Ours</td><td>V</td><td>58.9</td><td>89.1</td><td>0.0000</td><td>1</td></tr></table>

<table><tr><td rowspan="2">Task 2</td><td rowspan="2">HML</td><td colspan="2">Previously Currently</td><td rowspan="2">Both</td></tr><tr><td>known</td><td>known</td></tr><tr><td>Base model</td><td></td><td>91.5</td><td>76.9</td><td>88.5</td></tr><tr><td>Ours</td><td>V</td><td>91.6</td><td>91.2</td><td>91.5</td></tr></table>

![](images/d6470df382737e0853c53fe7fa94a48bab3c632d4cbe84cc3b18f33f82acb0c6.jpg)

![](images/8f8c0bdd46191e4e2a669021ed1fec102434d13ee289fa0a08ffe4db8276d374.jpg)

ground track fieldharborshipstorage tanktennis court

Fig. 6. The t-SNE visualization of proposal embeddings learned without (a) and with (b) the HML module.

45.2% to 51.2%, while reducing both WI and A-OSE to zero. This result demonstrates the effectiveness of the hyperbolic radius as an uncertainty cue for known–unknown discrimination. When DOL and HUL are combined, U-Recall further increases to 58.9%, outperforming the baseline by 13.7 percentage points. Meanwhile, WI remains zero and A-OSE decreases from 8 to 1. Although K-mAP decreases moderately from 91.9% to 89.1%, the combined model achieves a substantially better balance between unknown-object discovery and known-class detection. These results also indicate that DOL and HUL are complementary: DOL improves class-agnostic foreground discovery, whereas HUL facilitates the subsequent discrimination between known and unknown objects.

Effectiveness of HML. We further investigate the effectiveness of HML during incremental learning. As reported in the lower part of Table V, adding HML improves the mAP of previously known classes from 91.5% to 91.6%, indicating that the model preserves the knowledge acquired in the preceding stage. More notably, the mAP of the currently introduced classes increases from 76.9% to 91.2%, corresponding to an improvement of 14.3 percentage points. Consequently, the overall mAP increases from 88.5% to 91.5%. These results suggest that HML substantially improves the learning of newly introduced classes without degrading the performance of previously known classes, thereby achieving a better balance between stability and plasticity during incremental learning.

Visualization and Analysis.

To further analyze the effect of HML on representation learning, we visualize the proposal embeddings learned with and without HML using t-SNE. As shown in Fig. 6, without HML, embeddings from different categories exhibit substantial overlap and relatively dispersed intra-class distributions. In contrast, incorporating HML produces more compact intra-class clusters and clearer separation between different classes. This visualization demonstrates that HML improves the discriminability of proposal representations, supporting its role in reducing interference between previously learned and novel classes during incremental learning.

## VI. CONCLUSION

This work investigated hyperbolic geometry for OWOD in remote sensing imagery. We proposed HyRS-OWOD, which employs DOL to disentangle class-specific information from objectness. To discriminate unknown objects from known classes, we presented the HUL strategy based on the hyperbolic radius of proposal embeddings. To mitigate the forgetting effect in incremental learning, we introduced HML, which encourages the model to learn discriminative and stable embeddings by improving intra-class compactness and inter-class separability. Extensive experiments on multiple remote sensing benchmarks demonstrated that HyRS-OWOD improves unknownobject recall, reduces known–unknown confusion, and effectively retains previously acquired knowledge during incremental learning. These results validate the potential of hyperbolic representation learning for openworld object detection in remote sensing imagery.

## REFERENCES

[1] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS journal of photogrammetry and remote sensing, vol. 159, pp. 296–307, 2020.

[2] X. Zhang, T. Zhang, G. Wang, P. Zhu, X. Tang, X. Jia, and L. Jiao, “Remote sensing object detection meets deep learning: A metareview of challenges and advances,” IEEE Geoscience and Remote Sensing Magazine, vol. 11, no. 4, pp. 8– 44, 2023.

[3] K. Joseph, S. Khan, F. S. Khan, and V. N. Balasubramanian, “Towards open world object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 5830–5840.

[4] Y. Li, Y. Wang, W. Wang, D. Lin, B. Li, and K.- H. Yap, “Open world object detection: A survey,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 2, pp. 988–1008, 2024.

[5] C. Lang, G. Cheng, J. Wu, Z. Li, X. Xie, J. Li, and J. Han, “Toward open-world remote sensing imagery interpretation: Past, present, and future,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 4, pp. 8–44, 2024.

[6] L. Fang, Z. Yang, T. Ma, J. Yue, W. Xie, P. Ghamisi, and J. Li, “Open-world recognition in

remote sensing: Concepts, challenges, and opportunities,” IEEE Geoscience and Remote Sensing Magazine, vol. 12, no. 2, pp. 8–31, 2024.

[7] Z. Tan, Z. Jiang, Z. Yuan, and H. Zhang, “Opodet: Toward open world potential oriented object detection in remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[8] M. Hu, W. Yin, W. Diao, X. Gao, and X. Sun, “Unveiling the unknown: A sam guided open world object detection method for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–15, 2026.

[9] A. Gepperth and B. Hammer, “Incremental learning algorithms and applications,” in European symposium on artificial neural networks (ESANN), 2016.

[10] V. Khrulkov, L. Mirvakhabova, E. Ustinova, I. Oseledets, and V. Lempitsky, “Hyperbolic image embeddings,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 6417–6427.

[11] J. Hong, P. Fang, W. Li, J. Han, L. Petersson, and M. Harandi, “Curved geometric networks for visual anomaly recognition,” IEEE transactions on neural networks and learning systems, vol. 35, no. 12, pp. 17 921–17 934, 2023.

[12] Z. Sun, J. Li, and Y. Mu, “Exploring orthogonality in open world object detection,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 17 302– 17 312.

[13] Y. Wang, Z. Yue, X.-S. Hua, and H. Zhang, “Random boxes are open-world object detectors,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 6233– 6243.

[14] M. A. Pourhoseingholi, A. R. Baghestani, and M. Vahedi, “How to control confounding effects by statistical analysis,” Gastroenterology and Hepatology From Bed to Bench, vol. 5, pp. 79 – 83, 2012.

[15] B. Chen, W. Peng, X. Cao, and J. Roning,¨ “Hyperbolic uncertainty aware semantic segmentation,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 2, pp. 1275–1290, 2023.

[16] L. Vinh Tran, Y. Tay, S. Zhang, G. Cong, and X. li, “Hyperml: A boosting metric learning approach in hyperbolic space for recommender systems,” in Proceedings of the 13th International Conference on Web Search and Data Mining, 01 2020, pp. 609–617.

[17] Y. Cui, Z. Yu, W. Peng, Q. Tian, and L. Liu, “Rethinking few-shot class-incremental learning with open-set hypothesis in hyperbolic geometry,” IEEE Transactions on Multimedia, vol. 26, pp. 5897–5910, 2023.

[18] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, realtime object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 779–788.

[19] J. Redmon and A. Farhadi, “Yolo9000: better, faster, stronger,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 7263–7271.

[20] ——, “Yolov3: An incremental improvement,” arXiv preprint arXiv:1804.02767, 2018.

[21] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Optimal speed and accuracy of object detection,” arXiv preprint arXiv:2004.10934, 2020.

[22] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 580–587.

[23] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” Advances in Neural Information Processing Systems, vol. 28, 2015.

[24] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Har-´ iharan, and S. Belongie, “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2117–2125.

[25] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision. Springer, 2020, pp. 213–229.

[26] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable detr: Deformable transformers for end-to-end object detection,” arXiv preprint arXiv:2010.04159, 2020.

[27] W. Hua and Q. Chen, “A survey of small object detection based on deep learning in aerial images,” Artificial Intelligence Review, vol. 58, no. 6, p. 162, 2025.

[28] L. Wen, Y. Cheng, Y. Fang, and X. Li, “A comprehensive survey of oriented object detection in remote sensing images,” Expert Systems with Applications, vol. 224, p. 119960, 2023.

[29] J. Pan, Y. Liu, Y. Fu, M. Ma, J. Li, D. P. Paudel, L. Van Gool, and X. Huang, “Locate anything on earth: Advancing open-vocabulary object detection for remote sensing community,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, 2025, pp. 6281– 6289.

[30] A. Gupta, S. Narayan, K. J. Joseph, S. Khan, F. S. Khan, and M. Shah, “Ow-detr: Open-world detection transformer,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9225–9234.

[31] S. Ma, Y. Wang, Y. Wei, J. Fan, T. H. Li, H. Liu, and F. Lv, “Cat: Localization and identification cascade detection transformer for open-world object detection,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 19 681–19 690.

[32] O. Zohar, K.-C. Wang, and S. Yeung, “Prob: Probabilistic objectness for open world object detection,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 11 444–11 453.

[33] O. Zohar, A. Lozano, S. Goel, S. Yeung, and K.-C. Wang, “Open world object detection in the era of foundation models,” ArXiv, vol. abs/2312.05745, 2023.

[34] Y. He, W. Chen, S. Wang, T. Liu, and M. Wang, “Recalling unknowns without losing precision: An effective solution to large model-guided open world object detection,” IEEE Transactions on Image Processing, vol. 34, pp. 729–742, 2025.

[35] X. Xi, Y. Huang, Z. Zhong, and R. Luo, “Umb: Understanding model behavior for open-world object detection,” in Advances in Neural Information Processing Systems, vol. 37. Curran Associates, Inc., 2024, pp. 74 233–74 261.

[36] X. Xi, X. Fu, W. Wang, and R. Luo, “OW-VAP: Visual attribute parsing for open world object detection,” in Forty-second International Conference on Machine Learning, 2025.

[37] Z. Li, Z. Xiang, J. West, and K. Khoshelham, “From open vocabulary to open world: Teaching vision language models to detect novel objects,” 2026.

[38] N. Saini, A. Dubey, D. Das, and C. Chattopadhyay, “Advancing open-set object detection in remote sensing using multimodal large language model,” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 451–458.

[39] P. Mettes, M. Ghadimi Atigh, M. Keller-Ressel, J. Gu, and S. Yeung, “Hyperbolic deep learning in computer vision: A survey,” International Journal of Computer Vision, vol. 132, no. 9, pp. 3484–3508, Sep. 2024.

[40] W. Peng, T. Varanka, A. Mostafa, H. Shi, and G. Zhao, “Hyperbolic deep neural networks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 12, pp. 10 023–10 044, 2022.

[41] O.-E. Ganea, G. Becigneul, and T. Hofmann,´ “Hyperbolic neural networks,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc., 2018, p. 5350–5360.

[42] Y. Wang, H. Wang, W. Lu, and Y. Yan, “Hygge: Hyperbolic graph attention network for reasoning over knowledge graphs,” Information Sciences,

vol. 630, pp. 190–205, 2023.

[43] I. Chami, Z. Ying, C. Re, and J. Leskovec, “Hy-´ perbolic graph convolutional neural networks,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019.

[44] Z. Gao, C. Xu, F. Li, Y. Jia, M. Harandi, and Y. Wu, “Exploring data geometry for continual learning,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 24 325–24 334.

[45] B. Zhang, H. Jiang, S. Feng, X. Li, Y. Ye, and R. Ye, “Hyperbolic knowledge transfer with class hierarchy for few-shot learning,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, 7 2022, pp. 3723–3729, main Track.

[46] Y. Guo, X. Wang, Y. Chen, and S. X. Yu, “Clipped hyperbolic classifiers are superhyperbolic classifiers,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2022, pp. 1–10.

[47] B. Chen, W. Peng, X. Cao, and J. Roning,¨ “Hyperbolic uncertainty aware semantic segmentation,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 2, pp. 1275–1290, 2024.

[48] M. G. Atigh, J. Schoep, E. Acar, N. Van Noord, and P. Mettes, “Hyperbolic image segmentation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 4443–4452.

[49] C. Lang, A. Braun, L. Schillingmann, and A. Valada, “On hyperbolic embeddings in object detection,” in Pattern Recognition. Cham: Springer International Publishing, 2022, pp. 462–476.

[50] F. Kong, Y. Chen, J. Cai, and D. Modolo, “Hyperbolic learning with synthetic captions for open-world detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 762–16 771.

[51] T. Doan, X. Li, S. Behpour, W. He, L. Gou, and L. Ren, “Hyp-ow: Exploiting hierarchical structure learning with hyperbolic distance enhances open world object detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, 2024, pp. 1555–1563.

[52] M. Hamzaoui, L. Chapel, M.-T. Pham, and S. Lefevre, “Hyperbolic prototypical network for\` few shot remote sensing scene classification,” Pattern Recognition Letters, vol. 177, pp. 151– 156, 2024.

[53] X. Li, F. Xu, F. Liu, R. Xia, Y. Tong, L. Li, Z. Xu, and X. Lyu, “Hybridizing euclidean and hyperbolic similarities for attentively refining representations in semantic segmentation of remote sensing images,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[54] W. Cui, X. Xu, J. Chen, Z. Feng, H. Zhao, Y. Hao, Y. Tian, J. Wang, and C. Xia, “Representation of multirelations of geographic scenes based on hyperbolic space,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[55] Q. Yang, S. Zhang, J. Li, Y. Sun, Q. Han, and Y. Sun, “Hyperboloid-embedded siamese network for change detection in remote sensing images,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 9240–9252, 2024.

[56] P. Sun, R. Zhang, Y. Jiang, T. Kong, C. Xu, W. Zhan, M. Tomizuka, L. Li, Z. Yuan, C. Wang et al., “Sparse r-cnn: End-to-end object detection with learnable proposals,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14 454–14 463.

[57] S. Chen, P. Sun, Y. Song, and P. Luo, “Diffusiondet: Diffusion model for object detection,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19 773– 19 786.

[58] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for dense object detection,”´ in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[59] R. Girshick, “Fast r-cnn,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 1440–1448.

[60] A. R. Dhamija, M. Gunther, J. Ventura, and T. E.¨ Boult, “The overlooked elephant of object detection: Open set,” in 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 2020, pp. 1010–1019.

[61] D. Miller, L. Nicholson, F. Dayoub, and N. Sunderhauf, “Dropout sampling for robust¨ object detection in open-set conditions,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018, pp. 3243–3249.

[62] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[63] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Har-´ iharan, and S. Belongie, “Feature pyramid networks for object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2117–2125.

[64] X. Xi, Y. Huang, J. Lin, and R. Luo, “Ktcn: enhancing open-world object detection with knowledge transfer and class-awareness neutralization,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, ser. IJCAI ’24, 2024.

[65] Y. He, W. Chen, S. Wang, T. Liu, and M. Wang, “Recalling unknowns without losing precision: An effective solution to large model-guided open

world object detection,” IEEE Transactions on Image Processing, vol. 34, pp. 729–742, 2025.

![](images/a2de310eb8686214f7b373f9aabb2a02563c622a5133bb434ada10097f85fd49.jpg)

Wuzhou Li received the B.Sc., M.Sc., and Ph.D. degrees from Wuhan University, Wuhan, China, in 2014, 2020, and 2024, respectively. She is currently a Lecturer with the School of Computer Science and Artificial Intelligence, Wuhan Textile University, Wuhan, China.

Her research interests include computer vision and deep learning, especially on object detection in remote sensing.

![](images/985a0db971cd78fda0fe77b61f7e746eac9e72ad3dd9debc7a03eebe2a0bf7f2.jpg)

Jiawei Zhou received the B.Sc., M.Sc., and Ph.D. degrees from Wuhan University, Wuhan, China, in 2017, 2021, and 2025, respectively.

His research interests include deep learning, computer vision, and remote sensing image object detection.

![](images/fda02b16fea5395b186847a49da239ca6503798776e164765427ab8b8b59872e.jpg)

Shenghang Wang received the B.S. degree in Electrical Engineering Technology from Northern Kentucky University, Highland Heights, KY, USA, in 2025. He is currently pursuing the M.S. degree with the Electrical and Computer Engineering, Ohio State University, Columbus, OH, USA.

His research interests include computer vision and point cloud processing.

![](images/076c279e249650afadb98c78e6b14e21c031b6d34512816f924d3dd741e39570.jpg)

Xiang Li received the B.Sc. degree from Wuhan University, Wuhan, China, in 2014, and the Ph.D. degree from the Institute of Remote Sensing and Digital Earth, Chinese Academy of Sciences, Beijing, China, in 2019. He is currently a Professor with with the School of Artificial Intelligence, Wuhan University, Wuhan, China.

His research interests include deep learning, computer vision, and remote sensing image interpretation.