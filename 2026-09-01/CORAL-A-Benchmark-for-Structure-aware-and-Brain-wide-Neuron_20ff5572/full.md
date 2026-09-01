# CORAL: A Benchmark for Structure-aware and Brain-wide Neuron Reconstruction in Light Microscopy

Zekang Yang1,2, Jiamin Li2,3,4, Zhenghua Li1,2, Jiaqi Fan2,3,4, Zengcai Guo2,3,4\*, Xiaolin Hu1,2,5 \* 1Department of Computer Science and Technology,

Institute for AI, BNRist, Tsinghua University, Beijing 100084, China 2Tsinghua Laboratory of Brain and Intelligence (THBI),

IDG/McGovern Institute for Brain Research, Tsinghua University, Beijing 100084, China 3School of Basic Medical Sciences, Tsinghua University, Beijing, 100084, China 4Tsinghua-Peking Center for Life Sciences, Beijing, 100084, China 5Chinese Institute for Brain Research (CIBR), Beijing 100010, China {yzk25, li-zh24, jq-fan24} @mails.tsinghua.edu.cn, li-jm20@tsinghua.org.cn, {guozengcai,xlhu} @tsinghua.edu.cn

## Abstract

Automatic neuron reconstruction from light microscopy images is a central problem in computational neuroanatomy. While recent methods have achieved encouraging results on local image blocks, it remains unclear whether such progress translates to reconstruction that is both structurally accurate and scalable to the whole-brain scale. We present CORAL, the first benchmark for structure-aware evaluation of automatic neuron reconstruction from light microscopy images at both local and whole-brain scales. Built on a high-quality whole-brain fMOST dataset with carefully curated annotations, CORAL establishes two progressive tasks: blocklevel reconstruction, which evaluates reconstruction methods under limited spatial context, and brain-wide reconstruction, which assesses complete neuron reconstruction at the whole-brain scale. To account for topological correctness beyond geometric distance similarity, we introduce a structure-aware metric based on fiber prediction. To further achieve complete neuron reconstruction across the entire brain, we develop a brain-wide neuron tracing framework that extends arbitrary local reconstruction methods to the whole-brain scale through an iterative localto-global process. Using this benchmark, we provide the first structure-aware comparison of mainstream methods for local neuron reconstruction and further evaluate their performance in brain-wide reconstruction. Our results underscore the importance of structure-aware evaluation and the need for more robust methods for complete neuron reconstruction. All datasets and code are available at https://github.com/yang-ze-kang/CORAL.

## 1 Introduction

Reconstructing neuronal morphologies from light microscopy images is a fundamental problem in computational neuroanatomy, with broad importance for analyzing neuronal cell types, long-range projections, and large-scale brain connectivity [1, 2, 3]. Over the past decade, advances in image enhancement, 3D segmentation, and tracing algorithms have substantially improved the accuracy and scalability of automatic neuron reconstruction [4, 5, 6]. As a result, modern methods can already achieve promising geometric accuracy on local image blocks.

Despite these advances, a fundamental question remains largely unanswered: how far are current automatic methods from achieving structurally accurate neuron reconstruction at the whole-brain scale? Existing public benchmarks [7, 8] mainly evaluate reconstruction performance on local image blocks and isolated neuronal structures. However, in large-scale fMOST imaging data, many neurons span distances far beyond a single local volume, meaning that only partial neuronal structures are visible within any given local image. As a result, these benchmarks provide limited insight into whole-brain reconstruction performance and thus fail to capture its central challenges, including global topological correctness and long-range continuity preservation.

A second limitation of existing evaluation protocols is that many commonly used metrics are primarily geometric rather than structure-aware. Segmentation metrics, such as Dice-style overlap measures quantify foreground agreement at the voxel level, while geometry-based metrics evaluate spatial position or connectivity consistency between prediction and reference [9, 10]. However, neuron reconstruction is fundamentally a tree structure prediction problem: a reconstruction can still be biologically incorrect even when most points are well aligned, if a small number of connectivity errors alter the branching structure or long-range trajectories of the neuronal tree. In particular, root-to-leaf fibers are often more critical than local geometric similarity for assessing whether a reconstruction preserves the correct neuronal organization. This calls for a benchmark that evaluates not only local geometric fidelity, but also the correctness of global neuronal structure.

To address these limitations, we present CORAL, a benchmark for structure-aware evaluation of automatic neuron reconstruction from light microscopy images at both local and whole-brain scales. CORAL is built on a high-quality whole-brain fMOST dataset with carefully curated neuron annotations and supports evaluation under two progressive settings: block-level reconstruction, which is designed to develop reconstruction algorithms, and brain-wide reconstruction, which evaluates complete neuron reconstruction at the whole-brain scale. To better assess reconstruction quality, we further introduce structure-aware metrics based on keypoint prediction and fber prediction, which explicitly measure the recovery of bifurcation/termination events and root-to-leaf neuronal fibers. In addition, to enable standardized large-scale evaluation of existing reconstruction methods in the brain-wide setting, we develop a whole-brain tracing framework that extends arbitrary local reconstruction algorithms to the whole-brain scale.

Using this benchmark, we conducted a standardized large-scale evaluation of representative reconstruction methods under both block-level and brain-wide settings. Our experiments yield three main findings. First, in two-stage reconstruction methods, the segmentation method and the reconstruction method are coupled, making algorithm optimization more challenging. Second, geometry-based metrics are insufficient to characterize final reconstruction quality: minor local errors can lead to global topological failures, whereas structure-aware keypoint- and fiber-based metrics provide a substantially more comprehensive assessment of reconstruction quality. Third, current methods remain far from achieving structurally correct reconstruction and fully automated brain-wide reconstruction. Even small local structural errors can propagate into large-scale errors across the whole brain, imposing higher requirements on automated reconstruction methods. These findings demonstrated that CORAL provides a rigorous testbed for developing reconstruction methods and evaluating brain-wide neuron reconstruction.

## 2 Related Work

## 2.1 Datasets

Existing public datasets for automated neuron reconstruction are predominantly constructed from small-scale volumetric image blocks. The DIADEM [7] dataset is one of the earliest standardized benchmarks, providing manually annotated neuron reconstructions across several imaging conditions. The BigNeuron [8] project expands the diversity with collections, yet still limited to small-volume image blocks. More recent datasets, including CWMBS [6] and NeuroFly [11], further increase variability in morphology and imaging modalities, but are likewise derived from cropped sub-volumes. The Brain Image Library (BIL) [12] hosts a large collection of whole-brain image datasets and neuron annotations, supporting neuroscientists in tracing neuronal circuits and comprehensively map cell types. However, due to the lack of a standardized evaluation protocol for automatic reconstruction algorithms, these data have not been widely used in the automatic neuron reconstruction community.

To our knowledge, only Liu et al. [6] have evaluated reconstruction performance on four complete neurons selected from BIL. Nevertheless, their study did not establish a standard evaluation protocol.

## 2.2 Methods

Neuron reconstruction has been studied for several decades. Early algorithms primarily relied on conventional computational techniques, such as path pruning [13, 4], graph search [14, 15], active contours [16], and hand-crafted tubular-structure cues [5]. With the advent of deep learning, researchers have increasingly used deep models to segment and enhance neuronal structures before applying traditional reconstruction algorithms, achieving better performance than directly applying reconstruction methods [17, 18]. Subsequent improvements have mainly focused on strengthening segmentation models, including improving foreground prediction accuracy [19, 6] and preserving the continuity of linear structures [20]. In recent years, some researchers have sought to develop deep learning-based reconstruction methods by predicting neuronal orientations [21] or the locations of subsequent tracing points [22].

## 2.3 Metrics

A diverse set of metrics has been proposed to evaluate neuronal reconstruction from multiple perspectives. From a segmentation viewpoint, voxel-wise metrics such as F1-score and Dice coefficient are commonly used to measure the overlap between predicted neuronal regions and ground truth, reflecting foreground classification accuracy. Metrics designed for neuron reconstruction can be categorized into two following groups:

Geometry-based metrics. Peng et al. [23] introduced distance-based metrics that measure the minimum spatial distance between reconstructed and ground-truth structures, including spatial distance (SD) and significant spatial distance (SSD), which are the most commonly used metrics in neuron reconstruction evaluation. Liu et al. [6] treated predicted and ground-truth point pairs whose distance is below a predefined threshold as true positives, and accordingly define precision, recall, F1 score (referred to as Point-F1 for simplicity), and Miss-Extra Scores (MES). Zhang et al. [10] further consider point connectivity by treating an edge pair as a true positive when both its endpoints and length are matched, and propose the Length-F1 metric to evaluate predicted-edge accuracy. Limitations: These metrics assess reconstruction quality only from a local geometric perspective, whereas the correctness of the topological structure is essential and cannot be overlooked. The left panel of Figure 1 illustrates an example showing why structure-aware is essential. The prediction contains the same number of nodes and spatial locations with the ground truth, causing SD, SSD, and Point-F1 to collapse. The prediction traces is routed onto a neighboring fiber and occurs a merging error, leaving only one fiber correctly recovered. Locally, this corresponds to only one incorrectly predicted edge, resulting in Length-F1 having an incorrect high value.

![](images/c8e6e29a0d52926bc91628dc43f3749a2e087c238bf1ba5226a4b29ed64f55f9.jpg)  
Figure 1: Comparison of different metrics. The left uses a representative prediction error to highlight the limitations of geometry-based metrics. The right illustrates the limitations of DIADEM.

Structure-aware metrics. Feng et al. [5] adopted the critical node metric to evaluate the prediction accuracy at key locations (bifurcations and leafs) that affect structural correctness. Tree edit distance (TED) [24] has also been introduced to evaluate neuron reconstruction quality. DIADEM metric [25] further extends TED by checking whether each matched node has a corresponding matched ancestor and whether the difference between their respective path lengths to the ancestor is below a predefined threshold. Limitations: DIADEM has several design limitations that have prevented its widespread adoption: (1) It relies on a complex computation protocol and produces only an abstract score, making it less suitable for inspecting bad cases or guiding algorithm improvement. (2) It relies on a directed tree structure rooted at a known soma location, which makes it unsuitable as a block-level metric, where local image blocks often lack the soma and predicted reconstructions are collections of undirected graphs. (3) As shown on the right side of Figure 1, DIADEM still has a design limitation. In practical tracing, dense neuronal fibers or merging errors may allow a matched node to reach the matched ancestor node through a different path with the same path length, causing DIADEM to spuriously count it as a correct prediction.

![](images/404a0987348dad1ad99940b21de8a464f451aa10b6bd10bbb33b8b87f6a5f534.jpg)

## 3 Dataset Description

## 3.1 Data Curation and Annotation Protocol

Our mouse brain samples underwent a standard tissue preparation protocol for fMOST imaging, including dehydration, resin infiltration, embedding and imaging [26, 27]. The acquired raw image tiles were subjected to a series of preprocessing steps, comprising pixel-wise dual-channel registration, image stitching, artifact removal, and signal enhancement. The in-plane resolution is 0.35μm in the x-y axes, and the axial resolution is 1µm along the z axis. We will also upload our mouse brain images to the Brain Image Library [12].

To facilitate the development and evaluation of automatic reconstruction algorithms, we selected a dataset containing 32 neurons for annotation. For quality control, we established a three-stage annotation protocol involving three independent annotators. First, neuronal fibers were annotated by the first annotator. Next, a different annotator independently reviewed all reconstructed fibers from scratch; any missing branches or incomplete structures required re-annotation. Finally, a third annotator performed global cross-validation. All annotation and verification were performed using Fast Neuron Tracing (FNT) software [28] and were exported in the standardized SWC format [29]. Compared to existing datasets, our dataset is, to the best of our knowledge, the first to achieve near-exhaustive annotation of all visible neurons within an entire brain volume.

![](images/0580b93e18262cc54c960f251f0cef43b746c116911dfadf94e27af773b21487.jpg)

(a) Spatial distribution of 32 neurons.  
(b) Lengths of neurons  
![](images/a6a30a816c6799486a0883885bfaf7f02aa20ad1ffabc2d861ca80890071816a.jpg)

![](images/53b54b8a2ec89490e0dca96de42d1568ec51f536085ace0f9889251d4c3309b8.jpg)  
(c) Distribution of fiber lengths.  
Figure 2: Visualization of dataset statistics. (a) Spatial distribution of the 32 neurons in the mouse brain from multiple views. (b) Total length of each neuron, with blue denoting neurons whose somata are located in the left hemisphere and orange denoting those whose somata are located in the right hemisphere. (c) Length distribution of dendrites (orange) and axons (blue).

## 3.2 Dataset Statistics

Our dataset includes 32 neurons with complete morphological reconstructions, among which 15 have their somata located in the left hemisphere and 17 in the right hemisphere. The neurons are sparselydistributed across the cerebral cortex, as shown in Figure 2a. Figure 2b shows the total length of each reconstructed neuron. The shortest neuron spans 695 μm (1,584 voxels), whereas the longest reaches 110,292 μm (148,842 voxels), with neurons of varying sizes distributed across both the left and right hemispheres. Figure 2c further shows the length distributions of dendrites and axons. In total, the dataset contains 1,421 dendritic fibers and 4,931 axonal fibers. The dendrite lengths range from 33 to 429µm (62–1,067 voxels), whereas axon lengths exhibit a much broader range, from 103 to 9,643 µm (273–22,720 voxels), highlighting the challenge of preserving long-range continuity during axon tracing.

## 4 CORAL Benchmark

Figure 3 provides an overview of CORAL, whose ultimate goal is to evaluate the performance of algorithms in brain-wide neuron reconstruction. Due to the high resolution of brain images, directly importing the whole volume for brain-wide neuron reconstruction is impractical. To address this challenge, we developed a whole-brain tracing framework (WBTF) (Sec 4.1) that can equip arbitrary reconstruction algorithms with the ability to reconstruct brain-wide neurons. CORAL defines two progressive tasks: block-level reconstruction and brain-wide reconstruction (Sec 4.2). The diverse block-level images, together with the WBTF, enable reconstruction algorithms to be developed and trained on block images and then deployed for brain-wide tracing inference. Moreover, we design the Fiber metric, a more comprehensive yet simple and intuitive metric, to evaluate the performance of different algorithms (Sec 4.3).

![](images/b81e6650ee7bc275f4b44a373f5a1393d0e4b715cd4e34e0ad13632184f0cf5e.jpg)  
Figure 3: Overview of the CORAL benchmark. (a) shows the two benchmark tasks. (b) shows the whole-brain tracing framework. (c) shows the two structure-aware metrics (Keypoint and Fiber) where d denotes the distance between matched keypoints and τ is the corresponding distance threshold; The green region indicates the overlapping area; FIoU denotes the fiber intersection-overunion of overlapped lengths between matched fibers, and η is the FIoU threshold for valid matching.

## 4.1 Whole-brain tracing framework

As shown in Figure 3, the proposed whole-brain tracing framework consists of three main components: heuristic search, local tracing, and growing. First, for heuristic search, we partition the high-resolution whole-brain volume into a 3D grid. Each grid position corresponds to a block image centered at that position, and block images associated with neighboring grid positions overlap with each other. When tracing a neuron, the algorithm starts from the grid position containing the soma. It then selects leaf nodes, as well as nodes close to the boundary of the already traced region, from the current reconstruction as candidate search locations. By progressively moving the field of view, the framework expands the reconstruction and obtains an intermediate result over the brain-wide. Second, for local tracing, given the block image corresponding to the current grid point, any neuron reconstruction algorithm can be applied to obtain a local reconstruction result. Third, for growing, the framework compares the local reconstruction in the current field of view with the overlapping part of the existing global reconstruction, and stitches newly traced subtrees from the current view into the global reconstruction. See implementation details in the appendix B.

## 4.2 Task Settings

We use a hemisphere-based partition throughout the benchmark: data from the left hemisphere are used for training, while data from the right hemisphere are reserved for testing. This setting prevents data leakage during testing.

Block-level reconstruction. Considering the diversity of neuronal morphology, sparsity, and signal intensity across the whole brain, we cropped 1,904 diverse blocks from the entire brain volume for training and testing different reconstruction methods. Each block has the size of $3 0 0 \times 3 0 0 \times 3 0 0$ and paired with its corresponding SWC annotation. We assigned 907 left-hemisphere blocks to the training set and 1,021 right-hemisphere blocks to the test set, while holding out eight blocks spanning both hemispheres. Compared with previous block-level datasets [6], our block-level dataset is ten times larger and captures greater diversity in neuronal structures and local imaging environments, making it better suited for training reconstruction algorithms.

Brain-wide reconstruction. In this task, algorithms start from the initial tree at the soma location of each neuron and reconstruct the complete neuron across the brain-wide. Among the 17 neurons in the right hemisphere, three extend into the left hemisphere. To prevent data leakage, we selected the remaining 14 neurons as the test set. Among these 14 neurons, the shortest neuron to be reconstructed has a total length of 1,404 µm (3,560 voxels), whereas the longest reaches 86,680 µm (200,067 voxels). In total, 746 dendrites and 2785 axons are required to be reconstructed. Figure 3 shows the number of axons and dendrites to be reconstructed for each neuron.

## 4.3 Evaluation Metrics

We adopt two structure-aware metrics, Keypoint metric and Fiber metric, to evaluate reconstruction quality. The Keypoint metric measures how accurately an algorithm predicts critical locations. Unlike previous metrics [5], we separately evaluate the reconstruction accuracy of bifurcations and leafs, as they affect reconstruction quality through different patterns.

Inspired by the primary goal of neuron reconstruction, which is to trace all neurites (dendrites and axons) starting from the soma, as Figure 3c shows, we design the Fiber metric to directly measure the reconstruction performance of each neurite. Given the ground truth tree $T _ { g }$ and the predicted tree $T _ { p } ,$ we first decompose them into sets of fibers $\mathcal { F } _ { g } = \{ \bar { f } _ { 1 } , f _ { 2 } , \dots , f _ { m } \}$ and $\mathcal { F } _ { p } = \{ f _ { 1 } ^ { \prime } , f _ { 2 } ^ { \prime } , . . . , f _ { n } ^ { \prime } \}$ where each fiber corresponds to a root-to-leaf path in the tree. Each fiber is a set of continuous edges, let $\mathcal { C } _ { f }$ denotes the midpoints of edges in fiber $f _ { i }$ and $\delta _ { c }$ denotes the length of the edge corresponding to midpoint $c .$ To quantify the geometric similarity between two fibers, we define the fiber overlap as the sum of the lengths of edges whose corresponding midpoints in the ground-truth and predicted fibers are within a distance threshold $\tau { : }$

$$
\mathrm { O v e r l a p } ( f _ { g } , f _ { p } ) = \frac { 1 } { 2 } ( \sum _ { \mathbf { c } \in \mathcal { C } _ { g } } \mathbb { I } \big ( \mathrm { d i s t } ( \mathbf { c } , f _ { p } ) \leq \tau \big ) \cdot \delta _ { c } + \sum _ { \mathbf { c } \in \mathcal { C } _ { p } } \mathbb { I } \big ( \mathrm { d i s t } ( \mathbf { c } , f _ { g } ) \leq \tau \big ) \cdot \delta _ { c } ) ,
$$

The green region in Figure 3c represents the overlap region. We then can define fiber intersectionover-union (FIoU) based on fiber overlap:

$$
\mathrm { F I o U } ( f _ { g } , f _ { p } ) = \frac { \mathrm { O v e r l a p } ( f _ { g } , f _ { p } ) } { \mathrm { L e n g t h } ( f _ { g } ) + \mathrm { L e n g t h } ( f _ { p } ) - \mathrm { O v e r l a p } ( f _ { g } , f _ { p } ) } ,
$$

where $L e n g t h ( f )$ denotes the total length of fiber $f .$ Then we compute the FIoU between each pair of predicted and ground truth fibers, and perform optimal matching based on the FIoU to identify matched fiber pairs M. The unmatched fibers in $\mathcal { F } _ { g }$ and $\mathcal { F } _ { p }$ are considered false negatives (FN) and

false positives (FP), respectively. The fiber pair is considered correctly identified only if its FIoU exceeds the threshold η are considered true positives (TP):

$$
\mathrm { T P } = \sum _ { ( f _ { g } , f _ { p } ) \in \mathcal { M } } \mathbb { I } \big ( \mathrm { F I o U } ( f _ { g } , f _ { p } ) \geq \eta \big ) ,
$$

Conversely, matched fiber pairs with an FIoU below the threshold $\eta$ are also counted as false positives and false negatives. Then we have the definitions of precision, recall, and F1 for Fiber metric:

$$
\mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { | \mathcal { F } ^ { \mathrm { p r e d } } | } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { | \mathcal { F } ^ { \mathrm { g t } } | } } , \qquad \mathrm { F } 1 = { \frac { 2 * \mathrm { P r e c i s i o n } * \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } } .
$$

As shown in Figure 1, the Fiber metric provides a more accurate characterization of reconstruction quality. Compared with DIADEM, it is more intuitive and flexible, making it a comprehensive structure-aware metric.

## 5 Experiments and Results

## 5.1 Block-level Reconstruction

Recent high-performing neuron reconstruction algorithms mainly adopt a two-stage pipeline consisting of a segmentation model followed by a reconstruction method. Therefore, we first selected ten segmentation models, including state-of-the-art 3D segmentation models [30, 31, 32, 33, 34, 35] as well as models specifically designed for curvilinear-structure neuron segmentation [36, 37, 38, 6]. We then combined these segmentation models with different reconstruction methods [4, 15, 5, 14]. In addition, we compared two learning-based reconstruction methods [21, 22].

The segmentation results are reported in Appendix Table A.1, showing that SegMamba and IVNet achieve the best segmentation performance. Figure 4 shows the reconstruction results produced by different methods and presents commonly used metrics with our recommended metrics, showing that the combination of SegMamba and neuTube achieves the best structure-aware reconstruction quality (Fiber-F1: 46.8). Table 1 shows the speeds of different reconstruction methods. Different methods exhibit large differences in reconstruction speed and can be roughly divided into three orders of magnitude. Kimimaro is the fastest, followed by neuTube and SPE-DNR; APP2 and SmartTrace are slower, while NETracer is the slowest. Detailed experimental results are provided in the Appendix. From the experimental results, we draw the following three main findings:

In two-stage reconstruction methods, the segmentation method and the reconstruction method are coupled. Previous studies have shown that traditional reconstruction methods can achieve better performance when preceded by segmentation preprocessing. Howerer, better segmentation performance does not imply better reconstruction quality: clDice performed substantially worse than SegMamba and IVNet in segmentation performance (Dice: 48.04 vs. 54.66 vs. 54.46), it achieved comparable, or even better, reconstruction quality when paired with the same reconstruction method (Fiber-F1: 45.8 vs. 46.8 vs. 45.6). Moreover, when clDice was used as the segmentation model, neuTube, APP2, and Kimimaro all achieved consistently strong performance, whereas the strong performance of SegMamba and IVNet depended more heavily on the choice of reconstruction algorithm. This suggests that reconstruction algorithms are affected by distinct error patterns produced by different segmentation models, indicating a complex coupling between segmentation method and reconstruction method.

Geometry-based metrics are not suitable for evaluating the quality of neuron reconstruction. Methods that perform well in terms of local geometric reconstruction do not necessarily achieve better structural reconstruction quality. For example, Kimimaro obtained the best SSD, Point-F1, Length-F1, and Leaf-F1 scores across multiple segmentation models, but its Bifurcation-F1 and Fiber-F1 were substantially lower than those of other methods. In contrast, neuTube and APP2 achieved only moderate performance on geometry-based metrics, yet performed strongly on Bifurcation-F1 and Fiber-F1. Figure 5 shows the correlations among different metrics computed from our largescale experimental results. From a statistical perspective, geometry-based metrics exhibit certain correlations with each other, but show no clear correlation with structure-based metrics.

![](images/a138d1c063d31df3353dd18f4ee12d5652100f93e5771f6af7d95d75494a2877.jpg)  
Figure 4: Results of different reconstruction methods on the block-level task. The top three metrics (SSD, Point-F1, Length-F1) are geometry-based, whereas the bottom three metrics (Leaf-F1, Bifurcation-F1, Fiber-F1) are structure-aware metrics that we recommend for evaluating neuron reconstruction quality. In each radar plot, the axes denote the segmentation methods, and the closed curves correspond to the reconstruction methods. NETracer does not rely on segmentation methods and therefore forms an approximately circular curve.

Table 1: Average speed of different reconstruction methods.
<table><tr><td>Method</td><td>Speed (s/cube)</td></tr><tr><td>APP2 [4]</td><td>192.857</td></tr><tr><td>SmartTrace [15]</td><td>155.760</td></tr><tr><td>neuTube [5]</td><td>12.032</td></tr><tr><td>Kimimaro [14]</td><td>0.891</td></tr><tr><td>SPE-DNR [21]</td><td>12.005</td></tr><tr><td>NetTracer [22]</td><td>1025.043</td></tr></table>

![](images/869fd8fc4ba4a9c83f0a3d98bdc0ed7d9a891a4da6c75ad037f26afd94879250.jpg)  
Figure 5: Pearson correlation of metrics

Learning-based end-to-end reconstruction algorithms have great potential. It provides a potential path toward addressing the optimization difficulties caused by the coupling between segmentation and reconstruction methods in two-stage pipelines. However, existing learning-based methods, such as SPE-DNR and NETracer, mainly predict local neurite information and show limited ability to reconstruct long-range neurites, as indicated by their low Fiber-F1 scores. In contrast, the strong performance of SegMamba, which explicitly incorporates long-range context modeling, further demonstrates the importance of modeling long-range information for tracing long neurites.

## 5.2 Brain-wide Reconstruction

We selected several representative methods by jointly considering their reconstruction performance and efficiency, and evaluated them at the brain-wide scale through our proposed whole-brain tracing framework. Because tracing speed varies substantially across methods and neuron sizes differ greatly the runtime for reconstructing a single neuron ranged from 30 seconds to 13.35 hours. In total, evaluating all methods required more than 400 GPU-hours. The results are shown in Figure 6.

As shown in the figure, clDice with Kimimaro achieves the best performance in dendrite reconstruction, whereas SwinUNETR with APP2 performs best in axon reconstruction. Although SegMamba with Kimimaro achieves the best overall performance, its Fiber-F1 remains only 35.1%, highlighting the substantial gap that still exists toward fully automated reconstruction. While existing methods can reconstruct dendrites reasonably well, they still struggle to reconstruction long-range axonal structures, highlighting the challenge of long-distance continuity in brain-wide reconstruction. Kimimaro achieves better reconstruction quality for dendrites, whereas no reconstruction method demonstrates a significant advantage in axon reconstruction. In addition, segmentation methods that lead to better dendrite reconstruction are primarily distributed in the upper-right region of the radar plot, while those that improve axon reconstruction are mainly concentrated in the lower-left region, highlighting the challenge of developing algorithms that perform well on both structures.

![](images/b5650297b025a2cf3ac67975e7373f76aa8cb64aad1a0e622d49ce05fdc8e28a.jpg)  
Figure 6: Whole-brain reconstruction performance across nine segmentation methods and three reconstruction methods. Points denote micro-F1 estimates and horizontal whiskers denote 95%BCa bootstrap confidence intervals based on 10,000 neuron-resampling replicates.

Figure 7 shows the representative errors that occur in brain-wide reconstruction. As illustrated in the figure, errors in brain-wide neuron reconstruction mainly fall into three categories. (1) Break error: the predicted fiber terminates prematurely and fails to follow the correct continuation in the ground truth, leading to missing branches or incomplete long-range projections. (2) Merge error I (same-neuron crossover): two nearby fibers from the same neuron are incorrectly connected, which alters the intrinsic branching organization of the target neuron. (3) Merge error I (other-neuron crossover): the tracing result incorrectly crosses from the target neuron to a different neuron, causing the predicted reconstruction of neuron 1 to erroneously absorb fibers belonging to neuron 2.

![](images/a2acde14dc2607a4f59cf805832cb51c41c4bf9ab65def808219aa3daf1386a8.jpg)  
(a) Break Error (premature termination)

![](images/07cae7ac1af7beb5c9b5e8936276736edaa294c3582b1c3867943372c43ceb38.jpg)  
(b) Merge Error I (same neuron crossover)

![](images/376b687e85514d1f7b190a40400075960e2e6e66f24fe6831f2c4691f5f77a6e.jpg)  
(c) Merge Error II (other neuron crossover)  
Figure 7: Visualization of reconstruction errors in brain-wide neuron reconstruction. The ground-truth annotations of neuron 1 and neuron 2 are shown in green and blue, respectively, while the prediction for neuron 1 is shown in red. The arrows along the tubular structures indicate the fiber direction.

These three error types can have a substantial impact on automatic brain-wide neuron reconstruction: a single local error may propagate and affect the reconstruction at the whole-brain scale. For example (1) a Break Error near the proximal axon can cause large range axonal branches to be missed; (2) Merge Error I may cause the tracing process to jump onto another fiber and trace backward toward the soma, producing many false positives and false negatives; and (3) Merge Error II may lead the reconstruction to follow another neuron, generating numerous false positives while also wasting tracing time. These errors highlight the central challenge of brain-wide neuron reconstruction: accurate local geometry alone is insufficient for structurally correct reconstruction. A reliable method must preserve continuity, branching structure, and neuronal identity over long distances.

## 6 Conclusion

We introduced CORAL, the first benchmark for automatic brain-wide neuron reconstruction from light microscopy images. By unifying block-level and brain-wide reconstruction tasks, the structure-aware metric, and the whole-brain neuron tracing Framework, CORAL provides a standardized platform for developing and testing brain-wide neuron reconstruction methods. We conducted extensive experiments on this benchmark and identified several key insights that may help guide future research in automated neuron reconstruction. We hope CORAL will serve as a efficient platform for future research on brain-wide neuron reconstruction.

## References

[1] Johan Winnubst, Erhan Bas, Tiago A Ferreira, Zhuhao Wu, Michael N Economo, Patrick Edson, Ben J Arthur, Christopher Bruns, Konrad Rokicki, David Schauder, et al. Reconstruction of 1,000 projection neurons reveals new cell types and organization of long-range connectivity in the mouse brain. Cell, 179(1):268–281, 2019.

[2] Lijuan Liu, Zhixi Yun, Linus Manubens-Gil, Hanbo Chen, Feng Xiong, Hongwei Dong, Hongkui Zeng, Michael Hawrylycz, Giorgio A Ascoli, and Hanchuan Peng. Connectivity of single neurons classifies cell subtypes in mouse brains. Nature methods, 22(4):861–873, 2025.

[3] Le Gao, Haifang Wang, Yu Chen, Zhaoqin Chen, Li Deng, Dechen Liu, Xiaojing Ding, Ziwei Le, Yuan Liu, Yun Du, et al. Integrative analysis of single-neuron projectomes links connectome, transcriptome, and function in the mouse cortex. Neuron, 114(1):86–104, 2026.

[4] Hang Xiao and Hanchuan Peng. App2: automatic tracing of 3d neuron morphology based on hierarchical pruning of a gray-weighted image distance-tree. Bioinformatics, 29(11):1448–1454, 2013.

[5] Linqing Feng, Ting Zhao, and Jinhyun Kim. neutube 1.0: a new design for efficient neuron reconstruction software based on the swc format. eneuro, 2(1), 2015.

[6] Min Liu, Shuhan Wu, Runze Chen, Zhuangdian Lin, Yaonan Wang, and Erik Meijering. Brain image segmentation for ultrascale neuron reconstruction via an adaptive dual-task learning network. IEEE transactions on medical imaging, 43(7):2574–2586, 2024.

[7] Kerry M Brown, Germán Barrionuevo, Alison J Canty, Vincenzo De Paola, Judith A Hirsch, Gregory SXE Jefferis, Ju Lu, Marjolein Snippe, Izumi Sugihara, and Giorgio A Ascoli. The diadem data sets: representative light microscopy images of neuronal morphology to advance automation of digital reconstructions. Neuroinformatics, 9(2):143–157, 2011.

[8] Linus Manubens-Gil, Zhi Zhou, Hanbo Chen, Arvind Ramanathan, Xiaoxiao Liu, Yufeng Liu, Alessandro Bria, Todd Gillette, Zongcai Ruan, Jian Yang, et al. Bigneuron: a resource to benchmark and predict performance of algorithms for automated tracing of neurons in light microscopy datasets. Nature Methods, 20(6):824–835, 2023.

[9] Hanchuan Peng, Zongcai Ruan, Fuhui Long, Julie H Simpson, and Eugene W Myers. V3d enables real-time 3d visualization and quantitative analysis of large-scale biological image data sets. Nature biotechnology, 28(4):348–353, 2010.

[10] Han Zhang, Chao Liu, Yifei Yu, Jianhua Dai, Ting Zhao, and Nenggan Zheng. Pyneval: A python toolbox for evaluating neuron reconstruction performance. Frontiers in Neuroinformatics, 15:767936, 2022.

[11] Rubin Zhao, Yang Liu, Shiqi Zhang, Zijian Yi, Yanyang Xiao, Fang Xu, Yi Yang, and Pencheng Zhou. Neurofly: A framework for whole-brain single neuron reconstruction. arXiv preprint arXiv:2411.04715, 2024.

[12] Mariah Kenney, Iaroslavna Vasylieva, Greg Hood, Ivan Cao-Berg, Luke Tuite, Rozita Laghaei, Megan C Smith, Alan M Watson, and Alexander J Ropelewski. The brain image library: A community-contributed microscopy resource for neuroscientists. Scientific Data, 11(1):1212, 2024.

[13] Hanchuan Peng, Fuhui Long, and Gene Myers. Automatic 3d neuron tracing using all-path pruning. Bioinformatics, 27(13):i239–i247, 2011.

[14] William Silversmith, J. Alexander Bae, Peter H. Li, and A. M. Wilson. Kimimaro: Skeletonize densely labeled 3d image segmentations, September 2021.

[15] Hanbo Chen, Hang Xiao, Tianming Liu, and Hanchuan Peng. Smarttracing: self-learning-based neuron reconstruction. Brain informatics, 2(3):135–144, 2015.

[16] Yu Wang, Arunachalam Narayanaswamy, Chia-Ling Tsai, and Badrinath Roysam. A broadly applicable 3-d neuron tracing method based on open-curve snake. Neuroinformatics, 9(2):193– 217,2011.

[17] Rongjian Li, Tao Zeng, Hanchuan Peng, and Shuiwang Ji. Deep learning segmentation of optical microscopy images improves 3-d neuron reconstruction. IEEE transactions on medical imaging, 36(7):1533–1541, 2017.

[18] Heng Wang, Donghao Zhang, Yang Song, Siqi Liu, Yue Wang, Dagan Feng, Hanchuan Peng, and Weidong Cai. Segmenting neuronal structure in 3d optical microscope images via knowledge distillation with teacher-student network. In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), pages 228–231. IEEE, 2019.

[19] Bo Yang, Weixun Chen, Huiqiong Luo, Yinghui Tan, Min Liu, and Yaonan Wang. Neuron image segmentation via learning deep features and enhancing weak neuronal structures. IEEE Journal of Biomedical and Health Informatics, 25(5):1634–1645, 2020.

[20] Haiyang Yan, Yanchao Zhang, Zhenchen Li, Jinyue Guo, Hao Zhai, Jiazheng Liu, Yongwei Zhong, Jingbin Yuan, Lijun Shen, Linlin Li, et al. Glancing beyond patch: Spatial contextual cues for 3d neuron segmentation. IEEE Transactions on Medical Imaging, 2025.

[21] Weixun Chen, Min Liu, Hao Du, Miroslav Radojević, Yaonan Wang, and Erik Meijering. Deeplearning-based automated neuron reconstruction from 3d microscopy images using synthetic training images. IEEE Transactions on Medical Imaging, 41(5):1031–1042, 2021.

[22] Chao Liu, Yangbo Jiang, and Nenggan Zheng. Netracer: A topology-aware iterative tracing approach for tubular structure extraction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20593–20602, 2025.

[23] Hanchuan Peng, Zongcai Ruan, Deniz Atasoy, and Scott Sternson. Automatic reconstruction of 3d neuron structures using a graph-augmented deformable model. Bioinformatics, 26(12):i38– i46, 2010.

[24] Todd A. Gillette and John J. Grefenstette. On comparing neuronal morphologies with the constrained tree-edit-distance. Neuroinformatics, 7(3):191–194, 2009.

[25] Todd A Gillette, Kerry M Brown, and Giorgio A Ascoli. The diadem metric: comparing multiple reconstructions of the same neuron. Neuroinformatics, 9(2):233–245, 2011.

[26] Yadong Gang, Hongfu Zhou, Yao Jia, Ling Liu, Xiuli Liu, Gong Rao, Longhui Li, Xiaojun Wang, Xiaohua Lv, Hanqing Xiong, et al. Embedding and chemical reactivation of green fluorescent protein in the whole mouse brain for optical micro-imaging. Frontiers in neuroscience, 11:121, 2017.

[27] Hui Gong, Shaoqun Zeng, Cheng Yan, Xiaohua Lv, Zhongqin Yang, Tonghui Xu, Zhao Feng, Wenxiang Ding, Xiaoli Qi, Anan Li, et al. Continuously tracing brain-wide long-distance axonal projections in mice at a one-micron voxel resolution. Neuroimage, 74:87–98, 2013.

[28] Le Gao, Sang Liu, Lingfeng Gou, Yachuang Hu, Yanhe Liu, Li Deng, Danyi Ma, Haifang Wang, Qiaoqiao Yang, Zhaoqin Chen, et al. Single-neuron projectome of mouse prefrontal cortex. Nature neuroscience, 25(4):515–529, 2022.

[29] Giorgio A Ascoli, Duncan E Donohue, and Maryam Halavi. Neuromorpho. org: a central resource for neuronal morphologies. Journal of Neuroscience, 27(35):9247–9251, 2007.

[30] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[31] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021.

[32] Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 574–584, 2022.

[33] Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, Dong Yang, Holger R Roth, and Daguang Xu. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI brainlesion workshop, pages 272–284. Springer, 2021.

[34] Saikat Roy, Gregor Koehler, Constantin Ulrich, Michael Baumgartner, Jens Petersen, Fabian Isensee, Paul F Jaeger, and Klaus H Maier-Hein. Mednext: transformer-driven scaling of convnets for medical image segmentation. In International conference on medical image computing and computer-assisted intervention, pages 405–415. Springer, 2023.

[35] Zhaohu Xing, Tian Ye, Yijun Yang, Du Cai, Baowen Gai, Xiao-Jian Wu, Feng Gao, and Lei Zhu. Segmamba-v2: Long-range sequential modeling mamba for general 3d medical image segmentation. IEEE Transactions on Medical Imaging, 2025.

[36] Min Liu, Huiqiong Luo, Yinghui Tan, Xueping Wang, and Weixun Chen. Improved v-net based image segmentation for 3d neuron reconstruction. In 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 443–448. IEEE, 2018.

[37] Suprosanna Shit, Johannes C Paetzold, Anjany Sekuboyina, Ivan Ezhov, Alexander Unger, Andrey Zhylka, Josien PW Pluim, Ulrich Bauer, and Bjoern H Menze. cldice-a novel topologypreserving loss function for tubular structure segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16560–16569, 2021.

[38] Yaolei Qi, Yuting He, Xiaoming Qi, Yuan Zhang, and Guanyu Yang. Dynamic snake convolution based on topological geometric constraints for tubular structure segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6070–6079, 2023.

[39] Robert C Cannon, Dennis A Turner, GK Pyapali, and HV Wheal. An on-line archive of reconstructed hippocampal neurons. Journal of neuroscience methods, 84(1-2):49–54, 1998.

[40] Z Xing, T Ye, Y Yang, G Liu, and L Zhu. Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation. arxiv 2024. arXiv preprint arXiv:2401.13560.

## Contents of Appendix

A SWC Format and Evaluation Metrics. 14   
A.1 SWC Format 14   
A.2 Spatial Distance (SD) 14   
A.3 Substantial Spatial Distance (SSD) . 14   
A.4 Point Metrics 15   
A.5 Length-F1 15   
A.6 Keypoint Metrics. 15   
A.7 DIADEM 16   
A.8 Fiber Metrics 16   
B Whole-Brain Neuron Tracing Framework 17   
B.1 Overlapping Cube Grid Construction . 17   
B.2 Local Cube Tracing 17   
B.3 Topology-aware Growing 18   
B.4 Heuristic-guided Search . 18   
C Experiment settings 19   
C.1 Segmentation Training and Testing Configuration 19   
C.2 Implementation of Reconstruction Methods 19   
D Supplementary Experimental Results . 19   
D.1 Segmentation Results 19   
D.2 Detailed Results on the Block-level Task. 19   
D.3 Detailed Results on the Brain-wide Task . 19   
D.4 Parameter Sensitivity of Fiber Metric. 21   
D.5 Generalization Performance on the Out-of-domain Test Set 22   
D.6 Performance of Individual Human Annotators 22   
E Visualization of Bad Case in Brain-wide . 23   
F Limitations 23

## A SWC Format and Evaluation Metrics

This section defines the representation and reconstruction metrics used in our benchmark. The definitions follow our evaluation implementation. We denote the ground-truth and predicted reconstructions by $T _ { g }$ and $T _ { p } ,$ respectively. When required, a scale vector $\mathbf { s } = ( s _ { x } , s _ { y } , s _ { z } )$ is applied coordinate-wise to trānsform both reconstructions into a common coordinate system.

## A.1 SWC Format

We use the SWC format [39] to represent neuronal morphology as a rooted tree, or a forest when a local block contains multiple disconnected components, embedded in $\mathbb { R } ^ { 3 }$ . Each row of an SWC file describes one sampled skeleton node as a seven-tuple

$$
v _ { i } = ( i , t _ { i } , x _ { i } , y _ { i } , z _ { i } , r _ { i } , p _ { i } ) ,
$$

where i is the node identifier, $t _ { i }$ is its anatomical type, $( x _ { i } , y _ { i } , z _ { i } )$ are its spatial coordinates, $r _ { i }$ is the local radiu $^ { 1 S , }$ and $p _ { i }$ is the identifier of its parent. The conventional type codes 1, 2, 3, and 4 denote soma, axon, basal dendrite, and apical dendrite, respectively. A node with $p _ { i } = - 1$ is a root. Every non-root node and its parent form a straight edge in three-dimensional space; hence, the parent field induces a directed tree structure. In a complete reconstruction the root generally corresponds to the soma, whereas a component clipped by a block boundary may have an artificial root.

For the geometric metrics below, let ${ \cal R } _ { h } ( T )$ denote the topology-preserving resampling used in our implementation: each chain between consecutive key nodes (roots, bifurcations, or leaves) is resampled at approximately uniform spacing h, while the key nodes are retained. For two finite point sets $\bar { X }$ and Y, define the nearest-neighbor distance

$$
d ( \mathbf { x } , Y ) = \operatorname* { m i n } _ { \mathbf { y } \in Y } \| \mathbf { x } - \mathbf { y } \| _ { 2 } .
$$

## A.2 Spatial Distance (SD)

Spatial distance measures the average geometric discrepancy between two reconstructions [23]. In the implementation, both trees are first scaled and then resampled using $h _ { s } .$ , which is also used as the substantial-distance threshold. Let $X _ { g }$ and $X _ { p }$ be the node-coordinate sets of $R _ { h _ { s } } ( \mathbf { s } _ { } \odot T _ { g } )$ and $R _ { h _ { s } } ( \mathbf { s } \odot T _ { p } )$ . The two directed distances are

$$
\mathrm { S D } _ { g \to p } = \frac { 1 } { | X _ { g } | } \sum _ { \mathbf { x } \in X _ { g } } d ( \mathbf { x } , X _ { p } ) , \qquad \mathrm { S D } _ { p \to g } = \frac { 1 } { | X _ { p } | } \sum _ { \mathbf { x } \in X _ { p } } d ( \mathbf { x } , X _ { g } ) ,
$$

and the reported score is

$$
\mathrm { S D } = \frac { \mathrm { S D } _ { g \to p } + \mathrm { S D } _ { p \to g } } { 2 } .
$$

The directed terms reflect geometric omission and commission errors, respectively. A smaller SD indicates closer overall spatial agreement. We use $h _ { s } = 2$ by default.

## A.3 Substantial Spatial Distance (SSD)

SSD suppresses small localization deviations and averages only nearest-neighbor distances strictly larger than $h _ { s }$ [23, 10]. Using the same resampled sets as SD, define

$$
S _ { g \to p } = \{ \mathbf { x } \in X _ { g } : d ( \mathbf { x } , X _ { p } ) > h _ { s } \} , \qquad S _ { p \to g } = \{ \mathbf { x } \in X _ { p } : d ( \mathbf { x } , X _ { g } ) > h _ { s } \} .
$$

For $( a , b ) \in \{ ( g , p ) , ( p , g ) \}$ , the directed SSD is

$$
\mathrm { S S D } _ { a  b } = \{ \begin{array} { l l } { \displaystyle \frac { 1 } { | { \cal S } _ { a  b } | } \sum _ { { \bf x } \in { \cal S } _ { a  b } } d ( { \bf x } , X _ { b } ) , } & { | { \cal S } _ { a  b } | > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} 
$$

and

$$
\mathrm { S S D } = \frac { \mathrm { S S D } _ { g \to p } + \mathrm { S S D } _ { p \to g } } { 2 } .
$$

Thus, SSD describes the magnitude of substantial errors rather than their frequency; lower is better. The implementation also computes the proportion of substantially different nodes,

$$
\mathrm { S S D } _ { \mathcal { Y } _ { 0 } } = \frac { | S _ { g  p } | + | S _ { p  g } | } { | X _ { g } | + | X _ { p } | } .
$$

## A.4 Point Metrics

Point metrics measure bidirectional nearest-node coverage [6]. In our implementation, each tree is resampled with step $h _ { p }$ and the resulting coordinates are then scaled. Let $Y _ { g }$ and $Y _ { p }$ denote these two node sets. A predicted point and a ground-truth point are counted as covered when their respective nearest-neighbor distances are strictly smaller than $\tau _ { p } \colon$

$$
\begin{array} { r } { \mathcal { C } _ { p } = \{ \mathbf x \in Y _ { p } : d ( \mathbf x , Y _ { g } ) < \tau _ { p } \} , \qquad \mathcal { C } _ { g } = \{ \mathbf y \in Y _ { g } : d ( \mathbf y , Y _ { p } ) < \tau _ { p } \} . } \end{array}
$$

The point precision, recall, and F1 score are therefore

$$
P _ { \mathrm { p o i n t } } = \frac { \left| \mathcal { C } _ { p } \right| } { \left| Y _ { p } \right| } , \qquad R _ { \mathrm { p o i n t } } = \frac { \left| \mathcal { C } _ { g } \right| } { \left| Y _ { g } \right| } , \qquad \mathrm { F 1 } _ { \mathrm { p o i n t } } = \frac { 2 P _ { \mathrm { p o i n t } } R _ { \mathrm { p o i n t } } } { P _ { \mathrm { p o i n t } } + R _ { \mathrm { p o i n t } } } .
$$

Let $N _ { \mathrm { m i s s } } = | Y _ { g } | - | \mathcal { C } _ { g } |$ and $N _ { \mathrm { e x t r a } } = | Y _ { p } | - | \mathcal { C } _ { p } |$ be the numbers of missed and extra points. The miss-extra score (MES) is

$$
\mathrm { M E S } = \frac { | Y _ { g } | - N _ { \mathrm { m i s s } } } { | Y _ { g } | + N _ { \mathrm { e x t r a } } } .
$$

The nearest-neighbor associations are directional and need not be one-to-one. If either point set is empty, our implementation returns zero for all four point scores. We use $h _ { p } = 2$ and $\tau _ { p } = 4$ by default.

## A.5 Length-F1

Length-F1 evaluates matched edge length and incorporates local connectivity in addition to point proximity [10]. Both trees are resampled with step $h _ { l }$ and then scaled. For every ground-truth edge $\boldsymbol { e } _ { g } = \left( u , v \right)$ , we find candidate predicted edges near u and v and project the endpoints to the candidates. Let û and  be the projected points and $P _ { p } ( \hat { u } , \hat { v } )$ the unique path between them in the predicted tree. The ground-truth edge is accepted when

$$
\operatorname* { m a x } \{ \mathrm { d i s t } ( u , e _ { p } ^ { u } ) , \mathrm { d i s t } ( v , e _ { p } ^ { v } ) \} \leq \tau \upsilon , \qquad \left| L ( P _ { p } ( \hat { u } , \hat { v } ) ) - L ( e _ { g } ) \right| < \rho _ { l } L ( e _ { g } ) ,
$$

where dist $; ( \cdot , e )$ is Euclidean point-to-segment distance, and no part of the predicted path has already been assigned to another ground-truth edge. This last constraint prevents duplicated credit. Let $M _ { g }$ be the set of accepted ground-truth edges and let $L _ { p } ^ { \mathrm { m a t c h } }$ be the sum of the corresponding predicted path lengths. Then

$$
\displaystyle P _ { \mathrm { l e n g t h } } = \frac { L _ { p } ^ { \mathrm { m a t c h } } } { L ( T _ { p } ) } , \qquad \displaystyle R _ { \mathrm { l e n g t h } } = \frac { L ( M _ { g } ) } { L ( T _ { g } ) } , \qquad \displaystyle \mathrm { F l _ { l e n g t h } } = \frac { 2 P _ { \mathrm { l e n g t h } } R _ { \mathrm { l e n g t h } } } { P _ { \mathrm { l e n g t h } } + R _ { \mathrm { l e n g t h } } + \epsilon } .
$$

We use $h _ { l } = 2 , \tau _ { l } = 2 , \rho _ { l } = 0 . 2$ , and the numerical guard $\epsilon = 1 0 ^ { - 6 }$ . Compared with Point-F1, Length-F1 rejects spatially close structures whose connectivity or intervening path length is inconsistent.

## A.6 Keypoint Metrics

We define two classes of topologically important nodes. A bifurcation is a non-root node with more than one child, and a leaf is a node with no children. In block-level evaluation, an artificial root with exactly one child is additionally treated as a leaf because it represents a neurite truncated by the block boundary; this rule is disabled for whole-brain evaluation.

Let $\textstyle { \mathcal { K } } _ { g }$ and $\displaystyle { { \cal K } _ { p } }$ be the scaled coordinate sets of the selected ground-truth and predicted keypoint types. We compute their pairwise Euclidean distances and use the Hungarian algorithm to obtain the minimum-cost one-to-one assignment $\mathcal { M } _ { k }$ . An assigned pair is a true positive only when its distance is strictly smaller than $\tau _ { k } \colon$

$$
\mathrm { T P } _ { k } = \sum _ { ( \mathbf { x } , \mathbf { y } ) \in \mathcal { M } _ { k } } \mathbb { I } ( \| \mathbf { x } - \mathbf { y } \| _ { 2 } < \tau _ { k } ) .
$$

With $\mathrm { F P } _ { k } = | { \cal K } _ { p } | - \mathrm { T P } _ { k }$ and $\mathrm { F N } _ { k } = | \mathcal { K } _ { g } | - \mathrm { T P } _ { k }$ , the scores are

$$
P _ { \mathrm { k e y } } = \frac { \mathrm { T P } _ { k } } { | K _ { p } | } , \qquad R _ { \mathrm { k e y } } = \frac { \mathrm { T P } _ { k } } { | K _ { g } | } , \qquad \mathrm { F 1 _ { k e y } } = \frac { 2 P _ { \mathrm { k e y } } R _ { \mathrm { k e y } } } { P _ { \mathrm { k e y } } + R _ { \mathrm { k e y } } } .
$$

The main keypoint score uses the union of bifurcations and leaves. We additionally repeat the same matching independently for each class and report Bifurcation-Precision/Recall/F1 and Leaf-Precision/Recall/F1. We use $\tau _ { k } = 5$

## A.7 DIADEM

For reproducibility, we define the DIADEM value exactly as computed by our evaluation implementation. Each non-root edge of a scaled tree is divided into $n _ { e } = \dot { \left| L ( e ) / h _ { d } \right| }$ equal subsegments. Their midpoints form a sample set $Q ( T )$ , and every midpoint q has weight $w ( \mathbf { q } ) = L ( e ) / n _ { e }$ . A ground-truth sample is covered if its nearest predicted sample is within $\tau _ { d } ,$ and vice versa:

$$
\begin{array} { r } { \mathcal { H } _ { g } = \{ \mathbf { q } \in Q ( T _ { g } ) : d ( \mathbf { q } , Q ( T _ { p } ) ) \leq \tau _ { d } \} , \qquad \mathcal { H } _ { p } = \{ \mathbf { q } \in Q ( T _ { p } ) : d ( \mathbf { q } , Q ( T _ { g } ) ) \leq \tau _ { d } \} . } \end{array}
$$

The length-weighted recall, precision, and reported score are

$$
R _ { d } = \frac { \sum _ { \mathbf { q } \in \mathcal { H } _ { g } } w ( \mathbf { q } ) } { \sum _ { \mathbf { q } \in Q ( T _ { g } ) } w ( \mathbf { q } ) } , \qquad P _ { d } = \frac { \sum _ { \mathbf { q } \in \mathcal { H } _ { p } } w ( \mathbf { q } ) } { \sum _ { \mathbf { q } \in Q ( T _ { p } ) } w ( \mathbf { q } ) } , \qquad \mathrm { D I A D E M } = \frac { 2 P _ { d } R _ { d } } { P _ { d } + R _ { d } } .
$$

We use $h _ { d } = 1$ and $\tau _ { d } = 2$ . If both trees have zero total edge length, the score is one; if only one is empty, it is zero. This implementation is a length-weighted bidirectional coverage score and should be distinguished from the classical ancestor-aware DIADEM protocol [25, 10].

## A.8 Fiber Metrics

Given a rooted SWC tree T, a fiber is the unique root-to-leaf path

$$
f = ( v _ { i _ { 1 } } , v _ { i _ { 2 } } , \ldots , v _ { i _ { K } } ) ,
$$

where $v _ { i _ { 1 } }$ is a root, $v _ { i _ { K } }$ is a leaf, and each node is the parent of the next. Its length is

$$
L ( f ) = \sum _ { k = 1 } ^ { K - 1 } \| \mathbf { x } _ { i _ { k + 1 } } - \mathbf { x } _ { i _ { k } } \| _ { 2 } .
$$

Each input tree is first resampled with step $h _ { 0 } = 2$ For block-level evaluation, each predicted component is re-rooted at the node nearest to a ground-truth root whenever their distance in the input coordinate system is at most 20; this operation changes the root and edge directions but does not translate coordinates. The coordinates are then scaled and root-to-leaf paths are extracted.

For FIoU computation, each fiber is again resampled along arc length with step $h _ { f } .$ Let $Z _ { f } =$ $\left( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { m } \right)$ be this ordered point sequence, $\mathbf { c } _ { i } = ( \mathbf { z } _ { i } + \mathbf { z } _ { i + 1 } ) / 2$ its ith segment midpoint, and $\delta _ { i } = \lVert \mathbf { z } _ { i + 1 } - \mathbf { z } _ { i } \rVert _ { 2 }$ the corresponding segment length. The directed overlap is

$$
O ( f _ { a } \to f _ { b } ) = \sum _ { i = 1 } ^ { | Z _ { f _ { a } } | - 1 } \mathbb { I } ( d ( \mathbf { c } _ { i } , Z _ { f _ { b } } ) \leq \tau _ { f } ) \delta _ { i } .
$$

The distance is thus evaluated against the resampled points of the other fiber. We symmetrize the overlap and define fiber intersection-over-union (FIoU) as

$$
{ \cal O } ( f _ { a } , f _ { b } ) = \frac { { \cal O } ( f _ { a } \to f _ { b } ) + { \cal O } ( f _ { b } \to f _ { a } ) } { 2 } , \qquad \mathrm { F I o U } ( f _ { a } , f _ { b } ) = \frac { { \cal O } ( f _ { a } , f _ { b } ) } { L ( f _ { a } ) + L ( f _ { b } ) - { \cal O } ( f _ { a } , f _ { b } ) + \epsilon _ { \mathrm { i o t } } } .
$$

We use $h _ { f } = 1 , \tau _ { f } = 5 .$ and $\epsilon _ { \mathrm { i o u } } = 1 0 ^ { - 7 }$ . Fiber pairs with a length ratio below 0.5 or a centroid distance above 100 are pruned before exact overlap computation. In the direction-aware whole-brain setting, a pair is also pruned when the distance between its roots or between its leaves exceeds half of the longer fiber length.

Let ${ \mathcal { F } } _ { p }$ and $\mathcal { F } _ { g }$ be the predicted and ground-truth fiber sets. We use the Hungarian algorithm to find the one-to-one assignment that maximizes total FIoU. If $\mathcal { M } _ { f }$ denotes the assignment, a pair is correct only if its FIoU reaches η:

$$
\mathrm { T P } _ { \mathrm { f i b e r } } = \sum _ { ( f _ { p } , f _ { g } ) \in \mathcal { M } _ { f } } \mathbb { I } ( \mathrm { F I o U } ( f _ { p } , f _ { g } ) \geq \eta ) .
$$

Accordingly,

$$
\mathrm { F P _ { f b e r } } = | \mathcal { F } _ { p } | - \mathrm { T P _ { f b e r } } , \qquad \mathrm { F N _ { f b e r } } = | \mathcal { F } _ { g } | - \mathrm { T P _ { f b e r } } ,
$$

and the fiber precision, recall, and F1 score are

$$
P _ { \mathrm { f i b e r } } = \frac { \mathrm { T P } _ { \mathrm { f i b e r } } } { | \mathcal { F } _ { p } | } , \qquad R _ { \mathrm { f i b e r } } = \frac { \mathrm { T P } _ { \mathrm { f i b e r } } } { | \mathcal { F } _ { g } | } , \qquad \mathrm { F 1 } _ { \mathrm { f i b e r } } = \frac { 2 P _ { \mathrm { f i b e r } } R _ { \mathrm { f i b e r } } } { P _ { \mathrm { f i b e r } } + R _ { \mathrm { f i b e r } } + \epsilon _ { \mathrm { f i } } } .
$$

We use $\epsilon _ { \mathrm { f 1 } } = 1 0 ^ { - 1 2 }$ and set $\eta = 0 . 8$ for block-level evaluation and $\eta = 0 . 7 5$ for whole-brain evaluation. In the whole-brain setting, axon and dendrite scores are additionally obtained by repeating the assignment within fibers whose leaf types are 2 and $3 / 4$ , respectively. Unlike local geometric metrics, Fiber-F1 gives credit only when an entire root-to-leaf trajectory is recovered with sufficient spatial overlap.

## B Whole-Brain Neuron Tracing Framework

We design a local-to-global tracing framework for whole-brain neuron reconstruction from ultra-large 3D microscopy volumes. The central idea is to decompose whole-brain reconstruction into an iterative local-to-global process. Specifically, the brain volume is partitioned into overlapping 3D cubes, and the reconstruction is grown from the soma by repeatedly performing three steps: search, trace, and grow. The search module identifies candidate nodes for further expansion based on the current partial neuron tree; the tracing module predicts a local skeleton within the selected cube; and the growing module attaches the newly traced local structure to the candidate nodes. Repeating this procedure yields a whole-brain neuron tree assembled from cube-level predictions.

## B.1 Overlapping Cube Grid Construction

To enable scalable tracing in an ultra-large brain volume, we organize the brain region of interest as an overlapping 3D grid of local cubes. Let the region of interest be denoted by $\Omega \subset \mathbb { R } ^ { 3 }$ . We parameterize the grid by a cube size $\left( c _ { x } , c _ { y } , c _ { z } \right)$ and a step size $\left( { { s _ { x } } , { s _ { y } } , { s _ { z } } } \right)$ , where typically $s _ { i } < c _ { i }$ for each axis so that adjacent cubes overlap with each other. This overlapping design allows the same neurite segment to be observed in multiple neighboring cubes, which is important for preserving local continuity across cube boundaries and for supporting reliable cross-cube merging.

Let $( x _ { 0 } , y _ { 0 } , z _ { 0 } )$ denote the minimum coordinate of the brain region of interest. Each cube, indexed by $( i , j , k )$ , corresponds to a a cubic region defined as

$$
C _ { i , j , k } = [ x _ { 0 } + i s _ { x } , ~ x _ { 0 } + i s _ { x } + c _ { x } ) \times [ y _ { 0 } + j s _ { y } , ~ y _ { 0 } + j s _ { y } + c _ { y } ) \times [ z _ { 0 } + k s _ { z } , ~ z _ { 0 } + k s _ { z } + c _ { z } ) . ~
$$

Here, the cube origin is determined by the grid index and the step size, while the cube extent is determined by the fixed cube size. In this way, the entire brain volume is covered by a set of partially overlapping local subvolumes that serve as the basic processing units of our framework.

The grid plays two roles in our method. First, it provides a discrete search space for whole-brain exploration, allowing the reconstruction process to move from one local region to another in a structured manner rather than scanning the full volume exhaustively. Second, it defines the spatial support of each local tracing step, so that cube-level predictions can later be associated and merged into a global neuron tree. Because neighboring cubes share overlapping content, branches that pass through cube boundaries can still be matched based on their common local geometry, which substantially improves the robustness of local-to-global assembly.

## B.2 Local Cube Tracing

For each selected cube, we perform local neuron tracing within the corresponding image subvolume Our framework is modular and can be readily coupled with different local tracing algorithms. This design makes the overall method flexible, as improvements in cube-level tracing can be directly incorporated into the whole-brain reconstruction pipeline.

Formally, for each cube $C _ { i , j , k } .$ , we extract an image volume $I _ { i , j , k }$ from the corresponding region of the whole-brain volume. The local tracing module then produces a cube-level skeleton

$$
S _ { i , j , k } = \mathcal { T } ( I _ { i , j , k } ) ,
$$

where $\tau$ denotes the local tracing operator. Since tracing is performed in the local coordinate system of the cube, the predicted skeleton is subsequently mapped back to the global coordinate system by adding the spatial offset of the cube origin.

To improve computational efficiency during iterative reconstruction, we cache the tracing result of each cube on disk. As a result, revisiting a previously processed cube does not require recomputing its local skeleton prediction. This significantly reduces the overall cost of the search-trace-merge procedure, particularly when multiple frontier nodes are associated with the same cube.

## B.3 Topology-aware Growing

To progressively construct a whole-brain neuronal morphology, we introduce a topology-aware growing module that incrementally integrates locally traced subtrees into a global reconstruction. Starting from a candidate frontier node, the module expands the global tree by identifying valid junctions and merging non-redundant structures in a topology-consistent manner.

Grid-based localization. Given a candidate expansion node $p _ { c } \in \mathbb { R } ^ { 3 }$ , we first determine the most relevant spatial region by selecting the cube whose center is closest to $p _ { c } \colon$

$$
\begin{array} { r } { ( i ^ { * } , j ^ { * } , k ^ { * } ) = \arg \underset { i , j , k } { \operatorname* { m i n } } \left\| \mathrm { c e n t e r } ( C _ { i , j , k } ) - p _ { c } \right\| _ { 2 } . } \end{array}
$$

This ensures that subsequent operations are performed within the most spatially relevant local context. Global-local subtree extraction. Within the selected cube $C _ { i ^ { * } , j ^ { * } , k ^ { * } }$ , we extract a local subtree $t _ { g } ~ \subset ~ T$ from the current global reconstruction $T _ { \ast }$ , restricted to the cube region. The subtree is re-rooted at $p _ { c } ,$ serving as the reference structure for subsequent matching.

Candidate junction selection. To identify potential connection points, we select a set of candidate junction nodes from $S _ { i ^ { * } , j ^ { * } , k ^ { * } }$ . Specifically, we retrieve the $N _ { c }$ nearest nodes to the node $p _ { c }$ that belong to distinct connected components:

$$
\mathcal { P } _ { J } = \{ p _ { j } ^ { ( 1 ) } , . . . , p _ { j } ^ { ( N _ { c } ) } \} .
$$

This design avoids redundant candidates from the same subtree and improves robustness in complex branching regions.

Fiber-level overlap matching. For each candidate junction node $p _ { j } \in \mathcal { P } _ { J } ,$ we consider its corresponding subtree $t _ { j }$ (re-rooted at $p _ { j } )$ and decompose both $t _ { j }$ and $t _ { g }$ into sets of fibers $( \mathrm { i . e . }$ , root-to-leaf paths). We define an overlap-based matching criterion between a candidate fiber $f _ { 1 } \subset t _ { j }$ and a reference fiber $f _ { 2 } \subset t _ { g }$ . Let overlap( $f _ { 1 } , f _ { 2 } )$ denote their shared arc length. $\mathbf { A }$ valid junction is identified if:

$$
\frac { \mathrm { o v e r l a p } ( f _ { 1 } , f _ { 2 } ) } { \mathrm { l e n g t h } ( f _ { 1 } ) } > \tau \quad \mathrm { o r } \quad \frac { \mathrm { o v e r l a p } ( f _ { 1 } , f _ { 2 } ) } { \mathrm { l e n g t h } ( f _ { 2 } ) } > \tau ,
$$

where $\tau$ is a predefined overlap threshold. This criterion ensures that the candidate subtree shares sufficient structural consistency with the existing reconstruction.

Conflict-aware merging. Once a valid junction is detected, we remove overlapping segments from $t _ { j }$ to avoid duplication. Specifically, the overlapping portion of the fiber is detached by severing its parent connection, preserving only the novel structure. The pruned subtree is then attached to the global tree $T$ as a child of ${ \dot { p } } _ { c }$ , provided that its total length exceeds a minimum threshold $L _ { \mathrm { m i n } }$ , which prevents the introduction of spurious short branches.

## B.4 Heuristic-guided Search

To progressively explore the whole-brain volume and expand the reconstructed neuronal morphology, we propose a heuristic-guided search strategy that iteratively grows a global tree from a set of frontier nodes. Starting from the soma annotation, an initial global tree $T$ is obtained via local tracing, and its frontier nodes are used to initialize a candidate set $\mathcal { Q } .$

At each iteration, a candidate node $p _ { c } \in \mathcal { Q }$ is selected to guide further exploration. The search is driven by two types of structurally informative nodes: $l e a f s ,$ which naturally indicate continuation directions of neuronal fibers, and margin nodes, which lie close to the boundary of the currently explored region and encourage expansion into unvisited areas. A node $p \in T$ is considered a margin candidate if its distance to the boundary of the current cube C satisfies

$$
d _ { \mathrm { m a r g i n } } ( p , C ) \leq \tau ^ { \prime } ,
$$

where $\tau ^ { \prime }$ is a predefined threshold.

Given a selected node $p _ { c } .$ we perform local tracing (Section B.2) and topology-aware merging (Section B.3) to expand the global tree. Newly attached subtrees introduce additional frontier nodes, from which new candidates are extracted. Specifically, leaf nodes of the newly merged subtree are added to $\mathcal { Q } ,$ while internal nodes located near cube boundaries are included as margin candidates. To avoid redundant exploration, we maintain a visited set V and ensure that each node is processed at most once.

To improve robustness against error accumulation, we adopt a breadth-first search (BFS) strategy instead of depth-first search (DFS). This choice encourages more uniform spatial exploration and reduces the risk that early merging errors propagate along a single incorrect branch.

In addition, we employ a cube-level caching mechanism to improve efficiency and stability. For each spatial cube, previously traced results are reused; however, fibers that have already been incorporated into the global tree are explicitly removed from the cached results before reuse. This prevents duplicated structures and mitigates the reinforcement of erroneous connections.

Overall, the proposed search procedure can be summarized as an iterative expansion process:

$$
Q \gets \mathrm { I n i t } ( T ) , \quad \mathbf { w h i l e } ~ \mathcal { Q } \neq \emptyset : \quad p _ { c } \gets \mathrm { P o p } ( \mathcal { Q } ) , \quad T \gets \mathrm { E x p a n d } ( T , p _ { c } ) , \quad \mathcal { Q } \gets \mathcal { Q } \cup \mathrm { U p d a t e } ( T ) .
$$

This heuristic-guided design balances exploration and reliability by integrating structural priors, traversal strategy, and redundancy control, enabling scalable and robust whole-brain reconstruction.

## C Experiment settings

## C.1 Segmentation Training and Testing Configuration

All segmentation models are trained under a unified protocol for fair comparison. During training, we randomly crop 3D blocks of size $1 2 8 \times 1 2 8 \times 1 2$ 8 from annotated cubes of size $3 0 0 \times 3 0 0 \times 3 0 0$ For validation and testing, inference is performed on cubes of size $3 0 0 \times 3 0 0 \times 3 0 0$ using a slidingwindow strategy with a window size of $1 2 8 \times 1 2 8 \times 1 2 8$ and an overlap ratio of 0.5. We optimize the models using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 4 }$ . The loss function is Dice loss. Training is conducted for 10,000 steps with a batch size of 4 on 4 GPUs using mixed-precision distributed training. We use a linear warmup schedule at the beginning of training, followed by cosine annealing. For preprocessing, training images are scaled to [0, 1], and random flipping is applied along the three spatial axes as data augmentation. For validation and testing, we use percentile-based intensity normalization by mapping the 1st to 99th percentiles to [0, 1]. All experiments were conducted on four NVIDIA GeForce RTX 3090 GPUs.

## C.2 Implementation of Reconstruction Methods

We slightly modify and to better fit our benchmark setting. The original implementations usually return only the brightest traced tree in a 3D volume. However, a local cube in our dataset may contain multiple disconnected neurite fragments. Therefore, we use an iterative trace-and-remove strategy: after each tracing step, the reconstructed tree is converted into a mask and removed from the image volume, and the tracer is then rerun on the residual image. The process stops when the returned SWC is empty or contains too few nodes. All recovered SWC trees are finally merged into a single output. This modification enables and to reconstruct multiple trees from one cube.

## D Supplementary Experimental Results

## D.1 Segmentation Results

Table A.1 presents the segmentation performance of ten different methods on the test set.

## D.2 Detailed Results on the Block-level Task

Table A.2 presents the detailed quantitative results of different tracing methods on the block-level reconstruction task, evaluated using keypoint-based and fiber-based metrics.

## D.3 Detailed Results on the Brain-wide Task

Table A.3 presents the detailed quantitative results of different tracing methods on the brain-wide reconstruction task, evaluated using fiber-based metrics.

Table A.1: Comparison of different segmentation methods on the block-level task.
<table><tr><td>Method</td><td>Dice↑</td><td>clDice↑</td><td>NSD↑</td><td>#Params(M)↓</td><td>GFLOPs↓</td><td>Latency(ms)↓</td></tr><tr><td>VNet [30]</td><td>53.69</td><td>64.21</td><td>81.90</td><td>45.71</td><td>1011.39</td><td>52.92</td></tr><tr><td>nnU-Net [31]</td><td>50.51</td><td>57.24</td><td>76.21</td><td>31.42</td><td>2141.14</td><td>33.29</td></tr><tr><td>UNETR [32]</td><td>45.53</td><td>55.87</td><td>73.78</td><td>121.35</td><td>195.61</td><td>37.84</td></tr><tr><td>SwinUNETR [33]</td><td>50.40</td><td>61.12</td><td>78.60</td><td>15.51</td><td>199.68</td><td>92.01</td></tr><tr><td>MedNeXt [34]</td><td>53.30</td><td>64.78</td><td>81.38</td><td>61.97</td><td>505.18</td><td>306.76</td></tr><tr><td>SegMamba [40]</td><td>54.66</td><td>65.08</td><td>82.13</td><td>64.24</td><td>1553.65</td><td>150.27</td></tr><tr><td>IVNet [36]</td><td>54.46</td><td>65.14</td><td>82.45</td><td>53.62</td><td>1183.02</td><td>60.00</td></tr><tr><td>clDice [37]</td><td>48.04</td><td>63.21</td><td>72.25</td><td>31.42</td><td>2141.14</td><td>33.29</td></tr><tr><td>DSCNet [38]</td><td>53.07</td><td>64.01</td><td>81.41</td><td>4.23</td><td>423.73</td><td>178.18</td></tr><tr><td>ADTL-Net [6]</td><td>12.76</td><td>16.11</td><td>20.62</td><td>7.69</td><td>286.91</td><td>43.09</td></tr></table>

Table A.2: Detailed block-level reconstruction performance of different methods. Best results are shown in bold, and second-best results are underlined.
<table><tr><td colspan="2">Method</td><td>SD↓</td><td>SSD↓</td><td>Point-F1↑</td><td>MES↑</td><td>Length-F1↑</td><td>Leaf-F1↑</td><td>Bifurcation-F1↑</td><td>Fiber-F1↑</td></tr><tr><td rowspan="10"></td><td>DSCNet</td><td>13.6505</td><td>42.6313</td><td>0.8502</td><td>0.7636</td><td>0.8022</td><td>0.4082</td><td>0.4250</td><td>0.4082</td></tr><tr><td>clDice</td><td>14.5049</td><td>56.2745</td><td>0.8546</td><td>0.7546</td><td>0.8152</td><td>0.3319</td><td>0.3923</td><td>0.4581</td></tr><tr><td>nn-UNet</td><td>12.9215</td><td>41.1503</td><td>0.8608</td><td>0.7786</td><td>0.8136</td><td>0.4392</td><td>0.4353</td><td>0.4287</td></tr><tr><td>IVNet</td><td>13.3140</td><td>41.9819</td><td>0.8562</td><td>0.7727</td><td>0.8106</td><td>0.4190</td><td>0.4507</td><td>0.4483</td></tr><tr><td>MedNeXt</td><td>15.9024</td><td>48.6796</td><td>0.8439</td><td>0.7485</td><td>0.7922</td><td>0.3904</td><td>0.3975</td><td>0.3252</td></tr><tr><td>SegMamba</td><td>13.5240</td><td>43.9229</td><td>0.8574</td><td>0.7725</td><td>0.8122</td><td>0.4454</td><td>0.4662</td><td>0.4424</td></tr><tr><td>SwinUNETR</td><td>12.1924</td><td>41.6085</td><td>0.8520</td><td>0.7656</td><td>0.8031</td><td>0.3629</td><td>0.4408</td><td>0.3325</td></tr><tr><td>UNETR</td><td>11.4839</td><td>38.4829</td><td>0.8455</td><td>0.7472</td><td>0.7902</td><td>0.3107</td><td>0.4213</td><td>0.2875</td></tr><tr><td>VNet</td><td>12.6861</td><td>41.3826</td><td>0.8570</td><td>0.7766</td><td>0.8108</td><td>0.4257</td><td>0.4403</td><td>0.4395</td></tr><tr><td>DSCNet</td><td>10.8461</td><td>39.6433</td><td>0.8624</td><td>0.7763</td><td>0.8054</td><td>0.5771</td><td>0.3570</td><td>0.4210</td></tr><tr><td rowspan="8">Kimimaro</td><td>clDice</td><td>9.8950</td><td>34.5246</td><td>0.8735</td><td>0.7924</td><td>0.8218</td><td>0.6103</td><td>0.3767</td><td>0.4487</td></tr><tr><td>nn-UNet</td><td>10.6941</td><td>39.7011</td><td>0.8717</td><td>0.7880</td><td>0.8189</td><td>0.5958</td><td>0.3818</td><td>0.4301</td></tr><tr><td>IVNet</td><td>10.7001</td><td>39.6400</td><td>0.8665</td><td>0.7841</td><td>0.8158</td><td>0.5706</td><td>0.3310</td><td>0.4109</td></tr><tr><td>MedNeXt</td><td>11.8287</td><td>44.5038</td><td>0.8649</td><td>0.7798</td><td>0.8050</td><td>0.5168</td><td>0.3338</td><td>0.3038</td></tr><tr><td>SegMamba</td><td>10.8848</td><td>39.9199</td><td>0.8618</td><td>0.7760</td><td>0.8114</td><td>0.5838</td><td>0.3979</td><td>0.4297</td></tr><tr><td>SwinUNETR</td><td>11.0525</td><td>31.7336</td><td>0.8433</td><td>0.7382</td><td>0.7743</td><td>0.4959</td><td>0.3378</td><td>0.3249</td></tr><tr><td>UNETR</td><td>14.8150</td><td>30.3950</td><td>0.7890</td><td>0.6472</td><td>0.7057</td><td>0.3923</td><td>0.2382</td><td>0.1904</td></tr><tr><td>VNet</td><td>10.9187</td><td>39.4200</td><td>0.8622</td><td>0.7766</td><td>0.8088</td><td>0.5659</td><td>0.3095</td><td>0.4063</td></tr><tr><td rowspan="9">neuTube</td><td>DSCNet</td><td>16.4049</td><td>52.8636</td><td>0.8263</td><td>0.7214</td><td>0.7935</td><td>0.4689</td><td>0.4403</td><td>0.4369</td></tr><tr><td>clDice</td><td>13.4700</td><td>45.2117</td><td>0.8415</td><td>0.7572</td><td>0.8099</td><td>0.4939</td><td>0.4507</td><td>0.4574</td></tr><tr><td>nn-UNet</td><td>15.4649</td><td>51.4791</td><td>0.8345</td><td>0.7375</td><td>0.8027</td><td>0.4810</td><td>0.4446</td><td>0.4475</td></tr><tr><td>IVNet</td><td>16.8210</td><td>54.6989</td><td>0.8262</td><td>0.7221</td><td>0.7959</td><td>0.4771</td><td>0.4569</td><td>0.4563</td></tr><tr><td>MedNeXt</td><td>17.9398</td><td>56.7850</td><td>0.8188</td><td>0.7127</td><td>0.7871</td><td>0.4724</td><td>0.4250</td><td>0.4430</td></tr><tr><td>SegMamba</td><td>16.4829</td><td>54.9054</td><td>0.8313</td><td>0.7285</td><td>0.8028</td><td>0.4913</td><td>0.4906</td><td>0.4683</td></tr><tr><td>SwinUNETR</td><td>14.8432</td><td>48.1729</td><td>0.8237</td><td>0.7257</td><td>0.7899</td><td>0.4541</td><td>0.4208</td><td>0.4020</td></tr><tr><td>UNETR</td><td>15.7476</td><td>45.8373</td><td>0.7983</td><td>0.6805</td><td>0.7503</td><td>0.3659</td><td>0.2874</td><td>0.3064</td></tr><tr><td>VNet</td><td>16.4064</td><td>53.4323</td><td>0.8259</td><td>0.7250</td><td>0.7952</td><td>0.4732</td><td>0.4453</td><td>0.4449</td></tr><tr><td rowspan="9">smartTrace</td><td>DSCNet</td><td>15.4830</td><td>42.2836</td><td>0.8213</td><td>0.7094</td><td>0.7876</td><td>0.3051</td><td>0.4303</td><td>0.3974</td></tr><tr><td>clDice</td><td>12.1445</td><td>34.3846</td><td>0.8106</td><td>0.7389</td><td>0.7795</td><td>0.3365</td><td>0.3870</td><td>0.4178</td></tr><tr><td>nn-UNet</td><td>14.8389</td><td>40.4214</td><td>0.8157</td><td>0.7152</td><td>0.7836</td><td>0.3041</td><td>0.4205</td><td>0.4021</td></tr><tr><td>IVNet</td><td>15.9178</td><td>42.6343</td><td>0.8189</td><td>0.7109</td><td>0.7873 0.7579</td><td>0.3055</td><td>0.4388</td><td>0.4141</td></tr><tr><td>MedNeXt</td><td>18.7440</td><td>49.0162</td><td>0.7907</td><td>0.6646</td><td></td><td>0.2781</td><td>0.4139</td><td>0.3910</td></tr><tr><td>SegMamba</td><td>15.8813</td><td>42.9193</td><td>0.8245</td><td>0.7193</td><td>0.7934</td><td>0.3251</td><td>0.4623</td><td>0.4092</td></tr><tr><td>SwinUNETR</td><td>14.1917</td><td>38.5577</td><td>0.8053</td><td>0.7141</td><td>0.7673</td><td>0.2903</td><td>0.4233</td><td>0.3951</td></tr><tr><td>UNETR</td><td>16.8839</td><td>36.5758</td><td>0.7834</td><td>0.6620</td><td>0.7391</td><td>0.2655</td><td>0.3973</td><td>0.3622</td></tr><tr><td>VNet</td><td>16.3725</td><td>43.0692</td><td>0.8237</td><td>0.7040</td><td>0.7923</td><td>0.3222</td><td>0.4424</td><td>0.4124</td></tr><tr><td rowspan="10">SPE-DNR</td><td>DSCNet clDice</td><td>16.3057</td><td>42.5090 52.9042</td><td>0.8239</td><td>0.6949 0.3415</td><td>0.7323 0.5486</td><td>0.1966 0.0657</td><td>0.0281 0.0078</td><td>0.0713 0.0309</td></tr><tr><td>nn-UNet</td><td>34.7587</td><td></td><td>0.5729</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IVNet</td><td>33.4240</td><td>52.2203</td><td>0.6036</td><td>0.3692</td><td>0.5694</td><td>0.0650</td><td>0.0096</td><td>0.0243</td></tr><tr><td></td><td>15.1397</td><td>39.7491</td><td>0.8293</td><td>0.7026</td><td>0.7208</td><td>0.1769</td><td>0.0357</td><td>0.0565</td></tr><tr><td>MedNeXt</td><td>20.0328</td><td>48.3904</td><td>0.8074</td><td>0.6558</td><td>0.7073</td><td>0.1674</td><td>0.0332</td><td>0.0587</td></tr><tr><td>SegMamba</td><td>15.9594 22.7519</td><td>42.3837 45.9359</td><td>0.8293 0.7519</td><td>0.6996 0.5723</td><td>0.7316 0.6445</td><td>0.1914 0.1165</td><td>0.0332 0.0251</td><td>0.0689 0.0363</td></tr><tr><td>SwinUNETR UNETR</td></table>

Table A.3: Whole-brain reconstruction results. Each value is the micro-F1 estimate followed by its 95% BCa bootstrap confidence interval (10,000 resamples). The best and second-best estimates in each metric are shown in bold and underlined, respectively.
<table><tr><td rowspan=1 colspan=11>Micro-F1 (95% BCa CI)ReconstructionMethodFiber              Dendrite              Axon</td></tr><tr><td rowspan=1 colspan=3>DSCnet     0.249</td><td rowspan=1 colspan=2>[0.196, 0.314]</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=2>[0.216, 0.479]</td><td rowspan=1 colspan=1>0.122 [</td><td rowspan=1 colspan=1>0.063, 0.201]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>cIDice</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=2>[0.234, 0.384]</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=2>[0.317, 0.567]</td><td rowspan=1 colspan=1>0.216[</td><td rowspan=1 colspan=1>0.147, 0.324]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>nn-UNet</td><td rowspan=1 colspan=1>0.269 [</td><td rowspan=1 colspan=2>0.210, 0.391]</td><td rowspan=1 colspan=1>0.323[</td><td rowspan=1 colspan=2>0.172, 0.491]</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=1>[0.112, 0.358]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>IVNet</td><td rowspan=1 colspan=1>0.296 [</td><td rowspan=1 colspan=2>0.250, 0.365]</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=2>[0.293, 0.555]</td><td rowspan=1 colspan=1>0.234 [0</td><td rowspan=1 colspan=1>.159, 0.313]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>APP2    MedNeXt   0.235 [</td><td rowspan=1 colspan=2>0.185, 0.279]</td><td rowspan=1 colspan=1>0.476 </td><td rowspan=1 colspan=2>[0.376, 0.560]</td><td rowspan=1 colspan=1>0.115 [0</td><td rowspan=1 colspan=1>.070, 0.178]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>SegMamba</td><td rowspan=1 colspan=1>0.304 [0</td><td rowspan=1 colspan=2>.252, 0.395]</td><td rowspan=1 colspan=1>0.418</td><td rowspan=1 colspan=2>[0.319, 0.571]</td><td rowspan=1 colspan=1>0.225 [0</td><td rowspan=1 colspan=1>.153, 0.332]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>SwinUNETR</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=2>[0.286, 0.362]</td><td rowspan=1 colspan=1>0.523[</td><td rowspan=1 colspan=2>0.402, 0.610]</td><td rowspan=1 colspan=1>0.244 [0</td><td rowspan=1 colspan=1>.168, 0.301]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>UNETR</td><td rowspan=1 colspan=1>0.247</td><td rowspan=1 colspan=2>[0.190, 0.302]</td><td rowspan=1 colspan=1>0.297 [0</td><td rowspan=1 colspan=2>.152, 0.414]</td><td rowspan=1 colspan=1>0.171 [0</td><td rowspan=1 colspan=1>.089, 0.257]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>VNet</td><td rowspan=1 colspan=1>0.264</td><td rowspan=1 colspan=2>[0.215, 0.325]</td><td rowspan=1 colspan=1>0.320 [0</td><td rowspan=1 colspan=2>.168, 0.495]</td><td rowspan=1 colspan=1>0.190 [0</td><td rowspan=1 colspan=1>.114, 0.269]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>DSCnet     0.316 [</td><td rowspan=1 colspan=2>0.266, 0.368]</td><td rowspan=1 colspan=1>0.705 [0</td><td rowspan=1 colspan=2>.657, 0.763]</td><td rowspan=1 colspan=1>0.161 [0</td><td rowspan=1 colspan=1>.095, 0.233]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>cIDice</td><td rowspan=1 colspan=1>0.318 [</td><td rowspan=1 colspan=2>0.273, 0.359]</td><td rowspan=1 colspan=1>0.744 [0</td><td rowspan=1 colspan=2>.688, 0.803]</td><td rowspan=1 colspan=1>0.148 [0</td><td rowspan=1 colspan=1>.094, 0.214]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>nn-UNet</td><td rowspan=1 colspan=1>0.324 [</td><td rowspan=1 colspan=2>0.291, 0.353]</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=2>[0.682, 0.804]</td><td rowspan=1 colspan=1>0.164 [</td><td rowspan=1 colspan=1>0.110, 0.202]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>IVNet</td><td rowspan=1 colspan=1>0.269 [</td><td rowspan=1 colspan=2>0.235, 0.293]</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=2>[0.571, 0.723]</td><td rowspan=1 colspan=1>0.118[</td><td rowspan=1 colspan=1>0.080, 0.154]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Kimimaro</td><td rowspan=1 colspan=1>MedNeXt</td><td rowspan=1 colspan=1>0.214 [</td><td rowspan=1 colspan=2>0.184, 0.254]</td><td rowspan=1 colspan=1>0.587 [0</td><td rowspan=1 colspan=2>.524, 0.635]</td><td rowspan=1 colspan=1>0.060[</td><td rowspan=1 colspan=1>0.032, 0.114]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SegMamba</td><td rowspan=1 colspan=1>0.351 [0</td><td rowspan=1 colspan=1>.277,</td><td rowspan=1 colspan=1>0.402]</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.512</td><td rowspan=1 colspan=1>0.708</td><td rowspan=1 colspan=1>0.241[</td><td rowspan=1 colspan=1>0.160, 0.303]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SwinUNETR</td><td rowspan=1 colspan=1>0.275 [0</td><td rowspan=1 colspan=1>.248, 0</td><td rowspan=1 colspan=1>.306]</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=1>0.615</td><td rowspan=1 colspan=1>0.701</td><td rowspan=1 colspan=1>0.128[</td><td rowspan=1 colspan=1>0.085, 0.170]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>UNETR</td><td rowspan=1 colspan=1>0.188 [</td><td rowspan=1 colspan=1>0.159, 0</td><td rowspan=1 colspan=1>.223]</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.031[</td><td rowspan=1 colspan=1>0.015, 0.074]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>VNet</td><td rowspan=1 colspan=1>0.264 [</td><td rowspan=1 colspan=2>0.240, 0.293]</td><td rowspan=1 colspan=1>0.637 [0</td><td rowspan=1 colspan=2>.576, 0.707]</td><td rowspan=1 colspan=1>0.116[</td><td rowspan=1 colspan=1>0.071, 0.162]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>DSCnet</td><td rowspan=1 colspan=1>0.239 [</td><td rowspan=1 colspan=2>0.206, 0.278]</td><td rowspan=1 colspan=1>0.416 [0</td><td rowspan=1 colspan=2>.361, 0.476]</td><td rowspan=1 colspan=1>0.135</td><td rowspan=1 colspan=1>[0.084, 0.201]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>cIDice</td><td rowspan=1 colspan=1>0.289 [</td><td rowspan=1 colspan=2>0.233, 0.393]</td><td rowspan=1 colspan=1>0.404</td><td rowspan=1 colspan=2>[0.309, 0.519]</td><td rowspan=1 colspan=1>0.217[</td><td rowspan=1 colspan=1>0.124, 0.379]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>nn-UNet</td><td rowspan=1 colspan=1>0.252 [</td><td rowspan=1 colspan=2>0.191, 0.312]</td><td rowspan=1 colspan=1>0.349 </td><td rowspan=1 colspan=2>[0.245, 0.468]</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=1>[0.126, 0.258]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>IVNet      0.267 [</td><td rowspan=1 colspan=2>0.213, 0.331]</td><td rowspan=1 colspan=1>0.448 </td><td rowspan=1 colspan=2>[0.290, 0.558]</td><td rowspan=1 colspan=1>0.169</td><td rowspan=1 colspan=1>[0.093, 0.261]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>neuTube   MedNeXt   0.308 [</td><td rowspan=1 colspan=2>0.247, 0.394]</td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1>0.261.</td><td rowspan=1 colspan=1>0.580</td><td rowspan=1 colspan=1>0.232</td><td rowspan=1 colspan=1>[0.135, 0.355]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>SegMamba</td><td rowspan=1 colspan=1>0.252 [0</td><td rowspan=1 colspan=2>.213, 0.299]</td><td rowspan=1 colspan=1>0.530 </td><td rowspan=1 colspan=1>0.444.</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.118[</td><td rowspan=1 colspan=1>0.062, 0.179]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>SwinUNETR</td><td rowspan=1 colspan=1>0.264 [0</td><td rowspan=1 colspan=2>.228, 0.313]</td><td rowspan=1 colspan=1>0.450 [0</td><td rowspan=1 colspan=1>.337,</td><td rowspan=1 colspan=1>0.578]</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>[0.083, 0.234]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>UNETR</td><td rowspan=1 colspan=1>0.183 [</td><td rowspan=1 colspan=2>0.157, 0.225]</td><td rowspan=1 colspan=1>0.288</td><td rowspan=1 colspan=1>0.257</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.093[</td><td rowspan=1 colspan=1>0.064, 0.171]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>VNet       0.238 [</td><td rowspan=1 colspan=2>0.202, 0.290]</td><td rowspan=1 colspan=1>0.552</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.624</td><td rowspan=1 colspan=1>0.083 [0</td><td rowspan=1 colspan=1>.047, 0.150]</td><td rowspan=1 colspan=1></td></tr></table>

## D.4 Parameter Sensitivity of Fiber Metric

We evaluated top 18 methods (ranked by performance at the threshold of 0.75) across IoU thresholds η. The results are presented in Table A.4. The performance of all methods generally decreases as the IoU threshold increases, which is expected since stricter overlap requirements make it more challenging to achieve high scores. Notably, the relative ranking of methods remains largely consistent across different thresholds, indicating that the evaluation is robust to variations in η.

Table A.4: Performance comparison under different IoU thresholds η. The best results are shown in bold, and the second-best results are underlined.
<table><tr><td>Method</td><td>0.50</td><td>0.55</td><td>0.60</td><td>0.65</td><td>0.70</td><td>0.75</td><td>0.80</td><td>0.85</td><td>0.90</td><td>0.95</td></tr><tr><td>SegMamba+Kimimaro</td><td>0.4024</td><td>0.3934</td><td>0.3818</td><td>0.3738</td><td>0.3638</td><td>0.3495</td><td>0.3405</td><td>0.3262</td><td>0.3124</td><td>0.2928</td></tr><tr><td>SwinUNETR+APP2</td><td>0.4016</td><td>0.3905</td><td>0.3781</td><td>0.3631</td><td>0.3432</td><td>0.3285</td><td>0.3125</td><td>0.2975</td><td>0.2730</td><td>0.2567</td></tr><tr><td>nn-UNet+Kimimaro</td><td>0.3626</td><td>0.3551</td><td>0.3487</td><td>0.3422</td><td>0.3326</td><td>0.3230</td><td>0.3171</td><td>0.3096</td><td>0.3005</td><td>0.2904</td></tr><tr><td>clDice+Kimimaro</td><td>0.3520</td><td>0.3445</td><td>0.3390</td><td>0.3331</td><td>0.3276</td><td>0.3200</td><td>0.3130</td><td>0.3087</td><td>0.3038</td><td>0.2924</td></tr><tr><td>DSCNet+Kimimaro</td><td>0.3606</td><td>0.3515</td><td>0.3456</td><td>0.3348</td><td>0.3267</td><td>0.3159</td><td>0.3094</td><td>0.3013</td><td>0.2927</td><td>0.2814</td></tr><tr><td>MedNeXt+neuTube</td><td>0.3721</td><td>0.3610</td><td>0.3466</td><td>0.3356</td><td>0.3203</td><td>0.3066</td><td>0.2952</td><td>0.2834</td><td>0.2690</td><td>0.2484</td></tr><tr><td>SegMamba+APP2</td><td>0.3953</td><td>0.3810</td><td>0.3635</td><td>0.3433</td><td>0.3252</td><td>0.3044</td><td>0.2876</td><td>0.2722</td><td>0.2554</td><td>0.2456</td></tr><tr><td>IVNet+APP2</td><td>0.3667</td><td>0.3555</td><td>0.3432</td><td>0.3296</td><td>0.3110</td><td>0.2960</td><td>0.2736</td><td>0.2485</td><td>0.2373</td><td>0.2262</td></tr><tr><td>clDice+neuTube</td><td>0.3428</td><td>0.3323</td><td>0.3225</td><td>0.3124</td><td>0.3007</td><td>0.2886</td><td>0.2781</td><td>0.2633</td><td>0.2500</td><td>0.2332</td></tr><tr><td>clDice+APP2</td><td>0.3587</td><td>0.3458</td><td>0.3355</td><td>0.3190</td><td>0.3013</td><td>0.2838</td><td>0.2700</td><td>0.2580</td><td>0.2432</td><td>0.2234</td></tr><tr><td>SwinUNETR+Kimimaro</td><td>0.3150</td><td>0.3060</td><td>0.2998</td><td>0.2936</td><td>0.2845</td><td>0.2749</td><td>0.2698</td><td>0.2619</td><td>0.2500</td><td>0.2359</td></tr><tr><td>IVNet+Kimimaro</td><td>0.3116</td><td>0.2996</td><td>0.2921</td><td>0.2852</td><td>0.2766</td><td>0.2686</td><td>0.2605</td><td>0.2554</td><td>0.2439</td><td>0.2341</td></tr><tr><td>nn-UNet+APP2</td><td>0.3398</td><td>0.3284</td><td>0.3164</td><td>0.3001</td><td>0.2849</td><td>0.2683</td><td>0.2474</td><td>0.2182</td><td>0.2057</td><td>0.1934</td></tr><tr><td>IVNet+neuTube</td><td>0.3151</td><td>0.3068</td><td>0.2982</td><td>0.2903</td><td>0.2778</td><td>0.2666</td><td>0.2570</td><td>0.2451</td><td>0.2297</td><td>0.2150</td></tr><tr><td>VNet+Kimimaro</td><td>0.2976</td><td>0.2936</td><td>0.2826</td><td>0.2785</td><td>0.2716</td><td>0.2640</td><td>0.2588</td><td>0.2542</td><td>0.2490</td><td>0.2403</td></tr><tr><td>SwinUNETR+neuTube</td><td>0.3187</td><td>0.3088</td><td>0.2993</td><td>0.2891</td><td>0.2762</td><td>0.2632</td><td>0.2492</td><td>0.2378</td><td>0.2249</td><td>0.2047</td></tr><tr><td>VNet+APP2</td><td>0.3227</td><td>0.3140</td><td>0.3054</td><td>0.2902</td><td>0.2759</td><td>0.2617</td><td>0.2482</td><td>0.2294</td><td>0.2163</td><td>0.2068</td></tr><tr><td>nn-UNet+neuTube</td><td>0.3046</td><td>0.2953</td><td>0.2852</td><td>0.2762</td><td>0.2641</td><td>0.2530</td><td>0.2437</td><td>0.2291</td><td>0.2136</td><td>0.1966</td></tr></table>

Table A.5: Whole-brain reconstruction results on C-166 dataset. Each value is the micro-F1 estimate followed by its 95% BCa bootstrap confidence interval (10,000 resamples). The best and second-best estimates in each metric are shown in bold and underlined, respectively.
<table><tr><td rowspan=1 colspan=11>Micro-F1 (95% BCa CI)ReconstructionMethodFiber              Dendrite              Axon</td></tr><tr><td rowspan=1 colspan=2>DSCnet     0.155</td><td rowspan=1 colspan=2>[0.082, 0.243]</td><td rowspan=1 colspan=1>0.331 [0</td><td rowspan=1 colspan=2>.195, 0.484]</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=2>[0.027, 0.162]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>cIDice</td><td rowspan=1 colspan=1>0.296[</td><td rowspan=1 colspan=2>0.238, 0.379]</td><td rowspan=1 colspan=1>0.532[</td><td rowspan=1 colspan=2>0.388, 0.640]</td><td rowspan=1 colspan=1>0.174[</td><td rowspan=1 colspan=2>0.088, 0.277]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>nn-UNet</td><td rowspan=1 colspan=1>0.285[</td><td rowspan=1 colspan=2>0.230, 0.379]</td><td rowspan=1 colspan=1>0.535</td><td rowspan=1 colspan=2>[0.361, 0.649]</td><td rowspan=1 colspan=1>0.162[</td><td rowspan=1 colspan=2>0.087, 0.272]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=1>IVNetAPP2MedNeXtSegMambaSwinUNETRVNet</td><td rowspan=1 colspan=1>0.320 [0</td><td rowspan=1 colspan=2>.259, 0.420]</td><td rowspan=1 colspan=1>0.505 [0</td><td rowspan=1 colspan=2>.344, 0.637]</td><td rowspan=1 colspan=1>0.216[</td><td rowspan=1 colspan=2>0.114, 0.352]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=2>[0.008, 0.045]</td><td rowspan=1 colspan=1>0.072 [[</td><td rowspan=1 colspan=2>0.030, 0.128]</td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=2>[0.000, 0.009]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.263[</td><td rowspan=1 colspan=2>0.217, 0.332]</td><td rowspan=1 colspan=1>0.506[</td><td rowspan=1 colspan=2>0.395, 0.591]</td><td rowspan=1 colspan=1>0.168</td><td rowspan=1 colspan=2>[0.086, 0.260]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.211[</td><td rowspan=1 colspan=2>0.150, 0.291]</td><td rowspan=1 colspan=1>0.232</td><td rowspan=1 colspan=2>[0.181, 0.296]</td><td rowspan=1 colspan=1>0.192</td><td rowspan=1 colspan=2>[0.096, 0.323]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.253</td><td rowspan=1 colspan=2>[0.203, 0.346]</td><td rowspan=1 colspan=1>0.500 [0</td><td rowspan=1 colspan=2>.318, 0.642]</td><td rowspan=1 colspan=1>0.104</td><td rowspan=1 colspan=2>[0.036, 0.267]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>DSCnet     0.050</td><td rowspan=1 colspan=2>[0.019, 0.118]</td><td rowspan=1 colspan=1>0.160 [0</td><td rowspan=1 colspan=2>.071, 0.319]</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=2>[0.001, 0.015]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>cIDice</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=2>[0.170, 0.293]</td><td rowspan=1 colspan=1>0.578 [0</td><td rowspan=1 colspan=1>.506,</td><td rowspan=1 colspan=1>0.668]</td><td rowspan=1 colspan=1>0.071[</td><td rowspan=1 colspan=2>0.035, 0.149]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>nn-UNet</td><td rowspan=1 colspan=1>0.143 [</td><td rowspan=1 colspan=2>0.103, 0.206]</td><td rowspan=1 colspan=1>0.392 [0</td><td rowspan=1 colspan=1>.295, 0</td><td rowspan=1 colspan=1>.488]</td><td rowspan=1 colspan=1>0.038[</td><td rowspan=1 colspan=2>0.019, 0.091]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>IVNetKimimaroMedNeXt</td><td rowspan=1 colspan=1>0.218 [</td><td rowspan=1 colspan=2>0.164, 0.326]</td><td rowspan=1 colspan=1>0.559[</td><td rowspan=1 colspan=1>0.483, 0</td><td rowspan=1 colspan=1>.647]</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=2>[0.034, 0.169]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.194</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.272</td><td rowspan=1 colspan=1>0.512[</td><td rowspan=1 colspan=1>0.462,</td><td rowspan=1 colspan=1>0.560]</td><td rowspan=1 colspan=1>0.058[</td><td rowspan=1 colspan=2>0.017, 0.138]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SegMamba</td><td rowspan=1 colspan=1>0.154</td><td rowspan=1 colspan=1>0.106.</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.370[</td><td rowspan=1 colspan=1>0.277, 0</td><td rowspan=1 colspan=1>.540]</td><td rowspan=1 colspan=1>0.064[</td><td rowspan=1 colspan=2>0.031, 0.128]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SwinUNETR</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.120.</td><td rowspan=1 colspan=1>0.234</td><td rowspan=1 colspan=1>0.478</td><td rowspan=1 colspan=2>[0.399, 0.553]</td><td rowspan=1 colspan=1>0.016[</td><td rowspan=1 colspan=1>0.005,</td><td rowspan=1 colspan=1>0.034]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>VNet</td><td rowspan=1 colspan=1>0.153[</td><td rowspan=1 colspan=1>0.102,</td><td rowspan=1 colspan=1>0.264]</td><td rowspan=1 colspan=1>0.385[</td><td rowspan=1 colspan=2>0.298, 0.550]</td><td rowspan=1 colspan=1>0.058[</td><td rowspan=1 colspan=1>0.025,</td><td rowspan=1 colspan=1>0.132]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>DSCnet</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=2>[0.131, 0.270]</td><td rowspan=1 colspan=1>0.456 [0</td><td rowspan=1 colspan=2>.353, 0.551]</td><td rowspan=1 colspan=1>0.081[</td><td rowspan=1 colspan=2>0.028, 0.160]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>cIDice</td><td rowspan=1 colspan=1>0.258</td><td rowspan=1 colspan=2>[0.201, 0.365]</td><td rowspan=1 colspan=1>0.443 [0</td><td rowspan=1 colspan=1>.342,</td><td rowspan=1 colspan=1>0.531]</td><td rowspan=1 colspan=1>0.145</td><td rowspan=1 colspan=2>[0.056, 0.303]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>nn-UNet</td><td rowspan=1 colspan=1>0.230 [</td><td rowspan=1 colspan=2>0.160, 0.317]</td><td rowspan=1 colspan=1>0.374 [0</td><td rowspan=1 colspan=1>.208,</td><td rowspan=1 colspan=1>0.550]</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=2>[0.041, 0.211]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>IVNetneuTubeMedNeXt</td><td rowspan=1 colspan=1>0.221</td><td rowspan=1 colspan=2>[0.167, 0.304]</td><td rowspan=1 colspan=1>0.365 [0</td><td rowspan=1 colspan=1>.275,</td><td rowspan=1 colspan=1>0.499]</td><td rowspan=1 colspan=1>0.086</td><td rowspan=1 colspan=2>[0.029, 0.217]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.291</td><td rowspan=1 colspan=2>[0.201, 0.437]</td><td rowspan=1 colspan=1>0.366 [0</td><td rowspan=1 colspan=2>.241, 0.521]</td><td rowspan=1 colspan=1>0.240[</td><td rowspan=1 colspan=2>0.089, 0.450]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SegMamba</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=2>[0.205, 0.405]</td><td rowspan=1 colspan=1>0.425 [0</td><td rowspan=1 colspan=1>.335,</td><td rowspan=1 colspan=1>0.549]</td><td rowspan=1 colspan=1>0.202[</td><td rowspan=1 colspan=2>0.105, 0.371]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SwinUNETR</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=2>[0.013, 0.030]</td><td rowspan=1 colspan=1>0.021 [0</td><td rowspan=1 colspan=1>.013,</td><td rowspan=1 colspan=1>0.041]</td><td rowspan=1 colspan=1>0.016[</td><td rowspan=1 colspan=2>0.001, 0.028]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>VNet</td><td rowspan=1 colspan=1>0.255 [</td><td rowspan=1 colspan=2>0.198, 0.345]</td><td rowspan=1 colspan=1>0.416 [0</td><td rowspan=1 colspan=1>.301,</td><td rowspan=1 colspan=1>0.533]</td><td rowspan=1 colspan=1>0.137 [0</td><td rowspan=1 colspan=2>.055, 0.293]</td><td rowspan=1 colspan=1></td></tr></table>

## D.5 Generalization Performance on the Out-of-domain Test Set

Table A.5 presents the performance of different models on the out-of-domain test set. C-166 is an independent whole-brain sample prepared and annotated following the same procedures and was used as the out-of-domain test set. It contains 15 neurons. All models were trained on the CORAL dataset and evaluated on C-166.

## D.6 Performance of Individual Human Annotators

To evaluate individual human performance in neuron annotation, we recruited five professional annotators. Each annotator was randomly assigned five neurons for annotation. Table A.6 reports their annotation performance for dendrites, axons, and the overall (Fiber) neuronal structure, along with the mean and standard deviation across the five annotators. The results show relatively small inter-annotator variation, which is substantially lower than the performance gap between current state-of-the-art methods and human annotators. This indicates that existing algorithms remain far from achieving human-level performance.

Table A.6: Performance of Individual Human Annotators.
<table><tr><td>Human</td><td colspan="3">Dendrite</td><td colspan="3">Axon</td><td colspan="3">Fiber</td></tr><tr><td></td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>human1</td><td>0.897</td><td>0.912</td><td>0.901</td><td>0.746</td><td>0.674</td><td>0.708</td><td>0.958</td><td>0.843</td><td>0.895</td></tr><tr><td>human2</td><td>0.993</td><td>0.977</td><td>0.984</td><td>0.935</td><td>0.785</td><td>0.849</td><td>0.946</td><td>0.820</td><td>0.875</td></tr><tr><td>human3</td><td>0.845</td><td>0.958</td><td>0.857</td><td>0.758</td><td>0.685</td><td>0.720</td><td>0.967</td><td>0.828</td><td>0.887</td></tr><tr><td>human4</td><td>0.976</td><td>0.948</td><td>0.961</td><td>0.966</td><td>0.810</td><td>0.880</td><td>0.970</td><td>0.840</td><td>0.900</td></tr><tr><td>human5</td><td>0.853</td><td>0.936</td><td>0.867</td><td>0.758</td><td>0.625</td><td>0.682</td><td>0.965</td><td>0.746</td><td>0.830</td></tr><tr><td>Meanstd</td><td>0.9130.069</td><td>0.9460.025</td><td>0.9140.057</td><td>0.8320.108</td><td>0.7160.078</td><td>0.7680.090</td><td>0.9610.010</td><td>0.8160.040</td><td>0.8770.028</td></tr></table>

## E Visualization of Bad Case in Brain-wide

Figure 8 illustrates the impact of break and merge errors during whole-brain reconstruction. A break error in an axon causes multiple downstream neurites to be missed. Merge Error I leads the tracing process back toward the soma in the opposite direction, resulting in numerous false-positive predictions while leaving the affected region untraced. Merge Error II causes the reconstruction to erroneously trace into another neuron, producing a large number of false-positive predictions.

![](images/0ef17b06bacc55b9d63c7cc6706b0ddc3e79453d1beddff67b78e733f1722b47.jpg)  
Figure 8: Visualization of reconstruction errors in brain-wide. The ground-truth annotations of neuron 1 and neuron 2 are shown in green and blue, respectively, while the prediction for neuron 1 is shown in red. The arrows along the tubular structures indicate the fiber direction.

## F Limitations

Despite these strengths, CORAL has several limitations. The current benchmark is built from a limited number of fully annotated whole-brain neurons and therefore does not yet cover the full diversity of neuronal morphologies and imaging conditions. Moreover, all data are derived from mice, and the benchmark has not yet been evaluated on other species; thus, its cross-species generalizability remains to be established. We view CORAL as a first step toward standardized brain-wide evaluation and expect future extensions to broaden data diversity, protocol coverage, and evaluation criteria.