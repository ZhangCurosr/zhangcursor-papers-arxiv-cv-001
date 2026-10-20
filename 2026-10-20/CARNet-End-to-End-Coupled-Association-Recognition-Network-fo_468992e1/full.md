# CARNet: End-to-End Coupled Association-Recognition Network for Fine-Grained Multi-Drone Collaborative Detection

Zhao Zhang<sup>a</sup>, Ruigang Fu<sup>a,∗</sup>, Yu Zhang<sup>a</sup>, Jiajun Xiang<sup>a</sup>, Zixuan Xiao<sup>a</sup>, Ping Zhong<sup>a</sup>

<sup>a</sup>College of Electronic Science and Technology, National University of Defense Technology, Changsha, China

## Abstract

Multi-drone collaborative detection aims to enhance target recognition by leveraging complementary observations from overlapping aerial viewpoints. Existing approaches typically follow a two-stage pipeline that first establishes crossview correspondences and then performs fusion-based recognition. However, this decomposition sufers from inconsistent optimization objectives, redundant pairwise modeling, and irreversible error propagation, where incorrect associations inevitably undermine subsequent recognition performance. To overcome these limitations, we propose the Coupled Association-Recognition Network (CAR-Net), a unified framework that jointly optimizes cross-view target association and recognition fusion through a shared target-pair representation. To improve association reliability in densely populated scenes with visually similar targets, CARNet explicitly incorporates spatial cues into pairwise relation modeling. Moreover, we introduce two complementary causal-inspired intervention strategies: the first enhances cross-view association by emphasizing appearance-confusing pairs, while the second bolsters the robustness of fusion-based recognition by learning stable multi-view representations under incomplete visual evidence. In addition, we present MDMT-CD, a fine-grained extension of the MDMT dataset that enriches the original vehicle annotations with detailed category labels, providing the first benchmark specifically designed for multi-drone collaborative vehicle detection. Extensive experiments demonstrate that CARNet consistently outperforms existing state-of-the-art methods. The MDMT-CD dataset and source code will be

available at https://github.com/bigdreaming11/CARNet.

Keywords: Multi-drone collaborative detection, Cross-view target association, Fine-grained object fusion recognition, Causal-inspired intervention

## 1. Introduction

Single-drone object detection is prone to classification errors and struggles to handle challenging scenarios such as occlusion [1]. In contrast, multi-drone systems enable collaborative observation of the same target from diferent aerial viewpoints, providing complementary structural and appearance information [2].

A natural implementation is a two-stage pipeline: an association network first establishes correspondences, and a subsequent fusion classifier recognizes each matched target [3]. However, the two stages are optimized with heterogeneous objectives, leading to inconsistencies in representation learning and redundant pairwise modeling across modules. Moreover, the cascade design introduces a structural bottleneck where association errors are irrevocably propagated to the recognition stage [4]. Once the first stage incorrectly links two diferent vehicles, subsequent feature fusion inevitably aggregates conflicting evidence, making the final prediction unrecoverable. This design is also computationally redundant and ineficient due to duplicated pairwise modeling across stages. This motivates a unified formulation in which target association and fusion recognition are learned through the same target-pair representation.

We address this bottleneck with a position-aware all-pairs relation. For every target in one drone view and every target in the other, we construct a fused relation by combining their cross-view visual descriptors. LGSGMM [5]demonstrat , positional information provides valuable geometric cues that improve feature discriminability and correspondence estimation under appearance ambiguity. Motivated by this observation, we encode each target center using a learnable Fourier embedding and inject the resulting positional representation into the fused relation, allowing it to jointly model visual appearance and spatial consistency across views. Each fused target-pair relation is mapped to a C + 1 distribution containing C fine-grained vehicle categories and one no-match state. Consequently, the relation tensor serves as a unified target-pair representation that jointly performs cross-view target association and fusing recognition within a single prediction stage, eliminating the conventional separation between association and fusion-based recognition. This unified formulation enables CARNet to replace the conventional association network and fusion recognition with a single prediction stage.

However, even with the unified target-pair modeling scheme, association learning may still struggle to capture discriminative correspondence evidence from challenging target pairs. Existing supervision paradigms typically optimize pairwise relations based on matching labels, while overlooking the varying discriminative dificulty among diferent target pairs. Since easy negative pairs provide straightforward optimization signals, the model tends to prioritize learning easily separable appearance patterns, whereas the fine-grained identity cues and critical correspondence evidence embedded in highly similar target pairs remain insuficiently explored, resulting in incorrect associations in complex scenarios.

Motivated by Pearl’s causal hierarchy[6], CARNet introduces two complementary intervention strategies to guide the model toward learning stable crossview correspondence evidence. The first enhances cross-view correspondence by assigning greater learning emphasis to appearance-confusing non-matching target pairs, encouraging the model to devote more optimization capacity to challenging correspondence decisions. The second intervention perturbs the visual evidence within target pairs and enforces consistency between the original and intervened relation representations, encouraging the model to learn stable correspondence representations that remain robust under visual variations and incomplete observations.

Our contributions are summarized as follows:

• We establish MDMT-CD, a fine-grained extension of the MDMT dataset by enriching the original single-category vehicle annotations with fine-grained category labels.

• We propose CARNet, a unified pairwise relation framework that integrates cross-view correspondence and fusion recognition in a single prediction stage.

• We propose two causal-inspired intervention strategies that improve crossview correspondence in densely distributed scenes with appearance-similar targets while enhancing the robustness of multi-view fusion recognition.

• CARNet achieves state-of-the-art collaborative detection and cross-view association performance on the MDMT-CD datasets, the best association performance on the infrared and cross-modal subsets of M3Track, and further demonstrates excellent generalization on the cross-camera multi-object tracking benchmarks Wildtrack and MvMHAT.

## 2. Related Work

## 2.1. Multi-Drone Collaborative Perception

Multi-drone collaborative perception exploits complementary observations from multiple UAVs to alleviate limited fields of view, occlusion, small object scales, and unstable imaging conditions. Early studies mainly focused on collaborative tracking. Zhu et al. [7] proposed an agent-sharing network that exchanges reliable target templates and performs view-aware fusion for multidrone single-object tracking. Liu et al. [8] introduced the MDMT benchmark and MIA-Net, which combine within-view tracking with cross-drone identity authentication to recover targets occluded in individual views. Wu et al. [9] further enhanced multi-drone multi-object tracking through temporal–spatial feature interaction. Beyond UAV-specific settings, Nguyen et al. [10] formulated multi-camera multi-object tracking as a single-stage global association problem, jointly reasoning about detections and cross-camera correspondences in overlapping views. Zheng et al. [11] further improved multi-object association under occlusion through motion-guided cross-temporal aggregation and hierarchical matching.

Recent research has increasingly investigated feature-level collaborative detection. DiscoNet [12] learns a collaboration graph under teacher–student distillation, whereas Where2comm [13] communicates only spatial regions with high perceptual confidence. V2X-ViT [14], CoBEVT [15] and CoBEVFlow [16] further address heterogeneous observations, sparse cross-agent fusion, communication redundancy, and temporal asynchrony. Although several of these methods were initially developed for vehicle-based agents, their feature alignment and communication mechanisms are also applicable to UAV collaboration.

For UAV-specific three-dimensional perception, UAV3D [17] provides benchmarks for collaborative 3D detection and tracking, while UCDNet [18] maps multi-UAV image features into a common 3D space using depth priors and geometric consistency. MCOP [19] extends collaboration from bounding-box detection to dense occupancy prediction with altitude-aware compression. V2U4Real [20] studies real-world vehicle-to-UAV cooperative detection, and AirCopBench [21] evaluates multi-drone embodied perception and reasoning. Nevertheless, most existing methods output trajectories, boxes, or occupancy states, while target-level correspondence and fine-grained collaborative recognition remain insuficiently coupled.

## 2.2. Cross-View Target Association

Cross-view target association aims to determine whether targets observed by diferent UAVs correspond to the same physical object. Compared with conventional multi-object tracking, the task is considerably more challenging due to drastic viewpoint changes, severe rotations, scale variations, target occlusions, and dense distributions of visually similar objects.

Existing UAV-oriented methods mainly exploit geometric consistency to establish cross-view correspondences. Li et al. [22] represented each target using triangular topological sequences and performed globally consistent association through two-stage assignment. Liu et al. [23] introduced the MDMT benchmark and proposed MIA-Net, which first establishes image-level correspondences using SIFT features and then associates tracked targets across UAVs through geometric verification. Qiao et al. [24] proposed STCA, which estimates inter-view transformations, identifies common visible regions, and performs bidirectional cross-drone association. Tan et al. [25] further formulated target association as a graph matching problem and exploited reliable spatial relationships to improve robustness under missed and false detections.

Besides these dedicated UAV-oriented methods, appearance-based ReID models and deep image matching networks are also widely adopted as association baselines. However, appearance-based methods such as Re-ID [26], OSNet [27], ReIDMamba [28], CrossMOT [29], CT [30], AGW [31], mainly rely on visual similarity and therefore struggle to distinguish densely distributed targets with highly similar appearances. Deep image matching methods, such as Super-

Glue [32], LoFTR [33] and Roma [34], often experience noticeable performance degradation under the large rotation diferences commonly encountered between UAV views. Moreover, MIA-Net depends on handcrafted SIFT features for image registration, resulting in relatively low computational eficiency. Furthermore, most existing methods perform target association as an independent preprocessing stage, making it dificult to jointly optimize cross-view association and downstream collaborative recognition.

## 3. Fine-Grained MDMT Annotation

The original MDMT dataset assigns all vehicle targets to a single category, car, which limits its capability for fine-grained semantic analysis. In this work, we extend MDMT by retaining the original cross-view identity annotations and further refining the object labels into seven fine-grained categories: car, bus, van, truck, tricycle, bicycle, and pedestrian. This extension enables more comprehensive evaluation of multi-view target perception and association under diverse object types. In addition, we manually verify the entire dataset and correct annotation errors from the original release to improve annotation reliability. Each annotation record consists of the object category, target identity (ID), and bounding box information.

## 4. Method

Given a pair of UAV images $( I ^ { A } , I ^ { B } )$ , the onboard object detector first extracts M target instances from $I ^ { A }$ and N target instances from $I ^ { B } ,$ . Conventional collaborative detection methods generally decompose the task into two successive stages. First, cross-view object association is performed to establish correspondences between targets observed from diferent UAVs. Then, multi-view feature fusion is conducted for the matched target pairs to obtain the final fused recognition results.

In contrast, we formulate object association and multi-view feature fusion as a unified end-to-end framework, enabling one-stage collaborative object detection for multi-drone systems. The overall pipeline of the proposed method is illustrated in Fig. 1.

![](images/d693acc15d2a13a3de7daa989390308b445f57e5f2010666699c6634f83e6a64.jpg)  
Figure 1: Overall pipeline of the proposed CARNet. Given two drone views, target features are extracted and used to construct position-aware all-pairs relations. Each target-pair relation is mapped to a unified $C + 1$ distribution for joint cross-view association and collaborative recognition, while two causal-inspired intervention are applied during training.

## 4.1. Position-Aware Pairwise Relation Modeling

Fistly, a shared backbone is employed to process both UAV images. Subsequently, ROI pooling extracts one feature vector for each detected target, resulting in the object feature sets $Z ^ { A } \in \mathbb { R } ^ { M \times D } , Z ^ { B } \in \mathbb { R } ^ { N \times D }$ . When visually similar targets are densely distributed, appearance features alone are often insuficient to distinguish diferent instances. To alleviate this ambiguity, we introduce positional information to facilitate cross-view association. Specifically, following the use of learned positional encodings in modern feature matching [32], the center coordinates of each target are mapped by a learnable Fourier encoder, yielding a positional feature $p \in \mathbb { R } ^ { D }$ which has the same dimensionality as the visual descriptor. For each cross-view candidate pair $( i , j )$ , we construct a position-aware pairwise relation representation by jointly encoding the appearance descriptors and positional information of the object pair:

$$
r _ { i j } = \left[ z _ { i } ^ { A } , z _ { j } ^ { B } , p _ { i } ^ { A } , p _ { j } ^ { B } \right]\tag{1}
$$

Together with the Fourier positional diference, the relation representation supplies complementary evidence for association and collaborative detection. Thus, the pairwise relation matrix between the two views is obtained as:

$$
r \in \mathbb { R } ^ { M \times N \times { 4 D } }\tag{2}
$$

An MLP maps each relation feature $r _ { i j }$ to a $( C + 1 )$ -dimensional relation dis-

tribution:

$$
\pi _ { i j } = [ \pi _ { i j , 1 } , \dots , \pi _ { i j , C } , \pi _ { i j , \emptyset } ] .\tag{3}
$$

Here, C is the number of fine-grained target categories. $\pi _ { i j , c }$ denotes the probability that pair (i j) is a valid correspondence with category c, where $c \in$ $\{ 1 , \ldots , C \} . \ \pi _ { i j , \emptyset }$ denotes the probability that pair (i j) belongs to the no-match class. The final decoder uses a fast bidirectional best-match strategy to determine valid correspondences. Only accepted target pairs are used to update the singleview detection categories.

## 4.2. Hard-Distinguishing Intervention

Dense scenes contain many visually similar vehicles. This may create an appearance shortcut: visually similar targets are treated as matched targets. Although the pairwise loss supervises all valid target pairs, the optimization contribution of dificult relations remains limited because the relation distribution is highly imbalanced and dominated by easy no-match pairs. To suppress this shortcut, we introduce a hard non-match intervention. This intervention does not generate new training pairs. Instead, it selects appearance-confusing no-match pairs from existing all-pairs relations and imposes additional no-match supervision on them.

Let ${ { \mathcal { P } } ^ { + } }$ denote the set of ground-truth matched target pairs. For each target i in view A, we construct a candidate non-match set by excluding its ground-truth correspondence:

$$
{ N } _ { i } = \{ j \mid ( i , j ) \notin \mathcal { P } ^ { + } \} .\tag{4}
$$

Here, j indexes a target in view B. $N _ { i }$ contains all valid targets in view B that should not be matched with target i.

The visual descriptors are first normalized. Then, the most confusing nonmatch target is selected according to cosine similarity:

$$
j _ { i } ^ { - } = \arg \operatorname* { m a x } _ { j \in N _ { i } } \frac { z _ { i } ^ { A } \cdot z _ { j } ^ { B } } { \| z _ { i } ^ { A } \| _ { 2 } \| z _ { j } ^ { B } \| _ { 2 } } .\tag{5}
$$

where · denotes inner product and $\| \cdot \| _ { 2 }$ denotes the $\ell _ { 2 }$ norm. The selected target $j _ { i } ^ { - }$ is the no-match target that is most similar to i in the descriptor space.

The intervention set is defined as:

$$
{ \mathcal { H } } = \{ ( i , j _ { i } ^ { - } ) \mid N _ { i } \neq \emptyset \} .\tag{6}
$$

Where H contains the selected hard non-match pairs. Each pair in $\mathcal { H }$ is a true negative pair, but it has high visual similarity. For these intervened pairs, the model is encouraged to classify them as unmatched:

$$
\mathcal { L } _ { \mathrm { h d i } } = - \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { ( i , j ) \in \mathcal { H } } \log \pi _ { i j , \emptyset } .\tag{7}
$$

Where $| \mathcal { H } |$ is the number of selected Hard-Distinguishing pairs. The selected pairs in H are already included in $_ \mathrm { ~ \textit ~ { ~ N ~ } ~ }$ and are therefore also supervised by ${ \mathcal { L } } _ { \mathrm { p a i r } }$ . The role of $\mathcal { L } _ { \mathrm { h d i } }$ is to increase the optimization weight of these Hard-Distinguishing pairs.

## 4.3. Visual-Evidence Perturbation Intervention

The second intervention acts on visual evidence. Cross-view observations are often incomplete because of viewpoint changes, occlusion, illumination variation, and scale changes. To simulate this condition, independent dropout masks are applied to target descriptors:

$$
\widetilde { z } _ { i } ^ { A } = m _ { i } ^ { A } \odot z _ { i } ^ { A } , \quad \widetilde { z } _ { j } ^ { B } = m _ { j } ^ { B } \odot z _ { j } ^ { B } .\tag{8}
$$

Where $m _ { i } ^ { A }$ and $m _ { j } ^ { B }$ are independently sampled dropout masks for the two views. $\widetilde { z } _ { i } ^ { A }$ and $\widetilde { z } _ { j } ^ { B }$ are the perturbed descriptors after partial removal of visual evidence.

The intervened relation representation $\widetilde { r } _ { i j }$ is reconstructed using the same relation module as the clean relation $r _ { i j }$ . The position embeddings are not dropped. Thus, the intervention is restricted to the visual descriptors. A consistency constraint is imposed between the original and intervened relations:

$$
\mathcal { L } _ { \mathrm { v e i } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( i , j ) \in \mathcal { V } } \Vert \widetilde { r } _ { i j } - \mathrm { s g } ( r _ { i j } ) \Vert _ { 2 } ^ { 2 } .\tag{9}
$$

where $_ \mathrm { ~ \textit ~ { ~ N ~ } ~ }$ denotes the set of valid target pairs used for relation consistency, and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operation. The clean relation is treated as a fixed target. The consistency constraint encourages the relation representation to preserve reliable correspondence information under visual perturbations.

## 4.4. Training Objective

The basic supervision is imposed on all valid target pairs. For each pair $( i , j ) \in$ $\mathcal { V } ,$ we define its relation label as:

$$
y _ { i j } = \left\{ { \begin{array} { l l } { c _ { i } , } & { ( i , j ) \in { \mathcal { P } } ^ { + } , } \\ { \emptyset , } & { ( i , j ) \not \in { \mathcal { P } } ^ { + } , } \end{array} } \right.\tag{10}
$$

where $c _ { i } \in \{ 1 , . . . , C \}$ is the fine-grained category label of target i. Thus, a positive pair is supervised by its fine-grained category, while an unmatched pair is supervised as no-match. The all-pairs relation classification loss is defined as

$$
\mathcal { L } _ { \mathrm { p a i r } } = - \frac { 1 } { \vert \mathcal { V } \vert } \sum _ { ( i , j ) \in \mathcal { V } } \log \pi _ { i j , y _ { i j } } .\tag{11}
$$

This loss optimizes all valid target pairs, where matched pairs are supervised by fine-grained categories and unmatched pairs are supervised by the no-match state.

The overall objective is

$$
{ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { p a i r } } + \lambda _ { 1 } { \mathcal { L } } _ { \mathrm { h d i } } + \lambda _ { 2 } { \mathcal { L } } _ { \mathrm { v e i } } .\tag{12}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are learnable scalar parameters that control the contributions of the two intervention losses. After training, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ converge to approximately 0.027 and 0.042, respectively.

## 5. Experiments

## 5.1. Datasets and Experimental Settings

We evaluate the proposed method on four benchmarks. MDMT-CD is used for multi-drone collaborative detection and target association, while M3Track are used for cross-view target association. To further examine generalization, we also conduct experiments on two cross-camera multi-person tracking benchmarks, Wildtrack and MvMHAT.

MDMT-CD contains small-scale aerial targets, dense appearance-similar objects, severe occlusions, and large viewpoint variations, making cross-view association challenging. M3Track contains low-altitude UAV and ground-camera views, including both visible-spectrum and infrared sequences, and is used to evaluate association under modality and viewpoint changes. The image resolutions of MDMT and M3Track are 1920×1080 and 640×512, respectively.

For collaborative detection, we report mAP@50 and mAP@50–90. For association on MDMT-CD and M3Track, we follow the MDMT protocol and report MDA. For Wildtrack and MvMHAT, we report MOTA and IDF1. All experiments are conducted on two NVIDIA RTX 4090 GPUs with a batch size of 4. YOLOv8 is used as the single-drone detector. For cross-camera tracking experiments, we replace the target association module of MIA-Net with CARNet while keeping the remaining components unchanged. In the association experiments, we set the number of categories C to 1.
<table><tr><td colspan="3">Method mA  $P _ { 5 0 }$   $m A P _ { 5 0 } .$  -90 Latency (ms)</td></tr><tr><td>Single Drone</td><td>23.45</td><td>16.12 一</td></tr><tr><td>UVF (AAAI 2026)</td><td>16.40</td><td>11.00 173</td></tr><tr><td>MVCNN (CVPR 2015)</td><td>17.39 11.55</td><td>88</td></tr><tr><td>CMLP</td><td>17.96</td><td>12.09 103</td></tr><tr><td>SMDC (AAAI 2023)</td><td>20.99</td><td>14.19 94</td></tr><tr><td>ARA (AAAI 2025)</td><td>21.31</td><td>14.41 97</td></tr><tr><td>DS (IGRSL 2015)</td><td>22.43</td><td>15.15 86</td></tr><tr><td>DC</td><td>23.71</td><td>15.89 83</td></tr><tr><td>TMC (TPAMI 2023)</td><td>23.67</td><td>16.33 89</td></tr><tr><td>CARNet(Ours)</td><td>25.23 17.21</td><td>48</td></tr></table>

Table 1: Comparison of collaborative detection methods on the MDMT-CD dataset. The best scores are marked in bold.

## 5.2. Comparison of Collaborative Detection Performance

Since there are currently no mature methods specifically designed for collaborative object detection in multi-drone systems, we construct a strong baseline by combining the state-of-the-art cross-view object association method LGS-GMM [5] with representative multi-view classification models, forming an "target association–fusing recognition" pipeline. Specifically, the compared fusion methods include DS [35], MVCNN [36], TMC [37], SMDC [38], ARA [39], and UVF [40]. In addition, we consider two simple yet widely adopted engineering baselines. The first selects the final category according to the detection confidence of a single drone, referred to as DC (Decision by Confidence). The second directly concatenates the features extracted from diferent views and feeds them into an MLP for classification, referred to as CMLP (Concatenation + MLP). To evaluate the efectiveness of CARNet, we compare it with the aforementioned single-view, decision-level fusion, and feature-level fusion approaches. The quantitative results are reported in Table 1, while qualitative comparisons are presented in Fig. 2.

![](images/16c2d7d9c8a15039c7c690d22465cf5023055af12d24c928f216d7aa1b167e01.jpg)  
Figure 2: Visualization of collaborative detection results on the MDMT-CD dataset. Green numbers denote single-drone predictions, and red numbers denote fused predictions. Boxes without red numbers are not associated.

As shown in Table 1, decision-level fusion methods based on single-drone logits or class probabilities, such as DS, TMC, SMDC, and ARA, consistently outperform visual feature fusion methods such as MVCNN and UVF. This is because category outputs directly provide task-relevant discriminative evidence, while evidence modeling, uncertainty estimation, and unreliable-view rejection further alleviate cross-view prediction conflicts, thereby improving multi-drone collaborative recognition accuracy. However, these methods do not fully exploit the visual and positional information of the targets. CARNet further models all M×N crossview target pairs through a unified C+1 relation distribution. By jointly exploiting visual interactions, spatial cues, and intervention-based training, it improves association among densely distributed similar vehicles and performs association, fusion, and recognition within a single framework. Compared with the single-drone baseline, CARNet improves mA $P _ { 5 0 }$ by 1.78% (25 23%−23 45%). Compared with the strongest existing collaborative detection baseline, TMC, CARNet improves mA $P _ { 5 0 }$ by 1.56% (25 23% − 23 67%) . Meanwhile, CARNet reduces the latency from 89 ms to 48 ms, demonstrating both higher accuracy and better inference

<table><tr><td rowspan="2">Method</td><td rowspan="2">(%)</td><td rowspan="2">Precision Recall MDA Latency (%)</td><td rowspan="2"></td><td rowspan="2">(ms)</td></tr><tr><td>(%)</td></tr><tr><td>ReIDMamba (TMM 2026)</td><td>30.07</td><td>28.96</td><td>14.22</td><td>107</td></tr><tr><td>Re-ID (ACM MM 2023)</td><td>31.53</td><td>28.27</td><td>15.31</td><td>18</td></tr><tr><td>OSNet (ICCV 2019)</td><td>33.11</td><td>30.13</td><td>18.11</td><td>21</td></tr><tr><td>TTS (Drones 2022)</td><td>26.97</td><td>24.57</td><td>12.06</td><td>1284</td></tr><tr><td>GTA-Net (Drones 2025)</td><td>34.64</td><td>32.31</td><td>19.27</td><td>1034</td></tr><tr><td>MambaGlue (CVPR 2025)</td><td>8.27</td><td>8.64</td><td>2.87</td><td>44</td></tr><tr><td>SuperGlue (CVPR 2020)</td><td>9.94</td><td>8.09</td><td>3.05</td><td>38</td></tr><tr><td>LoFTR (CVPR 2021)</td><td>11.24</td><td>10.01</td><td>4.07</td><td>625</td></tr><tr><td>E-LoFTR (CVPR 2024)</td><td>10.87</td><td>10.01</td><td>3.66</td><td>517</td></tr><tr><td>JamMa (CVPR 2025)</td><td>16.77</td><td>14.21</td><td>5.94</td><td>633</td></tr><tr><td>RoMa (CVPR 2024)</td><td>18.31</td><td>12.68</td><td>6.73</td><td>1176</td></tr><tr><td>MIA-Net (TMM 2023)</td><td>47.60</td><td>48.81</td><td>36.88</td><td>820</td></tr><tr><td>LGSGMM (TCSVT 2026)</td><td>54.17</td><td>56.11</td><td>39.29</td><td>80</td></tr><tr><td>CARNet(Ours)</td><td>56.67</td><td>49.22</td><td>40.81</td><td>48</td></tr></table>

Table 2: Comparison of cross-view object association methods on the MDMT-CD dataset. The best scores are marked in bold.

eficiency.

## 5.3. Comparison of Cross-View Object Association Performance

## 5.3.1. Comparison on MDMT-CD

To further evaluate the efectiveness of the proposed method in scenarios with densely distributed similar targets, we compare it with representative state-ofthe-art cross-view object association methods on the MDMT-CD dataset, including Re-ID , OSNet , ReIDMamba , GTA-Net, TTS [41], MIA-Net , SuperGlue , LoFTR , E-LoFTR , JamMa, MambaGlue [42], and RoMa . Among them, MIA-Net is based on SIFT, while SuperGlue, LoFTR, E-LoFTR, JamMa, MambaGlue, and RoMa are state-of-the-art deep learning-based image matching methods that estimate homography matrices for cross-view object mapping. The quantitative results are reported in Table 2. As shown in the table, image matching-based object association methods generally struggle with large rotation variations between views, resulting in the poorest performance on the MDMT-CD dataset. Appearance similarity-based association methods are also prone to incorrect associations due to the presence of numerous visually similar objects in UAV views. In contrast, the proposed CARNet achieves a 3.93% improvement over MIA-Net in terms of MDA (40.81% vs. 36.88%) while maintaining more than 10× higher computational eficiency. Compared with the current state-of-the-art LGS-GMM, CARNet achieves comparable association performance with significantly improved eficiency. Fig. 3 shows the qualitative association results. As can be observed, CARNet can efectively associate densely distributed and visually similar objects, even under large rotation and viewpoint variations between image pairs. Moreover, it maintains robust association performance in nighttime scenarios with sparse object appearances. This improvement benefits from the incorporation of positional information and the proposed full object-pair modeling strategy, which jointly exploit spatial and visual relationships for discriminative association.

![](images/1c2b6664b1a636abe969c0b35d12ad25d52f951a8cd3d9dcb1167c8ea6c1afc9.jpg)  
Figure 3: Visualization of cross-view target association results on the MDMT-CD dataset. Red lines indicate incorrect associations, while yellow lines denote correct associations.

## 5.3.2. Comparison on M3Track

To further evaluate the efectiveness of the proposed method in scenarios with severe non-rigid target deformation and cross-modal variations, we compare it with representative state-of-the-art cross-view object association methods on the M3Track dataset, including OSNet, CrossMOT , CT , AGW , ReIDMamba, GTA-Net, TTS, MambaGlue, and MIA-Net. For the cross-modal scenario, we further compare CARNet with representative cross-modal person re-identification methods, including DLM [43], DSAF [44], and LAReVIT [45]. The quantitative results are reported in Table 3. As shown in the table, the proposed method achieves the best overall performance among all competing methods.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Precision Recall MDA</td></tr><tr><td>(%)</td><td>(%)</td><td>(%)</td></tr><tr><td rowspan="10">Visible Light</td><td>TTS (Drones 2022)</td><td>9.89</td><td>18.21</td><td>3.48</td></tr><tr><td>GTA-Net (Drones 2025)</td><td>16.78</td><td>13.23</td><td>5.46</td></tr><tr><td>MambaGlue (CVPR 2025)</td><td>17.31</td><td>14.21</td><td>6.33</td></tr><tr><td>MIA-Net (TMM 2023)</td><td>29.77</td><td>33.21</td><td>15.67</td></tr><tr><td>ReIDMamba (TMM 2026)</td><td>39.99</td><td>43.2319.86</td><td></td></tr><tr><td>OSNet (ICCV 2019)</td><td>40.21</td><td>46.3220.76</td><td></td></tr><tr><td>CrossMOT (IJCV 2024)</td><td>39.77</td><td></td><td>49.3621.07</td></tr><tr><td>CT (ICONIP 2021)</td><td>42.48</td><td></td><td>54.5024.85</td></tr><tr><td>AGW (TPAMI 2021)</td><td>43.42</td><td></td><td>56.4027.47</td></tr><tr><td>LGSGMM (TCSVT 2026)</td><td>46.05 66.14</td><td></td><td>46.3830.05 40.2833.40</td></tr><tr><td rowspan="10">Infrared</td><td>CARNet(Ours) TTS (Drones 2022)</td><td>3.04</td><td>16.78</td><td>4.01</td></tr><tr><td>GTA-Net (Drones 2025)</td><td>7.81</td><td>12.96</td><td>4.61</td></tr><tr><td>ReIDMamba (TMM 2026)</td><td>25.47</td><td>32.11</td><td>10.24</td></tr><tr><td>OSNet (ICCV 2019)</td><td>27.87</td><td></td><td>35.2812.88</td></tr><tr><td>MambaGlue (CVPR 2025)</td><td>26.34</td><td></td><td>36.4212.97</td></tr><tr><td>CrossMOT (IJCV 2024)</td><td>24.36</td><td></td><td>42.9913.20</td></tr><tr><td>MIA-Net (TMM 2023)</td><td>28.03</td><td></td><td>39.7714.22</td></tr><tr><td>CT (ICONIP 2021)</td><td>28.82</td><td></td><td>37.2514.62</td></tr><tr><td>AGW (TPAMI 2021)</td><td>39.58</td><td></td><td>51.1622.54</td></tr><tr><td>LGSGMM (TCSVT 2026)</td><td>41.67</td><td></td><td>37.49 24.59</td></tr><tr><td rowspan="8">Cross Modal</td><td>CARNet(Ours)</td><td>61.41</td><td>36.84</td><td>29.91</td></tr><tr><td>GTA-Net (Drones 2025)</td><td>7.66</td><td>11.29</td><td>4.36</td></tr><tr><td>ReIDMamba (TMM 2026)</td><td>15.03</td><td>16.23</td><td>5.31</td></tr><tr><td>CT (ICONIP 2021)</td><td>14.04</td><td>18.13</td><td>6.19</td></tr><tr><td>AGW (TPAMI 2021)</td><td>14.47</td><td>18.68</td><td>6.41</td></tr><tr><td>MIA-Net (TMM 2023)</td><td>18.36</td><td>12.29</td><td>8.74</td></tr><tr><td>DASF (TMM 2025)</td><td>27.24</td><td></td><td>18.7611.32</td></tr><tr><td>DLM (TPAMI 2025)</td><td>26.32</td><td></td><td>20.7812.06</td></tr><tr><td>LAReVIT (CVPR 2026)</td><td></td><td>29.18</td><td>23.0714.79</td><td></td></tr><tr><td>LGSGMM (TCSVT 2026)</td><td></td><td>28.94</td><td>26.6216.10</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CARNet(Ours)</td><td>42.89</td><td></td><td>21.0316.43</td><td></td></tr></table>

Table 3: Comparison of Diferent Methods on M3Track dataset. The best scores are marked in bold.

![](images/9013148d8962e3a5419caa0edeae4d7be9f9dd29f37a5c1fe681276e71b23bfa.jpg)  
Figure 4: Visualization of cross-view target association results on the visible-light subset of the M3Track dataset.

Fig. 4, Fig. 5 and Fig. 6 shows qualitative association results. Appearancebased association methods, such as AGW and CT, mainly rely on target visual similarity and therefore struggle to distinguish densely distributed targets with similar appearances. LGSGMM implicitly introduces positional cues to guide attention aggregation. However, when targets are heavily occluded or observed across diferent modalities, inaccurate attention aggregation may introduce misleading context and reduce feature discriminability. As shown in the figure, CAR-Net can still accurately associate targets under severe non-rigid deformations, dense crowds, and heavy occlusions. This robustness benefits from explicitly incorporating positional information into pairwise relation modeling, which enables more reliable association under challenging scenarios. Moreover, CARNet maintains strong performance in both same-modal and cross-modal association scenarios.

## 5.4. Generalization to Cross-Camera Multi-Object Tracking

We further evaluate CARNet on natural cross-view tracking datasets, including Wildtrack and MvMHAT. As shown in Table 4, we compare our method with representative multi-camera tracking and association methods, including KSP-DO[46], KSP-DO-track[46], GLMB-DO[46], DMCT[47], DMCT Stack[47], ReST[48], TSMMT[49], Tracktor++[50], CenterTrack[51], TraDeS[52], TrackFormer[53], DeepCC[54], and SVT. TSMMT uses temporal information from adjacent video frames, while CARNet only uses the current cross-view image pair without relying on preceding or subsequent frames.

![](images/330e4dac9ebc7cad06cbe13e8d8fb07ff7c820f64aec8a3960271215346698b7.jpg)  
Figure 5: Visualization of cross-view target association results on the infrared modality of the M3Track dataset.

Despite this more restricted input setting, CARNet achieves the best results on both datasets. On Wildtrack, CARNet obtains 86.9% MOTA and 88.5% IDF1. On MvMHAT, it achieves 72.3% MOTA and 52.0% IDF1. These results demonstrate that the proposed position-aware all-pairs relation modeling is efective for natural cross-view object association and can provide strong association performance even without temporal video cues.

## 5.5. Ablation Study

To evaluate the efectiveness of position-aware relation modeling and the proposed intervention strategy, we conduct ablation studies on the MDMT dataset. As shown in Table 5, removing either module causes a significant decrease in MDA, indicating that both components are critical for robust cross-view correspondence. Position-aware relation modeling improves the discriminability of visually similar targets by incorporating spatial cues, while the intervention strategy enhances the learning of discriminative correspondence evidence from appearance-confusing target pairs.

![](images/1458f2321081a5e253ac5e5c10c52bdbfbfe4d251bbd509e7d8b3b035563a88e.jpg)

Figure 6: Visualization of cross-modal target association results between visible and infrared modalities on the M3Track dataset.  
![](images/b40072676d938edea3e1c0145541606743da01bacad6de82d26c5c7d82cb3ee1.jpg)  
Figure 7: Efect of diferent Drop rates on mAP50.

To investigate the efect of Visual-Evidence Perturbation Intervention, we conduct an ablation study on MDMT-CD with diferent dropout rates, as shown in Fig. 7. As the dropout rate increases, collaborative recognition performance first improves and then declines. This is expected: moderate perturbation prevents the model from over-relying on unstable visual dimensions and encourages stable relation representations under partial evidence loss, improving robustness to viewpoint changes, occlusion, and appearance variations. However, excessive dropout removes too much discriminative evidence, weakening reliable association and recognition.

<table><tr><td>Dataset</td><td>Method</td><td>MOTA (%) IDF1 (%)</td></tr><tr><td rowspan="9">Wildtrack</td><td>KSP-DO(CVPR 2018)</td><td>69.6</td><td>73.2</td></tr><tr><td>KSP-DO-track(CVPR 2018)</td><td>72.2</td><td>78.4</td></tr><tr><td>GLMB-DO (TPAMI 2018)</td><td>70.1</td><td>72.5</td></tr><tr><td>DMCT (Arxiv 2020)</td><td>72.8</td><td>77.8</td></tr><tr><td>DMCT Stack (Arxiv 2020)</td><td>74.6</td><td>81.9</td></tr><tr><td>ReST (CVPR 2023)</td><td>84.9</td><td>86.7</td></tr><tr><td>TSMMT(TCSVT 2025)</td><td>86.4</td><td>88.1</td></tr><tr><td>CARNet(Ours)</td><td>86.9</td><td>88.5</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="8">MvMHAT</td><td>Tracktor++ (ICCV 2019)</td><td>66.5</td><td>46.1</td></tr><tr><td>CenterTrack (ECCV 2020)</td><td>63.5</td><td>38.1</td></tr><tr><td>TraDeS (CVPR 2021)</td><td>69.5</td><td>44.9</td></tr><tr><td>TrackFormer(CVPR 2022)</td><td>70.4</td><td>49.6</td></tr><tr><td>DeepCC (CVPR 2018)</td><td>63.9</td><td>44.4</td></tr><tr><td>SVT (TPAMI 2022)</td><td>65.4</td><td>47.6</td></tr><tr><td>TSMMT(TCSVT 2025)</td><td>71.6</td><td>51.6</td></tr><tr><td>CARNet(Ours)</td><td>72.3</td><td>52.0</td></tr></table>

Table 4: Comparison on two natural scenario datasets: Wildtrack and MvMHAT. The best scores are marked in bold.

## 5.6. Paramater and Glops Analysis

We further evaluate the scalability of CARNet by replacing the feature extraction backbone with networks of diferent capacities. As shown in Table 6, CAR-Net can be flexibly adapted to diferent eficiency requirements. With the same ResNet-50 backbone, CARNet uses only about two-thirds of the parameters of LGSGMM and achieves better MDA performance, demonstrating the parameter eficiency of the proposed pairwise relation framework.

When lightweight deployment is required, smaller backbones can be adopted. ResNet-18 greatly reduces the model size and computation while maintaining an acceptable MDA decrease. MobileNetV3 and ShufleNetV2 further reduce the computational cost, but the performance drop becomes more evident. These results show that CARNet provides a flexible accuracy-eficiency trade-of for different deployment scenarios.

<table><tr><td colspan="4">Method Position Strategy 1 Strategy 2 MDA (%)</td></tr><tr><td></td><td></td><td>√</td><td>23.19</td></tr><tr><td></td><td>√</td><td>√</td><td>31.80</td></tr><tr><td></td><td>√</td><td>√</td><td>33.39</td></tr><tr><td>CARNet</td><td>√ √</td><td>√</td><td>40.81</td></tr></table>

Table 5: Ablation study of the proposed modules on the MDMT-CD dataset. “Position” denotes the Position information, “Strategy 1” denotes the Hard-Distinguishing Intervention, and “Strategy 2” denotes the Visual-Evidence Intervention.

We further evaluate CARNet with diferent single-drone detectors. As shown in Table 7, CARNet consistently improves collaborative recognition across all evaluated detectors. Moreover, larger gains are achieved with weaker detectors, demonstrating the robustness and detector-agnostic nature of the proposed framework.
<table><tr><td>Method</td><td>Backbone</td><td colspan="3">Parameters (M) GFLOPs (G) MDA (%)</td></tr><tr><td rowspan="6">CARNet</td><td>ResNet101</td><td>54.88</td><td>1294.33</td><td>41.04</td></tr><tr><td>ResNet50</td><td>24.56</td><td>680.18</td><td>40.81</td></tr><tr><td>ResNet18</td><td>11.83</td><td>304.55</td><td>35.93</td></tr><tr><td>MobileNetV3</td><td>1.35</td><td>28.97</td><td>29.27</td></tr><tr><td>ShuffleNetV2</td><td>0.86</td><td>18.41</td><td>28.22</td></tr><tr><td>ResNet50</td><td>35.88</td><td>680.20</td><td>39.29</td></tr></table>

Table 6: Ablation study of diferent backbones on the MDMT-CD dataset. The input image resolution is 1920 × 1080.

## 6. Conclusion

In this paper, we proposed CARNet and established MDMT-CD, a finegrained benchmark for collaborative target detection. Extensive experiments demonstrate that CARNet achieves state-of-the-art performance on MDMT-CD, M3Track, Wildtrack, and MvMHAT. These results show that the proposed method is efective for collaborative detection, cross-view target association, and cross-camera multi-object tracking, providing a strong and practical solution for multi-drone perception. However, CARNet relies on the outputs of single-drone detectors. If a target is missed by the detector, the subsequent association and fusion stages cannot recover it. Inaccurate bounding-box localization may also afect the extracted appearance features and positional representations. Future work will therefore explore the joint optimization of single-drone detection, cross-view target association, and multi-view fusion recognition.

<table><tr><td colspan="3">Single Detector  $\mathrm { m A P } _ { 5 0 } ( { \mathrm S } )$  (%)  $\mathrm { m A P } _ { 5 0 } ( \mathrm { M } )$  (%) Gain (%)</td></tr><tr><td> $\mathrm { Y O L O v 3 - T i n y }$ </td><td>18.36 24.15</td><td>+5.79</td></tr><tr><td> $\mathrm { Y O L O v } 3$ </td><td>19.73 24.22</td><td>+4.49</td></tr><tr><td> $\Upsilon \mathrm { O L O v } 5 { \mathrm { - s } }$ </td><td>22.78 25.03</td><td>+2.25</td></tr><tr><td> $\mathrm { Y O L O v 8 – s }$ </td><td>23.45 25.23</td><td>+1.78</td></tr><tr><td> $\Upsilon \mathrm { O L O v } 1 2 \mathrm { - s }$ </td><td>23.64 25.37</td><td>+1.73</td></tr><tr><td> $\Upsilon \mathrm { O L O v } 2 6 \mathrm { - s }$ </td><td>23.59 25.33</td><td>+1.74</td></tr></table>

Table 7: Robustness of CARNet to diferent single-drone detectors.

## References

[1] Y. Liu, S. Lei, H.-C. Li, J. Pan, X. Geng, Spformer: Superpixel-perceptive transformer for infrared small target detection, Pattern Recognition (2026) 114721.

[2] X. Liu, S. Li, R. An, Z. Wang, Y. Xian, X. Bai, Topological feature learning for cross-view multi-target track-to-track association in uav-based observation, Pattern Recognition (2026) 114650.

[3] J. Peng, Q. Li, Y. Tan, D. Zhao, Z. Yuan, J. Chen, H. Wang, Y. Jiang, SkyNet: Multi-drone cooperation for real-time person identification and localization, in: IEEE INFOCOM 2023 - IEEE Conference on Computer Communications, 2023, pp. 1–10. doi:10.1109/INFOCOM53939.2023.10228978.

[4] A. Lu, Y. Cheng, Y. Hu, Z. Cao, J. Liu, Z. Li, Skyseg: Collaborative onboard semantic segmentation with heterogeneous uavs in the wild, arXiv preprint arXiv:2605.24014 (2026).

[5] Z. Zhang, R. Fu, Z. Xiao, J. Xiang, Y. Song, P. Zhong, Two-stage multidrone multi-target association: Laplacian-guided feature enhancement and sgmm-based error filtering approach, IEEE Transactions on Circuits and Systems for Video Technology (2026).

[6] I. Guyon, D. Janzing, B. Schölkopf, Causality: Objectives and assessment, in: Causality: Objectives and Assessment, PMLR, 2010, pp. 1–42.

[7] P. Zhu, J. Zheng, D. Du, L. Wen, Y. Sun, Q. Hu, Multi-drone-based single object tracking with agent sharing network, IEEE Transactions on Circuits and Systems for Video Technology 31 (2021) 4058–4070. doi:10.1109/ TCSVT.2020.3045747.

[8] Z. Liu, Y. Shang, T. Li, G. Chen, Y. Wang, Q. Hu, P. Zhu, Robust multidrone multi-target tracking to resolve target occlusion: A benchmark, IEEE Transactions on Multimedia 25 (2023) 1462–1476.

[9] H. Wu, H. Sun, K. Ji, G. Kuang, Temporal-spatial feature interaction network for multi-drone multi-object tracking, IEEE Transactions on Circuits and Systems for Video Technology 35 (2025) 1165–1179. doi:10.1109/ TCSVT.2024.3478758.

[10] P. Nguyen, K. G. Quach, C. N. Duong, S. L. Phung, N. Le, K. Luu, Multicamera multi-object tracking on the move via single-stage global association approach, Pattern Recognition 152 (2024) 110457.

[11] Y. Zheng, H. Qi, L. Li, S. Li, Y. Huang, C. He, D. Wang, Motion-guided and occlusion-aware multi-object tracking with hierarchical matching, Pattern Recognition 151 (2024) 110369.

[12] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, W. Zhang, Learning distilled collaboration graph for multi-agent perception, in: Advances in Neural Information Processing Systems, volume 34, 2021, pp. 29541–29552.

[13] Y. Hu, S. Fang, Z. Lei, Y. Zhong, S. Chen, Where2comm: Communicationeficient collaborative perception via spatial confidence maps, in: Advances in Neural Information Processing Systems, volume 35, 2022, pp. 4874– 4886.

[14] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, J. Ma, V2x-vit: Vehicle-toeverything cooperative perception with vision transformer, in: European Conference on Computer Vision, 2022, pp. 107–124.

[15] R. Xu, Z. Tu, H. Xiang, W. Shao, B. Zhou, J. Ma, Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers, in: Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, 2023, pp. 989–1000.

[16] S. Wei, Y. Wei, Y. Hu, Y. Lu, Y. Zhong, S. Chen, Y. Zhang, Asynchronyrobust collaborative perception via bird’s eye view flow, in: Advances in Neural Information Processing Systems, volume 36, 2023.

[17] H. Ye, R. Sunderraman, S. Ji, Uav3d: A large-scale 3d perception benchmark for unmanned aerial vehicles, in: Advances in Neural Information Processing Systems, volume 37, 2024, pp. 55425–55442. doi:10.52202/ 079017-1760.

[18] P. Tian, Z. Wang, P. Cheng, Y. Wang, Z. Wang, L. Zhao, M. Yan, X. Yang, X. Sun, Ucdnet: Multi-uav collaborative 3-d object detection network by reliable feature mapping, IEEE Transactions on Geoscience and Remote Sensing 63 (2025).

[19] Z. Lin, W. Chen, X. Jin, Y. Yang, L. Fan, Y. Zhang, Y. Zhang, Z. Zhang, Mcop: Multi-uav collaborative occupancy prediction, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 27242– 27251.

[20] W. Li, H. Xiang, T. Wang, S. Wu, Q. Xia, C. Wang, C. Wen, V2u4real: A real-world large-scale dataset for vehicle-to-uav cooperative perception, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 4728–4737.

[21] J. Zha, Y. Fan, T. Zhang, G. Chen, Y. Chen, C. Gao, X. Chen, Aircopbench: A benchmark for multi-drone collaborative embodied perception and reasoning, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2026, pp. 1507–1515.

[22] X. Li, L. Wu, Y. Niu, A. Ma, Multi-target association for uavs based on triangular topological sequence, Drones 6 (2022) 119. doi:10.3390/ drones6050119.

[23] Z. Liu, Y. Shang, T. Li, G. Chen, Y. Wang, Q. Hu, P. Zhu, Robust multi-drone multi-target tracking to resolve target occlusion: A benchmark, IEEE Transactions on Multimedia 25 (2023) 1462–1476. doi:10.1109/TMM.2021. 3124578.

[24] Y. Qiao, H. Fan, Q. Wang, T. Zhao, Y. Tang, Stca: High-altitude tracking via single-drone tracking and cross-drone association, Remote Sensing 16 (2024) 3861. doi:10.3390/rs16203861.

[25] Q. Tan, X. Yang, C. Qiu, W. Liu, Y. Li, Z. Zou, J. Huang, Graph-based target association for multi-drone collaborative perception under imperfect detection conditions, Drones 9 (2025) 300. doi:10.3390/drones9040300.

[26] L. He, X. Liao, W. Liu, X. Liu, P. Cheng, T. Mei, Fastreid: A pytorch toolbox for general instance re-identification, in: Proceedings of the 31st ACM international conference on multimedia, 2023, pp. 9664–9667.

[27] K. Zhou, Y. Yang, A. Cavallaro, T. Xiang, Omni-scale feature learning for person re-identification, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 3702–3712.

[28] H. Gu, Q. Yang, L. Pu, S. Han, Y. Ding, Reidmamba: Learning discriminative features with visual state space model for person re-identification, IEEE Transactions on Multimedia (2026).

[29] S. Hao, P. Liu, Y. Zhan, K. Jin, Z. Liu, M. Song, J.-N. Hwang, G. Wang, Divotrack: A novel dataset and baseline method for cross-view multi-object tracking in diverse open scenes, International Journal of Computer Vision 132 (2024) 1075–1090.

[30] M. Wieczorek, B. Rychalska, J. D˛abrowski, On the unreasonable efectiveness of centroids in image retrieval, in: International Conference on Neural Information Processing, Springer, 2021, pp. 212–223.

[31] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S. C. Hoi, Deep learning for person re-identification: A survey and outlook, IEEE transactions on pattern analysis and machine intelligence 44 (2021) 2872–2893.

[32] P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich, SuperGlue: Learning feature matching with graph neural networks, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 4938–4947.

[33] J. Sun, Z. Shen, Y. Wang, H. Bao, X. Zhou, LoFTR: Detector-free local feature matching with transformers, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 8922–8931.

[34] J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, M. Felsberg, Roma: Robust dense feature matching, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19790–19800.

[35] G. Dong, G. Kuang, Target recognition via information aggregation through dempster–shafer’s evidence theory, IEEE Geoscience and Remote Sensing Letters 12 (2015) 1247–1251.

[36] H. Su, S. Maji, E. Kalogerakis, E. Learned-Miller, Multi-view convolutional neural networks for 3d shape recognition, in: Proceedings of the IEEE international conference on computer vision, 2015, pp. 945–953.

[37] Z. Han, C. Zhang, H. Fu, J. T. Zhou, Trusted multi-view classification with dynamic evidential fusion, IEEE transactions on pattern analysis and machine intelligence 45 (2022) 2551–2566.

[38] W. Liu, Y. Chen, X. Yue, C. Zhang, S. Xie, Safe multi-view deep classification, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 2023, pp. 8870–8878.

[39] W. Liu, Y. Chen, X. Yue, Enhancing multi-view classification reliability with adaptive rejection, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 2025, pp. 18969–18977.

[40] L. Lv, Q. Guo, L. Zhang, L. Du, B. Jiang, L. Chen, X. Liang, Uncertaintyguided view-strength-aware feature utilization for multi-view classification,

in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2026, pp. 24198–24206.

[41] X. Li, L. Wu, Y. Niu, A. Ma, Multi-target association for uavs based on triangular topological sequence, Drones 6 (2022) 119.

[42] K. Ryoo, H. Lim, H. Myung, Mambaglue: Fast and robust local feature matching with mamba, in: 2025 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2025, pp. 5758–5765.

[43] M. Ye, Z. Wu, B. Du, Dual-level matching with outlier filtering for unsupervised visible-infrared person re-identification, IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[44] Y. Jiang, X. Cheng, H. Yu, X. Liu, H. Chen, G. Zhao, Dsaf: Dual space alignment framework for visible-infrared person re-identification, IEEE Transactions on Multimedia (2025).

[45] H. Xu, G. Niu, Bit: Matching-based bi-directional interaction transformation network for visible-infrared person re-identification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 40386–40396.

[46] T. Chavdarova, P. Baqué, S. Bouquet, A. Maksai, C. Jose, T. Bagautdinov, L. Lettry, P. Fua, L. Van Gool, F. Fleuret, Wildtrack: A multi-camera hd dataset for dense unscripted pedestrian detection, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 5030–5039.

[47] Q. You, H. Jiang, Real-time 3d deep multi-camera tracking, arXiv preprint arXiv:2003.11753 (2020).

[48] C.-C. Cheng, M.-X. Qiu, C.-K. Chiang, S.-H. Lai, Rest: A reconfigurable spatial-temporal graph model for multi-camera multi-object tracking, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 10051–10060.

[49] H. Wu, H. Sun, K. Ji, G. Kuang, Temporal-spatial feature interaction network for multi-drone multi-object tracking, IEEE Transactions on Circuits and Systems for Video Technology 35 (2024) 1165–1179.

[50] P. Bergmann, T. Meinhardt, L. Leal-Taixe, Tracking without bells and whistles, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 941–951.

[51] X. Zhou, V. Koltun, P. Krähenbühl, Tracking objects as points, in: European conference on computer vision, Springer, 2020, pp. 474–490.

[52] J. Wu, J. Cao, L. Song, Y. Wang, M. Yang, J. Yuan, Track to detect and segment: An online multi-object tracker, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12352– 12361.

[53] T. Meinhardt, A. Kirillov, L. Leal-Taixe, C. Feichtenhofer, Trackformer: Multi-object tracking with transformers, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 8844– 8854.

[54] E. Ristani, C. Tomasi, Features for multi-target multi-camera tracking and re-identification, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6036–6046.