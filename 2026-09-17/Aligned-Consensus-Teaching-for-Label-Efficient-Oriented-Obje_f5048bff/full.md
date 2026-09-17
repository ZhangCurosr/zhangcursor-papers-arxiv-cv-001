# Aligned Consensus Teaching for Label-Efficient Oriented Object Detection in Weakly-Aligned Visible-Infrared Imagery

Qi Ming<sup>1</sup> Xiaxin Yuan<sup>2</sup> Jiahuan Zhou<sup>3</sup> Jiangmeng Li<sup>4</sup> Xudong Zhao<sup>5</sup>

Zhanchao Huang<sup>6</sup> Juan Fang<sup>1</sup> Shaoguang Huang<sup>7</sup> Aleksandra Pizurica<sup>8</sup>

<sup>1</sup>Beijing University of Technology <sup>2</sup>University of Science and Technology of China <sup>3</sup>Peking University

<sup>4</sup>Institute of Software, Chinese Academy of Sciences <sup>5</sup>Beijing Institute of Technology

<sup>6</sup>Fuzhou University <sup>7</sup>China University of Geosciences Wuhan

<sup>8</sup>Telecommunications and Information Processing, Ghent University

chaser.ming@gmail.com xiaxinyuan@mail.ustc.edu.cn

Aleksandra.Pizurica@UGent.be

## Abstract

Visible-infrared object detection (VIOD) detects objects with oriented bounding boxes from paired visible and infrared images. Existing methods depend on costly dualmodality annotations. Semi-supervised learning can reduce this burden, but extending itfrom single-modal detection to VIOD is challenging. In the practical image-pair-level setting considered here, only a few pairs are labeled in both modalities, while the rest are completely unlabeled. This limited supervision creates three challenges: (i) too few labeled boxes for robust cross-modal alignment; (ii) pseudolabel errors caused by branch-wise misses accumulate during self-training; and (iii) tail-class annotations become critically scarce as the labeling budget decreases. We propose Aligned Consensus Teacher (ACT) for label-efficient VIOD in this setting. Its Cycle-Consistent Region Alignment (CRA) combines cycle consistency and sparse anchors with reliability-weighted regional matching. Cross-Modal Consensus Mean-Teacher (CMC-MT) forms consensus pseudo labels under pair-preserving views to recover branch-wise misses and supervise unlabeled pairs. Text-Guided Cross-Modal Instance Augmentation (TG-CMIA) uses a visionlanguage scene prior to compose tail-class instance pairs while preserving RGB–IR offsets. To the best of our knowledge, ACT is the first framework to study semi-supervised VIOD under this image-pair-level setting. Experiments on DroneVehicle and VEDAI show consistent gains across annotation ratios. With 10% labeled pairs on DroneVehicle, ACT reaches 94.3% ofthe mAP obtained by the same detector under full supervision. Code and models will be available on GitHub tofacilitatefuture work.

## 1. Introduction

Object detection in aerial images supports a wide range of applications, including traffic surveillance, urban management, and disaster rescue [28, 34]. Aerial objects appear at arbitrary orientations, so detectors predict rotated bounding boxes [3, 7, 35]. However, detectors that use only visible (RGB) images often degrade at night or under adverse weather [40]. Infrared (IR) images capture the radiated heat of objects and reveal clear silhouettes even in low light, so the two modalities are complementary [12, 28, 41]. By jointly exploiting them, visible-infrared object detection (VIOD) works under both daytime and low-light conditions.

Existing VIOD detectors are mostly fully supervised and thus rely on abundant annotations for both modalities [28, 40, 41]. Such annotations are costly to obtain, because the visible and infrared images must be annotated separately, which roughly doubles the labeling effort. Learning VIOD detectors from only a few labeled pairs is therefore of great practical value. Semi-supervised learning offers a natural way to reduce this cost, because it trains the detector with a few image pairs labeled in both modalities and abundant completely unlabeled pairs [11, 36].

However, extending semi-supervised learning from single-modal object detection to VIOD remains underexplored. Under this limited paired supervision, three challenges arise. (i) Insufficient Cross-Modal Alignment Supervision: The two sensors differ in field of view and imaging time, so the same object shows intrinsic position and size deviations [40, 41]. Only labeled pairs provide matched boxes for regional correspondence. On DroneVehicle, reducing the labeling budget to 1% decreases their number from 286.8k to 3.2k, leaving insufficient supervision for robust cross-modal alignment (Figure 1(a)). (ii) Accumulated

(c) Tail-class recovery at the 1% budget  
![](images/207c83b66141258dfaa5591a6bbf0396fb5637a17ab0ae3b08637227a1e32449.jpg)

![](images/e160e67ef0fc15a3c71111270a6983126bf48a5938893d195bad0ea3781bed98.jpg)  
Figure 1. Motivation on DroneVehicle. (a) A 1% labeling budget provides about 90× fewer matched boxes for RGB–IR alignment. (b) A branch-wise miss yields incomplete pseudo labels whose errors propagate through EMA self-training. (c) ACT recovers classwise AP toward the full-supervision reference at the 1% budget.

Pseudo-Label Noise: Visibility differences between RGB and IR cause branch-wise misses and unequal recall across the two modalities. In Figure 1(b), RGB recall is 73.2%, whereas IR recall reaches 91.1% at the 10% budget. Such misses produce incomplete pseudo labels whose errors can accumulate through EMA self-training. (iii) Tail-Class Annotation Scarcity: Aerial vehicles follow a long-tailed distribution. As the labeled subset shrinks, the absolute number of annotated tail instances drops sharply even when the class proportions remain similar. Figure 1(c) shows the resulting performance gap. When the same supervised detector is trained with 1% rather than all labels, AP drops substantially more on the minority categories than on car. Tail classes consequently receive too little supervision to learn their diverse appearances and orientations [17, 29, 39, 42].

To address these challenges, we propose Aligned Consensus Teacher (ACT), a label-efficient framework for image-pair-level semi-supervised VIOD. First, a Cycle-Consistent Region Alignment module combines target-free cycle consistency with sparse cross-modal anchors and reliability-weighted matching to regularize regional correspondence under limited paired supervision. Next, a Cross-Modal Consensus Mean-Teacher forms consensus pseudo labels from the CRA-regularized fused branch under pairpreserving views to recover branch-wise misses and supervise unlabeled pairs. Finally, a Text-Guided Cross-Modal Instance Augmentation composes minority-class instance pairs in both modalities to enrich tail-class supervision. Extensive experiments on DroneVehicle and VEDAI show that ACT consistently outperforms state-of-the-art methods under all annotation budgets. With only 10% labeled pairs on DroneVehicle, it reaches 94.3% of the mAP obtained by the same detector under full supervision. The main contributions are summarized as follows:

• We study image-pair-level semi-supervised VIOD and propose ACT to learn from a few jointly labeled RGB– IR pairs and many completely unlabeled pairs. To the best of our knowledge, ACT is the first framework for this setting.

• To regularize cross-modal correspondence under limited paired supervision, we design a Cycle-Consistent Region Alignment (CRA) module that combines target-free cycle consistency, sparse cross-modal anchors, and reliability-weighted regional matching.

• To reduce pseudo-label errors from branch-wise misses, we develop a Cross-Modal Consensus Mean-Teacher (CMC-MT) that derives reliable pseudo labels from cross-modal consensus under pair-preserving views.

• To address tail-class annotation scarcity, we introduce a Text-Guided Cross-Modal Instance Augmentation (TG-CMIA) strategy that composes minority-class instance pairs coherently across both modalities under a vision-language scene prior.

## 2. Related Work

Visible-Infrared Object Detection. Oriented object detection is essential in aerial scenes, where objects appear at arbitrary angles [28]. Existing methods adapt two-stage proposal pipelines to rotated regions [3, 26, 35] or use one-stage dense detectors [7, 20], with later advances in rotation-equivariant representation and geometry-aware localization [8, 23, 37, 38]. These RGB-only detectors can degrade under poor illumination. Visible-infrared detectors instead exploit complementary sensing through crossmodal alignment [40, 44], feature calibration [41], or statespace fusion [4, 47]. Because the paired views are weakly aligned, they learn spatial correspondence from densely labeled boxes. Most therefore remain fully supervised and cannot obtain such correspondence supervision when only a few image pairs are labeled.

Semi-Supervised Object Detection. Semi-supervised object detection trains on a few labeled images and many unlabeled images [36]. Most methods follow mean-teacher self-training, where an EMA teacher generates pseudo labels for a student [21, 36, 46]; this paradigm has also been extended to oriented objects [11, 32]. Subsequent methods improve pseudo-label quality through detection consistency, dense supervision, localization-aware scoring, adaptive assignment, or label polishing [15, 16, 30, 33, 43, 46]. However, they assume a single modality and do not model cross-modal displacement or complementary branch-wise misses. Sparsely annotated multimodal detection instead studies missing instances within an image [14] or annotations available in only one modality [13]. These settings differ from image-pair-level semi-supervised VIOD, where a few pairs are labeled in both modalities, the remaining pairs are completely unlabeled, and augmentation must preserve RGB–IR pairing.

## 3. Methodology

## 3.1. Overview

Figure 2 illustrates ACT for image-pair-level semisupervised VIOD. CRA regularizes regional correspondence on labeled RGB–IR pairs, CMC-MT generates consensus pseudo labels for unlabeled pairs under pairpreserving views, and TG-CMIA composes tail-class pairs on the labeled branch. The student learns from these supervised and unsupervised objectives. At inference, only the fused EMA teacher and oriented detection head are retained.

## 3.2. Cycle-Consistent Region Alignment

RGB and infrared images exhibit object-level displacement, while limited labeled boxes are insufficient to learn robust cross-modal alignment. CRA addresses this problem by coupling cycle consistency with sparse correspondence anchors and reliability-weighted regional matching.

Cycle-Consistent Alignment with Sparse Anchors. Let $f _ { R }$ and $f _ { T }$ denote the RGB and infrared feature maps, let $\textit { b } = \left( c _ { x } , c _ { y } , w , h , \theta \right)$ be an RGB region, and define $r _ { m } ( b ) = \operatorname { R o I } ( f _ { m } , b )$ using RoI-aligned regional features [9]. A lightweight forward head $T _ { \psi }$ predicts its center and scale offset from the regional features of both modalities. A reverse head $T _ { \psi ^ { \prime } }$ then maps the transformed region back to the RGB frame:

$$
\begin{array} { r l } & { \Delta _ { b } ^ { R  T } = T _ { \psi } ( [ r _ { R } ( b ) , r _ { T } ( b ) ] ) , } \\ & { \quad b _ { R  T } = T ( b , \Delta _ { b } ^ { R  T } ) , } \\ & { \quad b _ { R  T  R } = T ( b _ { R  T } , T _ { \psi ^ { \prime } } ( [ r _ { T } ( b _ { R  T } ) , r _ { R } ( b _ { R  T } ) ] ) ) . } \end{array}\tag{1}
$$

Here, T adjusts the center and scale while retaining the orientation. The round trip imposes cycle consistency without a matched infrared box for every RGB source region. Class-consistent RGB–IR matches define A.

$$
\mathcal { L } _ { \mathrm { c y c } } = \frac { 1 } { | \mathcal { R } | } \sum _ { b \in \mathcal { R } } \| ( b _ { R  T  R } - b ) _ { 1 : 4 } \| _ { 1 } ,\tag{2a}
$$

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { \operatorname* { m a x } ( | \boldsymbol { A } | , 1 ) } \sum _ { b \in \boldsymbol { A } } \ell _ { \mathrm { S L 1 } } \big ( \Delta _ { b } ^ { R  T } - \Delta _ { b } ^ { * } \big ) .\tag{2b}
$$

Here, R contains RGB ground-truth regions from the labeled pairs, and $\Delta _ { b } ^ { * }$ is encoded from a matched infrared box. Equation (2a) is target-free because it needs no matched infrared offset for every source region; this does not imply optimization on the unlabeled branch. Since a cycle alone admits mutually consistent yet incorrect mappings, the sparseanchor loss in Eq. (2b) resolves this ambiguity and prevents transformation drift.

Reliability-Weighted Regional Matching. Even with geometric constraints, appearance agreement varies across regions because illumination and sensing conditions affect the two modalities differently. We denote the regional features before and after mapping by $z _ { R } ^ { b } = \mathrm { R o I } ( f _ { R } , b )$ and $z _ { T } ^ { b _ { R }  T } = \mathrm { R o I } ( f _ { T } , b _ { R  T } )$ . A coarse occupancy score $G _ { b } \in$ [0, 1] from the offline scene prior refines their visual agreement and yields the regional confidence

$$
\begin{array} { r l } { \omega _ { b } ^ { \mathrm { v i s } } } & { = \sigma \Big ( \cos ( z _ { R } ^ { b } , z _ { T } ^ { b _ { R }  T } ) \Big ) , } \\ { \omega _ { b } } & { = \mathrm { c l i p } \big ( \omega _ { b } ^ { \mathrm { v i s } } [ 1 + \lambda _ { G } ( G _ { b } - 0 . 5 ) ] , 0 , 1 \big ) , } \\ { \mathcal { L } _ { \mathrm { c m a } } } & { = \frac { \sum _ { b \in \mathcal { R } } \omega _ { b } [ 1 - \cos ( z _ { R } ^ { b } , z _ { T } ^ { b _ { R }  T } ) ] } { \epsilon + \sum _ { b \in \mathcal { R } } \omega _ { b } } . } \end{array}\tag{3}
$$

Here, σ is the sigmoid function, clip bounds the weight to [0, 1], and ϵ ensures numerical stability. The weight suppresses uncertain correspondence signals in the alignment objective rather than directly rescaling the fused features. For neighboring source regions, let $\mathcal { N } = \{ ( i , j ) \ \mid$ $b _ { i } , b _ { j } \in \mathcal { R } , 0 \mathrm { ~ < ~ } \| p _ { i } - p _ { j } \| _ { 2 } < r _ { s } \big \}$ , where p<sub>i</sub> is the center of $b _ { i } .$ . We define $\begin{array} { r } { \mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { ( i , j ) \in \mathcal { N } } \Vert \bar { \Delta } _ { i } ^ { R  T } - } \end{array}$ $\Delta _ { j } ^ { R  T } \Vert _ { 2 } ^ { 2 } / \operatorname* { m a x } ( \vert \mathcal { N } \vert , 1 )$ to discourage abrupt offset changes within radius $r _ { s }$ . The complete CRA objective is

$$
{ \mathcal { L } } _ { \mathrm { C R A } } = \lambda _ { a } { \mathcal { L } } _ { \mathrm { a l i g n } } + \lambda _ { c } { \mathcal { L } } _ { \mathrm { c y c } } + \lambda _ { m } { \mathcal { L } } _ { \mathrm { c m a } } + \lambda _ { s } { \mathcal { L } } _ { \mathrm { s m o o t h } } .\tag{4}
$$

CRA consequently combines limited box-level anchors with cycle consistency and reliability-aware matching. The regularized cross-modal representation provides the basis for the fused consensus in CMC-MT without adding an inference-time alignment branch.

## 3.3. Cross-Modal Consensus Mean-Teacher

CRA regularizes regional correspondence, but the few labeled pairs remain insufficient for learning an accurate detector. Moreover, a missed target in either modality can produce incomplete supervision, and this error may circulate through the student–teacher update. CMC-MT addresses this problem by generating pseudo labels from fused crossmodal evidence under pair-preserving views.

Pair-Preserving Teacher–Student Views. For each unlabeled pair $( I _ { R } , I _ { T } )$ , the EMA teacher model receives a weak view and the trainable student model receives a strong view [21, 22, 27]. Both modalities share the same geometric transformation H, while modality-specific photometric operators produce the weak and strong observations:

$$
I _ { m } ^ { v } = \mathcal { P } _ { m } ^ { v } \big ( \mathcal { H } ( I _ { m } ) \big ) , \qquad m \in \{ R , T \} , \quad v \in \{ w , s \} .\tag{5}
$$

![](images/455667078d7f085412d153d254393d3635ac07a991d829f066ee170d4658036e.jpg)  
Figure 2. Training and inference overview of ACT. CRA learns regional correspondence from labeled pairs, CMC-MT converts unlabeled pairs into consensus supervision, and TG-CMIA composes tail-class RGB–IR instance pairs on the labeled branch. Only the fused EMA teacher is retained for inference.

Thus, the RGB–IR geometry is preserved within each view, and teacher boxes remain valid for the student view because the weak and strong observations use the same H. This paired construction avoids the coordinate inconsistency that would arise if either modality were transformed independently.

Cross-Modal Consensus Supervision. The teacher processes the weak pair and aggregates the CRA-regularized RGB and infrared evidence through its fused branch. Let $\Phi _ { \bar { \theta } }$ and $D _ { \bar { \theta } }$ denote the teacher fusion and oriented detection functions. The consensus pseudo-label set is

$$
\begin{array} { r l } { \mathcal { P } _ { \mathrm { c o n } } ^ { w } } & { = D _ { \bar { \theta } } ( \Phi _ { \bar { \theta } } ( I _ { R } ^ { w } , I _ { T } ^ { w } ) ) , } \\ { \widetilde { B } _ { \mathrm { c o n } } } & { = \{ ( b , c ) \in \mathcal { P } _ { \mathrm { c o n } } ^ { w } | s _ { b } \geq \tau \} . } \end{array}\tag{6}
$$

RGB and infrared responses are complementary, so the fused prediction can retain target evidence when an individual branch misses or weakly observes an object. The resulting consensus labels can consequently be more complete than supervision drawn from either displaced branch alone. They supervise the student detector on the strong paired view after a short burn-in stage. The teacher is updated as the exponential moving average of the student after each optimization step [31]. This feedback loop turns completely unlabeled pairs into training targets, while fused consensus prevents branch-wise misses from recurring through selftraining.

## 3.4. Text-Guided Cross-Modal Instance Augmentation

CMC-MT exploits unlabeled pairs, but it cannot create additional ground-truth instances for tail classes. As the labeling budget decreases, tail-class annotations become critically scarce. Conventional single-modal instance composition [6] can increase instance counts, but it does not preserve the correspondence of an RGB–IR pair. TG-CMIA couples reliable tail-instance selection with scene-aware, offset-preserving pair composition.

Consensus-Guided Tail-Instance Sampling. TG-CMIA first constructs an instance bank from the labeled pairs. An RGB–IR crop pair $\iota _ { i }$ is retained only when the two annotations agree in class and their centers satisfy the expected cross-modal displacement:

$$
\begin{array} { r } { \mathcal { B } = \left\{ \iota _ { i } \left| \begin{array} { c } { c _ { R } ^ { i } = c _ { T } ^ { i } \in \mathcal { C } _ { \mathrm { t a i l } } , } \\ { \| p _ { R } ^ { i } - p _ { T } ^ { i } \| _ { 2 } < \tau _ { d } \operatorname* { m a x } ( w _ { i } , h _ { i } ) } \end{array} \right. \right\} . } \end{array}\tag{7}
$$

Here, $\mathcal { C } _ { \mathrm { t a i l } } ~ = ~ \{ \mathrm { t r u c k , f r e i g h t ~ c a r , v a n } \}$ denotes the three target categories used by TG-CMIA. The variables $p _ { R } ^ { i }$ and $p _ { T } ^ { i }$ are the paired box centers, $( w _ { i } , h _ { i } )$ is the size of the RGB source box, and $\tau _ { d } = 0 . 6$ . Each bank entry stores the paired crops, their oriented boxes, and their natural relative offset. Thus, consensus denotes agreement between the paired annotations rather than a teacher prediction, and it filters inconsistent instance pairs before augmentation.

To direct the limited composition budget toward scarce categories, an instance $\iota _ { i }$ is sampled according to

$$
P ( \iota _ { i } ) = \frac { n _ { c _ { i } } ^ { - 1 } } { \sum _ { \iota _ { j } \in B } n _ { c _ { j } } ^ { - 1 } } ,\tag{8}
$$

where $n _ { c _ { i } }$ is the number of bank instances from class $c _ { i } .$ The inverse-frequency distribution increases the sampling probability of the rarest available classes without altering the original labels. It follows the general principle of frequencyaware rebalancing [2].

Scene-Aware Offset-Preserving Composition. Visionlanguage pretraining provides semantic scene representations without task-specific box labels [24]. For each target pair, a frozen Qwen2.5-VL-32B model [1] generates the scene prior $ { \boldsymbol { S } } = ( G , d , \rho )$ once without annotation input. Here, G is ${ \bf { a } } \ 2 \times 2$ occupancy grid, d is a day–night descriptor, and $\rho$ contains the two modality reliabilities. During detector training, G selects the composition cell, d permits one additional inserted pair at night, and $\rho$ scales the modalityspecific feathering masks. Once an instance pair is selected, TG-CMIA applies the same translation t to its RGB and infrared box centers:

$$
\begin{array} { c } { { p _ { R } ^ { \prime } = p _ { R } + t , \qquad p _ { T } ^ { \prime } = p _ { T } + t , } } \\ { { p _ { T } ^ { \prime } - p _ { R } ^ { \prime } = p _ { T } - p _ { R } . } } \end{array}\tag{9}
$$

The second relation shows that the original cross-modal displacement is preserved after composition. The paired crops are blended into their corresponding regions without changing their relative scale or orientation, and the transformed oriented boxes are appended to both annotation sets. A composed instance may remain faint in nighttime RGB imagery while appearing clear in infrared, yet the two annotations stay geometrically paired. TG-CMIA thus adds targeted ground-truth supervision for tail classes without introducing cross-modal inconsistency.

Overall Training Objective. Let D denote the labeled pair set and let $\mathcal { D } _ { l } ^ { + } = \mathcal { A } _ { \mathrm { T G } } ( \mathcal { D } _ { l } ; B , S )$ denote its TG-CMIAaugmented counterpart. The complete student objective is

$$
\mathcal { L } _ { \mathrm { A C T } } = \mathcal { L } _ { \mathrm { d e t } } ^ { l } ( \mathcal { D } _ { l } ^ { + } ) + \lambda _ { u } \mathcal { L } _ { \mathrm { d e t } } ^ { u } ( \widetilde { \mathcal { B } } _ { \mathrm { c o n } } ) + \mathcal { L } _ { \mathrm { C R A } } ,\tag{10}
$$

where the three terms correspond to supervised detection on the augmented labeled pairs, consensus supervision on the unlabeled pairs, and cross-modal alignment, respectively. The student is optimized by Eq. (10), while the teacher is updated as the exponential moving average of the student.

## 4. Experiments

## 4.1. Datasets and Evaluation Protocol

DroneVehicle. DroneVehicle [28] contains 28,439 RGB–IR image pairs collected from daytime and nighttime scenes. It covers five vehicle categories with oriented annotations in both modalities. The official split contains 17,990 training, 1,469 validation, and 8,980 test pairs.

VEDAI. VEDAI [25] covers four vehicle categories. Our split contains 772 training pairs and 192 validation pairs. Its average object width is approximately 14 pixels, compared with 50 pixels on DroneVehicle. We use it to evaluate ACT under limited data and substantially smaller objects.

Image-pair-level protocol. For each budget $p ,$ we select a fixed subset containing p% of the training image pairs. Both modalities are labeled in the selected pairs, while all annotations are removed from the remaining pairs. We use $p ~ \in ~ \{ 1 , 5 , 1 0 \}$ on DroneVehicle and $p \in \{ 5 , 1 0 , 1 5 \}$ on VEDAI. These ratios correspond to 180, 900, and 1,799 labeled pairs on DroneVehicle, and 38, 77, and 115 labeled pairs on VEDAI.

Evaluation metric. Following semi-supervised detection practice [11, 36], we evaluate against infrared validation annotations and report rotated mAP at 0.5 IoU with 11-point AP interpolation [5]. All methods use identical splits and evaluation code.

## 4.2. Implementation Details

Detector architecture. The detector uses two ResNet-50 streams [10]. At each backbone stage, adaptive feature sampling and inter-modality cross-attention calibrate and exchange complementary features [41]. The resulting multiscale representations are fused by an FPN [19] and passed to an $\mathsf { S } ^ { 2 }$ A-Net oriented detection head [7].

Training settings. We implement ACT with MMRotate 0.3.4 [48] on one NVIDIA A800 GPU. SGD uses a 0.001 initial learning rate, 0.9 momentum, 0.0001 weight decay, and 500-iteration linear warm-up. Each iteration samples one labeled and one unlabeled pair at $5 1 2 \times 6 4 0$ on DroneVehicle or 1024×1024 on VEDAI. DroneVehicle is trained for six epochs with decays at epochs 4 and 5. VEDAI uses twelve epochs with decays at epochs 8 and 11. Both add two refinement epochs at learning rate 0.0005.

Semi-supervised configuration. Labeled and unlabeled pairs share resizing and geometric flips. Pair-preserving weak and strong views differ only in photometric augmentation. The teacher follows the student via EMA with 0.999 momentum. After a 2,000-iteration burn-in, fused predictions above 0.4 confidence supervise strong views with weight $\lambda _ { u } ~ = ~ 1 . 0$ . We evaluate thresholds in {0.3, 0.4, 0.5, 0.7} and select 0.4; complete run details are supplementary. The CRA weights $\left( \lambda _ { a } , \lambda _ { c } , \lambda _ { m } , \lambda _ { s } \right)$ are $( 1 . 0 , 0 . 5 , 0 . 5 , 0 . 3 )$ , and $\lambda _ { G } = 0 . 3$ . Sparse anchors require at least 0.5 cross-modal IoU, and the neighborhood radius $r _ { s }$ is 64 pixels. With probability 0.8, TG-CMIA adds one to four target-class pairs and at most one extra pair at night.

Table 1. DroneVehicle results across labeling budgets.
<table><tr><td>Method</td><td>Input</td><td>1%</td><td>5%</td><td>10%</td></tr><tr><td>Supervised</td><td></td><td></td><td></td><td></td></tr><tr><td>C{Former [41]</td><td>V+I V+I</td><td>0.299 0.366</td><td>0.485 0.599</td><td>0.575 0.663</td></tr><tr><td>DMM [47] SM3Det [18]</td><td>I</td><td>0.389</td><td>0.548</td><td>0.609</td></tr><tr><td>M2D-LIF [45]</td><td>V+I</td><td>0.271</td><td>0.491</td><td>0.602</td></tr><tr><td>Semi-supervised</td><td></td><td></td><td></td><td></td></tr><tr><td>MT [31]</td><td>V+I</td><td>0.452</td><td>0.578</td><td>0.612</td></tr><tr><td>SOOD [11]</td><td>I</td><td>0.476</td><td>0.581</td><td>0.594</td></tr><tr><td></td><td>I</td><td>0.477</td><td></td><td></td></tr><tr><td>DT [46]</td><td></td><td></td><td>0.594</td><td>0.621</td></tr><tr><td>MCL [32] ACT (ours)</td><td>I V+I</td><td>0.492 0.511</td><td>0.592 0.644</td><td>0.614 0.683</td></tr></table>

## 4.3. Comparison with State-of-the-Art

Comparison protocol. Tables 1 and 2 compare supervised detectors trained only on labeled subsets with semisupervised detectors that also use unlabeled pairs. All methods share the same subsets and evaluation code. MT and DT denote Mean-Teacher and Dense Teacher; I and V+I denote infrared-only and paired RGB–IR input, respectively. Bold and underlined values mark the best and second-best results. For resolution-sensitive DMM, we report native-resolution VEDAI scores.

Results on DroneVehicle. ACT obtains 0.511, 0.644, and 0.683 mAP at the 1%, 5%, and 10% budgets. It outperforms the strongest semi-supervised competitor by 0.019, 0.050, and 0.062, respectively, and surpasses every supervised detector, including DMM, SM3Det, and M2D-LIF. Its gain over C<sup>2</sup>Former widens from +0.108 at 10% to +0.212 at 1%, demonstrating the growing value of unlabeled pairs as annotations shrink. With 10% labeled pairs, ACT recovers 94.3% of the 0.724 full-supervision mAP.

Results on VEDAI. ACT obtains 0.541, 0.545, and 0.632 mAP, exceeding the corresponding second-best results by 0.126, 0.101, and 0.126. With only 38 labeled pairs at the 5% budget, it already surpasses every competing 15% result, whose best mAP is 0.506. Supervised detectors degrade sharply with only 38–115 labeled pairs, whereas ACT exploits all remaining unlabeled pairs. This advantage generalizes to smaller objects and varied alignment.

Label efficiency. Figure 3(a) shows the largest gain at 1%;   
at 10%, ACT attains 94.3% of full supervision.

Per-category comparison. Table 3 reports AP at the 10% budget. Car and bus are near saturation, while the main differences occur on truck, van, and freight car. ACT ranks first on all three target categories and outperforms DMM by

Table 2. VEDAI results across labeling budgets.
<table><tr><td>Method</td><td>Input</td><td>5%</td><td>10%</td><td>15%</td></tr><tr><td>Supervised</td><td></td><td></td><td></td><td></td></tr><tr><td>C2Former [41]</td><td>V+I</td><td>0.130</td><td>0.286</td><td>0.342</td></tr><tr><td>DMM [47]</td><td>V+I</td><td>0.090</td><td>0.221</td><td>0.272</td></tr><tr><td>SM3Det [18]</td><td>I</td><td>0.127 0.064</td><td>0.329</td><td>0.484 0.158</td></tr><tr><td>M2D-LIF [45]</td><td>V+I</td><td></td><td>0.080</td><td></td></tr><tr><td>Semi-supervised</td><td></td><td></td><td></td><td></td></tr><tr><td>MT [31]</td><td>V+I</td><td>0.371</td><td>0.444</td><td>0.488</td></tr><tr><td>SOOD [11]</td><td>I</td><td>0.335</td><td>0.428</td><td>0.506</td></tr><tr><td>DT [46]</td><td>I</td><td>0.284</td><td>0.440</td><td>0.366</td></tr><tr><td>MCL [32]</td><td>I</td><td>0.415</td><td>0.443</td><td>0.483</td></tr><tr><td>ACT (ours)</td><td>V+I</td><td>0.541</td><td>0.545</td><td>0.632</td></tr></table>

Table 3. Per-category AP on DroneVehicle at the 10% annotation ratio.
<table><tr><td>Method</td><td>car</td><td>truck bus</td><td>van</td><td>freight mAP</td></tr><tr><td>C2Former [41] 0.896 0.442 0.872 0.340</td><td></td><td></td><td></td><td>0.326 0.575</td></tr><tr><td>DMM [47]</td><td>0.900 0.555 0.8880.449</td><td></td><td></td><td>0.5210.663</td></tr><tr><td>SM3Det [18] 0.897 0.474 0.881 0.3600.4340.609</td><td></td><td></td><td></td><td></td></tr><tr><td>M2D-LIF [45] 0.889 0.508 0.778 0.412 0.4230.602</td><td></td><td></td><td></td><td></td></tr><tr><td>MT [31]</td><td>0.890 0.492 0.880 0.3640.4350.612</td><td></td><td></td><td></td></tr><tr><td>SOOD [11]</td><td>0.897 0.462 0.862 0.3670.3830.594</td><td></td><td></td><td></td></tr><tr><td>DT [46]</td><td></td><td></td><td></td><td>0.897 0.492 0.8680.4130.4410.622</td></tr><tr><td>MCL [32]</td><td></td><td></td><td></td><td>0.8980.467 0.864 0.3990.4420.614</td></tr><tr><td>ACT (ours)</td><td></td><td></td><td></td><td>0.896 0.612 0.896 0.4820.5280.683</td></tr></table>

Table 4. Component ablation on DroneVehicle.
<table><tr><td colspan="3">Components</td><td colspan="4">Annotation ratio</td></tr><tr><td>CMC-MT</td><td>TG-CMIA</td><td>CRA</td><td>1%</td><td>5%</td><td>10%</td><td>100%</td></tr><tr><td></td><td></td><td></td><td>0.299</td><td>0.485</td><td>0.575</td><td>0.724</td></tr><tr><td>√</td><td></td><td></td><td>0.489</td><td>0.619</td><td>0.665</td><td>一</td></tr><tr><td>√</td><td>√</td><td></td><td>0.510</td><td>0.639</td><td>0.679</td><td>一</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.511</td><td>0.644</td><td>0.683</td><td>一</td></tr></table>

0.057, 0.033, and 0.007, respectively. Infrared-only semisupervised methods lag further behind, supporting the use of paired cues and targeted TG-CMIA supervision.

## 4.4. Ablation and Component Analysis

Overall component contribution. Table 4 sequentially adds the three mechanisms to the supervised baseline at the 1%, 5%, and 10% budgets. Each row reports the peak result of that configuration; the 100% column is only the fullsupervision upper bound. ACT improves the limited-label baseline by 21.2, 15.9, and 10.8 mAP points and reaches 70.6%, 89.0%, and 94.3% of the reference, respectively. Because the mechanisms are added sequentially, each difference is a conditional rather than isolated causal gain.

Component-wise analysis. Effect of CMC-MT. CMC-MT brings the largest incremental gains of 19.0, 13.4, and 9.0 mAP points. The gain is largest at the 1% budget, where nearly all training pairs are unlabeled and reliable pseudo labels are therefore most valuable. Figure 3(b) explains this behavior. Relative to the IR branch, fused consensus improves recall by 1.1 and 2.0 points and F1 by 1.0 and 1.5 points at the 5% and 10% budgets. Precision also increases by 0.9 and 1.2 points, so the recovered branch-wise misses do not come from indiscriminate box retention. Meanwhile, RGB recall trails fused consensus by 19.6 and 19.9 points. These diagnostics connect complementary cross-modal recovery to more reliable supervision on unlabeled pairs.

![](images/7da76a6a5d28811e4720eadf9c414d616d864892325e3522e1c44fb1335e0f3e.jpg)  
(a) Label efficiency

![](images/7d8f34b6596fc63de5081ac9c7282cfe54fdb569d77e3cda4089794ecda69d57.jpg)  
(b) Consensus pseudo-label quality

![](images/df5d988d4af22490812cd15b1102ac99b1bc36160d87ea1b3943a8165176cf1b.jpg)  
(c) Target-class AP with TG-CMIA  
Figure 3. Quantitative analysis on DroneVehicle. (a) Label efficiency across annotation ratios. (b) Teacher pseudo-label quality across RGB, IR, and fused branches. (c) Class-wise effects of TG-CMIA under the two lowest labeling budgets.

Effect of TG-CMIA. TG-CMIA adds 2.1, 2.0, and 1.4 mAP points. Figure 3(c) shows that the target-class mean rises from 0.252 to 0.276 at 1% and from 0.459 to 0.480 at 5%. Its larger gain at 1% is consistent with more severe tailclass scarcity under the lowest budget. At 10%, it rises from 0.511 to 0.522, while car and bus remain nearly unchanged. On the nighttime subset, TG-CMIA adds 1.4 mAP points and improves van AP by 3.7 points.

Effect ofCRA. Finally, CRA adds 0.1, 0.5, and 0.4 mAP points across the three budgets. Because it is added after CMC-MT and TG-CMIA, these values measure its conditional contribution to the complete pipeline. The smaller gain at 1% is consistent with the scarcity of labeled anchors at the lowest budget. Nevertheless, its improvement at every budget shows that regional correspondence remains complementary to consensus supervision. CRA regularizes the fused representation from which CMC-MT derives pseudo labels, linking better regional correspondence to the subsequent self-training stage.

## 4.5. Robustness and Efficiency

Robustness across seeds. The main comparison follows the same single-run peak protocol for all methods. As a robustness check, three runs at the representative 10% budget obtain 0.681 ± 0.001 mAP, only 0.002 below the headline result. Complete per-seed results at the 1%, 5%, and 10%

RGB input  
Baseline  
ACT (ours)  
![](images/1f4d41f1a468b4c266e3857669749ccf72e1649ad9d363290521a742ff570ac4.jpg)  
Figure 4. Qualitative results on DroneVehicle. Rows show day and night scenes; columns show RGB input, baseline predictions (orange), and ACT predictions (green). Red dashed boxes mark targets recovered by ACT.

## budgets are provided in the supplementary material.

Pseudo-label threshold sensitivity. In a controlled sameseed six-epoch study at the 10% budget, $\tau = 0 . 3 , 0 . 4 , 0 . 5 .$ and 0.7 yield 0.647, 0.674, 0.657, and 0.658 mAP, respectively. All four runs use the same labeled subset and training schedule, with only τ changed. Relative to the selected value, the other settings reduce mAP by 2.7, 1.7, and 1.6 points, respectively. Thus, τ = 0.4 provides the best empirical balance between pseudo-label coverage and reliability. We retain this value across annotation budgets rather than tuning it separately for each reported result.

Reference-modality consistency. Primary results use infrared annotations. To rule out dependence on this reference, we also evaluate the same 10% models against RGB annotations. ACT obtains 0.645 mAP versus 0.528 for its supervised baseline, preserving an 11.7-point advantage despite spatial RGB–IR displacement. Thus, the conclusion remains consistent across both reference modalities.

Efficiency. ACT optimizes 150.19M parameters: 120.83M belong to the deployed fused detector, while 29.36M come from training-only alignment and auxiliary modality paths. TG-CMIA also operates only during training and adds no inference parameters. At inference, ACT retains the EMA teacher and its fused branch, matching the supervised baseline’s graph. Both models therefore use 120.83M parameters and reach 3.2 FPS (batch size one, 512 × 640) on an A800 GPU. Thus, the additional mechanisms increase training cost without enlarging the deployment footprint.

## 4.6. Qualitative Detection Analysis

Figure 4 compares ACT with the supervised baseline in daytime and nighttime scenes. In the daytime example, the baseline misses several tightly packed vehicles, whereas ACT recovers more of the group with orientations consistent with the road layout. At night, the RGB observation is nearly dark, but the corresponding IR view retains clear vehicle responses; ACT consequently produces more complete detections in the parking area. The highlighted recoveries show that complementary cross-modal evidence benefits both crowded layouts and severe illumination degradation. They also localize the improvement to genuine baseline misses rather than merely denser predictions on already detected objects. The two scenes therefore separate recovery in a crowded daytime layout from recovery under severe nighttime appearance degradation, complementing the aggregate accuracy reported in the tables.

Limitations. ACT assumes that RGB and IR frames are available as synchronized pairs. CRA accommodates object-level displacement, but it does not remove this pairing requirement. Moreover, the current evaluation focuses on aerial vehicle categories, and tail classes remain difficult when nighttime RGB evidence is severely degraded. Performance under missing modalities, stronger temporal misalignment, and broader object taxonomies has not yet been established. These boundaries motivate extensions to incomplete sensor inputs and stronger vision-language priors.

## 5. Conclusion

This paper studies image-pair-level semi-supervised VIOD, where few RGB–IR pairs are jointly labeled and the rest are unlabeled. ACT combines CRA, CMC-MT, and TG-CMIA to address insufficient alignment supervision, accumulated pseudo-label noise, and tail-class scarcity, respectively. Experiments on DroneVehicle and VEDAI show consistent gains, especially at low labeling budgets. The strongest gains confirm the value of unlabeled pairs when paired supervision is scarce. With 10% labeled pairs on DroneVehicle, ACT retains 94.3% of full-supervision mAP, demonstrating label efficiency across datasets and scene conditions.

Regional correspondence and fused consensus enable label-efficient self-training, while paired augmentation alleviates tail-class scarcity. Nighttime tail classes remain challenging under severely degraded RGB observations, motivating stronger remote-sensing vision-language priors.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[2] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9268–9277, 2019.

[3] Jian Ding, Nan Xue, Yang Long, Gui-Song Xia, and Qikai Lu. Learning RoI transformer for oriented object detection in aerial images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2849–2858, 2019.

[4] Wenhao Dong, Haodong Zhu, Shaohui Lin, Xiaoyan Luo, Yunhang Shen, Guodong Guo, and Baochang Zhang. Fusion-mamba for cross-modality object detection. IEEE Transactions on Multimedia, 27:7392–7406, 2025.

[5] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. International Journal ofComputer Vision, 88(2):303–338, 2010.

[6] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2918–2928, 2021.

[7] Jiaming Han, Jian Ding, Jie Li, and Gui-Song Xia. Align deep features for oriented object detection. IEEE Transactions on Geoscience and Remote Sensing, 60:1–11, 2022.

[8] Jiaming Han, Jian Ding, Nan Xue, and Gui-Song Xia. Re-Det: A rotation-equivariant detector for aerial object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2786– 2795, 2021.

[9] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-´ shick. Mask R-CNN. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2961–2969, 2017.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[11] Wei Hua, Dingkang Liang, Jingyu Li, Xiaolong Liu, Zhikang Zou, Xiaoqing Ye, and Xiang Bai. SOOD: Towards semisupervised oriented object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15558–15567, 2023.

[12] Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon. Multispectral pedestrian detection: Bench-

mark dataset and baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1037–1045, 2015.

[13] Hang Jin, Chenqiang Gao, Junjie Guo, Fangcen Liu, Kanghui Tian, and Qinyao Chang. DOD-SA: Infraredvisible decoupled object detection with single-modality annotations. arXiv preprint arXiv:2508.10445, 2025.

[14] Chan Lee, Seungho Shin, Gyeong-Moon Park, and Jung Uk Kim. Multispectral pedestrian detection with sparsely annotated label. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 4482–4490, 2025.

[15] Gang Li, Xiang Li, Yujie Wang, Yichao Wu, Ding Liang, and Shanshan Zhang. PseCo: Pseudo labeling and consistency training for semi-supervised object detection. In Proceedings of the European Conference on Computer Vision (ECCV), pages 457–472, 2022.

[16] Hengduo Li, Zuxuan Wu, Abhinav Shrivastava, and Larry S. Davis. Rethinking pseudo labels for semi-supervised object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1314–1322, 2022.

[17] Jiaming Li, Xiangru Lin, Wei Zhang, Xiao Tan, Yingying Li, Junyu Han, Errui Ding, Jingdong Wang, and Guanbin Li. Gradient-based sampling for class-imbalanced semisupervised object detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 16390–16400, 2023.

[18] Yuxuan Li, Xiang Li, Yunheng Li, Yicheng Zhang, Yimian Dai, Qibin Hou, Ming-Ming Cheng, and Jian Yang. SM3Det: A unified model for multi-modal remote sensing object detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 6717–6725, 2026.

[19] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He,´ Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2117–2125, 2017.

[20] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2980–2988, 2017.

[21] Yen-Cheng Liu, Chih-Yao Ma, Zijian He, Chia-Wen Kuo, Kan Chen, Peizhao Zhang, Bichen Wu, Zsolt Kira, and Peter Vajda. Unbiased teacher for semi-supervised object detection. In International Conference on Learning Representations (ICLR), 2021.

[22] Yen-Cheng Liu, Chih-Yao Ma, and Zsolt Kira. Unbiased teacher v2: Semi-supervised object detection for anchor-free and anchor-based detectors. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9819–9828, 2022.

[23] Qi Ming, Liuqian Wang, Juan Fang, Xudong Zhao, Yucheng Xu, Ziyi Teng, Yue Zhou, Xiaoxi Hu, Xiaohan Zhang, and Yufei Guo. Hilbert curve-encoded rotation-equivariant oriented object detector with locality-preserving spatial mapping. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8071–8079, 2026.

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen

Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763, 2021.

[25] Sebastien Razakarivony and Frederic Jurie. Vehicle detection in aerial imagery: A small target detection benchmark. Journal ofVisual Communication and Image Representation, 34:187–203, 2016.

[26] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in Neural Information Processing Systems, volume 28, pages 91–99, 2015.

[27] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A. Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. FixMatch: Simplifying semi-supervised learning with consistency and confidence. In Advances in Neural Information Processing Systems, volume 33, pages 596–608, 2020.

[28] Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning. IEEE Transactions on Circuits and Systems for Video Technology, 32(10):6700–6713, 2022.

[29] Jingru Tan, Xin Lu, Gang Zhang, Changqing Yin, and Quanquan Li. Equalization loss v2: A new gradient balance approach for long-tailed object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1685–1694, 2021.

[30] Yihe Tang, Weifeng Chen, Yijun Luo, and Yuting Zhang. Humble teachers teach better students for semi-supervised object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3132–3141, 2021.

[31] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, volume 30, pages 1195– 1204, 2017.

[32] Chenxu Wang, Chunyan Xu, Xiang Li, Yuxuan Li, Xu Guo, Ziqi Gu, and Zhen Cui. Multi-clue consistency learning to bridge gaps between general and oriented objects in semisupervised detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7582–7590, 2025.

[33] Xinjiang Wang, Xingyi Yang, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, and Wayne Zhang. Consistent-teacher: Towards reducing inconsistent pseudo-targets in semi-supervised object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3240–3249, 2023.

[34] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. DOTA: A large-scale dataset for object detection in aerial images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3974–3983, 2018.

[35] Xingxing Xie, Gong Cheng, Jiabao Wang, Xiwen Yao, and

Junwei Han. Oriented R-CNN for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3520–3529, 2021.

[36] Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan Wang, Fangyun Wei, Xiang Bai, and Zicheng Liu. End-toend semi-supervised object detection with soft teacher. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3060–3069, 2021.

[37] Xue Yang, Junchi Yan, Ziming Feng, and Tao He. R3Det: Refined single-stage detector with feature refinement for rotating object. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 3163–3171, 2021.

[38] Xue Yang, Junchi Yan, Qi Ming, Wentao Wang, Xiaopeng Zhang, and Qi Tian. Rethinking rotated object detection with gaussian wasserstein distance loss. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 11830–11841, 2021.

[39] Weiping Yu, Taojiannan Yang, and Chen Chen. Towards resolving the challenge of long-tail distribution in UAV images for object detection. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 3258–3267, 2021.

[40] Maoxun Yuan, Yinan Wang, and Xingxing Wei. Translation, scale and rotation: Cross-modal alignment meets RGBinfrared vehicle detection. In Proceedings of the European Conference on Computer Vision (ECCV), pages 509–525, 2022.

[41] Maoxun Yuan and Xingxing Wei. C<sup>2</sup>former: Calibrated and complementary transformer for rgb-infrared object detection. IEEE Transactions on Geoscience and Remote Sensing, 62:1–12, 2024.

[42] Fangyuan Zhang, Tianxiang Pan, and Bin Wang. Semisupervised object detection with adaptive class-rebalancing self-training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3252–3261, 2022.

[43] Lei Zhang, Yuxuan Sun, and Wei Wei. Mind the gap: Polishing pseudo labels for accurate semi-supervised object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 3463–3471, 2023.

[44] Lu Zhang, Xiangyu Zhu, Xiangyu Chen, Xu Yang, Zhen Lei, and Zhiyong Liu. Weakly aligned cross-modal learning for multispectral pedestrian detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5127–5137, 2019.

[45] Tianyi Zhao, Boyang Liu, Yanglei Gao, Yiming Sun, Maoxun Yuan, and Xingxing Wei. Rethinking multi-modal object detection from the perspective of mono-modality feature learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 6364–6373, 2025.

[46] Hongyu Zhou, Zheng Ge, Songtao Liu, Weixin Mao, Zeming Li, Haiyan Yu, and Jian Sun. Dense teacher: Dense pseudolabels for semi-supervised object detection. In Proceedings of the European Conference on Computer Vision (ECCV), pages 35–50, 2022.

[47] Minghang Zhou, Tianyu Li, Chaofan Qiao, Dongyu Xie, Guoqing Wang, Ningjuan Ruan, Lin Mei, Yang Yang, and Heng Tao Shen. DMM: Disparity-guided multispectral

mamba for oriented object detection in remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 63:1–13, 2025.

[48] Yue Zhou, Xue Yang, Gefan Zhang, Jiabao Wang, Yanyi Liu, Liping Hou, Xue Jiang, Xingzhao Liu, Junchi Yan, Chengqi Lyu, Wenwei Zhang, and Kai Chen. MMRotate: A rotated object detection benchmark using PyTorch. In Proceedings of the 30th ACM International Conference on Multimedia, pages 7331–7334, 2022.