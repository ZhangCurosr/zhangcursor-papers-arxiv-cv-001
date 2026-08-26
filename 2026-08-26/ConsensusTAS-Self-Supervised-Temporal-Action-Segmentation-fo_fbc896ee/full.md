# ConsensusTAS: Self-Supervised Temporal Action Segmentation for Long-Horizon Construction Videos

Xiaoshan Zhou<sup>1</sup> and Yafei Sun<sup>2</sup>

Abstract— Recognizing sequential construction activities is important for collaborative human–robot work; for example, robots are able to understand workers’ current and upcoming actions and provide timely tool delivery or physical support. However, despite extensive research on construction worker activity recognition, existing studies have been limited to classifying activity categories, such as climbing, lifting, and walking, instead of recognizing fine-grained activity transitions from long-horizon sequences. Addressing this problem is challenging because annotating action temporal boundaries in long construction videos is time-consuming. In this study, we propose ConsensusTAS, a label-free, self-supervised learning approach to segment continuous video streams into distinct activity phases by exploiting the internal consensus of candidate segmentations. We evaluated our algorithm on three public datasets, where it outperformed state-of-the-art methods, achieving an F1@10 of 73.08 on GTEA, an F1@10 of 64.33 on Breakfast, and an F1@50 of 33.50 on static-camera videos from Assembly101. We also tested it on real-world construction videos, where post-hoc evaluation showed that the model successfully recognized and segmented actions within the composite activity of bricklaying, such as spreading mortar on a brick, placing the brick, pressing, and aligning. Compared with other temporal action segmentation models that require computationally intensive large vision–language models, our method can run on a CPU, which provides practical value for video surveillance and human–robot collaboration on mobile robotic platforms.

## I. INTRODUCTION

Anticipating workers’ next steps is essential for advancing collaborative construction robotics from supporting structured actions, such as passing a tool, toward coordinating assistance throughout an entire workflow. Current research has sought to understand which activities workers are performing, such as tiling and masonry [1]. The next step is to identify their fine-grained constituent actions. For example, hand-sawing a panel involves positioning the panel, measuring and marking the cutting line, grasping the saw, and cutting. Recognizing these individual actions and their transitions offers a more detailed understanding of the workflow than classifying the entire sequence as panel cutting.

Temporal action segmentation (TAS) of long-horizon videos is challenging because manually annotating framelevel action labels is time-consuming [2]. Although several public datasets capturing activities such as breakfast preparation and tool assembly provide testbeds for TAS model development and evaluation [3], [4], no comparable dataset is currently available for the construction domain. Developing such a dataset requires videos of workers performing natural, multi-step activities on construction sites, such as bricklaying and rebar typing, and a label-free TAS method that can partition these videos into distinct activity phases, which is the purpose of methodological development in this study.

Self-supervised learning offer a promising means of reducing this annotation burden. These methods exploit information inherent in the videos, such as visual similarity between frames and temporal continuity at action boundaries, to discover recurring action phases. For example, SSCAP learns discriminative frame representations through self-supervision and then uses the co-occurrence patterns of sub-actions to infer their temporal structure [2]. More recent approaches introduced object-centric features to identify boundaries of changes in human–object interactions [5], and used temporally consistent unbalanced optimal transport to generate pseudo-labels for sub-actions of varying durations [6].

Despite this progress, existing unsupervised TAS methods have several limitations. SSCAP learns co-occurrence relationships across a corpus of videos, making its segmentation dependent on recurring sub-action patterns [2]. OTAS focuses primarily on detecting boundaries from global and object-centric visual changes; however, visually salient changes do not always correspond to meaningful activity transitions, particularly in construction videos containing camera motion, occlusion, or changes in the surrounding environment [5]. Optimal-transport-based methods improve temporal consistency but require iterative optimization and may remain sensitive to the initial frame–action affinity matrix and the assumed number of latent actions [6].

To address these limitations, we build on the internalobjective design of a progression-aware segmentation framework recently introduced for temporal phase discovery in time series [7] and extend it to the video domain. In ConsensusTAS, candidates generated under varied temporal scales and randomized settings are scored using representationderived internal criteria, and elite candidates are converted into a continuous boundary density from which the final segmentation is decoded. This consensus-based design reduces dependence on any single initialization or temporal scale while retaining a fully label-free workflow.

## II. METHOD

## A. Problem Formulation

Given a video representd by a sequence of frame-level features

$$
X = ( x _ { 1 } , \ldots , x _ { T } ) , \qquad x _ { t } \in \mathbb { R } ^ { D } ,\tag{1}
$$

the objective is to infer an ordered boundary set

$$
B = \{ b _ { 1 } , \ldots , b _ { K } \} , \qquad 1 < b _ { 1 } < \cdot \cdot \cdot < b _ { K } < T ,\tag{2}
$$

without action labels during prediction or calibration.

Here T is sequence length, D is feature dimension, and K is the number of boundaries. The inferred boundaries partition the video into K+1 anonymous temporal segments.

## B. Feature Preprocessing

Each dimension is standardized using its temporal median and median absolute deviation (MAD):

$$
\widetilde { \boldsymbol { x } } _ { t , d } = \mathrm { c l i p } \left( \frac { \boldsymbol { x } _ { t , d } - \mathrm { m e d i a n } _ { t } ( \boldsymbol { x } _ { t , d } ) } { 1 . 4 8 2 6 \mathrm { M A D } _ { t } ( \boldsymbol { x } _ { t , d } ) + \epsilon } , - 8 , 8 \right) .\tag{3}
$$

Features are centered, projected to at most 32 dimensions using randomized PCA. Each projected frame vector is then L2 normalized.

## C. Multiscale boundary evidence

For each sampled scale s and smoothing setting, leftand right-context representations are compared around every frame. The raw boundary score combines cosine dissimilarity and Euclidean motion:

$$
e _ { t } = 0 . 6 5 \left( 1 - \cos ( \ell _ { t } , r _ { t } ) \right) + 0 . 3 5 \left\| \ell _ { t } - r _ { t } \right\| _ { 2 } .\tag{4}
$$

After robust normalization, the default foreground weighting is

$$
e _ { t } ^ { \prime } = e _ { t } \left( 0 . 2 5 + 0 . 7 5 ~ \mathrm { M A } _ { 5 } ( \hat { f } ) _ { t } \right) .\tag{5}
$$

## D. Candidate Population and Internal Reward

The default population contains 48 candidates. Each samples a temporal scale, smoothing width, 55–100% of projected dimensions, and a score quantile uniformly between 0.72 and 0.92. Gaussian perturbation has standard deviation 0.015. For candidate $i ,$ agreement is the mean tolerancebased boundary F1 against the remaining population:

$$
A _ { i } = \frac { 1 } { N - 1 } \sum _ { j \neq i } F _ { \mathrm { m a t c h } } ( B _ { i } , B _ { j } ) .\tag{6}
$$

Boundary contrast $C _ { i }$ is the mean boundary-evidence at its proposed boundaries; compactness $S _ { i }$ compares withinsegment feature variance with global variance. Complexity is

$$
P _ { i } = { \frac { | B _ { i } | \log T } { T } } .\tag{7}
$$

The implemented reward is

$$
R _ { i } = 0 . 5 0 A _ { i } + 0 . 2 5 C _ { i } + 0 . 2 5 S _ { i } - 0 . 0 8 P _ { i } .\tag{8}
$$

The top 25% candidates are retained, with a minimum of three elites.

![](images/fd675757b1d0ef3389ff352a2f8a66be821849416fc52cc0668e6ae4a78f09c8.jpg)

Fig. 1. Sample images from the GETA dataset.  
![](images/f1f393b5be58071f394d0fba24e81657f8d97404eb12b195e11009e97be407e4.jpg)

Fig. 2. Sample images from the Breakfast dataset.  
![](images/8caa04e729127508d4e962483a6e2f75eba8914c1b9ed8a389a0d7ac1110ac6e.jpg)  
Fig. 3. Sample images from the Assembly101 dataset.

## E. Consensus Density

Elite boundaries contribute Gaussian kernels:

$$
d ( t ) = \sum _ { i \in \mathcal { E } } w _ { i } \sum _ { b \in B _ { i } } \exp \left[ - \frac { ( t - b ) ^ { 2 } } { 2 h ^ { 2 } } \right] ,\tag{9}
$$

where

$$
w _ { i } = \frac { \exp ( ( R _ { i } - R _ { \operatorname* { m a x } } ) / 0 . 1 ) } { \sum _ { j \in \varepsilon } \exp ( ( R _ { j } - R _ { \operatorname* { m a x } } ) / 0 . 1 ) } .\tag{10}
$$

The density is normalized by its maximum. Peaks above a dataset/configuration-specific density quantile are retained while enforcing a minimum gap.

## III. EXPERIMENTAL PROTOCOL

## A. Dataset

1) GTEA: Georgia Tech Egocentric Activities (GTEA) contains first-person kitchen videos recorded with a headmounted camera. Four participants each perform seven activities, including making a sandwich, tea or coffee, giving 28 videos in the TAS version. All 28 local feature sequences and four official splits were used. See Fig. 1 [8].

2) Breakfast: The Breakfast Actions Dataset (Breakfast) records people preparing ten breakfast-related items, such as cereal, fried egg, and packages.. It contains 1712 videos from 52 participants. Breakfast is one of the most established benchmarks for TAS models and it contains more variation than GTEA in people, environments, viewpoints and action sequences. See Fig. 2 [3].

3) Assembly101: Assembly101 is a much larger dataset involving participants assembling and disassembling 101 types of toy vehicles. In this study, we used a subset C10119, the static-view feature package with coarse action labels, such as attaching or detaching a vehicle component. It contains 350 useable recording (12 of the 362 recordings in the downloaded package lacked usable labels and were excluded) and 680 resulting assembly/disassembly activity crops. See Fig. 3 [4].

TABLE I  
GTEA RESULTS OVER THREE SEEDS AND FOUR FOLDS. BASELINE RESULTS ARE REPORTED BY AOUAIDJIA ET AL. [9].
<table><tr><td>Method</td><td>F1@10</td></tr><tr><td>ConsensusTAS</td><td> ${ \bf 7 3 . 0 8 \pm 2 . 1 8 }$ </td></tr><tr><td>ASESM, manually selected interval</td><td>70.4</td></tr><tr><td>ASESM, automatic interval</td><td>57.0</td></tr><tr><td>SemiTAS comparison, 5% labels</td><td>59.8</td></tr><tr><td>SemiTAS comparison, 10% labels</td><td>71.5</td></tr></table>

TABLE II

BREAKFAST RESULTS OVER SEEDS AND OFFICIAL FOLDS.
<table><tr><td>Method</td><td>F1@10</td><td>F1@25</td><td>F1@50</td></tr><tr><td>TSA, published</td><td>58.0</td><td></td><td></td></tr><tr><td>SaM, published</td><td>55.9</td><td></td><td></td></tr><tr><td>Uniform 300, local</td><td>63.09</td><td>58.41</td><td>33.60</td></tr><tr><td>Uniform 350, local</td><td>63.36</td><td>59.18</td><td>33.38</td></tr><tr><td>ASESM, local</td><td>63.12</td><td>58.59</td><td>32.77</td></tr><tr><td>ASESM, published</td><td>63.2</td><td></td><td></td></tr><tr><td>ConsensusTAS</td><td colspan="3">64.33 ± 0.70  ${ \bf 6 2 . 4 5 \pm 0 . 9 5 }$   ${ \bf 4 4 . 7 3 \pm 1 . 3 1 }$ </td></tr></table>

## B. Evaluation Metrics

Class-agnostic segment F1 is reported at temporal IoU thresholds $\tau \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 5 0 \}$ . Background is excluded and anonymous predicted segments are greedily matched to reference segments:

$$
F 1 @ \tau = 1 0 0 \frac { 2 T P _ { \tau } } { 2 T P _ { \tau } + F P _ { \tau } + F N _ { \tau } } .\tag{11}
$$

## IV. RESULTS

Tables I, II, and III summarize the results on GTEA, Breakfast, and Assembly101, respectively. Under the classagnostic segment-overlap protocol, ConsensusTAS achieved the strongest reported performance among the compared unsupervised methods on GTEA and Breakfast. On GTEA, ConsensusTAS v1.1 obtained F1 scores of 73.08, 71.07, and 52.40 at overlap thresholds of 10%, 25%, and 50%, respectively. On Breakfast, it achieved 64.33, 62.45, and 44.73. The improvement at F1@50 is particularly notable, indicating that the discovered segments maintain greater temporal overlap with the reference phases rather than merely recovering their approximate locations.

On the C10119 static-view subset of Assembly101, ConsensusTAS outperformed uniform segmentation, changepoint detection, KMeans with smoothing, and the implemented ASOT variants. Its comparison with ASESM was threshold-dependent. Relative to the retrospectively best AS-ESM grid configuration, ConsensusTAS had a paired difference of −5.01 at F1@10, with a 95% confidence interval of $[ - 6 . 4 1 , - 3 . 5 8 ]$ , but a difference of +7.28 at F1@50, with a 95% confidence interval of [5.22, 9.30]. Thus, ASESM recovered more segments under the lenient overlap criterion, whereas ConsensusTAS produced substantially better highoverlap localization. The strongest ASESM configuration was identified through an extensive parameter grid evaluated on the test set though (See Table IV).

TABLE III  
ASSEMBLY101 TEST RESULTS ON 167 ELIGIBLE ACTIVITY CROPS.
<table><tr><td>Method</td><td>F1@10</td><td>F1@25</td><td>F1@50</td></tr><tr><td>Uniform</td><td>48.94</td><td>40.06</td><td>18.05</td></tr><tr><td>Change point</td><td>47.69</td><td>43.81</td><td>23.78</td></tr><tr><td>KMeans + smoothing</td><td> $4 0 . 6 2 \pm 0 . 0 4$ </td><td> $3 7 . 0 1 \pm 0 . 0 6$ </td><td> $2 0 . 4 1 \pm 0 . 2 7$ </td></tr><tr><td>Adapted ASOT</td><td> $4 2 . 5 6 \pm 0 . 3 7$ </td><td> $3 9 . 5 0 \pm 0 . 3 9$ </td><td> $2 2 . 2 5 \pm 0 . 3 7$ </td></tr><tr><td>Local full ASOT</td><td> $2 8 . 8 4 \pm 4 . 3 2$ </td><td> $2 3 . 6 0 \pm 4 . 9 2$ </td><td> $1 1 . 9 0 \pm 2 . 8 8$ </td></tr><tr><td>ConsensusTAS</td><td> $5 4 . 7 9 \pm 0 . 1 0$ </td><td> $5 2 . 6 0 \pm 0 . 0 8$ </td><td> $\mathbf { 3 3 . 5 0 \pm 0 . 2 4 }$ </td></tr><tr><td>ASESM,  $K = 1 6 ,$  gap 225</td><td> ${ \bf 5 9 . 8 1 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 3 . 5 5 \pm 0 . 0 7 }$ </td><td> $2 6 . 2 1 \pm 0 . 1 8$ </td></tr></table>

TABLE IV

ASSEMBLY101 TEST RESULTS ON 167 ELIGIBLE ACTIVITY CROPS.
<table><tr><td>ASESM configuration</td><td>F1@10 F1@25 F1@50</td><td></td></tr><tr><td>Label-free selected:  $K = 1 6 ,$  gap 30</td><td>11.40</td><td>4.55 0.97</td></tr><tr><td>Grid sensitivity: K = 16, gap 225</td><td>59.81 53.55</td><td>26.21</td></tr><tr><td>ConsensusTAŠ</td><td>54.79</td><td>52.60 33.50</td></tr></table>

TABLE V

ASSEMBLY101 ABLATIONS. TEST-SET ANALYSES ARE DIAGNOSTIC.
<table><tr><td>Variant</td><td>F1@10 F1@25</td><td></td><td>F1@50 Interpretation</td></tr><tr><td>Full</td><td>54.79</td><td>52.60 33.50</td><td>Reference</td></tr><tr><td>ConsensusTAS No consistency</td><td>54.80</td><td>52.60 32.96</td><td>Small strict-overlap loss</td></tr><tr><td>reward No foreground</td><td>55.49</td><td>53.42 34.38</td><td>Foreground term unfavorable</td></tr><tr><td>suppression Population 12</td><td>55.00</td><td></td><td>Large population not essential</td></tr><tr><td>Unweighted</td><td>54.76</td><td>52.83 52.59</td><td>33.18 33.52 Weighting has little effect</td></tr><tr><td>density No density</td><td>48.27</td><td>45.73</td><td>28.42 Large degradation</td></tr><tr><td>decoder No internal calibration</td><td>49.73</td><td>47.48</td><td>29.98 Large degradation</td></tr></table>

![](images/73ae7d2b602b851caad747f1c10d2517358448adcd695eda68663c7d2d162c9d.jpg)  
Fig. 4. Timeline and consensus density for the bricklaying video.

In addition, we conducted one-component-at-a-time ablations using the Assembly101 test subset. All ablations retained the same temporal configuration: scales ( $1 0 , 2 0 , 4 0 , 8 0 ) , L _ { \operatorname* { m i n } } = 3 0 , K _ { \operatorname* { m a x } } = 4 0 , h = 1 2 , Q _ { \mathrm { d e n s i t y } } =$ 0.80, and $G _ { \mathrm { m i n } } = 4 5 $ . Experiments used seeds 7, 17, and 27 over 167 eligible activity crops. As shown in Table V, removing density decoding reduced F1@50 from 33.50 to 28.42, while removing internal calibration reduced it to 29.98. These results identify consensus-density decoding and temporal-scale calibration as the most influential components on the Assembly101 dataset, while not yet excluding other heuristics’ role in long, untrimmed recordings containing distinct motion dynamics or irrelevant inactivity.

## V. REAL-WORLD DEMONSTRATIONS

We further examined whether ConsensusTAS could produce interpretable phase boundaries in a previously unseen, unlabeled bricklaying video. This demonstration used a training-free OpenCV front end. Frames were sampled at target 10 frames/s. Each frame was resized to 160 × 160. A normalized $1 6 \times 8$ HSV histogram gives $h _ { t } ~ \in ~ \mathbb { R } ^ { 1 2 8 }$ Grayscale structure and Canny edges (thresholds 60 and 140) are resized to $1 6 \times 1 6 ,$ giving $g _ { t } , e _ { t } \ \in \ \mathbb { R } ^ { 2 5 6 }$ . Farneback flow uses pyramid scale 0.5, three levels, window 15, three iterations, polynomial neighborhood 5, and sigma 1.2. Flow magnitude $m _ { t }$ and directional grids $a _ { t }$ each contribute 64 dimensions. The descriptor is

Bricklaying video: medoid frame of each predicted phase  
![](images/a4f0161c00f830730042f4973aebdbbbbc79dada8ab0d47ae40ca59bd11fd235.jpg)  
Fig. 5. Qualitative visualization of the nine anonymous phases discovered by ConsensusTAS in an unlabeled bricklaying video.

$$
x _ { t } = [ h _ { t } ; g _ { t } ; e _ { t } ; m _ { t } ; a _ { t } ] \in \mathbb { R } ^ { 7 6 8 } ,\tag{12}
$$

then robustly standardized, projected to 32 dimensions using randomized PCA, and $\ell _ { 2 }$ -normalized.

ConsensusTAS divided the 20.9-s video into nine anonymous phases. In Fig. 4, the upper panel displays the predicted intervals, while the lower panel shows the consensus boundary density averaged over seeds:

$$
\bar { d } ( t ) = \frac { 1 } { 3 } \sum _ { r \in \{ 7 , 1 7 , 2 7 \} } d _ { r } ( t ) .\tag{13}
$$

The density peaks coincide with the decoded phase boundaries and provide a visual indication of agreement across stochastic runs. To summarize the visual content of each phase, we extracted the medoid frame whose visual feature is closet to the phase’s mean descriptor:

$$
\mu _ { i } = { \frac { 1 } { | S _ { i } | } } \sum _ { t \in S _ { i } } z _ { t } ,\tag{14}
$$

$$
t _ { i } ^ { * } = \underset { t \in S _ { i } } { \arg \operatorname* { m i n } } \left\| z _ { t } - \mu _ { i } \right\| _ { 2 } .\tag{15}
$$

Figure 5 presents the nine medoid frames in temporal order. The inferred phases were evaluated post-hoc by a human reviewer and their interpretability and practical plausibility are shown in Table VI.

## VI. CONCLUSIONS

This study introduced ConsensusTAS, a label-free approach to TAS for long-horizon construction videos. Preliminary results demonstrated its superior performance on public benchmarks and produced plausible action segmentation of real-world construction activities. Future work will scale the method across diverse activities and establish a video benchmark for context-aware human–robot collaboration in long-horizon construction workflows.

TABLE VI  
TENTATIVE VISUAL INTERPRETATION OF THE IDENTIFIED PHASES.
<table><tr><td>Phase</td><td>Interval</td><td>Visual interpretation</td></tr><tr><td>1</td><td>0.0–7.0 s</td><td>Holding the brick, collecting and spreading mortar</td></tr><tr><td>2</td><td>7.0–9.3 s</td><td>Completing mortar application</td></tr><tr><td>3</td><td></td><td>9.3-10.0 s Transition into placement</td></tr><tr><td>4</td><td></td><td>10.0–13.2 s Inserting and initially positioning the closure brick</td></tr><tr><td>5</td><td></td><td>13.2–15.6 s Pressing and aligning the brick</td></tr><tr><td>6</td><td></td><td>15.6–16.7 s Short joint or edge adjustment</td></tr><tr><td>7</td><td></td><td>16.7–17.8 s Continued alignment or mortar adjustment</td></tr><tr><td>8</td><td></td><td>17.8–18.6 s Brief finishing movement</td></tr><tr><td>9</td><td></td><td>18.6–20.9 s Final placement and hand withdrawal</td></tr></table>

## REFERENCES

[1] E. Monfared and Y. Alipouri, “On-site construction worker activity monitoring using deep learning,” Journal of Computing in Civil Engineering, vol. 40, no. 4, p. 04026040, 2026. [Online]. Available: https://doi.org/10.1061/JCCEE5.CPENG-7234

[2] Z. Wang, H. Chen, X. Li, C. Liu, Y. Xiong, J. Tighe, and C. Fowlkes, “Sscap: Self-supervised co-occurrence action parsing for unsupervised temporal action segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

[3] H. Kuehne, A. Arslan, and T. Serre, “The language of actions: Recovering the syntax and semantics of goal-directed human activities,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 780–787.

[4] F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, and A. Yao, “Assembly101: A large-scale multi-view video dataset for understanding procedural activities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 21 064–21 074.

[5] Y. Li, Z. Xue, and H. Xu, “OTAS: Unsupervised boundary detection for object-centric temporal action segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 6423–6432.

[6] M. Xu and S. Gould, “Temporally consistent unbalanced optimal transport for unsupervised action segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 618–14 627.

[7] X. Zhou, C. C. Menassa, and V. R. Kamat, “Self-supervised evolutionary learning for progression-aware segmentation of neurodynamic time series during safety-critical decision making,” Cognitive Neurodynamics, vol. 20, p. 145, 2026.

[8] A. Fathi, X. Ren, and J. M. Rehg, “Learning to recognize objects in egocentric activities,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2011, pp. 3281–3288.

[9] K. Aouaidjia, W. Zhang, A. Li, and C. Zhang, “Improving action segmentation via explicit similarity measurement,” Engineering Appli cations of Artificial Intelligence, vol. 164, p. 113251, 2026.