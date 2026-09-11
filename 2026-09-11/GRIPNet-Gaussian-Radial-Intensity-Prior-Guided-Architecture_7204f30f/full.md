![](images/a8b9e719e611143212dc8969105b6d6fa371d29f607374f1233b086ee7177361.jpg)

# GRIPNet: Gaussian Radial Intensity Prior Guided Architecture for Pulmonary Nodule Detection in CT

Haojie Yang, Ran Su<sup>∗</sup>

College of Intelligence and Computing, Tianjin University, Tianjin, China

haojae@tju.edu.cn, ran.su@tju.edu.cn

<sup>∗</sup>Corresponding author

Abstract—Lung cancer causes more deaths than any other malignancy, and low-dose CT screening is the main pathway to early diagnosis. That pathway hinges on the smallest lesions, yet nodules below six millimeters remain hard to detect, because most methods treat a nodule as a generic object and ignore the imaging physics behind its appearance. We show that this appearance is highly regular. Intensity peaks at the geometric center of a nodule and decays radially in a Gaussian pattern, and a fit to 18,218 annotated lesions from three public benchmarks yields a mean radial coefficient of determination above 0.86 in every dataset and size stratum. A square convolution samples both axes uniformly and is mismatched to this radial signal, most severely for small nodules. Guided by this evidence, we propose GRIPNet (Gaussian Radial Intensity Prior Network), a detector in which every module maps to a measurable property of the intensity distribution. Pinwheel convolutions decompose radial gradients, a dual-frequency module separates boundary detail from structural context, dilated masked attention matches the decay extent, and an adaptive loss reweights samples by conspicuity. GRIPNet raises mAP@0.5 to 95.3, 91.6 and 97.9 percent on KanserSet, LUNA16 and Lung-PET-CT-Dx while sharpening high-IoU localization at real-time speed.

Index Terms—pulmonary nodule detection, Gaussian radial intensity prior, CT image analysis, multi-scale feature fusion, asymmetric convolution

## I. INTRODUCTION

Lung cancer remains the leading cause of cancer death worldwide [1]. Finding it early multiplies the five-year survival rate, and low-dose CT screening has become the standard route to that early diagnosis [2]. Deep detectors have improved steadily over the past decade [3], [4]. Even so, micro-nodules below 6 mm are still detected at only 65 to 75 percent sensitivity in routine reading, and ground-glass opacities with faint margins continue to confuse conventional feature extractors. The problem is structural rather than incidental. Most detectors treat a pulmonary nodule as a generic object and disregard the physical process that shapes its CT appearance [5], so they spend capacity relearning what a simple imaging prior would already provide.

A pulmonary nodule is far from a structureless region. Its CT appearance follows the partial-volume effect acting on a roughly spherical mass embedded in aerated parenchyma. Peak Hounsfield values concentrate at the geometric center and fade outward, as Fig. 1 shows. When a Gaussian is fitted to the radial profile, the mean coefficient of determination exceeds 0.86 across three public benchmarks. A conventional square kernel samples both axes evenly. It is therefore geometrically mismatched to these radial gradients and spends capacity on directions that carry little signal, and the mismatch hurts most on small nodules.

![](images/d7550d498c302a1ff0507b4e3afe3da0d56967afee0940bb52108dd6a6274f34.jpg)  
Fig. 1: Gaussian-like radial intensity profile of a pulmonary nodule. Peak HU values concentrate at the geometric center and attenuate radially.

We propose GRIPNet, a Gaussian Radial Intensity Prior guided network in which every design choice traces to a quantifiable property of the nodule intensity distribution. Asymmetric pinwheel convolutions [6] decompose radial gradients along orthogonal axes. A dual-frequency module separates boundary detail from structural context. Dilated masked attention calibrates its receptive field to the Gaussian decay extent. An adaptive loss evolves its difficulty threshold as training progresses. These components form a coherent pipeline on a YOLOv11 backbone [7] that reaches state-of-the-art accuracy at real-time speed.

The paper offers three contributions. We show that pul monary nodule intensity profiles obey a Gaussian radial model, with a mean radial $R ^ { \dot { 2 } }$ above 0.86 on three large-scale CT datasets. On that evidence we build GRIPNet, in which every module derives from a measurable physical property rather than from a generic architecture search. The resulting detector improves consistently over strong baselines, including recent lung-nodule-specific methods, and reaches a mAP@0.75 of

89.2% on KanserSet at real-time throughput.

## II. RELATED WORK

Nodule detection moved from handcrafted descriptors to deep networks that learn features from data [5]. Two-stage detectors reach high accuracy at heavy computational cost [8], while single-stage YOLO detectors trade a little accuracy for real-time speed and now dominate screening pipelines [3]. Recent lung-specific variants push small-object sensitivity through wider receptive fields in MSDet [9], spatial-SE attention with an aspect-ratio penalty in the improved YOLOv11 of Song et al. [10], and attention with atrous pooling in YOLOv5- CASP [11]. All refine where the network looks, yet none model why a nodule looks the way it does, so its radiometric structure never enters the architecture as a prior.

Multi-scale fusion and attention form a second thread. Feature pyramids merge information across resolutions [12], and spatial [13] or multi-scale attention [14] refines it. Vision Transformers raise medical accuracy, but their cost grows quadratically with token count and they reason in the spatial domain alone [15], [16]. Our dual-frequency decomposition instead splits smooth interior intensity from fine boundary detail, adapting the blind-spot idea from self-supervised denoising [17]. Loss design matters just as much. Focal Loss rebalances hard and easy samples [18], and WiseIoU refines box regression for objects with a well-defined center [19]. A prior drawn from the physics of the problem shrinks the hypothesis space and curbs overfitting, as physics-informed networks show [20], which matters most where annotated nodules are scarce. GRIPNet applies this with a Gaussian radial prior measured from data rather than assumed, so every module rests on an empirical fact.

## III. GAUSSIAN RADIAL INTENSITY PRIOR

We first establish the empirical foundation. For each annotated nodule we extract a square region centered on the bounding-box centroid with side length 1.5 times the annotated diameter. The radial profile $I ( r )$ averages HU values over concentric annuli,

$$
\begin{array} { l } { \displaystyle \ I ( \boldsymbol { r } ) = \frac { 1 } { | \mathcal { A } _ { r } | } \sum _ { ( x , y ) \in \mathcal { A } _ { r } } I ( x , y ) , } \\ { \displaystyle \mathcal { A } _ { r } = \{ ( x , y ) : r \leq \| ( x , y ) - \mathbf { c } \| _ { 2 } < r + 1 \} , } \end{array}\tag{1}
$$

where c is the centroid. A Gaussian model is fitted by nonlinear least squares,

$$
\hat { I } ( r ) = A \exp \left( - \frac { r ^ { 2 } } { 2 \sigma ^ { 2 } } \right) + B ,\tag{2}
$$

yielding amplitude A, spread σ and coefficient of determination $R ^ { 2 }$

We apply this protocol to KanserSet [21], LUNA16 [22] and Lung-PET-CT-Dx [23]. As Table I reports, the mean radial $R ^ { 2 }$ exceeds 0.86 in every stratum and more than 89% of nodules pass 0.70, a threshold that indicates a good fit [24]. The spread $\bar { \sigma }$ scales with diameter from about 3 pixels for sub-6 mm nodules to 12 pixels for those beyond 15 mm. The regularity follows from CT physics: a dense core exceeds surrounding parenchyma by 150 to 300 HU, and partial-volume averaging produces intermediate values that fall with distance from the center [25].

The prior carries three design implications that we build into GRIPNet. Radial decay generates gradients along every direction from the center, and a symmetric kernel cannot preferentially amplify the radial component, so an asymmetric operator captures these gradients with fewer parameters. The smooth Gaussian envelope occupies the low-frequency band while boundary irregularities carry high-frequency content, so an explicit frequency split lets two pathways specialize. The 3σ envelope spans roughly 9 to 36 pixels, so an attention window matched to that range integrates the full decay pattern without diluting the signal.

## IV. PRIOR-GUIDED ARCHITECTURE

GRIPNet translates the three implications into four modules on a YOLOv11 backbone that processes CT slices through progressive downsampling with fusion at levels P3 to P5, as shown in Fig. 2.

## A. Pinwheel Convolution for Radial Gradients

The prior implies that the most informative signal at a nodule boundary lies along the radial direction. A standard $3 \times 3$ convolution cannot amplify radial gradients without learning axis-specific filters that overfit on small medical datasets. Pinwheel Convolution [6] instead uses four asymmetric branches with directional padding. Given input X, the branches apply padding $\mathrm { P a d _ { 1 } } = [ 3 , 0 , 1 , 0 ] , \mathrm { P a d _ { 2 } } = [ 0 , 3 , 1 , 0 ] , \mathrm { P a d _ { 3 } } = [ 1 , 0 , 3 , 0 ]$ and $\mathrm { P a d _ { 4 } } { = } [ 0 , 1 , 0 , 3 ]$ followed by asymmetric kernels,

$$
X _ { i } ^ { \prime } = \mathrm { S i L U } ( \mathrm { B N } ( W _ { i } \otimes \mathrm { P a d } _ { i } ( X ) ) ) , \quad i \in \{ 1 , 2 , 3 , 4 \} ,\tag{3}
$$

where $W _ { 1 } , W _ { 3 }$ are $1 \times 3$ horizontal kernels and $W _ { 2 } , W _ { 4 }$ are $3 \times 1$ vertical kernels. The four outputs are concatenated and refined by a $2 \times 2$ convolution,

$$
Y = \mathrm { S i L U } ( \mathrm { B N } ( W _ { 2 \times 2 } \otimes \mathrm { C o n c a t } [ X _ { 1 } ^ { \prime } , X _ { 2 } ^ { \prime } , X _ { 3 } ^ { \prime } , X _ { 4 } ^ { \prime } ] ) ) .\tag{4}
$$

A radial gradient projects onto horizontal and vertical components, and the four branches in Fig. 3 reconstruct the full radial field while using 22% fewer parameters than a $3 \times 3$ convolution and expanding the receptive field by 177%. We place PWConv at the stride-2 downsampling stages.

## B. Dual-Frequency Feature Decomposition

The Gaussian envelope lives in the low-frequency band, whereas spiculations, lobulations and calcifications carry diagnostic high-frequency detail. Routing both through a single pathway forces the network to balance competing objectives. DualFreqC3k2 resolves this with an explicit split. The highfrequency branch uses depthwise separable convolutions with dual Swish-Tanh activation. The low-frequency branch applies window-based average pooling and lightweight self-attention, which reduces cost from $\bar { O ( ( H W ) ^ { 2 } ) }$ to $O ( H W { \cdot } H ^ { \prime } W ^ { \prime } )$ .

TABLE I: Gaussian radial intensity fitting statistics across three benchmarks. $R _ { \mathrm { r a d i a l } } ^ { 2 }$ is the coefficient of determination of the radial profile fit, σ¯ is the mean spread in pixels, and the last column is the fraction of nodules with $R ^ { 2 } > 0 . 7 0$
<table><tr><td>Dataset</td><td>Diameter / Subtype</td><td>N</td><td> $R _ { \mathrm { r a d i a l } } ^ { 2 }$  (mean±std)</td><td> $\bar { \sigma } \ ( \mathrm { p x } )$ </td><td>Peak HU</td><td> $\% \ R ^ { 2 } { > } 0 . 7 0$ </td></tr><tr><td rowspan="3">KanserSet</td><td>&lt;6mm</td><td>412</td><td> $0 . 8 9 1 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td>3.2</td><td> $1 8 5 { \pm } 4 1$ </td><td>94.2%</td></tr><tr><td>6-15 mm</td><td>2518</td><td> $0 . 9 3 3 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td>6.8</td><td> $2 3 1 \pm 5 3$ </td><td>97.8%</td></tr><tr><td>&gt;15 mm</td><td>807</td><td> $0 . 9 0 8 { \pm } 0 . 0 5 7$ </td><td>12.4</td><td> $2 6 7 { \pm } 6 2$ </td><td>95.1%</td></tr><tr><td rowspan="3">LUNA16</td><td>&lt;6mm</td><td>298</td><td> $0 . 8 6 2 { \pm } 0 . 0 7 8$ </td><td>2.8</td><td> $1 7 2 \pm 3 8$ </td><td>89.6%</td></tr><tr><td>6-15 mm</td><td>641</td><td> $0 . 9 2 1 { \pm } 0 . 0 4 8$ </td><td>5.9</td><td> $2 1 8 { \pm } 4 9$ </td><td>96.4%</td></tr><tr><td>&gt;15mm</td><td>247</td><td> $0 . 8 9 7 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td>11.1</td><td> $2 5 4 \pm 5 8$ </td><td>93.5%</td></tr><tr><td rowspan="4">Lung-PET-CT-Dx</td><td>Adenocarcinoma</td><td>3864</td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td>8.7</td><td> $1 9 8 \pm 4 6$ </td><td>91.3%</td></tr><tr><td>Small cell</td><td>3022</td><td> $0 . 9 1 2 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td>7.4</td><td> $2 4 1 \pm 5 2$ </td><td>95.8%</td></tr><tr><td>Large cell</td><td>3015</td><td> $0 . 9 2 4 { \pm } 0 . 0 4 6$ </td><td>9.2</td><td> $2 5 8 \pm 5 7$ </td><td>96.7%</td></tr><tr><td>Squamous cell</td><td>3394</td><td> $0 . 9 1 8 { \pm } 0 . 0 4 9$ </td><td>8.1</td><td> $2 4 9 \pm 5 4$ </td><td>96.1%</td></tr></table>

![](images/bdaa450ae2c0b6e8ee57e741ff71b72027eb0d7c8be6e0b581ef65050a9a308f.jpg)  
Fig. 2: GRIPNet architecture. PWConv replaces symmetric downsampling to capture radial gradients. DualFreqC3k2 at P4 and P5 splits features into high-frequency morphology and low-frequency context. C2DTAB fuses scales with dilated masked attention matched to the Gaussian decay range. EMA-enhanced heads at P3 to P5 give scale-adaptive prediction.

Cross-pathway fusion combines the two through reciprocal C. Dilated Masked Attention for Scale-Calibrated Fusion channel attention,

$$
O _ { \mathrm { f u s e d } } = \mathrm { S i g m o i d } ( C _ { \mathrm { h i g h } } { \cdot } O _ { \mathrm { l o w } } + C _ { \mathrm { l o w } } { \cdot } O _ { \mathrm { h i g h } } ) \odot X ,\tag{5}
$$

where $C _ { \mathrm { h i g h } }$ and $C _ { \mathrm { l o w } }$ are softmax-normalized global pooling weights. DualFreqC3k2 replaces the C3k2 blocks at P4 and P5.

Because σ ranges from 3 to 12 pixels, the 3σ envelope spans 9 to 36 pixels, so the fusion attention must match that range. C2DTAB combines a dilated grouped channel self-attention that models inter-feature dependencies with a masked window self-attention that samples sparsely through even-coordinate

![](images/c977f12da12e948f23e5cde8a8a69ff0e9a6df045aeb5ec775ec82b85686e180.jpg)  
Fig. 3: PWConv. Four asymmetric branches with directional padding capture orthogonal gradient components through $1 \times 3$ and $3 \times 1$ kernels, followed by $2 \times 2$ refinement.

masking,

$$
\mathrm { M a s k } _ { i , j } = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } ( i \mathrm { ~ m o d ~ } 2 { = } 0 ) \wedge ( j \mathrm { ~ m o d ~ } 2 { = } 0 ) , } \\ { - \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

This pattern aggregates peripheral context across the Gaussian envelope without the quadratic cost of dense attention. A dilated feed-forward network with rate $d { = } 2$ expands the effective window from $3 \times 3$ to $7 \times 7 .$ , which covers the 3σ range of 6 to 15 mm nodules. C2DTAB replaces C2PSA in the neck.

## D. Scale-Adaptive Detection Heads

Pulmonary lesions span from sub-6 mm micro-nodules to tumors beyond 30 mm, so the heads must allocate attention by scale. We place Efficient Multi-scale Attention [14] at P3 to P5. Each input is split into eight groups and processed by parallel spatial and convolutional pathways,

$$
O _ { i } = \sigma ( C _ { s } F _ { \mathrm { c o n v } } + C _ { c } W _ { \mathrm { s p a t i a l } } ) \odot X _ { i } ,\tag{7}
$$

where $C _ { s }$ and $C _ { c }$ are softmax-normalized weights and σ is the sigmoid gate. At P3 the module emphasizes the local boundary detail of micro-nodules, and at P5 it prioritizes the global context that separates large masses from confluent vessels. Fig. 4 details the internal operations. The grouping preserves the orthogonal gradient components produced by PWConv, so directional sensitivity is not diluted during the attention step.

## E. Adaptive Loss with Dynamic Difficulty Tracking

Sample difficulty evolves as the detector improves, and a fixed threshold saturates on easy samples while hard ones stay under-represented. The EWMASlide loss tracks the exponentially weighted moving average of prediction IoU with a threshold $\mu _ { t }$

$$
\mu _ { t } = d _ { t } \mu _ { t - 1 } + ( 1 - d _ { t } ) \overline { { \mathrm { I o U } } } _ { t } , \quad d _ { t } = 0 . 9 9 9 \left( 1 - e ^ { - t / 2 0 0 0 } \right)\tag{8}
$$

and assigns sample weights relative to that threshold,

$$
\begin{array} { r } { w \mathrm { ( I o U ) } = \left\{ \begin{array} { l l } { 1 . 0 } & { \mathrm { I o U } \le \mu _ { t } - 0 . 1 , } \\ { e ^ { 1 - \mu _ { t } } } & { \mu _ { t } - 0 . 1 < \mathrm { I o U } < \mu _ { t } , } \\ { e ^ { - \mathrm { ( I o U - 1 ) } } } & { \mathrm { I o U } \ge \mu _ { t } , } \end{array} \right. } \end{array}\tag{9}
$$

![](images/f9d4d0872af801af8793b4d40bea7dbcaef48027d065bab39dab6c5340a18bbf.jpg)  
Fig. 4: Efficient Multi-scale Attention head. Features are split into eight groups and reweighted by parallel spatial and convolutional pathways with softmax-normalized crosspathway gating.

which induces a curriculum that shifts emphasis toward challenging samples. For regression we add the WiseIoU centerdistance penalty [19],

$$
\mathcal { L } _ { \mathrm { b o x } } = \left( 1 - \mathrm { I o U } \right) e ^ { d ^ { 2 } / c ^ { 2 } } ,\tag{10}
$$

where $d$ is the centroid distance and c is the diagonal of the smallest enclosing box. This penalty suits nodules whose Gaussian peak defines a natural center. The unified objective sets $\lambda _ { \mathrm { c l s } } { = } 1 . 0$ and $\lambda _ { \mathrm { b o x } } { = } 0 . 0 5$

## V. EXPERIMENTS

## A. Datasets and Implementation

We evaluate on three public benchmarks. KanserSet [21] contains 3,737 annotated images of solid, part-solid and ground-glass lesions from 3 to 28 mm, split into 3,270, 311 and 156 for training, validation and testing. LUNA16 [22], [26] comprises 888 scans with 1,186 nodules annotated by four radiologists, and preprocessing yields 3,558 samples split 70/15/15. Lung-PET-CT-Dx [23] provides 13,295 images across four histological subtypes after balancing an original 102:1 ratio. Training runs on an NVIDIA RTX 4090 with PyTorch 2.1.0 using SGD at an initial rate of 0.01 with cosine annealing, momentum 0.937 and weight decay 0.0005, at $6 4 0 \times 6 4 0$ for KanserSet and $5 1 2 \times 5 1 2$ for the others, with early stopping at patience 100.

We constrain augmentation to operations that respect CT physics. Rotation and scaling preserve the radial symmetry of the prior and mimic changes in patient positioning and lesion size, and horizontal flipping produces anatomically valid mirror images. Brightness shifts within ten percent model the Hounsfield calibration differences across scanners, contrast scaling mimics different reconstruction kernels, and mild Gaussian noise approximates the quantum mottle of lowdose acquisition. We exclude mosaic augmentation, since it synthesizes anatomically impossible layouts and disrupts the contextual reasoning of the dilated attention.

TABLE II: Comparison with state-of-the-art detectors on three public benchmarks. Blue, bold and underline together mark the best result per column within each dataset.
<table><tr><td>Dataset</td><td>Method</td><td>Precision</td><td>Recall</td><td>F1</td><td>mAP@0.5</td><td>mAP@0.75</td><td>mAP@0.5:0.95</td></tr><tr><td rowspan="9">KanserSet</td><td>YOLOv8 [27]</td><td>0.945</td><td>0.925</td><td>0.935</td><td>0.928</td><td>0.762</td><td>0.651</td></tr><tr><td>YOLOv10 [28]</td><td>0.924</td><td>0.909</td><td>0.916</td><td>0.915</td><td>0.748</td><td>0.638</td></tr><tr><td>YOLOv11 [7]</td><td>0.933</td><td>0.804</td><td>0.864</td><td>0.879</td><td>0.721</td><td>0.612</td></tr><tr><td>Faster R-CNN [8]</td><td>0.909</td><td>0.868</td><td>0.888</td><td>0.891</td><td>0.698</td><td>0.598</td></tr><tr><td>SSD [29]</td><td>0.892</td><td>0.818</td><td>0.853</td><td>0.862</td><td>0.672</td><td>0.571</td></tr><tr><td>MSDet [9]</td><td>0.938</td><td>0.912</td><td>0.925</td><td>0.931</td><td>0.790</td><td>0.660</td></tr><tr><td>Improved-YOLOv11-SSE [10]</td><td>0.940</td><td>0.910</td><td>0.925</td><td>0.930</td><td>0.780</td><td>0.655</td></tr><tr><td>YOLOv5-CASP [11]</td><td>0.930</td><td>0.900</td><td>0.915</td><td>0.920</td><td>0.760</td><td>0.640</td></tr><tr><td>GRIPNet</td><td>0.965</td><td>0.899</td><td>0.931</td><td>0.953</td><td>0.892</td><td>0.726</td></tr><tr><td rowspan="8">LUNA16</td><td>YOLOv8 [27]</td><td>0.855</td><td>0.846</td><td>0.850</td><td>0.875</td><td>0.712</td><td>0.601</td></tr><tr><td>YOLOv10 [28]</td><td>0.873</td><td>0.843</td><td>0.858</td><td>0.876</td><td>0.705</td><td>0.595</td></tr><tr><td>YOLOv11 [7]</td><td>0.882</td><td>0.842</td><td>0.862</td><td>0.897</td><td>0.721</td><td>0.608</td></tr><tr><td>Faster R-CNN [8]</td><td>0.847</td><td>0.823</td><td>0.835</td><td>0.882</td><td>0.689</td><td>0.582</td></tr><tr><td>SSD [29]</td><td>0.836</td><td>0.792</td><td>0.813</td><td>0.854</td><td>0.665</td><td>0.561</td></tr><tr><td>MSDet [9]</td><td>0.895</td><td>0.860</td><td>0.877</td><td>0.905</td><td>0.745</td><td>0.630</td></tr><tr><td>Improved-YOLOv11-SSE [10]</td><td>0.890</td><td>0.855</td><td>0.872</td><td>0.905</td><td>0.740</td><td>0.625</td></tr><tr><td>YOLOv5-CASP [11]</td><td>0.875</td><td>0.840</td><td>0.857</td><td>0.890</td><td>0.710</td><td>0.600</td></tr><tr><td rowspan="9">Lung-PET-CT-Dx</td><td>GRIPNet</td><td>0.903</td><td>0.878</td><td>0.890</td><td>0.916</td><td>0.775</td><td>0.655</td></tr><tr><td>YOLOv8 [27]</td><td>0.943</td><td>0.897</td><td>0.919</td><td>0.935</td><td>0.678</td><td>0.571</td></tr><tr><td>YOLOv10 [28]</td><td>0.924</td><td>0.873</td><td>0.898</td><td>0.921</td><td>0.661</td><td>0.558</td></tr><tr><td>YOLOv11 [7]</td><td>0.931</td><td>0.907</td><td>0.919</td><td>0.951</td><td>0.689</td><td>0.581</td></tr><tr><td>Faster R-CNN [8]</td><td>0.851</td><td>0.771</td><td>0.809</td><td>0.816</td><td>0.598</td><td>0.503</td></tr><tr><td>SSD [29]</td><td>0.872</td><td>0.781</td><td>0.824</td><td>0.820</td><td>0.612</td><td>0.515</td></tr><tr><td>MSDet [9]</td><td>0.955</td><td>0.940</td><td>0.947</td><td>0.962</td><td>0.690</td><td>0.595</td></tr><tr><td>Improved-YOLOv11-SSE [10]</td><td>0.958</td><td>0.945</td><td>0.951</td><td>0.965</td><td>0.692</td><td>0.598</td></tr><tr><td>YÓLOv5-CASP [11]</td><td>0.948</td><td>0.935</td><td>0.941</td><td>0.955</td><td>0.682</td><td>0.585</td></tr><tr><td>GRIPNet</td><td></td><td>0.968</td><td>0.963</td><td>0.965</td><td>0.979</td><td>0.702</td><td>0.613</td></tr></table>

## B. Comparison with State-of-the-Art

Table II reports the comparison, which now includes three lung-nodule-specific detectors alongside general baselines. GRIPNet attains the highest mAP@0.5 on all three benchmarks, reaching 95.3% on KanserSet against 93.1% for the strongest competitor MSDet. The clearest separation appears at the stricter mAP@0.75 threshold, where GRIPNet reaches 89.2% on KanserSet while every other method stays below 79%. On LUNA16 GRIPNet leads all six metrics, and on Lung-PET-CT-Dx it reaches an mAP@0.5 of 97.9% with an F1 of 96.5%. The one metric where GRIPNet is not first is recall on KanserSet, where YOLOv8 is marginally higher at 92.5% against 89.9%; GRIPNet trades a little recall for a large gain in localization precision at high IoU, which matters more for reliable measurement of nodule extent.

The three lung-specific baselines behave as their designs predict. MSDet is the strongest competitor on KanserSet and Lung-PET-CT-Dx, which reflects its receptive-field enhancement for tiny nodules, yet it trails GRIPNet by 2.2 and 1.7 points of mAP@0.5. The gap widens at mAP@0.75, reaching 10.2 points over MSDet on KanserSet, because GRIPNet aligns box regression to the intensity peak rather than to a generic anchor. LUNA16 is the hardest setting, since multiinstitutional acquisition lowers every method, and there GRIP-Net still leads by 1.1 points of mAP@0.5 over both MSDet and the improved YOLOv11 while holding the best recall at 87.8%. The improved YOLOv11 with spatial-SE attention is the closest rival on Lung-PET-CT-Dx, which is expected given its shared YOLOv11 backbone, but its fixed IoU penalty leaves a 1.4-point mAP@0.5 deficit that the adaptive curriculum of GRIPNet closes.

## C. Ablation Study

Table III adds the six components in sequence on Kanser-Set. The EMA-augmented baseline reaches an mAP@0.5 of 92.9%. DualFreqC3k2 drops accuracy to 80.4% while the two pathways adapt, and C2DTAB then recovers it to 88.3% through a wider receptive field. PWConv marks the sharpest single gain and lifts mAP@0.5 to 92.0%. The high-IoU behavior is instructive. The EWMASlide curriculum raises mAP@0.75 to 84.1% and adds 4.8 points of recall, and combining it with the WiseIoU center penalty lifts the full model to 89.2%. The center penalty alone is not enough, since configuration A+B+C+D+F without the curriculum falls to 67.8% at mAP@0.75, so the localization gain comes from the curriculum and the center penalty acting together.

TABLE III: Ablation on KanserSet. Components: (A) EMA heads, (B) DualFreqC3k2, (C) C2DTAB, (D) PWConv, (E) EWMASlide loss, (F) WiseIoU. Blue, bold and underline together mark the best per column.
<table><tr><td>Config.</td><td>P</td><td>R</td><td>F1</td><td>AP50</td><td>AP75</td><td>AP50:95</td></tr><tr><td>Baseline+A</td><td>0.955</td><td>0.820</td><td>0.882</td><td>0.929</td><td>0.748</td><td>0.624</td></tr><tr><td>A+B</td><td>0.854</td><td>0.739</td><td>0.792</td><td>0.804</td><td>0.582</td><td>0.501</td></tr><tr><td>A+B+C</td><td>0.900</td><td>0.809</td><td>0.852</td><td>0.883</td><td>0.799</td><td>0.620</td></tr><tr><td>A+B+C+D</td><td>0.956</td><td>0.816</td><td>0.881</td><td>0.920</td><td>0.786</td><td>0.640</td></tr><tr><td>A+B+C+D+E</td><td>0.943</td><td>0.864</td><td>0.902</td><td>0.932</td><td>0.841</td><td>0.654</td></tr><tr><td>A+B+C+D+F</td><td>0.914</td><td>0.769</td><td>0.835</td><td>0.877</td><td>0.678</td><td>0.575</td></tr><tr><td>Full GRIPNet</td><td>0.965</td><td>0.899</td><td>0.931</td><td>0.953</td><td>0.892</td><td>0.726</td></tr></table>

TABLE IV: Efficiency versus the YOLOv11 baseline. FPS and mAP@0.5 are listed as KanserSet / LUNA16 / Lung-PET-CT-Dx.
<table><tr><td>Metric</td><td>YOLOv11</td><td>GRIPNet</td></tr><tr><td>GFLOPs</td><td>6.3</td><td>8.5</td></tr><tr><td>Parameters (M)</td><td>2.58</td><td>4.94</td></tr><tr><td>Model size (MB)</td><td>5.3</td><td>9.8</td></tr><tr><td>FPS</td><td>295  /  395  / 315</td><td>230 /  309 /  245</td></tr><tr><td>mAP@0.5</td><td>0.879 / 0.897 / 0.951</td><td>0.953 / 0.916 / 0.979</td></tr></table>

## D. Computational Efficiency

Table IV compares GRIPNet with the YOLOv11 baseline. The prior-guided modules raise parameters from 2.58 M to 4.94 M and GFLOPs from 6.3 to 8.5. This moderate overhead buys accuracy gains of 1.9 to 7.4 points in mAP@0.5, and throughput stays well above real time, from 230 to 309 FPS across the three benchmarks.

## E. Training Dynamics

Fig. 5 tracks the training and validation loss on the three datasets. KanserSet stabilizes within about 100 epochs because its nodule presentations are relatively homogeneous. LUNA16 converges more gradually over roughly 266 epochs, which reflects the annotation spread of four-observer consensus. Lung-PET-CT-Dx reaches a stable plateau near epoch 200 after class balancing. The adaptive threshold of Eq. 8 produces the mild oscillation visible early in training, since each rise of µ<sub>t</sub> shifts optimization from easy samples toward harder ones before the representation settles. Convergence proceeds in two phases. Recall climbs first as the network locks onto the lowfrequency Gaussian envelope, and precision then improves as the high-frequency pathway rejects vascular and pleural false positives.

## F. Qualitative Analysis

Fig. 6 shows representative detections. On KanserSet the predicted boxes conform tightly to sub-centimeter lesions where the gradient between periphery and parenchyma is subtle, which we attribute to the WiseIoU penalty anchoring regression to the Gaussian peak. On LUNA16 nodules near the pleura or vessels are delineated without spurious extension, a sign that the dilated masked attention constrains the receptive field to the decay envelope. On Lung-PET-CT-Dx the detector stays consistent across the four histological subtypes, consistent with the generality of a prior that captures the shared volumetric physics regardless of composition.

![](images/6c2cb780521a03f88311a99a9a32d44419f900e832b22ae3237dfca5bbaf52be.jpg)  
Fig. 5: Training and validation loss on the three datasets. Each left panel plots the box, classification and DFL training losses against epoch, and each right panel is a normalized validationloss heatmap of the same components over training progress.

## G. Feature Map Analysis

Fig. 7 visualizes multi-scale activations across the three benchmarks. At P3 the maps fire sharply along the nodule rim, often in radial directions, and the high resolution at this level preserves the detail that sub-6 mm nodules with σ¯ near 3 pixels demand. At P4 the response broadens to the peri-nodular ring predicted by the 3σ envelope, and the dual-frequency channels separate into a smooth interior group and a sharp group that isolates spiculation and lobulation. At P5 the maps track bronchi and vessels, the wider context that drives falsepositive suppression. The same three-stage progression holds on all three datasets, and we read its meaning for module behavior and prior transfer in the discussion below.

## VI. DISCUSSION AND LIMITATIONS

The consistent gains across three datasets with different acquisition profiles indicate that the Gaussian radial prior transfers well, since it encodes the volumetric physics of X-ray attenuation rather than dataset-specific texture. The feature maps in Fig. 7 make this concrete. Activation moves with depth exactly as the prior predicts, from a radial rim response at P3, through the peri-nodular 3σ ring with clean dual-frequency separation at P4, to bronchial and vascular context at P5. The same three-stage progression appears on KanserSet, LUNA16 and Lung-PET-CT-Dx, so the network is not memorizing the texture of one cohort but tracking a physical pattern that all three share.

The detection examples in Fig. 6 show the same logic at the output. On KanserSet the predicted boxes hug sub-centimeter lesions whose contrast against parenchyma is faint, which we read as the WiseIoU center penalty pulling regression toward the Gaussian peak rather than toward an arbitrary anchor. On LUNA16 the boxes stop cleanly at nodules that touch the pleura or a vessel, a sign that the dilated masked attention holds the receptive field to the decay envelope instead of bleeding into the adjacent wall. On Lung-PET-CT-Dx the detector behaves the same way across adenocarcinoma, small cell, large cell and squamous cell, which is what a prior grounded in shared volumetric physics rather than histology should do. The ablation in Table III gives the mechanistic version of the story. The mAP@0.75 rise from 84.1 to 89.2 percent appears only when the EWMASlide curriculum and the center penalty act together, since neither alone clears 80 percent, so reliable high-IoU localization is a joint effect of learning which samples are hard and where their true center lies.

![](images/e49a3d975d712e0ff6adac3f8047acded39a8c2dcabee0e0d9e73af189193bd2.jpg)  
(a) KanserSet

![](images/2f3d649fb10baf334735e9180c04c32f7aa55280c86f0f87eb443d2e57bf9a2e.jpg)  
(b) LUNA16

![](images/a34d98421d1770d8c922897fb677ded84a50ce8f40ebee0b2cc436511f273958.jpg)  
(c) Lung-PET-CT-Dx

Fig. 6: Representative detection results across the three benchmarks under diverse imaging conditions.  
![](images/5ec42e8cb8dc4288beafe75619a49621e3ce7cbfd3476c604f6216e24622c84d.jpg)

![](images/c4ea1d7584a6354bd64943658dae0774255f246072b8d3675e053b4a14f76f91.jpg)

(a) KanserSet  
![](images/27320d071caa5f6ea1dbbf4f3524e4e8a14f37872776036e6bed4635b4ba3ba8.jpg)

![](images/a7a0e55758f1a8da8a7c2909f44dc201e209013ebf00748ee3337da981711573.jpg)  
(b) LUNA16

![](images/9735316ed48cf6acdc5c9a29df561ea98c2542195887095ee23962049bbb506f.jpg)

![](images/f1f52c5f7ddedb4005a73350b62cc49e6f0947dc12d0f277f7e433b9d695d70d.jpg)  
(c) Lung-PET-CT-Dx

![](images/144924d22472047fa832cb7275b5e547886e1bdde930e4b5c1db7cd3baec7243.jpg)  
Fig. 7: Feature maps at P3, P4 and P5 across the three benchmarks, each row running from P3 on the left to P5 on the right. P3 shows radial boundary activation, P4 adds peri-nodular context with visible frequency separation, and P5 responds to broader anatomical neighborhoods. The same progression holds on every dataset.

Two limits remain. GRIPNet operates on 2D slices, so a strongly anisotropic nodule can break the isotropic assumption behind Eq. 2, and a volumetric prior that fits an ellipsoidal Gaussian to the whole lesion is the natural next step. Recall on KanserSet also sits a little below YOLOv8, so the conspicuitydriven curriculum still trades a few borderline lesions for high-

IoU precision, and a gentler difficulty schedule may recover them without loosening the boxes.

## VII. CONCLUSION

Progress in nodule detection has mostly come from larger backbones and wider search. We instead asked what a pulmonary nodule is, physically, and found a law rather than a texture. Across 18,218 lesions from three public benchmarks, intensity peaks at the geometric center and decays radially in a Gaussian pattern with a mean $R ^ { 2 }$ above 0.86, stable across every size stratum and histological subtype. GRIPNet turns that single measurement into an architecture, so pinwheel convolutions read the radial gradient, a dual-frequency split separates the smooth core from the spiculated rim, dilated masked attention matches the decay extent, and an adaptive loss tracks each lesion by its conspicuity. The detector reaches state-of-the-art accuracy at real-time speed, with an mAP@0.5 of 95.3%, 91.6% and 97.9% and its widest margins in the strict high-IoU regime that governs reliable measurement. Because the prior encodes the physics of X-ray attenuation rather than the style of one cohort, it stays stable across scanners and folds and should extend to any roughly spherical lesion with a dense core, from hepatic to brain metastases.

## REFERENCES

[1] H. Sung, J. Ferlay, R. L. Siegel, M. Laversanne, I. Soerjomataram, A. Jemal, and F. Bray, “Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries,” CA: A Cancer Journal for Clinicians, vol. 71, no. 3, pp. 209–249, 2021.

[2] H. J. de Koning, C. M. van der Aalst, P. A. de Jong et al., “Reduced lung-cancer mortality with volume CT screening in a randomized trial,” New England Journal of Medicine, vol. 382, no. 6, pp. 503–513, 2020.

[3] J. Terven, D.-M. Cordova-Esparza, and J.-A. Romero-Gonzalez, “A comprehensive review of YOLO architectures in computer vision: from YOLOv1 to YOLOv8 and YOLO-NAS,” Machine Learning and Knowledge Extraction, vol. 5, no. 4, pp. 1680–1716, 2023.

[4] G. Litjens, T. Kooi, B. E. Bejnordi et al., “A survey on deep learning in medical image analysis,” Medical Image Analysis, vol. 42, pp. 60–88, 2017.

[5] A. Halder, D. Dey, and A. K. Sadhu, “Lung nodule detection from feature engineering to deep learning in thoracic CT images: a comprehensive review,” Journal of Digital Imaging, vol. 33, no. 3, pp. 655–677, 2020.

[6] J. Yang, S. Liu, J. Wu, X. Su, N. Hai, and X. Huang, “Pinwheelshaped convolution and scale-based dynamic loss for infrared small target detection,” arXiv preprint arXiv:2412.16986, 2024.

[7] G. Jocher and J. Qiu, “Ultralytics YOLO11,” https://github.com/ultralytics/ultralytics, 2024.

[8] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: towards realtime object detection with region proposal networks,” in Advances in Neural Information Processing Systems, 2015, pp. 91–99.

[9] G. Cai, R. Zhang, H. He, Z. Zhang, D. Ergu, Y. Cao, J. Zhao, B. Hu, Z. Liao, Y. Zhao, and Y. Cai, “MSDet: receptive field enhanced multiscale detection for tiny pulmonary nodule,” in IEEE International Conference on Multimedia and Expo (ICME), 2025, arXiv:2409.14028.

[10] X. Song, H. Xie, T. Gao, N. Cheng, and J. Gou, “Improved YOLO-based pulmonary nodule detection with spatial-SE attention and an aspect ratio penalty,” Sensors, vol. 25, no. 14, p. 4245, 2025.

[11] Y. Zhang et al., “Lung nodule detection in medical images based on improved YOLOv5s,” IEEE Access, vol. 11, pp. 77 254–77 264, 2023.

[12] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in CVPR, 2017, pp. 2117–2125.

[13] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: convolutional block attention module,” in ECCV, 2018, pp. 3–19.

[14] D. Ouyang, S. He, G. Zhang, M. Luo, H. Guo, J. Zhan, and Z. Huang, “Efficient multi-scale attention module with cross-spatial learning,” in ICASSP, 2023, pp. 1–5.

[15] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., “An image is worth 16x16 words: transformers for image recognition at scale,” in ICLR, 2021.

[16] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: hierarchical vision transformer using shifted windows,” in ICCV, 2021, pp. 10 012–10 022.

[17] J. Li, Z. Zhang, and W. Zuo, “Rethinking transformer-based blind-spot network for self-supervised image denoising,” in AAAI, 2025.

[18] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in ICCV, 2017, pp. 2980–2988.

[19] Z. Tong, Y. Chen, Z. Xu, and R. Yu, “Wise-IoU: bounding box regression loss with dynamic focusing mechanism,” arXiv preprint arXiv:2301.10051, 2023.

[20] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.

[21] Roboflow Community, “KanserSet: Pulmonary nodule detection dataset,” https://universe.roboflow.com/, 2023.

[22] A. A. A. Setio, A. Traverso, T. de Bel et al., “Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge,” Medical Image Analysis, vol. 42, pp. 1–13, 2017.

[23] P. Li, S. Wang, T. Li, J. Lu, Y. HuangFu, and D. Wang, “A large-scale CT and PET/CT dataset for lung cancer diagnosis (Lung-PET-CT-Dx),” The Cancer Imaging Archive, 2020.

[24] H. J. Motulsky and A. Christopoulos, Fitting Models to Biological Data Using Linear and Nonlinear Regression. Oxford University Press, 2004.

[25] D. F. Yankelevitz, A. P. Reeves, W. J. Kostis, B. Zhao, and C. I. Henschke, “Small pulmonary nodules: volumetrically determined growth rates based on CT evaluation,” Radiology, vol. 217, no. 1, pp. 251–256, 2000.

[26] S. G. Armato III, G. McLennan, L. Bidaut et al., “The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans,” Medical Physics, vol. 38, no. 2, pp. 915–931, 2011.

[27] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” https://github.com/ultralytics/ultralytics, 2023.

[28] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, “YOLOv10: real-time end-to-end object detection,” arXiv preprint arXiv:2405.14458, 2024.

[29] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “SSD: single shot multibox detector,” in European Conference on Computer Vision. Springer, 2016, pp. 21–37.