# Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR

Bilal Sidiqi<sup>1,2</sup>, Yipei Wang<sup>1,2[0000-0002-9589-7177]</sup>, Wen

<sub>Yan</sub>1,2[0000-0002-3962-5994]<sub>,</sub> <sub>Shaheer</sub> <sub>U.</sub> <sub>Saeed</sub>1,3,4[0009-0002-9889-9853]<sub>,</sub>

Daniel C. Alexander<sup>1,5[0000-0003-2439-350X]</sup>, and Yipeng

<sub>Hu</sub>1,2[0000-0003-4902-0486]

<sup>1</sup> UCL Hawkes Institute, University College London, London, UK bilal.sidiqi.21@ucl.ac.uk

2 Department of Medical Physics and Biomedical Engineering, University College London, London, UK

Centre for Bioengineering, School of Engineering and Materials Science, Queen Mary University of London, London, United Kingdom

4 Digital Environment Research Institute, Queen Mary University of London, London, United Kingdom

<sup>5</sup> Department of Computer Science, University College London, London, UK

Abstract. Boundary delineation of anatomical structures in prostate MRI is often compromised by regional ambiguity. Automated segmentation methods relying on global overlap loss functions tend to be insensitive to surface deviations, leading to significant errors in ambiguous regions. Most existing uncertainty-aware methods treat uncertainty as a post-hoc quality indicator or prediction filter. We propose Contrastive Uncertainty Learning (CUL), a supervised framework that uses predictive uncertainty to directly guide representation learning. Using Monte Carlo Dropout, CUL identifies voxel-wise uncertainty and structures the embedding geometry accordingly: low-uncertainty, correctly predicted features form stable voxels while high-uncertainty or misclassified features are actively repelled, preventing feature collapse at dificult boundaries. We evaluate CUL on a 589-scan 3D prostate MRI multistructure segmentation task (8 anatomical structures) using a ViT with UperNet decoder architecture. Average Dice (83.38%) remains comparable to the baseline (83.35%), while surface distance decreases significantly $( p \ < \ 0 . 0 0 1 )$ across 6 of 8 anatomical structures. A hard-case analysis reveals a 22× asymmetry between significantly improved and degraded cases (18.8% vs 0.9%). Ablation studies show that removing uncertainty-guided sample selection eliminates improvements for the transition zone, while replacing the contrastive objective with a boundary loss additionally removes bladder improvements, demonstrating that both uncertainty-guided sampling and the contrastive objective are essential for the observed gains. These findings demonstrate that the proposed uncertainty-guided representation learning is a useful mechanism for reducing the risk of extreme errors in safety-critical medical segmentation

## 1 Introduction

Deep learning based segmentation is becoming important for quantitative analysis of prostate MRI, enabling automated delineation of prostate zones and associated structures for prostate cancer diagnosis, risk stratification, and treatment planning [1]. Despite strong average performance, convolutional and transformer models often fail on boundary ambiguous cases, producing large boundary deviations that are not fully reflected by global overlap metrics [2]. A key challenge lies in learning representations that remain robust in regions of high predictive uncertainty, typically near boundaries where standard losses such as Dice or cross-entropy do not explicitly incorporate confidence information.

Predictive uncertainty estimates represent model confidence and are widely used in medical image analysis. Bayesian approximations, including Monte Carlo dropout [3] and deep ensembles [4], correlate uncertainty with segmentation error and support quality assurance, error detection, and selective prediction in safetycritical applications.

Uncertainty has also been incorporated into training strategies, particularly in semi-supervised segmentation. In teacher–student frameworks, uncertainty is used to filter unreliable pseudo-labels or weight consistency losses to mitigate confirmation bias [5]. Uncertainty guided contrastive learning similarly excludes high uncertainty pixels from positive pair construction to stabilise representations [6, 7]. Prototype based and memory based approaches construct class prototypes from confident regions while preventing ambiguous voxels from contaminating cluster centres [8]. These methods improve robustness when labelled data are limited, using uncertainty as a filtering or weighting mechanism.

Directly using uncertainty to structure embedding geometry in fully supervised segmentation remains underexplored, particularly for intrinsically ambiguous multi-structure MRI segmentation.

Contrastive learning provides a principled mechanism for structuring representation space. Self-supervised frameworks such as MoCo [9], SimCLR [10], and supervised contrastive learning [11] demonstrate their ability to shape embeddings. Contrastive learning at the pixel and voxel levels has been adopted in segmentation to enhance intra-class compactness and inter-class separation. However, these methods typically define positive pairs solely by class identity or spatial proximity, without considering predictive confidence.

We hypothesise that structuring embedding space based on predictive uncertainty can reduce catastrophic boundary deviations while preserving overall segmentation accuracy. Confident same-class voxels should anchor compact clusters, whereas high-uncertainty voxels should be prevented from distorting these regions of feature space. We therefore propose Contrastive Uncertainty Learning (CUL), a supervised contrastive uncertainty learning framework that: (1) treats predictive uncertainty as a representation level supervisory signal rather than solely a post-hoc reliability estimate or filtering mechanism; (2) structures representation geometry by aligning low uncertainty same class voxel features while preventing ambiguous high uncertainty features from collapsing into confident clusters; (3) demonstrates significant surface distance (SD) improvement across

6 of 8 anatomical structures $( p < 0 . 0 0 1 )$ , with a 22× asymmetry between significantly improved and degraded hard cases; and (4) validates each component’s contribution via two ablations: removing uncertainty guidance achieves 5/8 significant SD anatomical structures; replacing the contrastive term with a boundary loss achieves $4 / 8 ,$ confirming uncertainty guided selection as the decisive factor, improving transition zone, while the contrastive mechanism additionally accounts for bladder. To our knowledge, uncertainty has not been explicitly used as a contrastive structuring signal to improve segmentation robustness.

![](images/84958b9baccdb5a65e31a7f7da3545867d99da48eefa20d3b6dee5248e2e7155.jpg)  
Fig. 1: Overview of the proposed Contrastive Uncertainty Learning (CUL) framework. Prostate MRI is processed by a segmentation network to produce voxel wise predictions and an associated uncertainty map. Voxel embeddings are partitioned into low and high uncertainty regions and used to construct contrastive pairs. Low-uncertainty same-class features are pulled together, while high-uncertainty or misclassified features are repelled, improving boundary robustness.

## 2 Method

## 2.1 Model Architecture

For our proposed CUL, we use a 3D model based on a Vision Transformer (ViT) encoder pretrained [12] using masked autoencoding, followed by a UperNet head [13] adapted for the segmentation task. For specific formulation of model architecture, let $\boldsymbol { x } \in \mathbb { R } ^ { D \times H \times \smile }$ denote the T2W MRI input and $y \in \mathbb { R } ^ { C \times D \times H \times W }$ the voxel-wise ground-truth masks. The encoder-adapter stack produces multiscale (s) spatial features $\{ F ^ { ( s ) } \} _ { s = 1 } ^ { 4 } = f _ { \theta } ( x )$ which are decoded using the UperNet decoder into voxel-wise logits $\boldsymbol { Z } ^ { \bullet } = g _ { \phi } \big ( \{ F ^ { ( s ) } \} \big ) \in \mathbb { R } ^ { C \times D \times H \times W }$ . Segmentation probabilities are then defined as $P = \operatorname { s o f t m a x } ( Z )$ . For the baseline segmentation we have a Dice + Cross-Entropy (DiceCE) loss: $\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { D i c e C E } } ( Z , y )$

Predictive Uncertainty Estimation For computing our proposed CUL loss, we first compute predictive uncertainty map using Monte Carlo (MC) Dropout in the decoder. Specifically, given $M ( = 5 )$ stochastic forward passes, let $P ^ { ( m ) }$ be the probability map from pass m. The predictive entropy for voxel-wise uncertainty map is then computed as $\begin{array} { r } { U = - \sum _ { c = 1 } ^ { \overline { { C } } } \bar { P } _ { c } \log ( \bar { P } _ { c } + \overset { \cdot } { \epsilon } ) } \end{array}$ where $\begin{array} { r } { \bar { P } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } } \end{array}$ P<sup>(m)</sup> is predictive probability, $C = 8$ is the number of segmentation classes (zones), $P _ { c }$ is the probability for class c and ϵ is numerical constant for stability. The map is normalised per sample: $\begin{array} { r } { \tilde { U } = \frac { U - \operatorname* { m i n } ( U ) } { \operatorname* { m a x } ( U ) - \operatorname* { m i n } ( U ) + \epsilon } . } \end{array}$

## 2.2 Contrastive Uncertainty Learning (CUL)

Uncertainty Guided Sampling and Pair Construction: Using the uncertainty map, CUL operates on decoder features (before the final classifier), which are projected into a contrastive embedding space. We first define the lowuncertainty mask and prediction-error map:

$$
M _ { \mathrm { l o w } } = { \bf 1 } \Big ( \tilde { U } < \tau \Big ) , E _ { \mathrm { m a p } } = { \bf 1 } ( \hat { y } \neq y ) ,\tag{1}
$$

where $\tau ( = 0 . 5 )$ is a fixed uncertainty threshold and $\hat { y }$ is the predicted label. From these, we define an anchor pool and a negative pool:

$$
M _ { \mathrm { a n c h o r } } = M _ { \mathrm { l o w } } \cdot ( 1 - E _ { \mathrm { m a p } } ) , ~ M _ { \mathrm { n e g } } = \pm ( ( 1 - M _ { \mathrm { l o w } } ) + E _ { \mathrm { m a p } } > 0 ) .\tag{2}
$$

Terminology: Anchors are voxels $( e _ { p } )$ sampled from decoder features using the mask $M _ { \mathrm { a n c h o r } }$ , corresponding to low-uncertainty and correctly predicted regions. For each anchor, a positive sample is defined as a same-class voxel drawn from the anchor pool, excluding the anchor itself when possible. Negatives $\left( e _ { n } \right)$ are sampled using $M _ { \mathrm { n e g } }$ , comprising voxels that are either high-uncertainty or misclassified, and are not restricted to diferent-class locations.

Uncertainty-Aware Contrastive Loss: Projected features (anchors and negatives) are $\ell _ { 2 } .$ -normalised, and for each anchor $e _ { p } ^ { i }$ , one positive $e _ { p ( i ) } ^ { i }$ is sampled from the same-class confident set, and a set of negatives ${ \mathcal { N } } _ { i }$ is sampled using $M _ { \mathrm { n e g } } .$ . The contrastive term is then computed voxel-wise as:

$$
\mathcal { L } _ { \mathrm { C U L - c t r } } ^ { ( i ) } = - \log \frac { \exp ( \sin ( e _ { p } ^ { i } , e _ { p ( i ) } ^ { i } ) / T ) } { \exp ( \sin ( e _ { p } ^ { i } , e _ { p ( i ) } ^ { i } ) / T ) + \sum _ { k \in N _ { i } } \exp ( \sin ( e _ { p } ^ { i } , e _ { n } ^ { k } ) / T ) } ,\tag{3}
$$

where $T$ is a learnable temperature parameter. The full training objective is defined as sum of DiceCE and CUL (weighted by learnable parameter $\lambda _ { c t r } )$ $\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { c t r } } \mathcal { L } _ { \mathrm { C U L - c t r } }$

## 3 Experiments and Results

Dataset and Experimental Setup: For our experiments, we employ a multiinstitute pelvis MR dataset [14], consisting of 589 T2W 3D MRI input samples of size $\left( \mathrm { D { = } 6 4 , } \mathrm { H { = } 2 5 6 , } \mathrm { W { = } 2 5 6 } \right)$ with anatomy/zone masks. In total, there are 8 anatomical structures; bladder, bone, obturator internus, central gland, transition zone, rectum, seminal vescicle and neurovascular bundle. The dataset was split into train, validation and test sets (412, 60, 117) and the inputs were preprocessed and augmented using standard preprocessing and augmentation (centre cropping, rotation, flipping, zooming and intensity normalisation). Training used an NVIDIA A40 GPU for 100 epochs with learning rate $1 e - 4 .$ . The Dice with cross entropy loss was adopted to train our baseline segmentation model using the Adam optimiser. Batch size was set to 4 with 5 Monte Carlo (MC) samples computed for the uncertainty map during training. For performance evaluation, we use Dice, Hausdorf-distance and Surface Distance, capturing the overall overlap as well as worst-case boundary errors and surface errors. To isolate each component’s contribution, we evaluate two ablations under identical training conditions: a class-based variant replacing uncertainty-guided anchor selection with ground-truth class identity only; and a boundary loss variant replacing the contrastive term with a Kervadec-style distance-transform boundary loss.

Table 1: Overall segmentation performance for Baseline, CUL, and the two ablations. SD(mm) is significantly improved which is consistent with CUL’s design: predictive uncertainty concentrates at local boundary regions, so improvement manifests in surface-sensitive metrics rather than in global overlap (Dice) or gross deviation extremes (HD95(mm)).
<table><tr><td>Metric</td><td>Baseline</td><td>CUL</td><td>Class-Based</td><td>Classic BL</td></tr><tr><td>Dice (↑)</td><td> $0 . 8 3 3 5 \pm 0 . 0 6 4 7$ </td><td> $0 . 8 3 3 8 \pm 0 . 0 6 5 1$ </td><td> $0 . 8 3 3 1 \pm 0 . 0 6 3 7$ </td><td> $\mathbf { 0 . 8 3 3 8 \pm 0 . 0 6 3 4 }$ </td></tr><tr><td>HD95(mm) (↓)</td><td> $5 . 9 3 4 3 \pm 8 . 1 4 1 4$ </td><td> $\mathbf { 5 . 4 1 3 0 \pm 4 . 8 8 2 9 }$ </td><td> $5 . 3 7 3 6 \pm 4 . 6 5 5 8$ </td><td> $5 . 6 2 5 3 \pm 5 . 2 3 0 6$ </td></tr><tr><td>SD(mm) (↓)</td><td> $1 . 6 0 7 6 \pm 2 . 5 1 0 1$ </td><td> $\mathbf { 1 . 4 1 9 0 \pm 1 . 8 1 3 5 }$ </td><td> $1 . 4 3 8 5 \pm 1 . 8 8 7 3$ </td><td> $1 . 4 6 4 1 \pm 1 . 9 1 8 8$ </td></tr></table>

Overall Quantitative Performance: Table 1 summarises overall segmentation performance across 117 test cases. CUL achieves comparable Dice performance to the baseline (0.8338 vs 0.8335), indicating that uncertainty-aware contrastive learning does not on average degrade global overlap. As CUL specifically focuses on boundary deviations, we observe consistent improvement in boundary sensitive metrics. Mean HD95 decreases from 5.93 mm to 5.41 mm, and mean surface distance (SD) decreases from 1.61 mm to 1.42 mm.

The HD95 standard deviation is notably reduced $( 8 . 1 4  4 . 8 8 ~ \mathrm { m m } )$ , suggesting suppression of extreme outliers, though formal variance testing (Brown-Forsythe $\scriptstyle { p = 0 . 5 3 8 ) }$ does not confirm this at $\alpha { = } 0 . 0 5$ , likely reflecting limited statistical power at $n = 1 1 7$ ; we therefore treat it as a descriptive finding. Paired Wilcoxon signed-rank testing confirms that SD reduction is statistically significant $\left( { p { < } 0 . 0 0 0 1 } \right)$ , while Dice $( p { < } 0 . 1 8 1 )$ and HD95 $( p { < } 0 . 8 1 6 )$ median shifts are not. Dice is expected to remain unchanged because CUL targets local boundary uncertainty rather than volumetric overlap. Although HD95 is not significant, its mean decreases by 0.49 mm and variance is descriptively reduced, suggesting suppression of extreme outliers despite limited statistical power.

Table 2: Case-wise performance distribution (n=117). Significant improvement/degradation thresholds: SD ±0.1 mm, HD95 ±0.5 mm.
<table><tr><td>Metric</td><td>Improved / Degraded (%)</td><td> $\mathbf { S i g + \lambda / \lambda S i g - \lambda ( \% ) }$ </td></tr><tr><td></td><td> $6 4 \textrm { / 5 3 } ( 5 4 . 7 \textrm { / 4 5 . 3 } )$ </td><td>Ratio</td></tr><tr><td>Dice (↑)</td><td></td><td> $1 2 \textrm { / 8 } ( 1 0 . 3 \textrm { / 6 . 8 } )$ </td></tr><tr><td>HD95(mm)(↓) SD(mm) (↓)</td><td> $6 6 ~ / ~ 5 1 ~ ( 5 6 . 4 ~ / ~ 4 3 . 6 )$   $7 9 \ / \ 3 8 \ ( 6 7 . 5 \ / \ 3 2 . 5 )$ </td><td>1.5×  $2 2 \textrm { / 1 } ( 1 8 . 8 \textrm { / 0 . 9 } )$  22×</td></tr></table>

Table 3: HD95(mm) change stratified by baseline dificulty (median=4.0032).
<table><tr><td>Subset</td><td>n</td><td>∆HD95 (mm)</td></tr><tr><td>Easy (&lt;4.0032)</td><td>58</td><td>≈0</td></tr><tr><td>Hard (≥4.0032)</td><td>59</td><td>-1.1131</td></tr></table>

Case-Wise Performance Distribution: To characterise model behaviour beyond aggregate statistics, we analyse per-case performance (Table 2) across 117 test cases. A case is classified as notably improved or degraded if its mean metric change across all 8 structures exceeds a fixed medically critical absolute threshold (SD: ±0.1 mm; HD95: ±0.5 mm). Dice improves in 54.7% of cases versus 45.3% degraded, indicating only a modest net overlap gain. In contrast, boundarysensitive metrics show clear asymmetry. HD95 improves in 56.4% of cases (+15 net), while SD improves in 67.5% of cases (+41 net). When considering significant changes, SD exhibits strong directional bias: 18.8% of cases show significant reduction compared to only 0.9% significant increase (22 vs 1 cases). This 22× ratio suggests asymmetric boundary refinement rather than random fluctuation. Figure 2 demonstrates the case where CUL has improved the boundary uncertainty the most (HD95=17.11 vs 4.09) and also the most degraded case (HD95=10.97 vs 12.80).

Hard-Case Robustness Analysis: We stratify cases using the median baseline HD95 threshold (4.00) to distinguish easy and hard subsets (Table 3). CUL achieves the largest boundary improvement in hard cases, with a mean HD95 reduction of 1.113 mm, whereas easy cases exhibit only marginal change (≈ 0). Structure-wise Analysis: Table 4 reports mean surface distance per anatomical structure (paired Wilcoxon,α=0.05). CUL achieves significant SD improvement in 6/8 structures:bladder, bone, central gland, rectum, transition zone, and neurovascular bundle. Obturator internus and seminal vesicle show no significant change;both have inter-subject variance exceeding three times their mean, suggesting high case to case variability. Dice remains largely unchanged across all structures $\cdot | \varDelta | { < } 0 . 0 0 4 )$ . HD95 reaches significance only for rectum (p=0.004). Notably, the largest quantitative SD gains appear in bone(−0.16 mm) and central gland(−0.61 mm).

Ablation Study: To isolate the contribution of each CUL component, Table 4 also reports the two ablations. The class-based ablation (uncertainty guided selection removed) reaches $5 / 8$ significant structures, failing only on transition zone $\scriptstyle \left( p = 0 . 6 7 6 \right)$ which CUL improves $\scriptstyle \left( p = 0 . 0 0 2 \right)$ . This shows uncertainty guidance provides additional benefit beyond class-based contrastive learning. The boundary loss ablation reaches only $4 / 8 ,$ , additionally failing to improve bladder $\scriptstyle ( p = 0 . 1 1 5 )$ ), suggesting geometric loss augmentation cannot replicate uncertaintyguided contrastive structuring. The stepwise pattern, $\mathrm { C U L } 6 / 8 >$ class-based 5/8 > boundary loss $4 / 8 ,$ , isolates uncertainty guided sample selection as the decisive factor for transition zone, and the contrastive mechanism as the decisive factor for bladder.

Table 4: Per-structure SD (mm ↓), n=117, paired Wilcoxon vs. Baseline. $^ { * } p { < } 0 . 0 5 , ^ { * * } p { < } 0 . 0 1 , ^ { * * * } p { < } 0 . 0 0 1$
<table><tr><td>Structure</td><td>Baseline</td><td>CUL</td><td>Class-Based</td><td>Classic BL</td></tr><tr><td>Bladder</td><td> $1 . 1 7 \pm 1 . 8 2$ </td><td> $0 . 8 5 ^ { * * * }$ </td><td>0.86*</td><td>0.87</td></tr><tr><td>Bone</td><td> $1 . 2 0 \pm 1 . 9 4$ </td><td>1.03***</td><td> $1 . 0 2 ^ { * * }$ </td><td> $1 . 0 1 ^ { * * * }$ </td></tr><tr><td>Obturator Int.</td><td> $1 . 4 3 \pm 3 . 4 7$ </td><td>1.37</td><td>1.30</td><td>1.37</td></tr><tr><td>Central Gland</td><td> $2 . 5 0 \pm 1 3 . 4$ </td><td>1.89***</td><td>2.10***</td><td>2.02**</td></tr><tr><td>Transition Zone</td><td> $1 . 5 9 \pm 1 . 1 3$ </td><td>1.50**</td><td>1.55</td><td>1.58</td></tr><tr><td>Rectum</td><td> $1 . 2 2 \pm 0 . 9 8$ </td><td>1.15***</td><td>1.14***</td><td>1.13***</td></tr><tr><td>Seminal Vesicle</td><td> $1 . 7 8 \pm 5 . 4 3$ </td><td>1.65</td><td>1.62</td><td>1.81</td></tr><tr><td>NVB</td><td> $1 . 9 7 \pm 2 . 2 9$ </td><td>1.93***</td><td>1.92***</td><td>1.92***</td></tr><tr><td>Sig. structures</td><td></td><td>6/8</td><td>5/8</td><td>4/8</td></tr></table>

## 4 Discussion and Conclusion

For our multi-structure MRI segmentation, we propose a supervised contrastive framework for improving geometric robustness, by estimating predictive uncertainty as a representation-level supervisory signal.

Our experiments on a multi-organ prostate MRI segmentation task show that CUL maintains comparable global overlap performance to the baseline, while consistently improving boundary-sensitive metrics. Case-wise and dificulty-stratified analyses further reveal that CUL yields the strongest benefits on hard cases characterised by large boundary deviations. Moreover, comparison with the two ablations clarifies the source of these gains, isolating the uncertainty guided mechanism as the key driver of CUL’s improvements in the transition zone.

These improvements have practical implications for radiotherapy planning. In pelvic organ segmentation, surface distance errors of even a few millimetres can translate to incorrect planning target volume (PTV) margins, afecting dose coverage and organ at risk (OAR) sparing. Similar precision requirements are of importance for targeted biopsy and focal therapy. The significant SD reductions, directly targets the boundary precision that afects contour acceptance in clinical workflows. The 22× asymmetry between significant improvements and significant degradations in SD further suggests CUL reduces the tail of outlier errors that typically require manual re-contouring. Future work includes adaptive uncertainty thresholding, additional ablations, and integration with multi-scale, domain-robust training.

![](images/d288964997a78b14945537bfa7aed20c5b3ae2f2da2c2e72465cdd9ce4f5b420.jpg)  
(b) Degraded case  
Fig. 2: Qualitative comparison between an improved case (top) and a degraded case (bottom). The top row highlights CUL’s role as geometric “insurance”, where it efectively corrects extreme boundary outliers in cases characterised by high regional ambiguity (e.g. central gland and both neurovascular bundles). CUL organises representations to stabilise boundaries by anchoring confident regions and isolating ambiguity. This visualisation demonstrates the framework’s capacity to correct significant spatial deviations while maintaining the integrity of stable predictions.

## References

1. B. Turkbey and M. A. Haider, “Deep learning-based artificial intelligence applications in prostate mri: brief summary,” The British Journal of Radiology, vol. 95, no. 1131, p. 20210563, 2022.

2. M.-K. Fassia et al., “Deep learning prostate mri segmentation accuracy and robustness: a systematic review,” Radiology: Artificial Intelligence, vol. 6, no. 4, p. e230138, 2024.

3. Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” International Conference on Machine Learning (ICML), 2016.

4. B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” NeurIPS, 2017.

5. Y. Ouali, C. Hudelot, and M. Tami, “Semi-supervised semantic segmentation with cross-consistency training,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12 674–12 684.

6. T. Wang, J. Lu, Z. Lai, J. Wen, and H. Kong, “Uncertainty-guided pixel contrastive learning for semi-supervised medical image segmentation,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJ-CAI), 2022, pp. 1444–1450.

7. Y. Hua, X. Shu, Z. Wang, and L. Zhang, “Uncertainty-guided voxel-level supervised contrastive learning for semi-supervised medical image segmentation,” International Journal of Neural Systems, vol. 32, no. 4, p. 2250016, 2022.

8. X. Chen, J. Yao, R. Zhang et al., “Class prototypes for semi-supervised medical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2022.

9. K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 9729–9738.

10. T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International Conference on Machine Learning (ICML), 2020, pp. 1597–1607.

11. P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 18 661–18 673.

12. Y. Wang, Y. Xu, W. Yi, S. U. Saeed, N. Thorley, A. Ng, Y. Zhou, W. Yan, D. Barratt, S. Punwani, V. Kasivisvanathan, M. Emberton, D. C. Alexander, and Y. Hu, “Profound: A moderate-sized vision foundation model for multi-task prostate imaging,” 2026. [Online]. Available: https://arxiv.org/abs/2603.03961

13. T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified perceptual parsing for scene understanding,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 418–434.

14. Y. Li, Y. Fu, I. J. Gayo, Q. Yang, Z. Min, S. U. Saeed, W. Yan, Y. Wang, J. A. Noble, M. Emberton et al., “Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration,” Medical Image Analysis, vol. 90, p. 102935, 2023.