# Data-Efficient Crosswalk Segmentation from Overhead CCTV via Confidence- and Geometry-Guided Pseudo-Labeling

Preprint, September 2026

Abdirashid Omar and Jonghyuk Park Department of Data Science, Graduate School of Kookmin University Seoul 02707, Republic of Korea

## Abstract

Pixel-level annotation of fixed trafic-camera imagery is expensive, while crosswalk models trained from streetlevel imagery face a substantial viewpoint and appearance shift when applied to elevated CCTV. We investigate a data-eficient target-domain pipeline using 241 manually annotated CCTV images and 5,926 unlabeled CCTV frames. A source-domain experiment trains a 31.0M-parameter custom U-Net on 3,300 first-person-view (FPV) images and obtains 93.05% IoU on its 330-image FPV test split. This result is a source baseline, not transferred performance: the released CCTV notebook instantiates a 42.0M-parameter DeepLabV3-ResNet50 from torchvision weights, and no compatible mapping from the U-Net checkpoint is implemented. Training on 201 manual CCTV images and selecting on 40 held-out manual masks yields 88.91% IoU. The model then predicts all unlabeled frames; image-level certainty and a largest-component area prior rank the candidates, and the top 1,000 attain mean certainty 0.976 and mean combined score 0.988. A repository audit shows that the reported second-stage 98.52% IoU was measured on a 150-image split containing only teacher-generated pseudo-masks: because of a directory-layout mismatch, the executed combined-data loader found zero manual samples and split 1,000 pseudo-labeled samples into 850/150. We therefore report 98.52% as internal pseudolabel agreement rather than human-ground-truth accuracy. The defensible target-domain result is 88.91% IoU on the 40 manual validation images. Batch-one FP32 inference at 512×512 requires 12.98 ms (77.03 FPS) on an NVIDIA RTX A6000 48 GB. These findings support the practicality of confidence-and-geometry filtering, while also showing why pseudo-label evaluation must remain isolated from the labels used for self-training.

Keywords crosswalk segmentation · CCTV · pseudo-labeling · semi-supervised learning · domain shift · intelligent transportation

## 1 Introduction

Crosswalk localization is a useful perception primitive for trafic monitoring and pedestrian-safety systems. Classical methods exploit repeated stripe edges and geometric regularity [1]; recent systems instead learn crosswalk appearance from data [2, 3]. Fixed overhead cameras remain dificult because perspective compression, small foreground scale, dynamic occlusion, shadows, nighttime illumination, and camera-specific backgrounds difer sharply from pedestrian- or vehicle-level imagery.

Collecting target-domain video is easy, but dense annotation is not. This asymmetry motivates a simple question: can a small manually labeled CCTV set and a larger unlabeled targetdomain pool support useful binary crosswalk segmentation under FPV-to-CCTV shift? We study that question in the public rashiedomar/crosswalk-cctv project. The empirical pipeline trains a supervised CCTV model, predicts 5,926 unlabeled frames, rejects implausible masks with confidence and foreground-area checks, and uses the highest-ranked masks for a second training stage.

This paper is deliberately conservative. Saved notebook outputs provide strong evidence for the supervised target model and for real-time throughput, but they do not provide an independent human-annotated evaluation of the second-stage checkpoint. Our audit additionally identifies a loader mismatch that excluded all manual samples from the executed second stage. Accordingly, the paper separates three diferent quantities: source-domain FPV test IoU, held-out manual CCTV validation IoU, and agreement with generated pseudo-masks. Mixing them would overstate the evidence.

Our contributions are:

• an empirical characterization of the FPV-to-overhead-CCTV viewpoint gap for binary crosswalk segmentation;

• a lightweight pseudo-label ranking rule that combines pixel certainty with a crosswalk-area prior, selecting 1,000 samples from 5,926 unlabeled frames;

• a reproducible audit of dataset construction and evaluation semantics, including a clear distinction between 88.91% manual-mask IoU and 98.52% internal pseudomask agreement; and

• a measured batch-one throughput of 77.03 FPS at 512× 512 on an RTX A6000.

## 2 Related Work

## 2.1 Crosswalk perception

Crosswalk detection has been studied from mobile, vehicle, and overhead viewpoints. Ivanchenko et al. used line grouping and figure-ground reasoning on a camera phone [1]. Berriel et al. used automatically collected street and map imagery and emphasized cross-database evaluation [2]. Liang and Seo combined SegNet-style decoding with residual features for zebra-crossing segmentation [3], while Verma and Ukkusuri detected crossings in satellite imagery for pedestrian-network completion [4].

![](images/e75d9e53c51075ac59af117d90ff5958f3afa468437a3e1ec38becf2f0103d5a.jpg)  
Figure 1: Audited experimental pipeline. The FPV model is a source-domain reference; the released code does not implement a verified U-Net-to-DeepLab parameter transfer. Green denotes human-annotated target-domain training and validation, orange denotes pseudo-label generation, and the dashed red note records the executed second-stage protocol.

Fixed urban CCTV is neither street-level nor near-orthographic: crosswalks may be distant, oblique, partly outside the frame, or repeatedly occluded by vehicles.

## 2.2 Segmentation and semi-supervision

U-Net established an encoder-decoder design with skip connections for dense prediction [5]; residual learning enabled deeper visual backbones [6]. DeepLabV3 uses atrous convolution and multi-scale context [7], and DeepLabV3+ adds an explicit decoder [8].

Pseudo-labeling treats confident predictions as targets [9]. Mean Teacher stabilizes targets through weight averaging [10], while FixMatch couples thresholded predictions with strong augmentation [11]. Dense prediction has motivated segmentation-specific methods including ClassMix [12], Cross Pseudo Supervision [13], and selective self-training in ST++ [14]. Under domain shift, AdaptSegNet aligns structured outputs [15], CyCADA combines cycle-consistent adaptation [16], and DACS mixes source and target samples while using pseudo-labels [17]. Our method is intentionally simpler: a single teacher ranks whole CCTV images by certainty and a binary geometry prior.

## 3 Method

## 3.1 Problem formulation

Let $\mathcal { D } _ { L } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { L } }$ denote labeled target-domain images and binary masks, and let $\mathcal { D } _ { U } = \{ u _ { j } \} _ { j = 1 } ^ { N _ { U } }$ denote unlabeled CCTV frames. A network $f _ { \theta }$ produces a foreground probability map

$$
p = f _ { \theta } ( x ) , \qquad p \in [ 0 , 1 ] ^ { H \times W } ,\tag{1}
$$

and the binary prediction is $\hat { y } _ { k } = \mathcal { H } [ p _ { k } \ge 0 . 5 ]$

## 3.2 Source-domain baseline

The saved FPV notebook partitions 3,300 images into 2,640 training, 330 validation, and 330 test samples. Contrary to the repository README’s “U-Net + ResNet34” description, the executable notebook defines a custom four-level U-Net with channels 64–128–256–512, a 1,024-channel bottleneck, and 31,043,521 parameters; no ResNet encoder is instantiated. It trains for 30 epochs with Adam at $1 0 ^ { - 4 }$ , batch size 8, and the same binary objective used below. The best validation IoU is 92.44% at epoch 21 (one-indexed), and the restored checkpoint obtains 93.05% on the FPV test set.

The CCTV notebook separately instantiates torchvision DeepLabV3-ResNet50 with default pretrained weights and replaces the final output layer for one class. It attempts to load the custom U-Net state dictionary with strict=False, but does not record matched keys or implement a parameter mapping. Because the module namespaces and tensors belong to diferent architectures, the available implementation does not establish meaningful checkpoint transfer. We therefore treat the FPV experiment only as a source-domain baseline.

## 3.3 Supervised CCTV model

The target model is DeepLabV3 with a ResNet50 backbone [6,7]. Images and masks are resized to 512×512. The 241 humanlabeled images are split with seed 42 into 201 training and 40 validation samples. The training loader uses batch size 8 and drops one incomplete sample, so 200 examples are processed per epoch. Optimization uses AdamW [18], learning rate $1 0 ^ { - 4 }$ weight decay $1 0 ^ { - 4 }$ , and 30 epochs.

For pixels indexed by k, the objective is

$$
\mathcal { L } _ { \mathrm { s u p } } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } } ,\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { { D i c e } } } = 1 - \frac { 2 \sum _ { k } p _ { k } y _ { k } + \epsilon } { \sum _ { k } p _ { k } + \sum _ { k } y _ { k } + \epsilon } .\tag{3}
$$

The notebook applies color jitter and random horizontal flipping to training images. Its horizontal flip is not synchronized with the mask, however, and therefore adds alignment noise. This defect should be corrected in any rerun; it does not change the provenance of the archived validation score.

## 3.4 Confidence- and geometry-guided selection

For each unlabeled image, the teacher produces $p _ { j }$ and pseudomask $\hat { y } _ { j }$ . The implementation defines image-level certainty as the mean distance from the binary decision boundary,

$$
c _ { j } = \frac { 2 } { H W } \sum _ { k = 1 } ^ { H W } | p _ { j , k } - 0 . 5 | .\tag{4}
$$

It extracts the largest contour in $\hat { y } _ { j }$ , computes its area ratio

$$
r _ { j } = \frac { A ( \mathrm { l a r g e s t } ( \hat { y } _ { j } ) ) } { H W } ,\tag{5}
$$

and assigns

$$
g _ { j } = \left\{ \begin{array} { l l } { { 1 . 0 , } } & { { 0 . 0 5 < r _ { j } < 0 . 4 0 , } } \\ { { 0 . 5 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. \qquad s _ { j } = \frac { c _ { j } + g _ { j } } { 2 } .\tag{6}
$$

AI Hub CCTV Dataset (Testing) 23011\_2021110407501943\_142.jpg...

# DOMAIN GAP: FPV Training Data vs AI Hub CCTV Testing Data

![](images/f3d1b78edf0851a819e0539d98d1c8683b68b0be3061672c2af424501005b684.jpg)

FPV Dataset (Training) synthetic\_standard\_01815\_img\_std\_46...  
![](images/e0bb5ec3c3fd3ee85cd4dd94554eab55f396b5f7fe327ea92076e7329004c5ef.jpg)

AI Hub CCTV Dataset (Testing) 84011 2021090906293060 031.jpg...  
![](images/8b11d9610c91f1376f682d3111f0a555b4c19736170f5e5e8521b7cd41fbbd8d.jpg)

AI Hub CCTV Dataset (Testing) 54011 2021110413580290 068.jpg...  
![](images/7ce313af903229cd60cfbb0d6ae075906d4b3f8c078a2d7bac5d0e75e3f05888.jpg)  
Figure 2: Examples recovered from the repository’s FPV-versus-AI-Hub comparison. FPV imagery (left) is near-horizontal and crosswalk-centered; fixed CCTV imagery (right) is elevated, wider, and more cluttered. The overlaid confidence text comes from the source-model analysis and is not ground-truth IoU on CCTV.

Candidates with $s _ { j } \geq 0 . 7$ are sorted, and at most 1,000 are retained. All selected samples satisfy the area rule. Their certainty is approximately 0.976, while the combined score has mean 0.9882, median 0.9882, and range 0.9879–0.9888. This distinction resolves inconsistent uses of “confidence” in the saved summary and README.

## 3.5 Second-stage training and protocol audit

The intended design combines human and pseudo-labeled data. The executed notebook does not realize that design. Its initial manual-data loader searches recursively and finds finetuning/images and finetuning/masks. The secondstage loader instead iterates over immediate subdirectories and searches for an additional images/masks pair below each one. It consequently prints Original: 0, Pseudo: 1000, and then randomly splits only the pseudo-labeled set into 850 training and 150 validation images. A later “1,241 samples” message is a hard-coded string, not the loaded dataset length.

The model continues from the supervised checkpoint and trains for 20 epochs with AdamW at $\bar { 5 } \times 1 0 ^ { - 5 }$ . Its best 98.52% IoU at epoch 19 measures overlap with pseudo-masks generated by its own teacher lineage. It is useful as a self-training diagnostic but cannot establish an accuracy gain. The public repository excludes the images and model checkpoints, so the second-stage checkpoint cannot be reevaluated here on the 40 manual masks.

Table 1: Dataset composition and executed use.
<table><tr><td>Subset or pool</td><td>Images</td><td>Role</td></tr><tr><td>FPV train / val / test</td><td>2,640 / 330 / 330</td><td>source</td></tr><tr><td>Manual CCTV train</td><td>201</td><td>supervised</td></tr><tr><td>Manual CCTV valida-</td><td>40</td><td>human eval.</td></tr><tr><td>tion Unlabeled CCTV pool</td><td>5,926</td><td></td></tr><tr><td>Selected pseudo-labels</td><td>1,000</td><td>teacher input stage two</td></tr><tr><td>Nominal combined pool</td><td>1,241</td><td>documented intent</td></tr><tr><td>Executed stage-two pool</td><td>1,000</td><td>0 human + 1,000 pseudo</td></tr><tr><td>Executed train / val</td><td>850 / 150</td><td>pseudo only</td></tr></table>

## 4 Experimental Setup

## 4.1 Data

The CCTV frames derive from the AI-Hub urban-road trafic CCTV resource [19]. Table 1 distinguishes recorded data from intended and executed stage-two pools. The unlabeled frames are used only for teacher prediction and pseudo-label selection.

![](images/af2f04af51d2300271c33afe99eb02f2a850114731cd0a660f5bc51194ea0c1d.jpg)  
Figure 3: Six high-ranked pseudo-labels recovered from the repository. Red overlays are automatically generated masks, not human ground truth. The displayed score averages pixel certainty (about 0.976) and the binary geometry score (1.0), yielding about 0.988. Occluding vehicles remain visible beneath the translucent masks.

## 4.2 Metrics and implementation

For binary masks, the reported intersection-over-union is

$$
\mathrm { I o U } ( \hat { y } , y ) = \frac { \sum _ { k } \hat { y } _ { k } y _ { k } + \epsilon } { \sum _ { k } \hat { y } _ { k } + \sum _ { k } y _ { k } - \sum _ { k } \hat { y } _ { k } y _ { k } + \epsilon } .\tag{7}
$$

Dice/F1 is related by Dice = 2 IoU /(1 + IoU) [20], but the archived experiments report IoU. Notebook validation averages batch-level IoUs rather than accumulating a dataset-wide confusion matrix. The manual validation set contains exactly five full batches; the pseudo-only validation set ends with a smaller batch and is therefore not strictly sample-weighted.

Experiments ran with PyTorch on an NVIDIA RTX A6000 with 48 GB VRAM. The CCTV notebook reports 41.99M trainable parameters, which we round to 42.0M; the README’s “approximately 39M” is not consistent with the saved model printout. No mixed-precision context appears in the timing cell, so Table 2 describes FP32 inference.

## 5 Results

Table 3 presents each value with its actual protocol. The 93.05% FPV test result establishes in-domain source performance. The 88.91% CCTV result is the strongest target-domain number backed by human masks: the best of 30 epochs on the fixed 40- image validation set. Because that split also selects the checkpoint, it should be called validation, not an independent test set.

Table 2: Recorded inference configuration.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Model</td><td>DeepLabV3-ResNet50</td></tr><tr><td>Parameters</td><td>41.99M</td></tr><tr><td>Input / batch</td><td>512×512 RGB / 1</td></tr><tr><td>Precision</td><td>FP32 (no autocast recorded)</td></tr><tr><td>Hardware</td><td>NVIDIA RTX A6000, 48 GB</td></tr><tr><td>Warm-up / timed runs</td><td>10 / 100</td></tr><tr><td>Latency</td><td>12.98 ms per image</td></tr><tr><td>Throughput</td><td>77.03 FPS</td></tr></table>

The pseudo-label statistics show that the ranking rule selects highly certain, area-valid masks. The 98.52% second-stage value is numerically higher but answers a diferent question. It measures how well the continued model fits teacher-generated targets in a pseudo-only split and must not be presented as a 9.61-point target-domain improvement.

Figure 4 confirms that the supervised CCTV model can follow several crosswalk components, including oblique boundaries, crowded scenes, and nighttime imagery. It also exposes the failure mode that motivated target-domain supervision: the FPVonly source model can return almost no foreground under distant, dark CCTV conditions. Figure 3 shows plausible selected masks, but visual plausibility is not a substitute for human-mask evaluation.

![](images/8c77822717484fb481da172aa4a238d88292e55065a70b4474ac5290023b7d93.jpg)  
Figure 4: Qualitative evidence with provenance kept explicit. The upper grid shows four samples from the 40-image humanannotated CCTV validation set used for the supervised model: each group contains input, human ground-truth overlay (green), and prediction overlay (red). The lower strip shows the FPV source model’s near-empty response to a dificult night CCTV frame; it has no human mask and is included only as a domain-shift failure example. The upper panel is not an iteration-two evaluation.

Table 3: Results separated by label source and evaluation protocol. Values across rows are not directly comparable when the domains or targets difer.
<table><tr><td>Stage</td><td>Evaluation target and split</td></tr><tr><td>FPV baseline</td><td>Human FPV masks; 330 test 93.05%</td></tr><tr><td>Supervised CCTV</td><td>Human CCTV masks; 40 valida- 88.91% tion</td></tr><tr><td>Pseudo selection Stage-two internal</td><td>No human targets; top 1,000 Teacher pseudo-masks; 150 vali- 98.52%</td></tr><tr><td></td><td>dation</td></tr><tr><td></td><td>Stage two, human CCTV Checkpoint/data unavailable; not run</td></tr></table>

## 6 Discussion and Limitations

The reliable conclusion is narrower than the repository README suggests. A modest manual target set is suficient to train a strong CCTV validation model, and the teacher can generate visually plausible masks at scale. The certainty-plus-area rule is inexpensive and domain-specific: it removes masks with a largest foreground component below 5% or above 40% of the frame. At the same time, high certainty can reflect calibration or class imbalance rather than correctness, and an area prior cannot detect a confidently segmented road region of plausible size.

Four limitations determine the next experiment. First, 40 validation images are too few for a definitive estimate and are used for checkpoint selection. Second, the executed second stage contains no manual masks, so it is not the intended semisupervised mixture. Third, random splitting of nearby video frames may leak scene and temporal redundancy; future splits should be camera- or sequence-disjoint. Fourth, unsynchronized image/mask flipping introduces training noise. These issues are implementation and evaluation limitations, not merely presentation details.

A corrected evaluation should (i) keep the same 40 manual frames fixed for comparison, or preferably add a camera-disjoint manual test set; (ii) train on exactly 201 manual plus 1,000 pseudo-labeled samples; (iii) apply geometric transforms jointly to image and mask; (iv) report confidence calibration and pseudo-mask quality on a manually audited subset; and (v) provide per-camera results with bootstrap confidence intervals. Ablations should compare manual-only training, all pseudo-labels, certainty-only selection, geometry-only selection, and their combination. Only such an experiment can establish whether filtered pseudo-labels improve human-ground-truth IoU.

The recorded 77.03 FPS demonstrates server-GPU feasibility, not embedded deployment. It excludes video decoding, resizing, transfer, and post-processing, and should not be extrapolated to edge devices without end-to-end measurement.

## 7 Conclusion

We examined a practical pipeline for crosswalk segmentation under FPV-to-CCTV domain shift. A custom U-Net reaches 93.05% IoU on FPV test data, while a separately initialized DeepLabV3-ResNet50 reaches 88.91% IoU on 40 humanannotated CCTV validation images after training on 201 manual examples. Confidence and a 5–40% largest-component area prior select 1,000 visually plausible pseudo-labels from 5,926 unlabeled frames. The reported 98.52% second-stage score, however, is pseudo-only internal agreement because the executed loader omitted all manual samples. The study therefore supports the eficiency and real-time practicality of pseudo-label generation, but does not yet prove a human-ground-truth accu racy gain from self-training. Corrected mixed-data training and camera-disjoint manual evaluation are the essential next steps.

## Data and Code Availability

Code, notebooks, numerical summaries, pseudo-label metadata, and the experiment figures used in this paper are publicly available at rashiedomar/crosswalk-cctv. The repository commit audited for this manuscript is 7e4a9d7f59e7de56f867a2e49ab971408e5007a0. Large datasets and trained checkpoints are excluded by the repository’s ignore rules; consequently, the final checkpoint could not be independently rerun on the manual CCTV masks from the public artifacts alone. The source CCTV resource is described by AI-Hub [19] and remains subject to its access and use terms.

## References

[1] V. Ivanchenko, J. Coughlan, and H. Shen, “Detecting and locating crosswalks using a camera phone,” in IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 1–8, 2008.

[2] R. F. Berriel, F. S. Rossi, A. F. de Souza, and T. Oliveira-Santos, “Automatic large-scale data acquisition via crowdsourcing for crosswalk classification: A deep learning approach,” Computers & Graphics, vol. 68, pp. 32–42, 2017.

[3] H. Liang and S. Seo, “Detection of zebra-crossing areas based on deep learning with combination of SegNet and ResNet,” Journal of the Korean Society of Surveying, Geodesy, Photogrammetry and Cartography, vol. 39, no. 3, pp. 141–148, 2021.

[4] R. Verma and S. V. Ukkusuri, “Crosswalk detection from satellite imagery for pedestrian network completion,” Transportation Research Record: Journal of the Transportation Research Board, vol. 2678, no. 7, pp. 845–856, 2024.

[5] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention, pp. 234–241, 2015.

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

[7] L.-C. Chen, G. Papandreou, F. Schrof, and H. Adam, “Rethinking atrous convolution for semantic image segmentation,” arXiv preprint arXiv:1706.05587, 2017.

[8] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schrof, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European Conference on Computer Vision, pp. 801– 818, 2018.

[9] D.-H. Lee, “Pseudo-label: The simple and eficient semisupervised learning method for deep neural networks,” in ICML Workshop on Challenges in Representation Learning, 2013.

[10] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Advances in Neural Information Processing Systems, vol. 30, pp. 1195– 1204, 2017.

[11] K. Sohn, D. Berthelot, C.-L. Li, Z. Zhang, N. Carlini, E. D. Cubuk, A. Kurakin, H. Zhang, and C. Rafel, “FixMatch: Simplifying semi-supervised learning with consistency and confidence,” in Advances in Neural Information Processing Systems, vol. 33, pp. 596–608, 2020.

[12] V. Olsson, W. Tranheden, J. Pinto, and L. Svensson, “Class-Mix: Segmentation-based data augmentation for semisupervised learning,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1369–1378, 2021.

[13] X. Chen, Y. Yuan, G. Zeng, and J. Wang, “Semi-supervised semantic segmentation with cross pseudo supervision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2613–2622, 2021.

[14] L. Yang, W. Zhuo, L. Qi, Y. Shi, and Y. Gao, “ST++: Make self-training work better for semi-supervised semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4268– 4277, 2022.

[15] Y.-H. Tsai, W.-C. Hung, S. Schulter, K. Sohn, M.-H. Yang, and M. Chandraker, “Learning to adapt structured output space for semantic segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7472–7481, 2018.

[16] J. Hofman, E. Tzeng, T. Park, J.-Y. Zhu, P. Isola, K. Saenko, A. A. Efros, and T. Darrell, “CyCADA: Cycleconsistent adversarial domain adaptation,” in Proceedings of the 35th International Conference on Machine Learning, pp. 1989–1998, 2018.

[17] W. Tranheden, V. Olsson, J. Pinto, and L. Svensson, “DACS: Domain adaptation via cross-domain mixed sampling,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1379–1389, 2021.

[18] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[19] National Information Society Agency, “Ai-hub: Cctv traffic video for solving trafic problems (urban roads).” AI-Hub dataset no. 165, 2021. Accessed 8 September 2026.

[20] L. R. Dice, “Measures of the amount of ecologic association between species,” Ecology, vol. 26, no. 3, pp. 297–302, 1945.