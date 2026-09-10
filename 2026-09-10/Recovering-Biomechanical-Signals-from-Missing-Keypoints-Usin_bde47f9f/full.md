# Recovering Biomechanical Signals from Missing Keypoints Using Temporal Interpolation in Monocular Gait Analysis

Shubham Rajeshkumar Jariwala

Information Systems Technology and Design, Singapore University of Technology and Design, 8 Somapah Road, Singapore 487372, Singapore

ORCID: 0009-0005-4735-4759

Corresponding author: Shubham Jariwala (email: shubhamrajeshkumar\_jariwala@mymail.sutd.edu.sg)

## Abstract

Monocular pose estimation enables low-cost gait analysis but is sensitive to missing keypoints caused by occlusion, detection errors, or efficiency-driven model reduction. While prior work on recovering missing joints focuses on complex learned models, the effectiveness of simple temporal methods remains underexplored. We evaluate knee-angle estimation under a missingankle-keypoint condition and test a first-order temporal interpolation scheme as a recovery mechanism. Across 527 frames of monocular walking video (428 with valid baseline detections), removing the ankle keypoint increased mean angular error to $2 3 . 4 ^ { \circ } \pm 4 6 . 7 ^ { \circ }$ and collapsed signal variance to near zero. Temporal interpolation reduced error to $1 . 1 ^ { \circ } \pm 6 . 7 ^ { \circ }$ (Wilcoxon signedrank, $\mathsf { p } < 1 0 ^ { - 5 0 } )$ and restored variance and smoothness to within a few percent of baseline. These results indicate that gait signals possess sufficient temporal redundancy for a simple, computationally trivial interpolation scheme to recover a critical missing joint, without resorting to learned reconstruction models. The findings support low-complexity, real-time-compatible designs for gait analysis in resource-constrained or occlusion-prone monocular settings.

## Keywords

gait analysis; pose estimation; temporal interpolation; missing keypoints; knee angle; monocular video

## 1. Introduction

Monocular human pose estimation has become an attractive route to low-cost, camera-only gait analysis, avoiding the expense and lab constraints of marker-based motion capture [1,2]. Its practical value, however, depends on the reliability of individual keypoint estimates, which degrade under occlusion, motion blur, low-confidence detections, or the use of lighter models chosen for real-time throughput.

Existing approaches to recovering missing pose information generally rely on learned priors. These include Bayesian and geometric-constraint models that infer occluded joints directly [7,8], temporal-context networks that exploit information across frames [3,9], and full 3D reconstruction pipelines such as OpenCap [4] or deep learning-based monocular capture systems [5], which fuse multi-view or biomechanically constrained information.

It remains unclear whether a much simpler approach treating the missing keypoint purely as a temporal gap and filling it from adjacent frames is sufficient to recover a biomechanically

meaningful signal. This is the question addressed here, using knee-angle estimation as the target signal and ankle-keypoint loss as the perturbation, since the ankle is both commonly occluded in monocular views and directly load-bearing for the knee-angle calculation.

## 2. Methods

## 2.1 Pipeline

Monocular walking video (a single participant, male, height 1.70 m, mass 64 kg) was processed frame-by-frame with a YOLO pose estimator (YOLOv8-pose family) [6] to extract the 17 COCO-format keypoints [10]. The knee angle θ was computed from the hip (H), knee (K), and ankle (A) keypoints as the angle at K between vectors K→H and K→A:

$$
\theta = a r c c o s [ ( K {  } H \cdot K {  } A ) / ( \mathbb { I } K {  } H \mathbb { I } \ \mathbb { K } {  } A \mathbb { I } ) \jmath \quad ( I )
$$

## 2.2 Conditions

Three conditions were compared over the same 527-frame sequence: (i) full baseline, knee angle computed from all detected keypoints; (ii) missing ankle, the ankle keypoint discarded for every frame, simulating a hard occlusion or detector dropout; and (iii) temporal interpolation, in which the missing ankle position is recovered using a first-order exponential smoothing update, ankle′ₜ $= 0 . 7 { \cdot } \mathrm { a n k l e } _ { \mathrm { t } - 1 } ^ { \prime } + 0 . 3$ ·ankleₜ (using the raw detected position directly when no prior recovered value exists), and the knee angle is recomputed from this smoothed position.

Even in the full-baseline condition, YOLO failed to return valid keypoints for 99 of 527 frames (baseline detection rate: 81.2%), reflecting realistic detector dropout on a single monocular recording rather than an idealized signal. All angular-error and statistical comparisons below use the 428 frames with valid baseline detections, so that recovery performance is measured against a real, rather than synthetic, reference signal.

## 2.3 Metrics and statistics

For each condition we computed: (i) mean angular error relative to the full-baseline knee angle, (ii) signal variance, and (iii) signal smoothness, defined as the mean absolute frame-to-frame change, mean(|θₜ − θₜ₋₁|). Missing-ankle versus interpolated errors were compared using the Wilcoxon signed-rank test, chosen over a paired t-test because per-frame error distributions are right-skewed, with standard deviations exceeding their means.

The video recording used in this study was of the author, who provided consent for its use in this research. No third-party participant data were collected.

## 3. Results

Table 1 summarizes mean angular error, standard deviation, variance, and smoothness for the three conditions, computed over the 428 frames with valid baseline detections.

<table><tr><td colspan="1" rowspan="1">Condition</td><td colspan="1" rowspan="1">Mean error (°)</td><td colspan="1" rowspan="1">SD (°)</td><td colspan="1" rowspan="1">Variance</td><td colspan="1" rowspan="1">Smoothness</td></tr><tr><td colspan="1" rowspan="1">Full baseline</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2177.8</td><td colspan="1" rowspan="1">1.97</td></tr><tr><td colspan="1" rowspan="1">Missing ankle</td><td colspan="1" rowspan="1">23.4</td><td colspan="1" rowspan="1">46.7</td><td colspan="1" rowspan="1">0.0                 0</td><td colspan="1" rowspan="1">.01</td></tr><tr><td colspan="1" rowspan="1">Interpolated</td><td colspan="1" rowspan="1">1.1</td><td colspan="1" rowspan="1">6.7</td><td colspan="1" rowspan="1">2192.0</td><td colspan="1" rowspan="1">2.12</td></tr></table>

Table 1. Angular error, variance, and smoothness across the full baseline, missing-ankle, and interpolated conditions (n = 428 valid frames).

Removing the ankle keypoint degrades the signal on all three measures simultaneously: error rises sharply, and variance and smoothness both collapse toward zero, indicating the knee-angle trace becomes effectively frozen once the ankle is lost consistent with the angle calculation losing one of its two defining vectors, rather than merely becoming noisier.

Temporal interpolation reverses this collapse rather than only reducing the point-wise error: variance and smoothness after interpolation both return to within a few percent of baseline, alongside the large reduction in mean error (Wilcoxon signed-rank test, missing-ankle vs. interpolated, $\mathfrak { p } \approx 7 . 6 \times 1 0 ^ { - 5 9 } )$ . This indicates that the recovered signal is not just closer to baseline on average, but restores the underlying temporal dynamics of the gait cycle.

![](images/e9d9ce405618a9e711d1cb4edccc6b8e151f7c3bf145e8c92944625970338482.jpg)

![](images/caced319867bd01866492b946c38069f2aa5cc214116eccd6023994eeb2eafc5.jpg)

![](images/58f3d2647d5e40db9bd7d84005e2cfebde33e8da92398b427b3cf79cb790cf60.jpg)  
Figure 1: pipeline diagram and knee-angle trajectory / error / smoothness comparison across conditions.

## 4. Discussion

These results show that a first-order temporal interpolation arguably the simplest possible recovery strategy is sufficient to recover a biomechanically meaningful knee-angle signal after losing a keypoint that is directly load-bearing for the computation. This suggests that gait signals carry strong short-timescale redundancy: the ankle's position over a single missing frame is well predicted by its immediately preceding position, at typical video frame rates and walking cadences.

This has a practical implication for monocular gait-analysis system design: robustness to occasional keypoint dropout (occlusion, low-confidence detections, or the 81.2% baseline detection rate observed even without any injected perturbation in this dataset) does not necessarily require a learned reconstruction model. A negligible-cost interpolation step can substantially mitigate this class of failure.

## 5. Limitations

This study uses a single participant and a single walking sequence, so the generalizability of the recovery magnitude $( 2 3 . { \overset { \vartriangle } { 4 ^ { \circ } } } \to { \bar { 1 } } . 1 ^ { \circ } )$ to other gait patterns, camera angles, or occlusion durations is untested. The interpolation scheme was evaluated only for single-frame ankle loss; performance under longer occlusion runs (multiple consecutive missing frames) is expected to degrade and was not characterized here. All reference (“ground truth”) angles are themselves derived from the same monocular pipeline's own detections rather than an independent motioncapture reference, so absolute error magnitudes should be interpreted as internal consistency rather than validated accuracy against a gold standard.

## 6. Conclusion

Simple temporal interpolation recovers a knee-angle signal from a missing, load-bearing keypoint with high fidelity $( 2 3 . 4 ^ { \circ }  1 . 1 ^ { \circ }$ mean error) in a monocular gait pipeline, without requiring a learned reconstruction model. This supports low-complexity designs for robust, realtime-compatible gait analysis in occlusion-prone or resource-constrained monocular settings.

## Acknowledgements

The author thanks the Agency for Science, Technology and Research (A\*STAR), Singapore, for the opportunity to undertake this work during an attachment.

## CRediT author statement

Shubham Jariwala: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization.

## Data availability statement

The analysis code and experiment CSV files supporting this study will be made publicly available in a GitHub repository upon acceptance of this manuscript. Prior to acceptance, the repository is maintained privately by the corresponding author to protect priority of the findings. The monocular video recording is not publicly shared to protect participant identifiability; it is available from the corresponding author on reasonable request.

## Declaration of competing interests

The author declares no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, OpenPose: realtime multi-person 2D pose estimation using part affinity fields, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017.

[2] J. Martinez, R. Hossain, J. Romero, J.J. Little, A simple yet effective baseline for 3D human pose estimation, in: Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017.

[3] A. Zeng, X. Sun, F. Huang, M. Liu, Q. Xu, Learning temporal representations for human pose estimation, in: Proc. Eur. Conf. Comput. Vis. (ECCV), 2018.

[4] A. Mundt et al., OpenCap: 3D human movement dynamics from smartphone videos, PLoS Comput. Biol. 19 (10) (2023).

[5] H. Zhou et al., Monocular human motion capture with deep learning, IEEE Trans. Pattern Anal. Mach. Intell. (2021).

[6] G. Jocher, A. Chaurasia, J. Qiu, Ultralytics YOLOv8 [software], version 8.0.0, Ultralytics, 2023. https://github.com/ultralytics/ultralytics.

[7] A. A. Dursun, T. E. Tuncer, Estimation of partially occluded 2D human joints with a Bayesian approach, Digit. Signal Process. 114 (2021) 103056.

[8] X. Guo, Y. Dai, Occluded joints recovery in 3D human pose estimation based on distance matrix, in: Proc. 24th Int. Conf. Pattern Recognit. (ICPR), IEEE, 2018, pp. 1325–1330.

[9] M. Ghafoor, A. Mahmood, Quantification of occlusion handling capability of a 3D human pose estimation framework, IEEE Trans. Multimedia (2021).

[10] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft COCO: common objects in context, in: Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740–755.