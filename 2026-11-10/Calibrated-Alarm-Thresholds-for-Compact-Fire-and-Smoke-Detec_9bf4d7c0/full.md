# Calibrated Alarm Thresholds for Compact Fire and Smoke Detectors

Vikas Sandireddy

California, USA

vikas.sandireddy@jessup.edu

Abstract—Compact object detectors such as YOLOv8n are a common choice for camera-based fire and smoke detection on low-power devices. These detectors are usually evaluated with box-level metrics such as mean average precision, and their outputs are turned into decisions with a fixed confidence threshold such as 0.25 or 0.5. A fire monitoring system, however, makes one decision per image: raise an alarm or not. In this paper we evaluate two publicly available compact detectors on the D-Fire test set at the image level. We compare three simple rules for turning box detections into one alarm score per image, and we use a split conformal procedure to choose the alarm threshold from held-out images so that the expected fraction of missed fire and smoke images stays below a chosen target. We find that the same fixed threshold leads to very different miss rates for the two detectors, 17.3 and 8.4 percent at a threshold of 0.5, while the calibrated threshold meets miss-rate targets of 10, 5 and 1 percent for both detectors and all three rules. At a 5 percent target, the false alarm rate on images without fire or smoke stays between 1.2 and 3.0 percent. We also find that smoke-only images stay below the target even when the overall target is met, and we describe the main failure cases.

Index Terms—fire detection, smoke detection, conformal prediction, threshold selection, artificial intelligence, computer vision

## I. INTRODUCTION

Early detection of fire and smoke can limit damage and save lives. Camera-based detection is attractive because cameras are cheap and already installed in many places. Early vision methods used hand-crafted color rules for flame pixels [1]. Later work moved to convolutional neural networks (CNNs) [2], and today most systems use object detectors from the YOLO family [3]. Small variants of these detectors, such as YOLOv5n or YOLOv8n, are popular because they can run on low-power devices [4], [5].

Most papers in this area report box-level metrics such as precision, recall and mean average precision (mAP) on benchmarks like D-Fire [4]. A deployed system, however, has to answer a simpler question for every frame: is there fire or smoke in this image or not? Two choices sit between the detector output and this answer. The first is how to turn a set of boxes into one decision. The second is which confidence threshold to use. In practice the threshold is usually fixed, for example at 0.25 [6], and evaluations of detector calibration often fix it at 0.3 or 0.5 [7]. Detector confidence scores are not calibrated probabilities [8], so the same threshold can mean different things for different models. For a fire alarm, the most important quantity is the fraction of fire and smoke images that are missed, and a fixed threshold gives no control over it.

This work examines these two choices for compact detectors on D-Fire and uses a split conformal procedure [9] to choose the alarm threshold. The rest of the paper is organized as follows. Section II reviews related work, Section III describes the dataset, the detectors and the threshold selection method, Section IV presents and discusses the results, and Section V concludes the paper.

## II. RELATED WORK

## A. Vision-Based Fire and Smoke Detection

Celik and Demirel [1] proposed a rule-based color model in the YCbCr color space for classifying flame pixels and reported fire detection rates of up to 99% on their test images. Rule-based color methods are sensitive to changes in brightness and to shadows, which can cause false alarms [2]. Muhammad et al. [2] proposed a CNN inspired by GoogleNet for fire detection in CCTV surveillance videos. De Venancioˆ et al. [4] released the D-Fire dataset of 21,527 images, trained a YOLOv4 detector on it, and pruned its convolutional filters so that it could run on a Raspberry Pi 4. Alkhammash [5] compared YOLOv9t, YOLOv10n and YOLOv11n for smoke and fire detection on two datasets, one of which is D-Fire. Feng et al. [6] proposed HIFE-YOLO, a lightweight detector based on YOLO11n, and trained it on D-Fire. Besides box-level metrics, they report a background false alarm rate, defined as the share of images without any annotation that receive at least one predicted box at a confidence of 0.25. Outside D-Fire, Dewangan et al. [10] treat wildfire smoke detection from fixed cameras as a binary image classification problem and note that clouds, fog and haze are a common source of false positives.

## B. Post-Detection Processing and Calibration

Joshi et al. [11] proposed an uncertainty-aware postdetection framework for compact fire and smoke detectors. A small Confidence Refinement Network combines a dropoutbased uncertainty estimate with color, edge and texture features of each box and outputs a refined confidence score. Applied to YOLOv5n and YOLOv8n on D-Fire, it gave higher precision and recall than non-maximum suppression (NMS), Soft-NMS and several color- and edge-based filters, at a modest extra processing time. The refined scores are then thresholded at a fixed value of 0.5. Our work is complementary: we leave the scores unchanged and study how the final threshold is chosen.

Guo et al. [8] showed that modern neural networks are often poorly calibrated, meaning that their confidence scores do not match their actual accuracy. For object detectors, Kuzucu et al. [7] point out that choosing an operating threshold is not trivial, that common evaluations fix it at values such as 0.3 or 0.5, and that simple post-hoc calibrators work well. In fire detection, Zhang et al. [12] added a differentiable calibration-error loss to fire image classifiers and reduced their calibration error on the DFAN and EdgeFireSmoke datasets. Better-calibrated scores are useful, but they do not by themselves guarantee a given miss rate.

## C. Error Control With Statistical Guarantees

Conformal prediction gives finite-sample guarantees for the output of any trained model, as long as the calibration data and the test data are exchangeable [9]. Conformal risk control extends this idea to the expected value of monotone losses such as the false negative rate [13]. In binary classification, Tong et al. [14] proposed the Neyman-Pearson umbrella algorithm, which picks the threshold from an order statistic of held-out scores of one class so that the error on that class stays below a chosen level with high probability. Luo et al. [15] used conformal prediction to tune warning systems in robotics so that they have a provable false negative rate. For object detection, Andeol et al. [16] proposed sequential conformal risk control,´ which gives guarantees on the boxes and classes produced by a detector. In the fire domain, Dayan [17] applied conformal risk control to wildfire spread prediction and controlled the pixel-level false negative rate. Our setting is different from these works: we control the miss rate of an image-level alarm built on top of a camera-based fire and smoke detector.

## III. METHODOLOGY

## A. Dataset

We use the D-Fire dataset [4]. It contains 21,527 images in four categories: fire only, smoke only, fire and smoke, and none. Images in the last category contain neither fire nor smoke, but often show scenes that can be mistaken for them. The authors split the dataset into a training set of 17,221 images and a test set of 4,306 images. All our experiments use only the 4,306 test images, which contain 2,878 fire boxes and 2,315 smoke boxes. We call an image positive if it has at least one fire or smoke box, and negative otherwise.

## B. Detectors

We did not train any detector. Instead we evaluate two publicly available compact detectors that were trained on D-Fire:

• YOLOv8n [18], fine-tuned on D-Fire and released on Hugging Face (https://huggingface.co/rabahdev/ fire-smoke-yolov8n). Its model card reports an mAP@50 of 0.754 on the D-Fire test split.

• YOLOv5s [19], released by the authors of D-Fire (https: //github.com/pedbrgs/Fire-Detection).

Both models were run once over the test images at an input size of 640 pixels, keeping every box with a confidence of at least 0.001 and at most 100 boxes per image. For NMS we used the default IoU threshold of each framework, 0.7 for YOLOv8n and 0.6 for YOLOv5s. All later analysis works on these saved boxes, so no further inference is needed.

## C. Box-Level Metrics

To place our detectors next to published results, we first compute the usual box-level metrics: precision and recall at a confidence of 0.25 with an IoU of 0.5, average precision at IoU 0.5 for each class (AP@50), and its mean over the two classes (mAP@50), as well as mAP averaged over IoU thresholds from 0.5 to 0.95 (mAP@50–95). We use greedy matching and 101-point interpolation.

## D. From Boxes to an Image Score

For an image I, let $c _ { 1 } , \ldots , c _ { m }$ be the confidences of its predicted boxes and $a _ { 1 } , \ldots , a _ { m }$ their areas as a fraction of the image area. We compare three rules that map these boxes to one score s(I):

$$
s _ { \operatorname* { m a x } } ( I ) = \operatorname* { m a x } _ { j } c _ { j } ,\tag{1}
$$

$$
s _ { \mathrm { n o r } } ( I ) = 1 - \prod _ { j \in T _ { 3 } ( I ) } ( 1 - c _ { j } ) ,\tag{2}
$$

$$
s _ { \mathrm { a r e a } } ( I ) = \operatorname* { m a x } _ { j : a _ { j } \geq 0 . 0 0 1 } c _ { j } ,\tag{3}
$$

where $T _ { 3 } ( I )$ holds the three most confident boxes. The score is 0 when no box qualifies. Rule (1) is the standard choice: an alarm is raised if any single box is confident enough. Rule (2) is the noisy-OR used in multiple instance learning [20], which lets several weak boxes add up. Rule (3) ignores very small boxes, covering less than 0.1% of the image, on the idea that they are often noise. An alarm is raised when $s ( I ) \geq t$ for a threshold t. Both classes count: an alarm is correct on a positive image whether it comes from a fire box or a smoke box.

## E. Image-Level Metrics

Let P and N be the sets of positive and negative images. For a threshold t, the detection rate is $\mathrm { D R } ( t ) \ = \quad$ $\begin{array} { r } { \frac { 1 } { | P | } \sum _ { I \in P } \mathbf { 1 } [ s ( I ) \geq t ] } \end{array}$ and the false alarm rate is $\mathrm { F A R } ( t ) =$ $\begin{array} { r } { \frac { 1 } { | N | } \sum _ { I \in N } \mathbf { 1 } [ s ( I ) \geq t ] } \end{array}$ . The miss rate is $1 - \mathrm { D R } ( t )$ . We also report the area under the ROC curve (AUC) of each score.

## F. Calibrated Threshold Selection

We want a threshold that keeps the miss rate below a target α, for example 5%. Let $s _ { ( 1 ) } \leq \cdots \leq s _ { ( n ) }$ be the sorted scores of n positive calibration images. We set

$$
\hat { t } = s _ { ( k ) } , \qquad k = \lfloor \alpha ( n + 1 ) \rfloor ,\tag{4}
$$

and raise an alarm for every image whose score is at least t<sup>ˆ</sup> (if $k = 0$ , every image raises an alarm). If a new positive image is exchangeable with the calibration images, its score is equally likely to fall in any position among the n + 1 scores. It falls below $s _ { ( k ) }$ with probability at most $k / { ( n + 1 ) } \leq \alpha ,$ so

$$
\operatorname* { P r } \left[ s ( I _ { \mathrm { n e w } } ) < \hat { t } \right] \leq \alpha .\tag{5}
$$

<table><tr><td>Metric</td><td>YOLOv8n</td><td>YOLOv5s</td></tr><tr><td>Precision</td><td>0.685</td><td>0.678</td></tr><tr><td>Recall</td><td>0.724</td><td>0.781</td></tr><tr><td>mAP@50</td><td>0.746</td><td>0.771</td></tr><tr><td>mAP@50-95</td><td>0.424</td><td>0.429</td></tr><tr><td>AP@50, fire</td><td>0.684</td><td>0.720</td></tr><tr><td>AP@50, smoke</td><td>0.807</td><td>0.822</td></tr></table>

This is the split conformal argument [9] applied to positive images only. The guarantee holds on average over the choice of calibration set. The Neyman-Pearson umbrella algorithm [14] uses the same kind of order statistic but picks k so that the guarantee holds with high probability, which needs more data. Only positive images are needed to set $\hat { t } ,$ and the false alarm rate is whatever the detector delivers at that threshold.

Since we have only one labeled split, we repeat the following 200 times. We randomly divide the 4,306 test images into two halves, keeping the share of each category the same in both. We compute t<sup>ˆ</sup> from the positive images in the first half and measure DR and FAR on the second half. We report the mean and standard deviation over the 200 repeats for $\alpha \in \{ 0 . 1 0 , 0 . 0 5 , 0 . 0 1 \}$ , and compare with the fixed thresholds 0.25 and 0.5 on the same halves.

## IV. RESULTS AND DISCUSSION

## A. Box-Level Results

Table I shows the box-level metrics. Our mAP@50 for YOLOv8n is 0.746, close to the 0.754 reported on its model card, which suggests that our evaluation matches the standard one. For both detectors, fire is the harder class: its AP@50 is 0.684 and 0.720, compared with 0.807 and 0.822 for smoke. Joshi et al. [11] observe the same pattern for compact YOLOv5n and YOLOv8n detectors on D-Fire, and for YOLOv5n their post-detection framework gave its largest improvement on the fire class. This suggests that fire remains the weaker class for compact detectors on this dataset.

## B. Image-Level Results With Fixed Thresholds

Table II shows the image-level results. Even though mAP@50 is only around 0.75, the AUC at the image level is between 0.990 and 0.996. Many errors that hurt box-level metrics, such as a box that is slightly misplaced or a missed second box in the same image, do not matter for an alarm.

The fixed thresholds, however, behave very differently for the two detectors. With the max rule at t = 0.5, YOLOv8n misses 17.3% of positive images while YOLOv5s misses 8.4%. At t = 0.25 the miss rates are 4.3% and 2.3%. The same threshold thus gives a twofold difference in missed fires, and without labeled data the user cannot know this. At t = 0.25, the false alarm rate of the max rule is 2.4% for YOLOv8n and 2.9% for YOLOv5s. This is the same quantity that Feng et al. [6] call the background false alarm rate, and their value for YOLO11n on D-Fire, 3.24%, is in the same range. The noisy-OR rule raises scores, so at a fixed threshold it detects more but also raises more false alarms, which again shows that a fixed threshold has no fixed meaning.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Rule</td><td rowspan="2">AUC</td><td colspan="2">t = 0.25</td><td colspan="2">t = 0.5</td></tr><tr><td>DR</td><td>FAR</td><td>DR</td><td>FAR</td></tr><tr><td rowspan="2">YOLOv8n</td><td>Max Noisy-OR</td><td>0.993 0.994</td><td>0.957 0.972</td><td>0.024</td><td>0.827</td><td>0.007</td></tr><tr><td>Area-gated</td><td>0.990</td><td>0.942</td><td>0.032 0.023</td><td>0.924 0.815</td><td>0.011 0.006</td></tr><tr><td rowspan="3">YOLOv5s</td><td>Max</td><td>0.995</td><td>0.977</td><td>0.029</td><td>0.916</td><td>0.009</td></tr><tr><td>Noisy-OR</td><td>0.996</td><td>0.987</td><td>0.038</td><td>0.958</td><td>0.014</td></tr><tr><td>Area-gated</td><td>0.992</td><td>0.959</td><td>0.025</td><td>0.901</td><td>0.007</td></tr></table>

TABLE III

RESULTS WITH CALIBRATED THRESHOLDS
<table><tr><td rowspan="2">Model</td><td rowspan="2">Rule</td><td colspan="2">α = 0.10</td><td colspan="2">α = 0.05</td><td colspan="2">α= 0.01</td></tr><tr><td>Miss</td><td>FAR</td><td>Miss</td><td>FAR</td><td>Miss</td><td>FAR</td></tr><tr><td rowspan="2">YOLOv8n</td><td>Max</td><td>0.098</td><td>0.013</td><td>0.048</td><td>0.023</td><td>0.009</td><td>0.092</td></tr><tr><td>Noisy-OR Area-gated</td><td>0.099 0.099</td><td>0.010</td><td>0.048</td><td>0.019</td><td>0.009</td><td>0.084</td></tr><tr><td rowspan="4">YOLOv5s</td><td>Max</td><td>0.099</td><td>0.015 0.008</td><td>0.049 0.049</td><td>0.030 0.014</td><td>0.009 0.009</td><td>0.191</td></tr><tr><td></td><td>0.099</td><td></td><td></td><td></td><td></td><td>0.069</td></tr><tr><td>Noisy-OR</td><td></td><td>0.005</td><td>0.049</td><td>0.012</td><td>0.009</td><td>0.067</td></tr><tr><td>Area-gated</td><td>0.099</td><td>0.008</td><td>0.049</td><td>0.019</td><td>0.010</td><td>0.219</td></tr></table>

## C. Calibrated Thresholds

Table III shows the results of the calibrated thresholds over the 200 random splits. In all 18 settings (two detectors, three rules, three targets), the mean miss rate on the evaluation halves is at or just below the target α. The detection rate varies from split to split with a standard deviation of about 0.01, so a single split can land slightly above the target. This is expected, since the guarantee holds on average over calibration sets [9].

The thresholds themselves show why calibration is needed. For a 5% miss target with the max rule, the median calibrated threshold is 0.277 for YOLOv8n and 0.420 for YOLOv5s. A fixed threshold of 0.5 is too high for both, and a single fixed value cannot suit both models.

The cost in false alarms is small for moderate targets and grows quickly for strict ones. At a 5% target, the false alarm rate is between 1.2% and 3.0%. At a 1% target, it rises to between 6.7% and 21.9%. Among the rules, noisy-OR gives the lowest false alarm rate at the 5% target for both detectors, as Fig. 1 also suggests at low false alarm rates. The area-gated rule is the worst at the 1% target (19.1% and 21.9% false alarms). Removing small boxes removes real evidence: some of the hardest positive images contain only a small, distant fire or a thin smoke plume, and the threshold must then drop to almost zero to catch them.

## D. Results per Category

The guarantee covers positive images as a whole, not each type of image. Fig. 2 breaks the alarm rate down by D-Fire category at the 5% target with the max rule. Images with both fire and smoke are almost always detected (0.996 for YOLOv8n and 0.992 for YOLOv5s), and fire-only images are above the target (0.954 and 0.981). Smoke-only images, however, are detected only 91.8% and 91.4% of the time, below the 95% target. The overall target is met because the easier categories make up for them. For a system whose main purpose is early warning, where faint smoke is often the first sign of a fire, this matters.

![](images/757ee2e2d514853c84487918a6ac28955c221295fc25eed2483eda9d7e98d8c3.jpg)

Fig. 1. Image-level ROC curves for the three rules. The horizontal axis is on a log scale to show the region of low false alarm rates.  
![](images/0b02cebc9b46a33a94aeb05c61f36582c990ae7a7fc713d727df567e9f7f2a41.jpg)  
Fig. 2. Alarm rate per D-Fire category with the max rule at a 5% miss target (mean and standard deviation over 200 splits). For the first three categories this is the detection rate; for “none” it is the false alarm rate. The dotted line marks 0.95.

## E. Failure Cases

Fig. 3 shows the four negative images with the highest scores and the four positive images with the lowest scores for YOLOv8n. The false alarms are mostly things that look like smoke or fire: a forest in fog, clouds over mountains, and a warm wall lamp in a bedroom. This matches the observation of Dewangan et al. [10] that clouds, fog and haze cause false positives. The highest-scoring negative images of YOLOv5s (not shown) are similar and also include street lights at night and decorative lights. Several false alarms on both detectors come from fixed hillside cameras, where the detector marks a small spot near the horizon. We cannot tell from the images whether some of these are distant smoke that was not labeled.

The misses fall into two groups. Some are very small targets, such as a tiny fire on the horizon at sunset or a faint plume in haze. Others have labels that are arguably ambiguous, such as a large white plume from the cooling towers of a power plant that is labeled as smoke. All of the misses shown have a score of 0.00, meaning that the detector produced no box for them at all, even at a confidence of 0.001. No choice of threshold can recover such images, which explains why very strict targets lead to many false alarms. We also noticed that two of the missed images are near-duplicates of each other. Feng et al. [6] checked D-Fire for exact duplicates and found none, but did not check for near-duplicates.

## F. Limitations

This study uses a single dataset and two detectors, so the results may not carry over to other datasets or models. The calibration assumes that new images resemble the calibration images, which may not hold when cameras, scenes or weather change after deployment. The guarantee also holds on average over all positive images, not for each category, as Section IV-D shows for smoke-only images. Finally, we work with single images, while real systems process video and could use information over time.

## V. CONCLUSION

We studied compact fire and smoke detectors as image-level alarms. On D-Fire, two public detectors separate positive and negative images very well (AUC above 0.99), but common fixed thresholds lead to very different miss rates for different models. A simple split conformal threshold, computed from the positive images in half of the test set, met miss-rate targets of 10%, 5% and 1% for both detectors and all three rules, at a false alarm rate of 1.2% to 3.0% for a 5% target. Noisy-OR was the best of the three rules, and ignoring small boxes hurt at strict targets. Smoke-only images remained below the target. In future work, we plan to calibrate separate thresholds per category, combine calibration with post-detection rescoring methods, test the approach on other datasets and on video, and study high-probability guarantees such as the Neyman-Pearson approach [14].

## REFERENCES

[1] T. C¸ elik and H. Demirel, “Fire detection in video sequences using a generic color model,” Fire Safety Journal, vol. 44, no. 2, pp. 147–158, 2009. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0379711208000568

[2] K. Muhammad, J. Ahmad, I. Mehmood, S. Rho, and S. W. Baik, “Convolutional neural networks based fire detection in surveillance videos,” IEEE Access, vol. 6, pp. 18 174–18 183, 2018.

[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” 2016, arXiv:1506.02640. [Online]. Available: https://arxiv.org/abs/1506.02640

[4] P. V. A. B. de Venancio, A. C. Lisboa, and A. V. Barbosa,ˆ “An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices,” Neural Computing and Applications, vol. 34, no. 18, pp. 15 349–15 368, 2022. [Online]. Available: https://doi.org/10.1007/s00521-022-07467-z

False alarm (none), score 0.69 False alarm (none), score 0.65

![](images/108e6795b6f567535c92630655163fc289c92be27ae5f6b89416ab6a6d9def71.jpg)  
Fig. 3. Failure cases of YOLOv8n with the max rule at the 5% miss target. Top: negative images with the highest scores (false alarms). Bottom: positive images with the lowest scores (misses). Green boxes are ground truth and red boxes are predictions above the threshold.

[5] E. H. Alkhammash, “A comparative analysis of YOLOv9, YOLOv10, YOLOv11 for smoke and fire detection,” Fire, vol. 8, no. 1, 2025. [Online]. Available: https://www.mdpi.com/2571-6255/8/1/26

[6] S. Feng, S. Feng, J. Hu, and Q. Su, “HIFE-YOLO: Hierarchical information flow enhancement for lightweight fire and smoke detection,” Fire, vol. 9, no. 11, p. 478, 2026. [Online]. Available: https://www.mdpi.com/2571-6255/9/11/478

[7] S. Kuzucu, K. Oksuz, J. Sadeghi, and P. K. Dokania, “On calibration of object detectors: Pitfalls, evaluation and baselines,” in Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXI. Berlin, Heidelberg: Springer-Verlag, 2024, p. 185–204. [Online]. Available: https://doi.org/10.1007/978-3-031-72664-4 11

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning - Volume 70, ser. ICML’17. JMLR.org, 2017, p. 1321–1330.

[9] A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction,” Found. Trends Mach. Learn., vol. 16, no. 4, p. 494–591, Mar. 2023. [Online]. Available: https://doi.org/10.1561/2200000101

[10] A. Dewangan, Y. Pande, H.-W. Braun, F. Vernon, I. Perez, I. Altintas, G. W. Cottrell, and M. H. Nguyen, “FIgLib & SmokeyNet: Dataset and deep learning model for real-time wildland fire smoke detection,” Remote Sensing, vol. 14, no. 4, 2022. [Online]. Available: https://www.mdpi.com/2072-4292/14/4/1007

[11] A. S. Joshi, G. J. William, and S. S. Joshi, “Uncertainty-aware postdetection framework for enhanced fire and smoke detection in compact deep learning models,” in 2025 International Conference on Smart Multimedia (ICSM), 2025, pp. 1–7.

[12] Z. Zhang, X. Zhou, and X. Gong, “Learning to calibrate for reliable visual fire detection,” 2025, arXiv:2502.09872. [Online]. Available: https://arxiv.org/abs/2502.09872

[13] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal risk control,” 2025, arXiv:2208.02814. [Online]. Available: https://arxiv.org/abs/2208.02814

[14] X. Tong, Y. Feng, and J. J. Li, “Neyman-Pearson classification algorithms and NP receiver operating characteristics,” Science Advances, vol. 4, no. 2, p. eaao1659, 2018. [Online]. Available: https: //www.science.org/doi/abs/10.1126/sciadv.aao1659

[15] R. Luo, S. Zhao, J. Kuck, B. Ivanovic, S. Savarese, E. Schmerling, and M. Pavone, “Sample-efficient safety assurances using conformal prediction,” Int. J. Rob. Res., vol. 43, no. 9, p. 1409–1424, Aug. 2024. [Online]. Available: https://doi.org/10.1177/02783649231221580

[16] L. Andeol, L. Mossina, A. Mazoyer, and S. Gerchinovitz, “Conformal´ object detection by sequential risk control,” 2025, arXiv:2505.24038. [Online]. Available: https://arxiv.org/abs/2505.24038

[17] B. Dayan, “Conformal risk control for safety-critical wildfire evacuation mapping: A comparative study of tabular, spatial, and graph-based models,” 2026, arXiv:2603.22331. [Online]. Available: https://arxiv.org/abs/2603.22331

[18] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” https: //github.com/ultralytics/ultralytics, 2023.

[19] G. Jocher, “YOLOv5 by Ultralytics,” https://github.com/ultralytics/ yolov5, 2020.

[20] C. Zhang, J. Platt, and P. Viola, “Multiple instance boosting for object detection,” in Advances in Neural Information Processing Systems, Y. Weiss, B. Scholkopf, and J. Platt, Eds., vol. 18. MIT¨ Press, 2005. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2005/file/56c82ccd658e09e829f16bb99457bcbc-Paper.pdf