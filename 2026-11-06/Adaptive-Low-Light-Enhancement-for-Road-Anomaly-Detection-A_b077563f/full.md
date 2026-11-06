# Adaptive Low-Light Enhancement for Road Anomaly Detection: A Multi-Architecture Benchmark and Edge Deployment Study

Obed Honour Eje

Oluwatosin Kolade

Mechatronics Engineering

Mechanical Engineering

Glory Bagai

Federal University of Technology Minna

Minna, Nigeria

Obafemi Awolowo University

Ile-Ife, Nigeria

obed.m2200396@st.futminna.edu.ng

Mechatronics Engineering

okolade@student.oauife.edu.ng

Federal University of Technology Minna

Minna, Nigeria

kuzayet.m2204354@st.futminna.edu.ng

Taliha Abiodun Folorunso

Federal University of Technology Minna

Mechatronics Engineering

Minna, Nigeria

funso.taliha@futminna.edu.ng

Abstract—Road anomaly detection systems on autonomous vehicles must function reliably under degraded visual conditions including darkness, rain, and Harmattan dust haze. This paper benchmarks four object detection architectures, YOLOv8n, YOLOv12n, RT-DETR, and Faster R-CNN, under physically grounded lighting and weather degradation, and shows that uniform application of the SCI low-light enhancement network is actively harmful at dusk-level lighting (−16.1% to −30.7% mAP@50). We propose a lightweight, training-free luminanceadaptive router that selects between no enhancement, moderate SCI, and denoising plus strong SCI based on per-frame Rec. 601 luminance, recovering most of the dusk penalty while preserving low-light gains, and we validate the full pipeline on NVIDIA Jetson Nano edge hardware.

Index Terms—road anomaly detection, low-light image enhancement, adaptive routing, YOLO, RT-DETR

## I. INTRODUCTION

Road-traffic crashes remain a severe public-health and economic burden in Nigeria: the Federal Road Safety Corps recorded 5,421 deaths from 9,570 crashes in 2024, a 7% rise over 2023 [1], and the WHO places Nigeria’s road-traffic death rate at 21.4 per 100,000, among the highest in Africa [2]. Poor infrastructure, potholes, eroded shoulders, poor lighting, weak signage, is repeatedly implicated, and the World Bank estimates a socio-economic cost of USD 29 billion in 2016, over 7% of GDP [3]. Automated pothole and speed-bump detection is a safety-critical perception task for advanced driver-assistance systems (ADAS) and autonomous vehicles on West African roads, yet most detectors are trained and validated under favourable daytime illumination. Reliability degrades sharply under conditions routine in Nigeria: unlit night-time roads, the rainy season, and the Harmattan, a seasonal Saharan dust haze that can cut visibility to 200–500 m between November and February [6]. These conditions reduce contrast, add signal-dependent noise, and increase atmospheric scattering, weakening the features detectors depend on.

A natural mitigation is a front-end enhancement module that restores visibility before detection [9], but enhancement optimised for human perception does not necessarily benefit, and can actively harm, machine vision, since the two objectives are misaligned [10], [11]. Retraining the detector on augmented data is more robust but computationally expensive and impractical to repeat per degradation type on resourceconstrained edge hardware such as the Jetson Nano.

Vision-based road-damage detection has moved from classical image processing toward CNN detectors, with YOLO variants now predominant [5], [15]; multi-country benchmarks such as RDD2020/2022 [16, 17] target daytime conditions, few studies address low light explicitly [18], and African contexts remain under-represented. Detector design spans twostage detectors such as Faster R-CNN [14], one-stage detectors such as the YOLO series [4], [12], [19], and realtime transformer detectors such as RT-DETR, the first DETRstyle model to outperform comparable YOLOs in speed and accuracy [13]; comparative benchmarking of these families under degradation remains sparse. Low-light enhancement has advanced from Retinex theory to deep networks such as Zero-DCE [7], EnlightenGAN [8], and the lightweight SCI network [9], but such enhancement can harm downstream detection through noise, edge blur, and colour artefacts [10], [11]. Adverse weather poses a related challenge: rain and haze are commonly modelled via the Koschmieder scattering equation [20], with physically grounded studies reporting detection drops of 25–30% under extreme rain [22]. Harmattan haze is well characterised in the atmospheric-science literature [6] but is essentially absent from the detection-robustness literature, a gap this paper begins to address.

Closest to our approach, IA-YOLO [23], GDIP-YOLO [24], and DENet/DE-YOLO [25] integrate learned, alwayson enhancement end-to-end with the detector, while YOLO-DER [26] and ClassLIE [27] make a binary enhance/bypass decision after additional training; none provides a trainingfree, three-way routing decision with a full-bypass branch for road-anomaly detection. Benchmarks confirm lightweight YOLO variants are typically the only models approaching real-time on a Jetson Nano [28], often requiring TensorRT, quantisation, or pruning [29]; we extend these benchmarks by jointly accounting for the added cost of the enhancement front-end.

![](images/45f41030e6648ec43a50c92073fa57839b2b67fd7af079cd961cce80a08e4380.jpg)  
Fig. 1. The proposed luminance-adaptive routing architecture.

This paper investigates whether front-end image enhancement can serve as a practical, real-time perception-recovery strategy for road-anomaly detection on edge-deployed vehicles. Our contributions are fourfold: a four-architecture benchmark (YOLOv8n, YOLOv12n [4],[12], RT-DETR [13], Faster R-CNN [14]) under simulated lighting/weather degradation; evidence that uniform SCI enhancement harms dusk-level detection (−16.1% to −30.7% mAP@50); a lightweight, training-free luminance-adaptive router that recovers most of this loss; and a Jetson Nano edge-latency benchmark of the full pipeline.

## II. METHODOLOGY

The overall architecture of the proposed perception-recovery pipeline is illustrated in fig. 1. Rather than forcing all images through a static enhancement front-end, the system ingests degraded frames and computes a lightweight perceptual luminance score to dynamically route the signal. Depending on this score, the frame is either bypassed entirely, passed through a moderate enhancement network, or heavily denoised and maximally enhanced before detection. This dynamic routing ensures that computational resources on constrained edge hardware are only expended when genuinely necessary to recover features.

## A. Dataset Construction

The evaluation corpus was constructed by merging two publicly available road-anomaly datasets under a targeted classconsolidation protocol: the Rajdalsaniya Pothole Detection Dataset (Kaggle, 2022; 2,077 images, single “pothole” class) [30] and the speed-unmarked-bump dataset v3 (Roboflow Universe, 2023; 2,367 images, “pothole” and “speed bump” classes, with manhole-cover variants excluded) [31]. The merged corpus comprises 4,444 images across two classes (pothole, speed bump), resized to 640 × 640 px and partitioned

70/15/15 into training, validation, and test splits. All degradation pipeline development and router threshold derivation were conducted on the 667-image validation split. The heldout 666-image test split was reserved for final confirmation of generalisability and evaluated after all methodological decisions were fixed.

## B. Degradation Pipeline

A physically grounded degradation pipeline was built from scratch after preliminary trials with albumentations-based brightness-shift augmentation caused catastrophic model collapse (YOLOv8n map@50 at dusk fell from 0.88 to 0.056) due to unrealistic linear brightness artefacts. Lighting degradation was simulated via gamma compression combined with signal-dependent (Poisson-like) noise at four severity levels, calibrated to target mean luminance values: clean $( \gamma = 1 . 0 , \mathrm { n o }$ noise, mean luminance ≈ 0.462), dusk $( \gamma = 1 . 6 , \sigma = 0 . 0 2 5$ ≈ 0.332), night $( \gamma = 2 . 8 , \sigma = 0 . 0 6 0 , \approx 0 . 2 0 7 )$ , and severe $( \gamma = 4 . 2 , \sigma = 0 . 1 1 0$ , plus 5-pixel horizontal motion blur, ≈ 0.141), with noise variance scaling with local pixel brightness to approximate Poisson shot noise. Weather degradation was simulated independently for three conditions: light rain (400 streaks, 7 px) and heavy rain (700 streaks, 10 px) via a custom scene-adaptive streak renderer, and Harmattan fog via the Koschmieder atmospheric-scattering equation [20] with a depth-gradient transmission map and a warm dust tint (RGB multiplier [1.0, 0.97, 0.92]). Lighting and weather degradations were applied independently, not compounded, and a visual sanity check preceded full-batch generation for each condition.

## C. Detection Architectures and Training

Four detection architectures spanning three design paradigms were trained and evaluated under identical data splits: YOLOv8n [4] and YOLOv12n [12] (anchor-based and anchor-free, attention-enhanced one-stage CNN detectors, nano variants), RT-DETR [13] (transformer-based, end-to-end, anchor-free), and Faster R-CNN with a ResNet-50 Feature Pyramid Network backbone [14] (two-stage, region-proposalbased). YOLOv8n, YOLOv12n, and RT-DETR were trained using the Ultralytics framework; Faster R-CNN was trained using torchvision. All four models were trained on the clean (non-degraded) training split and evaluated, without retraining, on the degraded validation images, isolating the effect of test-time visual degradation and preprocessing from any domain-adaptation effect of training-time augmentation.

## D. Low-Light Enhancement Module

Self-Calibrated Illumination (SCI) [9] was selected as the enhancement module for its favourable accuracy-to-cost ratio: 0.0003 M parameters, 0.0619 GFLOPs, and approximately 1.7 ms per frame on a TITAN X. The enhanced output R is obtained as follows:

$$
L = \mathcal { H } _ { \theta } ( I ) , \quad \hat { R } = \mathrm { c l i p } \left( \frac { I } { L } , \ 0 , \ 1 \right)\tag{1}
$$

where I is the degraded input image and L is the illumination map estimated by the SCI network H .Two pretrained weight variants released by the original authors were used: medium.pt for moderate low-light conditions and difficult.pt for severely dark scenes. SCI was applied as a fixed, nonfine-tuned front-end preprocessor; it was not retrained or adapted to the road-anomaly domain. SCI was deliberately not evaluated on weather-degraded images: Harmattan fog frames are already high-luminance due to the white haze overlay, and applying SCI to them produced complete whiteout; weatherspecific restoration (dehazing, deraining) was judged to require domain-specific modules and is left as future work.

## E. Luminance-Adaptive Router

To address the finding that uniform SCI enhancement is harmful when the input signal is already adequate, a luminance-adaptive router was developed. Perceptual luminance is computed per frame using Rec. 601 weights, adding approximately 1.2 ms of overhead on a Jetson Nano:

$$
L ( I ) = 0 . 2 9 9 R + 0 . 5 8 7 G + 0 . 1 1 4 B\tag{2}
$$

Routing thresholds were set as coarse round-number boundaries between the measured mean luminance centroids of each degradation level (dusk ≈ 0.332, night ≈ 0.207, severe ≈ 0.141): 0.40 separates adequate-light from dusk, and 0.20 separates dusk from severe darkness. No threshold search was performed against detection metrics; the boundaries require only luminance statistics, not detection labels, preserving the router’s training-free and label-free character.

$$
\begin{array} { r } { \mathcal { P } ( I ) = \left\{ \begin{array} { l l } { I } & { \mathrm { i f ~ } L \geq 0 . 4 0 \quad \mathrm { ( B r a n c h ~ A ) } } \\ { \hat { R } _ { \mathrm { m e d i u m } } ( I ) } & { \mathrm { i f ~ } 0 . 2 0 \leq L < 0 . 4 0 \quad \mathrm { ( B r a n c h ~ B ) } } \\ { \hat { R } _ { \mathrm { d i f f i c u l t } } ( \mathcal { D } ( I ) ) } & { \mathrm { i f ~ } L < 0 . 2 0 \quad \mathrm { ( B r a n c h ~ C ) } } \end{array} \right. } \end{array}\tag{3}
$$

The router is a fixed, rule-baseddecision function; it involves no learned parameters and required no additional training.

## F. Evaluation Protocol

Detection accuracy was measured using mAP@50 (mean Average Precision at an IoU threshold of 0.50), computed separately per lighting level, per weather condition, and per router branch, and compared across three configurations: (i) baseline (no preprocessing), (ii) uniform SCI applied indiscriminately to every frame, and (iii) the proposed luminanceadaptive router. Relative degradation was additionally reported as percentage retention of clean-condition mAP@50. All evaluations used the Ultralytics evaluation loop for YOLOv8n, YOLOv12n, and RT-DETR, and a torchvision-based evaluation loop for Faster R-CNN, applied consistently across all three preprocessing configurations.

## G. Edge Deployment Setup

Edge feasibility was assessed by deploying all four detectors, the SCI module, and the luminance-adaptive router on an NVIDIA Jetson Nano. Per-stage latency (luminance computation, enhancement, detection) and end-to-end framesper-second (FPS) were measured for each model on each router branch, using 50 timed runs per configuration following a 10-run warm-up period that was excluded from reported statistics. This protocol isolates the fixed cost of luminance computation and SCI enhancement from the variable cost of the downstream detector backbone.

![](images/1426ae5770f1c554de956a09475dcf4789af7304ad2a632a543383f2dda4946f.jpg)  
Fig. 2. Baseline mAP@50 under lighting degradation.

## III. RESULTS

## A. Baseline Robustness Under Degradation

Under baseline (no-preprocessing) conditions, mAP@50 declined sharply from clean to severe lighting for all four architectures (Fig. 1), with a pronounced non-linear cliff between dusk and night. RT-DETR was the most robust, retaining 98% of clean performance at dusk and 37% at severe darkness, while Faster R-CNN retained only 28% at severe, the weakest of the four. Under weather degradation, light rain caused negligible loss (90–100% retention) and Harmattan fog produced graduated degradation (50–70% retention, RT-DETR strongest), but heavy rain caused a catastrophic collapse to 24–31% retention across all architectures (Fig. 2), substantially worse than any lighting condition. RT-DETR’s global attention mechanism consistently gave it an advantage under partial occlusion (fog, rain), whereas Faster R-CNN’s two-stage region-proposal pipeline was the most fragile across nearly every degraded condition.

## B. Uniform Enhancement is Harmful at Dusk

Applying SCI uniformly caused significant mAP@50 degradation at dusk across all four architectures (Table I), ranging from -16.1% (RT-DETR) to -30.7% (Faster R-CNN). The same pass was conversely beneficial at night and severe darkness for RT-DETR and YOLOv12n, and consistently harmful for Faster R-CNN except at severe. This confirms enhancement value is condition- and architecture-dependent, not uniformly positive.

## C. Adaptive Router Recovers the Dusk Penalty

The router substantially recovered the plain-SCI dusk penalty (Table I), reducing the residual loss to 0.001–0.073 mAP@50 below baseline across both validation and heldout test splits, with RT-DETR, YOLOv12n, and YOLOv8n remaining within 0.007 of baseline dusk performance on unseen test data. At night, RT-DETR retained the strongest gain under the router (+0.045 validation, +0.071 test split), YOLOv12n showed consistent recovery (+0.036 test), while YOLOv8n and Faster R-CNN showed additional losses, the latter exceeding its plain-SCI loss and confirming the fixed thresholds are not uniformly optimal for two-stage detectors. Baseline degradation trends were consistent across splits, with absolute mAP@50 values 2–6 points lower on unseen test data as expected.

TABLE I  
COMPLETE MAP@50 RESULTS. D COLUMNS SHOW DELTA VS BASELINE
<table><tr><td>Model</td><td>System</td><td>Clean</td><td>Dusk</td><td>Night</td><td>Severe</td><td>Dusk D</td><td>Night D</td><td>Severe D</td></tr><tr><td rowspan="3">RT-DETR</td><td>Baseline</td><td>0.883</td><td>0.861</td><td>0.612</td><td>0.324</td><td></td><td></td><td></td></tr><tr><td>Plain SCI</td><td>0.883</td><td>0.722</td><td>0.679</td><td>0.387</td><td>-0.139</td><td>+0.067</td><td>+0.063</td></tr><tr><td>Adaptive</td><td>0.883</td><td>0.847</td><td>0.657</td><td>0.338</td><td>-0.014</td><td>+0.045</td><td>+0.014</td></tr><tr><td rowspan="3">YOLOv12n</td><td>Baseline</td><td>0.894</td><td>0.867</td><td>0.570</td><td>0.305</td><td></td><td></td><td></td></tr><tr><td>Plain SCI</td><td>0.894</td><td>0.706</td><td>0.601</td><td>0.368</td><td>-0.161</td><td>+0.031</td><td>+0.063</td></tr><tr><td>Adaptive</td><td>0.894</td><td>0.829</td><td>0.577</td><td>0.294</td><td>-0.038</td><td>+0.007</td><td>-0.011</td></tr><tr><td rowspan="3">YOLOv8n</td><td>Baseline</td><td>0.881</td><td>0.855</td><td>0.590</td><td>0.268</td><td></td><td></td><td></td></tr><tr><td>Plain SCI</td><td>0.881</td><td>0.713</td><td>0.582</td><td>0.282</td><td>-0.142</td><td>-0.008</td><td>+0.014</td></tr><tr><td>Adaptive</td><td>0.881</td><td>0.836</td><td>0.554</td><td>0.218</td><td>-0.019</td><td>-0.036</td><td>-0.050</td></tr><tr><td rowspan="3">Faster R-CNN</td><td>Baseline</td><td>0.779</td><td>0.755</td><td>0.548</td><td>0.215</td><td></td><td></td><td></td></tr><tr><td>Plain SCI</td><td>0.779</td><td>0.523</td><td>0.442</td><td>0.208</td><td>-0.232</td><td>-0.106</td><td>-0.007</td></tr><tr><td>Adaptive</td><td>0.779</td><td>0.687</td><td>0.407</td><td>0.163</td><td>-0.068</td><td>-0.141</td><td>-0.052</td></tr></table>

ArchiteqtureRobustness: mAP@50 Retention AcrossyAlltConditions  
![](images/ceea6a231a0e8e4508ede49ff7527aa6ca0725bf5bc638f412488b79cf2593ab.jpg)  
Fig. 3. Architecture robustness: mAP@50 retention (percentage of clean baseline) across all lighting and weather conditions.

## D. Edge Deployment Throughput and Latency

YOLOv8n was the only architecture approaching realtime operation (15-25.8 FPS) across all branches (Table II); YOLOv12n remained close (11.4-18.6 FPS). RT-DETR and Faster R-CNN fell well short at every branch despite stronger degradation robustness, confirming a direct accuracy-latency trade-off. SCI added a fixed overhead independent of detector (16 ms Branch B, 36 ms Branch C), making its relative cost largest for the fastest models.

## IV. DISCUSSION

These results show that front-end enhancement is not a universally safe intervention: SCI improves detection only when the input signal is genuinely inadequate, and actively degrades it when applied to scenes, such as dusk, that already carry sufficient luminance for the detector’s feature extractor.

TABLE II  
JETSON NANO PIPELINE LATENCY. FPS THRESHOLDS: ≥20 REAL-TIME,≥15 SAFETY MINIMUM, <15 BELOW SAFETY
<table><tr><td>Model</td><td>Branch</td><td>Lum. (ms)</td><td>Enh. (ms)</td><td>Det. (ms)</td><td>FPS</td></tr><tr><td rowspan="3">RT-DETR</td><td>A (Clean)</td><td>1.2</td><td>0</td><td>120.4</td><td>8.23</td></tr><tr><td>B (Dusk)</td><td>1.2</td><td>8.3</td><td>121.3</td><td>7.65</td></tr><tr><td>C (Severe)</td><td>1.2</td><td>27.3</td><td>120.9</td><td>6.69</td></tr><tr><td rowspan="3">YOLOv12n</td><td>A (Clean)</td><td>1.2</td><td>0</td><td>52.7</td><td>18.54</td></tr><tr><td>B (Dusk)</td><td>1.2</td><td>13.2</td><td>52.4</td><td>14.97</td></tr><tr><td>C (Severe)</td><td>1.2</td><td>33.8</td><td>52.4</td><td>11.44</td></tr><tr><td rowspan="3">YOLOv8n</td><td>A (Clean)</td><td>1.2</td><td>0 16.2</td><td>37.5</td><td>25.83</td></tr><tr><td>B (Dusk)</td><td>1.2</td><td></td><td>37.6</td><td>18.18</td></tr><tr><td>C (Severe)</td><td>1.2</td><td>36.0</td><td>36.8</td><td>13.52</td></tr><tr><td rowspan="3">Faster R-CNN</td><td>A (Clean)</td><td>2.0</td><td>0</td><td>224.5</td><td>4.42</td></tr><tr><td>B (Dusk)</td><td>2.0</td><td>8.3</td><td>223.6</td><td>4.28</td></tr><tr><td>C (Severe)</td><td>2.0</td><td>38.5</td><td>222.0</td><td>3.81</td></tr></table>

The luminance-adaptive router demonstrates that a simple, training-free, threshold-based decision is sufficient to capture most of this effect, recovering the majority of the dusk penalty without any additional model training. RT-DETR’s consistent robustness across both lighting and weather conditions is plausibly attributable to its global self-attention mechanism, which is less reliant on local high-frequency texture than the convolutional backbones of the YOLO models and Faster R-CNN.

## V. CONCLUSION

This paper benchmarked four object-detection architectures for road-anomaly detection under physically grounded lighting and weather degradation and showed that uniform low-light enhancement is actively harmful at dusk, degrading mAP@50 by 16–31% across architectures. A lightweight, training-free luminance-adaptive router, selecting between no enhancement, moderate SCI, and denoising plus strong SCI based on perframe Rec. 601 luminance, recovered 71–90% of the plain-SCI dusk penalty on the validation split and maintained dusk performance within 0.007 of baseline on the held-out test split, with RT-DETR retaining a night-level gain of +0.071 mAP@50 on unseen data. Edge validation on a Jetson Nano confirmed that only YOLOv8n, and marginally YOLOv12n, are suitable for real-time deployment at this hardware tier.

## VI. FUTURE WORKS

Immediate extensions include real-world data collection under Nigerian night-time, Harmattan, and rainy-season conditions to validate the synthetic findings, and compounded degradation benchmarking combining lighting and weather simultaneously. Weather-specific restoration modules (dehazing, deraining), per-architecture threshold calibration for twostage detectors, and model compression for sub-Jetson-Nano deployment remain open directions.

## REFERENCES

[1] Federal Road Safety Corps (FRSC), “2024 annual road traffic crash report,” press briefing by Corps Marshal S. Mohammed, Abuja, Nigeria, Jan. 2025.

[2] World Health Organization, Global Status Report on Road Safety 2023. Geneva, Switzerland: WHO, 2023. [Online]. Available: https://www.who.int/publications/i/item/9789240086517

[3] World Bank Global Road Safety Facility, “The single organizational model for road safety in Nigeria,” World Bank Group, Washington, DC, USA, 2023. [Online]. Available: https://www.globalroadsafetyfacility.org/sites/default/files/2023-10/ Nigeria%20The%20%E2%80%9CSingle%20Organization%20Road% 20Safety%E2%80%9D%20Institutional%20Model%2C%20its% 20Efficacy%20and%20Replicability.pdf

[4] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[5] Y. Safyari, M. Mahdianpari, and H. Shiri, “A review of vision-based pothole detection methods using computer vision and machine learning,” Sensors, vol. 24, no. 17, p. 5652, 2024.

[6] A. C. Anuforom, “Spatial distribution and temporal variability of Harmattan dust haze in sub-Sahel West Africa,” Atmos. Environ., vol. 41, no. 39, pp. 9079–9090, 2007.

[7] C. Guo et al., “Zero-reference deep curve estimation for low-light image enhancement,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 1780–1789.

[8] Y. Jiang et al., “EnlightenGAN: Deep light enhancement without paired supervision,” IEEE Trans. Image Process., vol. 30, pp. 2340–2349, 2021.

[9] L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward fast, flexible, and robust low-light image enhancement,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5637–5646.

[10] H. Liu, F. Jin, H. Zeng, H. Pu, and B. Fan, “Image enhancement guided object detection in visually degraded scenes,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 10, pp. 14164–14177, 2023.

[11] Y. Wang et al., “Is underwater image enhancement all object detectors need?,” IEEE J. Ocean. Eng., vol. 49, no. 2, pp. 606–621, 2023.

[12] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric realtime object detectors,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 38, 2026, pp. 78433–78457.

[13] Y. Zhao et al., “DETRs beat YOLOs on real-time object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16965–16974.

[14] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), vol. 28, 2015.

[15] A. Dhiman and R. Klette, “Pothole detection using computer vision and learning,” IEEE Trans. Intell. Transp. Syst., vol. 21, no. 8, pp. 3536– 3550, 2019.

[16] D. Arya et al., “Crowdsensing-based road damage detection challenge (CRDDC’2022),” in Proc. IEEE Int. Conf. Big Data (Big Data), 2022, pp. 6378–6386.

[17] D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, and Y. Sekimoto, “RDD2022: A multi-national image dataset for automatic road damage detection,” Geosci. Data J., vol. 11, no. 4, pp. 846–862, 2024.

[18] Y. Zanevych et al., “Evaluation of pothole detection performance using deep learning models under low-light conditions,” Sustainability, vol. 16, no. 24, p. 10964, 2024.

[19] J. Terven, D.-M. Cordova-Esparza, and J.-A. Romero-Gonz´ alez, “A´ comprehensive review of YOLO architectures in computer vision: From YOLOv1 to YOLOv8 and YOLO-NAS,” Mach. Learn. Knowl. Extr., vol. 5, no. 4, pp. 1680–1716, 2023.

[20] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 12, pp. 2341–2353, 2010.

[21] Z. Chen et al., “Object detection for autonomous vehicles under adverse weather conditions,” Expert Syst. Appl., 2025, Art. no. 128994.

[22] S. S. Halder, J.-F. Lalonde, and R. de Charette, “Physics-based rendering for improving robustness to rain,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 10203–10212.

[23] W. Liu et al., “Image-adaptive YOLO for object detection in adverse weather conditions,” in Proc. AAAI Conf. Artif. Intell., vol. 36, no. 2, 2022, pp. 1792–1800.

[24] S. Kalwar et al., “GDIP: Gated differentiable image processing for object detection in adverse conditions,” arXiv preprint arXiv:2209.14922, 2022.

[25] Q. Qin, K. Chang, M. Huang, and G. Li, “DENet: Detection-driven enhancement network for object detection under adverse weather conditions,” in Proc. Asian Conf. Comput. Vis. (ACCV), 2022, pp. 2813–2829.

[26] R. Gao, M. H. Omar, and M. Mahmuddin, “YOLO-DER: A dynamic enhancement routing framework for adverse weather vehicle detection,” Electronics, vol. 14, no. 24, p. 4851, 2025.

[27] Z. Wei, Y. Wang, L. Sun, A. V. Vasilakos, and L. Wang, “ClassLIE: Structure- and illumination-adaptive classification for low-light image enhancement,” IEEE Trans. Artif. Intell., vol. 5, no. 9, pp. 4765–4775, 2024.

[28] H. Feng, G. Mu, S. Zhong, P. Zhang, and T. Yuan, “Benchmark analysis of YOLO performance on edge intelligence devices,” Cryptography, vol. 6, no. 2, p. 16, 2022.

[29] D. K. Alqahtani, M. A. Cheema, and A. N. Toosi, “Benchmarking deep learning models for object detection on edge computing devices,” in Proc. Int. Conf. Service-Oriented Comput., Singapore: Springer, 2024, pp. 142–150.

[30] Rajdalsaniya, “Pothole detection dataset,” Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/rajdalsaniya/ pothole-detection-dataset

[31] “Speed-unmarked-bumb dataset, v3,” Roboflow Universe, 2023. [Online]. Available: https://universe.roboflow.com/pothole-detection-1nczj/ speed-unmarked-bumb