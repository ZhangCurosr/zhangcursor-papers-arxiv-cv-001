# CapsuleMotion: A Lightweight Real-Time Visual Motion Predictor for Capsule Endoscopy

Oliver Bause<sup>1</sup>

0009-0003-5388-2959

Julia Werner<sup>1</sup> 0009-0006-0279-1776

Oliver Bringmann<sup>1</sup> 0000-0002-1615-507X

Abstract—Video Capsule Endoscopy (VCE) is a non-invasive medical examination that allows for the observation of the small intestine, which is otherwise difficult to access. A fundamental challenge persists in the form of their limited size in order to still be swallowable. The resulting restricted battery capacity, however, contradicts with the power-intensive nature of image capture and transmission. Therefore, we propose CapsuleMotion, a patient-specific dynamic capsule behavior that utilizes the available energy in a goal-oriented manner to increase the likelihood of a complete screening of the gastrointestinal tract. By investigating and combining metrics from the on-device image compression, CapsuleMotion predicts the motion between two successive frames. The camera’s frame rate will be modified in accordance with the predicted magnitude of motion. Furthermore, prior to entering the small intestine, the capsule operates in a low power mode with a significantly reduced frame rate. In this mode, the LocalizationNet is employed to determine the current organ, provided that motion was predicted. The proposed framework is evaluated on the Rhode Island VCE dataset and deployed on an ultra-low power single-core RISC-V demonstrator with an integrated hardware accelerator. CapsuleMotion demonstrated the capability to reduce electric energy consumption by up to 20.66% in comparison with conventional capsules that lack a dynamic frame rate. Additionally, the accuracy of detecting the entry point of the small intestine has been improved.

Index Terms—Video Capsule Endoscopy, On-Device Motion Estimation, Capsule Localization, Image Compression

## I. INTRODUCTION

Introduced in the early 2000s, wireless Video Capsule Endoscopy (VCE) utilizes a pill-sized device that patients swallow to examine the gastrointestinal (GI) tract [1, 2]. The capsule integrates a miniature image sensor, LED illumination, a transmitter, a battery, and a microcontroller, with the potential to incorporate additional sensors to expand diagnostic capabilities. As the Video Capsule (VC) travels through the GI tract, it captures images and transmits them to an external on-body receiver for later clinical analysis. This technology is primarily used to identify pathologies in the small intestine, a region otherwise inaccessible to standard gastroscopy or colonoscopy [3].

A significant challenge for VCE is the limited battery life, a direct result of the strict physical constraints required to ensure the capsule remains swallowable. For instance, the Medtronic PillCam™ SB3 offers an operational window of 8 to 12 hours at a frame rate of 2 to 6 frames per second (fps) [4]. However, as the time required for a capsule to traverse the GI tract varies greatly among patients and can exceed 12 hours, the procedure often risks being incomplete. This limitation can result in blind spots where potential pathologies in the small intestine or colon remain undetected.

![](images/c090b7a38ebfc36fafd1aff3894e4ae378443446aad098ac622d7cf250d10e15.jpg)  
Fig. 1. Proposed pipeline to train a motion predictor based on features that can be extracted from the image compression and combining it with a lightweight CNN to achieve a more precise localization and energy efficient screening.

Our Contribution: To increase the probability of a complete screening, this work introduces a patient-specific dynamic system behavior that adjusts the frame rate depending on the current location within the GI tract and the capsule’s estimated motion. The proposed pipeline is illustrated in Figure 1. First, CapsuleMotion investigates metrics from a hardware-suitable Adaptive Golomb-Rice (AGR) coding compression technique [5, 6] that supports the RAW Bayer output from the image sensor without requiring further preprocessing. These metrics are then used to predict the Mean Structural Similarity Index Measure (MSSIM) [7] of two successive frames with a ridge regression model [8]. In the event that the MSSIM falls below a certain threshold, motion is assumed, and the organ where the frame was taken is predicted on-device with a combination of a lightweight Convolutional Neural Network (CNN) and a Hidden Markov Model (HMM) with Viterbi Decoding, termed LocalizationNet [9], which contains only 64,000 8-bit weights. The motion-triggered localization reduces the amount of consecutive mislabeling within a difficult scene. As a result, this enables the deployment of a restrictive low power mode that drastically reduces the frame rate to save energy and is active until the Area of Interest (AoI), the small intestine, is reached. After the localization detect the transition into the AoI, the system will be set into the normal operation mode and the frame rate is adjusted according to CapsuleMotion’s prediction. The pipeline has been validated with the Rhode Island (RI) gastroenterology VCE dataset [10] and demonstrated on an ultra-low power singlecore RISC-V System-on-Chip (SoC) [11] with the hardware accelerator UltraTrail [12] integrated.

## II. RELATED WORK

Visual motion estimation has been widely studied in computer vision through visual odometry and ego-motion estimation. Raudies et al. conducted a review of classical methods based on optical flow, feature tracking, and geometric reconstruction, emphasizing the trade-off between accuracy and computational cost [13].

In the context of VCE, the process of motion estimation is characterized by a heightened degree of complexity, primarily due to tissue deformation, low-texture surfaces, and illumination changes. Early work by Liu et al. estimated capsule motion directly from monocular video for redundancy reduction in VCE recordings [14]. Later, Spyrou et al. evaluated feature-based visual odometry methods for capsule localization, showing that classical descriptors such as SIFT and SURF can provide useful motion information despite challenging imaging conditions [15].

However, recent approaches have shifted toward the utilization of deep learning methodologies. Turan et al. proposed Deep EndoVO, an RCNN-based framework for end-toend monocular pose estimation from endoscopic video [16]. While the effectiveness of these methods is evident, their high computational complexity restricts their applicability to devices with limited resources. These methods are intended for implementation subsequent to screening and prior to analysis by medical professionals. Therefore, their deployment within the VC to execute real-time motion estimation, which directly impacts the device’s operation, is not feasible.

Sensor-based localization approaches, in contrast, utilize onboard accelerometers, gyroscopes, or magnetometers to estimate capsule motion and orientation. Despite their computational efficiency, the sensors require space within the capsule, and their accuracy is limited by several factors. Among these are sensor drift, accumulated integration errors, magnetic interference, and patient movement, which can introduce significant localization uncertainty over time. Consequently, numerous systems mandate the use of external sensors, magnetic tracking hardware, or sensor-fusion techniques to ensure the reliability of pose estimates [17].

## III. METHODOLOGY

## A. Rhode Island VCE Dataset

The RI VCE dataset [10] was used to develop and validate the proposed pipeline. It consists of 424 complete studies with a total of over five millions frames labeled with their respective anatomical organ in the GI tract. LocalizationNet was trained using the official downsampled training and validation splits [9]. However, to train the ridge regression models, the 339 complete studies from the training and validation splits were utilized and randomly divided into an 80/20% split by study. It was imperative to ensure that the regression model precisely predicted the MSSIM of successive frames, rather than the MSSIM of two random frames from different patients. Furthermore, the MSSIM of sequenced frames at varying frame rates was analyzed to verify the assumption that it can be used as a motion predictor. As anticipated, the mean MSSIM between successive frames exhibits a decline in value as the frame rate is reduced. Nonetheless, the evaluation of both models was conducted by utilizing the official test set, which encompasses 85 complete studies.

## B. LocalizationNet

In preceding work [9], we presented the ultra-low power LocalizationNet that is a combination of a lightweight CNN with a quantized HMM requiring only 5.31 µJ per inference to precisely determine the organ of the current frame. Additional hardware-level optimizations further reduced it to just 1.53 µJ per inference. To enhance the image classification results, the CNN was combined with Viterbi decoding which computes the most likely sequence of hidden states given the HMM and the CNN predictions based on [18]. The sequence of hidden states directly corresponds to the presumed path traversed by the capsule through the organs. Furthermore, by reducing the frame rate, the accuracy of the network was increased which also allowed a reduction of the HMM window size and resulted in an overall power reduction before entering the AoI. The improvement in accuracy can be attributed to a reduction in the number of sequential frames analyzed from scenes that were challenging for the network to predict, leading to a decrease in consecutive mislabelings.

## C. Bayer Compression Pipeline

The efficient AGR compression pipeline, demonstrated in [5], achieved great compression results and, thus, was adapted for this framework. The compression was validated on the RI dataset and achieved similar results with an average compression ratio (CR) of 3.99 and a Peak Signal-to-Noise Ratio (PSNR) of 38.88 dB. It was then adjusted to also log different metrics during compression which were utilized to train the ridge regression MSSIM prediction model.

## D. CapsuleMotion

To estimate the motion of the capsule, we introduce the CapsuleMotion pipeline as illustrated in Figure 1. The goal is to predict the MSSIM between two sequential frames. Conversely, if the MSSIM exceeds 0.95, the images are deemed to be so similar that the motion between these two frames is considered negligible. Consequently, the captured image is deemed obsolete and is not transmitted to the on-body receiver. If it is higher than 0.85, the motion is slow and the capsule’s frame rate can be reduced to save power. Otherwise, the device operates at the pre-defined normal frame rate, typically between 2 and 6 fps. However, calculating the MSSIM ondevice is too computationally complex and energy intensive for an ultra-low power SoC which would evaporate any possible energy and time savings. Thus, the CapsuleMotion pipeline trains a simple ridge regression model that predicts the MSSIM between two frames with a few selected metrics that can be easily and efficiently extracted during the compression of a frame.

The pipeline starts by identifying possible metrics which can be extracted during the image compression. By compressing the images of the RI’s train set, 37 different metrics were identified:

• Sum of Absolute Difference (SAD) of each DCT component between two frames (4),

• Hamming Distance (HD) and length difference within the three encoded bitstrings (6),

• CR and PSNR (2),

• and SAD between frame slices by slicing the image into a $5 \times 5$ grid (25).

Then, an iterative feature selection based on the Variance Inflation Factor (VIF) eliminates stepwise metrics to mitigate multicollinearity in regression models. In each iteration, the algorithm computes the $\mathrm { V I F } _ { j }$ for each remaining independent metric $X _ { j }$ by first constructing an auxiliary ordinary least squares (OLS) regression. In this auxiliary model, $X _ { j }$ is treated as the dependent variable and regressed against all other independent variables in the current subset:

$$
X _ { j } = \beta _ { 0 } + \sum _ { k \neq j } \beta _ { k } X _ { k } + \epsilon .\tag{1}
$$

From this auxiliary regression, the coefficient of determination $R _ { j } ^ { 2 }$ is extracted, which quantifies the proportion of variance in $\check { X _ { j } }$ that is explained by the other predictors. The $\mathrm { V I F } _ { j }$ for variable $X _ { j }$ is then calculated as:

$$
\mathrm { V I F } _ { j } = { \frac { 1 } { 1 - R _ { j } ^ { 2 } } } .\tag{2}
$$

The variable that exhibits the highest VIF is identified. In the event that the maximum value exceeds a threshold of 10, as this is regarded as high [19] , the corresponding metric is eliminated from the feature subset, and the process is repeated. This cycle continues iteratively until the $\operatorname { V I F } _ { j }$ values of all remaining metrics fall below the specified threshold. Subsequently, ridge regression models were trained and validated on the validation set, incorporating all potential metric combinations of up to 10 features. This process resulted in the development of over 4,000 trained models. The objective was to select the best-performing one per number of features, which could then deployed into the VC.

## IV. RESULTS

## A. Feature Selection

After the iterative VIF-based feature elimination, 12 metrics remained in the set for possible ridge regression models. For each number of features, the best performing model on the validation set was selected and tested against the RI test set. The achieved coefficient of determination $( \mathbf { R } ^ { 2 } )$ and Root Mean Square Error (RMSE) of these 10 models are illustrated in Figure 2. The regression model demonstrates minimal enhancement in accuracy when more than four features are employed. Given that each additional feature increases the computational load, the analysis of the hardware demonstrator has been restricted to the four smallest models which are listed with their utilized features in Table I. However, even the calculation of the 4 metrics plus the inference of model 4 requires with 3.85 ms significantly less time than the calculation of the SSIM with 26.41 ms directly.

![](images/42ea62a0b19284ad62878d3ef96d7d1ec0adceb8445c4821e6ef242482f920d2.jpg)  
Fig. 2. SSIM predictor performance over different number of metrics utilized.

TABLE I  
METRICS USED IN THE FOUR SMALLEST RIDGE REGRESSION MODELS AND THEIR CORRESPONDING INFERENCE TIME TO COMPUTE THEM.
<table><tr><td rowspan="2">Metric</td><td colspan="4">Used in Model</td><td rowspan="2">Inference Time [ms]</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>SAD of AGR coefficient y2</td><td>X</td><td>X</td><td>X</td><td>X</td><td>1.792</td></tr><tr><td>HD of encoded y bitstring</td><td>=</td><td>X</td><td>1</td><td>X</td><td>0.0003</td></tr><tr><td>SAD of AGR coefficient cr</td><td></td><td>=</td><td>X</td><td>X</td><td>1.764</td></tr><tr><td>SAD of 64 × 64px center crop</td><td>=</td><td>=</td><td>X</td><td>X</td><td>0.279</td></tr><tr><td>RISC-V optimized SSIM</td><td>=</td><td>=</td><td>=</td><td></td><td>26.41</td></tr><tr><td>Total Model Inference Time [ms] | 1.795 1.799 3.844 3.848</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B. Hardware Demonstrator

The demonstrator was equipped with the 320 × 320px NanEyeC camera, four LEDs, a transceiver, and a single-core RISC-V PULPissimo SoC [11]. The SoC’s die size is just $1 . 8 \times 2 \mathrm { m m ^ { 2 } }$ and has the ultra-low power hardware accelerator UltraTrail integrated which executes the CNN part of the LocalizationNet. During compression and inference, the core was clocked at 200 MHz, while it was reduced to 25 MHz during image capturing and idle times to save power. The resulting electric energy consumptions split by module are listed in Table II.

## C. Test Study Simulation

All 85 test studies from the RI dataset were emulated on the demonstrator to validate its performance. Various operation modes were executed and compared against a baseline VC mode that captures images at a constant frame rate of 2 fps and directly transmit them to the on-body receiver without further analysis. Figure 3 illustrates the achieved electric energy savings across the different modes in comparison to the baseline. While the image compression [5] and the LocalizationNet [9] stand-alone achieved an average reduction by 10.90% and 8.01%, respectively, their combination even reached up to 16.01%. CapsuleMotion, however, further pushed the average savings to 20.47% and 20.66% for the regression models 1 and 2, respectively, while using only a Viterbi decoding window size of 5 instead of the 20 or 30 of the LocalizationNet modes. As model 3 and 4 require 114.2% more computational time than the smaller models, the energy savings achieved by the dynamic frame rate are canceled out by the more complex models, resulting in average savings of just 12.3%. Besides the power savings, the models needs to detect the entry into the AoI precisely. A late detection risks blind spots in the beginning of the small bowel while an early detection wastes battery by capturing and transmitting stomach images. The LocalizationNets with a Viterbi window size of 20 and 30 achieved a median detection delay of 86.5 and 65.5 s, respectively. This is still within the area of the small bowel, that is accessible by the classic gastroscopy. However, all CapsuleMotion models reduced the median delay across all models to an early detection at −17.5 s.

TABLE II  
POWER AND ELECTRIC ENERGY CONSUMPTION OF THE DEMONSTRATOR’S MODULES
<table><tr><td>Task</td><td>Module</td><td>IT [ms]</td><td>Power [mW]</td><td>Energy [μJ]</td></tr><tr><td>Image</td><td>NanEyeC</td><td>12.79</td><td>8.51</td><td>108.93</td></tr><tr><td>Capture</td><td>LEDs</td><td>12.79</td><td>14.78</td><td>189.15</td></tr><tr><td></td><td>Core</td><td>12.79</td><td>1.06</td><td>13.56</td></tr><tr><td>LocalizationNet</td><td>Core + Acc.</td><td>0.33</td><td>4.62</td><td>1.53</td></tr><tr><td>Compression</td><td>Core</td><td>51.8 - 94.7</td><td>4.28</td><td>221.7 - 405.3</td></tr><tr><td>CapsuleMotion</td><td>Core</td><td>1.8 - 3.89</td><td>4.28</td><td>7.72 - 16.65</td></tr><tr><td>Transmission</td><td>Transceiver</td><td>7.5 - 51.2</td><td>9.2</td><td>69 - 475.1</td></tr><tr><td>Idle</td><td>System</td><td>-</td><td>0.43</td><td>一</td></tr></table>

![](images/4ed37eba6ecccf11a01b3eb8e56bbb4a34041b2fb214f10af7dbee8167f5513a.jpg)  
Fig. 3. Electric energy savings achieved by the compression [5], LocalizationNet [9] with a HMM window size of 20 and 30 at a reduced fps of 0.25 and 0.5, respectively, the combination of LocalizationNet and compression, and the presented CapsuleMotion models 1 to 4 normalized against a baseline VCE.

## V. CONCLUSION

This work presents the efficient ridge regression-based motion estimator CapsuleMotion. It enables a patient-specific dynamic VCE screening without the need of additional onboard accelerometers or gyroscopes by extracting existing metrics from the compression pipeline and using them to predict the capsule’s motion. In combination with the lightweight LocalizationNet, the average electric energy savings per screening reached up to 20.66% compared to a standard capsule without a dynamic frame rate. At the same time, the median detection delay of the small bowel was reduced from more than 60 seconds late to just 17.5 s early. For future research, we want to validate this approach on further datasets and evaluate other possible motion predictors based on additional sensors, sensor fusions, or lightweight CNNs.

## REFERENCES

[1] G. Iddan, G. Meron, A. Glukhovsky, and P. Swain, “Wireless capsule endoscopy,” Nature, vol. 405, no. 6785, pp. 417–417, 2000.

[2] P. Swain, G. J. Iddan, G. Meron, and A. Glukhovsky, “Wireless capsule endoscopy of the small bowel: development, testing, and first human trials,” in Biomonitoring and Endoscopy Technologies, vol. 4158. SPIE, 2001, pp. 19–23.

[3] G. Costamagna et al., “A prospective trial comparing small bowel radiographs and video capsule endoscopy for suspected small bowel disease,” Gastroenterology, vol. 123, no. 4, pp. 999–1005, 2002.

[4] Medtronic, “PillCam™ SB3 System,” https://www.medtronic.com/covidien/en-nz/ products/capsule-endoscopy/pillcam-sb-3-system.html/, 2025, [Online; accessed 7- May-2025].

[5] O. Bause, J. Gamerdinger, J. Werner, and O. Bringmann, “Image compression with bubble-aware frame rate adaptation for energy-efficient video capsule endoscopy,” in Proceedings of the 48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2026, accepted for publication. [Online]. Available: https://arxiv.org/abs/2604.25464

[6] R. Rice and J. Plaunt, “Adaptive variable-length coding for efficient compression of spacecraft television data,” IEEE Transactions on Communication Technology, vol. 19, no. 6, pp. 889–897, 1971.

[7] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[8] G. C. McDonald, “Ridge regression,” Wiley Interdisciplinary Reviews: Computational Statistics, vol. 1, no. 1, pp. 93–100, 2009.

[9] O. Bause, J. Werner, P. Palomero Bernardo, and O. Bringmann, “Smart video capsule endoscopy: Raw image-based localization for enhanced gi tract investigation,” in Neural Information Processing. Singapore: Springer Nature Singapore, 2026, pp. 33–47.

[10] A. Charoen, A. Guo, P. Fangsaard, S. Taweechainaruemitr, N. Wiwatwattana, T. Charoenpong, and H. G. Rich, “Rhode island gastroenterology video capsule endoscopy data set,” Scientific Data, vol. 9, no. 1, p. 602, 2022.

[11] P. P. Bernardo et al., “A scalable risc-v hardware platform for intelligent sensor processing,” in 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2024, pp. 1–5.

[12] P. Palomero Bernardo, P. Schmid, C. Gerum, and O. Bringmann, “Compiler-aware ai hardware design for edge devices,” in Proceedings of the 8th International Workshop on Edge Systems, Analytics and Networking, 2025, pp. 31–36.

[13] F. Raudies and H. Neumann, “A review and evaluation of methods estimating egomotion,” Computer Vision and Image Understanding, vol. 116, no. 5, pp. 606–633, 2012.

[14] H. Liu, N. Pan, H. Lu, E. Song, Q. Wang, and C.-C. Hung, “Wireless capsule endoscopy video reduction based on camera motion estimation,” Journal of digita imaging, vol. 26, no. 2, pp. 287–301, 2013.

[15] E. Spyrou, D. K. Iakovidis, S. Niafas, and A. Koulaouzidis, “Comparative assessment of feature extraction methods for visual odometry in wireless capsule endoscopy,” Computers in biology and medicine, vol. 65, pp. 297–307, 2015.

[16] M. Turan, Y. Almalioglu, H. Araujo, E. Konukoglu, and M. Sitti, “Deep endovo: A recurrent convolutional neural network (rcnn) based visual odometry approach for endoscopic capsule robots,” Neurocomputing, vol. 275, pp. 1861–1870, 2018.

[17] M. A. Ali, N. Tom, F. N. Alsunaydih, and M. R. Yuce, “Recent advancements in localization technologies for wireless capsule endoscopy: A technical review,” Sensors, vol. 25, no. 1, p. 253, 2025.

[18] J. Werner, C. Gerum, M. Reiber, J. Nick, and O. Bringmann, “Precise localization within the gi tract by combining classification of cnns and time-series analysis of hmms,” in International Workshop on Machine Learning in Medical Imaging. Springer, 2023, pp. 174–183.

[19] S. Sheather, A modern approach to regression with R. Springer Science & Business Media, 2009.