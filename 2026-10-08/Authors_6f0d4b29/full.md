Title: trACT: temporal revelation Airborne Camera Trap

## Authors:

Oliver Bimber<sup>1\*</sup>, Rakesh John Amala Arokia Nathan<sup>1</sup>, Mohamed Youssef<sup>1</sup>, Vinayak Lal Bhatnagar<sup>1</sup>, Ralf Berger<sup>2</sup>, Klaus Hackländer<sup>3,4</sup>

## Affiliations:

<sup>1</sup>Department of Computer Science, Johannes Kepler University Linz, Austria

<sup>2</sup>Institute of Space Research, German Aerospace Center, Berlin, Germany

<sup>3</sup>BOKU University, Institute of Wildlife Biology and Game Management, , Vienna, Austria

<sup>4</sup>German Wildlife Foundation (Deutsche Wildtier Stiftung), Hamburg, Germany

\*Corresponding author. Email: oliver.bimber@jku.at

Abstract: Effective remote monitoring and surveillance using drones are frequently impeded by severe environmental and thermal clutter, dynamic vegetation, target camouflage, and system latency. Drawing inspiration from the hunting strategies of birds of prey that hover and stabilize their vision to isolate subtle ground motion, we introduce trACT (temporal revelation Airborne Camera Trap), a lightweight, real-time aerial robotics framework designed for autonomous consumer drones. The system integrates Temporal Max Pooling (TMP), a low-level signal processing method that transforms imperceptible movement across a rolling integration window into robust value and time encodings, with self-supervised motion anomaly detection to isolate target motion from background environmental motion caused by wind gusts and drone drift. To overcome mechanical and processing delays, trACT combines motion prediction with automated gimbalstabilized optical zoom verification and equitable multi-target verification balancing. Extensive realworld field experiments in densely forested wildlife habitats and surveillance scenarios demonstrate that trACT successfully bridges the gap between wide-area aerial monitoring and precise, autonomous target verification under challenging operational conditions.

One Sentence Summary: Inspired by birds of prey, trACT is an airborne camera trap designed for autonomous target detection and monitoring in cluttered environments.

## Main Text:

## INTRODUCTION

Sustainable wildlife management, including conservation, control and use, relies on the effective monitoring of animal populations, behaviors, and habitat conditions, as well as the implementation of robust anti-poaching measures. Traditional approaches depend heavily on manual field surveys and patrolling, which are labor-intensive, costly, and often limited in spatial and temporal coverage. To overcome these limitations, camera traps (1-3) and camera drones (4-11) have emerged as highly effective vision sensors for large-scale wildlife monitoring, offering the ability to survey extensive areas while minimizing disturbance to wildlife.

Recent advances in signal processing and deep learning have enabled the automatic detection and counting from aerial imagery using drones. Several studies have employed object detection frameworks to identify and monitor various species, including mammals and birds, achieving accurate population estimation over large geographical regions (12- 20). Aerial synthetic aperture imaging can mitigate vegetation-induced occlusion for detection (20,21), provided that the approximate locations of the animals are known and they remain stationary during the scanning process. These approaches have significantly reduced the manual effort required for wildlife surveys while improving monitoring efficiency. In addition to wildlife monitoring, these aerial perception systems play a critical role in anti-poaching operations (22-24), where detection of human intruders, vehicles, or suspicious activities enables rapid response and enhanced protection of endangered species in large and often inaccessible conservation areas.

Beyond spatial-based detection and counting, extracting motion information from video data remains fundamental to studying wildlife behavioral ecology, migration patterns, and population dynamics (1-3, 25-28). Existing computer vision approaches include absolute frame differencing (27), optical flow and background modeling (25), spatiotemporal segmentation (1), and deep neural networks (2,3,28). Although these methods demonstrate promising results under relatively simple, near-field conditions, achieving real-time performance often requires compromising spatial resolution or increasing computational overhead. Furthermore, they are frequently unsuitable for far-field (high-altitude aerial) monitoring, suffering significant performance degradation when targets and their motion occupy only a few pixels or are obscured by occlusion, camouflage, and thermal clutter.

Beyond wildlife management, vision-based surveillance applications suffer from similar problems. They encompasses aerial small-target monitoring (29-35), counter-drone defense (36-40), traffic surveillance (41-44), maritime safety (45-50), border security (51, 52), perimeter and railway intrusion detection (53-56), search and rescue operations (57-59) and security anomaly detection (60). These systems are deployed on unmanned aerial vehicles (UAVs) or on fixed, trackside and pan-tilt cameras. Most approaches perform detection on a frame-by-frame basis, only a few leverage inter-frame motion under platform ego-motion (32-35, 40, 55, 58, 60) or actively reposition to verify detected candidates (44, 48). Performance typically degrades in the far field due to challenges such as few-pixel targets, cluttered and dynamic backgrounds, occlusion, platform jitter, motion blur and the tradeoff between real-time processing and resolution on constrained hardware.

![](images/ee0825d911416beae8d0dbf98e075226cf26d0eb09538e6d334ba9971c48a371.jpg)  
Fig. 1. Airborne camera trap principle. Automatic waypoint flights with a camera drone enable the flexible scanning of large regions from high altitudes, while docking stations support continuous charging and remote monitoring (A). Motion detection from higher altitudes draws inspiration from the fixation strategies of birds of prey, such as common kestrels (Falco tinnunculus), which hover while stabilizing their heads to detect the motion of ground-level prey (B). Our drone mimics this behavior to detect subtle motion signals from a reference pose across a sliding time window, using wide-angle RGB or thermal images. Once motion is detected, the camera gimbal rotates to center and zoom in on the target. The resulting magnified image (daylight or night-vision RGB) is then used for verification. This scanning process runs directly on the drone hardware at a frequency of approximately 1Hz (C). Wideangle thermal images recorded from higher altitudes (86 m AGL in this example) obscure moving red deer within thermal clutter and limited spatial detail (D). Temporal Max Pooling (TMP) highlights the animals' motion trails with dark pixels (E). Automatic centering and zooming on the detected targets enable optical verification even from high altitudes (F). Boxes highlight the same ground area. See Supplementary Movie S1.

To address these problems, we propose the temporal revelation Airborne Camera Trap, trACT, a system designed for immediate deployment on consumer drones. Unlike conventional camera traps, trACT enables aerial, large-area coverage across flexible observation sites and provides continuous monitoring via automated waypoint flights (Fig. 1A). Operating such a system presents significant challenges: extreme visual clutter from dense vegetation, thermal environmental noise, and motion artifacts caused by wind and drone drift. Additionally, achieving reliable detection from higher altitudes is difficult due to constrained spatial and temporal resolutions, as well as limited onboard processing power.

![](images/c06cc42d720c019119db88066658ad8c16239112356926ba87261011de87775a.jpg)  
Fig. 2. The trACT processing pipeline. Temporal Max Pooling (TMP) (1.) serves as the initial signal processing component, detecting motion within a rolling image integration window to yield value and time encodings of motion trails. These channels are used for the self-supervised training and inference of a motion anomaly detection network (2.), which distinguishes targets from common background motion caused by wind (such as swaying vegetation and drone drift). Combined with Kalman filtering (3.), the TMP time channel enables the prediction of the target's actual position once processing latency has elapsed. An optional mapping strategy ensures balanced verification coverage across multiple detected targets. (4.) The predicted positions control the camera gimbal (centering and zooming) for optical verification (5.) alongside laser range-finding to acquire distance and GPS coordinates. This process loops on the drone’s hardware at approximately 1Hz. The magnified verification image then undergoes an offline classification to filter out outliers and identify the target (6.). Finally, the identified targets are plotted on a map using their GPS coordinates (7.).

We overcome these challenges by making the following contributions:

1. Our approach is inspired by the hunting strategies of birds of prey like common kestrels (Falco tinnunculus). When hunting, these birds hover while stabilizing their heads to detect ground-level prey motion (61,62), as illustrated in Fig. 1B and Supplementary Movie S1. To mimic this behavior, our drone hovers during inspection while rotating its camera gimbal to center and verify moving detections using optical zoom and laser-range finding for GPS mapping (Fig. 1C, Supplementary Movie S1). This scanning process runs continuously on the drone hardware (specifically, the remote controller's processor) at a frequency of around 1Hz.

2. We introduce Temporal Max Pooling, TMP as a simple, low-level, and lightweight signal processing framework for motion detection. By per-pixel pooling the maximal temporal value differences and integrated motion times over a substantially large (time) rolling integration window and normalizing the result, TMP transforms subtle, often imperceptible motion in individual frame-pairs into large and robustly detectable motion trails at almost no computational cost. TMP has also been field-tested in rainforest conditions and helped locate the last remaining wild Sumatran rhinos.

3. We apply Noise-Contrastive Estimation, NCE (63) to the TMP signal to isolate anomalous motion patterns caused by moving targets, such as animals or humans. This selfsupervised learning approach requires training on only a few drone videos of empty forests recorded under varying wind conditions. The method learns the statistical distribution of background motion (primarily caused by wind, such as swaying vegetation, and drone drift) and identifies pixels that deviate from this baseline.

4. We combine the time channel from the TMP signal with Kalman filtering to predict the target's position while compensating for imaging and processing delays. These coordinates then guide the camera gimbal and laser-range finder for optical zoom verification and GPS mapping.

5. We present an algorithmic solution to ensure balanced verification coverage in case multiple targets are visible at the same time. This avoids a bias towards targets with the strongest motion signals.

6. We apply vision-based reasoning for outlier removal and classification, achieving an overall recall of 80% under realistic conditions.

7. We evaluated our approach through two field experiments – one focusing on wildlife monitoring and the other on surveillance.

Figure 2 illustrates the interaction of these steps.

## RESULTS

The trACT process (Fig. 2) is executed at a specified waypoint, drone heading, and gimbal pitch. Although this pose can be chosen freely, the gimbal yaw is set to zero degrees to align it with the drone heading. TMP integral images, along with their value and time encodings, are computed for this reference pose. Details on how TMP integrals are calculated are provided in Materials and Methods.

Besides the regular flight mode (F), three operating modes are supported. The manual mode (M) enables user verification of anomalous motion patterns in the value-encoding (���<sub>�</sub>) displayed on the drone's remote control (RC). A touch selection on the RC triggers optical verification (also displayed on the RC) and GPS mapping. Further details on optical verification, mapping, and offline classification are provided in Materials and Methods.

The automatic single-target mode (AS) supports fully automatic detection and verification of a single target. If multiple targets are visible, the one with the strongest motion signal is prioritized. Details regarding motion anomaly detection and prediction are provided in Materials and Methods.

The automatic multi-target mode (AM) ensures that multiple detected targets are verified in a balanced manner without prioritizing the strongest motions. Details on multi-target verification are provided in Materials and Methods.

In addition, details regarding offline outlier removal, the classification of detected targets using vision reasoning, and the corresponding mapping procedures are provided in the Materials and Methods.

We first present the results of technical validations conducted under controlled conditions, followed by field experiments in wildlife monitoring and surveillance under challenging conditions.

## Technical Validation

Minimizing the time interval between reference-pose imaging and optical verification (achieved by centering and zooming in on detected motion) is critical, as targets may continue to move and motion prediction has inherent limitations. In our hardware implementation (see Materials and Methods), imaging requires a constant 33ms, while temporal max pooling (TMP) computations take 35ms on average (min: 20ms, max: 82ms). Motion anomaly detection averages 45ms (min: 24ms, max:123 ms), motion anomaly prediction averages 5ms (min: 0.2ms, max: 30ms), multi-target balancing averages 0.4ms (min: 0.065ms, max: 0.39ms), and optical verification averages 830ms (min: 818ms, max: 885ms). The optical verification step includes 800ms for mechanical gimbal rotation, with the remaining time allocated to zoom imaging and laser range measurements. This results in a total processing time of 946ms on average (approximately 1Hz) between initial imaging and final verification – an interval during which the target may shift position. Kalman filtering applied to the TMP time-encoding $( T M P _ { t } )$ enables the prediction of the target's location once the processing delay has elapsed, as explained in Materials and Methods. To account for runtime variations, we continuously measure and update the average processing time.

Imaging is performed using either wide-angle RGB or thermal cameras, while verification relies on the drone’s RGB zoom camera operating in either daylight or night-vision mode.

![](images/11280a9fd8ac4ece4d6fa2e86b9513e4d9b41408a20894d903f2f4b412f504a7.jpg)  
Fig. 3. Motion anomaly detection and multi-target verification. Wide-angle RGB experiment recorded from 120m above ground level, with background motion caused by wind gusts (row A) and drone drift (row C) clearly visible in the motion signal’s value $( T M P _ { v } )$ and time $( T M P _ { t } )$ encodings. Relative time (from oldest to newest) is color-coded in $T M P _ { t }$ . Close-ups are shown in row B. Motion anomaly detection efficiently filters out background motion and isolates the anomalous

motion patterns caused by two walking people. Ensuring balanced verification coverage across multiple detected targets is achieved by tracking how frequently reference-pose areas (circles) have been previously verified. The example in (D) illustrates three moving targets across 16 time intervals, with the next target to be verified indicated by a cross. See Supplementary Movie S1.

Figure 3 illustrates a wide-angle RGB experiment recorded from an altitude of 120m above ground level (AGL) under wind speeds of 14-25 km/h. In row A, local tree motion caused by wind gusts appears in both TMP signal channels, $T M P _ { v }$ and $T M P _ { t }$ . In row B, global environmental motion is visible in both channels due to drone drift. Row C presents closeup views. Our motion anomaly detection efficiently filters out these background signals, successfully isolating the anomalous motion trails of two walking people whose predicted positions are marked with crosses. An example of the automatic AM mode is shown in D. This mode balances verification coverage across all detected targets and avoids the bias toward the strongest motion signal as it is the case in the AS mode. See Supplementary Movie S1.

![](images/5e3a1b94223e5b583d0ecdeb2acbff5bc53699afaff12c9d46e43c1872af56b7.jpg)

![](images/6bc43aff52ef30ba438718d9e64f089272ed4abee5951e9f78fe67244b56a13d.jpg)  
Fig. 4. Partial occlusion. Verification images captured from an altitude of 40m AGL under a wind speed of 14-18 km/h demonstrate the successful wide-angle RGB detection of a moving quadruped robot under partial forest occlusion. The top-left inserts display the $T M P _ { v }$ channel highlighting the previous (circles) and current (box) verification regions. Notably, the system detected not only the robot but also nearby unoccluded people, whose verification images are omitted here. See Supplementary Movie S1.

Figure 4 shows verification images from wide-angle RGB recordings captured at an altitude of 40m AGL under a wind speed of 14-18 km/h. This experiment aimed to detect a moving, partially occluded quadruped robot in a forest environment. In AM mode, the system successfully detected not only the robot but also unoccluded people nearby. See Supplementary Movie S1.

The technical validation experiments described above were conducted under relatively controlled conditions: targets exhibited minimal camouflage and occlusion, and wind speeds were low. Because the approximate locations of the targets were known, sample video data could be recorded and processed offline to demonstrate and validate the individual processing steps. In contrast, the following sections present results from realworld field campaigns for wildlife monitoring and surveillance under more challenging conditions, featuring heavily camouflaged and hidden targets at unknown locations, with wind speeds reaching up to 40km/h. Under these operational conditions, all computations were executed live on the drone’s onboard hardware.

## Field Experiment 1: Wildlife Monitoring

Wildlife monitoring field experiments were conducted from August 28 to 31, 2026, at the Klepelshagen estate in the Uckermark region of Germany. Spanning approximately 2,500 hectares, the area is operated by the German Wildlife Foundation. Adequate monitoring locations were established by experienced personnel. The drone was manually flown to these designated waypoints to comply with visual line-of-sight regulations.

![](images/594f0e4d6118938f72a26c5e0298f6e9e82d6e20238b448e86b3ab8cd0f4d868.jpg)  
Fig. 5. Wildlife monitoring. Sample verification images and mapped locations of various species detected during field experiments at the Klepelshagen estate and the surrounding nature park “Am Stettiner Haff”. Detections were performed both manually and automatically for single and multiple targets. See Supplementary Movies S2-S4.

A total of eight flights were conducted over a four-day experimental period (4h 26min total flight time). Flight operations spanned from dusk, through daytime, to dawn, at altitudes ranging from 60 to 120m AGL (ground resolution per pixel 8cm - 16cm) and in wind speeds between 10 and 40km/h. Wide-angle imaging was performed exclusively using the drone's thermal camera, as motion signals appear stronger in the thermal spectrum than in RGB imaging – even amid severe environmental thermal clutter. Subsequent optical verification was conducted using the drone's zoom camera (ranging from 2x to 70x magnification, depending on the flight altitude and target species) in either daylight or night-vision mode, which the camera selects automatically based on ambient light levels.

Various species were detected, ranging from small mammals, such as European hares (Lepus europaeus), invasive American mink (Neogale vison), and European badgers (Meles meles), to larger animals including roe deer (Capreolus capreolus), red deer (stags and hinds of Cervus elaphus), wild boar (Sus scrofa), and cattle (Bos taurus). Most of these animals were well camouflaged and/or occluded by vegetation. Figure 5 presents example verification images captured by the drone, with their corresponding detections automatically mapped. Supplementary Movie S2 demonstrates manual detection (manual mode, M). Supplementary Movie S3 illustrates automated single-target detection and verification (automatic single-target mode, AS). Supplementary Movie S4 illustrates automated multitarget detection and verification (automatic multi-target mode, AM).

In all modes, trACT can detect false positives if the motion detection fails. An offline classification (Fig. 2, step 6) is applied to automatically filter out false positives (i.e. verification images that do not show animals) and to classify the detected animal. Rather than relying on conventional pattern-matching approaches (64-77) – such as wildlife-finetuned classifiers (e.g., You Only Look Once, YOLO (78)) or dual encoders for image-text similarity (e.g., Contrastive Language-Image Pre-training, CLIP (79)) – we leverage the cognitive interpretation capabilities of advanced vision reasoning models. See Materials and Methods for details.

Throughout all field experiment flights, trACT captured a total of 1,276 verification images, 579 of which contained animals (recall: 46.0%). Following automatic outlier removal with vision reasoning (i.e., filtering out images without animals), the recall increased to 74.8% (precision: 99.8%, F1 score: 85.5%, accuracy: 88.3%). Furthermore, the species of the detected animals were classified with a recall of 87.6% (precision: 98.7%, F1 score: 92.8%, accuracy: 86.6%). However, a significant performance discrepancy in automatic classification was observed between daylight and night-vision images. Night-vision image classification performed worse than daylight image classification. Note that the metrics above combine daylight and night-vision verification data across all classified species. Statistics for individual flights and species are provided in the Supplementary Material (Supplementary Tables S1 and S2).

## Field Experiment 2: Surveillance

Surveillance field experiments were conducted on September 2, 2026, at the Ruhleben Fighting City facility in Berlin – a forested training area spanning approximately 25 hectares used primarily by the Berlin Police. Volunteers stood and walked beneath the forest canopy or in open areas. To comply with visual line-of-sight regulations, the drone was manually piloted to the designated waypoints.

![](images/d1179aaa8db467c30d08104bf8eb2e34ea8f33118b02ff843bfaa5d1081ea2f9.jpg)  
Fig. 6. Surveillance. Examples of walking and standing individuals detected under partial occlusion using motion signals. (A) A standing person revealed only by slight body movements (manual detection), with wide-angle RGB and thermal images (left and center) and the verification image (right). (B) Automatically detected walking people under varying degrees of occlusion, displaying the motion signals (���<sub>�</sub> channel) and verification images. Note that the numbers indicate the size (in pixels) of the detected motion trails. (C) Automatic detection of a walking person where the thermal signal is lost in background clutter, showing the wide-angle thermal image, $T M P _ { v }$ channel, and RGB verification image. See Supplementary Movie S5.

Three flights took place between 11:00 and 17:00 local time, at altitudes ranging from 85m to 120m AGL under wind speeds between 4km/h and 25km/h. Wide-angle imaging was performed exclusively using the drone's thermal camera, resulting in typical ground resolutions from 11cm to 16cm. Subsequent optical verification in daylight mode utilized the zoom camera at magnifications from 2x to 20x, depending on flight altitude. See Supplementary Movie S5.

Across all flights, trACT captured 413 verification images, of which 185 contained people (yielding a recall of 44.8%). After automatic outlier removal using vision reasoning (specifically filtering out images without people) recall increased to 83.8%, achieving 100% precision, a 91.2% F1 score, and 92.7% accuracy. The mapping result of these experiments is shown in Supplementary Figure S1.

## DISCUSSION

This work introduces trACT, a lightweight bio-inspired robotics framework for consumer drones that bridges wide-area aerial scanning with precise, autonomous target verification. Its key contributions include Temporal Max Pooling (TMP) for motion trail extraction, selfsupervised motion anomaly detection via Noise-Contrastive Estimation (NCE) to filter environmental clutter, Kalman-filtered motion prediction to compensate for system latency, and multi-target balancing for equitable verification coverage. All steps are optimized to achieve real-time detection and verification on resource-constrained edge devices, such as drone remote controllers. Under realistic field conditions during wildlife monitoring and surveillance, trACT achieves a raw recall of approximately 45%, which increases to 80% following vision-reasoning-based outlier removal.

Although TMP remains robust under dense partial occlusion and temperature clutter, optical verification and classification may fail when obstruction is excessive. Although occlusionremoval techniques such as synthetic aperture sensing (21) offer a potential solution, they require wide-area drone maneuvers to sample from multiple perspectives and are restricted to largely static targets. This runs counter to trACT's core principle of detecting moving targets from a hovering position. Consequently, integrating these complementary sampling strategies presents a compelling direction for future research.

We found that TMP alone serves as a promising approach for reconstructing and visualizing the flight trajectories of volant animals, such as insects, bats, and birds, with initial experiments presented in Supplementary Figures S2 and S4, and Supplementary Movies S6-S9, S12, S16. Conventional methods often falter when target detection fails within individual video frames or single difference frames for background subtraction (80,81). In contrast, TMP integrates a broad temporal window of image differences, resulting in distinct and robustly detectable motion trails. Furthermore, it can be readily adapted for static ground-based cameras rather than being restricted to aerial platforms.

During an earlier wildlife conservation field expedition, we evaluated TMP in rainforest environments to detect wild Sumatran rhinos (see Supplementary Figures S3 and S4, and Supplementary Movies S10-S17). Notably, automated features such as motion anomaly detection, motion prediction, multi-target coverage, and classification were not yet implemented at that time, requiring optical verification to be performed via manual drone control. Nevertheless, TMP performed exceptionally well, particularly under conditions of extreme thermal clutter during hot daylight hours, when motion signals are nearly imperceptible in raw thermal imagery.

Employing self-supervised NCE to distinguish between anomalous motion (e.g., animals, humans) and background dynamics (e.g., wind, drone drift) is straightforward and eliminates the need for labeled training data. Furthermore, incorporating greater variability into the training set, such as varying wind speeds, flight altitudes, and target velocities, yields more robust detections. However, because the framework currently performs independent per-pixel evaluation, we believe that leveraging richer spatial and temporal context could further enhance performance. Exploring these spatiotemporal neighborhoods represents a promising direction for future research.

Of the total processing time of 946ms, 800ms is consumed by mechanical gimbal movement for optical verification. This constraint limits the system's ability to verify fast-moving targets, highlighting the need for faster gimbal mechanisms in future developments. Expanded pitch and yaw rotation ranges (e.g., those featured on the DJI Mavic 400) would allow for wider coverage without requiring drone rotation, while also eliminating the need to return the gimbal to the reference pose. Overall, this would result in smaller cumulative gimbal rotations and reduced repositioning times.

The limited flight endurance of multirotor drones remains a significant constraint, as hovering demands substantially more energy than forward flight. Future platforms leveraging the energy-efficient flight strategies of birds (82) could provide viable alternatives capable of stable, kestrel-like hovering. Finally, trACT is not restricted to aerial platforms, but can be readily adapted to terrestrial robotic systems or stationary mastmounted cameras.

Temporal Max Pooling

![](images/24bad26ba717dba679049d876351c3232f9f4cec947eb5ce2f99e3a223648907.jpg)  
Fig. 7. Temporal Max Pooling. Pairwise differences of images within a fixed integration window are stacked, and the maximum values are pooled from the stack for each pixel. After normalization, this yields amplified motion trail values (A). Embedding the relative time at which the maximum value is pooled per pixel produces a time encoding (B). These two channels, $T M P _ { v }$ and $T M P _ { t }$ , form our signal basis for training and inference.

Let the rolling integration window (Fig. 7A) consist of a sequence of � gray scale video frames (color frames are converted to grayscale) denoted as $\{ I _ { 1 } , I _ { 2 } , \dots , I _ { N } \}$ , where each frame $I _ { k }$ consist pixels at spatial coordinates $( x , y )$ . First, the pairwise absolute differences between consecutive frames are computed as:

$$
D _ { k } ( x , y ) = | I _ { k + 1 } ( x , y ) - I _ { k } ( x , y ) | \quad f o r k = 1 , 2 , \ldots , N - 1 .
$$

Next, the stack of difference images is projected into a single image by pooling the maximum value at each pixel coordinate across time:

$$
T M P ( x , y ) = m \underset { k } { a x } D _ { k } ( x , y ) .
$$

Finally, the projected image is scaled to the range [0,1] using global min-max normalization:

$$
T M P _ { v } ( x , y ) = \frac { T M P ( x , y ) - m i n ( T M P ) } { m a x ( T M P ) - m i n ( T M P ) } ,
$$

which emphasizes the value $( T M P _ { v } )$ of the strongest motion signal within the integration window.

Alongside the magnitude of the strongest motion signal, its relative timing is also determined (Fig. 7B). Let $T S { = } \{ 1 , 2 , \ldots , N \}$ represent a sequence of time steps corresponding to the order of recorded video frames within a rolling integration window. The temporal

encoding $( T M P _ { t } )$ is derived by projecting the max-pooled motion signals as explained above, assigning their respective pooling time steps, and normalizing the result to the range [0,1]:

$$
\begin{array} { r } { T M P _ { t } ( x , y ) = \frac { T S ( T M P _ { v } ( x , y ) ) } { N } , } \end{array}
$$

where $T S ( T M P _ { v } )$ denotes the time step corresponding to the max-pooled value $T M P _ { v }$ Consequently, the newest position in a motion trail corresponds to $T M P _ { t } { = } 1$ , and the oldest corresponds to $T M P _ { t } { = } 0$

In addition to the rolling integration window size �, a frame-skipping interval � (where � frames are skipped before the next frame is ingested into the window) can be defined. Assuming a constant camera frame rate � (in Hz) the total integrated time window � (in seconds) is given by:

$$
T = \frac { ( N - 1 ) \cdot ( S + 1 ) } { F } .
$$

The effective sampling rate, � (in Hertz, Hz), of the resulting motion integral is:

$$
R = \frac { F } { S + 1 } .
$$

For a constant video frame rate �, slower motion dynamics can be captured as elongated motion trails by increasing the integration window size � and/or the frame-skipping interval �. While increasing � reduces the effective sampling rate, it improves overall processing performance and significantly mitigates memory constraints associated with storing large integration windows.

Due to hardware limitations in memory and camera frame rate, we used $N { = } 1 0$ and S=0 for all experiments. For better visibility under direct sunlight, the live visualizations of $T M P _ { v }$ on the drone’s RC were inverted: dark regions represent high motion values, while bright regions indicate low motion.

The two signal channels $( T M P _ { v }$ and $T M P _ { t } )$ are the basis for all following training, inference, and estimation steps.

## Motion Anomaly Detection

To enable self-supervised learning in the absence of ground-truth motion labels, we adopted the Noise-Contrastive Estimation (NCE) strategy described in (63). NCE formulates the learning problem as a binary classification task that discriminates between observed data and artificially generated samples drawn from predefined noise distributions. The distribution of the noise must only be substantially different from the distribution of the observed data. Following this principle, we constructed pseudo-labeled training data by independently modeling the two components of each TMP channels, $T M P _ { v }$ and $T M P _ { t }$ . As illustrated in Fig. 8, the resulting distributions of anomalous (animals) and normal (wind) motion pixels exhibit substantially different statistical characteristics, providing a basis for distinguishing the two classes in the TMP space. Specifically, the $T M P _ { v }$ was modeled using a Gaussian distribution, whereas the $T M P _ { t }$ was modeled using a uniform distribution. The resulting samples were assigned binary labels, with 1 denoting anomalous motion (e.g., an animal or person motion) and 0 denoting normal motion (e.g., caused by wind or drone drift). This formulation enables the motion-detection model to be trained without requiring manually annotated data.

![](images/ebef49f01fc5a7e77cf76a8cea86b12dee83657d75c151f0f54d62ec9d718d6c.jpg)  
Fig. 8. TMP distributions of anomalous motion vs normal motion. Histograms showing pixel counts (y-axis) across TMP channels $T M P _ { v }$ and $T M P _ { t }$ (x-axes) for anomalous animal motion and normal wind-induced motion. Similar general distributions are observed across both high (approx. 40km/h) and low (approx. 20km/h) wind speeds, as well as between slow (standing with minor body movements) and fast (walking) animal motions. A total of 684 selected TMP images were aggregated to compute the histograms. Note that the $T M P _ { t }$ channel is discretized to 9 relative time steps (corresponding to 9 difference images $D _ { k } )$ with $N { = } 1 0$ (see Temporal Max Pooling).

Based on these pseudo-labeled samples, we employed a three-layer multilayer perceptron (MLP) to learn the distinction between normal and anomalous motion in the TMP space. Each input sample consists of the $T M P _ { v }$ and $T M P _ { t }$ . The MLP learns a nonlinear decision boundary between the two classes and classifies each TMP according to its predicted probability of representing anomalous motion. The network was optimized using binary cross-entropy (BCE) loss.

To improve the representation of the TMP inputs, we incorporated the positional encoding function introduced in Neural Radiance Fields, NeRF (81). Positional encoding maps the input into a higher-dimensional embedding, enabling the MLP to represent high-frequency variations more effectively. The encoding function is defined as

$$
\gamma ( p ) = ( s i n ( 2 ^ { 0 } \pi p ) , c o s ( 2 ^ { 0 } \pi p ) , \ldots , s i n ( 2 ^ { L - 1 } \pi p ) , c o s ( 2 ^ { L - 1 } \pi p ) ) ,
$$

where � denotes an input component and � is the number of frequency levels. The encoding function was applied independently to the two components of each ���, i.e., the $T M P _ { v }$ and $T M P _ { t }$ , with �=16.

For inference, we precomputed a lookup table containing the predicted class for every possible combination of $T M P _ { v }$ and $T M P _ { t }$ . Specifically, the $T M P _ { v }$ was discretized in 255 value steps, while the $T M P _ { t }$ was discretized in 9 time steps (assuming $N { = } 1 0 )$ . This lookuptable eliminates the need for repeated forward passes and therefore enables real-time motion detection with computational requirements suitable for deployment on the drone's RC.

The MLP generates a binary prediction mask for all pixels in the input image. As a postprocessing step, connected components below a predefined minimum motion trail size are removed to suppress isolated detections and small spurious responses. The minimum motion trail size is an adjustable parameter that can be configured through the drone's RC. It depends on flight altitude and target size. For each retained motion trail segment, the pixel exhibiting the maximum $T M P _ { t }$ is identified, and its image coordinates are used to estimate the latest position of the corresponding motion.

Training and inference pipelines were implemented in Python 3.12.8 using PyTorch 2.8.0 and trained on an NVIDIA RTX 5090 GPU. Following validation of the Python implementation, the inference pipeline was reimplemented in Java to ensure compatibility with the drone's RC platform and facilitate real-time deployment.

## Motion Prediction

The $T M P _ { t }$ channel contains the relative time steps of motion appearance per pixel. The coordinates and stored time steps of all pixels belonging to a segmented motion trail are passed to a Kalman filter (84). Combined with the average processing time (continuously measured and updated from previous iterations), the Kalman filter predicts the pixel coordinates of the motion trail for the next iteration. This coordinate is then used for centering and zoom verification. We use the KalmanFilter class of the OpenCV Library (85) with a constant-velocity state model ${ \boldsymbol { s } } = ( x , y , v _ { x } , v _ { y } ) ^ { \mathrm { T } }$ . Using the coordinates $( x , y )$ of all the trail pixels and their times, converted into seconds,

$$
t ( x , y ) = N \cdot T M P _ { t } ( x , y ) / R
$$

derived from the $T M P _ { t }$ channel, where � is the rolling integration window size and � is the effective sampling rate (see Temporal Max Pooling), the filter estimates the velocity $( v _ { x } , v _ { y } )$ of the moving target. The trail pixels are grouped by their time $t ( x , y )$ and processed in temporal order. The state is initialized with the mean position of the trail pixels with the oldest $T M P _ { t } ,$ i.e., at $T M P _ { t } { = } 0$ and zero velocity. For each group, the filter predicts the state forward and then corrects it using the positions of the pixels from that time step together. The pixel coordinates of the motion trail for the next iteration are predicted by extrapolating the estimated latest position of the corresponding motion with the estimated velocity over the average processing time.

## Optical Verification and Mapping

The predicted pixel coordinates of the motion trail are mapped to the gimbal's pitch and yaw angles to center the target on the ground after rotation. Once centered, a verification image is captured (daylight or night-vision RGB) at a pre-selected zoom level using the zoom camera. Simultaneously, a laser rangefinder measures the target's distance and calculates its ground GPS coordinates. All verification images, along with the corresponding GPS and drone coordinates, target distances, and timestamps, are stored on the remote controller. After landing, the data is transferred to a PC, where the verification images are classified and outliers are filtered out. The remaining data is then visualized in our custom, browserbased mapping application powered by OpenStreetMap.

## Classification

Rather than relying on conventional pattern-matching approaches (86-91), such as finetuned object detectors and classifiers for wildlife or surveillance applications (e.g., YOLO (78)) or dual-encoder models for image–text similarity (e.g., CLIP (79)), we leverage the emerging capabilities of vision-reasoning models for semantic interpretation and contextual understanding. Recent studies have demonstrated the use of vision-reasoning models for anomaly understanding in surveillance videos (92-95), wildlife species identification (96), and semantic retrieval of camera-trap imagery (97).

In our trACT processing framework, we employ Qwen3-VL-30B-A3B-Instruct (98) as a label-free semantic verification stage. Unlike conventional classifiers that generally require task-specific training or annotated data, the model can perform zero-shot classification through in-context prompting, allowing it to identify images containing animals and determine their corresponding species without additional fine-tuning. We therefore use the model as a post-processing step (cf. Fig. 2, step 6) to automatically filter and clean the verification dataset, retaining images that contain animals together with their predicted species (field experiment 1), as well as images containing people (field experiment 2). This process reduces the need for manual inspection and provides semantically verified samples for subsequent analysis.

For detection and species identification in our wildlife monitoring experiments, we use the following in-context prompt:

prompt = (   
"Look at this image carefully. Does it contain any animal? "   
"Do not show your reasoning or thinking process. "   
"Respond with ONLY the following two lines, nothing else:\n"   
"Line 1: YES or NO\n"   
"Line 2: the animal species"   
)

For filtering images containing people in our surveillance experiments, we use a separate prompt:

prompt = (

"Look at this image carefully. Does it contain any people? "

"Do not show your reasoning or thinking process. "

"Respond with ONLY the following line, nothing else:\n"

"Line 1: YES or NO\n"

Only verification images with a positive response $\textstyle { \binom { 6 6 } { 5 } } \mathrm { Y e s } ^ { 9 } )$ on line 1 are retained. For wildlife, the species indicated on line 2 is stored alongside the image.

## Multi-Target Verification

We define a covermap as a list of previously verified circular regions within the reference image. Each region (�) is represented by its center coordinates $( m _ { x } , m _ { y } )$ and a counter $( m _ { c } )$ indicating how many times the region has been previously verified. The process begins with an empty covermap list. For each imaging iteration that results in a new TMP image and the corresponding list of (�) detected motion trails (�) at coordinates $( t _ { x } , t _ { y } )$ and priority $t _ { p } = ( 1 . . n )$ , the following steps are executed:

1. priority order the list of detected motion trails

2. for each motion trail �, determine how often its region has been verified previously:

a. determine closest existing verification region (�) in the covermap b. if distance between $( m _ { x } , m _ { y } )$ and $( t _ { x } , t _ { y } ) < \mathrm { R A D I U S }$ then return $m _ { c } .$ , if outside any previously visited verification region the return 0

3. select motion trail(s) with lowest number of verifications, and if multiple exist with the same low number of verifications, select the one among them with the highest $t _ { p }$

4. verify the selected motion trail at $( t _ { x } , t _ { y } )$

5. if the verified motion trail is outside any verification region in the covermap, create a new entry

$( m _ { x } , m _ { y } ) { = } ( t _ { x } , t _ { y } ) , m _ { c } { = } n$ for the closest verification region.

6. reduce $m _ { c }$ of all � by 1, and remove those entries from the covermap with $m _ { c } { = } 0$

A motion trail takes higher priority than another if its segmented size (post-detection) is larger. Note, that RADIUS can be adjusted. It can be derived from the verification camera's ground coverage, which depends on the field of view, zoom level, and flight altitude.

## Drone Implementation

Our experimental results were recorded using a commercial DJI Matrice 30T drone (see Data Acquisition for details). trACT is deployed within a custom Android application compiled for the ARM64-based DJI RC Plus remote controller. The RC is powered by a Snapdragon 865 SoC (featuring a 1+3+4 octa-core configuration: one prime, three performance, and four efficiency cores) and runs a 64-bit Android 10 operating system. Developed using the DJI SDK (v5.18), our custom application processes live video streams received directly from the aircraft at a frame rate of F=30Hz. Custom controller mappings allow the user to toggle between the different modes (F, M, AS, AM), switch cameras, adjusting zoom level, choosing imaging settings (exposure, daylight and night vision), and adjust parameters (N (integration window size) and S (frame-skipping interval) for TMP, and R (RADIUS) and B (minimum motion trail size) for covermap).

## Data Acquisition

A DJI Matrice 30T drone was used for all experiments. It is equipped with a thermal camera (uncooled VOx microbolometer with a $6 1 ^ { \circ }$ diagonal field of view, 9.1 mm focal length, f/1.0, and a resolution of 640×512 pixels at 30 Hz), a wide-angle camera (1/2-inch CMOS, $8 4 ^ { \circ }$ diagonal field of view, 4.5 mm focal length, f/2.8, and a 12 MP resolution at 30 Hz), a zoom camera (1/2-inch CMOS, 21-75 mm focal length, f/2.8-f/4.2, and a 48 MP resolution at 30Hz), and a laser rangefinder with a range of 3m to 1,200m (0.5×12m vertical area at 20% remission). The gimbal features an angular vibration range of $\pm 0 . 0 1 ^ { \circ }$ and a controllable range of $\pm 9 0 ^ { \circ }$ (pitch) and $- 1 2 0 ^ { \circ } \mathrm { t o } + 4 5 ^ { \circ }$ (yaw). The drone was manually flown to designated waypoints to comply with visual line-of-sight regulations.

Verification images and laser-ranged GPS coordinates were stored on the remote controller and subsequently transferred to our custom desktop mapping software post-flight. Classification (including automatic outlier removal and species identification) was also performed using our custom post-flight desktop software.

## Supplementary Materials:

Fig. S1. Mapping of surveillance experiments.

Fig. S2. Temporal Max Pooling for volant animals.

Fig. S3. Detecting a semi-wild Sumatran rhino (Diceorhinus sumatrensis) in thermal clutter during daylight.

Fig. S4. Additional TMP results obtained in a tropical rainforest ecosystem.

Tab. S1. Automatic outlier removal performance.

Tab. S2. Automatic species classification.

Movie S1. Airborne camera trap principle.

Movie S2. Wildlife monitoring with manual detection.

Movie S3. Wildlife monitoring with automated single-target detection and verification.

Movie S4. Wildlife monitoring with multi-target detection and verification.   
Movie S5. Surveillance examples.   
Software S1. trACT software for DJI.   
Software S2. trACT software for Windows.   
Data S1. Data for wildlife monitoring field experiments (Fig. 5).   
Data S2. Data for surveillance field experiments (Fig. 6).

## References:

1. Z. Zhang, Z. He, G. Cao, W. Cao, Animal detection from highly cluttered natural scenes using spatiotemporal object region proposals and patch verification. IEEE Trans. Multimedia 18, 2079-2092 (2016). https://doi.org/10.1109/TMM.2016.2594138.

2. M. Riechmann, R. Gardiner, K. Waddington, R. Rueger, F. Fol Leymarie, S. Rueger, Motion vectors and deep neural networks for video camera traps. Ecol. Inform. 69, 101657 (2022). https://doi.org/10.1016/j.ecoinf.2022.101657.

3. M. Klasen, V. Steinhage, Wildlife 3D multi-object tracking. Ecol. Inform. 71, 101790 (2022). https://doi.org/10.1016/j.ecoinf.2022.101790.

4. N. Aliane, Drones and AI-driven solutions for wildlife monitoring. Drones 9, 455 (2025). https://doi.org/10.3390/drones9070455.

5. U. P. S. Lundquist, S. Afridi, C. Berthelot, N. Ngoc Dat, K. Hlebowicz, E. Iannino, L. Laporte-Devylder, G. Maalouf, G. May, K. Meier, C. A. Molina Catricheo, E. G. A. Rolland, C. Rondeau Saint-Jean, V. Shukla, T. Burghardt, A. L. Christensen, B. R. Costelloe, M. Damen, A. Flack, K. Jensen, H. S. Midtiby, M. Mirmehdi, F. Remondino, T. Richardson, B. Risse, D. Tuia, M. Wahlberg, D. Cawthorne, S. Bullock, W. Njoroge, S. Mutisya, M. Watson, E. Pastucha, WildDrone: Autonomous drone technology for monitoring wildlife populations. Front. Robot. AI 12, 1695319 (2026).https://doi.org/10.3389/frobt.2025.1695319

6. J. C. Hodgson, S. M. Baylis, R. Mott, A. Herrod, R. H. Clarke, Precision wildlife monitoring using unmanned aerial vehicles. Sci. Rep. 6, 22574 (2016).

7. T. Petso, R. S. Jamisola, Wildlife conservation using drones and artificial intelligence in Africa. Sci. Robot. 8, eadm7008 (2023). https://doi.org/10.1126/scirobotics.adm7008.

8. R. Nath Tripathi, M. G. Ghazi, R. Badola, S. A. Hussain, Feasibility study of UAV based ecological monitoring and habitat assessment of cervids in the floating meadows of Keibul Lamjao National Park in Manipur, India. Measurement 229, 114411 (2024). https://doi.org/10.1016/j.measurement.2024.114411.

9. E. J. Pinel-Ramos, F. Aureli, S. Wich, F. Rodrigues de Melo, C. Rezende, F. Brandão, F. C. S. Alves de Melo, D. Spaan, An assessment of the effectiveness of RGB-camera drones to monitor arboreal mammals in tropical forests. Drones 9, 622 (2025). https://doi.org/10.3390/drones9090622.

10. H. L. Larsen, K. Møller-Lassesen, E. M. E. Enevoldsen, S. B. Madsen, M. T. Obsen, P. Povlsen, D. Bruhn, C. Pertoldi, S. Pagh, Drone with mounted thermal infrared cameras for monitoring terrestrial mammals. Drones 7, 680 (2023). https://doi.org/10.3390/drones7110680.

11. B. Wagner, S. W. Garnick, M. F. Ryan, J. L. Isaac, A. Begg, C. R. Nitschke, Thermal drone surveys to detect arboreal fauna: Improving population estimates and threatened species monitoring. Ecol. Appl. 35, e70091 (2025). https://doi.org/10.1002/eap.70091.

12. L. Ortenzi, J. Aguzzi, C. Costa, S. Marini, D. D'Agostino, L. Thomsen, F. C. De Leo, P. V. Correa, D. Chatzievangelou, Automated species classification and counting by deep-sea mobile crawler platforms using YOLO. Ecol. Inform. 82, 102788 (2024). https://doi.org/10.1016/j.ecoinf.2024.102788.

13. P. Gao, D. Zhong, Q. Qi, C. Ling, C. Qiu, B. Wang, X. Du, M. Gao, FDM-YOLO: Real-time small-target UAV wildlife detection via attention-guided cross-modality fusion. Ecol. Inform. 95, 103697 (2026). https://doi.org/10.1016/j.ecoinf.2026.103697.

14. S. Alphonse, S. B. Prathiba, A. Sharma, YOLOv11-Lite architecture for wildlife detection from drone images. Front. Artif. Intell. 9, 1777913 (2026). https://doi.org/10.3389/frai.2026.1777913.

15. A. Elkholy, A. M. Sarhan, A. E. Hassanien, WL-YOLOv11: Enhanced YOLOv11 for wildlife detection and counting in complex environments. Neural Comput. Appl. 38, 85 (2026).

16. C. Liu, P. Wang, Y. Gong, A. Cheng, YOLO-WL: A lightweight and efficient framework for UAV-based wildlife detection. Sensors 26, 790 (2026). https://doi.org/10.3390/s26030790.

17. E. Corcoran, M. Anschau, A. Sudholz, G. Hamilton, Automated detection of wildlife using drones: Synthesis, opportunities and constraints. Methods Ecol. Evol. 12, 1103-1114 (2021). https://doi.org/10.1111/2041-210X.13581.

18. R. N. Tripathi, K. Agarwal, V. Tripathi, R. Badola, S. A. Hussain, Conservation in action: Cost-effective UAVs and real-time detection of the globally threatened swamp deer (Rucervus duvaucelii). Ecol. Inform. 85, 102913 (2025). https://doi.org/10.1016/j.ecoinf.2024.102913

19. B. S. Krishnan, L. R. Jones, J. A. Elmore, S. Samiappan, K. O. Evans, M. B. Pfeiffer, B. F. Blackwell, R. B. Iglay, Fusion of visible and thermal images improves automated detection and classification of animals for drone surveys. Sci. Rep. 13, 10385 (2023).

20. D. C. Schedl, I. Kurmi, O. Bimber, Airborne optical sectioning for nesting observation. Sci. Rep. 10, 7254 (2020).

21. I. Kurmi, D. C. Schedl, O. Bimber, Airborne optical sectioning. J. Imaging 4, 102 (2018).

22. E. Bondi, F. Fang, M. Hamilton, D. Kar, D. Dmello, J. Choi, R. Hannaford, A. Iyer, L. Joppa, M. Tambe, R. Nevatia, SPOT poachers in action: Augmenting conservation drones with automatic detection in near real time, in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI Press, 2018), pp. 7741-7746.

23. M. A. Olivares-Mendez, C. Fu, P. Ludivig, T. F. Bissyandé, S. Kannan, M. Zurad, A. Annaiyan, H. Voos, P. Campoy, Towards an autonomous vision-based unmanned aerial system against wildlife poachers. Sensors 15, 31362-31391 (2015). https://doi.org/10.3390/s151229861.

24. S. G. Penny, R. L. White, D. M. Scott, L. MacTavish, A. P. Pernetta, Using drones and sirens to elicit avoidance behaviour in white rhinoceros as an anti-poaching tactic. Proc. R. Soc. London Ser. B 286, 20191135 (2019).

25. A. Tazeem, A. Tewari, M. Ahmed, D. Sharma, Motion detection using mixture of Gaussians for wildlife photography, in Proceedings of the 2025 8th International Conference on Computing Methodologies and Communication (ICCMC) (IEEE, 2025), pp. 1810-1815 . https://doi.org/10.1109/ICCMC65190.2025.11140724.

26. B. Maslen, G. Popovic, D. Wang, A. Jansen, D. Warton, The motion picture: Leveraging movement to enhance AI object detection in ecology. Ecol. Evol. 15, e71996 (2025).

27. S.-P. Yong, A. L. W. Chung, W. K. Yeap, P. Sallis, Motion detection using drone's vision, in Proceedings of the 2017 Asia Modelling Symposium (AMS) (IEEE, 2017), pp. 108-112. https://doi.org/10.1109/AMS.2017.25.

28. A. Nakajima, H. Oku, K. Motegi, Y. Shiraishi, Wild animal recognition method for videos using a combination of deep learning and motion detection. Jpn. J. Inst. Ind. Appl. Eng. 9, 38-45 (2021). https://doi.org/10.12792/jjiiae.9.1.38.

29. K. Nguyen, F. Liu, C. Fookes, S. Sridharan, X. Liu, A. Ross, Person recognition in aerial surveillance: A decade survey. IEEE Trans. Biom. Behav. Identity Sci. 8, 3- 19 (2026).

30. L. Xu, Y. Zhao, Y. Zhai, L. Huang, C. Ruan, Small object detection in UAV images based on YOLOv8n. Int. J. Comput. Intell. Syst. 17, 223 (2024).

31. X. Cao, H. Wang, X. Wang, B. Hu, DFS-DETR: Detailed-feature-sensitive detector for small object detection in aerial images using transformer. Electronics 13, 3404 (2024).

32. J. Zhao, W. Lv, Z. Li, W. Zhang, H. Wang, F. Shuang, ARCAF-YOLO: Eventbased multimodal algorithm for small object detection in aerial images. Signal Image Video Process. 20, 283 (2026).

33. X. Fan, G. Wen, Z. Gao, J. Chen, H. Jian, An unsupervised moving object detection network for UAV videos. Drones 9, 150 (2025).

34. L. Wang, F. Zhang, Decoupling ego-motion from target dynamics via dual-interval motion cues for UAV detection. https://arxiv.org/abs/2605.22605 (2026).

35. U. Verma, M. M. M. Pai, R. M. Pai, Contextual information based anomaly detection for multi-scene aerial videos. Sci. Rep. 15, 25805 (2025).

36. J. Yang, D. Wang, H. Yin, H. Li, J. Yu, UAV-DETR: DETR for anti-drone target detection. https://arxiv.org/abs/2603.22841 (2026).

37. P. Chen, H. Sa, Y. Hu, Y. Cheng, J. Wang, SDD-YOLO: A small-target detection framework for ground-to-air anti-UAV surveillance with edge-efficient deployment. https://arxiv.org/abs/2603.25218 (2026).

38. N. Alshaer, R. Abdelfatah, T. Ismail, H. Mahmoud, Vision-based UAV detection and tracking using deep learning and Kalman filter. Comput. Intell. 41, e70026 (2025).

39. W. Wang, J. Fu, J. Song, K. Li, H. Qiao, J. Liu, H. Sun, X. Cao, Dist-tracker: A small object-aware detector and tracker for UAV tracking, in Proceedings of the 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (IEEE, 2025), pp. 6603-6611.

40. A. Mahalanobis, A. Tall, Efficient drone detection using temporal anomalies and small spatio-temporal networks. Sensors 26, 170 (2026).

41. J. Ye, C. Kyrkou, A review of vision-based vehicle detection for UAV-based traffic monitoring: Experimental insights and future directions. IEEE Trans. Intell. Transp. Syst. 27, 10514-10532 (2026).

42. A. Khanpour, T. Wang, A. Vahidi-Shams, W. Ectors, F. Nakhaie, A. Taheri, C. Claudel, UAV-based intelligent traffic surveillance system: Real-time vehicle detection, classification, tracking, and behavioral analysis. https://arxiv.org/abs/2509.04624 (2025).

43. M. Worachairungreung, N. Kulpanich, P. Sae-Ngow, K. Thanakunwutthirot, S. Doddacha, J. Nilnarong, S. Ninsawat, AI-assisted vehicle detection and traffic flow analysis from CCTV imagery using YOLO in the Bangkok metropolitan area. Transp. Res. Interdiscip. Perspect. 38, 102104 (2026).

44. T. Li, Z. Bian, H. Lei, F. Zuo, Y.-T. Yang, Q. Zhu, Z. Li, K. Ozbay, Multi-level traffic-responsive tilt camera surveillance through predictive correlated online learning. Transp. Res. Part C Emerg. Technol. 167, 104804 (2024).

45. S. Cheng, Y. Zhu, S. Wu, Deep learning based efficient ship detection from dronecaptured images for maritime surveillance. Ocean Eng. 285, 115440 (2023).

46. Y. Huang, Y. Zheng, P. Wu, Y. Zhang, J. Liu, Y. Liu, Multi-scale oriented detection with shared convolution for UAV-enabled maritime safety surveillance. Research 8, 0920 (2025).

47. S. Ahmed, O. Pizarro, Maritime small object detection from UAVs using deep learning with altitude-aware dynamic tiling, in Proceedings of OCEANS 2025 Brest (IEEE, 2025), pp. 1-9.

48. A. B. Kulas, F. Petric, S. Bogdan, Aerial maritime vessel detection and identification, in Proceedings of the 2025 International Conference on Unmanned Aircraft Systems (ICUAS) (IEEE, 2025), pp. 44-51.

49. N. Lopac, K. Severinski, N. Grubišić, J. Lerga, YOLO-driven vessel detection and multi-object tracking in fixed-camera marina environments. Technologies 14, 482 (2026).

50. X. Han, Q. Han, Y. Fan, D. Mu, MAAT: A marine-aware adaptive tracker for robust and real-time multi-object tracking in maritime environments. J. Mar. Sci. Eng. 14, 738 (2026).

51. K. R. Akshatha, A. Kotegar Karunakar, S. B. Shenoy, A. K. Pai, N. Hunjanal Nagaraj, S. S. Rohatgi, Human detection in aerial thermal images using Faster R-CNN and SSD algorithms. Electronics 11, 1151 (2022).

52. K. Taneja, S. Dwivedi, S. Gupta, V. Raj, YOLO-based real-time border security surveillance system using deep learning, in Proceedings of the 2025 International Conference on Next Generation Information System Engineering (NGISE) (IEEE, 2025), vol. 1, pp. 1-6.

53. S. Pitafi, T. Anwar, I. D. M. Widia, Z. Sharif, B. Yimwadsana, A novel machine learning model for perimeter intrusion detection using intrusion image dataset. PLOS ONE 19, e0313890 (2024).

54. N. Thai-Nghe, H.-H. Nguyen, W. Park, Q. T. Ngo, M. T. Truong, Human intrusion detection for security cameras using YOLOv8, in Proceedings of the International Conference on Intelligent Systems and Data Science (Springer Nature, Singapore, 2023), pp. 220-227.

55. D. Wu, Y. Tian, F. Gao, X. Wei, C. Wang, Dual weakly supervised anomaly detection and unsupervised segmentation for real-time railway perimeter intrusion monitoring. Sensors 25, 6344 (2025).

56. X. Chen, Y. Tian, M. Li, B. Lv, S. Zhang, Z. Qu, J. Wu, S. Cheng, Automatic detection of foreign object intrusion along railway tracks based on MACENet. PLOS ONE 20, e0329303 (2025).

57. S. Yeom, Thermal image tracking for search and rescue missions with a drone. Drones 8, 53 (2024).

58. P. Fraternali, L. Morandini, R. Motta, Enhancing search and rescue missions with UAV thermal video tracking. Remote Sens. 17, 3032 (2025).

59. Z. Song, Y. Yan, Y. Cao, S. Jin, F. Qi, Z. Li, T. Lei, L. Chen, Y. Jing, J. Xia, X. Liang, An infrared dataset for partially occluded person detection in complex environment for search and rescue. Sci. Data 12, 300 (2025).

60. Y. Qian, S. Ye, C. Wang, X. Cai, J. Qian, J. Wu, UCF-Crime-DVS: A novel eventbased dataset for video anomaly detection with spiking neural networks, in Proceedings of the AAAI Conference on Artificial Intelligence (2025), vol. 39, pp. 6577-6585.

61. C. T. O'Rourke, T. Pitlik, M. Hoover, E. Fernández-Juricic, Hawk eyes II: Diurnal raptors differ in head movement strategies when scanning from perches. PLOS ONE 5, e12169 (2010).

62. S. A. Kane, A. H. Fulton, L. J. Rosenthal, When hawks attack: Animal-borne video studies of goshawk pursuit and prey-evasion strategies. J. Exp. Biol. 218, 212-222 (2015).

63. M. Gutmann, A. Hyvärinen, Noise-contrastive estimation: A new estimation principle for unnormalized statistical models, in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (JMLR Workshop and Conference Proceedings, 2010), pp. 297-304.

64. C. Liu, P. Wang, Y. Gong, A. Cheng, YOLO-WL: A lightweight and efficient framework for UAV-based wildlife detection. Sensors 26, 790 (2026).

65. R. Hu, Y. Chen, K. Xu, L. Zhang, C. Yue, H. Pi, X. Chen, X. Lin, YOLIP: An enhanced framework for UAV-assisted wildlife monitoring based on YOLO integrated with the CLIP model. Sensors 26, 3436 (2026).

66. D. Nguyen Ngoc, T. S. Richardson, M. Watson, K. Meier, J. Kline, S. Reid, G. Maalouf, D. Hine, M. Mirmehdi, T. Burghardt, WildLive: Near real-time visual wildlife tracking onboard UAVs, in 5th Workshop on Computer Vision for Animal Behavior Tracking and Modeling (CV4Animals), IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025).

67. P. Jia, Y. Zhang, A lightweight algorithm for wildlife detection in outdoor environments based on you only look once version 8 network. Eng. Appl. Artif. Intell. 159, 111544 (2025).

68. S. Sastry, S. Khanal, A. Dhakal, A. Ahmad, N. Jacobs, TaxaBind: A unified embedding space for ecological applications, in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (IEEE, 2025), pp. 1765-1774.

69. K. Backman, J. Wood, M. Brandimarti, C. T. Beranek, A. Roff, Human inspired deep learning to locate and classify terrestrial and arboreal animals in thermal drone surveys. Methods Ecol. Evol. 16, 1239-1254 (2025).

70. T. Gadot, Ș. Istrate, H. Kim, D. Morris, S. Beery, T. Birch, J. Ahumada, To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images. IET Comput. Vis. 18, 1193-1208 (2024).

71. M. Mulero-Pázmány, S. Hurtado, C. Barba-González, M. L. Antequera-Gómez, F. Díaz-Ruiz, R. Real, I. Navas-Delgado, J. F. Aldana-Montes, Addressing significant challenges for animal detection in camera trap images: A novel deep learningbased approach. Sci. Rep. 15, 16191 (2025).

72. Z. Du, D. Wu, Q. Wen, F. Xu, Z. Liu, C. Li, R. Luo, An improved lightweight model for protected wildlife detection in camera trap images. Sensors 25, 7331 (2025).

73. G. May, E. Dalsasso, B. Kellenberger, D. Tuia, POLO – Point-based, multi-class animal detection, in Computer Vision – ECCV 2024 Workshops, A. Del Bue, C. Canton, J. Pont-Tuset, T. Tommasi, Eds. (Lecture Notes in Computer Science, Springer, 2025), vol. 15624, pp. 169-177.

74. A. He, X. Li, X. Wu, C. Su, J. Chen, S. Xu, X. Guo, ALSS-YOLO: An adaptive lightweight channel split and shuffling network for TIR wildlife detection in UAV imagery. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. 17, 17308-17326 (2024).

75. S. Stevens, J. Wu, M. J. Thompson, E. G. Campolongo, C. H. Song, D. E. Carlyn, L. Dong, W. M. Dahdul, C. Stewart, T. Berger-Wolf, W.-L. Chao, Y. Su, BioCLIP: A vision foundation model for the tree of life, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE, 2024), pp. 19412-19424.

76. V. Gabeff, M. Rußwurm, D. Tuia, A. Mathis, WildCLIP: Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models. Int. J. Comput. Vis. 132, 3770-3786 (2024).

77. Q. Ye, M. Ma, X. Zhao, B. Duan, L. Wang, D. Ma, ADD-YOLO: An algorithm for detecting animals in outdoor environments based on unmanned aerial imagery. Measurement 242, 116019 (2025).

78. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You only look once: Unified, realtime object detection, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE, 2016), pp. 779-788.

79. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in Proceedings of the 38th International Conference on Machine Learning (ICML) (PMLR, 2021), vol. 139, pp. 8748-8763.

80. Y. M. N. Ratnayake, D. C. Amarathunga, A. Zaman, A. G. Dyer, A. Dorin, Spatial monitoring and insect behavioural analysis using computer vision for precision pollination. Int. J. Comput. Vis. 131, 591–606 (2023).

81. Z. M. N. Ratnayake, A. N. Toosi, J. Cook, R. Rader, A. Dorin, AGRICAM: A track-mounted crop pollination monitoring robot. https://arxiv.org/abs/2608.29237 (2026).

82. S. L. Jeger, V. Wüest, C. Toumieh, D. Floreano, Adaptive morphing of wing and tail for stable, resilient, and energy-efficient flight of avian-inspired drones. npj Robot. 2, 8 (2024).

83. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, NeRF: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99-106 (2021).

84. R. E. Kalman, A new approach to linear filtering and prediction problems. J. Basic Eng. 82, 35-45 (1960). https://doi.org/10.1115/1.3662552.

85. G. Bradski, The OpenCV library. Dr. Dobb's J. Softw. Tools 25, 120-125 (2000).

86. J. D. Santamaria, C. Isaza, J. H. Giraldo, CATALOG: A camera trap languageguided contrastive learning model, in Proceedings of the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (IEEE, 2025), pp. 1197- 1206.

87. H. Markoff, S. H. Bengtson, M. Ørsted, Vision transformers for zero-shot clustering of animal images: A comparative benchmarking study .https://arxiv.org/abs/2602.03894 (2026).

88. J. D. Santamaria, C. Isaza, J. H. Giraldo, WildIng: A wildlife image invariant representation model for geographical domain shift. Int. J. Comput. Vis. 134, 183 (2026).

89. J. Gu, S. Stevens, E. Campolongo, M. Thompson, N. Zhang, J. Wu, A. Kopanev, Z. Mai, A. White, J. Balhoff, W. Dahdul, BioCLIP 2: Emergent properties from scaling hierarchical contrastive learning. Adv. Neural Inf. Process. Syst. 38, 102778-102811 (2026).

90. V. G. Warnasooriya, A. Hajian, W. Ruangsang, S. Aramvith, Real-time video anomaly detection using YOLO pose estimation and CLIP-based semantic scoring. https://arxiv.org/abs/2608.31074 (2026).

91. E. Dumitru, S. Spînu, A multi-task deep learning framework for real-time intelligent video surveillance with temporal event validation. https://arxiv.org/abs/2607.03131 (2026).

92. C. Huang, B. Wang, W. Wang, J. Wen, C. Liu, L. Shen, X. Cao, VAD-R1: Towards video anomaly reasoning via perception-to-cognition chain-of-thought. Adv. Neural Inf. Process. Syst. 38, 118486-118518 (2026).

93. S. Agarwal, A. Konwer, A. P. Shah, Reasoning-guided grounding: Elevating video anomaly detection through multimodal large language models. https://arxiv.org/abs/2605.02912 (2026).

94. Y. Duan, H. Li, A. Li, S. Feng, L. Kong, N. Liu, J. Zhang, X. Sheng, Y. Du, AgenticVAU: Multi-agent explore-verify reasoning for video anomaly understanding. https://arxiv.org/abs/2608.03779 (2026).

95. S. Yao, A. Danesh Pazho, N. Rashvand, H. Tabkhi, Are multimodal LLMs ready for surveillance? A reality check on zero-shot anomaly detection in the wild. https://arxiv.org/abs/2603.04727 (2026).

96. W. Zhou, M. Siripuram, X. Yan, Z. Liu, Y. Ding, Can edge-deployable visionlanguage models identify species?. https://arxiv.org/html/2609.11916 (2026).

97. V. Gabeff, B. Maquignaz, J. Shan, S. Mamooler, G. Sumbul, B. Costelloe, D. Tuia, A. Mathis, Prompting-MammAlps: Fine-grained text-to-video retrieval for cameratrap data, in Proceedings of the European Conference on Computer Vision (Springer Nature Switzerland, Cham, 2026), pp. 37-57.

98. S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, K. Zhu, Qwen3-VL technical report. https://arxiv.org/abs/2511.21631 (2025).

Acknowledgements: We thank the staff at Klepelshagen estate for supporting the wildlife monitoring field experiments in Germany. Drone flights at the Klepelshagen estate and surrounding areas were authorized by the District Administrator of Vorpommern-Greifswald. We also thank the staff at the German Aerospace Center (DLR) Berlin and the Berlin Police for organizing and supporting the surveillance field experiments in Berlin, Germany. Drone flights in Fighting City Berlin were authorized by the Berlin Police. All legal requirements regarding the protection of uninvolved personnel have been met. Furthermore, drone flights in Way Kambas National Park and Ujung Kulon National Park in Indonesia were authorized by the Indonesian Ministry of Forestry. We extend our sincere gratitude to all drone pilots from Yayasan Badak Indonesia (YABI / Rhino Foundation of Indonesia) and the Sumatran Rhino Sanctuary (SRS), who were trained during this campaign, for their invaluable assistance with data acquisition (drone flights and video recording).

## Funding:

The Indonesia expedition was partially funded and co-organized by Specialized Wildlife Services Ltd, the Yayasan Badak Indonesia - the Rhino Foundation of Indonesia (YABI), and the Sumatran Rhino Sanctuary (SRS).

## Author contributions:

Conceptualization: OB

Methodology: OB

Software: RJAAN, MY, VLB

Investigation/Implementation: OB, RJAAN, MY, VLB, RB, KH

Resources: KH, RB, OB

Visualization: OB, RJAAN, MH, VLB

Funding acquisition: OB

Project administration: OB, KH, RB

Supervision: OB

Writing – original draft: OB

Writing – review & editing: OB, RJAAN, MY, VLB, RB, KH

## Competing interests:

Authors declare that they have no competing interests.

## Data, code, and materials availability:

The main movies, data, and code for this study have been deposited in the database https://zenodo.org/records/23057130. Additional supplementary material on the TMP results from preceding pilot studies in rainforest environments and with volant animals has been archived in a separate database: https://zenodo.org/records/23079962. No new materials were generated in this study.

## Supplementary Materials:

Supplementary Materials in this file: Figs. S1 to S3 Tabs. S1 to S2 Legends for Movie S1 to S5 Screenshots and Legends for Software S1 to S2 Legends for Data S1 to S2

The main movies, data, and code for this study have been deposited in the database https://zenodo.org/records/23057130.

Additional supplementary material on the TMP results from preceding pilot studies in rainforest environments and with volant animals has been archived in a separate database: https://zenodo.org/records/23079962.

Statistics for individual flights and species of wildlife monitoring experiments Table S1 provides individual classification statistics for flights under daylight and nightvision conditions. The accuracy of night-vision flights is notably lower than that of daylight flights. Table S2 provides classification statistics per species.
<table><tr><td rowspan=1 colspan=1>date</td><td rowspan=1 colspan=1>flight number / type</td><td rowspan=1 colspan=1>recall</td><td rowspan=1 colspan=1>precision</td><td rowspan=1 colspan=1>F1 score</td><td rowspan=1 colspan=1>accuracy</td></tr><tr><td rowspan=1 colspan=1>27/8/2026</td><td rowspan=1 colspan=1>1 / day light</td><td rowspan=1 colspan=1>75.0 %</td><td rowspan=1 colspan=1>100.0 %</td><td rowspan=1 colspan=1>85.7 %</td><td rowspan=1 colspan=1>95.6 %</td></tr><tr><td rowspan=3 colspan=1>28/8/2026</td><td rowspan=1 colspan=1>1 / day light</td><td rowspan=1 colspan=1>94.1 %</td><td rowspan=1 colspan=1>100.0 %</td><td rowspan=1 colspan=1>97.0 %</td><td rowspan=1 colspan=1>99.0 %</td></tr><tr><td rowspan=1 colspan=1>2 / day light</td><td rowspan=1 colspan=1>69.4 %</td><td rowspan=1 colspan=1>100.0 %</td><td rowspan=1 colspan=1>82.0 %</td><td rowspan=1 colspan=1>87.0 %</td></tr><tr><td rowspan=1 colspan=1>2 / night vision</td><td rowspan=1 colspan=1>43.6 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>60.7 %</td><td rowspan=1 colspan=1>52.1 %</td></tr><tr><td rowspan=1 colspan=1>29/8/2026</td><td rowspan=1 colspan=1>1 / day light</td><td rowspan=1 colspan=1>88.9 %</td><td rowspan=1 colspan=1>99.4 %</td><td rowspan=1 colspan=1>93.9 %</td><td rowspan=1 colspan=1>88.4 %</td></tr><tr><td rowspan=1 colspan=1>30/8/2026</td><td rowspan=1 colspan=1>1 / day light</td><td rowspan=1 colspan=1>97.8 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>98.9 %</td><td rowspan=1 colspan=1>98.9 %</td></tr><tr><td rowspan=4 colspan=1>31/8/2026</td><td rowspan=1 colspan=1>1 / day light</td><td rowspan=1 colspan=1>96.4 %</td><td rowspan=1 colspan=1>87.1 %</td><td rowspan=1 colspan=1>91.5 %</td><td rowspan=1 colspan=1>89.4 %</td></tr><tr><td rowspan=1 colspan=1>1 / night vision</td><td rowspan=1 colspan=1>18.7 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>31.5 %</td><td rowspan=1 colspan=1>58.9 %</td></tr><tr><td rowspan=1 colspan=1>2 / day light</td><td rowspan=1 colspan=1>33.3 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>50.0 %</td><td rowspan=1 colspan=1>97.0 %</td></tr><tr><td rowspan=1 colspan=1>3 / day light</td><td rowspan=1 colspan=1>89.5 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>94.4 %</td><td rowspan=1 colspan=1>94.4 %</td></tr></table>

Tab. S1. Automatic outlier removal performance. Per-flight statistics on outlier removal (post-filtering verification images that contain animals). One to three flights were conducted per day, as indicated by the flight number. Flight type denotes whether night vision or daylight imaging was used, with the zoom camera switching automatically based on lighting conditions. Overall, performance is lower in night vision mode due to fewer image features and the absence of color saturation. A total of 1,276 verification images were captured, of which 579 contained animals (ground truth). Of these, 440 passed the outlier removal step.

<table><tr><td rowspan=1 colspan=1>species</td><td rowspan=1 colspan=1>recall</td><td rowspan=1 colspan=1>precision</td><td rowspan=1 colspan=1>F1 score</td><td rowspan=1 colspan=1>accuracy</td></tr><tr><td rowspan=1 colspan=1>European Hares (Lepus Europaeus)</td><td rowspan=1 colspan=1>70.4 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>82.6 %</td><td rowspan=1 colspan=1>98.2 %</td></tr><tr><td rowspan=1 colspan=1>European Badgers (Meles Meles)</td><td rowspan=1 colspan=1>83.3 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>90.9 %</td><td rowspan=1 colspan=1>99.5 %</td></tr><tr><td rowspan=1 colspan=1>Wild Boar (Sus Scrofa)</td><td rowspan=1 colspan=1>50.0 %</td><td rowspan=1 colspan=1>25.0 %</td><td rowspan=1 colspan=1>33.3 %</td><td rowspan=1 colspan=1>99.1 %</td></tr><tr><td rowspan=1 colspan=1>Cattle (Bos Taurus)</td><td rowspan=1 colspan=1>83.5 %</td><td rowspan=1 colspan=1>100 %</td><td rowspan=1 colspan=1>91.0 %</td><td rowspan=1 colspan=1>96.1 %</td></tr><tr><td rowspan=1 colspan=1>Roe Deer (Capreolus Capreolus) / Red Deer(Stags and Hinds of Cervus Elaphus)</td><td rowspan=1 colspan=1>89.5 %</td><td rowspan=1 colspan=1>99.2 %</td><td rowspan=1 colspan=1>94.1 %</td><td rowspan=1 colspan=1>92.5 %</td></tr></table>

Tab. S2. Automatic species classification. Per-species statistics on automatic species identification using the 440 verification images after outlier removal. Note that wild boar images were underrepresented, with only two captured. The classification model could not differentiate between deer subspecies (i.e., roe deer and red deer)

## Mapping of Surveillance Experiments

GPS locations of detected individuals (people) often reveal spatial patterns along forest paths and roads, whereas others were verified deep within the forest at designated positions.

![](images/2a462217398cb4d0196d7f9f9925d193227fad6de2688fe0047cd4ba19c25914.jpg)  
Fig. S1. Mapping of surveillance experiments. The satellite image displays the Ruhleben Fighting City and surrounding facilities in Berlin, alongside the mapped positions of verified targets.

## Temporal Max Pooling for Volant Animals

Applying TMP to volant animals (e.g., insects or birds) provides a straightforward way to visualize their flight paths. This approach may support future biological studies investigating such flight behavior. Because Fig. S2 presents only the TMP results without the additional trACT components, the corresponding video files and software have been archived in a separate database (https://zenodo.org/records/23079962).

![](images/0ec51c4ea362f8d56cf94a396c677ce817854638893e03f22119fc32acca2931.jpg)  
Fig. S2. Temporal Max Pooling for volant animals. The ��� channel enables the visualization of flight paths. Examples shown are a swarm of dragonflies (Anisoptera) (A), swarm of mosquitos (Nematocera) (B), flock of European starlings (Sturnus vulgaris) (C), and bat (Chiroptera) (D). See Supplementary Movies S6-S9 (https://zenodo.org/records/23079962).

## Temporal Max Pooling Results in Rainforest Environments

TMP (specifically only the $T M P _ { v }$ channel) was initially used and evaluated during a field expedition in Sumatra’s Way Kambas National Park (TNWK) and Java’s Ujung Kulon National Park (TNUK) during May 3-14, 2026. While the primary objective of the expedition was to confirm the presence of the critically endangered wild Sumatran rhinoceros (Diceorhinus sumatrensis) in TNWK, our method also proved highly effective for detecting various other wildlife species and humans in tropical rainforest ecosystems (Figs. S3-S4, Supplementary Movies S10-S17, and Supplementary Software S3). Because Figs. S3-S4 depicts the findings of a pilot study preceding trACT, the corresponding video files and software have been archived in a separate database (https://zenodo.org/records/23079962). Note that Supplementary Software S3 computes only $T M P _ { v }$ , as used during the expedition.

![](images/b361fba53c5f7784b896460817606366a718d50a4f01d672acaa801dca3f116f.jpg)  
Fig. S3. Detecting a semi-wild Sumatran rhino (Diceorhinus sumatrensis) in thermal clutter during daylight. The example illustrates the detection of a rhino’s motion in the $T M P _ { v }$ channel (B) amid the thermal clutter of a fixed focal-length thermal camera (A). The detection was achieved from an altitude of approximately 140m above ground level (AGL) and was verified using a telezoom RGB camera (D) while it also remained undetectable in wide-angle RGB recordings (C). See Supplementary Movie S10 (https://zenodo.org/records/23079962). Note that the rhino is semi-wild and lives in a large forested paddock area in TNWK for a breeding program. It was searched and detected for training purposes. .

![](images/217945b58b1731af26319f964b336714cc3a6675f84345013f92f9ca820aac89.jpg)  
Fig. S4. Additional TMP results obtained in a tropical rainforest ecosystem. Shown is the ��� channel. Body and branch motions induced by walking persons (A,B), flying and perching birds (C,D), smaller mammals such as long-tailed macaques (Macaca fascicularis) and black monkeys (Presbytis thomasi) (E,F), flight trajectories of hunting bats and insects (G), and the first visual sighting of a wild Sumatran rhino (Diceorhinus sumatrensis) in TNWK since 2017 (H). Note that A-F were recorded during daylight, while G-H were captured at night. Acquisition altitudes ranged from 119-165m AGL. See Supplementary Movies S10-S17 (https://zenodo.org/records/23079962) corresponding to A-H, respectively.

Anti-poaching examples are illustrated in Figs. S4A,B, where the body motion of a walking person and the associated branch motion were detected in thermal and RGB recordings acquired from an altitude of 140-143m AGL, despite strong thermal clutter and partial occlusion (see also Supplementary Movies S10 and S11). Bird flight trajectories (captured from 165m AGL) and perching hornbills (captured from 125m AGL) were detected in RGB recordings, as shown in Figs. S4C,D (see also Supplementary Movies S12 and S13). Smaller mammals, such as a long-tailed macaque (recorded from 152m AGL, Fig. S4E and

Supplementary Movie S14) and a black monkey (recorded from 119m AGL, Fig. S4F and Supplementary Movie S15), were detected in thermal recordings based either on body motion or on conspicuous branch motion. While the examples above were recorded during daylight at temperatures of up to 35°Celsius, the hunting bats and their insect prey are shown in Fig. S4G (see Supplementary Movie S16) were recorded at night from the ground. As in the case of the flying birds in Fig. S4C, TMP implicitly recovers the flight trajectories of both insects and bats. The detection of a wild Sumatran rhino (the first visual sighting in TNWK since 2017) is shown in Fig. S4H (see Supplementary Movie S17). The discovery was made at night using thermal recordings acquired from an altitude of 152m AGL. While TMP clearly detected its motion, the thermal signal alone would have been sufficient in this case, as the animal was not visually occluded and its thermal signature was clearly distinguishable from the cool ambient environment. While the shape and motion patterns in the thermal signal alone suggested that the animal was a rhinoceros, definitive confirmation was obtained through ground verification, during which fresh footprints, scratch marks, and environmental DNA (eDNA) were found.

Movie S1. Airborne camera trap principle. This video illustrates a hawk’s fixation strategy and demonstrates its application to a drone. It first shows how trACT uses motion signals to find and track a herd of red deer through dense forest occlusions and thermal clutter, achieving automatic detection and verification. It then presents technical verification experiments conducted under controlled conditions, focusing on motion anomaly detection, multi-target verification, and partial occlusion.

Movie S2. Wildlife monitoring with manual detection. This video presents two examples of manual detection and automatic verification using motion signals from wildlife monitoring field experiments. A roe deer (Capreolus capreolus) was detected in dense, low-tree vegetation, and a red deer stag (Cervus elaphus) was located hiding in a cornfield.

Movie S3. Wildlife monitoring with automated single-target detection and verification. This video presents eight examples of manual detection and automatic verification using motion signals from wildlife monitoring field experiments. A red deer stag (Cervus elaphus) and a roe deer (Capreolus capreolus) were detected in dense blackberry bushes; two European hares (Lepus europaeus) were found hiding in tall grass and open fields; two American minks (Neovison vison) chasing each other in an open field; two roe deer hiding in tall grass were captured in both daylight and night-vision modes; and a European badger (Meles meles) was detected under a tree.

Movie S4. Wildlife monitoring with multi-target detection and verification. This video presents three examples of automatic detection and verification for multiple targets using motion signals. The footage features roe deer (Capreolus capreolus) hiding in grass in night-vision mode, and cattle (Bos taurus) in a pasture during daylight.

Movie S5. Surveillance examples. This video presents multiple examples of manual and automatic detection and verification for both unoccluded individuals and people partially occluded by forest vegetation, covering single and multiple targets.

![](images/9d2c886c0a1eb04f6eac4d41f83344ccde3d0e1620aa0083c6b7e0f00e2692b2.jpg)  
Software S1. trACT software for DJI. Screenshot of the Android application installed on the DJI RC Plus remote controller, used with the DJI Matrice 30T for all field experiments. The application supports switching models (F, M, AS, AM), configuring parameters (TMP and camera), and displaying raw video and $T M P _ { v }$ streams alongside flight and drone telemetry. Manual detection is supported via touchscreen input on the target. Note that due to dual-use regulations, access to this application is restricted and available upon request.

![](images/cf39d396a26a6e4e790685f7de6b80c255a76939c8a86da6046450aebcfa80df.jpg)  
Software S2. trACT software for Windows. Screenshot of the Windows software (with Python source code available) demonstrating $\mathrm { \ t r { A C T ^ { \prime } s } }$ core functions (including TMP computations, motion detection, and multi-target verification) using preloaded sample videos, with support for custom video uploads.

Data S1. Data for wildlife monitoring field experiments (Fig. 5). The dataset contains all trACT verification images for the wildlife monitoring experiments (both before and after outlier removal and visual reasoning-based species classification) along with manually selected and classified ground truth images. The data are organized by flight (date and flight number, as listed in Table S1). A companion Python script for outlier removal and species classification is also provided.

Data S2. Data for surveillance field experiments (Fig. 6). The dataset contains all trACT verification images for the surveillance experiments (both before and after outlier removal) along with manually curated ground truth images. Because the flights were conducted on a single day, corresponding images are stored in the same folders. A companion Python script for outlier removal is also provided. Note that all faces have been blurred for privacy, which may prevent the reproduction of our outlier removal results.