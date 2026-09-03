# Doppio: A Dataset for Contactless Weight Estimation of Falling Particles

Simon Kiefhaber<sup>∗,1</sup> , Jan-Martin O. Steitz<sup>∗,1</sup> , Julia Grabinski<sup>1</sup> Christoph Reich<sup>1,2,3,4</sup> , Paul Wagner<sup>1</sup> , Max Zimmermann<sup>1</sup> , Simone Schaub-Meyer<sup>1,3,5</sup> , and Stefan Roth<sup>1,3,5</sup>

<sup>1</sup> TU Darmstadt <sup>2</sup> TU Munich <sup>3</sup> ELIZA <sup>4</sup> MCML <sup>5</sup> hessian.AI <sup>\*</sup> equal contribution {name.surname}@visinf.tu-darmstadt.de https://visinf.github.io/doppio

Abstract. Measuring the mass of powder, including falling particles, is a common task in industrial applications. While scales are efective for static measurements, many applications require contactless sensing, where existing solutions are often costly, application-specific, and technically complex. In this work, we investigate computer vision as a practical alternative for contactless mass estimation. As an accessible real-world case study, we focus on cofee grinding and introduce Doppio, a novel video dataset capturing videos of falling ground cofee, paired with precise, per-frame ground-truth weight measurements. To demonstrate contactless measuring, we evaluate deep learning-based approaches ranging from purely spatial feed-forward networks to recurrent spatio-temporal models. These models are analyzed with respect to their predictive accuracy and computational trade-ofs. We demonstrate that deep learningbased computer vision models accurately estimate the cumulative weight of falling particles, establishing a solid foundation for future vision-based contactless measurement solutions.

Keywords: Weight Estimation · Video Dataset · Cofee Analysis

## 1 Introduction

The precise measurement of powder mass is a fundamental task common in numerous industry applications, including solid dosage in pharmaceutical manufacturing [1], additive manufacturing via laser metal deposition [2], and the food industry [30]. Mechanical vibrations often prohibit the application of scales or weight cells. Additionally, they only support batch processing, restricting material flow. For a continuous mode of manufacturing or contactless applications, other specialized measurement approaches are required. These often include highly specialized hardware such as microwave radar or X-ray [12,25], significantly increasing engineering efort, technical complexity, and total production cost. A simple, cost-eficient, and general contactless measuring solution remains lacking.

Therefore, there is a significant incentive to move toward of-the-shelf contactless measurement solutions. Computer vision ofers a compelling alternative.

Optical sensors are less prone to mechanical vibration and universally available as low-cost, high-resolution video cameras. Additionally, current computer vision approaches demonstrate efective video understanding [38,54]. Video-based weight estimation using computer vision can aid manufacturers reduce costs and technical complexity while increasing system durability.

As access to industry-scale systems that use powder flow is limited, we turn to cofee bean grinding as a case study. The grinding of cofee beans provides a relevant setting, as industry applications frequently rely on the dosing of finegrained materials. Taking a closer look, during cofee grinding, cofee particles agglomerate mainly due to the triboelectric efect [41], but also because of capillary liquid bridges from the lipid fraction released from the cofee beans [20,52], and powder caking due to compaction in the grinder [8]. As a result, depending on agglomerate size, we observe diferent speeds and occlusions of smaller clumps and particles, making the weight estimation of falling ground cofee a challenging problem and, therefore, an adequate substitute model.

To the best of our knowledge, no published work has explored the use of computer vision for dynamic and contactless weight estimation of falling agglomerated particles. Our core contribution is to present and release the first specialized dataset for vision-based weight estimation of ground cofee: The Doppio dataset comprises videos capturing a diverse range of roast types as they fall from a grinder at multiple grind settings, providing a benchmark to facilitate future research in this domain. Along with the dataset, we present a range of models, from a simple time-based model to recurrent deep neural networks, to showcase that weight information can be extracted from the continuous video signal.

## 2 Related Work

Vision-Based Contactless Weight Estimation. To bypass the high cost and mechanical limitations of physical scales, estimation of mass through computer vision finds application in broader agricultural and industrial contexts [37,39,42]. In livestock monitoring, various approaches for weight estimation have been introduced, demonstrating a shift toward contactless sensors to reduce operational complexity [9]. In the context of vegetable and food processing, geometric feature mapping alongside multi-form shape properties has been used to estimate weights [27,43]. Crucially, when dealing with materials in motion, recent work has explored tracking frameworks; for instance, deep learning-based object detection has been used to estimate the cumulative mass flow of irregular crops moving rapidly along a harvester conveyor belt [26]. More broadly, deep learning-based end-to-end image-to-mass frameworks have successfully modeled the complex relationship between an item’s visual geometry, volume, and hidden material densities to predict physical weight [6,53] or calories [19,28,40,45].

Mass Flow Estimation of Particles. In the domain of mass flow estimation for particles in industrial settings, it is common to employ measurement methods that use modalities other than RGB imagery [47]. Optical sensors, using photoreceptors and lasers, have been employed to estimate mass flow rates by measuring the length of falling particle clusters [16]. For pharmaceutical tablet manufacturing, both capacitance-based sensing [24] and X-ray sensors [12] have demonstrated real-time, in-line mass flow rate monitoring in a non-invasive manner. Microwave Doppler radar has been applied to solid-flow measurements, with falling beans serving as a model [25]. Using an image-based analysis, Gao et al. [13] recover the particle size distribution from a laser-illuminated stream of pneumatically conveyed particles using a high-speed charge-coupled device camera and contour-based image processing. While these approaches demonstrate that reliable mass flow estimation and particle characterization are achievable in industrial pipelines, they generally rely on a pneumatically controlled mass flow or dedicated, specialized hardware (e.g., X-ray, solid-state laser, or microwave Doppler radar).

Deep-Learning-Based Cofee Analysis. Translating visual volume into a precise weight metric requires accounting for particle size and material density. Due to limited access to industrial particle-flow systems, we utilize cofee grinding as an accessible case study. In the context of cofee processing, research has focused on analyzing the following granular variations. For example, cofee granularity classification has been approached using AlexNet [29,34].Exploring the use of simple consumer hardware [35] demonstrated that using traditional computer vision techniques, such as Canny edge detection [3] and connected components labeling [7], allows for measuring fine particles (200-1600 µm) using mobile-phone cameras. Following this approach, shape analysis was used on ground cofee to determine the distribution of particle sizes [44]. Further work in the cofee industry focused on classifying raw materials [4] and on quality control [22,36] during the roasting process. Several studies have leveraged deep learning architectures to categorize bean characteristics. For instance, a classification dataset comprising 8 k images of unroasted cofee beans was proposed to identify defects and bean varieties [11]. Regarding the roasting process, standard deep learning-based vision architectures such as LeNet [32], ResNet [18], DenseNet [23], and MobileNet-V2 [48] were trained and benchmarked on individual beans at various roast levels [17]. Similar classifications on images containing bulk quantities of roasted beans were also performed using neural networks [33]. While these works establish the efectiveness of using deep learning-based vision models for identifying cofee-specific features, they remain limited to static, non-temporal analysis.

While these studies focus on analyzing the cofee, they do not quantify how much is being ground in a dynamic setting. In this work, we extend these concepts by moving beyond static, categorized, or volumetric analysis toward spatiotemporal understanding for eficient, real-time contactless weight estimation of falling particles.

![](images/0759087ed6bfa910c13c0f6af28caf854bd22e3f7e90e6748b75786fad311b7c.jpg)  
Fig. 1. Data acquisition pipeline for a single sequence of the Doppio dataset. Crops of the display and cofee are taken relative to the ArUco markers. The display crop is then processed by an optical character recognition (OCR) approach, and a time-lag compensation is applied. Finally, we construct a sequence of image-weight pairs.

## 3 The Doppio Dataset

Due to the lack of publicly available datasets containing videos of falling particles with per-frame weight annotations, we record a dedicated dataset—Doppio. We will release our Doppio dataset, evaluation metrics, and data loader with the publication of this paper. We use a cofee grinder to produce falling cofee particles and capture videos of the falling cofee. In these videos, we also capture the machine’s display showing the cumulative weight of the cofee determined by an integrated scale.

## 3.1 Data Acquisition

For our recordings, we use a Fiorenzato AllGround Sense cofee grinder. This grinder ofers discrete grind size adjustments and an integrated scale, well-suited for our data acquisition. The scale provides the real-time weight measurements, which we record to obtain ground-truth weight measurements. To ensure sufficient variability, we record sequences spanning 25 diferent grind sizes and 3 diferent types of cofee beans. Each video in the dataset shows a continuous grinding process that yields about 32 g of ground cofee. Figure 1 (left) provides an overview of our data acquisition setup.

To ensure a consistent spatial arrangement of our capturing setup, we attach ArUco markers [14] to all critical components in the scene. This includes the floor, the main camera, the grinder, the lighting, and the tripods. For robust pose estimation and tracking, each object is equipped with at least three markers. A Logitech C920 webcam is used to detect these markers and maintain spatial calibration between recordings.

![](images/8b65bed801cc38d0211e16bac6bdca0ff5a8244239eb28dc47667768b5459b91.jpg)  
Fig. 2. Example weight sequence. Visualization of the raw and smoothed weights of one exemplary sequence of our Doppio dataset. Additionally, the start ■, mid ■, and end ■ sections are highlighted.

To record the main sequence of falling particles, it is important to capture the fast-moving cofee particles without motion blur, so they remain clearly visible. Therefore, we employ a Canon EOS R5 camera, recording at 60 fps at a 4 K resolution, with a fast shutter speed of 1/2000 s, an aperture of f/3.2, and an ISO value of 1000. In Fig. 1, we show an example frame captured by the camera with these settings.

## 3.2 Postprocessing and Annotation

We extract a 512×512 px crop of the falling cofee from each frame. To determine the cumulative weight, we extract a second crop that contains the machine’s display (cf . Fig. 1 (middle)). Then, we apply an optical character recognition (OCR) pipeline based on Tesseract OCR [51] to read the display’s content. Since the weight is unreadable in some frames due to the display refreshing its content while the frame is captured, we filter out OCR misdetections and fill missing values. We identify misdetections by assuming a monotonic growth in weight and by limiting the maximum increase between consecutive frames. The missing values are then linearly interpolated from their closest valid neighbors.

We also compensate for the time lag between when the ground cofee appears in the camera frame and when the weight is measured (cf. Fig. 1 (right)). This ofset is caused by the time it takes for the cofee to fall onto the scale after passing through the cropped window. To that end, we estimate and apply a per-sequence time ofset. This ofset is estimated per sequence using the first frame in which ground cofee appears and the frame in which the scale detects an initial weight.

Since the integrated scale measurements show only a single digit after the decimal point, we apply a moving average window of seven frames to the raw measurements as a smoothing operator. This avoids ramp-like trajectories in our data. An illustrative example of this smoothing efect is presented in Fig. 2.

Table 1. Dataset statistics overview. We report statistics for training, validation, and test splits: mean weight increment and standard deviation per frame, minimum and maximum number of frames per sequence, and number of sequences per split.
<table><tr><td colspan="6">Split Mean  $\mathbf { ( g / f r a m e ) }$  Std  $\mathbf { ( g / f r a m e ) }$  Min frames Max frames Sequences</td></tr><tr><td>train</td><td>0.0304</td><td>0.0193</td><td>810</td><td>1406</td><td>131</td></tr><tr><td>val</td><td>0.0284</td><td>0.0196</td><td>979</td><td>1416</td><td>13</td></tr><tr><td>test</td><td>0.0303</td><td>0.0192</td><td>839</td><td>1480</td><td>75</td></tr></table>

## 3.3 Dataset Splits

We split our dataset into 131 training, 13 validation, and 75 test sequences. Further statistics of the splits are reported in Tab. 1. We ensure each combination of grind size and cofee bean type is present in the training and test set.

We further split the validation and test sequences into the following subsequences: Start and end contain the first and last 256 frames of each sequence, respectively. Mid contains 256 consecutive frames sampled from in-between the start and end sections (cf . Fig. 2). The $f u l l$ setting contains the entire sequence. These sub-sequences enable more detailed model evaluations, as the grinder’s behavior varies throughout the grinding process. For example, at the start, slightly fewer grounds fall than in the middle. Towards the end, the machine’s internal control loop stops or slows down the grinding process to reach the predefined target weight more accurately (cf. Fig. 2 (right top)). We include a video of the complete grinding process in our supplemental material to demonstrate this.

## 3.4 Evaluation Metrics

For our evaluation metrics, we define a doppio (a double shot of espresso, a common espresso serving) as 16 g of ground cofee. We measure the mean absolute error (MAE) per doppio. We term this metric $\mathrm { M A E } _ { \mathbb { 0 } } .$ , and we define it as

$$
\mathrm { M A E } _ { 0 } = \frac { 1 6 } { N } \sum _ { i = 1 } ^ { N } \frac { | w _ { i } ( L _ { i } ) - \hat { w _ { i } } ( L _ { i } ) | } { w _ { i } ( L _ { i } ) } ,\tag{1}
$$

where N is the number of sample sequences, $w _ { i } ( j )$ is the measured ground truth weight at the $j \mathrm { - t h }$ frame for the i-th sequence. $\hat { w } _ { i } ( j )$ denotes the estimated weight. $L _ { i }$ is the length of the i-th sequence, and therefore $w _ { i } ( L _ { i } )$ refers to the last element of the sequence, so the MAE is only calculated after the entire sequence is processed.

Since our dataset contains per-frame weight annotations, we can measure the MAE at each time step and calculate the area between the predicted and groundtruth weight measurements. To enable comparison of variable-length sequences, we normalize this metric by N, the number of frames within a sequence. We define this metric as

$$
\mathrm { M A E } _ { \perp } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { L _ { i } } \sum _ { j = 1 } ^ { L _ { i } } | w _ { i } ( j ) - \hat { w _ { i } } ( j ) | .\tag{2}
$$

![](images/10b9cbd56d3692fbb1a3af0843fa7c7f0fdfc2364cefe85d5e5f3615b74dfced.jpg)  
Fig. 3. Metrics visualization. $\mathrm { M A E } _ { \perp }$ (gray shaded area ■) is calculated between the prediction and ground truth for each frame. $\mathrm { M A E } _ { \mathbb { Q } }$ (red ■ dashed line) is computed at the end of the sequence and measures the final discrepancy between the predicted and ground truth weight, while normalizing to the discrepancy per doppio (i.e., 16 g).

We provide a visualization of how $\mathrm { M A E } _ { \mathfrak { d } }$ and $\mathrm { M A E } _ { \boxed { \cdot } }$ relate to the sequence of frames in Fig. 3.

## 4 Methods for Contactless Weight Estimation

To systematically address the challenge of contactless weight estimation, we evaluate models of various complexity, ranging from simple linear regression to deep spatio-temporal architectures. First, we establish a na¨ıve, time-based baseline to quantify the predictive boundaries achievable without any visual inputs, followed by more sophisticated computer vision models.

## 4.1 Time-Based Baseline

The na¨ıve approach to estimating the weight of falling particles is a time-based model calibrated for each new combination of input material and particle size. We build this baseline model using linear regression for each possible combination of bean type and grind size, enabling us to assess whether more complex visionbased models are needed or whether the assumption of a constant fall rate is suficient to solve this task.

## 4.2 Vision-Based Models

To predict the target weights across our video data, we process each frame individually to estimate the weight diference caused by the cofee visible at that time step. Per-frame diferences are then accumulated using a cumulative sum to obtain the total weight of the processed sequence up to any given frame. For our task of estimating the cumulative weights of falling particles, we explore three vision-based approaches: a feed-forward approach that accumulates no temporal information, a recurrent architecture that can capture long sequences, and a temporal approach that considers multiple previous frames for its predictions.

Since the weight can only increase monotonically over time, we limit our models to output only positive weight estimates by using a softplus [10] activation function.

As inputs for all of our vision-based models, we use either a single RGB frame $\mathbf { \Psi } _ { I _ { t } } \mathbf { \bar { \Psi } } \in \mathbb { R } ^ { H \times W \times C }$ with $C = 3$ or a concatenation of the current frame and the pixel-wise diference between the current and previous frame, denoted as $\varDelta I _ { t } .$ , doubling the channel dimension to $C = 6$

Our feed-forward approach uses a standard feed-forward network (FFN) to directly predict the weight diference from only the input at the current time step, ignoring all previous frames. Standard FFNs lack temporal awareness, meaning that falling particles visible in multiple frames can skew the results. Thus, we also employ a recurrent architecture to track these particles accurately over time. We aim to capture long-term temporal dependencies by employing a gated recurrent unit (GRU) [5] coupled with a feed-forward backbone acting as a feature extractor. The extracted spatial features are passed to the GRU, which updates its hidden state and outputs the predicted weight diferences. Since long-term temporal context may be unnecessary for this task, we also evaluate temporal convolutional networks (TCNs) [31] as an alternative to GRUs, using the same spatial features. In the TCNs, we apply causal convolutions along the time axis to also consider information from previous frames within a limited time window.

## 5 Experiments

## 5.1 Training Setup

For all experiments, we train our deep-learning-based models on sub-sampled sequences of 128 consecutive frames within each batch. Our training runs for 200 epochs using a one-cycle learning rate schedule with cosine annealing [50], with the first 10 % of iterations used for warmup.

To improve robustness and generalization, we apply random adjustments to the brightness and contrast of each sequence. Furthermore, we introduce a domain-specific temporal augmentation by randomly inserting frames containing no falling cofee into 10 % of the frames of each train sequence. This encourages the network to correctly predict a zero-weight delta when no particles are present.

As the learning objective, we employ a combined loss that supervises both the final accumulated weight and the intermediate per-frame diferences, enforcing correct temporal dynamics. Our loss on the last accumulated weight is

$$
\mathcal { L } _ { \mathrm { s e q } } = d ( w _ { i } ( L _ { i } ) , \hat { w } _ { i } ( L _ { i } ) ) ,\tag{3}
$$

where $d ( x , y )$ measures the distance between x and $y . \ w _ { i } ( L _ { i } )$ is the ground truth weight and $\hat { w } _ { i } ( L _ { i } )$ the estimated weight, both at the end of the i-th sequence denoted by $L _ { i }$ . To enforce correct temporal dynamics, we compute a diference loss between the predicted weight deltas and the ground-truth per frame $j \colon$

$$
\mathcal { L } _ { \delta } = \frac { 1 } { L _ { i } - 1 } \sum _ { j = 2 } ^ { L _ { i } } d ( w _ { i } ( j ) - w _ { i } ( j - 1 ) , \hat { w } _ { i } ( j ) - \hat { w } _ { i } ( j - 1 ) ) .\tag{4}
$$

Table 2. Results on diferent input representations. We report results of our models with and without frame diference features $\varDelta I _ { t }$ on Doppio test, using $\mathrm { M A E } _ { \mathbb { 0 } }$ and $\mathrm { M A E } _ { \perp }$ (both ↓) for the full sequences. We highlight the best results for each metric per row in orange ■.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2"></td><td colspan="2">Without  $\varDelta I _ { t }$ </td><td colspan="2">With  $\varDelta I _ { t }$ </td></tr><tr><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td></tr><tr><td rowspan="4">FN</td><td>MobileNet-V3-S</td><td>0.28</td><td>0.39</td><td>0.19</td><td>0.25</td></tr><tr><td>MobileNet-V3-L</td><td>0.29</td><td>0.37</td><td>0.20</td><td>0.26</td></tr><tr><td>ResNet-18</td><td>0.24</td><td>0.33</td><td>0.26</td><td>0.30</td></tr><tr><td>ResNet-34</td><td>0.25</td><td>0.33</td><td>0.21</td><td>0.24</td></tr><tr><td rowspan="4">GRU</td><td>MobileNet-V3-S</td><td>0.35</td><td>0.32</td><td>0.35</td><td>0.31</td></tr><tr><td>MobileNet-V3-L</td><td>0.24</td><td>0.26</td><td>0.21</td><td>0.23</td></tr><tr><td>ResNet-18</td><td>0.19</td><td>0.26</td><td>0.20</td><td>0.24</td></tr><tr><td>ResNet-34</td><td>0.19</td><td>0.26</td><td>0.27</td><td>0.33</td></tr><tr><td rowspan="4">CN</td><td>MobileNet-V3-S</td><td>0.31</td><td>0.37</td><td>0.24</td><td>0.31</td></tr><tr><td>MobileNet-V3-L ResNet-18</td><td>0.19</td><td>0.23</td><td>0.29</td><td>0.33</td></tr><tr><td>ResNet-34</td><td>0.18</td><td>0.23</td><td>0.18</td><td>0.24</td></tr><tr><td></td><td>0.18</td><td>0.23</td><td>0.17</td><td>0.23</td></tr></table>

Our final loss is a combination of $\mathcal { L } _ { \mathrm { s e q } }$ and $\mathcal { L } _ { \delta }$ , weighted by a combination factor $\lambda \in \mathbb { R } ^ { + } , i . e . , \mathcal { L } = \mathcal { L } _ { \mathrm { s e q } } + \lambda \mathcal { L } _ { \delta }$ . As the expected weights in our setting are numerically small, we use the smooth L1 [15] distance for $d ( x , y )$ across all experiments. An evaluation of alternative distance functions is provided in the supplement.

Since all of our models described in Sec. 4.2 use a feed-forward network for feature extraction, we finetune ImageNet-pretrained [46] ResNets [18] and MobileNet-V3s [21] of diferent sizes on our proposed dataset. For our GRU architectures, we use a single GRU layer with a hidden dimension size of 256. In our TCNs, we use 4 causal convolution blocks with kernel size $5 ,$ hidden dimension 256, and a dilation that doubles in each block, starting at 1.

## 5.2 Doppio Evaluation Results

Design Choices. As described in Sec. 4.2, we propose two input modes for our approaches: one that uses only the current frame $I _ { t } ,$ and another that uses the current frame and its pixel-wise diference with the previous frame, denoted as $\varDelta I _ { t }$ , stacked along the channel dimension. In Tab. 2, we analyze the impact of this design decision using diferent architectures in combination with varioussized backbones. For our feed-forward architectures, we find that using frame diferences consistently improves the accuracy of all tested backbones except for ResNet-18 at very low computational cost, as only the first layer of each backbone needs to be modified to accommodate the six input channels. Given these results, we use the frame diferences as additional inputs for all subsequent FFNs. The diferences between the results of our GRU-based architectures in Tab. 2 are less pronounced, but given the large gap in accuracy for ResNets and the rather small gap for MobileNets, we chose to use the simpler inputs without frame diferences in our subsequent GRU-based architectures. Analogously, for the TCN-based architectures, we observe a significant improvement in the accuracy of the MobileNet-V3-L backbone when the frame diferences $\varDelta I _ { t }$ are excluded. Therefore, we decided not to use them in this setting.

Table 3. Results on Doppio. We report results of our models on Doppio test, using $\mathrm { M A E } _ { \mathbb { 0 } }$ and $\mathrm { M A E } _ { \perp }$ (both ↓) for diferent sections and for the full sequences. Best results are highlighted for our FFNs, GRUs, and TCNs individually in orange ■ and the overal best results over all of our models in red ■.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="2">Full</td><td colspan="2">Start</td><td colspan="2">Mid</td><td colspan="2">End</td></tr><tr><td>MAE。</td><td>MAE□</td><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td>MAE□</td><td>MAE</td><td>MAE□</td><td>MAE0</td><td>MAE</td></tr><tr><td colspan="2">Time baseline</td><td>0.54</td><td>0.57</td><td>0.98</td><td>0.25</td><td>0.75</td><td>0.20</td><td>2.95</td><td>0.68</td></tr><tr><td rowspan="3">N</td><td>MobileNet-V3-S MobileNet-V3-L</td><td>0.19 0.20</td><td>0.25 0.26</td><td>0.29 0.29</td><td>0.12 0.12</td><td>0.29 0.28</td><td>0.13 0.13</td><td>0.57 0.57</td><td>0.12 0.11</td></tr><tr><td>ResNet-18 ResNet-34</td><td>0.26</td><td>0.30</td><td>0.36</td><td>0.14</td><td>0.28</td><td>0.13</td><td>0.64</td><td>0.13</td></tr><tr><td></td><td>0.21</td><td>0.24</td><td>0.32</td><td>0.13</td><td>0.23</td><td>0.12</td><td>0.63</td><td>0.13</td></tr><tr><td rowspan="2">GRU</td><td>MobileNet-V3-S MobileNet-V3-L</td><td>0.35 0.24</td><td>0.32 0.26</td><td>0.37 0.31</td><td>0.13</td><td>0.32</td><td>0.13</td><td>0.75</td><td>0.13</td></tr><tr><td>ResNet-18</td><td>0.19</td><td>0.26</td><td>0.35</td><td>0.12 0.13</td><td>0.28 0.24</td><td>0.13 0.13</td><td>0.59 0.55</td><td>0.12 0.12</td></tr><tr><td rowspan="4">T</td><td>ResNet-34</td><td>0.19</td><td>0.26</td><td>0.31</td><td>0.12</td><td>0.26</td><td>0.13</td><td>0.56</td><td>0.12</td></tr><tr><td>MobileNet-V3-S</td><td>0.31</td><td>0.37</td><td>0.44</td><td>0.16</td><td>0.35</td><td>0.14</td><td>0.51</td><td>0.11</td></tr><tr><td>MobileNet-V3-L</td><td>0.19</td><td>0.23</td><td>0.27</td><td>0.12</td><td>0.24</td><td>0.12</td><td>0.47</td><td>0.11</td></tr><tr><td>ResNet-18 ResNet-34</td><td>0.18 0.18</td><td>0.23 0.23</td><td>0.31 0.30</td><td>0.12 0.12</td><td>0.23 0.22</td><td>0.12 0.12</td><td>0.47 0.46</td><td>0.11 0.11</td></tr></table>

We further investigate the choice of loss weighting factor λ for the diferent architecture-types and find that λ = 10 performs best for the feed-forward network, while $\lambda = 1 0 0$ is best for GRU- and TCN-based architectures. Therefore, we use these difering hyperparameters for all subsequent experiments. For completeness, we provide the full tables for these experiments in the supplement.

Main Results. Tab. 3 presents our main performance evaluation. All of our proposed vision-based architectures consistently outperform the Time baseline across all sub-sequences and metrics. When analyzing the purely spatial feedforward networks (FFNs), the MobileNet-V3-S architecture achieves the highest overall accuracy. However, the performance margins are small, as the MobileNet-V3-L and ResNet-34 architectures achieve nearly identical accuracies.

When moving to temporal models, the trend shifts towards larger backbones, as all ResNet-based models outperform their MobileNet counterparts. In the case of GRUs, all MobileNet-based models are worse than in the FFN setting. This is likely due to the aggressive feature reduction, which, on the one hand, allows for high compute eficiency in MobileNets, but, on the other hand, limits the models’ outputs to rather coarse features, lacking fine-grained details. For ResNets, we observe higher overall accuracies when used as backbones for GRUs and TCNs than when used as backbones for FFNs. In fact, our overall best model is a TCN using a ResNet-34 backbone, scoring a $\mathrm { M A E } _ { \mathfrak { d } }$ of 0.18 on the full test sequence.

Table 4. Leave-one-out validation. We train our TCN (w/ ResNet-34) only on training samples of certain types of cofee beans and report the overall $\mathrm { M A E } _ { \mathfrak { d } } \ ( \downarrow )$ on the $f u l l$ test sequences of Doppio. We also evaluate each type of bean individually and report the diference to the overall $\mathrm { M A E } _ { \mathbb { 0 } }$ indicated in gray ■.
<table><tr><td>Training Split</td><td>All</td><td>A</td><td>B</td><td>C</td></tr><tr><td>{A, B, C}</td><td>0.18</td><td>0.18 (+0.00)</td><td> $0 . 1 8 \left( + 0 . 0 0 \right)$ </td><td> $0 . 1 8 \left( + 0 . 0 0 \right)$ </td></tr><tr><td>{A, B}</td><td>0.22</td><td> $0 . 2 3 \left( + 0 . 0 1 \right)$ </td><td> $0 . 2 0 \left( - 0 . 0 2 \right)$ </td><td> $0 . 2 4 \ : ( + 0 . 0 2 )$ </td></tr><tr><td>{A, C}</td><td>0.38</td><td> $0 . 3 1 \left( - 0 . 0 7 \right)$ </td><td> $0 . 4 4 \left( + 0 . 0 6 \right)$ </td><td> $0 . 4 1 \left( + 0 . 0 3 \right)$ </td></tr><tr><td>{B, C}</td><td>0.28</td><td> $0 . 3 0 \left( + 0 . 0 2 \right)$ </td><td> $0 . 2 7 \left( - 0 . 0 1 \right)$ </td><td> $0 . 2 8 \left( + 0 . 0 0 \right)$ </td></tr></table>

![](images/6515ab50bd09fc57bb3dea7141f7e452e9a44dbc4056441f20b14922f57c63c0.jpg)  
Fig. 4. Saliency visualization. Grad-CAM maps [49] of our TCN with a ResNet-34 backbone. Saliency is mostly located in regions with falling cofee grounds, if present. Color encoding: ■■■ (low → high saliency).

To confirm that our models track only the visible cofee per frame, we analyze our TCN (w/ ResNet-34) using Grad-CAM [49]. Figure 4 shows qualitative Grad-CAM results. In general, the model only pays attention to falling cofee if present in the frame, and it only relies on cues from the background when no coffee is present. However, the last sample also shows that the network sometimes only pays attention to larger lumps as they fall, ignoring smaller ones.

Generalization. All experiments until this point evaluate exactly the same types of cofee beans in the training and test sets. This leaves open whether our proposed methods overfit only to the three types of beans we captured or generalize to new types. To gain insights into the generalization capabilities, we train our best-performing model, the TCN with a ResNet-34 backbone, on subsets of our training dataset containing only two types, and we evaluate them on the full test set in Tab. 4. We find that the overall accuracy of these models is lower than that of the model trained on the full training set, but this is most likely due to the reduced number of training samples. More interestingly, we find that the $\mathrm { M A E } _ { \mathfrak { d } }$ of the unseen bean type is only slightly higher than the $\mathrm { M A E } _ { \mathbb { 0 } }$ of the types contained in the respective training subset. This demonstrates that our methods can generalize to unseen bean types. Another insight from this experiment is that models trained on bean type B are significantly more accurate than those without it. We observe that this type of cofee tends to produce fewer lumps than the other two, and these data seem beneficial for overall training.

![](images/fbf5d1c73139bf4d0b510e837223eda01ecab40beab9f2b4d1682c4afe017039.jpg)  
Fig. 5. Cost-accuracy tradeofs of our proposed methods and backbones. We measure the memory and FLOPs for processing a 60-frame sequence (1 s of video) and report accuracy on the full test-set sequence.

Computational Cost vs. Accuracy. To evaluate trade-ofs between computational cost and model accuracy, we measure the memory footprint and the number of required floating-point operations (FLOPs) for all proposed architectures on a 60-frame sequence (1 s of video material). The results of this evaluation are presented in Fig. 5. Our lightweight MobileNet-V3 backbone uses minimal computational power and memory; further, it delivers competitive accuracy with much larger models like ResNet-34 when used as FFN architecture. However, when we use these small backbones in our GRU- or TCN-based architecture, we see a large decrease in accuracy. This suggests that the highly compressed spatial features extracted by MobileNets lack the capacity needed to build meaningful recurrent states within the GRU and TCN.

In contrast, the ResNet-based FFNs perform worse than their GRU and TCN-based counterparts. Overall, the TCN with a ResNet backbone achieves the highest accuracy across all experiments, but comes at a drastically increased cost in required compute operations, compared to our more lightweight backbones.

## 6 Conclusion

In this paper, we present a case study on contactless weight estimation for falling particles using ground cofee. To enable this research, we captured and introduced Doppio, a dataset containing precise, per-frame ground-truth weight measurements of falling cofee grounds at diferent grind sizes and cofee bean types. We utilized our dataset to conduct a comprehensive evaluation of distinct architectures, including feed-forward networks, GRUs, and TCNs. We benchmarked a diverse range of backbone architectures, ranging from lightweight MobileNet-V3 variants to ResNets. Our analysis of computational costs highlights the trade-ofs between architectural families. We demonstrated that highly compressed models are well-suited for eficient feed-forward processing, but heavier convolutional networks are necessary to build meaningful temporal contexts in GRUs. Further, we demonstrate the generalization capabilities of our model to unseen types of beans. A promising direction for future work is the exploration of quantization-aware training and model quantizations to ensure these models can be eficiently deployed on low-cost, resource-constrained hardware. In conclusion, this work provides an easily accessible dataset and testbed for robust, real-time, vision-based mass estimation of falling particles.

Acknowledgments SK has been funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC-3057. J-MS has been funded by the State of Hesse through LOEWE emergenCITY (Grant no. LOEWE/1/12/519/03/05.001(0016)/72). JG has been funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 866008). Christoph Reich is supported by the Konrad Zuse School of Excellence in Learning and Intelligent Systems (ELIZA) through the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the German Federal Ministry of Education and Research. We also acknowledge the support of the European Laboratory for Learning and Intelligent Systems (ELLIS) and Munich Center for Machine Learning (MCML). SSM and PW have been funded by the DFG – project No. 529680848. Finally, we thank L. Kammeyer, J. Milkovits, and F. Wichert for their help with recording this dataset.

## References

1. Blackshields, C.A., Crean, A.M.: Continuous powder feeding for pharmaceutical solid dosage form manufacture: A short review. Pharm. Dev. Technol. 23(6), 554– 560 (2018). https://doi.org/10.1080/10837450.2017.1339197

2. Breese, P.P., Hauser, T., Regulin, D., Seebauer, S., Rupprecht, C.: In situ measurement and closed-loop control for powder supply processes: Retrofittable solution in the context of laser metal deposition. Int. J. Adv. Manuf. Technol. 116(3), 889–903 (2021). https://doi.org/10.1007/s00170-021-07438-z

3. Canny, J.: A computational approach to edge detection. IEEE TPAMI 8(6), 679– 698 (1986). https://doi.org/10.1109/TPAMI.1986.4767851

4. Chen, P., Jhong, S., Hsia, C.: Semi-supervised learning with attention-based CNN for classification of cofee beans defect. In: ICCE-TW. pp. 411–412 (2022). https: //doi.org/10.1109/ICCE-Taiwan55306.2022.9869187

5. Cho, K., van Merrienboer, B., G¨ul¸cehre, C¸ ., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.: Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: EMNLP. pp. 1724–1734 (2014). https://doi. org/10.3115/v1/D14-1179

6. Dehais, J., Anthimopoulos, M., Shevchik, S., Mougiakakou, S.: Two-view 3D reconstruction for food volume estimation. IEEE Trans. Multimed. 19(5), 1090–1099 (2017). https://doi.org/10.1109/TMM.2016.2642792

7. Di Stefano, L., Bulgarelli, A.: A simple and eficient connected components labeling algorithm. In: ICIAP. pp. 322–327 (1999). https://doi.org/10.1109/ICIAP.1999. 797615

8. Do˘gan, M., Aslan, D., G¨urmeri¸c, V., Ozg¨ur, A., Sara¸c, M.G.: Powder caking and<sup>¨</sup> cohesion behaviours of cofee powders as afected by roasting and particle sizes: Principal component analyses (PCA) for flow and bioactive properties. Powder Technol. 344, 222–232 (2019). https://doi.org/10.1016/j.powtec.2018.12.030

9. Dohmen, R., Catal, C., Liu, Q.: Computer vision-based weight estimation of livestock: A systematic literature review. N. Z. J. Agric. Res. 65(2-3), 227–247 (2022). https://doi.org/10.1080/00288233.2021.1876107

10. Dugas, C., Bengio, Y., B´elisle, F., Nadeau, C., Garcia, R.: Incorporating second-order functional knowledge for better option pricing. In: NIPS. vol. 13, pp. 472–478 (2000), https://proceedings.neurips.cc/paper/2000/hash/ 44968aece94f667e4095002d140b5896-Abstract.html

11. Febriana, A., Muchtar, K., Dawood, R., Lin, C.Y.: USK-COFFEE Dataset: A multi-class green arabica cofee bean dataset for deep learning. In: CyberneticsCom. pp. 469–473 (2022). https://doi.org/10.1109/CyberneticsCom55287.2022. 9865489

12. Ganesh, S., Troscinski, R., Schmall, N., Lim, J., Nagy, Z., Reklaitis, G.: Application of X-Ray sensors for in-line and noninvasive monitoring of mass flow rate in continuous tablet manufacturing. J. Pharm. Sci. 106(12), 3591–3603 (2017). https://doi.org/10.1016/j.xphs.2017.08.019

13. Gao, L., Yan, Y., Lu, G.: Contour-based image segmentation for on-line size distribution measurement of pneumatically conveyed particles. In: I2MTC. pp. 1–5 (2011). https://doi.org/10.1109/IMTC.2011.5944318

14. Garrido-Jurado, S., Mu˜noz-Salinas, R., Madrid-Cuevas, F.J., Mar´ın-Jim´enez, M.J.: Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognit. 47(6), 2280–2292 (2014). https://doi.org/10.1016/J. PATCOG.2014.01.005

15. Girshick, R.B.: Fast R-CNN. In: ICCV. pp. 1440–1448 (2015). https://doi.org/10. 1109/ICCV.2015.169

16. Grift, T.: Fundamental mass flow measurement of solid particles. Part. Sci. Technol. 21(2), 177–193 (2003). https://doi.org/10.1080/02726350307492

17. Hassan, E.: Enhancing cofee bean classification: A comparative analysis of pretrained deep learning models. Neural Comput. Appl. 36(16), 9023–9052 (2024). https://doi.org/10.1007/s00521-024-09623-z

18. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

19. He, Y., Xu, C., Khanna, N., Boushey, C.J., Delp, E.J.: Food image analysis: Segmentation, identification and weight estimation. In: ICME. pp. 1–6 (2013). https://doi.org/10.1109/ICME.2013.6607548

20. Herminghaus, S.: Dynamics of wet granular matter. Adv. Phys. 54(3), 221–261 (2005). https://doi.org/10.1080/00018730500167855

21. Howard, A., Pang, R., Adam, H., Le, Q.V., Sandler, M., Chen, B., Wang, W., Chen, L., Tan, M., Chu, G., Vasudevan, V., Zhu, Y.: Searching for MobileNetV3. In: ICCV. pp. 1314–1324 (2019). https://doi.org/10.1109/ICCV.2019.00140

22. Hsia, C.H., Lee, Y.H., Lai, C.F.: An explainable and lightweight deep convolutional neural network for quality detection of green cofee beans. Appl. Sci. 12(21), 10966 (2022). https://doi.org/10.3390/app122110966

23. Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: CVPR. pp. 2261–2269 (2017). https://doi.org/10.1109/ CVPR.2017.243

24. Huang, Y.S., Medina-Gonz´alez, S., Straiton, B., Keller, J., Marashdeh, Q., Gonzalez, M., Nagy, Z., Reklaitis, G.V.: Real-time monitoring of powder mass flowrates for plant-wide control of a continuous direct compaction tablet manufacturing process. J. Pharm. Sci. 111(1), 69–81 (2022). https://doi.org/10.1016/j.xphs.2021.06. 005

25. Isa, M., Wu, Z.: Microwave Doppler radar sensor for solid flow measurements. In: EuMC. pp. 1508–1510 (2006). https://doi.org/10.1109/EUMC.2006.281364

26. Jang, S.H., Moon, S.P., Kim, Y.J., Lee, S.H.: Development of potato mass estimation system based on deep learning. Appl. Sci. 13(4) (2023). https://doi.org/10. 3390/app13042614

27. Kamiwaki, Y., Fukuda, S.: A machine learning-assisted three-dimensional image analysis for weight estimation of radish. Horticulturae 10(2), 142 (2024). https: //doi.org/10.3390/horticulturae10020142

28. Konstantakopoulos, F.S., Georga, E.I., Fotiadis, D.I.: A review of image-based food recognition and volume estimation artificial intelligence systems. IEEE Rev. Biomed. Eng. 17, 136–152 (2024). https://doi.org/10.1109/RBME.2023.3283149

29. Krizhevsky, A., Sutskever, I., Hinton, G.E.: ImageNet classification with deep convolutional neural networks. Commun. ACM 60(6), 84–90 (2017). https://doi.org/ 10.1145/3065386

30. Kruppa, F., Weiß, U., Oberdorfer, B., Wilke, B.: Increasing the dosing accuracy of a screw dosing device by inline measurement of the product density. Packag. Technol. Sci. 36(3), 185–194 (2023). https://doi.org/10.1002/pts.2703

31. Lea, C., Flynn, M.D., Vidal, R., Reiter, A., Hager, G.D.: Temporal convolutional networks for action segmentation and detection. In: CVPR. pp. 1003–1012 (2017). https://doi.org/10.1109/CVPR.2017.113

32. Lecun, Y., Bottou, L., Bengio, Y., Hafner, P.: Gradient-based learning applied to document recognition. Proc. IEEE 86(11), 2278–2324 (1998). https://doi.org/10. 1109/5.726791

33. Leme, D.S., da Silva, S.A., Barbosa, B.H.G., Bor´em, F.M., Pereira, R.G.F.A.: Recognition of cofee roasting degree using a computer vision system. Comput. Electron. Agr. 156, 312–317 (2019). https://doi.org/https://doi.org/10.1016/j. compag.2018.11.029

34. Leonard, F., Akbar, H.: Cofee grind size detection by using convolutional neural network (CNN) architecture. J. Appl. Sci. Eng. Technol. Educ. 4(1), 133–145 (2022). https://doi.org/10.35877/454RI.asci842

35. Lertsawatwicha, P., Siriborvornratanakul, T.: Measuring particle size distribution of ground cofee using computer vision. Int. J. Inf. Technol. 15(6), 2961–2967 (2023). https://doi.org/10.1007/s41870-023-01364-x

36. Liang, C., Xu, Z., Zhou, J., Yang, C., Chen, J.: Automated detection of cofee bean defects using multi-deep learning models. In: APWCS. pp. 1–5 (2023). https: //doi.org/10.1109/APWCS60142.2023.10234059

37. Liang, T., Yuan, Z.: Computer-vision-based non-contact paste concentration measurement. In: ICaMaL. pp. 1–9 (2024). https://doi.org/10.1109/ICaMaL62577. 2024.10919773

38. Lin, J., Gan, C., Han, S.: TSM: Temporal shift module for eficient video understanding. In: ICCV. pp. 7083–7093 (2019). https://doi.org/10.1109/ICCV.2019. 00718

39. Mathiassen, J.R., Misimi, E., Toldnes, B., Bondø, M., Østvik, S.O.: High-speed weight estimation of whole herring (Clupea harengus) using 3D machine vision. J. Food Sci. 76(6), E458–E464 (2011). https://doi.org/10.1111/j.1750-3841.2011. 02226.x

40. Meyers, A., Johnston, N., Rathod, V., Korattikara, A., Gorban, A., Silberman, N., Guadarrama, S., Papandreou, G., Huang, J., Murphy, K.P.: Im2Calories: Towards an automated mobile vision food diary. In: ICCV. pp. 1233–1241 (2015). https: //doi.org/10.1109/ICCV.2015.146

41. M´endez Harper, J., McDonald, C.S., Rheingold, E.J., Wehn, L.C., Bumbaugh, R.E., Cope, E.J., Lindberg, L.E., Pham, J., Kim, Y.H., Dufek, J., Hendon, C.H.: Moisture-controlled triboelectrification during cofee grinding. Matter 7(1), 266– 283 (2024). https://doi.org/10.1016/j.matt.2023.11.005

42. Nyalala, I., Okinda, C., Nyalala, L., Makange, N., Chao, Q., Chao, L., Yousaf, K., Chen, K.: Tomato volume and mass estimation using computer vision and machine learning algorithms: Cherry tomato model. J. Food Eng. 263, 288–298 (2019). https://doi.org/10.1016/j.jfoodeng.2019.07.012

43. Puri, M., Zhu, Z., Yu, Q., Divakaran, A., Sawhney, H.: Recognition and volume estimation of food intake using a mobile device. In: WACV. pp. 1–8 (2009). https: //doi.org/10.1109/WACV.2009.5403087

44. Ren, Z., Zeng, J., Yang, Z., Tang, H., Wang, J., Jiang, L., Feng, W.: Digital image analysis for contact and shape recognition of cofee particles in grinding. Powder Technol. 440, 119717 (2024). https://doi.org/10.1016/j.powtec.2024.119717

45. Ruede, R., Heusser, V., Frank, L., Roitberg, A., Haurilet, M., Stiefelhagen, R.: Multi-task learning for calorie prediction on a novel large-scale recipe dataset enriched with nutritional information. In: ICPR. pp. 4001–4008 (2021). https: //doi.org/10.1109/ICPR48806.2021.9412839

46. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M.S., Berg, A.C., Fei-Fei, L.: ImageNet large scale visual recognition challenge. Int. J. Comput. Vis. 115(3), 211–252 (2015). https://doi.org/10.1007/S11263-015-0816-Y

47. Samadi, M., Rostampour, V., Abdollahpour, S.: A review of solid particles mass flow rate measuring methods: Screening analytic hierarchy process for methods prioritization. J. Braz. Soc. Mech. Sci. Eng. 44(8), 359 (2022). https://doi.org/ 10.1007/s40430-022-03663-z

48. Sandler, M., Howard, A.G., Zhu, M., Zhmoginov, A., Chen, L.: MobileNetV2: Inverted residuals and linear bottlenecks. In: CVPR. pp. 4510–4520 (2018). https: //doi.org/10.1109/CVPR.2018.00474

49. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-CAM: Visual explanations from deep networks via gradient-based localization. Int. J. Comput. Vis. 128(2), 336–359 (2020). https://doi.org/10.1007/ S11263-019-01228-7

50. Smith, L.N., Topin, N.: Super-convergence: Very fast training of neural networks using large learning rates. In: SPIE, Artif. Intell. Mach. Learn. Multi-Dom. Oper. Appl. vol. 11006, pp. 369–386 (2019). https://doi.org/10.1117/12.2520589

51. Smith, R.: An overview of the tesseract OCR engine. In: ICDAR. pp. 629–633 (2007). https://doi.org/10.1109/ICDAR.2007.4376991

52. Speer, K., K¨olling-Speer, I.: The lipid fraction of the cofee bean. Braz. J. Plant Physiol. 18(1), 201–216 (2006). https://doi.org/10.1590/ S1677-04202006000100014

53. Standley, T., Sener, O., Chen, D., Savarese, S.: image2mass: Estimating the mass of an object from its image. In: CoRL. vol. 78, pp. 324–333 (2017), https: //proceedings.mlr.press/v78/standley17a.html

54. Tang, Y., Bi, J., Xu, S., Song, L., Liang, S., Wang, T., Zhang, D., An, J., Lin, J., Zhu, R., et al.: Video understanding with large language models: A survey. IEEE Trans. Circuits Syst. Video Technol. 36(2), 1355–1376 (2026). https://doi.org/10. 1109/TCSVT.2025.3566695

# Doppio: A Dataset for Contactless Weight Estimation of Falling Particles Supplementary Material

Table A.1. Results on diferent loss weightings. We report results of our models for $\lambda = 1 0$ and $\lambda = 1 0 0$ on Doppio test, using $\mathrm { M A E } _ { \mathbb { 0 } }$ and $\mathrm { M A E } _ { \perp }$ (both ↓) for the $f u l l$ sequences. We highlight the best results for each metric per row in orange ■.
<table><tr><td colspan="2" rowspan="2">Backbone</td><td colspan="2">λ = 10</td><td colspan="2">λ = 100</td></tr><tr><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td></tr><tr><td rowspan="4">N ResNet-18 ResNet-34</td><td>MobileNet-V3-S</td><td>0.19</td><td>0.25</td><td>0.33</td><td>0.24</td></tr><tr><td>MobileNet-V3-L</td><td>0.20</td><td>0.26</td><td>0.34</td><td>0.25</td></tr><tr><td></td><td>0.26</td><td>0.30</td><td>0.30</td><td>0.21</td></tr><tr><td></td><td>0.21</td><td>0.24</td><td>0.26</td><td>0.20</td></tr><tr><td rowspan="4">GRU</td><td>MobileNet-V3-S MobileNet-V3-L</td><td>0.37 0.26</td><td>0.34</td><td>0.35</td><td>0.32</td></tr><tr><td>ResNet-18</td><td>0.20</td><td>0.23</td><td>0.24</td><td>0.26</td></tr><tr><td>ResNet-34</td><td>0.24</td><td>0.24 0.20</td><td>0.19</td><td>0.26</td></tr><tr><td></td><td></td><td></td><td>0.19</td><td>0.26</td></tr><tr><td rowspan="4">N</td><td>MobileNet-V3-S MobileNet-V3-L</td><td>0.31</td><td>0.26</td><td>0.31</td><td>0.37</td></tr><tr><td>ResNet-18</td><td>0.25</td><td>0.22</td><td>0.19</td><td>0.23</td></tr><tr><td>ResNet-34</td><td>0.18</td><td>0.23</td><td>0.18</td><td>0.23</td></tr><tr><td></td><td>0.22</td><td>0.24</td><td>0.18</td><td>0.23</td></tr></table>

## A Loss Weighting

As described in Sec. 5.2, we provide the full evaluation of the loss weighting factor λ for the diferent architecture-types in Tab. A.1. As reported in the main paper, we find that $\lambda = 1 0$ performs best for $\mathrm { M A E } _ { \mathfrak { d } }$ of the feed-forward network, and the diferences between the $\mathrm { M A E } _ { \boxed { \cdot } }$ in both settings are rather small. At the same time, we find $\lambda = 1 0 0$ to be the best for GRU- and TCN-based architectures in terms of $\operatorname { M A E } _ { \mathfrak { d } _ { \mathfrak { d } } }$ , and the $\mathrm { M A E } _ { \perp }$ is only slightly smaller for $\lambda = 1 0$

Table A.2. Results on diferent distance measures. We report results of our models trained with diferent distance measures within their respective loss functions on Doppio test, using $\mathrm { M A E } _ { \mathbb { 0 } }$ and $\mathrm { M A E } _ { \perp }$ (both ↓) for the full sequences. We highlight the best results for each metric per row in orange ■.
<table><tr><td rowspan="3"></td><td rowspan="3">Backbone</td><td colspan="2">Smooth L1</td><td colspan="2">MAE</td><td colspan="2">MSE</td></tr><tr><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td><td> $\mathbf { M A E } _ { \mathbb { 0 } }$ </td><td> $\mathbf { M A E } _ { \perp }$ </td></tr><tr><td></td><td></td><td>0.22</td><td></td><td>0.20</td><td></td></tr><tr><td rowspan="4">FN</td><td>MobileNet-V3-S MobileNet-V3-L</td><td>0.19</td><td>0.25</td><td></td><td>0.26</td><td></td><td>0.26</td></tr><tr><td>ResNet-18</td><td>0.20 0.26</td><td>0.26 0.30</td><td>0.26 0.27</td><td>0.20 0.25</td><td>0.20 0.32</td><td>0.23</td></tr><tr><td>ResNet-34</td><td>0.21</td><td>0.24</td><td>0.25</td><td>0.24</td><td>0.23</td><td>0.31</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.30</td></tr></table>

## B Distance Measures

Our loss terms defined in Eq. (3) and Eq. (4) require a distance measure. We provide the evaluation of diferent distance functions in Tab. A.2. As we reported in the main paper, we find that smooth L1 performs best in our case. MAE-based distance functions outperforming MSE-based ones in our case can be explained by the fact that the predicted weights per step are all below ${ \mathrm { ~ } } _ { } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ ~ }  \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ } \mathrm { ~ } \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ ~ } \mathrm \mathrm { ~ ~ } \mathrm \mathrm { ~ ~ ~ } \mathrm \mathrm \mathrm { ~ ~ ~ } \mathrm \mathrm \mathrm { ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ ~ ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm { ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ } \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm \mathrm $ so the quadratic penalty actually reduces their magnitude.