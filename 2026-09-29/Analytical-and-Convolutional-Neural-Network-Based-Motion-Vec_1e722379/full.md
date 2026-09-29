# Analytical and Convolutional Neural Network-Based Motion-Vector Propagation for Efficient Video Object Detection

Ashiyana Abdul Majeed, Member, IEEE,, Mahmoud Meribout, Senior Member, IEEE, and Neethu Joseph

Abstract—Continuous video analytics requires accurate localization at low latency within embedded power budgets. This paper presents a hardware-software design methodology that reuses codec motion vectors (MVs) between detector invocations. Two alternative models support translation and scale changes: analytical motion-vector propagation (Analytical-MV) and learned propagation using a convolutional neural network (CNN) (CNN-MV). The learned model uses convolutional operations and independent object updates suited to parallel execution on an edge graphics processing unit (GPU). Analytical-MV combines a harmonic-mean precision-recall score (F1) of 0.909 with a mean end-to-end latency of 9.03 ms and an energy of 0.177 J per frame, yielding the lowest latency and energy among the evaluated configurations. Relative to detection on every frame, it reduces mean latency by 25.9% and energy per frame by 36.4%. CNN-MV offers a different trade-off: its fastest configuration raises recall from Analytical-MV’s 0.871 to 0.890, and lowers mean power from 19.64 to 17.32 W, at 18.42 ms latency and 0.319 J/frame. It is therefore useful when recall or operating power matters more than minimum latency and energy. Execution on a deep learning accelerator (DLA) further reduces time-averaged GPU utilization relative to GPU execution. Hostprocessing optimization substantially improves both latency and energy, demonstrating the value of jointly designing temporal models and their execution pipelines.

Index Terms—Video object detection, intelligent traffic monitoring, physical AI, edge computing, compressed-domain video analysis, motion vectors, energy efficiency.

## I. INTRODUCTION

Continuous video object detection is an important component of intelligent traffic-monitoring infrastructure, where roadside edge devices must localize vehicles under strict latency, energy, and thermal constraints. Similar constraints arise in physical-AI systems such as autonomous vehicles, inspection drones, and mobile robots, which continuously process video while sharing limited computational resources with control, planning, and communication workloads. These tasks require frame-level localization at video rate within the compute, power, and thermal limits of compact edge devices. Processing every frame independently repeats substantial work because neighboring frames often contain the same objects. The resulting costs extend beyond neural inference to preprocessing and data movement, making complete-pipeline efficiency essential for sustained operation.

Temporal reuse offers a way to reduce this repeated computation. Video codecs already encode motion vectors (MVs) for inter-frame prediction [1], and compressed-domain learning has demonstrated its value for visual analysis [2]. Reusing this information can reduce detector workload while maintaining frame-by-frame outputs. The challenge is to preserve localization quality while limiting the cost of preparing motion information and executing updates.

This paper adopts a hardware-software design methodology for edge graphics processing units (GPUs). Its artificial intelligence (AI) model for bounding-box estimation is a convolutional neural network (CNN) designed around GPUfriendly convolutional operations and independent updates of objects. This structure exposes parallel work across objects and allows the model, central processing unit (CPU) processing, scheduling, and accelerator assignment to be considered together. The design aims to utilize the available hardware resources efficiently, with its effectiveness evaluated through end-to-end pipeline measurements.

The framework supports two alternative bounding-box estimation methods: analytical motion-vector propagation (Analytical-MV) and CNN-based motion-vector propagation (CNN-MV). Both methods account for object translation and scale variation, including their simultaneous occurrence. Fullframe detector refreshes provide the mechanism for discovering newly visible objects, including those entering from any of the four image borders, whereas propagation updates only previously detected objects.

The main contributions are:

• Analytical motion model. A coupled translation-scale estimator propagates the bounding boxes of previously detected objects between full-frame detector refreshes. The refresh mechanism enables the discovery of newly appearing objects.

• Learned motion model. An object-local CNN provides an alternative approach for propagating bounding boxes within the same detector-refresh framework.

• Hardware-aware model and pipeline design. GPUfriendly convolutions and independent object updates are combined with optimized host processing and evaluated under alternative scheduling and accelerator assignments.

• Measured operating trade-offs. Complete-pipeline accuracy, latency, power, energy, and utilization measurements identify the analytical method’s efficiency advantage and the conditions favoring learned propagation.

## II. RELATED WORK

Reducing spatial-temporal redundancy for efficient edgebased industrial video analytics (RESPIRE) [3] reduces redundant transmission and processing in industrial edge video analytics. Context-aware detection [4] and collaborative edge intelligence [5] address complementary recognition and scheduling problems. Our focus is on local box propagation and its complete execution cost.

Majeed et al. [6] review heterogeneous edge scheduling, emphasizing memory contention, accelerator compatibility, and transition costs. Sali et al. [7] review real-time object detectors and associated hardware for autonomous vehicles. These surveys motivate joint algorithm-hardware evaluation; our contribution is a measured temporal-propagation pipeline.

Deep feature flow (DFF) [8] warps key-frame features using optical flow, reporting 73.1% mean average precision (mAP) at 20.25 frames per second (FPS) on ImageNet video object detection (VID), using an NVIDIA K40 GPU and Intel i7-4790 CPU. Flow-guided feature aggregation (FGFA) [9] aggregates aligned neighboring features, reaching 76.3% mAP at 733 ms/frame with a 101-layer residual network (ResNet-101) on a K40 GPU and Intel E5-2670 v2 CPU. Our box-level updates avoid optical-flow inference and feature warping.

Mobile temporal detectors [10], [11] show that recurrent inference can be practical on constrained hardware. Looking Fast and Slow [11] interleaves feature extractors with long shortterm memory (LSTM) memory; its quantized asynchronous configuration reports 59.3% mAP at 72.3 FPS on a Pixel 3 phone. CNN-MV instead uses explicit box history and codec vectors without a recurrent hidden state.

Motion-vector propagation (MVP) [12] uses deterministic motion-vector grids to propagate key-frame detections, reporting 60.9% mAP at an intersection over union (IoU) threshold of 0.5 and 10.3 FPS on an NVIDIA RTX 3090 GPU. Its separate translation and scale tests restrict accepted motion. Analytical-MV extends this path with occupied-cell masking and a coupled translationscale fit; the controlled comparison is discussed in Section VI. See Without Decoding [13] combines box-aligned features with recurrent refinement for compressed-video tracking. Section VI examines its performance and the distinction from our feed-forward model and embedded measurements.

Because these studies use different datasets and timing protocols, their reported values are not directly comparable with ours. Instead, they motivate the joint treatment of propagation, preprocessing, and resource assignment that follows.

## III. METHODOLOGY

## A. Pipeline Overview

The system alternates full-frame detection with motionvector propagation (Fig. 1). Detection initializes boxes and refreshes them on intra-coded frames (I-frames) or propagation failure, subject to the triggers below. Analytical-MV runs on the CPU; CNN-MV uses CPU preprocessing followed by GPU or deep learning accelerator (DLA) inference. Both paths validate boxes before emitting each frame’s output.

Let $B _ { t } = [ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ denote an object’s bounding box in frame t, and let $M _ { t + 1 }$ denote the codec motion vectors associated with the transition to frame $t + 1$ . Analytical-MV estimates box displacement and scale analytically; CNN-MV predicts a candidate box from motion in the object’s region of interest (ROI) and recent geometry. Both depend on the codec’s motion quality and may accumulate errors under occlusion, deformation, or abrupt motion; neither can introduce a previously unseen object without a detector refresh. For CNN-MV, the deployed TensorRT configuration uses the two most recent boxes when history is available:

$$
\tilde { B } _ { t + 1 } = g _ { \theta } ( \bar { B } _ { t - 1 } , \bar { B } _ { t } , \bar { M } _ { t + 1 } ^ { \mathrm { R O I } } ) ,\tag{1}
$$

where $\bar { B } _ { t - 1 }$ and ${ \bar { B } } _ { t }$ are normalized box-history inputs and $\bar { M } _ { t + 1 } ^ { \mathrm { R O I } }$ is the normalized and padded ROI-filtered motionvector tensor. The recorded runtime interprets the four output values as a normalized xyxy candidate box, then validates, clips, or rejects it before converting it back to pixel coordinates.

## B. Analytical Propagation

The analytical path summarizes backward-reference codec motion vectors over a $3 \times 3$ grid within each previous-frame box. Raw codec vectors are converted to pixel-domain source and destination centers, and $\mathbf { d } _ { i }$ is defined as the forward displacement from the reference-frame block center to the current-frame block center; for example, a one-pixel rightward block motion gives $d _ { x , i } ~ = ~ + 1$ . The unmodified MVP rule is retained as a validation reference, while Analytical-MV excludes empty cells rather than interpreting them as zero motion. Let O be the set of occupied cells, $\mathbf { g } _ { i } = [ g _ { x , i } , g _ { y , i } ] ^ { \mathsf { T } }$ the center of cell i relative to the bounding-box center, and $\mathbf { d } _ { i } = [ d _ { x , i } , d _ { y , i } ] ^ { \mathsf { T } }$ the mean forward displacement in that cell. Propagation is attempted only when $| \mathcal { O } | \geq 2$

The updated method first tests pure translation. It computes $\begin{array} { r } { \bar { \bf d } = | \bar { \mathcal { O } } | ^ { - 1 } \sum _ { i \in \mathcal { O } } { \bf d } _ { \mathrm { ~ } } } \end{array}$ <sub>i</sub> and accepts the translation when the standard deviations of both displacement components do not exceed $\tau _ { \mathrm { t r } } = 1$ pixel. If this test fails, a pure uniform-scale hypothesis about the box center is evaluated using the offcenter occupied cells:

$$
r _ { i } = \frac { \| \mathbf { g } _ { i } + \mathbf { d } _ { i } \| _ { 2 } } { \| \mathbf { g } _ { i } \| _ { 2 } } .\tag{2}
$$

The hypothesis is accepted when std $. ( r _ { i } ) < \tau _ { \mathrm { s c } } = 0 . 1$ , with scale $s = \mathrm { m e a n } ( r _ { i } )$ and zero translation.

When neither test explains the motion, we implement a coupled translation and uniform-scale model test,

$$
\mathbf { d } _ { i } = \mathbf { t } + \alpha \mathbf { g } _ { i } , \qquad s = 1 + \alpha ,\tag{3}
$$

where $\textbf { t } = [ t _ { x } , t _ { y } ] ^ { \top }$ . The three unknowns $( t _ { x } , t _ { y } , \alpha )$ are obtained by linear least squares over the occupied cells. A rank-deficient system is rejected. For a full-rank fit, the pixeldomain residual is

$$
\varepsilon _ { \mathrm { f i t } } = \sqrt { \frac { 1 } { | \mathcal { O } | } \sum _ { i \in \mathcal { O } } \left\| \mathbf { d } _ { i } - \left( \mathbf { t } + \alpha \mathbf { g } _ { i } \right) \right\| _ { 2 } ^ { 2 } } ,\tag{4}
$$

and propagation is accepted when $\varepsilon _ { \mathrm { f i t } } \le \tau _ { \mathrm { f i t } }$ , with $\tau _ { \mathrm { f i t } } = 1 0$ pixels in the evaluated implementation. The accepted translation updates the box center, the accepted scale multiplies its width and height, and the resulting normalized box is clamped to the image bounds. If any active box lacks sufficient occupied cells or fails all motion tests, propagation for that frame is rejected, and the object detector is invoked.

## C. Object-Local Motion-Vector Selection

CNN-MV preprocessing first filters valid backwardreference codec vectors, fits a robust global similarity transform between source and destination block centers, subtracts that transform to obtain residual motion, and retains coherent residual vectors. The model’s residual addition combines the predicted box delta with the previous normalized box; the runtime receives the resulting candidate box, not a delta to add again. Per-object ROI selection then reduces each retained record to eight geometric features and pads or truncates the result to the TensorRT binding $\mathrm { m } \mathrm { v } = ( 1 , 1 , 2 0 0 0 , 8 )$ . These full-frame and per-object operations account for substantial host work before the compact CNN executes. A representative record is

$$
[ w , h , x _ { s } , y _ { s } , x _ { d } , y _ { d } , m _ { x } , m _ { y } ] ,\tag{5}
$$

where w and h are block dimensions, $( x _ { s } , y _ { s } )$ is the source block center, $( x _ { d } , y _ { d } )$ is the destination block center, and $( m _ { x } , m _ { y } )$ is the motion vector or residual motion after background compensation. Non-geometric bookkeeping fields from parser outputs are not supplied to the deployed network. The recorded configuration uses source-coordinate ROI selection: a vector row is retained when its configured source center lies inside the previous box,

$$
x _ { 1 } \leq x _ { s } \leq x _ { 2 } , \qquad y _ { 1 } \leq y _ { s } \leq y _ { 2 } .\tag{6}
$$

Each retained feature is standardized as $( z _ { j } - \mu _ { j } ) / \sigma _ { j }$ using training-set statistics stored in the model configuration; only box coordinates are divided by frame dimensions. If fewer than 2000 vectors remain, the tensor is zero-padded; otherwise, it is truncated to the first 2000 rows. The four-value-normalized candidate box is then subjected to the validation gates below.

![](images/7aec7019f2444ce3e0668108e1bd2740a91860d9dadb3745cc52a91d7a616317.jpg)  
Fig. 2. CNN-MV architecture verified against the model definition and exported graph: 10,548 learned parameters. Dimensions are batch, channels, height, and width. Activation and spatial-reduction details are specified in Section III-D.

## D. CNN-MV Propagation Model

The DLA- and GPU-targeted TensorRT engines accept box history ${ \mathrm { b b } } = ( 1 , 8 , 1 , 1 )$ and ROI motion mv=(1,1,2000,8), returning the normalized candidate box $\mathtt { b l o o x } = ( 1 , 4 , 1 , 1 )$ ). History concatenates bbox[t-2] and bbox[t-1]; the current box fills a missing slot from older history.

Figure 2 summarizes the 10,548-parameter CNN-MV model, verified against its definition and exported Open Neural Network Exchange (ONNX) graph. Each box-branch convolution is followed by a rectified linear unit (ReLU). Each motionbranch convolution uses unit stride and one-pixel padding, followed by ReLU and 2 × 2 max pooling with stride 2, yielding spatial dimensions $2 0 0 0 \times 8 \to 1 0 0 0 \times 4 \to 5 0 0 \times 2$ Three $5 \times 1$ average pools with stride (5, 1) reduce this to $4 \times 2 ;$ nearest-neighbor width upsampling produces $4 \times 4 ,$ followed by $4 \times 4$ average pooling to $1 \times 1$ . The first three head convolutions use ReLU; the final linear output is added to the latest box: $\tilde { B } _ { t + 1 } = \bar { B } _ { t } + \Delta \bar { B } _ { t + 1 }$

![](images/bd1825e4e30c05b81a295635eee6d70d43aa69d6d20c509b1095eb5430fe9b89.jpg)  
Fig. 1. Motion-vector-assisted video analytics pipeline. The detector path refreshes boxes on I-frames or tracking failures. The motion path uses the CPU Analytical-MV engine or CPU preprocessing followed by CNN-MV inference on DLA/GPU.

Both TensorRT engines use 16-bit floating-point precision (FP16); the supplied training script uses 32-bit floatingpoint precision (FP32) tensors without automatic mixed precision. The exported graph requires 5,769,088 convolutional multiply-accumulate operations (MACs) per object, equivalent to 11,538,176 floating-point operations (FLOPs) when the multiply and addition counts are counted separately. These counts exclude bias addition, activations, pooling, resizing, residual addition, and host preprocessing.

With fixed padded tensors, propagation scales approximately as $O ( N _ { B } )$ for $N _ { B }$ active boxes. Independent object updates permit batching and concurrent scheduling; parallel variants use 16 motion-engine contexts (Table I). DLA execution requests core 0, with no GPU fallback layers listed in the FP16 engine-creation log. This is build-time partition evidence, not layer-level runtime profiling.

## E. Refresh, Validation, and Fallback Logic

Propagation updates existing boxes; only a full-frame detector refresh can initialize newly visible objects, including those entering at any image border. Because the evaluated system lacks a dedicated border-search detector, an entering object may remain undetected until the next refresh. The CNN-MV runners invoke the detector on I-frames, on the first detection frame, after five consecutive frames without detections, and when optional periodic refresh is enabled; the recorded runs use the default interval of 0.

On propagation frames, the stationary gate holds a box when residual motion intensity per pixel is below $1 0 ^ { - 4 }$ ; otherwise, CNN-MV proposes a candidate. A candidate is rejected if it is non-finite, empty, less than two pixels wide or high after clipping, less than 65% visible, outside the area-ratio range [0.3, 3.5] relative to the previous box, or shifted by more than the larger of 70 pixels and 1.75 times the previous-box diagonal. Rejected candidates retain the previous box with a score decay of 0.98. Full-frame fallback occurs when the invalid-candidate ratio reaches 0.60 or when the propagated area changes by more than 2.5 relative to the last full-frame detections. The measured tables use this recorded 0.60 policy and exclude results from the later 0.80 policy.

## IV. VALIDATION

The University at Albany Detection and Tracking benchmark (UA-DETRAC) traffic-video benchmark [14] is used to evaluate the framework in a controlled traffic-monitoring setting. Its fixed-camera road scenes represent an edge-based intelligent transportation application in which continuous vehicle localization must be performed within limited compute and power budgets. The same propagation and heterogeneousexecution principles may also apply to autonomous vehicles, inspection drones, and mobile robots; however, validation under moving-camera conditions is outside the scope of the present study. Version 11 of You Only Look Once (YOLO) (YOLOv11) is the experimental key-frame detector and is not part of the propagation architecture. Other detectors that return conventional boxes could be substituted, but would require separate accuracy, refresh policy, TensorRT, and timing validation.

TABLE I  
CONFIGURATION IDENTIFIERS (IDS). CNN-MV SHARES ONECHECKPOINT AND VALIDATION POLICY. PRE/POST DENOTESPREPROCESSING/POSTPROCESSING; SHADING MARKS THE PRINCIPALOPERATING POINTS.
<table><tr><td>ID</td><td>Method</td><td>Motion execution</td><td>Host implementation</td></tr><tr><td>CO</td><td>YOLO-only</td><td>None</td><td>Detector on every frame</td></tr><tr><td>C1</td><td>MVP-YOLO (adapted reference)</td><td>CPU analytical</td><td>Original MVP grid propagation</td></tr><tr><td></td><td>C2 Analytical-MV (ours)</td><td>CPU analytical</td><td>Masked grid plus coupled translationscale fit</td></tr><tr><td>C3</td><td>CNN-MV (ours)</td><td>DLA</td><td>Serial original</td></tr><tr><td>C4</td><td>CNN-MV (ours)</td><td>DLA</td><td>Serial optimized pre/post</td></tr><tr><td>C5</td><td>CNN-MV (ours)</td><td>DLA</td><td>Parallel original</td></tr><tr><td>C6</td><td>CNN-MV (ours)</td><td>DLA</td><td>Parallel optimized pre/post</td></tr><tr><td>C7</td><td>CNN-MV (ours)</td><td>GPU</td><td>Serial original</td></tr><tr><td></td><td>C8 CNN-MV (ours)</td><td>GPU</td><td>Serial optimized pre/post</td></tr><tr><td>C9</td><td>CNN-MV (ours)</td><td>GPU</td><td>Parallel original</td></tr><tr><td></td><td>C10 CNN-MV (ours)</td><td>GPU</td><td>Parallel optimized pre/post</td></tr></table>

MVP-YOLO adapts the released MVP implementation [12] using the common YOLO TensorRT engine while retaining its original, untrained grid-propagation rule. It is distinct from our Analytical-MV and CNN-MV methods.

## A. Videos and Baselines

Table I defines the 11 configurations evaluated on the same 25-video subset. A YOLOv11 detector trained on UA-DETRAC serves as the common full-frame detector. All configurations share the same videos, annotations, class list, trained YOLOv11 TensorRT engine, a confidence threshold of 0.5, an non-maximum suppression (NMS) IoU of 0.45, a maximum of 300 detections, and a 20-frame warm-up. Consequently, differences among the configurations arise from choices in propagation, routing, and execution rather than from the detector. Each benchmark processes 28,177 frames; removing 20 warm-up frames from each video leaves 27,677 timed rows. Accuracy is evaluated on 27,893 annotated frames.

## B. Training

CNN-MV training samples are constructed from six UA-DETRAC sequences that are distinct from all 25 sequences used for the final evaluation. Samples contain adjacent annotated boxes and motion-vector files, with frame-normalized box coordinates, pairwise IoU of at least 0.25, and matching classes. Within each of the six training sequences, samples are randomly divided using seed 7 and a validation fraction of 0.2, yielding 40,167 training and 10,042 validation samples. Although this internal validation split is sample-level and may contain temporally adjacent samples across its partitions, the final 25-video system evaluation is sequence-disjoint from the CNN-MV training data.

Training uses the Adam optimizer for 100 epochs, a batch size of 64, a learning rate of 0.001, zero weight decay, Smooth L1 loss with beta 1.0, 2000 motion-vector rows, source-coordinate ROI selection, residual-vector postprocessing, and stationary-box gating at $1 0 ^ { - 4 } .$ . The best checkpoint occurs at epoch 94, with validation loss $1 . 3 0 4 9 \times 1 0 ^ { - 6 }$ . The training script, checkpoint configuration, exported graph, and normalization statistics document the model configuration; a reproducible release should also include the sample-split manifests.

## C. Reference Protocols

Predictions are evaluated against UA-DETRAC annotations using class-agnostic one-to-one vehicle matching at $\mathrm { I o U } \geq 0 . 5$ processed in descending IoU order. The logged car, van, bus, and others categories are collapsed before matching, and predictions with at least 50% of their area inside an official ignore region are excluded. Metrics comprise precision, recall, the harmonic mean of precision and recall (F1) score, mean matched IoU, and counts of true positives (TPs), false positives (FPs), and false negatives (FNs).

Routing counts use all 28,177 processed frames; stationary holds count as MV-routed. YOLO-only invokes the detector on every frame.

Standard average precision (AP) at IoU 0.50 and 0.75 $( \mathbf { A P } _ { 5 0 } , \mathbf { A P } _ { 7 5 } ) .$ , mAP over 0.500.95 $( \mathrm { m A P _ { 5 0 : 9 5 } } ) _ { : }$ , and per-class AP require confidence sweeps. Retained scores are truncated at 0.5, and non-CNN prediction archives are incomplete, precluding those metrics. Reruns must preserve routing, retain low-score predictions, and use class-aware, confidence-ranked matching.

## D. On-Device Measurement Setup

On-device experiments use an NVIDIA Jetson AGX Orin Developer Kit with TensorRT 10.3. The live reader decodes the stored UA-DETRAC videos and extracts codec vectors, so decoding remains within the measured pipeline; the experiments do not demonstrate a decoder bypass. Under MAX\_FREQ setting, the recorded CPU, GPU, external memory controller (EMC), and DLA maxima are 2.2016 GHz, 1.3005 GHz, 3.199 GHz, and 1.6 GHz, respectively.

The common YOLO detector and all CNN-MV engines use FP16 inference. YOLO exposes images=(1,3,960,960) and output0=(1,8,18900). DLA variants request core 0, while GPU variants use a separately built plan for the same model. The DLA engine-creation log reports no GPU fallback layers [15]; YOLO remains on the GPU in every configuration.

Improved host processing batches active boxes, vectorizes residual-neighbor construction, batches box validation and history matching, reuses geometry, and removes intermediate work without changing model inputs or routing. Parallel improved variants add multithreaded per-object scheduling. Random sample consensus (RANSAC)/iteratively reweighted least squares (IRLS) preprocessing uses a 2.5-pixel fit threshold, at least 20 inliers, a 1.5-pixel residual threshold, a 2- pixel residual-coherence tolerance, support of at least 3 vectors (including the candidate) in its grid cell and neighboring cells, and an empty fallback when no coherent residual support remains.

Each method runs after a reboot. The measured -no-video pass begins timing at local frame read/decode and ends after detection postprocessing. Tegrastats [16] is active only from the first post-warm-up frame through final measured-frame postprocessing. Model loading, warm-up, annotation drawing, Moving Picture Experts Group (MPEG)-4 Part 14 (MP4) video-file generation, and result serialization are excluded; annotated videos are generated in a separate unmonitored pass.

Mean power sums complete samples from VDD\_GPU\_SOC (GPU and system on chip (SoC)), VDD\_CPU\_CV (CPU and computer vision (CV) engines), and VIN\_SYS\_5V0 (5-volt system input) within the measured window. Energy per frame is mean rail-sum power divided by effective throughput, without idle subtraction, and is not wall-socket energy. For stage estimates, each frame is assigned the nearest telemetry sample. Stage energy is the mean of frame-matched power multiplied by stage duration; stage power is the corresponding durationweighted mean. These allocations are not independent electrical measurements of individual stages. YOLO and motion inference are subcomponents of inference/compute and must not be added to it again.

## V. RESULTS

All results use the configuration IDs in Table I. Tables separate the non-CNN configurations (C0-C2), CNN-MV/DLA (C3-C6), and CNN-MV/GPU (C7-C10); shaded rows identify Analytical-MV (C2) and serial optimized CNN-MV/GPU (C8). Latency is summarized by the mean, 95th percentile (P95), and 99th percentile (P99); FPS denotes throughput, and FPS/W denotes throughput per watt.

## A. Accuracy and Routing

YOLO-only leads in F1 (0.933), followed by MVP-YOLO (0.914). Analytical-MV reaches 0.909, trading approximately 2.4 percentage points for efficiency. CNN-MV variants cluster around precision/recall/F1 of 0.934/0.890/0.911. Relative to Analytical-MV, they gain approximately 1.9 percentage points of recall and 0.2 percentage points of F1, with lower precision and matched IoU (Table II).

MVP-YOLO propagates 25.3% of frames, Analytical-MV 52.0%, and CNN-MV 44.1-44.5%, including stationary holds under the 0.60 fallback policy. Equivalently, Analytical-MV reduces the fraction of detector-processed frames from 100% for framewise YOLO and 74.7% for MVP-YOLO to 48.0%, while CNN-MV reduces it to approximately 55.5%. The different routing rates are intentional outcomes of the complete propagation and fallback policies. These comparisons, therefore, characterize complete-system operating points rather than propagation accuracy under a matched detector budget.

## B. Latency and Per-Video Variation

Analytical-MV leads at 110.69 FPS and 9.03 ms mean latency: throughput improves by 35.0%, and latency falls by 25.9% relative to YOLO-only. Its P95 is nevertheless higher (16.90 versus 13.28 ms), reflecting expensive detectorrecovery frames (Table III).

TABLE II  
ACCURACY AND ROUTING FOR THE 11 CONFIGURATIONS. EVERY CONFIGURATION USES 25 VIDEOS, 27,893 ANNOTATED FRAMES, AND 28,177 PROCESSED FRAMES. PREC. DENOTES PRECISION; IOU IS AVERAGED OVER MATCHED PAIRS. DETECTOR AND MV PERCENTAGES USE THE PROCESSED-FRAME TOTAL.
<table><tr><td>ID</td><td colspan="4">Accuracy</td><td colspan="3">Detection counts</td><td colspan="2">Detector routing</td><td colspan="2">MV routing</td></tr><tr><td></td><td>Prec.</td><td>Recall</td><td>F1</td><td>IoU</td><td>TP</td><td>FP</td><td>FN</td><td>Frames</td><td>%</td><td>Frames</td><td>%</td></tr><tr><td>C0</td><td>0.957</td><td>0.910</td><td>0.933</td><td>0.904</td><td>197,005</td><td>8,940</td><td>19,522</td><td>28,177</td><td>100.0</td><td>0</td><td>0.0</td></tr><tr><td>Cl</td><td>0.957</td><td>0.874</td><td>0.914</td><td>0.901</td><td>189,343</td><td>8,594</td><td>27,184</td><td>21,048</td><td>74.7</td><td>7,129</td><td>25.3</td></tr><tr><td>C2</td><td>0.951</td><td>0.871</td><td>0.909</td><td>0.880</td><td>188,543</td><td>9,708</td><td>27,984</td><td>13,515</td><td>48.0</td><td>14,662</td><td>52.0</td></tr><tr><td>C3</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,795</td><td>13,730</td><td>23,732</td><td>15,647</td><td>55.5</td><td>12,530</td><td>44.5</td></tr><tr><td>C4</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,795</td><td>13,730</td><td>23,732</td><td>15,647</td><td>55.5</td><td>12,530</td><td>44.5</td></tr><tr><td>C5</td><td>0.933</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,814</td><td>13,796</td><td>23,713</td><td>15,739</td><td>55.9</td><td>12,438</td><td>44.1</td></tr><tr><td>C6</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,795</td><td>13,730</td><td>23,732</td><td>15,646</td><td>55.5</td><td>12,531</td><td>44.5</td></tr><tr><td>C7</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,793</td><td>13,732</td><td>23,734</td><td>15,646</td><td>55.5</td><td>12,531</td><td>44.5</td></tr><tr><td>C8</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,794</td><td>13,731</td><td>23,733</td><td>15,646</td><td>55.5</td><td>12,531</td><td>44.5</td></tr><tr><td>C9</td><td>0.933</td><td>0.891</td><td>0.911</td><td>0.874</td><td>192,835</td><td>13,775</td><td>23,692</td><td>15,739</td><td>55.9</td><td>12,438</td><td>44.1</td></tr><tr><td>C10</td><td>0.934</td><td>0.890</td><td>0.911</td><td>0.874</td><td>192,793</td><td>13,732</td><td>23,734</td><td>15,646</td><td>55.5</td><td>12,531</td><td>44.5</td></tr></table>

TABLE III

COMPLETE-PIPELINE TIMING FOR THE 11 CONFIGURATIONS. LATENCIES ARE MILLISECONDS: THROUGHPUT EXCLUDES 20 WARM-UP FRAMES PER VIDEO. ENGINE IS THE MEAN PER-CALL ENGINE LATENCY, COMPUTE IS SYNCHRONIZED PER-FRAME INFERENCE TIME, AND CALLS/FRAME IS THE AVERAGE NUMBER OF DETECTOR OR MOTION-ENGINE INVOCATIONS. FOR CNN-MV, ENGINE FPS IS BASED ON COMPUTE BECAUSE MULTIPLE MOTION-ENGINE CALLS MAY OCCUR PER FRAME.
<table><tr><td rowspan="2">ID</td><td colspan="2">Throughput (FPS)</td><td colspan="3">End-to-end latency (ms)</td><td colspan="2">Inference time (ms)</td><td rowspan="2">Calls/ frame</td></tr><tr><td>Pipeline</td><td>Engine</td><td>Mean</td><td>P95</td><td>P99</td><td>Engine/call</td><td>Compute/frame</td></tr><tr><td>CO</td><td>82.00</td><td>249.31</td><td>12.20</td><td>13.28</td><td>13.68</td><td>4.01</td><td>4.01</td><td>1.000</td></tr><tr><td>C1</td><td>85.33</td><td>247.99</td><td>11.72</td><td>15.90</td><td>18.39</td><td>4.03</td><td>4.25</td><td>0.747</td></tr><tr><td>C2</td><td>110.69</td><td>249.16</td><td>9.03</td><td>16.90</td><td>18.24</td><td>4.01</td><td>3.44</td><td>0.479</td></tr><tr><td>C3</td><td>11.67</td><td>193.10</td><td>85.71</td><td>124.63</td><td>138.71</td><td>0.37</td><td>5.18</td><td>6.017</td></tr><tr><td>C4</td><td>48.46</td><td>194.94</td><td>20.64</td><td>36.20</td><td>41.55</td><td>0.37</td><td>5.13</td><td>6.017</td></tr><tr><td>C5</td><td>11.70</td><td>171.53</td><td>85.44</td><td>121.97</td><td>134.85</td><td>0.36</td><td>5.83</td><td>6.213</td></tr><tr><td>C6</td><td>51.96</td><td>178.94</td><td>19.25</td><td>32.06</td><td>36.44</td><td>0.37</td><td>5.59</td><td>6.017</td></tr><tr><td>C7</td><td>12.05</td><td>339.11</td><td>82.99</td><td>118.24</td><td>129.95</td><td>0.37</td><td>2.95</td><td>6.017</td></tr><tr><td>C8</td><td>54.29</td><td>339.08</td><td>18.42</td><td>30.30</td><td>33.59</td><td>0.37</td><td>2.95</td><td>6.017</td></tr><tr><td>C9</td><td>11.84</td><td>191.70</td><td>84.46</td><td>120.72</td><td>132.90</td><td>0.36</td><td>5.22</td><td>6.214</td></tr><tr><td>C10</td><td>53.08</td><td>198.09</td><td>18.84</td><td>31.52</td><td>35.45</td><td>0.37</td><td>5.05</td><td>6.017</td></tr></table>

Serial-improved GPU execution is the fastest CNN-MV variant (54.29 FPS, 18.42 ms mean, 30.30 ms P95). Against serial-improved DLA execution, it reduces mean latency by 10.7%, P95 by 16.3%, and energy by 11.4%, with an F1 difference below 0.00001. Parallel scheduling improves optimized DLA latency by 6.7% to 19.25 ms but slightly slows optimized GPU execution. Original CNN-MV variants remain host-bound at 11.67-12.05 FPS.

Table IV separates propagation from detector recovery/fallback: mean costs are 3.68 versus 14.99 ms for Analytical-MV and 11.63 versus 24.32 ms for serial-improved GPU CNN-MV.

For the serial-improved GPU CNN-MV, per-video throughput ranges from 38.73-93.58 FPS, F1 ranges from 0.466- 0.967, and energy ranges from 0.166-0.453 J/frame. The optimized serial DLA variant shares the lowest- and highest-F1 sequences, MVI\_40141 and MVI\_40131. Figure 3 compares their outputs with YOLO-only, illustrating the variation hidden by aggregate metrics.

## C. Power, Energy, and Utilization

Under the rail-sum measurement protocol, optimized CNN-MV draws 17.20-17.79 W, below all non-CNN configurations. Original CNN-MV draws only 13.58-14.14 W, but prolonged host processing raises energy to 1.127-1.212 J/frame (Table V).

Serial-improved GPU CNN-MV leads its family at 0.319 J/frame and 3.134 FPS/W; Analytical-MV remains better at 0.177 J/frame and 5.637 FPS/W. Serial-improved DLA execution lowers mean GPU utilization from 15.5% to 11.8% (Table VI), a proxy for headroom that requires a concurrentworkload experiment to assess its sharing benefit.

TABLE IV  
ROUTE-SPECIFIC TIMING IN MILLISECONDS. MOTION IS THE PER-FRAME MOTION-COMPUTE CONTRIBUTION; PROPAGATION AND RECOVERY/FALLBACKCOLUMNS REPORT COMPLETE-FRAME LATENCY CONDITIONED ON THE ROUTE. YOLO IS MEAN DETECTOR-ENGINE LATENCY PER CALL. CALLS/FRAMEINCLUDES DETECTOR AND MOTION-ENGINE INVOCATIONS; C0 HAS NO PROPAGATION ROUTE.
<table><tr><td>ID</td><td>YOLO</td><td colspan="2">Motion compute</td><td colspan="2">Propagation route</td><td colspan="2">YOLO recovery/fallback</td><td>Calls/</td></tr><tr><td></td><td>Mean</td><td>Mean</td><td>P95</td><td>Mean</td><td>P95</td><td>Mean</td><td>P95</td><td>frame</td></tr><tr><td>Cl</td><td>4.03</td><td>1.24</td><td>3.74</td><td>3.93</td><td>8.25</td><td>14.43</td><td>16.34</td><td>0.747</td></tr><tr><td>C2</td><td>4.01</td><td>1.52</td><td>3.32</td><td>3.68</td><td>6.65</td><td>14.99</td><td>17.66</td><td>0.479</td></tr><tr><td>C3</td><td>4.00</td><td>2.96</td><td>8.05</td><td>76.83</td><td>92.85</td><td>94.19</td><td>130.32</td><td>6.017</td></tr><tr><td>C4</td><td>4.01</td><td>2.91</td><td>7.97</td><td>13.40</td><td>19.79</td><td>27.07</td><td>38.30</td><td>6.017</td></tr><tr><td>C5</td><td>4.00</td><td>3.60</td><td>8.69</td><td>76.80</td><td>91.31</td><td>93.34</td><td>127.11</td><td>6.213</td></tr><tr><td>C6</td><td>4.02</td><td>3.36</td><td>8.18</td><td>12.36</td><td>16.79</td><td>25.29</td><td>33.89</td><td>6.017</td></tr><tr><td>C7</td><td>4.01</td><td>0.72</td><td>2.03</td><td>74.66</td><td>87.94</td><td>90.83</td><td>122.94</td><td>6.017</td></tr><tr><td>C8</td><td>3.99</td><td>0.74</td><td>2.03</td><td>11.63</td><td>15.49</td><td>24.32</td><td>31.64</td><td>6.017</td></tr><tr><td>C9</td><td>4.02</td><td>2.97</td><td>7.62</td><td>75.88</td><td>89.84</td><td>92.22</td><td>125.54</td><td>6.214</td></tr><tr><td>C10</td><td>4.00</td><td>2.83</td><td>7.34</td><td>11.87</td><td>16.20</td><td>24.90</td><td>33.23</td><td>6.017</td></tr></table>

TABLE V

MEASURED RAIL POWER AND ESTIMATED ENERGY EFFICIENCY. TOTAL POWER IS THE THREE-RAIL SUM. FRAME MEAN ASSIGNS THE NEARESTTELEMETRY SAMPLE TO EACH TIMED FRAME; ENERGY USES THE MEASURED-WINDOW MEAN DIVIDED BY PIPELINE FPS, WITH NO IDLE SUBTRACTION.
<table><tr><td>ID</td><td colspan="2">Total power (W)</td><td colspan="3">Mean rail power (W)</td><td>Frame mean</td><td colspan="2">Energy efficiency</td></tr><tr><td></td><td>Mean</td><td>P95</td><td>GPU-SoC</td><td>CPU-CV</td><td>5V</td><td>(W)</td><td>J/frame</td><td>FPS/W</td></tr><tr><td>CO</td><td>22.88</td><td>23.54</td><td>12.98</td><td>3.19</td><td>6.70</td><td>22.90</td><td>0.279</td><td>3.584</td></tr><tr><td>Cl</td><td>20.60</td><td>21.73</td><td>11.28</td><td>3.00</td><td>6.33</td><td>20.50</td><td>0.241</td><td>4.142</td></tr><tr><td>C2</td><td>19.64</td><td>21.24</td><td>10.30</td><td>3.15</td><td>6.19</td><td>19.50</td><td>0.177</td><td>5.637</td></tr><tr><td>C3</td><td>14.14</td><td>14.73</td><td>5.92</td><td>3.19</td><td>5.04</td><td>14.23</td><td>1.212</td><td>0.825</td></tr><tr><td>C4</td><td>17.45</td><td>18.63</td><td>8.18</td><td>3.45</td><td>5.82</td><td>17.33</td><td>0.360</td><td>2.777</td></tr><tr><td>C5</td><td>14.12</td><td>14.73</td><td>5.90</td><td>3.19</td><td>5.03</td><td>14.22</td><td>1.206</td><td>0.829</td></tr><tr><td>C6</td><td>17.79</td><td>18.94</td><td>8.33</td><td>3.56</td><td>5.90</td><td>17.65</td><td>0.342</td><td>2.920</td></tr><tr><td>C7</td><td>13.58</td><td>14.13</td><td>5.90</td><td>2.79</td><td>4.89</td><td>13.68</td><td>1.127</td><td>0.887</td></tr><tr><td>C8</td><td>17.32</td><td>18.62</td><td>8.48</td><td>3.17</td><td>5.67</td><td>17.21</td><td>0.319</td><td>3.134</td></tr><tr><td>C9</td><td>13.62</td><td>14.13</td><td>5.90</td><td>2.79</td><td>4.92</td><td>13.72</td><td>1.150</td><td>0.870</td></tr><tr><td>C10</td><td>17.20</td><td>18.52</td><td>8.43</td><td>3.15</td><td>5.62</td><td>17.09</td><td>0.324</td><td>3.086</td></tr></table>

TABLE VI

UTILIZATION AND TEMPERATURE OVER THE MEASURED TELEMETRY WINDOWS. UTILIZATION IS THE TIME-AVERAGED RECORDED PERCENTAGE; TEMPERATURES ARE IN DEGREES CELSIUS. A DASH DENOTES UNAVAILABLE TELEMETRY, WHILE 0.0 IS A RECORDED, ROUNDED VALUE AND DOES NOT ESTABLISH THE ABSENCE OF SHORT ACCELERATOR ACTIVITY.
<table><tr><td>ID</td><td colspan="4">Mean utilization (%)</td><td colspan="2">CPU temperature (°C)</td><td colspan="2">GPU temperature (°C)</td></tr><tr><td></td><td>CPU</td><td>GPU</td><td>DLA0</td><td>EMC</td><td>Mean</td><td>Max.</td><td>Mean</td><td>Max.</td></tr><tr><td>CO</td><td>9.2</td><td>33.1</td><td></td><td>13.9</td><td>56.3</td><td>59.0</td><td>52.8</td><td>55.0</td></tr><tr><td>Cl</td><td>8.9</td><td>25.8</td><td></td><td>10.8</td><td>57.6</td><td>58.8</td><td>54.0</td><td>55.3</td></tr><tr><td>C2</td><td>9.7</td><td>21.8</td><td></td><td>9.2</td><td>57.1</td><td>58.8</td><td>52.9</td><td>54.2</td></tr><tr><td>C3</td><td>8.1</td><td>2.9</td><td>0.0</td><td>1.3</td><td>54.9</td><td>56.4</td><td>50.4</td><td>52.1</td></tr><tr><td>C4</td><td>12.1</td><td>11.8</td><td>1.4</td><td>6.1</td><td>56.8</td><td>58.3</td><td>52.2</td><td>53.7</td></tr><tr><td>C5</td><td>8.3</td><td>3.1</td><td>0.0</td><td>1.4</td><td>54.7</td><td>56.9</td><td>50.3</td><td>52.0</td></tr><tr><td>C6</td><td>12.9</td><td>13.0</td><td>2.0</td><td>6.6</td><td>56.8</td><td>58.4</td><td>52.2</td><td>53.5</td></tr><tr><td>C7</td><td>8.3</td><td>3.6</td><td>0.0</td><td>1.0</td><td>54.3</td><td>56.1</td><td>49.8</td><td>51.5</td></tr><tr><td>C8</td><td>13.0</td><td>15.5</td><td>0.0</td><td>5.4</td><td>56.2</td><td>58.3</td><td>51.9</td><td>53.3</td></tr><tr><td>C9</td><td>8.3</td><td>3.3</td><td>0.0</td><td>1.0</td><td>54.2</td><td>55.5</td><td>49.9</td><td>51.6</td></tr><tr><td>C10</td><td>12.9</td><td>14.9</td><td>0.0</td><td>5.3</td><td>55.7</td><td>58.3</td><td>51.5</td><td>53.2</td></tr></table>

## D. Stage-Level Analysis

Host preprocessing dominates CNN-MV (Table VII). Optimization reduces it from 73.54 to 9.09 ms on DLA and

from 73.50 to 9.08 ms on GPU, lowering end-to-end latency by 75.9% and 77.8%, respectively, and energy by 70.3% and

![](images/a5018557a575183dfaf8c2fea97d7bea919945d211ebd47bcec5a83aa55211c7.jpg)

Highest per-video F1: MVI\_40131 Sequence-level F1: YOLO 0.969; CNN-MV/DLA 0.967. All selected frames use dla\_motion  
![](images/0dcdbd4e13b8762c882b296a42deb45095d27889ac4b5407ce01dd2d0bc5e642.jpg)  
Fig. 3. YOLO-only versus serial-improved CNN-MV/DLA on successful motion frames nearest each sequence’s quartiles: MVI\_40141 frames 402/797/1196 (sequence F1: 0.783/0.466) and MVI\_40131 frames 411/822/1227 (0.969/0.967). F1 pairs are YOLO/CNN-MV. Outputs lack ground-truth overlays and do not establish per-frame accuracy.

## 71.7%, without materially changing accuracy.

A preliminary CuPy preprocessing path increased latency through small-tensor transfers and was discarded. CPU batching/vectorization was retained, with multithreading in parallel variants. GPU placement reduces optimized serial compute time from 5.13 to 2.95 ms/frame.

Figure 4 illustrates the temporal separation of CNN-MV activity on DLA core 0 and YOLO activity on the GPU.

Table VIII allocates frame-matched rail power according to each stage’s measured duration. Optimized serial preprocessing energy falls from 1038.7 to 158.1 mJ/frame on DLA and from 997.3 to 156.8 mJ/frame on GPU. The corresponding aggregate and rail-level power measurements remain available in Table V; small differences between stage totals and aggregate energy are due to frame matching and stage accounting.

## VI. DISCUSSION AND COMPARISON WITH PRIOR WORK

Our framework targets continuous vehicle localization for edge-based traffic monitoring and operates on detector outputs within one edge platform. This differs from RESPIRE’s transmission/processing reduction [3], contextual recognition inside a detector [4], and cross-device collaborative scheduling [5].

The scheduling review of Majeed et al. [6] explains why accelerator assignment must account for contention and transition overhead. Our measurements complement that perspective by separating host optimization from whole-model DLA/GPU placement; we do not propose a layer-partitioning scheduler. The detector-hardware survey by Sali et al. [7] addresses autonomous-vehicle perception more broadly, whereas our evaluation focuses on temporal box reuse on a single edge platform.

Like DFF and FGFA, learnable spatio-temporal sampling (LSTS) [17], progressive sparse local attention (PSLA) [18], sequence level semantics aggregation (SELSA) [19], and memory enhanced global-local aggregation (MEGA) [20] retain richer visual features through temporal alignment or aggregation. Our box updates avoid this intermediate computation but require detector refresh for discovery and recovery. Different datasets and mAP/F1 protocols preclude a direct ranking of accuracy.

![](images/0857f79e857473c10fecb266beac32174e383cd01a38d43c572b5c3941cc8187.jpg)  
Fig. 4. Nsight Systems timeline: CNN-MV on DLA core 0 (purple) and YOLO on GPU (blue), with thread, TensorRT, CUDA API, and OS activity below  
TABLE VII

MEAN COMPONENT LATENCY IN MILLISECONDS PER FRAME. COMPUTE INCLUDES YOLO AND MOTION-MODEL INFERENCE, WHICH ARE SHOWN SEPARATELY FOR INTERPRETATION AND MUST NOT BE ADDED AGAIN. OVERHEAD CONTAINS SYNCHRONIZATION, SCHEDULING, AND RESIDUAL BOOKKEEPING; IN PARALLEL VARIANTS, IT IS REPORTED AFTER ANY OVERLAP ALREADY PRESENT IN THE MEASURED FRAME TIME.
<table><tr><td>ID</td><td>Decode</td><td>Preprocess</td><td colspan="3">Compute (total and components)</td><td>Postprocess</td><td>Overhead</td></tr><tr><td></td><td></td><td></td><td>Total</td><td>YOLO</td><td>Motion</td><td></td><td></td></tr><tr><td>CO</td><td>0.60</td><td>3.90</td><td>4.01</td><td>4.01</td><td></td><td>1.69</td><td>2.00</td></tr><tr><td>Cl</td><td>1.64</td><td>2.84</td><td>4.25</td><td>3.01</td><td>1.24</td><td>1.32</td><td>1.67</td></tr><tr><td>C2</td><td>1.61</td><td>1.83</td><td>3.44</td><td>1.92</td><td>1.52</td><td>0.87</td><td>1.27</td></tr><tr><td>C3</td><td>1.67</td><td>73.54</td><td>5.18</td><td>2.22</td><td>2.96</td><td>2.29</td><td>3.03</td></tr><tr><td>C4</td><td>1.65</td><td>9.09</td><td>5.13</td><td>2.22</td><td>2.91</td><td>1.80</td><td>2.97</td></tr><tr><td>C5</td><td>1.68</td><td>74.68</td><td>5.83</td><td>2.23</td><td>3.60</td><td>2.20</td><td>1.06</td></tr><tr><td>C6</td><td>1.67</td><td>9.10</td><td>5.59</td><td>2.23</td><td>3.36</td><td>1.83</td><td>1.06</td></tr><tr><td>C7</td><td>1.68</td><td>73.50</td><td>2.95</td><td>2.23</td><td>0.72</td><td>1.91</td><td>2.96</td></tr><tr><td>C8</td><td>1.65</td><td>9.08</td><td>2.95</td><td>2.21</td><td>0.74</td><td>1.78</td><td>2.96</td></tr><tr><td>C9</td><td>1.67</td><td>74.53</td><td>5.22</td><td>2.24</td><td>2.97</td><td>1.91</td><td>1.13</td></tr><tr><td>C10</td><td>1.66</td><td>9.18</td><td>5.05</td><td>2.22</td><td>2.83</td><td>1.84</td><td>1.12</td></tr></table>

TABLE VIII

ESTIMATED STAGE ENERGY (MJ/FRAME), AVERAGED FROM FRAME-MATCHED RAIL POWER MULTIPLIED BY RECORDED STAGE DURATION. YOLO AND MOTION ARE INCLUDED IN COMPUTE AND DO NOT NEED TO BE ADDED AGAIN. A DASH INDICATES NO MOTION STAGE. THESE ARE TELEMETRY-BASED ALLOCATIONS WITHOUT IDLE SUBTRACTION; THEY ARE NOT DIRECT COMPONENT MEASUREMENTS.
<table><tr><td>ID</td><td>Decode</td><td>Preprocess</td><td colspan="3">Compute (total and components)</td><td>Postprocess</td><td>Overhead</td></tr><tr><td></td><td></td><td></td><td>Total</td><td>YOLO</td><td>Motion</td><td></td><td></td></tr><tr><td>CO</td><td>13.6</td><td>89.3</td><td>91.8</td><td>91.8</td><td></td><td>38.4</td><td>45.9</td></tr><tr><td>Cl</td><td>33.7</td><td>58.6</td><td>87.5</td><td>62.3</td><td>25.2</td><td>27.0</td><td>34.4</td></tr><tr><td>C2</td><td>31.5</td><td>36.4</td><td>67.5</td><td>38.2</td><td>29.4</td><td>17.1</td><td>25.0</td></tr><tr><td>C3</td><td>23.8</td><td>1038.7</td><td>74.1</td><td>32.2</td><td>41.9</td><td>32.6</td><td>43.2</td></tr><tr><td>C4</td><td>28.6</td><td>158.1</td><td>90.0</td><td>39.5</td><td>50.5</td><td>31.3</td><td>51.9</td></tr><tr><td>C5</td><td>23.9</td><td>1053.3</td><td>83.2</td><td>32.3</td><td>50.8</td><td>31.2</td><td>15.3</td></tr><tr><td>C6</td><td>29.4</td><td>161.3</td><td>100.1</td><td>40.3</td><td>59.8</td><td>32.6</td><td>19.2</td></tr><tr><td>C7</td><td>22.9</td><td>997.3</td><td>40.9</td><td>31.1</td><td>9.8</td><td>26.1</td><td>40.6</td></tr><tr><td>C8</td><td>28.3</td><td>156.8</td><td>51.8</td><td>39.1</td><td>12.7</td><td>30.9</td><td>51.5</td></tr><tr><td>C9</td><td>22.9</td><td>1013.3</td><td>71.9</td><td>31.4</td><td>40.5</td><td>26.2</td><td>15.8</td></tr><tr><td>C10</td><td>28.4</td><td>157.3</td><td>87.3</td><td>38.9</td><td>48.5</td><td>31.7</td><td>19.5</td></tr></table>

Against MVP [12], Analytical-MV adds occupied-cell masking and coupled translation-scale fitting. With a common detector, it reduces MVP-YOLO’s latency from 11.72 to 9.03 ms and energy from 0.241 to 0.177 J/frame, with F1 changing from 0.914 to 0.909. CNN-MV adds a learned, nonrecurrent alternative and measured host/accelerator tradeoffs.

See Without Decoding [13] combines box-aligned feature extraction (BAFE) with bidirectional long short-term memory (BiLSTM) refinement, reporting 89.62% mAP at IoU 0.5 on static-camera Multi-Object Tracking 2017 (MOT17) sequences and 42 simultaneous 30-FPS streams on an unspecified 16- gigabyte GPU. CNN-MV instead uses feed-forward updates and measures embedded latency, energy, and placement; stream capacity alone does not establish any of these. Our timing includes decoding, so no decoder-bypass benefit is claimed.

The two methods are alternatives: Analytical-MV favors latency and energy, whereas CNN-MV trades those advantages for higher recall and lower mean power. Host optimization accounts for most of the improvement over the original CNN-MV implementation; GPU placement then favors speed and energy, while DLA placement lowers GPU utilization. Scheduling must be chosen based on the surrounding workload, since parallelism improves DLA execution but not GPU execution in the evaluated configuration.

The present study is limited to 25 fixed-camera traffic videos. The different detector-routing rates are outcomes of each complete propagation-and-fallback policy; consequently, the evaluation captures system-level operating trade-offs but does not isolate propagation quality under a matched detector schedule. Moreover, Tegrastats provides power measurements at a coarser temporal resolution than the per-frame latency, and stage-level energy is therefore estimated through allocation rather than measured independently. The Nsight results provide system-level execution timelines rather than layerlevel traces. Although the six CNN-MV training sequences are distinct from the 25 evaluation sequences, the internal trainingvalidation split is sample-level rather than sequence-disjoint.

Future work will extend the evaluation to larger and more diverse datasets, including moving-camera scenarios, using sequence-disjoint training and validation partitions, and controlled detector schedules. Further investigation will consider adaptive refresh and fallback mechanisms, standard class-wise AP/mAP evaluation, and more detailed component-level energy profiling. The framework will also be extended to support workload-aware runtime scheduling that dynamically selects among Analytical-MV, CNN-MV, GPU execution, and DLA execution based on accuracy, latency, energy, and resourcecontention requirements.

## VII. CONCLUSION

This paper presents two alternative compressed-domain propagation methods, evaluated using a common YOLO-based framework on a UA-DETRAC traffic-video subset and a Jetson AGX Orin platform. Analytical-MV jointly estimates translation and scale and is the overall efficiency leader in this evaluation. CNN-MV produces candidate boxes through learned residual updates, improves recall over Analytical-MV, and draws lower mean power than the non-CNN methods. GPU placement provides the fastest CNN-MV execution, while DLA placement reduces time-averaged GPU utilization. Across these configurations, optimized CPU batching, vectorization, and multithreaded scheduling show that host processing is as important as neural inference to end-to-end latency and energy. These results demonstrate the potential of motionvector propagation for resource-constrained traffic-monitoring infrastructure and provide a basis for future evaluation in moving-camera physical-AI systems, including autonomous vehicles, drones, and mobile robots.

## REFERENCES

[1] ITU-T, “Advanced video coding for generic audiovisual services,” International Telecommunication Union, Recommendation H.264, Aug. 2024. [Online]. Available: https://www.itu.int/rec/T-REC-H.264- 202408-I/en

[2] C.-Y. Wu, M. Zaheer, H. Hu, R. Manmatha, A. J. Smola, and P. Krähenbühl, “Compressed video action recognition,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6026– 6035.

[3] X. Dai, P. Yang, X. Zhang, Z. Dai, and L. Yu, “Respire: Reducing spatial–temporal redundancy for efficient edge-based industrial video analytics,” IEEE Transactions on Industrial Informatics, vol. 18, no. 12, pp. 9324–9334, 2022.

[4] Y. Zhang, H. Bai, Y. Xu, Y. He, Q. Zhu, and H. Sheng, “Single-stage related object detection for intelligent industrial surveillance,” IEEE Transactions on Industrial Informatics, vol. 20, no. 4, pp. 5539–5549, 2024.

[5] M. Zhang, J. Cao, Y. Sahni, Q. Chen, S. Jiang, and L. Yang, “Blockchain-based collaborative edge intelligence for trustworthy and real-time video surveillance,” IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 1623–1633, 2023.

[6] A. A. Majeed, M. Meribout, and S. M. Sali, “Scheduling techniques of ai models on modern heterogeneous edge gpu—a critical review,” IEEE Transactions on Industrial Informatics, vol. 22, no. 4, pp. 2641–2652, 2026.

[7] S. Sali, A. Meribout, A. Majeed, M. Meribout, J. Pablo, V. Tiwari, and A. Baobaid, “Real-time object detection and associated hardware accelerators targeting autonomous vehicles,” Engineered Science, vol. 38, p. 1865, 2025. [Online]. Available: http://dx.doi.org/10.30919/ es1865

[8] X. Zhu, Y. Xiong, J. Dai, L. Yuan, and Y. Wei, “ Deep Feature Flow for Video Recognition ,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jul. 2017, pp. 4141–4150. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/CVPR.2017.441

[9] X. Zhu, Y. Wang, J. Dai, L. Yuan, and Y. Wei, “Flow-guided feature aggregation for video object detection,” in 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 408–417.

[10] M. Zhu and M. Liu, “Mobile video object detection with temporallyaware feature maps,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 5686–5695.

[11] M. Liu, M. Zhu, M. White, Y. Li, and D. Kalenichenko, “Looking fast and slow: Memory-guided mobile video object detection,” 2019. [Online]. Available: https://arxiv.org/abs/1903.10172

[12] B. Huang, N. Wang, W. Yao, and S. Dev, “Mvp: Motion vector propagation for zero-shot video object detection,” 2025. [Online]. Available: https://arxiv.org/abs/2509.18388

[13] A. Duché, C. Chatelain, and G. Gasso, “See without decoding: Motion-vector-based tracking in compressed video,” 2026. [Online]. Available: https://arxiv.org/abs/2602.00153

[14] L. Wen, D. Du, Z. Cai, Z. Lei, M.-C. Chang, H. Qi, J. Lim, M.-H. Yang, and S. Lyu, “Ua-detrac: A new benchmark and protocol for multi-object detection and tracking,” Computer Vision and Image Understanding, vol. 193, p. 102907, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1077314220300035

[15] NVIDIA, “Tensorrt: Building and launching a dla loadable,” NVIDIA TensorRT Documentation, 2026, accessed: Feb. 21, 2026. [Online]. Available: https://docs.nvidia.com/deeplearning/tensorrt/latest/inferencelibrary/dla-build-and-run.html

[16] “Tegrastats utility,” NVIDIA Jetson Linux Developer Guide, 2024, accessed: Feb. 21, 2026. [Online]. Available: https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/ JetsonLinuxDevelopmentTools/TegrastatsUtility.html

[17] Z. Jiang, Y. Liu, C. Yang, J. Liu, P. Gao, Q. Zhang, S. Xiang, and C. Pan, “Learning where to focus for efficient video object detection,” in Computer Vision – ECCV 2020, A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm, Eds. Cham: Springer International Publishing, 2020, pp. 18–34.

[18] C. Guo, B. Fan, J. Gu, Q. Zhang, S. Xiang, V. Prinet, and C. Pan, “Progressive sparse local attention for video object detection,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3908–3917.

[19] H. Wu, Y. Chen, N. Wang, and Z.-X. Zhang, “Sequence level semantics aggregation for video object detection,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9216–9224.

[20] Y. Chen, Y. Cao, H. Hu, and L. Wang, “Memory enhanced global-local aggregation for video object detection,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10 334– 10 343.