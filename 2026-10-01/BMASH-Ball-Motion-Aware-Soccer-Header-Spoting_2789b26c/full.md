# BMASH: Ball-Motion-Aware Soccer Header Spoting

Ahmed Endris Hasen<sup>∗</sup>   
ahmed.e.hasen@jyu.fi   
Faculty of Information Technology,   
University ofJyväskylä   
Jyväskylä, Finland   
Nikolaos Passalis   
passalis@csd.auth.gr   
Faculty of Sciences,   
Aristotle University of Thessaloniki   
Thessaloniki, Greece   
Muhammad Shahzad Khan   
muhammad.s.khan@jyu.fi   
Faculty of Information Technology,   
University ofJyväskylä   
Jyväskylä, Finland   
Jenni Raitoharju   
jenni.k.raitoharju@jyu.fi   
Faculty of Information Technology,   
University of Jyväskylä   
Jyväskylä, Finland

![](images/d2e0356f2b4b20544f530a31ae4b072ceda2c2b679d0beafda19c874a0dc85d9.jpg)  
Figure 1: Overview of the proposed BMASH framework. The action-recognition branch and the ball-detection branch are trained separately using task-specific data. Given a sampled video window, the Video Swin branch processes the full stack of� frames and produces action-score features $\mathbf { f } _ { v } ,$ while the YOLO ball detector is applied frame-wise and its detections are summarized into compact ball-motion features $\mathbf { f } _ { b } .$ . The two feature vectors are concatenated and passed to a lightweight fusion MLP to predict whether the input window contains a header.

## Abstract

Recent advances in computer vision have made broadcast sports videos increasingly useful for event analysis, performance assessment, and player-safety applications. In soccer, however, header spotting remains a challenging problem due to the subtle and shortlived nature of header events. This paper focuses on soccer header spotting: identifying moments in broadcast videos where the ball contacts a player’s head. We first adapt and evaluate Video Swin as a strong action-recognition baseline for this task, and then introduce BMASH, a ball-motion-aware fusion framework that integrates detector-derived ball features. BMASH combines Video Swin action representations with ball-presence and motion features from framelevel soccer-ball detections, integrating player-action context with ball dynamics to distinguish headers from visually similar events. We evaluate BMASH using game-level splits with separate test matches and rotating validation folds, considering both centeredwindow classification and continuous full-video spotting. Results show that Video Swin provides a strong baseline for header spotting, while BMASH improves clip-level AP and ROC-AUC over the corresponding Video Swin baseline. In continuous full-video spotting, BMASH achieves a comparable event-level F1-performance with a diferent precision–recall trade-of.

## CCS Concepts

• Computing methodologies → Activity recognition and understanding; Object detection; Neural networks.

## Keywords

Soccer Video Analysis, Header Spotting, Action Recognition, Ball Detection, Vision Transformer

## ACM Reference Format:

Ahmed Endris Hasen, Muhammad Shahzad Khan, Nikolaos Passalis, andJenni Raitoharju. 2026. BMASH: Ball-Motion-Aware Soccer Header Spotting. In 9th Int. Workshop on Multimedia Content Analysis in Sports (MMSports ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3841455.3841525

## 1 Introduction

Recently, progress in sports video understanding has been driven by large-scale benchmarks, advances in spatiotemporal deep learning, and automated broadcast analysis [1, 27]. These developments have substantially expanded the scope of computer vision in sports, enabling applications such as event recognition, automatic high light generation, tactical analysis, player tracking, performance assessment, and athlete health monitoring [34, 36]. Among these applications, action spotting has become a fundamental problem in soccer video understanding, aiming to localize the occurrence of semantic events within broadcast matches [8, 10, 12, 34]. Large-scale benchmarks, such as SoccerNet [10], have accelerated research in this direction by providing annotated broadcast matches for evaluating event localization methods.

Despite this progress, soccer header spotting remains considerably less explored than other action spotting. A soccer header is characterized by a brief ball-head interaction that frequently lasts only a few frames and occupies a very small image region [29]. Detecting this subtle interaction, therefore, represents a challenging fine-grained recognition problem that requires reasoning about player motion, surrounding context, and ball dynamics rather than relying solely on global scene appearance [11, 35].

The dificulty is further afected by the characteristics of broadcast soccer videos. Camera motion, rapid zooming, viewpoint changes, motion blur, occlusion, compression artifacts, and limited spatial resolution of distant players all reduce the visibility of both the ball and the head-ball contact [10]. Moreover, many visually similar actions, including crosses, clearances, long passes, aerial duels, throw-ins, and shots, share highly similar motion patterns and ball trajectories, making it dificult to distinguish heading events from other airborne ball interactions [12, 29].

Beyond its importance for automatic match analysis, reliable header spotting is increasingly motivated by player-safety research. Numerous biomechanical and epidemiological studies have investigated repetitive soccer headers and their potential relationship with head-impact exposure, concussion risk, and long-term neurological outcomes [19]. However, estimating heading exposure currently relies heavily on labor-intensive manual video review or wearable sensing followed by extensive human verification [19, 20]. Recent computer vision systems, such as DeepImpact [29], have demonstrated that automated video screening can substantially reduce this manual efort by identifying candidate heading events from broadcast footage. Consequently, accurate header spotting constitutes an essential first stage toward scalable video-based head-impact monitoring and downstream biomechanical analysis.

Despite recent progress in sports video understanding, soccer header spotting remains comparatively underexplored as a standalone fine-grained recognition problem. Reliable soccer header spotting requires both accurate spatiotemporal understanding of player actions and awareness of the ball’s presence and movement.

Modern video recognition architectures can capture player motion, body interactions, and broadcast context, while ball detections provide additional information about the ball’s presence and movement. However, ball information alone is not suficient, since many non-header actions also involve airborne ball motion. Thus, ball motion is used as supporting evidence for visual action features.

To address these challenges, we adapt Video Swin to the soccer header spotting task and introduce BMASH (Ball-Motion-Aware Soccer Header Spotting), a lightweight late-fusion framework that integrates full-frame action representations with detector-derived ball-motion information. BMASH combines spatiotemporal Video Swin action representations with ball-motion information derived from frame-level soccer-ball detections.

We evaluate BMASH on a soccer header dataset constructed from multiple broadcast matches. To ensure a realistic evaluation and avoid overlap between training and test footage, we use gamelevel splits with unseen test matches. Evaluation is conducted at both the temporal-window level, using labeled header and nonheader clips, and the full-video level, where the model is applied to continuous broadcast video using sliding-window inference. The main contributions of this work are:

• We adapt Video Swin as a strong transformer-based actionrecognition baseline for soccer header spotting and compare it with EficientNet3D and TSM-ResNet50 under a unified header/non-header protocol.

• We propose BMASH, a lightweight ball-motion-aware latefusion framework that combines Video Swin-based action representations with compact detector-derived ball features for soccer header spotting.

• We evaluate the models using game-level splits with unseen test videos across diferent input-window configurations, and further analyze their performance in continuous fullvideo spotting.

The rest of the paper is organized as follows. Section 2 discusses prior and related works on action spotting, header detection, and head-impact analysis. Section 3 presents the BMASH framework. Section 4 describes the dataset, compared methods, and experimental setup. Section 5 presents the results and analysis. Section 6 concludes the paper and discusses future work.

## 2 Related Work

## 2.1 Action Spotting in Soccer

Action spotting aims to temporally localize semantic events within long, untrimmed sports videos. In soccer, this problem has become one of the core tasks in video understanding following the introduction of SoccerNet [10], which established the first large-scale benchmark for temporal localization of events in full broadcast matches. SoccerNet-v2 [8] further expanded both the annotation scale and benchmark tasks, enabling broader research on long-form soccer video understanding. Recent surveys show that SoccerNet is the most widely used benchmark for soccer action spotting and temporal event localization [12, 36].

Research has progressively shifted from learning global clip representations toward explicitly modeling temporal context surrounding candidate events. NetVLAD++ [13] introduced temporallyaware feature aggregation by separately encoding information before and after candidate timestamps. Cioppa et al. [4] introduced a context-aware loss function (CALF) for soccer action spotting, which models the temporal neighborhood around annotated events rather than treating each timestamp in isolation. Subsequent methods explored multiple-scene representations to exploit broadcast transitions [30], dense temporal anchor prediction for precise localization [31], and transformer-based reasoning capable of modeling long-range temporal dependencies while addressing class imbalance and temporal uncertainty [33]. Unified frameworks, such as OSL-ActionSpotting [3], have further improved soccer action spotting by providing standardized implementations of major spotting algorithms.

Recent SoccerNet challenges have shifted attention toward denser ball-centric understanding by introducing Ball Action Spotting, where headers constitute only one of several fine-grained ball interaction classes alongside passes, shots, throw-ins, goals and oth ers [5]. Cioppa et al. [5] presented the best-performing EficientNetbased 2D–3D model, where 2D convolutions encode appearance from individual frames and 3D convolutions model temporal information across stacked frames. Parallel research has also investigated anticipating future ball actions before they occur [7].

These studies demonstrate growing interest in modeling subtle ball-player interactions. However, most existing approaches target large-scale, multi-class event spotting, where headers constitute only one of many event categories and can be dificult to model due to class imbalance and limited examples. In contrast, our work focuses on dedicated binary soccer header spotting as a standalone fine-grained recognition problem.

## 2.2 Soccer Header and Head-Impact Detection

Compared with general soccer action spotting, dedicated computer vision methods for automated soccer header detection have received considerably less attention, leaving a gap between soccer action spotting and automated head-impact analysis. Prior work has explored computer vision pipelines for estimating soccer head-impact exposure from broadcast video, including approaches that combine ball detection, tracking, localized video crops, and temporal classification to identify potential heading events [29]. Although the approach achieved high header detection sensitivity on full-match tests, the reported precision dropped to 21.1% due to a large number of false-positive detections, highlighting the dificulty of robust header spotting in real full match scenarios [29]. In such pipelines, the temporal classifier receives cropped video regions centered on the detected ball, so missed or incorrect ball detections can directly lead to crops that omit the relevant header action or focus on an irrelevant region. These studies show the feasibility of using broadcast footage for header-related analysis, while also motivating further work on robust header spotting in realistic match videos. In contrast to ball-centered crop pipelines, our framework preserves full-frame spatiotemporal reasoning and uses detector-derived ball information as supporting evidence through late fusion.

Beyond computer vision, extensive biomechanics research has investigated soccer heading through wearable sensing. Kern et al. [20] developed a neural network capable of distinguishing true headers from false-positive recordings from xPatch wearable head-impact sensors placed behind the players’ right ear over the mastoid process, using kinematic measurements verified by broadcast video. Kenny et al. [19] quantified heading biomechanics in collegiate soccer using an instrumented mouthpiece with tri-axis accelerometers and gyroscopes, synchronized with video verification. In a subsequent longitudinal study, the same group investigated individual heading exposure using a large set of video-confirmed headers collected over an extended observation period [18]. Additional investigations have characterized heading burden and head-injury-risk situations in elite football and extended exposure analysis to professional football [9]. The broader head-impact literature consistently demonstrates that wearable sensors alone generate substantial numbers of false-positive events, making video confirmation essential for reliable exposure estimation [20, 22]. More recently, quantitative video analysis has been recognized as an efective complementary alternative for scalable head-impact monitoring without wearable sensors and extensive manual annotation [2].

## 2.3 Video Action Recognition and Ball Tracking

Deep learning for video understanding has evolved from convolutionbased architectures toward transformer-based spatiotemporal representation learning. Early approaches, such as the Temporal Shift Module (TSM) [23], introduced temporal reasoning into 2D convolutional networks by shifting feature channels across adjacent frames. More recent transformer-based architectures have substantially improved spatiotemporal video understanding. Among these, Video Swin Transformer [25] employs hierarchical shifted-window attention to eficiently model long-range spatiotemporal patterns, making it a strong backbone for dedicated soccer header spotting.

Soccer ball detection and tracking have also been widely studied as the ball is small, fast-moving, frequently blurred, and often occluded in broadcast footage [16, 17, 21]. DeepBall [21] introduced a dedicated deep-learning framework for long-shot soccer ball detection, while subsequent work used temporal consistency to improve robustness when the ball is temporarily invisible [16]. Semi-supervised learning has further improved player and ball detection in SoccerNet broadcasts [32].

Prior work in other ball sports also highlights the value of temporal object modeling. TrackNet [14] showed that temporal information improves localization of small, high-speed tennis balls in broadcast video, while TrackNetV4 [28] further explored motionaware feature fusion for ball tracking in tennis, badminton, and table tennis. These studies provide useful motivation for incorporating ball-motion information into header spotting, and our work investigates whether compact detector-derived ball-motion descriptors can complement transformer-based video representations for dedicated soccer header spotting.

## 3 Method

## 3.1 Overview of our Method

BMASH is a ball-motion-aware late-fusion framework for binary soccer header spotting in broadcast video. Given a short temporal window sampled from a match, the model predicts whether the window contains a header or a visually similar non-header action. The task is challenging because the defining action evidence, namely ball-head contact, is often brief, small in the image, and partially obscured by players, camera motion, or motion blur. At the same time, many non-header actions also exhibit similar airborne ball motion, making ball visibility alone insuficient for reliable prediction. BMASH then combines full-frame action understanding with detector-derived ball information as supporting evidence.

We denote the sampled window as $\mathbf { X } = \{ \mathbf { x } _ { t _ { 1 } } , \mathbf { x } _ { t _ { 2 } } , \ldots , \mathbf { x } _ { t _ { T } } \}$ and assign it a binary label $y \in \{ 0 , 1 \}$ , where $y = 1$ denotes a header and $y = 0$ denotes a non-header action. The goal is to learn a classifier $F ( \mathbf { X } )$ that estimates

$$
{ \hat { y } } = F ( \mathbf { X } ) = P ( y = 1 \mid \mathbf { X } ) ,\tag{1}
$$

where $\hat { y }$ is the predicted probability that the input window contains a header.

To address this, BMASH combines two complementary sources of information from the same input window. The first source is a global spatiotemporal action representation extracted from the full broadcast clip using a Video Swin Transformer. This branch captures player motion, body configuration, player-player inter action, and surrounding scene context. The second source is an explicit ball-motion representation derived from frame-level soccerball detections, summarizing ball visibility and movement over time. These window-level representations are aligned and fused through a lightweight classifier, allowing explicit ball information to complement full-frame visual action understanding.

Figure 1 illustrates the overall BMASH pipeline. A video window is sampled into� frames and processed by both branches. The Video Swin branch produces a 6-dimensional action-score feature vector, while the ball-detection branch summarizes detector outputs into a 10-dimensional ball-motion descriptor. The two representations are concatenated and passed through a lightweight fusion MLP that predicts the probability that the input window contains a header. This late-fusion design allows detector-derived ball information to complement full-frame spatiotemporal action understanding without making the final prediction depend only on the presence of a detected ball.

## 3.2 Video Swin Action Recognition

The Video Swin branch provides the spatiotemporal action representation used by BMASH. It learns window-level video representations from the sampled frames. Given an input video window, we sample � frames and denote the resulting clip as $\textbf { X } =$ $\{ \mathbf { x } _ { t _ { 1 } } , \mathbf { x } _ { t _ { 2 } } , \ldots , \mathbf { x } _ { t _ { T } } \}$ , where $\mathbf { x } _ { t _ { i } }$ is the frame sampled at time $t _ { i } .$ The sampled frames are processed together as a spatiotemporal clip.

Video Swin processes the complete sampled window, allowing the model to capture player motion, body pose changes, nearby player interactions, and broader scene context. The Video Swin encoder processes X and outputs two-class softmax scores, denoted by �<sub>��</sub> and $ { \mathcal { P } } H$ for the non-header and header classes. When multiple temporal clips are sampled from the same input window, the same notation refers to the scores from each sampled clip. These scores are complementary probabilities. For fusion, we summarize them into a 6-dimensional score feature vector:

$$
\mathbf { f } _ { v } = [ \mu _ { N H } , \mu _ { H } , m _ { N H } , m _ { H } , \sigma _ { N H } , \sigma _ { H } ] ,\tag{2}
$$

where $\mu , m ,$ and � denote the mean, maximum, and standard deviation of the non-header and header scores across the temporal clips sampled from the same window. Thus, $\mathbf { f } _ { v } \in \mathbb { R } ^ { 6 }$

Using the sampled video window is important because soccer headers are defined not only by the ball location, but also by the surrounding action context. Body orientation, aerial duels, nearby opponents, and the phase of play can help distinguish headers from other ball-in-play actions. However, the actual ball-head contact can still be small and brief, so BMASH complements the action representation with an explicit ball-motion branch.

## 3.3 Ball Detection and Motion Encoding

The ball-detection and motion descriptor branch provides explicit ball-related information from the same sampled video window. We use a YOLO11s [15] object detector pretrained on the COCO dataset [24] and fine-tuned for soccer-ball detection using separate SoccerNet tracking data. The fine-tuned ball detector is then applied to each sampled frame in the input window to detect the ball. For each frame, we keep the highest-confidence ball candidate and encode its detection information, i.e., the detection confidence, normalized position and normalized bounding-box area. If no ball candidate is detected, the detection values are set to zero. The frames are processed independently without any additional cliplevel adjustment of the detections.

For each sampled frame $t _ { i } ,$ the detector output is represented as $\mathbf { b } _ { t _ { i } } = [ r _ { i } , c _ { i } , x _ { i } , y _ { i } , a _ { i } ]$ , where $r _ { i }$ indicates whether the ball is detected, $c _ { i }$ is the detection confidence, $( x _ { i } , y _ { i } )$ is the normalized ball-center position, and $a _ { i }$ is the normalized bounding-box area. If no ball is detected, all values are set to zero. Across the input window, the frame-level detections $\{ \mathbf { b } _ { t _ { 1 } } , \mathbf { b } _ { t _ { 2 } } , . . . , \mathbf { b } _ { t _ { T } } \}$ are summarized into a 10-dimensional ball-motion descriptor:

$$
\mathbf { f } _ { b } = [ \mu _ { c } , m _ { c } , \sigma _ { c } , \mu _ { r } , \mu _ { a } , m _ { a } , \mu _ { x } , \mu _ { y } , \mu _ { d } , m _ { d } ] ,\tag{3}
$$

where $\mu _ { c } , m _ { c } ,$ and $\sigma _ { c }$ are the mean, maximum, and standard deviation of the ball-detection confidence; $\mu _ { r }$ is the detection rate; $\mu _ { a }$ and $m _ { a }$ are the mean and maximum normalized bounding-box area; $\mu _ { x }$ and $\mu _ { y }$ are the mean normalized ball-center positions; and $\mu _ { d }$ and $m _ { d }$ are the mean and maximum displacement between consecutive detected ball centers. Thus, $\mathbf { f } _ { b } \in \mathbb { R } ^ { 1 0 }$

Ball detections are used as supporting evidence rather than as a standalone decision signal. This is important because visible ball motion appears in many non-header events, while true headers may still have weak or missing ball detections due to occlusion, blur, or small ball size. The ball-detection branch, thus, provides structured ball-related information that complements the Video Swin based action representation learned from the sampled video window.

## 3.4 Ball-Motion-Aware Fusion

BMASH combines the two window-level feature vectors using late fusion. Since the Video Swin branch and the ball-motion branch operate on the same sampled video window X, their outputs are aligned at the window level. The final fusion input is obtained by concatenating the 6-dimensional Video Swin action-score feature vector $\mathbf { f } _ { v }$ and the 10-dimensional ball-motion descriptor $\mathbf { f } _ { b } \mathbf { : }$

$$
\mathbf { f } = [ \mathbf { f } _ { v } ; \mathbf { f } _ { b } ] \in \mathbb { R } ^ { 1 6 } ,\tag{4}
$$

where [·; ·] denotes feature concatenation. The fused 16-dimensional vector is passed to a lightweight multi-layer perceptron (MLP) classifier:

$$
{ \hat { y } } = C ( \mathbf { f } ) ,\tag{5}
$$

where �ˆ is the predicted probability that the input window contains a header and $C ( \cdot )$ denotes the fusion MLP classifier. The MLP consists of two hidden layers with ReLU activations and dropout, followed by a single output unit whose sigmoid activation gives the header probability. During fusion training, the Video Swin and ball-detection branches were kept fixed, and only the fusion MLP was optimized using the extracted feature representations.

The fusion classifier is trained using weighted binary crossentropy with logits. We use class weighting to account for differences in the number of header and non-header samples within each training fold. The header-class weight $w _ { h }$ is computed as the ratio of non-header to header samples in the training fold. The loss can be written as

$$
\mathcal { L } = - w _ { h } y \log ( \hat { y } ) - \big ( 1 - y \big ) \log ( 1 - \hat { y } ) ,\tag{6}
$$

where $y \in \{ 0 , 1 \}$ is the ground-truth label and �ˆ is the predicted header probability. When threshold tuning is used, the decision threshold is selected on the validation split and then applied to the held-out test set.

The fusion classifier learns to combine the two evidence sources. Strong ball motion without compatible player posture may correspond to a cross, clearance, long pass, or shot. Conversely, a visually plausible header may still occur when the ball detector is uncertain due to blur, occlusion, or small ball size. The late-fusion design allows BMASH to use ball-motion information when it supports the action context, without treating ball detections as a standalone decision rule. When ball evidence is weak or missing, the classifier can rely more on the Video Swin action representation.

## 4 Experiments

## 4.1 Dataset

The dataset was constructed from SoccerNet broadcast videos recorded at 25 fps. We used all the seven matches from the SoccerNet Ball Action Spotting annotations, from which header timestamps were extracted. To increase data diversity, we additionally added five raw SoccerNet broadcast matches (Wigan Athletic–Birmingham City, Cardif City–Queens Park Rangers, Napoli–Inter Milan, Real Madrid–1. FC Union Berlin, Arsenal–Racing Club de Lens) and manually annotated header timestamps. The selected matches were split at the game level, so that windows from the same match do not appear in both training and test sets.

Each annotated timestamp was converted into a short temporal window. Header windows were centered on visible ball-head contact. Candidate events were reviewed frame by frame, and unclear cases were excluded when ball-head contact could not be verified because of severe occlusion or camera cuts. Non-header windows were sampled from visually similar ball-in-play situations, such as crosses, clearances, passes, shots, throw-ins, aerial duels, and other airborne-ball interactions, while ensuring that the selected windows did not contain visible ball-head contact. This centered-window setting isolates the recognition problem by evaluating whether a candidate temporal window contains a header or a visually similar non-header action. The separate full-video analysis in Section 5.3 evaluates the complete spotting setting, where candidate windows are generated from continuous broadcast video using sliding-window inference.

Table 1: Distribution of the dataset across the experiment.
<table><tr><td>Split</td><td>Non-header</td><td>Header</td><td>Total</td></tr><tr><td>Train/validation pool</td><td>1067</td><td>947</td><td>2014</td></tr><tr><td>Test set</td><td>268</td><td>261</td><td>529</td></tr><tr><td>Total</td><td>1335</td><td>1208</td><td>2543</td></tr></table>

Table 2: Rotating game-level validation folds. Counts are given as header/non-header.
<table><tr><td>Fold</td><td>Validation game</td><td>Train</td><td>Validation</td></tr><tr><td>0</td><td>Brentford-Bristol City</td><td>847/957</td><td>100/110</td></tr><tr><td>1</td><td>Hull City-Sheffield Wednesday</td><td>822/957</td><td>125/110</td></tr><tr><td>2</td><td>Leeds United-West Bromwich</td><td>879/957</td><td>68/110</td></tr><tr><td>3</td><td>Middlesbrough-Preston North End</td><td>820/957</td><td>127/110</td></tr><tr><td>4</td><td>Reading-Fulham</td><td>893/957</td><td>54/110</td></tr><tr><td>5</td><td>Stoke City-Huddersfield Town</td><td>819/957</td><td>128/110</td></tr><tr><td>6</td><td>Cardiff City-Queens Park Rangers</td><td>839/949</td><td>108/118</td></tr><tr><td>7</td><td>Napoli-Inter Milan</td><td>856/959</td><td>91/108</td></tr><tr><td>8</td><td>Real Madrid-1. FC Union Berlin</td><td>861/965</td><td>86/102</td></tr><tr><td>9</td><td>Arsenal-Racing Club de Lens</td><td>887/988</td><td>60/79</td></tr></table>

The test set remained fixed across all folds and contains two matches: BN (Blackburn Rovers–Nottingham Forest) and WB (Wigan Athletic–Birmingham City), a total of 529 temporal windows derived from annotated timestamps, with 261 headers and 268 nonheaders. The remaining ten matches formed the training/validation pool. For rotating game-level validation, each fold used one match for validation and the other nine matches for training. Table 1 summarizes the class distribution across the training/validation pool and the fixed test set, while Table 2 lists the validation match used in each fold.

## 4.2 Compared Methods

While many methods have been proposed for soccer action spotting, our comparison focuses on models that are most relevant to header spotting: ball-action spotting models that include header as one of the action classes, a header specific recognition method, and Video Swin as a main action-recognition baseline.

For generic ball-action spotting, we evaluate a 12-class SoccerNet Ball Action EficientNet3D model [5], where header is one ofseveral classes, including pass, drive, high pass, cross, shot, and throw-in. In this Ball Action Spotting setting, header obtains approximately 0.56 AP@1, indicating that headers are among the more dificult ball-action classes in full-video event spotting. On our centered clip-level test set, the same header-score evaluation gives high precision but low recall at the default 0.5 threshold (precision = 0.9230, recall = 0.4303, F1 = 0.5870). This comparison addresses whether a specialized binary header detector provides an advantage over a generic ball-action model in which header is only one class among several actions. Similarly, prior header-detection work using TSM ResNet50 with ball tracking and ball-centered cropping reported high sensitivity but only 21.1% precision, highlighting the dificulty of realistic full-video header detection [29].

For direct comparison on our dataset, we trained and evaluated EficientNet3D, TSM-ResNet50, and Video Swin [25] under the same game-level binary header and non-header protocol. EficientNet3D provides a convolutional spatiotemporal baseline, TSM-ResNet50 represents a temporal-shift 2D CNN baseline related to prior headerdetection work, and Video Swin serves as the strongest actionrecognition baseline. BMASH uses the Video Swin branch and adds detector-derived ball-motion features through late fusion, allowing us to compare full-frame action recognition with ball-motion-aware fusion under the same dataset, split, and evaluation protocol.

## 4.3 Experimental Setup

This section presents the experimental setup and implementation details for BMASH and the baseline models. All models were trained on temporal windows from the training matches, selected using rotating validation folds, and tested on the same fixed held-out matches. For each fold, one match was used for validation, and the checkpoint with the best validation performance was used as the final model for test evaluation. This procedure produced ten independently trained models, one for each rotating validation match. Table 3 reports the mean and standard deviation of the fixed test-set metrics across these ten trained models. EficientNet3D and TSM-ResNet50 used 33-frame and 17-frame windows, respectively. Video Swin (along with BMASH) used two temporal settings, with 16-frame and 32-frame windows, to analyze the efect of inputwindow length.

EficientNet3D and TSM-ResNet50 were trained using AdamW [26] for 30 epochs with learning rates of $1 \times 1 0 ^ { - 3 }$ and $3 \times 1 0 ^ { - 4 }$ , respectively. Video Swin [25] was implemented based on MMAction2 [6] using a Video Swin-Tiny backbone pretrained on Kinetics-400 and fine-tuned on our soccer header dataset for header spotting task. The model was trained for 30 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - 3 }$ . Video Swin-16 uses 16-frame clips at 960×544, while Video Swin-32 uses 32-frame clips at 1280×736 with gradient accumulation. For BMASH, the fusion MLP was trained for 100 epochs using the Video Swin action and ball-motion features, binary cross-entropy with logits, AdamW, and batch size 16. The ball detector used a YOLO11s [15] model pretrained on COCO [24] and fine-tuned for soccer-ball detection for 20 epochs with input size 1280, using batch size 6 and learning rate 0.01.

All experiments were conducted on the CSC Roihu environment using NVIDIA GH200 GPU nodes with Python 3.12, PyTorch 2.10, MMAction2/MMEngine, and Ultralytics YOLO.

## 4.4 Evaluation Metrics

We evaluated the performance using average precision (AP), F1- score, precision, recall, balanced accuracy, and ROC-AUC. AP and

ROC-AUC summarize how well the model ranks header and nonheader windows independently of a single decision threshold. Precision, recall, F1-score, and balanced accuracy measure classification performance after converting model scores into binary predictions.

Full-video spotting was evaluated separately as an event-detection task. The model was applied to continuous broadcast videos using sliding-window inference with 2.0-second windows and a stride of 0.5 seconds. Window-level scores were smoothed and processed with non-maximum suppression to merge nearby detections and prevent repeated predictions for the same event. Predicted events were matched with annotated header timestamps using a tolerance of ±2 seconds. Matched predictions were counted as true positives, unmatched predictions as false positives, and unmatched annotations as false negatives. We report event-level precision, recall, F1-score, and false positives per match. True negatives were not used for event-level full-video evaluation, since continuous match footage contains a very large number of non-header moments.

## 5 Results Analysis

## 5.1 Overall Performance

Table 3 shows the performance on the test set. Results are reported as mean ± standard deviation across the validation configurations reported in Table 2. TSM-ResNet50 achieves high recall but low precision and balanced accuracy, indicating that it tends to overpredict the header class. Video Swin provides a much stronger visual action baseline, improving AP and ROC-AUC. Adding ballderived features produces a clear improvement in the 16-frame setting: BMASH-16 increases AP from 0.8858 to 0.8966 and F1- score from 0.7416 to 0.7961. The 32-frame configuration produces a stronger visual baseline, with Video Swin-32 reaching 0.9423 AP and 0.8636 F1-score. BMASH-32 achieves the highest AP and ROC-AUC overall, reaching 0.9530 AP and 0.9507 ROC-AUC, while maintaining essentially the same F1-score as Video Swin-32. These results indicate that detector-derived ball information primarily improves score ranking and class separation, whereas the strongest visual backbone already provides highly competitive thresholdbased classification performance.

## 5.2 Per-Fold and Qualitative Analysis

Figure 2 shows the per-fold F1-score comparison between the proposed BMASH model and all evaluated baseline methods. Figure 3 shows a representative header case, with the detected ball and header action highlighted. Figure 4 shows the precision–recall across folds. BMASH-32 is concentrated in the upper-right region, indicating a stronger balance between precision and recall than the other evaluated models. This compact distribution across folds also indicates stable behavior under diferent game-level validation splits.

## 5.3 Full-Video Header Spotting Analysis

In addition to temporal-window clip classification, we evaluate fullvideo header spotting on the two test broadcast matches. In this setting, the models process continuous video streams and convert dense sliding-window scores into discrete header-event predictions. The two videos contain a total of 261 annotated header events.

Table 3: Performance comparison of all evaluated models. Results are reported as mean ± standard deviation across the validation folds. The best result for each metric is shown in bold.
<table><tr><td>Method</td><td>AP</td><td>F1</td><td>Precision</td><td>Recall</td><td>Balanced Acc.</td><td>ROC-AUC</td></tr><tr><td>EfficientNet3D</td><td> $0 . 7 0 6 1 { \scriptstyle \pm 0 . 0 5 4 4 }$ </td><td> $0 . 6 6 0 6 { \scriptstyle \pm 0 . 0 4 2 9 }$ </td><td> $0 . 5 9 2 7 { \scriptstyle \pm 0 . 0 6 9 9 }$ </td><td> $0 . 7 9 3 1 { \scriptstyle \pm 0 . 1 8 0 9 }$ </td><td> $0 . 6 0 9 4 { \scriptstyle \pm 0 . 0 5 1 7 }$ </td><td> $0 . 7 1 2 4 { \scriptstyle \pm 0 . 0 5 1 7 }$ </td></tr><tr><td>TSM-ResNet50</td><td> $0 . 5 8 8 9 { \scriptstyle \pm 0 . 0 6 4 2 }$ </td><td> $0 . 6 6 3 1 { \scriptstyle \pm 0 . 0 0 3 4 }$ </td><td> $0 . 5 0 0 8 { \scriptstyle \pm 0 . 0 0 8 3 }$ </td><td> $\mathbf { 0 . 9 8 2 0 { \scriptstyle \pm 0 . 0 2 0 6 } }$ </td><td> $0 . 5 1 3 8 { \pm } 0 . 0 1 5 3$ </td><td> $0 . 5 9 1 3 { \scriptstyle \pm 0 . 0 4 5 7 }$ </td></tr><tr><td>Video Swin-16</td><td> $0 . 8 8 5 8 { \scriptstyle \pm 0 . 0 2 5 0 }$ </td><td> $0 . 7 4 1 6 { \pm } 0 . 0 6 8 3$ </td><td> $0 . 8 7 8 6 { \scriptstyle \pm 0 . 0 4 6 3 }$ </td><td> $0 . 6 5 2 5 { \scriptstyle \pm 0 . 1 0 4 0 }$ </td><td> $0 . 7 7 9 4 { \pm } 0 . 0 3 8 8$ </td><td> $0 . 8 8 3 1 { \scriptstyle \pm 0 . 0 2 4 7 }$ </td></tr><tr><td>BMASH-16</td><td> $0 . 8 9 6 6 { \scriptstyle \pm 0 . 0 2 4 4 }$ </td><td> $0 . 7 9 6 1 { \scriptstyle \pm 0 . 0 5 7 5 }$ </td><td> $0 . 8 0 9 6 { \scriptstyle \pm 0 . 0 6 9 1 }$ </td><td> $0 . 8 0 2 3 { \scriptstyle \pm 0 . 1 1 7 7 }$ </td><td> $0 . 8 0 1 3 { \scriptstyle \pm 0 . 0 4 1 6 }$ </td><td> $0 . 8 9 9 0 { \scriptstyle \pm 0 . 0 2 2 3 }$ </td></tr><tr><td>Video Swin-32</td><td> $0 . 9 4 2 3 { \scriptstyle \pm 0 . 0 1 3 1 }$ </td><td> $0 . 8 6 3 6 { \scriptstyle \pm 0 . 0 3 6 4 }$ </td><td> $0 . 9 0 2 2 { \scriptstyle \pm 0 . 0 2 8 0 }$ </td><td> $0 . 8 3 3 3 { \scriptstyle \pm 0 . 0 7 4 9 }$ </td><td> $0 . 8 7 1 5 { \scriptstyle \pm 0 . 0 2 7 9 }$ </td><td> $0 . 9 4 1 8 { \pm } 0 . 0 1 3 9$ </td></tr><tr><td> $\mathbf { B M A S H } – 3 2$ </td><td> $\mathbf { 0 . 9 5 3 0 { \pm 0 . 0 1 0 3 } }$ </td><td> $\mathbf { 0 . 8 6 3 9 { \pm 0 . 0 3 8 3 } }$ </td><td> $\mathbf { 0 . 9 1 5 2 { \pm 0 . 0 2 9 0 } }$ </td><td> $0 . 8 2 4 1 { \scriptstyle \pm 0 . 0 8 1 2 }$ </td><td> $\mathbf { 0 . 8 7 3 5 { \pm 0 . 0 2 8 9 } }$ </td><td> $\mathbf { 0 . 9 5 0 7 { \scriptstyle \pm 0 . 0 1 1 4 } }$ </td></tr></table>

![](images/60c5a9a68e09dc1cacb706f7b8c4858b7b9c294ce2356db470b4b1f7e6f13486.jpg)

Figure 2: Per-fold F1-score comparison of the proposed BMASH models and the baseline methods.  
![](images/76624093a8248272dd04288b27a7b9e2d6c1ae410791b595891392fc549b1cb9.jpg)  
Figure 3: Qualitative examples of header spotting. The red box indicates the detected ball, and the blue box highlights the player in the header action.

![](images/584d4c4ca586d3acc126ff52d6a8ae4d2e588273863a72d4eadcdf43dcc30940.jpg)

Table 4 summarizes the event-level results across both full broadcast videos. Video Swin-32 achieves the highest overall F1-score (0.636), together with the highest precision among the 32-frame models (0.617). BMASH-32 achieves almost the same F1-score (0.635) while detecting more true headers than Video Swin-32 (175 vs. 171) and obtaining the highest recall (0.671), at the cost of more false positives. Video Swin-16 attains the same recall as BMASH-32 but produces substantially more false positives, whereas BMASH-16 achieves the highest precision (0.621) and the fewest false positives (97), although with reduced recall (0.609). Overall, these results illustrate the trade-of between recall and precision in full-video header spotting.

Figure 4: Precision–recall results across folds. Points show individual folds and stars show the mean for each model.  
Table 4: Full-video header spotting results across the two test videos, using two second temporal matching.
<table><tr><td>Model</td><td>GT</td><td>Pred</td><td>TP</td><td>FP</td><td>FN</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>BMASH-32</td><td>261</td><td>290</td><td>175</td><td>115</td><td>86</td><td>0.603</td><td>0.671</td><td>0.635</td></tr><tr><td>Video Swin-32</td><td>261</td><td>277</td><td>171</td><td>106</td><td>90</td><td>0.617</td><td>0.655</td><td>0.636</td></tr><tr><td>BMASH-16</td><td>261</td><td>256</td><td>159</td><td>97</td><td>102</td><td>0.621</td><td>0.609</td><td>0.615</td></tr><tr><td>Video Swin-16</td><td>261</td><td>302</td><td>175</td><td>127</td><td>86</td><td>0.580</td><td>0.671</td><td>0.622</td></tr></table>

Compared with temporal-window classification, full-video spotting is substantially more challenging because dense sliding-window inference over long broadcast videos accumulates false positive detections and introduces temporal localization errors. The addition of ball-motion features produces performance comparable to the Video Swin baseline, with BMASH-32 favoring higher recall while Video Swin-32 provides a slightly better balance between precision and recall.

Table 5 presents the results for each test video separately. Performance varies across the two matches, indicating that full-video spotting is influenced by factors such as camera viewpoint, event density, ball visibility, and visually similar aerial-ball situations. All models achieve higher precision on WB than on BN, while BN generally produces more false positive detections.

Table 5: Full-match header spotting results on the two broadcast test videos, reported separately for each video.
<table><tr><td>Video</td><td>Model</td><td>GT</td><td>Pred</td><td>TP</td><td>FP</td><td>FN</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="4">BN</td><td>BMASH-32</td><td>111</td><td>133</td><td>71</td><td>62</td><td>40</td><td>0.534</td><td>0.640</td><td>0.582</td></tr><tr><td>BMASH-16</td><td>111</td><td>137</td><td>75</td><td>62</td><td>36</td><td>0.547</td><td>0.676</td><td>0.605</td></tr><tr><td>Video Swin-32</td><td>111</td><td>127</td><td>70</td><td>57</td><td>41</td><td>0.551</td><td>0.631</td><td>0.588</td></tr><tr><td>Video Swin-16</td><td>111</td><td>160</td><td>85</td><td>75</td><td>26</td><td>0.531</td><td>0.766</td><td>0.627</td></tr><tr><td rowspan="4">WB</td><td>BMASH-32</td><td>150</td><td>157</td><td>104</td><td>53</td><td>46</td><td>0.662</td><td>0.693</td><td>0.678</td></tr><tr><td>BMASH-16</td><td>150</td><td>119</td><td>84</td><td>35</td><td>66</td><td>0.706</td><td>0.560</td><td>0.625</td></tr><tr><td>Video Swin-32</td><td>150</td><td>150</td><td>101</td><td>49</td><td>49</td><td>0.673</td><td>0.673</td><td>0.673</td></tr><tr><td>Video Swin-16</td><td>150</td><td>142</td><td>90</td><td>52</td><td>60</td><td>0.634</td><td>0.600</td><td>0.616</td></tr></table>

## 6 Conclusion

We presented BMASH, a ball-motion-aware framework for soccer header spotting in broadcast videos. BMASH combines Video Swinbased action representations with detector-derived ball-motion descriptors using lightweight late fusion, allowing ball motion to support the Video Swin based action understanding. We evaluate BMASH against EficientNet3D, TSM-ResNet50, and Video Swin baselines using game-level splits, with both temporal-window cliplevel classification and full-video spotting analysis.

At the clip windows level, BMASH-32 improved average precision from 0.9423 to 0.9530 and ROC-AUC from 0.9418 to 0.9507 compared with the matched Video Swin-32 baseline, while maintaining almost similar F1-score. This indicates that detector-derived ball fea tures improve the ranking of header and non-header samples. In the full-video setting, Video Swin-32 and BMASH-32 achieved nearly similar event-level F1-scores, with BMASH-32 detecting more true headers and obtaining higher recall. These results highlight the dificulty of continuous full-match spotting, where overlapping windows, visually similar airborne-ball actions, missed ball detections, and many false positives afect event-level performance.

The present study has some limitations. Full-video spotting was evaluated on only two held-out broadcast matches, so larger and more diverse full-match evaluations are needed to better assess generalization. In addition, the current ball descriptor summarizes simple detector statistics, such as visibility, position, scale, and displacement, but does not yet model full ball trajectories, speed, acceleration, or player-relative ball geometry.

Future work will extend the training and evaluation to larger and more diverse datasets, investigate trajectory-aware and playerrelative ball representations, improve post-processing for continuous spotting, and explore the integration of video predictions with wearable-sensor measurements for downstream head-impact analysis.

## Acknowledgments

This work was part ofFinland’s Ministry ofEducation and Culture’s Doctoral Education Pilot under Decision No. VN/3137/2024-OKM-6 (The Finnish Doctoral Program Network in Artificial Intelligence, AI-DOC). We thank CSC – IT Center for Science, Finland, for providing the computational resources used in this work. We also thank Vili Pesonen, Ronja Suomala, and Miikka Pasanen for their help with the manual annotation and verification ofsoccer header events in the broadcast videos used in this study.

## References

[1] Sara Akan and Songül Varlı. 2023. Use of deep learning in soccer videos analysis: survey. Multimedia Systems 29, 3 (2023), 897–915.

[2] Thomas Aston and Filipe Teixeira-Dias. 2025. Quantitative video analysis of head acceleration events: a review. Frontiers in Bioengineering and Biotechnology 13 (2025).

[3] Yassine Benzakour, Bruno Cabado, Silvio Giancola, Anthony Cioppa, Bernard Ghanem, and Marc Van Droogenbroeck. 2024. OSL-ActionSpotting: A unified library for action spotting in sports videos. In IEEE International Workshop on Sport, Technology and Research. 132–137.

[4] Anthony Cioppa, Adrien Deliege, Silvio Giancola, Bernard Ghanem, Marc Van Droogenbroeck, Rikke Gade, and Thomas B Moeslund. 2020. A context-aware loss function for action spotting in soccer videos. In IEEE/CVF conference on computer vision and pattern recognition. 13126–13136.

[5] Anthony Cioppa, Silvio Giancola, Vladimir Somers, Floriane Magera, Xin Zhou, Hassan Mkhallati, Adrien Deliège, Jan Held, Carlos Hinojosa, Amir M Man sourian, et al. 2024. SoccerNet 2023 challenges results. Sports Engineering 27, 2 (2024), 24.

[6] MMAction2 Contributors. 2020. OpenMMLab’s Next Generation Video Understanding Toolbox and Benchmark. https://github.com/open-mmlab/mmaction2.

[7] Mohamad Dalal, Artur Xarles, Anthony Cioppa, Silvio Giancola, Marc Van Droogenbroeck, Bernard Ghanem, Albert Clapés, Sergio Escalera, and Thomas B Moeslund. 2025. Action anticipation from soccernet football video broadcasts. In Computer Vision and Pattern Recognition Conference. 6080–6091.

[8] Adrien Deliege, Anthony Cioppa, Silvio Giancola, Meisam J Seikavandi, Jacob V Dueholm, Kamal Nasrollahi, Bernard Ghanem, Thomas B Moeslund, and Marc Van Droogenbroeck. 2021. Soccernet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. In IEEE/CVF conference on computer vision and pattern recognition. 4508–4519.

[9] Tanner M Filben, N Stewart Pritchard, Logan E Miller, Christopher M Miles, Jillian E Urban, and Joel D Stitzel. 2021. Header biomechanics in youth and collegiate female soccer. Journal ofbiomechanics 128 (2021), 110782.

[10] Silvio Giancola, Mohieddine Amine, Tarek Dghaily, and Bernard Ghanem. 2018. Soccernet: A scalable dataset for action spotting in soccer videos. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 1711– 1721.

[11] Silvio Giancola, Anthony Cioppa, Julia Georgieva, Johsan Billingham, Andreas Serner, Kerry Peek, Bernard Ghanem, and Marc Van Droogenbroeck. 2023. Towards active learning for action spotting in association football videos. In IEEE/CVF conference on computer vision and pattern recognition. 5098–5108.

[12] Silvio Giancola, Anthony Cioppa, Bernard Ghanem, and Marc Van Droogenbroeck. 2025. Deep learning for action spotting in association football videos. In Pattern Recognition and Computer Vision in the New AI Era. 427–459.

[13] Silvio Giancola and Bernard Ghanem. 2021. Temporally-aware feature pooling for action spotting in soccer broadcasts. In IEEE/CVF conference on computer vision and pattern recognition. 4490–4499.

[14] Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih Peng. 2019. Tracknet: A deep learning network for tracking high-speed and tiny objects in sports applications. In IEEE international conference on advanced video and signal based surveillance.

[15] Glenn Jocher and Jing Qiu. 2024. Ultralytics YOLO11. https://github.com/ ultralytics/ultralytics

[16] Paresh R Kamble, Avinash G Keskar, and Kishor M Bhurchandi. 2019. Ball tracking in sports: a survey. Artificial Intelligence Review 52, 3 (2019), 1655–1705.

[17] Paresh R Kamble, Avinash G Keskar, and Kishor M Bhurchandi. 2019. A deep learning ball tracking system in soccer videos. Opto-Electronics Review 27, 1 (2019), 58–69.

[18] Rebecca Kenny, Marko Elez, Adam Clansey, Naznin Virji-Babul, and Lyndia C Wu. 2024. Individualized monitoring of longitudinal heading exposure in soccer. Scientific Reports 14, 1 (2024), 1796.

[19] Ryan P. W. Kenny et al. 2022. Head Impact Exposure and Biomechanics in University Soccer Players. Annals ofBiomedical Engineering (2022).

[20] Jan Kern, Thomas Lober, Joachim Hermsdörfer, and Satoshi Endo. 2022. A neural network for the detection of soccer headers from wearable sensor data. Scientific Reports 12, 1 (2022), 18128.

[21] Jacek Komorowski, Grzegorz Kurzejamski, and Grzegorz Sarwas. 2019. Deepball: Deep neural-network ball detector. In International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications.

[22] Enora Le Flao, Gunter P Siegmund, and Robert Borotkanics. 2022. Head impact research using inertial sensors in sport: a systematic review of methods, demographics, and factors contributing to exposure. Sports Medicine 52, 3 (2022), 481–504.

[23] Ji Lin, Chuang Gan, and Song Han. 2019. TSM: Temporal Shift Module for Eficient Video Understanding. In IEEE/CVF International Conference on Computer Vision. 7083–7093.

[24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European Conference on Computer Vision. 740–755.

[25] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. 2022. Video swin transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3202–3211.

[26] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[27] Binod Naik, Mohammad F. Hashmi, and Neeraj D. Bokde. 2022. A Comprehensive Review of Computer Vision in Sports: Open Issues, Future Trends and Research Directions. Applied Sciences 12, 9 (2022), 4429.

[28] Arjun Raj, Lei Wang, and Tom Gedeon. 2025. Tracknetv4: Enhancing fast sports object tracking with motion attention maps. In IEEE International Conference on Acoustics, Speech and Signal Processing.

[29] Ahmad Rezaei and Lyndia C Wu. 2022. Automated soccer head impact exposure tracking using video and deep learning. Scientific reports 12, 1 (2022), 9282.

[30] Yuzhi Shi, Hiroaki Minoura, Takayoshi Yamashita, Tsubasa Hirakawa, Hironobu Fujiyoshi, Mitsuru Nakazawa, Yeongnam Chae, and Björn Stenger. 2022. Action spotting in soccer videos using multiple scene encoders. In International

Conference on Pattern Recognition. 3183–3189

[31] Joao VB Soares, Avijit Shah, and Topojoy Biswas. 2022. Temporally precise action spotting in soccer videos using dense detection anchors. In IEEE International Conference on Image Processing. 2796–2800.

[32] Renaud Vandeghen, Anthony Cioppa, and Marc Van Droogenbroeck. 2022. Semisupervised training to improve player and ball detection in soccer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3481–3490.

[33] Artur Xarles, Sergio Escalera, Thomas B Moeslund, and Albert Clapés. 2023. Astra: An action spotting transformer for soccer videos. In International Workshop on Multimedia Content Analysis in Sports. 93–102.

[34] Hao Xu, Arbind Agrahari Baniya, Sam Well, Mohamed Reda Bouadjenek, Richard Dazeley, and Sunil Aryal. 2025. Deep Learning for Sports Video Event Detection: Tasks, Datasets, Methods, and Challenges. arXiv preprint arXiv:2505.03991 (2025).

[35] Jinglin Xu, Guohao Zhao, Sibo Yin, Wenhao Zhou, and Yuxin Peng. 2024. Finesports: A multi-person hierarchical sports video dataset for fine-grained action understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21773–21782.

[36] Fucheng Zheng, Duaa Zuhair Al-Hamid, Peter Han Joo Chong, Cheng Yang, and Xue Jun Li. 2025. A review of computer vision technology for football videos. Information 16, 5 (2025), 355.