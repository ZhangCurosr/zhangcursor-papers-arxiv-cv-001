# Can We Anticipate Violence? Multimodal Learning from Pre-Incident Behavioral Cues

Sindhuja Penchala, Mohammed Yusuf Mujawar, Noorbakhsh Amiri Golilarz, Sudip Mittal, and Shahram Rahimi

Department of Computer Science, The University of Alabama, Tuscaloosa, AL 35487, USA {spenchala, mmujawar}@crimson.ua.edu, {namirigolilarz, sudip.mittal, srahimi1}@ua.edu

## Abstract

Detecting violence after it begins is important from recognizing behavioral cues that appear immediately beforehand. This work studies short-horizon pre-incident risk recognition from multimodal video signals. We construct a binary Normal-versus-Risky setting from temporally annotated XD-Violence clips, using 443 samples with source-level separation across training, validation, and test sets. Each sample consists of a variable-length pre-incident clip, with its duration determined by the observable behavioral context preceding the incident. The incident itself is excluded from all input clips. We evaluate three complementary information sources: facial-region appearance, temporally aligned audio, and body-motion features derived from tracked keypoints. Controlled ablations are performed with Swin-Tiny, ViT-Tiny, and DeiT-Tiny to measure the contribution of each modality under the same split. Results show that combining all modalities is more efective than using any other combination alone. The best configuration, Deit-Tiny with audio, facial appearance, and motion, achieves 91.21% accuracy, 88.96% balanced accuracy, 93.65% F1-score, and 96.38% ROC-AUC on the held-out test set. These results suggest that complementary appearance, acoustic, and kinematic cues provide useful evidence for recognizing elevated pre-incident risk.

Keywords: Pre-incident risk recognition, multimodal learning, behavioral cues, violence anticipation, behavior analysis.

## 1 Introduction

Violent incidents in public and monitored environments can escalate rapidly, leaving only a short window for timely intervention. Automated video understanding has therefore become increasingly important for applications such as public-space surveillance, transportation hubs, campuses, healthcare facilities, and online video moderation, where continuous human monitoring is often impractical. Existing video anomaly and violence-detection methods have made substantial progress in recognizing fights, assaults, shootings, and other abnormal events, supported by large-scale benchmarks such as UCF-Crime and XD-Violence [17, 22]. Multimodal approaches further improve recognition by combining complementary visual, motion, and audio cues, particularly when the visual stream is degraded by occlusion or poor illumination [14, 21, 22]. However, these systems, including online variants that operate causally, are designed to recognize violent or anomalous evidence already present in the observed sequence. For safety-critical applications, a complementary question is whether observable behavior before an incident can distinguish emerging risk from normal activity.

Recent work has expanded violence detection beyond single visual streams by incorporating complementary audio, motion, and semantic information. Following the audio-visual XD-Violence benchmark [22], attention-based fusion methods modeled cross-modal interactions between visual and audio features through co-attention and bilinear or additive fusion [14, 21]. Other work added caption-derived textual features to audio-visual input and strengthened multi-scale temporal modeling [13]. More recent methods improve cross-modal interaction through semantic feature alignment that also addresses modality asynchrony [10] and through gated fusion combined with multi-scale temporal modeling [1]. Although these approaches improve multimodal violence recognition, they are trained and evaluated to score frames that already contain violent evidence; pre-incident frames are treated as negatives, so early responses are counted as false alarms rather than as anticipation.

Treating early responses as anticipation requires a diferent formulation, where the model must make a decision using only behavior observed before violent evidence appears. This problem has received less attention, although human observers have been shown to anticipate dangerous CCTV incidents above chance [20]. Blunsden and Fisher studied pre-fight recognition using spatio-temporal cuboid features and hierarchical AdaBoost [2]. They showed that pre-segmented pre-fight sequences could be distinguished from normal, fight, and post-fight behavior. However, performance dropped on continuous video, where pre-fight frames were often confused with normal activity or fighting. Their continuous classifier also used a temporal window centered on the current frame, which could include observations after incident onset, and the evaluation was based on acted scenarios. Most recent violence-detection methods are still mainly reactive and are trained to recognize violence once it is already visible in the observed sequence [1, 10, 13, 14, 21]. This leaves an important gap: whether modern learned representations of human behavior can recognize short-horizon risk using only observations captured before an incident begins.

To address this gap, we study pre-incident risk recognition using only observations captured before annotated incident onset. The incident itself is excluded from the input, distinguishing our setting from event-present violence detection. We examine complementary cues from facial appearance, audio context, and pose-derived body motion, and evaluate them individually and in combination across multiple transformer backbones. Our objective is not to infer latent human intent, but to determine whether observable cues preceding an incident provide useful evidence of near-term risk.

• We formulate pre-incident risk recognition as a setting in which models observe only behavior captured before annotated incident onset, and curate a dataset of pre-onset clips with graded risk annotations.

• We evaluate facial, audio, and body-motion cues individually and in combination across multiple transformer backbones to measure the contribution of each source.

• We show that multimodal behavioral cues observed before incident onset carry discriminative evidence of near-term risk, supporting proactive rather than reactive violence analysis.

The remainder of this paper is organized as follows. Section 2 reviews prior work on video anomaly detection, multimodal violence recognition, feature fusion, and event anticipation. Section 3 presents the task formulation, multimodal representations, and fusion framework. Section 4 describes the dataset curation, experimental setup, ablation studies, results, and discussions. Finally, Section 5 concludes the paper.

## 2 Related Work

In this section, we review prior work on weakly supervised anomaly detection, multimodal violence recognition, generalizable video anomaly detection, and event anticipation.

## 2.1 Weakly Supervised Video Anomaly and Violence Detection

Weakly supervised video anomaly detection learns temporal abnormality from video-level labels without requiring dense frame annotations. Sultani et al. [17] introduced the UCF-Crime benchmark and a multiple-instance learning formulation for anomaly detection in long, untrimmed videos. Wu et al. [22] extended this setting to audio-visual violence detection through XD-Violence, while RTFM [18] improved temporal anomaly localization using feature-magnitude learning. These methods provide important foundations for violence and anomaly recognition, but primarily identify abnormal evidence within the observed video.

## 2.2 Multimodal Violence Detection

Multimodal violence detection has largely been developed on XD-Violence, combining visual features with audio and, in some methods, optical flow or generated captions. Methods difer mainly in how modalities interact, including co-attention and bilinear fusion [14], stacked self- and co-attention units [21], caption-augmented multi-scale temporal networks [13], sparse alignment of audio and flow into the RGB feature space [10], and gated fusion with multi-scale bottleneck transformers [1]. Graph-based fusion operators have also been proposed for integrating heterogeneous features in video anomaly detection [6]. Across these methods, the emphasis is on how violent evidence is combined across modalities once it is present, rather than on what the modalities reveal beforehand.

## 2.3 Video Anomaly Detection

Recent studies have also sought to make anomaly detection transfer beyond fixed training categories, often by leveraging vision-language models. [9] identifies local spatial patterns through imagetext alignment to generalize to novel anomalies, and LaGoVAD [12] allows anomaly definitions to be specified in natural language at inference time for open-world detection. SteerVAD [3] instead steers anomaly-sensitive attention heads within a frozen multimodal large language model, achieving competitive performance with little training data. Because these methods ground anomalies in semantic descriptions of the event itself, they broaden what can be detected but do not address the behavior that precedes an incident.

## 2.4 Action and Anomaly Anticipation

Anticipation methods infer upcoming events from preceding observations. The Anticipative Video Transformer [8] predicts next actions in egocentric video, and [4] introduced video anomaly anticipation as a task, together with the semi-supervised NWPU Campus benchmark and a model that jointly detects and anticipates anomalies. More recently, PULS [16] predicts future latent states with a video world model and reports an anticipatory advantage shortly before anomaly onset on UCF-Crime and XD-Violence. For violence specifically, [5] fuse visual, audio, and caption features and add a Siamese onset branch that models transitional dynamics around violence onset, while earlier work by [2] classified pre-fight behavior using hand-crafted features.

Our work difers from these approaches in three respects. First, in contrast to onset-aware detection, our inputs end before annotated onset and never include the incident itself. Second, whereas prior anomaly anticipation relies on generic scene representations, we focus on humancentered cues, namely facial appearance, audio, and body motion, and measure the contribution of each. Finally, rather than semi-supervised benchmarks recorded in fixed campus scenes, our clips are drawn from diverse, unconstrained XD-Violence videos spanning movies and in-the-wild footage, and are annotated with graded pre-incident risk.

![](images/128d91bcbc52b9a24070493d66e47a2386fd6e918dc5542153b00ff25640e92e.jpg)  
Figure 1: Overview of the proposed multimodal pre-incident risk classification framework. The preprocessing stage maps XD-Violence source videos into binary Normal and Risky clips and aligns facial appearance, audio, and pose-derived motion inputs. The facial stream uses a pretrained visual backbone followed by a temporal Transformer, the audio stream processes log-magnitude spectrograms using a three-stage CNN, and the motion stream models pose-derived kinematic features using a temporal Transformer. The resulting 256-dimensional visual, 128-dimensional audio, and 64-dimensional motion representations are concatenated and passed through a late-fusion classification head to predict Normal or Risky behavior.

## 3 Methodology

We formulate pre-incident recognition as binary clip classification using three synchronized modalities: facial appearance, audio, and pose-derived motion. As illustrated in Figure 1, each modality is encoded independently, and the resulting clip-level representations are combined through late feature fusion to predict either Normal or Risky.

## 3.1 Input Construction

For each annotated clip, we sample T = 32 ordered frames at 8 frames per second from an observation window ending at the clip boundary. This corresponds to a maximum duration of approximately four seconds. Shorter sequences are padded, and a validity mask prevents padded positions from

contributing to temporal attention. Audio and motion features are extracted from the same interval.

$$
\mathbf { z } _ { t } = \mathrm { G E L U } \left( \mathbf { W } _ { v } \mathbf { f } _ { t } + \mathbf { b } _ { v } \right) \in \mathbb { R } ^ { 2 5 6 } .\tag{1}
$$

DeiT layer 8·14×14  
DeiT-Tiny Feature-Depth Activation Maps Using Facial Expressions No-Risk Samples  
![](images/bf9c833e2472c527488085811791f304bd86620b2d9ad1c0669eaa9884a72c63.jpg)

![](images/6b043b9d1c6d39713906dde3689551349e7e9262f5d5dac5a3fe00ef05f4dae0.jpg)

![](images/72b22a3490368265f47ba721445eb1a5d9c764cfabe1c1a80d9dd680bb0a1e0d.jpg)  
(a) No-Risk samples.

![](images/dad3e9c0c8bbb39da646070776f3ab839b178637ecd6962180b287ea61a9a8ae.jpg)

![](images/a3d3c9cf495b7697757cf9abe454a8ce6dbee0b22feeda8872c2babf7bfa900d.jpg)  
DeiT layer 12 · 14×14

DeiT-Tiny Feature-Depth Activation Maps Using Facial Expressions Risky Samples  
![](images/e2b25ab63051fff3b525caf00e46189217ea8e75ac4cbe82488858034a65a756.jpg)

![](images/bba3e0b0d5b6136cee709415e266e5105d0a699d4cb48a50af4054c18ec92b72.jpg)

![](images/41d5b59c904c96fdc743be2b4bd08f4ffd0b7c9c62ab923e7fb25f46eb188aac.jpg)  
(b) Risky samples.

![](images/de4eada2f14b9b4a958f9a16bb15795a592b97f28da5a108d829eae74a4fb8bb.jpg)

![](images/aa44b76969cb3f80644bd6f3284d81b2a492b91c247e30f24f3aa7fe083140b6.jpg)  
Figure 2: Feature-depth activation maps from the DeiT-Tiny facial-appearance branch for representative No-Risk and Risky samples. The first column shows the original facial crop, followed by activation maps from layers 1, 4, 8, and 12. Red and yellow regions indicate relatively strong feature responses, green and cyan indicate moderate responses, and blue and purple indicate weaker responses within each layer. These colors show activation strength.

A one-layer temporal Transformer with eight attention heads aggregates the 32 frame representations. Learned positional embeddings preserve frame order, while a classification token produces

the final facial representation $\mathbf { e } _ { v } \in \mathbb { R } ^ { 2 5 6 }$ . The pretrained visual backbone remains frozen during multimodal training to reduce overfitting. Representative layer-wise activation maps from the facial-appearance encoder are presented in Figure 2.

## 3.2 Facial-Appearance Stream

Facial regions are localized using the available COCO-17 pose annotations. Visible facial landmarks are used when available; otherwise, the head region is estimated from the upper part of the person bounding box. For frames containing multiple people, the largest detected person is selected. Facial boxes are temporally smoothed, cropped with a small contextual margin, and resized to $2 2 4 \times 2 2 4$ pixels.

We evaluate ImageNet-pretrained Swin-Tiny, ViT-Tiny, and DeiT-Tiny backbones [7, 11, 19]. Each backbone independently encodes the facial crop at every temporal position. Its output is projected to a common 256-dimensional space:

## 3.3 Audio Stream

Audio is converted to mono, resampled to 16 kHz, normalized, and aligned with the visual observation window. We compute a standardized log-magnitude spectrogram using a 512-point short-time Fourier transform, a 400-sample Hann window, and a 160-sample hop.

The spectrogram is processed by a three-stage convolutional encoder with 32, 64, and 128 channels. Each stage contains convolution, batch normalization, GELU activation, and max pooling. Adaptive average pooling produces a 128-dimensional audio representation $\mathbf { e } _ { a } \in \mathbb { R } ^ { 1 2 8 }$ . Figure 3 visualizes the input acoustic representations and their transformation across the three audio-CNN stages.

![](images/d0b083f4bce5f545f8489239d43b2d9d0084743e044c03285c2401364c7db929.jpg)  
Figure 3: Class-averaged audio representations for Normal and Risky pre-incident clips using the DeiT-Tiny three-modality configuration. From left to right, the figure shows the mean absolute waveform, mean log-magnitude spectrogram, and learned audio representations at three successive network depths (32, 64, and 128 channels). The visualization illustrates how diferences in the input acoustic patterns are progressively transformed into higher-level feature representations by the audio branch.

Pose frame 3  
Pose frame 2  
Pose frame 2  
DeiT-Tiny Motion Feature Maps Using Keypoint Poses No-Risk Samples  
Pose frame 1  
Pose frame 3  
Pose frame 4  
![](images/b2fae771fa8334f0a2401648863e27e48adfc2394855e0b79bcdb160bee17a6a.jpg)  
(a) No-Risk samples.

DeiT-Tiny Motion Feature Maps Using Keypoint Poses Risky Samples  
Pose frame 1  
Pose frame 4  
![](images/756fabc3c54ef96c58312d01ca0acb3428222e70e01d20fdec8627bdcc5dc0e3.jpg)  
(b) Risky samples.  
Figure 4: Representative pose-based motion sequences for No-Risk and Risky pre-incident clips. Four temporally ordered frames are shown for each example. Red markers denote detected body joints, blue lines denote skeletal connections, and green boxes indicate person detections. These colors are visualization annotations and do not directly represent risk; classification is based on temporal changes in the detected keypoint positions and derived motion features.

## 3.4 Motion Stream

COCO-17 keypoints extracted by the YOLO pose [15] estimator are used to calculate 12 kinematic descriptors: mean and maximum speed, acceleration, and jerk; speed dispersion; mean and maximum wrist and ankle speeds; and overall keypoint-motion energy.

Motion measurements are normalized by body scale and aggregated across detected people using permutation-invariant mean and maximum statistics. After standardization using training-set statistics, the features are projected to 64 dimensions. A one-layer temporal Transformer with four attention heads produces the clip-level motion representation $\mathbf { e } _ { m } \in \mathbb { R } ^ { 6 4 }$ . Representative temporally ordered pose sequences used to derive these motion features are shown in Figure 4.

## 3.5 Multimodal Fusion and Classification

The three modality representations are concatenated:

$$
\mathbf { e } = \mathbf { e } _ { v } \parallel \mathbf { e } _ { a } \parallel \mathbf { e } _ { m } \in \mathbb { R } ^ { 4 4 8 } ,\tag{2}
$$

where the facial, audio, and motion streams contribute 256, 128, and 64 dimensions, respectively. Layer normalization, dropout, and a linear classifier transform the fused representation into Norma and Risky probabilities. A fixed threshold of 0.5 is used for binary classification.

The network is optimized using class-weighted, label-smoothed cross-entropy with $\ell _ { 2 }$ regularization:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { W C E } } + \lambda \mathcal { L } _ { \mathrm { r e g } } . } \end{array}\tag{3}
$$

## 4 Experiments and Results

We evaluate the proposed framework using curated pre-incident clips from XD-Violence. The experiments examine the contribution of facial appearance, audio, and body motion, and compare diferent visual backbones under the same data partitions. We also provide qualitative visualizations to better understand the information captured by each modality.

## 4.1 Experimental Setup

The experimental set was constructed from the XD-Violence dataset [22]. Source videos were manually reviewed to identify behavior occurring before a violent incident. For each Risky example, only the portion preceding the incident was retained, while the incident itself was excluded from the model input. Because the amount of observable pre-incident behavior varies across videos, the resulting clips have variable durations.

Each clip was manually categorized as No Risk, Low Risk, Medium Risk, or High Risk using visual appearance, body motion, and audio context. For binary classification, Low, Medium, and High-Risk clips were grouped into the Risky class, while No-Risk clips formed the Normal class. Clips originating from the same source video were kept within a single training, validation, or test partition to avoid information leakage. These labels represent observable pre-incident behaviora cues and are not intended to contain violence frames.

Table 1: Final ablation results for individual and combined pre-incident modalities across diferent model configurations. We report Accuracy, Balanced Accuracy, Precision, Recall, F1-score, and ROC-AUC on the held-out test set. Bold values indicate the best performance obtained for each metric across all evaluated configurations.
<table><tr><td>Ablation</td><td>Model</td><td>Accuracy</td><td>Bal. Acc.</td><td>Precision</td><td>Recall</td><td>F1</td><td>ROC-AUC</td></tr><tr><td>Motion only</td><td>Temporal Transformer</td><td>82.42</td><td>76.08</td><td>82.86</td><td>93.55</td><td>87.88</td><td>86.21</td></tr><tr><td>Audio only</td><td>Audio CNN</td><td>67.03</td><td>73.05</td><td>92.11</td><td>56.45</td><td>70.00</td><td>78.75</td></tr><tr><td>Facial appearance only</td><td>ViT-Tiny</td><td>87.91</td><td>85.62</td><td>90.48</td><td>91.94</td><td>91.20</td><td>91.94</td></tr><tr><td>Facial appearance only</td><td>Swin-Tiny</td><td>89.01</td><td>87.35</td><td>91.94</td><td>91.94</td><td>91.94</td><td>95.22</td></tr><tr><td>Facial appearance only</td><td>DeiT-Tiny</td><td>87.91</td><td>85.62</td><td>90.48</td><td>91.94</td><td>91.20</td><td>94.22</td></tr><tr><td>Audio + Motion</td><td>Audio CNN + Temporal Transformer</td><td>78.02</td><td>75.61</td><td>85.00</td><td>82.26</td><td>83.61</td><td>88.71</td></tr><tr><td>Facial appearance + Motion</td><td>Swin-Tiny + Temporal Transformer</td><td>84.62</td><td>83.20</td><td>90.00</td><td>87.10</td><td>88.52</td><td>93.83</td></tr><tr><td>Audio + Facial appearance</td><td>ViT-Tiny</td><td>86.81</td><td>83.90</td><td>89.06</td><td>91.94</td><td>90.48</td><td>93.10</td></tr><tr><td>Audio + Facial appearance</td><td>Swin-Tiny</td><td>89.01</td><td>86.43</td><td>90.62</td><td>93.55</td><td>92.06</td><td>96.11</td></tr><tr><td>Audio + Facial appearance</td><td>DeiT-Tiny</td><td>87.91</td><td>84.71</td><td>89.23</td><td>93.55</td><td>91.34</td><td>93.66</td></tr><tr><td>Audio + Facial appearance + Motion</td><td>ViT-Tiny</td><td>86.81</td><td>84.82</td><td>90.32</td><td>90.32</td><td>90.32</td><td>94.33</td></tr><tr><td>Audio + Facial appearance + Motion</td><td>Swin-Tiny</td><td>90.11</td><td>87.24</td><td>90.77</td><td>95.16</td><td>92.91</td><td>96.11</td></tr><tr><td>Audio + Facial appearance + Motion</td><td>DeiT-Tiny</td><td>91.21</td><td>88.96</td><td>92.19</td><td>95.16</td><td>93.65</td><td>96.38</td></tr></table>

## 4.2 Results and Discusion

The complete test-set comparison across the single-, dual-, and three-modality configurations is reported in Table 1 for all evaluated configurations. Among the tested models, the three-modality DeiT-Tiny configuration achieved the strongest overall performance, with 91.21% accuracy, 88.96% balanced accuracy, 92.19% precision, 95.16% recall, 93.65% F1-score, and 96.38% ROC-AUC on the held-out test set.

## 4.2.1 Ablation Studies

We perform controlled ablation experiments to measure the contribution of three modalities: facial appearance, audio, and body motion. Each modality is first evaluated independently, followed by pairwise combinations and the complete three-modality configuration. The audio branch uses the Audio CNN, while the motion branch uses the Temporal Transformer described in Section 3.

For configurations containing facial appearance, we compare ViT-Tiny, Swin-Tiny, and DeiT-Tiny to examine whether multimodal performance depends on the selected visual backbone. All configurations use the same training, validation, and test partitions, allowing the efect of individual modalities and multimodal fusion to be compared consistently.

## 4.2.2 Risk-Level Analysis

Although the model is trained only with binary Normal and Risky labels, we also analyze its predictions using the original No-Risk, Low-Risk, Medium-Risk, and High-Risk annotations. As shown in Figure 5, the predicted risk score is lowest for No-Risk clips and generally increases from Low to Medium and High Risk. This trend is consistent across the training, validation, and test sets.

This result is important because the model is never explicitly trained to separate Low, Medium, and High Risk. Even so, its predictions follow the progression of the original risk levels. This suggests that the model is capturing meaningful changes in pre-incident behavior, rather than only learning a simple Normal-versus-Risky boundary. The result also provides evidence that usefu behavioral cues can appear before the annotated incident begins.

![](images/6966376cdf0097ddb97c3628c075e2b616ec7a39327b96ae9fcf3802b1c96e7e.jpg)  
Figure 5: Mean predicted Risky scores across the original risk levels for the three-modality DeiT-Tiny model. Although the model is trained for binary Normal-versus-Risky classification, the average predicted risk score generally increases from No Risk to Low, Medium, and High Risk across the training, validation, and test partitions. The dashed line indicates the fixed decision threshold of 0.50.

## 4.2.3 Qualitative Analysis

We examine the internal representations of the three modality branches to understand the complementary information captured from the pre-incident observation window. Figure 2 shows activation maps from successive layers of the DeiT-Tiny facial-appearance encoder for representative No-Risk and Risky samples. Earlier layers preserve localized facial structure and fine-grained appearance boundaries, while deeper layers produce increasingly abstract and spatially distributed response patterns. The resulting representations demonstrate that the visual branch captures both facial appearance and surrounding contextual information relevant to clip-level risk classification.

Figure 3 presents class-averaged audio representations for 29 Normal and 62 Risky test clips. Normal clips exhibit comparatively stable acoustic patterns, whereas Risky clips show stronger temporal variation and increased acoustic activity toward the end of the observation window. These diferences remain visible across successive convolutional stages, highlighting that the audio encoder progressively transforms the input waveform and spectrogram into discriminative higher-leve representations.

Figure 4 presents temporally ordered pose sequences used to construct the motion representation. The No-Risk examples maintain relatively stable postures and trajectories across the observation window. In contrast, the Risky examples exhibit more pronounced changes in body configuration, limb movement, spatial position, and interpersonal interaction. These temporal variations are summarized by the kinematic descriptors and subsequently encoded by the motion Transformer.

Together, the visualizations show that the three branches capture complementary evidence: the audio branch represents temporal acoustic activity, the facial branch captures hierarchical appearance information, and the motion branch models changes in body dynamics. Their combination therefore provides a richer representation of pre-incident behaviour than any individual modality alone.

## 5 Conclusion

This work examined whether multimodal behavioral cues observed before annotated incident onset can support short-horizon pre-incident risk recognition. By combining facial appearance, synchronized audio, and pose-derived motion, the framework captures complementary information about appearance, acoustic context, and body dynamics rather than relying on a single source of evidence. The experimental results show that multimodal integration provided better results, with the threemodality DeiT-Tiny configuration achieving the strongest overall performance of 91.21% accuracy, 88.96% balanced accuracy, 93.65% F1-score, and 96.38% ROC-AUC on the held-out test set. These findings suggest that useful discriminative information can be present in the period preceding an incident, highlighting the potential of moving violence analysis beyond event-present detection toward earlier risk recognition. In future work, we plan to expand the dataset with more diverse pre-incident scenarios and develop a more robust multimodal model capable of capturing tempora relationships across modalities and recognizing risk earlier and more reliably before incident onset.

## Acknowledgments

The authors acknowledge the support and resources provided by the Bioinspired Robotics, AI, Imaging and Neurocognitive Systems (BRAINS) Laboratory and Predictive Analytics and Technology Integration(PATENT) Laboratory at The University of Alabama.

## References

[1] Bilal Ahmad, Mustaqeem Khan, and Muhammad Sajjad. Gated fusion networks for multimodal violence detection. AI, 6(10):259, 2025.

[2] Scott J Blunsden and Robert B Fisher. Pre-fight detection-classification of fighting situations using hierarchical adaboost. In International Conference on Computer Vision Theory and Applications, volume 1, pp. 303–308. SCITEPRESS, 2009.

[3] Zhaolin Cai, Fan Li, Huiyu Duan, Lijun He, and Guangtao Zhai. Steering and rectifying latent representation manifolds in frozen multi-modal llms for video anomaly detection. arXiv preprint arXiv:2602.24021, 2026.

[4] Congqi Cao, Yue Lu, Peng Wang, and Yanning Zhang. A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20392–20401. IEEE, 2023.

[5] Chih-Yung Chang, Syu-Jhih Jhang, Yu-Ting Chin, I-Hsiung Chang, and Diptendu Sinha Roy. A multimodal framework for violent behavior recognition in surveillance videos. Neurocomputing, 684:133532, 2026.

[6] Dexuan Ding, Lei Wang, Liyun Zhu, Tom Gedeon, and Piotr Koniusz. Learnable expansion of graph operators for multi-modal feature fusion. In International Conference on Learning Representations, volume 2025, pp. 74263–74285, 2025.

[7] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[8] Rohit Girdhar and Kristen Grauman. Anticipative video transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 13485–13495. IEEE, 2021.

[9] Yalong Jiang. Local patterns generalize better for novel anomalies. In International Conference on Learning Representations, volume 2025, pp. 49969–49995, 2025.

[10] Wenping Jin, Li Zhu, and Jing Sun. Aligning first, then fusing: A novel weakly supervised multimodal violence detection method. Knowledge-Based Systems, 322:113709, 2025.

[11] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 9992–10002. Ieee, 2021.

[12] Zihao Liu, Xiaoyu Wu, Jianqin Wu, Xuxu Wang, and Linlin Yang. Language-guided open-world video anomaly detection under weak supervision. In International Conference on Learning Representations, volume 2026, pp. 153197–153224, 2026.

[13] Gwangho Na, Jaepil Ko, and Kyungjoo Cheoi. Leveraging multi-modality and enhanced temporal networks for robust violence detection. Machine Learning and Knowledge Extraction, 6 (4):2422–2434, 2024.

[14] Wen-Feng Pang, Qian-Hua He, Yong-jian Hu, and Yan-Xiong Li. Violence detection in videos based on fusing visual and audio information. In ICASSP 2021-2021 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 2260–2264. IEEE, 2021.

[15] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779–788, 2016.

[16] Abu Anas Ibn Samad. Latent clarity: Bridging world-model kinematics to semantic manifolds for video anomaly anticipation. arXiv preprint arXiv:2607.03558, 2026.

[17] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6479–6488. IEEE, 2018.

[18] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4955– 4966. IEEE, 2021.

[19] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-eficient image transformers & distillation through attention. In International conference on machine learning, pp. 10347–10357. PMLR, 2021.

[20] Tom Troscianko, Alison Holmes, Jennifer Stillman, Majid Mirmehdi, Daniel Wright, and Anna Wilson. What happens next? the predictability of natural behaviour viewed through cctv cameras. Perception, 33(1):87–101, 2004.

[21] Dong-Lai Wei, Chen-Geng Liu, Yang Liu, Jing Liu, Xiao-Guang Zhu, and Xin-Hua Zeng. Look, listen and pay more attention: Fusing multi-modal information for video violence detection. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1980–1984. IEEE, 2022.

[22] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In European conference on computer vision, pp. 322–339. Springer, 2020.

Test score distribution  
![](images/a6c0107ee505f152dd0ac26d13c60d11c8f573c82a6131c923a7b3575ac312d9.jpg)  
Figure 6: Distribution of predicted Risky probabilities for Normal and Risky clips in the held-out test set. Most Normal clips receive scores near zero, whereas most Risky clips receive scores near one, showing clear separation between the two classes. The dashed line indicates the fixed classification threshold of 0.5. Blue observations to the right of the threshold correspond to false positives, while orange observations to the left correspond to false negatives.

## A appendix

Figure 6 shows the distribution of predicted Risky probabilities produced by the final three-modality DeiT-Tiny model on the held-out test set. Most Normal clips receive scores close to zero, whereas most Risky clips receive scores close to one, showing a clear separation between the two classes. The dashed line represents the fixed decision threshold of 0.5.

The binary prediction is obtained as

$$
\hat { y } = \left\{ \begin{array} { l l } { \mathrm { N o r m a l } , } & { p ( \mathrm { R i s k y } \mid \mathcal { X } ) < 0 . 5 , } \\ { \mathrm { R i s k y } , } & { p ( \mathrm { R i s k y } \mid \mathcal { X } ) \ge 0 . 5 , } \end{array} \right.\tag{4}
$$

where X represents the multimodal input. Normal samples appearing to the right of the threshold correspond to false positives, while Risky samples appearing to the left correspond to false negatives. The limited overlap between the two distributions indicates that the model separates most test clips with predicted scores away from the decision boundary, although a small number of dificult or confidently incorrect cases remain.

Table 2 summarizes the annotation criteria used to assign the four original risk levels. The labels are based on observable cues across visual appearance, body motion, and audio. No-Risk clips contain stable and non-aggressive behavior, while Low- and Medium-Risk clips represent progressively stronger behavioral changes and escalation. High-Risk clips contain stronger pre-incident cues occurring close to the annotated incident onset. For the binary experiments, Low-, Medium-, and High-Risk clips are grouped into the Risky class, while No-Risk clips form the Normal class.

Table 2: Pre-Incident Risk Annotation Rubric
<table><tr><td rowspan=1 colspan=7>Label       Vision                   Motion                  Audio                   Risk Interpre-tation</td></tr><tr><td rowspan=2 colspan=7>No Risk      Individuals  ap-     Minimal    body      Background ambi-      The environmentpear calm and non- movement and stable ence or normal conver- appears safe and sta-aggressive.         No trajectories. Normal sation only. No shout- ble with no visible in-threatening posture, interpersonal distance ing, screaming, im- dicators of aggressionweapon     visibility, with smooth and pact sounds, or sud- or escalation.or confrontation is predictable motion den loud audio spikes.observed. People may patterns. No suddenbe walking, standing, acceleration or ag-sitting, or interacting gressive gestures.normally.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=5>Low         Mild suspicious or</td><td rowspan=10 colspan=2>Slight increase in     Slightly elevated     Early behavioralmovement intensity audio activity such as indicators   suggestinterpersonal focus. louder speech, crowd potential tension orradual    approach noise, or environmen- discomfort, but noetween   individu- tal tension without immediate threat isls, mild trajectory explicit     aggressive observed.changes, or increased sounds.ody orientation to-ard another person.</td></tr><tr><td rowspan=1 colspan=5>Risk   tense visual behavior</td></tr><tr><td rowspan=1 colspan=5>may appear, such as or</td></tr><tr><td rowspan=1 colspan=5>prolonged1staring, G</td></tr><tr><td rowspan=1 colspan=5>followingbehavior, b</td></tr><tr><td rowspan=1 colspan=5>defensive    posture, a</td></tr><tr><td rowspan=1 colspan=5>or unusual attention</td></tr><tr><td rowspan=1 colspan=5>toward another in- b</td></tr><tr><td rowspan=1 colspan=5>dividual. No direct w</td></tr><tr><td rowspan=1 colspan=5>aggression occurs.</td></tr><tr><td rowspan=1 colspan=5>Medium     Clear behavioral</td><td rowspan=1 colspan=1>Noticeablein-</td><td rowspan=6 colspan=1>Raised     voices,      Behavioralpat-escalation becomes crease in motion arguments, aggressive ternsindicateantensity,      abrupt speech tone, sudden significant likelihoodjectory changes, audio spikes, or in- of an aggressive eventeduced interpersonal creased environmental occurring soon. Thedistance, and rapid disturbance may be situation is unstablemovements present.                   and escalating.may indicatingpossibleconflict escalation.</td></tr><tr><td rowspan=1 colspan=5>Risk</td><td rowspan=1 colspan=1>creaseinmotion</td></tr><tr><td rowspan=1 colspan=5>visible. Aggressive i</td><td></td></tr><tr><td rowspan=1 colspan=5>body posture, arm tra</td><td></td></tr><tr><td rowspan=2 colspan=5>raising, threatening rgestures,     chasingbehavior, or visible bodyconfrontationappear.</td><td></td></tr><tr><td rowspan=1 colspan=1>bodicat</td></tr><tr><td rowspan=6 colspan=7>Screaming,   im-     The scene showsRisk                   behavior and rapid motion pact sounds, weapon- strong pre-incidentis strongly visible. patterns, aggressive related        sounds, aggression cues indi-Direct confrontation, acceleration, physical explosions, or highly cating an imminentattack preparation, engagement attempts, elevated audio in- violent or dangeroushandling, collision-like move- tensity may occur event.striking posture, or ment, or intense immediately beforephysically aggressive approachdynamics the event.interaction is clearly are observed.observed before theincident starts.</td></tr><tr><td rowspan=1 colspan=5>Riskincidentbehavior</td><td rowspan=1 colspan=1>andrapidmotion</td><td rowspan=1 colspan=1>pact sound</td></tr><tr><td rowspan=1 colspan=5>is strongly visible.</td></tr><tr><td rowspan=3 colspan=5>三</td><td rowspan=1 colspan=1>Direct confrontation,</td></tr><tr><td rowspan=1 colspan=2>ng postard,ing,</td></tr><tr><td rowspan=1 colspan=1>lly aggressive</td></tr></table>