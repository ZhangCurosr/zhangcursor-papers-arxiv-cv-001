# 4D Reconstruction with Events & RGB

Dulanga Weerakoon<sup>1</sup>, Darshana Rathnayake<sup>1</sup>, Van Minh Do<sup>1</sup>, Hemanth Sabbella<sup>3</sup>, Kaichen Zhou<sup>2</sup>, Fangneng Zhan<sup>2</sup>, Alok Prakash<sup>1</sup>, Sanjay E. Sarma<sup>2</sup>, Archan Misra<sup>3</sup>, and Paul Pu Liang<sup>2</sup>

Singapore-MIT Alliance for Research & Technology Centre <sup>2</sup> Massachusetts Institute of Technology 3 Singapore Management University

![](images/542ae77b7553d60ed79fcb86f8abfb5e83b0cb2d797b02bee2afc041e4a8d43d.jpg)  
Fig. 1: Overview of our event-augmented 4D reconstruction pipeline. Pairs of RGB frames and their corresponding event voxel grids are processed within a sliding window and passed through MonST3R to predict per-window pointmaps. Camera intrinsics and extrinsics estimated from overlapping windows align consecutive pointmaps into a single coordinate system, yielding a dynamic point cloud that captures scene geometry and motion over time.

Abstract. Reconstructing dynamic 3D scenes over time (4D reconstruction) from a single moving camera is central to robotics and AR/VR applications, yet state-of-the-art feed-forward methods rely solely on RGB frames and therefore inherit the failure modes of conventional cameras, including motion blur and temporal aliasing under fast or non-uniform camera and object motion. Event cameras report per-pixel brightness changes asynchronously at microsecond resolution, ofering a complementary signal that is largely immune to these artifacts, but whether this modality can be incorporated into existing RGB-based 4D reconstruction pipelines without a full architectural redesign remains an open question. In this preliminary study, we investigate a minimal extension: we augment Monst3R, an existing RGB-only 4D reconstruction model, with an additional event-camera input branch, leaving the rest of the architecture unchanged. Across 3 benchmark datasets, this simple RGBevent fusion reduces Abs Rel by 3.4% on average relative to the RGB-only Monst3R baseline, and a sensitivity analysis shows that the improvement

grows with the magnitude of camera and object motion, with the largest gains observed under high object motion conditions where RGB-only reconstruction degrades most. These early results suggest that even a lightweight, non-invasive fusion of event data can meaningfully improve 4D reconstruction robustness, motivating a more thorough architectural investigation in future work.

Keywords: 4D Reconstruction, 3D Vision, Event Processing

## 1 Introduction

Reconstructing dynamic three-dimensional scenes over time, referred to as 4D reconstruction, is a foundational problem in computer vision with applications in AR/VR content creation, robotics, autonomous driving, and digital twins. Recent advances in vision-language models (VLMs), such as CL4D [6], have also begun exploring dynamic 4D point cloud representations as an alternative to conventional 2D video inputs, enabling richer spatio-temporal and geometric reasoning and further motivating accurate 4D reconstruction directly from videos. Recent feed-forward models, such as MonST3R [21], recover per-frame depth and camera pose directly from RGB video without per-scene optimization. However, because these models rely solely on RGB frames, they inherit the failure modes of conventional cameras: motion blur and temporal aliasing under fast or non-uniform camera and object motion, precisely the conditions where accurate motion estimation matters most.

Event cameras ofer a complementary sensing modality. By reporting perpixel brightness changes asynchronously at microsecond resolution, they capture motion information largely unafected by blur or frame-rate limitations. Prior work has explored event-based depth and pose estimation as standalone tasks, but whether an event stream can be incorporated into an existing RGB-based 4D reconstruction pipeline without a full architectural redesign remains less explored.

In this preliminary work, we present an early exploration of event-augmented 4D reconstruction through a minimal extension to the MonST3R pipeline as shown in Figure 1: we augment MonST3R with an additional event-voxel input branch, leaving the rest of the architecture unchanged, and fuse it with RGB features via concatenation before processing through the standard MonST3R pipeline. We hypothesize that RGB and event data are complementary—RGB provides spatial information for depth, while events provide motion cues for modeling dynamic scenes, particularly under fast object and camera motion—and that even this minimalist, concatenation-based fusion, without a more principled fusion mechanism, is suficient to realize this complementarity.

To test this hypothesis, following prior work, we evaluate on benchmark datasets for both camera pose estimation and video depth estimation. We observe that fusing event voxels and RGB achieves higher reconstruction accuracy than using either modality alone, and that this benefit is particularly pronounced under high object motion.

While this work is a preliminary study on 4D reconstruction with RGB and events, we argue that more principled multi-modal fusion mechanisms are needed to achieve substantially higher reconstruction accuracy, given the fundamental diferences between the two modalities: RGB is a constant-frame-rate, synchronous modality, while events are adaptive and asynchronous.

We make the following key contributions:

– We extend MonST3R with a minimal event-voxel input branch, requiring no other architectural changes, and show that fusing RGB and event voxels reduces depth and pose estimation error relative to RGB-only MonST3R, an event-to-RGB translation baseline, and an event-only variant.

– We conduct a sensitivity study under varying camera and object motion, establishing that RGB-event fusion provides the largest gains under high object motion.

## 2 Related Work

Feedforward 4D reconstruction. Prior work on 4D reconstruction primarily used RGB frames as input. Early work such as DUSt3R [17] proposed predicting a separate pointmap for each pair of RGB frames. MonST3R [21] extended the DUSt3R [17] pointmap paradigm to dynamic scenes. Both MonST3R and DUSt3R predict pointmaps followed by a postprocessing step that aligns them into a single coordinate system. Works such as CUT3R [16] and Fast3R [20] instead unify pointmap prediction and alignment into a single feed-forward pipeline, directly predicting aligned pointmaps in one coordinate system without a separate postprocessing step. More recently, VGGT [15] and PAGE4D [23] scale up training data and model capacity with transformer-based architectures, achieving more accurate 3D/4D scene reconstruction.

Despite this large body of work on 4D reconstruction, all prior methods take a single monocular RGB video as input and are feed-forward, and none incorporate an event stream. Robustness to fast motion remains an open challenge across these methods, motivating the present work: fusing event and RGB data for 4D reconstruction.

Event cameras and RGB event fusion. Event cameras have been studied extensively for their high temporal resolution, negligible motion blur, and high dynamic range [3]. RGB and event fusion has been explored successfully for related but narrower tasks, including video deblurring [8], frame interpolation/extrapolation [11, 13], and event-based visual odometry and SLAM [14], where events are typically used to recover information that has already been lost in blurred or dropped RGB frames. Its potential for 4D scene reconstruction, that is, jointly recovering time-varying geometry and appearance rather than a single deblurred or interpolated frame, remains comparatively underexplored.

Event-augmented 4D reconstruction. More recently, EAG3R [19] presented the first event-augmented 4D reconstruction method, combining RGB and event data to target low-light conditions. Its design and evaluation are organized around illumination robustness: an SNR-aware fusion module favors RGB features in well-lit regions and event features in dark regions, and its central experiments vary illumination from day to night while holding the motion regime relatively constant. This leaves comparatively less direct evidence for a complementary motivation for event cameras: their ability to capture fast, non-rigid scene dynamics that a fixed-frame-rate RGB camera under-samples or blurs, independent of illumination. The present work targets precisely this gap, evaluating explicitly under varying camera and object motion rather than varying illumination.

## 3 Methodology

This section describes our approach for 4D reconstruction using multi-modal RGB and event inputs. We first introduce the event representation, followed by the proposed network architecture and the dataset preparation pipeline.

An asynchronous event stream over the time interval $( t , t + \varDelta )$ is defined as

$$
\mathcal { E } _ { ( t , t + \Delta ) } = \{ e _ { i } = ( p _ { i } , x _ { i } , y _ { i } , t _ { i } ) \} _ { i = 0 } ^ { N } ,\tag{1}
$$

where $p _ { i } \in \{ - 1 , 1 \}$ denotes the event polarity corresponding to an increase or decrease in brightness, $( x _ { i } , y _ { i } ) \in \mathbb { R } ^ { H \times W }$ represents the pixel coordinates, and $t _ { i }$ denotes the event timestamp.

Our framework builds upon the MonST3R architecture and extends it to support multi-modal inputs consisting of RGB images and event data. The network takes two RGB images, $I _ { 1 } , I _ { 2 } \in \mathbb { R } ^ { W \times H \times 3 }$ , together with two event voxel grids, $\mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } \in \mathbb { R } ^ { W \times H \times B }$ , where $B = 5$ denotes the number of temporal bins. Here, H and W represent the image height and width, respectively.

The event stream is converted into a voxel grid representation [24] by accumulating events within each temporal bin as

$$
\mathcal { E } _ { V } ( b , x , y ) = \sum _ { e _ { i } \in \mathcal { E } _ { ( t , t + t _ { 1 } ) } } p _ { i } \mathbf { 1 } \left\{ b = \left\lfloor B \cdot \frac { t _ { i } - t } { t _ { 1 } } \right\rfloor , x = x _ { i } , y = y _ { i } \right\} ,\tag{2}
$$

where $\mathbf { 1 } \{ \cdot \}$ is the indicator function.

## 3.1 Model Architecture

We extend MonST3R to jointly process RGB images and event voxel grids while preserving the original output representations. For each view, the 3-channel RGB image and the 5-channel event voxel grid are concatenated to form an 8-channel input tensor. Since the pretrained ViT encoder expects a 3-channel input, we introduce a lightweight adapter consisting of a single convolution layer that projects the 8-channel input to a 3-channel feature representation. The adapted inputs from the two views are then processed by the shared Siamese ViT encoder, followed by the original MonST3R decoder and prediction heads without further architectural modifications.

## 3.2 Dataset Preparation

The datasets used in this work provide only RGB image sequences. To obtain the corresponding event data, we use the V2E [7] simulator to generate asynchronous event streams between consecutive RGB frames. The simulator first interpolates intermediate RGB frames, efectively increasing the temporal resolution of the original video sequence. It then generates events by modeling the brightness changes between the interpolated frames. Finally, the generated event streams are accumulated into voxel grid representations according to Equation 2, which are used together with the corresponding RGB images as inputs to the proposed network.

## 3.3 Training Details

Similar to MonST3R, we train the proposed variants on a number of prior datasets, each augmented with synthetically generated event data as described in Section 3.2: PointOdyssey [22], TartanAir [18], and Spring [9]. Remaining training configurations are identical to the original MonST3R implementation.

## 4 Results

Evaluation Datasets. We evaluate on Sintel [1], Bonn [10], KITTI [4], and PointOdyssey [22], and adopt the standard evaluation metrics used in the 4D reconstruction literature, following MonST3R [21].

Baseline. We primarily use MonST3R as the base model with diferent input modalities. (a) RGB is the MonST3R model with the pretrained weights from the original paper implementation. (b) RGB (HyperE2VID) also uses the pretrained MonST3R weights, but RGB frames are generated from event voxels using HyperE2VID [2] instead of the sensor’s original RGB frames. (c) Ev. Vox. retrains MonST3R using event voxels (bin size = 5) alone, without RGB input. (d) RGB + Ev. Vox. retrains MonST3R using concatenated RGB and event voxels (bin size = 5).

Camera Pose Estimation. Table 1 compares camera pose estimation error on Sintel across four input variants to MonST3R. RGB combined with event voxels achieves the lowest error on all three metrics (ATE mean, RPE trans mean, RPE rot mean). Reconstructing RGB from events via HyperE2VID before passing it to MonST3R degrades performance relative to using the original RGB frames directly, indicating that direct fusion of event voxels is more efective than a translation-based approach, which may inherently introduce errors during eventto-RGB conversion. Event voxels alone generally achieve a higher error than RGB + Event voxels, underscoring the importance of fusing RGB and event data for accurate depth reconstruction.

Table 1: Camera pose estimation on Sintel with MonST3R as the base model. Bold indicates the best result per column. ↓: lower is better.
<table><tr><td>Input</td><td>ATE mean ↓ RPE trans mean ↓ RPE rot mean ↓</td><td></td></tr><tr><td>RGB</td><td>0.108</td><td>0.042 0.732</td></tr><tr><td>RGB (HyperE2VID)</td><td>0.132</td><td>0.067 0.767</td></tr><tr><td>Ev. Vox.</td><td>0.154</td><td>0.076 0.532</td></tr><tr><td>RGB + Ev. Vox.</td><td>0.106</td><td>0.038 0.476</td></tr></table>

Table 2: Video depth estimation comparison across three datasets with MonST3R as the base model. Bold indicates the best result per column. ↓: lower is better, ↑: higher is better.
<table><tr><td></td><td colspan="2">Sintel</td><td colspan="2">Bonn</td><td colspan="2">KITTI</td></tr><tr><td>Input</td><td></td><td>Abs Rel ↓ δ&lt;1.25 ↑|Abs Rel ↓ δ&lt;1.25 ↑|Abs Rel ↓ δ&lt;1.25 ↑</td><td></td><td></td><td></td><td></td></tr><tr><td>RGB</td><td>0.345</td><td>56.2</td><td>0.065</td><td>96.3</td><td>0.106</td><td>89.3</td></tr><tr><td>RGB (HyperE2VID)</td><td>0.464</td><td>53.3</td><td>0.085</td><td>93.3</td><td>0.143</td><td>79.9</td></tr><tr><td>Ev. Vox.</td><td>0.47</td><td>50.2</td><td>0.088</td><td>92.5</td><td>0.165</td><td>78.3</td></tr><tr><td>RGB + Ev. Vox.</td><td>0.331</td><td>58.1</td><td>0.064</td><td>96.4</td><td>0.101</td><td>90.6</td></tr></table>

Video Depth Estimation. Table 2 reports video depth estimation error on three datasets: Sintel, Bonn, and KITTI. As with camera pose estimation, RGB combined with event voxels achieves the lowest error on both metrics across all three datasets. A MonST3R variant trained with event voxels alone achieves the highest error, exceeding even the translation-based approach of converting events to RGB with an event-to-RGB model before passing them to MonST3R. This further reinforces that spatial information from RGB, coupled with motionspecific cues from event data, is necessary for accurate 4D reconstruction.

Event+RGB against Camera Motion. In Figure 2, we showcase the trajectories predicted by the RGB + Ev. Vox. variant with MonST3R as the base model. We observe that smooth, uniform camera motion (Figure 2b) is easier to estimate, while more non-uniform motion (Figure 2a) accumulates higher error over time, resulting in inaccurate trajectories.

Furthermore, in Figure 3a we measure the sensitivity of reconstruction error to camera motion on a subset of the PointOdyssey test set with MonST3R. We divide the PointOdyssey test set into three levels of camera motion using the ground-truth camera poses. RGB and event voxel fusion achieves lower Log RMSE across all three levels of camera motion, underscoring the importance of event and RGB fusion for accurate 4D reconstruction.

Event+RGB against Object Motion. In Figure 3b, we further measure the sensitivity of reconstruction error to object motion. In this sensitivity study, we divide the PointOdyssey test set into two levels of object motion based on ground-truth depth variation. While RGB and event voxel fusion achieves lower Log RMSE across both levels of object motion, gains are more pronounced under high object motion, achieving a 0.327 lower Log RMSE in this category. This further reinforces the importance of the complementary event and RGB modalities for accurate 4D reconstruction, particularly under high object motion.

![](images/f6c4d4ca3e7015ac5235e075431d8574c79fef0a3415b2e90d39bd175ec440df.jpg)  
(a) ambush\_4

![](images/655e600b122942559e51fa37f59a45e2c904e2b65b4aa7ea2fd93d1b90a13e30.jpg)  
(b) shaman\_3  
Fig. 2: Predicted vs. ground-truth camera trajectories on two Sintel sequences with RGB + Ev. Vox.

Visualization of RGB+Event Voxels based 4D Reconstruction. In Figure 4, we showcase point cloud reconstructions produced by the RGB + Ev. Vox. variant, with MonST3R as the base model, from pairs of RGB frames and corresponding event voxels. Across these visualizations, we observe that scene geometry remains intact using our proposed fusion of RGB and event data for 4D reconstruction.

## 5 Discussion & Future Work

This preliminary study shows that a minimal event-camera branch added to MonST3R reduces both depth and pose estimation error across the benchmarks considered, with the largest gains under high camera and object motion. The design is deliberately simple: an event-voxel branch concatenated with RGB features, with no other architectural changes. This leaves several directions open for a more thorough follow-up study.

– Event-specific encoders. We encode events as dense voxel grids processed by the same backbone used for RGB. Architectures designed for the sparse, asynchronous structure of event data, such as Spiking Neural Networks (SNNs) [5] or event transformers [12], may extract more informative motion cues than a standard RGB-based backbones.

– Adaptive multi-modal fusion. We fuse modalities by concatenation, which treats RGB and event features uniformly regardless of local motion. Events are inherently adaptive to motion, while RGB provides uniform spatial coverage independent of it; fusion mechanisms such as cross-attention or gating could weight each modality’s contribution based on local motion cues rather than combining them uniformly.

![](images/208131426ec5361a8406c5ee5b856c5497eeafba97d865cb252757a9a04f318d.jpg)  
(a) Camera motion

![](images/4065c2f5658ed2497c1e00423566d79603caef5379a796991866def89c5c6a65.jpg)  
(b) Object motion  
Fig. 3: Sensitivity of reconstruction error to camera and object motion magnitude on a subset of the PointOdyssey test set with MonST3R. RGB + Ev. Vox. consistently achieves lower Log RMSE than RGB alone, with the gap widening under high object motion (Figure 3b). Camera motion is divided into three levels (Low, Medium, High) based on the ground-truth camera poses, and object motion is divided into two levels (Low, High) based on the ground-truth depth variation.

– More Recent 4D base models. We build on MonST3R, which is no longer the most accurate feed-forward 4D reconstruction model. Applying the same event-branch extension to more recent architectures, such as VGGT [15] or PAGE4D [23], would test whether the benefit of event fusion generalizes across base models rather than being specific to MonST3R.

– Real event data. Sintel, Bonn, KITTI and PointOdyssey do not natively contain event streams, so the event modality used in our evaluation is synthetically generated using V2E simulator. Validating these gains on real event-camera captures is an important next step before drawing conclusions about real-world deployment.

## 6 Conclusion

We present a preliminary study on 4D reconstruction using a combination of RGB frames and event voxels. To this end, we propose a minimalist design that concatenates event voxels with RGB frames before passing them through MonST3R, retrained to incorporate this additional modality. Across multiple datasets for camera pose estimation and video depth estimation, we observe that combining event voxels and RGB frames achieves lower reconstruction error than either modality alone, indicating that spatial information from RGB and motion cues from event voxels play complementary roles in accurate 4D reconstruction.

![](images/3068ce2cb836a61e461d35663ded1a1b2dab06949d7cef2b6f64dcc9dac938a4.jpg)  
(a)

![](images/619528b677858e25f6407e3d3a850ba59251c62256028159cd0b83418533b012.jpg)  
(b)

![](images/5b26e7bc37f94464a18c3aac9be83b58f164c4196408508a9bcd4f079b4313ea.jpg)  
(c)

![](images/f3cbea88e45c8851d668cb3813e4320db5dc6e97bc78f189ca53d17050d4323e.jpg)  
(d)  
Fig. 4: Qualitative 4D reconstructions produced by the RGB + Ev. Vox. variant with MonST3R as the base model.

This preliminary work lays the foundation for more principled multi-modal fusion approaches combining events and RGB for 4D reconstruction, which we leave for future work.

## 7 Acknowledgments

This research was supported by the Singapore National Research Foundation (NRF), Prime Minister’s Ofice, Singapore, under its Campus for Research Excellence and Technological Enterprise (CREATE) programme. The Mens, Manus, and Machina (M3S) is an interdisciplinary research group (IRG) of the Singapore-MIT Alliance for Research and Technology (SMART) centre.

## References

1. Butler, D.J., Wulf, J., Stanley, G.B., Black, M.J.: A naturalistic open source movie for optical flow evaluation. In: A. Fitzgibbon et al. (Eds.) (ed.) European Conf. on Computer Vision (ECCV). pp. 611–625. Part IV, LNCS 7577, Springer-Verlag (Oct 2012)

2. Ercan, B., Eker, O., Saglam, C., Erdem, A., Erdem, E.: Hypere2vid: Improving event-based video reconstruction via hypernetworks. IEEE Transactions on Image Processing 33, 1826–1837 (2024)

3. Gallego, G., Delbrück, T., Orchard, G., Bartolozzi, C., Taba, B., Censi, A., Leutenegger, S., Davison, A.J., Conradt, J., Daniilidis, K., et al.: Event-based vision: A survey. IEEE transactions on pattern analysis and machine intelligence 44(1), 154–180 (2020)

4. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The kitti dataset. The international journal of robotics research 32(11), 1231–1237 (2013)

5. Ghosh-Dastidar, S., Adeli, H.: Spiking neural networks. International journal of neural systems 19(04), 295–308 (2009)

6. Hewagamage, K., Senavirathne, I., Amarasinghe, S., Gallella, H., Weerakoon, D., Subbaraju, V., Rodrigo, R.: Cl4d: Contrastive language–4d pretraining for visionlanguage reasoning in dynamic scenes. In: European Conference on Computer Vision. pp. 594–612. Springer (2026)

7. Hu, Y., Liu, S.C., Delbruck, T.: v2e: From video frames to realistic dvs events. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 1312–1321. IEEE (2021)

8. Kim, T., Cho, H., Yoon, K.J.: Frequency-aware event-based video deblurring for real-world motion blur. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 24966–24976. IEEE (2024)

9. Mehl, L., Schmalfuss, J., Jahedi, A., Nalivayko, Y., Bruhn, A.: Spring: A highresolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4981–4991. IEEE (2023)

10. Palazzolo, E., Behley, J., Lottes, P., Giguère, P., Stachniss, C.: ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. In: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2019), https://www.ipb.uni-bonn.de/pdfs/palazzolo2019iros.pdf

11. Rathnayake, D., Weerakoon, D., Radhakrishnan, M., Misra, A.: Neurolidar: Adaptive frame rate depth sensing via neuromorphic event-lidar fusion. arXiv preprint arXiv:2605.16805 (2026)

12. Sabater, A., Montesano, L., Murillo, A.C.: Event transformer. a sparse-aware solution for eficient event data processing. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 2676–2685. IEEE (2022)

13. Tulyakov, S., Gehrig, D., Georgoulis, S., Erbach, J., Gehrig, M., Li, Y., Scaramuzza, D.: Time lens: Event-based video frame interpolation. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16150–16159. IEEE (2021)

14. Vidal, A.R., Rebecq, H., Horstschaefer, T., Scaramuzza, D.: Ultimate slam? combining events, images, and imu for robust visual slam in hdr and high-speed scenarios. IEEE Robotics and Automation Letters 3(2), 994–1001 (2018)

15. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5294–5306. IEEE (2025)

16. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10510–10522. IEEE (2025)

17. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20697–20709. IEEE (2024)

18. Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., Scherer, S.: Tartanair: A dataset to push the limits of visual slam. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 4909–4916. IEEE (2020)

19. Wu, X., Yu, Y., Lyu, X., Huang, Y., Wang, B., Zhang, B., Wang, Z., Qi, X.: Eag3r: Event-augmented 3d geometry estimation for dynamic and extreme-lighting scenes. Advances in Neural Information Processing Systems 38, 73314–73342 (2026)

20. Yang, J., Sax, A., Liang, K.J., Henaf, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21924–21935. IEEE (2025)

21. Zhang, J., Herrmann, C., Hur, J., Jampani, V., Cole, F., Sun, D., Yang, M.H., et al.: Monst3r: A simple approach for estimating geometry in the presence of motion. In: International Conference on Learning Representations. vol. 2025, pp. 82863–82886 (2025)

22. Zheng, Y., Harley, A.W., Shen, B., Wetzstein, G., Guibas, L.J.: Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In: ICCV (2023)

23. Zhou, K., Wang, Y., Chen, G., Beaudouin, G., Zhan, F., Liang, P., Wang, M.: Page-4d: Disentangled pose and geometry estimation for vggt-4d perception. In: International Conference on Learning Representations. vol. 2026, pp. 36401–36414 (2026)

24. Zhu, A.Z., Yuan, L., Chaney, K., Daniilidis, K.: Unsupervised event-based learning of optical flow, depth, and egomotion. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 989–997. IEEE (2019)