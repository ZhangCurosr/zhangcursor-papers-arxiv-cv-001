# Beyond Pixel Similarity: Task-Aware Evaluation of GAN-Based Synthetic Sonar Data for Robotic Perception

Hannan Ejaz Keen<sup>1</sup>, Muhammad Moazam Fraz <sup>2</sup> and Karsten Berns<sup>3</sup>

Abstract— Synthetic data can reduce the cost of collecting and annotating training data for robotic perception, but generating sensor observations that preserve the characteristics relevant to downstream perception remains challenging, particularly for sonar imagery. In this work, we investigate whether conventional image-fidelity metrics adequately reflect the downstream perception performance of GAN-generated synthetic sonar data. We employ a Pix2Pix conditional generative adversarial network with four discriminator configurations characterized by different receptive fields: PixelGAN, PatchGAN-16, PatchGAN-70, and ImageGAN. The models are trained using sonar imagery from two datasets and evaluated using conventional image-fidelity metrics, including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE). To complement these pixel-level measures with task-oriented evaluation, YOLOX-S, YOLOX-L, and Faster R-CNN detectors are trained exclusively on real sonar imagery and subsequently evaluated on the GAN-generated images using identical test samples and annotations across all discriminator configurations. The results reveal a discrepancy between image-fidelity and downstream object-detection performance: the configuration achieving the best SSIM, PSNR, and MSE does not consistently yield the best detection performance. In particular, PatchGAN configurations achieve strong downstream detection results despite not achieving the highest pixel-level similarity scores. These findings suggest, for the datasets and models considered, pixellevel image-fidelity metrics alone may not consistently capture the task-relevant realism of synthetic sonar observations and motivate the use of task-aware evaluation for synthetic sensor data intended for robotic perception.

## I. INTRODUCTION

Reliable perception in underwater robotics increasingly relies on imaging sonar because acoustic sensing remains effective in environments where optical cameras are degraded by limited illumination, turbidity, and suspended particles. However, sonar imagery exhibits sensor- and environmentdependent characteristics, including speckle noise, target highlights, acoustic shadows, reverberation, and strong variation with range, viewing angle, and seabed conditions. These characteristics make the development of robust sonar perception models challenging and increase the demand for large and diverse annotated datasets. In practice, collecting and annotating such datasets requires specialized equipment and repeated surveys across different environments, making real-world data acquisition costly and time-consuming [1], [2].

![](images/c16167e5b49a650b1a2c440c3b4ada1df999b6aec8461980db6df76956d40d84.jpg)  
(a)

![](images/5f5cbbe1d4bbb68bfe6e3b59f0e76e4eb3eb92a70ba6402a5d614d40963fde73.jpg)  
(b)  
Fig. 1: Comparative Visualization of Sonar Data: (a) Real sonar image (b) Simulated sonar image [3].

Synthetic data provides a potential alternative by enabling large numbers of labeled observations to be generated without repeated real-world acquisition. Previous studies have demonstrated that synthetic sonar observations generated from simulated underwater scenes can support downstream perception [1]. However, conventional sonar simulation can be difficult to model accurately because realistic observations depend on acoustic propagation, scattering, reverberation, sensor characteristics, and environmental interactions as shown in Fig. 1. Consequently, simulation may compute scene geometry while failing to replicate the appearance characteristics of real sonar imagery, limiting its usefulness for perception.

Generative adversarial networks (GANs) provide one approach for learning this appearance transformation. GANbased methods have been applied to generate or translate sonar imagery from simulated or conditioned representation and have demonstrated potential for downstream underwater perception [4], [5], [6]. These studies motivate the use of learned image translation for synthetic sonar generation. However, they also raise a fundamental research question:

## RQ: Does image similarity to real sonar data indicate how useful synthetic sonar data is for robotic perception?

Synthetic images are commonly evaluated using metrics such as Structural Similarity Index (SSIM), Peak Signalto-Noise Ratio (PSNR), and Mean Squared Error (MSE). Although these metrics quantify similarity to a reference image, they primarily measure pixel-level or local structural features [7]. They do not directly assess whether taskrelevant characteristics, such as target boundaries, acoustic shadows, and local contrast patterns, are preserved sufficiently for object detection. Prior work on synthetic data and sonar perception has similarly highlighted the importance of evaluating generated data according to downstream task performance rather than visual similarity alone [8]-[9].

![](images/07f7a02b1bb4a9188fdb20636e63e1e068c130a63cf86d5292bf0f6707997d29.jpg)  
Fig. 2: Pix2Pix-based synthetic sonar generation and evaluation pipeline with different discriminator receptive fields

This issue is particularly relevant in conditional image-toimage translation. In Pix2Pix [10], the discriminator evaluates image realism at a particular spatial scale determined by its receptive field. A pixel-level discriminator focuses on local intensity statistics, while PatchGAN discriminators evaluate increasingly larger local regions; an image-level discriminator instead evaluates broader spatial consistency. These different receptive fields may therefore influence whether the generator prioritizes local structures or global appearance characteristics.

In this work, we investigate whether conventional imagefidelity metrics adequately reflect the downstream perception performance of GAN-generated synthetic sonar data. Within a Pix2Pix framework, we compare four discriminator configurations: PixelGAN, PatchGAN-16, PatchGAN-70, and ImageGAN. The generated images are evaluated using SSIM, PSNR, and MSE and, independently through YOLOX-S, YOLOX-L, and Faster R-CNN detectors. The detectors are trained exclusively on real sonar imagery and subsequently evaluated on the generated images, with identical test samples and annotations used across all discriminator configurations. The experiments were originally conducted as part of the authors’ doctoral research [2] and are revisited here from the perspective of task-aware synthetic-sensor evaluation.

Our results reveal a discrepancy between image-fidelity evaluation and downstream detection performance in the experimental settings considered. The discriminator configuration achieving the strongest pixel-level fidelity does not consistently provide the strongest downstream detection performance. In particular, PatchGAN configurations frequently achieve higher detector performance despite lower SSIM, PSNR, or MSE values. These findings indicate that pixel-level image-fidelity metrics alone may not consistently characterize the task-relevant realism of synthetic sonar ob-

servations.

## II. METHODOLOGY

## A. Sonar Datasets and Conditioning

We use two sonar datasets with annotations suitable for conditional image generation and downstream object detection: the Marine Debris dataset [11] and the Underwater Acoustic Target Detection (UATD) dataset [12]. The datasets differ in acquisition conditions and therefore provide distinct sonar-image characteristics. The Marine Debris data is acquired in a controlled pool environment, while UATD contains imagery acquired in a more variable river environment.

For conditional image generation, object masks are used as the input representation. The Marine Debris dataset provides segmentation masks. UATD provides bounding-box annotations; therefore, binary masks are generated by assigning pixels inside each annotated bounding box to the object class and the remaining pixels to the background. This provides a common conditioning representation for both datasets.

Because the source sonar data is acquired as video sequences, consecutive frames can contain substantial redundancy. Similar images are therefore filtered during preprocessing. In addition, data augmentation is investigated to improve generation robustness; mirroring and random jitter are retained, while other tested transformations introduce undesirable artifacts [2].

## B. Pix2Pix-Based Sonar Image Generation

We employ a conditional GAN based on Pix2Pix [10]. The generator uses a U-Net architecture, while the discriminator is varied to investigate the effect of receptive field. Across all experiments, the generator architecture, training data, preprocessing, and optimization protocol are kept fixed. An overview of the experimental pipeline is shown in Fig. 2.

The conditional adversarial objective is combined with a (L<sub>1</sub>) reconstruction loss:

$$
G ^ { * } = \arg \operatorname* { m i n } _ { G } \operatorname* { m a x } _ { D } L _ { c G A N } ( G , D ) + \lambda L _ { L 1 } ( G )\tag{1}
$$

where the adversarial term encourages the generated sonar image to appear consistent with the target domain and the (L1) term encourages correspondence with the paired reference image [10].

Four discriminator configurations are evaluated (Fig. 2):

• PixelGAN: pixel-level discrimination with a minimal receptive field;

• PatchGAN-16: discrimination over local (16 × 16) regions;

• PatchGAN-70: discrimination over local (70 × 70) regions;

• ImageGAN: discrimination over the full image, corresponding to a (286 × 286) receptive field.

This controlled design isolates the effect of discriminator spatial scale on the generated sonar imagery.

## C. Evaluation Protocol

The generated images are evaluated from two complementary perspectives.

Image fidelity: SSIM, PSNR, and MSE are computed between each generated image and its corresponding reference sonar image. Higher SSIM and PSNR and lower MSE indicate greater image-level similarity.

Task-level evaluation: YOLOX-S, YOLOX-L, and Faster R-CNN are trained exclusively on real sonar imagery. The trained detectors are then applied to generated sonar images from each discriminator configuration. The same test samples and annotations are used for all four configurations. We report Average Precision (AP) averaged over IoU thresholds from 0.50 to 0.95, together with recall. We denote this metric as AP throughout the paper. The detectors are trained only on real sonar data and then tested on the generated images. This allows us to assess how useful the synthetic images are for detectors trained on real sonar data.

We refer to this property as task-relevant realism, meaning the extent to which a generated sonar image preserves features useful for downstream detection. This term does not imply complete physical or sensor-level realism.

TABLE I: Image-fidelity performance of discriminator configurations. Best result in each metric/dataset is highlighted.
<table><tr><td>Discriminator</td><td colspan="3">Marine Debris Dataset</td><td colspan="3">UATD Dataset</td></tr><tr><td rowspan="3">PixelGAN PatchGAN-16</td><td>SSIM</td><td>PSNR</td><td>MSE</td><td>SSIM</td><td>PSNR</td><td>MSE</td></tr><tr><td>0.586</td><td>24.265</td><td>49.532</td><td>0.755</td><td>31.845</td><td>21.078</td></tr><tr><td>0.546</td><td>23.090</td><td>52.079</td><td>0.729</td><td>30.995</td><td>23.759</td></tr><tr><td>PatchGAN-70</td><td>0.561</td><td>23.558</td><td>51.209</td><td>0.749</td><td>31.331</td><td>23.014</td></tr><tr><td>ImageGAN</td><td>0.548</td><td>23.192</td><td>52.174</td><td>0.760</td><td>32.034</td><td>20.800</td></tr></table>

## III. RESULTS AND DISCUSSION

## A. Image-Fidelity Results

Table I summarizes the image-fidelity performance on the Marine Debris and UATD datasets.

The results show that the preferred discriminator depends on the dataset. PixelGAN obtains the strongest SSIM, PSNR, and MSE results on Marine Debris, whereas ImageGAN achieves the strongest values on UATD. Thus, conventional image-fidelity metrics would favor different discriminator configurations depending on the dataset.

However, numerical image similarity does not fully describe the observed appearance. Fig. 3 shows that configurations with strong pixel-level scores can generate smoother or blurrier images, whereas PatchGAN configurations can preserve sharper local structures.

For the Marine Debris dataset, PixelGAN achieves the highest SSIM and PSNR among the evaluated configurations. However, as shown in Fig. 3a, its generated images appear comparatively blurred, whereas PatchGAN produces sharper local structures. Similarly, for the UATD dataset, although ImageGAN achieves the highest SSIM, PSNR, and lowest MSE, Fig. 3b shows that the ImageGAN-generated image contains fewer details than those produced by the PatchGAN configurations.

These observations suggest that conventional pixel-level fidelity metrics do not necessarily reflect the characteristics of synthetic sonar images that are most relevant for downstream perception.

## B. Downstream Detection Results

Table II reports detector performance on generated sonar images. The detectors were trained only on real sonar imagery and then evaluated on the generated images.

TABLE II: Downstream object-detection performance on generated sonar imagery. Entries show AP@[0.50:0.95] / Recall.
<table><tr><td rowspan=1 colspan=1>Discriminator</td><td rowspan=1 colspan=1>YOLOX-S</td><td rowspan=1 colspan=1>YOLOX-L</td><td rowspan=1 colspan=1>Faster R-CNN</td></tr><tr><td rowspan=1 colspan=4>Marine Debris Dataset</td></tr><tr><td rowspan=1 colspan=1>PixelGANPatchGAN-16PatchGAN-70ImageGAN</td><td rowspan=1 colspan=1>0.76 / 0.540.81  / 0.560.81 / 0.550.79 / 0.56</td><td rowspan=1 colspan=1>0.67 / 0.460.76 / 0.510.72 / 0.480.71 / 0.49</td><td rowspan=1 colspan=1>0.66 / 0.430.66 / 0.430.67  / 0.420.61 / 0.45</td></tr><tr><td rowspan=1 colspan=4>UATD Dataset</td></tr><tr><td rowspan=1 colspan=1>PixelGANPatchGAN-16PatchGAN-70ImageGAN</td><td rowspan=1 colspan=1>0.62 / 0.290.67  / 0.330.72 / 0.340.68 / 0.34</td><td rowspan=1 colspan=1>0.61 / 0.300.71 / 0.360.74 / 0.360.69 / 0.33</td><td rowspan=1 colspan=1>0.55 / 0.250.54 / 0.260.58 / 0.290.64 / 0.32</td></tr></table>

The downstream results reveal a different ranking from the image-fidelity metrics. On Marine Debris, PixelGAN achieves the highest SSIM, PSNR, and lowest MSE, but PatchGAN-16 and PatchGAN-70 obtain higher YOLOX-S AP, while PatchGAN-16 obtains the highest YOLOX-L AP. On UATD, ImageGAN provides the strongest image-fidelity scores, yet PatchGAN-70 obtains the highest YOLOX-S and YOLOX-L AP.

The discrepancy is particularly clear when comparing the best image-level and task-level configurations. For UATD, ImageGAN achieves SSIM = 0.760, PSNR = 32.034 dB, and MSE = 20.800, but PatchGAN-70 achieves higher YOLOX-S and YOLOX-L AP values of 0.72 and 0.74, respectively.

![](images/1f819f93f004bd7fac52ad1e4002e081349ec8463784fa50dfdd24b922446f39.jpg)  
Fig. 3: Qualitative comparison of generated sonar images across discriminator configurations and datasets.

Similarly, on Marine Debris, PixelGAN has the strongest image-fidelity scores, whereas PatchGAN-16 achieves the strongest YOLOX-S and YOLOX-L AP.

These results indicate, in our experiments, maximizing similarity to a reference image does not necessarily maximize the preservation of features required by a detector trained on real sonar imagery.

## C. Discussion: Image Fidelity Versus Task-Relevant Realism

The observed mismatch can be explained by the different properties measured by the two evaluation approaches. SSIM, PSNR, and MSE reward close correspondence to the reference image at the pixel or local-structure level. A generated image may therefore obtain a favorable score by reproducing overall intensity and structural statistics while smoothing or modifying small target structures.

In contrast, the detector-based evaluation is sensitive to whether generated images retain cues that are discriminative for object localization. The stronger downstream performance of PatchGAN configurations suggests that image characteristics beyond pointwise similarity are important for synthetic sonar perception. One possible explanation is that discriminator configurations operating over local regions encourage image characteristics that are more useful to the detector, although this mechanism is not directly isolated in our experiments. However, the results do not indicate a universally optimal discriminator: the preferred configuration varies with dataset and detector.

The main finding is therefore not that one particular discriminator is universally superior, but that the ranking of generation methods depends on the evaluation criterion. These results suggest that image-fidelity metrics should be complemented with task-aware evaluation when synthetic sonar imagery is intended for robotic perception.

Importantly, the detector transfer experiment provides evidence of task-relevant realism, not a complete measurement of physical sensor realism. Likewise, the present study does not quantify a reduction of the simulator-to-real domain gap because it does not directly compare raw simulator outputs with real sonar under the same transfer protocol. Instead, it demonstrates that different metrics of synthetic-image quality can lead to different conclusions about the same generated data.

## IV. CONCLUSION

This work investigates whether conventional imagefidelity metrics adequately characterize the task-relevant realism of GAN-generated synthetic sonar imagery. Within a Pix2Pix framework, four discriminator configurations (PixelGAN, PatchGAN-16, PatchGAN-70, and ImageGAN) are evaluated using SSIM, PSNR, and MSE as well as downstream object detection with YOLOX-S, YOLOX-L, and Faster R-CNN are trained exclusively on real sonar data.

Across the two sonar datasets considered, the configuration achieving the strongest image-fidelity scores does not consistently achieve the strongest object-detection performance. PatchGAN configurations frequently produce stronger downstream detection despite lower pixel-level similarity. These results indicate that SSIM, PSNR, and MSE capture only part of the quality relevant to synthetic sonar perception and should therefore be complemented with task-aware evaluation.

For synthetic sonar data intended for robotic perception, these results support evaluating data according to its intended use rather than image similarity alone. Future work should extend the analysis to additional sonar sensors and datasets and directly compare raw simulator outputs, generated observations, and real sonar to quantify broader domain-gap effects.

## REFERENCES

[1] S. Lee, B. Park, and A. Kim, “Deep learning from shallow dives: Sonar image generation and training for underwater object detection,” CoRR, vol. abs/1810.07990, 2018. [Online]. Available: http://arxiv.org/abs/1810.07990

[2] H. E. Keen, “Traversability mapping in post-flood environment,” doctoralthesis, Rheinland-Pfalzische Technische Universit¨ at¨ Kaiserslautern-Landau, 2024. [Online]. Available: https://nbnresolving.de/urn:nbn:de:hbz:386-kluedo-83477

[3] W.-S. Choi, D. R. Olson, D. Davis, M. Zhang, A. Racson, B. Bingham, M. McCarrin, C. Vogt, and J. Herman, “Physics-based modelling and simulation of multibeam echosounder perception for autonomous underwater manipulation,” Frontiers in Robotics and AI, vol. 8, p. 706646, 2021.

[4] M. Sung, J. Kim, M. Lee, B. Kim, T. Kim, J. Kim, and S.-C. Yu, “Realistic sonar image simulation using deep learning for underwater object detection,” Int. J. Control Autom. Syst., vol. 18, no. 3, pp. 523– 534, Mar. 2020.

[5] A. Noren, “Enhancing simulated sonar images with cyclegan for deep´ learning in autonomous underwater vehicles,” 2021.

[6] M. Jegorova, A. I. Karjalainen, J. Vazquez, and T. Hospedales, “Fullscale continuous synthetic sonar data generation with markov conditional generative adversarial networks,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 3168– 3174.

[7] H. E. Keen, A. Haider, and K. Berns, “Denoising and segmentation of sonar images for rescue operations,” in ISR Europe 2023; 56th International Symposium on Robotics, 2023, pp. 426–431.

[8] L. Liu, M. Muelly, J. Deng, T. Pfister, and L.-J. Li, “Generative modeling for small-data object detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

[9] Q. Ma, L. Jiang, W. Yu, R. Jin, Z. Wu, and F. Xu, “Training with noise adversarial network: A generalization method for object detection on sonar image,” in 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 2020, pp. 718–727.

[10] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” CVPR, 2017.

[11] D. Singh and M. Valdenegro-Toro, “The marine debris dataset for forward-looking sonar semantic segmentation,” in 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021, pp. 3734–3742.

[12] K. Xie, J. Yang, and K. Qiu, “A dataset with multibeam forwardlooking sonar for underwater object detection,” Scientific Data, vol. 9, no. 1, p. 739, 2022.