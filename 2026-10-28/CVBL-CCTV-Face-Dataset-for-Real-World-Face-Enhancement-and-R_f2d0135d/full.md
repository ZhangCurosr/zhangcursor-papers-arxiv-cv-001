# CVBL-CCTV-Face Dataset for Real-World Face Enhancement and Recognition in Surveillance

Manish Das, Ajay Kumar Yadav, Shiv Ram Dubey, Satish Kumar Singh Computer Vision and Biometrics Lab, Indian Institute of Information Technology, Allahabad {pmm2023001, rsi2022502, srdubey, sk.singh}@iiita.ac.in

Abstract—In real-world surveillance settings, facial captures from closed-circuit television systems suffer from degradations such as low resolution, motion blur, noise, and occlusions, which undermine dependable face recognition. In this study, we propose a framework for super-resolution and identity recognition, aimed at surveillance applications. For this purpose, we present the CVBL-CCTV-Face Dataset which is a dataset of real-world images taken in an unpaired manner where each pair consists of a low-quality (LQ) image of a face captured by a 8MP IP Dome CCTV camera together with a high-quality (HQ) reference image that is taken in controlled settings. In order to cope with the various problems posed by such datasets, we construct a modular system that involves face detection, deep learning-based superresolution of low-quality face images, and identity recognition through L2-normalized embeddings extracted by an embedding model. Enhancement models are trained and tested according to three different setups, and the system is deployed for identity verification in a real-time unconstrained environment.

Index Terms—CCTV surveillance, image super-resolution, deep learning, real-time enhancement, unpaired dataset

## I. INTRODUCTION

Benchmarking systems for face recognition has largely been done on face images obtained from studios and synthetically degraded images. However, there has never been a benchmarking database for closed circuit television (CCTV) images containing faces with multiple uncontrolled degradations such as low resolution because of the distance between the object and the camera, motion blur due to motion of the subject, sensor noise, partial occlusion and wide pose variation. Thus, a need for a dataset with real LQ CCTV face images alongside HQ face images taken separately from the subjects in an unpaired manner arises. This motivates the creation of the CVBL-CCTV-Face Dataset. In this regard, we propose the CVBL-CCTV-Face Dataset which is a real world and unpaired collection of Low Quality (LQ) face images taken from a 8MP IP dome CCTV camera alongside High Quality (HQ) face images taken separately. The dataset includes two versions, Version Aand Version B, differing in the number of HQ references per identity, enabling analysis under varying reference richness. To validate the dataset, we build a modular benchmark pipeline (see Fig. 1) consisting of MTCNN [1] face detection, transformer/diffusion-based enhancement, and L2- distance identity matching. Enhancement models are evaluated under three configurations as illustrated in Fig. 2: (i) pretrained models, (ii) fine-tuning on synthetically degraded HQ images, and (iii) fine-tuning on real LQ samples. Finally, we demonstrate the real-time surveillance system that processes live CCTV streams, enhancing detected faces and verifying identities with bounding box overlays (see Fig. 3). The key contributions of this work are:

![](images/f63baec08ed15e54891163e1a55155884aa36bf94ef6cb2b8712b428dda5ebac.jpg)  
Fig. 1. Overview of the proposed face enhancement and recognition pipeline integrating face detection, enhancement, and identity recognition.

![](images/be80aa341cefd438497b6443165e57aa564f21b2cf0104d030c14a985961555a.jpg)  
Fig. 2. Training and evaluation configurations: (i) Pretrained models, (ii) Finetuning on synthetic HQ degradation, (iii) Fine-tuning on real LQ samples. Emb Extractor denotes the embedding extraction model.

CVBL-CCTV-Face Dataset: A real-world, unpaired dataset of 223 identities with authentic CCTV-degraded LQ images and separately acquired HQ references. Unlike prior datasets that simulate degradation synthetically, ours captures compound real-world distortions with no spatial alignment between LQ–HQ pairs.

• Benchmark Pipeline: A validation framework with MTCNN face detection, four transformer/diffusion enhancement models, and L2 embedding-based recognition, also deployed for real-time CCTV face enhancement and identification with overlaid bounding boxes and labels.

• Training Configuration Analysis: Three training setups (pretrained, synthetic HQ degradation fine-tuning, real LQ fine-tuning) evaluated on the dataset to characterize its challenges on surveillance performance.

![](images/fc8b594eeb099dc9d475d75cf27346a08bd8721e21a3c4f4874624b4a5277915.jpg)  
Fig. 3. Real-time surveillance system with live CCTV stream processing, on-the-fly enhancement, and identity recognition.

![](images/346384f1e2fd49266dd069ae6965a1974b8175d68b623fdf4053dc45cc72aea3.jpg)  
Fig. 4. CVBL-CCTV-Face Dataset: top row shows high-quality (HQ) references, bottom row shows corresponding low-quality (LQ) images.

## II. RELATED WORK

Restoring faces from real-world surveillance captures is dif ficult because of complex degradations. Early super-resolution (SR) methods [2]–[5] targeted pixel fidelity (PSNR, SSIM [6]) by creating LQ–HQ pairs via bicubic downsampling and using architectures with dense/residual connections, recursion, nonlocal attention, and other advanced modules. However, these techniques struggle with the heterogeneous distortions found in authentic CCTV data. Further, methods like BSRGAN [7] and RealESRGAN [8] synthesize varied degradations, while perceptual [9] and GAN-based losses [10], [11] improve visual quality at the cost of artifacts. Subsequent work mitigated these artifacts through frequency division [12], network optimization [13], adaptive loss re-weighting [14], [15], and leveraging local statistics [16]. Diffusion-based priors have recently advanced blind SR: StableSR [17], and DiffBIR [18] adapt pretrained diffusion backbones. Recent architectures like PiSA-SR [19] disentangle pixel-level and semantic enhancement using dual-LoRA branches, while TSD-SR [20] aligns SR outputs with recognition utility via target score distillation. To improve efficiency, methods like SinSR [21] and OSEDiff [22] reduce iterative overheads while preserving output consistency. Additionally, specialized enhancements have been put forward in surveillance systems. Whereas the learning-based models such as GFPGAN [23] have proposed component-aware priors and VQFR [24] has used vector quantization to stabilize latent features, transformer models such as CodeFormer [25] and RestoreFormer++ [26] make use of codebook priors and key–value memory for dynamic reconstructions. But these models are mainly aimed at expression analysis and reidentification under relatively controlled environments, using synthetic degradations. All these studies underscore the importance of super-resolution in multimedia face applications. While there is progress in super-resolution and enhancements, the usage of synthetic degradations has limited the real-life application of these techniques. It therefore becomes necessary to develop data sets and techniques specifically designed for

![](images/48b9c8627da6b49541fdf060adac4f8f11762df3ec6ac2e15d1d69a9f37ef4be.jpg)  
Fig. 5. Examples of typical challenges in the CVBL-CCTV-Face Dataset. Row 1: Blurred Faces due to motion and poor focus. Row 2: Faces with Caps causing occlusion. Row 3: Partial Faces obstructed by doors or other objects. Row 4: Side Faces with significant pose variation. Row 5: Head-Down Faces.

surveillance environments.

## III. PROPOSED DATASET

## A. CVBL-CCTV-Face Dataset

CVBL-CCTV-Face Dataset is a collection of unpaired facial images in the real-world scenario of face enhancement and face recognition under surveillance degradations. An 8MP IP dome CCTV camera (model: EW-IP8050D) is employed to record the low quality (LQ) images in unconstrained indoor environments, which may cause various artifacts, such as pose variation, illumination variation, partial occlusion and noise from the sensor. Considering the practical surveillance environment, the high quality (HQ) images are captured independently in a controlled way. Some samples are demonstrated in Fig. 4. We provide the dataset in two different versions. Version A includes 102 identities, where each identity has one HQ image (totally 102 HQ images) and multiple LQ images (total 8090 images). The version represents the actual situation of having a limited number of HQ images. Version B includes 130 identities, where each identity has 15 to 25 HQ images (totally 2,612 HQ images) and 7,572 LQ images. The two versions share 9 identities and after deleting the duplication, we have a final count of 223 identities. For experimentation, the final set is split with identity-level separation, where we have the test set, which includes 130 identities (1,581 LQ and 528 HQ images), and the remaining identities (14,081 LQ across 223 folders and 2,186 HQ images, augmented via standard transformations) make our training set.

![](images/df6ffff046de4f471679f766a164504790465d870b0b1b907dc1bf1099262150.jpg)

Fig. 6. Proposed pipeline: (a) MTCNN face detection; (b) Super-resolution via facial enhancement models; (c) Inception-ResNet-v1 embedding extraction and L2 norm identity matching (M × 512 SR, P × 512 HR embeddings).  
![](images/eb41e8925365621d719b8cc5b2d774b449eb045c8a4b09fe706de0ca9f619107.jpg)  
Fig. 7. Illustration of various challenges encountered by MTCNN during face detection, highlighting the challenges in proposed CVBL-CCTV-Face Dataset.

## B. Challenges in the Dataset

Real-world surveillance footage presents several inherent challenges (refer Fig. 5) for facial enhancement and recognition, as reflected in the CVBL-CCTV-Face Dataset :

• Blurred Faces: The facial clarity is often degraded by motion and defocus, which makes enhancement and recognition complicated.

• Occlusions by Accessories: Caps or hat items occlude many important landmarks, which affects identification.

• Partially Visible Faces: Walls or doors obstruct many facial regions, which affects restoration.

• Pose Variations: Nonfrontal views hinder facial structures and hamper models trained mainly on frontal faces.

• Downward Head Tilts: Key features like eyes and mouth become less visible when faces are angled downwards.

These challenges necessitate the development of robust enhancement models capable of handling diverse degradations and pose variations, ensuring reliable performance under unconstrained real-world surveillance conditions.

## IV. METHODOLOGY

## A. Face Enhancement and Recognition Pipeline

We propose a modular pipeline to systematically evaluate the impact of facial image enhancement on face recognition performance on the propose CVBL-CCTV-Face Datset. As shown in Fig.6, the pipeline is organized into three core stages: face detection and alignment, super-resolution enhancement, and embedding extraction and identity matching.

![](images/069400f8fd475d9e334620a23d2a5895b7de4deb438a0962249ef342c62bba6e.jpg)  
Fig. 8. Face enhancement examples using pretrained CodeFormer [25], RestoreFormer++ [26], PiSA-SR [19], and TSD-SR [20] (left to right).

1) Face Detection and Alignment: The first stage of the pipeline focuses on detecting and geometrically aligning face regions to prepare for subsequent enhancement and recognition. We employ the Multi-task Cascaded Convolutional Neural Network (MTCNN) [1] for this purpose. MTCNN [1] operates in three cascaded stages (P-Net, R-Net, O-Net) to progressively refine bounding boxes and localize five facial landmarks for alignment. Figure 7 shows six example frames from the CVBL-CCTV-Face Dataset that reveal common failure modes of MTCNN in realistic surveillance scenes. In the top row, the left frame completely misses the face and instead highlights a non facial region, while the middle and right frames produce false positives such as hands and toes. In the bottom row, the left frame again fails to localize the face, and the middle and right continue to misidentify non facial areas. These errors arises from real world factors like low resolution, changing illumination, motion blur, occlusions, and varied poses and illustrate how MTCNN performance degrades outside controlled environments.

2) Super-Resolution and Enhancement: As can be seen from Fig. 6 (part (b)), the aligned LQ face crops are enhanced with a collection of deep learning-based enhancement models specifically trained to address some of the most common surveillance-related image degradations like blur, noise, and compression artifacts. These models comprise CodeFormer [25], RestoreFormer++ [26], PiSA-SR [19], and TSD-SR [20]. Each of the above-mentioned models addresses some particular problems associated with low-quality face enhancement. CodeFormer [25] is a codebook-augmented transformer that allows balancing visual sharpness and identity preservation by introducing an easily adjustable fidelity weight. Restore-Former++ [26] introduces a dynamic key-value memory for high-fidelity prior retrieval with multi-scale feature aggregation. PiSA-SR [19] is a dual-LoRA model built upon a pretrained Stable Diffusion backbone, which allows optimizing pixel and semantic fidelity independently. TSD-SR [20] aligns super-resolution with face recognition and relies on distillation of target identity scores, making it robust to extreme image degradations. Fig. 8 demonstrates examples of enhancements of LQ faces obtained with different pre-trained models.

3) Embedding Extraction and Recognition: As shown in part (c) of Fig. 6, super-resolved (SR) images are passed through the Inception-ResNet-v1 [27] network to extract 512- dimensional facial embeddings. This hybrid architecture combines multi-scale Inception modules with residual connections for effective feature representation. The embeddings are $\ell _ { 2 ^ { - } }$ normalized, ensuring scale-invariant comparisons based on angular similarity. Let M and P denote the number of SR and high-quality (HR) faces, respectively, forming $M \times 5 1 2$ and $P \times 5 1 2$ matrices. Identity matching is performed by computing Euclidean distances between embeddings: $d ( \mathbf { e } _ { \mathrm { S R } } , \mathbf { e } _ { \mathrm { H R } } ) =$ $\| \mathbf { e } _ { \mathrm { S R } } - \mathbf { e } _ { \mathrm { H R } } \| _ { 2 } ,$ . Each SR face is matched to the nearest HR identity, and accuracy is the proportion of correct matches.

## V. EXPERIMENTAL SETTINGS

## A. Training Configurations for Face Enhancement Models

Evaluation of the efficiency of different training approaches to face enhancement and recognition is done using experiments under three setups, each representing a certain degree of domain adaptation. Experiments are performed on the CVBL-CCTV-Face Dataset, as detailed in Section III. Evaluation procedure is divided into two parts – face enhancement and identity recognition. Face detection and face alignment are performed using MTCNN face detector, while features are extracted using the Inception-ResNet-v1 network, hybrid architecture pretrained on VGGFace2 [28]. 512-dimensional embeddings generated by the network are then used for identity matching. The performance of face recognition is assessed in terms of Top-1 identification accuracy. Experimental setup is as follows:

1) Setting 1: Pretrained Models: The enhancement models are applied to the low-quality (LQ) images in this setting without any fine-tuning. The pretrained weights are being utilized by the models to assess generalization on diverse surveillance conditions. As shown in the first column of Fig. 9, pretrained models demonstrate improvements over raw LQ inputs but they often struggle to capture domain-specific characteristics, reducing their robustness in real-world surveillance scenarios.

2) Setting 2: Fine-Tuning on Synthetically Degraded HQ Images: In this configuration, high-quality images from the CVBL-CCTV-Face Dataset are artificially degraded using techniques like Gaussian blur, noise addition, and compression artifacts, which emulate real-world surveillance scenarios. Enhancement networks are fine-tuned using these degraded images so that the network learns common degradations. As can be seen in the second column of Fig. 9, the use of artificial degradations for fine-tuning results in substantial improvements in recognition accuracy.

3) Setting 3: Fine-Tuning on Real LQ Images: In this setting, models are fine-tuned directly on real low-quality surveillance images containing complex artifacts like occlusions, pose variations, motion blur, and lighting changes. As shown in the third column of Fig. 9, direct real LQ fine-tuning did not improve over the baseline, indicating that current architectures struggle with uncontrolled real-world distortions, which provides an open challenge motivating future work.

## B. Real-Time Enhancement and Recognition System

We evaluate the real-time system on the CVBL-CCTV-Face Dataset under diverse live conditions, with enhancement applied as a preprocessing step prior to identity matching. As shown in Fig. 10, a gallery of high-resolution (HR) reference faces is created using Inception-ResNet-v1, producing 512-d embeddings. The faces are detected via MTCNN, enhanced using the proposed models, and passed through Inception-ResNet-v1 to extract embeddings. These are compared to the HR gallery using cosine similarity. The closest match is displayed with a bounding box and identity label.

![](images/9ba9be547be247c408746a65da034cd645e3244e61fe75f13b8240c75d82fa1a.jpg)  
Fig. 9. Visual comparison between LQ/HQ inputs and restored outputs under different training strategies. Each column shows outputs from CodeFormer, TSD-SR, and RestoreFormer++ models for Pretrained, Fine-tuned on synthetic degradation, and Fine-tuned on real LQ images.

![](images/e1e4417c6553c4f3a113cdd7b0e4baa7a5bce8527b3f8a93db542755656ebb00.jpg)  
Fig. 10. Real-time results: Left shows enrolled HR faces; right shows enhanced live detections with matched identities.

## VI. RESULTS

## A. Face Enhancement and Recognition Performance

As shown in Fig. 11, the first five columns depict LQ inputs and their HQ references, with red boxes marking incorrect identity matches. The next five columns show the corresponding TSD-SR enhanced outputs and their corrected matches. In column 1, the LQ image was misclassified as ID 32 $( L _ { 2 } ~ = ~ 0 . 9 3 3 2 )$ , but after enhancement, it correctly matches ID $1 0 8 \ : \ : ( L _ { 2 } = \ : 0 . 7 5 6 2 )$ as seen in Column 6. Column 7 retains ID 101, with distance improving from 0.7603 to 0.6988. Column 8 also keeps ID 115, improving from 0.7200 to 0.4700. Column 9 corrects a mismatch as seen in Column 4 from ID 71 (0.8557) to ID 81 (0.8088). Column 10, despite occlusion, recovers ID 100, improving from ID 9 (1.023) to 0.980. These results confirm the enhancement models effectiveness in reducing embedding gaps and correcting identity mismatches under real-world CCTV conditions. Fig. 12 further compares CodeFormer, Restore Former++, PiSA-SR, and TSD-SR (all fine-tuned on synthetic degradations)

![](images/142976333740b7e3f27089fa257a5a57b717645d016c45cda29e19b7a7cc6d09.jpg)  
Fig. 11. Facial enhancement and identity recognition results from the CVBL-CCTV-Face Dataset. Incorrect identity matches are highlighted with red boxes and their corrected matches after enhancement are shown with green boxes.

![](images/3362a4f174616f4ab00077e94aac8d76b461fd58f2ce3824c06befe1e0b10843.jpg)  
Fig. 12. Qualitative face enhancement results on four identities from the CVBL-Face dataset. Each row corresponds to a different individual. From left to right: (1) original low-quality (LQ) surveillance face crop, followed by enhanced outputs from (2) CodeFormer [25], (3) RestoreFormer++ [26], (4) PiSA-SR [19], and (5) TSD-SR [20].

across four identities. For identity 1, the $L _ { 2 }$ distances are 0.890 (CodeFormer), 0.730 (PiSA-SR), 0.930 (RestoreFormer++), and 0.773 (TSD-SR); for identity 2: 0.680, 0.620, 0.617, and 0.604; identity 3: 0.841, 0.824, 0.766, and 0.795; and identity 4: 0.748, 0.707, 0.668, and 0.679 respectively. Overall, finetuning consistently improves identity similarity, with TSD-SR and RestoreFormer++ preserving identities most effectively.

## B. Real-Time Face Enhancement and Recognition Results

We evaluated on an NVIDIA GeForce RTX 3090 GPU at approximately 25 FPS under diverse CCTV conditions, including motion blur, occlusion, illumination changes, and non-frontal poses. Fig. 10 illustrates the results, where the left panel shows high-resolution (HR) enrolled faces and the right panel shows live detections. Detected low-quality (LQ) faces are enhanced on-the-fly, and embeddings are extracted and matched to the HR gallery using dot product similarity. Predicted identities are annotated with similarity scores: Person3 (0.65), Person4 (0.66), Person5 (0.64), and Person2 (0.41). The incorrect match of Person2 highlights sensitivity to ambiguous low-quality inputs, yet overall results validate enhancement as a necessary preprocessing step for real-time recognition under surveillance degradations.

TABLE I  
BENCHMARK COMPARISON OF REPRESENTATIVE ENHANCEMENT METHODS USING TOP-1 RECOGNITION ACCURACY (%)
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=3>Training Setting</td></tr><tr><td rowspan=1 colspan=1>Pre-Trained</td><td rowspan=1 colspan=1>FT (Real LQ)</td><td rowspan=1 colspan=1>FT (Synthetic)</td></tr><tr><td rowspan=1 colspan=1>PiSA-SR [19]</td><td rowspan=1 colspan=1>64.77</td><td rowspan=1 colspan=1>3.23</td><td rowspan=1 colspan=1>67.81</td></tr><tr><td rowspan=1 colspan=1>TSD-SR [20]</td><td rowspan=1 colspan=1>64.45</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>70.08</td></tr><tr><td rowspan=1 colspan=1>CodeFormer [25]</td><td rowspan=1 colspan=1>57.69</td><td rowspan=1 colspan=1>40.20</td><td rowspan=1 colspan=1>59.20</td></tr><tr><td rowspan=1 colspan=1>RestoreFormer++ [26]</td><td rowspan=1 colspan=1>59.46</td><td rowspan=1 colspan=1>62.75</td><td rowspan=1 colspan=1>67.49</td></tr></table>

<sup>a</sup>Baseline (No enhancement): 62.60%.

## C. Evaluation Under Different Training Configurations

We evaluated three enhancement strategies on CVBL-CCTV-Face Dataset with realistic degradations from surveillance, shown in Table I. The pipeline contains face detection (MTCNN), enhancement (Transformer and Diffusion-based models) and feature extraction (Inception-ResNet-v1 trained on VGGFace2). We evaluated the performance by means of $L _ { 2 }$ distance and top-1 identification accuracy. Baseline (without enhancement) yields 62.6% identification accuracy.

1) Evaluation with Pre-trained Models: The identification rate for PiSA-SR and TSD-SR is 64.77% and 64.45%, respectively, using publicly pre-trained models without any finetuning, which is slightly higher than baseline and demonstrates poor generalization on real CCTV degradations.

2) Fine Tuning on Real Low Quality Images: Different fine tuning approaches yield different identification rates, with RestoreFormer++ getting 62.75%, CodeFormer and TSD-SR obtaining 40.20% and 33.33%, respectively. This difference can be explained by the fact that real LQ images are captured independently, are spatially unaligned and have multiple uncontrolled degradations, unlike synthetic degraded HQ images.

3) Fine-Tuning on Synthetically Degraded HQ Images: Synthetic fine tuning proved to be the most successful: TSD-SR obtains 70.08% accuracy, PiSA-SR - 67.81% and Restore-Former++ - 67.49%.

## VII. CONCLUSION

The CVBL-CCTV-Face Dataset can be considered as an essential benchmark that can be used to improve face enhancement and recognition in real-world surveillance situations. The pipeline created using this dataset, consisting of face detection, deep learning-based image enhancement, and identity recognition, performs well in different conditions. Of all the training methods tested, those involving synthetic degradations of highresolution images resulted in the best recognition accuracy, which proves the validity of degradation modeling. Yet, the problem is that the actual scenario in real-world surveillance is associated with low-quality real-life degradations, which are much more complicated and unpredictable. Models cannot cope with this situation at the moment, yet it is a promising way forward. Our real-time system proves the feasibility of the suggested pipeline, and these findings can become the foundation for developing face recognition systems.

## VIII. FUTURE WORK

We will extend the identification benchmark by adding verification metrics such as FAR/FRR, inference latency, and computational cost across different hardware platforms in our future work.

## ACKNOWLEDGMENT

This work was supported by the I-DAPT Hub Foundation, IIT (BHU) Varanasi. We acknowledge the fellowship and research support from Indian Institute of Information Technology Allahabad. We sincerely thank Mr. Hritu Raj, Mr. Vikas Rajput, Mr. Samar Borkar, Mr. Anubhav, Mr. Shirshak Mandal, Mr. Govind Anand, and Mr. Prathmesh Pampattiwar for their support in creation of the dataset used in this study.

## REFERENCES

[1] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE signal processing letters, vol. 23, no. 10, pp. 1499–1503, 2016.

[2] C. Dong, C. C. Loy, K. He, and X. Tang, “Learning a deep convolutional network for image super-resolution,” in Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13, pp. 184–199, Springer, 2014.

[3] M. Haris, G. Shakhnarovich, and N. Ukita, “Deep back-projection networks for super-resolution,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1664–1673, 2018.

[4] J. Kim, J. K. Lee, and K. M. Lee, “Deeply-recursive convolutional network for image super-resolution,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1637–1645, 2016.

[5] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “Swinir: Image restoration using swin transformer,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 1833– 1844, 2021.

[6] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[7] K. Zhang, J. Liang, L. Van Gool, and R. Timofte, “Designing a practical degradation model for deep blind image super-resolution,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 4791–4800, 2021.

[8] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-esrgan: Training realworld blind super-resolution with pure synthetic data,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 1905– 1914, 2021.

[9] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11- 14, 2016, Proceedings, Part II 14, pp. 694–711, Springer, 2016.

[10] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta,´ A. Aitken, A. Tejani, J. Totz, Z. Wang, et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4681–4690, 2017.

[11] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and C. Change Loy, “Esrgan: Enhanced super-resolution generative adversarial networks,” in Proceedings of the European conference on computer vision (ECCV) workshops, pp. 0–0, 2018.

[12] W. Li, K. Zhou, L. Qi, L. Lu, and J. Lu, “Best-buddy gans for highly detailed image super-resolution,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 36, pp. 1412–1420, 2022.

[13] Y. Zhang, B. Ji, J. Hao, and A. Yao, “Perception-distortion balanced admm optimization for single-image super-resolution,” in European Conference on Computer Vision, pp. 108–125, Springer, 2022.

[14] L. Sun, J. Liang, S. Liu, H. Yong, and L. Zhang, “Perception-distortion balanced super-resolution: A multi-objective optimization perspective,” IEEE Transactions on Image Processing, 2024.

[15] X. Wang, K. Yu, C. Dong, X. Tang, and C. C. Loy, “Deep network interpolation for continuous imagery effect transition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1692–1701, 2019.

[16] J. Liang, H. Zeng, and L. Zhang, “Details or artifacts: A locally discriminative learning approach to realistic image super-resolution,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5657–5666, 2022.

[17] J. Wang, Z. Yue, S. Zhou, K. C. Chan, and C. C. Loy, “Exploiting diffusion prior for real-world image super-resolution,” International Journal of Computer Vision, vol. 132, no. 12, pp. 5929–5949, 2024.

[18] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang, and C. Dong, “Diffbir: Toward blind image restoration with generative diffusion prior,” in European Conference on Computer Vision, pp. 430– 448, Springer, 2024.

[19] L. Sun, R. Wu, Z. Ma, S. Liu, Q. Yi, and L. Zhang, “Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach,” arXiv preprint arXiv:2412.03017, 2024.

[20] L. Dong, Q. Fan, Y. Guo, Z. Wang, Q. Zhang, J. Chen, Y. Luo, and C. Zou, “Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution,” arXiv preprint arXiv:2411.18263, 2024.

[21] Y. Wang, W. Yang, X. Chen, Y. Wang, L. Guo, L.-P. Chau, Z. Liu, Y. Qiao, A. C. Kot, and B. Wen, “Sinsr: diffusion-based image superresolution in a single step,” in Proceedings of the IEEE/CVF conference on computer vision and patZhangtern recognition, pp. 25796–25805, 2024.

[22] R. Wu, L. Sun, Z. Ma, and L. Zhang, “One-step effective diffusion network for real-world image super-resolution,” Advances in Neural Information Processing Systems, vol. 37, pp. 92529–92553, 2024.

[23] X. Wang, Y. Li, H. Zhang, and Y. Shan, “Towards real-world blind face restoration with generative facial prior,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 9168–9178, 2021.

[24] Y. Gu, X. Wang, L. Xie, C. Dong, G. Li, Y. Shan, and M.-M. Cheng, “Vqfr: Blind face restoration with vector-quantized dictionary and parallel decoder,” in European Conference on Computer Vision, pp. 126–143, Springer, 2022.

[25] S. Zhou, K. Chan, C. Li, and C. C. Loy, “Towards robust blind face restoration with codebook lookup transformer,” Advances in Neural Information Processing Systems, vol. 35, pp. 30599–30611, 2022.

[26] Z. Wang, J. Zhang, T. Chen, W. Wang, and P. Luo, “Restoreformer++: Towards real-world blind face restoration from undegraded key-value pairs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 15462–15476, 2023.

[27] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inception-v4, inception-resnet and the impact of residual connections on learning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 31, 2017.

[28] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, “Vggface2: A dataset for recognising faces across pose and age,” in 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pp. 67–74, IEEE, 2018.