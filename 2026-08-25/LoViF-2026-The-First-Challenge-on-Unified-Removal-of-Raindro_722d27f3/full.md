# LoViF 2026 The First Challenge on Unified Removal of Raindrops and Reflections: Methods and Results

Zewei He<sup>†,B</sup> , Xi Tong<sup>†</sup> , Yu Chen<sup>†</sup> , Xingyu Liu<sup>†</sup> , Xin Li<sup>†</sup> , Zepeng Wang, Jiagao Hu, Fuhao Li, Yuxuan Chen, Fei Wang, Daiguo Zhou, Minmin Yi, Chuanrui Zhang, Liwen Zhang, Yeongjin Jeong, Hyunjin Cho, Jiwon Lee, Minsang Kim, Jae Woong Soh, Jin-Hui Jiang, Rong-Lin Jian, Chih-Chung Hsu, Youngjin Oh, Junhyeong Kwon, Junyoung Park, Jae Hyun Park, Sung Ju Lee, Nam Ik Cho, Vishwajeet Shukla, Himanshu Baurai, Zhiqi Zhang, Kui Jiang, Zhaocheng Yu, Runzhe Li, Dawei Fan, Hao Li, Zhanshuo Zhang, Fan Ji, Jiangmeng Li, Xiongxin Tang, Fanjiang Xu, Shangquan Sun, Anh-Kiet Duong, Petra Gomez-Krämer, Jean-Michel Carozza, Ruibo Zhang, Dexiang Hong, Xinyan Liu, Shengeng Tang, Weidong Chen, Tzu-Hsuan Weng, and Min-Te Sun

zeweihe@zju.edu.cn

Data page: https://github.com/hezw2016/RDRF-dataset

Abstract. This workshop paper comprehensively reviews the First Challenge on Unified Removal of Raindrops and Reflections. The challenge aims to address a frequently encountered practical problem in the field of autonomous driving, i.e., raindrop-reflection composite degradation on rainy days. This competition attracted 149 registered participants and received 12 valid final submissions with corresponding fact sheets, significantly contributing to the progress of unified removal of raindrops and reflections. All the methods are developed and evaluated on our real-shot RainDrop and ReFlection (RDRF) dataset. A detailed analysis of the submitted methods and corresponding results is provided in this report, which highlights efective approaches and provides interesting insights for future research.

Keywords: Raindrop removal · Reflection removal · Challenge

## 1 Introduction

On rainy days, raindrops and reflections frequently co-occur in autonomous driving scenarios, posing challenges for onboard visual recognition systems or vehicle cameras during recording. Adherent raindrops and the reflections from camera side significantly reduce the visibility of captured images [41], and may lead to severe driving safety hazards. Therefore, Unified Removal of Raindrops and Reflections $\mathrm { ( U R ^ { 3 } ) }$ is a practical problem that requires urgent solution.

Unified removal of raindrops and reflections (UR<sup>3</sup>) aims to develop a unified model that can handle the raindrop-reflection composite degradation. Previously, researchers treated raindrop removal and reflection removal as two separate tasks [17, 18, 28, 31, 33, 47]. Though these methods can achieve relatively good performance in removing the target type of degradation (i.e., raindrop or reflection) from a single image, they often fail to remove both types at the same time.

To facilitate the research in this direction, we organized the First Challenge on Unified Removal of Raindrops and Reflections in conjunction with the Second Low-level Vision Frontiers (LoViF) Workshop at ECCV 2026. This challenge aims to find a practical solution capable of simultaneously eliminating this raindrop-reflection composite degradation, thereby enhancing the clarity of captured images. We hope this endeavor can provide useful support for applications such as autonomous driving, photography, and video surveillance [9, 10].

This challenge is held as part of the LoViF Workshop, which hosts a series of related challenges, including the Second Real-World All-in-One Image Restoration Challenge [5], the Unified Removal of Raindrops and Reflections Challenge (ours), the Ultra-Low Bitrate Image Compression Challenge [23], the AIGC Image Compression Challenge [21], the Day and Night Raindrop Removal for Dual-Focused Images Challenge [40], and the ActPhysCause Challenge [43].

## 2 UR<sup>3</sup> Challenge

This UR<sup>3</sup> challenge is the first competition organized to advance the development of unified removal of raindrops and reflections. The details of the whole challenge are as follows:

## 2.1 RDRF dataset

The dataset used in this challenge is the RainDrop and ReFlection (RDRF) dataset [24], which is captured with a dedicated hardware system. In total, the RDRF dataset consists of 307 unique scenes. During this challenge, it is divided into a training set (216 scenes with 9003 image pairs) and a validation set (91 scenes with 277 image pairs). The above images are shared with the research community under a CC BY-NC-SA 4.0 copyright license on https://github. com/hezw2016/RDRF-dataset.

In addition, we also captured 100 image pairs from 55 scenes as the testing set for this challenge. To ensure the fairness of this competition, only the degraded images were publicly released.

## 2.2 Evaluation protocol

For all the submitted methods, we followed the scoring function from [22] to determine the ranking. It involves three widely-used metrics, i.e., PSNR, SSIM

Table 1: Quantitative results on the $\mathrm { U R ^ { 3 } }$ Challenge test set. The bold and underline indicate the best and second-best results.
<table><tr><td>Rank</td><td>Team</td><td>Leader</td><td>PSNR↑</td><td></td><td></td><td>SSIM↑ LPIPS↓|Extra Data|</td><td>Score↑</td></tr><tr><td>1</td><td>Xedit Master</td><td>Jiagao Hu</td><td>32.3993</td><td>0.9645</td><td>0.0443</td><td>√</td><td>41.8226</td></tr><tr><td>2</td><td>tysl</td><td>Minmin Yi</td><td>32.3006</td><td>0.9648</td><td>0.0416</td><td>√</td><td>41.7405</td></tr><tr><td>3</td><td>GIST-IVL</td><td>Yeongjin Jeong</td><td>31.6556</td><td>0.9680</td><td>0.0388</td><td>√</td><td>41.1416</td></tr><tr><td>4</td><td>ACVLAB</td><td>Jin-Hui Jiang</td><td>31.0060</td><td>0.9603</td><td>0.0432</td><td>x</td><td>40.3933</td></tr><tr><td>5</td><td>SNU-ISPL</td><td>Youngjin Oh</td><td>30.7389</td><td>0.9572</td><td>0.0526</td><td>√</td><td>40.0474</td></tr><tr><td>6</td><td></td><td>the_last_token Vishwajeet Shukla</td><td>30.3074</td><td>0.9580</td><td>0.0559</td><td>√</td><td>39.6075</td></tr><tr><td>7</td><td>AIIA-Lab</td><td>Zhiqi Zhang</td><td>30.3203</td><td>0.9568</td><td>0.0563</td><td>x</td><td>39.6067</td></tr><tr><td>8</td><td>ISCAS Optics</td><td>Dawei Fan</td><td>30.1058</td><td>0.9575</td><td>0.0611</td><td>x</td><td>39.3752</td></tr><tr><td>9</td><td>sunsean</td><td>Shangquan Sun</td><td>29.7094</td><td>0.9562</td><td>0.0652</td><td>√</td><td>38.9455</td></tr><tr><td>10</td><td>ULR</td><td>Anh-Kiet Duong</td><td>29.0555</td><td>0.9260</td><td>0.1107</td><td>x</td><td>37.7620</td></tr><tr><td>11</td><td>ustc_pi_lab</td><td>Ruibo Zhang</td><td>28.2417</td><td>0.9457</td><td>0.0851</td><td>x</td><td>37.2730</td></tr><tr><td>12</td><td>daniel0902</td><td>Tzu-Hsuan Weng</td><td>27.8509</td><td>0.9440</td><td>0.0800</td><td>x</td><td>36.8909</td></tr></table>

Table 2: Quantitative results on the $\mathrm { U R ^ { 3 } }$ Challenge test set. The results are obtained with submitted inference codes under strict single-frame setting. The bold and underline indicate the best and second-best results.
<table><tr><td>Rank</td><td>Team</td><td>Leader</td><td>PSNR↑</td><td></td><td></td><td>SSIM↑ LPIPS↓|Extra Data|</td><td>Score↑</td></tr><tr><td>1</td><td>Xedit Master</td><td>Jiagao Hu</td><td>|32.3993</td><td>0.9645</td><td>0.0443</td><td>√</td><td>41.8226</td></tr><tr><td>2</td><td>tysl</td><td>Minmin Yi</td><td>32.3006</td><td>0.9648</td><td>0.0416</td><td>√</td><td>41.7405</td></tr><tr><td>3</td><td>ACVLAB</td><td>Jin-Hui Jiang</td><td>31.0060</td><td>0.9603</td><td>0.0432</td><td>x</td><td>40.3933</td></tr><tr><td>4</td><td>SNU-ISPL</td><td>Youngjin Oh</td><td>30.7389</td><td>0.9572</td><td>0.0526</td><td>√</td><td>40.0474</td></tr><tr><td>5</td><td>GIST-IVL†</td><td>Yeongjin Jeong</td><td>30.5155</td><td>0.9573</td><td>0.0516</td><td>√</td><td>39.8308</td></tr><tr><td>6</td><td></td><td>the_last_token Vishwajeet Shukla</td><td>30.3074</td><td>0.9580</td><td>0.0559</td><td>√</td><td>39.6075</td></tr><tr><td>7</td><td>ISCAS Optics</td><td>Dawei Fan</td><td>30.1058</td><td>0.9575</td><td>0.0611</td><td>x</td><td>39.3752</td></tr><tr><td>8</td><td>sunsean</td><td>Shangquan Sun</td><td>29.7094</td><td>0.9562</td><td>0.0652</td><td>√</td><td>38.9455</td></tr><tr><td>9</td><td>AIIA-Lab†</td><td>Zhiqi Zhang</td><td>29.7247</td><td>0.9544</td><td>0.0657</td><td>x</td><td>38.9404</td></tr><tr><td>10</td><td>ustc_pi_lab</td><td>Ruibo Zhang</td><td>28.2417</td><td>0.9457</td><td>0.0851</td><td>x</td><td>37.2730</td></tr><tr><td>11</td><td>daniel0902</td><td>Tzu-Hsuan Weng</td><td>27.8509</td><td>0.9440</td><td>0.0800</td><td>x</td><td>36.8909</td></tr><tr><td>12</td><td>ULR†</td><td>Anh-Kiet Duong</td><td>28.2267</td><td>0.9215</td><td>0.1176</td><td>x</td><td>36.8534</td></tr></table>

and LPIPS. The final score is calculated by re-weighting them as:

$$
\mathrm { S c o r e = P S N R ( Y ) + 1 0 \times S S I M ( Y ) - 5 \times L P I P S } .\tag{1}
$$

For PSNR and SSIM, the evaluation is conducted on the Y channel of the YCbCr color space. For LPIPS, the image pixel values are normalized to the range of [−1, 1] before computing the perceptual distance (based on AlexNet) between the restored and the ground-truth images.

## 2.3 Challenge phases

There are two phases in this challenge, i.e., the development and testing phases. The details are as follows:

Development phase: In this phase, we released the training set in our RDRF dataset. The degraded images in the validation set were also released. Each participant can upload their restored images to the challenge platform to obtain the corresponding score on the validation set. During this phase, we received 1176 submissions from 56 participants.

Testing phase: In this phase, we released the ground-truth images in the validation set. The degraded images in the testing set were also released. Each participant can upload their restored images to the challenge platform to obtain the corresponding score on the testing set. During this phase, we received 406 submissions from 38 participants.

## 3 Challenge Results

We have summarized the challenge results in Table 1. Team Xedit Master achieved the best overall performance in the challenge, ranking top with a final score of 41.8226. This team also achieved the highest PSNR value (32.3993). Team tysl ranked second with a final score of 41.7405. This team secured the second position across three indicators (i.e., PSNR, SSIM and LPIPS). Team GIST-IVL ranked third with a final score of 41.1416. This team also achieved the highest SSIM and LPIPS values (0.9680 and 0.0388).

Due to the acquisition protocol of the RDRF dataset, a single ground-truth image may correspond to multiple degraded images. Consequently, multi-frame fusion can enhance restoration performance. We found that some submitted methods utilized multiple images of the same scene to produce the final output, which was somewhat unfair compared with single-frame methods. This challenge was originally designed as a single-image restoration task. Therefore, the inference should rely on only one input image. Using multiple images or frames is inconsistent with the task setting and does not match the intended practical scenario.

We re-tested the results with submitted inference codes under strict singleframe setting. Table 2 shows the results. Team ACVLAB ranked third by removing multi-frame fusion.

## 4 Teams and Methods

## 4.1 Xedit Master

This team proposes Progressive Cascaded Removal and Refinement, a two-stage framework built upon an RDNet remover and a NAFNet refiner, as illustrated in Fig. 1. In the first stage, the remover jointly suppresses raindrops and reflections. In the second stage, the refiner corrects the remaining artifacts and recovers fine image details. By separating degradation removal from fidelity-oriented refinement, the method allows the two networks to be optimized for complementary objectives.

![](images/78975954696581bf52a5480b07f0883f421f9a782ac672d914eead8ea4a3ccb6.jpg)  
Fig. 1: The pipeline of the method proposed by Team Xedit Master.

The first-stage remover adopts RDNet [47] with a larger FocalNet-XL backbone [38]. Building upon the trained remover, the second stage adopts the SIDD configuration of NAFNet [4], with a width of 64 and initialization from SIDDpretrained weights. Specifically, the team freezes RDNet-XL and applies it to the entire training set to obtain intermediate restorations. These results are used as the inputs to NAFNet, while the corresponding clean images serve as supervision. The refiner is trained for 200 epochs using a PSNR-based loss and a cosine-annealing learning-rate schedule to correct residual artifacts remaining after the first-stage restoration.

Training Details. The models are trained sequentially. The team first follows the three-resolution curriculum to optimize RDNet-XL and then trains NAFNet using the frozen RDNet-XL outputs. In addition to the oficial challenge training data, they use the DeRaindrop [28] and RainDS [30] datasets for raindrop removal, together with OpenRR-5K [2] and RR4K for reflection removal.

Testing Details. During testing, each degraded image is first processed by RDNet-XL for unified raindrop and reflection removal. The resulting image is then passed once through NAFNet to obtain the final restoration.

Implementation Details. The method is implemented on eight NVIDIA A100 GPUs using bf16 mixed precision. The remover adopts the FocalNet-XL version of RDNet, while the refiner adopts the NAFNet architecture configured with a base channel width of 64 following its standard SIDD denoising setting.

## 4.2 tysl

This team proposes a difusion-guided two-stage framework that combines a generative clean-image prior with a supervised fidelity-oriented refiner, as illustrated in Fig. 2. In the first stage, they adapt the pretrained GenSIRR model [20] to the $\mathrm { U R ^ { 3 } }$ task using LoRA [16]. Specifically, a task-oriented prompt guides the difusion model to remove reflections and raindrops while completing the content occluded by raindrops. Since the generated result may contain local geometric or color inconsistencies, the method uses it as a restoration prior instead of directly treating it as the final output.

![](images/cfae0df3d7b0fc135d49986c1b03a2a8e5099003dc893b211899c1b6d268b1a3.jpg)  
Fig. 2: The pipeline of the method proposed by Team tysl.

![](images/2415314b689da56d390bc9252e3f87b3193132a6fd4787c0da9af8f3a3e78ddb.jpg)  
Fig. 3: The overall framework of Team GIST-IVL.

In the second stage, a supervised refiner takes the original degraded image and the difusion prior as inputs. Two shallow convolutional branches first extract their features, which are aligned and exchanged through bidirectional deformable attention [48] to correct local inconsistencies in the difusion prior. A ConvNeXt-based estimator [25] further predicts a low-resolution spatial prompt map to adaptively modulate the refinement features. The prompted features are concatenated and fused by HiT-SR [45], after which NAFNet predicts a residual correction that is added to the difusion prior.

Training Details. Training comprises two stages. First, the pretrained Gen-SIRR model is adapted with LoRA for 20,000 iterations using a task-specific restoration prompt, a learning rate of $1 \times 1 0 ^ { - 4 }$ . The refiner is then trained for 60,768 iterations, taking the degraded image and the GenSIRR-generated restoration prior as inputs and the corresponding clean image as the target.

![](images/f8d618238161e4a8d983357377478000b50f0c5004777748d5f73f6142783096.jpg)  
Fig. 4: The architecture of Stage III in MAFNet-UR<sup>3</sup>.

Testing Details. During testing, the adapted GenSIRR model uses 24 denoising steps to generate a restoration prior for each test image. The generated prior and the original degraded image are then jointly fed into the trained refiner to produce the restored output.

Implementation Details. The second-stage refiner is trained on four NVIDIA A100 GPUs, each with 80 GB of memory. The two stages are trained independently. During the first-stage adaptation, only the LoRA parameters are updated, while the pretrained GenSIRR weights remain frozen.

Ensembles and Fusion Strategies. For each test image, the adapted Gen-SIRR model is sampled using 20 fixed random seeds, ranging from 1 to 20. Each generated restoration prior is independently processed by the refiner, and the resulting 20 restored outputs are averaged to obtain the final prediction.

## 4.3 GIST-IVL

Team GIST-IVL proposes MAFNet-UR<sup>3</sup>, a three-stage framework that exploits multiple observations of the same scene for unified raindrop and reflection removal, as illustrated in Fig. 3. In Stage I, a pretrained WeatherRemover model [29] is fine-tuned with raindrop-free targets that retain reflections, producing an intermediate result for subsequent reflection removal. Building on this output, Stage II adapts FUMO [37] to the UR<sup>3</sup> task. An intensity prior is generated by Qwen3-VL [1], while a high-frequency prior is extracted through wavelet decomposition. Conditioned on these complementary cues, a Stable Difusion 2.1-based model [32] with a ControlNet-style branch [44] performs one-step restoration. The preliminary result is then refined by NAFNet using the two priors and the Stage I output as guidance. Stage III, illustrated in Fig. 4, integrates the input with the preceding stage outputs. Raindrop and reflection masks are estimated from their luminance diferences, after which a shared NAFBlock encoder extracts multi-scale features and gated spatial fusion modules assign mask-aware weights. The fused features are further processed by Fast Fourier Convolution blocks [11] and reconstructed by an anchor-based residual decoder, suppressing corrupted regions while preserving reliable structures from cleaner observations.

![](images/b5fd632bf5b4ea4bfe44db0f03c888d2b1ec664e8c0de2a0cd9aa5b1d40808ff.jpg)  
Fig. 5: The overall framework of Team ACVLAB.

Training Details. Training follows the three-stage pipeline. Stage I is optimized for 50k iterations using Adam and a Charbonnier loss. In Stage II, the difusion model and refiner are trained for 200k and 25k iterations, respectively, using the oficial $\mathrm { U R ^ { 3 } }$ training data and Stage I outputs. Stage III is trained for 100k iterations on 256 × 256 patches.

Testing Details. At inference, Stage I first suppresses raindrops while preserving reflection cues for subsequent processing. Stage II then generates the required priors and performs restoration at $1 0 8 0 \times 7 2 0$ resolution using bf16 precision. Finally, Stage III jointly processes the grouped observations and produces the final result by predicting a residual correction over a weighted anchor.

Implementation Details. Stages I, II, and III contain 24.32M, 1.873B, and 38.14M parameters, respectively. Measured on an NVIDIA GeForce RTX 5090, the corresponding runtimes are 1.007, 4.843, and 0.764 seconds per image.

Ensembles and Fusion Strategies. Stage II employs an 8-way D4 self-ensemble, and Stage III further combines an internal 8-way D4 ensemble with learned maskaware fusion across multiple observations of the same scene.

Note. MAFNet-UR<sup>3</sup> uses multiple observations of the same scene during inference, which difers from the standard single-image setting adopted by the challenge. For transparency and fair comparison, its results under the multiobservation and single-image settings are reported separately in Tables 1 and 2.

## 4.4 ACVLAB

This team proposes RDNet-SGCR, a single feed-forward restoration network illustrated in Fig. 5. The method builds upon RDNet by introducing Semanticand-Geometry-Guided Cascaded Refinement (SGCR). It is motivated by the complementary characteristics of the two degradations: raindrops typically appear as sparse, localized, high-frequency occlusions, whereas reflections are generally low-frequency, semi-transparent layer-mixing artifacts. The backbone uses frozen FocalNet-L and DINOv2 [27] encoders to extract appearance and semantic features, which are fused and modulated by a content-adaptive prompt. A frozen Depth-Anything-V2 model [39] provides depth and surface-normal cues, injected through a zero-initialized geometry adapter to preserve the RDNet initialization. Four reversible sub-networks progressively estimate the restoration residual, with the last three outputs supervised as refinement stages.

![](images/1dab072d913d16bf75b6e152ad9a6113f078fb28693e7077b1893e9c9acf2ed7.jpg)  
Fig. 6: The overall framework of Team SNU-ISPL.

Training Details. Training proceeds in three steps. First, RDNet is initialized from its oficial reflection-removal checkpoint, equipped with a frozen DINOv2 semantic branch, and fine-tuned on OpenRR-5K [2] for 20 epochs. Second, the resulting backbone is adapted to the RDRF training set using random 640 × 640 crops, Adam with a learning rate of $1 \times 1 0 ^ { - 4 }$ and zero weight decay, a batch size of 1, and mixed precision. Third, the full SGCR model is fine-tuned on 512 × 512 crops with online depth and surface-normal guidance. Its three refinement stages are jointly trained with stage-wise supervision, contraction loss, and finalstage LPIPS loss. Ten percent of the competition data is reserved for validation. EMA, gradient clipping, rain-safe augmentation, and warm-up evaluation are also adopted.

Testing Details. During testing, depth and surface normals are first estimated from the input and injected through the learned geometry adapter. The resulting features are then processed by the reversible restoration cascade. For high-resolution images, the team optionally adopts overlapping 512 × 512 tiled inference with 25% overlap and triangular-window blending.

Implementation Details. The complete pipeline contains approximately 428M parameters, including 267.2M trainable parameters and 160.8M parameters in the frozen encoders. On an NVIDIA A100 40GB GPU, a single forward pass takes 206ms and uses 2.28GiB of peak memory.

Ensembles and Fusion Strategies. At inference, a four-way rain-safe testtime augmentation (TTA) comprising the identity, 180<sup>◦</sup> rotation, horizontal flip, and vertical flip is applied, while 90<sup>◦</sup> and 270<sup>◦</sup> rotations are excluded to avoid unrealistic horizontal rain streaks. For each transformed input, the depth map and surface normals are recomputed to maintain geometric consistency.

## 4.5 SNU-ISPL

This team adopts the hierarchical Transformer architecture of DINOLight [26] and introduces DINOClear, a data-centric adaptation for unified raindrop and reflection removal, as illustrated in Fig. 6. Rather than increasing architectural complexity or model capacity, the method improves generalization through a sequential three-phase training curriculum. It further leverages hierarchical selfsupervised features from DINOv2 to provide complementary semantic and finegrained structural priors for distinguishing genuine scene content from raindrop and reflection degradation artifacts. The method employs an Adaptive Feature Fusion Module to combine shallow geometric and structural features with deeper semantic representations from a distilled DINOv2 ViT-B/14 encoder. The fused features are injected into the restoration network through SFDINO blocks, whose Auxiliary Cross-Attention sequentially operates in the spatial and frequency domains to supplement restoration features with self-supervised visual priors.

Training Details. Training follows a three-phase curriculum: (1) 50k iterations on synthetic reflection pairs generated online from 13,700 source pairs, (2) 50k iterations on $^ { 8 , 1 3 8 }$ real reflection pairs collected from OpenRR-5K [2], Zhang et al. [46], and $\mathrm { S I R ^ { 2 } + }$ [36], and (3) 150k iterations on RDRF for target-domain adaptation.

Testing Details. At inference, high-resolution images are processed using slidingwindow inference with 448 × 448 patches and an adjustable overlap ratio.

Implementation Details. The restoration network contains 9.51M trainable parameters. For a 448×448 input, it requires 193.2 GFLOPs excluding the frozen DINOv2 encoder and 369.08 GFLOPs including it.

Ensembles and Fusion Strategies. The final submission employs an 8-way D4 ensemble, where predictions from transformed inputs are inverse-transformed and averaged. Stochastic weight averaging is also applied after training.

## 4.6 the\_last\_token

This team introduces an ensemble-based restoration framework that combines eight Restormer-family predictions through frequency-split fusion, as illustrated in Fig. 7. They compare several restoration architectures, including Restormer [42], X-Restormer [7], DRSformer [6], InstructIR [12], and MambaIR [15], together with the difusion-prior DAI model [17]. Based on scene-disjoint validation, Restormer and X-Restormer are selected as the primary backbones. The models are optimized using a diferentiable surrogate of the oficial score, combining negative luma-domain PSNR and SSIM with RGB LPIPS. The objective further includes an FFT-domain reconstruction loss, a degradation-aware weighted L<sub>1</sub> loss, and a Charbonnier loss.

Training Details. Restormer and X-Restormer are initialized from released deraining checkpoints and fine-tuned using AdamW with linear warm-up, cosine decay, gradient clipping, bf16 autocast, and EMA-based model selection. The training set contains 9,003 challenge pairs and 5,161 additional pairs, including 2,975 synthetic joint-degradation pairs generated from Cityscapes [13], 1,018 reflection pairs, and 1,168 raindrop pairs. Online synthetic degradation is further applied to 20% of the crops.

Testing Details. Each model performs overlap-add tiled inference using 256 × 256 windows with a 32-pixel overlap, reflection padding, and Hann-window blending. The reconstructed predictions are spatially aligned before fusion.

![](images/41319308f9f8e579f7476d4c433863bdb733b8c7de9dbfd74a91f9a319de4077.jpg)  
Fig. 7: The pipeline of the method proposed by Team the\_last\_token.

![](images/2dfb9edaee3fcc6b4895227adec932795ee9c5d326f595565be65c2aef838bff.jpg)  
Fig. 8: The pipeline of the method proposed by Team AIIA-Lab.

Implementation Details. On a single NVIDIA H200 GPU, the per-image runtimes of X-Restormer and Restormer are 1.16 and 0.67 seconds without TTA, and 3.60 and 2.35 seconds with four-way TTA, respectively.

Ensembles and Fusion Strategies. The final ensemble contains eight predictions generated from a Restormer-family checkpoint pool, with one checkpoint evaluated both with and without four-view orientation-preserving TTA. Lowfrequency components are averaged, while high-frequency residuals are fused using a pixel-wise median to suppress inconsistent details.

## 4.7 AIIA-Lab

The team proposes a multi-stage pipeline for joint raindrop and reflection removal, adopting the Multi-Scale Deraining Transformer (MSDT) [3] as the backbone and augmenting it with checkpoint-based scene-specific selection, interframes fusion, and Flux [19] refinement. Its overall pipeline is illustrated in Fig. 8. Training Details.In Stage I, MSDT is trained for 500 epochs on 256 × 256 patches with batch size 1, using Adam and cosine learning-rate decay from $1 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 \times 1 0 ^ { - 6 }$ after 3 warm-up epochs. Augmentation includes random cropping, flipping, 90-degree rotations, and gamma and saturation adjustments, while several high-performing checkpoints are retained. In Stage II, their outputs are aggregated by mean or median fusion to form pseudo ground truth, which is then used to fine-tune Flux [19] as a refinement model.

Testing Details. At test time, they employ a sliding-window scheme to handle arbitrary resolutions: overlapping predictions are averaged, and edge padding with recomposition addresses non-divisible dimensions. In addition, A set of strong checkpoints of MSDT proceed each test image independently. For each scene, the best-performing checkpoint is selected case-by-case.

Implementation Details. The team uses MSDT as the primary restoration backbone and a fine-tuned FLUX model for pseudo-GT-based refinement. Both stages are developed exclusively using the oficial challenge data.

Ensembles and Fusion Strategies. Outputs of the same scene are fused via mean or median fusion. Mean fusion aggregates shared structural information; median fusion is more robust to local outliers and residual artifacts.

Note. This team trains its residual correction model using multiple observations of the same scene. Results for the multi-observation and standard single-image settings are reported separately in Tables 1 and 2.

## 4.8 ISCAS Optics

This team uses XResformer [8] as the backbone and proposes a coarse-to-fine training strategy, along with a self-ensemble testing strategy.

Training Details. Training follows a four-stage coarse-to-fine schedule with random cropping, flipping, and rotation augmentation. The model is first trained for 300k iterations on $2 5 6 \times 2 5 6$ patches using $\ell _ { 1 }$ loss, followed by another 300k iterations with $\mathcal { L } _ { \ell _ { 1 } } + 0 . 0 1 \times \mathcal { L } _ { \mathrm { F F T } }$ . The patch size is then increased to $3 2 0 \times 3 2 0$ for 210k iterations with $\ell _ { 1 }$ loss, and finally to $3 8 4 \times 3 8 4$ for another 210k iterations using the combined loss. The FFT loss provides frequency-domain supervision for recovering high-frequency details.

Ensembles and Fusion Strategies. The method combines eight-way geometric self-ensemble with multi-resolution inference. Predictions from rotated and flipped inputs are inversely transformed and averaged. Full-image inference is further fused with sliding-window inference using 384 × 384 patches and a 320- pixel overlap, balancing global context and local detail restoration.

## 4.9 sunsean

This team proposes a two-branch ensemble framework that combines a direct restoration model (RD) with the difusion-based DAI model [17]. Given a degraded input image $I _ { \mathrm { i n p u t } }$ , the two branches independently produce restored predictions $I _ { \mathrm { R D } }$ and $I _ { \mathrm { D A I } }$ . The final output is obtained by linear blending. The two branches provide complementary restoration characteristics, with the RD branch performing direct deterministic restoration and the DAI branch providing difusion-based refinement.

Testing Details. The RD branch directly processes the degraded image to produce $I _ { \mathrm { R D } }$ . For high-resolution inputs, the DAI branch adopts overlapping tiled inference using $5 1 2 \times 5 1 2$ patches with a 64-pixel overlap. The tile predictions are subsequently merged to reconstruct the full-resolution output.

Ensembles and Fusion Strategies. The DAI branch uses a scene-adaptive checkpoint selection strategy. For relatively simple scenes, inference is performed using a single checkpoint. For more challenging scenes, two complementary checkpoints independently process the same input, and their predictions are averaged with equal weights. The resulting DAI prediction is then linearly fused with the RD output using the weight defined above.

## 4.10 ULR

This team proposes a conditional difusion-based restoration framework based on Stable Difusion 2.1. DINOv3 [34] features are clustered using Afinity Propagation [14] to group images with similar degradations and provide cluster-level conditioning. A scene-level pseudo-reference is also generated by pixel-wise median aggregation and concatenated with each input image.

Training Details. The model is trained for up to 2,000 epochs using Adam with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a batch size of 1. Images are resized to $7 2 0 \times 1 0 8 0$ and randomly flipped horizontally. The DINOv3 encoder and VAE are frozen, while only the U-Net is updated.

Testing Details. Each input is concatenated with the scene-level pseudo-reference Inference is performed in two modes: one uses cluster-level DINOv3 features, while the other replaces them with zeros to reduce dependence on clustering. Ensembles and Fusion Strategies. The ensemble consists of two models initialized from the same pretrained weights and trained with diferent random seeds and data orderings.

Note. This method uses multiple degraded images from the same scene for conditioning and is therefore evaluated separately from standard single-image methods in Tables 1 and 2.

## 4.11 ustc\_pi\_lab

This team proposes a two-stage restoration framework that combines Histoformer with a signed multi-band residual subtractor, as shown in Fig. 9. In Stage I, Histoformer [35] serves as the restoration backbone to model long-range dependencies between regions with similar intensity statistics and produce an initial restored image. In Stage II, a compact residual subtractor processes a 21-channel representation of the degraded input, the Stage-I output, their diference, low- and high-frequency features, and an in-mask prior through a shared convolutional trunk with multiple residual heads, using the direct head to predict a signed residual for final refinement.

Training Details. The two stages are trained sequentially. Histoformer is initialized from a public checkpoint and fine-tuned on the oficial UR<sup>3</sup> paired data. Its cached outputs are then used to train the Stage-II direct branch for 30K steps on $5 1 2 \times 5 1 2$ patches, with a width of 64 and a synthetic-overlay probability of

![](images/d210da9b328db31736c3c4d2c163598853aa3db287fc86d416ffc6190bf0e631.jpg)  
Fig. 9: The pipeline of the method proposed by Team ustc\_pi\_lab.

0.20. The objective combines image reconstruction, residual supervision, multiband consistency, and clean-region regularization. The pretrained Histoformer checkpoint is counted as external model usage.

Testing Details. During testing, each degraded image is first processed by the fine-tuned Histoformer. The spatial mask prior is subsequently generated or loaded, and the resulting multi-band representation is fed into the direct residual branch to predict the correction layer in a single forward pass.

Implementation Details. The Stage-II residual model adopts the selected direct branch with a width of 64. Both stages are integrated into a single deterministic inference pipeline.

## 4.12 daniel0902

This team proposes a lightweight single-image mixture-of-experts network with a shared convolutional backbone, six residual blocks, and 192 base channels. A spatial router activates the top-2 of eight convolutional experts to adaptively handle raindrop- and reflection-degraded regions. The network uses dual reconstruction heads to predict the restored transmission (T) and reflection layer (R), constrained by the layer-consistency loss $( \| ( T + R ) - I \| )$ . This decomposition enables joint reflection removal and recovery of raindrop-occluded content, while only (T) is retained during inference.

Training Details. The model is trained from scratch on the oficial training set for 360 epochs using Adam, with 256 × 256 crops, a batch size of 24, and a learning rate of $1 \times 1 0 ^ { - \hat { 4 } }$ . Augmentation includes horizontal flipping and, during continued training, photometric transformations applied with a probability of 0.25, such as exposure and gamma adjustment, color jitter, mild noise, and veiling glare.

Testing Details. During testing, each image is restored independently using a single checkpoint and full-image inference at 1080\times 720 , without tiling. A twoview flip-based self-ensemble averages the predictions from the original image and its horizontally flipped version.

Implementation Details. This network contains 10.23M parameters and uses top-2 sparse expert activation. Training is performed on one NVIDIA H100 GPU with mixed precision and takes approximately 9 hours. Inference uses fp16 autocasting, with a reported runtime of approximately 28.5 seconds per image on an RTX 4060 laptop GPU and 1.3 seconds per image on an H100 GPU.

## Teams and Afiliations

## Organizers

Members:

Zewei He<sup>1,2</sup> (zeweihe@zju.edu.cn)

Xingyu Liu<sup>1,2</sup> (xingyu\_liu@zju.edu.cn)

Xi Tong<sup>1,2</sup> (tongxi\_zju@zju.edu.cn)

Yu Chen<sup>1,2</sup> (yuchen2024@zju.edu.cn)

Xin Li<sup>3</sup> (xin.li@ustc.edu.cn)

Afiliations: <sup>1</sup> Huanjiang Laboratory, Zhuji, China <sup>2</sup> Zhejiang University, Hangzhou, China <sup>3</sup> University of Science and Technology of China, Hefei, China

## Participants

Please check in our supplementary material.

## Acknowledgment

This work was supported in part by National Natural Science Foundation of China under Grant No. 52305590, Zhejiang Provincial Natural Science Foundation of China under Grant No. LQ24F010004, and Tianshan Talent Cultivation Plan - Science and Technology Innovation Team Project under Grant No. 2024TSYCTD0011.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

2. Cai, J., Yang, K., Ouyang, L., Fu, L., Ding, J., Shen, J., Meng, Z.: Openrr-5k: A large-scale benchmark for reflection removal in the wild. In: IEEE International Conference on Multimedia Information Processing and Retrieval. pp. 14–19 (2025)

3. Chen, H., Chen, X., Lu, J., Li, Y.: Rethinking multi-scale representations in deep deraining transformer. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 1046–1053 (2024)

4. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: ECCV. pp. 17–33 (2022)

5. Chen, X., Li, H., Dong, J., Pan, J., Li, X., et al.: The second LoViF 2026 challenge on real-world all-in-one image restoration: Methods and results. In: European Conference on Computer Vision Workshops (ECCVW) (2026)

6. Chen, X., Li, H., Li, M., Pan, J.: Learning A Sparse Transformer Network for Efective Image Deraining. In: CVPR. pp. 5896–5905 (2023)

7. Chen, X., Li, Z., Pu, Y., Liu, Y., Zhou, J., Qiao, Y., Dong, C.: A comparative study of image restoration networks for general backbone network design. arXiv preprint arXiv:2310.11881 (2023)

8. Chen, X., Li, Z., Pu, Y., Liu, Y., Zhou, J., Qiao, Y., Dong, C.: A comparative study of image restoration networks for general backbone network design. In: European Conference on Computer Vision. pp. 74–91. Springer (2024)

9. Chen, Z., He, Z., Lu, Z.M.: DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention. IEEE Transactions on Image Processing 33, 1002–1015 (2024)

10. Chen, Z., He, Z., Lu, Z., Sun, X., Lu, Z.M.: Prompt-based test-time real image dehazing: a novel pipeline. In: ECCV. pp. 432–449 (2024)

11. Chi, L., Jiang, B., Mu, Y.: Fast fourier convolution. NeurIPS (2020)

12. Conde, M.V., Geigle, G., Timofte, R.: Instructir: High-quality image restoration following human instructions. In: ECCV (2024)

13. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: CVPR (2016)

14. Frey, B.J., Dueck, D.: Clustering by passing messages between data points. science 315(5814), 972–976 (2007)

15. Guo, H., Li, J., Dai, T., Ouyang, Z., Ren, X., Xia, S.T.: Mambair: A simple baseline for image restoration with state-space model. In: ECCV (2024)

16. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

17. Hu, J., Yang, C., Zhou, Z., Fang, J., Yang, X., Tian, Q., Shen, W.: Dereflection Any Image with Difusion Priors and Diversified Data. In: AAAI (2026)

18. Hu, Q., Wang, H., Guo, X.: Single image reflection separation via dual-stream interactive transformers. In: NeurIPS. vol. 37, pp. 55228–55248 (2024)

19. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)

20. Li, M., Hu, J., Wang, H., Hu, Q., Wang, J., Guo, X.: Rectifying latent space for generative single-image reflection removal. In: CVPR. pp. 8397–8407 (2026)

21. Li, X., Jia, C., Gao, Y., et al.: LoViF 2026 AIGC image compression challenge: Methods and results. In: European Conference on Computer Vision Workshops (ECCVW) (2026)

22. Li, X., Jin, Y., Jin, X., Wu, Z., Li, B., Wang, Y., Yang, W., Li, Y., Chen, Z., Wen, B., et al.: Ntire 2025 challenge on day and night raindrop removal for dual-focused images: Methods and results. In: CVPR Workshop. pp. 1172–1183 (2025)

23. Ling, X., Zhou, C., Li, X., Lu, G., et al.: LoViF 2026 challenge on ultra-low bitrate image compression: Methods and results. In: European Conference on Computer Vision Workshops (ECCVW) (2026)

24. Liu, X., He, Z., Chen, Y., Zhu, C., Chen, Z., Luo, X., Lu, Z.M.: Unified removal of raindrops and reflections: A new benchmark and a novel pipeline. In: ECCV (2026)

25. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: CVPR. pp. 11976–11986 (2022)

26. Oh, Y., Kwon, J., Cho, N.I.: Dinolight: Robust ambient light normalization with self-supervised visual prior integration. arXiv preprint arXiv:2603.12579 (2026)

27. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024)

28. Qian, R., Tan, R.T., Yang, W., Su, J., Liu, J.: Attentive Generative Adversarial Network for Raindrop Removal from a Single Image. In: CVPR. pp. 2482–2491 (2018)

29. Qu, W., Liang, S., Pan, C., Yang, Z., Zhou, G., Fu, X., Liu, B., Wang, C., Elazab, A.: Weatherremover: All-in-one adverse weather removal with multi-scale feature map compression. IEEE Transactions on Artificial Intelligence (2025)

30. Quan, R., Yu, X., Liang, Y., Yang, Y.: Removing Raindrops and Rain Streaks in One Go. In: CVPR. pp. 9143–9152 (2021)

31. Quan, Y., Deng, S., Chen, Y., Ji, H.: Deep learning for seeing through window with raindrops. In: ICCV. pp. 2463–2471 (2019)

32. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR. pp. 10684–10695 (2022)

33. Shao, M.W., Li, L., Meng, D.Y., Zuo, W.M.: Uncertainty Guided Multi-Scale Attention Network for Raindrop Removal from a Single Image. IEEE Transactions on Image Processing 30, 4828–4839 (2021)

34. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

35. Sun, S., Ren, W., Gao, X., Wang, R., Cao, X.: Restoring Images in Adverse Weather Conditions via Histogram Transformer. In: ECCV (2024)

36. Wan, R., Shi, B., Duan, L.Y., Tan, A.H., Kot, A.C.: Benchmarking Single-Image Reflection Removal Algorithms. In: ICCV. pp. 3922–3930 (2017)

37. Xu, T., Zhang, C., Zhai, G., Liu, X.: FUMO: Prior-modulated coarse-to-fine restoration for single image reflection removal. arXiv preprint arXiv:2603.19036 (2026)

38. Yang, J., Li, C., Dai, X., Gao, J.: Focal modulation networks. In: NeurIPS (2022)

39. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. In: NeurIPS (2024)

40. Yao, S., Jin, Y., et al.: LoViF 2026 the third challenge on day and night raindrop removal for dual-focused images: Methods and results. In: European Conference on Computer Vision Workshops (ECCVW) (2026)

41. You, S., Tan, R.T., Kawakami, R., Ikeuchi, K.: Adherent Raindrop Detection and Removal in Video. IEEE Transactions on Pattern Analysis and Machine Intelligence 38(9), 1721–1733 (2016)

42. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Eficient transformer for high-resolution image restoration. In: CVPR (2022)

43. Zhang, J., Lyu, Q., Gao, R., Liu, E., Mo, T., Liang, Z., Liu, S., Wu, X., Li, X., Wang, W., Wang, G., Wang, K., Chen, T., Torr, P., Lin, L.: The LoViF 2026 Act-PhysCause challenge: A baseline report on action-conditioned video world models. In: European Conference on Computer Vision Workshops (ECCVW) (2026)

44. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: ICCV. pp. 3836–3847 (2023)

45. Zhang, X., Zhang, Y., Yu, F.: HiT-SR: Hierarchical transformer for eficient image super-resolution. In: ECCV. pp. 483–500 (2024)

46. Zhang, X., Ng, R., Chen, Q.: Single image reflection separation with perceptual losses. In: CVPR. pp. 4786–4794 (2018)

47. Zhao, H., Li, M., Hu, Q., Guo, X.: Reversible decoupling network for single image reflection removal. In: CVPR. pp. 26430–26439 (2025)

48. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)