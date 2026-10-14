# Dual-Domain Sparse-View 4D CBCT Reconstruction with Spatial-Temporal Consistency

Learning Qi Lai<sup>1</sup> and YuTong He<sup>1,2</sup>

Contributing authors: q.lai@siat.ac.cn; heyt3@mail.sustech.edu.cn;

## Abstract

Four-dimensional cone-beam computed tomography (4D CBCT) provides phaseresolved volumetric images for image-guided radiotherapy. Yet good reconstruction quality usually needs many projections. This increases scan time and radiation dose. We propose STCNet, a sparse-view 4D CBCT reconstruction framework that uses spatial-temporal consistency learning. STCNet reconstructs high-quality images from only 16 projections in each respiratory phase. STCNet integrates physics-driven positional encoding to enforce geometric consistency in the projection domain and introduces temporal encoding in latent space to capture respiratory motion dynamics across phases to capture dynamic respiratory motion across phases. A dual-domain collaborative strategy jointly optimizes both projection and image domains, enhancing structural fidelity and temporal coherence. We tested STCNet on the TCIA-LUNG dataset and a CIRS thorax phantom dataset. STCNet performed better than conventional methods and recent deep learning baselines in image quality, artifact reduction, and structural preservation. On TCIA-LUNG, STCNet reached PSNR values of 33.311, 32.695, and 32.658 dB at phases 2, 5, and 8. The corresponding SSIM values were 0.845, 0.823, and 0.825. On CIRS, STCNet achieved PSNR values up to 36.497 dB and SSIM values up to 0.877. These results support STCNet as a promising framework for low-dose, phase-resolved 4D CBCT reconstruction in image-guided radiotherapy.

## 1 Introduction

Image-guided radiation therapy (IGRT) has become integral to contemporary radiotherapy by enabling treatment delivery to be aligned with patient anatomy at the time of irradiation [1–4]. This role is especially critical for thoracic malignancies, where respiratory motion can introduce substantial uncertainty in tumor position during treatment and thereby compromise targeting accuracy [3–6]. In this setting, robust on-board imaging is essential not only for setup verification, but also for motion-aware assessment of the tumor and surrounding normal tissues [1–4]. Cone-beam computed tomography (CBCT) is widely used for volumetric image guidance on the treatment unit. [1–5] However, conventional three-dimensional CBCT has well-recognized limitations in the thorax because respiratory motion is averaged into a single reconstructed volume, often causing motion blur, streak artifacts, and loss of boundary definition [3, 6–8]. These degradations can reduce confidence in target localization and limit the reliability of image guidance for moving lesions [3–6].

Four-dimensional CBCT (4D CBCT) was developed to overcome this limitation by sorting projection data into respiratory-correlated bins and reconstructing phaseresolved image volumes [7, 9]. By preserving temporal information across the breathing cycle, 4D CBCT can improve visualization of moving targets and provide a more informative representation of respiratory motion than conventional CBCT [8, 10–13]. This capability is particularly relevant for radiotherapy workflows that depend on accurate motion characterization, including target verification, treatment adaptation, and motion management [3, 10, 13]. Despite its clinical appeal, routine use of 4D CBCT remains constrained by image quality limitations. Because the acquired projections are divided across respiratory phases, each phase is reconstructed from only a small subset of views, making 4D CBCT inherently a sparse-view problem [7, 13, 14]. The resulting undersampling can produce severe streaking artifacts, structural distortion, and unstable intensity representation, particularly when the number of projections per phase is further reduced to limit imaging dose [10, 12–14]. These limitations directly afect the clinical interpretability of phase-resolved images. Dose eficiency is therefore a central challenge in 4D CBCT. In practice, obtaining acceptable image quality often requires many projections acquired over multiple breathing cycles, increasing scan time and imaging dose [15]. This is undesirable in radiotherapy, where repeated on-board imaging may be needed throughout a treatment course [1, 3, 4]. Meth ods that enable high-quality reconstruction from substantially fewer projections could therefore improve the practicality of low-dose 4D CBCT for routine clinical guidance [16, 17]. A broad range of strategies has been proposed to address sparse-view 4D CBCT reconstruction. Motion-compensated methods exploit inter-phase deformation to reduce respiratory artifacts and improve reconstruction fidelity [7, 10]. Regularized iterative approaches, including temporal nonlocal and motion-compensated formulations, can further suppress noise and undersampling artifacts [11, 12, 14]. However, these approaches often depend on accurate motion estimation, may be computationally intensive, and can oversmooth fine structures when the acquisition is extremely sparse. Deep learning has recently emerged as a powerful alternative for sparse-view CT and CBCT reconstruction [17–23]. Prior studies have explored image-domain artifact reduction [16, 23, 24], sinogram- or projection-domain synthesis [19, 20], ultra-sparseview volumetric reconstruction [21, 25], and generative modeling frameworks based on difusion or flow models [26, 27]. Deep learning has also been used to derive fourdimensional CBCT information from highly limited acquisition data, highlighting the potential of data-driven reconstruction for low-dose imaging [15]. Nevertheless, many existing methods primarily address either data completion or image enhancement in isolation, without fully exploiting the coupled geometric structure of projection acquisition and the temporal continuity of respiratory motion that define the 4D CBCT problem [19, 25, 28]. For 4D CBCT, these two forms of prior information are fundamentally complementary. Projection measurements are governed by fixed acquisition geometry, whereas reconstructed anatomy evolves in a temporally correlated manner across respiratory phases [11–13, 23]. A reconstruction framework that jointly models both relationships may therefore recover missing information more faithfully than approaches operating in only a single domain or without explicit temporal consistency [15, 17, 25]. In this work, we propose STCNet, a sparse-view 4D CBCT reconstruction framework based on spatial-temporal consistency learning. STCNet is designed for low-dose phase-resolved imaging using only 16 projections per respiratory phase. The framework combines projection-domain completion with image-domain refinement after Feldkamp-Davis-Kress reconstruction. In the projection domain, physics-aware positional encoding is used to represent angular acquisition geometry, while temporal encoding is introduced to capture respiratory dynamics across phases. The completed projections are then reconstructed and further refined to improve structural fidelity and suppress residual sparse-view artifacts.

![](images/abc2a8f31bd98190d5df082483c5251dcdcdd4af84f427e0f1012c8dac9f24bb.jpg)  
Fig. 1: Overview of the proposed STCNet framework for dual-domain sparse-view 4D CBCT reconstruction with spatial-temporal consistency learning. Sparse-view projections are sorted into P = 10 respiratory phases, with N = 16 projections per phase. The projection-domain generation module (PDGM) incorporates physics-aware positional encoding of acquisition angles and LSTM-based temporal consistency encoding to generate K = 256 dense-view projections per phase. FDK reconstruction converts the generated projections into initial phase-resolved volumes, which are subsequently refined by the image-domain correction module (IDCM) to produce the final 4D CBCT sequence.

We evaluate STCNet using the TCIA-LUNG dataset [29, 30]. Its performance is compared with conventional reconstruction approaches and representative deep learning baselines under an ultra-sparse acquisition setting. Our aim is to determine whether explicit modeling of geometric and temporal consistency can improve image quality and anatomical fidelity, and thereby support a more clinically feasible pathway toward low-dose 4D CBCT for image-guided radiotherapy.

## 2 Results

## 2.1 Dataset Definition

Two datasets were used to evaluate the proposed STCNet: CIRS thorax phantom dataset for experimental validation and the publicly available TCIA-LUNG dataset [5, 6, 29, 30] for evaluating generalization performance. The CIRS dataset was acquired in our laboratory and is a private dataset, providing experimentally measured CBCT projections under controlled respiratory motion. The TCIA-LUNG dataset was used to assess the applicability of the proposed method beyond the controlled phantom setting. The acquisition and reconstruction parameters are summarized in Table 1.

Table 1: Acquisition parameters for the TCIA-LUNG and CIRS phantom datasets
<table><tr><td>Parameter</td><td>TCIA-LUNG [5, 6, 29, 30]</td><td>CIRS phantom</td></tr><tr><td>DSD/DSO (mm)</td><td>1500/1000</td><td>1800/1000</td></tr><tr><td>Pixel size  $( \mathrm { m m } ^ { 2 } )$ </td><td> $0 . 9 5 \times 0 . 9 5$ </td><td> $0 . 9 5 \times 0 . 9 5$ </td></tr><tr><td>Detector size</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td> $1 0 2 4 \times 1 0 2 4$ </td></tr><tr><td>Angle range</td><td>0°-360°</td><td>0°-360°</td></tr><tr><td>Voxel size (mm³)</td><td> $0 . 9 8 \times 0 . 9 8 \times 1$ </td><td> $0 . 9 8 \times 0 . 9 8 \times 1$ </td></tr><tr><td>Dimension</td><td> $5 1 2 \times 5 1 2 \times \mathrm { N } _ { z }$ </td><td> $5 1 2 \times 5 1 2 \times \mathrm { N } _ { z }$ </td></tr></table>

Remark: : N denotes the number of slices, which varied across cases/datasets.

CIRS Thorax Phantom Dataset To evaluate the reconstruction performance of STCNet under controlled experimental conditions, a dynamic CIRS thorax phantom was scanned using a CBCT imaging system in our laboratory. The phantom was scanned along a circular acquisition trajectory and contained a spherical target serving as a tumor surrogate. During acquisition, the target underwent prescribed sinusoidal motion to simulate respiratory-induced tumor displacement.

A total of 2,560 projection images were acquired over a full $3 6 0 ^ { \circ }$ rotation, with an angular interval of approximately $0 . 1 4 ^ { \circ }$ between adjacent views. Projections were acquired continuously while the target followed the predefined respiratory trajectory. Each projection was assigned to a respiratory phase according to its position within the motion cycle. The acquired projections were retrospectively sorted into 10 respiratory phase bins, with projections corresponding to the same phase across diferent respiratory cycles grouped together. These phase-specific projection subsets were used for subsequent sparse-view reconstruction experiments.

The source-to-object distance (SOD) and source-to-detector distance (SDD) were 1,000 mm and 1,800 mm, respectively. The detector comprised 3072×3072 pixels, with a physical pixel size of $\mathrm { 0 . 1 7 2 \times 0 . 1 7 2 ~ m m ^ { 2 } }$ . The X-ray tube voltage and current were set to 120 kVp and 0.4 mA, respectively, with an exposure time of [to be specified] ms per projection.

Before reconstruction, the acquired projections were corrected using the corresponding flat-field and dark-field images. The corrected projections were subsequently converted into line-integral measurements through a negative logarithmic transformation. The reconstructed volumes had a matrix size of [to be specified] voxels and a voxel spacing of [to be specified] mm. Unless otherwise specified, all reconstruction methods evaluated on the CIRS dataset used the same acquisition geometry and preprocessing pipeline to ensure a consistent comparison.

The CIRS dataset was used for experimental validation of STCNet using physically acquired sparse-view CBCT projections. This controlled setting enabled evaluation of phase-resolved reconstruction quality and recovery of the moving target in the presence of respiratory motion.

TCIA-LUNG Dataset The publicly available TCIA-LUNG dataset [5, 6, 29, 30] was used to evaluate the generalization performance of STCNet beyond the controlled phantom setting. In contrast to the CIRS phantom experiments, this dataset provided patient anatomical structures for assessing the applicability of the proposed method to more complex thoracic anatomy.

## 2.2 Comparison Methods

We compared STCNet with both conventional reconstruction methods and representative deep learning baselines. The conventional methods included FDK, simultaneous algebraic reconstruction technique (SART) [31], and SART-TV / OSCTV-based sparse-view reconstruction [14]. The deep learning baselines included N-Net, CyCN-Net [9], and AttUNet [16] . We also evaluated the projection-domain module alone (PDGM) to assess the contribution of dual-domain refinement.

## 2.3 Evaluation Metrics

We evaluated reconstruction quality using mean absolute error (MAE) [32], peak signal-to-noise ratio (PSNR) [33], and structural similarity index (SSIM) [34]. MAE measures the absolute intensity diference between the reconstructed image and the reference image, with lower values indicating better reconstruction accuracy. PSNR quantifies fidelity in terms of signal-to-error ratio, with higher values indicating better image quality.SSIM measures structural similarity and perceptual consistency between images, with higher values indicating better structural preservation.

## 2.4 Implement Details

All experiments were implemented in Python using the PyTorch framework. Training and validation were conducted on a workstation equipped with an Intel CPU, 128 GB RAM, and an NVIDIA A6000 GPU with 48 GB memory.

For TCIA-LUNG, each phase volume was forward projected into 256-view dense projection data, and 16 uniformly sampled views were selected as sparse-view input. The PDGM and IDCM were trained as the two sequential components of the overall framework. During inference, the trained STCNet model took sparse-view phaseresolved projection data as input and generated the final reconstructed 4D CBCT volumes.

## 2.5 Quantitative Comparison

Table 2 presents the quantitative results on axial slices from the CIRS thorax phantom and TCIA-LUNG datasets at respiratory phases 2, 5, and 8.

Table 2: Quantitative comparison of reconstruction performance on axial slices from the CIRS Phantom and TCIA-LUNG datasets.
<table><tr><td rowspan="3">Methods</td><td colspan="9">CIRS Phantom</td></tr><tr><td colspan="3">Phase2</td><td colspan="3">Phase5</td><td colspan="3">Phase8</td></tr><tr><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>FDK [35]</td><td>0.196</td><td>22.316</td><td>0.324</td><td>0.194</td><td>22.289</td><td>0.349</td><td>0.189</td><td>22.326</td><td>0.335</td></tr><tr><td>SART [31]</td><td>0.129</td><td>25.893</td><td>0.349</td><td>0.123</td><td>25.719</td><td>0.346</td><td>0.124</td><td>25.892</td><td>0.349</td></tr><tr><td>SART-TV [14]</td><td>0.113</td><td>26.735</td><td>0.457</td><td>0.117</td><td>26.741</td><td>0.461</td><td>0.121</td><td>26.658</td><td>0.449</td></tr><tr><td>N-Net [9]</td><td>0.071</td><td>20.322</td><td>0.506</td><td>0.071</td><td>20.313</td><td>0.505</td><td>0.072</td><td>20.126</td><td>0.499</td></tr><tr><td>CycN-Net [9]</td><td>0.102</td><td>23.338</td><td>0.423</td><td>0.102</td><td>23.239</td><td>0.422</td><td>0.103</td><td>23.160</td><td>0.421</td></tr><tr><td>AttUnet[16]</td><td>0.068</td><td>24.493</td><td>0.261</td><td>0.066</td><td>24.358</td><td>0.265</td><td>0.062</td><td>24.374</td><td>0.258</td></tr><tr><td>PGDM</td><td>0.063</td><td>30.513</td><td>0.711</td><td>0.061</td><td>30.408</td><td>0.717</td><td>0.065</td><td>30.501</td><td>0.710</td></tr><tr><td>STCNet</td><td>0.018</td><td>36.497</td><td>0.871</td><td>0.017</td><td>36.392</td><td>0.874</td><td>0.018</td><td>36.261</td><td>0.877</td></tr><tr><td rowspan="2"></td><td colspan="10">TCIA-LUNG</td></tr><tr><td>Phase2</td><td></td><td></td><td></td><td>Phase5</td><td></td><td></td><td>Phase8</td><td></td></tr><tr><td>FDK [35]</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>SART [31]</td><td>0.180</td><td>21.585</td><td>0.309</td><td>0.174</td><td>22.077</td><td>0.317</td><td>0.182</td><td>21.687</td><td>0.308</td></tr><tr><td></td><td>0.073</td><td>22.573</td><td>0.536</td><td>0.076</td><td>22.360</td><td>0.530</td><td>0.068</td><td>22.177</td><td>0.528</td></tr><tr><td>SART-TV [14]</td><td>0.071</td><td>23.453</td><td>0.540</td><td>0.064</td><td>23.377</td><td>0.538</td><td>0.067</td><td>23.264</td><td>0.537</td></tr><tr><td>N-Net [9]</td><td>0.136</td><td>22.025</td><td>0.473</td><td>0.129</td><td>22.137</td><td>0.468</td><td>0.133</td><td>22.021</td><td>0.471</td></tr><tr><td>CycN-Net [9]</td><td>0.113</td><td>22.483</td><td>0.564</td><td>0.109</td><td>22.514</td><td>0.557</td><td>0.107</td><td>22.488</td><td>0.561</td></tr><tr><td>AttUnet [16]</td><td>0.174</td><td>23.631</td><td>0.293</td><td>0.171</td><td>23.797</td><td>0.274</td><td>0.168</td><td>24.124</td><td>0.292</td></tr><tr><td>PGDM</td><td>0.059</td><td>30.332</td><td>0.713</td><td>0.061</td><td>29.744</td><td>0.715</td><td>0.058</td><td>29.738</td><td>0.713</td></tr><tr><td>STCNet</td><td>0.035</td><td>33.311</td><td>0.845</td><td>0.032</td><td>32.695</td><td>0.823</td><td>0.039</td><td>32.658</td><td>0.825</td></tr></table>

Remark: the Bold and underlined indicate this metric’s best and second-best performance, respectively. ↑(↓) indicates that the larger (smaller) the value, the better the performance.

STCNet achieved the lowest MAE and the highest PSNR and SSIM at all reported phases on both datasets (Table 2). On the CIRS thorax phantom dataset, SART-TV improved PSNR from 22.289–22.326 dB with FDK [35] to 26.658–26.741 dB, whereas the deep-learning baselines showed mixed performance across metrics. PDGM ranked second for all three metrics at each evaluated phase. STCNet further increased PSNR to 36.497, 36.392, and 36.261 dB at phases 2, 5, and 8, respectively, with corresponding SSIM values of 0.871, 0.874, and 0.877 and MAE values of 0.018, 0.017, and 0.018. Its PSNR gains of 5.760–5.984 dB over PDGM support the additional benefit of image-domain correction in the phantom experiments.

On the TCIA-LUNG dataset, SART and SART-TV improved all three metrics relative to FDK, while the deep-learning baselines showed mixed performance across metrics. PDGM ranked second at all evaluated phases, achieving PSNR values of 29.738–30.332 dB and SSIM values of 0.713–0.715. STCNet further increased PSNR to 33.311, 32.695, and 32.658 dB at phases 2, 5, and 8, respectively, with corresponding SSIM values of 0.845, 0.823, and 0.825 and MAE values of 0.035, 0.032, and 0.039. The PSNR gains of 2.920–2.979 dB over PDGM support the generalization performance of the dual-domain framework beyond the controlled phantom setting.

Figure 2 additionally compares PSNR and SSIM in the coronal and sagittal planes. The CIRS dataset was used for experimental validation, whereas the TCIA-LUNG dataset was used to assess generalization beyond the controlled phantom setting. The coronal and sagittal comparisons in Fig. 2 show a similar trend: STCNet consistently achieved the highest PSNR and SSIM across the displayed phases on both datasets. Together with the axial-slice results, these findings demonstrate improved reconstruction fidelity and structural similarity across the evaluated anatomical planes and respiratory phases, supporting the complementary roles of projection-domain generation and image-domain correction.

## 2.6 Reconstruction Quality

We then tested the model on the CIRS thorax phantom dataset. Figure 3 shows the results.

The FDK [35] again exhibited severe artifacts, particularly in the axial plane, which made tumor-marker boundaries dificult to identify. SART [31] and SART-TV [14] reduced some of these errors but still left visible artifacts and incomplete boundaries. The deep-learning baselines did not perform consistently: some methods improved smoothness, but they often missed fine structural details or distorted the marker boundaries.

STCNet provided the clearest reconstructions on this dataset. Marker boundaries were more accurate, structures appeared more continuous, and residual artifacts were fewer. These results suggest that STCNet can generalize beyond the training-like data setting.

## 2.7 Generalization Ability

We first tested STCNet on the TCIA-LUNG dataset. Figure 4 shows coronal, sagittal, and axial results at phases 2, 5, and 8.

![](images/e06d1b7bf4e04af79a86ef504ecf8cfa4feec0e3f626067d4823788b1248676f.jpg)  
(a)

![](images/32eea00d8f394539aa2016b73cbaec33ee424f310ab2a597e24533fbdc1faf88.jpg)  
(b)

![](images/24c180ae8c958bfe2729ae60e2cbd41caf0f4f52f44fdf94eff32dd3c8a47723.jpg)  
(c)

![](images/ed03575572251acc16a4345f7f470951cc04800666b45b99bff19309a96132e9.jpg)  
(d)  
Fig. 2: Quantitative comparison across planes and phases: The figure shows PSNR and SSIM results for representative phases in coronal and sagittal views. Panel (a) shows TCIA-LUNG coronal results. Panel (b) shows TCIA-LUNG sagittal results. Panel (c) shows CIRS coronal results. Panel (d) shows CIRS sagittal results.

The FDK [35] produced strong streak artifacts under the sparse-view setting. The ribs and spine looked blurred and broken. Soft tissue boundaries were also unclear. SART and SART-TV improved the overall appearance, but both methods still had visible noise and smoothing efects. SART [31] kept more structure than FDK, but it introduced noisy textures. SART-TV [14] reduced some noise, but it also blurred fine details. The deep learning baselines showed mixed results. N-Net and CycN-Net [9] reduced some artifacts, but they did not recover fine structures well. AttUNet [16] produced smoother images, but it still caused anatomical distortion in some regions.

STCNet provided the best overall visual quality among the tested methods. The reconstructions showed fewer streak artifacts. The diaphragm region was more clearly delineated. Bone and soft tissue boundaries were more natural. The overall anatomical structure was also closer to the ground truth. PDGM alone already improved the image quality a lot. The full STCNet improved it further after image-domain correction.

![](images/f7a539bb48efff1bfc1dbc2f15ab1d484dbc5c6a54b93a584199bcc15746d0fe.jpg)  
Fig. 3: Reconstruction results on the CIRS phantom dataset: The figure shows coronal, sagittal, and axial slices at phases 2, 5, and 8 for diferent methods. The red and yellow dashed lines mark tumor marker boundary positions across phases.

## 2.8 HU Error Map Analysis

We also compared HU diference maps on the TCIA-LUNG dataset (Figure 5). These maps show the diference between each reconstruction and the ground truth. The FDK method showed large error regions across all planes. The largest errors appeared near dense structures and tissue boundaries. SART reduced part of the error, but it still showed scattered noise-like deviations. SART-TV reduced some noisy errors, but it also produced boundary loss in transition areas.

N-Net and CycN-Net still showed clear deviations near the lung boundary and mediastinal regions. AttUNet reduced some errors, but noticeable distortion remained in dificult areas. PDGM reduced HU errors by a large margin. This result shows that projection completion is important for sparse-view reconstruction. STCNet reduced the errors even more after adding image-domain correction. The remaining errors became smaller and more localized. This pattern was clear near the pulmonary hilum, airway regions, and soft tissue interfaces.

![](images/92c82618e43f67f52f375a22bfb2748532add5e61f46ab8689726d324f9100a7.jpg)  
Fig. 4: Reconstruction results on the TCIA-LUNG dataset: The figure shows coronal, sagittal, and axial slices at phases 2, 5, and 8 for diferent methods. The red and yellow dashed lines mark the highest and lowest diaphragm positions and show the motion path across phases.

## 2.9 Ablation Study

We used ablation experiments to test the role of each module. Tables 3, 4, and Figure 6 show the results.

On the TCIA-LUNG dataset, the baseline model reached an MAE of 0.162, a PSNR of 23.916 dB, and an SSIM of 0.382. When we added PDGM, the model substantially improved. The MAE dropped to 0.059. The PSNR increased to 30.352 dB. The SSIM increased to 0.747. These changes show that projection-domain generation is a key part of the framework.

When we added IDCM alone, the model also improved. But the gain was smaller than the gain from PDGM. This result suggests that image-domain correction is useful, but it depends on a good initial reconstruction. When we combined PDGM and IDCM, the model achieved the best results on both datasets. On TCIA-LUNG, STC-Net reached an MAE of 0.036, a PSNR of 32.536 dB, and an SSIM of 0.824. On CIRS,

![](images/75a067f2319257e3004047077e5af1c7776711d811081b1e7fef4211c7e0ba54.jpg)  
Fig. 5: HU diference maps on the TCIA-LUNG dataset: The maps show axial, coronal, and sagittal diferences between reconstructed images and ground truth. Red and blue represent positive and negative HU errors.

Table 3: Ablation study of PDGM and IDCM on the TCIA-LUNG dataset
<table><tr><td>Baseline</td><td>PDGM</td><td>IDCM</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>√</td><td></td><td></td><td>0.162</td><td>23.916</td><td>0.382</td></tr><tr><td>√</td><td>√</td><td></td><td>0.059</td><td>30.352</td><td>0.747</td></tr><tr><td>√</td><td></td><td>√</td><td>0.162</td><td>22.360</td><td>0.695</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.036</td><td>32.536</td><td>0.824</td></tr></table>

Remark: the PDGM and IDCM represent Projection Domain Generation Module and Image Domain Correction Module. The Bold and underline indicate this metric’s best and second-best performance, respectively.

Table 4: Ablation study of PDGM and IDCM on the CIRS phantom dataset
<table><tr><td>Baseline</td><td>PDGM</td><td>IDCM</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>√</td><td></td><td></td><td>0.102</td><td>24.916</td><td>0.332</td></tr><tr><td>√</td><td>√</td><td></td><td>0.064</td><td>31.134</td><td>0.783</td></tr><tr><td>√</td><td></td><td>√</td><td>0.162</td><td>29.697</td><td>0.695</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.018</td><td>36.543</td><td>0.878</td></tr></table>

\* Remark: the PDGM and IDCM represent Projection Domain Generation Module and Image Domain Correction Module. The Bold and underline indicate this metric’s best and second-best performance, respectively.

STCNet reached an MAE of 0.018, a PSNR of 36.543 dB, and an SSIM of 0.878. Figure 6 shows that this advantage remained stable across phases. These findings support the idea that the two modules work well together.

## 3 Discussion

This study presents STCNet, a dual-domain framework for sparse-view 4D CBCT reconstruction. The method combines projection completion and image refinement. The method also models both spatial geometry and respiratory motion. The results show that STCNet improves image quality on both TCIA-LUNG and CIRS data.

The results support two main points. First, projection-domain recovery matters a lot in sparse-view 4D CBCT. PDGM alone already produced large gains over the baseline and other methods. This result suggests that early recovery of missing projection information is important. Second, image-domain correction adds further value after reconstruction. IDCM reduced residual artifacts and improved local structure. The best performance came from combining both modules.

The spatial-temporal design also played an important role. Positional encoding helped the model learn the geometric relation between sparse angles. Temporal encoding helped the model use the continuity between breathing phases. These two parts likely explain why the method preserved phase-wise structure better than the comparison methods.

![](images/bedd4b1bee8a86fe83898845122f44a584e4cddab200f7e4ecb84cf4417e71c3.jpg)  
Fig. 6: Ablation study of STCNet: The figure shows phase-wise PSNR and SSIM results for the baseline, baseline + PDGM, baseline + IDCM, and STCNet. Panels (a) and (b) show TCIA-LUNG results. Panels (c) and (d) show CIRS results.

The study also has practical value. Sparse-view reconstruction can help reduce imaging dose and scan burden in 4D CBCT. STCNet used only 16 projections per phase and still produced strong results. This suggests that the method may support lower-dose imaging protocols in radiotherapy guidance. Improved phase-resolved image fidelity may also benefit downstream radiotherapy tasks, such as target localization, motion assessment, and adaptive decision-making, although these applications were not directly evaluated here.

Still, this work has several limitations. The first limitation is the sampling setting. We mainly trained and tested the model under a 16-view-per-phase design. The current results do not show how the model performs under more extreme sparse-view settings, such as 10 views or fewer. The second limitation is the lack of explicit anatomical priors. The method does not directly use structural knowledge about airways, vessels, or organ topology. This may reduce reconstruction reliability in cases with larger motion or more complex anatomy. The third limitation is the evaluation scope. We focused on image reconstruction metrics and visual quality. We did not test downstream tasks such as tumor localization, motion estimation, or dose calculation.

Future work can address these points. One direction is to add anatomical priors from 4D CT or MRI. Another direction is to design adaptive view selection strategies for sparse acquisition. A third direction is to test the method in taskbased radiotherapy settings. These steps would help clarify the clinical value of the framework.

## 4 Methodology

In this section, We will introduce STCNet in detail, including Problem Definition, Projection Domain Generation Module (PDGM), Image Domain Correction Module (IDCM), Total Loss and some details of the network architecture. The workflow is shown in Figure 7.

![](images/121b5d49c2c3a49229cb755fb486fba0f2d83861eacfef0b0ab24b132ad42539.jpg)  
Fig. 7: Overview of the STCNet framework: The workflow starts with sparse-view projections sorted by respiratory phase. The PDGM uses a Swin-UNet backbone with positional and temporal encoding to generate dense-view projections. The method then applies FDK reconstruction. The IDCM refines the reconstructed volumes and outputs final phase-resolved 4D CBCT images.

## 4.1 Architecture Overview

We formulate sparse-view 4D CBCT reconstruction as a phase-resolved volumetric imaging problem with $P = 1 0$ respiratory phases. For each phase t, a sparse projection set is denoted as

$$
{ \cal S } _ { t } \ = \ \Big \{ p _ { n } ^ { ( t ) } \Big \} _ { n = 1 } ^ { N } ,\tag{1}
$$

where $N = 1 6$ is the number of sparse projections and $p _ { n } ^ { ( t ) }$ is the projection acquired at angle $\theta _ { n } ^ { ( t ) }$ . The corresponding angle set is

$$
\Theta _ { t } \ = \ \Big \{ \theta _ { n } ^ { ( t ) } \Big \} _ { n = 1 } ^ { N } ,\tag{2}
$$

Given the sparse-view input $\boldsymbol { S } _ { t } ,$ the goal is to reconstruct a high-quality phaseresolved image volume $\mathcal { T } _ { t }$ for each respiratory phase.

The PDGM first maps the sparse-view projection set to a dense-view projection representation:

$$
\hat { \mathcal { P } } _ { t } ~ = ~ F _ { \mathrm { P D G M } } ( S _ { t } , \Theta _ { t } )\tag{3}
$$

where

$$
\hat { \mathcal P } _ { t } ~ = ~ \left\{ \hat { p } _ { k } ^ { ( t ) } \right\} _ { k = 1 } ^ { K } ,\tag{4}
$$

and $K = 2 5 6$ is the number of synthesized projections for each phase.

The densified projections are then reconstructed by FDK to obtain the initial image-domain volume:

$$
\mathcal { T } _ { t } ^ { ( 0 ) } = \mathrm { F D K } ( \hat { \mathcal { P } } _ { t } ) , \qquad t = 1 , \dotsc , P\tag{5}
$$

The image-domain correction module then refines the initial reconstruction:

$$
\mathcal { T } _ { t } ~ = ~ F _ { \mathrm { I D C M } } ( \mathcal { T } _ { t } ^ { ( 0 ) } , t = 1 , \ldots , P )\tag{6}
$$

Accordingly, the overall STCNet pipeline can be expressed as

$$
F _ { \mathrm { S T C N e t } } = F _ { \mathrm { I D C M } } \circ F _ { \mathrm { F D K } } \circ F _ { \mathrm { P D G M } }\tag{7}
$$

## 4.2 Projection Domain Generation Module (PDGM)

The PDGM is designed to generate dense-view projections from sparse-view input within each respiratory phase. This step addresses the intrinsic data insuficiency of sparse-view 4D CBCT and provides a more complete projection set for subsequent reconstruction. During training, the generated dense-view projections are supervised by the corresponding ground-truth projection set $\hat { \mathcal { P } } _ { t } ^ { g t }$ . The optimization target is

$$
\operatorname * { a r g m i n } _ { ( \hat { \mathcal { P } } _ { t } , \hat { \mathcal { P } } _ { t } ^ { g t } ) \in \mathcal { D } } \left\| \hat { \mathcal { P } } _ { t } ^ { g t } - \hat { \mathcal { P } } _ { t } \right\| _ { 2 } ^ { 2 } ,\tag{8}
$$

where D denotes the projection-domain training dataset.

We used Swin-UNet as the backbone architecture of the PDGM.[23] For a sparse projection tensor $S _ { t } \in \mathbb { R } ^ { c \times n \times h \times w }$ , the encoder extracts hierarchical projection features:

$$
X _ { s } = \operatorname { E n c o d e r } ( S _ { t } ) ,\tag{9}
$$

where $X _ { s } \in \mathbb { R } ^ { c ^ { \prime } \times n ^ { \prime } \times h ^ { \prime } \times w ^ { \prime } }$ denotes the latent feature representation.

Physics-aware positional encoding Because sparse-view reconstruction is strongly constrained by acquisition geometry, we introduced a physics-aware positional encoding strategy to preserve angular relationships among projection views. Instead of using a simple normalized angle index, each acquisition angle θ is mapped to the unit circle as

$$
v ( \theta ) = ( \sin \theta , \cos \theta ) ,\tag{10}
$$

which explicitly preserves the circular geometry of rotational acquisition and more faithfully represents the relative relationship between projection angles.

To improve the representation of fine geometric variation, we further applied Fourier feature mapping to the positional representation. This design was motivated by the spectral bias of neural networks, which tend to learn low-frequency functions more easily than high-frequency ones.[33] For an input coordinate $p ,$ the positional encoding is defined as

$$
\gamma ( p ) = \left( \sin ( 2 ^ { 0 } \pi p ) , \cos ( 2 ^ { 0 } \pi p ) , \ldots , \sin ( 2 ^ { L - 1 } \pi p ) , \cos ( 2 ^ { L - 1 } \pi p ) \right) ,\tag{11}
$$

where in our implementation, we set $L = 1 0$ for spatial coordinates and $L = 4$ for angular coordinates to balance representation capacity and computational cost.

Temporal consistency encoding A key property of 4D CBCT is that anatomical structures evolve continuously across respiratory phases rather than changing independently from one phase to another. To exploit this temporal continuity, we introduced a temporal encoding mechanism in the latent feature space. After Swin-UNet encoding, the feature maps are reshaped into sequential embeddings and arranged in respiratory phase order. These phase-wise latent features are then processed by a long short-term memory (LSTM) module to model inter-phase dependencies and breathing dynamics.

Compared with directly applying recurrent operations to raw high-resolution projections, temporal modeling in the latent feature space is more eficient and more stable. It allows the network to capture phase-specific motion patterns while preserving compact semantic information. Let the temporal embedding be denoted by $H _ { t }$ The spatial feature, positional encoding, and temporal feature are fused as

$$
X = X _ { s } + \mathrm { P E } + H _ { t } ,\tag{12}
$$

where PE is the positional encoding term. The fused feature $X$ is then passed through the decoder to generate dense-view projections:

$$
{ \hat { \mathcal { P } } } _ { t } = { \mathrm { D e c o d e r } } ( X ) ,\tag{13}
$$

After projection generation, FDK reconstruction is applied to obtain the initial volumetric image for each phase:

$$
I _ { t } ^ { ( 0 ) } = \mathrm { F D K } ( \hat { \mathcal { P } } _ { t } ) , \qquad t = 1 , \ldots , P .\tag{14}
$$

## 4.3 Image domain correction module

Although projection completion substantially improves sparse-view reconstruction, residual artifacts and local distortions may still remain after FDK reconstruction. We therefore introduced an image-domain correction stage to further refine the reconstructed volumes.

The IDC M takes the initial reconstructed image

$$
I _ { t } ^ { ( 0 ) } \in \mathbb { R } ^ { c \times h \times w }
$$

as input and produces the corrected image $I _ { t }$ . We implemented this module using an AttU-Net-based architecture. During training, the image-domain refinement is supervised by the corresponding reference image $\ddot { I } _ { t } ^ { g t }$ , with the optimization target

$$
\underset { ( \hat { I } _ { t } , I _ { t } ^ { g t } ) \in \mathcal { D } _ { 1 } } { \arg \operatorname* { m i n } } \ \left\| I _ { t } ^ { g t } - I _ { t } \right\| _ { 2 } ^ { 2 } .\tag{15}
$$

where $\mathcal { D } _ { 1 }$ denotes the image-domain training dataset.

The corrected output is defined as

$$
I _ { t } = \mathrm { I D C } \mathrm { M } ( I _ { t } ^ { ( 0 ) } ) .\tag{16}
$$

## 4.4 Loss Function

Projection-domain loss: The PDGM was trained using a composite loss that combines mean squared error (MSE), structural similarity (SSIM) loss, and adjacent-slice gradient consistency loss.

The MSE term is defined as

$$
L _ { \mathrm { M S E } } = \frac { 1 } { N _ { p } } \sum _ { i = 1 } ^ { N _ { p } } \left( \hat { p } _ { g t } ( i ) - \hat { p } _ { t } ( i ) \right) ^ { 2 } ,\tag{17}
$$

where $N _ { p }$ denotes the total number of pixels in the projection set.

To preserve structural similarity between generated and reference projections, we used an SSIM-based loss

$$
L _ { \mathrm { S S I M } } = 1 - \mathrm { S S I M } ( \hat { p } _ { t } , \hat { p } _ { g t } ) .\tag{18}
$$

To encourage local continuity and reduce structural inconsistency between adjacent slices, we further introduced a gradient consistency loss

$$
L _ { S } = \frac { 1 } { N _ { p } } \sum _ { i = 1 } ^ { N _ { p } } \| \nabla \hat { p } _ { t } ( i , z ) - \nabla \hat { p } _ { t } ( i , z + 1 ) \| _ { 2 } ^ { 2 } ,\tag{19}
$$

where $\hat { p } _ { t } ( i , z )$ and $\hat { p } _ { t } ( i , z + 1 )$ denote adjacent slices along the depth direction, and $\nabla$ represents the spatial gradient operator.

The final PDGM training objective is

$$
L _ { \mathrm { P D } } = \alpha L _ { \mathrm { M S E } } + \beta L _ { \mathrm { S S I M } } + \lambda L _ { S } ,\tag{20}
$$

with empirically set constants $\alpha , \beta , \lambda .$

Image-domain loss: The IDC M is trained using a combined image-domain loss composed of MSE and SSIM terms. The MSE component is

$$
L _ { \mathrm { M S E } } ^ { \prime } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \left( I _ { \mathrm { g t } } ( j ) - I ( j ) \right) ^ { 2 } ,\tag{21}
$$

where M denotes the number of image pixels or voxels.

The image-domain SSIM loss is

$$
L _ { \mathrm { S S I M } } ^ { \prime } = 1 - \mathrm { S S I M } ( I , I _ { \mathrm { g t } } ) .\tag{22}
$$

The final image-domain training objective is

$$
\begin{array} { r } { L _ { \mathrm { t o t a l } } ^ { \prime } = \delta L _ { \mathrm { M S E } } ^ { \prime } + \epsilon L _ { \mathrm { S S I M } } ^ { \prime } , } \end{array}\tag{23}
$$

and in all experiments we set $\delta = 0 . 4$ and $\epsilon = 0 . 6$

## 4.5 Reconstruction procedure

For each respiratory phase, STCNet reconstructs the final image in three steps. First, the sparse-view projection subset $S _ { t }$ and its known acquisition angles $\theta _ { t }$ are used as input to the PDGM. Second, the PDGM generates a dense-view projection set $\hat { \mathcal { P } } _ { t } ^ { \phantom { \dagger } } .$ which is reconstructed by FDK to obtain the initial image volume $\dot { I _ { t } ^ { ( 0 ) } }$ . Third, the IDCM refines $I _ { t } ^ { ( 0 ) }$ to produce the final corrected image $I _ { t }$ . Repeating this process for all 10 respiratory phases yields the full 4D CBCT sequence.

## 5 Declarations

## Data Availability

The TCIA-LUNG dataset is publicly available through The Cancer Imaging Archive [29]. The associated NSCLC 4D lung imaging data used in this study are described in [30]. The CIRS phantom data are available from the corresponding author upon reasonable request and subject to institutional policy.

## Code availability

Code has publicly available at: https://github.com/Lai-KK/STCNet.git. Author contribution

QI LAI: Conceptualization, Methodology, Formal analysis, Writing-Original draft; YUTONG HE: Methodology, Validation, Experiments; ZHICHENG LI: Data curation, Investigation, Resources; LEI WANG: Writing Review,Funding acquisition; YAN YAN: Software,Validation,Data curation; YANZHONG MA: Investigation, Data curation; XIAOKUN LIANG: Writing Review, Funding acquisition; DONG LIANG: Data curation, Investigation, Resources; LEI JIN: Writing Review, Funding acquisition. Conflict of interest/Competing interests

The authors declare that they have no conflicts of interest or competing interests related to this work.

## Acknowledgments

This work was supported by the National Key Research and Development Program of China (2024YFF1206901); in part by the Shenzhen General Program under Grant Ref. JCYJ20240813154703005; in part by the National Natural Science Fund for Young Scientists Fund Program C (82502447).

## References

[1] Franzone, P., Fiorentino, A., Barra, S., Cante, D., Masini, L., Cazzulo, E., Todisco, L., Gabriele, P., Garibaldi, E., Merlotti, A., et al.: Image-guided radiation therapy (igrt): practical recommendations of italian association of radiation oncology (airo). La radiologia medica 121, 958–965 (2016)

[2] Cuijpers, J.P., Dahele, M., Jonker, M., Kraan, B., Senan, S., Slotman, B., Verbakel, W.F.: Analysis of components of variance determining probability of setup errors in cbct-guided stereotactic radiotherapy of lung tumors. Medical physics 44(2), 382–388 (2017)

[3] Caillet, V., Booth, J.T., Keall, P.: Igrt and motion management during lung sbrt delivery. Physica Medica 44, 113–122 (2017)

[4] Johnson-Hart, C.N., Price, G.J., Faivre-Finn, C., Aznar, M.C., Herk, M.: Residual setup errors towards the heart after image guidance linked with poorer survival in lung cancer patients: do we need stricter igrt protocols? International Journal of Radiation Oncology\* Biology\* Physics 102(2), 434–442 (2018)

[5] Roman, N.O., Shepherd, W., Mukhopadhyay, N., Hugo, G.D., Weiss, E.: Interfractional positional variability of fiducial markers and primary tumors in locally advanced non-small-cell lung cancer during audiovisual biofeedback radiotherapy. International Journal of Radiation Oncology\* Biology\* Physics 83(5), 1566–1572 (2012)

[6] Balik, S., Weiss, E., Jan, N., Roman, N., Sleeman, W.C., Fatyga, M., Christensen, G.E., Zhang, C., Murphy, M.J., Lu, J., et al.: Evaluation of 4-dimensional computed tomography to 4-dimensional cone-beam computed tomography deformable image registration for lung cancer adaptive radiation therapy. International Journal of Radiation Oncology\* Biology\* Physics 86(2), 372–379 (2013)

[7] Ou, Z., Xie, J., Teng, Z., Wang, X., Jin, P., Du, J., Ding, M., Li, H., Chen, Y., Niu, T.: Pnmc: four-dimensional conebeam ct reconstruction combining prior network and motion compensation. Computers in Biology and Medicine 171, 108145 (2024)

[8] Qi, Z., Chen, G.-H.: Extraction of tumor motion trajectories using piccs-4dcbct: a validation study. Medical physics 38(10), 5530–5538 (2011)

[9] Zhi, S., Kachelrieß, M., Pan, F., Mou, X.: Cycn-net: A convolutional neural network specialized for 4d cbct images refinement. IEEE Transactions on Medical Imaging 40(11), 3054–3064 (2021)

[10] Brehm, M., Paysan, P., Oelhafen, M., Kachelrieß, M.: Artifact-resistant motion estimation with a patient-specific artifact model for motion-compensated conebeam ct. Medical physics 40(10), 101913 (2013)

[11] Wang, J., Gu, X.: Simultaneous motion estimation and image reconstruction (smeir) for 4d cone-beam ct. Medical physics 40(10), 101912 (2013)

[12] Jia, X., Tian, Z., Lou, Y., Sonke, J.-J., Jiang, S.B.: Four-dimensional cone beam ct reconstruction and enhancement using a temporal nonlocal means method. Medical physics 39(9), 5592–5602 (2012)

[13] Zhang, H., Ma, J., Bian, Z., Zeng, D., Feng, Q., Chen, W.: High quality 4d conebeam ct reconstruction using motion-compensated total variation regularization. Physics in Medicine & Biology 62(8), 3313 (2017)

[14] Mascolo-Fortin, J., Matenine, D., Archambault, L., Despr´es, P.: A fast 4d cone beam ct reconstruction method based on the osc-tv algorithm. Journal of X-ray Science and Technology 26(2), 189–208 (2018)

[15] Yang, P., Ge, X., Tsui, T., Liang, X., Xie, Y., Hu, Z., Niu, T.: Four-dimensional cone beam ct imaging using a single routine scan via deep learning. IEEE Transactions on Medical Imaging 42(5), 1495–1508 (2022)

[16] Chan, Y., Liu, X., Wang, T., Dai, J., Xie, Y., Liang, X.: An attention-based deep convolutional neural network for ultra-sparse-view ct reconstruction. Computers in Biology and Medicine 161, 106888 (2023)

[17] Zhang, C., Liu, L., Dai, J., Liu, X., He, W., Chan, Y., Xie, Y., Chi, F., Liang, X.: Xtransct: ultra-fast volumetric ct reconstruction using two orthogonal x-ray projections for image-guided radiation therapy via a transformer network. Physics in Medicine & Biology 69(8), 085010 (2024)

[18] Dong, G., Dai, J., Li, N., Zhang, C., He, W., Liu, L., Chan, Y., Li, Y., Xie, Y., Liang, X.: 2d/3d non-rigid image registration via two orthogonal x-ray projection images for lung tumor tracking. Bioengineering 10(2), 144 (2023)

[19] Li, Y., Li, K., Zhang, C., Montoya, J., Chen, G.-H.: Learning to reconstruct computed tomography images directly from sinogram data under a variety of data acquisition conditions. IEEE transactions on medical imaging 38(10), 2469–2481 (2019)

[20] Lee, H., Lee, J., Kim, H., Cho, B., Cho, S.: Deep-neural-network-based sinogram

synthesis for sparse-view ct image reconstruction. IEEE Transactions on Radiation and Plasma Medical Sciences 3(2), 109–119 (2019) https://doi.org/10.1109/ TRPMS.2018.2867611

[21] Zhang, Z., Liang, X., Dong, X., Xie, Y., Cao, G.: A sparse-view ct reconstruction method based on combination of densenet and deconvolution. IEEE transactions on medical imaging 37(6), 1407–1417 (2018)

[22] Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., Wang, M.: Swinunet: Unet-like pure transformer for medical image segmentation. In: European Conference on Computer Vision, pp. 205–218 (2022). Springer

[23] Xie, S., Zheng, X., Chen, Y., Xie, L., Liu, J., Zhang, Y., Yan, J., Zhu, H., Hu, Y.: Artifact removal using improved googlenet for sparse-view ct reconstruction. Scientific reports 8(1), 6700 (2018)

[24] Han, Y., Ye, J.C.: Framing u-net via deep convolutional framelets: Application to sparse-view ct. IEEE transactions on medical imaging 37(6), 1418–1429 (2018)

[25] Zhang, C., Xie, Y., Liang, X.: Xgenrecon: A new perspective in ultra-sparse volumetric cbct reconstruction through geometry-controlled x-ray projection generation. IEEE Transactions on Radiation and Plasma Medical Sciences (2024)

[26] Xia, W., Cong, W., Wang, G.: Patch-based denoising difusion probabilistic model for sparse-view ct reconstruction. arXiv preprint arXiv:2211.10388 (2022)

[27] Liu, X., Liang, X., Deng, L., Tan, S., Xie, Y.: Learning low-dose ct degradation from unpaired data with flow-based model. Medical Physics 49(12), 7516–7530 (2022)

[28] Heylen, R., Schramm, G., Suetens, P., Nuyts, J.: 4d cbct reconstruction with tv regularization on a dynamic software phantom. In: 2019 IEEE Nuclear Science Symposium and Medical Imaging Conference (NSS/MIC), pp. 1–3 (2019). Ieee

[29] Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Mafitt, D., Pringle, M., et al.: The cancer imaging archive (tcia): maintaining and operating a public information repository. Journal of digital imaging 26, 1045–1057 (2013)

[30] Hugo, G.D., Weiss, E., Sleeman, W.C., Balik, S., Keall, P.J., Lu, J., Williamson, J.F.: Data from 4d lung imaging of nsclc patients. (No Title) (2016)

[31] Andersen, A.H., Kak, A.C.: Simultaneous algebraic reconstruction technique (sart): a superior implementation of the art algorithm. Ultrasonic imaging 6(1), 81–94 (1984)

[32] Chai, T., Draxler, R.R., et al.: Root mean square error (rmse) or mean absolute

error (mae). Geoscientific model development discussions 7(1), 1525–1534 (2014)

[33] Poobathy, D., Chezian, R.M.: Edge detection operators: Peak signal to noise ratio based comparison. IJ Image, Graphics and Signal Processing 10, 55–61 (2014)

[34] Chen, M.-J., Bovik, A.C.: Fast structural similarity index algorithm. Journal of Real-Time Image Processing 6, 281–287 (2011)

[35] Feldkamp, L.A., Davis, L.C., Kress, J.W.: Practical cone-beam algorithm. Josa a 1(6), 612–619 (1984)