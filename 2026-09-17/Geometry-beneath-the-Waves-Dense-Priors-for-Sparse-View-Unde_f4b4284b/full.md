# Geometry beneath the Waves: Dense Priors for Sparse-View Underwater 3D Gaussian Splating

Harvey Caldeira University of Bristol Bristol, United Kingdom jv21451@bristol.ac.uk

Shaoyu Cai National University of Singapore Singapore, Singapore shaoyucai@nus.edu.sg

Haoran Wang University of Bristol Bristol, United Kingdom yp22378@bristol.ac.uk

Rachel Fu   
Oceanx   
New York, USA   
rachel.fu@oceanx.org   
Guoxi Huang   
University of Bristol   
Bristol, United Kingdom   
guoxi.huang@bristol.ac.uk   
Nantheera Anantrasirichai University of Bristol Bristol, United Kingdom   
N.Anantrasirichai@bristol.ac.uk

![](images/b91f30a12074e9a12b74e5b91f88418f5ea4e4ddec2cbe8560505fcc119ed33f.jpg)  
Figure 1: Qualitative comparison between VGGT (top) and our method (bottom) on the Curasao scene. From left to right: rendered depth, reconstructed point cloud, and RGB rendering.

## 1 Introduction

Underwater 3D reconstruction supports applications ranging from marine ecosystem monitoring and subsea inspection to underwater archaeology, education, and immersive visualisation. 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] has made real-time photore alistic novel-view rendering practical, while underwater variants such as UW-GS [Wang et al. 2025a] and RUSplatting [Jiang et al. 2025] incorporate physically based image-formation models to separate medium efects from scene radiance. Their reconstruction quality, however, remains fundamentally limited by the geometry used for initialisation. Water directly undermines this geometry. Wavelength-dependent absorption, backscatter, and suspended particulates reduce contrast and introduce high-frequency artefacts, degrading the feature correspondences required by classical Structure-from-Motion pipelines such as COLMAP [Schonberger and Frahm 2016]. These failures become particularly severe under turbid conditions and sparse camera trajectories, where insuficient overlap further weakens geometric estimation.

Feed-forward geometry foundation models such as VGGT [Wang et al. 2025b] ofer an appealing alternative by regressing camera poses, dense depth, and point maps from unposed images in a single pass, without explicit feature matching. Applied zero-shot underwater, however, they exhibit two characteristic failure modes. First, volumetric water-column noise arises when scattering and suspended particles are interpreted as solid geometry, filling open water with spurious point clusters. Second, high-frequency surface dispersion causes predicted surfaces to fragment or jitter instead of converging to coherent boundaries. Although 3DGS initialised from these predictions often recovers the global scene structure, it also inherits both forms of geometric noise.

View sparsity introduces a second, complementary bottleneck. RUSplatting expands its feature tracks using temporal frame interpolation, but the resulting images are synthesised purely in 2D and are not constrained by the underlying scene geometry. When processed by a feed-forward reconstruction backbone, these photometrically and geometrically inconsistent frames can produce severe spatial artefacts. We address both limitations by adapting feed-forward geometry to the underwater domain and replacing image-space interpolation with geometry-guided view synthesis.

## 2 Our approach

## 2.1 Domain-Adapted Feed-forward Geometry

Reliable underwater geometric supervision is scarce: paired data are limited, while pseudo-labels from real images inherit the same scattering, absorption, and backscatter artefacts the model must overcome. Post-hoc alignment methods, including depth inversion, foreground scaling, histogram matching, and afine fitting, also fail to generalise across water types because the depth signals exhibit non-linear volumetric distortions that no global transformation can reconcile. We therefore reverse the supervision strategy: instead of adapting noisy depth targets, we preserve clean geometry and degrade the corresponding images. Our training set contains 22,000 images, comprising 1,000 Hypersim [Roberts et al. 2021] scenes, 1,000 TartanAir [Wang et al. 2020] images, and 20,000 unposed SA-1B [Kirillov et al. 2023] images. We use metric groundtruth depth where available and Depth Anything 3 [Lin et al. 2026] pseudo-depth for SA-1B. The RGB images are then transformed using SyreaNet’s [Wen et al. 2023] physically guided model of wavelength-dependent absorption, scattering, and backscatter. By independently sampling attenuation coeficients, backscatter fractions, and transmission parameters for each frame, we expose the encoder to a broad range of turbidity levels and colour casts while retaining geometrically consistent supervision.

We adapt VGGT using the Fin3R [Ren et al. 2025] teacher–student framework. A frozen teacher processes the clean image, while the student receives its synthetically degraded underwater counterpart. Training minimises $\mathcal { L } = \| y _ { s } - y _ { t } \| ^ { 2 }$ encouraging the student to reproduce the teacher’s geometric predictions despite underwater appearance distortions. To preserve VGGT’s pretrained reconstruction capability, we insert LoRA [Hu et al. 2022] adapters only into its visual aggregator encoder, while keeping all downstream prediction heads frozen. For each pretrained weight matrix W, LoRA learns a low-rank update W<sup>′</sup> = W + AB, with trainable matrices A and B. The student therefore learns to encode degraded underwater imagery into the feature space expected by the original model, while preserving its pretrained geometric priors and metric-scale predictions.

## 2.2 Geometry-Guided Radiance Field Rendering

Rather than densifying feature tracks through image-space interpolation, we generate additional observations from explicit 3D geometry (Fig. 2). The fine-tuned VGGT first predicts a dense, structurally coherent point cloud from the unposed input views. This geometry initialises an intermediate 3DGS proxy, from which we render pseudo-views along the estimated camera trajectory. Each rendered view is paired with a pseudo depth map.

We then run COLMAP jointly over the captured and rendered views, increasing camera overlap and extending feature tracks across wider baselines. The resulting track graph and depth priors initialise RUSplatting, which performs the final radiance optimisation and underwater colour restoration. Unlike 2D flow-based interpolation, each pseudo-view is obtained by projecting an explicit 3D representation. The additional correspondences are therefore anchored to the estimated scene structure, providing dense geometric constraints for the final reconstruction.

![](images/fa20c70dc6f3f1b2623b414e423a4b2bfbf46f4e3d8235e28a3a1a75f4cf6bb1.jpg)  
Figure 2: Pipeline overview. LoRA-adapted VGGT predicts dense geometry, a 3DGS proxy renders geometry-guided pseudo-views, COLMAP expands tracks, and RUSplatting performs final optimisation.

Table 1: Performance comparison. Red, orange, and yellow indicate the best, second-, and third-best results, respectively.
<table><tr><td rowspan="2">Initialisation → Renderer</td><td rowspan="2">Iter.</td><td colspan="3">SeaThru-NeRF</td><td colspan="3">Submerged3D</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>COLMAP → 3DGS</td><td>30k</td><td>25.68</td><td>0.7735</td><td>0.1965</td><td>23.48</td><td>0.7385</td><td>0.3333</td></tr><tr><td>VGGT → 3DGS</td><td>30k</td><td>20.62</td><td>0.5380</td><td>0.3066</td><td>19.38</td><td>0.5001</td><td>0.4915</td></tr><tr><td>FT-VGGT → 3DGS</td><td>30k</td><td>22.01</td><td>0.6240</td><td>0.2726</td><td>19.58</td><td>0.4998</td><td>0.4568</td></tr><tr><td>COLMAP → RUSplatting</td><td>15k</td><td>24.37</td><td>0.7611</td><td>0.3103</td><td>22.97</td><td>0.7114</td><td>0.3514</td></tr><tr><td>VGGT → RUSplatting</td><td>15k</td><td>19.76</td><td>0.4824</td><td>0.4463</td><td>20.41</td><td>0.5177</td><td>0.4403</td></tr><tr><td>FT-VGGT → RUSplatting</td><td>15k</td><td>22.41</td><td>0.6165</td><td>0.3567</td><td>19.34</td><td>0.4857</td><td>0.4597</td></tr><tr><td>Ours (geometry-guided)</td><td>15k</td><td>27.11</td><td>0.8634</td><td>0.2188</td><td>23.49</td><td>0.7045</td><td>0.3322</td></tr></table>

## 3 Experiments and Findings

We fine-tune the LoRA adapters for 20 epochs using mixed precision on eight NVIDIA GH200 superchips, with a per-GPU batch size of 18 and four-step gradient accumulation, yielding an efective batch size of 576. The validation loss on 2,200 held-out synthetic images decreases steadily before plateauing, indicating stable convergence without evident overfitting. We evaluate on SeaThru-NeRF [Levy et al. 2023] and Submerged3D [Jiang et al. 2025], each containing four scenes with approximately 20 views. Images are processed at 720p, with every eighth frame held out. Following the original settings, we use 30k iterations for 3DGS and 15k for RUSplatting, render one pseudo-view between consecutive frames, and retain up to 10<sup>6</sup> VGGT points after applying a 30% confidence threshold.

Table 1 reports dataset averages. On the SeaThru-NeRF scenes, our pipeline improves COLMAP-initialised RUSplatting from 24.37 to 27.11 dB PSNR, with gains of 0.102 SSIM and 0.092 LPIPS. It also improves by up to 7.35 dB over raw feed-forward initialisation. On Submerged3D, our method achieves the best average PSNR of 23.49,dB and LPIPS of 0.3322, although gains are smaller due to greater lighting and colour variation. VGGT adaptation also improves the geometric prior. On SeaThru-NeRF, PSNR rises from

20.62 to 22.01 dB with 3DGS and from 19.76 to 22.41 dB with RUSplatting, while LPIPS decreases from 0.4463 to 0.3567. Qualitatively, adaptation suppresses floating water-column points and recovers more coherent depth as shown in Fig. 1.

Two limitations remain: i) COLMAP fails on IUI3-RedSea dataset because scattering artefacts disrupt SIFT matching; ii) low-amplitude prediction noise persists after density control under sparse capture. These limitations motivate future work on geometry-aware pruning, semantic water-column masking, and iterative view synthesis to investigate whether refinement yields cumulative gains.

## References

Edward J. Hu, Yelong Shen, Phillip Wallis, et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Zhuodong Jiang, Haoran Wang, Guoxi Huang, et al. 2025. RUSplatting: Robust 3D Gaussian Splatting for Sparse-View Underwater Scene Reconstruction. In BMVC.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, et al. 2023. 3D gaussian splatting for real-time radiance field rendering. ACM TOG 42, 4 (2023), 139–1.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, et al. 2023. Segment Anything. In IEEE/CVF International Conference on Computer Vision ICCV. 3992–4003.

Deborah Levy, Amit Peleg, Naama Pearl, et al. 2023. SeaThru-NeRF: Neural Radiance Fields in Scattering Media. In CVPR. 56–65.

Haotong Lin, Sili Chen, Jun Hao Liew, et al. 2026. Depth Anything 3: Recovering the Visual Space from Any Views. In ICLR.

Weining Ren, Hongjun Wang, Xiao Tan, et al. 2025. Fin3R: Fine-Tuning Feed-Forward 3D Reconstruction Models via Monocular Knowledge Distillation. In NeurIPS.

Mike Roberts, Jason Ramapuram, Anurag Ranjan, et al. 2021. Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding. In ICCV.

Johannes L Schonberger andJan-Michael Frahm. 2016. Structure-from-motion revisited. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4104–4113.

Haoran Wang, N. Anantrasirichai, Fan Zhang, et al. 2025a. UW-GS: Distractor-aware 3D gaussian splatting for enhanced underwater scene reconstruction. In WACV.

Jianyuan Wang, Minghao Chen, Nikita Karaev, et al. 2025b. VGGT: Visual Geometry Grounded Transformer. In CVPR.

Wenshan Wang, Delong Zhu, Xiangwei Wang, et al. 2020. TartanAir: A Dataset to Push the Limits of Visual SLAM. In IROS. 4909–4916.

Junjie Wen, J. Cui, Z. Zhao, et al. 2023. SyreaNet: A Physically Guided Underwater Image Enhancement Framework Integrating Synthetic and Real Images. In ICRA.