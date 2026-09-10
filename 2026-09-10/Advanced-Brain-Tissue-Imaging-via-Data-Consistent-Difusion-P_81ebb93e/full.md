# Advanced Brain Tissue Imaging via Data-Consistent Difusion Priors in Laminographic X-Ray Nanoimaging

Wenxuan Fang<sup>1,2</sup>, Abraham L. Levitan<sup>2</sup>, Ana Diaz<sup>2</sup>, Carles Bosch<sup>3</sup>, Adrian Wanner<sup>2</sup>, Andreas T. Schaefer<sup>3</sup>, Mirko Holler<sup>2</sup>, Tomas Aidukas<sup>2</sup>, Nicholas W. Phillips<sup>2,5</sup>, Yuxin Zhang<sup>3</sup>, Alexandra Pacureanu<sup>4</sup>, Manuel Guizar-Sicairos<sup>1,2\*</sup>, Luis Barba<sup>2\*</sup>

<sup>\*</sup>Ecole Polytechnique F´ed´erale de Lausanne, Lausanne, Switzerland. <sup>´</sup> <sup>2</sup>Paul Scherrer Institute, Villigen, Switzerland. <sup>3</sup>Sensory Circuits and Neurotechnology Lab, The Francis Crick Institute, 1 Midland Road, London, NW1 1AT, UK. <sup>4</sup>ESRF, The European Synchrotron, Grenoble, France. <sup>5</sup>Mineral Resource CSIRO, Clayton, Victoria, Australia.

\*Corresponding author(s). E-mail(s): manuel.guizar-sicairos@psi.ch; luis.barba-flores@psi.ch; Contributing authors: wenxuan.fang@epfl.ch;

## Abstract

Nanoscale imaging of mammalian brains is critical for connectomics. X-ray laminography enables high-throughput imaging of extended, plate-like biological specimens. However, the tilted acquisition geometry leads to incomplete Fourier-space coverage, giving rise to a missing-cone of information. Conventional reconstruction methods cannot recover unmeasured information within the cone, resulting in artifacts that distort fine brain structures. While resolving these requires modeling 3D structure, direct 3D deep learning approaches are limited by data scarcity and computational cost. Here we introduce LUCID (Laminography with Unified Consistent Difusion), a framework that combines multi-view difusion priors with projection-domain data consistency. LUCID integrates complementary 3D structural information while enforcing strict alignment with the laminography forward model. On simulated datasets, LUCID substantially improves spatial fidelity and restores missing Fourier components, outperforming

baseline methods. Applied to experimental laminography data, LUCID generalizes robustly despite being trained exclusively on fully sampled tomographic volumes, and efectively recovers unmeasured Fourier information.

Keywords: Difusion Prior, Laminography, X-ray, Brain imaging, Ptychography, Nanoscale, Connectomics

## 1 Introduction

The nanoscale architecture of the brain’s connectome, a comprehensive map of neural circuitry, holds the key to deciphering the structural underpinnings of cognition, behavior, and neurological disorders [1, 2]. To recover the connectome, the synapic connections between neurons must be resolved, which requires resolutions on the order of 10–20 nm. At this resolution, synaptic features become resolvable, enabling the iden tification of synaptic interfaces and providing critical information about the wiring principles of the central nervous system. However, achieving three-dimensional (3D) imaging at this resolution across millimeter-scale tissue volumes remains a formidable challenge. While electron microscopy (EM)-based techniques, such as serial sectioning and focused ion beam/scanning electron microscopy (FIB/SEM), provide a resolution suficient to reliably identify synapses [3, 4], their reliance on thin-sectioning or ablation introduces artifacts and limits scalability. X-ray imaging provides a nondestructive alternative, and recent advances in synchrotron-based ptychography have demonstrated nanometric resolution in a wide variety of specimens of tens of microns in diameter [5–9]. While nanoscale X-ray imaging can resolve synaptic structures [10, 11], conventional X-ray computed tomography (CT) relies on rotation around an axis perpendicular to the X-ray beam. Applying this geometry to extended brain tissue would require cutting cuboid-like samples to allow full angular coverage. However, partitioning large brain volumes into such geometries without disrupting or losing delicate neural structures is generally not feasible. In contrast, sectioning brain tissue into thick slices is experimentally practical and preserves structural integrity over large fields of view.

In contrast, sectioning brain tissue into thick slices is experimentally practical and preserves structural integrity over extended fields of view. X-ray laminography [12], with its tilted imaging geometry, is naturally suited to such plate-like specimens and avoids the need for cylindrical sample preparation. Although the preparation strategy resembles serial sectioning in electron microscopy, X-rays can penetrate substantially thicker tissue sections, enabling higher-throughput volumetric imaging of neural tissue.

Computed laminography (CL) employs a tilted rotation axis forming an angle smaller than 90<sup>◦</sup> with respect to the X-ray beam. Compared with limited-angle tomography, laminography provides more uniform efective thickness across projection angles and reduces the extent of missing information in Fourier space. In limited-angle tomography, incomplete angular coverage gives rise to a missing wedge in reciprocal space, leading to severe directional artefacts and anisotropic resolution degradation [13, 14]. By contrast, the tilted acquisition geometry of laminography distributes the missing information into a biconical region rather than a wedge, resulting in more balanced sampling and improved structural preservation across views. These properties make laminography particularly well suited for high-resolution 3D imaging of thick neural sections.

Despite the aforementioned advantages, the imaging geometry of laminography gives rise to an ill-posed reconstruction problem. As illustrated in Fig. 1a, the rotation axis is tilted with respect to the incident X-ray beam, which leads to an incomplete coverage of the 3D frequency domain, leaving a characteristic biconical region unsampled along the rotation axis [7, 12] as shown in Fig. 1b. This missing-cone geometry induces a highly anisotropic transfer of information, as further visualized in a 2D Fourier slice in Fig. 1c, where entire regions of Fourier space are absent. This geometry results in limited resolution along the axis normal to the sample plane [15], manifesting as a pronounced axial elongation, interlayer aliasing, and directional blurring. Fine neural features are therefore smeared across adjacent layers, becoming so distorted as to be no longer visible. Addressing this loss of information along the vertical axis is essential for achieving faithful three-dimensional reconstruction in laminography. Fig. 1 provides a conceptual schematic of the laminography acquisition geometry. It is intended to illustrate the principle of laminography and does not represent the specific experimental setup, where synaptic-resolution data were acquired using ptychographic X-ray laminography.

CL reconstruction is commonly performed using a modified or extended version of the Filtered Back Projection (FBP) algorithm [16]. Unfortunately, because FBP provides no means to compensate for the missing cone, it is unable to correct for missing-cone related artifacts. Therefore, most existing eforts to correct for such artifacts have focused on iterative reconstruction strategies. These approaches can partially mitigate this issue by roughly estimating the missing information. They do this by minimizing a data error metric using algebraic [17] or regularized iterative solvers [18]. To further incorporate prior knowledge, anisotropic and edge-preserving regularization schemes [19] have been introduced to suppress inter-slice aliasing and reduce noise. Although efective in controlled settings, such methods require careful tuning of regularization terms and hyperparameters, and often lead to oversmoothing of fine structural details.

A closely related problem in tomography is the missing-wedge artifacts encountered in limited-angle acquisition geometries, particularly in cryo-electron tomography (cryo-ET). To mitigate these efects, several learning-based reconstruction approaches have been proposed, including GAN-based frameworks [20] operating jointly in sinogram and tomographic domains, as well as more recent iterative and self-supervised methods such as IsoNet [21] and DeepDeWedge [22]. These approaches demonstrated the potential of learned priors to recover missing structural information. Similarly, deep learning–based methods have recently been explored for CL reconstruction and mitigation of missing cone artifacts. Some propose post-processing networks applied after reconstruction [23]. These approaches remain decoupled from the reconstruction and do not enforce data consistency, as the denoised outputs are not re-constrained by the measured projections. Consequently, they primarily act as artifact-correction modules rather than generative models capable of recovering genuinely missing information.

![](images/f835aacd56bd97d098710187cfbf13bc2817bcd0924264928be07de140f75824.jpg)

(a) Laminography Setup  
![](images/33d5cf8f4927272cc3f5f42398eb131e63a4deee41dd0ec2a976bdfce3a4ef01.jpg)  
(b) Fourier Space and Missing Cone  
(c) Fourier Space 2D Slice  
Fig. 1: Laminography acquisition geometry and the missing-cone problem. (a) Schematic of X-ray laminography with a plane-wave beam. A parallel X-ray beam illuminates a tilted specimen, which rotates around an axis inclined by an angle θ with respect to the beam. The transmitted intensity is recorded by a pixelated detector. (b) Corresponding sampling pattern in 3D Fourier space (k-space). The tilted rotation geometry leads to incomplete coverage, leaving a biconical region (missing cone) unsampled along the rotation axis. (c) Representative 2D Fourier slice illustrating the missing-cone region (dark wedge), which results in anisotropic frequency loss. This incomplete sampling fundamentally limits reconstruction fidelity and induces artifacts. It should be noted that (a) is provided as a schematic to explain the concept of laminography, but the experimental data shown in this paper was not acquired with a parallel beam geometry but rather using ptychographic X-ray laminography [7].

The approach in [24] introduced a generative network that relies on an unattainable dual-scale CT-CL fusion strategy to construct artifact-free ground truth (GT) labels for training. Acquiring such perfectly aligned, high-fidelity fusion targets is physically intractable for intact biological tissues.

The missing-cone problem in laminography is intrinsically 3D, and in principle requires processing the full volumetric data to recover lost information. In contrast to conventional tomography, due to the tilted geometry of the rotation axis, the laminography reconstructions and therefore also the recovery of missing-cone information cannot be split into 2D slices. However, direct 3D learning-based reconstruction faces substantial practical barriers. Volumetric data is inherently scarce, and operating on high-resolution 3D volumes demands prohibitively large computational resources. Both training and inference that operate on complete volumes become slow or simply infeasible, making fully 3D approaches dificult to deploy in realistic imaging workflows. Existing methods for laminography that resort to independent slice-by-slice 2D reconstructions [23, 24] are inadequate, because they inevitably sacrifice volumetric consistency and fail to capture structural dependencies across the volume.

The existing methods mentioned above, iterative, regularized, or learning-based, have been developed and validated primarily in industrial settings that involve structures with limited diversity and heterogeneity, such as printed circuit boards or integrated circuits [17, 18, 23–27]. In contrast, brain tissue exhibits extreme structural heterogeneity, with diverse cell types, complex neuropil textures, and densely interconnected synaptic networks. These characteristics introduce unique reconstruction challenges, making it especially hard to exploit the information which does exist about their prior distribution.

Difusion models [28–31] have shown remarkable capability in image restoration and reconstruction, suggesting strong potential for addressing the missing cone problem. Yet their implementation in laminography reconstruction, as well as their application to a challenging field like 3D brain imaging remain largely unexplored.

Here, we propose LUCID (Laminography with Unified Consistent Difusion), a physics-guided generative framework that unifies the strengths of data-driven difusion priors and measurement-consistent optimization. Rather than operating independently on individual slices or relying on post processing, LUCID employs a multi-axial difusion strategy across axial, sagittal, and coronal planes, enabling complementary structural learning. Crucially, the difusion model is embedded directly within the iterative laminography reconstruction loop, allowing the generative prior to evolve jointly with projection-domain data-consistency updates. This tight coupling reconciles learned anatomical priors with the governing physics of the acquisition process, steering reconstructions toward physically faithful and anatomically plausible solutions.

In this work, we first introduce the LUCID framework and evaluate its performance on simulated laminography datasets generated from high-fidelity tomographic volumes, which enable quantitative analysis of artifacts and restoration of the missing cone. In parallel, we establish an experimental benchmark by acquiring, curating, and publicly releasing what is, to our knowledge, the first nanoscale ptychographic X-ray laminography dataset of mouse brain tissue. We then demonstrate robust cross-domain generalization on real experimental laminography data. Across both simulated and experimental settings, LUCID consistently restores structural information, reduces interlayer aliasing, and recovers fine features that are severely degraded by conventional methods. We observe better performance on simulated data, presumably due to substantial diferences in acquisition and resolution between the training data and the experiment data. Finally, we assess the reliability and adherence to the GT. The recovered structures remain tightly constrained by the measured projections, indicating that the generative expressivity of the difusion prior is efectively regulated by physical data consistency.

![](images/b1c7ea78f45a6facdaae7c88b7eba052c6bdd1aeb6f9b9017bfce1a125873fe8.jpg)  
Fig. 2: Overview of the LUCID reconstruction framework. The reconstruction is formulated as an iterative process that alternates between a learned generative prior and laminography data consistency. (a) LUCID pipeline. Starting from Gaussian noise $x _ { T }$ , the volume is progressively refined through multiple denoising steps (MDP), interleaved with laminography data consistency updates, yielding the final reconstruction $x _ { 0 }$ . (b) Multi-view Difusion Prior Module (MDP). The current 3D estimate $x _ { t }$ is decomposed into axial or sagittal, or coronal slices, each processed by a dedicated 2D difusion denoiser to restore anatomically plausible structures. (c) Laminography Data Consistency Module (LDC). The volume is projected using the forward operator $\scriptstyle A _ { \theta }$ to generate synthetic projections $\hat { y } ,$ , which are compared with measured projections y<sub>measured</sub> via a data fidelity loss to enforce consistency with physical measurements. The two modules are coupled in an iterative loop, ensuring that reconstruction remains both measurement-consistent and anatomically coherent.

## 2 Results

## 2.1 LUCID framework

(a)

(b)

We introduce LUCID, a physics-guided generative reconstruction framework that tightly integrates multi-view difusion priors with laminography data consistency. As illustrated in Fig. 2, LUCID is formulated as an iterative refinement process, in which a 3D volume is progressively updated by alternating between two modules.

Training of the difusion prior. To learn a high-fidelity anatomical prior, we train difusion models on tomography reconstructions [10], which provide isotropic, high-resolution 3D volumes under a well-conditioned imaging geometry. Each volume is decomposed into axial, sagittal, and coronal slices, enabling eficient learning of volumetric structures using 2D models.

For each view, a denoising difusion model is trained to learn the reverse of a fixed forward noising process, which gradually perturbs clean samples $x _ { 0 }$ into Gaussian noise $x _ { T }$ . The model is optimized to predict and remove noise at each step, thereby capturing the distribution of realistic structures across orientations.

Iterative reconstruction with coupled modules. At inference, reconstruction is initialized from Gaussian noise $x _ { T }$ and proceeds through a sequence of alternating updates, as depicted in Fig. 2a. At each iteration $t ,$ the current estimate $x _ { t }$ is refined by two complementary operations:

(i) Multi-view Difusion Prior Module (MDP). The current 3D volume $x _ { t }$ is decomposed into axial, sagittal, and coronal slices in Fig. 2b. Each view is processed by its corresponding 2D difusion denoiser, which removes noise and restores anatomically plausible structures. The denoised slices are then reassembled into an updated 3D estimate $x _ { t - 1 }$ . This step primarily acts as a learned prior, promoting structural coherence and compensating for missing information.

(ii) Laminography Data Consistency Module $( L D C )$ . The updated volume is passed to the LDC module in Fig. $\mathrm { 2 c , }$ where data consistency is enforced. Specifically, the current estimate is projected using the laminography forward operator $\scriptstyle A _ { \theta }$ to generate simulated projections $\hat { y } .$ These are compared with the measured projections y<sub>measured</sub> via a data fidelity loss $\| \hat { y } - y _ { \mathrm { m e a s u r e d } } \| _ { 2 } ^ { 2 }$ . The resulting gradient is used to update the volume via a gradient descent step with a fixed step size. In practice, this update is performed for a small number of inner iterations within each difusion step, ensuring consistency with the measurements.

Coupled refinement. The MDP and LDC modules are applied sequentially and iteratively throughout the reverse difusion process in Fig. 2a, forming a closed-loop optimization. The difusion prior provides strong regularization by restricting solutions to the manifold of realistic brain structures, while the data consistency module anchors the reconstruction to physically observed data.

## 2.2 Performance on a simulated laminography dataset

We evaluate our method on a simulated laminography dataset generated using the laminography forward model applied to high-resolution full-angle tomography volumes, which serve as the ground truth for quantitative evaluation. The volume used for simulation is strictly held out from training, ensuring a separation between training and evaluation data and enabling an unbiased assessment of generalization performance.

Qualitative Evaluation: Fig. 3 presents a qualitative comparison of reconstruction results obtained with FBP, gradient-descent reconstruction (GD), and the proposed LUCID framework against the GT. The diferences are evident both at the global structural level and in localized regions of interest (ROIs).

![](images/e1f7cdb730b1d3e66083d15b708d945b1e4926ec29aa29388225fd867e07ce10.jpg)  
Fig. 3: Qualitative comparison of reconstruction fidelity on representative brain regions. Left: reconstructed slices from FBP, GD, LUCID, and GT. Colored boxes indicate regions of interest (ROIs), with corresponding zoom-ins shown on the right. FBP sufers from severe blurring and elongation due to missing-cone artifacts, while Gradient Descent (GD) partially improves structural sharpness but retains anisotropic distortions. In contrast, LUCID recovers fine-scale features with improved contrast and continuity, closely matching the GT. Highlighted ROIs correspond to biologically relevant structures: orange and yellow boxes indicate putative synapses, which are poorly resolved in FBP and GD but clearly recovered in LUCID; purple and green boxes indicate putative plasma membranes, whose continuity and boundary definition are substantially improved by LUCID.

FBP exhibits pronounced missing-cone artifacts, leading to strong axial blurring and loss of structural contrast. Fine cellular features are largely obscured, and synaptic-like structures in the axial direction are indistinguishable. GD partially mitigates these degradations through iterative refinement, yielding improved sharpness; however, residual anisotropic artifacts and inconsistent local contrast persist, particularly along the missing-cone direction. In contrast, LUCID substantially enhances structural fidelity and continuity. As highlighted in the zoomed ROIs, synaptic-like structures (orange and yellow boxes) that are indistinct in FBP and GD become clearly delineated, with improved contrast and spatial definition. Similarly, putative plasma membranes (purple and green boxes) exhibit enhanced continuity and sharper boundaries, approaching the appearance observed in the GT.

These results demonstrate that LUCID not only suppresses missing-cone–induced anisotropy but also restores biologically meaningful ultrastructural features, yielding reconstructions that are both visually consistent and anatomically plausible.

(a)  
(b)  
![](images/b8d5d912d95236b9ce4f7e00ac790dd03b01cf68c56a1e509098a0a61f598234.jpg)  
Fig. 4: Quantitative and spectral evaluation of reconstruction quality and uncertainty. (a) Spatial-domain error maps, i.e. diference to GT, for GD, and LUCID. Conventional methods exhibit strong anisotropic errors aligned with the missing-cone direction, whereas LUCID substantially reduces both the magnitude and spatial extent of residuals. (b) Uncertainty estimation of LUCID reconstructions. Top: Voxel-wise uncertainty map computed from multiple stochastic reconstructions, highlighting elevated uncertainty in regions afected by missing-cone information loss. Bottom: Overlay of the reconstruction and uncertainty map, showing that regions of high uncertainty spatially coincide with structurally ambiguous areas. (c) Fourierdomain analysis. Slices of the 3D Fourier magnitude are shown for FBP, GD, LUCID, and GT, with the missing-cone region delineated by dashed lines. FBP and GD exhibit pronounced spectral missing energy in the cone, while LUCID recovers substantially more energy within the missing cone, approaching the GT distribution. All reconstructions are angular undersampling, especially visible in FBP outside the missing-cone region.

Figure 4a shows voxel-wise diference maps between the reconstructions and the GT, revealing where reconstruction errors are spatially concentrated. When comparing the diference GD to the GT, these errors form continuous bright bands, indicating that these discrepancies coincide with membrane structures that typically define synaptic boundaries, implying that their loss critically afects the reliable identification of synapses. In contrast, in the diference map calculated with the LUCID reconstruction, the membrane-aligned residuals largely vanish and the remaining diferences are low and approximately isotropic.

Quantitative Evaluation: Next, we quantitatively evaluated our framework’s performance on this simulated ptychographic X-ray laminography (PyXL) dataset using peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and root mean squared error (RMSE) as metrics to assess reconstruction fidelity. Importantly, all evaluations are performed on volumes not seen during training, ensuring a fair evaluation of generalization performance.

Table 1 summarizes the quantitative performance of FBP, GD, and the proposed LUCID framework. Across all spatial-domain metrics, LUCID achieves the highest reconstruction fidelity. Compared with conventional FBP, LUCID improves PSNR by more than 13 dB and reduces RMSE by nearly 75%, reflecting its strong capability to suppress noise and geometric artifacts. Relative to iterative GD, LUCID provides consistent gains in both pixel-wise accuracy and perceptual similarity, confirming that the integration of difusion priors and data-consistency constraints yields more stable and anatomically faithful reconstructions.

Uncertainty Evaluation: To assess the reliability and stability of our reconstruction framework, we further quantify uncertainty in Fig. 4b by analyzing the variability across multiple stochastic runs of the multi-view difusion model. Specifically, we run the LUCID reconstruction 20 times, and compute a voxel-wise standard deviation volume as an empirical uncertainty map.

The spatial distribution of uncertainty reveals strong structural consistency with known limitations of laminography. Regions exhibiting the highest uncertainty are not randomly scattered; instead, they form coherent patterns that align with areas afected by missing-cone artifacts. These regions correspond to Fourier-domain directions that are under-constrained by the acquisition geometry. Their elevated uncertainty therefore reflects the model’s awareness of where the projections cannot uniquely determine the solution, forcing the difusion prior to play a dominant role. In contrast, wellsampled regions—such as high-contrast boundaries, membranes, and features lying near the central plane—show significantly lower uncertainty, indicating high model confidence where the data provides suficient coverage.

The uncertainty map is further visualized by overlaying the standard deviation onto the reconstructed slice as shown in Fig. 4b bottom. This representation highlights that coherent ultrastructural patterns such as large neurites and dense organelles exhibit low uncertainty, while difuse textures and elongated structures aligned with the missing cone display markedly higher uncertainty. This contrast confirms that the difusion model does not simply “hallucinate” structures in ill-posed regions; it also expresses reduced confidence when the solution cannot be reliably inferred from the data.

Importantly, the uncertainty encodes meaningful structure tied to the physical limitations of the imaging geometry. As such, the uncertainty map can serve as a diagnostic tool for downstream neuroanatomical analysis. In tasks such as connectome reconstruction where microstructural interpretation requires strict reliability, these uncertainty estimates help distinguish between confidently recovered neural processes and regions where additional measurements or complementary priors may be necessary. Moreover, this information can be propagated to downstream segmentation algorithms, providing spatial and directional cues on where feature representations are less certain and should be treated with caution.

Fourier Space Comparison: We compare the reconstructed Fourier frequency spectra of FBP, GD, and LUCID against the GT in Fig. 4c. The FBP result exhibits pronounced loss of information within the missing-cone region. Mild angular undersampling artefacts are visible outside the cone due to the finite number of projection angles used in the simulated acquisition (see Dataset composition and usage in the Methods section). GD partially recovers these frequencies, but remains anisotropic and incomplete. In contrast, LUCID substantially enhances spectral density within the missing cone, approaching the isotropic distribution of the GT. This enrichment of Fourier-space energy confirms that LUCID efectively fills the missing cone region, restoring spatial frequency content.

To specifically evaluate the efectiveness of our method in addressing the missingcone problem, we computed three Fourier-domain metrics: Cone Spectral Fidelity $\mathrm { ( C S F _ { c o n e } ) }$ , Cone Energy Diference $\left( \Delta E _ { \mathrm { c o n e } } \right)$ , and Cone Energy $\left( E _ { \mathrm { c o n e } } \right)$ , which are described in Section 4.1. $\mathrm { C S F _ { c o n e } }$ quantifies the correlation between the reconstructed and GT spectra within the missing-cone region, while $\Delta E _ { \mathrm { c o n e } }$ measures the relative energy discrepancy in this region. Table 1 presents complementary frequency-domain analyses. LUCID achieves the highest cone spectral fidelity and the largest recovered spectral energy within the missing-cone region, highlighting its ability to restore previously unsampled Fourier components. The reduced spectral deviation further indicates improved isotropy and a more balanced frequency distribution. Taken together, these metrics demonstrate that LUCID not only enhances spatial-domain fidelity but also provides superior Fourier-space completion, accurately recovering high-frequency information that conventional methods fail to reconstruct.

Table 1: Quantitative comparison of diferent reconstruction methods. Best results are shown in bold.
<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>RMSE↓</td><td> $\mathrm { C S F _ { c o n e } } \uparrow$ </td><td> $\Delta E _ { \mathrm { c o n e } }$  ↑</td><td> $E _ { \mathrm { c o n e } } \uparrow$ </td></tr><tr><td>FBP</td><td>16.74</td><td>0.4783</td><td>0.1456</td><td>0.3070</td><td>-0.9633</td><td>0.009</td></tr><tr><td>GD</td><td>26.78</td><td>0.9787</td><td>0.0457</td><td>0.6923</td><td>-0.8877</td><td>0.025</td></tr><tr><td>LUCID</td><td>30.14</td><td>0.9902</td><td>0.0364</td><td>0.8742</td><td>-0.5056</td><td>0.099</td></tr></table>

## 2.3 Performance on a experimental laminography dataset

To further validate the capability of LUCID, we evaluate its performance on a real experimental laminography dataset acquired using PyXL [32]. Unlike the simulated data, the experimental laminography measurements originate from a diferent domain, and with a distinct spatial resolution. These discrepancies make real-data evaluation a stringent test of model generalization.

Despite being trained exclusively on simulated tomographic data, LUCID generalizes efectively to experimental laminography without any retraining or fine-tuning. Fig. 5 demonstrates that LUCID restores structural information that is degraded by the missing-cone acquisition geometry in real data.

While FBP and GD reconstructions sufer from elongation and directional blurring, particularly along linear neural features, LUCID recovers these line-like structures with continuity and contrast. The result in Fig. 5a exhibits reduced anisotropy, smoother intensity transitions, and clearer tissue boundaries, showing that LUCID compensates missing information.

(a)  
![](images/8b91eb329ea82c74d1a90afb87c08a9c137efcf36ea2642ce7647adf8272ceb8.jpg)  
Fig. 5: Reconstruction results on experimental laminography data. (a) Spatial-domain comparison of FBP, GD, and LUCID reconstructions on real data. Colored boxes indicate ROIs with corresponding zoom-ins shown on the right. (b) Fourier-domain analysis of the corresponding reconstructions. Slices of the 3D Fourier magnitude are shown, with dashed lines indicating the missing-cone region. FBP and GD display significant spectral gaps, whereas LUCID recovers substantially more energy within the missing cone, resulting in a more isotropic frequency distribution. The reported cone energy quantifies the recovered spectral content within this region.

This improvement is further supported by Fourier-domain analysis, shown in Fig. 5b, where LUCID recovers significantly more spectral energy within the missingcone region compared to FBP and GD. The resulting frequency distribution is more isotropic, confirming that the restoration of spatial structures is directly linked to improved recovery of missing Fourier components.

## 3 Discussion

Here, we present LUCID, a physics-guided generative framework trained on highquality nanoscale ptychographic-tomography brain volumes and applied to both simulated and real ptychographic laminography data. Our results demonstrate that LUCID efectively restores the information lost due to the missing-cone in laminography, yielding high-fidelity reconstructions with enhanced isotropy and structural continuity. By successfully filling the missing Fourier regions, LUCID overcomes an important limitation in X-ray laminography, enabling faithful recovery of fine neural features such as elongated axonal and dendritic structures which are challenging to reconstruct accurately.

Beyond the proposed reconstruction framework, the first released nanoscale brain laminography dataset represents an important resource for the community. Public datasets have played a central role in accelerating progress in fields such as tomography, cryo-electron microscopy, and connectomics. We anticipate that the availability of experimental laminography data will similarly facilitate the development of reconstruction methods and help establish common benchmarks.

In addition, LUCID produces high-quality volumetric reconstructions that can facilitate downstream brain-imaging studies. In particular, in domains where access to suficient or diverse training data is restricted, such as due to privacy or acquisition constraints, LUCID’s generative capability provides a data-eficient route to augment existing datasets. This is especially meaningful for neuroscience, for which acquiring large annotated 3D datasets remains technically and ethically challenging. Our findings indicate that integrating difusion priors with physical constraints significantly enhances the model’s ability to recover coherent neural microarchitecture, providing a robust tool for advancing connectomic analysis and tissue-level interpretation.

To take advantage of these capabilities, we needed to overcome several additional challenges. High-quality 3D generative modeling still demands substantial computational resources, particularly when directly processing volumetric data. To address this, LUCID employs a multi-view difusion strategy, integrating axial, sagittal, and coronal perspectives. This approach achieves volumetric consistency through eficient 2D inference, greatly reducing computational cost while maintaining 3D coherence, ofering a practical balance between accuracy and scalability. Moreover, an inherent domain gap exists between tomography and laminography, as the two modalities difer not only in sampling geometry but also in the statistical distribution of reconstructed voxel intensities, including contrast, dynamic range, and frequency-dependent energy content. To address this challenge, we applied normalization-based domain adaptation, ensuring that LUCID’s outputs remain closely aligned with the physical characteristics of real laminographic data.

Overall, LUCID demonstrates that physics-guided difusion modeling can bridge modality and resolution gaps in X-ray nanoimaging. The framework provides a data-consistent approach for reconstructing neural tissue imaged with minimally invasive X-ray laminography. By unifying physical modeling and generative intelligence, LUCID establishes a foundation for reliable large-volume brain reconstruction, opening new opportunities for multiscale neuroimaging and other domains facing similar data incompleteness challenges.

## 4 Methods

## 4.1 Dataset and implementation details

We validate our approach on high-resolution ptychographic X-ray computed tomography (PXCT) datasets of mouse brain tissue acquired at the coherent small-angle X-ray scattering (cSAXS) beamline of the Swiss Light Source (SLS), Paul Scherrer Institute (PSI), Villigen, Switzerland [10]. The experimental preparation and imaging protocol are designed for structural preservation and contrast at synaptic resolution [10], ensuring the dataset serves as a robust benchmark for evaluating reconstruction algorithms.

Animals. Animals used in this study were around 16.4 week old wild-type male mice of C57Bl/6 background . All animal protocols were approved by the Ethics Committee of the board of the Francis Crick Institute and the United Kingdom Home Ofice under the Animals (Scientific Procedures) Act 1986. All animal IDs are listed in Supplementary Information 1.

Sample preparation. Brain tissue samples with preserved ultrastructure were prepared as described previously [10]. In brief, mice were sacrificed and a 600 µmthick horizontal section of the dorsal olfactory bulb extending 3\*3 mm2 was sliced in ice-cold dissecting bufer (phosphate bufer 65 mM, 0.6 mM CaCl2, 150 mM sucrose, $\mathrm { 3 0 0 \pm 2 0 ~ m O s m / L ) }$ using a Leica VT1200S vibratome and quickly transferred to icecold fixative (1.25% glutaraldehyde and 2.5% paraformaldehyde in 150mM sodium cacodylate bufer, $\mathrm { p H 7 . 4 0 , 3 0 0 \pm 2 0 m O s m / L }$ containing 0.02% sodium azide). After overnight fixation at $4 ^ { \circ } \mathrm { C }$ the fixative was washed with ice-cold wash bufer (150mM sodium cacodylate bufer, pH $\mathrm { 7 . 4 0 , 3 0 0 \pm 2 0 \ m O s m / L ) }$ before being stained with a ROTO protocol [33] using a Leica automated tissue processor (EMTP). The staining process involved bufered osmium (2% osmium tetroxide in 150mM sodium cacodylate bufer $\mathrm { p H }$ 7.40 for 1h30min at $2 0 ^ { \circ } \mathrm { C } )$ followed directly by bufered potassium ferrocyanide (3% in the same bufer described for the first osmium, for 1h30min at 20°C), 1% thiocarbohydrazide “TCH” (aq, 45 min at 30°C), 2% osmium tetroxide (aq, 3h at 20°C), 2% filtered uranyl acetate (aq, overnight at $4 ^ { \circ } \mathrm { C } ,$ followed by $2 \mathrm { h } , 5 0 ^ { \circ } \mathrm { C } )$ and lead aspartate ${ } ^ { 6 6 } \mathrm { L A } ^ { 9 }$ (pH 5.5, prepared as in [34], for 2h $5 0 ^ { \circ } \mathrm { C } )$ . Six 10 min washes in double distilled water preceded every staining step from TCH onwards, always at $2 0 ^ { \circ } \mathrm { C }$ except warmer washes at $5 0 \mathrm { { } ^ { \circ } C }$ before and after TCH and before LA. Samples were then dehydrated at 20°C by successive washes in ethanol solutions with increased concentration (70%, 90%, 90%, 2x100%) and transferred to acetonitrile (2x100%, 30 and 60 min). The TGPAP–DDM resin consists of the tri-functional epoxy resin TGPAP and the hardener DDM at a weight ratio of $\mathrm { D D M } { \cdot } \mathrm { T G P A P } = 1 { \cdot } 2 \ [ 1 0 ]$ . DDM was first dissolved in acetonitrile heated to $7 0 ~ ^ { \circ } \mathrm { C }$ and subsequently, TGPAP was added. The samples were incubated in 1:3 resin: acetonitrile for 2 h at room temperature, 1:1 resin: acetonitrile for 2–24 h at room temperature and subsequently the samples were placed in 1:1 resin: acetonitrile and cured for 12–72 h at $8 0 ~ ^ { \circ } \mathrm { C }$ . As the boiling point of acetonitrile is at $8 2 \ { } ^ { \circ } \mathrm { C } ,$ it is important to keep the sample container lid suficiently open such that the acetonitrile can evaporate during the curing process.

Sample trimming into pillars for PXCT. Resin-embedded brain samples were trimmed using a diamond knife to the approximate region of interest, as defined by preliminary microCT imaging. Cylindrical pillars of targeted histological layers were extracted using a 30 keV Ga-ion beam of 13 nA on a Zeiss NVision 40 Gallium FIB-SEM at PSI. Samples were mounted onto dedicated PXCT holders using an integrated micromanipulator, with Ga-assisted carbon deposition to ensure mechanical stability. Subsequent fine polishing was carried out in multiple stages, progressively reducing Ga-beam currents from 65 nA to 2.5 nA to achieve smooth, damage-minimized surfaces suitable for coherent X-ray illumination. The preparation of the brain samples is described in detail in [10].

PXCT instrumentation and acquisition. All PXCT measurements were performed on the OMNY instrument [35] at cSAXS. Coherent X-rays of 6.2 keV photon energy, corresponding to a wavelength of 2 <sup>˚</sup>A were produced using a fixed-exit doublecrystal Si(111) monochromator. For most datasets, the illumination optics consisted of a Fresnel zone plate (FZP) of 220 µm diameter and 60 nm outermost zone width with a focal length of 66.0 mm at 6.2 keV, fabricated at PSI, while the last beamtime e19533 employed an FZP from XRNanotech with 250 µm diameter and 30 nm outermost zone width with focal length of 37.5 mm at 6.2 keV. In both cases, a gold central stop with 40 µm and an order-sorting aperture with 30 µm diameter were used to block the unscattered beam, and structured illumination was optimized through locally displaced zones in the FZP. The FZPs were fabricated with designed wavefront aberrations to improve imaging quality [36]. A secondary source was defined by a 20 µm horizontal slit placed approximately 12 m from the undulator source, and the sample was positioned slightly downstream of the focal spot, with beam diameters of 8 µm for the first four beamtimes, and 5 µm for beamtime e19533. The far-field X-ray difraction patterns were detected using an invaccum Eiger 1.5M [37] detector placed approximately 7.2 m downstream [10].

PyXL instrumenation and acquisition. Samples of about 5 micron thickness were cut from metal-stained, dehydrated and resin-embedded brain tissue specimens using a diamond knife and deposited on a silicon nitride membrane for measurement. PyXL measurements on mouse brain were taken with the laminographic nano-imaging (LamNI) instrument [32] at the cSAXS beamline, of the SLS, PSI, Switzerland. Coherent X-ray photons with energy of 6.2 keV where focused by a FZP of 170 micron diameter and 60 outermost zonewidth fabricated with designed wavefront aberrations [36]. Ptychography scans were taken with a circular FOV of 27 micron diameter on the plane of the sample. On the plane perpendicular to the X-ray propagation the scan followed a Fermat spiral pattern [38] with an average step size of 0.5 microns. Each ptychogram had 1082 scanning points, at each scanning point an Eiger 1.5M [37] measured a far-field difraction pattern with an exposure time of 0.1 seconds. In total 752 projections were measured at sample orientation angles uniformly distributed between 0 and 360 degrees, yielding a reconstruction pixel size of 27 nm.

Ptychographic reconstruction and tomography. 2D Projection reconstructions from the first four beamtimes were obtained using several hundred iterations of the diference map algorithm followed by maximum likelihood refinement, whereas data from beamtime e19533 were reconstructed with two probe modes and 600 iterations of ML using the ptychoshelves package [39]. Projections were aligned using a tomographic consistency-based algorithm, followed by modified FBP with Hanning or ramp filters [40]. Non-rigid corrections were applied to mitigate sample drift and deformation [41]. For the PyXL ptychography reconstructions were carried out using 500x500 pixels from the detector, resulting in a pixel size of 27.9 nm. The ptychoshelves package [39] was used with 400 iterations of diference map [42] followed by maximum likelihood refinement [43, 44]. For both PXCT and PyXL projections were post-processed and aligned following the methods in [40].

Dataset composition and usage. The PXCT data comprises 10 tomograms of mouse brain pillars, each containing rich ultrastructural detail including neurons, synapses, and fine neuropil textures. Voxel sizes range from 38 nm to 81 nm, with volumetric dimensions varying across samples, e.g., $5 1 2 \times 7 6 8 \times 7 6 8$ voxels, $4 0 0 \times 7 6 8 \times$ 768 voxels, 256×896×896 voxels, down to 128×736×736 voxels. For the training, slices were randomly sampled from the first nine tomograms, separately for axial $( 7 3 6 \times 7 3 6$ voxels), sagittal $( 1 2 8 \times 7 3 6$ voxels), and coronal $( 1 2 8 \times 7 3 6$ voxels) orientations to enable multi-view learning. For the inference, we generate laminography projections from the remaining tomogram of PXCT with $1 2 8 \times 7 3 6 \times 7 3 6$ voxels to do simulation, and use real laminography projections from $\mathrm { P y X L }$ for experimental validation.

During simulation, laminographic projections were synthetically generated from the tomography volume using a laminography forward operator, matching the acquisition geometry. In particular, projections were simulated at a fixed tilt angle of $\theta = 6 1 ^ { \circ }$ ， reproducing the characteristic missing-cone sampling in Fourier space. For the given object thickness $( T \approx 4 . 7 \mu \mathrm { m } )$ and spatial resolution $( \Delta r \approx 3 7 \mathrm { n m } )$ , the laminographic sampling criterion predicts that $N = 7 2 3$ projections are required to achieve Nyquist angular sampling, according to $N = \pi { \frac { T } { \Delta r } }$ tan θ [7]. In contrast, we simulated only 360 uniformly spaced projections, corresponding to an angularly undersampled regime. This is evident in the gaps shown in the Fourier-domain gaps observed for FBP in Fig. 4.

Despite this, LUCID yields stable and structurally faithful reconstructions, highlighting the ability of difusion-based priors to compensate for incomplete angular sampling beyond the limits imposed by classical reconstruction theory.

Evaluation. To quantitatively assess reconstruction fidelity in the image domain, we use PSNR, SSIM, RMSE. Let I denote the GT volume and <sup>ˆ</sup>I the reconstructed volume, both defined on the voxel set Ω, where |Ω| denotes the total number of voxels in the 3D volume. All metrics are computed over the 3D brain volume.

We first define the mean squared error (MSE) as

$$
\mathrm { M S E } ( I , \hat { I } ) = \frac { 1 } { \left| \Omega \right| } \sum _ { i \in \Omega } \bigl ( I _ { i } - \hat { I } _ { i } \bigr ) ^ { 2 } .\tag{1}
$$

The dynamic range L is estimated from the GT as

$$
L = \operatorname* { m a x } _ { i \in \Omega } I _ { i } - \operatorname* { m i n } _ { i \in \Omega } I _ { i } .\tag{2}
$$

The PSNR used in our experiments is then given by

$$
\mathrm { P S N R } ( I , \hat { I } ) = 2 0 \log _ { 1 0 } \left( \frac { L } { \sqrt { \mathrm { M S E } ( I , \hat { I } ) } } \right) ,\tag{3}
$$

where higher values indicate better reconstruction quality.

The RMSE is defined as

$$
\mathrm { R M S E } ( I , \hat { I } ) = \sqrt { \mathrm { M S E } ( I , \hat { I } ) } = \sqrt { \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \bigl ( I _ { i } - \hat { I } _ { i } \bigr ) ^ { 2 } } .\tag{4}
$$

For SSIM, we adopt the global formulation consistent with our implementation [45]. Let

$$
\mu _ { I } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } I _ { i } , \qquad \mu _ { \hat { I } } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \hat { I } _ { i } ,\tag{5}
$$

$$
\sigma _ { I } ^ { 2 } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } ( I _ { i } - \mu _ { I } ) ^ { 2 } , \qquad \sigma _ { \hat { I } } ^ { 2 } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } ( \hat { I } _ { i } - \mu _ { \hat { I } } ) ^ { 2 } ,\tag{6}
$$

$$
\sigma _ { I \hat { I } } = \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } ( I _ { i } - \mu _ { I } ) ( \hat { I } _ { i } - \mu _ { \hat { I } } ) .\tag{7}
$$

The data range L defined above is used to construct the SSIM stabilisation constants

$$
C _ { 1 } = ( k _ { 1 } L ) ^ { 2 } , \qquad C _ { 2 } = ( k _ { 2 } L ) ^ { 2 } ,\tag{8}
$$

with fixed parameters $k _ { 1 } = 0 . 0 1$ and $k _ { 2 }$ = 0.03. The structural similarity index between I and <sup>ˆ</sup>I is thus

$$
\mathrm { S S I M } ( I , \hat { I } ) = \frac { ( 2 \mu _ { I } \mu _ { \hat { I } } + C _ { 1 } ) ( 2 \sigma _ { I \hat { I } } + C _ { 2 } ) } { ( \mu _ { I } ^ { 2 } + \mu _ { \hat { I } } ^ { 2 } + C _ { 1 } ) ( \sigma _ { I } ^ { 2 } + \sigma _ { \hat { I } } ^ { 2 } + C _ { 2 } ) } .\tag{9}
$$

Higher SSIM values indicate greater perceptual similarity between the reconstruction and the GT.

To evaluate spectral completeness, especially in the missing-cone region, we compute three Fourier-domain metrics.

In-cone spectral energy.

$$
E _ { \mathrm { c o n e } } = \frac { \sum _ { \mathbf { k } \in \mathcal { C } } | F _ { \hat { I } } ( \mathbf { k } ) | ^ { 2 } } { \sum _ { \mathbf { k } \in \mathcal { C } } | F _ { I } ( \mathbf { k } ) | ^ { 2 } } ,\tag{10}
$$

where C denotes the missing-cone region in Fourier space.

In-cone energy diference.

$$
\Delta E _ { \mathrm { c o n e } } = \frac { \sum _ { \mathbf { k } \in \mathcal { C } } \left( | F _ { \hat { I } } ( \mathbf { k } ) | ^ { 2 } - | F _ { I } ( \mathbf { k } ) | ^ { 2 } \right) } { \sum _ { \mathbf { k } \in \mathcal { C } } | F _ { I } ( \mathbf { k } ) | ^ { 2 } } .\tag{11}
$$

Cone spectral correlation.

$$
\mathrm { C S F _ { c o n e } } = \frac { \langle | F _ { \hat { I } } | , | F _ { I } | \rangle c } { \| F _ { \hat { I } } \| c \cdot \| F _ { I } \| c } ,\tag{12}
$$

which ranges from 0 to 1, with higher values indicating closer agreement with the GT.

Together, these spatial and spectral metrics provide a comprehensive evaluation of reconstruction fidelity and Fourier-space completion.

Training details As for model architecture, the difusion model is based on U Net with an encoder and decoder consisting of resnet blocks [46]. We trained difusion priors using AdamW with a cosine learning-rate schedule and 500 warm-up steps. The training data were extracted from axial, sagittal, and coronal slices of the datasets. In practice, two difusion models were trained: one using axial slices and one using the combined sagittal and coronal slices. Because sagittal and coronal views share identical spatial dimensions and similar image statistics, a single model was trained on both orientations and subsequently applied to each view during inference. Batch sizes were set to 10 for axial slices and 60 for sagittal/coronal slices to balance memory usage and training eficiency. Smaller batches used for the larger axial slices and larger batches used for the sagittal/coronal views. We augment the data by random rotation and scaling during the training phase to reduce overfitting. Data augmentation is a process to synthetically generate additional training samples for the purpose of avoiding overfitting and increasing robustness in the image domain. The implementation of all methods in this work is based on the PyTorch library, and all experiments are run on a single NVIDIA A100-PCIE.

## 4.2 Inverse problem in X-ray laminography

Reconstructing a volumetric object from X-ray projections is an inverse problem. Given an unknown 3D object $x \in \mathbb { R } ^ { n }$ , the measurement process is modelled as

$$
y = \mathcal { A } ( x ) + \eta ,\tag{13}
$$

where $\mathcal { A } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ denotes the forward projection operator which is given by the acquisition geometry, and η represents measurement noise. In conventional CT, A corresponds to a Radon transform and is well-conditioned when suficient angular coverage is available.

In laminography, the rotation axis has an angle $\beta ~ < ~ 9 0 ^ { \circ }$ relative to the beam propagation direction, as opposed to the conventional $9 0 °$ of CT. This geometry is advantageous for imaging specimens that are extended in 2D. Despite this favourable measurement geometry laminography produces a characteristic deficit in the Fourier domain: in contrast to computed tomography, which achieves complete reciprocal-space coverage, laminography introduces a cone-shaped region of missing frequencies. The resulting incomplete sampling makes the laminographic operator A ill-conditioned, leading in the reconstructions to axial elongation, mixing between adjacent layers, and highly directional blurring artifacts. In this work, we address this challenge by integrating a learned generative prior with explicit physical constraints, forming a unified, data-consistent difusion framework for 3D brain imaging.

## 4.3 Denoising difusion probabilistic models

Denoising difusion probabilistic models (DDPMs) [47] learn complex data distributions via a gradual forward noising process and a learned reverse denoising process. Let $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { d }$ denote a clean data sample, and let $\{ { \bf x } _ { t } \} _ { t = 1 } ^ { T }$ denote a sequence of latent variables indexed by the difusion time step $t \in \{ 1 , \ldots , T \}$ . The forward process is defined as a Markov chain $q ,$ where each transition adds Gaussian noise:

$$
q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) = \mathcal { N } \Big ( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } \Big ) ,\tag{14}
$$

where $\mathcal { N } ( \cdot ; \pmb { \mu } , \pmb { \Sigma } )$ denotes a multivariate normal distribution with mean $\pmb { \mu }$ and covariance $\pmb { \Sigma } , \pmb { \mathrm { I } } \in \mathbb { R } ^ { d \times d }$ is the identity matrix, and $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ is a predefined variance schedule controlling the noise magnitude at each step.

By recursively composing the linear Gaussian transitions of the Markov chain, the marginal distribution of $\mathbf { x } _ { t }$ conditioned on the original sample $\mathbf { x } _ { \mathrm { 0 } }$ admits a closed-form expression:

$$
q ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { t } ; \sqrt { \alpha _ { t } } \mathbf { x } _ { 0 } , ( 1 - \alpha _ { t } ) \mathbf { I } ) , \quad \alpha _ { t } = \prod _ { j = 1 } ^ { t } ( 1 - \beta _ { j } ) .\tag{15}
$$

The reverse process $p _ { \theta }$ is parameterized by a neural network $s _ { \theta }$ that predicts the noise component at step t:

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } ) = \mathcal { N } \bigg ( \mathbf { x } _ { t - 1 } ; \frac { 1 } { \sqrt { 1 - \beta _ { t } } } \big ( \mathbf { x } _ { t } + \beta _ { t } s _ { \theta } ( \mathbf { x } _ { t } , t ) \big ) , \beta _ { t } \mathbf { I } \bigg ) .\tag{16}
$$

During inference, a sample is generated by initializing $\mathbf { x } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and iteratively applying the reverse transitions. In LUCID, difusion models serve as learned priors of clean brain structures, providing biologically and statistically coherent regularization for the ill-posed laminography reconstruction.

## 4.4 LUCID: Laminography with Unified Consistent Difusion

To incorporate 3D anatomical coherence while maintaining computational tractability, LUCID adopts a multi-view difusion strategy. Difusion priors are learned from axial, sagittal, and coronal slices. In practice, two DDPMs are trained: one on axial slices and one on the combined sagittal and coronal slices, which share identical dimensions. Each model learns the slice distribution in its respective orientation, capturing complementary structural cues across views.

## 4.4.1 Multi-view difusion prior

At difusion step $t ,$ LUCID performs stochastic denoising updates using view-specific difusion models. For each orientation (axial, sagittal, or coronal), the noisy sample is updated as

$$
\mathbf { x } _ { t - 1 } ^ { \mathrm { v i e w } } = \frac { 1 } { \sqrt { 1 - \beta _ { t } } } \Big ( \mathbf { x } _ { t } ^ { \mathrm { v i e w } } + \beta _ { t } s _ { \theta , \mathrm { v i e w } } \big ( \mathbf { x } _ { t } ^ { \mathrm { v i e w } } , t \big ) \Big ) + \sqrt { \beta _ { t } } \mathbf { z } _ { t } ,\tag{17}
$$

where $s _ { \theta , \mathrm { v i e w } }$ denotes the view-specific denoiser that predicts the noise component, $\beta _ { t }$ is the difusion variance schedule, and $\mathbf { z } _ { t } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . The denoisers are applied in one view using the corresponding view-specific difusion model, then refined by the laminography data consistency module. The resulting volume is passed to the next view-specific denoising step. The views are cycled every three steps, so the axial, sagittal, and coronal priors are incorporated sequentially.

To enable the application of physical measurement operators, we further compute a noise-free estimate of the clean volume using Tweedie’s formula. Specifically, from the current noisy state $\mathbf { x } _ { t } ,$ we form

$$
\hat { \mathbf { x } } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \Big ( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \mathbf { \epsilon } _ { \theta } ( \mathbf { x } _ { t } , t ) \Big ) ,\tag{18}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { j } ) } \end{array}$ and $\epsilon _ { \theta }$ denotes the aggregated noise prediction obtained from the multi-view denoisers. This estimate serves as a deterministic approximation of the underlying clean volume at step t.

## 4.4.2 Laminography projection-domain data consistency

To enforce consistency with the measured laminography data, projection-domain constraints are imposed on the noise-free estimate $\hat { \mathbf { x } } _ { 0 }$ . Simulated laminography projections are obtained via the forward operator

$$
\hat { \mathbf { y } } = \mathcal { A } _ { \boldsymbol { \theta } } ( \hat { \mathbf { x } } _ { 0 } ) ,\tag{19}
$$

where $\scriptstyle A _ { \theta }$ denotes the laminography projection operator at tilt angle θ. The simulated projections $\hat { \mathbf { y } }$ are compared with the experimentally measured projections y<sub>measured</sub>, and the estimate is refined by minimizing the data-fidelity term through gradient descent:

$$
\hat { \mathbf { x } } _ { 0 } ^ { ( k + 1 ) } = \hat { \mathbf { x } } _ { 0 } ^ { ( k ) } - \eta \nabla _ { \hat { \mathbf { x } } _ { 0 } ^ { ( k ) } } \big \| \mathcal { A } _ { \theta } ( \hat { \mathbf { x } } _ { 0 } ^ { ( k ) } ) - \mathbf { y } _ { \mathrm { m e a s u r e d } } \big \| _ { 2 } ^ { 2 } ,\tag{20}
$$

where $\eta$ is the step size and k indexes the inner data-consistency iterations. This projection-domain correction enforces adherence to the physical measurements while preserving the anatomical priors provided by the difusion model.

## 4.4.3 Domain translation for real laminography inference

To bridge the gap between simulated tomography-domain training data and real laminography measurements, we employ a reversible domain translation module. A forward mapping $f _ { \mathrm { t r a n s } }$ aligns real reconstructions to the tomography-like intensity manifold via core-region histogram matching and global monotonic quantile interpolation. After difusion-based denoising, an approximate inverse mapping $f _ { \mathrm { i n v } }$ restores the output to the laminography domain by applying the inverse quantile transform, removing auxiliary padding, and reintegrating data-consistent regions. This bidirectional mapping enables stable difusion inference while preserving compatibility with the physical laminography operator.

## 4.4.4 Iterative sampling loop

The full reconstruction process alternates between (i) multi-view difusion prior module, (ii) laminography projection-domain consistency module, and (iii) controlled noise reintroduction. Iterating these steps from $t = T$ to $t = 1$ yields a final estimate $\mathbf { x } _ { \mathrm { 0 } }$ that is simultaneously physically consistent and anatomically plausible.

## 5 Data availability

All data supporting the findings described in this manuscript are available in the article and in the Supplementary Information. A supplementary structured table including metadata supporting the measurements presented is available via Zenodo. The PXCT data from pillar-shaped samples are available at Zenodo. And the $\mathrm { P y X L }$ brain datasets are available for preview at Zenodo. All datasets can be viewed and downloaded from the links provided in Supplementary Information 1.

## 6 Code availability

The code of LUCID is available under MIT license at Github https://github.com/ Athenaxr/LUCID.git.The exact version used in this work (v1.0.0) has been archived at Zenodo: https://doi.org/10.5281/zenodo.20430172. The ptychography reconstruction code is available from https://www.psi.ch/en/sls/csaxs/software (license:https: //www.psi.ch/sites/default/files/import/sls/csaxs/ComputingEN/License.txt).

## References

[1] Abbott, L.F., Bock, D.D., Callaway, E.M., Denk, W., Dulac, C., Fairhall, A.L., Fiete, I., Harris, K.M., Helmstaedter, M., Jain, V., et al.: The mind of a mouse. Cell 182(6), 1372–1376 (2020)

[2] Helmstaedter, M.: Synaptic-resolution connectomics: towards large brains and connectomic screening. Nature Reviews Neuroscience 27(2), 101–120 (2026)

[3] Knott, G., Marchman, H., Wall, D., Lich, B.: Serial section scanning electron microscopy of adult brain tissue using focused ion beam milling. Journal of Neuroscience 28(12), 2959–2964 (2008)

[4] Merchan-Perez, A., Rodriguez, J.-R., AlonsoNanclares, L., Schertel, A., DeFelipe, J.: Counting synapses using fib/sem microscopy: a true revolution for ultrastructural volume reconstruction. Frontiers in Neuroanatomy 3, 944 (2009)

[5] Dierolf, M., Menzel, A., Thibault, P., Schneider, P., Kewish, C.M., Wepf, R., Bunk, O., Pfeifer, F.: Ptychographic x-ray computed tomography at the nanoscale. Nature 467(7314), 436–439 (2010)

[6] Yu, Y.-S., Farmand, M., Kim, C., Liu, Y., Grey, C.P., Strobridge, F.C., Tyliszczak, T., Celestre, R., Denes, P., Joseph, J., et al.: Three-dimensional localization of nanoscale battery reactions using soft x-ray tomography. Nature communications 9(1), 921 (2018)

[7] Holler, M., Odstrcil, M., Guizar-Sicairos, M., Lebugle, M., M¨uller, E., Finizio, S., Tinti, G., David, C., Zusman, J., Unglaub, W., et al.: Three-dimensional imaging of integrated circuits with macro-to nanoscale zoom. Nature Electronics 2(10), 464–470 (2019)

[8] Aidukas, T., Phillips, N.W., Diaz, A., Poghosyan, E., M¨uller, E., Levi, A.F., Aeppli, G., Guizar-Sicairos, M., Holler, M.: High-performance 4-nm-resolution x-ray tomography using burst ptychography. Nature 632(8023), 81–88 (2024)

[9] Beck, A., Holler, M., Aidukas, T., Menzel, A., Guizar-Sicairos, M., Bokhoven, J.A., Ihli, J.: In-situ ptychographic nanotomography captures activation, mobility, and deactivation of supported catalysts. preprint (2025)

[10] Bosch, C., Aidukas, T., Holler, M., Pacureanu, A., M¨uller, E., Peddie, C.J., Zhang, Y., Cook, P., Collinson, L., Bunk, O., et al.: Nondestructive x-ray tomography of brain tissue ultrastructure. Nature Methods, 1–8 (2025)

[11] Laugros, A., Cloetens, P., Bosch, C., Schoonhoven, R., Pavlovic, L., Kuan, A.T., Livingstone, J., Zhang, Y., Kim, M., Hendriksen, A., et al.: Self-supervised image restoration in coherent x-ray neuronal microscopy. BioRxiv, 2025–02 (2025)

[12] Nikitin, V., Wildenberg, G., Mittone, A., Shevchenko, P., Deriy, A., De Carlo, F.: Laminography as a tool for imaging large-size samples with high resolution. Synchrotron Radiation 31(4), 851–866 (2024)

[13] Arslan, I., Tong, J.R., Midgley, P.A.: Reducing the missing wedge: High-resolution dual axis tomography of inorganic materials. Ultramicroscopy 106(11-12), 994– 1000 (2006)

[14] Bartesaghi, A., Sprechmann, P., Liu, J., Randall, G., Sapiro, G., Subramaniam, S.: Classification and 3d averaging with missing wedge correction in biological electron tomography. Journal of structural biology 162(3), 436–450 (2008)

[15] Du, W., Iacoviello, F., Mirza, M., Zhou, S., Bu, J., Feng, S., Grant, P.S., Jervis,

R., Brett, D.J., Shearing, P.R.: X-ray computed laminography: A brief review of mechanisms, reconstruction, applications and perspectives. Materials Today (2025)

[16] Helfen, L., Baumbach, T., Cloetens, P., Baruchel, J.: Phase-contrast and holographic computed laminography. Applied Physics Letters 94(10), 104103 (2009)

[17] Que, J.-M., Cao, D.-Q., Zhao, W., Tang, X., Sun, C.-L., Wang, Y.-F., Wei, C.-F., Shi, R.-J., Wei, L., Yu, Z.-Q., et al.: Computed laminography and reconstruction algorithm. Chinese Physics C 36(8), 777 (2012)

[18] Lu, J., Liu, Y., Zhang, P., Li, Z., Yang, M., Gui, Z.: An anisotropic alternating regularization-based reconstruction algorithm for cone beam computed laminography. NDT & E International 138, 102898 (2023)

[19] Kang, I., Jiang, Y., Holler, M., Guizar-Sicairos, M., Levi, A.F., Klug, J., Vogt, S., Barbastathis, G.: Accelerated deep self-supervised ptycho-laminography for threedimensional nanoscale imaging of integrated circuits. Optica 10(8), 1000–1008 (2023)

[20] Ding, G., Liu, Y., Zhang, R., Xin, H.L.: A joint deep learning model to recover information and reduce artifacts in missing-wedge sinograms for electron tomography and beyond. Scientific reports 9(1), 12803 (2019)

[21] Liu, Y.-T., Zhang, H., Wang, H., Tao, C.-L., Bi, G.-Q., Zhou, Z.H.: Isotropic reconstruction for electron tomography with deep learning. Nature communications 13(1), 6482 (2022)

[22] Wiedemann, S., Heckel, R.: A deep learning method for simultaneous denoising and missing wedge reconstruction in cryogenic electron tomography. Nature Communications 15(1), 8255 (2024)

[23] Zou, X., Shi, W., Du, M., Xing, Y.: Artifact reduction in rotational computed laminography using a deep learning method. Optics and Lasers in Engineering 187, 108881 (2025)

[24] Liu, T., Shi, L., Peng, B., Jia, T., Xu, X., Liu, B., Liu, Q.: Laminodif: Artifactfree computed laminography in non-destructive testing via difusion model. arXiv preprint arXiv:2601.07254 (2026)

[25] Jia, T., Wei, C., Zhu, M., Shi, R., Wang, Z., Cui, X., Liu, B.: The multi-scale fusion reconstruction algorithm of ct and cl. Physica Scripta 98(10), 105114 (2023)

[26] Ghandourah, E.E., Hamidi, S.H.A., Mohd Salleh, K.A., Wahab, M.N., Banoqitah, E.M., Alhawsawi, A.M., Moustafa, E.B.: Evaluation of welding imperfections with x-ray computed laminography for ndt inspection of carbon steel plates. Journal of Nondestructive Evaluation 42(3), 77 (2023)

[27] Xu, F., Helfen, L., Baumbach, T., Suhonen, H.: Comparison of image quality in computed laminography and tomography. Optics Express 20(2), 794–806 (2012)

[28] Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., Chen, E.: A survey on multimodal large language models. National Science Review 11(12), 403 (2024)

[29] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020)

[30] Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019)

[31] Lee, S., Chung, H., Park, M., Park, J., Ryu, W.-S., Ye, J.C.: Improving 3d imaging with pre-trained perpendicular 2d difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10710–10720 (2023)

[32] Holler, M., Odstrˇcil, M., Guizar-Sicairos, M., Lebugle, M., Frommherz, U., Lachat, T., Bunk, O., Raabe, J., Aeppli, G.: Lamni–an instrument for x-ray scanning microscopy in laminography geometry. Journal of synchrotron radiation 27(3), 730–736 (2020)

[33] Pallotto, M., Watkins, P.V., Fubara, B., Singer, J.H., Briggman, K.L.: Extracellular space preservation aids the connectomic analysis of neural circuits. Elife 4, 08206 (2015)

[34] Walton, J.: Lead asparate, an en bloc contrast stain particularly useful for ultrastructural enzymology. Journal of Histochemistry & Cytochemistry 27(10), 1337–1342 (1979)

[35] Holler, M., Raabe, J., Diaz, A., Guizar-Sicairos, M., Wepf, R., Odstrcil, M., Shaik, F.R., Panneels, V., Menzel, A., Sarafimov, B., et al.: Omny—a tomography nano cryo stage. Review of Scientific Instruments 89(4) (2018)

[36] Odstrˇcil, M., Lebugle, M., Guizar-Sicairos, M., David, C., Holler, M.: Towards optimized illumination for high-resolution ptychography. Optics express 27(10), 14981–14997 (2019)

[37] Guizar-Sicairos, M., Johnson, I., Diaz, A., Holler, M., Karvinen, P., Stadler, H.- C., Dinapoli, R., Bunk, O., Menzel, A.: High-throughput ptychography using eiger: scanning x-ray nano-imaging of extended regions. Optics express 22(12), 14859–14870 (2014)

[38] Huang, X., Yan, H., Harder, R., Hwu, Y., Robinson, I.K., Chu, Y.S.: Optimization of overlap uniformness for ptychography. Optics Express 22(10), 12634–12644 (2014)

[39] Wakonig, K., Stadler, H.-C., Odstrˇcil, M., Tsai, E.H., Diaz, A., Holler, M., Usov, I., Raabe, J., Menzel, A., Guizar-Sicairos, M.: Ptychoshelves, a versatile highlevel framework for high-performance analysis of ptychographic data. Applied Crystallography 53(2), 574–586 (2020)

[40] Odstrˇcil, M., Holler, M., Raabe, J., Guizar-Sicairos, M.: Alignment methods for nanotomography with deep subpixel accuracy. Optics Express 27(25), 36637– 36652 (2019)

[41] Odstrcil, M., Holler, M., Raabe, J., Sepe, A., Sheng, X., Vignolini, S., Schroer, C.G., Guizar-Sicairos, M.: Ab initio nonrigid x-ray nanotomography. Nature communications 10(1), 2600 (2019)

[42] Thibault, P., Dierolf, M., Menzel, A., Bunk, O., David, C., Pfeifer, F.: Highresolution scanning x-ray difraction microscopy. Science 321(5887), 379–382 (2008)

[43] Thibault, P., Guizar-Sicairos, M.: Maximum-likelihood refinement for coherent difractive imaging. New Journal of Physics 14(6), 063004 (2012)

[44] Odstrˇcil, M., Menzel, A., Guizar-Sicairos, M.: Iterative least-squares solver for generalized maximum-likelihood ptychography. Optics express 26(3), 3108–3123 (2018)

[45] Wang, Z., Simoncelli, E.P., Bovik, A.C.: Multiscale structural similarity for image quality assessment. In: The Thrity-seventh Asilomar Conference on Signals, Systems & Computers, 2003, vol. 2, pp. 1398–1402 (2003). Ieee

[46] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical Image Computing and Computer-assisted Intervention, pp. 234–241 (2015). Springer

[47] Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

## 7 Acknowledgements

We acknowledge the Paul Scherrer Institute, Villigen PSI, Switzerland, for the provision of synchrotron radiation beamtime at the cSAXS beamline of the Swiss Light Source. The work of L.B was supported by the Swiss Data Science Center (SDSC) under the CHIP project grant no. C22-11L. A.L.L. and N.W.P were supported by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement no.884104 (PSI-FELLOW-III-3i). T.A. was supported by funding from the Swiss National Science Foundation (SNF), project number 200021 196898. This work was also supported by the Francis Crick Institute, which receives its core funding from Cancer Research UK (CC2036 to A.T.S.), the UK Medical Research Council (CC2036 to A.T.S.), and the Wellcome Trust (CC2036 and 110174/Z/15/Z to A.T.S.). It was also supported by a Physics of Life grant (EP/W024292/1) to A.T.S. and A.P. funded by EPSRC and Wellcome.

## 8 Author contributions

W.F., L.B., and M.G.-S. conceived the research project. W.F. and L.B. developed the core algorithms. A.L.L. provided technical support throughout the project. C.B.P. contributed biological expertise and guided the validation of neuroanatomical structures. A.D., A.W., A.T.S., A.P., M.H., N.W.P., T.A, Y.Z. and M.G.-S. secured beamtime and performed the experiments, acquiring the brain imaging datasets and performing ptychographic and tomographic reconstructions.

All authors contributed to the interpretation of results and provided critical feedback on the manuscript.