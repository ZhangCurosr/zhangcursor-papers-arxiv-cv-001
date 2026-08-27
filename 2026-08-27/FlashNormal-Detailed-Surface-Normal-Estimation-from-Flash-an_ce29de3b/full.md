# FlashNormal: Detailed Surface Normal Estimation from Flash and No-Flash Images

Ruiyang Chen<sup>†</sup>, Feiran Li<sup>†</sup>, Heng Guo<sup>∗</sup>, and Zhanyu Ma, Senior Member, IEEE

![](images/9e9c143e7d48b40f9c389aaef542d1fa3931ef11d2f5b6e15b75555415d42a84.jpg)  
Fig. 1. We propose FlashNormal, a diffusion-based surface normal estimator taking flash/no-flash image pairs as input. Compared with single image-based normal estimation method [1], FlashNormal can recover high-quality surface normals with rich details, while reducing the shape-reflectance ambiguity as shown in the case of EARPHONE POSTER, whose actual shape is a plane.

Abstract—High-quality surface normal estimation is preferred for detailed surface shape recovery and image editing. Existing single image-based methods, though being a practical setup, often struggle to recover fine surface details and are sensitive to inherent shape-reflectance ambiguity. While photometric stereo achieves high-fidelity surface normal estimation from images under varying lights, its applicability is strictly limited by requiring a multi-illumination capture setup. To this end, we propose FlashNormal, a diffusion-based surface normal estimator from flash/no-flash image pairs. While retaining high practicability on modern smartphones, our proposal takes advantage of flash-induced shading variations, and leverages curvatureguided detail enhancement strategy, improving surface detail recovery and mitigating shape-reflectance ambiguity effectively. To evaluate our proposed method, we further present EvalFlash, the first real-world flash/no-flash evaluation dataset containing 20 objects aligned with ground-truth surface normals for quantitative benchmarking. Extensive experiments demonstrate the effectiveness of FlashNormal over state-of-the-art single imagebased methods and show a significant out-performance over flash/no-flash-based normal estimation method on EvalFlash.

Index Terms—Surface normal estimation, diffusion model,

flash/no-flash images.

## I. INTRODUCTION

URFACE normal maps represent a 2.5-dimensional ex-S pression of geometry. High-quality surface normals are required by various computer vision tasks, such as object relighting, augmented reality applications, and advanced rendering techniques [2]–[9]. However, producing detail-rich surface normal maps usually requires either a 3D scanning process with high-accuracy scanners [10], or the collection of lightcontrolled images for photometric stereo [11]–[14], both of which can be cumbersome and resource-intensive.

As a simpler alternative, recent advances in learning-based surface normal estimation [15]–[19] have demonstrated the capability of surface normal estimation from a single RGB image. Despite their popularity, these methods struggle to handle geometric details and may produce ambiguous shape recoveries, as a single image input cannot effectively distinguish between geometric and texture details due to the inherent shape-reflectance ambiguity. For example, as shown in the top row of Fig. 1, the geometric details of estimated surface normals are insufficient and over-smoothed. Also, the bottom row of Fig. 1 shows that a single image alone cannot provide enough information to determine if the depicted scene is a truly 3D shape or merely a 2D plane.

In this work, we introduce FlashNormal, a practical and effective approach for surface normal estimation. Drawing inspiration from photometric stereo—where shading differences across images under varying lighting help resolve ambiguities between shape and reflectance—we utilize flash/no-flash image pairs as a minimal yet accessible source of shading cues to reduce shape estimation ambiguity. Unlike prior twoshot surface normal estimation methods [20], [21], our method further benefits from the zero-shot generalization capabilities of diffusion priors [22], which help alleviate shape-reflectance ambiguity and improve estimation accuracy.

In addition to addressing shape-reflectance ambiguity, we further enhance geometric details in surface normal estimation. Specifically, we introduce a curvature-guided detail enhancement strategy that directs more focus to regions with pronounced geometric features during training. Moreover, inspired by techniques from small object detection [23], [24], we propose a zoomed-pixels strategy to refine fine-scale surface details. Together, these two strategies effectively contribute to producing detail-rich surface normal maps.

To evaluate the effectiveness of FlashNormal in real-world scenarios, we introduce EvalFlash, the first real-world flash/noflash image dataset with labeled ground-truth (GT) surface normals. EvalFlash comprises 20 objects with diverse shapes and materials, where object meshes are scanned and carefully aligned with the captured views to generate high-quality GT surface normal maps, allowing for quantitative evaluation in real-world scenes.

To summarize, our main contributions are as follows:

• We propose FlashNormal, a diffusion-based surface normal estimator taking in flash/no-flash images, boosting the shape recovery accuracy by reducing shapereflectance ambiguity while maintaining a practical setup;

• We introduce curvature-guided detail enhancement strategy and zoomed-pixels strategy, shown to be effective in improving geometric details of surface normal estimates;

• Building upon precise 3D scanning and manual alignment, we build a real-world evaluation dataset named EvalFlash. To the best of our knowledge, this is the first dataset for benchmarking flash/no-flash-based surface normal estimation with the corresponding GT surface normal maps.

## II. RELATED WORKS

In this section, we briefly review recent advances in singleview normal estimation based on their settings.

a) Surface normal estimation from single image: Surface normal estimation from a single image is an ill-posed problem due to shape-reflectance ambiguity. Both Li et al. [25] and Sang et al. [26] adopt CNN-based cascade networks to recover spatially-varying BRDFs (SVBRDF) and surface normals from a flash image. Tiwari et al. [27] further introduce an attention-based hourglass network to refine the quality of surface normals and reflectance. Most recent works explore Stable Diffusion (SD) [22] prior in surface normal estimation.

Specifically, GeoWizard [15] fine-tunes SD [22] model on a large-scale dataset with paired depth, normal, and images, where a geometry switcher is applied to address scene-level and object-level inputs. Since surface normal estimation is a deterministic task, GenPercept [28] and E2E-FT [1] reduce the inherent stochasticity of diffusion models by single-step denoising process. StableNormal [16] further introduces a semantic-guided refinement process to improve the accuracy and sharpness of estimated surface normals. Metric3Dv2 [29] brings a versatile geometric foundation model for depth and surface normal estimation trained on 16M indoor and outdoor scenes. Despite the practical setting of single-shot surface normal estimation, the above methods still struggle to resolve the inherent shape-reflectance ambiguity, leading to unsatisfied shape recovery accuracy and blurry surface normal estimation.

b) Surface normal estimation with photometric stereo: Photometric stereo is well-posed for surface normal estimation given images captured under varying illuminations of a Lambertian surface [11], [12], [30]–[35]. To address general non-Lambertian reflectance, learning-based methods such as CNN-PS [11], PS-FCN [12], SPLINE-Net [35] are proposed. However, these methods are restricted to darkroom settings due to their requirements for images taken under point lights. To tackle this limitation, Guo et al. [36] introduce a patch-based illumination strategy to extend the application range of photometric stereo to unknown environment light. UniPS [13] and SDM-UniPS [14], as the most recent photometric stereo methods, achieve surface normal estimation for general-reflectance surfaces under unknown natural illuminations. Despite the high-quality normal estimates from photometric stereo, multilight images are required for the input, which limits their application range, especially for casual capture scenarios.

c) Surface normal estimation from flash/no-flash images: The flash/no-flash setup, as widely available on modern smartphones and cameras, can provide images under varying lighting conditions for reducing shape-reflectance ambiguity while retaining practicality. Cao et al. [21] estimate surface normals of Lambertian surfaces based on the image ratio between flash/no-flash pairs. However, their approach requires a depth map as input to resolve shape ambiguity. In followup work, Xia et al. [37] extend normal estimation to non-Lambertian surfaces via introducing a near-infrared (NIR) camera and NIR flash to capture additional cues. To avoid the need for depth or NIR inputs, Boss et al. [20] introduce a cascaded network and inverse-rendering loss to estimate shape, illumination, and SVBRDFs. Li et al. [38] further demonstrate that an inverse rendering pipeline combined with the flash/noflash setup can be used for translucent surface reconstruction. Despite these advances, surface normal estimation quality on real images remains limited due to the domain gap created by synthetic datasets used in training. In contrast, we show that the diffusion prior, capturing real-world data distribution, can achieve accurate surface normal recovery.

## III. FLASHNORMAL

As illustrated in Fig. 2, FlashNormal comprises two key components: the flash/no-flash VAE and the diffusion priorbased surface normal estimator. The former takes flash/noflash image pairs as input to extract shape variation information to address the shape-reflectance ambiguity. The latter denoises the latent representation obtained from the flash/noflash VAE in a single-step, generating detailed surface normal maps enhanced by a curvature-guided detail enhancement strategy. Also, following the practice of StableNormal [16], we takes a coarse initial surface normal map from Metric3Dv2 [29] as input, shown to be effective in accelerating convergence speed of the training process.

![](images/1bd604a3e2013ddf9a0e4f1f54300c73cf28c3cc34bf5c92754e8e3a5658c93d.jpg)  
Fig. 2. Pipeline of FlashNormal: Flash/no-flash image pairs and a coarse shape initialization are individually encoded by the VAE and then concatenated into a latent representation, which serves as guidance for normal estimator. The normal estimator denoises the latent representation to generate the latent surface normal, which is then decoded to produce the final estimated surface normal map.

## A. Flash/no-flash VAE

Flash/no-flash image pairs are inherently complementary, helping to enhance surface normal estimation accuracy. A straightforward approach to input these image pairs into network is to concatenate them at the original resolution into 6-channel before applying VAE for compression. However, directly concatenating flash/no-flash image pairs as input could be problematic as pre-trained VAEs are generally trained on large-scale datasets of 3-channel images, and fine-tuning them with 6-channel inputs would lead to a mismatch with the pretrained prior.

To address this problem, we employ a pre-trained VAE to encode the flash and no-flash images separately, yielding two 4-channel latent space representations. Subsequently, we concatenate the flash and no-flash representations along the channel dimension within the latent space, resulting in an 8- channel flash/no-flash guidance feature map:

$$
\mathbf { z } ^ { ( f n f ) } = \operatorname { c o n c a t } \left( { \mathcal { E } } ( C ^ { ( f ) } ) , { \mathcal { E } } ( C ^ { ( n f ) } ) \right) ,\tag{1}
$$

where $\mathbf { z } ^ { ( f n f ) }$ represents the latent flash/no-flash features, and $C ^ { ( f ) }$ and $C ^ { \bar { ( } n f { ) } }$ denote the flash/no-flash image pairs, respectively.

In addition to flash/no-flash image pairs, we incorporate an initial coarse surface normal map as input, obtained from an off-the-shelf efficient surface normal estimation method Metric3Dv2 [29]. Following the practice of StableNormal [16], this coarse normal initialization could help accelerating the convergence of training process. We choose the no-flash image because it better matches the training distribution of Metric3Dv2 [29], whereas using the flash image or averaging two predictions would introduce a stronger domain shift. To encode the coarse surface normal map, we utilize a pre-trained VAE to obtain a latent representation $\mathbf { z } ^ { ( s t r ) }$ . We then combine it with $\mathbf { z } ^ { ( f n f ) }$ to form the final feature map $\hat { \mathbf { z } } _ { t }$ that is input into the subsequently network:

$$
\hat { \mathbf { z } } _ { t } = \mathrm { c o n c a t } \left( \mathbf { z } ^ { ( f n f ) } , \mathbf { z } ^ { ( s t r ) } \right) .\tag{2}
$$

## B. Diffusion prior guided normal estimator

We employ a Diffusion U-Net to denoise the feature maps $\hat { \mathbf { z } } _ { t }$ obtained in the previous step. We adopt a deterministic singlestep diffusion strategy in our network design. This approach reduces randomness during the denoising process, making it more suitable for the deterministic task of surface normal estimation, and leads to more stable and reliable outputs. Since the single-step generation process is independent of the number of time steps t, we set it to 999. The generation process is guided by the text prompt “detailed surface normal map from Digital Single-Lens Reflex (DSLR) image”. During training, the input to the Diffusion U-Net $f _ { \theta }$ is $\hat { \mathbf { z } } _ { t } ,$ , and the network outputs the latent surface normal estimation $\hat { \mathbf { z } } _ { 0 } \mathrm { : }$

$$
\hat { \mathbf { z } } _ { 0 } = - f _ { \theta } ( \hat { \mathbf { z } } _ { t } , t = 9 9 9 ) .\tag{3}
$$

We adopt an end-to-end training strategy. A VAE decoder D is utilized to decode the latent representation $\hat { \mathbf { z } } _ { 0 } .$ , producing the predicted surface normal in pixel space: $\hat { \mathbf { y } } = \mathcal { D } \left( \hat { \mathbf { z } } _ { 0 } \right)$ To enhance detail areas surface normal estimation, we employ two methods as follows.

a) Curvature-guided geometry enhancement: We propose a training strategy that leverages curvature information to enhance fine details in surface normals. Geometric details correspond to strong variations in normals, represented as curvature—higher in detailed regions and lower in smooth areas. By incorporating curvature maps into supervision, we put more attention on detail regions in surface normal estimation. As discussed in Heep et al. [39], curvatures are derived from the first and second fundamental forms:

$$
\begin{array} { c } { { { \pmb I } _ { i j } = \partial _ { i } { \pmb x } \cdot \partial _ { j } { \pmb x } , } } \\ { { { \pmb I } _ { i j } = - \partial _ { i } { \pmb x } \cdot \partial _ { j } { \pmb n } , } } \end{array}\tag{4}
$$

where $i , j$ denotes the uv coordinate in screen space. The first fundamental form measures the 3D distance between two points projected onto a 2D surface. The second fundamental form represents the gradient of the normal map. Given a set of surface normals, we can calculate $\partial _ { i } { \pmb x }$ since ${ \pmb n } \cdot \partial _ { i } { \pmb x } = 0$ This gives:

$$
\partial _ { u } \pmb { x } = \pmb { e _ { x } } - \frac { \pmb { n _ { x } } } { \pmb { n _ { z } } } \cdot \pmb { e _ { z } } , ~ \partial _ { v } \pmb { x } = \pmb { e _ { y } } - \frac { \pmb { n _ { y } } } { \pmb { n _ { z } } } \cdot \pmb { e _ { z } } ,\tag{5}
$$

where $e _ { x } , e _ { y } , e _ { z }$ are the unit vectors in the respective coordinate axis directions. Combining Eq. (4) with Eq. (5), we derive the curvature $\kappa _ { i }$ by solving the following generalized eigenvalue problem:

$$
\boldsymbol { \kappa } _ { i } \cdot \boldsymbol { I } \cdot \boldsymbol { v } _ { i } = \boldsymbol { \cal { I } } \boldsymbol { I } \cdot \boldsymbol { v } _ { i } .\tag{6}
$$

By combining equations (4), (5) and (6), we generate a curvature map from the surface normal map to supervise geometric surface details. Specifically, we set an empirical threshold of 0.002, identifying regions with curvature values exceeding this threshold as detail-rich areas. The corresponding detail mask M is then created to focus supervision on these regions. With this mask, we minimize the difference between curvature maps derived from GT and estimated surface normal maps in detail-rich areas:

$$
\mathcal { L } _ { \mathrm { c u r } } = \frac { 1 } { N _ { d } } \sum _ { i , j } \left( \mathcal { M } _ { i , j } \cdot ( k _ { i , j } ^ { * } - \hat { k } _ { i , j } ) \right) ^ { 2 } ,\tag{7}
$$

where (i, j) denotes the pixel coordinates, and $k ^ { * }$ and $\hat { k }$ are the GT and predicted curvature map derived from their respective surface normal map. $N _ { d }$ denotes the total number of valid pixels with a value of 1 in the detail mask M.

In addition to the curvature map loss, we supervise the network by minimizing the angular difference between the GT and predicted surface normals. Considering the detail-rich region dominant the angular error, leading to less attention to smooth and flat regions, we assign more weight to smooth surface normal areas. Specifically, the loss is designed as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { m a e } } = \frac { 1 } { N _ { d } } \sum _ { i , j } \mathcal { M } _ { i , j } \cdot \operatorname { a r c c o s } \left( \frac { \pmb { n } _ { i , j } ^ { * } \cdot \hat { \pmb { n } } _ { i , j } } { \| \pmb { n } _ { i , j } ^ { * } \| \| \hat { \pmb { n } } _ { i , j } \| } \right) + } \\ { \displaystyle 1 0 \cdot \frac { 1 } { N _ { s } } \sum _ { i , j } ( 1 - \mathcal { M } _ { i , j } ) \cdot \operatorname { a r c c o s } \left( \frac { \pmb { n } _ { i , j } ^ { * } \cdot \hat { \pmb { n } } _ { i , j } } { \| \pmb { n } _ { i , j } ^ { * } \| \| \hat { \pmb { n } } _ { i , j } \| } \right) , } \end{array}\tag{8}
$$

where $n ^ { * }$ and nˆ are the GT and predicted surface normal maps, respectively. $N _ { s }$ denotes the total number of valid pixels with a value of 0 in the detail mask M. In this way, the detailed and smooth regions can be well recovered considering both the curvature loss and mean angular error loss. The training process is finalized by minimizing the following total loss function to optimize the parameters of the Diffusion U-Net:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m a e } } + \lambda \cdot \mathcal { L } _ { \mathrm { c u r } } ,\tag{9}
$$

![](images/65a950f57d7a68094c931b845bf301a0c2733402d9ce0ebfbd3ffaa0ac4b02ad.jpg)  
Fig. 3. Flash/no-flash image pairs of BADGE and EARPHONE POSTER. We show the shading variation between flash and no-flash images through the green boxes.

where λ denotes the weight of total loss. We empirically set it to 100 in all of our experiments.

b) Zoomed-pixels for detail enhancement: We observe that the region of interest (ROI) in the training and testing data could occupy varying image area. Small ROI leads to blurry surface normal estimation due to limited information. To handle this problem, we borrow the idea from object detection techniques, where upscaling small objects improves detection accuracy [23], [24]. Specifically, we propose a zoomed-pixels strategy for detail enhancement. Specifically, we use bounding boxes to zoom in on regions of interest and resize them to 512 × 512, ensuring that objects occupy at least 80% of the final input image. This operation is also applied during inference, enhancing input informativeness and strengthening the guidance from flash/no-flash pairs, leading to improved accuracy on surface normal estimation. At inference time, the ROI can be obtained automatically by first predicting an object mask using SAM 2 [40] or multimodal large models, and then converting the mask into a bounding box and valid region. This formulation can also be extended to multi-object scenes when the target regions are segmented reliably, while inaccurate masks may lead to inaccurate ROIs and weaken the zoomed-pixels strategy.

## C. Analyses regarding flash inputs

We demonstrate that incorporating flash inputs helps to enhance the detail richness in estimated surface normals and mitigate shape-reflectance ambiguity.

a) Flash inputs help to estimate detail-rich surface normals: Incorporating flash aids in illuminating fine details of object surfaces, making the geometric structure more distinguishable. For instance, as shown on the left of Fig. 3, the noflash image is affected by the randomness of ambient lighting, resulting in blurry fine details and obscured concave-convex surfaces. In contrast, the flash image enhances the visibility of these details by providing brighter and more uniform illumination. Furthermore, the shading variations between the flash and no-flash images facilitate a clearer justification of the underlying geometric structure.

b) Flash inputs help to reduce shape-reflectance ambiguity: In this paper, all ambiguity mentioned refers to shapereflectance ambiguity. As illustrated on the right of Fig. 3, we analyze the appearance of EARPHONE POSTER under flash and no-flash conditions. In the no-flash image, it is unclear whether the image depicts a real pair of earphones or merely a poster. However, with flash input, the flash halo is concentrated in one spot, and no shading variation is observed in other parts of the earphone. If it is an actual pair of earphones, there would be more diverse shading variations across their surface. This insight allows us to conclude that the object is a poster of earphones. In this way, the shading variation observed between flash and no-flash images provides strong hints to determine the true structure of objects, effectively reducing shape-reflectance ambiguity.

![](images/e468b9d6b575b194ca9c6a1d26ccf775de89d81a3fada8fd140fb893e261972e.jpg)

![](images/9b03d8843e347e75a6df91fecec60c831af5fb7a6ace0cf5c268f7792054ef67.jpg)  
Fig. 4. Overview of our EvalFlash benchmark. We illustrate the GT surface normals, flash (top), and no-flash (bottom) images of EvalFlash.

![](images/0d622d69b755ac98cad78d1c0d6e38edcde515852f1bfe843036397f79a88c8c.jpg)  
Fig. 5. Capture setup of EvalFlash. (Top) Flash/no-flash capture setup and 3D scanner used for GT shape acquisition. (Bottom) Flash/no-flash image observations and the aligned surface normal map rendered from the mesh.

## IV. DATASET

We here introduce our rendered training dataset Flash100K and a large-scale synthetic testset EvalFlash-synth for comprehensively evaluating the robustness of surface normal estimation methods. We also present the first-ever real-captured dataset EvalFlash for benchmarking flash/no-flash surface normal estimation methods.

## A. A large-scale training dataset: Flash100K

Our FlashNormal requires large-scale flash/no-flash photorealistic image pairs with the GT surface normals to fine-tune SD [22]. While there already exists an off-the-shelf dataset for this purpose [20], it fails to reflect real-world object

![](images/a589a2c44377227bb1120ddba5c889f39c38eb756e29de794d74e410f066b2e8.jpg)  
3D assets

![](images/c9ab37f5eaa9fd478efbaec42e071e97707ee56b5366130a65b115b14540ae37.jpg)  
HDR Env. map

![](images/a2293edc9c20add73b6dd80ab58d9998dfe6d6e72700cf1d0daaf3921eb169ce.jpg)

![](images/442578bea5e0dc46868acd1eb5ba8dbf8b1344c7375a24889acd991344721c9d.jpg)

![](images/b97d5e060c4fa00a4484d442f232b598c8540b1660ebbc29229b18a02d3a54a9.jpg)  
Flash input  
No-flash input  
GT Normal  
Mask

Fig. 6. Four examples from our rendered photorealistic training dataset Flash100K. We show four 3D assets and the corresponding HDR environment maps in the first row and the bottom 4 rows show the rendering data.

appearances due to the random combination of materials and shapes. Therefore, we rendered a more comprehensive and photorealistic flash/no-flash dataset with carefully composed shapes and materials, which contains 100, 000 flash/no-flash image pairs and corresponding GT surface normals, wishing to narrow the domain gaps between rendered data and the real-world counterpart as much as possible. The 3D assets for our training dataset Flash100K are derived from the Gbuffer Objaverse (gObjaverse) dataset [41]. During rendering, after initially choosing 20, 000 samples from gObjaverse [41], we manually removed non-object assets to finalize selecting 17, 000 3D assets with high-quality mesh and physics-based rendering material. Additionally, we collect 15 HDR environmental maps to provide diverse scene illumination. We use the Cycles renderer in Blender [42] for physically-based rendering. Each 3D object is normalized to fit within a unit sphere. We position six virtual cameras with a focal length of 30mm at the top, bottom, left, right, front, and back of the object with a consistent camera-object distance of 500mm. To keep in line with the real-world layout of most smartphones and consumer cameras, we co-locate the point light source with the camera to capture flash and no-flash image pairs. For each capture, one of the 15 environmental maps is selected to provide ambient illumination, and two images are shot with and without the point light activated. In this way, a largescale dataset containing 100, 000 flash/no-flash image pairs is created, which are shown to be essential for FlashNormal to produce detailed surface normal maps. Figure 6 illustrates four example training samples. We show four 3D assets about these examples and their corresponding HDR environment maps during rendering on the top. The rendering training data of Flash100K is shown at the bottom.

## B. A comprehensive large-scale synthetic evaluation dataset: EvalFlash-synth

Since our method leverages diffusion prior for surface normal estimation, a comprehensive evaluation of its robustness and stable normal estimation capabilities requires a large-scale synthetic test set. To this end, we introduce EvalFlash-synth, the first synthetic test set specifically designed to assess normal estimation performance under flash/no-flash conditions. Comprising 839 high-quality test samples, this dataset is constructed by rendering 3D assets from gObjaverse [41] that are disjoint from the training set and exhibit rich geometric details and diverse material properties. The rendering methodology used for the test set is consistent with that used for the training set, ensuring a fair and reliable evaluation.

## C. A real-world evaluation dataset: EvalFlash

A real-captured dataset with GT surface normal maps is essential for quantitatively evaluating the real-world performance of our proposed FlashNormal. Since there does not exist such a dataset<sup>1</sup> in our setting, we decided to capture one by ourselves. Specifically, as illustrated in Fig. 4, we have collected data from 20 objects characterized by intricate shapes and diverse materials. Objects were specifically chosen for their rich geometric details and diverse reflectance properties, ensuring thorough evaluation on highly detailed surface reconstruction and robustness across varying material types. The images were captured using a Canon EOS R5 camera at a resolution of 8192 × 5464 with the flash both on and off, followed by manual segmentation of the silhouette of objects. To collect the GT, as demonstrated in Fig. 5, we utilize the EinScan-SP scanner to obtain the 3D mesh of each object. A typical mesh contains about 102k vertices and 206k faces, which is sufficient to support accurate rendering of ground-truth normal maps for quantitative evaluation. To align the scanned meshes with the object masks, we follow the methodology established by the DiLiGenT dataset [10] and render the GT surface normal maps. Similar to the dataset collection of DiLiGenT dataset [10], the alignment and rendering process is timeconsuming, taking us about 10 days for building the scale of 20 objects. To verify the alignment quality quantitatively, we measured the overlap between the RGB object masks and the rendered masks from the aligned meshes, and obtained an average IoU above 99.0% across the 20 objects. This confirms that the mesh-to-image alignment is sufficiently accurate for quantitative evaluation, and small residual alignment errors are unlikely to affect the overall MAE trends substantially.

## V. EXPERIMENT

We compare FlashNormal with state-of-the-art methods on surface normal estimation. We also employ 3D reconstruction as an application to demonstrate the superiority of ours.

## A. Experimental settings

a) Details about implementation: Our model is finetuned from Stable Diffusion v2 [22], leveraging its pretrained image diffusion weights rather than training from scratch. We train FlashNormal on Flash100K using AdamW optimizer [44] with a fixed learning rate of 3e-5. The batch size is set to 256 with the data distributed across 8 NVIDIA A6000 GPUs. The training spans 10 epochs, which requires approximately 2 days to complete. As our employed NVIDIA A6000 GPU can only process 2 batches at a time, we employ a gradient accumulation strategy by setting the batch sizes by conducting 16 accumulations per card, resulting in a total batch size of $2 \times 1 6 \times 8 = 2 5 6$ before back-propagation. During preprocessing, we scale all the flash/no-flash image pairs to the range of [−1, 1] to match the input requirements of our Flash/no-flash VAE. We employ the accelerate library [45] for efficient multi-GPU training and the xFormers library [46] for optimized attention implementation. FlashNormal is efficient. It takes about 1s to predict the normal map of a 512×512 image on an NVIDIA RTX A6000 GPU, including preprocessing steps such as ROI extraction and image resizing.

b) Baselines and metrics: We select the state-of-theart single image-based surface normal estimation methods including E2E-FT [1], Metric3Dv2 [29], StableNormal [16] and LX18 [25]; photometric stereo method SDM-UniPS [14]; and flash/no-flash based surface normal estimation method MV20 [20] for comparison. We input the no-flash images to the single image-based methods, and let the others take in the flash/no-flash pairs. We use the released code and checkpoint of the baselines in the following experiments.

For evaluation, we follow StableNormal [16] and employ the mean angular error (MAE) in degree, the root mean square error (RMSE), and the mean square error (MSE) to measure the difference between the estimated and GT surface normal maps.

## B. Benchmark evaluation on EvalFlash

As shown in Table I, we present the average metric values on 20 objects. Compared to the second-best method E2E-FT [1], FlashNormal improves the surface normal estimation accuracy by 6.6%. Compared with the state-of-the-art flash/noflash surface normal estimation method MV20 [20], we improve performance by 58.4%, demonstrating the effectiveness of FlashNormal. Table II and Table III report the per-object quantitative results of 20 objects of EvalFlash. For certain objects, E2E-FT [1] demonstrate superior performance compared to Metric3Dv2 [29], whereas Metric3Dv2 [29] outperforms them on others. On the other hand, our method consistently achieves the smallest MAE and RMSE across most tested objects, showcasing its stability in estimating surface normals for objects with rich details. We also show the MSE value across 20 objects in EvalFlash, as shown in Table IV and Table V.

![](images/777acde6c98111b4fb0f182eca4a46afb0ed4387fc05c0cea50e93475c912a07.jpg)  
Fig. 7. Surface normal estimation of PINEAPPLE and LADY from baseline methods and ours. We show the estimated surface normals and the corresponding angular error distributions with MAE in degrees shown on the top.

TABLE I  
COMPARISON ON EvalFlash AND EvalFlash-synth. WE CALCULATE THE AVERAGE VALUES OVER THE TWO TEST SETS.
<table><tr><td rowspan="2">Method</td><td colspan="3">EvalFlash</td><td colspan="3">EvalFlash-synth</td></tr><tr><td>MAE</td><td>RMSE</td><td>MSE</td><td>MAE</td><td>RMSE</td><td>MSE</td></tr><tr><td>E2E-FT [1]</td><td>16.02</td><td>0.33</td><td>0.04</td><td>17.50</td><td>0.38</td><td>0.06</td></tr><tr><td>Metric3Dv2 [29]</td><td>16.37</td><td>0.34</td><td>0.04</td><td>16.14</td><td>0.35</td><td>0.05</td></tr><tr><td>StableNormal [16]</td><td>18.53</td><td>0.36</td><td>0.05</td><td>18.48</td><td>0.38</td><td>0.06</td></tr><tr><td>LX18 [25]</td><td>38.83</td><td>0.88</td><td>0.27</td><td>45.39</td><td>1.02</td><td>0.38</td></tr><tr><td>SDM-UniPS [14]</td><td>21.89</td><td>0.44</td><td>0.07</td><td>26.32</td><td>0.52</td><td>0.11</td></tr><tr><td>MV20 [20]</td><td>36.01</td><td>0.72</td><td>0.21</td><td>55.38</td><td>1.21</td><td>0.53</td></tr><tr><td>FlashNormal (Ours)</td><td>14.96</td><td>0.31</td><td>0.03</td><td>15.47</td><td>0.33</td><td>0.04</td></tr></table>

In addition, we show a qualitative comparison in Fig. 7 on PINEAPPLE and LADY, representing specular and diffuse surfaces, respectively. While existing methods like E2E-FT [1] and StableNormal [16] perform well on only one of the two, our FlashNormal robustly estimates plausible surface normals for both, demonstrating its adaptability to diverse reflectances. Both Metric3Dv2 [29] and MV20 [20] can only produce coarse estimations, while ours can capture fine geometric details. SDM-UniPS [14] produces blurry predictions, likely due to an insufficient number of multi-light image inputs.

## C. Evaluation on EvalFlash-synth

We use the large-scale synthetic test set EvalFlash-synth to comprehensively compare the robustness and generalizability of different methods. Table I presents the quantitative results. Our proposed FlashNormal achieves the smallest MAE and RMSE on the average results across 839 test samples. Compared to the second-best method Metric3Dv2 [29], FlashNormal improves the surface normal estimation accuracy by 4.2%. Compared with the state-of-the-art flash/no-flash surface normal estimation method MV20 [20], we achieve a performance gain of 72.1%.

## D. Evaluation on causally captured images

Besides the comparison on EvalFlash, which is captured by a high-end camera, we also evaluate FlashNormal on flash/noflash images causally captured by a smartphone (Redmi K60) to show the robustness of our proposal w.r.t. capturing devices.

As shown in Fig. 8, we show the influence of shapereflectance ambiguity on single-image surface normal estimation. From a single image, E2E-FT [1] cannot disentangle the shape and image texture shown on near-planar boxes or the mirror. As a result, the recoveries are heavily deviated from their actual shapes, as demonstrated by their surface normals and the visualized meshes integrated from the normal maps [47]. On the other hand, Metric3Dv2 [29] can estimate more realistic surface normals, despite the results are also relatively blurry. Given the diffusion prior and shading variations from flash image pairs, our FlashNormal appears to be effective in reducing the shape-reflectance ambiguity and achieves plausible surface normal estimation. This demonstrates that by introducing flash inputs, our method effectively reduces shape reflectance ambiguity, thanks to shading variations.

To assess detailed surface shape recovery, we capture three objects with rich geometric features. As shown in Fig. 9, both E2E-FT [1] and Metric3Dv2 [29] fail to capture fine details in their surface normals, as seen in the normal maps and integrated meshes. This highlights the importance of shadow and specular distribution in flash image pairs for detail recovery. Additionally, curvature map supervision during training helps FlashNormal enhance geometric detail reconstruction. For SANTA and SQUIRREL, which have intricate surface textures, our method outperforms others by capturing complex details in the estimated surface normals. The meshes generated from these normals reveal pronounced protrusions, closely resembling the real objects. In contrast, competing methods produce smoother normals and meshes, indicating their limitations in handling highly detailed surfaces.

E. Comparison with single-image-based mesh generation methods

Currently, there are lots of works toward generating 3D shapes from single image [48]–[51], among which CLAY [48] and One-2-3-45++ [49] demonstrate state-of-the-art performances. Therefore, we compare these methods with ours in terms of mesh generation. As shown in Fig. 10, we show snapshots of reconstructed meshes by image-to-3D methods. Their front view does not align well with the actual appearance of the objects. In our comparison, the key criterion is not the overall plausibility of a completed 3D shape, but whether the visible front-view geometry remains faithful to the observed input. CLAY [48] and One-2-3-45++ [49] may generate plausible 3D geometry, but they can also hallucinate or distort visible structures, making their front-view geometry less faithful to the actual observation. Such hallucinated geometry also makes direct alignment-based evaluation less appropriate in our setting. Additionally, their side view is relatively smooth and does not effectively reflect the original details of the surface of the object. On the other hand, we show the mesh integrated by BiNi [47], whose input is estimated by our method. FlashNormal performs deterministic, pixelaligned prediction guided by physical shading cues, which is more suitable for recovering visible-surface normals and local details faithfully observed in the input images. Our mesh preserves better surface details and maintains geometric consistency across different views.

TABLE II  
QUANTITATIVE COMPARISON (MAE/RMSE) ON SURFACE NORMAL ESTIMATION BETWEEN BASELINE METHODS AND OURS. THE BEST AND SECOND-BEST RESULTS ARE EMBOLDENED AND UNDERLINED, RESPECTIVELY. THESE ARE THE 10 OBJECTS RESULTS FROM EVALFLASH.
<table><tr><td>Method</td><td>FORTUNE BAG</td><td>PINEAPPLE</td><td>LADY</td><td>PINKCAT</td><td>GUANYU</td><td>EYGPTCAT</td><td>SHAKESPEARE</td><td>MONKEY</td><td>TREE</td><td>MONSTER</td></tr><tr><td>E2E-FT [1]</td><td>15.36 / 0.32</td><td>14.45 / 0.29</td><td>19.84 / 0.45</td><td>17.22 / 0.35</td><td>20.02 / 0.41</td><td>15.09 / 0.32</td><td>16.42 / 0.34</td><td>14.52 / 0.29</td><td>18.71 / 0.40</td><td>12.91 / 0.26</td></tr><tr><td>Metric3Dv2 [29]</td><td>14.74 / 0.31</td><td>16.03 / 0.32</td><td>18.75 / 0.44</td><td>20.31 / 0.41</td><td>14.88 / 0.31</td><td>12.00 / 0.27</td><td>18.70 / 0.39</td><td>15.54 / 0.31</td><td>15.48 / 0.33</td><td>16.44 / 0.33</td></tr><tr><td>StableNormal [16]</td><td>15.50 / 0.32</td><td>18.27 / 0.35</td><td>18.15 / 0.42</td><td>20.45 / 0.39</td><td>19.67 / 0.38</td><td>15.19 / 0.31</td><td>20.82 / 0.42</td><td>17.33 / 0.33</td><td>18.10 / 0.35</td><td>15.99 / 0.31</td></tr><tr><td>LX18 [25]</td><td>31.70 / 0.70</td><td>32.10 / 0.74</td><td>40.98 / 0.86</td><td>41.92 / 0.86</td><td>36.90 /  0.77</td><td>40.26 / 0.88</td><td>33.49 / 0.75</td><td>29.53 / 0.70</td><td>36.43 / 0.81</td><td>35.61  / 0.83</td></tr><tr><td>SDM-UniPS [14]</td><td>21.95 / 0.46</td><td>18.51 / 0.36</td><td>24.01 / 0.51</td><td>36.66 / 0.68</td><td>29.79 / 0.58</td><td>17.73 / 0.38</td><td>19.51 / 0.41</td><td>16.54 / 0.33</td><td>23.00 / 0.46</td><td>16.67 / 0.33</td></tr><tr><td>MV20 [20]</td><td>25.01 / 0.55</td><td>33.19 / 0.69</td><td>45.92 / 0.86</td><td>44.00 / 0.83</td><td>33.42 / 0.68</td><td>42.05 / 0.82</td><td>33.85 / 0.68</td><td>31.76 / 0.67</td><td>34.96 / 0.71</td><td>38.30 / 0.75</td></tr><tr><td>FlashNormal (Ours)</td><td>14.68 / 0.30</td><td>11.13 / 0.23</td><td>17.79 / 0.40</td><td>18.95 / 0.38</td><td>16.45 / 0.33</td><td>11.96 / 0.25</td><td>14.98 / 0.31</td><td>11.52 / 0.24</td><td>17.01 / 0.34</td><td>14.62 / 0.28</td></tr></table>

TABLE III

QUANTITATIVE COMPARISON (MAE/RMSE) ON SURFACE NORMAL ESTIMATION BETWEEN BASELINE METHODS AND OURS. THE BEST AND SECOND-BEST RESULTS ARE EMBOLDENED AND UNDERLINED, RESPECTIVELY. THESE ARE THE OTHER 10 OBJECTS RESULTS FROM EVALFLASH.
<table><tr><td>Method</td><td>HUGO</td><td>ANDERSON</td><td>HAPPYDOG</td><td>MONKEYHEAD</td><td>LUXUN</td><td>SITTINGBUDDHA</td><td>GOETHE</td><td>LION</td><td>PERSIMMON</td><td>ToY</td></tr><tr><td>E2E-FT [1]</td><td>14.87 / 0.30</td><td>17.69 / 0.36</td><td>14.00 / 0.29</td><td>14.15 / 0.29</td><td>15.69 / 0.33</td><td>13.15 / 0.27</td><td>17.65 / 0.37</td><td>18.20 / 0.37</td><td>14.58 / 0.32</td><td>15.51 / 0.34</td></tr><tr><td>Metric3Dv2 [29]</td><td>18.77 / 0.38</td><td>19.00 / 0.38</td><td>13.28 / 0.28</td><td>13.74 / 0.27</td><td>18.89 / 0.38</td><td>15.69 / 0.31</td><td>19.59 / 0.39</td><td>15.64 / 0.32</td><td>13.12 / 0.29</td><td>16.91 / 0.36</td></tr><tr><td>StableNormal [16]</td><td>20.41 / 0.39</td><td>21.33 / 0.42</td><td>19.48 / 0.38</td><td>17.45 / 0.34</td><td>19.84 / 0.38</td><td>17.89 / 0.36</td><td>20.89 / 0.41</td><td>18.11 / 0.37</td><td>17.42 / 0.37</td><td>18.44 / 0.39</td></tr><tr><td>LX18 [25]</td><td>30.66 / 0.62</td><td>31.61 / 0.64</td><td>28.17 / 0.61</td><td>49.03 / 0.89</td><td>28.82 / 0.63</td><td>38.09 / 0.73</td><td>30.10 / 0.63</td><td>43.89 / 1.04</td><td>48.35 / 1.12</td><td>50.77 / 1.13</td></tr><tr><td>SDM-UniPS [14]</td><td>23.62 / 0.46</td><td>21.79 / 0.44</td><td>21.02 / 0.43</td><td>14.55 / 0.29</td><td>17.87 / 0.36</td><td>22.61 / 0.43</td><td>21.70 / 0.43</td><td>22.86 / 0.47</td><td>17.02 / 0.36</td><td>31.34 / 0.61</td></tr><tr><td>MV20 [20]</td><td>33.59 / 0.85</td><td>41.31 / 0.96</td><td>36.74 / 0.90</td><td>43.29 / 0.98</td><td>32.41 / 0.93</td><td>52.38 / 1.11</td><td>34.93 / 0.89</td><td>33.09 / 0.69</td><td>40.67 / 0.81</td><td>47.46 / 0.90</td></tr><tr><td>FlashNormal(ours)</td><td>13.79 / 0.29</td><td>14.55 / 0.31</td><td>13.13 / 0.27</td><td>13.60  / 0.26</td><td>14.61 / 0.31</td><td>14.08 / 0.28</td><td>15.66 / 0.33</td><td>18.21 / 0.37</td><td>15.68 / 0.32</td><td>16.83 / 0.35</td></tr></table>

TABLE IV

QUANTITATIVE COMPARISON (MSE) ON SURFACE NORMAL ESTIMATION BETWEEN BASELINE METHODS AND OURS. THE BEST AND SECOND-BEST RESULTS ARE EMBOLDENED AND UNDERLINED, RESPECTIVELY.THESE ARE THE 10 OBJECTS RESULTS FROM EVALFLASH.
<table><tr><td>Method</td><td>FORTUNE BAG</td><td>PINEAPPLE</td><td>LADY</td><td>PINKCAT</td><td>GUANYU</td><td>EYGPTCAT</td><td>SHAKESPEARE</td><td>MONKEY</td><td>TREE</td><td>MONSTER</td></tr><tr><td>E2E-FT [1]</td><td>0.03</td><td>0.03</td><td>0.07</td><td>0.08</td><td>0.05</td><td>0.03</td><td>0.04</td><td>0.03</td><td>0.05</td><td>0.02</td></tr><tr><td>Metric3Dv2 [29]</td><td>0.04</td><td>0.03</td><td>0.06</td><td>0.06</td><td>0.03</td><td>0.02</td><td>0.05</td><td>0.03</td><td>0.04</td><td>0.04</td></tr><tr><td>StableNormal [16]</td><td>0.04</td><td>0.05</td><td>0.10</td><td>0.07</td><td>0.06</td><td>0.03</td><td>0.05</td><td>0.03</td><td>0.06</td><td>0.06</td></tr><tr><td>LX18 [25]</td><td>0.18</td><td>0.27</td><td>0.34</td><td>0.24</td><td>0.29</td><td>0.35</td><td>0.27</td><td>0.22</td><td>0.33</td><td>0.28</td></tr><tr><td>SDM-UniPS [14]</td><td>0.05</td><td>0.06</td><td>0.09</td><td>0.15</td><td>0.11</td><td>0.06</td><td>0.06</td><td>0.04</td><td>0.06</td><td>0.04</td></tr><tr><td>MV20 [20]</td><td>0.10</td><td>0.14</td><td>0.24</td><td>0.19</td><td>0.12</td><td>0.23</td><td>0.16</td><td>0.14</td><td>0.15</td><td>0.19</td></tr><tr><td>FlashNormal(ours)</td><td>0.03</td><td>0.01</td><td>0.05</td><td>0.05</td><td>0.04</td><td>0.02</td><td>0.03</td><td>0.02</td><td>0.04</td><td>0.03</td></tr></table>

TABLE V

QUANTITATIVE COMPARISON (MSE) ON SURFACE NORMAL ESTIMATION BETWEEN BASELINE METHODS AND OURS. THE BEST AND SECOND-BEST RESULTS ARE EMBOLDENED AND UNDERLINED, RESPECTIVELY. THESE ARE THE OTHER 10 OBJECTS RESULTS FROM EVALFLASH.
<table><tr><td>Method</td><td>HUGO</td><td>ANDERSON</td><td>HAPPYDOG</td><td>MONKEYHEAD</td><td>LUXUN</td><td>SITTINGBUDDHA</td><td>GOETHE</td><td>LION</td><td>PERSIMMON</td><td>ToY</td></tr><tr><td>E2E-FT [1]</td><td>0.03</td><td>0.04</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.02</td><td>0.04</td><td>0.04</td><td>0.03</td><td>0.04</td></tr><tr><td>Metric3Dv2 [29]</td><td>0.04</td><td>0.04</td><td>0.03</td><td>0.02</td><td>0.05</td><td>0.03</td><td>0.05</td><td>0.03</td><td>0.03</td><td>0.04</td></tr><tr><td>StableNormal [16]</td><td>0.04</td><td>0.05</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.07</td><td>0.05</td><td>0.07</td></tr><tr><td>LX18 [25]</td><td>0.21</td><td>0.26</td><td>0.19</td><td>0.24</td><td>0.22</td><td>0.33</td><td>0.24</td><td>0.29</td><td>0.30</td><td>0.33</td></tr><tr><td>SDM-UniPS [14]</td><td>0.07</td><td>0.08</td><td>0.05</td><td>0.02</td><td>0.06</td><td>0.05</td><td>0.06</td><td>0.07</td><td>0.05</td><td>0.12</td></tr><tr><td>MV20 [20]</td><td>0.24</td><td>0.26</td><td>0.25</td><td>0.26</td><td>0.28</td><td>0.31</td><td>0.29</td><td>0.17</td><td>0.23</td><td>0.27</td></tr><tr><td>FlashNormal(ours)</td><td>0.02</td><td>0.03</td><td>0.02</td><td>0.02</td><td>0.03</td><td>0.02</td><td>0.03</td><td>0.04</td><td>0.03</td><td>0.04</td></tr></table>

## F. Ablation study

a) Ablation on geometric detail recovery from different setups: As shown in Fig. 11, we demonstrate the effectiveness of flash input, curvature-guided geometry enhancement, and zoomed-pixels strategy in enhancing the accuracy of surface normal estimation with rich detail. When any of these modules is omitted, the MAE value of the estimated surface normals increases, and geometric details are lost, particularly at the neck of PINKCAT and the head of LADY. Table VI further reveals that each of the three modules individually reduces the MAE by 3.4%, 3.5%, and 4.1% on EvalFlash, demonstrating the improved detail estimation ability of our method.

b) Ablation on different input settings: As shown in Fig. 12, flash inputs benefit shape-reflectance ambiguous surfaces, such as the DRUM POSTER and MIRROR, by using shading variations to differentiate flat surfaces from true 3D

TEXTUREDBOX

Flash/no-flash images

FlashNormal (Ours)

MIRROR

E2E-FT [1]

DRUMPOSTER

![](images/f54374cac1501c109f82692142aee0fcf2d8931311c8392b73dae0f576d24f23.jpg)  
Metric3Dv2 [29]  
FANPOSTER  
SANTA  
SQUIRREL

Fig. 8. Qualitative comparison with baseline methods in terms of shape-reflectance ambiguity. Our method can estimate high-quality ambiguous-less surface normal maps, leading to visibly flatter meshes after surface normal integration.

![](images/5989f0194a79635bcf2c6e9e1c066c48e1c67676295e21447cec61fdf3294835.jpg)  
Fig. 9. Estimated surface normal maps and reconstructed meshes of detail-rich objects. Compared to the peer methods, our estimated surface normals present richer details, leading to higher-quality structured meshes.

TABLE VI  
ABLATION STUDY ON EvalFlash AND EvalFlash-synth. WE CALCULATED THE AVERAGE VALUES OF TWO TESTSETS.
<table><tr><td></td><td colspan="3">EvalFlash</td><td colspan="3">EvalFlash-synth</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td>MSE</td><td>MAE</td><td>RMSE</td><td>MSE</td></tr><tr><td>FlashNormal (Ours)</td><td>14.96</td><td>0.31</td><td>0.03</td><td>15.47</td><td>0.33</td><td>0.04</td></tr><tr><td>w/o flash input</td><td>15.48</td><td>0.32</td><td>0.03</td><td>15.88</td><td>0.35</td><td>0.05</td></tr><tr><td>w/o curvature-guided detail enhancement</td><td>15.50</td><td>0.33</td><td>0.04</td><td>16.07</td><td>0.35</td><td>0.05</td></tr><tr><td>w/o zoomed pixels strategy</td><td>15.60</td><td>0.32</td><td>0.04</td><td>17.96</td><td>0.38</td><td>0.06</td></tr><tr><td>w/o global shape initialization</td><td>16.33</td><td>0.34</td><td>0.04</td><td>18.23</td><td>0.39</td><td>0.07</td></tr></table>

structures. Additionally, the last row of Table VI and Fig. 13 demonstrate that providing an easy and fast-to-obtain coarse surface normal initialization accelerates convergence of training process, enhancing the quality of the predicted surface normals when training for the same number (10) of epochs. These experiments indicate the importance of both flash inputs and coarse shape initialization in the learning process.

c) Sensitivity to the loss weight λ: We additionally evaluated prompt variants by removing “DSLR” or “detailed” and observed only minor changes, indicating that the prediction is dominated by flash/no-flash visual conditioning rather than prompt wording. The quantitative comparison is reported in Table VII.

![](images/aea015c3233789d3fa453a51aee3726055b765d2eb4c2a43267f05d3237bcfed.jpg)

![](images/a156bb21f23c0cf96cca235096c2be419a10bbd8b90661ffe68846378512dfd1.jpg)  
Fig. 10. Mesh generation results of SHAKESPEARE (left side) and MON-KEY (right side) by image-to-3D methods [48], [49] and ours. The 3D mesh visualizations are taken as snapshots.

![](images/971544dc6739001fe4e84b3c8c4e87612e49f583b0ddf17a619ff18c842e0a4e.jpg)  
Fig. 11. Ablation on geometric detail recovery from different setups.  
Fig. 12. Ablation study on flash input.

d) Sensitivity to prompt variants: We additionally evaluated prompt variants by removing “DSLR” or “detailed” and observed only minor changes, indicating that the prediction is dominated by flash/no-flash visual conditioning rather than prompt wording. The quantitative comparison is reported in Table VIII.

![](images/e1ad49b82f60c1ef9645a39163f2ed5f2d3cd5f4aea9eac01039607631e5d925.jpg)  
Fig. 13. Ablation study on coarse shape initialization. Though the initial surface normal is lack of detail, it can help enhance the accuracy of estimated surface normal from FlashNormal.

TABLE VII  
SENSITIVITY STUDY ON THE LOSS WEIGHT λ. WE REPORT THE AVERAGEVALUES ON EVALFLASH AND EVALFLASH-SYNTH.
<table><tr><td>λ</td><td colspan="3">EvalFlash</td><td colspan="3">EvalFlash-synth</td></tr><tr><td></td><td>MAE</td><td>RMSE</td><td>MSE</td><td>MAE</td><td>RMSE</td><td>MSE</td></tr><tr><td>10</td><td>15.46</td><td>0.34</td><td>0.11</td><td>16.12</td><td>0.35</td><td>0.05</td></tr><tr><td>1000</td><td>15.23</td><td>0.33</td><td>0.05</td><td>15.98</td><td>0.34</td><td>0.05</td></tr><tr><td>100 (used in paper)</td><td>14.96</td><td>0.31</td><td>0.03</td><td>15.47</td><td>0.33</td><td>0.04</td></tr></table>

![](images/278b23d8a82f4b65c5419f91f1389d1d37385e96eeb07b2cce7b645e3965b1ba.jpg)  
Fig. 14. Comparison between E2E-FT [1] and ours about zoomed pixels strategy. Valid regions are resized to the same resolution.

## G. Analyses regarding zoomed-pixels for inputting image to E2E-FT

While zoomed pixels is a simple yet effective method for fitting variously captured images to networks, in the main paper, we stick to the default settings of our peer methods without applying this method. Therefore, to more comprehensively demonstrate the superiority of our proposal, we here show that even zoomed pixels (adding bounding box) for inputting image of E2E-FT [1], its performance is still inferior to our method. As shown in Fig. 14, the two columns on the left are the input RGB images for our proposal, and we input the no-flash image with and without a bounding box into E2E-FT [1], respectively. It can be observed that the quality of estimated surface normals is effectively improved after adding the bounding box. However, its MAE value is still higher than ours, both in synthetic data BUDDHA and real data PINEAPPLE. We attribute this phenomenon to the fact that E2E-FT [1] has not zoomed pixels during training, making the algorithm less sensitive to the presence of bounding boxes.

TABLE VIII  
SENSITIVITY STUDY ON PROMPT VARIANTS. WE REPORT THE AVERAGE VALUES ON EVALFLASH AND EVALFLASH-SYNTH.
<table><tr><td>Prompt variant</td><td colspan="3">EvalFlash</td><td colspan="3">EvalFlash-synth</td></tr><tr><td></td><td>MAE</td><td>RMSE</td><td>MSE</td><td>MAE</td><td>RMSE</td><td>MSE</td></tr><tr><td>w/o “DSLR”</td><td>15.16</td><td>0.31</td><td>0.03</td><td>15.73</td><td>0.34</td><td>0.04</td></tr><tr><td>w/o &quot;detailed&quot;</td><td>15.21</td><td>0.32</td><td>0.03</td><td>15.66</td><td>0.34</td><td>0.05</td></tr><tr><td>full prompt (used in paper)</td><td>14.96</td><td>0.31</td><td>0.03</td><td>15.47</td><td>0.33</td><td>0.04</td></tr></table>

## H. Robustness analyses

a) Robustness analyses w.r.t. different camera setups: In real-world image acquisition, the camera focal length f and camera-to-object distance z could vary dramatically. To assess the robustness of FlashNormal under these conditions, we conduct tests on images rendered with fixed z but varying f, and fixed f but varying z. As shown in Fig. 15, E2E-FT [1] presents blurred predictions when the valid surface area is small, resulting in higher MAE errors. By contrast, our zoomed pixels strategy maintains informative inputs across different values of f and z, enabling consistently high-quality surface normal estimations.

b) Robustness analyses w.r.t. varying illuminations & color temperatures: In real-world scenarios, illumination conditions can vary dramatically. We here show the robustness analyses w.r.t. varying illuminations & color temperatures. As shown in Fig. 16, synthetic (top) and real-world (bottom) experimental results show that our proposal can consistently estimate reliable surface normals under ambient light with varying intensities and color temperatures, even under the case of dark and flat illuminations. These results demonstrate the robustness and applicability of our method.

c) Automatic ROI acquisition with SAM 2 [40]: We further add practical experiments on automatic ROI acquisition using SAM 2 [40]. As shown in Fig. 17, five representative cases, including three single-object examples and two multiobject scenes, demonstrate that the predicted masks can be converted into valid ROIs and lead to coherent surface normal predictions. These examples also show that the method can handle multi-object scenes when the target regions are segmented correctly, while mask failures may still reduce robustness in cluttered cases.

## I. Suiting FlashNormal to single-image input

While our proposal in general assumes a pair of flash/noflash images as input, it can still work reasonably well in case of a single ambient-light image. As shown in Fig. 18, given an ambient-light image and its algorithmically delighted version via StableDelight [52], our proposal can still estimate highquality surface normal maps.

f=24mm  
f=85mm  
z=10m  
z=3m  
![](images/390af32904dcca68fd3ed3d6f1589db3ed7f0b9d37a20378e51f392cd7f99839.jpg)

![](images/c728e87b168c56218ecec275541171334613de68ce93ea6f0d9ed9d6cf39003f.jpg)

![](images/c3f6f16ac76b549ce8e4b022638b975b85dcf177d07d183561853b7c89a3df2a.jpg)  
Fig. 15. Analysis on the robustness against varying focal length f and camera-to-object distance z. The top and middle four rows show the image observations and estimated surface normal maps. Valid regions are resized to the same resolution. The bottom row summarizes the normal estimation errors measured by MAE under varying f and z.

J. Application: Multi-view normal integration for 3D reconstruction

Beyond single-view surface normal estimation, our method excels in multi-view normal integration for complete 3D reconstruction. As shown in Fig. 19, we apply our method to two real-world objects, SHAKESPEARE and MONKEY to estimate surface normal maps of 17 distinct views. These maps are integrated using SuperNormal [3], yielding detailed, geometrically consistent meshes. Compared to E2E-FT [1], for SHAKESPEARE, our method produces richer geometric features and a more complete reconstruction of its head. Additionally, using the same method as EvalFlash, we generate 17 GT normal maps and compute an average multi-view MAE of 17.62, outperforming E2E-FT [1] (19.51) by 9.7%. Similarly, for MONKEY, our estimated surface normals maintain greater detail and geometric consistency across views, yielding reconstructed meshes with richer geometric features compared to E2E-FT [1], whose results are smoother. We also compute the average multi-view MAE for MONKEY, achieving an MAE of 12.86, compared to E2E-FT [1]’s 17.83, reflecting a 27.8% improvement. These results demonstrate the superior geometric consistency, stability, and overall quality of 3D reconstruction produced by our method.

![](images/56aeddb39cf989ba3b31ef2fd7b2a27535cf07c971120810e99d7a77c3d30787.jpg)  
Fig. 16. Analysis on the robustness w.r.t. varying illuminations & color temperatures. The top three rows show the synthetic experimental results and bottom two rows show the real-world experimental results.

## VI. LIMITATIONS AND FUTURE WORKS

Although our proposal can predict surface normals with rich details and reduce shape-reflectance ambiguity in general, it still struggles in certain extreme cases. For example, as shown in the top row of Fig. 20, our method shows substandard performance when dealing with transparent objects due to their unique reflective properties, which make it difficult to clearly distinguish the surface of the object. Additionally, the relatively low number of transparent objects in our training dataset limits the prior knowledge our method has about the GT surface normals of such objects. Furthermore, as shown in the bottom row of Fig. 20, given flat objects with highly complex and realistic textures, our approach tends to misinterpret them as 3D. This occurs because, after introducing flash input, the shading variations in textured areas become minimal, making it difficult to distinguish textures from actual surface structures. In future work, we aim to address these limitations by augmenting the training dataset with more diverse data.

![](images/a295bcfa210fe4cc7f48686be927db20ca692d381ff88e4c1ca4340982fa6e27.jpg)

Fig. 17. Representative ROI extraction and surface normal prediction results using SAM 2 [40]. The first three rows are single-object cases, and the last two rows are multi-object cases.  
![](images/23af5d889d2d3698a2b768b693fba829a9260f8737bd29dbb8ccd7a78b8b8e40.jpg)  
Fig. 18. Qualitative result using an in-the-wild image processed by StableDelight [52].

In the future, we will explore surface normal estimation for objects with special materials, as discussed earlier. We also plan to extend the flash/no-flash setup to scene-level applications, and study its effects in complex environments. Additionally, we will investigate how to leverage this setup to improve the quality and temporal consistency of surface normal estimation in dynamic scenes.

## VII. CONCLUSION

We introduce FlashNormal, a robust and practical surface normal estimator leveraging flash/no-flash image pairs. By integrating diffusion priors, curvature-guided geometry enhancement, and the zoomed-pixels strategy, FlashNormal achieves fine-grained normal estimation while effectively mitigating shape-reflectance ambiguity. To support training and evaluation, we provide Flash100K, a large-scale photorealistic dataset, and EvalFlash, the first real-world dataset tailored to flash/no-flash-based surface normal estimation. Extensive experiments demonstrate that FlashNormal outperforms stateof-the-art methods, delivering significant improvements in surface normal estimation accuracy and downstream multiview 3D reconstruction quality.

![](images/0d63f4eeb92acc9a0682c9153da5499ee8b6c41d5edd84dfa8aa82c3f85aedc3.jpg)

![](images/6badba5c16803c5a3c96eca1b1e4f7c7742195f1906ae9d11dfa133eb1048d35.jpg)  
Fig. 19. 3D reconstruction via integrating multi-view surface normal maps estimated by E2E-FT [1] and ours. The 3D mesh visualizations are taken as snapshots.  
Fig. 20. Failure cases from our proposal. Red boxes highlight areas where surface normals are incorrect.

## ACKNOWLEDGMENT

This work was supported by Beijing Major Science and Technology Project No. Z251100007125021, Hebei Natural Science Foundation Project No. 242Q0101Z, Beijing-Tianjin-Hebei Basic Research Funding Program No. F2024502017, and the National Natural Science Foundation of China (Grant Nos. 62472044, U24B20155, and U23B2052).

## REFERENCES

[1] G. Martin Garcia, K. Abou Zeid, C. Schmidt, D. de Geus, A. Hermans, and B. Leibe, “Fine-tuning image-conditional diffusion models is easier than you think,” in WACV, 2025.

[2] Z. Kuang, K. Olszewski, M. Chai, Z. Huang, P. Achlioptas, and S. Tulyakov, “Neroic: Neural rendering of objects from online image collections,” ACM Trans. Graph., vol. 41, no. 4, pp. 1–12, 2022.

[3] C. Xu and T. Takafumi, “SuperNormal: Neural surface reconstruction via multi-view normal integration,” in CVPR, 2024, pp. 20 581–20 590.

[4] J. Wang, P. Wang, X. Long, C. Theobalt, T. Komura, L. Liu, and W. Wang, “Neuris: Neural reconstruction of indoor scenes using normal priors,” in ECCV, 2022, pp. 139–155.

[5] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, “MonoSDF: Exploring monocular geometric cues for neural implicit surface reconstruction,” in NeurIPS, vol. 35, 2022, pp. 25 018–25 032.

[6] X. Zhu, R. Yi, X. Wen, C. Zhu, and K. Xu, “Relighting scenes with object insertions in neural radiance fields,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 7, pp. 6787–6802, 2025.

[7] J. Ding, Y. He, B. Yuan, Z. Yuan, P. Zhou, J. Yu, and X. Lou, “Ray reordering for hardware-accelerated neural volume rendering,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 11, pp. 11 413–11 422, 2024.

[8] J. Zhang, Y. Zheng, Z. Li, Q. Dai, and X. Yuan, “Gbr: Generative bundle refinement for high-fidelity gaussian splatting with enhanced mesh reconstruction,” IEEE Trans. Circuits Syst. Video Technol., 2025.

[9] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3d gaussian splatting: Survey, technologies, challenges, and opportunities,” IEEE Trans. Circuits Syst. Video Technol., 2025.

[10] B. Shi, Z. Wu, Z. Mo, D. Duan, S.-K. Yeung, and P. Tan, “A benchmark dataset and evaluation for non-lambertian and uncalibrated photometric stereo,” in CVPR, 2016, pp. 3707–3716.

[11] S. Ikehata, “CNN-PS: Cnn-based photometric stereo for general nonconvex surfaces,” in ECCV, 2018, pp. 3–18.

[12] G. Chen, K. Han, and K.-Y. K. Wong, “PS-FCN: A flexible learning framework for photometric stereo,” in ECCV, 2018, pp. 8–14.

[13] S. Ikehata, “Universal photometric stereo network using global lighting contexts,” in CVPR, 2022, pp. 12 591–12 600.

[14] S. Ikehata, “Scalable, detailed and mask-free universal photometric stereo,” in CVPR, 2023, pp. 13 198–13 207.

[15] X. Fu, W. Yin, M. Hu, K. Wang, Y. Ma, P. Tan, S. Shen, D. Lin, and X. Long, “GeoWizard: Unleashing the diffusion priors for 3D geometry estimation from a single image,” in ECCV, 2024, pp. 241–258.

[16] C. Ye, L. Qiu, X. Gu, Q. Zuo, Y. Wu, Z. Dong, L. Bo, Y. Xiu, and X. Han, “Stablenormal: Reducing diffusion variance for stable and sharp normal,” ACM Trans. Graph., vol. 43, pp. 1 – 18, 2024.

[17] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and C. Shen, “Metric3D: Towards zero-shot metric 3d prediction from a single image,” in ICCV, 2023, pp. 9043–9053.

[18] L. Qiu, G. Chen, X. Gu, Q. zuo, M. Xu, Y. Wu, W. Yuan, Z. Dong, L. Bo, and X. Han, “RichDreamer: A generalizable normal-depth diffusion model for detail richness in Text-to-3D,” in CVPR, 2024, pp. 9914– 9925.

[19] X. Long, Y.-C. Guo, C. Lin, Y. Liu, Z. Dou, L. Liu, Y. Ma, S.-H. Zhang, M. Habermann, C. Theobalt et al., “Wonder3d: Single image to 3d using cross-domain diffusion,” in CVPR, 2024, pp. 9970–9980.

[20] M. Boss, V. Jampani, K. Kim, H. P. A. Lensch, and J. Kautz, “Twoshot spatially-varying brdf and shape estimation,” in CVPR, 2020, pp. 3982–3991.

[21] X. Cao, M. Waechter, B. Shi, Y. Gao, B. Zheng, and Y. Matsushita, “Stereoscopic flash and no-flash photography for shape and albedo recovery,” in CVPR, 2020, pp. 3430–3439.

[22] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10 684–10 695.

[23] J. Xu, Y. Li, and S. Wang, “Adazoom: Adaptive zoom network for multiscale object detection in large scenes,” arXiv preprint arXiv:2106.10409, 2021.

[24] B. S. Mahyar Najibi and L. S. Davis, “Autofocus: Efficient multi-scale inference,” in ICCV, 2019, pp. 9745–9755.

[25] Z. Li, Z. Xu, R. Ramamoorthi, K. Sunkavalli, and M. Chandraker, “Learning to reconstruct shape and spatially-varying reflectance from a single image,” ACM Trans. Graph., vol. 37, pp. 1 – 11, 2018.

[26] S. Sang and M. Chandraker, “Single-shot neural relighting and SVBRDF estimation,” in ECCV, 2020, pp. 85–101.

[27] A. Tiwari, S. Ikehata, and S. Raman, “MERLiN: Single-shot material estimation and relighting for photometric stereo,” in ECCV, 2024, pp. 251–269.

[28] G. Xu, Y. Ge, M. Liu, C. Fan, K. Xie, Z. Zhao, H. Chen, and C. Shen, “Diffusion models trained with large data are transferable visual models,” arXiv preprint arXiv:2403.06090, 2024.

[29] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Trans. Pattern Anal. Mach. Intell., 2024.

[30] R. J. Woodham, “Photometric method for determining surface orientation from multiple images,” Opt. Eng., vol. 19, no. 1, pp. 139–144, 1980.

[31] Y. Ju, M. Jian, C. Wang, C. Zhang, J. Dong, and K.-M. Lam, “Estimating high-resolution surface normals via low-resolution photometric stereo images,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 4, pp. 2512–2524, 2023.

[32] T. Luo, J. Shen, and X. Li, “Accurate normal and reflectance recovery using energy optimization,” IEEE Trans. Circuits Syst. Video Technol., vol. 25, no. 2, pp. 212–224, 2014.

[33] Y. Li, Q. Hu, Z. Ouyang, and S. Shen, “Neural reflectance decomposition under dynamic point light,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 4, pp. 2195–2208, 2023.

[34] W. Meng, H. Han, X. Lu, Y. Yin, G. Pan, and Q. Zheng, “Lac-ps: A light direction selection policy under the accuracy constraint for photometric stereo,” IEEE Trans. Circuits Syst. Video Technol., 2025.

[35] Q. Zheng, Y. Jia, B. Shi, X. Jiang, L.-Y. Duan, and A. C. Kot, “SPLINE-Net: Sparse photometric stereo through lighting interpolation and normal estimation networks,” in ICCV, 2019, pp. 8549–8558.

[36] H. Guo, Z. Mo, B. Shi, F. Lu, S.-K. Yeung, P. Tan, and Y. Matsushita, “Patch-based uncalibrated photometric stereo under natural illumination,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 7809– 7823, 2021.

[37] Z. Xia, J. Lawrence, and S. Achar, “A dark flash normal camera,” in ICCV, 2021, pp. 2430–2439.

[38] C. Li, T. Ngo, and H. Nagahara, “Inverse rendering of translucent objects using physical and neural renderers,” in CVPR, 2023, pp. 12 510–12 520.

[39] M. Heep and E. Zell, “An adaptive screen-space meshing approach for normal integration,” in ECCV, 2024, pp. 445–461.

[40] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. Alwala,¨ N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer,´ “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.

[41] Q. Zuo, X. Gu, Y. Dong, Z. Zhao, W. Yuan, L. Qiu, L. Bo, and Z. Dong, “High-fidelity 3d textured shapes generation by sparse encoding and adversarial decoding,” in ECCV. Springer, 2024, pp. 52–69.

[42] B. O. Community, Blender - a 3D modelling and rendering package, Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. [Online]. Available: http://www.blender.org

[43] Y. Aksoy, C. Kim, P. Kellnhofer, S. Paris, M. Elgharib, M. Pollefeys, and W. Matusik, “A dataset of flash and ambient illumination pairs from the crowd,” in ECCV, 2018, pp. 119–135.

[44] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2019.

[45] S. Gugger, L. Debut, T. Wolf, P. Schmid, Z. Mueller, S. Mangrulkar, M. Sun, and B. Bossan, “Accelerate: Training and inference at scale made simple, efficient and adaptable.” https://github.com/huggingface/ accelerate, 2022.

[46] B. Lefaudeux, F. Massa, D. Liskovich, W. Xiong, V. Caggiano, S. Naren, M. Xu, J. Hu, M. Tintore, S. Zhang, P. Labatut, D. Haziza, L. Wehrstedt, J. Reizenstein, and G. Sizov, “xformers: A modular and hackable transformer modelling library,” https://github.com/ facebookresearch/xformers, 2022.

[47] X. Cao, H. Santo, B. Shi, F. Okura, and Y. Matsushita, “Bilateral normal integration,” in ECCV, 2022, pp. 552–567.

[48] L. Zhang, Z. Wang, Q. Zhang, Q. Qiu, A. Pang, H. Jiang, W. Yang, L. Xu, and J. Yu, “Clay: A controllable large-scale generative model for creating high-quality 3d assets,” ACM Trans. Graph., vol. 43, no. 4, pp. 1–20, 2024.

[49] M. Liu, R. Shi, L. Chen, Z. Zhang, C. Xu, X. Wei, H. Chen, C. Zeng, J. Gu, and H. Su, “One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion,” in CVPR, 2024, pp. 10 072–10 083.

[50] Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, D. Liu, F. Liu, K. Sunkavalli, T. Bui, and H. Tan, “Lrm: Large reconstruction model for single image to 3d,” arXiv preprint arXiv:2311.04400, 2023.

[51] P. Wang, H. Tan, S. Bi, Y. Xu, F. Luan, K. Sunkavalli, W. Wang, Z. Xu, and K. Zhang, “Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction,” arXiv preprint arXiv:2311.12024, 2023.

[52] Stable-X, “StableDelight: Revealing Hidden Textures by Removing Specular Reflections,” GitHub repository, available: https://github.com/ Stable-X/StableDelight. Accessed: 2026-04-22.

![](images/4cf5f91ac00c74309f985236339ecf7731a1b5e312168249fc9de4d4da5705ea.jpg)  
Ruiyang Chen received the B.S degree in computer science and technology from Beijing University of Technology, Beijing, China, in 2025. He is currently pursuing the M.S. degree with the College of Artificial Intelligence, Beijing University of Posts and Telecommunications. His research interests include deep learning and computer vision.

![](images/4c1e1f0026e0fd6b256a49e128af59dfc760c82ed9d31245436442583af1742a.jpg)

Feiran Li received his Ph.D. in Information Science from Osaka University in 2023. Before that, He obtained M.Eng. in Robotics from Nara Institute of Science & Technology in 2019, and B.Eng. in Mechanical Engineering from East China University of Science & Technology in 2017. His research interests lie in computational photography and geometric computer vision.

![](images/04ab3069b96490eb6df4f4233fd08a22fdc433d86796e1bc43836d5b84a0b2bf.jpg)

Heng Guo (Member, IEEE) received the BE and ME degrees from University of Electronic Science and Technology, and the PhD degree from Osaka University, in 2015, 2018, and 2022. He is currently a specially-appointed research Professor at Beijing University of Posts and Telecommunications (BUPT). Before joining BUPT, he was a specially-appointed assistant professor at Osaka University from 2022 to 2023. His research interest includes computational photography, physics-based computer vision, and 3D reconstruction.

![](images/0e17240de6e05cfde4476daad67e8ba41fb3bdd78dca824e6fdc00cf9b8237b5.jpg)

Zhanyu Ma (Senior Member, IEEE) received the Ph.D. degree in electrical engineering from the KTH Royal Institute of Technology, Sweden, in 2011. He has been a Professor with Beijing University of Posts and Telecommunications, Beijing, China, since 2019. From 2012 to 2013, he was a Post-Doctoral Research Fellow with the School of Electrical Engineering, KTH. He was an Associate Professor with Beijing University of Posts and Telecommunications, Beijing, China, from 2014 to 2019. His research interests include pattern recognition and machine learning fundamentals, with a focus on applications in computer vision and multimedia signal processing.