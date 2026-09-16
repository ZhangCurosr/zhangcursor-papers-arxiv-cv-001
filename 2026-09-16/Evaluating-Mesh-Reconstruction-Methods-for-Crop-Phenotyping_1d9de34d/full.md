# Evaluating Mesh Reconstruction Methods for Crop Phenotyping

Karanvir Singh<sup>a</sup>, Theo Morales<sup>b</sup>, Binh-Son Hua<sup>b</sup> and Mukesh Saini<sup>a</sup>

<sup>a</sup>Indian Institute ofTechnology Ropar, Rupnagar 140001, Punjab, India

<sup>b</sup>Trinity College Dublin, The University of Dublin, College Green D02 PN40, Dublin 2, Ireland

## A R T I C L E I N F O

Keywords:   
3D Mesh Reconstruction   
Neural Radiance Fields   
3D Gaussian Splatting   
Agriculture   
Cauliflower   
Phenotyping

## A BS T R AC T

Phenotyping an agricultural crop is crucial for studying its entire life cycle, as it provides vital insights to improve yield and, ultimately, food production. Doing the same for crops grown on remote sites is a challenge for the specialists who cannot be available on-site. 3D reconstruction techniques ofer a promising solution to this problem by enabling crop digitization, allowing specialists to access the resulting 3D crop models from anywhere at any time. In this work, we evaluate recent 3D reconstruction pipelines for crop phenotyping. We focus on 7 mesh reconstruction pipelines and measure the fidelity and consistency of their outputs qualitatively and quantitatively. Our results suggest that the meshes produced by the GGGS, PGSR, and 2DGS are preferable to the other pipelines, owing to their quantitative metrics and visually pleasing outputs. The GGGS pipeline is better than the second-best pipeline (2DGS) by about 27% on the radar chart with 5 dimensions, namely, User ratings, Chamfer distance, LPIPS, PSNR, and SSIM.

## 1. Introduction

Food production is of utmost importance for human survival and prosperity, as it enables humankind to persevere and flourish. The sustainability of food production depends on crop yield, which in turn is afected by crop health. Phenotypes are the observable properties of a crop at diferent stages of its life cycle, which can help to deduce crop health. The process of identifying these phenotypes is called Crop phenotyping. Many applications, such as breeding, crop management, and crop processing [30], depend on the crop phenotyping process for their success.

Visual analysis is one of the foundational techniques for phenotyping, in which experts manually inspect crops for leaf color, size, and shape, flowers, or signs of disease. However, it is often dificult for experts to be physically available on-site to perform such inspections. Hence, there is a strong need to accurately digitize the crops in a format that is eficient to work with, even in remote settings. A 3D triangle mesh is a widely adopted format for representing and visualizing digitized scenes or objects. It also allows editing, animating, and relighting the digitized scenes [15], which is helpful for crop phenotyping and growth models [33]. The process of obtaining a 3D mesh from multiple 2D images of a scene or an object is called 3D reconstruction. Traditionally, 3D reconstruction was done using traditional computer visionbased approaches. Recently, there has been a shift from traditional to modern deep-learning-based techniques [23].

Despite the maturity of this field, its application in the crop phenotyping domain remains non-trivial, as the phenotyping process requires meshes to have clear photometric details to represent the crop in the real world with high fidelity. These details include all morphological elements, such as color, leaf count, and all parts of the crop’s shoot system.

Furthermore, these 3D meshes should also be eficient to load, render, and store for the experts working at their workstations remotely.

Although there are studies that explore the traditional [41] and modern [46] 3D reconstruction pipelines from an agritech perspective in an exhaustive manner. But, to the best of our knowledge, there is no such study that experimentally compares 3D mesh reconstruction pipelines in agricultural settings. In this work, we provide a comparative analysis of 3D mesh reconstruction pipelines for crop phenotyping. On exploring the current literature for the reconstruction pipelines which output 3D triangle meshes, the following pipelines are selected for evaluation: Alicevision [14], SuGaR [15], 3DGS-to-PC [36], NeRF2Mesh [37], 2DGS [17], PGSR [4], and GGGS [49]. We also publicly release our Cauliflower-13 dataset, which was briefly explored in our preliminary work [34]. This dataset captures a growing cauliflower plant for 13 consecutive days. Each sample for the day consists of RGB images captured from 120 camera angles, consistently maintained using a photogrammetry setup. Furthermore, we perform quantitative analysis of the pipelines using 4 metrics: the Chamfer distance [13], PSNR [10], SSIM [29], and LPIPS [50]. We also validate them qualitatively through a user study in which we ask questions about the perceived realism of leaf veins, leaf edges, stems, petioles (branches), and soil. We find that the GGGS pipeline is the best out of the seven, beating the second-best, 2DGS, by 27% on the radar chart.

In summary, our contributions are:

• A new dataset for benchmarking 3D reconstruction pipelines for crop phenotyping;

• An up-to-date evaluation study of traditional and stateof-the-art methods for mesh reconstruction on crop phenotyping data;

• An evaluation protocol with documentation to support future research and evaluations in this domain.

## 2. Background

3D reconstruction pipelines can be classified into Traditional and Deep learning-based [46]. Traditional pipelines rely on geometry and image processing techniques, which involve sub-processes such as camera calibration, camera pose estimation, feature matching, and geometric calculations before reconstructing the final 3D scene. Deep learning pipelines learn the mappings from input images to 3D scenes directly, without considering the above-mentioned intermediates. But recently, a new paradigm, Gaussian Splatting, has emerged, which is based on the intersection of traditional and deep learning paradigms.

## 2.1. Traditional Pipelines

Traditional pipelines can be further classified into active and passive pipelines. The active pipelines use real-time sensors that emit a signal to measure the position of the target surface as a collection of relative depth values, which in turn is used to reconstruct the actual target in the real world. Kinect [51], Lidar [3], and Laser Scanning [1] are some of the prominent traditional active pipelines. Conversely, passive pipelines involve capturing images via a single or multiple cameras, which are then processed by 3D computer vision techniques to reconstruct the 3D surface. Structure from Motion (SfM) [31], Multi-View-Stereo (MVS) [35], COLMAP [32], and Alicevision-Meshroom [14] are some of the major traditional passive pipelines. Active pipelines are better than passive ones in terms of accuracy and real-time outputs, but they are very expensive due to their complex hardware requirements. Whereas the passive pipelines do not require the emission of any signals, as a benefit, there is no interference from the environment. All the traditional pipelines always output the 3D scene as either a point cloud or a mesh.

## 2.2. Deep Learning Pipelines

With the dawn of the deep learning era comes another 3D representation, that is, the radiance fields, notably Neural Radiance Fields (NeRF [26]). NeRF attempts to learn a radiance field via a neural network to include the light propagation models in the output scene. These models capture the advanced view-dependent lighting efects such as specular highlights, reflections, and refraction, which are not possible in traditional pipelines. However, NeRF-based pipelines sufer from long training times and slow rendering speeds. Plenoxels [11] addresses this problem by using a voxel grid instead of the neural network, resulting in two orders of magnitude faster training time but at the cost of some memory and fidelity loss. Instant Neural Graphics Primitives (NGP [27]) address the same problem by considering a smaller neural network along with a hash table. It attains training times even lower than Plenoxels. Within the same paradigm, recent foundational models called Image-to-3D have emerged, which generate a 3D mesh from a single image. Sam3D [38] and Trellis2 [43] are some of the notable methods in this sub-domain. These models are trained on massive datasets of millions of 2D images with their corresponding 3D shapes. An overview of both the abovementioned paradigms is shown in Figure 1.

![](images/74104359f027c541d3e1f77e9f8f7f507b7475c4b94dd3539028862ab9cd3621.jpg)  
Figure 1: A basic classification of 3D Reconstruction Pipelines into Traditional and Deep learning Pipelines

## 2.3. 3D Gaussian Splatting - A middle way.

3D Gaussian Splatting (3DGS), discovered recently [20], is a point-based rendering method. It is based on elements from both of the above paradigms. It uses Gaussian primitives as a base for further processing, where these primitives are initialized by the traditional pipelines (SfM [32] in the vanilla version). Each primitive has parameters that are optimized via deep-learning-based approaches to reduce a loss function, which is a weighted sum of two well-known terms, $L _ { 1 }$ and ���� [52], between the rasterized and the input images (training set). The parameter set consists of a position in space as the mean (�) of that Gaussian primitive, and a covariance matrix (Σ), where Σ is composed of the scale (�) and the rotation orientation (�) of the same primitive. There is another parameter, opacity (�), which describes the amount of light transmission through the primitive. Furthermore, the color appearance when viewed from diferent directions is controlled by the spherical harmonics (��) parameter. Traditional representations (point cloud, mesh) could be easily rendered via the traditional graphics pipelines as implemented in tools like Blender [8], Meshlab [6], and Cloudcompare [7]. However, for visualizing 3DGS scenes, a specialized parallel rasterization algorithm, such as SIBR Core [2], is used.

## 3. 3D Mesh Reconstruction Pipelines

The above-discussed paradigms reconstruct 3D scenes as diferent outputs, namely point clouds, meshes, radiance fields, and splats. But as discussed earlier, our work considers mesh representations only, due to their widespread adoption and other application-oriented capabilities. This mesh preference has inspired recent NeRF and Gaussian-splatting-based works to produce mesh representations of 3D scenes rather than their usual outputs. After exploring the current literature for suitable works that output a mesh, we consider seven 3D reconstruction pipelines for comparison as explained below. An overview of these pipelines is presented in Table 1.

Table 1  
3D Mesh Reconstruction Pipelines Considered along with the observations made after the experiments. The preference ranks were decided from the radar chart, as in Figure 8.
<table><tr><td rowspan=1 colspan=1>Pipeline</td><td rowspan=1 colspan=1>Year</td><td rowspan=1 colspan=1>Based On</td><td rowspan=1 colspan=1>Meshing</td><td rowspan=1 colspan=1>Textured</td><td rowspan=1 colspan=1>Preference</td><td rowspan=1 colspan=1>Mentionable Defects</td></tr><tr><td rowspan=1 colspan=1>Alicevision Meshroom [14]</td><td rowspan=1 colspan=1>2021</td><td rowspan=1 colspan=1>Photogrammetry</td><td rowspan=1 colspan=1>Delaunay</td><td rowspan=1 colspan=1> $\curlyvee \mathtt { e s }$ </td><td rowspan=1 colspan=1> $\overline { { { 4 ^ { t h } } } }$ </td><td rowspan=1 colspan=1>Poor Leaf Edges</td></tr><tr><td rowspan=1 colspan=1>3DGS-to-PC [36]</td><td rowspan=1 colspan=1>2025</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>Poisson</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> $\overline { { 7 ^ { t h } } }$ </td><td rowspan=1 colspan=1>Poor Mesh</td></tr><tr><td rowspan=1 colspan=1>SuGaR [15]</td><td rowspan=1 colspan=1>2024</td><td rowspan=1 colspan=1>3DGS</td><td rowspan=1 colspan=1>Poisson</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\overline { { 6 ^ { t h } } }$ </td><td rowspan=1 colspan=1>Visible Seams</td></tr><tr><td rowspan=1 colspan=1>2D Gaussian Splatting [17]</td><td rowspan=1 colspan=1>2024</td><td rowspan=1 colspan=1>2DGS</td><td rowspan=1 colspan=1>TSDF</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> ${ \overline { { 2 ^ { n d } } } }$ </td><td rowspan=1 colspan=1>Highly Smooth</td></tr><tr><td rowspan=1 colspan=1>NeRF2Mesh [37]</td><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>Grid-based NeRF</td><td rowspan=1 colspan=1>IMR</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\overline { { 5 ^ { t h } } }$ </td><td rowspan=1 colspan=1>Soil Details missing</td></tr><tr><td rowspan=1 colspan=1>PGSR [4]</td><td rowspan=1 colspan=1>2025</td><td rowspan=1 colspan=1>Planar-based GS</td><td rowspan=1 colspan=1>TSDF</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> $\overline { { 3 ^ { r d } } }$ </td><td rowspan=1 colspan=1>High Chamfer distance</td></tr><tr><td rowspan=1 colspan=1>GGGS [49]</td><td rowspan=1 colspan=1>2026</td><td rowspan=1 colspan=1>Stochastic Solids</td><td rowspan=1 colspan=1>TSDF</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> $\overline { { 1 ^ { s t } } }$ </td><td rowspan=1 colspan=1>Tiny Granular Holes</td></tr></table>

## 3.1. Alicevision Meshroom

Alicevision 3D reconstruction pipeline [14] consists of a sequential set of stages that takes the set of images as input and finally outputs a 3D mesh. The initial stage extracts features in all the images. This pipeline supports SIFT [25], DSP-SIFT [9], and AKAZE [39] features. Then, it proceeds with the feature matching stage, where feature descriptors are matched for all pairs of images. The next stage is the SfM stage, where the already found matches are fused into tracks. A track is considered a candidate for a 3D point if it is visible from multiple views. The group of these tracks is used to solve the camera calibration and generate a sparse 3D representation. The next stage is the depth map stage, which finds the depth of all the pixels associated with the previously calibrated cameras by Semi-Global Matching (SGM) [16]. The depth maps at this stage are of very low resolution, which are then upscaled with a brute force method [14]. Afterward, all the depth maps are used to re-project the depth values as 3D points, which are merged into a single dense point cloud via KD Trees. The next stage extracts the mesh out of the fused points via Delaunay triangulation. Then, the final stage is the texturing stage, in which the UV maps are extracted from the images of certain camera views that provide the best texture for the mesh region corresponding to that particular texture map. As a side note, while executing this pipeline in our case, we also aligned the meshes to the COLMAP sparse point cloud via the Iterative Closest Point (ICP) [6] to attain a comparable orientation with the other pipelines’ meshes.

## 3.2. 3DGS-to-PC

3DGS-to-PC [36] pipeline converts the Gaussian splats of a scene into a very dense point cloud and ultimately into a 3D mesh. One of its benefits is that it doesn’t need any retraining, as done in the other 3D reconstruction pipelines. Its input consists of the COLMAP sparse point cloud and the 3D Gaussian splats [20] of the scene. The pipeline focuses on the sampling of points from the Gaussians, where the number of points sampled from each Gaussian is proportional to its volume. The Mahalanobis distance between the sampled point and the corresponding Gaussian mean is used to ensure that the outliers are not sampled. The color assignment for each of the newly sampled points is also a major highlight of this pipeline. For fast rendering, only the $0 ^ { t h }$ order spherical harmonics are used, which removes view-dependent efects of Gaussian colors. The idea is to balance the loss of these efects by having more newly sampled Gaussians with single color values. Here, the notion of contribution is introduced, which is directly proportional to the product of the opacity and transmittance values of the Gaussians. Each Gaussian would have a maximum contribution value, which gets updated as the training proceeds. The Gaussians that have this maximum contribution value lower than the overall mean contribution value are removed. As a result, the heavily occluded Gaussians get removed, and the Gaussians that are closer to the surface remain. Afterward, the point cloud is cleaned to remove any noisy outliers. Finally, Poisson reconstruction [19] with a Poisson depth of 10 is used to extract the mesh, on which 10 iterations of Laplacian smoothing [40] are further applied to attain the final mesh.

## 3.3. SuGaR

The path from a 3DGS scene to a mesh can also be bridged by another idea of Surface-aligned Gaussian Splatting (SuGaR) [15]. This pipeline’s primary idea is to ensure that the Gaussians output by the original 3D Gaussian Splatting pipeline are aligned to the surface. The input consists of a COLMAP sparse point cloud and the 3D Gaussian splats. There are three stages involved here, namely Regularization, Mesh Extraction, and Joint Refinement.

The first stage has 3 sub-stages, where the 1st sub-stage is basically the optimization of vanilla Gaussians’ properties as described in section 2.3, the 2nd sub-stage ensures that only Gaussians with opacities � > 0.5 remain, and the 3rd sub-stage includes new regularization term based on the signed distance function (SDF) of the surface attained by the updated Gaussians and the ideal SDF (when Gaussians lie on the surface). The second stage extracts a coarse mesh from the output of the first stage. To attain this, �-level set points (�=0.3) along with their normals are computed with the help of depth maps of the Gaussians along the training viewpoints. Then, Poisson reconstruction [19] (depth 10) followed by mesh simplification via quadric error metric [12] is applied to reconstruct a coarse mesh from the computed points and normals. The third stage refines the coarse mesh by binding new Gaussians to its triangle faces. However, these new Gaussians have a smaller number of parameters, like 2 scaling factors instead of 3, and only 1 rotation parameter instead of 2, because they are bound to a 2D triangle, which is planar instead of the original 3D blobs. Finally, after further optimization, the refined textured mesh is extracted from the refined SuGaR model along with the textures of square size 8.

## 3.4. 2D Gaussian Splatting

2D Gaussian Splatting (2DGS) [17] is a novel idea, where the Gaussians are 2D in nature instead of 3D, implying that they are planar now. There are some challenges in surface (mesh) reconstruction from 3DGS because the rasterization of 3DGS lacks multi-view consistency; that is, when a 3D Gaussian is viewed from diferent angles, the corresponding projections are made on diferent intersection planes, which is not the case when the Gaussian is a 2D disk. Furthermore, the 3DGS representation doesn’t consider surface normals, which are necessary for accurate 3D surface reconstruction.

The input to the 2DGS pipeline is the sparse point cloud from COLMAP. This pipeline adds two new regularization terms to the usual photometric loss $( L _ { 1 }$ and ����). One of them is about depth distortion, which ensures that the 2D Gaussians along each ray are concentrated at a specific optimized distance. The other term is for normal consistency, which ensures that the 2D Gaussians are locally aligned with the surface, which is done by aligning the 2D splats’ normals with the normals from the depth maps. Apart from that, the same adaptive control strategy from 3DGS is used to increase the number of Gaussians. For mesh extraction, the optimized 2D splats are used to render the depth maps of the training data itself. Later, these depth maps are fused by Truncated Signed Distance Fusion (TSDF) using Open3D [53].

## 3.5. NeRF2Mesh

NeRF2Mesh [37], being a NeRF-based approach, extracts a textured mesh from the multi-view RGB images in a two-stage process. The first stage’s main task is to learn the Geometry and Appearance. A shallow Multi-layerperceptron (MLP) is used to learn the Geometry Density Grid. The Appearance is decomposed into difuse color $( c _ { d } )$ and specular color $( c _ { s } )$ . Difuse color is the solid component, which is independent of any viewing directions, whereas the specular color is the shiny component, which depends upon the viewing directions. Both of these components are learned via two diferent MLPs. The $c _ { d }$ being fixed can directly be converted into an RGB texture image, whereas $c _ { s } ,$ , being a function of viewing direction, requires a fragment shader to render it. The original NeRF loss function is enhanced by adding an L2 regularization term for $c _ { s }$ along with the entropy regularization on the rendering weights. The Marching Cubes [24] algorithm is applied on the converged Geometry Density Grid to acquire a coarse mesh, which is further refined in the second stage.

In the second stage, the appearance and geometry are optimized jointly along with the vertex positions and face densities, alias iterative mesh refinement (IMR) via diferential rendering [22]. After convergence, the geometry is refined. But the appearance is still in the color grid. Hence, the UV coordinates of the refined mesh are unwrapped. After which, the $c _ { d }$ and $c _ { s }$ are baked onto that as two diferent texture images $I _ { d }$ and $I _ { s } .$ . Additionally, during the execution of this pipeline on our dataset, the input images were down-sampled by a factor of 2 for faster convergence. Consequently, the final meshes were re-scaled and re-aligned to the COLMAP sparse point cloud via the ICP algorithm to attain comparable size and orientation.

## 3.6. PGSR

Planar-based Gaussian Splatting for Eficient and Highfidelity Surface Reconstruction (PGSR) focuses specifically on the depth accuracy and global geometry preservation during the reconstruction process. It flattens each of the 3D Gaussians along the axis for which the scaling factor is the minimum; the same axis also corresponds to the normal of the corresponding Gaussian. Since these flat Gaussians are fitted onto the surface, the depth maps are more consistent for the surface as compared to the cases where the Gaussians are not treated as planes, as in some previous methods ( [5], [18]).

To fit the Gaussians along the actual surface, the PGSR method optimizes the Gaussians using the loss function, which considers the terms related to flattening, photometry, and geometry. The flattening loss is the $L _ { 1 }$ norm of the minimum scaling factors for all Gaussians. The photometric loss consists of the usual $L _ { 1 }$ loss and $L _ { S S I M }$ on the exposureadjusted rendered images. The geometry loss is further composed of three aspects, namely single-view regularization, multi-view geometric consistency, and multi-view photometric consistency. The single-view regularization terms ensure that the final alpha-blended normal map is similar to the normal map attained via the rendered depth map. The multiview geometric consistency loss considers the Homography matrix, which maps the pixels in one viewpoint to the pixels in another viewpoint (neighboring frame). It ensures that the forward and backward projection errors are minimized. The multi-view photometric consistency ensures that the normalized cross-correlation [45] between the original frame and the neighboring frame is close to 1. Apart from the loss function, it follows the original 3DGS work for the initialization, along with the densification strategy as in AbsGS [44]. It extracts the TSDF field [28] and then performs Marching Cubes [24] over the TSDF field to extract the final mesh.

## 3.7. GGGS

Geometry Grounded Gaussian Splatting (GGGS) considers the Gaussians as stochastic solids whose transmission function is smooth as opposed to the step-wise nature in the case of standard 2D splats, which results in a more accurate geometry reconstruction. Treating them as stochastic solids allows to attain attenuation coeficients inside the Gaussian primitives, which ultimately allows for smooth optimization and accurate depth maps. This treatment is equivalent to the rasterization rendering of the original Gaussians.

The optimization loss function is composed of the usual photometric loss [20], 2DGS consistency loss [17], and multiview regularization loss from the PGSR [4] as mentioned previously. For geometric regularization, the median depth is considered, which is the point on the ray where the transmission value reaches 0.5. This transmission is the product of individual transmission values of the Gaussians along the ray calculated via RaDe-GS [48]. The densification strategy from the Gaussian opacity fields [47] is used. The 3D mesh is extracted via the TSDF fusion as implemented by Open3D [53].

## 4. Evaluation

We evaluated the pipelines’ outputs quantitatively and qualitatively. This section provides details of the dataset, metrics used, quantitative evaluation, user study, and result analysis.

## 4.1. Dataset

Our dataset (‘Cauliflower-13’) consists of a cauliflower crop in a pot as the subject of image acquisition. Images were captured on a day-to-day basis from multiple views. The apparatus included a turntable, a tripod (Amazon Basics 60 Inch), a smartphone camera (Redmi Note 8 Pro), and a white cloth for the background. We captured 120 images for each day, up to a total of 13 days. Basically, we rotated the turntable 5 times at diferent height levels. At each height, 24 images were taken at a diference of $1 5 ^ { \circ } { }$ between each consecutive image at the same height. These points, from where the images were to be taken, were marked on the turntable. The heights were adjusted with the help of the rotatable knob of the tripod, which was always kept at a fixed location marked on the floor. The consistency of image acquisition was maintained throughout all 13 days. All the above-mentioned systematic specifications make this dataset a suitable testbed to evaluate the latest 3D reconstruction methods.

We are releasing our dataset publicly this time, as it was explored briefly in our previous preliminary work [34], where only the Alicevision [14] pipeline was used to create crop assets and visualize them in a Virtual Reality (VR) application.

## 4.2. Metrics

We used 4 metrics for evaluation, namely Chamfer distance, PSNR, SSIM, and LPIPS. The ground truth for Chamfer distance was computed via the COLMAP dense reconstruction [32], whereas for the other 3 metrics, the original images along with their pose information were treated as the ground truth. We discuss details of these metrics below.

Chamfer Distance. Chamfer distance is defined between two sets of point clouds as a measure of their similarity. It is not sensitive to the density of the point cloud unless there are very strong outliers, as it is an averaged metric. Chamfer Distance (CD) can be calculated as in Equation 1, where � and � are the vertices in the point clouds  and , respectively.

$$
C D = \frac { 1 } { \left| \mathcal { A } \right| } \sum _ { \mathbf { a } \in \mathcal { A } } \operatorname* { m i n } _ { \mathbf { b } \in \mathcal { B } } \left| \left| \mathbf { a } - \mathbf { b } \right| \right| _ { 2 } ^ { 2 } + \frac { 1 } { \left| \mathcal { B } \right| } \sum _ { \mathbf { b } \in \mathcal { B } } \operatorname* { m i n } _ { \mathbf { a } \in \mathcal { A } } \left| \left| \mathbf { b } - \mathbf { a } \right| \right| _ { 2 } ^ { 2 }\tag{1}
$$

PSNR. Peak Signal-to-Noise Ratio measures the ratio of the square of the maximum pixel value $( M A X _ { I } )$ to that of the noise intensity, which is calculated as the Mean Squared Error (���) between the individual pixels of the reconstructed image and the ground truth image corresponding to the same pose at log scale in decibels (dB). Then, this ���� value is averaged over all the poses for a sample mesh to attain a final average PSNR metric.

$$
P S N R = 1 0 * l o g _ { 1 0 } \frac { M A X _ { I } ^ { 2 } } { M S E }\tag{2}
$$

SSIM. Structural Similarity Index considers the inter-pixel dependencies, especially for the case of spatially proximate pixels, which carry the information about the structure of the object in the image [42]. It is calculated between two blocks (�, �) of the images as in Equation 3.

$$
S S I M ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + c _ { 1 } ) ( 2 \sigma _ { x y } + c _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + c _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + c _ { 2 } ) }\tag{3}
$$

where $\mu$ and $\sigma$ are the sample mean and sample variance for the corresponding � and � blocks, and $c _ { 1 } , c _ { 2 }$ are the constants determined by the dynamic range of the pixel values of the blocks. Then, this SSIM index is averaged over all the blocks considered in the images to attain the SSIM for the rendered and the ground truth image for a particular pose. Finally, the same is averaged over all the poses to attain the SSIM for the mesh.

LPIPS. Learned Perceptual Image Patch Similarity measures the similarity between two images, which coincides with human judgment [50]. It involves the use of pre-trained backbone networks to extract features from the images to be compared, which are then further used to compute the perceptual distance. We used the AlexNet [21] as the backbone. The images to be compared also go through preprocessing to convert them to tensors of the specific shape as desired by the AlexNet backbone network. The LPIPS metric is calculated between the rendered image and the ground-truth images for each pose. It is basically the weighted diference of the extracted feature maps, where the weights are specific to the backbone network. Finally, the metric is averaged over all the poses to attain the final LPIPS for the mesh.

## 4.3. Quantitative Evaluation

All seven pipelines were fed the same 13 image sets for the corresponding days, yielding 13 meshes for each pipeline. Since the Chamfer distance calculations are concerned with point clouds, the point cloud format, which stores only the vertices of the mesh, was considered against the ground truth generated by dense COLMAP reconstruction. The values corresponding to this metric can be observed in Figure 2. For the calculations of the other three metrics, the final 3D meshes were rendered from the same poses corresponding to those of the original input images, with the ambient lighting of white color and an intensity value of 1. The PSNR, SSIM, and LPIPS values for the pipelines can be observed in Figures 3, 4, and 5, respectively. Lower LPIPS values are better, implying that the final output and ground truth are more similar perceptually.

![](images/431a5c80fabb1fc4bccd1d81a63109ee6ebdbe1acacba08a6ae6788287b10848.jpg)  
Figure 2: Chamfer distance for our Cauliflower-13 dataset 3DGS-to-PC 2DGS SuGaR NeRF2Mesh PGSR GGGS Alicevision. The Chamfer distance in the case of the Alicevision pipeline was the lowest owing to the ICP alignment step that was performed after the dense reconstruction stage to make it comparable to all other COLMAP-based pipelines. Besides, the Chamfer distance of the 2DGS pipeline was lower than all the pipelines except the NeRF2Mesh pipeline for some samples.

![](images/5742325ce653c187de40b63ceb0f694a1554e38b451f959cd764fa9fe7dd159c.jpg)  
Figure 3: PSNR values for our Cauliflower-13 dataset 3DGSto-PC 2DGS SuGaR NeRF2Mesh PGSR GGGS Alicevision. The computed PSNR values were low for all the meshes, considering that the lighting conditions used for rendering the meshes were not exactly similar to those of the real scenes of the photographs. However, from a comparison perspective, the GGGS pipeline has the highest PSNR values, followed by PGSR, 2DGS, and SuGaR.

## 4.4. User Study

We conducted a user study on 32 participants with basic agriculture knowledge. They were asked the following 6 questions, which required ranking the meshes generated from the seven considered 3D reconstruction methods. The meshes were displayed in a web browser with interactive mouse controls to view the meshes from any desired angle. The user could swap the meshes back and forth and decide the rank they want to give for each of the questions below.

![](images/01de0e328aec806d7086a20c2ef11161aa608f16359430e877b8e72b54c887c8.jpg)  
Figure 4: SSIM metric for our Cauliflower-13 dataset 3DGSto-PC 2DGS SuGaR NeRF2Mesh PGSR GGGS Alicevision. The SSIM values for the GGGS pipeline were the highest, followed by 2DGS, PGSR, and SuGaR. The 3DGSto-PC pipeline shows the lowest SSIM values, indicating poor reconstruction based on the structural aspect.

![](images/c8c0a68a20e85227cece05274d13d1b59cbc1ed1efd35b05e0fafc2e30590af4.jpg)  
Figure 5: LPIPS for our Cauliflower-13 dataset 3DGS-to-PC 2DGS SuGaR NeRF2Mesh PGSR GGGS Alicevision. A lower LPIPS score implies better reconstruction. Here, LPIPS corresponding to GGGS were the lowest, implying the best, followed by Alicevision, which was followed by the and PGSR. A lower LPIPS value is considered closer to human perception.

1. Please rank the leaf colors of all the methods in a relative manner from the lowest to the highest. Here, the highest implies the closest to actual images.

2. Please rank the visibility of veins in all the methods in a relative manner from the lowest to the highest. Here, highest implies the veins are visible in the best possible manner.

3. For which of the methods do you see artifacts (noise or defects) at the edges of the leaves? (Yes / No)

![](images/1d2e0123bc0ab95c4115860ffb01857928ba644b39aba1eed40363efa94c3d48.jpg)  
Figure 6: The Average Sentiment scores for each of the considered aspects from the User-study can be observed for all 7 pipelines.

4. Please rank the appearance of the soil details in all the methods in a relative manner from the lowest to the highest. (It may include features like the fallen leaves, pebbles, etcetera). Most details imply the highest rank.

5. Please rank the perceived realism for the shape and appearance of the stem in all the methods in a relative manner from the lowest to the highest. Highest implies maximum realism.

6. Please rank the perceived realism for the shape and appearance of the branches (petioles) in all the methods in a relative manner from the lowest to the highest. Highest implies maximum realism.

The responses were converted into numerical data based on their rankings by mapping the ranks to an integer sentiment. Higher rank implies more positive sentiment towards that method. The highest rank had a value of 7, and the lowest rank had a value of 1. All the other ranking values lie between 1 and 7. For each ranking-related question, each mesh is given a final numerical sentiment $( S e n t i m e n t _ { M e s h } )$ , which is calculated as

$$
S e n t i m e n t _ { M e s h } = \sum _ { \forall i } \frac { R a n k _ { i } } { \# P a r t i c i p a n t s }\tag{4}
$$

where $R a n k _ { i }$ is the numerical value for the rank assigned by the $i _ { t h }$ participant. For the question related to the artifact on the edges, the answers were Yes or No. The majority were naysayers for GGGS, NeRF2Mesh, and SuGaR; that is, they didn’t find the artifacts on the edge of leaves, whereas for other pipelines, the majority of the participants found the artifacts on the leaf edges. Besides, the results of the rankingbased questions of the User study can be observed in Figure 6. The final User Study Rank was calculated by averaging the ranking aspects for all 7 methods, as can be observed in Figure 7.

![](images/3774ce324d04426c9570001eeefd2684e76b393f1606cbef7163edea1e4440c2.jpg)  
Figure 7: The final User Study rank was calculated by considering the 5 ranking-based questions. The overall sentiment is highest for GGGS, followed by PGSR, 2DGS, and Alicevision.

## 4.5. Result Analysis

Further summarizing the qualitative metrics, the Chamfer distance, PSNR, SSIM, and LPIPS were averaged across all 13 samples to obtain a single quantity for each pipeline. Based on this quantity, all the pipelines were arranged in 4 separate ranking orders for the corresponding metrics. Combining them with the User-study final ranking orders, we represented them via 5 dimensions on a radar chart, as in Figure 8. The points on this radar chart are marked as per the rankings instead of the true metric values because the values being on diferent scales are not suitable for direct comparison. For clarity, it can be reiterated that the points further away from the center are higher in preference rankings (or better) as compared to the closer ones. We further computed the area covered by the spread of each pipeline on the radar chart as in Figure 9. The pipeline with the highest area was given the maximum overall preference or the best rank.

![](images/f25f97e03c547c53da477aaf2b5e4956679c99e7d978e86edec6abeacc524889.jpg)  
Figure 8: This radar chart summarizes the preference rankings of the methods based on five aspects: Userstudy, Chamfer Distance, LPIPS, SSIM, and PSNR. Points away from the center are better than the ones closer to the center on the preference scale. It can be seen that 2DGS is still one of the balanced choices. 3DGStoPC SuGaR NeRF2Mesh 2DGS GGGS Alicevision PGSR

![](images/e59809c9d5e69dfddee69747735cf54ae30145386a59c866b51849ebd319a720.jpg)  
Figure 9: Area values computed on the radar chart, which are directly proportional to overall preference rankings (The higher the area, the higher the preference)

![](images/183731c82c23c9a6ae81654ee90205cb11cf4316862f377eebabb116f85935bb.jpg)  
(a) Cleaned Point Cloud

![](images/61c50f637a868d817c103554731d77993d39257e89611e15ffa2108d48b1d55c.jpg)  
(b) Cleaned Mesh  
Figure 10: 3DGS-to-PC Pipeline Example - Day 9

## 4.5.1. Observations

The GGGS pipeline is at the top of the ranking order, followed by 2DGS, PGSR, Alicevision, NeRF2Mesh, SuGaR, and 3DGS-to-PC in the same order. Between 2DGS and PGSR, the Chamfer distance, LPIPS, and SSIM were better for 2DGS, while PGSR had better PSNR and User ratings. Alicevision, being the representative of the traditional photogrammetry pipelines, ensured a low LPIPS value, indicating that the reconstruction was perceptually closer to humans, second to GGGS. NeRF2Mesh was the highly preferred method for accurate leaf colors as per the User Study. The photorealistic details of the veins on the leaves were very clear. SuGaR pipeline was one of the pioneers to perform reconstruction by aligning the Gaussians along a 2D surface, which was mastered by 2DGS later. But SuGaR additionally generates the UV textures, which is not the case with the latest Gaussian-based methods, as they just work with the vertex colors only.

The mesh reconstruction of the 3DGS-to-PC method was poor, but its point cloud reconstruction abilities were good, as can be observed via the Chamfer distance values, which was better than even the GGGS, PGSR, and NeRF2Mesh pipelines. However, the Chamfer distance metric considers the point clouds only, rather than the entire mesh. Consequently, the final mesh in this case was poorly reconstructed. This can be seen in one of the samples (Day-9 of our dataset) as in Figure 10. Hence, this pipeline can be a good candidate for the quick reconstruction of a point cloud from the 3DGS output if the point cloud is the only requirement.

## 4.5.2. Visual Results

The cleaned output meshes, after the removal of floaters from the reconstruction outputs, of the other 6 pipelines except the 3DGS-to-PC can be observed from the front and the top in Figure 12 and Figure 15, respectively. These figures correspond to the cauliflower plant on the same day (Day 9) output by diferent pipelines. Furthermore, the normal maps corresponding to the same can be seen in Figure 13 and Figure 16 for the front and top views, respectively. Similarly, the mesh geometry obtained from all the pipelines can be seen in Figure 14 and Figure 17.

![](images/99884bb2d69f3af09108845947defb1d59e9a3de6ddccf2739c0748b9a4f3e83.jpg)  
(a) Oblique Frontal View

![](images/584e7e73e3de5c941d648bf3e9e4065c069bf823c14159d6dba455fa404d7f83.jpg)  
(b) Approximate Top View  
Figure 11: Day 1 to 13 meshes generated via the 2DGS pipeline. These demonstrate the temporal growth of the cauliflower crop.

Although there were very strong ratings for GGGS in almost all aspects, but it had some caveats too, which were mainly its larger mesh size and a large amount of granular artifacts on the mesh surface. The 2DGS meshes had a lighter size as compared to the GGGS meshes, which made it possible to load all 13 2DGS meshes at one time in a rendering software like Meshlab [6] on a high-end desktop with NVIDIA RTX 4090 32GB VRAM, as can be seen in Figure 11. This lightweight quality of the meshes is highly preferable to the phenotyping experts, as the simultaneous loading of these multiple meshes is helpful to observe the temporal traits and growth of the crop.

## 4.5.3. Visual Defects

Considering the visual results, none of the pipelines outputted the perfect mesh like a real plant. There were some observable defects with every pipeline briefly mentioned in Table 1 also. GGGS had multiple tiny granules on the surface of the whole mesh, as can be seen on magnifying Figures 12 and 15. The mesh surface of the 2DGS and PGSR had contour-like patterns, which are due to the voxel-space-based triangulation procedure involved there. Alicevision had issues with reconstruction around the leaf edges. NeRF2Mesh had clear visible holes and no visual soil details in the pot region, indicating the lack of reconstruction in the peripheral regions. SuGaR’s reconstructed surface had a stitched appearance, as various seams were visible when magnifying the surface. 3DGS-to-PC mesh structure was poor, as can be seen in Figure 10.

## 5. Conclusion

In this study, we have explored the 3D reconstruction prowess of seven pipelines, namely Alicevision Meshroom, 3DGS-to-PC, SuGaR, 2DGS, NeRF2Mesh, PGSR, and GGGS. We have used our unreleased Cauliflower-13 dataset as the pipeline inputs. We evaluated them based on 4 quantitative metrics, namely, Chamfer distance, PSNR, LPIPS, and SSIM. Alongside, we also conducted a user study with a questionnaire curated to the phenotyping aspects of the crops. Experiments on our dataset revealed that recent Gaussian

![](images/51ee686c721bb12c1e79e59a9f2615a7223b807afb59a53dd849030d5e7e390d.jpg)  
(a) Alicevision Meshroom

![](images/e1aac332e331a2d025b138dfba26d453a95e469f0ac1592320d2516bfaacc04f.jpg)  
(b) SuGaR

![](images/9a12171bbf2f969268f94e69b82f3a3c4232420f01849c38e72dca8a465b7a5a.jpg)  
(c) 2D Gaussian Splats

![](images/68a4feaa4c78398602c46a195b01cea715b0085b905457aed4793f86bf8ea56c.jpg)  
(d) NeRF2Mesh

![](images/cc7d66945623400bdf642677812a8eadffe435c0ee2d138a5cb189ba5fe1281e.jpg)  
(e) PGSR

![](images/d72a817d0b9afd91d7635983244c9d69e8840a78a3e9fe603d3bab4e4b360d82.jpg)  
(f) GGGS  
Figure 12: Front View of Day 9 Cauliflower for all the Pipelines

Splatting-based methods have outperformed NeRF-based and traditional photogrammetry-based methods. In particular, the GGGS pipeline is the most preferred, and it beats the secondbest pipeline (2DGS) by around 27% on the radar chart. However, 2DGS is more eficient in terms of storage space, which is also an important factor to consider for temporal phenotyping. Overall, all the considered pipelines had some notable defects, implying there is scope for the perfect 3D reconstruction pipeline. But on an ending note, the ongoing advancements in the field of Gaussian Splatting are still promising as they bridge the gap between the traditional graphics pipelines and the recent rendering-based paradigms, making it beneficial from a practical standpoint of speed and eficiency.

![](images/d7d912546d33174bee2f946d6e670411f70bed1f4e828d6ae36525318471048e.jpg)  
(a) Alicevision Meshroom

![](images/3cdd4ecb6a32e5a12e3c0e30f68be7f61e9fae5bbcb5d221dac6032e33fd1e97.jpg)  
(b) SuGaR

![](images/aee64544434485a666c301060d725cb47e81096f0d1478c74db3b61d2ccd0d90.jpg)  
(c) 2D Gaussian Splats

![](images/ba2d033e75150e1a63f185ff7a7932d7219061a7fff42344739f5ab12bfa29cd.jpg)  
(d) NeRF2Mesh

![](images/1f8412f964b4775df63e6b5696fed5d59b143e0ac31f3c70727d4166488be137.jpg)  
(e) PGSR

![](images/1335d27b77d579d94b80461c6ddaa81baee1c0245b41341604371daba7a9edbd.jpg)  
(f) GGGS  
Figure 13: Normal Maps from the Front View of Day 9 Cauliflower for all the Pipelines.

## Acknowledgements

The work was supported by TIH-AwaDH at IIT Ropar, which is a technology innovation hub for agriculture & water technology development sponsored by DST.

## References

[1] Baltsavias, E.P., 1999. A comparison between photogrammetry and laser scanning. ISPRS Journal of Photogrammetry and Remote Sensing 54, 83–94. URL: https://www.sciencedirect.com/science/article/ pii/S0924271699000143, doi:https://doi.org/10.1016/S0924-2716(99) 00014-3.

[2] Bonopera, S., Esnault, J., Prakash, S., Rodriguez, S., Thonat, T., Benadel, M., Chaurasia, G., Philip, J., Drettakis, G., 2020. sibr: A system for image based rendering. URL: https://gitlab.inria.fr/ sibr/sibr\_core.

[3] Borkowski, A.S., Kubrat, A., 2024. Integration of laser scanning, digital photogrammetry and bim technology: A review and case studies. Eng 5, 2395–2409. URL: https://www.mdpi.com/2673-4117/5/4/125, doi:10.3390/eng5040125.

[4] Chen, D., Li, H., Ye, W., Wang, Y., Xie, W., Zhai, S., Wang, N., Liu, H., Bao, H., Zhang, G., 2025. Pgsr: Planar-based gaussian splatting for eficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics 31, 6100–6111. URL: http://dx. doi.org/10.1109/TVCG.2024.3494046, doi:10.1109/tvcg.2024.3494046.

[5] Cheng, K., Long, X., Yang, K., Yao, Y., Yin, W., Ma, Y., Wang, W., Chen, X., 2024. Gaussianpro: 3d gaussian splatting with progressive propagation. URL: https://arxiv.org/abs/2402.14650, arXiv:2402.14650.

[6] Cignoni, P., Callieri, M., Corsini, M., Dellepiane, M., Ganovelli, F., Ranzuglia, G., 2008. Meshlab: an open-source mesh processing tool, in: Association, T.E. (Ed.), Eurographics Italian Chapter Conference, pp. 129–136. doi:10.2312/localchapterevents/italchap/ italianchapconf2008/129-136.

![](images/366682380ab84b4cb91762ee0597c50bcd9227cbff0182a1373292bc49be9f26.jpg)  
(a) Alicevision Meshroom

![](images/69c844a693d360c762e0908febf364853dcdc51dc6e0ecc906cbbf73602e86c4.jpg)  
(b) SuGaR

![](images/06c2b87bcf586e5084c443a5ea68929064852d87519fe7cc99e38b7406ca984d.jpg)

![](images/1bb4fa69f39c7a4eab9115a86e7183e27e2da34e5413e927dab4843fb87d0d31.jpg)  
(d) NeRF2Mesh

(c) 2D Gaussian Splats  
![](images/1fb97b62dfb6d93345532228e92ba8815c9a716b61de1959fe9f63ff18be269e.jpg)  
(e) PGSR

![](images/189fde05f5b3087e0f0076ca05f8682102e83bdaf294feafc9e1d7cc89ce3e93.jpg)  
(f) GGGS  
Figure 14: Geometrical Front View of Day 9 Cauliflower for all the Pipelines

[7] CloudCompare Project, . CloudCompare - 3D point cloud and mesh processing software. http://www.cloudcompare.org.

[8] Community, B.O., 2018. Blender - a 3D modelling and rendering package. Blender Foundation. Stichting Blender Foundation, Amsterdam. URL: http://www.blender.org.

[9] Dong, J., Soatto, S., 2015. Domain-size pooling in local descriptors: Dsp-sift, in: 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5097–5106. doi:10.1109/CVPR.2015.7299145.

[10] Fardo, F.A., Conforto, V.H., de Oliveira, F.C., Rodrigues, P.S., 2016. A formal evaluation of psnr as quality measurement parameter for image segmentation algorithms. URL: https://arxiv.org/abs/1605.07116, arXiv:1605.07116.

[11] Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A., 2022. Plenoxels: Radiance fields without neural networks, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5491–5500. doi:10.1109/CVPR52688.2022.00542.

[12] Garland, M., Heckbert, P.S., 1997. Surface simplification using quadric error metrics, in: Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, ACM Press/Addison-Wesley Publishing Co., USA. p. 209–216. URL: https://doi.org/10. 1145/258734.258849, doi:10.1145/258734.258849.

[13] Goranci, G., Jiang, S., Kiss, P., Szilagyi, E., Yang, Q., 2025. Fully dynamic algorithms for chamfer distance. URL: https://arxiv.org/ abs/2512.16639, arXiv:2512.16639.

[14] Griwodz, C., Gasparini, S., Calvet, L., Gurdjos, P., Castan, F., Maujean, B., De Lillo, G., Lanthony, Y., 2021. Alicevision meshroom: An open-source 3d reconstruction pipeline, in: Proceedings of the 12th ACM Multimedia Systems Conference, Association for Computing Machinery, New York, NY, USA. p. 241–247. URL: https://doi.org/ 10.1145/3458305.3478443, doi:10.1145/3458305.3478443.

[15] Guédon, A., Lepetit, V., 2024. Sugar: Surface-aligned gaussian splatting for eficient 3d mesh reconstruction and high-quality mesh rendering, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5354–5363. doi:10.1109/CVPR52733. 2024.00512.

![](images/569af2ece2952f7d79601ff81f9fba72cd131545d1275a7e8d4f4d02221685fd.jpg)  
(a) Alicevision Meshroom

![](images/03ecfa44dc3cb35422ec43f3981feccd427dfda4c4bf25a66aafb8837969545b.jpg)  
(b) SuGaR

![](images/a0e56c054640ce47e899e29b6227c6036ae9a6e4124cd5a778f5fec77a6896b3.jpg)

![](images/3f08a18ce207240c02dbbe8cb811829c40c7cb4b1a4c3d14153f7dfefb7d9694.jpg)  
(d) NeRF2Mesh

(c) 2D Gaussian Splats  
![](images/f462bd578117905107dc74a0fd7bbdf6505088de1812cbee5df37fa43fda4b5e.jpg)  
(e) PGSR

![](images/6e2dca60b226cd32cc44101116998a922f11f0432df8c7c64ed90bf90d6cedbd.jpg)  
(f) GGGS  
Figure 15: Top View of Day 9 Cauliflower for all the Pipelines

[16] Hirschmuller, H., 2008. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on Pattern Analysis and Machine Intelligence 30, 328–341. doi:10.1109/TPAMI.2007.1166.

[17] Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S., 2024. 2d gaussian splatting for geometrically accurate radiance fields, in: Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24, ACM. p. 1–11. URL: http://dx.doi.org/10. 1145/3641519.3657428, doi:10.1145/3641519.3657428.

[18] Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., Ma, Y., 2023. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. URL: https://arxiv.org/abs/2311.17977, arXiv:2311.17977.

[19] Kazhdan, M., Bolitho, M., Hoppe, H., 2006. Poisson surface reconstruction, in: Proceedings of the Fourth Eurographics Symposium on Geometry Processing, Eurographics Association, Goslar, DEU. p. 61–70.

[20] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42. URL: https://repo-sam.inria.fr/ fungraph/3d-gaussian-splatting/.

[21] Krizhevsky, A., Sutskever, I., Hinton, G.E., 2017. Imagenet classification with deep convolutional neural networks. Commun. ACM 60, 84–90. URL: https://doi.org/10.1145/3065386, doi:10.1145/3065386.

[22] Laine, S., Hellsten, J., Karras, T., Seol, Y., Lehtinen, J., Aila, T., 2020. Modular primitives for high-performance diferentiable rendering. ACM Trans. Graph. 39. URL: https://doi.org/10.1145/3414685. 3417861, doi:10.1145/3414685.3417861.

![](images/ab506f89d92b8d358eb337bcbc69e132fe83361cae07944701b5778c0d68b15a.jpg)

![](images/21651851bf269af6fe7fe401d6008c1f0b5997fa8d5f29e74829f28af12e3bac.jpg)  
(b) SuGaR

(a) Alicevision Meshroom  
![](images/50d59e6a302050c71e0877469f8bd9d5ef7da40bf2f5a3760ebb44796a3987b0.jpg)

![](images/76b1aeb4da19cf661f6ccd199c4532d8703b27cc761a5d0e0f180aaed57a3e53.jpg)

(c) 2D Gaussian Splats  
![](images/184cf3be59b428d6397636a08f548e596d98be697f5bbb44fc1067c972da9d85.jpg)  
(e) PGSR

(d) NeRF2Mesh  
![](images/856d736217e7162e541de612fa1b9e04d528d366a578adedc8b85c6c3aa41ce9.jpg)  
(f) GGGS  
Figure 16: Normal Maps from the Top View of Day 9 Cauliflower for all the Pipelines

[23] Liu, S., Yang, M., Xing, T., Yang, R., 2025. A survey of 3d reconstruction: The evolution from multi-view geometry to nerf and 3dgs. Sensors 25. URL: https://www.mdpi.com/1424-8220/25/18/5748, doi:10.3390/s25185748.

[24] Lorensen, W.E., Cline, H.E., 1987. Marching cubes: A high resolution 3d surface construction algorithm. SIGGRAPH Comput. Graph. 21, 163–169. URL: https://doi.org/10.1145/37402.37422, doi:10.1145/ 37402.37422.

[25] Lowe, D.G., 2004. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vision 60, 91–110. URL: https://doi.org/ 10.1023/B:VISI.0000029664.99615.94, doi:10.1023/B:VISI.0000029664. 99615.94.

[26] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R., 2021. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99–106. URL: https: //doi.org/10.1145/3503250, doi:10.1145/3503250.

[27] Müller, T., Evans, A., Schied, C., Keller, A., 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics 41, 1–15. URL: http://dx.doi.org/10.1145/ 3528223.3530127, doi:10.1145/3528223.3530127.

[28] Newcombe, R.A., Fitzgibbon, A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J., Kohi, P., Shotton, J., Hodges, S., 2011. KinectFusion: Real-time dense surface mapping and tracking , in: 2011 IEEE International Symposium on Mixed and Augmented Reality, IEEE Computer Society, Los Alamitos, CA, USA. pp. 127– 136. URL: https://doi.ieeecomputersociety.org/10.1109/ISMAR.2011. 6092378, doi:10.1109/ISMAR.2011.6092378.

![](images/e96a8ca52d2e1e57eb42a7d7e4f7511b097c4f1d944f7d7ea81f0b7e5d97f0b4.jpg)  
(a) Alicevision Meshroom

![](images/68963bd12192ad1583894ace75c4b537b19ce82d1553196143911cae6d7517b0.jpg)  
(b) SuGaR

![](images/b734a7a01ea873b3e692f5951623a59abe271ff56fc7081cf61e5d59f6a9e881.jpg)

![](images/e02fffcfc2b83e54f65cddcbbc1473229cb039f81986a7d0584ffd276d2e746a.jpg)  
(d) NeRF2Mesh

(c) 2D Gaussian Splats  
![](images/e684a9954fd38e054aa2fc1ef7e7a1d7a072521943e5cbc95b453587f0038686.jpg)  
(e) PGSR

![](images/a07532ba8f6796a51eec09dc3dde23f7925122b48b648ca34158c455aa4a1202.jpg)  
(f) GGGS  
Figure 17: Geometrical Top View of Day 9 Cauliflower for all the Pipelines

[29] Nilsson, J., Akenine-Möller, T., 2020. Understanding ssim. URL: https://arxiv.org/abs/2006.13846, arXiv:2006.13846.

[30] Pieruschka, R., Schurr, U., 2019. Plant phenotyping: Past, present, and future. Plant Phenomics 2019, 7507131. URL: https://spj.science. org/doi/abs/10.34133/2019/7507131, doi:10.34133/2019/7507131, arXiv:https://spj.science.org/doi/pdf/10.34133/2019/7507131.

[31] Rodríguez, T., Sturm, P., Gargallo, P., Guilbert, N., Heyden, A., Jauregizar, F., Menéndez, J.M., Ronda, J.I., 2005. Photorealistic 3d reconstruction from handheld cameras. Machine Vision and Applications 16, 246–257. URL: https://doi.org/10.1007/s00138-005-0179-4, doi:10.1007/s00138-005-0179-4.

[32] Schonberger, J.L., Frahm, J.M., 2016. Structure-from-motion revisited, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4104–4113.

[33] Singh, K., Saddik, A.E., Saini, M., 2026. A step closer towards the digital twin of the plant. ACM Trans. Multimedia Comput. Commun. Appl. 22. URL: https://doi.org/10.1145/3774885, doi:10. 1145/3774885.

[34] Singh, K., Saini, M., 2024. Towards digital twin of crops for growth modelling using virtual reality, in: Proceedings of the 5th ACM International Conference on Multimedia in Asia, Association for Computing Machinery, New York, NY, USA. pp. 1–5. URL: https: //doi.org/10.1145/3595916.3626368, doi:10.1145/3595916.3626368.

[35] Strecha, C., Hansen, W.v., Gool, L.v., Fua, P., Thoennessen, U., 2008. On benchmarking camera calibration and multi-view stereo for high resolution imagery. URL: https://publica.fraunhofer.de/handle/ publica/360317, doi:10.1109/CVPR.2008.4587706.

[36] Stuart, L.A.G., Pound, M.P., 2025. 3dgs-to-pc: Convert a 3d gaussian splatting scene into a dense point cloud or mesh. URL: https: //arxiv.org/abs/2501.07478, arXiv:2501.07478.

[37] Tang, J., Zhou, H., Chen, X., Hu, T., Ding, E., Wang, J., Zeng, G., 2023. Delicate textured mesh recovery from nerf via adaptive surface refinement. URL: https://arxiv.org/abs/2303.02091, arXiv:2303.02091.

[38] Team, S.D., Chen, X., Chu, F.J., Gleize, P., Liang, K.J., Sax, A., Tang, H., Wang, W., Guo, M., Hardin, T., Li, X., Lin, A., Liu, J., Ma, Z., Sagar, A., Song, B., Wang, X., Yang, J., Zhang, B., Dollár, P., Gkioxari, G., Feiszli, M., Malik, J., 2025. Sam 3d: 3dfy anything in images. URL: https://arxiv.org/abs/2511.16624, arXiv:2511.16624.

[39] Pablo Alcantarilla (Georgia Institute of Technology), Jesus Nuevo (TrueVision Solutions AU), A.B., 2013. Fast explicit difusion for accelerated features in nonlinear scale spaces, in: Proceedings of the British Machine Vision Conference, BMVA Press. pp. 13.1–13.11. doi:http://dx.doi.org/10.5244/C.27.13.

[40] Vollmer, J., Mencl, R., Muller, H., 1999. Improved Laplacian Smoothing of Noisy Surface Meshes. Computer Graphics Forum doi:10.1111/1467-8659.00334.

[41] Vázquez-Arellano, M., Griepentrog, H.W., Reiser, D., Paraforos, D.S., 2016. 3-d imaging systems for agricultural applications—a review. Sensors 16. URL: https://www.mdpi.com/1424-8220/16/5/618, doi:10.3390/s16050618.

[42] Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E., 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 600–612. doi:10.1109/TIP.2003. 819861.

[43] Xiang, J., Chen, X., Xu, S., Wang, R., Lv, Z., Deng, Y., Zhu, H., Dong, Y., Zhao, H., Yuan, N.J., Yang, J., 2025. Native and compact structured latents for 3d generation. URL: https://arxiv.org/abs/2512.14692, arXiv:2512.14692.

[44] Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y., 2024. Absgs: Recovering fine details for 3d gaussian splatting. URL: https://arxiv.org/abs/2404. 10484, arXiv:2404.10484.

[45] Yoo, J.C., Han, T.H., 2009. Fast normalized cross-correlation. Circuits Syst. Signal Process. 28, 819–843. URL: https://doi.org/10.1007/ s00034-009-9130-7, doi:10.1007/s00034-009-9130-7.

[46] Yu, S., Liu, X., Tan, Q., Wang, Z., Zhang, B., 2024a. Sensors, systems and algorithms of 3d reconstruction for smart agriculture and precision farming: A review. Computers and Electronics in Agriculture 224, 109229. URL: https://www.sciencedirect. com/science/article/pii/S0168169924006203, doi:https://doi.org/10. 1016/j.compag.2024.109229.

[47] Yu, Z., Sattler, T., Geiger, A., 2024b. Gaussian opacity fields: Eficient adaptive surface reconstruction in unbounded scenes. URL: https: //arxiv.org/abs/2404.10772, arXiv:2404.10772.

[48] Zhang, B., Fang, C., Shrestha, R., Liang, Y., Long, X., Tan, P., 2024. Rade-gs: Rasterizing depth in gaussian splatting. URL: https://arxiv.org/abs/2406.01467, arXiv:2406.01467.

[49] Zhang, B., Jiang, C., Li, H., Shen, S., Tan, P., 2026. Geometrygrounded gaussian splatting. URL: https://arxiv.org/abs/2601.17835, arXiv:2601.17835.

[50] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O., 2018. The unreasonable efectiveness of deep features as a perceptual metric. URL: https://arxiv.org/abs/1801.03924, arXiv:1801.03924.

[51] Zhang, Z., 2012. Microsoft kinect sensor and its efect. IEEE MultiMedia 19, 4–10. doi:10.1109/MMUL.2012.24.

[52] Zhao, H., Gallo, O., Frosio, I., Kautz, J., 2017. Loss functions for image restoration with neural networks. IEEE Transactions on Computational Imaging 3, 47–57. doi:10.1109/TCI.2016.2644865.

[53] Zhou, Q.Y., Park, J., Koltun, V., 2018. Open3d: A modern library for 3d data processing. URL: https://arxiv.org/abs/1801.09847, arXiv:1801.09847.