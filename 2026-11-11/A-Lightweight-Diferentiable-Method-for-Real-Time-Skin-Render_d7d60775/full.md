# A Lightweight Diferentiable Method for Real-Time Skin Rendering and Digitization

Mario Marquez-Balduque<sup>a</sup>, Carlos Rodríguez-Pardo<sup>b,c,d</sup> and Jorge Lopez-Moreno<sup>a</sup>

<sup>a</sup>Universidad Rey Juan Carlos, Spain

<sup>b</sup>Politecnico di Milano, Milan, Italy

<sup>c</sup>Euro-Mediterranean Center on Climate Change (CMCC), Lecce, Italy

<sup>d</sup>RFF-CMCC European Institute on Economics and the Environment (EIEE), Milan, Italy

## A R T I C L E I N F O

Keywords:   
Diferentiable Rendering   
Screen-Space Subsurface Scattering   
Real-Time Rendering   
Skin Appearance   
Inclusive Digitization

## A BS T R AC T

Inverse rendering for skin typically relies on ofline path tracing to estimate physical properties, which fail to map directly to the approximate shaders utilized in interactive applications, such as game engines. We bridge this gap by directly estimating subsurface scattering parameters tailored for real-time rendering pipelines. We introduce a diferentiable Screen-Space Subsurface Scattering (SSSS) framework that enables the joint optimization of spatially varying base textures and scattering distances. To account for local variations in skin thickness, we estimate these parameters across distinct facial regions, generating a blended profile map ready for production runtime engines. Our pipeline utilizes a low-cost, single-light, single-view capture setup designed for accessible and practical appearance acquisition. Optimization is driven by a masked loss function targeting regions where subsurface scattering efects are dominant. We validate our approach on a real-world dataset of 45 individuals across the Monk Skin Tone Scale, ensuring demographic generalization. Furthermore, we complement this evaluation with synthetic data, a multi-light setup validating our single-light assumption, and a tilted-view configuration confirming angular consistency. Results demonstrate that our method efectively recovers scattering parameters from lightweight inputs, yielding realistic skin appearance directly compatible with standard real-time shaders.

## 1. Introduction

Accurately simulating human skin remains a significant challenge in computer graphics due to its physical complexity and the high sensitivity of the human visual system to facial appearance. As a layered, anisotropic, and spatially varying medium, skin involves intricate subsurface scattering, fine-scale geometric details, and directionally dependent light transport. To meet the computational constraints of interactive applications, the industry often relies on screenspace approximations rather than full volumetric path tracing. However, the perceptual sensitivity of observers ensures that even subtle rendering inaccuracies appear unnatural. Additionally, the global diversity of skin tones demands methods that generalize across the full spectrum of human pigmentation while maintaining both physical plausibility and real-time performance.

Traditional approaches for capturing skin optical properties often require sophisticated hardware, such as hyperspectral sensors or polarimetric imaging systems [1, 2, 3]. While these methods yield accurate results, they are typically expensive, restricted to controlled laboratory settings, and produce physical parameters (e.g., extinction coeficients) that do not map directly to real-time shaders. Furthermore, they often struggle to scale across the diverse facial expressions and skin tones encountered in real-world scenarios. To address these limitations, we introduce a low-cost, replicable singleview, single-light capture setup. Our compact configuration facilitates the practical acquisition of skin appearance without requiring specialized environments.

![](images/21000e6da99c647e5382c816ebeeb97f11b730ab0f3e00e7608df6083f58d1d7.jpg)  
Figure 1: Our lightweight pipeline estimates spatially-varying subsurface scattering from single-image strip-light captures. Shown: rendered avatars across the Monk skin tone scale using optimized per-region scattering parameters.

Instead of estimating ofline physical parameters like absorption or extinction coeficients through complex pathtracing intermediaries, we directly optimize the specific parameters used in real-time rendering pipelines. We ground our approach in the Bidirectional Subsurface Scattering Reflectance Distribution Function (BSSRDF), specifically utilizing the Burley Normalized Difusion profile [4], the standard for real-time physically based skin shading. To enable optimization, we implement a diferentiable version of the Screen-Space Subsurface Scattering (SSSS) algorithm incorporating this profile. This model provides control via the scattering distance parameter, which governs the distance light travels beneath the surface. Tuning this parameter manually is dificult due to its strong coupling with material appearance and the lack of direct visual references. Our approach automates this process by directly optimizing scattering distance textures from photographic data, enabling eficient and plausible appearance reconstruction.

![](images/ec1b687c721fc24a1fd96c7473c823d93a9516cc4c05d6450a2c9b548110ac1f.jpg)  
Figure 2: Monk Skin Tone Scale color swatches [5]. Image credit: © Google, used under the CC BY 4.0 license.

Prior work frequently underexplores skin tone diversity, focusing primarily on mid-range tones and resulting in biased models that generalize poorly across lighter or darker skin. To address this limitation and drive representative skin modeling, we anchor our framework in the Monk Skin Tone Scale [5] (Fig. 2). Unlike traditional scales designed primarily for dermatological UV response (e.g., Fitzpatrick [6]), the Monk scale provides a perceptually balanced distribution covering the broad spectrum of human skin tones. Our primary goal is to leverage this scale to deliver a lightweight diferentiable inverse rendering method that is physically grounded, equitable across diverse populations, and directly compatible with real-time production pipelines.

## Our key contributions are:

1. A lightweight, low-cost, single-light, single-view capture setup explicitly designed for the practical acquisition of facial skin patches across diverse skin tones.

2. A diferentiable rendering framework introducing a novel optimization stage to estimate state-of-the-art real-time scattering parameters directly from image data, bypassing complex path-tracing intermediaries.

3. A masked loss formulation that isolates gradients within shadowed regions, where multiple scattering is physically dominant, ensuring accurate and stable optimization of subsurface parameters.

4. A large dataset of calibrated facial skin patches and their estimated optical properties, sampled across the Monk Skin Tone Scale to drive equitable and representative skin modeling.

## 2. Related Work

In this section, we review prior work on skin rendering, categorizing approaches into subsurface scattering approximations, biophysical and material models, physical acquisition methods, and inverse rendering strategies.

Subsurface Scattering Approximations. Realistic skin rendering requires simulating the complex transport of light through layers of varying opacity. As described by the BSS-RDF [4], light transport in translucent media falls into three regimes: single scattering (thin materials), multiple scattering (thick materials), and intermediate scattering (turbid media) (Fig. 3). While full simulation is computationally expensive, most methods approximate the multiple scattering component to achieve visual plausibility [7]. The dipole model [8] remains a foundational approximation for ofline rendering, modeling subsurface transport via virtual light source pairs. Subsequent improvements include the multipole method [9] for multi-layered materials and the directional dipole [10, 11], which introduces anisotropy to better capture detailed surface features. For higher precision, the quantized difusion model [12] discretizes the difusion process, ofering finer control over scattering behavior.

![](images/486a2db9507e535004918597afb6537999518f58e959948bedf76dace902484d.jpg)  
Figure 3: Illustration of subsurface scattering regimes. Left: a rendered ear where the yellow box highlights a single scattering dominant area (thin cartilage), while the green box shows a region dominated by multiple scattering, emphasizing the variability of light transport in human skin. Right: representation of single, intermediate, and multiple scattering. Adapted from [13].

For real-time pipelines, scattering is typically separated into surface and subsurface components. Translucency efects (single scattering) are often approximated using thickness maps generated via inverted-normal ambient occlusion [14]. For multiple scattering, screen-space techniques are the standard. [15] and [16] approximate difusion by convolving irradiance maps with weighted Gaussian kernels, while [17] proposed an exponential difusion profile matching Monte Carlo simulations. Recently, neural approaches such as [18, 19] have proposed encoding scattering profiles into lightweight networks, though screen-space convolution remains the industry standard for its predictability and direct integration with standard shading models.

Biophysical and Skin Material Models. Human skin exhibits significant spatial heterogeneity. Early measurementbased approaches [20] demonstrated that fixed scattering parameters fail to capture the variation between regions like the forehead, cheeks, and ears. [21] introduced layered models to account for epidermal and dermal variations, while [22] utilized region-based lookup tables, and [23] extended layered models to simulate cosmetics. Beyond empirical layers, recent biophysical models explicitly simulate physiological processes, such as sun-induced erythema and tanning [24], or multi-layer heterogeneous volume transport [25].

While volumetric biophysical models ofer high biological fidelity, they require computationally demanding Monte Carlo solvers. In contrast, we utilize the Disney 2015 BSSRDF [4] as our parametric foundation. While the earlier Disney BRDF [26]—widely referred to in the industry as Burley 12—provided a robust model for surface reflection, it lacked volumetric scattering controls. The 2015 extension, commonly known as Burley 15, introduces a physically derived scattering distance parameter that ensures energy conservation while supporting anisotropic difusion. This balance of physical plausibility and artist-friendly control makes Burley 15 ideal for our optimization pipeline. Inspired by [27], we optimize a scalar scattering distance per facial region under this model and blend them at runtime to achieve anatomical fidelity without sacrificing real-time eficiency.

Skin Appearance Acquisition. High-fidelity skin capture traditionally relies on complex hardware to separate reflectance components. Systems such as Light Stages [28], time-multiplexed lighting rigs [29], and polarization-based setups [30, 31] deliver ground-truth quality. Recent advances include spectral reconstruction methods [32] and singleshot polarized capture systems [33] that recover high-quality geometry and appearance. However, these methods require specialized equipment and controlled environments. In contrast, we focus on a lightweight acquisition regime to democratize skin reconstruction. Our diferentiable framework optimizes scattering parameters from a single-view, singlelight setup, eliminating the need for costly optical components or synchronized lighting arrays. Furthermore, this simplified pipeline remains robust across the full spectrum of human skin tones. This combination of hardware accessibility and tonal generalization provides a scalable solution for realistic digital human creation.

Inverse Rendering for Material Estimation. Material estimation generally falls into learning-based or optimizationbased categories. Learning-based methods [34, 35, 36] train neural networks to infer parameters from images, ofering fast inference but often lacking generalization to out-ofdistribution examples. Optimization-based methods minimize the reconstruction error between observed and rendered images. For example, [37] optimized spatially varying parameters for fabrics using a diferentiable rasterizer. In the domain of skin, [2] and [3] have successfully estimated parameters from hyperspectral data. Most closely related to our work is [38], which performs inverse rendering for textured translucent appearance. However, while [38] focuses on general translucent objects, our pipeline is specialized for human skin, incorporating domain-specific constraints (such as the Monk scale and facial region masks) and directly optimizing the screen-space parameters used in game engines. Furthermore, while [25] estimates biological parameters from a pre-extracted albedo map for ofline volume rendering, our framework optimizes directly from raw photographs under directional lighting, decoupling surface reflection (albedo, roughness, specular) and subsurface scattering for real-time screen-space execution.

## 3. Overview and Rendering Model

This section outlines our proposed pipeline for skin appearance estimation and describes the real-time rendering techniques integrated into our engine to simulate subsurface scattering.

## 3.1. Overview

Our approach builds upon the difusion-based skin model introduced by [17], which extends Burley 12 with a subsurface scattering parameter �. We adopt Burley 15 model for its physical grounding and proven photorealism [37].

Unlike prior works that fix subsurface parameters to fit surface properties, we explicitly estimate the subsurface scattering terms that dominate skin appearance. We define the scattering distance as $\begin{array} { r } { d = { \frac { 1 } { s } } } \end{array}$ (assuming a unit mean free path). We replace the standard uniform parameterization with an optimized homogeneous scattering parameter per facial region, which are then blended to create a spatially varying map for local anatomical variations.

We focus on the human face due to its perceptual importance in recognition [39] and the significant variation in skin thickness that influences light difusion [40]. Consequently, our acquisition targets facial skin patches.

We utilize a single-view, single-light capture setup to extract skin data. To ensure representational diversity, we select subjects spanning the Monk Skin Tone Scale [5], capturing the optical variability caused by difering melanin and hemoglobin levels. Our proposed architecture proceeds in two sequential phases: first, we optimize a homogeneous scattering distance � for each individual skin patch; second, we generate a full facial scattering map by blending these values based on anatomical regions, similar to the profile blending in [27]. To execute this initial patch-fitting phase, our inverse rendering pipeline directly optimizes a spatially varying Burley 15 texture stack by backpropagating reconstruction errors through a diferentiable rasterizer, extending the core optimization framework from [37]. The complete pipeline is illustrated in Fig. 4.

This automated approach eliminates manual parameter tuning, driving skin rendering directly from photographic observations. Accordingly, the remainder of the paper details our acquisition protocol (Section 4.1), the tensorized scattering architecture (Section 4.2), implementation details (Section 5), and our experimental results and validation (Sections 6.1 and 6.2).

## 3.2. Rendering Model

We implement a custom real-time rendering engine using WebGL and Babylon.js [41]. This choice provides a broadly deployable, cross-platform verification environment that runs across a wide range of browsers and devices, including resource-constrained and low-end platforms, allowing us to assess whether the fitted scattering distances produce realistic and physically consistent appearances under practical crossdevice rendering constraints. In addition to visualizing the final assets, the engine acts as a synthetic data generator, producing reference images with known ground-truth parameters to validate our optimization procedure.

The shading model relies on two primary inputs: a thickness texture and a spatially varying scattering distance texture. In the absence of accurate data textures (measured or artist-painted), local thickness is approximated automatically via inverted-normal SSAO [14]. The idea is to flip the surface normal, so that the AO hemisphere points into the object instead. Nearby surfaces encountered by those inward-facing samples then provide a rough indication of how close other parts of the object are. When inward-facing polygons are close to one another, they generate more occlusion; this is interpreted as the region being locally thinner.

![](images/f897203bbf7d1f52c8fceeb6ef511fd17d21f44ee97202605a685e54da732355.jpg)

Figure 4: Overview of our inverse skin rendering pipeline: we optimize Burley 15 texture parameters and a homogeneous scattering distance using a diferentiable rasterizer, then generate spatially varying subsurface profiles via profile blending for final rendering.  
![](images/1e2a37670e2ae395a4e3696cf5912cedbd9b92b2f78f34d521d0c22b096840dd.jpg)  
(a)

![](images/8d7052e40961f45027c4f80536618d51ed220847453ea4ffd2a54c3d4ac25e86.jpg)  
(b)

![](images/ffb5fce69e18495c57370aae75a896112847728514c97d22af38f70ddcebc842.jpg)  
(c)  
Figure 5: Texture attachments of our rendering engine: (a) thickness, (b) region, and (c) blended scattering distance. For the region texture (b), the red channel encodes discrete region tags, while the green channel stores the interpolation factor for scattering distance blending .

To encode local scattering behavior, we generate the scattering distance texture via profile blending. Homogeneous scattering values optimized from patches are interpolated using the thickness texture (Fig. 5a) as a blending weight. This produces a smooth, continuous map (Fig. 5c) that accounts for thinner skin areas (e.g., ears, nose) without requiring unique profiles per pixel. Optionally, an anatomical region map (Fig. 5b) can be supplied to assign distinct scattering baselines to specific zones.

Scattering Distance Scaling. The scattering distance texture is parameterized in metric units rather than arbitrary scalar weights. Following the convention of real-time engines such as Unreal Engine 5, the normalized texture domain [0, 1] maps linearly to a scattering distance range of [0, 1.0] cm. To align this with our engine’s world-space coordinate system (measured in meters), we apply a fixed dimensional conversion factor of 0.01 ( $1 . 0 \mathrm { c m } = 0 . 0 1 \mathrm { m } )$ . Unlike screenspace approximations that rely on projected pixel distances, we compute the true 3D Euclidean distance between the surface point and the sample point using the depth bufer. This ensures that the difusion profile remains physically consistent regardless of camera distance or field of view.

Multiple Scattering. We simulate multiple scattering using the screen-space approach proposed by [42], which integrates the difusion profile by distributing samples within a 2D screen-space disk and reconstructing the 3D surface geometry under perspective projection. We employ the exponential approximation of the Burley profile:

![](images/8f17113ffb43bfb15b97a543c14e2f4c617187291c8b5b59e0eb4077725d594a.jpg)  
(a)

![](images/fd6ede22d60c68c594f01c93a98fdf11e814cea669e8085e1248171a4f394791.jpg)  
(b)  
Figure 6: Left: Fibonacci sampling pattern used for importance sampling the difusion profile. Right: Chromatic fringing observed in captures due to spectral dependencies in the projector light.

$$
R ( r , d ) = A \ \frac { e ^ { - r / d } + e ^ { - r / 3 d } } { 8 \pi d r }\tag{1}
$$

where � is the difuse volume albedo, � represents the 3D Euclidean distance, and � is the scaled scattering distance inherent to the material properties. This formulation simplifies light transport through layered skin into a dual-exponential profile over a single homogeneous volume. Because it does not model individual epidermal and dermal layers, the recovered parameter � represents an efective parameter for the real-time shader rather than an intrinsic scattering coeficient $( \sigma _ { s } )$

To minimize Monte Carlo variance, these samples are generated in polar coordinates $( \boldsymbol { \theta } _ { i } , \boldsymbol { r } _ { i } )$ via an analytical Importance Sampling scheme. The resulting Probability Density Function (PDF) $p ( r )$ and Complementary Cumulative Distribution Function (CCDF) $\bar { P } ( r )$ derived from the difusion profile are defined as:

$$
p ( r ) = \frac { r R ( r , d ) } { A } , \quad \bar { P } ( r ) = \frac { 1 } { 4 } e ^ { - r / d } + \frac { 3 } { 4 } e ^ { - r / 3 d }\tag{2}
$$

According to [42], �(�) models the linear profile shape along a radial ray; the full 2� circle integration factor is explicitly accounted for during numerical CCDF inversion and applied within the final difuse irradiance estimator $( 2 \pi / N )$

The sampling radius $r _ { i } ~ \in ~ [ 0 , r _ { \operatorname* { m a x } } ]$ is generated by inverting the CCDF in Eq. 2 via inverse transform sampling. The angular coordinate $\theta _ { i }$ follows a Fibonacci pattern [43] (Fig. 6a) combined with a stochastic rotationaljitter to convert structured aliasing into high-frequency temporal noise.

![](images/b40adfcd099a97d053a9cdfc318df8b0edd945a72c24d010cea2a079bc5669e6.jpg)  
(a)

![](images/8211d31de7bd4ace2d05ea49f6a903002a6ce6b532ea6f204fe65493ea7b0e56.jpg)  
(b)

![](images/b6ab0843020f736b63228be0d185040102a7c1297c5c2c1b43cbc72fd8463609.jpg)  
(c)  
Figure 7: Final render produced with our engine (a). Insets highlight regions dominated by (b) multiple scattering and (c) single scattering.

For a target fragment with view-space position ${ \textbf { x } } \in$ $\mathbb { R } ^ { 3 }$ , the local 2D polar ofset is projected to determine its corresponding screen-space coordinate $\mathbf { u } _ { i } ~ = ~ [ u _ { i } , v _ { i } ] ^ { T }$ The true 3D view-space position of this surface sample, $\mathbf { x } _ { i } ,$ is reconstructed directly by applying the standard inverse perspective projection matrix to the normalized screen coordinates and the depth map $D ( \mathbf { u } _ { i } )$ .

The true 3D Euclidean distance through the scattering medium is subsequently evaluated as $s _ { i } ~ = ~ \| \mathbf { x } - \mathbf { x } _ { i } \| _ { 2 } .$ Following the bilateral formulation in [42], we evaluate a spectral difusion weight $k _ { i , c }$ for each color channel � ∈ {�, �, �} that balances the 3D radial profile response against the 2D sampling density:

$$
k _ { i , c } = \frac { s _ { i } \cdot R ( s _ { i } , d _ { c } ) } { r _ { i } \cdot R ( r _ { i } , r _ { \operatorname* { m a x } } ) }\tag{3}
$$

where $d _ { c }$ denotes the material scattering distance for the respective spectral channel, and $r _ { \mathrm { m a x } }$ represents the global boundary domain limit.

The final scattered difuse radiance $I _ { c } ( \mathbf { x } )$ is obtained via a normalized weighted average of the incident difuse irradiance field $L _ { c } ( \mathbf { u } _ { i } )$ across all � screen-space samples:

$$
I _ { c } ( \mathbf { x } ) = A _ { c } \left( \frac { \sum _ { i = 1 } ^ { N } k _ { i , c } L _ { c } ( \mathbf { u } _ { i } ) } { \sum _ { i = 1 } ^ { N } k _ { i , c } } \right)\tag{4}
$$

where $A _ { c }$ is the channel albedo, and $L _ { c } ( \mathbf { u } _ { i } )$ represents the standard Lambertian incident difuse irradiance sampled at the projected screen-space coordinate $\mathbf { u } _ { i } .$ . To ensure full reproducibility, the implementation details of the shader are provided in Section 5.3 of the Supplementary Material.

Single Scattering. For a complete skin shader, we also account for single scattering (translucency) through thin geometry (e.g., ears). We implement the approximation proposed by [14], which estimates light transmission using the Beer-Lambert law and the thickness texture. As our optimization framework focuses exclusively on the dominant multiple scattering component, we employ this term solely for visual completeness in the final renders (Fig. 7).

## 4. Skin Fitting Pipeline

## 4.1. Acquisition

Estimating physically meaningful texture maps—such as normals, roughness, albedo, and scattering distances—from skin images typically demands complex acquisition systems. Common approaches rely on multi-view setups, polarized lighting, or hyperspectral imaging [2, 3, 1]. While accurate, these systems sufer from limited scalability and high deployment costs. Furthermore, spatially varying estimation is highly sensitive to lighting directionality, local geometry, and microstructural detail [21, 44, 30], making replication outside controlled laboratories dificult.

To address these limitations, we propose an acquisition setup that balances simplicity with fidelity and is lightweight in both hardware and operational complexity compared to such laboratory rigs. Rather than requiring custom optical components or electronic synchronization, our system relies entirely on consumer-grade devices, eliminates complex geometric and radiometric calibration protocols, and is easily deployed in standard indoor environments, specifically in a darkened room to eliminate ambient light interference. As shown in Fig. 8a, the setup consists of:

1. Subject support. A rigid horizontal platform stabilizes the target surface. Participants place a specific facial region (e.g., cheek or forehead) on a fixed marker to ensure positional repeatability.

2. Illumination source. A projector is used to emit both highcontrast structured line patterns and uniform white light. It is positioned approximately 20 cm from the subject, tilted upwards at $3 0 ^ { \circ }$ relative to the surface plane. This low-angle configuration creates strong directional lighting, maximizing the visibility of subsurface scattering efects while allowing for controlled binary illumination.

3. Imaging system. A high-resolution RGB camera (DSLR) is mounted behind the projector at a distance of approximately 50 cm from the subject. This near-coaxial alignment minimizes geometric distortion and allows the direct capture of reflectance and shading cues under controlled illumination.

Although we target a near-coaxial alignment, our pipeline ensures robustness against angular deviations. To maintain a lightweight workflow, we avoid complex projector-camera calibration and structured-light triangulation, relying instead on monocular depth estimation [45]. We map the network’s normalized output (0 for the near plane, 1 for the far plane) to a physical depth range of 2 cm that bounds the local facial relief, anchored at a measured camera-to-subject distance of 47 cm. All geometric and exposure settings are fixed to maintain consistency across subjects (Section 5).

Despite its simplicity, this setup successfully captures the subtle shading gradients and scattering cues required for material estimation, avoiding the cost and complexity of multi-camera or hyperspectral systems [21, 2]. While specialized hardware ofers higher raw fidelity, we demonstrate that optimized diferentiable rendering can extract productionquality assets from this minimal input.

We acquire four photographs per skin patch under distinct binary illumination patterns (Fig. 8b). We utilize vertically aligned line patterns to minimize projector-induced chromatic aberration and color fringing (Fig. 6b). Unlike the complex patterns conventional in structured-light reconstruction [46, 47, 48], our simplified linear patterns reduce spectral interference—a critical consideration given the wavelengthdependent nature of subsurface scattering. Fig. 9 details the sequence:

![](images/8644b732b9aabfd0b8487f5925b2082dbb29afe7c81cfe012e1ae8e4f300d710.jpg)

![](images/167e1202ce2e5adaebda96c225ea4d28721ef8d4c10c3654021839cf87f06ea7.jpg)  
(a)

![](images/bdbe8b7c11f5cd56e3953b37e56e14727bafc292a1010dd0b00022209a58b928.jpg)  
(b)

Figure 8: Hardware acquisition setup and illumination patterns. (a) Physical configuration (left) and corresponding schematic ray-diagram (right) illustrating the scanning system geometry. (b) Temporal sequence of projected light patterns used during data acquisition: ambient baseline, structured stripes, inverted stripes, and uniform flat-field illumination.  
![](images/ec1e7ec579bc74837e75fc88b4eebae7060e426f8632af2c8f1782dc6f809b50.jpg)  
(a)

![](images/70148dfb359f2bc233ca72e7fb3f037a22fd83d9544ae5c5891d6d20ca2283ee.jpg)  
(b)

![](images/9ac8eef92f86d6d01fba218a0fb586c8b3765a26b73eaafc0fa8e598c6c91ec1.jpg)  
(c)

![](images/3df952b47ebe79daf9fce462a5b560a100e2b1e59b460c2d0f2d8d76bbc78175.jpg)  
(d)  
Figure 9: Localized view of the captured raw frame sequence during acquisition. (a) Ambient baseline frame (brightnessenhanced for visibility) showcasing residual environmental light. (b) Active surface capture under the high-frequency structured stripe pattern. (c) Corresponding active capture under the inverted stripe pattern. (d) Fully illuminated frame under uniform flat-field projection.

• Photo 1 (9a): Captured with a black pattern to record ambient residuals. This image is subtracted from subsequent captures to isolate direct illumination.

• Photos 2 & 3 (9b, 9c): Captured using complementary vertical line patterns. These images highlight the skin’s directional response.

• Photo 4 (9d): Captured under full white illumination. This provides a uniformly lit reference for estimating global properties like albedo, decoupling shading from geometric variation.

To isolate subsurface scattering efects, we generate a light mask by computing the diference between the patterned image (9b) and its inverse (9c). We binarize this diference map using Otsu’s thresholding [49], which automatically determines an optimal threshold by maximizing the betweenclass variance of the intensity distribution. To ensure spatial coherence, we apply morphological dilation followed by opening and closing operations. This process produces two distinct versions of the mask to balance optimization accuracy with visual reconstruction:

![](images/0e57b22d28e984a1b8d32409fa4f2f31a82a492244f0c2b6ba6ddd1fba5d6b57.jpg)  
(a)

![](images/1cddcae15cbefacb752de3dec663cb8fc2c790b4245ce03e59b91e4e40245948.jpg)  
(b)

![](images/7681a0734866cf1ef5b5145c7bd265246e117c00f8ccd643ab3dfd4d47430111.jpg)  
(c)

![](images/eeb564b1f282a1d0cb6ee4342ca0f2963829a01be4f8b48ddae89a37e72171de.jpg)  
(d)  
Figure 10: Optimization input data evaluated across diferent anatomical skin regions (rows). From left to right: (a) Uniformly illuminated reference frames, (b) active strip-lit captures displaying subsurface scattering efects, (c) light masks, and (d) estimated monocular depth maps.

1. Loss Mask: An inverted, strictly binary shadow mask  (where 1 corresponds to shadowed regions) used to focus the optimization loss. Maintaining a hard binary edge here is critical; any blurring of the mask during optimization would alter the edges of the illumination pattern and corrupt the scattering halo estimation.

2. Rendering Mask: A non-inverted light mask <sub>smooth</sub> (where 1 corresponds to lit regions) smoothed via an 11 × 11 Gaussian filter. This blur emulates projector defocus and prevents hard-edge artifacts in the final render, ensuring a smooth transition where the specular component is added to the masked difuse component.

The resulting preprocessed images and the depth map (Fig. 10) serve as the input for our diferentiable rasterizer optimization.

## 4.2. Tensorized SSSS

Porting the screen-space multiple scattering algorithm to a diferentiable framework presents a challenge: we must convert per-pixel shader operations into batched tensor computations to ensure eficiency and parallelization. This requires careful handling of texture sampling, coordinate projection, and memory management to maintain diferentiability.

Key aspects of this transformation include:

1. Batch Processing – Instead of iterating per pixel, we process the entire image and all scattering samples simultaneously as high-dimensional tensors.

2. View-Space Reconstruction – To ensure 1:1 consistency with the rendering engine, we reconstruct the 3D position of every pixel using depth and camera intrinsics, rather than relying on 2D screen-space approximations.

3. Diferentiable Sampling – We replace discrete texture lookups with diferentiable bilinear interpolation, allowing gradients to propagate through the scattering inputs.

4. Memory Optimization – We broadcast operations across dimensions to minimize loop overhead, a critical requirement for performance in tensorized frameworks.

The pipeline begins by computing the difuse irradiance tensor $\mathbf { L } _ { T }$ modulated by the smoothed rendering mask $\mathcal { M } _ { \mathrm { s m o o t h } }$ to filter the incoming illumination before difusion: ${ \bf L } _ { T } = \mathcal { M } _ { \mathrm { s m o o t h } } \odot ( { \bf n } \cdot { \bf l } ) _ { + }$ , where � and � denote the surface normal and light direction tensors.

In a standard shader, difusion is computed via a loop over � samples for each pixel. Imitating this loop in a tensor framework is ineficient. Instead, we adopt a fully vectorized approach where the entire image is replicated � times, creating tensors of shape (�, �, �, �). This allows us to apply the scattering kernel across all pixels and samples simultaneously (Fig. 11). The process is as follows:

We define a virtual camera matching the real-world acquisition system (focal length 50 mm, sensor width 23.5 mm). Using the estimated depth map (Fig. 10d), we unproject every pixel into 3D view-space coordinates $\mathbf { P } _ { v } .$ To avoid highdimensional matrix inversions during backpropagation, we linearize depth and unproject coordinates component-wise:

$$
\mathbf { z } = \frac { P _ { 2 3 } } { ( 2 \cdot \mathbf { D } _ { T } - 1 ) - P _ { 2 2 } }\tag{5}
$$

$$
\mathbf { x } _ { v } = \frac { \mathbf { U } \odot \mathbf { z } } { P _ { 0 0 } } , \quad \mathbf { y } _ { v } = \frac { \mathbf { V } \odot \mathbf { z } } { P _ { 1 1 } }\tag{6}
$$

where $\mathbf { P } _ { v } = [ \mathbf { x } _ { v } , \mathbf { y } _ { v } , - \mathbf { z } ] ^ { T }$ , the tensors �, � represent the normalized grid coordinates, and $P _ { 0 0 } , P _ { 1 1 } , P _ { 2 2 } , P _ { 2 3 }$ are the perspective projection coeficients derived from the camera geometry and clipping planes. This explicit tensorized reconstruction eliminates the need for 2D pixel-distance approximations, ensuring pixel-level geometric alignment with the WebGL forward rasterizer while decoupling the simulation from the fixed image plane for robust angular consistency, as validated in Section 6.2.

We utilize the precomputed sampling tensors ������� and ������� representing the Fibonacci angles and stratified Burley Inverse CCDF normalized radii, respectively. The angles are shifted by the stochastic jitter pattern, while the normalized radii are scaled by the maximum scattering distance $r _ { \mathrm { m a x } }$ . These updated polar coordinates are then projected to obtain the spatial ofsets $( \mathbf { x } _ { o } , \mathbf { y } _ { o } )$

We apply these ofsets to $\mathbf { P } _ { v }$ in view space and project the resulting positions back into Normalized Device Coordinates (NDC) to form a sampling grid. This perspective projection is defined component-wise as:

$$
{ \bf U } _ { s } = \frac { ( { \bf x } _ { v } + { \bf x } _ { o } ) \cdot P _ { 0 0 } } { { \bf z } } , \quad { \bf V } _ { s } = \frac { ( { \bf y } _ { v } + { \bf y } _ { o } ) \cdot P _ { 1 1 } } { { \bf z } }\tag{7}
$$

where $\mathbf { U } _ { s }$ and ${ \bf V } _ { s }$ are stacked to form the final 2D sampling coordinate grid �. To accurately replicate the hardwareaccelerated bilinear filtering used in standard shaders, we retrieve the incident irradiance $( \mathbf { L } _ { T } )$ and depth $( \mathbf { D } _ { T } )$ at these coordinates using grid sampling with bilinear interpolation. This ensures our optimization kernel mathematically matches the runtime renderer while maintaining full diferentiability.

![](images/dbafd6ca8d669579d764216113f1ddebf833cb91dc20c517b6e965977afca65d.jpg)  
Figure 11: Comparison of the shader-based approach (left) vs. our tensorized scheme (right). By broadcasting the image across the sample dimension �, we perform view-space reconstruction and integration in a single parallelized pass.

We calculate the true 3D Euclidean distance between the surface point $\mathbf { P } _ { v }$ and the sampled point ${ \bf P } _ { s }$ . This distance is fed into the Burley difusion profile to compute the scattering weight (Eq. 3). The weighted samples are summed along the sample dimension (�) to produce the integrated difuse irradiance tensor.

Finally, this integrated difuse irradiance is multiplied element-wise by the albedo tensor ${ \bf A } _ { T }$ to obtain the final scattered difuse radiance component. For visualization during optimization, we add the specular lobe, modulated by the rendering mask $\mathcal { M } _ { \mathrm { s m o o t h } } .$ to produce the complete rendered image. To ensure full reproducibility, the tensorized implementation is provided in Section 5.4 of the Supplementary Material.

## 5. Implementation details

Capture Hardware To capture the input data, we utilize a Sony �6100 (ILCE-6100L), a consumer-grade camera that balances afordability with high image quality. Its highresolution sensor and reliable low-light performance make it an efective instrument for acquiring precise material data, while its accessible price point ensures framework reproducibility. The camera is connected via USB to a PC and remotely controlled using the Sony Imaging Edge desktop application, which provides a live view, full manual parameter control, and automated interval shooting. All images were captured using fixed camera parameters to ensure strict consistency across the dataset. Specifically, we established a resolution of 6020 × 4024 pixels, an ISO of 100, an exposure time of $1 / 2 0 \mathrm { s } ,$ , an aperture of $f / 2 0 .$ , a white balance of 6000 K, and a 50 mm focal length.

Illumination is provided by an Optoma projector rated at 3500 lumens, which projects the sequence of structured light patterns. We synchronize the camera and projector using a combination of Sony Imaging Edge and an AutoHotkey script. While Imaging Edge manages the timed shooting in continuous mode, the AutoHotkey script coordinates each projection frame change with the camera’s shutter trigger, ensuring precise temporal synchronization between pattern display and image acquisition.

Input processing The RAW captures (.ARW) are decoded using rawpy and subsequently center-cropped to $4 8 0 \times 3 6 0$ px to satisfy GPU memory constraints. To ensure chromatic accuracy, we calibrate the raw sensor space to linear sRGB using ${ \mathrm { ~ a ~ } } 3 \times 3$ Color Correction Matrix (CCM) computed via Ridge Regression on an X-Rite ColorChecker, normalizing pixel intensities to the [0, 1] range. For complete processing and calibration details, see Section 5.1 of the Supplementary Material.

Optimization details Our pipeline optimizes a subset of material property tensors from a single view under a single light source, specifically, the spatially varying albedo ${ \bf A } _ { T }$ and roughness $\rho _ { T }$ , alongside the homogeneous scattering distance ${ \bf S } _ { T }$ . Due to the limited geometric cues of this underconstrained setup, we do not optimize surface normals and tangents, relying instead on a fixed Sobel-filtered map derived from a pre-estimated depth map [45]. To comply with commercial production standards, the parameter $\mathbf { S } _ { T } \in [ 0 , 1 ]$ (encoding up to 1.0 cm) maps to world-space distance in meters via the metric conversion $d = \mathbf { S } _ { T } \times 0 . 0 1$ , where $d _ { \mathrm { o p t } }$ represents the resulting homogeneous scattering color texture map. Achieving a clean separation of the difuse albedo ${ \bf A } _ { T }$ is critical to avoid bias during the scattering optimization; since the Burley difusion profile (Eq. 1) scales directly with �, any specular energy incorrectly baked into the albedo would artificially shift its chromatic tone. To capture these highfrequency highlights without corrupting the core material parameters, we introduce an auxiliary residual specular tensor $\mathbf { \Delta } \Delta _ { \mathrm { s p e c } , T }$ . This tensor functions as a temporary residual bufer that absorbs specular discrepancies during optimization, mitigating their bleeding into the base material maps, and is discarded after convergence.

Textures are initialized using physically informed priors: the unwarped IOR map is initialized at a uniform value of 0.145 (linearly mapping to a physical $\mathrm { I O R } \approx 1 . 4 4 )$ , and the homogeneous scattering distance map ${ \bf S } _ { T }$ begins at a uniform 0.5. To isolate a clean skin albedo independent of lighting, hair, or deep shadows, we perform �-means clustering $( k = 2 )$ on the $a ^ { * }$ and $b ^ { * }$ chromatic channels in CIE LAB space. This adaptive mask is refined by excluding the brightest 10% and darkest 5% of pixels. A robust mean color is then computed via 2� Sigma Clipping on the remaining pixels to provide a stable initialization baseline (see Section 5.2 of the Supplementary Material for complete algorithmic details).

We employ a multi-stage optimization strategy using the Adam optimizer [50]. The objective function balances a masked $\mathcal { L } _ { 1 }$ reconstruction loss with an $ { \mathcal { L } } _ { 2 }$ roughness regularization prior:

$$
\mathcal { L } = \frac { 1 } { \left| \Omega \right| } \left\| \hat { \mathbf { I } } \odot \mathcal { M } _ { \mathrm { o p t } } - \mathbf { I } \odot \mathcal { M } _ { \mathrm { o p t } } \right\| _ { 1 } + \lambda _ { \mathrm { p r i o r } } \frac { 1 } { \left| \Omega \right| } \left\| \rho _ { T } - 0 . 6 \right\| _ { 2 } ^ { 2 }\tag{8}
$$

Table 1  
Optimization Schedule and Hyperparameters.
<table><tr><td>Configuration</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td></tr><tr><td>Optimized Parameters Frozen Parameters</td><td> $\Delta _ { \mathrm { s p e c } , T } , \rho _ { T }$   $\mathbf { A } _ { T } , \mathbf { S } _ { T }$ </td><td> $\mathbf { A } _ { T } , \pmb { \rho } _ { T } , \pmb { \Delta } _ { \mathsf { s p e c } , T }$   ${ \bf S } _ { T }$ </td><td> ${ \bf S } _ { T }$ </td></tr><tr><td>Iterations</td><td> $7 0 0$ </td><td>500</td><td> $\mathbf { A } _ { T } , \pmb { \rho } _ { T } , \pmb { \Delta } _ { \mathsf { s p e c } , T }$  600</td></tr><tr><td>Learning Rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Active Mask  $( \mathcal { M } _ { \mathrm { o p t } } )$ </td><td>Full Image (1)</td><td> $\mathsf { F u l l \ l m a g e \left( 1 \right) }$ </td><td>Shadow Mask (M)</td></tr><tr><td>Roughness Prior  $( \lambda _ { \mathsf { p r i o r } } )$ </td><td> $1 \times 1 0 ^ { - 2 }$ </td><td> $1 \times 1 0 ^ { - 2 }$ </td><td>None</td></tr></table>

where � is the captured reference tensor, $\mathcal { M } _ { \mathrm { o p t } }$ denotes the stage-dependent optimization mask and |Ω| is the total number of elements within the active spatial domain. The $ { \mathcal { L } } _ { 2 }$ value prior minimizes the transfer of high-frequency albedo details into the roughness map under single-view constraints, stabilizing the reflection envelope without suppressing regional variability. The anchor value of 0.6 corresponds to the default skin roughness baseline adopted by industrystandard real-time engines. Within this framework, the complete rendered image is formulated as a diferentiable function $\hat { \bf I } = f ( { \bf A } _ { T } , \pmb { \rho } _ { T } , { \pmb { \Delta } } _ { \mathrm { s p e c } , T } , { \bf S } _ { T } )$ , whose active parameter dependencies, hyperparameter weights, and spatial domains adapt dynamically according to the multi-stage schedule outlined in Table 1 and detailed below:

• Stage 1 (Specular & Roughness Pre-fit): Parameters $\Delta _ { \mathrm { s p e c } , T }$ and $\rho _ { T }$ are optimized; ${ \bf A } _ { T }$ and ${ \bf S } _ { T }$ are frozen. This stage fits global reflection, capturing unmodeled specular highlights into $\Delta _ { \mathrm { s p e c } , T }$ . Freezing albedo preserves its clustering initialization, shielding base color from early lighting errors and cross-parameter interference.

• Stage 2 (Surface Maps Refinement): Parameters $\mathbf { A } _ { T } , \pmb { \rho } _ { T }$ and $\mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } } \mathbf { { \Delta } }$ are optimized; ${ \bf S } _ { T }$ is frozen. Unfreezing ${ \bf A } _ { T }$ absorbs difuse details, allowing the $\mathcal { L } _ { 2 }$ prior to smooth $\rho _ { T }$ . Retaining active $\Delta _ { \mathrm { s p e c } , T }$ prevents residual specular discrepancies from being incorrectly baked into albedo.

• Stage 3 (Subsurface Scattering Fit): Parameter ${ \bf S } _ { T }$ is optimized; $\mathbf { A } _ { T } , \pmb { \rho } _ { T }$ , and $\Delta _ { \mathrm { s p e c } , T }$ are frozen. This isolates the low-frequency subsurface scattering pass from highfrequency surface variations, preventing the coupling inherent to Eq. (1) between linear albedo scaling and nonlinear decay � (verified synthetically in Figures 15 and 16). Restricting the loss domain $\mathcal { M } _ { \mathrm { o p t } }$ to the shadow mask  forces the optimizer to fit the multiple scattering halos exclusively, removing interference from surface reflections and preventing direct lighting discrepancies from skewing � (Figure 18).

This schedule is held constant across all captured patches and skin tones, requiring no per-subject hyperparameter tuning. The learning rates, iteration counts, and prior weight for surface estimation (Stages 1 and 2) build upon established baseline configurations [37], with the residual specular map $\Delta _ { \mathrm { s p e c } , T }$ sharing the same learning rate $( 5 \times 1 0 ^ { - 3 } )$ . For Stage 3, we reduce the learning rate to $1 \times 1 0 ^ { - 3 }$ over 600 iterations to ensure stable convergence against the lower gradient magnitudes of subsurface scattering, particularly across darker skin tones where the signal-to-noise ratio is lower. The progressive evolution of material maps across these stages is illustrated in Section 2 of the Supplementary Material.

## 6. Results

We evaluate our method qualitatively and quantitatively on both real and synthetic data to demonstrate its reconstruction fidelity, computational eficiency, and robustness across diverse skin tones. Our evaluation is structured into two main parts: (1) an analysis of fitting results on real facial patches across the Monk Skin Tone Scale, evaluating perceptual metrics, runtime benchmarks, and material plausibility (Section 6.1); and (2) a comprehensive validation study verifying parameter isolation via synthetic ground truth, assessing our shadow-masked loss through ablation, and benchmarking our single-light setup against a multi-light configuration (Section 6.2). Validation of angular consistency via tilted-view experiments is provided in Section 3 of the Supplementary Material.

We built a dataset of 225 real skin patches from 45 participants across five facial regions using a single-light setup. A subset of 17 subjects (85 samples) was additionally captured using a multi-light setup for validation.

Ethics, consent, and data handling . The acquisition protocol was reviewed and approved by the Research Ethics Committee of our institution prior to data collection. All participants received written information describing the purpose and procedures of the study and provided written informed consent prior to participation, including consent for the processing of their personal data. Participation was voluntary, and participants could withdraw their consent at any time without adverse consequences. Skin images were associated with generic participant identifiers, while the correspondence between these identifiers and the contact information of the participants was stored separately under restricted access available only to the research team. Contact information was used solely when a participant needed to be contacted for a repeated acquisition. Only non-identifying skin-patch images were used for scientific dissemination, and any additional compatible research use requires prior anonymization. Personal data were handled according to applicable international regulations.

## 6.1. Fitting Results

We apply our inverse rendering pipeline to real facial skin patches sampled across the Monk Skin Tone Scale. We evaluate reconstruction quality using the perceptual metric FLIP [51], computed over masked regions to isolate the subsurface scattering halo. Fig. 12 shows representative examples from light, medium, and dark skin tones. The reconstructions closely match the input photographs, particularly in the shadowed transition regions where scattering dominates.

We report complementary metrics—LPIPS [52], SSIM [53], and PSNR—in Table 2. While SSIM is sensitive to luminance variations in shadowed regions (resulting in lower scores), perceptual metrics like FLIP and LPIPS yield consistently low error values. This supports the perceptual plausibility of our reconstructions across the full skin tone spectrum.

Table 2  
![](images/e4afcd7752d7870919af82fdf611b6d730245802e4a4df150854f36f50c9b934.jpg)  
Figure 12: Reconstruction results on real facial skin patches across the Monk Skin Tone Scale. Rows display diferent facial regions: neck (top), cheek (middle), and nose (bottom). Columns show (left to right): the captured input image with strip lighting, the rendered reconstruction, the masked input image, the masked render, and the masked perceptual error map (FLIP). The optimized scattering distance for each patch is indicated as $d _ { o p t } .$

Quantitative reconstruction errors across the Monk Scale, computed on masked image regions.
<table><tr><td>Monk Range</td><td>FLIP ↓</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td></tr><tr><td>Monk 1-3</td><td> $0 . 0 9 3 \pm 0 . 0 2 6$ </td><td> $0 . 0 8 6 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 5 \pm 0 . 0 6 6$ </td><td> $2 8 . 6 1 \pm 1 . 6 8$ </td></tr><tr><td>Monk 4-6</td><td> $0 . 0 9 3 \pm 0 . 0 3 1$ </td><td> $0 . 0 8 8 \pm 0 . 0 1 8$ </td><td> $0 . 6 1 4 \pm 0 . 0 6 6$ </td><td> $2 8 . 6 7 \pm 1 . 8 9$ </td></tr><tr><td>Monk 7-10</td><td> $0 . 0 7 7 \pm 0 . 0 2 1$ </td><td> $0 . 1 1 5 \pm 0 . 0 1 5$ </td><td> $0 . 5 0 3 \pm 0 . 0 7 2$ </td><td> $2 9 . 6 5 \pm 1 . 3 5$ </td></tr></table>

Table 3

Computational cost per skin patch $( 4 8 0 \times 3 6 0 ~ \mathsf { p x } ,$ $N = 1 2 8$ scattering samples) across optimization stages on an AMD Ryzen 9 7900 CPU (12 cores, 24 threads) and an NVIDIA GeForce RTX 4070 Ti GPU. The Overall column reports cumulative values for time and iterations, mean iteration latency, and maximum peak VRAM.
<table><tr><td>Metric</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Overall</td></tr><tr><td>Execution Time (s)</td><td>51.1</td><td>38.3</td><td>109.0</td><td>198.4</td></tr><tr><td>Iterations</td><td>700</td><td>500</td><td>600</td><td>1800</td></tr><tr><td>Time per Iteration (ms)</td><td>72.9</td><td>76.6</td><td>181.7</td><td>110.2</td></tr><tr><td>Peak VRAM (GB)</td><td>2.5</td><td>2.6</td><td>9.4</td><td>9.4</td></tr></table>

In addition to perceptual quality, we benchmark the computational eficiency of our pipeline. Because all skin patches are standardized to $4 8 0 \times 3 6 0$ px and evaluated under our fixed multi-stage schedule, runtime is deterministic across subjects. Table 3 details the stage-wise computational breakdown on an NVIDIA GeForce RTX 4070 Ti (12 GB). Inverse optimization converges in under 3.5 minutes (198.4 s) per patch with a peak footprint of 9.4 GB during Stage 3 integration, confirming that our pipeline enables fast and accessible digitization on consumer desktop hardware.

We visualize the optimized Burley 15 material stack alongside the auxiliary residual specular map in Fig. 13.

![](images/7a62b0c0f93878a6126eb5071bcb431664f05d51234432249dd62a72e436c75e.jpg)

Figure 13: Optimized Burley 15 material stack across diferent skin tones from single-view, single-light capture. Columns show (left to right): full-light input image, estimated albedo, roughness and residual specular.  
![](images/abbf43b9ff72a897b2f2545e98121de4d85a1ec278ef76b0b53a285fe47f2c68.jpg)  
Figure 14: Radial scattering profiles plotted as $r R ( r )$ in logarithmic space to visualize fallof and efective range. From left to right: light, medium, and dark skin tones. Curves show channel-wise reflectance decay (RGB) alongside estimated albedo. The trend of decreasing scattering distance with darker skin tones confirms the physical plausibility of our optimization.

The estimated textures are spatially smooth and consistent across the dataset. Utilizing a monocular depth-based normal prior, rather than relying on photometric cues, regularizes the single-view optimization, mitigating parameter instability across complex structures like the ear. While this depth prior acts as a low-frequency approximation that lacks sharp pore-level details—yielding smoother roughness maps—and screen-space shading partially bakes self-shadows into albedo across steep relief (e.g., ear folds), our multi-light validation confirms the scattering parameter � converges consistently. This demonstrates that subsurface optimization remains efectively decoupled from localized shadowing and microgeometric inaccuracies.

Crucially, the estimated scattering distance maps exhibit intuitive trends across the skin tone spectrum. As illustrated in Fig. 14, darker skin tones correspond to shorter scattering distances, consistent with the reduced subsurface light transport in melanin-rich tissue. These findings demonstrate the physical plausibility of our RGB-based optimizations without the need for hyperspectral data. Furthermore, our masking strategy efectively focuses supervision on shadowed regions, improving multiple scattering estimates in darker skin where the efect is subtle but directionally dependent.

![](images/9491137a301065f228fb9e22fa5dac66e7c338cba7532c03a18c9c6c2fbb9ae6.jpg)  
Figure 15: Synthetic validation using known ground truth. $d _ { \mathrm { r e f } }$ denotes the ground-truth scattering distance, and $d _ { \mathsf { o p t } }$ is the optimized value. The near-identical maps confirm the numerical self-consistency of � under the inverse rendering formulation.

## 6.2. Validation

Validation on Synthetic Skin Patches We validate our pipeline using synthetic skin patches rendered under known Burley parameters [4] with an identical sampling budget $( N = 1 2 8 )$ to ensure numerical consistency during inverse optimization. In the first experiment, we fix all surface attributes (albedo, normals, roughness, IOR) and optimize only the scattering distance. This isolates the scattering estimation stage. As shown in Fig. 15, the estimated scattering maps $( d _ { o p t } )$ closely match the reference ground truth $( d _ { r e f } )$ , and the rendered reconstructions are visually indistinguishable from the inputs.

In the second experiment, we fix the scattering distance and vary the albedo across the ten synthetic tones of the Monk Scale to test robustness to pigmentation changes. While Figure 16 shows a representative subset, Section 4 of the Supplementary Material details the full 10-tone evaluation alongside the real-world dataset distribution (� = 45) crossreferenced against participant self-assessments.

Specular Isolation Validation Our pipeline employs a residual specular map to absorb high-frequency highlights, mitigating their leakage into the albedo map. To validate this strategy, we optimize a synthetic scene where the groundtruth decomposition between albedo and specular reflection is known. Fig. 17 compares the optimized residual map against the ground-truth specular contribution. The close resemblance confirms that our robust mean albedo computation efectively shields the difuse component, allowing the residual map to accurately capture the specular response. Furthermore, this separation minimizes the risk of unmodeled specular energy corrupting other surface attributes, directly validating why our framework achieves highly stable and smooth roughness maps with suppressed artifact bleeding.

Masked Loss Validation To isolate the contribution of our shadow-masked loss during Stage $3 ( \mathcal { M } _ { \mathrm { o p t } } = \mathcal { M } )$ , we performed an ablation study comparing this formulation against unmasked full-image loss supervision $( \mathcal { M } _ { \mathrm { o p t } } = 1 )$ under identical rendering conditions (Figure 18). In the unmasked setting, incident irradiance across directly illuminated stripes is orders of magnitude higher than within shadowed regions. Consequently, backpropagation gradients are dominated by surface reflectance components (difuse albedo and specular highlights), introducing severe gradient competition against the subtle, low-frequency subsurface difusion halo. The optimizer attempts to compensate for subtle direct-light discrepancies by skewing the scattering distance parameter �.

![](images/b11ad82226b8a0eab0ad18b36badc772e66964ceee3be394ea09538584b31ef4.jpg)

Figure 16: Synthetic validation across the Monk Scale. We fix scattering distance and vary albedo (Monk 01, 05, 10). The low perceptual error (FLIP) and consistent parameter recovery confirm robustness to albedo variation.  
![](images/c600ef4dca3e754e6f43b88a959a17ae1b1db3d192a49022bfc9a3c41fc23738.jpg)  
Figure 17: Validation of specular isolation. We compare the ground-truth specular contribution (Reference) against the optimized residual specular map (Render). The low error demonstrates that our albedo initialization enables the optimizer to efectively capture and absorb specular reflections within the residual map, mitigating their contamination of the base material parameters.

Interestingly, while the relative spectral proportionality $( d _ { r } > d _ { g } > d _ { b } )$ is naturally preserved due to the lower absorption of melanin and hemoglobin at longer wavelengths, the unmasked optimization yields an artificially saturated and magnitude-shifted �. This magnitude shift truncates the difusion tail along transition boundaries. In contrast, restricting the loss domain strictly to the binary shadow mask  decouples parameter estimation from direct surface illumination, forcing the objective function to evaluate the pure radial decay of multiple scattering. This targeted supervision eliminates parameter overcompensation, ensuring that the estimated parameter � yields a reconstruction closer to the reference capture, decreasing the perceptual FLIP error from 0.11 to 0.06 (45% reduction) in shadowed transition regions, and from 0.15 to 0.10 across the full patch.

![](images/8231cc593d337d4766c11764ee4a3a1bdbc451c86ea40dc0bdcffd32fca51801.jpg)  
Figure 18: Ablation of Stage 3 loss supervision domain. Comparing unmasked optimization $( \mathcal { M } _ { \mathrm { o p t } } = \mathbf { 1 }$ , top) against our shadow-masked loss $( \mathcal { M } _ { \mathrm { o p t } } = \mathcal { M }$ , bottom). Supervising unlit regions isolates subsurface halos, preventing direct light from skewing the scattering distance $d _ { \mathsf { o p t } }$ (insets) and reducing FLIP error across both the full patch and masked shadow regions.

Comparison with Multi-Light Acquisition To evaluate robustness under real-world conditions, we compare our single-light results against a multi-light setup (three synchronized projectors). This configuration allows us to introduce an additional preliminary optimization stage to the pipeline. In this new Stage 1, we leverage the multi-light observations to specifically optimize and refine the initial Sobel-filtered normal map, improving geometric consistency before proceeding with the standard material estimation steps (now shifted to Stages 2, 3, and 4).

We compare the scattering distance maps estimated from our standard single-light pipeline against those produced using this augmented four-stage multi-light configuration. Despite the improved geometric detail provided by the refined normals, the single-light method yields scattering distances that are visually and numerically consistent (mean $\mathcal { L } _ { 1 }$ diference < 2%). As shown in Fig. 19, discrepancies are minimal and confined to occluded regions.

Fig. 20 illustrates the material maps derived from the multi-light setup. By leveraging additional lighting data, this configuration improves the estimation of surface properties—particularly roughness—and better handles challenging facial regions such as the nose or ears. Although the setup is still constrained by the limited angular coverage of the three projectors, the multi-light approach provides more detailed material stacks. However, the recovered scattering distance parameters and the resulting rendering reconstructions are nearly identical to those produced by the single-view baseline. These results confirm that for the specific task of estimating subsurface scattering, our single-light, single-view approach remains both suficient and robust.

![](images/6697dfe8ee255fb20f2e17e77ba4b077f15cb92fd6719a3df2cc4059da679bb4.jpg)

Figure 19: Multi-light results for the skin patches analyzed in Fig. 12. Each row shows (left to right): photo, render, masked pairs, and FLIP map. Consistency with single-view $d _ { o p t }$ validates our simplified setup’s robustness despite additional lighting data.  
![](images/58f8bb5ef3972f95d68923f0f78b0c2bca7db198c03ac4eb82c4a8ef8a126611.jpg)  
Figure 20: Optimized material stack from the multi-light setup. Extra lighting data enables a more detailed roughness map and cleaner parameter separation, reducing the information baked into the residual specular map.

## 7. Conclusions

We presented a lightweight diferentiable inverse rendering pipeline for estimating subsurface scattering parameters of human skin directly from single-view, single-light captures. By optimizing the same parameters used in standard realtime shading models, our method establishes a direct bridge between physical acquisition and production-ready rendering pipelines, eliminating the need for complex optical setups or intermediate physical volumetric parameter conversion.

Our experiments demonstrate that physically plausible scattering distances can be estimated from minimal input data. The recovered parameters produce a realistic skin appearance across a wide range of skin tones, exhibiting consistent numerical convergence in synthetic benchmarks and high perceptual reconstruction fidelity on real-world captures. The proposed shadow-aware loss formulation improves robustness in regions dominated by multiple scattering, enabling reliable estimation even under challenging pigmentation conditions. Furthermore, our dataset includes a diverse collection of calibrated skin samples aligned with the Monk Skin Tone Scale, supporting inclusive and scalable digital human modeling.

A key result of this work is that production-ready subsurface scattering parameters can be estimated from an accessible capture setup without relying on laboratory-grade acquisition systems. Rather than requiring specialized hardware or complex measurement procedures, our approach makes the problem tractable through diferentiable optimization, physically motivated parameterization, and carefully designed supervision. This substantially lowers the barrier to high-quality skin digitization while preserving compatibility with real-time rendering workflows.

The method relies on a locally homogeneous scattering assumption, and monocular depth priors. These assumptions allow the inverse problem to remain practical while still producing parameters that are physically plausible and visually efective. In the single-light setting, the ambiguity between albedo and scattering is addressed through a multi-stage optimization strategy that stabilizes parameter recovery. In thin or highly translucent structures, such as nose cartilage or the ear helix, omitting explicit single-scattering optimization may lead to slightly larger estimated scattering distances, as the model explains missing translucency through the available parameters. Our regional blending scheme constrains these estimates within realistic ranges and preserves a consistent appearance across facial regions. Similarly, omitting nonlocal visibility handling causes residual self-shadows to be partially absorbed into the albedo in areas of steep relief.

Future work can build on this foundation by jointly recovering additional scattering components, incorporating stronger learned priors, integrating diferentiable visibility testing to resolve self-shadowing, and extending the capture protocol to less constrained environments. More flexible acquisition strategies could further improve the fidelity of reconstruction while maintaining the accessibility of the proposed setup.

By combining accessible acquisition, diferentiable optimization, physically grounded constraints, and validation across diverse skin tones, our method provides a scalable path toward realistic and inclusive digital human creation for real-time applications.

## Acknowledgments

This work has been supported by the Spanish Agencia Estatal de Investigación (AEI) under Project PID2024-159930OB-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERDF, EU

## References

[1] H. Ha, I. Hwang, N. Monzon, J. Cho, D. Kim, S.-H. Baek, A. Muñoz, D. Gutierrez, M. H. Kim, Polarimetric BSSRDF Acquisition of Dynamic Faces, ACM Transactions on Graphics 43 (6) (2024) 1–11. doi:10.1145/3687767.

[2] C. Aliaga, M. Xia, X. Xie, A. Jarabo, G. Braun, C. Hery, A Hyperspectral Space of Skin Tones for Inverse Rendering of Biophysical Skin Properties, Computer Graphics Forum 42 (4) (7 2023). doi: 10.1111/cgf.14887.

[3] X. Li, G. C. Guarnera, A. Lin, A. Ghosh, Practical Measurement and Neural Encoding of Hyperspectral Skin Reflectance, in: 2024 International Conference on 3D Vision (3DV), IEEE, 2024, pp. 1301– 1309. doi:10.1109/3DV62453.2024.00116.

[4] B. Burley, Extending the Disney BRDF to a BSDF with integrated subsurface scattering, SIGGRAPH Course: Physically Based Shading in Theory and Practice. ACM, New York, NY 19 (7) (2015) 9.

[5] E. Monk, The monk skin tone scale (2023).

[6] T. B. Fitzpatrick, The Validity and Practicality of Sun-Reactive Skin Types I Through VI, Archives of Dermatology 124 (6) (1988) 869. doi:10.1001/archderm.1988.01670060015008.

[7] C. Che, F. Luan, S. Zhao, K. Bala, I. Gkioulekas, Towards Learningbased Inverse Subsurface Scattering, in: 2020 IEEE International Conference on Computational Photography (ICCP), IEEE, 2020, pp. 1–12. doi:10.1109/ICCP48838.2020.9105209.

[8] H. W. Jensen, S. R. Marschner, M. Levoy, P. Hanrahan, A practical model for subsurface light transport, in: Proceedings of the 28th annual conference on Computer graphics and interactive techniques, ACM, New York, NY, USA, 2001, pp. 511–518. doi:10.1145/383259.383319.

[9] C. Donner, H. W. Jensen, Light difusion in multi-layered translucent materials, ACM Transactions on Graphics 24 (3) (2005) 1032–1039. doi:10.1145/1073204.1073308.

[10] E. d’Eon, A better dipole, Eugene d’Eon 14 (2012).

[11] J. R. Frisvad, T. Hachisuka, T. K. Kjeldsen, Directional Dipole Model for Subsurface Scattering, ACM Transactions on Graphics 34 (1) (2014) 1–12. doi:10.1145/2682629.

[12] E. D’Eon, G. Irving, A quantized-difusion model for rendering translucent materials, ACM Transactions on Graphics 30 (4) (2011) 1–14. doi:10.1145/2010324.1964951.

[13] Y. N. Mishra, Droplet Size, Concentration, and Temperature Mapping in Sprays Using SLIPI-Based Techniques, Ph.D. thesis, Lund University (2018).

[14] C. Barré-Brisebois, M. Bouchard, Approximating Translucency for a Fast Cheap and Convincing Subsurface Scattering Look, in: Game developers conference, Vol. 6, 2011.

[15] J. Jimenez, D. Gutierrez, Screen-Space Subsurface Scattering, in: W. Engel (Ed.), GPU Pro: Advanced Rendering Techniques, AK Peters Ltd., 2010, pp. 335–351.

[16] J. Jimenez, K. Zsolnai, A. Jarabo, C. Freude, T. Auzinger, X. Wu, J. von der Pahlen, M. Wimmer, D. Gutierrez, Separable Subsurface Scattering, Computer Graphics Forum 34 (6) (2015) 188–197. doi: 10.1111/cgf.12529.

[17] P. H. Christensen, An approximate reflectance profile for eficient subsurface scattering, in: ACM SIGGRAPH 2015 Talks, ACM, New York, NY, USA, 2015, pp. 1–1. doi:10.1145/2775280.2792555.

[18] T. TG, D. M. Tran, H. W. Jensen, R. Ramamoorthi, J. R. Frisvad, Neural SSS: Lightweight Object Appearance Representation, Computer Graphics Forum 43 (4) (7 2024). doi:10.1111/cgf.15158.

[19] T. TG, J. R. Frisvad, R. Ramamoorthi, H. W. Jensen, NeuPreSS: Compact Neural Precomputed Subsurface Scattering for Distant Lighting of Heterogeneous Translucent Objects, Computer Graphics Forum 43 (7) (10 2024). doi:10.1111/cgf.15234.

[20] T. Weyrich, W. Matusik, H. Pfister, J. Lee, A. Ngan, H. W. Jensen, M. Gross, A measurement-based skin reflectance model for face rendering and editing, None TR 71 (4) (2005) 1–13.

[21] A. Ghosh, T. Hawkins, P. Peers, S. Frederiksen, P. Debevec, Practical modeling and acquisition of layered facial reflectance, ACM Transactions on Graphics 27 (5) (2008) 1–10. doi:10.1145/1409060.1409092.

[22] S. Tariq, A. Gardner, I. Llamas, A. Jones, P. Debevec, G. Turk, Eficient estimation of spatially varying subsurface scattering parameters for relighting, Tech. rep. (2006).

[23] D. Lanza, J. R. Padrón-Grife, A. Pranovich, A. Muñoz, J. R. Frisvad, A. Jarabo, Practical Appearance Model for Foundation Cosmetics, Computer Graphics Forum 43 (4) (7 2024). doi:10.1111/cgf.15148.

[24] X. He, M. Huang, R. Fu, J. Guo, J. Yuan, Y. Wang, Y. Guo, Biophysically-based Simulation of Sun-induced Skin Appearance Changes, in: R. Chen, T. Ritschel, E. Whiting (Eds.), Pacific Graphics Conference Papers and Posters, The Eurographics Association, 2024. doi:10.2312/pg.20241286.

[25] Q. Wang, F. Luan, Y. Dai, Y. Huo, H. Bao, R. Wang, A biophysicalbased skin model for heterogeneous volume rendering, Computational Visual Media 11 (2) (2025) 289–303.

[26] B. Burley, W. D. A. Studios, Physically-based shading at disney, in: Acm siggraph, Vol. 2012, 2012, pp. 1–7.

[27] T. Yatagawa, H. Todo, Y. Yamaguchi, S. Morishima, Data compression for measured heterogeneous subsurface scattering via scattering profile blending, The Visual Computer 36 (3) (2020) 541–558. doi:10.1007/ s00371-018-01626-x.

[28] P. Debevec, T. Hawkins, C. Tchou, H.-P. Duiker, W. Sarokin, M. Sagar, Acquiring the reflectance field of a human face, in: Proceedings of the 27th annual conference on Computer graphics and interactive techniques - SIGGRAPH ’00, ACM Press, New York, New York, USA, 2000, pp. 145–156. doi:10.1145/344779.344855.

[29] A. Ghosh, G. Fyfe, B. Tunwattanapong, J. Busch, X. Yu, P. Debevec, Multiview face capture using polarized spherical gradient illumination, ACM Transactions on Graphics 30 (6) (2011) 1–10. doi:10.1145/ 2070781.2024163.

[30] W.-C. Ma, T. Hawkins, P. Peers, C.-F. Chabert, M. Weiss, P. E. Debevec, others, Rapid Acquisition of Specular and Difuse Normal Maps from Polarized Spherical Gradient Illumination., Rendering Techniques 9 (10) (2007) 2.

[31] G. Fyfe, K. Nagano, L. Huynh, S. Saito, J. Busch, A. Jones, H. Li, P. Debevec, Multi-View Stereo on Consistent Face Topology, Computer Graphics Forum 36 (2) (2017) 295–309. doi:10.1111/cgf.13127.

[32] Y. Gitlina, G. C. Guarnera, D. S. Dhillon, J. Hansen, A. Lattas, D. Pai, A. Ghosh, Practical Measurement and Reconstruction of Spectral Skin Reflectance, Computer Graphics Forum 39 (4) (2020) 75–89. doi:10.1111/cgf.14055.

[33] J. Riviere, P. Gotardo, D. Bradley, A. Ghosh, T. Beeler, Single-shot high-quality facial geometry and skin appearance capture, ACM Transactions on Graphics 39 (4) (8 2020). doi:10.1145/3386569. 3392464.

[34] V. Deschaintre, M. Aittala, F. Durand, G. Drettakis, A. Bousseau, Single-image SVBRDF capture with a rendering-aware deep network, ACM Transactions on Graphics 37 (4) (2018) 1–15. doi:10.1145/ 3197517.3201378.

[35] X. Zhou, M. Hašan, V. Deschaintre, P. Guerrero, K. Sunkavalli, N. K. Kalantari, A Semi-Procedural Convolutional Material Prior, Computer Graphics Forum 42 (6) (9 2023). doi:10.1111/cgf.14781.

[36] C. Rodriguez-Pardo, D. Pascual-Hernandez, J. Rodriguez-Vazquez, J. Lopez-Moreno, E. Garces, Single-image reflectance and transmittance estimation from any flatbed scanner, Computers & Graphics 127 (2025) 104186. doi:10.1016/j.cag.2025.104186.

[37] E. Garces, V. Arellano, C. Rodriguez-Pardo, D. Pascual-Hernandez, S. Suja, J. Lopez-Moreno, Towards Material Digitization with a Dualscale Optical System, ACM Transactions on Graphics 42 (4) (2023) 1–13. doi:10.1145/3592147.

[38] P. Weier, J. Riviere, R. Guseinov, S. Garbin, P. Slusallek, B. Bickel, T. Beeler, D. Vicini, Practical Inverse Rendering of Textured and Translucent Appearance, ACM Transactions on Graphics 44 (4) (2025) 1–16. doi:10.1145/3730855.

[39] V. Bruce, A. Young, Understanding face recognition, British journal of psychology 77 (3) (1986) 305–327.

[40] S. L. Jacques, Optical properties of biological tissues: a review, Physics in Medicine & Biology 58 (11) (2013) R37.

[41] J. Moreau-Mathis, Babylon. js Essentials, Packt Publishing Ltd, 2016.

[42] E. Golubev, Eficient screen-space subsurface scattering using Burley’s normalized difusion in real-time, ACM SIGGRAPH Courses: Advances in Real-Time Rendering in Games Course (2018).

[43] J. H. Hannay, J. F. Nye, Fibonacci numerical integration on a sphere, Journal of Physics A: Mathematical and General 37 (48) (2004) 11591– 11601.

[44] P. Gotardo, J. Riviere, D. Bradley, A. Ghosh, T. Beeler, Practical dynamic facial appearance modeling and acquisition, ACM Transactions on Graphics 37 (6) (2018) 1–13. doi:10.1145/3272127.3275073.

[45] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao, Depth Anything V2 (2024). URL https://arxiv.org/abs/2406.09414

[46] R. J. Woodham, Photometric method for determining surface orientation from multiple images, Optical engineering 19 (1) (1980) 139–144.

[47] J. Salvi, J. Pagès, J. Batlle, Pattern codification strategies in structured light systems, Pattern Recognition 37 (4) (2004) 827–849. doi: 10.1016/j.patcog.2003.10.002.

[48] C. Wu, B. Wilburn, Y. Matsushita, C. Theobalt, High-quality shape from multi-view stereo and shading under general illumination, in: CVPR 2011, IEEE, 2011, pp. 969–976. doi:10.1109/CVPR.2011. 5995388.

[49] N. Otsu, A Threshold Selection Method from Gray-Level Histograms, IEEE Transactions on Systems, Man, and Cybernetics 9 (1) (1979) 62–66. doi:10.1109/TSMC.1979.4310076.

[50] D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization (1 2017). doi:10.48550/arXiv.1412.6980.

[51] P. Andersson, J. Nilsson, T. Akenine-Möller, M. Oskarsson, K. Åström, M. D. Fairchild, FLIP, Proceedings of the ACM on Computer Graphics and Interactive Techniques 3 (2) (2020) 1–23. doi:10.1145/3406183.

[52] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The Unreasonable Efectiveness of Deep Features as a Perceptual Metric, in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2018, pp. 586–595. doi:10.1109/CVPR.2018.00068.

[53] Zhou Wang, A. Bovik, H. Sheikh, E. Simoncelli, Image quality assessment: from error visibility to structural similarity, IEEE Transactions on Image Processing 13 (4) (2004) 600–612. doi:10.1109/TIP.2003. 819861.

# A Lightweight Diferentiable Method for Real-Time Skin Rendering and Digitization

Supplementary Material

Mario Marquez-Balduque<sup>1</sup>, Carlos Rodríguez-Pardo<sup>2,3,4</sup>, Jorge Lopez-Moreno<sup>1</sup>

<sup>1</sup>Universidad Rey Juan Carlos, Spain

<sup>2</sup>Politecnico di Milano, Milan, Italy

<sup>3</sup>Euro-Mediterranean Center on Climate Change (CMCC), Lecce, Italy

<sup>4</sup>RFF-CMCC European Institute on Economics and the Environment (EIEE), Milan, Italy

## 1. Introduction

This supplementary document provides extended validation, dataset analysis, and core implementation details supporting the main manuscript, organized as follows:

• Map Evolution Across Optimization Stages (Section 2): Illustrates the step-by-step visual progression of active and frozen maps across optimization stages, justifying our staged schedule in decoupling surface reflections and subsurface difusion.

• Tilted View Validation (Section 3): Demonstrates the angular consistency and physical grounding of our 3D depth-aware optimization under varying camera orientations.

• Dataset across the Monk Scale (Section 4): Presents comprehensive synthetic validation across the full Monk Skin Tone Scale by isolating albedo variations under fixed scattering parameters, and evaluates realworld participant distributions by cross-referencing subjective self-identification against our objective color-matching algorithm via a unified confusion matrix.

• Replicability and Algorithmic Details (Section 5): Provides concrete pipeline blueprints, including RAW color calibration via Ridge Regression (Section 5.1), robust albedo initialization (Section 5.2), and our GPU shader (Section 5.3) and diferentiable tensorized implementations (Section 5.4).

## 2. Map Progression Across Optimization Stages

To complement the optimization methodology described in the main manuscript, Figure 1 illustrates the step-by-step visual progression of active and frozen material maps across our three optimization stages.

During Stage 1, freezing the albedo ${ \bf A } _ { T }$ forces the reconstruction loss to temporarily drive the active roughness map $( \rho _ { T } )$ to fit low-frequency difuse shading gradients. In Stage 2, unfreezing ${ \bf A } _ { T }$ provides the direct path to resolve these difuse reconstruction errors. As the albedo rapidly absorbs the spatial shading details, the reconstruction gradient on other maps vanishes. Consequently, the $ { \mathcal { L } } _ { 2 }$ roughness prior smoothly regularizes $\rho _ { T }$ while leaving the residual specular map virtually unperturbed, matching its role as a bufer designed for unmodeled specular highlights rather than difuse light transport.

![](images/90bbe362d5f56eb9c47f535b284aa47fabdc0798a9db33eda66be37b06b14914.jpg)  
Figure 1: Map progression across optimization stages. Green and red denote active and frozen maps, respectively. Stage 1 fits global reflection, capturing unmodeled specular highlights into the residual map while shielding initialized albedo. Stage 2 unfreezes albedo to absorb difuse spatial details, enabling the $\mathcal { L } _ { 2 }$ prior to smooth roughness while preserving the specular residual. Stage 3 optimizes scattering distance over shadowmasked regions (Input) with surface maps frozen.

## 3. Tilted View Validation

To verify that our frontal-view optimization yields physically grounded parameters rather than view-dependent overfitted solutions, we perform tilted-view cross-validation on both synthetic and real data. For synthetic targets, we optimize parameters independently from frontal and novel tilted views, verifying that the estimated scattering distance $( d _ { \mathrm { o p t } } )$ consistently matches the ground truth $( d _ { \mathrm { r e f } } )$ . For real targets, we evaluate acquisitions of identical skin regions under varying camera angles. As shown in Fig. 2, estimated scattering distances remain invariant across orientations, yielding minimal reconstruction error.

Our pipeline avoids 2D projection approximations by executing scattering calculations directly in 3D view space using the estimated geometric depth. This integration ensures that the Screen-Space Subsurface Scattering (SSSS) algorithm naturally accounts for surface curvature and perspective foreshortening. By computing physical distances rather than simple 2D pixel ofsets, the model recovers orientationinvariant scattering profiles.

![](images/d90efebeea33e0fa3d52d10a2e844b87b069d3b54edc1a4659a1bb822341a5d0.jpg)  
Figure 2: Tilted view validation. In synthetic cases (top), $d _ { \mathsf { o p t } }$ closely tracks $d _ { \mathrm { r e f } } .$ For real captures (bottom), scattering distances remain stable across views, demonstrating that 3D depth-aware computations prevent parameter shrinkage caused by perspective foreshortening.

## 4. Dataset across the Monk Scale

Figure 3 presents the full synthetic validation workflow, confirming optimization stability across all ten Monk classes. By fixing the ground-truth scattering distance and varying only the target albedo, we verify that our method decouples scattering parameters independently of surface reflectance.

Figure 4 cross-references participant skin tone selfassessments against our automatically computed baseline using a unified confusion matrix $( N = 4 5 )$ . As expected from a Mediterranean–Caucasian population, the sample distribution concentrates around intermediate Monk tones (3–6), highlighting the representative challenges of realworld skin digitization. For automated classification, the mean skin color is extracted from the neutral frontal-light capture to eliminate specular and shadow contamination, and subsequently mapped via CIE LAB distance to uniformly lit reference exemplars from the MST dataset.

Discrepancies between computed tones and selfassessments in the matrix reveal inherent human perception biases and visual ambiguities when isolating pure albedo from overall lightness under casual observation. Furthermore, the original Monk scale is inherently perceptual, with reference colors defined under uncalibrated lighting conditions. Consequently, standardized computational environments and objective physical optimization pipelines are essential for consistent, illumination-invariant tone classification.

## 5. Reproducibility and Algorithmic Details

## 5.1. RAW Processing and Color Calibration

RAW images (6020 × 4024 px) are decoded via rawpy, bypassing all non-linear ISP operations to enforce a linear gamma $( \gamma = 1 . 0 )$ and native camera color space. Normalized 16-bit float intensities ([0, 1]) are downsampled using bicubic interpolation to 1204 × 804 px to preserve global spatial context, and center-cropped to a final operating resolution of $4 8 0 \times 3 6 0 \mathrm { p x }$

To map the linear RAW space to standard D65 linear sRGB, we extract the mean RAW intensity of the 24 patches from an X-Rite ColorChecker Classic target (Fig. 5) to build the measurement matrix �. Target sRGB reference values are linearized via gamma decoding $( \gamma = 2 . 2 )$ to form matrix �.

To suppress noise amplification, the $3 \times 3$ Color Correction Matrix (CCM) � is computed channel-wise using  -regularized Ridge Regression. For each target channel column $\mathbf { y } _ { c } \left( c \in \left\{ r , g , b \right\} \right)$ , the optimization yields:

$$
\mathbf { c } _ { c } = \left( \mathbf { X } ^ { T } \mathbf { X } + \alpha \mathbf { I } \right) ^ { - 1 } \mathbf { X } ^ { T } \mathbf { y } _ { c }\tag{1}
$$

where $\alpha ~ = ~ 1 0 ^ { - 3 }$ and � is the identity matrix. The final matrix $\mathbf { C } = [ \mathbf { c } _ { r } ^ { T } ; \mathbf { c } _ { g } ^ { T } ; \mathbf { c } _ { b } ^ { T } ]$ maps the image via $\mathbf { I } _ { \mathrm { l i n e a r } } = \mathbf { I } _ { \mathrm { r a w } } \mathbf { C } ^ { T }$ Inverse gamma encoding $( 1 / 2 . 2 )$ is applied strictly for display previews.

## 5.2. Albedo Initialization

Initializing the albedo from a flat-lit, difuse-only baseline is critical for downstream parameter decoupling. Arbitrary or global-mean initialization risk optimization collapse into local minima due to facial features, hair, or micro-shadow contamination. If specular highlights contaminate the initial albedo estimate, the optimization struggles to isolate independent light transport components. Securing a clean, purely difuse baseline guarantees that high-frequency specular details are absorbed into the residual specular map rather than getting baked into the estimated albedo or roughness maps.

We perform our initialization strategy in two stages (Algorithm 1):

![](images/b7d61a70445d2f90d92163efcbe55412a9f46cf2320d4cbed61d56afe0efb90b.jpg)  
Figure 3: Full synthetic validation across the Monk Skin Tone Scale. Per tone (left to right): input, optimized reconstruction, and FLIP error map. Inset swatches denote $d _ { \mathrm { r e f } }$ (known reference value) and $d _ { \mathsf { o p t } }$ (optimized value). Low perceptual errors confirm estimation invariance across diverse skin tones.

![](images/d6e8889f73152965960b993575ef9230bc8f5b1d86752db390072fef2b375abf.jpg)  
Figure 4: Unified confusion matrix cross-referencing participant self-identified (Perceived) skin tones against our automatic appearance-based color-matching algorithm (Computed) across the Monk Skin Tone Scale (� = 45). The clustering behavior highlights systematic discrepancies between subjective human self-assessment and objective physical albedo extraction, driven by visual perception biases and the uncalibrated nature of the original scale.

![](images/938ff45cb85663568730cd007ed4daf1d4236c50bd6e1eeed046b9151f5b03fc.jpg)  
Figure 5: Physical color calibration setup under the projector’s flat-field white illumination. The 24 patch coordinates are automatically extracted to fit the 3 × 3 CCM.

Algorithm 1 Adaptive Albedo Initialization Algorithm 2 Burley Multiple Scattering Shader   
Require: Require:   
� : Normalized RGB input image � : 2D screen-space texture coordinates [�, �]   
Ensure: � : Screen-space scattering map storing spatially blended $d _ { \mathrm { o p t } }$ values   
$\mathbf { c } _ { \mathrm { i n i t } }$ : Robust initialization RGB color vector � : Screen-space depth texture map   
1: function E A S M (�) � : Screen-space surface albedo texture map   
⊳ Step 1: Color Space Transformation � : Screen-space difuse irradiance texture map   
2: $\mathbf { I } _ { \mathrm { l a b } } $ ConvertRGBToLab(�) � : Sample count ∈ [2, 128]   
������� : Array of vec2(�, �) (Fibonacci angles, Stratified Burley   
⊳ Step 2: Background Discarding & Feature Extraction Inverse CCDF normalized radii (Eq. 2))   
3: $\mathcal { M } _ { \mathrm { v a l i d } } ^ { \mathrm { ~ - ~ } }  \{ ( x , y ) \mid \mathbf { I } _ { \mathrm { l a b } } [ x , y , L ] > \bar { 2 } 0 \}$ $\mathbf { M } _ { \mathrm { p r o j } } , \mathbf { M } _ { \mathrm { p r o j } } ^ { - 1 }$ : Camera projection matrix and its inverse   
4: $\mathbf { P } \gets \mathbf { \mathbf { I } } _ { \mathrm { l a b } } [ \mathcal { M } _ { \mathrm { v a l i d } } , ( a , b ) ]$ Ensure:   
� : Scattered difuse radiance vector for the target fragment   
⊳ Step 3: Chromatic Feature Clustering 1: procedure EVALUATEMULTIPLESCATTERING   
5: $c _ { 1 } , c _ { 2 } \gets$ ExecuteKMeans(�, $K = 2 )$ ⊳ Subsurface Scattering Setup   
2: $w _ { \mathrm { s c a l e } }  0 . 0 1 , \quad \epsilon  1 0 ^ { - 6 }$   
⊳ Step 4: Skin Cluster Identification & Mapping 3: $d \gets S ( u ) . r g b \times w _ { \mathrm { s c a l e } }$   
6: $i d x _ { \mathrm { s k i n } } $ argmax(GetCentroi $\boldsymbol { \mathrm { I } } ( \boldsymbol { C } _ { k } ) [ \boldsymbol { a } ] )$ 4: $r _ { \mathrm { m a x } } \gets \operatorname* { m a x } ( d _ { r } , d _ { g } , d _ { b } )$   
7: $\mathcal { M } _ { \mathrm { r a w } }  \mathrm { M a p T o I m a g e S p a c e } ( C _ { i d x _ { \mathrm { s k i n } } } )$ 5: $I  [ 0 . 0 \quad \begin{array} { r r r } { 0 . 0 } & { 0 . 0 } \end{array} \quad 0 . 0 ] ^ { T }$   
⊳ Step 5: Luminance Filtering (Specularities & Shadows) 6: <sub>�</sub> <sub>←</sub> [<sub>0.0</sub> <sub>0.0</sub> $0 . { \overset { \cdot } { 0 } } \mathbf { \overset { \cdot } { \doublebarwedge } }$   
8: $L _ { \mathrm { s k i n } }  \mathbf { I } _ { \mathrm { l a b } } [ \mathcal { M } _ { \mathrm { r a w } } , L ]$   
9: $T _ { \mathrm { b r i g h t } }  \cdot$ ComputePercentile $\boldsymbol { L } _ { \mathrm { s k i n } } , 9 0 )$ ⊳ Step 1: Target View-Space 3D Position Reconstruction   
10: $T _ { \mathrm { d a r k } } \gets \mathrm { C }$ omputePercentile( $L _ { \mathrm { s k i n } } , 5 )$ 7: � ← Unproject(�, $D ( u ) , \hat { M } _ { \mathrm { p r o j } } ^ { - 1 } )$   
11: $\dot { \mathcal { M } } _ { \mathrm { f i n a l } }  \dot { \mathcal { M } } _ { \mathrm { r a w } } \land ( T _ { \mathrm { d a r k } } < \mathbf { I } _ { \mathrm { l a b } } [ : , : , L ] < T _ { \mathrm { b r i g h t } } )$   
⊳ Step 2: Generate noise pattern to avoid $\theta _ { i }$ periodicity   
12: return ${ \mathcal { M } } _ { \mathrm { f i n a l } }$ 8: jitter $ 2 . 0 \times \pi \times$ Noise(�)   
13: end function   
14: function INITIALIZEALBEDO(�) 9: for $i \gets 0$ to $N - 1$ do   
⊳ Step 3: Fibonacci & Burley Profile Importance Sampling   
⊳ Step 1: Adaptive Mask Generation 10: $\theta _ { i } \gets \mathrm { s a m p l e s } [ i ] . x + .$ jitter   
15:  ← EXTRACTADAPTIVESKINMASK(�)   
11: $r _ { i }$ ← samples[�].� × �   
⊳ Step 2: Active Pure-Skin Pixel Gathering 12: ofset $ [ r _ { i } \cos \theta _ { i } r _ { i } $ sin $\theta _ { i }$ $0 . 0 \big | ^ { T }$   
16: $\mathbf { P } _ { \mathrm { s k i n } } \mathbf { \bar { \Pi } }  \mathbf { I } [ \mathcal { M } ]$   
17: $\mathbf { c } _ { \mathrm { i n i t } }  [ 0 \quad 0 \quad 0 ] ^ { T }$ ⊳ Step 4: Neighbor Screen Projection   
13: $u _ { i } \gets \mathrm { P r o j e c t } ( x +$ ofset, $M _ { \mathrm { p r o j } } )$   
⊳ Step 3: Channel-wise Statistical Sigma Clipping   
18: for channel $c \in \{ R , G , B \}$ do ⊳ Step 5: Neighbor View-Space Position Reconstruction   
14: $D _ { i } \gets D ( u _ { i } )$   
20: 19: $\mathbf { v }  \mathbf { P } _ { \mathrm { s k i n } } [ : , c ]$ $\mu _ { c } , \sigma _ { c } \gets \mathrm { C }$ omputeMean(�), ComputeStdDev(�) 15: $x _ { i } \gets$ Unproject $u _ { i } , D _ { i } , M _ { \mathrm { p r o j } } ^ { - 1 } )$   
⊳ Step 6: Subsurface Difusion Weights $( E q . 3 )$   
⊳ Apply 2� Outlier Filter   
16: $s _ { i } \gets \operatorname* { m a x } ( \lVert x - x _ { i } \rVert _ { 2 } , \epsilon )$   
21: $\mathbf { v } _ { \mathrm { c l i p p e d } }  \{ x \in \mathbf { v } \mid | x - \mu _ { c } | < 2 \sigma _ { c } \}$   
22: 23: end for $\mathbf { c } _ { \mathrm { i n i t } } [ c ] \gets \cdot$ ComputeMean( $\mathbf { v } _ { \mathrm { c l i p p e d } } )$ 17: 18: $r R r \gets s _ { i } \times R ( s _ { i } , d )$ $p r \gets r _ { i } \times R ( r _ { i } , r _ { \operatorname* { m a x } } )$   
19: $k _ { i }  r R r / p r$   
24: return ${ \bf c } _ { \mathrm { i n i t } }$   
⊳ Step 7: Irradiance Gathering   
25: end function   
20: $L _ { i } \gets L ( u _ { i } ) . r g b$   
21: $I \gets I + ( k _ { i } \times L _ { i } )$   
22: $K \gets K + \dot { k _ { i } }$   
23: end for   
⊳ Step 8: Normalized weighted Integration $( E q .$ 4)   
24: $I  \bar { A } ( u ) . r g b \times ( I /$ max(�, �))   
25: return �   
26: end procedure

Algorithm 3 Burley Multiple Scattering Tensorized   
Require:   
�, � : Grid coordinate tensors. (�, �) in [−1, 1]   
${ \bf s } _ { T }$ : Homogeneous scattering distance tensor. $( H , W , 3 )$   
${ \bf D } _ { T }$ : Depth bufer tensor map. $( H , W )$   
${ \bf A } _ { T }$ : Surface albedo tensor map. $( H , W , 3 )$   
$\mathbf { L } _ { T }$ : Difuse irradiance tensor map. $( 1 , 3 , H , W )$   
� : Sample count ∈ [2, 128]   
������� : Fibonacci angles tensor. (�)   
������� : Stratified Burley Inverse CCDF radii tensor. (�) (Eq. 2)   
� : Camera intrinsic parameters $\{ z _ { \mathrm { n e a r } } , z _ { \mathrm { f a r } } , w _ { \mathrm { s e n s o r } } , f _ { \mathrm { f o c a l } } \}$   
Ensure:   
$\mathbf { I } _ { T }$ : Scattered difuse radiance tensor. $( H , W , 3 )$   
1: procedure EVALUATEMULTIPLESCATTERING   
⊳ Subsurface Scattering Setup   
2: $w _ { \mathrm { s c a l e } }  0 . 0 1 , \quad \epsilon  1 0 ^ { - 6 }$   
3: $d \gets S _ { T } \times w _ { \mathrm { s c a l e } }$ $\triangleright ( H , W , 3 )$   
4: $r _ { \operatorname* { m a x } } \gets d . \operatorname* { m a x } ( )$   
⊳ Step 1: Target View-Space 3D Position Reconstruction   
5: $z \gets \mathrm { L i n e a r i z e D e p t h } ( D _ { T } , \Phi _ { c } )$ $\triangleright ( H , W ) ( \mathrm { E q . } 5 )$   
6: $x _ { v } \gets \mathrm { U n p r o j e c t } ( U , z , \Phi _ { c } )$ $\triangleright ( H , W ) ( { \mathrm { E q . } }$ 6)   
7: $y _ { v } \gets \mathrm { U n p r o j e c t } ( V , z , \Phi _ { c } )$ ⊳ $( H , W )$ (Eq. 6)   
8: $\begin{array} { r } { P _ { v }  [ x _ { v } \quad y _ { v } \quad - z ] ^ { T } } \end{array}$ $\triangleright ( H , W , 3 )$   
⊳ Step 2: Generate noise pattern to avoid � periodicity   
9: jitter $ 2 . 0 \times \pi \times$ Random(1, $H , W )$ $\triangleright ( 1 , H , W )$   
⊳ Step 3: Fibonacci & Burley Profile Importance Sampling   
10: $\theta \gets s a m p l e s _ { \theta } . \mathrm { v i e w } ( N , 1 , 1 ) +$ + jitter ⊳ $( N , H , W )$   
11: $r \gets s a m p l e s _ { r } . \mathrm { v i e w } ( N , 1 , 1 ) \times r _ { \mathrm { m a x } }$ $\triangleright ( N , 1 , 1 )$   
12: $x _ { o } \gets \cos \theta \odot r$ ⊳ $( N , H , W )$   
13: $y _ { o } \gets \sin \theta \odot r$ ⊳ $( N , H , W )$   
⊳ Step 4: Neighbor Screen Projection   
14: $U _ { s } \gets \mathrm { P r o j e c t } ( x _ { v } + x _ { o } , z , \Phi _ { c } )$ ⊳ $\left( N , H , W \right) \left( \mathrm { E q . } 7 \right)$   
15: $V _ { s } \gets \mathrm { P r o j e c t } ( y _ { v } + y _ { o } , z , \Phi _ { c } )$ ⊳ (�, �, �) (Eq. 7)   
16: $G \gets \mathrm { S t a c k } ( U _ { s } , V _ { s } , \mathrm { d i m } = - 1 )$ ⊳ $( N , H , W , 2 )$   
⊳ Step 5: Neighbor View-Space Position Reconstruction   
17: $D _ { s } \gets \mathrm { G r i d S a m p l e } ( D _ { T } , G )$ $\triangleright ( N , H , W )$   
18: $z _ { s } \gets \mathrm { L i n e a r i z e D e p t h } ( D _ { s } , \Phi _ { c } )$ $\triangleright ( N , H , W ) ( \mathrm { E q . ~ } 5 )$   
19: $x _ { s } \gets \mathrm { U n p r o j e c t } ( U _ { s } , z _ { s } , \Phi _ { c } )$ $\triangleright \left( H , W \right) \left( \mathrm { E q . } 6 \right)$   
20: $y _ { s } \gets \mathrm { U n p r o j e c t } ( V _ { s } , z _ { s } , \Phi _ { c } )$ $\triangleright \left( H , W \right) \left( \mathrm { E q . } 6 \right)$   
21: $\begin{array} { r l } { P _ { s } \gets \left[ x _ { s } \quad y _ { s } \quad - z _ { s } \right] ^ { I } } \end{array}$ ⊳ $( N , H , W , 3 )$   
⊳ Step 6: Subsurface Difusion Weights $( E q . 3 )$   
22: $s \gets \operatorname* { m a x } ( \lVert P _ { v } - P _ { s } \rVert _ { 2 } , \epsilon )$ $> ( N , H , W , 1 )$   
23: $r R r \gets s \odot R ( s , d )$ ⊳ $( N , H , W , 3 )$   
24: $p r \gets r \odot R ( r , r _ { \operatorname* { m a x } } )$ $\triangleright ( N , 1 , 1 )$   
25: $k  r R r \oslash p r$ ⊳ $( N , H , W , 3 )$   
⊳ Step 7: Irradiance Gathering & Dimension Collapse   
26: $L _ { T } \gets L _ { T } . \mathrm { e x p a n d } ( N )$ ⊳ $( N , 3 , H , W )$   
27: $L _ { s } \gets \mathrm { G r i d S a m p l e } ( L _ { T } , G )$ ⊳ $( N , 3 , H , W )$   
28: $L _ { s } \gets L _ { s } . \mathrm { p e r m u t e } ( 0 , 2 , 3 , 1 )$ ⊳ $( N , H , W , 3 )$   
29: $\begin{array} { r } { I _ { T }  \sum _ { \mathrm { d i m } = 0 } ( k \odot L _ { s } ) } \end{array}$ $\triangleright ( H , W , 3 )$   
30: $\begin{array} { r } { K _ { T }  \sum _ { \mathrm { d i m } = 0 } k } \end{array}$ $\triangleright ( H , W , 3 )$   
⊳ Step 8: Normalized weighted Integration $( E q . 4 )$   
31: $I _ { T } \gets A _ { T } \odot \left( I _ { T } \oslash \operatorname* { m a x } ( K _ { T } , \epsilon ) \right)$ $\triangleright ( H , W , 3 )$   
32: return $I _ { T }$   
33: end procedure

1. Adaptive Skin Masking: The input is converted to CIE $L ^ { * } a ^ { * } b ^ { * }$ space. Chromatic channels $( a ^ { * } , b ^ { * } )$ are clustered via �-means $\left( k \ = \ 2 \right)$ after filtering lowluminance background pixels $( L ^ { * } \leq 2 0 )$ . The skin cluster is identified by the maximum $a ^ { * }$ centroid. Highfrequency specularity and deep shadow artifacts are eliminated by removing the top 10% and bottom 5% luminance percentiles within the raw mask.

2. Statistical Sigma Clipping: Active skin pixels are mapped back to linear RGB space. For each spectral channel, a 2� Sigma Clipping filter eliminates remaining outliers (e.g., edge pixels, macro-shadow boundary artifacts) from the distribution. The robust mean of the clipped pixels forms the final initialization vector ${ \bf { c } } _ { \mathrm { { i n i t } } }$

## 5.3. GPU Shader Implementation

Algorithm 2 details our screen-space multiple scattering approximation within the fragment shader, mapping directly to the mathematical formulations described in the main manuscript. All equation references within the pseudocode correspond to the main text.

## 5.4. Burley Multiple Scattering Diferentiable Tensorized Implementation

While forward rendering executes via the GPU fragment shader, inverse optimization requires an auto-diferentiable SSSS model to backpropagate loss gradients. We translate the screen-space difusion pass into vectorized PyTorch operations by broadcasting inputs across the � importance sample coordinates into a unified high-dimensional tensor. This architecture enables simultaneous 3D view-space reconstruction, projection, and diferentiable bilinear texture lookups, removing sequential per-pixel execution loops. The tensorized execution logic is formalized in Algorithm 3, where tuples within the inline comments denote multidimensional tensor shapes, and equation references correspond to the formulations established in the main text.