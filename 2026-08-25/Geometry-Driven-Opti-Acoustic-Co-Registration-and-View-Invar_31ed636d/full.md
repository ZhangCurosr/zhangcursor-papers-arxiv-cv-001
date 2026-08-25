# Geometry-Driven Opti-Acoustic Co-Registration and View-Invariant Reflectivity Mapping for Side-Scan Sonar

Taqi Hamoda, and Nuno Gracias, Member, IEEE

Abstract—Side-Scan Sonar (SSS) is a primary modality for large-scale underwater mapping, yet automated perception and cross-modal alignment are severely bottlenecked by acoustic complexities such as speckle noise, shadows, and extreme viewpoint dependencies. Traditional handcrafted descriptors and modern deep learning matchers fail to bridge the physical domain gap between optical and acoustic imagery without 3D geometric constraints. To overcome these limitations, we propose a novel geometry-driven framework for pixel-level optiacoustic co-registration and view-invariant reflectivity mapping. Our method utilizes Structure-from-Motion (SfM) to reconstruct a dense 3D seafloor mesh, acting as a geometric anchor between the visual and acoustic domains. We introduce a First Bottom Return (FBR) extraction algorithm to dynamically correct nonlinear altitude drift caused by uncalibrated SfM reconstruction. Furthermore, we apply an inverse Lambertian model and a dual-Gaussian weighting function to isolate the intrinsic seabed reflectivity, effectively neutralizing slant-range propagation loss and geometric view-dependence. By deterministically associating these isolated acoustic properties with optical pixels, our pipeline generates highly accurate, strictly co-registered multi-modal datasets. This automated, physics-guided approach eliminates the need for manual annotation and paves the way for advanced selfsupervised learning in benthic habitat mapping.

Index Terms—side-scan sonar, opti-acoustic matching, viewinvariant matching, benthic habitat mapping.

## I. INTRODUCTION

U <sup>NDERWATER</sup> <sup>robotic</sup> <sup>perception</sup> <sup>is</sup> <sup>severely</sup> <sup>constrained</sup> by the physical properties of the marine environment. Due to rapid absorption and scattering of electromagnetic radiation, high-resolution optical sensors are typically limited to operational ranges of under 10 meters [1], [2]. Conversely, acoustic energy propagates with minimal attenuation, allowing sound waves to travel hundreds of meters even in turbid or light-deprived waters. Consequently, Side-Scan Sonar (SSS) has emerged as the primary sensing modality for large-scale seafloor mapping, target detection, and autonomous navigation [3], [4].

Unlike optical cameras, SSS image formation is strictly governed by acoustic acquisition dynamics [3], [6]. As illustrated in Figure 2, laterally mounted transducers on a moving platform emit fan-shaped acoustic pulses perpendicular to the vehicle’s trajectory. The returning echoes are recorded over time and converted to range by assuming a constant sound speed of 1500 m/s, yielding a continuous 2D backscatter intensity map [3], [4]. Traditionally, this process is approximated via a Lambertian reflection model:

$$
I ( x , y ) = \rho ( x , y ) \cdot \cos ( \theta ( x , y ) ) \cdot L ( x , y ) ,\tag{1}
$$

![](images/e37655069c290def6beb967e4946809241da78a45bb44d62ad5e1e4c1afb9e25.jpg)  
Fig. 1: The Lambertian physics model for the SSS modality. Figure adapted from [5]

where the recorded intensity $I ( x , y )$ is a function of the intrinsic seabed reflectivity $\rho ( x , y )$ , the local incidence angle $\theta ( x , y )$ , and the acoustic propagation loss $L ( x , y )$ [3], [6].

However, this simplified theoretical model fails to capture the true complexities of acoustic propagation. Real SSS transducers emit non-uniform power profiles that diminish at the beam edges and fluctuate across side lobes [7]. Furthermore, topographic occlusions generate stark acoustic shadows, while multiplicative speckle noise inherently corrupts the returning signal [2], [8]. Consequently, SSS imagery is highly viewpoint-dependent; traversing the same seafloor structure from different survey directions yields drastically altered intensity patterns and shadow geometries.

## A. State of the Art in Feature Extraction and Matching

These severe physical artifacts pose a significant bottleneck for automated perception. Traditional handcrafted keypoint extractors (e.g., SIFT, SURF, AKAZE) rely heavily on local intensity gradients, which are easily disrupted by acoustic speckle and view-dependent backscatter, leading to poor repeatability [1], [10], [11].

To overcome this, recent approaches have transitioned to deep-learned detectors (e.g., SuperPoint) and Transformerbased matchers (e.g., LightGlue, LoFTR) [2], [11]. Despite their immense success in optical domains, recent acoustic benchmarks reveal severe performance degradation under wave distortion and stripe noise, exposing critical architectural trade-offs [2]:

• Sparse Methods (e.g., SuperPoint + LightGlue): Maintain relative accuracy under noise but yield incredibly sparse correspondences, failing to provide the dense alignment required for precise mapping.

![](images/bfe12ba8cafb0c6f5e969fedf7f013824ac70100b714d0cd08a6aeffe83df502.jpg)  
Fig. 2: Side-Scan Sonar deployment on a towfish connected to a survey vessel. Figure adapted from [9].

• Dense Methods (e.g., LoFTR): Achieve high spatial coverage but remain hyper-sensitive to low-contrast acoustic artifacts, resulting in prohibitive computational footprints and low inlier rates.

A foundational limitation is that these networks are typically deployed ”out-of-the-box” using pre-trained optical weights [2]. Without domain-specific fine-tuning, they fail to generalize to acoustic physics, relying instead on 2D homographic approximations that disregard 3D seafloor topography. This domain gap is exacerbated in cross-modal alignment (e.g., pairing SSS with optical imagery). To enforce structural consistency across differing sensor physics, symmetric epipolar distance is often introduced:

$$
d ( p _ { 1 } , p _ { 2 } ) = \frac { ( p _ { 2 } ^ { \top } F p _ { 1 } ) ^ { 2 } } { | F p _ { 1 } | ^ { 2 } } + \frac { ( p _ { 1 } ^ { \top } F ^ { \top } p _ { 2 } ) ^ { 2 } } { | F ^ { \top } p _ { 2 } | ^ { 2 } } ,\tag{2}
$$

where $\boldsymbol { F } \in \mathbb { R } ^ { 3 \times 3 }$ is the fundamental matrix [11]. However, optimizing this constraint requires features that are invariant to the underlying physics of both modalities—a requirement that current optical models simply cannot fulfill.

## B. Contributions

To address acoustic degradation, viewpoint dependency, and the severe domain gap in underwater multi-modal perception, this paper introduces a physics-informed, geometry-driven framework for SSS imagery. The primary contributions of this work are:

• Geometric Opti-Acoustic Co-Registration: We develop a novel cross-modal alignment framework that leverages reconstructed 3D optical geometry (SfM) as an anchor to achieve deterministic, pixel-level registration between visual and acoustic data.

• Dynamic Altitude Correction: We introduce a First Bottom Return (FBR) extraction algorithm utilizing a specialized cost function to dynamically correct nonlinear altitude drift and scale artifacts inherent in uncalibrated SfM reconstructions.

• Physics-Guided Reflectivity Isolation: We formulate an inverse Lambertian model paired with a dual-Gaussian weighting function to successfully disentangle intrinsic seabed reflectivity from slant-range propagation loss and geometric view-dependence.

• Automated Dataset Generation: We provide a scalable pipeline to generate perfectly co-registered opti-acoustic datasets, eliminating the need for manual acoustic annotation and enabling future self-supervised learning for benthic habitat mapping.

## II. METHODOLOGY

To establish a robust pixel-level correspondence between side-scan sonar (SSS) and optical imagery, we propose a geometry-driven pipeline [1]. This framework reconstructs a

shared 3D spatial representation to bridge the disparate sensor geometries, altitudes, and physical modalities.

## A. Geometric Reconstruction and Calibration

Following data acquisition via a dense, high-overlap lawnmower survey, we process the optical imagery using Vehicle Inertial Measurement Unit (IMU)-constrained Structure-from-Motion (SfM). This yields a precise 3D seafloor mesh that serves as the geometric ground truth. However, the lack of a calibrated sensor model introduces non-linear altitude offsets relative to the seabed. These artifacts stem from unconstrained optimization behaviors within the SfM solver:

• Dynamic Focal Length Estimation: Jointly optimizing the focal length induces an artificial zooming effect, causing spatial errors to propagate non-linearly away from the optical center.

• Unconstrained Radial Distortion: The solver minimizes reprojection errors locally but fails to model watercolumn-induced radial distortions, amplifying altitude offsets.

## B. Altitude Offset Correction via First Bottom Return

To resolve these non-linear altitude offsets, we utilize First Bottom Return (FBR) extraction from the SSS waterfall imagery to determine the true Autonomous Underwater Vehicle (AUV) altitude [3]. The raw sonar channels are smoothed using a median filter to reduce speckle noise, followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance edge contrast. A greedy algorithm tracks the FBR by minimizing the following cost function over a 100-bin sliding window:

$$
\mathcal { C } ( b ) = 0 . 4 \cdot \nabla ( b ) + 0 . 3 \cdot \mathcal { D } _ { \mathrm { n a d i r } } ( b ) + 0 . 3 \cdot \mathcal { D } _ { \mathrm { p a s t } } ( b )\tag{3}
$$

where ∇(b) is the gradient magnitude, ${ \mathcal { D } } _ { \mathrm { n a d i r } } ( b )$ is the spatial distance from the nadir, and ${ \mathcal { D } } _ { \mathrm { p a s t } } ( b )$ is the distance relative to preceding FBR locations. The FBR-derived ground-truth altitude dynamically corrects the reconstructed 3D geometry using an adaptive temporal window of the 50 closest pings, successfully compensating for non-linear drift.

## C. Physics-Guided Reflectivity Isolation

To extract the view-invariant material reflectivity, an inverse Lambertian reflection model is applied to the geometrically aligned SSS data and 3D mesh [6]. Mesh vertices with occluded lines-of-sight are classified as acoustic shadows via raytracing and discarded. Valid vertices are mapped to specific SSS bins, and raw intensities are divided by the cosine of the local incidence angle θ and normalized along-track to remove slant-dependent propagation loss. Finally, relative reflectivity values are projected back onto the 3D mesh using a dual-Gaussian weighting function ${ \boldsymbol { \mathcal W } } ( { \boldsymbol { \theta } } , r _ { s } )$

$$
\mathcal { W } ( \theta , r _ { s } ) = \exp \left( - \frac { ( \theta - \pi / 4 ) ^ { 2 } } { 2 ( \pi / 1 6 ) ^ { 2 } } \right) \cdot \exp \left( - \frac { r _ { s } ^ { 2 } } { 2 ( 1 2 ) ^ { 2 } } \right)\tag{4}
$$

This isolates the final view-invariant reflectivity $\rho _ { v }$ as a weighted average, neutralizing the sonar’s emission physics and angular sensitivity to yield highly accurate co-registered data.

![](images/6af2df004abdb77da04cd79bea771ed7e8e5f814f01fe8db9cb2c8722d4498d7.jpg)  
(a)

![](images/cddeeb8655f9baf08d3b083eeb0083eba15208930733f3752d628eb1165417c5.jpg)  
(b)

![](images/5f3c132e6956df6ca3ffb583b4bc2628b34328ba31192527152ca1480040ab3a.jpg)  
(c)

![](images/90949d025de7197ff5be72672b24eb5b5bfa9425bbcdf6d3ff52ca77f51e57cc.jpg)  
(d)  
Fig. 3: Results of the offset correction pipeline: (a) offset between SSS waterfall and geometry reprojection, (b) First Bottom Return (FBR) detection performance, (c) after global offset correction, and (d) after local offset correction. OpenCV’s Rainbow colormap is applied to the relative acoustic reflectivity.

## III. RESULTS AND DISCUSSION

Following the geometric and physics-based corrections detailed in our methodology, we evaluated the framework’s performance across diverse benthic environments, specifically rocky and seagrass regions.

## A. Altitude Correction Performance

Uncalibrated Structure-from-Motion (SfM) reconstructions inherently introduce non-linear altitude drift between the estimated vehicle pose and the seabed [1]. However, the application of our dynamic, local First Bottom Return (FBR) offset correction successfully compensated for these non-linear scaling artifacts. As demonstrated in Figure 3, the local temporal correction achieved precise structural alignment between the raw SSS waterfall imagery and the corresponding geometric reprojection, vastly outperforming global offset adjustments.

## B. Reflectivity Mapping Outcomes

By applying the inverse Lambertian reflection model, the pipeline successfully isolated the view-invariant material reflectivity $( \rho _ { v } )$ from transient acoustic artifacts [6]. Figure 4 illustrates the estimated seabed reflectivity maps, demonstrating the removal of geometric view-dependence and slantdependent propagation loss. The integration of shadow filtering and dual-Gaussian blending ensured that the final mapped intensities accurately represent intrinsic seabed properties. Furthermore, these isolated reflectivity values facilitated the accurate synthesis of SSS intensities for novel-view acoustic simulation (Figure 5).

![](images/bc5acf3a34841f326cf599d1bbdcf2fb175193375947582e666d3f27ca62a6b9.jpg)  
(a)

![](images/f8ffb79ab143bdb1a18de5df18f3f081e800785001682a6f0518b21d0d3d15e8.jpg)  
(b)  
Fig. 4: Estimated seabed reflectivity maps: (a) mountainous region mesh and (b) seagrass region mesh. Brighter areas indicate higher acoustic reflectivity.

## C. Opti-Acoustic Co-Registration Results

The spatial bridge established between localized acoustic bins and the 3D SfM point cloud achieved a highly accurate, deterministic pixel-level co-registration. Figure 6 showcases this successful cross-modal alignment across both the mountainous and seagrass test regions. By eliminating spatial ambiguity in multi-modal sensor fusion, this automated pipeline yields a high-fidelity proxy for ground truth. Ultimately, this generated dataset provides the foundational data necessary to train Self-Supervised Learning (SSL) architectures for benthic classification, entirely bypassing the need for manual acoustic annotation.

## IV. CONCLUSION AND FUTURE WORK

This paper presented a novel geometry-driven and physicsinformed framework to overcome the severe domain gap in opti-acoustic co-registration. While traditional handcrafted extractors and modern deep-learning matchers struggle with acoustic scattering, speckle noise, and extreme geometric distortions [2], [10], our approach successfully mitigates these challenges. By reconstructing a dense 3D SfM mesh as a structural anchor and dynamically correcting non-linear altitude drift via First Bottom Return (FBR) extraction, we established a deterministic pixel-level mapping between the visual and acoustic domains. Furthermore, the integration of an inverse Lambertian physics model successfully disentangled viewpoint-dependent artifacts from intrinsic seabed reflectivity, yielding highly accurate, view-invariant co-registration [6], [7].

## A. Future Work

Future work will focus on rigorous quantitative evaluation and framework refinement through three primary avenues:

![](images/3ff11a31227ab7f9483c103c5745aab2729ec0a22bd1a716a19a109ad21cca9f.jpg)  
Fig. 5: Acoustic simulation pipeline: (left) patch from the SSS waterfall, (middle) corresponding geometric reprojection, and (right) synthesized intensities based on the isolated reflectivity.

• Advanced Acoustic Modeling: Framework fidelity can be improved by replacing the simplified Lambertian approximation with advanced acoustic scattering models and bathymetric priors to better capture complex, heterogeneous seabeds [6], [7].

• Synthetic Benchmarking: Utilizing synthetic data generated from the isolated reflectivity values will provide a rigorous benchmark to quantitatively evaluate the framework’s novel-view acoustic simulation capabilities.

• Self-Supervised Learning Integration: We plan to leverage the automatically generated, perfectly coregistered datasets to train Self-Supervised Learning (SSL) architectures for automated benthic habitat mapping, fully bypassing the bottleneck of manual acoustic annotation.

## ACKNOWLEDGMENTS

This work was supported by Spanish Government through the project ”Automated Seabed Analysis through Self-Supervised Deep Learning Sonar Technology (ASSiST)” under grant PID2023-149413OB-I00.

## REFERENCES

[1] C. Lei, H. Rajani, N. Gracias, R. Garcia, and H. Wang, “A geometrically consistent matching framework for side-scan sonar mapping,” 2025. [Online]. Available: https://arxiv.org/abs/2509.11255

[2] O. Katrusha, D. Prylipko, and K. Yefremov, “Change detection in side-scan sonar imagery based on deep learning feature matching methods,” Eastern-European Journal of Enterprise Technologies, vol. 6, no. 2 (138), pp. 52–62, Dec. 2025. [Online]. Available: https://journals.uran.ua/eejet/article/view/346940

[3] A. Burguera and G. Oliver, “High-resolution underwater mapping using side-scan sonar,” PLOS ONE, vol. 11, no. 1, pp. 1–41, 01 2016. [Online]. Available: https://doi.org/10.1371/journal.pone.0146396

[4] H. Rajani, V. Franchi, B. M.-C. Valles, R. Ramos, R. Garcia, and N. Gracias, “Benthicat: An opti-acoustic dataset for advancing benthic classification and habitat mapping,” 2025. [Online]. Available: https://arxiv.org/abs/2510.04876

[5] P. Blondel and B. Murton, Handbook of seafloor sonar imagery. Scopus, 1997, cited by: 158. [Online]. Available: https://www.scopus.com/inward/record.uri?eid=2-s2.0-0031422050& partnerID=40&md5=ecb1a03ce5ac7ff380a2b6d941cc0fe0

![](images/21d2995028863dd16e77121b823ec4ba27eda731cd112b1dc68d2b5788b0373b.jpg)  
(a)

![](images/e659b280576c89941d85e69d740e58d87b02e4d85a23306d281a54df7c4d92fa.jpg)  
(b)

![](images/201c4ac4beb1f9b2ae740ac743c16ea13d956207c0832f8091f2caac7e8c1fbb.jpg)  
(c)

![](images/980213dd398c51e0d743210ea7d41deed7151efb5e5305e36b0da02e83987220.jpg)  
(d)

![](images/fe8321d1fe1921978e498701bb89ed143459bac3931e1031c4bc5b31a48c2e6d.jpg)  
(e)

![](images/c76dee84a4cdd074de51eec5cc578521a555b8fe43b3d267fc8a0fb4ed15bc79.jpg)  
(f)  
Fig. 6: Geometric opti-acoustic co-registration results across two test regions. Top row (rocky seabed): (a) 3D point cloud, (b) optical image, and (c) SSS waterfall. Bottom row (seagrass bed): (d) 3D point cloud, (e) optical image, and (f) SSS waterfall.

[6] C. Lei, H. Rajani, N. Gracias, R. Garcia, and H. Wang, “Physdnet: Physics-guided decomposition network of side-scan sonar imagery,” 2025. [Online]. Available: https://arxiv.org/abs/2511.19539

[7] E. Coiras, Y. Petillot, and D. M. Lane, “Multiresolution 3-d reconstruction from side-scan sonar images,” IEEE Transactions on Image Processing, vol. 16, no. 2, pp. 382–390, 2007. [Online]. Available: https://ieeexplore.ieee.org/document/4060928

[8] A. Preciado-Grijalva, B. Wehbe, M. B. Firvida, and M. Valdenegro-Toro, “Self-supervised learning for sonar image classification,” 2022. [Online]. Available: https://arxiv.org/abs/2204.09323

[9] D. Dondurur, “Chapter 1 - introduction,” in Acquisition and Processing of Marine Seismic Data. Elsevier, 2018, pp. 1–35. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ B9780128114902000013

[10] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, pp. 91–110, 2004. [Online]. Available: https://api.semanticscholar.org/CorpusID:174065

[11] Y. Fu, X. Luo, X. Qin, H. Wan, J. Cui, and Z. Huang, “Deep learning-based feature matching algorithm for multi-beam and side-scan images,” Remote Sensing, vol. 17, no. 4, 2025. [Online]. Available: https://www.mdpi.com/2072-4292/17/4/675