# From Sentinel-2 to 1-m Imagery: Real-World ×10 Cross-Sensor Super-Resolution via Visual Autoregressive Modeling

Tailai Chen<sup>†</sup>, Yuluxin Fu<sup>†</sup>, Jie Deng<sup>∗</sup>, Shu Luo, Yanan Wu, and Danfeng Hong, Senior Member, IEEE

Abstract—Reconstructing 1-m imagery from Sentinel-2 combines two difficulties. The input and the reference are observed by different satellites, which differ in spatial resolution, spectral response, radiometry, acquisition time, and local geometry, so no synthetic degradation operator links them. The required magnification of ten in each spatial dimension also far exceeds the factors of two to four at which remote sensing super-resolution is usually developed. We study this real-world ×10 problem using YREB-SR, which pairs 2022 Sentinel-2 RGB composites with 2022 Jilin-1 RGB references resampled to a 1-m grid over the Yangtze River Economic Belt. The benchmark contains 11,559 geographically matched scene pairs, with 6,409, 2,135, and 3,015 assigned to training, validation, and testing, respectively. We develop CrossVAR, a visual autoregressive model that predicts the 1-m latent as a nine-scale hierarchy conditioned on the Sentinel-2 input and estimates the remaining quantization residual with a diffusion refiner run without injected noise. Complete 1600×1600 scenes are reconstructed in overlapping windows that start on the Sentinel-2 pixel grid, as every training patch does, and every window is decoded at the size on which the decoder was trained before the windows are blended with Gaussian weights. On all 3,015 test scenes, CrossVAR achieves 27.60 dB PSNR, 0.7725 SSIM, 0.1451 LPIPS, and 7.15 FID, the best value of each metric among the compared methods. These results establish a strong autoregressive baseline for large-scale cross-sensor reconstruction. To facilitate reproducibility, the source code and implementation are publicly available at https://github.com/SHANOAL/RSSR.

Index Terms—Remote sensing, image super-resolution, crosssensor reconstruction, visual autoregressive modeling, Sentinel-2, Yangtze River Economic Belt.

## I. INTRODUCTION

ENTINEL-2 provides frequently, openly available obser-S vations for monitoring large geographic areas, but its

![](images/87e8a6787bd13fda26c6b7848f5284c0a0559b64e74a8c4d33f822cb78e83452.jpg)  
Fig. 1. PSNR and model time per complete 1600 × 1600 test scene of YREB-SR. Marker area grows with the number of parameters.

10-m visible bands do not resolve many roads, roofs, field boundaries, and small water features [1]. Very-high-resolution imagery can reveal these structures, although its coverage and acquisition conditions differ from those of medium-resolution missions. Super-resolution (SR) offers a way to estimate finer spatial structure from the available observations. Estimating 1-m imagery from Sentinel-2 is, however, considerably harder than the SR settings commonly studied in remote sensing, for two reasons.

First, the input and the reference are observed by different sensors. Most remote sensing SR methods learn from pairs in which the low-resolution (LR) image is synthesized from the high-resolution (HR) image by bicubic downsampling or a predefined blur kernel [2], [3]. The two images then share the sensor, the acquisition time, and the radiometric processing, and the degradation is known exactly. Models trained on such pairs lose accuracy on real acquisitions, because the simulated degradation departs from the real one [4]. Independent satellites differ in optics and modulation transfer function, spectral response, radiometric calibration and product generation, atmospheric state, and acquisition date. In YREB-SR, a single three-channel input is a Sentinel-2 composite from 2022, whereas its Jilin-1 reference was acquired in 2022. Land-cover change and seasonal appearance thus add to the sensor differences. Their footprints are geographically matched, but exact pixel-wise correspondence is not assumed. The LR image is therefore not a degraded copy of the reference, and no degradation operator links the two. A model must learn the cross-sensor relation from paired data without treating every sensor or temporal disagreement with the reference as missing detail. Real-world benchmarks pair independent acquisitions for this reason, including Sentinel-2 with WorldView-2 in MuS2 [5], with SPOT 6/7 in WorldStrat [6], and with NAIP aerial imagery in S2-NAIP [7], as well as Landsat 8, Sentinel-2, and PlanetScope in L8S2PS [8].

Second, the magnification is far larger than in common practice. Most remote sensing SR methods operate at factors of two to four, because the problem becomes rapidly more illposed at higher magnification [9], and real-world benchmarks likewise use moderate factors, such as three in L8S2PS [8], or provide several Sentinel-2 acquisitions per reference [5], [6]. Reconstructing 1-m imagery from a single 10-m Sentinel-2 image requires ×10, so each input pixel is expanded into 100 output pixels, compared with 16 at ×4. The nominal ×10 ratio describes the output grid; it does not imply that one observed LR pixel is the arithmetic mean of the corresponding 100 reference pixels. Most of the structure in a 1-m output, such as individual buildings, narrow roads, and field boundaries, lies below the input sampling and must come from learned image priors. Strong priors produce sharp detail that the observation does not necessarily support, and the cross-sensor differences make the reference an imperfect target for learning such priors. The large factor also enlarges every scene, as a 160 × 160 Sentinel-2 input becomes a 1600 × 1600 output that exceeds the crop size used in training, so complete scenes must be assembled from overlapping windows without visible seams. Reconstruction-oriented CNNs and transformers provide fidelity baselines for this setting [10]–[12], and adversarial and diffusion models offer perceptual priors [13], [14], so their comparison must use the same complete scenes and assess both reference fidelity and perceptual similarity.

Visual autoregressive modeling (VAR) predicts an image through a coarse-to-fine sequence of latent maps rather than a raster-ordered sequence of individual pixels [15], and the hierarchy has also been conditioned on LR images for SR [16]. The hierarchy suits a large magnification, since its coarse scales fix the scene layout that the Sentinel-2 input constrains, and its finer scales add local structure from the learned prior. CrossVAR encodes the Sentinel-2 input into one condition token per latent position, predicts the 1-m latent as nine code maps with a next-scale transformer, and estimates the residual that the codes leave with a residual refiner. Two steps of this inference add error unrelated to the reference. First, the residual refiner samples the residual with injected noise, which adds variation that no reference supports. Second, the assembly of complete scenes departs from the conditions of training. At ×4, a 16-pixel latent position spans exactly four input pixels, so every window on the latent grid also starts on the input grid. At ×10, input pixels span 10 output pixels and latent positions span 16, so the two grids coincide only every 80 output pixels, and overlapping windows mostly start inside an input pixel, whereas every training patch starts on the input grid. Fusing the windows and decoding the scene in one pass further applies the VQ decoder to a latent field sixteen times larger than those seen in training. CrossVAR removes the first error by deterministic residual refinement and the second by aligned window decoding, which starts every window on the input grid and decodes it at the training size. Both act at inference and leave the trained weights unchanged. Fig. 1 places CrossVAR among the compared methods by fidelity and cost on the complete test scenes.

We evaluate CrossVAR on the real paired benchmark described in Section III and summarized in Table I. The model processes the natural-color input alone; auxiliary spectral bands, temporal sequences, semantic labels, and geographic metadata are not provided to it. The contributions of this work are as follows.

• A real cross-sensor ×10 benchmark. YREB-SR contains 11,559 Sentinel-2/Jilin-1 scene pairs with a fixed scenelevel partition and documented sampling, geospatial standardization, and quality control. The test unit is the complete scene, not a selected crop.

• An autoregressive reconstruction of complete scenes. CrossVAR predicts the 1-m latent coarse to fine from the Sentinel-2 input, refines the quantization residual without injected noise, and reconstructs 1600 × 1600 scenes by aligned window decoding, which keeps every window on the input grid and at the training size of the decoder.

• A multi-family empirical comparison. We evaluate reconstruction, continuous-representation, adversarial, diffusion, and autoregressive baselines with common RGB metrics on 3,015 scenes. We report fidelity, perceptual quality, and efficiency for every method.

## II. RELATED WORK

## A. Super-Resolution Datasets

Most remote sensing SR datasets synthesize the LR image from the HR image by bicubic downsampling or a predefined blur kernel [2], [3]. Real-world datasets instead pair independent acquisitions of the same area. MuS2 pairs multiple Sentinel-2 revisits with WorldView-2 references [5], WorldStrat matches SPOT 6/7 imagery with multiple Sentinel-2 acquisitions [6], S2-NAIP pairs Sentinel-2 with NAIP aerial imagery [7], and OpenSR-test curates cross-sensor sets for evaluating Sentinel-2 SR [17]. RRSISR captures LR and HR images of the same scene with a spectral camera [4], and L8S2PS aligns Landsat 8, Sentinel-2, and PlanetScope images at a factor of three [8]. YREB-SR pairs a single Sentinel-2 composite with a 1-m Jilin-1 reference at ×10 and evaluates complete scenes.

## B. Super-Resolution

Reconstruction networks learn a direct mapping from the LR to the HR image, with residual convolutional blocks in EDSR [10], window-based transformer blocks in SwinIR [11], and top-k token selection for remote sensing in TTST [12]. Arbitrary-scale methods decouple the output grid from a fixed factor, as LIIF queries a continuous representation from local features [18] and Thera integrates a Gaussian point-spread function with neural heat fields [19]. Generative methods pursue perceptually plausible detail [13], [20] under the tradeoff between distortion and perceptual quality [21]. Satlas-SR applies adversarial training to Sentinel-2 [7], diffusion priors are explored by EDiffSR, RSDiffSR, SGDM, and TexADiff [14], [22]–[24], SEN2SR emphasizes radiometric and spatial consistency with the input [25], and WaveDiffUR addresses ultra-large magnification with a wavelet-domain diffusion solver [9].

## C. Visual Autoregressive Modeling

Vector-quantized autoencoders map images to discrete latent codes [26], [27], and VAR models these codes by next-scale prediction from coarse to fine [15]. VARSR conditions this hierarchy on LR prefix tokens and estimates the quantization residual with a diffusion refiner [16]. These models are devel oped on natural images at moderate factors. CrossVAR applies next-scale prediction to real Sentinel-2/Jilin-1 pairs at ×10, refines the residual without injected noise, and reconstructs complete scenes in windows aligned with the input grid.

## III. YREB-SR DATASET CONSTRUCTION

## A. Study Area and Sampling Design

The Yangtze River Economic Belt (YREB) extends from the mountainous upper reaches of the Yangtze River to the densely urbanized and intensively cultivated middle and lower reaches. The study area encompasses all administrative regions of 11 provincial-level units and relevant county-level divisions in eight additional provinces, comprising 1,173 county-level administrative units and more than 2.4 million km<sup>2</sup> of land [28], [29]. The pronounced variation in topography, climate, settlement density, vegetation, agricultural systems, and urban morphology provides a geographically diverse setting for evaluating real-world remote sensing super-resolution.

Candidate locations were selected using the built-up class of ESA WorldCover as a spatial guide [30]. Stratified random sampling was conducted across the upstream, midstream, and downstream portions of the YREB to capture regional differences in surface structure and urbanization intensity. A minimum center-to-center distance of 3 km was imposed between sampling locations. Because each scene covers approximately $1 . 6 \times 1 . 6 ~ \mathrm { k m ^ { 2 } }$ , this constraint prevents direct overlap between scene footprints and reduces immediate spatial redundancy. Although the sampling strategy emphasizes builtup environments containing high-frequency structures such as buildings, roads, and fragmented impervious surfaces, the resulting scenes also encompass vegetation, cropland, water bodies, bare land, and mixed rural landscapes. Figure 2 shows the geographic distribution of the resulting scene pairs and their dataset assignments.

## B. Cross-Sensor Image Sources

1) Sentinel-2 Low-Resolution Images: The low-resolution (LR) images were derived from Sentinel-2 observations accessed through Google Earth Engine [31]. Source acquisitions with reported cloud coverage exceeding 20% were excluded, after which median compositing was applied to the remaining 2022 observations to reduce residual cloud and atmospheric contamination. Sentinel-2 provides a native spatial resolution of 10 m in the blue (B2), green (B3), and red (B4) bands [1]. These bands were retained and ordered as B4–B3–B2 to form natural-color RGB inputs.

For each sampling location, a $1 6 0 \times 1 6 0 \times 3 ~ \mathrm { L R }$ scene was extracted at the native 10-m grid spacing, corresponding to a nominal ground footprint of $1 . 6 \times 1 . 6 ~ \mathrm { k m ^ { 2 } }$ . Each benchmark sample provides a single three-channel Sentinel-2 image as the model input. Other Sentinel-2 bands, multitemporal sequences, Sentinel-1 observations, semantic labels, and geographic metadata are not supplied to the reconstruction models. This design isolates the ability of each method to reconstruct fine spatial detail from natural-color medium-resolution imagery.

2) Jilin-1 High-Resolution Reference Images: The highresolution (HR) references were derived from 0.75-m RGB imagery acquired in 2022 by the Jilin-1 satellite constellation [32]. The source imagery was resampled to a uniform 1-m grid, from which 1600 × 1600 × 3 RGB scenes were extracted. Each HR scene therefore covers the same nominal $1 . 6 \times 1 . 6 ~ \mathrm { k m ^ { 2 } }$ footprint as its corresponding Sentinel-2 input.

The Jilin-1 images are regarded as cross-sensor references rather than strictly simultaneous ground truth. The LR and HR observations differ in acquisition date, sensor optics, modulation transfer function, spectral response, radiometric representation, atmospheric conditions, and product-generation chain. These differences are intentionally retained because YREB-SR is designed to evaluate practical cross-sensor reconstruction rather than inversion of an analytically specified synthetic degradation.

## C. Geospatial and Radiometric Standardization

For each sampling location, the Jilin-1 and Sentinel-2 images were extracted over the same geospatial extent using their associated georeferencing information. The Jilin-1 imagery was resampled by cubic interpolation to a 1600 × 1600 grid at a nominal 1-m ground sampling distance. The complete Sentinel-2 raster was independently reprojected and resampled over the corresponding extent to a 160 × 160 grid before extraction of bands B4, B3, and B2. Only locations for which both modalities were available and readable were retained, resulting in 11,559 complete RGB scene pairs.

All standardized pairs are stored in the WGS 84 geographic coordinate system (EPSG:4326). Within each pair, the LR and HR rasters have identical geographic bounds and resolutionconsistent affine transforms. Source projection information, spatial extent, affine transformation parameters, and NoData masks are propagated through the standardization procedure whenever available.

Sentinel-2 digital values are divided by 10,000 and stored as float32 surface reflectance, whereas the HR references retain their uint8 RGB representation. No histogram matching, color transfer, or sensor-to-sensor radiometric normalization is applied. The benchmark therefore preserves the actual cross-sensor radiometric discrepancy rather than imposing an artificially harmonized color relationship.

The quality-control procedure verifies file readability, paired filenames, band availability, raster dimensions, data types, geographic bounds, affine transforms, and NoData values. Dataset-level streaming statistics are additionally computed for each band, including the minimum, maximum, mean, standard deviation, and value histogram. Missing, corrupted, or unmatched files are recorded in an error log and excluded before dataset partitioning.

![](images/2974d2dbf16ab6e13fd60565f61267012f8201b066ecf29b0ae164d8cd6a8435.jpg)

![](images/5244fd92f093ab95f37d1142fe43e1f7e989d0c544bf3491a900c5fbcde2acf8.jpg)  
Fig. 2. Spatial distribution of the 11,559 cross-sensor scene pairs in YREB-SR across the Yangtze River Economic Belt. Colors indicate the training, validation, and test subsets, while blue lines represent the Yangtze River network.

Unlike conventional SR datasets, the LR images in YREB-SR are not generated from the HR references using bicubic downsampling, Gaussian filtering, additive noise, or a predefined degradation kernel. Instead, each pair consists of independent observations acquired by two satellite systems. The LR–HR relationship consequently includes sensorspecific optical responses, radiometric differences, temporal and seasonal changes, atmospheric variability, and residual local misregistration. It cannot be represented by a single deterministic degradation operator.

The pairs are geographically matched at the scene level, but exact pixel-wise correspondence is not assumed. A 10- m Sentinel-2 pixel nominally corresponds to a $1 0 ~ \times ~ 1 0$ region on the 1-m reference grid; however, its measured value is determined by the Sentinel-2 imaging process rather than by direct averaging of the corresponding 100 Jilin-1 pixels. This distinction separates YREB-SR from synthetically downsampled benchmarks and more closely reflects operational cross-sensor reconstruction.

## D. Dataset Partitioning and Patch Generation

Following quality control, the 11,559 scenes were randomly shuffled using a fixed seed and partitioned at the scene level into 6,409 training, 2,135 validation, and 3,015 test pairs. These subsets account for 55.45%, 18.47%, and 26.08% of the complete dataset, respectively. Scene identities are mutually exclusive across the three subsets. Furthermore, the 3-km minimum sampling distance ensures that their image footprints do not directly overlap, although the random partition is not intended to constitute a geographically blocked holdout experiment.

Each training and validation scene is divided into a nonoverlapping 4 × 4 grid. This produces aligned $4 0 \times 4 0 ~ \mathrm { L R }$ and $4 0 0 \times 4 0 0 ~ \mathrm { H R }$ patch pairs while preserving the nominal ×10 scale. The procedure yields 102,544 training and 34,160 validation patch pairs. During model training, random horizontal and vertical flips are applied identically to the LR and HR members of each pair to preserve correspondence.

The validation subset is used for model selection and hyperparameter configuration, whereas the fixed test subset is excluded from both parameter estimation and configuration selection. All competing methods use the same official partition. Test predictions are reconstructed and evaluated over all 3,015 complete $1 6 0 0 \times 1 6 0 0$ scenes, without border removal or selection of favorable test crops. The resulting protocol provides identical spatial support for interpolation-based, convolutional, transformer-based, continuous-representation, adversarial, and autoregressive methods.

Overall, YREB-SR differs from commonly used synthetic SR datasets in three principal respects. First, it evaluates an extreme ×10 reconstruction from native 10-m Sentinel-2 imagery to a 1-m VHR reference. Second, the LR images are independent cross-sensor observations rather than artificial degradations of the HR targets. Third, evaluation is conducted on a large and geographically heterogeneous collection of complete scenes under a fixed protocol. Together, these characteristics establish a reproducible and practically relevant benchmark for real-world cross-sensor remote sensing super-resolution.

TABLE I  
SUMMARY OF THE YREB-SR BENCHMARK
<table><tr><td>Attribute</td><td>Setting</td></tr><tr><td colspan="2">Cross-sensor image pairs</td></tr><tr><td>Sensor pair</td><td>Sentinel-2 RGB → 1-m RGB reference</td></tr><tr><td>Spectral channels</td><td> $\mathbf { B 4 - B 3 - B 2 }  \mathbf { R - G - B }$ </td></tr><tr><td>Nominal GSD</td><td> $1 0 \mathrm { ~ m ~ } {  } \mathrm { ~ 1 ~ m ~ } ( \times 1 0 )$ </td></tr><tr><td>Full-scene grids</td><td> $1 6 0 \times 1 6 0 \to 1 6 0 0 \times$  1600 pixels</td></tr><tr><td>Training grids</td><td> $4 0 \times 4 0  4 0 0 \times 4 0 0 ~ \mathrm { p i x e l s }$ </td></tr><tr><td>Storage types</td><td>float32 LR; uint8 HR</td></tr><tr><td>Spatial reference</td><td>EPSG:4326 with pairwise matched bounds</td></tr><tr><td colspan="2">Official partition and evaluation</td></tr><tr><td>Full scenes</td><td>6,409 training; 2,135 validation; 3,015 test</td></tr><tr><td>Aligned patches</td><td>102,544 training; 34,160 validation</td></tr><tr><td>Test protocol</td><td>3,015 complete scenes without border crop- ping</td></tr></table>

## IV. METHOD

CrossVAR reconstructs a 1-m image window by window, and Fig. 3 shows its stages. Section IV-A defines the task and the latent representation of a window. The condition encoder in Section IV-B maps the Sentinel-2 window to condition tokens, the next-scale transformer in Section IV-C predicts nine code maps of the latent from these tokens, and the residual refiner in Section IV-D estimates the residual that the codes leave. Section IV-E gives the training objective, and Section IV-F reconstructs complete scenes from windows.

## A. Preliminaries

Let $\mathbf { x } \in \mathbb { R } ^ { 3 \times H \times W }$ denote the standardized Sentinel-2 input and $\textbf { y } \in ~ [ 0 , 1 ] ^ { 3 \times 1 0 H \times 1 0 W }$ its Jilin-1 reference. We learn a reconstruction $\hat { \mathbf { y } } = F _ { \theta } ( \mathbf { x } )$ on the nominal 1-m grid. As specified in Section III, y is a geographically matched crosssensor reference rather than an exactly registered observation, so $F _ { \theta }$ is learned from the paired data instead of being derived from a degradation operator. CrossVAR operates on windows of $4 0 \times 4 0$ input pixels, the size of the training patches, and reconstructs each window at $4 0 0 \times 4 0 0$ pixels.

A VQ autoencoder represents a $4 0 0 \times 4 0 0$ window by a latent on a $2 5 \times 2 5$ grid. Its encoder E reduces each spatial dimension by a factor of 16, and its decoder G maps a latent back to the image. The reference latent ${ \bf z } ^ { * } = E ( { \bf y } )$ is represented by $K = 9$ code maps $\mathbf { r } _ { 1 : K }$ drawn from one codebook shared by all scales. The map r<sub>k</sub> has $s _ { k } \times s _ { k }$ positions with $( s _ { 1 } , \ldots , s _ { 9 } ) = ( 1 , 2 , 3 , 4 , 6 , 9 , 1 3 , 1 8 , 2 5 )$ , and the maps are obtained by residual quantization from coarse to fine,

$$
\mathbf { r } _ { k } = Q _ { k } \big ( \mathbf { z } ^ { * } - \mathbf { z } _ { < k } \big ) , \qquad \mathbf { z } _ { < k } = \sum _ { j < k } \phi _ { j } \big ( \mathbf { r } _ { j } \big ) ,\tag{1}
$$

where $Q _ { k }$ assigns the nearest code to each position of its argument downsampled to $s _ { k } \times s _ { k }$ , and $\phi _ { j }$ upsamples the code embeddings $\mathbf { e } ( \mathbf { r } _ { j } )$ to the $2 5 \times 2 5$ grid and applies a residual

convolution. The autoencoder stays frozen, and CrossVAR predicts the latent zˆ of a window, which the decoder maps to the 1-m window ${ \hat { \mathbf { y } } } = G ( { \hat { \mathbf { z } } } )$ .

## B. Condition Encoding

The Sentinel-2 window is first resized to the 400×400 output grid by bicubic interpolation, so that the condition and the reconstruction share pixel coordinates. The condition encoder is a convolutional network with four stride-2 stages, which reduce the resized window by a factor of 16 as the VQ encoder does. It outputs a condition token $\mathbf { c } _ { n } \in \mathbb { R } ^ { D }$ at every position n of the $2 5 \times 2 5$ latent grid, and $C ( \mathbf { x } )$ denotes these 625 tokens. Every latent position therefore has a condition token computed from the Sentinel-2 content around the same location. A learned start token $\mathbf { s } \in \mathbb { R } ^ { D }$ follows the condition tokens, and the condition tokens and the start token form the prefix of the transformer input.

## C. Next-Scale Prediction

The next-scale transformer predicts the code maps $\mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { K }$ in K steps. Its input sequence consists of the prefix followed by one token map for each scale from the second on. The token map of scale k is the latent accumulated from the coarser code maps, downsampled to $s _ { k } \times s _ { k }$ and projected to D channels,

$$
{ \bf t } _ { k } = { { W } _ { \mathrm { i n } } } \left. { { A } _ { s _ { k } } } { { \left( { { \bf { z } } _ { < k } } \right) } } + { { \ell } _ { k } } , \right.\tag{2}
$$

where $A _ { s }$ is area downsampling ${ \mathrm { t o ~ } } s \times s , W _ { \mathrm { i n } }$ is a linear projection, and $\ell _ { k }$ is a learned embedding of scale k added at every position. The prefix receives learned position embeddings. Each transformer block applies self-attention and a feedforward layer, and the layer normalization before each of them takes its scale and shift from the start token.

All tokens are placed on the 25 × 25 latent grid. Rotary position embedding rotates the queries and keys of every attention layer by the two-dimensional coordinates of a token, and position $( i , j )$ of an s × s map receives the coordinates $( 2 5 i / s , 2 5 j / s )$ . The condition tokens keep their grid coordinates, so relative positions in attention correspond to ground distances for the condition tokens and the tokens of all scales.

Attention follows the order of the scales. A token of scale k attends to the prefix and to all tokens of scales up to $k ,$ so every code map depends only on the input and on the coarser maps,

$$
p ( \mathbf { r } _ { 1 : K } \mid \mathbf { x } ) = \prod _ { k = 1 } ^ { K } p _ { \theta } { \big ( } \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } , C ( \mathbf { x } ) { \big ) } ,\tag{3}
$$

which is the factorization of next-scale prediction [15], [16]. The output at the start token gives the distribution of r , and the output at each position of $\mathbf { t } _ { k }$ gives the distribution of the code at the same position of $\mathbf { r } _ { k }$ , so all positions of a scale are predicted in parallel. An output head with a layer normalization and a linear layer maps the final transformer feature $\mathbf { h } _ { k , n }$ at position n of scale k to logits over the codebook. Training uses teacher forcing, in which the reference code maps form $\mathbf { z } _ { < k }$ in Eq. (2). At inference, the transformer selects the most probable code at every position, the accumulated latent $\hat { \mathbf { z } } _ { < k + 1 } = \hat { \mathbf { z } } _ { < k } + \phi _ { k } \big ( \hat { \mathbf { r } } _ { k } \big )$

takes the place of $\mathbf { z } _ { < k + 1 }$ as the input of the next scale, and cached keys and values let each step process only the tokens of its own scale.

## D. Deterministic Residual Refinement

The codes come from a finite codebook, so the accumulated latent $\hat { \mathbf { z } } _ { < K + 1 }$ cannot match a continuous latent exactly. After the last scale, the residual refiner estimates the remaining residual at every position n of the $2 5 \times 2 5$ map. Its target is

$$
\delta _ { n } = { \left( { { \mathbf { z } } ^ { * } } - { \mathbf { z } } _ { < K } \right) } _ { n } - \mathbf { e } { \left( \hat { \mathbf { r } } _ { K , n } \right) } ,\tag{4}
$$

the part of the latent at the last scale that the embedding of the code $\hat { \mathbf { r } } _ { K , n }$ selected by the transformer does not represent. The refiner is a small diffusion model applied to every position separately. Its denoiser is a stack of residual MLP blocks, each modulated by the diffusion step and by the condition

$$
\mathbf { c } _ { n } = \mathrm { L N } \big ( \mathbf { h } _ { K , n } + W _ { \mathrm { i n } } \mathbf { e } \big ( \hat { \mathbf { r } } _ { K , n } \big ) \big ) ,\tag{5}
$$

where $W _ { \mathrm { i n } }$ is the projection of Eq. (2) and LN denotes layer normalization, so the condition joins the transformer feature at n with the embedding of the code selected there. The refiner is trained with the diffusion loss ${ \mathcal { L } } _ { \mathrm { d i f f } }$ of improved denoising diffusion probabilistic models [33], and its estimate $\hat { \pmb \delta }$ completes the predicted latent $\hat { \mathbf { z } } = \hat { \mathbf { z } } _ { < K + 1 } + \delta$

At inference, the refiner estimates the residual of each position with a short reverse diffusion process, and we omit the index n below. Starting from ${ \bf u } _ { T } \sim \mathcal { N } ( { \bf 0 , I } )$ , each step computes

$$
\mathbf { u } _ { t - 1 } = { \pmb { \mu } } _ { \theta } ( \mathbf { u } _ { t } , t , \mathbf { c } ) + \tau \sigma _ { t } { \pmb { \epsilon } } _ { t } , \qquad { \pmb { \epsilon } } _ { t } \sim { \mathcal { N } } ( \mathbf { 0 } , { \bf I } ) ,\tag{6}
$$

where $\sigma _ { t }$ is the standard deviation of the reverse step, and standard sampling uses $\tau = 1$ . The injected noise makes the estimate $\hat { \pmb { \delta } } = \mathbf { u } _ { 0 }$ a random draw whose variation is unrelated to the reference. For a fixed condition, the expected squared error of such a draw decomposes as

$$
\mathbb { E } \big \lVert \boldsymbol { \delta } - \boldsymbol { \hat { \delta } } \big \rVert ^ { 2 } = \big \lVert \boldsymbol { \delta } - \mathbb { E } \boldsymbol { \hat { \delta } } \big \rVert ^ { 2 } + \mathbb { E } \big \lVert \boldsymbol { \hat { \delta } } - \mathbb { E } \boldsymbol { \hat { \delta } } \big \rVert ^ { 2 } ,\tag{7}
$$

so the sampling variance adds error without moving the estimate toward the reference. Deterministic residual refinement sets $\tau = 0$ in Eq. (6), so every reverse step follows the predicted mean. The initial state u remains random, but its effect on the output becomes negligible, as quantified in Section V-H. The refinement reuses the trained refiner and its number of steps, so it requires no training and no additional computation.

## E. Training Objective

The condition encoder, the next-scale transformer, and the residual refiner are trained jointly with the objective

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { d } \mathcal { L } _ { \mathrm { d i f f } } + \mathcal { L } _ { 1 } + \lambda _ { s } \mathcal { L } _ { \mathrm { S S I M } } , } \end{array}\tag{8}
$$

while the VQ autoencoder stays frozen. The code cross-entropy $\mathcal { L } _ { \mathrm { C E } }$ is averaged over the positions of each scale, and the scales are weighted in proportion to their side length $s _ { k }$ . The last code map holds about half of all 1,265 positions, so a plain average over positions would let it outweigh the coarser maps that set the tone and layout of the window. The L1 loss $\mathcal { L } _ { 1 }$ and the SSIM loss $\mathcal { L } _ { \mathrm { S S I M } }$ compare the reference window with the decoded image $G ( { \bf z } _ { < K } + \tilde { { \bf e } } _ { K } )$ , where $\mathbf { z } _ { < K }$ accumulates the reference code maps of the first $K - 1$ scales and $\tilde { \mathbf { e } } _ { K }$ is a straight-through estimate of the embeddings of the last code map. In the forward pass, $\tilde { \mathbf { e } } _ { K }$ takes the embedding of the most probable code at each position, and its gradient is that of the probability-weighted mean of the embeddings of the most probable codes. The pixel losses thus train the logits of the last scale through the frozen decoder.

![](images/439a0c09b26283fcdf9c9ff5786268813edfceece754eb39e69bda7c66ee25a5.jpg)  
Fig. 3. Overview of CrossVAR. Panels 1 to 3 reconstruct one window. The condition encoder maps the Sentinel-2 window, resized to the output grid, to one condition token per latent position, and the next-scale transformer predicts nine code maps from $\mathbf { \hat { l } } \times \mathbf { 1 }$ to 25 × 25. Deterministic residual refinement (DRR) estimates the residual of the codes without injected noise, and the frozen VQ decoder outputs the 1-m window. Aligned window decoding (AWD) in panel 4 starts every window of a complete scene on the input pixel grid, reconstructs it through panels 1 to 3, and blends the decoded windows with Gaussian weights.

## F. Aligned Window Decoding of Complete Scenes

Sections IV-B to IV-D reconstruct one window at the size of a training patch. A complete $1 6 0 \times 1 6 0$ input requires a 1600 × 1600 output and $\mathrm { ~ \ i ~ } 1 0 0 \times 1 0 0$ latent field, whereas the model is trained on a $2 5 \times 2 5$ latent grid. We therefore process the resized condition image in overlapping 400 × 400 windows. A training patch covers $4 0 \times 4 0$ input pixels, so its corner lies both on the input grid, with a spacing of 10 output pixels, and on the latent grid, with a spacing of 16, and the two grids coincide every 80 output pixels. A window that starts elsewhere places the input pixels at offsets from the latent positions that never occur in training. We therefore start every window at a multiple of 80 pixels, which places the windows of a complete scene at 0, 320, 640, 960, and 1200 pixels along each axis. An overlap of a quarter window would start them at 0, 304, 608, 912, and 1200 pixels, so three of every five window rows and columns would begin inside an input pixel.

Each window must also be decoded at its training size. The VQ decoder follows the VQGAN design [27], in which group normalization computes its statistics over all positions of a feature map and self-attention at the latent resolution weights all positions. Both operations were trained on $2 5 \times 2 5$ latents. Fusing the window latents and decoding the $1 0 0 \times 1 0 0$ field in one pass, as in latent tiling, therefore normalizes every window with statistics of the whole scene and lets each position attend to sixteen times as many positions as in training. We instead decode every window on its own and blend the decoded windows,

$$
\hat { \mathbf { y } } ( p ) = \frac { \sum _ { j : p \in \Omega _ { j } } \mathbf { w } _ { j } ( p ) G \big ( \hat { \mathbf { z } } _ { j } \big ) ( p ) } { \sum _ { j : p \in \Omega _ { j } } \mathbf { w } _ { j } ( p ) } ,\tag{9}
$$

where $p$ is an output pixel, $\Omega _ { j }$ is the support of window $j$ on the output grid, $\hat { \mathbf { z } } _ { j }$ is its refined latent, and ${ \bf w } _ { j }$ is a positive Gaussian weight map centered on the window. Every window is thus predicted, refined, and decoded as a training patch. Aligned window decoding changes only how complete scenes are tiled and decoded and requires no training.

## V. EXPERIMENTS

## A. Implementation and Evaluation

CrossVAR is trained on the 102,544 training patches with the objective of Eq. (8), where $\lambda _ { d } = 2$ and $\lambda _ { s } = 0 . 0 5$ , and all of its parameters are initialized from released natural-image weights [16]. All results of CrossVAR and its ablations use the same trained weights. Inference selects the most probable code at every position, uses $\tau = 0$ , and assembles complete scenes by aligned window decoding.

TABLE II  
TRAINING BUDGETS OF THE RETRAINED BASELINES
<table><tr><td>Method</td><td>Batch</td><td>Steps</td></tr><tr><td>EDSR-baseline</td><td>16</td><td>150k</td></tr><tr><td>SwinIR</td><td>8</td><td>150k</td></tr><tr><td>TTST</td><td>4</td><td>150k</td></tr><tr><td>Real-ESRGAN+</td><td>4</td><td>150k</td></tr></table>

PSNR on RGB and SSIM [34] on the luminance channel measure fidelity to the reference, LPIPS [35] measures perceptual distance, and FID [36] compares the distributions of predictions and references. All methods are evaluated with the same code on the 3,015 complete test scenes, without border cropping or output color correction.

## B. Comparison Methods

EDSR-baseline [10], SwinIR [11], and TTST [12] represent reconstruction networks, and Thera [19] represents a continuous image field. Real-ESRGAN+ [13] and Satlas-SR [7] represent adversarial models, EDiffSR [14] and TexADiff [24] represent diffusion models, and VARSR [16] is the released natural-image model. EDSR-baseline, SwinIR, TTST, Real-ESRGAN+, and EDiffSR are trained on the training partition with their outputs adapted to ×10, and Table II lists the budgets of the first four. TexADiff and VARSR are applied with their released weights. RSDiffSR and image-only SGDM are not compared because a released implementation or image-only checkpoint was not available.

## C. Quantitative Results

Table III reports the four metrics on the 3,015 test scenes. CrossVAR obtains 27.6008 dB PSNR, 0.7725 SSIM, 0.1451 LPIPS, and 7.1486 FID, the best value of every metric. The compared methods fall into two groups. The reconstruction networks and the continuous image field attain the highest fidelity among the baselines, led by Thera with 25.30 dB PSNR and 0.6855 SSIM, but their LPIPS of 0.57 to 0.68 and FID of 37.8 to 68.3 indicate smooth textures. The adversarial models reverse this balance. Satlas-SR lowers LPIPS to 0.2150 and FID to 16.49, while its PSNR falls 2.00 dB below that of Thera. CrossVAR does not follow this trade-off. It exceeds Thera by 2.30 dB PSNR and 0.087 SSIM and lowers the LPIPS and FID of Satlas-SR by 33% and 57%, respectively.

The diffusion models and the released VARSR stay close to bicubic interpolation. EDiffSR reaches 16.10 dB, TexADiff applies its released ×4 weights to the ×10 grid and reaches 14.88 dB, and VARSR transferred from natural images reaches 14.18 dB with an FID of 173.31. Priors learned from natural images or synthetic degradations therefore do not transfer to the cross-sensor ×10 task without training on real pairs.

Average scores can conceal scenes on which a method fails. Table IV therefore compares CrossVAR with every retrained baseline scene by scene. CrossVAR attains a higher PSNR, a higher SSIM, and a lower LPIPS than each of these methods on every one of the 3,015 test scenes. Against Thera, the PSNR gain has a median of 2.30 dB, a 5th percentile of 1.39 dB, and a minimum of 0.09 dB, and the 95% bootstrap interval of the mean gain spans 2.28 to 2.32 dB. Against Satlas-SR, the mean PSNR gain is 4.30 dB, and LPIPS is lower by 0.070.

## D. Spectral, Complexity, and Regional Analysis

Fig. 4 examines where the gain of CrossVAR arises. Panel (a) compares the radially averaged power spectrum of each reconstruction with that of the Jilin-1 reference. Below 0.05 cycles per meter, the Nyquist frequency of the Sentinel-2 input, the reconstruction networks and Thera stay within 1.8 dB of the reference and CrossVAR within 0.2 dB, since this band is observed in the input. Above it, the methods separate. Between 0.05 and 0.15 cycles per meter, the reconstruction networks and Thera retain 10.6 to 14.0 dB less power than the reference, Satlas-SR 1.4 dB less, and CrossVAR 1.8 dB less. The mean gradient magnitude of the reconstructions reaches 0.28 to 0.34 of that of the reference for the reconstruction networks and Thera, 0.77 for Satlas-SR, and 0.74 for CrossVAR. CrossVAR therefore keeps nearly as much fine structure as the adversarial model while its PSNR is 4.30 dB higher.

Panel (b) groups the test scenes into quartiles of the mean gradient magnitude of the reference, from the smoothest to the most textured quarter. PSNR falls with complexity for every method, and CrossVAR remains the most accurate in each quartile. Its gain over Thera decreases from 2.52 dB in the smoothest quartile to 2.00 dB in the most textured one, and its LPIPS rises from 0.116 to 0.173, compared with 0.493 to 0.632 for Thera and 0.179 to 0.250 for Satlas-SR.

Panel (c) maps the PSNR gain over Thera at the scene centers. The gain is positive across the whole basin. It averages 1.91 dB in the upper reaches west of 111<sup>◦</sup>E (859 scenes), 2.36 dB in the middle reaches between 111<sup>◦</sup>E and $1 1 6 ^ { \circ } \mathrm { E }$ (490 scenes), and 2.49 dB in the lower reaches (1666 scenes). The upper reaches also have the lowest PSNR for every method, 26.71 dB for CrossVAR against 27.99 dB in the lower reaches.

## E. Coarse-to-Fine Reconstruction

Fig. 5 follows the reconstruction across the nine scales. On the 427 validation scenes, the first six code maps set the overall tone and raise PSNR from 18.55 to 21.54 dB. The layout of buildings, roads, and water appears at $1 3 \times 1 3 .$ , and the 18 × 18 and $2 5 \times 2 5$ maps add the fine structure and raise PSNR to 25.63 dB. Deterministic residual refinement adds a further 1.78 dB and sharpens the textures in all four examples.

## F. Qualitative Results

Figure 6 compares four scenes with urban, water, transport, and rural content. In the urban scene, the reconstruction networks and Thera merge neighboring buildings into blurred patches, whereas CrossVAR separates individual buildings and keeps the narrow field roads in the enlarged regions continuous. In the water scene, the levees between ponds become faint in the outputs of the reconstruction networks, and CrossVAR reproduces them as thin continuous lines close to the reference.

TABLE III  
FULL-SCENE ×10 RECONSTRUCTION ON YREB-SR. ALL REPORTED SCORES USE THE SAME 3,015 TEST PAIRS. BEST AND SECOND-BEST RESULTS ARE BOLD AND UNDERLINED.
<table><tr><td>Method</td><td>Representation</td><td>PSNR (dB)↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td></tr><tr><td colspan="6">Reconstruction and continuous-representation methods</td></tr><tr><td>Bicubic</td><td>Interpolation</td><td>14.5051</td><td>0.4626</td><td>0.9534</td><td>215.9498</td></tr><tr><td>EDSR-baseline [10]</td><td>CNN</td><td>24.0986</td><td>0.6596</td><td>0.6831</td><td>68.2941</td></tr><tr><td>SwinIR [11]</td><td>Transformer</td><td>24.5078</td><td>0.6706</td><td>0.6371</td><td>58.0257</td></tr><tr><td>TTST [12]</td><td>RS transformer</td><td>24.7157</td><td>0.6739</td><td>0.6201</td><td>48.9027</td></tr><tr><td>Thera [19]</td><td>Neural field</td><td>25.2978</td><td>0.6855</td><td>0.5679</td><td>37.8334</td></tr><tr><td colspan="6">Perceptual and generative methods</td></tr><tr><td>Real-ESRGAN+ (paired) [13]</td><td>GAN</td><td>21.9581</td><td>0.6179</td><td>0.3269</td><td>47.2463</td></tr><tr><td>Satlas-SR [7]</td><td>RS GAN</td><td>23.2978</td><td>0.6328</td><td>0.2150</td><td>16.4924</td></tr><tr><td>EDiffSR [14]</td><td>RS diffusion</td><td>16.1011</td><td>0.4386</td><td>0.4529</td><td>99.6146</td></tr><tr><td>TexADiff [24]</td><td>Texture-aware diffusion</td><td>14.8779</td><td>0.4332</td><td>0.8325</td><td>175.9166</td></tr><tr><td colspan="6">Visual autoregressive methods</td></tr><tr><td>VARSR [16]</td><td>Natural-image VAR</td><td>14.1794</td><td>0.4381</td><td>0.6840</td><td>173.3056</td></tr><tr><td>CrossVAR (ours)</td><td>Cross-sensor VAR</td><td>27.6008</td><td>0.7725</td><td>0.1451</td><td>7.1486</td></tr></table>

(a)  
![](images/4a8c63a62dc401d957a7f29d08fac95e79a0bb0224b503e7614218ccb3faa48c.jpg)

$$
\mathsf { m } ^ { - 1 } )
$$

(b)  
![](images/c0cb6b5230b50909297e7538421b99b371b288c7063793e6aaa91be499017b02.jpg)

(c)  
![](images/6517c25ab99f1ddb0b2ab9e521189d1f34da811a8b098d340df229c064e72625.jpg)  
Fig. 4. Scene-wise analysis on the 3,015 test scenes. (a) Radially averaged power spectrum of the reconstructions relative to the Jilin-1 reference. The dotted line marks the Nyquist frequency of the Sentinel-2 input. (b) Mean PSNR in quartiles of the mean gradient magnitude of the reference. (c) PSNR gain of CrossVAR over Thera at the scene centers.  
TABLE IV

SCENE-WISE COMPARISON OF CROSSVAR WITH THE RETRAINED   
BASELINES ON THE 3,015 TEST SCENES. DIFFERENCES ARE CROSSVAR   
MINUS BASELINE, AND THE LAST COLUMN GIVES THE SHARE OF SCENES ON WHICH CROSSVAR IS BETTER IN ALL THREE METRICS.

<table><tr><td>Baseline</td><td>∆PSNR (dB)</td><td>∆SSIM</td><td>∆LPIPS</td><td>Better (%)</td></tr><tr><td>EDSR-baseline</td><td>+3.50</td><td>+0.113</td><td>-0.538</td><td>100</td></tr><tr><td>SwinIR</td><td>+3.09</td><td>+0.102</td><td>-0.492</td><td>100</td></tr><tr><td>TTST</td><td>+2.89</td><td>+0.099</td><td>-0.475</td><td>100</td></tr><tr><td>Thera</td><td>+2.30</td><td>+0.087</td><td>-0.423</td><td>100</td></tr><tr><td>Real-ESRGAN+</td><td>+5.64</td><td>+0.155</td><td>-0.182</td><td>100</td></tr><tr><td>Satlas-SR</td><td>+4.30</td><td>+0.140</td><td>-0.070</td><td>100</td></tr></table>

Across the four scenes, the reconstruction networks produce smooth textures and the adversarial and autoregressive models restore sharper local structure, consistent with the spectra in Fig. 4(a) and with their lower LPIPS.

## G. Computational Cost

Table V reports the cost of reconstructing one complete scene. CrossVAR requires 2.77 s of model execution per scene, less than the released VARSR and the iterative diffusion models but more than the one-pass reconstruction networks. Decoding each window on its own lowers the peak memory from 15,109 MiB to 9,219 MiB with 6% more model time.

## H. Ablation Study

Table VI isolates the two components on the 3,015 test scenes. All four settings use the same weights and evaluator. The base model samples the residual with τ = 1, tiles the scene with an overlap of a quarter window, and decodes the fused scene latent in one pass.

Aligned window decoding improves all four metrics at both temperatures and raises PSNR by 1.08 dB with τ = 1 and by 1.33 dB with τ = 0, on every test scene. Deterministic residual refinement raises PSNR by 0.34 dB and 0.60 dB and improves SSIM and LPIPS on every scene, while FID prefers the sampled residual once the windows are aligned [21]. Together, the two components raise PSNR from 25.93 to 27.60 dB and lower FID from 10.29 to 7.15.

Deterministic refinement also makes the output nearly independent of the random seed. On 64 validation patches, two runs with different seeds agree to 52.5 dB PSNR with $\tau = 0 ,$ against 31.5 dB with $\tau = 1$

TABLE V  
FULL-SCENE EFFICIENCY. TIMES ARE MILLISECONDS PER SCENE AND MEMORY IS PEAK MIB.
<table><tr><td>Method</td><td>Parameters (M)</td><td>Precision</td><td>Model time</td><td>I/O-inclusive</td><td>Peak memory</td></tr><tr><td>Bicubic</td><td></td><td>CPU</td><td>26.44</td><td>127.91</td><td></td></tr><tr><td>EDSR-baseline</td><td>1.39</td><td>FP32</td><td>2.92</td><td>384.07</td><td>124</td></tr><tr><td>SwinIR</td><td>11.99</td><td>FP32</td><td>106.41</td><td>440.12</td><td>499</td></tr><tr><td>TTST</td><td>21.76</td><td>FP32</td><td>189.63</td><td>503.57</td><td>1,632</td></tr><tr><td>Thera</td><td>5.85</td><td>FP32</td><td>303.50</td><td>717.46</td><td>4,534</td></tr><tr><td>Real-ESRGAN+</td><td>20.32</td><td>FP32</td><td>44.96</td><td>446.78</td><td>2,283</td></tr><tr><td>Satlas-SR</td><td>16.73</td><td>FP32</td><td>42.30</td><td>404.91</td><td>2,163</td></tr><tr><td>EDiffSR</td><td>20.40</td><td>FP32</td><td>88,213.74</td><td>88,696.91</td><td>4,974</td></tr><tr><td>TexADiff</td><td>3,525.17</td><td>FP16</td><td>9,766.84</td><td>11,621.83</td><td>14,588</td></tr><tr><td>VARSR</td><td>1,210.86†</td><td>FP16</td><td>3,273.50</td><td>3,370.45</td><td>15,140</td></tr><tr><td>CrossVAR (ours)</td><td>1,210.25†</td><td>FP16</td><td>2,765.14</td><td>2,849.16</td><td>9,219</td></tr></table>

Note: times are milliseconds per complete scene on one RTX 4090. <sup>†</sup>Includes the frozen VQ autoencoder.

(a)  
Sentinel-2  
13 × 13  
18 × 18  
25 × 25  
Refined  
(b)  
![](images/f5e1aab0b6728acbb670e1554d8dea5d65288610f017afd686f45cabf8ddbbc0.jpg)

![](images/64fcd24b28f987cc9c1c5544972237ed3bca11ed325a980f8715e16bc3d3235b.jpg)  
Fig. 5. Coarse-to-fine reconstruction. (a) Four test windows decoded after the 13 × 13, 18 × 18, and 25 × 25 code maps and after deterministic residual refinement, with PSNR against the Jilin-1 reference. (b) Mean PSNR after every scale on 427 validation scenes, with the interquartile range shaded.

TABLE VI  
ABLATION OF DETERMINISTIC RESIDUAL REFINEMENT (DRR) AND ALIGNED WINDOW DECODING (AWD) ON THE 3,015 TEST SCENES
<table><tr><td>DRR</td><td>AWD</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td></tr><tr><td rowspan="3">√</td><td></td><td>25.93</td><td>0.6960</td><td>0.1796</td><td>10.29</td></tr><tr><td></td><td>26.27</td><td>0.7076</td><td>0.1695</td><td>9.81</td></tr><tr><td>√</td><td>27.01</td><td>0.7509</td><td>0.1539</td><td>6.38</td></tr><tr><td>√</td><td>√</td><td>27.60</td><td>0.7725</td><td>0.1451</td><td>7.15</td></tr></table>

Table VII compares where the windows start, where they are fused, and at which size the decoder runs. Latent fusion blends the window latents before decoding, and RGB fusion blends windows decoded on their own as in Eq. (9). Starting the windows on the input grid gives most of the gain, 0.99 to 1.15 dB for every fusion rule. Decoding each window at the training size adds a further 0.31 dB, and RGB fusion of the decoded windows is best on all four metrics. This assembly was selected on 427 complete validation scenes, formed from the 16 patches of every fifth validation scene, where it reached 27.64 dB against 26.32 dB for latent fusion with one-pass decoding.

Fig. 7 tests the window start directly on the 427 validation scenes. Single windows were started 0.2 to 0.8 input pixels away from the grid in each axis and decoded on their own. PSNR falls steadily as the start moves away from the nearest input-pixel boundary, by 0.6 to 0.7 dB for a shift of 0.2 input pixels in one axis and by up to 2.2 dB when both axes start near the middle of an input pixel, and the aligned start is best in 87% of the scenes.

![](images/9b93e93fac498abf734878d1835232baa28ed6ba311ad21e0003a24837f2a414.jpg)  
Fig. 6. Complete-scene and local-detail comparisons on four test scenes. Seven learned methods are shown between the Sentinel-2 input and the Jilin-1 reference, and red and blue boxes mark the enlarged regions. Only the Sentinel-2 input is contrast-stretched for display.

TABLE VII  
COMPLETE-SCENE ASSEMBLY RULES WITH τ = 0 ON THE 3,015 TEST SCENES
<table><tr><td>Windows</td><td>Fusion</td><td>Decoding</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td></tr><tr><td>On grid</td><td>None</td><td>Scene</td><td>26.39</td><td>0.7406</td><td>0.1804</td><td>10.98</td></tr><tr><td>Off grid</td><td>Latent</td><td>Scene</td><td>26.27</td><td>0.7076</td><td>0.1695</td><td>9.81</td></tr><tr><td>Off grid</td><td>Latent</td><td>Window</td><td>26.51</td><td>0.7145</td><td>0.1595</td><td>8.94</td></tr><tr><td>Off grid</td><td>RGB</td><td>Window</td><td>26.45</td><td>0.7131</td><td>0.1558</td><td>7.96</td></tr><tr><td>On grid</td><td>Latent</td><td>Scene</td><td>27.26</td><td>0.7645</td><td>0.1609</td><td>9.08</td></tr><tr><td>On grid</td><td>Latent</td><td>Window</td><td>27.58</td><td>0.7708</td><td>0.1506</td><td>8.21</td></tr><tr><td>On grid</td><td>RGB</td><td>Window</td><td>27.60</td><td>0.7725</td><td>0.1451</td><td>7.15</td></tr></table>

## VI. DISCUSSION

## A. Fidelity and Perceptual Quality at a Large Magnification

Distortion and perceptual quality are known to compete in SR [21], and the baselines in Table III follow this pattern. $\mathbf { A t } \times 1 0 .$ the competition is governed by the frequencies that the Sentinel-2 input does not observe. A network trained with a pixel loss predicts the conditional mean of these frequencies, and because many fine structures are consistent with the same input pixels, this mean carries little power, as Fig. 4(a) shows. An adversarial model restores the power, but the restored structures need not lie where the reference places them, which lowers PSNR. CrossVAR selects the fine structure as discrete codes conditioned on coarser code maps, and the coarse maps fix the layout that the input constrains (Fig. 5). A likely explanation for its gain in both fidelity and perceptual quality is that the fine codes are placed relative to an already correct layout, so the reconstruction keeps both the power and much of the position of fine structure. The residual temperature offers an explicit control over the remaining balance. With aligned windows, τ = 1 gives the lowest FID and τ = 0 the best PSNR, SSIM, and LPIPS (Table VI), and τ = 0 is used throughout because the benchmark measures agreement with the reference.

## B. Window Alignment in Tiled Reconstruction

The misalignment addressed in Section IV-F arises whenever the number of output pixels per input pixel does not divide the latent stride. With a stride of 16, every window on the latent grid also starts on the input grid at $\times 2 , \times 4 , \times 8 ,$ and ×16, which is a probable reason why tiling at these common factors has not exposed the effect. At ×3 and ×6 the two grids coincide every 48 output pixels, and at ×10 every 80. Fig. 7 shows that a start 0.2 input pixels away from the grid already costs 0.6 to 0.7 dB. Latent generative SR models that tile complete scenes at such factors may therefore benefit from starting their windows on the common grid and from decoding each window at the size used in training.

![](images/bc8485ab72a5c04875b25a8572124828628173ec31dd07d30ff7443591741a2f.jpg)  
Fig. 7. PSNR change of single windows on 427 validation scenes when the window start is moved away from the Sentinel-2 pixel grid. The outlined cell is the aligned start.

## C. Limitations

The Sentinel-2 composites and the Jilin-1 references come from different sensors and different acquisition dates, and their footprints are matched geographically rather than registered pixel by pixel. Part of the measured error therefore reflects sensor and temporal differences that no reconstruction from the input can remove, and the absolute values of the referencebased metrics should be read with this in mind.

Structures smaller than a Sentinel-2 pixel, such as individual buildings and narrow field boundaries, are predicted from the learned prior. These structures are not observed in the input, so the reconstructions are suited to visual interpretation and to mapping at the level of the observed layout rather than to measuring individual objects below 10 m.

The benchmark covers the Yangtze River Economic Belt, and its random scene-level partition is not a geographically blocked holdout. Transfer to other climates and landscapes and the use of other spectral bands remain to be tested. CrossVAR also requires 2.77 s and 9.2 GB per complete scene with about 1.2 billion parameters, much more than the compact reconstruction networks. Finally, on validation patches, taking the code maps of the first eight scales from the reference raises PSNR from 27.36 to 29.23 dB, whereas decoding the unquantized reference latent reaches 32.45 dB, so most of the remaining error lies in the last scale and the continuous residual.

## VII. CONCLUSION

We studied real-world ×10 reconstruction from a single Sentinel-2 RGB composite using the 11,559 geographically matched pairs in YREB-SR. CrossVAR predicts the 1-m latent as nine code maps and a continuous residual, refines the residual without injected noise, and reconstructs complete scenes in windows that keep the input alignment and the decoder size of training. On the fixed 3,015-scene test split, CrossVAR obtains 27.60 dB PSNR, 0.7725 SSIM, 0.1451 LPIPS, and 7.15 FID. Scene-wise and spectral analyses show that the advantage over the retrained baselines holds across scene complexity and region and that CrossVAR keeps most of the power of fine structure above the Sentinel-2 Nyquist frequency while placing it far more accurately than adversarial models. Future work will estimate the continuous residual more closely from the Sentinel-2 input, add multi-temporal and multi-spectral observations, and reduce the inference cost.

## CREDIT AUTHORSHIP CONTRIBUTION STATEMENT

Tailai Chen and Yuluxin Fu: Writing – original draft, Software, Conceptualization, Formal analysis. Shu Luo: Data curation, Resources. Jie Deng, Yanan Wu and Danfeng Hong: Supervision, Conceptualization, Writing – review & editing, Resources, Funding acquisition.

## DECLARATION OF COMPETING INTEREST

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## FUNDING

This research was supported by the National Natural Science Foundation of China (Grant No. 32402311).

## ACKNOWLEDGMENT

The authors thank all collaborators and institutions involved in this research for their valuable contributions and assistance. All authors approved the final version of the manuscript.

## REFERENCES

[1] M. Drusch, U. Del Bello, S. Carlier et al., “Sentinel-2: ESA’s optical high-resolution mission for GMES operational services,” Remote Sensing of Environment, vol. 120, pp. 25–36, 2012. [Online]. Available: https://doi.org/10.1016/j.rse.2011.11.026

[2] P. Wang, B. Bayram, and E. Sertel, “A comprehensive review on deep learning based remote sensing image super-resolution methods,” Earth Science Reviews, vol. 232, p. 104110, 2022.

[3] Y. Qi, M. Lou, Y. Liu, L. Li, Z. Yang, and W. Nie, “Advancing image super-resolution techniques in remote sensing: A comprehensive survey,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 231, pp. 68–100, 2026.

[4] J. Wang, L. Xiang, L. Liu et al., “Toward real-world remote sensing image super-resolution: A new benchmark and an efficient model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–13, 2025.

[5] P. Kowaleczko, T. Tarasiewicz, M. Ziaja et al., “A real-world benchmark for Sentinel-2 multi-image super-resolution,” Scientific Data, vol. 10, p. 644, 2023.

[6] J. Cornebise, I. Orsoli ˇ c, and F. Kalaitzis, “Open high-resolution satellite´ imagery: The WorldStrat dataset – with application to super-resolution,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 25 979–25 991.

[7] P. Wolters, F. Bastani, and A. Kembhavi, “Zooming out on zooming in: Advancing super-resolution for remote sensing,” arXiv preprint arXiv:2311.18082, 2023. [Online]. Available: https://arxiv.org/abs/2311. 18082

[8] F. Meng, S. Wu, Y. Xiao et al., “Deep learning for remote sensing image superresolution: A comprehensive review and a new real-world benchmark,” IEEE Geoscience and Remote Sensing Magazine, vol. 14, no. 5, pp. 396–432, 2026.

[9] Y. Shi, L. Han, D. Dancy, and L. Han, “WaveDiffUR: A diffusion SDE-based solver for ultra magnification super-resolution in remote sensing images,” arXiv preprint arXiv:2412.18996, 2024. [Online]. Available: https://arxiv.org/abs/2412.18996

[10] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, “Enhanced deep residual networks for single image super-resolution,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017, pp. 136–144.

[11] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “SwinIR: Image restoration using swin transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2021, pp. 1833–1844.

[12] Y. Xiao, Q. Yuan, K. Jiang, J. He, C.-W. Lin, and L. Zhang, “TTST: A topk token selective transformer for remote sensing image super-resolution,” IEEE Transactions on Image Processing, vol. 33, pp. 738–752, 2024.

[13] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training realworld blind super-resolution with pure synthetic data,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2021, pp. 1905–1914.

[14] Y. Xiao, Q. Yuan, K. Jiang, J. He, X. Jin, and L. Zhang, “EDiffSR: An efficient diffusion probabilistic model for remote sensing image superresolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024.

[15] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via next-scale prediction,” in Advances in Neural Information Processing Systems, vol. 37. Curran Associates, Inc., 2024, pp. 84 839–84 865. [Online]. Available: https://doi.org/10.52202/079017-2694

[16] Y. Qu, K. Yuan, J. Hao et al., “Visual autoregressive modeling for image super-resolution,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267, 2025, pp. 50 926–50 948.

[17] C. Aybar, D. Montero, S. Donike, F. Kalaitzis, and L. Gomez-Chova, ´ “A comprehensive benchmark for optical remote sensing image superresolution,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[18] Y. Chen, S. Liu, and X. Wang, “Learning continuous image representation with local implicit image function,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 8628–8638.

[19] A. Becker, R. C. Daudt, D. Narnhofer et al., “Thera: Aliasing-free arbitrary-scale super-resolution with neural heat fields,” Transactions on Machine Learning Research, 2025. [Online]. Available: https: //openreview.net/forum?id=GU8YOfmqyg

[20] C. Ledig, L. Theis, F. Huszar et al., “Photo-realistic single image superresolution using a generative adversarial network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 4681–4690.

[21] Y. Blau and T. Michaeli, “The perception-distortion tradeoff,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6228–6237.

[22] C. Zhu, Y. Liu, S. Huang, and F. Wang, “Taming a diffusion model to revitalize remote sensing image super-resolution,” Remote Sensing, vol. 17, no. 8, p. 1348, 2025.

[23] C. Wang and W. Sun, “Semantic guided large scale factor remote sensing image super-resolution with generative diffusion prior,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 220, pp. 125–138, 2025.

[24] E. Zhang, S. Zhao, D. Muhtar, Z. Li, X. Zhang, and P. Xiao, “Remote sensing image super-resolution for imbalanced textures: A texture-aware diffusion framework,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 38 238–38 247.

[25] C. Aybar, J. Contreras, S. Donike, E. Portales-Juli´ a, G. Mateo-Garc\` ´ıa, and L. Gomez-Chova, “A radiometrically and spatially consistent super-´ resolution framework for Sentinel-2,” Remote Sensing of Environment, vol. 334, p. 115222, 2026.

[26] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[27] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12 873–12 883.

[28] State Council of the People’s Republic of China, “Reply on the territorial spatial plan for the yangtze river economic belt– yangtze river basin (2021–2035),” State Council of the People’s Republic of China, Beijing, China, Tech. Rep. State Letter [2024] No. 26, 2024, in Chinese. [Online]. Available: https: //www.gov.cn/zhengce/content/202402/content 6931103.htm

[29] J. Deng, D. Hong, C. Li, and N. Yokoya, “Joint super-resolution and segmentation for 1-m impervious surface area mapping in China’s Yangtze River Economic Belt,” GIScience & Remote Sensing, vol. 63, no. 1, p. 2610548, 2026. [Online]. Available: https://doi.org/10.1080/15481603.2025.2610548

[30] D. Zanaga, R. Van De Kerchove, D. Daems et al., “ESA WorldCover 10 m 2021 v200,” 2022, dataset. [Online]. Available: https://doi.org/10.5281/zenodo.7254221

[31] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, “Google Earth Engine: Planetary-scale geospatial analysis for everyone,” Remote Sensing of Environment, vol. 202, pp. 18–27, 2017. [Online]. Available: https://doi.org/10.1016/j.rse.2017.06.031

[32] Chang Guang Satellite Technology Co., Ltd., “Jilin-1 satellite data service platform,” Jilin-1 Satellite Mall, n.d., accessed: Aug. 27, 2026. [Online]. Available: https://www.jl1mall.com

[33] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139, 2021, pp. 8162–8171.

[34] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[35] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[36] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

![](images/e1aa52264a15e1d0eef5c4be57bc0f50e32649e9fbfeb8a443881dd6dc7b0518.jpg)

Tailai Chen is currently pursuing the Ph.D. degree with the Institute of Automation, Chinese Academy of Sciences, Beijing, China. His research interests include computer vision, open-world learning, mul timodal learning, and scientific machine learning.

![](images/f7edc542cf27cd2bee2b6ed697b45ead9ee0c77ee6077acbad13e14f32e4ddd6.jpg)

Danfeng Hong (Senior Member, IEEE) received the Dr.-Ing. degree (summa cum laude) in signal and information processing from the Signal Processing in Earth Observation (SiPEO), Technical University of Munich, Munich, Germany, in 2019. He is currently a Full Professor with Southeast University, Nanjing, China, and previously served as a Full Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China. His research interests include artificial intelligence, multimodal perception, foundation models, hyperspectral

![](images/ddfd6035c070fa735a6a2c4064cfc4f72c1b829a5f769f2673d487eb0f6578b8.jpg)

Yuluxin Fu received the B.S. degree in Artificial Intelligence from the College of Information and Electrical Engineering, China Agricultural University, Beijing, China. She is currently pursuing the Ph.D. degree with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China. Her research interests include remote sensing vision foundation models and fine-grained water body extraction.

remote sensing, and Earth science. Dr. Hong serves as an Associate Editor for IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE (TPAMI), IEEE TRANSACTIONS ON IMAGE PROCESS ING (TIP), and IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING (TGRS). He has been recognized as a Highly Cited Researcher by Clarivate Analytics since 2022.

![](images/0474dc8ba0b304b0c2f8dbe23fa705c05546e6a0e64241b1f05cec6205716dae.jpg)

Jie Deng received the Ph.D. degree in plant pathology from China Agricultural University, Beijing, China, in 2023. He was a Postdoctoral Researcher with the Aerospace Information Research Institute, Chinese Academy of Sciences, from 2023 to 2025. He is currently an Associate Professor with China Agricultural University. His research interests include agricultural remote sensing, crop disease monitoring, and artificial intelligence.

![](images/fbd70d5e5503e06e41a159b61adeb6799b4eb25f6ce99c3a53a442a0c384da01.jpg)

Shu Luo received the B.E. degree from Jiangsu University, Zhenjiang, China. He is currently pursuing the M.S. degree in Agricultural Engineering and Information Technology at Beijing Forestry University, Beijing, China. His research interests include forestry remote sensing and deep learning.

![](images/dd09d3e0b4519a601d1cbed29a3aaf96757576e9b335edba55cfc2e95c889956.jpg)

Yanan Wu received the Ph.D. degree from the School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China, in 2024. She con ducted postdoctoral research with the School of Com puter Science, Carleton University, Ottawa, Canada, from 2024 to 2025. She is currently a Lecturer with the College of Information and Electrical Engineering, China Agricultural University, Beijing, China. Her research interests include open-world multi-modal information processing, data-driven foundation model adaptation, and smart agriculture. She has published more than 20 peer-reviewed papers in prestigious international conferences and journals, including CVPR, ICCV, ICLR, AAAI, ECCV, IJCAI, ACM Multimedia, and IEEE Transactions on Circuits and Systems for Video Technology, etc. She was also a Visiting Scholar with the Department of Computer Science and Software Engineering, Concordia University, Montreal, Canada, in 2023.