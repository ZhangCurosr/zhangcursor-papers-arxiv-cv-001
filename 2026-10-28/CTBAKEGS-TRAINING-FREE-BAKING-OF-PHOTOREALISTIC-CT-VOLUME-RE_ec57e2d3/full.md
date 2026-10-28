# CTBAKEGS: TRAINING-FREE BAKING OF PHOTOREALISTIC CT VOLUME RENDERINGS INTO 3D GAUSSIAN SPLATS

A PREPRINT

Haruki Fukuda\*<sup>1</sup>, Constantin Lux<sup>2</sup>, Wolf Schweitzer<sup>2</sup>, Tadashi Hosoya<sup>1</sup>, and Lars Ebert<sup>3</sup>

<sup>1</sup>Department of Forensic Medicine, Gunma University Graduate School of Medicine, Maebashi, Japan <sup>2</sup>Institute of Forensic Medicine, Universität Zürich, Zürich, Switzerland <sup>3</sup>Biometrics/Forensic Imaging, Zurich Forensic Science Institute, Zurich, Switzerland

## ABSTRACT

Photorealistic volume rendering of computed tomography (CT) may convey anatomy intuitively, but it needs the CT data, dedicated software and a powerful graphics processor. Three-dimensional Gaussian splatting (3DGS) displays photorealistic scenes in real time on less powerful devices, but existing medical approaches train each scene for minutes to hours. We aimed to convert a CT volume rendering directly into a 3DGS model without training. Splats were placed on an adaptive octree of millimetre-sized cells. Their colour and opacity were computed on the graphics processor with the renderer’s lighting, cell opacity was converted analytically into splat opacity, and faint splats were retained stochastically above the 1/255 opacity floor of standard 3DGS rasterisers. Six public CT scans (1.5-mm voxels) were converted with three transfer functions and compared with converged references by peak signal-to-noise ratio (PSNR), structural similarity and a learned perceptual metric. Two-million-splat models reached a PSNR of 32.6 +/- 0.9 dB for bone, 30.1 +/- 1.0 dB for contrastenhanced vessels and 32.3 +/- 1.5 dB for skin and bone. A conversion pass (usually two or three per model) took 4-30 s on a current laptop and 9-60 s on a desktop with a 4-GB graphics card. On the laptop, the models rendered at 68 frames per second at 800 x 800 pixels. Training-free conversion produced real-time 3DGS models of photorealistic CT renderings in about a minute or less per conversion pass on ordinary computers, with fidelity that we consider sufficient for interactive viewing.

Keywords Computer graphics · Imaging (three-dimensional) · Software · Tomography (X-ray computed) · Image processing (computer-assisted)

## 1 Background

Photorealistic volume rendering of computed tomography (CT) simulates light scattering in tissue and has the capacity to depict anatomy with apparent lifelike depth and shading [1–4]. It speeds anatomical understanding compared with conventional CT images and serves surgical planning, teaching, patient communication and documentation [5–8]. However, it needs the CT data, dedicated software and a powerful graphics processing unit (GPU), so results are usually shared as still images or videos.

Three-dimensional Gaussian splatting (3DGS) represents a scene as millions of semi-transparent Gaussians rasterised in real time [9]. Standard 3DGS viewers display such models directly, so a 3DGS model of a CT rendering could be viewed, also in head-mounted displays, without the CT data or renderer. For medical and scientific volumes, most methods optimise Gaussians iteratively against tens to over a thousand rendered views per scene, for minutes to over an hour on a dedicated GPU [10–16]; one refines an analytic initialisation [17]. Feed-forward networks predict Gaussians within seconds but must first be trained on about 100–1,000 volumes with rendered views [18, 19]. Most optimised models must be rebuilt, or extended [15, 16], when the transfer function changes.

![](images/8f35b00779873339b034ff0295e7b98128b1788f1391de7993bcf8b9f293cb43.jpg)  
Fig. 1 Training-free conversion of a CT volume rendering into 3D Gaussian splats. The CT volume is shown with a chosen transfer function, which maps Hounsfield units to colour and opacity; the boxes marked Step 1–5 correspond to the steps in the text. A light volume, shared with the photorealistic renderer, stores the visibility of a sky dome in 24 directions and the transmittance towards a key light with soft shadows. The volume is divided into an adaptive octree of millimetre-sized cubic cells, split only where opacity changes, down to sub-voxel size; the finest size is adjusted to reach the requested number of splats. Each cell is sampled at 27 stratified points with the colour and lighting of the volume renderer, giving one colour with baked lighting and an opacity. Opacity mapping replaces each cell by a Gaussian with the same total opacity, fainter at the centre and wider than the cell (s = 0.55h; top); in a profile through a row of cells, the overlapping Gaussians add up to the opacity of the cells (bottom). Faint splats, whose opacity would fall below the 1/255 floor of standard rasterisers, are retained stochastically with raised opacity. The output is a standard 3DGS file (.ply or .spz); the example shows case 2 with the contrast-enhanced vessel transfer function. CT, computed tomography; 3DGS, three-dimensional Gaussian splatting

A volume, however, already defines what a Gaussian representation needs: where the material is, how opaque it is and, once lit, its colour. Gaussian kernels have long served direct volume rendering, notably in elliptical weighted average (EWA) splatting [20]. Instead of optimising or learning Gaussians from rendered images, we aimed to bake the appearance of a lit CT rendering directly into Gaussians, so that conversion is immediate on ordinary hardware. The contribution is not a new rendering model but the removal of the prerequisites of existing 3DGS pipelines: rendered training views, a dedicated GPU and per-scene training time. We describe CTBakeGS and measure how faithfully it reproduces the original rendering.

## 2 Methods

## 2.1 Conversion

Principle. A CT volume stores voxel-wise radiodensity values in Hounsfield units. A direct volume rendering is fully defined by three inputs: the CT data, a transfer function that maps Hounsfield units to colour and opacity [21], and the virtual lighting. A 3DGS model, in turn, is a set of Gaussian splats. Each splat is a small, “soft”, semi-transparent blob in space, described by a position, a size and orientation (scale and rotation; a sphere or a stretched ellipsoid), an opacity and a colour; its opacity is highest at the centre and fades smoothly with distance from it [9]. A splat differs from a voxel in three respects. First, a voxel is one sample of a regular grid, so its position and size are fixed by the grid, whereas a splat can be placed anywhere and have any size. Second, a voxel stores a measured quantity, in CT the X-ray attenuation in Hounsfield units, and says nothing about appearance; a splat stores the appearance itself, that is, how that piece of the scene looks. Third, a voxel volume becomes an image only through volume rendering (transfer function, lighting and ray casting) [21], whereas splats are simply projected onto the screen and blended in depth order [9, 20], which graphics processors do very quickly.

CTBakeGS derives splats directly from the volume: the volume is divided into small cells, each cell becomes one splat, and each splat receives the colour and opacity that the lit volume rendering gives to its cell.

This takes five steps, described in turn below (Fig. 1): (1) virtual lighting is precomputed, so that it can be stored in the splat colours; (2) the volume is divided into cells, which set the positions and sizes of the splats; (3) colour and opacity are sampled within each cell; (4) cell opacity is converted into splat opacity, as neighbouring splats overlap to generate a smooth realistic appearance; and (5) faint splats, which would vanish below the 1/255 opacity floor of standard rasterisers, are retained stochastically with raised opacity. An anti-aliasing compensation is applied when the splats are drawn. The result is written as a standard 3DGS file (.ply or .spz).

Step 1: lighting. In CTBakeGS, each splat has a single view-independent colour that does not change with the viewing direction, whereas 3DGS models trained from images usually store view-dependent colour. Lighting and shadows must therefore be computed before conversion and stored (baked) in that colour. CTBakeGS uses the lighting of its own photorealistic volume renderer, which approximates global illumination with a precomputed light volume of about 112 cells along the longest axis. For each light-volume cell, rays cast through the transfer-function opacity give the visibility of a sky dome in 24 directions and the transmittance towards a key light, which produces soft shadows. A sample’s colour is multiplied by the visibility-weighted sky light plus the transmittance-weighted key light and then tone-mapped. The renderer has a real-time view (one pass per frame), which supplies the colours for conversion, and a progressive view, which averages jittered frames until the image converges.

Step 2: cells. The cells determine where splats are placed and how large they are. Cells of one voxel each would spend many splats on homogeneous regions and would tie the splat count to the voxel count. Cells are therefore organised as an adaptive octree that is refined only where opacity changes. Starting from cells four voxels wide, a cell is split into eight when its opacity varies strongly, that is, when the maximum transfer-function opacity over its Hounsfield-unit range, $\alpha _ { \mathrm { m a x } } ,$ exceeds the smaller of the opacities at the two ends of that range by more than $\alpha _ { \mathrm { m a x } } / 2$ (and $\alpha _ { \mathrm { m a x } } > 0 . 0 1 )$ Homogeneous regions thus keep one large splat, whereas boundaries receive small splats, down to sub-voxel size. The split test extends 1.5 cell widths beyond the cell, so that large splats do not protrude through nearby surfaces. Cells are cubes measured in millimetres (unit: the finest voxel spacing), so that thick-slice data do not produce elongated cells.

The number of splats follows from the finest permitted cell size. To reach a chosen target count, this size is adjusted, assuming a power law between cell size and splat count, and the conversion is repeated until the count is within 8% of the target, usually after two or three conversion passes. If the target falls between two octree levels, as happened once (case 3, bone, one-million target), the count overshoots.

Step 3: colour and opacity. Each cell is now reduced to one colour and one opacity. The cell is sampled on the GPU at 27 stratified points $( 3 \times 3 \times 3 )$ . At each point, the transfer function gives the extinction, and the colour is shaded with the lighting model and light volume of the real-time view (Step 1). The point colours are averaged with weights for extinction and for the transmittance towards the most open axis direction, so that points lying deep inside the cell do not darken it. From the mean extinction σ¯ and the cell size h, the cell opacity is

$$
\alpha _ { \mathrm { c e l l } } = 1 - \exp ( - \bar { \sigma } h ) .\tag{1}
$$

Step 4: opacity mapping. Each cell is replaced by an isotropic Gaussian splat with standard deviation s. To avoid gaps and ripples between splats, s is chosen so that each splat extends beyond its cell. Neighbouring splats therefore overlap, and a ray crosses several of them; if every splat simply carried the cell opacity, the model would be too opaque. With the optical depth of the cell

$$
\tau _ { \mathrm { c e l l } } = - \ln ( 1 - \alpha _ { \mathrm { c e l l } } ) ,\tag{2}
$$

the optical depths of a lattice of isotropic Gaussians with spacing h and standard deviation s add up along a ray to that of the volume when each splat carries

$$
\tau _ { \mathrm { s p l a t } } = \tau _ { \mathrm { c e l l } } \frac { h ^ { 2 } } { 2 \pi s ^ { 2 } } .\tag{3}
$$

With $s = 0 . 5 5 h$ , the narrowest width for which Gaussians on a lattice of spacing h sum to a constant optical depth within 1% along each lattice axis, the splat opacity is

$$
\alpha _ { \mathrm { s p l a t } } = 1 - \exp ( - 0 . 5 2 6 \tau _ { \mathrm { c e l l } } ) .\tag{4}
$$

Step 5: faint splats. Standard 3DGS rasterisers skip splats whose opacity is below 1/255, and compact formats such as .spz store opacity with 8 bits, so a splat with $\alpha _ { \mathrm { s p l a t } }$ below this floor vanishes. In a thin translucent layer, every cell is below the floor, although many such cells along an oblique ray add up to a visible veil; discarding them removes the whole layer. CTBakeGS therefore keeps a faint splat $( \alpha _ { \mathrm { s p l a t } } < \alpha _ { 0 }$ , with $\alpha _ { 0 } = 0 . 0 1$ , about 2.5/255, so that it stays above the floor after the anti-aliasing compensation) with probability $\tau _ { \mathrm { s p l a t } } / \tau _ { 0 }$ , where $\tau _ { 0 } = - \ln ( 1 - \alpha _ { 0 } )$ , and assigns it the opacity $\alpha _ { 0 }$ . The expected optical depth of the lattice is unchanged, and fewer splats are needed. The choice uses a hash of the cell index, so the conversion remains deterministic.

Anti-aliasing compensation. To avoid aliasing, 3DGS rasterisers enlarge each projected Gaussian by a small screenspace filter (variance $0 . 3 \mathrm { p i x e l } ^ { 2 } )$ . This makes small splats appear thicker than intended [22]. As in Mip-Splatting [22], the splat opacity is therefore multiplied at drawing time by

$$
w = \sqrt { \frac { \operatorname * { d e t } ( \Sigma ) } { \operatorname * { d e t } ( \Sigma + 0 . 3 I ) } } ,\tag{5}
$$

where Σ is the projected covariance and I the identity matrix.

## 2.2 Implementation

CTBakeGS is a single HTML file (JavaScript, WebGL 2) that runs locally in a web browser. It is available at https://github.com/BocchiSlice/CTBakeGS; version 1.0 was used for all measurements.

## 2.3 Evaluation

Data. We used six CT scans from two scanner vendors in the TotalSegmentator dataset v2 (CC BY 4.0) [23]: case 1 (s0289), head CT angiography; case 2 (s0138), heart and thoracic aorta; case 3 (s1340), thorax to pelvis; case 4 (s0970), polytrauma; case 5 (s1121), pelvis-to-leg CT angiography; case 6 (s0931), neck-to-leg CT angiography (Supplementary Table S1). All have 1.5-mm isotropic voxels (215–519 slices; 4.9–56.7 million voxels). For the thick-slice test, the 1.5-mm slices of case 1 were also averaged along the slice direction in groups spanning 5 mm, giving voxels of 1.5 × 1.5 × 5 mm.

Views and reference. Each scan was rendered with three preset transfer functions (bone; contrast-enhanced vessels; skin and bone), giving 18 scenes plus three for the 5-mm data. For each scene, 48 wide views (16 azimuths at three elevations) and 16 close-up views were rendered at 800 × 800 pixels on a black background. Close-up views were centred on visible surface points chosen by farthest-point sampling, at 0.3 times the wide distance. The reference image for each view was rendered by the progressive view of the renderer (Step 1), from the same volume, transfer function and lighting, by averaging 128 jittered frames. The same view averaged over 256 frames differed from it by a peak signal-to-noise ratio (PSNR; see Metrics) of 54.7–58.7 dB, a root-mean-square (RMS) difference below half an 8-bit grey level and far smaller than the differences of the 3DGS models, so the reference was considered converged.

The 3DGS models were rasterised from the same cameras.

Metrics. Each 3DGS model was compared with the reference by PSNR, computed over the whole image and over the foreground (pixels non-black in either image), the structural similarity index (SSIM) [24] and the learned perceptual image patch similarity (LPIPS, VGG backbone) [25]. PSNR is 20 log of the full-scale value divided by the RMS pixel difference; 20 dB more means a tenfold smaller RMS difference, and 32 dB corresponds to about 2.5% of full scale, or 6 of 255 grey levels in an 8-bit image. We also recorded splat counts, conversion times and frame rates; frame rates were measured in the built-in WebGL rasteriser of CTBakeGS.

Variants. To test how splat count and the individual conversion steps affect fidelity, we evaluated models with about 1, 2 (main setting) and 3 million splats, and five ablations, each of which removes or replaces one step:

• no opacity mapping (Step 4; cell opacity used as is);

• no anti-aliasing compensation;

• without faint splats (Step 5; splats below the 8-bit floor discarded);

• a uniform grid with one cell per voxel instead of the adaptive octree (Step 2), compared with the octree at the same number of splats;

• voxel-unit instead of millimetre cells (Step 2; 5-mm data only).

Hardware. Measurements used Chrome 155 on a 14-inch Apple MacBook Pro (2024; M4 Max, 14-core CPU, 32-core GPU, 36 GB memory, macOS 26.6). Speed was also measured on a Windows desktop (Chrome 155; 2018; Intel Core i7-8700, 16 GB memory, NVIDIA GeForce GTX 1050 Ti, 4 GB, Windows 11) and conversion time on a 13-inch Apple MacBook Air (Chrome 137; 2017; 1.8-GHz dual-core Intel Core i5, Intel HD Graphics 6000, 8 GB memory, macOS 12.6).

Statistics. Metrics were averaged over the views of each scene, and the case was the unit of analysis (mean±standard deviation over six cases). Variants were compared with the main setting by the two-sided exact Wilcoxon signed-rank test; with six cases the smallest attainable p value is 0.031, and p values are exploratory and not adjusted for multiple comparisons. Analyses used Python 3.9 with NumPy 2.0, SciPy 1.13, pandas 2.3 and lpips 0.1.4 (PyTorch 2.8).

Table 1 Image quality of the 3DGS models against converged photorealistic references
<table><tr><td rowspan="2">Look, views</td><td colspan="3">PSNR (dB)</td><td rowspan="2">SSIM 2M</td><td rowspan="2">LPIPS 2M</td><td rowspan="2">PSNR, real-time photorealistic (dB)</td></tr><tr><td>1M</td><td>2M</td><td>3M</td></tr><tr><td>Bone, wide</td><td> $3 2 . 0 2 { \pm } 1 . 4 2$ </td><td> $3 2 . 6 1 { \pm } 0 . 8 9$ </td><td> $3 2 . 5 2 { \pm } 0 . 8 6$ </td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 0 2 8 { \pm } 0 . 0 1 4$ </td><td> $4 4 . 6 1 { \pm } 1 . 2 6 $ </td></tr><tr><td>Bone, close-up</td><td> $2 8 . 4 6 { \pm } 1 . 8 7$ </td><td> $2 9 . 6 3 { \pm } 1 . 5 5 $ </td><td> $3 0 . 2 1 { \pm } 1 . 4 2 $ </td><td> $0 . 9 4 9 { \pm } 0 . 0 1 2$ </td><td> $0 . 1 3 5 { \pm } 0 . 0 4 1$ </td><td> $4 5 . 8 9 { \pm } 1 . 6 4 $ </td></tr><tr><td>Vessels, wide</td><td> $2 9 . 1 7 { \pm } 1 . 1 5$ </td><td> $3 0 . 0 9 { \pm } 0 . 9 8 $ </td><td> $3 0 . 4 3 { \pm } 0 . 8 7$ </td><td> $0 . 9 6 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 0 3 6 { \pm } 0 . 0 1 6$ </td><td> $4 2 . 2 8 { \pm } 1 . 4 7$ </td></tr><tr><td>Vessels, close-up</td><td> $2 5 . 2 3 { \pm } 1 . 5 5$ </td><td> $2 6 . 5 6 { \pm } 1 . 6 2$ </td><td> $2 7 . 2 9 { \pm } 1 . 6 0 $ </td><td> $0 . 8 9 5 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 1 5 0 { \pm } 0 . 0 1 9$ </td><td> $4 3 . 8 9 { \pm } 1 . 5 1 $ </td></tr><tr><td>Skin and bone, wide</td><td> $3 1 . 7 8 { \pm } 1 . 8 0 $ </td><td> $3 2 . 3 2 { \pm } 1 . 5 1 $ </td><td> $3 2 . 1 9 { \pm } 1 . 1 8$ </td><td> $0 . 9 6 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 0 9 1 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $4 7 . 1 2 { \pm } 0 . 7 9$ </td></tr><tr><td>Skin and bone, close-up</td><td> $2 8 . 2 4 \pm 1 . 0 4$ </td><td> $2 9 . 0 9 { \pm } 1 . 0 1 $ </td><td> $2 9 . 4 3 { \pm } 0 . 9 4 $ </td><td> $0 . 9 2 5 { \pm } 0 . 0 1 3$ </td><td> $0 . 3 1 4 { \pm } 0 . 0 2 9$ </td><td> $4 6 . 7 0 { \pm } 1 . 1 9$ </td></tr></table>

Mean ± standard deviation over six cases; metrics were first averaged over the 48 wide or 16 close-up views of each scene. Model had approximately 1, 2 or 3 million (M) splats. Higher PSNR and SSIM and lower LPIPS indicate closer pixel-level agreement with the reference; they do not measure anatomical accuracy. LPIPS, learned perceptual image patch similarity; PSNR, peak signal-to-noise ratio; SSIM, structural similarity index; 3DGS, three-dimensional Gaussian splatting.

Table 2 Per-case image quality with two million splats, PSNR (dB) / SSIM / LPIPS
<table><tr><td>Look</td><td>Views</td><td>Case 1</td><td> $\mathbf { C a s e } 2$ </td><td> $\mathrm { C a s e } 3$ </td><td>Case 4</td><td>Case 5</td><td>Case 6</td></tr><tr><td>Bone</td><td>wide</td><td>33.6/0.982/0.031</td><td>32.0/0.971/0.054</td><td>33.1/0.980/0.023</td><td>31.1/0.979/0.022</td><td>33.1/0.980/0.018</td><td>32.6/0.985/0.017</td></tr><tr><td>Bone</td><td>close-up</td><td>30.0/0.961/0.178</td><td>31.2/0.948/0.192</td><td>30.0/0.945/0.124</td><td>27.1/0.935/0.118</td><td>30.9/0.965/0.085</td><td>28.5/0.941/0.114</td></tr><tr><td>Vessels</td><td>wide</td><td>30.2/0.966/0.042</td><td>30.4/0.942/0.065</td><td>30.3/0.955/0.034</td><td>28.3/0.962/0.030</td><td>31.3/0.969/0.019</td><td>30.1/0.969/0.024</td></tr><tr><td>Vessels</td><td>close-up</td><td>27.4/0.925/0.156</td><td>28.7/0.911/0.174</td><td>25.9/0.861/0.163</td><td>24.0/0.877/0.147</td><td>27.4/0.903/0.120</td><td>26.1/0.893/0.138</td></tr><tr><td>Skin and bone</td><td>wide</td><td>32.3/0.968/0.088</td><td>29.8/0.927/0.170</td><td>32.9/0.963/0.087</td><td>32.0/0.968/0.070</td><td>34.5/0.976/0.064</td><td>32.5/0.967/0.069</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>29.4/0.942/0.306</td><td>28.7/0.935/0.356</td><td>29.9/0.922/0.322</td><td>28.2/0.904/0.314</td><td>30.5/0.926/0.267</td><td>27.9/0.921/0.316</td></tr></table>

Metrics were averaged over the 48 wide or 16 close-up views of each scene; case numbers as in Supplementary Table S1. LPIPS, learned perceptual image patch similarity; PSNR, peak signal-to-noise ratio; SSIM, structural similarity index.

## 3 Results

## 3.1 Fidelity

Splat counts were within 8% of the targets in every scene except one (case 3, bone, 1.37 million for the one-million target). With two million splats, wide-view PSNR was 32.6±0.9 dB for bone, 30.1±1.0 dB for contrast-enhanced vessels and 32.3±1.5 dB for skin and bone, with SSIM of 0.960–0.980 and LPIPS of 0.028–0.091 (Table 1; Fig. 2). Close-up views were about 3 dB lower; the real-time photorealistic view reached 42–47 dB. Errors lay mainly along the edges of dense structures for bone and vessels and were spread over the translucent layer for skin and bone (Fig. 2). Fidelity was lowest in close-up views of case 4, the largest volume, for bone and vessels (27.1 and 24.0 dB), and in wide views of case 2 for skin and bone (29.8 dB; other cases 32.0–34.5 dB; Table 2).

## 3.2 Splat count and speed

Compared with two million splats, one million lowered close-up PSNR in all cases for bone (median −1.2 dB), vessels (−1.3 dB) and skin and bone (−0.9 dB), and three million raised it in all cases (+0.6, +0.8 and +0.4 dB); close-up LPIPS followed the same pattern (Supplementary Table S2). A two-million-splat pass took 14.3 s on average (4.2–29.5 s), and rendering ran at 68 frames per second (800 × 800 pixels; Table 3). A desktop with a 4-GB graphics card needed 29.4 s per pass and rendered 28 frames per second. The 2017 laptop needed 106 s (30–216 s) and rendered 7 frames per second.

## 3.3 Ablations

Without opacity mapping, PSNR was lower in all cases for every look (median −4.5 and −5.1 dB for bone in wide and close-up views, −1.3 and −1.9 dB for vessels, −5.3 and −5.9 dB for skin and bone; Supplementary Table S2). Without anti-aliasing compensation, wide-view PSNR was lower in all cases for bone (−3.8 dB) and vessels (−0.9 dB), with little change for skin and bone and in close-up views. At equal splat count, a uniform grid with one cell per voxel was as faithful as the octree for vessels and skin and slightly better for bone (+0.4 and +0.6 dB, all cases). The voxel grid, however, fixes the splat count (0.14–2.86 million); in the two smallest volumes, two-million-splat octree models exceeded it by 1.3–2.7 dB for bone and vessels. With 5-mm slices, millimetre cells gave PSNR similar to the 1.5-mm data (bone, wide views: 34.3 vs 33.6 dB), whereas voxel-unit cells gave 22.5 dB (Supplementary Table S3; per-case details in Supplementary Table S1). Without faint splats, PSNR was lower in all cases in wide views (bone −0.3 dB, vessels −0.4 dB, skin and bone −1.6 dB) and in close-up views of vessels and skin and bone (−0.5 and −1.4 dB), with little change in LPIPS; the loss was largest in case 2, where wide-view PSNR for skin and bone fell from 29.8 to 23.8 dB (Table 4; Fig. 3).

a Bone  
b Contrast-enhanced vessels c Skin and bone  
![](images/ec722e540536dbf55024ffb00969f4985a83d2865dfc37e31a7a3a535198d2a8.jpg)  
Fig. 2 Representative results: converged references, 3DGS models and their differences. (a) Bone, case 1; (b) contrast-enhanced vessels, case 6; (c) skin and bone, case 3; one wide view each, cropped to the body. Top: converged photorealistic volume rendering (reference); middle: 3DGS model with two million splats; bottom: difference from the reference (mean absolute difference over the colour channels, 0 to ≥ 50 grey levels of 255). Insets (yellow frames) show a close-up view of the same scene; dashed boxes mark the region it covers in the wide view. Errors lie mainly along the edges of dense structures for bone and vessels and are spread over the translucent layer for skin and bone. 3DGS, three-dimensional Gaussian splatting

Table 3 Splat counts, conversion times and frame rates
<table><tr><td>Model</td><td>1M</td><td>2M</td><td>3M</td></tr><tr><td>Splats (million)</td><td>1.04 (0.92–1.37)</td><td>2.00 (1.89–2.15)</td><td>2.97 (2.76–3.23)</td></tr><tr><td>Conversion time, MacBook Pro (s)</td><td>11.0 (2.2–24.0)</td><td>14.3 (4.2–29.5)</td><td>17.1 (3.1–30.7)</td></tr><tr><td>Frame rate, MacBook Pro (fps)</td><td>122 (97–137)</td><td>68 (62–74)</td><td>48 (43–53)</td></tr><tr><td>Conversion time, Windows desktop (s)</td><td>23.1 (4.6–50.0)</td><td>29.4 (8.8–60.2)</td><td>35.1 (7.4–63.0)</td></tr><tr><td>Frame rate, Windows desktop (fps)</td><td>52 (36–60)</td><td>28 (22–33)</td><td>19 (16–22)</td></tr><tr><td>Conversion time, MacBook Air 2017 (s)</td><td>85 (22–193)</td><td>106 (30–216)</td><td>a</td></tr></table>

Mean (range) over six cases × three looks. Conversion time is one conversion pass at the final cell size; reaching the target splat count usually took two or three passes; frame rates at 800 × 800 pixels in Chrome. <sup>a</sup>Not reported: three-million-splat conversions completed in only 10 of 18 scenes on this computer, the others exceeding its graphics memory. The .ply files hold 68 bytes per splat (128–146 MB for the 2-M models).

Table 4 Per-case PSNR (dB) with two million splats, with / without faint splats
<table><tr><td>Look</td><td>Views</td><td>Case 1</td><td>Case 2</td><td>Case 3</td><td>Case 4</td><td>Case 5</td><td>Case 6</td></tr><tr><td>Bone</td><td>wide</td><td>33.6 / 33.4</td><td>32.0 / 31.6</td><td>33.1 / 32.5</td><td>31.2 / 30.8</td><td>33.1 / 32.9</td><td>32.6 / 32.6</td></tr><tr><td>Bone</td><td>close-up</td><td>30.0 / 29.7</td><td>31.2 / 30.7</td><td>30.0 / 29.2</td><td>27.1 / 26.7</td><td>30.9 / 30.0</td><td>28.5 / 28.5</td></tr><tr><td>Vessels</td><td>wide</td><td>30.2 / 30.1</td><td>30.4 / 29.7</td><td>30.3 / 29.9</td><td>28.3 / 27.9</td><td>31.3 / 30.9</td><td>30.1 / 29.7</td></tr><tr><td>Vessels</td><td>close-up</td><td>27.4 / 27.3</td><td>28.7 / 27.7</td><td>25.9 / 25.4</td><td>24.0 / 23.5</td><td>27.4 / 26.6</td><td>26.1 / 25.6</td></tr><tr><td>Skin and bone</td><td>wide</td><td>32.3 / 29.8</td><td>29.8 / 23.8</td><td>32.9 / 31.4</td><td>32.0 / 31.2</td><td>34.5 / 33.7</td><td>32.5 / 30.7</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>29.4 / 27.4</td><td>28.7 / 23.7</td><td>29.9 / 28.6</td><td>28.2 / 27.2</td><td>30.5 / 29.8</td><td>27.9 / 26.5</td></tr></table>

PSNR averaged over the 48 wide or 16 close-up views of each scene; case numbers as in Supplementary Table S1. With: the full method (Step 5); without: splats below the 8-bit opacity floor discarded (ablation). PSNR, peak signal-to-noise ratio.

## 4 Discussion

CTBakeGS converted CT volume renderings into standard 3DGS models in about a minute or less per conversion pass, usually two or three passes per model, on a current laptop and on a 2018 desktop with a 4-GB graphics card, without training or per-scene optimisation. The idea is simple, but three steps made it work: the opacity mapping for overlapping splats, the baked lighting and the retention of faint splats above the 8-bit opacity floor. It is deterministic as every splat derives from its cell’s voxels; a new transfer function needs only another conversion; and, with the virtual lighting baked in, standard viewers show the lit result. Because the viewer runs in a web browser, no software has to be installed, and a converted model could be packaged with the viewer page and sent, for example, to patients as their CT scan in a form that runs on ordinary devices. Because splats are a common format, a volume rendering could also be combined with splat models of other origin, such as reconstructions of a crime scene.

The fidelity results may define the scope of use. Bone reached about 33 dB in wide and 30 dB in close-up views, corresponding to an RMS pixel difference of about 2.3% and 3.3% of full scale, or 6 and 8 of 255 grey levels, and skin and bone about 29 dB in close-up views (3.5%; 9 of 255), whereas the real-time volume renderer reached 42–47 dB (0.45–0.8%; 1–2 of 255). Contrast-enhanced vessels were weakest in close-up views (27 dB; 4.7%; 12 of 255), where the edges of dense bone and thin vessels were smoothed. In case 4, the largest volume, the same splat budget gave coarser cells, and close-up fidelity rose with more splats (vessels 22.8, 24.0 and 24.8 dB with 1, 2 and 3 million splats), so large volumes may need more splats.

Values reported for trained models (about 21–37 dB [10, 12, 18]) used different data, resolutions and references and are not directly comparable. Optimisation [10–17], feed-forward networks [18, 19] and hybrid streaming [26] remain preferable when the highest fidelity matters and their cost is acceptable. CTBakeGS does not compete with them on fidelity but removes their prerequisites.

Faint splats were the main source of error in translucent layers. Standard 3DGS rasterisers discard opacities below 1/255, and the thin outer soft-tissue layer of the skin-and-bone look consists of such faint cells, which are visible in the volume rendering only because many of them add up along oblique rays. Without the retention of Step 5, this layer was largely missing in case 2 and the ribs showed through it (Fig. 3); its wide-view PSNR fell from 29.8 to 23.8 dB, and bone and vessels in that case lost 0.5 and 0.7 dB (Table 4). In exploratory runs, lowering the discard threshold to 0.001 or not splitting faint cells did not restore the layer (+0.1 and +0.3 dB); retaining faint splats stochastically with raised opacity did. The retained splats are individually denser than the tissue they represent, which gives the layer a fine grain at close range. Isotropic splats may still represent thin translucent sheets poorly, and 3DGS rasterisers sort whole splats rather than samples along each ray [27]. The translucent skin layer is itself a simplification: real skin has a complex, spectrally varying reflectance with subsurface scattering in its upper layers [28], and easy-to-use, realistic skin rendering from CT remains for future research.

a Volume rendering (reference)  
b 3DGS without faint splats  
c 3DGS with faint splats  
![](images/2cd5041c82052c85e32171cac936a9943b06deb61960e29be7e050d71e31d294.jpg)  
Fig. 3 Faint splats: case 2, skin and bone, two million splats. (a) Converged photorealistic volume rendering (reference); (b) 3DGS model without faint splats (ablation of Step 5); (c) 3DGS model with faint splats (CTBakeGS). Top row: one wide view, cropped to the body; middle row: close-up of the dashed box; bottom row: difference from the reference as in Fig. 2 (0 to ≥ 50 grey levels of 255). Without them, the thin outer soft-tissue layer, whose splats fall below the 8-bit opacity floor, is missing and the ribs show through (PSNR 22.5 dB in this view); with them it is restored (30.1 dB), with a fine grain visible at close range. 3DGS, three-dimensional Gaussian splatting; PSNR, peak signal-to-noise ratio

At equal splat count, the octree was not more faithful than a uniform grid. Its benefit is that the splat count can be chosen, for example two million for every volume, instead of being dictated by the number of voxels, which ranged from 0.14 to 2.86 million here.

This study has limitations. Six cases from one public dataset were used, with 1.5-mm voxels, which are slightly coarser than the maximum slice thickness of 1.25 mm proposed for three-dimensional reconstruction in forensic casework [29]; clinical thin-slice scans have up to ten times more voxels and convert more slowly. We also stopped at three million splats; close-up fidelity was still rising at that count, so more splats may improve it further at the cost of file size and frame rate, whereas wide-view fidelity had levelled off. The evaluation has two boundaries of its own. Fidelity was measured against the converged view of the same renderer, whose lighting only approximates global illumination, so agreement with other photorealistic renderers is unknown, and PSNR, SSIM and LPIPS do not establish anatomical accuracy. Nor did we test the use of the models: whether they aid teaching or communication, or are suitable for diagnostic reading, requires reader studies. Finally, two choices in the method were made rather than optimised. The opacity floor of the retained faint splats $( \alpha _ { 0 } = 0 . 0 1 )$ is roughly the smallest value that stays above the 1/255 cut-off after the anti-aliasing compensation, which reduces the drawn opacity of small splats, and the two interact; and lighting, transfer function and clipping are fixed at conversion, so that view-independent colours lose specular highlights.

Several of these limitations point to the next steps. A larger series, including clinical thin-slice scans, should be tested, and conversion of thin-slice and very large volumes should become faster, for example by caching the Hounsfield-unit ranges of the volume for the octree. It also remains to be determined which computer is the minimum for displaying (rather than converting) the models, how they compare with volume rendering and path tracing in this respect and in compressibility. Whether fine structures such as hairline fractures are preserved also needs to be tested, and the error between the model and the rendering could be used to place more splats where details are lost. Fidelity could rise further with view-dependent colours (spherical harmonics), which would restore highlights, and with flat or anisotropic splats for thin translucent layers [30]. The converted model could also be edited directly, cut with a clip plane or recoloured by segmented organ, without reconverting. Above all, reader studies are needed to show whether the models aid teaching and communication.

## 5 Conclusions

We could demonstrate that photorealistic CT volume renderings can be converted directly into standard 3D Gaussian splats, without training, and displayed in a web browser. Conversion took about a minute or less per pass on a current laptop and on a 2018 desktop with a 4-GB graphics card, and under 4 minutes per pass on a 2017 laptop with integrated graphics. The models reproduce the rendering with root-mean-square errors of about 2.3–3.1% of full scale in wide views. This approach could make complex CT data easier to share and view interactively without the CT data or a volume renderer, with potential value for communication and teaching.

## Abbreviations

3DGS, three-dimensional Gaussian splatting; CT, computed tomography; EWA, elliptical weighted average; GPU, graphics processing unit; LPIPS, learned perceptual image patch similarity; PSNR, peak signal-to-noise ratio; RMS, root-mean-square; SSIM, structural similarity index.

## Acknowledgements

Not applicable.

## Declaration of the use of artificial intelligence

The software code was written with the assistance of an artificial intelligence model (Claude Opus 5.5, Anthropic), which was also used to help draft, edit and translate the manuscript. The authors reviewed, tested and revised all code and text and take full responsibility for the content of this article.

## Authors’ contributions

Conceptualization, H.F., W.S., C.L.; methodology, H.F.; software, H.F. and W.S. ; validation, H.F.; formal analysis, H.F.; investigation, H.F.; resources, H.F.; data curation, H.F.; writing—original draft preparation, H.F.; writing—review and editing, H.F., C.L., W.S., T.H. and L.E. ; visualization, H.F.; supervision, H.F.; project administration, H.F.; funding acquisition, H.F. All authors have read and agreed to the published version of the manuscript.

## Funding

This work was supported in part by KAKENHI Grant Number 25K20648 and Overseas Research Fellowships (grant number 202560103) from the Japan Society for the Promotion of Science.

## Availability of data and materials

The source code and the evaluation scripts are available at https://github.com/BocchiSlice/CTBakeGS. The evaluation outputs are available from the corresponding author on reasonable request.

## Declarations

Ethics approval and consent to participate. Not applicable. This study used only publicly available, de-identified CT data (TotalSegmentator dataset, CC BY 4.0).

Consent for publication. Not applicable.

Competing interests. The authors declare that they have no competing interests.

## References

[1] Kroes T, Post FH, Botha CP (2012) Exposure Render: an interactive photo-realistic volume rendering framework. PLoS On 7:e38586. doi:10.1371/journal.pone.0038586

[2] Jönsson D, Sundén E, Ynnerman A, Ropinski T (2014) A survey of volumetric illumination techniques for interactive volume rendering. Comput Graph Forum 33:27–51. doi:10.1111/cgf.12252

[3] Dappa E, Higashigaito K, Fornaro J, Leschka S, Wildermuth S, Alkadhi H (2016) Cinematic rendering – an alternative to volume rendering for 3D computed tomography imaging. Insights Imaging 7:849–856. doi:10.1007/s13244-016-0518-1

[4] Eid M, De Cecco CN, Nance JW et al (2017) Cinematic rendering in CT: a novel, lifelike 3D visualization technique. AJR Am J Roentgenol 209:370–379. doi:10.2214/AJR.17.17850

[5] Elshafei M, Binder J, Baecker J et al (2019) Comparison of cinematic rendering and computed tomography for speed and comprehension of surgical anatomy. JAMA Surg 154:738–744. doi:10.1001/jamasurg.2019.1168

[6] Binder JS, Scholz M, Ellmann S et al (2021) Cinematic rendering in anatomy: a crossover study comparing a novel 3D reconstruction technique to conventional computed tomography. Anat Sci Educ 14:22–31. doi:10.1002/ase.1989

[7] Pachowsky ML, Morf H, Simon D et al (2022) Cinematic rendering in rheumatic diseases—photorealistic depiction of pathologies improves disease understanding for patients. Front Med (Lausanne) 9:946106. doi:10.3389/fmed.2022.946106

[8] Ebert LC, Schweitzer W, Gascho D, Ruder T, Flach PM, Thali MJ, Ampanozi G (2017) Forensic 3D visualization of CT data using cinematic volume rendering: a preliminary study. AJR Am J Roentgenol 208:233–240. doi:10.2214/AJR.16.16499

[9] Kerbl B, Kopanas G, Leimkühler T, Drettakis G (2023) 3D Gaussian splatting for real-time radiance field rendering. ACM Trans Graph 42:139. doi:10.1145/3592433

[10] Niedermayr S, Neuhauser C, Petkov K, Engel K, Westermann R (2024) Application of 3D Gaussian splatting for cinematic anatomy on consumer class devices. In: Linsen L, Thies J (eds) Vision, Modeling, and Visualization (VMV 2024). The Eurographics Association. doi:10.2312/vmv.20241195

[11] Kleinbeck C, Schieber H, Engel K, Gutjahr R, Roth D (2025) Multi-layer Gaussian splatting for immersive anatomy visualization. IEEE Trans Vis Comput Graph 31:2353–2363. doi:10.1109/TVCG.2025.3549882

[12] Li C, Tong Y, Chen K et al (2025) ClipGS: clippable Gaussian splatting for interactive cinematic visualization of volumetric medical data. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. Lecture Notes in Computer Science, vol 15969. Springer, Cham, pp 127–137. doi:10.1007/978-3-032-05127-1\_13

[13] Gao Z, Planche B, Zheng M, Choudhuri A, Chen T, Wu Z (2025) 6DGS: enhanced direction-aware Gaussian splatting for volumetric rendering. In: International Conference on Learning Representations (ICLR 2025). arXiv:2410.04974

[14] Tang K, Yao S, Wang C (2025) iVR-GS: inverse volume rendering for explorable visualization via editable 3D Gaussian splatting. IEEE Trans Vis Comput Graph 31:3783–3795

[15] Dyken L, Sewell A, Usher W et al (2026) Volume encoding Gaussians: transfer function-agnostic 3D Gaussians for volume rendering. IEEE Trans Vis Comput Graph (PacificVis 2026). arXiv:2504.13339

[16] Gao Z, Planche B, Zheng M, Choudhuri A, Chen T, Wu Z (2026) FactorSplat: appearance-controllable Gaussian proxies for medical volume rendering. arXiv:2610.02382

[17] Li Q, Wang Y, Cai Y, Du H, Liu L (2026) Variable basis mapping for real-time volumetric visualization. arXiv:2601.09417

[18] Gao Z, Planche B, Zheng M et al (2026) Render-FM: feedforward model for real-time photorealistic volumetric rendering. In: European Conference on Computer Vision (ECCV 2026), in press. arXiv:2505.17338

[19] Wang Y, Li Q, Cai Y, Liu L (2026) VVGT: visual volume-grounded transformer. arXiv:2604.12217

[20] Zwicker M, Pfister H, van Baar J, Gross M (2001) EWA volume splatting. In: Proceedings of IEEE Visualization 2001. IEEE, pp 29–36, 538. doi:10.1109/VISUAL.2001.964490

[21] Max N (1995) Optical models for direct volume rendering. IEEE Trans Vis Comput Graph 1:99–108. doi:10.1109/2945.468400

[22] Yu Z, Chen A, Huang B, Sattler T, Geiger A (2024) Mip-Splatting: alias-free 3D Gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 19447–19456. doi:10.1109/CVPR52733.2024.01839

[23] Wasserthal J, Breit HC, Meyer MT et al (2023) TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol Artif Intell 5:e230024. doi:10.1148/ryai.230024

[24] Wang Z, Bovik AC, Sheikh HR, Simoncelli EP (2004) Image quality assessment: from error visibility to structural similarity. IEEE Trans Image Process 13:600–612. doi:10.1109/TIP.2003.819861

[25] Zhang R, Isola P, Efros AA, Shechtman E, Wang O (2018) The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp 586–595

[26] Kleinbeck C, Theelke L, Schieber H, Eck U, von Eisenhart-Rothe R, Roth D (2026) Hybrid foveated path tracing with peripheral Gaussians for immersive anatomy. In: 2026 IEEE Conference on Virtual Reality and 3D User Interfaces (VR). doi:10.1109/VR67842.2026.00078

[27] Radl L, Steiner M, Parger M, Weinrauch A, Kerbl B, Steinberger M (2024) StopThePop: sorted Gaussian splatting for view-consistent real-time rendering. ACM Trans Graph 43:64. doi:10.1145/3658187

[28] Gitlina Y, Guarnera GC, Dhillon DS, Hansen J, Lattas A, Pai D, Ghosh A (2020) Practical measurement and reconstruction of spectral skin reflectance. Comput Graph Forum 39:75–89. doi:10.1111/cgf.14055

[29] Ford JM, Decker SJ (2016) Computed tomography slice thickness and its effects on three-dimensional reconstruction of anatomical structures. J Forensic Radiol Imaging 4:43–46. doi:10.1016/j.jofri.2015.10.004

[30] Huang B, Yu Z, Chen A, Geiger A, Gao S (2024) 2D Gaussian splatting for geometrically accurate radiance fields. In: ACM SIGGRAPH 2024 Conference Papers, article 32. doi:10.1145/3641519.3657428

## Supplementary material

Table S1 Cases (TotalSegmentator dataset v2)
<table><tr><td>Case</td><td>ID</td><td>Description</td><td>Matrix (voxels)</td><td>Voxels (million)</td><td>MacBook Pro (s)</td><td>Windows desktop (s)</td><td>MacBook Air 2017 (s)</td></tr><tr><td>1</td><td>s0289</td><td>Head CT angiography</td><td> $1 4 5 \times 1 4 5 \times 2 3 3$ </td><td>4.9</td><td>5.4</td><td>11.1</td><td>46</td></tr><tr><td>2</td><td>s0138</td><td>Heart and thoracic aorta</td><td> $2 2 1 \times 2 2 1 \times 2 1 5$ </td><td>10.5</td><td>12.7</td><td>25.4</td><td>82</td></tr><tr><td>3</td><td>s1340</td><td>Thorax to pelvis</td><td> $2 7 5 \times 2 7 5 \times 4 3 1$ </td><td>32.6</td><td>18.5</td><td>37.9</td><td>131</td></tr><tr><td>4</td><td>s0970</td><td>Polytrauma</td><td> $3 3 3 \times 3 3 3 \times 5 1 1$ </td><td>56.7</td><td>20.8</td><td>43.2</td><td>166</td></tr><tr><td>5</td><td>s1121</td><td>Pelvis-to-leg CT angiography</td><td> $2 8 1 \times 2 8 1 \times 5 1 9$ </td><td>41.0</td><td>20.5</td><td>42.4</td><td>148</td></tr><tr><td>6</td><td>s0931</td><td>Neck-to-leg CT angiography</td><td> $2 4 5 \times 1 8 7 \times 4 7 9$ </td><td>21.9</td><td>8.0</td><td>16.6</td><td>60</td></tr></table>

All images are resampled to 1.5-mm isotropic voxels in the released dataset; the last dimension is the number of slices. The last three columns give the conversion time, one pass at the final cell size, for two million splats.

Table S2 Paired comparisons of each variant with the full method (two million splats)
<table><tr><td>Look</td><td>Views</td><td>Variant</td><td>∆PSNR (dB)</td><td>p</td><td>∆SSIM</td><td>p</td><td>∆LPIPS</td><td>p</td></tr><tr><td>Bone</td><td>wide</td><td>1 M splats</td><td>-0.885</td><td>0.156</td><td>-0.003</td><td>0.312</td><td>+0.005</td><td>0.094</td></tr><tr><td>Bone</td><td>wide</td><td>3M splats</td><td>+0.036</td><td>1.000</td><td>-0.000</td><td>0.562</td><td>-0.001</td><td>0.438</td></tr><tr><td>Bone</td><td>wide</td><td>No anti-aliasing compensation</td><td>-3.832</td><td>0.031</td><td>-0.010</td><td>0.031</td><td>+0.001</td><td>0.562</td></tr><tr><td>Bone</td><td>wide</td><td>No opacity mapping</td><td>-4.498</td><td>0.031</td><td>-0.012</td><td>0.062</td><td>+0.001</td><td>1.000</td></tr><tr><td>Bone</td><td>wide</td><td>Without faint splats</td><td>-0.274</td><td>0.031</td><td>-0.001</td><td>0.031</td><td>+0.001</td><td>0.031</td></tr><tr><td>Bone</td><td>wide</td><td>Uniform grid (1 cell/voxel)</td><td>-0.171</td><td>0.438</td><td>-0.000</td><td>0.438</td><td>+0.001</td><td>0.219</td></tr><tr><td>Bone</td><td>wide</td><td>Uniform grid vs octree, equal count</td><td>+0.393</td><td>0.031</td><td>+0.001</td><td>0.031</td><td>-0.001</td><td>0.031</td></tr><tr><td>Bone</td><td>close-up</td><td>1 M splats</td><td>-1.239</td><td>0.031</td><td>-0.010</td><td>0.031</td><td>+0.010</td><td>0.031</td></tr><tr><td>Bone</td><td>close-up</td><td>3M splats</td><td>+0.614</td><td>0.031</td><td>+0.006</td><td>0.031</td><td>-0.006</td><td>0.062</td></tr><tr><td>Bone</td><td>close-up</td><td>No anti-aliasing compensation</td><td>-0.398</td><td>0.062</td><td>-0.002</td><td>0.094</td><td>-0.001</td><td>0.031</td></tr><tr><td>Bone</td><td>close-up</td><td>No opacity mapping</td><td>-5.061</td><td>0.031</td><td>-0.037</td><td>0.031</td><td>+0.005</td><td>0.562</td></tr><tr><td>Bone</td><td>close-up</td><td>Without faint splats</td><td>-0.445</td><td>0.062</td><td>-0.006</td><td>0.062</td><td>-0.004</td><td>0.031</td></tr><tr><td>Bone</td><td>close-up</td><td>Uniform grid (1 cell/voxel)</td><td>-0.594</td><td>0.062</td><td>-0.006</td><td>0.062</td><td>+0.005</td><td>0.062</td></tr><tr><td>Bone</td><td>close-up</td><td>Uniform grid vs octree, equal count</td><td>+0.554</td><td>0.031</td><td>+0.006</td><td>0.031</td><td>-0.006</td><td>0.031</td></tr><tr><td>Vessels</td><td>wide</td><td>1 M splats</td><td>-0.926</td><td>0.031</td><td>-0.007</td><td>0.031</td><td>+0.007</td><td>0.031</td></tr><tr><td>Vessels</td><td>wide</td><td>3M splats</td><td>+0.445</td><td>0.062</td><td>+0.002</td><td>0.156</td><td>-0.003</td><td>0.062</td></tr><tr><td>Vessels</td><td>wide</td><td>No anti-aliasing compensation</td><td>-0.854</td><td>0.031</td><td>-0.005</td><td>0.062</td><td>+0.001</td><td>0.312</td></tr><tr><td>Vessels</td><td>wide</td><td>No opacity mapping</td><td>-1.327</td><td>0.031</td><td>-0.010</td><td>0.031</td><td>+0.004</td><td>0.062</td></tr><tr><td>Vessels</td><td>wide</td><td>Without faint splats</td><td>-0.385</td><td>0.031</td><td>-0.002</td><td>0.031</td><td>+0.003</td><td>0.031</td></tr><tr><td>Vessels</td><td>wide</td><td>Uniform grid (1 cell/voxel)</td><td>-0.035</td><td>0.562</td><td>+0.000</td><td>1.000</td><td>+0.000</td><td>0.562</td></tr><tr><td>Vessels</td><td>wide</td><td>Uniform grid vs octree, equal count</td><td>-0.004</td><td>0.438</td><td>+0.000</td><td>0.042</td><td>+0.000</td><td>0.438</td></tr><tr><td>Vessels</td><td>close-up</td><td>1 M splats</td><td>-1.313</td><td>0.031</td><td>-0.016</td><td>0.031</td><td>+0.024</td><td>0.031</td></tr><tr><td>Vessels</td><td>close-up</td><td>3M splats</td><td>+0.764</td><td>0.031</td><td>+0.008</td><td>0.031</td><td>-0.011</td><td>0.031</td></tr><tr><td>Vessels</td><td>close-up</td><td>No anti-aliasing compensation</td><td>-0.049</td><td>0.094</td><td>-0.000</td><td>0.312</td><td>-0.000</td><td>0.844</td></tr><tr><td>Vessels</td><td>close-up</td><td>No opacity mapping</td><td>-1.926</td><td>0.031</td><td>-0.018</td><td>0.031</td><td>+0.009</td><td>0.031</td></tr><tr><td>Vessels</td><td>close-up</td><td>Without faint splats</td><td>-0.528</td><td>0.031</td><td>-0.005</td><td>0.031</td><td>+0.001</td><td>0.688</td></tr><tr><td>Vessels</td><td>close-up</td><td>Uniform grid (1 cell/voxel)</td><td>-0.184</td><td>0.562</td><td>-0.000</td><td>1.000</td><td>+0.001</td><td>0.562</td></tr><tr><td>Vessels</td><td>close-up</td><td>Uniform grid vs octree, equal count</td><td>-0.055</td><td>0.438</td><td>+0.000</td><td>0.438</td><td>-0.000</td><td>0.156</td></tr><tr><td>Skin and bone</td><td>wide</td><td>1 M splats</td><td>-0.645</td><td>0.062</td><td>-0.008</td><td>0.031</td><td>+0.011</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>wide</td><td>3M splats</td><td>-0.115</td><td>0.562</td><td>+0.002</td><td>0.438</td><td>-0.004</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>wide</td><td>No anti-aliasing compensation</td><td>-0.308</td><td>0.156</td><td>+0.000</td><td>1.000</td><td>-0.002</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>wide</td><td>No opacity mapping</td><td>-5.274</td><td>0.031</td><td>-0.015</td><td>0.031</td><td>-0.001</td><td>0.562</td></tr><tr><td>Skin and bone</td><td>wide</td><td>Without faint splats</td><td>-1.632</td><td>0.031</td><td>-0.003</td><td>0.031</td><td>-0.001</td><td>0.438</td></tr><tr><td>Skin and bone</td><td>wide</td><td>Uniform grid (1 cell/voxel)</td><td>-0.242</td><td>0.219</td><td>-0.003</td><td>0.312</td><td>+0.006</td><td>0.438</td></tr><tr><td>Skin and bone</td><td>wide</td><td>Uniform grid vs octree, equal count</td><td>-0.125</td><td>0.438</td><td>+0.000</td><td>0.893</td><td>-0.001</td><td>0.438</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>1 M splats</td><td>-0.857</td><td>0.031</td><td>-0.009</td><td>0.031</td><td>+0.012</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>3M splats</td><td>+0.355</td><td>0.031</td><td>+0.004</td><td>0.094</td><td>-0.007</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>No anti-aliasing compensation</td><td>+0.112</td><td>0.031</td><td>+0.000</td><td>0.031</td><td>-0.000</td><td>0.031</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>No opacity mapping</td><td>-5.866</td><td>0.031</td><td>-0.025</td><td>0.031</td><td>+0.001</td><td>0.438</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>Without faint splats</td><td>-1.382</td><td>0.031</td><td>-0.004</td><td>0.062</td><td>-0.004</td><td>0.094</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>Uniform grid (1 cell/voxel)</td><td>-0.521</td><td>0.094</td><td>-0.006</td><td>0.156</td><td>+0.007</td><td>0.156</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>Uniform grid vs octree, equal count</td><td>-0.073</td><td>0.562</td><td>+0.000</td><td>0.844</td><td>-0.003</td><td>0.062</td></tr></table>

Median of the per-case differences (variant minus full method) over six cases; metrics were first averaged over the 48 wide or 16 close-up views of each scene. p: two-sided exact Wilcoxon signed-rank test over six cases (smallest attainable value 0.031); exploratory, not adjusted for multiple comparisons.

Table S3 Thick slices: case 1, PSNR (dB) / SSIM / LPIPS
<table><tr><td>Look</td><td>Views</td><td>1.5-mm data, millimetre cells</td><td>5-mm slices, millimetre cells</td><td>5-mm slices, voxel-unit cells</td></tr><tr><td>Bone</td><td>wide</td><td>33.61 / 0.982 / 0.031</td><td>34.33 / 0.983 / 0.032</td><td>22.54 / 0.898 / 0.109</td></tr><tr><td>Bone</td><td>close-up</td><td>30.03 / 0.961 / 0.178</td><td>31.67 / 0.966 / 0.193</td><td>17.63 / 0.723 / 0.367</td></tr><tr><td>Vessels</td><td>wide</td><td>30.22 / 0.966 / 0.042</td><td>31.02 / 0.964 / 0.043</td><td>24.64 / 0.901 / 0.112</td></tr><tr><td>Vessels</td><td>close-up</td><td>27.39 / 0.925 / 0.156</td><td>28.00 / 0.924 / 0.181</td><td>20.07 / 0.761 / 0.309</td></tr><tr><td>Skin and bone</td><td>wide</td><td>32.26 / 0.968 / 0.088</td><td>32.87 / 0.970 / 0.087</td><td>20.46 / 0.843 / 0.171</td></tr><tr><td>Skin and bone</td><td>close-up</td><td>29.36 / 0.942 / 0.306</td><td>29.77 / 0.950 / 0.289</td><td>16.34 / 0.756 / 0.413</td></tr></table>

Two million splats in all three conditions. Each condition is compared with the reference rendered from the same data (1.5-mm or 5-mm).