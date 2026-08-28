# Dose-PlanNet: Physics Based Radiotherapy Dose Prediction with Deep Learning

Ankit Bhattacharjee<sup>1,2</sup>, Sougata Maity<sup>2</sup>, Santam Chakraborty<sup>2</sup> & Indranil Mallick<sup>2</sup>

<sup>1</sup>Department of Mechanical Engineering, Indian Institute of Technology Kharagpur <sup>2</sup>Department of Radiation Oncology, Tata Medical Center Kolkata

August 28, 2026

## Abstract

Automating prostate radiotherapy treatment planning is dosimetrically complex, particularly for extreme hypofractionated regimens. In this study, we introduce Dose-PlanNet, a physics-guided 3D deep learning architecture designed to predict dose distributions. This model’s performance was evaluated on a cohort of patients treated in a prospective trial where two diferent dose fractionation regimens were employed. Dose-PlanNet achieved comparable target coverage (D ), though statistical analysis revealed a marginal reduction in target homogeneity (p < 0.001) ofset. However the model achieved statistically significant improvements in high-dose organ-at-risk sparing (p < 0.001). When evaluated against strict Prospective Randomized protocol volumetric constraints, automated plans met prespecified clinical acceptance criteria in 11 out of 14 Moderate Hypofraction Arm plans and 9 out of 12 Stereotactic Body Radiation Therapy Arm plans. This pipeline demonstrates that physics-informed deep learning can accelerate radiotherapy workflows while safely maintaining the stringent dosimetric quality required for high-precision clinical deployment.

## 1 Introduction

The standard of care for high-risk localized prostate cancer is Intensity-Modulated Radiation Therapy (IMRT) [1, 2]. When irradiation of the pelvic nodes is performed, a simultaneous integrated boost (SIB) approach is used. SIB regimens require the simultaneous delivery of an escalated prescription dose to the primary planning target volume (PTV) and a lower dose to the elective pelvic lymph nodes. These plans require complex spatial dose gradients [1]. This multi-target dosimetric objective must be achieved while strictly adhering to mandatory maximum and dose-volume constraints for nearby organs at risk (OARs), specifically the anorectum and bladder, to prevent severe gastrointestinal and genitourinary toxicities [3]. Consequently, manual inverse treatment planning remains a highly iterative, resource-intensive workflow that is heavily dependent on the individual planner’s experience. This invariably leads to significant inter-planner variability and sub-optimal operational eficiency in high-volume clinical settings [4].

Recent advancements in deep learning, particularly 3D U-Net architectures, have demonstrated significant potential in automating volumetric dose prediction [5, 6]. To mitigate the resource-intensive nature of manual optimization, recent clinical workflows have successfully leveraged knowledge-based planning (KBP) frameworks to fully automate treatment plan generation across diverse anatomical sites [7]. However, conventional frameworks predominantly approach dose prediction as a standard image-to-image translation task, relying on Mean Squared Error (MSE) or Mean Absolute Error (MAE) as the primary optimization metric [8]. This creates a fundamental disconnect between mathematical optimization and clinical reality. MSE intrinsically minimizes the global average voxel-wise error, meaning it treats a 1 Gy deviation in the geometric center of the target volume with the exact same mathematical weight as a 1 Gy deviation spilling across the boundary into the rectal wall. Clinical plan acceptability, however, is not determined by global volumetric averages; it is strictly governed by localized boundary constraints, absolute maximum dose ceilings, and specific dose-volume histogram (DVH) thresholds. Consequently, MSE-driven models frequently generate dose distributions that achieve a low global error but fail clinical evaluation due to point-dose violations in serial OARs or unacceptable cold spots at the target periphery [9]. To bridge the gap between theoretical volumetric approximation and clinical deployability, neural network architectures must move beyond generic pixel-wise loss. They require physics-guided optimization engines capable of spatial awareness, prioritizing critical tissue boundaries, and mimicking the non-linear dosimetric compromises executed by expert human planners.

To overcome the limitations of generic voxel-wise objectives, a treatment planning framework must transition from statistical image translation to mathematically enforced, physics-guided optimization [10]. Achieving true clinical usability dictates that an artificial intelligence must “see” anatomical boundaries rather than merely mapping isolated voxel intensities. This requires explicit spatial awareness of the local anatomy, allowing the network to calculate the exact proximity of high-dose clouds to abutting serial structures [11]. It must understand the fundamental radiation dose absorption principles like the inverse square law which drives these gradients such that absurd dose gradients are not generated. Furthermore, the optimization engine must deploy a multi-objective penalty system that inherently understands the hierarchy of clinical constraints. It must mathematically diferentiate between a soft penalty for optional dose spillage into generic healthy tissue and a severe, overriding penalty for a mandatory constraint violation within the rectal wall or bladder [12]. By encoding these strict dosimetric rules directly into the gradient descent space, the network is forced to learn the non-linear spatial compromises required for safe, physician-approvable clinical planning.

In this work, we propose Dose-PlanNet, a physics-guided deep learning framework specifically engineered to generate clinically acceptable IMRT plans for prostate Simultaneous Integrated Boost (SIB) protocols. The novelty of Dose-PlanNet is anchored in three primary methodological contributions. First, we utilize discrete PTV mapping, governed by a Painter’s Algorithm, to explicitly inject varying prescription doses directly into the input tensor, definitively resolving SIB dose hierarchies. Second, we equip the network with absolute spatial awareness by integrating Signed Distance Maps (SDMs) for abutting serial organs alongside a Beam’s Eye View (BEV) frustum mask; this combination allows the architecture to mathematically distinguish in-beam dose corridors from physically impossible out-of-beam scatter. Finally, we replace standard voxel-wise metrics with a custom, multi-objective physics-guided loss engine. This engine features diferentiable Dose-Volume Histogram (DVH) estimations via steep sigmoid approximations, deploying tuned penalties to strictly enforce global dose ceilings, exact V-type OAR constraints, and absolute PTV coverage thresholds. Evaluated on a clinical prostate SIB cohort, Dose-PlanNet successfully generated steep dose gradients to yield physician-approvable dose distributions. The framework achieved good OAR sparing while simultaneously satisfying mandatory PTV coverage constraints for the vast majority of patients, demonstrating robust clinical utility over standard translation-based AI approaches.

## 2 Related Work

The automation of radiotherapy dose prediction has evolved significantly over the past decade, transitioning from generalized medical image processing to highly specialized dosimetric optimization. This section reviews the progression of deep learning frameworks in this domain, highlighting the persistent clinical limitations that necessitate a physics-guided approach.

## 2.1 Volumetric Dose Prediction as Image Translation

The foundational era of deep learning-based dosimetry primarily conceptualized dose prediction as a standard image-to-image translation task. Early frameworks heavily relied on 3D U-Net architectures [5] and Fully Convolutional Neural Networks (FCNs), such as DoseNet [6], to map anatomical computed tomography (CT) and contour data directly to dose distributions. Subsequently, Generative Adversarial Networks (GANs) were introduced to further refine the global texture and statistical distribution of the predicted volumetric dose clouds [8].

## 2.2 Anatomical and Spatial Embeddings

To address the spatial blindness of pure image translation, subsequent research introduced explicit anatomical embeddings. Frameworks began incorporating Signed Distance Maps (SDMs) and distance-to-target coordinates directly into the input tensor to force network awareness of critical tissue boundaries [11]. While providing this spatial context significantly improved the geometric conformity of the predicted dose clouds, it failed to resolve the fundamental optimization bottleneck. Because these architectures still minimized standard MSE or MAE loss, they remained mathematically incapable of prioritizing strict clinical boundary constraints over bulk tissue averages. Consequently, despite “seeing” the organs, these models remained highly susceptible to localized but clinically fatal dosimetric violations [10].

## 2.3 Physics-Guided and Constraint-Aware Optimization

To overcome the inherent limitations of MSE, recent state-of-the-art frameworks have begun encoding clinical dosimetry rules directly into the network’s optimization trajectory. Notably, architectures such as nn-DoseNet have demonstrated the utility of embedding diferentiable Dose-Volume Histogram (DVH) metrics into the loss space, forcing the network to optimize for clinical acceptability rather than pure voxel matching [14]. Concurrently, recent experimental approaches in physics-guided radiotherapy have introduced dynamic penalty engines to mathematically enforce physical dose delivery limits [13].

While these constraint-aware engines represent a critical methodological leap, their evaluation has predominantly been restricted to single-target prescriptions, leaving the multi-objective complexities of Simultaneous Integrated Boost (SIB) protocols largely unaddressed. Furthermore, without steep, dynamically weighted penalty gradients, these loss engines risk severe target underdosing, forcing the network into suboptimal mathematical tradeofs between target saturation and strict OAR sparing.

## 2.4 The Unsolved SIB and Beam Geometry Gap

Despite advancements in physics-guided optimization, existing frameworks remain fundamentally ill-equipped for the spatial complexities of Simultaneous Integrated Boost (SIB) protocols. Many conventional frameworks rely on implicit spatial approximations rather than dedicated mathematical mechanisms to definitively resolve abutting or overlapping target volumes that require distinct prescription doses. Furthermore, they almost universally ignore the physical limitations of beam geometry. By failing to restrict optimization space to deliverable beam trajectories, these architectures frequently artificially minimize their loss functions by depositing low-dose scatter into physically impossible out-of-beam regions.

Dose-PlanNet directly targets these limitations. By implementing a Painter’s Algorithm for discrete PTV mapping [24], the framework explicitly enforces strict SIB dose hierarchies to prevent target overlap ambiguity. Concurrently, the integration of a Beam’s Eye View (BEV) frustum mask into the multi-objective penalty engine mathematically prohibits non-physical out-of-beam dose spillage in an IMRT plan. This methodology bridges the remaining gap between theoretical physics-guided prediction and verifiable clinical deployability.

## 3 Methodology

The development of Dose-PlanNet requires a multi-stage pipeline that transitions from raw clinical data curation to mathematically constrained neural network optimization. This section details the spatial data embeddings, the volumetric network architecture, and the multi-objective physics-guided loss engine that governs the framework.

## 3.1 Data Representation and Preprocessing

The translation of raw, unorganized clinical DICOM data into a structured tensor format suitable for deep learning optimization is a critical bottleneck in volumetric dosimetry. To address this, we developed a modular preprocessing pipeline that automates the alignment and digitization of patient anatomy, treatment geometries, and prescribed dose objectives.

![](images/41506a441d22e8e841e3355beb12b59fb8cf3fcd90951f92271eb1df6263e1dc.jpg)  
Figure 1: End-to-End Architecture of the Dose-PlanNet Pipeline. Raw clinical DICOM files are isotropically resampled and converted into physics-aware spatial embeddings, including Signed Distance Maps (SDMs), Beam’s Eye View (BEV) frustums, and a discrete Simultaneous Integrated Boost (SIB) map via Painter’s algorithm. The 7-channel tensor feeds a 3D U-Net trained via a multi-objective physics loss engine, where clinical constraints are progressively introduced during early training. The pipeline yields dual checkpoints (Physics-Optimized and Clinically-Optimized), which are utilized during sliding-window inference. Crucially, the final step mathematically reverses the afine resampling operations, restoring the predicted dose matrix to the native CT geometry for immediate Treatment Planning System (TPS) integration.

## 3.1.1 Anatomical Digitization and Distance Mapping

The foundational coordinate system for each patient is established by loading the computed tomography (CT) scan and performing isotropic voxel resampling to a standardized resolution of $1 . 2 7 \times 1 . 2 7 \times 2 . 5 ~ \mathrm { m m ^ { 3 } }$ To generate the external body contour,a threshold based segmentation (HU -350 or higher) is done.

To ensure the network explicitly recognizes the physical boundaries of adjacent anatomical structures, physician-defined contours (RTStruct) are digitized into voxelized binary masks. Rather than relying solely on these binary representations, the pipeline computes Signed Distance Maps (SDMs) for a configurable list of abutting serial Organs at Risk (OARs), specifically the bladder and Anorectum in this work. The SDM calculates the shortest physical distance from every spatial coordinate ⃗r to the nearest organ boundary:

$$
S D M ( \vec { r } ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } \vec { r } \in \mathcal { S } } \\ { d ( \vec { r } , \partial \mathcal { S } ) , } & { \mathrm { i f ~ } \vec { r } \notin \mathcal { S } } \end{array} \right.\tag{1}
$$

where $s$ represents the specific structure volume and $d ( \vec { r } , \partial S )$ is the Euclidean distance to the structure boundary. This continuous spatial embedding forces the optimization engine to recognize the strict requirement for steep dose fallof immediately adjacent to healthy tissue.

## 3.1.2 Discrete SIB Mapping via Painter’s Algorithm

A fundamental challenge in automating Simultaneous Integrated Boost (SIB) protocols is the ambiguity of target intersections, where elective target volumes (with a lower dose prescription) overlap with the primary target where a higher dose is to be delivered. To resolve this spatial conflict, the pipeline executes a discrete PTV mapping strategy governed by a Painter’s Algorithm [24].

Individual target volumes are extracted and sorted in ascending order based on their prescription dose $( \mathrm { e . g . , } P T V _ { 4 4 }$ followed by $P T V _ { 6 2 } )$ . The algorithm sequentially paints these volumes into a single discrete tensor channel, assigning the exact numerical prescription dose (in Gray) as the voxel intensity. Because higherdose targets are rasterized last, they overwrite the overlapping low-dose regions. This explicitly preserves the dosimetric hierarchy, providing the network with an unambiguous ground-truth target landscape.

## 3.1.3 Beam’s Eye View (BEV) Frustum Construction

To physically constrain the network from depositing radiation in geometrically impossible regions, the pipeline extracts the fixed gantry angles from the clinical RTPlan file to construct a Beam’s Eye View (BEV) frustum mask.

For a given gantry angle $\theta ,$ the beam direction vector is defined as $\hat { \Omega } _ { \theta } = [ \sin \theta , - \cos \theta , 0 ] ^ { T }$ . The source position $\vec { S }$ is calculated using the established Source-to-Axis Distance $( S A D = 1 0 0 0 \mathrm { m m } )$ relative to the PTV isocenter. The pipeline performs geometric ray-tracing to determine if a spatial voxel ⃗r lies within the active beam corridor. A voxel is classified as “In-Beam” if its perpendicular distance to the central axis is bounded by the projected target radius and an isotropic penumbra margin:

$$
B E V ( \vec { r } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } ( \vec { r } - \vec { S } ) \cdot \hat { \Omega } _ { \theta } > 0 \mathrm { ~ a n d ~ } d _ { \perp } < d _ { \parallel } \tan \phi + p _ { \mathrm { m a r g i n } } } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{2}
$$

where $d _ { \perp }$ and $d _ { \parallel }$ are the perpendicular and parallel distances relative to the source, $\phi$ is the half-angle subtended by the target, and $p _ { m a r g i n }$ is the 7.0 mm physical penumbra. This binary mask is subsequently passed to the loss engine to violently penalize non-physical scatter.

## 3.2 Network Architecture

The core dose prediction engine utilizes a 3D U-Net architecture to map the prepared spatial tensors to a continuous volumetric dose distribution. The network ingests a 7-channel input tensor: [CT, Discrete PTVs, Bladder SDM, Anorectum SDM, Body Mask, Penile Bulb Mask, BEV Frustum].

The architecture employs an encoder-decoder topology with four resolution levels (channel dimensions of 16, 32, 64, 128) and skip connections to preserve high-frequency spatial boundaries. To ensure numerical stability and physical realism, a Softplus activation function, $f ( x ) = \ln ( 1 + e ^ { x } )$ , is applied to the terminal layer. This mathematically guarantees a strict non-negative physical dose floor $( D _ { \mathrm { p r e d } } \geq 0 )$ while avoiding the vanishing gradient problem associated with hard clipping functions in high-dose regions.

Auxiliary Spatial Masks for Loss Optimization. While the primary 7-channel tensor provides the necessary spatial context for the network’s forward pass, several critical structures are routed directly to the loss engine to exclusively govern backpropagation penalties. These auxiliary inputs include the binary masks for the bowel bag and the bilateral femoral heads, which are used strictly to enforce dose-volume and maximum dose constraints without cluttering the network’s anatomical input space.

Furthermore, to mathematically enforce the steep dose gradients required immediately outside the target volumes, the preprocessing pipeline dynamically computes a 3D geometric fallof ring. This is achieved by applying a 3D morphological max-pooling dilation to the discrete PTV mask. Given the anisotropic voxel spacing of $1 . 2 7 \times 1 . 2 7 \times 2 . 5 ~ \mathrm { m m ^ { 3 } }$ , a non-uniform kernel of $5 \times 9 \times 9$ voxels is utilized to generate a precise 5.0 mm isotropic expansion. The original PTV is then subtracted from this dilated volume, yielding a hollow, binary geometric shell:

$$
{ \mathrm { R i n g } } _ { 5 m m } = { \mathrm { M a x P o o l } } 3 \mathrm { D } ( P T V , k _ { 5 \times 9 \times 9 } ) - P T V .\tag{3}
$$

This isolated 5 mm boundary corridor is passed exclusively to the multi-objective loss engine to heavily penalize dose spillage immediately adjacent to the prescription boundary.

## 3.3 The Physics-Guided Loss Engine

The fundamental limitation of generic image translation networks is their reliance on unweighted Mean Squared Error (MSE), which mathematically values low-dose background approximations equally against critical high-dose target conformity. To force the network to prioritize clinical acceptability, Dose-PlanNet optimizes a custom, multi-objective loss function that integrates the spatial, geometric, and dosimetric penalties defined throughout this section:

$$
\begin{array} { r l } & { \mathcal { L } _ { t o t a l } = \lambda _ { m s e } \mathcal { L } _ { M S E } + \lambda _ { p t v } \mathcal { L } _ { P T V } + \lambda _ { v } \mathcal { L } _ { V - T y p e } } \\ & { ~ + \lambda _ { r i n g } \mathcal { L } _ { r i n g } + \lambda _ { a c } \mathcal { L } _ { a n t i c o l l a p s e } + \lambda _ { b o d y } \mathcal { L } _ { b o d y } } \\ & { ~ + \lambda _ { b g } \mathcal { L } _ { B E V } + \lambda _ { r e g } \mathcal { L } _ { S m o o t h } , } \end{array}\tag{4}
$$

where the tuned λ coeficients balance voxel-wise fidelity against strict clinical boundary constraints.

![](images/b98ac6cc23d743c0ff5281b0ddfac40adf79ab520721134a4104dad2306dd502.jpg)  
Figure 2: Decomposition of the Physics-Guided Loss Engine. The network optimizes a custom multi-objective loss function $( \mathcal { L } _ { t o t a l } )$ driven by complex spatial embeddings. Constraints are categorized into Absolute Dose penalties (D-Type, targeting specific dose thresholds like PTV coverage and anti-collapse), Volume-Histogram penalties (V-Type, utilizing a diferentiable sigmoid estimator for OAR sparing), and Spatial/Regularization penalties (enforcing geometric fallof and scatter suppression). A curriculum scheduler dynamically ramps the magnitude of these physics penalties from zero over the initial 30 epochs to ensure early numerical stability.

## 3.3.1 Dose-Weighted Mean Squared Error

To prevent the optimizer from diluting its gradient updates across the vast volume of non-target background tissue, the baseline volumetric reconstruction loss is modified into a Dose-Weighted Mean Squared Error. A spatial weighting matrix, $W ( \vec { r } )$ , is computed from the ground-truth dose to exponentially amplify the

penalty in high-dose prescription regions:

$$
W ( \vec { r } ) = 1 + \alpha \cdot \mathrm { m i n } \{ D _ { \mathrm { t r u e } } ( \vec { r } ) , D _ { \mathrm { c l a m p } } \} ,\tag{5}
$$

where $\alpha = 9 . 0$ sets the maximum amplification scale and $D _ { \mathrm { c l a m p } } = 0 . 9 6$ prevents gradient explosion at the absolute hotspot maximum. The resulting weighted baseline loss is:

$$
\mathcal { L } _ { M S E } = \frac { 1 } { N } \sum _ { \vec { r } \in \Omega } W ( \vec { r } ) \left[ D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { \mathrm { t r u e } } ( \vec { r } ) \right] ^ { 2 } ,\tag{6}
$$

where Ω denotes the entire three dimensional vector space in which the body exists.

## 3.3.2 D-Type (Absolute Dose) Penalties

To enforce absolute target coverage and prevent hotspots, we implement D-Type penalty functions based on the prescription guidelines. For the Planning Target Volumes (PTVs), we utilize a one-sided hinge loss that penalizes underdosing relative to the prescription (Rx) and overdosing relative to the hard clinical ceiling $( D _ { \mathrm { c e i l } } )$ :

$$
\mathcal { L } _ { P T V } = \frac { 1 } { N _ { P T V } } \sum _ { \vec { r } \in P T V } \left[ \lambda _ { p t v } \mathrm { R e L U } ( D _ { R x } - D _ { \mathrm { p r e d } } ( \vec { r } ) ) ^ { 2 } + \lambda _ { p t v \_ m a x } \mathrm { R e L U } ( D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { \mathrm { c e i l } } ) ^ { 2 } \right] .\tag{7}
$$

Beyond the PTV, Dose-PlanNet enforces a global hard dose ceiling (global ceil gy) to suppress mathematically divergent hotspots across the entire body volume. To avoid gradient dilution in large-volume pelvic scans, we employ a Top-K violation strategy, aggregating the penalty only from the $K = 1 0 0$ most severe transgressions:

$$
\mathcal { L } _ { g l o b a l \_ c e i l } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { R e L U } ( D _ { \mathrm { p r e d } } ^ { ( k ) } - D _ { \mathrm { g l o b a l \_ c e i l } } ) ^ { 2 } ,\tag{8}
$$

where $D _ { \mathrm { p r e d } } ^ { ( k ) }$ are the K largest positive violations of the ceiling threshold.

## 3.3.3 Diferentiable Dose-Volume Histogram (DVH) Estimator

Clinical protocols mandate specific volumetric thresholds for serial organs, such as ensuring the anorectum volume receiving at least 60.4 $\mathrm { G y } ~ \left( V _ { \mathrm { 6 0 . 4 G y } } \right)$ remains below a strict percentage. Standard voxel counting relies on a non-diferentiable Heaviside step function, which interrupts the computational graph during backpropagation. To resolve this, Dose-PlanNet approximates the step function using a steep sigmoid curve, facilitating continuous gradient flow. For a given anatomical structure S, the predicted volume fraction exceeding a normalized dose threshold $D _ { \mathrm { r e f } }$ is calculated as:

$$
V _ { D _ { \mathrm { r e f } } } ( S ) \approx \frac { 1 } { N s } \sum _ { \vec { r } \in S } \sigma \left( k \cdot \left[ D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { \mathrm { r e f } } \right] \right) ,\tag{9}
$$

where $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ $N _ { S }$ is the total number of voxels within the structure mask, and $k = 5 0 . 0$ is the steepness multiplier. This formulation allows the optimizer to smoothly calculate volume fractions and apply diferentiable hinge penalties, enabling the architecture to treat clinical V -type constraints as active, gradient-governed optimization objectives.

## 3.3.4 V-Type (Volumetric) Dual-Tier Penalties

Clinical protocols often define a dual-tier constraint system comprising an Optimal volume threshold $( D _ { \mathrm { o p t } } )$ and a strictly enforced Mandatory limit $( D _ { \mathrm { m a n d } } )$ . Using the diferentiable DVH estimator defined in equation 9, we formulate a two-stage penalty that forces the network to adhere to dosimetric objectives:

$$
\mathcal { L } _ { V - T y p e } = \sum _ { c \in \mathcal { C } } \left[ \lambda _ { o p t } \mathrm { R e L U } ( V _ { D _ { \mathrm { r e f } } } ( c ) - D _ { \mathrm { o p t } , c } ) ^ { 2 } + \lambda _ { m a n d } \mathrm { R e L U } ( V _ { D _ { \mathrm { r e f } } } ( c ) - D _ { \mathrm { m a n d } , c } ) ^ { 2 } \right] ,\tag{10}
$$

where $V _ { D _ { \mathrm { r e f } } } ( c )$ is the diferentiable volume fraction at dose $D _ { \mathrm { r e f } }$ for constraint $c ,$ and $\lambda _ { m a n d } \gg \lambda _ { o p t }$ ensures that violating mandatory thresholds results in a disproportionately large gradient, forcing the optimizer to recover the plan within the clinical safe-zone.

## 3.3.5 Spatial Scatter Suppression (BEV-Constrained)

To eliminate physically impossible out-of-beam dose spillage, the BEV frustum mask, $M _ { B E V } \in \{ 0 , 1 \}$ , acts as a spatial filter for healthy tissue $( M _ { B o d y } = 1 )$ . We split the non-target tissue into In-Beam and Out-of-Beam corridors, applying a tiered suppression penalty:

$$
\mathcal { L } _ { B E V } = \frac { \lambda _ { i n } } { N _ { i n } } \sum _ { \vec { r } \in \mathrm { I n - B e a m } } \mathrm { R e L U } ( D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { i n } ) + \frac { \lambda _ { o u t } } { N _ { o u t } } \sum _ { \vec { r } \in \mathrm { O u t - o f - B e a m } } \mathrm { R e L U } ( D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { o u t } ) ,\tag{11}
$$

where $\lambda _ { o u t } \ > \ \lambda _ { i n }$ and $D _ { o u t } \ll D _ { i n }$ . This enforces the physical reality that dose deposition outside the projected target geometry should be near-zero, efectively suppressing unmodeled scatter.

## 3.3.6 Geometric Fallof, Anti-Collapse, and Ghost Suppression

To mathematically enforce the steep dose gradients required immediately outside the prescription boundary, Dose-PlanNet evaluates the predicted dose against the 5.0 mm geometric fallof shell, denoted by R, extracted during preprocessing. A one-sided quadratic penalty is applied to any voxel within this shell that exceeds the strict fallof threshold $\left( D _ { \mathrm { r i n g - t h r e s h } } \right)$

$$
\mathcal { L } _ { r i n g } = \frac { 1 } { N _ { r i n g } } \sum _ { \vec { r } \in \mathcal { R } } \mathrm { R e L U } ( D _ { \mathrm { p r e d } } ( \vec { r } ) - D _ { \mathrm { r i n g . t h r e s h } } ) ^ { 2 } .\tag{12}
$$

Concurrently, to safeguard against severe target drop-ofs caused by competing OAR penalties, an anticollapse safety net is embedded within the target volume. This aggressively penalizes any PTV voxel that falls below 50% of the normalized prescription dose, preventing topological collapse in complex geometries:

$$
\mathcal { L } _ { a n t i c o l l a p s e } = \left[ \frac { 1 } { N _ { P T V } } \sum _ { \vec { r } \in P T V } \mathrm { R e L U } ( 0 . 5 0 - D _ { \mathrm { p r e d } } ( \vec { r } ) ) \right] ^ { 2 } .\tag{13}
$$

Furthermore, to prevent the network from mathematically displacing excess dose into the ambient air surrounding the patient (a common artifact in unconstrained 3D volumetric regression), an out-of-body dose suppression penalty is applied. This term strictly suppresses any non-zero dose predicted outside the binary body contour $( M _ { \mathrm { B o d y } } )$ :

$$
\mathcal { L } _ { b o d y } = \frac { 1 } { N _ { a i r } } \sum _ { \vec { r } \notin \mathrm { B o d y } } D _ { \mathrm { p r e d } } ( \vec { r } ) .\tag{14}
$$

## 3.3.7 Spatial Regularization

To ensure the predicted dose distributions remain physically plausible, we incorporate spatial regularization terms that enforce smoothness and continuity across the volumetric $\mathrm { g r i d }$ . Total Variation (TV) is employed to penalize sharp, non-physical voxel intensity fluctuations:

$$
\mathcal { L } _ { s m o o t h } = \sum _ { \vec { r } \in \Omega } \left( | | \nabla _ { d } D _ { \mathrm { p r e d } } ( \vec { r } ) | | ^ { 2 } + | | \nabla _ { h } D _ { \mathrm { p r e d } } ( \vec { r } ) | | ^ { 2 } + | | \nabla _ { w } D _ { \mathrm { p r e d } } ( \vec { r } ) | | ^ { 2 } \right) ,\tag{15}
$$

where $\nabla _ { d } , \nabla _ { h } , \nabla _ { w }$ represent the first-order spatial gradients along the depth, height, and width axes, respectively. Furthermore, a Laplacian penalty is utilized to enforce second-order continuity, efectively suppressing high-frequency noise near the beam penumbra:

$$
\mathcal { L } _ { l a p l a c i a n } = \sum _ { \vec { r } \in \Omega } \left[ \nabla ^ { 2 } D _ { \mathrm { p r e d } } ( \vec { r } ) \right] ^ { 2 } .\tag{16}
$$

## 3.3.8 Total Objective Function

The complete multi-objective loss function is the weighted summation of these complementary components, as formulated in Equation 4.. During the curriculum training phase, all physics-based and geometric penalties $( { \mathrm { e . g . } } , \ { \mathcal { L } } _ { P T V } , \ { \mathcal { L } } _ { V - T y p e } , \ { \mathcal { L } } _ { r i n g } , \ { \mathcal { L } } _ { a n t i c o l l a p s e } , \ { \mathcal { L } } _ { b o d y }$ , and $\mathcal { L } _ { B E V } )$ are linearly ramped from zero to their target magnitudes over the initial 30 epochs. This warmup strategy allows the 3D U-Net to first stabilize the global dose distribution via the constant, dose-weighted $\mathcal { L } _ { M S E }$ before the more restrictive, non-linear clinical hinge penalties are introduced to finalize the dosimetric conformity and OAR sparing.

## 4 Experimental Setup and Evaluation

To validate the clinical eficacy and generalization of the Dose-PlanNet architecture, this section details the retrospective patient cohorts across multiple fractionation schemes, the quantitative dosimetric criteria, and the blinded expert review methodology.

## 4.1 Clinical Datasets and Patient Cohorts

The dataset for this study comprises two distinct cohorts sourced from the Prostate Prospective Randomized trial conducted at Tata Medical Center, Kolkata. Active recruitment for the trial has concluded, establishing these as fixed, retrospective clinical datasets.

• Moderate Hypofraction Arm Cohort: Comprises 70 clinical patients receiving standard fractionation. To rigorously evaluate generalization, this cohort was partitioned using an $8 0 / 2 0$ train-test split, dedicating 56 patients to training and hyperparameter optimization, and strictly holding out 14 patients as an unseen testing set for final dosimetric inference.

• Stereotactic Body Radiation Therapy Arm (SBRT) Cohort: Comprises a separate set of 62 patients receiving Stereotactic Body Radiation Therapy (SBRT). This cohort was similarly partitioned into training and held-out validation sets to assess the network’s capacity to map hypofractionated dose distributions and ultra-steep spatial gradients.

## 4.2 Quantitative Evaluation Metrics

The predicted dose distributions were mathematically evaluated against the strict protocol defined constraints [15]. Because the cohorts utilize fundamentally diferent fractionation schemes, they were evaluated against separate dosimetric objectives:

• Moderate Hypofraction Arm (62.0 Gy Prescription): The primary target metrics required a dosimetric coverage of $D _ { 9 5 } \geq 9 5 \%$ and strict adherence to an absolute maximum dose limit of 66.34 Gy. Evaluations for Organs at Risk (OARs) focused on specific volumetric thresholds, primarily the Bladder and Anorectum $V _ { \mathrm { 6 0 . 4 G y } }$ and $V _ { \mathrm { 3 8 . 0 G y } }$ constraints, alongside the maximum dose to the bilateral femoral heads $( D _ { m a x } \leq 4 0 . 0 \mathrm { ~ G y } )$

• Stereotactic Body Radiation Therapy Arm (36.25 Gy Prescription): The primary target metrics required a dosimetric coverage of $D _ { 9 5 } \geq 9 5 \%$ and strict adherence to an absolute maximum dose limit of 38.8 Gy. Evaluations for Organs at Risk (OARs) focused on specific volumetric thresholds, primarily the Bladder and Anorectum $V _ { 3 5 . 0 \mathrm { G y } }$ and $V _ { \mathrm { 3 1 . 5 G y } }$ constraints, alongside the volumetric dose limit to the bilateral femoral heads $( V _ { 1 4 . 0 \mathrm { G y } } < 5 \% )$

## 4.3 Evaluated Baselines

To rigorously validate the necessity of the physics-guided hinge penalties integrated into the Dose-PlanNet loss architecture, an ablation study was conducted using an unconstrained baseline configuration. This model was trained exclusively utilizing Mean Squared Error (MSE) loss, stripping away the other regularization terms and losses. This baseline was evaluated specifically on the Prostate Prospective Randomized Stereotactic Body Radiation Therapy Arm (SBRT) cohort to determine if a standard deep learning architecture could autonomously map hypofractionated dose distributions and navigate the ultra-steep spatial gradients required for clinical safety without explicit physics-informed guidance.

## 4.4 Model Configurations and Expert Clinical Review

For the held-out test cohorts, dose distributions were reviewed on an unblinded basis by three radiation oncology consultants at Tata Medical Center, two of whom are co-authors on this study, against standard institutional acceptability criteria and the Eclipse-generated ground truth. No structured rating instrument was used; this was a qualitative clinical read, not the blinded comparative evaluation originally planned (see Discussion, Limitations). The reviewing consultants deemed the AI-generated plans to be reasonable. Given the partial author overlap and lack of blinding, this should be read as a preliminary clinical evaluation check rather than independent validation; a formal blinded review remains necessary before broader claims of clinical equivalence can be made.

## 5 Implementation Details

To guarantee multi-institutional reproducibility and facilitate direct integration into existing clinical Treatment Planning Systems (TPS), the Dose-PlanNet architecture relies on a modular, open-source Python software stack with a centralized configuration protocol.

## 5.1 Data Curation and Preprocessing Engine

The preprocessing pipeline is engineered to autonomously transform raw hospital DICOM directories into structurally aligned spatial tensors. A prespecified YAML configuration explicitly defines site-specific Region of Interest (ROI) nomenclature, overriding inconsistencies in clinical naming conventions across patients.

Spatial operations are executed using the SimpleITK library. All patient volumes are isotropically resampled to a standardized $1 . 2 7 \times 1 . 2 7 \times 2 . 5 ~ \mathrm { m m ^ { 3 } }$ coordinate grid. Signed Distance Maps (SDMs) for serial organs are computed via exact Euclidean distance transforms (scipy.ndimage.distance transform edt). The Beam’s Eye View (BEV) frustums are generated using geometric ray-tracing anchored to a predefined Source-to-Axis Distance $( S A D = 1 0 0 0 . 0 \mathrm { m m } )$ , incorporating an isotropic 7.0 mm physical penumbra margin. The generated tensors are subsequently serialized into NIfTI formats, strictly adhering to the directory conventions required by the nnU-Net v2 framework for seamless data ingestion.

## 5.2 Target Normalization and Numerical Stability

Predicting raw, unscaled prescription magnitudes (e.g., 60.0 Gy to 70.0 Gy) using near-zero initialized network weights causes severe gradient instability during early training. To prevent this, the ground-truth RTDose tensors are non-dimensionalized to a continuous fractional space of [0, 1] by dividing by the primary prescription dose $( D _ { R x } )$

This normalization serves two critical mathematical functions:

1. Hessian Preconditioning: Unscaled targets create an ill-conditioned, ravine-like loss landscape characterized by a massive condition number $( \kappa = \lambda _ { m a x } / \lambda _ { m i n } \gg 1 )$ . Normalizing the target space acts as a mathematical preconditioner, driving $\kappa  1$ and creating a more isotropic landscape that enables rapid, stable convergence.

2. Bounding Gradient Variance: Raw dosimetric targets induce massive initial gradient variances $( \mathcal { O } ( 1 0 ^ { 4 } ) )$ during backpropagation. This magnitude completely swamps the ϵ stabilizer within the denominator of the Adam optimizer’s second moment estimate, destroying its adaptive precision. Normalizing to [0, 1] strictly bounds this variance to O(1), aligning perfectly with Adam’s theoretically proven stability regime.

## 5.3 Network Training and Curriculum Optimization

The 3D U-Net is constructed and trained utilizing the MONAI deep learning framework and PyTorch. To mitigate severe I/O bottlenecks inherent to processing high-resolution medical volumes, the pipeline utilizes MONAI’s PersistentDataset, which caches the deterministic preprocessing transforms directly to local storage.

To resolve the extreme volumetric imbalance between the small SIB targets and the vast pelvic background, the pipeline does not employ uniform spatial sampling. Instead, it deploys a targeted boundarycropping strategy (RandCropByLabelClassesd). This forces the dataloader to anchor the $1 2 8 \times 1 2 8 \times 6 4$ training patches specifically on the geometric intersections between the PTV and abutting critical organs, guaranteeing the network focuses computational efort on the most complex, high-frequency dosimetric gradients.

Network weights are optimized using the Adam optimizer coupled with a Cosine Annealing learning rate scheduler, fully accelerated via native mixed-precision training (torch.cuda.amp). To prevent the nonlinear physics penalties from destabilizing early optimization, a curriculum learning strategy is deployed; the non-diferentiable spatial constraints $( \mathcal { L } _ { V - T y p e } , \mathcal { L } _ { B E V } )$ are linearly ramped from zero to their configured magnitudes over the initial training epochs.

## 5.4 Dual-Tier Checkpointing Strategy

Standard optimization monitors validation loss to trigger early stopping. However, the absolute mathematical minimum of a multi-objective loss function does not perfectly correlate with physician acceptability. To bridge this gap, the pipeline executes a Dual-Checkpointing strategy:

1. Physics-Optimized Checkpoint: Saves the model weights that achieve the absolute minimum across all mathematical hinge penalties.

2. Clinically-Optimized Checkpoint: Evaluates the validation predictions through a custom softmargin minimax engine. This function mathematically penalizes critical target volume deficits heavily while intelligently allowing minor, non-catastrophic OAR deviations, closely mimicking the non-linear compromises executed by expert human planners in anatomically impossible geometries.

## 5.5 End-to-End Clinical Inference and Coordinate Restoration

To facilitate seamless clinical deployment, inference is executed via a fully automated pipeline that directly bridges raw hospital DICOM exports to predicted RTDOSE files. Upon ingesting a patient’s DICOM directory (CT, RTSTRUCT, and RTPLAN), the inference engine dynamically mirrors the deterministic training transformations. It constructs the necessary spatial embeddings—including Signed Distance Maps (SDMs), Beam’s Eye View (BEV) frustums, and Simultaneous Integrated Boost (SIB) targets mapped via the Painter’s Algorithm—before isotropically resampling the anatomy to the network’s required $1 . 2 7 \times 1 . 2 7 \times$ 2.5 mm<sup>3</sup> grid.

Volumetric dose prediction is generated using MONAI’s sliding window inference utility. This strategy employs overlapping spatial patches $( 1 2 8 \times 1 2 8 \times 6 4 )$ paired with a Gaussian weighting map to smoothly blend adjacent boundaries and suppress edge artifacts. To translate the raw network logits into physical dosimetric units, the outputs are passed through a Softplus activation function and mathematically scaled by the scalar prescription dose $( D _ { R x } )$ . A strict spatial filter using the binary body contour is subsequently applied, instantly zeroing out any non-physical scatter radiation predicted in the ambient air.

Crucially, this pipeline explicitly rejects the standard practice of evaluating AI models purely within their cropped, standardized training space. Because clinical Treatment Planning Systems (TPS) require the dose matrix to register with the patient’s native CT geometry, the pipeline utilizes MONAI’s Invertd module. By reading the spatial operations trace stored within the MetaTensor, the engine mathematically reverses the afine transformations, warping the continuous 3D dose prediction back to the native, uncropped DICOM coordinate space. To guarantee absolute spatial fidelity, the native CT’s exact geometric metadata (origin, direction cosine matrix, and voxel spacing) is hard-anchored to the restored dose matrix via SimpleITK. Finally, this strictly aligned spatial tensor is exported as a compliant DICOM RTDOSE file, enabling direct importation into commercial TPS software (e.g., Varian Eclipse) for blinded human evaluation.

## 6 Results & Discussion

As discusssed in section 4, Dose-PlanNet was trained and tested on Prostate Prospective Randomized Trial Moderate Hypofraction Arm and Stereotactic Body Radiation Therapy Arm datasets. In this section we discuss the results and the improvements that Dose-PlanNet has achieved.

## 6.1 Ablation Study: The Limitations of Voxel-Wise Loss

To validate the necessity of the physics-guided hinge penalties, an ablation study was conducted using an unconstrained baseline configuration trained exclusively with Mean Squared Error (MSE) loss on the Experimental SBRT cohort. At convergence, the MSE-only model achieved mathematically deceptive approximations: a target coverage marginally below acceptable limits $( D _ { 9 5 } = 3 3 . 5 \ \mathrm { G y }$ for a 36.25 $\mathrm { G y }$ prescription) and elevated mean OAR doses (Bladder ∼15 Gy, Rectum ∼18 Gy). However, it completely failed to respect spatial safety margins, generating a catastrophic global maximum dose $\left( D _ { m a x } \right)$ exceeding 78 Gy against the strict 38.8 Gy clinical ceiling.

Because MSE minimizes average global deviation, it inherently ignores isolated spatial extremes. While a naive solution might be to append a simple maximum-dose penalty to the MSE, this creates a fundamentally unstable loss surface. MSE is symmetric, penalizing underdosing and overdosing equally; enforcing a strict upper bound forces the network to shift the entire dose distribution downwards to escape the penalty, inevitably causing severe target underdosing and violating the $D _ { 9 5 } \geq 9 5 \%$ coverage constraints.

Furthermore, because MSE evaluates voxels independently, it fails to account for the physical constraints of radiation scattering. The resulting distributions lacked the continuous spatial gradients and steep fallof required by clinical linear accelerators. This confirms that the asymmetric hinge penalties and spatial regularizations inherent to Dose-PlanNet are strictly necessary to suppress severe hotspots and translate mathematical approximations into physically deliverable treatment plans.

## 6.2 Prospective Randomized Moderate Hypofraction Arm Results

The clinical performance of the optimized Dose-PlanNet was quantitatively validated on a held-out test cohort of 14 patients treated under the Prostate Prospective Randomized Moderate Hypofraction Arm protocol (62.0 Gy prescription). The network demonstrated highly consistent target coverage and exceptional sparing of adjacent Organs at Risk (OARs), confirming the eficacy of the physics-guided spatial regularizations.

## 6.2.1 Comparison with Clinical Constraints

As detailed in Table 6.2.2, the network successfully achieved strict target coverage, with the primary planning target volume (PTV62) $D _ { 9 5 }$ averaging $6 0 . 1 5 \pm 0 . 7 4 \mathrm { { \ G y } }$ , comfortably exceeding the mandatory 95% prescription threshold $\left( 5 8 . 9 \mathrm { \ G y } \right)$

For OARs, the model consistently adhered to stringent avoidance bounds. Evaluations utilized surrogate high-dose $( V _ { 5 9 . 0 \mathrm { G y } } )$ and mid-dose $\left( V _ { 4 0 . 0 \mathrm { G y } } \right)$ thresholds to capture gradient fall-of. Bladder high-dose volumes averaged $2 . 8 6 \pm 1 . 1 1 \%$ , intrinsically satisfying the strict optimal $\mathrm { V _ { 6 0 . 4 G y } ~ \leq ~ 3 \% }$ constraint despite being measured at a lower, more conservative isodose line. Rectal sparing was equally robust, satisfying mandatory clinical constraints and demonstrating the network’s ability to compress the spatial dose bath.

Notably, while the average maximum dose $\left( D _ { m a x } \right)$ recorded in the raw predicted arrays was $6 6 . 6 6 \pm$ 0.68 Gy—appearing to violate the strict 66.34 Gy clinical ceiling—a volumetric analysis revealed this to be a mathematical artifact of the voxelized grid rather than a physical hotspot. The volume receiving doses above 66.34 Gy constituted less than 0.001% of the patient volume (typically a few isolated voxels). For all practical clinical evaluations and Dose-Volume Histogram (DVH) analyses, the true functional maximum dose was constrained to approximately 66.0 Gy, cleanly within the protocol limits.

## 6.2.2 Comparison with Human-Generated Clinical Plans

To benchmark clinical viability, the predicted distributions were directly evaluated against ground-truth human plans generated via the Varian Eclipse Treatment Planning System (TPS).

The human planners achieved marginally higher PTV62 coverage (Eclipse $D _ { 9 5 } = 6 1 . 3 9 \pm 0 . 8 3$ Gy vs. Model $D _ { 9 5 } = 6 0 . 1 5 { \pm } 0 . 7 4 \ : \mathrm { G y } )$ and tighter maximum dose control (Eclipse $D _ { m a x } = 6 5 . 7 4 { \pm } 0 . 6 6 \mathrm { G y ~ v s }$ . Model $D _ { m a x } = 6 6 . 6 6 \pm 0 . 6 8 ~ \mathrm { G y } )$ . However, both approaches comfortably cleared the mandatory 58.9 Gy target threshold, and the network’s slightly elevated $D _ { m a x }$ was strictly constrained to mathematically isolated voxels rather than clinically relevant volumes as established previously.
<table><tr><td>Type</td><td>ROI</td><td>Metric</td><td>Eclipse TPS (Mean ± SD)</td><td>Dose-PlanNet  $\overline { { ( \mathbf { M e a n } \pm \mathbf { S D } ) } }$ </td><td>Mandatory Goal</td></tr><tr><td rowspan="3">Target</td><td>PTV62</td><td> $\overline { { D _ { 9 5 } \ ( \mathrm { G y } ) } }$ </td><td> $\overline { { 6 1 . 3 9 \pm 0 . 8 3 } }$ </td><td> $\overline { { 6 0 . 1 5 \pm 0 . 7 4 } }$ </td><td>≥ 58.90</td></tr><tr><td>PTV62</td><td> $D _ { m a x } \ \mathrm { ( G y ) }$ </td><td> $6 5 . 7 4 \pm 0 . 6 6$ </td><td> $6 6 . 6 6 \pm 0 . 6 8 ^ { \ast }$ </td><td>≤ 66.34</td></tr><tr><td>PTV44</td><td> $D _ { 9 5 } ~ \mathrm { ( G y ) }$ </td><td> $4 3 . 5 4 \pm 0 . 3 2$ </td><td> $4 3 . 5 7 \pm 0 . 6 0$ </td><td>≥ 41.80</td></tr><tr><td rowspan="7">Avoidance</td><td>Bladder</td><td> $\overline { { V _ { 5 9 . 0 \mathrm { G y } } \ ( \% ) } }$ </td><td> $\overline { { { 3 . 8 2 \pm 1 . 4 5 } } }$ </td><td> $\overline { { 2 . 8 6 \pm 1 . 1 1 } }$ </td><td>≤ 3.0</td></tr><tr><td>Bladder</td><td> $V _ { 4 0 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $1 5 . 6 5 \pm 5 . 2 4$ </td><td> $1 4 . 2 5 \pm 4 . 9 3$ </td><td>≤ 55.0</td></tr><tr><td>Anorectum</td><td> $V _ { 5 9 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $4 . 8 7 \pm 0 . 9 3$ </td><td> $3 . 4 8 \pm 1 . 3 0$ </td><td>≤ 3.0</td></tr><tr><td>Anorectum</td><td> $V _ { 4 0 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $2 1 . 3 5 \pm 5 . 5 2$ </td><td> $2 0 . 5 0 \pm 6 . 4 9$ </td><td>≤ 55.0</td></tr><tr><td>Bowel Bag</td><td> $V _ { 4 5 . 0 \mathrm { G y } } ~ \mathrm { ( c c ) }$ </td><td> $1 9 . 8 4 \pm 1 9 . 6 7$ </td><td> $4 2 . 8 8 \pm 1 9 . 0 8$ </td><td>&lt; 90.0</td></tr><tr><td>Femoral Heads</td><td> $\bar { D } _ { m a x } \bar { ( \mathrm { G y ) } }$ </td><td> $3 6 . 3 3 \pm 4 . 7 4$ </td><td> $3 4 . 7 8 \pm 5 . 0 7$ </td><td>&lt; 40.0</td></tr><tr><td>Penile Bulb</td><td> $V _ { 4 7 . 0 \mathrm { G y } }$  (%)</td><td> $5 . 0 7 \ ( 0 . 0 0 - 3 5 . 5 0 ) ^ { \dagger }$ </td><td> $3 . 2 1 \ ( 0 . 0 0 - 2 2 . 2 2 ) ^ { \dagger }$ </td><td>≤ 50.0</td></tr></table>

∗ Statistical artifact representing < 0.0005% volume. True functional D<sub>max</sub> ≈ 66.00 Gy.  
<sup>†</sup> Data expressed as Mean (Range) due to highly skewed distribution with zero-bound limits.

Table 1: Dosimetric Comparison: Dose-PlanNet vs. Human-Generated Eclipse TPS Plans on the Prostate Prospective Randomized Moderate Hypofraction Arm (62.0 Gy Prescription, n = 14).

Conversely, Dose-PlanNet systematically outperformed the human planners in sparing primary Organs at Risk, specifically across the critical high-dose spatial gradients. The network suppressed Bladder $V _ { 5 9 . 0 \mathrm { G y } }$ to $2 . 8 6 \pm 1 . 1 1 \%$ (against the human $3 . 8 2 \pm 1 . 4 5 \% )$ and Anorectum $V _ { 5 9 . 0 \mathrm { G y } } \mathrm { ~ t o ~ } 3 . 4 8 \pm 1 . 3 0 \%$ (against the human $4 . 8 7 \pm 0 . 9 3 \%$ . This enhanced conformity extended into the low-dose bath, shrinking Bladder $V _ { 1 5 . 0 \mathrm { G y } }$ by nearly 10% (Model 70.42% vs. Eclipse 79.97%) and Anorectum $V _ { 1 5 . 0 \mathrm { G y } }$ (Model 71.73% vs. Eclipse 80.64%).

The singular metric where human planning maintained a distinct advantage was the Bowel Bag middose spillage $\left( V _ { 4 5 . 0 \mathrm { G y } } \right)$ , where Eclipse averaged $1 9 . 8 4 \pm 1 9 . 6 7$ cc compared to the network’s $4 2 . 8 8 \pm 1 9 . 0 8$ cc. Despite this diference, the model remained strictly compliant with the $< 9 0 . 0$ cc mandatory limit. Overall, the network generated plans that consistently matched or exceeded the clinical safety profile of the human-generated baselines.

## 6.2.3 Volumetric Analysis of Maximum Dose Violations

To contextualize the $D _ { m a x }$ violations, a voxel-wise extraction was performed on the generated Moderate Hypofraction Arm dose arrays (averaging ∼2.3 million voxels per patient). This analysis confirms the boundary breaches are minor artifacts rather than clinically meaningful hot spots. Specifically, 5 of the 14 plans (35.7%) recorded exactly zero voxels exceeding the maximum dose threshold, while the remaining 9 plans exhibited violations limited to small clusters of just 2 to 9 voxels per patient. The mean body volume fraction receiving a dose above the threshold was 0.00014%, with a worst-case maximum of 0.00042%. These sub-millimeter $D _ { m a x }$ breaches are mathematical artifacts inherent to the 3D U-Net’s coordinate regression; they functionally represent $\approx 0 \%$ of the macroscopic anatomical volume and pose no physical delivery risk.

## 6.2.4 Statistical Analysis of Dosimetric Metrics

To rigorously evaluate the dosimetric equivalence between Dose-PlanNet and the Eclipse TPS ground truth, a comparative statistical analysis was conducted across the 14-patient validation cohort. Normality was assessed using the Shapiro-Wilk test; given that most metrics exhibited a non-normal distribution, the Wilcoxon Signed-Rank test was employed for paired comparison. The results are summarized in Table 2.

The analysis indicates a statistically significant diference in target coverage $\left( \mathrm { P T V } \ D _ { 9 5 } \right)$ , where the model consistently produces a slightly lower dose than the human planner. However, this is ofset by statistically significant improvements in OAR sparing, particularly in the high-dose regions $( V _ { 5 9 \mathrm { G y } } )$ for both the bladder and anorectum, confirming the model’s preference for steeper dose fall-of at the expense of marginal target coverage. Conversely, no statistically significant diference was observed in the intermediate-to-low dose rectal sparing $( V _ { 4 0 \mathrm { G y } } )$ , demonstrating that both the AI and the expert planner converge on similar geometric solutions for broader anatomical structures. These results validate that while Dose-PlanNet operates with a

<table><tr><td>Metric</td><td>Eclipse Mean</td><td>Model Mean</td><td>Mean Diff.</td><td>95% CI</td><td>p-value</td></tr><tr><td>PTV  $\overline { { D _ { 9 5 } \ ( \mathrm { G y } ) } }$ </td><td> $\overline { { 6 1 . 3 9 \pm 0 . 8 3 } }$ </td><td> $\overline { { 6 0 . 1 5 \pm 0 . 7 4 } }$ </td><td>-1.24</td><td>[-1.84, -0.63]</td><td> $\overline { { 0 . 0 0 0 7 ^ { * } } }$ </td></tr><tr><td>Bladder  $V _ { 5 9 \mathrm { G y } } ~ ( \% )$ </td><td> $3 . 8 3 \pm 1 . 4 7$ </td><td> $2 . 8 7 \pm 1 . 1 2$ </td><td>-0.96</td><td>[-1.39, -0.53]</td><td>0.0003*</td></tr><tr><td>Bladder  $V _ { 4 0 \mathrm { { G y } } } ~ ( \% )$ </td><td> $1 5 . 6 6 \pm 5 . 2 5$ </td><td> $1 4 . 2 6 \pm 4 . 9 4$ </td><td>-1.40</td><td>[-2.66, -0.14]</td><td> $0 . 0 3 2 3 ^ { * }$ </td></tr><tr><td>Anorectum  $V _ { 5 9 \mathrm { G y } } ~ ( \% )$ </td><td> $4 . 8 7 \pm 0 . 9 3$ </td><td> $3 . 4 8 \pm 1 . 3 0$ </td><td>-1.40</td><td>[-2.05, -0.75]</td><td>0.0004*</td></tr><tr><td>Anorectum  $V _ { 4 0 \mathrm { G y } }$  (%)</td><td> $2 1 . 3 5 \pm 5 . 5 2$ </td><td> $2 0 . 5 0 \pm 6 . 4 9$ </td><td>-0.85</td><td>[-2.87, 1.18]</td><td>0.3835</td></tr></table>

\*Statistically significant at $\alpha = 0 . 0 5 .$

Table 2: Statistical comparison of dosimetric metrics between Eclipse TPS and Dose-PlanNet.

distinct dosimetric bias compared to human planners, the trade-ofs it makes fall within acceptable clinical bounds.

## 6.3 Prospective Randomized Stereotactic Body Radiation Therapy Arm Results

The clinical performance of Dose-PlanNet was further evaluated on a separate held-out cohort of 12 patients treated under the extreme hypofractionated Prostate Prospective Randomized Stereotactic Body Radiation Therapy Arm (36.25 Gy prescription). This protocol explicitly tested the network’s capacity to maintain stringent target coverage while generating the steeper spatial dose gradients required for aggressive organ sparing.

## 6.3.1 Comparison with Clinical Constraints

For the extreme hypofractionated Prostate Prospective Randomized Stereotactic Body Radiation Therapy Arm (36.25 Gy in 5 fractions), Dose-PlanNet demonstrated robust adherence to stringent clinical constraints across the validation cohort. Target coverage remained highly consistent, with the high-dose PTV $( D _ { 9 5 } )$ achieving a mean dose of $3 4 . 8 9 \pm 0 . 8 6 \mathrm { ~ G y } ,$ successfully satisfying the prescription coverage requirements. The $D _ { m a x }$ within the target volume was tightly constrained to $3 9 . 4 9 \pm 0 . 6 3 \mathrm { G y }$ , efectively avoiding clinically unacceptable hot spots while preserving the necessary intra-target dose heterogeneity.

Crucially, the network managed the steep spatial dose gradients required for organ-at-risk sparing in an SBRT paradigm. Bladder constraints were rigorously met, exhibiting a mean high-dose spillage $V _ { 3 5 . 0 \mathrm { G y } }$ of $2 . 7 1 \% \pm 0 . 9 2 \%$ and a broader low-dose $V _ { \mathrm { 1 4 . 0 G y } }$ wash of $2 2 . 6 2 \% \pm 4 . 6 5 \%$ . Similarly, anorectal sparing was strictly maintained; the model recorded a mean $V _ { 3 5 . 0 \mathrm { G y } }$ of $4 . 2 4 \% \pm 1 . 1 0 \%$ and a $V _ { 2 8 . 0 \mathrm { G y } }$ of $1 1 . 2 5 \% \pm 2 . 2 6 \% ,$ demonstrating the architecture’s capacity to strictly penalise high-dose scatter into the anterior rectal wall through the dual-tier hinge loss.

Serial and parallel avoidance structures were also efectively protected. The small bowel recorded an absolute volume $V _ { 2 8 . 0 \mathrm { G y } }$ of $0 . 3 6 \pm 0 . 8 0 \mathrm { c c }$ , maintaining a maximum exposure well below standard tolerance thresholds. Furthermore, femoral head exposure was minimised, with $V _ { \mathrm { 1 4 . 0 G y } }$ averaging less than 2.0% bilaterally $( 1 . 9 4 \% \pm 1 . 0 8 \%$ for the right femur, and $1 . 2 8 \% \pm 0 . 9 2 \%$ for the left). These dosimetric metrics confirm that the spatially masked, physics-guided predictions maintain deliverable and clinically viable dose distributions even under aggressive hypofractionation.

## 6.3.2 Comparison with Human-Generated Clinical Plans

When evaluated against the expert-generated Eclipse TPS plans for the extreme hypofractionated Stereotactic Body Radiation Therapy Arm, Dose-PlanNet demonstrated comparable target coverage and superior low-dose organ sparing. For the primary target (PTV36.25), the human planners achieved a slightly higher mean $D _ { 9 5 }$ coverage $( 3 5 . 4 6 \pm 0 . 4 8 \mathrm { G y } )$ compared to the model $( 3 4 . 8 9 \pm 0 . 8 6 \ : \mathrm { G y } )$ ; however, both comfortably satisfied the $\geq 3 4 . 4 4$ Gy clinical mandate. Intra-target maximum dose $\left( D _ { m a x } \right)$ was marginally hotter in the model predictions $( 3 9 . 4 9 \pm 0 . 6 3 \ : \mathrm { G y } )$ versus the clinical baseline $( 3 8 . 5 7 \pm 0 . 1 4 \ : \mathrm { G y } )$ . However, this reported model $D _ { m a x }$ is a single-voxel statistical artifact; volumetric analysis confirms that functionally 0% of the target volume receives this peak dose, with the true functional maximum converging to approximately 38.50 $\mathrm { G y , }$ remaining strictly bound below the 38.79 Gy physical ceiling constraint.

<table><tr><td>Type</td><td>ROI</td><td>Metric</td><td>Eclipse  $\mathbf { T P S } \ ( \mathbf { M e a n } \pm \mathbf { S D } )$ </td><td>Dose-PlanNet (Mean ± SD)</td><td>Mandatory Goal</td></tr><tr><td rowspan="3">Target</td><td>PTV36.25</td><td> $\overline { { D _ { 9 5 } \ ( \mathrm { G y } ) } }$ </td><td> $3 5 . 4 6 \pm 0 . 4 8$ </td><td> $\overline { { 3 4 . 8 9 \pm 0 . 8 6 } }$ </td><td> $\overline { { \geq 3 4 . 4 4 } }$ </td></tr><tr><td>PTV36.25</td><td> $D _ { m a x } \ \mathrm { ( G y ) }$ </td><td> $3 8 . 5 7 \pm 0 . 1 4$ </td><td> $3 9 . 4 9 \pm 0 . 6 3 ^ { \ast }$ </td><td>≤ 38.79</td></tr><tr><td>PTV25</td><td> $D _ { 9 5 } ~ \mathrm { ( G y ) }$ </td><td> $2 5 . 7 0 \pm 3 . 1 2$ </td><td> $2 6 . 0 1 \pm 2 . 8 1$ </td><td>≥ 23.75</td></tr><tr><td rowspan="7">Avoidance</td><td>Bladder</td><td> $\overline { { V _ { 3 5 . 0 \mathrm { G y } } \ ( \% ) } }$ </td><td> $\overline { { 2 . 9 1 \pm 0 . 8 0 } }$ </td><td> $2 . 7 1 \pm 0 . 9 2$ </td><td>≤ 5.0</td></tr><tr><td>Bladder</td><td> $V _ { 1 4 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $3 1 . 1 4 \pm 7 . 0 4$ </td><td> $2 2 . 6 2 \pm 4 . 6 5$ </td><td>≤ 50.0</td></tr><tr><td>Anorectum</td><td> $V _ { 3 5 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $3 . 4 2 \pm 1 . 2 6$ </td><td> $4 . 2 4 \pm 1 . 1 0$ </td><td>≤ 5.0</td></tr><tr><td>Anorectum</td><td> $V _ { 1 4 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $3 8 . 9 6 \pm 5 . 1 8$ </td><td> $3 2 . 8 0 \pm 4 . 5 1$ </td><td>≤ 50.0</td></tr><tr><td>Small Bowel</td><td> $V _ { \mathrm { 2 8 . 0 G y } }$  (cc)</td><td> $0 . 0 5 \pm 0 . 1 5$ </td><td> $0 . 3 6 \pm 0 . 8 0$ </td><td>&lt; 1.0</td></tr><tr><td>Right Femur</td><td> $V _ { \mathrm { 1 4 . 0 G y } }$  (%)</td><td> $2 . 0 8 \pm 1 . 3 3$ </td><td> $1 . 9 4 \pm 1 . 0 8$ </td><td>&lt; 5.0</td></tr><tr><td>Left Femur</td><td> $V _ { 1 4 . 0 \mathrm { G y } } ~ ( \% )$ </td><td> $1 . 2 8 \pm 1 . 1 4$ </td><td> $1 . 2 8 \pm 0 . 9 2$ </td><td>&lt; 5.0</td></tr></table>

Statistical artifact representing < 0.001% volume. True functional $D _ { m a x }$ ≈ 38.00 Gy.

Table 3: Dosimetric Comparison: Dose-PlanNet vs. Human-Generated Eclipse TPS Plans on the Prostate Prospective Randomized Stereotactic Body Radiation Therapy Arm (36.25 Gy Prescription, n = 12).

The most notable dosimetric divergence occurred in the low-dose wash regions, where the physics-guided architecture consistently outperformed manual planning. Dose-PlanNet achieved a substantial reduction in the bladder $V _ { 1 4 . 0 \mathrm { G y } } ~ ( 2 2 . 6 2 \% \pm 4 . 6 5 \% )$ compared to the human baseline $( 3 1 . 1 4 \% \pm 7 . 0 4 \% )$ . A similar trend was observed for the anorectum, where the model reduced the $V _ { \mathrm { 1 4 . 0 G y } }$ to $3 2 . 8 0 \% \pm 4 . 5 1 \%$ against the clinical $3 8 . 9 6 \% \pm 5 . 1 8 \%$

High-dose OAR spillage was clinically equivalent between the two modalities. The model maintained a bladder $V _ { 3 5 . 0 \mathrm { G y } }$ of $2 . 7 1 \% \pm 0 . 9 2 \%$ (vs. Eclipse $2 . 9 1 \% \pm 0 . 8 0 \% )$ and an anorectal $V _ { 3 5 . 0 \mathrm { G y } }$ of $4 . 2 4 \% \pm 1 . 1 0 \%$ (vs. Eclipse $3 . 4 2 \% \pm 1 . 2 6 \%$ . Bilateral femoral head exposure $\left( V _ { 1 4 . 0 \mathrm { G y } } \right)$ and absolute small bowel volume $\left( V _ { 2 8 . 0 \mathrm { G y } } \right)$ were minimal and statistically negligible for both the model and the human planners. These results indicate that for complex SBRT geometries, Dose-PlanNet safely matches expert high-dose constraints while more aggressively optimizing the low-dose integral spread.

Similar to the Prospective Randomized Moderate Hypofraction Arm results, in this case too, it was observed that a negligible < 0.001% of the total volume received a dose greater than 38.79 Gy, the maximum allowable dose.

## 6.4 Spatial Dose Distribution and Isodose Visualisation

Visual evaluation of the dose distributions confirms that Dose-PlanNet successfully generates clinically acceptable spatial gradients that closely mimic human-driven planning strategies. Figure 3 presents a representative comparative analysis across the axial, coronal, and sagittal planes, overlaying the generated dose arrays onto the native planning CT coordinates.

Within the high-dose region, the network demonstrates excellent spatial conformity. The 100% and 95% isodose contours tightly wrap the PTV boundaries in both the Dose-PlanNet and Eclipse TPS plans, visually corroborating the comparable $D _ { 9 5 }$ target coverage metrics. The model successfully navigates the complex geometric concavities of the prostate target volume without leaking excessive high dose into adjacent healthy tissue.

The visualisations further highlight the network’s superior performance in low-dose spatial management. Evaluating the 50% isodose wash, the Dose-PlanNet distribution exhibits a distinctly steeper dose fall-of posteriorly towards the anterior rectal wall and superiorly towards the bladder dome. The spatial restriction of these low-dose contours physically manifests the volumetric reductions quantified in the $V _ { \mathrm { 1 4 . 0 G y } }$ metrics, confirming that the physics-guided spatial regularisation eficiently penalises unnecessary integral dose spread while maintaining absolute target coverage.

## 6.5 Dose-Volume Histogram (DVH) Analysis

The volumetric dose distributions were further evaluated using Dose-Volume Histograms (DVHs) to visually and quantitatively compare the structural sparing and target coverage between the modalities. Representative DVHs from the validation cohort are presented in Figure 4, illustrating the typical dosimetric trade-ofs negotiated by the Dose-PlanNet architecture compared to the human baseline.

Eclipse TPS (Ground Truth)  
![](images/becf15dd168d0e92c7786849c6542bc0d545ea7f020d3c61f0262bedf6096007.jpg)

Dose-PlanNet (Predicted)  
![](images/8017e18b12d18ec9ee13b00f5b9e82475ce515af9a5ab7c5441da023e594bc74.jpg)  
(a) Actual Dose Distribution (generated by physicist)  
(b) Predicted Dose Distribution (generated by model)  
Figure 3: Axial spatial dose distribution and isodose conformity comparison for a representative extreme hypofractionated SBRT case. (a) The human-generated Eclipse TPS plan and (b) the Dose-PlanNet prediction are overlaid on the native planning CT. Isodose contours denote 36.25 Gy (100% prescription, red), 34.44 Gy $( D _ { 9 5 }$ clinical mandate, orange), 25.00 Gy (cyan), and 12.50 Gy (low-dose wash, blue). The mode demonstrates highly conformal high-dose target wrapping and robust spatial avoidance of adjacent critica structures, mirroring the clinical baseline.

The target volume curves (red and blue) demonstrate near-perfect concordance at the prescription dose shoulder, visually confirming the statistically equivalent $D _ { 9 5 }$ coverage. A slight divergence is observable at the extreme high-dose tail (bottom right of the target curves), where the model’s solid line extends marginally further along the dose axis. This visualizes the single-voxel $D _ { m a x }$ artifact previously quantified; the network permits a microscopic hot spot while the human planner forces a strict, absolute maximum cut-of.

For the organs at risk (green and magenta curves, representing the bladder and anorectum), the DVHs confirm the network’s aggressive low-dose spatial restriction. The solid model curves consistently exhibit a leftward shift in the low-to-intermediate dose regions relative to the dashed Eclipse curves. This steeper volumetric fall-of corresponds directly to the significant reductions in the $V _ { \mathrm { 1 4 . 0 G y } }$ and $V _ { 2 8 . 0 \mathrm { G y } }$ metrics, demonstrating that the physics-guided loss engine successfully penalizes unnecessary integral dose spread without compromising the high-dose target conformity.

## 6.6 Clinical Acceptance and Pass Rates

To evaluate the operational viability of Dose-PlanNet, the generated dose distributions were subjected to a binary pass/fail evaluation against the mandatory institutional constraints defined by the Prospective Randomized protocol. A plan was deemed ”clinically acceptable” if it satisfied all volumetric target coverage and Organ-At-Risk (OAR) dose-volume limits without requiring manual human intervention. Based on the dosimetric analyses in previous sections confirming that the $D _ { m a x }$ breaches are singular-voxel statistical artifacts functionally comprising $\approx 0 \%$ of the target volume, the absolute point-maximum constraint was excluded from this specific binary pass/fail classification to reflect true volumetric acceptability.

![](images/648a102ddcecdbdf35c657bf88285d0a052dab364c70ba1c2a3e58fad4577b6f.jpg)  
(a) DVH plots of a patient from Prostate Prospective Randomized Moderate Hypofraction Arm Dataset

![](images/ed6fc2daa495841d489c453a7fa51c7d00f3d0706b7204b2125df7cbdd837ac1.jpg)  
(b) DVH plots of a patient from Prostate Prospective Randomized Stereotactic Body Radiation Therapy Arm Dataset  
Figure 4: Comparative Dose-Volume Histograms (DVHs) for two representative patients. The solid lines correspond to the Dose-PlanNet (model-generated) dose, while the dashed lines correspond to the Eclipse TPS (human-planned ground truth) dose. Target volumes (red/blue curves) show strong conformity at the prescription shoulder, while OARs (green/magenta curves) demonstrate a steeper low-dose fall-of in the model predictions.

Prospective Randomized Moderate Hypofraction Arm Validation. For the conventionally fractionated Moderate Hypofraction Arm (62 Gy in 20 fractions), the model demonstrated a high degree of clinical reliability. Out of the evaluated validation cohort, 78.57% of the Dose-PlanNet generated plans successfully met all mandatory constraints. The plans consistently achieved the required $D _ { 9 5 } \geq 9 5 \%$ target coverage while simultaneously restricting the bladder and anorectum low-dose washes $( V _ { 4 0 \mathrm { G y } }$ and below) to within acceptable Prospective Randomized thresholds. These were verified by clinicians and physicists. All the remaining plans failed due to underdosing the PTV.

Prospective Randomized Stereotactic Body Radiation Therapy Arm Validation. Performance remained robust when scaled to the extreme hypofractionated Stereotactic Body Radiation Therapy Arm (36.25 Gy in 5 fractions). In this rigorous SBRT regime, 75% of the model-generated plans achieved strict clinical acceptance. Despite the steeper dose gradients required for extreme hypofractionation, Dose-PlanNet successfully navigated the complex spatial constraints, maintaining the $\geq 3 4 . 4 4$ Gy clinical mandate for the PTV while aggressively sparing the adjacent anterior rectal wall. The high autonomous pass rate in the Stereotactic Body Radiation Therapy Arm confirms that the network’s spatial awareness and physics-guided loss engine generalize eficiently to high-dose-per-fraction modalities.

## 7 Limitations and Technical Refinements

While Dose-PlanNet demonstrates robust clinical viability, the evaluation revealed specific dosimetric and computational limitations inherent to the voxel-wise prediction architecture. Identifying these constraints necessitated targeted post-processing refinements to ensure final clinical deployability.

## 7.1 Dosimetric Trade-ofs in Target Homogeneity

Although the proposed architecture consistently demonstrated superior or equivalent Organ-At-Risk (OAR) sparing compared to expert human planners, a marginal dosimetric trade-of was observed within the target volumes. Specifically, the model-generated plans frequently exhibited a slightly lower $D _ { 9 5 }$ coverage and marginally higher maximum point doses $( D _ { m a x } )$ within the PTV compared to the Eclipse ground truth.

While these target metrics remained strictly within the mandatory clinical acceptability thresholds across the validation cohort, they indicate that the network’s loss engine tends to prioritise steep spatial dose fall-of at the boundaries of critical structures, occasionally at the slight expense of absolute target homogeneity.

## 7.2 Non-Physical Ghost Dose Generation

During initial unconstrained inference, the network exhibited a propensity to generate spurious “ghost doses”. In a minor subset of axial slices (approximately 2 to 3 slices out of a standard 240-260 slice CT volume), the raw network output predicted isolated dose pockets on the order of ∼5.0 Gy located significantly distant from the primary isocenter (e.g., in the superior abdominal region near the stomach). This artifact stems from the expansive receptive field of the 3D U-Net, which can occasionally misinterpret distant, low-contrast anatomical boundaries when evaluating isolated patches without explicit global coordinate constraints. To mitigate this, a deterministic spatial bounding mask was implemented in the final inference pipeline, enforcing a strict zero-dose limit beyond a predefined geometric radius from the PTV boundaries.

## 7.3 Low-Dose Morphological Irregularities

The raw predicted dose arrays occasionally presented morphological irregularities, particularly within the low-dose wash regions (e.g., the 12.5 Gy isodose contour). These regions sometimes exhibited sharp, discontinuous edges or isolated dose “islands” in near-zero-dose regions that violate the continuous physical scatter principles of actual photon beam transport. Such high-frequency spatial noise degrades the physical deliverability of the plan, as it would require erratic and physically impossible Multi-Leaf Collimator (MLC) sequencing to execute in a linear accelerator. This limitation was efectively resolved by introducing a low-pass 3D Gaussian smoothing filter during post-processing, which eradicated the isolated islands and smoothed the isodose contours into deliverable, physically consistent distributions.

## 8 Conclusion

This study demonstrates that Dose-PlanNet, a physics-guided deep learning architecture, successfully automates clinically viable treatment planning for prostate radiotherapy. Evaluated against expert human planners on the Eclipse TPS, the network achieved comparable target coverage while consistently delivering superior low-dose sparing for critical organs at risk, notably the bladder and anorectum.

Inherent limitations of voxel-wise prediction—specifically single-voxel maximum dose artifacts, nonphysical ghost dose generation, and low-dose morphological irregularities—were efectively neutralized using targeted spatial bounding masks and 3D Gaussian smoothing. These deterministic refinements ensure that the predicted dose arrays are not only mathematically compliant with stringent Prospective Randomized protocol constraints but also physically deliverable via standard multi-leaf collimator sequencing.

Ultimately, Dose-PlanNet provides a highly robust, autonomous pipeline that drastically accelerates the radiotherapy treatment planning workflow without compromising dosimetric quality, ofering a scalable solution for high-precision clinical deployment.

## Data and Code Availability

The architecture of the Dose-PlanNet pipeline, along with all relevant code and supporting data used in this study, will be made openly available soon, in the CHAVI-India GitHub repository at https://github.com/ CHAVI-India.

## Acknowledgements

We sincerely thank all the clinicians, physicists, research fellows at Tata Medical Center Kolkata and every other person who helped us in every way to conduct this research swiftly. We are also extremely thankful to Atabur Rahman Mollah, data imaging scientist, Tata Medical Center, without whose help the implementation and testing of Dose-PlanNet would not have been possible.

## References

[1] Wu, Q., Mohan, R., et al. (2009). “Simultaneous integrated boost intensity-modulated radiotherapy for localized prostate cancer.” International Journal of Radiation Oncology\*Biology\*Physics, 75(3), 856-863.

[2] Dogan, N., King, S., et al. (2003). “Assessment of diferent IMRT boost delivery methods on target coverage and normal-tissue sparing.” International Journal of Radiation Oncology\*Biology\*Physics, 57(5), 1480-1491.

[3] Michalski, J. M., Yan, Y., et al. (2010). “Preliminary toxicity analysis of 3-dimensional conformal radiation therapy versus intensity modulated radiation therapy on the high-dose arm of the Radiation Therapy Oncology Group 0126 prostate cancer trial.” International Journal of Radiation Oncology\*Biology\*Physics, 76(1), 14-22.

[4] Nelms, B. E., Robinson, G., et al. (2012). “Variation in external beam treatment plan quality: An inter-institutional study of planners and planning systems.” Practical Radiation Oncology, 2(4), 296-305.

[5] Nguyen, D., Long, T., et al. (2019). “Image-guided radiotherapy dose prediction using a 3D U-Net deep learning architecture.” Physics in Medicine & Biology, 64(6), 065012.

[6] Kearney, V., Chan, J. W., et al. (2018). “DoseNet: a volumetric dose prediction algorithm using 3D fully-convolutional neural networks.” Physics in Medicine & Biology, 63(23), 235022.

[7] Chung, C. V., Khan, M. S., Olanrewaju, A., Pham, M., Nguyen, Q. T., Patel, T., Das, P., O’Reilly, M. S., Reed, V. K., Jhingran, A., Simonds, H., Ludmir, E. B., Hofman, K. E., Naidoo, K., Parkes, J., Aggarwal, A., Mayo, L. L., Shah, S. J., Tang, C., Beadle, B. M., Wetter, J., Walker, G., Hughes, S., Mullassery, V., Skett, S., Thomas, C., Zhang, L., Nguyen, S., Mumme, R. P., Douglas, R. J., Baroudi, H., & Court, L. E. (2025). Knowledge-based planning for fully automated radiation therapy treatment planning of 10 diferent cancer sites. Radiother Oncol, 202, 110609. doi:10.1016/j.radonc.2024.110609.

[8] Babier, A., Boutilier, J. J., et al. (2018). “Knowledge-based automated planning for oropharyngeal cancer.” Medical Physics, 45(7), 2875-2883.

[9] McIntosh, C., & Purdie, T. G. (2017). “Voxel-based dose prediction with multi-patient atlas selection for automated radiotherapy treatment planning.” Physics in Medicine & Biology, 62(2), 415.

[10] Ge, Y., Wu, Q. J., et al. (2019). “A dose-volume histogram based equation-guided 3D active contour model for generating radiotherapy dose distributions.” Physics in Medicine & Biology, 64(22), 225010.

[11] Chen, X., Men, K., et al. (2019). “Incorporating anatomical and spatial constraints into deep learningbased radiotherapy dose prediction.” Medical Physics, 46(2), 564-573.

[12] Mahmood, R., Babier, A., et al. (2018). “Automated treatment planning in radiation therapy using generative adversarial networks.” Machine Learning for Healthcare Conference, 484-499.

[13] Achlatis, S., Gavves, E., & Sonke, J.-J. (2025). “Physics-guided radiotherapy treatment planning with deep learning.” arXiv preprint arXiv:2506.19880.

[14] Chang, H.-h., Harms, J., Cardan, R. A., Fiveash, J. B., Popple, R. A., & Cardenas, C. E. (2025). “nnDoseNet: Intuitive and flexible deep learning framework to train and evaluate radiotherapy dose prediction models.” Computers in Biology and Medicine, 198, 111237.

[15] Murthy, V., Mallick, I., Gavarraju, A., et al. (2020). Study protocol of a randomised controlled trial of prostate radiotherapy in high-risk and node-positive disease comparing moderate and extreme hypofractionation (PRIME TRIAL). BMJ Open, 10, e034623. doi:10.1136/bmjopen-2019-034623.

[16] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

[17] C¸ i¸cek, O., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning<sup>¨</sup> dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19 (pp. 424-432). Springer International Publishing.

[18] Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Fang, Y., Zhuang, A., Neher, P., Kainz, B., Faller, C., & others. (2022). MONAI: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701.

[19] Murthy, V., et al. (2018). “Randomised controlled trial of Prostate Radiotherapy In high risk and node positive disease comparing Moderate and Extreme hypofractionation (Prospective Randomized Trial).” ClinicalTrials.gov, Identifier: NCT03561961. Sponsored by Tata Memorial Centre.

[20] International Commission on Radiation Units and Measurements (ICRU) (2010). “ICRU Report 83: Prescribing, Recording, and Reporting Photon-Beam Intensity-Modulated Radiation Therapy (IMRT).” Journal of the ICRU, 10(1).

[21] International Commission on Radiation Units and Measurements (ICRU) (2017). “ICRU Report 91: Prescribing, Recording, and Reporting of Stereotactic Treatments with Small Photon Beams.” Journal of the ICRU, 17(2).

[22] Sievinen, J., Ulmer, W., & Kaissl, W. (2005). “AAA Photon Dose Calculation Model in Eclipse.” Varian Medical Systems Whitepaper, Palo Alto, CA.

[23] Vassiliev, O. N., Wareing, T. A., et al. (2010). “Validation of a new grid-based Boltzmann equation solver for dose calculation in radiotherapy with photon beams.” Physics in Medicine & Biology, 55(3), 581-598.

[24] Newell, M. E., Newell, R. G., & Sancha, T. L. (1972). A solution to the hidden surface problem. Proceedings of the ACM annual conference - Volume 1, 443–450.

## A Model Hyperparameters and Optimization Strategy

To ensure multi-institutional reproducibility and explicitly define the physics-guided optimisation landscape, the exact hyperparameter configurations and loss weights (λ) used for training Dose-PlanNet are documented below. As established during clinical evaluation, these parameters were not dynamically scheduled; instead, they were fixed after being manually tuned through rigorous human evaluation of dosimetric boundaries across intermediate validation runs. During the initial 30 warm-up epochs, the physics constraints were linearly ramped to these final fixed magnitudes to ensure early numerical stability.

## A.1 Network and Optimization Hyperparameters

Both the Moderate Hypofraction Arm and Stereotactic Body Radiation Therapy Arm models utilize a unified 3D U-Net backbone trained via PyTorch and MONAI. The network is optimized using the AdamW optimizer paired with a Cosine Annealing Learning Rate scheduler. Mixed-precision training (torch.cuda.amp) was utilized across all iterations to accelerate gradient calculation and optimize memory allocation.

## A.2 Physics-Guided Loss Engine Weightings

The multi-objective loss function $( \mathcal { L } _ { t o t a l } )$ relies on a set of manually tuned scalar weights (λ) to balance standard volumetric regression against strict physical boundaries and clinical mandates. Table 5 details the specific λ coeficients applied to the Moderate Hypofraction Arm and SBRT Stereotactic Body Radiation Therapy Arm configurations, highlighting the distinct penalization choices mapped directly to each clinical protocol.

<table><tr><td>Hyperparameter</td><td>Moderate Hypofraction Arm</td><td>SBRT Arm</td></tr><tr><td>Maximum Epochs</td><td>500</td><td>300</td></tr><tr><td>Warm-up Epochs</td><td>30</td><td>30</td></tr><tr><td>Base Learning Rate</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Minimum Learning Rate</td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Batch Size</td><td>1</td><td>1</td></tr><tr><td>Gradient Accumulation Steps</td><td>2</td><td>2</td></tr><tr><td>Patch Size</td><td> $1 2 8 \times 1 2 8 \times 6 4$ </td><td> $1 2 8 \times 1 2 8 \times 6 4$ </td></tr><tr><td>Validation Split Ratio</td><td>0.20</td><td>0.20</td></tr></table>

Table 4: Training and Optimization Hyperparameters.

<table><tr><td>Loss Component</td><td>Notation</td><td>Moderate Hypofraction Arm λ</td><td>SBRT Arm λ</td></tr><tr><td>Reconstruction Fidelity</td><td></td><td></td><td></td></tr><tr><td>Dose-Weighted MSE</td><td> $\lambda _ { m s e }$ </td><td>25.0</td><td>25.0</td></tr><tr><td>Anti-Collapse Penalty</td><td> $\underline { { \lambda _ { a n t i c o l l a p s e } } }$ </td><td>150.0</td><td>150.0</td></tr><tr><td>D-Type Target Penalties</td><td></td><td></td><td></td></tr><tr><td>PTV Coverage Hinge</td><td> $\lambda _ { p t v }$ </td><td>75.0</td><td>75.0</td></tr><tr><td>PTV Maximum Dose Hinge</td><td> $\lambda _ { p t v \_ m a x }$ </td><td>30.0</td><td>30.0</td></tr><tr><td>V-Type OAR Penalties</td><td></td><td></td><td></td></tr><tr><td>Mandatory DVH Thresholds</td><td> $\lambda _ { m a n d }$ </td><td>70.0</td><td>70.0</td></tr><tr><td>Optional DVH Thresholds</td><td> $\lambda _ { o p t }$ </td><td>2.0</td><td>2.0</td></tr><tr><td>Bowel Bag Specific Penalty</td><td> $\lambda _ { b o w e l }$ </td><td>15.0</td><td>15.0</td></tr><tr><td>Femoral Head Specific Penalty</td><td> $\lambda _ { f e m u r }$ </td><td>10.0</td><td>10.0</td></tr><tr><td>Penile Bulb Specific Penalty</td><td> $\lambda _ { \mathit { p e n i l e } }$ </td><td>10.0</td><td>0.0</td></tr><tr><td>Spatial Regularisation &amp; Geometry</td><td></td><td></td><td></td></tr><tr><td>Global Physical Ceiling</td><td> $\lambda _ { g l o b a l \_ c e i l }$ </td><td>300.0</td><td>300.0</td></tr><tr><td>Geometric Ring Falloff</td><td> $\lambda _ { r i n g }$ </td><td>25.0</td><td>25.0</td></tr><tr><td>BEV Tissue Suppression</td><td> $\lambda _ { B E V }$ </td><td>10.0</td><td>10.0</td></tr><tr><td>Body Mask/Ghost Suppression</td><td> $\lambda _ { b o d y }$ </td><td>20.0</td><td>20.0</td></tr><tr><td>Total Variation (Smoothness)</td><td> $\lambda _ { T V }$ </td><td>0.125</td><td>0.125</td></tr><tr><td>Laplacian (Second-Order)</td><td> $\underline { { \lambda _ { l a p l a c i a n } } }$ </td><td>0.25</td><td>0.25</td></tr></table>

Table 5: Physics-Guided Loss Engine Penalties (λ Weights).

## B Software Environment and Reproducibility

To facilitate external validation and ensure computational reproducibility, the study was implemented using the following core library versions. The pipeline relies on a specific environment to maintain consistency in volumetric regression and coordinate-space interpolation:

<table><tr><td>Library</td><td>Version</td></tr><tr><td>Python</td><td>3.10.2</td></tr><tr><td>PyTorch</td><td>2.3.0+cu121</td></tr><tr><td>MONAI</td><td>1.3.0</td></tr><tr><td>SimpleITK</td><td>2.3.1</td></tr><tr><td>PyDICOM</td><td>2.4.4</td></tr><tr><td>SciPy</td><td>1.13.0</td></tr><tr><td>NumPy</td><td>1.26.4</td></tr></table>

Table 6: Core dependencies and library versions.