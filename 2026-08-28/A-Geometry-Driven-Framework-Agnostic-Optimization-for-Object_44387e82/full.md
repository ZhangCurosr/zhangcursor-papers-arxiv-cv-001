# A Geometry-Driven, Framework-Agnostic Optimization for Object

Pose Estimation

Wei Chen, Tao Zhen, Jing Zhang, Zhongchen Shi, Liang Xie, Erwei Yin

Abstract—Current object pose estimation research remains predominantly model-centric, focusing on architectural innovations and post-processing refinements. This paper introduces a data-centric optimization by proposing a novel, physically grounded rotation representation through principal axes alignment. Our method aligns the object’s coordinate system with its inherent geometric axes, derived from inertial properties, yielding three key advantages: Inherent Stability—leveraging the energyminimizing property of principal axes provides a robust representation that is less sensitive to noise and occlusions; Symmetry-Aware Canonicalization—explicitly resolving rotational ambiguities for symmetric objects at the data level, which fundamentally eliminates label confusion during network training; and Framework Agnosticism—the optimization is applied purely at the dataset level, ensuring plug-and-play compatibility with existing networks without any architectural modification. We validate the framework across diverse category-level and instance-level models. Extensive experiments demonstrate consistent and significant accuracy improvements, while preserving the integrity of the baseline network. This work establishes a new, geometrydriven direction for enhancing pose estimation, circumventing the need for complex network redesign.

Index Terms—6D pose estimation, rotation optimization, dataset optimization, and moment of inertia.

## I. INTRODUCTION

STIMATING the 6 Degree-of-Freedom (6D) and 3D translation—is a foundational computer vision task essential for bridging perceptual understanding with physical interaction. Robust and accurate 6D pose estimation underpins critical applications such as augmented reality (requiring precise virtual-real alignment) [1], autonomous driving (relying on exact localization of obstacles and agents), and robotic manipulation (demanding real-time, millimeter-accurate control of tools and workpieces) [2], [3].

Significant progress has been made in 6D object pose estimation for both instance-level and category-level tasks, as evidenced by extensive recent work [4]–[12]. However, the prevailing research focus has been on improving neural networks themselves, primarily through three types: (1) designing increasingly complex architectures with multi-stage pipelines, such as attention modules, and sophisticated feature fusion mechanisms [5], [6], [13]; (2) developing specialized loss functions to address challenges such as object symmetries, and non-linear error metrics [14]–[17]; and (3) exploring diverse rotation representations, including quaternions, axis-angle, and continuous embeddings, to facilitate more accurate regression [4], [18].

While these network-centric advancements have contributed to higher accuracy, they introduce notable practical limitations. First, integrating such enhancements often entails considerable implementation complexity. For instance, the adoption of specialized rotation representations [18] not only requires algorithmic adaptations but may also be incompatible with frameworks that do not directly regress rotation parameters, such as keypoint-based [19] or correspondence-driven methods [6]. Second, post-refinement modules [12], [20] are conceptually straightforward. However, they inevitably increase inference latency. This added computational overhead can compromise real-time performance, making such approaches less suitable for latency-sensitive applications, such as robotic manipulation or interactive augmented reality. Third, for symmetric objects, existing methods either rely on handcrafted primitives or loss functions, or employ complex architectures to achieve an approximate mapping; neither of these approaches provides a principled solution for resolving rotational ambiguity.

![](images/c535175e5e42d941c6089bce88fcccb3fad8665e32652916fc31bd85846530a5.jpg)  
Fig. 1. Semantic illustration of proposed mechanism.The main contribution of previous methods mainly focused on network layout update, including new loss functions or new network architectures. Our proposed method only updates the rotation coordinate for pose representation, which is easy to apply and perfectly compatible with existing methods.

To overcome the aforementioned issues, in this paper, we design a simple mechanism that improves the pose estimation accuracy without changing any layout details of the existing network or increasing the running time.

The proposed method, termed the π-Mechanism (Principal Axes of Inertia, PAI), establishes a canonical object coordinate system by aligning it with the three principal axes of inertia derived from the object’s 3D geometry. This alignment is achieved by calculating the rotation matrix that maps the original object coordinates to this new, intrinsic reference frame. Consequently, all ground-truth pose annotations in the training set are transformed into this stable coordinate system via this matrix. A network is then trained or fine-tuned to regress poses relative to this canonical frame. The key advantages of this approach are threefold: 1) It is model-agnostic, as it operates purely on the dataset without altering the network architecture;2) it is computationally lightweight, requiring only a single matrix multiplication per object during pre-processing;

3) it resolves the ambiguous rotations of symmetric objects in a simple and analytical way.

To summarize, this paper makes the following primary contributions to the field of 6D object pose estimation:

• A Data-Centric, Geometry-Driven Framework: We introduce a paradigm shift from modifying network architectures to optimizing their geometric input. The core of this framework is the proposed π-Mechanism, a plugand-play, model-agnostic module that canonicalizes an object’s coordinate system by aligning it with its intrinsic Principal Axes. This establishes a stable reference frame rooted in classical mechanics, which is applied through a simple, one-time data preprocessing step requiring no changes to existing network architectures, loss functions, or training pipelines.

• Symmetry-Aware Canonicalization via Inertial Principal Axes: We propose a data-level canonicalization scheme that explicitly resolves rotational ambiguities for symmetric objects. By aligning objects to their inherent principal axes and disambiguating their orientations with deterministic rules, our method eliminates label confusion at the source, leading to significantly improved pose estimation accuracy on symmetric objects without any network modification.

• Extensive and Conclusive Empirical Validation: We demonstrate consistent and significant accuracy improvements across multiple benchmarks (LINEMOD, LINEMOD-OCC, NOCS-REAL), covering both instance-level and category-level tasks under various challenges. The results robustly validate that the introduced geometric stability directly translates to enhanced learning performance and generalization.

## II. RELATED WORK

## A. Network Architecture Optimization

Recent advances in 6D pose estimation have focused on enhancing network architectures through multimodal fusion, computational efficiency, and cross-domain generalization. Early approaches, such as PoseCNN [4], relied primarily on RGB input followed by geometric post-processing. Subsequent work emphasized the fusion of multimodal data, as seen in DenseFusion [21], which performs pixel-wise fusion of RGB and point cloud features. More recent architectures have incorporated transformer-based modules for cross-modal attention, exemplified by Trans6D [22]. To improve feature extraction under occlusion, GatedUniPose [23] integrated gated convolutions with UniRepLKNet backbones [24], reporting notable gains on challenging benchmarks. Another line of research targets generalization across object categories and sensing conditions; for instance, FoundationPose [12] leverages neural implicit representations and large-scale synthetic data to achieve strong cross-domain performance on standardized benchmarks. While these architectural innovations push the state of the forward accuracy, they often introduce increased model complexity, require careful tuning, or depend on specialized training data—factors that can limit their practical adoption and ease of deployment.

## B. Pose Refinement

Pose refinement techniques aim to enhance initial pose estimates through geometric or learning-based optimization. Classical iterative closest point (ICP) methods [25] minimize pointto-point distances but often struggle in textureless or cluttered scenes. The advent of deep learning shifted the paradigm toward learned refinement: DeepIM [26] pioneered an iterative update scheme using differentiable rendering, while GDR-Net [27] introduced a geometry-guided direct refinement module within a single forward pass. More recent approaches integrate physical constraints and explicit uncertainty modeling. For example, Repose [28] employs deep reinforcement learning to adaptively minimize feature-metric residuals. Coupled Iterative Refinement [29] proposes a differentiable bidirectional PnP layer (BD-PnP) that jointly refines 2D–3D correspondences and the final pose while dynamically rejecting outliers. In the category-level setting, GeoReF [30] addresses intra-class shape variation via geometric alignment using hierarchical semantic (HS) layers and cross-cloud transformation, reporting significant gains over prior refinement baselines.

## C. Rotation Representation

Rotation representation significantly impacts pose estimation performance. Traditional representations include rotation matrices (9D) and Euler angles (3D), but both suffer from discontinuities and constraints (orthogonality/gimbal lock) [18]. Quaternions (4D) became popular for their compactness and continuity, yet ambiguity (q = -q) complicates loss formulation [31]. Recent works focus on learning-friendly representations: [18] decoupled rotation into two orthogonal axes, avoiding singularities while enabling Gram-Schmidt recovery. [31] addressed gradient misalignment in Riemannian manifolds (e.g., SO(3)) by projecting Euclidean gradients onto tangent spaces, improving convergence for quaternion/6D/9D representations. In [6], the authors decoupled the rotation representation into two vectors, which are perpendicular to each other, to improve the rotation estimation accuracy. For category-level pose estimation, GPV-Pose [32] integrated geometry-guided point voting and decoupled rotation losses to handle intraclass shape variations. Recent trends also explore implicit representations (e.g., NeRFs for refinement) and equivariant networks [33] for rotation-invariant point cloud features. However, challenges persist in applying these new representations to different methods.

## D. Symmetries in Pose Estimation

In the field of 6D object pose estimation, handling object symmetry is a core challenge. Existing approaches primarily address it at the levels of loss function design, output representation, and post-processing. Early methods based on correspondence matching often suffered from pose ambiguity due to symmetry. Subsequent research has improved by explicitly modeling symmetry. One category of methods introduces symmetry-aware loss functions during training, such as treating symmetric poses as equivalent labels and computing a minimum distance loss to accommodate multiple valid solutions [4]. Another line of work focuses on designing symmetry-invariant intermediate representations, for example, predicting object-local coordinates or symmetry-aware keypoints to avoid ambiguity at the regression stage [16], [19], [21]. In recent years, methods have managed symmetry by defining symmetry-robust surface coordinate mappings or aggregating symmetric equivalent solutions through voting mechanisms [34], [35]. Besides, some methods designed symmetry-invariant loss functions and normalizing the output pose space [17]. Others proposed a theoretically grounded rotation normalization method that maps all visually equivalent rotations to a canonical representation with multi-regressors. While significant progress has been made for common symmetric objects, efficiently and uniformly modeling symmetry for objects with complex, continuous, or partial symmetry remains an open research direction.

## III. PROPOSED METHOD

While contemporary research predominantly focuses on advancing network architectures for object pose estimation, we identify a fundamental yet overlooked weakness: the reliance on arbitrarily defined, and thus geometrically unstable, object coordinate systems. This section presents our core innovation: a data-centric optimization paradigm that replaces these ad hoc frames with a canonical, physically grounded coordinate system defined by the object’s principal axes of inertia. Our method is agnostic to network architectures, offering a plug-and-play preprocessing step that systematically enhances training data to achieve more stable and efficient learning of rotation representations.

The main steps of our proposed π-Mechanism are shown in Fig. 2. The proposed mechanism comprises four key stages. First, the principal axes of inertia for the target object are computed to determine its intrinsic geometric orientation. Subsequently, a rotational transformation matrix is constructed from these three orthogonal axes. This matrix is then applied to the original dataset, translating all data points into a new, canonical coordinate system defined by the principal axes. Finally, the neural network is either retrained from scratch or fine-tuned using the transformed dataset, enabling it to learn features within this normalized and physically meaningful frame of reference. We explain the details of the above steps in the following sections.

A. Geometrically Stable Representation via Principal Axes Alignment

1) Principal Axes: The core of our geometric optimization lies in replacing the arbitrarily defined object coordinate system with one grounded in the physical properties of the object itself. We achieve this by defining the canonical frame based on the object’s principal axes of inertia, which are derived from its mass distribution geometry. Formally, for a object 3D model $\mathcal { M } = \{ \mathbf { v } _ { i } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { N }$ , where ${ \bf v } _ { i }$ means the $i ^ { t h }$ vertex of the 3D model, we first compute the centroid $\mathbf { c } = \frac { 1 } { N } \Sigma _ { i } \mathbf { v } _ { i }$ , of $\mathcal { M } .$

Then the inertia tensor $\mathcal { T } _ { t e n s o r } \in \mathbb { R } ^ { 3 \times 3 }$ , which characterizes the mass distribution relative to c, is calculated as:

$$
{ \mathcal { T } } _ { t e n s o r } = \sum _ { i = 1 } ^ { N } ( \lVert \mathbf { v } _ { i } - \mathbf { c } \rVert ^ { 2 } \cdot \mathbf { I } - ( \mathbf { v } _ { i } - \mathbf { c } ) ( \mathbf { v } _ { i } - \mathbf { c } ) ^ { \top } ) ,\tag{1}
$$

where I is the $3 \times 3$ identity matrix. According to the relative theory [36], [37], the principal axes are obtained via eigendecomposition of this symmetric tensor:

$$
{ \bf I } _ { t e n s o r } \cdot { \bf e } _ { j } = \lambda _ { j } \cdot { \bf e } _ { j } , \quad j \in \{ 1 , 2 , 3 \} .\tag{2}
$$

The eigenvectors $\mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \mathbf { e } _ { 3 }$ define the orthogonal principal axes. These three vectors constitute our intrinsic coordinated frame $\mathcal { F } _ { P A } = \mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \mathbf { e } _ { 3 }$

There are several advantages of our proposed mechanism: 1) rotation stability: the inertia tensor is diagonal in the proposed coordinate $\mathcal { F } _ { P A }$ , which means rotations around these axes are decoupled and correspond to the object’s natural modes of rotation with minimal dynamic disturbation across different axes (as shown in Fig. 3, the non-diagonal elements are close to zero in $\mathcal { F } _ { P A } )$ . This property translates directly to reduced sensitivity to perturbations in pose estimation. 2) symmetry invariance: the proposed axes naturally align with the object’s intrinsic symmetry axes. ensuring that symmetric transformations (such as rotations by multiples of $1 8 0 ^ { \circ }$ for two-fold symmetric objects) leave the axes representation unchanged. Based on this property, we can detect and eliminate rotation ambiguity automatically (as illustrated in Fig. 7, where symmetric rotations map to identical canonical poses).

2) The Alignment Transformation: Based on the established principal axes as the intrinsic object frame $\mathcal { F } _ { P A }$ , we now formalize the canonical transformation that aligns any object instance from its original, arbitrary coordinate system $\mathcal { F } _ { o r i g }$ to $\mathcal { F } _ { P A }$ . Specifically, we apply a pure rotation to the object’s coordinate system. The transformation is defined as follows.

Let ${ \bf V } = [ { \bf v } _ { 1 } , { \bf v } _ { 2 } , { \bf v } _ { 3 } ] \in \mathbb { R } ^ { 3 \times 3 }$ be the matrix whose columns are the three principal axes computed in the above section. We construct the alignment rotation matrix $\mathbf { R } _ { p } ^ { o } \in S O ( 3 )$ as:

$$
\mathbf { R } _ { p } ^ { o } = \mathbf { V } ,\tag{3}
$$

which represents the rotation from the original object coordinate system to the principal axes system.

Given a ground-truth pose in the original frame, represented as a rotation $\mathbf { R } _ { o r i } .$ , the corresponding rotation in the aligned frame is obtained by (shown in Fig. 4):

$$
\begin{array} { r } { \mathbf { R } _ { \mathrm { P A } } = \mathbf { R } _ { p } ^ { o } \cdot \mathbf { R } _ { \mathrm { o r i g } } . } \end{array}\tag{4}
$$

Notably, the translation component remains unchanged. It is defined with respect to the object’s centroid, which is consistent across both coordinate systems.

This alignment is applied as a preprocessing step to the entire training dataset. For each training sample, the 3D model vertices $\mathcal { M } _ { \mathrm { o r i g } }$ are canonically oriented via $\mathcal { M } _ { \mathrm { P A } } ~ =$ $\mathbf { R } _ { p } ^ { o } \cdot \mathcal { M } _ { \mathrm { o r i g } } ,$ , and the associated rotation labels are transformed accordingly. During inference, a network predicts a rotation $\mathbf { R _ { \mathrm { p r e d } \to \mathrm { P A } } }$ relative to $\{ O _ { \mathrm { P A } } \}$ , which can be converted back to the original frame for evaluation or application as $\mathbf { R } _ { \mathrm { p r e d } \to \mathrm { o r i g } } = \mathbf { R } _ { p } ^ { o T } \cdot \mathbf { R } _ { \mathrm { p r e d } \to \mathrm { P A } }$ . This bi-directional, rotationonly mapping ensures that our geometrically stable representation is seamlessly integrated into the standard pose estimation pipeline.

![](images/4bab97f0a636055f62fd7448a51262b4d017409684814ac02ee5388e403c3095.jpg)  
Fig. 2. Overview of the proposed mechanism and its usage. There are four main steps of our proposed mechanism. First, we calculate the principal axes of the object’s inertia, and then we form the transformation rotation using these three principal axes. Third, we access the new object coordinate for the datasets with the transformation rotation. Then, we eliminate the rotation ambiguity with the proposed module if the objects are symmetric. Finally, we retrain the network using datasets in new coordinates, eliminating rotation ambiguity.

![](images/d36ce5bbb61ec7e6cbd0167c906a26c011f9e69a2bc25ec2b28e1110a4c9ca5e.jpg)

![](images/221af5ed0e95455cd456852c0ccc51711a64e8ab7ada3b4cafde1e45fe40a955.jpg)  
Fig. 3. Physics property of π-Mechanism. For original coordinates, the non-diagonal elements are non-zero. For the object in Principal Axes, the non-diagonal elements are close to zero.

## B. Canonical Reduction of Rotational Ambiguity for Symmetric Objects

A fundamental property of rigid-body dynamics ensures that any axis of rotational symmetry of an object must coincide with a principal axis of its inertia tensor. This arises because the inertia tensor, as a second-order moment integral of the mass distribution, remains invariant under the symmetry operations defining the object. Consequently, the eigen-decomposition of the inertia tensor naturally reveals and aligns with the object’s intrinsic geometric symmetries.

![](images/e1c9bb9bc3fa703d26c7e9e268d6329ad9cbaf1767c72673ca98becea1b92dd2.jpg)  
Fig. 4. Illustration of the effect of π-Mechanism. The above row shows the geometric relation between the object and its coordinates. When the rotation value is 0, the orientation of the object is not aligned well with the axes of the object coordinate. The second row of images shows the relation when applying our proposed mechanism to the original object relation. The orientation of the object is aligned well with its coordinate.

According to this, another pivotal advantage of our approach is its explicit handling of symmetry through physical properties. In the following, we will describe how we handle the symmetric object for pose estimation. We first detect whether an object is symmetric. If symmetry is confirmed, classification rules are applied, ambiguous rotation labels are mapped to a canonical space, and the network is trained using these canonical rotations.

1) Symmetry Detection & Classification: The core premise is that an object is rotationally symmetric if and only if a rotation around a certain axis by some angle θ results in a shape that is congruent to its original configuration.

Formally, let $\bar { \mathcal { P } } \in \mathbb { R } ^ { N \times 3 }$ denote the centered object point cloud (with centroid at the origin), and $\mathbf { V } ^ { \prime } = [ \mathbf { v } ^ { \prime } { } _ { 1 } , \mathbf { \bar { v } } ^ { \prime } { } _ { 2 } , \mathbf { \bar { v } } ^ { \prime } { } _ { 3 } ] \in$ $\mathbb { R } ^ { 3 \times 3 }$ be its matrix of inertial principal axes. The symmetry detection is performed independently for each principal axis ${ \bf v } _ { k } \ ( k = 1 , 2 , 3 )$ as follows:

• Angular Space Discretization: An angular resolution $\Delta \theta ( \mathrm { e . g . , 5 ^ { \circ } } )$ is defined. A sequence of test angles is generated: $\Theta = \{ \theta _ { i } \ | \ \theta _ { i } = i \cdot \Delta \theta _ { \vdots }$ $i =$ $1 , 2 , . . . , \left\lfloor 3 6 0 / \Delta \theta \right\rfloor \Sigma$

• Rotation and Shape Matching: For each $\theta _ { i } \in \Theta$ , the rotation matrix $\mathbf { R } ( \mathbf { v } _ { k } , \theta _ { i } ) \in \mathrm { S O } ( 3 )$ about axis $\mathbf { v } _ { k }$ is computed. The point cloud is rotated to obtain $\mathbf { P } _ { i } ^ { \prime } = \mathbf { P } \cdot \mathbf { R } ( \mathbf { v } _ { k } , \theta _ { i } ) ^ { T }$

The shape congruence error $d _ { i }$ is then calculated as the bidirectional Chamfer Distance between the original P and the rotated $\mathbf { P } _ { i } ^ { \prime } \colon$

$$
\begin{array} { l } { \displaystyle d _ { i } = D _ { C D } ( { \bf P } , { \bf P } _ { i } ^ { \prime } ) = } \\ { \displaystyle \frac { 1 } { | { \bf P } | } \sum _ { { \bf x } \in { \bf P } } \operatorname* { m i n } _ { { \bf y } \in { \bf P } _ { i } ^ { \prime } } \| { \bf x } - { \bf y } \| _ { 2 } } \\ { + \displaystyle \frac { 1 } { | { \bf P } _ { i } ^ { \prime } | } \sum _ { { \bf y } \in { \bf P } _ { i } ^ { \prime } } \displaystyle \operatorname* { m i n } _ { { \bf x } \in { \bf P } } \| { \bf y } - { \bf x } \| _ { 2 } . } \end{array}\tag{5}
$$

## • Symmetry Angle Determination:

Given a predefined symmetry tolerance threshold $\tau \ ( \mathrm { d e } -$ termined by point cloud scale and noise), an angle $\theta _ { i }$ is considered valid if $d _ { i } < \tau .$ , indicating self-congruence after rotation. The set of all valid angles for axis $\mathbf { v } _ { k }$ is:

$$
S _ { k } = \{ \theta _ { i } \in \Theta \cup \{ 0 ^ { \circ } , 3 6 0 ^ { \circ } \} \mid d _ { i } < \tau \} .\tag{6}
$$

The angles $0 ^ { \circ }$ and $3 6 0 ^ { \circ }$ , representing the identity rotation, are included by default.

The rotation symmetry of an object can be automatically detected based on the above calculation. We classify the symmetry of the object as follows.

We first check the symmetry of the object about the first axis of $\mathcal { F } _ { P A }$ based on the structure of $S _ { 1 }$

• Continuous Symmetry: If $S _ { 1 }$ contains all angles in Θ $( \mathrm { i } . \mathbf { e } . , d _ { i } < \tau$ for all $\theta _ { i } )$ , the object possesses continuous rotational symmetry about this axis (e.g., cylinder, cone).

• N-fold Discrete Symmetry: If $S _ { 1 } = \{ 0 ^ { \circ } , 3 6 0 ^ { \circ } / N , 2 \times$ $3 6 0 ^ { \circ } / N , . . . , 3 6 0 ^ { \circ } \}$ for an integer $N > 1$ , the object has N-fold discrete rotational symmetry (e.g., a square prism has 4-fold symmetry).

• 2-fold Symmetry: If $S _ { 1 } = \{ 0 ^ { \circ } , 1 8 0 ^ { \circ } , 3 6 0 ^ { \circ } \}$ , the object has 2-fold (bilateral) symmetry about this axis (e.g., a cuboid rotated about the axis through the center of its faces).

• No Symmetry: If $S _ { 1 } = \{ 0 ^ { \circ } , 3 6 0 ^ { \circ } \}$ , the object is asymmetric about this axis.

We then check the symmetry of the object about the second axis based on the structure of $S _ { 2 }$ with the same operation. Without doubt, the symmetry of the object about the second axis can still be classified into four categories: Continuous Symmetry, N-fold Discrete Symmetry, 2-fold Symmetry, and

No Symmetry. Then we classify the whole symmetry of an object based on the category combination of those two axes (as shown in $\operatorname { F i g } . 5 )$ . Please note that, due to the geometric constraint [18], we do not need to check the third axis.

2) Mapping Ambiguous Rotations: Following the symmetry classification defined in Fig. 5, a critical step is to resolve the inherent rotational ambiguities for symmetric object categories. This process aims to map all physically equivalent poses within a category to a single, unique canonical pose in the dataset. This ensures a bijective mapping between an object’s appearance and its pose label, which is essential for stable network training.

Specifically, to systematically address object symmetry at the data-preprocessing stage, we propose a principal axesbased method (shown in Fig. 6, Fig. 7). We first align the object point cloud to the principal axes frame. Within this canonical frame, the algorithm performs a rotation-congruence test around each of the two principal axes at discrete angular intervals. For each rotated point cloud, the bidirectional Chamfer distance to the original is computed; if this distance falls below a predefined geometric tolerance threshold, the rotation angle is validated as a symmetry angle. The set of all such valid angles for an axis constitutes its rotational ambiguity set, which subsequently classifies the axis as having continuous, Nfold discrete, 2-fold, or no rotational symmetry. For different symmetry classifications, we employ the following strategies to map the ambiguous rotation labels to a unique set.

In contrast to prior approaches that necessitate additional network branches or specialized loss functions, our proposed solution is straightforward, analytical, and operates directly on the dataset level. As illustrated in Fig. 8, we design distinct geometric vectors for different symmetry categories according to the following rules: If an object exhibits rotational symmetry only around the first axis (as shown in the last row of Fig. 5), this axis is taken as the primary vector. The plane perpendicular to it is equally divided into N sectors using N vectors, where N corresponds to the fold of rotational symmetry. If a second symmetry axis exists, it is treated as the primary vector, and the same division is applied to its perpendicular plane. Finally, mergeable vectors are combined, resulting in the configuration depicted in Fig. 8.

Once the geometric vectors are defined, we utilize the rotation labels from datasets of symmetric objects to reorient these vectors accordingly. As demonstrated in Fig. 7, rotations belonging to the same ambiguous set yield transformed geometric vectors that are indistinguishable in direction—thereby transferring ambiguity from the rotation label (which relates to object appearance) to the vector directions. A reference direction is then established to select a unique vector, which corresponds to a canonical rotation. This process allows the network to be trained exclusively on rotation labels that are free of ambiguity. Please note that any direction can serve as the reference direction, but it should be unique for a specific object within a particular dataset.

The core advantage of this method is rooted in its physical foundation. The inertial principal axes framework transforms the problem from an infinite search in $S O ( 3 )$ to a finite verification of three deterministic, physically-grounded directions. It provides an intrinsic and stable object-centric reference frame, guaranteeing consistent symmetry analysis across all instances of the same category. This not only guides the detection process but also enables subsequent disambiguation rules to be based on stable and unique geometric directions. Thus, the process does not merely detect ambiguities but physically constrains them, which makes the rotation disambiguation simple and effective.

![](images/74262e3406b0cf50178914692a48f56e5041d13a234ff2e2be53c3bcf0a7f69d.jpg)  
Fig. 5. A Taxonomy of 3D Objects Based on Rotational Symmetry Around Two Principal Axes. We systematically categorize objects’ shapes by their rotational symmetry properties around two independent principal inertial axes. The rows and columns represent the four fundamental symmetry types for a single axis: Continuous, N-fold Discrete, 2-fold, and Asymmetric. Each cell provides one or two canonical 3D object examples (without considering the color texture) exhibiting the corresponding pairwise combination of symmetries.

![](images/a973074d209f55aad50ab1a4c4125282a555c546fc0e8619990a742d39f151d1.jpg)  
Fig. 6. Pipeline for resolving rotational ambiguity in symmetric objects. Left top: Computation of the principal axes of inertia from the object’s geometry. Right top: Symmetry type classification (e.g., two-fold symmetry) based on the principal axes. Right bottom: Generation of canonical candidate vectors. Left bottom: Final unambiguous rotation representation obtained after disambiguation.

![](images/c4c6194b4d549f72a20f1940909839e6215aeccb4f95415a31948d6627d29e4b.jpg)  
Fig. 7. Illustration of rotation disambiguation for symmetric objects. Left column: Distribution of rotation viewpoints in the original ambiguous space. Middle column: 3D model of the object in the TLESS dataset (obj29, obj27, obj14), highlighting its intrinsic symmetry. Right column: Unique distribution of rotation viewpoints after ambiguity resolution.

![](images/4da840c433ea88f7f246b00a33b0db27329f98b7cd7a0c086223c1193e97c09e.jpg)  
Fig. 8. Illustration of Geometric Vectors for Different Symmetry Types.

Although our method can eliminate the rotation ambiguities of symmetrical objects in the widely used datasets, such as LINEMOD [38], YCB-V [4], and T-LESS [39], it still has some limitations. For example, it cannot collapse the full symmetry group of polyhedra like the octahedron or dodecahedron into a single canonical pose. While one can still use our method to automatically alleviate the rotation ambiguities of polyhedra like the octahedron or dodecahedron, and then manually eliminate the remaining rotation ambiguities.

## IV. EXPERIMENTS

In this section, we deploy our π-Mechanism with different pose estimation methods to show the effectiveness of the proposed mechanism. For all baselines, we use their official codebases and training configurations without any modification (e.g., network architecture, hyperparameters, loss functions, or data augmentation pipelines). Our principal axes alignment (including rotation disambiguation) is applied solely as a preprocessing step to the 3D models and their associated pose annotations in the training and testing sets. All experiments are conducted on a PC with NVIDIA GeForce RTX 4090D GPU.

## A. Datasets and Evaluation Metrics

We evaluate our method on three public benchmarks that span both instance-level and category-level pose estimation, demonstrating its broad applicability. We use rotation errors or ADD-(S) metric for comparison.

LINEMOD [40] is a widely used instance-level 6D object pose estimation dataset which consists of 13 different objects with significant shape variation. We use the standard split of 15% for training and 85% for testing. Please note, for symmetric objects ‘Egg box’ and ‘Glue’ in LINEMOD, we only use ADD-(S) metric for comparison.

LINEMOD-OCC [41] is derived from LINEMOD dataset for occlusion scenes. It contains 1214 images, which are annotated for 8 objects under different levels of occlusion.

NOCS-REAL [42] is the first real-world dataset for categorylevel 6D object pose estimation. The training set has 4300 real images of 7 scenes with 6 categories. For each category, there are 3 unique instances. In the testing set, there are 2750 real images spread across 6 scenes of the same 6 categories as the training set. In each test scene, there are about 5 objects, which makes the dataset cluttered and challenging.

## B. Toy Experiments on Synthetic LINEMOD

We first validate the proposed method on a synthetic LINEMOD dataset. This dataset is rendered from the original LINEMOD CAD models [40] by applying random rotations to each object while keeping the translation fixed at zero. Please note that samples vary only in rotation, and are free of noise or background clutter to isolate the effect of our geometric preprocessing on rotation estimation.

We employ two representative backbones for rotation regression: PointNet [43] and SI-Mamba [44]. PointNet is a foundational architecture for direct point cloud processing, widely adopted in subsequent research [45], [46]. SI-Mamba integrates the hierarchical feature extraction of PointNet++ [47] with the efficient sequence modeling of Mamba [48], [49], representing a recent advance in network design.

The results in Table I show that our principal axes alignment improves rotation prediction accuracy by 3.3% for PointNet and 4.0% for SI-Mamba. This consistent gain across two distinct architectures demonstrates that the proposed datacentric optimization enhances rotation estimation inherently and is invariant to the choice of network backbone.

## C. Experiments on LINEMOD

Building upon the demonstrated improvement in rotation estimation within a controlled synthetic environment, we now transition to evaluating our method under more challenging, real-world conditions. To this end, we employ the standard LINEMOD benchmark and integrate our preprocessing with two established pose estimation frameworks: G2L-Net [50] and GDR-Net [27]. This experiment is designed to validate the practical effectiveness and general applicability of our geometric alignment approach.

Comparison on G2L-Net. G2L-Net is a real-time 6D pose estimation framework that processes RGB-D images in a divide-and-conquer fashion. It begins by extracting coarse object point clouds via 2D detection, followed by a translation localization network that performs 3D segmentation and predicts translation. The points are then transformed into a canonical coordinate system, where a rotation localization network estimates an initial rotation along with a residual refinement. By leveraging viewpoint-aware embedding features, the method achieves state-of-the-art accuracy on benchmark such as LINEMOD while maintaining high frame rates. We retrain G2L-Net using the official setup from [50].

As shown in Table II, integrating our π-Mechanism with G2L-Net yields consistent improvement across all objects in the LINEMOD dataset. Notably, the mean rotation error decreases from 9.61 to 7.09—a relative reduction of 26.2%. This significant enhancement demonstrates that the proposed geometric alignment effectively stabilizes the rotation estimation, complementing the strong baseline performance of G2L-Net.

Comparison on GDR-Net GDR-Net is an end-to-end framework that directly regresses 6D object pose from a single RGB image. Its key innovation lies in leveraging dense 2D–3D correspondences, structured in a patch-based representation, and enhancing feature discrimination through a surface region attention (SRA) mechanism. We retrain GDR-Net on the LINEMOD dataset using its original loss function based on the $R _ { 6 D }$ representation [18], while integrating our proposed π-Mechanism as the only modification.

The results demonstrate that our π-Mechanism consistently reduces the rotation error across categories. As shown in Table II, the mean rotation error decreases from 2.01 to 1.83—an 8.96% relative improvement. This confirms that even for a strong correspondence-based baseline, canonical alignment provides a measurable gain in rotation estimation accuracy.

TABLE I  
ROTATION ERROR FOR DIFFERENT METHODS ON THE SYNTHETIC LINEMOD DATASET. THE BEST RESULT IS BOLDED.
<table><tr><td>Method</td><td> $\overline { { \mathrm { A p e } } }$ </td><td>Bench Vise</td><td>Camera</td><td>Can</td><td>Cat</td><td>Driller</td><td>Glue</td><td>Hole Puncher</td><td>Iron</td><td>Lamp</td><td>Phone</td><td>Average</td></tr><tr><td>PointNet [43] PointNet + π-Mechanism</td><td>12.28 12.12</td><td>12.49 10.30</td><td>11.76 10.27</td><td>13.71</td><td>13.11</td><td>11.30 11.26</td><td>10.72 11.41</td><td>10.23 12.34</td><td>11.62 11.42</td><td>10.47 10.01</td><td>11.09 9.71</td><td>11.71 11.32</td></tr><tr><td></td><td></td><td></td><td></td><td>13.16</td><td>12.51</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Si-Mamba [44]</td><td>10.85 10.48</td><td>11.04</td><td>10.42 10.05</td><td>11.34 11.94</td><td>11.22 10.26</td><td>11.06 10.49</td><td>10.13 9.86</td><td>12.55</td><td>10.17</td><td>11.57</td><td>11.34 10.72</td><td>11.06</td></tr><tr><td>Si-Mamba +π-Mechanism</td><td></td><td>10.58</td><td></td><td></td><td></td><td></td><td></td><td>11.46</td><td>10.67</td><td>10.25</td><td></td><td>10.61</td></tr></table>

TABLE II  
ROTATION ERROR FOR DIFFERENT METHODS ON LINMOD DATASET. THE BEST RESULT IS BOLDED.
<table><tr><td>Method</td><td>Ape</td><td>Bench Vise</td><td>Camera</td><td>Can</td><td>Cat</td><td>Driller</td><td>Duck</td><td>Hole Puncher</td><td>Iron</td><td>Lamp</td><td>Phone</td><td>Average</td></tr><tr><td>G2L [50]</td><td>7.37</td><td>15.34</td><td>5.99</td><td>8.96</td><td>6.03</td><td>13.04</td><td>6.84</td><td>5.13</td><td>17.20</td><td>12.54</td><td>7.24</td><td>9.61</td></tr><tr><td>G2L + π-Mechanism</td><td>6.25</td><td>11.94</td><td>5.83</td><td>6.30</td><td>5.23</td><td>9.37</td><td>6.58</td><td>3.94</td><td>7.37</td><td>9.79</td><td>5.44</td><td>7.09</td></tr><tr><td>GDR-Net [27]</td><td>2.11</td><td>1.85</td><td>1.81</td><td>1.82</td><td>2.02</td><td>2.02</td><td>1.98</td><td>1.96</td><td>2.33</td><td>1.87</td><td>2.30</td><td>2.01</td></tr><tr><td>GDR-Net+π-Mechanism</td><td>1.83</td><td>1.82</td><td>1.67</td><td>1.73</td><td>1.80</td><td>1.83</td><td>1.83</td><td>1.88</td><td>2.04</td><td>1.68</td><td>2.19</td><td>1.83</td></tr></table>

![](images/99a68bb27a5ae6141e94a03b1d11e6e6e5a1042f1889371f0556c7372803748d.jpg)  
Fig. 9. Mean rotation errors with standard deviation (SD). The left bar is the mean rotation error and SD of the baseline method; The right bar is the mean rotation error and SD of the proposed mechanism.

Fig. 10 provides a qualitative comparison of challenging samples from the LINEMOD dataset. The visualization employs a consistent color scheme, with green representing the ground-truth, blue representing the baseline method, and red representing our approach. It can be observed that our method (red) consistently produces bounding boxes that are more accurately aligned with the ground-truth (green) geometry, particularly under the texture-less scene(Row 1). In contrast, the baseline predictions (blue) frequently exhibit noticeable rotational errors or instability. These visual results corroborate our quantitative findings, demonstrating that alignment to a stable, intrinsic coordinate system enables more robust and accurate rotation estimation under diverse challenging conditions.

To rigorously evaluate the robustness and reproducibility of our method beyond a single training run, we conducted a statistical experiment using G2L-Net [6] as the baseline on the

LINEMOD dataset. We performed five independent trials, each with random weight initialization and data shuffling. For each trial, we trained two models: the original G2L-Net (Baseline) and G2L-Net trained with our principal axes-aligned data (Ours). Fig. 9 presents the box plots of the mean rotation error across all 5 times for both configurations.

The results are statistically compelling. Our method achieves a mean rotation error of 7.14, with a standard deviation of 0.07. In contrast, the baseline yields a higher mean of 9.65 and a notably higher standard deviation of 0.18. This demonstrates that our approach not only decreases the rotation error by 26% but also significantly reduces the variance in outcomes by 61%. The reduced variance indicates that the trained models are more stable and less dependent on random initialization when learning a rotation representation anchored in the geometrically stable principal axes frame.

## D. Experiments on LINEMOD-OCC

To further evaluate robustness under severe occlusion, we test the proposed mechanism on the challenging LINEMOD-OCC dataset. We adopt LC (Linear Covariance) [14] as the baseline, which introduces a loss function enabling direct endto-end 6D pose regression from RGB images. By formulating the pose in Lie algebra, LC operates in the tangent space of the SO(3) rotation manifold, providing a geometrically consistent and stable learning signal that improves both accuracy and convergence for direct regression networks.

The results are summarized in Table III. With the integration of our π-Mechanism, pose accuracy measured by the ADD metric improves by 3.4% on average across all object categories. A notable case is the ‘Ape’ object, whose accuracy rises from 44.44% to 48.97%, corresponding to a 10.2% relative gain. These results confirm that our geometric alignment enhances pose estimation even under highly occluded conditions.

Fig. 11 presents a qualitative comparison on the challenging LINEMOD-OCC dataset with LC [14] as the baseline. We follow the scheme: ground-truth (green), LC baseline prediction (blue), and our method’s prediction (red). As shown, the LC baseline (blue) frequently produces bounding boxes with large rotational errors when the target object is partially obscured. In contrast, our method (red), leveraging the stable representation aligned with the object’s principal axes, consistently estimates poses that are much closer to the ground truth (green), even under severe occlusion. This visual evidence directly supports the quantitative improvements reported in TABLE III, demonstrating that our data-centric optimization significantly enhances robustness in real-world occlusion scenarios.

![](images/4facbf596bd2e5f2db2e46d1ec42ed33feb9118a603f278382f06a8c832813a6.jpg)  
Fig. 10. The Quantitative results for the LINEMOD dataset. The green boxes are the ground truth. The red boxes are our results. The blue boxes are the results of the baseline.

## E. Experiments on Symmetric Objects

To test the effectiveness of our π-Mechanism on symmetric objects, we pick up some typical symmetric objects from the widely used TLESS dataset (shown in Fig. 7). We then report the rotation errors of these objects in TABLE IV. From the table, we can see that the rotation errors drop dramatically with the optimized dataset.

## F. Category-level Tasks

Unlike instance-level tasks, category-level pose estimation is challenged by significant intra-class variation in shape and appearance. To validate the applicability of our proposed mechanism in this setting, we conduct experiments on the NOCS dataset [42].

A key adaptation for the category-level task is the computation of the Principal Axes of Inertia. Since objects within a category can have substantially different geometries, we compute the principal axes for each object instance independently. For example, the ”camera” category in NOCS contains three distinct instances with markedly different shapes, as illustrated in Fig. 12. During training, the respective principal axes for each instance are derived and used for canonical alignment. This instance-specific alignment ensures that our geometric normalization remains effective despite intra-class shape variation.

TABLE III  
R LINEMOD-OCCLUDE GDR-LC [14] ADD(-S)-0.1 ( ). T IS BOLDED. FOR SYMMETRIC OBJECTS ‘EGG BOX’ AND ‘GLUE’, THE ADD-S SCORES ARE REPORTED.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Ape</td><td rowspan=1 colspan=1>Can</td><td rowspan=1 colspan=1>Cat</td><td rowspan=1 colspan=1>Driller</td><td rowspan=1 colspan=1>Duck</td><td rowspan=1 colspan=1>Egg box</td><td rowspan=1 colspan=1>Glue</td><td rowspan=1 colspan=1>Hole Puncher</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>LCLC+π-M</td><td rowspan=1 colspan=1>44.44%48.97%</td><td rowspan=1 colspan=1>89.06%90.22%</td><td rowspan=1 colspan=1>49.87%51.39%</td><td rowspan=1 colspan=1>87.81%89.46%</td><td rowspan=1 colspan=1>56.08%57.31%</td><td rowspan=1 colspan=1>62.81%65.37%</td><td rowspan=1 colspan=1>68.88%70.86%</td><td rowspan=1 colspan=1>72.89%76.61%</td><td rowspan=1 colspan=1>66.69%68.99%</td></tr></table>

![](images/da2503f85da1cbb70902ca2ad402beedeccd49e5f1b23c28abc15881de8e93c8.jpg)

![](images/d08baa2c148f58f89f5fac803a9ef8b82e2573daabee9a40259d27a99eac684d.jpg)

![](images/963fd5a0ba97851ddaf8ac0aadcf21200a415a3e9565bef9de94a075ee56ebab.jpg)

![](images/f5d853d33afe0acbb04739f03bea4b9bae9714bfa1ab6a95a57936359e6c96ef.jpg)  
Fig. 11. The Quantitative results for LINEMOD-OCC dataset. The green boxes are the ground truth. The red boxes are our results. The blue boxes are the results of the baseline.

TABLE IV  
ROTATION ERROR FOR DIFFERENT METHODS ON TLESS SYMMETRIC OBJECTS. THE BEST RESULT IS BOLDED.
<table><tr><td>Method</td><td>obj 14</td><td>obj 27</td><td>obj 29</td></tr><tr><td>G2L [50]</td><td>31.51</td><td>71.70</td><td>108.83</td></tr><tr><td>G2L + π-Mechanism</td><td>2.64</td><td>10.11</td><td>17.79</td></tr></table>

![](images/db8bc42b277252817e0dbdeced76785eab778d7db5d8ac5c6cfc68b036c9aa69.jpg)  
Fig. 12. ‘Camera’ category with different instances

For the category-level pose estimation experiments, we adopt the FS-Net pipeline as our baseline. Following the evaluation protocol in [18], we use the geodesic error to assess different rotation representations. The quantitative results are presented in Table V.

As shown in Table V, our proposed π-Mechanism achieves the best performance for rotation estimation. It reduces the mean rotation error from 11.27 to 8.92—a relative improvement of 19.6% over the baseline. This result confirms that our geometry-driven alignment generalizes effectively to the category-level setting, where handling significant intra-class shape variation is critical.

## G. Convergence and Learning Dynamics

To move beyond empirical metrics and understand the fundamental mechanism behind our performance gains, we investigate how principal axes alignment reshapes the underlying learning problem. We analyze three interconnected aspects: optimization convergence, feature representation, and loss landscape geometry, all evaluated using the same network architecture where the only variable is the coordinate representation (original vs. aligned).

We monitor the training dynamics. As shown in Fig. 13, the rotation loss for networks trained on our aligned data converges to a lower plateau compared to training on original data. This indicates that regressing rotations relative to the stable principal axes constitutes a simpler and more well-posed learning objective.

To summarize, by aligning the training data to the geometrically stable principal axes, we reparameterize the rotation estimation task into an intrinsically ”easier” space. This reformulation results in a lower loss landscape, provides more stable gradients, and enables the network—without any architectural changes—to learn more coherent and robust feature representations. This fundamental improvement in learning dynamics directly explains the consistent accuracy gains observed across diverse networks and datasets.

TABLE V  
ROTATION ERROR COMPARISON. RESULTS ON DATASET NOCS-REAL FOR METHOD FS-NET [6] IN ROTATION PREDICTION ERROR. THE BEST RESULT IS BOLDED.
<table><tr><td></td><td>Category</td><td>Bottle</td><td>Bowl</td><td>Can</td><td>Camera</td><td>Laptop</td><td>Mug</td><td>Average</td></tr><tr><td rowspan="2">different representation</td><td>FS-Net</td><td>5.71</td><td>3.85</td><td>3.97</td><td>21.59</td><td>8.00</td><td>13.77</td><td>9.48</td></tr><tr><td>FS-Net+π-Mechanism</td><td>5.34</td><td>3.64</td><td>3.58</td><td>20.99</td><td>7.57</td><td>12.40</td><td>8.92</td></tr></table>

![](images/42fd3b2959ee56a74fa9b59d06401aceb8d7a9e6113a263f1e2fd64a51f01048.jpg)

![](images/a251a114fd9a92019bda32cc0fcc8b4900e5a8568bf399767919108a30bac7cc.jpg)  
Fig. 13. Loss value comparison. Left: the loss value comparison between original datasets and datasets in Principal Axes coordinates. Right: the loss ratio between original and principal over training iterations.

## V. CONCLUSION

In this work, we have introduced a fundamental shift in perspective for 6D object pose estimation: from modifying deep networks to optimizing their geometric input. Our proposed method aligns object coordinates with their intrinsic Principal Axes, establishing a canonical, physicsinformed rotation representation that is inherently more stable. Extensive experimentation—from synthetic poses to occluded real-world scenes, and from instance-level to category-level tasks—demonstrates that this data-centric, model-agnostic optimization consistently and significantly improves accuracy across diverse architectures. The core of our contribution is threefold. First, we provide a geometrically principled solution that directly addresses the instability of arbitrary object coordinates, thereby enhancing robustness against noise and occlusion. Second, we propose an analytical yet straightforward method to resolve ambiguous rotation labels for symmetric objects. Third, we establish a plug-and-play paradigm that decouples geometric optimization from network design, enabling immediate performance gains without retraining or modifying existing models. This work not only presents an effective mechanism but also opens a new, promising direction for advancing pose estimation through geometric and datacentric reasoning, potentially influencing other vision tasks reliant on stable 3D representations.

## REFERENCES

[1] E. Marchand, H. Uchiyama, and F. Spindler, “Pose estimation for augmented reality: a hands-on survey,” IEEE transactions on visualization and computer graphics, vol. 22, no. 12, pp. 2633–2651, 2016.

[2] M. Zhu, K. G. Derpanis, Y. Yang, S. Brahmbhatt, M. Zhang, C. Phillips, M. Lecce, and K. Daniilidis, “Single image 3d object detection and pose estimation for grasping,” in Robotics and Automation (ICRA), 2014 IEEE International Conference on. IEEE, 2014, pp. 3936–3943.

[3] J. Tremblay, T. To, B. Sundaralingam, Y. Xiang, D. Fox, and S. Birchfield, “Deep object pose estimation for semantic robotic grasping of household objects,” arXiv preprint arXiv:1809.10790, 2018.

[4] Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, “Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes,” arXiv preprint arXiv:1711.00199, 2017.

[5] Y. He, W. Sun, H. Huang, J. Liu, H. Fan, and J. Sun, “Pvn3d: A deep point-wise 3d keypoints voting network for 6dof pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 632–11 641.

[6] W. Chen, X. Jia, H. J. Chang, J. Duan, S. Linlin, and A. Leonardis, “Fs-net: Fast shape-based network for category-level 6d object pose estimation with decoupled rotation mechanism,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021.

[7] Y. Weng, H. Wang, Q. Zhou, Y. Qin, Y. Duan, Q. Fan, B. Chen, H. Su, and L. J. Guibas, “Captra: Category-level pose tracking for rigid and articulated objects from point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 13 209–13 218.

[8] K. Chen and Q. Dou, “Sgpa: Structure-guided prior adaptation for category-level 6d object pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 2773–2782.

[9] M.-F. Li, X. Yang, F.-E. Wang, H. Basak, Y. Sun, S. Gayaka, M. Sun, and C.-H. Kuo, “Ua-pose: Uncertainty-aware 6d object pose estimation and online object completion with partial references,” in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), June 2025, pp. 1180–1189.

[10] X. Lin, W. Yang, Y. Gao, and T. Zhang, “Instance-adaptive and geometric-aware keypoint learning for category-level 6d object pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 040–21 049.

[11] J. Huang, H. Yu, K.-T. Yu, N. Navab, S. Ilic, and B. Busam, “Matchu: Matching unseen objects for 6d pose estimation from rgb-d images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 10 095–10 105.

[12] B. Wen, W. Yang, J. Kautz, and S. Birchfield, “Foundationpose: Unified 6d pose estimation and tracking of novel objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 17 868–17 879.

[13] W. Chen, J. Duan, H. Basevi, H. J. Chang, and A. Leonardis, “Ponitposenet: Point pose network for robust 6d object pose estimation,” in The IEEE Winter Conference on Applications of Computer Vision (WACV), March 2020.

[14] F. Liu, Y. Hu, and M. Salzmann, “Linear-covariance loss for end-toend learning of 6d pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 14 107–14 117.

[15] T.-C. Hsiao, H.-W. Chen, H.-K. Yang, and C.-Y. Lee, “Confronting ambiguity in 6d object pose estimation via score-based diffusion on se(3),” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 352–362.

[16] H. Zhao, S. Wei, D. Shi, W. Tan, Z. Li, Y. Ren, X. Wei, Y. Yang, and S. Pu, “Learning symmetry-aware geometry correspondences for 6d object pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 14 045– 14 054.

[17] N. Mo, W. Gan, N. Yokoya, and S. Chen, “Es6d: A computation efficient and symmetry-aware 6d pose regression framework,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 6718–6727.

[18] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 5745–5753.

[19] S. Peng, Y. Liu, Q. Huang, X. Zhou, and H. Bao, “Pvnet: Pixelwise voting network for 6dof pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4561–4570.

[20] Y. Li, G. Wang, X. Ji, Y. Xiang, and D. Fox, “DeepIM: Deep Iterative Matching for 6D Pose Estimation,” 2018. [Online]. Available: http://arxiv.org/abs/1804.00175

[21] C. Wang, D. Xu, Y. Zhu, R. Martin-Martin, C. Lu, L. Fei-Fei, and S. Savarese, “Densefusion: 6d object pose estimation by iterative dense fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[22] Z. Zhang, W. Chen, L. Zheng, A. Leonardis, and H. J. Chang, “Trans6D: Transformer-Based 6D Object Pose Estimation and Refinement,” in Computer Vision – ECCV 2022 Workshops, L. Karlinsky, T. Michaeli, and K. Nishino, Eds. Cham: Springer Nature Switzerland, 2023, pp. 112–128.

[23] L. Feng, M. Xu, L. Wen, and Z. Shen, “Gatedunipose: A novel approach for pose estimation combining unireplknet and gated convolution,” 2024. [Online]. Available: https://arxiv.org/abs/2409.07752

[24] X. Ding, Y. Zhang, Y. Ge, S. Zhao, L. Song, X. Yue, and Y. Shan, “Unireplknet: A universal perception large-kernel convnet for audio video point cloud time-series and image recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 5513–5524.

[25] Y. Chen and G. Medioni, “Object modelling by registration of multiple range images,” Image and vision computing, vol. 10, no. 3, pp. 145–155, 1992.

[26] Y. Li, G. Wang, X. Ji, Y. Xiang, and D. Fox, “Deepim: Deep iterative matching for 6d pose estimation,” in The European Conference on Computer Vision (ECCV), September 2018.

[27] G. Wang, F. Manhardt, F. Tombari, and X. Ji, “GDR-Net: Geometryguided direct regression network for monocular 6d object pose estimation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 16 611–16 621.

[28] S. Iwase, X. Liu, R. Khirodkar, R. Yokota, and K. M. Kitani, “Repose: Fast 6d object pose refinement via deep texture rendering,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 3303–3312.

[29] L. Lipson, Z. Teed, A. Goyal, and J. Deng, “Coupled iterative refinement for 6d multi-object pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[30] L. Zheng, T. H. E. Tse, C. Wang, Y. Sun, H. Chen, A. Leonardis, W. Zhang, and H. J. Chang, “Georef: Geometric alignment across shape variation for category-level object pose refinement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 10 693–10 703.

[31] J. Chen, Y. Yin, T. Birdal, B. Chen, L. J. Guibas, and H. Wang, “Projective Manifold Gradient Layer for Deep Rotation Regression,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). New Orleans, LA, USA: IEEE, Jun. 2022, pp. 6636–6645. [Online]. Available: https://ieeexplore.ieee.org/document/ 9878614/

[32] Y. Di, R. Zhang, Z. Lou, F. Manhardt, X. Ji, N. Navab, and F. Tombari, “Gpv-pose: Category-level object pose estimation via geometry-guided point-wise voting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 6781– 6791.

[33] Y. Yuyang, L. Wen, A. Sheng, X. Qingshan, Y. Shangshu, Y. Guo, Z. Yin, S. Siqi, and C. Wang, “Raloc: Enhancing outdoor lidar localization via rotation awareness,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025.

[34] G. Pitteri, M. Ramamonjisoa, S. Ilic, and V. Lepetit, “On object symmetries and 6d pose estimation from images,” in 2019 International Conference on 3D Vision (3DV). IEEE, 2019, pp. 614–622.

[35] T. Hodan, D. Barath, and J. Matas, “Epos: Estimating 6d pose of objects with symmetries,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[36] C. C. Pagano and M. T. Turvey, “Eigenvectors of the inertia tensor and perceiving the orientation of a hand-held object by dynamic touch,” Perception & Psychophysics, vol. 52, no. 6, pp. 617–24, 1992.

[37] Pagano, C. Christopher, Turvey, and T. M., “The inertia tensor as a basis for the perception of limb orientation.” Journal of Experimental Psychology, 1995.

[38] S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit, “Gradient response maps for real-time detection of textureless objects,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 5, pp. 876–888, 2012.

[39] T. Hodan, P. Haluza, S. Obdr<sup>ˇ</sup> zˇalek, J. Matas, M. Lourakis, and X. Zab-´ ulis, “T-less: An rgb-d dataset for 6d pose estimation of texture-less objects,” in 2017 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2017, pp. 880–888.

[40] S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, and N. Navab, “Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes,” in Asian conference on computer vision. Springer, 2012, pp. 548–562.

[41] E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, and C. Rother, “Learning 6d object pose estimation using 3d object coordinates,” in The European Conference on Computer Vision (ECCV). Springer, 2014, pp. 536–551.

[42] H. Wang, S. Sridhar, J. Huang, J. Valentin, S. Song, and L. J. Guibas, “Normalized object coordinate space for category-level 6d object pose and size estimation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 2642–2651.

[43] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (CVPR), July 2017.

[44] A. Bahri, M. Yazdanpanah, M. Noori, S. Dastani, M. Cheraghalikhani, G. A. V. Hakim, D. Osowiechi, F. Beizaee, I. Ben Ayed, and C. Desrosiers, “Spectral informed mamba for robust point cloud processing,” in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), June 2025, pp. 11 799–11 809.

[45] C. R. Qi, W. Liu, C. Wu, H. Su, and L. J. Guibas, “Frustum pointnets for 3d object detection from rgb-d data,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[46] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph cnn for learning on point clouds,” ACM Transactions on Graphics (TOG), 2019.

[47] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in Advances in Neural Information Processing Systems, 2017, pp. 5099–5108.

[48] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[49] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in International Conference on Machine Learning (ICML), 2024.

[50] W. Chen, X. Jia, H. J. Chang, J. Duan, and A. Leonardis, “G2l-net: Global to local network for real-time 6d pose estimation with embedding vector features,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.