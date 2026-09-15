# G-ray: Ray-Level Relative Geometric Position Encoding in Multi-View Vision Transformers under Camera Heterogeneity

Shuo Zhang , Student Member, IEEE, Xin Su , Member, IEEE, Wei Wang , Member, IEEE,

Jun Liu , Xinrui Zeng , Yongsen Chen , Chenjie Wang , Guibo Zhu ,

Jinqiao Wang , Member, IEEE, Bin Luo , Senior Member, IEEE, and Liangpei Zhang , Fellow, IEEE

![](images/549c341c90a214ff95ff88148e421435ec3b5d619b1f17630c9738f04dac4367.jpg)  
Fig. 1. Overview of G-ray under camera heterogeneity. Top, representative configurations include (a) distributed camera collections, (b) rigid multi-camera rigs, and (c) variable-FoV cameras. Middle, G-ray uses camera-local ray angles as positional coordinates across FoVs and projection models, illustrated by horizontal- and vertical-angle maps. Bottom, qualitative examples show 16-view 3D reconstruction on ScanNet++ from mixed non-pinhole DSLR and pinhole iPhone inputs, and NVS on RE10K under controlled FoV variation (z=2.0), with per-example PSNR values in dB. The top-row configurations illustrate application settings, not the capture setups of the bottom-row examples.

Abstract—We study relative position encoding for multi-view vision Transformers under camera heterogeneity, including varying fields of view (FoVs) or projection models. Existing rotary relative position encodings commonly use image-plane positional coordinates, producing projection-dependent relative phases and inconsistent geometric cues for cross-projection attention. We introduce G-ray, a ray-level relative position encoding whose rotary phases are parameterized by camera-local ray angles. The same camera-local ray pair induces the same relative phase across projections, providing projection-invariant positional consistency. G-ray can be used directly or integrated with existing encodings, retaining complementary geometric cues without additional learned parameters. We validate G-ray in three host encodings, RoPE, GTA, and RayRoPE, across 3D reconstruction and novel-view synthesis (NVS). Across three heterogeneous 3D reconstruction benchmarks at 50 views, Gray leads all six averaged metrics and reduces mean pointmap relative error by 45.8% over MapAnything, with calibration supplied to both. Trained exclusively on homogeneous pinhole images, the 3D reconstruction model handles mixed pinhole and non-pinhole inputs without retraining and remains competitive on homogeneous pinhole 3D reconstruction protocols. For NVS, GTA and RayRoPE improve with G-ray under joint viewpoint and FoV variation. The project’s webpage is available at https: //g-ray-project.github.io/.

Index Terms—Position encoding, multi-view vision, camera heterogeneity, 3D reconstruction, novel-view synthesis.

## I. INTRODUCTION

M <sup>ULTI-VIEW</sup> <sup>3D</sup> <sup>perception</sup> <sup>is</sup> <sup>a</sup> <sup>foundational</sup> <sup>capabil-</sup>ity for embodied intelligence, augmented reality, au- ity for embodied intelligence, augmented reality, autonomous driving, and mobile mapping, where a system needs to build spatially grounded representations of its surroundings from observations acquired across viewpoints [1]–[4]. To extend coverage, reduce blind spots, and capture complementary near- and far-field detail, practical systems often use a variety of camera configurations. Three recurring configurations illustrate this diversity (Fig. 1(a)–(c)). Distributed camera collections combine independently acquired views, potentially mixing devices, FoVs, and projection models [5]–[11]. Rigid multi-camera rigs maintain fixed relative camera poses while combining different FoVs or projection models [2], [3], [12], [13]. Variable-FoV cameras, including continuous-zoom and pan-tilt-zoom units, vary their angular coverage over time [5], [6], [14]. We use camera heterogeneity to describe such configurations, where imaging geometry such as FoV or projection model varies across views, in contrast to homogeneous input whose views share a single imaging geometry [15], [16]. Any perception model built on heterogeneous input must therefore integrate views that differ not only in where they observe the scene but also in how 3D viewing rays are mapped onto image locations.

Classical geometric formulations use calibrated camera models [17], [18] to convert image measurements from different projections into viewing rays. This ray-based representation thus can accommodate camera heterogeneity during cross-view geometric reasoning [19], [20]. The limitations of these formulations lie elsewhere. They rely on computationally expensive iterative optimization, and their accuracy can degrade under weak texture, dynamic content, and sparse views [21], [22]. This cost and brittleness motivated the shift to feed-forward multi-view vision Transformers, which replace staged geometric optimization with a single learned system [21], [23]–[30]. The shift, however, also discarded the explicit camera model. Early designs in this family inherit the positional encoding of language and single-image vision Transformers, which tells a token where it sits within its own image but nothing about the camera that produced it [31]–[37]. The problem of camera heterogeneity is therefore not resolved but dropped from the formulation.

Subsequent feed-forward methods restore camera awareness through two families of positional encodings. Absolute camera encodings, including Plucker-ray embeddings [30], [38], [39],¨ 3D point positional encoding [40], and CamRay tokens [29], attach viewing geometry to individual tokens by addition or concatenation. Because geometry is encoded per token rather than per token pair, cross-view relative structure must be inferred indirectly from absolute features, without an explicit guarantee of consistent positional behavior across viewpoints [33], [41], [42]. Relative camera encodings instead incorporate camera parameters into token-pair interactions [43], [44]. GTA and PRoPE combine camera-transform components with rotary image-grid subspaces [41], [42], while projectionbased rotary encodings such as RayRoPE, URoPE, and CRePE obtain rotary coordinates by projecting geometric quantities into a query camera [45]–[47]. Although these approaches introduce geometric information into relative attention, their grid-index or projected-pixel components still form relative phases from image-plane positional coordinates. In the lowerleft panel of Fig. 2, wide- and tele-FoV projections map the same pair of viewing rays to different image-plane displacements. These displacements induce unequal relative phases even though the underlying ray relation is unchanged, making the positional treatment projection-dependent. We refer to this effect as cross-projection phase mismatch, a source of inconsistent geometric cues for cross-projection attention.

![](images/711565a1e282cf63516498d61d1a8d43cac4139956425d5f1d7f995299c4e666.jpg)  
Fig. 2. Cross-projection phase mismatch and ray-angle consistency. Wide- and tele-FoV projections map the same camera-local ray pair to different imageplane displacements, yielding unequal relative phases (left). G-ray instead uses ray-angle coordinate differences, so the same ray pair induces the same relative phase across projections (right).

We address this mismatch with G-ray, a ray-level relative position encoding that lifts the positional coordinate from the image plane to ray-angle space. G-ray converts each positional sample to a camera-local ray using a view-specific image-to-ray map and represents it by horizontal and vertical angles $( \theta _ { x } , \theta _ { y } )$ . This conversion expresses viewing direction in radians, giving the positional coordinates a shared physical meaning across projections. In the lower-right panel of Fig. 2, the wide- and tele-FoV observations recover the same angular separation for the same ray pair and therefore the same relative phase. G-ray thus provides projection-invariant positional consistency, with relative phases determined by camera-local rayangle coordinate differences rather than projection-dependent image-plane displacements.

G-ray defines ray-angle rotary transformations that can serve directly as a relative position encoding or be integrated into existing geometric encodings (Fig. 4). Integration replaces the host’s grid-index or projected pixel coordinates with ray-angle pairs while retaining its complementary geometric components, such as depth, disparity, pose, and camera-center information. The image-to-ray map $\mathcal { U } _ { c }$ is derived from intrinsics for pinhole views and supplied by a calibrated projection model or unit-ray map for non-pinhole views. An external estimator can also supply this geometry when calibration is unavailable. The conversion itself introduces no learned parameters and requires neither token depth nor camera extrinsics, although the host encoding may still use both.

We test G-ray in 3D reconstruction and novel-view synthesis (NVS) using three host encodings. For 3D reconstruction, we use G-ray directly as the ray-angle rotary position encoding in a ViT-based model, replacing grid-index RoPE. For NVS, we replace either GTA’s image-grid coordinates [41] or RayRoPE’s query-camera projected pixels [45] within the same LVSM backbone [30], retaining each host encoding’s remaining geometric components. These instantiations test whether shared ray-angle coordinates benefit both grid-based and projection-based rotary formulations across tasks under camera heterogeneity. Notably, the 3D reconstruction model trained exclusively on homogeneous pinhole inputs handles mixed pinhole and non-pinhole inputs without retraining.

On heterogeneous 3D reconstruction across PandaSet, ETH3D under controlled FoV variation, and mixed-projection ScanNet++, G-ray leads all six dataset-averaged metrics at 50 views among methods evaluated on all three benchmarks. On mixed-projection ScanNet++ at 50 views, G-ray reduces both pointmap and ray-depth relative errors by about 74% compared with MapAnything when both use supplied calibration, while increasing point inliers from 26.2% to 75.1%. In NVS, integrating G-ray into GTA improves all three main Objaverse regimes and substantially improves RE10K results under controlled FoV variation. For RayRoPE, the largest gain among the three main Objaverse regimes occurs at novel viewpoints with additional FoV change. Further experiments show more gradual degradation above the training resolution, effective 3D reconstruction with AnyCalib-estimated calibration [48], and competitive accuracy on homogeneous pinhole 3D reconstruction protocols.

Our contributions are as follows.

• We introduce G-ray, a direct ray-level angular encoding that gives relative phase a shared geometric interpretation and provides projection-invariant positional consistency. This encoding addresses the cross-projection phase mismatch we identify in rotary relative position encodings based on image-plane coordinates (§III).

• We formulate G-ray for direct use as a ray-angle rotary position encoding and for integration with existing geometric encodings. Instantiations in RoPE, GTA, and RayRoPE cover grid-based and projection-based formulations while retaining each host encoding’s complementary geometry and adding no learned parameters (§III-E).

• We validate G-ray in 3D reconstruction and NVS under camera heterogeneity. 3D reconstruction evaluation spans distributed camera collections (ScanNet++), rigid multi-camera rigs (PandaSet), and variable-FoV inputs (ETH3D under controlled FoV variation). G-ray leads all six dataset-averaged metrics at 50 views and achieves the lowest ray-direction error in each setting. In NVS, both GTA and RayRoPE improve with G-ray under joint viewpoint and FoV variation (§IV).

• We demonstrate pinhole-to-non-pinhole transfer in 3D reconstruction, applying the same model trained on homogeneous pinhole inputs to heterogeneous views at inference without retraining. We further evaluate G-ray for resolution extrapolation and 3D reconstruction with calibration estimated from the input images (§IV).

## II. RELATED WORK

## A. Camera Heterogeneity in 3D Perception

Camera heterogeneity is not confined to a particular task. It arises when input views use different FoVs or projection models, whether acquired by multiple devices or a single variable-FoV camera. Distributed camera collections combine independently acquired views that may differ in device, FoV, and projection model, as in crowd-sourced 3D reconstruction and scene understanding [7]–[9]. Rigid multi-camera rigs maintain fixed relative camera poses while combining different FoVs or projection models, as in autonomous driving and egocentric sensing [2], [3], [12]. Variable-FoV cameras, including continuous-zoom and pan-tilt-zoom units, change their angular coverage over time [5], [6], [14]. Surveillance tasks involve detection and tracking across camera networks with overlapping or non-overlapping FoVs, using static, pantilt-zoom, and omnidirectional cameras [6], [14], [49]. We use camera heterogeneity for variation in imaging geometry within one multi-view input, which may also involve viewpoint changes.

Recent learning-based systems address selected portions of this design space. PanoVGGT reconstructs panoramic imagery [16], UniDAC estimates metric depth across camera models [15], X-Lens combines ray maps with projectionaware features for mixed pinhole and fisheye rigs [13], and UAVFF3D studies feed-forward 3D reconstruction under changes in aerial-view geometry [4]. These works demonstrate the value of projection-aware modeling, but each is tied to a particular task, camera family, or complete system. Gray instead asks whether heterogeneous views can share one relative positional coordinate, and evaluates that coordinate in both 3D reconstruction and NVS.

## B. Classical and Feed-Forward Visual Geometry

Classical image-based geometry resolves the view-specific projection explicitly when combining observations. Structurefrom-motion and multi-view stereo estimate correspondences, cameras, and scene structure through triangulation and geometric optimization [19], [20]. Per-scene NVS methods similarly cast calibrated pixels into camera rays and optimize a radiance field or Gaussian scene representation [50], [51]. Calibrated multi-camera detection and tracking systems also use projection geometry to associate observations across views [6], [14]. When an appropriate camera model is supplied, these formulations can unproject pinhole, fisheye, or zoom observations into a common geometric space. Their generality, however, often depends on reliable calibration and correspondence, staged processing, iterative optimization, or per-scene fitting.

Feed-forward models replace much of this processing with learned cross-view aggregation. For 3D reconstruction, DUSt3R introduced pairwise pointmap regression [23], MASt3R added dense matching features [24], and MUSt3R generalized the formulation to multiple views [25]. VGGT jointly predicts cameras, depth, pointmaps, and tracks [21], while VGGT-Ω, π<sup>3</sup>, Fast3R, Depth Anything 3, and MapAnything extend robustness, scale, equivariance, or optional geometric conditioning [22], [26]–[29]. For NVS, pixelSplat and MVSplat predict Gaussian scene representations [52], [53], whereas LVSM directly maps context views to targetview appearance with minimal explicit 3D inductive bias [30]. Multi-view Transformers have also advanced camera-only 3D detection and temporal tracking through 3D queries, positionaware image features, bird’s-eye-view aggregation, and objectcentric temporal propagation [54]–[57]. These models make visual geometry substantially faster and more unified, but most are trained and evaluated with a shared perspective projection or a homogeneous camera rig. This leaves open which coordinate should parameterize relative position when one attention layer receives tokens formed under different projection geometries.

## C. Positional Encoding for Vision Transformers

Self-attention is permutation equivariant and therefore requires an explicit source of positional structure [31], [32]. Absolute encodings associate each token with a learned or fixed positional feature. Relative encodings instead parameterize token-pair relations, including window-relative biases and the directional, context-dependent offsets of iRPE [35], [36]. Conditional positional encoding generates local position cues from the input neighborhood and improves extrapolation to unseen image sizes [37], and ViTAR trains with perturbed positions so that a model tested above its training resolution encounters positional values it has effectively already seen [58]. RoPE applies position-dependent orthogonal rotations so that querykey products depend on relative displacement [33], and its two-dimensional variants extend this mechanism to image lattices [34].

Most image positional encodings are designed around a regular two-dimensional lattice. They improve locality, translation structure, or resolution extrapolation while presuming that positions from different images share the same image-plane metric. Multi-view rotary coordinates inherit this assumption through either token-grid indices, as in standard 2D RoPE and the spatial subspace of GTA, or pixels obtained by projection into a query camera, as in RayRoPE. The two sources differ computationally, but both remain tied to projection-specific image coordinates. G-ray changes the coordinate supplied to the relative rotation from an image-plane pair to camera-local horizontal and vertical ray angles.

## D. Camera-Aware Encoding in Multi-View Transformers

Camera-aware encoding in multi-view Transformers falls into two families that parallel the absolute-versus-relative distinction in standard positional encoding.

a) Absolute camera encodings: attach viewing geometry to individual tokens through addition, concatenation, or adapter-based conditioning. Representative designs encode 3D points or Plucker rays as token-aligned features [38], [40],¨ MapAnything represents optional calibration through Cam-Ray inputs [29], the released Pi3X extension of $\pi ^ { \mathrm { 5 } }$ accepts optional poses, intrinsics or rays, and depth [27],<sup>1</sup> WorldMirror incorporates camera and depth priors through any-prior prompting [59], and OmniVGGT uses a GeoAdapter to inject arbitrary subsets of geometric modalities [60]. Camera-aware generation methods similarly encode per-token Plucker rays,¨ depth-aware projected paths, or relative ray and orientation cues [39], [44], [47]. Because geometry is carried per token rather than expressed in pairwise attention, cross-view relative structure must be inferred indirectly from individual token features, without an explicit guarantee of consistent positional behavior across viewpoints.

b) Relative camera encodings: incorporate camera parameters into the token-pair interaction itself. GTA combines a camera-transformation subspace with a regular image-grid subspace [41], PRoPE models relative viewing-frustum relationships [42], and DPPE factorizes camera transformations for scalable multi-view attention [43]. RayRoPE projects a depth-parameterized point on each ray into a query camera and includes its query-frame pixel coordinate and disparity in the rotary position [45]. URoPE lifts key-view pixels to multiple depth anchors, projects them into the query view, and assigns the resulting positions to attention heads [46]. Although these approaches introduce geometric information into relative attention, their image-plane rotary components remain projectiondependent, so cross-projection phase mismatch persists.

G-ray isolates a narrower common component in this landscape. It replaces only the projection-dependent coordinate pair used by a relative rotary encoding with camera-local ray angles, while retaining complementary ray features, pose transformations, disparity, uncertainty, or depth-dependent branches of the surrounding method.

## III. RAY-LEVEL RELATIVE GEOMETRIC POSITION ENCODING

G-ray replaces image-plane positional coordinates with camera-local ray-angle coordinates while retaining the host encoding’s complementary geometry and surrounding architecture. Fig. 3 organizes the resulting computation into three stages. An image-to-ray interface maps each image-plane positional sample to a camera-local viewing ray, in closed form for a pinhole camera and through a per-pixel unitray map for a non-pinhole camera (Fig. 3A). Horizontal and vertical ray angles form patch-aligned coordinate maps and parameterize each token’s multi-frequency block rotation (Fig. 3B). The rotations are applied to queries, keys, and values before attention, followed by the inverse query rotation on the output (Fig. 3C). The image-to-ray conversion is view-specific, while the ray-angle parameterization and rotary construction are shared across projections. Sec. III-A identifies the positional variable we replace, Sec. III-B defines the ray-angle coordinate, and Sec. III-C defines the attention computation. Sec. III-D explains the resulting properties and calibration requirement, with a qualitative diagnostic of positional consistency. Sec. III-E describes the replacement in three host encodings.

## A. Problem Formulation

We consider a rotary encoding parameterized by a twodimensional positional coordinate. Let a multi-view input contain images $\{ \mathbf { I } _ { c } \} _ { c = 1 } ^ { C } .$ , and let token i of view c carry a feature $\mathbf { x } _ { c , i }$ and a positional coordinate $\pmb { \xi } _ { c , i } ~ \in ~ \mathbb { R } ^ { 2 }$ . The encoding assigns each token an orthogonal transformation $\mathbf { E } ( \xi _ { c , i } )$ , yielding the query-key interaction

![](images/96f151876bd9985bfd2db201282afc4336f578a52c0d086cb55f75399184ee05.jpg)  
Fig. 3. G-ray from image-to-ray geometry to ray-level relative attention. (A) View-specific unprojection and normalization convert an image-plane positional sample to a camera-local unit ray, using pinhole intrinsics or a non-pinhole unit-ray map. (B) Horizontal and vertical angles, computed with atan2, form ray-angle coordinate maps and parameterize multi-frequency block rotations. (C) Queries, keys, and values are rotated before attention, followed by the inverse query rotation on the output.

$$
\left( \mathbf { E } ( \pmb { \xi } _ { c , i } ) \mathbf { q } _ { c , i } \right) ^ { \top } \left( \mathbf { E } ( \pmb { \xi } _ { d , j } ) \mathbf { k } _ { d , j } \right) = \mathbf { q } _ { c , i } ^ { \top } \mathbf { E } ( \pmb { \xi } _ { c , i } ) ^ { \top } \mathbf { E } ( \pmb { \xi } _ { d , j } ) \mathbf { k } _ { d , j } .\tag{1}
$$

The positional effect of this rotary component depends on the coordinate pair $( \xi _ { c , i } , \xi _ { d , j } )$ . We study which coordinates should parameterize it under camera heterogeneity.

Existing encodings supply $\boldsymbol { \xi }$ as token-grid indices or querycamera projected pixel coordinates, both defined on the image plane. For a zero-skew pinhole camera, a ray with horizontal angle $\theta _ { x }$ projects to $u = c _ { x } + f _ { x }$ tan $\theta _ { x }$ . Consequently, equal pixel displacements can represent different ray-angle differences, both across focal lengths and between the center and periphery of one image. Non-pinhole cameras, such as fisheye cameras, map ray directions to image coordinates differently, so this correspondence also depends on the projection model. Relative phases derived from these projection-dependent displacements therefore provide inconsistent geometric cues for cross-projection attention.

G-ray instead expresses position through viewing direction, using angular units shared across projections. We therefore replace $\xi _ { c , i }$ with a camera-local ray-angle coordinate $\theta _ { c , i }$ before the rotary phase is formed. The conversion uses the view-specific image-to-ray map

$$
\mathcal { U } _ { c } : \Omega _ { c } \to \mathbb { R } ^ { 3 } \setminus \{ \mathbf { 0 } \} ,\tag{2}
$$

which sends a location in the valid image domain $\Omega _ { c }$ to a nonzero camera-frame direction. This reparameterization removes projection dependence from the positional phase. The host encoding’s complementary geometry, including pose conditioning, depth prediction, and camera-transform subspaces, is retained (Sec. III-E).

## B. Ray-Angle Coordinates

G-ray turns every image-plane positional sample into a pair of camera-local ray angles. Let $\mathbf { p } _ { c , i } = ( u _ { c , i } , v _ { c , i } , 1 ) ^ { \top }$ be a positional sample of token i in view c. Unprojection followed by normalization gives the unit viewing ray

$$
\mathbf { r } _ { c , i } = \frac { \mathcal { U } _ { c } ( \mathbf { p } _ { c , i } ) } { \Vert \mathcal { U } _ { c } ( \mathbf { p } _ { c , i } ) \Vert _ { 2 } } = ( r _ { x , c , i } , r _ { y , c , i } , r _ { z , c , i } ) ^ { \top } ,\tag{3}
$$

which we parameterize by its horizontal and vertical angles

$$
\theta _ { x , c , i } = \operatorname { a t a n 2 } \left( r _ { x , c , i } , r _ { z , c , i } \right) ,\tag{4}
$$

$$
\theta _ { y , c , i } = \operatorname { a t a n 2 } \left( r _ { y , c , i } , r _ { z , c , i } \right) .\tag{5}
$$

The ray-angle coordinate of the sample is $\pmb { \theta } _ { c , i } = ( \theta _ { x , c , i } , \theta _ { y , c , i } )$ shown as the two angle maps of Fig. 3B. The two-argument arctangent preserves quadrant information, with principal values in $( - \pi , \pi ]$ . The occupied angular range depends on the view’s FoV and projection model. For forward-facing rays, both angles lie in $( - \pi / 2 , \pi / 2 )$

The projection model enters through $\mathcal { U } _ { c } .$ , which is instantiated differently for pinhole and non-pinhole projections (Fig. 3A). A pinhole camera with intrinsics $\mathbf { K } _ { c }$ admits the closed form $\mathcal { U } _ { c } ( \mathbf { p } ) = \mathbf { K } _ { c } ^ { - 1 } \mathbf { p }$ . For zero skew, (4) and (5) reduce to $\theta _ { x } = \mathrm { a t a n 2 } ( u - c _ { x } , f _ { x } )$ and $\theta _ { y } = \mathrm { a t a n 2 } ( v - c _ { y } , f _ { y } )$ . For non-pinhole cameras, we represent $\mathcal { U } _ { c }$ as a per-pixel unit-ray map. This map can be computed from a parametric projection model, supplied by factory calibration, or predicted by an external estimator. Our mixed-projection evaluation uses the four-coefficient fisheye model of the ScanNet++ DSLR stream. This model maps a ray at polar angle ψ from the optical axis to the normalized radius

$$
\rho ( \psi ) = \psi \left( 1 + k _ { 1 } \psi ^ { 2 } + k _ { 2 } \psi ^ { 4 } + k _ { 3 } \psi ^ { 6 } + k _ { 4 } \psi ^ { 8 } \right) ,\tag{6}
$$

which we numerically invert to obtain $\psi$ and reconstruct the ray (Appendix A-A). The same angular parameterization in (4) and (5) then applies to both pinhole and non-pinhole views.

## C. Relative Angular Attention

The ray-angle coordinate enters attention through the standard rotary construction. For head dimension $d _ { h }$ , G-ray builds the block-diagonal transformation

$$
\mathbf { E } _ { c , i } = \mathbf { R } _ { \Omega } ( \theta _ { x , c , i } ) \oplus \mathbf { R } _ { \Omega } ( \theta _ { y , c , i } ) ,\tag{7}
$$

where ⊕ denotes block-diagonal concatenation and $\mathbf { R } _ { \Omega } ( \theta )$ applies a bank of planar rotations to disjoint channel pairs. We parameterize a geometric bank of angular frequencies as

$$
\omega _ { k } = \alpha \beta ^ { - k / n _ { \Omega } } ,\tag{8}
$$

in which α is a frequency scale, $\beta$ is a frequency base, and $n _ { \Omega }$ is the number of frequency components in one angular subspace, with $k = 0 , \ldots , n \Omega - 1$ . For axis $a \in \{ x , y \}$ , the rotary phase of token i in view c at frequency k is

$$
\varphi _ { c , i , k } ^ { a } = \omega _ { k } \theta _ { a , c , i } .\tag{9}
$$

Each channel pair is rotated using cos $\varphi _ { c , i , k } ^ { a }$ and sin $\varphi _ { c , i , k } ^ { a } ,$ so the positional coordinate and frequency jointly determine the phase supplied to attention. Ray angles and image-plane coordinates have different numerical scales, so α controls the range of rotation phases. Sec. IV-D2 ablates the frequency scale.

G-ray transforms queries, keys, and values and applies the inverse query transformation to the attention output

$$
\widetilde { \mathbf { q } } _ { c , i } = \mathbf { E } _ { c , i } \mathbf { q } _ { c , i } , \qquad \widetilde { \mathbf { k } } _ { d , j } = \mathbf { E } _ { d , j } \mathbf { k } _ { d , j } ,
$$

$$
\begin{array} { r } { \widetilde { \mathbf v } _ { d , j } = \mathbf E _ { d , j } \mathbf v _ { d , j } , } \end{array}\tag{10}
$$

(11)

$$
\widetilde { \mathbf { q } } _ { c , i } ^ { \top } \widetilde { \mathbf { k } } _ { d , j } = \mathbf { q } _ { c , i } ^ { \top } \mathbf { E } _ { c , i } ^ { \top } \mathbf { E } _ { d , j } \mathbf { k } _ { d , j } ,\tag{12}
$$

$$
\begin{array} { r } { \mathbf o _ { c , i } = \mathbf E _ { c , i } ^ { - 1 } \mathrm { { A t t e n t i o n } } ( \mathbf E _ { c , i } \mathbf q _ { c , i } , \{ \mathbf E _ { d , j } \mathbf k _ { d , j } , \mathbf E _ { d , j } \mathbf v _ { d , j } \} _ { d , j } ) . } \end{array}\tag{13}
$$

Because planar rotations compose by angle addition, the transpose-query rotation in (12) subtracts the query phase from the key phase. The relative phase for each axis and frequency is therefore

$$
\begin{array} { r l } & { \Delta \varphi _ { c i , d j , k } ^ { a } = \varphi _ { d , j , k } ^ { a } - \varphi _ { c , i , k } ^ { a } } \\ & { \qquad = \omega _ { k } \big ( \theta _ { a , d , j } - \theta _ { a , c , i } \big ) . } \end{array}\tag{14}
$$

For the same ray pair, Fig. 2 uses $\Delta \varphi _ { w }$ and $\Delta { { \varphi } _ { t } }$ to denote this relative phase under wide- and tele-FoV projections.

![](images/c59017c358f74baa70a53c92dd33d2f9e7f9efb5581ae0fd5a46c81fba5dddd3.jpg)  
Fig. 4. Integrating G-ray with existing rotary encodings. (A) Grid-index coordinates in RoPE or GTA, or query-camera projected-pixel coordinates in RayRoPE, are replaced by camera-local ray-angle coordinates. (B) Ray-angle differences determine relative phases within the existing attention interface, while the host encoding’s complementary geometric components are retained. Each variant is trained independently.

Thus, $\mathbf { E } _ { c , i } ^ { \top } \mathbf { E } _ { d , j }$ is parameterized by horizontal and vertical angular offsets rather than projection-dependent pixel offsets. These offsets compare camera-local coordinates, while any complementary camera-pose transformations remain part of the host encoding. The formulation in (11) to (13) supports within-view and cross-view attention in encoder and encoderdecoder architectures.

## D. Properties and Requirements

Replacing the positional coordinate changes what the rotary phase measures. We describe the resulting positional consistency and coordinate range, together with their calibration requirement.

1) Projection-Invariant Positional Consistency: An angular increment has a shared physical meaning across projections. The same camera-local ray-angle coordinate difference therefore produces the same relative phase, whereas equal imageplane displacements need not represent equal changes in viewing direction.

Fig. 5 illustrates this positional consistency on ETH3D image pairs under controlled FoV variation. Each narrow-FoV view is a cropped and resized version of its wide-FoV counterpart. Wide-view patch tokens serve as queries and narrow-view patch tokens as keys. The maps average attention weights over global-attention blocks and heads and display the wide-view responses, with a red rectangle marking the shared FoV. In the richly textured examples, all compared methods concentrate attention within the shared region. In the weakly textured examples, grid-index RoPE assigns more attention outside that region, whereas G-ray maintains a clearer concentration within it.

The rightmost column shows position-only (PE-only) priors to separate the positional contribution from image content. For rotary encodings, this probe assigns identical constant features to queries and keys before rotation. The relative-phase identity for a bank of planar rotations parameterized by ξ is

$$
\left( \mathbf { E } ( \pmb { \xi } _ { i } ) \mathbf { q } \right) ^ { \top } \left( \mathbf { E } ( \pmb { \xi } _ { j } ) \mathbf { k } \right) = \mathbf { q } ^ { \top } \mathbf { E } ( \pmb { \xi } _ { j } - \pmb { \xi } _ { i } ) \mathbf { k }\tag{15}
$$

![](images/fcbc0ccfdac9d08119e9e8df30568d97e83864d9f632ef2138f424b15e2d9315.jpg)  
Fig. 5. Cross-view attention under controlled FoV variation on ETH3D. The top two rows show wide-FoV images and their resized narrow-FoV crops. Red rectangles mark the shared FoV, and dashed lines link each region to its crop. Attention maps average wide-to-narrow responses over global-attention blocks and heads. The right column shows position-only priors (PE-only), computed for the rotary encodings using constant query and key features before rotation. In the weakly textured examples, G-ray concentrates attention more clearly within the shared FoV than grid-index RoPE. Its PE-only prior reflects angular alignment, whereas the grid-index prior is nearly uniform.

so the content-free interaction depends only on the coordinate difference $\Delta \pmb { \xi } = \pmb { \xi } _ { j } - \pmb { \xi } _ { i }$ . With constant features, it reduces to a sum of cosines of frequency-weighted coordinate differences (Appendix A-D). For grid-index RoPE, these differences are measured in token-grid indices, and the displayed prior is nearly uniform. For G-ray, $\xi ~ = ~ ( \theta _ { x } , \theta _ { y } )$ , so the prior instead reflects camera-local ray-angle differences and favors small angular offsets in the displayed examples. The multifrequency cosine interaction need not decrease monotonically with angular separation. These examples illustrate how rayangle coordinates provide a positional prior aligned with the shared FoV, while the learned query and key features also shape the final attention distribution.

2) Resolution-Independent Coordinate Range: The rayangle coordinate range does not grow with input resolution. At a fixed FoV, changing the resolution alters the sampling density of the angular domain but leaves its extent unchanged. After the image-to-ray map is updated, nearby token centers across resolutions correspond to nearby viewing rays and therefore similar ray-angle coordinates. Exactly corresponding continuous image locations retain identical rays, as shown in Appendix A-B. Token-grid indices, in contrast, span a larger range as resolution increases, requiring positional values outside the training range, a difficulty also studied in image classification [58]. G-ray avoids this coordinate-range extrapolation through ray-level angular coordinates, since a higherresolution input samples the same angular interval more finely. Sec. IV-E2 evaluates the complete network at up to twice the training resolution.

3) Calibration Requirement: These properties require viewspecific image-to-ray calibration. The angular map and the rotations are deterministic, so the encoding introduces no learned parameters and needs neither token depth nor camera extrinsics, although a host may keep such cues for its own purposes. The image-to-ray map $\mathcal { U } _ { c }$ is the interface to viewspecific imaging geometry. It can be obtained from pinhole intrinsics, a calibrated non-pinhole model such as (6), or an external ray-map estimator. The positional phases therefore depend on the accuracy of the supplied rays, regardless of how the map is obtained. We therefore quantify controlled calibration error in Sec. IV-E3 and off-the-shelf estimated image-to-ray geometry in Sec. IV-E4.

## E. Coordinate Replacement in Existing Encodings

G-ray replaces the positional coordinates supplied to a host encoding’s rotary interface. Fig. 4 summarizes the token-grid and projection-based cases. Grid-index RoPE uses a patch index pair $( m , n )$ . G-ray instead maps the corresponding patch center through $\mathcal { U } _ { c }$ and Eqs. (4) and (5) to obtain $( \theta _ { x } , \theta _ { y } )$ for the same rotary interface. A projection-based encoding obtains a pixel pair $( u _ { q } , v _ { q } )$ by projecting a geometric quantity into the query camera. G-ray converts this pair to query-camera ray angles while retaining the host’s other geometric components. When a host attaches several projected samples to one token, the same image-to-ray and ray-to-angle mapping is applied to each sample before the host’s own positional aggregation. Both paths therefore replace the image-plane positional coordinate rather than the geometric formulation built around it.

We evaluate one direct instantiation and two integrations with existing geometric encodings. In the 3D reconstruction model, G-ray serves directly as the rotary position encoding in place of grid-index RoPE, allocating equal channel subspaces to $\theta _ { x }$ and $\theta _ { y }$ . The model also applies the value and output transformations in (13), enabling matched comparisons with grid-index RoPE and absolute CamRay conditioning.

In LVSM [30] with GTA [41], head dimensions are split between camera-transform and image-plane grid subspaces. Gray replaces only the grid coordinates, retaining the cameratransform subspace and CamRay conditioning.

In LVSM with $\mathrm { R a y R o P E }$ [45], the rotary coordinates contain the query-frame camera center, query-camera projected pixels, and disparity. G-ray replaces only the projected pixel coordinates. Camera-center and disparity components, depth prediction, uncertainty treatment, and extrinsic projection are retained. Appendix B-C gives the explicit rotation blocks and frequency settings for these three host encodings.

## IV. EXPERIMENTS

We evaluate G-ray along four complementary dimensions. Sec. IV-B compares multi-view 3D reconstruction under camera heterogeneity and homogeneous pinhole 3D reconstruction protocols. Sec. IV-C evaluates transfer to novel view synthesis through coordinate replacement in two host encodings. Sec. IV-D examines individual design choices under matched training conditions. Sec. IV-E examines robustness to controlled FoV variation and resolution changes, calibration requirements, and computational cost.

## A. Experimental Setup

a) Datasets: We evaluate 3D reconstruction on three datasets spanning complementary forms of camera heterogeneity. PandaSet [12] provides a rigid six-camera automotive rig with camera-specific intrinsics and FoVs. On ETH3D [8], we evaluate controlled FoV variation, synthesized by center cropping and resizing with zoom factors $z ~ \sim ~ U [ 1 . 0 , 3 . 0 ]$ while retaining one native-FoV input as a wide-view anchor.

ScanNet++ [9] is evaluated under three protocols with distinct roles. The mixed-projection protocol provides the main 3D reconstruction comparison, combining approximately equal numbers of DSLR frames under the calibrated four-coefficient fisheye model and undistorted pinhole iPhone frames from the same scene. Covisibility-based sampling produces input sets of 2 to 50 views with both projection-model and device differences. The undistorted mixed-device protocol uses both streams after pinhole undistortion for the matched positionalencoding ablation and calibration diagnostics. The highresolution DSLR-only protocol uses undistorted pinhole DSLR frames for the controlled FoV variation stress test in Fig. 9.

For NVS, RealEstate10K [61] (RE10K) provides realworld videos for comparing a matched-FoV baseline with controlled FoV variation synthesized by center cropping and resizing. One context view retains its native FoV, while the other context and each target independently sample zoom factors from $U [ 1 , z _ { \mathrm { m a x } } ]$ , with $z _ { \mathrm { m a x } } \in \{ 1 . 5 , 2 . 0 , 2 . 5 \}$ . The baseline z=1.0 applies no additional FoV change. For evaluation, we preserve the 16:9 aspect ratio at 256×144 resolution with patch size 8, rather than using the 256×256 square training format. This choice avoids upsampling after cropping typical 640×360 source frames up to z=2.5. A 256×256 square evaluation would instead require upsampling beyond z≈1.4.

Objaverse [62] evaluates NVS under camera heterogeneity, with the FoV independently sampled for each original view during rendering. Two such views serve as context, and three regimes vary the target camera pose and FoV. FoV change keeps a context camera pose and applies an additional FoV change. Novel viewpoint places the target at a pose not seen in the context views, retaining its independently sampled FoV without further modification. Novel viewpoint + FoV change combines an unseen target pose with an additional change to its sampled FoV. The two FoV-change regimes thus introduce further FoV variation beyond the differences already present in the independently sampled views. Appendix C defines the complete target sets and reports the additional protocols involving radial camera motion.

We also evaluate homogeneous pinhole 3D reconstruction protocols [27] to assess compatibility when input views share one imaging geometry. These protocols assess relative pose on TUM RGB-D [63] and Sintel [64], video depth on Sintel and KITTI [65], and multi-view 3D reconstruction on NRGBD [66], 7-Scenes [67], DTU [68], and undistorted ETH3D [8]. The undistorted ETH3D split in these protocols is distinct from the ETH3D setting with controlled FoV variation in Table I.

b) Metrics: For the main 3D reconstruction evaluation with 2 to 50 input views, we report pointmap relative error (Points rel ↓), ray-direction error $( \mathrm { R a y s ~ e r r } ^ { \circ }$ ↓), depth relative error (Depth rel ↓), point inliers at 3% (Pts Inl. ↑), pose ATE (↓), and depth inliers at 3% (Dp. Inl. ↑). On PandaSet and ETH3D, Depth rel and depth inliers use standard z-depth errors. On mixed-projection ScanNet++, both non-pinhole DSLR and pinhole iPhone views are evaluated using ray depth, the distance along the viewing ray. The Depth rel and Dp. Inl. columns therefore use the depth convention appropriate to each dataset. Calibration diagnostics use 16 views. For NVS, we report PSNR (dB ↑), SSIM (↑), and LPIPS (↓). For the homogeneous pinhole 3D reconstruction protocols, we report trajectory ATE and RPE, video-depth AbsRel and δ<1.25, and 3D reconstruction accuracy, completeness, and normal consistency. 3D reconstruction accuracy and completeness are reported in millimetres on DTU and in metres on the other datasets, without averaging across these units.

c) Implementation: For 3D reconstruction, we use a VGGT-based architecture [21] with a DINOv2 ViT-L/14 backbone and alternating frame and global attention. G-ray replaces grid-index RoPE in the attention aggregator and applies the value and output rotations in (13). Training uses homogeneous pinhole inputs from mixed indoor and outdoor datasets, first at a 224-pixel long side and then at 518 pixels, on 8 NVIDIA A100 GPUs.

All NVS variants share the LVSM decoder-only backbone [30] and use 2 context views and 1 target view at 256×256 training resolution. Each encoding variant is trained independently for 80 k steps, separately on RE10K and Objaverse. Appendix B provides the architecture details, training datasets, losses, optimization schedules, and frequency settings for both tasks.

d) Baselines: 3D reconstruction methods are grouped by whether external image-to-ray calibration is supplied. With supplied calibration, we report G-ray, $\pi ^ { 3 } \mathrm { - } \mathrm { \mathbf { X } }$ [27], WorldMirror, and MapAnything [29]. For settings without externally supplied calibration, we report image-only DA3 [28], DA3- Nested, MapAnything, $\pi ^ { 3 }$ [27], VGGT [21], VGGT-Ω [22], OmniVGGT, and WorldMirror, together with G-ray driven by AnyCalib-estimated ray maps [48]. For pinhole cameras the supplied calibration is represented by intrinsic parameters, and for non-pinhole projections it additionally includes model-specific unprojection or an equivalent unit-ray map. All methods in the heterogeneous 3D reconstruction comparison use the same sampled views and evaluation metrics, with input resizing adapted to each model’s resolution requirements.

NVS baselines include Plucker-ray features with grid-index¨ RoPE, GTA [41], PRoPE [42], and RayRoPE [45]. GTA + G-ray replaces token-grid position indices with camera-local ray-angle coordinates, while RayRoPE + G-ray replaces querycamera projected pixel coordinates. Both variants retain the complementary geometric components of their host encoding.

## B. Multi-View 3D Reconstruction

We first evaluate 3D reconstruction under camera heterogeneity, then examine compatibility with homogeneous pinhole inputs. Table I reports three-dataset averages at 50 views, Fig. 6 summarizes performance from 2 to 50 views, and Fig. 7 compares reconstructed scene geometry. Complete per-dataset results are provided in Appendix Table X.

1) Heterogeneous Protocols: With supplied calibration, Gray leads all six cross-dataset metrics at 50 views, including 0.0744 pointmap relative error and 58.1% point inliers (Table I). Without externally supplied calibration, G-ray + AnyCalib leads on average pointmap error, ray-direction error, and pose ATE, while VGGT-Ω leads on depth error and point and depth inliers.

The clearest gains occur on mixed-projection ScanNet++, where the pinhole-trained model processes both pinhole and non-pinhole inputs through their ray maps without retraining. With ground-truth ray maps, G-ray leads all six metrics, achieving 0.0583 pointmap relative error and 75.1% point inliers. On PandaSet and ETH3D, G-ray achieves the lowest ray-direction errors $( 0 . 2 7 ^ { \circ }$ and 0.15<sup>◦</sup>), while $\pi ^ { 3 } \mathrm { - } \mathrm { \mathbf { X } }$ and VGGT-Ω, respectively, lead most other metrics. The gains therefore depend on the form of camera heterogeneity rather than extending uniformly to every benchmark and metric. Appendix D gives the complete per-dataset results and analysis.

2) Homogeneous Pinhole 3D Reconstruction Protocols: The preceding benchmarks contain cross-view differences in FoV or projection model. We next examine performance when input views share a homogeneous pinhole imaging geometry. Table II summarizes relative pose, video depth, and multiview 3D reconstruction under the homogeneous pinhole 3D reconstruction protocols [27]. Complete results appear in Appendix E.

VGGT-Ω achieves the strongest overall results under these protocols. G-ray with supplied calibration attains the secondbest Sintel pose ATE (0.110 vs. 0.171 for VGGT and 0.219 for MapAnything) and the second-best indoor 3D reconstruction accuracy on NRGBD and 7-Scenes, including 0.019 Acc on dense 7-Scenes versus 0.018 for VGGT-Ω. It also improves over MapAnything with the same calibrated input on TUM pose (0.0122 vs. 0.0179 ATE) and on all four indoor Acc columns. It trails VGGT-Ω, and on Sintel video depth, KITTI AbsRel, DTU Acc, and undistorted ETH3D Acc it also trails VGGT.

The 3D reconstruction model with G-ray retains competitive accuracy under homogeneous pinhole inputs, supporting the compatibility of ray-level angular encoding when camera heterogeneity is absent.

## C. Coordinate Replacement in Novel View Synthesis

We evaluate whether ray-angle coordinate replacement improves NVS in both grid-index and projection-based rotary encodings. Within each host, paired variants share the LVSM backbone, complementary geometric components, and training budget. G-ray replaces the image-plane positional coordinates with ray-angle coordinates, with per-host frequency settings specified in Appendix B-C.

a) RealEstate10K: Ray-angle coordinates improve both host encodings under controlled FoV variation (Table III). At $z _ { \mathrm { m a x } } { = } 2 . 5 , \mathrm { G T A + G { - } r a y }$ achieves 20.12 dB, compared with 13.45 dB for GTA and 13.03 dB for PRoPE. RayRoPE + Gray obtains the highest PSNR in every setting, with gains over RayRoPE of 1.11, 1.13, and 1.10 dB at $z _ { \mathrm { m a x } } { = } 1 . 5 , \ 2 . 0$ , and 2.5, respectively. The corresponding gain on the matched-FoV baseline is 0.12 dB. These results support the use of ray-angle coordinates when synthesizing real-world scenes from views with different imaging geometries.

b) Objaverse: Both host encodings benefit from G-ray when synthesizing unseen viewpoints under heterogeneous FoVs (Table IV). GTA + G-ray improves PSNR by 0.78 dB for FoV change, 0.44 dB for Novel viewpoint, and 0.88 dB for Novel viewpoint + FoV change. Replacing RayRoPE’s projected pixel coordinates yields gains of 0.48 dB at novel viewpoints and 1.32 dB when their native FoVs are additionally changed, while retaining depth, disparity, uncertainty, and camera transformations. These paired results support applying the same coordinate replacement to two distinct rotary formulations.

TABLE I  
MEAN 50-VIEW 3D RECONSTRUCTION RESULTS ACROSS MIXED-PROJECTION SCANNET++, PANDASET, AND ETH3D UNDER CONTROLLED FOV VARIATION. THE W/C GROUP RECEIVES EXTERNAL IMAGE-TO-RAY CALIBRATION THROUGH CAMERA-MODEL PARAMETERS AND UNPROJECTION OR AN EQUIVALENT UNIT-RAY MAP. THE W/O C GROUP RECEIVES NO EXTERNAL CALIBRATION AND INCLUDES IMAGE-ONLY METHODS AND G-RAY WITH ANYCALIB-ESTIMATED RAY MAPS. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN.
<table><tr><td>Setting Method</td><td></td><td>Source / year</td><td>Points rel ↓ Rays err°</td><td></td><td></td><td></td><td>Depth rel ↓ Pts Inl. ↑ Pose ATE ↓ Dp. Inl. ↑</td><td></td></tr><tr><td rowspan="4">w/C</td><td>G-ray</td><td>This work</td><td>0.0744</td><td>0.38</td><td>0.0744</td><td>58.1</td><td>0.0221</td><td>51.9</td></tr><tr><td>π3-X [27]</td><td>Release 2025†</td><td>0.1130</td><td>2.32</td><td>0.0798</td><td>48.8</td><td>0.0425</td><td>42.3</td></tr><tr><td>WorldMirror [59]</td><td>ICML 2026</td><td>0.1338</td><td>2.26</td><td>0.0936</td><td>40.3</td><td>0.0441</td><td>38.2</td></tr><tr><td>MapAnything [29]</td><td>3DV 2026</td><td>0.1372</td><td>2.11</td><td>0.1199</td><td>37.5</td><td>0.0616</td><td>28.6</td></tr><tr><td rowspan="10">w/o C</td><td>G-ray + AnyCalib</td><td>This work</td><td>0.1347</td><td>2.49</td><td>0.1407</td><td>43.4</td><td>0.0702</td><td>24.4</td></tr><tr><td>DA3 [28]</td><td>ICLR 2026</td><td>0.3441</td><td>6.74</td><td>0.2827</td><td>25.3</td><td>0.1885</td><td>16.9</td></tr><tr><td>DA3-Nested [28]</td><td>ICLR 2026</td><td>0.3278</td><td>6.04</td><td>0.2387</td><td>25.7</td><td>0.1444</td><td>15.8</td></tr><tr><td>MapAnything [29] 3DV 2026</td><td></td><td>0.2404</td><td>3.90</td><td>0.2037</td><td>27.0</td><td>0.1208</td><td>14.0</td></tr><tr><td>π³ [27]</td><td>ICLR 2026</td><td>0.2828</td><td>4.29</td><td>0.1939</td><td>34.8</td><td>0.1537</td><td>21.3</td></tr><tr><td>VGGT [21]</td><td>CVPR 2025</td><td>0.3359</td><td>4.31</td><td>0.2081</td><td>22.8</td><td>0.1795</td><td>19.4</td></tr><tr><td>VGGT-Ω [22]</td><td>CVPR 2026</td><td>0.2011</td><td>4.12</td><td>0.1329</td><td>46.1</td><td>0.0780</td><td>38.1</td></tr><tr><td>OmniVGGT [60]</td><td>CVPR 2026</td><td>0.2708</td><td>4.30</td><td>0.3198</td><td>25.8</td><td>0.1815</td><td>13.9</td></tr><tr><td>WorldMirror [59]</td><td>ICML 2026</td><td>0.1944</td><td>3.34</td><td>0.1554</td><td>32.0</td><td>0.0839</td><td>25.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup> π<sup>3</sup>-X is the authors’ December 2025 model release, not a separate conference publication. <sup>‡</sup> DA3-Nested is a released DA3 variant and shares its publication.

![](images/6739df01caabc10d934bf94535b659f6cf62ebb3f0aa5ab8205677af73ca0119.jpg)  
Fig. 6. Mean 3D reconstruction performance across mixed-projection ScanNet++, PandaSet, and ETH3D with 2 to 50 input views. ScanNet++ uses ray-depth metrics. The dashed G-ray curves use AnyCalib-estimated ray maps. Table I gives the 50-view results. Lower is better for pointmap, ray-direction, depth, and pose errors. Higher is better for point and depth inlier rates.

The benefit depends on the context-to-target relationship. When the target shares a pose with one context view, RayRoPE + G-ray scores 27.97 dB versus 28.42 dB for RayRoPE, a decrease of 0.46 dB computed before rounding. Thus, the benefit is not uniform across all FoV changes and is larger for synthesis at unseen viewpoints. Additional results involving radial camera motion are reported in Appendix C.

c) Qualitative Comparison: Fig. 8 illustrates the effects of coordinate replacement under FoV variation. In the RE10K examples, GTA and Plucker exhibit blur and structural errors,¨ while GTA + G-ray better preserves scene layout. RayRoPE + G-ray reduces the ghosting and banding visible in baseline RayRoPE predictions. The Objaverse examples show similar improvements along short viewpoint trajectories with changing FoV. These examples illustrate the structural improvements reflected in the quantitative comparisons. Additional targetview comparisons are provided in Appendix C-B.

![](images/c0fcc02c8c4d28ba0bb9b90ed4a5ee8256d9cffafbdd45bc4fbc287537108ac0.jpg)  
Fig. 7. Qualitative 3D reconstruction under camera heterogeneity. Rows show mixed-projection ScanNet++, ETH3D under controlled FoV variation, and the PandaSet multi-camera rig. Columns show inputs, ground truth, G-ray, MapAnything with supplied calibration, and image-only VGGT-Ω. Within each row, all point clouds share ground-truth-defined display bounds and identical rendering viewpoints, FoVs, and settings. PandaSet ground truth is sparse LiDAR, whereas the predictions are dense image-based 3D reconstructions.

TABLE II  
SELECTED RESULTS UNDER HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS [27]. G-RAY AND MAPANYTHING RECEIVE GROUND-TRUTH PINHOLE CALIBRATION, WHILE VGGT AND VGGT-Ω ARE IMAGE-ONLY. INDOOR 3D RECONSTRUCTION ACCURACY (ACC) IS IN METRES. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN. APPENDIX E PROVIDES THE COMPLETE RESULTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Source / year</td><td colspan="2">Rel. pose ATE</td><td colspan="2">Video depth AbsRel ↓</td><td colspan="4">Indoor Acc ↓</td></tr><tr><td>TUM</td><td>Sintel</td><td>Sintel</td><td>KITTI</td><td>NRGBD-s</td><td>NRGBD-d</td><td>7Sc-s</td><td>7Sc-d</td></tr><tr><td>G-ray</td><td>This work</td><td>0.0122</td><td>0.110</td><td>0.357</td><td>0.092</td><td>0.037</td><td>0.023</td><td>0.038</td><td>0.019</td></tr><tr><td>MapAnything [29]</td><td>3DV 2026</td><td>0.0179</td><td>0.219</td><td>0.401</td><td>0.083</td><td>0.088</td><td>0.036</td><td>0.059</td><td>0.023</td></tr><tr><td>VGGT [21]</td><td>CVPR 2025</td><td>0.0090</td><td>0.171</td><td>0.324</td><td>0.082</td><td>0.058</td><td>0.029</td><td>0.055</td><td>0.025</td></tr><tr><td>VGGT-Ω [22]</td><td>CVPR 2026</td><td>0.0044</td><td>0.040</td><td>0.251</td><td>0.072</td><td>0.022</td><td>0.014</td><td>0.029</td><td>0.018</td></tr></table>

## D. Ablation Studies

The ablations examine coordinate representation, value and output rotations, and frequency scale under matched training conditions.

1) Coordinate Replacement under a Matched 3D Reconstruction Protocol: To isolate the positional encoding, all variants in Table V use the same ViT-S/14 backbone and 45- epoch training schedule.

a) Ray-Angle PE vs. Image-Plane Grid-Index RoPE: Gray outperforms grid-index RoPE on all six reported metrics across both datasets. On ScanNet++, pointmap relative error, ray-direction error, and pose ATE decrease by approximately 15.5%, 19.7%, and 11.5%, respectively. On ETH3D under controlled FoV variation, the corresponding reductions are 8.6%, 14.6%, and 2.3%. The improvement across both datasets supports ray-angle coordinates over grid indices within the same rotary attention formulation.

b) Ray-Angle PE vs. Absolute CamRay: G-ray also outperforms absolute CamRay conditioning on all six columns. Relative reductions in pointmap error, ray-direction error, and pose ATE are 13.2%, 19.7%, and 6.8% on ScanNet++, and 7.9%, 15.9%, and 2.0% on ETH3D. Under this matched setting, relative ray-angle encoding is more effective than the evaluated absolute camera-ray conditioning design.

c) Value and Output Encoding: Applying the rotations only to queries and keys, without the value and inverse output rotations, degrades all six metrics. ScanNet++ ray error increases from 15.9<sup>◦</sup> to 21.2<sup>◦</sup> (+33%) and ETH3D ray error from 11.1<sup>◦</sup> to 13.3<sup>◦</sup> (+20%). The result supports applying the rotations to values and outputs in addition to queries and keys.

2) NVS Frequency Scale: Table VI evaluates the frequency scale α of (8) in $\mathrm { G T A + G \mathrm { - } r a y }$ . The matched-FoV baseline and mild FoV variation $( z _ { \mathrm { m a x } } { = } 1 . 5 )$ achieve their highest PSNR at α=2, with 20.97 and 20.74 dB, respectively. Stronger FoV variation $( z _ { \mathrm { m a x } } { = } 2 . 0 $ and 2.5) favors α=10, with 20.41 and

![](images/11b9b1691b30d3d9e3e26fd4f6eb7ca00030c88fa1dfb446cff389e51b44fa18.jpg)  
Fig. 8. Qualitative NVS under FoV heterogeneity. The upper block shows RE10K under controlled FoV variation at 256×144 resolution and $z _ { \mathrm { m a x } } { = } 2 . 0 .$ One context retains its native FoV, while the remaining views sample zoom factors from U[1, 2]. The lower block shows Objaverse with changing FoV along short viewpoint trajectories. Columns contain two context views, the ground-truth target, and predictions from Plucker, GTA, GTA + G-ray, RayRoPE, and¨ RayRoPE + G-ray. Numbers report target-view PSNR (dB ↑). Both G-ray variants better preserve scene structure in these examples.

## TABLE III

RE10K NVS UNDER CONTROLLED FOV VARIATION, MEASURED BY PSNR (DB ↑) AT 256×144 RESOLUTION. THE MATCHED-FOV BASELINE USES z=1.0. OTHERWISE, ONE CONTEXT RETAINS ITS NATIVE FOV AND THE REMAINING VIEWS INDEPENDENTLY SAMPLE ZOOM FACTORS FROM $U [ 1 , z _ { \mathrm { m a x } } ]$ . PAIRED ROWS COMPARE EACH HOST ENCODING WITH AND WITHOUT G-RAY. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.

<table><tr><td rowspan="2">Method</td><td>Homogeneous</td><td colspan="3">Heterogeneous imaging geometry</td></tr><tr><td>z=1.0</td><td> $z _ { \mathrm { m a x } } { = } 1 . 5$ </td><td> $z _ { \mathrm { m a x } } { = } 2 . 0$ </td><td> $z _ { \mathrm { m a x } } { = } 2 . 5$ </td></tr><tr><td>Plücker [30] ICLR 2025 (LVSM)</td><td>20.67</td><td>18.22</td><td>17.50</td><td>17.05</td></tr><tr><td>PRoPE [42] NeurIPS 2025</td><td>19.80</td><td>15.19</td><td>13.80</td><td>13.03</td></tr><tr><td>GTA [41]</td><td>19.06</td><td>15.12</td><td>14.00</td><td>13.45</td></tr><tr><td>ICLR 2024 GTA + G-ray This work</td><td>20.62</td><td>20.72</td><td>20.41</td><td>20.12</td></tr><tr><td>RayRoPE [45]</td><td>21.35</td><td>20.98</td><td>20.65</td><td>20.32</td></tr><tr><td>ECCV 2026 RayRoPE + G-ray This work</td><td>21.47</td><td>22.09</td><td>21.78</td><td>21.42</td></tr></table>

20.12 dB. This setting also gives the lowest LPIPS under all four protocols. We use α=10 consistently for GTA + G-ray, balancing performance under larger FoV differences with a 0.35 dB difference from the matched-FoV PSNR maximum.

## TABLE IV

OBJAVERSE NVS UNDER THREE TARGET-VIEW REGIMES, MEASURED BY PSNR (DB ↑). ALL REGIMES USE TWO CONTEXT VIEWS WITH   
INDEPENDENTLY SAMPLED FOVS AND THE SAME LVSM BACKBONE.   
FOV CHANGE APPLIES AN ADDITIONAL FOV CHANGE AT A CONTEXT POSE. NOVEL VIEWPOINT USES AN UNSEEN TARGET POSE WITHOUT FURTHER MODIFYING ITS SAMPLED FOV, AND THE FINAL REGIME COMBINES BOTH CHANGES. PAIRED ROWS COMPARE EACH HOST   
ENCODING WITH AND WITHOUT G-RAY. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.

<table><tr><td>Method</td><td>FoV change</td><td>Novel viewpoint</td><td>Novel viewpoint + FoV change</td></tr><tr><td>Plücker [30] ICLR 2025 (LVSM)</td><td>26.20</td><td>18.88</td><td>11.17</td></tr><tr><td>PRoPE [42] NeurIPS 2025</td><td>26.07</td><td>19.69</td><td>10.67</td></tr><tr><td>GTA [41] ICLR 2024</td><td>25.72</td><td>19.11</td><td>10.92</td></tr><tr><td>GTA + G-ray This work</td><td>26.50</td><td>19.55</td><td>11.80</td></tr><tr><td>RayRoPE [45] ECCV 2026</td><td>28.42</td><td>19.55</td><td>10.68</td></tr><tr><td>RayRoPE + G-ray This work</td><td>27.97</td><td>20.03</td><td>12.00</td></tr></table>

## E. Analysis

Following the attention diagnostic in Sec. III-D1, we evaluate robustness to controlled FoV variation and resolution changes, dependence on calibration, and inference cost.

MATCHED POSITIONAL-ENCODING ABLATION FOR 3D RECONSTRUCTION. ALL VARIANTS USE VIT-S/14 WITH EMBEDDING DIMENSION 384 AND THE SAME TRAINING SCHEDULE. SCANNET++ USES APPROXIMATELY EQUAL NUMBERS OF UNDISTORTED PINHOLE DSLR AND IPHONE VIEWS. ETH3D USES CONTROLLED FOV VARIATION. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.
<table><tr><td></td><td colspan="3">ScanNet++</td><td colspan="3">ETH3D (FoV var.)</td></tr><tr><td>Variant</td><td>Pts↓</td><td>Ray°↓</td><td>Pose↓</td><td>Pts↓</td><td>Ray°↓</td><td>Pose↓</td></tr><tr><td>G-ray (full)</td><td>0.820</td><td>15.9</td><td>0.408</td><td>0.984</td><td>11.1</td><td>0.385</td></tr><tr><td>w/o v, o encoding</td><td>0.927</td><td>21.2</td><td>0.444</td><td>0.996</td><td>13.3</td><td>0.389</td></tr><tr><td>→ Grid-index RoPE</td><td>0.970</td><td>19.8</td><td>0.461</td><td>1.077</td><td>13.0</td><td>0.394</td></tr><tr><td>→ Absolute CamRay</td><td>0.945</td><td>19.8</td><td>0.438</td><td>1.068</td><td>13.2</td><td>0.393</td></tr></table>

TABLE VI

FREQUENCY-SCALE ABLATION FOR GTA + G-RAY ON RE10K, WITH FREQUENCY BASE FIXED AT $\beta { = } 1 0 0$ . PSNR (DB ↑) IS MEASURED AT 256×144 RESOLUTION UNDER THE FOV PROTOCOLS OF TABLE III. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN. THE SELECTED SETTING IS α=10.
<table><tr><td rowspan="2">α</td><td rowspan="2">Homogeneous z=1.0</td><td colspan="3">Heterogeneous imaging geometry</td></tr><tr><td> $z _ { \mathrm { m a x } } { = } 1 . 5$ </td><td> $z _ { \mathrm { m a x } } { = } 2 . 0$ </td><td> $z _ { \mathrm { m a x } } { = } 2 . 5$ </td></tr><tr><td>1</td><td>20.66</td><td>20.40</td><td>19.98</td><td>19.65</td></tr><tr><td>2</td><td>20.97</td><td>20.74</td><td>20.33</td><td>20.01</td></tr><tr><td>4</td><td>20.77</td><td>20.63</td><td>20.26</td><td>19.98</td></tr><tr><td>6</td><td>20.76</td><td>20.53</td><td>20.08</td><td>19.72</td></tr><tr><td>8</td><td>20.54</td><td>20.58</td><td>20.30</td><td>20.04</td></tr><tr><td>10</td><td>20.62</td><td>20.72</td><td>20.41</td><td>20.12</td></tr><tr><td>15</td><td>20.31</td><td>20.50</td><td>20.23</td><td>19.96</td></tr><tr><td>20</td><td>20.39</td><td>20.51</td><td>20.19</td><td>19.90</td></tr><tr><td>40</td><td>20.25</td><td>20.60</td><td>20.31</td><td>20.01</td></tr></table>

1) Controlled FoV Variation: Fig. 9 evaluates increasing FoV disparity on 15 high-resolution ScanNet++ scenes using undistorted pinhole DSLR frames. For each 16-view input, one view retains its native FoV and the remaining views uniformly sample zoom factors up to $z _ { \mathrm { h i g h } } \in \{ 1 . 0 , \ldots , 5 . 0 \}$ . The source frames and camera poses remain fixed, while center cropping and resizing change the angular coverage of each view. As $z _ { \mathrm { h i g h } }$ increases, MapAnything with supplied calibration exhibits fewer point inliers and higher pose ATE, whereas Gray remains comparatively stable on both metrics. VGGT-Ω improves over VGGT but still trails G-ray at large FoV differences. Together with the matched ablation in Table V, this result supports robustness to controlled FoV variation within the pinhole projection model.

2) Resolution Extrapolation: Although ray-angle coordinates do not explicitly depend on pixel scale, resolution generalization must be evaluated for the complete network. We test input resolutions above the 518-pixel training long side. On ETH3D, we sweep patch-aligned long-side resolutions corresponding to nominal scales $\{ 1 . 0 , 1 . 2 5 , 1 . 5 , 1 . 7 5 , 2 . 0 \} \times 5 1 8$ at 4 views, and compare G-ray against MapAnything with calibrated input and VGGT. Fig. 10 compares matched-FoV inputs with controlled FoV variation that retains one native-FoV view and samples zoom factors uniformly up to $z _ { \mathrm { h i g h } } { = } 3$

Without FoV variation, doubling the input resolution reduces point inliers from 66.9% to 28.9% for MapAnything and from 54.9% to 13.6% for VGGT, compared with 66.5% to 58.8% for G-ray. Under the $z _ { \mathrm { h i g h } } { = } 3$ protocol the absolute accuracy is lower for all methods, and the same ordering persists, with G-ray declining from 50.2% to 45.5% while MapAnything and VGGT again drop substantially. Pose ATE follows the same pattern. The results support improved resolution transfer on ETH3D with both matched and heterogeneous FoVs.

![](images/5e7a22ba8bbd9e2fe17dd33c1dd2ebf4a13632ec738d795aaa4dee9325624bc6.jpg)

![](images/5cb0fc551ffb9ba1d42fefbf8bfe295b2c9c788df40c85851e075601118cebef.jpg)  
Fig. 9. Controlled FoV variation on ScanNet++ using undistorted pinhole DSLR frames. One view retains its native FoV, while the remaining views are center-cropped and resized with zoom factors sampled uniformly up to $z _ { \mathrm { h i g h } } ~ \in ~ \{ 1 . 0 , \ldots , 5 . 0 \}$ . The panels report point inlier rate (%, ↑) and pose ATE RMSE (↓). G-ray remains comparatively stable as cross-view FoV differences increase.

TABLE VII  
SENSITIVITY TO INTRINSIC-CALIBRATION ERRORS, AVERAGED OVER ETH3D, UNDISTORTED MIXED-DEVICE SCANNET++, AND PANDASET AT 16 VIEWS. ONLY THE SUPPLIED INTRINSICS CHANGE, WHILE IMAGES AND EVALUATION TARGETS REMAIN FIXED. LOWER IS BETTER EXCEPT FOR POINT INLIER RATE. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.
<table><tr><td>Condition</td><td>Points rel</td><td> $\scriptstyle { \mathrm { R a y s ~ e r r } } ^ { \circ }$ </td><td>Depth rel</td><td>Pose ATE</td><td>Pts Inl.</td></tr><tr><td>Correct K</td><td>0.0756</td><td>0.191</td><td>0.0733</td><td>0.0213</td><td>57.7</td></tr><tr><td>+5% focal</td><td>0.0881</td><td>0.995</td><td>0.0852</td><td>0.0280</td><td>51.8</td></tr><tr><td>+10% focal</td><td>0.1087</td><td>1.899</td><td>0.1035</td><td>0.0393</td><td>43.0</td></tr><tr><td>+20% focal</td><td>0.1540</td><td>3.552</td><td>0.1412</td><td>0.0638</td><td>31.3</td></tr><tr><td>Shuffled K</td><td>0.4770</td><td>5.778</td><td>0.3880</td><td>0.2925</td><td>12.0</td></tr></table>

3) Sensitivity to Intrinsic Calibration: Because G-ray derives relative phases from view-specific calibration, we distinguish sensitivity to calibration error from robustness to FoV differences in correctly calibrated images. We perturb only the conditioning intrinsics while leaving images, poses, groundtruth rays, and depth unchanged. The focal trials multiply $f _ { x }$ and $f _ { y }$ by $1 + \epsilon$ for $\epsilon \in \lbrace 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \rbrace$ while fixing the principal point. The shuffled condition cyclically assigns intrinsics from another view in the same input set. Table VII reports averages over ETH3D, ScanNet++, and PandaSet at 16 views. ScanNet++ uses the undistorted mixed-device protocol.

All five reported metrics degrade monotonically as the uniform focal perturbation increases. At +5%, point and depth errors remain substantially closer to the correct-calibration condition than to the shuffled condition, but ray-direction error already rises from $0 . 1 9 1 ^ { \circ }$ to 0.995<sup>◦</sup>. Assigning intrinsics to the wrong views produces the largest degradation on every metric. Thus, robustness to cross-view FoV variation does not remove the need for accurate, correctly associated image-toray calibration.

![](images/f28b5b5f7ae468a0689627dd9728fabca12eebdf872f56fbaac8500ee715b955.jpg)  
Fig. 10. Resolution extrapolation on ETH3D. Columns report point inlier rate (%, ↑) and pose ATE RMSE (↓), with shared axis ranges across rows. The top row uses matched $\mathrm { F o V s . }$ The bottom row uses controlled FoV variation with one native-FoV anchor and uniformly sampled zoom factors up to $z _ { \mathrm { h i g h } } { = } 3$ . Input long-side resolutions are patch-aligned nominal scales of $5 1 8 \times \{ 1 . 0 \}$ 1.25, 1.5, 1.75, 2.0}. G-ray degrades more gradually than MapAnything with supplied calibration and image-only VGGT.

4) Off-the-Shelf Intrinsic Estimates: To test G-ray without supplied calibration, we use the single-image estimator Any-Calib [48] to recover image-to-ray geometry. Table VIII(a) reports calibration accuracy on three datasets, and panel (b) evaluates 16-view ScanNet++ 3D reconstruction with estimated conditioning geometry while retaining ground-truth rays and depth as evaluation targets. For the undistorted mixed-device protocol, we also compare multi-view VGGT estimates [21].

Calibration quality varies across datasets, with the largest focal and principal-point errors on PandaSet. On undistorted ScanNet++, VGGT estimates yield more accurate 3D reconstruction than single-image AnyCalib estimates, but both trail ground-truth calibration. On mixed-projection ScanNet++, AnyCalib ray maps achieve 60.2% point inliers versus 72.9% with ground-truth ray maps, demonstrating non-pinhole 3D reconstruction with estimated geometry while retaining a measurable calibration gap. Appendix F-A details the camera models, metric conventions, and per-protocol analysis.

5) Runtime and Memory: We measure the computational cost of coordinate replacement using paired end-to-end inference trials. Both variants share the same trained checkpoint,

ESTIMATED IMAGE-TO-RAY GEOMETRY AND 3D RECONSTRUCTION ACCURACY. (A) ANYCALIB CALIBRATION ACCURACY, EVALUATED INDEPENDENTLY OF 3D RECONSTRUCTION. (B) G-RAY 3D RECONSTRUCTION ON SCANNET++ USING ESTIMATED CALIBRATION, WITH GROUND-TRUTH RAYS AND DEPTH RETAINED AS EVALUATION TARGETS. SCANNET++ IN (A) AND UNDIST. MIXED IN (B) USE UNDISTORTED PINHOLE DSLR AND IPHONE VIEWS. MIXED PROJ. COMBINES NON-PINHOLE DSLR AND PINHOLE IPHONE VIEWS. PANEL (B) USES THE ABBREVIATED METRICS OF TABLE I. BOLD VALUES INDICATE THE BEST RESULT FOR EACH METRIC WITHIN EACH PROTOCOL, INCLUDING GROUND-TRUTH CALIBRATION.

(a) Calibration accuracy
<table><tr><td>Dataset</td><td>Rays err° ↓ Rel. focal ↓ PP err px ↓</td><td></td><td></td></tr><tr><td>ETH3D</td><td>1.47</td><td>7.1%</td><td>1.44</td></tr><tr><td>ScanNet++</td><td>2.04</td><td>8.6%</td><td>0.78</td></tr><tr><td>PandaSet</td><td>4.53</td><td>24.7%</td><td>22.35</td></tr><tr><td>Average</td><td>2.68</td><td>13.5%</td><td>8.19</td></tr></table>

(b) End-to-end 3D reconstruction with estimated conditioning geometry
<table><tr><td>Protocol</td><td>Source</td><td>Pts ↓</td><td> $\mathrm { R a y ^ { o } \downarrow }$ </td><td>Depth ↓</td><td>ATE↓</td><td>Inl. ↑</td></tr><tr><td rowspan="3"></td><td>GT K</td><td>0.0449</td><td>0.161</td><td>0.0393</td><td>0.0204</td><td>81.8</td></tr><tr><td>Undist. mixed AnyCalib K</td><td>0.1161</td><td>2.543</td><td>0.1112</td><td>0.0641</td><td>50.3</td></tr><tr><td>VGGT K</td><td>0.0656</td><td>1.177</td><td>0.0666</td><td>0.0366</td><td>71.0</td></tr><tr><td rowspan="2">Mixed proj.</td><td>GT ray map</td><td>0.0588</td><td>0.709</td><td>0.0466</td><td>0.0238</td><td>72.9</td></tr><tr><td>AnyCalib ray map</td><td>0.0879</td><td>2.164</td><td>0.0714</td><td>0.0514</td><td>60.2</td></tr></table>

prediction heads, and pre- and post-processing. Only the coordinates supplied to the rotary encoding change between grid indices and calibrated ray angles. Fig. 11 reports latency and peak GPU memory for 2 to 50 views. The two modes follow nearly indistinguishable scaling curves. At 50 views, Gray and the grid-coordinate baseline respectively use 12.403 and 12.401 GiB on average, while their mean latencies are 6.504 and 6.494 s. The paired latency difference is 9.6 ms, with a 95% confidence interval from −20.0 to 39.2 ms that includes zero. These measurements show similar runtime and memory usage, with no detectable latency increase under the evaluated conditions.

## V. DISCUSSION AND LIMITATIONS

a) Calibration Dependence: G-ray requires viewspecific image-to-ray geometry, supplied through a calibrated camera model or a unit-ray map. Tables VII and VIII quantify its sensitivity to calibration errors. Accuracy decreases monotonically across the tested focal perturbations and drops sharply when intrinsics are assigned to the wrong views. 3D reconstruction with either single-image or multi-view estimates remains less accurate than with ground-truth calibration, and estimation quality varies across domains. These results support using estimated calibration but do not establish calibration-free operation.

b) Training Scale and Homogeneous-Input Compatibility: Training data scale and computational resources are important factors in the performance of multi-view vision Transformers. VGGT-Ω, for example, reports training on 128 H100 GPUs, using approximately 4 million supervised scenes/sequences and 18 million unlabeled videos [22]. Our 3D reconstruction model is trained on 20 million images from the public-dataset mixture in Appendix B-A, using 8 A100

![](images/368e27e58d356b40f82469433ed7bccdae36a5870e6343232d581b530b7fe8e1.jpg)  
Fig. 11. End-to-end inference time and peak GPU memory with 2 to 50 input views. Markers and bands show means and one standard deviation over ten paired trials with alternating execution order on an NVIDIA RTX 4090. Grid-index RoPE and G-ray use the same checkpoint, inference procedure, BF16 precision, and calibrated 518×518 inputs.

GPUs. G-ray addresses cross-projection phase mismatch, a source of inconsistent geometric cues for cross-projection attention, through ray-level angular encoding without additional learned parameters. With supplied calibration, this model outperforms VGGT-Ω on all six cross-dataset averaged metrics under camera heterogeneity (Table I) and remains competitive under homogeneous pinhole 3D reconstruction protocols (Table II). These systems differ in architecture, calibration inputs, and training data, so this comparison does not isolate training efficiency. The results motivate larger-scale training of G-ray, while its performance at a matched data and compute budget remains to be evaluated.

c) Efficiency Scope: G-ray introduces no learned parameters. The paired measurements in Fig. 11 show nearly identical runtime and peak GPU memory scaling for G-ray and gridindex RoPE across 2 to 50 views. This comparison isolates coordinate replacement for a fixed architecture, input resolution, GPU, and software configuration. Other architectures, kernels, and hardware require separate benchmarking.

d) Applicability Boundary: On Objaverse, the largest gains among the three main regimes occur when novelviewpoint synthesis is combined with additional FoV change. For FoV change at a context pose, RayRoPE + G-ray is 0.46 dB below RayRoPE, while GTA + G-ray improves over GTA. The target shares one context pose, but the two context views still have different poses. The benefit therefore depends on the host encoding and target-view regime, and these comparisons do not isolate why RayRoPE’s projected pixel coordinates perform better in the observed-pose regime.

The evaluated resolution and projection-model ranges, and the distinction between controlled FoV variation and optical zoom, are discussed further in Appendices F-B and F-C.

## VI. CONCLUSION

We presented G-ray, a ray-level relative position encoding that lifts the positional coordinate from the image plane to camera-local ray-angle space. Ray angles provide a shared physical unit across projections, so the resulting relative phase offers projection-invariant positional consistency that imageplane coordinates cannot provide. The encoding operates by replacing the image-plane coordinate pair used to form rotary phases with ray angles. The surrounding attention computation and the model’s complementary geometric branches are retained. The method consumes a view-specific image-to-ray map, introduces no learned parameters, and enables a model trained exclusively on pinhole images to generalize to nonpinhole inputs at inference by supplying the corresponding ray map.

Across distributed camera collections, rigid multi-camera rigs, and variable-FoV configurations, G-ray leads all six threedataset average 3D reconstruction metrics at 50 views among methods with complete results. NVS experiments show that integrating G-ray into GTA and RayRoPE improves synthesis under controlled FoV variation on RE10K and at novel viewpoints with additional FoV change on Objaverse. On homogeneous pinhole 3D reconstruction protocols, accuracy is retained under the angular encoding, confirming compatibility when camera heterogeneity is absent. Further analysis shows that accuracy degrades more gradually than compared baselines under resolution extrapolation beyond the training scale, and that most of it is retained when calibration comes from an off-the-shelf estimator.

The observed calibration sensitivity and the lack of improvement for RayRoPE + G-ray under same-pose FoV changes define current applicability limits and motivate further study of position encoding under camera heterogeneity.

## ACKNOWLEDGMENT

This work was supported by the National Key R&D Program of China under Grants 2022ZD0160601 and 2022YFB3903404.

## REFERENCES

[1] X. Lin, T. Lin, L. Huang, H. Xie, and Z. Su, “BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 9007–9016. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2025/html/Lin BIP3D Bridging 2D Images and 3D Perception for Embodied Intelligence CVPR 2025 paper.html

[2] X. Pan, N. Charron, Y. Yang, S. Peters, T. Whelan, C. Kong, O. Parkhi, R. Newcombe, and C. Y. Ren, “Aria Digital Twin: A New Benchmark Dataset for Egocentric 3D Machine Perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20 133–20 143. [Online]. Available: https://openaccess.thecvf.com/ content/ICCV2023/html/Pan Aria Digital Twin A New Benchmark Dataset for Egocentric 3D ICCV 2023 paper.html

[3] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 621–11 631. [Online]. Available: https://openaccess.thecvf. com/content CVPR 2020/html/Caesar nuScenes A Multimodal Dataset for Autonomous Driving CVPR 2020 paper.html

[4] X. Yang, Y. Wang, H. Li, and Y. Zhang, “UAVFF3D: A Geometry-Aware Benchmark for Feed-Forward UAV 3D Reconstruction,” 2026. [Online]. Available: https://arxiv.org/abs/2605.17942

[5] X. Wang, “Intelligent multi-camera video surveillance: A review,” Pattern Recognition Letters, vol. 34, no. 1, pp. 3–19, 2013. [Online]. Available: https://doi.org/10.1016/j.patrec.2012.07.005

[6] P. Natarajan, P. K. Atrey, and M. Kankanhalli, “Multi-Camera Coordination and Control in Surveillance Systems: A Survey,” ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 11, no. 4, pp. 1–30, 2015. [Online]. Available: https://doi.org/10.1145/2710128

[7] Z. Li and N. Snavely, “MegaDepth: Learning Single-View Depth Prediction from Internet Photos,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2041–2050. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2018/ html/Li MegaDepth Learning Single-View CVPR 2018 paper.html

[8] T. Schops, J. L. Schonberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, and A. Geiger, “A Multi-view Stereo Benchmark with High-Resolution Images and Multi-camera Videos,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. [Online]. Available: https://doi.org/10.1109/CVPR.2017.272

[9] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12–22. [Online]. Available: https://openaccess.thecvf.com/content/ ICCV2023/html/Yeshwanth ScanNet A High-Fidelity Dataset of 3D Indoor Scenes ICCV 2023 paper.html

[10] Y. Zhou, L. Quang, C. Nieto-Granda, and G. Loianno, “CoPeDadvancing multi-robot collaborative perception: A comprehensive dataset in real-world environments,” IEEE Robotics and Automation Letters, vol. 9, no. 7, pp. 6416–6423, 2024. [Online]. Available: https://doi.org/10.1109/LRA.2024.3406207

[11] Y. Hou, B. Zou, M. Zhang, R. Chen, S. Yang, Y. Zhang, J. Zhuo, S. Chen, J. Chen, and H. Ma, “AGC-Drive: A large-scale dataset for real-world aerial-ground collaboration in driving scenarios,” in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, vol. 38, 2025, pp. 111 315–111 337. [Online]. Available: https://papers.nips.cc/paper files/paper/2025/ hash/910ac01a3e5c5e7e67d64f23c0e1a740-Abstract-Datasets and Benchmarks Track.html

[12] P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang, Y. Wang, and D. Yang, “PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving,” in IEEE International Intelligent Transportation Systems Conference, 2021. [Online]. Available: https://doi.org/10.1109/ITSC48978.2021.9565009

[13] H. Zhou, S. Liu, Y. He, B. Zhang, F. Fu, C. Hou, X. Hou, L. Han, and W. Sui, “X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras,” 2026. [Online]. Available: https: //arxiv.org/abs/2607.12993

[14] T. I. Amosa, P. Sebastian, L. I. Izhar, O. Ibrahim, L. S. Ayinla, A. A. Bahashwan, A. Bala, and Y. A. Samaila, “Multi-camera multi-object tracking: A review of current trends and future advances,” Neurocomputing, vol. 552, p. 126558, 2023. [Online]. Available: https://doi.org/10.1016/j.neucom.2023.126558

[15] G. C. Ganesan, Y. Guo, L. Ren, and X. Liu, “UniDAC: Universal Metric Depth Estimation for Any Camera,” 2026. [Online]. Available: https://arxiv.org/abs/2603.27105

[16] Y. Guo, M. Chao, L. Wang, T. Zhao, H. Dai, Y. Zhang, J. Yu, and Y. Shi, “PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 36 444–36 453. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/html/ Guo PanoVGGT Feed-Forward 3D Reconstruction from Panoramic Imagery CVPR 2026 paper.html

[17] Z. Zhang, “A flexible new technique for camera calibration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 11, pp. 1330–1334, 2000.

[18] F. Rameau, J. Park, O. Bailo, and I. S. Kweon, “MC-Calib: A generic and robust calibration toolbox for multi-camera systems,” Computer Vision and Image Understanding, vol. 217, p. 103353, 2022. [Online]. Available: https://doi.org/10.1016/j.cviu.2021.103353

[19] J. L. Schonberger and J.-M. Frahm, “Structure-from-Motion Revisited,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016. [Online]. Available: https://doi.org/10.1109/ CVPR.2016.445

[20] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “MVSNet: Depth Inference for Unstructured Multi-view Stereo,” in European Conference

on Computer Vision, 2018, pp. 785–801. [Online]. Available: https://doi.org/10.1007/978-3-030-01237-3 47

[21] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual Geometry Grounded Transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5294–5306. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2025/html/Wang VGGT Visual Geometry Grounded Transformer CVPR 2025 paper.html

[22] J. Wang, M. Chen, S. Zhang, N. Karaev, J. Schonberger, P. Labatut,¨ P. Bojanowski, D. Novotny, A. Vedaldi, and C. Rupprecht, “VGGT-Ω,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. [Online]. Available: https: //arxiv.org/abs/2605.15195

[23] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D Vision Made Easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697–20 709. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/Wang DUSt3R Geometric 3D Vision Made Easy CVPR 2024 paper.html

[24] V. Leroy, Y. Cabon, and J. Revaud, “Grounding Image Matching in 3D with MASt3R,” in Proceedings of the European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, vol. 15130, 2024, pp. 71–91. [Online]. Available: https://www.ecva.net/ papers/eccv 2024/papers ECCV/html/9080 ECCV 2024 paper.php

[25] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, “MUSt3R: Multi-view Network for Stereo 3D Reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 1050–1060. [Online]. Available: https://openaccess.thecvf.com/content/ CVPR2025/html/Cabon MUSt3R Multi-view Network for Stereo 3D Reconstruction CVPR 2025 paper.html

[26] J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli, “Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 21 924–21 935. [Online]. Available: https://openaccess.thecvf.com/ content/CVPR2025/html/Yang Fast3R Towards 3D Reconstruction of 1000 Images in One Forward CVPR 2025 paper.html

[27] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Permutation-Equivariant Visual Geometry Learning,” in International Conference on Learning Representations, 2026. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ hash/11a09e0aaa74867c6b0719c639fc09f8-Abstract-Conference.html

[28] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, Y. Zhao, S. Peng, H. Guo, X. Zhou, G. Shi, J. Feng, and B. Kang, “Depth Anything 3: Recovering the Visual Space from Any Views,” in International Conference on Learning Representations, 2026. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ hash/e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html

[29] N. Keetha, N. Muller, J. Sch¨ onberger, L. Porzi, Y. Zhang, T. Fischer,¨ A. Knapitsch, D. Zauss, E. Weber, N. Antunes, J. Luiten, M. Lopez-Antequera, S. R. Bulo, C. Richardt, D. Ramanan, S. Scherer, and\` P. Kontschieder, “MapAnything: Universal Feed-Forward Metric 3D Reconstruction,” in International Conference on 3D Vision, 2026. [Online]. Available: https://openreview.net/forum?id=h9J2UUVWat

[30] H. Jin, H. Jiang, H. Tan, K. Zhang, S. Bi, T. Zhang, F. Luan, N. Snavely, and Z. Xu, “LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias,” in International Conference on Learning Representations, 2025, oral presentation. [Online]. Available: https://openreview.net/forum?id=QQBPWtvtcn

[31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention Is All You Need,” in Advances in Neural Information Processing Systems, vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[32] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[33] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding,” Neurocomputing, vol. 568, p. 127063, 2024. [Online]. Available: https://doi.org/10.1016/j.neucom.2023.127063

[34] B. Heo, S. Park, D. Han, and S. Yun, “Rotary Position Embedding for Vision Transformer,” in Proceedings of the European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, vol. 15068, 2024, pp. 289–305. [Online]. Available: https://www.ecva.net/ papers/eccv 2024/papers ECCV/html/1584 ECCV 2024 paper.php

[35] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10 012–10 022. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/Liu Swin Transformer Hierarchical Vision Transformer Using Shifted Windows ICCV 2021 paper.html

[36] K. Wu, H. Peng, M. Chen, J. Fu, and H. Chao, “Rethinking and Improving Relative Position Encoding for Vision Transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10 033–10 041. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/Wu Rethinking and Improving Relative Position Encoding for Vision Transformer ICCV 2021 paper.html

[37] X. Chu, Z. Tian, B. Zhang, X. Wang, and C. Shen, “Conditional Positional Encodings for Vision Transformers,” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=3KWnuT-R1bh

[38] V. Sitzmann, S. Rezchikov, W. T. Freeman, J. B. Tenenbaum, and F. Durand, “Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering,” in Advances in Neural Information Processing Systems, vol. 34, 2021. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html

[39] M. Yin, J. Lu, W. Hu, W. Zhao, S. Ying, and K. Han, “RayPE: Ray-Space Positional Encoding for 3D-Aware Video Generation,” 2026. [Online]. Available: https://arxiv.org/abs/2606.27345

[40] C. Shu, J. Deng, F. Yu, and Y. Liu, “3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3580–3589. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/html/Shu 3DPPE 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object ICCV 2023 paper.html

[41] T. Miyato, B. Jaeger, M. Welling, and A. Geiger, “GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers,” in International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=uJVHygNeSZ

[42] R. Li, B. Yi, J. Liu, H. Gao, Y. Ma, and A. Kanazawa, “Cameras as Relative Positional Encoding,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2025/hash/ 17a7075094632c88cccdd86270ad715b-Abstract-Conference.html

[43] S. Kenney and T. Suzuki, “DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View Transformers,” 2026. [Online]. Available: https://arxiv.org/abs/2606.31585

[44] C. Zhang, B. Li, M. Wei, Y.-P. Cao, C. C. Gambardella, D. Phung, and J. Cai, “Unified Camera Positional Encoding for Controlled Video Generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 38 027–38 037. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/ html/Zhang Unified Camera Positional Encoding for Controlled Video Generation CVPR 2026 paper.html

[45] Y. Wu, M. Jeon, J.-H. R. Chang, O. Tuzel, and S. Tulsiani, “RayRoPE: Projective Ray Positional Encoding for Multi-view Attention,” in European Conference on Computer Vision, 2026. [Online]. Available: https://rayrope.github.io/

[46] Y. Xie, D. Meng, C. Peng, Y. Hu, Q. Herau, M. Tomizuka, and W. Zhan, “URoPE: Universal Relative Position Embedding across Geometric Spaces,” in European Conference on Computer Vision, 2026, oral presentation. [Online]. Available: https://urope-pe.github.io/

[47] S. Jin, Y. Kim, S. Park, and J. C. Ye, “CRePE: Curved Ray Expectation Positional Encoding for Unified-Camera-Controlled Video Generation,” 2026. [Online]. Available: https://arxiv.org/abs/2605.12938

[48] J. Tirado-Gar´ın and J. Civera, “AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 8044–8055. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2025/html/Tirado-Garin AnyCalib On-Manifold Learning for Model-Agnostic Single-View Camera Calibration ICCV 2025 paper.html

[49] M. Song, D. Tao, and S. J. Maybank, “Sparse Camera Network for Visual Surveillance – A Comprehensive Survey,” 2013. [Online]. Available: https://arxiv.org/abs/1302.0446

[50] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,” in European Conference on Computer Vision, 2020, pp. 405–421. [Online]. Available: https://doi.org/10.1007/ 978-3-030-58452-8 24

[51] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian¨ Splatting for Real-Time Radiance Field Rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023. [Online]. Available: https://doi.org/10.1145/3592433

[52] D. Charatan, S. Li, A. Tagliasacchi, and V. Sitzmann, “pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 457–19 467. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/ html/Charatan pixelSplat 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D CVPR 2024 paper.html

[53] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, “MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images,” in Proceedings of the European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, vol. 15079, 2024, pp. 370–386. [Online]. Available: https://www.ecva.net/papers/eccv 2024/papers ECCV/html/ 3187 ECCV 2024 paper.php

[54] Y. Wang, V. C. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. Solomon, “DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 164. PMLR, 2022, pp. 180–191. [Online]. Available: https: //proceedings.mlr.press/v164/wang22b.html

[55] Y. Liu, T. Wang, X. Zhang, and J. Sun, “PETR: Position Embedding Transformation for Multi-View 3D Object Detection,” in European Conference on Computer Vision, 2022, pp. 531–548. [Online]. Available: https://www.ecva.net/papers/eccv 2022/papers ECCV/html/ 4420 ECCV 2022 paper.php

[56] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai, “BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers,” in European Conference on Computer Vision, 2022. [Online]. Available: https://www.ecva.net/papers/eccv 2022/papers ECCV/html/ 694 ECCV 2022 paper.php

[57] S. Wang, Y. Liu, T. Wang, Y. Li, and X. Zhang, “Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3621–3631. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/html/ Wang Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection ICCV 2023 paper.html

[58] Q. Fan, Q. You, X. Han, Y. Liu, Y. Tao, H. Huang, R. He, and H. Yang, “ViTAR: Vision Transformer with Any Resolution,” 2024. [Online]. Available: https://arxiv.org/abs/2403.18361

[59] Y. Liu, Z. Min, Z. Wang, J. Wu, T. Wang, Y. Yuan, Y. Luo, and C. Guo, “WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting,” in International Conference on Machine Learning, 2026. [Online]. Available: https://arxiv.org/abs/2510.10726

[60] H. Peng, H. Li, Y. Dai, Y. Lan, Y. Luo, T. Qi, Z. Zhang, Y. Zhan, J. Zhang, W. Xu, and Z. Liu, “OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 36 485–36 497. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/ html/Peng OmniVGGT Omni-Modality Driven Visual Geometry Grounded Transformer CVPR 2026 paper.html

[61] T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely, “Stereo Magnification: Learning View Synthesis using Multiplane Images,” ACM Transactions on Graphics, vol. 37, no. 4, pp. 65:1–65:12, 2018. [Online]. Available: https://doi.org/10.1145/3197517.3201323

[62] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A Universe of Annotated 3D Objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 142–13 153. [Online]. Available: https: //openaccess.thecvf.com/content/CVPR2023/html/Deitke Objaverse A Universe of Annotated 3D Objects CVPR 2023 paper.html

[63] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in Proc. of the International Conference on Intelligent Robot Systems (IROS), Oct. 2012. [Online]. Available: https://cvg.cit.tum.de/data/datasets/ rgbd-dataset

[64] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black, “A naturalistic open source movie for optical flow evaluation,” in European Conf. on Computer Vision (ECCV), ser. Part IV, LNCS 7577, A. Fitzgibbon et al. (Eds.), Ed. Springer-Verlag, Oct. 2012, pp. 611–625. [Online]. Available: https://sintel.is.tue.mpg.de/

[65] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, June 2012, pp. 3354–3361. [Online]. Available: http://dx.doi.org/10.1109/CVPR.2012. 6248074

[66] D. Azinovic, R. Martin-Brualla, D. B. Goldman, M. Nießner, and´ J. Thies, “Neural rgb-d surface reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 6290–6301. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022/html/Azinovic Neural RGB-D Surface Reconstruction CVPR 2022 paper.html

[67] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon, “Scene coordinate regression forests for camera relocalization in rgb-d images,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2013. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2013/ html/Shotton Scene Coordinate Regression 2013 CVPR paper.html

[68] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, and H. Aanaes, “Large scale multi-view stereopsis evaluation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2014. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2014/html/Jensen Large Scale Multi-view 2014 CVPR paper.html

[69] G. Baruch, Z. Chen, A. Dehghan, T. Dimry, Y. Feigin, P. Fu, T. Gebauer, B. Joffe, D. Kurz, A. Schwartz, and E. Shulman, “ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. [Online]. Available: https://openreview.net/forum?id=tjZjv qh CE

[70] Y. Yao, Z. Luo, S. Li, J. Zhang, Y. Ren, L. Zhou, T. Fang, and L. Quan, “Blendedmvs: A large-scale dataset for generalized multi-view stereo networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. [Online]. Available: https://openaccess.thecvf.com/content CVPR 2020/html/Yao BlendedMVS A Large-Scale Dataset for Generalized Multi-View Stereo Networks CVPR 2020 paper.html

[71] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. M. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra, “Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021. [Online]. Available: https://arxiv.org/abs/2109.08238

[72] M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb, and J. M. Susskind, “Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 10 912–10 922. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/ Roberts Hypersim A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding ICCV 2021 paper.html

[73] Y. Li, L. Jiang, L. Xu, Y. Xiangli, Z. Wang, D. Lin, and B. Dai, “Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3205–3215. [Online]. Available: https: //raw.githubusercontent.com/city-super/MatrixCity/main/README.md

[74] M. Fonder and M. Van Droogenbroeck, “Mid-air: A multimodal dataset for extremely low altitude drone flights,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019. [Online]. Available: https://openaccess.thecvf.com/content CVPRW 2019/html/ UAVision/Fonder Mid-Air A Multi-Modal Dataset for Extremely Low Altitude Drone Flights CVPRW 2019 paper.html

[75] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3d: Learning from rgb-d data in indoor environments,” in International Conference on 3D Vision (3DV), 2017. [Online]. Available: https://niessner.github.io/Matterport/

[76] T. Wu, J. Zhang, X. Fu, Y. Wang, J. Ren, L. Pan, W. Wu, L. Yang, J. Wang, C. Qian, D. Lin, and Z. Liu, “Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 803–814. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2023/html/Wu OmniObject3D Large-Vocabulary 3D Object Dataset for Realistic Perception Reconstruction and CVPR 2023 paper.html

[77] Y. Zhou, Y. Wang, J. Zhou, W. Chang, H. Guo, Z. Li, K. Ma, X. Li, Y. Wang, H. Zhu, M. Liu, D. Liu, J. Yang, Z. Fu, J. Chen, C. Shen, J. Pang, K. Zhang, and T. He, “Omniworld: A multi-domain and multimodal dataset for 4d world modeling,” arXiv preprint arXiv:2509.12201, 2025. [Online]. Available: https://yangzhou24.github.io/OmniWorld/

[78] A. Avetisyan, C. Xie, H. Howard-Jenkins, T.-Y. Yang, S. Aroudj, S. Patra, F. Zhang, D. Frost, L. Holland, C. Orme, J. Engel, E. Miller, R. Newcombe, and V. Balntas, “SceneScript: Reconstructing scenes with an autoregressive structured language model,” in Proceedings of the European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, vol. 15119, 2024, pp. 247–263. [Online]. Available: https://www.ecva.net/papers/eccv 2024/papers ECCV/html/ 7833 ECCV 2024 paper.php

[79] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, “Taskonomy: Disentangling task transfer learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2018/ html/Zamir Taskonomy Disentangling Task CVPR 2018 paper.html

[80] F. Tosi, Y. Liao, C. Schmitt, and A. Geiger, “Smd-nets: Stereo mixture density networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 8942–8952. [Online]. Available: https: //openaccess.thecvf.com/content/CVPR2021/html/Tosi SMD-Nets Stereo Mixture Density Networks CVPR 2021 paper.html

[81] H. Xia, Y. Fu, S. Liu, and X. Wang, “Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 22 378–22 389. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/ Xia RGBD Objects in the Wild Scaling Real-World 3D Object Learning CVPR 2024 paper.html

[82] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht, “Dynamicstereo: Consistent dynamic depth from stereo videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 13 229–13 239. [Online]. Available: https://openaccess.thecvf.com/ content/CVPR2023/html/Karaev DynamicStereo Consistent Dynamic Depth From Stereo Videos CVPR 2023 paper.html

[83] P.-H. Huang, K. Matzen, J. Kopf, N. Ahuja, and J.-B. Huang, “Deepmvs: Learning multi-view stereopsis,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2018/html/ Huang DeepMVS Learning Multi-View CVPR 2018 paper.html

[84] W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer, “Tartanair: A dataset to push the limits of visual slam,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020. [Online]. Available: https://theairlab.org/tartanair-dataset/

[85] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” 2020. [Online]. Available: https://arxiv.org/abs/2001.10773

[86] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov, “Scalability in perception for autonomous driving: Waymo open dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. [Online]. Available: https://openaccess.thecvf.com/content CVPR 2020/html/ Sun Scalability in Perception for Autonomous Driving Waymo Open Dataset CVPR 2020 paper.html

[87] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2018/ html/Zhang The Unreasonable Effectiveness CVPR 2018 paper.html

## APPENDIX A METHOD DETAILS

## A. Image-to-Ray Maps

A pinhole camera with zero skew instantiates $\mathcal { U } _ { c }$ through the intrinsic matrix

$$
\mathbf { K } _ { c } = \left[ \begin{array} { c c c } { f _ { x , c } } & { 0 } & { c _ { x , c } } \\ { 0 } & { f _ { y , c } } & { c _ { y , c } } \\ { 0 } & { 0 } & { 1 } \end{array} \right] ,\tag{16}
$$

with $\mathcal { U } _ { c } ( \mathbf { p } ) = \mathbf { K } _ { c } ^ { - 1 } \mathbf { p }$ . For the four-coefficient fisheye projection in (6), we first normalize the image-plane sample using the intrinsics. We then numerically invert $\rho ( \psi )$ to recover the polar angle ψ from the normalized radius. Together with the azimuth ϕ of the normalized sample, this gives the unit ray $\mathcal { U } _ { c } ( \mathbf { p } ) = ( \sin \psi$ cos ϕ, sin ψ sin $\phi , \cos \psi ) ^ { \intercal }$ . The camera model may also provide a forward projection $\Pi _ { c } ,$ but G-ray requires only the image-to-ray map.

## B. Consistency under Image Resampling

Let A denote the coordinate transformation that a resize or crop induces on the retained image domain, so that $\mathbf { p ^ { \prime } } = \mathbf { A p }$ Suppose the updated image-to-ray map satisfies

$$
\begin{array} { r } { \mathcal { U } _ { c } ^ { \prime } ( { \bf A p } ) \propto \mathcal { U } _ { c } ( { \bf p } ) , } \end{array}\tag{17}
$$

with a positive proportionality factor. Then $\mathbf { p } ^ { \prime }$ and p represent the same camera-local ray and receive the same ray angles under (4) and (5). For a pinhole camera the same operation updates the intrinsics as $\mathbf { K } _ { c } ^ { \prime } = \mathbf { A } \mathbf { K } _ { c } ,$ so that

$$
{ \mathbf { K } _ { c } ^ { \prime } } ^ { - 1 } { \mathbf { p } } ^ { \prime } = ( \mathbf { A } \mathbf { K } _ { c } ) ^ { - 1 } \mathbf { A } \mathbf { p } = \mathbf { K } _ { c } ^ { - 1 } \mathbf { p } ,\tag{18}
$$

which verifies (17) for that case. This equality concerns exactly corresponding continuous image locations. Patch centers resampled onto a different discrete grid need not coincide with those locations and generally represent nearby, rather than identical, viewing rays.

## C. Multiple Positional Samples per Token

A host encoding may assign one or more two-dimensional positional samples to each token, denoted by $\xi _ { c , i , \ell }$ for sample ℓ. Standard 2D RoPE uses a single token-lattice index, whereas a projection-based encoding can provide several pixel locations obtained in a query camera. G-ray maps each sample independently through (3) to (5), while the host encoding retains its aggregation over ℓ. The main text omits the sample index for clarity.

## D. Content-Free Attention Score under Rotary Encoding

This subsection derives the content-free (position-only) attention score used in Fig. 5 and Sec. III-D1. We first consider a rotary encoding parameterized by coordinates ξ, then specialize the result to grid-index RoPE and G-ray.

1) Relative Phase of Planar Rotations: Let $\mathbf { E } ( \pmb { \xi } )$ denote a block-diagonal bank of planar rotations applied to a feature vector, as in standard RoPE and (7). For any query and key features q, k $\in \mathbb { R } ^ { d _ { h } }$

$$
\left( \mathbf { E } ( \pmb { \xi } _ { i } ) \mathbf { q } \right) ^ { \top } \left( \mathbf { E } ( \pmb { \xi } _ { j } ) \mathbf { k } \right) = \mathbf { q } ^ { \top } \mathbf { E } ( \pmb { \xi } _ { i } ) ^ { \top } \mathbf { E } ( \pmb { \xi } _ { j } ) \mathbf { k } .\tag{19}
$$

Because planar rotations compose by angle addition,

$$
\mathbf { E } ( \pmb { \xi } _ { i } ) ^ { \top } \mathbf { E } ( \pmb { \xi } _ { j } ) = \mathbf { E } ( \pmb { \xi } _ { j } - \pmb { \xi } _ { i } ) .\tag{20}
$$

The positional contribution therefore depends on the relative displacement $\Delta \xi = \pmb { \xi } _ { j } - \pmb { \xi } _ { i }$ rather than on absolute coordinates. 2) Content-Free Probe: The position-only diagnostic removes image content by setting $\mathbf { q } = \mathbf { k } = \mathbf { 1 }$ (a constant vector) before applying $\mathbf { E } ( \cdot )$ . Consider a two-channel rotation block at frequency ω with scalar positional coordinate $\xi \in \{ \xi ^ { ( 1 ) } , \xi ^ { ( 2 ) } \}$ The two components of ξ are grid-index coordinates for RoPE and $( \theta _ { x } , \theta _ { y } )$ for G-ray. The corresponding planar rotation acts on a two-dimensional feature pair $( x _ { 1 } , x _ { 2 } )$ as

$$
R _ { \omega \xi } \left[ \begin{array} { l } { x _ { 1 } } \\ { x _ { 2 } } \end{array} \right] = \left[ \cos ( \omega \xi ) x _ { 1 } - \sin ( \omega \xi ) x _ { 2 } \right] .\tag{21}
$$

Substituting the constant pair (1, 1) at positions i and j and taking the inner product yields

$$
\begin{array} { r l } & { \left( R _ { \omega \xi _ { i } } \left[ 1 \right] \right) ^ { \top } \left( R _ { \omega \xi _ { j } } \left[ 1 \right] \right) } \\ & { \qquad = \left( \cos \omega \xi _ { i } - \sin \omega \xi _ { i } \right) \left( \cos \omega \xi _ { j } - \sin \omega \xi _ { j } \right) } \\ & { \qquad + \left( \sin \omega \xi _ { i } + \cos \omega \xi _ { i } \right) \left( \sin \omega \xi _ { j } + \cos \omega \xi _ { j } \right) } \\ & { \qquad = 2 \cos \left( \omega ( \xi _ { i } - \xi _ { j } ) \right) . } \end{array}\tag{22}
$$

Summing over frequency channels and over the two axes of ξ therefore gives the content-free score

$$
s ( i , j ) = \sum _ { a \in \{ 1 , 2 \} } \sum _ { k } 2 \cos \bigl ( \omega _ { k } \bigl ( \xi _ { i } ^ { ( a ) } - \xi _ { j } ^ { ( a ) } \bigr ) \bigr ) ,\tag{23}
$$

where $\omega _ { k }$ follows the frequency schedule of the evaluated encoding. Equation (23) is the raw content-free dot product underlying the PE-only column of Fig. 5. Attention scaling and normalization are applied afterward, so this expression describes the positional score rather than the final attention weights.

3) Specialization to Grid-Index RoPE and G-ray: Grid-index RoPE uses the token-lattice coordinate pair ${ \pmb \xi } = ( m , n )$ . When two views share a patch grid, the same lattice offsets produce the same positional scores regardless of focal length or principal point. The grid-based score therefore does not distinguish their FoVs, even when their visible angular regions differ. G-ray instead sets ${ \pmb \xi } = ( \theta _ { x } , \theta _ { y } )$ via Eqs. (4) and (5). Relative phase then depends on camera-local ray-angle differences. Identical ray-angle coordinates maximize this position-only score, but its multi-frequency cosine form does not generally decrease monotonically with angular separation. In the wide/zoom examples of Fig. 5, the resulting prior emphasizes the shared angula region. The two encodings thus share the cosine mechanism in (23) but differ in the geometric meaning of their positional coordinates.

## APPENDIX B TRAINING DETAILS

## A. 3D Reconstruction

The 3D reconstruction model uses a VGGT-based architecture [21] with a DINOv2 ViT-L/14 backbone (embedding dimension 1024, 24 layers, 16 heads, and patch size 14) and 24 pairs of alternating frame and global attention blocks. A DPT head predict camera-space pointmaps, and a linear patch head predicts world-space pointmaps. A separate pose-regression head processes the final patch features with residual convolutions and global average pooling, followed by an MLP with translation and quaternion output branches.

The training mixture comprises ARKitScenes [69], BlendedMVS [70], Habitat-Matterport 3D (HM3D) [71], Hypersim [72], MatrixCity [73], MegaDepth [7], Mid-Air [74], Matterport3D [75], OmniObject3D [76], OmniWorld-Game [77], Aria Synthetic Environments (ASE) [78], ScanNet++ [9], Taskonomy [79], UnrealStereo4K [80], WildRGB-D [81], Dynamic Replica [82], MVS-Synth [83], TartanAir [84], Virtual KITTI 2 [85], and the Waymo Open Dataset [86].

We initialize the image encoder with pretrained DINOv2 weights and train the 3D reconstruction model in two stages. Both stages use 8 NVIDIA A100 GPUs with distributed data parallelism. Stage 1 trains for 120 epochs at a 224-pixel long side using AdamW with a peak learning rate of $5 \times 1 0 ^ { - 5 }$ and weight decay of 0.05. The schedule uses 5% linear warmup from $1 0 ^ { - 8 }$ followed by cosine decay to $\bar { 1 } 0 ^ { - 8 }$ . Stage 1 uses BF16 mixed precision, gradient clipping at max-norm 1.0, and up to 240 images per GPU. Each batch samples 2 to 24 views per sequence and a shared aspect ratio between 0.33 and 1.0, with the short side aligned to the 14-pixel patch size. Stage 2 fine-tunes the Stage-1 checkpoint for 60 epochs at a 518-pixel long side. It uses an initial learning rate of $1 0 ^ { - 7 }$ , cosine decay to $1 0 ^ { - 1 0 }$ without warmup, and up to 48 images per GPU. Both stages are configured for 800 gradient-update steps per epoch without gradient accumulation, corresponding to approximately 96 k and 48 k updates, respectively.

The training objective is a weighted sum of (i) a camera loss (λ=5.0) computing L1 error on the final predicted absolute translation and quaternion rotation, (ii) a camera-space point loss (λ=1.0) with confidence-weighted L2 regression and a normal-based gradient loss after outlier filtering at the 98th percentile, and (iii) a world-space point loss (λ=1.0) using the same formulation. Ground truth is scale-normalized by the mean world-point distance.

## B. Novel View Synthesis

All NVS variants share the LVSM decoder-only backbone [30], with 6 transformer layers, model dimension 1152, 8 heads, feed-forward dimension 1024, and patch size 8. Training uses 2 context views and 1 target view at 256×256 resolution. Models are trained separately on RE10K and Objaverse. The Objaverse training set contains approximately 72 k objects, each rendered from 8 viewing directions with additional FoV and camera-distance variants.

Each positional encoding variant is trained independently on a single NVIDIA RTX 4090 GPU with batch size 4 and FP16 mixed precision. We use AdamW with a learning rate of $4 { \times } 1 0 ^ { - 4 } , \ \beta _ { 1 } { = } 0 . 9 , \ \beta _ { 2 } { = } 0 . 9 5$ , and weight decay 0.5. Training runs for 80 k steps, with 2 500 steps of linear warmup followed by cosine decay.

The loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { M S E } } + 0 . 5 \mathcal { L } _ { \mathrm { L P I P S } }$ , where $\mathcal { L } _ { \mathrm { M S E } }$ is the pixel-wise mean squared error between the predicted and ground-truth target view, and ${ \mathcal { L } } _ { \mathrm { L P I P S } }$ is the AlexNet-based perceptual loss [87]. No explicit depth, pose, or adversarial losses are used. In RayRoPE, per-layer depth and uncertainty predictions are used within the positional encoding and receive no separate supervision.

## C. Positional Encoding Frequency Parameters

G-ray retains each host encoding’s allocation of channels to geometric components and uses host-specific frequency settings. An angular subspace with $n _ { \Omega }$ frequency components applies independent two-channel rotations,

$$
\mathbf { R } _ { \Omega } ( \theta ) = \bigoplus _ { k = 0 } ^ { n _ { \Omega } - 1 } \left[ \cos ( \omega _ { k } \theta ) \right. \quad \left. \cos ( \omega _ { k } \theta ) \right] .\tag{24}
$$

For 3D reconstruction and GTA, the frequencies follow (8), with the number of components determined by the angular subspace. The frequency scale $\alpha$ and base $\beta$ correspond to freq\_scale and freq\_base in these implementations. All ray-angle coordinates are expressed in radians.

a) 3D Reconstruction: The aggregator uses α=1 and $\beta { = } 1 0 0$ . Its head channels are divided equally between $\theta _ { x }$ and $\theta _ { y } ,$ giving $n _ { \Omega } { = } d _ { h } / 4$ frequencies per angle. The value and inverse-output transformations in (13) are enabled.

b) $G T A + G \cdot r a y \colon$ The angular subspace uses $\alpha { = } 1 0$ and $\beta { = } 1 0 0$ , while the remaining geometric components retain the host’s formulation. The frequency-scale ablation in Sec. IV-D2 varies α with $\beta$ fixed.

c) RayRoPE + G-ray: RayRoPE uses a period-based parameterization rather than the scale/base convention above. For its ray-angle coordinates, the maximum period is $P _ { \mathrm { m a x } } { = } 8 ( \pi / 2 ) { = } 4 \pi$ and the ratio between consecutive frequencies is $b _ { \mathrm { R } } { = } 3 ,$

$$
\omega _ { k } ^ { \mathrm { R } } = \frac { 2 \pi } { P _ { \mathrm { m a x } } } b _ { \mathrm { R } } ^ { k } , \qquad k = 0 , \ldots , n _ { \Omega } - 1 .\tag{25}
$$

In the implementation, this period is specified by max\_period = MAX\_FOV ×8, with $\mathsf { M A X \_ F O V } = \pi / 2$ . RayRoPE’s freq\_base denotes $b _ { \mathrm { R } }$ , not the decay base β in (8). The frequency settings for its other geometric components are unchanged.

## APPENDIX C ADDITIONAL NVS EVALUATION

## A. Objaverse Target-View Definitions

All six Objaverse protocols evaluate the same 8 000 test objects under heterogeneous-FoV inputs. Each object is rendered from eight viewing directions, with an independently sampled FoV and a corresponding camera distance for each native view. Two native views from different directions are held fixed as context across the six protocols. For each native view, a focal variant changes its FoV at the same pose, while a dolly variant changes only the camera distance and retains the native FoV. Here, native FoV denotes the per-view sampled FoV, not a common FoV shared by all views.

The three main regimes in Table IV differ in the target camera pose and FoV. FoV change uses the two focal variants at the context poses. Novel viewpoint uses the six native views at unseen poses, without further modifying their independently sampled FoVs. Novel viewpoint + FoV change uses the six focal variants at those unseen poses.

The additional protocols include targets with radial camera motion. FoV / dolly change combines two focal and two dolly variants at the context viewing directions, for four targets per object. Novel viewpoint + FoV / dolly change combines six focal and six dolly variants at unseen viewing directions, for twelve targets. Original mixed protocol combines the four FoV/doll targets with the six native novel-viewpoint targets, for ten targets. It excludes the focal and dolly variants at unseen viewing directions and is not an average over all six protocols. The FoV/dolly protocols mix changes in FoV and camera distance, rather than isolating either factor.

Method publication details are given in Tables III and IV.

TABLE IX  
ADDITIONAL OBJAVERSE NVS PROTOCOLS, MEASURED BY PSNR (DB ↑), USING THE INPUTS AND MODELS OF TABLE IV. ORIGINAL MIXED PROTOCOL COMBINES FOV/DOLLY TARGETS AT OBSERVED VIEWING DIRECTIONS WITH NATIVE-FOV TARGETS AT NOVEL VIEWPOINTS. IT IS NOT AN AVERAGE OVER ALL TARGET SETS. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.
<table><tr><td>Method</td><td>FoV / dolly change</td><td>Novel viewpoint + FoV / dolly change</td><td>Original mixed protocol</td></tr><tr><td>Plücker</td><td>26.43</td><td>13.14</td><td>20.52</td></tr><tr><td>PRoPE</td><td>26.41</td><td>12.80</td><td>21.21</td></tr><tr><td>GTA</td><td>26.12</td><td>12.97</td><td>20.68</td></tr><tr><td> $\mathbf { G } \mathbf { T } \mathbf { A } + \mathbf { G } \mathbf { - } \mathbf { r } \mathbf { a } \mathbf { y }$ </td><td>26.62</td><td>13.72</td><td>21.13</td></tr><tr><td>RayRoPE</td><td>28.19</td><td>12.85</td><td>21.32</td></tr><tr><td>RayRoPE + G-ray</td><td>28.01</td><td>14.23</td><td>21.68</td></tr></table>

Table IX reports results under these additional protocols. GTA + G-ray improves over GTA in all three protocols. RayRoPE + G-ray improves by 1.38 dB in Novel viewpoint + FoV / dolly change and by 0.36 dB in Original mixed protocol, but is 0.18 dB lower in FoV / dolly change. Together with the main results, these comparisons show that gains depend on the target-view setting.

## B. Additional NVS Visualizations

Figures 12 and 13 provide additional NVS comparisons on RE10K [61] and Objaverse [62], respectively. Each row contains two context views, a ground-truth target, and predictions from the Plucker baseline [30], GTA [41], GTA + G-ray, RayRoPE [45],¨ and $\mathrm { R a y R o P E + G – r a y }$ . The paired columns show how ray-angle coordinates affect scene structure and texture in individual examples. The overlaid PSNR values measure individual target images and complement the aggregate results in Tables III and IV.

Ref View  
Ref View 0  
GT Target  
![](images/6c4ff7cd5ec4b930a4358a1244473a622647114665fdcc00202484c0f4269fd5.jpg)  
Fig. 12. Additional RE10K NVS examples under controlled FoV variation with $z _ { \mathrm { m a x } } { = } 2 . 0 .$ Rows show different scenes. From left to right, columns contain two context views, the ground-truth target, and predictions from Plucker, GTA, GTA + G-ray, RayRoPE, and RayRoPE + G-ray. Numbers on predictions¨ report target-image PSNR (dB ↑). The paired host-encoding columns allow comparison of structural alignment, blur, and ghosting.

![](images/2b755ae8c031fce13b9f74b1b13a81ce9fc36dc27a1ddce8320b8e1f274e906d.jpg)

Fig. 13. Additional Objaverse NVS examples under FoV heterogeneity. Rows show different objects. From left to right, columns contain two context views, the ground-truth target, and predictions from Plucker, GTA, GTA + G-ray, RayRoPE, and RayRoPE + G-ray. Numbers on predictions report target-image¨ PSNR (dB ↑). These examples illustrate changes in object shape and appearance; the gains from ray-angle coordinates vary across targets.

# APPENDIX D

# PER-DATASET 3D RECONSTRUCTION RESULTS

Table X reports the per-dataset 50-view results used to compute the three-dataset averages in Table I. Method publication details are given in Table I.

TABLE X  
PER-DATASET 3D RECONSTRUCTION RESULTS AT 50 INPUT VIEWS. MIXED-PROJECTION SCANNET++ COMBINES NON-PINHOLE DSLR AND PINHOLE IPHONE VIEWS AND USES RAY-DEPTH METRICS. THE W/C GROUP RECEIVES EXTERNAL IMAGE-TO-RAY CALIBRATION THROUGH CAMERA-MODEL PARAMETERS AND UNPROJECTION OR AN EQUIVALENT UNIT-RAY MAP. THE W/O C GROUP RECEIVES NO EXTERNAL CALIBRATION AND INCLUDES IMAGE-ONLY METHODS AND G-RAY WITH ANYCALIB-ESTIMATED RAY MAPS. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC WITHIN EACH DATASET.
<table><tr><td>Dataset</td><td colspan="2">Setting Method</td><td></td><td>Points rel ↓ Rays err° ↓ Depth rel ↓ Pts Inl. ↑ Pose ATE ↓ Dp. Inl. ↑</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="18">ScanNet++ (dist. fish.+pin.)</td><td rowspan="4">w/C</td><td>G-ray</td><td>0.0583</td><td>0.72</td><td>0.0449</td><td>75.1</td><td>0.0246</td><td>66.0</td></tr><tr><td>π3-X</td><td>0.2080</td><td>4.29</td><td>0.1052</td><td>25.8</td><td>0.1005</td><td>21.8</td></tr><tr><td>WorldMirror</td><td>0.2347</td><td>4.16</td><td>0.1010</td><td>27.4</td><td>0.0969</td><td>22.9</td></tr><tr><td>MapAnything</td><td>0.2246</td><td>3.39</td><td>0.1729</td><td>26.2</td><td>0.1312</td><td>9.6</td></tr><tr><td>DA3</td><td>G-ray + AnyCalib</td><td>0.0872</td><td>2.18</td><td>0.0707</td><td>61.4</td><td>0.0518</td><td>36.7</td></tr><tr><td></td><td></td><td>0.4646</td><td>9.94</td><td>0.2501</td><td>14.2</td><td>0.2152</td><td>5.1</td></tr><tr><td rowspan="5">w/o C</td><td>DA3-Nested</td><td>0.4549</td><td>9.58</td><td>0.2367</td><td>15.5</td><td>0.2012</td><td>5.9</td></tr><tr><td>MapAnything</td><td>0.2806</td><td>5.28</td><td>0.1906</td><td>24.1</td><td>0.1536</td><td>7.8</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>0.2846</td><td>6.95</td><td>0.1724</td><td>22.4</td><td>0.1457</td><td>7.0</td></tr><tr><td>VGGT</td><td>0.4260</td><td>7.96</td><td>0.1942</td><td>14.2</td><td>0.2068</td><td>9.6</td></tr><tr><td>VGGT-Ω</td><td>0.4133</td><td>7.83</td><td>0.1736</td><td>16.3</td><td>0.1556</td><td>9.9</td></tr><tr><td></td><td>OmniVGGT WorldMirror</td><td>0.2035 0.3306</td><td>3.98</td><td>0.1225 0.1405</td><td>34.1 20.9</td><td>0.1130 0.1349</td><td>12.5 16.5</td></tr><tr><td rowspan="10">PandaSet</td><td rowspan="5">w/C</td><td></td><td></td><td>5.78</td><td></td><td></td><td></td><td></td></tr><tr><td>G-ray π3-X</td><td>0.1099 0.0959</td><td>0.27</td><td>0.1378</td><td>34.7</td><td>0.0117 0.0106</td><td>27.5</td></tr><tr><td>WorldMirror</td><td>0.1264</td><td>2.33</td><td>0.1115</td><td>45.7 21.7</td><td>0.0127</td><td>28.1</td></tr><tr><td>MapAnything</td><td>0.1233</td><td>2.39</td><td>0.1508</td><td></td><td>0.0171</td><td>19.2</td></tr><tr><td></td><td></td><td>2.58</td><td>0.1406</td><td>35.8</td><td></td><td>26.5</td></tr><tr><td rowspan="6">DA3 w/o C</td><td>G-ray + AnyCalib</td><td>0.2191 0.4725</td><td>3.71</td><td>0.2607</td><td>22.0</td><td>0.0820</td><td>10.8</td></tr><tr><td>DA3-Nested</td><td>0.4172</td><td>8.47</td><td>0.5302 0.4039</td><td>12.2</td><td>0.2980 0.1591</td><td>7.3</td></tr><tr><td></td><td>0.3245</td><td>6.45 4.86</td><td>0.3325</td><td>13.9 22.0</td><td>0.1349</td><td>5.0</td></tr><tr><td>MapAnything</td><td>0.5000</td><td>4.15</td><td>0.3355</td><td>18.9</td><td>0.2549</td><td>8.4</td></tr><tr><td> $\pi ^ { 3 }$  VGGT</td><td>0.4241</td><td>4.07</td><td>0.3604</td><td>11.1</td><td>0.2161</td><td>9.2</td></tr><tr><td>VGGT-Ω</td><td>0.1676</td><td>4.04</td><td>0.2098</td><td>30.6</td><td>0.0680</td><td>9.1</td></tr><tr><td></td><td>OmniVGGT</td><td>0.4339</td><td>7.18</td><td>0.7329</td><td>6.3</td><td>0.2925</td><td>15.6 4.0</td></tr><tr><td></td><td>WorldMirror</td><td>0.1854</td><td>3.42</td><td>0.2717</td><td>20.0</td><td>0.0767</td><td>13.1</td></tr><tr><td rowspan="9">ETH3D</td><td rowspan="4">w/C</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>G-ray π3-X</td><td>0.0549</td><td>0.15</td><td>0.0405</td><td>64.6</td><td>0.0300</td><td>62.2</td></tr><tr><td>WorldMirror</td><td>0.0352 0.0401</td><td>0.33 0.24</td><td>0.0228 0.0290</td><td>74.7 71.8</td><td>0.0163 0.0227</td><td>76.9 72.5</td></tr><tr><td>MapAnything</td><td>0.0636</td><td>0.35</td><td>0.0461</td><td>50.4</td><td>0.0365</td><td>49.7</td></tr><tr><td>G-ray + AnyCalib</td><td></td><td>0.0977</td><td></td><td>0.0908</td><td>46.8</td><td>0.0767</td><td>25.6</td></tr><tr><td>DA3</td><td></td><td>0.0952</td><td>1.56 1.82</td><td>0.0679</td><td>49.5</td><td>0.0524</td><td>38.4</td></tr><tr><td>w/o C</td><td>DA3-Nested</td><td>0.1113</td><td>2.11</td><td>0.0756</td><td>47.7</td><td>0.0730</td><td>36.4</td></tr><tr><td></td><td>MapAnything</td><td>0.1162</td><td>1.57</td><td>0.0879</td><td>34.9</td><td>0.0738</td><td>25.8</td></tr><tr><td rowspan="5"></td><td> $\pi ^ { 3 }$ </td><td>0.0638</td><td>1.79</td><td>0.0738</td><td>63.0</td><td>0.0603</td><td>47.8</td></tr><tr><td>VGGT</td><td>0.1575</td><td>0.90</td><td>0.0699</td><td>43.1</td><td>0.1155</td><td>39.6</td></tr><tr><td>VGGT-Ω</td><td>0.0225</td><td>0.49</td><td>0.0152</td><td>91.6</td><td>0.0105</td><td>88.7</td></tr><tr><td>OmniVGGT</td><td>0.1750</td><td>1.74</td><td>0.1038</td><td>36.9</td><td>0.1388</td><td>25.1</td></tr><tr><td>WorldMirror</td><td>0.0672</td><td>0.82</td><td>0.0541</td><td>55.0</td><td>0.0400</td><td>45.7</td></tr></table>

a) Three-Dataset Average: With supplied calibration, G-ray leads all six cross-dataset metrics at 50 views, including a pointmap relative error of 0.0744, a ray-direction error of 0.38<sup>◦</sup>, and 58.1% point inliers. π<sup>3</sup>-X ranks second on five metrics, while MapAnything ranks second on ray-direction error. Without externally supplied calibration, G-ray + AnyCalib achieves the lowest average pointmap error (0.1347), ray-direction error (2.49<sup>◦</sup>), and pose ATE (0.0702). VGGT-Ω remains strongest in this group on depth error and point and depth inliers. The largest gains over image-only methods occur on mixed-projection ScanNet++, while the other datasets reveal complementary strengths of the baselines.

b) ScanNet++ (Mixed Projections, Ray Depth): This protocol tests generalization from pinhole training images to mixed pinhole and non-pinhole inputs through the calibrated ray-map interface. With ground-truth ray maps, G-ray achieves the best result on every reported metric, including 0.0583 pointmap relative error, 0.0449 ray-depth relative error, and 75.1% point inliers. Among the other calibrated methods, π<sup>3</sup>-X and WorldMirror improve over MapAnything on several depth and pose metrics but remain less accurate than G-ray on pointmaps and ray depth. Replacing ground-truth calibration with AnyCalib estimates yields 0.0872 pointmap error and 61.4% point inliers. This estimated-calibration variant ranks second across all six ScanNet++ metrics, behind G-ray with ground-truth calibration and ahead of every image-only baseline.

c) PandaSet (Rigid Multi-Camera $R i g ) . ~ \pi ^ { 3 }  – X$ leads most metrics on PandaSet, while G-ray achieves the lowest raydirection error (0.27<sup>◦</sup>) and ranks second on pointmap error, depth error, pose ATE, and depth inliers. WorldMirror with supplied calibration improves over its image-only counterpart, but trails G-ray and $\pi ^ { 3 } \mathrm { - } \mathrm { \mathbf { X } }$ on the primary error measures. Unlike mixed-projection ScanNet++, PandaSet uses a rigid rig with a fixed imaging geometry for each camera. The results show that the advantage of G-ray depends on the evaluation setting and does not extend to every metric of every heterogeneous rig.

d) ETH3D (Controlled FoV Variation): VGGT-Ω leads most geometry metrics on ETH3D. G-ray achieves the lowest ray-direction error (0.15<sup>◦</sup>). Among methods with supplied calibration, $\pi ^ { 3 } \mathrm { - } \mathrm { \mathbf { X } }$ and WorldMirror are strong, with MapAnything trailing on all six ETH3D columns. Here, center cropping induces FoV differences within a shared pinhole projection model. The gains are concentrated in ray-direction accuracy, rather than a uniform improvement across all 3D reconstruction metrics.

## APPENDIX E

## HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS

This appendix reports complete relative-pose, video-depth, and multi-view 3D reconstruction results under the homogeneous pinhole 3D reconstruction protocols [27] summarized in Sec. IV-B2. The undistorted ETH3D split here differs from the ETH3D setting with controlled FoV variation in Table I. Accuracy and completeness are measured in millimetres on DTU and in metres on the other 3D reconstruction datasets. Normal consistency is dimensionless. KITTI video-depth evaluation uses 64-frame windows for MapAnything and complete sequences for the other methods. Method publication details are given in Table II.

TABLE XI  
RELATIVE-POSE ACCURACY ON TUM AND SINTEL UNDER HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS. ATE AND TRANSLATIONAL RPE USE EACH DATASET’S DISTANCE UNITS, WHILE ROTATIONAL RPE IS IN DEGREES. LOWER IS BETTER FOR ALL METRICS. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN.
<table><tr><td rowspan="2">Method</td><td colspan="3">TUM</td><td colspan="3">Sintel</td><td rowspan="2">Avg ATE ↓</td></tr><tr><td>ATE ↓</td><td> $\mathrm { R P E } _ { t } \downarrow$ </td><td>RPEr↓</td><td>ATE ↓</td><td> $\mathrm { R P E } _ { t } \downarrow$ </td><td>RPEr↓</td></tr><tr><td>G-ray</td><td>0.0122</td><td>0.0113</td><td>0.364</td><td>0.110</td><td>0.0598</td><td>0.406</td><td>0.061</td></tr><tr><td>MapAnything</td><td>0.0179</td><td>0.0187</td><td>0.359</td><td>0.219</td><td>0.0747</td><td>0.656</td><td>0.119</td></tr><tr><td>VGGT</td><td>0.0090</td><td>0.0095</td><td>0.314</td><td>0.171</td><td>0.0582</td><td>0.449</td><td>0.090</td></tr><tr><td>VGGT-Ω</td><td>0.0044</td><td>0.0046</td><td>0.274</td><td>0.040</td><td>0.0255</td><td>0.232</td><td>0.022</td></tr></table>

TABLE XII

VIDEO-DEPTH ACCURACY ON SINTEL AND KITTI UNDER HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS. LOWER IS BETTER FOR ABSREL, SQREL, AND RMSE, WHILE HIGHER IS BETTER FOR δ<1.25. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN.
<table><tr><td rowspan="2">Method</td><td colspan="4">Sintel</td><td colspan="4">KITTI</td></tr><tr><td>AbsRel ↓</td><td>SqRel ↓</td><td>RMSE↓</td><td>δ&lt;1.25 ↑</td><td>AbsRel ↓</td><td>SqRel ↓</td><td>RMSE↓</td><td>δ&lt;1.25 ↑</td></tr><tr><td>G-ray</td><td>0.357</td><td>3.74</td><td>6.20</td><td>0.560</td><td>0.092</td><td>1.66</td><td>5.52</td><td>0.943</td></tr><tr><td>MapAnything</td><td>0.401</td><td>4.01</td><td>5.55</td><td>0.620</td><td>0.083</td><td>0.774</td><td>3.97</td><td>0.946</td></tr><tr><td>VGGT</td><td>0.324</td><td>3.02</td><td>6.20</td><td>0.612</td><td>0.082</td><td>0.710</td><td>4.46</td><td>0.933</td></tr><tr><td>VGGT-Ω</td><td>0.251</td><td>4.59</td><td>4.90</td><td>0.834</td><td>0.072</td><td>0.638</td><td>3.88</td><td>0.952</td></tr></table>

TABLE XIII

INDOOR MULTI-VIEW 3D RECONSTRUCTION UNDER HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS. ACCURACY (ACC) AND COMPLETENESS (COMP) ARE MEASURED IN METRES, WITH LOWER VALUES BETTER. HIGHER NORMAL CONSISTENCY (NC) IS BETTER. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC AND DATASET.
<table><tr><td rowspan="2">Method</td><td colspan="3">NRGBD-sparse</td><td colspan="3">NRGBD-dense</td><td colspan="3">7-Scenes-sparse</td><td colspan="3">7-Scenes-dense</td></tr><tr><td>Acc</td><td>Comp</td><td>NC</td><td>Acc</td><td>Comp</td><td>NC</td><td>Acc</td><td>Comp</td><td>NC</td><td>Acc</td><td>Comp</td><td>NC</td></tr><tr><td>G-ray</td><td>0.037</td><td>0.041</td><td>0.899</td><td>0.023</td><td>0.020</td><td>0.870</td><td>0.038</td><td>0.037</td><td>0.772</td><td>0.019</td><td>0.023</td><td>0.685</td></tr><tr><td>MapAnything</td><td>0.088</td><td>0.100</td><td>0.852</td><td>0.036</td><td>0.024</td><td>0.850</td><td>0.059</td><td>0.120</td><td>0.722</td><td>0.023</td><td>0.022</td><td>0.672</td></tr><tr><td>VGGT</td><td>0.058</td><td>0.073</td><td>0.871 0.029</td><td></td><td>0.019</td><td>0.857</td><td>0.055</td><td>0.066</td><td>0.724</td><td>0.025</td><td>0.023</td><td>0.670</td></tr><tr><td>VGGT-Ω</td><td>0.022</td><td>0.018</td><td>0.9180.014</td><td></td><td>0.008</td><td>0.882</td><td>0.029</td><td>0.032</td><td>0.7660.018</td><td></td><td>0.018 0.688</td><td></td></tr></table>

TABLE XIV  
3D RECONSTRUCTION ON DTU AND UNDISTORTED ETH3D UNDER HOMOGENEOUS PINHOLE 3D RECONSTRUCTION PROTOCOLS. ACCURACY (ACC) AND COMPLETENESS (COMP) ARE MEASURED IN MILLIMETRES ON DTU AND METRES ON ETH3D, WITH LOWER VALUES BETTER. HIGHER NORMAL CONSISTENCY (NC) IS BETTER. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN. THIS ETH3D SPLIT DIFFERS FROM THE CONTROLLED FOV VARIATION SETTING IN TABLE I.
<table><tr><td>Method</td><td colspan="3">DTU (mm)</td><td colspan="3">ETH3D (m)</td></tr><tr><td></td><td>Acc↓</td><td>Comp ↓</td><td>NC ↑</td><td>Acc ↓</td><td>Comp ↓</td><td>NC ↑</td></tr><tr><td>G-ray</td><td>3.16</td><td>2.12</td><td>0.708</td><td>0.303</td><td>0.329</td><td>0.831</td></tr><tr><td>MapAnything</td><td>2.22</td><td>1.91</td><td>0.687</td><td>0.490</td><td>0.578</td><td>0.804</td></tr><tr><td>VGGT</td><td>2.61</td><td>2.14</td><td>0.687</td><td>0.277</td><td>0.283</td><td>0.849</td></tr><tr><td>VGGT-Ω</td><td>1.22</td><td>1.89</td><td>0.680</td><td>0.074</td><td>0.062</td><td>0.923</td></tr></table>

## APPENDIX F ADDITIONAL EVALUATION DETAILS AND SCOPE

## A. Off-the-Shelf Intrinsic Estimates

We separately evaluate AnyCalib, a single-image estimator that regresses per-pixel viewing rays and recovers modelspecific calibration [48], on the three 3D reconstruction datasets. The ScanNet++ calibration-only subset uses the undistorted mixed-device protocol rather than the mixed-projection protocol. Table VIII(a) measures the estimator’s calibration accuracy independently of 3D reconstruction. Relative focal error averages the relative errors of $f _ { x }$ and $f _ { y } ,$ while principal-point error is measured in pixels.

AnyCalib’s average focal errors are 7.1% on ETH3D and 8.6% on ScanNet++, with larger focal and principal-point errors on PandaSet. Unlike the uniform focal perturbations in Sec. IV-E3, estimated calibration contains coupled, image-dependent errors, motivating an end-to-end 3D reconstruction evaluation.

Table VIII(b) reports 16-view G-ray 3D reconstruction when conditioning geometry is replaced by off-the-shelf estimates while ground-truth rays and depth remain the evaluation targets. On undistorted mixed-device ScanNet++, we compare ground truth intrinsics with estimates from single-image AnyCalib [48] and multi-view VGGT [21]. On the mixed-projection protocol, DSLR views retain their non-pinhole projection while iPhone views use undistorted pinhole images, and both the ground-truth and estimated conditioning geometry are represented as unit-ray maps. AnyCalib uses the four-coefficient Kannala–Brandt model for non-pinhole DSLR views and the pinhole model for iPhone views. Depth error uses z-depth for the undistorted protocol and ray depth for the mixed-projection protocol.

On undistorted mixed-device ScanNet++, VGGT-estimated intrinsics outperform single-image AnyCalib, with point relative error of 0.066 versus 0.116 and point inliers of 71.0% versus 50.3%. Both estimators remain less accurate than ground-truth calibration. On the mixed-projection protocol, AnyCalib-estimated unit-ray maps yield a pointmap relative error of 0.088, a ray-direction error of 2.16<sup>◦</sup>, and point inliers of 60.2%, compared with 0.059, 0.71<sup>◦</sup>, and 72.9% under ground-truth ray maps. These results demonstrate the use of estimated image-to-ray geometry for non-pinhole 3D reconstruction while quantifying the remaining gap to ground-truth calibration.

## B. Resolution and Projection-Model Scope

The resolution evaluation in Fig. 10 covers four-view ETH3D inputs up to 2× the 518-pixel training long side, using calibrated pinhole images and digital resampling. Mixed-projection ScanNet++ further evaluates non-pinhole DSLR and pinhole iPhone inputs with ray-depth metrics. These results establish evidence for the tested resolutions and calibrated projection models, rather than general support for arbitrary non-pinhole systems. Panoramic, catadioptric, and rolling-shutter imaging require appropriate image-to-ray geometry and separate end-to-end evaluation.

## C. Controlled FoV Variation Scope

Objaverse varies FoV during rendering, while the real-image evaluations synthesize controlled FoV variation through cropping and resizing. The aspect-preserving 256×144 RE10K evaluation avoids upsampling typical 640×360 source images within the tested zoom range. These protocols do not capture changes in point-spread function, distortion, or exposure that may accompany optical zoom, so robustness to those effects remains to be evaluated.