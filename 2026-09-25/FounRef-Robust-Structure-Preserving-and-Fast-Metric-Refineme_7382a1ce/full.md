# FounRef: Robust, Structure-Preserving, and Fast Metric Refinement of Frozen Monocular Foundation Priors with Sparse Anchors

Dan Halperin

Mirko Mählisch

University of the Bundeswehr Munich, Germany

## Abstract

Dense metric depth from cameras is essential to real-world 3D applications, yet achieving accuracy, faithful surface geometry, and fast inference simultaneously remains challenging. Monocular foundation models provide rich, transferable geometric priors but lack reliable metric scale, while depthcompletion networks recover metric depth at the cost of geometric fidelity, cross-domain robustness, or speed. We present FounRef, a training-free method that aligns a frozen monocular foundation prior with sparse metric anchors to produce dense metric depth. FounRef is modular by design: its depth prior, anchor source, and refinement solver can each be replaced independently. We instantiate FounRef with MoGe-2 and LiDAR anchors. FounRef validates each anchor against the prior’s dense depth prediction, rejecting inconsistencies caused by cross-sensor misalignment that geometryonly filters cannot detect. It then applies global and local metric corrections through a structure-preserving solver, retaining the prior’s fine-grained geometry. FounRef requires no task-specific training and operates out of the box across unfamiliar cameras and scenes. On out-of-domain data, it delivers up to 24% lower depth error, 92% lower surfacenormal noise, and almost 15× faster inference than DMD3C, a state-of-the-art depth-completion network. By decoupling metric alignment from geometry prediction, FounRef provides an accurate, geometrically faithful, and eficient approach to dense metric depth that can directly benefit from future advances in foundation models and metric sensors.

## 1 Introduction

Estimating metric depth from a single image is fundamentally ill posed. An image provides strong cues about the relative structure of a scene, but it does not provide enough geometric information to determine metric scale reliably across arbitrary scenes and camera settings (Guizilini et al., 2023). Nevertheless, dense metric depth is essential for real-world applications such as autonomous systems, mapping, and the construction of 3D datasets.

Recent monocular foundation models produce increasingly detailed and transferable depth predictions, such as

![](images/484c9b4832e6df123401861b0586d2924596ee79b0c2614d39ae5507a120da69.jpg)  
Figure 1: Accuracy is not the only axis. Metric error against surface dispersion on KITTI (n=500 frames, identical anchors for every method); marker area grows with runtime. A single configuration change moves our point to 0.663 m at 11.2<sup>◦</sup> without afecting runtime

MoGe-2 (Wang et al., 2025a), UniDepth (Piccinelli et al., 2024), DepthPro (Bochkovskii et al., 2025) and Metric3D-V2 (Hu et al., 2024). They often recover sharp boundaries and plausible surface geometry on unfamiliar scenes. However, their metric calibration remains imperfect, especially under domain shift and most notably at long ranges. In many practical settings, a sensor rig provides only an RGB image, camera intrinsics, and sparse metric anchors, such as a LiDAR scan projected into the image. The challenge is therefore not only to obtain metric depth, but also to preserve object boundaries and smooth physical surfaces while remaining eficient enough to process large image collections.

Depth-completion networks address this problem by learning to combine RGB images and sparse depth measurements, from early sparse-to-dense regression (Ma and Karaman, 2018) to recent models such as DMD3C (Liang et al., 2025) and OMNI-DC (Zuo et al., 2025). Their accuracy can be high, but it is tied to the training procedure, the available training data, and the sparse-depth patterns seen during training; adapting them to a new sensor rig or data distribution may require retraining or fine-tuning. Moreover, as illustrated in Figure 1, optimizing metric error alone does not guarantee coherent surfaces: a method can improve depth metrics while introducing artifacts plainly visible in normal maps and at object boundaries (Figure 6).

To avoid this trade-of, FounRef uses a frozen monocular foundation model as a geometric prior, sparse metric anchors as direct observations, and a lightweight per-image refinement procedure. It globally calibrates the prior, validates anchors against a locally refined reference surface, and propagates the retained corrections along surfaces while preserving depth discontinuities. Our algorithm refines the prior rather than replacing it, so its output remains limited by the prior where anchors are sparse; better priors lead to better refined depth maps. Retraining for a new sensor setup is expensive and can cause a model to fit sparse, noisy anchor points rather than preserve the surface between them. Instead, FounRef exposes metric evidence per image without retraining, while retaining coherent geometry for downstream tasks.

## 2 Related Work

Depth datasets and evaluation. The KITTI depthcompletion benchmark (Uhrig et al., 2017), built on the KITTI dataset (Geiger et al., 2013), and NYU Depth V2 (Silberman et al., 2012) provide refined dense targets for depthcompletion evaluation. In contrast, nuScenes (Caesar et al., 2020), the Waymo Open Dataset (Sun et al., 2020), and GOOSE (Mortimer et al., 2024) retain the sparse and imperfect projected measurements encountered in newly collected data. Together, they test transfer across sensor configurations, scene types, and target quality.

Monocular depth foundation models. Modern monocular models, such as Depth Anything V2 (Yang et al., 2024), Depth Anything V3 (Lin et al., 2025a) Metric3D v2 (Hu et al., 2024), UniDepth (Piccinelli et al., 2024), Depth Pro (Bochkovskii et al., 2025), and MoGe-2 (Wang et al., 2025a), provide detailed dense geometry. However, monocular models are still biased metrically, particularly under domain shift and at long range. Thus, FounRef uses this geometry as a frozen prior and grounds it with direct metric LiDAR anchors.

Learned depth completion. Depth-completion methods predict dense metric depth from an RGB image and sparse depth observations (Ma and Karaman, 2018). They learn image-guided depth propagation, for example through nonlocal afinities in NLSPN (Park et al., 2020), transformer context in CompletionFormer (Zhang et al., 2023), or foundation-model distillation in DMD<sup>3</sup>C (Liang et al., 2025). Other foundation-model conditioning methods include PromptDA (Lin et al., 2025b) and Prior Depth Anything (Wang et al., 2025b), while BP-Net (Tang et al., 2024) is a propagation-based completion baseline. These learned, scene-dependent priors require ofline training and depend on the training distribution, sensor configuration, and input sparsity. On the other hand, OMNI-DC (Zuo et al., 2025) targets zero-shot transfer across datasets and sparse-depth patterns. In contrast, FounRef is training-free and adapts a frozen monocular prior to each image using only sparse metric anchors.

Optimization-based depth refinement. Explicit refinement methods propagate sparse depth constraints through a regularized correction field. TGV<sup>2</sup> (Bredies et al., 2010) regularizes both the correction gradient and its spatial change, favoring locally afine corrections, while screened Poisson refinement (Kazhdan and Hoppe, 2013) balances anchor fitting with spatial smoothness. The fast bilateral solver (Barron and Poole, 2016) instead solves an edge-aware correction on a low-dimensional bilateral grid, allowing eficient propagation within similar regions while limiting propagation across boundaries.

Anchor outlier handling. Projected LiDAR anchors can be corrupted by occlusion, calibration error, moving objects, temporal registration, and projection artifacts. KITTI provides a refined dense target for ofline evaluation, obtained by multimodal careful voting, but such a target is unavailable when processing data from a new sensor rig. RePLAy (Zhu et al., 2024) uses a parameter-free epipolar-geometry test to remove baseline-induced projective occlusions. FounRef is complementary: it tests each anchor against a locally refined dense reference surface, allowing it to reject disagreements caused by a broader range of cross-modal and temporal inconsistencies.

## 3 Method

FounRefconverts a frozen monocular depth prior into a dense metric depth map using sparse metric anchors. Given an input RGB image I, a frozen foundation model predicts a dense prior depth map $D _ { \mathrm { p r i o r } }$ and, when available, a validity mask M. The refinement stage receives the prior and a set of projected metric anchors $\mathcal { A } = \{ ( \mathbf { u } _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { u } _ { i }$ is an image location and $z _ { i }$ is its measured metric depth. The output is a dense metric depth map D. Pixels marked invalid by the prior are excluded from fitting and assigned the maximum depth cap $D _ { \mathrm { m a x } }$ in the final output. The algorithm is defined in Algorithm 1.

Global monotone calibration. The prior predicts depth on the wrong scale. We correct this in log space, where a scale factor becomes a shift. Writing $t _ { i } = \log D _ { \mathrm { p r i o r } } ( \mathbf { u } _ { i } )$ and $y _ { i } = \log z _ { i }$ for each fitting anchor, we fit a straight line

![](images/43120873a6d16f02b4312f3d4dd9092ec6a167ce58d1ce9a2599c180cfe78bb9.jpg)  
Figure 2: Global calibration of four priors on the same frame. Top: raw predictions use incompatible depth scales; DA-V2 is relative. Bottom: after fitting the same sparse anchors, all priors share a metric scale while preserving their edges and relative layout. All depth panels use the 0–80m range. Pixels a prior marks invalid (sky) are drawn at the far end (MoGe-2 provides an invalid map, while depth pro predicts them at 10, 000m).

## Algorithm 1: FounRef pipeline

1. Predict the frozen monocular prior and its validity mask: $( D _ { \mathrm { p r i o r } } , M ) = \mathrm { P r i o r } ( I )$

2. Cap the anchors at $D _ { \mathrm { m a x } } ~ = ~ 5 0 m$ and split them into fixed fitting and held-out sets: $( \mathcal { A } _ { \mathrm { f i t } } , \mathcal { A } _ { \mathrm { t e s t } } ) \ =$ $\mathrm { S p l i t } ( \mathrm { C a p } ( \mathcal { A } , D _ { \mathrm { m a x } } ) , 8 0 / 2 0 )$

3. Globally calibrate the prior using the fitting anchors: $D _ { \mathrm { c a l } } = \mathrm { C a l i b r a t e } ( D _ { \mathrm { p r i o r } } , A _ { \mathrm { f i t } } )$

4. If validation is enabled, construct the locally corrected reference surface: $D _ { 1 } = \mathrm { L i g h t S o l v e } ( D _ { \mathrm { c a l } } , A _ { \mathrm { f i t } } )$

5. Filter fitting and held-out anchors against this reference: A<sup>+</sup> = {(u<sub>i</sub>, z<sub>i</sub>) ∈ A : | log z<sub>i</sub> − log D<sub>1</sub>(u<sub>i</sub>)| ≤ τ}.

6. Run the full solver using only retained fitting anchors: D<sub>2</sub> = FullSolve(D<sub>cal</sub>, A<sup>+</sup> ).

$$
\begin{array} { l } { f ( t ) = \displaystyle \alpha t + \beta \colon } \\ { \displaystyle ( \hat { \alpha } , \hat { \beta } ) = \arg \operatorname* { m i n } _ { \alpha , \beta } \sum _ { i } \rho _ { \mathrm { H u b e r } } ( y _ { i } - f ( t _ { i } ) ) } \end{array}\tag{1}
$$

where $\rho _ { \mathrm { H u b e r } }$ is the Huber loss, solved with iteratively reweighted least squares (Huber, 1964).

One line cannot fit every distance equally well. We therefore split the anchors into K bins by depth and record the average error the line makes in each bin. Interpolating these K values gives a small per-distance correction c, and the calibrated prior is

$$
\log D _ { \mathrm { c a l } } ( \mathbf { x } ) = f ( \log D _ { \mathrm { p r i o r } } ( \mathbf { x } ) ) + c ( \log D _ { \mathrm { p r i o r } } ( \mathbf { x } ) )\tag{2}
$$

We use $K = 2 4$ , chosen once on validation as the smallest value that saturated calibration accuracy; the mapping is monotone, sparse bins are damped, and extrapolation is disabled. With too few anchors we use $f$ alone; with none, we keep the prior.

Anchor validation. To filter cross-modal noise, we validate anchors before the final refinement. Validation requires a metric surface that is locally more accurate than global calibration alone. We obtain this surface by running a lightweight local solver on the fitting anchors from the current sweep only, producing $D _ { 1 }$ . For an anchor i projected at $\mathbf { u } _ { i }$ , we retain the anchor when

$$
| \log z _ { i } - \log D _ { 1 } ( \mathbf { u } _ { i } ) | \leq \tau ,\tag{3}
$$

and discard it otherwise, and where $\tau$ is a dataset-specific threshold in log space. Because the test operates in log depth, it measures relative disagreement rather than an absolute error that increases with range. We select that threshold $\tau$ from a validation operating curve. Cleaner datasets, such as KITTI (Geiger et al., 2013), permit a more relaxed threshold, whereas noisier datasets, such as nuScenes (Caesar et al., 2020) and GOOSE (Mortimer et al., 2024), require stricter filtering.

The aim is to discard only a small fraction of unreliable anchors while preserving a dense metric point cloud. This is particularly important under temporal accumulation: additional sweeps improve coverage only when their outliers are removed. Since FounRef is training-free, it cannot learn to suppress these errors from data; explicit anchor validation is therefore necessary.

Structure-preserving local solver. We refine the calibrated prior with a single additive correction field in log depth:

$$
\log D ( \mathbf { x } ) = \log D _ { \mathrm { c a l } } ( \mathbf { x } ) + b ( \mathbf { x } )\tag{4}
$$

where b is a smooth correction field. Thus, the solver adjusts the prior’s surfaces rather than redrawing their structure. For

anchor $i ,$ the desired correction is

$$
t _ { i } = \log z _ { i } - \log D _ { \mathrm { c a l } } ( \mathbf { u } _ { i } ) .\tag{5}
$$

We estimate b by minimizing

$$
E ( b ) = \sum _ { i } w _ { i } \big ( b ( { \mathbf { u } } _ { i } ) - t _ { i } \big ) ^ { 2 } + \lambda b ^ { \top } L b ,\tag{6}
$$

Here, $b ( u _ { i } )$ is the correction interpolated from the bilateral grid at anchor $u _ { i } , w _ { i }$ is its confidence, and λ sets the strength of the bilateral smoothness prior. The bilateral Laplacian (Barron and Poole, 2016) penalizes diferences between corrections at vertices that are nearby in image position and similar in prior depth. Thus, anchors share a correction along a predicted surface, while predicted depth discontinuities limit propagation across surfaces.

We construct L on a bilateral grid over image position and log prior depth, and five bistochastic normalization steps (Barron and Poole, 2016). Two pixels are coupled only when they are close in the image and lie at similar prior depth. A correction can therefore spread smoothly along a surface while stopping at its depth boundaries: for example, an anchor on a nearby pole corrects the pole without pulling the background towards it.

The resulting sparse positive-definite system is solved on grid vertices with Jacobi-preconditioned conjugate gradients. We solve at half resolution, aggregating anchors in each $2 \times 2$ block as a weighted mean in log-depth so that returns from diferent surfaces never average into a depth that belongs to neither. Only the correction is upsampled and applied to the full-resolution prior, which keeps the prior’s detail intact. The weights $w _ { i }$ combine agreement with a robust local consensus and a depth-dependent confidence term, so the solver trusts the prior near the camera and the anchors where the prior starts to drift.

We use a pure shift field rather than a local scale-and-shift field. We also exclude image intensity from the bilateral grid: although it can improve metric accuracy, it can imprint image texture into the recovered geometry.

## 4 Experiments

We run all experiments on a single NVIDIA A100 GPU using PyTorch 2.10 and CUDA 12.8. Each competing network remains resident in a separate process throughout evaluation on a dataset, running in fp16 precision when possible.

Data. We use the five datasets described above. Our deployment setting assumes an RGB image, camera intrinsics when required by the monocular prior, and sparse direct metric-depth anchors, optionally augmented with temporally accumulated anchors when context is available.

Since monocular priors become substantially less reliable at longer ranges (Table 3), we cap input anchors at 50 m, where monocular priors remain suficiently reliable for metric alignment. For full context, we additionally report farrange error beyond this cap. This range retains the challenging medium while avoiding a comparison dominated by unreliable far-range estimates.

When refined dense ground truth is available, namely on KITTI and NYU Depth V2, we report dense-depth metrics for direct comparison with prior work. In addition, we report error on a filtered held-out subset of the raw metric measurements. For every image, we split these measurements deterministically, using a fixed seed of 42, into 80% fitting anchors and 20% held-out anchors. This sensor-grounded metric does not replace oficial benchmark metrics; rather, it measures agreement with independent observations that were not used during refinement. It is the primary evaluation protocol for nuScenes, Waymo, and GOOSE, where refined dense ground truth is unavailable. For these datasets, we construct denser metric targets by accumulating measurements from neighboring frames, up to approximately ten frames. Accumulation substantially improves coverage, but also introduces artifacts from moving objects, ego-motion error, occlusions, and sensor noise. Therefore,we filter the accumulated points using the structural consistency of an independent Metric3D-V2 prior, which exhibits strong metric and structural performance in our prior-selection experiment (Table 3, Figure 2). This avoids biasing the evaluation target toward FounRef, which uses MoGe-2 as its refinement prior, while applying the same surface-consistency criterion used to identify outlier anchors. Points that disagree with the estimated visible scene geometry are removed before evaluation, producing cleaner and more spatially coherent targets than unfiltered temporal accumulation.

Finally, we evaluate 500 images from each dataset’s validation set, providing a consistent sample size across domains while keeping all comparisons computationally tractable.

Prior selection. The monocular foundation model is the backbone of FounRef: it provides the initial depth estimate, fine geometric structure, and a substantial fraction of the total runtime. We therefore compare several frozen foundation models (Table 3, Figure 2). None is fine-tuned or retrained.

All priors are evaluated on the same KITTI images with identical anchors, global calibration, and local refinement.

We report three axes. Accuracy on the dense target: RMSE, AbsRel, and RMSE beyond 50 m, where priors difer most. Structure, as a pair of spectral measures: incoherence, the high-frequency energy on flat surfaces, where any such content is noise, and detail, the same quantity on finely textured regions, where it is real scene structure. They have to be read together: a washed-out prediction scores well on incoherence alone, so only a prior that is low on the first and high on the second is both clean and sharp. And runtime, measured per frame.

![](images/acb4f7cabcc66886fde8c8bab2bfde2959b9bca25ea5b334a36e4fe4a225d5fa.jpg)  
RGB

![](images/13fb4f23f331bce404aecc01f06e1722a4b03bb151ab5c7a48b8ba8eedd23f98.jpg)  
raw anchors

![](images/881ff5f6e91e865995b8a6480f14323785bfcf6650bae6347face3d6db24f4dd.jpg)  
prior D<sub>1</sub> (ref.)

![](images/bf1ef9d89f40519e7c57557be3ea01f02ab2ec4144f08128381a25b26c7552bb.jpg)  
RePLAy mask

![](images/23761ede35c491262c1c55b653888064ea4cbfe2f9c096eb0d3ed0208d65e56b.jpg)  
RePLAy filtered

![](images/a13494c71698e4fab9b2d0b145e9113b354dfcf5716ddc1edb6fa1ba17b1aa32.jpg)  
ours mask

![](images/05fd7bd7ab2a966d2a3e6f11e60afca401a51f04c055bb68039ffb6004ad0fba.jpg)  
ours filtered

Figure 3: Anchor filtering on a cluttered KITTI crop. RePLAy retains returns through windshields, mirrors, and moving-vehicle silhouettes, whereas FounRef rejects them because they disagree with the locally refined smooth surface $D _ { 1 }$  
![](images/14beea6884d65c084b24edfbdaeb4bf2da13cb0a968228e01d31b7eb4d69acbd.jpg)  
RGB

![](images/92431c570a375369f7680798aa35d7e3c7b2cb704abacc17677c592c802e9041.jpg)  
calibrated prior

![](images/b8f5e456a3901c1c070fdbb270dca1ccd1e51648471a8de7f26b3e75bd783a57.jpg)  
ours (bilateral)

![](images/7159006a97d2361a6f7e6e4fdf5303b16dd177542372437198d10fd8aa4bebb8.jpg)  
Poisson

![](images/3f5bf33d5f4f272e55ae3e81b51a657dfea65d8a3540ddd838b62f7554875aba.jpg)  
TGV

Figure 4: Solver comparison on a sparse-anchor nuScenes crop. All methods start from the same calibrated prior and anchors. Pixel-wise Poisson and TGV produce anchor-centred surface bumps, whereas FounRef shares corrections on a coarse bilateral grid. Depth panels share the [1, 50] m range.  
![](images/83157699c8dca6d3161f10abc61032e069c2eb5ec638e8e22401bc7d4b58f115.jpg)  
Figure 5: One solver, four operating points on nuScenes (0–50 m). $\sigma _ { s }$ is the bilateral grid’s spatial bandwidth in pixels; reducing it localizes corrections, improving RMSE but increasing normal dispersion.

Solver selection. The local solver determines how sparse metric corrections propagate across the depth map. We compare the fast bilateral solver, screened edge-aware Poisson refinement and $\mathrm { T G V ^ { 2 } } { \mathrm { - } } \mathrm { L } 2$ refinement. To isolate the solver, the prior, its global calibration, and the per-anchor confidence weights are computed once and handed byte-identically to every method; only the regularizer difers, and all four solve the same unknown, the log-residual $r = \log ( D / D _ { \mathrm { c a l } } )$ . All methods run on the same 100 KITTI frames (seed 42), scored against KITTI’s accumulated dense depth maps (Table 2; qualitative results and the bandwidth trade-of appear in Figures 4 and 5.

Anchor filtering. Sparse anchors are not uniformly reliable. Even a single sweep can contain outliers caused by sensor misalignment, moving objects, or occlusions. These errors become more frequent when anchors are accumulated over time. We therefore test every anchor against a reference surface before the final refinement.

The reference surface is produced from the monocular prior in two stages. We first apply global calibration and then refine the calibrated prior with the lightweight local solver, producing the reference depth map $D _ { 1 }$ . An anchor is removed when its log-depth difers from $D _ { 1 }$ by more than τ. We use $\tau = 0 . 4 5$ on KITTI; for the noisier nuScenes data, we use the stricter threshold $\tau = 0 . 2$ , selected from a validation operating curve (See Appendix).

We compare against RePLAy as a representative geometrybased point-cloud filter. Because RePLAy targets projective occlusions using LiDAR geometry, it does not address the broader cross-sensor and temporal inconsistencies captured by our surface-consistency test.

We evaluate the filter on KITTI. First, we report the fraction of capped anchors removed by the filter. Next, using KITTI ground truth, we compare the relative depth error of retained and removed anchors across depth ranges. This reveals whether the filter removes primarily incorrect anchors rather than simply discarding many measurements. Finally, we run the full refinement with and without filtering and report the resulting depth accuracy and runtime. We repeat this evaluation for single-sweep and temporally accumulated anchors because the two settings contain diferent types and levels of noise (Table 4, Figure 3).

Depth-completion benchmark. We compare FounRef with supervised sparse-to-dense and zero-shot depthcompletion methods. Each baseline is evaluated using its publicly released checkpoint, without dataset-specific recalibration or fine-tuning, and receives the same sparse input as FounRef.

Table 1: Depth completion across five validation sets (n = 500) using identical anchors for every method and no dataset-specific recalibration. KITTI and NYU use dense ground truth; the other datasets use a held-out 20% of the cloud. RMSE and far (RMSE over target depths $3 0 < z \le 5 0$ m) are in metres, surf. is normal dispersion (<sup>◦</sup>), and ms is mean latency on the Waymo validation set, our worst-case setting, on an A100 (most models cannot run at fp16 precision)
<table><tr><td rowspan="2"></td><td colspan="3">KITTI</td><td colspan="3">nuScenes</td><td colspan="3">Waymo</td><td colspan="3">GOOSE</td><td colspan="3">NYU</td><td></td></tr><tr><td>RMSE↓ far↓</td><td></td><td>surf.↓</td><td>RMSE↓</td><td>far↓</td><td>surf.↓</td><td>RMSE↓ far↓</td><td></td><td>surf.↓</td><td>RMSE↓</td><td>far↓</td><td>surf.↓</td><td>RMSE↓</td><td>far↓</td><td>surf.↓</td><td>ms↓</td></tr><tr><td>Ours Ours  $( \sigma _ { s } 4 )$ </td><td>0.726 0.663</td><td>2.08 1.83</td><td>10.8 11.2</td><td>0.687 0.489</td><td>2.00 1.27</td><td>3.5 3.6</td><td>0.745 0.587</td><td>1.43 1.10</td><td>1.7 2.2</td><td>1.687 1.073</td><td>4.42 3.24</td><td>8.8 8.9</td><td>0.174 0.164</td><td>0.40 0.37</td><td>6.6 6.7</td><td>78 78</td></tr><tr><td>OMNI-DC</td><td>1.137</td><td>3.61</td><td>45.8</td><td>0.353</td><td>0.87</td><td>34.6</td><td>0.590</td><td>1.12</td><td>41.7</td><td>0.702</td><td>2.01</td><td>33.1</td><td>0.099</td><td>0.20</td><td>10.2</td><td>1011</td></tr><tr><td>Prior-DA</td><td>1.583</td><td>4.83</td><td>41.4</td><td>0.661</td><td>2.19</td><td>16.6</td><td>0.833</td><td>1.69</td><td>13.2</td><td>0.742</td><td>2.03</td><td>15.8</td><td>0.135</td><td>0.25</td><td>13.0</td><td>1621</td></tr><tr><td>PromptDA</td><td>2.237</td><td>7.32</td><td>11.9</td><td>2.058</td><td>7.41</td><td>8.8</td><td>2.770</td><td>6.48</td><td>5.8</td><td>1.545</td><td>4.91</td><td>11.0</td><td>0.218</td><td>0.40</td><td>11.1</td><td>580</td></tr><tr><td>BP-Net</td><td>0.747</td><td>2.11</td><td>45.1</td><td>0.602</td><td>1.66</td><td>35.0</td><td>0.932</td><td>1.53</td><td>49.2</td><td>0.693</td><td>1.85</td><td>30.6</td><td>1.055</td><td>0.79</td><td>23.3</td><td>1081</td></tr><tr><td>NLSPN</td><td>0.951</td><td>2.47</td><td>38.5</td><td>0.834</td><td>2.11</td><td>24.3</td><td>0.756</td><td>1.39</td><td>27.3</td><td>0.764</td><td>1.82</td><td>27.6</td><td>2.720</td><td>0.76</td><td>21.8</td><td>347</td></tr><tr><td>DMD3C</td><td>0.677</td><td>1.92</td><td>40.1</td><td>0.523</td><td>1.53</td><td>18.7</td><td>0.774</td><td>1.56</td><td>25.6</td><td>0.805</td><td>1.92</td><td>23.4</td><td>1.205</td><td>2.10</td><td>20.6</td><td>1219</td></tr></table>

Table 2: Solver comparison on KITTI val against accumulated dense depth. All solvers receive identical prior, calibration, anchors, and weights; only the regularizer difers. RMSE: error over all valid pixels;far: RMSE in the 30–50 m band. surf.: median angle between the output normals and calibrated prior normals in smooth regions. imp.: correction at anchor pixels relative to a few pixels away, so 1.0 leaves no trace of the anchor pattern.
<table><tr><td>solver</td><td>RMSE↓</td><td>far↓</td><td></td><td>surf. ↓ imp. →1 ms ↓</td><td></td></tr><tr><td>Ours</td><td>1.050</td><td>2.583</td><td> $1 . 7 7 ^ { \circ }$ </td><td>0.38</td><td>14</td></tr><tr><td>Screened Poisson</td><td>0.880</td><td>1.904</td><td> $4 . 7 5 ^ { \circ }$ </td><td>5.25</td><td>42</td></tr><tr><td>TGV²-L2</td><td>0.934</td><td>1.949</td><td>6.32°</td><td>一</td><td>300</td></tr></table>

The benchmark spans five datasets and two complementary settings. KITTI and NYU Depth V2 provide standard benchmarks with refined dense ground truth, including domains on which some baselines were trained. nuScenes, Waymo, and GOOSE assess transfer to previously unseen driving and unstructured outdoor data, where raw sensor measurements are sparse and imperfect. Together, these settings test whether FounRef generalizes across sensor configurations, scene types, and input distributions rather than only within the training domain of a learned completion model.

Importantly, we evaluate both metric accuracy and surface structure. This distinction reveals not only where methods transfer successfully across distributions, but also whether a gain in point-wise accuracy is achieved by overfitting sparse measurements and damaging the depth surface between them. A solver optimized solely to fit anchors could obtain lower point-wise error while introducing local artifacts; our evaluation therefore measures whether metric correction is achieved without sacrificing coherent scene geometry.

Table 3: Prior selection (KITTI val., n=100, A100, fp16). All rows: FounRef refined. Scored to 80 m so each prior’s bias beyond the 50 m trust cap is visible; anchors stop at 50 m, so the >50 m column measures unsupported extrapolation. Structure is measured in log-depth on LiDAR-derived masks: incoh = high-frequency spectral fraction on flat surfaces, detail = the same on RGB-textured regions. Read them together: low+low = washed, high detail + low incoh = clean and sharp.
<table><tr><td></td><td colspan="3">accuracy</td><td colspan="2">structure</td><td></td></tr><tr><td>prior</td><td>RMSE (↓)</td><td>AbsRel (↓)</td><td>RMSE&gt;50m (↓) incoh (↓) detail (↑) ms (↓)</td><td></td><td></td><td></td></tr><tr><td>MoGe-2</td><td>2.83</td><td>0.029</td><td>14.57</td><td>0.007</td><td>0.025</td><td>49</td></tr><tr><td>Metric3D-V2</td><td>2.76</td><td>0.028</td><td>14.41</td><td>0.008</td><td>0.004</td><td>77</td></tr><tr><td>DA-V3</td><td>2.98</td><td>0.034</td><td>15.00</td><td>0.020</td><td>0.020</td><td>107</td></tr><tr><td>DA-V2</td><td>3.05</td><td>0.037</td><td>15.44</td><td>0.050</td><td>0.112</td><td>55</td></tr><tr><td>UniDepth-V2</td><td>3.59</td><td>0.047</td><td>17.28</td><td>0.020</td><td>0.007</td><td>49</td></tr><tr><td>DepthPro</td><td>2.83</td><td>0.032</td><td>14.38</td><td>0.213</td><td>0.035</td><td>205</td></tr></table>

## 5 Discussion and Limitations

Our experiments show that dense metric depth should not be evaluated by point-wise error alone. Across prior selection, solver selection, and depth completion, lower RMSE can coincide with noisier surfaces, stronger anchor imprinting, or substantially higher runtime. These are distinct properties: metric accuracy determines whether the prediction is correctly grounded, structural fidelity determines whether the geometry remains coherent between sparse measurements, and runtime determines whether the method is practical for large-scale processing or online use. FounRef addresses this three-way objective by combining a transferable monocular prior, direct metric anchors, and lightweight structurepreserving refinement.

Table 3 identifies MoGe-2 as the most suitable prior for this balance. Metric3D-V2 achieves slightly lower refined metric error but preserves considerably less fine detail, whereas Depth Pro and Depth Anything introduce more high-frequency surface noise. MoGe-2 therefore provides the strongest overall combination of metric compatibility, coherent geometry, retained detail, and runtime. The fact that every tested prior improves beyond global calibration also indicates that FounRef is not specific to MoGe-2 and can directly benefit from future monocular foundation models.

![](images/867f94ad18c23f919e26179e8e075c6431b4088ed78d97a15f5c374dfbefb421.jpg)  
Figure 6: Qualitative bake-of across five datasets. Metric accuracy doesn’t stand on its own, without appropriate runtime and coherent structure.

Table 4: Anchor-level filter comparison on KITTI validation $( n { = } 1 0 0 )$ ). Errors are relative errors at dropped/kept anchors against dense GT or the independent reference $D _ { 1 }$ . Higher drop/keep ratios indicate better separation. Runtime is the filter’s own cost on one A100 (fp16). Temporal is over 10 sweeps.The oracle measures the diference between the raw and filtered point clouds, as processed by the Authors of KITTI.
<table><tr><td>set</td><td>filter</td><td>drop (%)</td><td>vs. GT err. (D/K) X</td><td>VS.  $D _ { 1 }$ </td><td>ms</td></tr><tr><td>single</td><td>Ours RePLAy</td><td>2.3 4.5</td><td>1.08 / 0.04 271.10 / 0.03 37 0.49 / 0.04 12 0.43 / 0.04 11 452</td><td>err. (D/K) X</td><td>13</td></tr><tr><td>temporal Ours</td><td>Oracle RePLAy Oracle</td><td>2.5 2.2 2.2 1.5</td><td>0.80 / 0.04 20 0.88 / 0.03 29 0.26 / 0.0470.27 / 0.0568 0.87 / 0.03 290.59 / 0.0512</td><td>0.95 / 0.0332 0.64 / 0.0416</td><td>13 868</td></tr></table>

The filtering and solver experiments clarify the remaining sources of error. FounRef removes only a small fraction of anchors, but the rejected measurements are substantially less reliable than those retained. Unlike geometry-only filtering, the surface-consistency test can reject cross-modal and temporal disagreements such as moving-object silhouettes, calibration errors, and occlusion artifacts. The solver comparison reveals a related trade-of: Poisson and TGV fit the anchors more aggressively and obtain lower metric error, but produce local bumps and visible anchor imprinting. The bilateral solver instead propagates corrections along surfaces defined by the prior and limits propagation across depth discontinuities, accepting a modest metric penalty in exchange for cleaner geometry and lower runtime.

The cross-domain benchmark exposes the limitations of learned depth completion and provides a diagnostic crosssection of current approaches. DMD3C is highly competitive on KITTI, but this advantage does not transfer consistently to nuScenes, Waymo, GOOSE, and NYU Depth V2. More generally, no learned baseline remains uniformly accurate across all five datasets. Supervised methods are sensitive to their training domains, sensor configurations, and sparsedepth patterns, while recent zero-shot and foundation-guided methods improve transfer at the cost of substantial runtime, surface degradation, or inconsistent metric accuracy. Consequently, none of the evaluated methods is an unambiguous out-of-the-box choice when accuracy, geometric coherence, cross-domain robustness, and eficiency are considered jointly.

FounRef behaves diferently because it preserves the geometry of a frozen monocular prior and estimates only the metric correction supported by the sparse anchors. This separation allows it to inherit the prior’s cross-domain structural generalization without learning a fixed RGB-sparse-depth mapping. As a result, FounRef achieves the lowest normal dispersion across all evaluated datasets while remaining metrically competitive and considerably faster than the learned alternatives. The more aggressive $\sigma _ { s } { = } 4$ configuration further improves metric accuracy without increasing runtime, but moves toward the same accuracy-structure trade-of observed in the solver study.

FounRef nevertheless remains limited by the quality of its prior. It can correct the metric placement of surfaces already represented by the monocular model, but cannot reliably recover geometry that the prior omits or interprets incorrectly. GOOSE illustrates this limitation: LiDAR may return through vegetation or fences, while the image prior represents the visible foreground as a coherent surface. FounRef tends to preserve that visible surface rather than create a sparse see-through reconstruction, which can increase pointwise error. Structure preservation is therefore an inductive bias rather than an unconditional advantage. Because Foun-Ref is modular and directly inherits the geometry of its prior, stronger future priors could further improve both accuracy and structure, making the same lightweight framework an even stronger competitor to state-of-the-art depth-completion methods.

Overall, FounRef lies on an empirical Pareto frontier between metric accuracy, surface coherence, cross-domain robustness, and runtime. More aggressive local fitting can further reduce error, but increasingly exposes the sparse-anchor pattern and degrades surface quality, while still retaining substantially lower normal dispersion than the competing methods. A natural next step is a zero-shot refinement model that moves this frontier without sacrificing the structural generalization of the frozen prior or the eficiency of the current solver. Methods such as OMNI-DC and Prior-DA already move toward broader transfer and foundation-prior integration, but the benchmark shows that substantial gaps remain in surface coherence and runtime.

## 6 Conclusion

We introduced FounRef, a training-free method that converts a frozen monocular prior and sparse metric anchors into dense metric depth. Global calibration, surface-based anchor validation, and bilateral refinement provide metric grounding while preserving the prior’s geometry. Across five datasets, FounRef remains metrically competitive with learned methods while producing more coherent surfaces at substantially lower runtime. Its modular design can directly benefit from stronger future priors and metric sensors without retraining.

Use of Generative AI. Generative AI was used for language editing. The authors verified all technical content, analyses, citations, and claims, and take full responsibility for the manuscript.

## References

Jonathan T. Barron and Ben Poole. The Fast Bilateral Solver. In Proceedings ofthe European Conference on Computer Vision, pages 617–632, 2016.

Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. In Proceedings of the International Conference on Learning Representations, 2025.

Kristian Bredies, Karl Kunisch, and Thomas Pock. Total Generalized Variation. SIAM Journal on Imaging Sciences, 3(3):492–526, 2010.

Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020.

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision Meets Robotics: The KITTI Dataset. The International Journal ofRobotics Research, 32(11):1231– 1237, 2013.

Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares Ambrus, and Adrien Gaidon. Towards Zero-Shot Scale-Aware Monocular Depth Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9233–9243, 2023.

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation. arXiv preprint arXiv:2404.15506, 2024.

Peter J. Huber. Robust Estimation of a Location Parameter. The Annals ofMathematical Statistics, 35(1):73–101, 1964.

Michael Kazhdan and Hugues Hoppe. Screened Poisson Surface Reconstruction. ACM Transactions on Graphics, 32(3):29:1–29:13, 2013.

Yingping Liang, Yutao Hu, Wenqi Shao, and Ying Fu. Distilling Monocular Foundation Model for Fine-grained Depth Completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the Visual Space from Any Views. arXiv preprint arXiv:2511.10647, 2025a.

Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, and Bingyi Kang. Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Fangchang Ma and Sertac Karaman. Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 4796–4803, 2018.

Peter Mortimer, Raphael Hagmanns, Miguel Granero, Thorsten Luettel, Janko Petereit, and Hans-Joachim Wuensche. The GOOSE Dataset for Perception in Unstructured Environments. In Proceedings of the IEEE International Conference on Robotics and Automation, 2024.

Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu, and In So Kweon. Non-Local Spatial Propagation Network for Depth Completion. In Proceedings of the European Conference on Computer Vision, 2020.

Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal Monocular Metric Depth Estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor Segmentation and Support Inference from RGBD Images. In Proceedings of the European Conference on Computer Vision, pages 746–760, 2012.

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Sheng Zhao, Shuyang Cheng, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2446–2454, 2020.

Jie Tang, Fei-Peng Tian, Boshi An, Jian Li, and Ping Tan. Bilateral Propagation Network for Depth Completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9763–9772, 2024.

Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and Andreas Geiger. Sparsity Invariant

CNNs. In Proceedings of the International Conference on 3D Vision, pages 11–20, 2017.

Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details. arXiv preprint arXiv:2507.02546, 2025a.

Zehan Wang, Siyu Chen, Lihe Yang, Jialei Wang, Ziang Zhang, Hengshuang Zhao, and Zhou Zhao. Depth Anything with Any Prior. arXiv preprint arXiv:2505.10565, 2025b.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In Advances in Neural Information Processing Systems, 2024.

Youmin Zhang, Xianda Guo, Matteo Poggi, Zheng Zhu, Guan Huang, and Stefano Mattoccia. CompletionFormer: Depth Completion with Convolutions and Vision Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18527– 18536, 2023.

Shengjie Zhu, Girish Chandar Ganesan, Abhinav Kumar, and Xiaoming Liu. RePLAy: Remove Projective LiDAR Depthmap Artifacts via Exploiting Epipolar Geometry. arXiv preprint arXiv:2407.19154, 2024.

Yiming Zuo, Willow Yang, Zeyu Ma, and Jia Deng. OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

## A Appendix

This appendix ablates the pipeline stage by stage, reports how the method behaves as the metric evidence is thinned and corrupted, examines the two solver settings that admit a continuous choice, and compares every method qualitatively across the five datasets.

Every number reported here uses the default configuration; nothing was retuned for the appendix. These experiments use their own frame subset and reference for screening the evaluation target, so absolute values are not directly comparable with the main tables — each table is internally consistent, and the comparisons drawn are within a table.

Pipeline break-down. Table 5 adds one stage of the pipeline at a time on KITTI, Waymo and GOOSE, using the cloud before any cleaning so the noise the anchor filter exists for is present. Global calibration fixes the metric scale of the monocular prior; the first solve $D _ { 1 }$ produces the surface the filter tests each anchor against; the second solve $D _ { 2 }$ runs on the anchors that survive it.

![](images/c39904b64b035d32078edea36bba9b6fd9f23738a702791781ef10ccd0bd55e6.jpg)  
Figure 7: The pipeline on one nuScenes frame, by row: left the state after a step, right what that step changed (blue nearer, red farther, grey unchanged, each panel on its own scale). Top: anchors in, and what survives the anchor filter (4.7% removed), dilated for legibility.

Each stage pays for itself. The first solve $D _ { 1 }$ removes half to two thirds of the calibrated prior’s error, and $D _ { 2 }$ improves on $D _ { 1 }$ by 19% on KITTI and 14% on Waymo — that margin is what filtering the cloud buys, since $D _ { 1 }$ and $D _ { 2 }$ are the same solver difering in the anchors they receive. GOOSE is the exception at 2.7% the other way: of-road returns through vegetation disagree with the image-visible surface for reasons the threshold cannot separate from noise.

Figure 7 shows the whole progression on one frame: each stage in turn, the anchor cloud the filter accepts and rejects, and the change every step applies. It is the visual counterpart of the dispersion column — what the stages move is the metric field, not the geometry.

<table><tr><td>stage</td><td>kitti</td><td>waymo</td><td>goose</td></tr><tr><td>RMSE (m)</td><td></td><td></td><td></td></tr><tr><td>calibrated prior</td><td>1.407</td><td>2.522</td><td>3.268</td></tr><tr><td> $+ D _ { 1 }$  (filter reference)</td><td>0.698</td><td>0.959</td><td>1.101</td></tr><tr><td>+ D2 (full method)</td><td>0.567</td><td>0.826</td><td>1.131</td></tr><tr><td>RMSE 30–50 m</td><td></td><td></td><td></td></tr><tr><td>calibrated prior</td><td>4.220</td><td>4.423</td><td>8.713</td></tr><tr><td>十  $D _ { 1 }$  (filter reference)</td><td>2.021</td><td>1.836</td><td>2.916</td></tr><tr><td> $+ D _ { 2 }$  (full method)</td><td>1.636</td><td>1.581</td><td>3.032</td></tr><tr><td>normal disp. (deg)</td><td></td><td></td><td></td></tr><tr><td>calibrated prior</td><td>10.64</td><td>3.15</td><td>8.94</td></tr><tr><td>十  $D _ { 1 }$  (filter reference)</td><td>11.65</td><td>5.40</td><td>9.37</td></tr><tr><td>十  $D _ { 2 }$  (full method)</td><td>11.78</td><td>4.83</td><td>9.54</td></tr></table>

Table 5: Pipeline ablation on the accumulated, unfiltered cloud. Anchors and held-out target are screened by the method’s own MoGe reference; no competitor appears here, so these are inference numbers rather than the cross-method protocol of the main tables.

Behaviour as anchors are thinned and mis-projected. How many anchors a frame carries is a property of the rig, not of the method: a 32-beam scanner returns roughly half what a 64-beam one does, returns thin out with range and on dark or specular surfaces, and accumulation is available only when ego-motion is. Figure 8 therefore sweeps the retained fraction from 100% to $1 \% ,$ with every method receiving the identical thinned cloud on each frame and scored against the unchanged target.

![](images/acff9d116e776dca30c39d8b07c818b73a0f107f0cd7cfe2628659568d15e2fd.jpg)  
Figure 8: Anchor-density sweep.

Table 6 gives the numbers behind the curves. The methods separate by how they degrade, not by where they start. From full density to $1 \%$ our error grows by 1.4–2.1× on every dataset, the most uniform response in the table, because the prior supplies the geometry and the anchors only fix its metric scale — removing anchors weakens the scale estimate without touching the surface. DMD3C, which regresses depth from the points themselves, grows by 2.9–12.9× and is unusable at the sparse end.

<table><tr><td>method</td><td>100%</td><td>50%</td><td>25%</td><td>10%</td><td>5%</td><td>1%</td></tr><tr><td>kitti</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours</td><td>0.832</td><td>0.865</td><td>0.906</td><td>0.969</td><td>1.024</td><td>1.150</td></tr><tr><td>OMNI-DC</td><td>1.095</td><td>0.887</td><td>0.663</td><td>0.708</td><td>0.796</td><td>0.991</td></tr><tr><td>Prior-Depth-Anything</td><td>1.667</td><td>0.760</td><td>0.753</td><td>0.841</td><td>0.921</td><td>1.167</td></tr><tr><td>DMD3C</td><td>0.548</td><td>0.536</td><td>0.533</td><td>0.563</td><td>0.661</td><td>1.596</td></tr><tr><td>nuscenes</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours</td><td>0.968</td><td>1.067</td><td>1.177</td><td>1.327</td><td>1.433</td><td>1.655</td></tr><tr><td>OMNI-DC</td><td>0.537</td><td>0.598</td><td>0.689</td><td>0.843</td><td>1.021</td><td>1.852</td></tr><tr><td>Prior-Depth-Anything</td><td>0.898</td><td>0.981</td><td>1.073</td><td>1.266</td><td>1.448</td><td>1.811</td></tr><tr><td>DMD3C</td><td>1.118</td><td>1.537</td><td>2.131</td><td>3.347</td><td>4.784</td><td>8.832</td></tr><tr><td>waymo</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours</td><td>0.917</td><td>1.001</td><td>1.109</td><td>1.282</td><td>1.442</td><td>1.766</td></tr><tr><td>OMNI-DC</td><td>0.695</td><td>0.719</td><td>0.747</td><td>0.829</td><td>0.911</td><td>1.269</td></tr><tr><td>Prior-Depth-Anything</td><td>1.052</td><td>1.144</td><td>1.275</td><td>1.530</td><td>1.769</td><td>2.473</td></tr><tr><td>DMD3C</td><td>0.992</td><td>1.184</td><td>1.650</td><td>2.649</td><td>3.928</td><td>9.892</td></tr><tr><td>goose</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours</td><td>1.578</td><td>1.800</td><td>2.062</td><td>2.445</td><td>2.718</td><td>3.247</td></tr><tr><td>OMNI-DC</td><td>0.710</td><td>0.633</td><td>0.668</td><td>0.772</td><td>0.901</td><td>1.421</td></tr><tr><td>Prior-Depth-Anything</td><td>0.849</td><td>0.912</td><td>1.048</td><td>1.271</td><td>1.477</td><td>2.129</td></tr><tr><td>DMD3C</td><td>1.117</td><td>1.365</td><td>1.739</td><td>2.690</td><td>4.097</td><td>10.720</td></tr><tr><td>nyu</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours</td><td>0.269</td><td>0.294</td><td>0.300</td><td>0.315</td><td>0.330</td><td>0.521</td></tr><tr><td>OMNI-DC</td><td>0.144</td><td>0.173</td><td>0.214</td><td>0.258</td><td>0.328</td><td>0.704</td></tr><tr><td>Prior-Depth-Anything</td><td>0.184</td><td>0.211</td><td>0.245</td><td>0.266</td><td>0.306</td><td>0.486</td></tr><tr><td>DMD3C</td><td>1.385</td><td>1.989</td><td>2.666</td><td>3.566</td><td>5.016</td><td>17.832</td></tr></table>

Table 6: Anchor-density sweep: RMSE (m) as a fraction of the returns is withheld. Every method receives the identical thinned cloud on each frame, drawn once per frame and shared, and is scored against the same target. 100% is the temporally accumulated cloud: 135 351 anchors per frame on KITTI (31.6% of all pixels), 35 815 on nuScenes (2.5%), 109 119 on Waymo (4.4%) and 102 247 on GOOSE (4.2%).

Two baselines are non-monotone: OMNI-DC improves from 1.095 to 0.663 on KITTI and from 0.710 to 0.633 on GOOSE as the cloud is thinned, and Prior-Depth-Anything from 1.667 to 0.753 on KITTI. This is not an artefact of the sweep — the thinned subset is drawn once per frame and given to every method alike, and our own curve is monotone on all five datasets. It follows from how those two clouds are accumulated. KITTI is registered by scan-to-scan ICP, which drifts, and GOOSE is accumulated from ego-motion alone, so anything that moves smears along its trajectory; nuScenes and Waymo use dataset ego-poses with per-object motion compensation, and neither shows the efect. Accumulation on KITTI and GOOSE therefore injects anchors that are individually wrong. A completion network propagates each measurement almost as a hard constraint, so a wrong anchor writes a wrong depth into its neighbourhood, and thinning removes a share of them and lets the image evidence take over. Our solve is afected far less because a bilateral cell pools several anchors — 3.9 per vertex at the default $\sigma _ { s } -$ so no single measurement moves the field.

Table 9 corrupts the anchors instead of removing them. Only the input is perturbed; the evaluation target is the unperturbed held-out cloud, so the numbers measure damage to the prediction rather than a moved target. Depth noise is comparatively benign — a 5% multiplicative error moves every dataset by about 3% or less — whereas projection error is not: a 5-pixel shift raises KITTI error by 95%. FounRef is thus robust to moderate depth noise but remains sensitive to extrinsic or timing miscalibration, motivating accurate sensor synchronisation.

Solver sensitivity. The solver exposes two continuous settings, the bilateral cell $\sigma _ { s }$ and the smoothness weight λ, and a handful of discrete ones — solve resolution, conjugategradient budget, bistochastic steps — which Table 8 ablates separately. Table 7 sweeps $\sigma _ { s }$ at fixed λ around the default configuration and reports the accuracy–structure pair each setting produces.

Neither axis is fragile. Normal dispersion moves by a few percent across the whole range from $\sigma _ { s } = 1 6$ down to 3, an order of magnitude in cell size, and degrades only at $\sigma _ { s } = 2$ (Waymo 4.85 → 6.00, GOOSE 9.35 → 11.18), where the cell is too fine for neighbouring anchors to be pooled at all. Accuracy, meanwhile, improves monotonically as the cell shrinks: the default $\sigma _ { s } = 1 6$ is the least accurate setting in the sweep, giving up 21–33% of its RMSE to $\sigma _ { s } = 4$ on the four outdoor datasets and 6% on NYU, and the frame-averaged dispersion does not object.

Figure 9 is why we keep it anyway. At $\sigma _ { s } = 4$ the correction stops travelling along road and facade surfaces and concentrates into small blobs at individual anchors: the extra accuracy is bought by pulling the depth map locally towards each measurement, which is the surface damage the method exists to avoid.

The structure metric cannot report it. Normal dispersion is the median angle between each surface normal and its own $1 5 \times 1 5$ local mean over every valid pixel, so it describes the typical pixel, and anchor blobs are a sparse minority that no median will follow. That robustness is deliberate — the damage it exists to catch in the learned baselines is global corrugation, for which the median is the right summary — but it leaves the metric blind to localised damage of our own, and a high quantile of the same angle would be the natural way to expose it.

We take the coarser cell and report the point accuracy it gives up, since a depth map used for reconstruction or 3D projection needs the surface rather than the last decimal of RMSE.

Table 8 toggles each structural component of the solver alone, with runtime, so that a component’s contribution can be read against its cost. Solve time is dominated by the conjugate-gradient budget at half resolution, and moves only for the two components that touch one of those: the iteration count (27 and 42 ms) and the solve resolution (40 ms).

<table><tr><td rowspan="2"> $\sigma _ { s }$ </td><td colspan="2">KITTI</td><td colspan="2">nuScenes</td><td colspan="2">Waymo</td><td colspan="2">GOOSE</td><td colspan="2">NYU</td></tr><tr><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td></tr><tr><td>16 (default)</td><td>0.832</td><td>12.01</td><td>0.968</td><td>6.94</td><td>0.917</td><td>4.85</td><td>2.254</td><td>9.35</td><td>0.269</td><td>7.53</td></tr><tr><td>12</td><td>0.794</td><td>11.88</td><td>0.908</td><td>6.84</td><td>0.859</td><td>4.80</td><td>2.043</td><td>9.29</td><td>0.264</td><td>7.53</td></tr><tr><td>8</td><td>0.736</td><td>11.71</td><td>0.826</td><td>6.72</td><td>0.791</td><td>4.77</td><td>1.798</td><td>9.18</td><td>0.258</td><td>7.53</td></tr><tr><td>6</td><td>0.700</td><td>11.60</td><td>0.776</td><td>6.65</td><td>0.748</td><td>4.78</td><td>1.651</td><td>9.18</td><td>0.255</td><td>7.53</td></tr><tr><td>4</td><td>0.656</td><td>11.49</td><td>0.713</td><td>6.61</td><td>0.704</td><td>4.85</td><td>1.513</td><td>9.36</td><td>0.253</td><td>7.53</td></tr><tr><td>3</td><td>0.630</td><td>11.64</td><td>0.677</td><td>6.67</td><td>0.683</td><td>5.06</td><td>1.443</td><td>9.75</td><td>0.253</td><td>7.55</td></tr><tr><td>2</td><td>0.607</td><td>12.37</td><td>0.655</td><td>7.06</td><td>0.674</td><td>6.00</td><td>1.365</td><td>11.18</td><td>0.258</td><td>7.60</td></tr></table>

Table 7: Solver operating points: the bilateral cell σ swept at constant λ, over 50 frames per dataset. RMSE in metres and normal dispersion in degrees side by side, so the accuracy–structure trade of each setting reads across a row.
<table><tr><td rowspan="2">variant</td><td colspan="2">KITTI</td><td colspan="2">nuScenes</td><td colspan="2">Waymo</td><td colspan="2">GOOSE</td><td colspan="2">NYU</td><td rowspan="2">ms</td></tr><tr><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td><td>RMSE</td><td>disp.</td></tr><tr><td>default configuration</td><td>0.832</td><td>12.01</td><td>0.968</td><td>6.94</td><td>0.917</td><td>4.85</td><td>2.254</td><td>9.35</td><td>0.269</td><td>7.53</td><td>35</td></tr><tr><td>full-resolution solve</td><td>0.793</td><td>11.52</td><td>1.021</td><td>6.58</td><td>0.931</td><td>4.62</td><td>2.622</td><td>9.35</td><td>0.282</td><td>7.50</td><td>40</td></tr><tr><td>CG 2 iterations</td><td>0.868</td><td>11.25</td><td>1.195</td><td>6.54</td><td>1.074</td><td>4.46</td><td>3.744</td><td>9.69</td><td>0.293</td><td>7.48</td><td>27</td></tr><tr><td>CG 16 iterations</td><td>0.831</td><td>12.16</td><td>0.961</td><td>7.27</td><td>0.916</td><td>5.02</td><td>2.327</td><td>9.44</td><td>0.270</td><td>7.52</td><td>42</td></tr><tr><td>bistochastic 8 steps</td><td>0.832</td><td>11.97</td><td>0.967</td><td>6.92</td><td>0.917</td><td>4.84</td><td>2.253</td><td>9.20</td><td>0.269</td><td>7.53</td><td>35</td></tr></table>

Table 8: Solver components, each toggled alone from the default configuration, over 50 frames per dataset; RMSE in metres and normal dispersion in degrees side by side, ‘ms’ the mean solve time. Full-resolution solve drops the half-resolution grid. CG iterations is the conjugategradient budget of the final solve. Bistochastic normalisation is the grid balancing that makes the bilateral operator well conditioned. A component is worth keeping only if switching it of costs more than the runtime it saves.

<table><tr><td>perturbation</td><td>kitti</td><td>nuscenes</td><td>waymo</td><td>goose</td><td>nyu</td></tr><tr><td colspan="6">projection shift</td></tr><tr><td>0px</td><td>0.832</td><td>0.968</td><td>0.917</td><td>2.254</td><td>0.269</td></tr><tr><td>1px</td><td>0.911</td><td>0.969</td><td>0.932</td><td>2.251</td><td>0.269</td></tr><tr><td>2px</td><td>1.042</td><td>1.003</td><td>0.972</td><td>2.254</td><td>0.270</td></tr><tr><td>3px</td><td>1.219</td><td>1.043</td><td>1.033</td><td>2.264</td><td>0.272</td></tr><tr><td>5px</td><td>1.624</td><td>1.151</td><td>1.178</td><td>2.293</td><td>0.274</td></tr><tr><td colspan="6">multiplicative depth noise</td></tr><tr><td>0%</td><td>0.832</td><td>0.968</td><td>0.917</td><td>2.254</td><td>0.269</td></tr><tr><td>1%</td><td>0.834</td><td>0.968</td><td>0.919</td><td>2.255</td><td>0.268</td></tr><tr><td>2%</td><td>0.836</td><td>0.969</td><td>0.922</td><td>2.253</td><td>0.268</td></tr><tr><td>5%</td><td>0.850</td><td>0.978</td><td>0.945</td><td>2.250</td><td>0.266</td></tr></table>

Table 9: Anchor perturbation, ours. Only the anchors are corrupted; the evaluation target is the unperturbed held-out cloud (dense GT on KITTI/NYU), so the numbers measure damage to the prediction rather than a moved target.

Figures 10, 11 and 12 close the appendix with every method on the same frames, seen three ways: as predicted depth, as the surface normals of those same predictions, and re-projected to 3D. The depth maps are the representation in which the methods are hardest to tell apart, and the clouds the one in which they are easiest.

![](images/26e7e0000af943bb118c86c17096fb1290c0a50978ef3a8de31258f0277bd59d.jpg)  
Figure 9: Solver sensitivity, two frames per dataset: the anchors given to the solver, depth at the default $\sigma _ { s } = 1 6 ,$ at $\sigma _ { s } = 4 ,$ and their signed diference (blue nearer, red farther, grey unchanged, each panel on its own scale). At $\sigma _ { s } = 4$ the correction concentrates into blobs at individual anchors instead of travelling along surfaces — the 21–33% RMSE it buys is paid for in surface damage. NYU is omitted: its two settings difer by under a centimetre.

![](images/8e75468ea1eb1267a863fcbc72274f49df8ed130c2deb185b0c02814ce57f03d.jpg)  
Figure 10: Predicted depth for every evaluated method on one frame of each dataset, alongside the input image. Frames are drawn at random from each dataset, except GOOSE, where a forest track was chosen over the open field the draw returned. Our sky and other pixels where th monocular prior is undefined are drawn at the depth cap, since we emit no prediction there. The supervised networks hold up on the driving scenes that resemble their training data and come apart away from them, most visibly indoors. On the driving rows every method returns a plausible map — and on those same frames their normal dispersion spans 1.7 to 56 degrees (Figure 11). What separates these methods is not visible in this representation.

![](images/fd4df749105c6b0a1ff170de2f3623c9d9385389eae3b8ec59cbbf844ba8aefc.jpg)  
Figure 11: Surface normals of the same predictions, with each panel’s RMSE (m) and normal dispersion (deg) beneath it. The completion networks imprint the sensor into the surface: the ruling visible across the road is the LiDAR sweep pattern of the anchor column reproduced as geometry. The point metric does not register it — DMD3C and BP-Net are both more accurate than we are on the KITTI frame while carrying seven to eight times the normal error, because reproducing a measurement exactly is rewarded by RMSE whether or not the surfac between measurements survives. We estimate a smooth metric correction over a prior we do not modify, so the anchor pattern cannot enter the geometry in the first place.

![](images/cdc0fec020a3bdcbcd57516e14132c6a52fe31c89b47bbbf299cbe93c1141b29.jpg)  
Figure 12: The same depth maps, turned back into 3D. For each scene the upper row is the input image and each method’s depth map; the lower row is the anchors the solver was given, drawn over the image, and each depth map re-projected into a coloured point cloud, all seen from a direction the camera never occupied (35<sup>◦</sup> azimuth, 18<sup>◦</sup> elevation). From the original viewpoint all four look alike by construction, since each is consistent with the same image. The depth maps are indeed hard to separate — but only ours re-projects to continuous surfaces. On Waymo the baselines smear the buildings and trees behind the road into sheets of stray points that hang over the scene; on NYU, where OMNI-DC and Prior-DA are metrically ahead of us, the room still resolves for all three and only DMD3C disintegrates. This is what the normal-dispersion column measures, and what a downstream consumer of the depth map receives.