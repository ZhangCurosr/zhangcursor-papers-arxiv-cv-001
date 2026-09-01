# Failure or Drift? Evaluating Monocular SLAM under Synthetic and Real-World Corruptions

Abhay Skaria Thomas<sup>1</sup> , Shashank Agnihotri<sup>1</sup> , and Margret Keuper<sup>1,2</sup>

<sup>1</sup> Machine Learning Group, University of Mannheim, Germany <sup>2</sup> Max-Planck-Institute for Informatics, Saarland Informatics Campus, Germany {shashank.agnihotri,keuper}@uni-mannheim.de

Abstract. Visual SLAM is commonly evaluated on clean trajectories, although deployment failures are often caused by adverse weather, illumination, blur, and sensor artifacts. Controlled corruptions are attractive because they isolate such factors, but a synthetic stress test is useful only when it leads to the same engineering conclusion as the condition it is intended to approximate. This work examines that question for monocular SLAM. We evaluate a classical feature-based system and two learned trackers under image-space, geometry-aware, and compound corruptions, and compare their behavior with adverse conditions from 4Seasons. Rather than reducing robustness to a single trajectory error, the evaluation separates explicit tracking failure from drift accumulated by methods that remain active. The results show that learned trackers largely replace catastrophic loss with sustained, and sometimes severe, drift. More importantly, the apparent ordering of the learned systems changes with the physical fidelity of the corruption: structured rain and fog proxies preserve the real-world ordering, whereas a simple illumination proxy does not. Code is available in this: GitHub repository.

Keywords: Visual SLAM · camera tracking · robustness · synthetic corruptions · distribution shift · adverse conditions

## 1 Introduction

Camera tracking is a prerequisite for autonomous driving, mobile robotics, augmented reality, and online 3D reconstruction. The standard evaluation of a monocular SLAM system nevertheless remains dominated by clean benchmark trajectories. Such measurements are necessary, but they do not describe the conditions under which a system will be used. Cameras encounter rain, fog, low illumination, motion blur, compression, exposure variation, and sensor noise [10, 11, 27, 34, 36]. These changes weaken visual correspondences and can disrupt both local tracking and the optimization that carries information across time.

The resulting failure is not always explicit. A feature-based system can lose tracking and return no usable trajectory. A learned tracker can remain active while its error accumulates over the sequence. The first outcome is easy to detect; the second can appear well-behaved until an incorrect pose reaches mapping, planning, or control. Robustness must therefore distinguish whether a trajectory is available from whether the available trajectory remains accurate.

Real adverse-condition datasets provide the most faithful evidence, but they make controlled diagnosis dificult. In 4Seasons [42], for example, weather and illumination change together with route geometry, trafic, camera response, and motion. Synthetic corruptions provide the complementary intervention: the route, calibration, timestamps, and ground truth can remain fixed while the visual input changes. This control has made common-corruption benchmarks useful across vision tasks. For a stateful system such as SLAM, however, visual plausibility is not suficient. A corruption may look similar to an adverse condition while changing a diferent subset of correspondences and, consequently, favoring a diferent tracker.

This work studies the external validity of such stress tests. We compare ORB-SLAM2 [28], DPVO [39], and DROID-SLAM [38] without corruption-specific adaptation. Clean performance is established on KITTI [14] sequences 00–10. The controlled study uses 115 corrupted variants of KITTI sequence 00, covering image-space, geometry-aware, and compound efects at five severity levels. Four 4Seasons [42] sequences provide reference, rain, winter or fog-like, and evening conditions for a conservative synthetic-to-real comparison.

The findings expose two independent decisions. First, ORB-SLAM2 often fails by returning no evaluable trajectory, whereas both learned systems continue in every archived synthetic attempt and carry the risk of substantial drift. Second, the corruption model changes the comparison between the learned trackers. DPVO has lower APE in 30 of 43 paired image-space settings, while DROID-SLAM has lower APE in all 33 paired geometry-aware settings and 14 of 15 compound settings. Rain and winter preserve the compound-proxy ordering in real data; evening illumination reverses it. A benchmark constructed only from convenient image transformations can therefore recommend the wrong architecture for the condition it is intended to represent.

Our main contributions are:

– a controlled evaluation of three monocular SLAM paradigms across imagespace, geometry-aware, compound, and natural adverse conditions;

– a failure-aware protocol that reports trajectory validity before APE and RPE conditioned on a valid output, avoiding comparisons that conflate tracking loss with exporter density;

– evidence that corruption fidelity changes tracker selection, together with a direct analysis of which synthetic conclusions transfer to real adverse data.

## 2 Related Work

In the following, we review the monocular SLAM paradigms and robustness benchmarks most relevant to this work, before discussing synthetic corruption protocols and their validity as proxies for real adverse conditions.

## 2.1 Monocular Tracking and Dense Scene Representations

ORB-SLAM2 [28] combines ORB [33] keypoints with geometric pose estimation, local mapping, loop closure, and relocalization. Its explicit front end provides interpretable failure signals, but initialization and tracking depend on repeatable local descriptors. DPVO [39] replaces handcrafted keypoints with learned patch correspondences and couples a recurrent update operator with diferentiable bundle adjustment. DROID-SLAM [38] uses dense learned correspondences and repeatedly refines poses and depth through dense bundle adjustment. These systems span the three behaviors central to this study: discrete correspondence failure, continued local odometry with accumulated drift, and learned tracking supported by global multi-frame optimization.

Recent systems couple tracking more tightly to dense reconstruction. GO-SLAM [44] jointly optimizes camera poses and an implicit scene representation. Gaussian Splatting SLAM [24], SplaTAM [20], and GS-SLAM [43] use 3D Gaussians for tracking and mapping. MASt3R-SLAM [29] builds a real-time dense pipeline around a two-view reconstruction prior, while SLAM3R [23] directly regresses and registers local pointmaps. Their progress makes adverse-condition evaluation increasingly important: an appearance shift can afect not only pose estimation, but also geometry, rendering, map updates, and memory.

## 2.2 Robustness Evaluation for SLAM

KITTI [14] and the TUM RGB-D benchmark [37] established reproducible trajectory evaluation under nominal conditions. TartanAir [41] broadened the range of simulated environments, motion, and weather, while 4Seasons [42] introduced repeated outdoor recordings across seasons, illumination, and weather. Such datasets reveal realistic failures, but adverse conditions remain entangled with route and sensor diferences [1, 5, 17, 32].

Measuring Robustness of Visual SLAM [32] demonstrated the importance of evaluating variation across more than a small set of trajectories. VSLAM-LAB [12, 13] addresses a complementary source of variation by standardizing compilation, dataset preparation, configuration, execution, and evaluation. SLAM Adversarial Lab [15] provides modular perturbations, interpretable severity units, and a search for failure boundaries. Here, we ask a diferent question: does the conclusion obtained from a controlled perturbation remain stable when the proxy becomes more physically structured and is then tested on natural adverse data?

## 2.3 Synthetic Corruptions and Their Real-World Validity

ImageNet-C [16] established parameterized common corruptions as a practical robustness protocol. The imagecorruptions package [25,26] exposes many of these transformations for arbitrary images. 3D Common Corruptions [18] incorporates depth and scene geometry into efects such as fog and focus changes. Robust-Spring [30] extends corruption evaluation to optical flow, scene flow, and stereo with temporal, stereo, and depth consistency.

Are Synthetic Corruptions A Reliable Proxy For Real-World Corruptions? [1, 6, 7] shows that strong aggregate correspondence in semantic segmentation can coexist with corruption-specific mismatch. The question is more demanding for SLAM because errors propagate through a stateful estimation process. A local correspondence change can alter subsequent poses, optimization, and map state. We therefore assess synthetic-to-real validity through failure behavior and relative method ordering, not through equality of absolute trajectory errors.

## 3 Experimental Design

To isolate the efect of visual degradation, all methods are evaluated under a common protocol without corruption-specific adaptation. The following subsections describe the datasets and corruptions, evaluated systems, and failure-aware trajectory evaluation.

## 3.1 Datasets and Corruptions

The clean baseline contains the eleven KITTI [14] odometry training sequences with ground-truth trajectories, evaluated in monocular mode. The controlled corruption study is generated from sequence 00. This sequence is suficiently long to expose both local tracking errors and long-range drift, while keeping route, calibration, timestamps, and ground truth fixed across conditions. The use of a single controlled route is intentional for diagnosis and is treated as a limitation for generalization.

The image-space branch contains 13 transformations from the imagecorruptions package [25, 26]: brightness, contrast, defocus blur, elastic transform, frost, Gaussian noise, impulse noise, JPEG compression, motion blur, pixelation, shot noise, snow, and zoom blur. The geometry-aware branch follows 3D Common Corruptions [18] and contains far focus, near focus, flash, fog, ISO noise, low light, and rain. These efects use depth or physically motivated scene structure where applicable. The compound branch combines brightness with contrast, fog with contrast, and rain with contrast. Each family has five nominal severity levels. The complete synthetic collection therefore contains 65 image-space, 35 geometry-aware, and 15 compound sequence variants.

The real-world comparison uses four neighborhood sequences from 4Seasons [42]. We use neighborhood 2 train as a cloudy-afternoon reference, neighborhood 3 train as rainy afternoon, neighborhood 4 train as winter cloudy morning, and neighborhood 6 train as cloudy evening. The corresponding synthetic conditions are clean KITTI, rain with contrast, fog with contrast, and brightness with contrast. These are condition-level matches rather than paired recordings; route, camera, motion, and scene geometry remain diferent.

## 3.2 Systems and Execution

ORB-SLAM2 [28] represents sparse feature-based SLAM with loop closure and relocalization. DPVO [39] represents learned patch-based visual odometry without classical loop closure. DROID-SLAM [38] represents dense learned correspondence with global multi-frame optimization. Released weights and dataset-appropriate settings are used without fine-tuning on corrupted data, test-time restoration, or condition-specific adaptation.

All runs are executed through VSLAM-LAB [12, 13]. Dataset and experiment YAML files define image paths, timestamps, calibration, method modules, and method-specific parameters. Every execution archives its trajectory, logs, runtime records, and evaluation outputs. ORB-SLAM2 is executed on CPU; DPVO and DROID-SLAM use NVIDIA A100 or H100 GPUs. The archive contains more than 350 executions. Missing configurations are retained as missing and are never converted into failures or imputed measurements. Quantitative comparisons between DPVO and DROID-SLAM use only conditions for which both results are present.

## 3.3 Failure-Aware Evaluation

For an attempted run r, we first record whether it returns a non-empty trajectory that can be associated with ground truth,

$$
V _ { r } = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { e v a l u a b l e ~ t r a j e c t o r y ~ r e t u r n e d , } } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{1}
$$

The valid-run rate is the mean of $V _ { r }$ over archived attempts. This binary measure is deliberately narrower than trajectory coverage. ORB-SLAM2 commonly exports keyframes, whereas the learned systems can export poses at a diferent temporal density. Some archived corruption sequences are also subsampled. Dividing the number of exported poses by the original number of frames would therefore mix keyframe density, input sampling, and genuine tracking loss.

For $V _ { r } = 1$ , poses are associated within 0.01 s and the monocular trajectory is aligned by a similarity transform $S \in \mathrm { S i m } ( 3 )$ . We report translation RMSE for the standard absolute and relative pose errors used by the TUM RGB-D benchmark [37],

$$
\mathrm { A P E } _ { \mathrm { R M S E } } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \mathrm { t r a n s } \Big ( T _ { i } ^ { - 1 } S \hat { T } _ { i } \Big ) \right\| _ { 2 } ^ { 2 } } ,\tag{2}
$$

$$
\mathrm { R P E } _ { \mathrm { R M S E } } = \sqrt { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Big \| \mathrm { t r a n s } \Big ( ( T _ { i } ^ { - 1 } T _ { i + \varDelta } ) ^ { - 1 } ( \hat { T } _ { i } ^ { - 1 } \hat { T } _ { i + \varDelta } ) \Big ) \Big \| _ { 2 } ^ { 2 } } .\tag{3}
$$

APE measures accumulated global disagreement; RPE measures local motion inconsistency. Both are conditioned on a valid trajectory so that a method cannot obtain a favorable error by returning no output.

To compare the learned trackers, we use

$$
\Delta \mathrm { A P E } = \mathrm { A P E } _ { \mathrm { D P V O } } - \mathrm { A P E } _ { \mathrm { D R O I D - S L A M } } ,\tag{4}
$$

where a positive value favors DROID-SLAM. For synthetic-to-real analysis, each compound result is averaged over its five severity levels before it is compared with the corresponding real condition. We also report descriptive Spearman correlations [35] between geometry-aware and compound severity curves. Since each correlation contains only five severity points, it is used as a diagnostic rather than an inferential statistic.

## 4 Results

We organize the results around four questions concerning how the systems fail, whether corruption fidelity changes tracker selection, how conclusions vary across proxy layers, and which findings transfer to real adverse conditions.

## 4.1 Do the Systems Fail, or Do They Drift?

The clean KITTI runs provide an important reference before corruption is introduced. ORB-SLAM2, DPVO, and DROID-SLAM each return an evaluable trajectory on all 11 sequences. ORB-SLAM2 exports a sparser keyframe trajectory, so its raw number of stored poses is not interpreted as cross-method coverage. Yet clean input does not produce a stable accuracy ordering. ORB-SLAM2 has the lowest median conditional APE (8.2 m), but reaches 518.0 m on sequence 01. DPVO and DROID-SLAM have median APEs of 64.9 m and 82.0 m, respectively, and their relative performance changes with the route: DPVO is substantially better on sequence 01, whereas DROID-SLAM is better on sequence 08. Even without corruption, successful execution and dependable global accuracy are therefore diferent properties.

Figure 1 shows how corruption changes this scenario. DPVO and DROID-SLAM return an evaluable trajectory for every archived synthetic attempt. ORB-SLAM2 is valid in 37 of 63 image-space attempts, 15 of 23 geometry-aware attempts, and 10 of 15 compound attempts. The image-space archive contains 25 empty ORB-SLAM2 trajectories and one synchronization failure. Gaussian noise, impulse noise, and snow produce no evaluable ORB-SLAM2 trajectory at any recorded severity; brightness and JPEG compression are substantially less destructive.

The corruption results expose a sharper diference between explicit failure and silent degradation. ORB-SLAM2 often stops returning an evaluable trajectory, whereas the learned systems continue to produce poses while their aligned APE frequently exceeds 100 m under geometry-aware and compound efects. This distinction matters operationally. Loss of tracking can trigger relocalization or sensor fallback; a continuous but inaccurate estimate may pass through the rest of the autonomy stack without a comparable warning.

![](images/3d568ee1e86de81852d58804db1dd72dea62ee10ac445ad14f843a0ff713ef26.jpg)  
Fig. 1: Clean and corrupted inputs reveal diferent aspects of failure. (a) All three systems return an evaluable trajectory on each clean KITTI sequence; under synthetic corruption, ORB-SLAM2 frequently returns no evaluable trajectory, while DPVO and DROID-SLAM remain valid in every archived attempt. Missing, unexecuted configurations are excluded from the denominators. (b) Conditional APE across clean KITTI sequences 00–10; diamonds mark sequence 00, the nominal reference used for the corruption sweep. The clean distribution shows strong route dependence and is not a matched degradation curve for every corrupted archive. (c) Conditional APE for paired DPVO and DROID-SLAM settings after tracking continues.

Clean data reveal route-dependent drift; corruption changes the failure mechanism. All three systems return evaluable trajectories on clean KITTI, although their errors vary strongly by route. Under corruption, ORB-SLAM2 often fails explicitly, whereas DPVO and DROID-SLAM remain active and can accumulate substantial drift.

## 4.2 Does Corruption Fidelity Change Which Tracker Appears Robust?

It does, and the reversal is systematic. Among the 43 paired image-space settings with complete learned-method results, DPVO has lower APE in 30. Brightness and contrast favor DPVO at every tested severity. A benchmark restricted to these inexpensive global transformations would therefore select DPVO.

The geometry-aware branch produces the opposite result. DROID-SLAM has lower APE in all 33 paired settings, including every available severity of low light, fog, rain, flash, ISO noise, and focus changes. The compound branch largely preserves that ordering: DROID-SLAM has lower APE in 14 of 15 settings, with fog and contrast at severity 5 as the only reversal. Figure 2 shows the full distribution rather than only its mean.

![](images/43b0a6f33eea073a516539245df20d3fcc147ec75d835acd4181cdd94acc4da7.jpg)  
Fig. 2: The corruption model changes tracker selection. Each point is a paired condition with complete DPVO and DROID-SLAM results; $\varDelta \mathrm { A P E } > 0$ favors DROID-SLAM. DPVO has lower APE in 30/43 image-space settings, whereas DROID-SLAM has lower APE in 33/33 geometry-aware settings and 14/15 compound settings.

This result should not be read as a component ablation. The experiments do not isolate the contribution of dense correspondences, recurrent updates, or global optimization. They do show that an image-level proxy is not simply a less accurate version of a geometry-aware proxy. The two can alter correspondence support diferently and lead to opposite decisions about the preferred system.

Corruption fidelity changes the preferred tracker. The image-space study mostly favors DPVO, while every paired geometry-aware setting and 14 of 15 compound settings favor DROID-SLAM.

## 4.3 Are the Trends Consistent across Image-Space, Geometry-Aware, and Compound Efects?

There is no method-independent mapping between the proxy layers. Figure 3(b) compares geometry-aware APE curves with their compound counterparts. For $\mathrm { \ D P V O }$ , low light and brightness with contrast have identical severity ordering $( \rho = 1 . 0 0 )$ , while rain and rain with contrast are almost unrelated $( \rho = - 0 . 1 0 )$ DROID-SLAM shows the opposite pattern: its rain curves correlate strongly $( \rho = 0 . 8 0 )$ , whereas its low-illumination curves do not $( \rho = - 0 . 1 0 )$ . Fog is similarly method dependent, with $\rho = 0 . 9 0$ for DPVO and $\rho = - 0 . 2 0$ for DROID-SLAM.

Nominal severity is not a common dificulty coordinate either. Under geometryaware rain, DROID-SLAM APE decreases from 111.9 m at severity 1 to 87.9 m at severity 5, while DPVO remains between 125.5 m and 129.2 m. This decrease does not mean that the tracker improves as rain becomes more severe. The corruption changes which image structure remains usable and, in turn, the trajectory reached by the optimizer. For SLAM, severity orders the corruption parameters, not necessarily the estimation dificulty.

![](images/92e9625c79ad5e33d04c4c89a2782febf0582a07786fab2416d6e8a86ff5cb42.jpg)

![](images/c39454fffa501baea0f421a7e7dacbb43436a9f345997e4726363b4642f0ba47.jpg)  
Fig. 3: Synthetic-to-real evidence across matched conditions. (a) Learned-tracker margin through the proxy ladder; $\varDelta \mathrm { A P E } > 0$ favors DROID-SLAM. Rain and winter retain the compound-proxy ordering, while real evening illumination reverses to DPVO. (b) Spearman correlation between geometry-aware and compound severity curves; blue circles denote DPVO and orange diamonds denote DROID-SLAM. High agreement between two synthetic curves does not guarantee real-world transfer.

A single mean corruption score or a single correlation can therefore hide the mechanism of interest. Agreement depends on the tracker, the physical efect, the metric, and the portion of the sequence over which the trajectory remains evaluable.

No single relationship holds across proxy layers. Correlations change with the tracker and condition, and nominal corruption severity does not consistently order SLAM dificulty.

## 4.4 Which Conclusions Transfer to Real Adverse Conditions?

Figure 3(a) follows the learned-tracker margin from a simple image-space transform to a geometry-aware efect, its compound proxy, and the corresponding 4Seasons condition. Rain provides the strongest agreement. DROID-SLAM is better by 17.96 m for rain with contrast averaged over severities and by 16.40 m on the real rainy sequence. Fog with contrast also selects DROID-SLAM, and the same ordering appears in the real winter sequence, with margins of 2.27 m and 5.15 m, respectively.

Evening illumination is the counterexample. Brightness with contrast favors DROID-SLAM by 5.51 m, but the real cloudy evening favors DPVO by 10.85 m. Notably, DPVO has $\rho = 1 . 0 0$ between the geometry-aware low-light and compound severity curves. Perfect agreement between the two synthetic constructions still fails to preserve the real tracker ordering. Global brightness and contrast do not capture camera-specific noise, automatic exposure, spatially varying illumination, headlights, motion-dependent blur, or the route itself [11, 27, 36].

The transfer result is therefore two out of three adverse conditions, not a claim of synthetic-real equivalence. Absolute error magnitudes are not comparable

across diferent routes, sensors, and motion profiles. The useful conclusion is narrower: structured rain and fog proxies can reveal an architectural weakness that persists in natural data, while an intuitive illumination proxy can select the wrong tracker.

Only some conclusions transfer. Rain and winter preserve the compound-proxy ordering; evening illumination does not, and none of the synthetic settings predicts absolute real-world error.

## 5 Discussion

The experiments expose not only diferent levels of robustness, but fundamentally diferent ways of failing. ORB-SLAM2 [28] uses the same ORB [33] representation for tracking, local mapping, relocalization, and loop detection. This coupling is both a strength and a weakness. When a suficient set of repeatable keypoints survives, geometric verification, bundle adjustment, and loop closure can reject inconsistent matches and correct accumulated drift. When noise, blur, or structured occlusion destabilizes that feature set, the corruption simultaneously weakens frame-to-map tracking and the evidence required for relocalization or loop closure. The frequent empty trajectories under noise and snow are therefore consistent with a threshold-like breakdown of the shared feature representation. The route-dependent clean results follow the same reasoning: explicit global correction is valuable when reliable correspondences and revisited regions are available, but it cannot compensate for insuficient or ambiguous visual evidence.

DPVO [39] replaces detector-selected keypoints with a sparse set of learned feature patches. A recurrent update operator estimates patch trajectories across nearby frames, predicts a confidence weight for each correspondence factor, and updates patch depths and camera poses through diferentiable bundle adjustment. This soft, learned correspondence model provides a plausible mechanism for the observed tracking continuity after the ORB front end has already failed. However, DPVO is a visual odometry system: its online optimization is restricted to a local window and does not include loop closure or a global correction backend. Small but persistent correspondence biases can consequently remain locally compatible while accumulating into substantial global drift.

DROID-SLAM [38] difers from DPVO in two important respects. It reasons over dense feature correlations rather than a sparse patch set, and its backend performs global bundle adjustment over the keyframe history, including long-range graph connections for loop closure. Dense evidence provides redundancy when individual image regions become unreliable, while joint multi-frame refinement can distribute and correct local correspondence errors. This design is consistent with DROID-SLAM obtaining lower APE across the available geometry-aware comparisons and in 14 of the 15 compound settings. Nevertheless, global optimization can only be as reliable as the evidence supplied to it. If a scene-wide corruption systematically biases the dense correspondence field, bundle adjustment may produce a smooth and internally consistent trajectory that remains globally incorrect. Continued output, low local residuals, or low RPE therefore do not by themselves certify low accumulated drift. Since this work does not ablate individual architectural components, these links should be interpreted as mechanism-consistent explanations rather than causal attributions.

This distinction also sharpens the diference between availability and trustworthiness. DPVO [39] and DROID-SLAM [38] already predict confidence weights that control how individual correspondence factors contribute to bundle adjust ment. These weights are learned indirectly through the pose-estimation objective; they are not supervised as calibrated probabilities that the final trajectory is correct. They should therefore not be interpreted as trajectory-level uncertainty without additional validation. A deployment-facing system should expose signals that are tested against accumulated drift and impending tracking loss, potentially combining correspondence confidence, geometric residuals, graph connectivity, agreement between local and global estimates, and temporal consistency. The relevant question is not only whether the tracker can return another pose, but whether that pose remains suficiently constrained to be trusted.

The results further show why corruption design must follow the mechanism being tested. Image-space transformations remain useful for broad and inexpensive screening, particularly for compression artifacts, global intensity changes, blur, and high-frequency noise [2–4,22]. They primarily probe the stability of the visual front end. Geometry-aware corruptions additionally alter which parts of the scene remain informative as a function of depth and spatial structure, while compound corruptions test whether several individually manageable changes interact inside the correspondence and optimization pipelines. The reversal of the preferred learned tracker across diferent proxy layers shows that no single image-space corruption score can represent robustness in general. A useful evaluation should instead form a proxy ladder: image-space corruptions for breadth, geometry aware and temporally coherent corruptions for mechanism fidelity, and natural adverse-condition sequences for external validation. RobustSpring [30] provides an instructive example of preserving temporal, stereo, and depth consistency when corrupting correspondence data.

Finally, recent dense systems couple tracking quality to the state of the reconstructed map. Gaussian Splatting SLAM [24] optimizes camera poses directly against an incrementally updated Gaussian scene representation. An adverse observation can therefore influence both the current pose estimate and the map that will be used to track subsequent frames. MASt3R-SLAM [29] combines a learned two-view reconstruction prior with pointmap matching, local fusion, loop closure, and global optimization. SLAM3R [23] takes a substantially diferent route by directly regressing local pointmaps and registering them into a global reconstruction without explicitly solving for camera parameters. These systems may continue producing visually plausible dense output even when geometry becomes incomplete, warped, or inconsistent. Robustness evaluation must consequently distinguish trajectory validity, map validity, and systems validity. In addition to APE and RPE, future evaluations should measure reconstruction accuracy and completeness, cross-view geometric consistency, rendering degradation, map growth, runtime stability, and memory consumption.

## 6 Limitations

The controlled sweep uses one KITTI [14] trajectory and one archived execution per condition, so scene dependence, geographical diversity [9], initialization variance, and runtime nondeterminism are not measured. The binary validity criterion distinguishes evaluable output from missing output but does not quantify how much of a truncated trajectory is supported in time. Several configurations are absent from the archive; all paired analyses use only complete DPVO–DROID-SLAM conditions and expose their sample counts. The 4Seasons [42] conditions are semantically matched rather than paired recordings of the same route and camera, so they support comparisons of relative behavior, not causal or numerical equivalence. The study is monocular and trajectory-centered, and does not cover other imaging modalities [31], inertial fusion, map accuracy, reconstruction completeness, runtime, memory, or energy.

## 7 Future Work

An extension will introduce temporally coherent, sensor-aware corruptions and a broader set of matched natural conditions, together with map-quality and resource measurements. The immediate aim is to establish which controlled proxies reproduce specific field failure modes, rather than to compress accuracy, continuity, reconstruction, and eficiency into one score. Structured graphical formulations provide another possible direction [19, 21]. Alternative feature representations may provide another analysis direction [40]. Internal representation analysis is another complementary direction [8]. Additionally adding synthetic corruptions to more real-world datasets and extend the benchmarking and study to more SLAM methods.

## 8 Conclusion

This work evaluates monocular SLAM under controlled and natural visual degradation. ORB-SLAM2 frequently fails by returning no evaluable trajectory, while DPVO and DROID-SLAM continue producing poses and instead accumulate drift. These are diferent operational failures and should be reported separately. More importantly, the apparent ranking of the learned trackers changes with the corruption model: image-space efects mostly favor DPVO, whereas geometry-aware and compound efects strongly favor DROID-SLAM.

Synthetic corruptions are therefore valuable as controlled diagnostic interventions, but not as calibrated forecasts of deployment error. Rain and winter preserve the compound-proxy ordering in real data; evening illumination does not. Robustness evaluation should ask whether tracking remains available, whether its error stays bounded, and whether the conclusion survives the natural condition the proxy is intended to represent.

## Acknowledgments

Shashank Agnihotri and Margret Keuper acknowledge support by DFG Research Unit 5336 – Learning to Sense. Margret Keuper acknowledges funding by BMFTR project TrackOpt (01IS24074A-D). The authors acknowledge support by the state of Baden-Württemberg through bwHPC. The authors acknowledge support by the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant INST 35/1597-1 FUGG. The authors gratefully acknowledge the computing time provided on the high-performance computer HoreKa by the National High-Performance Computing Center at KIT (NHR@KIT). This center is jointly supported by the Federal Ministry of Research, Technology and Space and the Ministry of Science, Research and the Arts of Baden-Württemberg, as part of the National High-Performance Computing (NHR) joint funding program. HoreKa is partly funded by the German Research Foundation (DFG).

## Resources

The dataset is available at: https://data.dws.informatik.uni-mannheim.de/ machinelearning/Failure\_or\_Drift\_dataset/.

The code is available at: https://github.com/abhaythomas/master\_thesis\_ vslamlab\_robustness.

## References

1. Agnihotri, S., Caspary, J.Y., Schwarz, L., Gao, X., Schmalfuss, J., Bruhn, A., Keuper, M.: Flowbench: Benchmarking optical flow estimation methods for reliability and generalization. Transactions on Machine Learning Research (2025)

2. Agnihotri, S., Gandikota, K.V., Grabinski, J., Chandramouli, P., Keuper, M.: On the unreasonable vulnerability of transformers for image restoration – and an easy fix. In: 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). pp. 3709–3719 (2023). https://doi.org/10.1109/ICCVW60793.2023. 00398

3. Agnihotri, S., Grabinski, J., Keuper, J., Keuper, M.: Beware of aliases–signal preservation is crucial for robust image restoration. arXiv preprint arXiv:2406.07435 (2024)

4. Agnihotri, S., Grabinski, J., Keuper, M.: Improving feature stability during upsampling–spectral artifacts and the importance of spatial context. In: Euro pean Conference on Computer Vision. pp. 357–376. Springer (2024)

5. Agnihotri, S., Jung, S., Keuper, M.: CosPGD: an eficient white-box adversarial attack for pixel-wise prediction tasks. In: Forty-first International Conference on Machine Learning (2024), https://openreview.net/forum?id=CXZqGJonmt

6. Agnihotri, S., Schader, D., Jakubassa, J., Sharei, N., Kral, S., Weber, R., Keuper, M., et al.: Semsegbench & detecbench: Benchmarking reliability and generalization beyond classification. arXiv preprint arXiv:2505.18015 (2025)

7. Agnihotri, S., Schader, D., Sharei, N., Kaçar, M.E., Keuper, M.: Are synthetic corruptions a reliable proxy for real-world corruptions? In: Computer Vision and Pattern Recognition Workshop (2025), https://arxiv.org/abs/2505.04835, synthetic Data for Computer Vision Workshop

8. Alshami, E., Agnihotri, S., Schiele, B., Keuper, M.: Aim: Amending inherent interpretability via self-supervised masking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 993–1003 (2025)

9. Basu, A., Singh, M., Agnihotri, S., Keuper, M., Radhakrishnan, V.B.: Geodiv: Framework for measuring geographical diversity in text-to-image models. In: Vondrick, C., Hariharan, B., Rafel, C., Pinto, L., Yang, D., Faust, A. (eds.) International Conference on Learning Representations. vol. 2026, pp. 83680– 83730 (2026), https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 86edb2450c32577c08c6046be45ff643-Paper-Conference.pdf

10. Fatima, M., Agnihotri, S., Bock, M., Gandikota, K.V., Van Laerhoven, K., Moeller, M., Keuper, M.: γ-quant: Towards learnable quantization for low-bit pattern recog nition. In: DAGM German Conference on Pattern Recognition. pp. 57–72. Springer (2025)

11. Fatima, M., Agnihotri, S., Gandikota, K.V., Moeller, M., Keuper, M.: Rawdet-7: A multi-scenario benchmark for object detection and description on quantized raw images. arXiv preprint arXiv:2602.03760 (2026)

12. Fontan, A., Contributors: VSLAM-LAB: Oficial implementation. GitHub repository, https://github.com/VSLAM-LAB/VSLAM-LAB (2025), accessed 2026-08-03

13. Fontan, A., Fischer, T., Civera, J., Milford, M.: VSLAM-LAB: A comprehensive framework for visual SLAM methods and datasets. In: Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 9013–9020 (2025). https://doi.org/10.1109/IROS60139.2025.11247218

14. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? the KITTI vision benchmark suite. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 3354–3361 (2012). https://doi.org/10.1109/ CVPR.2012.6248074

15. Hefny, M., Dantu, K., Ko, S.Y.: SLAM Adversarial Lab: An extensible framework for visual SLAM robustness evaluation under adverse conditions. arXiv preprint arXiv:2603.17165 (2026). https://doi.org/10.48550/arXiv.2603.17165

16. Hendrycks, D., Dietterich, T.: Benchmarking neural network robustness to common corruptions and perturbations. In: International Conference on Learning Represen tations (2019)

17. Hofmann, J., Agnihotri, S., Saikia, T., Brox, T.: Towards improving robustness of compressed cnns. In: ICML Workshop on Uncertainty and Robustness in Deep Learning (UDL) (2021)

18. Kar, O.F., Yeo, T., Atanov, A., Zamir, A.: 3D Common Corruptions and Data Augmentation . In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18963–18974 (2022)

19. Kardoost, A., Keuper, M.: Solving minimum cost lifted multicut problems by node agglomeration. In: Asian Conference on Computer Vision. pp. 74–89. Springer (2018)

20. Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan, D., Luiten, J.: SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21357–21366 (2024)

21. Keuper, M., Schmidt, T., Rodriguez-Franco, M., Schamel, W., Brox, T., Burkhardt, H., Ronneberger, O.: Hierarchical markov random fields for mast cell segmentation in electron microscopic recordings. In: 2011 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. pp. 973–978. IEEE (2011)

22. Keuper, M., Schmidt, T., Temerinac-Ott, M., Padeken, J., Heun, P., Ronneberger, O., Brox, T.: Blind deconvolution of widefield fluorescence microscopic data by regularization of the optical transfer function (otf). In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 2179–2186 (2013)

23. Liu, Y., Dong, S., Wang, S., Yin, Y., Yang, Y., Fan, Q., Chen, B.: SLAM3R: Real-time dense scene reconstruction from monocular RGB videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16651–16662 (2025)

24. Matsuki, H., Murai, R., Kelly, P.H.J., Davison, A.J.: Gaussian splatting SLAM. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18039–18048 (2024)

25. Michaelis, C., Contributors: imagecorruptions: Python package to corrupt arbitrary images. GitHub repository, https://github.com/bethgelab/imagecorruptions (2019), accessed 2026-08-03

26. Michaelis, C., Mitzkus, B., Geirhos, R., Rusak, E., Bringmann, O., Ecker, A.S., Bethge, M., Brendel, W.: Benchmarking robustness in object detection: Autonomous driving when winter is coming. arXiv preprint arXiv:1907.07484 (2019). https: //doi.org/10.48550/arXiv.1907.07484

27. Müller, P., Braun, A., Keuper, M.: Classification robustness to common optical aberrations. In: 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). pp. 3634–3645. IEEE (2023)

28. Mur-Artal, R., Tardós, J.D.: ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. IEEE Transactions on Robotics 33(5), 1255–1262 (2017). https://doi.org/10.1109/TRO.2017.2705103

29. Murai, R., Dexheimer, E., Davison, A.J.: MASt3R-SLAM: Real-time dense SLAM with 3d reconstruction priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16695–16705 (2025)

30. Oei, V., Schmalfuss, J., Mehl, L., Bartsch, M., Agnihotri, S., Keuper, M., Bulling, A., Bruhn, A.: RobustSpring: Benchmarking robustness to image corruptions for optical flow, scene flow and stereo. In: International Conference on Learning Representations (2026), https://openreview.net/forum?id=RebPBMrMmk

31. Poggi, N., Agnihotri, S., Keuper, M.: Smart Eyes for Silent Threats: VLMs and In-Context Learning for THz Imaging. arXiv preprint arXiv:2507.15576 (2025)

32. Prokhorov, D., Zhukov, D., Barinova, O., Konushin, A., Vorontsova, A.: Measuring robustness of visual SLAM. In: 2019 16th International Conference on Machine Vision Applications. pp. 1–6 (2019). https://doi.org/10.23919/MVA.2019.8758020

33. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: ORB: An eficient alternative to SIFT or SURF. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 2564–2571 (2011). https://doi.org/10.1109/ICCV.2011.6126544

34. Sommerhof, H., Agnihotri, S., Saleh, M., Moeller, M., Keuper, M., Choubey, B., Kolb, A.: Task driven sensor layouts-joint optimization of pixel layout and network parameters. In: 2024 IEEE International Conference on Computational Photography (ICCP). pp. 1–10. IEEE (2024)

35. Spearman, C.: The proof and measurement of association between two things. The American Journal of Psychology 15(1), 72–101 (1904). https://doi.org/10.2307/ 1412159

36. Stracke, L., Nimmermann, L., Agnihotri, S., Keuper, M., Blanz, V.: Vision at night: Exploring biologically inspired preprocessing for improved robustness via color and contrast transformations. arXiv preprint arXiv:2509.24863 (2025)

37. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for the evaluation of RGB-D SLAM systems. In: Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 573–580 (2012). https://doi.org/10.1109/IROS.2012.6385773

38. Teed, Z., Deng, J.: DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras. In: Advances in Neural Information Processing Systems. vol. 34, pp. 16558–16569 (2021)

39. Teed, Z., Lipson, L., Deng, J.: Deep patch visual odometry. In: Advances in Neural Information Processing Systems. vol. 36, pp. 39033–39051 (2023)

40. Walter, J.P., Agnihotri, S., Keuper, M.: Images as tables: In-context learning with tabPFN for low-data detection of AI-generated images. In: 2nd ICML Workshop on Foundation Models for Structured Data (2026), https://openreview.net/forum? id=6uEcEWSEfs

41. Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., Scherer, S.: TartanAir: A dataset to push the limits of visual SLAM. In: Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 4909–4916 (2020). https://doi.org/10.1109/IROS45743.2020.9341801

42. Wenzel, P., Wang, R., Yang, N., Cheng, Q., Khan, Q., von Stumberg, L., Zeller, N., Cremers, D.: 4Seasons: A cross-season dataset for multi-weather SLAM in autonomous driving. In: Proceedings of the German Conference on Pattern Recog nition (GCPR) (2020)

43. Yan, C., Qu, D., Xu, D., Zhao, B., Wang, Z., Wang, D., Li, X.: GS-SLAM: Dense visual SLAM with 3d gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19595–19604 (2024)

44. Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: GO-SLAM: Global optimization for consistent 3d instant reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3727–3737 (2023)