# Differentiable Inverse Rendering Recovers Intrinsic Properties from Intravascular Optical Coherence Tomography

Ajay Manicka1\*, Karim Kadry1, Sebastián Ruiz-Lopera1,2, Maryam Alsharqi¹, Sanjana Arji3, Samuel Ruipérez-Campillo1,4, Néstor Uribe-Patarroyo1,2, Brett E. Bouma1,2, Farhad Nezami3, Elazer R. Edelman1,3

1Institute of Medical Engineering and Science, MIT, 77 Massachusetts Ave, Cambridge, MA, 02139, USA.

2Wellman Center for Photomedicine, Harvard Medical School and Massachusetts General Hospital, 399 Revolution Dr, Somerville, MA, 02145, USA.

3Brigham and Women's Hospital, Harvard Medical School, 75 Francis St, Boston, MA, 02115, USA.

4Department of Computer Science, ETH Zurich, Rämistrasse 101, Zurich, 8092, Switzerland.

\*Corresponding author(s). E-mail(s): ajay\_m@mit.edu;

Contributing authors: kkadry@mit.edu; srulo@mit.edu; maryam7@mit.edu; sarji@bwh.harvard.edu; sruipere@mit.edu; uribepatarroyo.nestor@mgh.harvard.edu bbouma@mit.edu; frikhtegarnezami@bwh.harvard.edu; ere@mit.edu;

## Abstract

Medical images encode intrinsic tissue properties that indicate disease progression, yet these properties remain entangled with the system-dependent physics of image formation. Differentiable inverse rendering asks whether they are recoverable by inverting the imaging system's forward physics rather than through post-hoc correction. We address this in optical coherence tomography (OCT), focusing on the optical attenuation coefficient, a per-pixel marker of pathology and cellular composition. Conventional recovery relies on closed-form inversions limited to simplified light-tissue models, or learned methods coupled to specific imaging configurations. We instead recover per-pixel attenuation from a single OCT image by gradient descent through a differentiable physics-based forward model. We verified the method against analytical solutions, validated it on five in vitro phantoms, and applied it without modification to 29,269 clinical intravascular OCT images from 128 patients. The inversion needs no training data or segmentations and is portable across platforms and retrospective archives. Replacing the forward model or recovering another optical parameter leaves the optimization procedure unchanged. We demonstrate per-pixel recovery is tractable because gradient information scales to this high-dimensional problem, where gradient-free methods become intractable. Our work supports quantitative tissue characterization from OCT images and provides a framework for extension across imaging modalities and organ systems.

Keywords: Inverse Rendering, Medical Imaging, Differentiable Physics, Tissue Characterization, Visual Computing

## Introduction

The interrogation of biological matter through diverse forms of energy constitutes the foundation of modern medical diagnostic practice. Whether utilizing magnetic gradients in magnetic resonance imaging [1], high-frequency sound waves in ultrasound [2], ionizing radiation in computed tomography [3], or backscattered light in optical coherence tomography (OCT) [4], modern imaging modalities share a common operative principle. They apply energy to tissue and infer anatomy from how the energy is differentially reflected, scattered or absorbed by the underlying material [5]. Historically, the advancement of these technologies has focused on hardware innovations to improve spatial resolution and contrast, providing clinicians with high-fidelity anatomical visualizations of organ systems and diseased tissue [6, 7]. However, while these anatomical maps are essential, they are not direct measurements of tissue composition: each image is a projection of the underlying intrinsic properties through the physics of the imaging system, entangling composition with modality-specific acquisition artifacts. In clinical applications, intrinsic properties such as stiffness, conductivity, and pH dictate disease progression and physiological stability [8-10], yet remain inaccessible from the projected image alone. This paper inverts that projection. Although the imaging chain is a complex, modality-specific forward operator whose inverse is ill-posed, we demonstrate intrinsic tissue composition can be recovered directly from clinical imaging through differentiable inverse rendering, with OCT as our case study

The technical foundation for this inversion comes from parallel development in computer vision. Physics-based forward models have been refined in medical imaging [6], yet they are almost exclusively used in alternate directions. They simulate, correct, or render images rather than recover tissue properties from them. The computer vision community is closing an analogous gap through differentiable rendering [11, 12], which treats the image-formation process as a differentiable function and uses gradient-based optimization to recover scene parameters from observed images [13]. The principle that gradient descent can invert any differentiable forward model to recover its input extends naturally beyond natural images and has begun to enter medical imaging in modality-specific applications such as x-ray and ultrasound [14–17].

In biomedical optical imaging, the optical attenuation coefficient (OAC) is a particularly informative quantitative tissue property, characterizing how rapidly the signal is attenuated with depth. Attenuation serves as a robust discriminator of pathology across both acoustic [18-21] and optical [22-28] regimes, and is sensitive to fundamental cellular mechanisms, including necrosis and apoptosis [29], making it a functional proxy for tissue viability beyond the shape and size of anatomical features captured by standard imaging. While IV-OCT provides high anatomical resolution [30, 31], the image entangles the OAC with the effects of complex wave propagation and system-specific acquisition parameters, making it difficult to extract intrinsic composition from image intensity [32, 33].

Other biomedical optical modalities recover different optical properties. Photoacoustic and diffuse optical methods derive contrast from optical absorption, where quantitative recovery is itself an ill-posed problem because the internal fluence is unknown [34, 35], and intravascular photoacoustic catheters remain preclinical [36]. OCT attenuation is instead dominated by scattering, the mechanism carrying lipid, calcium, and cellular density information in plaque [37, 38], and is well established as a quantitative marker across many tissue types [39–43]. Because IV-OCT is already standard of care during percutaneous coronary intervention [44, 45], inverting its image formation extracts an additional quantitative channel from images that are already acquired. OCT is a case study here as the framework requires only that the forward operator be differentiable with respect to the property of interest.

Existing approaches to OCT attenuation recovery fall into three categories [46]. Curve-fitting (CF) methods extract a single attenuation coefficient by fitting signal decay over a manually segmented region [47], which is robust for uniform tissues but unable to capture intra-tissue heterogeneity. Depth-resolved (DR) methods compute per-pixel attenuation from the ratio of local intensity to remaining energy [48, 49], but the closed-form formulation constrains them to single-scattering assumptions and makes the resulting maps sensitive to noise sources like speckle and local signal artifacts. Recent work in this area addresses the noise sensitivity issue using despeckling to improve local region homogeneity [50]. Learning-based methods can incorporate richer forward physics through synthetic training data [51], but the learned amortized inverse is fixed after training and coupled to the simulated distribution. The stationary iteration method estimates depth-dependent attenuation and backscattering without explicit layer segmentation [52], and optimization also estimates these parameters via an implicit neural representation of optical properties through an OCT imaging model using multi-angle measurements [53]. Our method instead optimizes any differentiable forward model by gradient descent from a single B-scan.

Our work targets a central question in differentiable inverse rendering of whether intrinsic tissue properties can be inferred directly from medical images by framing property recovery as the inverse of the imaging system's forward physics. To examine this in OCT, we built an automated and generalizable physics-based inverse rendering framework that extracts intrinsic tissue properties directly from standard clinical OCT images without the need for segmentations. The framework addresses all three limitations in a single formulation by delivering per-pixel attenuation while operating on an explicit forward model that can incorporate arbitrary physics and optimizing each image independently. Because the forward model is modular, incorporating additional scattering physics, such as the Extended Huygens-Fresnel model [54], or recovering further optical parameters requires only modifying the forward model, not the inversion procedure.

By inverting the path-integrated Beer-Lambert (B-L) model [55], we recovered depth-resolved, pixel-level OAC maps. We verified our method on five optically homogeneous physical phantoms against the conventional CF method and applied it to 29,269 clinical coronary IV-OCT frames [56–59] (over a 128-patient intravascular lithotripsy cohort) to demonstrate spatially resolved attenuation recovery and faithful image rendering across varying vessel morphologies. IV-OCT routinely guides percutaneous coronary intervention. Images are acquired with a catheter-based scanner that illuminates the arterial wall with a near-infrared laser and reconstructs depth-resolved cross-sections from the backscattered light. Current clinical practice relies on viewing anatomical features such as plaque shape and size, while treatment decisions depend on tissue composition. These results support the potential of per-pixel OAC maps to enable quantitative plaque characterization and guide interventional decision-making [29, 60].

The differentiable inverse rendering approach generalizes to any imaging modality whose forward physics can be expressed as a differentiable operator. Changing the forward physics can change the properties that are recovered. We can also control the optimization loop, loss function, and regularization framework to guide optimization. Our work demonstrates this paradigm in OCT, joining differentiable forward models already demonstrated for other modalities across physics-based medical imaging [61, 62]. We make the following contributions:

• A gradient-based differentiable inverse rendering framework for per-pixel optical attenuation recovery from OCT images without training data or prior segmentation.

• Quantitative validation against curve-fitting references across 5,120 B-scan inversions on 5 physical liquid intralipid phantoms each imaged at two focal positions, with systematic characterization of optimizer reproducibility, noise robustness, and parameter sensitivity in agar phantoms.

• Retrospective application to 128 clinical IV-OCT coronary pullbacks of atherosclerotic tissue demonstrating spatially resolved attenuation and backscattering recovery.

Differentiable inverse rendering posits that intrinsic property recovery from images can be cast as a tractable inverse problem driven by gradient information. Our work contributes evidence for this paradigm in biomedical imaging, with OCT as the use case. We introduce a differentiable inverse rendering framework for OCT with validation across phantoms and demonstration in clinical acquisitions. Gradient information is vital because it captures the intrinsic relationships between tissue regions and guides the optimizer toward a physically consistent solution. The inversion is portable across different imaging systems and applicable to retrospective clinical archives, addressing the domain-shift barrier in clinical AI deployment. Gradient descent through an explicit forward model decouples the complexity of the physics from the availability of an inverse, whereas closed-form estimators require physics simple enough to invert analytically and learned estimators are bound to their training distribution, making the forward model a replaceable component and the approach transferable beyond OCT.

## Results

Our method takes a single OCT image as input along with prespecified scanner and optimization parameters. It returns a per-pixel map of the optical attenuation coefficient µ. An OCT image is composed of depth profiles called A-lines; a two-dimensional cross-section (B-scan) is formed by laterally concatenating adjacent A-lines. The framework operates by passing a candidate attenuation map through a differentiable forward model that simulates a simplified OCT image by combining Beer-Lambert attenuation with confocal amplitude modulation, then iteratively updating µ by gradient descent to minimize the discrepancy between the simulated and measured B-scan. We first verify the forward and inverse components on synthetic data before applying the framework to physical phantoms and clinical intravascular OCT images.

## Forward and inverse models achieve exact recovery on computational phantoms

We compared four optimizers while increasing the number of axially coupled attenuation parameters within one synthetic A-line from 25 to 5,000 across eight problem sizes (Figure 1). All methods shared the forward model, bounds, and eight paired initializations, with solver-specific optimization budgets. All achieved below 0.1% mean MAPE at the smallest size. While Adam and L-BFGS-B maintained this accuracy throughout, Random Search and CMA-ES became worse reaching MAPE values of 30.32% and 100.91% respectively. Supplementary experiments examined lateral scaling on synthetic data, measured-data recovery, and A-line coupling (Supplementary Figure 1). At 475 parameters on the measured A-line with axial TV, the gradient-based methods achieved lower intensity MSE and closer CF agreement. With fixed gain and no lateral TV, each A-line can be optimized independently using the mean squared error between predicted and measured intensities and the axial TV regularizer that couples neighboring pixels in a single A-line together by penalizing differences in attenuation. Lateral TV couples neighboring A-lines together within a B-scan. Estimating one gain from the entire B-scan also couples the A-lines because each A-line contributes to the shared gain. We therefore optimize all A-lines in a B-scan together in the agar-phantom and clinical inversions.

The differentiable forward model was first verified against the analytical Beer-Lambert solution. Five hundred synthetic A-lines, each containing 2 to 5 homogeneous layers with randomized attenuation coefficients spanning $[ 0 . 1 , 1 5 . 0 ] ~ \mathrm { m m } ^ { - 1 }$ over a depth range of $0 . 4 \mathrm { - } 0 . 8$ mm (100 depth pixels at $\Delta z = 4$ µm; Figure $\mathrm { 2 g ) }$ , were passed through the differentiable forward model to produce the synthetic OCT B-scan (Figure 2h) and through the analytical Beer-Lambert solution to produce the target OCT B-scan. Representative A-line profiles for $2 \mathrm { - } , 3 \mathrm { - }$ , and 4-layer configurations confirm numerical agreement (Figure 2a-f). Across all 50,000 pixels, the residuals were five orders of magnitude below the lowest signal $( R ^ { 2 } = 1 . 0 0 0 0 , \mathrm { M A E } = 5 . 0 { \times } 1 0 ^ { - 6 } \mathrm { d B }$ , max erro $\mathrm { r } = 6 . 1 \times 1 0 ^ { - 5 } \ : \mathrm { d B }$ Figure 2i), indicating virtually exact agreement.

The inverse optimization was then evaluated on 500 synthetic B-scans generated by the forward model from randomized ground-truth attenuation maps, testing whether the optimizer can recover known attenuation maps from its own rendered image. We show one example B-scan optimization result $( \mathrm { F i g u r e ~ 2 j - o } )$ , concordance scatter plots at four optimization snapshots across all 500 phantoms $( \mathrm { F i g u r e ~ 2 p - s } )$ , and the full convergence trajectory of MAPE and $R ^ { 2 }$ over 30,000 iterations (Figure 2t). Each attenuation map was initialized from scrambled conditions $( R ^ { 2 } < 0 . 0 0 0 1 , \mathrm { M A P E = 7 7 . 0 \% ; F i g u r e 2 j , p ) }$ , and the optimizer converged monotonically to near-perfect recovery (Figure 2t). Across all 500 phantoms, the recovered tissue $\mu$ achieved final $\mathrm { M A P E } = 0 . 1 5 \%$ concordance $R ^ { 2 } = 0 . 9 9 9 7 , \mathrm { M A E } = 0 . 0 0 7 \pm 0 . 0 0 1 \ \mathrm { m m ^ { - 1 } }$ , and regression slope = 0.998. These results confirm the numerical correctness of both the forward operator and the optimization procedure, establishing a baseline for evaluation on physical phantoms. Recovery under forward-model mismatch is evaluated in Supplementary Note 7.

## Recovered attenuation tracks scatterer concentration across physical phantoms

Physical phantoms provide a controlled setting in which the imaging system parameters are fully specified and reference attenuation values are available from conventional curve-fitting analysis, enabling direct quantitative comparison with the optimization-based recovery. All 512 B-scans per acquisition were processed independently using the single-scattering forward model, totaling 5,120 B-scan inversions across five liquid intralipid phantoms $( 0 . 5 \mathrm { - } 3 . 0 \% )$ each imaged at two focal positions (H and L; Figure 3a-c). The optimization workflow on a representative 5.0% intralipid-in-agar B-scan is shown in Supplementary Figure 2d-i, with the recovered $\mu$ volume for the 3.0% H liquid acquisition (Figure 3d,e) demonstrating spatially coherent attenuation within the tissue band.

Recovered attenuation coefficients, evaluated within matched CF windows, increased monotonically with intralipid concentration across all 10 acquisitions (Figure 3f). Frame-to-frame variability was low (standard deviation $0 . 0 2 5 { - } 0 . 0 5 3 ~ \mathrm { \ m m ^ { - 1 } }$ across 512 B-scans per volume). Each concentration was imaged at two focal positions; H and L yielded slopes of 0.663 and 0.629 $\mathrm { m m } ^ { - 1 \% }$ with $R ^ { 2 } = 0 . 9 9 0$ and $R ^ { 2 } = 0 . 9 9 3$ , comparable to the CF reference (slopes of 0.623 and $0 . 6 2 5 \mathrm { ~ m m } ^ { - 1 } \% )$ . The framework recovers a full $\mathrm { p e r - p i x e l ~ O A C }$ map across the tissue volume (Figure 3e), whereas CF collapses each spatially averaged measurement location to a single slope value. Absolute values were $0 . 0 8 3 { - } 0 . 1 7 6 ~ \mathrm { m m ^ { - 1 } }$ below CF, but the linear relationship and relative ordering are preserved. Example A-lines (Figure 3g) show lower intensity RMSE for DiffOCT than for CF (0.29 and 0.52 versus 0.95 and 1.76 dB), although attenuation agreement remains conditional on the shared CF-derived gain.

## Attenuation recovery is reproducible across initializations and robust to noise

The identifiability study quantified the reproducibility of the recovered attenuation under two distinct perturbations of the initialization: the random draw of the initial $\mu$ map from a fixed distribution (seed sweep), and the width and location of the initialization distribution itself (range sweep) (Figure 4a-d). In the seed sweep, each acquisition was optimized 20 times with independent initializations drawn from a fixed uniform distribution (Figure 4a). Intra-acquisition coefficients of variation (CoV) ranged from 0.18% to 0.32% across the 10 acquisitions $( \mathrm { m e a n } = 0 . 2 7 \pm 0 . 0 4 \%$ ; Figure 4c), indicating that, given a fixed initialization distribution, the optimizer yields reproducible tissue-mean attenuation.

In the range sweep, 10 distinct ranges for the uniform distribution were tested with 5 seeds per condition (Figure 4b), spanning narrow $( [ 3 , 7 ] \ \mathrm { m m ^ { - 1 } } )$ , wide $( [ 0 , 1 0 ] , \ [ 0 , 2 0 ] \ \mathrm { \ m m ^ { - 1 } } )$ , fixed-value ([1, 1], [5, 5], [10, 10], $[ 1 5 , 1 5 ] \mathrm { m m } ^ { - 1 } )$ , biased $\left( \left[ 7 , 1 5 \right] \mathrm { m m } ^ { - 1 } \right)$ , and informed (CF reference) conditions. Seven of the 10 conditions yielded consistent recovery across all 10 acquisitions: after averaging the five seeds per condition, every condition mean was within ±25% of the corresponding acquisition's mean across these seven conditions. The corresponding per-acquisition initialization range CoV was 2.1%-13.1% (Figure 4c,d). The three conditions [0, 3], [1, 1], and $\bar { [ 1 5 , 1 5 ] } \ \mathrm { m m ^ { - 1 } }$ failed this criterion and yielded different recovered values, consistent with the non-convexity expected from the cumulative-sum forward operator. This criterion describes numerical consistency rather than accuracy, since pixel-wise ground truth is unavailable for measured data. Initialization-induced uncertainty in the clinical maps is quantified in Supplementary Note 8.

The noise robustness study quantified how additive Gaussian noise degrades recovery across all 10 acquisitions over a sweep from $\mathrm { S N R } = - 3$ dB up to each phantom's baseline acquisition SNR in 1 dB steps (Figure 4e- $\mathrm { - g ) }$ . Following the noise model in OCT [63], zero-mean circular Gaussian noise was added to each complex-valued B-scan at a controlled SNR computed using the rigorous definition that accounts for the noise floor bias in the numerator. For every phantom, three recovery metrics tracked SNR consistently. Recovered tissue $\mu$ deviated only marginally from the noiseless baseline over the first 4-5 dB of added noise and then diverged sharply as SNR fell further (Figure $_ \mathrm { 4 g ) }$ . Baseline concordance $R ^ { 2 }$ (recovered $\mu$ versus its noiseless recovery) increased monotonically with SNR across every phantom. The spatial standard deviation of recovered tissue $\mu$ decreased monotonically with SNR, reflecting smoother and more spatially consistent recoveries at high SNR.

## Recovered attenuation maps generalize to clinical intravascular OCT

To demonstrate the method transfers from phantoms to clinical acquisition, the framework was applied to intravascular OCT with the same optimization algorithm and reconstruction loss, without TV regularization. Per-frame optimization was run independently on 29,269 cross-sections spanning 128 separate patient pullbacks drawn from four clinical trial cohorts [56-59]; each Cartesian frame was converted to a polar representation for optimization. Each frame is solved independently, as the method carries no state or learned parameters between frames. Figure 5 shows 5 example frames from a single representative pullback. Because scanner-specific calibration metadata was unavailable for this acquisition, we adopted measured confocal parameters reported for the C7-XR / Dragonfly catheter family $( z _ { R } = 0 . 5 9 \ : \mathrm { m m } , z _ { f } = 1 . 0 6 \ : \mathrm { m m } )$ [64], together with an axial pixel size of 0.01932 mm/pixel estimated from the ILUMIEN OPTIS A-scan range. Combined with the undocumented post-processing transformations applied to the log-compressed clinical images, the recovered attenuation values capture relative tissue contrast rather than absolute quantities

The per-pixel attenuation coefficient $\mu$ was first optimized with the backscattering fraction α held fixed at unity (Stage 1) assuming a 15 dB noise floor. Across every frame of every patient, the recovered OAC maps are spatially resolved and track the tissue architecture visible in the raw images, resolving the vessel wall, lumen boundary, and calcified plaque deposits, confirming that the framework generalizes from controlled phantom to the complex radial scan of catheter-based IV-OCT. Using the recovered $\mu$ maps, the forward model synthesizes rendered IV-OCT images that reproduce the overall geometry of each cross-section. Aggregated over all 29,269 frames, rendered and target B-scans agree closely on tissue coverage $( \mathrm { D i c e } = 0 . 9 5 7 \pm 0 . 0 0 7 )$ , and pixel-level rendering quality reaches a mean PSNR of $1 4 . 8 3 \pm 0 . 9 9$ dB (Equation (13)).

The Stage 1 residual maps (right column of Figure 5) reveal structured, non-random differences concentrated along the inner wall. These arise because µ plays a dual role in the Beer-Lambert forward model as in Equation (5): it controls per-pixel brightness and cumulative attenuation, so the optimizer must satisfy both simultaneously with a single variable. Introducing a per-pixel backscattering parameter α (Stage 2) decouples these roles: $\mu$ is frozen and $\alpha \in [ 1 0 ^ { - 1 0 } , 1 0 ^ { - 2 } ]$ is optimized as an independent brightness term. This reduces residuals across the cohort and raises the mean PSNR to $2 2 . 6 3 \pm 0 . 7 4 \mathrm { d B } ( \mathrm { D i c e } = 0 . 9 5 4 \pm 0 . 0 0 9 ;$ Figure $5 ,$ bottom row) without altering the recovered attenuation structure. A detailed derivation of the brightness-attenuation coupling and the two-stage procedure is provided in Supplementary Note $6 .$ The recovered maps carry information that the standalone IV-OCT image does not. Pixel brightness is not a tissue property. It confounds local backscattering with the cumulative attenuation of every voxel already traversed, the confocal gate, and an arbitrary system gain such that identical tissue appears at different intensities depending on depth and position relative to the catheter. Inverting the forward model removes these confounders so equal $\mu$ denotes equal optical property anywhere in the frame.

No pixel-level ground-truth attenuation exists for in vivo coronary tissue, so this experiment demonstrates that the method transfers unmodified to clinical acquisition geometry rather than validating recovered attenuation values. PSNR and Dice quantify forward-model fidelity, not the accuracy of $\mu ,$ which is demonstrated with the computational and physical phantoms.

## Discussion

We investigated whether intrinsic tissue properties can be recovered directly from medical images by recasting property extraction as an inverse problem that explicitly accounts for the system-dependent effects of image formation rather than a post-processing task. We tested this hypothesis in OCT by developing a differentiable inverse rendering framework that recovers the optical attenuation coefficient (OAC) per pixel from a single OCT image, without training data or prior segmentation. By embedding the Beer-Lambert model as a differentiable forward operator and optimizing µ directly via gradient descent, the framework produces per-pixel parameters with direct physical meaning and an optimization procedure decoupled from the imaging configuration.

The optimizer comparison supports gradient-based attenuation recovery as the number of axially coupled parameters increases (Figure 1). All four methods achieve low error at small problem sizes, but the gradient-free algorithms lose accuracy as dimensionality grows. Random Search runs only a forward pass, so its computational cost stays flat, but its error grows steadily with problem size. CMA-ES fails on both fronts. Its error also grows with dimensionality, and each evaluation updates a covariance matrix whose size grows as the square of the number of parameters, making it computationally prohibitive at large sizes. Adam adds only a backward pass to compute gradients, while L-BFGS-B additionally maintains a short history of recent gradient and position updates to build its quasi-Newton search direction. Only the gradient-based methods remain viable across the full scaling range, underscoring the necessity of gradient information for high-dimensional inverse problems in imaging.

We then verified numerical correctness on synthetic phantoms (Figure 2), and we examined 5,120 inversions on liquid intralipid phantoms and found that recovered OAC tracks scatterer concentration linearly $( R ^ { 2 } =$ 0.990–0.993) with slopes of $0 . 6 2 9 { - } 0 . 6 6 3 ~ \mathrm { m m ^ { - 1 } \% }$ under CF-derived gain calibration (Figure 3), and that agar-phantom recovery is reproducible across initializations $( \mathrm { C o V } ~ = ~ 0 . 1 8 \% - 0 . 3 2 \%$ Figure 4). The noise robustness study confirmed that recovery degrades predictably with decreasing SNR: across every phantom, baseline concordance and per-pixel spatial consistency tracked SNR monotonically, while recovered $\mu$ deviated only marginally from its noiseless value over the first few dB of added noise and diverged sharply only once noise approached the signal amplitude. The physics-based forward model therefore acts as a stable regularizer across realistic acquisition SNRs and loses this regularization only when the signal itself is lost. The OCT inverse problem is fundamentally ill-posed, meaning that many distinct attenuation maps can produce nearly identical observed images. Our reproducible recovery across initializations indicates that the physics-based forward model yields consistent tissue-mean attenuation within the initialization ranges examined, and agreement with CF in liquid intralipid phantoms (Figure 3) supports the accuracy of tissue-mean recovery.

Applying the same algorithm to clinical coronary IV-OCT without modification recovered spatially resolved OAC maps (Figure 5), demonstrating that physics-based recovery transfers across acquisition geometries in a data-agnostic way that learned methods cannot achieve without retraining. The recovered maps produce synthesized images that match the acquired B-scans with a mean PSNR of $1 4 . 8 3 \pm 0 . 9 9 ~ \mathrm { d B }$ , a modest value explained by the dual role that the OAC in the Beer-Lambert forward model plays in both per-pixel brightness and cumulative attenuation along depth. The optimizer must find a single µ map that simultaneously satisfies both constraints. Introducing a per-pixel backscattering parameter α decouples these roles and raises the mean PSNR to 22.63 ± 0.74 dB (Supplementary Note 6) without altering the recovered attenuation structure. This sequential procedure, constrained by literature-informed bounds on $\mu$ and α [46, 50], reduces but does not remove the ambiguity between attenuation and backscattering in a single intensity image. The modest µ-only PSNR therefore reflects the under-specification of the single-parameter forward model rather than a failure of the inversion procedure. Despite lack of absolute calibration, the recovered maps preserve spatial contrast consistent with anatomical features, which is the primary signal used in clinical decision-making

The ability to extract a well-established tissue biomarker [38, 47] directly from routine IV-OCT imaging has direct implications for interventional cardiology, where treatment decisions hinge on tissue composition that is often inferred indirectly from anatomical features observed in imaging. Attenuation-based plaque assessment has already reached commercial deployment in the form of the OCT-derived index of plaque attenuation, which discriminates high-risk plaque features in vivo against expert consensus OCT classification [65], and this supports that the underlying signal carries clinical meaning. Such systems are, however, closed and vendor-bound pipelines that reduce a region to a single aggregate index, whereas the framework presented here recovers attenuation per pixel through an explicit forward model that can be substituted to recover other optical properties without altering the inversion. Per-pixel OAC maps could serve as a quantitative substrate for decisions that depend on spatial contrast, including vulnerable-plaque identification, calcified-region delineation for lithotripsy or atherectomy planning, and objective monitoring of plaque evolution across serial acquisitions of the same patient. Because the inversion depends on forward physics rather than a learned prior tied to specific scanners and samples, such analyses can be applied retrospectively to existing IV-OCT archives. These applications are most directly enabled once absolute calibration is available; the results here demonstrate the spatially resolved intrinsic properties and the relative contrast that would underpin them.

We observed a systematic underestimation of recovered attenuation relative to curve-fitting references, but the difference was small at every concentration $( 0 . 0 8 3 - 0 . 1 7 6 ~ \mathrm { m m ^ { - 1 } } )$ , and concentration slopes were close to CF under shared gain calibration (Figure 3f). Both methods share the homogeneous Beer-Lambert limit, but CF fits a single attenuation to each depth window, acting as a first-order approximation of the signal decay, while DiffOCT fits an attenuation value at every depth pixel under TV regularization. The DiffOCT prediction therefore tracks the measured signal closely, whereas the CF line behaves as a regression line through it, with lower intensity RMSE for DiffOCT in both example A-lines (Figure 3g). The small offset is consistent with this difference in model order. In these homogeneous samples, part of the local variation that DiffOCT follows reflects residual speckle and noise, so lower RMSE indicates closer signal reproduction rather than more accurate attenuation by itself. DiffOCT thus reproduces the CF attenuation to within $0 . 2 ~ \mathrm { m m ^ { - 1 } }$ while additionally resolving attenuation with depth, which a single slope cannot represent in heterogeneous tissue. Because the loss mask is fixed from measured intensities, the offset does not arise from predictions falling below the noise floor and being excluded from the loss. Such systematic offsets are pervasive in the existing OAC retrieval literature. Reported values for calcified plaque alone span an order of magnitude depending on the estimation method, from $1 1 . 1 \pm 4 . 9 ~ \mathrm { m m } ^ { - 1 }$ by curve fitting on averaged A-scans [47] to $0 . 8 7 \pm 0 . 2 4 ~ \mathrm { m m } ^ { - 1 }$ by depth-resolved analysis [49], yet the relative ordering of tissue types is preserved within each study [46]. Tissue differentiation depends on the relative ordering of the OAC across tissues rather than on absolute magnitudes, and our recovered OAC preserves this ordering through a linear relationship to intralipid concentration across the phantom series. This has important implications for clinical settings. When scanner-specific calibration metadata is unavailable, recovered OAC values are no longer absolute but retain the within-slice relative contrast needed for tissue discrimination. Methods to convert relative OAC into absolute cross-patient values, such as reference-tissue anchoring or self-calibrating forward models, will be vital for extending the framework to cases requiring population-level comparisons such as drug-trial endpoints.

Solving this inversion at clinically relevant scales is tractable only because of the structural advantage of gradient information. Each backward pass delivers one derivative per parameter, while gradient-free methods update based on a scalar loss whose information content does not scale with problem size (Figure 1), which explains the convergence failure in gradient-free approaches with increasing problem size. Beyond this computational tractability, physics-based differentiable inverse rendering carries two deployment advantages over analytical and data-driven methods that are particularly relevant for digital health. First, the framework is fully modular. The forward model, optimizer, and regularization terms can each be modified independently without restructuring the inversion procedure. This flexibility is unavailable to closed-form analytical approaches, which require the forward physics to admit an analytical inverse, and to learning-based approaches, which are coupled to the physics of their training distribution. Second, because the inversion is agnostic to acquisition configuration, it naturally sidesteps the scanner-to-scanner domain shift that is the dominant barrier to clinical deployment, allowing a single implementation to serve heterogeneous clinical environments regardless of scanner vendor, acquisition protocol, or post-processing pipeline. As physics-based differentiable models become increasingly common across medical imaging, including ultrasound [15, 17], X-ray [14], and MRI [16], their shared portability and interpretability provide a foundation for quantitative imaging tools that can scale across the infrastructure of digital health.

However, while our framework can recover per-pixel OAC from clinical OCT images without retraining, it has several limitations. First, full optimization is computationally costly (approximately 35-50 s per B-scan on a modern GPU; Supplementary Note 4), which is tractable for offline research but incompatible with real-time catheter-lab use. Second, the clinical demonstration was conducted without access to scanner-specific system parameters or post-processing metadata, so the recovered values should be interpreted as evidence of geometric transfer and relative attenuation extraction rather than population-level clinical validation of absolute OAC values. This can be overcome in future developments by having access to the raw data and optical specifications of the scanner. Related to this, no reference standard for per-pixel attenuation exists in vivo, since histology is destructive and cannot be registered to individual OCT pixels. Third, our forward model is a simplified intensity-domain formulation that omits phase, coherent interference, and the speckle realization that arises from them. We adopt this simplification deliberately because phase-resolved simulation requires scanner-specific spectral parameters that are unavailable for retrospective clinical data. Speckle is nonetheless pervasive in IV-OCT and, because $\mu$ is fitted at every pixel, residual speckle can appear as local attenuation fluctuations (Figure 3g). In the phantom experiments, we mitigate this using Gaussian spatial filtering applied to input B-scans, which other despeckling methods can replace, and an anisotropic total-variation penalty (Eq. (8)) that encourages spatial homogeneity within tissue regions. Fourth, tissue-mean recovery is reproducible across seeds $( \mathrm { C o V = 0 . 1 8 \% - 0 . 3 2 \% } )$ and the seven selected initialization conditions (CoV $= 2 . 1 \% - 1 3 . 1 \%$ Figure 4c-d), but more aggressive perturbations of the initialization range produce different recovered values. Initialization-induced uncertainty in the clinical tissue means and individual pixels is quantified in Supplementary Note 8. Lastly, the single-scattering Beer-Lambert forward model may become insufficient at higher optical depths and in cases where multiple scattering dominates, although preliminary comparisons with the Extended Huygens-Fresnel model $( g = 0 . 7 $ , a value previously used for tissue in OCT simulations [66]) yielded nearly identical recovered attenuation at our phantom concentrations (mean difference $< 1 \% ;$ Supplementary Note 5).

This work highlights several directions for extension. Posterior sampling and Bayesian inverse-problem formulations can attach per-pixel uncertainty bounds to recovered OAC and characterize the local-minima behavior we observed under aggressive initialization perturbation. More thorough treatment of multiple-scattering forward models can extend quantitative recovery into deeper tissue regimes, building on the Extended Huygens-Fresnel implementation reported in Supplementary Note 5. Joint estimation of system parameters such as $z _ { f }$ and $z _ { R }$ with $\mu$ is a natural extension of the framework and should be feasible with constraints that make these parameters identifiable.

This framework permits quantitative tissue characterization from routine clinical intravascular imaging using physics-based differentiable inverse rendering. By recovering physically interpretable attenuation maps from a single IV-OCT image without training data or scanner calibration, it can support interventional cardiologists in plaque characterization and treatment planning, and enable retrospective analysis of existing IV-OCT archives.

## Methods

## Framework Overview

The framework recovers per-pixel optical properties from a measured OCT B-scan by iteratively optimizing a differentiable forward model to reproduce the observed image (Fig. 6). Given a log-scaled target B-scan $I _ { \mathrm { t a r g e t } } ( z , x )$ , where z denotes depth and x the lateral A-line index, the method initializes a spatial map of tissue attenuation coefficients $\mu ( z , x )$ and passes it through a physics-based forward model $\mathcal { F }$ that simulates OCT image formation, producing a predicted B-scan $\hat { I } ( z , x ) = \mathcal { F } ( \mu ; \pmb \theta )$ , where θ denotes the fixed system parameters (focal geometry, pixel size, noise floor, and system gain). A masked mean squared error (MSE) loss, with total variation (TV) regularization on $\mu$ for spatial smoothness, is computed between $\hat { I }$ and $I _ { \mathrm { t a r g e t } }$ over tissue pixels identified by a noise floor mask. Because the forward model is implemented as a differentiable computational graph in $\mathrm { P y ' }$ Torch [67], the gradient $\partial \mathcal { L } / \partial \mu$ is obtained via automatic differentiation, and $\mu$ is updated by the Adam optimizer. After convergence, the final $\mu ( z , x )$ map represents the recovered depth-resolved optical attenuation coefficient at each pixel.

The forward model $\mathcal { F }$ is modular: in this work we implement a single-scattering Beer-Lambert model, but the optimization procedure is agnostic to the specific forward physics and requires only that the model is differentiable with respect to $\mu .$ Pre-processing steps applied before optimization include subtraction of the noise floor in linear intensity scale and automatic initialization of the system gain offset by median-matching between predicted and target tissue intensities.

## Differentiable Forward Model

The detected OCT intensity at depth z can be modeled as the product of the local backscattering amplitude, the round-trip attenuation along the beam path, and depth-dependent sensitivity and confocal functions corrupted by an additive noise floor [48, 50, 68]:

$$
I ( z ) = G \cdot \alpha ( z ) \cdot \mu ( z ) \cdot \exp \Bigl ( - 2 \int _ { 0 } ^ { z } \mu ( u ) d u \Bigr ) \cdot S ( z ) \cdot H ( z ) + N ( z ) ,\tag{1}
$$

where $G$ is the system gain, representing the incident irradiance on the sample scaled by detection efficiency, $\alpha ( z )$ is the unitless local backscattering fraction [50], written as $\alpha ( z , x )$ for a B-scan, $\mu ( z )$ is the attenuation coefficient $[ \mathrm { m } \mathrm { m } ^ { - 1 } ]$ , the exponential term describes round-trip Beer-Lambert attenuation from the surface to depth $z , S ( z )$ is the dimensionless sensitivity fall-off function in linear intensity scale, and $N ( z )$ is the estimated mean depth-dependent noise-floor intensity. The confocal gate $H ( z )$ models the depth-dependent collection efficiency of the focusing optics [68]:

$$
H ( z ) = \frac { 1 } { \left( \displaystyle \frac { z - z _ { f } } { z _ { R } } \right) ^ { 2 } + 1 } ,\tag{2}
$$

where $z _ { f }$ is the focal plane position and $z _ { R }$ is the Rayleigh range of the sample arm beam. Sensitivity fall-off can occur in both spectral-domain OCT, due to finite spectrometer resolution, and swept-source OCT, due to the finite instantaneous linewidth of the swept laser [69]. The swept-source system used for the phantom studies (based on a Thorlabs SL131090 light source) had negligible sensitivity fall-off over the imaging range spanned by the phantom signal, which was placed close to the zero-path delay. We used $S ( z ) = 1$ for the agar-phantom inversions, applied the measured $S ( z )$ during preprocessing for the liquid phantoms, and assumed $S ( z ) = 1$ for the clinical data because scanner-specific sensitivity calibration was unavailable.

Prior to optimization, the noise floor $N ( z )$ is subtracted and the sensitivity correction is applied in linear intensity scale [50]:

$$
I _ { \mathrm { c o r r } } ( z ) = \frac { I ( z ) - N ( z ) } { S ( z ) } = G \cdot \alpha ( z ) \cdot \mu ( z ) \cdot \exp \Bigl ( - 2 \int _ { 0 } ^ { z } \mu ( u ) d u \Bigr ) \cdot H ( z ) .\tag{3}
$$

This expression describes the mean corrected intensity where individual measurements retain residual noise fluctuations that can affect the recovered $\mu .$ Taking $1 0 \log _ { 1 0 } ( \cdot )$ of both sides converts the multiplicative forward model into an additive expression in log scale:

$$
I _ { \mathrm { d B } } ( z ) = G _ { \mathrm { d B } } + 1 0 \cdot \log _ { 1 0 } \alpha ( z ) + 1 0 \cdot \log _ { 1 0 } \mu ( z ) + H _ { \mathrm { d B } } ( z ) - { \frac { 2 0 } { \ln 1 0 } } \cdot \int _ { 0 } ^ { z } \mu ( u ) d u ,\tag{4}
$$

where $G _ { \mathrm { d B } } = 1 0 \cdot \log _ { 1 0 } G$ is the system gain [dB] and $H _ { \mathrm { d B } } ( z ) = 1 0 \cdot \log _ { 1 0 } H ( z )$ is the confocal function correction [dB]. In discrete form, the integral is replaced by a cumulative sum over depth pixels with axial spacing $\Delta z ,$ where pixel i accumulates attenuation from all pixels above it but not itself:

$$
I _ { \mathrm { d B } } ( z _ { i } ) = G _ { \mathrm { d B } } + 1 0 \cdot \log _ { 1 0 } \alpha ( z _ { i } ) + 1 0 \cdot \log _ { 1 0 } \mu ( z _ { i } ) + H _ { \mathrm { d B } } ( z _ { i } ) - \frac { 2 0 } { \ln 1 0 } \cdot \sum _ { j = 0 } ^ { i - 1 } \mu ( z _ { j } ) \cdot \Delta z .\tag{5}
$$

This log-domain formulation is implemented as a differentiable $\mathrm { P y }$ Torch module in which $\mu ( z , x )$ is the optimizable parameter tensor and, for measured-data attenuation recovery, $\alpha ( z , x ) = 1$ at every tissue pixel, so that $1 0 \cdot \log _ { 1 0 } \alpha = 0$ in $\mathrm { E q . ~ ( 5 ) }$ and α drops out of the optimization. Because every operation in $\mathrm { E q . ~ ( 5 ) }$ is differentiable, the gradient $\partial I _ { \mathrm { d B } } / \partial \mu$ is computed exactly via automatic differentiation and propagated to the optimizer $\left( \mathrm { F i g . ~ 6 } \right)$ . All $\mu$ values are updated simultaneously. Here, differentiation through the cumulative sum makes the gradient from the intensity error with respect to $\mu ( z _ { j } )$ combine its local contribution with contributions from deeper pixels included in the loss. In the clinical second stage, $\mu$ is frozen and $\alpha ( z , x )$ is optimized per pixel (Supplementary Note 6).

## Inverse Problem Formulation

Given a noise-floor-corrected target B-scan $I _ { \mathrm { t a r g e t } } ( z , x )$ in log scale, we seek the attenuation map that minimizes the reconstruction error over tissue pixels:

$$
\mu ^ { * } ( z , x ) = \arg \operatorname* { m i n } _ { \mu } \ \frac { 1 } { | \mathcal { M } | } \sum _ { ( z , x ) \in \mathcal { M } } \left[ [ \mathcal { F } ( \mu ; \pmb { \theta } ) ] ( z , x ) - I _ { \mathrm { t a r g e t } } ( z , x ) \right] ^ { 2 } ,\tag{6}
$$

where $\mathcal { F } ( \mu ; \pmb \theta )$ is the forward model prediction (Eq. (5)), θ denotes the fixed system parameters, and $\mathcal { M }$ is the set of tissue pixels whose measured intensity exceeds the noise floor threshold, fixed before optimization. For the agar-phantom and clinical inversions, pixels below this threshold are excluded from both the loss computation and the cumulative attenuation sum in the forward model, preventing noise-dominated regions from influencing the recovered attenuation. To promote spatial smoothness in the recovered attenuation map, the loss is augmented with an anisotropic total variation (TV) penalty [51]:

$$
\mathrm { T V } ( \mu ) = \frac { 1 } { \left| \mathcal { M } _ { z } \right| } \sum _ { ( z , x ) \in \mathcal { M } _ { z } } \left| \mu ( z + 1 , x ) - \mu ( z , x ) \right| + \frac { 1 } { \left| \mathcal { M } _ { x } \right| } \sum _ { ( z , x ) \in \mathcal { M } _ { x } } \left| \mu ( z , x + 1 ) - \mu ( z , x ) \right| ,\tag{7}
$$

where $\mathcal { M } _ { z }$ and $\mathcal { M } _ { x }$ are the sets of adjacent tissue pixel pairs along the depth and lateral axes, respectively. The total loss is:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } ( \mu ) = \lambda _ { \mathrm { M S E } } \cdot { \mathcal { L } } ( \mu ) + \lambda _ { \mathrm { T V } } \cdot \mathrm { T V } ( \mu ) ,\tag{8}
$$

where the reconstruction term $\begin{array} { r } { \mathcal L ( \mu ) = \frac { 1 } { | \mathcal { M } | } \sum _ { ( z , x ) \in \mathcal { M } } \left[ \hat { I } _ { \mathrm { d B } } ( z , x ) - I _ { \mathrm { t a r g e t } } ( z , x ) \right] ^ { 2 } } \end{array}$ is the masked MSE of $\operatorname { E q }$ (6), with $[ \mathcal { F } ( \mu ; \pmb { \theta } ) ] ( z , x ) = \hat { I } _ { \mathrm { d B } } ( z , x )$ the predicted log-scale intensity at pixel $( z , x )$ from Eq. (5), $I _ { \mathrm { t a r g e t } } ( z , x )$ the corresponding measured intensity [dB], and $| { \mathcal { M } } |$ the number of tissue pixels above the noise floor. The weights $\lambda _ { \mathrm { M S E } }$ and $\lambda _ { \mathrm { T V } }$ control the relative weighting of the reconstruction and regularization terms. For the agar-phantom inversions, we weight each loss equally $\lambda _ { \mathrm { M S E } } = \lambda _ { \mathrm { T V } }$ . The clinical cohort inversions used no TV penalty, whereas the five frames shown in Figure 5 and analyzed in Supplementary Note 8 used $\lambda _ { \mathrm { M S E } } = \lambda _ { \mathrm { T V } }$ . The optimizer comparisons specify TV separately because it is disabled in the synthetic scaling sweeps, enabled axially in the measured A-line sweep, and varied in the matched joint and independent fits (Supplementary Note 1). The sensitivity of the recovered attenuation to $\lambda _ { \mathrm { T V } }$ is characterized in the sensitivity study (Supplementary Note 3).

The attenuation map is optimized using the Adam optimizer. At each iteration, gradients $\partial \mathcal { L } / \partial \mu$ are computed via backpropagation through the forward model and applied to update $\mu .$ After each optimizer step, $\mu$ is projected onto the feasible interval $[ \mu _ { \mathrm { m i n } } , \mu _ { \mathrm { m a x } } ]$ and, for the agar-phantom and clinical inversions, noise-floor pixels are re-zeroed. In these inversions, the optimizer tracks the best reconstruction loss throughout training and restores the corresponding $\mu$ at convergence.

The system gain $G _ { \mathrm { d B } }$ is initialized prior to optimization by running the forward model with the initial $\mu$ and computing the median offset between predicted and target tissue intensities. This closed-form initialization places the predicted B-scan in the correct intensity range before optimization begins. For the agar-phantom and clinical inversions, $G _ { \mathrm { d B } }$ is recomputed at each iteration as the MSE-optimal additive offset:

$$
G _ { \mathrm { d B } } ^ { * } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( z , x ) \in \mathcal { M } } \big [ I _ { \mathrm { t a r g e t } } ( z , x ) - \mathcal { F } ( \mu ; \pmb \theta ) \big | _ { G _ { \mathrm { d B } } = 0 } ( z , x ) \big ] ,\tag{9}
$$

where $\mathcal { F } ( \mu ; \pmb { \theta } ) \big | _ { G _ { \mathrm { d B } } = 0 }$ is the forward model prediction with gain set to zero. This per-iteration recomputation minimizes the residual offset for the current $\mu ;$ it does not independently calibrate the attenuation scale.

## Computational Verification Protocol

To verify numerical correctness of the forward and inverse models, we conducted two complementary tests. First, the forward model was evaluated on 500 synthetic A-lines, each containing 2 to 5 homogeneous layers with randomized attenuation coefficients drawn from $[ 0 . 1 , 1 5 . 0 ] ~ \mathrm { m m } ^ { - 1 }$ . With the confocal function disabled, system gain set to unity, and $\alpha = 1$ , the forward model output was compared pixel-by-pixel against the analytical Beer-Lambert solution computed independently. Agreement was quantified by the coefficient of determination $( R ^ { 2 } )$ and the maximum absolute error across all pixels.

Second, the inverse optimization was evaluated on 500 synthetic B-scans with spatially varying, per-pixel ground-truth attenuation. Each B-scan was generated by the forward model with known $\mu , \alpha ,$ confocal parameters, and system gain. The attenuation map was then scrambled to random initial values and recovered by the optimizer over 30,000 iterations, with α and $G _ { \mathrm { d B } }$ frozen at their true values to isolate the recovery of $\mu .$

To assess scaling with coupled parameters, the same attenuation field was sampled within a 0.16 mm tissue band at $K \ = \ 2 5 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 , 0 0 0 , 2 , 0 0 0$ and 5,000 depths in one A-line. Axial spacing was $\Delta z = 0 . 1 6 / K$ mm; focal position and Rayleigh range in pixels were rescaled to preserve the physical confocal geometry. Random Search [70], CMA-ES [71], Adam [72], and L-BFGS-B [73] shared the target, masks, fixed gain and backscattering fraction, OAC bounds of $[ 0 . 0 1 , \dot { 2 0 } ] \ \dot { \mathrm { m m } } ^ { - 1 }$ , and eight paired initializations drawn uniformly from [0.01, 10] mm−1. Each optimizer used a nominal budget of 20,000 iterations (Adam, Random Search, and L-BFGS-B) or objective evaluations (CMA-ES). Recovery accuracy was quantified by the mean absolute percentage error (MAPE) between recovered and ground-truth tissue attenuation:

$$
\mathrm { M A P E } = \frac { 1 } { | \mathcal { T } | } \sum _ { ( z , x ) \in \mathcal { T } } \frac { | \hat { \mu } ( z , x ) - \mu _ { \mathrm { t r u e } } ( z , x ) | } { \mu _ { \mathrm { t r u e } } ( z , x ) } \times 1 0 0 \% ,\tag{10}
$$

where $\tau$ denotes the set of tissue pixels. The MAPE is evaluated on the recovered attenuation coefficient rather than on image intensity. Wall-clock time per evaluation (ms/eval) was also recorded to quantify per-iteration compute cost. Per-evaluation cost differs across methods because each evaluation carries different work: Random Search runs only a forward pass; Adam adds a backward pass to compute gradients; L-BFGS-B additionally maintains a running history of past gradients to build its quasi-Newton search direction, and that bookkeeping cost grows as the parameter count increases; CMA-ES on top of evaluation must update and occasionally re-decompose a covariance matrix over the parameters, so its per-evaluation cost climbs steeply with problem size and dominates at large problem scales. Supplementary Note 1 describes the synthetic B-scan sweep, with 40 parameters per A-line, and the measured-data comparisons.

## Physical Phantom Validation

For the liquid comparison, five intralipid concentrations (0.5%, 1.0%, 1.5%, 2.0%, and 3.0%) were imaged at two focal positions (H and L; Figure 3a-c), with 512 B-scans of 500 × 500 pixels per acquisition. The system used a 1300 nm source with 95 nm bandwidth and an LSM03 scan lens (NA ≈ 0.05), and axial sampling in water was 7.35 μm. Despeckled 16-bit intensities were converted as $I _ { \mathrm { d B } } = I _ { \mathrm { r a w } } \times 1 5 0 / 6 5 5 3 5$ . Linear intensities were corrected as $I _ { \mathrm { c o r r , l i q } } ( z ) = [ I ( z ) - N ( z ) ] / [ S ( z ) \cdot H ( z ) ] $ using the measured noise and sensitivity curves and calibrated confocal function $( z _ { R } = 0 . 4 3 1 5$ mm, with a separate focal position for each B-scan). Both CF and DiffOCT were fitted to these noise-, sensitivity-, and confocal-corrected liquid-phantom intensities. Fitting extended from the start of each A-line's 100-pixel CF window to its last valid depth. Only finite, positive corrected measurements entered the fixed loss mask; excluded observations within this depth interval remained in the cumulative attenuation path.

For each A-line, CF supplied a constant $\mu _ { \mathrm { C F } }$ , fitted as a log-linear decay over a 100-pixel depth window (0.735 mm), and the natural-log intensity at the window start, $b _ { s }$ . Its equivalent DiffOCT gain, $\begin{array} { r l } { G _ { \mathrm { d B } } } & { { } = } \end{array}$ $1 0 b _ { s } / \ln ( 1 0 ) - 1 0 \log _ { 1 0 } \mu _ { \mathrm { C F } }$ , was held fixed throughout optimization. Thus, gain calibration uses both the

CF intercept and attenuation estimate. With $\alpha = 1$ , maps were initialized uniformly over $[ 0 , 5 ] ~ \mathrm { m m } ^ { - 1 }$ and optimized with Adam for 15,000 iterations, using a cosine learning-rate decay from $1 0 ^ { - 3 }$ to $1 0 ^ { - 5 }$ , bounds of $[ 1 0 ^ { - 6 } , 2 0 ] ~ \mathrm { m m } ^ { - 1 }$ , and $\lambda _ { \mathrm { M S E } } = \lambda _ { \mathrm { T V } } = 1 0 0 0$ with axial and lateral $\mathrm { T V } .$

Additionally, five different in vitro phantoms were fabricated at different intralipid-in-agar concentrations (1.0%, 1.5%, 2.0%, 3.0%, and 5.0%), each imaged at two focal positions, and were used for visualizing the optimization workflow (Supplementary Figure 2b) and for the identifiability and noise robustness studies (Figure 4). Each phantom was imaged with a wavelength-swept source OCT system with negligible sensitivity fall-off (central wavelength 1300 nm, bandwidth 90 nm, described in detail elsewhere [50]), with an LSM04 scan lens $( \mathrm { N A } = 0 . 0 3 1 5 )$ , each acquisition producing a volumetric stack of 512 B-scans at $5 0 0 \times 5 0 0$ pixels per frame (500 depth samples × 500 A-lines). Intensity $I _ { \mathrm { r a w } }$ data were stored as 16-bit TIFF files in log-intensity scale, with the mapping $I _ { \mathrm { d B } } = I _ { \mathrm { r a w } } \times ( 1 3 0 / 6 5 5 3 5 )$ yielding values in the 0–130 dB.

The two tomograms from each phantom were registered in three dimensions via a fast Fourier transform to automatically determine the axial shift δz together with small lateral shifts arising from deviation from pure axial translation. Speckle was reduced by Gaussian spatial filtering with $1 / e ^ { 2 } { \mathrm { - d i a m e t e r s } }$ of $7 \times 1 5$ pixels along the fast-scan and slow-scan axes, respectively, applied to all volumes prior to optimization.

The paired tomograms and the recovered axial shift were used to estimate the focal plane position $z _ { f }$ and effective Rayleigh range $z _ { R }$ via logarithmic-intensity matching [74]. Specifically, the optimal parameters were found by minimizing the mean-squared difference between the two confocal-corrected tomograms:

$$
\begin{array} { r l r } & { } & { \{ z _ { f } ^ { * } , z _ { R } ^ { * } \} = \arg \underset { z _ { f } , z _ { R } } { \operatorname* { m i n } } \frac { 1 } { N } \sum _ { z , x , y } \big | I _ { \mathrm { d B } } ^ { ( 1 ) } ( z , x , y ) - H _ { \mathrm { d B } } ( z ; z _ { f } , z _ { R } ) } \\ & { } & { \quad \quad - \big [ I _ { \mathrm { d B } } ^ { ( 2 ) } ( z , x , y ) - H _ { \mathrm { d B } } ( z - \delta z ; z _ { f } , z _ { R } ) \big ] \big | ^ { 2 } , } \end{array}\tag{11}
$$

where $I ^ { ( 1 ) }$ and $I ^ { ( 2 ) }$ are the two registered tomograms from the same phantom, $N$ is the number of voxels, and $H _ { \mathrm { d B } }$ is the confocal gate (Eq. (2)) expressed in log scale.

Independent reference attenuation coefficients were obtained by conventional curve-fitting (CF) slope analysis performed in MATLAB on the same phantom volumes. After noise-floor subtraction and correction for the calibrated confocal function, a linear model $I _ { \mathrm { l n } } ( z ) = b _ { 0 } + b _ { 1 } \cdot z$ , with z the depth-pixel index, was fitted to the natural-logarithmic intensity of 40 depth pixels (11-50 below the surface peak) within the thresholded tissue mask. The attenuation coefficient was then computed as $\mu _ { \mathrm { C F } } = - b _ { 1 } / ( 2 \delta _ { z } )$ , where $\delta _ { z }$ is the axial pixel size in water and the factor of $1 / 2$ accounts for the double-pass geometry.

Each of the 512 B-scans per acquisition was processed independently using the single-scattering forward model $\left( \mathrm { E q . \ ( 5 ) } \right)$ with noise floor subtraction $\left( \mathrm { E q . \ ( 3 ) } \right)$ and per-iteration gain reoptimization $\left( \mathrm { E q . \ ( 9 ) } \right)$ . Recovered attenuation coefficients were evaluated exclusively over tissue pixels defined by a binary segmentation mask derived from independent MATLAB ground-truth labeling, ensuring that reported $\mu$ statistics excluded noise-floor-dominated regions where the optimizer assigns zero attenuation.

## Identifiability

The identifiability study tested whether the recovered attenuation is independent of optimization randomness and initialization. In the first experiment, each acquisition was optimized 20 times with different random seeds controlling the initial $\mu$ values, while keeping all other parameters fixed. The coefficient of variation $\mathrm { ( C o V ) }$ of the recovered tissue $\mu$ across seeds was used to quantify reproducibility. In the second experiment, the initialization range was varied across 10 conditions spanning narrow $( [ 1 , 1 ] \ \mathrm { \ m m ^ { - 1 } } )$ , wide $( [ 0 , 2 0 ] \ \mathrm { m m ^ { - 1 } } )$ , and biased $( [ 7 , 1 5 ] \ \mathrm { m m ^ { - 1 } } )$ ranges, with 5 seeds per condition per acquisition. Agreement of the converged $\mu$ across initialization conditions was assessed by the CoV and by the range of recovered tissue means.

## Noise Robustness

The robustness study quantified how additive noise degrades attenuation recovery. Zero-mean, circular Gaussian noise $\sim \mathcal { N } ( 0 , \sigma _ { n } )$ was added to the complex-valued data at controlled signal-to-noise ratios (SNR). The noise variance $\sigma _ { n } ^ { 2 }$ to achieve a target SNR was calculated as

$$
\sigma _ { n } ^ { 2 } = \bar { I } _ { n } \left( \frac { R - R ^ { \prime } } { 1 + R ^ { \prime } } \right) ,\tag{12}
$$

where R and $R ^ { \prime }$ are the original and target SNRs, defined with the rigorous definition $R = ( \bar { I } _ { s } - \bar { I } _ { n } ) / \bar { I } _ { n }$ [63], where $\bar { I } _ { s }$ and ${ \bar { I } } _ { n }$ represent the average intensity at a signal-dominated region (a slab covering 10 depth pixels below the phantom surface) and a noise-dominated region (the air region above the phantom surface), respectively. The noise injection protocol was applied to one B-scan from each of the 10 phantom acquisitions, with the noiseless recovery serving as the baseline reference. Degraded-SNR cases varied from -3 dB up to each phantom's original SNR in 1 dB steps. Degradation was quantified by the change in recovered tissue $\mu$ relative to the noiseless condition. For reference, SNR = 20 dB corresponds to a signal intensity two orders of magnitude above the noise floor, while $\mathrm { S N R } = 0$ dB corresponds to noise equal to the signal.

## Application to Clinical IV-OCT

The clinical demonstration used 29,269 pre-IVL intravascular OCT frames across 128 patients from the Shockwave Disrupt CAD-I through CAD-IV clinical trial cohorts [56-59]. IV-OCT images were stored as 500 × 500 pixel Cartesian cross-sections. The five representative frames displayed in Figure 5 were acquired with an ILUMIEN OPTIS system and a C7-XR Dragonfly catheter (Abbott Vascular).

Each Cartesian cross-section was converted to a polar representation (250 depth × 500 A-lines) by extracting radial A-lines from the catheter center outward. The Rayleigh range $z _ { R }$ and focal plane position $z _ { f }$ were set to 31 pixels (≈0.59 mm) and 55 pixels (≈1.06 mm) respectively, adopting the measured confocal parameters reported by Gargesha et al. for the C7-XR / Dragonfly catheter family [64]. The axial pixel size of 0.01932 mm/pixel follows from the ILUMIEN OPTIS A-scan range of 4.83 mm in contrast over 250 depth samples per the device's 510(k) summary (FDA K141453, https://www.accessdata.fda.gov/CDRH510K K141453.pdf); the numerical reconstruction used 0.020 mm/pixel. Per-pixel attenuation maps $\mu ( z , x )$ were recovered over 25,000 iterations with the backscattering fraction α held fixed at unity (Stage 1). Our optimized $\mu$ maps were then frozen and a per-pixel backscattering map $\alpha \in [ 1 0 ^ { - 1 0 } , 1 0 ^ { - 2 } ]$ (bounds informed by supplementary of [50]) was optimized over an additional 25,000 iterations (Stage 2) to decouple per-pixel brightness from cumulative attenuation. Additional details of this two-stage procedure are given in Supplementary Note 6. The noise floor was subtracted in linear intensity scale and the system gain was recomputed at each iteration. Rendering quality was assessed by the peak signal-to-noise ratio (PSNR) between the forward model prediction and the noise-floor-corrected target image, computed in log-intensity scale over tissue pixels as

$$
\mathrm { P S N R } = 1 0 \log _ { 1 0 } \left( \frac { D ^ { 2 } } { \mathrm { M S E } } \right) ,\tag{13}
$$

where $\begin{array} { r } { \mathrm { M S E } = \frac { 1 } { | T | } \sum _ { ( z , x ) \in \mathcal { T } } \left( \hat { I } _ { \mathrm { d B } } ( z , x ) - I _ { \mathrm { d B } } ( z , x ) \right) ^ { 2 } } \end{array}$ is the mean squared residual between the rendered $\left( \hat { I } _ { \mathrm { d B } } \right)$ and target $\left( I _ { \mathrm { d B } } \right)$ log-scale images evaluated only on the tissue pixel set $\tau$ defined by the noise-floor mask, and D = maxτ IdB − minτ $I _ { \mathrm { d B } }$ is the log-scale dynamic range of the tissue signal in the target. Target images were encoded with a full log-scale of 130 dB and a tissue mask threshold of 15 dB, giving a tissue dynamic range D ≈ 115 dB that is essentially constant across frames. Higher PSNR values indicate that the rendered image matches the target more closely relative to the available tissue signal range; because the denominator grows with MSE, a k-fold reduction in MSE increases PSNR $\mathrm { b y ~ } 1 0 \log _ { 1 0 } k$ dB.

In addition to PSNR, tissue-coverage agreement between the rendered and target B-scans was quantified by the Dice overlap of their binarized tissue masks. Both images were binarized at the 15 dB noise floor threshold that the optimizer uses to define tissue pixels, and the Dice coefficient was computed over the full Cartesian frame as $\mathrm { D i c e } = 2 | A \cap B | / ( | A | + | B | )$ , where A and B are the sets of pixels above threshold in the target and rendered images respectively. This metric captures whether the two images agree on which pixels carry signal above the noise floor, complementing the intensity-sensitive PSNR.

## Statistical Analysis

All quantitative analyses were performed in Python (v3.10) using NumPy (v1.24), SciPy (v1.11), and Matplotlib (v3.7). The study is descriptive throughout: no inferential hypothesis tests were applied, as this work compares recovered properties to known references and reports per-pixel error distributions rather than testing population effects. Continuous quantities computed across frames, phantoms, or random seeds are reported as mean ± standard deviation unless otherwise noted, and per-pixel distributions are summarized by median, mean absolute error, and the fraction of pixels within fixed dB tolerance envelopes $( \pm 3 , \pm 1 0 , \pm 2 0 \ \mathrm { d B } )$ . For the phantom validation, linear regression of recovered tissue attenuation against intralipid concentration was fit by ordinary least squares and reported as slope, intercept, and the coefficient of determination $R ^ { 2 }$ . For the identifiability study, the within-acquisition coefficient of variation (CoV) of recovered tissue-mean $\mu$ was computed as the standard deviation divided by the mean across the perturbation set and reported as a percentage. Initialization-range CoV was computed after averaging five seeds per condition, using the seven selected phantom conditions or all ten clinical conditions (Figure $^ \mathrm { 4 c , d }$ and Supplementary Figure 8b). For the noise robustness study, recovery degradation was quantified as the absolute relative change in recovered tissue $\mu$ versus the noiseless baseline $( | \Delta \mu | / \mu _ { \mathrm { b a s e l i n e } } )$ , and spatial agreement was tracked as the pixel-wise $R ^ { 2 }$ of recovered $\mu$ versus the noiseless baseline at each SNR.

For Figure 1, attenuation MAPE (Eq. (10)) was computed separately for each run against the known tissue attenuation. Runtime per objective evaluation was total solver wall time divided by the actual number of evaluations, including solver overhead. Points and error bars report the arithmetic mean and standard deviation across N = 8 paired initializations at each problem size. These repeats measure numerical variability on the same synthetic target.

For Figure 3f, DiffOCT attenuation was first averaged over valid pixels within each 100-pixel CF window (0.735 mm), then equally over paired A-lines to obtain one mean per B-scan. CF used the same A-lines; lines with nonfinite or nonpositive CF estimates, empty valid windows, or fallback gain were excluded from both summaries. Each acquisition contributed 512 frame means. Violin densities and black mean markers use values within $[ Q _ { 1 } - 1 . 5 \mathrm { I Q R } , Q _ { 3 } + 1 . 5 \mathrm { I Q R } ]$ , applied separately to each method, concentration, and focal position; $\mathrm { I Q R } = Q _ { 3 } - Q _ { 1 }$ . This filtering affects only the display (486–512 retained frames per violin). Regression fits use all frame means, averaged to five equally weighted concentration means for each focal position; the displayed pooled CF line uses ten acquisition means. $R ^ { 2 }$ measures concentration linearity rather than agreement between methods. Attenuation bias is the difference between unfiltered DiffOCT and CF volume means. In Figure 3g, intensity RMSE is the square root of the mean squared dB residual over the same valid CF-window pixels for both methods.

## Ethics Statement

The physical phantom experiments involved no human subjects, animal subjects, or biological tissue and required no ethical approval. The clinical intravascular OCT data were acquired as part of the Shockwave Disrupt CAD-I through CAD-IV (ClinicalTrials.gov Identifiers NCT02650128, NCT03328949, NCT03595176, and NCT04151628) study [56–59], sponsored by Shockwave Medical, Inc. All necessary patient consents were secured for the original studies. Ethical approval for secondary use of this dataset was obtained from the Institutional Review Board at Mass General Brigham under protocol IRB #2022P001679 (Personalized lesion modification optimizes atherosclerosis intervention); the approval is active and exempt. All protected health information was de-identified according to the Health Insurance Portability and Accountability Act (HIPAA) requirements, and identifiers such as names and birth dates were removed from structured data sources.

## Data Availability

The clinical intravascular OCT data analyzed in this study were provided by Shockwave Medical, Inc. and are not publicly available because their sharing is restricted by the data use agreement with Shockwave Medical, Inc. The intralipid phantom OCT data generated and analyzed in this study are available from the corresponding author, Ajay Manicka (ajay\_m@mit.edu), upon reasonable request. Requests will normally be answered within 3 weeks. The source data for Figures 1–5 are provided in Supplementary Data 1. Additional material is provided as Supplementary Information.

## Code Availability

All code used in this study, including the code to generate the synthetic phantoms, will be released on GitHub and archived on Zenodo with a DOI upon publication. The code is GPU-accelerated but also CPU-compatible

## Acknowledgements

The authors would like to thank Taylor Cannon and Louis Leger for their helpful discussions throughout the duration of this article. We also thank Shockwave Medical Inc. for providing the rich dataset used in this study. We acknowledge Albert Swiston and MIT Supercloud for their support and computing resources. Figures were made using Biorender software and Python visualization libraries.

## Funding

This work was supported by a grant to E.R.E and F.N. from the National Institutes of Health (R01 HL161069).   
F.N. graciously acknowledges career funding from American Heart Association (25CDA1452999).

## Competing Interest

The authors declare no competing interests.

## Author Information

A.M. conceived, designed, and executed the project. K.K. provided technical support for the inverse modeling design. S.R-L. conducted experimental testing and validation on intralipid phantoms. M.A. provided substantial editorial contributions to the manuscript and helped design studies. S.A. helped develop figures and contributed to the discussion of results. S.R-C helped perform analyses and contributed to the discussion of results. N.U-P. and B.E.B. provided guidance on OCT physics and forward model design. F.N. and E.R.E. supervised and advised A.M. throughout all stages of the project. All authors contributed to the review and editing of the manuscript.

## References

[1] Berger, A.: How does it work? magnetic resonance imaging.(clinical review). British medical journal 324(7328), 35–36 (2002)

[2] Harvey, C.J., Pilcher, J.M., Eckersley, R.J., Blomley, M.J., Cosgrove, D.O.: Advances in ultrasound. Clinical radiology 57(3), 157–177 (2002)

[3] Buzug, T.M.: Computed tomography. In: Springer Handbook of Medical Technology, pp. 311–342. Springer, Berlin, Heidelberg (2011)

[4] Drexler, W., Fujimoto, J.G.: Optical Coherence Tomography: Technology and Applications. Springer, Berlin, Heidelberg (2008)

[5] Shung, K.K., Smith, M., Tsui, B.M.: Principles of Medical Imaging. Academic Press, San Diego, CA (2012)

[6] Webb, A.: Introduction to Biomedical Imaging. John Wiley & Sons, Hoboken, NJ (2022)

[7] Oppelt, A.: Imaging Systems for Medical Diagnostics: Fundamentals, Technical Solutions and Applications for Systems Applying Ionizing Radiation, Nuclear Magnetic Resonance and Ultrasound. John Wiley & Sons, Erlangen, Germany (2006)

[8] Levental, K.R., Yu, H., Kass, L., Lakins, J.N., Egeblad, M., Erler, J.T., Fong, S.F., Csiszar, K., Giaccia, A., Weninger, W., et al.: Matrix crosslinking forces tumor progression by enhancing integrin signaling. Cell 139(5), 891–906 (2009)

[9] Webb, B.A., Chimenti, M., Jacobson, M.P., Barber, D.L.: Dysregulated ph: a perfect storm for cancer progression. Nature Reviews Cancer 11(9), 671–677 (2011)

[10] Gabriel, C., Gabriel, S., Corthout, Y.: The dielectric properties of biological tissues: I. literature survey. Physics in medicine & biology 41(11), 2231 (1996)

[11] Li, T.-M., Aittala, M., Durand, F., Lehtinen, J.: Differentiable monte carlo ray tracing through edge sampling. ACM Transactions on Graphics (TOG) 37(6), 1–11 (2018)

[12] Pharr, M., Jakob, W., Humphreys, G.: Physically Based Rendering: From Theory to Implementation. MIT Press, Cambridge, MA (2023)

[13] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)

[14] Gopalakrishnan, V., Dey, N., Golland, P.: Intraoperative 2d/3d image registration via differentiable x-ray rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11662–11672 (2024)

[15] Bertramo, N., Duguey, G., Gopalakrishnan, V.: Diffus: Differentiable ultrasound rendering from volumetric imaging. In: International Workshop on Advances in Simplifying Medical Ultrasound, pp. 23–32 (2025). Springer

[16] Finkelstein, A., Vladimirov, N., Zaiss, M., Perlman, O.: Multi-parameter molecular mri quantification using physics-informed self-supervised learning. Communications Physics 8(1), 164 (2025)

[17] Simson, W., Zhuang, L., Sanabria, S.J., Antil, N., Dahl, J.J., Hyun, D.: Differentiable beamforming for ultrasound autofocusing. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 428–437 (2023). Springer

[18] Kuc, R.: Clinical application of an ultrasound attenuation coefficient estimation technique for liver pathology characterization. IEEE Transactions on Biomedical Engineering (6), 312–319 (2007)

[19] Berger, G., Laugier, P., Thalabard, J., Perrin, J.: Global breast attenuation: control group and benign breast diseases. Ultrasonic imaging 12(1), 47–57 (1990)

[20] Sagar, K.B., Agemura, D.H., D O'Brien Jr, W., Pelc, L.R., Rhyne, T.L., Wann, L.S., Komorowski, R.A., Warltier, D.C.: Quantitative ultrasonic assessment of normal and ischaemic myocardium with an acoustic

microscope: relationship to integrated backscatter. Cardiovascular research 24(6), 447–455 (1990)

[21] Sugata, Y., Murakami, K., Ito, M., Shiina, T., Yamamoto, Y.: An application of ultrasonic tissue characterization to the diagnosis of cataract. Acta Ophthalmologica 70(S204), 35–39 (1992)

[22] Xu, C., Schmitt, J.M., Carlier, S.G., Virmani, R.: Characterization of atherosclerosis plaques by measuring both backscattering and attenuation coefficients in optical coherence tomography. Journal of biomedical optics 13(3), 034003–034003 (2008)

[23] McLaughlin, R.A., Scolaro, L., Robbins, P., Saunders, C., Jacques, S.L., Sampson, D.D.: Parametric imaging of cancer with optical coherence tomography. Journal of biomedical optics 15(4), 046029–046029 (2010)

[24] Barwari, K., Bruin, D.M., Cauberg, E.C., Faber, D.J., Leeuwen, T.G., Wijkstra, H., Rosette, J., Laguna, M.P.: Advanced diagnostics in renal mass using optical coherence tomography: a preliminary report. Journal of endourology 25(2), 311–315 (2011)

[25] Barwari, K., Bruin, D.M., Faber, D.J., Leeuwen, T.G., Rosette, J.J., Laguna, M.P.: Differentiation between normal renal tissue and renal tumours using functional optical coherence tomography: a phase i in vivo human study. BJU international 110(8b), 415–420 (2012)

[26] Tomlins, P.H., Adegun, O., Hagi-Pavli, E., Piper, K., Bader, D., Fortune, F.: Scattering attenuation microscopy of oral epithelial dysplasia. Journal of biomedical optics 15(6), 066003–066003 (2010)

[27] Zhang, Q., Wu, X., Tang, T., Zhu, S., Yao, Q., Gao, B.Z., Yuan, X.: Quantitative analysis of rectal cancer by spectral domain optical coherence tomography. Physics in Medicine & Biology 57(16), 5235 (2012)

[28] Van Der Schoot, J., Vermeer, K.A., Boer, J.F., Lemij, H.G.: The effect of glaucoma on the optical attenuation coefficient of the retinal nerve fiber layer in spectral domain optical coherence tomography images. Investigative ophthalmology & visual science 53(4), 2424–2430 (2012)

[29] Meer, F.J., Faber, D.J., Aalders, M.C., Poot, A.A., Vermes, I., Leeuwen, T.G.: Apoptosis-and necrosis-induced changes in light attenuation measured by optical coherence tomography. Lasers in medical science 25(2), 259–267 (2010)

[30] Huang, D., Swanson, E.A., Lin, C.P., Schuman, J.S., Stinson, W.G., Chang, W., Hee, M.R., Flotte, T., Gregory, K., Puliafito, C.A., et al.: Optical coherence tomography. science 254(5035), 1178–1181 (1991)

[31] Bouma, B.E., Villiger, M., Otsuka, K., Oh, W.-Y.: Intravascular optical coherence tomography. Biomedical optics express 8(5), 2660–2686 (2017)

[32] Stone, G.W., Maehara, A., Lansky, A.J., De Bruyne, B., Cristea, E., Mintz, G.S., Mehran, R., McPherson, J., Farhat, N., Marso, S.P., et al.: A prospective natural-history study of coronary atherosclerosis. New England journal of medicine 364(3), 226–235 (2011)

[33] Virmani, R., Kolodgie, F.D., Burke, A.P., Farb, A., Schwartz, S.M.: Lessons from sudden coronary death: a comprehensive morphological classification scheme for atherosclerotic lesions. Arteriosclerosis, thrombosis, and vascular biology 20(5), 1262–1275 (2000)

[34] Wang, L.V., Hu, S.: Photoacoustic tomography: in vivo imaging from organelles to organs. science 335(6075), 1458–1462 (2012)

[35] Cox, B., Laufer, J.G., Arridge, S.R., Beard, P.C.: Quantitative spectroscopic photoacoustic imaging: a review. Journal of biomedical optics 17(6), 061202–061202 (2012)

[36] Iskander-Rizk, S., Van Der Steen, A.F., Van Soest, G.: Photoacoustic imaging for guidance of interventions in cardiovascular medicine. Physics in Medicine & Biology 64(16), 16–01 (2019)

[37] Yabushita, H., Bouma, B.E., Houser, S.L., Aretz, H.T., Jang, I.-K., Schlendorf, K.H., Kauffman, C.R., Shishkov, M., Kang, D.-H., Halpern, E.F., et al.: Characterization of human atherosclerosis by optical coherence tomography. Circulation 106(13), 1640–1645 (2002)

[38] Van Soest, G., Goderie, T., Regar, E., Koljenović, S., Leenders, G.L., Gonzalo, N., Noorden, S., Okamura, T., Bouma, B.E., Tearney, G.J., et al.: Atherosclerotic tissue characterization in vivo by optical coherence

[39] Scolaro, L., McLaughlin, R.A., Klyen, B.R., Wood, B.A., Robbins, P.D., Saunders, C.M., Jacques, S.L., Sampson, D.D.: Parametric imaging of the local attenuation coefficient in human axillary lymph nodes assessed using optical coherence tomography. Biomedical optics express 3(2), 366–379 (2012)

[40] Cauberg, E.C., Bruin, D.M., Faber, D.J., Reijke, T.M., Visser, M., La Rosette, J.J., Leeuwen, T.G.: Quantitative measurement of attenuation coefficients of bladder biopsies using optical coherence tomography for grading urothelial carcinoma of the bladder. Journal of biomedical optics 15(6), 066013–066013 (2010)

[41] Gong, P., McLaughlin, R.A., Liew, Y.M., Munro, P.R., Wood, F.M., Sampson, D.D.: Assessment of human burn scars with optical coherence tomography by imaging the attenuation coefficient of tissue after vascular masking. Journal of biomedical optics 19(2), 021111–021111 (2014)

[42] Qin, N., Liu, Y., Huang, L., Xin, Y., Zhang, X., Hu, X., Li, Q.: Research on optical properties of cardiovascular tissues based on oct data. Journal of Innovative Optical Health Sciences 14(01), 2140007 (2021)

[43] Almasian, M., Bosschaart, N., Leeuwen, T.G., Faber, D.J.: Validation of quantitative attenuation and backscattering coefficient measurements by optical coherence tomography in the concentration-dependent and multiple scattering regime. Journal of biomedical optics 20(12), 121314–121314 (2015)

[44] Tearney, G.J., Regar, E., Akasaka, T., Adriaenssens, T., Barlis, P., Bezerra, H.G., Bouma, B., Bruining, N., Cho, J.-m., Chowdhary, S., et al.: Consensus standards for acquisition, measurement, and reporting of intravascular optical coherence tomography studies: a report from the international working group for intravascular optical coherence tomography standardization and validation. Journal of the American College of Cardiology 59(12), 1058–1072 (2012)

[45] Yonetsu, T., Jang, I.-K.: Cardiac optical coherence tomography: History, current status, and perspective. JACC: Asia 4(2), 89–107 (2024)

[46] Chang, S., Bowden, A.K.: Review of methods and applications of attenuation coefficient measurements with optical coherence tomography. Journal of biomedical optics 24(9), 090901–090901 (2019)

[47] Meer, F.J., Faber, D.J., Sassoon, D.M.B., Aalders, M.C., Pasterkamp, G., Leeuwen, T.G.: Localized measurement of optical attenuation coefficients of atherosclerotic plaque constituents by quantitative optical coherence tomography. IEEE transactions on medical imaging 24(10), 1369–1376 (2005)

[48] Vermeer, K.A., Mo, J., Weda, J.J., Lemij, H.G., Boer, J.F.: Depth-resolved model-based reconstruction of attenuation coefficients in optical coherence tomography. Biomedical optics express 5(1), 322–337 (2013)

[49] Liu, S., Sotomi, Y., Eggermont, J., Nakazawa, G., Torii, S., Ijichi, T., Onuma, Y., Serruys, P.W., Lelieveldt, B.P., Dijkstra, J.: Tissue characterization with depth-resolved attenuation coefficient and backscatter term in intravascular optical coherence tomography images. Journal of biomedical optics 22(9), 096004–096004 (2017)

[50] Cannon, T.M., Bouma, B.E., Uribe-Patarroyo, N.: Layer-based, depth-resolved computation of attenuation coefficients and backscattering fractions in tissue using optical coherence tomography. Biomedical Optics Express 12(8), 5037–5056 (2021)

[51] Yu, J., Wang, Y., Guo, W., Gao, Y., Sun, Y., Kang, J.U.: End-to-end reconstruction of oct optical properties and speckle-reduced structural intensity via physics-based learning. arXiv preprint arXiv:2602.02721 (2026)

[52] Wang, Y., Wei, S., Kang, J.U.: Depth-dependent attenuation and backscattering characterization of optical coherence tomography by stationary iterative method. Journal of biomedical optics 28(8), 085002–085002 (2023)

[53] Li, Y., Zhang, C., Huang, T., Fan, Y., Ning, G., Liao, H.: Computational multi-angle optical coherence tomography using implicit neural representation. Optics & Laser Technology 184, 112551 (2025)

[54] Thrane, L., Yura, H.T., Andersen, P.E.: Analysis of optical coherence tomography systems based on the extended huygens-fresnel principle. Journal of the Optical Society of America A 17(3), 484–490 (2000)

[55] Oshina, I., Spigulis, J.: Beer-lambert law for optical tissue diagnostics: current state of the art and the main limitations. Journal of biomedical optics 26(10), 100901–100901 (2021)

[56] Ali, Z.A., Brinton, T.J., Hill, J.M., Maehara, A., Matsumura, M., Karimi Galougahi, K., Illindala, U., Götberg, M., Whitbourn, R., Van Mieghem, N., et al.: Optical coherence tomography characterization of coronary lithoplasty for treatment of calcified lesions: first description. JACC: Cardiovascular Imaging 10(8), 897–906 (2017)

[57] Ali, Z.A., Nef, H., Escaned, J., Werner, N., Banning, A.P., Hill, J.M., De Bruyne, B., Montorfano, M., Lefevre, T., Stone, G.W., et al.: Safety and effectiveness of coronary intravascular lithotripsy for treatment of severely calcified coronary stenoses: the disrupt cad ii study. Circulation: Cardiovascular Interventions 12(10), 008434 (2019)

[58] Kereiakes, D.J., Hill, J.M., Shlofmitz, R.A., Klein, A.J., Riley, R.F., Price, M.J., Herrmann, H.C., Bachinsky, W., Waksman, R., Stone, G.W., et al.: Intravascular lithotripsy for treatment of severely calcified coronary arteries: 2-year results—disrupt cad iii study. Cardiovascular Interventions 16(19), 2472–2474 (2023)

[59] Saito, S., Yamazaki, S., Takahashi, A., Namiki, A., Kawasaki, T., Otsuji, S., Nakamura, S., Shibata, Y., Investigators, D.C.I., et al.: Intravascular lithotripsy for vessel preparation in calcified coronary arteries prior to stent placement—japanese disrupt cad iv study 1-year results—. Circulation reports 4(9), 399–404 (2022)

[60] Fujino, A., Mintz, G.S., Matsumura, M., Lee, T., Kim, S.-Y., Hoshino, M., Usui, E., Yonetsu, T., Haag, E.S., Shlofmitz, R.A., et al.: A new optical coherence tomography-based calcium scoring system to predict stent underexpansion. EuroIntervention 13(18), 2182–2189 (2018)

[61] Chen, N., Brady, D.J., Lam, E.Y.: Differentiable imaging: progress, challenges, and outlook. Advanced Devices & Instrumentation 6, 0117 (2025)

[62] Barbastathis, G., Ozcan, A., Situ, G.: On the use of deep learning for computational imaging. Optica 6(8), 921–943 (2019)

[63] Uribe-Patarroyo, N., Post, A.L., Ruiz-Lopera, S., Faber, D.J., Bouma, B.E.: Noise and bias in optical coherence tomography intensity signal decorrelation. OSA continuum 3(4), 709–741 (2020)

[64] Gargesha, M., Shalev, R., Prabhu, D., Tanaka, K., Rollins, A.M., Costa, M., Bezerra, H.G., Wilson, D.L.: Parameter estimation of atherosclerotic tissue optical properties from three-dimensional intravascular optical coherence tomography. Journal of medical imaging 2(1), 016001–016001 (2015)

[65] Cai, X., Cao, Y., Lu, Q., Jing, J., Wei, K., Hu, Y., Wang, M., Zhou, S., Chen, Y.: In vivo quantitative assessment of atherosclerotic plaque: A validation study on the optical coherence tomography-derived index of plaque attenuation. Medicine 104(36), 44196 (2025)

[66] Hartinger, A.E., Nam, A.S., Chico-Calero, I., Vakoc, B.J.: Monte carlo modeling of angiographic optical coherence tomography. Biomedical optics express 5(12), 4338–4349 (2014)

[67] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., Lerer, A.: Automatic differentiation in PyTorch. NeurIPS. NeurIPS-W (2017)

[68] Faber, D.J., Meer, F.J.v.d., Aalders, M.C., Leeuwen, T.G.v.: Quantitative measurement of attenuation coefficients of weakly scattering media using optical coherence tomography. Optics express 12(19) 4353–4365 (2004)

[69] Choma, M.A., Sarunic, M.V., Yang, C., Izatt, J.A.: Sensitivity advantage of swept source and fourier domain optical coherence tomography. Optics express 11(18), 2183–2189 (2003)

[70] Solis, F.J., Wets, R.J.-B.: Minimization by random search techniques. Mathematics of operations research 6(1), 19–30 (1981)

[71] Hansen, N., Ostermeier, A.: Completely derandomized self-adaptation in evolution strategies. Evolutionary computation 9(2), 159–195 (2001)

[72] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

[73] Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.: A limited memory algorithm for bound constrained optimization. SIAM Journal on scientific computing 16(5), 1190–1208 (1995)

[74] Dwork, N., Smith, G.T., Pauly, J.M., Bowden, A.K.E.: Automated estimation of oct confocal function parameters from two b-scans. In: CLEO: Applications and Technology, pp. 1-4 (2016). Optica Publishing Group

![](images/68fb4e01f23780399e67c778619990a9ec854b214def970caa0b4248bb775e99.jpg)

Inverse verification

## Figure 2

![](images/45fa896eb81ed49c43ce489971ba5e6a5d47c4957b46729bfce821a3398e9a2a.jpg)

(a) OCT Instrument  
![](images/6a2ae31d8516ffc128b5286d7de6fa53fe94d84f5f2580ddf2951fc7c2aa672f.jpg)

(b) OCT Parameters  
(c) Liquid Intralipid Phantoms  
![](images/18fac593d86adcf5c514478cc32983e5d8769f30a1411ebe88de8ee30e175163.jpg)

(d) OCT Phantom (3% Liquid IL)  
![](images/d1eda20add859eb4ffd7e8a3d2e99e8a7b604e05ad2e23deff8b85f807e1d96a.jpg)

(e) Recovered μ (SS)  
![](images/6d195f5106a20c2ee1b012403b6445bfe11d2bf6800db55f1d6a85bb4d6ee37e.jpg)

(f) Intralipid Concentration  
![](images/74e15f104f308bb673a0ddc2aca57fa67a7995ddd898134efa2c8b4ff848b384.jpg)  
(g) Example A-lines

![](images/3981f66a042ba6b6713bbdf48a62237a97ba31b7976bdfb25d684b4918ca8205.jpg)

![](images/6cfa108578848e41c18168dae2ba105ecf25046d28b6acdc139fccab66e0b66a.jpg)  
Depth within CF window (mm)

Figure 4
<table><tr><td>(a) Seed Sweep (b) In. Range Sweep</td><td colspan="4">(c) Identifiability Summary</td></tr><tr><td></td><td>In-vitro Phantom Acq.</td><td>Concentration and FP</td><td>Seed CoV (20 seeds) [%]</td><td>Init CoV (selected) [%]</td></tr><tr><td></td><td>A0</td><td>1.0% (F1)</td><td>0.23</td><td>13.1</td></tr><tr><td></td><td>A1</td><td>1.0% (F2)</td><td>0.18</td><td>8.3</td></tr><tr><td></td><td>A2</td><td>1.5% (F1)</td><td>0.31</td><td>7.6</td></tr><tr><td></td><td>A3</td><td>1.5% (F2)</td><td>0.31</td><td>5.9</td></tr><tr><td></td><td>A4</td><td>2.0% (F1)</td><td>0.29</td><td>7.8</td></tr><tr><td></td><td>A5</td><td>2.0% (F2)</td><td>0.28</td><td>4.7</td></tr><tr><td></td><td>A6</td><td>3.0% (F1)</td><td>0.32</td><td>6.6</td></tr><tr><td></td><td>A7</td><td>3.0% (F2)</td><td>0.27</td><td>4.7</td></tr><tr><td></td><td>A8</td><td>5.0% (F1)</td><td>0.24</td><td>2.8</td></tr><tr><td></td><td>A9</td><td>5.0% (F2)</td><td>0.28</td><td>2.1</td></tr></table>

![](images/34992bcfeb0f3e221c49a96cb8f831aec7b226a5db1b93c87859c2be19b97913.jpg)

<table><tr><td>(e) SNR Sweep</td><td colspan="3">(f) Robustness Summary</td></tr><tr><td>ise SNR [dB]</td><td>|∆µ| vs baseline (%)</td><td> $R ^ { 2 } \ ( { \mu } \ \mathbf { v } \mathbf { s }$  SNR curve)</td><td>Max σµ [mm−1]</td></tr><tr><td>Original</td><td> $0 ~ ( \mathrm { r e f } )$ </td><td>1.000</td><td></td></tr><tr><td>16</td><td> $8 . 5 \pm 8 . 2$ </td><td>0.915</td><td>1.385</td></tr><tr><td>12</td><td> $1 9 . 5 \pm 1 2 . 9$ </td><td>0.778</td><td>1.922</td></tr><tr><td>8</td><td> $3 4 . 6 \pm 1 1 . 2$ </td><td>0.585</td><td>2.190</td></tr><tr><td>4</td><td> $5 1 . 8 \pm 6 . 9$ </td><td>0.362</td><td>2.336</td></tr><tr><td>iise 0</td><td> $6 3 . 8 \pm 1 1 . 5$ </td><td>0.197</td><td>2.554</td></tr></table>

![](images/617a222914e0e73b0affd718bb9c7339a0872a1efeef31f83b215e9418753310.jpg)

GT

OAC

Figure 5

Rendered (μ)

Alpha

Rendered (µ + α)

Residual

![](images/353bc172f7ecc8fe6f67e9641a4e8ceb8a17e424eacb17885229002035312d93.jpg)

![](images/8047adbc20aac2986fd57fb09c012b90ee5044bb8e27ca668fc7145d97bb3cdb.jpg)

## Figure Captions

Fig. 1. Gradient-based optimization maintains attenuation recovery accuracy as the number of coupled depth parameters increases. Solid curves show attenuation MAPE against known ground truth (Equation (10)); dashed curves show wall time per objective evaluation. Problem size K counts depth parameters in one synthetic A-line, spanning eight sizes from 25 to 5,000 within a fixed 0.16 mm tissue band. Random Search, CMA-ES, Adam, and L-BFGS-B share the forward model, bounds, and initial values, with TV disabled and solver-specific optimization budgets. GF and GB denote gradient-free and gradient-based optimizers, respectively. Points and error bars denote mean ± one standard deviation across $N = 8$ paired initializations; curves interpolate the means. Timing includes solver overhead and is normalized by actual evaluation counts on NVIDIA Tesla V100-PCIE-32GB GPUs.

Fig. 2. Forward and inverse model verification. $\mathbf { ( a ) - ( f ) }$ Representative A-line profiles for 2-layer (a,b), 3-layer (c,d), and 4-layer (e,f) configurations, showing input attenuation µ (left) and corresponding dB intensity (right); red dashed lines indicate the analytical Beer-Lambert solution. (g) Input attenuation map µ for a representative B-scan (500 A-lines, 2–5 layers each). (h) Synthetic OCT B-scan generated by the forward model. (i) Absolute error map between forward model output and analytical solution $( R ^ { 2 } = 1 . 0 0 0 0$ $\mathrm { M A E } = 5 . 0 \times 1 0 ^ { - 6 }$ dB, max error $= 6 . 1 \times 1 0 ^ { - 5 } ~ \mathrm { d B } )$ $\mathbf { \eta } ( \mathbf { j } ) - ( \mathbf { o } )$ Inverse model verification on a representative synthetic B-scan: (j) scrambled initial µ; (k) ground-truth µ; (1) recovered µ after 30,000 iterations; (m) predicted B-scan at convergence; (n) concordance scatter plot $( R ^ { 2 } = 0 . 9 9 9 6$ , slope = 0.998, MAPE $= 0 . 1 5 \% )$ ; (o) central A-line depth profile: scrambled initial $\mu$ before optimization (gray dotted), ground-truth µ (blue dashed), and recovered $\mu$ (red solid). The gray dotted curve is the random starting point, so the comparison of interest is recovered versus ground truth, which overlap to within the line width. (p)-(s) Concordance scatter plots of recovered versus ground-truth $\mu$ across all 500 computational phantoms at four optimization snapshots. In each panel the red solid line is the ordinary least-squares fit. Snapshots are taken at iteration 0 $( R ^ { 2 } < 0 . 0 0 0 1$ $\mathrm { M A P E } = 7 7 . 0 \% )$ , iteration 1,000 $( R ^ { 2 } = 0 . 0 2 , \mathrm { M A P E } = 5 8 . 4 \% )$ , iteration 10,000 $( R ^ { 2 } = 0 . 9 3$ , M $\mathrm { I A P E } = 8 . 9 \% $ , and iteration 30,000 $( R ^ { 2 } = 0 . 9 9 9 7$ MAPE $= 0 . 1 5 \% )$ . (t) Convergence of MAPE (red, left axis) and concordance $R ^ { 2 }$ (blue, right axis) over 30,000 iterations on a logarithmic iteration axis, reaching final values of $R ^ { 2 } = 0 . 9 9 9 7$ and $\mathrm { M A P E } = 0 . 1 5 \%$

Fig. 3. Attenuation recovery in liquid intralipid phantoms. (a) Swept-source OCT instrument used for phantom imaging. (b) OCT system and image processing parameters. (c) Liquid intralipid phantoms at five concentrations $( 0 . 5 \mathrm { - } 3 . 0 \% )$ , each imaged at two focal positions (H and L). (d,e) Measured intensity and recovered single-scattering attenuation for the 3.0% H acquisition, displayed over the same 1.65 mm region from each CF-window start. (f) Distributions of B-scan mean attenuation within CF windows (N = 512 frames). Blue and orange denote DiffOCT at H and L; adjacent green violins denote CF. Black markers are retained-frame means; regression lines use unfiltered volume means, with the dashed CF line pooling H and L. (g) Two example A-lines from B-scan 256 show corrected measured intensity, CF fit, and DiffOCT prediction (top), with corresponding attenuation profiles (bottom). Prediction color and profile shading indicate local $\mu _ { \mathrm { D i f f O C T } } - \mu _ { \mathrm { C F } } ;$ annotations report intensity RMSE within the CF window.

Fig. 4. Identifiability and noise robustness. (a) Seed sweep experimental design: each acquisition is optimized 20 times with different random initializations, yielding consistent tissue-mean OAC. (b) Initialization range sweep: the µ initialization interval is varied across 10 conditions with 5 seeds each. (c) Identifiability summary table for all 10 acquisitions (concentration and focal position indicated), reporting per-acquisition seed CoV across the 20 random seeds and per-acquisition init CoV across the 7 initialization ranges satisfying the consistency criterion in (d). d Recovered tissue µ across the 10 initialization ranges for all 10 acquisitions (top), and the corresponding deviation from each acquisition's mean (bottom); the dashed boxes mark the 7 conditions for which every acquisition lies within ±25% of its mean across these conditions, after averaging five seeds per condition. The three conditions [0, 3], [1, 1], and [15, 15] $\mathrm { m m } ^ { - 1 }$ do not satisfy this criterion. (e) Noise robustness experimental design: Gaussian noise is injected at controlled signal-to-noise ratios (SNR) into phantom B-scans, which are then optimized to recover OAC maps. (f) Noise robustness summary table reporting $| \Delta \mu |$ from the noiseless baseline, $R ^ { 2 }$ of recovered µ versus its noiseless baseline, and the maximum spatial standard deviation $\sigma _ { \mu }$ of recovered tissue $\mu ,$ at six SNR levels (Original, 16, 12, 8, 4, and 0 dB). Values of $| \Delta \mu |$ are mean ± SD across $N = 1 0$ acquisitions. (g) Per-acquisition curves of $| \Delta \mu | , R ^ { 2 }$ , and $\sigma _ { \mu }$ versus SNR for all 10 phantoms. Degradation remains marginal over the first 4–5 dB of added noise below each phantom's baseline SNR and diverges sharply as SNR falls further.

Fig. 5. Clinical IV-OCT attenuation and backscattering fraction estimation. The framework was applied without modification to all 29,269 frames over the 128 patients. We show 5 representative cross-sections from a single pullback here. The white scale bars delineate 500µm. Columns show, left to right: ground-truth IV-OCT (GT, dB): recovered OAC map $( \mu , \ m m ^ { - 1 } )$ ; Stage 1 rendered image from OAC map with $\alpha ~ = ~ 1 ~ \left( \mathrm { d B } \right)$ ; recovered per-pixel backscattering map (α); Stage 2 rendered image from the sequential $\mu + \alpha$ model (dB); and the final pixel-wise residual (rendered - ground-truth, dB). Recovered attenuation maps are spatially resolved across all frames and track the tissue architecture visible in the raw images, including the vessel wall, lumen boundary, and calcified deposits. Across the full 128-patient cohort, Stage 1 rendered images reproduce the target geometry with Dice $= 0 . 9 5 7 \pm 0 . 0 0 7$ at the 15 dB noise floor and $\mathrm { P S N R } = 1 4 . 8 3 \pm 0 . 9 9 \ : \mathrm { d B }$ . Optimizing α with µ frozen (Stage 2) decouples per-pixel brightness from cumulative attenuation and raises the cohort-wide PSNR to 22.63 ± 0.74 dB $( \mathrm { D i c e } = 0 . 9 5 4 \pm 0 . 0 0 9 )$ . Cohort values are mean ± SD across N = 29,269 frames.

Fig. 6. Framework overview. A measured OCT B-scan is compared against the output of a differentiable physics-based forward model. Automatic differentiation provides gradients of the reconstruction loss with respect to the per-pixel attenuation map $\mu ( z , x )$ , which is iteratively updated via gradient-based optimization until the predicted image matches the observed measurement. As optimization proceeds, the predicted OCT image $\hat { I } ( z , x )$ converges to the target image $I _ { \mathrm { t a r g e t } } ( z , x )$ and the recovered attenuation map $\mu ( z , x )$ converges to our estimated optical attenuation coefficient at each pixel. Created in BioRender. Edelman, E. (2026) https://BioRender.com/uffg8fl