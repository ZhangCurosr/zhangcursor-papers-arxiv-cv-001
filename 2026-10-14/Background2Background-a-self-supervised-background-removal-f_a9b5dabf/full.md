# Background2Background: a self-supervised background removal framework for high dimensional microscopy.

Guangrui Ding<sup>1,5,#</sup>, Haonan Lin<sup>2,#</sup>, Rong Tang<sup>1,5</sup>, Bethany Weinberg<sup>3,5</sup>, Jianpeng Ao<sup>1,5</sup>, Qing Xia<sup>1,5</sup>, Guorong Hu<sup>1,5</sup>, Dingcheng Sun<sup>3,5</sup>, Jiaze Yin<sup>1,5</sup>, Lei Tian<sup>1,3,5</sup>, Ji-Xin Cheng<sup>1,3,4,5\*</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, Boston University, Boston, MA, USA, 02215

<sup>2</sup> The Wallace H. Coulter Department of Biomedical Engineering, Georgia Institute of Technology and Emory University, Atlanta, GA, 30332

<sup>3</sup> Department of Biomedical Engineering, Boston University, Boston, MA, USA, 02215

<sup>4</sup> Department of Chemistry, Boston University, Boston, MA, USA, 02215

<sup>5</sup> Photonics Center, Boston University, Boston, MA, USA, 02215

<sup>#</sup> Contributing equally

<sup>\*</sup>Corresponding author: jxcheng@bu.edu

## Abstract

High-dimensional microscopy captures features across space, time, and spectrum, but complex backgrounds obscure signals. We present Background2Background (B2B), a selfsupervised background removal framework leveraging the background’s larger correlation relative to the signal. Local pixel shuffling disrupts signal correlations while preserving background structure, transforming signals into effective “noise” for Noise2Noise-based background removal. Across fluorescence, stimulated Raman scattering and mid-infrared photothermal imaging, B2B achieves over 40-fold background suppression and reveals hidden signatures.

## Main

Modern optical microscopy captures biological information across multiple dimensions, including depth, time, and spectrum. Volumetric and temporal fluorescence imaging<sup>1</sup> reveal three-dimensional cellular architecture and dynamic processes, while hyperspectral vibrational imaging<sup>2</sup> maps biochemical heterogeneity through molecular spectral signatures. However, these multi-dimensional datasets often contain complex backgrounds that vary across space, time, and spectrum. Such backgrounds can overwhelm signals and distort quantitative measurements, limiting the biological insights that can be reliably extracted.

Computational background removal can improve image contrast without increasing system complexity or acquisition time. Model-based methods<sup>3-6</sup> generally assume that background is confined to low spatial frequencies and require explicit modeling of the background frequency distribution. Such modeling becomes increasingly challenging in high-dimensional microscopy where background is heterogeneous in different domains. Learning-based algorithms offer greater flexibility but require paired training data and simulated backgrounds for supervised learning<sup>7,8</sup>. Mismatches between simulated and experimentally measured data can therefore limit their performance and generalizability.

Here, we introduce Background2Background (B2B), a self-supervised framework for removing heterogeneous backgrounds in high-dimensional microscopy. B2B exploits differences in correlation length between biological signals and backgrounds to recast background estimation as a denoising problem. It generates a pair of locally randomized images from the input, disrupting the short-range correlations of signals while preserving the longer-range correlations of the background. The pair thus shares background structure, with randomized signals acting as decorrelated “noise.” A network trained to predict one image from the other using Noise2Noise (N2N)<sup>9</sup> learns to recover the sSared background without requiring groundtruth. The estimated background is then used to correct the original images.

B2B enables label-free stimulated Raman scattering (SRS)<sup>10</sup> mapping of carbohydrates and nucleic acids in the fingerprint window, high-throughput mid-Infrared photothermal (MIP)<sup>11</sup> imaging of protein and lipid distribution in C. elegans, tracking of actin dynamics in wide-field fluorescence microscopy, and one-photon calcium imaging of neuron activity in a mouse brain.

We first explored the physical basis for distinguishing background from biological signals. In wide-field fluorescence, defocus broadens the point spread function (PSF), causing fluorescence from out-of-focus structures to spread over a larger lateral area. Together with diffusely distributed autofluorescence, this produces a spatially varying background that obscures in-focus signals (Fig. 1a, b). SRS is accompanied by competing optical heterodyne processes such as cross-phase modulation (XPM) with broad spectral response, while MIP contains broad water absorption peaks (Fig. 1c). These background sources are diffusely distributed throughout the sample, giving rise to spatially broad background signals. In general, the background exhibits a longer correlation length than signals (Fig. 1d), which builds a foundation for background elimination.

To implement B2B, we introduce local randomization (LRAND), which replaces each pixel with a randomly selected pixel within a local cube (Fig. 1e). Enlarging the cube size perturbs structures at progressively longer correlation scales, thereby increasing the Fourier-domain difference between the original and randomized stacks. Changes in the curvature of this difference therefore reflect transitions between characteristic scales. We use the second zero crossing of its second derivative as an estimate of the transition to the longer-range background and select a cube size that disrupts local signal correlations while largely preserving the background (Fig. 1f and Fig. S1). We then use a pair of independently randomized images to train a U-Net<sup>12</sup> within the N2N framework to estimate the background (Fig. 1g). Finally, we extract the signal by subtracting the estimated background from the raw images or dividing the raw images by it (Fig. 1h).

(a)  
![](images/f67e129a47e47cb34bebd2ef244501d6742df681d54c73c1612ba73e9f63d253.jpg)  
(c)

(b)  
![](images/ebbdeb44a0980caca50a93c70f0b40fbc6ffd759b10f727ec08159182c8899df.jpg)  
(e)

Local Randomization (LRAND): making signal as “noise”  
![](images/71209700e5b6feec0c0c8764bfe494d95e47600a0e687350b5207c50beb2971b.jpg)  
(g)

Noise2Noise: removing “noise”

![](images/d5e24a62e021d6fa55590a9327e58577db8ee24e1d8cea99e86fcd3e9a4b284e.jpg)

![](images/3c860761f8d341e4f887e538165e9ccddd544800a69225028c67027e131ad835.jpg)

![](images/735ad6cba6e6eec6613c47e44cbb6a356a9ae924afaa89973b22eaf55426ac7e.jpg)

![](images/f65684d64f2b5f22cf3ace89537f9c4c83192c869d33bc5a2bb2bfc88ec6ce90.jpg)

![](images/18ad984622e320f3f37450997e9933232722bc070494f05244a21b10badee627.jpg)

![](images/978791b4fd7ad0db9064820a23fd92625c3ff46d518ffc36df884584fe6019d0.jpg)

![](images/d2acd287158894ecd8428bd1ce1f38ecfe260e0ac02b4adaa41965356116da40.jpg)  
(h)  
Signal extraction:

![](images/d239a54f040982cc6bc16f5c0582de23166c856b769d2c8c94d80744a53de74d.jpg)

(i)  
![](images/f8cfb20e2f5648fc98aa6225928c239baf75738b4c26a76f495e021086c4e7fe.jpg)

![](images/f725e68d177e4440e85a19c4aaabe52d724e7a8e794f0b6338177498cef994bc.jpg)

![](images/a1d326a3bdfb24e6027c74df92a60fca202e1f540b139fb0b435c6b6d65096f8.jpg)

![](images/870c97ed532638493eb3cf473c67497220a75172dcc848ab5284dd08a23aaf3f.jpg)

(k)  
![](images/b6266eebf2987d95b9009eb4a3b10b165995069662ea86693220d5f59da0f28e.jpg)

(1)  
![](images/e21ca4fe2374b5cc3fa264c487bd7ca7706573f310e79c13dd176ffa30bc9e9c.jpg)

![](images/6f96be8ae36a1d4304e8873ee812fbae2630951684050305d4da6f19424f0f85.jpg)

![](images/f8eb4647d8704d1cd76c6e348fcb4c5ac038eddb3d738d05b41196b4c379e68f.jpg)

(m)  
(n)  
![](images/2ed4e8db6aebe3fd6d1444f107ef0bdd3a35eb59764756643b50b0e9ba16ee34.jpg)

![](images/7292be97e193e53ede0dc64d40166ef42117f89a7b930d0274ea74b0fbf92e93.jpg)

(0)  
![](images/8ccafa0fc36510d6195a68f45c85cf7c40614d62e2a13715f975a80878436d31.jpg)

![](images/3e1d8157ce492e107fc4feb770c617e3fb907aee56b9c5803bffb8ecf4974719.jpg)

![](images/1d0b037c64a669d43b27282e6d930bdb988dc7beded78055c99f6e052b0673d0.jpg)  
(p)  
Figure 1. Principle and validation of B2B.  
Diff MIP

![](images/c2f787025afb900dde2c9460abc882002dba803b7eeb1799eb174fea91477720.jpg)

(q)  
![](images/632eca1ca691712260f12f7fd4c36f0bf479508368fde49eccb0f8681a116a84.jpg)

(a) Comparison of in-focus and out-of-focus PSFs. (b) Distribution of target dye and autofluorescence (AF). (c) Representative spectral profiles of two chemical bands (C1 and C2) and a broader background (BKG). (d) Schematic comparison of the characteristic correlation lengths of signal and background.

(e) Local randomization (LRAND) operation. (f) Selection rule of optimal cube size. (g) N2N to estimate the background. (h) Signal extraction.

(i) Raw and B2B-processed wide-field (WF) and 2D structured illumination microscopy (2D-SIM) images, with 3D-SIM as the reference. (j, k) SSIM and PSNR evaluation.

(l) Conventional SRS, B2B-processed SRS and donut-SRS reference images. (m) Spectrum of lipid droplet (LD) and background region. (n) SBR evaluation.

(o) Single-wavenumber MIP, B2B-processed MIP, and differential-MIP reference images. (p) Spectrum of selected ROIs. (q) SSIM and RMSE evaluation.

We validated B2B using volumetric fluorescence, hyperspectral SRS, and hyperspectral MIP imaging. For fluorescence, B2B was applied to widefield and 2D structured illumination microscopy (SIM) stacks, achieving 75x and 60x background removal respectively. We used 3D-SIM as the reference<sup>6</sup> (Fig. 1i) and quantified improvements using the structure similarity index measure (SSIM; Fig. 1j) and peak signal-to-noise ratio (PSNR; Fig. 1k). We next evaluated B2B on silent region hyperspectral SRS imaging of PA-d31 treated cell (Fig. 1l), using donut-SRS<sup>13</sup> as a reference. B2B reduced the background 40-fold and recovered the PA-d31 spectrum (Fig. 1m). It achieved a signal-to-background ratio (SBR) of 46, compared with 27 for donut-SRS (Fig. 1n). This advantage reflects B2B’s adaptive removal of spatially heterogeneous XPM backgrounds, whereas donut-SRS applies uniform subtraction. B2B also outperformed rolling ball and sliding paraboloid algorithms (Fig. S2), which failed to recover signals in lipid rich cytoplasm area (Fig. S3). Finally, we evaluated B2B using hyperspectral MIP imaging of azide treated cell, which provides an isolated Infrared (IR) peak in the cell silent window. A previously reported differential MIP approach based on on- and off- resonance measurements served as the reference<sup>14</sup> (Fig. S4). B2B enabled background reduction over 150 times and closely reproduced the reference chemical maps and spectrum (Fig. 1o, p), as further statistically quantified by SSIM and root mean square error (RMSE) (Fig. 1q).

Because B2B does not require high-SBR training data, it is well suited for discovery-driven research, as demonstrated by label-free fingerprint SRS imaging of carbohydrates in single cells. Raman bands near $1 1 0 0 \ c m ^ { - 1 }$ provide distinctive chemical contrast but have low cross sections and are often obscured by noise and background. We therefore combined the selfsupervised denoiser ${ \mathsf { S P E N D } } ^ { 1 5 }$ with B2B to recover these weak features. The 1048 cm<sup>-1</sup> band, assigned to C-O stretching in carbohydrates, primarily from glycogen<sup>16</sup> (Fig. 2a, b), revealed distinct spatial distributions from that at the C-C skeletal stretching band near 1125 cm<sup>-1</sup> (Fig. 2c). Carbohydrate-rich pools were located near lipid droplets, consistent with electron microscopy observations<sup>17</sup>. Spectra from these regions are shown in Fig. 2d. SPEND alone reduced random noise but did not resolve these spatial and spectral features (Fig. S5 and S6). The peak near $1 0 9 0 ~ { \mathsf { c m } } ^ { - 1 }$ in the nucleolus (Fig. S7) is attributed to ${ \tt - P O } _ { 2 } { } ^ { 1 8 }$ associated with nucleic acid and other phosphate-containing biomolecules. The SRS images in the highwavenumber C-H window provided a morphology reference (Fig. S8).

Hyperspectral MIP mesoscope provides high-throughput biochemical imaging over large field of view $( \mathsf { F o V } ) ^ { 1 9 }$ . However, nonuniform illumination and water absorption introduce spatially and spectrally varying backgrounds that obscure chemical contrast. B2B removed this background across an entire worm, revealing protein (amide II and amide I) and lipid (ester band) distributions (Fig. 2e) and their overlay (Fig. 2f), with clearly resolvable spectral bands (Fig. 2g, h). B2B thus combines high-throughput mesoscopic imaging with local molecular specificity to produce chemically interpretable maps of large samples.

![](images/d4d0b2445d9050c896ed0d8edea833c7653c237c65265b02133a4e04487c306f.jpg)

(a)  
SRS  
![](images/f64ffaaffd5d29eb6ac1f4fbaa6da8ee973aa5c389dcb3a2925ed4ba91793d28.jpg)

![](images/65b79ee31238c7ac114b2fb62789ae47bedfd60c5fb6cc778a525e6fa23c59c0.jpg)

![](images/cebfe7dc01992005e68ccd1cd84424bfce9d73c5d9073c0b30fe9e0625d43d02.jpg)

(d)  
![](images/5dab3d108744e3d00cb323041ba6d9e6c03c23e8e4425ee05e21a0beae1701b8.jpg)  
Wavenumber (cm-1)

(e)  
![](images/6447be7339ff3b3a0302070b16cb285932f09d978cf34c2600bd1f1d2ebe073f.jpg)

![](images/43508f1bfd58500d8f331210bd30c374ad966067d360aeb18923c71f1ec9f30b.jpg)  
Lipid

(f)  
![](images/1671ff5406fec23b9db96f9b91291c02e26f08831e028c52b9221e8bc047815e.jpg)  
Merge

(g)  
(h)  
![](images/fe79be2225d2bf0b730b2f75856134e1aafb67eb954ff8c08e74a88355fcbcf3.jpg)

![](images/4ecbcb9b765806cd3f5f2ef5f8fbe36c57c6169d92bc9d85cd7a639bb8fd9419.jpg)

![](images/fbe0f3d3502bbd077d15c0fe349034ae01b5e7d2c948c43cb7b2c516d7b4f78e.jpg)

![](images/156fd70e29dc43b04c8a1305d5fb99cb0ed9db2a5358ad97b4db3c64b66bb7d8.jpg)

![](images/0f2b69edfb5bfde58018f3987c7e8e87d921bee22ef27d8b5f5ac5e056d44f4e.jpg)

Wavenumber (cm-1)  
(1)  
![](images/4ea7ee1f48166a1f413b0e6b30141f410725a4679413f62f1a92f88882da6d58.jpg)  
1 Photon

![](images/cebfd059f7e6d7c82a05395956268c011e4c13c8897077892b985bafe354dee1.jpg)  
B2B

![](images/6a2d90fc49948c4db0e8ecf3155b51bcf58d2b2c0df0e23e25bb644df1f5f078.jpg)  
2 Photon

(m) - 1P — B2B -2P  
![](images/099527fc716fda482c59be6b6a7cc3d465c5ad1b12f706456382b4135dd8f49c.jpg)

(n)  
![](images/bf2fb6a69eaf4be1e28f5eace133f859cdea392b586826ca1dfbd78bd24b686e.jpg)

## Figure 2. Applications of B2B to SRS, MIP, and fluorescence modalities.

(a) Raw and SPEND denoised SRS image at $1 0 4 8 \mathsf { c m } ^ { - 1 }$ . (b) B2B processed SRS image at $1 0 4 8 \mathsf { c m } ^ { - 1 }$ (c) B2B processed SRS image at 1125 cm<sup>-1</sup>. Magnified views of the boxed region at 1048 cm<sup>-1</sup> and 1125 $\mathsf { c m } ^ { - 1 }$ are shown. Scale Bar: $3 \ : \mu m .$ . (d) B2B-corrected spectra from lipid-rich, carbohydrate (Carb.) -rich and nucleolus region.

(e) Raw and B2B-processed Amide II, I and lipid channels. (f) Merged chemical map after B2B processing. (g, h) Spectra from ROI 1 and 2 before and after B2B processing.

(i) Temporal color-coded projections before and after B2B processing. (j, k) Magnified views of the indicated region before and after B2B processing.

(l) Max intensity projection of single photon, B2B processed and two photon calcium imaging. (m)

Temporal profiles of selected neurons. (n) Pearson correlation analysis of single photon and B2B results.   
P=4.9e-7. 1P: 1 photon. 2P: 2 photon.

Wide-field fluorescence microscopy is widely used to study biological systems, but out-offocus emission and autofluorescence can obscure signals. Here, we applied B2B to wide-field recordings of propagating actin waves (Fig. 2i), revealing their spatiotemporal dynamics<sup>5</sup>. A representative frame is shown in Fig. S9. Temporal features buried by background were visible after B2B (Fig. 2j, k). Finally, we applied B2B to retrieve neuronal activities in wide-field calcium imaging<sup>7</sup>. B2B improved the localization of active neurons (Fig. 2l) and recovered their temporal activity profiles (Fig. 2m). Correlation analysis showed increased agreement with the two-photon reference, further supporting B2B’s ability to recover neuronal activity (Fig. 2n).

Together, these results establish B2B as a broadly applicable framework for background suppression across diverse high-dimensional images, enabling applications ranging from label-free chemical mapping to fluorescence imaging of biological dynamics. The current implementation uses a deterministic estimator and does not explicitly model uncertainty in the recovered background. Incorporating probabilistic or generative models may further improve robustness in datasets with more complex or overlapping signal and background distributions.

## Methods

## Hyperspectral SRS microscope

A custom-built hyperspectral stimulated Raman scattering microscope is used to perform hyperspectral SRS imaging. A femtosecond pulse laser source (Insight, DeepSee+, spectra-Physics) operating at 80MHz provided two synchronized outputs: a tunable pump beam ranging from 680nm to 1300nm and a fixed stokes beam at 1040nm. The pump beam is tuned to 800nm for the C-H region, to 931nm for 1100 cm<sup>-1</sup> fingerprint region. The Stokes beam is modulated at 2.5MHz by an acoustic optical modulator (1205-C, Isomet) and chirped with a 15cm glass rod (SF57, Schott) before the merging of two beams. After spatial and temporal recombination, the two beams were further chirped using five glass rods to obtain picosecond pulse durations. A motorized delay stage controlled the time delay between the pump and Stokes pulses, thereby tuning the Raman shift through spectral focusing.

The collinear beams were scanned using a 2D Galvo scanner (GVS102, Thorlabs) and focused onto the sample through a 60X water immersion objective (NA=1.2, UPlanApo/IR, Olympus). The transmitted light was collected by an oil condenser (NA=1.4, U-AAC, Olympus). A photodiode (S3994-01, Hamamatsu) is used to collect signals after blocking the Stokes beam. Stimulated Raman gain signals were demodulated at 2.5 MHz using a lock-in amplifier (UHFLI, Zurich Instrument).

## MIP Mesoscope

Large-Field of View (FoV) MIP imaging was performed using a previously reported lab-built MIP mesoscope<sup>19</sup>. The MIP mesoscope operates in a co-propagating configuration, in which the mid-infrared pump and visible probe beams are combined and focused onto the sample through the same objective. Mid-IR excitation is provided by a QCL (MIRcat 2400, Daylight Solutions) outputting pulses at 600 kHz with 300 ns pulse duration, which induces a transient thermal modulation. The IR beam is combined with a 532-nm continuous-wave visible probe beam (Samba 532, Cobolt) via a dichroic mirror. The combined beam is scanned by an X–Y pair of galvanometric mirrors (Saturn 1B, ScannerMAX). A pair of concave mirrors $( f _ { S L } =$ 200 �� , CM508-200-P01, Thorlabs; $f _ { T L } = 5 0 0$ �� , CM750-500-P01, Thorlabs) relay the scanned beam to the back pupil of a 15×/0.4 NA reflective objective (50105-01, Newport), expanding the beam diameter by 2.5× to fill the objective aperture. The all-reflective scanning relay eliminates chromatic aberrations, enabling IR spectroscopic imaging without realignment across visible and IR wavelengths. The probe beam, carrying the photothermal modulation, is collected by a 20×/0.45 NA refractive objective (AMEP4763EO, Olympus) with an axial offset and detected by an amplified photodiode (PDA100A2, Thorlabs). A bias tee (ZFBT-4R2GW+, Mini-Circuits) separates the DC and AC components of the photodiode output. The AC signal is demodulated using a lock-in amplifier (HF2LI, Zurich Instruments) to recover the photothermal signal.

## B2B training and prediction

To enhance the robustness of our model, we augment the datasets by 4-fold through flipping and rotating at 180 degrees. After augmentation, the training set is comprised of 36 stacks, with 10% for validation and 90% for training. A 4-layer Unet architecture based on the CSBDeep framework is employed. The training was conducted on a commercial graphics processing unit (GPU, RTX 4090, Nvidia). Training time is different based on different datasets.

## Open-source data

WF fluorescence, 2D-SIM, 3D-SIM data in Fig. 1i is from Dark-based Optical sectioning<sup>6</sup>. SRS dataset in Fig. 1l is from Donut-Shaped detection SRS<sup>13</sup>. MIP imaging of Azide in Fig.1o is from MIP imaging of Azido photothermal probe (App)<sup>14</sup>. WF fluorescence video in Fig. 2i is from Background Filter SIM<sup>5</sup>. Calcium imaging of neuron in Fig. 2l is from DeepWonder<sup>7</sup>.

## Sample preparation

U87 cells used in Fig. 2a-d were seeding in a 35mm glass-bottom dish overnight. Cells were incubated within the medium that contains analogs at a concentration of 20�� for 48 hours. Cells were fixed with 10% neutral buffered formalin for 30 minutes and then washed with phosphate-buffered saline (PBS) three times.

C. elegans used in Fig.2e-h were cultured on nematode growth medium (for 1L: phosphate buffer 20mM, CaCl2 0.8mM, MgSO4 0.8mM, agar 17.5g, NaCl 3.0g, peptone 2.5g, cholesterol 0.005g) and grown at 15°C. Plates were seeded with OP-50 Escherichia coli. For imaging, young adult worms (1-3 days old) were mounted via picking into 1% NaN3 in M9 buffer on a 24x60mm #1 thickness cover slide. Double-sided tape was used as a spacer, and another #1 thickness cover slide was lowered over the top of the tape to sandwich the sample.

## Acknowledgements.

This work is supported by NIH grants R35GM136223, R01 EB032391, and R01 EB035429 to JXC.

## Author contributions.

G.D., H.L., and J.C. co-designed the project. G.D. proposed the B2B framework. G.D., H.L., R.T., B.W., J.A., Q.X., J.Y., and D.S. conducted the SRS and MIP experiments. G.D. and G.H. conducted the fluorescence imaging experiment. L.T., H.L., and J.C. supervised the project. G.D., H.L., and J.C. wrote the manuscript. All authors read the manuscript.

## Data availability.

The datasets have been made publicly available on https://github.com/buchenglab/B2B

## Code availability.

The original codes have been made publicly available on https://github.com/buchenglab/B2B

## References

1 Mertz, J. Introduction to Optical Microscopy. 2 edn, (Cambridge University Press, 2019).

2 Cheng, J.-X., Min, W., Ozeki, Y. & Polli, D. Stimulated Raman scattering microscopy: Techniques and applications. (Elsevier, 2021).

3 Rodrigues, M. C. M. & Militzer, M. Application of the rolling ball algorithm to measure phase volume fraction from backscattered electron images. Materials Characterization 163, 110273 (2020). https://doi.org:https://doi.org/10.1016/j.matchar.2020.110273

4 Schindelin, J. et al. Fiji: an open-source platform for biological-image analysis. Nature Methods 9, 676-682 (2012). https://doi.org:10.1038/nmeth.2019

5 Mo, Y. et al. Quantitative structured illumination microscopy via a physical model-based background filtering algorithm reveals actin dynamics. Nature Communications 14, 3089 (2023). https://doi.org:10.1038/s41467-023-38808-8

6 Cao, R. et al. Dark-based optical sectioning assists background removal in fluorescence microscopy. Nature Methods 22, 1299-1310 (2025). https://doi.org:10.1038/s41592-025-02667-6

7 Zhang, Y. et al. Rapid detection of neurons in widefield calcium imaging datasets after training with synthetic data. Nature Methods 20, 747-754 (2023). https://doi.org:10.1038/s41592-023-01838-7

8 Li, B. et al. Deep-3D microscope: 3D volumetric microscopy of thick scattering samples using a wide-field microscope and machine learning. Biomedical Optics Express 13, 284-299 (2022). https://doi.org:10.1364/BOE.444488

9 Lehtinen, J. et al. in Proceedings of the 35th International Conference on Machine Learning Vol. 80 (eds Dy Jennifer & Krause Andreas) 2965--2974 (PMLR, Proceedings of Machine Learning Research, 2018).

10 Freudiger, C. W. et al. Label-Free Biomedical Imaging with High Sensitivity by Stimulated Raman Scattering Microscopy. Science 322, 1857-1861 (2008). https://doi.org:doi:10.1126/science.1165758

11 Zhang, D. et al. Depth-resolved mid-infrared photothermal imaging of living cells and organisms with submicrometer spatial resolution. Science Advances 2, e1600521 (2016). https://doi.org:doi:10.1126/sciadv.1600521

12 Ronneberger, O., Fischer, P. & Brox, T. in International Conference on Medical image computing and computer-assisted intervention. 234-241 (Springer).

13 Ao, J., Zhu, Y., Teng, X., Zhang, M. & Cheng, J.-X. Background-Free Stimulated Raman Scattering Imaging via Donut-Shaped Detection. Chemical & Biomedical Imaging (2026). https://doi.org:10.1021/cbmi.6c00031

14 Xia, Q. et al. Click-free imaging of carbohydrate trafficking in live cells using an azido

photothermal probe. Science Advances 10, eadq0294 (2024). https://doi.org:doi:10.1126/sciadv.adq0294

15 Ding, G. et al. Self-supervised elimination of non-independent noise in hyperspectral imaging. Newton 1 (2025). https://doi.org:10.1016/j.newton.2025.100195

16 Pastrana-Otero, I., Majumdar, S., Gilchrist, A. E., Harley, B. A. C. & Kraft, M. L. Identification of the Differentiation Stages of Living Cells from the Six Most Immature Murine Hematopoietic Cell Populations by Multivariate Analysis of Single-Cell Raman Spectra. Analytical Chemistry 94, 11999-12007 (2022). https://doi.org:10.1021/acs.analchem.2c00714

17 Mayeuf-Louchart, A. et al. Glycogen Dynamics Drives Lipid Droplet Biogenesis during Brown Adipocyte Differentiation. Cell Reports 29, 1410-1418.e1416 (2019). https://doi.org:https://doi.org/10.1016/j.celrep.2019.09.073

18 Wiercigroch, E. et al. Raman and infrared spectroscopy of carbohydrates: A review. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy 185, 317-335 (2017). https://doi.org:https://doi.org/10.1016/j.saa.2017.05.045

19 Tang, R. et al. Mid-Infrared Photothermal Mesoscopy with Millimeter Field of View and Sub-micron Spatial Resolution. bioRxiv, 2025.2011.2003.686410 (2025). https://doi.org:10.1101/2025.11.03.686410

# Background2Background: a self-supervised background removal framework for high dimensional microscopy

Guangrui Ding<sup>1,5,#</sup>, Haonan Lin<sup>2,#</sup>, Rong Tang<sup>1,5</sup>, Bethany Weinberg<sup>3,5</sup>, Jianpeng Ao<sup>1,5</sup>, Qing Xia<sup>1,5</sup>, Guorong Hu<sup>1,5</sup>, Dingcheng Sun<sup>3,5</sup>, Jiaze Yin<sup>1,5</sup>, Lei Tian<sup>1,3,5</sup>, Ji-Xin Cheng<sup>1,3,4,5\*</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, Boston University, Boston, MA, USA, 02215

<sup>2</sup> The Wallace H. Coulter Department of Biomedical Engineering, Georgia Institute of Technology and Emory University, Atlanta, GA, 30332

<sup>3</sup> Department of Biomedical Engineering, Boston University, Boston, MA, USA, 02215

<sup>4</sup> Department of Chemistry, Boston University, Boston, MA, USA, 02215

<sup>5</sup> Photonics Center, Boston University, Boston, MA, USA, 02215

<sup>#</sup> Contributing equally

<sup>\*</sup>Corresponding author: jxcheng@bu.edu

(a)  
![](images/1fa71878b3b35d4ee087a412ae03a8edaab1a9f96a3bcf6b3e9a674644058fbc.jpg)  
(b)

![](images/c5920382ce4c6cc072179622830f545f9f44b5446a53415672f9a7dac55f6812.jpg)  
Img1(r0) = LRAND(Img0|r = r0)

![](images/869e2134879715fb4fba319e0df5ee3b18f03daff221b974909cbcc2d2dc4d2c.jpg)

![](images/251f1fef8311940988d89aa4a600f1047bb0c0012dc5b4934f2c1110d2eab746.jpg)

(d)  
![](images/2b7b2d4b8935c265478dcf6aa7a0dee5a965a80e078b4d61f843c389bd2008a3.jpg)  
Figure S1. Determination of the optimal LRAND radius $\mathsf { R } _ { 1 } .$ (a) LRAND applied to the raw image using different radius r. (b) Calculation of Fourier domain difference between the original and randomized images, together with its first and second derivative with respect to r. (c, d) Example of Fourier domain difference and its corresponding first derivative.

(a)  
![](images/0975489b4c262b407a1b2ba3e0de8a44ee177053835a50a0ad73bf7363eddf6c.jpg)

(b)  
![](images/d31b9fd9c7e2c684cc02e3d14e5b7704912336361b6df82e298d59db6df2c145.jpg)  
SRS

(c)  
![](images/8d643fe25296e43feb01b60c6da83cc706a7f707b76fbbe74cc96268e1fb2cac.jpg)  
Donut-SRS  
B2B

(d)  
![](images/457ec4b5b9a7d2d94b95c717d37f1fb079037b1ee95d19291a3a62608fc0cfdb.jpg)  
Figure S2. Spatial performance of rolling ball and sliding paraboloid methods. Scale bar: 10 ��.

(e)  
![](images/2847d1591236909ac66f381db0e3f43148c8bbbf8a4a09fbb27270275b7a8d0a.jpg)  
Sliding Paraboloid

![](images/65f33e987268d6cfd2b901ae96c5298915507a41ad8cf9b1d1b70bdd5183b605.jpg)

![](images/1d81764d5e4dd7c8364c938bb321870a44776f52e33fee4f251769e7f1e37512.jpg)

(c)  
![](images/b55958ba960177bd12221707ca51c726d3d735652fb3c48cd7a285e7fce73ec3.jpg)

![](images/90ca236ae47e880c293280fcbe078fc9cf80a45e57d4c7ddc390053b1343e1d7.jpg)

![](images/e1a1ad0e767f701a70eed3b6e1d810eadeb45e3e6ad9d995d058d5c4bfdb90b6.jpg)  
Figure S3. Spectral results of each method.

![](images/b391695f86586fd06501eef2bc38d1689950676e330f984d75d797429e933865.jpg)  
Figure S4. Azide signal extraction via subtraction of images at three different wavenumbers. (a) Intensity-normalized spectrum of Azide in silent region. (b) subtraction process.

1048 cm-1  
![](images/44f1a13c5537cf7ff38de32495b8003ea922e567fbb9d0f61a115aac42548dd2.jpg)

(b)  
1125 cm-1  
![](images/1aebc5ef5b361f876925a71592eb73e54fea8fde33aea406799ee3f747301aa7.jpg)

![](images/a31021479654370a26e41786ec50ed1e82c7594f81dcb5eb8b6366deac701b91.jpg)

(d)  
![](images/216a7e27429e4c298ec9bc26bc6cc16e4d2e8bc1ce08373d0f983fdc67ae9ccf.jpg)  
Figure S5. Comparison of raw and SPEND processed SRS image. (a, b) Raw SRS images at 1048 $\mathsf { c m } ^ { - 1 }$ , (b) 1125 $\mathsf { c m } ^ { - 1 }$ . (c, d) SPEND processed SRS images at (c) $1 0 4 8 ~ \mathsf { c m } ^ { - 1 }$ , (d) 1125 cm<sup>-1</sup>. Scale Bar: 1 $\mu m$

(a)  
![](images/d7d9020c4456399b2d96078194e08f96ee9117a7d26db754eda688596b47c2e1.jpg)

![](images/7e982f01d2d70cca4d38e8d3a09ecce446eae7d7bd0801c0f2d4a76303bf2509.jpg)  
Figure S6. Spectrum profile of selected regions. (a) Raw. (b) SPEND processed.

(a)  
![](images/0dbe9a7f048253f29c3ac6fac0d1912c67c59631408c10c85164dbc795187b25.jpg)

(b)  
![](images/ea4a1426fc57cd1bbe2a5d0b08ee36a09ab3206479d917e3463ef96bb593c4ab.jpg)  
Figure S7. SRS image at $1 0 9 0 \ c m ^ { - 1 } ,$ . (a) Raw and SPEND processed images. (b) B2B processed image. Scale bar: $1 0 \mu m$

![](images/7bc0591b3ba52df84d00cf00b9d87a27fb93c01ca138692479e7f8f0836b6d1b.jpg)  
Figure S8. Stack average of hyperspectral SRS imaging of U87 in CH region. Scale bar: 10 ��.

(a)  
![](images/9560653aa23f80d3c4b8ad533daa150508778586b41810fc6d9b7db5a90414c7.jpg)

(b)  
![](images/0422baf493d3403e42008de6140bced1cf4753d4f2bc7a51a9b00aacdb0a2df3.jpg)

(c)  
![](images/87123de6b0be06bd0de3d1bef1d3edb51a703c2f9d3cfb0ca6b14a57a5ce5c66.jpg)

![](images/297bbfc78e16f6cf95f849e7a295a8bbab6b60a74e7dddb4219b1edc3307ff2f.jpg)  
Figure S9. Single frame comparison of widefield fluorescence images before and after B2B processing. (a) Raw, (b) B2B at 20s. (c, d) Magnified views of selected regions in a and b respectively. Scale bar: (a) 5 ��. (d) 500 ��.