# Beyond CCT: Multi-channel SPD recovery for per-shot camera color correction

Shuwei Yue<sup>1,∗</sup>, Hongke Wei<sup>2</sup>, and Qiao Chen<sup>1</sup>

<sup>1</sup>Shenzhen Polytechnic University, Shenzhen, China

<sup>2</sup>Arashi Vision Inc. (Insta360), Shenzhen, China

<sup>∗</sup>Corresponding author: shuwei.yue@connect.polyu.hk

## Abstract

Modern cameras correct color via two or three pre-calibrated 3×3 matrices interpolated by Correlated Color Temperature (CCT). This collapses the illuminant SPD to a scalar. LEDs, fluorescents, and mixed sources sit off the Planckian locus, and metameric illuminants with the same chromaticity demand different optimal CCMs that a CCT pipeline cannot distinguish. We recover the illuminant SPD instead — non-negative least squares against a basis dictionary, conditioned on the camera whitepoint and a multi-channel ambient-light sensor (ALS). A closedform 3×3 CCM is then fit per shot. On a synthetic benchmark of 300 illuminants ×3,100 reflectances, mean angular error drops from 1.89<sup>◦</sup> to 1.20<sup>◦</sup>, with a 1.7× tail reduction on the LED-like off-Planckian group (n=45) where metameric ambiguity is tangible. The gain holds under realistic ALS imperfections and observation noise, and the same off-Planckian advantage carries over to 1,684 real measured SPDs.

## Introduction

The Color Correction Matrix (CCM) is a 3×3 linear map from white-balanced camera RGB to a colorimetric standard (CIE XYZ or sRGB). The optimal CCM depends on the scene illuminant. The standard Adobe DNG pipeline [1] stores two reference CCMs, one at T =2856 K and one at T =6500 K. At runtime it interpolates the two by the estimated Correlated Color Temperature (CCT). The illuminant SPD E(λ) — a function over wavelength — is collapsed to a single scalar.

This works on Planckian illuminants. Blackbody radiators are fully described by temperature, so CCT and the SPD are equivalent. Modern indoor and architectural lighting often sits off this locus. LEDs, fluorescents, and mixed sources do not follow the blackbody curve. Two sources can share xy chromaticity to within $\Delta _ { x y } { < } 0 . 0 0 5$ yet differ in normalized SPD shape by more than 30<sup>◦</sup>. Both drive the same white-point reading and the same CCT estimate. They still need different optimal CCMs. CCTbased interpolation cannot tell them apart.

We present a spectral-aware per-shot CCM that replaces the CCT scalar with a recovered SPD. We recover the SPD from two cheap observations: the camera’s own white-point response and a multi-channel ambient-light sensor (ALS). The ALS is a small commodity chip with 8 to 14 narrow-band photodiodes, already used in mobile phones for ambient sensing. With K+3 observations against a small SPD basis, recovery is a non-negative least squares problem. The recovered SPD then synthesizes a virtual color chart, and we fit a scene-specific CCM to that chart in closed form.

![](images/047ae3148c4e29dffe9810c559540e03a82cf7e4d3c6dfa56fc3f716e5b8d19a.jpg)

![](images/816dc0d4c140df647760735cc179fef2541f8ab2f8204740038f8fb2662836bf.jpg)  
Figure 1. The metameric problem and our fix. (a) Two real EMPIR illuminants share xy chromaticity to within ∆ =0.003 but differ in normalized SPD shape by 43<sup>◦</sup>. A CCT-driven CCM cannot tell them apart, yet they need different optimal CCMs. (b) On the LED-like Duv group of our benchmark (n=45, statistically meaningful), our spectral pipeline cuts worst-25% angularerrorfrom 5.3<sup>◦</sup> (Adobe DNG) to 2.8<sup>◦</sup> — a 1.9× reduction. A more extreme group (|Duv|≥0.03, n=5) shows a larger directional gain that we report only for context.

Figure 1 previews the argument: a real metameric pair on the left, the worst-case error reduction on the right. The full method comparison and ablations are in Sec. 4; a real-data sanity check on the WP-only fallback (without ALS hardware) is in Sec. 4.5.

Contributions The non-parametric NNLS form lifts CCM conditioning from chromaticity to K+3 dimensions. Against six baselines on 300 EMPIR illuminants ×3,100 CAVE reflectances, the spectral advantage is most reliable on the LEDlike off-Planckian group (n=45), where worst-25% error drops by 1.7× (4.79<sup>◦</sup> → 2.80<sup>◦</sup>) relative to the strongest non-spectral baseline (CST-MLP <sub>×6</sub>, the recent xy-MLP of Tedla et al. [2]); a smaller extreme subset (n=5) shows a larger directional gain. A channel-count ablation finds a sharp knee at K=8, matching commercially available multispectral sensors [3]. A metamericpair stress test isolates SPD-recovery accuracy from downstream CCM behavior.

## Related work

Three lines of work bear on a spectral-aware CCM. Classical color constancy estimates an illuminant chromaticity but stops short of the CCM stage. CCT- and xy-conditioned CCM pipelines scalarize the illuminant to one or two numbers, which a metameric pair defeats by construction. Spectral-recovery work tackles reflectance rather than the illuminant or the CCM. We position our method against each in turn.

Classical color constancy Statistical priors — gray-world [4], gray-edge [5] — and chromaticity-prior approaches that exploit saturated colors [6] estimate an illuminant chromaticity but stop there; downstream CCM remains a separate, illuminantconditioned step.

CCT-based CCM The DNG two-CCT pipeline [1] and Karaimer & Brown’s three-CCT extension [7] both parameterize the illuminant in 1D for the CCM stage. We treat upstream AWB as orthogonal to the CCM-stage spectral conditioning we address here.

Beyond CCT Tedla et al. [2] share our diagnosis that off-Planckian sources need richer conditioning than CCT, and use xy chromaticity (2D) as input to a small MLP that predicts a $3 \times 3$ or $3 \times 6$ root-polynomial [8] mapping. Their input is still the white point projected to chromaticity, so two SPDs sharing xy (a metameric pair) yield identical CCMs by construction. We lift the conditioning dimension from 2 to $K { + 3 }$ where K is the ALS channel count: a multi-channel sensor resolves the spectral shape that the xy projection discards.

Spectral upsampling and HSI Recovering reflectance spectra from RGB [9] and learning illuminant SPDs from hyperspectral data [10] solve neighboring problems; we recover the illuminant SPD given known camera response and an auxiliary multichannel observation.

## Method

We treat per-shot CCM design as a spectral-recovery problem. Each shot follows three stages. First, we recover an illuminant SPD E<sup>ˆ</sup> from the camera white point and a multi-channel ALS reading by a small non-negative least squares against a fixed SPD basis. Second, we plug $\bar { E }$ into the imaging equations to synthesize a virtual 24-patch chart in raw camera RGB and reference CIE XYZ. Third, we fit a 3×3 CCM to that chart in closed-form least squares, apply it to the captured pixels, and chromatically adapt to a D65 reference frame. The rest of this section formalizes each stage and the ALS filter bank.

Forward model We discretize wavelength on [400,700] nm at 10 nm $( N _ { \lambda } { = } 3 1 )$ . Let $E \in \mathbb { R } ^ { N _ { \lambda } }$ be the unknown SPD; let $\mathbf { S \in }$ $\mathbb { R } ^ { N _ { \lambda } \times 3 }$ be the camera SSF, $\mathbf { C } \in \mathbb { R } ^ { N _ { \lambda } \times 3 }$ the CIE 1931 CMFs, A ∈ $\mathbb { R } ^ { N _ { \lambda } \times K }$ the ALS filter bank, and $\mathbf { R } \in \mathbb { R } ^ { N _ { \lambda } \times 2 4 }$ a fixed 24-patch ColorChecker-style reflectance set. The camera white-point and ALS observations are wavelength integrals of E against the SSF and ALS filters, respectively. Per-patch camera RGB and CIE XYZ are the same integrals weighted by the patch reflectance.

NNLS recovery over an SPD basis We constrain E to the nonnegative span of a basis $\mathbf { B } \in \mathbb { R } ^ { N _ { \lambda } \times M } .$ To build B, we maxnormalize each EMPIR PhotoLED SPD [11] so basis atoms describe shape, not magnitude; absolute scale is absorbed into the recovered coefficients $\hat { \alpha } ,$ and we observe condition numbers $\kappa ( \mathbf { F } ) < 5 0$ across the test illuminant set. We pick M=30 representative atoms by k-means on a held-out 200-illuminant pool for the main experiments; Table 2(a) shows results are insensitive to $M \in [ 1 0 , 1 0 0 ]$ . Stacking the camera white-point and ALS readings into $\mathbf { y } { = } [ \mathbf { a } \mathbf { l } \mathbf { s } ; \mathbf { \bar { w } } { \mathbf { p } } ] ^ { \top } \in \mathbb { R } ^ { \Breve { K } { + } 3 }$ and the corresponding forward operator $\mathbf { F } { = } [ \bar { \mathbf { A } } ^ { \top } \mathbf { B } ; \mathbf { S } ^ { \top } \mathbf { B } ]$ , we solve

$$
\hat { \boldsymbol { \alpha } } = \arg \operatorname* { m i n } _ { { \boldsymbol { \alpha } } \geq 0 } \| { \bf F } { \boldsymbol { \alpha } } - { \bf y } \| _ { 2 } ^ { 2 } , \quad \hat { E } = { \bf B } \hat { \boldsymbol { \alpha } } .\tag{1}
$$

We use the active-set NNLS in scipy.optimize.nnls. Nonnegativity alone regularizes the inversion; we add no Tikhonov term. Train and test illuminant sets are disjoint and grouped by |Duv| (Sec. 4.1).

Per-shot virtual calibration With $\hat { E }$ in hand, we synthesize a virtual 24-patch chart by plugging E<sup>ˆ</sup> into the forward model. For each patch $k ,$ we obtain a synthesized camera RGB $\hat { \mathbf { r } } _ { k }$ and CIE $\mathbf { X Y Z } \hat { \mathbf { x } } _ { k }$ as wavelength integrals of $\hat { E } \cdot \mathbf { R } _ { k }$ against S and C respectively. We fit $\mathbf { M } \in \bar { \mathbb { R } ^ { 3 \times 3 } }$ as an XYZ→RGB CCM by closed-form least squares. This direction matches the DNG ColorMatrix convention. The camera RGB and XYZ are defined up to a free exposure / normalization scale that propagates into M; we pin the green-row middle entry $M _ { 2 2 } = 1$ for gauge-fixing, one of several admissible normalizations:

![](images/a880a0227cfb7d37dfc8d1ecd56d7836b6d6395cc3f68872670fc1d30c81dd65.jpg)  
Figure 2. Nikon D40 spectral sensitivity (filled bands) overlaid with the 14-channel ALS filter bank used in our experiments.

$$
\mathbf { M } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { M } } \sum _ { k = 1 } ^ { 2 4 } \| \hat { \mathbf { r } } _ { k } - \mathbf { M } \hat { \mathbf { x } } _ { k } \| _ { 2 } ^ { 2 } \quad \mathrm { s . t . } M _ { 2 2 } = 1 .\tag{2}
$$

At inference, we map a captured RGB r to scene-frame XYZ by $\mathbf { x } ( \mathbf { r } ) { = } \mathbf { M } ^ { - 1 } \mathbf { r }$ . The matrices we recover have condition numbers in the $1 0 ^ { 1 } – 1 0 ^ { 2 }$ range, so the inversion is numerically stable. We then apply Bradford chromatic adaptation [12] from $\hat { E } \mathrm { : }$ white point $\dot { \mathbf { C } } ^ { \dagger } \dot { E }$ to D65 for evaluation.

ALS filter bank Our main experiments use K idealized Gaussian filters with FWHM 25 nm and peaks uniformly spaced over 400– 700 nm (Fig. 2); this serves as a clean control for the channelcount ablation. An AS7341-like response with asymmetric peaks at the AS7341 channel centers, 5% adjacent-channel crosstalk, and 3% DC offset is reported in Sec. 4.3.

## Experiments

We evaluate the pipeline on a synthetic benchmark of 300 illuminants $\times 3 ,$ 100 reflectances against six baselines (CCT-driven, xy-conditioned, spectral), then probe robustness, OOD generalization, a metameric-pair stress test, and finally a measured-SPD evaluation plus real-RAW sanity check.

The camera spectral sensitivity is the Nikon D40 from [13]. Test illuminants are 300 EMPIR sources disjoint from the basis pool (random seed 0), split by signed Duv into Planckian (|Duv|<0.003, n=156), mild (0.003−0.01, n=94), LED-like $( 0 . 0 1 - 0 . 0 3 , ~ n { = } 4 5 )$ , and extreme $( \geq 0 . 0 3 , \ n { = } 5 ;$ see Fig. 3). Test reflectances are 100 random pixels per scene over 31 CAVE scenes [15], 3,100 vectors total. The calibration target is a 24- patch ColorChecker-style reflectance set distinct from the CAVE test reflectances, so the evaluation is a cross-distribution test. We report per-pair angular error on D65 XYZ as both the mean and the worst-25% mean (average of the upper quartile, which exposes the metameric tail); CIE ∆E<sub>2000</sub> [16] tracks the same ranking and is reported inline.

Baselines Oracle: per-illuminant least-squares CCM with the true SPD known; a non-causal upper bound. 2-CST DNG [1]: Adobe ColorMatrix1/2 with $1 / T$ interpolation by McCamy CCT [17]. 2-CST SELF: same interpolation with self-calibrated matrices (D40 SSF × Planckian × ColorChecker reflectance).

![](images/92df6808722ac3291c7af17d742740ce83c0969d78aa9d667df677e591ab0141.jpg)  
Figure 3. EMPIR test illuminants on CIE 1931 xy, color-coded by signed Duv. The dataset spans Planckian sources, daylight phases, and significantly off-locus LEDs.

2D-NN: nearest-neighbor lookup in xy over basis-illuminant CCMs. $C S T \cdot M L P \left[ 2 \right] :$ a reimplementation of both heads of Tedla et al.: a 2-layer 32-unit MLP from xy to either (i) the 8 free entries of a 3×3 CCM $( ^ {  } \mathbf { C S T - M L P } _ { 3 \times 3 } ^ { 3 9 } )$ , or (ii) an 18-element $3 \times 6$ matrix that maps Finlayson root-polynomial features [8] to $\mathrm { X Y Z \ ( \mathrm { ^ { * } C S T - M L P } } _ { 3 \times 6 } \mathrm { ^ { , 9 } ) }$ . Both heads trained on the 200 basis illuminants for 2,000 Adam steps at $\scriptstyle 1 5 = 1 0 ^ { - 3 }$ with per-patch cosine loss. Spectral WP-only: our framework with $K { = } 0 .$ Spectral $W P { + } A L S ;$ the full method with $K { = } 1 4$

Main result: synthetic benchmark by Duv Table 1 reports mean and worst-25% angular error per Duv group. ${ \mathrm { W P } } { \mathrm { + A L S } }$ is the best non-Oracle method at every group. Against the strongest non-spectral baseline $( \mathbf { C S T - M L P } _ { 3 \times 6 } , \ 1 . 8 9 ^ { \circ }$ overall), WP+ALS reduces mean error by 36% (1.20<sup>◦</sup>). The headline gain is statistically grounded on the LED-like group $( n { = } 4 5 )$ , where the metameric problem is tangible: worst-25% drops from $4 . 7 9 ^ { \circ }$ to $2 . 8 0 ^ { \circ }$ , a 1.71× reduction. The Extreme group’s tail is even larger but rests on only $n { = } 5$ illuminants and is reported directionally. The xy-conditioned baselines (2D-NN, CST-MLP) beat the CCT-driven baselines at low Duv, confirming Tedla et al.’s diagnosis. They fall apart off the locus, where WP+ALS has $K { + 3 }$ conditioning dimensions instead of 2. The 3×6 root-polynomial extension of Tedla shrinks but does not close the gap (1.89<sup>◦</sup> vs. $1 . 2 0 ^ { \circ }$ overall, $2 . 2 7 ^ { \circ }$ vs. $1 . 3 1 ^ { \circ }$ on LED-like). On Planckian, ALS adds only 0.50<sup>◦</sup> over WP-only — positive but small.

The visual gap is pronounced on saturated chart patches: metameric ambiguity translates most directly into hue and saturation errors, where WP+ALS approaches the Oracle floor while WP-only and the CCT baselines retain large per-patch residuals on red, orange, and blue primaries (visible in the worst-25% column of Table 1, dominated by these patches).

Perceptual color difference Mean $\Delta E _ { 2 0 0 0 }$ [16] for DNG / CST-$\mathbf { M L P } _ { 3 \times 3 } / \mathbf { W P }$ -only / WP+ALS: overall $2 . 1 0 / 1 . 5 5 / 1 . 1 2 / 0 . 6 9 ;$ LED-like $\left( n { = } 4 5 \right) 2 . 2 0 / 1 . 6 5 / 1 . 3 2 / \mathbf { 0 . 8 0 }$ ; Extreme $( n { = } 5 ,$ directional) $5 . 9 4 / 8 . 6 6 / 5 . 6 6 / 2 . 7 8 .$ . Same ranking as angular error.

Robustness and basis representativeness We test the proposed pipeline along four axes (Table 2). (a) Channel count × basis size: sweeping $K \in \{ 0 , 4 , 8 , 1 4 , 2 8 \}$ against $M \in \{ 1 0 , 3 0 , 5 0 , 1 0 0 \}$ on the off-Planckian subset, the $K { = } 8$ knee is robust across M (within $0 . 1 6 ^ { \circ }$ of the row minimum at every $M ) ,$ consistent with the classical 5–8-dimensional rank of natural illuminant SPDs.

(b) Realistic ALSfilter shape: an AS7341-like response — asymmetric peaks at the AS7341 channel centers, 5% crosstalk, 3% DC offset; a conservative non-idealization, not manufacturermeasured curves — at $K { = } 8$ raises mean error from $1 . 2 1 ^ { \circ }$ to $1 . 3 2 ^ { \circ } ( + 8 . 7 \% ) .$ (c) Observation noise: multiplicative Gaussian noise on (wp,als) at $\sigma \in \{ 1 , 3 , 5 \}$ % degrades WP+ALS mean error from $1 . 2 0 ^ { \circ }$ to $1 . 2 3 ^ { \circ } \ ( + 2 . 8 \%$ at 5%); the WP+ALS–DNG crossover would require σ ≈ 50%. (d) Cross-database (out-ofdistribution): on 23 CIE reference illuminants $( \mathrm { F L } _ { 1 - 9 } , \mathrm { H P } _ { 1 - 5 } ,$ LED-B/V/RGB) held out from EMPIR, an EMPIR-only basis under-performs DNG $( 2 . 7 5 ^ { \circ }$ vs. 2.21<sup>◦</sup>): the basis cannot represent fluorescent Hg lines. Augmenting the pool with 7 CIE references (one per major class) and re-running k-means restores the spectral lead in this controlled test $( 2 . 0 0 ^ { \circ }$ vs. DNG $2 . 2 5 ^ { \circ }$ on the remaining 16 hold-outs). A small class-representative augmentation can substantially improve OOD behavior; the count and selection strategy are tuned for this benchmark.

Metameric pair stress test We pick 50 EMPIR pairs $\left( E _ { a } , E _ { b } \right)$ with $\Delta _ { x y } < 0 . 0 0 5$ but SPD-shape angle larger than $3 0 ^ { \circ }$ (where SPD-shape angle is arccos $\left( { \bar { \langle E _ { a } , E _ { b } \rangle } } / { \| E _ { a } \| \| E _ { b } \| } \right)$ in degrees on the unit-normalized SPDs). Both sides are held out from the basis. WP-only NNLS collapses the pair (recovered pair angle $1 2 . 6 ^ { \circ }$ , true 35<sup>◦</sup>); ${ \mathrm { W P + A L S } }$ recovers $2 5 . 2 ^ { \circ }$ , with per-side SPD error $9 . 3 ^ { \circ }$ vs. 24. $. 6 ^ { \circ } - \mathrm { ~ a ~ } 2 . 6 \times$ improvement. Fig. 1(a) shows a comparable pair. On these cherry-picked pairs the per-patch CCM error is slightly higher for $\mathbf { W P + A L S _ { \alpha } } ( 1 . 8 7 ^ { \circ } )$ than WPonly $( 1 . 2 9 ^ { \circ } )$ : WP-only converges on a single “compromise” SPD that fits both sides forgivingly, while WP+ALS exposes basisdiscretization error. Better SPD recovery does not linearly translate into lower CCM error; the basis and CCM objective are not perfectly aligned. What WP+ALS uniquely provides is the ability to identify the correct side of a metamer; on the broader benchmark above, where reflectances vary freely, this converts to a measurable CCM-error gain.

Measured-SPD evaluation and real RAW sanity check We replace the EMPIR test illuminants with the 1,684 spectroradiometer-measured SPDs from Beyond RGB [18] (1,209 lab + 475 outdoor); the rest of the protocol (Nikon D40 SSF, CAVE reflectance, AS7341-like ALS synthesized from the measured SPD) is unchanged. This validates the illuminant side of the pipeline, not the full RGB+ALS capture loop. The same per-Duv pattern survives the swap to measured SPDs in both mean and worst-25% (Table 3): ${ \mathrm { W P } } { \mathrm { + A L S } }$ leads on LED-like and Extreme groups in both metrics, while $\mathrm { C S T - M L P } _ { 3 \times 6 }$ wins on Planckian and Mild. The LED-like advantage on Beyond RGB is smaller than on the synthetic benchmark (0.24<sup>◦</sup> mean $/ \ 0 . 1 7 ^ { \circ }$ worst-25% here vs $0 . 9 6 ^ { \circ } ~ / ~ 1 . 9 9 ^ { \circ }$ in Table 1); we attribute this to the broader test-illuminant distribution and the sub-optimality of the LED-biased EMPIR basis on daylight-dominated outdoor scenes. Spectral conditioning pays off where chromaticity is ambiguous, but the size of that pay-off depends on how off-locus the test set is.

We sanity-check on 117 real Nikon D40 captures with per-image white-point measurements but no ALS hardware (Fig. 4 shows four). The dataset is 84% near-Planckian; all three WP-only methods land within $0 . 5 ^ { \circ }$ in mean angular error (2-CST SELF $3 . 0 0 ^ { \circ }$ , Spectral WP-only 3.09<sup>◦</sup>, 2-CST DNG 3.42<sup>◦</sup>). This is the predicted behavior on a near-Planckian distribution: the WP-only fallback recovers the SPD’s 1D position on the locus, and the ALS adds nothing it can resolve there. End-to-end validation of ${ \mathrm { W P } } { \mathrm { + A L S } }$ on synchronized RGB+ALS captures remains future work.

Table 1. Mean and worst-25% angular error (degrees) per Duv group on 300×3,100 (illuminant, reflectance) pairs after Bradford CAT to D65. 3×3 and 3×6 are the linear and root-polynomial heads of Tedla et al.’s CST-MLP [2] (3×6 uses Finlayson root-polynomial features [8]). Best non-Oracle entry per row in bold. The Extreme group has only = illuminants; its worst-25% averages 1–2 samples and is reported for completeness, not statistical inference. Statistical weight rests on the LED-like group (n=45). Paired bootstrap on the WP+ALS vs CST-MLP <sub>×</sub> mean gap (illuminant-level resampling, B=1000): overall ∆ = +0.70<sup>◦</sup>, 95% CI [+0.40, +1.19]<sup>◦</sup>; LED-like ∆ = +0.97<sup>◦</sup>, 95% CI [+0.67, +1.37]<sup>◦</sup>; Extreme ∆=+5.62<sup>◦</sup>, 95% CI [+0.01, +14.56]<sup>◦</sup> (the wide CI is the n=5 caveat made explicit).
<table><tr><td rowspan="2">|Duv| group</td><td colspan="3">CCT-driven</td><td colspan="3">xy-conditioned</td><td colspan="2">Spectral (ours)</td></tr><tr><td>Oracle</td><td>DNG [1]</td><td>SELF</td><td>2D-NN</td><td>3×3 [2]</td><td>3×6</td><td>WP-only</td><td>WP+ALS</td></tr><tr><td colspan="9">Mean angular error</td></tr><tr><td>Planckian &lt;0.003 (n=156)</td><td>0.95</td><td>2.45</td><td>3.18</td><td>1.75</td><td>1.79</td><td>1.33</td><td>1.55</td><td>1.05</td></tr><tr><td>Mild 0.003–0.01 (n=94)</td><td>0.87</td><td>2.73</td><td>3.44</td><td>2.01</td><td>1.99</td><td>2.21</td><td>1.82</td><td>1.21</td></tr><tr><td>LED-like 0.01–0.03 (n=45)</td><td>0.81</td><td>3.09</td><td>4.26</td><td>2.78</td><td>2.41</td><td>2.27</td><td>2.18</td><td>1.31</td></tr><tr><td>Extreme ≥0.03 (n=5)</td><td>2.36</td><td>11.08</td><td>26.75</td><td>6.60</td><td>14.31</td><td>10.11</td><td>9.23</td><td>4.49</td></tr><tr><td>Overall (n=300)</td><td>0.93</td><td>2.77</td><td>3.81</td><td>2.07</td><td>2.15</td><td>1.89</td><td>1.86</td><td>1.20</td></tr><tr><td colspan="9">Worst-25% angular error (mean of upper quartile)</td></tr><tr><td>Planckian &lt; 0.003</td><td>2.02</td><td>4.29</td><td>5.33</td><td>3.74</td><td>3.61</td><td>2.77</td><td>3.21</td><td>2.27</td></tr><tr><td>Mild 0.003–0.01</td><td>1.92</td><td>5.05</td><td>6.47</td><td>4.57</td><td>4.15</td><td>5.97</td><td>3.92</td><td>2.83</td></tr><tr><td>LED-like 0.01–0.03</td><td>1.88</td><td>5.32</td><td>8.50</td><td>6.37</td><td>4.67</td><td>4.79</td><td>4.08</td><td>2.80</td></tr><tr><td>Extreme ≥0.03</td><td>7.10</td><td>35.88</td><td>88.34</td><td>18.86</td><td>45.66</td><td>33.39</td><td>30.97</td><td>12.63</td></tr><tr><td>Overall</td><td>2.06</td><td>5.29</td><td>7.73</td><td>4.71</td><td>4.74</td><td>4.68</td><td>4.08</td><td>2.74</td></tr></table>

![](images/6cd479ce1470e411e8f96efe15157d033dd39aabdc0d556f2d0234bce5e33d87.jpg)  
Figure 4. Four real Nikon D40 captures. Top two rows near-Planckian (tungsten museum, daylight outdoor). Bottom two rows off the locus (LED hallway |Duv|=0.012, mall LED |Duv|=0.021). Left column shows the scene with the chart highlighted. The four right columns show the 24-patch chart as-shot (raw camera response, before any white-balance, so each row shows its own illuminant cast), corrected by 2-CST DNG, by our spectral white-point pipeline, and the D65 ground truth. Per-shot mean angular error ∆ relative to the D65 GT is annotated on the two corrected columns; the as-shot column is shown for context only. Spectral-WP edges out or matches DNG, with the largest gain on the most distant illuminant (tungsten, 1.46<sup>◦</sup> vs. 2.36<sup>◦</sup>).

## Discussion

We close with one implication from the channel-count ablation, then the limitations.

Implication: where 8 channels come from The knee of the WP+ALS error curve sits at K=8, robust across basis sizes M ∈ [10,100] (Sec. 4.3). This matches the classical 5–8-dimensional rank of natural illuminant SPDs and the channel count of commodity multispectral ALS chips [3]: spectral information past 8 narrow bands gives diminishing returns on the CCM endpoint. A practical reading: an 8-channel commodity ALS provides enough spectral conditioning for the off-Planckian band where

xy-conditioned baselines fail.

When does better SPD recovery yield a better CCM? The metameric stress test (Sec. 4.5) shows an apparent paradox: WP+ALS recovers metameric SPDs 2.6× more accurately than WP-only, yet per-patch CCM error on those cherry-picked pairs is slightly worse (1.87<sup>◦</sup> vs 1.29<sup>◦</sup>). The mechanism is that WPonly converges on a single “compromise” SPD that fits both sides of a metameric pair forgivingly; WP+ALS resolves the pair but then pays basis-discretization error on a single side. SPD-recovery accuracy is therefore not sufficient for CCM accuracy. It is, however, necessary on the main benchmark, where reflectances span a wide distribution: there, the ability to identify the correct metamer side converts directly into the 1.7× worst-25% reduction on LED-like illuminants reported in Sec. 4.1. The stress test is a pathological boundary; the main benchmark is the deployment proxy.

Table 2. Robustness of ${ \mathsf { W P } } { \mathrm { + A L S } }$ along four axes. Numbers are mean / worst-25% angular error in degrees on the relevant test set; for reference, DNG is 2.77/5.29 on the same scale (Table 1). Cross-DB OOD test set: 23 hold-out CIE reference illuminants (FL/HP/LED), with the bottom row using a basis pool augmented by 7 CIE reference SPDs.
<table><tr><td>Axis / condition</td><td>mean / worst-25 (°)</td></tr><tr><td>Baseline (ideal  $\overline { { K { = } 1 4 , { \mathsf { n o } } { \mathsf { n o } } { \mathsf { i s e } } ) } }$ </td><td> $\overline { { 1 . 2 0 / 2 . 7 4 } }$ </td></tr><tr><td>AS7341-like filters  $( K { = } 8 )$ </td><td>1.32/3.25</td></tr><tr><td>5%observation noise</td><td>1.23/2.83</td></tr><tr><td>OOD, EMPIR-only basis</td><td>2.75/5.40</td></tr><tr><td>OOD, basis +7 CIE refs</td><td>2.00/4.21</td></tr></table>

Table 3. Beyond RGB [18] mean and worst-25% angular error (degrees) by Duv group on 1,684 measured-SPD scenes. Groups: Plk = Planckian (n=868), Mild (n=473), LED = LED-like (n=276), Ext = Extreme (n=67). Best per row in bold. ${ \mathsf { W P } } { \mathrm { + A L S } }$ leads on LED and Ext in both metrics; the worst-25% advantage on LED is smaller than the mean advantage $( 0 . 1 7 ^ { \circ }$ vs $0 . 2 4 ^ { \circ }$ over CST-${ \sf M L P } _ { 3 \times 6 } )$ , reflecting the broader-distribution test illuminants compared to the synthetic benchmark.
<table><tr><td>Method</td><td>PIk</td><td>Mild</td><td>LED</td><td>Ext</td></tr><tr><td>Mean angular error</td><td></td><td></td><td></td><td></td></tr><tr><td>DNG</td><td>2.12</td><td>1.91</td><td>2.51</td><td>6.97</td></tr><tr><td> $\mathsf { C S T - M L P } _ { 3 \times 6 }$ </td><td>1.72</td><td>1.67</td><td>2.13</td><td>9.50</td></tr><tr><td> ${ \mathsf { W P - o n l y } }$ </td><td>2.67</td><td>2.31</td><td>3.16</td><td>5.76</td></tr><tr><td> ${ \mathsf { W P } } { \mathrm { + A L S } }$ </td><td>2.17</td><td>1.83</td><td>1.89</td><td>5.14</td></tr><tr><td>Worst-25% angular error</td><td></td><td></td><td></td><td></td></tr><tr><td>DNG</td><td>4.40</td><td>3.71</td><td>5.40</td><td>21.61</td></tr><tr><td> $\mathsf { C S T - M L P } _ { 3 \times 6 }$ </td><td>3.66</td><td>3.67</td><td>4.50</td><td>31.64</td></tr><tr><td> ${ \mathsf { W P - o n l y } }$ </td><td>4.84</td><td>4.69</td><td>6.26</td><td>15.70</td></tr><tr><td> ${ \mathsf { W P } } { \mathrm { + A L S } }$ </td><td>4.56</td><td>4.01</td><td>4.33</td><td>15.50</td></tr></table>

Limitations and future work We isolate the spectralconditioning contribution by holding the prediction head fixed at the non-parametric NNLS form throughout. The orthogonal axis

— training a parametric head (MLP, root-polynomial, etc.) on the same K+3-dimensional input — is the learned-amortization extension noted as future work; the CST-MLP comparators receive only xy chromaticity by their original design [2]. The ALS observations are simulated; the AS7341-like and noise analyses cover the tractable parts of the sim-to-real gap, but calibration drift and ALS–RGB spatial mismatch are unaddressed. The 117 D40 RAW captures (Sec. 4.6, Fig. 4) sanity-check the WPonly fallback only and the dataset is 84% near-Planckian, where the ALS adds no information the white point does not already provide; end-to-end synchronized $\mathbf { R } \mathbf { G } \mathbf { B } { + } \mathbf { A } \mathbf { L } S$ validation on off-Planckian sources is the cleanest hardware-side next step. The single-illuminant SPD assumption also breaks down under indoor mixed lighting (e.g., daylight through a window plus an LED downlight). We test one camera (Nikon D40); cross-sensor generalization is unverified, although principled dual-mapping schemes [14] suggest a tractable extension. Embedded-ISP latency on the per-shot NNLS+LSQ pipeline is not measured here.

## Conclusion

Two-CCM interpolation by CCT is a 1D model of a higherdimensional quantity. A few extra spectral observations recover SPD shape and fit a scene-specific CCM that approaches the perilluminant Oracle within $0 . 3 ^ { \circ }$ on Planckian, mild, and LED-like groups. Eight ALS channels suffice (a K=8 knee robust across basis sizes $M \in [ 1 0 , 1 0 0 ] \backslash$ ). On the LED-like group (n=45) where the metameric problem is tangible, ${ \mathrm { W P } } { \mathrm { + A L S } }$ reaches a worst-25% 1.7× lower than the strongest non-spectral baseline (CST-${ \mathrm { M L P } } _ { 3 \times 6 } ) { \mathrm { ; } }$ the lead is robust to AS7341-like filter shape and 5% observation noise. Cross-database generalization requires augmenting the basis with class-representative reference SPDs; on Beyond RGB measured illuminants, the spectral lead is largest off the Planckian locus, as the metameric-problem framing predicts.

## References

[1] Adobe Systems, Digital Negative (DNG) Specification, version 1.6.0.0, https://helpx.adobe.com/camera-raw/digital-negative.html, 2021.

[2] S. Tedla, J. E. Little, H. C. Karaimer, and M. S. Brown, Off the Planckian locus: Using 2D chromaticity to improve in-camera color, arXiv preprint arXiv:2511.17133 (2025).

[3] ams Osram, AS7341 11-Channel Multi-Spectral Digital Sensor, Datasheet, https://ams-osram.com/products/sensors/spectralsensors/ams-as7341, 2023.

[4] G. Buchsbaum, A spatial processor model for object colour perception, J. Franklin Inst., 310, 1 (1980).

[5] J. van de Weijer, T. Gevers, and A. Gijsenij, Edge-based color constancy, IEEE Trans. Image Process., 16, pg. 2207 (2007).

[6] S. Yue and M. Wei, Color constancy from a pure color view, J. Opt. Soc. Am. A., 40, pg. 602 (2023).

[7] H. C. Karaimer and M. S. Brown, Improving color reproduction accuracy on cameras, In Proc. CVPR (2018).

[8] G. D. Finlayson, M. Mackiewicz, and A. Hurlbert, Color correction using root-polynomial regression, IEEE Trans. Image Process., 24, pg. 1460 (2015).

[9] B. Arad and O. Ben-Shahar, Sparse recovery of hyperspectral signal from natural RGB images, In Proc. ECCV (2016).

[10] Y. Li, Q. Fu, and W. Heidrich, Multispectral illumination estimation using deep unrolling network, In Proc. ICCV (2021).

[11] EMPIR PhotoLED, EMPIR PhotoLED Spectral Database, version 2, European Metrology Programme for Innovation and Research, 2018.

[12] K. M. Lam, Metamerism and Colour Constancy, Ph.D. thesis, University of Bradford, 1985.

[13] J. Jiang, D. Liu, J. Gu, and S. Susstrunk, What is the space of¨ spectral sensitivity functions for digital color cameras?, In Proc. WACV (2013).

[14] S. Yue and M. Wei, Effective cross-sensor color constancy using a dual-mapping strategy, J. Opt. Soc. Am. A., 41, pg. 329 (2024).

[15] F. Yasuma, T. Mitsunaga, D. Iso, and S. K. Nayar, Generalized assorted pixel camera: Post-capture control of resolution, dynamic range and spectrum, Technical Report CUCS-061-08, Department of Computer Science, Columbia University, 2008.

[16] G. Sharma, W. Wu, and E. N. Dalal, The CIEDE2000 colordifference formula: Implementation notes, supplementary test data, and mathematical observations, Color Res. Appl., 30, pg. 21 (2005).

[17] C. S. McCamy, Correlated color temperature as an explicit function of chromaticity coordinates, Color Res. Appl., 17, pg. 142 (1992).

[18] O. Glatt, Y. Ater, W.-S. Kim, S. Werman, O. Berby, Y. Zini,

S. Zelinger, S. Lee, H. Choi, and E. Soloveichik, Beyond RGB: A real world dataset for multispectral imaging in mobile devices, In Proc. WACV, pp. 4344–4354 (2024).