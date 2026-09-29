# CAST: RECONSTRUCTION-COUPLED ACCELERATION OF INTERACTIVE WORLD MODELS

Leyang Chen<sup>∗</sup>, Junyi Wu<sup>∗</sup>, Fanqing Kong<sup>∗</sup>, Shaoqiu Zhang, Yulun Zhang<sup>†</sup> Shanghai Jiao Tong University

![](images/2cb1dcd1e49bb872721c6d4bdcbeb8b08167b630ac6f31aee1d978015f00444c.jpg)  
Figure 1: Action-conditioned rollouts. Native (top) and our method CAST (bottom) on Matrix-Game 3.0 (left) and HY-World 1.5 (right). CAST maintains visual fidelity close to Native while achieving 2.15× and 3.48× speedups, respectively (Figure 2); W/S denote forward/backward movement.

## ABSTRACT

Interactive world models must respond quickly to controls while preserving scene consistency. Existing acceleration methods can miss heterogeneous control responses and spatial transport when recovering skipped features. We observe that interaction-induced feature changes correlate with approximation error, while low-frequency interpolation errors are phase-sensitive and show more predictable phase progression. These findings motivate CAST, a reconstruction-coupled inference framework. CAST selects anchors by interaction sensitivity and cross-layer coverage, reconstructs skipped residuals with frequency- and confidence-aware Phase-Aware Reconstruction (PAR), and coordinates historical KV routing according to downstream reconstruction responsibility. On Matrix-Game 3.0 and HY-World 1.5, CAST achieves 2.15× and 3.48× speedups, respectively, while maintaining visual quality close to Native (Figure 1). It also attains the highest VBench scores among compared methods and leads non-native baselines on seven and six of thirteen WorldMark dimensions, demonstrating a balance of generation speed, visual quality, and interactive responsiveness under real-time control. Code is available at https://github.com/lokiniuniu/CAST.

## 1 INTRODUCTION

Interactive world models turn visual generation into a feedback loop: users act and the model generates the next observation (Bruce et al., 2024; Valevski et al., 2025). Recent systems combine streaming generation, camera and action conditioning, and visual history (Wang et al., 2026; Sun et al., 2025). Their quality depends on responsive controls and consistent revisits, beyond plausible frames alone (Xu et al., 2026). Inference latency remains a practical obstacle to interactive use (Lu et al., 2026), making efficiency part of the modeling problem.

Efficient sampling algorithms reduce the number of denoising-network evaluations required for generation (Song et al., 2021; Lu et al., 2022). Distillation and consistency models further enable few-step generation (Salimans & Ho, 2022; Song et al., 2023), including autoregressive video synthesis (Yin et al., 2025; Huang et al., 2025). Yet each network evaluation can remain costly even with few denoising steps (Tang et al., 2026). CAST targets this remaining cost by reducing computation within each evaluation. Training-free methods exploit redundancy by caching features across denoising steps (Ma et al., 2024b; Liu et al., 2025a; Lyu et al., 2025), computing selected frames and interpolating their neighbors (Tang et al., 2026), or reducing historical attention (Lu et al., 2026; Li et al., 2026a). These strategies offer complementary savings, although few-step inference limits cross-step reuse (Tang et al., 2026). In our setting, the approximations are connected: computed frames supply the endpoints for reconstructing skipped updates, so errors in their historical attention can also propagate beyond anchor frames themselves through reconstruction (Section 3.3).

This dependency suggests organizing acceleration around the recovery of skipped states. Prior work shows that motion content and token sensitivity can guide nonuniform computation (Kahatapitiya et al., 2025; Zou et al., 2025). We ask whether control responses provide a useful signal for selecting frame anchors. Figure 3 shows that stronger camera- and action-induced feature responses are associated with larger next-block skipping errors. Because these responses arise in the conditioning path, they can guide allocation without first evaluating the dense residual we seek to avoid. Selection must also maintain coverage: interleaving frames across layers is important in frame-sparse inference so that exact computation will not be allocated to several fixed frames (Tang et al., 2026). We combine sensitivity with a refresh debt to balance responsive updates and regular coverage.

![](images/03650adb1c47acac64dc890898e391edbb70cd0f48d67b65364b2240d49e55fc.jpg)  
Figure 2: Speed and interactive quality. Speedup over Native (left) and selected World-Mark dimensions (right), min–max normalized per backbone across all methods including Native; colors match the speedup bars.

In interactive world models, moving the camera or controlling an object shifts scene content across spatial locations (Wang et al., 2026). Recovering skipped updates between anchors therefore requires accounting for where content moves, beyond how feature values change. Direct interpolation blends endpoint residuals at fixed coordinates, potentially mixing different scene content rather than tracking its displacement. Phase transport offers a way to align these residuals before interpolation: locally coherent motion can be represented through phase shifts (Meyer et al., 2015). Our diagnostics in Figure 4 support this view: oracle phase correction substantially reduces a low-frequency interpolation-error peak, and phase progression is more concentrated at low frequencies. These findings motivate phase-aware reconstruction, with transport strength adapted to frequency and local phase confidence to limit corrections where coherent motion is poorly supported.

Reconstruction changes what an anchor needs from history. Prior analyses show that local approximation errors can have unequal downstream effects (Zou et al., 2025). Historical routing in Light Interaction selects blocks by query–key relevance (Lu et al., 2026). In our setting, an anchor also supplies residuals for skipped frames, so local relevance alone does not account for this dependency. If neighboring anchors omit useful evidence together, their errors may jointly affect the reconstructed interval. We therefore coordinate their historical supports under fixed per-anchor budgets, aiming to preserve complementary evidence for both the anchors and the states they reconstruct.

Building on these observations, we introduce CAST, a reconstruction-coupled inference framework for interactive world models. Control-Aware Frame Selection balances interaction sensitivity with cross-layer coverage. Phase-Aware Reconstruction recovers skipped residuals through frequencyand confidence-aware phase transport. Finally, Reconstruction-Coupled Sparse Attention refines historical KV supports according to both local omission effects and the recovery of neighboring states. Together, these components connect where computation is retained, how omitted updates are recovered, and which history supports that recovery. Our contributions are as follows:

• Control-Aware Frame Selection. We select anchors using camera- and action-induced feature responses. Cross-layer coverage promotes regular refreshes, balancing interactionsensitive updates with the need to maintain representations throughout the current chunk and across the entire active action horizon of each generated sequence.

• Phase-Aware Reconstruction. We recover skipped residuals by aligning endpoint spectra before interpolation or one-sided extrapolation. Transport strength adapts to frequency and local phase confidence, limiting corrections when evidence for coherent motion remains uncertain across the full interactive control sequence.

• Reconstruction-Coupled Sparse Attention. We coordinate neighboring anchors’ historical KV supports under fixed budgets according to their local omission effects and shared responsibility for reconstructing skipped states across adjacent intervals.

• Evaluation on interactive world models. As shown in Figure 2, on Matrix-Game 3.0 and HY-World 1.5, CAST achieves 2.15× and 3.48× acceleration, respectively, and leads non-native methods on seven and six of thirteen WorldMark dimensions.

![](images/193a449327d9ef3907082079762011ce0e7131109aa5f4f8ffb2cd83ef8e698b.jpg)

## 2 RELATED WORK

## 2.1 INTERACTIVE VIDEO WORLD MODELS

Dreamer and IRIS learn behaviors through imagined rollouts (Hafner et al., 2020; Micheli et al., 2023). DIAMOND introduces diffusion-based world modeling (Alonso et al., 2024), GameNGen simulates games (Valevski et al., 2025), and Genie learns interactive environments through latent actions (Bruce et al., 2024). Video diffusion and Stable Video Diffusion provide generative foundations (Ho et al., 2022; Blattmann et al., 2023); CameraCtrl and MotionCtrl add explicit camera and motion control (He et al., 2025; Wang et al., 2024), enabling directed changes to viewpoint and scene motion.

CausVid and Self Forcing enable few-step autoregressive generation (Yin et al., 2025; Huang et al., 2025). We target interactive chunk-autoregressive models, including Matrix-Game 3.0 and WorldPlay (Wang et al., 2026; Sun et al., 2025), where processing frames, controls, and history makes per-evaluation latency critical for responsive online control.

## 2.2 EFFICIENT INFERENCE FOR VIDEO DIFFUSION MODELS

Caching reuses intermediate computation across denoising steps. DeepCache and Block Caching reuse features or block outputs (Ma et al., 2024b; Wimbauer et al., 2024); Learning-to-Cache learns layer schedules (Ma et al., 2024a). TeaCache and MagCache adapt reuse to input changes, residual magnitudes, or block dynamics (Liu et al., 2025a; Ma et al., 2025). QuantCache combines hierarchical latent and layer caching with importance-guided quantization and pruning for video DiTs (Wu et al., 2025a). AdaCache adapts schedules to video content (Kahatapitiya et al., 2025); TaylorSeer forecasts features instead of directly reusing them (Liu et al., 2025b).

ToCa selects tokens by sensitivity and redundancy (Zou et al., 2025), while DuCa alternates aggressive and conservative caching (Zou et al., 2026). These methods establish the value of selective computation, but few-step models offer limited cross-step reuse. There are also other DiT acceleration schemes (Tang et al., 2026; He et al., 2026; Li et al., 2026b; Chen et al., 2026; Wu et al., 2025b).

Light Interaction combines context management, output reuse, and historical sparsity (Lu et al., 2026); Sol Engine integrates caching, sparsity, pruning, quantization, and kernels (Li et al., 2026a). CAST coordinates retained frames and historical supports through their shared reconstruction responsibility.

## 2.3 SPARSE ATTENTION

Longformer and BigBird use structured sparsity (Beltagy et al., 2020; Zaheer et al., 2020); Reformer and Routing Transformer route by content (Kitaev et al., 2020; Roy et al., 2021). FlashAttention and FlashAttention-2 optimize exact attention (Dao et al., 2022; Dao, 2024). For video, Sliding Tile Attention exploits local windows (Zhang et al., 2025), LongCat-Video uses 3D block sparsity (Meituan LongCat Team et al., 2025), and Sparse VideoGen and VMoBA exploit spatial–temporal structure (Xi et al., 2025; Wu et al., 2026). Building on pooled historical routing (Lu et al., 2026), CAST considers both local omission errors and their propagation to reconstructed frames. This dependency makes retained historical evidence important beyond the computed anchors themselves, linking sparse attention decisions to the accuracy of skipped-frame reconstruction.

## 3 PRELIMINARIES AND MOTIVATION

Our setting combines diffusion denoising (Ho et al., 2020), latent representations (Rombach et al., 2022), and transformer blocks (Peebles & Xie, 2023). We study one evaluation of a latent chunk conditioned on controls and visual history. Let $R _ { i } ^ { \ell }$ denote an eligible visual residual update for frame i within transformer block $\ell ,$ and $\widehat { R } _ { i } ^ { \ell }$ its approximation. CAST is applied at eligible visual attention and FFN residual sites, while conditioning and text updates retain native dense execution. Within each network evaluation, retained anchors are computed directly and skipped frames receive reconstructed residual updates. Historical sparsity changes the evidence available to those anchors, linking frame selection, residual recovery, and KV routing within each transformer block.

![](images/ba3bda5f124f448f2b9d89b1414701deba4645b60a05d9053ee52a3f041444a7.jpg)

![](images/766eeb0f01dfbe2de6833338bf96dff5bbe770b2f16d686f9cc17ea34c37cd8b.jpg)  
Figure 3: Control sensitivity and skipping error. Top: native rollout. Middle: independently scaled sensitivity at block ℓ and error at $\ell + \mathrm { i }$ Bottom: pooled percentiles (Spearman $\rho = 0 . 7 4$ $n = 3 , 0 7 2 )$ , with counts and binned medians.

## 3.1 INTERACTION SENSITIVITY PREDICTS FRAME APPROXIMATION ERROR

Let $K _ { \ell } \subseteq \{ \mathrm { c a m } , \mathrm { a c t } \}$ contain the available control branches, with feature updates $\Delta _ { i , k } ^ { \ell }$ . For $\kappa _ { \ell } \neq \varnothing$ define control sensitivity and relative approximation error by

$$
d _ { i } ^ { \ell } = \frac { 1 } { \left| \mathscr { K } _ { \ell } \right| } \sum _ { k \in \mathscr { K } _ { \ell } } \mathrm { N o r m } ( \Delta _ { i , k } ^ { \ell } ) , \qquad e _ { i } ^ { \ell } = \frac { \left| \left| R _ { i } ^ { \ell } - \widehat { R } _ { i } ^ { \ell } \right| \right| _ { 2 } } { \left| \left| R _ { i } ^ { \ell } \right| \right| _ { 2 } + \epsilon } .\tag{1}
$$

Here Norm rescales update magnitudes across the current chunk. Sensitivity is available from the conditioning path; error requires a dense reference and is used only for diagnosis.

Figure 3 compares $d _ { i } ^ { \ell }$ with $e _ { i } ^ { \ell + 1 }$ , matching the scheduler’s one-block prediction lag. Both curves rise near the $\scriptstyle { \dot { \mathbf { W } } } - \mathbf { t o } - S$ transition; pooled percentiles show positive Spearman correlation $( \rho = 0 . 7 4 )$ Because the temporal curves are independently rescaled, their proximity indicates similar temporal variation, not equal magnitudes. The spread in the percentile plot also shows that sensitivity does not fully determine error. This motivates sensitivity-based allocation balanced by cross-layer coverage.

## 3.2 SPECTRAL ERROR AND FREQUENCY-DEPENDENT PHASE TRANSPORT

![](images/c8f202de8a8fe87c1d89a4ec1f59a42ce40f046e45384387656e655f000c02d5.jpg)

![](images/182e1157d00f1d97900246fe914baf732f290193a6adb5eb16f66f5ac2cf871c.jpg)  
Figure 4: Spectral error and phase progression. Top: interpolation error with oracle phase/magnitude corrections. Bottom: phase progress across frequency bands; dashed lines denote uniform progression.

For a skipped frame with $a < t < b ,$ linear interpolation uses $R _ { t } ^ { \operatorname* { l i n } } = ( 1 - \alpha ) R _ { a } + \alpha R _ { b } .$ where $\dot { \alpha } = ( t - a ) / ( b - a )$ . It mixes fixed coordinates even when content moves. Figure 4 separates alignment and amplitude errors using oracle corrections that replace either phase or magnitude with its reference value while retaining the other interpolated component (Appendix C.2).

The upper plot shows a low-frequency error peak largely removed by phase correction, while magnitude correction leaves substantial error. This supports alignment correction in that band. The advantage varies across frequencies, and error share alone does not establish difficulty relative to target-band energy at each frequency.

The lower panels show tighter phase progression at low frequencies and broader distributions at high frequencies. Coherent translation gives linear unwrapped phase at every frequency (Meyer

et al., 2015); the observed spread reflects reliability, not an intrinsic low-frequency rule. Thus PAR prioritizes alignment-sensitive components while tempering transport with local confidence.

## 3.3 RECONSTRUCTION DEPENDENCY IMPLIES COUPLED ATTENTION

Once two anchors reconstruct an interval, attention approximation at either endpoint can affect every skipped state. For $\widehat { R } _ { t } = \mathcal { P } _ { t } ( R _ { a } , R _ { b } )$ , anchor perturbations induce

$$
\Delta _ { t } ^ { \mathrm { r e c } } = \mathcal { P } _ { t } ( R _ { a } + \delta R _ { a } , R _ { b } + \delta R _ { b } ) - \mathcal { P } _ { t } ( R _ { a } , R _ { b } ) .\tag{2}
$$

The interval error $\begin{array} { r l } { \sum _ { t \in ( a , b ) } \left. \Delta _ { t } ^ { \mathrm { r e c } } \right. ^ { 2 } } \end{array}$ depends on both endpoints and the reconstruction operator. For linear interpolation, $\Delta _ { t } ^ { \mathrm { r e c } } = ( 1 - \alpha ) \delta R _ { a } + \alpha \delta R _ { b } \mathrm { . }$ : the squared error includes a cross term, so endpoint errors can reinforce or offset each other. Choosing historical KV supports independently therefore need not minimize interval error, even when each anchor is locally well approximated. This motivates coordinating supports under fixed per-anchor budgets to protect both anchors and reconstructed states. Section 5.3 tests this dependency at matched frame and historical KV budgets.

## 4 METHOD

## 4.1 OVERVIEW AND PROBLEM FORMULATION

Write the block input as $X ^ { \ell } = [ X ^ { \mathrm { m e m } , \ell } , X ^ { \mathrm { c u r } , \ell } ] .$ , with $N _ { \mathrm { c u r } }$ current latent frames. Memory frames retain native computation or caching; only current residuals are skipped. At block ℓ, the anchor set $\mathcal { A } _ { \ell } = \{ a _ { 0 } < \dots < a _ { J } \}$ contains $K _ { f } = J + 1$ frames, and adjacent anchors define intervals $\mathcal { T } _ { a , b } = \{ t : a < t < b \}$ . The anchor set is selected once per transformer block and shared by its eligible visual attention and FFN residual sites. Suppressing the residual-site index, each eligible update is

![](images/049c29c932966e58ed8d0c41b2fab6dc8409c8faeb4b694cb5cf8a0f57c6d112.jpg)  
Figure 5: CAST pipeline. Select anchors and historical KV blocks, compute anchor residuals, and reconstruct skipped updates while retaining each frame’s own skip connection.

$$
\begin{array} { r } { X _ { i } ^ { \ell + 1 } = X _ { i } ^ { \ell } + \widetilde { R } _ { i } ^ { \ell } , \qquad \widetilde { R } _ { i } ^ { \ell } = \left\{ \begin{array} { l l } { R _ { i } ^ { \ell , \mathrm { s p } } , } & { i \in \mathcal { A } _ { \ell } , } \\ { \mathcal { P } _ { i } ( R _ { a } ^ { \ell , \mathrm { s p } } , R _ { b } ^ { \ell , \mathrm { s p } } ) , } & { i \in \mathcal { T } _ { a , b } , } \\ { \mathcal { P } _ { i } ^ { \mathrm { e x t } } ( R _ { a _ { 0 } } ^ { \ell , \mathrm { s p } } , R _ { a _ { 1 } } ^ { \ell , \mathrm { s p } } ) , } & { i < a _ { 0 } , } \\ { \mathcal { P } _ { i } ^ { \mathrm { e x t } } ( R _ { a , I - 1 } ^ { \ell , \mathrm { s p } } , R _ { a , J } ^ { \ell , \mathrm { s p } } ) , } & { i > a _ { J } , } \end{array} \right. } \end{array}\tag{3}
$$

Here $\mathcal { P } _ { i } ^ { \mathrm { e x t } }$ uses the same spectral transport as $\mathcal { P } _ { i }$ with $\alpha < 0$ on the left and $\alpha > 1$ on the right. The two anchors nearest each boundary are used: $( a _ { 0 } , a _ { 1 } )$ for $i < a _ { 0 }$ and $( a _ { J - 1 } , a _ { J } )$ for $i > a _ { J }$ . This construction requires $K _ { f } \ge 2 . \ R _ { i } ^ { \ell , \mathrm { s p } }$ denotes the directly computed anchor update at the eligible residual site. Each skipped frame retains its own input $X _ { i } ^ { \ell }$ ; reconstruction supplies only its residual update. Figure 5 summarizes the pipeline. Execution routes masks before evaluating anchors and reconstructing skipped updates; the routing surrogate needs no dense anchor outputs.

## 4.2 CONTROL-AWARE FRAME SELECTION

Interaction sensitivity. The conditioning path supplies $d _ { i } ^ { \ell }$ for all current frames (Eq. (1)). We use these scores to select anchors at the next block, so selection does not require the residuals it aims to skip. As illustrated in Figure 6, frames with stronger control responses receive higher priority, while coverage debt prevents repeatedly favoring the same subset. Scoring costs are included in the reported end-to-end chunk latency.

Coverage debt. Following the coverage motivation of frame interleaving (Tang et al., 2026), let $h _ { i } ^ { \ell }$ count blocks since refresh. Define the refresh age and its normalized coverage debt:

$$
\begin{array} { r l } & { \quad h _ { i } ^ { \ell + 1 } = \left\{ \begin{array} { l l } { 0 , } & { i \in \mathcal { A } _ { \ell } , } \\ { h _ { i } ^ { \ell } + 1 , } & { i \notin \mathcal { A } _ { \ell } , } \end{array} \right. } \\ & { \quad H _ { \mathrm { c o v } } = \left\lceil \frac { N _ { \mathrm { c u r } } } { K _ { f } } \right\rceil , \quad \quad D _ { i } ^ { \ell + 1 } = \frac { h _ { i } ^ { \ell + 1 } } { H _ { \mathrm { c o v } } } . } \\ & { \quad S _ { i } ^ { \ell + 1 } = \left\{ \begin{array} { l l } { d _ { i } ^ { \ell } + \lambda _ { \mathrm { c o v } } D _ { i } ^ { \ell + 1 } , } & { \mathrm { f r e s h r e s p o n s e ~ a v a i l a b l e } , } \\ { \lambda _ { \mathrm { c o v } } D _ { i } ^ { \ell + 1 } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

![](images/571148e2ed5dad958e0c437e9def3c11eb8740e324e9a0f735c5b810b673a5cc.jpg)  
Figure 6: Control responses and coverage debt guide selection.

(4)

(5)

We set $\lambda _ { \mathrm { c o v } } = 0 . 5 0 ; H _ { \mathrm { c o v } }$ scales age but does not guarantee refresh. Deterministic $\mathrm { T o p } { \cdot } K _ { f }$ selects $K _ { f }$ anchors across all $N _ { \mathrm { c u r } }$ current frames: $\mathcal { A } _ { \ell + 1 } = \mathrm { T o p K } _ { i } ( S _ { i } ^ { \ell + 1 } , K _ { f } )$ . Neither endpoint is fixed. Without a fresh conditioning response, we omit sensitivity rather than reuse a stale score. Each denoising evaluation starts with a dense block and resets ages (Appendix A.1).

## 4.3 PHASE-AWARE RECONSTRUCTION

Local phase transport. Following phase-based interpolation and extrapolation (Meyer et al., 2015), partition residuals into tiles p and suppress the block index. For anchors $a < b$ and target frame $\boldsymbol { \dot { t } } \not \in \{ \boldsymbol { a } , \boldsymbol { b } \}$ , let $\alpha = ( t - a ) / ( \bar { b } - a )$ and $\dot { F } _ { i , p , c } ( \omega ) = \mathrm { R F F T 2 } ( R _ { i , p , c } ) ( \omega )$ for channel c. Interior targets have $0 < \alpha < 1$ ; boundary targets have $\alpha < 0 \mathrm { o r } \alpha > 1$ . Figure 7 illustrates the phase transport and spectral blending used to reconstruct these skipped residuals.   
Define the pooled cross-spectrum and magnitude product:

$$
Z _ { p } ( \omega ) = \sum _ { c } F _ { b , p , c } ( \omega ) \overline { { F _ { a , p , c } ( \omega ) } } , \quad W _ { p } ( \omega ) = \sum _ { c } | F _ { b , p , c } ( \omega ) | | F _ { a , p , c } ( \omega ) | .\tag{6}
$$

The pooled cross-spectrum favors shared displacement over weak-channel evidence, but it neither estimates optical flow nor resolves phase unwrapping.

On reliable bins, set $\theta _ { p } = \mathrm { A r g } Z _ { p }$ and $U _ { p } = e ^ { \mathrm { i } \theta _ { p } } ;$ otherwise $U _ { p } ~ = ~ 1$ . Fractional powers use the fixed branch $\begin{array} { r } { \dot { U } _ { p } ^ { \beta } = e ^ { \mathrm { i } \beta \theta _ { \tau } } } \end{array}$

Frequency-confidence affinity. Let $\Omega _ { + }$ be the reliable non-DC spectral bins, with multiplicity weights $\kappa _ { \omega }$ for the real FFT. A tile-level consensus score across the reliable spectral bins is

$$
c _ { p } = \frac { \sum _ { \omega \in \Omega _ { + } } \kappa _ { \omega } | Z _ { p } ( \boldsymbol { \hat { \omega } } ) | } { \sum _ { \omega \in \Omega _ { + } } \kappa _ { \omega } W _ { p } ( \omega ) + \epsilon } \in [ 0 , 1 ] .\tag{7}
$$

Agreement across channel phases raises ${ \mathit { c } } _ { p } ,$ whereas conflicting evidence reduces it. With normalized radial frequency $\nu _ { \omega } \in [ 0 , 1 ]$ , define

$$
\begin{array} { r l } & { \tau ( \nu ) = \tau _ { \mathrm { l o w } } + ( \tau _ { \mathrm { h i g h } } - \tau _ { \mathrm { l o w } } ) \nu ^ { q } , } \\ & { g _ { p , \omega } = \sigma \Big ( \frac { c _ { p } - \tau ( \nu _ { \omega } ) } { T } \Big ) , } \end{array}\tag{8}
$$

where $q , T > 0$ and $\tau _ { \mathrm { h i g h } } ~ \geq ~ \tau _ { \mathrm { l o w } }$ . Thus high frequencies require stronger transport evidence. Frequency supplies a prior and confidence estimates reliability, without target access.

![](images/7c6780aa625adf5ffdc46a2536ea3e3f89848fed9eb0acb764d0ec03aa01dcf8.jpg)  
Figure 7: Phase-aware reconstruction with $Q =$ ${ \overline { { Z } } } ,$ where $Z$ is defined in Eq. (6).

Unified reconstruction. PAR aligns endpoint spectra before interpolation or one-sided extrapolation, applying the following rule to each channel and spatial tile:

$$
\widehat { F } _ { t , p , c } ( \omega ) = ( 1 - \alpha ) F _ { a , p , c } ( \omega ) U _ { p } ( \omega ) ^ { g _ { p , \omega } \alpha } + \alpha F _ { b , p , c } ( \omega ) U _ { p } ( \omega ) ^ { - g _ { p , \omega } ( 1 - \alpha ) } .\tag{9}
$$

An inverse real FFT recovers $\widehat { R } _ { t , p , c } ,$ and the tiles are assembled into $\widehat { R } _ { t } = \mathcal { P } _ { t } ( R _ { a } , R _ { b } )$ . Setting $g = 0$ gives direct value interpolation or extrapolation; setting $g = 1$ gives full endpoint phase transport. The anchor endpoints are reproduced exactly. Numerical conventions are given in Appendix A.2.

## 4.4 RECONSTRUCTION-COUPLED SPARSE ATTENTION

![](images/35f3bb2632fa911b814452937de30579c2fc16d785709dc25e8175db0991d59c.jpg)  
Figure 8: Initialize each anchor’s mask, then refine it using local and neighboring reconstruction gains while preserving a fixed historical KV budget at every anchor.

Local KV-omission importance. As in pooled historical routing (Lu et al., 2026), divide each head’s historical KV into M equal-size blocks. Pooled queries and keys/values give

$$
s _ { i , m } = \frac { \bar { q } _ { i } ^ { \top } \bar { k } _ { m } } { \sqrt { d _ { h } } } , \qquad \pi _ { i , m } = \mathrm { s o f t m a x } _ { m } ( s _ { i , m } ) , \qquad \bar { o } _ { i } = \sum _ { m } \pi _ { i , m } \bar { v } _ { m } .\tag{10}
$$

Pooled one-block omission scores initialize the masks in Figure 8:

$$
\delta _ { i , m } = \frac { \pi _ { i , m } } { 1 - \pi _ { i , m } + \epsilon } \big ( \bar { o } _ { i } - \bar { v } _ { m } \big ) , \qquad u _ { i , m } = \mathopen { } \mathclose \bgroup \left\| P \delta _ { i , m } \aftergroup \egroup \right\| _ { 2 } ^ { 2 } .\tag{11}
$$

We use $P = W _ { O } ^ { ( h ) }$ , the corresponding head slice of the native attention output projection. This inexpensive proxy excludes current/text interactions and later nonlinearities, so it estimates pooledoutput deviation rather than the complete nonlinear block error after all residual updates.

Pairwise reconstruction risk. Let $x _ { i , m } \in \{ 0 , 1 \}$ F̂indicate whether anchor i retains block m. For interval $( a , b )$ , define

$$
A _ { a , b } = \sum _ { t \in \mathcal { T } _ { a , b } } ( 1 - \alpha _ { t } ) ^ { 2 } , \quad B _ { a , b } = \sum _ { t \in \mathcal { T } _ { a , b } } \alpha _ { t } ^ { 2 } , \quad C _ { a , b } = \sum _ { t \in \mathcal { T } _ { a , b } } \alpha _ { t } ( 1 - \alpha _ { t } ) .\tag{12}
$$

Table 1: WorldMark results; higher is better. Bold marks the best non-native score per backbone. Speedups are taken from our matched runtime evaluation in Table 2 and are not measured on the WorldMark trajectories used to compute the quality and control scores.
<table><tr><td rowspan="3">Method</td><td colspan="7">Action Dynamics↑</td><td colspan="2">World Memory↑</td><td colspan="2">Visual Quality↑</td><td>Efficiency</td></tr><tr><td colspan="2">Direction Accuracy</td><td colspan="2">Direction Purity</td><td colspan="2">Motion Stability</td><td colspan="2">Response Latency</td><td rowspan="2">Local Global Revisit Perceptual Aesthetic Speedup↑</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td>Trans.</td><td>Rot.</td><td>Trans.</td><td>Rot.</td><td>Trans.</td><td>Trans.</td><td>Rot.</td><td></td></tr><tr><td colspan="10">Matrix-Game 3.0</td><td rowspan="3" colspan="2"></td></tr><tr><td colspan="10"></td></tr><tr><td>Native</td><td></td><td></td><td></td><td>98.053 87.203 83.605 82.966 79.832 98.184 93.636 96.296 76.763 47.755 80.124</td><td></td><td></td><td></td><td></td><td></td><td>77.174</td><td>58.688</td><td>1× 0.62×</td></tr><tr><td>SVG</td><td>97.44382.404</td><td></td><td>84.411 77.97480.68797.674 92.50896.28776.25345.63278.826</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>75.809</td><td>58.118</td></tr><tr><td>BSA</td><td>97.306 86.877</td><td></td><td>84.061</td><td>82.903</td><td>82.822 98.427 93.748 96.268 76.502</td><td></td><td></td><td></td><td>44.809</td><td>80.591</td><td>77.362</td><td>59.192 0.94×</td></tr><tr><td>MagCache</td><td>97.489</td><td>84.181 83.779</td><td>79.269</td><td>72.931</td><td></td><td>92.715 92.408 96.316</td><td></td><td>67.411</td><td>43.916 77.593</td><td>71.070</td><td>53.661</td><td>1.38×</td></tr><tr><td>TeaCache</td><td>97.972 86.158</td><td>83.572</td><td>81.514</td><td>79.612</td><td>96.616</td><td>92.949</td><td>96.137</td><td>75.433</td><td>46.561 78.638</td><td>75.654</td><td>57.635 55.727</td><td>1.44×</td></tr><tr><td>Sol Engine</td><td>97.915 86.390 97.944 87.566</td><td>82.195</td><td>81.299</td><td>73.957</td><td></td><td>95.513 91.914 96.231</td><td></td><td>65.351</td><td>45.327 77.215</td><td>72.049</td><td>57.653</td><td>1.56× 1.61×</td></tr><tr><td>Light Interaction CAST</td><td>98.02986.932</td><td>84.042</td><td>83.392 85.35081.995</td><td>76.010</td><td>98.695 79.83098.97391.52396.33776.531</td><td>593.728 96.304 70.812</td><td></td><td></td><td>45.720 80.999 46.64581.514</td><td>75.302 71.871</td><td>56.727</td><td>2.15×</td></tr><tr><td colspan="9"></td><td></td><td></td><td></td></tr><tr><td>HY-World 1.5 / WorldPlay</td><td></td><td></td><td>81.871 84.70556.30785.93583.29583.51591.07055.84087.545</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.990</td><td></td></tr><tr><td>Native</td><td colspan="10">90.32987.029</td><td></td><td>58.725</td><td>1×</td></tr><tr><td>BSA</td><td>91.751 82.674</td><td>78.691</td><td>80.189</td><td>52.570</td><td>88.723</td><td>85.132</td><td>99.118 92.510</td><td>52.945</td><td>87.884</td><td>78.927</td><td>59.356</td><td>0.48×</td></tr><tr><td>SVG</td><td>89.773 76.378</td><td>74.093</td><td>79.965</td><td>34.723</td><td>84.741</td><td>80.642</td><td>81.549</td><td>88.634 55.142</td><td>85.737</td><td>73.290</td><td>56.371</td><td>0.92×</td></tr><tr><td>TeaCache</td><td>91.862 73.061</td><td>78.389</td><td>73.369</td><td>30.601</td><td>81.831</td><td>81.21296.732</td><td></td><td>93.656 54.934</td><td>88.935</td><td>78.801</td><td>59.494</td><td>1.12×</td></tr><tr><td>MagCache</td><td></td><td>89.00773.72477.73774.15032.31081.16880.33898.72196.05350.32791.135</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.887</td><td>56.768</td><td>1.38×</td></tr><tr><td>Sol Engine</td><td>92.713 85.68880.25583.497</td><td>91.20280.07376.15278.050</td><td></td><td></td><td>29.890 85.492 73.665 98.923 95.430 54.391 90.894</td><td></td><td></td><td></td><td></td><td>77.718</td><td>58.917</td><td>1.56×</td></tr><tr><td>Light Interaction</td><td></td><td></td><td></td><td></td><td>55.552 93.45085.06097.003 95.13654.97991.517</td><td></td><td></td><td></td><td></td><td>78.244</td><td>57.940</td><td>2.59×</td></tr><tr><td>CAST</td><td>92.70086.77480.68684.39953.84394.23385.93098.83894.56655.43989.652</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.685</td><td>58.706</td><td>3.48×</td></tr></table>

If phase and confidence are held fixed, the phase factors in Eq. (9) have unit magnitude. The triangle inequality therefore bounds the propagated perturbation by $\bar { ( 1 - \alpha _ { t } ) } \| \delta R _ { a } \| + \bar { \alpha } _ { t } \| \delta R _ { b } \|$ . Squaring and summing, then substituting the pooled omission proxies, yields the routing surrogate

$$
\begin{array} { r } { \mathcal { R } _ { a , b , m } ( x _ { a , m } , x _ { b , m } ) = A _ { a , b } ( 1 - x _ { a , m } ) u _ { a , m } + B _ { a , b } ( 1 - x _ { b , m } ) u _ { b , m } } \\ { + 2 C _ { a , b } ( 1 - x _ { a , m } ) ( 1 - x _ { b , m } ) \sqrt { u _ { a , m } u _ { b , m } } . } \end{array}\tag{13}
$$

The endpoint terms account for each anchor’s reconstruction responsibility, while the joint-omission cross term explicitly couples neighboring masks by penalizing simultaneous omission of the same historical block. This frozen-transport surrogate omits cross-block interactions and changes in estimated phase and confidence, so it does not bound the full nonlinear reconstruction error (Appendix B).

Fixed-budget best responses. Let $\mathcal { E }$ contain adjacent-anchor pairs. We minimize

$$
\begin{array} { r l } { \mathcal { I } ( \boldsymbol { x } ) = \lambda _ { \mathrm { a n c } } \displaystyle \sum _ { i , m } ( 1 - x _ { i , m } ) u _ { i , m } + \lambda _ { \mathrm { r e c } } \left[ \sum _ { ( a , b ) \in \mathcal { E } } \sum _ { m } \mathcal { R } _ { a , b , m } + \mathcal { R } _ { \mathrm { b d r y } } ( \boldsymbol { x } ) \right] , } & { } \\ { \mathrm { s u b j e c t ~ t o } \quad \displaystyle \sum _ { m } x _ { i , m } = K _ { \mathrm { K V } } \quad \mathrm { f o r ~ e v e r y ~ a n c h o r } \ i , } & { } \end{array}\tag{14}
$$

with nonnegative weights and a fixed $K _ { \mathrm { K V } }$ . Here $\mathcal { R } _ { \mathrm { b d r y } }$ accounts for both anchors used to extrapolate skipped frames outside the anchor span (Appendix B.2). Text and permitted currentcontext interactions remain available; only historical visual blocks enter this budget. We initialize $\mathcal { M } _ { i } = \mathrm { T o p K } _ { m } ( u _ { i , m } , K _ { \mathrm { K V } } )$ . With neighboring masks fixed, define $G _ { i , m }$ as the reduction in $\mathcal { I }$ when anchor i retains block m. The conditional objective is linear in the anchor’s mask, so its response is

$$
\mathcal { M } _ { i } \gets \mathrm { T o p K } _ { m } ( G _ { i , m } , K _ { \mathrm { K V } } ) .\tag{15}
$$

Directional refinement passes update each anchor’s mask using local gains and reconstruction gains from updated neighboring masks. With strict improvements and ties preserved, repeated passes converge to a coordinate-wise local optimum; the fixed $L _ { s } = 2$ passes (forward and backward) need not reach that optimum for every evaluated input.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Models and protocol. We evaluate distilled Matrix-Game 3.0 and HY-World 1.5 (World-Play) (Wang et al., 2026; Sun et al., 2025), using one and two RTX A6000 GPUs per rollout, respectively; HY uses model/tensor parallel inference. Defaults retain 50% of current frames

![](images/435e004c436140a20217b7acca77111ea81d5ef1f9c5b46ba1df375ef979805c.jpg)  
Figure 9: Baseline rollouts on WorldMark. Matrix-Game 3.0 (left) and HY-World 1.5 (right).  
and 20% of historical KV blocks, with two directional refinement passes. Paired comparisons share checkpoints, initial observations, controls, and seeds. Mean steady-state latency uses CUDA synchronization, excludes startup and warm-up, and includes scoring, routing, reconstruction, and memory retrieval; VAE decoding is excluded from all reported steady-state speedup measurements.

Benchmarks and baselines. WorldMark (Xu et al., 2026) measures thirteen dimensions of control, memory, and visual quality, using 500 samples for every method on both backbones. The Light Interaction-style evaluation (Lu et al., 2026) uses the same 400 trajectories, evaluator, per-backbone hardware, and timing scope for all methods. It reports native-reference and self-comparison PSNR, SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018), VBench (Huang et al., 2024), and speedup. Native-reference fidelity measures agreement with the base model, not ground-truth correctness. Baselines are Native, SVG (Xi et al., 2025), LongCat-Video BSA (Meituan LongCat Team et al., 2025), TeaCache (Liu et al., 2025a), MagCache (Ma et al., 2025), Sol Engine (Li et al., 2026a), and Light Interaction (Lu et al., 2026). All baselines are re-run in our unified evaluation environment using their released implementations when available; Tables 1 and 2 contain no imported published scores. Appendix C.1 details the protocols and matched evaluation conditions.

## 5.2 MAIN RESULTS: QUALITY–EFFICIENCY TRADE-OFF

Tables 1 and 2 compare behavior and rollout quality, ranking non-native methods.

Table 2: VBench and other quality results. Bold: best nonnative score per backbone; arrows: preferred direction.
<table><tr><td rowspan="2">Method</td><td colspan="3">vs. Original</td><td colspan="2">Self-Comparison</td><td rowspan="2"></td><td rowspan="2">VBench↑ Speedup↑</td></tr><tr><td>PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">Matrix-Game 3.0</td></tr><tr><td>Native</td><td></td><td></td><td></td><td>17.07 0.5264</td><td>0.3283</td><td>0.7504</td><td>1×</td></tr><tr><td>SVG</td><td>12.380.4170</td><td>0.5587</td><td></td><td>14.480.4949</td><td>0.4406</td><td>0.7511</td><td>0.62×</td></tr><tr><td>BSA</td><td>13.340.4228</td><td></td><td>0.5795</td><td>16.660.5326</td><td>0.4094</td><td>0.7336</td><td>0.94×</td></tr><tr><td>MagCache</td><td>15.02</td><td>0.4902</td><td>0.4504</td><td>14.060.5398</td><td>0.3837</td><td>0.7192</td><td>1.38×</td></tr><tr><td>TeaCache</td><td>19.03 0.5619</td><td></td><td>0.3818</td><td>18.840.5765</td><td>0.3602</td><td>0.7146</td><td>1.44×</td></tr><tr><td>Sol Engine</td><td>13.81 0.4628</td><td>0.5262</td><td></td><td>16.01 0.6018</td><td>0.3751</td><td>0.7151</td><td>1.56×</td></tr><tr><td>Light Int.</td><td>14.67</td><td>0.4638</td><td>0.4442</td><td>15.590.4881</td><td>0.3658</td><td>0.7415</td><td>1.61×</td></tr><tr><td>CAST</td><td>15.73 0.4875</td><td>0.4657</td><td></td><td>19.350.5478</td><td>0.3581</td><td>0.7570</td><td>2.15×</td></tr><tr><td colspan="8">HY-World 1.5 / WorldPlay</td></tr><tr><td>Native</td><td></td><td></td><td></td><td>18.600.5678</td><td>0.5131</td><td>0.7201</td><td>1×</td></tr><tr><td>BSA</td><td>15.940.4639</td><td>0.3755</td><td></td><td>15.440.4205</td><td>0.3720</td><td>0.7943</td><td>0.48×</td></tr><tr><td>SVG</td><td>19.480.6028</td><td>0.2209</td><td></td><td>17.75 0.5299</td><td>0.2187</td><td>0.8082</td><td>0.92×</td></tr><tr><td>TeaCache</td><td>20.900.6588</td><td>0.1892</td><td></td><td>18.860.5743</td><td>0.2054</td><td>0.8150</td><td>1.12×</td></tr><tr><td>MagCache</td><td>18.76 0.6684</td><td>0.2390</td><td></td><td>18.28 0.5597</td><td>0.2247</td><td>0.8137</td><td>1.38×</td></tr><tr><td>Sol Engine</td><td>23.750.6540</td><td>0.2574</td><td></td><td>16.330.3829</td><td>0.2948</td><td>0.7293</td><td>1.56×</td></tr><tr><td>Light Int.</td><td>24.810.6500</td><td>0.1788</td><td></td><td>18.83 0.4570</td><td>0.5070</td><td>0.7846</td><td>2.59×</td></tr><tr><td>CAST</td><td>22.740.6745</td><td>0.2213</td><td></td><td>18.890.5798</td><td>0.2052</td><td>0.8178</td><td>3.48×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Quality–efficiency comparison. CAST achieves the highest speedup/VBench on both backbones: 2.15×/0.7570 on Matrix and 3.48×/0.8178 on HY, versus Light Interaction’s 1.61×/0.7415 and $2 . { \overset { \smile } { 5 } } 9 \times / 0 . 7 8 4 6$ On HY, TeaCache reaches similar VBench (0.8150) at only 1.12× speedup.

Interactive behavior and world memory. CAST leads seven of thirteen WorldMark dimensions on Matrix and six on HY (Table 1). Matrix local/global/revisit scores reach 76.531/46.645/81.514 versus Light Interaction’s 70.812/45.720/80.999. On HY, rotational stability/translational response latency improve to 94.233/85.930 from 93.450/85.060;

translation stability (53.843 versus 55.552) and revisit memory (89.652 versus 91.517) remain lower.

Rollout fidelity and self-comparison. On Matrix, CAST leads non-native self-comparison PSNR (19.35 versus TeaCache’s 18.84) and LPIPS (0.3581 versus 0.3602). It leads all three metrics on HY, narrowly exceeding TeaCache: 18.89 versus 18.86 PSNR, 0.5798 versus 0.5743 SSIM, and 0.2052 versus 0.2054 LPIPS. Figure 9 shows matched rollouts.

## 5.3 ABLATION STUDIES

Table 3 compares progressive/fixedbudget designs on Matrix-Game 3.0. Native retains all frames and historical KV; intermediate configurations retain 50% of frames with dense KV, and full CAST retains 20% of historical KV. Fixed-budget comparisons match checkpoints, prompts, controls, noise, candidate sets, and budgets.

Full CAST cuts latency from 4,319 to 2,007 ms/chunk (2.15×), raising VBench from 0.7504 to 0.7570 and revisit memory from 80.124 to 81.514.

Table 3: Progressive configurations and fixed-budget ablations on Matrix-Game 3.0. A: uniform/interleaved anchors + linear reconstruction; B: CAFS + linear reconstruction; C: CAFS + PAR. A–C use independent KV masks. Fixed-budget variants A–D use frame retention $r _ { f } = 0 . 5 0$ and historical KV retention r = 0.20. T-Motion, Local, Global, and Revisit are WorldMark scores. Speedup is relative to Native; latency is measured in ms/chunk and excludes VAE decoding for every configuration.
<table><tr><td>Variant</td><td>VBench↑ LPIPS↓ T-Motion↑ Local↑ Global↑ Revisit↑ Latency↓ Speedup↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A: Uniform</td><td>0.7218 0.5128</td><td>74.218 70.864</td><td>38.927</td><td>72.991</td><td>1,934</td><td>2.23×</td></tr><tr><td>B: + CAFS</td><td>0.7346 0.4974</td><td>76.482 72.943</td><td>41.586</td><td>75.648</td><td>1,951</td><td>2.21×</td></tr><tr><td>C: + PAR</td><td>0.7479 0.4776</td><td>78.219 75.046</td><td>44.387</td><td>79.206</td><td>1,976</td><td>2.19×</td></tr><tr><td>Native</td><td>0.7504</td><td>79.832 76.763</td><td>47.755</td><td>80.124</td><td>4,319</td><td>1.00×</td></tr><tr><td>+ CAFS</td><td>0.7370 0.4960</td><td>76.800 74.000</td><td>43.200</td><td>76.700</td><td>2,800</td><td>1.54×</td></tr><tr><td>+ CAFS+ PAR</td><td>0.7600 0.4550</td><td>80.200 76.900</td><td>47.500</td><td>82.000</td><td>2,825</td><td>1.53×</td></tr><tr><td>D: Full CAST</td><td>0.7570 0.4657</td><td>79.830 76.531</td><td>46.645</td><td>81.514</td><td>2,007</td><td>2.15×</td></tr></table>

T-Motion/local memory remain near Native; global memory falls from 47.755 to 46.645. Native + CAFS uses linear residual interpolation and two-anchor linear extrapolation outside the anchor span. PAR replaces recovery at matched frame retention; RCSA adds sparse historical access with coupled masks. RCSA changes KV retention; C to D isolates coupling at fixed sparsity.

Contribution at fixed budgets. At fixed frame/KV retention, A uses uniform/interleaved anchors, linear reconstruction, and independent KV masks; B adds CAFS, C replaces reconstruction with PAR, and D adds coupling (full CAST). A to B improves T-Motion/global memory by 2.264/2.659 points. B to C lowers LPIPS by 0.0198 and adds 3.558 revisit points. C to D adds 2.258/2.308 global/revisit points and lowers LPIPS by 0.0119. Respective costs are 17/25/31 ms/chunk. A to D lowers LPIPS by 0.0471 and adds 7.718 global-memory points for 73 ms/chunk. Coupling trades runtime for quality; gains depend on addition order.

## 5.4 EFFICIENCY ANALYSIS

Runtime and overhead. Table 4 compares Native, independent routing (C), and CAST with matched precision, resolution, chunk shape, sampling, and hardware per backbone. Steady-state latency includes selection, routing, reconstruction, and retrieval, excluding warm-up and VAE decoding. C combines CAFS/PAR with independent historical KV routing.

Table 4: Mean steady-state runtime (ms/chunk); OH includes routing and reconstruction.
<table><tr><td>Variant</td><td>Core</td><td>Route</td><td>Recon.</td><td>Total</td><td>OH</td><td>Speedup</td></tr><tr><td colspan="7">Matrix-Game 3.0</td></tr><tr><td>Native</td><td>4,319</td><td>0</td><td>0</td><td>4,319</td><td>0.0%</td><td>1.00×</td></tr><tr><td>Indep. (C)</td><td>1,887</td><td>43</td><td>46</td><td>1,976</td><td>4.5%</td><td>2.19×</td></tr><tr><td>CAST</td><td>1,891</td><td>74</td><td>42</td><td>2,007</td><td>5.8%</td><td>2.15×</td></tr><tr><td colspan="7">HY-World 1.5</td></tr><tr><td>Native</td><td>8,713</td><td>0</td><td>0</td><td>8,713</td><td>0.0%</td><td>1.00×</td></tr><tr><td>Indep. (C)</td><td>2,303</td><td>55</td><td>74</td><td>2,432</td><td>5.3%</td><td>3.58×</td></tr><tr><td>CAŚT</td><td>2,297</td><td>127</td><td>80</td><td>2,504</td><td>8.3%</td><td>3.48×</td></tr></table>

Overhead. Core time falls from 4,319 ms to 1,887/1,891 ms for C/CAST on Matrix and from 8,713 ms to 2,303/2,297 ms on HY. CAST’s routing/reconstruction costs 74/42 ms on Matrix and 127/80 ms on HY (5.8%/8.3% of latency). Coupling adds 31/72 ms/chunk over C, reducing speedup from 2.19×/3.58× to 2.15×/3.48×. On Matrix, this accompanies Table 3’s memory/LPIPS gains. HY’s larger native-to-accelerated core-time ratio yields greater speedup despite its higher overhead share.

Default operating point. Appendix C.3.3 identifies $( r _ { f } , r _ { \mathrm { K V } } ) = ( 0 . 5 0 , 0 . 2 0 )$ as a quality–latency knee. At 20% KV retention, cutting frame retention from 50% to 30% raises speedup from 2.15× to 2.52×, but lowers VBench from 0.7570 to 0.7441 and global memory from 46.645 to 42.738. Raising frame retention to 60% adds only 0.0031 VBench/0.574 global-memory points for 356 ms/chunk. At 50% frame retention, 30% KV retention adds 0.741/0.749 global/revisit points for 71 ms/chunk. Four versus two passes add only 0.136/0.158 global/revisit points for 31 ms/chunk; one forward–backward pair $( L _ { s } = 2 )$ captures most coupling gains at the selected operating point.

## 6 CONCLUSION

We introduced CAST, which connects anchor selection, residual reconstruction, and historical KV routing through their downstream effects. It balances control sensitivity with coverage, recovers skipped residuals using PAR, and coordinates historical supports to protect that recovery. CAST achieves 2.15× and 3.48× speedups on Matrix-Game 3.0 and HY-World 1.5, with VBench scores of 0.7570 and 0.8178. It also leads non-native methods on seven and six of thirteen WorldMark dimensions, respectively, demonstrating benefits for interactive behavior and visual quality. These findings highlight reconstruction dependencies as a useful guide for coordinating frame computation and historical attention under limited inference budgets.

## REFERENCES

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in Atari. In NeurIPS, 2024.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments. In ICML, 2024.

Leyang Chen, Junyi Wu, Shaoqiu Zhang, and Yulun Zhang. Worlddyncache: Risk-controlled latent dynamics approximation for diffusion world model. arXiv preprint arXiv:2608.01845, 2026.

Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, 2024.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In NeurIPS, 2022.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In ICLR, 2020.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In ICLR, 2025.

Wenkun He, Yuchao Gu, Junyu Chen, Junyi Wu, Wenhang Ge, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, et al. Dc-gen: Post-training diffusion acceleration with deeply compressed latent space. In ECCV, 2026.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In NeurIPS, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. In NeurIPS, 2022.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. In NeurIPS, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Chenyang Zhang, Michael S. Ryoo, and Tian Xie. Adaptive Caching for Faster Video Generation with Diffusion Transformers. In ICCV, 2025.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2020.

Yitong Li, Junsong Chen, Haopeng Li, Haozhe Liu, Jincheng Yu, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Sol video inference engine: Agent-native full-stack acceleration framework for efficient video generation. arXiv preprint arXiv:2606.23743, 2026a.

Zhiteng Li, Hanxuan Li, Junyi Wu, Kai Liu, Haotong Qin, Linghe Kong, Guihai Chen, Yulun Zhang, and Xiaokang Yang. Dvd-quant: data-free video diffusion transformers quantization. In ICLR, 2026b.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In CVPR, 2025a.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers. In ICCV, 2025b.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. In NeurIPS, 2022.

Jiacheng Lu, Haoyi Zhu, Sipei Yi, Enze Xie, Yu Li, and Cheng Zhuo. Light interaction: Training-free inference acceleration for interactive video world models. arXiv preprint arXiv:2605.31158, 2026.

Zhengyao Lyu, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. In ICLR, 2025.

Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching. In NeurIPS, 2024a.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. DeepCache: Accelerating diffusion models for free. In CVPR, 2024b.

Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, and Qi Tian. MagCache: Fast video generation with magnitude-aware cache. In NeurIPS, 2025.

Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. LongCat-Video Technical Report. arXiv preprint arXiv:2510.22200, 2025.

Simone Meyer, Oliver Wang, Henning Zimmer, Max Grosse, and Alexander Sorkine-Hornung. Phase-based frame interpolation for video. In CVPR, 2015.

Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are Sample-Efficient World Models. In ICLR, 2023.

William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In ICCV, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR, 2022.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. TACL, 2021.

Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. In ICLR, 2022.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In ICLR, 2021.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. In ICML, 2023.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Jian Tang, Jiawei Fan, Qingbin Liu, and Zheng Wei. FIS-DiT: Breaking the few-step video inference barrier via training-free frame interleaved sparsity. arXiv preprint arXiv:2605.11869, 2026.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In ICLR, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 2004.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. In SIGGRAPH, 2024.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, Christian Rupprecht, Daniel Cremers, Peter Vajda, and Jialiang Wang. Cache Me if You Can: Accelerating Diffusion Models through Block Caching. In CVPR, 2024.

Jianzong Wu, Liang Hou, Haotian Yang, Ye Tian, Pengfei Wan, Di ZHANG, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. In ICLR, 2026.

Junyi Wu, Zhiteng Li, Zheng Hui, Yulun Zhang, Linghe Kong, and Xiaokang Yang. QuantCache: Adaptive importance-guided quantization with hierarchical latent and layer caching for video generation. In ICCV, 2025a.

Junyi Wu, Zhiteng Li, Haotong Qin, Yulun Zhang, and Xiaokang Yang. Flashedit: Decoupling speed, structure, and semantics for precise image editing. arXiv preprint arXiv:2509.22244, 2025b.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, Jianfei Chen, Ion Stoica, Kurt Keutzer, and Song Han. Sparse VideoGen: Accelerating video diffusion transformers with spatial-temporal sparsity. In ICML, 2025.

Xiaojie Xu, Zhengyuan Lin, Kang He, Yukang Feng, Xiaofeng Mao, Yuanyang Yin, Yongtao Ge, and Kaipeng Zhang. WorldMark: A unified benchmark suite for interactive video world models. arXiv preprint arXiv:2604.21686, 2026.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models. In CVPR, 2025.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In NeurIPS, 2020.

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast Video Generation with Sliding Tile Attention. In ICML, 2025.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR, 2018.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating Diffusion Transformers with Token-wise Feature Caching. In ICLR, 2025.

Chang Zou, Shikang Zheng, Evelyn Zhang, Runlin Guo, Haohang Xu, Zhengyi Shi, Conghui He, Xuming Hu, and Linfeng Zhang. Rethinking Token-wise Feature Caching: Accelerating Diffusion Transformers with Dual Feature Caching. IEEE TIP, 2026.

Appendix overview. The appendix provides frame-scheduling and spectral-reconstruction details, the reconstruction-risk analysis, and the configuration and evaluation protocols supporting the main-text results and their interpretation under the stated modeling assumptions.

## A FRAME SCHEDULING AND SPECTRAL RECONSTRUCTION DETAILS

## A.1 CONTROL-SCORE NORMALIZATION AND SCHEDULING

For module $k \in \{ \mathrm { c a m } , \mathrm { a c t } \}$ , define $r _ { i , k } ^ { \ell } = \mathrm { R M S } ( \Delta _ { i , k } ^ { \ell } )$ over the frame’s spatial and channel coordinates. We use independent min–max normalization across the current chunk:

$$
\mathrm { N o r m } ( \Delta _ { i , k } ^ { \ell } ) = \frac { r _ { i , k } ^ { \ell } - \operatorname* { m i n } _ { j } r _ { j , k } ^ { \ell } } { \operatorname* { m a x } _ { j } r _ { j , k } ^ { \ell } - \operatorname* { m i n } _ { j } r _ { j , k } ^ { \ell } + \epsilon } .\tag{16}
$$

A constant module response contributes zero. If only one control branch exists, average the available branch rather than treating the absent branch as evidence of low sensitivity. This convention bounds $d _ { i } ^ { \ell }$ in [0, 1] and prevents arbitrary feature scales from dominating the debt term. It is not robust to outliers, so this normalization can be sensitive to extreme responses.

All $N _ { \mathrm { c u r } }$ current frames participate in deterministic Top- $K _ { f }$ selection under Eq. (5); ties use increasing frame index. Neither the first nor the last frame is mandatory. Matrix selects five of ten current latent frames, and HY selects two of four. When $K _ { f } = N _ { \mathrm { c u r } } ,$ , all current frames are computed and reconstruction is bypassed. Positive coverage debt raises the priority of repeatedly skipped frames; $H _ { \mathrm { c o v } }$ remains a nominal age scale rather than a strict refresh guarantee. Skipped frames between two selected anchors use PAR, while frames outside the selected-anchor span use one-sided PAR extrapolation from the two nearest anchors on that side.

The first block of each denoising evaluation is dense and resets ages. An anchor set is then selected once per block and shared across eligible visual attention and FFN residual sites. Fresh all-frame conditioning responses from block ℓ supply $d _ { i } ^ { \ell }$ for block $\ell + 1 ;$ ; where no fresh response is available, selection uses coverage debt alone. Conditioning and text updates remain dense, so the score does not require computing the visual residual being skipped.

## A.2 PHASE, CONFIDENCE, AND REAL-VALUED RECONSTRUCTION

Use spatial FFTs independently in each channel, with orthonormal normalization. This paper uses disjoint tiles; smaller boundary tiles are transformed at their own size, avoiding hidden padding amplification in the norm argument. Overlapping-window variants are outside the evaluation scope.

The ratio $| Z _ { p } ( \omega ) | / ( W _ { p } ( \omega ) + \epsilon )$ is an amplitude-weighted circular resultant length across channels. It approaches one when nonzero channel cross-spectra share a phase, and approaches zero when those phases cancel. Equation (7) aggregates this evidence over reliable non-DC bins. Reliability requires $W _ { p }$ and $| Z _ { p } |$ | to exceed documented numerical thresholds; the thresholds are $W _ { p } ( \omega ) > \eta _ { p }$ and $| Z _ { p } ( \omega ) | > \eta _ { p }$ , where $\eta _ { p } = 1 0 ^ { - 6 }$ max<sub>ω</sub> $W _ { p } ( \omega ) + 1 0 ^ { - 8 }$ , with FP32 accumulation and $\epsilon = 1 0 ^ { - 8 }$ If no reliable bin exists, set $c _ { p } = 0$ and $U _ { p } = \mathbf { \bar { 1 } }$ . A confidence score is not a calibrated probability of correct motion or a guarantee that phase transport will recover the target.

For a real transform, mirrored frequency bins must receive conjugate phase factors. Define phases on one representative of each conjugate pair and assign the negative phase to its mirror. Self-conjugate bins, including DC and applicable Nyquist points, receive $\mathbf { \bar { \boldsymbol { U } } } _ { p } = \mathbf { \dot { \boldsymbol { 1 } } }$ so that their coefficients remain real. The real-FFT weights $\kappa _ { \omega }$ account for the omitted conjugate half when summing energies. With these conventions the full-spectrum multiplier has unit modulus and preserves real-valuedness. Radial frequency is normalized by the largest represented spatial frequency radius.

For a translation with an unambiguous phase branch, substitution into Eq. (9) at $g = 1$ gives

$$
\widehat F _ { t } = ( 1 - \alpha ) F _ { a } e ^ { - \mathrm { i } \alpha \omega ^ { \top } d } + \alpha F _ { a } e ^ { - \mathrm { i } \omega ^ { \top } d } e ^ { \mathrm { i } ( 1 - \alpha ) \omega ^ { \top } d } = F _ { a } e ^ { - \mathrm { i } \alpha \omega ^ { \top } d } .\tag{17}
$$

This identity also holds for real α outside [0, 1] under the same translation and phase-branch assumptions. Boundary reconstruction uses Eq. (9) without clipping α or adding anchors. Extrapolation is not a convex blend: even with fixed transport, its worst-case perturbation amplification is $\left| 1 - \alpha \right| + \left| \alpha \right|$ Long extrapolation distances, changing motion, and occlusion can therefore degrade recovery. The routing surrogate accounts for propagated anchor errors, not the intrinsic error of extending a local motion model beyond its supporting anchors. This identity applies to coherent, identifiable transport modes, not arbitrary deformation or ambiguous Nyquist content. Principal endpoint phase only determines displacement modulo 2π at each frequency. Fractional powers of wrapped phase are generally not equal to fractional powers of the true unwrapped displacement. High consensus can coexist with wrapping ambiguity; the method does not claim to remove this limitation. Phase wrapping remains a limitation of principal-branch transport. The confidence- and frequency-dependent gate reduces aggressive corrections in unreliable regions but does not resolve phase ambiguity.

## B RECONSTRUCTION RISK AND HISTORICAL ROUTING

## B.1 EXACT SINGLE-BLOCK OMISSION IN THE POOLED MODEL

Removing historical block m and renormalizing the remaining pooled attention gives, for $\pi _ { i , m } < 1$

$$
\bar { o } _ { i } ^ { ( - m ) } = \frac { \bar { o } _ { i } - \pi _ { i , m } \bar { v } _ { m } } { 1 - \pi _ { i , m } } , \qquad \bar { o } _ { i } ^ { ( - m ) } - \bar { o } _ { i } = \frac { \pi _ { i , m } } { 1 - \pi _ { i , m } } \big ( \bar { o } _ { i } - \bar { v } _ { m } \big ) .\tag{18}
$$

The positive stabilizer in Eq. (11) modifies this identity near unit mass. It is exact only before stabilization and only for the stated pooled distribution. We use $P = W _ { O } ^ { ( h ) }$ , the corresponding head slice of the native attention output projection, with no additional learned or sketched projection. A pooled projected vector does not upper-bound a spatial residual tensor without further assumptions. The proxy concerns the attention residual site; subsequent FFN nonlinearities are not covered by this local estimate. The FFN residual is computed and reconstructed separately using the same block-level anchors as the corresponding visual attention residuals.

## B.2 A CONDITIONAL PROPAGATION BOUND

Freeze $U _ { p }$ and $g _ { p , \omega }$ while perturbing anchor residuals. Let $T _ { a , }$ <sub>t</sub> and $T _ { b , t }$ denote the resulting phase multipliers and transforms, including disjoint-tile assembly. Parseval’s identity gives $\| T _ { a , t } \overset { \bf { \bar { \alpha } } } { v } \| _ { 2 } =$ $\Vert \boldsymbol { v } \Vert _ { 2 }$ and similarly for $T _ { b , t }$ under Appendix A.2’s conventions.

Proposition 1 (Frozen-transport interval bound). For endpoint perturbations $v _ { a } , v _ { b }$ andfixed transport operators, the reconstruction change satisfies

$$
\sum _ { t \in \mathcal { Z } _ { a , b } } \left\| ( 1 - \alpha _ { t } ) T _ { a , t } v _ { a } + \alpha _ { t } T _ { b , t } v _ { b } \right\| _ { 2 } ^ { 2 } \leq A _ { a , b } \left\| v _ { a } \right\| _ { 2 } ^ { 2 } + B _ { a , b } \left\| v _ { b } \right\| _ { 2 } ^ { 2 } + 2 C _ { a , b } \left\| v _ { a } \right\| _ { 2 } \left\| v _ { b } \right\| _ { 2 } .\tag{19}
$$

Proof. Expand the squared norm at each t. The two individual terms follow from norm preservation. The cross term is at most $2 \alpha _ { t } ( 1 - \alpha _ { t } ) \left\| v _ { a } \right\| _ { 2 } \left\| v _ { b } \right\| _ { 2 }$ by Cauchy–Schwarz. Summing over the interval gives the three interval-dependent coefficients used in Eq. (12). □

For a single omitted block, insert $v _ { a } = ( 1 - x _ { a , m } ) \delta R _ { a , m }$ and $v _ { b } = ( 1 - x _ { b , m } ) \delta R _ { b , m }$ . Binary masks satisfy $( 1 - x ) ^ { 2 } = 1 - x$ . Replacing their true squared residual norms by $u _ { a , m }$ and $u _ { b , m }$ produces Eq. (13). For integer frame indices with gap $n = b - a \geq 1$ , the coefficients have closed forms

$$
A _ { a , b } = B _ { a , b } = \frac { ( n - 1 ) ( 2 n - 1 ) } { 6 n } , \qquad C _ { a , b } = \frac { n ^ { 2 } - 1 } { 6 n } .\tag{20}
$$

They vanish for adjacent frames $( n = 1 )$ , which have no skipped-frame responsibility.

For zero-based current-frame indices, define the left and right boundary sets $\mathcal { T } _ { L } = \{ t : 0 \leq t < a _ { 0 } \}$ and $\mathcal { T } _ { R } = \{ t : a _ { J } < t < N _ { \mathrm { c u r } } \}$ . Boundary extrapolation uses $( a _ { 0 } , a _ { 1 } )$ on the left and $( a _ { J - 1 } , a _ { J } )$ on the right. For $q \in \{ L , R \}$ , let $( a _ { q } , b _ { q } )$ denote the corresponding pair and define

$$
A _ { q } = \sum _ { t \in \mathcal { T } _ { q } } ( 1 - \alpha _ { t } ) ^ { 2 } , \qquad B _ { q } = \sum _ { t \in \mathcal { T } _ { q } } \alpha _ { t } ^ { 2 } ,
$$

$$
C _ { q } = \sum _ { t \in \mathbb { Z } _ { q } } | \alpha _ { t } ( 1 - \alpha _ { t } ) | , \qquad \alpha _ { t } = \frac { t - a _ { q } } { b _ { q } - a _ { q } } .\tag{21}
$$

For frozen transport, the per-frame perturbation is bounded by $\left| 1 - \alpha _ { t } \right| \left\| v _ { a _ { q } } \right\| _ { 2 } + \left| \alpha _ { t } \right| \left\| v _ { b _ { q } } \right\| _ { 2 } .$ . Squaring and summing gives the boundary analogue of Eq. (19); its cross coefficient is $C _ { q } \geq 0$ , even outside [0, 1]. Substituting the pooled omission proxies gives

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { b d r y } } ( x ) = \displaystyle \sum _ { q \in \{ L , R \} } \displaystyle \sum _ { m } \big [ A _ { q } ( 1 - x _ { a _ { q } , m } ) u _ { a _ { q } , m } + B _ { q } ( 1 - x _ { b _ { q } , m } ) u _ { b _ { q } , m } } \\ & { ~ + ~ 2 C _ { q } ( 1 - x _ { a _ { q } , m } ) ( 1 - x _ { b _ { q } , m } ) \sqrt { u _ { a _ { q } , m } u _ { b _ { q } , m } } \big ] . } \end{array}\tag{22}
$$

Empty boundary sets contribute zero. When $K _ { f } = 2 ,$ both boundary sets use the same anchor pair, but contain disjoint target frames; both contributions are included. Interior intervals contribute pairwise responsibility through Eq. (13); boundary pairs contribute the same responsibility under one-sided extrapolation. Together they give Eq. (14).

Three qualifications matter. First, recomputing U and g after an anchor perturbation introduces an additional operator-change term; the proposition does not bound it. Second, pooling and projection can distort omission norms. Third, summing per-block risks discards interactions among different omitted blocks and their joint softmax renormalization. Equation (14) is therefore a tractable surrogate rather than a global certificate for sparse inference. Moreover, the conservative norm bound removes the phase orientation itself: its coefficients depend on temporal interpolation responsibility, not on the realized phase or confidence values. The interior bound also applies to direct linear interpolation; the boundary version uses absolute cross coefficients for extrapolation. The claim supported by this derivation is coupling to the reconstruction dependency and norm-preserving operator class; phase-specific routing benefits require empirical evidence or a sharper estimator.

## B.3 RETAIN GAINS AND COORDINATE-WISE REFINEMENT

For anchor i with left/right neighbors l/r when present, fix their masks and define the retain gain

$$
\begin{array} { r l } & { G _ { i , m } = \lambda _ { \mathrm { a n c } } u _ { i , m } } \\ & { \phantom { G _ { i , m } = } + \lambda _ { \mathrm { r e c } } \mathbf { 1 } _ { l \mathrm { e x i s t s } } \left[ B _ { l , i } u _ { i , m } + 2 C _ { l , i } ( 1 - x _ { l , m } ) \sqrt { u _ { l , m } u _ { i , m } } \right] } \\ & { \phantom { G _ { i , m } = } + \lambda _ { \mathrm { r e c } } \mathbf { 1 } _ { r \mathrm { e x i s t s } } \left[ A _ { i , r } u _ { i , m } + 2 C _ { i , r } ( 1 - x _ { r , m } ) \sqrt { u _ { i , m } u _ { r , m } } \right] } \\ & { \phantom { G _ { i , m } = } + \lambda _ { \mathrm { r e c } } G _ { i , m } ^ { \mathrm { b d r y } } . } \end{array}\tag{23}
$$

where the boundary contribution is

$$
\begin{array} { r l r } {  { G _ { i , m } ^ { \mathrm { b d r y } } = \sum _ { q \in \{ L , R \} } \{ { \bf 1 } _ { i = a _ { q } } [ A _ { q } u _ { i , m } + 2 C _ { q } ( 1 - x _ { b _ { q } , m } ) \sqrt { u _ { i , m } u _ { b _ { q } , m } } ]  } } \\ & { } & \\ & { } & {  + { \bf 1 } _ { i = b _ { q } } [ B _ { q } u _ { i , m } + 2 C _ { q } ( 1 - x _ { a _ { q } , m } ) \sqrt { u _ { i , m } u _ { a _ { q } , m } } ] \} . } \end{array}\tag{24}
$$

Holding all other masks fixed, $\mathcal { I } =$ constant $\textstyle - \sum _ { m } x _ { i , m } G _ { i , m }$ . Thus the largest $K _ { \mathrm { K V } }$ gains solve the single-anchor problem exactly. A neighboring retention removes the joint-omission contribution, showing complementary coupling; large individual gains can still favor shared retention.

For efficiency, an implementation may restrict each anchor to a fixed candidate set $\mathcal { C } _ { i }$ of size $M _ { c } \ge K _ { \mathrm { K V } }$ , obtained from its local scores and neighboring proposals. Candidate sets are fixed before refinement; discarded coordinates remain zero and still count as omissions in neighboring gains. The same candidate sets must be used by the independent baseline. The convergence statement is then relative to these sets. Full-history pooled scoring and candidate construction are still counted in runtime; a small $M _ { c }$ does not make candidate discovery free. The unrestricted mathematical objective is recovered with $\mathcal { C } _ { i } = \{ 1 , \ldots , M \}$ , with no candidate truncation.

Proposition 2 (Termination and conditional optimality). Withfixed scores, coefficients, and candidate sets, exact Top-K best responses that accept only strict decreases in $\mathcal { I }$ terminate afterfinitely many changes. Afull unchanged directional refinement pass certifies coordinate-wise optimality over the candidates while holding all neighboring masksfixed.

Proof. There are at most $\prod _ { i } { \binom { | { \mathcal { C } } _ { i } | } { K _ { \mathrm { K V } } } }$ feasible masks. Every accepted change strictly decreases ${ \mathcal { I } } ,$ , so no mask configuration can recur. At termination, the best response at every anchor fails to improve the objective while all other masks remain fixed. No feasible single-anchor replacement can therefore improve it. This does not imply a globally optimal joint mask. □

Tie handling is necessary: deterministic tie-breaking alone does not establish strict descent. In floating point, accept only a decrease larger than a reported tolerance, yielding an approximate coordinate-wise condition. Stopping at the directional-pass cap preserves the budget but provides no convergence certificate. In the special cases $K _ { \mathrm { K V } } = { \dot { M } }$ or $\boldsymbol { M } \boldsymbol { \bar { = } } 0$ , historical routing is bypassed; if fewer than $K _ { \mathrm { K V } }$ blocks exist, all are kept and the actual available budget is reported.

```latex
Algorithm 1 Fixed-budget reconstruction-coupled routing
1 Input: ordered anchors, omission scores u, fixed candidates $\mathcal { C } _ { i } .$ , interval coefficients, budget $K _ { \mathrm { K V } }$
boundary coefficients from Eq. (21), and directional-pass cap $L _ { s } .$
2 Initialize $\mathcal { M } _ { i } \gets \mathrm { T o p K } _ { m \in \mathcal { C } _ { i } } \{ \hat { u } _ { i , m } , K _ { \mathrm { K V } } \}$ for every anchor; construct binary masks x.
3 For directional refinement pass $s = 1 , \ldots , L _ { s } { \mathrm { : } }$
Set changed ← false.
For i in forward order if s is odd, backward order otherwise:
Compute $G _ { i , m }$ using the latest neighbor masks (Eq. (23)).
7 Let $\begin{array} { r } { \dot { \mathcal { M } } _ { i } ^ { \prime } \gets \mathrm { T o p K } _ { m \in \mathcal { C } _ { i } } ( G _ { i , m } , K _ { \mathrm { K V } } ^ { \mathrm { ' } } ) . } \end{array}$
8 $\begin{array} { r } { \mathbf { I f } \sum _ { m \in \mathcal { M } _ { i } ^ { \prime } } G _ { i , m } > \sum _ { m \in \mathcal { M } _ { i } } G _ { i , m } { : } } \end{array}$
9 Replace ${ \mathcal { M } } _ { i } ,$ update $x ,$ and set changed ← true.
10 If no mask changed, stop.
11 Return: historical masks with exactly $K _ { \mathrm { K V } }$ retained blocks per anchor.
```

## B.4 BLOCK EXECUTION ORDER

Select the anchor set once per block using fresh lagged control scores and updated debt, or debt alone when no fresh score is available. At each eligible visual attention site, form pooled historical statistics and refine masks using Algorithm 1; evaluate anchor attention residuals and reconstruct skipped updates. At each eligible FFN site, compute anchor FFN residuals and reconstruct the remaining updates separately with the same anchors. Every update is added to its own current-frame hidden state. Conditioning and text updates remain dense, all permitted current-frame K/V are constructed, and native causal permissions are preserved. Debt and fresh all-frame control signals then determine the next block’s anchors using information already available from the current evaluation.

## B.5 COMPLEXITY AND IMPLEMENTATION

Let each current frame contain S tokens of width $D ,$ each historical block contain $B _ { v }$ tokens, and $N _ { 0 }$ denote retained current/text KV tokens. Direct current-frame attention changes from ${ \cal O } ( N _ { \mathrm { c u r } } S ( N _ { 0 } +$ $M B _ { v } ) D )$ to $O ( K _ { f } S ( N _ { 0 } + K _ { \mathrm { K V } } B _ { v } ) D )$ , and directly evaluated current-frame FFNs change from $O ( N _ { \mathrm { c u r } } S D ^ { 2 } )$ to $\hat { O } ( K _ { f } S D ^ { 2 } )$ . Any all-frame conditioning or current-KV projections remain explicit costs; these expressions do not imply that the entire block scales by the anchor ratio.

For tiles of at most $B _ { p }$ spatial positions, batched anchor FFTs and skipped-frame inverse FFTs cost $O ( N _ { \mathrm { c u r } } S D \log B _ { p } )$ . Pooled scoring costs $O ( K _ { f } M D )$ , while $L _ { s }$ directional refinement passes over fixed candidate sets of size $M _ { c }$ cost $\mathsf { \bar { O } } ( L _ { s } K _ { f } \dot { M } _ { c } )$ plus Top- $\mathbf { \nabla } . K$ selection. Memory-context processing, the dense warm-up block, KV storage, control scoring, and tensor gathering remain in the total. Historical sparsity reduces attention reads without automatically shrinking the resident KV cache. Boundary extrapolation reuses the first or last adjacent-anchor pair and adds no directly computed visual residuals; unlike residual copying, it incurs spectral reconstruction work for boundary targets. The anchor budget and asymptotic reconstruction cost are unchanged, but measured latency need not be identical. Net wall-clock gains therefore require the runtime accounting in Section 5.4; operation-count savings alone do not establish interactive speed.

## C EVALUATION DETAILS AND ADDITIONAL ANALYSES

## C.1 IMPLEMENTATION AND EVALUATION RECORD

Qualitative examples use Matrix sample fr005\_006 and HY sample ts000\_006.

Tables 5 and 6 summarize backbone, CAST, and evaluation settings. Backbone identities and chunk conventions follow the Matrix release, its model configuration, and the HY release. The tables specify hooks, hardware, numerical settings, and evaluators; timing follows Section 5.4. We use the released Light Interaction block-sparse backend with CAST’s historical-block selection policy.

Native-chunk compatibility. Matrix uses ten current latent frames and dynamically selects five anchors at $r _ { f } = 0 . 5 0$ ; HY uses four and dynamically selects two. Both backbones select over the entire current chunk without mandatory endpoints. The anchor set is shared across eligible visual residual sites within each transformer block. Matrix budget experiments use the discrete counts $K _ { f } \in \{ 3 , 5 , 6 \}$ }, corresponding to r<sub>f</sub> ∈ {0.30, 0.50, 0.60}.

Benchmark scope and provenance. WorldMark uses 500 samples for every method on both backbones. The Light Interaction-style evaluation uses the same 400 trajectories for every method: 200 left–right and 200 forward–backward. All baselines, including Native and Sol Engine, are re-run in our unified environment; the tables contain no scores imported from published result tables. Within each backbone, methods share the evaluator, per-rollout hardware, and timing scope. Results are point estimates without confidence intervals. Speedups in Table 1 come from our matched runtime evaluation in Table 2, not the WorldMark trajectories. Non-native rankings exclude Native.

Controls and accounting. Paired runs share initial observations, controls, and seeds; calibration and evaluation trajectories are disjoint. Matched-component ablations fix K<sub>f</sub>, K<sub>KV</sub>, block layouts, candidate sets, and dense-block schedules. Matrix uses one RTX A6000 per rollout; HY uses two for model/tensor parallel inference. The eight-GPU RTX A6000 server parallelizes independent runs on unused devices. For each of three seeds, timing uses ten warm-up and fifty measured chunks with CUDA synchronization. Mean steady-state latency includes scoring, routing, reconstruction, and retrieval, but excludes VAE decoding and the initial startup phase.

WorldMark’s thirteen dimensions comprise translation/rotation variants of direction accuracy, direction purity, motion stability, and response latency; local, global, and revisit memory; and perceptual and aesthetic quality. The Light Interaction protocol reports PSNR, SSIM, and LPIPS against paired native rollouts and under self-comparison, together with VBench. Agreement with Native measures approximation fidelity rather than ground-truth correctness.

Table 5: Backbone and execution configuration (8× RTX A6000 server).
<table><tr><td>Field</td><td>Matrix-Game 3.0</td><td>HY-World 1.5</td></tr><tr><td>Checkpoint and code provenance</td><td>Skywork/Matrix-Game-3.0, base_distilled_model;released code with CAST integration</td><td>tencent/HY-WorldPlay, ar_distilled_action_model; HunyuanVideo 8B path</td></tr><tr><td></td><td>GPU model and count 1× RTX A6000 per rollout; batch 1</td><td>2× RTX A6000 per rollout; model/tensor parallel inference; batch 1</td></tr><tr><td>Software and attention backend</td><td>Python 3.12; PyTorch 2.6; CUDA 12.4; Triton 3.2; FlashAttention 2.7 for dense paths; released Light Interaction</td><td>Python 3.10; PyTorch 2.6; CUDA 12.4; Triton 3.2; FlashAttention 2.7 for dense paths; released Light Interaction</td></tr><tr><td>Model / attention / FFT precision</td><td>block-sparse execution backend BF16 model and QKV; FP32 softmax and pooled reductions; FP32 / complex64 FFT; pooled reductions; FP32 / complex64 FFT; no INT8/FP8</td><td>block-sparse execution backend BF16 model and QKV; FP32 softmax and no FP8</td></tr><tr><td>Resolution and native chunk</td><td>704 × 1280; steady-state 40 decoded / 10 new latent frames; separate 57-frame startup</td><td>480 × 832; steady-state 16 decoded / 4 latent frames; first chunk has 13 decoded frames</td></tr><tr><td>Denoising schedule</td><td>3 evaluations; retain the released distilled scheduler and its native timestep ordering</td><td>4 evaluations; native few-step AR scheduler; no denoising cache</td></tr><tr><td>layout</td><td>History and KV block Keep native camera-aware retrieval and temporal context; blocks (t, h, w) = (1, 8, 8) in token coordinates; mask partial blocks</td><td>Keep native reconstituted context memory and causal permissions; blocks (1, 8, 8) in token coordinates; mask partial blocks</td></tr><tr><td>skip scope</td><td>Control-score hooks / Dense control branches where present (released action blocks 0–14); skip only current visual attention/FFN residuals; use debt-only selection where no fresh control</td><td>Dense native action-conditioning path at verified interfaces; skip only current visual attention/FFN residuals; never skip text or conditioning updates</td></tr><tr><td>Current-frame KV construction</td><td>Project K/V for all current visual tokens, retain all permitted current/text KV; sparse Q computation only for selected frames</td><td>Project K/V for all current visual tokens; preserve native text/visual masks and permitted current/text KV</td></tr><tr><td>Dense / sparse schedule</td><td>First transformer block dense at every denoising evaluation; all later eligible visual residuals sparse; reset debt per</td><td>First transformer block dense at every denoising evaluation; all later eligible visual residuals sparse; reset debt per</td></tr></table>

Table 6: CAST and evaluation settings for both backbones under the shared comparison protocol.
<table><tr><td>Field</td><td>Matrix-Game 3.0</td><td>HY-World 1.5</td></tr><tr><td> $r _ { f } , K _ { f } , \lambda _ { \mathrm { c o v } }$ </td><td>0.50, 5 of 10, 0.50; dynamic Top-5 over all current frames; boundary extrapolation uses the two nearest anchors</td><td>0.50, 2 of 4, 0.50; dynamic Top-2 over all current frames; boundary extrapolation uses the two nearest anchors</td></tr><tr><td>Tiles and reliable spectral bins</td><td> $8 \times 8$  spatial token tiles, disjoint;  $W _ { p } , | \tilde { Z _ { p } | } > 1 0 ^ { - 6 }$  max  $W _ { p } + 1 0 ^ { - 8 }$  ; FP32  $( 0 . 2 5 , 0 . 7 5 , 2 , 0 . 1 0 ) ;$ </td><td> $8 \times 8$  spatial token tiles, disjoint;  $W _ { p } , | Z _ { p } | > 1 0 ^ { - 6 }$  max  $W _ { p } + 1 0 ^ { - 8 }$  ; FP32  $( 0 . 2 5 , 0 . 7 5 , 2 , 0 . 1 0 )$  ; principal phase</td></tr><tr><td> $\tau _ { \mathrm { l o w } } , \tau _ { \mathrm { h i g h } } , q , T$ </td><td>principal phase branch; DC/Nyquist transport disabled</td><td>branch; DC/Nyquist transport disabled  $K _ { \mathrm { K V } } = \lceil 0 . 2 0 M \rceil ;$  target  $r _ { \mathrm { K V } } = 0 . 2 0 ;$ </td></tr><tr><td> $K _ { \mathrm { K V } , } r _ { \mathrm { K V } , } M _ { c }$ </td><td> $K _ { \mathrm { K V } } = \lceil 0 . 2 0 M \rceil ; \mathrm { t a r g e t } r _ { \mathrm { K V } } = 0 . 2 0 ;$   $M _ { c } = M$  (full candidate pool); bypass if  $M = 0$ </td><td> $M _ { c } = M ; { \mathrm { r e p o r t } }$  realized block fraction</td></tr><tr><td>Projection P and scaling</td><td> $P = W _ { O } ^ { ( h ) }$  , native attention output-projection head slice; no additional learned/sketched projection; FP32 pooled reductions</td><td>Same native output-projection head slice and FP32 reductions; squared Euclidean norm</td></tr><tr><td>Routing weights / stopping</td><td> $( \lambda _ { \mathrm { a n c } } , \lambda _ { \mathrm { r e c } } ) = ( 1 , 1 ) ; L _ { s } = 2$  directional passes; accept  $\mathrm { \hat { \Delta } } J > 1 0 ^ { - 6 } \mathrm { \tilde { \ m a x } } ( J , 1 0 ^ { - 8 } )$ </td><td> $( \lambda _ { \mathrm { a n c } } , \lambda _ { \mathrm { r e c } } ) = ( 1 , 1 ) ; L _ { s } = 2$  directional passes; same relative tolerance</td></tr><tr><td>Calibration / evaluation / seeds</td><td>32 disjoint calibration trajectories; retain main-text 500 WorldMark / 400 LI counts; paired seed 42; sensitivity seeds 0, 1, 2 Preserve benchmark controls; 12-iteration</td><td>32 disjoint calibration trajectories; retain main-text 500 WorldMark samples per method and 400 LI counts; same seed plan Preserve benchmark controls; diagnostic</td></tr><tr><td>Trajectories and horizons</td><td>diagnostic rollout:  $5 7 + 1 1 \times 4 0 = 4 9 7$  decoded frames; startup excluded from steady-state timing</td><td>horizons 125, 253, 509 decoded frames (32, 64, 128 latent frames)</td></tr><tr><td>Warm-up and repetitions</td><td>10 unmeasured chunks; 50 timed chunks per seed; 3 seeds; CUDA synchronization; scoring, routing, reconstruction, and mean steady-state latency</td><td>Same repetitions and timing scope; retrieval included; VAE decoding excluded; startup excluded</td></tr><tr><td>Quality/control evaluators</td><td>Official VBench and WorldMark protocols; LPIPS-AlexNet; RGB [0, 1] for PSNR/SSIM</td><td>Same evaluators, revisions, and r preprocessing; standalone VGGT diagnostic is separate from WorldMark</td></tr><tr><td>Uniform-anchor controlled baseline</td><td>Table 3, Variant A: uniform/interleaved anchors, linear residual recovery, independent KV routing</td><td>Same controlled ablation definition</td></tr><tr><td>Sol Engine</td><td>Official publicly released implementation, Official publicly released implementation, fully re-run in our unified environment</td><td>fully re-run with the same evaluator, hardware, and timing scope as CAST</td></tr></table>

## C.2 DIAGNOSTIC MEASUREMENTS

Oracle frame difficulty. The frame-difficulty diagnostic replays identical block inputs through the dense reference. For an eligible interior frame, its neighboring anchors are fixed, its direct residual computation is omitted, and Eq. (1) measures the resulting error. Figure 3 compares sensitivity at block ℓ with error at $\ell + 1$ , matching the deployed prediction lag. This is an association diagnostic rather than a causal guarantee for frame selection.

Oracle corrections and phase progression. Let $F _ { t }$ and $F _ { t } ^ { \mathrm { l i n } }$ be the spatial Fourier transforms of the reference and interpolated residuals. The two diagnostic oracle corrections are

$$
F _ { t } ^ { \mathrm { p h a s e } } = | F _ { t } ^ { \mathrm { l i n } } | e ^ { \mathrm { i } \mathrm { A r g } F _ { t } } , \qquad F _ { t } ^ { \mathrm { m a g } } = | F _ { t } | e ^ { \mathrm { i } \mathrm { A r g } F _ { t } ^ { \mathrm { l i n } } } .\tag{25}
$$

These reference-based diagnostics do not run at inference. For coherent constant-velocity translation,

$$
R _ { b } ( x ) = R _ { a } ( x - d ) \implies F _ { b } ( \omega ) = F _ { a } ( \omega ) e ^ { - \mathrm { i } \omega ^ { \top } d } , \qquad F _ { t } ( \omega ) = F _ { a } ( \omega ) e ^ { - \mathrm { i } \alpha \omega ^ { \top } d } .\tag{26}
$$

This implies linear unwrapped phase at every frequency, not preferential low-frequency linearity. To compare trajectories in Figure 4, we fix one coefficient within a channel and spatial region and define

$$
p _ { t , \omega } = \frac { \tilde { \phi } _ { t } ( \omega ) - \tilde { \phi } _ { a } ( \omega ) } { \tilde { \phi } _ { b } ( \omega ) - \tilde { \phi } _ { a } ( \omega ) } .\tag{27}
$$

Uniform progression gives $p _ { t , \omega } = \alpha$ , with endpoints fixed to zero and one. Negative values indicate motion opposite to the net endpoint phase displacement. Weak-amplitude coefficients, near-zero

endpoint phase differences, and inconsistent phase branches can distort this normalization. Temporal comparisons in Figure 3 independently rescale the two curves and preserve the ℓ-to-ℓ + 1 lag.

Spectral decomposition. For radial band B, the absolute error-energy fraction is

$$
\eta ( \mathcal { B } ) = \frac { \sum _ { p , c , \omega \in \mathcal { B } } \kappa _ { \omega } | F _ { t , p , c } - F _ { t , p , c } ^ { \mathrm { l i n } } | ^ { 2 } } { \sum _ { p , c , \omega } \kappa _ { \omega } | F _ { t , p , c } - F _ { t , p , c } ^ { \mathrm { l i n } } | ^ { 2 } + \epsilon } .\tag{28}
$$

Oracle phase and magnitude corrections use Eq. (25) and the same tile/FFT conventions. Phase is undefined for zero-energy spectra. Let $\phi _ { i , p , c } \doteq \mathrm { A r g } F _ { i , p , c } ;$ the transport diagnostic measures the energy-weighted circular distance

$$
\begin{array} { r } { \left| \operatorname { w r a p } _ { [ - \pi , \pi ) } \left( \phi _ { t , p , c } - \phi _ { a , p , c } - \alpha \theta _ { p } \right) \right| . } \end{array}\tag{29}
$$

Routing diagnostics. With the anchor set, candidate pool, and per-anchor KV budget fixed, Table 8 compares independent omission-importance Top-K masks with coupled masks against a densehistory reference. Anchor perturbations and interval reconstruction errors are measured on identical block inputs, separating the immediate approximation from subsequent rollout drift. The objective is a frozen-transport surrogate; the reported errors measure the resulting residual approximations rather than certify the bound for the complete nonlinear inference process.

Generation quality and controls. The quality score Q uses the official VBench quality aggregation (Huang et al., 2024). Paired RGB PSNR uses a declared intensity range and the same initial observation, control sequence, and random seed. It measures closeness to the dense generator, which can itself be wrong. Fixed-input residual diagnostics isolate approximation effects from subsequent context drift accumulated over later chunks in free-running rollouts.

The camera-control diagnostic compares inferred and commanded relative rotations in a common coordinate system. The rotation error is the geodesic angle arccos $( \mathrm { c l i p } ( ( \mathrm { t r } ( R _ { \mathrm { c m d } } ^ { \top } R _ { \mathrm { e s t } } ) - 1 ) / 2 , - 1 , 1 ) )$ Translation-direction error is reported only for commands with nonzero movement. The complementary camera-pose diagnostic uses VGGT-1B (Wang et al., 2025). Its OpenCV world-to-camera matrices are converted to camera-to-world poses, commanded poses are transformed to the same axes, and both trajectories are expressed relative to the first camera. Relative rotations and unit translation directions are evaluated in degrees, without fitting a similarity transform to each generated rollout. Translation-direction scoring requires finite poses, a proper orthonormal rotation (tolerance $1 0 ^ { - 3 } )$ and nonzero translation norm $( > 1 0 ^ { - 6 }$ in normalized coordinates). The valid-pose rates are 94.6% (Matrix) and 96.2% (HY). This diagnostic complements the official WorldMark evaluation. Visual quality alone does not establish discrete action consistency. Revisit consistency assesses accumulated drift. These protocols assess control adherence and accumulated drift alongside the benchmark quality metrics.

Latency measurement. We report mean steady-state chunk latency over fifty timed chunks per seed and three seeds, following ten unmeasured warm-up chunks. CUDA synchronization brackets the measured execution, including retrieval, scoring, routing, and reconstruction while excluding VAE decoding; startup is excluded. Speedup divides matched Native latency by method latency at identical resolution, chunk size, and per-backbone hardware. Generation throughput is decoded frames per measured time and does not by itself equal the interaction response rate. Quality comparisons remain point estimates without confidence intervals for the reported evaluation samples.

## C.3 ADDITIONAL ABLATIONS AND QUALITATIVE COMPARISONS

## C.3.1 CONTROLLED DESIGN ALTERNATIVES

The main paper retains the progressive component table; the following one-factor comparisons separate the choice of selector and reconstruction operator under the same end-to-end evaluation.

Controlled frame-selection and reconstruction ablations. Following the standard one-factor-ata-time organization used in diffusion-acceleration studies, Table 7 changes one design choice while holding the rest of the pipeline fixed. In the first panel, all selectors use PAR and coupled routing; in the second, all reconstruction operators use control-aware anchors and coupled routing. Each row is evaluated end-to-end with the same metrics, rather than mixing selector-only and reconstruction-only diagnostics with different evaluation targets in one set of columns.

Table 7: Controlled frame-selection and reconstruction ablations on Matrix-Game 3.0. Every row uses $r _ { f } = 0 . 5 0 , r _ { \mathrm { K V } } = 0 . 2 0$ , and the same end-to-end evaluation. T-Motion, Local, Global, and Revisit are representative WorldMark scores; all are higher-is-better.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">VBench↑</td><td rowspan="2">LPIPS↓</td><td colspan="4">WorldMark↑</td></tr><tr><td>T-Motion</td><td>Local</td><td>Global</td><td>Revisit</td></tr><tr><td colspan="7">Frame selection; PAR and coupled routing fixed</td></tr><tr><td>Uniform/interleaved</td><td>0.7437</td><td>0.4884</td><td>76.914</td><td>73.478</td><td>42.806</td><td>77.138</td></tr><tr><td>Latent-motion magnitude</td><td>0.7506</td><td>0.4769</td><td>79.312</td><td>75.087</td><td>44.578</td><td>79.341</td></tr><tr><td>Control sensitivity</td><td>0.7570</td><td>0.4657</td><td>79.830</td><td>76.531</td><td>46.645</td><td>81.514</td></tr><tr><td colspan="7">Reconstruction; control-aware selection and coupled routing fixed</td></tr><tr><td>Linear interpolation</td><td>0.7392</td><td>0.4948</td><td>77.103</td><td>73.337</td><td>42.164</td><td>76.782</td></tr><tr><td>Phase transport without confidence</td><td>0.7495</td><td>0.4781</td><td>78.864</td><td>75.008</td><td>44.612</td><td>79.203</td></tr><tr><td>Full PAR</td><td>0.7570</td><td>0.4657</td><td>79.830</td><td>76.531</td><td>46.645</td><td>81.514</td></tr></table>

With reconstruction and routing fixed, control sensitivity improves VBench by 0.0133 over uniform/interleaved selection, reduces LPIPS by 0.0227, and raises global and revisit memory by 3.839 and 4.376 points. Latent-motion magnitude is consistently intermediate, suggesting that generic motion is useful but does not fully capture action-conditioned frame importance. With selection and routing fixed, full PAR improves VBench by 0.0178 over linear interpolation, reduces LPIPS by 0.0291, and gains 4.481 global-memory and 4.732 revisit-memory points. Phase transport without confidence recovers part of this gap, while the remaining gain from frequency-dependent confidence is visible across both visual and memory metrics. Residual-domain measurements are reported separately in Appendix C.2; these tables use uniform end-to-end metrics for direct comparison.

## C.3.2 FIXED-INPUT COUPLING DIAGNOSTICS

Does reconstruction coupling add value? We compare C and D on identical block inputs with the same control-aware anchors, historical candidate pool, and per-anchor KV cardinality. Independent routing retains the Top-K blocks under the local omission score $u _ { i , m } ;$ coupled routing starts from those masks and applies one forward and one backward directional refinement pass using the reconstruction-coupled objective in Eq. (14). This changes which historical blocks are retained without changing the 20% historical KV retention budget at any anchor.

Table 8: Fixed-input coupling ablation on Matrix-Game 3.0. Anchor and interval errors are normalized residual errors under identical anchors and KV budgets. Local, Global, and Revisit are WorldMark memory scores measured on the paired rollouts for both routing variants.
<table><tr><td>KV routing</td><td>Anchor error↓</td><td>Interval error↓</td><td>VBench↑</td><td>Local↑</td><td>Global↑</td><td>Revisit↑</td></tr><tr><td>Independent (C)</td><td>0.0831</td><td>0.1127</td><td>0.7479</td><td>75.046</td><td>44.387</td><td>79.206</td></tr><tr><td>Coupled,  $L _ { s } = 2 \left( \mathrm { D } \right)$ </td><td>0.0774</td><td>0.1013</td><td>0.7570</td><td>76.531</td><td>46.645</td><td>81.514</td></tr></table>

Coupling reduces anchor-local error by 6.9% and interval reconstruction error by 10.1%. The larger reduction between anchors is accompanied by gains of 1.485, 2.258, and 2.308 points in local, global, and revisit memory. This pattern supports the reconstruction-risk objective: coupled routing selects historical evidence that is useful both to each endpoint and to the frames reconstructed between adjacent endpoints. C and D share the same budget, ruling out additional KV capacity as the cause.

## C.3.3 BUDGET AND REFINEMENT SENSITIVITY

Sensitivity to computation budgets. We vary the retained-frame ratio $r _ { f } = K _ { f } / N _ { \mathrm { c u r } }$ and retainedhistory ratio $r _ { \mathrm { K V } } \doteq K _ { \mathrm { K V } } / M$ one at a time on Matrix-Game 3.0, while fixing $L _ { s } = 2$ . Table 9 reports both visual quality and the WorldMark global and revisit memory scores most directly affected by current-frame and historical-context budgets in these comparisons.

Table 9: Budget sensitivity on Matrix-Game 3.0 with two directional refinement passes. Ratios are retained fractions; Global and Revisit are WorldMark memory scores, and latency is milliseconds per chunk. All settings use identical prompts, actions, and paired random seeds.
<table><tr><td></td><td></td><td></td><td></td><td colspan="2">WorldMark↑</td><td></td><td></td></tr><tr><td> $r _ { f }$ </td><td>rKV</td><td>VBench↑</td><td>LPIPS↓</td><td>Global</td><td>Revisit</td><td>Latency↓</td><td>Speedup↑</td></tr><tr><td>0.30</td><td>0.20</td><td>0.7441</td><td>0.4936</td><td>42.738</td><td>77.246</td><td>1,717</td><td>2.52×</td></tr><tr><td>0.50</td><td>0.10</td><td>0.7512</td><td>0.4788</td><td>42.911</td><td>77.926</td><td>1,909</td><td>2.26×</td></tr><tr><td>0.50</td><td>0.20</td><td>0.7570</td><td>0.4657</td><td>46.645</td><td>81.514</td><td>2,007</td><td>2.15×</td></tr><tr><td>0.50</td><td>0.30</td><td>0.7584</td><td>0.4608</td><td>47.386</td><td>82.263</td><td>2,078</td><td>2.08×</td></tr><tr><td>0.60</td><td>0.20</td><td>0.7601</td><td>0.4529</td><td>47.219</td><td>82.487</td><td>2,363</td><td>1.83×</td></tr></table>

Reducing $r _ { f }$ from 0.50 to 0.30 saves 290 ms but lowers VBench by 0.0129 and reduces global and revisit memory by 3.907 and 4.268 points. Raising $r _ { f }$ to 0.60 provides only 0.0031 additional VBench and less than one point of memory improvement, while reducing speedup from 2.15× to 1.83×. Historical context shows a similar saturation pattern. Retaining 10% rather than 20% of historical KV blocks loses 3.734 global-memory and 3.588 revisit-memory points, whereas increasing the ratio from 20% to 30% adds 71 ms for gains of only 0.741 and 0.749 points. We therefore use $( r _ { f } , r _ { \mathrm { K V } } ) = ( 0 . 5 0 , 0 . 2 0 )$ as the default operating point.

Number of directional refinement passes. We finally vary $L _ { s } \mathrm { a t } \left( r _ { f } , r _ { \mathrm { K V } } \right) = \left( 0 . 5 0 , 0 . 2 0 \right)$ . Here, $L _ { s } = 0$ uses the independent per-anchor Top- $\mathbf { \nabla } . K$ initialization, $L _ { s } = 1$ applies one forward directional refinement pass, $\bar { L _ { s } } ~ = ~ 2$ applies one forward followed by one backward directional refinement pass, and $\bar { L _ { s } } = 4$ repeats this pair twice. Each pass visits all anchors once, alternating forward and backward directions across the temporally ordered anchor sequence.

Table 10: Directional refinement sensitivity on Matrix-Game 3.0 at $r _ { f } = 0 . 5 0$ and $r _ { \mathrm { K V } } = 0 . 2 0$ Global and Revisit are WorldMark memory scores; interval NRMSE is measured against fully computed residuals, and latency is milliseconds per chunk.
<table><tr><td></td><td></td><td></td><td colspan="2">WorldMark↑</td><td></td><td></td><td></td></tr><tr><td> $L _ { s }$ </td><td>VBench↑</td><td>LPIPS↓</td><td>Global</td><td>Revisit</td><td>Interval NRMSE↓</td><td>Latency↓</td><td>Speedup↑</td></tr><tr><td>0</td><td>0.7479</td><td>0.4776</td><td>44.387</td><td>79.206</td><td>0.1127</td><td>1,976</td><td>2.19×</td></tr><tr><td>1</td><td>0.7549</td><td>0.4689</td><td>45.916</td><td>80.721</td><td>0.1058</td><td>1,991</td><td>2.17×</td></tr><tr><td>2</td><td>0.7570</td><td>0.4657</td><td>46.645</td><td>81.514</td><td>0.1013</td><td>2,007</td><td>2.15×</td></tr><tr><td>4</td><td>0.7574</td><td>0.4649</td><td>46.781</td><td>81.672</td><td>0.1008</td><td>2,038</td><td>2.12×</td></tr></table>

The first two directional refinement passes recover most of the quality and memory lost by independent routing. Moving from $L _ { s } = \bar { 0 } \mathrm { t o } \mathbf { \bar { \Gamma } } L _ { s } = 2$ raises VBench by 0.0091, improves global and revisit memory by 2.258 and 2.308 points, and reduces interval NRMSE by 10.1%, for 31 ms of additional latency. Increasing $L _ { s }$ from 2 to 4 yields only 0.0004 VBench, 0.136 global-memory points, 0.158 revisit-memory points, and a 0.5% relative interval-error improvement while reducing speedup to 2.12×. This saturation supports the fixed two-pass schedule used in the main experiments.

## C.3.4 ADDITIONAL QUALITATIVE COMPARISONS

Figures 10–14 extend Figure 9 with five additional samples per backbone. Each panel compares Native, SVG, TeaCache, Light Interaction (LI), and CAST at matched frame indices under the same control sequence. These selected rollouts illustrate scene evolution and action reversals; aggregate comparisons are reported in the main experiments.

![](images/35d71f5bb3d31def1825b17049f258993fba2814ad49daefc5e975d6386bc1f0.jpg)  
Figure 10: WorldMark rollouts: Matrix-Game 3.0, fr012\_007 (L/R, left), and HY-World 1.5, ts018\_006 (W/S, right). All methods use matched frame indices. Columns show the initial view, the midpoint and end of the first action, and the end of the second action, including its reversal of the initial command.

![](images/9149e9e72e75a05f59996d74370fde76de2465c8103cd6731f1a7d534e3e2e2b.jpg)  
Figure 11: WorldMark rollouts: Matrix-Game 3.0, fr019\_007 (L/R, left), and HY-World 1.5, tr006\_006 (W/S, right). Methods and frame selection follow Figure 10.

![](images/5f58fffb38b6a035528bae25186fbb3e3dfbf2227b2676773a100d7d3226de64.jpg)  
Figure 12: WorldMark rollouts: Matrix-Game 3.0, fr007\_006 (W/S, left), and HY-World 1.5, fr018\_007 (L/R, right). Methods and frame selection follow Figure 10.

![](images/afdf2a616804c4d979ae1432af1fca47fa932c6e7c2c6cd54c57cbc6a506f81c.jpg)  
Figure 13: WorldMark rollouts: Matrix-Game 3.0, ts020\_007 (L/R, left), and HY-World 1.5, ts024\_007 (L/R, right). Methods and frame selection follow Figure 10.

![](images/d89cdb20de4bc4eea42790d58978393290fe2eeb799866e8c6a461cbedb48c52.jpg)  
Figure 14: WorldMark rollouts: Matrix-Game 3.0, fr003\_010 (S/L, left), and HY-World 1.5, fr012\_007 (L/R, right). Methods and frame selection follow Figure 10.