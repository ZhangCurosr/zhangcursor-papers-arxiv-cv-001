# PROXIMAL-ONLY TRANSMISSION MATRIX RECOVERY OF AN ARBITRARILY DEFORMED GRADED-INDEX MULTIMODE FIBER

PREPRINT

Cole Reynolds Weyl Labs Cole@WeylLabs.com

## ABSTRACT

The multimode fiber is among the thinnest imaging conduits available, carrying hundreds to thousands of spatial modes through a cross-section comparable to a human hair, but its endoscopic capabilities are currently limited by the sensitivity of the transmission matrix to the fiber’s deformed state. Proximal-only recovery of the fiber’s transmission matrix is an appealing approach for enabling general use multimode fiber endoscopy, and within the last decade, machine learning techniques have been applied to both single-ended and double-ended transmission matrix recovery tasks. We present a new approach to this interdisciplinary problem and show that neural networks can generalize to recover transmission matrices of an arbitrarily deformed graded-index multimode fiber from proximal measurements alone.

## 1 Introduction

A Multimode fiber (MMF) propagates information between its proximal and distal ends through the reciprocal relationship

$$
\psi _ { \mathrm { d i s t } } = U \psi _ { \mathrm { p r o x } } , \qquad \psi _ { \mathrm { p r o x } } = U ^ { T } \psi _ { \mathrm { d i s t } }\tag{1}
$$

where $U$ is the fiber’s transmission matrix (TM). In MMF endoscopy, the fiber’s distal tip is guided to a scene described by an unknown scattering matrix S, and probe signals from the proximal end propagate through the fiber, interact with the scene, and a portion of that initial signal energy propagates back through the fiber to the proximal end which can then be measured:

$$
\psi _ { \mathrm { m e a s u r e d } } = U ^ { T } S U \psi _ { \mathrm { p r o b e } } .\tag{2}
$$

For a known N × N TM, the available scene content $S$ can be recovered, in principle, with N linearly independent probe signals. This is the basis of MMF endoscopy, and it is demonstrably capable of resolving subcellular structure [1–4].

Despite these successes, MMF endoscopy applications are limited by knowledge of the underlying fiber’s TM because the TM changes with the fiber’s geometry: bends and twists during endoscopy change the TM, and since the distal tip at the scene cannot be accessed, the TM cannot be directly measured or calibrated using Eq. (1). Every other requirement of MMF endoscopy has been achieved experimentally: probe delivery, scene interrogation, and image formulation all follow from Eq. (1) once U is known, thus recovery of the deformed fiber’s TM is the bottleneck to general use. Proximal-only recovery methods [5] try to obtain the TM through the use of reflectors located at the distal end of the fiber. Reflection matrices within a fiber, unlike a varying and unknown scene scattering matrix outside a fiber, are constant across measurements: for K reflectors with reflection matrices $\{ R _ { k } \}$ , one probe signal provides the K measurements

$$
\psi _ { \mathrm { m e a s u r e d } } ^ { ( k ) } = U ^ { T } \tilde { R } _ { k } U \psi _ { \mathrm { p r o b e } } = M _ { k } \psi _ { \mathrm { p r o b e } } ,\tag{3}
$$

Where ${ \tilde { R } } _ { i }$ denotes the interaction with $R _ { i }$ as well as any prior or subsequent reflection, transmission, or propagation matrices.

This approach is appealing because it re-frames the historically physics problem as a machine learning problem: given a set of measurements $\{ \bar { M } _ { k } \}$ , does there exist a mapping that accurately encodes the state of an arbitrarily deformed fiber, and can that encoded state be decoded to recover the TM? Deep learning’s canonical successes never needed to ask this question because it already had empirical evidence of the answer: a finite nervous system performs vision and language, so those domains provably admit finite representations, and the architectures that succeeded there were built to exploit relevant abstract structure such as locality, invariance, and equivariance. A physics problem has no such guarantee. Whether a fiber’s transmission matrix admits a finite parameterization under deformation is a property of the fiber, and prior works provide little evidence that the inductive biases of image-domain architectures are aligned with the fiber’s dynamics. A learned model must recover the TM of a fiber state it has never encountered, and thus the model must capture and encode what the fiber does under deformation rather than what a sample of deformations looked like. In practice, the encoding must be equivariant in the informal sense that a physical action on the fiber, a bend or a twist, acts on the encoded state as a corresponding structural operation, and it is this property that allows the model to recover states it has not seen. In this work we present a new approach that suggests a finite set of graded-index (GRIN) training fibers is enough to learn that structure, and we demonstrate capabilities on a synthetic commercial grade fiber under realistic measuring conditions.

## 2 Related Work

Currently, the most successful approach to MMF endoscopic imaging is to make the fixed-TM assumption hold: the TM is characterized before deployment, with access to both facets, and the fiber’s geometry is then locked for the duration of imaging. The Cižmár group has carried this paradigm the furthest. They first demonstrated lensless imaging through<sup>ˇ</sup> a standard MMF [6] and then later demonstrated high-fidelity fluorescence endoscopy deep in the living brain [1].

The general TM deformation problem has also been addressed directly from the fiber side. Flaes et al. [7] showed that light transport through parabolic-profile GRIN fibers is intrinsically more robust to bending than through step-index fibers, and predicted that sufficiently precise parabolic-index fibers would make flexible probes feasible. Subsequent work pursued bending resilience through a hybrid multimode-multicore design [8], and more recently through tomographic measurement of the refractive-index perturbations of commercial GRIN fibers, showing that the measured profiles account for the observed decline of imaging under bending, identifying the fibers that endure it best, and demonstrating a flexible endoscope built on one of them [9]. This approach aims to solve the TM deformation problem by making the TM sufficiently deformation-invariant. This bend-resilience narrows the gap between the calibrated TM and the true underlying TM, but the gap still grows with the severity of the deformation and cannot be accounted for without some form of active sensing. Nevertheless, tolerance aids recovery and we argue in Section 3 that the mechanism behind the bend-resilience of graded-index fibers, low-dimensional parameterization, is what enables general TM recovery.

## 3 Theory

We first show that the TM for an arbitrarily deformed ideal GRIN fiber admits a finite representation while the TM for a step-index fiber does not.

## 3.1 Graded-Index Fiber

In the weakly-guided approximation [10], the Hamiltonian for a bent and twisted ideal GRIN fiber is the linear combination of operators (see appendix A)

$$
{ \cal H } ( z ) = c _ { 0 } p ^ { 2 } + c _ { 1 } r ^ { 2 } + c _ { 2 } \kappa ( z ) \cdot r + c _ { 3 } \tau ( z ) { \cal L } _ { z } ,\tag{4}
$$

This set of deformation operators is not closed under commutation since $[ p ^ { 2 } , r ^ { 2 } ] = - 2 i ( { \pmb r } \cdot { \pmb p } + { \pmb p } \cdot { \pmb r } ) = - 2 i D$ , but the set with D included does close. The non-trivial commutation relations are

![](images/29136892729c6e48e183f19b9129f5af75da1438983bbe79ae3b0a8fdff86e6b.jpg)  
Figure 1: Architecture results for 3 held-out GRIN fibers in different recovery regimes. Only every 4th column and row are shown for visual clarity.

$$
\begin{array} { r l r l r l r l } { { } [ x _ { j } , p _ { l } ] = i \delta _ { j l } { \cal I } , } & { } & { [ x _ { j } , p ^ { 2 } ] = 2 i p _ { j } , } & { } & { [ x _ { j } , D ] = 2 i x _ { j } , } & { } & { [ x _ { j } , L _ { z } ] = - i \varepsilon _ { j l } x _ { l } , } \\ { { } [ p _ { j } , r ^ { 2 } ] = - 2 i x _ { j } , } & { } & { [ p _ { j } , D ] = - 2 i p _ { j } , } & { } & { [ p _ { j } , L _ { z } ] = - i \varepsilon _ { j l } p _ { l } , } & { } & { } & { } \\ { { } [ p ^ { 2 } , r ^ { 2 } ] = - 2 i D , } & { } & { [ p ^ { 2 } , D ] = - 4 i p ^ { 2 } , } & { } & { } & { } & { } \\ { { } [ r ^ { 2 } , D ] = 4 i r ^ { 2 } , } & { } & { } & { } & { } & { } \end{array}\tag{5}
$$

where ε is the two-dimensional Levi-Civita symbol and the remaining relations $[ x _ { j } , r ^ { 2 } ] , [ p _ { j } , p ^ { 2 } ] , [ r ^ { 2 } , L _ { z } ] , [ p ^ { 2 } , L _ { z } ]$ , and $[ D , L _ { z } ]$ all vanish. Thus, $\mathfrak { g } = \mathrm { s p a n } \{ I , x , y , p _ { x } , p _ { y } , L _ { z } , p ^ { 2 } , r ^ { 2 } , D \}$ is a nine-dimensional Lie algebra that contains $H ( z )$ for all deformations.

Having the general Hamiltonian contained within a finite-dimensional Lie algebra is what makes the proximal-only TM recovery problem tractable. The TM for any deformed fiber takes the path-ordered form

$$
U = { \mathcal { P } } \exp \left( - i \int _ { 0 } ^ { L } H ( z ) d z \right) .\tag{6}
$$

Dividing the fiber into segments of length $\Delta z$ over which H is constant, Eq. (6) is the limit of the ordered product

$$
U = \operatorname * { l i m } _ { \Delta z \to 0 } e ^ { - i H ( z _ { M } ) \Delta z } \cdot \cdot \cdot e ^ { - i H ( z _ { 2 } ) \Delta z } e ^ { - i H ( z _ { 1 } ) \Delta z } ,\tag{7}
$$

and neighboring segments combine through the Baker–Campbell–Hausdorff (BCH) formula,

$$
\begin{array} { r } { e ^ { A } e ^ { B } \ = \ \exp \Bigl ( A + B + \frac 1 2 [ A , B ] + \frac 1 { 1 2 } \bigl ( [ A , [ A , B ] ] + [ B , [ B , A ] ] \bigr ) + \cdots \Bigr ) . } \end{array}\tag{8}
$$

But since g is closed, the expansion in Eq. (8) for the GRIN fiber can always be expressed as a linear combination of elements in the Lie algebra,

$$
e ^ { A } e ^ { B } = \exp \left( \sum _ { i } \alpha _ { i } g _ { i } \right) , \qquad g _ { i } \in { \mathfrak { g } } ,\tag{9}
$$

and the same holds for any pair of exponentials built from g. An ordering change therefore adds no new terms and only changes the coefficients: applying Eq. (9) to both $e ^ { - A } e ^ { - B }$ and $e ^ { A } e ^ { B }$ , and then to the product of those two results gives

$$
e ^ { A } e ^ { B } = e ^ { B } e ^ { A } X ,
$$

$$
X = e ^ { - A } e ^ { - B } e ^ { A } e ^ { B } \ = \ \mathrm { e x p } \left( \sum _ { i } \beta _ { i } g _ { i } \right) \mathrm { e x p } \left( \sum _ { i } \gamma _ { i } g _ { i } \right) \ = \ \mathrm { e x p } \left( \sum _ { i } \delta _ { i } g _ { i } \right) ,\tag{10}
$$

where $\beta , \gamma$ , and δ are each fixed by A and B through Eq. (8). Exchanging two segments therefore costs one further element of the same group rather than an expansion without end, and applying Eq. (9) across every segment of Eq. (7) collapses the ordered product to a single exponential,

$$
U = f ( \pmb { a } ) = \exp \left( \sum _ { i } a _ { i } g _ { i } \right) , \quad g _ { i } \in \mathfrak { g } ,\tag{11}
$$

where a is fixed by the deformation profile [11, 12]. The interpretation is that while the physical geometry of any fiber is path-ordered, the TM dynamics of the GRIN fiber are not, and recovery from finite measurements is possible because those dynamics have a finite parameterization.

## 3.2 Step-Index Fiber

The step-index fiber replaces the quadratic well with the canonical finite-well profile, but we assume a general potential of the form $\begin{array} { r } { \dot { V } ( \pmb { r } ) = et { } { ' } \sum _ { m } c _ { m } r ^ { m } } \end{array}$ . The GRIN fiber is the special case $c _ { m \neq 2 } = 0 .$ . Using the identity $[ A , B C ] = [ A , B ] C + B \left[ A , C \right]$ , every $r ^ { m }$ for m $\geq 3$ in the Hamiltonian can be generated by r and $r ^ { 2 }$ , but only $m = 2$ creates a finite Lie algebra:

$$
\left[ p ^ { 2 } , r \right] = \frac { 1 } { r } \left( - 2 i \pmb { r } \cdot \pmb { p } - 1 \right) ,\tag{12}
$$

$$
[ p ^ { 2 } , r ^ { 4 } ] = r ^ { 2 } [ p ^ { 2 } , r ^ { 2 } ] + [ p ^ { 2 } , r ^ { 2 } ] r ^ { 2 }\tag{13}
$$

$$
= - 2 i \left( r ^ { 2 } D + D r ^ { 2 } \right) .\tag{14}
$$

The non-quadratic Hamiltonian is

$$
H ( z ) = c _ { 0 } p ^ { 2 } + c _ { 1 } \kappa ( z ) \cdot r + c _ { 2 } L _ { z } + \sum _ { m = 0 } ^ { \infty } c _ { m + 3 } r ^ { m }\tag{15}
$$

and performing the same analysis (Eqs. (6), (7), and (8)) results in a TM with no finite parameterization. The BCH formula does not terminate and the path-ordering does not become tractable. The number of operators needed to describe the fiber grows with the number of bends rather than remaining fixed, and in the continuum limit, where the curvature varies smoothly along the fiber and the number of segments is unbounded, no truncation of the expansion is uniformly valid. Recovering the TM of a step-index fiber is consequently equivalent to recovering $\kappa ( z )$ at every point along its length, which no finite set of proximal measurements can determine.

## 3.3 Real Fibers

For the parabolic profile, the z-dependent deformation functions $\kappa ( z )$ and $\tau ( z )$ never appear in the TM individually: closure consolidates each generator’s accumulated contribution into a single scalar, so the TM of Eq. (11) depends on the deformation history only through history-dependent, weighted integrals of the deformation profile, and its dynamics are exhausted by how those nine scalars evolve. A non-parabolic profile cannot be similarly compressed because each application of Eq. (8) in an effort to consolidate terms comes at the cost of an infinite expansion, and recovering the TM becomes tantamount to recovering the deformation functions themselves point by point along the fiber.

In practice, commercial GRIN fibers are not perfectly parabolic, and contain many defects that break this ideal structure. Fortunately, the defects in the spatial refractive profile for GRIN fibers are perturbative, and parameter consolidation introduces higher-order BCH terms that decay rapidly, and general recovery from finite measurements only needs the dynamics to be well approximated by some finite representation.

Demonstrating general recovery requires data sampled from fibers whose curvature admits no convenient parameterization and this data would be expensive to obtain experimentally. Consequently, we establish the architecture’s capabilities on synthesized fibers, where deformation states are sampled from a continuum. Details about the Hamiltonian used to describe the fiber’s dynamics can be found in Appendix A, and details about how the fiber is deformed and how the TM and reflection measurements are obtained can be found in Appendix B.

## 4 Methods

## 4.1 Proximal-Only Recovery System for MMF Endoscopy

In a physically realizable MMF endoscopic system, an input signal propagates down the fiber, interacts with the reflectors and the scene, and their respective return signals are measured back at the proximal end. The $N \times N$ reflection and scene matrices can be obtained from N linearly independent input signals using an amplitude + phase measuring technique such as off-axis holography.

![](images/d83caf0a33241aaecf0d2b5cbe4d8114f2f268536dd5687770b38f11fa85027a.jpg)  
Figure 2: Visual representation of our proximal-end recovery system. The input signal (blue) propagates down the fiber, reflects off of reflectors $R _ { 1 } , R _ { 2 } ^ { \bar { } } , R _ { 3 }$ , and the scene $S ,$ and then propagates back. The TM is determined from the measurement signals (red) and then uses the scene signal (green) to construct the final image.

All matrices and signals are represented in the Laguerre-Gauss spatial mode basis — the propagation-invariant modes (PIM) for the ideal GRIN fiber. These PIMs carry a complex $\bar { e } ^ { i \ell \varphi }$ factor, where propagation back to the proximal end after reflection is the transpose composed with the $\ell  - \ell$ operator $P$ for circularly polarized light. Writing $\mathcal { P } [ X ] \equiv P X P$ , the round trip reflection measurements are $M _ { i } = \mathcal { P } \bar { [ } U ^ { T } ] R _ { i } U$

Since these measurements are bilinear, at least three reflectors are needed to break all symmetries and uniquely determine U [5, 13]. In this work we use a straight partial reflector $R _ { 1 }$ , a tilted partial reflector $R _ { 2 } ,$ , and a second tilted partial reflector $R _ { 3 }$ with its tilt axis making a $. \pi / 4$ angle with $R _ { 2 } { \ ' } s$ tilt axis. We note that neither $R _ { i }$ nor ${ \tilde { R } } _ { i }$ need to be explicitly known or fabricated with extreme precision in practice. The architecture only requires that all bilinear-induced symmetries are sufficiently broken by the set of reflectors.

The architecture solves for a fiber’s bent $\mathrm { T M } , U _ { 2 } .$ , given the fiber’s unbent TM, $U _ { 1 }$ , and the three reflector measurements $\{ M _ { 1 } , M _ { 2 } , M _ { 3 } \}$ . The unknown $U _ { 2 }$ is obtained indirectly by solving for $V = U _ { 2 } U _ { 1 } ^ { \dagger }$ instead, where $U _ { 2 } = V U _ { 1 }$ and $V$ is the operator that describes the mode couplings due to the deformation. Using ${ \mathcal { P } } [ X Y ] = { \mathcal { P } } [ X ] { \mathcal { P } } [ Y ]$

$$
M _ { i } = \mathcal { P } \big [ ( V U _ { 1 } ) ^ { T } \big ] R _ { i } ( V U _ { 1 } ) = \mathcal { P } [ U _ { 1 } ^ { T } ] \mathcal { P } [ V ^ { T } ] R _ { i } V U _ { 1 } = \mathcal { P } [ U _ { 1 } ^ { T } ] A _ { i } U _ { 1 }\tag{16}
$$

$$
A _ { i } = \mathcal { P } [ V ^ { T } ] R _ { i } V = \mathcal { P } [ U _ { 1 } ^ { * } ] M _ { i } U _ { 1 } ^ { \dagger } ,\tag{17}
$$

where $A _ { i }$ is a known since both $M _ { i }$ and $U _ { 1 }$ are known.

## 4.2 Architecture

We use a bilinear architecture [14, 15] with one encoder per measurement $A _ { i } ,$ , and replace the correspondence decoder in [15] with a MLP that outputs V:

$$
h _ { j k } ^ { i } = \big \langle \phi _ { j } ^ { i } \big | A _ { i } \big | \psi _ { k } ^ { i } \big \rangle ,\tag{18}
$$

$$
g _ { n } ^ { i } = \sigma ( W _ { j k n } h _ { j k } ^ { i } + c _ { n } ) ,\tag{19}
$$

$$
\mathsf { c a t } [ \mathbf { R e } ( V ) , \mathbf { I m } ( V ) ] = \mathbf { M L P } ( \mathsf { c a t } [ \mathbf { g } ^ { 1 } , \mathbf { g } ^ { 2 } , \mathbf { g } ^ { 3 } ] ) ,\tag{20}
$$

where $\sigma$ is a non-linear activation function (we use tanh in this paper) and the MLP contains two hidden layers with the leaky ReLU activation function $( \mathbf { g }  \mathbf { z } _ { 1 }  \mathbf { z } _ { 2 } )$ . Outputting the entire V from a single MLP is impractical for large number of modes, and the presented architectures instead partition $V$ into subsets and then use a separate, smaller MLP for each subset, $V _ { i } = \mathbf { M } \mathbf { L } \mathbf { \dot { P } } _ { i } ( \mathbf { z } _ { 2 } )$ . Complex h values are handled in σ by $\mathbf { h }  \mathrm { c a t } [ | \mathbf { h } | , \mathrm { R e } ( \mathbf { h } )$ , Im(h)]. No learned or prescribed residual gating mechanism is used in the presented architectures for simplicity, but the iterative refinement procedure still applies: an initial output $\hat { V } _ { 1 }$ can be applied to $U _ { 1 }$ within $A _ { i }$ to "deform" $U _ { 1 }$ toward $U _ { 2 }$ . Inference is then performed using the updated $A _ { i }$ matrices, and the estimate composes as $V = \hat { V } _ { 2 } \hat { V } _ { 1 } , \hat { U } _ { 2 } = V U _ { 1 }$ . Two inference passes are applied during training and testing.

In practice, illumination and detection involve distinct optical systems and each imposes its own transformation on the measurement, so what the architecture sees is $B A _ { i } C$ rather than $A _ { i }$ , where the transformations B and $C$ are fixed properties of the system rather than of the fiber [16]. These transformations break the reciprocity relation of Eq. (1) but they do not obstruct recovery since each feature $h _ { j k } ^ { i }$ is a bilinear form in the measurements and the filters are mutable: for invertible $B , C$ and synthetically trained filters ψ, ϕ, the experimentally trained (or fine-tuned) filters would become $( B ^ { - 1 } ) ^ { \dag } \phi$ and $C ^ { - 1 } \psi$ , leaving $h _ { j k } ^ { i }$ intact (experimental filters for non-invertible transformations would instead move towards the available orthogonal projection, $h _ { j k } ^ { i }$ would change, and downstream weights would likewise need to be fine-tuned).

More generally speaking, the encoder does not commit to the basis in which the measurement is expressed: $A _ { i }$ is an operator and its rank is a basis-free property, thus the set of features the encoder can compute is unchanged under any invertible change of basis on either side of the measurement. This distinguishes it from the locality and translation structure that image-domain architectures impose on a pixel basis that has no counterpart in a measurement whose content is mode coupling.

## 5 Results

We trained our architecture under the protocol described in Table 2 on a synthesized DRAKA WideCap OM5 50 µm core fiber at $\lambda = 5 3 2$ nm with both circular-polarization states $( \sigma _ { + } , \sigma _ { - } )$ and $N = 3 0 0$ modes per polarization (600 total modes). We trained three separate models with varying levels of zero-mean gaussian noise, $\sigma = \{ 0 . 0 , 0 . 0 4 , 0 . 1 6 \}$ added to $\{ M _ { i } \}$ . We also trained a model on a synthesized step-index fiber as a reference with $N = 3 0 3$ using the Linearly Polarized basis and zero-mean $\sigma = 0 . 0 2$ gaussian noise added to its {M }. The evaluation metric is fiber fidelity $F = | \mathrm { t r } ( V _ { \mathrm { p r e d } } ^ { \dagger } V _ { \mathrm { g t } } ) | / ( \| V _ { \mathrm { p r e d } } \| _ { F } \| V _ { \mathrm { g t } } \| _ { F } )$ where $V _ { \mathrm { p r e d } }$ is the model output and $V _ { \mathrm { g t } }$ is the ground truth matrix.

## 5.1 Learned Curvature Encoding

To examine how the architecture organizes measurement information from these arbitrarily deformed GRIN fibers, we constructed 820 fibers each wound into a single circle of constant curvature, with radii of curvature between 3 and 60 cm. While these fibers are unphysical (many of them wind into themselves), they isolate a single scalar deformation parameter and allow it to be swept continuously. Furthermore, they appear in neither the training nor the testing set, and radii of curvature larger than 40 cm are never seen by the trained models. Figure 4 shows the main MLP’s second hidden layer activations $\mathbf { z } _ { 2 }$ for each fiber projected onto the first three principal components. The architecture only sees the three measurements $\{ M _ { i } \}$ , and plotting these projections as a function of curvature shows the parameter traversing a three-dimensional manifold within the model’s latent space. The first two components correspond to a local phase and the third component corresponds to a coarse global structure. The encoded curvature loses its structure around $R = 7 { \mathrm { c m } }$ , in agreement with the testing dataset results, and provides one possible explanation why some stray fibers with large R in Figure 3 were poorly recovered: their latent vector sits near the region with broken structure and was incorrectly decoded. The step-index model, despite the exact same encoder, could not encode the curvature as well, which is in agreement with [7].

![](images/901f94b371fb7868b3147585319d4f6d3133176663aba8de803499bcea587973.jpg)

![](images/5eb000d19c4a56751deef8c03f899cea9bf7ce3a10e4da0e7b09cf8e6606f183.jpg)  
Figure 3: Recovery fidelity on the thousand held-out fibers for the three GRIN models trained at detector noise $\sigma = 0 , 0 . 0 4$ and 0.16, and the step-index model. each evaluated at its own σ. Top: Scatter plot of F as a function of the minimum bend radius, with a rolling mean over 80 fibers. Individual scatter points are shown for the $\sigma = 0 . 1 6$ model. Bottom: Histogram of the same fidelities.

Table 1: Fidelity recovery for the $\sigma ~ = ~ 0 . 1 6$ GRIN model on held-out testing data with mean initial fidelity ${ \boldsymbol { F } } =$ $| \mathrm { t r } ( U _ { 2 } U _ { 1 } ^ { \dagger } ) | / ( \sqrt { N } \lVert U _ { 2 } U _ { 1 } ^ { \dagger } \rVert _ { F } ) = 0 . 1 8 1 6$
<table><tr><td>Subset</td><td>n</td><td>Mean  $F$ </td><td> $p _ { 5 }$ </td><td>Worst</td></tr><tr><td>All held-out fibers</td><td>1000</td><td>0.897</td><td>0.551</td><td>0.045</td></tr><tr><td>Min bend radius  $\geq 5 \mathrm { c m }$ </td><td>808</td><td>0.953</td><td>0.869</td><td>0.603</td></tr><tr><td>Min bend radius  $\geq 7 \mathrm { c m }$ </td><td>692</td><td>0.964</td><td>0.919</td><td>0.671</td></tr><tr><td>Min bend radius  $\geq 1 0 \mathrm { c m }$ </td><td>566</td><td>0.970</td><td>0.934</td><td>0.671</td></tr></table>

Table 2: Training protocol
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Training deformations</td><td>19,000</td></tr><tr><td>Testing deformations</td><td>1000</td></tr><tr><td> $\mathrm { L o s s ^ { 1 } }$ </td><td>min  $\lVert Q ( \theta ) V _ { \mathrm { p r e d } } - V \rVert _ { F } ^ { 2 } / \lVert V \rVert _ { F } ^ { 2 }$  θ</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Schedule</td><td>cosine annealing to 0</td></tr><tr><td>Epochs</td><td>40</td></tr><tr><td>Batch size</td><td>24</td></tr><tr><td>Inference passes</td><td>2</td></tr></table>

<sup>1</sup> In the circular polarization basis $Q ( \theta ) = { \mathrm { D i a g } } ( e ^ { i \theta } I _ { N } , e ^ { - i \theta } I _ { N } ) ,$  
where θ is determined analytically.

We then looked to see if curvature is encoded by the model in higher order harmonics within $\mathbf { z } _ { 2 }$ with a dense sampling of the same constant curvature fibers from $R \stackrel { \cdot } { = } 4 0$ cm to $R = 2 0$ cm. Figure 5 shows that curvature is indeed very clearly encoded in higher order harmonics, and extents until the 12-14th harmonic for this range of R.

## 5.2 Learned Relative Positional Encoding

Another encoded property is the location of a particular deformation along the fiber. For a fiber with a localized defect that is partitioned into $n _ { \mathrm { t o t a l } }$ sections, the TM can be represented by

$$
U ( n ) = \left( e ^ { - i H _ { 0 } \Delta z } \right) ^ { m - n } U _ { \mathrm { d e f e c t } } \left( e ^ { - i H _ { 0 } \Delta z } \right) ^ { n } ,\tag{21}
$$

where there are m straight fiber segments, the defect is described by the remaining $n _ { \mathrm { t o t a l } } - m$ segments, n indicates where the defect exists along the fiber, and $H _ { 0 }$ is the unbent Hamiltonian. Changing n corresponds to changing X in Eq. (10), and to see if the model learned to encode this relational structure, we constructed 10,800 fibers in which the same localized spiral deformation (complex Gabor wavelet) is translated along the middle quarter of the fiber (Figure 6a). Moving the defect conjugates the TM by straight propagation, $U ( n ) = e ^ { - i H _ { 0 } n \Delta z } U ( 0 ) \sp { \sp { . } } e ^ { i H _ { 0 } n \Delta z }$ , so each element only acquires a phase $e ^ { - i ( \bar { \beta } _ { j } - \beta _ { k } ) n \Delta z }$ , where $\beta _ { j }$ are the propagation constants, and position enters the TM in the same way it enters a rotary positional encoding.

Figure 6b shows that the architecture learned this structure. The latent $\mathbf { z } _ { 2 }$ oscillates with a period within 0.4% of $2 \pi / \Omega$ and over a few millimeters its two largest principal components trace a circle, so the defect position is encoded as an angle while a slower component across the full sweep carries the coarse position along the fiber. Neither the defect position nor the self-imaging period is supplied to the architecture at any point, and like the curvature ablation, this relational structure is something it arrives at from the measurements alone.

## 6 Discussion and Limitations

The parameters of the simulated fiber and proximal-only system were chosen to go beyond proof-of-concept: the refractive index perturbations are the measured profiles of a Prysmian DRAKA WideCap OM5 50 µm core fiber and the Hamiltonian of Eq. (35) contains the spin-orbit coupling together with all three birefringence contributions. The deformation regime spans minimum bend radii from 3 to 40 cm with out-of-plane displacement, non-adiabatic curvature changes, and twists. The straight calibration $U _ { 1 }$ retains a fidelity of 0.12-0.22 (mean 0.18) against the deformed $U _ { 2 }$ indicating that these are deformations under which a fixed calibration is unusable. The measurements are collected in a realistic, non-ideal manner where the fiber is perturbed independently during each probe signal (Eq. (36)) so that no $M _ { i }$ corresponds to a single fiber state, and with system noise added on top of that (Figure 8). Each of these choices moves the setting toward a working instrument and away from the regime in which recovery would be easiest.

Furthermore, our construction poses the hardest version of the recovery problem. Each deformation operator $V$ is recovered in a single step between the straight calibration state $U _ { 1 }$ and a severely deformed $U _ { 2 } ,$ , and nothing in the formulation requires this, since $V = U _ { 2 } U _ { 1 } ^ { \dagger }$ holds for any previously known state in place of $U _ { 1 }$ . A deployed endoscope deforms continuously, so the most recently recovered TM could and should serve as the reference where each update only needs to recover the incremental deformation accumulated over one time step, which is a far better conditioned problem than the full excursion from straight. From this perspective, we consider the fidelities in Table 1 to be a lower bound on what the same architecture would achieve operating as a closed tracking loop with appropriate algorithms.

![](images/058679873fd9a95af309235a37d94f91f81aa2a086a1a16fd04f9c94642abf3f.jpg)

![](images/286ea05e80220142c00656dd567c24839ce54e4824253d30d6e6d87155257269.jpg)  
Figure 4: The three largest principal components of z for the initial 620 wound fibers plus an additional 200 fibers in the $R \in [ 4 0 , 6 0 ]$ cm range for the $\sigma = 0 . 1 6$ GRIN fiber (top) and step-index fiber (bottom), colored by radius of curvature. Gray lines indicate trajectory path. The architectures never saw bends with $R \geq 4 0 \mathrm { c m } .$ , so that curvature regime is outside of the training distribution but still traverses the manifold of both architectures as if it was in the distribution and demonstrates generalized capabilities. However, curvature information is much better compressed in the GRIN encoder, maintaining low-dimensional structure until about $R = 7 { \mathrm { c m } }$ whereas low-dimensional representation in the step-index encoder breaks down near $R = 2 0 \mathrm { c m }$

![](images/0b852dc71845119ee9d827120b8ffaf901cb93b8f61cd504b42d3354f92d0724.jpg)  
Figure 5: Curvature of the σ = 0.16 GRIN model encoded in z<sub>2</sub> as higher-ordered harmonics.

While the fibers are synthetic, the path from these results to an experimentally verified device does not require discarding the synthetic data. Every term in Eq. (35) corresponds to an identified physical effect or fiber defect rather than a fitted degree of freedom: the parabolic potential and its measured perturbation, the bend-induced tilt, the twist rotation, the spin-orbit coupling, and the three birefringence contributions each have a separate physical origin, so a discrepancy between simulated and experimental measurements can be attributed to particular terms and corrected rather than absorbed into a global mismatch. The same holds for the measurement chain, where the sources we do not currently model are characterizable on a bench, and once characterized they enter the simulator in the same way that the acquisition jitter and system noise already do. Figure 3 shows that recovery is robust to measurement noise, with the $\sigma = 0 . 1 6$ measurements containing significant magnitude and phase errors, indicating that the architecture is not relying on artifacts particular to noise-free data. We therefore expect a practical route to an instrument to be pretraining on synthetic data made as faithful as the characterization allows, followed by fine-tuning on a comparatively small experimental set.

## 7 Conclusion

In this work we introduce a bilinear neural network architecture for proximal-only TM recovery that achieves an average fidelity recovery of $F = 0 . 8 9 7$ on arbitrarily deformed GRIN MMFs with 600 guided modes $( F \ge 0 . 9 5 $ when restricted to bending curvatures larger than 5 cm). We also perform several ablations demonstrating the learned model’s ability to identify structure and generalize to fibers outside of the training distribution. The next step is to reproduce these capabilities on experimentally acquired measurements, and we believe the results here establish that the effort is warranted.

![](images/18eacf20625694fca7ac7821e6f83bcaff58af2b30de6d507146ac9a3da73bf3.jpg)  
(a) The first $( z _ { 0 } = 3 7 5$ mm) and last $( z _ { 0 } = 6 2 5$ mm) of the 10,800 fibers, where $z _ { \mathrm { 0 } }$ (indicated by the dot) is the center of the spiral deformation.

![](images/fadfefdae8d4d5170c1b72a3da3afd850b3a636feea1329bb790a4ee85b802b8.jpg)  
(b) The three largest principal components of $\mathbf { z } _ { 2 }$ for all 10,800 fibers, colored by $z _ { 0 } .$  
Figure 6: A localized spiral deformation translated along the middle quarter of the fiber. None of these fibers appear in the training or testing sets.

## References

[1] Sergey Turtaev, Ivo T. Leite, Tristan Altwegg-Boussac, Janelle M. P. Pakan, Nathalie L. Rochefort, and Tomáš Cižmár. High-fidelity multimode fibre-based endoscopy for deep brain in vivo imaging. <sup>ˇ</sup> Light: Science & Applications, 7:92, 2018. doi: 10.1038/s41377-018-0094-x.

[2] Sebastian A. Vasquez-Lopez, Raphaël Turcotte, Vadim Koren, Martin Ploschner, Zahid Padamsey, Martin J. Booth, Tomáš Cižmár, and Nigel J. Emptage. Subcellular spatial resolution achieved for deep-brain imaging <sup>ˇ</sup> in vivo using a minimally invasive multimode fiber. Light: Science & Applications, 7(1):110, 2018. doi:

10.1038/s41377-018-0111-0.

[3] Shay Ohayon, Antonio M. Caravaca-Aguirre, Rafael Piestun, and James J. DiCarlo. Minimally invasive multimode optical fiber microendoscope for deep brain fluorescence imaging. Biomedical Optics Express, 9(4):1492–1509, 2018. doi: 10.1364/BOE.9.001492.

[4] Zhong Wen, Zhenyu Dong, Qilin Deng, Chenlei Pang, Clemens F. Kaminski, Xiaorong Xu, Huihui Yan, Liqiang Wang, Songguo Liu, Jianbin Tang, Wei Chen, Xu Liu, and Qing Yang. Single multimode fibre for in vivo lightfield-encoded endoscopic imaging. Nature Photonics, 17(8):679–687, 2023. doi: 10.1038/s41566-023-01240-x.

[5] R. Y. Gu, Reza Nasiri Mahalati, and Joseph M. Kahn. Design of flexible multi-mode fiber endoscope. Optics Express, 23(21):26905–26918, 2015. doi: 10.1364/OE.23.026905.

[6] Tomáš Cižmár and Kishan Dholakia. Exploiting multimode waveguides for pure fibre-based imaging. <sup>ˇ</sup> Nature Communications, 3:1027, 2012. doi: 10.1038/ncomms2024.

[7] Dirk E. Boonzajer Flaes, Jan Stopka, Sergey Turtaev, Johannes F. de Boer, Tomáš Tyc, and Tomáš Cižmár.<sup>ˇ</sup> Robustness of light-transport processes to bending deformations in graded-index multimode waveguides. Physical Review Letters, 120:233901, 2018. doi: 10.1103/PhysRevLett.120.233901.

[8] Yang Du, Sergey Turtaev, Ivo T. Leite, Adrian Lorenz, Jens Kobelke, Katrin Wondraczek, and Tomáš Cižmár.<sup>ˇ</sup> Hybrid multimode-multicore fibre based holographic endoscope for deep-tissue neurophotonics. Light: Advanced Manufacturing, 3(3):408–416, 2022. doi: 10.37188/lam.2022.029.

[9] Sergey Turtaev, Tomáš Tyc, Ulf Poßner, Tina Eschrich, Torsten Poßner, Yang Du, André Gomes, Bernhard Messerschmidt, and Tomáš Cižmár. Deformation enduring conveyance of structured light through multimode <sup>ˇ</sup> waveguides and its exploitation for flexible hair-thin endoscopes, 2025.

[10] Allan W. Snyder and John D. Love. Optical Waveguide Theory. Chapman and Hall, London, 1983.

[11] Wilhelm Magnus. On the exponential solution of differential equations for a linear operator. Communications on Pure and Applied Mathematics, 7(4):649–673, 1954. doi: 10.1002/cpa.3160070404.

[12] James Wei and Edward Norman. Lie algebraic solution of linear differential equations. Journal ofMathematical Physics, 4(4):575–577, 1963. doi: 10.1063/1.1703993.

[13] Szu-Yu Lee, Vicente J. Parot, Brett E. Bouma, and Martin Villiger. Reciprocity-induced symmetry in the round-trip transmission through complex systems. APL Photonics, 5(10):106104, 2020. doi: 10.1063/5.0021285.

[14] Roland Memisevic and Geoffrey Hinton. Unsupervised learning of image transformations. In 2007 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1–8. IEEE, 2007. doi: 10.1109/CVPR. 2007.383036.

[15] Cole Reynolds. Neural phase correlation, 2026. URL https://arxiv.org/abs/2606.18496.

[16] Shreyas Bharadwaj, Gyeong Hun Kim, Brett E. Bouma, and Martin Villiger. Matrix symmetry and normality in multimode fiber transport. APL Photonics, 11(9):096101, 2026. doi: 10.1063/5.0337399.

[17] Moty Heiblum and Jay H. Harris. Analysis of curved optical waveguides by conformal transformation. IEEE Journal ofQuantum Electronics, 11(2):75–83, 1975. doi: 10.1109/JQE.1975.1068563.

[18] Martin Plöschner, Tomáš Tyc, and Tomáš Cižmár. Seeing through chaos in multimode fibres.<sup>ˇ</sup> Nature Photonics, 9 (8):529–535, 2015. doi: 10.1038/nphoton.2015.112.

[19] Hui Cao, Tomáš Cižmár, Sergey Turtaev, Tomáš Tyc, and Stefan Rotter. Controlling light propagation in multimode<sup>ˇ</sup> fibers for imaging, spectroscopy, and beyond. Advances in Optics and Photonics, 15(2):524–612, 2023. doi: 10.1364/AOP.484298.

[20] R. Clark Jones. A new calculus for the treatment of optical systems. i. description and discussion of the calculus. Journal ofthe Optical Society ofAmerica, 31(7):488–493, 1941. doi: 10.1364/JOSA.31.000488.

[21] Ivan P. Kaminow. Polarization in optical fibers. IEEE Journal ofQuantum Electronics, 17(1):15–22, 1981. doi: 10.1109/JQE.1981.1070626.

[22] R. Ulrich, S. C. Rashleigh, and W. Eickhoff. Bending-induced birefringence in single-mode fibers. Optics Letters, 5(6):273–275, 1980. doi: 10.1364/OL.5.000273.

[23] R. Ulrich and A. Simon. Polarization optics of twisted single-mode fibers. Applied Optics, 18(13):2241–2251, 1979. doi: 10.1364/AO.18.002241.

## A Propagation Hamiltonian

## A.1 Scalar Approach

The scalar field in a weakly guiding fiber obeys the Helmholtz equation $\nabla ^ { 2 } E + k _ { 0 } ^ { 2 } n ^ { 2 } ( x , y , z ) E = 0$ . The ansatz $E = \psi ( x , y , z ) e ^ { i k z }$ with $k = n _ { 0 } k _ { 0 }$ , the free-space wavenumber $k _ { 0 } = 2 \pi / \lambda$ and $n _ { 0 }$ the index on the fiber axis, gives

$$
\nabla ^ { 2 } \psi + 2 i k \partial _ { z } \psi + \left( k _ { 0 } ^ { 2 } n ^ { 2 } - k ^ { 2 } \right) \psi = 0 .\tag{22}
$$

When $\psi$ varies slowly w.r.t z on the scale of a wavelength, the approximation $| \partial _ { z } ^ { 2 } \psi | \ll | 2 k \partial _ { z } \psi |$ can be made and the $\partial _ { z } ^ { 2 } \psi$ term can be neglected. What remains is a 2-D Schrödinger-like equation on the xy-plane in which the propagation coordinate z plays the role of time,

$$
i \partial _ { z } \psi = { \cal H } \psi , \qquad { \cal H } = \frac { p ^ { 2 } } { 2 k } + V ( { \bf r } ) , \qquad V = \frac { k } { 2 } \left( 1 - \frac { n ^ { 2 } } { n _ { 0 } ^ { 2 } } \right) ,\tag{23}
$$

with $\pmb { p } = - i \nabla _ { \perp }$ acting on the two transverse $( x , y )$ coordinates only. For the ideal parabolic profile $n ^ { 2 } ( r ) =$ $n _ { 0 } ^ { 2 } \left[ 1 - 2 \Delta ( r / a ) ^ { 2 } \right]$ , with core radius a and

$$
\Delta = { \frac { n _ { 0 } ^ { 2 } - n _ { \mathrm { c l } } ^ { 2 } } { 2 n _ { 0 } ^ { 2 } } } = { \frac { \mathrm { N A } ^ { 2 } } { 2 n _ { 0 } ^ { 2 } } }\tag{24}
$$

is the relative index contrast between the axis and the cladding, Eq. (23) gives a harmonic potential,

$$
V _ { 0 } = \frac { k } { 2 } \Omega ^ { 2 } r ^ { 2 } , \qquad \Omega = \frac { \sqrt { 2 \Delta } } { a } ,\tag{25}
$$

and the unperturbed problem is the 2-D isotropic oscillator, whose eigenmodes are the Laguerre–Gauss modes $| \ell , m \rangle$ with propagation constants $\beta _ { g }$ for each mode group index $g = | \ell | + 2 m$

Bending enters through the geometry. A fiber bent to a local radius of curvature R is equivalent to a straight fiber whose index profile is tilted across the core [7, 17]. To first order in $r / R$ the equivalent profile is $n _ { \mathrm { e q } } ^ { 2 } \approx n ^ { 2 } \left( 1 + 2 \kappa \cdot r \right)$ where κ is the curvature vector with magnitude $1 / R$ pointing along the bend normal, and substituting into Eq. (23) adds the term $- k \kappa \cdot r$ , linear in the transverse coordinate. The bend also compresses the glass on the inside and stretches it on the outside, changing the density and with it the index. Writing $n - 1$ as proportional to density and using $\varepsilon _ { x x } = \varepsilon _ { y y } = - \sigma \varepsilon _ { z z }$ , where $\sigma$ is the Poisson ratio, that correction leaves the form of the term alone and rescales the curvature itself [18],

$$
\eta = 1 - \left( 1 - 2 \sigma \right) \frac { n _ { 0 } - 1 } { n _ { 0 } } ,\tag{26}
$$

so the fiber bends as though its curvature were ηκ. For fused silica $\sigma = 0 . 1 7$ , giving $\eta = 0 . 7 8 6$ at $n _ { 0 } = 1 . 4 7 8 3$ , against the $0 . 7 7 \pm 0 . 0 2$ measured in [18]. The bend term is therefore

$$
V _ { \mathrm { b e n d } } ( z ) = - k \eta \pmb { \kappa } ( z ) \pmb { \cdot } \pmb { r } .\tag{27}
$$

Twist is a cross-section that rotates along the fiber [10], through an accumulated angle $\theta ( z )$ at rate $\tau ( z ) = d \theta / d z$ , and collecting all contributions gives the ideal-fiber Hamiltonian

$$
H _ { \mathrm { i d e a l } } ( z ) = \frac { p ^ { 2 } } { 2 k } + \frac { k \Omega ^ { 2 } } 2 r ^ { 2 } - k \eta \pmb { \kappa } ( z ) \pmb { \cdot } \pmb { r } - \tau ( z ) L _ { z } .\tag{28}
$$

Realistically, a manufactured fiber is not an ideal parabola. Its measured index deviates from the parabolic profile by the perturbation $\delta n ^ { 2 } ( x , y )$ , and substituting $n ^ { 2 } = n _ { 0 } ^ { \frac { 1 } { 2 } } \bigl [ 1 - 2 \Delta ( r / a ) ^ { 2 } \bigr ] + \delta n ^ { 2 }$ into Eq. (23) leaves the harmonic potential untouched and adds the perturbative term,

$$
\delta V ( \pmb { r } ) = - \frac { k } { 2 n _ { 0 } ^ { 2 } } \delta n ^ { 2 } ( x , y ) .\tag{29}
$$

Crucially, this term breaks the degeneracy of the mode groups: the ideal parabola assigns a single $\beta _ { g }$ to every state of a given $^ { g , }$ and $\delta V$ non-trivially perturbs each mode’s propagation constant from the group value.

## A.2 Vector Approach

GRIN fibers are not currently manufactured well enough for the scalar, single-polarization approach to be valid [9, 19]: ellipticity, refraction perturbations, and stresses introduced during the fiber draw couple E’s polarization states and should be accounted for. The E-field obeys the vector wave equation [10]

$$
\nabla ^ { 2 } { \boldsymbol { E } } + k _ { 0 } ^ { 2 } n ^ { 2 } { \boldsymbol { E } } + \nabla \bigl ( { \boldsymbol { E } } \cdot \nabla \ln n ^ { 2 } \bigr ) = 0 .\tag{30}
$$

Using the two-state Jones vector ψ [20] in the E-field ansa $^ { \mathrm { t z , } }$ the scalar Hamiltonian in $\operatorname { E q . }$ . (23) carries over unchanged but the last term of Eq. (30) introduces the spin-orbit coupling term $H _ { \mathrm { S O } }$

$$
H _ { \mathrm { S O } } = - \frac { 1 } { 4 k } \left( \begin{array} { l l } { \partial _ { - } w _ { + } } & { \partial _ { - } w _ { - } } \\ { \partial _ { + } w _ { + } } & { \partial _ { + } w _ { - } } \end{array} \right) , \qquad \partial _ { \pm } = \partial _ { x } \pm i \partial _ { y } , \quad w _ { \pm } = \partial _ { \pm } \ln n ^ { 2 } ,\tag{31}
$$

where $\psi$ is projected onto $e _ { \pm } = ( e _ { x } \pm i e _ { y } ) / \sqrt { 2 }$ because the architecture assumes circular polarization.

Material anisotropy contributes three further terms acting on the polarization index. (1) Frozen stresses leave a fixed linear birefringence [21] whose slow axis sits at an angle α set by the quadrupole component of the measured profile:

$$
H _ { 1 } = \frac { b _ { 1 } } { 2 } \big ( e ^ { - 2 i \alpha } \sigma _ { + } + e ^ { 2 i \alpha } \sigma _ { - } \big ) ,\tag{32}
$$

where $\sigma _ { \pm } = ( \sigma _ { x } \pm i \sigma _ { y } ) / 2 . ( 2 )$ Bending compresses the glass on the inside of the bend and stretches it on the outside. The resulting stress-optics birefringence [22] has its principal axes set by the bend plane and its magnitude is quadratic in curvature. Define $\phi ( z )$ to be the angle $\kappa ( z )$ makes with the x-axis of the co-rotating spatial profile,

$$
H _ { 2 } = \frac { b _ { 2 } \kappa ^ { 2 } } { 2 } \left( e ^ { - 2 i \phi } \sigma _ { + } + e ^ { 2 i \phi } \sigma _ { - } \right) ,\tag{33}
$$

where $\kappa = | \kappa |$ . This is identical to $H _ { 1 }$ with the bend plane in place of the frozen axis and $b _ { 2 } \kappa ^ { 2 }$ in place of $b _ { 1 }$ with the sign of $b _ { 2 }$ fixing which of the two axes is slow. (3) Twisting shears the glass, and the photoelastic response to that shear is a circular birefringence

$$
H _ { 3 } = b _ { 3 } \tau ( z ) \sigma _ { z } .\tag{34}
$$

Collecting every contribution gives the Hamiltonian used for our synthesized fibers,

$$
H ( z ) = \frac { p ^ { 2 } } { 2 k } + \frac { k \Omega ^ { 2 } } { 2 } r ^ { 2 } + \delta V ( r ) - k \eta \kappa ( z ) \cdot r - \tau ( z ) L _ { z } + H _ { \mathrm { S O } } + H _ { 1 } + H _ { 2 } + H _ { 3 } .\tag{35}
$$

## B Fiber Model and Dataset Synthesis

## B.1 Index profile and mode basis

We simulate the Prysmian DRAKA WideCap OM5 fiber, with a $5 0 \mu \mathrm { m }$ core and NA 0.200, at $\lambda = 5 3 2$ nm. The index profile is the ideal parabolic profile with the measured perturbation $\Delta n ^ { 2 } ( x , y )$ reported in [9].

## B.2 Deformation synthesis

Each deformation is generated by partitioning the fiber into n equal-length sections, where n is uniformly sampled from $\{ 1 , 2 , 3 , 4 , 5 , 6 \}$ . Each section is assigned a curvature vector κ whose magnitude corresponds to a radius of curvature drawn uniformly from 3 to 40 cm and whose direction is drawn uniformly on the circle, so that the deformation is not confined to a single plane. The resulting piecewise-constant $\kappa ( z )$ is convolved with a Gaussian kernel to produce a continuous curvature profile, which makes the curvature vary smoothly along the fiber while remaining non-adiabatic. A twist $\tau ( z )$ is applied over the full length, with a total accumulation uniformly sampled from [−2 rad, 2 rad].

## B.3 Transmission matrix construction

The smoothed deformation profile is discretized into 100 segments of constant κ and $\tau ,$ the Hamiltonian in Eq. (35) is evaluated on each segment in the retained mode basis, and the transmission matrix is the ordered product of segment propagators as in Eq. (7). The undeformed transmission matrix $U _ { 1 }$ is obtained from the same construction with $\kappa = 0$ and $\tau = 0$ , so that $U _ { 1 }$ carries the fiber’s defects.

Table 3: Simulated DRAKA fiber parameters.
<table><tr><td>Symbol</td><td>Meaning</td><td>Value</td><td>Source</td></tr><tr><td colspan="4">Fiber (Prysmian DRAKA WideCap OM5)</td></tr><tr><td> $\lambda$ </td><td>build wavelength</td><td>532 nm</td><td> $\mathrm { N . A . }$ </td></tr><tr><td> $n _ { 0 }$ </td><td>index on axis</td><td>1.4783</td><td>manufacturer</td></tr><tr><td> $\mathrm { N A }$ </td><td>numerical aperture</td><td>0.200</td><td>manufacturer</td></tr><tr><td> $a$ </td><td>core radius</td><td>25 µm</td><td>manufacturer</td></tr><tr><td> $R _ { \mathrm { c l } }$ </td><td>cladding radius</td><td>62.5 µm</td><td>manufacturer</td></tr><tr><td> $\Delta$ </td><td>index contrast</td><td> $9 . 1 5 \times 1 0 ^ { - 3 }$ </td><td>Eq. (24)</td></tr><tr><td> $L$ </td><td>fiber length</td><td>1.0 m</td><td> $\mathrm { N . A . }$ </td></tr><tr><td> $k _ { 0 }$ </td><td>vacuum wavenumber</td><td> $1 . 1 8 1 \times 1 0 ^ { 7 } \mathrm { m ^ { - 1 } }$ </td><td> $2 \pi / \lambda$ </td></tr><tr><td> $k$ </td><td>axial wavenumber</td><td> $1 . 7 4 6 \times 1 0 ^ { 7 } \mathrm { m ^ { - 1 } }$ </td><td> $n _ { 0 } k _ { 0 }$ </td></tr><tr><td> $\Omega$ </td><td>oscillator frequency</td><td> $\mathrm { 5 4 1 2 m ^ { - 1 } }$ </td><td> ${ \sqrt { 2 \Delta } } / a$ </td></tr><tr><td> $\sigma$ </td><td>Poisson ratio</td><td>0.17</td><td>fused silica [18]</td></tr><tr><td> $\eta$ </td><td>elasto-optic curvature factor</td><td>0.786</td><td>Eq. (26)</td></tr><tr><td colspan="4">Polarization coefficients</td></tr><tr><td> $\alpha$ </td><td>frozen-stress axis</td><td> $+ 1 8 . 5 ^ { \circ }$ </td><td>[9], quadrupole of  $\delta n ^ { 2 }$ </td></tr><tr><td> $b _ { 1 }$ </td><td>frozen linear biref.</td><td> $3 . 0 \mathrm { r a d m } ^ { - 1 }$ </td><td>no source</td></tr><tr><td> $b _ { 2 }$ </td><td>bend stress-optic</td><td> $6 . 2 5 \times 1 0 ^ { - 3 } \mathrm { r a d m }$ </td><td>[22]</td></tr><tr><td> $b _ { 3 }$ </td><td>twist activity</td><td> $0 . 1 4$ </td><td>[23]; 0.13–0.16</td></tr></table>

Table 4: Step-index fiber parameters. The fiber is mode-matched to the DRAKA fiber of Table 3: same core radius, same $n _ { 0 } .$ , same wavelength, same defect field $\delta n ^ { 2 }$ , same polarization coefficients, and similar number of guided modes.
<table><tr><td>Symbol</td><td>Meaning</td><td>Value</td><td>Source</td></tr><tr><td colspan="4">Fiber (step-index, mode-matched to the DRAKA fiber)</td></tr><tr><td> $\lambda$ </td><td>build wavelength</td><td>532 nm</td><td>matched</td></tr><tr><td>n0</td><td>core index</td><td>1.4783</td><td>matched</td></tr><tr><td> $n _ { \mathrm { c l } }$ </td><td>cladding index</td><td>1.47131</td><td>Eq. (24)</td></tr><tr><td> $\mathrm { N A }$ </td><td>numerical aperture</td><td>0.1436</td><td>chosen for 452 guided modes</td></tr><tr><td> $a$ </td><td>core radius</td><td>25 µm</td><td>matched</td></tr><tr><td> $R _ { \mathrm { c l } }$ </td><td>cladding radius</td><td>62.5 µm</td><td>matched</td></tr><tr><td> $\Delta$ </td><td>index contrast</td><td> $4 . 7 2 \times 1 0 ^ { - 3 }$ </td><td>Eq. (24)</td></tr><tr><td> $V$ </td><td>normalized frequency</td><td>42.40</td><td>k0aNA</td></tr><tr><td> $L$ </td><td>fiber length</td><td>1.0m</td><td>matched</td></tr><tr><td> $k _ { 0 }$ </td><td>vacuum wavenumber</td><td> $1 . 1 8 1 \times 1 0 ^ { 7 } \mathrm { m ^ { - 1 } }$ </td><td> $2 \pi / \lambda$ </td></tr><tr><td> $k$ </td><td>axial wavenumber</td><td> $1 . 7 4 6 \times 1 0 ^ { 7 } \mathrm { m ^ { - 1 } }$ </td><td> $n _ { 0 } k _ { 0 }$ </td></tr><tr><td> $\Omega _ { \mathrm { e f f } }$ </td><td>oscillator frequency</td><td> $\mathrm { 3 0 4 8 m ^ { - 1 } }$ </td><td> ${ \sqrt { 2 \Delta } } / a$ </td></tr><tr><td> $\sigma$ </td><td>Poisson ratio</td><td>0.17</td><td>fused silica [18]</td></tr><tr><td> $\eta$ </td><td>elasto-optic curvature factor</td><td>0.786</td><td>Eq. (26)</td></tr><tr><td colspan="4">Polarization coefficients (unchanged from Table 3)</td></tr><tr><td> $\alpha$ </td><td>frozen-stress axis</td><td> $+ 1 8 . 5 ^ { \circ }$ </td><td>matched</td></tr><tr><td> $b _ { 1 }$ </td><td>frozen linear biref.</td><td> $3 . 0 \mathrm { r a d m } ^ { - 1 }$ </td><td>matched</td></tr><tr><td> $b _ { 2 }$ </td><td>bend stress-optic</td><td> $6 . 2 5 \times 1 0 ^ { - 3 }$  rad m</td><td>matched</td></tr><tr><td> $b _ { 3 }$ </td><td>twist activity</td><td>0.14</td><td>matched</td></tr></table>

## B.4 Reflector stack and measurement acquisition

The distal stack consists of a straight partial reflector $R _ { 1 } = \rho _ { 1 } I ,$ a tilted partial reflector $R _ { 2 } .$ , and a second tilted partial reflector $R _ { 3 }$ whose tilt axis makes an angle of $\pi / 4$ with that of $R _ { 2 }$ . Both $R _ { 2 }$ and $R _ { 3 }$ are tilted 0.6 deg. Reflector $R _ { 2 }$ is simulated 3mm behind $R _ { 1 }$ , and $R _ { 3 }$ is simulated 5mm behind $R _ { 2 }$ . The transmission matrices $T _ { i }$ are the scalars $t _ { i } I$ that satisfy energy conservation.

Each measurement $M _ { i }$ is assembled from N linearly independent probe signals, and each probe is subject to an independent rigid misalignment of the launch. We model it to first order in the four generators $\mathcal { I } _ { a } ^ { \cdot } \in \{ X , Y , \tilde { P } _ { x } , P _ { y } \}$ of transverse displacement and tilt, so that the measured matrix becomes

![](images/4373fb34426edb6f7df1314d9134878d0ca831f982fdaff2d1d87e05bef00859.jpg)  
Figure 7: Generated fiber deformations in the held-out test set.

$$
M ^ { \prime } = M + i \sum _ { a } \varepsilon _ { a } J _ { a } M D _ { a } + i M \sum _ { a } J _ { a } D _ { a } , \qquad D _ { a } = \mathrm { d i a g } ( d _ { j , a } ) ,\tag{36}
$$

where $d _ { j , a }$ is a randomly sampled from a Gaussian independently for every probe $j ,$ whose standard deviation $\sigma _ { j }$ is itself assigned per fiber, stratified over $\sigma _ { j } \in [ 0 , 0 . 1 0 ]$ so that the population spans the range with mean 0.05, and $\varepsilon _ { a } = ( 1 , 1 , - 1 , - 1 )$ is the parity of each generator under the reciprocity flip P. The $\sigma _ { j }$ varies for each synthetic fiber’s $\{ M _ { i } \}$ measurements, but $\bar { U } _ { 1 }$ is constructed with a smaller $\sigma _ { j } = 0 . 0 1$ since care can be taken before training to ensure it’s more accurate with techniques like multi-look averaging. The measurements are assumed single-look and therefore contain more "jitter" noise. Two terms appear because the same physical misalignment is seen on the launch and again on the return path, and the right-acting operator $\textstyle \sum _ { a } J _ { a } D _ { a }$ is dense rather than diagonal. Zero-mean Gaussian noise of standard deviation σ is then added to the reflection measurements.

![](images/177ed2afb44aae3d016ccb5c4f7e3376c907330afed2f6dfbab5dd63cd41b4de.jpg)

![](images/88a94f13c43cd8696591927f07d860598dfa1ac1543bf458526218cdf4e84cb7.jpg)

![](images/c38366fe4b2590bc4284b24d5798d889b7cf5101500794d34790f38b8aa5ff77.jpg)

![](images/1113d47762c8bd467f2dd2728b5a9e37b0d796f55b17a53c81da9363a464cc9a.jpg)

![](images/fef532fc09c2bd7cbc6e1645bca548bcbca688a0541ab128bf2a125bdc2364e3.jpg)

![](images/3884654e7a16a0ed0e265c024c3df560daf1319d714483c39b0a4f5b8f412b81.jpg)  
Figure 8: The third reflector measurement $M _ { 3 }$ of a single deformed fiber at three noise levels. The top row is the measurement as the architecture receives it, where hue is phase and brightness is magnitude. The bottom row is the phase deviation of each matrix element from its ideal value, $\mathrm { a r g } ( M _ { \mathrm { i d e a l } } ^ { * } \mathrm { \hat { ~ } o ~ } M _ { \mathrm { m e a s u r e d } } \mathrm { \bar { ) } }$ , where ◦ is the elementwise product and $M _ { \mathrm { i d e a l } } ^ { \ast }$ is the measurement that a single, unperturbed fiber state would produce without added noise. The $\sigma = 0$ column therefore isolates the per-probe deformation applied during acquisition, which is present in every column.