# Factorized Gradients for Scalable Highly-expressive Parametric Diffeomorphisms

Amit Aflalo<sup>1</sup> Eran Treister<sup>1</sup> Chaim Baskin<sup>2</sup> Oren Freifeld<sup>1</sup>

<sup>1</sup>The Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Israel <sup>2</sup>School of Electrical and Computer Engineering, Ben-Gurion University of the Negev, Israel

(a) Forward  
![](images/8dfd7919cafaf6d282fdb1a537555064e5686fee3c848bcad11a5af7bda5c4e5.jpg)

$$
\left( N _ { \mathrm { t e s s } } \right)
$$

(b) Gradient computation  
![](images/334136a21ed7d184813a2aebfa11c19ad6193316231c5bffd5bc17262ddeb246.jpg)  
(c) Memory

$$
\left( N _ { \mathrm { t e s s } } \right)
$$

![](images/ec147226a0f7bb0ef00114afe8cf6f0478e8fb12f02f80612015785c8084da12.jpg)

$$
\left( N _ { \mathrm { t e s s } } \right)
$$

Figure 1: Quantitative comparison, in 2D, of performance scaling with tessellation size $( N _ { \mathrm { t e s s } } ) _ { \mathrm { : } }$ between the proposed method, called FG-CPAB, and libcpab [10], the standard implementation of CPAB diffeomorphisms [13]. The evaluation was performed on a single RTX A6000 GPU with 125,000 2D points and a batch size of 16. (a) Forward pass latency. (b) Gradient computation latency. (c) Peak GPU memory allocation. In 2D, the domain is partitioned into $C = 4 N _ { \mathrm { t e s s } } ^ { 2 }$ triangular cells. Notably, the standard implementation does not support $\dot { N } _ { \mathrm { t e s s } } > 1 2$ , whereas ours scales efficiently.

## Abstract

Diffeomorphisms provide a flexible, topology-preserving mathematical tool for modeling complex spatial or temporal deformations, and are used in various scientific fields. However, making these deformations simultaneously highly expressive and computationally efficient remains challenging. Continuous Piecewise-Affine Based (CPAB) transformations [13, 14] offer an attractive solution by parameterizing a family of diffeomorphisms via continuous velocity fields that are piecewise affine w.r.t. a chosen tessellation of the domain. Importantly, $d = \dim ( { \bar { \theta } } )$ , the dimension of the parameter vector θ, depends on the fineness of the tessellation rather than on the data resolution. Despite this compact parameterization, CPAB optimization has long been hindered by the tight coupling between trajectory integration and gradient computation w.r.t. θ. We introduce FG-CPAB, a factorized-gradient formulation for scalable CPAB transformations. By separating integration from parameter-space projection, our method reduces gradient computation complexity from $\mathcal { O } ( d T N )$ to $\mathcal { O } ( T N + d C )$ , where T is the number of integration steps, N is the number of transformed points, and C is the number of tessellation cells. This reformulation yields speedups of up to four orders of magnitude $( \mathrm { e . g . , } \approx 1 0 ^ { 4 } \times$ in 2D), substantial memory savings $( \mathrm { e . g . , \approx 5 0 0 \times }$ reduction in peak GPU memory usage in 2D), and dramatically higher expressiveness. This makes, for the first time, fine-tessellation CPAB practical in 2D and 3D, enabling highly expressive parametric diffeomorphisms. For code and resources, see our project page.

## 1 Introduction

A diffeomorphism is a differentiable invertible function whose inverse is also differentiable. Diffeomorphisms are a natural choice in spatial modeling tasks that require smooth, invertible, and topology-preserving deformations. They are widely used in applications such as registration, spatial alignment, and geometric warping, where folding, tearing, or self-intersection would result in geomet rically invalid deformations. The standard way to construct such transformations is via integration of either a stationary velocity field (SVF) [2] or a time-dependent velocity field as in Large Deformation Diffeomorphic Metric Mapping (LDDMM) [4]. However, in practice, such methods often rely on a dense-grid discretization of the spatial domain. As a result, the number of parameters controlling the diffeomorphisms grows directly with the resolution of the domain (e.g., with the number of pixels in 2D). In modern 2D and especially 3D settings, this leads to high memory use, slow optimization, and difficult optimization landscapes.

Continuous Piecewise-Affine Based (CPAB) transformations, introduced by Freifeld et al. [13, 14], offer a compelling alternative. Instead of a dense-grid discretization, this approach tessellates the domain into cells (e.g., triangles in the 2D case) and constrains the velocity field to be a continuous piecewise-affine (CPA) function. A CPAB transformation is, by definition [13], obtained by integrating a CPA velocity field. Under a mild, easily enforced regularity condition, a CPAB transformation is a diffeomorphism<sup>1</sup>. A key point is that the number of parameters controlling a CPAB transformation depends on the tessellation, not the data resolution, thereby decoupling model complexity from data resolution. These transformations provide a compact low-dimensional parameterization while still capturing rich nonlinear deformations and have been used in diverse deep learning settings, including 1D applications such as time-series alignment and trainable activation functions [22, 28, 34, 38, 7, 26, 29, 23, 33], parameter-efficient multi-task fine-tuning [27], and 2D vision and generative modeling tasks [19, 11, 31, 32, 35, 37, 18].

Despite this compact parameterization, CPAB transformations remained hard to scale in practice. The bottleneck lies in computing the loss gradient, $\nabla _ { \boldsymbol { \theta } } \mathcal { L } ,$ which conventionally requires propagating parameter sensitivities through the transformation $x \mapsto \Phi ^ { \theta } ( x )$ . In the standard formulation, sensitivities w.r.t. all $d = \dim ( \theta )$ parameters are propagated through all $T$ integration steps for each of the N spatial points, yielding a backward complexity of $\mathcal { O } ( d T N )$ . This tight coupling between parameter dimension and trajectory integration makes optimization prohibitively expensive once tessellations become fine or spatial inputs become large. Consequently, existing CPAB implementations are often limited to coarse tessellations, which restricts practical expressiveness. Moreover, even with coarse tessellations in 2D, CPAB optimization is too slow for most practical applications. In 3D, the problem worsens, and scalable CPAB optimization has remained largely out of reach, despite the wide use of diffeomorphisms in this setting [5, 39, 42, 16, 8, 36, 17, 25, 6].

In this paper, we address this bottleneck with FG-CPAB, a factorized-gradient formulation that separates trajectory integration from parameter-space projection. Instead of propagating per-parameter sensitivities during integration, we perform a single adjoint sweep that accumulates low-dimensional cell-wise geometric statistics and recovers all parameter gradients through one projection onto the CPA basis. This reduces the backward complexity from O(dTN) to $\mathcal { O } ( \bar { T } N + \bar { d } C )$ , where $C$ is the number of tessellation cells. We further introduce a sparse basis formulation that reduces memory use and allows the final projection step to fully exploit basis sparsity, thereby realizing the full computational benefit of the factorized gradients in practice.

Combined with optimized GPU kernels, the resulting method makes CPAB practical at scales that were previously inaccessible. For example, in 2D, our implementation achieves up to 75× faster forward passes, $1 0 ^ { 4 } \times$ faster backward passes, and over 500× lower peak memory compared with the standard implementation [10]. Such gains enable efficient optimization of fine-tessellation CPAB diffeomorphisms in both 2D and 3D, and make CPAB competitive as a modern primitive for large-scale nonrigid spatial modeling. To summarize, our main contributions are:

• FG-CPAB. We introduce a gradient-factorized backward pass based on discrete-adjoint moment accumulation. Unlike standard adjoint methods, which still carry parameter dependence through the backward integration, our method exploits the CPA basis structure to accumulate parameterindependent cell-wise moments and recover gradients only in a final sparse projection step.

• An efficient implementation. We combine factorized gradients, optimized GPU kernels, and a sparse basis construction to achieve large runtime and memory gains in practice.

• Scalable diffeomorphic optimization in 2D and 3D. We demonstrate that the resulting formulation makes fine-tessellation CPAB practical for spatial modeling and competitive with other diffeomorphic primitives in both 2D and 3D.

## 2 Background and Related Work

Let $\Omega \subseteq \mathbb { R } ^ { K }$ denote the spatial domain where $K \in \{ 1 , 2 , 3 \}$ . A diffeomorphism is a differentiable bijection $\Phi : \Omega  \Phi ( \Omega ) \dot { \subseteq } \mathbb { R } ^ { K }$ with a differentiable inverse. Such transformations preserve topology, avoiding tearing, folding, and self-intersection. A standard way to construct such a diffeomorphism Φ is to integrate a well-behaved velocity field v over time. Given an initial point $\pmb { p } ^ { ( 0 ) } = p _ { 0 } \in \Omega$ , its trajectory follows the ordinary differential equation (ODE)

$$
{ \pmb { \dot { p } } } ^ { ( t ) } = { \pmb v } ( { \pmb p } ^ { ( t ) } , t ) , \quad t \in [ 0 , 1 ] ,\tag{1}
$$

with the resulting mapping defined as $\Phi ( p _ { 0 } ) = p ^ { ( 1 ) }$ . Some frameworks, such as Large Deformation Diffeomorphic Metric Mapping (LDDMM) [4], use time-varying velocity fields. A widely used alternative is the Stationary Velocity Field (SVF) formulation [2], which assumes the velocity field is constant over time $( { \mathrm { i . e . , } } v ( p , t ) = v ( p ) )$ . This simplification significantly reduces computational complexity while still producing a rich class of diffeomorphisms.

Dense vs. Compact Parameterizations. Traditional methods for constructing diffeomorphisms [2, 4] parameterize v densely on the spatial grid. While this enables highly expressive deformations, it couples the number of parameters directly to the resolution of the domain. Even in 2D, this can already involve hundreds of thousands of parameters, and in 3D, the parameter count can reach millions, leading to high memory requirements and a large parameter space that can be difficult to optimize without explicit regularization. While alternative formulations restrict dense velocity fields to finite-dimensional Lie algebras [40, 41], they do not directly address the scaling bottlenecks.

Continuous Piecewise-Affine Based (CPAB) transformations [13, 14] address this limitation by imposing structure on the velocity field. Following [13], we consider tessellations of Ω into $C$ convex polyhedral cells, possibly unbounded, and refer to these as convex polyhedral tessellations. A continuous piecewise-affine (CPA) function, w.r.t. a given tessellation, is a continuous function whose restriction to each cell is affine. A simplicial tessellation is the special case in which all cells are simplices: intervals in 1D, triangles in 2D, and tetrahedra in 3D. Our implementation uses simplicial tessellations when zero boundary conditions are imposed. Otherwise, we extend Ω to $\mathbb { R } ^ { K }$ and augment the simplicial tessellation with unbounded convex polyhedral cells, following [14]. For simplicity, the presentation below focuses on simplicial cells, but the formulation also applies to the additional unbounded cells.

CPA functions are Lipschitz continuous, and a CPAB transformation is defined by integrating a CPA velocity field. These fields form a finite-dimensional linear space, constructed as follows. Within each cell $c ,$ the stationary velocity field v takes the form

$$
{ \pmb v } _ { c } ( { \pmb p } ) = { \bf A } _ { c } { \widetilde { \pmb p } } , \qquad { \pmb p } \in \mathrm { c e l l } c ,\tag{2}
$$

where $\mathbf { A } _ { c } \in \mathbb { R } ^ { K \times ( K + 1 ) }$ contains the affine velocity coefficients for cell $^ { c , }$ and $\widetilde { \pmb { p } } = ( \pmb { p } ^ { \top } , 1 ) ^ { \top }$ denotes homogeneous coordinates. Defining the augmented matrix $\widetilde { \mathbf { A } } _ { c } = \left\lceil \mathbf { A } _ { c } ^ { \top } \ \mathbf { 0 } \right\rceil ^ { \top } \in \mathbb { R } ^ { ( K + 1 ) \times ( K + 1 ) }$ , we can specialize Eq. 1 to the stationary affine velocity field within cell $c ,$ obtaining $\dot { \widetilde { \pmb p } } ^ { ( t ) } = \widetilde { \bf A } _ { c } \widetilde { \pmb p } ^ { ( t ) }$ Its closed-form solution is $\widetilde { \pmb { p } } ^ { ( t ) } = \mathrm { e x p } ( t \widetilde { \bf A } _ { c } ) \widetilde { \pmb { p } } ^ { ( 0 ) }$ , where exp denotes the matrix exponential. This solution is valid as long as the trajectory remains in c throughout [0, t].

To ensure that the global velocity field v is continuous, we impose linear constraints requiring the affine velocity fields of neighboring cells to agree on their shared boundaries. Concatenating the cell-wise affine coefficients of a CPA velocity field yields a vector of length $D = C K ( K + 1 )$ , which must lie in the null space of the continuity constraint system. Let $d \leq D$ denote the dimension of this null space and let $\mathbf { B } \in \mathbb { R } ^ { D \times d }$ have columns forming a basis for it. The stacked coefficients of any CPA velocity field can therefore be written as

$$
\mathrm { v e c } \big ( \{ \mathbf { A } _ { c } \} _ { c = 1 } ^ { C } \big ) = \mathbf { B } \pmb { \theta }\tag{3}
$$

where $\pmb { \theta } \in \mathbb { R } ^ { d }$ is the vector of parameters controlling the velocity field. Crucially, this guarantees that any choice of θ produces a CPA velocity field, shifting the dimensionality of the optimization problem from the spatial resolution of the data to the complexity of the tessellation. The dimension $d ,$ which controls the expressiveness of the CPA velocity fields, increases with tessellation refinement.

We remark that, as shown in [13], additional optional linear constraints such as zero boundary conditions or ones that imply volume preservation can be simply added to the constraint system above. For a more detailed explanation of CPAB transformations, see [13, 14, 1].

Previous CPAB implementations. The original CPAB framework [13] and its main successor libcpab [10] compute gradients by propagating sensitivities through the integration trajectory (see [12] for details). This couples the spatial integration to the parameter dimension d, yielding a prohibitive $\mathcal { O } ( d T N )$ backward complexity that restricts prior methods to coarse tessellations. Although early CPAB work [13, 14] also considered MCMC-based inference, this approach scales poorly in the fine-tessellation regime and cannot provide the gradients required for end-to-end learning; see Appendix J.

Adjoint methods and factorization. Adjoint sensitivity methods are standard for differentiating through ODE solvers with reduced memory cost [9]. However, for CPAB transformations, a generic adjoint formulation does not address the main computational bottleneck: gradients still involve the dependence of the velocity field on all d basis parameters. Moreover, as noted by ANODE [15], continuous adjoint formulations can yield inconsistent gradients under forward/backward discretiza tion mismatch. Our approach is therefore closer in spirit to a discretize-then-optimize adjoint, but specialized to the CPAB structure: the linear CPA basis allows trajectory-dependent terms to be accumulated as cell-local quantities before a final projection onto the parameter basis.

## 3 Method

We introduce FG-CPAB, a gradient-factorization technique based on discrete-adjoint moment accumulation that decouples spatial trajectory integration from the parameter dimension. Existing CPAB implementations compute gradients w.r.t. the transformation parameters θ by propagating forward sensitivities through the trajectory, incurring complexity $\mathcal { O } ( d T \bar { N } )$ , where T is the number of integration steps, N is the number of spatial points, and $\bar { d } = \mathrm { d i m } ( \boldsymbol { \theta } )$ . Since $d$ grows with tessellation resolution and spatial dimension, this becomes prohibitive for fine tessellations and higher-dimensional inputs.

Our method’s novelty is not merely in the use of a discrete adjoint, which alone does not remove the coupling to the parameter dimension. Rather, we also exploit the linear CPA basis to reorganize adjoint contributions into cell-local geometric moments that are independent of d. The full parameter gradient is then recovered by a single projection onto the basis, shifting all dependence on θ to this final step and reducing the backward complexity from $\mathcal { O } ( d T N )$ to $\mathcal { O } ( T N + \overline { { d C } } )$ . This enables a much higher d (i.e., finer tessellations), hence also much more expressive diffeomorphisms.

Furthermore, to maximize these efficiency gains, we also introduce a sparse formulation of the CPA basis matrix B (detailed in Appendix F). This sparse construction improves the practical efficiency and memory footprint of both the forward and backward passes.

## 3.1 Trajectory integration and gradient computation

In the forward pass, the affine ODE within cell c admits the update

$$
\widetilde { \pmb { p } } ^ { ( t + 1 ) } = \exp ( h \widetilde { \mathbf { A } } _ { c } ) \widetilde { \pmb { p } } ^ { ( t ) }\tag{4}
$$

where t indexes integration steps of size h. The update is exact provided that the trajectory remains within the same cell. Accordingly, we precompute $\mathbf { T } _ { c } = \exp ( h \widetilde { \mathbf { A } } _ { c } )$ , so that each forward step reduces to a matrix-vector multiplication. When a step crosses a cell boundary, in both the forward and gradient computation steps, we assume that the chosen integration step h is sufficiently small, so the update is well approximated by averaging the velocity over the step. This is consistent with the standard CPAB formulation [10]. For the gradient computation, however, we use a different discretization. Specifically, we adopt an explicit midpoint rule (second-order Runge–Kutta),

$$
\begin{array} { r } { \pmb { p } ^ { ( t + 1 ) } = \pmb { p } ^ { ( t ) } + h \mathbf { A } _ { c } \widetilde { \pmb { p } } _ { \mathrm { m i d } } , \qquad \pmb { p } _ { \mathrm { m i d } } = \pmb { p } ^ { ( t ) } + \frac { h } { 2 } \mathbf { A } _ { c } \widetilde { \pmb { p } } ^ { ( t ) } . } \end{array}\tag{5}
$$

Using different forward and backward schemes is also the approach adopted in the original CPAB formulation [13] and is also natural from the Optimize-then-Discretize viewpoint [9]. However, since our velocity field is usually not differentiable at inter-cell boundaries, an Optimize-then-Discretize adjoint may yield inconsistent gradients (see ANODE [15]), because the backward adjoint trajectory can pass through a different sequence of cells than the forward trajectory. We therefore adopt a Discretize-then-Optimize treatment for the backward pass, and combine it with checkpointing (see § 3.3) for memory efficiency. See Appendix I for discussion and a numerical gradient check.

## 3.2 Adjoint factorization via moment accumulation

Let $\mathbf { A } _ { c , \mathrm { l i n } } \in \mathbb { R } ^ { K \times K }$ denote the linear block of $\mathbf { A } _ { c } .$ , and let $\mathbf { B } _ { i , c }$ denote the i-th basis field’s affine matrix in cell $c .$ Differentiating Eq. 5 w.r.t. $\theta _ { i }$ (the i-th component of θ), using $\partial { \bf A } _ { c } / \partial \theta _ { i } = { \bf B } _ { i , c }$ yields the sensitivity recursion

$$
\frac { \partial \pmb { p } ^ { ( t + 1 ) } } { \partial \theta _ { i } } = h \mathbf { B } _ { i , c } \widetilde { \pmb { p } } _ { \mathrm { m i d } } ^ { ( t ) } + \frac { h ^ { 2 } } { 2 } \mathbf { A } _ { c , \mathrm { l i n } } \mathbf { B } _ { i , c } \widetilde { \pmb { p } } ^ { ( t ) } + \mathbf { J } \frac { \partial \pmb { p } ^ { ( t ) } } { \partial \theta _ { i } } ,\tag{6}
$$

$$
\begin{array} { r } { \mathrm { w h e r e } \quad \mathbf { J } : = \frac { \partial \pmb { p } ^ { ( t + 1 ) } } { \partial \pmb { p } ^ { ( t ) } } = \mathbf { I } + h \mathbf { A } _ { c , \mathrm { l i n } } + \frac { h ^ { 2 } } { 2 } \mathbf { A } _ { c , \mathrm { l i n } } ^ { 2 } } \end{array}\tag{7}
$$

is the Jacobian of one midpoint step with respect to the current position.Note that J depends on $\mathbf { A } _ { c , \mathrm { l i n } }$ but not on the translation component of $\mathbf { A } _ { c }$ (see Appendix H). The translation still influences the forward trajectory through the positions themselves.

A direct implementation of Eq. 6 would still require storing or propagating the full N × d sensitivities. Instead, we introduce the adjoint vector $\boldsymbol { \lambda } ^ { ( t ) }$ , defined as the gradient of the loss w.r.t. the position at time t, and propagate it backward in time. The adjoint satisfies the discrete recurrence

$$
\begin{array} { r } { \boldsymbol { \lambda } ^ { ( t ) } = \left( \mathbf { I } + h \mathbf { A } _ { c , \mathrm { l i n } } ^ { \top } + \frac { h ^ { 2 } } { 2 } ( \mathbf { A } _ { c , \mathrm { l i n } } ^ { \top } ) ^ { 2 } \right) \boldsymbol { \lambda } ^ { ( t + 1 ) } = \mathbf { J } ^ { \top } \boldsymbol { \lambda } ^ { ( t + 1 ) } . } \end{array}\tag{8}
$$

Using this adjoint recurrence and left-multiplying Eq. 6 by $\lambda ^ { ( t + 1 ) \top }$ yields

$$
\lambda ^ { ( t + 1 ) \top } \frac { \partial \pmb { p } ^ { ( t + 1 ) } } { \partial \theta _ { i } } = \lambda ^ { ( t + 1 ) \top } \left( h \mathbf { B } _ { i , c } \widetilde { \pmb { p } } _ { \mathrm { m i d } } ^ { ( t ) } + \frac { h ^ { 2 } } { 2 } \mathbf { A } _ { c , \mathrm { l i n } } \mathbf { B } _ { i , c } \widetilde { \pmb { p } } ^ { ( t ) } \right) + \lambda ^ { ( t ) \top } \frac { \partial \pmb { p } ^ { ( t ) } } { \partial \theta _ { i } } .\tag{9}
$$

The sensitivity recurrence propagates basis contributions through subsequent timesteps, making it difficult to group them by cell and factor out the basis. We note, however, that summing over all timesteps produces a telescoping series, so that all propagated terms cancel exactly (see Appendix G), leaving local contributions that can be grouped by cell to factor out the basis. Concretely, let $\mathcal { L }$ denote a loss and let $c _ { n } ^ { ( t ) }$ denote the cell containing point n at step t. Then

$$
\begin{array} { r l } & { \frac { \partial \mathcal { L } } { \partial \theta _ { i } } = \displaystyle \sum _ { n = 1 } ^ { N } \sum _ { t = 0 } ^ { T - 1 } \lambda _ { n } ^ { ( t + 1 ) \top } \left( h \mathbf { B } _ { i , c _ { n } ^ { ( t ) } } \widetilde { \pmb { p } } _ { n , \mathrm { m i d } } ^ { ( t ) } + \frac { h ^ { 2 } } { 2 } \mathbf { A } _ { c _ { n } ^ { ( t ) } , \mathrm { l i n } } \mathbf { B } _ { i , c _ { n } ^ { ( t ) } } \widetilde { \pmb { p } } _ { n } ^ { ( t ) } \right) . } \end{array}\tag{10}
$$

Moment accumulation. The remaining direct terms are bilinear in the adjoint λ and the homogeneous position ${ \widetilde { \pmb { p } } } .$ Since the basis slice $\mathbf { B } _ { i , c }$ is fixed for each cell c, we use the Frobenius identity $\begin{array} { r } { \lambda ^ { \top } { \bf A } \mathbf { B } _ { i , c } \tilde { \pmb { p } } = \langle { \bf B } _ { i , c } , ( { \bf A } ^ { \top } \pmb { \lambda } ) \tilde { \pmb { p } } ^ { \top } \rangle _ { F } } \end{array}$ to separate the basis from the trajectory-dependent factors. We then accumulate these factors over all visits to each cell. Defining $\mathcal { T } _ { c } : = \{ ( n , t ) \mid c _ { n } ^ { ( t ) } = c \}$ , the moment matrix for cell c is

$$
\mathbf { M } _ { c } = \sum _ { ( n , t ) \in \mathbb { Z } _ { c } } \left[ h \lambda _ { n } ^ { ( t + 1 ) } \widetilde { \pmb { p } } _ { n , \mathrm { m i d } } ^ { ( t ) \top } + \frac { h ^ { 2 } } { 2 } \left( \mathbf { A } _ { c , \mathrm { l i n } } ^ { \top } \pmb { \lambda } _ { n } ^ { ( t + 1 ) } \right) \widetilde { \pmb { p } } _ { n } ^ { ( t ) \top } \right] .\tag{11}
$$

Crucially, $\mathbf { M } _ { c } \in \mathbb { R } ^ { K \times ( K + 1 ) }$ is decoupled from $\mathbf { B } _ { i , c } ,$ hence independent of the dimension d.

Gradient recovery. Once the moment matrices have been accumulated, all parameter gradients are recovered by a single sparse basis projection

$$
\left| \frac { \partial \mathcal { L } } { \partial \theta _ { i } } = \sum _ { c = 1 } ^ { C } \langle \mathbf { B } _ { i , c } , \mathbf { M } _ { c } \rangle _ { F } , \qquad \mathrm { e q u i v a l e n t l y } \qquad \nabla _ { \theta } \mathcal { L } = \mathbf { B } ^ { \top } \operatorname { v e c } \bigl ( \{ \mathbf { M } _ { c } \} _ { c = 1 } ^ { C } \bigr ) \right|\tag{12}
$$

The core factorization result is Eq. 12. The trajectory-dependent computation is independent of $d ,$ with all explicit d-dependence confined to the final projection step. In practice, this final projection further benefits from a sparse basis construction of $\mathbf { \hat { B } } ,$ which we derive in Appendix F. This sparse formulation reduces the memory footprint of the basis representation and improves the efficiency of the projection step.

## 3.3 Memory-efficient adjoint via segmented checkpointing

While factorization removes dependence on d, the reverse sweep still requires access to intermediate forward states. In our GPU implementation, each thread processes a single spatial point, so storing the full trajectory would require $\bar { \mathcal { O } } ( T N )$ storage across all threads (equivalently, $\mathcal { O } ( T )$ per thread), which increases register pressure and can spill to local memory. Thus, we use segmented checkpointing with segment length $S$ (we use $S = 4 )$ . Instead of storing all $T + 1$ states along the trajectory, each thread stores only the states at segment boundaries, giving $\lceil T / S \rceil$ checkpoints per point. During the reverse sweep, each segment is replayed from its checkpoint and immediately used for adjoint propagation and moment accumulation. This reduces storage from $\mathcal { O } ( T N )$ to ${ \mathcal O } ( \dot { N } ( T / S + S ) )$ ) overall, or from $\mathcal { O } ( T )$ to $\mathcal { O } ( T / S + S )$ per thread, at the cost of only constant-factor recomputation. See Appendix D.

## 3.4 Gradient computation algorithm

Incorporating segmented checkpointing, the factorization yields three sequential phases:

1. Forward checkpointing $( \mathcal { O } ( T N ) )$ ). Integrate the trajectory forward, caching only sparse checkpoints at boundaries of length S.

2. Adjoint sweep + moment accumulation $( \mathcal { O } ( T N ) )$ . Iterate backward through the segments. For each segment, recompute the local forward trajectory from the checkpoint, propagate the adjoint via the transposed midpoint rule (Eq. 8), and atomically accumulate the products into $\mathbf { M } _ { c } \left( \mathrm { E q . } 1 1 \right)$

3. Sparse basis projection $( \mathcal { O } ( d C ) )$ . Recover all d gradients via Eq. 12.

The total gradient computation cost is $\mathcal { O } ( T N + d C )$ : integration and adjoint propagation are independent of d, and projection is independent of N and T.

## 4 Results

In this section, we evaluate FG-CPAB along several axes: computational efficiency vs. the standard CPAB implementation, integration runtime vs. optimized diffeomorphic baselines, expressiveness and optimization behavior vs. dense SVFs, performance on 3D deformable registration, and CPAB as a low-dimensional geometry code for operator learning. We also show that composing CPAB transformations (which are not algebraically closed under composition) increases expressivenes further, and conclude with ablations on tessellation resolution, basis sparsity, and solver step count.

## 4.1 Acceleration and memory savings over the standard CPAB implementation

FG-CPAB substantially improves CPAB scaling over the standard libcpab [10] implementation. As Figure 1 shows, the gains are largest in the gradient computation phase, where standard CPAB is bottlenecked by parameter-dependent sensitivity propagation.

In 2D, FG-CPAB achieves up to $7 5 \times$ faster forward passes, $1 0 ^ { 4 } \times$ faster gradient computation, and over 500× lower peak memory. Under our benchmark setup, the standard implementation does not run beyond $\bar { N } _ { \mathrm { t e s s } } ~ = ~ 1 2$ because its backward pass couples the parameter dimension directly to kernel-launch geometry (see Appendix A.1), whereas FG-CPAB continues to scale to finer resolutions. This pattern is similar in other dimensions (see Appendix $\mathbf { A } ) \colon$ in 3D, up to 60× forward acceleration, $4 , 4 7 7 \times$ backward acceleration, and a 180× reduction in peak memory; in 1D, up to a 301× backward speedup. These results match the proposed complexity reduction: by decoupling trajectory integration from parameter-space projection, FG-CPAB removes the main bottleneck that has historically restricted CPAB to coarse tessellations.

## 4.2 Forward and gradient computation runtime

We next evaluate FG-CPAB against optimized VoxelMorph SVF [3] and torch-lddmm [24] baselines on standard 2D and 3D grids. All methods are run with a batch size of 32. FG-CPAB uses a $1 5 \times 1 5$ tessellation in 2D $( 4 \times \mathsf { \bar { 1 5 } ^ { 2 } }$ triangles) and a $1 5 \times 1 5 \times 1 5$ tessellation in 3D $\mathrm { ( 5 \times 1 5 ^ { 3 } }$ tetrahedra). For each method, we measure the latency of the forward spatial transformation and the latency of computing the gradients w.r.t. the transformation parameters θ.

FG-CPAB is faster than both baselines, as reported in Table 1. In 2D, it reduces total runtime by 2.4× relative to VoxelMorph and by 21.4× relative to torch-lddmm. The same trend holds in 3D, with speedups of $2 . 3 \times$ and 21.2×, respectively. These improvements are due to the combination of CPAB’s low-dimensional parameterization and the proposed factorized gradient computation, which avoids propagating parameter-dependent sensitivities during trajectory integration.
<table><tr><td></td><td colspan="3">2D Integration (512 × 512)</td><td colspan="3">3D Integration (128 × 128 × 128)</td></tr><tr><td>Method</td><td>Forward (ms)</td><td>Gradient (ms)</td><td>Peak Mem (MiB)</td><td>Forward (ms)</td><td>Gradient (ms)</td><td>Peak Mem (MiB)</td></tr><tr><td>FG-CPAB</td><td> $\mathbf { 1 . 6 9 \pm 0 . 0 6 }$ </td><td> ${ \bf 8 . 6 4 \pm 0 . 0 2 }$ </td><td>197.97</td><td> ${ \bf 3 1 . 1 0 \pm 0 . 1 8 }$ </td><td> $\mathbf { 1 1 7 . 5 0 \pm 0 . 3 4 }$ </td><td>2351.61</td></tr><tr><td>VoxelMorph SVF</td><td> $1 1 . 4 1 \pm 0 . 0 2$ </td><td> $1 3 . 4 2 \pm 0 . 0 8$ </td><td>1330.13</td><td> $1 1 9 . 1 4 \pm 0 . 0 3$ </td><td> $2 2 0 . 2 1 \pm 0 . 2 4$ </td><td>14432.13</td></tr><tr><td>torch-lddmm</td><td> $5 1 . 2 3 \pm 0 . 4 1$ </td><td> $1 6 9 . 7 5 \pm 0 . 0 6$ </td><td>3162.13</td><td> $7 3 2 . 8 1 \pm 0 . 4 0$ </td><td> $2 4 1 4 . 1 2 \pm 0 . 5 3$ </td><td>43288.13</td></tr></table>

Table 1: Forward transformation and parameter gradient (∇<sub>θ</sub>) latency, alongside peak memory footprint, on 2D and 3D data.

## 4.3 Expressiveness and optimization stability compared with dense SVFs

We compare FG-CPAB against a dense stationary velocity field (SVF) baseline on a synthetic registration benchmark designed to isolate optimization behavior under a matched budget (1000 fit steps). In both 2D $( 5 1 2 \times 5 1 2 )$ and 3D $( 1 2 8 \times 1 2 8 \times 1 2 8 )$ , we generate blob-based source images, warp them with sampled smooth deformations to form targets, and fit either a CPAB transformation (using FG-CPAB) or a dense SVF integrated with 7 scaling-and-squaring steps. FG-CPAB uses a 15 × 15 tessellation in 2D $( 4 \times 1 5 ^ { 2 }$ triangles) and a $1 5 \times 1 5 \times \bar { 1 5 }$ tessellation in 3D $\mathrm { ( 5 \times 1 5 ^ { 3 } }$ tetrahedra); full implementation details are provided in Appendix C.

Table 2 reports reconstruction error, fitting time, and the fraction of non-positive Jacobian determinants, while Figure 2 shows qualitative results. Under the same optimization budget, FG-CPAB consistently reaches lower residual error, converges faster, and exhibits fewer topology violations. These gains arise from the compact CPAB parameterization: optimizing within the low-dimensional space spanned by B acts as an implicit regularizer, substantially improving optimization behavior, potentially due to a smoother optimization landscape than that of a dense SVF parameterization.

Importantly, our claim is not that dense SVFs lack representational power. Rather, the comparison highlights a practical trade-off between asymptotic flexibility and optimization efficiency. Dense SVFs can represent highly detailed deformations, but their large parameter space makes them much harder to optimize reliably under realistic compute budgets. In contrast, CPAB offers a more compact parameterization that appears to facilitate efficient optimization while retaining sufficient expressiveness for the evaluated deformations.

To further probe this trade-off, Appendix C.1 studies extended-budget optimization and regularized dense SVFs. Those results show that heavily regularized dense SVFs can eventually approach, and in some cases surpass, CPAB in final reconstruction error, but only after orders of magnitude more optimization steps. Thus, the main advantage of CPAB is practical: it achieves strong alignment quality much more reliably and with substantially less compute.
<table><tr><td>Dim</td><td>Params (C,S)</td><td>MSE C↓</td><td>MSE S↓</td><td> $\mathrm { F i t } { \bf C } \left( { \bf s } \right) \downarrow$ </td><td>Fit S (s) ↓</td><td> $\left| J \right| \leq 0 \left( \% \right) \left( \mathbf { C } , \mathbf { S } \right) \downarrow$ </td></tr><tr><td>2D</td><td>842 / 524288</td><td> $\mathbf { 8 . 8 5 3 e { \cdot } 0 5 \pm 3 . 9 e { \cdot } 0 5 }$ </td><td> $2 . 6 5 1 \mathrm { e } { - } 0 3 \pm 1 . 2 \mathrm { e } { - } 0 3$ </td><td> ${ \bf 0 . 5 2 5 \pm 0 . 0 4 4 }$ </td><td> $1 . 3 5 6 \pm 0 . 0 0 7$ </td><td>0.028% / 0.178%</td></tr><tr><td>3D</td><td>8232 / 6291456</td><td> $7 . 2 5 5 \mathrm { e } { \bf - 0 . 0 5 } \pm 2 . 5 \mathrm { e } { \bf - 0 . 5 }$ </td><td> $1 . 6 7 6 \mathrm { e } { - } 0 4 \pm 4 . 1 \mathrm { e } { - } 0 5$ </td><td> ${ \bf 6 . 1 4 0 \pm 0 . 0 2 1 }$ </td><td> $1 2 . 9 0 6 \pm 0 . 0 2 1$ </td><td>0.012% / 0.021%</td></tr></table>

Table 2: Quantitative synthetic registration results comparing FG-CPAB (C) and dense SVF (S) under the same optimization budget in 2D and 3D. Metrics report final reconstruction MSE, wall-clock fitting time, and the percentage of spatial locations with non-positive Jacobian determinant.

![](images/5e5bc373e2be0d691298d8aae2846abd859ff61db2f05fd823a867d615e4a9a9.jpg)  
Figure 2: Qualitative comparison of CPAB and dense SVF on synthetic deformation fitting. Left to right: Source image, deformed target, absolute alignment error after optimization with CPAB, and absolute alignment error after optimization with dense SVF under the same optimization budget. CPAB achieves near-zero residual error despite its compact parameterization, whereas dense SVF more often exhibits structured residual artifacts under the matched budget.

## 4.4 CPAB for 3D deformable registration

We evaluate FG-CPAB in the classical instance-specific optimization setting for deformable registration. For each test volume pair, we directly optimize the compact CPAB parameter vector θ (using the factorized gradients from § 3). No offline pre-training on any dataset is performed. This setup highlights the practical ease of optimization unlocked by our method: the compact parameterization and $\mathcal { \bar { O } } ( T N + \mathbf { \bar { \it d C } } )$ backward pass enable fast, stable, topology-preserving convergence per volume pair. Full implementation details, including the optimizer and loss function, are provided in Appendix E.1.

Table 3 reports results on the IXI brain MRI benchmark. A single CPAB transformation reaches $0 . 7 6 0 \pm 0 . { \mathrm { \bar { 0 2 3 } } }$ Dice with near-perfect topology preservation $( < 0 . 0 0 0 1 \%$ non-positive Jacobians). This performance is competitive with strong learned baselines, despite requiring no offline pretraining. Direct per-pair optimization converges in ≈ 5 seconds per volume pair on an RTX A6000 GPU, compared with ≈ 135 seconds for LDDMM. FG-CPAB therefore offers a substantially better accuracy-runtime trade-off than classical optimization-based diffeomorphic methods.

We further observe that sequential composition of CPAB transformations increases expressiveness while preserving topology. Composing i transformations raises the Dice score from 0.760 (i = 1) to 0.765 (i = 4), with the fraction of non-positive Jacobians remaining below 0.0001 throughout. These results demonstrate that the proposed factorized formulation makes fine-tessellation CPAB a practical, optimization-friendly primitive for 3D deformable registration.

Table 3: Registration performance comparison on the IXI dataset. For FG-CPAB, i indicates the number of composed transformations.
<table><tr><td>Model</td><td>Dice</td><td> $| J | \leq 0 ( \% )$ </td></tr><tr><td>Affine</td><td> $0 . 3 8 6 \pm 0 . 1 9 5$ </td><td></td></tr><tr><td>SyN</td><td> $0 . 6 4 5 \pm 0 . 1 5 2$ </td><td> $< 0 . 0 0 0 1$ </td></tr><tr><td>NiftyReg</td><td> $0 . 6 4 5 \pm 0 . 1 6 7$ </td><td> $0 . 0 2 0 \pm 0 . 0 4 6$ </td></tr><tr><td>LDDMM</td><td> $0 . 6 8 0 \pm 0 . 1 3 5$ </td><td> $< 0 . 0 0 0 1$ </td></tr><tr><td>VoxelMorph</td><td> $0 . 7 3 2 \pm 0 . 1 2 3$ </td><td> $1 . 5 2 2 \pm 0 . 3 3 6$ </td></tr><tr><td>CycleMorph</td><td> $0 . 7 3 7 \pm 0 . 1 2 3$ </td><td> $1 . 7 1 9 \pm 0 . 3 8 2$ </td></tr><tr><td>MIDIR</td><td> $0 . 7 4 2 \pm 0 . 1 2 8$ </td><td> $< 0 . 0 0 0 1$ </td></tr><tr><td>ViT-V-Net</td><td> $0 . 7 3 4 \pm 0 . 1 2 4$ </td><td> $1 . 6 0 9 \pm 0 . 3 1 9$ </td></tr><tr><td>CoTr</td><td> $0 . 7 3 5 \pm 0 . 1 3 5$ </td><td> $1 . 2 9 8 \pm 0 . 3 4 3$ </td></tr><tr><td>PVT</td><td> $0 . 7 2 7 \pm 0 . 1 2 8$ </td><td> $1 . 8 5 8 \pm 0 . 3 1 4$ </td></tr><tr><td>nnFormer</td><td> $0 . 7 4 7 \pm 0 . 1 3 5$ </td><td> $1 . 5 9 5 \pm 0 . 3 5 8$ </td></tr><tr><td>TransMorph</td><td> $0 . 7 5 3 \pm 0 . 1 2 3$ </td><td> $1 . 5 7 9 \pm 0 . 3 2 8$ </td></tr><tr><td>ScaMorph</td><td> $0 . 7 5 5 \pm 0 . 1 3 2$ </td><td> $1 . 4 0 0 \pm 0 . 3 2 9$ </td></tr><tr><td>FG-CPAB</td><td> $0 . 7 6 0 \pm 0 . 0 2 3$ </td><td>&lt; 0.0001</td></tr><tr><td>FG-CPAB (i = 2)</td><td> $0 . 7 6 2 \pm 0 . 0 2 4$ </td><td>&lt; 0.0001</td></tr><tr><td>FG-CPAB (i = 3)</td><td> $0 . 7 6 3 \pm 0 . 0 2 4$ </td><td>&lt; 0.0001</td></tr><tr><td>FG-CPAB (i = 4)</td><td> $\mathbf { 0 . 7 6 5 \pm 0 . 0 2 4 }$ </td><td> $< 0 . 0 0 0 1$ </td></tr><tr><td></td><td></td><td></td></tr></table>

Figure 3: Dice score vs. tessellation size on IXI. At tessellation 30 we have $5 \times 3 0 ^ { 3 } =$ 135, 000 unique cells.  
![](images/50d2eaa367da907a9424322992c32df5e92160e8dd21b17e9f4496db1736e220.jpg)

## 4.5 CPAB as a low-dimensional geometry code for neural operator learning

DIMON [39] learns geometry-dependent PDE operators by transporting inputs and solutions from varying domains $\{ \Omega ^ { \theta } \}$ to a reference domain $\Omega ^ { 0 }$ , and conditioning the operator on a low-dimensional geometry code. In the original DIMON pipeline, this code is obtained by compressing LDDMM high-dimensional deformation representation with principal component analysis (PCA).

We consider the same reaction–diffusion setting, but replace the geometry encoding with CPAB. For each target domain $\Omega ^ { \theta } .$ , we fit a CPAB diffeomorphism $\Phi _ { \theta } : \Omega ^ { \breve { 0 } } \to \Omega ^ { \theta }$ and use the corresponding parameter vector $\pmb \theta \in \mathbb { R } ^ { d }$ directly as the geometry code. Unlike PCA-based LDDMM encodings, CPAB provides a native low-dimensional diffeomorphic parameterization and therefore removes the need for a separate compression step. We match the CPAB parameter count to that of the PCA encoding.

This has two advantages: the code is directly tied to the transport map, and its dimension is controlled by the tessellation, yielding an interpretable expressiveness–complexity trade-off. Table 4 shows that FG-CPAB attains the lowest test relative $L ^ { 2 }$ error under the same operator backbone, suggesting that compact diffeomorphic parameterizations can serve as effective geometry descriptors for neural operator learning. We defer implementation details to Appendix E.2.

<table><tr><td>Experiment</td><td>Geometry model</td><td>Test rel.  $L ^ { 2 }$  mean</td><td>Test rel.  $L ^ { 2 }$  std</td></tr><tr><td>FG-CPAB</td><td>FG-CPAB parameterization</td><td>0.052811</td><td>0.022436</td></tr><tr><td>LDDMM deformation PCA</td><td>PCA deformation code</td><td>0.082821</td><td>0.033168</td></tr></table>

Table 4: Reaction–diffusion operator-learning results using different geometry encodings. Lower is better for both metrics.

## 4.6 CPAB composition for non-stationary velocity fields

Table 3 shows that FG-CPAB already performs competitively with, and in our setting better than, LD-DMM. However, the two models differ in expressiveness: a single CPAB transformation is generated by a stationary CPA velocity field, whereas LDDMM can represent more general deformations through time-varying velocity fields. Therefore, there exist orientation-preserving diffeomorphisms that can not be represented (or even well approximated) by a single CPAB diffeomorphism. Fortunately, the efficiency of our implementation also makes it practical to compose m CPAB transformations,

$$
\Phi = \Phi _ { m } \circ \cdots \circ \Phi _ { 1 } .\tag{13}
$$

Although such a composition is not necessarily itself a CPAB transformation (CPAB diffeomorphisms form a subset, but not a subgroup, of the diffeomorphism group and are not closed under composition [13]), it remains a diffeomorphism, since diffeomorphisms are closed under composition. This provides a practical way to increase expressiveness while retaining CPAB’s efficiency at each stage.

Figure 4 illustrates this point on a synthetic example with two blobs following crossing trajectories. A single CPAB transformation cannot model this motion accurately and leaves a visible residual error. In contrast, a composition of CPAB transformations successfully captures the deformation by distributing the motion across multiple stages. Thus, CPAB composition provides a simple and efficient way to approximate deformations that would otherwise require a non-stationary velocity field, bringing CPAB closer in spirit to the flexibility of LDDMM, but at a much lower cost.

## 4.7 Ablations and hyper-parameter analysis

Basis sparsity. We first ablate the sparse basis construction by comparing sparse and dense representations of the CPA basis. The sparse formulation yields substantial memory savings, with additional runtime benefits that become more pronounced at larger tessellation sizes. This shows that sparsity is an important complement to gradient factorization for making high-resolution CPAB practical. Full results are provided in Appendix B.2.

Tessellation resolution. We next study tessellation resolution, the main expressiveness knob of CPAB. Performance improves as the tessellation becomes finer, both on IXI (Figure 3) and in the synthetic setting, showing that the additional local flexibility is useful in practice. This result also highlights the importance of our efficient backward pass, which makes these larger tessellations practical to optimize. Full results are provided in Appendix B.1.

![](images/66793664d2eb6c5f53cb3f07e7924c5c900f6a045e355e597a4834cf1a778fe1.jpg)  
Figure 4: A composition of CPAB transformations can model deformations that a single stationary CPAB cannot. From left to right: source image, target image, absolute error for a single CPAB transformation, and absolute error for a composition of CPAB transformations. The target requires the two blobs to follow crossing trajectories, which cannot be captured well by a single stationary velocity field. While a single CPAB leaves substantial residual error, the composed model greatly reduces the error by realizing the deformation through multiple diffeomorphic stages.

Solver step count. We also study the number of ODE solver steps used for CPAB integration. All results in the paper use 50 CPAB solver steps unless noted otherwise. Accuracy improves rapidly at low step counts and then largely plateaus around 50 steps, while runtime continues to increase. This supports using 50 steps as a robust default, consistent with previous CPAB implementations. Full results are provided in Appendix B.3.

## 5 Discussion

Our results show that FG-CPAB makes CPAB substantially more practical as a diffeomorphic primitive for modern spatial or temporal modeling. By decoupling trajectory integration from parameter-space projection, the proposed formulation removes the main computational bottleneck of standard CPAB and enables efficient optimization with much finer tessellations in both 2D and 3D. In turn, this improved scalability translates into better practical expressiveness, competitive registration accuracy, and strong topology preservation.

While adjoint methods are standard for general ODEs [9], the linear CPA basis permits an exact factorization into cell-local moments that prior general-purpose adjoint solvers cannot exploit. The key insight is the reorganization of sensitivities into parameter-independent geometric statistics that are later projected through the sparse CPA basis. This delivers the $\mathcal { \bar { O } } ( T N + \mathcal { \bar { d } } C )$ complexity and the dramatic practical speedups reported here.

These efficiency gains, however, do not eliminate the dimensionality limitations of tessellation-based models. Although the factorization extends directly to any spatial dimension K, with $K \times ( K + 1 )$ moments and parameter dependence confined to the final projection, the number of cells grows exponentially with K. We therefore restrict our empirical claims to spatial dimensions $\bar { K \ } \leq 3$ Extending the approach to higher-dimensional domains would likely require additional structure, such as separable, low-rank, or adaptively refined velocity fields.

Overall, this work shifts CPAB from a promising but difficult-to-scale formulation toward a useful, optimization-friendly tool for large-scale diffeomorphic learning in both classical optimization and modern deep-learning pipelines.

## References

[1] Amit Aflalo and Oren Freifeld. A regularity condition under which a cpab transformation is a diffeomorphism, 2026. URL https://arxiv.org/abs/2610.32369. 2, 4

[2] John Ashburner. A fast diffeomorphic image registration algorithm. Neuroimage, 38(1):95–113, 2007. 2, 3

[3] Guha Balakrishnan, Amy Zhao, Mert R Sabuncu, John Guttag, and Adrian V Dalca. Voxelmorph: a learning framework for deformable medical image registration. IEEE transactions on medical imaging, 38(8):1788–1800, 2019. 7, 30

[4] M Faisal Beg, Michael I Miller, Alain Trouvé, and Laurent Younes. Computing large deformation metric mappings via geodesic flows of diffeomorphisms. International journal ofcomputer vision, 61(2):139–157, 2005. 2, 3, 30

[5] Jian Bi, Qianliang Wu, Jianjun Qian, Lei Luo, and Jian Yang. Shaping without tearing: Controllable diffeomorphic deformations for topology-preserving 3d point cloud augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 19711– 19719, 2026. 2

[6] Mathieu Carrière, Marc Theveneau, and Théo Lacombe. Diffeomorphic interpolation for efficient persistence-based topological optimization. Advances in Neural Information Processing Systems, 37:27274–27294, 2024. 2

[7] Irit Chelly, Shahaf E Finder, Shira Ifergane, and Oren Freifeld. Trainable highly-expressive activation functions. In European Conference on Computer Vision, 2024. 2

[8] Junyu Chen, Eric C Frey, Yufan He, William P Segars, Ye Li, and Yong Du. Transmorph: Transformer for unsupervised medical image registration. Medical image analysis, 82:102615, 2022. 2, 23

[9] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018. 4, 5, 10

[10] Nicki S. Detlefsen. libcpab. https://github.com/SkafteNicki/libcpab, 2018. 1, 2, 4, 5, 6, 14, 28, 30

[11] Nicki Skafte Detlefsen, Oren Freifeld, and Søren Hauberg. Deep diffeomorphic transformer networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4403–4412, 2018. 2

[12] Oren Freifeld. Deriving the cpab derivative. Technical report, 2018. 4

[13] Oren Freifeld, Soren Hauberg, Kayhan Batmanghelich, and John W Fisher. Highly-expressive spaces of well-behaved transformations: Keeping it simple. In Proceedings of the IEEE International Conference on Computer Vision, pages 2911–2919, 2015. 1, 2, 3, 4, 5, 9, 29

[14] Oren Freifeld, Søren Hauberg, Kayhan Batmanghelich, and John W. Fisher. Transformations based on continuous piecewise-affine velocity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(12):2496–2509, 2017. doi: 10.1109/TPAMI.2016.2646685. 1, 2, 3, 4, 28, 29

[15] Amir Gholami, Kurt Keutzer, and George Biros. Anode: Unconditionally accurate memoryefficient gradients for neural odes. International Joint Conferences on Artificial Intelligence (IJCAI), 2019. 4, 5

[16] Kunal Gupta and Manmohan Chandraker. Neural mesh flow: 3d manifold mesh generation via diffeomorphic flows. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1747–1758. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/file/1349b36b01e0e804a6c2909a6d0ec72a-Paper.pdf. 2

[17] Kun Han, Shanlin Sun, Thanh-Tung Le, Xiangyi Yan, Haoyu Ma, Chenyu You, and Xiaohui Xie. Hybrid neural diffeomorphic flow for shape representation and generation via triplane. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7707–7717, 2024. 2

[18] Adam Hartshorne, Allen Paul, Tony Shardlow, and Neill DF Campbell. Arc-flow: Articulated, resolution-agnostic, correspondence-free matching and interpolation of 3d shapes under flow fields. In 2025 International Conference on 3D Vision (3DV), pages 11–33. IEEE, 2025. 2

[19] Søren Hauberg, Oren Freifeld, Anders Boesen Lindbo Larsen, John Fisher, and Lars Hansen. Dreaming more data: Class-dependent distributions over diffeomorphisms for learned data augmentation. In Artificial intelligence and statistics, pages 342–350. PMLR, 2016. 2

[20] Malte Hoffmann, Benjamin Billot, Douglas N Greve, Juan Eugenio Iglesias, Bruce Fischl, and Adrian V Dalca. Synthmorph: learning contrast-invariant registration without acquired images. IEEE transactions on medical imaging, 41(3):543–558, 2021. 19

[21] IXI Project. IXI Dataset. https://brain-development.org/ixi-dataset/. Information eXtraction from Images, EPSRC Grant GR/S21533/02. 30

[22] Ilya Kaufman, Ron Shapira Weber, and Oren Freifeld. Cyclic diffeomorphic transformer nets for contour alignment. In 2021 IEEE International Conference on Image Processing (ICIP), pages 349–353, 2021. doi: 10.1109/ICIP42928.2021.9506570. 2

[23] Alaa Kryeem, Shmuel Raz, Dana Eluz, Dorit Itah, Hagit Hel-Or, and Ilan Shimshoni. Personal ized monitoring in home healthcare: An assistive system for post hip replacement rehabilitation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 1868–1877, October 2023. 2

[24] Brian Lee. torch-lddmm: Gpu and cpu implementation of lddmm image registration. https: //github.com/brianlee324/torch-lddmm. GitHub repository, accessed 2026-04-26. 7

[25] Yuchen Liu, Ling Wang, Xiaolin Ning, Yang Gao, and Defeng Wang. Enhancing unsupervised learning in medical image registration through scale-aware context aggregation. iScience, 28(2), 2025. 2

[26] Krishna S Mantri, Xinzhi Wang, Carola-Bibiane Schönlieb, Bruno Ribeiro, Beatrice Bevilacqua, and Moshe Eliasof. Digraf: Diffeomorphic graph-adaptive activation function. Advances in Neural Information Processing Systems, 37:3649–3681, 2024. 2

[27] Krishna Sri Ipsit Mantri, Carola-Bibiane Schönlieb, Bruno Ribeiro, Chaim Baskin, and Moshe Eliasof. Ditask: Multi-task fine-tuning with diffeomorphic transformations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25218–25229, 2025. 2

[28] Iñigo Martinez, Elisabeth Viles, and Igor G Olaizola. Closed-form diffeomorphic transformations for time series alignment. In International Conference on Machine Learning, pages 15122–15158. PMLR, 2022. 2, 14, 30

[29] Nour Neifar, Achraf Ben-Hamadou, Afef Mdhaffar, Mohamed Jmaiel, and Bernd Freisleben. Leveraging statistical shape priors in gan-based ecg synthesis. IEEE Access, 12:36002–36015, 2024. 2

[30] Daniel Rueckert, Luke I Sonoda, Carmel Hayes, Derek LG Hill, Martin O Leach, and David J Hawkes. Nonrigid registration using free-form deformations: application to breast mr images. IEEE transactions on medical imaging, 18(8):712–721, 2002. 19

[31] Pola Schwöbel, Frederik Rahbæk Warburg, Martin Jørgensen, Kristoffer Hougaard Madsen, and Søren Hauberg. Probabilistic spatial transformer networks. In Uncertainty in artificial intelligence, pages 1749–1759. PMLR, 2022. 2

[32] Guy Shacht, Dov Danon, Sharon Fogel, and Daniel Cohen-Or. Single pair cross-modality super resolution. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 6378–6387, 2021. 2

[33] Ron Shapira Weber, Shahar Ben Ishay, Andrey Lavrinenko, Shahaf E Finder, and Oren Freifeld. Timepoint: Accelerated time series alignment via self-supervised keypoint and descriptor learning. In International Conference on Machine Learning, 2025. 2

[34] Ron A Shapira Weber, Matan Eyal, Nicki Skafte, Oren Shriki, and Oren Freifeld. Diffeomorphic temporal alignment nets. Advances in neural information processing systems, 32, 2019. 2

[35] Nicki Skafte and Søren Hauberg. Explicit disentanglement of appearance and perspective in generative models. Advances in Neural Information Processing Systems, 32, 2019. 2

[36] Shanlin Sun, Kun Han, Deying Kong, Hao Tang, Xiangyi Yan, and Xiaohui Xie. Topologypreserving shape reconstruction and registration via neural diffeomorphic flow. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20845–20855, 2022. 2

[37] Hexiang Wang, Fengqi Liu, Qianyu Zhou, Ran Yi, Xin Tan, and Lizhuang Ma. Continuous piecewise-affine based motion model for image animation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 5427–5435, 2024. 2

[38] Ron Shapira Weber and Oren Freifeld. Regularization-free diffeomorphic temporal alignment nets. In International Conference on Machine Learning, pages 30794–30826. PMLR, 2023. 2

[39] Minglang Yin, Nicolas Charon, Ryan Brody, Lu Lu, Natalia Trayanova, and Mauro Maggioni. A scalable framework for learning the geometry-dependent solution operators of partial differential equations. Nature Computational Science, 4:928–940, 2024. doi: 10.1038/s43588-024-00732-2. 2, 9, 23, 30

[40] Miaomiao Zhang and P Thomas Fletcher. Finite-dimensional lie algebras for fast diffeomorphic image registration. In International conference on information processing in medical imaging, pages 249–260. Springer, 2015. 3

[41] Miaomiao Zhang and P Thomas Fletcher. Fast diffeomorphic image registration via fourierapproximated lie algebras. International Journal of Computer Vision, 127(1):61–73, 2019. 3

[42] Zhiwei Zhao, Changqing Liu, Yingguang Li, Zhibin Chen, and Xu Liu. Diffeomorphism neural operator for various domains and parameters of partial differential equations. Communications Physics, 8:15, 2025. doi: 10.1038/s42005-024-01911-3. 2

## A Performance

In this section, we evaluate the computational efficiency of our optimized CPAB implementation against the strongest available baselines. In 2D and 3D, we compare against the standard libcpab [10] implementation. In 1D, we additionally compare against DIFW [28], which to our knowledge is the fastest available implementation in the 1D setting, although it only supports 1D transformations. To ensure reliable measurements, all reported results represent the mean of 50 execution iterations, preceded by a 10-iteration warmup phase. All experiments were performed on a standardized workload of 125,000 points with a batch size of 16.

## A.1 Why the standard backward stops scaling

The failure of the standard implementation at fine tessellation is not an intrinsic limitation of CPAB, but a consequence of how the reference backward pass is organized on the GPU. In previous implementations, sensitivities are propagated separately for every parameter throughout the full integration trajectory. As tessellation becomes finer, the parameter dimension d grows, and the backward pass therefore scales directly with d.

The key hardware bottleneck is that the parameter axis is encoded directly in the backward-kernel launch geometry. As a result, increasing tessellation enlarges the kernel grid itself and eventually reaches a device-dependent launch ceiling; this is the launch-limit problem referred to in the main text. In practice, once tessellations become moderately fine, the reference implementation is governed by this parameter-dependent kernel structure rather than by the geometric integration alone.

FG-CPAB avoids this bottleneck by removing parameter dependence from the integration itself. The fused adjoint kernel is launched only over points and batch elements, accumulates parameterindependent cell-wise moments, and defers the dependence on d to a separate final projection step. This is the practical reason the optimized implementation continues to run at tessellation levels for which the standard backward becomes unusable.

(a) Forward  
![](images/64a697388f7d9e8faef87e913907ed5094f4006595c8a2c13417cd6cb119552c.jpg)

(b) Gradient computation  
![](images/473be0978409350963e9670b086e0cfe63c0d9cc00111045e26b89c2d05d0ab0.jpg)

(c) Memory  
![](images/e3bdfd0cde987368f1ba39587eba6ef1484227a5b39775a915771a2ade890266.jpg)  
Figure 5: Performance scaling with tessellation size $( N _ { \mathrm { t e s s } } ) .$ . Comparison of our optimized 3D CPAB vs. standard implementation on a single RTX A6000 GPU. (a) Forward pass latency. (b) Gradient computation latency. (c) Peak GPU memory allocation. Our approach yields up to a 60× speedup in the forward pass and a 4477× speedup in the gradient computation, while reducing peak memory by over 180×. Notably, the standard implementation does not support $N _ { \mathrm { t e s s } } > 5$ , whereas ours scales efficiently.

Tessellation size $\left( N _ { \mathrm { t e s s } } \right)$  
(a) Forward  
![](images/f0323b4be7891a240227a62a11eb492de89010e3c10640a36c29b12f1e6412a0.jpg)

$$
\left( N _ { \mathrm { t e s s } } \right)
$$

(b) Gradient computation  
![](images/5e7c681f59e1510d5d7eb374a41c8d2d347a7b887973fa7337b28510ee53c2d9.jpg)

(c) Memory  
![](images/8b64d7903a2749936b4e4af0096b224bb3a9cc0c200c5e235ee03c0f987945d0.jpg)  
Tessellation size $\left( N _ { \mathrm { t e s s } } \right)$  
Figure 6: Performance scaling with tessellation size $( N _ { \mathrm { t e s s } } ) .$ . Comparison of our optimized 1D CPAB vs. libcpab implementation on a single RTX A6000 GPU. (a) Forward pass latency. (b) Gradient computation latency. (c) Peak GPU memory allocation. Our approach yields up to a 7× speedup in the forward pass and a 301× speedup in the gradient computation, while reducing peak memory by 49×.

(a) Forward  
![](images/ab6c0e39693817e1e83516bd5adf4c777e46f5d444772f75d9b310990b459776.jpg)

(b) Gradient computation  
![](images/1bae08e1ff02ea621a5a23c0c0e8f69fbe2bfc9f77e8f17e9c6dd78ee59210c1.jpg)  
Tessellation size $( N _ { \mathrm { t e s s } } )$

(c) Memory  
![](images/ebba136fc3ea17d06fb1383fc973a1907341dc15d8cef70414187fccd76ddd65.jpg)  
Figure 7: Performance scaling with tessellation size $( N _ { \mathrm { t e s s } } ) .$ . Comparison of our optimized 1D CPAB vs. the DIFW implementation on a single RTX A6000 GPU. (a) Forward pass latency. (b) Gradient computation latency. (c) Peak GPU memory allocation. Our approach yields up to a 2× speedup in the forward pass and a 104× speedup in the gradient computation, while reducing peak memory by 196×.

## B Ablations and hyper-parameter analysis

## B.1 Tessellation resolution hyper-parameter analysis

We study the effect of tessellation resolution on CPAB performance, since tessellation size is the main factor controlling the trade-off between expressiveness and computational cost.

Setup. We vary the tessellation resolution while keeping the rest of the optimization protocol fixed. We evaluate this both on the IXI brain MRI registration benchmark (Table 3, Figure 3) and on the synthetic fitting task described in § C.1. The purpose of this ablation is to isolate how much of the final performance comes from access to finer tessellations.

Findings. Figure 8 shows a clear and consistent trend: increasing tessellation resolution improves alignment quality up to the largest resolutions tested. This confirms that finer CPAB parameterizations provide additional local flexibility that is useful for modeling complex deformations. At the same time, the results highlight a central practical limitation of previous CPAB implementations: without an efficient backward pass, these larger tessellations are too slow or memory-intensive to optimize reliably, especially in 3D.

Conclusion. The ablation demonstrates that the scalability enabled by gradient factorization is directly tied to model quality. In other words, our method is valuable not only because it makes CPAB faster, but because it enables operation in a more expressive tessellation regime that leads to better registration results.

(a) 2D  
![](images/d16070bafe6820c6c7eb3d6e6cb8ae6d59646371812e0217d5440e237d187596.jpg)

(b) 3D  
![](images/fd49fe2651b0fc0f50a9a22bf8969a942f1c3e81bea8bde530fe6b3bbb4a8039.jpg)  
Figure 8: Effect of tessellation resolution on registration performance. Increasing the tessellation resolution improves deformation quality by providing a more expressive piecewise-affine parameterization. This trend is observed across both real and synthetic registration settings, supporting the claim that the scalability unlocked by gradient factorization is important not only for efficiency, but also for final task performance.

## B.2 Sparse basis ablation

We compare sparse and dense representations of the CPA basis matrix B for 3D transformations on a $\mathrm { 1 2 8 ^ { 3 } }$ grid. As tessellation resolution increases, B grows quickly in both dimensions: the rows correspond to the stacked cell-wise affine coefficients, while the columns correspond to the free CPAB parameters. Since finer tessellations increase both the number of cells and the number of free parameters, a dense basis becomes increasingly expensive, especially in 3D.

Table 5: Sparse vs. Dense CPA basis across tessellation size. Forward/backward are mean runtime in ms (lower is better); peak is memory in MiB (lower is better).
<table><tr><td rowspan="2">Tess.</td><td colspan="3">Sparse basis</td><td colspan="3">Dense basis</td></tr><tr><td>Fwd (ms)</td><td>Bwd (ms)</td><td>Peak (MiB)</td><td>Fwd (ms)</td><td>Bwd (ms)</td><td>Peak (MiB)</td></tr><tr><td>2</td><td>10.8</td><td>51.9</td><td>384</td><td>10.9</td><td>51.9</td><td>384</td></tr><tr><td>3</td><td>11.3</td><td>54.0</td><td>384</td><td>11.3</td><td>54.1</td><td>384</td></tr><tr><td>4</td><td>11.5</td><td>54.9</td><td>385</td><td>11.4</td><td>55.0</td><td>385</td></tr><tr><td>5</td><td>11.6</td><td>55.6</td><td>385</td><td>11.6</td><td>55.7</td><td>385</td></tr><tr><td>6</td><td>11.7</td><td>56.1</td><td>386</td><td>11.7</td><td>56.3</td><td>386</td></tr><tr><td>7</td><td>11.9</td><td>57.0</td><td>388</td><td>12.0</td><td>57.1</td><td>388</td></tr><tr><td>8</td><td>12.2</td><td>57.4</td><td>390</td><td>12.5</td><td>57.7</td><td>390</td></tr><tr><td>9</td><td>12.6</td><td>58.0</td><td>392</td><td>13.0</td><td>58.4</td><td>392</td></tr><tr><td>10</td><td>13.8</td><td>59.6</td><td>395</td><td>14.5</td><td>60.3</td><td>501</td></tr><tr><td>11</td><td>14.4</td><td>59.9</td><td>399</td><td>15.8</td><td>61.5</td><td>914</td></tr><tr><td>12</td><td>15.4</td><td>62.8</td><td>403</td><td>17.8</td><td>65.1</td><td>1580</td></tr><tr><td>13</td><td>15.8</td><td>62.5</td><td>408</td><td>19.7</td><td>65.7</td><td>2607</td></tr><tr><td>14</td><td>16.9</td><td>66.2</td><td>414</td><td>23.4</td><td>71.9</td><td>4140</td></tr><tr><td>15</td><td>17.2</td><td>65.1</td><td>423</td><td>26.9</td><td>74.1</td><td>6360</td></tr><tr><td>16</td><td>19.5</td><td>72.6</td><td>432</td><td>34.4</td><td>86.2</td><td>9492</td></tr><tr><td>17</td><td>18.4</td><td>67.6</td><td>438</td><td>41.1</td><td>87.7</td><td>13818</td></tr><tr><td>18</td><td>20.5</td><td>75.4</td><td>450</td><td>52.0</td><td>104.8</td><td>19674</td></tr><tr><td>19</td><td>20.5</td><td>72.1</td><td>462</td><td>64.3</td><td>113.4</td><td>27467</td></tr><tr><td>20</td><td>23.5</td><td>83.4</td><td>474</td><td>85.0</td><td>140.3</td><td>37678</td></tr></table>

Table 5 shows that the sparse basis mainly improves memory usage. At small tessellations, sparse and dense implementations behave similarly, but at tessellation size 20, peak memory drops from 37678 MiB to 474 MiB. Runtime also improves, though more modestly: forward latency decreases from 85.0 ms to 23.5 ms and backward latency from 140.3 ms to 83.4 ms.

This trend is expected. The main runtime speedup in our method comes from gradient factorization, which removes parameter-dependent integration from the backward pass. In contrast, sparsifying B mainly reduces the cost of storing and applying the final basis projection. Its primary effect is therefore memory saving, with a secondary but smaller effect on runtime.

## B.3 Solver step hyper-parameter analysis

We study how the number of ODE solver steps affects the trade-off between runtime and registration accuracy. Unless otherwise noted, all results in the paper use 50 CPAB solver steps, matching the standard setting used in prior CPAB implementations.

Setup. We use the same synthetic registration protocol as in § C.1 and vary only the number of CPAB solver steps. The sweep is performed for 2D transformations with a 30 × 30 tessellation, so the goal is to isolate the effect of integration accuracy on optimization behavior without changing the rest of the training setup.

Findings. Table 6 shows a clear runtime-accuracy trade-off. Very small step counts are fastest per iteration, but they incur substantially worse registration error. Increasing the number of solver steps from 5 to 20 reduces the mean MSE from $\phantom { - } 1 . 4 2 \phantom { - } \times 1 0 ^ { - 2 } \phantom { - } \mathrm { t o } 1 . 0 4 \times 1 0 ^ { - 4 }$ , i.e., by more than two orders of magnitude. Beyond 20 steps, however, the curve largely plateaus. The best mean MSE in the full sweep is $8 . 3 0 \times \dot { 1 0 } ^ { - 5 }$ at 65 steps, while 50 steps already achieves $8 . 8 9 \times 1 0 ^ { - 5 }$ . Extending the sweep to 100 steps yields $8 . 3 1 \times 1 0 ^ { - 5 }$ , essentially the same accuracy, but increases mean fitting time from 0.287 s to 0.468 s and backward latency from 0.415 ms to 0.720 ms.

Conclusion. These results suggest that the paper’s default choice of 50 solver steps is a robust operating point: it lies in the accuracy plateau, captures most of the attainable improvement, and avoids the noticeably higher cost of longer integrations.

<table><tr><td> $n _ { \mathrm { s o l v e r } }$ </td><td>Forward (ms) ↓</td><td>Backward (ms) ↓</td><td>Fit time (s) ↓</td><td>Registration MSE ↓</td></tr><tr><td>5</td><td>0.1054</td><td>0.1822</td><td>0.2559</td><td> $1 . 4 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>10</td><td>0.1068</td><td>0.1792</td><td>0.2490</td><td> $2 . 2 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>15</td><td>0.1080</td><td>0.2080</td><td>0.2486</td><td> $3 . 2 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>20</td><td>0.1138</td><td>0.2279</td><td>0.2438</td><td> $1 . 0 4 \times { { 1 0 } ^ { - 4 } }$ </td></tr><tr><td>25</td><td>0.1206</td><td>0.2523</td><td>0.2482</td><td> $9 . 4 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>30</td><td>0.1248</td><td>0.2910</td><td>0.2483</td><td> $9 . 1 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>35</td><td>0.1310</td><td>0.3201</td><td>0.2482</td><td> $9 . 0 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>40</td><td>0.1376</td><td>0.3649</td><td>0.2558</td><td> $9 . 7 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>45</td><td>0.1441</td><td>0.3805</td><td>0.2686</td><td> $8 . 7 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>50</td><td>0.1527</td><td>0.4145</td><td>0.2869</td><td> $8 . 8 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>55</td><td>0.1584</td><td>0.4497</td><td>0.3066</td><td> $9 . 0 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>60</td><td>0.1659</td><td>0.4748</td><td>0.3220</td><td> $8 . 6 0 \times { { 1 0 } ^ { - 5 } }$ </td></tr><tr><td>65</td><td>0.1730</td><td>0.5147</td><td>0.3433</td><td> $8 . 3 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>70</td><td>0.1765</td><td>0.5392</td><td>0.3599</td><td> $8 . 4 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>75</td><td>0.1838</td><td>0.5673</td><td>0.3762</td><td> $8 . 6 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>80</td><td>0.1901</td><td>0.6109</td><td>0.4003</td><td> $8 . 4 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>85</td><td>0.1960</td><td>0.6306</td><td>0.4134</td><td> $8 . 4 7 \times { { 1 0 } ^ { - 5 } }$ </td></tr><tr><td>90</td><td>0.2011</td><td>0.6623</td><td>0.4319</td><td> $8 . 7 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>95</td><td>0.2082</td><td>0.6966</td><td>0.4520</td><td> $8 . 8 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>100</td><td>0.2115</td><td>0.7197</td><td>0.4678</td><td> $8 . 3 1 \times 1 0 ^ { - 5 }$ </td></tr></table>

Table 6: Full CPAB ODE solver-step sweep over $n _ { \mathrm { s o l v e r } } \in \{ 5 , 1 0 , \dots , 1 0 0 \}$ . Lower values are better. Accuracy improves dramatically from the very low-step regime and then plateaus, while runtime continues to increase with the number of solver steps.

## C Implementation Details

## C.1 Optimization Stability and Practical Expressiveness Compared with Dense SVFs

Data and deformation generation. We follow a simulation-based registration protocol where correspondences are known by construction [20]. We evaluate in both 2D and 3D with spatial sizes $5 1 2 \times 5 1 2$ and $1 2 8 \times 1 2 8 \times 1 2 8$ , respectively. For each case, we synthesize a moving image as a sum of 9 random Gaussian blobs (blob standard deviations sampled uniformly from [0.03, 0.12]), add Gaussian intensity noise with standard deviation 0.01, and normalize intensities to [0, 1]. We then sample a smooth random deformation from a coarse control lattice and upsample it to full resolution (FFD-like deformation model [30]), using control sizes $8 \times 8$ in 2D and $6 \times 6 \times 6$ in 3D with deformation amplitude 0.03. The fixed image is obtained by warping the moving image with this sampled transformation, yielding synthetic moving/fixed pairs with exact ground-truth deformation fields.

Model configuration. CPAB uses tessellations of size $1 5 \times 1 5$ in 2D and $1 5 \times 1 5 \times 1 5$ in 3D. The dense SVF baseline is parameterized as a full-resolution velocity field and integrated using scalingand-squaring with 7 squaring steps. This yields a comparison between a compact low-dimensional diffeomorphic parameterization and a standard dense voxel-wise one.

Optimization protocol. Both methods are optimized against the same reconstruction objective using Adam. For the fixed-budget comparison reported in the main paper, each method is run for 1000 iterations with 5 restarts per target registration. Per dimension, results are aggregated over 100 independently generated target registrations. This fixed-budget setting is intended to compare practical optimization behavior and time-to-accuracy rather than asymptotic best-case fitting after extensive tuning.

Metrics. We report the final mean squared error (MSE) between the recovered image and the target, wall-clock fitting time, and the fraction of sampled spatial locations with non-positive Jacobian determinant after transformation. The Jacobian statistic is included to quantify topology violations.

Fixed-budget behavior. Under this matched optimization budget, FG-CPAB consistently achieves lower residual error, shorter fitting times, and fewer topology violations than dense SVF (Table 2). One possible explanation is the compact CPAB parameterization, which restricts optimization to the low-dimensional space spanned by B. In contrast, dense SVF involves a substantially larger number of free parameters. These results suggest that the reduced parameterization may facilitate more efficient optimization while retaining sufficient expressiveness for the evaluated registration tasks.

Extended-budget and regularized dense SVF. To separate optimization effects from asymptotic fitting behavior, we also evaluate dense SVF under much larger iteration budgets, both with and without explicit smoothness regularization. Figure 9 shows that unregularized dense SVF typically plateaus early at a relatively high error, whereas CPAB rapidly reaches a low-error solution. We then equip dense SVF with standard smoothness regularization and optimize for 100,000 iterations (Figure 10). Regularization substantially improves the optimization stability of dense SVF, but convergence remains much slower than for CPAB.

In 2D, regularized dense SVF still does not match CPAB’s reconstruction quality within the tested budget. In 3D, regularized dense SVF eventually attains a lower final MSE, but only after approximately 50,000–60,000 iterations. Since CPAB is also faster per iteration, this crossover occurs only after a very large wall-clock time budget.

Interpretation. These experiments indicate that dense SVFs may offer greater asymptotic fitting flexibility in some regimes, but that this flexibility is difficult to realize in practice without strong regularization and a substantially extended optimization budget. CPAB occupies a different and practically attractive point in the trade-off: it provides sufficient diffeomorphic expressiveness for complex deformations while yielding substantially better optimization stability, faster time-to-solution, and fewer topology violations out of the box.

![](images/c57d765f7defc9647ec4f1526bf7f32246cf6b76e0c31769db6c9bf8a3ffde34.jpg)

![](images/7725492bbc44fe8d80f45da8d76b77ca9b57cdc4178b2fb0892471f60a821399.jpg)  
Figure 9: Convergence without regularization. Optimization trajectories for CPAB and dense SVF on the synthetic fitting task without additional regularization, shown for both 2D and 3D settings. In both cases, CPAB rapidly reaches a low reconstruction error within the first few hundred iterations, whereas dense SVF plateaus early at a substantially higher MSE. This highlights the optimization difficulty of the dense SVF parameterization under a matched iteration budget, despite its much larger number of degrees of freedom.

(a) 2D  
![](images/4849c0576e2a7d0fdef634d98c7d6091f09ed4f724ca6901d37fcd8fc083a4c9.jpg)

(b) 3D  
![](images/d395a8c7bac372ee86540d5536f0b9d779c70a5ac1ab57c300c3b772b1815bbe.jpg)  
Figure 10: Convergence with regularization. Optimization trajectories for CPAB and dense SVF when dense SVF is equipped with standard smoothness regularization, shown for both 2D and 3D settings over an extended budget of 100,000 iterations. CPAB demonstrates extremely rapid initial convergence in both settings, stabilizing within the first few hundred iterations. In 2D (a), CPAB achieves and maintains a consistently lower fitting error. In 3D (b), regularized dense SVF converges much more slowly, but eventually surpasses CPAB to achieve a lower final mean squared error after approximately 50,000 to 60,000 iterations.

## D Segmented Checkpointing

Segmented checkpointing is the memory mechanism that makes the $\mathcal { O } ( T N )$ adjoint practical on GPU hardware. The reverse sweep requires access to the forward states $\pmb { p } ^ { ( 0 ) } , \ldots , \pmb { p } ^ { ( T ) }$ , because both the adjoint update and the moment contribution at timestep t depend on the state and active cell at that timestep. In our implementation these trajectories are thread-private. Storing the full trajectory for every point would therefore require $\mathcal { O } ( T )$ per-thread storage, increase register pressure, and, once registers are exhausted, spill to local memory, reducing occupancy and increasing memory traffic.

Segmented replay strategy. Let T denote the number of integration steps and S the segment length. We partition the trajectory into

$$
K = \left\lceil { \frac { T } { S } } \right\rceil
$$

segments, where segment k covers the timestep range $[ k S , \operatorname* { m i n } ( ( k + 1 ) S , T ) )$ ). The method then proceeds in two phases.

In the first phase, we integrate forward once and store only the $K + 1$ states at segment boundaries. In the second phase, we process segments in reverse order. For each segment, we restart from its boundary checkpoint, replay the forward dynamics within that segment, and immediately sweep backward through the reconstructed states to propagate the adjoint and accumulate the moments. Once the segment has been processed, its temporary states are discarded.

Memory-compute trade-off. The resulting per-thread storage cost is

$$
{ \mathcal { O } } ( T / S + S ) ,
$$

consisting of $\mathcal { O } ( T / S )$ checkpoint states and $\mathcal O ( S )$ temporary states for the active segment. This replaces the $\mathcal { O } ( T )$ storage required by the full trajectory. The trade-off is a constant-factor recomputation overhead: each state is visited once in the checkpointing pass and once again during replay. Although this does not change the asymptotic complexity, the constant matters in practice because reduced register pressure and lower local-memory traffic can substantially improve GPU occupancy.

Relation to the adjoint factorization. Checkpointing changes only how intermediate forward states are materialized; it does not alter the reverse sweep itself. Within each segment, the adjoint recursion and moment accumulation are exactly those of Eq. 8 and Eq. 11. Consequently, segmentation changes memory usage and runtime, but not the factorized gradient in Eq. 12.

Choice of segment length. The segment length S interpolates between two extremes. When $S = 1$ every timestep is checkpointed: checkpoint count is maximal, but replay buffers are minimal. When $S \geq T$ , the method reduces to a single-segment replay (“no checkpointing” in Table 7): checkpoint overhead is minimal, but per-thread replay buffers are largest. The runtime trade-off is therefore typically non-monotonic. Small S increases checkpoint bookkeeping, whereas large S increases temporary state, register pressure, and local-memory traffic. Best performance is usually attained at an intermediate S; unless otherwise noted, we use ${ \dot { S } } = 4$ throughout the paper. Table 7 reports this trade-off empirically.

Table 7: Checkpoint segment size sweep. Forward/backward are mean runtime in ms (lower is better). Bold indicates the fastest value in each metric.
<table><tr><td>Ckpt seg.</td><td>Segments</td><td>Fwd (ms)</td><td>Bwd (ms)</td></tr><tr><td>2</td><td>25</td><td>0.898</td><td>4.161</td></tr><tr><td>3</td><td>17</td><td>0.885</td><td>3.977</td></tr><tr><td>4</td><td>13</td><td>0.882</td><td>3.860</td></tr><tr><td>5</td><td>10</td><td>0.885</td><td>3.905</td></tr><tr><td>6</td><td>9</td><td>0.891</td><td>3.899</td></tr><tr><td>7</td><td>8</td><td>0.889</td><td>4.302</td></tr><tr><td>8</td><td>7</td><td>0.893</td><td>4.644</td></tr><tr><td>9</td><td>6</td><td>0.889</td><td>4.699</td></tr><tr><td>10</td><td>5</td><td>0.887</td><td>5.056</td></tr><tr><td>11</td><td>5</td><td>0.891</td><td>4.905</td></tr><tr><td>12</td><td>5</td><td>0.887</td><td>5.073</td></tr><tr><td>13</td><td>4</td><td>0.887</td><td>5.246</td></tr><tr><td>14</td><td>4</td><td>0.885</td><td>5.198</td></tr><tr><td>15</td><td>4</td><td>0.892</td><td>5.284</td></tr><tr><td>16</td><td>4</td><td>0.885</td><td>5.683</td></tr><tr><td>17</td><td>3</td><td>0.893</td><td>5.976</td></tr><tr><td>18</td><td>3</td><td>0.905</td><td>5.868</td></tr><tr><td>19</td><td>3</td><td>0.901</td><td>5.828</td></tr><tr><td>20</td><td>3</td><td>0.898</td><td>5.923</td></tr><tr><td>21</td><td>3</td><td>0.897</td><td>5.956</td></tr><tr><td>22</td><td>3</td><td>0.897</td><td>6.080</td></tr><tr><td>23</td><td>3</td><td>0.900</td><td>6.231</td></tr><tr><td>24</td><td>3</td><td>0.898</td><td>6.468</td></tr><tr><td>25</td><td>2</td><td>0.897</td><td>6.773</td></tr><tr><td>50 (no ckpt)</td><td>1</td><td>0.899</td><td>7.339</td></tr></table>

## E Application-Specific Experimental Details

We provide additional experimental details for the two applications considered in the main paper: 3D deformable registration and neural operator learning. These include the datasets, model configurations, and optimization protocols.

## E.1 CPAB for 3D deformable registration

Table 3 is produced using an atlas-to-patient registration protocol on the IXI dataset.

Data and preprocessing. We follow the public TransMorph IXI atlas-to-patient protocol [8]. It contains 576 preprocessed T1-weighted MRI volumes, divided into 403 training, 58 validation, and 115 test subjects. Each volume is skull-stripped, affinely aligned, segmented using FreeSurfer, normalized to [0, 1], and represented at a resolution of 160 × 192 × 224. Because FG-CPAB requires no offline training, we evaluate it directly on all 115 test subjects.

The fixed image is the common atlas distributed with the TransMorph IXI data, originally obtained from the CycleMorph protocol, together with its corresponding segmentation. Registration is optimized from the atlas to each patient using MRI intensities; anatomical labels are used only for evaluation. Dice is evaluated over the 30 anatomical labels used by the TransMorph protocol.

Model configuration and optimization. In all CPAB configurations, we use a 20 × 20 × 20 tessellation. For the composed variants, multiple CPAB transformations are applied sequentially during each optimization step. The parameters are fitted independently for every test subject using Adam for 200 iterations.

Loss function. The objective function comprises a local normalized cross-correlation (LNCC) similarity loss (window size 9, sample stride 2, weight 1.0) and a diffusion regularization penalty (L2 gradient of the displacement field, weight 3.0).

## E.2 CPAB as a low-dimensional geometry code for operator learning

We follow the reaction–diffusion benchmark of DIMON [39], where solutions on a family of annuluslike domains are transported to a reference circular annulus and a neural operator is conditioned on a low-dimensional geometry code. We keep the DIMON data generation, transport pipeline, and operator architecture unchanged, and vary only the geometry representation.

In the DIMON baseline, geometry is encoded by applying PCA to an LDDMM-based deformation representation. In our variant, we fit a CPAB diffeomorphism from the reference domain to each target domain and use the CPAB parameters directly as the geometry code. We use a 6 × 6 CPAB tessellation, yielding 122 parameters. For a fair comparison, we also set the DIMON baseline to 122 PCA coefficients. Hence both methods use geometry codes of identical dimension, and differences in performance reflect the quality of the underlying geometry parameterization rather than code size.

## F Nodal Sparse Basis

The basis B introduced in § 2 can be constructed efficiently through a nodal parameterization of the CPA velocity field. Instead of working directly with cell-wise affine coefficients, we first represent the velocity field by its values at the tessellation vertices and then impose the CPAB constraints in this nodal space. This yields a sparse construction of a basis spanning the same space as the constraint-based construction.

Nodal parameterization. Let $n _ { V }$ denote the number of vertices in the tessellation, and let ${ \textbf { \em u } } \in$ $\mathbb { R } ^ { K n _ { V } }$ be the stacked vector of nodal velocities, with one K-dimensional velocity vector per vertex. Let $\pmb { a } \in \mathbb { R } ^ { D }$ denote the stacked vector of all cell-wise affine coefficients, where D is the total number of affine coefficients across all cells. Since the velocity field is affine inside each simplex, the affine coefficients on a cell are uniquely determined by the velocities at that cell’s $K + 1$ vertices. Therefore, the global affine parameter vector can be written as

$$
\begin{array} { r } { \pmb { a } = \mathbf { N } \pmb { u } , \qquad \mathbf { N } \in \mathbb { R } ^ { D \times K n _ { V } } , } \end{array}\tag{14}
$$

where N is a sparse linear map from nodal velocities to cell-wise affine parameters.

Construction of the nodal map. The matrix N is assembled cell by cell from the mesh geometry. For a simplex c, let

$$
\widetilde { \pmb { x } } _ { c , 0 } , \hdots , \widetilde { \pmb { x } } _ { c , K } \in \mathbb { R } ^ { K + 1 }
$$

denote its vertices in homogeneous coordinates, and let

$$
\pmb { v } _ { c , 0 } , \hdots , \pmb { v } _ { c , K } \in \mathbb { R } ^ { K }
$$

be the corresponding nodal velocities. Inside the simplex, the CPA velocity field is represented as

$$
{ \pmb v } _ { c } ( { \pmb x } ) = { \bf A } _ { c } { \widetilde { \pmb x } } , \qquad { \bf A } _ { c } \in \mathbb { R } ^ { K \times ( K + 1 ) } .
$$

The local affine matrix $\mathbf { A } _ { c }$ is determined by the interpolation conditions

$$
\mathbf { A } _ { c } \widetilde { \pmb { x } } _ { c , j } = \pmb { v } _ { c , j } , \qquad j = 0 , \dots , K .
$$

Collecting the homogeneous vertex coordinates and nodal velocities into

$$
\mathbf { X } _ { c } = [ \widetilde { \pmb { x } } _ { c , 0 } \quad \cdots \quad \widetilde { \pmb { x } } _ { c , K } ] \in \mathbb { R } ^ { ( K + 1 ) \times ( K + 1 ) } , \qquad \mathbf { V } _ { c } = [ { \pmb { v } } _ { c , 0 } \quad \cdots \quad { \pmb { v } } _ { c , K } ] \in \mathbb { R } ^ { K \times ( K + 1 ) } ,
$$

gives

$$
\mathbf { A } _ { c } \mathbf { X } _ { c } = \mathbf { V } _ { c } .
$$

Since the simplex vertices are affinely independent, $\mathbf { X } _ { c }$ is invertible, and thus

$$
\mathbf { A } _ { c } = \mathbf { V } _ { c } \mathbf { X } _ { c } ^ { - 1 } .
$$

This is the local interpolation rule in matrix form. To express it as a linear operator on stacked vectors, we apply the standard identity

$$
\mathrm { v e c } ( \mathbf { U M } ) = ( \mathbf { M } ^ { \top } \otimes \mathbf { I } ) \mathrm { v e c } ( \mathbf { U } ) ,
$$

which yields

$$
\mathrm { v e c } \left( \mathbf { A } _ { c } \right) = ( \mathbf { X } _ { c } ^ { - T } \otimes \mathbf { I } _ { K } ) \mathrm { v e c } ( \mathbf { V } _ { c } ) .
$$

Hence the local nodal-to-affine operator is

$$
\mathbf { N } _ { c } = \mathbf { X } _ { c } ^ { - T } \otimes \mathbf { I } _ { K } .
$$

The global sparse matrix N is obtained by assembling these local operators over all cells according to the mesh connectivity. Its sparsity follows from locality: each cell depends only on its own $K + { \bar { 1 } }$ vertices, so each row of N contains nonzero entries only in the columns corresponding to those local nodal degrees of freedom.

Constraint enforcement in nodal coordinates. In the standard CPAB formulation, valid affine coefficients must satisfy the global linear constraint system

$$
\mathbf { L } \pmb { a } = \mathbf { 0 } ,
$$

which enforces continuity across shared cell boundaries together with any imposed boundary conditions. Substituting the nodal parameterization from Eq. 14 gives

$$
\mathbf { L N } u = \mathbf { 0 } .\tag{15}
$$

Defining

$$
\mathbf { C } : = \mathbf { L N } ,\tag{16}
$$

the admissible nodal configurations are exactly those in the null space of C. Let

$$
\mathbf { Z } \in \mathbb { R } ^ { K n _ { V } \times d }\tag{17}
$$

be a basis for null(C), so that every valid nodal vector can be written as

$$
\begin{array} { r } { { \pmb u } = { \bf Z } { \pmb \theta } , \qquad { \pmb \theta } \in \mathbb { R } ^ { d } . } \end{array}\tag{18}
$$

Resulting sparse CPA basis. Combining Eq. 14 and Eq. 18 yields

$$
\begin{array} { r } { { \pmb a } = { \bf N } { \bf Z } \theta . } \end{array}\tag{19}
$$

Hence the CPA basis can be written as

$$
\mathbf { B } = \mathbf { N } \mathbf { Z } .\tag{20}
$$

This shows that the low-dimensional parameter vector θ can be interpreted as coordinates in a constrained nodal basis, while B maps these coordinates to the cell-wise affine representation used throughout the paper.

The main advantage of this construction is sparsity: N is local by design, and the resulting basis inherits the mesh structure of the tessellation. Crucially, on a standard simplex mesh, the nodal formulation automatically satisfies continuity across shared cell boundaries. Consequently, the vast majority of constraints in C = LN evaluate exactly to zero. The only remaining active constraints are global boundary conditions (e.g., zero velocity at the domain boundaries). Finding the null space Z of this drastically reduced system geometrically amounts to identifying the unconstrained interior nodes, yielding a natural sparse basis matrix. In practice, simply thresholding numerical noise from standard dense linear algebra routines fully recovers this exact physical sparsity. This provides an efficient way to assemble B as a sparse matrix while exactly preserving the same globally continuous CPA velocity fields as the direct algebraic construction.

Remark. The null-space basis Z can often be constructed directly from the nodal constraints, avoiding an SVD. For more general constraints, one can remove the redundant rows of C before computing its null space numerically, substantially reducing the cost of the SVD. Although sparsity is not guaranteed in general, the resulting basis can remain sparse in practice for certain constraint structures, enabling efficient storage and computation.

## G Proof of the Telescoping Property of the Discrete Adjoint

We prove the claim made in § 3.2 that summing the sensitivity contraction identity Eq. 9 over all timesteps produces a telescoping series in which all propagated terms cancel exactly, leaving only the accumulated direct contributions.

For notational brevity, fix a single point ${ \pmb p } _ { n }$ (the argument is identical and independent for every point) and a single parameter component $\theta _ { i } .$ . Define the scalar quantities

$$
\begin{array} { r } { s ^ { ( t ) } : = \boldsymbol { \lambda } ^ { ( t ) ^ { \top } } \frac { \partial \pmb { p } ^ { ( t ) } } { \partial \theta _ { i } } , \qquad \boldsymbol { d } ^ { ( t ) } : = \boldsymbol { \lambda } ^ { ( t + 1 ) ^ { \top } } \left( h \mathbf { B } _ { i , c } \widetilde { \pmb { p } } _ { \mathrm { m i d } } ^ { ( t ) } + \frac { h ^ { 2 } } { 2 } \mathbf { A } _ { c , \mathrm { l i n } } \mathbf { B } _ { i , c } \widetilde { \pmb { p } } ^ { ( t ) } \right) , } \end{array}
$$

where all quantities on the right-hand side of $d ^ { ( t ) }$ are evaluated at the appropriate cell c that the point occupies at timestep t.

The identity Eq. 9 then reads

$$
s ^ { ( t + 1 ) } = d ^ { ( t ) } + s ^ { ( t ) } \qquad \mathrm { ~ f o r ~ e a c h { t i m e s t e p } ~ } t = 0 , 1 , \ldots , T - 1 .\tag{21}
$$

Summing Eq. 21 from $t = 0$ to $t = T - 1$ yields

$$
\sum _ { t = 0 } ^ { T - 1 } s ^ { ( t + 1 ) } = \sum _ { t = 0 } ^ { T - 1 } d ^ { ( t ) } + \sum _ { t = 0 } ^ { T - 1 } s ^ { ( t ) } .
$$

The left-hand side expands to

$$
\boldsymbol { s } ^ { ( 1 ) } + \boldsymbol { s } ^ { ( 2 ) } + \cdots + \boldsymbol { s } ^ { ( T ) } ,
$$

while the second sum on the right-hand side expands to

$$
\boldsymbol { s } ^ { ( 0 ) } + \boldsymbol { s } ^ { ( 1 ) } + \cdots + \boldsymbol { s } ^ { ( T - 1 ) } .
$$

Rearranging immediately gives the telescoping cancellation

$$
s ^ { ( T ) } - s ^ { ( 0 ) } = \sum _ { t = 0 } ^ { T - 1 } d ^ { ( t ) } .\tag{22}
$$

Two boundary conditions complete the proof:

1. The initial position ${ \pmb p } ^ { ( 0 ) } = { \pmb p } _ { 0 }$ is the fixed input point and therefore independent of the parameters θ. Consequently,

$$
\frac { \partial { \pmb p } ^ { ( 0 ) } } { \partial \theta _ { i } } = { \bf 0 } \quad \Longrightarrow \quad s ^ { ( 0 ) } = 0 .
$$

2. At the final timestep the adjoint is defined by the loss gradient,

$$
\boldsymbol { \lambda } ^ { ( T ) } = \frac { \partial \mathcal { L } } { \partial \pmb { p } ^ { ( T ) } } ,
$$

so

$$
s ^ { ( T ) } = \lambda ^ { ( T ) ^ { \top } } \frac { \partial \pmb { p } ^ { ( T ) } } { \partial \theta _ { i } } = \frac { \partial \mathcal { L } } { \partial \theta _ { i } } .
$$

Substituting these two facts into Eq. 22 produces

$$
\frac { \partial \mathcal { L } } { \partial \theta _ { i } } = \sum _ { t = 0 } ^ { T - 1 } d ^ { ( t ) } .
$$

The right-hand side is precisely the sum of all direct (local) contributions of $\theta _ { i }$ across the trajectory. These contributions are exactly the terms that, after applying the Frobenius inner-product identity and summing over all points n that visit cell c at timestep t, assemble into the moment matrix $\mathbf { M } _ { c }$ of Eq. 11. The final gradient is then recovered by the cheap projection Eq. 12.

Because the derivation relies only on the chain rule and the definition of the exact discrete adjoint of the midpoint integrator, the cancellation is algebraic and holds to machine precision (no discretization error is introduced).

## H Why Translation Does Not Enter the State Jacobian

For an arbitrary spatial dimension K, consider a fixed cell c and decompose the affine velocity matrix as

$$
\mathbf { A } _ { c } = [ \mathbf { L } _ { c } \ \mathbf { \delta } \mathbf { b } _ { c } ] , \qquad \mathbf { L } _ { c } \in \mathbb { R } ^ { K \times K } , \quad \mathbf { b } _ { c } \in \mathbb { R } ^ { K } .
$$

For homogeneous coordinates $\widetilde { \pmb { p } } = ( \pmb { p } ^ { \top } , 1 ) ^ { \top }$ , the velocity field is

$$
\mathbf { A } _ { c } \widetilde { \pmb { p } } = \mathbf { L } _ { c } \pmb { p } + \pmb { b } _ { c } .
$$

Starting from the midpoint integrator in Eq. 5,

$$
{ \pmb p } _ { \mathrm { m i d } } = \bigg ( { \bf I } + \frac { h } { 2 } { \bf L } _ { c } \bigg ) { \pmb p } ^ { ( t ) } + \frac { h } { 2 } { \pmb b } _ { c } ,
$$

and therefore

$$
\pmb { p } ^ { ( t + 1 ) } = \left( \mathbf { I } + h \mathbf { L } _ { c } + \frac { h ^ { 2 } } { 2 } \mathbf { L } _ { c } ^ { 2 } \right) \pmb { p } ^ { ( t ) } + \left( h \mathbf { I } + \frac { h ^ { 2 } } { 2 } \mathbf { L } _ { c } \right) \pmb { b } _ { c } .
$$

Hence, the state Jacobian of one midpoint step is

$$
\frac { \partial \pmb { p } ^ { ( t + 1 ) } } { \partial \pmb { p } ^ { ( t ) } } = \mathbf { I } + h \mathbf { L } _ { c } + \frac { h ^ { 2 } } { 2 } \mathbf { L } _ { c } ^ { 2 } ,
$$

where $\mathbf { I } \in \mathbb { R } ^ { K \times K }$ . Thus, the Jacobian depends only on the linear block $\mathbf { L } _ { c } = \mathbf { A } _ { c , \mathrm { l i n } }$ , regardless of the spatial dimension. The translation $b _ { c }$ contributes only additive terms independent of $\mathbf { \boldsymbol { p } } ^ { ( t ) }$ , whose derivatives w.r.t. $\mathbf { \boldsymbol { p } } ^ { ( t ) }$ vanish.

## I Integrator mismatch

Source of the mismatch and relation to prior CPAB implementations. As discussed in § 3, our forward pass uses the matrix-exponential update, whereas the gradient computation uses the discrete adjoint of the explicit midpoint scheme in Eq. 5. Consequently, the resulting gradient is the exact derivative of the midpoint-discretized dynamics, but only an approximation to the derivative of the matrix-exponential forward computation.

This numerical pairing is not introduced by FG-CPAB. Reference CPAB implementations similarly combine a matrix-exponential forward transformation with a midpoint-discretized sensitivity computation [14, 10]. We retain the same forward and gradient discretizations to ensure a fair comparison with prior CPAB methods. Thus, the comparison evaluates our factorized organization of the gradient computation without introducing a different integration scheme as an additional source of performance or accuracy differences.

Effect of inconsistent cell assignments. We first consider the discrepancy caused by using different cells for the same state. Let $\mathbf { \dot { A } } _ { 1 }$ and $\widetilde { \mathbf { A } } _ { 2 }$ denote the augmented matrices representing the velocity fields in two neighboring cells, and consider

$$
\widetilde { \pmb { p } } _ { 1 } ^ { ( t + 1 ) } = \exp ( h \widetilde { \mathbf { A } } _ { 1 } ) \widetilde { \pmb { p } } ^ { ( t ) } , \qquad \widetilde { \pmb { p } } _ { 2 } ^ { ( t + 1 ) } = \exp ( h \widetilde { \mathbf { A } } _ { 2 } ) \widetilde { \pmb { p } } ^ { ( t ) } .\tag{23}
$$

A Taylor expansion gives

$$
\widetilde { \pmb { p } } _ { 1 } ^ { ( t + 1 ) } - \widetilde { \pmb { p } } _ { 2 } ^ { ( t + 1 ) } = h \big ( \widetilde { \bf A } _ { 1 } - \widetilde { \bf A } _ { 2 } \big ) \widetilde { \pmb { p } } ^ { ( t ) } + \mathcal { O } ( h ^ { 2 } ) .\tag{24}
$$

Therefore, for fixed cell matrices and a bounded state $( \mathrm { i } . \mathrm { e } . , \| \widetilde { p } ^ { ( t ) } \|$ remains bounded as $h  0 )$ , an inconsistent cell choice can introduce an $O ( h )$ local discrepancy.

Effect of different integrators within a fixed cell. We next compare the matrix-exponential and midpoint updates when both use the same cell matrix $\widetilde { \bf A }$ . The midpoint update can be written as

$$
\begin{array} { r } { \widetilde { \pmb { p } } _ { \mathrm { m i d } } ^ { ( t + 1 ) } = \left( \mathbf { I } + h \widetilde { \mathbf { A } } + \frac { h ^ { 2 } } { 2 } \widetilde { \mathbf { A } } ^ { 2 } \right) \widetilde { \pmb { p } } ^ { ( t ) } . } \end{array}\tag{25}
$$

Meanwhile,

$$
\begin{array} { r } { \exp ( h \widetilde { \mathbf { A } } ) = \mathbf { I } + h \widetilde { \mathbf { A } } + \frac { h ^ { 2 } } { 2 } \widetilde { \mathbf { A } } ^ { 2 } + \mathcal { O } \left( h ^ { 3 } \| \widetilde { \mathbf { A } } \| ^ { 3 } \right) . } \end{array}\tag{26}
$$

Hence,

$$
\widetilde { \pmb { p } } _ { \mathrm { e x p } } ^ { ( t + 1 ) } - \widetilde { \pmb { p } } _ { \mathrm { m i d } } ^ { ( t + 1 ) } = \mathcal { O } \left( h ^ { 3 } \| \widetilde { \mathbf { A } } \| ^ { 3 } \right) .\tag{27}
$$

For a fixed cell matrix over a finite integration interval, the $\mathcal { O } ( h ^ { 3 } )$ local error yields an $\mathcal { O } ( h ^ { 2 } )$ global error. Thus, the local integrator mismatch is of higher order than the potential ${ \mathcal { O } } ( h )$ discrepancy caused by inconsistent cell assignments.

Empirical gradient validation. The remaining question is whether this approximation is sufficiently accurate for gradient-based optimization. We therefore compared the gradient produced by our backward pass with a central finite-difference estimate of the gradient of the matrix-exponential forward computation. We measured cosine similarity and relative error across multiple tessellation configurations.

For the standard setting used throughout the paper, with 50 solver steps, the cosine similarity was above 0.998, while the relative error was below 0.05. The approximate gradient is therefore closely aligned with the finite-difference reference gradient in the regime used in our experiments.

Conclusion. The forward and backward discretizations are not algebraically identical, and our gradient should accordingly be interpreted as an approximation to the derivative of the matrixexponential forward computation. However, the mismatch is higher order within a fixed cell, follows the numerical treatment used by prior CPAB implementations, and produces gradients that are strongly aligned with finite-difference estimates. Empirically, it does not hinder stable first-order optimization in any of our experiments.

## J Why Gradient-Based Optimization Is Needed: Comparison with MCMC

The original CPAB work considered MCMC-based optimization (Metropolis algorithm with a Gaussian proposal) for standalone registration problems, while its journal extension subsequently introduced gradient-based optimization [13, 14]. Although MCMC can be effective in low-dimensional settings, it is not well suited to the fine-tessellation applications targeted in this work for two main reasons.

End-to-end differentiability. CPAB was originally introduced as a standalone transformation model outside modern automatic-differentiation frameworks. In many recent applications, however, a CPAB transformation is used as a differentiable component within a larger model. Training such a model requires propagating loss gradients through the CPAB parameters to preceding network components. MCMC performs inference over the CPAB parameters but does not provide the vector– Jacobian products required for end-to-end backpropagation.

Scaling with the parameter dimension. Standard MCMC methods, such as Metropolis sampling with a Gaussian proposal, generally become increasingly difficult to apply as the parameter dimension grows. Early CPAB experiments using MCMC involved relatively small parameter spaces; for example, the landmark experiment in the original CPAB work used d = 38 parameters. In contrast, our synthetic benchmarks use 842 parameters in 2D and 8,232 parameters in 3D, while the IXI configuration uses 20,577 parameters per CPAB transformation. This fine-tessellation regime strongly favors gradient-based optimization.

Empirical comparison. To verify this empirically, we benchmarked the official CPAB MCMC implementation using the same IXI atlas-to-patient registration setup as FG-CPAB. We allocated MCMC a wall-clock optimization budget 100× larger than that used by FG-CPAB. Nevertheless, MCMC achieved a Dice score of only 0.453, compared with 0.760 for gradient-based FG-CPAB.

These results do not imply that MCMC is unsuitable for every CPAB application. Rather, they show that it is uncompetitive for the high-dimensional, fine-tessellation problems considered here and cannot replace gradient computation when CPAB is used within an end-to-end differentiable model.

## K Assets, Licenses, and Terms of Use

This appendix is provided to document the external assets used in this work, including datasets, codebases, and benchmark implementations.

## K.1 Public datasets and benchmark data

Table 8: Datasets and benchmark data used in the paper.
<table><tr><td>Asset</td><td>How used in this paper</td><td>Citation</td><td>Version</td><td>License</td></tr><tr><td>IXI brain MRI dataset</td><td>3D deformable registra- [21] tion benchmark</td><td></td><td>link 26.4.26</td><td>Creative Commons CC BY-SA 3.0 license</td></tr><tr><td>DIMON diffusion benchmark ment in § 4.5 data / pipeline</td><td>reaction- Operator-learning experi- [39]</td><td></td><td>Commit - a22ed33</td><td>MIT License</td></tr></table>

## K.2 External code, software, and baseline implementations

Table 9: External codebases, libraries, and baseline implementations used by the project.
<table><tr><td>Asset</td><td>How used in this paper</td><td>Citation</td><td>Version / commit</td><td>License</td></tr><tr><td>libcpab</td><td>Primary CPAB baseline and reference implementa-</td><td>[10]</td><td>Commit - 22df6cd</td><td>MIT license</td></tr><tr><td>DIFW</td><td>tion 1D performance baseline in Appendix A</td><td>[28]</td><td>Commit - f970dd3</td><td>MIT license</td></tr><tr><td>VoxelMorph</td><td>Dense SVF baseline / run- time comparison</td><td>[3]</td><td>Commit - 6fea3e8</td><td>Apache-2.0 license</td></tr><tr><td>torch-lddmm</td><td>LDDMM runtime baseline</td><td>[4]</td><td>Commit - 86d5c43</td><td>Apache-2.0 license</td></tr></table>