# CVTSplat: Multi-view Reconstruction Using an Adaptive Centroidal Voronoi Tessellation

Briac Toussaint<sup>✉</sup> Max Planck Institute for Informatics Saarbrücken, Germany btoussai@mpi-inf.mpg.de

Diego Thomas   
Kyushu University   
Fukuoka, Japan   
thomas@ait.kyushu-u.ac.jp   
Edmond Boyer   
Inria   
Grenoble, France   
edmond.boyer@inria.fr

Jean-Sébastien Franco Inria Grenoble, France jean-sebastien.franco@inria.fr

## Abstract

Voronoi tessellations have recently emerged as an adaptive, spacepartitioning discretization for shape and appearance reconstruction from multi-view images. However, the irregular cells they produce make the repeated reconstruction of the tessellation during optimization costly. We therefore explore adaptive centroidal Voronoi tessellations (CVTs), which extend standard CVTs with non-uniform cell sizes while preserving the regular, well-shaped cells that make them eficient to compute. In CVTSplat, cell size is modulated by the evolving signed distance field, concentrating sites near the zero level set while keeping cells nearly centroidal. This regularity lets us specialize the whole pipeline, with GPU algorithms for nearest-neighbor queries and approximate Voronoi construction, and a rasterizer tailored to small, nearly isotropic cells. Our Voronoi construction is 5–10× faster than general-purpose alternatives in this regime. On DTU and BlendedMVS, CVTSplat reaches geometric accuracy competitive with the best Gaussiansplatting surface methods while training in about 2.5 minutes on a single GPU.

## CCS Concepts

• Computing methodologies → Reconstruction; Shape representations.

## Keywords

Multiview Reconstruction, Voronoi Diagram, Implicit Surface

## 1 Introduction

Christian Theobalt Max Planck Institute for Informatics Saarbrücken, Germany theobalt@mpi-inf.mpg.de

Multi-view reconstruction aims at recovering the geometry and appearance of a 3D scene from a collection of images acquired from diferent viewpoints. A central challenge is to design a rep resentation that provides suficient spatial resolution to capture fine surface details while remaining computationally eficient to optimize and render. Implicit surface methods coupled with diferentiable rendering have demonstrated high reconstruction quality [54], but their practical eficiency and compactness strongly depend on how the scene volume is discretized. Uniform discretizations, such as dense voxel grids, allocate the same capacity throughout space even though accurate geometry and appearance are primar ily required near the reconstructed surface. Conversely, adaptive representations concentrate capacity in relevant regions, but introduce irregular spatial configurations that can make neighborhood queries, geometric operations, and rendering more expensive.

![](images/eb8bb83672570809e64cdbc35507c95c654424aee80151bd7b633af146b4adb1.jpg)  
Figure 1: We represent the scene with Voronoi cells carrying trainable signed-distance and view-dependent appearance attributes, and directly rasterize them for diferentiable volume rendering (middle). Our adaptive CVT combines regular cells with increased spatial resolution near the zero level set, concentrating representation capacity where it is most needed (bottom right). Top left: signed-distance values mapped to hue; bottom right: randomized cell colors illustrate the adaptive refinement near the surface.

Voronoi-based representations provide an interesting direction for addressing this trade-of. Recent works have explored Voronoi and related cellular discretizations for neural rendering and reconstruction [14, 15, 36, 47, 50, 56]. However, these representations do not explicitly exploit one of the fundamental properties of Centroidal Voronoi Tessellations (CVTs): their tendency to produce regular, well-shaped cells. Such regularity is particularly attractive computationally, since irregular Voronoi cells may require visiting many neighboring sites when constructing the diagram [46]. At the same time, a standard CVT tends toward a uniform distribution of sites, thereby losing the main benefit of an adaptive representation.

We therefore investigate adaptive centroidal Voronoi tessella tions as a spatial representation for multi-view reconstruction. We modulate cell size using the implicit surface, yielding a nearcentroidal tessellation with increased site density around the zero level set (fig. 1). This combines the representational eficiency of an adaptive discretization with the computational benefits of regular Voronoi cells, while relying on a standard Voronoi diagram rather than a power diagram. Specifically, power diagrams ofer another way to achieve adaptivity, but require an additional weight per site and incur a higher construction cost [14, 49].

Based on this representation, we introduce CVTSplat, a diferentiable multi-view reconstruction method in which each Voronoi site carries a signed-distance value and compact appearance attributes decoded into view-dependent radiance. The near-centroidal structure of the tessellation lets us specialize every stage of the pipeline to this regime. Since well-shaped cells have few and nearby neighbors, the Voronoi diagram can be constructed from a small set of candidate sites. We deliberately accept an approximate diagram in the rare cases where this set is incomplete, as the resulting local overlaps are compatible with diferentiable primitive-based rendering [26]. Similarly, instead of ray tracing an exact diagram [15], we directly rasterize the cells [14] and exploit their small, nearly isotropic shape to tailor the rasterizer. Together, these choices make CVTSplat a regular yet surface-adaptive discretization for eficient geometry and appearance reconstruction.

The complete optimization pipeline is described in figure 2. It consists of two components: the adaptive tessellation controls the position of the sites, while the diferentiable rendering branch optimizes the signed-distance and appearance attributes. The tessellation is guided by the evolving implicit surface to produce a surface-adaptive discretization. Our main contributions are as follows:

(1) We introduce an adaptive centroidal Voronoi discretization for multi-view reconstruction, combining surface adap tivity with the regularity of centroidal tessellations. Our complete reconstruction pipeline achieves geometric accu racy competitive with state-of-the-art 3DGS-based methods [7, 13, 63] on DTU, while training in approximately 2.5 min utes on a single RTX 5090 and using 4 GB of VRAM.

(2) We exploit the regularity of the tessellation to make Voronoibased reconstruction computationally eficient. We introduce specialized GPU algorithms for approximate Voronoi construction, repeated �-nearest-neighbor queries, and Delaunay triangulation, substantially accelerating these operations compared to general-purpose approaches [4, 5, 15, 46, 49].

(3) We propose a rasterizer optimized for rendering small, isotropic primitives, further consolidating our framework.

## 2 Related Work

We review the literature most closely related to our approach along three directions: adaptive spatial discretizations for implicit 3D reconstruction (Sec. 2.1), explicit primitive-based representations (Sec. 2.2), and eficient GPU construction of 3D Voronoi and Delaunay diagrams (Sec. 2.3).

## 2.1 Adaptive Spatial Discretizations

Multi-view 3D reconstruction based on diferentiable rendering initially relied on continuous neural representations [37, 54], before increasingly adopting grid-based or hybrid spatial encodings to accelerate optimization and rendering [6, 11, 34, 40, 53]. The main objective of these representations is to allocate computation sparsely, by concentrating capacity around the reconstructed surface. However, grid-aligned discretizations typically follow a predefined spatial structure that either stays fixed during optimization or changes discretely across levels of detail.

Unstructured discretizations ofer greater flexibility in adapting both the number and spatial extent of their elements. Recent approaches have explored Voronoi or Power cells [14, 15, 47, 50], as well as tetrahedral representations [29, 36, 52, 56]. These works motivate spatial representations that retain the flexibility ofunstructured discretizations while achieving the accuracy and eficiency of highly optimized grid-based methods.

## 2.2 Explicit Splatting Representations

3D Gaussian Splatting [26] has become a widely used representation for high-quality novel-view synthesis due to its adaptive distribution of primitives and eficient rasterization. However, recovering coherent geometry from an unstructured set of Gaussians is highly challenging [7, 13, 18, 19, 21, 45, 61, 63]. Existing methods introduce additional constraints or regularization to suppress floaters and improve depth and normal consistency [7, 63], or dedicated procedures for extracting closed surfaces [13, 18].

Space-partitioning primitives such as tetrahedra and Voronoi cells provide an alternative splat support [15, 36] in which neighboring elements form a coherent spatial decomposition rather than an overlapping collection of primitives. This structure provides consistent spatial ordering, but maintaining it while the underlying sites move during optimization can introduce considerable computational overhead. Eficiently updating such adaptive representations therefore remains an important challenge.

## 2.3 GPU Voronoi & Delaunay Diagrams Construction

A 3D Voronoi diagram can be represented through its adjacency graph, i.e., the neighboring relationships between generating sites. This graph can be recovered indirectly through the dual Delaunay triangulation [15], or computed directly from the Voronoi cells. GPU Delaunay algorithms enable substantial parallelism [5, 15], but generally require multiple rounds of point insertion, tetrahedron splitting, flipping, and repair, making frequent reconstruction of a dynamically evolving diagram costly.

![](images/5515734b9d6ce02001bd718b26e1afbbc99f87eaa23c1f1039ec401194b3aef7.jpg)  
Figure 2: Overview of our approach during optimization. Top row: Construction of the tessellation. (a) The sites are processed to obtain a list of their 32 nearest neighbors, (b) before building the Voronoi diagram as an adjacency graph. (c) The center of mass of each cell is then computed to regularize the distribution of the sites via the CVT loss. (d) The Voronoi diagram is optionally converted to a Delaunay triangulation to extract the zero-level set with marching tetrahedra. Bottom row: Diferentiable volume rendering from posed images. (e) The sites’ attributes (learnable SDF and features) are smoothed with a Gaussian kernel. (f) The gradient of the SDF is also computed to produce normal vectors, which define a piecewise linear SDF over the cells. Then, for all viewpoints in the mini-batch, (g) a per-site color is computed based on the camera’s position, (h) before rasterizing the scene and comparing against the ground-truth image.

GPU methods for direct Voronoi construction were initially developed for relatively uniform point distributions [4, 46], providing large speedups over CPU implementations [31, 51]. More recently, Taveira et al. [49] extended this family of approaches to non-uniform point distributions. Following the clipping formulation of Ray et al. [46], Voronoi cells are constructed independently by successively clipping them against bisector planes induced by neighboring sites. Although cells can be processed in parallel, the clipping operations for each cell are sequential, and irregular cells may require significantly more operations. Leaving cells unregularized [15] can thereby increase the total construction cost.

## 3 Fast and Adaptive Spatial Data Structure

Our spatial representation consists of a set of sites $\{ \mathbf { x } _ { i } \} _ { 0 \leq i < N }$ (i.e. 3D points) that induce a Voronoi tessellation of the scene volume. Each site stores the geometric and appearance attributes used by the reconstruction, including a signed-distance value and viewdependent radiance features. The Voronoi cells provide the volumetric primitives used for rendering, while their dual Delaunay connectivity provides the local neighborhood structure required for mesh extraction. Because the site positions evolve throughout optimization, these structures must be updated repeatedly. A key property of our representation is that the sites remain close to a centroidal configuration, producing compact cells with a small number of nearby neighbors and enabling eficient GPU construction.

In the following, we first describe how we obtain this adaptive centroidal distribution (Sec. 3.1) and compute its centroids (Sec. 3.2). We then present our fast Voronoi construction (Sec. 3.3), its conversion to Delaunay tetrahedra (Sec. 3.4), and the KNN kernel supporting these operations (Sec. 3.5).

## 3.1 Adaptive Centroidal Voronoi Tessellation

![](images/6cb8471eb64ace9e8ffaeebf8fc2d716ebf4c3c574a2eab2b3732037d0748e91.jpg)  
Figure 3: Left: The volume and centroid of the cell are computed by casting rays from the site’s location. Right: Our biased CVT loss maintains the sites slightly closer to the implicit surface than the centroids x¯, resulting in stretched cells along the SDF gradient.

In a centroidal Voronoi tessellation (CVT), the sites $\mathbf { x } _ { i }$ coincide with the center of mass $\bar { \bf x } _ { i }$ of their cell (the centroids). It is not the case at first because the sites have arbitrary positions, but iteratively minimizing the CVT loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { C V T } } = \sum | | \mathbf { \tilde { x } } _ { i } - \bar { \mathbf { x } } _ { i } | | ^ { 2 } } \end{array}$ brings the tessellation closer to being centroidal, with regular cells [35]. Since we need an adaptive distribution rather than a uniform one, we propose to bias this loss using the normal vector of the SDF (see figure 3). Our biased CVT loss is as follows:

$$
\mathcal { L } _ { \mathrm { C V T } } = \sum _ { i } \frac { 1 } { \mathrm { s g } ( r _ { i } ) } | | \mathbf { x } _ { i } - \mathrm { s g } ( \bar { \mathbf { x } } _ { i } - \beta _ { i } r _ { i } \mathbf { n } _ { i } ) | | _ { 2 } ^ { 2 } .\tag{1}
$$

sg(·) is the stop-gradient operator and $r _ { i }$ is the radius of the cell. Adaptivity may also be achieved using a spatially-varying mass density when computing cell moments [9] but our approach is more direct. Intuitively, the centroid $\bar { \bf x } _ { i }$ is ofset by a fraction $\beta _ { i }$ of the cell radius along the SDF gradient $\mathbf { n } _ { i }$ . We divide by $r _ { i }$ to make the loss scale-invariant. $\beta _ { i }$ directly depends on the SDF $\hat { f } _ { i } \colon$

$$
\beta _ { i } = a \cdot \operatorname { t a n h } ( \frac { \hat { f } _ { i } } { 2 b } ) .\tag{2}
$$

�, $b > 0$ control the magnitude of the shift. Using $a = 0$ gives the standard centroidal tessellation. In our experiments, we set $a = 0 . 1$ and $b = 0 . 0 0 1$ which corresponds to shifting the centroid by 10% of the cell radius. This results in smaller cells near the zero level-set and larger cells further away, with cell radius growing approximately linearly with SDF (refer to the supplementary material for an empirical analysis). The size of the cells is not explicitly imposed, which lets them arrange themselves independently of their number and of the object’s surface area.

## 3.2 Computation of the Centroids

The CVT loss depends on the centroid $\bar { \mathbf { x } } _ { i : }$ , itself depending on the volume $\mathcal { N } _ { i } .$ . They can be computed using a tetrahedral decomposition of the cell [46] or its polyhedron representation [39]. We propose a simpler alternative, although approximate, based on Riemann sums. Evaluating these sums only requires knowledge of the bisector planes, directly available from the Voronoi adjacency graph. Specifically, we sample $n = 6 4$ rays $\mathbf { r } _ { k } \in S ^ { 2 }$ from a Fibonacci sphere lattice, and measure the distance $l _ { k }$ from the site $\mathbf { x } _ { i }$ to the nearest bisector plane along $\mathbf { r } _ { k }$ (see figure 3, left).

$$
\mathcal { V } _ { i } = \int _ { C _ { i } } d \mathbf { x } = \int _ { S ^ { 2 } } \int _ { 0 } ^ { l ( \mathbf { r } ) } s ^ { 2 } d s d \sigma = \int _ { S ^ { 2 } } \frac { l ( \mathbf { r } ) ^ { 3 } } { 3 } d \sigma \approx \frac { 4 \pi } { 3 n } \sum _ { k = 0 } ^ { n - 1 } l _ { k } ^ { 3 } .\tag{3}
$$

The volume integral over cell $C _ { i }$ is decomposed into a 1D integral and an integral over the unit sphere $S ^ { 2 } . l ( \mathbf { r } )$ is the distance from the site to the nearest clipping plane along r and �� is an infinitesimal surface element on $S ^ { 2 } .$ . Because the rays are approximately uniformly spread, their weight in the Riemann sum is $\textstyle { \frac { 4 \pi } { n } }$ . A similar computation gives the centroid $\bar { \bf x } _ { i }$

$$
\bar { \bf x } _ { i } - { \bf x } _ { i } = \frac { 1 } { \mathcal { V } _ { i } } \int _ { C _ { i } } { \bf x } - { \bf x } _ { i } d { \bf x } = \frac { 1 } { \mathcal { V } _ { i } } \int _ { S ^ { 2 } } \int _ { 0 } ^ { l ( { \bf r } ) } { \bf r } s \cdot { s } ^ { 2 } d s d \sigma\tag{4}
$$

$$
= \frac { 1 } { \mathcal { V } _ { i } } \int _ { S ^ { 2 } } \mathbf { r } \frac { l ( \mathbf { r } ) ^ { 4 } } { 4 } d \sigma \approx \frac { 1 } { \mathcal { V } _ { i } } \frac { 4 \pi } { n } \sum _ { k = 0 } ^ { n - 1 } \mathbf { r } _ { k } \frac { l _ { k } ^ { 4 } } { 4 } = \frac { 3 } { 4 } \frac { \sum \mathbf { r } _ { k } l _ { k } ^ { 4 } } { \sum l _ { k } ^ { 3 } } .\tag{5}
$$

$\mathcal { N } _ { i }$ and $\bar { \bf x } _ { i }$ are re-evaluated at every training iteration. The accuracy of these quadrature formulas is assessed in the appendix. Our approach does not preclude using a more direct computation [39, 46] if greater precision is required, since Voronoi vertices can be recovered from the Delaunay triangulation (Sec. 3.4).

## 3.3 Voronoi Diagram Construction

Following previous GPU approaches [4, 46, 49], we represent the Voronoi diagram as an adjacency graph. However, we propose a new algorithm designed specifically for near-centroidal distributions. We assume the sites are in general position as a simplification, this means there are no duplicates and five sites or more are not cospherical.

Cells as linear programs. The Voronoi cell of the site $\mathbf { x } _ { i }$ is:

$$
C _ { i } = \{ \mathbf { x } \in \mathbb { R } ^ { 3 } , | | \mathbf { x } - \mathbf { x } _ { i } | | _ { 2 } ^ { 2 } \leq | | \mathbf { x } - \mathbf { x } _ { j } | | _ { 2 } ^ { 2 } , \quad \forall j \neq i \} .\tag{6}
$$

Let $\textbf { x } = \textbf { x } _ { i } + \textbf { y } .$ The distance constraints in equation 6 can be reformulated geometrically as plane equations:

$$
| | \mathbf { x } - \mathbf { x } _ { i } | | _ { 2 } ^ { 2 } \leq | | \mathbf { x } - \mathbf { x } _ { j } | | _ { 2 } ^ { 2 }\tag{7}
$$

$$
\begin{array} { r l } { \Leftrightarrow } & { { } \| \mathbf { y } \| _ { 2 } ^ { 2 } \leq \| \mathbf { x } _ { i } + \mathbf { y } - \mathbf { x } _ { j } \| _ { 2 } ^ { 2 } } \end{array}\tag{8}
$$

$$
\begin{array} { r l } { \Leftrightarrow } & { { } ~ | | \mathbf { y } | | _ { 2 } ^ { 2 } \leq | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | _ { 2 } ^ { 2 } + 2 \mathbf { y } \cdot ( \mathbf { x } _ { i } - \mathbf { x } _ { j } ) + | | \mathbf { y } | | _ { 2 } ^ { 2 } } \end{array}\tag{9}
$$

$$
\Leftrightarrow \quad \mathbf { y } \cdot \left( \mathbf { x } _ { j } - \mathbf { x } _ { i } \right) \leq { \frac { 1 } { 2 } } | | \mathbf { x } _ { j } - \mathbf { x } _ { i } | | _ { 2 } ^ { 2 } .\tag{10}
$$

Therefore, the Voronoi cell is equivalently defined as the feasibility region of a linear program in R<sup>3</sup>: $\bar { \mathbb { R } ^ { 3 } }$

$$
C _ { i } = \big \{ \mathbf { x } \in \mathbb { R } ^ { 3 } , ( \mathbf { x } - \mathbf { x } _ { i } ) \cdot ( \mathbf { x } _ { j } - \mathbf { x } _ { i } ) \leq \frac { 1 } { 2 } \vert \vert \mathbf { x } _ { j } - \mathbf { x } _ { i } \vert \vert _ { 2 } ^ { 2 } , \quad \forall j \neq i \big \} .\tag{11}
$$

The Voronoi cell $C _ { i }$ is bounded by a subset of the bisector planes between $\mathbf { x } _ { i }$ and its neighboring sites. We call a neighbor $\mathbf { x } _ { j }$ active if its bisector plane with $\mathbf { x } _ { i }$ intersects $C _ { i }$ , and the corresponding constraint in (11) contributes to the boundary of $C _ { i } .$ Not all neigh boring sites are necessarily active; for example, removing $\mathbf { x } _ { k }$ does not change $C _ { i }$ in Fig. 4. Therefore, identifying the set of active neighbors, or equivalently their bisector planes, is suficient to define the Voronoi cell.

![](images/a2ea0f232ad72d196e9efb35fc6c332f6671982cab2f348ba1ca3bffe96308ec.jpg)  
Figure 4: Voronoi cell in 2D. The bisector plane $\mathcal { P } _ { k }$ is inactive since it does not participate in defining the cell $C _ { i }$ . Any neighboring point beyond the security radius, defined as twice the cell radius, is guaranteed to not participate.

Adjacency as 2Dfeasibility. An active bisector plane must contain points of $C _ { i }$ since it corresponds to a facet of the convex polyhedron enclosing the cell. Conversely, inactive bisector planes do not contain any points of the Voronoi cell. This leads to a simple test:

$$
\mathrm { T h e ~ s i t e } ~ r \mathrm { ~ i s ~ a ~ V o r o n o i ~ n e i g h b o r ~ o f ~ s i t e } ~ i\tag{12}
$$

$$
\iff { \mathcal { P } } _ { r } { \mathrm { ~ i s ~ a c t i v e ~ f o r ~ c e l l ~ } } C _ { i }\tag{13}
$$

$$
\Longleftrightarrow \mathcal { P } _ { r } \cap C _ { i } \neq \emptyset\tag{14}
$$

We parametrize $\mathcal { P } _ { r }$ as $\begin{array} { r } { \textbf { x } = ~ \frac { 1 } { 2 } \big ( \mathbf { x } _ { r } + \mathbf { x } _ { i } \big ) + u \mathbf { e } _ { 1 } + v \mathbf { e } _ { 2 } , } \end{array}$ , where $( \mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , ( \mathbf { x } _ { r } - \mathbf { x } _ { i } ) / | | \mathbf { x } _ { r } - \mathbf { x } _ { i } | | )$ is an orthonormal basis [10]. Substituting the parametric equation for $\mathcal { P } _ { r }$ into the remaining constraints of Eq. (11) yields a 2D linear program (see Fig. 5):

$$
a _ { j } u + b _ { j } v \leq c _ { j } , \qquad j \neq i , j \neq r ,\tag{15}
$$

and ${ \bf x } _ { r }$ is a neighbor if and only if this system is feasible.

![](images/b14004f3fb310603f94ce6fa4f8b5047636049e32d275a4458033101518b9242.jpg)  
Figure 5: The 2D inequalities enclose the feasibility region, which corresponds to a facet of the polyhedron wrapping the cell. Each line is the intersection of the reference plane $\mathcal { P } _ { r }$ with another bisector plane $\mathcal { P } _ { j }$

Feasibility test. We test feasibility with Seidel’s incremental algorithm [2, 48], which maintains a feasible point and updates it only when a new constraint is violated, by solving a 1D problem along that constraint’s boundary line. Instead of shufling the constraints randomly, we process them in increasing distance order, as returned by the KNN kernel (Sec. 3.5). The nearest planes are the most likely to bound the cell, so infeasibility is usually detected early.

GPUimplementation. We consider that only the 32 nearest neigh bors are likely to define active bisector planes, so there are 32 threads dedicated for building each cell. Each one tests a candidate plane $\mathcal { P } _ { r }$ against the 37 remaining constraints (31 other neighbors and the 6 planes of the scene bounding box). The 2D constraints are built on the fly from the site positions so no per-thread storage is needed, which is a significant bottleneck in previous approaches [4, 46] due to reduced GPU occupancy. At the end, each thread outputs a single bit classifying whether the neighbor is Voronoiadjacent.

Approximation. The assumption that 32 neighbors are enough generally holds in the case of our biased CVT. To further reduce the number of missing adjacencies, we center the KNN queries at the previously computed cell centroids rather than at the sites themselves. This substantially improves the completeness of the resulting Voronoi diagrams, as demonstrated in the supplementary material. More neighbors could be included to handle more irregular cells, though with a performance loss. If a true neighbor beyond the � nearest sites is missed, the cell will be enlarged and will overlap its neighbor. A missing constraint can also enlarge the feasible region of Eq. (15), so that a non-neighbor is wrongly marked as adjacent. Both errors are local and only afect cells near the few sites concerned.

## 3.4 Delaunay Triangulation

![](images/0e345bcab2616dbd71734772cd3bc89e9d8dabb3485f50ee0bb2aff45a2fc615.jpg)  
Figure 6: The adjacency matrix of the site � encodes which of its neighbors are adjacent to each other.

We convert the approximate Voronoi diagram into the dual Delaunay triangulation for mesh extraction, illustrated in figure 6. For conciseness, "Voronoi neighbor" will be shortened to "neighbor" in the following discussion.

The Delaunay tetrahedra are groups of 4 sites $( i , j , k , l )$ that are pairwise neighbors, and whose circumsphere does not contain any other site. Thus, we enumerate all possible 4-cliques, and then test whether their circumsphere is empty. Only ordered tuples $i < j < k < l$ are emitted to avoid duplicates.

We build a binary adjacency matrix M for each site �, that encodes the adjacency relation of its neighbors: $M _ { j , k }$ is 1 if sites � and � are neighbors. To form a quadruplet, we look for a neighbor � of � such that $M _ { j , l }$ and $M _ { k , l }$ are set to 1 in M.

We eficiently enumerate the quadruplets (�, �, �, �) by leveraging the binary nature of M: because each site has 32 neighbors at most, the rows of M can be encoded as 32-bit integers. Thus, � indices correspond to the direct neighbors of �, � indices correspond to the bits set in row $M _ { j } ,$ and � indices correspond to the bits set in the binary AND of rows $M _ { j }$ and $M _ { k }$ . The \_\_fs CUDA intrinsic lets us iterate over the toggled bits of an integer.

Not all quadruplets form Delaunay tetrahedra so we remove the ones containing other sites in their circumsphere. In practice, we only need to test the neighbors of � against the circumsphere. A proof is available in the supplementary document.

Because our Voronoi diagram may be incomplete, our triangulation is not guaranteed to be exact for cells with missing clipping planes or in case of numerical ambiguities in the circumsphere check. This can result in a few holes or non-manifold edges in the extracted mesh out of millions of triangles. We quantify these defects in the supplementary material.

## 3.5 Fast GPU KNN Search

We propose a fast K-nearest neighbor kernel, tailored for $K = 3 2$ and for repeated queries, to take advantage of our specific setting:

the sites move by small amounts between rebuilds of the nearestneighbors lists, which implies a high degree oftemporal consistency that can be leveraged. Like Jakob et al. [24], we start by building an acceleration structure to avoid a naive �(�<sup>2</sup>) brute force check. We use a classic LBVH [30] with up to 128 points per node. This gives a shallow hierarchy well-suited for querying a relatively large number of neighbors.

The nearest neighbors are gathered using ball-queries, i.e. gath ering all points within some distance of the query point, instead of the more common heap-based approach [24, 46]. The ball-query is restarted with a smaller or larger radius until the number of gathered points is between 32 and 64, using a binary search. They are subsequently sorted in increasing distance to the query point [42] and the first 32 are returned. Each query point has its own search radius which we save for later runs. The point cloud evolves slowly so the search radius is likely to be valid the next time the KNN routine is called, sparing repeated BVH traversals.

Heap-based approaches generally assign one thread per query point, and therefore need a per-thread traversal stack or priority queue, in addition to the heap already required to keep track of the K closest points. While the heap can be stored in registers [24], this reduces occupancy considerably for large values of K or spills the registers to global memory.

Our strategy addresses these issues: we assign one warp per query point, with all lanes visiting the BVH together. A single BVH traversal stack is needed for the whole warp, and the points can be loaded with coalesced memory accesses. The points within the search radius are compacted to a shared memory bufer sorted in parallel at the end, which is faster than iterative heap insertions.

## 4 Diferentiable Volume Rendering

After gathering the nearest neighbors of all sites and computing the Voronoi diagram, we can proceed with rendering of the scene. The volume rendering integral is typically evaluated with ray-marching [37, 54], by querying the radiance and density fields at arbitrary locations along the rays. Directly adopting this formulation for our discretization is ill-suited: interpolating point neighborhoods at each ray sample is costly [58], while interpolating over the Voronoi cells generally requires a tetrahedral mesh [36, 52]. Instead, we follow RadiantFoam [15] and use piecewise integration where each cell only contributes to the portion of the ray it intersects. Radiant-Foam [15] and SDFoam [47] use a constant density inside the cells which does not account for local surface orientation. In comparison, we define the SDF as a linear function in each cell to obtain nonlinearities once composed with NeuS [54]’s SDF-to-density conversion, and which crucially still gives a closed-form solution to the volume integral over the ray segments.

Three elements are necessary: (i) the SDF gradient at each site to define a piecewise SDF, (ii) per-primitive radiance and (iii) regularization terms to stabilize optimization. We reuse the nearest neighbors gathered during the Voronoi diagram construction (section 3.5) to regularize the sites’ attributes and to compute the SDF gradient. For view-dependent appearance, we favor a neural implicit approach instead of per-site spherical harmonics to reduce the number of appearance parameters, as explained in section 4.2.

Finally, section 4.3 covers our rasterization algorithm designed for small primitives.

## 4.1 Attributes Smoothing

Smoothing shape and appearance attributes is essential to stabilize optimization and recover sensible geometry in the case of discrete representations [53, 57]. Here, we leverage ideas from the Smoothed Particle Hydrodynamics (SPH) literature to define smoothing operators of arbitrary fields defined at the sites [44]. These operators are diferentiable with respect to spatial coordinates, which in turn enables us to define the gradient of the SDF in a diferentiable manner. Let �(�) be the smoothing operator:

$$
G ( u ) _ { i } = \sum _ { j \in N ( i ) } u _ { j } w _ { i j } / \sum _ { j \in N ( i ) } w _ { i j } .\tag{16}
$$

� can be any scalar or vector field held by the sites, and �(�)<sub>�</sub> denotes the smoothed field computed at the position of site �. N (�) is the set of neighbors of �, plus � itself. The smoothing weights are defined as follows:

$$
w _ { i j } = \mathscr { V } _ { j } \exp \left( { - \frac { | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } R _ { i } ^ { 2 } } } \right)\tag{17}
$$

$$
R _ { i } = \operatorname* { m a x } _ { j \in N ( i ) } ( | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | _ { 2 } ) .\tag{18}
$$

$\mathcal { V } _ { j }$ is the volume of cell � and � is chosen so that sites further away than the last neighbor would have small weights if they were included. Other smoothing schemes are possible [27, 28] but we found this simple one to be efective. The weights �<sub>��</sub> take the cell volume into account so that � approximates the convolution of the field � by a Gaussian kernel:

$$
( u ( \mathbf { y } ) * e ^ { - \frac { \| \mathbf { y } \| ^ { 2 } } { 2 \sigma ^ { 2 } R ^ { 2 } } } ) ( \mathbf { x } ) = \int _ { \mathbb { R } ^ { 3 } } u ( \mathbf { y } ) \exp \left( - \frac { \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } { 2 \sigma ^ { 2 } R ^ { 2 } } \right) d \mathbf { y }\tag{19}
$$

$$
\approx \sum _ { j } u _ { j } \exp \left( - \frac { | | \mathbf { x } - \mathbf { x } _ { j } | | ^ { 2 } } { 2 \sigma ^ { 2 } R ^ { 2 } } \right) \mathcal { V } _ { j } .\tag{20}
$$

Equation 20 is the Riemann sum of the integral discretized over all the sites, that we further divide by the sum of the weights to normalize �. Following [53, 57], we apply our Gaussian filter � on the raw SDF � to obtain smoother values:

$$
{ \hat { f } } = G ( f )\tag{21}
$$

Next, we obtain the gradient vector of the smoothed SDF by computing partial derivatives of � applied on $\hat { f } \colon$

$$
\mathbf { n } _ { i } = \nabla _ { \mathbf { x } _ { i } } G ( \hat { f } ) _ { i } .\tag{22}
$$

We are now able to define regularization objectives typically used in multi-view implicit surface reconstruction:

$$
\mathcal { L } _ { \mathrm { S D F } } = \sum _ { i } \frac { ( f _ { i } - \hat { f _ { i } } ) ^ { 2 } } { \operatorname* { m a x } ( | \operatorname { s g } \left( \hat { f _ { i } } \right) | , 0 . 0 1 ) }\tag{23}
$$

$$
\mathcal { L } _ { \mathbf { n } } = \sum _ { i } | | \frac { \mathbf { n } _ { i } } { | | \mathbf { n } _ { i } | | _ { 2 } } - \mathrm { s g } \left( G ( \frac { \mathbf { n } _ { i } } { | | \mathbf { n } _ { i } | | _ { 2 } } ) \right) | | _ { 2 } ^ { 2 }\tag{24}
$$

$$
\mathcal { L } _ { \mathrm { E i k } } = \sum _ { i } ( | | { \bf n } _ { i } | | _ { 2 } - 1 ) ^ { 2 }\tag{25}
$$

L<sub>SDF</sub> keeps $f$ and $\hat { f }$ close to each other, $\mathcal { L } _ { \bf n }$ is a smoothness term on the normalized gradient vectors and $\mathcal { L } _ { \mathrm { E i k } }$ is the classic eikonal regularization [17]. sg(·) is the stop gradient operator added to simplify the backward pass. Furthermore, we do not back-propagate loss gradients through the positions of the sites, by treating them as constants in this formulation: their positions are entirely managed by the CVT loss defined in section 3.1.

So far, the SDF was only defined at the sites. We extend it to an entire Voronoi cell using an afine approximation:

$$
{ \hat { f } } ( \mathbf { x } ) = { \hat { f } } _ { i } + ( \mathbf { x } - \mathbf { x } _ { i } ) \cdot \mathbf { n } _ { i } , \quad \forall \mathbf { x } \in C _ { i } .\tag{26}
$$

When considering all the cells, the signed-distance field will be piecewise linear with small discontinuities allowed between cells. The afine approximations let us obtain a closed-form solution to the volume rendering integral, as described in section 4.3, which may not be possible with typical interpolation schemes.

## 4.2 Appearance Representation

![](images/3abb8499d9faf7094878b2459350d3536d5476b7b5d88d65148ef354bbcb7add.jpg)  
Figure 7: The light probes, parametrized with spherical harmonics, are interpolated to produce view-dependent features $\mathbf { F } ^ { a , \mathbf { v } }$ and $\mathbf { F } ^ { a , \mathbf { w } }$ , based on the local view direction v and reflected vector w.

We assign a view-dependent color to each cell $C _ { i }$ to model the scene appearance. Like 3DGS [26], the color is constant inside a primitive, which gives a piecewise constant color field in our case. Our appearance parametrization takes inspiration from [53]. The main idea is to split view-independent components (e.g. texture) from view-dependent components of the light field (e.g. shading) to significantly reduce the number of parameters compared to a distinct spherical harmonic (SH) basis per primitive.

The view-independent component is parametrized by a feature vector $\mathbf { F } _ { i } ^ { s } \in \mathbb { R } ^ { 4 }$ , called the spatial features, assigned to each site. The view-dependent component is modeled by a dense voxel grid of spherical harmonic (SH) coeficients called the light probes. They are trilinearly interpolated at the location of the sites to produce a per-primitive SH basis, subsequently evaluated in the direction of a unit vector to obtain angular features $\mathbf { F } _ { i } ^ { a } \in \mathbb { R } ^ { 4 }$ . More specifically, we use two sets oflight probes: the first one is queried with the incident vector v and the other uses the reflected vector w, which give $\mathbf { F } _ { i } ^ { a , \mathbf { v } }$ and $\mathbf { F } _ { i } ^ { a , \mathbf { w } }$ respectively. The site color $\mathbf { c } _ { i }$ is obtained by decoding the spatial and angular features with a tiny MLP:

$$
\mathbf { c } _ { i } = \mathrm { M L P } ( \mathbf { F } _ { i } ^ { s } , \mathbf { F } _ { i } ^ { a , \mathbf { v } } , \mathbf { F } _ { i } ^ { a , \mathbf { w } } , ( \mathbf { n } _ { i } \cdot \mathbf { v } _ { i } ) ^ { 1 } , \cdot \cdot \cdot , ( \mathbf { n } _ { i } \cdot \mathbf { v } _ { i } ) ^ { 4 } )\tag{27}
$$

For regularization, we add a smoothness term on the spatial features, similar to the smoothness term on the normals:

$$
\mathcal { L } _ { \mathrm { F } ^ { s } } = \sum _ { i } | | \mathbf { F } _ { i } ^ { s } - s \mathrm { g } \big ( G ( \mathbf { F } _ { i } ^ { s } ) \big ) | | _ { 2 } ^ { 2 }\tag{28}
$$

The light probes are also regularized by applying a Gaussian filter $\mathcal { G }$ on the voxels and penalizing deviations between the smoothed and raw coeficients:

$$
\mathcal { L } _ { \mathrm { p r o b e s } } = \sum | | \beta - \operatorname { s g } ( \mathcal { G } ( \beta ) ) | | _ { 2 } ^ { 2 }\tag{29}
$$

The sum is applied on all the voxels and SH basis coeficients $\beta .$

In practice, the two voxel grids have resolution $3 2 ^ { 3 }$ , with 16 SH coeficients per voxel and each SH coeficient in $\mathbb { R } ^ { 4 }$ . The MLP has 16 neurons with 2 hidden layers and squareplus activations [3]. Because there are multiple sites located in the same voxel in general, the light probes represent a small proportion of all parameters (equivalent $\mathbf { t o } \approx 1$ parameter per primitive in our experiments).

## 4.3 Real-Time Rasterization

The scene is ready to be rasterized into an image once every site has been assigned a color for the current camera position. We use NeuS’ rendering equation [54] with reciprocal transmittance [38] to blend colors along a camera ray $\mathbf { o } + t \mathbf { r } , t > 0 ;$

$$
\mathbf { C } = \int _ { 0 } ^ { + \infty } \mathbf { c } ( t ) \sigma ( t ) T ( 0 , t ) d t\tag{30}
$$

$$
T ( 0 , t ) = e ^ { - \int _ { 0 } ^ { t } \sigma ( u ) d u }\tag{31}
$$

$$
{ \boldsymbol { \sigma } } ( t ) = s \Phi ( - s { \hat { f } } ( t ) ) | \mathbf { n } \cdot \mathbf { r } |\tag{32}
$$

Φ is the sigmoid function and $s > 0$ is the sharpness parameter controlling the density decay outside of the implicit surface. In the equations above, $\mathbf { c } ( t ) = \mathbf { c ( 0 } + t \mathbf { r } ) , \sigma ( t ) = \sigma ( \mathbf { 0 } + t \mathbf { r } )$ , etc.

Because the color is constant inside a cell and the SDF is piecewise linear (see figure 8), and by further assuming the cells do not overlap which is typically the case, we obtain a closed-form solution for equation 30. Complete derivations are provided in the supplementary document, we give the final formula below:

$$
\mathbf { C } = \sum \mathbf { c } _ { i } ( 1 - \alpha _ { i } ) \prod _ { j < i } \alpha _ { j }\tag{33}
$$

$$
\alpha _ { i } = \frac { 1 + e ^ { - s \hat { f } ( t _ { i } ^ { - } ) } } { 1 + e ^ { - s \hat { f } ( t _ { i } ^ { + } ) } } \in [ 0 , 1 ]\tag{34}
$$

We use the convention that � corresponds to the transparency of the cell. $\hat { f } ( t _ { i } ^ { - } )$ and $\hat { f } ( t _ { i } ^ { + } )$ are the piecewise SDF (equation 26) evaluated at the entry and exit points of the ray in the cell respectively. The sum in equation 33 is carried over the set of Voronoi cells intersected by the ray, ordered by increasing distance to the camera to guarantee geometrically-consistent blending (not depth, see [14] for a proof). Note that the fraction in eq. 34 should be inverted for back-facing cells so that $\alpha \leq 1$

In practice, we first assign each cell to every $8 \times 8$ pixel tile intersected by the projection of its bounding sphere, sort them by tile id and by distance, and then evaluate equation 33.

4.3.1 Chunked Rasterizer. We noticed using one thread per ray results in poor GPU utilization whenever some rays intersect more cells than the average, which occurs for grazing rays that are parallel to the surface. In the worst case, a single thread can keep the GPU busy while the rest have terminated long before. Thus, we propose to split the sum in equation 33 into contiguous ranges of 32 cells that we call chunks, with the goal of increasing parallelism during rasterization.

![](images/deaccd25645fbb96edc81fb8f5d4d838d5461f75e7dcc6618653833ac428b4d2.jpg)  
Figure 8: The color is constant but the density varies inside the cells. They are sorted by distance to the camera for popfree rasterization.

![](images/736406dd2dedebfc0803a432c6f7e33e758bfb632034cd47f9b5c87e7ead035c.jpg)  
Figure 9: The pixel tiles are split into chunks of 32 consecutive cells. All chunks are rasterized in parallel but the ones marked in red have been skipped by detecting occlusions dy namically. The final result (top left) is the �-blending of all the chunks in front-to-back order, evaluated in a subsequent pass.

As figure 9 shows, all chunks are scheduled to be processed in parallel, with chunks closer to the camera being handled first since we order them front-to-back. Naively processing the chunks all at once is not ideal because occluded cells would still be intersected and blended. In comparison, a sequential processing can exit as soon as the per-ray transmittance reaches zero. To retain this useful behavior, we propose a communication system based on per-pixel flags that are updated atomically by the chunks when they terminate their local rasterization task.

Each flag contains a transmittance accumulator $T _ { c }$ and a distance counter $D _ { c } ,$ updated jointly with an atomic compare-and-swap. $T _ { c }$ corresponds to the multiplication of the partial transmittance values of the chunks having completed so far. The distance counter $D _ { c }$ contains the (depth) index of the furthest completed chunk. When

a chunk starts, it reads the flags of its 8x8 pixel tile and decides if rasterization is needed based on the following table:
<table><tr><td rowspan=1 colspan=1>Early-exit decision table</td><td rowspan=1 colspan=1> $T _ { c } \geq \epsilon$ </td><td rowspan=1 colspan=1> $T _ { c } < \epsilon$ </td></tr><tr><td rowspan=1 colspan=1>chunk index $< D _ { c }$ </td><td rowspan=1 colspan=1>Cannot skip</td><td rowspan=1 colspan=1>Cannot skip</td></tr><tr><td rowspan=1 colspan=1>chunk index $> D _ { c }$ </td><td rowspan=1 colspan=1>Cannot skip</td><td rowspan=1 colspan=1>Can skip</td></tr></table>

Rasterization is skipped if all rays in the chunk can skip. The accumulator $T _ { c }$ is an upper bound of the true transmittance up to chunk $D _ { c } ,$ valid for all chunks with depth index $i \geq D _ { c }$ . It is an upper bound since earlier chunks may not have updated $T _ { c }$ when further ones read it. Therefore, a low value for $T _ { c }$ guarantees that further chunks will not contribute to the final color. On the other hand, chunks closer than the distance counter have to run regardless of $T _ { c } .$ . This scheme allows chunks pertaining to the same pixel tile to be processed fully in parallel if the GPU has enough thread blocks available, while hidden chunks will be opportunistically skipped. A distinct GPU kernel performs the final blending of each chunk’s output color and transmittance in a later pass.

The main cost in evaluating equation 33 is the computation of the ray/cell intersection intervals $( t _ { i } ^ { - } , t _ { i } ^ { + } )$ due to execution divergence: each ray intersects a diferent subset of cells. To reduce this cost, we let threads work on diferent cells concurrently. The alternative is to iterate over the cells sequentially, and only the threads whose ray intersects the current cell can make progress while the others are inactive. For a given chunk, we first load the attributes of its 32 cells cooperatively in shared memory.

Then, each thread performs a ray/sphere intersection test [20] against the cells’ bounding spheres, which produces a bit mask with one bit per cell. This part is sequential but relatively inexpensive. Next, each thread iterates over the bits set in its intersection mask and performs the more expensive ray/cell check against the corresponding cell, which need not be the same for all threads. The maximum number of bits set in the mask across threads determines the total number of iterations, often lower than 32 since a ray is unlikely to intersect all cells’ bounding spheres in the chunk. This optimization makes sense in our case because the cells are only a few pixels across and intersection tests are costly. Larger primitives would not benefit as much because more rays intersect them at once, and thus incur lower execution divergence.

The bisector planes are encoded as vectors $\mathbf { p } _ { i } ^ { k } \in \mathbb { R } ^ { 3 }$ , whose direction is the plane normal, oriented away from the site, and with length equal to the distance between the site and the plane.

$$
t _ { i } ^ { - } = \operatorname* { m a x } _ { k } \frac { \| \mathbf { p } _ { i } ^ { k } \| _ { 2 } ^ { 2 } - \left( \mathbf { o } - \mathbf { x } _ { i } \right) \cdot \mathbf { p } _ { i } ^ { k } } { \mathbf { r } \cdot \mathbf { p } _ { i } ^ { k } } ,\tag{35}
$$

$$
t _ { i } ^ { + } = \operatorname* { m i n } _ { k } \frac { \| \mathbf { p } _ { i } ^ { k } \| _ { 2 } ^ { 2 } - \left( \mathbf { o } - \mathbf { x } _ { i } \right) \cdot \mathbf { p } _ { i } ^ { k } } { \mathbf { r } \cdot \mathbf { p } _ { i } ^ { k } } .\tag{36}
$$

If $t _ { i } ^ { - } \geq t _ { i } ^ { + }$ , this indicates the ray does not intersect the cell.

## 5 Optimization and Implementation

## 5.1 Color remapping

The rendered color C in equation 33 is further composited with background pixel values sampled from a trainable envmap at infinity: ${ \bf C } _ { 2 } = { \bf C } + { \bf C } _ { \mathrm { e n v } } T$ , with $T = \prod \alpha _ { i }$ the transmittance. We use an octahedral mapping to parametrize the envmap [8], with a resolu tion of 1024<sup>2</sup>. Then, the color is tone-mapped to the [0, 1] interval: ${ \bf C } _ { 3 } = { \bf C } _ { 2 } / ( { \bf C } _ { 2 } + 1 . 0 )$ , before applying a trainable color correction curve: ${ \bf C } _ { \mathrm { f i n a l } } = \gamma ( { \bf C } _ { 3 } )$ . The color correction curve is a piecewise linear ramp mapping the [0, 1] interval to itself, distinct for each color channel and each camera [53].

## 5.2 Image Losses

Our data term is a weighted combination of a normalized L2 loss, a mask loss, a structural loss and a multi-view loss. We also add an L1 smoothing term on the depth, taken as the first intersection with the implicit surface.

$$
\mathcal { L } _ { \mathrm { R G B } } = \sum | | \mathbf { C } _ { \mathrm { f i n a l } } - \mathbf { C } _ { \mathrm { g t } } | | ^ { 2 } \frac { M } { \operatorname* { m a x } \left( \mathbf { C } _ { \mathrm { g t } } , 0 . 0 1 \right) }\tag{37}
$$

$$
\mathcal { L } _ { \mathrm { M a s k } } = \sum ( T - ( 1 - M ) ) ^ { 2 }\tag{38}
$$

$$
\mathcal { L } _ { \mathrm { D S S I M } } = \sum \bigl ( 1 - \mathrm { S S I M } ( \mathbf { C } _ { \mathrm { f i n a l } } , \mathbf { C } _ { \mathrm { g t } } ) \bigr ) M\tag{39}
$$

$$
\mathcal { L } _ { \mathrm { M V } } = \sum ( 1 - \mathrm { Z N C C } )\tag{40}
$$

$$
\mathcal { L } _ { D } = \sum \left| d - s \mathrm { g } \big ( \bar { d } \big ) \right|\tag{41}
$$

$\mathbf { C } _ { \mathrm { g t } }$ is the ground truth image, � is the mask, � is the transmittance, ZNCC is the cross-correlation computed on pairs of nearby view points, � is the depth and <sup>¯</sup>� is the averaged depth of the neighboring pixels. The sums are applied to all the pixels of the cameras in the mini-batch. We use coarse-to-fine supervision from the ground truth images, with four levels of detail in total. Our complete optimization objective is a weighted combination of all the losses:

$$
\begin{array} { r l } & { \mathcal { L } = \lambda _ { \mathrm { R G B } } \mathcal { L } _ { \mathrm { R G B } } + \lambda _ { \mathrm { M a s k } } \mathcal { L } _ { \mathrm { M a s k } } + \lambda _ { \mathrm { D S S I M } } \mathcal { L } _ { \mathrm { D S S I M } } } \\ & { \quad \quad + \lambda _ { \mathrm { M V } } \mathcal { L } _ { \mathrm { M V } } + \lambda _ { D } \mathcal { L } _ { D } + \lambda _ { \mathrm { C V T } } \mathcal { L } _ { \mathrm { C V T } } } \\ & { \quad \quad + \lambda _ { \mathrm { S D F } } \mathcal { L } _ { \mathrm { S D F } } + \lambda _ { \mathrm { n } } \mathcal { L } _ { \mathrm { n } } + \lambda _ { \mathrm { E i k } } \mathcal { L } _ { \mathrm { E i k } } } \\ & { \quad \quad \quad + \lambda _ { \mathrm { p r o b e s } } \mathcal { L } _ { \mathrm { p r o b e s } } + \lambda _ { \mathrm { F } ^ { s } } \mathcal { L } _ { \mathrm { F } ^ { s } } . } \end{array}\tag{42}
$$

We propose to additionally weigh the image-loss gradients perprimitive during the backward pass based on their projected size. This re-weighting makes the training seamless when the resolution of the training images is switched to the next level of detail in the coarse-to-fine pyramid.

$$
r _ { \mathrm { p r o j } } = r f _ { x } / z\tag{43}
$$

$$
g _ { \mathrm { m o d } } = 1 / r _ { \mathrm { p r o j } } ^ { 2 }\tag{44}
$$

� is the radius of the cell’s bounding sphere, $f _ { x }$ is the focal length and � is the depth of the site for the current viewpoint. $g _ { \mathrm { m o d } }$ is the per-primitive weight.

## 5.3 Initialization and Densification

Initially, we spread the sites in the scene bounding box based on a uniform distribution and apply 10 iterations of Lloyd’s relaxation to uniformize the cells (� = 0 in equation 1). The SDF is initialized from a sphere, while the rest of the learnable features are randomized based on a normal distribution.

We start with 200K sites, and increase their number every 10 iterations until we reach 4M sites by following a geometric progression. To place new sites, we pick a random viewpoint and render a depth map and a feature map. The feature map is the �-blending of the sites’ spatial features F<sup>�</sup>. A new site is placed at the depth sampled from a random valid pixel in the depth map, and its feature vector F<sup>�</sup> is sampled from the same pixel in the feature map.

The nearest-neighbors list is rebuilt each time new points are added, while the Voronoi diagram is recomputed more frequently: at each iteration in the beginning, and every five iterations towards the end. We train for 10K iterations, with four training images per mini-batch. New points are added starting from the 400th iteration, until the last 400 iterations. For the last 400 iterations, we replace the planar-based ZNCC with a nonplanar version to avoid suppressing fine-scale details. More details regarding the multi-view loss and the weight scheduling are given in the supplementary material.

## 6 Experimental validation

All experiments in this section used a workstation equipped with an RTX 5090 GPU and an Intel Core Ultra 7 265K with 20 cores to measure timings, with the exception of table 2 where we report each method’s timing and corresponding hardware configuration. Best, second and third best results are highlighted in the tables.

## 6.1 Multi-view Reconstruction

Evaluation on DTU. We evaluate the multi-view reconstruction accuracy of our approach on 15 scenes from the DTU dataset [1]. Quantitative results are reported in table 2 and qualitative results are shown in figure 10. We achieve state-of-the-art geometric accuracy, tied with GGGS [63] and BlobsToSpokes [13], while requiring a training time of approximately 2.5 minutes. Our approach reproduces high-frequency geometric details that GGGS and BlobsToSpokes miss, this is particularly noticeable in the close-ups on scans 24 and 106 shown at the top of fig. 10. We hypothesize the qualitative improvement does not lead to quantitative improvement due to noise in the ground truth point clouds and down-sampling of the input reconstructions by the oficial evaluation script.

For consistency with the 3DGS-like approaches [21], we discard vertices that project outside of the training masks as well as hidden vertices before Chamfer distance evaluation, since closed surfaces tend to be penalized under the DTU evaluation protocol.

Evaluation on BlendedMVS. We also compare our approach on 8 scenes from the BlendedMVS [59] dataset in table 1, with qualitative results presented in figure 11. We measure Chamfer distances as point-to-mesh distances in the normalized reference frame used for training, with the same evaluation protocol on this dataset as ProbeSDF [53]. The images were cropped before running PGSR and GGGS according to the 2DGS protocol [21], since they require a projection center in the middle of the images. We achieve high geometric accuracy using the same hyperparameters as on DTU, with GGGS slightly ahead quantitatively but with noticeable noise (see figure 11). PGSR fails to produce a mesh in two instances due to ambiguous white backgrounds which caused the GPU to run out of memory during mesh extraction.

Additional evaluations of photometric accuracy are provided in the supplementary material.

## 6.2 Training Speed Analysis on DTU

We evaluate in table 3 the two main configuration hyperparameters of our approach: the number of sites and the number of training iterations. Reducing either of these produces a small decrease in accuracy but lowers training time substantially. Qualitative results are shown in figure 12.

Table 1: Chamfer distance on BlendedMVS with 4M sites.
<table><tr><td>Method</td><td>Mean</td><td>Dog</td><td>Bear</td><td>Clock</td><td>Durian</td><td>Man</td><td>Sculpture</td><td>Stone</td><td>Jade</td></tr><tr><td>NeuS2 [55]</td><td>2.93</td><td>2.78</td><td>2.71</td><td>2.63</td><td>4.23</td><td>2.25</td><td>2.50</td><td>1.91</td><td>4.43</td></tr><tr><td>Voxurf [57]</td><td>2.64</td><td>2.28</td><td>2.20</td><td>1.88</td><td>2.98</td><td>2.11</td><td>1.75</td><td>3.96</td><td>3.99</td></tr><tr><td>ProbeSDF [53]</td><td>2.21</td><td>2.24</td><td>2.03</td><td>1.69</td><td>3.26</td><td>1.81</td><td>1.61</td><td>1.36</td><td>3.71</td></tr><tr><td>PGSR [7]</td><td>1.56</td><td>1.21</td><td>1.20</td><td>N/A</td><td>2.15</td><td>1.39</td><td>1.81</td><td>1.58</td><td>N/A</td></tr><tr><td>GGGS [63]</td><td>0.88</td><td>0.55</td><td>0.37</td><td>0.71</td><td>0.93</td><td>0.85</td><td>0.99</td><td>0.40</td><td>2.28</td></tr><tr><td>CVTSplat</td><td>1.13</td><td>0.91</td><td>0.49</td><td>0.97</td><td>1.44</td><td>0.91</td><td>1.21</td><td>0.60</td><td>2.53</td></tr></table>

![](images/53c8b80db2217183e042747b03bbf06da9508d35a5772074ac078922637e8276.jpg)  
Figure 10: Qualitative comparison on DTU against GGGS [63] and PGSR [7]. The ground truth meshes were obtained by applying Poisson reconstruction [25] on the ground truth point clouds. Our approach recovers finer details, especially visible on scans 24, 106 and 122.

Next, using the chunked rasterizer described in section 4.3.1 has the largest impact on runtime: the optimization takes 147s with it, compared to 280s using a baseline that assigns a single GPU thread per pixel similarly to 3DGS [26]. This almost gives a 2× speedup on the end-to-end optimization. Both rasterizers achieve nearly the same Chamfer distance and the remaining discrepancy can likely be attributed to diferences in floating-point arithmetic.

![](images/7a302ffee9e04831f6f52808f566c611cec44d9e9bf7caee9664b216b28f646a.jpg)  
Figure 11: Qualitative comparison on BlendedMVS against GGGS [63] and PGSR [7]. Errors are mesured relative to the unit bounding box enclosing the objects. Our approach produces smoother geometry than GGGS (see close-ups on man and clock, heatmaps on jade), although some concavities are better reconstructed with GGGS (clock, jade, dog). PGSR fails on two examples and is less accurate overall.

Table 2: Chamfer distance on the DTU benchmark in millimeters. Lower is better. Best, second-best, and third-best results are highlighted.
<table><tr><td>Method</td><td></td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td><td>Runtime</td><td>GPU</td></tr><tr><td rowspan="8">Ipicet</td><td>VolSDF [60]</td><td>1.14</td><td>1.26</td><td>0.81</td><td>0.49</td><td>1.25</td><td>0.70</td><td>0.72</td><td>1.29</td><td>1.18</td><td>0.70</td><td>0.66</td><td>1.08</td><td>0.42</td><td>0.61</td><td>0.55</td><td>0.86</td><td>&gt;12 h</td><td>RTX 2080 Ti</td></tr><tr><td>NeuS [54]</td><td>1.00</td><td>1.37</td><td>0.93</td><td>0.43</td><td>1.10</td><td>0.65</td><td>0.57</td><td>1.48</td><td>1.09</td><td>0.83</td><td>0.52</td><td>1.20</td><td>0.35</td><td>0.49</td><td>0.54</td><td>0.84</td><td>8h</td><td>RTX 2080 Ti</td></tr><tr><td>Neuralangelo [33]</td><td>0.37</td><td>0.72</td><td>0.35</td><td>0.35</td><td>0.87</td><td>0.54</td><td>0.53</td><td>1.29</td><td>0.97</td><td>0.73</td><td>0.47</td><td>0.74</td><td>0.32</td><td>0.41</td><td>0.43</td><td>0.61</td><td>&gt;128h</td><td>V100</td></tr><tr><td>NeuS2 [55]</td><td>0.56</td><td>0.76</td><td>0.49</td><td>0.37</td><td>0.92</td><td>0.71</td><td>0.76</td><td>1.22</td><td>1.08</td><td>0.63</td><td>0.59</td><td>0.89</td><td>0.40</td><td>0.48</td><td>0.55</td><td>0.70</td><td>5 min</td><td>RTX 3090</td></tr><tr><td>Voxurf [57]</td><td>0.91</td><td>0.73</td><td>0.45</td><td>0.34</td><td>0.99</td><td>0.66</td><td>0.83</td><td>1.36</td><td>1.31</td><td>0.78</td><td>0.53</td><td>1.12</td><td>0.37</td><td>0.53</td><td>0.51</td><td>0.76</td><td>15 min</td><td>A100</td></tr><tr><td>ProbeSDF [53]</td><td>0.65</td><td>0.74</td><td>0.34</td><td>0.34</td><td>1.02</td><td>0.71</td><td>0.62</td><td>1.34</td><td>0.94</td><td>0.70</td><td>0.53</td><td>0.96</td><td>0.36</td><td>0.45</td><td>0.47</td><td>0.68</td><td>2.5 min</td><td>RTX 3090</td></tr><tr><td>Geo-NeuS [12]</td><td>0.38</td><td>0.54</td><td>0.34</td><td>0.36</td><td>0.80</td><td>0.45</td><td>0.41</td><td>1.03</td><td>0.84</td><td>0.55</td><td>0.46</td><td>0.47</td><td>0.29</td><td>0.36</td><td>0.35</td><td>0.51</td><td>&gt; 12 h</td><td>RTX 3090</td></tr><tr><td>GeoSVR [32]</td><td>0.32</td><td>0.51</td><td>0.30</td><td>0.33</td><td>0.71</td><td>0.48</td><td>0.42</td><td>1.03</td><td>0.62</td><td>0.56</td><td>0.33</td><td>0.46</td><td>0.30</td><td>0.34</td><td>0.32</td><td>0.47</td><td>50 min</td><td>RTX 3090 Ti</td></tr><tr><td rowspan="8">Expict</td><td>SuGaR [19]</td><td>1.47</td><td>1.33</td><td>1.13</td><td>0.61</td><td>2.25</td><td>1.71</td><td>1.15</td><td>1.63</td><td>1.62</td><td>1.07</td><td>0.79</td><td>2.45</td><td>0.98</td><td>0.88</td><td>0.79</td><td>1.33</td><td>1h</td><td>RTX 4090</td></tr><tr><td>2DGS [21]</td><td>0.48</td><td>0.91</td><td>0.39</td><td>0.39</td><td>1.01</td><td>0.83</td><td>0.81</td><td>1.36</td><td>1.27</td><td>0.76</td><td>0.70</td><td>1.40</td><td>0.40</td><td>0.76</td><td>0.52</td><td>0.80</td><td>12 min</td><td>RTX 4090</td></tr><tr><td>GOF [61]</td><td>0.50</td><td>0.82</td><td>0.37</td><td>0.37</td><td>1.12</td><td>0.74</td><td>0.73</td><td>1.18</td><td>1.29</td><td>0.68</td><td>0.77</td><td>0.90</td><td>0.42</td><td>0.66</td><td>0.49</td><td>0.74</td><td>50 min</td><td>RTX 4090</td></tr><tr><td>RaDe-GS [62]</td><td>0.46</td><td>0.73</td><td>0.33</td><td>0.38</td><td>0.79</td><td>0.75</td><td>0.76</td><td>1.19</td><td>1.22</td><td>0.62</td><td>0.70</td><td>0.78</td><td>0.36</td><td>0.68</td><td>0.47</td><td>0.68</td><td>8 min</td><td>RTX 4090</td></tr><tr><td>MILo [18]</td><td>0.43</td><td>0.74</td><td>0.34</td><td>0.37</td><td>0.80</td><td>0.74</td><td>0.70</td><td>1.21</td><td>1.22</td><td>0.66</td><td>0.62</td><td>0.80</td><td>0.37</td><td>0.76</td><td>0.48</td><td>0.68</td><td>25 min</td><td>RTX 4090</td></tr><tr><td>PGSR [7]</td><td>0.34</td><td>0.54</td><td>0.44</td><td>0.37</td><td>0.78</td><td>0.57</td><td>0.49</td><td>1.06</td><td>0.63</td><td>0.59</td><td>0.47</td><td>0.50</td><td>0.30</td><td>0.37</td><td>0.34</td><td>0.52</td><td>30 min</td><td>RTX 4090</td></tr><tr><td>GGGS [63]</td><td>0.38</td><td>0.50</td><td>0.27</td><td>0.31</td><td>0.80</td><td>0.43</td><td>0.42</td><td>1.04</td><td>0.64</td><td>0.52</td><td>0.31</td><td>0.56</td><td>0.30</td><td>0.31</td><td>0.33</td><td>0.47</td><td>15 min</td><td>RTX 4090</td></tr><tr><td>BlobsToSpokes [13]</td><td>0.39</td><td>0.50</td><td>0.26</td><td>0.32</td><td>0.80</td><td>0.46</td><td>0.44</td><td>1.05</td><td>0.65</td><td>0.52</td><td>0.34</td><td>0.58</td><td>0.32</td><td>0.33</td><td>0.34</td><td>0.49</td><td>20 min</td><td>RTX 4090</td></tr><tr><td colspan="2">CVTSplat</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.76</td><td>0.45</td><td>0.46</td><td>0.97</td><td>0.75</td><td>0.49</td><td>0.38</td><td>0.58</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td><td>2.5 min</td><td>RTX 5090</td></tr></table>

Notes: Timings are approximate and are not hardware-normalized. GGGS and BlobsToSpokes use the 20k iterations setting.

Table 3: Training speed analysis on DTU.
<table><tr><td>Method (#Sites, #Optim. steps)</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td><td></td><td>Time (s)</td></tr><tr><td>1M Sites, 10K</td><td>0.38</td><td>0.54</td><td>0.30</td><td>0.30</td><td>0.82</td><td>0.50</td><td></td><td>0.47 1.02</td><td></td><td>0.75</td><td>0.53</td><td>0.33</td><td>0.66</td><td>0.30</td><td>0.34</td><td>0.34</td><td>0.50</td><td>87</td></tr><tr><td>2M Sites, 10K</td><td>0.36</td><td>0.51</td><td>0.30</td><td>0.30</td><td>0.82</td><td>0.48</td><td></td><td>0.47</td><td>1.00</td><td>0.74</td><td>0.51</td><td>0.33</td><td>0.65</td><td>0.30</td><td>0.33</td><td>0.33</td><td>0.50</td><td>110</td></tr><tr><td>4M Sites, 5K</td><td>0.36</td><td>0.58</td><td>0.30</td><td>0.30</td><td>0.75</td><td>0.46</td><td>0.46</td><td>0.97</td><td></td><td>0.74</td><td>0.50</td><td>0.39</td><td>0.75</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.50</td><td>79</td></tr><tr><td>4M Sites, 7.5K</td><td>0.35</td><td>0.50</td><td>0.30</td><td>0.30</td><td>0.75</td><td>0.46</td><td>0.46</td><td>0.97</td><td>0.74</td><td></td><td>0.49 0.39</td><td></td><td>0.65</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td><td>115</td></tr><tr><td>Basic Rasterizer (4M Sites, 10K)</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.76</td><td>0.45</td><td>0.46</td><td>0.97</td><td>0.76</td><td></td><td>0.49 0.41</td><td>0.58</td><td></td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td><td>280</td></tr><tr><td>Chunked Rasterizer (4M Sites, 10K)</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.76</td><td>0.45</td><td>0.46</td><td>0.97</td><td>0.75</td><td></td><td>0.49 0.38</td><td></td><td>0.58</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td><td>147</td></tr></table>

![](images/69f8f3ad76e5f961098b533841768fa2f409f92807a15e3ebd20835f492c0cf9.jpg)  
Figure 12: Test with fewer sites or fewer optimization steps. Top: Using fewer sites produces a predictable loss of geometric details with faster optimization. Bottom: Our model converges after 10K training steps, with marginal improvement beyond 7.5K steps.

## 6.3 Ablations on DTU

We ablate our losses and regularization terms in table 4, ordered by decreasing importance. Figure 13 compares against the base model qualitatively. Training using a linear radiance tone-mapped after rasterization, compared to a typical sigmoid-bounded nonlinear radiance, avoids a failure case on scan 65 due to large patches of saturated pixels, but makes little diference otherwise. Removing the multi-view loss (No ZNCC) or the depth smoothing loss has a strong impact overall. Setting � = 0.05 in the biased CVT loss (eq. 1), instead of � = 0.10 prevents the cells from shrinking suficiently which reduces accuracy but does not create hard failure cases. The reflection probes are important for recovering smooth geometry in specular regions (scan 110, 69). Adding an environment map helps compensating for slight inaccuracies in the supervision masks (scan 83). The rest of the components have lower importance: the SSIM, color correction curves and view probes are added for robustness against inconsistent illumination conditions. Removing one of them does not afect geometry meaningfully. Finishing training with the non planar ZNCC term instead of the planar ZNCC gives a qualitative improvement in fine details (scan 106). Finally, removing the normal smoothing term (eq. 24) slightly improves the Chamfer distance compared to our base model but also introduces artifacts (scan 110). The gradient modulation (eq. 44) does not change the end result but makes training more stable.

Table 4: Ablation study on DTU.
<table><tr><td>Method</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>No Linear Radiance</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.79</td><td>1.78</td><td>0.44</td><td>0.99</td><td>0.84</td><td>0.50</td><td>0.32</td><td>0.59</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.57</td></tr><tr><td>No ZNCC</td><td>0.40</td><td>0.57</td><td>0.33</td><td>0.31</td><td>0.80</td><td>0.52</td><td>0.52</td><td>1.05</td><td>0.83</td><td>0.54</td><td>0.38</td><td>1.07</td><td>0.29</td><td>0.35</td><td>0.32</td><td>0.55</td></tr><tr><td>No Depth Smoothing</td><td>0.44</td><td>0.50</td><td>0.46</td><td>0.29</td><td>0.75</td><td>0.45</td><td>0.46</td><td>0.97</td><td>0.75</td><td>0.50</td><td>0.50</td><td>0.70</td><td>0.29</td><td>0.36</td><td>0.31</td><td>0.52</td></tr><tr><td>Biased CVT (a = 0.05)</td><td>0.36</td><td>0.51</td><td>0.34</td><td>0.30</td><td>0.81</td><td>0.50</td><td>0.48</td><td>1.02</td><td>0.77</td><td>0.52</td><td>0.34</td><td>0.80</td><td>0.30</td><td>0.34</td><td>0.34</td><td>0.52</td></tr><tr><td>No Ref. Probes</td><td>0.35</td><td>0.56</td><td>0.30</td><td>0.29</td><td>0.76</td><td>0.45</td><td>0.50</td><td>0.97</td><td>0.82</td><td>0.49</td><td>0.38</td><td>0.81</td><td>0.30</td><td>0.32</td><td>0.31</td><td>0.51</td></tr><tr><td>No Envmap</td><td>0.35</td><td>0.52</td><td>0.29</td><td>0.29</td><td>0.80</td><td>0.44</td><td>0.44</td><td>1.12</td><td>0.76</td><td>0.53</td><td>0.39</td><td>0.64</td><td>0.28</td><td>0.31</td><td>0.30</td><td>0.50</td></tr><tr><td>Planar ZNCC Only</td><td>0.35</td><td>0.51</td><td>0.30</td><td>0.29</td><td>0.75</td><td>0.45</td><td>0.46</td><td>0.96</td><td>0.75</td><td>0.50</td><td>0.39</td><td>0.60</td><td>0.30</td><td>0.32</td><td>0.31</td><td>0.48</td></tr><tr><td>No Color Correction</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.75</td><td>0.44</td><td>0.46</td><td>0.97</td><td>0.76</td><td>0.50</td><td>0.41</td><td>0.59</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td></tr><tr><td>No View Probes</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.75</td><td>0.47</td><td>0.47</td><td>0.97</td><td>0.68</td><td>0.50</td><td>0.40</td><td>0.58</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td></tr><tr><td>No SSIM</td><td>0.35</td><td>0.51</td><td>0.30</td><td>0.29</td><td>0.75</td><td>0.46</td><td>0.46</td><td>0.97</td><td>0.70</td><td>0.51</td><td>0.32</td><td>0.60</td><td>0.30</td><td>0.33</td><td>0.31</td><td>0.48</td></tr><tr><td>No Gradient Mod.</td><td>0.34</td><td>0.49</td><td>0.29</td><td>0.29</td><td>0.75</td><td>0.44</td><td>0.46</td><td>0.99</td><td>0.79</td><td>0.48</td><td>0.41</td><td>0.52</td><td>0.28</td><td>0.31</td><td>0.30</td><td>0.48</td></tr><tr><td>No Normal Smoothing</td><td>0.35</td><td>0.50</td><td>0.29</td><td>0.29</td><td>0.74</td><td>0.45</td><td>0.49</td><td>0.93</td><td>0.75</td><td>0.46</td><td>0.43</td><td>0.54</td><td>0.29</td><td>0.31</td><td>0.30</td><td>0.47</td></tr><tr><td>Base Model</td><td>0.35</td><td>0.49</td><td>0.30</td><td>0.29</td><td>0.76</td><td>0.45</td><td>0.46</td><td>0.97</td><td>0.75</td><td>0.49</td><td>0.38</td><td>0.58</td><td>0.29</td><td>0.32</td><td>0.31</td><td>0.48</td></tr></table>

![](images/d72c2008ebafea2d98b382d2981fae763a2f1a79abd56da19b6333ddf256ae5a.jpg)  
Figure 13: We remove one loss on each example to ablate our model. Bottom: The multi-view (ZNCC), depth and normal losses as well as the reflection probes are important for geometric accuracy. Top: Training with a linear radiance avoids a failure case on scan 65. Not replacing the planar ZNCC with a nonplanar variant in the last phase of training suppresses fine details (scan 106). The envmap alleviates silhouette artifacts from inaccurate masks (scan 83).

## 6.4 KNN Eficiency

We measure the speed of our KNN (K=32) implementation against five baselines on the 4M points DTU checkpoints and present the results in table 5. For our approach, we distinguish between the case where the ball query search radius is already known (w. warmup) compared to when it is unknown which takes approximately 3× longer. In practice, most of the sites can rely on the search radius used in the KNN lookup of the previous training iteration.

Table 5: KNN timings (ms) on our DTU point clouds with 4M points. The \* symbol denotes CPU operations.
<table><tr><td></td><td>CVTSplat with/without warmup</td><td>RestrictedVoronoi [4]</td><td>CuPy-KNN [23]</td><td>Torch-kdtree [16]</td><td>Sklearn [43]</td><td>TrueKNN [41]</td></tr><tr><td>Build Tree</td><td>1.3</td><td>&lt;1</td><td>17.3</td><td>6742*</td><td>1925*</td><td>-</td></tr><tr><td>Query Tree</td><td>13.7 / 46.8</td><td>97.9</td><td>720</td><td>1026</td><td>25630*</td><td></td></tr><tr><td>Total</td><td>15.0 / 48.1</td><td>97.9</td><td>737</td><td>7768</td><td>27556</td><td>28543</td></tr></table>

MeshlessVoronoi [46] and RestrictedVoronoi [4] use the same strategy so we only discuss [4] which was slightly faster. The input points are partitioned on a dense voxel grid as a pre-process. Then, each query point visits the voxels in concentric rings and keeps track of the KNN using a heap in shared memory. Importantly, we doubled their hard-coded maximum search radius to accommodate our adaptive point distributions, otherwise these baselines fail to gather all requested neighbors.

CuPy-KNN [23] is a re-implementation of [24], based on a BVH and using a heap stored in the registers. Torch-kdtree [16] works similarly to Cupy-KNN but builds its acceleration structure on the CPU. Overall, the heap-based approaches are much less eficient than our gather-then-sort approach, even more so when temporal consistency is leveraged. TrueKNN [41] relies on the RT cores of the GPU to benefit from hardware-accelerated BVH traversal. This strategy is efective for uniform point distributions but under performs on our point clouds since several retries are needed for low-density regions. We also include scikit-learn [43] as a CPU reference.

## 6.5 Voronoi and Delaunay Eficiency

Table 6: Voronoi diagram construction timings (ms) on our trained DTU point clouds (4M points).

<table><tr><td></td><td>(Ours)</td><td>[46]</td><td>CVTSplat MeshlessVoronoi RestrictedVoronoi Paragram [4]</td><td>[49]</td><td>gDel3D [5]</td><td>RadFoam [15]</td></tr><tr><td>KNN</td><td>15.0</td><td>126.6</td><td>97.9</td><td></td><td>-</td><td></td></tr><tr><td>Adjacency Centroids</td><td>11.9 3.0</td><td>44.5</td><td>40.5</td><td>275.5</td><td>617.4</td><td>18995</td></tr><tr><td>Total</td><td>29.9</td><td>171.0</td><td>138.4</td><td>275.5</td><td>617.4</td><td>18995</td></tr></table>

![](images/ee53474585b16d1ddda52d9bd714a120f8e61b630b53606845cc0a96e535ba20.jpg)

![](images/5a2ea1da84bd02ee17171917e597d88e7eb7fe4ed655de6e1a4381c86c12e493.jpg)  
Figure 14: Timings for Voronoi diagram construction, evaluated on DTU checkpoints saved every 500 optimization steps. Reported numbers include KNN and centroids computations if applicable.

We compare the speed of our algorithm for Voronoi diagram construction against five GPU baselines in table 6 and in figure 14, on the 4M points DTU checkpoints. Paragram [49], gDel3D [5] and RadiantFoam [15] only output the Voronoi adjacency graph. Closer to our approach, MeshlessVoronoi [46] and RestrictedVoronoi [4] produce a list of KNN and also compute cell centroids. We ran both of them with single-precision arithmetic and K=32 neighbors, equivalently to our setting. In the case of RestrictedVoronoi [4], we only report the first round of KNN gathering which corresponds to what we are doing. Subsequent retries with larger K for incomplete cells increase their runtime by about 1.5×. The reported timing for our approach assumes the KNN search radius is known, to match the typical case encountered during optimization. Overall, our approach is consistently 5× faster than [46] and 10× faster than [49] in this context, and scales linearly.

Table 7: Mesh extraction timings on DTU.
<table><tr><td></td><td>Adjacency → Del. Marching Tet.</td><td></td><td>Total</td></tr><tr><td>All cells</td><td>11.5 ms</td><td>4.0 ms</td><td>15.5 ms</td></tr><tr><td>Zero-crossing only</td><td>4.5 ms</td><td>0.6 ms</td><td>5.1 ms</td></tr></table>

For mesh extraction, we convert the adjacency graph into a Delaunay triangulation (sec. 3.4) and subsequently apply marching tetrahedra on the smoothed SDF ${ \hat { f } } .$ It is suficient to only generate the tetrahedra that cross the zero level set, which divides the total mesh extraction time by 3× according to table 7. The zero level set is extracted as a triangle soup for real-time visualization during training. The duplicate vertices are merged on the CPU before outputting the final mesh on disk, which takes two seconds in total.

## 7 Limitations and Future Work

A limitation of our Voronoi diagram construction algorithm is its approximate nature, which makes it unsuitable for applications requiring exact adjacencies. Iterative retries with a larger number of neighbors in the spirit of Basselin et al. [4] could recover the missing adjacencies. A fallback to double precision could also improve robustness in degenerate configurations.

As evaluated in the supplementary material, we find the proportion of incomplete cells to be low in practice for our choice of hyperparameters. More aggressive adaptivity could be balanced using a larger number of neighbors but at an increased cost. Deriving theoretical upper bounds on the required number of neighbors based on cell regularity [22] could be useful but may be inapplicable in our case since we make no guarantee on sites spacing.

Extending our adaptive discretization to open scenes is an interesting direction for future research. The main obstacles are our naive sphere initialization and the potential lack of scalability for representing geometric and photometric details over large spatial extents. The former could be addressed using monocular priors [32] and the latter could benefit from adaptive densification based on image content [15, 26], or primitives with greater expressivity [50].

## 8 Conclusion

We have presented CVTSplat, an approach for multi-view reconstruction combining adaptive Voronoi tessellations with diferentiable implicit surface modeling. Our centroidal regularization plays a central role: beyond producing well-shaped cells, it enables us to specialize several components of the pipeline for eficiency. In particular, we introduced a Voronoi diagram construction algorithm that exploits the regularity of the tessellation. We further tailored neighbor queries, centroid computation, rasterization and mesh ex traction to our representation. Together, these developments yield a highly eficient pipeline that achieves geometric accuracy competitive with the state-of-the-art on two object-level reconstruction benchmarks.

## Acknowledgments

This work was partly supported by AMEDJP26wm0625404 and JSPS/KAKENHI JP26K14883.

## References

[1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. 2016. Large-Scale Data for Multiple-View Stereopsis. International Journal ofComputer Vision (2016), 1–16.

[2] Gupta Anupam and Sleator Danny. 2016. Seidel’s LP algorithm lecture. https: //www.cs.cmu.edu/\~15451-s16/lectures/lec15-16-v2.pdf.

[3] Jonathan T Barron. 2021. Squareplus: A softplus-like algebraic rectifier. arXiv preprint arXiv:2112.11687 (2021).

[4] Justine Basselin, Laurent Alonso, Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, and Bruno Lévy. 2021. Restricted power diagrams on the GPU. In Computer Graphics Forum, Vol. 40. Wiley Online Library, 1–12.

[5] Thanh-Tung Cao, Ashwin Nanjappa, Mingcen Gao, and Tiow-Seng Tan. 2014. A GPU Accelerated Algorithm for 3D Delaunay Triangulation. In Proceedings of the 18th Meeting ofthe ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games (I3D ’14). Association for Computing Machinery, New York, NY, USA, 47–54. doi:10.1145/2556700.2556710

[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. Tensorf: Tensorial radiance fields. In European conference on computer vision. Springer, 333–350.

[7] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. 2024. Pgsr: Planar-based gaussian splatting for eficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics 31, 9 (2024), 6100–6111.

[8] Zina H Cigolle, Sam Donow, Daniel Evangelakos, Michael Mara, Morgan McGuire, and Quirin Meyer. 2014. A survey of eficient representations for independent unit vectors. Journal of Computer Graphics Techniques 3, 2 (2014).

[9] Qiang Du, Vance Faber, and Max D. Gunzburger. 1999. Centroidal Voronoi Tessellations: Applications and Algorithms. SIAM Rev. 41 (1999), 637–676.

[10] Tom Duf, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, and Ryusuke Villemin. 2017. Building an orthonormal basis, revisited. J. Comput. Graph. Tech 6, 1 (2017).

[11] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. 2022. Plenoxels: Radiance fields without neural networks. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5491–5500.

[12] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wenbing Tao. 2022. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruc tion. Advances in Neural Information Processing Systems 35 (2022), 3403–3416.

[13] Diego Gomez, Antoine Guédon, Nissim Maruani, Bingchen Gong, and Maks Ovsjanikov. 2026. From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians. arXiv preprint arXiv:2604.07337 (2026).

[14] Shrisudhan Govindarajan, Daniel Rebain, Dor Verbin, Kwang Moo Yi, Anish Prabhu, and Andrea Tagliasacchi. 2026. Power Foam: Unifying Real-Time Difer entiable Ray Tracing and Rasterization. arXiv preprint arXiv:2604.24994 (2026).

[15] Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. 2025. Radiant foam: Real-time diferentiable ray tracing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 4135–4145.

[16] Thomas Grandits. [n. d.]. torch\_kdtree. https://github.com/thomgrand/torch\_ kdtree/tree/main\_torch. Software repository, main\_torch branch. Accessed 25 September 2026.

[17] Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. 2020. Implicit geometric regularization for learning shapes. arXiv preprint arXiv:2002.10099 (2020).

[18] Antoine Guédon, Diego Gomez, Nissim Maruani, Bingchen Gong, George Drettakis, and Maks Ovsjanikov. 2025. Milo: Mesh-in-the-loop gaussian splatting for detailed and eficient surface reconstruction. ACM Transactions on Graphics (TOG) 44, 6 (2025), 1–15.

[19] Antoine Guédon and Vincent Lepetit. 2024. Sugar: Surface-aligned gaussian splatting for eficient 3d mesh reconstruction and high-quality mesh rendering. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5354–5363.

[20] Eric Haines, Johannes Günther, and Tomas Akenine-Möller. 2019. Precision improvements for ray/sphere intersection. In Ray Tracing Gems: High-Quality and Real-Time Rendering with DXR and Other APIs. Springer, 87–94.

[21] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery. doi:10.1145/ 3641519.3657428

[22] Benoît Hudson, Gary L. Miller, and Todd Phillips. 2006. Sparse Voronoi Refinement. In International Meshing Roundtable Conference.

[23] Felix Igelbrink. [n. d.]. CuPy-KNN. https://github.com/mortacious/cupy-knn. Software repository. Accessed 25 September 2026.

[24] Johannes Jakob and Michael Guthe. 2021. Optimizing LBVH-Construction and Hierarchy-Traversal to accelerate kNN Queries on Point Clouds using the GPU. In Computer Graphics Forum, Vol. 40. Wiley Online Library, 124–137.

[25] Michael Kazhdan and Hugues Hoppe. 2013. Screened poisson surface reconstruction. ACM Transactions on Graphics (ToG) 32, 3 (2013), 1–13.

[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

[27] Dan Koschier, Jan Bender, Barbara Solenthaler, and Matthias Teschner. 2019. SPH Eurographics Tutorial. https://sph-tutorial.physics-simulation.org/.

[28] Dan Koschier, Jan Bender, Barbara Solenthaler, and Matthias Teschner. 2020. Smoothed particle hydrodynamics techniques for the physics based simulation of fluids and solids. arXiv preprint arXiv:2009.06944 (2020).

[29] Jonas Kulhanek and Torsten Sattler. 2023. Tetra-nerf: Representing neural radiance fields using tetrahedra. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 18412–18423.

[30] Christian Lauterbach, Michael Garland, Shubhabrata Sengupta, David Luebke, and Dinesh Manocha. 2009. Fast BVH construction on GPUs. In Computer graphics forum, Vol. 28. Wiley Online Library, 375–384.

[31] Bruno Lévy. 2026. Geogram: A Programming Library with Geometric Algorithms. https://github.com/BrunoLevy/geogram

[32] Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, and Lin Gu. 2026. Geosvr: Taming sparse voxels for geometrically accurate surface reconstruction. Advances in Neural Information Processing Systems 38 (2026), 108809–108837.

[33] Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-fidelity neural surface reconstruction. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 8456–8465.

[34] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. 2020. Neural sparse voxel fields. Advances in Neural Information Processing Systems 33 (2020), 15651–15663.

[35] Stuart Lloyd. 1982. Least squares quantization in PCM. IEEE transactions on information theory 28, 2 (1982), 129–137.

[36] Alexander Mai, Trevor Hedstrom, George Kopanas, Janne Kontkanen, Falko Kuester, and Jonathan T Barron. 2026. Radiance meshes for volumetric recon struction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8267–8277.

[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

[38] Bailey Miller, Hanyu Chen, Alice Lai, and Ioannis Gkioulekas. 2024. Objects as volumes: A stochastic geometry view of opaque solids. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 87–97.

[39] Brian Mirtich. 1996. Fast and accurate computation of polyhedral mass properties. Journal of graphics tools 1, 2 (1996), 31–50.

[40] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41, 4 (2022), 1–15.

[41] Vani Nagarajan, Durga Mandarapu, and Milind Kulkarni. 2023. Rt-knns unbound: Using rt cores to accelerate unrestricted neighbor search. In Proceedings ofthe 37th International Conference on Supercomputing. 289–300.

[42] NVIDIA Corporation. 2026. cub::WarpMergeSort. CUDA Core Compute Libraries documentation. Accessed: 2026-08-26. https://nvidia.github.io/cccl/unstable cub/api/classcub\_1\_1WarpMergeSort.html

[43] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. Journal ofMachine Learning Research 12 (2011), 2825–2830.

[44] Daniel J Price. 2012. Smoothed particle hydrodynamics and magnetohydrodynamics. J. Comput. Phys. 231, 3 (2012), 759–794.

[45] Lukas Radl, Felix Windisch, Thomas Deixelberger, Jozef Hladky, Michael Steiner, Dieter Schmalstieg, and Markus Steinberger. 2025. SOF: Sorted Opacity Fields for Fast Unbounded Surface Rconstruction. In SIGGRAPH Asia Conference Proceedings.

[46] Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, and Bruno Lévy. 2018. Meshless Voronoi on the GPU. ACM Transactions on Graphics (TOG) 37, 6 (2018), 1–12.

[47] Antonella Rech, Nicola Conci, and Nicola Garau. 2026. SDFoam: Signed-Distance Foam for explicit surface reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 300–308.

[48] Raimund Seidel. 1991. Small-dimensional linear programming and convex hulls made easy. Discrete & Computational Geometry 6, 3 (1991), 423–434.

[49] Bernardo Taveira, Carl Lindström, Maryam Fatemi, Lars Hammarstrand, and Fredrik Kahl. 2026. Scalable GPU Construction of 3D Voronoi and Power Dia grams. arXiv preprint arXiv:2605.06408 (2026).

[50] Bernardo Taveira, Carl Lindström, Joakim Johnander, and Fredrik Kahl. 2026. Diferentiable Voronoi Ray Tracing Beyond Rasterization Speeds. arXiv preprint arXiv:2608.17682 (2026).

[51] The CGAL Project. 2026. CGAL User and Reference Manual (6.2 ed.). CGAL Editorial Board. https://doc.cgal.org/6.2/Manual/packages.htm

[52] Diego Thomas, Briac Toussaint, Jean-Sébastien Franco, and Edmond Boyer. 2025. VortSDF: 3D Modeling with Centroidal Voronoi Tessellation on Signed Distance

Field. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. IEEE, 495–504. doi:10.1109/WACV61041.2025.00058

[53] Briac Toussaint, Diego Thomas, and Jean-Sébastien Franco. 2025. Probesdf: Light field probes for neural surface reconstruction. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 11026–11035.

[54] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689 (2021).

[55] Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. 2023. Neus2: Fast learning of neural implicit surfaces for multiview reconstruction. In 2023 IEEE/CVF international conference on computer vision (ICCV). IEEE, 3272–3283.

[56] Johannes Weidenfeller, Shaofei Wang, Philipp Fürnstahl, and Siyu Tang. 2026. ADELE-Adaptive Delaunay Grids for High-Fidelity Mesh-Native Reconstruction. arXiv preprint arXiv:2609.06723 (2026).

[57] Tong Wu, Jiaqi Wang, Xingang Pan, Xudong Xu, Christian Theobalt, Ziwei Liu, and Dahua Lin. 2023. Voxurf: Voxel-based Eficient and Accurate Neural Surface Reconstruction. In International Conference on Learning Representations (ICLR).

[58] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. 2022. Point-nerf: Point-based neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5428–5438.

[59] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. 2020. Blendedmvs: A large-scale dataset for generalized multiview stereo networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 1787–1796.

[60] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. Advances in neural information processing systems 34 (2021), 4805–4815.

[61] Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024. Gaussian opacity fields: Eficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (ToG) 43, 6 (2024), 1–13.

[62] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiao-Xiao Long, and Ping Tan. 2026. Rade-gs: Rasterizing depth in gaussian splatting. ACM Transactions on Graphics 45, 2 (2026), 1–14.

[63] Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan. 2026. Geometry-Grounded Gaussian Splatting. arXiv preprint arXiv:2601.17835 (2026).

## A Overview

This supplementary document provides additional implementation details, experiments, derivations and proofs. We report the optimization schedule (section B) followed by statistics on our adaptive Voronoi tessellation: study of mesh defects (section C), accuracy of cell moments computation (section D), frequency of incomplete cells and relationship between cell size and SDF (section E). Evaluations of rendering speed, photometric accuracy and model size are available in sections F, G and H respectively. Section I gives details regarding the multi-view loss and section J concludes with proofs omitted from the main paper.

## B Hyperparameters Scheduling

Our optimization schedule is detailed in table 8.

Table 8: Optimization schedule. Array values are linearly interpolated throughout training.

Training phases. 10,000 iterations: 400 initialization, 9200 main training,   
and 400 cleanup. No sites are added during initialization or cleanup.   
Point population. 200k initial sites up to 4M sites. Densification every   
10 iterations.   
Data and optimization. 4 image levels of detail, batch size: 4 images   
per batch using gradient accumulation, learning rate [0.025, 0.001].   
Structure updates. KNN every 10 iterations, Voronoi diagram construc  
tion frequency [1, 2, 5].   
Geometric regularization. $\angle \mathrm { S D F } = [ \theta . 4 , \ \theta . 1 ] , \ \mathcal { L } _ { \mathrm { n } } = [ \theta . \theta 5 , \ \theta . 1 ] ,$   
$\mathcal { L } _ { \mathrm { E i k } } = [ \Theta . \Theta \Theta 5 ] ,$ depth smoothing ${ \mathcal { L } } _ { \mathrm { D } } = [ 0 . 5 ] .$   
CVT Densification. L = 1, � =[0.05, 0.10], � =[0.01, 0.001,   
0.0001].   
Image losses. L<sub>RGB</sub> =[5, 10], L<sub>DSSIM</sub> =[1, 2], L<sub>MV</sub> =[0.1, 10],   
$\mathcal { L } _ { \mathrm { M a s k } } = [ 5 , \ \emptyset . 1 ] .$   
Appearance. NeuS sharpness � =[5, 15, 30, 60, 120], features   
smoothing $\mathcal { L } _ { \mathrm { F } ^ { s } } = [ \Theta . \Theta 5 ]$ , probes smoothing $\mathcal { L } _ { \mathrm { p r o b e s } } = [ \theta . 1 , \ 1 ]$   
1024<sup>2</sup> background envmap, two 32<sup>3</sup> spherical-harmonic probe grids.

## C Mesh Defects Analysis

Table 9 reports the average number of topological errors in our extracted meshes. We count the number of non-manifold edges, i.e. edges having more than two incident triangles, and boundary edges that have a single incident triangle. The implicit surface touches the bounding box walls in some scenes, which creates legitimate boundary edges. Therefore, we only count the ones within 95% of the bounding box that actually correspond to small holes in the extracted meshes. Missing Voronoi adjacencies can prevent the formation of tetrahedra and produce small holes, where a single triangle is missing in general. Non-manifold edges may result from inconsistent tetrahedron selection near degenerate configurations or from incomplete Voronoi neighbor lists. Overall, these defects are infrequent relative to the number of mesh triangles and could potentially be addressed by topological post-processing.

## D Cell Moments Estimation

We evaluate in Figure 15 the accuracy of the quadrature formulas for cell volume and centroid using varying numbers of rays sampled on a Fibonacci sphere lattice. Errors are relative and computed with respect to a reference using 8192 rays. Volume and radius errors are relative to their reference values while the centroid error corresponds to the centroid displacement divided by the reference cell radius. The radius is estimated from the maximum sampled distance to the cell boundary over the set of ray directions. The error curves for the volume and centroid are nearly overlapped and decrease as 1/�, where � is the number of rays. The radius error decreases more slowly, approximately as $1 / \sqrt { N }$

Table 9: Average mesh extraction defects by dataset.
<table><tr><td>Dataset</td><td>Non-manifold edges</td><td>Boundary edges</td><td>Triangles</td></tr><tr><td>DTU</td><td>2.7</td><td>2.3</td><td>6.4M</td></tr><tr><td>BlendedMVS</td><td>2.4</td><td>1.4</td><td>6.7M</td></tr></table>

With our default of 64 rays, the mean errors are 0.79% for the volume, 0.77% for the centroid, and 5.65% for the radius. The cell radius mainly serves to define an approximate bounding sphere during rasterization. A more accurate value could be computed using the maximum distance to the circumcenters of incident tetrahedra (i.e. the Voronoi vertices), but that would require computing the Delaunay triangulation from the adjacency graph every time it is updated which would add some overhead. Our approximate bound ing spheres only clip the furthest corners of the cells, that would not contribute significantly to the volume integral if included.

![](images/36238ede1152b87d9eaa771dbd683a4f45148607ec3d0b8a04f6d472a61aedce.jpg)  
Figure 15: Relative error curves when computing cell volume, centroid and bounding sphere radius using a varying number of ray samples. The volume and centroid error curves are nearly overlapped in the plot.

## E Distribution of cells

Figure 16 presents statistics about the number of Voronoi neighbors and their distribution within the KNN list. This test used 64 nearest neighbors instead of 32 for validation. A small proportion of cells has Voronoi neighbors beyond the 32nd nearest neighbor (≈ 0.1%). Extracting the KNN at the centroid decreases that rate substantially (≈ 0.0006%) but has little importance on multi-view optimization in practice. Incomplete cells are typically located away from the implicit surface and therefore do not participate in volume rendering or mesh extraction. Nevertheless, we use the ofset KNN strategy by default as it provides lower error rates at no additional cost.

![](images/d7cea26ccddb230d2e456319a15510ac5b1ec8ca800a51b76a38e4200ee699b0.jpg)

![](images/5601141a958068d405167b5583dde36698114a67bda94fba46729430806f1e87.jpg)

![](images/81e6c44aed1d9424c8cbc5badce56989e03cd171204465f98847ccedee8a34ad.jpg)

Figure 16: Histograms pooled over all 23 DTU and BlendedMVS scenes (92 million sites). The blue one queries KNN at each site while the red one queries at the cell centroid computed at the previous update. Left: Number of Voronoi neighbors per site. The mean is ≈ 14.2, corresponding to 14 bisector planes contributing to each cell on average. Middle and Right: zero-based index of the last detected Voronoi neighbor in the KNN list, with linear and logarithmic frequency scales respectively. For ≈ 60% of sites, all Voronoi neighbors lie among the first 16 nearest neighbors. The frequency decreases roughly exponentially beyond index 15, with ≈ 0.1% of cells not finding their last Voronoi neighbor within the 32-KNN list with the base strategy. Using shifted ball queries reduces the frequency of incomplete cells by a factor of 170 in comparison (≈ 600 cells in total out of 92M).  
![](images/e3d8b584f9e221cd71ff37e25125fba8a7e162830eb1034304e0ac34519b3a1e.jpg)

![](images/0b646de5670f44f05d1f7841573e366c55088fbdfdcea9102e6f735c89fbecf9.jpg)

![](images/156abd81b07e04819363cab4d97efaa2d42d828d576b98eb2f0e28dd1431731d.jpg)  
Figure 17: Cell-radius statistics averaged over all scenes. Left: the distribution of cell radius � against SDF �. The cyan curve fits the median radius in each distance bin, showing approximately linear growth with $| d | .$ Its similar branches on either side of zero indicate comparable cell size inside and outside the surface at equal distance. Middle: the distribution of $\log _ { 2 } ( r )$ . The red curve fits a power-law tail over $2 ^ { - 8 } \leq r \leq 2 ^ { - 4 }$ , containing 26.2% of sites. Within this range, each doubling of � corresponds to approximately 3.9× fewer sites per histogram bin $( h ( r ) / h ( 2 r ) \approx 3 . 9 )$ . Right: The number of cells decreases very quickly away from the implicit surface.

Figure 17 explores the relationship between cell size and signed distance induced by our adaptive CVT. The bounding box used for training spans the unit cube $[ - 1 , + 1 ] ^ { 3 }$ , the SDF and cell radius are given in this reference frame. We find an approximately afine relationship between cell radius and SDF, with most of the cells located close to the implicit surface.

Figure 18 illustrates the sensitivity of the point distribution to the hyperparameter � in the CVT loss.

Table 10: Average projected cell radius and rendering speed (mean ± standard deviation across scenes).
<table><tr><td>Dataset</td><td>Image size</td><td></td><td>Cell radius (px) Render time (ms)</td><td>FPS</td></tr><tr><td>DTU</td><td> $1 6 0 0 \times 1 2 0 0$ </td><td> $1 . 5 5 \pm 0 . 3 1$ </td><td> $3 . 6 3 \pm 0 . 2 9$ </td><td> $2 7 7 \pm 2 1$ </td></tr><tr><td>BlendedMVS</td><td> $7 6 8 \times 5 7 6$ </td><td> $1 . 2 1 \pm 0 . 1 5$ </td><td> $2 . 4 0 \pm 0 . 2 1$ </td><td> $4 1 9 \pm 3 6$ </td></tr></table>

## F Rendering speed

Table 10 reports the average size of the primitives once projected to the training views. Each primitive is weighted by its volume rendering weight to compute this metric. We also report the time taken to render an image from the dataset, averaged over all viewpoints and scenes. The timings are measured on an RTX 5090 GPU. The renderings are written to an ofscreen bufer so a small additional overhead would be needed in practice to display the result.

![](images/2830e38d4c811427721d75c52eaa94c1d8f490c3db7ddca11d243ff5fa3a5616.jpg)  
Figure 18: Adaptive discretization of a known SDF. Top: 2D slice with random colors and superimposed extracted mesh. Bottom: 2D slice with SDF mapped to hue. We fit the SDF using $N = 1 0 ^ { 5 }$ sites and vary the parameter � in the CVT loss. The mesh approximates the zero level set more accurately for $a > 0 .$ The proportion of cells � with missing bisector planes increases sharply for $a > 0 . 1$ when using 32 Euclidean nearest neighbors. Missing bisector planes can cause cells to clearly overlap for larger values of � (see � = 0.2, center of the horseshoe in the top panel)

## G Photometric Evaluation

We evaluate the photometric accuracy of our approach in table 11, on all available viewpoints and within the silhouette masks eroded by two pixels. All tested methods were trained using all viewpoints so this experiment measures fitting error rather than novel view synthesis fidelity. PGSR [7] and GGGS [63] train on cropped images so we evaluate our method in the same setting for consistency. PGSR and GGGS used half-resolution images on DTU, following their training setup, so we upscale their renderings before evaluation. Our method achieves an image quality comparable to PGSR but slightly below GGGS, with a larger gap on BlendedMVS.

Qualitative comparisons on DTU and BlendedMVS are shown in figures 19 and 20 respectively. We picked viewpoints for which our error is highest compared to GGGS to understand the limitations of our appearance model. The texture is generally well-reproduced in our case but some specular regions are missing due to the SH bases being shared by multiple sites.

Table 11: Masked image metrics on training views.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="2">DTU</td><td>CVTSplat</td><td>30.32</td><td>0.888</td><td>0.147</td></tr><tr><td>GGGS PGSR</td><td>30.76</td><td>0.872</td><td>0.253</td></tr><tr><td rowspan="2"></td><td>CVTSplat</td><td>30.33</td><td>0.866</td><td>0.265</td></tr><tr><td>GGGS</td><td>29.52</td><td>0.894</td><td>0.073</td></tr><tr><td rowspan="2">BMVS</td><td></td><td>32.43</td><td>0.953</td><td>0.046</td></tr><tr><td>PGSR</td><td>30.50</td><td>0.902</td><td>0.108</td></tr></table>

## H Model size

Table 12 presents a comparison of our model in terms of number of primitives and number of trainable parameters, compared to PGSR [7] and GGGS [63]. Despite using more primitives than GGGS, we have a similar total parameter count on DTU, and lower on BlendedMVS. Table 13 provides a breakdown of parameter usage for our approach.

Table 12: Average model size.
<table><tr><td>Dataset</td><td>Method</td><td>Primitives</td><td>Parameters</td></tr><tr><td rowspan="3">BMVS</td><td>CVTSplat</td><td>4M sites</td><td>39.35M</td></tr><tr><td>GGGS</td><td>1.1M Gaussians</td><td>64.21M</td></tr><tr><td>PGSR</td><td>0.5M Gaussians</td><td>29.95M</td></tr><tr><td rowspan="3">DTU</td><td>CVTSplat</td><td>4M sites</td><td>39.35M</td></tr><tr><td>GGGS</td><td>0.6M Gaussians</td><td>36.16M</td></tr><tr><td>PGSR</td><td>0.3M Gaussians</td><td>16.11M</td></tr></table>

Table 13: Parameter count breakdown.
<table><tr><td>Component</td><td>Total</td></tr><tr><td>Site positions (3 × 4M)</td><td>12M</td></tr><tr><td>Site distances (1 × 4M)</td><td>4M</td></tr><tr><td>Site features (4 × 4M)</td><td>16M</td></tr><tr><td>Two SH probes grids</td><td>4.2M</td></tr><tr><td>Environment map</td><td>3.1M</td></tr><tr><td>MLP</td><td>595</td></tr><tr><td>Total</td><td>39.3M</td></tr></table>

Reference  
CVTSplat  
GGGS  
PGSR  
![](images/92814c4a7708ed64f7656a77efaac0f55928a212e69433f6b732dee6d966cda1.jpg)  
Figure 19: Qualitative comparison of renderings on DTU. Our approach can represent high-frequency texture patterns (scan 24, house roof) but misses some specularities compared to GGGS and PGSR.

## I Details on the Multiview Loss

We use a multi-view loss inspired by PGSR [7]. Pairs of views that are close to each other are chosen at random during training (a mini-batch contains 4 views so that pairs can be formed). During the forward pass of volume rendering, the depth of the first zerocrossing of the SDF is recorded for each ray to form a depth map.

## I.1 Depth Computation

For a cell $C _ { i }$ having SDF $\hat { f } _ { i } ,$ position $\mathbf { x } _ { i }$ and SDF gradient $\mathbf { n } _ { i } ,$ we have the following approximation:

$$
{ \hat { f } } ( \mathbf { x } ) = { \hat { f } } _ { i } + ( \mathbf { x } - \mathbf { x } _ { i } ) \cdot \mathbf { n } _ { i }\tag{45}
$$

![](images/cf0c711ddb070978a9f9dec7b24f02ec0bf43cd2e2de4a5270a98b56ab0a2069.jpg)  
Figure 20: Qualitative comparison of renderings on Blended-MVS. PGSR failed to converge on the clock scene.

Then for a ray $\mathbf { r } ( t ) = \mathbf { o } + t \mathbf { v }$ , the zero-crossing is obtained by solving $\hat { f } ( \mathbf { r } ( t ^ { * } ) ) = 0 ;$

$$
{ \hat { f } } _ { i } + \left( \mathbf { o } + t ^ { * } \mathbf { v } - \mathbf { x } _ { i } \right) \cdot \mathbf { n } _ { i } = 0\tag{46}
$$

$$
\Leftrightarrow { \hat { f } } _ { i } + \left( \mathbf { o } - \mathbf { x } _ { i } \right) \cdot \mathbf { n } _ { i } + t ^ { * } \mathbf { v } \cdot \mathbf { n } _ { i } = 0\tag{47}
$$

$$
\Leftrightarrow t ^ { * } = - ( \hat { f } _ { i } + ( \mathbf { o } - \mathbf { x } _ { i } ) \cdot \mathbf { n } _ { i } ) / \mathbf { v } \cdot \mathbf { n } _ { i }\tag{48}
$$

The depth value $t ^ { * }$ is discarded if the point $\mathbf { r } ( t ^ { * } )$ falls outside of the cell, indicating that the implicit surface does not cross it. There may be several zero-crossings along a given ray, so the final depth value is taken as the minimum over the intersected cells. In case no zero-crossing occurs, we use $t ^ { * } = + \infty$

Once a depth map is obtained for both views in the pair, we can compute our multi-view loss. We implement two variants based on the zero-normalized cross-correlation (ZNCC). The first one is similar to prior works [7, 13] based on planar patches, whereas the other removes the planar assumption to recover finer details.

## I.2 Planar ZNCC

Let the pair of views be denoted as $( A , B )$ , with � the reference view and � the neighboring view. We compute a normal vector for each pixel $( u , v )$ in view � using centered finite diferences in the depth map ${ \mathcal { D } } _ { A }$ . Let n be the normal vector expressed in �’s reference frame. Pixels having invalid depth values (+∞) or invalid neighboring pixels are skipped. We check for visibility in view � using a cycle consistency check (same as PGSR [7]):

$$
d _ { A } = \mathcal { D } _ { A } ( u _ { A } , v _ { A } )\tag{49}
$$

$$
u _ { B } , v _ { B } = \pi _ { B } ( \pi _ { A } ^ { - 1 } ( u _ { A } , v _ { A } , d _ { A } ) )\tag{50}
$$

$$
d _ { B } = \mathcal { D } _ { B } ( u _ { B } , v _ { B } )\tag{51}
$$

$$
u _ { A } ^ { \prime } , v _ { A } ^ { \prime } = \pi _ { A } ( \pi _ { B } ^ { - 1 } ( u _ { B } , v _ { B } , d _ { B } ) )\tag{52}
$$

� denotes the projection operator, mapping 3D points to pixel coordinates and $\pi ^ { - 1 }$ its inverse that maps a pixel and depth value back to a 3D point. We consider the pixel to be occluded in view � if $\sqrt { ( u _ { A } - u _ { A } ^ { \prime } ) ^ { 2 } + ( v _ { A } - v _ { A } ^ { \prime } ) ^ { 2 } } > 2$ . The pixel may be rejected if the cycle consistency check fails due to the point projecting outside of �’s field of view, or if the depth in view B is invalid. Lastly, we also check that the normal is not too slanted in either view, with a rejection threshold of 85° between the normal and the viewing ray.

If these checks pass, we compute the ZNCC over a 5 × 5 pixel patch centered on the pixel of interest. Let $x _ { i , j }$ and $y _ { i , j }$ be the grayscale ground truth pixel values for views � and � respectively, sampled over the pixels in the patch. The grayscale values are obtained by taking the average of the RGB channels and multiplying by the mask channel. The patch forms a plane with normal n, placed at depth � along the ray originating from pixel (�, �).

$$
x _ { i , j } = \mathcal { I } _ { A } ( u + i , v + j ) \qquad - 2 \leq i , j \leq 2\tag{53}
$$

$$
d ^ { \prime } = d \frac { \mathbf { n } ^ { T } \mathbf { K } _ { A } ^ { - 1 } ( u , v , 1 ) ^ { T } } { \mathbf { n } ^ { T } \mathbf { K } _ { A } ^ { - 1 } ( u + i , v + j , 1 ) ^ { T } }\tag{54}
$$

$$
y _ { i , j } = \mathcal { J } _ { B } ( \pi _ { B } ( \pi _ { A } ^ { - 1 } ( u + i , v + j , d ^ { \prime } ) ) )\tag{55}
$$

In the case of $x _ { i , j }$ , we can directly sample the pixels at their centers, whereas $y _ { i , j }$ requires sampling image $\mathcal { T } _ { B }$ with bilinear interpolation so that $y _ { i , j }$ is a continuous function of the plane normal and depth. The sampling coordinates for $y _ { i , j }$ are obtained by intersecting the plane with the neighboring rays, which gives an intersection depth $d ^ { \prime } { \mathrm { , } }$ , then back-projecting to 3D space and finally projecting into view B. The $\mathrm { Z N C C } ~ z ~ \in ~ [ - 1 , 1 ]$ is then computed by evaluating the first- and second-order moments of the pixel samples, weighted with a Gaussian kernel.

$$
W = \sum w _ { i , j }\tag{56}
$$

$$
{ \overline { { x } } } = { \frac { 1 } { W } } \sum x _ { i , j } w _ { i , j }
$$

$$
\overline { { y } } = \frac { 1 } { W } \sum y _ { i , j } w _ { i , j }\tag{57}
$$

(58)

$$
\overline { { x ^ { 2 } } } = \frac { 1 } { W } \sum x _ { i , j } ^ { 2 } w _ { i , j }\tag{59}
$$

$$
\overline { { y ^ { 2 } } } = \frac { 1 } { W } \sum y _ { i , j } ^ { 2 } w _ { i , j }
$$

$$
\overline { { x y } } = \frac { 1 } { W } \sum x _ { i , j } y _ { i , j } w _ { i , j }\tag{60}
$$

$$
{ \overline { { x y } } } - { \overline { { x } } } \cdot { \overline { { y } } }\tag{61}
$$

$$
\sqrt { \overline { { { x ^ { 2 } } } } - \overline { { { x } } } ^ { 2 } + \epsilon } \cdot \sqrt { \overline { { { y ^ { 2 } } } } - \overline { { { y } } } ^ { 2 } + \epsilon }\tag{62}
$$

We take $\epsilon = 1 . 0 E { - 4 }$ for numerical stability. The Gaussian kernel is separable $( w _ { i , j } = w _ { i } w _ { j } )$ , with the five weights taken from the 7th row of Pascal’s triangle (1, 6, 15, 20, 15, 6, 1), where the first and last bins are omitted since they contribute very little.

## I.3 Nonplanar ZNCC

Assuming that the surface is locally planar tends to erase fine geometric details so we propose to remove the planar assumption in the ZNCC computation. Doing so is simple, we just use the actual depth of the neighboring pixels in view A, instead of taking the depth of the ray/plane intersection. This amounts to replacing equation 54 by $d ^ { \prime } = \mathcal { D } _ { A } ( u + i , v + j )$ . In practice, we actually use a separate implementation to take advantage of the fact that the pixel values $y _ { i , j }$ can be shared with neighboring pixels in view A, which is not the case in the planar case due to adjacent pixels having diferent plane orientations.

We evaluate mutual visibility over each pixel in view A in a similar manner to the planar case. First, by computing the normal as before and checking its angle with respect to both viewpoints, and second, by applying the same cycle consistency check. Invalid pixels or pixels for which the mutual visibility tests fail are skipped in the computation of the moments in equations 56-61.

## I.4 Loss

Our multi-view loss is $\mathcal { L } _ { \mathrm { M V } } = \left( 1 - \mathrm { Z N C C } \right)$ , summed over all valid pixels in view A. We use the planar ZNCC computation for most of the training since it is more stable, but switch to the nonplanar ZNCC near the end to recover finer details. Note that the grayscale ground truth images and depth maps follow the same coarse-to-fine training regime as the color losses, which means that the small 5 × 5 receptive field will actually cover a larger proportion of the field of view in the beginning.

## I.5 Backward Pass

The ZNCC computation performs the forward and backward passes in the same CUDA kernel for eficiency. It takes as input the depth maps ${ \mathcal { D } } _ { A }$ and ${ \mathcal { D } } _ { B } ,$ , the grayscale images $\mathcal { T } _ { A }$ and $\mathcal { I } _ { B }$ and intrinsic and extrinsic parameters of both cameras. It produces the loss value ${ \mathcal { L } } _ { M V }$ along with depth gradients for view $\mathrm { ~ A ~ } ( \partial \mathcal { L } _ { M V } / \partial \mathcal { D } _ { A } )$ . The depth gradients are further propagated to the SDF values during the backward pass of the volume rendering.

## J Derivations and Proofs

This section gives more details on the computation of the SDF gradient from the neighbors’ SDF (sec. J.1), the computation of the volume rendering integral in closed form (J.2), and the check to filter non-Delaunay tetrahedra when converting the Voronoi adjacencies to a Delaunay triangulation (J.3).

## J.1 SDF Gradient Computation

We summarize the expressions used to define the smoothed SDF ${ \hat { f } } = G ( f )$ and show derivations for its gradient $\mathbf { n } = \nabla _ { \mathbf { x } } G ( \hat { f } )$ . The smoothing kernel � is defined by:

$$
G ( u ) _ { i } = \sum _ { j \in { \cal N } ( i ) } u _ { j } w _ { i j } / \sum _ { j \in { \cal N } ( i ) } w _ { i j }\tag{63}
$$

$$
w _ { i j } = \mathscr { V } _ { j } \exp \left( { - \frac { | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } R _ { i } ^ { 2 } } } \right)\tag{64}
$$

$$
R _ { i } = \operatorname* { m a x } _ { j \in N ( i ) } ( | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | _ { 2 } ) .\tag{65}
$$

For computation of the SDF gradient, we consider that the neighborhood $N ( i )$ , the cell volumes $\mathcal { V } _ { j }$ and the kernel radius $R _ { i }$ are constants. Let $\begin{array} { r } { W _ { i } \ = \ \sum _ { j \in N ( i ) } w _ { i j } } \end{array}$ be an intermediate variable to simplify the notation.

$$
\nabla _ { \mathbf { x } _ { i } } w _ { i j } = - \frac { \mathbf { x } _ { i } - \mathbf { x } _ { j } } { \sigma ^ { 2 } R _ { i } ^ { 2 } } w _ { i j }\tag{66}
$$

$$
\nabla _ { \mathbf { x } _ { i } } W _ { i } = \sum \nabla _ { \mathbf { x } _ { i } } w _ { i j }\tag{67}
$$

$$
\mathbf { n } _ { i } = \nabla _ { \mathbf { x } _ { i } } G ( \hat { f } ) _ { i }\tag{68}
$$

$$
= \sum _ { j \in N ( i ) } \hat { f } _ { j } \nabla _ { \mathbf { x } _ { i } } \left( w _ { i j } / W _ { i } \right)\tag{69}
$$

$$
= \sum _ { j \in N ( i ) } \hat { f } _ { j } \left( \nabla _ { \mathbf { x } _ { i } } w _ { i j } \cdot W _ { i } - w _ { i j } \cdot \nabla _ { \mathbf { x } _ { i } } W _ { i } \right) / W _ { i } ^ { 2 }\tag{70}
$$

$$
= \sum _ { j \in N ( i ) } \hat { f } _ { j } { \bf a } _ { i j } .\tag{71}
$$

When back-propagating loss gradients with respect to $\mathbf { n } _ { i } ,$ the weights $\mathbf { a } _ { i j }$ on $\hat { f } _ { j }$ in equation 71 are treated as constants.

## J.2 Closed-form Volume Integral

Assuming the cells are not overlapping, the volume integral can be split into one integral per cell, which are then blended to produce the color of the ray o + �r. To simplify the notation, let $[ a , b ] = [ t _ { i } ^ { - } , t _ { i } ^ { + } ]$ For each cell, we have:

$$
\mathbf { C } _ { i } = \int _ { a } ^ { b } \mathbf { c } _ { i } \sigma ( t ) T ( a , t ) d t\tag{72}
$$

$$
\begin{array} { r } { T ( a , t ) = e ^ { - \int _ { a } ^ { t } \sigma ( u ) d u } } \end{array}\tag{73}
$$

$$
\boldsymbol { \sigma } ( t ) = s \Phi ( - s \hat { f } ( t ) ) | \mathbf { n } _ { i } \cdot \mathbf { r } |\tag{74}
$$

Because the color $\mathbf { c } _ { i }$ is constant inside the cell, it can be moved outside of the integral in equation 72.

$$
= T ( a , a ) - T ( a , b ) = 1 - T ( a , b )\tag{75}
$$

(76)

$$
\int _ { a } ^ { b } \sigma ( t ) d t = \int _ { a } ^ { b } s \Phi ( - s { \hat { f } } ( t ) ) | \mathbf { n } _ { i } \cdot \mathbf { r } |\tag{77}
$$

$$
= s | \mathbf { n } _ { i } \cdot \mathbf { r } | \int _ { a } ^ { b } \Phi ( - s { \hat { f } } ( t ) ) d t\tag{78}
$$

$$
= s | \mathbf { n } _ { i } \cdot \mathbf { r } | \int _ { a } ^ { b } { \frac { 1 } { 1 + \exp ( s { \hat { f } } ( t ) ) } } d t\tag{79}
$$

Recall that the linear SDF approximation is:

$$
{ \hat { f } } ( t ) = { \hat { f } } ( \mathbf { o } + t \mathbf { r } )\tag{80}
$$

$$
{ \bf \Gamma } = \hat { f } _ { i } + \left( { \bf o } - { \bf x } _ { i } \right) \cdot { \bf n } _ { i } + t { \bf r } \cdot { \bf n } _ { i }\tag{81}
$$

Now, we perform a change of variable in the integral:

$$
u = s \hat { f } ( t )
$$

$$
d u = s \mathbf { r } \cdot \mathbf { n } _ { i } d t\tag{82}
$$

(83)

$$
\int _ { a } ^ { b } \sigma ( t ) \mathrm { d } t = s | \mathbf { n } _ { i } \cdot \mathbf { r } | \int _ { s { \hat { f } } ( a ) } ^ { s { \hat { f } } ( b ) } { \frac { 1 } { 1 + e ^ { u } } } { \frac { \mathrm { d } u } { s \mathbf { n } _ { i } \cdot \mathbf { r } } }\tag{84}
$$

$$
\mathbf { \Sigma } = \operatorname { s g n } ( \mathbf { n } _ { i } \cdot \mathbf { r } ) \int _ { s \hat { f } ( a ) } ^ { s \hat { f } ( b ) } { \frac { \mathrm { d } u } { 1 + e ^ { u } } }\tag{85}
$$

$$
\mathbf { \Sigma } = - s \mathrm { g n } ( \mathbf { n } _ { i } \cdot \mathbf { r } ) [ \ln ( 1 + e ^ { - u } ) ] _ { s \hat { f } ( a ) } ^ { s \hat { f } ( b ) }\tag{86}
$$

$$
\mathbf { \Sigma } = \operatorname { s g n } ( \mathbf { n } _ { i } \cdot \mathbf { r } ) \ln \left( { \frac { 1 + e ^ { - s { \hat { f } } ( a ) } } { 1 + e ^ { - s { \hat { f } } ( b ) } } } \right) .\tag{87}
$$

Consequently,

$$
T ( a , b ) = \exp \left( - \int _ { a } ^ { b } \sigma ( t ) { \mathrm { d } } t \right)\tag{88}
$$

$$
= \left( { \frac { 1 + e ^ { - s { \hat { f } } ( a ) } } { 1 + e ^ { - s { \hat { f } } ( b ) } } } \right) ^ { - s \mathrm { g n } ( \mathbf { n } _ { i } \cdot \mathbf { r } ) } .\tag{89}
$$

This gives the final expression for the transparency:

$$
\alpha _ { i } = T ( t _ { i } ^ { - } , t _ { i } ^ { + } ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 + \exp \bigl ( - s \hat { f } ( t _ { i } ^ { - } ) \bigr ) } { 1 + \exp \bigl ( - s \hat { f } ( t _ { i } ^ { + } ) \bigr ) } , } & { \mathbf { n } _ { i } \cdot \mathbf { r } < 0 , } \\ { \displaystyle \frac { 1 + \exp \bigl ( - s \hat { f } ( t _ { i } ^ { + } ) \bigr ) } { 1 + \exp \bigl ( - s \hat { f } ( t _ { i } ^ { - } ) \bigr ) } , } & { \mathbf { n } _ { i } \cdot \mathbf { r } > 0 . } \end{array} \right.\tag{90}
$$

The accumulated color is then

$$
\mathbf { C } = \sum _ { i } \mathbf { c } _ { i } \left( 1 - \alpha _ { i } \right) \prod _ { j < i } \alpha _ { j } .\tag{91}
$$

## J.3 Delaunay Triangulation

Assuming points in general position, we prove here that if a tetrahedron is not a Delaunay tetrahedron, then each of its four corners must have a Voronoi neighbor (not necessarily the same) inside its circumsphere.

By contraposition, this will show that a tetrahedron for which any of its four vertices has no Voronoi neighbors inside of the circumsphere must be a Delaunay tetrahedron.

Proof: Assume the tetrahedron IJKL is not a Delaunay tetrahedron. Without loss of generality, we look at corner I. The circumsphere of IJKL contains at least one site strictly inside of it by definition. Let C be the center of the circumsphere and P the ofending site: $| | C - P | | < r = | | C - I | |$ , with � the radius of the circumsphere. In other words, C is closer to P than to I, implying that C is not inside the cell of I by definition of a Voronoi cell.

Because C is outside of I’s cell, there must be a site Q, Voronoi neighbor of I, that defines a bisector plane such that C is on the wrong side:

$$
( C - I ) \cdot ( Q - I ) > \frac { 1 } { 2 } \vert \vert Q - I \vert \vert _ { 2 } ^ { 2 }\tag{92}
$$

All that remains to show is that $\mathrm { Q }$ must be inside the circumsphere of the tetrahedron:

$$
| | Q - C | | ^ { 2 } = | | Q - I + I - C | | ^ { 2 }\tag{93}
$$

$$
= | | Q - I | | ^ { 2 } + 2 ( Q - I ) \cdot ( I - C ) + | | I - C | | ^ { 2 }\tag{94}
$$

$$
= | | Q - I | | ^ { 2 } - 2 ( Q - I ) \cdot ( C - I ) + | | I - C | | ^ { 2 }\tag{95}
$$

$$
{ > } | | Q { - } I | | _ { 2 } ^ { 2 }
$$

$$
< | | I - C | | ^ { 2 } = r ^ { 2 }\tag{96}
$$

We have shown that � has a Voronoi neighbor � inside the circumsphere, which concludes the proof.