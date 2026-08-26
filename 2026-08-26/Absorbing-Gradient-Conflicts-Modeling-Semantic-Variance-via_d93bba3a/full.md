# Absorbing Gradient Conflicts: Modeling Semantic Variance via Kent Distributions for Cross-Modal Hashing

Hengjie Zhu<sup>1,2</sup> , Dayan Wu<sup>1∗</sup> , Zihao Zhang<sup>1,2</sup> , Xinze Liu<sup>1,2</sup> , Jingxuan Yu<sup>3</sup> , Peng Fu<sup>1</sup> , Zheng Lin<sup>1</sup> , Weiping Wang<sup>1</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences <sup>3</sup>School of Cyber Science and Engineering, Southeast University {zhuhengjie, wudayan, zhangzihao, liuxinze, fupeng, linzheng, wangweiping}@iie.ac.cn, yujingxuan24@seu.edu.cn

## Abstract

Supervised proxy-based deep cross-modal hashing has become the dominant paradigm for large-scale retrieval. However, prevalent methods model class proxies as deterministic points in the embedding space. This rigid assumption causes severe gradient conflicts in multi-label scenarios, where gradient conflicts arising from label co-occurrence lead to severe gradient contention and optimization collapse. To resolve this, we propose Kent-based Distributional Proxy Hashing (KDPH), a novel framework that shifts proxy representation from static points to flexible anisotropic Kent distributions on the hypersphere. Unlike point proxies that must shift their positions to accommodate conflicting gradients, KDPH absorbs these conflicts by dynamically adjusting its directional variance. This allows the proxy to maintain a stable semantic mean direction while stretching to cover diverse label correlations. Furthermore, to ensure stable training of these geometric parameters, we derive a tailored loss function incorporating the Cayley transform to enforce strict orthogonality. To the best of our knowledge, KDPH is the first framework to successfully introduce the Kent distributions into cross-modal hashing. Experiments on three benchmark datasets demonstrate that KDPH mitigates proxy collapse and chaotic oscillation, significantly outperforms state-of-the-art methods. Code is available at https://github.com/Senmo996/KDPHofficial-code.

## 1 Introduction

Driven by the surge in multimodal data [Xu et al., 2024], Cross-Modal Hashing Retrieval has emerged as a standard method for large-scale information retrieval [Cao et al., 2022; Singh and Gupta, 2022; Yan et al., 2025]. By projecting highdimensional features into a compact Hamming space, this approach circumvents the prohibitive computational and storage costs of conventional Euclidean methods, enabling efficient similarity approximation via bitwise XOR operations [Wang et al., 2022; Lu et al., 2019; Sun et al., 2022]. Despite rapid progress towards Supervised Deep Cross-Modal Hashing Retrieval—leveraging deep neural networks and semantic supervision for end-to-end representation learning—fundamental challenges still persist. Real-world data are inherently multifaceted, with single instances often associated with multiple overlapping labels. This multi-label characteristic poses severe difficulties for existing supervised frameworks, which struggle to capture complex, non-exclusive semantic correlations, necessitating more robust modeling strategies.

![](images/09adac65ce6bce4299f3121b9eeb004014f1e1cfa0e5c11161cdcad283ad028e.jpg)  
(a) Baseline: Proxy Collapse

![](images/407190abd253ea243ca8d9cce61283ea09b3ad036b3460cee5f58243677198fe.jpg)  
(b) Ours: Well-Separated

![](images/96cc53042508dbb40ee6fd8fc710bc5c8970f458ac004da77231cd3d146aac66.jpg)

![](images/5a561d99aae0cd95ca5138586db234e68c1d5dd403d7479cdf1c727a438b2d15.jpg)  
(c) Baseline: Proxy Oscillating  
(d) Ours: Fast Convergence  
Figure 1: Comparison of proxy optimization behaviors. (a)-(b) Conventional proxy learning suffers from proxy collapse, where learned proxies move toward the anchor and become poorly separated, whereas KDPH preserves distinct proxy distributions. (c)- (d) Conventional methods exhibit slow and oscillatory optimization trajectories, while KDPH achieves smoother trajectories and faster convergence. Colors denote different categories.

Despite the dominance of proxy-based strategies in deep cross-modal hashing [Tu et al., 2023; Huo et al., 2024b], fundamental limitations persist in multi-label scenarios. The prevailing paradigm typically abstracts each semantic category as a singular, deterministic point [Teh et al., 2020; Kim et al., 2020; Movshovitz-Attias et al., 2017; Qian et al., 2019] within the embedding space. While effective for singlelabel tasks, this rigid zero-variance assumption provokes severe gradient contention when confronted with the complex semantic conflicts inherent in label co-occurrence. Specifically, in a point-proxy framework, a multi-label sample simultaneously exerts pulling forces on divergent proxies. Modeled as rigid points, these proxies lack the spatial capacity to accommodate such conflicting semantic signals. Consequently, this unresolved contention drives the model towards optimization collapse. These two limitations manifesting in two degenerate states: (1) Chaotic Proxy Oscillation, where the proxy, unable to satisfy conflicting gradients, is forced to constantly shift its position, preventing the model from anchoring to a stable semantic center; and (2) Proxy Collapse, where the optimization process forcibly pulls proxies of distinct categories together to minimize aggregate loss—particularly causing lowfrequency tail classes to collapse onto high-frequency head classes—thereby eliminating fine-grained decision boundaries and impairing discriminative power.

To address these challenges, we propose the Kent-based Distributed Proxy Hashing (KDPH) framework. We depart from the single-point paradigm by modeling each class proxy as a learnable Kent distribution—an anisotropic spherical probability model defined by a mean direction and an elliptical covariance structure. Unlike rigid point proxies that must drastically shift position to satisfy conflicting gradients, this distributed representation resolves conflicts by changing specific directional variances, thereby endowing each class with a semantic shape and volume to flexibly absorb the contextual variance introduced by label co-occurrence. Specifically, observing the anisotropic geometry of the Kent distribution, we design a tailored loss function to achieve gradient decoupling mechanism by stretching shape rather than moving centroid. This approach effectively resolves conflict, preventing chaotic oscillation and proxy collapse. Furthermore, by incorporating the Cayley transform to enforce strict orthogonality for the principal axes and utilizing this probabilistic module solely to guide the learning process, KDPH achieves state-of-the-art performance with zero additional computational overhead during inference, successfully resolving the persistent trade-off between retrieval accuracy and efficiency.

In summary, our contributions are as follows:

• We introduce KDPH, a novel framework that reformulates proxy-based hashing by substituting rigid point proxies with anisotropic Kent distributions. To the best of our knowledge, KDPH is the first work to demonstrate that modeling proxy uncertainty through directional distributions can eliminate the optimization collapse caused by gradient contention in multi-label cross-modal retrieval.

• We derive a manifold-constrained loss function leveraging the Cayley transform, which enables proxies to adaptively absorb gradient conflicts via directional variance rather than centroid displacement, thereby effectively eradicating Proxy Collapse and suppressing Chaotic Proxy Oscillation during the learning process.

• Extensive experiments demonstrate that KDPH achieves state-of-the-art performance on three benchmarks, empirically validating the efficacy of our gradient decoupling strategy and showing superior robustness particularly in complex multi-label scenarios.

## 2 Related Work

## 2.1 Supervised Deep Cross-Modal Hashing and Proxy Learning

Supervised deep cross-modal hashing has advanced significantly, positioned such as CNN-RNN and adversarial baselines [Jiang and Li, 2017; Li et al., 2018; Bai et al., 2020]. Recently, graph and transformer-based models [Xu et al., 2019; Tu et al., 2022; Liu et al., 2023; Chen et al., 2024; Liu et al., 2024] have further pushed the boundaries of performance. In parallel, these models improving their algorithmic design from simple pairwise constraints [Qin et al., 2024] to adaptive gradient-triplet losses [Zhu et al., 2025a] or sophisticated geometric [Qin et $a l .$ , 2025b] strategies which ranging from spherical mutual information to boundary sensitive mining [Qin et al., 2025a]—to refine the Hamming space structure. To address the scalability bottlenecks of pairwise comparisons, proxy-based metric learning has been effectively incorporated. By representing categories as distinct point proxy, this paradigm allows the model to efficiently organize the embedding space [Tu et al., 2023; Huo et al., 2024b], and has recently explored hyperbolic geometries for hierarchical modeling [Huo et al., 2024c]. However, a major challenge persists that the deterministic point formulation’s inherent zero-variance assumption fundamentally limits its applicability to multi-label scenarios with high contextual variance.

## 2.2 Probabilistic Embeddings and Non-Isotropy

Probabilistic embedding learning has emerged as a robust alternative to deterministic point estimates [Lin et al., 2018]. To respect the unique geometry of hyperspherical feature spaces, the field has evolved from early Euclidean Gaussian models [Shi and Jain, 2019] to spherical probability density functions, such as the von Mises-Fisher (vMF) distribution [Zhe et al., 2018; Park et al., 2019; Li et al., 2021]. However, the efficacy of vMF-based approaches remains constrained by their inherent isotropy. One major challenge is the implicit assumption of uniform variance in all directions and this rigid simplification fails to accommodate the complex, anisotropic semantic manifolds inherent in multi-label data. Although recent metric learning approaches have explored non-isotropy through lightweight mechanisms, such as variance scaling or geometric regularization [Roth et al., 2022; Kirchhof et $a l . .$ , 2022], these low-parameter heuristics often lack the sufficient geometric flexibility required for large-scale multi-label cross-modal hashing, where semantic manifolds are densely entangled. To bridge this gap, we propose the KDPH, which introduces the parameter-rich Kent distribution to deliberately increase geometric capacity. We further facilitate stable training by resolving gradient contention and optimization collapse via the Cayley transform and a tailored loss function.

![](images/c45ce8afa8c0a0ed50708306614292207cf1c75ebeb6dc266511ef42929409ae.jpg)  
Figure 2: Overview of the Kent-based Distributed Proxy Hashing (KDPH) framework. It consists of a CLIP-based feature extraction module and a Kent-based Distributional Proxy Learner. As shown in the right panel, we model class proxies as learnable anisotropic distributions rather than static points. Through the gradient decoupling mechanism by separating centroid $\mu$ and variance B, Γ, the mode dynamically stretches the proxy’s semantic volume to accommodate multi-label variance, effectively mitigating proxy collapse and oscillation.

## 3 Method

## 3.1 Formulation and Architecture

Given a training dataset $O \ = \ \{ ( i _ { n } , t _ { n } , l _ { n } ) \} _ { n = 1 } ^ { N }$ with images $i _ { n } ,$ texts $\mathbf { \Delta } _ { t _ { n } } .$ , and multi-label vectors $l _ { n } \in \{ \bar { 0 } , 1 \} ^ { C }$ , we extract D-dimensional semantic features $\pmb { f } _ { n } ^ { i }$ and $\pmb { f } _ { n } ^ { t }$ via a dual-stream Transformer backbone initialized with pre-trained CLIP. Modality-specific hashing networks $H _ { I } , \dot { H _ { T } } : \mathbb { R } ^ { D } $ $\mathbb { R } ^ { K }$ then project these features into K-dimensional continuous representations $\boldsymbol { h } _ { n } ^ { i }$ and $\boldsymbol { h } _ { n } ^ { t }$ . The final binary hash codes are obtained via the sign function:

$$
b _ { n } ^ { i } : = \mathrm { s i g n } ( { \pmb h } _ { n } ^ { i } ) \in \{ - 1 , 1 \} ^ { K } , \quad { \pmb b } _ { n } ^ { t } : = \mathrm { s i g n } ( { \pmb h } _ { n } ^ { t } ) \in \{ - 1 , 1 \} ^ { K }\tag{1}
$$

## 3.2 Kent-based Distributed Proxy Mechanism

Conventional hashing methods typically abstract each class as deterministic point whose rigid zero-variance assumption fundamentally unsuitable for multi-label scenarios, and conflicting semantic signals from co-occurring labels lead to gradient contention and optimization collapse. To resolve this, KDPH shifts the paradigm from static points to distributional proxies on the hypersphere [Wang and Isola, 2020]. By modeling each class as a flexible Kent distribution with learnable centroid and variance, our framework effectively absorbs the gradients variance arising from label co-occurrence.

## Kent Distribution for Proxy Modeling

To instantiate this distributional paradigm, we employ the generalized Kent distribution. This strategic choice addresses the inherent limitations of simpler spherical models, such as the von Mises-Fisher (vMF) distribution. While vMF offers mathematical convenience, its isotropic nature rigidly imposes uniform variance across all dimensions, rendering it ill-suited for capturing the complex, directional manifolds typical of multi-label features. In contrast, the Kent distribution is intrinsically anisotropic. It provides the necessary geometric flexibility to model elliptical contours on the hypersphere $\mathbb { S } ^ { K - 1 }$ , where K denotes the hash code dimension, distinct from the feature dimension $D .$

Formally, the probability density function (PDF) for a class c is defined as:

$$
p ( z | \Theta _ { c } ) = \frac { 1 } { c ( \kappa , B ) } \exp \left( \kappa _ { c } \pmb { \mu } _ { c } ^ { T } z + \sum _ { i = 1 } ^ { N _ { a x e s } } \beta _ { c , i } ( \gamma _ { c , i } ^ { T } z ) ^ { 2 } \right)\tag{2}
$$

where $c ( \kappa , B )$ denotes the normalizing constant. The distribution is parameterized by $\Theta _ { c } = \{ \mu _ { c } , \kappa _ { c } , \Gamma _ { c } , B _ { c } \}$ , in which each component serves a distinct semantic role: the mean direction $\pmb { \mu } _ { c } ^ { \star } \in \mathbb { S } ^ { K - 1 }$ acts as the semantic centroid representing the stable core identity of the class, while $\kappa _ { c }$ governs the global concentration of samples around this centroid. Crucially, to model non-uniform variance, $\Gamma _ { c }$ defines the orthogonal principal axes, and $B _ { c } = \{ \beta _ { c , i } \}$ controls the anisotropic shaping along these dimensions. By learning distinct $\beta _ { c , i }$ values, the proxy can dynamically stretch or compress along specific semantic directions. This mechanism allows the distribution to accommodate the high semantic variance introduced by multilabel co-occurrences without drastically shifting the centroid $\pmb { \mu } _ { c }$

## 3.3 Hash learning

To jointly optimize the deep networks and proxy parameters $\Theta _ { c }$ while respecting manifold geometry, we introduce a differentiable reparameterization technique. This mechanism is integrated with a Kent-based log-likelihood scoring function and a multi-task objective to strictly enforce both semantic discrimination and quantization constraints.

## Orthogonal Frame Construction via Cayley Transform

Directly optimizing $\pmb { \mu } _ { c }$ and $\mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma }$ is challenging due to the strict orthogonality required by the Kent distribution. To address this, we construct a unified orthonormal basis using the Cayley

Transform. Specifically, for each class c, we learn a skewsymmetric parameter matrix $\mathbf { A } _ { c } \in \mathbb { R } ^ { K \times K }$ . By exploiting the mapping between the Lie algebra ${ \mathfrak { s o } } ( K )$ and the Lie group $S \dot { O ( K ) }$ , we generate a strict rotation matrix $\mathbf { Q } _ { c }$

$$
\mathbf { Q } _ { c } = ( \mathbf { I } + \mathbf { A } _ { c } ) ^ { - 1 } ( \mathbf { I } - \mathbf { A } _ { c } ) \in S O ( K )\tag{3}
$$

This transformation ensures that $\mathbf { Q } _ { c }$ remains orthogonal regardless of the updates to $\mathbf { A } _ { c } .$ . We then define the proxy components by slicing $\mathbf { Q } _ { c } \mathbf { . }$ : the first column serves as the centroid $\mu _ { c } ,$ while the following columns form the shape axes $\mathbf { \Gamma } _ { \mathbf { C } } .$ This parameterization naturally enforces geometric constraints, guaranteeing orthogonality not only between the centroid and the axes but also pairwise among all principal axes within $\mathbf { \Gamma } _ { \mathbf { C } } ,$ effectively bypassing the need for expensive orthogonalization procedures.

## Anisotropic Energy Scoring and Parameterization

To evaluate the semantic affinity between a query sample and the class proxies, we first project the continuous hash code h onto the unit hypersphere, $\dot { z } = h / \| h \| _ { 2 } \in \mathbb { S } ^ { K - 1 }$ , ensuring geometric consistency with the directional statistics.

While the canonical Kent distribution defines a rigorous probability density function, its optimization is hindered by a computationally intractable normalization constant dependent on high-dimensional Bessel functions. Since our primary objective in hashing is discriminative ranking rather than exact density estimation, we circumvent this bottleneck by reinterpreting the log-density formulation as a geometric Compatibility Score. We define the anisotropic energy score $S ( z , c )$ for a class c as follows:

$$
S ( z , c ) = \kappa _ { c } ( \pmb { \mu } _ { c } ^ { T } z ) + \sum _ { i = 1 } ^ { N _ { \mathrm { a x e s } } } \beta _ { c , i } ( \gamma _ { c , i } ^ { T } z ) ^ { 2 } - \mathcal { R } ( \kappa , \beta )\tag{4}
$$

Here, we employ a logarithmic regularization term $\mathcal { R } ( \kappa , \beta ) =$ $\log ( \kappa _ { c } + \| \dot { \beta _ { c } } \| _ { 1 } )$ . This term serves as a soft magnitude constraint, effectively transforming the probabilistic objective into a flexible quadratic energy metric on the hypersphere, preserving the topological benefits of Kent geometry while avoiding the gradient instability of Bessel approximations.

To strictly enforce the definition of concentration and magni tude, the parameters $\kappa _ { c }$ and $\beta _ { c , i }$ are constrained to be positive via a Softplus activation on learnable scalars $\hat { \kappa } _ { c }$ and $\hat { \beta } _ { c , i }$ . Crucially, regarding the shape constraints, while the canonical Kent distribution requires $2 | \beta _ { c , i } | < \kappa _ { c }$ for uni-modality, we adopt a generalized Fisher-Bingham formulation by removing this upper bound. By allowing the network to learn $\beta _ { c }$ freely, the proxy gains the flexibility to model complex, potentially multi-modal distributions. This relaxation is particularly advantageous for capturing the high variance and distinct semantic manifolds inherent in multi-label data.

## Distribution-Aware Proxy Triplet Loss

To strictly enforce discriminative margins within the hyperspherical manifold, we employ a structure-preserving Distribution-Aware Proxy Triplet Objective. Unlike standard cross-entropy which treats classes as independent points, this loss explicitly optimizes the anisotropic energy alignment between samples and learnable Kent distributions. By maximizing the energy score for relevant classes while minimizing it for irrelevant ones, the objective ensures that an instance resides firmly within the high-density acceptance region of its ground-truth semantic distributions, effectively pushing decision boundaries away from the ambiguous overlapping zones.

Formally, given a normalized query sample z, a positive proxy distribution $c _ { p } \in \mathcal { P }$ (where $\mathcal { P }$ is the set of ground-truth labels), and a negative proxy $c _ { n } \in { \mathcal { N } } .$ , we enforce a strict separation constraint based on the energy score $S ( \cdot ) \colon$

$$
S ( z , c _ { p } ) > S ( z , c _ { n } ) + m\tag{5}
$$

where m is a fixed margin hyperparameter governing the minimum semantic gap.

To accommodate the multi-label setting where a sample is associated with multiple semantic centroids, the final loss aggregates these constraints over all valid triplets. This formulation allows the gradients to dynamically adjust the shape parameters $\beta _ { c }$ of the Kent distributions, absorbing the conflict from hard negatives:

$$
\mathcal { L } _ { P r o x y } = \frac { 1 } { | \mathcal { P } | | \mathcal { N } | } \sum _ { y \in \mathcal { P } } \sum _ { j \in \mathcal { N } } \operatorname* { m a x } \bigl ( 0 , S ( z , c _ { n } ) - S ( z , c _ { p } ) + m \bigr )\tag{6}
$$

## Multi-Modal Irrelevance Regularization

To strictly separate semantically disjoint samples, we incorporate the Multi-Modal Irrelevance Loss [Huo et $a l .$ , 2024b]. This objective penalizes high similarities between irrelevant pairs—defined as instances sharing no common labels (i.e., $l _ { i } \cdot l _ { j } = 0 )$ . Taking the image intra-modal regularization as an example, the loss is defined as the average cosine similarity over all such pairs in a batch:

$$
\mathcal { L } _ { r e g . i } = \frac { \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \mathbb { I } ( l _ { i } \cdot l _ { j } = 0 ) \cos ( h _ { i } ^ { x } , h _ { j } ^ { x } ) } { \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \mathbb { I } ( l _ { i } \cdot l _ { j } = 0 ) }\tag{7}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function and B is the batch size. The text intra-modal loss $\mathcal { L } _ { r e g _ { - } t }$ and the cross-modal loss $\mathcal { L } _ { r e g . i t }$ are computed analogously by substituting the continues hash feature pairs with $( \check { h } ^ { y } , \check { h ^ { y } } )$ and $( h ^ { x } , h ^ { y } )$ , respectively. The final regularization term aggregates these three components: $\mathcal { L } _ { R e g } = \mathcal { L } _ { r e g . i } + \mathcal { L } _ { r e g . t } + \mathcal { L } _ { r e g . i t } .$

## Cross-Modal Contrastive Alignment

To bridge the heterogeneity between modalities, we employ the symmetric InfoNCE [He et al., 2020] objective to project representations into a shared, modality-invariant hypersphere. This contrastive loss maximizes the mutual information between matched image-text pairs. The image-to-text alignment loss is formulated as:

$$
\mathcal { L } _ { i  t } = - \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \log \frac { \exp ( \cos ( h _ { k } ^ { x } , h _ { k } ^ { y } ) / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( \cos ( h _ { k } ^ { x } , h _ { j } ^ { y } ) / \tau ) }\tag{8}
$$

where τ is a learnable temperature parameter scaling the distribution sharpness. The text-to-image loss $\mathcal { L } _ { t  i }$ is computed symmetrically by swapping the anchor and positive samples. The final cross-modal alignment objective is the sum of both directions: $\mathcal { L } _ { C M } = \mathcal { L } _ { i  t } + \mathcal { L } _ { t  i }$

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="3">MIRFLICKR-25K</td><td colspan="3">NUS-WIDE</td><td colspan="3">MS COCO</td></tr><tr><td>16bits</td><td>32bits</td><td>64bits</td><td>16bits</td><td>32bits</td><td>64bits</td><td>16bits</td><td>32bits</td><td>64bits</td></tr><tr><td rowspan="14">I2T</td><td>DCPH</td><td>TKDE&#x27;22</td><td>0.7679</td><td>0.7733</td><td>0.7764</td><td>0.6161</td><td>0.6156</td><td>0.6180</td><td>0.5206</td><td>0.5806</td><td>0.5939</td></tr><tr><td>DCMHT</td><td>MM&#x27;22</td><td>0.8263</td><td>0.8272</td><td>0.8441</td><td>0.6799</td><td>0.6992</td><td>0.7038</td><td>0.6447</td><td>0.6757</td><td>0.6915</td></tr><tr><td>nivMF</td><td>ECCV’22</td><td>0.8241</td><td>0.8245</td><td>0.8352</td><td>0.7023</td><td>0.6902</td><td>0.7024</td><td>0.7075</td><td>0.7242</td><td>0.7523</td></tr><tr><td>MIAN</td><td>TKDE&#x27;22</td><td>0.8123</td><td>0.8220</td><td>0.8355</td><td>0.6130</td><td>0.5894</td><td>0.6057</td><td>0.5350</td><td>0.5579</td><td>0.5404</td></tr><tr><td>DSPH</td><td>TCSVT’23</td><td>0.8129</td><td>0.8482</td><td>0.8541</td><td>0.6830</td><td>0.6979</td><td>0.7162</td><td>0.7044</td><td>0.7510</td><td>0.7694</td></tr><tr><td>DNPH</td><td>MC&#x27;24</td><td>0.8108</td><td>0.8269</td><td>0.8229</td><td>0.6689</td><td>0.6811</td><td>0.6939</td><td>0.6438</td><td>0.6910</td><td>0.7294</td></tr><tr><td>DNpH</td><td>TMM&#x27;24</td><td>0.8423</td><td>0.8552</td><td>0.8558</td><td>0.6921</td><td>0.7022</td><td>0.7071</td><td>0.6727</td><td>0.6903</td><td>0.6860</td></tr><tr><td>RDPH</td><td>TMM&#x27;24</td><td>0.7420</td><td>0.7749</td><td>0.7867</td><td>0.6330</td><td>0.6317</td><td>0.6523</td><td>0.5867</td><td>0.7151</td><td>0.7368</td></tr><tr><td>DDBH</td><td>TCSVT&#x27;24</td><td>0.8450</td><td>0.8534</td><td>0.8610</td><td>0.7041</td><td>0.7145</td><td>0.7229</td><td>0.7165</td><td>0.7454</td><td>0.7681</td></tr><tr><td>DNcH</td><td>ESWA&#x27;25</td><td></td><td></td><td></td><td>0.7150</td><td>0.7305</td><td>0.7387</td><td>0.7199</td><td>0.7341</td><td>0.7509</td></tr><tr><td>DAGtH</td><td>ESWA&#x27;25</td><td>0.8268</td><td>0.8514</td><td>0.8599</td><td>0.6827</td><td>0.6882</td><td>0.6927</td><td></td><td></td><td></td></tr><tr><td>KDPH</td><td>Ours</td><td>0.8770</td><td>0.8639</td><td>0.8639</td><td>0.7430</td><td>0.7365</td><td>0.7392</td><td>0.7636</td><td>0.7761</td><td>0.7843</td></tr><tr><td></td><td>TKDE&#x27;22</td><td>0.7992</td><td>0.8046</td><td>0.8061</td><td>0.6187</td><td>0.6292</td><td>0.6394</td><td>0.5981</td><td>0.5879</td><td>0.5745</td></tr><tr><td rowspan="14">T2I</td><td>DCPH DCMHT</td><td>MM&#x27;22</td><td>0.8116</td><td>0.8137</td><td>0.8281</td><td>0.6876</td><td>0.7104</td><td>0.7253</td><td>0.6513</td><td>0.6832</td><td></td></tr><tr><td>nivMF</td><td>ECCV’22</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7052</td></tr><tr><td></td><td>TKDE&#x27;22</td><td>0.8125</td><td>0.8102</td><td>0.8213</td><td>0.6912</td><td>0.6952</td><td>0.6924</td><td>0.6421</td><td>0.6892</td><td>0.7042</td></tr><tr><td>MIAN</td><td>TCSVT’23</td><td>0.8058</td><td>0.8151</td><td>0.8182</td><td>0.6365</td><td>0.6312</td><td>0.6416</td><td>0.5489</td><td>0.5493</td><td>0.5365</td></tr><tr><td>DSPH</td><td>MC&#x27;24</td><td>0.8000</td><td>0.8238</td><td>0.8294</td><td>0.6997</td><td>0.7153</td><td>0.7304</td><td>0.7040</td><td>0.7556</td><td>0.7713</td></tr><tr><td>DNPH</td><td></td><td>0.8015</td><td>0.8176</td><td>0.8166</td><td>0.6871</td><td>0.6994</td><td>0.7182</td><td>0.6468</td><td>0.7012</td><td>0.7388</td></tr><tr><td>DNpH</td><td>TMM&#x27;24 TMM&#x27;24</td><td>0.8147</td><td>0.8292</td><td>0.8361</td><td>0.6992</td><td>0.7137</td><td>0.7139</td><td>0.6562</td><td>0.6860</td><td>0.6928</td></tr><tr><td>RDPH</td><td>TCSVT&#x27;24</td><td>0.7200</td><td>0.7504 0.8318</td><td>0.7713</td><td>0.6732 0.7194</td><td>0.7067</td><td>0.7190</td><td>0.5925</td><td>0.6910</td><td>0.7139</td></tr><tr><td>DDBH</td><td></td><td>0.8245</td><td></td><td>0.8390</td><td></td><td>0.7211</td><td>0.7325</td><td>0.7167</td><td>0.7394</td><td>0.7595</td></tr><tr><td>DNcH</td><td>ESWA&#x27;25</td><td></td><td></td><td></td><td>0.7196</td><td>0.7343</td><td>0.7430</td><td>0.7130</td><td>0.7306</td><td>0.7398</td></tr><tr><td>DAGtH</td><td>ESWA&#x27;25</td><td>0.8072</td><td>0.8270</td><td>0.8382</td><td>0.6936</td><td>0.7042</td><td>0.7098</td><td></td><td></td><td></td></tr><tr><td>KDPH</td><td>Ours</td><td>0.8351</td><td>0.8439</td><td>0.8480</td><td>0.7473</td><td>0.7399</td><td>0.7458</td><td>0.7627</td><td>0.7785</td><td>0.7795</td></tr></table>

Table 1: mAP comparisons on three benchmark datasets with references. The best results are highlighted in bold.

## Uniform Distribution Constraint for Hashing

To maximize code entropy and prevent bit-collapse, we enforce a Uniform Distribution Constraint via Optimal Transport. Specifically, we minimize the Wasserstein-1 distance between the learned batch codes $\pmb { B } ^ { x }$ and a sampled ideal uniform distribution $A \in \{ - 1 , 1 \} ^ { B \times K }$ . This is formulated as finding the optimal permutation matrix $O ^ { x }$ that minimizes the transport cost:

$$
\mathcal { L } _ { U , x } = \operatorname* { m i n } _ { O ^ { x } \in \mathcal { O } } - \mathrm { t r } ( O ^ { x } A ( B ^ { x } ) ^ { T } )\tag{9}
$$

where O is the set of all $B \times B$ permutation matrices. The constraint is applied symmetrically to both modalities, yielding the total loss $\mathcal { L } _ { U } = \mathcal { L } _ { U , x } + \mathcal { L } _ { U , y }$

## Overall Objective Function

Our comprehensive training objective $\mathcal { L } _ { T o t a l }$ is a multi-task function, designed to integrate the four distinct components discussed previously. It simultaneously optimizes the model for (1) proxy learning, (2) cross-modal alignment, (3) semantic irrelevance separation, and (4) high-entropy hash code generation.The final objective is a weighted summation of these components:

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { P r o x y } + \epsilon \mathcal { L } _ { R e g } + \zeta \mathcal { L } _ { C M } + \eta \mathcal { L } _ { U }\tag{10}
$$

where $\epsilon , \zeta ,$ and η are scalar hyperparameters that balance the contribution of each term.

## 4 Experiment

To demonstrate the validity of our proposed KDPH framework, we conducted comprehensive experiments on three pub licly available cross-modal multi-label datasets MIRFLICKR-25K, NUS-WIDE and MS COCO. Besides, we introduce the datasets for the experiments and explain the details of the implementation of KDPH and evaluate metrics.

## 4.1 Experimental Settings

## Datasets

We evaluate KDPH on three standard multi-label datasets. MIRFLICKR-25K [Huiskes and Lew, 2008] contains 24,581 image-text pairs across 24 classes. NUS-WIDE [Chua et al., 2009] comprises 269,648 pairs; following standard protocols, we utilize a subset of 195,834 pairs from 21 frequent categories. MS COCO [Lin et al., 2014] consists of 123,289 images, each with 5 captions, spanning 80 categories. Across all datasets, we randomly sample 5,000 pairs as the query set and use the remainder as the database, from which 10,000 pairs are uniformly sampled to construct the training set.

## Baseline Methods

In our experiments, we selected 9 state-of-the-art deep cross-modal hashing methods for comparison, which contain DCPH[Tu et al., 2023], nivMF[Kirchhof et al., 2022], DCHMT[Tu et al., 2022], MIAN[Zhang et al., 2023], DSPH[Huo et al., 2024b], MITH[Liu et al., 2023], DNPH[Huo et al., 2024a], DNpH[Qin et al., 2024], DDBH[Qin et al., 2025a], DNcH[Zhu et al., 2025b] and DAGtH[Zhu et al., 2025a].

## Experimental Details

We implemented KDPH in PyTorch on an NVIDIA A800 GPU, optimizing via Adam (LR=0.001, batch size=128). The hyper-parameters are set as $\epsilon = 4 0 , \zeta = 0 . 3 , \eta = 0 . 1$ . Regarding the anisotropic geometry, we set the number of principal axes as $N _ { a x e s } \dot { = } { K \mathord { \left/ { \vphantom { K ^ { 2 } } 8 } \right. \kern - delimiterspace } 8 }$ , where K denotes the target hash code length $( \mathrm { e . g . , } N _ { a x e s } = 8$ for 64-bit codes). To ensure a fair comparison, we follow all baselines that use the identical CLIP backbone and data splits.

![](images/f39ac6eff4f4a99fee42f577dfd7f1aa93af1661444c7ca99197245f4ce788e0.jpg)

(a) Retrieval Performance vs. Semantic Complexity  
![](images/89cae918a7b6a1ecf411d16133143a8cbe7373f41c7925579724f45fb8fc05da.jpg)  
(b) Precision of ’Hard-to-Find’ Labels  
Figure 3: Impact of Semantic Complexity on Model Performance.

## 4.2 Main Results

To evaluate the effectiveness of our KDPH framework, we conduct a comprehensive comparison with state-of-the-art baselines on three benchmark datasets. The comparative results are summarized in Table 1. The proposed KDPH method outperforms all state-of-the-art baselines with significant performance improvements across all hash code lengths. This phenomenon demonstrates that the anisotropic Kent distributions possess stronger feature extraction and discriminative capabilities than the deterministic point-based baselines.In particular, our KDPH yields superiority over nivMF, a representative approach employing non-isotropic von Mises-Fisher distributions. It is observed that such methods suffer from the limitations of geometric inflexibility, particularly in scenarios enriched with complex semantic entanglements. This limitation is empirically evidenced by the substantial margin of over 7.92% on the challenging MS COCO dataset (I2T @ 16 bits).Despite the reliable performance achieved by previous methods, our KDPH benefits from the Kent-based distributional proxy and modeling elliptical semantic variance, demonstrates superior performance.

![](images/36bb4871893a191f556da1ffe2166945cf6d348ecb499bb351754295c349c313.jpg)

![](images/c8ebaed735b3d92625e58a1c0560d39a23a0745ae7db3e1ff0cdd64a65febead.jpg)

(a) High co-occurrence rate classes  
![](images/d0c423cd49962d33676be9c5aa337001d36f7f2f2d3e4a789bc92e1bf4a53cdc.jpg)

![](images/a2d6d3c9cf420855b6ac674de5a417f804a43d8e2ddfbe8c0ce6d6c9d8a2db5a.jpg)  
(b) Low co-occurrence rate classes  
Figure 4: Training dynamics comparing DSPH and KDPH.

Furthermore, to verify the performance of our KDPH under highly compressed spaces, we investigate the Bit-Efficiency. In comparison to the state-of-the-art methods (e.g., DCPH, DSPH) which suffer from sharp performance degradation under limited encoding capacity, KDPH maintains high discriminative power at 16 bits. Specifically, our KDPH yields an improvement of approximately 6.07% on MS COCO over the second-best method. These findings suggest that the meticulously designed Kent distributions offer superior information density, effectively absorbing gradient contentions even within highly compressed Hamming spaces. KDPH also achieves good balance between modalities, indicating that the optimization strategy successfully harmonizes visual and textual distributions within the shared latent space, thereby preventing the modality domination often observed in existing baselines.

## 4.3 Analysis

## Mitigation of Chaos Proxy Oscillation via KDPH

We first investigate the robustness of our KDPH approach in preventing optimization collapse. So we monitor the track of Proxy Flip Rate $R _ { f l i p } ^ { ( t ) }$ [Xie et al., 2016], defined as the ratio of samples changing their nearest proxy assignment between consecutive epochs. As illustrated in Figure 4, it can be observed that in Low Co-occurrence Classes (Fig. 4b), both the baseline and KDPH exhibit stable convergence, confirming that rigid point proxies suffice for simple, unimodal manifolds.In contrast, regarding High Co-occurrence Classes (Fig. 4a), the baseline suffers from severe oscillation and jagged precision curves, leading to a deficiency in optimization stability. Benefiting from the advantages of anisotropic Kent distributions, KDPH maintains exceptional stability with a flip rate < 1%. Specifically, our framework is designed to dynamically absorb gradient variance through shape deformation rather than through straightforward centroid shifting. In this way, the proposed method effectively mitigates the chaotic oscillations that plague traditional point-based methods, ensuring better adaptability to complex semantic environments.

## Prevention of Proxy Collapse via the KDPH

To comprehensively evaluate the effectiveness in prevention of proxy collapse, we track the Subordinate Label as the category with the least frequency in an sample. As illustrated in Figure 3, Baseline suffers from the limitations in high-complexity situations where rigid point proxies degrade rapidly under the pressure of dominant classes. In contrast, KDPH method consistently demonstrates the superiority and robustness over state-of-the-art competitors, yielding a significant improvement of 41.1% in scenarios with 5 labels. This promising performance confirms that the meticulously designed anisotropic distributions effectively mitigate the collapse issue by reserving sufficient semantic variance for tail concepts.

Despite the reliable performance achieved by the Baseline in single-label regimes due to the advantage of point proxies, it is noted that this advantage vanishes in multi-label scenarios. This slight redundancy introduced by KDPH serves as a necessary trade-off, providing the parametric flexibility required to bridge the semantic gaps between conflicting manifolds.

<table><tr><td colspan="3">Components</td><td rowspan="2">|mAP @ 64 bits</td></tr><tr><td>Kent</td><td> $\mathcal { L } _ { R e g }$ </td><td>LCM  $\mathcal { L } _ { U }$ </td></tr><tr><td></td><td>√</td><td>V V</td><td>|I → T 0.7615 0.7542</td></tr><tr><td>β = 0</td><td>√</td><td>V V</td><td>0.7693 0.7628</td></tr><tr><td>√</td><td></td><td>√ √</td><td>0.7326 0.7305</td></tr><tr><td>V</td><td>V</td><td>√</td><td>0.7566 0.7585</td></tr><tr><td>√</td><td>√</td><td>V</td><td>0.7537 0.7574</td></tr><tr><td>V</td><td>V</td><td>V √</td><td>0.7843 0.7795</td></tr></table>

Table 2: Ablation study of different components on the MS COCO.

## Ablation Study on Model Components

We conduct ablation studies by systematically evaluating the mAP of each component in our framework on the MS COCO dataset. The comparative results are summarized in Table 2. The performance metric exhibits a distinct hierarchy. Specifically, mAP decreases from 0.7693 with the isotropic vMF distribution (β = 0) to 0.7615 with rigid point proxies, suggesting only a marginal advantage for distributional proxies. Crucially, our anisotropic Kent model surpasses all state-of-the-art variants, owing to its capacity to capture anisotropic semantic conflicts, enabling effective recalibration of semantic interactions to resolve gradient contentions.Furthermore, the Multi-Modal Irrelevance Regularization $\mathcal { L } _ { R e g }$ and Cross-Modal Alignment $\mathcal { L } _ { C M }$ are designed to collaboratively enhance the contextual awareness and semantic capabilities, ensuring better adaptabil ity to nuanced semantic variations while Uniform Constraint $\mathcal { L } _ { U }$ provide essential complementary gains for modality invariance and code entropy.

![](images/849a14dd7fb429da53294a3a3e27e458f0ce1b83b7fb4c0230fa3642ae3678bc.jpg)  
(a) Baseline: Embedding Space

![](images/ddb6119d90801a4e987260952da8351fec1ba914a72a6a2e7aba73fec625d4a8.jpg)  
(b) Our Method: Embedding Space  
Figure 5: t-SNE visualization of the embedding spaces learned by DSPH and KDPH.

## Visualizing Distributions Across Multi-label Scenario

To empirically verify the topological superiority of KDPH, we visualized the embedding space of the complex Person category on MS COCO using t-SNE (Figure 5). The Baseline (Fig. 5a) exhibits severe proxy collapse, where proxies of distinct co-occurring classes (e.g., Car, Bicycle) physically collapse onto the central anchor. This visually confirms that rigid point proxies suffer from gradient contention, eliminating decision boundaries. In sharp contrast, KDPH (Fig. 5b) reveals a clear semantic-adaptive topology. The embedding space disentangles into coherent radial clusters based on highlevel affinity—grouping animate entities, vehicles, and food along distinct directional axes. This structured layout demonstrates that our anisotropic Kent proxies successfully capture latent semantic hierarchies, accommodating contextual diversity without destabilizing the core identity.

## 5 Conclusion

In this paper, we address the critical bottleneck of gradient contention in multi-label cross-modal hashing with the Kentbased Distributional Proxy Hashing (KDPH) framework. By shifting the paradigm from point proxies to anisotropic Kent distributions, our method enables proxies to absorb contextual conflicts by dynamically adjusting directional variance. This geometric flexibility allows the model to maintain stable semantic centroids while stretching to accommodate diverse label correlations. Extensive experiments on three benchmarks demonstrate that KDPH outperforms state-of-the-art methods.

## Acknowledgments

This work was supported by the Institute of Information Engineer ing, Chinese Academy of Sciences, under Grant E4101311F3, E4V06811F3. The reimbursement for this work was covered by this grant.

## References

[Bai et al., 2020] Cong Bai, Chao Zeng, Qing Ma, Jinglin Zhang, and Shengyong Chen. Deep adversarial discrete hashing for cross-modal retrieval. In Proceedings of the 2020 International Conference on Multimedia Retrieval, page 525–531, New York, NY, USA, 2020. Association for Computing Machinery.

[Cao et al., 2022] Min Cao, Shiping Li, Juntao Li, Liqiang Nie, and Min Zhang. Image-text retrieval: A survey on recent research and development. arXiv preprint arXiv:2203.14713, 2022.

[Chen et al., 2024] Bingzhi Chen, Zhongqi Wu, Yishu Liu, Biqing Zeng, Guangming Lu, and Zheng Zhang. Enhancing cross-modal retrieval via visual-textual prompt hashing. In Kate Larson, editor, Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 623–631. International Joint Conferences on Artificial Intelligence Organization, 8 2024. Main Track.

[Chua et al., 2009] Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhiping Luo, and Yantao Zheng. Nuswide: a real-world web image database from national university of singapore. In Proceedings ofthe ACM international conference on image and video retrieval, pages 1–9, 2009.

[He et al., 2020] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[Huiskes and Lew, 2008] Mark J. Huiskes and Michael S. Lew. The mir flickr retrieval evaluation. In MIR ’08: Proceedings of the 2008 ACM International Conference on Multimedia Information Retrieval, New York, NY, USA, 2008. ACM.

[Huo et al., 2024a] Yadong Huo, Qin Qibing, Jiangyan Dai, Wenfeng Zhang, Lei Huang, and Chengduan Wang. Deep neighborhood-aware proxy hashing with uniform distribution constraint for cross-modal retrieval. ACM Trans. Multimedia Comput. Commun. Appl., 20(6), March 2024.

[Huo et al., 2024b] Yadong Huo, Qibing Qin, Jiangyan Dai, Lei Wang, Wenfeng Zhang, Lei Huang, and Chengduan Wang. Deep semantic-aware proxy hashing for multi-label cross-modal retrieval. IEEE Transactions on Circuits and Systemsfor Video Technology, 34(1):576–589, 2024.

[Huo et al., 2024c] Yadong Huo, Qibing Qin, Wenfeng Zhang, Lei Huang, and Jie Nie. Deep hierarchy-aware proxy hashing with self-paced learning for cross-modal retrieval. IEEE Trans. on Knowl. and Data Eng., 36(11):5926–5939, November 2024.

[Jiang and Li, 2017] Qing-Yuan Jiang and Wu-Jun Li. Deep cross-modal hashing. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, July 2017.

[Kim et al., 2020] Sungyeon Kim, Dongwon Kim, Minsu Cho, and Suha Kwak. Proxy anchor loss for deep metric learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[Kirchhof et al., 2022] Michael Kirchhof, Karsten Roth, Zeynep Akata, and Enkelejda Kasneci. A non-isotropic probabilistic take on proxy-based deep metric learning. In European Conference on Computer Vision, pages 1–19. Springer, 2022.

[Li et al., 2018] Chao Li, Cheng Deng, Ning Li, Wei Liu, Xinbo Gao, and Dacheng Tao. Self-supervised adversarial hashing networks for cross-modal retrieval. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, June 2018.

[Li et al., 2021] Shen Li, Jianqing Xu, Xiaqing Xu, Pengcheng Shen, Shaoxin Li, and Bryan Hooi. Spherical confidence learning for face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[Lin et al., 2014] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common´ objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[Lin et al., 2018] Xudong Lin, Yueqi Duan, Qiyuan Dong, Jiwen Lu, and Jie Zhou. Deep variational metric learning. In Proceedings of the European Conference on Computer Vision, September 2018.

[Liu et al., 2023] Yishu Liu, Qingpeng Wu, Zheng Zhang, Jingyi Zhang, and Guangming Lu. Multi-granularity interactive transformer hashing for cross-modal retrieval. In Proceedings ofthe 31st ACM International Conference on Multimedia, MM ’23, page 893–902, New York, NY, USA, 2023. Association for Computing Machinery.

[Liu et al., 2024] Kaiming Liu, Yunhong Gong, Yu Cao, Zhenwen Ren, Dezhong Peng, and Yuan Sun. Dual semantic fusion hashing for multi-label cross-modal retrieval. In International Joint Conferences on Artificial Intelligence Organization, pages 4569–4577, 2024.

[Lu et al., 2019] Xu Lu, Lei Zhu, Zhiyong Cheng, Liqiang Nie, and Huaxiang Zhang. Online multi-modal hashing with dynamic query-adaption. In Proceedings ofthe 42nd international ACM SIGIR conference on research and development in information retrieval, pages 715–724, 2019.

[Movshovitz-Attias et al., 2017] Yair Movshovitz-Attias, Alexander Toshev, Thomas K. Leung, Sergey Ioffe, and Saurabh Singh. No fuss distance metric learning using proxies. In Proceedings of the IEEE International Conference on Computer Vision, Oct 2017.

[Park et al., 2019] Junyoung Park, Subin Yi, Yongseok Choi, Dong-Yeon Cho, and Jiwon Kim. Discriminative few-shot learning based on directional statistics. arXiv preprint arXiv:1906.01819, 2019.

[Qian et al., 2019] Qi Qian, Lei Shang, Baigui Sun, Juhua Hu, Hao Li, and Rong Jin. Softtriple loss: Deep metric learning without triplet sampling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, October 2019.

[Qin et al., 2024] Qibing Qin, Yadong Huo, Lei Huang, Jiangyan Dai, Huihui Zhang, and Wenfeng Zhang. Deep neighborhood-preserving hashing with quadratic spherical mutual information for cross-modal retrieval. IEEE Transactions on Multimedia, 26:6361–6374, 2024.

[Qin et al., 2025a] Qibing Qin, Yadong Huo, Wenfeng Zhang, Lei Huang, and Jie Nie. Deep discriminative boundary hashing for cross-modal retrieval. IEEE Transactions on Circuits and Systemsfor Video Technology, 35(10):10557– 10570, 2025.

[Qin et al., 2025b] Qibing Qin, Lei Wu, Wenfeng Zhang, Lei Huang, and Jie Nie. Deep semantic-consistent penalizing hashing for cross-modal retrieval. IEEE Transactions on Multimedia, 27:4613–4626, 2025.

[Roth et al., 2022] Karsten Roth, Oriol Vinyals, and Zeynep Akata. Non-isotropy regularization for proxy-based deep metric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7420–7430, 2022.

[Shi and Jain, 2019] Yichun Shi and Anil K Jain. Probabilistic face embeddings. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6902– 6911, 2019.

[Singh and Gupta, 2022] Avantika Singh and Shaifu Gupta. Learning to hash: a comprehensive survey of deep learningbased hashing methods. Knowledge and Information Systems, 64(10):2565–2597, 2022.

[Sun et al., 2022] Changchang Sun, Hugo Latapie, Gaowen Liu, and Yan Yan. Deep normalized cross-modal hashing with bi-direction relation reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4941–4949, 2022.

[Teh et al., 2020] Eu Wern Teh, Terrance DeVries, and Graham W Taylor. Proxynca++: Revisiting and revitalizing proxy neighborhood component analysis. In European conference on computer vision, pages 448–464. Springer, 2020.

[Tu et al., 2022] Junfeng Tu, Xueliang Liu, Zongxiang Lin, Richang Hong, and Meng Wang. Differentiable crossmodal hashing via multimodal transformers. In Proceedings of the 30th ACM International Conference on Multimedia, MM ’22, page 453–461, New York, NY, USA, 2022. Association for Computing Machinery.

[Tu et al., 2023] Rong-Cheng Tu, Xian-Ling Mao, Rong-Xin Tu, Binbin Bian, Chengfei Cai, Hongfa Wang, Wei Wei, and Heyan Huang. Deep cross-modal proxy hashing. IEEE

Trans. on Knowl. and Data Eng., 35(7):6798–6810, July 2023.

[Wang and Isola, 2020] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pages 9929–9939. PMLR, 2020.

[Wang et al., 2022] Yongxin Wang, Zhen-Duo Chen, Xin Luo, and Xin-Shun Xu. A high-dimensional sparse hashing framework for cross-modal retrieval. IEEE Transactions on Circuits and Systemsfor Video Technology, 32(12):8822– 8836, 2022.

[Xie et al., 2016] Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In International conference on machine learning, pages 478–487. PMLR, 2016.

[Xu et al., 2019] Ruiqing Xu, Chao Li, Junchi Yan, Cheng Deng, and Xianglong Liu. Graph convolutional network hashing for cross-modal retrieval. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 982–988. International Joint Conferences on Artificial Intelligence Organization, 7 2019.

[Xu et al., 2024] Cai Xu, Jiajun Si, Ziyu Guan, Wei Zhao, Yue Wu, and Xiyue Gao. Reliable conflictive multi-view learning. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 16129–16137, 2024.

[Yan et al., 2025] Shuanglin Yan, Jun Liu, Neng Dong, Liyan Zhang, and Jinhui Tang. Cross-modal collaborative representation learning for text-to-image person retrieval. In Proceedings ofthe thirty-fourth international joint conference on artificial intelligence, 2025.

[Zhang et al., 2023] Zheng Zhang, Haoyang Luo, Lei Zhu, Guangming Lu, and Heng Tao Shen. Modality-invariant asymmetric networks for cross-modal hashing. IEEE Transactions on Knowledge and Data Engineering, 35(5):5091– 5104, 2023.

[Zhe et al., 2018] Xuefei Zhe, Shifeng Chen, and Hong Yan. Directional statistics-based deep metric learning for image classification and retrieval. arXiv preprint arXiv:1802.09662, 2018.

[Zhu et al., 2025a] Congcong Zhu, Wei Hu, Jinkui Hou, Qibing Qin, Wenfeng Zhang, and Lei Huang. Deep adaptive gradient-triplet hashing for cross-modal retrieval. Expert Systems with Applications, 291:128566, 2025.

[Zhu et al., 2025b] Congcong Zhu, Qibing Qin, Wenfeng Zhang, and Lei Huang. Deep neighbor-coherence hashing with discriminative sample mining for supervised cross-modal retrieval. Expert Systems with Applications, 279:127365, 2025.

## A Extended Analysis for Main Results: Semantic Topological Evolution

To interpret the quantitative improvements reported in the main results from a topological perspective, we investigate the geometric organization of the learned proxy space. Specifically, we aggregate the 80 fine-grained categories of the MS COCO dataset into 12 semantic super-groups (e.g., animal, vehicle, electronic,furniture) based on the standard semantic taxonomy. This allows us to evaluate the Semantic Disentanglement capability of the model by measuring two key metrics: Intra-Group Similarity (the cosine cohesion within a super-group) and Inter-Group Similarity (the cosine overlap between distinct super-groups). An ideal semantic space should maximize the former while minimizing the latter, creating a structured manifold where conceptually related classes cluster tightly and unrelated ones are strictly separated.

Figure 6 visualizes this topological evolution. Subplot (a) plots the semantic trajectory of each super-group in a 2D coordinate system, where the y-axis represents Intra-Group Similarity (higher is better) and the x-axis represents Inter-Group Similarity (lower is better). The Baseline proxies (orange circles) are scattered loosely across the space, indicating a high degree of semantic entanglement where distinct concepts (e.g., furniture and electronic) share excessive similarity. In contrast, the KDPH proxies (blue stars) exhibit a distinct Top-Left Migration. The gray arrows highlight this consistent shift towards the Ideal Corner (high cohesion, low interference). For instance, the vehicle and outdoor groups show dramatic improvements in separation, verifying that our anisotropic distribution effectively pushes irrelevant semantic clusters apart.

Subplots (b) and (c) further quantify these gains. The Separation Score, defined as the difference between Intra- and Intergroup similarities, is consistently higher for KDPH across nearly all categories. The heatmap in Subplot (c) decomposes this improvement, revealing that KDPH achieves this separation primarily through a substantial reduction in Inter-Group similarity (blue cells in the middle row) and a simultaneous boost in Intra-Group cohesion (red cells in the top row). This confirms that the performance gains observed in Table 1 stem from a fundamentally better-structured semantic embedding space, where the distributed proxies enforce a rigorous hierar chy that aligns with human conceptual understanding.

## B Extended Analysis for Proxy Oscillation and Stability

To substantiate the robustness of the findings presented in RQ1 (Can our proposed model effectively mitigate the ’Chaos Proxy Oscillation’ problem?), we provide a comprehensive visualization of training dynamics across a broader spectrum of categories. While the main text highlights representative cases, this appendix expands the analysis to twelve additional classes from the MS COCO dataset. These categories are stratified into high and low co-occurrence groups to demonstrate the universal effectiveness of the KDPH framework across varying degrees of semantic complexity.

## B.1 Mitigation of Oscillation in Complex Scenarios

Figure 8 delineates the training trajectories for complex, high co-occurrence categories such asfork, microwave, pottedplant, mouse, keyboard, and laptop. In these scenarios, dense semantic overlaps typically induce severe gradient contention for traditional methods. Observing the baseline metrics (orange curves), it is evident that rigid point proxies suffer from persistent oscillation, with the Proxy Flip Rate fluctuating dangerously between 5% and 20% even in late training stages. This instability directly correlates with the jagged and suboptimal average precision curves, confirming that static point representations fail to anchor semantically ambiguous samples. In sharp contrast, the proposed KDPH framework (blue curves) effectively decouples these conflicting gradients, suppressing the flip rate to near-zero levels almost immediately. The resulting precision curves exhibit a smooth, monotonic ascent, confirming that the anisotropic Kent distributions successfully absorb the variance that otherwise destabilizes point-based models.

## B.2 Preservation of Stability in Simple Scenarios

Figure 9 illustrates the dynamics for semantically distinct, low co-occurrence categories, including sandwich, boat, cow, banana, airplane, and elephant. In these semantically pure environments, the baseline method performs adequately with minimal oscillation. Crucially, KDPH mirrors this stability, maintaining low flip rates and high precision without introducing unnecessary variance or computational noise. This comparative analysis serves as a vital validation of robustness: it demonstrates that the distributed proxy mechanism is adaptive, providing necessary topological flexibility for complex classes while behaving like a stable, confident estimator in unambiguous scenarios.

## B.3 Quantitative Correlation between Proxy Chaos and Performance

To move beyond visual inspection and quantitatively verify the causal link between proxy oscillation and retrieval degradation, we introduce a composite metric named the Chaos Index. This scalar, normalized between 0 and 1, aggregates four kinematic properties of the proxy’s learning trajectory: Path Efficiency (ratio of displacement to total path length), Directional Consistency (cosine similarity between consecutive steps), Movement Entropy (randomness of step magnitudes), and Autocorrelation (predictability based on vector dot products). The Chaos Index is computed by averaging the normalized entropy with the inverted values of the three ordered metrics, where a value of 1 indicates maximum stochasticity. We also define the Average Co-occurring Labels (ACL) to quantify semantic complexity: $\begin{array} { r } { A C L ( c ) \stackrel { \smile } { = } \frac { 1 } { | N _ { c } | } \sum _ { i \in N _ { c } } ( | l _ { i } | _ { 1 } \stackrel { \smile } { - } 1 ) } \end{array}$ representing the mean number of concurrent labels associated with class c.

Figure 10 presents the analysis for high-complexity classes $( \mathrm { A v g ~ } \bar { A } C L \stackrel { . } { \approx } 4 . 1 )$ on the MS COCO dataset (64 bits). A distinct inverse correlation is observed between the Chaos Index and retrieval performance (Precision@50). The baseline method (orange bars) exhibits elevated Chaos Indices across all dense categories, driven by low path efficiency and high movement entropy. This kinematic instability directly suppresses the precision (orange dotted line). Conversely, KDPH (blue bars) significantly minimizes the Chaos Index, transform ing the erratic trajectory into a directed, efficient optimization path. This stabilization translates directly into substantial performance gains, as evidenced by the superior Precision@50 scores (blue star line), confirming that resolving gradient contention is the underlying mechanism behind our performance improvements.

![](images/1eae5529508226e37d1d82825097a53692c80c5bc95869c4a1e062103b88981c.jpg)

![](images/0dd4dbb0a152911d623471d31958fe7baf90ba3ae773043e3ecbe2658a25cb43.jpg)

(c) Performance Gain (Ours vs Baseline)  
![](images/1c3ce5125287c6dc1ac67f96f48cb4df38459ab1bf6d3d465eced9671f3f9193.jpg)  
Figure 6: Analysis of Semantic Proxy Distribution Shifts. (a) The scatter plot illustrates the topological quality of semantic super-groups. KDPH (blue stars) consistently migrates towards the top-left Ideal Corner compared to the Baseline (orange circles), indicating higher intra-group cohesion and lower inter-group interference. (b) The Separation Scores confirm that KDPH establishes sharper boundaries between semantic concepts. (c) The heatmap details the specific contributions of Intra-Group Gain and Inter-Group Reduction to the overall performance improvement.

In parallel, Figure 11 details the metrics for low-complexity classes (Avg $A \bar { C } \bar { L } \approx 1 . 1 )$ . In these semantically sparse scenarios, the gradient signals are naturally coherent. Consequently, both the Baseline and KDPH exhibit low Chaos Indices and high, comparable Autocorrelation scores. The overlapping performance curves further reinforce our conclusion: the proposed anisotropic Kent distribution effectively reduces to a stable estimator in simple scenarios, only activating its shapedeforming capacity when high variance demands it. This con firms that KDPH provides an improvement without inference overhead, offering robustness in complex regimes without compromising the intrinsic stability of simple categories.

## C Extended Analysis for Quantifying Proxy Collapse via GTSD

To provide a fine-grained perspective on the Proxy Collapse phenomenon discussed in RQ2, we introduce the Ground-Truth Score Divergence (GTSD). While retrieval metrics like mAP measure global ranking, GTSD specifically quantifies the severity of the gradient competition within a single multi-label instance. It measures the disparity between the model’s confi dence in the best-recognized label versus the worst-recognized label. Formally, for a sample x with a ground-truth label set $\mathcal { L } _ { x } = \{ l _ { 1 } , . . . , \dot { l } _ { K } \}$ and a scoring function $S ( x , c )$ , GTSD is defined as:

$$
\mathrm { G T S D } ( x ) = \left( \operatorname* { m a x } _ { l _ { i } \in \mathcal { L } _ { x } } S ( x , l _ { i } ) \right) - \left( \operatorname* { m i n } _ { l _ { j } \in \mathcal { L } _ { x } } S ( x , l _ { j } ) \right)\tag{11}
$$

A high GTSD indicates that the model is collapsing onto a subset of dominant labels (head classes) while suppressing subordinate ones (tail classes). Conversely, a low GTSD implies intra-sample equity, where the proxy mechanism successfully maintains high compatibility for all co-occurring semantic concepts simultaneously.

Figure 12 elucidates the internal scoring dynamics on the MS COCO dataset as the semantic complexity (number of labels) increases from 3 to 8. Subplot (a) reveals that the

![](images/fd5b51d0ed29e5cccbab1e245a0c1d7f69991ea353cc8b8340b5d67adf1f8061.jpg)

![](images/59a717015690173c461c294ecdada2dce052454775cdf19c419064b20ba7e597.jpg)

![](images/13e729274ee9e4abca1c91384c18ae881b6b47d554c53d5db41067aa541d8b7c.jpg)

![](images/b3321faa597d068da08f06b9516f96825f0a2755e2991dd53f2695109b7c81c5.jpg)  
Figure 7: Parameter Sensitivity Analysis on MS COCO (@64 bits). We evaluate the impact of the loss weights $\epsilon , \eta , \zeta$ and the number of principal axes $N _ { a x e s }$ on the Image-to-Text (I2T) mAP. The results justify our default settings $( \epsilon = 4 0 , \eta = 0 . 1 , \zeta = 0 . 3 , N _ { a x e s } = 8 )$ .

Baseline (orange) suffers from an exacerbated divergence as the label density grows. This confirms that rigid point proxies, unable to satisfy conflicting gradients, prioritize the easiest category at the expense of others. In contrast, KDPH (blue) maintains a significantly tighter score distribution. Even more telling is Subplot (b), which shows that the Baseline achieves higher maximum scores $( > 0 . 9 )$ compared to KDPH $( \approx 0 . 6 )$ . This is not indicative of superior performance, but rather of overconfidence and collapse: the point proxy is pulled entirely towards the dominant centroid, abandoning the geometric structure required for the remaining labels.

Subplot (c) quantifies the relative improvement of KDPH over the Baseline in reducing GTSD. The benefit of our anisotropic modeling scales broadly with complexity, achieving a 14.8% reduction in divergence for samples with 8 labels. This empirically proves that the Kent-based distribution effectively utilizes its shape parameters to cover the semantic manifold of multiple labels, preventing the topological collapse that characterizes traditional point-based hashing.

## D Hyperparameter Sensitivity Analysis

To verify the robustness of KDPH and investigate the impact of different loss components and geometric constraints, we conducted a detailed sensitivity analysis on the MS COCO dataset with a hash code length of 64 bits. We specifically examined four critical hyperparameters: the weights for the regularization term (ϵ), the cross-modal alignment term (ζ), and the uniform distribution constraint (η), as well as the number of principal axes $( N _ { a x e s } )$ in the Kent distribution. The results are illustrated in Figure 7.

## D.1 Impact of Loss Coefficients

Irrelevance Regularization (ϵ). As shown in the top-left plot of Figure 7, the model’s performance exhibits a significant sensitivity to the Multi-Modal Irrelevance Regularization weight ϵ. The mAP improves steadily as ϵ increases from

1 to 40, confirming that strictly penalizing high similarities between semantically disjoint pairs is crucial for sharpening decision boundaries. However, setting ϵ too high (e.g., 100) causes a sharp performance degradation. We hypothesize that excessive regularization dominates the gradient optimiza tion, suppressing the learning of positive semantic correlations. Thus, $\epsilon = 4 0$ strikes the optimal balance.

Uniform Constraint (η). The top-right plot illustrates the effect of the Optimal Transport-based uniform constraint. We observe that a relatively small weight $( \eta = 0 . 1 )$ yields the best performance. As η increases, the performance generally declines. This suggests that while maximizing code entropy is beneficial for preventing bit collapse, over-enforcing uniformity may distort the intrinsic semantic manifold, forcing the embeddings into an unnatural distribution that hinders discrimination. Therefore, we treat this as a soft constraint with $\eta = 0 . 1$

Cross-Modal Alignment (ζ). The bottom-left plot analyzes the contribution of the contrastive alignment loss. The performance peaks at $\zeta = 0 . 3$ . While cross-modal alignment is essential for bridging the modality gap, an overly large ζ might cause the model to focus excessively on instance-level matching at the expense of class-level semantic structure learned by the proxy module.

## D.2 Impact of Anisotropic Geometry $( N _ { a x e s } )$

A unique contribution of KDPH is the modeling of anisotropic variance via the Kent distribution’s principal axes. The bottomright plot in Figure 7 investigates the optimal number of axes $N _ { a x e s }$ to model the shape parameter $\beta .$ . Interestingly, the performance does not increase monotonically with the number of axes. The best result is achieved at $\dot { N _ { a x e s } } = 8$ (which corresponds to $K / 8$ for 64-bit codes).

• When $N _ { a x e s }$ is too small (e.g., 4), the proxy lacks sufficient degrees of freedom to capture complex, multidirectional semantic variances.

• Conversely, increasing $N _ { a x e s }$ beyond 16 $( \mathbf { e . g . } , 3 2 \ \mathrm { o r } \ 6 3 )$ leads to a substantial drop in accuracy. This phenomenon can be attributed to the curse of dimensionality and overfitting: utilizing too many shaping parameters allows the proxy to model noise rather than meaningful semantic directions, destabilizing the optimization.

This finding validates our choice of $N _ { a x e s } = K / 8$ as a parsimonious yet effective configuration for modeling anisotropic semantic manifolds in the Hamming space.

![](images/1c4110015f111dfe50d2b146a077fc8e43c57906c639662e017460bf01437fcc.jpg)  
Figure 8: Supplementary for RQ1: Training Dynamics on High Co-occurrence Classes. The Baseline (orange) exhibits chaotic oscillation (high flip rates) and jagged precision curves due to gradient contention. KDPH (blue) achieves rapid stabilization and superior convergence, validating its ability to resolve semantic conflicts.

![](images/a0c40dc3f53928a8ff2684cde05784fb46bde08ae6b47c926c99e393802a7fb4.jpg)  
Figure 9: Supplementary for RQ1: Training Dynamics on Low Co-occurrence Classes. Both methods maintain high stability. This confirms that KDPH’s anisotropic modeling is robust and does not negatively impact learning in simple, distinct semantic scenarios.

![](images/fcd28780811c1a764a128220d5796dacb8be58f384f97bba4d60db3f38c44b18.jpg)

![](images/d332e568b495f5100fc432e78c3cac48e8af410b5cd7896b7a96ac72e65d11f6.jpg)

![](images/014de4375634c93b8506c53c8b7b1e8e4fe0b48bb157c915e29a55998e8b87e0.jpg)

![](images/0b9b66e35643e9f5addb7d81667743b32edef6f774e1e245e4cc3837d2682da9.jpg)

![](images/56a9f4b62aa3ee036f6710f7733b86a2e3355645cb7804123834b65154b6e2a0.jpg)  
Figure 10: Chaos Index vs. Real Performance (High Co-occurrence Classes). The charts correlate kinematic stability (Chaos Index, bars) with retrieval accuracy (Precision@50, lines). For complex classes (Avg ACL ≈ 4.1), the Baseline suffers from high chaos and low precision. KDPH effectively suppresses this chaos, resulting in higher path efficiency and significantly improved retrieval performance.

![](images/191a199dbf5edbc6ae13ff256d60645f64dbcb2076a0d7709564934532d546b8.jpg)

![](images/532fcd800d2a3943adc06869fa47cf912fdbbfda9a07da77ead77641b9648411.jpg)

![](images/d4f1fc8c2af89b858b304e7f3799ed4d8da8ab49ce97573defcc0d064e8b5ff8.jpg)

![](images/5eb9ddb45593aae365a8d678ba80349d451a9747323d94a87bcad6d9a21a387a.jpg)

![](images/e789fde49411290c3ad5990b8fca0cbbc943a994b1d6433202f2aba38e613059.jpg)  
Figure 11: Chaos Index vs. Real Performance (Low Co-occurrence Classes). In scenarios with low semantic interference (Avg ACL ≈ 1.1), both methods maintain low Chaos Indices and high Autocorrelation. This demonstrates that KDPH retains the optimization efficiency of point-based methods when gradient contention is absent.

![](images/73527832f538c69c9147397cea5fa5c356863160d73fca30375b431ac8f7bcd7.jpg)  
(a) Ground-Truth Score Divergence

![](images/25f33cfa962bf0d43bc55925824ce71de4073bba730b188e27b8d56dfa82e749.jpg)  
(b) Maximum Ground-Truth Score

![](images/f0deecd60844933a67bea806a407cf915995f483796635a9d5991d935b50e0d1.jpg)  
(c) Relative Improvement  
Figure 12: Ground-Truth Score Divergence (GTSD) Analysis. (a) As the number of labels increases, the Baseline exhibits a widening gap between the best and worst scored labels, indicating proxy collapse. KDPH maintains a stable, lower divergence. (b) The Baseline’s high maximum scores suggest overfitting to dominant labels. (c) KDPH demonstrates increasing efficacy in complex scenarios, reducing th divergence by up to 14.8% in dense multi-label settings.