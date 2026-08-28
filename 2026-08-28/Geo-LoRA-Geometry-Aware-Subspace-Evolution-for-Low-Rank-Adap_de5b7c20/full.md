# Geo-LoRA: Geometry-Aware Subspace Evolution for Low-Rank Adaptation in Continual Learning

Yibo Feng

University of Electronic Science and Technology of China

Chengdu, China

2022090917012@std.uestc.edu.cn

## Abstract

Rehearsal-free class-incremental learning (CIL) with LoRA adapters remains challenging because the low-rank subspaces updated across tasks evolve without geometric control, causing unstable shared representations and repetitive collapse of task-specific updates into previously occupied directions. We introduce Geo-LoRA, a geometry-aware framework that explicitly regulates how low-rank subspaces—shared and task-specific—evolve during continual learning. For the shared branch, Subspace Projection Preservation (SPP) constrains consecutive updates to follow smooth trajectories on the Grassmann manifold, and Adaptive Core–Slack Alignment (ACSA) decomposes transitions into principal and residual components, aligning the former while modulating the latter to balance stability and plasticity. For the task-specific branch, Median-Calibrated Block Overlap (MCBO) imposes a statistical constraint via normalized projection overlap, penalizing excessive reuse to mitigate subspace crowding. These constraints jointly regulate the evolution of all LoRA subspaces across layers and tasks without introducing additional adapter types beyond standard LoRA. Geo-LoRA provides a principled geometric formulation for continual low-rank adaptation and consistently achieves state-of-the-art performance across multiple benchmark datasets and diferent task lengths.

## CCS Concepts

• Computing methodologies → Machine learning; Computer vision; Transfer learning.

## Keywords

Continual Learning; Class-Incremental Learning; Low-Rank Adaptation; Parameter-Eficient Fine-Tuning

## ACM Reference Format:

Yibo Feng. 2026. Geo-LoRA: Geometry-Aware Subspace Evolution for Low-Rank Adaptation in Continual Learning. In Eficient Systems and AIfor the Planet (GreenMM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3840750.3841444

## 1 Introduction

Class-incremental learning (CIL)[11, 18, 19, 42] requires models to learn new classes over time without forgetting previously acquired knowledge. With the rise of pre-trained vision transformers (ViTs), recent work increasingly adopts parameter-eficient fine-tuning (PEFT)[3, 10, 41] to enable exemplar-free CIL, where the backbone remains frozen and lightweight modules handle continual adaptation. Among PEFT techniques, Low-Rank Adaptation (LoRA)[14] is particularly attractive due to its eficiency and its interpretation as learning task-dependent low-dimensional feature subspaces.

![](images/8d103839ca840aa1afec371e0d5c74b989860d5482d0ded517672dfe2f7c2005.jpg)  
Figure 1: Comparison of diferent PTM-based CIL methods on ImageNet-R dataset (20 tasks). The X-axis stands for the number of trainable parameters, and the Y-axis represents the average accuracy. For a fair comparison, all methods are based on pre-trained ViT-B/16-IN1K.

However, although existing LoRA-based CIL methods[37, 39, 43] have highlighted interference among low-rank updates, they still provide limited explicit control over the geometric evolution of these low-rank subspaces. Most approaches attach independent task-specific LoRA modules for each task, while recent dual-branch designs[12] introduce shared adapters across tasks. In both cases, low-rank directions learned at diferent stages are optimized mostly in isolation: shared subspaces may drift across tasks, and deep taskspecific updates may repeatedly occupy previously used directions. These efects lead to misaligned shared representations and subspace crowding in layers, undermining the stability[24]–plasticity[6, 33] trade-of.

These observations motivate a diferent perspective. Instead of viewing continual LoRA tuning as merely adding or freezing adapters, we formulate it as regulating the evolution of low-rank subspaces across tasks. This geometric viewpoint forms the basis of Geo-LoRA. Rather than modifying the LoRA architecture itself, Geo-LoRA introduces projection-based geometric objectives that guide low-rank updates to evolve within a stable yet flexible envelope.

To enable such geometric control, we adopt the Dual-LoRA architecture from CL-LoRA[12] as a structural backbone. This design separates continual adaptation into two regimes: shallow layers maintain a shared low-rank subspace that accumulates transferable structure, while deeper layers host task-specific LoRA updates for semantic specialization. This decomposition allows diferent geometric constraints to regulate shared and task-specific subspaces.

Importantly, our goal is not to propose a new LoRA architecture, but to introduce geometry-aware objectives that stabilize and structure the learning dynamics within this framework.

Within this hierarchy we design three complementary objectives. Subspace Projection Preservation (SPP) constrains consecutive shared updates using Grassmann-consistent projection distance to reduce cross-task drift. Adaptive Core–Slack Alignment (ACSA) aligns principal shared directions while adaptively moderating residual directions to balance stability and plasticity. Median Calibrated Block Overlap (MCBO) regulates deep-layer updates through normalized projection overlap, penalizing only statistically excessive reuse of historical directions. Together these components form a multi-level geometric regularization framework that controls subspace evolution without introducing additional adapter types and with modest training overhead.

As illustrated in Figure 1, Geo-LoRA achieves state-of-the-art (SOTA) performance while maintaining a small number of trainable parameters.

Our key contributions are summarized as follows:

(1) We provide a geometric subspace-evolution perspective on LoRA-based CIL, ofering a principled view of continual low-rank adaptation.

(2) We propose Geo-LoRA, integrating SPP, ACSA, and MCBO into the Dual-LoRA architecture to regularize shared and taskspecific subspaces under a unified projection-based principle.

(3) We achieve SOTA results across multiple datasets and tasks, demonstrating that geometry-aware subspace regulation provides an efective and scalable paradigm for exemplar-free continual learning.

## 2 Related work

## 2.1 Parameter-Eficient Fine-Tuning

Class-incremental learning (CIL) aims to acquire new classes over time without forgetting previously learned ones. Recent work increasingly leverages large pre-trained vision transformers (ViTs), whose strong general-purpose representations reduce the need to store exemplars. Parameter-eficient fine-tuning (PEFT) approaches, such as adapters[9, 22, 23, 25, 27, 31], prompt tuning[2, 15, 15– 17, 21, 38], and LoRA—enable continual adaptation while keeping the backbone frozen, thereby limiting full-parameter interference across tasks. However, many PEFT-based CIL methods still rely on task-specific modules that grow linearly with the number of tasks, leading to parameter overhead and fragmented cross-task representations. A few recent studies explore shared components across tasks, but they typically regularize parameters[12] or optimize prediction objectives at the module level[35], rather than explicitly modeling how low-rank feature subspaces evolve over time. Moreover, recent LoRA-based CIL methods, including InfLoRA[20] and related approaches[4, 12, 28, 34], have already highlighted the importance of interference among low-rank updates. Our work builds on this line of research, but difers in focusing explicitly on the geometric evolution of both shared and task-specific LoRA subspaces across tasks.

## 2.2 Geometric Evolution in LoRA

LoRA adapts frozen weights through low-rank matrices that can be interpreted as learning new directions in a task-dependent feature subspace[1, 13]. In multi-task or continual scenarios, these low-dimensional directions may rotate, collapse, drift, or interfere with one another, leading to misaligned representations and degraded transfer[4, 28, 34]. Prior work has attempted to alleviate such efects using orthogonality, sparsity, adaptive rank allocation, or shared/task-specific branch designs, but these methods mainly regulate parameters, overlap, or module composition rather than directly controlling the trajectories of the underlying low-rank subspaces[12]. Subspace-based formulations have also appeared in domain adaptation[5, 8, 40], yet they typically operate on feature covariance or classifier weights instead of LoRA-induced subspaces under continual task sequences. Moreover, existing methods rarely distinguish between principal directions that should remain stable and residual directions that can support plasticity, nor do they address the block-wise imbalance caused when deep adapters repeatedly reuse historical subspaces. Geo-LoRA addresses these gaps through three projection-based mechanisms—SPP, ACSA, and MCBO—that explicitly regulate shared drift, core-versus-slack evolution, and excessive historical reuse, providing a unified geometric perspective on continual low-rank adaptation.

## 3 Preliminaries

In rehearsal-free class-incremental learning (CIL), a model learns a sequence of tasks with disjoint label spaces without storing past samples. A pre-trained ViT $f _ { \theta }$ is frozen, and lightweight adapters are inserted into Transformer blocks:

$$
z = f _ { \boldsymbol \theta } ( x ) + f _ { \mathrm { a d a p t } } ( x ) .\tag{1}
$$

LoRA implements each adapter via a rank-� update �� to a frozen weight � :

$$
z = W x + A B x , \quad r \ll \operatorname* { m i n } ( d , k ) .\tag{2}
$$

Most CIL methods allocate task-specific LoRA modules $( A _ { t } , B _ { t } )$ for every task, causing linear parameter growth and limited knowledge sharing.

Geo-LoRA follows the Dual-LoRA architecture introduced in prior work, but reinterprets it through a geometric viewpoint. The first � blocks contain a single shared LoRA module used across all tasks, while deeper blocks host task-specific LoRA modules. Rather than treating these as independent parameter augmentations, Geo-LoRA regards both branches as evolving low-rank subspaces: the shallow shared subspace should evolve smoothly across tasks, whereas deep task-specific subspaces should either expand or reuse prior directions in a controlled way.

Formally, the feature update for task � at block � is

$$
\begin{array} { r } { z _ { i } ^ { t } = f _ { \theta } ^ { i } ( z _ { i - 1 } ^ { t } ) + \left\{ \begin{array} { l l } { A _ { s } ^ { i } B _ { s } ^ { i } z _ { i - 1 } ^ { t } , } & { i \leq l , } \\ { A _ { t } ^ { i } B _ { t } ^ { i } z _ { i - 1 } ^ { t } , } & { i > l . } \end{array} \right. } \end{array}\tag{3}
$$

This geometric reinterpretation provides the foundation for the three regularizers introduced next—SPP, ACSA, and MCBO—which collectively regulate how shared and task-specific LoRA subspaces evolve throughout continual learning.

![](images/36ab85e9c5cec128b26c7656ee39bb5442129b1b5ab06faf13fc2ef80941769c.jpg)  
Figure 2: Overview of Geo-LoRA. Geo-LoRA regulates the geometric evolution of LoRA-induced subspaces through three lightweight constraints. (1) SPP enforces smooth transitions of the shared low-rank subspace by minimizing Grassmannian projector drift. (2) ACSA decomposes subspace changes into core and slack components, preserving aligned directions while adaptively controlling residual activation. (3) MCBO suppresses excessive reuse of historical directions in deep task-specifi blocks via median-calibrated normalized overlaps. Together, these objectives coordinate stable yet flexible subspace evolution across tasks without increasing inference cost.

## 4 Method

In this section, we present Geo-LoRA, a unified geometric framework for continual low-rank adaptation, and Figure 2 summarizes its workflow. Instead of introducing separate LoRA modules for each task, Geo-LoRA explicitly constrains how LoRA-induced lowrank subspaces evolve across tasks. We adopt a dual-LoRA architecture with (i) a shallow shared branch for cross-task knowledge preservation and (ii) a deep task-specific branch for task-level spe cialization. Both branches are regularized by lightweight geometric constraints that balance stability and plasticity without adding inference-time overhead.

## 4.1 Subspace Projection Preservation

In Geo-LoRA, shallow LoRA modules are shared across tasks and continually refined, accumulating transferable structure in early Transformer layers. However, continual updates can drift the underlying subspace, so an input feature previously represented in $S _ { t - 1 }$ may be mapped to a rotated $S _ { t }$ . Since class prototypes from earlier tasks are fixed under $S _ { t - 1 }$ , such rotations induce feature–prototype misalignment and exacerbate forgetting.

To counter this, we regularize the evolution ofthe shallow shared subspaces via Subspace Projection Preservation (SPP). After finishing task �−1, we concatenate the up-projection matrices of all shallow shared LoRA modules to form $A _ { t - 1 } \in \mathbb { R } ^ { d \times q _ { t - 1 } } ;$ , and analogously obtain $A _ { t } \in \mathbb { R } ^ { d \times q _ { t } }$ after finishing task �. Here $q _ { t }$ denotes the total concatenated width (i.e., the sum of low-rank widths across the shallow shared LoRA modules), and $A _ { t }$ represents the current state of the shared branch after task � (shared parameters updated sequentially across tasks). This concatenation aggregates low-rank directions from all shallow shared layers into a single global representation of the shared subspace.

Basis invariance and Grassmann geometry. Crucially, LoRA identifies a low-rank subspace rather than a unique parameter matrix: any change of basis within the rank-� subspace (e.g., � →�� with invertible � ) leaves the represented subspace unchanged. Therefore, distances in the raw parameter space are not well-defined for comparing LoRA updates across tasks. Instead, we compare subspaces on the Grassmann manifold ${ \mathrm { G r } } ( q , d )$ , the set of all $q -$ dimensional linear subspaces of $\mathbb { R } ^ { d }$ , where each point corresponds to an equivalence class of bases spanning the same subspace.

We represent each subspace by its orthogonal projector (a basisinvariant representation):

$$
\begin{array} { r } { P _ { t - 1 } = A _ { t - 1 } ( A _ { t - 1 } ^ { \top } A _ { t - 1 } ) ^ { - 1 } A _ { t - 1 } ^ { \top } , \qquad P _ { t } = A _ { t } ( A _ { t } ^ { \top } A _ { t } ) ^ { - 1 } A _ { t } ^ { \top } . } \end{array}\tag{4}
$$

The projector � maps any vector to its orthogonal projection onto col(�), making it a canonical way to represent and compare subspaces independent of basis choices. (Equivalently, with thin-QR $A = Q R$ we compute $P = Q Q ^ { \top }$ , which is numerically stable and diferentiable.)

We then minimize the distance between consecutive projectors,

$$
\mathcal { L } _ { \mathrm { S P P } } = \Vert P _ { t } - P _ { t - 1 } \Vert _ { F } ^ { 2 } ,\tag{5}
$$

which is the squared chordal distance on the Grassmann manifold and thus measures subspace variation independent of basis choices.

Formally, for any input feature $x ,$

$$
\left\| P _ { t } x - P _ { t - 1 } x \right\| _ { 2 } ^ { 2 } \leq \left\| P _ { t } - P _ { t - 1 } \right\| _ { F } ^ { 2 } \left\| x \right\| _ { 2 } ^ { 2 } ,\tag{6}
$$

so reducing $\mathcal { L } _ { \mathrm { S P P } }$ directly limits feature drift in the shared branch, preserving compatibility with earlier prototypes and mitigating forgetting. Eq. (6) follows from $\| ( P _ { t } - P _ { t - 1 } ) x \| _ { 2 } \leq \| P _ { t } - P _ { t - 1 } \| _ { 2 } \| x \| _ { 2 } \leq$ $\| P _ { t } - P _ { t - 1 } \| _ { F } \| x \| _ { 2 }$ .

## 4.2 Adaptive Core–Slack Alignment

While SPP constrains the global evolution of the shared LoRA subspace, it treats all directions uniformly and does not distinguish between (i) rotations within the existing shared subspace and (ii) subspace expansion where previously inactive directions become activated by new tasks. These two forms of geometric variation have diferent implications: the former should be tightly regularized to preserve previously learned representations, whereas the latter reflects necessary plasticity. To address this, we introduce Adaptive Core–Slack Alignment (ACSA), which separates and jointly controls Grassmannian core alignment and residual subspace activation.

Grassmannian core alignment. Let the concatenated shallow shared LoRA matrices for tasks �−1 and � be $A _ { t - 1 } \in \mathbb { R } ^ { d \times q _ { t - 1 } }$ and $A _ { t } \in \mathbb { R } ^ { d \times q _ { t } }$ . We extract orthonormal bases via reduced QR decompositions:

$$
A _ { t - 1 } = Q _ { t - 1 } R _ { t - 1 } , \qquad A _ { t } = Q _ { t } R _ { t } ,
$$

where $Q _ { t - 1 } \in \mathbb { R } ^ { d \times r _ { t - 1 } }$ and $Q _ { t } \in \mathbb { R } ^ { d \times r _ { t } }$ have orthonormal columns, and $r _ { t - 1 } = \mathrm { r a n k } ( A _ { t - 1 } ) \ \leq \ q _ { t - 1 } , r _ { t } \ =$ rank $\left( A _ { t } \right) \ \leq \ q _ { t }$ . The matrices $Q _ { t - 1 }$ and $Q _ { t }$ parameterize points on the Grassmann manifolds $\mathrm { G r } ( r _ { t - 1 } , d )$ and $\mathrm { G r } ( r _ { t } , d )$

Let $r _ { c } ~ = ~ \mathrm { m i n } ( r _ { t - 1 } , r _ { t } )$ and define their principal-angle cores $\tilde { Q } _ { t - 1 } = Q _ { t - 1 } ( : , 1 : r _ { c } )$ and $\tilde { Q } _ { t } = Q _ { t } ( : , 1 : r _ { c } )$ . The quantity

$$
s _ { \mathrm { c o r e } } ^ { ( t ) } = \frac { \| \tilde { Q } _ { t - 1 } ^ { \top } \tilde { Q } _ { t } \| _ { F } ^ { 2 } } { r _ { c } } ,\tag{7}
$$

is a Grassmann-invariant similarity measure:

$$
\| \tilde { \mathcal { Q } } _ { t - 1 } ^ { \top } \tilde { \mathcal { Q } } _ { t } \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { r _ { c } } \cos ^ { 2 } \theta _ { i } ,\tag{8}
$$

where $\theta _ { i }$ are the principal angles between the two subspaces. Thus,

$$
\mathcal { L } _ { \mathrm { c o r e } } ^ { ( t ) } = 1 - s _ { \mathrm { c o r e } } ^ { ( t ) }\tag{9}
$$

penalizes rotations of the established shared subspace while $\mathbf { r e - }$ specting Grassmann geometry.

Residual slack subspace alignment. Beyond the Grassmann core, new tasks may introduce additional active directions. To quantify how these residual directions interact with previously learned subspaces, we measure symmetric overlaps between residual bases:

$$
\begin{array} { r l } & { e _ { \mathrm { s l a c k } } ^ { ( t ) } = \displaystyle \frac { 1 } { 2 r _ { c } } \left( \mathbb { I } [ r _ { t - 1 } > r _ { c } ] \ \lVert Q _ { t } ^ { \top } Q _ { t - 1 } ( : , r _ { c } + 1 : r _ { t - 1 } ) \rVert _ { F } ^ { 2 } \right. } \\ & { ~ \quad \quad \left. + \mathbb { I } [ r _ { t } > r _ { c } ] \lVert Q _ { t - 1 } ^ { \top } Q _ { t } ( : , r _ { c } + 1 : r _ { t } ) \rVert _ { F } ^ { 2 } \right) . } \end{array}\tag{10}
$$

Unlike $\operatorname { E q . } ( 7 ) ;$ these overlaps are not Grassmann invariants: they characterize how newly activated directions interact with the established shared subspace. When $r _ { t - 1 } = r _ { t } = r _ { c }$ , we define $e _ { \mathrm { s l a c k } } ^ { ( t ) } =$ $1 - s _ { \mathrm { c o r e } } ^ { ( t ) }$ to keep the regularizer active in the absence of residual dimensions.

To avoid uniformly penalizing all slack activations, we maintain an exponential moving reference:

$$
\tau _ { t } = \beta \tau _ { t - 1 } + ( 1 - \beta ) e _ { \mathrm { s l a c k } } ^ { ( t ) } , \qquad \beta \in ( 0 , 1 ) ,\tag{11}
$$

and apply an adaptive softplus penalty:

$$
\mathcal { L } _ { \mathrm { s l a c k } } ^ { ( t ) } = \log \bigl ( 1 + \exp ( e _ { \mathrm { s l a c k } } ^ { ( t ) } - \tau _ { t } ) \bigr ) .\tag{12}
$$

Finally, ACSA interpolates between stability (core) and plasticity (slack) according to the Grassmannian alignment level:

$$
\mathcal { L } _ { \mathrm { A C S A } } ^ { ( t ) } = s _ { \mathrm { c o r e } } ^ { ( t ) } \mathcal { L } _ { \mathrm { c o r e } } ^ { ( t ) } + \big ( 1 - s _ { \mathrm { c o r e } } ^ { ( t ) } \big ) \mathcal { L } _ { \mathrm { s l a c k } } ^ { ( t ) } .\tag{13}
$$

When two tasks share highly similar subspaces $( s _ { \mathrm { c o r e } } ^ { ( t ) } \approx 1 )$ , ACSA emphasizes Grassmann core preservation. When their alignment weakens, the emphasis shifts toward controlled plasticity in the slack directions, allowing the shared LoRA branch to expand without catastrophic interference.

## 4.3 Median-Calibrated Block Overlap

While the shallow shared branch of Geo-LoRA maintains global geometric coherence through SPP and ACSA, the deep task-specific branch is responsible for localized semantic specialization. In classincremental learning, these deep LoRA modules are repeatedly deployed on the same set of Transformer blocks across tasks. Without appropriate constraints, later tasks tend to reuse discriminative directions learned by earlier ones, resulting in subspace crowding and inter-task interference, which degrade both plasticity and retention.

To address this issue, we introduce Median-Calibrated Block Overlap (MCBO), a statistically adaptive geometric regularization that modulates representational reuse in the deep task-specific branch. In contrast to the Grassmann-based constraints in SPP and ACSA, MCBO does not operate on a fixed-rank manifold: the historical subspace expands as more tasks are observed, whereas each task-specific LoRA update remains low-rank. This rank mismatch makes standard Grassmannian distances ill-suited. Instead, MCBO measures how much the current update aligns with the accumulated historical subspace under the LoRA-induced metric, and uses a task-wise median to calibrate what constitutes “excessive” reuse.

Let ${ \mathcal { B } } _ { \mathrm { d e e p } }$ denote the set of deep Transformer blocks equipped with task-specific LoRA adapters. For any block $b \in \mathcal { B } _ { \mathrm { d e e p } } ,$ , we form an orthonormal basis of the historical subspace spanned by all previous task-specific updates:

$$
V _ { b } \in \mathbb { R } ^ { d \times r _ { b } } , \quad V _ { b } ^ { \top } V _ { b } = I .\tag{14}
$$

The current task introduces its own LoRA update at block �, denoted by $A _ { b } ^ { ( t ) } \in \mathbb { R } ^ { d \times r }$ . A natural (but scale-dependent) projection energy measuring how much $A _ { b } ^ { ( t ) }$ lies in the historical subspace is

$$
R _ { b } ^ { ( t ) } = { A _ { b } ^ { ( t ) \top } } V _ { b } V _ { b } ^ { \top } A _ { b } ^ { ( t ) } .\tag{15}
$$

Since the overall scale of $A _ { b } ^ { ( t ) }$ may vary across tasks and blocks, we normalize this projection in the current LoRA metric by defining

$$
G _ { b } ^ { ( t ) } = A _ { b } ^ { ( t ) \top } A _ { b } ^ { ( t ) } + \varepsilon I , \quad \varepsilon > 0 ,\tag{16}
$$

and the symmetrically normalized overlap matrix

$$
\widetilde { R } _ { b } ^ { ( t ) } = G _ { b } ^ { ( t ) - \frac { 1 } { 2 } } R _ { b } ^ { ( t ) } G _ { b } ^ { ( t ) - \frac { 1 } { 2 } } = G _ { b } ^ { ( t ) - \frac { 1 } { 2 } } A _ { b } ^ { ( t ) \top } V _ { b } V _ { b } ^ { \top } A _ { b } ^ { ( t ) } G _ { b } ^ { ( t ) - \frac { 1 } { 2 } } .\tag{17}
$$

This normalization removes scale dependence and isolates the geometric overlap between the current and historical low-rank subspaces under the metric induced by $A _ { b } ^ { ( t ) }$ . In practice, task-specific LoRA updates are low-rank, so $A _ { b } ^ { ( t ) \top } A _ { b } ^ { ( t ) }$ can be singular or illconditioned; the regularization term �� ensures positive definiteness and prevents numerical failures.

The block-level overlap score is then defined as

$$
S _ { b } ^ { ( t ) } = \frac { 1 } { r } \operatorname { t r } \big ( \widetilde { R } _ { b } ^ { ( t ) } \big ) = \frac { 1 } { r } \operatorname { t r } \Big ( A _ { b } ^ { ( t ) \top } V _ { b } V _ { b } ^ { \top } A _ { b } ^ { ( t ) } \big ( A _ { b } ^ { ( t ) \top } A _ { b } ^ { ( t ) } + \varepsilon I \big ) ^ { - 1 } \Big ) .\tag{18}
$$

Intuitively, $S _ { b } ^ { ( t ) }$ measures how strongly the current LoRA update at block � reuses representational directions learned by previous tasks: higher values indicate stronger reuse, while lower values correspond to expansion into novel task-specific directions.

Because diferent deep blocks naturally exhibit distinct reuse levels, MCBO avoids a global threshold and instead computes a task-wise median-calibrated reference level:

$$
\tau ^ { ( t ) } = \mathrm { m e d i a n } \{ S _ { b } ^ { ( t ) } \mid b \in \mathcal { B } _ { \mathrm { d e e p } } \} .\tag{19}
$$

Only blocks whose overlap exceeds this typical level are softly penalized via a softplus regularization:

$$
\mathcal { L } _ { \mathrm { M C B O } } ^ { ( t ) } = \frac { 1 } { | \mathcal { B } _ { \mathrm { d e e p } } | } \sum _ { b \in \mathcal { B } _ { \mathrm { d e e p } } } \log \bigl ( 1 + \exp ( S _ { b } ^ { ( t ) } - \tau ^ { ( t ) } ) \bigr ) .\tag{20}
$$

## 4.4 Training Objective

Geo-LoRA integrates task learning with geometric constraints into a unified, parameter-eficient optimization framework. The overall objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { \mathrm { s p p } } \mathcal { L } _ { \mathrm { S P P } } + \lambda _ { \mathrm { a c s a } } \mathcal { L } _ { \mathrm { A C S A } } + \lambda _ { \mathrm { m c b o } } \mathcal { L } _ { \mathrm { M C B O } } ,\tag{21}
$$

where $\mathcal { L } _ { \mathrm { c e } }$ trains the current task while the three geometric terms regulate the evolution of shared and task-specific low-rank subspaces. These constraints operate only during training and introduce no inference-time overhead

During inference, Geo-LoRA follows the standard prototypebased protocol. The shared branch produces the backbone feature, and each task-specific branch generates its corresponding representation. Classification is performed by cosine similarity to class prototypes computed during training. This parameter-eficient scheme maintains representational consistency across tasks, enabling smooth subspace evolution and reducing catastrophic forgetting under a rehearsal-free setting.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We evaluate Geo-LoRA on six rehearsal-free CIL benchmarks covering natural images, distribution shifts, fine-grained recognition, and heterogeneous visual domains. CIFAR-100 contains 100 natural categories. ImageNet-R and ImageNet-A each include 200 ImageNet-derived classes with artistic or adversarial distribution shifts. CUB200 comprises 200 fine-grained bird species. Following prior work, we construct a 50-class VTAB subset spanning diverse visual domains. OmniBenchmark provides 300 heterogeneous classes from multiple visual sources. All datasets are split into � equal-sized tasks to create CIL sequences of varying lengths.

Compared Methods. We compare Geo-LoRA with representative rehearsal-free PEFT-based CIL methods across prompt tuning, lightweight adapters, and LoRA variants, including L2P, Dual-Prompt, CODA-Prompt, InfLoRA, SD-LoRA, ACMap, SEMA, BiLoRA, and CL-LoRA. This suite covers the major parameter-eficient paradigms and forms a comprehensive evaluation.

Evaluation Metrics. Following standard protocol, we report the average accuracy where $A _ { t }$ is accuracy over all seen classes after task �, and the final accuracy $A _ { T }$ after completing the last task. These metrics assess both overall performance and long-term retention.

Implementation Details. We implement all models in PyTorch[26] and PILOT[30], using the same ViT-B/16[7] backbone with N = 12 Transformer blocks. Following prior CIL work, we adopt two representative pre-trained models—ViT-B/16-IN1K and ViT-B/16- IN21K—both initialized from ImageNet-21K, with the former further fine-tuned on ImageNet-1K. For LoRA, we use rank $r = 1 0$ with $A \in \mathbb { R } ^ { 7 6 8 \times 1 0 }$ and $B \in \mathbb { R } ^ { 1 0 \times 7 6 8 }$ . Task-shared LoRA adapters are inserted into the first $1 = 6$ blocks, and task-specific adapters into the remaining 6 blocks. All baseline implementations are based on publicly released oficial code or reported numbers from published papers. Except for SD-LoRA, which requires substantially more GPU memory and is therefore run on an NVIDIA H800, all other methods—including our Geo-LoRA—are trained and evaluated on NVIDIA RTX 3090.

## 5.2 Experimental Results

As shown in Tables 1 and 2, Geo-LoRA consistently surpasses all baselines across varying task numbers and both 1k- and 21kpretrained ViT backbones. The improvements are especially notable on challenging benchmarks such as ImageNet-R, ImageNet-A, VTAB, and CUB200. On ImageNet-R/A with 10-task and 20-task sequences—where artistic transformations and adversarial perturbations induce strong distribution shifts—Geo-LoRA achieves the best accuracy on both PTMs, for instance, our method achieves accuracy improvements of 2.19% and 1.25% on ImageNet-A (10 tasks) and ImageNet-R (20 tasks), respectively, when using the 1K pretrained model. Figure 3 further compares Geo-LoRA with the latest CL-LoRA on OmniBenchmark and CIFAR-100 (1k PTM). Geo-LoRA achieves higher final accuracy and maintains an advantage across most tasks, demonstrating more favorable stability–plasticity dynamics. To assess generalization under diverse regimes, we additionally run 5-task short-horizon and 40-task long-horizon experiments on ImageNet-A/R using the 1k backbone. As shown in Tables 3 and 4, Geo-LoRA remains the top performer even in these extreme settings, confirming strong scalability.

Meanwhile, we further evaluate the computational overhead on ImageNet-A with the 21k-pretrained backbone under both 10- task and 20-task settings. As reported in Table 5, we measure the total training time as well as the inference time during the final evaluation. Compared with the baselines, Geo-LoRA incurs only a slight increase in overall training time, while its inference time remains nearly unchanged. Given the clear performance gains, we believe this modest training overhead is well justified.

Table 1: Results (%) on ImageNet-R, ImageNet-A, VTAB and CUB200 with ViT-B/16-IN1K
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-R  $\left( T \mathrm { = } 2 0 \right)$ </td><td colspan="2">ImageNet-R  $\left( T \mathrm { = } 1 0 \right)$ </td><td colspan="2">ImageNet-A  $\left( T \mathrm { = } 2 0 \right)$ </td><td colspan="2">ImageNet-A  $\scriptstyle ( T = 1 0 )$ </td><td colspan="2">VTAB (T=5)</td><td colspan="2">CUB-200 (T=10)</td></tr><tr><td> $A _ { T }$ </td><td>Ã</td><td> $A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td>Ã</td><td> $A _ { T }$ </td><td> $\bar { A }$ </td><td> $A _ { T }$ </td><td>Á</td><td> $A _ { T }$ </td><td> $\bar { A }$ </td></tr><tr><td>L2P [36]</td><td>68.97</td><td>74.16</td><td>71.26</td><td>76.13</td><td>38.48</td><td>47.16</td><td>42.94</td><td>51.40</td><td>80.83</td><td>81.19</td><td>65.18</td><td>76.12</td></tr><tr><td>DualPrompt [35]</td><td>65.23</td><td>71.30</td><td>68.22</td><td>73.81</td><td>50.23</td><td>59.54</td><td>45.49</td><td>54.68</td><td>79.79</td><td>82.89</td><td>68.00</td><td>79.40</td></tr><tr><td>CODA-Prompt [29]</td><td>69.38</td><td>73.95</td><td>74.05</td><td>78.14</td><td>35.02</td><td>47.29</td><td>45.36</td><td>57.03</td><td>82.33</td><td>83.88</td><td>71.92</td><td>78.76</td></tr><tr><td>InfLoRA [20]</td><td>68.89</td><td>76.68</td><td>74.75</td><td>80.67</td><td>46.21</td><td>59.71</td><td>49.20</td><td>60.92</td><td>87.63</td><td>88.90</td><td>70.82</td><td>81.39</td></tr><tr><td>SD-LoRA [37]</td><td>75.26</td><td>80.22</td><td>77.34</td><td>82.04</td><td>52.71</td><td>62.21</td><td>55.96</td><td>64.95</td><td>75.71</td><td>87.03</td><td>77.48</td><td>85.59</td></tr><tr><td>ACMap [9]</td><td>73.75</td><td>79.49</td><td>75.06</td><td>80.52</td><td>53.29</td><td>63.79</td><td>53.21</td><td>64.92</td><td>89.35</td><td>92.76</td><td>77.62</td><td>85.03</td></tr><tr><td>SEMA [32]</td><td>74.53</td><td>81.75</td><td>78.00</td><td>83.56</td><td>48.52</td><td>60.02</td><td>53.29</td><td>63.71</td><td>89.64</td><td>91.26</td><td>77.31</td><td>83.07</td></tr><tr><td>BiLoRA [43]</td><td>72.07</td><td>77.04</td><td>77.45</td><td>80.36</td><td>40.75</td><td>57.84</td><td>53.26</td><td>65.65</td><td>81.24</td><td>79.21</td><td>75.48</td><td>83.19</td></tr><tr><td>CL-LoRA [12]</td><td>78.37</td><td>84.39</td><td>80.15</td><td>85.72</td><td>53.26</td><td>64.73</td><td>57.43</td><td>68.70</td><td>93.06</td><td>94.05</td><td>78.12</td><td>86.32</td></tr><tr><td>Geo-LoRA (Ours)</td><td>79.62</td><td>84.93</td><td>81.21</td><td>86.46</td><td>54.58</td><td>65.22</td><td>59.62</td><td>69.90</td><td>93.96</td><td>94.71</td><td>78.63</td><td>88.35</td></tr></table>

Table 2: Results (%) on ImageNet-R, ImageNet-A, VTAB and CUB200 with ViT-B/16-IN21K

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-R  $\left( T \mathrm { = } 2 0 \right)$ </td><td colspan="2">ImageNet-R  $\left( T \mathrm { = } 1 0 \right)$ </td><td colspan="2">ImageNet-A  $\left( T \mathrm { = } 2 0 \right)$ </td><td colspan="2">ImageNet-A  $\scriptstyle ( T = 1 0 )$ </td><td colspan="2">VTAB (T=5)</td><td colspan="2">CUB-200 (T=10)</td></tr><tr><td> $A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td>Á</td><td> $A _ { T }$ </td><td>Ã</td></tr><tr><td>L2P [36]</td><td>62.15</td><td>68.35</td><td>64.94</td><td>70.33</td><td>39.30</td><td>46.67</td><td>37.62</td><td>39.81</td><td>76.41</td><td>78.96</td><td>63.24</td><td>75.01</td></tr><tr><td>DualPrompt [35]</td><td>66.89</td><td>73.07</td><td>62.24</td><td>74.63</td><td>48.78</td><td>58.45</td><td>47.45</td><td>56.43</td><td>80.94</td><td>82.51</td><td>69.22</td><td>79.80</td></tr><tr><td>CODA-Prompt [29]</td><td>67.53</td><td>73.64</td><td>72.15</td><td>77.51</td><td>37.06</td><td>50.73</td><td>51.61</td><td>60.70</td><td>89.49</td><td>92.27</td><td>70.11</td><td>80.74</td></tr><tr><td>InfLoRA [20]</td><td>71.01</td><td>77.28</td><td>75.65</td><td>80.82</td><td>41.61</td><td>56.84</td><td>47.04</td><td>56.91</td><td>87.16</td><td>88.83</td><td>70.47</td><td>81.02</td></tr><tr><td>SD-LoRA [37]</td><td>74.31</td><td>80.00</td><td>77.01</td><td>81.92</td><td>45.89</td><td>56.60</td><td>50.10</td><td>59.17</td><td>72.77</td><td>85.24</td><td>63.74</td><td>82.61</td></tr><tr><td>ACMap [9]</td><td>72.95</td><td>78.82</td><td>73.50</td><td>79.50</td><td>51.58</td><td>62.56</td><td>56.19</td><td>65.19</td><td>87.56</td><td>91.21</td><td>76.42</td><td>83.09</td></tr><tr><td>SEMA [32]</td><td>69.60</td><td>77.84</td><td>74.82</td><td>81.39</td><td>46.94</td><td>58.36</td><td>45.95</td><td>57.89</td><td>90.86</td><td>91.99</td><td>79.62</td><td>84.95</td></tr><tr><td>BiLoRA [43]</td><td>72.41</td><td>79.28</td><td>77.95</td><td>81.52</td><td>40.75</td><td>55.18</td><td>48.91</td><td>61.47</td><td>81.20</td><td>80.47</td><td>78.34</td><td>85.04</td></tr><tr><td>CL-LoRA [12]</td><td>77.42</td><td>83.57</td><td>80.20</td><td>85.41</td><td>51.35</td><td>63.75</td><td>58.09</td><td>69.30</td><td>93.31</td><td>93.51</td><td>79.32</td><td>88.12</td></tr><tr><td>Geo-LoRA (Ours)</td><td>78.30</td><td>84.12</td><td>80.95</td><td>86.15</td><td>52.73</td><td>64.85</td><td>58.79</td><td>69.53</td><td>93.72</td><td>94.85</td><td>80.41</td><td>89.05</td></tr></table>

![](images/2cb4fb76a0d6c5cfc58466f7aacf24e07683a75703f0f58b0def2fb743c5a271.jpg)

![](images/f86f3fe973fec54b754df82d59cade191ecf1f612f6526fbf1a3b2e9479c634d.jpg)  
Figure 3: Results (%) on OmniBenchmark and Cifar100 across training stages with ViT-B/16-IN1K (�=20).

## 5.3 Experimental Analysis

The superior performance of Geo-LoRA on both short and long CIL sequences can be attributed to its explicit regulation of the evolution of feature subspaces—keeping representations geometrically aligned across tasks—whereas prior PEFT methods mainly act at token level and may not directly constrain the cumulative drift that can destabilize decision boundaries.

![](images/c709eea7aa83fd2f7b6db8c1138a3e4ea65737ce1f88339bc405945be8e51022.jpg)  
(a)

![](images/b04d1ebcce1abec2438f607cc5533f41d1b3ce312e21907591fb0aa9b47f17b9.jpg)

![](images/32e18ead4275b4cb44eb0d79fd68008d8f96ff654b6884938c056c6903538615.jpg)  
(c)

(b)  
![](images/6f25a05fe9a1193639bf554261e77dca6149961e6cf9c1e38e99b42530563614.jpg)  
(d)  
Figure 4: Empirical validation of the geometric mechanisms in Geo-LoRA. (a) SPP suppresses shared subspace drift over training epochs. (b) MCBO reduces median-calibrated overlap in deep task-specific LoRA across tasks, indicating controlled reuse of historical directions. (c) ACSA improves cross-task core alignment of shared subspaces. (d) ACSA maintains higher core alignment during training. These results demonstrate that the proposed constraints explicitly regulate subspace drift, overlap, and alignment, jointly enabling controlled geometric evolution of LoRA subspaces.

Table 3: Results (%) on ImageNet-R and ImageNet-A with ViT-B/16-IN1K ( �=40).  
Table 4: Results (%) on ImageNet-R and ImageNet-A with ViT-B/16-IN1K (�=5).
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-A</td></tr><tr><td> $A _ { T }$ </td><td>Á</td><td> $A _ { T }$ </td><td> $\bar { A }$ </td></tr><tr><td>L2P</td><td>60.62</td><td>65.82</td><td>27.42</td><td>38.09</td></tr><tr><td>DualPrompt</td><td>61.73</td><td>67.41</td><td>42.75</td><td>52.26</td></tr><tr><td>CODA-Prompt</td><td>63.93</td><td>70.39</td><td>27.93</td><td>38.39</td></tr><tr><td>InfLoRA</td><td>64.51</td><td>73.22</td><td>40.02</td><td>52.97</td></tr><tr><td>SD-LoRA</td><td>70.07</td><td>76.21</td><td>42.49</td><td>53.07</td></tr><tr><td>ACMap</td><td>71.10</td><td>77.78</td><td>43.55</td><td>53.75</td></tr><tr><td>SEMA</td><td>65.38</td><td>74.89</td><td>32.98</td><td>50.44</td></tr><tr><td>BiLoRA</td><td>64.05</td><td>68.99</td><td>31.80</td><td>47.26</td></tr><tr><td>CL-LoRA</td><td>73.37</td><td>81.54</td><td>43.45</td><td>56.97</td></tr><tr><td>Geo-LoRA (Ours)</td><td>73.90</td><td>82.19</td><td>44.04</td><td>57.53</td></tr></table>

Relative to single-branch LoRA approaches (e.g., BiLoRA and SD-LoRA), Geo-LoRA exhibits more reliable behavior by mitigating deep-layer subspace crowding. Single-branch LoRA often repeatedly reuses a narrow set of discriminative directions, limiting the capacity to encode genuinely novel task-specific variations. In contrast, SPP and ACSA stabilize the shared branch to accumulate taskagnostic structure coherently, while MCBO encourages the deep branch to explore new, non-conflicting directions. Consequently, Geo-LoRA better preserves earlier feature semantics while learning richer and less entangled representations for new tasks, improving stability and plasticity under comparable parameter budgets.

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-A</td></tr><tr><td> $A _ { T }$ </td><td> $\bar { A }$ </td><td> $A _ { T }$ </td><td> $\bar { A }$ </td></tr><tr><td>L2P</td><td>73.04</td><td>76.94</td><td>40.21</td><td>49.02</td></tr><tr><td>DualPrompt</td><td>69.99</td><td>72.24</td><td>54.72</td><td>62.93</td></tr><tr><td>CODA-Prompt</td><td>76.63</td><td>80.30</td><td>42.38</td><td>50.21</td></tr><tr><td>InfLoRA</td><td>76.95</td><td>81.81</td><td>49.37</td><td>63.27</td></tr><tr><td>SD-LoRA</td><td>77.30</td><td>82.30</td><td>55.83</td><td>65.15</td></tr><tr><td>ACMap</td><td>76.09</td><td>80.02</td><td>63.38</td><td>72.14</td></tr><tr><td>SEMA</td><td>75.92</td><td>82.27</td><td>56.47</td><td>64.62</td></tr><tr><td>BiLoRA</td><td>75.13</td><td>79.33</td><td>43.27</td><td>59.21</td></tr><tr><td>CL-LoRA</td><td>82.05</td><td>86.15</td><td>63.39</td><td>72.32</td></tr><tr><td>Geo-LoRA (Ours)</td><td>82.48</td><td>86.85</td><td>64.25</td><td>72.83</td></tr></table>

Table 5: Total training time (T) and inference time (I) comparison on ImageNet-A.
<table><tr><td>Method</td><td>T(20task)</td><td>I(20task)</td><td>T(10task)</td><td>I(10task)</td></tr><tr><td>Baseline</td><td>44.37 min</td><td>77.84 s</td><td>30.83 min</td><td>39.61 s</td></tr><tr><td>Geo-LoRA</td><td>46.85 min</td><td>77.60 s</td><td>32.27 min</td><td>39.54 s</td></tr></table>

Table 6: Ablation of geometric objectives on the ImageNet-A dataset with a 1K-pretrained model.
<table><tr><td>Method</td><td>20task</td><td>10task</td><td>5task</td></tr><tr><td>Baseline</td><td>53.26</td><td>57.43</td><td>63.39</td></tr><tr><td> $+ \ S \mathrm { P P + A C S A }$ </td><td>53.84</td><td>58.42</td><td>63.71</td></tr><tr><td> $+ \ S \mathrm { P P + M C B O }$ </td><td>54.10</td><td>59.04</td><td>63.98</td></tr><tr><td> $+ \mathrm { \ A C S A } + \mathrm { M C B O }$ </td><td>53.92</td><td>58.97</td><td>63.84</td></tr><tr><td>Full Geo-LoRA</td><td>54.58</td><td>59.62</td><td>64.25</td></tr></table>

Compared with prior dual-LoRA designs such as CL-LoRA, Geo-LoRA further benefits from a unified projection-based geometric principle that explicitly coordinates the evolution of shared and taskspecific subspaces. This coordination enables the shared branch to absorb cross-task common structure in a geometry-consistent manner, while the task-specific branch focuses on localized discrimination without collapsing into previously occupied directions, reducing interference within fine-grained subspaces.

Overall, these results suggest that geometry-aware subspace regulation is a principled and practical mechanism for parametereficient continual learning, improving long-term retention and task-specific adaptation across diverse and distribution-shifted CIL benchmarks.

## 5.4 Representation Dynamics Analysis

Beyond performance metrics, we further analyze the geometric dynamics targeted by Geo-LoRA from three aspects: (i) subspace drift in the shallow shared branch, (ii) cross-task core alignment of shared subspaces, and (iii) excessive reuse in deep task-specific blocks. As shown in Fig. 4, SPP markedly suppresses shared-subspace drift during training, ACSA consistently improves and maintains higher core alignment across tasks and epochs, and MCBO reduces median-calibrated overlap in deep task-specific LoRA blocks, indicating more controlled reuse of historical directions. These results provide direct empirical evidence that Geo-LoRA explicitly regu lates drift, alignment, and overlap during continual adaptation.

## 5.5 Ablation Study

We conduct ablation studies to evaluate the contribution of each component and the method design in Geo-LoRA.

Geometric objectives. As shown in Table 6, all three components provide substantial and complementary benefits: SPP improves cross-task consistency, ACSA refines the stability–plasticity balance, and MCBO strengthens deep-layer specialization. Removing any single term leads to a clear drop in performance, highlighting that Geo-LoRA relies on their joint interaction rather than any individual objective.

SPP normalization. Table 7 compares the Grassmann projector $A _ { t } ( A _ { t } ^ { \top } A _ { t } ) ^ { - 1 } A _ { t } ^ { \top }$ with the covariance-based normalization $A _ { t - 1 } ^ { \top } A _ { t - 1 } + \varepsilon I$ . The projector consistently yields higher accuracy, demonstrating superior geometric stability and robustness under low-rank updates.

Table 7: Comparison of SPP normalization schemes on ImageNet-R (20 tasks).
<table><tr><td>Normalization Choice</td><td> $A _ { T } \left( \% \right)$ </td><td>A (%)</td></tr><tr><td> $A _ { t - 1 } ( A _ { t - 1 } ^ { \top } A _ { t - 1 } + \varepsilon I ) ^ { - 1 } A _ { t - 1 } ^ { \top }$ </td><td>79.27</td><td>84.77</td></tr><tr><td> $A _ { t - 1 } ( A _ { t - 1 } ^ { \top } A _ { t - 1 } ) ^ { - 1 } A _ { t - 1 } ^ { \top }$ </td><td>79.62</td><td>84.93</td></tr></table>

## 6 Conclusion

We introduced Geo-LoRA, a unified geometric framework for continual low-rank adaptation that reframes LoRA updates as the evolution of task-dependent subspaces. By regulating the geometry of both shared and task-specific low-rank structures, Geo-LoRA provides a principled way to control how representational subspaces drift, expand, and interact across tasks. Through the complementary roles of SPP, ACSA, and MCBO, Geo-LoRA establishes a coherent hierarchy in which shallow layers preserve global consistency while deep layers manage localized plasticity. This geometry-aware design delivers strong accuracy and forgetting reductions over prior adapter-based CIL methods, without additional model capacity or replay. More broadly, our formulation suggests that continual adapter tuning can be viewed as managing subspace trajectories on evolving low-rank manifolds. We hope this perspective encourages future work on geometry-aware adaptation and scalable continual tuning for vision and multimodal foundation models.

## References

[1] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the efectiveness of language model fine-tuning. In Proceedings ofthe 59th annual meeting ofthe association for computational linguistics and the 11th international joint conference on natural language processing (volume 1: long papers). 7319–7328.

[2] Feidu Akmel, Fanman Meng, Mingyu Liu, Runtong Zhang, Asebe Teka, and Elias Lemuye. 2024. Few-shot class incremental learning via prompt transfer and knowledge distillation. Image and Vision Computing 151 (2024), 105251.

[3] Vladimir Araujo, Marie Francine Moens, and Tinne Tuytelaars. 2024. Learning to route for dynamic adapter composition in continual learning with language models. In Findings ofthe Association for Computational Linguistics: EMNLP 2024. 687–696.

[4] Prashant Shivaram Bhat, Shakib Yazdani, Elahe Arani, and Bahram Zonooz. 2025. Parameter Eficient Continual Learning with Dynamic Low-Rank Adaptation. arXiv preprint arXiv:2505.11998 (2025).

[5] Xinyang Chen, Sinan Wang, Jianmin Wang, and Mingsheng Long. 2021. Representation subspace distance for domain adaptation regression.. In ICML. 1749–1759.

[6] Shibhansh Dohare, J Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A Rupam Mahmood, and Richard S Sutton. 2024. Loss of plasticity in deep continual learning. Nature 632, 8026 (2024), 768–774.

[7] Alexey Dosovitskiy. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[8] Basura Fernando, Amaury Habrard, Marc Sebban, and Tinne Tuytelaars. 2013. Unsupervised visual domain adaptation using subspace alignment. In Proceedings ofthe IEEE international conference on computer vision. 2960–2967.

[9] Takuma Fukuda, Hiroshi Kera, and Kazuhiko Kawamoto. 2025. Adapter merging with centroid prototype mapping for scalable class-incremental learning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 4884– 4893.

[10] Xinyuan Gao, Songlin Dong, Yuhang He, Qiang Wang, and Yihong Gong. 2024. Beyond prompt learning: Continual adapter for eficient rehearsal-free continual learning. In European Conference on Computer Vision. Springer, 89–106.

[11] Jiangpeng He. 2024. Gradient reweighting: Towards imbalanced class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16668–16677.

[12] Jiangpeng He, Zhihao Duan, and Fengqing Zhu. 2025. CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 30534–30544.

[13] Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. 2021. Towards a unified view of parameter-eficient transfer learning. arXiv preprint arXiv:2110.04366 (2021).

[14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR 1, 2 (2022), 3.

[15] Zitong Huang, Ze Chen, Zhixing Chen, Erjin Zhou, Xinxing Xu, Rick Siow Mong Goh, Yong Liu, Wangmeng Zuo, and Chunmei Feng. 2024. Learning prompt with distribution-based feature replay for few-shot class-incremental learning. arXiv preprint arXiv:2401.01598 (2024).

[16] Zitong Huang, Ze Chen, Yuanze Li, Bowen Dong, Erjin Zhou, Yong Liu, Rick Siow Mong Goh, Chun-Mei Feng, and Wangmeng Zuo. 2024. Class balance matters to active class-incremental learning. In Proceedings of the 32nd ACM International Conference on Multimedia. 9445–9454.

[17] Yongwei Jiang, Yixiong Zou, Yuhua Li, and Ruixuan Li. 2025. Revisiting Poolbased Prompt Learning for Few-shot Class-incremental Learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 1303–1313.

[18] Qiwei Li, Yuxin Peng, and Jiahuan Zhou. 2024. Fcs: Feature calibration and separation for non-exemplar class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 28495–28504.

[19] Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40, 12 (2017), 2935–2947.

[20] Yan-Shuo Liang and Wu-Jun Li. 2024. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition. 23638–23647.

[21] Chenxi Liu, Zhenyi Wang, Tianyi Xiong, Ruibo Chen, Yihan Wu, Junfeng Guo, and Heng Huang. 2024. Few-shot class incremental learning with attention-aware self-adaptive prompt. In European Conference on Computer Vision. Springer, 1–18.

[22] Chengyan Liu, Linglan Zhao, Fan Lyu, Kaile Du, Fuyuan Hu, and Tao Zhou. 2024. CALA: A Class-Aware Logit Adapter for Few-Shot Class-Incremental Learning. arXiv preprint arXiv:2412.12654 (2024).

[23] Xialei Liu, Xusheng Cao, Haori Lu, Jia-wen Xiao, Andrew D Bagdanov, and Ming Ming Cheng. 2023. Class incremental learning with pre-trained vision-language models. arXiv preprint arXiv:2310.20348 (2023).

[24] Michael McCloskey and Neal J Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology oflearning and motivation. Vol. 24. Elsevier, 109–165.

[25] Aristeidis Panos, Yuriko Kobe, Daniel Olmeda Reino, RahafAljundi, and Richard E Turner. 2023. First session adaptation: A strong replay-free baseline for class incremental learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 18820–18830.

[26] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

[27] Zhi-Hong Qi, Da-Wei Zhou, Yiran Yao, Han-Jia Ye, and De-Chuan Zhan. 2025. Adaptive adapter routing for long-tailed class-incremental learning. Machine Learning 114, 3 (2025), 1–20.

[28] Fuli Qiao and Mehrdad Mahdavi. 2024. Learn more, but bother less: parameter eficient continual learning. Advances in Neural Information Processing Systems 37 (2024), 97476–97498.

[29] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. 2023. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11909–11919.

[30] Hai-Long Sun, Da-Wei Zhou, De-Chuan Zhan, and Han-Jia Ye. 2025. Pilot: A pre-trained model-based continual learning toolbox.

[31] Hai-Long Sun, Da-Wei Zhou, Hanbin Zhao, Le Gan, De-Chuan Zhan, and Han-Jia Ye. 2025. Mos: Model surgery for pre-trained model-based class-incremental learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 20699–20707.

[32] Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. 2025. Self-expansion of pre-trained models with mixture ofadapters for continual learning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 10087–10098.

[33] Maorong Wang, Nicolas Michel, Ling Xiao, and Toshihiko Yamasaki. 2024. Improving plasticity in online continual learning via collaborative learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 23460–23469.

[34] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuan-Jing Huang. 2023. Orthogonal subspace learning for language model continual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023. 10658–10671.

[35] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. 2022. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European conference on computer vision. Springer, 631–648

[36] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 139–149.

[37] Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, and Ying Wei. 2025. SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning. In The Thirteenth International Conference on Learning Representations. https: //openreview.net/forum?id=5U1rlpX68A

[38] Kunlun Xu, Yibo Feng, Jiangmeng Li, Yongsheng Qi, and Jiahuan Zhou. 2025. Class-aware Client Knowledge Interaction for Federated Continual Learning. arXiv preprint arXiv:2509.19674 (2025).

[39] Xiaobing Yu, Jin Yang, Xiao Wu, Peijie Qiu, and Xiaofeng Liu. 2025. FM-LoRA: Factorized Low-Rank Meta-Prompting for Continual Learning. In Proceedings of the Computer Vision and Pattern Recognition Conference. 6409–6418.

[40] Lei Zhang, Jingru Fu, Shanshan Wang, David Zhang, Zhaoyang Dong, and CL Philip Chen. 2019. Guide subspace learning for unsupervised domain adap tation. IEEE transactions on neural networks and learning systems 31, 9 (2019), 3374–3388.

[41] Wentao Zhang, Yujun Huang, Tong Zhang, Qingsong Zou, Wei-Shi Zheng, and Ruixuan Wang. 2023. Adapter learning in pretrained feature extractor for contin ual learning of diseases. In International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 68–78.

[42] Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. 2024. Class-incremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[43] Hao Zhu, Yifei Zhang, Junhao Dong, and Piotr Koniusz. 2025. BiLoRA: Almost-Orthogonal Parameter Spaces for Continual Learning. In Proceedings of the Computer Vision and Pattern Recognition Conference. 25613–25622.