# Hyperbolic Multimodal Continual Learning: A Closest-Admissible Solution

Jiahong Liu<sup>1</sup>, Ming Shen<sup>1</sup>, Xiaohao Liu<sup>2</sup>, Rex Ying<sup>3</sup>, Menglin Yang<sup>4</sup>, Tat-Seng Chua<sup>2</sup>, and Irwin King

<sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>National University of Singapore <sup>3</sup>Yale University <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)

Email: jiahong.liu21@gmail.com, king@cse.cuhk.edu.hk

Abstract. Existing continual-learning methods protect parameters, replayed examples, or Euclidean feature subspaces. When applied to hyperbolic multimodal models, they do not explicitly preserve the Lorentz geometry that jointly encodes within-modality similarity, cross-modal correspondence, and semantic hierarchy; sequential updates can therefore retain task scores while still distorting previously learned relations. We address this gap with Hyperbolic Multimodal Continual Learning (HMCL). We show that preserving the old multimodal geometry amounts to restricting all modalities to one shared hyperbolic isometry, which induces a family of admissible first-order parameter changes. We formulate a joint closest-admissible (CA) correction that retains the shared rotation best matching the candidate modal updates; its minimal-rotation (MR) special case fixes this rotation to zero. Both variants correct the displacement realized by AdamW, and task anchoring bounds within-task accumulation while preserving learning freedom. Across a unified 16-task classification–retrieval stream with three hyperbolic backbones, HMCL improves final performance and backward transfer over sequential fine-tuning and four continual-learning baselines; HMCL-CA gives the highest Overall score on every backbone. A modality-extended stream confirms the retrieval gains. Representation analyses find 81.2–95.5% less radial, angular, cross-modal, and paired-distance drift; ImageNet–WordNet results show better semantic ancestry and radial hierarchy. Code and data download links are in the code repository.

Keywords: continual learning; hyperbolic geometry; multimodal learning; vision–language representation learning

## 1 Introduction

Multimodal contrastive learning (MCL) has become a standard approach for constructing representations shared across data types [1–3]. Trained with large paired corpora and contrastive objectives, these models align modality-specific observations in a semantic space that supports efective zero-shot transfer [4–6]. Most existing systems place this shared space in a high-dimensional Euclidean domain and therefore treat the organization of semantic concepts as flat.

Semantic organization across modalities is rarely flat, however. A broad concept generally covers numerous specialized instances, creating asymmetric, partially ordered relations between diferent levels of abstraction. Uniform Euclidean neighborhoods do not explicitly distinguish such general-to-specific structure, making these relations dificult to represent without distortion.

Continual Learning of Multimodal Hyperbolic Representations  
![](images/a085267415c9da8610695338c2650a89583606d5bfa1b3880cb691e6d9a8090f.jpg)  
Figure 1. Continual adaptation of hyperbolic representations. Hyperbolic multimodal embeddings jointly encode task semantics and their hierarchy. Unconstrained sequential optimization may gradually distort this geometry and weaken structures learned from earlier tasks. Geometryaware updates can reduce this form of forgetting.

Hyperbolic Multimodal Learning. Negative-curvature geometry ofers a useful alternative for hierarchical semantics. MERU [7] and subsequent compositional-entailment learning [8] show that hyperbolic geometry can support efective multimodal representations. Geodesic proximity reflects semantic similarity, whereas radial position and entailment relations encode specificity. Generic concepts tend to lie nearer the origin, and more specific concepts tend to lie farther away. Hyperbolic embeddings can therefore represent alignment and abstraction in the same space.

Despite these advances, existing hyperbolic multimodal models have been developed mainly for static training distributions [7–10]. Real systems instead encounter new tasks, concepts, and domains over time, yet standard continual-learning mechanisms do not explicitly protect the Lorentz relations that organize multimodal similarity and hierarchy. How to adapt such models sequentially without deforming their learned geometry therefore remains open.

Continual Learning as Geometric Preservation. For a hyperbolic model, forgetting involves more than reduced accuracy on an earlier task. Parameter changes may deform within-modality neighborhoods, cross-modal correspondences, and the radial ordering that records semantic abstraction. Once these quantities drift, both similarity and hierarchy inherited from old tasks are altered. This observation motivates our central question: which parameter changes preserve the previously learned hyperbolic multimodal geometry while leaving suficient freedom to acquire a new task?

Our Approach. We view hyperbolic multimodal continual learning<sup>1</sup> as optimization over updates that satisfy geometric preservation constraints, which we call admissible updates. Under the assumptions in Section 5, old-task geometry is retained when all modalities change through the same hyperbolic isometry. A first-order form of this condition describes the corresponding parameter updates. We then pose a joint closest-admissible problem over the modal parameter changes and the generator of their shared rotation. Its unique solution is obtained from one Lyapunov equation and defines HMCL-CA; constraining the generator to zero yields the projection used by HMCL-MR. For the AdamW implementation evaluated in this work, HMCL corrects the parameter step produced after moment estimation and coordinate-wise scaling, and task anchoring limits its accumulation within a task.

Contributions. To the best of our knowledge, this work is the first to study continual hyperbolic multimodal representations explicitly through their geometry. Our main contributions are:

• We relate retention of old multimodal knowledge to a shared hyperbolic isometry and derive the associated set of first-order parameter updates.

• We formulate a joint closest-admissible problem whose shared-rotation solution is obtained from one Lyapunov equation. $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ permits the common rotation selected by this solution, while HMCL<sub>MR</sub> restricts the same admissible family to zero rotation. Task anchoring preserves admissibility and bounds within-task accumulation.

• Across three backbones, both HMCL variants improve final classification and retrieval scores, overall performance, and BWT over all external baselines, with CA attaining the best Overall score in every backbone block. Representation-space analyses further show that these task-level gains coincide with 81.2–95.5% less radial, angular, cross-modal, and paired-distance drift and with improvements in every reported ImageNet– WordNet hierarchy metric. The placement study identifies post-AdamW correction with task anchoring as the strongest stability–plasticity configuration.

Extension beyond the conference version. The ICML paper introduced the geometric preservation conditions and the zero-rotation MR update on a 15-task stream [11]. This article generalizes that foundation with the rotation-permitting CA update, optimizer-consistent analysis, and broader geometric evaluation:

(i) Closest-admissible CA solution. We express preservation as a joint constrained least-squares problem over every modal parameter change and one shared skew-symmetric generator. Its Lyapunov-equation solution gives the rotation-permitting $\mathrm { \ H M C L _ { C A } }$ update; fixing $\pmb { \Omega } = \pmb { 0 }$ recovers the conference-version $\mathrm { H M C L } _ { \mathrm { M R } }$ update and places both in one admissible family (Section 6.1).

(ii) Finite-step preservation and controlled accumulation. Following the backbones’ AdamW training paradigm, we correct the realized parameter displacement after moment estimation and preconditioning. We show that task-anchored CA and MR updates remain in the shared admissible family and that their accumulation is uniformly bounded, and we derive residual-drift and complexity results for practical low-rank protection (Section 6.2 and Appendix F).

(iii) Hierarchy-aware large-scale evaluation. We add ImageNet-1K to form a unified 16-task continual stream across MERU-L, MERU-B, and HyCoCLIP-B. Its WordNet structure directly verifies that the predictive-retention gains coincide with better preservation of semantic ancestry and abstraction order, rather than only higher aggregate scores (Section 7.4).

(iv) Controlled component and trajectory analysis. The component study shows that correcting AdamW’s realized displacement is stronger than correcting its raw gradient and that task anchoring provides a further stability gain; the complete post-AdamW update therefore gives the best stability–plasticity balance (Section 7.6). Representation-drift and hierarchy analyses further verify that the gain accompanies less radial and relational distortion (Sections 7.3–7.5).

## 2 Related Work

## 2.1 Hyperbolic Multimodal Learning

The volume of hyperbolic space grows exponentially with radius. This property makes hyperbolic space useful for hierarchical or scale-free data and has motivated a broad literature on non-Euclidean representation learning [12–21], including recent calls to extend non-Euclidean geometry to foundation models [22]. Hyperbolic contrastive learning has been developed for node embeddings and graph-level anomaly detection, together with explicit analyses of dimensional collapse [23–25]. Universal cone constructions further recover implicit hierarchies without prespecified parent–child relations [26]. In recommendation and retrieval, hyperbolic geometry captures long-tailed interactions and hierarchical neighborhoods [27–29]. In computer vision, it has also been studied for image representation [30], hierarchy-aware zero-shot recognition [31], and contrastive modeling of scene–object structure [32]; its benefits can depend on the task and radial geometry [33]. The same idea has recently been applied to multimodal learning, where images and language often have clear relations of abstraction and specificity.

MERU [7] introduced large-scale hyperbolic vision–language pretraining by mapping CLIP-like representations to a Lorentz hyperboloid. Later studies developed the setting in several directions. Pal et al. [8] modeled compositional entailment within each modality, relating complex scenes to visual primitives and compound expressions to simpler concepts. Ibrahimi et al. [34] analyzed the geometry and uncertainty encoded by hyperbolic vision–language embeddings, whereas Ramasinghe et al. [35] replaced proximity-based cross-modal alignment with an angular objective that accommodates the modality gap. Other work scales hyperbolic learning to multimodal large language models [9], filters underspecified image–text pairs using entailment-cone apertures [10], and applies hierarchical alignment to synthetic-caption detection [36] or open-vocabulary segmentation [37]. Low-rank adaptation has also become a broad foundation-model adaptation family, including multimodal and continual settings [38]; related work performs such adaptation directly on a hyperbolic manifold [39]. HySAC [40] organizes safety concepts through a hyperbolic hierarchy, while hyperbolic multimodal unlearning studies the selective removal of concepts from MERU [41]. These works focus on pretraining, task-specific adaptation, safety, or removal; they do not study preservation of shared Lorentz relations throughout sequential multi-task acquisition.

## 2.2 Continual Learning

Continual learning seeks to acquire tasks in sequence without erasing capabilities learned previously [3]. Existing methods typically rely on replay, regularization, or parameter isolation. Replay approaches retain representative examples and reuse them during later training, as in GEM [42] and DER [43]. Regularization methods such as EWC [44] penalize changes to parameters deemed important for earlier tasks. Parameter-isolation strategies instead reserve diferent model capacity for diferent tasks; examples include Progressive Neural Networks [45], PackNet [46], and Piggyback [47].

Continual vision–language learning introduces an additional requirement: retention of cross-modal alignment together with modality-specific features [48]. Unlike CLIP domain adaptation and generalization [49], it mus preserve a growing task history. Ni et al. [48] described Spatial Disorder, which combines within-modality rotation and between-modality deviation, as an important failure mode in continual CLIP adaptation. Later methods preserve zero-shot behavior through distillation [50], use mixture-of-experts adapters for task acquisition [51], or apply representation-level contrastive regularization, as in C-FLAT [52]. Personalized LLM research likewise identifies lifelong updating, evolving user memories, and forgetting-resistant adaptation as core challenges [53]. As a concrete parameter-eficient instance, PerFit models personalization through a shared low-rank representation shift and user-specific deviations, then intervenes directly in hidden states [54]. Related geometric work treats long-horizon memory as a nonuniform dynamical space [55] and tailors Lorentz representations to heterogeneous federated clients [56]. Hyperbolic continual learning has also been studied for hierarchical image recognition [57] and multimodal few-shot class-incremental learning [58], but these lines do not enforce preservation of one shared Lorentz isometry across modalities and sequential tasks. Our setting also considers the semantic hierarchy encoded by radial coordinates and entailment cones. This motivates update constraints that account for hyperbolic geometry as well as cross-modal alignment.

## 2.3 Projected Updates with Adaptive Optimizers

Projection-based continual-learning methods often express retention as a constraint on a gradient or a descent direction [42]. With scalar-step stochastic gradient descent, projecting the direction and projecting the resulting parameter displacement are equivalent up to the learning rate. AdamW breaks this equivalence through momentum, element-wise preconditioning, and decoupled decay [59]. Riemannian adaptive optimizers instead study parameters that themselves lie on a manifold and use manifold-wise gradients and updates [60]. Our setting is diferent. The Lorentz head is Euclidean-parameterized, while the continual-learning constraint requires earlier outputs to remain on one shared Lorentz-isometry orbit.

## 3 Preliminaries

## 3.1 Hyperbolic Geometry

Hyperbolic space is a non-Euclidean manifold of constant negative curvature. We work with its Lorentz, or hyperboloid, realization because it supports stable numerical optimization [14, 61]. Appendix B reviews the hierarchy motivation, Lorentz model, and Lorentz neural layers.

Lorentz model. Let the curvature be $- 1 / K$ for $K > 0$ . A �-dimensional hyperbolic space can be represented by the upper sheet of a hyperboloid in $( d + 1 )$ -dimensional Minkowski space. With metric tensor $\mathbf { G } = \mathrm { d i a g } ( - 1 , 1 , \dots , 1 ) \in$ $\mathbb { R } ^ { \bar { ( d + 1 ) } \times ( d + 1 ) }$ , the manifold is

$$
\mathbb { H } _ { K } ^ { d } = \{ \mathbf { x } \in \mathbb { R } ^ { d + 1 } \mid \mathbf { x } ^ { \top } \mathbf { G } \mathbf { x } = - K , x _ { 0 } > 0 \} .\tag{1}
$$

A point $\mathbf { x } = [ x _ { 0 } ; \mathbf { x } _ { [ 1 : d ] } ] ^ { \top } \in \mathbb { H } _ { K } ^ { d }$ contains one time-like component $x _ { 0 }$ and � space-like components $\mathbf { X } _ { \left[ 1 : d \right] }$ , consistent with the signature of the Lorentzian metric.

Lorentzian product and distance. For $\mathbf { x } , \mathbf { y } \in \mathbb { H } _ { K } ^ { d }$ , their Lorentzian inner product is

$$
\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } = \mathbf { x } ^ { \top } \mathbf { G } \mathbf { y } = - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } .\tag{2}
$$

The associated geodesic distance is

$$
d _ { \mathcal { L } } ^ { K } ( \mathbf { x } , \mathbf { y } ) = \sqrt { K } \operatorname { a r c o s h } ( - \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } / K ) = \sqrt { K } \operatorname { a r c o s h } ( - \mathbf { x } ^ { \top } \mathbf { G } \mathbf { y } / K ) .\tag{3}
$$

Lorentz transformation layer. Consider a Lorentz embedding $\mathbf { z } \in \mathbb { R } ^ { d + 1 }$ with positive time-like coordinate $z _ { 0 }$ and spatial block $\mathbf { z } _ { s } \in \mathbb { R } ^ { d }$ . Following [62], a Lorentz transformation layer maps it as

$$
f ( \mathbf { W } ; \mathbf { z } ) = \left[ \sqrt { K + \| \mathbf { W } \mathbf { z } \| _ { 2 } ^ { 2 } } \right] ,\tag{4}
$$

where $\mathbf { W } \in \mathbb { R } ^ { d \times ( d + 1 ) }$ . The reconstructed first coordinate ensures that the output remains on $\mathbb { H } _ { K } ^ { d }$ . We partition the parameters as $\mathbf { W } = [ \mathbf { w } _ { 0 } \mathbf { W } _ { s } ] , \quad \mathbf { w } _ { 0 } \in \mathbb { R } ^ { d \times 1 } , \mathbf { W } _ { s } \in \mathbb { R } ^ { d \times d }$ , so that $\mathbf { w } _ { 0 }$ and ${ \mathbf W } _ { s }$ act on the time-like and space-like inputs, respectively.

## 3.2 Hyperbolic Multimodal Learning

Let $\chi ^ { m , m ^ { \prime } } = \{ ( { \bf x } _ { i } ^ { m } , { \bf x } _ { i } ^ { m ^ { \prime } } ) \} _ { i = 1 } ^ { N } \subset X ^ { m } \times X ^ { m ^ { \prime } }$ be � semantically matched observations from modalities � and $m ^ { \prime }$ . A multimodal model embeds them as matrices $\mathbf { Z } ^ { m } , \mathbf { Z } ^ { m ^ { \prime } } \in \mathbb { R } ^ { N \times ( d + 1 ) }$ , whose �th rows $\mathbf { z } _ { i } ^ { m }$ and $\pmb { z } _ { i } ^ { m ^ { \prime } }$ lie on $\mathbb { H } _ { K } ^ { d }$

Contrastive similarity. Pairwise similarity is defined by negative hyperbolic distance:

$$
\begin{array} { r } { { \bf S } ^ { m  m ^ { \prime } } = - d _ { \mathcal { L } } ^ { K } ( { \bf Z } ^ { m } , { \bf Z } ^ { m ^ { \prime } } ) = - \sqrt { K } \operatorname { a r c o s h } \Bigl ( - { \bf Z } ^ { m } { \bf G } ( { \bf Z } ^ { m ^ { \prime } } ) ^ { \top } / K \Bigr ) . } \end{array}\tag{5}
$$

Here arcosh is evaluated element-wise and $S _ { i j } ^ { m  m ^ { \prime } } = - d _ { \mathcal { L } } ^ { K } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { j } ^ { m ^ { \prime } } )$ . The directional contrastive objective is

$$
\mathcal { L } _ { m  m ^ { \prime } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( S _ { i i } ^ { m  m ^ { \prime } } / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( S _ { i j } ^ { m  m ^ { \prime } } / \tau ) } ,\tag{6}
$$

where � is the temperature. Training uses the symmetric loss

$$
\mathcal { L } _ { \mathrm { c o n t r a s t } } = \frac { 1 } { 2 } \big ( \mathcal { L } _ { m  m ^ { \prime } } + \mathcal { L } _ { m ^ { \prime }  m } \big ) .\tag{7}
$$

Entailment cones. Hierarchy is represented by assigning each $\pmb { z } _ { i } ^ { m } \in \mathbb { H } _ { K } ^ { d }$ an entailment cone whose aperture is

$$
\mathrm { a p e r } ( \mathbf { z } _ { i } ^ { m } ) = \mathrm { s i n } ^ { - 1 } \left( \frac { 2 \kappa \sqrt { K } } { \| ( \mathbf { z } _ { i } ^ { m } ) _ { [ 1 : d ] } \| _ { 2 } } \right) ,\tag{8}
$$

with $\kappa = 0 . 1$ controlling behavior near the origin. Points with a smaller spatial norm represent broader concepts and receive wider cones; larger norms correspond to more specific concepts and narrower cones. The exterior angle from $\mathbf { z } _ { i } ^ { m }$ to $\pmb { z } _ { i } ^ { m ^ { \prime } }$ is

$$
\gamma _ { i } ^ { m , m ^ { \prime } } : = \langle \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } \rangle \mathcal { L } / K ,
$$

$$
\mathrm { e x t } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) = \mathrm { c o s } ^ { - 1 } \left( \frac { ( \mathbf { z } _ { i } ^ { m ^ { \prime } } ) _ { 0 } + ( \mathbf { z } _ { i } ^ { m } ) _ { 0 } \gamma _ { i } ^ { m , m ^ { \prime } } } { \| ( \mathbf { z } _ { i } ^ { m } ) _ { [ 1 : d ] } \| _ { 2 } \sqrt { ( \gamma _ { i } ^ { m , m ^ { \prime } } ) ^ { 2 } - 1 } } \right) .\tag{9}
$$

The entailment objective [15] penalizes a target that falls outside the source cone:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e n t a i l } } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) = \operatorname* { m a x } \Bigl ( 0 , \mathrm { e x t } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) - \mathrm { a p e r } ( \mathbf { z } _ { i } ^ { m } ) \Bigr ) . } \end{array}\tag{10}
$$

This loss imposes the desired cross-modal partial order, for example by locating a general textual description above its visual instance in the hierarchy.

## 4 Problem Formulation

## 4.1 Continual Hyperbolic Multimodal Learning

We study a sequence of multimodal tasks whose representations share one hyperbolic space. At every stage, the learner absorbs the current task while retaining the within-modality, cross-modal, and hierarchical relations established by earlier tasks. Our setup follows the multimodal contrastive continual-learning protocol of [63].

For each modality $m \in \{ 1 , \ldots , M \}$ , a frozen pretrained encoder $\mathcal { E } _ { m } : \boldsymbol { X } ^ { m }  \mathbb { H } _ { K } ^ { d }$ is followed at task � by a trainable Lorentz transformation $T _ { m } ^ { ( t ) } : \mathbb { H } _ { K } ^ { d }  \mathbb { H } _ { K } ^ { d }$ . Applied to data from the previous task, the encoder first produces

$$
\mathbf { Z } _ { t - 1 } ^ { m , * } = \mathcal { E } _ { m } ( X _ { t - 1 } ^ { m } ) \in \mathbb { R } ^ { N \times ( d + 1 ) } ,\tag{11}
$$

![](images/dd23b0f41ac7158be7cdc57d40c2040db5ae844c2a09516b7a03b47378112cea.jpg)  
Figure 2. Geometry of continual hyperbolic multimodal learning. Geodesic proximity encodes semantic similarity, while radial distance from the origin records specificity. Under our preservation conditions, all modalities undergo one common hyperbolic isometry. From the top-down view, first-order preservation of radial hierarchy limits a permitted update to the tangent direction.

where every row lies on $\mathbb { H } _ { K } ^ { d }$ and the superscript ∗ indicates a frozen pretrained representation. The stage-� transformation then maps these representations through the trainable Lorentz head:

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } = T _ { m } ^ { ( t ) } ( \mathbf { Z } _ { t - 1 } ^ { m , * } ) = f \Big ( \mathbf { W } ^ { m , t } ; \mathbf { Z } _ { t - 1 } ^ { m , * } \Big ) \in \mathbb { R } ^ { N \times ( d + 1 ) } . } \end{array}\tag{12}
$$

The same stage processes current-task observations as

$$
\mathbf { Z } _ { t } ^ { m , t } = T _ { m } ^ { ( t ) } \bigl ( \mathcal { E } _ { m } ( X _ { t } ^ { m } ) \bigr ) \in \mathbb { R } ^ { N \times ( d + 1 ) } .\tag{13}
$$

Our notation $\mathbf { Z } _ { q } ^ { m , s }$ therefore identifies the data task $q ,$ modality $m ,$ and processing stage �. For instance, $\mathbf { Z } _ { t - 1 } ^ { m , * }$ contains frozen features from task � − 1, whereas $\mathbf { Z } _ { t - 1 } ^ { m , t }$ contains their Lorentz embeddings after the stage-� transformation. Appendix A summarizes all symbols.

## 4.2 Geometric Stability and Plasticity

Stability. We describe retention of earlier hyperbolic knowledge through changes in the representation geometry. Because geodesic similarity is a monotone function of the Lorentzian inner product, we use three invariants<sup>2</sup>. For every pair of distinct modalities $m , m ^ { \prime } \in \{ 1 , \ldots , M \}$ :

(P1) Intra-modal preservation: $\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m , t } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } ) ^ { \top } } \end{array}$ ;

(P2) Inter-modal preservation: $\mathbf { Z } _ { t - 1 } ^ { m , t } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m ^ { \prime } , t } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m ^ { \prime } , t - 1 } ) ^ { \top }$ ;

(P3) Hierarchical preservation: $\begin{array} { r } { \| ( \mathbf { z } _ { t - 1 } ^ { m , t } ) _ { [ 1 : d ] } \| _ { 2 } = \| ( \mathbf { z } _ { t - 1 } ^ { m , t - 1 } ) _ { [ 1 : d ] } \| _ { 2 } . } \end{array}$

The first condition retains the relations internal to each modality. The second fixes cross-modal correspondences, and the third prevents a point from moving to a diferent radial level of the entailment hierarchy. Together, they protect semantic similarity, modality alignment, and specificity. Appendix C gives further justification.

Plasticity. Stability also leaves enough freedom to learn the current task. At stage �, the embeddings $\{ \mathbf { Z } _ { t } ^ { m , t } \} _ { m = 1 } ^ { M }$ fit the joint multimodal distribution $p _ { t } ^ { ( 1 , 2 , \ldots , M ) }$ . The transformation layers capture new task-specific patterns and associations without using all of their representation capacity.

Core dificulty. Conditions (P1)–(P3) describe this balance, but applying them directly during gradient-based training is dificult. Section 5 gives an equivalent geometric view that leads to practical constraints on parameter updates.

## 5 Geometric Conditions for Admissible Updates

We next convert the preservation requirements into a geometric description of the parameter changes that are permitted during continual adaptation.

## 5.1 Characterizing Preserved Representations

The following result identifies the transformation that can relate old-task embeddings before and after a learning stage without changing the three invariants.

Theorem 1 (Geometric Characterization of Preservation). Suppose the task-� − 1 representations satisfy thefull-rank condition in Appendix D.1, and the stage-to-stage correspondencefollows the continuous optimizer pathfrom the identity component. Conditions (P1), (P2), and (P3) are preservedfrom stage � − 1 to stage � ifand only ifone transformation

$$
\mathbf { R } = \left( \begin{array} { l l } { 1 } & { \mathbf { 0 } ^ { \top } } \\ { \mathbf { 0 } } & { \widetilde { \mathbf { R } } } \end{array} \right) , \qquad \widetilde { \mathbf { R } } \in \mathrm { S O } ( d ) ,
$$

acts on every modality, so that

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } \mathbf { R } ^ { \top } , \qquad m \in \{ 1 , \dotsc , M \} . } \end{array}
$$

Proofsketch. The intra- and inter-modal conditions preserve every required Lorentzian inner product among the task-� − 1 embeddings. Theorem 3 and Corollary 4 in Appendix D.1 then imply a unique Lorentz transformation $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ shared by all modalities. The forward sheet fixes the time orientation, while continuity from the identity fixes the determinant sign. Hierarchical preservation additionally fixes every spatial norm and therefore every time-like coordinate. Full rank then gives $\mathbf { L } ^ { \top } \mathbf { e } _ { 0 } = \mathbf { e } _ { 0 }$ , leaving only the block-diagonal spatial rotation R shown above. Appendix D contains the complete argument.

Role of non-degeneracy. Full rank is the worst-case identifiability condition: it uniquely determines the shared isometry over the entire ambient space. Practical HMCL protects only the retained low-rank subspace and therefore does not require this global uniqueness; deriving the update under the maximally constrained case gives a conservative, robust admissible rule.

## 5.2 First-Order Preservation of Hierarchy

On the Lorentz manifold, the time-like coordinate is determined by the spatial block. We exploit this dependence to obtain a local constraint that prevents a continual update from changing the radial level of an old embedding.

Corollary 1 (Updates under (P3)). Let $\Delta \mathbf { z } = \mathbf { z } _ { t - 1 } ^ { m , t } - \mathbf { z } _ { t - 1 } ^ { m , t - 1 }$ be the change applied to an old-task embedding $\mathbf { z } _ { t - 1 } ^ { m , t - 1 }$ If its time-like coordinate is invariant to first order, as required by (P3), then its spatial change obeys

$$
\begin{array} { r } { \mathbf { z } _ { [ 1 : d ] } ^ { \top } \Delta \mathbf { z } _ { [ 1 : d ] } = 0 . } \end{array}\tag{14}
$$

Takeaway. Corollary 1 requires the spatial displacement of an old embedding to be orthogonal to its current radial direction. In other words, a first-order hierarchy-preserving change lies along the tangent direction and does not introduce time-like drift and therefore avoids radial drift.

Geometric Interpretation. Figure 2 illustrates two implications of the analysis. Hyperbolic distance represents semantic similarity, while radius represents hierarchical specificity. Theorem 1 shows that both quantities remain unchanged across a stage transition when all modalities share the same hyperbolic isometry. From the top-down view, Corollary 1 further limits the local motion to a direction that keeps the radius fixed. A general continual-learning update is not subject to this condition and may distort the geometry even when its task loss is small.

## 6 Hyperbolic Multimodal Continual Learning

The preceding results describe the admissible family in representation space. We now transfer this family to the parameters of a Lorentz transformation layer and obtain a practical training update.

## 6.1 Geometrically Admissible HMCL Updates

Proposition 1 (Admissible Parameter Changes). Consider the Lorentz layerfrom Section 3.1, parameterized by $\mathbf { W } ^ { m , t } = \big [ \mathbf { w } _ { 0 } ^ { m , t } \mathbf { W } _ { s } ^ { m , t } \big ]$ . Assume that Theorem 1 holds at stage � − 1. Define the applied parameter displacement as $\Delta \mathbf { W } ^ { m } = \mathbf { W } ^ { m , t } - \mathbf { W } ^ { m , t - 1 }$ . In the infinitesimal regime $\| \Delta \mathbf { W } ^ { m } \| _ { F } \to 0 .$ , the collection ofmodal displacements is admissible ifand only ifthere is one skew-symmetric matrix � such that, simultaneouslyfor every modality �,

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , * } \big ( \Delta \mathbf { W } ^ { m } \big ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } , \qquad \pmb { \Omega } ^ { \top } = - \pmb { \Omega } . } \end{array}\tag{15}
$$

The left-hand side of (15) is the first-order change of an old spatial representation. The right-hand side is the tangent motion of the rotation exp(�). Crucially, the same � appears for every modality. The condition therefore permits a common change of spatial coordinates, but excludes independent modal rotations that would alter cross-modal relations.

Equation (15) includes both parameter blocks. Under the subspace regularity condition in Appendix E.2, the same output motion has a representative whose time-input column remains fixed; Appendix E.1 gives the precise condition. For every modality, we use the blockwise form

$$
\begin{array} { r } { \Delta \mathbf { w } _ { 0 } ^ { m } = \mathbf { 0 } , \qquad } \\ { \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 { : } d ] \pmb { \Omega } ^ { \top } . } \end{array}\tag{16}
$$

Corollary 2 (HMCL-CA: Closest-Admissible Update). Given candidate parameter displacements $\begin{array} { r l } { \{ \delta ^ { m } } & { { } = } \end{array}$ $[ \delta _ { 0 } ^ { m } \quad \delta _ { s } ^ { m } ] \} _ { m = 1 } ^ { M }$ , where $\pmb { \delta } _ { 0 } ^ { m } \in \mathbb { R } ^ { d \times 1 }$ is the column acting on the time-like input coordinate and $\delta _ { s } ^ { m } \in \mathbb { R } ^ { d \times d }$ is the block acting on the spatial input coordinates, HMCL-CA jointly selects the admissible displacements and their common rotation by solving

$$
\begin{array} { r l } { \underset { \{ \Delta \mathbf { W } ^ { m } \} , \Omega } { \mathrm { m i n i m i z e } } } & { \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { m = 1 } ^ { M } \| \Delta \mathbf { W } ^ { m } - \delta ^ { m } \| _ { F } ^ { 2 } } \\ { s u b j e c t t o } & { \displaystyle \Omega ^ { \top } = - \Omega , \quad \Delta \mathbf { w } _ { 0 } ^ { m } = \mathbf { 0 } , } \\ & { \displaystyle \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \mathbf { \Omega } ^ { \top } , } \\ & { m = 1 , \dots , M . } \end{array}\tag{17}
$$

We denote the solution by $\{ \delta ^ { \mathrm { C A } , m } \} _ { m = 1 } ^ { M }$ and $\Omega ^ { \star }$ . Under the regularity condition in Appendix E.2, the parameter displacements are unique and are obtained from one Lyapunov equation for $\Omega ^ { \star }$ . Thus, HMCL-CA retains the common rotation that best matches all modal candidates, rather than solving a separate rotationfor each modality

Corollary 3 (HMCL-MR: Minimal-Rotation Update). Selecting $\pmb { \Omega } = \pmb { 0 }$ in (17) gives, for each modality,

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 , s } ^ { m , * } \bigl ( \delta _ { s } ^ { \mathrm { M R } , m } \bigr ) ^ { \top } = \mathbf { 0 } , \qquad \delta _ { 0 } ^ { \mathrm { M R } , m } = \mathbf { 0 } , } \end{array}\tag{18}
$$

where $\mathbf { Z } _ { t - 1 , s } ^ { m , * } = \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ]$ . For an orthogonal projector $\mathbf { P } _ { t - 1 }$ onto the protected old-feature subspace, the unique closest solution is

$$
\delta _ { s } ^ { \mathrm { M R } , m } = \delta _ { s } ^ { m } \big ( \mathbf { I } - \mathbf { P } _ { t - 1 } \big ) .\tag{19}
$$

Here, $\mathbf { P } _ { t - 1 }$ projects onto the protected old-feature subspace. This is the closest candidate displacement that induces nofirst-order motion on that subspace.

HMCL-MR is therefore the $\mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Sigma } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Sigma } \mathbf { \Omega } \mathbf { \Sigma } \mathbf { \Omega } \mathbf { \Sigma } \mathbf { \Omega } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf \mathbf { \Sigma \Sigma } \mathbf { \Sigma } \mathbf \mathbf { \Sigma \Sigma } \mathbf { \Sigma } \mathbf \mathbf { \Sigma \Sigma } \mathbf \mathbf { \Sigma \Sigma } \mathbf \mathbf { \Sigma \Sigma \Sigma } \mathbf \mathbf  \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \mathbf \Sigma \mathbf \Sigma \mathbf \mathbf \Sigma \mathbf \Sigma \mathbf \mathbf \mathbf \Sigma \mathbf \Sigma \mathbf \mathbf \mathbf \Sigma \mathbf \Sigma \mathbf \mathbf \mathbf \mathbf \Sigma \mathbf \mathbf \mathbf \mathbf \Sigma \mathbf \mathbf \mathbf \Sigma \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \Sigma \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf$ member of the same closest-admissible problem. Appendix E.1 proves Proposition 1; Appendices E.2 and E.3 give the corresponding derivations.

In summary, hyperbolic preservation determines one admissible family: HMCL-CA selects its closest sharedrotation update, and HMCL-MR selects the zero-rotation member.

## 6.2 Stepwise Correction and Task Anchoring

We follow the standard training paradigm of the hyperbolic multimodal backbones, which optimizes their Euclideanparameterized Lorentz heads with AdamW. Because AdamW’s moment estimation and coordinate-wise scaling make its realized parameter displacement diferent from the raw gradient, HMCL applies post-AdamWprojection to that realized displacement. It then uses task anchoring to contract the corrected parameters toward the current task start. During task $t , J$ denotes the total number of optimizer steps and $k \in \{ 0 , \ldots , J - 1 \}$ indexes them, with $\mathbf { W } _ { 0 } ^ { m , t } = \mathbf { W } ^ { m , t - 1 }$ and $\mathbf { W } _ { J } ^ { m , t } = \mathbf { W } ^ { m , t }$ . A first subscript � denotes the within-task optimizer step; a second subscript after a comma denotes an input block, with 0 for the time-like input column and � for the spatial input block. Thus, for $r \in \{ \mathrm { o p t } , \star \}$

$$
\mathbf { W } _ { k } ^ { m , t } = [ \mathbf { w } _ { k , 0 } ^ { m , t } \mathbf { W } _ { k , s } ^ { m , t } ] , \qquad \delta _ { k } ^ { r , m , t } = [ \delta _ { k , 0 } ^ { r , m , t } \mathbf { \delta } \delta _ { k , s } ^ { r , m , t } ] .
$$

In particular, the single subscript in $\mathbf { W } _ { 0 } ^ { m , t }$ identifies the full parameter matrix at optimizer step 0; it is not a block label. With the AdamW state left implicit, one optimizer step produces

$$
\begin{array} { r l } & { \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } = \mathrm { O p t } _ { k } ^ { m } ( \mathbf { W } _ { k } ^ { m , t } , \mathbf { \mathbf { g } } _ { k } ^ { m , t } ) , } \\ & { \delta _ { k } ^ { \mathrm { o p t } , m , t } = \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { k } ^ { m , t } . } \end{array}
$$

Here $\mathbf { g } _ { k } ^ { m , t }$ is the raw gradient, whereas $\delta _ { k } ^ { \mathrm { o p t } , m , t }$ is the realized parameter displacement proposed by AdamW before HMCL correction. It already includes the learning rate, moment estimation, coordinate-wise preconditioning, and decoupled weight decay; it is not a gradient.

To state the geometric correction at the level of this proposed step, let $\pmb { \xi } _ { k } ^ { m , t } = [ \pmb { \xi } _ { k , 0 } ^ { m , t } \ \pmb { \xi } _ { k , s } ^ { m , t } ]$ denote an arbitrary parameter displacement at within-task step �, with the same time-input/spatial-input partition as $\mathbf { W } _ { k } ^ { m , t }$ . For any collection $\{ \pmb { \xi } _ { k } ^ { m , t } \} _ { m = 1 } ^ { M }$ , define its Lorentz defect by

$$
\mathcal { E } _ { \mathcal { L } , t - 1 } ( \{ \xi _ { k } ^ { m , t } \} ) = \operatorname* { m i n } _ { \Omega ^ { \top } = - \Omega } \sum _ { m = 1 } ^ { M } \left. \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 ; d ] ( \xi _ { k , s } ^ { m , t } ) ^ { \top } \right. _ { F } ^ { 2 } + \sum _ { m = 1 } ^ { M } \Vert \xi _ { k , 0 } ^ { m , t } \Vert _ { 2 } ^ { 2 } .
$$

Proposition 1 gives $\mathcal { E } _ { \mathcal { L } , t - 1 } = 0$ exactly for a fixed time-input column and one spatial Lorentz rotation shared across modalities. Let $\mathcal { P } _ { t - 1 } ^ { \mathcal { L } }$ be the Frobenius closest-point map onto this zero-defect family, denoted by $\mathcal { A } _ { t - 1 }$ . At optimizer step $k ,$ the defect is evaluated at $\pmb { \xi } _ { k } ^ { m , t } = \delta _ { k } ^ { \mathrm { o p t } , m , t }$ , whose two blocks are $\delta _ { k , 0 } ^ { \mathrm { o p t } , m , t }$ and $\delta _ { k , s } ^ { \mathrm { o p t } , m , t }$ . We collect the modal displacements as $\delta _ { k } ^ { \mathrm { o p t } , t } = ( \delta _ { k } ^ { \mathrm { o p t } , m , t } ) _ { m = 1 } ^ { M }$ , with analogous notation for $\delta _ { k } ^ { \star , t }$ . For $0 \leq \beta < 1$ , the combined update is

$$
\begin{array} { r l } & { \boldsymbol { \delta } _ { k } ^ { \star , t } = \boldsymbol { \mathcal { P } } _ { t - 1 } ^ { \mathcal { L } } ( \boldsymbol { \delta } _ { k } ^ { \mathrm { o p t } , t } ) , } \\ & { \mathbf { W } _ { k + 1 } ^ { m , t } = \boldsymbol { \beta } \mathbf { W } _ { 0 } ^ { m , t } + \left( 1 - \beta \right) \left( \mathbf { W } _ { k } ^ { m , t } + \boldsymbol { \delta } _ { k } ^ { \star , m , t } \right) . } \end{array}\tag{20}
$$

For a modal tuple $\mathcal { U } = ( \mathbf { U } ^ { m } ) _ { m = 1 } ^ { M }$ , write $\begin{array} { r } { \| \mathcal { U } \| _ { \oplus F } : = ( \sum _ { m } \| \mathbf { U } ^ { m } \| _ { F } ^ { 2 } ) ^ { 1 / 2 } } \end{array}$

Theorem 2 (Anchored Admissibility and Bounded Accumulation). For the update in (20) on a protected representation systemfixed throughout task �, the cumulative displacement ${ \bf A } _ { J } ^ { t } = ( { \bf A } _ { J } ^ { m } ) _ { m = 1 } ^ { M }$ , where $\mathbf { A } _ { J } ^ { m } = \mathbf { W } _ { J } ^ { m , t } - \mathbf { W } _ { 0 } ^ { m , t }$ satisfies $\{ \mathbf { A } _ { J } ^ { m } \} _ { m = 1 } ^ { M } \in \mathcal { A } _ { t - 1 }$ and $\mathscr { E } _ { \mathcal { L } , t - 1 } \big ( \{ \mathbf { A } _ { J } ^ { m } \} \big ) = 0$ . Thus, the protected representations undergo one shared first-order Lorentz rotation across all modalities. The projection is non-expansive in the joint Frobenius norm. $I f \| \delta _ { k } ^ { \mathrm { o p t } , t } \| _ { \oplus F } \leq u _ { \mathrm { m a x } }$ for every �, then task anchoring with $0 < \beta < 1$ also gives the uniform bound $\| \mathbf { A } _ { J } ^ { t } \| _ { \oplus F } \le ( \ddot { 1 } - \beta ) u _ { \mathrm { m a x } } / \beta .$

Takeaway. HMCL separates stability from plasticity at each optimizer step. Projection removes only the part of AdamW’s realized displacement that violates the shared Lorentz-isometry constraint, leaving the admissible component free to learn the current task. Task anchoring then limits cumulative drift without leaving this family.

## 7 Experiments

Our evaluation is organized around four questions concerning continual adaptation of hyperbolic multimodal models:

• RQ1: Does HMCL improve the stability–plasticity balance, measured by final performance and backward transfer, across hyperbolic multimodal backbones?

• RQ2: Does HMCL reduce radial, angular, cross-modal, and paired-distance drift on previously learned tasks?

• RQ3: Does the resulting representation preserve semantic hierarchies in quantitative and qualitative tests?

• RQ4: Within the unified HMCL update, what are the separate efects of correction placement and task anchoring?

Accordingly, Section 7.2 addresses RQ1, Section 7.3 addresses RQ2, Sections 7.4 and 7.5 address RQ3, and Section 7.6 addresses RQ4.

## 7.1 Experimental Setup

Task Stream and Datasets. The continual stream combines 14 multimodal classification tasks with the COCO [64] and Flickr30K [65] cross-modal retrieval tasks. ImageNet-1K [66] is inserted after Caltech-101, yielding 16 tasks in total. Non-replay methods never revisit earlier-task data, whereas replay baselines access it only through a bounded bufer. Appendices G.1 and G.2 provide the complete dataset statistics, canonical order, and alternative-order protocols.

Metrics. For retrieval, the implementation records image-to-text and text-to-image Recall@� for $k \in \{ 1 , 5 , 1 0 \}$ The main comparison uses symmetric R@5, i.e., the mean of image-to-text and text-to-image R@5, for both retrieval tasks. Classification performance is top-1 accuracy. Overall is the unweighted mean of the 14 final classification accuracies and the two final symmetric retrieval scores. We define signed backward transfer as $\mathrm { B W T = P e r f o r m a n c e _ { f i n a l } - P e r f o r m a n c e _ { r i g h t ~ a f t e r ~ t a s k } ; }$ ; thus, a negative value denotes forgetting. The main table reports this quantity separately for classification, retrieval, and all tasks. Reported entries are means and sample standard deviations over five paired seeds.

Baselines. Our comparison covers the principal families of continual-learning strategies under the same hyperbolic objective: ordinary sequential fine-tuning (Vanilla), parameter regularization (EWC), replay-based constrained optimization (GEM), vision–language distillation with flatness-aware optimization (C-FLAT), and dual-sided null-space projection (DNS) [42, 48, 52, 63]. This breadth tests HMCL against methods that preserve parameters, examples, representations, or update subspaces; none of them explicitly preserves a shared Lorentz isometry across modalities.

Implementation Details. We follow each backbone’s hyperbolic multimodal training paradigm: the pretrained encoder is frozen, continual adaptation is confined to a bias-free linear Lorentz head, and AdamW [59] optimizes the common objective. Classification and retrieval tasks use 10 and 15 epochs with learning rates $5 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 5 }$ , respectively; all use batch size 1024, curvature −0.1, and one accumulated AdamW step per epoch. Main HMCL runs apply post-AdamW correction followed by task anchoring. Every method shares the task order, cached 512-dimensional features, schedule, optimizer-step budget, and five paired seeds. Appendix H.5 gives the complete configurations and baseline memory budgets, while Appendices H.8, H.6, and H.7 report sensitivity and timing details.

## 7.2 Main Results (RQ1)

To answer RQ1, we compare final task performance and signed backward transfer under the matched 16-task protocol across all three hyperbolic multimodal backbones.

HMCL gives the strongest stability–plasticity balance across backbones. Table 1 shows that both complete HMCL variants outperform every external baseline on all three backbones in final classification and retrieval performance as well as in the Overall score. Their BWT values also move substantially toward zero, so higher final scores accompany stronger retention rather than sacrificing earlier tasks. $\mathrm { \ H M C L _ { C A } }$ gives the highest Overall score in every backbone block and consistently exceeds task-balanced GEM. DNS improves only modestly over Vanilla, suggesting that Euclidean null-space protection alone does not fully capture the constraints of hyperbolic representations. The agreement across classification and retrieval is informative because the former tests category discrimination, whereas the latter depends on preserving correspondence between modalities. Improvements in both settings are consistent with protecting their shared representation geometry, rather than benefiting only one prediction rule. Appendix H.1 provides the complete per-task results.

Table 1. Continual-learning results on the unified 16-task stream (mean ± sample standard deviation over five paired seeds). Classification is top-1 accuracy, retrieval is symmetric R@5, and BWT is final minus immediate performance. Light-blue rows denote the complete post-AdamW, task-pullback HMCL variants; boldface marks the best result in each backbone block. The Δ rows give $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ ’s relative improvement over Vanilla, using the reduction in negative magnitude for BWT. All method share the task order, cached features, training schedule, and optimizer-step budget.
<table><tr><td colspan="2"></td><td colspan="2">Classification</td><td colspan="2">Retrieval</td><td colspan="2">Overall</td></tr><tr><td>Backbone</td><td>Method</td><td>Acc ↑</td><td> $\mathbf { B W T _ { A } } \uparrow$ </td><td>R@5↑</td><td> $\mathbf { B W T } _ { \mathrm { R 5 } } \uparrow$ </td><td>Performance ↑</td><td>BWT↑</td></tr><tr><td rowspan="8">MERU-L</td><td>Vanilla</td><td> $4 0 . 2 5 1 \pm 0 . 2 0 2$ </td><td> $- 5 . 4 6 1 \pm 0 . 1 8 0$ </td><td> $2 9 . 6 2 5 \pm 0 . 1 1 0$ </td><td> $- 2 . 4 7 9 \pm 0 . 1 2 3$ </td><td> $3 8 . 9 2 2 \pm 0 . 1 6 5$ </td><td> $- 5 . 0 8 8 \pm 0 . 1 4 5$ </td></tr><tr><td>EWC</td><td> $4 0 . 3 2 1 \pm 0 . 1 8 7$ </td><td> $- 5 . 4 6 7 \pm 0 . 2 1 3$ </td><td> $3 0 . 2 1 5 \pm 0 . 1 4 6$ </td><td> $- 2 . 0 9 6 \pm 0 . 1 5 8$ </td><td> $3 9 . 0 5 8 \pm 0 . 1 5 3$ </td><td> $- 5 . 0 4 6 \pm 0 . 1 7 4$ </td></tr><tr><td>GEM</td><td> $4 1 . 6 1 3 \pm 0 . 2 8 8$ </td><td> $- 3 . 7 4 2 \pm 0 . 2 8 4$ </td><td> $2 9 . 2 2 0 \pm 0 . 2 9 9$ </td><td> $- 2 . 3 0 6 \pm 0 . 4 1 8$ </td><td> $4 0 . 0 6 4 \pm 0 . 2 7 6$ </td><td> $- 3 . 5 6 3 \pm 0 . 2 7 6$ </td></tr><tr><td>C-FLAT</td><td> $4 0 . 4 5 0 \pm 0 . 1 7 1$ </td><td> $- 3 . 9 4 5 \pm 0 . 1 0 8$ </td><td> $2 9 . 4 4 6 \pm 0 . 1 7 4$ </td><td> $- 2 . 0 3 2 \pm 0 . 1 1 5$ </td><td> $3 9 . 0 7 5 \pm 0 . 1 4 3$ </td><td> $- 3 . 7 0 6 \pm 0 . 0 8 6$ </td></tr><tr><td>DNS</td><td> $4 0 . 9 0 4 \pm 0 . 1 4 9$ </td><td> $- 4 . 9 1 2 \pm 0 . 1 7 4$ </td><td> $2 9 . 7 1 6 \pm 0 . 1 7 0$ </td><td> $- 2 . 4 5 5 \pm 0 . 1 0 0$ </td><td> $3 9 . 5 0 5 \pm 0 . 1 2 6$ </td><td> $- 4 . 6 0 5 \pm 0 . 1 4 9$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { M R } }$ </td><td> $4 3 . 8 9 1 \pm 0 . 0 8 5$ </td><td> $- 1 . 3 1 6 \pm 0 . 0 8 6$ </td><td> $\mathbf { 3 6 . 3 0 0 \pm 0 . 1 1 4 }$ </td><td> $\mathbf { - 1 . 9 2 1 \pm 0 . 0 5 7 }$ </td><td> $4 2 . 9 4 2 \pm 0 . 0 7 0$ </td><td> $\mathbf { - 1 . 3 9 2 \pm 0 . 0 7 0 }$ </td></tr><tr><td> $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ </td><td> $\mathbf { 4 3 . 9 8 7 \pm 0 . 0 8 7 }$ </td><td> $\mathbf { - 1 . 3 0 5 \pm 0 . 0 9 0 }$ </td><td> $3 6 . 0 1 1 \pm 0 . 1 3 6$ </td><td> $- 2 . 0 4 6 \pm 0 . 0 9 9$ </td><td> $\mathbf { 4 2 . 9 9 0 \pm 0 . 0 7 3 }$ </td><td> $- 1 . 3 9 8 \pm 0 . 0 7 8$ </td></tr><tr><td>∆ vs. Vanilla</td><td>+9.3%</td><td>+76.1%</td><td>+21.6%</td><td>+17.5%</td><td>+10.5%</td><td>+72.5%</td></tr><tr><td rowspan="8">MERU-B</td><td>Vanilla</td><td>43.326 ± 0.243</td><td>−6.617 ± 0.244</td><td> $3 0 . 5 3 5 \pm 0 . 0 9 3$ </td><td>−4.610 ± 0.120</td><td> $4 1 . 7 2 7 \pm 0 . 2 2 2$ </td><td>−6.366 ± 0.218</td></tr><tr><td>EWC</td><td>43.472 ± 0.210</td><td>−6.490 ± 0.217</td><td> $3 0 . 9 6 5 \pm 0 . 0 8 6$ </td><td> $- 4 . 2 9 1 \pm 0 . 1 7 2$ </td><td> $4 1 . 9 0 9 \pm 0 . 1 9 0$ </td><td></td></tr><tr><td>GEM</td><td>44.880 ± 0.272</td><td></td><td></td><td></td><td></td><td> $- 6 . 2 1 5 \pm 0 . 1 8 9$ </td></tr><tr><td>C-FLAT</td><td>42.561 ± 0.236</td><td>−3.803 ± 0.093 −4.700 ± 0.146</td><td> $3 2 . 0 1 0 \pm 0 . 3 7 3$ </td><td>−3.009 ± 0.238 −4.300 ± 0.237</td><td> $4 3 . 2 7 1 \pm 0 . 2 2 8$ </td><td> $- 3 . 7 0 4 \pm 0 . 0 5 6$ </td></tr><tr><td>DNS</td><td>43.769± 0.163</td><td></td><td> $3 1 . 4 5 9 \pm 0 . 1 6 1$  −6.265 ± 0.225 31.423 ± 0.073</td><td>−4.138 ± 0.193</td><td> $4 1 . 1 7 4 \pm 0 . 2 1 1$   $4 2 . 2 2 6 \pm 0 . 1 5 1$ </td><td> $- 4 . 6 5 0 \pm 0 . 1 2 4$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>−5.999 ± 0.217</td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { M R } }$ </td><td></td><td></td><td></td><td></td><td></td><td>45.736±0.042 -3.448±0.112 36.456±0.090 -1.392±0.162 44.576±0.036 -3.191±0.089</td></tr><tr><td> $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$  ∆ vs. Vanilla</td><td> $\mathbf { 4 5 . 7 4 6 \pm 0 . 0 7 8 }$  +5.6%</td><td> $\mathbf { - 3 . 1 5 9 \pm 0 . 1 3 4 }$  +52.3%</td><td> $\mathbf { 3 6 . 5 2 2 \pm 0 . 0 8 3 }$  +19.6%</td><td> $\mathbf { - 1 . 2 2 9 \pm 0 . 1 1 6 }$  +73.3%</td><td> $\mathbf { 4 4 . 5 9 3 \pm 0 . 0 7 1 }$  +6.9%</td><td> $\mathbf { - 2 . 9 1 8 \pm 0 . 1 1 1 }$  +54.2%</td></tr><tr><td rowspan="8">HyCoCLIP-B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla EWC</td><td> $4 0 . 9 8 8 \pm 0 . 1 6 0$ </td><td> $- 4 . 4 2 1 \pm 0 . 0 9 7$ </td><td> $6 2 . 0 3 4 \pm 0 . 1 8 2$ </td><td> $- 2 . 5 4 4 \pm 0 . 1 7 2$ </td><td> $4 3 . 6 1 8 \pm 0 . 1 5 2$ </td><td> $- 4 . 1 8 6 \pm 0 . 0 8 7$ </td></tr><tr><td></td><td> $4 1 . 1 2 1 \pm 0 . 1 2 9$ </td><td> $- 4 . 3 8 5 \pm 0 . 1 2 2$ </td><td> $6 3 . 3 6 7 \pm 0 . 1 4 8$ </td><td> $- 1 . 4 2 3 \pm 0 . 0 9 1$ </td><td> $4 3 . 9 0 1 \pm 0 . 1 1 8$ </td><td> $- 4 . 0 1 4 \pm 0 . 1 0 7$ </td></tr><tr><td>GEM</td><td> $4 2 . 1 1 1 \pm 0 . 3 1 8$ </td><td> $- 3 . 2 3 6 \pm 0 . 3 5 3$ </td><td> $6 2 . 5 8 6 \pm 0 . 1 2 5$ </td><td> $- 1 . 4 4 9 \pm 0 . 1 2 1$ </td><td> $4 4 . 6 7 0 \pm 0 . 2 6 8$ </td><td> $- 3 . 0 1 3 \pm 0 . 3 1 5$ </td></tr><tr><td>C-FLAT</td><td> $4 0 . 9 5 1 \pm 0 . 1 1 6$ </td><td>−3.699 ± 0.097</td><td> $6 2 . 0 1 0 \pm 0 . 0 4 2$ </td><td> $- 1 . 8 3 9 \pm 0 . 0 7 8$ </td><td> $4 3 . 5 8 3 \pm 0 . 1 0 4$ </td><td> $- 3 . 4 6 6 \pm 0 . 0 9 3$ </td></tr><tr><td>DNS</td><td> $4 1 . 3 5 6 \pm 0 . 1 0 5$ </td><td> $- 4 . 0 6 5 \pm 0 . 0 7 6$ </td><td> $6 2 . 5 3 6 \pm 0 . 1 8 9$ </td><td> $- 2 . 0 6 8 \pm 0 . 1 6 9$ </td><td> $4 4 . 0 0 4 \pm 0 . 1 0 8$ </td><td> $- 3 . 8 1 5 \pm 0 . 0 7 7$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { M R } }$ </td><td> $4 4 . 8 2 8 \pm 0 . 0 6 4$ </td><td> $- 0 . 6 3 5 \pm 0 . 0 6 9$ </td><td> $\mathbf { 7 2 . 8 0 1 \pm 0 . 0 8 7 }$ </td><td> $\mathbf { - 0 . 8 7 9 \pm 0 . 0 9 1 }$ </td><td> $4 8 . 3 2 5 \pm 0 . 0 5 3$ </td><td> $- 0 . 6 6 6 \pm 0 . 0 6 9$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { C A } }$  ∆ vs. Vanilla</td><td> $\mathbf { 4 5 . 1 0 8 \pm 0 . 0 3 1 }$  +10.1%</td><td> $\mathbf { - 0 . 3 8 8 \pm 0 . 0 6 1 }$  +91.2%</td><td> $7 2 . 7 7 2 \pm 0 . 0 7 3$  +17.3%</td><td> $- 0 . 9 3 7 \pm 0 . 0 5 4$  +63.2%</td><td> $\mathbf { 4 8 . 5 6 6 \pm 0 . 0 2 4 }$   $+ 1 1 . 3 \%$ </td><td> $\mathbf { - 0 . 4 5 6 \pm 0 . 0 5 1 }$  +89.1%</td></tr></table>

CA and MR expose complementary members of the admissible family. On the unified 16-task stream, CA generally gives higher classification and Overall scores, while MR is stronger on retrieval for MERU-L and HyCoCLIP-B and gives the best Overall BWT on MERU-L. This division is not universal: in the modality-extended stream below, CA also gives the stronger retrieval result on all three backbones. The evidence therefore supports a task-dependent stability–plasticity trade-of within one admissible family, rather than assigning a fixed empirical role to either variant.

HMCL remains stable throughout the task stream. Figure 3 normalizes each task by its own task-end score. On the displayed paired seed, Vanilla exhibits pronounced degradation on earlier tasks, whereas both HMCL variants keep the trajectories closer to their task-end reference. The contrast is visible across intermediate stages as well as at the final checkpoint, suggesting that the retention gains reflect sustained protection during adaptation rather than recovery only at the end of the stream. These old-task trajectories provide a stage-wise view of the five-seed BWT gains in Table 1; the displayed seed illustrates the temporal pattern rather than replacing the multi-seed comparison.

![](images/efa3dc1e08140e228d269ee152e658fa44b5ab7b5eac7f00b6a38d69f460a2b5.jpg)  
Figure 3. Stage-wise relative task-performance change for HyCoCLIP-B on seed 42 in the unified 16-task stream. Columns correspond to the completed training stage �, and rows correspond to evaluated task � in stream order. Each observed cell reports $1 0 0 ( P _ { i , t } - P _ { i , i } ) / P _ { i , i }$ , where $P _ { i , i }$ is the score immediately after task � is learned; thus, the outlined diagonal is zero, red denotes relative degradation, and blue denotes improvement. The gray upper triangle contains tasks not yet introduced. All panels share one color scale, the outlined rightmost column gives the final-stage changes, and (R) marks retrieval tasks evaluated with symmetric R@5; the remaining tasks use top-1 accuracy.

The gains persist under alternative task orders. Appendix G.2 evaluates two shufled HyCoCLIP-B streams with the main-table configurations fixed and no order-specific retuning. Across the paired seeds, both HMCL variants retain higher final classification, retrieval, and Overall scores and less negative BWT than Vanilla under either permutation. Their relative ranking changes with the order: MR leads in Overall on one permutation, whereas CA leads on the other. These results support robustness beyond the canonical sequence while reinforcing that the preferred admissible representative depends on the task stream.

Table 2. Robustness to new modalities in a separate protocol with one audio and one thermal retrieval task (mean ± standard deviation over three paired seeds). Cls. is top-1 accuracy, Ret. is symmetric R@5, Overall averages all final task scores, and BWT is final minus immediate performance. Both HMCL variants use after-AdamW correction and step pullback $( \beta = 0 . 0 1 5 )$ . Boldface marks the best result per backbone and metric.

HMCL remains efective when the stream introduces new modalities. Table 2 reports a separate protocol that adds one audio and one thermal retrieval task to the original stream. Both complete HMCL variants improve the Overall score and BWT over matched Vanilla training on every backbone. CA consistently improves retrieval R@5 while substantially reducing forgetting across the backbones. MR gives a slightly higher Overall score on the two MERU backbones, whereas CA is strongest on HyCoCLIP-B. The result shows that the shared geometric constraint remains useful when previously unseen modality types enter the stream. This extension is a distinct stress test: adaptation now accommodates additional cross-modal relations while retaining those learned earlier. The consistent retrieval

<table><tr><td>Backbone</td><td>Metric</td><td>Vanilla</td><td> $\mathbf { H M C L } _ { \mathrm { M R } }$ </td><td> $\mathbf { H M C L } _ { \mathbf { C A } }$ </td></tr><tr><td rowspan="4">MERU-L</td><td>Cls.</td><td> $4 2 . 5 1 4 \pm 0 . 0 9 2$ </td><td> $\mathbf { 4 3 . 8 9 4 \ : \pm 0 . 1 3 0 }$ </td><td> $4 3 . 1 8 8 \pm 0 . 0 4 6$ </td></tr><tr><td>Ret.</td><td> $1 6 . 2 2 1 \pm 0 . 0 4 8$ </td><td> $1 6 . 2 7 8 \pm 0 . 1 0 5$ </td><td> ${ \bf 1 8 . 2 2 4 \pm 0 . 0 7 7 }$ </td></tr><tr><td>Overall</td><td> $3 6 . 3 2 7 \pm 0 . 0 6 4$ </td><td> $\mathbf { 3 7 . 3 9 6 \pm 0 . 0 9 9 }$ </td><td> $3 7 . 3 1 5 \pm 0 . 0 2 6$ </td></tr><tr><td>BWT</td><td> $- 4 . 8 7 6 \pm 0 . 0 5 4$ </td><td> $- 1 . 6 5 0 \pm 0 . 0 3 8$ </td><td> $\mathbf { - 1 . 1 1 0 \pm 0 . 0 3 6 }$ </td></tr><tr><td rowspan="4">MERU-B</td><td>Cls.</td><td> $4 4 . 1 5 6 \pm 0 . 1 9 0$ </td><td> $\mathbf { 4 6 . 6 4 6 \pm 0 . 2 1 3 }$ </td><td> $4 5 . 8 1 5 \pm 0 . 0 5 0$ </td></tr><tr><td>Ret.</td><td></td><td>14.868 ± 0.131 17.000 ± 0.07618.482 ± 0.053</td><td></td></tr><tr><td></td><td>Overall 37.265 ± 0.116 39.670 ± 0.145</td><td></td><td> $3 9 . 3 8 4 \pm 0 . 0 2 8$ </td></tr><tr><td>BWT</td><td> $- 7 . 2 9 3 \pm 0 . 1 9 1$ </td><td>−4.354 ± 0.120 −2.871 ± 0.067</td><td></td></tr><tr><td rowspan="4">HyCoCLIP-B</td><td>Cls.</td><td> $4 2 . 8 6 9 \pm 0 . 1 5 4$ </td><td>44.563 ± 0.180</td><td> $\mathbf { 4 5 . 3 4 8 \pm 0 . 0 7 7 }$ </td></tr><tr><td>Ret.</td><td> $3 2 . 9 7 0 \pm 0 . 1 2 3$ </td><td>33.506 ± 0.123</td><td> $\mathbf { 3 5 . 4 0 7 \pm 0 . 0 8 1 }$ </td></tr><tr><td>Overall</td><td> $4 0 . 5 4 0 \pm 0 . 1 4 3$ </td><td>41.961 ± 0.155</td><td> $\mathbf { 4 3 . 0 0 9 \pm 0 . 0 7 2 }$ </td></tr><tr><td>BWT</td><td></td><td>−3.859 ± 0.062 −2.603 ± 0.056 −1.658 ± 0.017</td><td></td></tr></table>

gains suggest that geometric protection remains useful beyond the original image–text setting, although the changing MR–CA ranking cautions against treating either realization as uniformly preferable. Separate evaluation of earlier image–text tasks distinguishes retention of existing correspondences from new-modality gains.

![](images/ff7e1616c6ddeb94c7aec6863cd81da536f72ac4f2fd9238c3e78ed3611b0ff1.jpg)  
Figure 4. Fine-grained final retrieval performance on COCO and Flickr30K in the separate stream augmented with audio and thermal tasks. Each panel reports the mean over three paired seeds for one backbone; T2I and I2T denote text-to-image and image-to-text Recall@{1, 5, 10}, respectively. Gray, blue, and green bars denote Vanilla, $\mathrm { H M C L } _ { \mathrm { M R } } .$ , and $\mathrm { \ H M C L _ { C A } }$ Appendix H.4 reports exact means and standard deviations. Higher values indicate better retrieval.

The retrieval gains persist at fine granularity. Figure 4 shows final COCO and Flickr30K retrieval across backbones, directions, and recall cutofs; Appendix H.4 gives the complete numerical results. Under the same modality-extension protocol, $\mathrm { H M C L } _ { \mathrm { C A } }$ exceeds Vanilla and gives the highest score in all 36 backbone–dataset– direction–cutof comparisons. Its advantage therefore spans both retrieval directions and every recall cutof, not only symmetric R@5.

## 7.3 Quantitative Representation Drift (RQ2)

To answer RQ2, we go beyond task-level BWT, which records predictive change but does not identify which parts of the hyperbolic representation have moved. We compare each old task at its task-end checkpoint and at the final checkpoint using four complementary drift measures:

• Radial drift is the mean absolute change of the time-like coordinate, averaged over image and text anchors. It tests hierarchical preservation in Condition (P3).

• Angular drift is the mean absolute change of the normalized spatial Gram matrix, averaged over the image and text blocks. It removes magnitude efects and tracks the intra-modal relations in Condition (P1).

• Cross-modal drift is the mean absolute change of the normalized spatial image–text Gram block and therefore measures the inter-modal relations in Condition (P2).

• Paired-distance drift is the mean change in Lorentz geodesic distance between matched image–text anchors. It tests cross-modal alignment in Condition (P2) while remaining sensitive to hierarchy-related radial changes in Condition (P3).

Figure 5 shows lower drift for both HMCL variants across all 15 old tasks and all four measures. Both variants substantially suppress radial, angular, cross-modal, and paired-distance drift, with CA giving slightly lower mean drift throughout. For both variants, every paired comparison favors HMCL and is significant under an exact two-sided Wilcoxon signed-rank test $( p = 6 . 1 \times 1 0 ^ { - 5 }$ ; Bonferroni-adjusted $p = 2 . 4 \times 1 0 ^ { - 4 }$ over four measures). The agreement between MR and CA shows that the stability gain is not specific to the zero-rotation realization: both retain radial hierarchy, within-modality relations, and cross-modal geometry throughout the stream.

The joint behavior of these measures is more informative than any one measure alone. Small angular changes would not exclude radial contraction, and stable within-modality relations would not establish that image–text correspondence is retained. Lower paired-distance drift complements the normalized Gram measurements by tracking matched examples in the original Lorentz geometry. Taken together, the observations support preservation of both hierarchical and relational structure. They are consistent with the proposed mechanism, while remaining an empirical representation-space diagnostic rather than a proof that drift reduction alone causes the task-level gains.

![](images/94592d784edf2a0a40d32870b3e5b31f9c7399790d45facea375635509e328e8.jpg)  
Figure 5. Dataset-wise drift for MERU-L (seed 1024) on the unified 16-task stream, measured from each task-end checkpoint to the final checkpoint using 50 fixed test anchors per old task. The 15 plotted rows exclude SST-2 because it is the final task and hence has no post-task drift interval. Gray circles, blue diamonds, and green squares denote Vanilla, $\mathrm { H M C L } _ { \mathrm { M R } }$ , and $\mathrm { \ H M C L _ { C A } }$ , respectively; shaded rows identify retrieval tasks. Lower values indicate better preservation. In-panel percentages report each HMCL variant’s relative reduction in mean drift over the 15 plotted tasks compared with Vanilla.

## 7.4 ImageNet–WordNet Hierarchy (RQ3)

WordNet Evaluation Protocol. To answer RQ3 quantitatively, ImageNet measures new-task capacity and WordNet [67] measures whether the learned semantic hierarchy survives later tasks. Following the HyCoCLIP protocol [8] and standard hierarchical-classification evaluation [68], we report top-1 accuracy, tree-induced error (TIE), lowest-common-ancestor distance (LCA), ancestor Jaccard, and hierarchical precision and recall. Lower TIE and LCA are better; higher values are better for all other metrics. These metrics separate prediction correctness from semantic error severity: top-1 counts every mistake equally, whereas TIE and LCA distinguish nearby errors from cross-branch confusions; ancestor Jaccard measures shared ancestry, and hierarchical precision and recall summarize the correctness and coverage of predicted ancestor paths. Their joint use prevents an apparent hierarchy gain from being attributed to a change in accuracy or only one aspect of the tree path. We therefore compute the hierarchy scores from the same final predictions as top-1 and interpret improvement only when predictive retention and semantic proximity move consistently. Appendix H.1 gives the split, hierarchy mapping, retained ranks, and configuration details.

![](images/9ba8a3aa8fff1c148ce6a080bb299e6bd51e7b14e03d6c2d82e6862011214820.jpg)  
Figure 6. Radial geometry of the complete 20-class ImageNet primate WordNet subtree for HyCoCLIP-B (seed 42). (a) Geometry immediately after learning ImageNet; (b–d) geometry at the final checkpoint of the 16-task stream. Final representations are globally aligned to the task-end reference by a proper spatial rotation before display, so the remaining change reflects radial and relational distortion rather than an arbitrary global orientation. Node color denotes WordNet depth, and edges show the fixed subtree relations. “rad.” is the mean absolute radial change normalized by the task-end mean radius; “pair.” is the mean absolute change in pairwise Lorentz distance normalized by the task-end mean distance. Both quantities are computed in the original representation space rather than from the two-dimensional drawing.

HMCL preserves the ImageNet hierarchy. Table 3 reports the final HyCoCLIP-B checkpoints from Table 1. Both HMCL variants outperform all external baselines in top-1 accuracy and every hierarchy metric. Lower TIE and LCA together with higher ancestor Jaccard, hierarchical precision, and hierarchical recall show that their remaining errors preserve more of the correct WordNet ancestry. These concurrent gains show that hierarchy retention is not achieved by sacrificing classification performance. MR performs best on the ImageNet–WordNet measures, whereas CA gives the highest Overall score on the full stream, indicating a modest

Table 3. Final ImageNet accuracy and WordNet hierarchy metrics on the 16-task HyCoCLIP-B stream (five-seed means). HMCL uses the configurations in Table 1; shaded rows identify HMCL.
<table><tr><td>Method Top-1</td><td colspan="4">TIE LCA ↓ Jacc. ↑ H-Prec. ↑ H-Rec. ↑ 7</td></tr><tr><td>Vanilla</td><td>39.404 3.598</td><td>2.217 0.7869</td><td>0.8539</td><td>0.8560</td></tr><tr><td>EWC</td><td>40.045 3.549</td><td>2.199 0.7901</td><td>0.8560</td><td>0.8583</td></tr><tr><td>C-FLAT</td><td>38.934 3.634</td><td>2.237 0.7849</td><td>0.8521</td><td>0.8552</td></tr><tr><td>DNS</td><td>39.494 3.594</td><td>2.222 0.7875</td><td>0.8538</td><td>0.8568</td></tr><tr><td>HMCLMR HMCLCA 45.590</td><td>45.700 3.137 3.156</td><td>2.042 0.8162 2.051 0.8152</td><td>0.8743 0.8735</td><td>0.8757 0.8751</td></tr></table>

trade-of between hierarchy retention and aggregate performance. Thus, selecting a model solely by Overall can obscure diferences in how well semantic ancestry survives continual adaptation.

The radial view makes hierarchy retention visible. Figure 6 shows a representative checkpoint trajectory alongside the five-seed metrics. Vanilla contracts the depth rings and disrupts branch layout, whereas both HMCL variants stay close to the task-end geometry. Their smaller radial and pairwise-distance errors indicate preservation of both depth organization and relations among concepts. Because the views are aligned by a global spatial rotation, this contrast is not simply a diference in drawing orientation. The visualization complements the hierarchy metrics in Table 3: the former reveals changes in representation geometry, whereas the latter evaluates semantic ancestry in the final predictions. Appendix H.2 reports modality-specific norm distributions over all 15 old tasks.

## 7.5 Representation-Space Case Study (RQ3)

Construction of the Concept Set. To complement the quantitative answer to RQ3, we study hierarchy retention through image traversals on Flickr30K. From the original captions, nouns and adjectives are collected to form a text vocabulary with diferent levels of semantic abstraction. Appendix G.3 details how the concept set is prepared.

Traversal Protocol. For each image, we trace its embedding toward the manifold origin [ROOT], which represents the generic concept. We sample 20 locations in the origin’s tangent space, map them back to the Lorentz hyperboloid, and retrieve the nearest text representation at each location.

Qualitative Finding. Figure 7 shows a recurring contrast in the displayed examples. HMCL maintains local semantic continuity along the geodesic—for example, hat → fashion → style and relaxation → vacation → peace—whereas Vanilla introduces abrupt of-path transitions such as volleyball → interest and aloof → sunken. Because the HMCL paths become progressively more generic before reaching [ROOT], the result suggests preservation of intermediate abstraction ordering rather than only convergence to the shared endpoint. This qualitative evidence complements the WordNet and radial-geometry results in Section 7.4, thereby providing a representation-level answer to RQ3. Appendix G.3 provides additional Flickr30K and COCO traversals.

![](images/0858fb08f6af1ee47493b01a3e05320bfd82fe6ef15efdc2f0901ee4d1db262c.jpg)

![](images/773930caac283a8e5c068f322173f4a37fadf1ed4268efb6a0a327e6ab2b7f21.jpg)

![](images/ddcc1dc6fa6db212e2537396ca3031724a3da3306aa432f686cedba97b0ac650.jpg)

<table><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>hat</td><td>volleyball</td></tr><tr><td>fashion</td><td>interest</td></tr><tr><td>style</td><td>fashion</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr></table>

![](images/88b71cf98e698ea473c54e7cc5b09d27550f7ae59895a9d5d6ca246c3051b85b.jpg)

<table><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>defeat</td><td>workout</td></tr><tr><td>workout</td><td>pilates</td></tr><tr><td>peace</td><td>race</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr></table>

Figure 7. Hierarchical concept retrieval along image-to-origin geodesics. For each Flickr30K example, we move from the image representation toward the manifold origin [ROOT] and display the nearest text concept along the resulting specific-to-generic path. HMCL gives more coherent sequences in these examples, such as hat → fashion → style, while Vanilla more often shifts to less related concepts after sequential training.

## 7.6 Ablation Study (RQ4)

To answer RQ4, we isolate where the geometric correction is applied and whether task anchoring is added, while keeping the HyCoCLIP-B stream and evaluation metrics fixed.

Table 4 shows that all corrected configurations outperform Vanilla in the Overall score, BWT, and ImageNet accuracy. The consistent advantage of Post over Pre for both MR and CA shows that the constraint is most efective when applied to AdamW’s realized displacement rather than to the raw gradient. Pullback then complements the geometric correction by limiting cumulative within-task deviation, improving retention without removing the flexibility needed for the current task. Post+Pullback CA attains the highest Overall score, whereas Post+Pullback MR gives the strongest BWT and ImageNet retention.

The placement comparison separates optimizer alignment from the choice of admissible representative. Both MR and CA benefit from correcting the realized displacement,

Table 4. Focused update-placement ablation on the 16-task HyCoCLIP-B stream (five-seed mean ± sample standard deviation). Each placement reports both MR and CA. Pre and Post apply the correction to the raw gradient and realized AdamW displacement, respectively; Post+Pullback adds task anchoring.
<table><tr><td>Update</td><td>Corr.</td><td>Overall ↑ BWT↑</td><td>ImageNet ↑</td></tr><tr><td>Vanilla</td><td>一</td><td>43.618 ± 0.152 −4.186 ± 0.087</td><td>39.404 ± 0.058</td></tr><tr><td>Pre</td><td>MR CA</td><td>46.397 ± 0.124 47.172 ± 0.129</td><td>−2.907 ± 0.070 41.303 ± 0.095 −2.932 ± 0.058 40.871 ± 0.007</td></tr><tr><td>Post</td><td>MR CA</td><td>47.780 ± 0.106 47.785 ± 0.102</td><td>−1.548 ± 0.073 43.760 ± 0.032 −1.532 ± 0.07543.759 ± 0.043</td></tr><tr><td>Post + Pullback</td><td>MR CA</td><td>48.053 ± 0.072 −0.239 ± 0.057 46.257 ± 0.033 48.566 ± 0.024 −0.456 ± 0.05145.585 ± 0.052</td><td></td></tr></table>

consistent with the fact that AdamW’s momentum and adaptive scaling can change the direction proposed by the raw gradient. Task anchoring addresses a diferent source of deviation: locally corrected updates can still accumulate change over a task. The further retention gains with Pullback therefore support its complementary role within the integrated update, rather than an alternative to geometric correction.

The MR ablation uses stronger task anchoring than its main-table configuration. As detailed in Appendix H.8, this favors retention, whereas the main-table setting favors the Overall score; the configurations therefore represent diferent stability–plasticity trade-ofs within the same update.

## 8 Conclusion

We presented a continual-learning formulation for hyperbolic multimodal representations and characterized the updates that retain within-modal relations, cross-modal alignment, and semantic hierarchy. HMCL translates these representation-space conditions into a shared admissible family with two complementary realizations: CA permits a common rotation, whereas MR selects the identity representative. Across backbones, both variants improve final performance and retention over the external baselines; CA achieves the highest Overall scores, while MR gives the best ImageNet–WordNet hierarchy scores. The stage-wise, drift, modality-extension, and hierarchy analyses show that these gains coincide with more stable radial and relational structure rather than reflecting one aggregate metric alone. The component study further shows that geometric correction and task anchoring address complementary sources of change. Together, the results support shared geometric admissibility as a practical stability–plasticity principle for continual adaptation in non-Euclidean representation spaces.

## References

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.

[2] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in CVPR, 2020, pp. 9729–9738.

[3] D. Yu, X. Zhang, Y. Chen, A. Liu, Y. Zhang, P. S. Yu, and I. King, “Recent advances of multimodal continual learning: A comprehensive survey,” arXiv preprint arXiv:2410.05352, 2024.

[4] J. Han, R. Zhang, W. Shao, P. Gao, P. Xu, H. Xiao, K. Zhang, C. Liu, S. Wen, Z. Guo et al., “ImageBind-LLM: Multi-modality instruction tuning,” arXiv preprint arXiv:2309.03905, 2023.

[5] S. Chen, H. Li, Q. Wang, Z. Zhao, M. Sun, X. Zhu, and J. Liu, “VAST: A vision-audio-subtitle-text omni-modality foundation model and dataset,” NeurIPS, vol. 36, pp. 72 842–72 866, 2023.

[6] X. Liu, X. Xia, S.-K. Ng, and T.-S. Chua, “Principled multimodal representation learning,” arXiv preprint arXiv:2507.17343, 2025.

[7] K. Desai, M. Nickel, T. Rajpurohit, J. Johnson, and S. R. Vedantam, “Hyperbolic image-text representations,” in International Conference on Machine Learning. PMLR, 2023, pp. 7694–7731.

[8] A. Pal, M. van Spengler, G. M. D. di Melendugno, A. Flaborea, F. Galasso, and P. Mettes, “Compositional entailment learning for hyperbolic vision-language models,” in The Thirteenth International Conference on Learning Representations, 2025.

[9] P. Mandica, L. Franco, K. Kallidromitis, S. Petryk, and F. Galasso, “Hyperbolic learning with multimodal large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 382–398.

[10] W. Kim, S. Chun, T. Kim, D. Han, and S. Yun, “Hype: Hyperbolic entailment filtering for underspecified images and texts,” in European Conference on Computer Vision. Springer, 2024, pp. 247–265.

[11] J. Liu, M. Shen, X. Liu, R. Ying, M. Yang, T.-S. Chua, and I. King, “Hyperbolic multimodal continual learning,” in International Conference on Machine Learning, 2026.

[12] D. Krioukov, F. Papadopoulos, M. Kitsak, A. Vahdat, and M. Boguná, “Hyperbolic geometry of complex networks,” Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, vol. 82, no. 3, p. 036106, 2010.

[13] M. Nickel and D. Kiela, “Poincaré embeddings for learning hierarchical representations,” Advances in neural information processing systems, vol. 30, 2017.

[14] ——, “Learning continuous hierarchies in the Lorentz model of hyperbolic geometry,” in International Conference on Machine Learning. PMLR, 2018, pp. 3779–3788.

[15] O. Ganea, G. Bécigneul, and T. Hofmann, “Hyperbolic neural networks,” Advances in neural information processing systems, vol. 31, 2018.

[16] R. Shimizu, Y. Mukuta, and T. Harada, “Hyperbolic neural networks++,” arXiv preprint arXiv:2006.08210, 2020.

[17] I. Chami, Z. Ying, C. Ré, and J. Leskovec, “Hyperbolic graph convolutional neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[18] M. Yang, M. Zhou, T. Zhang, J. Liu, Z. Li, L. Pan, H. Xiong, and I. King, “Hyperbolic graph neural networks: A review of methods and applications,” arXiv preprint arXiv:2202.13852, 2022.

[19] M. Yang, H. Verma, D. C. Zhang, J. Liu, I. King, and R. Ying, “Hypformer: Exploring eficient transformer fully in hyperbolic space,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 3770–3781.

[20] P. Mettes, M. Ghadimi Atigh, M. Keller-Ressel, J. Gu, and S. Yeung, “Hyperbolic deep learning in computer vision: A survey,” International Journal ofComputer Vision, vol. 132, no. 9, pp. 3484–3508, 2024.

[21] J. Liu, M. Yang, and I. King, “Hyperbolic learning for structured data, knowledge, and memory: A tutorial,” in Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2026, pp. 13 351–13 355.

[22] N. He, J. Liu, B. Zhang, N. Bui, A. Maatouk, M. Yang, I. King, M. Weber, and R. Ying, “Position: Beyond Euclidean– foundation models should embrace Non-Euclidean geometries,” in Proceedings of the Fourth Learning on Graphs Conference, ser. Proceedings of Machine Learning Research, vol. 338, 2025.

[23] J. Liu, M. Yang, M. Zhou, S. Feng, and P. Fournier-Viger, “Enhancing hyperbolic graph embeddings via contrastive learning,” NeurIPS Workshop on Self-Supervised Learning: Theory and Practice, 2022.

[24] Y. Zhang, H. Zhu, M. Yang, J. Liu, R. Ying, I. King, and P. Koniusz, “Understanding and mitigating hyperbolic dimensional collapse in graph contrastive learning,” in Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025, pp. 1984–1995.

[25] Y. Fu, J. Li, J. Liu, Q. Xing, Q. Wang, and I. King, “HC-GLAD: Dual hyperbolic contrastive learning for unsupervised graph-level anomaly detection,” Neural Networks, p. 109009, 2026.

[26] M. Yang, J. Liu, I. King, and R. Ying, “UHCone: Universal hyperbolic cone for implicit hierarchical learning,” in ICML Workshop on Geometry-grounded Representation Learning and Generative Modeling, 2024.

[27] M. Yang, M. Zhou, J. Liu, D. Lian, and I. King, “HRCF: Enhancing collaborative filtering via hyperbolic geometric regularization,” in Proceedings ofthe ACM Web Conference 2022, 2022, pp. 2462–2471.

[28] M. Yang, Z. Li, M. Zhou, J. Liu, and I. King, “HICF: Hyperbolic informative collaborative filtering,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 2212–2221.

[29] Z. Qiu, J. Liu, Y. Chen, and I. King, “HiHPQ: Hierarchical hyperbolic product quantization for unsupervised image retrieval,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp. 4614–4622.

[30] V. Khrulkov, L. Mirvakhabova, E. Ustinova, I. Oseledets, and V. Lempitsky, “Hyperbolic image embeddings,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 6417–6427.

[31] S. Liu, J. Chen, L. Pan, C.-W. Ngo, T.-S. Chua, and Y.-G. Jiang, “Hyperbolic visual embedding learning for zero-shot recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 9270–9278.

[32] S. Ge, S. Mishra, S. Kornblith, C.-L. Li, and D. Jacobs, “Hyperbolic contrastive learning for visual representations beyond objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6840–6849.

[33] G. Moreira, M. Marques, J. P. Costeira, and A. Hauptmann, “Hyperbolic vs Euclidean embeddings in few-shot learning: Two sides of the same coin,” in Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, 2024, pp. 2071–2079.

[34] S. Ibrahimi, M. G. Atigh, N. Van Noord, P. Mettes, and M. Worring, “Intriguing properties of hyperbolic embeddings in vision-language models,” Transactions on Machine Learning Research, 2024.

[35] S. Ramasinghe, V. Shevchenko, G. Avraham, and A. Thalaiyasingam, “Accept the modality gap: An exploration in the hyperbolic space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 263–27 272.

[36] F. Kong, Y. Chen, J. Cai, and D. Modolo, “Hyperbolic learning with synthetic captions for open-world detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 762–16 771.

[37] Z. Peng, Z. Xu, Z. Zeng, C. Wen, Y. Huang, M. Yang, F. Tang, and W. Shen, “Understanding fine-tuning CLIP for open-vocabulary semantic segmentation in hyperbolic space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 4562–4572.

[38] M. Yang, J. Chen, J. Tao, Y. Zhang, J. Liu, J. Zhang, Q. Ma, H. Verma, R. Zhang, M. Zhou, I. King, and R. Ying, “Low-rank adaptation for foundation models: A comprehensive review,” arXiv preprint arXiv:2501.00365, 2025.

[39] M. Yang, R. B, A. Feng, B. Xiong, J. Liu, I. King, and R. Ying, “Hyperbolic fine-tuning for large language models,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[40] T. Poppi, T. Kasarla, P. Mettes, L. Baraldi, and R. Cucchiara, “Hyperbolic safety-aware vision-language models,” in Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2025, pp. 4222–4232.

[41] À. P. Vidal, K. Nasrollahi, T. B. Moeslund, and S. Escalera, “Machine unlearning in hyperbolic vs. Euclidean multimodal contrastive learning: Adapting alignment calibration to MERU,” in Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2025, pp. 1644–1653.

[42] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning,” Advances in neural information processing systems, vol. 30, 2017.

[43] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in neural information processing systems, vol. 33, pp. 15 920–15 930, 2020.

[44] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy ofsciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[45] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[46] A. Mallya and S. Lazebnik, “PackNet: Adding multiple tasks to a single network by iterative pruning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2018, pp. 7765–7773.

[47] A. Mallya, D. Davis, and S. Lazebnik, “Piggyback: Adapting a single network to multiple tasks by learning to mask weights,” in Proceedings ofthe European conference on computer vision (ECCV), 2018, pp. 67–82.

[48] Z. Ni, L. Wei, S. Tang, Y. Zhuang, and Q. Tian, “Continual vision-language representation learning with of-diagonal information,” in International Conference on Machine Learning. PMLR, 2023, pp. 26 129–26 149.

[49] J. Li, Y. Li, Y. Fu, J. Liu, Y. Liu, M. Yang, and I. King, “CLIP-powered domain generalization and domain adaptation: A comprehensive survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 5, pp. 5405–5424, 2026.

[50] Z. Zheng, M. Ma, K. Wang, Z. Qin, X. Yue, and Y. You, “Preventing zero-shot transfer degradation in continual learning of vision-language models,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2023, pp. 19 125–19 136.

[51] J. Yu, Y. Zhuge, L. Zhang, P. Hu, D. Wang, H. Lu, and Y. He, “Boosting continual learning of vision-language models via mixture-of-experts adapters,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 219–23 230.

[52] A. Bian, W. Li, H. Yuan, M. Wang, Z. Zhao, A. Lu, P. Ji, T. Feng et al., “Make continual learning stronger via C-FLAT,” Advances in Neural Information Processing Systems, vol. 37, pp. 7608–7630, 2024.

[53] J. Liu, Z. Qiu, Z. Li, Q. Dai, W. Yu, J. Zhu, M. Hu, M. Yang, T.-S. Chua, and I. King, “A survey of personalized large language models: Progress and future directions,” arXiv preprint arXiv:2502.11528, 2025.

[54] J. Liu, W. Yu, Q. Dai, Z. Li, J. Zhu, M. Yang, T.-S. Chua, and I. King, “PerFit: Exploring personalization shifts in representation space of LLMs,” in International Conference on Learning Representations, 2026.

[55] J. Liu, W. Yu, Z. Qiu, M. Yang, and I. King, “Memory has geometry: Non-uniform geometric memory for long-horizon personalized AI,” arXiv preprint arXiv:2609.17969, 2026.

[56] J. Liu, R. S. B. B, X. Fu, M. Yang, W. Zhang, R. Ying, and I. King, “FlatLand: Personalized graph federated learning via tailored Lorentz space,” in International Conference on Machine Learning, 2026.

[57] M. Ayoughi, M. G. Atigh, M. M. Derakhshani, C. G. Snoek, P. Mettes, and P. Groth, “Continual hyperbolic learning of instances and classes,” arXiv preprint arXiv:2506.10710, 2025.

[58] T. Doan, S. Behpour, X. Li, W. He, L. Gou, and L. Ren, “A streamlined approach to multimodal few-shot class incremental learning for fine-grained datasets,” arXiv preprint arXiv:2403.06295, 2024.

[59] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[60] G. Bécigneul and O. Ganea, “Riemannian adaptive optimization methods,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

[61] G. Mishne, Z. Wan, Y. Wang, and S. Yang, “The numerical stability of hyperbolic representation learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 24 925–24 949.

[62] W. Chen, X. Han, Y. Lin, H. Zhao, Z. Liu, P. Li, M. Sun, and J. Zhou, “Fully hyperbolic neural networks,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 5672–5686.

[63] X. Liu, X. Xia, S.-K. Ng, and T.-S. Chua, “Continual multimodal contrastive learning,” arXiv preprint arXiv:2503.14963, 2025.

[64] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in European Conference on Computer Vision. Springer, 2014, pp. 740–755.

[65] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions,” Transactions of the association for computational linguistics, vol. 2, pp. 67–78, 2014.

[66] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[67] G. A. Miller, “WordNet: A lexical database for English,” Communications of the ACM, vol. 38, no. 11, pp. 39–41, 1995.

[68] A. Kosmopoulos, I. Partalas, E. Gaussier, G. Paliouras, and I. Androutsopoulos, “Evaluation measures for hierarchical classification: A unified view and novel approaches,” Data Mining and Knowledge Discovery, vol. 29, no. 3, pp. 820–865, 2015.

[69] R. Sarkar, “Low distortion Delaunay embedding of trees in hyperbolic plane,” in International symposium on graph drawing. Springer, 2011, pp. 355–366.

[70] T.-Y. Lam, Introduction to quadraticforms overfields. American Mathematical Soc., 2005, vol. 67.

[71] V. Moretti, “The interplay of the polar decomposition theorem and the Lorentz group,” arXiv preprint math-ph/0211047, 2002.

[72] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009.

[73] L. Fei-Fei, R. Fergus, and P. Perona, “Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 1, 2004, pp. 178–178.

[74] L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101 – mining discriminative components with random forests,” in European Conference on Computer Vision. Springer, 2014, pp. 446–461.

[75] M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in 2008 Sixth Indian Conference on Computer Vision, Graphics and Image Processing, 2008, pp. 722–729.

[76] P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217–2226, 2019.

[77] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, “Describing textures in the wild,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 3606–3613.

[78] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi, “Fine-grained visual classification of aircraft,” arXiv preprint arXiv:1306.5151, 2013.

[79] Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner, “Gradient-based learning applied to document recognition,” Proceedings ofthe IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[80] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, and M. Welling, “Rotation equivariant CNNs for digital pathology,” arXiv preprint arXiv:1806.03962, 2018.

[81] J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick, and R. Girshick, “CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2901–2910.

[82] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in Proceedings ofthe 2013 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2013, pp. 1631–1642.

## Appendix

Supplementary derivations, implementation details, and extended results

Contents   
A Notation Table . 23   
B Additional Preliminaries 25   
B.1 Why Hyperbolic Geometry Represents Hierarchies 25   
B.2 Formal Description of the Lorentz Model 25   
B.3 Lorentz Neural Layers 26   
C Rationale for the Preservation Objectives . 27   
D Proofs for the Geometric Characterization 29   
D.1 Shared Isometry Induced by Gram Preservation 29   
D.2 Removing the Boost Component 31   
D.3 First-Order Hierarchy Preservation 31   
E Derivations for the HMCL Update 33   
E.1 Proof of Proposition 1 33   
E.2 Derivation of Corollary 2 34   
E.3 Proof of Corollary 3 34   
F Optimizer-Consistent HMCL: Derivations 36   
F.1 A Concrete Adaptive-Optimizer Mismatch 36   
F.2 Lorentz Strain of a Realized Optimizer Candidate 36   
F.3 Closest Projection of the Realized Step 38   
F.4 Task-Anchored Feasible Contraction 38   
F.5 Residual Drift under Low-Rank PCA 39   
F.6 Pseudocode, Optimizer State, and Complexity 40   
G Additional Implementation Details 41   
G.1 Dataset Details and Task Stream 41   
G.2 Task-Order Robustness 41   
G.3 Supplementary Materials for Case Study 42   
H Extended Experimental Protocol and Results 45   
H.1 ImageNet–WordNet Protocol and Per-Task Results 45   
H.2 Modality-Specific Radial Contraction . 45   
H.3 Stage-Wise Stability–Plasticity Diagnostics 46   
H.4 Fine-Grained Retrieval on the Modality-Extended Stream 46   
H.5 Reproducibility Notes . 47   
H.6 Robustness to AdamW Weight Decay 48   
H.7 Clean Single-Stage Training Time . 49   
H.8 Pullback Hyperparameter Analysis 49

## A Notation Table

Table 5 collects the symbols and conventions used in the main paper and this supplement.

Table 5. Summary of the notation used in this work.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td colspan="2">Hyperbolic geometry and the Lorentz model</td></tr><tr><td> $d$ </td><td>Intrinsic hyperbolic dimension; the Lorentz ambient space has dimension  $d + 1 .$ </td></tr><tr><td> $K$ </td><td>Squared hyperboloid radius  $( K > 0 ) ;$  the sectional curvature  $\mathrm { i s } - 1 / K .$ </td></tr><tr><td> $\mathbb { H } _ { K } ^ { d }$ </td><td>d-dimensional hyperbolic space with curvature  $- 1 / K .$ </td></tr><tr><td> $\mathbf { G }$ </td><td>Minkowski metric tensor,  $\hat { \mathbf { G } } = \operatorname { d i a g } ( - 1 , 1 , \ldots , 1 ) \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) }$ </td></tr><tr><td> $\langle \cdot , \cdot \rangle _ { \mathcal { L } }$ </td><td>Lorentzian inner product  $\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } = \mathbf { x } ^ { \top } \mathbf { G } \mathbf { y } .$ </td></tr><tr><td> $d _ { \mathcal { L } } ^ { K } ( \mathbf { x } , \mathbf { y } )$ </td><td>Geodesic distance,  $\sqrt { K }$  arcosh  $( - \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } / K )$ </td></tr><tr><td> $\mathbf { o }$ </td><td>Origin of the Lorentz model,  $( \sqrt { K } , 0 , \ldots , 0 ) ^ { \intercal } \in \mathbb { H } _ { K } ^ { d } .$ </td></tr><tr><td> $x _ { 0 } , \mathbf { X } [ 1 { : } d ]$ </td><td>Time-like coordinate and  $d$  space-like coordinates of x.</td></tr><tr><td> $\mathrm { S O ^ { + } } \bar { ( 1 , d ) } , \mathrm { S O } ( d )$ </td><td>Proper orthochronous Lorentz group and spatial rotation group.</td></tr><tr><td colspan="2">Multimodal learning</td></tr><tr><td></td><td>Number of modalities.</td></tr><tr><td>M</td><td>Modality indices,  $m , m ^ { \prime } \in \{ 1 , 2 , \ldots , M \}$ </td></tr><tr><td> $m , m ^ { \prime }$ </td><td>Number of paired samples in the current context.</td></tr><tr><td> $N$   $\chi m$ </td><td>Data space of modality m.</td></tr><tr><td> $\chi ^ { m , m ^ { \prime } }$ </td><td>Paired dataset</td></tr><tr><td> ${ \bf x } _ { i } ^ { m } , { \bf z } _ { i } ^ { m }$ </td><td> $\{ ( \mathbf { x } _ { i } ^ { m } , \mathbf { x } _ { i } ^ { m ^ { \prime } } ) \} _ { i = 1 } ^ { N } .$  The i-th input and its hyperbolic embedding for modality</td></tr><tr><td colspan="2">Continual learning framework</td></tr><tr><td></td><td></td></tr><tr><td>t  $J$ </td><td>Task or stage index.</td></tr><tr><td> $k$ </td><td>Total number of optimizer steps within a task. Within-task optimizer-step index,  $k \in \{ 0 , \ldots , J - 1 \}$ </td></tr><tr><td> $T _ { m } ^ { ( t ) }$ </td><td>Lorentz transformation for modality m at task</td></tr><tr><td> ${ \bf W } ^ { m , t }$ </td><td> $t .$  Final Lorentz-head parameters at task t.</td></tr><tr><td> $\mathbf { W } _ { k } ^ { m , t } = \big [ \mathbf { w } _ { k , 0 } ^ { m , t } \ \mathbf { W } _ { k , s } ^ { m , t } \big ]$ </td><td>Parameters after step  $k ,$  partitioned into the time-like input column and spatial input block;  $\mathbf { W } _ { 0 } ^ { m , t } =$   $\mathbf { W } ^ { m , t - 1 }$  and  $\mathbf { W } _ { J } ^ { m , t } = \mathbf { W } ^ { m , t }$ </td></tr><tr><td> $\widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t }$ </td><td>Optimizer-proposed parameter candidate before HMCL correction.</td></tr><tr><td> $\mathbf { g } _ { k } ^ { m , t }$ </td><td>Raw gradient at step k.</td></tr><tr><td> $X _ { t } ^ { m }$ </td><td>Input data for task t and modality  $m .$ </td></tr><tr><td> $\mathbf { Z } _ { a } ^ { m , * }$ </td><td>Frozen pretrained Lorentz representations of task-q data from modality</td></tr><tr><td> $\mathbf { Z } _ { q } ^ { \dot { m } , t }$ </td><td>The same data after the stage-t Lorentz head.</td></tr><tr><td> $f ( \mathbf { W } ; \cdot )$ </td><td>Lorentz layer in (22).</td></tr><tr><td colspan="2">Similarity measures and objectives</td></tr><tr><td> $\mathbf { S } ^ { m  m ^ { \prime } }$ </td><td>Similarity matrix from modality m to  $m ^ { \prime } ,$  with  $\mathbf { S } ^ { m  m ^ { \prime } }$ </td></tr><tr><td> $S _ { i j } ^ { m  m ^ { \prime } }$ </td><td>Entry  $( i , j )$  of  $\mathbf { S } ^ { m  m ^ { \prime } }$  , equal  $\mathrm { t o } - d _ { \mathcal { L } } ^ { K } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { j } ^ { m ^ { \prime } } ) .$ </td></tr><tr><td>T</td><td>Temperature parameter for contrastive learning.</td></tr><tr><td> $\scriptstyle { \mathcal { L } } _ { m \to m ^ { \prime } }$ </td><td>Unidirectional contrastive loss from modality m to m'.</td></tr><tr><td></td><td>Symmetric contrastive loss  $\begin{array} { r } { \frac { 1 } { 2 } ( \mathcal { L } _ { m  m ^ { \prime } } + \mathcal { L } _ { m ^ { \prime }  m } ) . } \end{array}$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { c o n t r a s t } }$ </td><td>Entailment-cone aperture at embedding z.</td></tr><tr><td>aper(z)  $\operatorname { e x t } ( \mathbf { z } , \mathbf { z } ^ { \prime } )$ </td><td>Exterior angle between embeddings z and  $\mathbf { z } ^ { \prime } .$ </td></tr><tr><td> $\pmb { K }$ </td><td>Boundary constant for entailment cones (typically  $\kappa = 0 . 1 )$ </td></tr><tr><td> $\scriptstyle { \mathcal { L } } _ { \mathrm { e n t a i l } }$ </td><td>Entailment loss for hierarchical relationships.</td></tr><tr><td colspan="2">Geometric analysis and parameter updates</td></tr><tr><td></td><td>General Lorentz transformation,  $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ </td></tr><tr><td>L</td><td>Spatial Lorentz rotation diag(1, R).</td></tr><tr><td>RR</td><td>Spatial rotation matrix in  ${ \mathrm { S O } } ( d )$ </td></tr><tr><td> $\delta ^ { m } = \big [ \delta _ { 0 } ^ { m } \delta _ { s } ^ { m } \big ]$ </td><td>Candidate parameter displacement, partitioned by input coordinate.</td></tr><tr><td> $\delta _ { k } ^ { \mathrm { o p t } , m , t }$ </td><td>Optimizer-proposed displacement  $\widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { k } ^ { m , t }$  before HMCL correction.</td></tr><tr><td> $\delta _ { k } ^ { \star , m , t }$ </td><td>Closest admissible displacement obtained from  $\delta _ { k } ^ { \mathrm { o p t } , m , t } .$ </td></tr><tr><td> $\mathbf { A } _ { k } ^ { m }$ </td><td>Cumulative displacement  $\mathbf { W } _ { k } ^ { m , t } - \mathbf { W } _ { 0 } ^ { m , t } .$ </td></tr><tr><td> $\boldsymbol { \mathcal { E } } _ { \mathcal { L } , t - 1 }$ </td><td>Defect from a shared first-order Lorentz rotation on protected representations</td></tr><tr><td> $\mathbf { \mathcal { A } } _ { t - 1 } , \mathbf { \mathcal { P } } _ { t - 1 } ^ { \mathcal { L } }$ </td><td>Zero-defect admissible family and its Frobenius closest-point map.</td></tr><tr><td> $\mathbf { V } _ { t - 1 } , \mathbf { P } _ { t - 1 }$ </td><td>Protected-subspace basis and projector, with  $\mathbf { P } _ { t - 1 } = \mathbf { V } _ { t - 1 } \mathbf { V } _ { t - 1 } ^ { \top }$ </td></tr><tr><td> $\beta$ </td><td>Pullback coefficient toward the task-start parameters.</td></tr></table>

## B Additional Preliminaries

## B.1 Why Hyperbolic Geometry Represents Hierarchies

Hyperbolic space is a homogeneous geometry with constant negative curvature. Its volume expands exponentially with radius: in � dimensions and curvature −1, a radius-� ball has volume

$$
V _ { \mathbb { H } ^ { n } } ( r ) \propto \int _ { 0 } ^ { r } \sinh ^ { n - 1 } ( t ) d t ,
$$

which is asymptotically proportional to $e ^ { ( n - 1 ) r }$ . Euclidean volume, in contrast, grows only as $r ^ { n }$ . In the twodimensional case, for instance, the hyperbolic disk area is $2 \pi ( \cosh r - 1 )$ rather than the Euclidean value $\pi r ^ { 2 }$ . This exponential capacity closely matches the growth of tree-structured data, which is why hyperbolic embeddings can represent taxonomies, trees, and knowledge graphs with low distortion [13, 69].

## B.2 Formal Description of the Lorentz Model

The Lorentz model embeds a �-dimensional hyperbolic manifold as one sheet of a hyperboloid in $( d + 1 )$ -dimensional Minkowski space.

Definition 1 (Minkowski Space and Metric). Minkowski space is $\mathbb { R } ^ { d + 1 }$ equipped with the indefinite metric

$$
\mathbf { G } = \mathrm { d i a g } ( - 1 , 1 , \ldots , 1 ) \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) } .
$$

Here $K > 0$ denotes the squared hyperboloid radius used below, so the corresponding sectional curvature $i s - 1 / K$ Its signature $( - , + , \ldots , + )$ distinguishes one time-like directionfrom � space-like directions.

Definition 2 (Lorentzian Inner Product). For $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d + 1 }$ , their Lorentzian (Minkowski) inner product is

$$
\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } = \mathbf { x } ^ { \top } \mathbf { G } \mathbf { y } = - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } .
$$

Definition 3 (Lorentz Manifold). The Lorentz realization of �-dimensional hyperbolic space with curvature $- 1 / K$ is

$$
\mathbb { H } _ { K } ^ { d } = \left\{ \mathbf { x } \in \mathbb { R } ^ { d + 1 } : \mathbf { x } ^ { \top } \mathbf { G } \mathbf { x } = - K , x _ { 0 } > 0 \right\} .
$$

We write $\mathcal { L } _ { K } ^ { d } = ( \mathbb { H } _ { K } ^ { d } , { \bf G } )$ when referring to the manifold together with its metric.

Remark 1. Expanding the manifold constraint gives

$$
- x _ { 0 } ^ { 2 } + \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } = - K , \qquad x _ { 0 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } = K .
$$

This equation describes a two-sheeted hyperboloid; $x _ { 0 } > 0$ selects its connected upper sheet. For any $\mathbf { x } \in \mathbb { H } _ { K } ^ { d } .$

$$
x _ { 0 } = \sqrt { K + \| \mathbf { x } _ { [ 1 : d ] } \| ^ { 2 } } \geq \sqrt { K } ,
$$

where $\mathbf { x } _ { [ 1 : d ] } = ( x _ { 1 } , \ldots , x _ { d } ) ^ { \top }$ contains the space-like coordinates.

Definition 4 (Lorentzian Distance). The length of the geodesic between x, $\mathbf { y } \in \mathcal { L } _ { K } ^ { d }$ is

$$
d _ { \mathcal { L } } ^ { K } ( \mathbf { x } , \mathbf { y } ) = \sqrt { K } \operatorname { a r c o s h } \left( - \frac { \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } } { K } \right) = \sqrt { K } \operatorname { a r c o s h } \left( - \frac { \mathbf { x } ^ { \top } \mathbf { G } \mathbf { y } } { K } \right) .\tag{21}
$$

Remark 2. For points on the upper hyperboloid, $\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } \leq - K ,$ , with equality ifand only $i f \mathbf { X } = \mathbf { y } .$ . The argument of arcosh is therefore at least one, so (21) is well defined. Substituting $\mathbf x = \mathbf y$ gives zero distance.

Remark 3 (Relation to Ambient Euclidean Distance). Far from the origin, geodesic distance increases approximately logarithmically with ambient Euclidean distance. This growth allows hyperbolic space to represent tree-like structures with low distortion.

The linear symmetries of the Lorentz model form a group of transformations that leave the Lorentzian inner product unchanged.

Definition 5 (Proper Orthochronous Lorentz Group). The proper orthochronous Lorentz group is

$$
\begin{array} { r } { \mathrm { S O } ^ { + } ( 1 , d ) = \left\{ \mathbf { L } \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) } : \mathbf { L } ^ { \top } \mathbf { G } \mathbf { L } = \mathbf { G } , \operatorname* { d e t } ( \mathbf { L } ) = 1 , L _ { 0 0 } \geq 1 \right\} . } \end{array}
$$

Remark 4. The metric constraint makes L an isometry, the determinant condition excludes orientation-reversing reflections, and $L _ { 0 0 } \ge 1$ preserves the time orientation ofthe upper hyperboloid. The group can also map any point on $\mathbb { H } _ { K } ^ { d }$ to any other point: for $\mathbf { x } , \mathbf { y } \in \mathbb { H } _ { K } ^ { d } ,$ , some $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ satisfies $\mathbf { L x } = \mathbf { y }$

## B.3 Lorentz Neural Layers

Definition 6 (Lorentz Transformation Layer [62]). Let $\mathbf { z } \in \mathbb { R } ^ { d + 1 }$ be a Lorentz representation, with time-like component $z _ { 0 } > 0$ and spatial block $\mathbf { z } _ { s } \in \mathbb { R } ^ { d }$ . A Lorentz transformation layer is defined by

$$
f ( \mathbf { W } ; \mathbf { z } ) = \left[ \sqrt { K + \| \mathbf { W } \mathbf { z } \| _ { 2 } ^ { 2 } } \right] , \qquad \mathbf { W } \in \mathbb { R } ^ { d \times ( d + 1 ) } .\tag{22}
$$

Remark 5. The time-like output in (22) is reconstructed from the spatial output so that the result satisfies the Lorentz manifold constraint.

For the block-wise analysis, we partition the weights as

$$
\mathbf { W } = [ \mathbf { w } _ { 0 } \mathbf { W } _ { s } ] , \qquad \mathbf { w } _ { 0 } \in \mathbb { R } ^ { d \times 1 } , \quad \mathbf { W } _ { s } \in \mathbb { R } ^ { d \times d } ,\tag{23}
$$

where ${ \bf w } _ { 0 }$ multiplies the time-like input and ${ \bf W } _ { s }$ acts on the space-like input coordinates.

## C Rationale for the Preservation Objectives

Continual learning aims to incorporate the current task while retaining information acquired earlier. For hyperbolic representations, that information is represented by geometry: pairwise relations encode semantic similarity, while radial position encodes semantic specificity and hierarchy [7]. We therefore use the following three requirements for representations of earlier data:

(P1) Intra-modal preservation:

$$
\mathbf { Z } _ { t - 1 } ^ { m , t } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m , t } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } ) ^ { \top } , \quad \forall m \in \{ 1 , \ldots , M \} ;
$$

(P2) Inter-modal preservation:

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m ^ { \prime } , t } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } \mathbf { G } ( \mathbf { Z } _ { t - 1 } ^ { m ^ { \prime } , t - 1 } ) ^ { \top } , \quad \forall m \neq m ^ { \prime } ; } \end{array}
$$

(P3) Hierarchical preservation:

$$
\| ( \mathbf { z } _ { t - 1 } ^ { m , t } ) _ { [ 1 : d ] } \| = \| ( \mathbf { z } _ { t - 1 } ^ { m , t - 1 } ) _ { [ 1 : d ] } \| , \quad \forall m \in \{ 1 , \ldots , M \} .
$$

Table 6. Links between the desired semantic properties and the hyperbolic quantities that encode them.
<table><tr><td>Semantic Property</td><td>Geometric Quantity</td><td>Condition</td></tr><tr><td colspan="3">– Directly constrained</td></tr><tr><td>Semantic similarity Semantic specificity</td><td> $d _ { \mathcal { L } } ( \mathbf { x } , \mathbf { y } ) \propto \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } }$   $d _ { \mathcal { L } } ( \mathbf { x } , \mathbf { 0 } ) \propto \Vert \mathbf { x } _ { [ 1 : d ] } \Vert$ </td><td>(P1), (P2) (P3)</td></tr><tr><td colspan="3">– Implied consequence (Proposition 2)</td></tr><tr><td>Concept partial order</td><td> $\mathrm { a p e r } ( \mathbf { x } ) , \mathrm { e x t } ( \mathbf { x } , \mathbf { y } )$ </td><td>Implied by the conditions</td></tr></table>

Table 6 links each preservation goal to the geometric quantity that represents it. Conditions (P1) and (P2) fix Lorentzian inner products. Since the distance in (21) is a monotone function of these products, the associated semantic similarities remain unchanged. Condition (P3) fixes the spatial norm. On the hyperboloid,

$$
d _ { \mathcal { L } } ^ { K } ( \mathbf { x } , \mathbf { 0 } ) = \sqrt { K } \operatorname { a r c o s h } \left( \sqrt { 1 + \| \mathbf { x } _ { [ 1 : d ] } \| _ { 2 } ^ { 2 } / K } \right) .
$$

Preserving the spatial norm therefore keeps a concept at the same level of specificity.

Together, these invariants are also suficient to retain the entailment relation used to encode concept order.

Proposition 2 (Invariance of the Concept Partial Order). Let $\mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } \in \mathbb { H } _ { K } ^ { d }$ be hyperbolic embeddings. If $\| ( \mathbf { z } _ { i } ^ { m } ) _ { [ 1 : d ] } \| _ { 2 }$ and $\langle { \pmb z } _ { i } ^ { m } , { \pmb z } _ { i } ^ { m ^ { \prime } } \rangle _ { \mathcal { L } }$ do not change, then the entailment relation $\mathbf { z } _ { i } ^ { m } \overset {  } { = } \mathbf { z } _ { i } ^ { m ^ { \prime } }$ is unchanged as well.

Proof. Entailment holds when $\mathcal { L } _ { \mathrm { e n t a i l } } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) = 0$ , or equivalently

$$
\mathrm { e x t } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) \leq \mathrm { a p e r } ( \mathbf { z } _ { i } ^ { m } ) .
$$

The two angles are

$$
\mathrm { a p e r } ( \mathbf { z } _ { i } ^ { m } ) = \mathrm { s i n } ^ { - 1 } \left( \frac { 2 \kappa \sqrt { K } } { \| ( \mathbf { z } _ { i } ^ { m } ) _ { [ 1 : d ] } \| } \right) ,\tag{24}
$$

$$
\mathrm { e x t } ( \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } ) = \mathrm { c o s } ^ { - 1 } \left( \frac { ( \mathbf { z } _ { i } ^ { m ^ { \prime } } ) _ { 0 } + ( \mathbf { z } _ { i } ^ { m } ) _ { 0 } \langle \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } \rangle _ { \mathcal { L } } / K } { \| ( \mathbf { z } _ { i } ^ { m } ) _ { [ 1 : d ] } \| \sqrt { \left( \langle \mathbf { z } _ { i } ^ { m } , \mathbf { z } _ { i } ^ { m ^ { \prime } } \rangle _ { \mathcal { L } } / K \right) ^ { 2 } - 1 } } \right) .\tag{25}
$$

The aperture depends only on the spatial norm, while the exterior angle depends on that norm and the Lorentzian inner product. The time-like coordinate follows from the manifold constraint. If these quantities are preserved, neither side of the entailment inequality changes, so the corresponding partial-order relation is invariant. □

## D Proofs for the Geometric Characterization

Definition 7 (Joint Embedding Matrix). Consider � paired examples from task $t \in \mathcal T$ , evaluated at stage $s \in S$ for modalities $m _ { 1 }$ and $m _ { 2 } .$ . Given $\mathbf { \widetilde { Z } } _ { t } ^ { m , s } \in \mathbb { R } ^ { N \times ( d + 1 ) }$ with rows $\mathbf { z } _ { t , i } ^ { m , s } \in \mathbb { H } _ { K } ^ { d }$ , define

$$
\begin{array} { r } { \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } : = \binom { \mathbf { Z } _ { t } ^ { m _ { 1 } , s } } { \mathbf { Z } _ { t } ^ { m _ { 2 } , s } } \in \mathbb { R } ^ { 2 N \times ( d + 1 ) } . } \end{array}
$$

The ordering in this vertical concatenation retains the pairing between the two modalities.

Definition 8 (Extended Lorentz Gram Matrix). For $\mathbf { G } = \mathrm { d i a g } ( - 1 , 1 , \dots , 1 )$ , the extended Gram matrix of the joint embeddings is

$$
\mathbf { G } _ { t , \mathrm { e x t } } ^ { s } : = \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } \mathbf { G } ( \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } ) ^ { \top }\tag{26}
$$

$$
\begin{array} { r l } { \mathbf { \Xi } } & { = \binom { \mathbf { Z } _ { t } ^ { m _ { 1 } , s } \mathbf { G } ( \mathbf { Z } _ { t } ^ { m _ { 1 } , s } ) ^ { \top } } { \mathbf { Z } _ { t } ^ { m _ { 2 } , s } \mathbf { G } ( \mathbf { Z } _ { t } ^ { m _ { 1 } , s } ) ^ { \top } } \quad \mathbf { Z } _ { t } ^ { m _ { 1 } , s } \mathbf { G } ( \mathbf { Z } _ { t } ^ { m _ { 2 } , s } ) ^ { \top } \} } \\ & { = \binom { m _ { 2 } , s } { \mathbf { Z } _ { t } ^ { m _ { 2 } , s } \mathbf { G } ( \mathbf { Z } _ { t } ^ { m _ { 1 } , s } ) ^ { \top } } \quad \mathbf { Z } _ { t } ^ { m _ { 2 } , s } \mathbf { G } ( \mathbf { Z } _ { t } ^ { m _ { 2 } , s } ) ^ { \top } \Biggr ) . } \end{array}\tag{27}
$$

Its diagonal blocks contain within-modality Lorentzianproducts, while the of-diagonal blocks record the cross-modal geometry.

Definition 9 (Lorentz Boost). For $\mathbf { v } \in \mathbb { R } ^ { d }$ with $\left\| \mathbf { v } \right\| < 1$ and $\gamma = ( 1 - \| \mathbf { v } \| ^ { 2 } ) ^ { - 1 / 2 }$ , the boost associated with v is

$$
\mathbf { B } = \left[ \begin{array} { l l } { \gamma } & { - \gamma \mathbf { v } ^ { \top } } \\ { - \gamma \mathbf { v } } & { \mathbf { I } + \displaystyle \frac { \gamma ^ { 2 } } { 1 + \gamma } \mathbf { v } \mathbf { v } ^ { \top } } \end{array} \right] .
$$

It mixes the time-like coordinate with the direction v without independently rotating the spatial axes.

Definition 10 (Lorentz Rotation). A spatial Lorentz rotation has block form

$$
\mathbf { R } = \left[ { \begin{array} { c c } { 1 } & { \mathbf { 0 } ^ { \top } } \\ { \mathbf { 0 } } & { \widetilde { \mathbf { R } } } \end{array} } \right] , \qquad \widetilde { \mathbf { R } } ^ { \top } \widetilde { \mathbf { R } } = \mathbf { I } , \quad \operatorname* { d e t } ( \widetilde { \mathbf { R } } ) = 1 .
$$

Thus $\widetilde { \mathbf { R } } \in { \mathrm { S O } } ( d )$ rotates only the spatial coordinates.

## D.1 Shared Isometry Induced by Gram Preservation

Lemma 1 (Witt’s Extension Theorem [70]). Let $( \boldsymbol { \mathcal { V } } , \boldsymbol { Q } )$ be a non-degenerate quadratic space. Any isometry $\phi : \mathcal { U } \to \mathcal { V } ^ { \prime }$ between subspaces $\mathcal { U } , \mathcal { V } ^ { \prime } \subseteq \mathcal { V }$ extends to an isometry ofall $o f ^ { \mathcal { N } }$

Remark 6. In matrix notation, $i f \mathbf { Z } _ { 1 } \mathbf { Q } \mathbf { Z } _ { 1 } ^ { \top } = \mathbf { Z } _ { 2 } \mathbf { Q } \mathbf { Z } _ { 2 } ^ { \top }$ for a non-degenerate symmetric Q, then an element T of

$$
O ( \mathbf { Q } ) = \{ \mathbf { T } : \mathbf { T } ^ { \top } \mathbf { Q } \mathbf { T } = \mathbf { Q } \}
$$

maps thefirst set ofrows to the second, i.e., $\mathbf { Z } _ { 2 } = \mathbf { Z } _ { 1 } \mathbf { T } ^ { \top }$

Lemma 2 (Hyperboloid-Preserving Isometries). Le $\mathbf { L } \in O ( 1 , d )$ be an orientation-preserving isometry that maps the upper hyperboloid

$$
\mathbb { H } _ { K } ^ { d } = \{ { \bf x } : { \bf x } ^ { \top } { \bf G } { \bf x } = - K , x _ { 0 } > 0 \}
$$

onto itself. Then L $\in { \mathrm { S O } } ^ { + } ( 1 , d )$

Proof. Because the upper sheet is invariant, applying L to $\mathbf { 0 } = ( \sqrt { K } , 0 , \ldots , 0 ) ^ { \top }$ gives $( { \bf L } \bullet ) _ { 0 } = \sqrt { K } L _ { 0 0 } > 0$ . The identity $\mathbf { L } ^ { \top } \mathbf { G } \mathbf { L } = \mathbf { G }$ further implies

$$
- L _ { 0 0 } ^ { 2 } + \sum _ { i = 1 } ^ { d } L _ { i 0 } ^ { 2 } = - 1 ,
$$

so $\begin{array} { r } { L _ { 0 0 } ^ { 2 } = 1 + \sum _ { i = 1 } ^ { d } L _ { i 0 } ^ { 2 } \geq 1 } \end{array}$ and therefore $L _ { 0 0 } \ge 1$ . Orientation preservation gives $\operatorname* { d e t } ( \mathbf { L } ) = 1$ . Together with $\mathbf { L } \in O ( 1 , d )$ , these are the defining conditions of ${ \mathrm { S O } } ^ { + } ( 1 , d )$ □

Theorem 3 (Uniqueness of the Lorentzian Isometry). Let $\mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } , \mathbf { Z } _ { t ^ { \prime } , \mathrm { a l l } } ^ { s ^ { \prime } } \in \mathbb { R } ^ { 2 N \times ( d + 1 ) }$ . Suppose

(1) their extended Gram matrices agree, $\mathbf { G } _ { t , \mathrm { e x t } } ^ { s } = \mathbf { G } _ { t ^ { \prime } , \mathrm { e x t } } ^ { s ^ { \prime } } ;$ and

(2) both joint matrices have full column rank $d + 1 ,$ and

(3) the stage-to-stage correspondence is induced by a continuous path of full-column-rank configurations with the same extended Gram matrix, beginning at the first configuration.

Then there is a unique $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ satisfying

$$
\mathbf { Z } _ { t ^ { \prime } , \mathrm { a l l } } ^ { s ^ { \prime } } = \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } \mathbf { L } ^ { \top } .
$$

Proof. Full column rank means that the rows of both joint matrices span the ambient Lorentzian space. Associate each row $\mathbf { z } _ { t , i } ^ { m , s }$ with the corresponding row $\mathbf { z } _ { t ^ { \prime } , i } ^ { m , s ^ { \prime } }$ . Equality of the extended Gram matrices gives

$$
\langle \mathbf { z } _ { t , i } ^ { m , s } , \mathbf { z } _ { t , j } ^ { m ^ { \prime } , s } \rangle _ { \mathcal { L } } = \langle \mathbf { z } _ { t ^ { \prime } , i } ^ { m , s ^ { \prime } } , \mathbf { z } _ { t ^ { \prime } , j } ^ { m ^ { \prime } , s ^ { \prime } } \rangle _ { \mathcal { L } }
$$

for every pair of samples and modalities. This correspondence is therefore an isometry on a generating set. Lemma 1 extends it to a global element $\mathbf { L } \in O ( 1 , d )$ such that

$$
\mathbf { Z } _ { t ^ { \prime } , \mathrm { a l l } } ^ { s ^ { \prime } } = \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } \mathbf { L } ^ { \top } .
$$

It remains to identify the connected component. Let $\mathbf { Z } ( \lambda )$ denote the path in condition (3), with $\mathbf { Z } ( 0 ) = \mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } .$ . Full rank makes the corresponding isometry unique, and

$$
\mathbf { L } ( \boldsymbol { \lambda } ) ^ { \top } = \left( \mathbf { Z } ( 0 ) ^ { \top } \mathbf { Z } ( 0 ) \right) ^ { - 1 } \mathbf { Z } ( 0 ) ^ { \top } \mathbf { Z } ( \boldsymbol { \lambda } )
$$

makes its continuity explicit. Thus $\mathbf { L } ( 0 ) = \mathbf { I }$ and $\mathbf { L } ( 1 ) = \mathbf { L }$ . Since the determinant of an element of $O ( 1 , d )$ is either +1 or −1, continuity gives det $\mathbf { L } ( \lambda ) = + 1$ throughout the path. All configurations remain on the forward sheet, so applying L(�) to the origin gives $L _ { 0 0 } ( \lambda ) > 0 ;$ the Lorentz constraint then gives $L _ { 0 0 } ( \lambda ) \ge 1$ . Hence $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ For uniqueness, assume that For uniqueness, assume that $\mathbf { L } _ { 1 }$ and and $\mathbf { L } _ { 2 }$ both satisfy the relation. Then both satisfy the relation. Then

$$
\mathbf { Z } _ { t , \mathrm { a l l } } ^ { s } ( \mathbf { L } _ { 1 } ^ { \top } - \mathbf { L } _ { 2 } ^ { \top } ) = \mathbf { 0 } .
$$

Since $\mathbf { Z } _ { t , \mathrm { a l l } } ^ { s }$ has column rank $d + 1$ , its right null space contains only zero; hence $\mathbf { L } _ { 1 } = \mathbf { L } _ { 2 }$

Corollary 4 (A Common Isometry for All Modalities). Under Theorem 3, the same $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ acts on both modalities:

$$
\begin{array} { r } { \mathbf { Z } _ { t ^ { \prime } } ^ { m _ { 1 } , s ^ { \prime } } = \mathbf { Z } _ { t } ^ { m _ { 1 } , s } \mathbf { L } ^ { \top } , \qquad \mathbf { Z } _ { t ^ { \prime } } ^ { m _ { 2 } , s ^ { \prime } } = \mathbf { Z } _ { t } ^ { m _ { 2 } , s } \mathbf { L } ^ { \top } . } \end{array}
$$

If, in addition, each modality-specific block has full column rank, then any modality-specific maps $\mathbf { L } _ { 1 }$ and $\mathbf { L } _ { 2 }$ satisfying these two relations must obey $\mathbf { L } _ { 1 } = \mathbf { L } _ { 2 } = \mathbf { L }$ .

Proof. Stacking the two modality-specific relations would produce

$$
\mathbf { Z } _ { t ^ { \prime } , \mathrm { a l l } } ^ { s ^ { \prime } } = { \binom { \mathbf { Z } _ { t } ^ { m _ { 1 } , s } \mathbf { L } _ { 1 } ^ { \top } } { \mathbf { Z } _ { t } ^ { m _ { 2 } , s } \mathbf { L } _ { 2 } ^ { \top } } } .
$$

Theorem 3 gives the unique transformation for the full stacked matrix, so both blocks admit the same restriction. Under the additional blockwise full-rank condition, subtracting the common relation from each modality-specific relation gives $\mathbf { Z } _ { t } ^ { m _ { i } , s } ( \mathbf { L } _ { i } ^ { \top } - \mathbf { L } ^ { \top } ) = \mathbf { 0 }$ . The right null space is then trivial, and hence $\mathbf { L } _ { i } = \mathbf { L }$ for $i \in \{ 1 , 2 \}$ □

Remark 7. Conditions (P1) and (P2) are jointly equivalent to

$$
\mathbf { G } _ { t - 1 , \mathrm { e x t } } ^ { t } = \mathbf { G } _ { t - 1 , \mathrm { e x t } } ^ { t - 1 } .
$$

Theorem 3 and Corollary 4 therefore show that preserving both within-modal and cross-modal relations determines one Lorentz transformation shared by all modalities.

## D.2 Removing the Boost Component

Lemma 3 (Generation of ${ \mathrm { S O } } ^ { + } ( 1 , d )$ [71]). Every element of $S O ^ { + } ( 1 , d )$ can be written as a finite composition of Lorentz rotations and boosts.

Remark 8. Boosts and spatial rotations are intrinsic linear transformations ofthe Lorentz model: $i f \mathbf { x } \in \mathbb { H } _ { K } ^ { d }$ , then both Bx and Rx remain on $\mathbb { H } _ { K } ^ { d }$

Theorem 4 (Elimination of Boosts; Restatement of Theorem 1). Suppose $\mathbf { L } \in \mathrm { S O } ^ { + } ( 1 , d )$ maps every old representation according to $\mathbf { z } _ { t - 1 } ^ { m , t } = \mathbf { L } \mathbf { z } _ { t - 1 } ^ { m , t - 1 }$ , and the stacked old-representation matrix has full column rank. If Condition (P3) holds for all such representations, then

$$
\mathbf { L } = \left[ \begin{array} { c c } { 1 } & { \mathbf { 0 } ^ { \top } } \\ { \mathbf { 0 } } & { \widetilde { \mathbf { R } } } \end{array} \right] , \qquad \widetilde { \mathbf { R } } \in \mathrm { S O } ( d ) ,
$$

i.e., L contains no nontrivial boost.

Proof. For every old representation on $\mathbb { H } _ { K } ^ { d }$ , Condition (P3) fixes the spatial norm. The manifold constraint

$$
z _ { 0 } = \sqrt { K + \| \mathbf { z } _ { [ 1 : d ] } \| _ { 2 } ^ { 2 } }
$$

then also fixes the time-like coordinate. In row-matrix form,

$$
\mathbf { Z } _ { t - 1 , \mathrm { a l l } } ^ { t - 1 } ( \mathbf { L } ^ { \top } \mathbf { e } _ { 0 } - \mathbf { e } _ { 0 } ) = \mathbf { 0 } , \qquad \mathbf { e } _ { 0 } = ( 1 , 0 , \ldots , 0 ) ^ { \top } .
$$

Full column rank makes the right null space trivial, so $\mathbf { L } ^ { \top } \mathbf { e } _ { 0 } = \mathbf { e } _ { 0 }$ . Writing L in time–space blocks and using $\mathbf { L } ^ { \top } \mathbf { G } \mathbf { L } = \mathbf { G }$ therefore gives

$$
\mathbf { L } = \mathrm { d i a g } ( 1 , \widetilde { \mathbf { R } } ) , \qquad \widetilde { \mathbf { R } } ^ { \top } \widetilde { \mathbf { R } } = \mathbf { I } .
$$

Finally, det $\mathbf { L } = 1$ implies det $\widetilde { \mathbf { R } } = 1$ . Thus $\widetilde { \mathbf { R } } \in { \mathrm { S O } } ( d )$ and no boost component remains.

## D.3 First-Order Hierarchy Preservation

The next lemma relates perturbations of the spatial and time-like coordinates on the Lorentz hyperboloid and leads directly to Corollary 1.

Lemma 4 (First-Order Change of the Time-Like Coordinate). Let $\mathbf { z } \in \mathbb { R } ^ { d + 1 }$ obey $z _ { 0 } ^ { 2 } - \| \mathbf { z } _ { [ 1 : d ] } \| _ { 2 } ^ { 2 } = K$ with $z _ { 0 } > 0$ . For an infinitesimal perturbation $\Delta \mathbf { z } ,$

$$
\Delta z _ { 0 } = \left( { \frac { \mathbf { z } _ { [ 1 : d ] } ^ { \top } } { z _ { 0 } } } \right) \Delta \mathbf { z } _ { [ 1 : d ] } .\tag{28}
$$

Proof. Define the manifold constraint

$$
\phi ( \mathbf { z } ) = z _ { 0 } ^ { 2 } - \| \mathbf { z } _ { [ 1 : d ] } \| _ { 2 } ^ { 2 } - K = 0 .\tag{29}
$$

For $\widetilde { \mathbf { z } } = \mathbf { z } + \Delta \mathbf { z } .$ , a first-order Taylor expansion gives

$$
\phi ( \widetilde { \mathbf { z } } ) = \phi ( \mathbf { z } ) + \nabla \phi ( \mathbf { z } ) ^ { \top } \Delta \mathbf { z } + O ( \| \Delta \mathbf { z } \| _ { 2 } ^ { 2 } ) .
$$

Both z and $\widetilde { \mathbf { z } }$ lie on the manifold to first order, so

$$
\nabla \phi ( \mathbf { z } ) ^ { \top } \Delta \mathbf { z } = 0 .\tag{30}
$$

Since

$$
\nabla \phi ( \mathbf { z } ) = \left[ \begin{array} { c } { 2 z _ { 0 } } \\ { - 2 \mathbf { z } _ { [ 1 : d ] } } \end{array} \right] ,
$$

Equation (30) becomes $2 z _ { 0 } \Delta z _ { 0 } - 2 \mathbf { z } _ { [ 1 : d ] } ^ { \top } \Delta \mathbf { z } _ { [ 1 : d ] } = 0$ . Dividing by $2 z _ { 0 } > 0$ proves (28).

Corollary 5 (Restatement of Corollary 1). Let $\Delta \mathbf { z } = \mathbf { z } _ { t - 1 } ^ { m , t } - \mathbf { z } _ { t - 1 } ^ { m , t - 1 }$ denote the update of an old-task embedding. If Condition (P3) makes its time-like coordinate invariant to first order, then

$$
\begin{array} { r } { \mathbf { z } _ { [ 1 : d ] } ^ { \top } \Delta \mathbf { z } _ { [ 1 : d ] } = 0 . } \end{array}\tag{31}
$$

Proof. Condition (P3) sets $\Delta z _ { 0 } = 0$ to first order. Lemma 4 simultaneously requires

$$
\Delta z _ { 0 } = \frac { \mathbf { z } _ { [ 1 : d ] } ^ { \top } \Delta \mathbf { z } _ { [ 1 : d ] } } { z _ { 0 } } .
$$

Because $z _ { 0 } > 0$ , combining the two statements yields (31).

## E Derivations for the HMCL Update

## E.1 Proof of Proposition 1

Proposition 3 (Restatement: Admissible Parameter Changes). Consider the Lorentz layer in Section 3.1, with $\mathbf { W } ^ { m , t } = [ \mathbf { \nabla } \mathbf { w } _ { 0 } ^ { m , t } \mathbf { W } _ { s } ^ { m , t } ]$ . Assume that Theorem 1 holds at stage � − 1. Define

$$
\Delta \mathbf { W } ^ { m } = \mathbf { W } ^ { m , t } - \mathbf { W } ^ { m , t - 1 } , \qquad \lVert \Delta \mathbf { W } ^ { m } \rVert _ { F } \to 0 .
$$

The collection of modal displacements is admissible to first order if and only if one skew-symmetric � satisfies, simultaneously for every modality,

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , * } ( \Delta \mathbf { W } ^ { m } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } , \qquad \pmb { \Omega } ^ { \top } = - \pmb { \Omega } . } \end{array}\tag{32}
$$

Proof. The spatial output of the Lorentz layer is linear in its parameters. Its block form is

$$
f _ { [ 1 : d ] } ( \mathbf { W } ^ { m , t } ; \mathbf { Z } _ { t - 1 } ^ { m , * } ) = \mathbf { Z } _ { t - 1 } ^ { m , * } [ 0 ] ( \mathbf { w } _ { 0 } ^ { m , t } ) ^ { \top } + \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \mathbf { W } _ { s } ^ { m , t } ) ^ { \top } .\tag{33}
$$

Consequently, the applied displacement changes the old spatial output by

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } [ 1 { : } d ] = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 { : } d ] + \mathbf { Z } _ { t - 1 } ^ { m , * } [ 0 ] ( \Delta \mathbf { w } _ { 0 } ^ { m } ) ^ { \top } + \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } . } \end{array}\tag{34}
$$

Combining the two update terms gives the compact form

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , t } [ 1 { : } d ] = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 { : } d ] + \mathbf { Z } _ { t - 1 } ^ { m , * } ( \Delta \mathbf { W } ^ { m } ) ^ { \top } . } \end{array}\tag{35}
$$

Theorem 1 also states that an admissible change of the old representations is a spatial rotation. In a neighborhood of the identity, write

$$
\mathbf { R } = \mathbf { I } + \pmb { \Omega } + O ( \| \pmb { \Omega } \| _ { F } ^ { 2 } ) , \qquad \pmb { \Omega } ^ { \top } = - \pmb { \Omega } .
$$

It follows that

$$
\begin{array} { r l } & { \mathbf { Z } _ { t - 1 } ^ { m , t } [ 1 : d ] = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \mathbf { R } ^ { \top } } \\ & { \qquad = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] + \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } + O ( \Vert \pmb { \Omega } \Vert _ { F } ^ { 2 } ) . } \end{array}\tag{36}
$$

Equating the first-order terms of (35) and (36) yields

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , * } ( \Delta \mathbf { W } ^ { m } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } , \qquad \pmb { \Omega } ^ { \top } = - \pmb { \Omega } . } \end{array}\tag{37}
$$

The same � follows from Theorem 1 for all modalities. Conversely, (37) matches the tangent motion of that common rotation. Its spatial norm is unchanged to first order, and the reconstructed time-like coordinate in (4) is therefore unchanged as well. □

The implementation fixes the time-input column and realizes the admissible output motion through $\Delta \mathbf { W } _ { s } ^ { m }$ . Such a representative exists exactly when

$$
\begin{array} { r } { \mathrm { c o l } \Big ( \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 { : } d ] \pmb { \Omega } ^ { \top } \Big ) \subseteq \mathrm { c o l } \Big ( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] \Big ) . } \end{array}\tag{38}
$$

The regularity condition in Appendix E.2 makes the two column spaces equal, so (38) holds for every admissible rotation. This gives the blockwise form in (16).

## E.2 Derivation of Corollary 2

Corollary 6 (Restatement: Closest-Admissible Update). Given candidate modal displacements $\begin{array} { l l } { \displaystyle { \{ \delta ^ { m } } \ = }  \end{array}$ $[ \delta _ { 0 } ^ { m } \ \delta _ { s } ^ { m } ] \} _ { m = 1 } ^ { M }$ , HMCL-CA jointly selects their closest admissible displacements and one shared rotation by solving

$$
\begin{array} { l l } { \displaystyle \underset { \{ \Delta \mathbf { W } ^ { m } \} , \Omega } { \mathrm { m i n i m i z e } } } & { \displaystyle \frac { 1 } { 2 } \sum _ { m = 1 } ^ { M } \| \Delta \mathbf { W } ^ { m } - \delta ^ { m } \| _ { F } ^ { 2 } } \\ { s u b j e c t t o } & { \displaystyle \Omega ^ { \top } = - \Omega , \quad \Delta \mathbf { w } _ { 0 } ^ { m } = \mathbf { 0 } , } \\ & { \displaystyle \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } = \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \Omega ^ { \top } , \quad m = 1 , \ldots , M . } \end{array}\tag{39}
$$

Under the regularity condition below, the modal displacements are unique and are obtained from one Lyapunov equation for the common $\Omega ^ { \star }$

We derive the joint HMCL-CA solution and make the shared rotation explicit. For each modality, let

$$
{ \bf X } _ { m } = { \bf Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] , \qquad { \bf Y } _ { m } = { \bf Z } _ { t - 1 } ^ { m , t - 1 } [ 1 { : } d ] .\tag{40}
$$

Assume that $\mathbf { Y } _ { m }$ has full column rank and that col $( \mathbf { X } _ { m } ) = \mathrm { c o l } ( \mathbf { Y } _ { m } )$ . Then

$$
{ \bf { X } } _ { m } = { \bf { Y } } _ { m } { \bf { C } } _ { m } , \qquad { \bf { C } } _ { m } = { \bf { Y } } _ { m } ^ { \dag } { \bf { X } } _ { m } ,\tag{41}
$$

where $\mathbf { C } _ { m }$ is invertible. Proposition 1 now gives

$$
\mathbf { C } _ { m } ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } = \pmb { \Omega } ^ { \top } \quad \Longleftrightarrow \quad \Delta \mathbf { W } _ { s } ^ { m } = \pmb { \Omega } \mathbf { C } _ { m } ^ { - \top } .\tag{42}
$$

The same � occurs for every �. Substituting (42) into the closest-admissible problem yields

$$
\operatorname* { m i n } _ { \Omega ^ { \top } = - \Omega } \frac { 1 } { 2 } \sum _ { m = 1 } ^ { M } \left. \Omega \mathbf { C } _ { m } ^ { - \top } - \delta _ { s } ^ { m } \right. _ { F } ^ { 2 } .\tag{43}
$$

Define

$$
\mathbf { H } = \sum _ { m = 1 } ^ { M } \mathbf { C } _ { m } ^ { - \top } \mathbf { C } _ { m } ^ { - 1 } , \qquad \mathbf { F } = \sum _ { m = 1 } ^ { M } \delta _ { s } ^ { m } \mathbf { C } _ { m } ^ { - 1 } .\tag{44}
$$

Since every $\mathbf { C } _ { m }$ is invertible, H is positive definite. Stationarity of (43) over the skew-symmetric matrices gives the Lyapunov equation

$$
\boldsymbol { \Omega } ^ { \star } \boldsymbol { \mathrm { H } } + \boldsymbol { \mathrm { H } } \boldsymbol { \Omega } ^ { \star } = 2 \mathrm { S k e w } ( \boldsymbol { \mathrm { F } } ) , \qquad \mathrm { S k e w } ( \boldsymbol { \mathrm { F } } ) = \frac { 1 } { 2 } ( \boldsymbol { \mathrm { F } } - \boldsymbol { \mathrm { F } } ^ { \top } ) .\tag{45}
$$

Positive definiteness of H gives a unique skew-symmetric solution. The closest admissible displacements are therefore

$$
\delta _ { s } ^ { \mathrm { C A } , m } = \Omega ^ { \star } \mathbf { C } _ { m } ^ { - \top } , \qquad \delta _ { 0 } ^ { \mathrm { C A } , m } = \mathbf { 0 } , \qquad m = 1 , \ldots , M .\tag{46}
$$

This construction solves for one rotation from all modal candidates. In the optimizer-consistent update, each $\delta _ { s } ^ { m }$ is the spatial block of the displacement proposed by the chosen optimizer.

## E.3 Proof of Corollary 3

Corollary 7 (HMCL-MR: Minimal-Rotation Update (Restatement)). Let $\delta _ { s } ^ { m }$ be an unconstrained candidate displacement and impose Proposition 1 while penalizing the induced rotation. In the minimum-rotation limit, $\Omega ^ { \star } = 0$ and

$$
\begin{array} { r } { \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } = \mathbf { 0 } , \qquad \Delta \mathbf { w } _ { 0 } ^ { m } = \mathbf { 0 } . } \end{array}\tag{47}
$$

The canonical solution is the null-space projection

$$
\Delta \mathbf { W } _ { s } ^ { m } = \delta _ { s } ^ { m } ( \mathbf { I } - \mathbf { P } _ { t - 1 } ) , \qquad \mathbf { P } _ { t - 1 } = ( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ) ^ { \top } \left( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ) ^ { \top } \right) ^ { \dagger } \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] .\tag{48}
$$

Proof. Begin with the general admissible family

$$
\begin{array} { r } { { \bf { Z } } _ { t - 1 } ^ { m , * } [ 1 : d ] ( \Delta { \bf { W } } _ { s } ^ { m } ) ^ { \top } = { \bf { Z } } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } , \qquad { \bf { \Omega } } { \bf { \Omega } } ^ { \top } = - \pmb { \Omega } . } \end{array}\tag{49}
$$

Among these displacements, consider the rotation-regularized closest point to $\pmb { \delta } _ { s } ^ { m }$ :

$$
\operatorname* { m i n } _ { \Delta \mathbf { W } _ { s } ^ { m } , \Omega } \frac { 1 } { 2 } \| \Delta \mathbf { W } _ { s } ^ { m } - \boldsymbol { \delta } _ { s } ^ { m } \| _ { F } ^ { 2 } + \frac { \lambda } { 2 } \| \underline { { \Omega } } \| _ { F } ^ { 2 } \quad \mathrm { s . t . } \quad ( 4 9 ) .\tag{50}
$$

With multiplier �, the Lagrangian is

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } = \frac { 1 } { 2 } \| \Delta \mathbf { W } _ { s } ^ { m } - \delta _ { s } ^ { m } \| _ { F } ^ { 2 } + \frac { \lambda } { 2 } \| \boldsymbol { \Omega } \| _ { F } ^ { 2 } } \\ & { \qquad + \left. \mathbf { \Delta } \mathbf { A } , \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 : d ] ( \Delta \mathbf { W } _ { s } ^ { m } ) ^ { \top } - \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] \pmb { \Omega } ^ { \top } \right. . } \end{array}\tag{51}
$$

Stationarity gives

$$
\Delta \mathbf { W } _ { s } ^ { m } = \delta _ { s } ^ { m } - \boldsymbol { \Lambda } ^ { \top } \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ,\tag{52}
$$

$$
\pmb { \Omega } = \frac { 1 } { \lambda } \operatorname { S k e w } \\Bigl ( ( \mathbf { Z } _ { t - 1 } ^ { m , t - 1 } [ 1 : d ] ) ^ { \top } \pmb { \Lambda } \Bigr ) ,\tag{53}
$$

where Skew $( \mathbf { A } ) = ( \mathbf { A } - \mathbf { A } ^ { \top } ) / 2$ . Substituting (52) into (49) yields

$$
\mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ) ^ { \top } \mathbf { \Lambda } = \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \delta _ { s } ^ { m } ) ^ { \top } + O ( \lambda ^ { - 1 } ) .\tag{54}
$$

In the limit $\lambda \to \infty$ , the rotation term vanishes and

$$
\begin{array} { r } { \mathbf { \Delta } \Lambda = \left( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ) ^ { \top } \right) ^ { \dagger } \mathbf { Z } _ { t - 1 } ^ { m , * } [ 1 { : } d ] ( \delta _ { s } ^ { m } ) ^ { \top } . } \end{array}\tag{55}
$$

Inserting this expression into (52) produces

$$
\Delta { \bf W } _ { s } ^ { m } = \delta _ { s } ^ { m } \left[ { \bf I } - ( { \bf Z } _ { t - 1 } ^ { m , * } [ 1 \colon d ] ) ^ { \top } \left( { \bf Z } _ { t - 1 } ^ { m , * } [ 1 \colon d ] ( { \bf Z } _ { t - 1 } ^ { m , * } [ 1 \colon d ] ) ^ { \top } \right) ^ { \dagger } { \bf Z } _ { t - 1 } ^ { m , * } [ 1 \colon d ] \right] .
$$

Equation (53) simultaneously gives $\mathbf { \Omega } ^ { \Omega ^ { \star } } = \mathbf { 0 }$ . Combining this spatial projection with $\Delta \mathbf { w } _ { 0 } ^ { m } = \mathbf { 0 }$ proves the stated canonical update. □

## F Optimizer-Consistent HMCL: Derivations

This appendix derives the direction–displacement mismatch, Lorentz defect, finite-step preservation result, feasible accumulation, and low-rank residual bound for the unified HMCL update. Throughout a fixed task, we omit the stage index � and retain the optimizer-step index $k ;$ in $( k , 0 )$ and $( k , s )$ , the second entry identifies the time-like input column (0) or the spatial input block (�), whereas $\mathbf { W } _ { 0 } ^ { m , t }$ denotes the task-start parameters. As established in the main paper, a common spatial rotation preserves the three HMCL conditions: CA estimates its skew generator �, while MR selects the identity member $\pmb { \Omega } = \pmb { 0 }$ . Both corrections act on the fully formed candidate displacement.

## F.1 A Concrete Adaptive-Optimizer Mismatch

Let $S \subseteq \mathbb { R } ^ { q }$ be the linear subspace of admissible vectorized parameter displacements and let � be its orthogonal projector. For a positive diagonal matrix $\mathbf { D } _ { \mathrm { o p t } } .$ , consider the preconditioned component of an AdamW step,

$$
{ \bf d } _ { \mathrm { p r e } } = - \eta { \bf D } _ { \mathrm { o p t } } { \bf u } , \qquad { \bf u } \in S .\tag{56}
$$

Proposition 4 (Direction–Displacement Mismatch). Suppose AdamW maps an admissible first-moment direction $\mathbf { u } \in S t o$

$$
\mathbf { q } ^ { \mathrm { A d a m W } } = - \eta \mathbf { D } _ { \mathrm { o p t } } \mathbf { u } - \eta \omega \pmb { \theta } ,\tag{57}
$$

where $\mathbf { D } _ { \mathrm { o p t } }$ is apositive diagonal preconditioner and � is the decoupled weight-decay coeficient. The preconditioned component remains admissible for every $\mathbf { u } \in \mathcal { S }$ if and only if

$$
( \mathbf { I } - \mathbf { I I } ) \mathbf { D } _ { \mathrm { o p t } } \mathbf { I I } = \mathbf { 0 } .\tag{58}
$$

The decay component is admissible exactly when $\omega = 0$ or $\pmb { \theta } \in { \cal S } .$

ProofofProposition 4. Because � projects onto S, every admissible direction can be written as u = �a for some a. The preconditioned direction remains in $s$ for every admissible input if and only if its component in $S ^ { \perp }$ vanishes:

$$
( \mathbf { I } - \mathbf { I } \mathbf { I } ) \mathbf { D } _ { \mathrm { o p t } } \mathbf { u } = ( \mathbf { I } - \mathbf { I } \mathbf { I } ) \mathbf { D } _ { \mathrm { o p t } } \mathbf { H } \mathbf { a } = \mathbf { 0 } \quad { \mathrm { f o r ~ e v e r y ~ } } \mathbf { a } .\tag{59}
$$

This is equivalent to $( \mathbf { I } - \mathbf { I I } ) \mathbf { D } _ { \mathrm { o p t } } \mathbf { I I } = \mathbf { 0 }$ . The decoupled decay term is $- \eta \omega \pmb \theta .$ . It lies in S exactly when $\omega = 0$ or $\pmb { \theta } \in \mathcal { S }$ . Adding either non-admissible component to an admissible one does not restore the constraint in general. □

A two-dimensional counterexample. Let $S = \operatorname { s p a n } ( [ 1 , 1 ] ^ { \top } )$ and choose the admissible direction $\mathbf { u } = [ 1 , 1 ] ^ { \top }$ With $\mathbf { D } _ { \mathrm { o p t } } = \mathrm { d i a g } ( 1 , 2 )$ and no decay, AdamW-like scaling gives

$$
\mathbf { d } _ { \mathrm { p r e } } = - \eta [ 1 , 2 ] ^ { \top } ,
$$

which is not in $s .$ . Equivalently, the normal vector $\mathbf { v } _ { \bot } = [ 1 , - 1 ] ^ { \top }$ satisfies $\mathbf { v } _ { \bot } ^ { \top } \mathbf { u } = 0$ but $\mathbf { v } _ { \bot } ^ { \top } \mathbf { d } _ { \mathrm { p r e } } = \eta \neq 0$ . The example is intentionally small: the same non-commutativity occurs between a low-rank representation projector and AdamW’s element-wise preconditioner in the Lorentz head.

## F.2 Lorentz Strain of a Realized Optimizer Candidate

We now specialize the mismatch to the Lorentz representation geometry. For modality $m ,$ , let $\mathbf { Y } ^ { m } \in \mathbb { R } ^ { N _ { m } \times d }$ contain the old spatial coordinates as rows, and let $\Delta \mathbf Y ^ { m }$ contain their first-order changes under the realized optimizer candidate. Define

$$
\begin{array} { r l } & { { \bf B } ^ { m } = \arg \underset { { \bf B } } { \operatorname* { m i n } } \| \Delta { \bf Y } ^ { m } - { \bf Y } ^ { m } { \bf B } ^ { \top } \| _ { F } ^ { 2 } , } \\ & { { \bf E } ^ { m } = \Delta { \bf Y } ^ { m } - { \bf Y } ^ { m } ( { \bf B } ^ { m } ) ^ { \top } . } \end{array}\tag{60}
$$

Thus $\Delta \mathbf { Y } ^ { m } = \mathbf { Y } ^ { m } ( \mathbf { B } ^ { m } ) ^ { \top } + \mathbf { E } ^ { m }$ . Write $\mathbf { B } ^ { m } = \mathbf { S } ^ { m } + \boldsymbol { \Omega } ^ { m }$ , where $\mathbf { S } ^ { m } = \mathrm { S y m } ( \mathbf { B } ^ { m } )$ and $\Omega ^ { m } = \mathrm { S k e w } ( \mathbf { B } ^ { m } )$ . For a row �, we use the equivalent column-vector form $\Delta \mathbf { y } _ { i } ^ { m } = \mathbf { B } ^ { m } \mathbf { y } _ { i } ^ { m } + \mathbf { e } _ { i } ^ { m }$

Proposition 5 (Lorentz Defect of a Realized Step). Under the rank condition of Theorem 1, a realized first-order change preserves Conditions (P1)–(P3) if and only if

$$
{ \bf E } ^ { m } = { \bf 0 } , \qquad { \bf S } ^ { m } = { \bf 0 } , \qquad { \pmb { \Omega } } ^ { m } = { \pmb { \Omega } } ^ { m ^ { \prime } } \quad f o r e \nu e r y m , m ^ { \prime } .\tag{61}
$$

Equivalently, the change is induced by one shared skew-symmetric generator.

Proof. Suppose first that the realized change preserves Conditions (P1)–(P3) to first order. Proposition 1 then gives one skew-symmetric generator � shared by all modalities, so

$$
\Delta { \bf Y } ^ { m } = { \bf Y } ^ { m } \Omega ^ { \top } \quad \quad \mathrm { f o r e v e r y } m .\tag{62}
$$

Under the stated rank condition, the least-squares generator in (60) is unique. Hence $\mathbf { E } ^ { m } = \mathbf { 0 } , \mathbf { B } ^ { m } = \pmb { \Omega }$ , and $\mathrm { S y m } ( \mathbf { B } ^ { m } ) = \mathbf { 0 }$ for every modality.

Conversely, suppose (61) holds. Then all modalities satisfy (62) for the same skew-symmetric matrix �. For an old spatial coordinate y, its induced time-like derivative under the reconstruction in (4) is

$$
\dot { z } _ { 0 } = \frac { \mathbf { y } ^ { \top } \pmb { \Omega } \mathbf { y } } { \sqrt { K + \| \mathbf { y } \| _ { 2 } ^ { 2 } } } = 0 .\tag{63}
$$

The first-order change of a spatial inner product is likewise

$$
( \Omega \mathbf { y } _ { i } ) ^ { \top } \mathbf { y } _ { j } + \mathbf { y } _ { i } ^ { \top } ( \Omega \mathbf { y } _ { j } ) = \mathbf { y } _ { i } ^ { \top } ( \Omega ^ { \top } + \Omega ) \mathbf { y } _ { j } = 0 .\tag{64}
$$

The same cancellation applies across modalities because the generator is shared. The Lorentz Gram matrices and spatial radii are therefore stationary to first order, establishing Conditions (P1)–(P3). □

A compact measure of the departure from this shared tangent direction is

$$
\mathcal { D } _ { \mathcal { L } } = \sum _ { m } \left( \| \mathbf { S } ^ { m } \| _ { F } ^ { 2 } + \| \mathbf { E } ^ { m } \| _ { F } ^ { 2 } \right) + \sum _ { m < m ^ { \prime } } \| \pmb { \Omega } ^ { m } - \pmb { \Omega } ^ { m ^ { \prime } } \| _ { F } ^ { 2 } .\tag{65}
$$

It vanishes exactly for a shared first-order rotation under the same assumptions.

Radial and cone-aperture drift. Let $r = \| \mathbf { y } \| _ { 2 }$ and $\mathbf { a } = \mathbf { S } \mathbf { y } + \pmb { \Omega } \mathbf { y } + \mathbf { e }$ . A Taylor expansion and $\mathbf { y } ^ { \top } \pmb { \Omega } \mathbf { y } = 0$ give

$$
\Delta r = \frac {  { \mathbf { y } } ^ { \top } (  { \mathbf { S } }  { \mathbf { y } } +  { \mathbf { e } } ) } { r } + O ( \|  { \mathbf { a } } \| _ { 2 } ^ { 2 } ) .\tag{66}
$$

For $h ( r ) = \sin ^ { - 1 } ( 2 \kappa / r )$

$$
h ^ { \prime } ( r ) = - \frac { 2 \kappa } { r \sqrt { r ^ { 2 } - 4 \kappa ^ { 2 } } } ,\tag{67}
$$

and therefore

$$
\Delta \mathrm { a p e r } ( \mathbf { y } ) = - \frac { 2 \kappa } { r \sqrt { r ^ { 2 } - 4 \kappa ^ { 2 } } } \Delta r + O ( ( \Delta r ) ^ { 2 } ) .\tag{68}
$$

In particular, decoupled decay has local generator $\mathbf { B } _ { \mathrm { d e c a y } } = - \eta \omega \mathbf { I } .$ . It produces $\Delta r = - \eta \omega r + O ( ( \eta \omega ) ^ { 2 } )$ and therefore increases the cone aperture to first order. This calculation is a geometric interpretation of the decay term, not an explanation of the main-table results, which use zero weight decay; Appendix H.6 separately evaluates nonzero decay.

Lorentz Gram drift. Let $\mathbf { z } _ { i } = \left[ z _ { i , 0 } ; \mathbf { y } _ { i } \right]$ and $\dot { \bf z } _ { i } = \left[ \dot { z } _ { i , 0 } ; { \bf a } _ { i } \right]$ . The first variation of an intra- or inter-modal Lorentz product is

$$
\frac { \mathrm { d } } { \mathrm { d } \epsilon } \left. \left( \mathbf { z } _ { i } + \epsilon \dot { \mathbf { z } } _ { i } \right) ^ { \top } \mathbf { G } \left( \mathbf { z } _ { j } + \epsilon \dot { \mathbf { z } } _ { j } \right) \right| _ { \epsilon = 0 } = \dot { \mathbf { z } } _ { i } ^ { \top } \mathbf { G } \mathbf { z } _ { j } + \mathbf { z } _ { i } ^ { \top } \mathbf { G } \dot { \mathbf { z } } _ { j } .\tag{69}
$$

For one shared skew generator, both time-like derivatives vanish and the spatial terms cancel. Symmetric strain, residual motion, or diferent modality generators remove this cancellation. Equation (69) therefore links the optimizer defect in (65) directly to the similarity matrices used by the contrastive objective.

## F.3 Closest Projection of the Realized Step

For one modality, omit the superscript � and split the optimizer candidate at step � into the time-like input column $\delta _ { k , 0 } ^ { \mathrm { { o p t } } }$ and spatial input block $\bar { \delta } _ { k , s } ^ { \mathrm { { o p t } } }$ . Let $\mathbf { P } = \mathbf { V } \mathbf { V } ^ { \top }$ be an orthogonal projector and $\mathbf { Q } = \mathbf { I } - \mathbf { P }$ . We write $\boldsymbol { \delta } _ { k } = \left[ \begin{array} { l l } { \delta _ { k , 0 } } & { \delta _ { k , s } } \end{array} \right]$ for a generic displacement at the same step. On the upper sheet of $\mathbb { H } _ { K } ^ { d }$ , zero Lorentz geodesic distance is equivalent to equality of the two points. Because the spatial block of (4) is linear in its input and the time-input parameter block is fixed, the canonical zero-motion condition used in (19) is equivalent to $\delta _ { k , s } \mathbf { P } = \mathbf { 0 }$ and $\delta _ { k , 0 } = \mathbf { 0 }$ . The following theorem and corollary establish the closest-point and finite-step preservation claims used in Theorem 2.

Theorem 5 (Closest Identity-Isometry Representative). Among displacements satisfying $\delta _ { k , s } \mathbf { P } = \mathbf { 0 }$ and $\delta _ { k , 0 } = \mathbf { 0 } ;$ the unique displacement closest to the realized optimizer candidate is

$$
\delta _ { k , s } ^ { \star } = \delta _ { k , s } ^ { \mathrm { o p t } } ( { \bf I } - { \bf P } ) , \qquad \delta _ { k , 0 } ^ { \star } = { \bf 0 } .\tag{70}
$$

Proof of Theorem 5. Every spatial matrix has the orthogonal decomposition

$$
\delta _ { k , s } ^ { \mathrm { o p t } } = \delta _ { k , s } ^ { \mathrm { o p t } } \mathbf { P } + \delta _ { k , s } ^ { \mathrm { o p t } } \mathbf { Q } .\tag{71}
$$

The two terms are orthogonal in the Frobenius inner product because $\mathbf P \mathbf Q = \mathbf 0$ . A feasible spatial displacement $\mathbf { D } _ { k , s }$ satisfies $\mathbf { D } _ { k , s } \mathbf { P } = \mathbf { 0 }$ and therefore $\mathbf { D } _ { k , s } = \mathbf { D } _ { k , s } \mathbf { Q }$ . Hence

$$
\lVert \mathbf { D } _ { k , s } - \delta _ { k , s } ^ { \mathrm { o p t } } \rVert _ { F } ^ { 2 } = \lVert \mathbf { D } _ { k , s } - \delta _ { k , s } ^ { \mathrm { o p t } } \mathbf { Q } \rVert _ { F } ^ { 2 } + \lVert \delta _ { k , s } ^ { \mathrm { o p t } } \mathbf { P } \rVert _ { F } ^ { 2 } .\tag{72}
$$

The second term is independent of $\mathbf { D } _ { k , s }$ , while the first is uniquely minimized by $\mathbf { D } _ { k , s } = \delta _ { k , s } ^ { \mathrm { o p t } } \mathbf { Q }$ . The time-like constraint has the unique closest value $\mathbf { d } _ { k , 0 } = \mathbf { 0 }$ . Combining the two blocks gives (70). □

This proof is independent of how the optimizer constructs $\delta _ { k } ^ { \mathrm { { o p t } } }$ . Learning rates, moment estimates, preconditioners, and decay afect the candidate, while the Lorentz-derived projection solves the same closest-point problem for the displacement actually proposed.

Corollary 8 (Exact Protected Lorentz Invariants). If an old frozen feature x lies in the range of $\mathbf { P } _ { t - 1 }$ , the update in (70) leaves its complete Lorentz representation unchanged after eachfinite step:

$$
\mathbf { z } _ { k + 1 } ^ { m } ( \mathbf { x } ) = \mathbf { z } _ { k } ^ { m } ( \mathbf { x } ) .\tag{73}
$$

Hence its hyperbolic relations, spatial radius, and entailment-cone aperture are unchanged.

Proof of Corollary 8. Let $\mathbf { x } = \mathbf { P } _ { t - 1 } \mathbf { x }$ . Equation (70) gives

$$
\begin{array} { r } { \delta _ { k , s } ^ { \star } \mathbf { x } = \delta _ { k , s } ^ { \mathrm { o p t } } ( \mathbf { I } - \mathbf { P } _ { t - 1 } ) \mathbf { P } _ { t - 1 } \mathbf { x } = \mathbf { 0 } , } \end{array}\tag{74}
$$

and the time-like input block is fixed. The spatial output of the Lorentz layer is therefore identical before and after the corrected step. Its reconstructed first coordinate, which is a deterministic function of the spatial norm, is identical as well. Thus each protected Lorentz point is unchanged. Every intra- and inter-modal Lorentz Gram entry, hyperbolic distance, spatial radius, and cone aperture formed from these points is consequently unchanged. □

## F.4 Task-Anchored Feasible Contraction

For a protected representation system fixed throughout task $t ,$ let $\mathcal { A } _ { t - 1 }$ be the joint admissible family in (17). It is linear because its defining constraints are linear in the modal displacements and their shared skew generator.

For a modal tuple $\mathcal { U } = ( \mathbf { U } ^ { m } ) _ { m = 1 } ^ { M }$ , define $\begin{array} { r } { \| \mathcal { U } \| _ { \oplus F } = \big ( \sum _ { m } \| \mathbf { U } ^ { m } \| _ { F } ^ { 2 } \big ) ^ { 1 / 2 } } \end{array}$ . Here $u _ { \mathrm { m a x } }$ bounds the realized AdamW displacement rather than the raw gradient. Its matrix-wide value can grow with the number of modal parameters even when the per-parameter step is small; the result requires only a finite bound, not $u _ { \mathrm { m a x } } \leq 1$ , although a larger value makes the numerical accumulation bound looser.

Corollary 9 (Within-Stage Admissibility and Bounded Accumulation). Let ${ \bf A } _ { 0 } ^ { m } = { \bf 0 } f o r$ every modality � denote the initial displacementfrom the task-start parameters. Given optimizer candidates $\delta _ { k } ^ { \mathrm { { o p t } , \it { t } } }$ , let $\mathbf { D } _ { k } ^ { t } = \mathcal { P } _ { t - 1 } ^ { \mathcal { L } } ( \delta _ { k } ^ { \mathrm { o p t } , t } ) =$ $( \mathbf { D } _ { k } ^ { m } ) _ { m = 1 } ^ { M }$ and define

$$
\mathbf { A } _ { k + 1 } ^ { m } = ( 1 - \beta ) ( \mathbf { A } _ { k } ^ { m } + \mathbf { D } _ { k } ^ { m } ) , \qquad 0 \leq \beta < 1 .\tag{75}
$$

Then $\{ \mathbf { A } _ { k } ^ { m } \} _ { m = 1 } ^ { M } \in \mathcal { A } _ { t - 1 } f o r$ every �. Relative to the uncontracted candidate, the task-start distance is reduced by $1 - \beta .$ The projection is non-expansive: $\| \mathbf { D } _ { k } ^ { t } \| _ { \oplus F } \leq \| \delta _ { k } ^ { \mathrm { o p t } , t } \| _ { \oplus F }$ . If the latter is at most $u _ { \mathrm { m a x } }$ for all �, then

$$
\begin{array} { r l } & { \displaystyle \| \mathbf { A } _ { J } ^ { t } \| _ { \oplus F } \leq \displaystyle \sum _ { j = 0 } ^ { J - 1 } ( 1 - \beta ) ^ { J - j } \| \mathbf { D } _ { j } ^ { t } \| _ { \oplus F } } \\ & { \qquad \leq \left\{ \begin{array} { l l } { J u _ { \operatorname* { m a x } } , } & { \beta = 0 , } \\ { \displaystyle \frac { ( 1 - \beta ) [ 1 - ( 1 - \beta ) ^ { J } ] } { \beta } u _ { \operatorname* { m a x } } , } & { 0 < \beta < 1 , } \\ { \displaystyle \sum _ { j } u _ { \operatorname* { m a x } } } & { ( 0 < \beta < 1 ) . } \end{array} \right. } \end{array}\tag{76}
$$

Proof of Corollary 9. The set $\mathcal { A } _ { t - 1 }$ is a linear subspace, so projection places every $\mathbf { D } _ { k } ^ { t }$ in it and induction keeps every $\mathbf { A } _ { k } ^ { t }$ in it. As the orthogonal projection onto this subspace, $\mathcal { P } _ { t - 1 } ^ { \mathcal { L } }$ is non-expansive in the joint Frobenius norm. Unrolling the recurrence gives

$$
\mathbf { A } _ { J } ^ { t } = \sum _ { j = 0 } ^ { J - 1 } ( 1 - \beta ) ^ { J - j } \mathbf { D } _ { j } ^ { t } .
$$

The triangle inequality gives the data-dependent bound in (76); substituting the proposal bound $u _ { \mathrm { m a x } }$ and summing the geometric series gives its two stated cases. □

## F.5 Residual Drift under Low-Rank PCA

Let $\mathbf { X } \in \mathbb { R } ^ { n \times p }$ contain the old frozen features, and let ${ \boldsymbol { \Sigma } } = \mathbf { X } ^ { \top } \mathbf { X } / n$ have eigenvalues $\lambda _ { 1 } \geq \cdots \geq \lambda _ { p } \geq 0$ . Let ${ \bf P } _ { r }$ be the orthogonal projector onto the top-� eigenspace and $\mathbf { Q } _ { r } = \mathbf { I } - \mathbf { P } _ { r }$ . For the zero-rotation restriction, $\mathbf { D } _ { k , s } = \mathbf { D } _ { k , s } \mathbf { Q } _ { r }$ , so

$$
\frac { 1 } { n } \| \mathbf { X D } _ { k , s } ^ { \top } \| _ { F } ^ { 2 } = \frac { 1 } { n } \| \mathbf { X Q } _ { r } \mathbf { D } _ { k , s } ^ { \top } \| _ { F } ^ { 2 }\tag{77}
$$

$$
\leq \| \mathbf { D } _ { k , s } \| _ { 2 } ^ { 2 } \frac { 1 } { n } \| \mathbf { X } \mathbf { Q } _ { r } \| _ { F } ^ { 2 }\tag{78}
$$

$$
= \| \mathbf { D } _ { k , s } \| _ { 2 } ^ { 2 } \sum _ { i > r } \lambda _ { i } .\tag{79}
$$

Thus the update is exact on the retained subspace, and its mean-squared efect outside that subspace is bounded by the PCA spectral tail. This is an average-energy statement. It does not bound the worst example or the worst earlier task.

The task-anchored update retains the same subspace property. The joint bound also applies to every modal block; in particular, $\mathbf { A } _ { J , s } = \mathbf { A } _ { J , s } \mathbf { Q } _ { \iota }$ . Combining (76) with (79) gives, for $0 < \beta < 1$

$$
\frac { 1 } { n } \| \mathbf { X } \mathbf { A } _ { J , s } ^ { \top } \| _ { F } ^ { 2 } \leq \left( \frac { 1 - \beta } { \beta } u _ { \operatorname* { m a x } } \right) ^ { 2 } \sum _ { i > r } \lambda _ { i } .\tag{80}
$$

The geometric correction controls the retained subspace exactly, whereas the anchor and the spectral tail jointly bound the average residual motion outside that subspace.

## F.6 Pseudocode, Optimizer State, and Complexity

Algorithms 1 and 2 instantiate the same post-optimizer, task-anchored update. They difer only in the closestadmissible correction: HMCL-MR fixes the shared rotation to zero, whereas HMCL-CA estimates one rotation jointly from all modal candidates.

Algorithm 1: Optimizer-Consistent HMCL-MR   
Input: Current-task data $\mathcal { D } _ { t } \mathrm { : }$ task-start Lorentz heads $\{ \mathbf { W } _ { 0 } ^ { m , t } \} _ { m = 1 } ^ { M } ;$ ; protected projector $\mathbf { P } _ { t - 1 } ;$ optimizer Opt;   
anchoring coeficient $\beta .$   
Output: Final task parameters $\{ \mathbf { W } _ { J } ^ { m , t } \} _ { m = 1 } ^ { M } .$   
for optimizer step $k = 0 , \ldots , J - 1$ do   
Compute the HMCL loss on $\mathcal { D } _ { t }$ and gradients $\{ \mathbf { g } _ { k } ^ { m , t } \} _ { m = 1 } ^ { M }$   
Let Opt update its state and propose $\{ \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } \} _ { m = 1 } ^ { M } ;$   
for $m = 1 , \ldots , M$ do   
$\delta _ { k } ^ { \mathrm { o p t } , m , t } \gets \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { k } ^ { m , t } ;$   
$\delta _ { k , 0 } ^ { \mathrm { \tilde { M R } } , m , t } \gets \mathbf { 0 } , \delta _ { k , s } ^ { \mathrm { \tilde { M R } } , m , t } \gets \delta _ { k , s } ^ { \mathrm { o p t } , m , t } ( \mathbf { I } - \mathbf { P } _ { t - 1 } ) ;$   
$\widehat { \mathbf { W } } _ { k + 1 } ^ { m , t } \gets \mathbf { W } _ { k } ^ { m , t } + \delta _ { k } ^ { \mathrm { M R } , m , t } ;$   
$\mathbf { W } _ { k + 1 } ^ { m , t }  \mathbf { W } _ { 0 } ^ { m , t } + ( 1 - \beta ) ( \widehat { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { 0 } ^ { m , t } ) ;$   
return $\{ \mathbf { W } _ { J } ^ { m , t } \} _ { m = 1 } ^ { M } .$

Algorithm 2: Optimizer-Consistent HMCL-CA   
Input: Current-task data $\mathcal { D } _ { t } \mathrm { : }$ ; task-start Lorentz heads $\{ \mathbf { W } _ { 0 } ^ { m , t } \} _ { m = 1 } ^ { M } ;$ protected matrices $\{ \mathbf { X } _ { m } , \mathbf { Y } _ { m } \} _ { m = 1 } ^ { M }$ from   
Appendix E.2; optimizer Opt; anchoring coeficient $\beta .$   
Output: Final task parameters $\{ \bar { \mathbf { W } } _ { J } ^ { m , t } \} _ { m = 1 } ^ { M } .$   
for $m = 1 , \ldots , M$ do   
$\mathbf { C } _ { m } \gets \mathbf { Y } _ { m } ^ { \dagger } \mathbf { X } _ { m } ;$   
H $\begin{array} { r } {  \sum _ { m = 1 } ^ { M } \mathbf { C } _ { m } ^ { - \top } \mathbf { C } _ { m } ^ { - 1 } ; } \end{array}$   
for optimizer step $k = 0 , \ldots , J - 1$ do   
Compute the HMCL loss on $\mathcal { D } _ { t }$ and gradients $\{ \mathbf { g } _ { k } ^ { m , t } \} _ { m = 1 } ^ { M }$   
Let Opt update its state and propose $\{ \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } \} _ { m = 1 } ^ { M } ;$   
for $m = 1 , \ldots , M$ do   
$\begin{array} { r l } { \llangle } & { { } \delta _ { k } ^ { \mathrm { o p t } , m , t } \gets \widetilde { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { k } ^ { m , t } ; } \end{array}$   
$\begin{array} { r } { \mathbf { F } _ { k } \gets \sum _ { m = 1 } ^ { M } \delta _ { k , s } ^ { \mathrm { o p t } , m , t } \mathbf { C } _ { m } ^ { - 1 } ; } \end{array}$   
Solve $\Omega _ { k } ^ { \star } \mathbf { H } + \mathbf { H } \Omega _ { k } ^ { \star } = 2 \operatorname { S k e w } ( \mathbf { F } _ { k } ) ;$   
for $m = 1 , \ldots , M$ do   
$\delta _ { k , 0 } ^ { \mathrm { C A } , m , t } \gets \mathbf { 0 } , \delta _ { k , s } ^ { \mathrm { C A } , m , t } \gets \Omega _ { k } ^ { \star } \mathbf { C } _ { m } ^ { - \top }$   
$\widehat { \mathbf { W } } _ { k + 1 } ^ { m , t } \gets \mathbf { W } _ { k } ^ { m , t } + \delta _ { k } ^ { \mathbf { C A } , m , t } ;$   
$\mathbf { W } _ { k + 1 } ^ { m , t }  \mathbf { W } _ { 0 } ^ { m , t } + ( 1 - \beta ) ( \widehat { \mathbf { W } } _ { k + 1 } ^ { m , t } - \mathbf { W } _ { 0 } ^ { m , t } ) ;$   
return $\{ \mathbf { W } _ { J } ^ { m , t } \} _ { m = 1 } ^ { M } .$

For $\mathbf { W } _ { k , s } \in \mathbb { R } ^ { d \times p }$ , the HMCL-MR correction $\delta _ { k , s } \gets \delta _ { k , s } - ( \delta _ { k , s } \mathbf { V } ) \mathbf { V } ^ { \top }$ costs $O ( d p r )$ and stores $O ( p r )$ values. The dense HMCL-CA solver adds one $d \times d$ Lyapunov equation per step, with $O ( d ^ { 3 } )$ time; a rank-� realization reduces this solve to $O ( r ^ { 3 } )$ . The PCA basis and the task-level CA factors are estimated once per stage. The optimizer state remains the state produced by the raw gradients; for AdamW, this state includes the first- and second-moment estimates. The guarantees concern the realized parameter trajectory rather than a transported optimizer state.

## G Additional Implementation Details

## G.1 Dataset Details and Task Stream

The unified journal benchmark contains 16 datasets and mixes image–text classification with bidirectional cross-modal retrieval.

## G.1.1 Classification Datasets

The classification portion includes CIFAR-10 and CIFAR-100 [72], Caltech-101 [73], ImageNet-1K [66], Food 101 [74], Flowers-102 [75], EuroSAT [76], DTD [77], FGVC Aircraft [78], MNIST [79], PCAM [80], Country211 [1], CLEVR [81], and SST-2 [82]. Collectively, these benchmarks cover coarse object categories, fine-grained recognition, textures, scenes, and language sentiment. We regard each dataset as a separate task and evaluate it using the standard image–text classification protocol.

## G.1.2 Retrieval Datasets

COCO [64] and Flickr30K [65] provide the retrieval stages. Both contain paired images and captions and are evaluated in the image-to-text and text-to-image directions.

## G.1.3 Task Stream Construction

The datasets follow one fixed sequential order shared by every method and random seed. At a given stage, non-replay methods optimize only on the current dataset and do not revisit examples from completed tasks. Replay-based baselines additionally access earlier tasks through their explicitly bounded memory bufers, whose capacities are reported with the method configurations. Classification and retrieval are interleaved so that the stream requires the model to alternate between diferent multimodal objectives.

## G.1.4 Dataset Statistics

Table 7 reports the train/test sizes, number of categories where applicable, and the evaluation measure for every stage.

Table 7. Composition and statistics of the 16-stage continual multimodal benchmark.
<table><tr><td>Dataset</td><td>Task</td><td>Classes</td><td>Train</td><td>Test</td><td>Evaluation</td></tr><tr><td>FGVC Aircraft [78]</td><td>Classification</td><td>100</td><td>3,334</td><td>3,333</td><td>Accuracy</td></tr><tr><td>Caltech-101 [73]</td><td>Classification</td><td>102</td><td>2,448</td><td>6,084</td><td>Accuracy</td></tr><tr><td>ImageNet-1K [66]</td><td>Classification</td><td>1,000</td><td>1,281,167</td><td>50,000</td><td>Accuracy</td></tr><tr><td>CIFAR-10 [72]</td><td>Classification</td><td>10</td><td>45,000</td><td>10,000</td><td>Accuracy</td></tr><tr><td>CIFAR-100 [72]</td><td>Classification</td><td>100</td><td>45,000</td><td>10,000</td><td>Accuracy</td></tr><tr><td>CLEVR [81]</td><td>Classification</td><td>8</td><td>4,500</td><td>5,000</td><td>Accuracy</td></tr><tr><td>Country211 [1]</td><td>Classification</td><td>211</td><td>31,650</td><td>21,100</td><td>Accuracy</td></tr><tr><td>DTD [77]</td><td>Classification</td><td>47</td><td>1,880</td><td>1,880</td><td>Accuracy</td></tr><tr><td>EuroSAT [76]</td><td>Classification</td><td>10</td><td>5,000</td><td>5,000</td><td>Accuracy</td></tr><tr><td>Flowers-102 [75]</td><td>Classification</td><td>102</td><td>1,020</td><td>6,149</td><td>Accuracy</td></tr><tr><td>Food-101 [74]</td><td>Classification</td><td>101</td><td>68,175</td><td>25,250</td><td>Accuracy</td></tr><tr><td>MNIST [79]</td><td>Classification</td><td>10</td><td>48,000</td><td>10,000</td><td>Accuracy</td></tr><tr><td>PCAM [80]</td><td>Classification</td><td>2</td><td>262,144</td><td>32,768</td><td>Accuracy</td></tr><tr><td>SST-2 [82]</td><td>Classification</td><td>2</td><td>6,920</td><td>1,821</td><td>Accuracy</td></tr><tr><td>COCO [64]</td><td>Retrieval</td><td>一</td><td>118,287</td><td>5,000</td><td>Recall@K</td></tr><tr><td>Flickr30K [65]</td><td>Retrieval</td><td>一</td><td>29,000</td><td>1,000</td><td>Recall@K</td></tr></table>

## G.2 Task-Order Robustness

We permute all tasks in the unified 16-task benchmark and keep the HyCoCLIP-B backbone, optimizer, training schedule, and main-table HMCL configurations fixed. The task-order RNG is independent of the five paired training seeds, and no hyperparameter is selected on either shufled stream. The two orders, generated once using order seeds 2608311 and 2608312, are:

seq2: Flickr30K → MNIST → PCAM → Food-101 → EuroSAT → Caltech-101 → SST-2 → COCO → Flowers-102 → CIFAR-10 → CLEVR → Aircraft → DTD → CIFAR-100 → Country211 → ImageNet. seq3: DTD → ImageNet → CLEVR → CIFAR-10 → Flickr30K → PCAM → Flowers-102 → EuroSAT → COCO → CIFAR-100 → Country211 → Aircraft → MNIST → Food-101 → Caltech-101 → SST-2.

Table 8 reports the direct five-seed results without relative-diference rows. On seq2, HMCL<sub>MR</sub> gives the highest Overall score and BWT, while $\mathrm { \ H M C L _ { C A } }$ gives the highest final retrieval score. On seq3, $\mathrm { H M C L } _ { \mathrm { C A } }$ gives the highest Overall score, while $\mathrm { H M C L } _ { \mathrm { M R } }$ gives the strongest retrieval score and marginally better Overall BWT. Both complete variants retain a clear advantage over Vanilla under each permutation, showing that the main result is not tied to the canonical task order. The separate AdamW weight-decay sweep is reported in Appendix H.6.

Table 8. Task-order robustness on HyCoCLIP-B using two fixed permutations of the unified 16-task stream (mean ± sample standard deviation over five paired seeds). Classification uses top-1 accuracy, retrieval uses symmetric R@5, and BWT is final minus immediate performance. Both HMCL variants use their fixed main-table configurations; no hyperparameter is retuned for either order.
<table><tr><td rowspan="2">Order Method</td><td rowspan="2"></td><td colspan="2">Classification</td><td colspan="2">Retrieval</td><td colspan="2">Overall</td></tr><tr><td>Acc ↑</td><td> $\mathbf { B W T _ { A } } \uparrow$ </td><td>R@5↑</td><td> $\mathbf { B W T } _ { \mathrm { R 5 } } \uparrow$ </td><td>Overall ↑</td><td>BWT ↑</td></tr><tr><td>seq2</td><td>Vanilla</td><td>42.095 ± 0.128</td><td>−3.890 ± 0.223</td><td>62.673 ± 0.095</td><td>−10.002 ± 0.092</td><td>44.668 ± 0.104</td><td>-4.654 ± 0.187</td></tr><tr><td>seq2</td><td>HMCLMR</td><td>45.441 ± 0.049</td><td>-0.198 ± 0.054</td><td>71.263± 0.082</td><td>-3.997 1 ± 0.101</td><td>48.669 ) ± 0.047</td><td>-0.673 ± 0.051</td></tr><tr><td>seq2</td><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td>45.222±0.020</td><td>−0.324±0.058</td><td>71.351 ± 0.071</td><td>-4.139 9 ± 0.057</td><td>48.488±0.016</td><td>−0.801±0.054</td></tr><tr><td>seq3</td><td>Vanilla</td><td>41.299 ± 0.168</td><td>-4.686 ± 0.051</td><td>62.551 ± 0.090</td><td>-5.551 ± 0.193</td><td>43.956 ± 0.147</td><td>-4.794 ± 0.050</td></tr><tr><td>seq3</td><td> $\mathrm { H M C L } _ { \mathrm { M R } }$ </td><td>45.278±0.062</td><td>−0.733±0.088</td><td>72.613± 0.081</td><td>−2.141 ± 0.117</td><td>48.695±0.053</td><td>−0.909 ± 0.065</td></tr><tr><td>seq3</td><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td>45.430 ± 0.070</td><td>−0.684±0.124</td><td>72.397±0.052</td><td>−2.579±0.034</td><td>48.801 ± 0.064</td><td>−0.920±0.113</td></tr></table>

## G.3 Supplementary Materials for Case Study

We further visualize how continual adaptation changes the visual–semantic hierarchy learned on Flickr30K. Nouns and adjectives extracted from the dataset captions augment the candidate text pool with concepts at diferent abstraction levels. Starting from an image embedding, we sample 20 locations on the geodesic leading to [ROOT], the manifold origin, and compare the nearest concepts recovered by HMCL and Vanilla.

Examples (1)–(3) search a joint pool of captions, nouns, and adjectives; examples (4)–(6) use captions alone. Blue rows show traversals immediately after the Flickr30K stage, while green rows show the same analysis after the full continual sequence. In these examples, HMCL gives a clearer movement from image-specific descriptions toward more general concepts. Vanilla shows less consistent results after later tasks, which is consistent with greater semantic and hierarchical drift.

![](images/81d12b22dcdfcaa609068cbd44315cd1cec2e134ae147f640117779f323cb0d2.jpg)  
(1)  
After Flickr30K After the full stream

![](images/577516fa398e027efa1aa1cb29ad5b58004569f7a778c52161588e7f400eebd7.jpg)  
(2)

(3)  
![](images/504130a84ff12b8e206acb052ff17560e4e1a59e9f52ba11d1362c87d881e401.jpg)

<table><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>A dog runs on the green grass near a wooden fence . A black and white dog is running in a grassy garden</td><td>surrounded by a white fence .</td></tr><tr><td>dog</td><td>handsome</td></tr><tr><td>sincerity</td><td>fondness</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td></td><td>A dog runs on the green grass near a wooden fence . A black and white dog is running in a grassy garden surrounded by a white fence .</td></tr><tr><td>dog</td><td>aloof</td></tr><tr><td>sincerity</td><td>pleasure</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>orange hat .</td><td>The man with pierced ears is wearing glasses and an The man with pierced ears is wearing glasses and an orange hat .</td></tr><tr><td>hat</td><td>fashion</td></tr><tr><td>fashion</td><td>style</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>orange hat .</td><td>The man with pierced ears is wearing glasses and an The man with pierced ears is wearing glasses and an orange hat .</td></tr><tr><td>fashion</td><td>kind</td></tr><tr><td>cousin</td><td>new</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>A girl in a pink shirt slides down an inflatable fun slide</td><td>Young girl sliding down an inflated slide .</td></tr><tr><td>summer</td><td>summer</td></tr><tr><td>conventional</td><td>quality</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>A girl in a pink shirt slides down an inflatable fun slide</td><td>Young girl sliding down an inflated slide .</td></tr><tr><td>summer</td><td>summer</td></tr><tr><td>quick</td><td>material</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr></table>

![](images/e89a17dc9b4aa91121383bebff62b5dfa14cb7590e20d79ca8884e671a6f0978.jpg)  
(4)

![](images/4e901be0521c1ca2178342d825211bc03bf74fadf1d2fb6968c10d6817f16ec8.jpg)  
(5)

(6)  
![](images/b5b742af724a3e7ab643b859965ec8728b7f818266b2b3d5d9c766f83b5ed76f.jpg)

After Flickr30K After the full stream
<table><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>Numerous Asian people working behind a counter .</td><td>Numerous Asian people working behind a counter .</td></tr><tr><td>An Asian food market where a woman is picking out An Asian food market where a woman is picking out food .</td><td>food .</td></tr><tr><td>Employees at a sushi restaurant prepare for dinner time rush .</td><td>A cyclist wearing a black helmet is riding by some black vans .</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>Numerous Asian people working behind a counter .</td><td>Employees at a sushi restaurant prepare for dinner time rush .</td></tr><tr><td>An Asian food market where a woman is picking out food .</td><td>A cyclist wearing a black helmet is riding by some black vans .</td></tr><tr><td>A cyclist wearing a black helmet is riding by some black vans .</td><td>Two wet dogs run into the surf at sunset .</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>2 boys in the foreground in a karate competition and coaches in background looking on with another coach sitting at table .</td><td>A young boy demonstrates karate in a gymnasium .</td></tr><tr><td>A young boy demonstrates karate in a gymnasium .</td><td>A little girl dressed in yellow splashes in a shallow</td></tr><tr><td>Man standing by a poster of religious beliefs .</td><td>pool . A child plays at a playground .</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>2 boys in the foreground in a karate competition and coaches in background looking on with another coach sitting at table .</td><td>Two people are sitting outdoors on a blanket near a tree .</td></tr><tr><td>A young boy demonstrates karate in a gymnasium .</td><td>A surfer jumps a wave .</td></tr><tr><td>Man standing by a poster of religious beliefs .</td><td>A couple enjoying a glass of white wine .</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>A young boy is sitting on the floor trying to take off his boots .</td><td>A woman wearing glasses and an orange dress is standing with another woman wearing glasses and a</td></tr><tr><td>A young boy puts on his boots at the top of a flight of</td><td>green dress . Three young men wearing hooded sweatshirts ,</td></tr><tr><td>stairs . A young boy demonstrates karate in a gymnasium .</td><td>standing at a bench . A couple enjoying a glass of white wine .</td></tr><tr><td>[ROOT]</td><td>[ROOT]</td></tr><tr><td>HMCL (ours)</td><td>Vanilla</td></tr><tr><td>A young boy is sitting on the floor trying to take off his boots .</td><td>An Indian girl playing a guitar .</td></tr><tr><td>A young boy puts on his boots at the top of a flight of</td><td>A couple enjoying a glass of white wine .</td></tr><tr><td>stairs .</td><td></td></tr><tr><td>couple enjoying a glass of white wine . [ROOT]</td><td>Two wet dogs run into the surf at sunset . [ROOT]</td></tr></table>

## H Extended Experimental Protocol and Results

## H.1 ImageNet–WordNet Protocol and Per-Task Results

We use the unified 16-task stream and metric convention defined in the main paper; Appendix G.1 gives the complete task order and dataset statistics. ImageNet-1K [66] uses its oficial training split and 50,000-image validation split, and each class is associated with a WordNet synset [67]. The hierarchy evaluation additionally reports TIE, LCA, ancestor Jaccard, hierarchical precision, and hierarchical recall.

The per-task and pullback analyses below use HyCoCLIP-B. Its cumulative protected basis follows the 80% explained-variance rule, with retained rank after each completed task

2, 12, 29, 29, 29, 29, 32, 32, 32, 32, 32, 33, 33, 33, 29.

For both HMCL variants, we select one fixed configuration on seed 42 using the Overall–BWT trade-of and reuse it for the remaining reported seeds. The pullback analysis below varies only its coeficient after fixing the correction family and all other training choices.

Table 9. Final performance for every task in the extended stream (five-seed mean ± sample standard deviation). COCO and Flickr30K use symmetric R@5; all other tasks use top-1 accuracy. The two HMCL rows use the complete post-AdamW task-pullback configurations from Table 1; shaded cells denote HMCL, and boldface marks the best result for each task.
<table><tr><td>Method</td><td>Aircraft</td><td>Caltech</td><td>ImageNet</td><td>C10</td><td>C100</td><td>CLEVR</td><td>COCO</td><td>Country211</td></tr><tr><td>Vanilla</td><td> $4 . 9 4 \pm 0 . 1 0$ </td><td> $7 4 . 2 6 \pm 0 . 1 5$ </td><td> $3 9 . 4 0 \pm 0 . 0 6$ </td><td> $8 7 . 8 0 \pm 0 . 1 2$ </td><td> $4 8 . 5 2 \pm 0 . 3 3 $ </td><td> $1 4 . 3 3 \pm 0 . 2 2$ </td><td> $4 8 . 8 4 \pm 0 . 2 0$ </td><td> $4 . 4 8 \pm 0 . 1 0$ </td></tr><tr><td>EWC</td><td> $5 . 0 2 \pm 0 . 1 2$ </td><td> $7 4 . 6 3 \pm 0 . 0 8$ </td><td> $4 0 . 0 4 \pm 0 . 0 6$ </td><td> $8 8 . 1 7 \pm 0 . 1 2$ </td><td> $4 9 . 4 7 \pm 0 . 2 9$ </td><td> $1 4 . 4 0 \pm 0 . 2 0$ </td><td> $5 0 . 1 7 \pm 0 . 1 8$ </td><td> $4 . 5 6 \pm 0 . 0 9$ </td></tr><tr><td>GEM</td><td> $4 . 9 9 \pm 0 . 1 9$ </td><td> $7 5 . 0 9 \pm 0 . 2 6$ </td><td> $3 9 . 2 5 \pm 0 . 1 2$ </td><td> $8 8 . 7 1 \pm 0 . 3 1$ </td><td> $5 0 . 6 0 \pm 0 . 4 6$ </td><td> $1 5 . 0 0 \pm 0 . 9 8$ </td><td> $4 9 . 3 6 \pm 0 . 2 1$ </td><td> $4 . 6 2 \pm 0 . 0 9$ </td></tr><tr><td>C-FLAT</td><td> $4 . 8 3 \pm 0 . 1 2$ </td><td> $7 4 . 0 3 \pm 0 . 2 2$ </td><td> $3 8 . 9 3 \pm 0 . 0 1$ </td><td> $8 7 . 7 0 \pm 0 . 1 3$ </td><td> $4 8 . 8 7 \pm 0 . 1 9$ </td><td> $1 6 . 8 0 \pm 0 . 3 8$ </td><td> $4 9 . 0 8 \pm 0 . 1 4$ </td><td> $4 . 6 0 \pm 0 . 0 6$ </td></tr><tr><td>HMCLMR</td><td> $5 . 7 7 \pm 0 . 1 1$ </td><td> $7 6 . 6 3 \pm 0 . 1 4$ </td><td> ${ \bf 4 5 . 7 0 \pm 0 . 0 3 }$ </td><td> $8 9 . 5 2 \pm 0 . 0 7$ </td><td> $5 6 . 3 1 \pm 0 . 1 1$ </td><td> $2 4 . 4 9 \pm 0 . 8 9$ </td><td> ${ \bf 6 0 . 1 2 \pm 0 . 0 9 }$ </td><td> $5 . 9 2 \pm 0 . 0 1$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td> ${ \bf 5 . 8 0 \pm 0 . 0 8 }$ </td><td> ${ \bf 7 6 . 9 1 \pm 0 . 1 1 }$ </td><td> $4 5 . 5 9 \pm 0 . 0 5$ </td><td> $\mathbf { 8 9 . 5 7 \pm 0 . 0 8 }$ </td><td> ${ \bf 5 6 . 7 7 \pm 0 . 2 0 }$ </td><td> ${ \bf 2 5 . 8 6 \pm 0 . 5 0 }$ </td><td> $6 0 . 1 0 \pm 0 . 0 9$ </td><td> ${ \bf 6 . 0 9 \pm 0 . 0 4 }$ </td></tr><tr><td>Method</td><td>DTD</td><td>EuroSAT</td><td>Flickr30K</td><td>Food101</td><td>MNIST</td><td>Flowers</td><td>PCAM</td><td>SST-2</td></tr><tr><td>Vanilla</td><td> $2 4 . 4 9 \pm 0 . 1 7$ </td><td> $4 1 . 0 0 \pm 0 . 6 5$ </td><td> $7 5 . 2 3 \pm 0 . 1 9$ </td><td> $5 6 . 7 8 \pm 0 . 2 0$ </td><td> $2 8 . 4 7 \pm 1 . 2 3 $ </td><td> $3 1 . 6 8 \pm 0 . 3 3 $ </td><td> $6 2 . 9 7 \pm 0 . 1 7$ </td><td> $5 4 . 7 0 \pm 0 . 3 5$ </td></tr><tr><td>EWC</td><td> $2 4 . 5 7 \pm 0 . 3 0$ </td><td> $4 0 . 6 4 \pm 0 . 8 7$ </td><td> $7 6 . 5 7 \pm 0 . 1 5$ </td><td> $5 6 . 9 9 \pm 0 . 1 4$ </td><td> $2 8 . 4 5 \pm 0 . 9 3$ </td><td> $3 1 . 9 2 \pm 0 . 3 7$ </td><td> $6 1 . 9 2 \pm 0 . 1 6$ </td><td> $5 4 . 9 0 \pm 0 . 2 6 $ </td></tr><tr><td>GEM</td><td> $2 5 . 9 6 \pm 0 . 4 4$ </td><td> $4 7 . 9 9 \pm 1 . 5 6 $ </td><td> $7 5 . 8 1 \pm 0 . 1 9$ </td><td> $5 6 . 8 5 \pm 0 . 6 0$ </td><td> ${ \bf 3 0 . 0 4 \pm 1 . 4 6 }$ </td><td> $3 1 . 8 7 \pm 0 . 6 5$ </td><td> $6 3 . 5 8 \pm 0 . 9 1 $ </td><td> $5 5 . 0 0 \pm 0 . 2 3 $ </td></tr><tr><td>C-FLAT</td><td> $2 4 . 3 5 \pm 0 . 3 9$ </td><td> $4 0 . 5 7 \pm 0 . 7 8$ </td><td> $7 4 . 9 4 \pm 0 . 1 7$ </td><td> $5 6 . 3 4 \pm 0 . 1 6$ </td><td> $2 7 . 4 7 \pm 1 . 1 4$ </td><td> $3 1 . 1 3 \pm 0 . 2 3 $ </td><td> ${ \bf 6 3 . 7 9 \pm 0 . 2 2 }$ </td><td> $5 3 . 8 9 \pm 0 . 3 3 $ </td></tr><tr><td>HMCLMR</td><td> $2 7 . 7 1 \pm 0 . 2 8$ </td><td> $5 5 . 8 7 \pm 0 . 5 0 $ </td><td> ${ \bf 8 5 . 4 8 \pm 0 . 1 1 }$ </td><td> $6 1 . 1 0 \pm 0 . 1 1$ </td><td> $2 9 . 5 2 \pm 0 . 4 7$ </td><td> ${ \bf 3 2 . 7 4 \pm 0 . 1 5 }$ </td><td> $6 0 . 8 0 \pm 0 . 0 8$ </td><td> ${ \pm } 5 5 . 5 2 \pm 0 . 0 8$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td> ${ \bf 2 7 . 8 9 \pm 0 . 1 9 }$ </td><td> ${ \pm 6 . 4 4 \pm 0 . 3 3 }$ </td><td> $8 5 . 4 5 \pm 0 . 1 3$ </td><td> ${ \bf 6 1 . 4 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 3 0 . 4 9 \pm 0 . 5 4 }$ </td><td> $3 2 . 4 8 \pm 0 . 1 5$ </td><td> $6 0 . 7 9 \pm 0 . 0 9$ </td><td> $5 5 . 3 5 \pm 0 . 1 3$ </td></tr></table>

The two HMCL variants attain the highest final score on 15 of the 16 tasks: $\mathrm { H M C L } _ { \mathrm { C A } }$ leads on ten classification tasks, while $\mathrm { H M C L } _ { \mathrm { M R } }$ leads on ImageNet-1K, both retrieval datasets, Flowers102, and SST-2. C-FLAT remains strongest on PCAM. This per-task breakdown agrees with the aggregate comparison: after matching both the optimizer-step budget and the global replay capacity, HMCL’s gain is distributed across classification and retrieval rather than being driven by a small subset of tasks. The complete per-run matrices, aggregation outputs, and source provenance accompany the extended experiment artifact.

## H.2 Modality-Specific Radial Contraction

As a modality-specific complement to Figure 6 in the main paper, we examine MERU-L at seed 1024 under the fixed 16-task protocol. We include every dataset with a post-task forgetting interval: all 15 old tasks in the sequence, excluding only the final SST-2 task because no later learning follows it. For each old task, we sample 500 fixed test anchors and compare their spatial norms immediately after learning that task with those at the final checkpoint after the complete sequence. This gives 7,500 image and 7,500 matched text anchors per method. Figure 8 pools these anchors across all 15 old tasks; the dashed lines mark the corresponding pooled task-end medians.

Vanilla retains only 0.87× of the image norm and 0.79× of the text norm at the median. The contraction appears in the task-level medians for 14 of 15 image tasks and all 15 text tasks. In contrast, $\mathrm { H M C L } _ { \mathrm { M R } }$ retains 1.00× and 0.99×, respectively, and no task-level median falls below 0.9× for either modality. Thus, the radial component of forgetting is a systematic contraction, especially for text, rather than merely nondirectional displacement. We avoid calling it a collapse to the origin because the Vanilla distributions remain nondegenerate.

![](images/5ba06b804e0d9ad8788228c491fd5567ce600c1d702c850949bbfd0f690424e0.jpg)  
Figure 8. Modality-specific spatial-norm distributions after the complete MERU-L continual stream (seed 1024). The distributions pool 500 fixed test anchors from each of the 15 old tasks in the 16-task sequence; SST-2 is excluded only because it is last and has no post-task forgetting interval. Bars show final-checkpoint norms of old-task image and text embeddings, and color-matched dashed lines show their pooled task-end medians. The inset values report the median per-anchor ratio ∥z<sub>final,space</sub> ∥<sub>2</sub>/∥z<sub>task-end,space</sub> ∥<sub>2</sub>.

Scope of the guarantee. The after-AdamW MR correction exactly preserves only the retained subspace represented $\mathbf { V } _ { t - 1 }$ . Because PCA controls average residual energy rather than the worst individual task, small drift can remain outside that subspace; the correction also leaves AdamW’s moment states unchanged. These limits explain why the empirical hierarchy drift is strongly reduced rather than identically zero.

## H.3 Stage-Wise Stability–Plasticity Diagnostics

![](images/d801260e2e4202dc093272bbfcd471bffd13f82a6fb6493925ca46177b5b9f27.jpg)  
(a) Stability analyses.

![](images/8830e48f0771963392777de58c2e042c47e28a1179b80818e6fe54d19af915c2.jpg)  
(b) Plasticity analyses.  
Figure 9. Stage-wise stability and plasticity diagnostics replotted from the original CMCL analysis of Liu et al. [63]. At stage �, panel (a) reports the mean deviation of previously learned modality-pair alignment scores from their preceding reference values; deviations close to zero indicate stable old-pair alignment. For � > 1, panel (b) reports the mean higher-order term used to assess the source plasticity condition $o ( \eta ) / \eta \leq 0 ;$ nonpositive values satisfy that condition. These theorem-specific quantities are not BWT or task-score gains.

The old-pair alignment deviations remain between −0.074 and 0.002 and approach zero at several stages. The higher-order terms are nonpositive throughout; after the two largest early-stage terms, their magnitudes fall below 0.06. Because this diagnostic uses the original CMCL protocol, it is included here as a theorem-specific consistency check rather than as a result on the current 16-task HMCL benchmark.

## H.4 Fine-Grained Retrieval on the Modality-Extended Stream

Figure 4 in the main paper summarizes final COCO and Flickr30K retrieval in the separate modality-extension protocol, which adds one audio and one thermal retrieval task to the original continual benchmark. Table 10

Table 10. Fine-grained final retrieval performance in the separate protocol augmented with audio and thermal tasks (mean ± standard deviation over three paired seeds). T2I and I2T denote text-to-image and image-to-text retrieval, respectively. The methods and configurations match Table 2. Boldface marks the best result for each backbone and metric; shaded cells denote HMCL.
<table><tr><td colspan="7">(a) COCO</td></tr><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="3">T2I</td><td colspan="3">I2T</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="3">MERU-L</td><td>Vanilla</td><td> $9 . 3 0 3 \pm 0 . 0 7 2$ </td><td> $2 3 . 8 4 4 \pm 0 . 0 6 5$ </td><td> $3 2 . 8 3 5 \pm 0 . 1 0 8$ </td><td> $1 5 . 1 1 3 \pm 0 . 1 1 4$ </td><td> $3 3 . 6 3 3 \pm 0 . 2 2 0$ </td><td> $4 3 . 7 7 3 \pm 0 . 2 4 7$ </td></tr><tr><td>HMCLMR</td><td> $8 . 8 4 8 \pm 0 . 0 6 7$ </td><td> $2 2 . 7 8 7 \pm 0 . 1 8 3$ </td><td> $3 1 . 9 5 5 \pm 0 . 2 4 5$ </td><td> $1 5 . 8 4 7 \pm 0 . 1 2 1$ </td><td> $3 4 . 5 7 3 \pm 0 . 5 4 9$ </td><td> $4 5 . 3 5 3 \pm 0 . 1 8 6$ </td></tr><tr><td> $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ </td><td> $\mathbf { 1 0 . 7 2 5 \pm 0 . 1 2 0 }$ </td><td> $\mathbf { 2 6 . 5 5 7 \pm 0 . 1 4 3 }$ </td><td> $\mathbf { 3 6 . 3 4 4 \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 1 8 . 0 4 0 \pm 0 . 2 5 0 }$ </td><td> $\mathbf { 3 7 . 2 9 3 \pm 0 . 2 3 9 }$ </td><td> $\mathbf { 4 7 . 9 1 3 \pm 0 . 3 4 8 }$ </td></tr><tr><td rowspan="3">MERU-B</td><td>Vanilla</td><td> $9 . 4 9 1 \pm 0 . 0 3 8$ </td><td> $2 3 . 6 3 5 \pm 0 . 1 1 3$ </td><td> $3 2 . 5 5 0 \pm 0 . 1 9 5$ </td><td> $1 3 . 0 6 7 \pm 0 . 5 2 2$ </td><td> $2 9 . 2 9 3 \pm 0 . 4 4 0$ </td><td> $3 8 . 7 4 0 \pm 0 . 5 9 6$ </td></tr><tr><td>HMCLMR</td><td> $1 0 . 7 8 2 \pm 0 . 0 4 5$ </td><td> $2 6 . 4 3 9 \pm 0 . 0 4 8$ </td><td> $3 6 . 1 6 6 \pm 0 . 1 3 7$ </td><td> $1 5 . 4 1 3 \pm 0 . 1 4 0$ </td><td> $3 4 . 1 1 3 \pm 0 . 0 9 0$ </td><td> $4 4 . 5 4 0 \pm 0 . 1 5 9$ </td></tr><tr><td>HMCLCA</td><td> $\mathbf { 1 } 2 . \mathbf { 0 7 1 } \pm \mathbf { 0 . 0 6 7 }$ </td><td> $\mathbf { 2 8 . 6 0 5 \pm 0 . 0 6 8 }$ </td><td> $\mathbf { 3 8 . 4 0 8 \pm 0 . 0 6 7 }$ </td><td> $\mathbf { 1 7 . 3 6 0 \pm 0 . 2 1 6 }$ </td><td> $\mathbf { 3 6 . 9 7 3 \pm 0 . 0 4 6 }$ </td><td> $\mathbf { 4 7 . 7 8 0 \pm 0 . 2 0 3 }$ </td></tr><tr><td rowspan="3">HyCoCLIP-B</td><td>Vanilla</td><td> $2 1 . 0 7 4 \pm 0 . 2 1 2$ </td><td> $4 2 . 9 0 1 \pm 0 . 3 3 4$ </td><td> $5 4 . 3 1 8 \pm 0 . 1 0 2$ </td><td> $3 4 . 9 0 0 \pm 0 . 1 8 0$ </td><td> $6 0 . 6 4 0 \pm 0 . 3 2 2$ </td><td> $7 1 . 8 4 7 \pm 0 . 1 5 3$ </td></tr><tr><td>HMCLMR</td><td> $2 0 . 2 1 9 \pm 0 . 0 6 3$ </td><td> $4 2 . 5 7 1 \pm 0 . 2 6 4$ </td><td> $5 4 . 0 1 9 \pm 0 . 2 7 7$ </td><td> $3 6 . 9 2 7 \pm 0 . 3 9 0$ </td><td> $6 2 . 6 6 0 \pm 0 . 2 7 8$ </td><td> $7 3 . 6 6 0 \pm 0 . 2 2 3$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td> $2 3 . 8 4 3 \pm 0 . 1 3 8$ </td><td> $\mathbf { 4 7 . 0 5 6 \pm 0 . 1 6 4 }$ </td><td> $\mathbf { 5 8 . 6 4 3 \pm 0 . 0 7 7 }$ </td><td> $\mathbf { 4 0 . 7 6 7 \pm 0 . 2 5 0 }$ </td><td> ${ \bf 6 6 . 8 3 3 \pm 0 . 1 6 3 }$ </td><td> $\mathbf { 7 6 . 8 2 7 \pm 0 . 1 2 9 }$ </td></tr></table>

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="5">(b) Flickr30K</td></tr><tr><td></td><td>T2I</td><td></td><td>I2T</td><td></td></tr><tr><td rowspan="3">MERU-L</td><td></td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Vanilla</td><td> $1 2 . 7 1 3 \pm 0 . 0 4 6$ </td><td> $2 9 . 2 2 7 \pm 0 . 2 6 6$ </td><td> $3 8 . 1 8 7 \pm 0 . 2 8 0$ </td><td> $1 9 . 5 0 0 \pm 0 . 2 6 5$ </td><td> $3 9 . 1 0 0 \pm 0 . 1 7 3$ </td><td> $4 9 . 9 3 3 \pm 0 . 4 0 4$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { M R } }$   $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ </td><td> $1 1 . 1 4 7 \pm 0 . 1 3 0$   $\mathbf { 1 4 . 6 0 0 \pm 0 . 1 6 0 }$ </td><td> $2 7 . 2 0 0 \pm 0 . 3 3 3$   $\mathbf { 3 3 . 0 4 0 \pm 0 . 3 6 5 }$ </td><td> $3 6 . 0 7 3 \pm 0 . 4 8 9$   $\mathbf { 4 3 . 1 4 7 \pm 0 . 0 8 1 }$ </td><td> $2 1 . 6 6 7 \pm 0 . 3 0 6$   $\mathbf { 2 2 . 6 3 3 \pm 0 . 4 0 4 }$ </td><td> $4 1 . 0 6 7 \pm 0 . 0 5 8$   $\mathbf { 4 4 . 5 0 0 \pm 0 . 1 0 0 }$ </td><td> $5 0 . 9 3 3 \pm 0 . 2 5 2$ </td></tr><tr><td rowspan="3">MERU-B</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\mathbf { 5 4 . 8 3 3 \pm 0 . 1 5 3 }$ </td></tr><tr><td>Vanilla</td><td> $1 1 . 8 1 3 \pm 0 . 0 6 1$ </td><td> $2 7 . 7 7 3 \pm 0 . 1 9 2$ </td><td> $3 6 . 6 1 3 \pm 0 . 4 2 8$ </td><td> $1 5 . 4 0 0 \pm 0 . 2 6 5$ </td><td> $3 4 . 6 3 3 \pm 0 . 7 5 1$ </td><td> $4 3 . 2 6 7 \pm 0 . 3 2 1$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { M R } }$   $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td> $1 3 . 9 6 7 \pm 0 . 1 4 2$   $\mathbf { 1 6 . 6 7 3 \pm 0 . 2 2 7 }$ </td><td> $3 1 . 7 8 0 \pm 0 . 1 4 0$   $\mathbf { 3 5 . 0 0 0 \pm 0 . 0 8 7 }$ </td><td> $4 1 . 3 0 0 \pm 0 . 3 0 0$   $\mathbf { 4 4 . 8 3 3 \pm 0 . 1 5 1 }$ </td><td> $1 8 . 7 3 3 \pm 0 . 4 9 3$   ${ \bf 2 0 . 3 6 7 \pm 0 . 0 5 8 }$ </td><td> $4 0 . 3 6 7 \pm 0 . 5 0 3$   $\mathbf { 4 3 . 3 6 7 \pm 0 . 5 6 9 }$ </td><td> $5 0 . 7 6 7 \pm 0 . 6 0 3$ </td></tr><tr><td rowspan="3">HyCoCLIP-B</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\mathbf { 5 4 . 5 6 7 \pm 0 . 2 5 2 }$ </td></tr><tr><td>Vanilla HMCLMR</td><td> $4 5 . 0 7 3 \pm 0 . 3 9 1$   $4 5 . 8 6 0 \pm 0 . 3 3 3$ </td><td> $7 0 . 9 4 0 \pm 0 . 3 0 8$   $7 2 . 2 0 7 \pm 0 . 4 6 9$ </td><td> $7 9 . 6 9 3 \pm 0 . 0 6 1$   $8 1 . 1 4 7 \pm 0 . 2 2 5$ </td><td> $6 4 . 4 3 3 \pm 0 . 8 0 2$   $6 6 . 4 3 3 \pm 0 . 8 1 4$ </td><td> $8 6 . 1 0 0 \pm 0 . 1 7 3$   $8 7 . 1 0 0 \pm 0 . 2 6 5$ </td><td> $9 2 . 2 0 0 \pm 0 . 1 0 0$ </td></tr><tr><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td> $\mathbf { 4 9 . 6 1 3 \pm 0 . 1 4 7 }$ </td><td> $\mathbf { 7 6 . 1 6 0 \pm 0 . 2 2 3 }$ </td><td> $\mathbf { 8 4 . 2 4 0 \pm 0 . 1 4 0 }$ </td><td> $\mathbf { 6 9 . 8 0 0 \pm 0 . 3 4 6 }$ </td><td> $\mathbf { 8 9 . 1 6 7 \pm 0 . 3 5 1 }$ </td><td> $9 2 . 9 0 0 \pm 0 . 2 6 5$   $\mathbf { 9 4 . 0 0 0 \pm 0 . 2 0 0 }$ </td></tr></table>

provides the corresponding text-to-image and image-to-text Recall@{1, 5, 10} means and standard deviations for all three backbones and methods. This protocol is reported separately from the unified 16-task ImageNet experiment.

## H.5 Reproducibility Notes

Before continual training, we validated the complete cached feature store against the expected sample counts, feature and label shapes, and label ranges.

The canonical task order is Aircraft, Caltech-101, ImageNet-1K, CIFAR-10, CIFAR-100, CLEVR, COCO, Country211, DTD, EuroSAT, Flickr30K, Food-101, MNIST, Flowers-102, PCAM, and SST-2. The paired seeds are 42, 123, 456, 789, and 1024. All main results use cached 512-dimensional backbone features. ImageNet stores 1,281,167 image features and one seven-prompt text prototype per class; class prototypes are paired to image features by label inside each minibatch.

For classification tasks, AdamW uses learning rate $5 \times 1 0 ^ { - 4 }$ , zero weight decay, 10 epochs, batch size 1024, and entailment weight 0.2. Retrieval tasks use learning rate $5 \times 1 0 ^ { - 5 }$ , zero weight decay, 15 epochs, batch size 1024, and the same entailment weight. “One optimizer step per epoch” means that gradients of all minibatch losses in that epoch are accumulated before a single AdamW update; gradients are cleared only at the epoch boundary. EWC accumulates its regularized loss in the same way. GEM likewise accumulates the current and memory gradients, applies its projection once when the epoch gradient violates a constraint, and then performs one AdamW step. Thus every method receives the same epoch-level parameter-update budget.

Let $\rho$ denote retained PCA variance. The fixed HMCL settings are

<table><tr><td rowspan="2">backbone</td><td colspan="3">MR</td><td colspan="3">CA</td></tr><tr><td> $\rho$ </td><td> $\beta$ </td><td> $k _ { \mathrm { m i n } }$ </td><td> $\rho$ </td><td>β</td><td> $k _ { \mathrm { m i n } }$ </td></tr><tr><td>MERU-L</td><td>.60</td><td>.14</td><td>8</td><td>.65</td><td>.13</td><td>8</td></tr><tr><td>MERU-B</td><td>.70</td><td>.075</td><td>8</td><td>.70</td><td>.075/.20</td><td>11</td></tr><tr><td> $\mathrm { H y C o C L I P { - B } }$ </td><td>.80</td><td>.125</td><td>8</td><td>.75</td><td>.15</td><td>1</td></tr></table>

where the MERU-B CA entry .075/.20 gives classification/retrieval pullback coeficients. These configurations are fixed after selection on seed 42 and then reused without change for the other four seeds.

EWC uses coeficient $1 0 ^ { - 4 }$ and memory size 512. GEM uses margin 0.5, the exact quadprog solver, and a task-balanced global bufer capped at 512 feature pairs; after all 16 tasks it contains 32 pairs per task. C-FLAT uses Co2L contrastive distillation with power 1.0 and memory size 512, perturbation radius $\rho = 0 . 2$ , and first-order coeficient $\lambda = 0 . 2 ;$ AdamW is its wrapped base optimizer, with base AdamW on task 1 and C-FLAT activated from task 2. DNS uses its dual-sided Euclidean null-space projection under the same task order, cached inputs, schedule, and update budget. Each run writes a task-end checkpoint and partial performance matrix, and is accepted only when all 16 tasks are complete and the matrix is a finite $1 6 \times 1 6$ lower-triangular array. The experiment artifact retains the per-run configurations, matrices, checkpoints, source hashes, exclusion audit, and aggregation outputs.

## H.6 Robustness to AdamW Weight Decay

The main-table experiments use zero weight decay. To test whether the conclusions depend on this choice, we sweep $\lambda \in \{ 0 , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ on HyCoCLIP-B while holding all other training conditions fixed. The $\mathrm { M R } _ { \mathrm { p r e } }$ control applies the minimal-rotation correction to the raw gradient before AdamW and uses no pullback. We use the distinct notation $\mathrm { M R } _ { \mathrm { p r e } }$ because this control is not the complete post-AdamW, pullback-equipped HMCL<sub>MR</sub> reported in the main comparison. $\mathrm { \ H M C L _ { C A } }$ uses the fixed complete configuration from the main comparison.

Table 11. Robustness to AdamW weight decay on HyCoCLIP-B (mean ± sample standard deviation over five paired seeds). Overall averages the final scores of 14 classification and two retrieval tasks; signed BWT is final minus immediate performance, so higher is better. All conditions share the task order, learning-rate schedule, epochs, batch size, loss, and update budget; only the weight-decay coeficient � changes. $\mathrm { { M R } _ { \mathrm { { p r e } } } }$ is the raw-gradient, pre-AdamW minimal-rotation control, whereas $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ uses the complete post-AdamW correction.
<table><tr><td rowspan="2">λ</td><td colspan="3">Overall ↑</td><td colspan="3">BWT ↑</td></tr><tr><td>Vanilla</td><td> $\mathrm { M R } _ { \mathrm { p r e } }$ </td><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td><td>Vanilla</td><td> $\mathrm { M R } _ { \mathrm { p r e } }$ </td><td> $\mathrm { H M C L } _ { \mathrm { C A } }$ </td></tr><tr><td>0</td><td> $4 3 . 6 1 8 \pm 0 . 1 5 2$ </td><td> $4 6 . 4 0 0 \pm 0 . 1 8 2$ </td><td> $\mathbf { 4 8 . 5 6 6 \pm 0 . 0 2 4 }$ </td><td> $- 4 . 1 8 6 \pm 0 . 0 8 7$ </td><td> $- 2 . 9 2 9 \pm 0 . 1 0 6$ </td><td> $\mathbf { - 0 . 4 5 6 \pm 0 . 0 5 1 }$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td> $4 3 . 6 1 8 \pm 0 . 1 5 1$ </td><td> $4 6 . 3 9 8 \pm 0 . 1 8 1$ </td><td> $\mathbf { 4 8 . 5 6 5 \pm 0 . 0 2 3 }$ </td><td> $- 4 . 1 8 5 \pm 0 . 0 8 8$ </td><td> $- 2 . 9 3 1 \pm 0 . 1 0 4$ </td><td> $\mathbf { - 0 . 4 5 8 \pm 0 . 0 5 0 }$ </td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td> $4 3 . 6 1 6 \pm 0 . 1 5 1$ </td><td> $4 6 . 3 9 9 \pm 0 . 1 8 3$ </td><td> $\mathbf { 4 8 . 5 6 5 \pm 0 . 0 2 2 }$ </td><td> $- 4 . 1 8 6 \pm 0 . 0 9 0$ </td><td> $- 2 . 9 2 8 \pm 0 . 1 0 6$ </td><td> $\mathbf { - 0 . 4 6 1 \pm 0 . 0 5 4 }$ </td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td> $4 3 . 6 0 6 \pm 0 . 1 5 1$ </td><td> $4 6 . 3 8 4 \pm 0 . 1 8 0$ </td><td> $\mathbf { 4 8 . 5 6 3 \pm 0 . 0 2 5 }$ </td><td> $- 4 . 1 9 1 \pm 0 . 0 8 8$ </td><td> $- 2 . 9 4 0 \pm 0 . 1 0 3$ </td><td> $\mathbf { - 0 . 4 5 9 \pm 0 . 0 5 1 }$ </td></tr></table>

Table 11 shows negligible sensitivity throughout the tested range. From � = 0 to $1 0 ^ { - 2 }$ , the Overall score of $\mathrm { M R _ { \mathrm { p r e } } }$ remains between 46.384 and 46.400, with BWT between −2.940 and −2.928; the Overall score of $\mathrm { \ H M C L _ { C A } }$ remains between 48.563 and 48.566, with BWT between −0.461 and −0.456, and it is strongest in every condition. The relatively mild aggregate forgetting of $\mathrm { M R _ { \mathrm { p r e } } }$ is concentrated in the 14 classification tasks (BWT approximately −2.69), whereas its two retrieval tasks have BWT approximately −4.65. Hence this control confirms that minimal rotation already protects much of the old classification geometry, while the complete CA correction gives the more consistent cross-task result. Most importantly, the ordering among Vanilla, $\mathrm { M R _ { \mathrm { p r e } } }$ , and $\mathrm { { H M C L } _ { \mathrm { { C A } } } }$ is unchanged by AdamW weight decay.

## H.7 Clean Single-Stage Training Time

We measure method-side cost on ImageNet-1K, the third task after sequentially learning Aircraft and Caltech-101. For every backbone–method pair, seed 42 uses the same cached features, 10-epoch schedule, batch size, and epoch-level optimizer-step budget as the main comparison. The timer brackets the complete fit\_task call with CUDA synchronization. It therefore includes cached-tensor transfer to the device, optimization and, for HMCL, geometric correction, pullback, and the method’s task-end state update, while excluding disk cache loading, evaluation, and checkpoint serialization.

The nine displayed jobs are run serially on the same RTX A6000. The dispatcher first waits for the preceding experiment queue to exit and then requires six consecutive five-second samples with no compute process, at most 2% utilization, and at most 32 MiB allocated mem-

![](images/8a40b8bd5d1815bb84da9bee67dda7201b9b98569387f1ae0eb58394962342e7.jpg)  
Figure 10. Clean ImageNet-1K training time at the third continual stage (seed 42). Each bar is one complete synchronized fit\_task timing on a common RTX A6000 under the matched schedule; values are seconds. Jobs are serialized, and accepted runs contain no foreign GPU compute PID.

ory. During a run, it records the GPU compute PIDs, utilization, memory, temperature, and power approximately once per second. The presence of any compute PID other than the timed process invalidates that attempt and triggers a fresh idle wait and retry. No displayed attempt was invalidated or retried.

Figure 10 shows that Vanilla and the two HMCL variants remain in the same computational regime: all measurements lie between 172.4 and 194.6 seconds. Averaging the within-backbone ratios to Vanilla gives 1.022× for $\mathrm { H M C L } _ { \mathrm { M R } }$ and 1.037× for $\mathrm { H M C L } _ { \mathrm { C A } }$ . Thus, in this clean single-run diagnostic, the complete HMCL corrections add only a small average cost relative to Vanilla. Because there is one timing run per pair, the diferences below 10% should not be interpreted as a statistically resolved ranking.

## H.8 Pullback Hyperparameter Analysis

We vary $\beta$ only after fixing the post-optimizer MR correction, task order, protected subspace, and all training hyperparameters. The selected setting uses $\beta = 0 . 1 2 5$ from a screening grid spanning 0.05 to 0.25. Relative to the matched $\beta = 0 . 2 0$ setting, reducing $\beta$ to 0.125 improves the Overall score by 0.6% and mean classification accuracy by 1.2%, while retrieval R@5 decreases by 2.1%. It also worsens forgetting, BWT, and final ImageNet retention. The pullback coeficient therefore selects the stability–plasticity operating point and is not treated as a method component in the main ablation.