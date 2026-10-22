# Compositionality with Primitives: A Unified View from Euclidean to Hyperbolic Vision-Language Models

Puru Vaish<sup>1</sup> Meelad Dashti<sup>1,3</sup> Christoph Brune<sup>1</sup> Pascal Mettes<sup>2</sup> Tatiana Tommasi<sup>3</sup> Nicola Strisciuglio<sup>1</sup>

<sup>1</sup>University of Twente, The Netherlands <sup>2</sup>University of Amsterdam, The Netherlands <sup>3</sup>Politecnico di Torino, Italy

## Abstract

Compositional zero-shot learning (CZSL) aims to recognize novel combinations of previously observed objects and attributes. Prior work showed that large-scale vision–language models encode decomposable semantic structures, that can be recomposed to enable generalization. The Geodesically Decomposable Embeddings (GDE)framework formalizes this by identifying directions in the tangent space at the Fréchet mean of the embedding distribution, which encode primitive visual concepts. Unseen attribute–object compositions can be synthesized through vector addition of these primitives. We revisit the assumption that compositional generalization requires explicit attribute-object pair re-composition. By means of new compositional scoring functions and extensive evaluations, we show that directly using the decomposed primitives suffices for recognizing unseen compositions. In that, we gain insights on how the geometry ofthe embedding manifold (Euclidean, spherical, or hyperbolic) and dataset structure affect compositional reasoning. Across models (CLIP, DINOv2, MERU, HyCoCLIP), we show that compositional structure consistently emerges as primitive representations, enabling inference on CZSL benchmarks (MIT-States, VAW-CZSL, UT-Zappos50K, Cloth ing16K) and group robustness on Waterbirds and CelebA.<sup>1</sup>

## 1. Introduction

Recognizing a sliced banana after seeing only sliced bread and whole bananas requires disentangling the attributes sliced and whole from the objects bread and banana, and reasoning over them independently. This ability reflects the broader principle of compositionality, a fundamental cognitive skill of humans clearly arising in language [14], whereby complex concepts are constructed from simpler, reusable ones [24]. At its core, it underpins the ability to generalize robustly to novel compositions without requiring direct supervision [14].

In Machine Vision, this paradigm is formalized as compositional zero-shot learning (CZSL) [20, 21]: given training images depicting a closed set of attribute-object pairs, the aim is to enable a model to recognize novel compositions of known concepts without having seen them during training. Open-world CZSL [18] extended this setting to the full Cartesian product of attributes and objects, significantly increasing task difficulty, and accounting for the long-tailed distribution of real-world generalization. Existing solutions typically attempt to learn compositional representations using conditional formulations of attribute-object representations [9, 35, 36], or learning compositional soft-prompt [17, 22], visual proxies [41] or invariant representations [42].

Recent work on Geodesically Decomposable Embeddings (GDE) [2] provides a different perspective: vision embeddings of pre-trained vision-language models already exhibit compositional semantic structures. These can be disentangled to perform compositional zero-shot classification. By projecting representations into the tangent space at the Fréchet mean of the embedding distribution of a dataset and performing a factorization step, attribute and object representations emerge as vector directions that embed basic visual concepts, called primitives. These can be recombined by vector arithmetics in the tangent space, thus synthesizing the representation of unseen object-attribute pairs and enabling their recognition. However, the focus in [2] was limited to CLIP spherical embeddings [28]. It remains unclear whether this view of compositionality extends to other space geometries, i.e. spaces with flat (Euclidean) or negative (hyperbolic) curvature, and whether explicit composition of primitives into pairs is necessary to achieve open-world compositional generalization. This question gains particular relevance for hyperbolic models [5, 23], where embeddings lie on the Lorentz space and representations encode compositional hierarchy in a tree-like fashion.

![](images/dd9304b4e8e09b93eb85300a356c69f94603636b8305b576aad6dbd83caea351.jpg)  
Figure 1. Geodesics initiating from the origin across Euclidean, spherical, and hyperbolic manifolds. The plots illustrate the structural differences imposed by space curvature: (a) Euclidean space is flat; (b) spherical space has positive curvature with origin estimated on the Fréchet mean of the dataset embeddings; (c) Hyperbolic space has negative curvature with a well-defined origin, resulting in tree-like hierarchical representation space. Data clusters are differentiated by color and marker type ({red, blue} × {square, circle}).

In this work, we place primitives at the center of the analysis and revisit compositional reasoning in VLMs. By means of novel compositional scoring functions, we demonstrate that primitives can be used directly to perform compositional inference without the need of explicitly constructing (unseen) pair representations. We demonstrate improved open-world compositional generalization, across Euclidean, spherical, and hyperbolic representation spaces, which hold different geometric and data distribution properties (see Figure 1). To perform this extensive investigation, we build upon the GDE framework and leverage its formulation beyond the original spherical setting of CLIP [28], to a broad family of models across geometries. We investigate Euclidean DINOv2 [1, 19] and hyperbolic VLMs (MERU [5], HyCoCLIP [23]), by utilizing corresponding manifold mapping functions between these geometries and the tangent space.

We also develop a formulation of primitives directly on the embedding manifold as the Fréchet means of samples sharing an attribute or object, and call them intrinsic primitives. We thus study manifold-native primitives as well as tangent-space GDE primitives, investigating how latent space geometry influences compositional abilities of models.

Across experiments, we find that compositional structure is a general emergent property of pre-trained models, showing through both GDE and intrinsic primitive representations across several CZSL benchmarks (MIT-States, VAW-CZSL, Clothing16K, UT-Zappos50K), with stronger evidence for datasets that contain samples with higher semantic variance. We further show that primitive-based inference is beneficial for improving recognition under group bias in the data through experiments on benchmark datasets for group robustness (Waterbirds, CelebA) using solely the object primitives, decoupled from the attribute primitives. The observed substantial and consistent increase of worst group accuracy indicates that biases and spurious correlations induced by attributes can be removed with primitive-based reasoning. The contributions of this work are therefore summarized as:

(i) Primitives as the basis for open-world compositional learning and generalization. We show that pretrained models consistently organize representation into primitives, either as tangent-space directions or intrinsic points on manifolds, which encode basic visual concepts of attributes and objects. We develop new compositional scoring functions and demonstrate with extensive experiments that primitives support direct CZSL inference and open-world generalization, making explicit attribute-object pair construction unnecessary.

(ii) Understanding of compositional power of pretrained models in relation to manifold geometry and dataset intrinsic variance. We find that, while compositional structure is geometry-agnostic, its task-wise optimal representation depends on the underlying geometry: tangent-space primitives are sufficient in Euclidean and spherical spaces, whereas hyperbolic representations benefit from intrinsic primitives aligned with their space geometry. We further shed light on the role that CZSL datasets and their intrinsic variance have in factorizing compositional primitives in pretrained models.

## 2. Related Work

Compositional zero-shot learning in VLMs. Early compositional zero-shot learning (CZSL) methods modeled attribute-object relations through shared embedding spaces [10, 38]. More recent studies have shifted attention to the compositional properties of Vision-Language models and highlighted that contrastive VLMs inherently struggle with text-based composition during standard inference, often behaving as a “bag of words” [34, 39]. This vulnerability is further underscored by benchmarks like SugarCrepe [8], which demonstrate that VLMs frequently rely on statistical hacks rather than genuine compositionality. In order to improve the compositional reasoning abilities of VLMs, subsequent research has explored adaptation strategies based on soft-prompt tuning [17, 22], or learning conditional attributeobject representations, where attribute frequency is coupled with the most likely co-occurring objects [9, 35, 36].

In [2], it was hypothesised that the pre-trained vision encoder of VLMs natively encodes compositional structures, induced by language-based pre-training. Thus, Geodesically Decomposable Embeddings (GDE) was proposed to factorize primitive representation directions in the tangent space at the Fréchet mean of the embeddings distribution, enabling training-free compositional reasoning via constructing unseen combination of primitives. The focus was restricted to spherical (CLIP-like) embeddings and to explicit construction of object-attribute pair representations for CZSL.

In this work, we fundamentally revisit the assumption that explicit pair re-composition is necessary for CZSL inference. We formulate new scoring functions for compositional classification that operate directly on independent object and attribute primitives. We show that leveraging these primitive representations is sufficient to recognize unseen compositions, bypassing pair recombination entirely. Furthermore, we expand the GDE framework to work across different geometries of the latent space (i.e. Euclidean, spherical and hyperbolic) and demonstrate the emergence of compositional structures in a broad family of visual encoders.

Our work differs from fine-tuning [17, 22] and conditional attribute learning [9, 35, 36], as we do not add learnable components and study the embedding capabilities of pretrained VLMs. It also shares a philosophical connection with Concept Bottleneck Models (CBMs) [13], which enforce intermediate, primitive-based concept representations for downstream inference. However, while CBMs require explicit bottleneck layers or fine-tuning with concept supervision, we analyze the compositional structure that inherently emerges within the embedding space of pretrained models.

Compositionality in hyperbolic VLMs. Most visionlanguage models operate on the unit hypersphere using cosine similarity. In contrast, hyperbolic models embed representations on the Lorentz model [5, 23, 27, 29, 37], where radius reflects semantic specificity, and geometry captures hierarchical structure: abstract concepts are placed closer to the origin, while more complex detailed concepts lie further away. HyCoCLIP [23] further incorporates compositional entailment supervision to align scene-object compositions with the hierarchical organization of the embedding space. While prior work has primarily focused on retrieval, entailment, and hierarchy preservation, we instead investigate how hyperbolic geometry shapes compositional reasoning. We show that primitives naturally align with the manifold structure, and that for group robustness intrinsic primitives provide a more semantically-coherent representation than tangent-space approximations in hyperbolic settings.

Compositionality in vision-only DINO. Compositional capabilities are not exclusive to cross-modal vision-language models. Self-supervised vision-only foundation models, such as DINOv2 [19], capture exceptional fine-grained, localized spatial semantics without using any language-signal during training. Recent efforts have explored post-hoc mechanisms to interface language with these frozen visual encoders. For instance, Talk2DINO [1] demonstrated that text embeddings can be aligned, via a learned projection, onto DINO patch-level features to guide open-vocabulary tasks. These findings suggest that a structured semantic organization emerges within the token embeddings of vision-only pre-trained encoders. Understanding the geometry underlying this intrinsic compositionality is therefore crucial, as such models provide powerful representations that are not influenced by language biases. In this work, we apply the GDE framework to DINOv2 and show that compositional structure naturally emerges in its (Euclidean) latent space, despite the absence of language supervision.

## 3. Preliminaries

In this section, we provide a brief overview of the mathematical background relevant to the notion of compositional embeddings and the GDE framework. Moreover, we provide the formalization of open-world CZSL and pair-based scoring functions used for recognition, and cover the foundational alignment of DINOv2 patch embeddings with text embeddings of CLIP done via the Talk2DINO framework.

## 3.1. Geodesically Decomposable Embeddings

VLMs encode images $x \in \mathcal { X }$ and their associated caption $y \in \mathcal { V }$ into shared embedding spaces via specific encoders $\phi _ { i m } : \mathcal { X } \to \mathcal { R } ^ { d } , \phi _ { t x } : \mathcal { Y } \to \mathcal { R } ^ { d }$ . Let $\mathcal { M } \subset \overline { { \mathcal { R } } } ^ { d }$ be the target d-dimensional representation manifold, with $\mathbf { u } \in \mathcal { M }$ being a generic point on the manifold. The text prompts can be created by a general template function $g : \mathrm { L } ^ { n }  \mathcal { V } \quad \forall n \in$ N where $\mathrm { L } ^ { n }$ are the n concepts to make a template for.

In the Geodesically Decomposable Embeddings (GDE) framework [2], visual representations from pre-trained VLMs are decomposed into primitive directions corresponding to objects and attributes, which can then be recombined for CZSL. Given N samples belonging to a dataset, their intrinsic (Fréchet) mean is $\begin{array} { r } { \pmb { \mu } = \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { M } } \sum _ { i = 1 , \dots N } d _ { \mathcal { M } } ( \mathbf { u } , \mathbf { u } _ { i } ) ^ { 2 } } \end{array}$ where $d _ { \mathcal { M } }$ denotes the geodesic distance on the manifold. Generally, a Riemannian gradient descent approach with a finite number of iterations is employed to compute the Fréchet mean on any manifold geometry (see Algorithm 1 in the supplementary material). Let $T _ { \mu } { \mathcal { M } }$ be the tangent space to the embedding manifold at $\mu ,$ and $\mathbf { t } _ { i } = \mathrm { L o g } _ { \mu } ( \mathbf { u } _ { i } ) \in T _ { \mu } { \mathcal { M } }$ the projection of an embedding vector to the tangent space via the logarithmic map. The function is locally invertible, and its inverse is the exponential map $\mathbf { u } _ { i } = \mathrm { E x p } _ { \mu } ( \mathbf { t } _ { i } ) \in \mathcal { M }$ . The exact form of the map functions depends on the manifold geometry detailed in Section 4.

Given a set of attributes $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { n } \}$ and objects $\mathcal { O } = \{ o _ { 1 } , o _ { 2 } , \dots , o _ { m } \}$ , GDE estimates an independent, isolated primitive semantic direction $\mathbf { v } _ { c }$ for any concept $c \in \mathcal { A } \cup \mathcal { O }$ via a weighted average of the projected tangent vectors containing that concept:

$$
\mathbf { v } _ { c } = \frac { \sum _ { i \in S _ { c } } w _ { i } \mathbf { t } _ { i } } { \sum _ { i \in S _ { c } } w _ { i } } , \qquad S _ { c } = \{ i : y _ { i } = c \} ,\tag{1}
$$

where $y _ { i }$ denotes the attribute or object label associated with sample i, and each sample is labelled by a compositional pair $( a _ { i } , o _ { i } ) \in \mathcal { A } \times \mathcal { O }$ . Following [2], each embedding is weighted according to its alignment with the corresponding attribute–object text representation

$$
w \propto \exp \Big ( \tau s _ { \mathcal { M } } \big ( \phi _ { i m } ( x ) , \phi _ { t x } ( g ( a , o ) ) \big ) \Big ) ,
$$

where τ is a temperature parameter and $s _ { \mathcal M } ( \cdot , \cdot )$ is a similarity function defined as a geometry-specific inner product.

![](images/0a1b17c455fc1ef76a1cfd08ed3f1291f6150f9702e2271b6941817d94dd57e8.jpg)  
Figure 2. Illustration of decomposition and primitive-based methods. For unseen samples of Blue Squares, the goal is to obtain meaningfu representations of the concepts Blue and Square from seen training samples. Then, the primitive directions in the tangent space $( \mathbf { v } _ { o }$ and $\mathbf { v } _ { a } )$ can independently be compared to the test samples projected to the same space. Alternatively, it is possible to calculate the intrinsic mean fo each concept label directly on the manifold $( \mu _ { a }$ and $\mu _ { o }$ and use it as a reference for the test sample via a manifold-specific inner product to estimate similarity. The pair-based alternatives proposed in previous work either rely on the projection on the manifold of the summed primitives in the tangent space $( \mathbf { v } _ { o } + \mathbf { v } _ { a } )$ , or leverage the textual object-attribute compositional description embedded in the manifold.

## 3.2. Open-World CZSL and Pair-based Scoring

In the open-world CZSL (OW-CZSL) setting [18], models are evaluated over the full Cartesian product ${ \mathcal { A } } \times { \mathcal { O } }$ of possible compositions at test time, regardless of which attribute-object pairs were observed during training. This removes the disambiguation advantage of closed candidate sets and provides a stronger test for compositional generalization. For the zero-shot inference baseline, we prompt VLMs using a pre-defined text template in the form of ‘a photo $\mathrm { ~ \textsf ~ { ~ o ~ f ~ } ~ a n ~ }$ {attribute} {object}’, which is embedded via the text encoder and used to score the vision embeddings $\phi _ { i m } ( x )$ . The scores computed using the function $s _ { \mathcal { M } }$ are conventionally converted into probabilities through a softmax over the valid label space. The baseline Zero-Shot Pair paradigm is based on the scoring function (Eq. 2), which calculates the probability assigned to an attribute-object pair and the prediction is obtained by selecting the label with max-probability (Eq. 3).

$$
\begin{array} { r } { s ( { a } , o | x ) = s _ { \mathcal { M } } \bigl ( \phi _ { i m } ( x ) , \phi _ { t x } \bigl ( g ( a , o ) \bigr ) \bigr ) , } \end{array}\tag{2}
$$

$$
( \hat { a } , \hat { o } ) = \underset { ( a , o ) \in \mathcal { A } \times \mathcal { O } } { \arg \operatorname* { m a x } } p ( a , o \mid x ) ,
$$

$$
p ( a , o \mid x ) = { \frac { \exp ( s ( a , o | x ) ) } { \sum _ { a ^ { \prime } \in { \mathcal { A } } } \sum _ { o ^ { \prime } \in O } \exp ( s ( a ^ { \prime } , o ^ { \prime } | x ) ) } } .\tag{3}
$$

Rather than using the text prompts, the GDE paper [2] proposed to combine all possible attribute-object primitive pairs and re-project them from the tangent space to the embedding manifold. For the GDE Pair paradigm, compositional recognition is achieved by using the scoring function:

$$
\begin{array} { r } { s ( \boldsymbol { a } , o | \boldsymbol { x } ) = s _ { \mathcal { M } } ( \phi _ { i m } ( \boldsymbol { x } ) , \mathrm { E x p } _ { \mu } ( \mathbf { v } _ { \boldsymbol { a } } + \mathbf { v } _ { o } ) ) . } \end{array}\tag{4}
$$

Here we raise the question of whether exhaustive pair recombination of primitives is necessary for OW-CZSL or if scores derived from single concept primitive are sufficient and perhaps better instead. However, we must first define scoring with primitives and their computation.

## 4. Primitive-based CZSL

The core focus of this work is the representation of visual concept as primitives (i.e. isolated attributes and objects representations), that form fundamental building blocks of compositional generalization. Rather than evaluating compositionality via pre-defined textual or visual pairs, we investigate how individual primitives emerge within different manifold geometries. Specifically, we study two distinct approaches for extracting these representations: intrinsic primitives computed directly on the underlying manifold via Riemannian operations, and tangent primitives extracted as localized linear vector directions via the Geodesically Decomposable Embeddings (GDE) framework. Under both paradigms, inference is performed zero-shot by scoring test images directly against these standalone primitives in their native mathematical domains. This eliminates the need for explicit primitive re-composition, currently used as standard in GDE. Figure 2 illustrates the differences between intrisic primitives, GDE tangent primitives, and GDE-Pair compositions on the hyperbolic geometry as a representative case.

## 4.1. Intrinsic Primitives via Fréchet Means

We define an intrinsic primitive as the manifold mean of samples sharing the same attribute or object label. Given a set of embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset { \mathcal { M } }$ , and a concept $c ,$ it is computed as the Fréchet mean of the subset, $S _ { c } ,$ associated with that concept:

$$
{ \pmb \mu } _ { c } = \arg \operatorname* { m i n } _ { { \bf u } \in \mathcal { M } } \sum _ { i \in S _ { c } } d _ { \mathcal { M } } ^ { 2 } ( { \bf u } , \phi _ { i m } ( x _ { i } ) ) ,\tag{5}
$$

where $d _ { \mathcal { M } }$ denotes the geodesic distance on the manifold. We compute separate intrinsic primitives for attributes and objects, using the Riemannian gradient descent for non-

Euclidean spaces and the arithmetic mean for Euclidean ones, yielding respectively $\mu _ { a }$ for each attribute $a \in { \mathcal { A } }$ and $\mu _ { o }$ for each object $o \in \mathcal { O }$

Given a test image $x ,$ denoting its embedding as $\phi _ { i m } ^ { x }$ for brevity, its similarity to the primitives is computed directly on the underlying manifold using the geometryspecific similarity function $s _ { \mathcal { M } }$ . In particular, for a reference pair of object (o) and attribute (a) and text template $g _ { c } : = g ( c ) \quad c \in \{ a , o \}$ with its corresponding text embedding $\phi _ { t x } ^ { c }$ , the Zero Shot Primitive score is defined as:

$$
\begin{array} { r } { s ^ { \mathrm { z s p } } ( a , o \mid \cdot ) = s _ { \mathcal { M } } ( \phi _ { i m } ^ { x } , \phi _ { t x } ^ { a } ) + s _ { \mathcal { M } } ( \phi _ { i m } ^ { x } , \phi _ { t x } ^ { o } ) . } \end{array}\tag{6}
$$

In contrast, our scoring function that uses intrinsic primitive individually does not rely on the text prompt embeddings:

$$
s ^ { \mathrm { i n t r p } } ( a , o \mid \cdot ) = s _ { \mathcal { M } } ( \phi _ { i m } ^ { x } , \pmb { \mu } _ { a } ) + s _ { \mathcal { M } } ( \phi _ { i m } ^ { x } , \pmb { \mu } _ { o } ) .\tag{7}
$$

The similarity functions used for the underlying spherical, Euclidean and hyperbolic manifolds are respectively

$$
\begin{array} { r l r } & { } & { s _ { \mathbb { S } ^ { d - 1 } } ( \mathbf { x } , \mathbf { y } ) = \mathbf { x } ^ { \top } \mathbf { y } , \qquad s _ { \mathbb { R } ^ { d } } ( \mathbf { x } , \mathbf { y } ) = \frac { \mathbf { x } ^ { \top } \mathbf { y } } { \left\| \mathbf { x } \right\| \left\| \mathbf { y } \right\| } , } \\ & { } & { s _ { \mathbb { H } ^ { d + 1 } } ( \mathbf { x } , \mathbf { y } ) = \displaystyle \sum _ { k = 1 } ^ { d } x _ { k } y _ { k } - x _ { 0 } y _ { 0 } . } \end{array}
$$

## 4.2. GDE Primitives on any Manifold

We exploit the GDE framework on different manifold geometries by integrating the logarithmic map $\operatorname { L o g } _ { \mu } : \mathcal { M } \to T _ { \mu } \mathcal { M }$ and the exponential map $\mathrm { E x p } _ { \mu } : T _ { \mu } \mathcal { M } \stackrel { . } {  } \mathcal { M }$ specifically defined for spherical, Euclidean and Lorentz spaces.

For Euclidean manifolds $\mathbb { R } ^ { d } .$ , the tangent space at every point is naturally identified with the manifold itself. In this case, the logarithmic and exponential maps reduce to ordinary subtraction and addition:

$$
\mathrm { L o g } _ { \mu } ( { \mathbf { u } } ) = { \mathbf { u } } - \mu \qquad \mathrm { E x p } _ { \mu } ( { \mathbf { t } } ) = \mu + { \mathbf { t } } .\tag{8}
$$

For spherical manifolds $\mathbb { S } ^ { d - 1 }$ , the logarithmic and exponential maps are:

$$
\begin{array} { r l } & { \mathrm { L o g } _ { \mu } ( \mathbf { u } ) = \operatorname { a r c c o s } ( \mu ^ { \top } \mathbf { u } ) \frac { \mathbf { u } - ( \mu ^ { \top } \mathbf { u } ) \mu } { \left\| \mathbf { u } - ( \mu ^ { \top } \mathbf { u } ) \mu \right\| } , } \\ & { \mathrm { E x p } _ { \mu } ( \mathbf { t } ) = \cos ( \left\| \mathbf { t } \right\| ) \pmb { \mu } + \sin ( \left\| \mathbf { t } \right\| ) \frac { \mathbf { t } } { \left\| \mathbf { t } \right\| } . } \end{array}\tag{9}
$$

For hyperbolic models, with embeddings lying on the Lorentz space, $\mathbb { H } ^ { d + 1 }$ the logarithmic and exponential maps are:

$$
\begin{array} { r l } & { \mathrm { L o g } _ { \pmb { \mu } } ( \mathbf { u } ) = \frac { \mathrm { a r c c o s h } \big ( - \langle \pmb { \mu } , \mathbf { u } \rangle _ { \mathcal { L } } \big ) } { \sqrt { \langle \pmb { \mu } , \mathbf { u } \rangle _ { \mathcal { L } } ^ { 2 } - 1 } } \Big ( \mathbf { u } + \langle \pmb { \mu } , \mathbf { u } \rangle _ { \mathcal { L } } \pmb { \mu } \Big ) , } \\ & { \mathrm { E x p } _ { \pmb { \mu } } ( \mathbf { t } ) = \mathrm { c o s h } ( \| \mathbf { t } \| _ { \mathcal { L } } ) \pmb { \mu } + \mathrm { s i n h } ( \| \mathbf { t } \| _ { \mathcal { L } } ) \frac { \mathbf { t } } { \| \mathbf { t } \| _ { \mathcal { L } } } , } \end{array}
$$

where $\| \mathbf { t } \| _ { \mathcal { L } } = \sqrt { \langle \mathbf { t , t } \rangle _ { \mathcal { L } } }$

(10)

The similarity functions for each geometry are the same defined at the end of Section 4.1.

The primitive directions $\left( \mathbf { v } _ { a } , \mathbf { v } _ { o } \right)$ , obtained via the GDE framework, and the projected test image representation $\mathbf { t } _ { x } = \mathrm { L o g } _ { \mu } ( \phi _ { i m } ( x ) )$ all lie in the same tangent space $T _ { \mu } { \mathcal { M } }$ where relationships are linear. We thus, in contrast to Eq. 4, define scoring functions for CZSL using individual GDE primitives, in which attribute and object scores are computed via standard cosine similarity in the tangent space as:

$$
s ^ { \mathrm { t a n } } ( a , o \mid x ) = s _ { \mathbb { R } ^ { d } } ( \mathbf { t } _ { x } , \mathbf { v } _ { a } ) + s _ { \mathbb { R } ^ { d } } ( \mathbf { t } _ { x } , \mathbf { v } _ { o } ) ,\tag{11}
$$

bypassing manifold projections entirely during inference.

## 5. Experiments

## 5.1. Setup

Datasets. We evaluate compositional recognition on four datasets spanning varying levels of semantic diversity and attribute-object coupling. MIT-States [10] contains 115 attributes paired with 245 objects, yielding 28,175 open-world compositions. VAW-CZSL [26, 31] is substantially larger, with 440 attributes and 541 objects (238,140 candidate compositions) sampled from images with larger scene context. UT-Zappos [38] is focused on footwear with 16 material attributes and 12 shoe types, while Clothing16K provides a broader clothing-domain benchmark with greater intracategory variability. Together, these datasets allow us to study compositionality under both highly diverse semantic settings (MIT-States, VAW-CZSL) and more constrained cases where sample variance is small and attributes are tightly coupled to object identity (UT-Zappos, Clothing16K). Models and baselines. We investigate models operating in Euclidean, spherical, and hyperbolic embedding spaces. Spherical models include CLIP: ViT-B/16 (GRIT dataset) and ViT-L/14 (OpenAI). For Euclidean representations, we use DINOv2 with Talk2DINO [1] bridge to the text embedding of CLIP. This preserve the native Euclidean embeddings of DINOv2 and enables to compute baseline results with CLIP-style text prompts (see Appendix B). We also experiment with MERU [5] and HyCoCLIP [23] hyperbolic models. CLIP-B, MERU-B, and HyCoCLIP-B share the same ViT backbone and are pretrained on the GRIT corpus [25], thus allowing to attribute performance differences primarily to the underlying manifold geometry.

Prompts and inference paradigms. We test the compositional zero-shot learning performance of the considered vision-language models under different settings, namely:

(i) ZS-Pair: direct zero-shot classification using pair-based text prompts A photo of an {attribute} {object} as defined in Eq. 2;

(ii) ZS-Prim: primitive-level zero-shot inference using text prompts with attribute-focus (A photo of an {attribute}) and object-focus (A photo of an {object}) using the scoring function defined in Eq. 6;

(iii) GDE-Pair: composing pair representations by adding visual primitives in the tangent space as $\mathbf { v } _ { a , o } = \mathbf { v } _ { a } + \mathbf { v } _ { o } ,$ and scoring the composed vector as defined in [2];

(iv) GDE-Prim: scoring the GDE object and attribute primitives independently using our function defined in Eq. 11;

(v) Int-Prim: scoring the intrinsic object and attribute primitives independently using our function defined in Eq. 7. Metrics. We compute the unseen pair accuracy (UPA) that evaluates the generalization to novel attribute-object compositions, and the seen pair accuracy (SPA) that measures the recognition performance on compositions observed during training (we report SPA in Appendix G). Models often exhibit a bias toward seen compositions, especially in the open-world setting where all feasible compositions are considered candidate predictions. Evaluating only one regime can provide a misleading picture of overall performance. Hence, as standard practice in OW-CZSL [18], we report the best harmonic mean of seen and unseen compositions:

$$
\mathrm { H M } = \frac { 2 \cdot \mathrm { S P A } \cdot \mathrm { U P A } } { \mathrm { S P A } + \mathrm { U P A } } ,
$$

that penalizes models that perform well on only one subset and therefore captures the ability to maintain strong recognition on both seen and unseen compositions simultaneously.

## 5.2. Results: CZSL via Primitives

Primitives improve open-world generalization. In Table 1 we reports the Harmonic Mean (HM) and Unseen Pair Accuracy (UPA) results of the CZSL performance of the considered pretrained models. Across models and datasets, replacing GDE pair composition with primitive-based inference consistently improves both metrics, indicating that explicit pair re-composition is not necessary for compositional generalization. On MIT-States, GDE primitives improve HM over GDE pairs by +1.53 for CLIP ViT-L/14, +1.66 for DINOv2, +1.89 for CLIP-B, +2.99 for MERU-B, and +3.66 for HyCoCLIP-B. The gains are often larger for UPA, including +3.85 for MERU-B and +5.04 for HyCoCLIP-B, indicating that primitives specifically strengthen open-world generalization. Notably, DINOv2 also benefits from primitive decomposition despite weak zero-shot pair prediction performance, improving HM by +1.65 on MIT-States and +0.90 on VAW-CZSL. As DINOv2 is trained without imagetext contrastive supervision, this suggests that compositional semantics intrinsically emerge also from large-scale visiononly self-supervision. Across geometries, intrinsic primitives provide the most stable formulation, particularly in hyperbolic space, consistently with representing visual concept directly on the embedding manifold rather than through tangent-space approximations. Overall, primitive representations provide a stronger tool for open-world compositional generalization than explicit pair composition.

Dataset structure affects compositional reasoning. UT-

![](images/bed14850bc9485dc731f72afba9349fe987feb8cc9c60b95fee3d174425a1cad.jpg)

![](images/3ee5d2e48541afa6792f1a403b6f6b6377c5311d29b218451b87024a5be2281e.jpg)  
Figure 3. Intrinsic dataset variance measured using squared geodesic distance to the dataset Fréchet mean across different geometries. From left to right: DINOv2 ViT-L/14 with Euclidean geometry, CLIP ViT-L/14 with spherical geometry, and HyCoCLIP-B with hyperbolic geometry. UT-Zappos exhibits substantially lower variance than the broader datasets, while geometry-specific whitening increases the variance of UTZ-W toward the same regime.

Zappos is a consistent exception to the advantages observed for primitive-based CZSL inference on the broader CZSL datasets. The dataset covers a narrow semantic domain and its samples have strongly coupled attribute–object relationship, with many attributes appearing across a restricted set of shoe categories (see Figure 6 in the Appendix). We therefore test whether effectiveness of primitive-based compositional inference depends on the semantic spread of the embedding distribution: we investigate the intrinsic dataset variance and causally link its value to compositional generalization performance. For embeddings $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ from dataset D on manifold M, we compute the intrinsic variance

$$
V ( \mathcal { D } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } d _ { \mathcal { M } } ^ { 2 } \left( \mathbf { x } _ { i } , \pmb { \mu } _ { \mathcal { D } } \right) ,\tag{12}
$$

where $\pmb { \mu } _ { \mathcal { D } }$ is the Fréchet mean and $d _ { \mathcal { M } } ( \cdot , \cdot )$ is the geodesic distance. Figure 3 reveals a similar pattern across Euclidean, spherical, and hyperbolic representations: MIT-States, VAW-CZSL, and Clothing16K have substantially greater intrinsic variance than UT-Zappos. This suggests that primitive decomposition benefits from representations spanning sufficiently diverse semantic contexts, whereas low-variance distributions allow pair representations to exploit stronger dataset-specific attribute-object co-occurrence patterns.

We study the role of intrinsic dataset variance as a causal factor to explain compositional reasoning limitations. We apply a geometry-specific whitening operation (detailed in Appendix E) to the embeddings of the considered models of the UTZappos dataset, and denote the resulting representations as UTZ-W. Whitening has the effect of increasing the intrinsic variance of UTZ-W (similarly to that of the broader datasets) compared to that of the original UT-Zappos.

We report CZSL results before and after whitening in

Table 1. Harmonic Mean (HM) and Unseen Pair Accuracy (UPA) results across models and datasets. Primitive-based inference consistently improves compositional generalization over GDE pair composition, particularly on unseen attribute–object pairs.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="5">Harmonic Mean (%, ↑)</td><td colspan="5">Unseen Pair Accuracy (%, ↑)</td></tr><tr><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>8.03</td><td>10.56</td><td>9.37</td><td>10.90</td><td>10.40</td><td>8.04</td><td>13.03</td><td>8.62</td><td>11.98</td><td>11.75</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>3.00</td><td>3.69</td><td>6.56</td><td>8.22</td><td>8.04</td><td>2.88</td><td>4.78</td><td>6.26</td><td>9.33</td><td>9.42</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>4.61</td><td>4.70</td><td>4.75</td><td>6.64</td><td>6.38</td><td>4.24</td><td>5.36</td><td>4.80</td><td>7.18</td><td>7.12</td></tr><tr><td>MIT-ttes</td><td>MERU-B</td><td>4.68</td><td>4.74</td><td>3.58</td><td>6.57</td><td>6.36</td><td>4.40</td><td>5.54</td><td>3.40</td><td>7.25</td><td>7.40</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>5.26</td><td>6.53</td><td>4.28</td><td>7.94</td><td>7.54</td><td>4.67</td><td>8.20</td><td>3.71</td><td>8.75</td><td>8.76</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>0.74</td><td>1.51</td><td>1.18</td><td>1.77</td><td>1.77</td><td>0.79</td><td>1.59</td><td>1.32</td><td>2.07</td><td>2.22</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>0.72</td><td>1.22</td><td>1.30</td><td>2.05</td><td>2.20</td><td>0.77</td><td>1.57</td><td>1.63</td><td>2.60</td><td>2.81</td></tr><tr><td>VA-CSL</td><td>CLIP-B (GRIT)</td><td>0.26</td><td>0.72</td><td>0.96</td><td>1.52</td><td>1.49</td><td>0.41</td><td>0.96</td><td>1.30</td><td>2.13</td><td>1.95</td></tr><tr><td></td><td>MERU-B</td><td>0.36</td><td>0.77</td><td>0.86</td><td>1.49</td><td>1.34</td><td>0.49</td><td>0.96</td><td>1.00</td><td>1.97</td><td>1.83</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>0.29</td><td>1.14</td><td>0.85</td><td>1.47</td><td>1.40</td><td>0.33</td><td>1.20</td><td>1.08</td><td>2.09</td><td>1.87</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>91.67</td><td>85.33</td><td>58.81</td><td>92.23</td><td>92.11</td><td>94.84</td><td>91.26</td><td>63.77</td><td>95.40</td><td>95.20</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>21.89</td><td>27.09</td><td>13.95</td><td>33.71</td><td>33.37</td><td>22.34</td><td>34.98</td><td>15.95</td><td>43.51</td><td>44.94</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>87.64</td><td>79.19</td><td>64.60</td><td>89.27</td><td>88.92</td><td>89.94</td><td>83.36</td><td>72.17</td><td>93.22</td><td>92.59</td></tr><tr><td>CIoth1[6k</td><td>MERU-B</td><td>86.25</td><td>76.99</td><td>67.57</td><td>88.73</td><td>88.07</td><td>90.14</td><td>83.09</td><td>74.35</td><td>92.32</td><td>92.52</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>87.01</td><td>87.29</td><td>70.47</td><td>88.84</td><td>88.31</td><td>89.18</td><td>91.13</td><td>75.31</td><td>92.89</td><td>92.52</td></tr></table>

Table 2. CZSL metrics before and after whitening across models and inference methods for the low-intrinsic variance UT-Zappos dataset. Whitening consistently improves primitive-based inference and restores its advantage over pair composition.
<table><tr><td colspan="6">Harmonic Mean (%, ↑) / After Whitening</td></tr><tr><td>Model</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td>CLIP ViT-L/14</td><td>11.76/-</td><td>9.61 / -</td><td>16.41 / 16.69</td><td>8.04 / 13.75</td><td>9.33 / 22.80</td></tr><tr><td>DINOv2 ViT-L/14</td><td>2.81 / -</td><td>0.38 / -</td><td>12.06 / 13.64</td><td>3.20 / 12.40</td><td>2.49 / 16.17</td></tr><tr><td>CLIP-B (GRIT)</td><td>2.33 / -</td><td>1.61 / -</td><td>10.81 / 13.53</td><td>9.04 / 12.79</td><td>8.95 / 14.65</td></tr><tr><td>MERU-B</td><td>3.38 / -</td><td>3.80 / -</td><td>10.39 / 12.79</td><td>8.99 /  8.80</td><td>9.36 / 16.32</td></tr><tr><td>HyCoCLIP-B</td><td>4.34 /-</td><td>3.94 / -</td><td>8.50 / 13.01</td><td>5.89 /  8.14</td><td>5.86 / 15.11</td></tr></table>

<table><tr><td colspan="6">Unseen Pair Accuracy (%, ↑) / After Whitening</td></tr><tr><td>Model</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td>CLIP ViT-L/14</td><td>21.89 / -</td><td>29.19 / -</td><td>29.46 / 36.81</td><td>18.03 / 24.43</td><td>22.85 / 36.83</td></tr><tr><td>DINOv2 ViT-L/14</td><td>9.84 / -</td><td>11.85 / -</td><td>21.95 / 28.34</td><td>12.64 / 19.78</td><td>16.08 / 28.42</td></tr><tr><td>CLIP-B (GRIT)</td><td>16.45 / -</td><td>11.63 / -</td><td>16.50 / 21.01</td><td>13.11 / 15.44</td><td>13.91 / 21.21</td></tr><tr><td>MERU-B</td><td>8.41 / -</td><td>8.51 / -</td><td>16.45 / 25.17</td><td>11.26 / 15.55</td><td>14.60 / 25.12</td></tr><tr><td>HyCoCLIP-B</td><td>11.95 /-</td><td>14.49 / -</td><td>14.65 / 23.69</td><td>12.16 / 16.08</td><td>14.01 / 24.80</td></tr></table>

Table 2: the HM and UPA values increase noticeably for both GDE- and intrinsic-primitive compositional inference, achieving comparable or largely superior performance in some cases than pair-based inference.

After whitening, intrinsic primitives achieve the highest HM for every model and the highest UPA for four of five models, falling only 0.05 points below the best MERU-B UPA results using pair-based inference. These results provide causal evidence that low intrinsic variance of the dataset (and its embedding distribution) contributes to the original failure mode of compositional inference. UT-Zappos is thus not a failure of primitive inference: its low-variance representation suppresses decomposition effectiveness.

## 5.3. Statistical Separation of GDE Primitives

A key premise of primitive-based inference is that the decomposition isolates object and attribute information into distinct semantic components. We test this by measuring the statistical dependence between object, attribute, and pair representations using Pearson cross-correlation (CORR) and mutual information (MI). We perform this analysis on the GDE primitives. The results show that, across datasets and model families, the dependence and mutual information between object and attribute primitive representations is consistently substantially lower than those of either object and pair or attribute and pair representations. For example, on MIT-States object and attribute primitive CORR remains between 0.444 and 0.473 across models, compared to 0.825– 0.895 for object primitive and pair representation, and the MI estimates are 13.0 versus 119.4–320.7. We observe similar results across the considered CZSL datasets and model representations. We report full results in Appendix D.

The higher dependence that we consistently observe between the primitive representations and the full pair representation are expected and confirm the disentanglement quality of the factorization that we perform. Object and attribute primitives retain information from the composition from which they are derived, while exhibiting weak dependence on one another. These results provide evidence that primitivebased decomposition, as we study it, recovers a factorized semantic structure in which object and attribute information is meaningfully separated rather than simply reproducing the original pair representation, and thus providing a solid tool for compositional reasoning.

## 5.4. Group Robustness via Primitives

Pre-trained vision-language models achieve strong zero-shot performance under distribution shift, but remain sensitive to spurious correlations and subgroup imbalance [40]. Existing approaches to group robustness typically rely on group annotations, additional optimization, group-specific prompts, adaptation, or class reweighting [7, 11, 12, 30, 32, 33].

We instead ask whether the compositional structure already encoded in and emerging from pretrained representations can itself be used for robust inference. The results of the statistical disentanglement indicate that decomposed primitives separate object and attribute semantics, allowing inference to retain information relevant to the object while suppressing attribute directions that may induce spurious correlations. We thus evaluate decomposed primitive for robustness to group bias on Waterbirds [30] and CelebA [16]. We do not use group annotations, additional data, or further optimization. Specifically, we classify images using only the object primitives, retaining only the object-score component of Eq. 6, Eq. 7 and Eq. 11 for zero-shot, intrinsic, and GDE primitives, respectively. Table 3 shows that this simple inference-time decomposition consistently improves group robustness. Both GDE and intrinsic object primitives improve worst-group accuracy over zero-shot object inference while largely preserving, and in several cases improving, average accuracy. These gains arise without an explicit debiasing objective or fairness supervision: they come solely from isolating semantic structure already present in the pretrained representations. This suggests that compositional factorization is not only useful for interpreting representations, but also provides a practical mechanism for reducing reliance on spurious attributes at inference time.

Table 3. Group-robustness results (%). Average Accuracy (Avg. Acc.), Worst-Group Accuracy (WG. Acc.), and their gap on Waterbirds and CelebA. GDE and intrinsic object primitives consistently improve over zero-shot object inference without retraining or group annotations, with especially large gains for the hyperbolic models MERU and HyCoCLIP.
<table><tr><td></td><td></td><td colspan="3">Zero Shot</td><td colspan="3">GDE-Primitive</td><td colspan="3">Intrinsic Primitive</td></tr><tr><td></td><td>Model</td><td>Avg. Acc. ↑</td><td>WG. Acc. ↑</td><td>Gap ↓</td><td>Avg. Acc. ↑</td><td>WG. Acc. ↑</td><td>Gap ↓</td><td>Avg. Acc. ↑</td><td>WG. Acc. ↑</td><td>Gap ↓</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>80.9</td><td>48.8</td><td>32.1</td><td>90.8</td><td>85.7</td><td>5.13</td><td>90.8</td><td>85.7</td><td>5.17</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>69.1</td><td>42.4</td><td>26.7</td><td>85.5</td><td>48.0</td><td>37.5</td><td>85.0</td><td>49.4</td><td>35.6</td></tr><tr><td>Waebids</td><td>CLIP-B (GRIT)</td><td>58.4</td><td>36.9</td><td>21.5</td><td>77.1</td><td>64.4</td><td>12.7</td><td>77.0</td><td>64.4</td><td>12.7</td></tr><tr><td></td><td>MERU-B</td><td>58.3</td><td>31.7</td><td>26.6</td><td>63.0</td><td>39.0</td><td>24.0</td><td>77.1</td><td>62.6</td><td>14.6</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>53.9</td><td>20.3</td><td>33.7</td><td>62.6</td><td>38.3</td><td>24.3</td><td>77.1</td><td>62.6</td><td>14.5</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>75.3</td><td>69.2</td><td>6.16</td><td>87.9</td><td>86.9</td><td>1.00</td><td>87.8</td><td>86.8</td><td>1.00</td></tr><tr><td>ClbbA</td><td>DINOv2 ViT-L/14</td><td>13.7</td><td>0.07</td><td>13.6</td><td>77.7</td><td>73.1</td><td>4.54</td><td>77.4</td><td>73.8</td><td>3.57</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>56.6</td><td>29.8</td><td>26.9</td><td>81.9</td><td>74.8</td><td>7.08</td><td>81.7</td><td>74.7</td><td>6.97</td></tr><tr><td></td><td>MERU-B</td><td>35.2</td><td>11.5</td><td>23.6</td><td>69.1</td><td>59.6</td><td>9.58</td><td>84.1</td><td>80.3</td><td>3.80</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>34.3</td><td>6.46</td><td>27.8</td><td>69.4</td><td>57.3</td><td>12.1</td><td>84.6</td><td>79.7</td><td>4.95</td></tr></table>

Intrinsic primitives yield better group robustness in hyperbolic spaces. The group robustness gains are especially pronounced for MERU and HyCoCLIP, where intrinsic object primitives yield large improvements in both average and worst-group accuracy. We conjecture that this advantage is related to the radial organization of semantic abstraction in the hyperbolic space. Intrinsic object primitives lie closer to the origin than compound object–attribute representations, consistent with their encoding broader, less attribute-specific concepts. GDE primitives, by contrast, are obtained through a local linear approximation around the Fréchet mean and can therefore remain more influenced by the local neighborhood structure induced by dominant attribute or background correlations. This provides a plausible geometric explanation for the stronger robustness of intrinsic primitives that lay directly on the manifold. The radial organization of these representations is further explored in Appendix H.

## 6. Conclusion

This work establishes primitive decomposition as a unifying principle for compositional zero-shot learning (CZSL) across Euclidean, spherical, and hyperbolic vision–language models. By systematically analyzing models with diverse manifold geometries, we demonstrate that compositional structure consistently emerges in pretrained vision embeddings in the form of primitive visual concept representations, which can be directly exploited for compositional zero-shot inference. Crucially, we show that explicit attribute-object pair re-composition is not necessary for CZSL: independently reasoning over attribute and object primitives, with specifically-designed scoring functions, can effectively support compositional open-world generalization.

We demonstrate primitives as a novel paradigm for CZSL inference with results being consistent for the tangent space GDE primitives and intrinsic primitives. Across experiments, GDE primitives provide robust and transferable primitive representations, consistently outperforming pair-based compositions for open-world CZSL while remaining simple and geometry-agnostic. Intrinsic primitives offer a complementary, geometry-aware refinement, particularly effective in hyperbolic spaces, where operating directly on the manifold improves stability by avoiding local linearization errors. We further demonstrate benefits of primitive-based inference in group robustness tasks, where object-primitive representations allow to perform unbiased inference without model re-training, avoiding spurious correlations induced by attribute characteristics.

Finally, this work provides the first comprehensive and unified view of primitive-based CZSL, demonstrating it as an emerging property of existing pretrained models.

## Acknowledgments

This publication is part of the project ROBUST: Trustworthy AI-based Systems for Sustainable Growth with project number KICH3.LTP.20.006, which is (partly) financed by the Dutch Research Council (NWO), Siemens Healthineers, and the Dutch Ministry of Economic Affairs and Climate Policy (EZK) under the program LTP KIC 2020–2023.

This publication is part of the project ReVision with file number VI.Vidi.243.176 of the research programme Vidi ENW which is (partly) financed by the Dutch Research Council (NWO) under the grant https://doi.org/ 10.61686/ERNED50610.

T.T. acknowledges the EU project ELSA-European Lighthouse on Secure and Safe AI, under the EU’s Horizon Europe Research and Innovation Programme (grant agreement no. 101070617)

This publication is part of the project HyperVision with file number VI.Vidi.243.085 of the research programme Vidi ENW which is (partly) financed by the Dutch Research Council (NWO) under the grant https://doi.org/ 10.61686/DCECG94893.

## References

[1] Luca Barsellotti, Lorenzo Bianchi, Nicola Messina, Fabio Carrara, Marcella Cornia, Lorenzo Baraldi, Fabrizio Falchi, and Rita Cucchiara. Talking to DINO: Bridging self-supervised vision backbones with language for open-vocabulary segmentation. In IEEE International Conference on Computer Vision, 2025. 2, 3, 5, 11

[2] Davide Berasi, Matteo Farina, Massimiliano Mancini, Elisa Ricci, and Nicola Strisciuglio. Not only text: exploring compositionality of visual representations in vision-language models. In IEEE Conference on Computer Vision and Pattern Recognition, 2025. 1, 2, 3, 4, 6

[3] Ines Chami, Albert Gu, Dat P Nguyen, and Christopher Re. Horopca: Hyperbolic dimensionality reduction via horospherical projections. In Proceedings of the 38th International Conference on Machine Learning, pages 1419–1429. PMLR, 2021. 12

[4] Paweł Czyz, Frederic Grabowski, Julia E. Vogt, Niko Beeren-˙ winkel, and Alexander Marx. On the Properties and Estimation of Pointwise Mutual Information Profiles. TMLR, 2024. 11

[5] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, 2023. 1, 2, 3, 5

[6] P.T. Fletcher, Conglin Lu, S.M. Pizer, and Sarang Joshi. Principal geodesic analysis for the study of nonlinear statistics of shape. IEEE Transactions on Medical Imaging, 23(8): 995–1005, 2004. 12

[7] Abinitha Gourabathina, Hyewon Jeong, Teya Bergamaschi, Marzyeh Ghassemi, and Collin Stultz. Robustness beyond known groups with low-rank adaptation, 2026. 7

[8] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality. In Advances in Neural Information Processing Systems, 2023. 2

[9] Xiaoming Hu and Zilei Wang. Leveraging sub-class discimination for compositional zero-shot learning. In AAAI Conference on Artificial Intelligence, 2023. 1, 2, 3

[10] Phillip Isola, Joseph J. Lim, and Edward H. Adelson. Discovering states and transformations in image collections. In IEEE Conference on Computer Vision and Pattern Recognition, 2015. 2, 5

[11] Nayeong Kim, Seong Joon Oh, and Suha Kwak. Groupcoop: Group-robust fine-tuning via group prompt learning, 2025. 7

[12] Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In International Conference on Learning Representations, 2023. 7

[13] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept Bottleneck Models. In International Conference on Machine Learning, 2020. 3

[14] Brenden M. Lake and Marco Baroni. Human-like systematic generalization through a meta-learning neural network. Nature, 623(7985), 2023. 1

[15] Jongmin Lee, Jang-Hyun Kim, and Hee-Seok Oh. spherepc: An r package for dimension reduction on a sphere. The R Journal, 14:167–181, 2022. https://doi.org/10.32614/RJ-2022-016. 12

[16] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In IEEE International Conference on Computer Vision. IEEE Computer Society, 2015. 8

[17] Xiaocheng Lu, Song Guo, Ziming Liu, and Jingcai Guo. Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning . In IEEE Conference on Computer Vision and Pattern Recognition, 2023. 1, 2, 3

[18] Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, and Zeynep Akata. Open world compositional zeroshot learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2021. 1, 4, 6

[19] Maxime Oquab, Timothée Darcet, Piotr Bojanowski, and et. al. DINOv2: learning robust visual features without supervision. Trans. Mach. Learn. Res., 2023. 2, 3

[20] Ishan Misra, Abhinav Gupta, and Martial Hebert. From red wine to red tomato: Composition with context. In IEEE Conference on Computer Vision and Pattern Recognition, 2017. 1

[21] Ans Munir, Faisal Z. Qureshi, Mohsen Ali, and Muhammad Haris Khan. Compositional zero-shot learning: A survey, 2025. 1

[22] Nihal V. Nayak, Peilin Yu, and Stephen Bach. Learning to compose soft prompts for compositional zero-shot learning. In International Conference on Learning Representations, 2023. 1, 2, 3

[23] Avik Pal, Max van Spengler, Guido Maria D’Amely di Melendugno, Alessandro Flaborea, Fabio Galasso, and Pascal Mettes. Compositional entailment learning for hyperbolic

vision-language models. In International Conference on Learning Representations, 2025. 1, 2, 3, 5

[24] Barbara Partee et al. Lexical semantics and compositionality. An invitation to cognitive science: Language, 1:311–360, 1995. 1

[25] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding multimodal large language models to the world. In International Conference on Learning Representations, 2024. 5

[26] Khoi Pham, Kushal Kafle, Zhe Lin, Zhihong Ding, Scott Cohen, Quan Tran, and Abhinav Shrivastava. Learning to predict visual attributes in the wild. In IEEE Conference on Computer Vision and Pattern Recognition, 2021. 5

[27] Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, and Rita Cucchiara. Hyperbolic safety-aware vision-language models. In IEEE Conference on Computer Vision and Pattern Recognition, 2025. 3

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning, 2021. 1, 2

[29] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Thalaiyasingam Ajanthan. Accept the modality gap: an exploration in the hyperbolic space. In IEEE Conference on Computer Vision and Pattern Recognition, 2024. 3

[30] Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: on the importance of regularization for worstcase generalization. In International Conference on Learning Representations, 2020. 7, 8

[31] Nirat Saini, Khoi Pham, and Abhinav Shrivastava. Disentangling visual embeddings for attributes and objects. In IEEE Conference on Computer Vision and Pattern Recognition, 2022. 5

[32] Seonguk Seo and Bohyung Han. Re-evaluating group robustness via adaptive class-specific scaling. In IEEE Winter Conference on Applications ofComputer Vision, 2025. 7

[33] Nimit Sohoni, Jared A. Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: fine-grained robustness in coarse-grained classification problems. In Advances in Neural Information Processing Systems, 2020. 7

[34] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: probing vision and language models for visiolinguistic compositionality. In IEEE Conference on Computer Vision and Pattern Recognition, 2022. 2

[35] Qingsheng Wang, Lingqiao Liu, Chenchen Jing, Hao Chen, Guoqiang Liang, Peng Wang, and Chunhua Shen. Learning Conditional Attributes for Compositional Zero-Shot Learning . In IEEE Conference on Computer Vision and Pattern Recognition, 2023. 1, 2, 3

[36] Peng Wu, Qiuxia Lai, Hao Fang, Guo-Sen Xie, Yilong Yin, Xiankai Lu, and Wenguan Wang. A conditional probability framework for compositional zero-shot learning. In IEEE International Conference on Computer Vision, 2025. 1, 2, 3

[37] Daiki Yoshikawa and Takashi Matsubara. PHyCLIP: ℓ<sub>1</sub>- product of hyperbolic factors unifies hierarchy and compositionality in vision-language representation learning. arXiv, 2026. 3

[38] Aron Yu and Kristen Grauman. Fine-grained visual comparisons with local learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2014. 2, 5

[39] Mert Yüksekgönül, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In International Conference on Learning Representations, 2023. 2

[40] Michael Zhang and Christopher Re. Contrastive adapters for foundation model group robustness. In Advances in Neural Information Processing Systems, 2022. 7

[41] Shiyu Zhang, Cheng Yan, Yang Liu, Chenchen Jing, Lei Zhou, and Wenjun Wang. Learning visual proxy for compositional zero-shot learning. In IEEE International Conference on Computer Vision, 2025. 1

[42] Tian Zhang, Kongming Liang, Ruoyi Du, Xian Sun, Zhanyu Ma, and Jun Guo. Learning invariant visual representations for compositional zero-shot learning. In European Conference on Computer Vision, 2022. 1

## A. Riemannian Gradient Descent

Given a collection of manifold-valued embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset { \mathcal { M } }$ , we seek a representative point that minimizes the sum of squared geodesic distances to all samples. This point is known as the intrinsic mean (or Fréchet mean) and is defined as

$$
\mu ^ { \star } = \arg \operatorname* { m i n } _ { \mu \in \mathcal { M } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } d _ { \mathcal { M } } ^ { 2 } ( \mu , f ( x _ { i } ) ) ,
$$

where $d _ { \mathcal { M } } ( \cdot , \cdot )$ denotes the geodesic distance on the manifold M. Unlike Euclidean averaging, optimization on a curved manifold requires moving along tangent spaces using the logarithmic and exponential maps. At an iterate $\boldsymbol { \mu } ^ { ( \bar { t } ) } \in \mathcal { M }$ , each sample $f ( x _ { i } )$ is first mapped to the tangent space $T _ { \mu ^ { ( t ) } } \mathcal { M }$ via the logarithmic map $\operatorname { L o g } _ { \mu ^ { ( t ) } } ( f ( x _ { i } ) )$ . The average tangent vector:

$$
\mathbf { v } ^ { ( t ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { L o g } _ { \mu ^ { ( t ) } } ( f ( x _ { i } ) )
$$

corresponds to the Riemannian gradient direction of the Fréchet objective. The iterate is then updated by mapping this tangent vector back onto the manifold using the exponential map:

$$
\mu ^ { ( t + 1 ) } = \mathrm { E x p } _ { \mu ^ { ( t ) } } \Big ( \eta \mathbf { v } ^ { ( t ) } \Big ) ,
$$

where $\eta > 0$ is the step size. Algorithm 1 summarizes the complete optimization procedure.

Algorithm 1 Intrinsic Mean Computation   
Require: Embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathcal { M } ,$ initial point   
$\pmb { \mu } ^ { ( 0 ) } \in \mathcal { M } ,$ step size η   
1: for $t = 0 , 1 , \dots , T - 1$ do   
2: $\begin{array} { r } { \mathbf { v } ^ { ( t ) }  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { L o g } _ { \mu ^ { ( t ) } } ( f ( x _ { i } ) ) } \end{array}$   
3: $\mu ^ { ( t + 1 ) } \gets \mathrm { E x p } _ { \mu ^ { ( t ) } } \big ( \eta \mathbf { v } ^ { ( t ) } \big )$   
4: end for   
5: return $\mu ^ { ( T ) }$

## B. Dense Feature Extraction via Talk2DINO

Unlike CLIP, MERU, or HyCoCLIP that inherently produce aligned image-text features from joint text pairs, vision-only models like DINOv2 yield dense patch embeddings without text alignment. To construct a competitive text-conditioned compositional baseline for DINOv2 that matches the open-world zero-shot capabilities of visionlanguage models, we interface its Euclidean representation space via Talk2DINO [1]. Given an image x and a textual query q, Talk2DINO produces patch embeddings $\{ \mathbf { p } _ { m } \} _ { m = 1 } ^ { M }$ together with a query embedding q. The spatial relevance map and the final pooled image representation are computed

respectively as

$$
\alpha _ { m } = \frac { \exp \left( \tau \frac { { \bf p } _ { m } ^ { \top } { \bf q } } { \| { \bf p } _ { m } \| \| { \bf q } \| } \right) } { \sum _ { k = 1 } ^ { M } \exp \left( \tau \frac { { \bf p } _ { k } ^ { \top } { \bf q } } { \| { \bf p } _ { k } \| \| { \bf q } \| } \right) } ,\tag{13}
$$

$$
f ( \{ p _ { m } \} _ { m = 1 } ^ { M } , q ) = \sum _ { m = 1 } ^ { M } \alpha _ { m } \mathbf { p } _ { m } ,\tag{14}
$$

where $\tau$ is a temperature parameter. This routing mechanism yields a single, unified Euclidean image representation conditioned dynamically on the queried attribute-object composition while strictly preserving DINOv2’s original underlying geometry.

## C. DINO Embeddings: Euclidean or Spherical?

Talk2DINO provides a language wrapper to DINO, without modifying its native visual embeddings. This enables us to compute for DINO the for fair comparison. The core compositionality analysis is performed on the native DINO vision embedding and confirms that properly addressing manifold geometry is key to effective decomposition: Table 4 shows that reducing the embeddings to spherical space harms performance, due to wrong manifold assumption for primitive extraction.

<table><tr><td></td><td></td><td colspan="3">Harmonic Mean  $( \% , \uparrow )$ </td><td colspan="3">Unseen Pair Accuracy  $( \% , \uparrow )$ </td></tr><tr><td>Dataset</td><td>M</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td>MIT</td><td>euc</td><td>6.56</td><td>8.22</td><td>8.04</td><td>6.26</td><td>9.33</td><td>9.42</td></tr><tr><td></td><td>sph</td><td>2.17</td><td>4.70</td><td>4.85</td><td>1.87</td><td>4.61</td><td>4.75</td></tr><tr><td>VAW</td><td>euc</td><td>1.30</td><td>2.05</td><td>2.20</td><td>1.63</td><td>2.60</td><td>2.81</td></tr><tr><td></td><td>sph</td><td>0.28</td><td>0.68</td><td>0.72</td><td>0.28</td><td>0.65</td><td>0.71</td></tr></table>

Table 4. Evaluation results utilizing the Talk2DINO as Euclidean (euc) or Spherical (sph) embeddings (after $\ell _ { 2 }$ normalisation). The native euclidean manifold substantially outperforms spherical.

## D. GDE Primitive Concept Separation

To assess whether the GDE primitives capture concept structure that is separable in the model’s embedding space, we measure the statistical dependence between each pair of primitives using two complementary estimators: the Pearson cross correlation (CORR), and a canonical correlation estimator of mutual information (MI) computed with the bench-mi mutual information package [4]. CORR captures linear alignment between two sets of vectors, while MI additionally captures non-linear dependence structure that a purely linear estimator would miss. We report both for the three primitive pairings: object-attribute, object-pair, and attribute-pair. Our analysis is limited to GDE primitives.

Across datasets and models, in Table 5, we observe trends consistent with our expectations: the object-attribute pairing is consistently the lowest correlated and carries the least mutual information relative to object-pair and attribute-pair. This suggests that object and attribute representations are comparatively more disentangled from one another, whereas information about objects and attributes is each more entangled with pairwise (relational) structure than with each other.

Table 5. Cross Correlation and Mutual Information estimates between the GDE primitives and the model embedding. Since GDE primtives live in euclidean space, usual statistics calculation methods apply.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">CLIP ViT-L/14</td><td colspan="2">DINOv2 ViT-L/14</td><td colspan="2">CLIP-B (GRIT)</td><td colspan="2">HyCoCLIP-B</td><td colspan="2">MERU-B</td></tr><tr><td>CORR</td><td>MI</td><td>CORR</td><td>MI</td><td>CORR</td><td>MI</td><td>CORR</td><td>MI</td><td>CORR</td><td>MI</td></tr><tr><td rowspan="3">MIT-States</td><td>object-attribute</td><td>0.453</td><td>13.0</td><td>0.456</td><td>13.0</td><td>0.444</td><td>13.0</td><td>0.453</td><td>13.0</td><td>0.473</td><td>13.0</td></tr><tr><td>object-pair</td><td>0.825</td><td>191.0</td><td>0.895</td><td>320.7</td><td>0.831</td><td>148.7</td><td>0.839</td><td>119.7</td><td>0.843</td><td>119.4</td></tr><tr><td>attribute-pair</td><td>0.530</td><td>77.9</td><td>0.516</td><td>129.0</td><td>0.511</td><td>44.5</td><td>0.513</td><td>44.2</td><td>0.531</td><td>44.3</td></tr><tr><td rowspan="3">VAW-CZSL</td><td>object-attribute</td><td>0.187</td><td>7.2</td><td>0.183</td><td>7.2</td><td>0.195</td><td>6.9</td><td>0.191</td><td>6.9</td><td>0.173</td><td>6.9</td></tr><tr><td>object-pair</td><td>0.575</td><td>38.1</td><td>0.713</td><td>61.0</td><td>0.565</td><td>28.1</td><td>0.565</td><td>26.6</td><td>0.576</td><td>29.9</td></tr><tr><td>attribute-pair</td><td>0.203</td><td>14.9</td><td>0.228</td><td>13.2</td><td>0.215</td><td>16.5</td><td>0.213</td><td>11.0</td><td>0.203</td><td>10.9</td></tr><tr><td rowspan="3">UT-Zappos</td><td>object-attribute</td><td>0.098</td><td>0.8</td><td>0.090</td><td>0.8</td><td>0.067</td><td>0.8</td><td>0.067</td><td>0.8</td><td>0.069</td><td>0.8</td></tr><tr><td>object-pair</td><td>0.872</td><td>4.8</td><td>0.930</td><td>5.1</td><td>0.896</td><td>5.5</td><td>0.909</td><td>5.6</td><td>0.929</td><td>18.0</td></tr><tr><td>attribute-pair</td><td>0.338</td><td>4.0</td><td>0.174</td><td>3.4</td><td>0.264</td><td>3.0</td><td>0.241</td><td>3.0</td><td>0.208</td><td>3.1</td></tr><tr><td rowspan="3">Clothing16k</td><td>object-attribute</td><td>0.404</td><td>2.7</td><td>0.454</td><td>2.7</td><td>0.419</td><td>2.7</td><td>0.427</td><td>2.7</td><td>0.419</td><td>2.7</td></tr><tr><td>object-pair</td><td>0.945</td><td>73.1</td><td>0.992</td><td>73.0</td><td>0.962</td><td>72.7</td><td>0.953</td><td>73.2</td><td>0.948</td><td>73.0</td></tr><tr><td>attribute-pair</td><td>0.606</td><td>83.1</td><td>0.488</td><td>83.2</td><td>0.581</td><td>83.4</td><td>0.612</td><td>83.7</td><td>0.609</td><td>83.2</td></tr></table>

We also note that object-pair estimates are consistently higher than attribute-pair estimates across datasets and models. This points to embeddings being generally more aligned with object-level information than with attribute-level information, and suggests attribute-awareness may be an area where current embeddings are comparatively weaker, a direction that could benefit from further study.

## E. Non-Euclidean Whitening

For Euclidean features, whitening can be applied directly in the ambient space. For spherical and hyperbolic features, we instead perform whitening in a tangent space, following the general principle of Principal Geodesic Analysis (PGA) [6]. The intrinsic mean $\pmb { \mu }$ is computed as described in Algorithm 1, after which the embeddings are mapped to the corresponding tangent space using the logarithmic map.

## E.1. Spherical Whitening

For spherical embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathbb { S } ^ { d - 1 }$ , we first compute the intrinsic mean $\pmb { \mu }$ and map each embedding to $T _ { \mu } \mathbb { S } ^ { d - 1 }$ using $\operatorname { L o g } _ { \mu } ,$ , as defined in $\operatorname { E q . 9 } .$ . Whitening is then performed on the resulting tangent vectors.

A spherical-specific restriction arises from the cut locus. For the unit sphere, the logarithmic representation around µ is unique only for tangent vectors with norm smaller than π. Arbitrarily rescaling a tangent vector beyond this radius causes the exponential map to wrap around the sphere. Distinct tangent vectors can then map to the same or nearby spherical locations, which may destroy semantic information encoded by their relative positions. We therefore constrain the norm of the projected tangent vector to remain within the injectivity radius with rescaling. This follows the same general tangent-space construction used in spherical PCA methods spherepc [15].

Algorithm 2 Spherical whitening   
Require: Embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathbb { S } ^ { d - 1 }$   
1: Compute intrinsic mean $\pmb { \mu }$ using Algorithm 1   
2: for $i = 1 , \ldots , N$ do   
3: $\mathbf { v } _ { i }  \operatorname { L o g } _ { \mu } ( f ( x _ { i } ) )$   
4: end for   
5: Compute covariance C of $\{ { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { N }$   
6: Compute whitening transform $\mathbf { \bar { W } }  \mathbf { C } ^ { - 1 / 2 }$   
7: for $i = 1 , \ldots , N$ do   
8: $\hat { \mathbf { v } } _ { i } \gets \mathbf { W } \mathbf { v } _ { i }$   
9: end for   
10: Rescale $\{ \hat { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { N }$ if required such that $\left\| \hat { \mathbf { v } } _ { i } \right\| < \pi$   
11: for $i = 1 , \ldots , \bar { N }$ do   
12: $\hat { f } ( x _ { i } ) \gets \mathrm { E x p } _ { \mu } ( \hat { \mathbf { v } } _ { i } )$   
13: end for   
14: return $\{ \hat { f } ( x _ { i } ) \} _ { i = 1 } ^ { N }$

## E.2. Hyperboloid Whitening

For hyperbolic embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathbb { H } ^ { d + 1 }$ , we again begin by computing an intrinsic mean $\pmb { \mu } .$ Unlike the spherical case, the ambient hyperboloid coordinates are equipped with the Lorentz metric rather than the Euclidean metric. We therefore first apply a Lorentz isometry that maps µ to the canonical hyperbolic origin o. This preserves hyperbolic distances while placing the tangent representation in a canonical coordinate system. Related use of hyperbolic isometries appears in HoroPCA [3].

After this transformation, the embeddings are mapped to $T _ { \mathbf { o } } \mathbb { H } ^ { d + 1 }$ using the logarithmic map as defined in Eq. 10. At the canonical origin, the tangent coordinates provide an ordinary Euclidean representation of the local Riemannian metric, and standard covariance whitening can therefore be applied. The whitened tangent vectors are finally mapped back to the hyperboloid using the exponential map.

Algorithm 3 Hyperboloid whitening   
Require: Embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N } \subset \mathbb { H } ^ { d + 1 }$   
1: Compute intrinsic mean µ using Algorithm 1   
2: Compute Lorentz isometry $B _ { \mu }$ such that $B _ { \mu } ( \mu ) = \mathbf { o }$   
3: for $i = 1 , \ldots , N$ do   
4: $\mathbf { u } _ { i }  B _ { \mu } ( f ( x _ { i } ) )$   
5: $\mathbf { v } _ { i }  \mathrm { L o g } _ { \mathbf { o } } ( \mathbf { u } _ { i } )$   
6: end for   
7: Compute covariance C of $\{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N }$   
8: Compute whitening transform $\bar { \mathbf { W } }  \mathbf { C } ^ { - 1 / 2 }$   
9: for $i = 1 , \ldots , N$ do   
10: $\hat { \mathbf { v } } _ { i } \gets \mathbf { W } \mathbf { v } _ { i }$   
11: ${ \hat { f } } ( x _ { i } ) \gets \mathrm { E x p } _ { \mathbf { o } } ( { \hat { \mathbf { v } } } _ { i } )$   
12: end for   
13: return $\{ \hat { f } ( x _ { i } ) \} _ { i = 1 } ^ { N }$

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Harmonic Mean  $( \% , \uparrow )$ </td><td colspan="2">Unseen Pair Accuracy (%, ↑)</td></tr><tr><td>Weighted</td><td>Uniform</td><td>Weighted</td><td>Uniform</td></tr><tr><td rowspan="5">MII</td><td>CLIP ViT-L/14</td><td>10.90</td><td>10.39</td><td>11.98</td><td>11.47</td></tr><tr><td>DINOv2 ViT-L/14</td><td>8.22</td><td>8.02</td><td>9.33</td><td>9.10</td></tr><tr><td>CLIP-B (GRIT)</td><td>6.64</td><td>6.12</td><td>7.18</td><td>6.62</td></tr><tr><td>MERU-B</td><td>6.57</td><td>6.59</td><td>7.25</td><td>7.14</td></tr><tr><td>HyCoCLIP-B</td><td>7.94</td><td>7.83</td><td>8.75</td><td>8.62</td></tr><tr><td rowspan="5">VAW</td><td>CLIP ViT-L/14</td><td>1.77</td><td>1.64</td><td>2.07</td><td>2.03</td></tr><tr><td>DINOv2 ViT-L/14</td><td>2.05</td><td>2.11</td><td>2.60</td><td>2.64</td></tr><tr><td>CLIP-B (GRIT)</td><td>1.52</td><td>1.37</td><td>2.13</td><td>2.01</td></tr><tr><td>MERU-B</td><td>1.49</td><td>1.43</td><td>1.97</td><td>1.87</td></tr><tr><td>HyCoCLIP-B</td><td>1.47</td><td>1.41</td><td>2.09</td><td>2.03</td></tr></table>

Table 6. Disabling noise estimation results in minor performance variations across models for GDE primitives validating that weighing carries no directional information.

In contrast to the sphere, hyperbolic space has no finite cut locus. The tangent radius is therefore not subject to the same wrapping ambiguity.

## F. Primitive Weighting Ablation

GDE primitive sample weights are scalar, carrying only confidence rather than semantic information. In Table 6, we provide additional results that ablate this design inherited from the baseline framework.

## G. Extended Open-World CZSL Metrics

This section reports additional evaluation metrics for openworld compositional zero-shot learning beyond the harmonic mean discussed in the main paper. In addition to composed pair performance, we separately evaluate object-level, attribute-level. These metrics provide a more detailed view of compositional behaviour. Object and attribute accuracies isolate recognition performance on the underlying semantic primitives, helping distinguish whether improvements arise from better primitive identification or from stronger compositional reasoning.

Table 7 reports object and attribute accuracies across all datasets and geometries. Consistent with the main results, primitive-based inference generally improves both object and attribute recognition, particularly on datasets with greater semantic diversity such as MIT-States and VAW-CZSL. These gains indicate that primitive decomposition improves not only pair retrieval, but also the separability of the underlying semantic factors. Table 8 reports seen and overall pair accuracies.

## H. Hierarchy of Hyperbolic Intrinsic Primitives

Hyperbolic embeddings encode hierarchy through radial distance from the origin, where larger radii correspond to more specific concepts. Driven by the strong group robustness results, and to further deepen the analysis of the properties of the intrinsic primitives, we examine the distributions of the radius of the representations of attributes, objects, and attribute-object compositions on the manifold of hyperbolic models.

We plot the computed distributions for HyCoCLIP in Figure 4 and for MERU in Figure 5 for the considered datasets. The embeddings of attribute-object compositions consistently occupy larger radii than their constituent primitives, consistently with the expected semantic hierarchy in hyperbolic spaces: compositions such as blue car are more specific than either primitives alone.

For MIT-States, VAW-CZSL, UT-Zappos, and Clothing16K, attribute primitives tend to lie closer to the origin and thus represent more general concepts, shared across several objects. Object primitives instead occupy deeper regions of the hierarchy, configuring as slightly more specific and complex semantic concepts in the considered datasets. In contrast, Waterbirds and CelebA data shows the opposite trend, with attributes occupying larger radii than objects. This suggests that attributes (e.g., backgrounds in Waterbirds or facial characteristics in CelebA) add complexity and further specifications to the object concepts in the dataset, possibly inducing biases and dominating the semantic structure of the data. The results of this analysis align with those achieved for group robustness tests where intrinsic primitives show exceptional de-biasing performance for hyperbolic models: the use of object primitives allows to disentangle the complexity induced by the attributes that cause bias and spurious correlations arising in the zero-shot predictions.

From a general perspective, these observations support the effectiveness of intrinsic primitive inference. Primitive embeddings occupy denser and better-covered regions of the representation space, whereas compositions correspond to deeper and sparser concepts. By decomposing representations into primitives rather than relying on locally estimated GDE primitives, intrinsic primitive means remain closer to the training distribution and better capture the dominant semantic structure of the dataset. This leads to more stable compositional representations and improved robustness under distribution shifts.

Table 7. Open-world Compositional Zero-Shot Learning results (%, ↑). Metrics include Object Accuracy and Attribute Accuracy.
<table><tr><td colspan="2"></td><td colspan="5">Object Accuracy (%, ↑)</td><td colspan="5">Attribute Accuracy (%, ↑)</td></tr><tr><td></td><td>Model</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>47.33</td><td>47.33</td><td>43.79</td><td>43.68</td><td>43.94</td><td>15.76</td><td>20.05</td><td>18.38</td><td>20.57</td><td>20.68</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>32.70</td><td>29.70</td><td>40.13</td><td>40.25</td><td>40.22</td><td>8.49</td><td>11.97</td><td>16.00</td><td>18.45</td><td>18.41</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>36.59</td><td>35.81</td><td>33.91</td><td>35.50</td><td>35.58</td><td>10.35</td><td>13.48</td><td>14.09</td><td>17.41</td><td>16.97</td></tr><tr><td>MIT-Stes</td><td>MERU-B</td><td>37.31</td><td>36.34</td><td>34.03</td><td>35.39</td><td>35.51</td><td>10.57</td><td>13.22</td><td>11.71</td><td>17.48</td><td>17.07</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>39.79</td><td>39.72</td><td>37.41</td><td>39.01</td><td>39.15</td><td>10.95</td><td>16.26</td><td>12.00</td><td>18.09</td><td>17.71</td></tr><tr><td>VA-SL</td><td>CLIP ViT-L/14</td><td>18.66</td><td>19.18</td><td>24.49</td><td>25.99</td><td>25.33</td><td>3.70</td><td>5.97</td><td>5.40</td><td>6.69</td><td>6.37</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>20.77</td><td>20.80</td><td>30.07</td><td>30.88</td><td>30.88</td><td>4.24</td><td>5.38</td><td>5.82</td><td>8.06</td><td>7.85</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>13.94</td><td>15.58</td><td>20.22</td><td>22.29</td><td>21.78</td><td>3.19</td><td>4.39</td><td>5.55</td><td>6.72</td><td>6.32</td></tr><tr><td></td><td>MERU-B</td><td>14.54</td><td>15.60</td><td>20.91</td><td>22.41</td><td>22.13</td><td>2.92</td><td>4.38</td><td>5.20</td><td>6.80</td><td>6.03</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>15.96</td><td>19.12</td><td>21.77</td><td>22.96</td><td>22.61</td><td>2.64</td><td>4.75</td><td>4.81</td><td>6.45</td><td>6.19</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>50.03</td><td>63.90</td><td>60.33</td><td>57.93</td><td>59.75</td><td>18.53</td><td>17.54</td><td>25.91</td><td>19.73</td><td>21.69</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>33.22</td><td>12.32</td><td>56.04</td><td>55.11</td><td>56.21</td><td>13.90</td><td>13.25</td><td>25.91</td><td>6.35</td><td>5.66</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>36.65</td><td>29.89</td><td>49.83</td><td>47.39</td><td>50.62</td><td>11.77</td><td>10.16</td><td>18.50</td><td>23.58</td><td>22.10</td></tr><tr><td>U--Pos</td><td>MERU-B</td><td>21.86</td><td>17.19</td><td>50.48</td><td>51.41</td><td>50.27</td><td>13.62</td><td>16.64</td><td>18.91</td><td>21.45</td><td>22.10</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>34.45</td><td>25.57</td><td>46.88</td><td>48.70</td><td>47.67</td><td>12.94</td><td>21.79</td><td>16.09</td><td>15.41</td><td>15.61</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>97.10</td><td>93.06</td><td>84.50</td><td>97.48</td><td>97.39</td><td>95.11</td><td>93.29</td><td>64.28</td><td>95.17</td><td>95.11</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>78.76</td><td>65.63</td><td>84.32</td><td>79.81</td><td>78.99</td><td>22.12</td><td>36.89</td><td>11.34</td><td>32.35</td><td>32.20</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>93.11</td><td>92.82</td><td>87.72</td><td>94.02</td><td>93.82</td><td>93.26</td><td>85.56</td><td>71.90</td><td>95.46</td><td>95.22</td></tr><tr><td>Clot16k</td><td>MERU-B</td><td>93.14</td><td>93.55</td><td>88.31</td><td>94.20</td><td>94.37</td><td>91.71</td><td>84.09</td><td>76.65</td><td>94.35</td><td>93.88</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>92.73</td><td>92.88</td><td>89.25</td><td>94.08</td><td>94.55</td><td>93.20</td><td>94.32</td><td>78.23</td><td>95.43</td><td>94.52</td></tr></table>

Table 8. Open-world Compositional Zero-Shot Learning results (%, ↑). Metrics include Seen Pair Accuracy and Overall Pair Accuracy.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="5">Seen Pair Accuracy (%, ↑)</td><td colspan="5">Overall Pair Accuracy (%, ↑)</td></tr><tr><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td><td>ZS Pair</td><td>ZS Prim.</td><td>GDE Pair</td><td>GDE Prim.</td><td>Int. Prim.</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>29.03</td><td>22.23</td><td>31.18</td><td>25.21</td><td>25.88</td><td>15.76</td><td>20.05</td><td>18.38</td><td>20.57</td><td>20.68</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>16.64</td><td>9.87</td><td>27.23</td><td>20.63</td><td>19.96</td><td>8.49</td><td>11.97</td><td>16.00</td><td>18.45</td><td>18.41</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>21.72</td><td>16.93</td><td>22.18</td><td>18.99</td><td>18.82</td><td>10.35</td><td>13.48</td><td>14.09</td><td>17.41</td><td>16.97</td></tr><tr><td>MITT-Stes</td><td>MERU-B</td><td>22.61</td><td>16.76</td><td>19.58</td><td>18.91</td><td>19.20</td><td>10.57</td><td>13.22</td><td>11.71</td><td>17.48</td><td>17.07</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>23.87</td><td>15.50</td><td>21.97</td><td>21.34</td><td>20.97</td><td>10.95</td><td>16.26</td><td>12.00</td><td>18.09</td><td>17.71</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>3.76</td><td>4.97</td><td>3.67</td><td>4.87</td><td>4.67</td><td>3.70</td><td>5.97</td><td>5.40</td><td>6.69</td><td>6.37</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>3.64</td><td>4.21</td><td>3.62</td><td>5.44</td><td>5.14</td><td>4.24</td><td>5.38</td><td>5.82</td><td>8.06</td><td>7.85</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>2.12</td><td>2.80</td><td>2.88</td><td>3.37</td><td>3.39</td><td>3.19</td><td>4.39</td><td>5.55</td><td>6.72</td><td>6.32</td></tr><tr><td>VA-SL</td><td>MERU-B</td><td>2.31</td><td>3.22</td><td>2.49</td><td>3.79</td><td>3.44</td><td>2.92</td><td>4.38</td><td>5.20</td><td>6.80</td><td>6.03</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>2.31</td><td>3.96</td><td>2.53</td><td>3.77</td><td>3.69</td><td>2.64</td><td>4.75</td><td>4.81</td><td>6.45</td><td>6.19</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>13.49</td><td>8.80</td><td>29.03</td><td>19.16</td><td>21.70</td><td>18.53</td><td>17.54</td><td>25.91</td><td>19.73</td><td>21.69</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>2.25</td><td>0.29</td><td>18.48</td><td>10.36</td><td>7.82</td><td>13.90</td><td>13.25</td><td>25.91</td><td>6.35</td><td>5.66</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>1.76</td><td>1.76</td><td>19.35</td><td>22.09</td><td>22.39</td><td>11.77</td><td>10.16</td><td>18.50</td><td>23.58</td><td>22.10</td></tr><tr><td>U-pos</td><td>MERU-B</td><td>3.23</td><td>4.11</td><td>21.41</td><td>22.78</td><td>23.07</td><td>13.62</td><td>16.64</td><td>18.91</td><td>21.45</td><td>22.10</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>9.48</td><td>3.71</td><td>17.69</td><td>11.24</td><td>13.88</td><td>12.94</td><td>21.79</td><td>16.09</td><td>15.41</td><td>15.61</td></tr><tr><td></td><td>CLIP ViT-L/14</td><td>98.72</td><td>91.30</td><td>87.21</td><td>98.98</td><td>98.72</td><td>92.41</td><td>86.90</td><td>58.86</td><td>92.76</td><td>92.62</td></tr><tr><td></td><td>DINOv2 ViT-L/14</td><td>48.59</td><td>42.46</td><td>52.94</td><td>52.17</td><td>57.54</td><td>19.22</td><td>28.42</td><td>9.00</td><td>30.65</td><td>30.53</td></tr><tr><td></td><td>CLIP-B (GRIT)</td><td>97.19</td><td>90.03</td><td>92.07</td><td>97.19</td><td>97.44</td><td>87.11</td><td>79.20</td><td>65.48</td><td>89.83</td><td>89.42</td></tr><tr><td>CIotht16k</td><td>MERU-B</td><td>97.70</td><td>93.09</td><td>93.09</td><td>97.95</td><td>97.44</td><td>85.64</td><td>78.35</td><td>69.47</td><td>88.95</td><td>88.69</td></tr><tr><td></td><td>HyCoCLIP-B</td><td>97.19</td><td>94.12</td><td>93.86</td><td>97.95</td><td>97.44</td><td>86.55</td><td>87.81</td><td>71.26</td><td>89.75</td><td>89.33</td></tr></table>

## I. Example Images for Datasets

We provide qualitative examples from each dataset used in our compositional zero-shot learning experiments in Figure 6. These samples are drawn from the test splits and illustrate the diversity of attribute-object compositions present in each domain.

Overall, most datasets exhibit substantial variation in both attributes and objects, reflecting the combinatorial nature of the task. In particular, Clothing16K contains a broad range of clothing categories and attributes, including different garment types, textures, and visual appearances, as well as a wider variety of poses and viewpoints. This results in higher visual diversity and more challenging compositional generalization settings.

![](images/1c7e19056be62ba9c53a22022468f7e5cd18643583533f6e70b24ac17cfcf241.jpg)  
Figure 4. Radius distributions for HyCoCLIP-B showing attribute primitives, object primitives, and attribute-object compositions in hyperbolic space. Dashed lines indicate mean radii. We find, objects intrinsic primitives are consistently occupy the higher radius stratum compared to attributes, except in the case of CelebA and Waterbirds. The pair embeddings themselves are significantly further away, which is consistent with the hierarchical entailment objective of these models.

In contrast, UT-Zappos is considerably more constrained, consisting exclusively of footwear categories. Moreover, the visual presentation of objects in this dataset tends to be relatively consistent, with limited variation in pose and viewpoint. This reduced geometric and semantic diversity aligns with our observation of lower compositional variability and helps explain its comparatively different behavior in our results.

![](images/ed1cf6f0f2b989775675b698f19de652bab43cd131c69432bf13ad288fd3c27c.jpg)  
Figure 5. Radius distributions for MERU showing attribute primitives, object primitives, and attribute-object compositions in hyperbolic space. Dashed lines indicate mean radii. Similar to HycoCLIP, we find, objects intrinsic primitives are consistently occupy the higher radius stratum compared to attributes, except in the case of CelebA and Waterbirds. The pair embeddings themselves are significantly further away which is consistent with the hierarchical entailment objective of these models.

![](images/0bceab738ae57d082543d10d8ef599bfdbfa7374cef2fea75887ea2e07de54ca.jpg)  
Figure 6. Random compositional examples across datasets.