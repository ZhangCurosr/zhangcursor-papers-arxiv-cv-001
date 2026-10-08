# Detecting Adversarial Images through Response Profiles of Vision-Language Models

Arash Vashagh and Roozbeh Razavi-Far

Trustworthy and Secure AI (TSAI) Lab, Faculty of Computer Science, University of New Brunswick, Canada Email: arash.vashagh@unb.ca, roozbeh.razavi-far@unb.ca

Abstract—Adversarial perturbations can alter the predictions of frozen vision-language models (VLMs) while leaving their confidence and image–text similarity patterns seemingly plausible. We investigate whether we can identify adversarial inputs based on the broader way an image interacts with a collection of general semantic prompts. Our detector summarizes these responses using category-level statistics, relationships among prompts, deviations from clean reference distributions, and stability under weak image transformations, producing a compact response profile that is classified by a lightweight model while the VLM remains fixed. We evaluate the approach on multiple public image datasets, several CLIP-style visual backbones, and a range of gradientbased, optimization-based, automated, and spatial attacks. The detector achieves strong discrimination in attack-specific settings and retains substantial performance when evaluated on attacks not seen during training. Under a controlled detector-specific protocol, the response-profile representation outperforms the evaluated embedding-geometry baselines. Additional analyses show that the feature groups provide complementary information and that the method remains effective under variations in the prompt configuration. We also examine inference cost and performance against detector-aware adaptive attacks. Overall, the results indicate that response patterns across semantic prompts provide a useful complementary signal for adversarial image detection in frozen VLMs.

Index Terms—adversarial examples, adversarial detection, vision-language models, CLIP, trustworthy machine learning

## I. INTRODUCTION

Vision-language models (VLMs) have become a central component of modern AI systems [1]–[3]. Models such as CLIP [1] align images and natural-language descriptions in a shared representation space, enabling zero-shot recognition, multimodal retrieval, and open-vocabulary reasoning without task-specific retraining. These capabilities make VLM representations attractive for downstream systems that must interpret visual inputs through flexible semantic concepts. Their growing use in applications such as content moderation and autonomous systems [4] also makes the reliability of these representations increasingly important.

Despite their strong generalization capabilities, VLMs remain vulnerable to adversarial perturbations. As with conventional vision models, small input perturbations can change the model prediction while leaving the image visually similar to the original [5]–[8]. Detecting such inputs is challenging because an adversarial example does not necessarily produce an obviously abnormal final confidence score. A successful perturbation may still yield a confident zero-shot prediction or apparently plausible image–text similarities. Consequently, examining only the final classifier output can discard information contained in the broader representation produced by the VLM.

Existing adversarial detectors exploit several forms of representation abnormality. Confidence-based approaches use quantities such as maximum softmax probability or energy scores [9], [10]. Other methods measure sensitivity to input transformations, density in feature space, local intrinsic dimensionality, Mahalanobis distance, or related representation statistics [11]–[13]. More recent VLM-specific methods analyze geometric properties of pretrained multimodal embeddings, attention responses, or learned semantic priors [14]–[17]. These approaches demonstrate that adversarial perturbations can leave detectable signatures beyond the final prediction.

Our motivation is that a frozen VLM provides another source of information that is naturally available but not fully represented by a single confidence score or embedding-space distance: its pattern of responses across many text prompts. Given an image and a collection of text prompts, a CLIP-style model produces a vector of image–text similarities rather than a single scalar output. This vector describes how the image is positioned relative to multiple directions in the joint vision– language space. Adversarial perturbations that alter the image representation can therefore change not only the predicted dataset class, but also the distribution, local relationships, reference deviations, and transformation behavior of these prompt-conditioned responses.

This observation motivates treating adversarial detection as a response-profiling problem. Instead of asking only whether the input has low confidence or lies far from clean samples in the image embedding space, we characterize how the input interacts with a fixed semantic reference space. We use general prompts that are separate from the dataset class prompts used by the zero-shot classifier. Their image–text similarities provide a reusable representation from which we can extract several complementary measurements.

The novelty of our approach lies in constructing a compact detector representation from multiple properties of this promptconditioned response. First, category-level statistics summarize how similarity mass is distributed across broad semantic groups. Second, we use a graph built from text-embedding relationships to measure global and local variation among prompt responses. Third, clean-reference residuals quantify how the observed response differs from response distributions estimated from clean training images. Fourth, graph diffusion measures how the response changes under smoothing over related prompts. Finally, weak transformations of the input provide stability measurements that characterize how the VLM response changes under small image modifications. These components are concatenated into a compact response profile and classified by a lightweight multilayer perceptron.

This design differs from existing VLM-based adversarial detectors in both the representation analyzed and how the pretrained model is used. Geometry-based methods such as GeoDetect operate primarily on distances or density-related properties of pretrained embeddings [15], while prompt-based irrelevant probing (PIP) uses attention responses generated by irrelevant probing questions [16]. Our method instead operates directly on the pattern of image–text similarities generated by a fixed prompt bank and combines several summaries of that response into a single detector representation. The underlying image encoder, text encoder, prompt embeddings, and prompt graph remain frozen; only the small detector operating on the extracted features is trained.

An additional motivation for this design is interpretability at the feature-group level. Because the final detector operates on an explicitly constructed low-dimensional representation rather than a newly learned VLM representation, we can evaluate the contributions of different signal families separately. We can therefore remove or combine category, graph, residual, and transformation-stability components without modifying the frozen backbone. This lets us study the detector as a composition of measurable response properties rather than an opaque classifier operating directly on high-dimensional pretrained embeddings.

We evaluate the proposed detector across CIFAR-10, STL-10, and Tiny-ImageNet using ViT-B/16, ViT-B/32, and ConvNeXt CLIP-style backbones. The evaluation covers PGD, MI-FGSM, CW, AutoAttack, and flow-based attacks under both standard attack-specific and cross-attack settings. We additionally evaluate a confidence-based MSP detector, compare against embedding-geometry detectors under a controlled protocol, measure the contribution of individual feature groups through controlled ablations, quantify inference cost, and study detectoraware attacks. Together, these experiments evaluate both the effectiveness and generalization of the proposed response representation and the contribution of its individual components.

Our contributions are summarized as follows:

• We introduce adversarial image detection through promptconditioned response profiling, using the pattern of image– text similarities produced by a frozen VLM as the basis for detection.

• We develop a compact representation that combines category-level statistics, prompt-graph measurements, clean-reference residuals, graph-diffusion features, and transformation-stability features while keeping the underlying VLM fixed.

• We use a lightweight MLP detector that operates only on the extracted response profile, enabling analysis of individual feature groups without fine-tuning the VLM backbone.

• We provide a multi-dataset, multi-backbone, and multiattack evaluation including standard and cross-attack detection, a confidence baseline, controlled comparisons with embedding-geometry detectors, feature ablations, runtime measurements, and detector-aware evaluation.

## II. BACKGROUND

This section reviews three groups of related work: adversarial example detection, VLM methods for semantic detection, and adversarial detection methods designed for VLMs. This context positions our prompt-conditioned response-profiling approach relative to existing confidence-, geometry-, and VLM-based detectors.

## A. Adversarial example detection

Adversarial example detection has been widely studied for image classifiers. Early methods use confidence-based scores, including maximum softmax probability (MSP) and energybased scoring, to identify abnormal inputs [9], [10]. Other approaches compare model behavior under input transformations, such as feature squeezing, or estimate sample density in deep feature spaces [11], [18]. Representation-based methods detect adversarial inputs using feature statistics or geometry, including local intrinsic dimensionality (LID), Mahalanobis distance, and statistical tests [12], [13], [19]. Nearest-neighbor methods further show that local representation structure can provide useful detection signals [20]. These methods can often identify adversarial inputs by detecting abnormal model behavior, but they are primarily designed for unimodal classifiers.

## B. VLMs for out-of-distribution and semantic detection

VLMs provide semantic image-text representations that are useful for out-of-distribution (OOD) detection. CLIP aligns images and text in a shared embedding space [1]. Building on this, maximum concept matching (MCM) uses image-text similarity scores for OOD detection [21]. CLIPN introduces negation-aware prompts for zero-shot OOD rejection [22]. Negative-label methods use auxiliary labels that describe concepts absent from the input or outside the target label set to improve semantic separation [23]–[25], while SimLabel uses consistency across related labels [26]. Recent work further analyzes the mechanisms and sensitivities of VLM-based OOD detection [27], and surveys summarize progress with CLIP-like models [28].

These works show the value of semantic alignment for detecting distribution shifts. However, they mainly address OOD detection (not adversarial detection) and usually rely on class-specific prompts or aggregated image-text similarities.

## C. Adversarial detection with VLMs

Recent work has studied adversarial detection directly in vision-language settings. GAD-VLP applies geometric measures to representations extracted from unimodal or multimodal vision-language pre-trained model (VLP) encoders [14]. These measures include LID, k-NN distance, Mahalanobis distance, and kernel density estimation (KDE). GeoDetect studies similar geometric signals and connects adversarial detection to offmanifold behavior in anisotropic VLP embedding spaces [15]. PIP detects adversarial examples in large vision-language models (LVLMs) using attention patterns induced by irrelevant probe questions [16]. Another recent method uses CLIP text embeddings as semantic priors to guide learned latent representations for adversarial detection [17].

These methods show that VLMs provide useful signals for adversarial detection. They mainly focus on embedding geom etry, attention maps, or learned latent spaces. In contrast, our method analyzes image-text similarity patterns from semantic prompts that are not tied to dataset labels. It models prompt relationships with a graph and extracts category, residual, and stability features from a frozen VLM.

## D. Relationship to prior VLM-based detectors

GAD-VLP and GeoDetect detect adversarial inputs from geometric deviations in VLP embedding spaces, using signals such as k-nearest-neighbor distance, Mahalanobis distance, KDE, and LID [14], [15]. PIP instead probes large VLMs with irrelevant questions and classifies attention-derived responses [16]. Other approaches use CLIP text embeddings as semantic priors for learned adversarial detectors [17].

Our method differs in the representation supplied to the detector. It operates on prompt-conditioned image–text responses and combines category-level statistics, prompt-graph measurements, clean-reference residuals, graph diffusion, and transformation stability. The VLM itself remains frozen, and only the lightweight detector is trained.

## III. METHOD

This section outlines the proposed framework. First, we formally define the problem setup and specify the threat model. Subsequently, we establish the semantic anchor space, followed by the graph-structured semantic representation, semantic residual modeling, and semantic stability under small transformations. Finally, we provide a technical description of the final detector.

## A. Problem setup and notation

Let $x \in \mathcal { X }$ denote an input image. The detector receives x and outputs a score indicating whether the image is clean or adversarial. We use a pretrained vision-language model as a frozen feature extractor, with image encoder $f : \mathcal { X } \to \mathbb { R } ^ { d }$ and text encoder $g : \mathcal { P } \overset { } {  } \mathbb { R } ^ { d }$ , where $\mathcal { P } = \{ t _ { 1 } , \ldots , t _ { M } \}$ is a set of semantic prompts, M is the number of prompts, and d is the embedding dimension.

The normalized image embedding is $z _ { I } ( x ) = f ( x ) / \| f ( x ) \| _ { 2 }$ and the normalized prompt embedding is $\begin{array} { r l } { z _ { T } ( t _ { i } ) } & { { } = } \end{array}$ $g ( t _ { i } ) / \| g ( t _ { i } ) \| _ { 2 }$ . Stacking all prompt embeddings gives

$$
Z _ { T } = \left[ \begin{array} { c } { z _ { T } ( t _ { 1 } ) ^ { \top } } \\ { \vdots } \\ { z _ { T } ( t _ { M } ) ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { M \times d } .\tag{1}
$$

The semantic similarity vector is $s ( x ) = Z _ { T } z _ { I } ( x )$ , where $s ( x ) \in \mathbb { R } ^ { M }$

Figure 1 illustrates the frozen VLM similarity computation for one CIFAR-10 test image.

Prompts are grouped into K semantic categories ${ \mathcal { C } } =$ $\{ C _ { 1 } , \ldots , C _ { K } \}$ . From $s ( x )$ , we construct a feature vector $\phi ( \boldsymbol { x } ) \in \mathbb { R } ^ { d _ { \phi } }$ , where $d _ { \phi }$ denotes the feature dimension. We then train a lightweight multilayer perceptron (MLP) detector $D _ { \theta } \ : \ \mathbb { R } ^ { d _ { \phi } } \ \to \ \mathbb { R }$ , where θ denotes the learnable MLP parameters. Higher detector scores indicate a higher likelihood of adversarial input.

## B. Threat model

For each experiment, the target VLM uses a fixed visual backbone, such as ViT-B/16 [29], ViT-B/32 [29], or ConvNeXt [30]. The selected backbone defines the CLIP-style VLM used in each experiment. We use this VLM to generate adversarial images by attacking its zero-shot image-text classification logits, and we also use the same frozen VLM to extract the semantic features used by the detector.

Let $\{ c _ { 1 } , \ldots , c _ { N } \}$ denote the dataset class names. To generate adversarial examples, the attacker constructs class-level prompts of the form “a photo of a {class}.” Let

$$
U = \left[ \overset { u _ { 1 } ^ { \top } } { \vdots } \right] \in \mathbb { R } ^ { N \times d }\tag{2}
$$

be the normalized text embeddings of these class-level prompts, where N is the number of dataset classes and $u _ { i } \in \mathbb { R } ^ { d }$ is the text embedding of the i-th class prompt.

The attack uses image-text logits $\ell ( x ) = \rho U z _ { I } ( x )$ , where ρ is the learned logit scale of the VLM if available.

Given a clean image x with class label $y _ { \mathrm { c l s } } .$ , the adversary creates $x _ { \mathrm { a d v } }$ by optimizing a loss over these logits. For untargeted gradient-based attacks, this objective can be written as

$$
x _ { \mathrm { a d v } } = \arg \operatorname* { m a x } _ { x ^ { \prime } \in B ( x ) } \mathcal { L } _ { \mathrm { C E } } ( \ell ( x ^ { \prime } ) , y _ { \mathrm { c l s } } ) ,\tag{3}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the cross-entropy loss and $B ( x )$ is the allowed perturbation set around x.

The attacker has white-box access to the target VLM used for attack generation, including its image encoder, class-level text embeddings, logits, labels, and gradients. This access enables attacks such as PGD [7], MI-FGSM [31], CW [32], AutoAttack [33], and flow-based attacks [34] to change the zero-shot classification scores by comparing image embeddings with dataset-class prompt embeddings.

## C. Semantic anchor space

We aggregate prompt-level similarities into category-level statistics to summarize how the input aligns with broad semantic groups.

![](images/963812d0516fe9ebbe1c32c1a3d34267549777fd095859e0c778c034d1d71e0a.jpg)  
Fig. 1: Conceptual example of image–text similarity computation for a CIFAR-10 test sample labeled as a ship using a CLIP-style VLM with ViT-B/16 as the backbone. The image encoder maps the input to the normalized image embedding $z _ { I } ( x )$ , while the text encoder maps the semantic prompts to text embeddings stacked in $Z _ { T }$ . The semantic response vector is computed as $s ( x ) = Z _ { T } z _ { I } ( x )$ and contains one similarity value per prompt. For visualization, the figure shows only selected high-scoring prompts and their associated semantic categories rather than the complete prompt set. The full detector uses al seven semantic categories defined in the method.

We normalize the similarity vector as $\hat { s } ( x ) = s ( x ) / \| s ( x ) \| _ { 2 }$ For each category $C _ { k }$ , define:

$$
\begin{array} { l l } { m _ { k } ( x ) = \displaystyle \sum _ { i \in C _ { k } } \operatorname* { m a x } ( 0 , \hat { s } _ { i } ( x ) ) , } & { \quad \mu _ { k } ( x ) = \displaystyle \frac { 1 } { | C _ { k } | } \sum _ { i \in C _ { k } } } \\ { \sigma _ { k } ( x ) = \displaystyle \sqrt { \frac { 1 } { | C _ { k } | } \sum _ { i \in C _ { k } } ( \hat { s } _ { i } ( x ) - \mu _ { k } ( x ) ) ^ { 2 } } , } & { \quad \psi _ { k } ( x ) = \displaystyle \operatorname* { m a x } _ { i \in C _ { k } } \hat { s } _ { i } ( x ) } \end{array}\tag{4}
$$

where $m _ { k } ( x )$ measures the positive semantic mass of category $C _ { k }$ , while $\mu _ { k } ( x ) , \sigma _ { k } ( x )$ , and $\psi _ { k } ( x )$ summarize the average, spread, and strongest prompt response within that category.

To obtain a normalized semantic distribution, we define:

$$
\tilde { m } _ { k } ( x ) = \frac { m _ { k } ( x ) } { \sum _ { j = 1 } ^ { K } m _ { j } ( x ) + \epsilon } ,\tag{5}
$$

where $\epsilon > 0$ is a small constant for numerical stability.

To measure how concentrated the semantic mass is across categories, we compute:

$$
H ( x ) = - \sum _ { k = 1 } ^ { K } \tilde { m } _ { k } ( x ) \log ( \tilde { m } _ { k } ( x ) + \epsilon ) .\tag{6}
$$

To capture separation between dominant categories, we define $\gamma ( x ) = \tilde { m } _ { ( 1 ) } ( x ) - \tilde { m } _ { ( 2 ) } ( x )$ , where $\tilde { m } _ { ( 1 ) } ( x )$ and $\tilde { m } _ { ( 2 ) } ( x )$

are the largest and second-largest normalized category masses. The entropy and margin summarize whether the semantic response is scattered (diffuse) across categories or concentrated sˆ<sub>i</sub>(x),<sub>around</sub> <sub>a</sub> <sub>dominant</sub> <sub>concept.</sub>

## D. Graph-structured semantic representation

This step models relationships between prompts to capture structured semantic behavior. The similarity vector is treated as a signal over a graph defined on the prompt space, where each graph node corresponds to one semantic prompt.

We define a prompt similarity matrix $Q = Z _ { T } Z _ { T } ^ { \top }$ . A sparse adjacency matrix $\overset { \cdot } { A } \in \mathbb { R } ^ { M \times M }$ is constructed from Q using top-r nearest neighbors and then symmetrized, where r is the number of graph neighbors. Let G denote the diagonal degree matrix with entries $\begin{array} { r } { \bar { G } _ { i i } = \sum _ { j = 1 } ^ { M } A _ { i j } } \end{array}$ . The graph Laplacian is $L = G - A$ . To measure global smoothness of the semantic signal, we compute $E _ { G } ( x ) = s ( x ) ^ { \top } L s ( x )$ . A low value of $E _ { G } ( x )$ indicates that related prompts have similar responses, while a high value indicates irregular semantic variation across the prompt graph. To capture local inconsistencies, we define node-wise deviations:

$$
\delta _ { i } ( x ) = s _ { i } ( x ) ^ { 2 } G _ { i i } - 2 s _ { i } ( x ) \sum _ { j = 1 } ^ { M } A _ { i j } s _ { j } ( x ) + \sum _ { j = 1 } ^ { M } A _ { i j } s _ { j } ( x ) ^ { 2 } .\tag{7}
$$

We summarize $\delta ( x )$ using its maximum, mean, variance, and the mean of its $k _ { \mathrm { g r a p h } }$ largest values:

$$
\mathrm { t o p k m e a n } ( \delta ( x ) ) = \frac { 1 } { k _ { \mathrm { g r a p h } } } \sum _ { i \in \mathcal { T } _ { k _ { \mathrm { g r a p h } } } ( x ) } \delta _ { i } ( x ) ,\tag{8}
$$

where $\mathcal { T } _ { k _ { \mathrm { g r a p h } } } ( x )$ indexes the $k _ { \mathrm { g r a p h } }$ largest entries of $\delta ( x )$ These statistics capture whether semantic irregularity is localized to a few prompts or spread across the graph. A visualization of the resulting prompt graph is provided in Section IV-A1.

## E. Semantic residual modeling

This step measures how much the semantic response deviates from clean data distributions. It allows detection based on distributional inconsistency rather than similarity magnitude alone.

For each category $C _ { k } ,$ we estimate a clean mean vector $\bar { s } _ { k }$ and covariance matrix $\Sigma _ { k }$ from clean training samples. The residual for category $C _ { k }$ is defined as:

$$
R _ { k } ( \boldsymbol { x } ) = ( s _ { k } ( \boldsymbol { x } ) - \bar { s } _ { k } ) ^ { \top } \Sigma _ { k } ^ { - 1 } ( s _ { k } ( \boldsymbol { x } ) - \bar { s } _ { k } ) ,\tag{9}
$$

where $s _ { k } ( x )$ is the subvector of $s ( x )$ restricted to prompts in $C _ { k }$

Let $k _ { 1 }$ and $k _ { 2 }$ be the indices of the two largest category masses. We define $R _ { \mathrm { t o p 1 } } ( x ) \ = \ R _ { k _ { 1 } } ( x )$ and $R _ { \mathrm { t o p 2 } } ( x ) \ =$ $R _ { k _ { 2 } } ( x )$ . These residuals measure how far the strongest semantic categories of the input deviate from clean-image statistics. We use the top two categories because adversarial perturbations may affect not only the dominant semantic category but also the nearest competing category.

## F. Semantic stability under small transformations

This step evaluates how semantic representations change under small input transformations. The goal is to detect instability that does not affect visual appearance but alters semantic behavior. Let A be a set of small image transformations. For each transformation $a \in A .$ , we compute the transformed image embedding $z _ { I } ( a ( x ) )$ and its semantic response $s ( a ( x ) )$ To measure embedding stability, we compute the average cosine similarity between the original and transformed image embeddings:

$$
S ( \boldsymbol { x } ) = \frac { 1 } { | \boldsymbol { \mathcal { A } } | } \sum _ { \boldsymbol { a } \in \boldsymbol { \mathcal { A } } } \boldsymbol { z } _ { I } ( \boldsymbol { x } ) ^ { \top } \boldsymbol { z } _ { I } ( \boldsymbol { a } ( \boldsymbol { x } ) ) .\tag{10}
$$

To measure semantic drift under transformations, we compute:

$$
\Delta _ { \cal A } ( x ) = \frac { 1 } { | { \cal A } | } \sum _ { a \in { \cal A } } \| s ( x ) - s ( a ( x ) ) \| _ { 2 } .\tag{11}
$$

We also propagate the semantic signal over the prompt graph using diffusion:

$$
s ^ { ( q + 1 ) } ( x ) = ( 1 - \alpha ) s ^ { ( q ) } ( x ) + \alpha G ^ { - 1 } A s ^ { ( q ) } ( x ) ,\tag{12}
$$

where $\alpha \in ( 0 , 1 )$ controls the diffusion strength, q indexes the diffusion iteration, and $\boldsymbol s ^ { ( 0 ) } ( \boldsymbol x ) = \boldsymbol s ( \boldsymbol x )$

After $T _ { \mathrm { d i f f } }$ diffusion steps, we compute the diffusion deviation:

$$
\Delta _ { \mathrm { d i f f } } ( x ) = \Vert s ^ { ( T _ { \mathrm { d i f f } } ) } ( x ) - s ( x ) \Vert _ { 2 } .\tag{13}
$$

We also compute the cosine similarity between the original and diffused semantic signals:

$$
C _ { \mathrm { d i f f } } ( x ) = \frac { s ( x ) ^ { \top } s ^ { ( T _ { \mathrm { d i f f } } ) } ( x ) } { \| s ( x ) \| _ { 2 } \| s ^ { ( T _ { \mathrm { d i f f } } ) } ( x ) \| _ { 2 } + \epsilon } .\tag{14}
$$

Finally, we compute $H _ { \mathrm { d i f f } } ( x )$ as the entropy of the categorylevel semantic masses after graph diffusion. The transformationstability features measure sensitivity to small image changes, while the diffusion features measure how the semantic response behaves under graph-based smoothing.

## G. Feature representation and detector

We combine all components into a unified feature vector that captures semantic alignment, graph structure, clean-distribution residuals, and stability under small transformations.

$$
\begin{array} { r } { { \bf m } ( x ) = [ m _ { 1 } ( x ) , \ldots , m _ { K } ( x ) ] , \quad \pmb { \mu } ( x ) = [ \mu _ { 1 } ( x ) , \ldots , \mu _ { K } ( x ) ] , } \\ { \pmb { \sigma } ( x ) = [ \sigma _ { 1 } ( x ) , \ldots , \sigma _ { K } ( x ) ] , \quad \pmb { \psi } ( x ) = [ \psi _ { 1 } ( x ) , \ldots , \psi _ { K } ( x ) ] . } \end{array}\tag{15}
$$

The final feature vector is constructed by concatenating category-level, graph-level, diffusion, residual, stability, and normalized embedding-norm features:

$$
\begin{array} { r l } & { \phi ( x ) = \big [ \mathbf { m } ( x ) , \pmb { \mu } ( x ) , \pmb { \sigma } ( x ) , \pmb { \psi } ( x ) , H ( x ) , \gamma ( x ) , } \\ & { \qquad E _ { G } ( x ) , \operatorname* { m a x } ( \delta ( x ) ) , \operatorname* { m e a n } ( \delta ( x ) ) , \mathrm { v a r } ( \delta ( x ) ) , } \\ & { \mathrm { t o p k m e a n } ( \delta ( x ) ) , \Delta _ { \mathrm { d i f f } } ( x ) , C _ { \mathrm { d i f f } } ( x ) , H _ { \mathrm { d i f f } } ( x ) , } \\ & { \qquad R _ { \mathrm { t o p 1 } } ( x ) , R _ { \mathrm { t o p 2 } } ( x ) , S ( x ) , \Delta _ { \mathcal { A } } ( x ) , \| z _ { I } ( x ) \| _ { 2 } \big ] . } \end{array}\tag{16}
$$

These components capture complementary effects of adversarial perturbations: category statistics summarize semantic alignment, graph features measure consistency among related prompts, residuals compare responses with clean-image statistics, and stability features measure sensitivity to small transformations. Together, they form a compact semantic profile. Figure 2 illustrates the proposed feature-vector construction using the values computed for one CIFAR-10 test image. A breakdown of the 43 feature dimensions is provided in Section IV-A4.

Before training the lightweight MLP detector $D _ { \theta }$ , we normalize the extracted features using statistics estimated from clean training samples. Let $\mu _ { \phi } \in \mathbb { R } ^ { d _ { \phi } }$ and $v _ { \phi } \in \mathbb { R } ^ { d _ { \phi } }$ denote the feature-wise mean and variance of $\phi ( x )$ over clean training images. We define the whitened feature vector as

$$
\phi _ { w } ( x ) = \frac { \phi ( x ) - \mu _ { \phi } } { \sqrt { v _ { \phi } + \epsilon _ { w } } } ,\tag{17}
$$

where $\epsilon _ { w } > 0$ is a small constant for numerical stability. The MLP detector is trained on $\phi _ { w } ( x )$ rather than the raw feature vector $\phi ( x )$

We train $D _ { \theta }$ using $\phi _ { w } ( x )$ and binary detection labels $y _ { \mathrm { d e t } } \in$ $\{ 0 , 1 \}$ , where $y _ { \mathrm { { d e t } } } = 0$ denotes a clean image and $y _ { \mathrm { { d e t } } } = 1$ denotes an adversarial image. The VLM backbone and prompt

![](images/710ae98a4fee06d0f086fbf6140aa54ebb15a0a80d7a1a9897ec0b075ea55e1e.jpg)  
Fig. 2: Example of feature-vector construction for a CIFAR-10 test image using a CLIP-style VLM with ViT-B/16 as backbone. The figure shows the actual feature groups computed by the pipeline for one sample: category statistics over the seven semantic prompt groups, prompt-graph features including graph energy and local deviations, clean-distribution residuals for the two dominant semantic categories, and stability features under small image transformations. These components are concatenated into the final feature vector ϕ(x) and passed to the MLP detector. For this example, the computed values include $H ( x ) = 1 . 9 2 3 0$ $\gamma ( x ) = 0 . 0 1 1 6 , E _ { G } ( x ) = 1 . 2 0 6 7 , R _ { \mathrm { t o p 1 } } = 4 . 8 5 2 5 , R _ { \mathrm { t o p 2 } } = 6 . 8 6 6 6 , S ( x ) = 0 . 6 7 8 1 , \mathrm { a n d } \Delta _ { A } ( x ) = 0 . 1 2 4 1$

embeddings remain fixed during training. The binary crossentropy loss is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { B C E } } = - \mathbb { E } \Big [ y _ { \mathrm { d e t } } \log \operatorname { s i g m o i d } ( D _ { \theta } ( \phi _ { w } ( x ) ) ) } \\ & { \qquad + \left( 1 - y _ { \mathrm { d e t } } \right) \log ( 1 - \operatorname { s i g m o i d } ( D _ { \theta } ( \phi _ { w } ( x ) ) ) ) \Big ] . } \end{array}\tag{18}
$$

To encourage adversarial samples to receive higher MLP scores than clean samples by at least a margin, we add:

$$
\mathcal { L } _ { \mathrm { m a r g i n } } = \mathbb { E } \left[ \operatorname* { m a x } ( 0 , \tau - D _ { \theta } ( \phi _ { w } ( x _ { \mathrm { a d v } } ) ) + D _ { \theta } ( \phi _ { w } ( x _ { \mathrm { c l e a n } } ) ) ) \right] ,\tag{19}
$$

where $\tau > 0$ is the desired score margin between adversarial and clean samples.

The final objective for training the MLP detector is ${ \mathcal { L } } =$ $\mathcal { L } _ { \mathrm { B C E } } + \lambda \mathcal { L } _ { \mathrm { m a r g i n } } .$ , where $\lambda > 0$ controls the contribution of the margin loss. At inference time, the trained MLP outputs a detector score, and an input is classified as adversarial if $D _ { \theta } ( \phi _ { w } ( x ) ) \geq \eta$ , where η is a threshold selected on validation data.

The feature extraction procedure follows Eqs. (4)–(16): the frozen VLM produces prompt similarities, the semantic profile is constructed and whitened using clean-training statistics, and the MLP is optimized with the objective above. Computational complexity is discussed in Section III-H, while implementation details, attack parameters, and hardware are given in Sections IV-A and IV-B.

## H. Computational complexity

Let r denote the number of graph neighbors, $T _ { \mathrm { d i f f } }$ the number of diffusion steps, |A| the number of input transformations, and $N _ { \mathrm { t r } }$ the number of training samples. The quantities M, $d ,$ and $d _ { \phi }$ are defined above.

The inference complexity is

$$
\mathcal { O } \big ( ( 1 + | A | ) C _ { \mathrm { m o d e l } } + | A | \cdot ( M d + M r ) + T _ { \mathrm { d i f f } } \cdot M r \big ) .\tag{20}
$$

where $C _ { \mathrm { m o d e l } }$ is the cost of a single forward pass of the vision language model. The first term accounts for computing image embeddings for the original input and transformed views. The second term corresponds to similarity computation and graphbased features across transformations. The third term captures diffusion on the prompt graph. In practice, $C _ { \mathrm { m o d e l } }$ dominates, yielding $\mathcal { O } ( ( 1 + | \mathcal { A } | ) C _ { \mathrm { m o d e l } } )$

The training complexity is

$$
\mathcal { O } \big ( N _ { \mathrm { t r } } \cdot [ ( 1 + | A | ) C _ { \mathrm { m o d e l } } + d _ { \phi } ] \big ) .\tag{21}
$$

This reflects that the backbone remains fixed during training and the trainable component operates only on the extracted feature vector.

## IV. EXPERIMENTAL SETUP

We evaluate the proposed detector across multiple image datasets, frozen VLM backbones, and adversarial attack families under standard, cross-attack, controlled-baseline, ablation, and detector-aware settings. This section summarizes the implementation choices, feature construction, detector configuration, attack parameters, and computational environment used throughout the experiments.

## A. Implementation details

This section provides additional details about the promptgraph construction, clean residual statistics, feature normalization, image transformations, detector architecture, attack settings, and computational complexity in our experiments.

1) Prompt graph construction: Prompt embeddings are computed once with the frozen text encoder and reused for all samples. We construct the adjacency matrix by connecting each prompt to its top-r nearest neighbors in normalized text embedding space. We then symmetrize the adjacency matrix before computing the graph Laplacian. This graph is fixed after construction and is shared across all images for a given prompt set. Figure 3 visualizes the resulting prompt graph.

2) Clean residual statistics and feature whitening: Clean residual statistics are estimated only from clean training samples. For each semantic category, we compute the clean mean vector and covariance matrix of the corresponding prompt similarity subvector. These statistics are then used to compute residual features for validation and test samples. In practice, covariance shrinkage is used with shrinkage coefficient 0.20 to improve numerical stability when estimating category-level covariance matrices.

Before training the detector, feature whitening is applied to the final feature vector. The feature-wise mean $\mu _ { \phi }$ and variance $v _ { \phi }$ are estimated from clean training samples, and each feature vector is normalized as

$$
\phi _ { w } ( x ) = \frac { \phi ( x ) - \mu _ { \phi } } { \sqrt { v _ { \phi } + \epsilon _ { w } } } ,\tag{22}
$$

where $\epsilon _ { w }$ is a small constant for numerical stability. This step prevents features with larger numerical ranges from dominating the MLP detector and improves training stability.

3) Transformations and hyperparameters: We use lightweight transformations to measure semantic stability. These include small spatial shifts, mild blur, and minor photometric changes such as contrast adjustment. The transformations are intentionally weak so that the main semantic content of the image is preserved. Stronger transformations may alter the meaning of clean images, while transformations that are too weak may fail to reveal instability introduced by adversarial perturbations.

The graph diffusion strength is set to $\alpha = 0 . 1 5$ to smooth the semantic signal over the prompt graph while preserving local variations. The margin-loss weight is set to $\lambda = 1 0 ^ { - 3 }$ . These values are chosen to balance detection performance, numerical stability, and computational efficiency.

4) Feature-vector dimensionality: The final feature vector has dimension $d _ { \phi } = 4 3$ in all experiments. This follows from the use of $K = 7$ semantic categories and 15 additional scalar features. Specifically, the category-level statistics consist of four vectors for each semantic category: positive semantic masses, category means, category standard deviations, and category maxima. These contribute $4 K = 2 8$ features.

The remaining 15 scalar features are: semantic entropy $H ( x )$ , semantic margin $\gamma ( x )$ , global graph energy $E _ { G } ( x )$ , four localized graph-anomaly statistics, three graph-diffusion and graph-entropy statistics, two clean-distribution residual scores, two transformation-stability scores, and the image-embedding norm. Therefore,

$$
d _ { \phi } = 4 K + 1 5 = 4 \times 7 + 1 5 = 4 3 .\tag{23}
$$

In our implementation, the localized graph-anomaly statistics are the maximum, mean, variance, and top-k mean of the nodewise graph deviations. The graph-diffusion and graph-entropy statistics are the diffusion deviation, diffusion cosine similarity, and graph entropy after diffusion. The two clean-distribution residual scores correspond to the residuals of the two categories with the largest semantic masses. The two transformationstability scores measure semantic stability under small input transformations.

5) Detector architecture: The detector is a lightweight multilayer perceptron trained on the whitened feature vector $\phi _ { w } ( x )$ . In our implementation, the MLP has hidden dimensions 256, 128, and 64, with ReLU activations and dropout after each hidden layer. The final layer outputs a single scalar detector score. The compact architecture combines the lowdimensional response-profile features without fine-tuning the VLM backbone. The VLM backbone, prompt embeddings, prompt graph, and clean residual statistics remain fixed during detector training. Figure 4 illustrates the detector architecture.

6) Attack settings: For the standard and cross-attack evaluations in Section V-A, we use five adversarial attacks: PGD, MI-FGSM, CW, AutoAttack, and Flow. All attacks optimize the zero-shot CLIP-style image-text classification objective constructed from dataset class prompts of the form “a photo of a {class}.”

For PGD and MI-FGSM, we use untargeted $\ell _ { \infty }$ -bounded attacks with random initialization. For CIFAR-10 and Tiny-ImageNet, we use perturbation budget $\epsilon _ { \mathrm { a t k } } ~ = ~ 0 . 0 4$ , step size $\alpha _ { \mathrm { a t k } } ~ = ~ 0 . 0 1$ , and 20 iterations. For STL-10, we use $\epsilon _ { \mathrm { a t k } } = 0 . 0 3 , \alpha _ { \mathrm { a t k } } = 0 . 0 0 7 5$ , and 10 iterations. MI-FGSM additionally uses momentum factor $\mu _ { \mathrm { m o m } } ~ = ~ 1 . 0$ with $\ell _ { 1 } -$ normalized gradients.

![](images/faec3ba85ab16f0d5f528398fb82d8ad7ad8cd3957a8a3e5310fa04a618e4915.jpg)  
Fig. 3: Visualization of the prompt graph used in the proposed detector. Nodes correspond to semantic prompts and are colored by semantic category. Black edges connect top-r nearest neighbors in the frozen text-embedding space and are symmetrized before graph construction. Node size is proportional to graph degree. Blue guide lines connect selected prompt labels to their corresponding nodes and are included only for readability; they are not graph edges. The graph provides the fixed structure used to compute graph-based response features.

The CW attack is implemented with 200 optimization steps, a learning rate of 0.005, a confidence parameter of $\kappa = 1 0$ , and a trade-off coefficient of $c = 5 . 0 \AA$ . It optimizes a sum of squared perturbation magnitude and a margin-based classification loss over the CLIP-style image-text logits.

The Flow attack learns a dense spatial deformation field and warps the image using bilinear sampling. It uses the same dataset-dependent $\epsilon _ { \mathrm { a t k } } , \alpha _ { \mathrm { a t k } }$ , and iteration count as PGD and MI-FGSM. The flow field is initialized randomly, clipped to the allowed range, and optimized using a smoothness term with a weight of 0.05.

For AutoAttack, we use the same dataset-dependent $\ell _ { \infty }$ perturbation budget as above, along with a batch size of 32. On CIFAR-10 and STL-10, we use the standard AutoAttack evaluation protocol. On Tiny-ImageNet, we use the implementation’s iterative fallback mode with 20 steps and step size 2/255 in pixel space.

## B. Hardware and runtime

The experiments were run using NVIDIA H100 NVL and NVIDIA A100 GPUs, depending on resource availability. Most large experimental batches were executed on a workstation equipped with two NVIDIA H100 NVL GPUs, each with approximately 95.8GB of memory. Several additional runs, including diagnostic and ablation runs, were executed on an NVIDIA A100 GPU through a Colab-based environment.

The standard attack-specific evaluation consisted of 45 model-dataset-attack configurations, corresponding to 3 datasets, 3 VLM backbones, and 5 attacks. Additional experiments included cross-attack evaluation, controlled baseline comparisons, ablations, runtime measurements, and detector-aware evaluation. The main experimental runs took approximately 70 hours across the available GPU resources. Runtime varies across datasets, backbones, and attacks. Attacks with iterative optimization, such as CW, are more computationally expensive than shorter iterative attacks such as PGD and MI-FGSM. The dominant computational cost is the frozen VLM forward pass over the original and transformed views. Training the MLP detector is comparatively inexpensive because it operates only on the extracted feature vectors.

![](images/5979703e3a044d1dc1466f6cbb5f488ea29b1c3d6e6c7494038688894a929bec.jpg)  
Fig. 4: Architecture of the lightweight MLP detector. The input is the whitened response-profile feature vector $\phi _ { w } ( x ) \in \mathbb { R } ^ { d _ { \phi } }$ The detector uses three hidden layers with dimensions 256, 128, and 64, each followed by ReLU activation and dropout, and a final linear layer that outputs a scalar detector score. Higher scores indicate a higher likelihood of adversarial input.

## V. RESULTS

This section evaluates the proposed detector from several complementary perspectives, including standard and crossattack detection, controlled comparisons with geometric baselines, inference-time overhead, feature ablations, and detector aware adaptive evaluation.

## A. Standard and cross-attack evaluation

We evaluate the proposed detector on CIFAR-10, STL-10, and Tiny-ImageNet using three CLIP-style VLM backbones: ViT-B/16, ViT-B/32, and ConvNeXt. Five attack families are considered: PGD, CW, MI-FGSM, Flow, and AutoAttack. For each dataset, we construct fixed, disjoint training, validation, and test partitions using a 70%/15%/15% split, and the same partitions are used across detector configurations.

In the standard setting, a separate detector is trained and evaluated for each attack family using adversarial examples generated by that attack on disjoint training, validation, and test samples. In the cross-attack setting, the detector is trained using PGD adversarial examples and then evaluated on CW, MI-FGSM, Flow, and AutoAttack examples without retraining.

Table I reports ROC-AUC on combined clean and adversarial test samples for both settings across all three datasets and backbones.

The detector achieves consistently high discrimination in the standard setting across datasets, attacks, and VLM backbones. Cross-attack performance also remains strong, indicating that a detector trained using PGD examples can retain substantial discriminative ability on attack families not observed during training.

Figure 5 provides a compact visualization of the cross-attack results across datasets, unseen attacks, and VLM backbones.

The cross-attack results are strongest for AutoAttack across most configurations and remain consistently high for MI-FGSM. Transfer to CW and Flow is also maintained across the evaluated backbones. Overall, the results indicate that the detector does not depend exclusively on attack-specific artifacts from the PGD training distribution.

TABLE I: ROC-AUC (%) on clean + adversarial samples. Left: standard attack-specific evaluation. Right: cross-attack evaluation with detectors trained on PGD.
<table><tr><td></td><td colspan="3">Standard</td><td colspan="3">Cross-attack</td></tr><tr><td>Attack</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ConvNeXt</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ConvNeXt</td></tr><tr><td colspan="7">CIFAR-10</td></tr><tr><td>PGD</td><td>98.87</td><td>97.78</td><td>98.96</td><td></td><td></td><td></td></tr><tr><td>CW</td><td>99.59</td><td>97.68</td><td>99.95</td><td>90.35</td><td>92.40</td><td>96.90</td></tr><tr><td>MI-FGSM</td><td>98.58</td><td>97.93</td><td>98.53</td><td>95.07</td><td>95.00</td><td>92.39</td></tr><tr><td>Flow</td><td>99.91</td><td>99.83</td><td>99.93</td><td>96.60</td><td>94.80</td><td>92.06</td></tr><tr><td>AutoAttack</td><td>99.01</td><td>98.56</td><td>98.86</td><td>97.84</td><td>96.20</td><td>97.81</td></tr><tr><td colspan="7">STL-10</td></tr><tr><td>PGD</td><td>98.02</td><td>96.60</td><td>98.67</td><td></td><td></td><td></td></tr><tr><td>CW</td><td>98.23</td><td>97.53</td><td>99.68</td><td>93.40</td><td>92.10</td><td>97.09</td></tr><tr><td>MI-FGSM</td><td>96.98</td><td>95.38</td><td>98.38</td><td>94.06</td><td>93.40</td><td>94.11</td></tr><tr><td>Flow</td><td>99.48</td><td>99.22</td><td>99.76</td><td>92.30</td><td>91.70</td><td>90.13</td></tr><tr><td>AutoAttack</td><td>99.44</td><td>99.31</td><td>99.46</td><td>98.52</td><td>93.00</td><td>98.26</td></tr><tr><td colspan="7">Tiny-ImageNet</td></tr><tr><td>PGD</td><td>98.15</td><td>94.65</td><td>98.97</td><td></td><td></td><td></td></tr><tr><td>CW</td><td>99.52</td><td>91.68</td><td>99.40</td><td>98.71</td><td>91.50</td><td>97.70</td></tr><tr><td>MI-FGSM</td><td>99.77</td><td>94.65</td><td>99.95</td><td>99.50</td><td>92.80</td><td>99.00</td></tr><tr><td>Flow</td><td>99.68</td><td>99.17</td><td>99.37</td><td>96.56</td><td>91.30</td><td>90.13</td></tr><tr><td>AutoAttack</td><td>99.87</td><td>91.04</td><td>99.83</td><td>99.81</td><td>92.20</td><td>97.61</td></tr></table>

## B. Controlled comparison with geometric baselines

We compare the proposed detector with four representative embedding-geometry detectors: k-nearest neighbors (KNN), Mahalanobis distance, kernel density estimation (KDE), and local intrinsic dimensionality (LID). The comparison uses CIFAR-10 and STL-10 with the same frozen ViT-B/16- quickgelu encoder, data splits, sample budgets, attack settings, and validation procedure for all methods.

For Table II, each detector is evaluated independently under PGD, CW, and Flow using the same detector-specific evaluation protocol. Adversarial examples are generated separately for each detector, and the attack procedure includes detector-aware optimization. The final column reports the mean ROC-AUC across the three attack families.

The proposed detector achieves the strongest mean ROC-AUC on both datasets. These results suggest that combining prompt-conditioned category statistics, graph structure, cleanreference residuals, diffusion behavior, and transformation stability provides complementary information to direct embeddingspace measurements.

![](images/50daf8cf5a96d7bdcfc884d96ce0137c72db7e6f5659519f6b4978d1b02d758e.jpg)  
Fig. 5: Cross-attack ROC-AUC (%) for detectors trained on PGD adversarial examples and evaluated on unseen attack families. Results are shown across three datasets and three VLM backbones. Higher values indicate stronger transfer of the learned response-profile detector across attacks.

TABLE II: Controlled comparison with geometric detectors. Entries are ROC-AUC (%), and the final column reports mean ROC-AUC across PGD, CW, and Flow.
<table><tr><td>Dataset</td><td>Detector</td><td>PGD</td><td>CW</td><td>Flow</td><td>Mean</td></tr><tr><td rowspan="5">CIFAR-10</td><td>KNN</td><td>73.5</td><td>70.8</td><td>63.2</td><td>69.2</td></tr><tr><td>Mahalanobis</td><td>74.1</td><td>69.9</td><td>64.0</td><td>69.3</td></tr><tr><td>KDE</td><td>72.8</td><td>68.7</td><td>65.1</td><td>68.9</td></tr><tr><td>LID</td><td>68.4</td><td>66.8</td><td>61.7</td><td>65.6</td></tr><tr><td>Proposed</td><td>82.4</td><td>78.6</td><td>76.9</td><td>79.3</td></tr><tr><td rowspan="5">STL-10</td><td>KNN</td><td>77.6</td><td>74.9</td><td>69.0</td><td>73.8</td></tr><tr><td>Mahalanobis</td><td>76.8</td><td>74.1</td><td>68.6</td><td>73.2</td></tr><tr><td>KDE</td><td>78.4</td><td>76.2</td><td>71.1</td><td>75.2</td></tr><tr><td>LID</td><td>67.0</td><td>64.8</td><td>62.7</td><td>64.8</td></tr><tr><td>Proposed</td><td>86.0</td><td>81.4</td><td>84.1</td><td>83.8</td></tr></table>

## C. Inference-time overhead

We measure batch-size-one, end-to-end GPU inference on an NVIDIA A100. Measurements include the image encoder, feature construction, and detector, while excluding data loading, model loading, and one-time calibration. The full detector processes the original image together with three weakly transformed views and therefore requires four image-encoder passes. We additionally evaluate a one-view configuration that processes only the original image and does not compute transformation-stability features.

The full configuration requires additional computation because the frozen image encoder is evaluated on four views of each input. The one-view configuration reduces the number of encoder evaluations, while the lightweight MLP itself contributes only a small portion of the total inference cost.

## D. MSP-based baseline

We compare against a confidence-based detector derived from maximum softmax probability (MSP), a widely used baseline for identifying abnormal inputs [9]. The baseline follows a CLIP-style zero-shot classification setting in which class probabilities are obtained from image–text similarity logits [1].

TABLE III: Batch-size-one inference latency and throughput on an NVIDIA A100.
<table><tr><td>Dataset</td><td>Method</td><td>Latency (ms/image)</td><td>FPS</td></tr><tr><td rowspan="5">CIFAR-10</td><td>MSP</td><td>5.88</td><td>170.10</td></tr><tr><td>GeoDetect-KNN</td><td>6.01</td><td>166.50</td></tr><tr><td>GeoDetect-Mahalanobis</td><td>6.12</td><td>163.30</td></tr><tr><td>Proposed, one view</td><td>12.83</td><td>77.97</td></tr><tr><td>Proposed, four views</td><td>29.27</td><td>34.16</td></tr><tr><td rowspan="5">STL-10</td><td>MSP</td><td>5.84</td><td>171.14</td></tr><tr><td>GeoDetect-KNN</td><td>6.15</td><td>162.68</td></tr><tr><td>GeoDetect-Mahalanobis</td><td>6.19</td><td>161.52</td></tr><tr><td>Proposed, one view</td><td>12.89</td><td>77.56</td></tr><tr><td>Proposed, four views</td><td>30.05</td><td>33.28</td></tr></table>

TABLE IV: ROC-AUC (%) of MSP-based detection across datasets and models.
<table><tr><td>Dataset / Attack</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ConvNeXt</td></tr><tr><td colspan="4">CIFAR-10</td></tr><tr><td>PGD</td><td>89.95</td><td>91.68</td><td>94.04</td></tr><tr><td>MI-FGSM</td><td>92.59</td><td>93.86</td><td>93.79</td></tr><tr><td>CW</td><td>70.39</td><td>65.89</td><td>73.02</td></tr><tr><td>Flow</td><td>94.04</td><td>93.69</td><td>95.83</td></tr><tr><td>AutoAttack</td><td>94.37</td><td>94.07</td><td>94.88</td></tr><tr><td colspan="4">STL-10</td></tr><tr><td>PGD</td><td>67.43</td><td>69.23</td><td>78.18</td></tr><tr><td>MI-FGSM</td><td>69.69</td><td>68.83</td><td>78.54</td></tr><tr><td>CW</td><td>62.62</td><td>58.27</td><td>74.23</td></tr><tr><td>Flow</td><td>74.82</td><td>72.27</td><td>79.27</td></tr><tr><td>AutoAttack</td><td>88.20</td><td>89.99</td><td>91.22</td></tr><tr><td colspan="4">Tiny-ImageNet</td></tr><tr><td>PGD</td><td>99.42</td><td>89.42</td><td>89.87</td></tr><tr><td>MI-FGSM</td><td>99.88</td><td>93.61</td><td>94.95</td></tr><tr><td>CW</td><td>99.98</td><td>99.50</td><td>98.77</td></tr><tr><td>Flow</td><td>99.45</td><td>90.31</td><td>68.05</td></tr><tr><td>AutoAttack</td><td>99.96</td><td>96.99</td><td>95.27</td></tr></table>

The baseline uses class-level prompts of the form “a photo of a {class}” and computes a softmax distribution over the dataset classes. Given image–text logits $z _ { i } ,$ the class probabilities are

$$
p _ { i } = \frac { \exp ( z _ { i } / T ) } { \sum _ { j } \exp ( z _ { j } / T ) } ,\tag{24}
$$

where $T$ is a temperature parameter. The detection score is MSP $\mathbf { \boldsymbol { \mathsf { \Pi } } } ( \mathbf { \boldsymbol { \mathscr { x } } } ) = \operatorname* { m a x } _ { i } p _ { i }$

We select the detection threshold using validation data. Table IV reports ROC-AUC across datasets, backbones, and attacks.

MSP performance varies substantially across datasets and attacks. In contrast, the proposed detector maintains consistently high ROC-AUC across the standard configurations in Table I, indicating that the response-profile representation provides a more stable detection signal than classification confidence alone.

TABLE V: Ablation study on CIFAR-10 with ViT-B/16 under PGD. Dim. denotes feature dimension, Acc. denotes accuracy, and TPR@5 denotes the true-positive rate at 5% false-positive rate. Acc., F1, AUC, and TPR@5 are reported in percent.
<table><tr><td>Variant</td><td>Dim.</td><td>Acc.</td><td>F1</td><td>AUC</td><td>TPR@5</td></tr><tr><td>Category only</td><td>30</td><td>87.08</td><td>87.30</td><td>94.69</td><td>74.10</td></tr><tr><td>Category + graph</td><td>38</td><td>90.75</td><td>90.82</td><td>96.71</td><td>83.95</td></tr><tr><td>Category + residual</td><td>32</td><td>88.12</td><td>88.05</td><td>95.29</td><td>76.45</td></tr><tr><td>Category + stability</td><td>32</td><td>92.10</td><td>92.05</td><td>97.78</td><td>88.25</td></tr><tr><td>Category + graph + residual</td><td>40</td><td>90.90</td><td>90.76</td><td>97.18</td><td>86.55</td></tr><tr><td>Category + graph + stability</td><td>40</td><td>93.60</td><td>93.53</td><td>98.55</td><td>92.20</td></tr><tr><td>Full model</td><td>43</td><td>94.08</td><td>93.94</td><td>98.70</td><td>92.85</td></tr></table>

## E. Ablation study

We conduct an ablation study on CIFAR-10 with the ViT-B/16 backbone under PGD. Each variant uses the same training, validation, and test protocol, and only the feature groups provided to the MLP detector are changed. This experiment measures the contribution of the major components of the response-profile representation.

Table V shows that category-level features alone provide a substantial detection signal. Adding graph features improves both ROC-AUC and TPR at 5% FPR, indicating that relationships among prompt responses provide additional information beyond category-level statistics. Residual features also improve over the category-only representation.

Transformation-stability features provide the largest individual improvement over category-level features. Combining category, graph, and stability features further improves performance, while the complete feature representation achieves the strongest overall result. The ablation therefore indicates that the detector benefits from complementary response properties rather than from a single feature group.

## VI. DISCUSSION AND FUTURE WORK

The detector uses 70 general prompts organized into seven broad categories: animal, vehicle, person, food, object, scene, and plant. The goal is not to reproduce dataset labels, but to provide a reusable semantic reference space from which multiple properties of the VLM response can be measured. The ablation results show that category-level statistics already provide a substantial detection signal, while graph, residual, and transformation-stability features provide complementary information. Transformation stability gives the largest individual improvement, and the complete response-profile representation achieves the strongest overall performance.

The cross-attack results further show that detectors trained on PGD retain strong discrimination against CW, MI-FGSM, Flow, and AutoAttack across the evaluated VLM backbones. In the controlled comparison, the proposed response-profile representation also achieves higher mean ROC-AUC than the evaluated embedding-geometry detectors under a common protocol. Together, these results support the use of promptconditioned response profiling as a detection signal that captures information beyond classification confidence and direct embedding-space geometry.

The transformed views contribute strongly to detection performance while also accounting for most of the additional inference cost measured in Section V-C. The detector-aware evaluation further shows that the response-profile representation retains substantial discriminative ability under adaptive optimization.

Future work can explore automatic prompt construction and weighting, more efficient transformation-stability features, broader multimodal architectures, real-world distribution shifts, and additional forms of detector-aware training and adaptive evaluation.

## VII. CONCLUSION

We presented a semantic-response profiling framework for adversarial image detection with frozen vision-language models. The detector summarizes image–text responses using category, graph, residual, diffusion, and transformation-stability features while leaving the VLM backbone fixed. Across multiple datasets, backbones, and attacks, the method achieves strong standard detection performance and strong cross-attack transfer. Controlled comparisons show that the proposed responseprofile representation provides information beyond confidencebased and geometric embedding scores, while feature ablations demonstrate complementary contributions from the different feature groups, with transformation stability providing the largest individual improvement. Detector-aware evaluation further shows that the detector retains substantial discrimination under adaptive optimization. Overall, these results support semantic-response profiling as an effective complementary signal for adversarial image detection in frozen VLMs.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[2] J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” Advances in neural information processing systems, vol. 34, pp. 9694–9705, 2021.

[3] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International conference on machine learning. PMLR, 2022, pp. 12 888–12 900.

[4] X. Zhou, M. Liu, E. Yurtsever, B. Zagar, W. Zimmer, H. Cao, and A. Knoll, “Vision language models in autonomous driving: A survey and outlook,” IEEE Transactions on Intelligent Vehicles, pp. 1–20, 2024.

[5] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” in International Conference on Learning Representations, 2014.

[6] I. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in International Conference on Learning Repre sentations, 2015.

[7] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks,” in International Conference on Learning Representations, 2018.

[8] Y. Zhao, T. Pang, C. Du, X. Yang, C. Li, N.-M. M. Cheung, and M. Lin, “On evaluating adversarial robustness of large vision-language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 54 111– 54 138, 2023.

[9] D. Hendrycks and K. Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks,” in International Conference on Learning Representations, 2017.

[10] W. Liu, X. Wang, J. Owens, and Y. Li, “Energy-based out-of-distribution detection,” Advances in neural information processing systems, vol. 33, pp. 21 464–21 475, 2020.

[11] X. Weilin, E. David, and Q. Yanjun, “Feature squeezing: Detecting adversarial examples in deep neural networks,” Proceedings 2018 Network and Distributed System Security Symposium, 2018.

[12] X. Ma, B. Li, Y. Wang, S. Erfani, S. Wijewickrema, G. Schoenebeck, D. Song, M. Houle, and J. Bailey, “Characterizing adversarial subspaces using local intrinsic dimensionality,” in 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceed ings. United States of America: International Conference on Learning Representations (ICLR), 2018.

[13] K. Lee, K. Lee, H. Lee, and J. Shin, “A simple unified framework for detecting out-of-distribution samples and adversarial attacks,” Advances in neural information processing systems, vol. 31, 2018.

[14] A. Hasanebrahimi, H. Huang, S. M. Erfani, J. Bailey, and C. Leckie, “GAD-VLP: Geometric adversarial detection for vision-language pretrained models,” 2025.

[15] A. Hasanebrahimi, H. Huang, C. Leckie, J. Bailey, and S. Erfani, “Geodetect: Geometric adversarial detection for vlps,” in Computer Vision – ECCV 2026. Cham: Springer Nature Switzerland, 2026, pp. 549–568.

[16] Y. Zhang, R. Xie, J. Chen, X. Sun, and Y. Wang, “Pip: Detecting adversarial examples in large vision-language models via attention patterns of irrelevant probe questions,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 11 175–11 183.

[17] J. Ma, P. Wang, and J. Liu, “Clip-guided adversarial example detection via latent mutual information,” in Neural Information Processing. Singapore: Springer Nature Singapore, 2026, pp. 245–260.

[18] R. Feinman, R. R. Curtin, S. Shintre, and A. B. Gardner, “Detecting adversarial samples from artifacts,” arXiv preprint arXiv:1703.00410, 2017.

[19] K. Grosse, P. Manoharan, N. Papernot, M. Backes, and P. McDaniel, “On the (statistical) detection of adversarial examples,” arXiv preprint arXiv:1702.06280, 2017.

[20] G. Cohen, G. Sapiro, and R. Giryes, “Detecting adversarial samples using influence functions and nearest neighbors,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 453–14 462.

[21] Y. Ming, Z. Cai, J. Gu, Y. Sun, W. Li, and Y. Li, “Delving into out-ofdistribution detection with vision-language representations,” Advances in neural information processing systems, vol. 35, pp. 35 087–35 102, 2022.

[22] H. Wang, Y. Li, H. Yao, and X. Li, “Clipn for zero-shot ood detection: Teaching clip to say no,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 1802–1812.

[23] X. JIANG, F. Liu, Z. Fang, H. Chen, T. Liu, F. Zheng, and B. Han, “Negative label guided ood detection with pretrained vision-language models,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 14 913–14 941.

[24] T. Li, G. Pang, X. Bai, W. Miao, and J. Zheng, “Learning transferable negative prompts for out-of-distribution detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 17 584–17 594.

[25] Z. He, C. Zhao, M. Shao, X. Wu, X. Zhao, D. Li, Q. Tian, and L. Yu, “Outof-distribution detection with positive and negative prompt supervision using large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, 2026, pp. 21 699–21 707.

[26] S. Zou, X. Tian, Q. Zhao, Z. Yang, and J. Zhang, “Simlabel: Consistencyguided ood detection with pretrained vision-language models,” in Australasian Joint Conference on Artificial Intelligence. Springer, 2025, pp. 110–121.

[27] Y. Lee, X. Cao, W. Ye, J. Yao, J. Song, and H. T. Shen, “An empirical analysis of vlm-based ood detection: Mechanisms, advantages, and sensitivity,” arXiv preprint arXiv:2509.13375, 2025.

[28] C. Li, E. Zhang, C. Geng, and S. Chen, “Recent advances in out-ofdistribution detection with clip-like models: A survey,” arXiv preprint arXiv:2505.02448, 2025.

[29] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[30] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A convnet for the 2020s,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 976–11 986.

[31] Y. Dong, F. Liao, T. Pang, H. Su, J. Zhu, X. Hu, and J. Li, “Boosting adversarial attacks with momentum,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 9185– 9193.

[32] N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” in 2017 ieee symposium on security and privacy (sp). Ieee, 2017, pp. 39–57.

[33] F. Croce and M. Hein, “Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,” in International conference on machine learning. PMLR, 2020, pp. 2206–2216.

[34] C. Xiao, J.-Y. Zhu, B. Li, W. He, M. Liu, and D. Song, “Spatially transformed adversarial examples,” in International Conference on Learning Representations, 2018.