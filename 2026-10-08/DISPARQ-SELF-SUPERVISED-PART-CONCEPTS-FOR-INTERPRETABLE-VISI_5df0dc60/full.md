# DISPARQ : SELF-SUPERVISED PART CONCEPTS FOR INTERPRETABLE VISION FOUNDATION MODELS

Adam Pardyl<sup>1,2∗</sup>, Siddhartha Gairola<sup>3∗</sup>, Sukrut Rao<sup>3</sup>, Adam Wrobel´ <sup>1,2</sup>, Bartosz Zielinski´ <sup>1,4</sup>, Bernt Schiele<sup>3</sup>, Dawid Rymarczyk<sup>1,5</sup>

![](images/d8241d99aa6c1b2dee992569db1fe1958db287f258651cb940458a60855d8aa9.jpg)  
Figure 1: DisParQ describes images through semantic visual concepts, their exclusive location assignment, and per-concept appearance attributes. Left: a learned interpretable bottleneck on a frozen VFM identifies part concepts and assigns each location to one concept. Attribute examples illustrate similar appearances within the same learned concept. The concept representation is learned without class labels, part annotations, or language. Right: Top four concepts explain a prediction.

## ABSTRACT

Concept-based vision models represent images through an intermediate layer of human-inspectable concepts, so what a model relies on can be traced to those concepts. However, those models are often limited to fixed categories or depend on language to define their concepts. We introduce DisParQ (Discrete Parts with Quantized attributes), a method that learns spatially grounded, discrete concept representations from a powerful frozen vision-only self-supervised backbone. It requires no class labels and no language supervision. Each image patch is assigned to exactly one concept from a learnable prototype dictionary and only a sparse subset of concepts may activate per image. To capture how each concept varies across images (e.g., the type of a “wheel”), we learn continuous residuals alongside the concepts, and then quantize them into discrete attributes. A spatial decoder reconstructs the backbone’s representation from the concepts and attributes alone, so successful reconstruction means that the discrete representation preserves the backbone’s information. We evaluate DisParQ across seven datasets, from general recognition (ImageNet, PartImageNet, Places) to fine-grained benchmarks (CUB, Cars, Dogs, Flowers). We show that DisParQ closely matches its frozen DINOv2 teacher on ImageNet linear probing (83.2% top-1), achieves higher concept consistency than language-aligned models, remains competitive on fine-grained recognition, and enables cross-category part-based retrieval.

## 1 INTRODUCTION

Self-supervised vision transformers produce rich feature representations, serving as versatile foun dation backbones for classification, segmentation, and retrieval (Caron et al., 2021; Oquab et al., 2024). As these models enter safety-critical domains, strong task performance alone is not sufficient, and we also need to understand what semantic concepts a model relies on and where it finds them during inference (Rudin, 2019; Rudin et al., 2022). This need has motivated research on inherently interpretable architectures. However, current paradigms make varying trade-offs in spatial grounding, concept representation, and supervision. Prototype-based networks (Chen et al., 2019; Janusz et al., 2026; Nauta et al., 2023; Rymarczyk et al., 2022) connect visual concepts to fixed class labels and usually require fine-tuning the entire backbone. Concept Bottleneck Models (CBMs) (Koh et al., 2020; Oikarinen et al., 2023; Yang et al., 2023; Yuksekgonul et al., 2023; Rao et al., 2024) often rely on human-specified or language-derived concepts, but lack localized spatial grounding (Prasse et al., 2025). Recent variants such as CFM (Wittenmayer et al., 2026) introduce spatial grounding, yet remain reliant on language-aligned backbones. Sparse autoencoders (SAEs) have been applied to the patch tokens of vision models (Lim et al., 2025; Zaigrajew et al., 2025). They extract features without concept annotations and provide patch-wise spatial attributions, but their continuous, potentially overlapping activations do not assign exactly one concept to each location. Moreover, these methods capture which concepts appear and where, but none encodes how each concept varies across images (e.g., the type of a “wheel”) as a separate attribute. Together, these methods leave open the combination of (1) spatial grounding, (2) one concept assignment per location, (3) attributes that capture how each concept varies across images, and (4) no class labels or language supervision.

![](images/4fccc3cd774cb3a4e0f7fb5bddc460cbbf14f3b07a739dc6e143bade1c1604ee.jpg)  
Figure 2: Concepts retrieve the same part across categories. A query region (black frame) and the four regions of other images with the closest DisParQ concept description (green frame).

We ask an essential question: Can a representation with all four properties be learned entirely from frozen, vision-only self-supervised features, without sacrificing the expressive power of the underlying backbone?

To answer this, we introduce DisParQ (Discrete Parts with Quantized attributes)<sup>1</sup>, a framework that discovers spatially grounded visual concepts from frozen self-supervised representations, assigns exactly one concept to each patch, and captures how each concept varies across images with an attribute, without class labels or language supervision. DisParQ maintains a learnable dictionary of prototype vectors, each representing a global concept. Every patch is assigned to one concept, and only a sparse subset of concepts can activate per image. To preserve fine appearance differences across occurrences of the same concept, we learn a separate attribute codebook per concept. Training proceeds in two stages: we first learn the high-level concepts with continuous residuals, then quantize these residuals into discrete attributes, resulting in a fully discrete bottleneck at inference.

Figure 1 illustrates how DisParQ describes images with spatially grounded part concepts and explains a class prediction. Although only a small set of concepts (at most 16) are active per image, this description costs almost no accuracy: DisParQ matches its frozen backbone on ImageNet (Deng et al., 2009) linear probing (83.2% vs. 83.4%, Figure 5) and stays within 0-2.2 p.p. of it on four fine-grained datasets (Table 3). Among self-supervised methods, it ranks first on most part-quality metrics on PartImageNet (PIN) (He et al., 2022), and its concepts are more consistent across images than language-grounded ones (Table 2). We further confirm this in a user study (Section 4.3).

We also introduce a part-to-part retrieval benchmark, which tests whether a concept names the same part across categories: given a part region, retrieve the same part in other images (Figure 2). DisParQ consistently leads self-supervised baselines (Tables 4 and 19) on PIN and CUB (Wah et al., 2011).

To summarize, our contributions are as follows. (C1) We propose DisParQ, a framework that learns part concepts with all four properties on a frozen self-supervised vision foundation backbone: spatial grounding, one concept per patch, attributes that capture how each concept varies across images, and no class labels or language supervision required. (C2) We introduce a part-to-part retrieval benchmark on PartImageNet and CUB that tests whether a concept names the same part across images and categories. (C3) We study the design of DisParQ on six frozen backbones and across feature extraction, concept budgets and selection, appearance quantization, decoder depth, gradient routes, and training objectives.

Table 1: Properties of interpretable concept representations. DisParQ is the only method with all nine: it states which part concepts occur, where they are and how each one varies across images, without any labels. (✓ yes, (✓) partially, × no)
<table><tr><td></td><td colspan="3">Training</td><td colspan="2">Where</td><td colspan="2">Which &amp; how</td><td colspan="2">Scope</td></tr><tr><td>Method</td><td>No class labels</td><td>No text</td><td>Frozen backbone</td><td>Concept maps</td><td>One per patch</td><td>Fully discrete</td><td>Within-concept attributes</td><td>Task- agnostic</td><td>Any backbone</td></tr><tr><td>ProtoPNet (Chen et al., 2019)</td><td>×</td><td></td><td>×</td><td>V</td><td>×</td><td>×</td><td>×</td><td>×</td><td>X</td></tr><tr><td>PIP-Net (Nauta et al., 2023)</td><td></td><td></td><td>×</td><td></td><td>√</td><td>X</td><td>×</td><td>×</td><td>×</td></tr><tr><td>ProtoQuant (Janusz et al., 2026)</td><td>(√)</td><td>V</td><td>V</td><td>V</td><td>(√)</td><td>(√)</td><td>X</td><td>×</td><td></td></tr><tr><td>PDiscoFormer (Aniraj et al., 2024)</td><td>×</td><td>L</td><td>(√)</td><td>v</td><td>(√)</td><td>×</td><td>(√)</td><td>×</td><td>×</td></tr><tr><td>LF-CBM (Oikarinen et al., 2023)</td><td>×</td><td>×</td><td>V</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td></td></tr><tr><td>SALF-CBM (Benou &amp; Riklin-Raviv, 2025)</td><td>×</td><td>×</td><td></td><td>V</td><td>×</td><td>×</td><td>×</td><td>×</td><td></td></tr><tr><td>PatchSAE (Lim et al., 2025)</td><td></td><td></td><td></td><td></td><td>×</td><td>×</td><td>X</td><td>V</td><td></td></tr><tr><td>MSAE (Zaigrajew et al., 2025)</td><td>V</td><td>V</td><td></td><td>V</td><td>×</td><td>X</td><td>×</td><td>L</td><td>V</td></tr><tr><td>CFM (Wittenmayer et al., 2026)</td><td>V</td><td>V</td><td></td><td>V</td><td>X</td><td>×</td><td>X</td><td>V</td><td>(√)</td></tr><tr><td>DisParQ (ours)</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td></td></tr></table>

We make the following empirical findings, detailed in Section 4.2. (F1) A discrete bottleneck costs almost no accuracy. With only a small set of active concepts $( k \le 1 6 )$ per image, DisParQ stays within $0 . 2 { \mathfrak { p } } . { \mathfrak { p } } .$ . of its teacher on ImageNet and within 2.2 p.p. on four fine-grained datasets, using one configuration for all of them (Figure 5 and Table 3). (F2) Vision-only self-supervision grounds a consistent part vocabulary. Without any language supervision, the concepts of DisParQ are more consistent across images than language-grounded ones, and DisParQ leads self-supervised baselines on most part-quality metrics (Tables 2 and 3). (F3) Concepts preserve their identity across categories. In part-to-part retrieval, DisParQ leads the self-supervised baselines on most metrics, by up to 3.5 mAP on PartImageNet and 5.9 p.p. R@1 on CUB (Tables 4 and 19).

With DisParQ, we show that frozen vision foundation models can be interpreted through discrete parts and their attributes, without any human supervision and at almost no cost in accuracy.

## 2 RELATED WORK

Prototype models classify images by matching patches to learned exemplars (Chen et al., 2019; Nauta et al., 2023), with newer ones using a VFM and quantized prototypes but still classifying through continuous similarity scores with class supervision (Janusz et al., 2026). Part discovery models instead learn a small, fixed set of parts, either without labels (Hung et al., 2019; Amir et al., 2023) or from class supervision (van der Klis et al., 2023; Aniraj et al., 2024). DisParQ learns a large shared dictionary without class labels or a task-specific classifier, letting each image select a few concepts to describe its parts. Concept bottleneck models predict through human-defined or language-generated concepts (Koh et al., 2020; Oikarinen et al., 2023), including spatial concept maps (Benou & Riklin-Raviv, 2025). Sparse autoencoders (SAEs) learn visual dictionaries from model activations (Lim et al., 2025; Zaigrajew et al., 2025), and CFM grounds concepts spatially and names them through a text vocabulary (Wittenmayer et al., 2026). These SAE dictionaries use continuous, overlapping codes that entangle concept identity with appearance. DisParQ learns concepts from visual features alone, assigns each patch exactly one concept, and encodes concept identity and within-concept attributes separately using integer indices alone (Table 1). Existing evaluations measure prototype purity, part alignment, or concept locality and consistency (Nauta et al., 2023; Aniraj et al., 2024; Wittenmayer et al., 2026). Our part-to-part retrieval benchmark tests whether concept descriptions preserve part identity across images and categories by retrieving the same part from all other images (Section 4.1). Appendix A gives the full related-work discussion.

## 3 DISPARQ

DisParQ provides an interpretable, discrete, spatially grounded concept bottleneck on top of a Vision Foundation Model (VFM). An overview of DisParQ is presented in Figure 3. We first extract latent features (Section 3.1), and organize them into a small set of exclusive concept regions to ensure spatial grounding (Section 3.2). Then we encode each region’s appearance to represent details of the concepts (Section 3.3). Lastly, a spatial decoder predicts the global representation from these concepts (Section 3.4). Training transforms continuous appearance attributes to discrete codes (Section 3.5). Additional details are provided in Appendix C.

![](images/c3c34764370dbd4c0912009e61f0b8ce3d2f91082e970fdabbe840eb76853068.jpg)  
Figure 3: DisParQ describes images with concepts and attributes. Fused and upsampled VFM features $z _ { u }$ are matched against prototypes $w _ { n }$ to determine which concepts occur (top-K identities $i _ { 1 : K } )$ and where they occur (assignments $p _ { u }$ at positions u). Discrete attributes $q _ { i _ { k } }$ encode how each concept appears in the image. The assignment map selects the learned concept embeddings $e _ { n }$ and attributes for the decoder, which predicts $\hat { c } ( x )$ by distillation from the VFM’s global representation.

## 3.1 FEATURE EXTRACTION

We build on a pretrained, frozen VFM encoder $f _ { \mathrm { V F M } }$ . An image x is passed through it, producing (1) encoded features (patch tokens) that serve as input to DisParQ, and (2) a global image representation $c ( x )$ (typically the cls token), which is later used as a distillation target. These two outputs serve different purposes. Local features ground concepts in image regions, while the global target encourages their collective description to preserve the image’s semantics.

As different details are encoded at different depths of the $f _ { \mathrm { V F M } }$ , we introduce a lightweight sharedquery attention module $f _ { \mathrm { f u s e } }$ that at each location combines features from different encoder blocks. Then, as the feature map is of low resolution, which results in low precision of concept activations, we upsample it with an AnyUp module $f _ { \mathrm { u p } }$ (Wimmer et al., 2026). AnyUp uses the original normalized image to guide feature upsampling, providing a finer grid for concept assignment. As a result we obtain spatial feature vectors $z _ { u } ,$ , where u is the patch position in the upsampled feature grid.

## 3.2 CONCEPT IDENTIFICATION AND ASSIGNMENT

Concept identification proceeds in two steps: we select a small set of concepts from the shared bank for each image (Figure 3), and then assign each spatial location to one of those concepts (Figure 4).

Selecting concepts. The shared bank must accommodate diverse visual structures across images. At the same time, any single image should have a compact description. That is why we train a large bank of N concept prototypes, from which we select a small subset of K concepts per image $( N = 1 0 2 4$ , K = 16 by default). Each prototype $w _ { n }$ , with $n \in \{ 1 , \ldots , N \}$ , is a learned reference vector. We compare it with the upsampled spatial features $z _ { u }$ to obtain a soft activation map $A _ { n } ,$ and average that map to get concept scores $\chi _ { n } \colon$

$$
\ell _ { n , u } = - \| z _ { u } - w _ { n } \| _ { 2 } ^ { 2 } , \qquad A _ { n , u } = \mathrm { s o f t m a x } _ { n } ( \ell _ { n , u } ) , \qquad \chi _ { n } = \frac { 1 } { H W } \sum _ { u } A _ { n , u } .\tag{1}
$$

Here $H \times W$ is the upsampled feature grid size, and the softmax makes all N prototypes compete at each location. During training, Sinkhorn–Knopp balancing (Sinkhorn & Knopp, 1967) adjusts the scores $\chi _ { n }$ across images in a batch to encourage balanced prototype usage. We select the K concepts with the highest balanced scores for each image. The bank index $i _ { k } \in \mathsf { \bar { \{ 1 , \dots , N \} } }$ identifies its kth selected concept, with $k \in \{ 1 , \ldots , K \}$

Assigning locations. Having chosen which K concepts participate, we then determine where they occur. We recompute the softmax over only the selected concepts, so spatial competition is restricted to the image’s chosen subset. This produces the post-selection soft maps $\bar { A } _ { i _ { k } }$ and hard identity map p shown in Figure 4:

![](images/5ee5b44c6c9fb63e05c89433238f7328559ab08a5945dc83308502318c153213.jpg)  
Figure 4: Separating concept identity and appearance. Selected-concept maps $\bar { A } _ { i _ { k } }$ determine exclusive spatial assignments $p _ { u }$ and pool each concept’s region into $\bar { z } _ { i _ { k } }$ . Its normalized deviation from the concept prototype, $r _ { i _ { k } }$ , describes the occurrence relative to the concept. Product codebooks $B _ { i _ { k } }$ quantize its projection. The resulting attribute $q _ { i _ { k } }$ is added to the concept embedding $e _ { i _ { k } }$ and broadcast to its assigned positions. Adding positional embeddings $\mathrm { P E } _ { u }$ forms the decoder tokens $t _ { u }$

$$
\begin{array} { r } { \bar { A } _ { i _ { k } , u } = \mathrm { s o f t m a x } _ { k } ( \ell _ { i _ { k } , u } ) , \qquad p _ { u } = i _ { \mathrm { a r g m a x } _ { k } } \bar { A } _ { i _ { k } , u } . } \end{array}\tag{2}
$$

The strongest response at each location determines $p _ { u }$ , making the concept assignment mutually exclusive. Training and inference details are specified in Appendix C.2.

## 3.3 ATTRIBUTE ENCODING AND QUANTIZATION

A concept such as a dog’s nose can vary in appearance across images. Describing every appearance difference with a new prototype would fragment the concept bank, whereas ignoring those differences would discard information needed to match the VFM representation. We therefore learn a concept-specific attribute that captures how each occurrence differs from its concept reference.

The attribute pathway in Figure 4 follows three steps: pooling region features, computing their deviation from the prototype, and quantizing that residual. For each selected concept $i _ { k } ,$ the soft map $\bar { A } _ { i _ { k } }$ weights the features $z _ { u }$ to form a region mean $\bar { z } _ { i _ { k } }$ . We normalize both vectors with ν and subtract the prototype from the region mean to compute the residual $r _ { i _ { k } } \colon$

$$
\bar { z } _ { i _ { k } } = \frac { \sum _ { u } \bar { A } _ { i _ { k } , u } z _ { u } } { \sum _ { u } \bar { A } _ { i _ { k } , u } } , \qquad r _ { i _ { k } } = \nu ( \bar { z } _ { i _ { k } } ) - \nu ( w _ { i _ { k } } ) .\tag{3}
$$

Here ν denotes $\ell _ { 2 }$ normalization. The residual $r _ { i _ { k } }$ describes the whole region relative to its concept reference, tying appearance information to that concept occurrence. We encode the residual with product quantization (PQ) (Jegou et al.´ , 2011) using concept-specific codebooks ${ { B } _ { { { i } _ { k } } } }$ . The result is a discrete appearance code and its quantized vector $q _ { i _ { k } }$ , known as an attribute. Pooling and quantization details are given in Appendix C.3.

## 3.4 CONCEPT DECODER

The decoder maps the interpretable concept description to a global image representation for downstream tasks. It combines concept identities, spatial assignments, and appearance attributes to predict the frozen VFM’s representation through feature distillation.

The right portion of Figure 4 shows how each spatial token combines the assigned concept’s embed ding, its appearance attribute, and a positional embedding $\mathrm { P E } _ { u }$ . A concept embedding $e _ { n }$ represents each concept in the decoder and is learned independently from the prototype used in previous steps. At each position $u \in \{ 1 , \dots , H W \}$ }, the identity $p _ { u }$ selects the concept embedding and attribute:

$$
t _ { u } = e _ { p _ { u } } + \mathrm { P E } _ { u } + q _ { p _ { u } } , \qquad { \hat { c } } ( x ) = f _ { \mathrm { d e c } } \left( [ t _ { \mathrm { C L S } } , t _ { 1 } , \dots , t _ { H W } ] \right) .\tag{4}
$$

A small transformer model integrates these tokens and predicts the VFM’s global representation from the output of its learned decoder CLS token $t _ { \mathrm { C L S } }$ (separate from the VFM’s CLS token).

Every image-dependent input to this decoder passes through the concept description. In the quantized model, the selected identities $i _ { 1 : K }$ , assignment map p, and discrete appearance codes completely specify that description. Decoding requires only these discrete quantities, learned lookup tables, and the decoder, making explicit what information survives the bottleneck. The output ${ \hat { c } } ( x )$ has the same dimension as the VFM representation and can be used for downstream tasks such as classification and retrieval.

## 3.5 TRAINING OBJECTIVES AND REGIME

We train through feature distillation without class labels, text descriptions, or region annotations. For an augmented image x and a sampled affine transformation T, each view $v \in \left\{ x , T ( x ) \right\}$ } predicts its corresponding frozen VFM target c(v):

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = \frac { 1 } { 2 } \sum _ { v \in \{ x , T ( x ) \} } \left[ 1 - \cos \bigl ( \hat { c } ( v ) , \mathrm { s t o p g r a d } ( c ( v ) ) \bigr ) \right] ,\tag{5}
$$

The loss is averaged over images, cos denotes cosine similarity and stopgrad stops gradients. Following PDiscoFormer (Aniraj et al., 2024), we regularize assignments to encourage coherent, confident regions and distinct, active concepts. Affine equivariance encourages spatial responses to follow image content under geometric transformations. Appendix C.5 specifies the losses (Figure 7) and weights (Table 6). Appendix D.3 gives the augmentation and optimization recipe.

Stage 1: discovering concepts. We jointly train feature fusion, concept prototypes, the continuous appearance projection, and the decoder. Continuous attributes provide flexibility during concept discovery, while an auxiliary distillation head encourages pooled region features to preserve global image information. Assignments are hard in the forward pass. Stage 2: quantizing attributes. We freeze feature fusion and the prototypes, replace continuous attributes with product-quantized vectors, and train the quantization module, concept embeddings, and decoder through $\mathcal { L } _ { \mathrm { g l o b a l } }$ . The auxiliary head is disabled. The VFM and AnyUp remain frozen throughout both stages. Appendices C.3, C.5, and C.6 detail quantization, gradient handling, and the stage transition.

## 4 EXPERIMENTAL EVALUATION

In this section, we compare DisParQ with concept dictionaries and prototype models. Section 4.1 describes the evaluation setup, Section 4.2 the quantitative results, and Section 4.3 the qualitative results and how DisParQ explains a prediction. The implementation details for training and evaluating DisParQ as well as the baselines are in Appendices C.6 and D.

## 4.1 EXPERIMENTAL SETUP

Datasets. We train one DisParQ model per dataset, without labels. ImageNet-1k (Deng et al., 2009) and Places365 (Zhou et al., 2018) test classification at scale. PartImageNet (He et al., 2022), with masks for 40 part classes, and CUB-200-2011 (Wah et al., 2011), with 15 keypoints per bird, test the part concepts. Stanford Cars (Krause et al., 2013), Stanford Dogs (Khosla et al., 2011) and Oxford Flowers (Nilsback & Zisserman, 2008) test fine-grained classification (Appendix D.1).

Backbone and baselines. The main self-supervised comparison uses frozen DINOv2-B/14 with registers (Oquab et al., 2024; Darcet et al., 2024), except CFM (Wittenmayer et al., 2026), whose released dictionary is built on CLIP-DINOiser (Wysoczanska et al.´ , 2024). We compare DisParQ to (i) sparse autoencoders trained on the same patch tokens: TopK SAE (Gao et al., 2025), Patch-SAE (Lim et al., 2025) and MSAE (Zaigrajew et al., 2025), (ii) the concept dictionary of CFM, and (iii) the label-free first stage of the prototype models PIP-Net (Nauta et al., 2023) and Proto-Quant (Janusz et al., 2026). As unranked references, gray rows give PDiscoFormer (Aniraj et al., 2024) and the classifier stage of PIP-Net and ProtoQuant, which use class labels, and the frozen backbone without a bottleneck. For spatial evaluation and part retrieval, the main self-supervised comparison uses $K = 1 6$ concepts per image, with DisParQ selecting them from $N = 1 0 2 4$ . TopK SAE is trained with this budget. PatchSAE, MSAE and CFM keep their 16 strongest concepts for these evaluations, but use unbudgeted representations for classification and image retrieval. Appendix D states selection rules and protocol exceptions. Appendix F.2 discusses more settings.

Classification. We evaluate the representation that each method passes downstream, which for DisParQ is the CLS token its decoder predicts from the discrete description. For the baselines, the probes read the pooled reconstruction of the sparse autoencoders, the pooled concept activations of CFM, and the prototype scores of PIP-Net and ProtoQuant (see Appendix D.5). We report the top-1 accuracy of a linear probe (LP) and of a weighted 20-nearest-neighbor classifier (kNN).

<table><tr><td>Method</td><td>Spatial grounding</td><td>Discrete</td><td>bottleneck IN-1k↑ Places ↑</td><td></td></tr><tr><td>ViT-B/16, supervised ProtoQuant</td><td>N/A (√)</td><td>N/A (√)</td><td>81.1 80.1</td><td></td></tr><tr><td>CLIP ViT-B/16</td><td>N/A</td><td>N/A</td><td>80.2</td><td>55.1</td></tr><tr><td>DN-CBM</td><td>X</td><td>X</td><td>79.5</td><td>55.1</td></tr><tr><td>CDM</td><td>X</td><td>(√)</td><td>79.3</td><td>52.6</td></tr><tr><td>CFM</td><td>(√)</td><td>X</td><td>78.9</td><td>55.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINOv2-B/14</td><td>N/A</td><td>N/A</td><td>83.4</td><td>55.4</td></tr><tr><td>DisParQ (ours)</td><td>V</td><td>√</td><td>83.2</td><td>55.3</td></tr></table>

![](images/b7f9de022872f96ffaea0ddb550abc65e840e6efc13ccd1e83ee4538f319448e.jpg)  
Figure 5: Left: Large-scale classification (linear probe, top-1). (✓ yes, (✓) partially, × no). Gray rows are unranked references. All methods and their sources are in Table 15 in Appendix F.1. Right: DisParQ achieves the highest perceived part consistency in our user study. Diamonds mark means and bars show interquartile ranges. $\ast \ast \ast _ { p } < 0 . 0 0 1$ , paired Wilcoxon, Holm-corrected.

Part concepts. On PartImageNet, we compare concept maps with part masks: foreground purity $( \mathrm { P u r _ { f g } } )$ , NMI and ARI after mapping each concept to its dominant part $( \mathrm { N M I } _ { m } , \mathrm { A R I } _ { m } )$ (van der Klis et al., 2023; Aniraj et al., 2024), and the mIoU (He et al., 2022) and FG-ARI (Locatello et al., 2020) of the resulting part segmentation. We add the locality, consistency and impurity of CFM (Wittenmayer et al., 2026), which ask whether a concept covers one part and whether it covers the same part in every image. On CUB, we report PIP purity (Nauta et al., 2023) and the normalized mean error of keypoint regression (Kp NME) (Aniraj et al., 2024). Appendix D.6 defines all metrics in detail.

Part-to-part retrieval. We introduce a benchmark that asks whether a concept names the same part across images and categories. Each annotated part region, a mask on PartImageNet or a visible keypoint on CUB, is described by the histogram of the concepts it contains, and the regions of all other images are ranked by cosine similarity. Appendix D.6 gives the details on the benchmark.

## 4.2 QUANTITATIVE RESULTS

We first test whether the discrete bottleneck keeps the backbone’s accuracy (F1), then the quality of its part concepts (F2), and finally whether a concept keeps its identity across categories (F3).

Classification. The discrete description costs almost no accuracy, although only concept identities, regions and quantized attributes reach the decoder. On ImageNet-1k, its linear probe reaches 83.2%, the highest of all concept models in Figure 5 (left) (+3.7 p.p. over DN-CBM) and only $0 . 2 \mathsf { p . p }$ . below its DINOv2 backbone, while the vision-language concept models lose 0.7 to 1.3 p.p. to their CLIP backbone. On Places365, it reaches 55.3%, 0.1 p.p. below both its backbone and CFM. Against the self-supervised baselines of Tables 2 and 3, whose best rows use the same DINOv2 backbone, its LP is $3 . 6 \mathsf { p } . \mathsf { p }$ . higher on PartImageNet and 4.0 to 9.3 p.p. higher on CUB, Cars and Dogs. None of the other methods has concepts that are both exclusive and discrete (Appendix F.1). Quantized attributes retain appearance lost by concepts alone; removing them costs 12.2 p.p. in LP (Appendix E.3).

Part concepts on PartImageNet. Among self-supervised methods, DisParQ is best on 7 of the 10 metrics of Table 2 and second on two more, $\mathrm { P u r _ { f g } }$ and FG-ARI, where the label-free stage of ProtoQuant leads. Its largest margins are on $\mathrm { A R I } _ { m }$ (62.3 vs. 56.3), locality (36.4 vs. 30.5) and consistency, where its concepts are more consistent than those of the language-grounded CFM (16.3 vs. 12.4). Its kNN accuracy is 4.8 p.p. above the best baseline, within 0.2 p.p. of the frozen backbone. It also exceeds the supervised PDiscoFormer on every part and grounding metric except FG-ARI.

Fine-grained datasets. On the four fine-grained datasets, DisParQ is best on all 10 metrics of Table 3 among self-supervised methods (tied on Flowers LP). Its kNN accuracy is 16.7 to 19.8 p.p. above the best baseline on CUB, Cars and Dogs. On CUB, its concepts also localize the keypoints best, with PIP purity 54.5 vs. 37.3 and keypoint error 9.36 vs. 9.87 (Appendix D.6 explains how the purity window is placed on exclusive maps). The supervised PDiscoFormer<sup>2</sup> scores higher on the CUB clustering metrics and on keypoint error, and lower on PIP purity (Table 16 in Appendix F.3).

Table 2: Part concepts on PartImageNet (val, as % except Imp.). Best and second best among self-supervised methods. Subscripts: gap to the best. Gray: unranked references. K/N: concepts per image/dictionary size. Protocol: Appendix D.5. DisParQ shows superior performance on 7/10.
<table><tr><td></td><td></td><td></td><td colspan="2">Downstream</td><td colspan="5">Part quality</td><td colspan="3">Concept grounding</td></tr><tr><td>Method</td><td>Backbone</td><td>K/N</td><td>LP↑</td><td>kNN↑</td><td>Purfg ↑</td><td>NMIm ↑ ARIm ↑ mIoU↑ FG-ARI↑</td><td></td><td></td><td></td><td>Loc.↑</td><td>Cons. ↑</td><td>Imp. ↓</td></tr><tr><td>Supervised (reference)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PDiscoFormer</td><td>DINOv2-B/14</td><td>50/50</td><td>89.8</td><td>88.1</td><td>54.9</td><td>48.4</td><td>37.9</td><td>29.1</td><td>28.8</td><td>34.9</td><td>2.0</td><td>0.505</td></tr><tr><td>PIP-Netstage 2 ProtoQuantstage 2</td><td>DINOv2-B/14 DINOv2-B/14</td><td>16/768</td><td>87.5 87.8</td><td>86.8 89.5</td><td>59.3 71.7</td><td>37.3 55.6</td><td>11.5 30.0</td><td>24.9 26.5</td><td>18.9 19.5</td><td>25.2 13.1</td><td>6.4 4.8</td><td>0.437 0.513</td></tr><tr><td></td><td></td><td>16/2048</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frozen backbone, no concept bottleneck (reference)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLS token Patch tokens, mean-pooled DINOv2-B/14 N/A</td><td>DINOv2-B/14 N/A</td><td></td><td>89.3</td><td>88.9</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td></td><td></td><td>85.7</td><td>79.3</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Self-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CFM</td><td>CLIP-DI-B/16</td><td>16/8192</td><td>72.6</td><td>69.2</td><td>54.7</td><td>51.4</td><td>24.5</td><td>22.7</td><td>12.5</td><td>22.9</td><td>12.4</td><td>0.378</td></tr><tr><td>TopK SAE</td><td>DINOv2-B/14</td><td>16/2048</td><td>80.5</td><td>73.5</td><td>68.8</td><td>58.0</td><td>36.0</td><td>24.8</td><td>16.1</td><td>20.6</td><td>9.8</td><td>0.462</td></tr><tr><td>PatchSAE</td><td>DINOv2-B/14</td><td>16/49152</td><td>84.1</td><td>77.2</td><td>76.2</td><td>60.1</td><td>34.9</td><td>25.0</td><td>17.0</td><td>24.7</td><td>4.4</td><td>0.455</td></tr><tr><td>MSAE</td><td>DINOv2-B/14</td><td>16/8192</td><td>85.6</td><td>79.5</td><td>59.0</td><td>50.3</td><td>23.5</td><td>20.5</td><td>12.1</td><td>18.0</td><td>8.1 10.2</td><td>0.460</td></tr><tr><td>PIP-Netstage 1 ProtoQuantstage 1</td><td>DINOv2-B/14 DINOv2-B/14</td><td>16/768 16/2048</td><td>76.9 85.2</td><td>71.5 83.9</td><td>68.5 82.1</td><td>54.2 68.7</td><td>34.3 56.3</td><td>29.0 33.7</td><td>19.1 28.4</td><td>30.5 18.1</td><td>7.3</td><td>0.376 0.514</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DisParQ (ours)</td><td>DINOv2-B/14</td><td>16/1024</td><td></td><td></td><td></td><td>89.2+3.6 88.7+4.8 81.5-0.6 69.8+1.1</td><td></td><td>62.3+6.0 35.9+2.2</td><td>26.3.2.1</td><td></td><td>36.4+5.9 16.3+3.9 0.385+0.009</td><td></td></tr></table>

Table 3: Fine-grained datasets (test, as % except Kp NME; CFM on CLIP-DINOiser). Marks as in Table 2. For PDiscoFormer some scores are reproduced as in Appendix F.4. <sup>∗</sup>4000 on CUB. Protocol: Appendix D.5. DisParQ consistently outperforms baseline methods on all but one metric.
<table><tr><td rowspan="2">Method</td><td rowspan="2">K/N</td><td colspan="4">CUB-200-2011</td><td colspan="2">Stanford Cars</td><td colspan="2">Stanford Dogs</td><td colspan="2">Oxford Flowers</td></tr><tr><td>LP↑</td><td>kNN↑</td><td>PIP pur. ↑</td><td>KpNME↓</td><td>LP↑</td><td>kNN↑</td><td>LP↑</td><td>kNN↑</td><td>LP↑</td><td>kNN↑</td></tr><tr><td colspan="10">Supervised (reference)</td></tr><tr><td>PDiscoFormer</td><td>K/K</td><td>86.8</td><td>74.8</td><td>40.4</td><td>2.87</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>99.6</td><td>99.7</td></tr><tr><td>PIP-Netstage 2</td><td>16/768</td><td>88.9</td><td>88.9</td><td>47.0</td><td>10.16</td><td>91.4</td><td>87.3</td><td>87.1</td><td>85.2</td><td>99.6</td><td>99.7</td></tr><tr><td>ProtoQuantstage 2</td><td>16/2048*</td><td>84.9</td><td>83.0</td><td>25.2</td><td>12.16</td><td>91.3</td><td>86.3</td><td>86.0</td><td>79.4</td><td>99.6</td><td>99.6</td></tr><tr><td colspan="10">Frozen backbone, no concept bottleneck (reference)</td></tr><tr><td>CLS token</td><td>N/A</td><td>90.4</td><td>86.7</td><td>N/A</td><td>N/A</td><td>93.2</td><td>85.6</td><td>88.6</td><td>86.8</td><td>99.7</td><td>99.7</td></tr><tr><td>Self-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">CFM</td></tr><tr><td></td><td>16/8192</td><td>54.8</td><td>42.8</td><td>16.1</td><td>12.84</td><td>71.0</td><td>61.3</td><td>56.0</td><td>52.5</td><td>90.2</td><td>88.8</td></tr><tr><td>TopK SAE</td><td>16/2048</td><td>36.9</td><td>24.3</td><td>8.7</td><td>10.92</td><td>41.9</td><td>22.7</td><td>61.3</td><td>41.1</td><td>99.4</td><td>99.0</td></tr><tr><td>PatchSAE</td><td>16/49152</td><td>62.0</td><td>33.4</td><td>6.9</td><td>19.12</td><td>74.6</td><td>35.8</td><td>77.6</td><td>53.5</td><td>99.3</td><td>99.0</td></tr><tr><td>MSAE</td><td>16/6144</td><td>73.5</td><td>42.2</td><td>15.1</td><td>12.21</td><td>80.3</td><td>41.7</td><td>79.4</td><td>55.2</td><td>99.6</td><td>99.4</td></tr><tr><td>PIP-Netstage 1</td><td>16/768</td><td>77.8</td><td>68.4</td><td>37.3</td><td>9.87</td><td>82.3</td><td>63.6</td><td>77.8</td><td>69.1</td><td>99.1</td><td>98.4</td></tr><tr><td>ProtoQuantstage 1</td><td>16/2048*</td><td>80.1</td><td>67.7</td><td>24.6</td><td>11.59</td><td>87.0</td><td>60.2</td><td>82.2</td><td>66.9</td><td>99.7</td><td>99.5</td></tr><tr><td>DisParQ (ours)</td><td>16/1024</td><td>89.4+9.3 85.5+17.1</td><td></td><td>54.5+17.2</td><td>9.36.0.51</td><td>91.0+4.0</td><td>83.4+19.8</td><td>87.8+5.6</td><td>85.8+16.7</td><td>99.7</td><td>99.7+0.2</td></tr></table>

Cross-category part retrieval. The concepts of DisParQ keep their identity across categories. Among self-supervised methods, DisParQ is best on 7 of the 8 metrics of Table 4 and second on part-to-part R@5. On PartImageNet, it leads part-to-part retrieval with an R@1 of 79.7% and an mAP of 30.5, vs. 79.2% and 27.0 for the best baselines, and image-to-image retrieval with an mAP of 76.1 vs. 61.8. An oracle based only on part-label co-occurrence groups reaches an R@1 of 43.9% (Appendix F.6). Against the supervised PDiscoFormer, DisParQ leads 4 of the 6 retrieval metrics at each of its three part counts, including part-to-part R@1 (79.7% vs. at most 71.8%) and imageto-image mAP (76.1 vs. at most 70.2), while PDiscoFormer reaches a higher part-to-part mAP. On CUB, DisParQ also leads R@1 (36.8% vs. 30.9%; see Table 19 in Appendix F.6).

Backbones and design choices. DisParQ is not tied to one pretraining objective. Across seven configurations of six frozen backbones, self-supervised (DINOv2 and DINOv3 (Simeoni et al. ´ , 2025)) and vision-language (CLIP (Radford et al., 2021), SigLIP2 (Tschannen et al., 2025), TIPSv2 (Cao et al., 2026) and CLIP-DINOiser), it stays within 2 p.p. of each backbone’s own linear probe on 6 of the 7 and within 3.5 p.p. on CLIP-DINOiser (see Figure 8 in Appendix E.1). At the same budget of 16/1024, DINOv2 gives the best concepts: it leads the other six configurations on 7 of the 8 concept metrics of Table 2, all but impurity, and on part-to-part retrieval. Appendix E presents our design choices for the bank size N, the per-image budget K and the losses in detail.

![](images/4e0225f765514283b4ac7040dfe72e9ccf259b75458c1e78fbefed47b87fcd4a.jpg)

Table 4: Retrieval on PartImageNet (val, as %). Marks as in Table 2. Dictionary methods appear at our N=1024 and their best N. <sup>‡</sup>At the method’s native resolution, on another query set (Appendix F.6). The controls and PCA are in Table 19. DisParQ is the best on all but one metric.
<table><tr><td rowspan="2">Method</td><td rowspan="2">K/N</td><td colspan="2">Accuracy</td><td colspan="3">Image → image</td><td colspan="3">Part → part</td></tr><tr><td>LP↑</td><td>kNN↑</td><td>R@1↑</td><td>R@5↑</td><td>mAP↑</td><td>R@1↑</td><td>R@5↑</td><td>mAP↑</td></tr><tr><td colspan="10">Supervised (reference)</td></tr><tr><td>PDiscoFormer‡</td><td>50/50</td><td>89.8</td><td>88.1</td><td>78.4</td><td>93.4</td><td>69.3</td><td>71.8</td><td>89.0</td><td>45.9</td></tr><tr><td>PIP-Netstage 2</td><td>16/768</td><td>87.5</td><td>86.8</td><td>77.6</td><td>91.9</td><td>64.2</td><td>65.4</td><td>85.2</td><td>21.1</td></tr><tr><td>ProtoQuantstage 2</td><td>16/2048</td><td>87.8</td><td>89.5</td><td>82.3</td><td>93.8</td><td>76.6</td><td>62.6</td><td>89.6</td><td>15.5</td></tr><tr><td colspan="10">Frozen backbone, no concept bottleneck (reference)</td></tr><tr><td>CLS token</td><td>N/A</td><td>89.3</td><td>88.9</td><td>81.0</td><td>92.9</td><td>75.1</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Patch tokens, mean-pooled N/A</td><td></td><td>85.7</td><td>79.3</td><td>57.5</td><td>82.2</td><td>44.7</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td colspan="10">Self-supervised</td></tr><tr><td>CFM‡</td><td>16/8192</td><td>72.6</td><td>69.2</td><td>47.0</td><td>75.4</td><td>34.6</td><td>31.6</td><td>68.8</td><td>16.0</td></tr><tr><td>TopK SAE</td><td>16/1024</td><td>78.1</td><td>73.0</td><td>52.6</td><td>79.2</td><td>42.7</td><td>56.2</td><td>86.5</td><td>17.2</td></tr><tr><td>TopK SAE</td><td>16/2048</td><td>80.5</td><td>73.5</td><td>53.8</td><td>77.5</td><td>43.3</td><td>58.3</td><td>86.9</td><td>18.1</td></tr><tr><td>PatchSAE</td><td>16/1024</td><td>82.9</td><td>76.5</td><td>56.8</td><td>80.8</td><td>44.6</td><td>41.9</td><td>83.6</td><td>11.8</td></tr><tr><td>PatchSAE</td><td>16/49152</td><td>84.1</td><td>77.2</td><td>56.8</td><td>81.1</td><td>44.4</td><td>48.6</td><td>82.9</td><td>8.0</td></tr><tr><td>MSAE</td><td>16/1024</td><td>85.2</td><td>79.9</td><td>58.9</td><td>81.7</td><td>46.1</td><td>41.4</td><td>82.2</td><td>12.5</td></tr><tr><td>MSAE</td><td>16/8192</td><td>85.6</td><td>79.5</td><td>58.3</td><td>82.3</td><td>45.4</td><td>41.2</td><td>79.9</td><td>12.6</td></tr><tr><td>PIP-Netstage 1</td><td>16/768</td><td>76.9</td><td>71.5</td><td>49.1</td><td>78.6</td><td>36.3</td><td>66.6</td><td>87.7</td><td>27.0</td></tr><tr><td>ProtoQuantstage 1</td><td>16/2048</td><td>85.2</td><td>83.9</td><td>72.5</td><td>90.4</td><td>61.8</td><td>79.2</td><td>92.8</td><td>22.7</td></tr><tr><td>DisParQ (ours)</td><td>16/1024</td><td>89.2+3.6</td><td>88.7+4.8</td><td>80.6+8.1</td><td>92.9+2.5 76.1+14.3</td><td></td><td>79.7+0.5</td><td></td><td>92.4-0.4 30.5+3.5</td></tr></table>

Figure 6: Part concepts on PartImageNet. (a) Part concept maps for DisParQ vs. baselines. DisParQ assigns each location exactly one concept, with no overlap, following object parts, while the baselines’ overlapping concepts merge parts or spread into the background. (b) Each method’s concept most consistent with the outlined sail, on four images. <sup>†</sup> trained with class labels.

## 4.3 QUALITATIVE RESULTS AND INTERPRETABILITY

Part maps. Figure 6 compares the concept maps of DisParQ with those of ProtoQuant, CFM and the supervised PDiscoFormer. DisParQ follows the annotated parts and stops at the object boundary: its non-overlapping concept regions match distinct parts of the object, such as the head, body, wings, feet and tail of the kite (Figure 6a, first row), where the baselines cover larger regions with one concept or spread into the background. For a part such as a sail, one concept of DisParQ marks it in all four images, while a baseline’s best-matching concept misses some of them or covers the whole object. Appendix G gives further examples on additional datasets, and failure cases.

Interpretability. DisParQ explains a prediction through its parts (Figure 1). Removing a concept from the discrete description and decoding the rest again measures how much the predicted class depends on that part, an occlusion test (Zeiler & Fergus, 2014) on parts instead of pixels. The four largest contributions account for 56-100% of the positive logit drops, and each retrieves the same part in other images, so a prediction reads as a short list of recognizable parts with their weights (details in Appendix G.5).

User study. We asked 120 participants to evaluate concepts of DisParQ, PIP-Net, and CFM on CUB, Cars, and PartImageNet. For each prototype, they rated how consistently it captures the same object part (1–5, where 5 is most consistent) and identified which of four regions in a new image it corresponds to. DisParQ prototypes were rated significantly more consistent than both baselines (Figure 5, right) and were matched to the correct part most often (69% vs. 60% for PIP-Net and 53% for CFM, random chance is 25%). Details are in Appendix F.8.

## 5 CONCLUSION

We presented DisParQ, which describes an image from a frozen self-supervised Vision Foundation Model (VFM) by a few discrete part concepts, one per location, each with a quantized attribute, learned without class labels or language. This description preserves the backbone’s accuracy, gives part concepts that are more consistent across images than those of language-grounded and sparseautoencoder dictionaries, and keeps a concept’s identity across categories, so a prediction reads as a short list of parts and their contributions. In future work, the discrete concepts could support searching by part and appearance, and correcting a prediction by removing or replacing concepts.

## ETHICS STATEMENT

We use existing public datasets and pretrained models for model training and benchmark evaluation, without using private datasets. To assess interpretation quality, we also conduct a user study on Clickworker that was compliant with our organization procedures and regulations. Participants were informed about the study and compensated for their participation, and we collected no identifying information.

DisParQ makes the representations of vision foundation models easier to inspect by expressing them through discrete part concepts. Although these concepts can reveal patterns used by a model, they do not by themselves establish that its predictions are fair, reliable, or causally grounded. The method inherits the biases and limitations of its frozen backbone and training data, so even visually coherent concepts may encode spurious correlations. Use in consequential applications would therefore require validation in the intended setting, including an assessment of errors and bias.

## REPRODUCIBILITY STATEMENT

To support reproduction of our experiments, we describe the model and training objectives in Section 3 and provide a highly detailed, technical description in Appendix C. Appendix D details the datasets and splits, pretrained backbones, evaluation metrics, and baseline protocols. Throughout these descriptions, we distinguish label-free concept learning from the use of labels for downstream probes and evaluation.

## ACKNOWLEDGMENTS

Adam Pardyl’s research was supported by a grant from the Faculty of Mathematics and Computer Science under the Strategic Programme Excellence Initiative at Jagiellonian University. Sid dhartha Gairola’s research was supported in part by ELSA – European Lighthouse on Secure and Safe AI, funded by the European Union under grant agreement No. 101070617. Sukrut Rao and Bernt Schiele were funded in part by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – GRK 2853/1 “Neuroexplicit Models of Language, Vision, and Action” – project number 471607914. The work of Bartosz Zielinski and Dawid Rymarczyk was funded´ by the “Interpretable and Interactive Multimodal Retrieval in Drug Discovery” project. This project (FENG.02.02-IP.05-0040/23) is carried out within the First Team programme of the Foundation for Polish Science co-financed by the European Union under the European Funds for Smart Economy 2021–2027 (FENG).

Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the European Commission can be held responsible for them.

We would like to thank and acknowledge the Max Planck Computing and Data Facility (MPCDF), whose HPC systems supported a substantial portion of our experiments, including the development of the proposed method, reproduction of baselines, and computation of the final results. Some experiments were performed on servers purchased with funds from the flagship project entitled “Artificial Intelligence Computing Center Core Facility” from the DigiWorld Priority Research Area within the Excellence Initiative – Research University program at Jagiellonian University in Krakow. ´ We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center:

ACK Cyfronet AGH) for providing computer facilities and support within computational grant no.   
PLG/2026/019504 and PLG/2025/018688.

## REFERENCES

Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. On the effectiveness of ViT features as local semantic descriptors. In ECCVW, 2023.

Ananthu Aniraj, Cassio F. Dantas, Dino Ienco, and Diego Marcos. PDiscoFormer: Relaxing part discovery constraints with vision transformers. In ECCV, 2024.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients ´ through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Itay Benou and Tammy Riklin-Raviv. Show and tell: Visually explainable deep neural nets via spatially-aware concept bottleneck models. In CVPR, 2025.

Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, Joshua Ainslie, Alex Bewley, Mithun Jacob, Rene Wagner, Washington Ramos, Krzysztof Choromanski, Mojtaba Seyedhosseini, Howard´ Zhou, and Andre Araujo. TIPSv2: Advancing vision-language pretraining with enhanced patch-´ text alignment. In CVPR, 2026.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K. Su. This looks like that: Deep learning for interpretable image recognition. In NeurIPS, 2019.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, 2023.

Subhabrata Choudhury, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Unsupervised part discovery from contrastive reconstruction. In NeurIPS, 2021.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In ICLR, 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

Jon Donnelly, Alina Jade Barnett, and Chaofan Chen. Deformable ProtoPNet: An interpretable image classifier using deformable prototypes. In CVPR, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Viktar Dubovik, Łukasz Struski, Jacek Tabor, and Dawid Rymarczyk. SIDE: Sparse information disentanglement for explainable artificial intelligence. arXiv preprint arXiv:2507.19321, 2025.

Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes challenge 2012 (VOC2012) results, 2012.

Thomas Fel, Ekdeep Singh Lubana, Jacob S. Prince, Matthew Kowal, Victor Boutin, Isabel Pa padimitriou, Binxu Wang, Martin Wattenberg, Demba E. Ba, and Talia Konkle. Archetypal SAE: Adaptive and stable dictionary learning for concept extraction in large vision models. In ICML, 2025.

Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever,´ Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In ICLR, 2025.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.

Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, Qihang Yu, and Alan Yuille. PartImageNet: A large, high-quality dataset of parts. In ECCV, 2022.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415, 2016.

Zixuan Huang and Yin Li. Interpretable and accurate fine-grained recognition via region grouping. In CVPR, 2020.

Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In ICLR, 2024.

Wei-Chih Hung, Varun Jampani, Sifei Liu, Pavlo Molchanov, Ming-Hsuan Yang, and Jan Kautz. SCOPS: Self-supervised co-part segmentation. In CVPR, 2019.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-Softmax. In ICLR, 2017.

Mikołaj Janusz, Adam Wrobel, Bartosz Zieli´ nski, and Dawid Rymarczyk. ProtoQuant: Quantization´ of prototypical parts for general and fine-grained image classification. In BMVC, 2026.

Herve J´ egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor´ search. IEEE TPAMI, 33(1), 2011.

Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li. Novel dataset for fine-grained image categorization: Stanford Dogs. In CVPRW, 2011.

Sunnie S. Y. Kim, Nicole Meister, Vikram V. Ramaswamy, Ruth Fong, and Olga Russakovsky. HIVE: Evaluating the human interpretability of visual explanations. In ECCV, 2022.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In ICML, 2020.

Philipp Krahenb¨ uhl and Vladlen Koltun. Efficient inference in fully connected CRFs with gaussian¨ edge potentials. In NeurIPS, 2011.

Jonathan Krause, Michael Stark, Jia Deng, and Fei-Fei Li. 3D object representations for fine-grained categorization. In ICCVW, 2013.

Hyesu Lim, Jinho Choi, Jaegul Choo, and Steffen Schneider. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. In ICLR, 2025.

Yoseph Linde, Andres Buzo, and Robert M. Gray. An algorithm for vector quantizer design.´ IEEE Transactions on Communications, 28(1), 1980.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In NeurIPS, 2020.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Chiyu Ma, Jon Donnelly, Wenjun Liu, Soroush Vosoughi, Cynthia Rudin, and Chaofan Chen. Interpretable image classification with adaptive prototype-based vision transformers. In NeurIPS, 2024.

Sachit Menon and Carl Vondrick. Visual classification via description from large language models. In ICLR, 2023.

Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. SPair-71k: A large-scale benchmark for semantic correspondence. arXiv preprint arXiv:1908.10543, 2019.

Meike Nauta, Ron van Bree, and Christin Seifert. Neural prototype trees for interpretable finegrained image recognition. In CVPR, 2021.

Meike Nauta, Jorg Schl¨ otterer, Maurice van Keulen, and Christin Seifert. PIP-Net: Patch-based¨ intuitive prototypes for interpretable image classification. In CVPR, 2023.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In ICVGIP, 2008.

Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. In ICLR, 2023.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien Mairal, Patrick Labatut,´ Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024.

Mateusz Pach, Koryna Lewandowska, Jacek Tabor, Bartosz Zielinski, and Dawid Rymarczyk. Lu- ´ cidPPN: Unambiguous prototypical parts network for user-centric interpretable computer vision. In ICLR, 2025.

Konstantinos P. Panousis, Dino Ienco, and Diego Marcos. Coarse-to-fine concept bottleneck models. In NeurIPS, 2024.

Konstantinos Panagiotis Panousis, Dino Ienco, and Diego Marcos. Sparse linear concept discovery models. In ICCVW, 2023.

Katharina Prasse, Patrick Knab, Sascha Marton, Christian Bartelt, and Margret Keuper. DCBM: Data-efficient visual concept bottleneck models. In ICML, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Sukrut Rao, Sweta Mahajan, Moritz Bohle, and Bernt Schiele. Discover-then-name: Task-agnostic¨ concept bottlenecks via automated concept discovery. In ECCV, 2024.

Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5), 2019.

Cynthia Rudin, Chaofan Chen, Zhi Chen, Haiyang Huang, Lesia Semenova, and Chudi Zhong. Interpretable machine learning: Fundamental principles and 10 grand challenges. Statistics Surveys, 16, 2022.

Dawid Rymarczyk, Łukasz Struski, Jacek Tabor, and Bartosz Zielinski. ProtoPShare: Prototypical´ parts sharing for similarity discovery in interpretable image classification. In KDD, 2021.

Dawid Rymarczyk, Łukasz Struski, Michał Gorszczak, Koryna Lewandowska, Jacek Tabor, and´ Bartosz Zielinski. Interpretable image classification with differentiable prototypes assignment. In ´ ECCV, 2022.

Mikołaj Sacha, Bartosz Jura, Dawid Rymarczyk, Łukasz Struski, Jacek Tabor, and Bartosz Zielinski.´ Interpretability benchmark for evaluating spatial misalignment of prototypical parts explanations. In AAAI, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3.´ arXiv preprint arXiv:2508.10104, 2025.

Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2), 1967.

Łukasz Struski, Dawid Rymarczyk, and Jacek Tabor. InfoDisent: Explainability of image classification models by information disentanglement. arXiv preprint arXiv:2409.10329, 2024.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Henaff,´ Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Hugues Turbe, Mina Bjelogrlic, Gianmarco Mengaldo, and Christian Lovis. ProtoS-ViT: Visual´ foundation models for sparse self-explainable classifications. In NeurIPSW, 2024.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

Robert van der Klis, Stephan Alaniz, Massimiliano Mancini, Cassio F. Dantas, Dino Ienco, Zeynep Akata, and Diego Marcos. PDiscoNet: Semantically consistent part discovery for fine-grained recognition. In ICCV, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

Jiaqi Wang, Huafeng Liu, Xinyue Wang, and Liping Jing. Interpretable image recognition by constructing transparent embedding space. In ICCV, 2021.

Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric Lenssen. AnyUp: Universal feature upsampling. In ICLR, 2026.

Kai Wittenmayer, Sukrut Rao, Amin Parchami-Araghi, Bernt Schiele, and Jonas Fischer. CFM: Language-aligned concept foundation model for vision. In ECCV, 2026.

Monika Wysoczanska, Oriane Sim´ eoni, Micha´ el Ramamonjisoa, Andrei Bursuc, Tomasz Trzci¨ nski,´ and Patrick Perez. CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary´ semantic segmentation. In ECCV, 2024.

Mengqi Xue, Qihan Huang, Haofei Zhang, Jingwen Hu, Jie Song, Mingli Song, and Canghong Jin. ProtoPFormer: Concentrating on prototypical parts in vision transformers for interpretable image recognition. In IJCAI, 2024.

Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In CVPR, 2023.

Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck models. In ICLR, 2023.

Vladimir Zaigrajew, Hubert Baniecki, and Przemysław Biecek. Interpreting CLIP with hierarchical sparse autoencoders. In ICML, 2025.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Sound-Stream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30, 2022.

Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In ECCV, 2014.

Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 40(6), 2018.

## APPENDIX

This appendix provides additional details and results for DisParQ. We first expand the discussion of related work (Appendix A) and discuss limitations (Appendix B). The architecture and training procedure are described in detail next (Appendix C). Then we present the experimental protocol, including the datasets, backbones, training procedures, and evaluation metrics used for DisParQ and the baselines (Appendix D). We examine each component through ablations (Appendix E) and provide extended results, including the full main-paper tables, a qualitative analysis of quantized attributes, and user-study details (Appendix F). Lastly, we show qualitative results across all five datasets and illustrate how DisParQ explains predictions (Appendix G), before describing how learned concepts can be named using a fixed vocabulary (Appendix H).

## Table of contents

B Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18   
C Architecture and training details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   
C.1 Feature extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   
C.2 Concept identification and assignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
C.3 Attribute encoding and quantization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21   
C.4 Concept decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
C.5 Training objectives and regime . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
C.6 Training stages and optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25   
D Experimental protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26   
D.1 Datasets and splits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26   
D.2 Backbones . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26   
D.3 Training DisParQ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27   
D.4 Training the baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28   
D.5 Evaluation protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30   
D.6 Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31   
E Ablation studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34   
E.1 Feature extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34   
E.2 Concept bank, selection, and maintenance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36   
E.3 Appearance attributes and quantization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38   
E.4 Concept decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40   
E.5 Gradient routes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41   
E.6 Training objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42   
F Additional results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44   
F.1 Large-scale classification in full . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   
F.2 Full PartImageNet comparison . 45   
F.3 Full CUB concept metrics . . . 48   
F.4 PDiscoFormer on Cars and Dogs . . . 48   
F.5 ProtoQuant and PIP-Net: frozen vs. fine-tuned backbone . . 48   
F.6 Retrieval results . . . 49   
F.7 What the quantized attributes keep . . . . 52   
F.8 User study details and results . . 52   
G Qualitative results . . . 54   
G.1 Interactive Concept Visualizer. . . 54   
G.2 PartImageNet. . . 57   
G.3 Fine-grained datasets . . 61   
G.4 Part-to-part retrieval . . . 63   
G.5 Class evidence of DisParQ and interpretability . . . 65   
G.6 What quantization keeps . . . . . 71   
G.7 How one concept varies across images . . 77   
H Automatic alignment of concept names . . . 82   
H.1 Concept descriptions and alignment targets . . . 82   
H.2 Concept descriptions to names . . 82   
H.3 Evaluating the retrieved names . . 83   
H.4 Results . . . 84   
H.5 Qualitative examples . . . 84

## A RELATED WORK

Prototypical parts and part discovery. ProtoPNet (Chen et al., 2019) introduced “this looks like that” reasoning: an image is classified by how closely its patches match prototypes learned for each class. Later work shares prototypes across classes to build compact vocabularies (Nauta et al., 2021; Rymarczyk et al., 2021; 2022; Nauta et al., 2023). Others refine the geometry of the prototype space (Wang et al., 2021; Donnelly et al., 2022) or separate color from other visual features (Pach et al., 2025). Recent variants build on Vision Transformers (Xue et al., 2024; Ma et al., 2024), including frozen foundation backbones (Struski et al., 2024; Turbe et al.´ , 2024; Dubovik et al., 2025; Janusz et al., 2026). Among these, ProtoQuant (Janusz et al., 2026) also quantizes its prototypes into a codebook, but still classifies from continuous similarity scores. Part discovery methods segment objects into parts, either without any labels (Hung et al., 2019; Choudhury et al., 2021; Amir et al., 2023) or with class labels as the only signal (Huang & Li, 2020; van der Klis et al., 2023; Aniraj et al., 2024). Prototypes are learned to serve a classifier, and discovered parts form a small, fixed set per category or dataset. DisParQ instead learns, without class labels, a large shared dictionary from which each image selects a few concepts, assigns each patch exactly one of them, and captures how each of them varies across images with an attribute, making a frozen vision foundation model interpretable at the level of parts (Table 1).

Concept bottlenecks and concept dictionaries. Concept bottleneck models (CBMs) make predictions through a layer of human-defined concepts (Koh et al., 2020). Recent CBMs let a language model write the concept list and score it with a vision-language model (Yuksekgonul et al., 2023; Oikarinen et al., 2023; Yang et al., 2023), and spatial variants score concepts on image regions (Panousis et al., 2024) or as concept maps (Benou & Riklin-Raviv, 2025). Sparse autoencoders (SAEs) instead learn an unnamed, overcomplete dictionary from a model’s activations. They have recently been used to interpret language models (Huben et al., 2024; Gao et al., 2025) and have since been applied to vision: PatchSAE (Lim et al., 2025) trains them on the patch tokens of a vision transformer, and later variants improve the dictionary itself (Zaigrajew et al., 2025; Fel et al., 2025). DN-CBM (Rao et al., 2024) names SAE concepts with CLIP, and the concept foundation model CFM (Wittenmayer et al., 2026) builds a spatially grounded concept dictionary on a language-aligned backbone and names its concepts from a predefined text vocabulary. Most CBMs name their concepts in advance, and SAE codes are continuous weights over concepts that overlap at each patch, which entangles a part’s identity with its appearance. DisParQ learns its concepts from visual features alone, assigns each patch exactly one concept, separates a part’s identity from how it varies across images (its attribute), and describes an image by at most 16 active concepts in integer indices alone, the budget at which we compare all dictionaries (Section 4.1).

Evaluating part concepts. Part concepts are usually scored by comparing each concept’s map with part annotations: through the purity of prototypes on annotated keypoints (Nauta et al., 2023), key point regression and clustering agreement with annotated parts (van der Klis et al., 2023; Aniraj et al., 2024), or the locality and consistency of concept maps (Wittenmayer et al., 2026). A separate benchmark tests whether a prototype’s activation map marks the image region that actually drives it (Sacha et al., 2024), and semantic correspondence matches keypoints between two given images, usually of the same category (Min et al., 2019; Amir et al., 2023). Our part-to-part retrieval bench mark instead starts from one annotated part region and asks whether its description finds the same part among the regions of all other images, across categories (Section 4.1 and Appendix D.6).

## B LIMITATIONS

In this section we discuss the limitations in our proposed method and study. Part annotations may be coarser than the learned concepts, but over-segmentation and incorrect semantic grouping can also cause failures (Appendix G.2). The frozen backbone passes on its features and biases, including part of the class structure of the attributes (Appendix G.7). Contextual backbone features and soft pooling can carry information from outside a displayed hard region, so spatial assignments do not establish pixel-level causal locality. Concepts remain unnamed without examples or alignment (Appendix H). They cover the whole image, including background, with the same budget of K=16 concepts regardless of its number of parts (Appendix E.2).

## C ARCHITECTURE AND TRAINING DETAILS

This section specifies details of DisParQ as introduced in Section 3, following the same narrative structure. It describes the canonical form used in the main experiments. Differences for studies with alternative configurations are described in the relevant sections.

The default DisParQ configuration uses frozen DINOv2-B/14 (Oquab et al., 2024) with registers (Darcet et al., 2024), a bank of $N = 1 0 2 4$ concepts, $K = 1 6$ selected concepts per image, and private $1 6 \times 6 4$ product quantization. Every patch is assigned to one concept. Table 5 recaps the principal symbols and dimensions used throughout this section. Spatial positions are flattened in row-major order. Batch statistics refer to the local device batch unless stated otherwise.

Table 5: Notation and dimensions for the canonical model. Feature width D and decoder width d are distinct quantities, both equal to 768 here. K counts concepts and G counts attribute groups.
<table><tr><td>Symbol</td><td>Meaning</td><td>Value / shape</td></tr><tr><td> $D , d$ </td><td>Feature width, decoder width</td><td>768,768</td></tr><tr><td> $B , b , { u }$ </td><td>Local batch size, image index, spatial index</td><td> $1 6 ; 1 \le b \le B ; 1 \le u \le H W$ </td></tr><tr><td> $z _ { u } , c ( x )$ </td><td>Spatial features, global target</td><td> $\mathbb { R } ^ { D }$ </td></tr><tr><td> $w _ { n } , e _ { n }$ </td><td>Concept prototype, decoder value</td><td> $\mathbb { R } ^ { D } , \mathbb { R } ^ { d }$ </td></tr><tr><td> $N , K$ </td><td>Bank size, selected concepts</td><td> $1 0 2 4 , 1 6$ </td></tr><tr><td> $i _ { k } , s _ { u } , p _ { u }$ </td><td>Global ID, local owner, spatial global ID</td><td> $p _ { u } = i _ { s _ { u } }$ </td></tr><tr><td> $\chi _ { b , n } , \psi _ { n }$ </td><td>Selection score, full-bank usage distribution</td><td>Scalars</td></tr><tr><td> $A , { \bar { A } }$ </td><td>Pre- and post-top-K soft maps</td><td> $N \times H W , K \times H W$ </td></tr><tr><td> $H \times W$ </td><td>Shared assignment and decoder grid</td><td> $3 2 \times 3 2$ </td></tr><tr><td> $\bar { z } _ { i _ { k } } , r _ { i _ { k } } , q _ { i _ { k } }$ </td><td>Pooled feature, residual, attribute</td><td> $\mathbb { R } ^ { D } , \mathbb { R } ^ { D } , \mathbb { R } ^ { d }$ </td></tr><tr><td> $B , j _ { i _ { k } , g }$ </td><td>Private codebooks, codeword index</td><td> $N \times G \times M \times ( d / G ) ; 0 \leq j < M$ </td></tr><tr><td> $G , { \ddot { M } }$ </td><td>PQ groups, entries per group</td><td> $1 6 , 6 4$ </td></tr><tr><td> $\rho , \eta$ </td><td>Usage EMA, selection dual</td><td> $\mathbb { R } ^ { N } \operatorname { e a c h }$ </td></tr><tr><td> $\bar { Z } , V _ { m }$ </td><td>Area-scaled and modulated region features</td><td> $D \times K$ </td></tr><tr><td> $W _ { \mathrm { c o n t } } , W _ { \mathrm { P Q } }$ </td><td>Continuous and quantization projections</td><td> $d \times D \mathrm { e a c h }$ </td></tr><tr><td> $\mathrm { P E } , t _ { u }$ </td><td>Position table, decoder token</td><td> $H W \times d , \mathbb { R } ^ { d }$ </td></tr></table>

## C.1 FEATURE EXTRACTION

Input images have size $2 2 4 \times 2 2 4$ and 3 RGB channels. The frozen DINOv2-B/14 encoder (Oquab et al., 2024) produces width- $D \ = \ 7 6 8$ patch features on a native $1 6 \times 1 6$ grid $( 1 4 \times 1 4$ patch size). Its final layer-normalized CLS token is the global distillation target $\boldsymbol { c } ( \boldsymbol { x } ) \in \mathbb { R } ^ { D }$ . CLS and register tokens are removed from the spatial sequence. The backbone and the pretrained upsampler remain in evaluation mode throughout training. Backbone substitutions and their target definitions are described in Appendix E.1.

To combine information across depths while retaining spatial correspondence, the lightweight attention pooling operator $f _ { \mathrm { f u s e } }$ consumes raw patch outputs $h _ { \lambda , \xi } \in \bar { \mathbb { R } } ^ { D }$ from the last four blocks of the encoder. We restrict fusion to these later blocks to favor semantic features and reduce the opportunity for concepts to organize around positional cues from earlier layers. A shared bias-free key projection $W _ { \mathrm { k e y } } \stackrel { \bullet } { \in } \mathbb { R } ^ { 6 4 \times \breve { D } }$ and a learned query $\omega ^ { \mathrm { f u s e } } \in \mathbb { R } ^ { 6 4 }$ form four 16-dimensional heads using scaled dot-product attention (Vaswani et al., 2017):

$$
\alpha _ { a , \lambda , \xi } = \mathrm { s o f t m a x } _ { \lambda } \left( \frac { ( \omega _ { a } ^ { \mathrm { f u s e } } ) ^ { \top } ( W _ { \mathrm { k e y } } h _ { \lambda , \xi } ) _ { a } } { \sqrt { 1 6 } } \right) , \qquad z _ { \xi } ^ { \mathrm { f u s e } } = \frac { 1 } { 4 } \sum _ { a = 1 } ^ { 4 } \sum _ { \lambda = 1 } ^ { 4 } \alpha _ { a , \lambda , \xi } h _ { \lambda , \xi } .\tag{6}
$$

Here ξ indexes the native patch grid, $\lambda \in \{ 1 , \ldots , 4 \}$ indexes the selected encoder blocks, and $a \in \{ 1 , \ldots , 4 \}$ indexes attention heads. Each head mixes full D-dimensional values. There is no value/output projection, extra normalization, residual connection, or positional encoding in fusion. It has $6 4 D + 6 4$ parameters.

The query is initialized from a truncated normal with standard deviation 0.02. Throughout this appendix, truncated-normal initialization uses mean zero and absolute truncation bounds [−2, 2], as in the default PyTorch initializer. The key projection entries are uniform on $[ - D ^ { - 1 / 2 } , D ^ { - 1 / 2 } ]$ Removing fusion leaves only the final-block patch features, as studied in Appendix E.1.

Image-guided upsampling. To represent finer concept boundaries than the native $1 6 \times 1 6$ grid permits, we upsample the fused features with image guidance. The multi-backbone AnyUp model (Wimmer et al., 2026), using its second released pretrained weight set, receives the ImageNetnormalized image and the fused feature map. It mixes unprojected feature values and preserves width D, producing $z _ { u } \in \mathbb { R } ^ { D }$ on a $3 2 \times 3 2$ grid. Its computation uses float32 before restoring the feature dtype. AnyUp’s parameters are frozen, but stage 1 propagates gradients through its feature input to fusion. Stage 2 evaluates the frozen feature path without gradients, because fusion is also frozen in that stage.

A less computationally expensive alternative is bilinear interpolation of the fused features to the same grid, with pixel-center coordinates (zero-based output coordinate $o \in \{ 0 , \ldots , 3 1 \}$ maps to source coordinate $( o + \frac { 1 } { 2 } ) / 2 - \frac { 1 } { 2 }$ along each axis, using boundary replication). Appendix E.1 compares this alternative, native-resolution features, and mixed AnyUp/bilinear training. Bilinear and AnyUp scores are broadly comparable, with metric-dependent differences, while AnyUp yields slightly more detailed qualitative concept maps.

## C.2 CONCEPT IDENTIFICATION AND ASSIGNMENT

Given the upsampled features, we first score the full concept bank and then restrict spatial competition to the selected concepts. The squared-distance competition in Equation 1 follows the prototype assignment used in PDiscoNet and PDiscoFormer (van der Klis et al., 2023; Aniraj et al., 2024), here extended to a large bank with per-image selection. It uses unnormalized prototypes and features. Each prototype $\bar { w _ { n } } \in \mathbb { R } ^ { D } , n \in \mathbf { \bar { \{ 1 , \dots , \bar { N } \} } }$ , is initialized from a truncated normal with standard deviation 0.02. Training adds independent standard Gumbel noise to each logit and uses temperature 1, with soft outputs (Jang et al., 2017). The mean activation score is $\begin{array} { r } { \chi _ { b , n } = ( H W ) ^ { - 1 } \sum _ { u } \bar { A _ { b , n , u } } . \operatorname { A } } \end{array}$ standard Gumbel sample is − log(− log U) for U uniform on $( 0 , 1 )$ , sampled independently for each image, concept, and location. Let B be the local batch size and $\dot { b } \in \{ \bar { 1 } , \ldots , B \}$ index its images. From the $B \times N$ score matrix, form

$$
\widetilde { \chi } _ { b , n } = \chi _ { b , n } - 0 . 5 \log ( \rho _ { n } + 1 0 ^ { - 8 } ) , \qquad Q ^ { ( 0 ) } = { \frac { \exp ( \widetilde { \chi } / 0 . 1 ) } { \sum _ { b , n } \exp ( \widetilde { \chi } _ { b , n } / 0 . 1 ) } } ,\tag{7}
$$

where $\rho _ { n }$ is the exponential moving average (EMA) of selection frequency, updated as specified in Appendix C.6. We use Sinkhorn–Knopp normalization (Sinkhorn & Knopp, 1967), following the use of batch-balanced prototype assignments in SwAV (Caron et al., 2020). Here balancing selects a subset of concepts rather than providing a swapped-prediction target. Ten iterations alternately divide each column by N times its sum and each row by its sum, clamping denominators to $1 0 ^ { - 8 }$ Normalization is performed in float32, under stop-gradient. The largest K entries in each final row select distinct IDs, retained in ranking order. Stable sorting favors lower bank IDs at exact ties. A per-concept score bias that is constant across images cancels under ideal Sinkhorn normalization, so the coverage term is not an independent balancing force in exact arithmetic. Top-K truncation also does not guarantee exactly balanced hard selections.

The bank-size and per-image-budget studies in Appendix E.2 vary N and K separately. Appendix E.2 compares peak scoring, direct mass-based selection, and removal of the coverage bias.

Balancing is per local batch, without gathering score matrices across devices. A training batch of one degenerates to ties. The post-selection maps A<sup>¯</sup> are recomputed from clean logits with a second, independent Gumbel sample. Slicing the first sampled maps would change the stochastic computation. Each location has local owner $s _ { u } = \arg \operatorname* { m a x } _ { k } \bar { A } _ { i _ { k } , u }$ and global concept identity $p _ { u } = i _ { s _ { u } }$ , with $k \in \{ 1 , \ldots , K \}$ . The K selected concepts need not all own a location: a selected concept can have positive soft mass but an empty hard region. Its attribute then contributes no spatial decoder token. These discrete choices have no derivative. An identity surrogate in Appendix C.5 supplies a gradient to the soft maps.

Batch-independent evaluation. During training, a second balancing calculation on the original view’s raw scores estimates a centered dual. If Q is its balanced matrix, set

$$
\delta _ { b , n } = \log \operatorname* { m a x } ( Q _ { b , n } , 1 0 ^ { - 3 0 } ) - \chi _ { b , n } / 0 . 1 , \qquad \eta _ { n } ^ { \mathrm { n e w } } = \mathrm { c e n t e r } _ { n } \left[ \frac { 1 } { B } \sum _ { b } \mathrm { c e n t e r } _ { n } ( \delta _ { b , n } ) \right] .\tag{8}
$$

Here center subtracts the mean over the N concept indices. In distributed training, we average this estimate across devices. We then update η by exponential moving average with decay 0.99 and recenter it, initializing it with the first estimate. Evaluation uses ordinary softmax and selects the largest K values of $\chi _ { n } ( x ) / 0 . 1 + \eta _ { n }$ . It omits the coverage term. The frozen PQ stage retains the dual rather than updating it.

## C.3 ATTRIBUTE ENCODING AND QUANTIZATION

Once the concept regions are defined, appearance encoding summarizes how each occurrence differs from its prototype. We first compute area-scaled pooled features, as in PDiscoNet (van der Klis et al., 2023), together with their soft region masses:

$$
\widetilde { z } _ { i _ { k } } = ( H W ) ^ { - 1 } \sum _ { u } \bar { A } _ { i _ { k } , u } z _ { u } , \qquad \mu _ { i _ { k } } = ( H W ) ^ { - 1 } \sum _ { u } \bar { A } _ { i _ { k } , u } .\tag{9}
$$

The weighted region mean in Equation 3 is stabilized as $\bar { z } _ { i _ { k } } = \widetilde { z } _ { i _ { k } } / ( \mu _ { i _ { k } } + 1 0 ^ { - 6 } )$ . The normalization $\nu ( y ) = \breve { y } / \operatorname* { m a x } ( \| y \| _ { 2 } , 1 0 ^ { - 1 2 } )$ is evaluated in float32 for each vector y. The residual is then cast back to the feature dtype without further normalization. Both stages pass stopgrad $. ( r _ { i _ { k } } )$ to the appearance encoder, preventing the appearance route from changing the concept reference or region features to make its own prediction easier. Assignment learning instead receives the identity-surrogate, auxiliary, and regularization signals described below. In stage 1, $q _ { i _ { k } } = W _ { \mathrm { c o n t } }$ stopgrad $. ( r _ { i _ { k } } )$ , where the shared bias-free projection $W _ { \mathrm { c o n t } } \in \mathbb { R } ^ { d \times D }$ is initialized from a truncated normal with standard deviation 0.02. Appendix E.5 tests restoring residual gradients, while Appendix E.3 removes the appearance channel entirely.

Product quantization. To make this appearance channel discrete, we use product quantization (PQ) (Jegou et al.´ , 2011). We project each residual to the decoder width with $\begin{array} { r l } { y _ { i _ { k } } } & { { } = } \end{array}$ $W _ { \mathrm { P Q } }$ stopgrad $( r _ { i _ { k } } ) \in \mathbb { R } ^ { d }$ and split it into G channel groups indexed by $g \in \{ 1 , \ldots , G \}$ . Each selected concept uses the codebooks of its bank identity $i _ { k }$

Each group is represented by one of M codewords:

$$
j _ { i _ { k } , g } = \arg \operatorname* { m i n } _ { j \in \{ 0 , \dots , M - 1 \} } \| y _ { i _ { k } } ^ { ( g ) } - \mathcal { B } _ { i _ { k } , g , j } \| _ { 2 } ^ { 2 } , \qquad q _ { i _ { k } } = \bigoplus _ { g = 1 } ^ { G } \mathcal { B } _ { i _ { k } , g , j _ { i _ { k } , g } } ,\tag{10}
$$

Here $B _ { i _ { k } , g , j }$ is the jth codeword in group g of concept $i _ { k }$ , and $\oplus$ concatenates the selected codewords into the quantized attribute vector $q _ { i _ { k } }$

The codebooks $B _ { i _ { k } , g }$ are concept-specific, so appearance codes can specialize to each concept. Product quantization represents variation through combinations of codewords rather than a separate entry for every complete attribute vector. The indices $j _ { i _ { k } , 1 : G }$ form the discrete appearance code. Figure 4 illustrates the attribute path and its connection to the decoder.

PQ surrogate. Hard codeword selection requires a surrogate gradient during training. Following the straight-through principle (Bengio et al., 2013), we retain hard forward values while differentiating a soft codeword mixture. We use G = 16 channel groups and M = 64 entries in each concept-specific codebook, with group width $d / G$ . For each group in Equation 10, training uses

$$
\begin{array} { r l r } {  { \pi _ { i _ { k } , g , j } = \mathrm { s o f t m a x } _ { j } ( - \| y _ { i _ { k } } ^ { ( g ) } - \mathcal { B } _ { i _ { k } , g , j } \| _ { 2 } ^ { 2 } ) , } } \\ & { } & { \bar { q } _ { i _ { k } } ^ { ( g ) } = \displaystyle \sum _ { j } \pi _ { i _ { k } , g , j } \mathcal { B } _ { i _ { k } , g , j } , \qquad q _ { i _ { k } } ^ { ( g ) } = \mathcal { B } _ { i _ { k } , g , j _ { i _ { k } , g } } + \bar { q } _ { i _ { k } } ^ { ( g ) } - \mathrm { s t o p g r a d } ( \bar { q } _ { i _ { k } } ^ { ( g ) } ) . } \end{array}\tag{11}
$$

The softmax temperature is fixed at 1. Here $\pi _ { i _ { k } , g , j }$ is the soft probability of codeword j for concept $i _ { k }$ and group g, and $\bar { q } _ { i _ { k } } ^ { ( g ) }$ is their weighted mixture. The backward pass includes the direct gradient to the chosen codeword and the gradient through the soft mixture, including its dependence on the codebooks. Thus, unlike the identity-gradient estimator in VQ-VAE (van den Oord et al., 2017), our surrogate differentiates the soft assignment probabilities as well as the codeword values. Evaluation performs hard lookup only. The shared bias-free projection and private tables are learned by global cosine distillation. There is no commitment loss, residual-reconstruction loss, codebook EMA, k-means initialization, or attribute-entropy penalty. Both projection and codebooks start from a truncated normal with standard deviation 0.02. Appendix E.3 compares continuous attributes, private and shared codebooks, and vector, residual, and product quantization at different bit rates.

## C.4 CONCEPT DECODER

The decoder predicts the global VFM representation (its CLS token) from concept identities, spatial assignments, and attributes, as in Equation 4. Spatial self-attention (Vaswani et al., 2017) lets the readout combine concepts according to their arrangement, while learned positions and a decoder CLS token follow the ViT readout pattern (Dosovitskiy et al., 2021). Its $\bar { N }$ learnable concept embeddings $e _ { n } \in \mathbb { R } ^ { d }$ are independent of the matching prototypes $w _ { n } ,$ allowing the representation used for decoding to differ from the reference used for feature matching. Assignment and decoding share the $H \times W$ grid, so identities and attributes are broadcast directly using $p _ { u } ,$ without any spatial pooling. Concept values, the $H W \times d$ positional table, and the learned decoder CLS token are initialized from a truncated normal with standard deviation 0.02.

The decoder has three pre-normalization blocks, four attention heads, and feed-forward width 4d with GELU (Hendrycks & Gimpel, 2016). Attention projections are bias-free, feed-forward layers have biases. LayerNorm (Ba et al., 2016) has learned affine parameters and $\epsilon = 1 0 ^ { - 5 }$ . There is no attention/transformer dropout, mask, or stochastic depth. A final LayerNorm(d) → Linear(d, D) maps the output CLS token to the predicted global representation ${ \hat { c } } ( x )$ The CLS input has no separate positional embedding. The output is not explicitly unit-normalized. Each block applies prenormalized self-attention with a residual connection, then a pre-normalized two-layer feed-forward network with a second residual connection. LayerNorm scales and offsets start at one and zero. Unless initialized explicitly above, linear weights and biases are uniform on $[ - d _ { \mathrm { i n } } ^ { - 1 / 2 } , d _ { \mathrm { i n } } ^ { - 1 / 2 } ]$ , where $d _ { \mathrm { i n } }$ is that layer’s input width. The concatenated attention query/key/value projection instead uses Xavier-uniform initialization (Glorot & Bengio, 2010).

Given fixed model parameters, decoding requires only the selected bank IDs $i _ { 1 : K }$ , the $H \times W$ local-owner map s, and the $K \times G$ PQ indices. Its nominal fixed-width payload is

$$
K \log _ { 2 } N + H W \log _ { 2 } K + K G \log _ { 2 } M ,\tag{12}
$$

or 2,720 bits at $H = W = 1 6$ and 5,792 bits at $H = W = 3 2$ . These counts exclude model parameters and serialization overhead. Private PQ tables contain NMd parameters (25,165,824 at d = 384 and 50,331,648 at $d = 7 6 8 )$ , plus Dd projection parameters. Compact per-image codes therefore do not imply a small decoder-side dictionary. Appendix E.3 separates attribute rate from sharing of the dictionary, and Appendix E.4 compares this decoder with pooling and an auxiliaryonly readout, then varies transformer depth.

## C.5 TRAINING OBJECTIVES AND REGIME

![](images/bd515620aed49c0f687bb74c4281e13d9e85e1f3a57617214487a39b19e67d3a.jpg)  
Figure 7: Training objectives. Two affine-related views share model parameters, and each prediction matches its own detached VFM target. Potts, entropy, slot presence, and orthogonality use the original view. Distillation and usage diversity average over views; bank presence takes maxima across both; equivariance compares aligned full-bank channels. †: auxiliary distillation is used only in continuous training. ‡: assignment regularizers train the front end in stage 1 but have no gradient to it during frozen PQ training. Orthogonality can still update region modulation in stage 2.

Table 6: Loss weights and view accounting. Frozen PQ still computes assignment regularizers, but they cannot train the frozen prototype-assignment path. Orthogonality can update the unfrozen modulation parameters.
<table><tr><td>Term</td><td>Stage 1</td><td>Stage 2</td><td>Views</td></tr><tr><td> $\mathcal { L } _ { \mathrm { g l o b a l } }$ </td><td>1</td><td>1</td><td>Mean of both</td></tr><tr><td> $\mathcal { L } _ { \mathrm { a u x } }$ </td><td>1</td><td>0</td><td>Mean of both</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { P o t t s } }$ </td><td>1</td><td>1</td><td>Original</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { e n t } }$ </td><td>1</td><td>0.5</td><td>Original</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { o r t h } }$ </td><td>1</td><td>1</td><td>Original</td></tr><tr><td> $\mathcal { L } _ { \mathrm { { s l o t } } }$ </td><td>1</td><td>1</td><td>Original</td></tr><tr><td> $\mathcal { L } _ { \mathrm { e q } }$ </td><td>1</td><td>1</td><td>Paired</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { d i v } }$ </td><td>0.1</td><td>0.1</td><td>Mean of both</td></tr><tr><td> $\mathcal { L } _ { \mathrm { b a n k } }$ </td><td>0.1</td><td>0.1</td><td>Maximum over both</td></tr></table>

Auxiliary region readout. The auxiliary head supervises pooled region content directly, complementing the decoder that sees concept identities and attributes. We adopt PDiscoFormer’s normalization-based modulation (Aniraj et al., 2024), which develops the part modulation of PDiscoNet (van der Klis et al., 2023). A learned LayerNorm $( [ D , { \dot { K } } ] )$ ) acts on the area-scaled $\widetilde { Z } = [ \widetilde { z } _ { i _ { 1 } } , \dots , \widetilde { z } _ { i _ { K } } ] \in \mathbb { R } ^ { D \times K }$ to form modulated features $V _ { m }$ . It normalizes jointly over feature and slot axes. These features support orthogonality and the stage-1 auxiliary head $f _ { \mathrm { a u x } } .$ Following PDiscoNet’s part dropout (van der Klis et al., 2023), we discourage reliance on a single informative region: the auxiliary branch zeros whole slots with slot dropout at probability 0.3 (scaling retained slots by $1 / 0 . 7 )$ , averages over all K slot positions, and applies LayerNorm(D) → Linear(D, D) → GELU → Linear(D, D). Modulation and dropout do not enter the primary residual branch.

Identity surrogate in continuous training. To let global distillation shape assignments despite hard ownership, we use a straight-through construction (Bengio et al., 2013; Jang et al., 2017) based on the selected soft maps A<sup>¯</sup>. During stage 1, we replace the identity contribution in Equation 4 by

$$
t _ { u } ^ { \mathrm { i d } } = e _ { p _ { u } } + \bar { e } _ { u } - \mathrm { s t o p g r a d } ( \bar { e } _ { u } ) , \qquad \bar { e } _ { u } = \sum _ { k } \bar { A } _ { i _ { k } , u } e _ { i _ { k } } .\tag{13}
$$

This has hard forward values but supplies gradients to soft spatial assignments and an additional gradient to concept values. It differentiates neither top-K selection nor hard attribute broadcasting. It connects global distillation to prototypes through A<sup>¯</sup> and to fusion through AnyUp’s feature inputs, while the appearance residual remains detached. It is disabled in stage 2 and evaluation.

Removing the surrogate or enabling residual gradients is studied in Appendix E.5. The auxiliaryonly alternative and the removal of auxiliary distillation are distinguished in Appendices E.4 and E.6.

Combined objective. Global and auxiliary distillation retain target information, while spatial and usage regularizers shape its organization into concepts. Table 6 specifies the weights of the nine terms shown in Figure 7. For each view $v \in \{ x , T ( x ) \}$ , the auxiliary prediction is $\hat { c } _ { \mathrm { a u x } } ( v ) =$ $f _ { \mathrm { a u x } } ( V _ { m } ( v ) )$ with the pooling and dropout above, and

$$
\mathcal { L } _ { \mathrm { a u x } } = \frac { 1 } { 2 } \sum _ { v \in \{ x , T ( x ) \} } [ 1 - \cos ( \hat { c } _ { \mathrm { a u x } } ( v ) , \mathrm { s t o p g r a d } ( c ( v ) ) ) ] .\tag{14}
$$

The total objective is the sum of the nine losses multiplied by their table weights, averaged over batch images wherever an image index remains. Both cosine losses and usage diversity average their two per-view values. Potts, entropy, orthogonality, and slot presence use only the original view. Bank presence takes maxima over both views, and equivariance compares them. Cosine similarity uses $\bar { \epsilon } = 1 0 ^ { - 8 }$ . The auxiliary head matches the same view-specific backbone target as the main decoder.

Feature-weighted Potts. We favor agreement between neighboring assignments when their features are similar, while weakening smoothing across feature discontinuities. This is a local soft Potts penalty with a Gaussian feature kernel, related to the appearance-dependent pairwise potentials of Krahenb¨ uhl & Koltun¨ (2011). We use it in place of PDiscoFormer’s unweighted totalvariation prior (Aniraj et al., 2024). Our adaptive temperature keeps confident logits from saturating the smoothing gradient, and the bandwidth adapts to feature-distance scale. Specifically, soften the clean selected logits without Gumbel noise using $C _ { i _ { k } , u } = \mathrm { s o f t m a x } _ { k } ( \ell _ { i _ { k } , u } / \tau _ { \mathrm { P } } )$ , where $\tau _ { \mathrm { P } } = \mathrm { m a x } ( 1 , \mathrm { m e d i a n } _ { b , u } ( \ell _ { b , u } ^ { ( 1 ) } - \ell _ { b , u } ^ { ( 2 ) } ) / 2 )$ and superscripts denote the two largest selected logits. Here $C$ denotes the spatial probabilities used by the Potts loss. Image indices are suppressed in the spatial expressions below. For each horizontal/vertical neighbor pair $( u , u ^ { \prime } )$ , counted once, let $\zeta _ { u u ^ { \prime } } \mathbf { \bar { = } } \lVert z _ { u } - \bar { z } _ { u ^ { \prime } } \rVert _ { 2 } ^ { 2 }$ denote its squared feature distance and set $\bar { \sigma ^ { 2 } } = \operatorname* { m a x } ( \dot { 1 0 } ^ { - 8 }$ , quantile {ζ<sub>uu′</sub>}). Then

$$
\mathcal { L } _ { \mathrm { P o t t s } } = \mathrm { m e a n } _ { b , ( u , u ^ { \prime } ) } \exp [ - \zeta _ { u u ^ { \prime } } / ( 2 \sigma ^ { 2 } ) ] \left( 1 - \sum _ { k } C _ { i _ { k } , u } C _ { i _ { k } , u ^ { \prime } } \right) .\tag{15}
$$

Features used for the kernel, temperature, and bandwidth are detached. Quantiles and medians are over the local batch. There is no normalization by total edge weight.

Entropy, orthogonality, and presence. The entropy term follows PDiscoFormer (Aniraj et al., 2024) and encourages confident spatial ownership. Orthogonality discourages redundant region features, following the part-decorrelation prior in PDiscoNet (van der Klis et al., 2023) and earlier SCOPS (Hung et al., 2019). We use squared Gram-matrix deviations, penalizing both positive and negative correlations, rather than the signed cosine sum written in PDiscoFormer. Selected-slot presence adapts PDiscoNet’s batch-presence term: spatial averaging prevents an isolated activation from satisfying it. We additionally apply this criterion to full-bank identities to supply gradients to concepts outside the selected subset.

For batch-indexed maps, $i _ { b , k }$ is the bank ID at selected-list position k in image $b ,$ and $k , l \in$ $\{ 1 , \ldots , K \}$ index selected-list positions. With $\widehat { V } _ { m }$ denoting channel-normalized columns of $V _ { m }$

$$
\mathcal { L } _ { \mathrm { e n t } } = - \operatorname* { m e a n } _ { b , u } \sum _ { k } \bar { A } _ { b , i _ { b , k } , u } \log ( \bar { A } _ { b , i _ { b , k } , u } + 1 0 ^ { - 8 } ) ,\tag{16}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o r t h } } = \operatorname* { m e a n } _ { b , k , l } \left[ \left( \widehat { V } _ { m , b } ^ { \top } \widehat { V } _ { m , b } \right) _ { k l } - \mathbf { 1 } _ { k = l } \right] ^ { 2 } , } \end{array}\tag{17}
$$

$$
\mathcal { L } _ { \mathrm { s l o t } } = 1 - \frac { 1 } { K } \sum _ { k } \operatorname* { m a x } _ { b , u } \mathrm { A v g P o o l } _ { 3 \times 3 } ( \bar { A } _ { b , i _ { b , k } } ) _ { u } ,\tag{18}
$$

$$
\mathcal { L } _ { \mathrm { b a n k } } = 1 - \frac { 1 } { N } \sum _ { n } \operatorname* { m a x } _ { \substack { b , u , v \in \{ x , T ( x ) \} } } \operatorname { A v g P o o l } _ { 3 \times 3 } ( A _ { b , n } ^ { v } ) _ { u } .\tag{19}
$$

Pooling uses stride 1 without padding. Orthogonality averages over all $B K ^ { 2 }$ entries. Slot presence is indexed by local list position. Bank presence is indexed by globally shared identity and can train unselected prototypes. Presence maxima are not gathered across devices.

Prototype-usage diversity. Subset balancing does not directly penalize concentrated full-bank spatial responses. We therefore add a differentiable usage penalty toward a detached, Sinkhornbalanced target, using the same balancing principle as above (Sinkhorn & Knopp, 1967; Caron et al., 2020). For each view, let $\psi _ { n } = B ^ { - 1 } \dot { \sum _ { b } _ { b } _ { b , n } }$ be the batch-mean full-bank activation, normalized to sum to one. To form a detached target $\psi ^ { \mathrm { b a l } }$ , apply the same ten-iteration balancing routine to $\mathrm { e x p } ( \mathrm { s t o p g r a d } ( \chi ) ) ^ { \intercal } \ \in \ \mathbb { R } ^ { N \times B }$ (with the column factor now $B )$ , transpose back, average over images, and normalize. The loss is

$$
\mathcal { L } _ { \mathrm { d i v } } = \sum _ { n } \psi _ { n } \big [ \log ( \psi _ { n } + 1 0 ^ { - 8 } ) - \log ( \psi _ { n } ^ { \mathrm { b a l } } + 1 0 ^ { - 8 } ) \big ] .\tag{20}
$$

The exponent here has temperature 1, distinct from the selection temperature 0.1. The final row normalization of the $N \times B$ matrix makes each prototype row sum to one. Thus, when denominator clamps are inactive, averaging over images and normalizing gives $\psi _ { n } ^ { \mathrm { b a l } } = 1 / N$ in exact arithmetic. The loss is therefore an epsilon-stabilized KL divergence toward uniform usage. Without the numerical stabilizers, it equals log $N - \mathcal { H } ( \psi )$ , where $\begin{array} { r } { \bar { \mathcal { H } } ( \psi ) = - \sum _ { n } \psi _ { n } \log \psi _ { n } } \end{array}$ . This term regularizes prototype usage rather than quantizer commitment.

Affine equivariance. To encourage concept identities to follow image content under changes of pose, we adapt the inverse-warp cosine objective of PDiscoNet and PDiscoFormer (van der Klis et al., 2023; Aniraj et al., 2024). Equivariant part maps were also encouraged in SCOPS (Hung et al., 2019). Our comparison uses active full-bank identities so that a change in the selected subset does not change which concepts are compared. For each image $b ,$ let $\mathcal { T } _ { b } ^ { 3 2 }$ contain the 32 bank indices with the largest max $\cdot ( \chi _ { b , n } ( x ) { \dot { , } } \chi _ { b , n } ( T ( x ) { \dot { ) } } ) .$ ). We compare their full-bank maps after inverse warping:

$$
\begin{array} { r } { { \cal { L } } _ { \mathrm { e q } } = 1 - \operatorname * { m e a n } _ { b , n \in { \cal Z } _ { b } ^ { 3 2 } } \cos \bigl ( A _ { b , n } ( x ) , \mathcal { W } _ { T } ^ { - 1 } A _ { b , n } ( T ( x ) ) \bigr ) . } \end{array}\tag{21}
$$

Cosine similarity acts on flattened maps. The resampling operator $\mathcal { W } _ { T } ^ { - 1 }$ first undoes translation, then rotation and scale, using two bilinear resampling operations. Map-space translations are rounded after scaling by the ratio of map to image resolution. No overlap mask removes filled borders. Using globally indexed maps avoids matching unrelated slot positions when the two views select different concepts.

The affine transforms T are sampled as described in Appendix D.3.

## C.6 TRAINING STAGES AND OPTIMIZATION

Two-stage curriculum. The continuous first stage lets concepts form before imposing a discrete appearance bottleneck. Freezing assignment in the second stage then gives quantization a stable concept reference. Stage 1 jointly learns fusion, matching prototypes, continuous appearance projection, concept embeddings, decoder, region modulation, and the auxiliary head. It uses detached residuals and the identity surrogate. Stage 2 starts from the stage-1 checkpoint with the lowest validation global cosine distance among checkpoints saved every ten epochs. It retains fusion, prototypes, decoder embeddings, positional and CLS embeddings, transformer and output head, and region modulation. Fusion and prototypes are frozen. Usage and dual estimates are retained without updates, prototype resampling is disabled, and the auxiliary head is removed. The continuous projection is discarded. A new PQ projection and private codebooks are initialized as specified in Appendix C.3. These, the concept embeddings, and decoder learn through global distillation. Region modulation remains trainable through orthogonality but does not enter the decoder’s appearance path. A fresh optimizer and learning-rate schedule are used for stage 2.

Freezing assignment parameters does not make the training forward deterministic: both stages use Gumbel noise and batch-balanced selection. Evaluation removes the noise, uses the stored dual for independent selection, and performs hard PQ lookup. Table 7 summarizes the gradient and state changes.

Table 7: Canonical stage transition. The backbone and AnyUp weights are frozen in both stages, residuals are detached in both.
<table><tr><td>Component or operation</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Fusion and matching prototypes</td><td>Learned</td><td>Retained, frozen</td></tr><tr><td>Appearance encoder</td><td>Continuous projection</td><td>New private PQ</td></tr><tr><td>Concept embeddings and decoder</td><td>Learned</td><td>Retained, learned</td></tr><tr><td>Auxiliary distillation</td><td>Enabled</td><td>Disabled</td></tr><tr><td>Identity gradient surrogate</td><td>Enabled</td><td>Disabled</td></tr><tr><td>AnyUp feature-input gradients</td><td>Enabled</td><td>Disabled</td></tr><tr><td>Usage and selection dual</td><td>Updated</td><td>Retained, fixed</td></tr><tr><td>Dead-prototype resampling</td><td>Enabled</td><td>Disabled</td></tr></table>

Usage and prototype resampling. Balancing alone cannot restore a prototype that receives little useful learning signal. We therefore replace persistently underused prototypes with current region features. Stage 1 initializes $\rho _ { n } = 1 / N$ . After computing both views, selected-ID counts are summed across views and devices and divided by the total number of selected IDs, forming a frequency vector. Usage is updated as 0.99 times its old value plus 0.01 times this frequency vector. After each optimizer step, prototypes with $\rho _ { n } < 1 0 ^ { - 4 }$ are replaced by normalized area-scaled region features drawn uniformly from the current batch’s pooled features across both views (without replacement when the pool is large enough), scaled to the mean active-prototype norm, plus Gaussian noise of standard deviation $1 \mathrm { { 0 } ^ { - 3 } }$ . If all prototypes are inactive, use the whole-bank mean norm. Reset the replaced prototype’s usage to $1 \bar { / } N$ , its dual to zero, and its value embedding from the replacement prototype. Resetting individual dual entries can temporarily change the vector’s mean. It is recentered at the next dual update. A common additive shift does not affect selection rankings. Optimizer moments are not reset. Under distributed training, one device samples from its own feature pool and broadcasts the changes. Appendix E.2 measures the effect of disabling this replacement.

The augmentation, optimizer, schedules and checkpoint selection of DisParQ are given in Appendix D.3, together with the schedules of every dataset.

Order of operations. For each training batch, form the two views and compute their frozen targets and fused, upsampled features. Independently for each view, compute full-bank maps, balance mean scores, select concepts, recompute selected maps, pool residuals, and decode using continuous or quantized attributes. Sum the weighted losses with the view accounting in Table 6, backpropagate, clip gradients, and take one optimizer step. In stage 1, update usage from both views and estimate the selection dual from the original view during the forward computation. After the optimizer step, resample inactive prototypes. In stage 2 these assignment states remain fixed. At evaluation, a single view preprocessed according to Appendix D.5 supplies ordinary soft maps, dual-based selection, hard ownership, and hard attribute codes. Because balancing and several regularizers use local batches, distributed execution is not equivalent to a single solve on the concatenated batch.

The objective-removal studies in Appendix E.6 distinguish preservation of downstream information from spatial and semantic organization.

## D EXPERIMENTAL PROTOCOL

This section specifies the datasets, backbones, training procedures and evaluation metrics used in Section 4 and Appendices E–G, including method-specific exceptions.

## D.1 DATASETS AND SPLITS

Table 8 lists ImageNet-1k (Deng et al., 2009), Places365-Standard (Zhou et al., 2018), PartImageNet (He et al., 2022), CUB-200-2011 (Wah et al., 2011), Stanford Cars (Krause et al., 2013), Stanford Dogs (Khosla et al., 2011) and Oxford Flowers (Nilsback & Zisserman, 2008). We train one DisParQ model per dataset without class labels or part annotations. We use class labels to fit classification probes and label the kNN reference set. Keypoint annotations train the keypoint regressors, and part annotations support concept evaluation. Supervised baseline training and selection among evaluated configurations are described in Section D.4.

We report results on the validation splits of ImageNet-1k, Places365 and PartImageNet and on the test splits of the other datasets. Appendix G.7 additionally uses the PartImageNet test split. On Oxford Flowers, we merge the training and validation splits, following PDiscoFormer (Aniraj et al., 2024).

Table 8: Datasets. Split sizes, available part annotations and corresponding results tables. Oxford Flowers uses the combined training and validation splits for training.
<table><tr><td>Dataset</td><td>Classes</td><td>Training</td><td>Evaluation</td><td>Part annotation</td><td>Tables</td></tr><tr><td>ImageNet-1k</td><td>1000</td><td>1,281,167</td><td>val 50,000</td><td>none</td><td>15</td></tr><tr><td>Places365</td><td>365</td><td>1,803,460</td><td>val 36,500</td><td>none</td><td>15</td></tr><tr><td>PartImageNet</td><td>158</td><td>20,466</td><td>val 1,206</td><td>40 part masks</td><td>2,4, 19</td></tr><tr><td>CUB-200-2011</td><td>200</td><td>5,994</td><td>test 5,794</td><td>15 keypoints</td><td>3, 16, 19</td></tr><tr><td>Stanford Cars</td><td>196</td><td>8,144</td><td>test 8,041</td><td>none</td><td>3</td></tr><tr><td>Stanford Dogs</td><td>120</td><td>12,000</td><td>test 8,580</td><td>none</td><td>3</td></tr><tr><td>Oxford Flowers</td><td>102</td><td>2,040</td><td>test 6,149</td><td>none</td><td>3</td></tr></table>

## D.2 BACKBONES

Table 9 lists DINOv2 (Oquab et al., 2024) with registers (Darcet et al., 2024), DINOv3 (Simeoni´ et al., 2025), TIPSv2 (Cao et al., 2026), CLIP (Radford et al., 2021), SigLIP2 (Tschannen et al., 2025) and CLIP-DINOiser (Wysoczanska et al.´ , 2024). In DisParQ, these feature extractors are frozen and receive 224 × 224 images. The input wrapper converts ImageNet-normalized images to each encoder’s pretrained normalization, while AnyUp retains ImageNet-normalized guidance. We fuse patch tokens from several layers as described in Appendix C.1, except for CLIP-DINOiser, which directly supplies its refined projected features. The global distillation targets are listed in the table.

AnyUp (Wimmer et al., 2026) increases the native $1 6 \times 1 6$ grid of patch-14 models to $3 2 \times 3 2$ and the 14 × 14 grid of patch-16 models to $2 8 \times 2 8$ . Baseline input resolutions, trainable tokens and scoring grids have the exceptions described in Sections D.4 and D.5.

The main controlled comparison uses DINOv2-B/14, with CFM evaluated on its released CLIP-DINOiser dictionary. Figure 8 covers seven configurations of six feature extractors because SigLIP2 has two targets. We use CLIP ViT-L/14 for the MSAE reference on its original backbone (Figure 23 and Table 16).

CLIP-DINOiser combines a frozen CLIP encoder with the authors’ released 1.8M-parameter adapter<sup>3</sup>. Two convolutions predict patch affinities that pool dense CLIP features into the denoised 512-dimensional map used by DisParQ. Its encoder is the listed OpenCLIP checkpoint (Cherti et al., 2023), trained on LAION-2B (Schuhmann et al., 2022). The separate CLIP ViT-B/16 row uses OpenAI weights.

Table 9: Feature extractors and targets. Checkpoints are Hugging Face Hub identifiers. The grid is the assignment grid of DisParQ, or the scoring grid of the CLIP-L/14 MSAE reference. Targets are the frozen global features distilled by DisParQ at 224 × 224. A dash indicates a baseline-only encoder.
<table><tr><td>Backbone</td><td>Checkpoint</td><td>Grid</td><td>Target</td></tr><tr><td>DINOv2-B/14, registers</td><td>facebook/dinov2-with-registers-base</td><td>32 × 32</td><td>CLS token</td></tr><tr><td>DINOv3-B/16</td><td>timm/vit_base-patch16_dinov3.lvd1689m</td><td>28 × 28</td><td>CLS token</td></tr><tr><td>TIPSv2-B/14</td><td>google/tipsv2-b14</td><td>32 × 32</td><td>CLS token</td></tr><tr><td>CLIP ViT-B/16</td><td>openai/clip-vit-base-patch16</td><td>28 × 28</td><td>CLS token after the last layer norm</td></tr><tr><td>SigLIP2-B/16</td><td>google/siglip2-base-patch16-224</td><td>28 × 28</td><td>attention-pooler output, or mean patch token</td></tr><tr><td>CLIP-DINOiser</td><td>laion/CLIP-ViT-B-16-laion2B-s34B-b88K 28× 28</td><td></td><td>projected CLIP image em-</td></tr><tr><td>CLIP ViT-L/14</td><td>openai/clip-vit-large-patch14</td><td>32 × 32</td><td>bedding</td></tr></table>

## D.3 TRAINING DISPARQ

The main DINOv2 experiments use the architecture, losses and two-stage curriculum of Appendix C, with N = 1024 concepts, K = 16 selected concepts per image and concept-specific PQ with $G =$ 16 groups of M = 64 codewords. Schedules and batch sizes vary across datasets, and Places365 uses mixed upsampling in stage 2.

Augmentation. Each training image produces two views. The first combines crop and photometric augmentations commonly used in self-supervised vision (Chen et al., 2020; Caron et al., 2021) to diversify appearance. The second applies a random affine transform to the normalized first view, following the affine view pairing used by PDiscoNet and PDiscoFormer (van der Klis et al., 2023; Aniraj et al., 2024). We draw one transform per local batch, providing known spatial correspondences for $\mathcal { L } _ { \mathrm { e q } } .$ . Table 10 specifies our recipe.

Table 10: Augmentations of DisParQ. The first-view transformations are applied in the listed order. The second view is an affine transform of the normalized first view.
<table><tr><td>Transformation</td><td>Probability</td><td>Parameters</td></tr><tr><td>First view</td><td></td><td></td></tr><tr><td>Random resized crop</td><td>1</td><td>scale [0.64, 1], ratio [3/4, 4/3], 224 × 224, bicubic</td></tr><tr><td>Horizontal flip</td><td>0.5</td><td></td></tr><tr><td>Color jitter</td><td>0.8</td><td>brightness and contrast 0.4, saturation 0.2, hue 0.1</td></tr><tr><td>Grayscale</td><td>0.2</td><td></td></tr><tr><td>Gaussian blur</td><td>0.5</td><td>kernel 9, σ ∈ [0.1, 2]</td></tr><tr><td>Normalization</td><td>1</td><td>mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)</td></tr><tr><td>Second view</td><td></td><td></td></tr><tr><td>Affine transform</td><td>1</td><td>rotation [—90°, 90°], translation ±10%, scale [0.7, 1.3], no shear, bilinear interpolation, zero fill in normalized units</td></tr></table>

Optimization. Tables 11 and 12 specify optimization and dataset-specific schedules. Both views are balanced separately within each local batch and jointly contribute to one optimizer update. Training loaders shuffle and drop incomplete batches. Learning rates are not automatically scaled with batch size.

Table 11: Optimization and checkpoint selection of DisParQ. Schedules and per-device batch sizes are in Table 12.  
Setting Value   
Optimizer AdamW (Loshchilov & Hutter, 2019), (β , β ) = (0.9, 0.999), ϵ = 10 −<sup>8</sup>   
Weight decay $1 0 ^ { - 4 }$ , on all trainable parameters   
Learning-rate schedule linear warm-up per epoch from $1 0 ^ { - 4 } \times$ the peak, then cosine decay (Loshchilov & Hutter, 2017)   
to 10−<sup>6</sup>   
Devices and accumulation one GPU, or four H100s for ImageNet-1k and Places365. No gradient accumulation   
Precision and clipping bfloat16 mixed precision and gradient-norm clipping at 1   
Seed 42 for main experiments. Ablation replication is specified in Appendix E   
Checkpoint selection lowest validation $\mathcal { L } _ { \mathrm { g l o b a l } }$ among checkpoints saved every 10 epochs (stage 1) or every epoch   
(stage 2). Stage 2 starts from the selected stage-1 checkpoint

Mixed upsampling on Places365. To reduce computation, Places365 uses AnyUp in a random 20% of stage-2 batches and bilinear interpolation otherwise. The other main training runs and downstream evaluation use AnyUp throughout. Appendix E.1 compares these alternatives.

## D.4 TRAINING THE BASELINES

We train the dataset-specific baselines on the training data in Section D.1, using the authors’ implementations where available. The main comparison uses DINOv2-B/14. Exceptions include the released CFM dictionary, released PDiscoFormer checkpoints and the MSAE reference on CLIP-L/14. Table 13 summarizes these models and their probe inputs. The vision-language and supervised reference entries in Figure 5 (left) and Table 15 reproduce results reported by CFM (Wittenmayer et al., 2026) and ProtoQuant (Janusz et al., 2026).

Sparse autoencoders. TopK SAE (Gao et al., 2025) is our re-implementation. PatchSAE (Lim et al., 2025)<sup>4</sup> and MSAE (Zaigrajew et al., 2025)<sup>5</sup> use the authors’ code and hyperparameters (Table 14). These models are fitted to frozen DINOv2-B/14 patch tokens for the main comparison. Following the MSAE authors, we remove its TopK constraint at inference.

On PartImageNet, we sweep dictionary sizes and, for MSAE, both loss weightings and the three published expansion factors. The main table presents the best evaluated configuration of each SAE by mean rank over the reported columns of Table 2. For the fine-grained table, we select one MSAE configuration jointly across all four datasets, requiring an available LP result on each dataset and averaging ranks over the ten reported metrics. In both selections, metrics are ranked in their preferred direction on unrounded values, missing metrics rank last, and exact ties follow the fixed candidate order. Candidates without a foreground-purity result are excluded on PartImageNet. A tie in the aggregate rank is also resolved by candidate order. These are best-of-sweep references selected using the reported validation or test scores, including annotated-part metrics where available. Representation fitting remains label-free, but this configuration selection uses evaluation labels. The selected MSAE sizes are 8192 on PartImageNet and 6144 on the fine-grained datasets with reverse and uniform weighting respectively. Figures 22 and 23 show the evaluated alternatives.

CFM. CFM (Wittenmayer et al., 2026) is a Matryoshka sparse autoencoder trained by its authors on CC12M features of CLIP-DINOiser. We use its released dictionary<sup>6</sup> without refitting it to our datasets.

PIP-Net and ProtoQuant. Both use the authors’ code<sup>7</sup> with the training settings in Table 14, adapted to frozen DINOv2-B/14 features. Stage 1 is label-free and appears with the self-supervised methods. Stage 2 adds supervised classification and is an unranked reference. ProtoQuant’s origina recipe first fine-tunes the backbone on the target dataset. Our adaptation keeps that backbone frozen to match the feature extractor used by the main self-supervised comparison.

Table 12: Training schedules of DisParQ with DINOv2-B/14 and width 768. The PartImageNet rows give the canonical schedule. Batch size is per device.
<table><tr><td>Stage</td><td>Epochs</td><td>Warm-up</td><td>Peak LR</td><td>Batch</td></tr><tr><td colspan="5">PartImageNet</td></tr><tr><td>Continuous concept discovery</td><td>150</td><td>10</td><td> $3 \cdot 1 0 ^ { - 4 }$ </td><td>16</td></tr><tr><td>Concept-specific PQ, 16 × 64</td><td>200</td><td>10</td><td> $1 0 ^ { - 3 }$ </td><td>16</td></tr><tr><td colspan="5">CUB-200-2011, Stanford Cars, Stanford Dogs, Oxford Flowers</td></tr><tr><td>Continuous concept discovery</td><td>200</td><td>40</td><td> $3 \cdot 1 0 ^ { - 4 }$ </td><td>16</td></tr><tr><td>Concept-specific PQ,  $1 6 \times 6 4$ </td><td>500</td><td>50</td><td> $1 0 ^ { - 3 }$ </td><td>16</td></tr><tr><td colspan="5">ImageNet-1k, Places365</td></tr><tr><td>Continuous concept discovery</td><td>50</td><td>10</td><td> $3 \cdot 1 0 ^ { - 4 }$ </td><td>64</td></tr><tr><td>Concept-specific PQ,  $1 6 \times 6 4$ </td><td>100</td><td>10</td><td> $1 0 ^ { - 3 }$ </td><td>64</td></tr></table>

Table 13: Baselines in the main comparison. Size denotes the global dictionary or prototype count, with PCA specified per image or class. Grid denotes the scoring resolution. The notation 16 → 32 indicates bilinear resampling from the native $1 6 \times 1 6$ maps to $3 2 \times 3 2$ . Probe inputs are used for LP and kNN.
<table><tr><td>Method</td><td> $\operatorname { c o d e }$ </td><td>Class labels</td><td>Size</td><td>Grid</td><td>Probe input</td></tr><tr><td>TopK SAE</td><td>ours</td><td>no</td><td>2048 latents</td><td> $1 6  3 2$ </td><td>mean-pooled reconstruction</td></tr><tr><td>PatchSAE</td><td>authors&#x27;</td><td>no</td><td>49,152 latents</td><td>16→32</td><td>mean-pooled reconstruction</td></tr><tr><td>MSAE</td><td>authors&#x27;</td><td>no</td><td>8192 latents (fine-grained: 6144)</td><td> $1 6  3 2$ </td><td>mean-pooled reconstruction</td></tr><tr><td>CFM</td><td>released</td><td>no</td><td>8192 latents</td><td> $2 8 \times 2 8$ </td><td>max- and mean-pooled</td></tr><tr><td>PIP-Net, stage 1 / 2</td><td>authors&#x27;</td><td>stage 2</td><td>768 prototypes</td><td> $3 2 \times 3 2 ( \mathrm { A n y U p } )$ </td><td>activations pooled prototype scores</td></tr><tr><td>ProtoQuant, stage 1 / 2</td><td>authors&#x27;</td><td>stage 2</td><td>2048 codes (CUB: 4000)</td><td> $1 6  3 2$ </td><td>pooled prototype scores</td></tr><tr><td>PDiscoFormer</td><td>released or retrained</td><td>yes</td><td>K parts + background</td><td> $1 6 \times 1 6 ( \mathrm { C U B } ;$   $3 7  3 2 )$ </td><td>mean part embedding</td></tr><tr><td>PCA, per image / class-wise</td><td>ours</td><td>class-wise only</td><td>16 per image or class</td><td> $1 6  3 2$ </td><td>none</td></tr></table>

Table 14: Baseline training settings. The table lists the recipes used for our retraining. MSAE runs for at least 21,000 steps, extending beyond 30 epochs on smaller datasets. CFM uses released weights and PCA is fitted directly to patch features.
<table><tr><td>Method</td><td>Optimizer</td><td>Learning rate</td><td>Batch</td><td>Length</td><td>Other settings</td></tr><tr><td>TopK SAE</td><td>Adam</td><td> $3 \cdot 1 0 ^ { - 4 }$ </td><td>64 images</td><td>20 epochs</td><td>8 latents per patch, 16 per image, auxiliary loss, weight 1/32</td></tr><tr><td>PatchSAE</td><td>Adam</td><td> $4 \cdot 1 0 ^ { - 4 } , 5 0 0$  warm-up steps</td><td>128 images</td><td>20,480 steps</td><td>L1 weight  $8 \cdot 1 0 ^ { - 5 }$  , ghost gradients</td></tr><tr><td>MSAE</td><td>AdamW</td><td> $1 0 ^ { - 4 }$ </td><td>4096 tokens</td><td>30 epochs,  $\geq 2 \mathrm { { \dot { 1 } } , 0 0 0 \ s t e p s }$ </td><td>nested levels 64, 128, . . . up to the dictionary size, uniform or reverse</td></tr><tr><td>PIP-Net</td><td>AdamW</td><td> $5 \cdot 1 0 ^ { - 4 } .$  classifier</td><td>128 / 64</td><td>10 + 60 epochs</td><td>weighting stage 1 / stage 2, features upsampled by AnyUp</td></tr><tr><td>ProtoQuant</td><td>AdamW</td><td>0.05 0.05</td><td>512</td><td>30 + 30 epochs</td><td>stage 1 / stage 2, temperature 0.1</td></tr><tr><td>PDiscoFormer, retrained</td><td>Adam</td><td> $1 . 4 1 4 \cdot 1 0 ^ { - 6 } { \mathrm { ~ b a s e , } }$  halved every 4 epochs</td><td>32 (PartImageNet:  $4 \times 3 2 ,$  accumulated)</td><td>28 epochs</td><td>position embedding, CLS and register tokens trained</td></tr></table>

PDiscoFormer. Tables 2 and 3 evaluate the authors’ released checkpoints<sup>8</sup> for PartImageNet, CUB and Oxford Flowers. We additionally retrain the evaluated part counts on PartImageNet, including a frozen-token variant at $K = 2 5$ (marked <sup>†</sup> in Figure 21), and train $K = 1 6$ models on Stanford Cars and Stanford Dogs, which have no released checkpoints. The retraining recipe updates the positional embeddings, CLS and register tokens while keeping the encoder blocks frozen, except in the explicitly frozen-token variant. Relative to the base rate in Table 14, learning rates are multiplied by $1 0 ^ { 3 }$ for these tokens and by $1 0 ^ { 4 }$ for prototypes, modulation and classifier parameters. Following the retraining procedure, we select the snapshot with the highest test accuracy. These supervised references therefore use the evaluation labels for checkpoint selection.

PDiscoFormer’s PartImageNet recipe combines the training and validation splits, so the validation images in Table 2 were also seen with their labels during representation learning. Its released CUB checkpoints retain their $5 1 8 \times 5 1 8$ input resolution. Appendix F.4 gives the retraining results, and Sections D.5 and D.6 describe how these models are scored.

PCA. We include two PCA references using 16 components per image, fitted to native $1 6 \times 1 6$ DINOv2-B/14 patch features. Each patch is assigned to the component with the largest absolute projection. Per-image PCA fits a new basis for each image, so its component identities have no shared meaning across images. Class-wise PCA fits one basis per training class and selects the basis using the evaluation image’s ground-truth class. Each class has distinct global component IDs, giving 16 times the number of classes in the full bank. This ground-truth-conditioned representation makes a classification probe circular, so we report no downstream probe for it.

On PartImageNet, PCA also estimates foreground from the sign of the first component. We orient its sign so that the mean projection of the four corner patches is nonpositive and retain patches with positive projection. On CUB, all patches are retained. This heuristic changes PCA’s spatial support and is part of the baseline definition.

## D.5 EVALUATION PROTOCOL

For locally evaluated models, we use the metric implementations in Section D.6 on the stated evaluation splits. The default budget is $K = 1 6$ concepts per image, with exceptions indicated by the $K / N$ column. Published reference scores retain their original protocols. We disable the test-time Gumbel noise in PDiscoFormer’s released implementation to make its concept maps deterministic.

Inputs. Our classification and image-to-image retrieval evaluations use a short-side resize to 256 followed by a $2 2 4 \times 2 2 4$ center crop. PIP-Net and ProtoQuant use the same crop-based probing pipeline. The dictionary baselines and PDiscoFormer are probed on cached features from squareresized images, preserving the protocol used to obtain their reported results. Thus using the same probe architecture does not imply identical input geometry or augmentation.

All reported mask and keypoint metrics use paired square resizing of images and annotations. The common input size is $2 2 4 \times 2 2 4 .$ , with nearest-neighbor interpolation for masks and corresponding coordinate scaling for keypoints. This retains the full annotated object and aligns spatial evaluation across methods. It follows PIP-Net’s full-image $2 2 4 \times 2 2 4$ protocol and $3 \bar { 2 } \times 3 \bar { 2 }$ purity windows (Nauta et al., 2023). For the PDiscoFormer part-discovery and CFM grounding metrics (Aniraj et al., 2024; Wittenmayer et al., 2026), we adopt their metric definitions in this common geometry. In particular, our square-resized PartImageNet evaluation differs from PDiscoFormer’s released center-crop evaluation.

Released PDiscoFormer CUB checkpoints retain the authors’ 518 × 518 input size, preserving their learned spatial configuration. We scale the purity window to $7 4 \times 7 4$ to keep its relative size close to 32/224. Keypoint regression uses image-normalized coordinates for every method, as defined in Section D.6.

Scoring grids. Table 13 lists the baseline grids. DisParQ and PIP-Net assign concepts on AnyUp’s 32 × 32 DINOv2-B/14 feature grid. The sparse autoencoders, ProtoQuant and PCA produce native 16 × 16 maps, which we resample bilinearly to $3 2 \times 3 2$ for scoring. CFM retains a 28 × 28 grid, and PDiscoFormer retains $1 6 \times 1 6$ on PartImageNet. Its $3 7 \times 3 7$ CUB maps are resampled to $3 2 \times 3 2$

Clustering metrics and part retrieval reduce annotations to these grids. Segmentation metrics instead upsample the concept maps to the image grid, and keypoint regression retains continuous target coordinates. The main tables therefore include resolution differences for CFM and PDiscoFormer, which can affect boundary-sensitive scores. Retrieval rows with a different region set are marked <sup>‡</sup> and accompanied by their own oracle.

Concept budget. TopK SAE is trained with at most 16 concepts per image, and its reconstruction retains this budget for downstream probing. PatchSAE, MSAE and CFM retain each image’s 16 strongest concepts for spatial metrics and part retrieval. Their LP, kNN and image-to-image retrieval instead use the unbudgeted representations listed in Table 13. Figure 22 includes other budgets and the unrestricted CFM dictionary used in its published evaluation.

Probing representations. Our probes use the predicted global representation $\hat { c } ( x )$ , which preserves the target definition for each backbone in Table 9. Baseline representations are listed in Table 13. We evaluate the representation passed downstream by each method.

1. Linear probe (LP). We apply batch normalization without affine parameters, followed by a linear classifier trained on the training split. Optimization uses SGD with momentum 0.9, no weight decay and learning rate $0 \mathrm { { . 1 \cdot { B _ { g l o b a l } } / 2 5 6 } }$ , where $B _ { \mathrm { g l o b a l } }$ is the global batch size. Training lasts 50 epochs, with 10 epochs of warm-up followed by cosine decay. For DisParQ, PIP-Net and ProtoQuant, training images are re-encoded under random resized crops and horizontal flips. Dictionary baselines and PDiscoFormer use precomputed vectors without probe-time augmentation. Cached dictionary probes use batches of 256. These choices preserve the recorded evaluation pipelines, so LP comparisons include differences in augmentation, image geometry and optimizer step counts.

2. kNN. We predict the class with the largest weighted vote among the 20 nearest training images under cosine similarity. A neighbor with similarity s contributes weight exp(s/0.07).

Fine-grained datasets. Table 3 reports LP and kNN accuracy on all four datasets, together with PIP purity and keypoint regression error on CUB. CUB provides annotations for 15 keypoints. Stanford Cars, Stanford Dogs and Oxford Flowers have no part annotations.

## D.6 METRICS

LP and kNN. We report top-1 classification accuracy on the evaluation split using the probes in Section D.5.

Image retrieval. Images are represented by the same frozen vectors used for probing. Each evaluation image queries the other evaluation images by cosine similarity, and a retrieved image is relevant when it shares the query’s class. Recall at rank r, denoted R@r, is the fraction of queries with at least one relevant result among the first r candidates. For each query, average precision (AP) averages the precision at ranks containing relevant results over the full ranking. Mean average precision (mAP) averages AP over queries. For image retrieval, queries with no other image of the same class are excluded from both recall and mAP.

Foreground purity. We reduce each mask to the scoring grid by a majority vote per cell. Let $C _ { n j }$ count cells assigned to global concept n with ground-truth label j, including the background label where it is annotated. Let $\mathcal { T } _ { \mathrm { f g } }$ be the foreground part labels and $N _ { \mathrm { f g } }$ the number of annotated foreground cells in the evaluated split. Then

$$
\mathrm { P u r } _ { \mathrm { f g } } = \frac { 1 } { N _ { \mathrm { f g } } } \sum _ { n } \operatorname* { m a x } _ { j \in \mathcal { I } _ { \mathrm { f g } } } C _ { n j } .
$$

A foreground cell left unassigned contributes to $N _ { \mathrm { f g } }$ but to no concept count, so rejecting foreground cannot improve purity by reducing its denominator. On CUB, counts use the grid cells containing visible keypoints.

NMI, ARI and their mapped variants. We compare the ground-truth part labels y with concept identities c over the annotated foreground cells of the evaluated split, following part-discovery evaluation (Choudhury et al., 2021; van der Klis et al., 2023; Aniraj et al., 2024). Unassigned foreground cells receive a separate cluster label. NMI normalizes mutual information by the arithmetic mean of the two entropies,

$$
\mathrm { N M I } = { \frac { 2 I ( y ; c ) } { H ( y ) + H ( c ) } } .
$$

The Rand index (RI) measures the fraction of cell pairs that both labelings place together or both place apart. ARI corrects this agreement for chance,

$$
\mathrm { A R I } = \frac { \mathrm { R I } - \mathbb { E } [ \mathrm { R I } ] } { 1 - \mathbb { E } [ \mathrm { R I } ] } ,
$$

where the expectation assumes random assignments with fixed cluster sizes, giving chance an expected score of zero.

Raw NMI and ARI penalize splitting one part across several pure concepts. For $\mathrm { N M I } _ { m }$ and $\mathrm { A R I } _ { m }$ we first map each concept to its dominant label on the evaluated split, arg $\operatorname* { m a x } _ { j } C _ { n j }$ , including background where annotated. We then compute NMI and ARI on the mapped labels, retaining a separate cluster for unassigned cells. On PartImageNet, this mapping uses at most 41 semantic labels, comprising 40 parts and background, in addition to the unassigned cluster. Mapping removes the penalty for splitting a part into concepts that receive the same label. The mapping is fitted on the evaluated split, so these are descriptive clustering scores.

mIoU and FG-ARI. Our mIoU (He et al., 2022) is a per-image matched-part score. We upsample the K concept maps to input resolution, assign each pixel to its strongest concept, and use Hungarian one-to-one matching to annotated parts. IoU is averaged over matched pairs, then images, excluding unmatched parts. This differs from dataset-level semantic mIoU. FG-ARI (Locatello et al., 2020) averages per-image ARI over annotated foreground pixels, rather than the split-wide grid cells used above.

Locality, consistency and impurity. We follow CFM’s metric implementation (Wittenmayer et al., 2026). We sort nonnegative map activations and find the cutoff where cumulative mass reaches 90%. All pixels at or above this cutoff are retained, including ties. Thus a uniform hard region retains its entire support. Part masks are dilated by $\mathbf { a \ 3 \times 3 }$ maximum filter.

Locality takes the best IoU between each part and any concept support, averages over images containing that part, and then over the 40 PartImageNet part labels. Consistency first averages each concept–part IoU over images in which either the concept or the part is present. It then selects the best concept for each part and averages over parts. Impurity measures natural-log entropy of the retained activation mass over the dilated label masks, including background, after normalizing those label masses. We average entropy over images within each concept, then over concepts active in at least five images.

As an approximate implementation check under CFM’s original center-crop/1 $4 \times 1 4$ protocol, its unrestricted released dictionary gives locality 44.3, consistency 15.3 and impurity 0.333 (published: 44.3, 15.3, 0.330). This differs from our common square-resize/28 × 28 evaluation.

PIP purity. We adapt PIP-Net’s top-image purity evaluation (Nauta et al., 2023) to each method’s concept maps on the CUB test split, merging left and right versions of each part.

1. For each concept, retain up to 10 images with its highest peak activations.

2. In each image, center a $3 2 \times 3 2 \cdot$ -pixel window on the map’s peak. If several cells share the maximum, choose the maximal cell nearest their centroid. Shift windows at image borders to retain their size. The released PDiscoFormer CUB models use the scaled $7 4 \times { \bar { 7 } } 4$ window described in Section D.5.

3. For each part, compute the fraction of retained images whose window contains its keypoint, using only images in which that part is visible.

4. Take the largest part-specific fraction as the concept’s purity, then average over concepts active in at least one image.

Kp NME. Keypoint regression tests whether concept maps locate the 15 CUB keypoints, following PDiscoNet and PDiscoFormer (van der Klis et al., 2023; Aniraj et al., 2024). Each concept map supplies its two centroid coordinates, normalized by image size, and its mass. Features are indexed by global concept identity, with zeros for an absent concept. We rank concepts by training-set usage and select the basis size from {16, 32, 64, 128, 256, 512} using a 20% holdout drawn from the training split with seed 0. Candidate sizes exceeding the available basis are skipped. If fewer than 16 concepts are available, we use all of them. After selection, the regressor is refitted on the ful training split. For each keypoint, ridge regression with regularization $\mathrm { \tilde { 1 0 } ^ { - 3 } }$ is fitted using training images where that keypoint is visible.

Kp NME is 100 times the mean Euclidean error over all visible test keypoints in image-normalized coordinates. Our PDiscoFormer entries in Tables 3 and 16 are recomputed with this evaluator. The original paper’s bounding-box-normalized errors are a different quantity.

Part-to-part retrieval. The benchmark in Section 4.1 compares regions through globally indexed concept identities.

1. Regions. On PartImageNet, each part label covering at least two cells of the scoring grid defines one region per image, giving 3,524 regions at $3 2 \times 3 2$ On CUB, each visible keypoint defines a one-cell region. To bound the quadratic cost of ranking, we sample 8,000 regions uniformly without replacement using seed 0. The 224 × 224 CUB evaluation of DisParQ yields 68,224 regions before sampling. Counts can differ slightly for the 518×518 PDiscoFormer inputs.

2. Descriptors. A region is represented by a histogram of the global concept IDs assigned to its cells, normalized to sum to one. For DisParQ, the histogram counts the globally indexed identities $p _ { u }$

3. Ranking. Each region queries regions from other images by cosine similarity, with random tie-breaking using seed 0. The same region set supplies queries and gallery candidates.

4. Scoring. A retrieved region is relevant when it has the query’s part label. R@r and mAP follow the definitions above. Queries without a relevant gallery candidate contribute zero to recall and are excluded from mAP.

Rows marked <sup>‡</sup> in Tables 4 and 19 use different grids and region sets with their own oracles. CFM’s 28 × 28 grid gives 3,466 PartImageNet regions and oracle R@1 44.5. Part-to-image retrieval compares the region histogram with a whole-image concept histogram, also normalized to sum to one. It excludes the query image and treats an image as relevant when it contains the queried part.

Oracle and chance. For part retrieval, the category-only reference is implemented as a cooccurrence-group oracle. It ranks regions sharing the query’s label group before other regions, with random ordering within both sets. We construct groups from the evaluated split by visiting part labels in label order. Each unassigned label starts a group and adds the remaining unassigned labels whose image-presence sets have Jaccard overlap of at least 0.6 with that seed label. This is a greedy grouping around the seed, without transitive merging.

For part-to-image retrieval, each gallery image receives the group of its lowest-index retained part label, or a separate sentinel group if it has no retained parts. Images sharing the query’s group are ranked first, with random ordering within both sets.

On PartImageNet, these groups capture supercategory co-occurrence, with 38 of the 40 part labels in 12 groups. The construction also assigns groups to the remaining labels. This measures retrieval explained by co-occurrence without identifying the particular part. It does not use bird-species labels on CUB. Chance ranks all candidates randomly. Both reference scores average five random rankings. Classification and image-to-image oracle entries instead use the ground-truth image class directly.

Concept contribution. The qualitative explanations in Figure 1, Section 4.3 and Appendix G.5 use a separate linear classifier on cˆ(x). It is fitted to features standardized using training-set means and standard deviations. Training uses AdamW for 200 epochs, learning rate $\overline { { 1 } } 0 ^ { - 3 }$ with cosine decay, weight decay $1 0 ^ { - 2 }$ , batches of 1024 and label smoothing 0.1. Its predicted class defines the logit used for explanations, while the quantitative LP results use the probe in Section D.5.

We measure a concept’s contribution by removing it from the discrete description and decoding the modified description again. This is an occlusion test (Zeiler & Fergus, 2014) at the concept level, related to concept interventions (Koh et al., 2020). The removed concept’s cells are reassigned to the strongest remaining concept among the already selected IDs $i _ { 1 : K }$ . The concept embeddings and attribute codes remain fixed, so only the spatial identity map p changes. We measure the drop in the original predicted-class logit and normalize the positive drops across the selected concepts to sum to one. These weights describe relative positive support under the intervention, rather than an additive decomposition of the classifier’s logit.

## E ABLATION STUDIES

Shared protocol. We organize the ablations around the components of Section 3 and Appendix C.

To reduce computational cost, most component studies train the continuous first stage with bilinear feature upsampling instead of AnyUp.

Unless specified otherwise, the component control uses PartImageNet, frozen DINOv2-B/14 with registers, 224 × 224 inputs, a 32 × 32 assignment/decoder grid, N = 1024, K = 16, and a threeblock width-768 decoder. It follows the 150-epoch stage-1 schedule with batch size 16 in Table 12. Each alternative changes the named component, retaining the remaining settings. Quantization uses a separate, equally trained continuation control, described in Appendix E.3. Backbone and conceptbudget comparisons specify their own references below.

Reading the results. Unless stated otherwise, gray reference rows report absolute scores and colored entries report variant-minus-reference differences. Percentage-valued metrics are shown on a 0–100 scale, and their differences are in percentage points (pp). Backbone impurity and quantization loss use raw units. Attribute rates are absolute bits per selected concept in every row. The backbone figure’s final panel reports each model’s LP and kNN differences from its own frozen target, without subtracting the DINOv2 reference again. Blue indicates improvement and red degradation, with separate color scales whose ranges are shown below each panel.

The fusion, upsampling, selection, appearance-removal, auxiliary-readout, decoder-depth, gradient, objective, and quantization comparisons report means over seeds 42, 43, and 44. Smaller numbers give sample standard deviations across absolute reference scores or paired variant-minus-reference differences. The backbone, bank-size, and concept-budget sweeps report single runs. We interpret small differences descriptively and do not infer significance from mean rankings. LP denotes linearprobe accuracy, and mAP denotes full-ranking image-retrieval mAP. CFM locality and consistency in the component panels use energy-threshold maps. Metric definitions, evaluation procedures, and training schedules are given in Appendices D.6, D.5, and D.3, respectively.

## E.1 FEATURE EXTRACTION

Backbone and global target. Figure 8 evaluates the quantized model across frozen feature extractors. The sweep follows the canonical two-stage settings, with the backbone-specific feature, target, and grid adaptations described in Appendix D.2 and Table 9. Decoder width remains 768, including for CLIP-DINOiser’s 512-dimensional projected features. Boundary-sensitive scores should be compared within a grid.

DINOv2 retains LP accuracy within 0.1 pp of its frozen target and achieves the strongest foreground purity in its grid group. DINOv3 leads most part-quality measures in the patch-16 group, although CLIP-DINOiser has higher CFM consistency (−3.0 versus −3.7 pp relative to DINOv2). Thus the metric ordering is not universal. SigLIP2’s pooler improves LP by 2.7 pp over GAP (−2.4 versus −5.1 pp relative to DINOv2), with mixed part-quality changes. The target definition therefore matters as well as the patch features. The final panel separates representation retention from each frozen target’s own probe accuracy.

Layer fusion. Removing the shared-query fusion of Appendix C.1 uses the final-block patch features alone. Figure 9 shows essentially unchanged downstream performance (+0.17 pp LP and −0.05 pp mAP), but foreground purity falls by 3.16 pp and mIoU by 1.44 pp. Multi-layer fusion therefore contributes primarily to concept quality in this comparison, rather than increasing downstream accuracy.

Upsampling. Figure 10 uses continuous AnyUp training as its reference. Replacing image-guided upsampling with bilinear interpolation at the same 32 × 32 resolution changes LP by +0.22 pp and mAP by −0.03 pp. Bilinear improves foreground purity by 2.07 pp but lowers mIoU and locality by 1.39 and 1.47 pp. These mixed differences support its use as a cheaper component-ablation control without establishing equivalence on every metric. The canonical method retains AnyUp for its slightly more detailed qualitative maps.

Native resolution and mixed-route training. Keeping the native 16 × 16 grid removes upsampling and changes the decoder grid accordingly. Relative to AnyUp, LP changes by −0.04 pp, but foreground purity falls by 6.55 pp, while FG-ARI rises by 2.42 pp, illustrating sensitivity to spatial granularity. The mixed alternative selects AnyUp with probability 0.2 and bilinear interpolation otherwise, sharing the draw across both views of a local batch, and uses full AnyUp for evaluation. It remains close in LP and mAP (+0.08 and −0.05 pp) and improves foreground purity by 2.17 pp in Figure 10. Expensive image-guided upsampling on every training batch is therefore not required to retain the measured downstream performance.

![](images/c0f9d504273e531dfa71d358ad6dac725c7ced4d60c73e0e8df49549ce4d6e88.jpg)

Figure 8: DisParQ across backbones. Quantized models on PartImageNet validation across seven configurations of six frozen feature extractors, with $K = 1 6$ and $N = 1 0 2 4$ , using the canonical seed 42. In the first four panels, the gray DINOv2 row reports absolute scores and other rows report differences from that reference. Percentage-valued metrics use pp for differences, while impurity uses raw entropy units. The final panel reports each model’s LP and kNN gaps relative to its corresponding frozen global target, without subtracting the DINOv2 row again. Blue indicates improvement and red degradation. Targets and scoring grids are specified in Table 9. Patch-14 and patch-16 models use $3 2 \times 3 2$ and 28 × 28 grids, respectively, which can affect boundary-sensitive scores.
<table><tr><td rowspan="2"></td><td colspan="2">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons. CFM Loc.</td><td></td></tr><tr><td>Bilinear baseline</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>No layer fusion</td><td>+0.17 (±0.11)</td><td>-0.05 (±0.03)</td><td>-0.21 (±0.29)</td><td>-1.44 (±0.3i)</td><td>-3.16 (±0.10)</td><td>-1.45 (±0.71)</td><td>-0.96 (±0.52)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-10 -5 0</td><td>5 10</td><td>-8</td><td>-4</td><td>0</td><td>4</td><td>8</td></tr><tr><td></td><td>Representation: ±10 pp</td><td></td><td></td><td></td><td>Spatial and semantic metrics: ±8 pp</td><td></td><td></td></tr></table>

Figure 9: Layer fusion versus final-block features under the bilinear stage-1 control. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

![](images/5350984eca38c380d87fb7386f8a8845d165f117c04cf8fa573c1b911a3c4ef4.jpg)  
Figure 10: Feature upsampling with continuous AnyUp training as the reference. The 20% AnyUp arm mixes AnyUp and bilinear interpolation during training and uses full AnyUp at evaluation. Native resolution changes both the assignment and decoder grids to 16×16. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

## E.2 CONCEPT BANK, SELECTION, AND MAINTENANCE

Bank size. Figure 11 varies the global vocabulary N from 1024 to 8192 at fixed $K = 1 6 .$ The budget sweeps use continuous DINOv2-B/14 discovery with AnyUp on a $3 2 \times 3 2$ grid and the 150-epoch stage-1 schedule in Table 12, sharing the $\dot { N } = 1 0 2 4 , \dot { K } \dot { = } 1 6$ reference. Each setting is a single completed run. Increasing N to 8192 raises foreground purity by 4.05 pp, but FG-ARI and mIoU fall by 1.85 and 2.53 pp, while consistency and locality fall by 10.47 and 3.23 pp. LP increases by 0.41 pp. Thus higher purity with a larger vocabulary does not translate into better spatial segmentation or more consistent concepts across images. The canonical bank balances these competing properties. Interpret purity alongside consistency using the metric definitions in Appendix D.6, since finer fragmentation can increase purity.

<table><tr><td></td><td colspan="2">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td></td><td>LP (val.)</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td></td><td>CFM Cons. CFM Loc.</td></tr><tr><td>N = 1024 (default)</td><td>89.47</td><td>75.65</td><td>26.30</td><td>35.88</td><td>81.48</td><td>16.30</td><td>36.37</td></tr><tr><td>N = 2048</td><td>+0.17</td><td>-0.35</td><td>-0.49</td><td>-0.45</td><td>+2.57</td><td>-4.89</td><td>-1.23</td></tr><tr><td>N = 4096</td><td>-0.17</td><td>+0.11</td><td>-1.26</td><td>-1.31</td><td>+3.50</td><td>-8.81</td><td>-2.21</td></tr><tr><td>N = 8192</td><td>+0.41</td><td>+0.01</td><td>-1.85</td><td>-2.53</td><td>+4.05</td><td>-10.47</td><td>-3.23</td></tr><tr><td></td><td colspan="2">-1 -0.5 0 0.5 Representation: ±1 pp</td><td colspan="5">-8 0 8 Spatial and semantic metrics: ±16 pp</td></tr></table>

Figure 11: Concept-bank size N at fixed $K = 1 6$ during continuous DINOv2-B/14 discovery with AnyUp, using the sweep setup in Appendix E.2. The gray $N = 1 0 2 4$ row gives absolute reference scores, and colored entries give differences in pp. Each row is one completed run.

Concepts per image. Figure 12 varies K from 4 to 64 at fixed $N = 1 0 2 4 .$ . The K = 32 and K = 64 configurations also increase the equivariance-map budget to 64 and 128, respectively, so these comparisons vary both budgets. The treatment of unmatched parts in mIoU follows Appendix D.6. With K = 4, LP remains close to the reference (+0.50 pp), but FG-ARI and mIoU fall by 14.37 and 15.31 pp, and consistency and locality fall by 12.49 and 16.00 pp. Increasing K beyond 16 also fails to improve these concept metrics. At K = 32, FG-ARI, mIoU, and foreground purity fall by 1.83, 2.17, and 3.84 pp. The per-image budget controls the spatial decomposition, and downstream accuracy alone does not identify a useful concept budget. At N = 1024, K = 16 gives the strongest reported spatial and semantic scores among the tested values.

![](images/6dbb5d5301a59cef7722bce86cb66fbe836c823fd33f6addd9d331b1c06f7d8f.jpg)  
Figure 12: Selected concept count K at fixed N = 1024 during continuous DINOv2-B/14 discovery with AnyUp. The sweep setup and accompanying equivariance budgets are described in Appendix E.2. The gray $K = 1 6$ row gives absolute reference scores, and colored entries give differences in pp. Each row is one completed run.

Joint bank and concept-budget sweep. Figure 13 extends these two slices to all 20 combinations of $K \in \{ 4 , 8 , 1 6 , 3 2 , 6 \bar { 4 } \}$ and $\mathbf { \bar { \Delta } } N \in \{ 1 \bar { 0 } 2 4 , \bar { 2 } 0 4 8 , 4 0 9 6 , 8 1 9 2 \}$ . LP varies from 88.14% to 90.22%, while foreground purity and CFM consistency vary more strongly. The highest observed purity is 86.60% at $\mathbf { \bar { \Phi } } ( K , N ) = \mathbf { \bar { \Phi } } ( 3 2 , 8 1 9 2 )$ , whereas the default (16, 1024) has the highest observed consistency (16.30%). Increasing the bank size therefore does not uniformly improve concept quality, and the effect depends on the selected concept budget. These single-run comparisons are descriptive.

![](images/62c4dd1412c0e28a0529309c350ad718fc55745bfe7e0a1ccf25fc2d9ba418d0.jpg)  
Figure 13: Joint sweep of selected concept count $K$ and bank size $N$ on PartImageNet with DINOv2-B/14, using the continuous AnyUp setup and equivariance budgets in Appendix E.2. Heights give absolute LP validation accuracy, foreground purity, and CFM consistency in percent. Vertical ranges differ across panels. Both horizontal axes use logarithmic (base-2) spacing. Dots mark all 20 measured configurations and the diamond marks the default $\left( K , N \right) = \left( 1 6 , 1 0 2 4 \right)$ . Surfaces connect adjacent observations without smoothing. Red and blue indicate scores below and above the default, respectively, with color intensity scaled separately for each metric. Each configuration is a single completed run.

Peak rather than mean scoring. The canonical selector averages each full-bank map over space (Appendix C.2). The peak alternative instead takes its maximum, retaining Sinkhorn balancing and all other settings. Figure 14 shows small downstream changes $( + 0 . 0 6 \ \mathrm { p p } \ \mathrm { L P } , + 0 . 0 5 \ \mathrm { p p } \ \mathrm { m A P } )$ but a 3.45 pp decrease in foreground purity. Selecting concepts by their most active cell does not improve the overall region semantics in this setting.

Coverage bias and direct mass selection. Setting the coverage-bias coefficient to zero retains Sinkhorn balancing and changes foreground purity by $- 0 . 7 9 \pm 1 . 4 6 \mathrm { p p }$ , with little mean change in FG-ARI or mIoU (Figure 14). These small differences relative to seed variation do not establish an independent benefit of the coverage term. Appendix C.2 explains its cancellation under ideal balancing. To compare the selectors, direct top-K selection by mean activation is contrasted with the zero-bias Sinkhorn row. LP differs by +0.03 pp, while consistency and locality fall by 1.38 and 1.63 pp.

Prototype resampling. Disabling the inactive-prototype replacement of Appendix C.6, while retaining usage tracking and selection, lowers mAP by 7.51 pp, foreground purity by 13.43 pp, and consistency by 8.52 pp (Figure 14). These are substantially larger changes than those from the scoring alternatives. Prototype replacement supports both retrieval and concept quality.

<table><tr><td rowspan="2"></td><td colspan="2">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons.</td><td>CFM Loc.</td></tr><tr><td>Bilinear baseline</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>Peak selector scoring</td><td>+0.06 (±0.31)</td><td>+0.05 (±0.08)</td><td>+0.18 (±0.63)</td><td>-0.20 (±0.42)</td><td>-3.45 (±0.59)</td><td>-1.26 (±1.48)</td><td>-0.26 (±0.65)</td></tr><tr><td>Zero coverage bias</td><td>-0.19 (±0.31)</td><td>+0.04 (±0.15)</td><td>+0.01 (±0.66)</td><td>-0.03 (±0.57)</td><td>-0.79 (±1.46)</td><td>-0.10 (±1.35)</td><td>+0.05 (±1.13)</td></tr><tr><td>Mass top-K, zero bias</td><td>-0.17 (±0.21)</td><td>-0.01 (±0.03)</td><td>+0.06 (±0.45)</td><td>-0.31 (±0.24)</td><td>-0.06 (±0.54)</td><td>-1.47 (±0.72)</td><td>-1.58 (±1.09)</td></tr><tr><td>No bank resampling</td><td>-2.31 (±0.06)</td><td>-7.51 (±0.45)</td><td>-3.81 (±0.49)</td><td>-6.91 (±0.09)</td><td>-13.43 (±1.27)</td><td>-8.52 (±1.25)</td><td>-6.41 (±0.96)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-10 -5 0</td><td>5 10-14</td><td></td><td>-7</td><td>0</td><td>7</td><td></td></tr><tr><td></td><td></td><td>Representation: ±10 pp</td><td></td><td>Spatial and semantic metrics: ±14 pp</td><td></td><td></td><td>14</td></tr></table>

Figure 14: Selection rules and prototype maintenance under the bilinear stage-1 control. The zerobias Sinkhorn and zero-bias mass rows form the direct selector comparison. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

## E.3 APPEARANCE ATTRIBUTES AND QUANTIZATION

Attributes during concept discovery. Removing the continuous appearance contribution sets $q _ { i _ { k } } =$ 0 in the decoder of Appendix C.4, while learning concepts and retaining identity, geometry, and the auxiliary branch. Figure 15 shows losses of 8.98 pp LP and 6.22 pp mAP, although FG-ARI and mIoU rise slightly. Concept identities and geometry alone do not retain the same downstream information. This discovery-stage experiment differs from removing attributes after the concept bank has been learned, considered next.

<table><tr><td colspan="3">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td></td><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons. CFM Loc.</td><td></td></tr><tr><td>Bilinear baseline</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>No attributes</td><td>-8.98 (±1.62)</td><td>-6.22 (±1.23)</td><td>+0.53 (±1.68)</td><td>+0.66 (±0.49)</td><td>-2.35 (±0.70)</td><td>-0.50 (±1.85)</td><td>+1.35 (±1.42)</td></tr><tr><td></td><td>-10 -5 0</td><td>5 10</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Representation: ±10 pp</td><td>-8</td><td>-4</td><td>0 Spatial and semantic metrics: ±8 pp</td><td>4</td><td>8</td></tr></table>

Figure 15: Continuous appearance attributes during concept discovery under the bilinear stage-1 control. This differs from removing attributes after freezing concept assignment. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

Quantization protocol and rate. Figure 16 compares attribute encoders with the same seedmatched bilinear stage-1 parent. Each arm freezes concept assignment, retains the decoder, and receives the same 200-epoch continuation under the PartImageNet stage-2 schedule in Table 12. The continuous arm retains its learned projection, while the discrete arms initialize fresh quantization modules. The reference is private PQ with G = 16 groups and M = 64 entries per group, as defined in Appendix C.3. A private codebook is indexed by concept identity. A shared codebook serves all concepts. Rates count only attribute indices per selected concept, excluding identities, geometry, and model parameters. All discrete alternatives use nearest-codeword forward values and differentiable soft-mixture surrogates. Their codebooks and input projections are learned through distillation. These arms compare codebook structures, using our common training objective and surrogate rather than the original VQ-VAE or SoundStream training losses.

Continuous and absent attributes. Continuing with continuous attributes instead of PQ improves LP and kNN by 0.46 and 0.71 pp, while mAP falls by 0.31 pp (Figure 16). Both arms receive the same continuation, controlling for additional training time. Removing attributes from this frozenassignment bilinear control lowers LP by 12.24 pp and mAP by 11.14 pp relative to private PQ. Both comparisons support retaining appearance information.

Private low-rate quantizers. Vector quantization (VQ) represents a full vector by a single codeword (Linde et al., 1980; van den Oord et al., 2017). Private VQ64 selects one full-width vector from 64 entries (6 bits per concept). Private PQ2×8 (Jegou et al.´ , 2011) splits the projected residual into two channel groups with eight entries each, also using 6 bits. Relative to private PQ16×64, they lose 1.69 and 2.05 pp LP, respectively, and both increase distillation loss by approximately 0.05 (Figure 16). Very compact attributes retain much more information than removing attributes, but incur a larger cost than the 96-bit reference. These comparisons jointly change the attribute-index rate and quantizer structure and capacity, rather than isolating bit rate alone.

Shared 12-bit quantizers. Shared VQ4096 selects a single full-width vector. Residual vector quantization (RVQ), as used in SoundStream (Zeghidour et al., 2022), instead quantizes successive residuals. Shared RVQ64+64 sums a first 64-entry codeword and a second 64-entry codeword fitted to the remaining residual. Shared PQ2×64 concatenates two channel-group codewords (Jegou´ et al., 2011). Each uses 12 bits per concept. Their LP changes are −1.84, −2.23, and −2.41 pp, respectively, with mAP changes of −0.40, −0.89, and −0.94 pp (Figure 16). At this rate, full-vector VQ has the highest mean LP and mAP among these three alternatives, but all trail the higher-rate reference.

Shared versus private product codebooks. Shared PQ16×64 matches the reference’s 96-bit attribute-index rate while sharing its codebooks across concepts. It changes LP by −0.17 pp, mAP by +0.03 pp, and kNN by +0.07 pp (Figure 16). Thus private codebooks do not provide a uniform downstream advantage in this comparison. They allow concept-specific appearance dictionaries, at the parameter cost specified in Appendix C.4. Sharing is a competitive alternative when reducing codebook storage is important.

<table><tr><td rowspan=2 colspan=8>Representation          Global distillation RateLP↑      mAP↑     kNN↑       Cosine loss ↓   bits/conceptContinuous</td></tr><tr><td rowspan=1 colspan=1>+0.46</td><td rowspan=1 colspan=1>(-0315)</td><td rowspan=1 colspan=1>(+0.71)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Continuous</td></tr><tr><td rowspan=1 colspan=1>No attributes</td><td rowspan=1 colspan=1>-12.24(±0.89)</td><td rowspan=1 colspan=1>-11.14(±1.18)</td><td rowspan=1 colspan=1>-11.67(±1.31)</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>+0.1355(±0.0059)</td><td rowspan=1 colspan=2>0</td></tr><tr><td rowspan=1 colspan=1>Private VQ64</td><td rowspan=1 colspan=1>-1.69(±0.05)</td><td rowspan=1 colspan=1>-0.58(±0.24)</td><td rowspan=1 colspan=1>-1.85(±0.36)</td><td rowspan=1 colspan=1>+0.0484(±0.0003)</td><td rowspan=1 colspan=2>6</td></tr><tr><td rowspan=1 colspan=1>Private PQ2×8</td><td rowspan=1 colspan=1>-2.05(±0.34)</td><td rowspan=1 colspan=1>-0.65(±0.25)</td><td rowspan=1 colspan=1>-1.79(±0.29)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.0492(±0.0003)</td><td rowspan=1 colspan=2>6</td></tr><tr><td rowspan=1 colspan=1>Shared VQ4096</td><td rowspan=1 colspan=1>-1.84(±0.16)</td><td rowspan=1 colspan=1>-0.40(±0.22)</td><td rowspan=1 colspan=1>-1.16(±0.48)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.0463(±0.0003)</td><td rowspan=1 colspan=2>12</td></tr><tr><td rowspan=1 colspan=1>Shared RVQ64+64</td><td rowspan=1 colspan=1>-2.23(±0.25)</td><td rowspan=1 colspan=1>-0.89(±0.38)</td><td rowspan=1 colspan=1>-1.91(±1.00)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.0542(±0.0003)</td><td rowspan=1 colspan=2>12</td></tr><tr><td rowspan=1 colspan=1>Shared PQ2×64</td><td rowspan=1 colspan=1>-2.41(±0.40)</td><td rowspan=1 colspan=1>-0.94(±0.21)</td><td rowspan=1 colspan=1>-1.83(±0.80)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.0536(±0.0003)</td><td rowspan=1 colspan=2>12</td></tr><tr><td rowspan=1 colspan=1>Shared PQ16×64</td><td rowspan=1 colspan=1>-0.17(±0.08)</td><td rowspan=1 colspan=1>+0.03(±0.09)</td><td rowspan=1 colspan=1>+0.07(±0.32)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.0099(±0.0002)</td><td rowspan=1 colspan=2>96</td></tr><tr><td rowspan=1 colspan=1>Private PQ16×64</td><td rowspan=1 colspan=1>88.97(±0.16)</td><td rowspan=1 colspan=1>74.08(±0.02)</td><td rowspan=1 colspan=1>87.47(±0.17)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0755(±0.0002)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96</td></tr><tr><td rowspan=1 colspan=8>-13     -6.5      0      6.5      13-0.15   0   0.15Representation: ±13 pp          Cosine loss: ±0.15</td></tr></table>

Figure 16: Attribute quantizers after equal 200-epoch continuation from a seed-matched, frozen bilinear concept model. Entries are means over three seeds. The gray private PQ16×64 row gives absolute reference scores, and other score entries give paired differences, in pp for LP, mAP, and kNN. Sample SDs appear beneath scores, using paired-difference SDs outside the reference row. Cosine loss is the single-view evaluation cosine distance between the decoder output and frozen global target, in raw units. Rates are absolute attribute-index bits per selected concept in every row and exclude model parameters.

## E.4 CONCEPT DECODER

Pooling instead of spatial attention. The zero-block alternative mean-pools the spatial tokens $t _ { u }$ and applies the same normalization and output projection, omitting the CLS token and all attention blocks. Figure 17 shows losses of 1.26 pp LP and 2.06 pp mAP relative to the three-block decoder, with smaller concept-quality changes. This is the pooling-only alternative to Appendix C.4. The attention blocks improve integration of the spatial concept description.

Transformer depth. Figure 17 varies depth from zero to six blocks at fixed width 768. Moving from one to three blocks improves LP by 0.51 pp and mAP by 0.31 pp. Increasing from three to six roughly doubles decoder parameters (23.42M to 44.68M) for 0.11 pp higher mean scores in each downstream metric, while consistency falls by 2.07 pp. Three blocks offer a useful capacity trade-off. Deeper decoding does not uniformly improve concept quality.

Auxiliary-only readout. Removing the spatial decoder also removes its global loss and appearance branch. The pooled-region auxiliary head becomes the downstream representation, and checkpoints are selected by auxiliary validation distillation loss. Figure 18 shows −0.75 pp LP and +0.35 pp mAP, with relatively small concept-quality changes. This system comparison changes both training and readout. Comparing the same auxiliary readout in both systems instead gives approximately $0 . 0 0 \pm 0 . 8 2$ pp LP and −0.05 ± 0.15 pp mAP. Concepts can therefore be learned through auxiliary distillation and the assignment objectives alone. The auxiliary readout consumes continuous pooled region features. The stage-1 spatial decoder instead consumes concept identities, spatia assignments, and continuous appearance attributes, which are quantized in the final model.

<table><tr><td rowspan="2"></td><td colspan="2">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons.</td><td>CFM Loc.</td></tr><tr><td>0 blocks (2.17M)</td><td>-1.26 (±0.31)</td><td>-2.06 (±0.16)</td><td>+0.46 (±0.37)</td><td>+0.05 (±0.39)</td><td>-0.23 (±1.20)</td><td>-0.97 (±0.60)</td><td>+0.21 (±0.55)</td></tr><tr><td>1 block (9.25M)</td><td>-0.51 (±0.37)</td><td>-0.31 (±0.04)</td><td>-0.69 (±0.36)</td><td>-0.45 (±0.40)</td><td>-0.70 (±0.25)</td><td>-0.67 (±1.22)</td><td>-0.41 (±0.75)</td></tr><tr><td>2 blocks (16.34M)</td><td>-0.28 (±0.47)</td><td>-0.08 (±0.10)</td><td>-0.65 (±0.37)</td><td>-0.40 (±0.81)</td><td>-0.04 (±0.33)</td><td>-0.51 (±1.11)</td><td>-0.91 (±0.77)</td></tr><tr><td>3 blocks (23.42M)</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>4 blocks (30.51M)</td><td>+0.00 (±0.29)</td><td>+0.06 (±0.05)</td><td>-0.78 (±0.29)</td><td>-0.41 (±0.47)</td><td>-0.59 (±0.35)</td><td>-1.58 (±0.46)</td><td>-0.77 (±0.58)</td></tr><tr><td>6 blocks (44.68M)</td><td>+0.11 (±0.15)</td><td>+0.11 (±0.07)</td><td>-1.33 (±0.44)</td><td>-1.41 (±0.29)</td><td>-0.81 (±0.70)</td><td>-2.07 (±0.86)</td><td>-1.32 (±0.61)</td></tr><tr><td></td><td>-3 -1.5 0</td><td>1.5 3 Representation: ±3 pp</td><td>-8 Spatial and semantic metrics: ±8 pp</td><td>-4</td><td>0</td><td>4</td><td>8</td></tr></table>

Figure 17: Decoder depth at fixed width 768 under the bilinear stage-1 control. Zero blocks denotes mean pooling, and labels include decoder parameter counts. The gray three-block row gives absolute reference scores. Colored entries are differences in pp, with SDs as defined in Appendix E.
<table><tr><td rowspan="2"></td><td colspan="2">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons. CFM Loc.</td><td></td></tr><tr><td>Bilinear baseline</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>Decoder-free (auxiliary)</td><td>-0.75 (±0.23)</td><td>+0.35 (±0.07)</td><td>-0.41 (±0.31)</td><td>-0.15 (±0.77)</td><td>-0.44 (±1.28)</td><td>-0.77 (±0.99)</td><td>+0.00 (±0.98)</td></tr><tr><td></td><td>-10 -5 0</td><td>5 10</td><td>-8</td><td>-4</td><td>0</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4</td><td>8</td></tr><tr><td></td><td>Representation: ±10 pp</td><td></td><td></td><td></td><td>Spatial and semantic metrics: ±8 pp</td><td></td><td></td></tr></table>

Figure 18: Spatial concept decoder versus the auxiliary-only region readout under the bilinear stage-1 control. The auxiliary readout consumes continuous pooled region features. The stage-1 decoder consumes concept identities, spatial assignments, and continuous appearance attributes. The comparison changes both training and readout, as discussed in Appendix E.4. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

## E.5 GRADIENT ROUTES

Identity surrogate. Disabling Equation 13 preserves hard forward identities but removes the decoder’s gradient to soft assignments through identity embeddings. Other training objectives remain active. Figure 19 shows −0.28 pp LP, −0.79 pp FG-ARI, and −1.00 pp consistency. The surrogate has modestly higher mean scores under this control, alongside the other sources of concept-learning gradients.

Residual gradients. The canonical residual in Appendix C.3 is detached. Restoring its derivative lets global distillation update pooled features and matching prototypes through the appearance projection as well as through the identity surrogate. LP and mAP change by −0.08 and +0.04 pp, while FG-ARI, mIoU, and foreground purity fall by 1.10, 0.67, and 0.69 pp (Figure 19). These results do not motivate adding this gradient route to the canonical model.

![](images/6dcc76aaf7ccadf42c8c744369024b18be365211b60b7f81bb291ad91fbb013f.jpg)  
Figure 19: Gradient routes under the bilinear stage-1 control. The identity-surrogate removal preserves hard forward identities, while the residual-gradient arm enables derivatives through the appearance residual. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

## E.6 TRAINING OBJECTIVES

We study the contribution of each auxiliary or regularization objective by setting one non-global loss weight in Table 6 to zero during continuous training, while retaining the global decoder loss and all remaining weights. Figure 20 reports the resulting changes. The objectives and weights are defined in Appendix C.5. Across these ablations, several objectives affect region quality more strongly than downstream performance. When auxiliary distillation $( \mathcal { L } _ { \mathrm { a u x } } )$ is removed, LP and mAP change little, but foreground purity and locality fall by 1.38 and 1.51 pp. The auxiliary signal supports region quality even when the retained spatial decoder preserves downstream information. Removing feature-weighted Potts regularization $( \mathcal { L } _ { \mathrm { P o t t s } } )$ lowers LP by only 0.12 pp, but mIoU and consistency fall by 6.94 and 7.91 pp. Feature-aware spatial regularization chiefly supports concept organization rather than downstream accuracy. Removing ${ \mathcal L } _ { \mathrm { o r t h } }$ lowers LP by 0.11 pp and both mIoU and consistency by 1.25 pp. Its effects are modest relative to Potts, but generally favor retaining distinct modulated region features.

The assignment and cross-view objectives reveal trade-offs between foreground purity and spatial and grounding metrics. Without assignment entropy regularization $( \mathcal { L } _ { \mathrm { e n t } } )$ , FG-ARI and mIoU rise by 2.45 and 1.65 pp, and consistency and locality rise by 1.28 and 2.16 pp, while foreground purity falls by 0.87 pp. The regularizer favors purity in this comparison, while several spatial and grounding metrics favor its removal. Similarly, without affine equivariance $( \mathcal { L } _ { \mathrm { e q } } )$ , mAP is unchanged and FG-ARI rises by 0.34 pp, but foreground purity and consistency fall by 4.76 and 2.14 pp. Cross-view agreement supports semantic stability beyond the quality of a single-image partition. Removing selected-region presence $( \mathcal { L } _ { \mathrm { s l o t } } )$ raises FG-ARI by 1.28 pp and mIoU by 0.66 pp but lowers foreground purity by 2.70 pp. Local activation of selected regions favors purity, with mixed effects on partition metrics.

When usage diversity $( { \mathcal { L } } _ { \mathrm { d i v } } )$ is removed, LP changes by +0.01 pp, but foreground purity and locality fall by 7.04 and 4.82 pp. Regularizing full-bank activation therefore remains useful alongside balancing the subset-selection scores. In comparison, removing bank presence $( \mathcal { L } _ { \mathrm { b a n k } } )$ lowers foreground purity by 0.68 pp and locality by 0.19 pp.

<table><tr><td colspan="3">Representation</td><td colspan="2">Spatial structure</td><td colspan="3">Semantic quality</td></tr><tr><td></td><td>LP</td><td>mAP</td><td>FG-ARI</td><td>mIoU</td><td>FG-Purity</td><td>CFM Cons. CFM Loc.</td><td></td></tr><tr><td>Bilinear baseline</td><td>89.49 (±0.15)</td><td>73.79 (±0.05)</td><td>20.92 (±0.57)</td><td>33.21 (±0.22)</td><td>79.14 (±0.51)</td><td>15.06 (±0.82)</td><td>33.24 (±0.66)</td></tr><tr><td>No auxiliary distillation</td><td>-0.03 (±0.28)</td><td>+0.00 (±0.06)</td><td>-0.57 (±0.48)</td><td>-0.53 (±0.41)</td><td>-1.38 (±0.80)</td><td>-0.93 (±1.09)</td><td>-1.51 (±0.89)</td></tr><tr><td>No Potts</td><td>-0.12 (±0.52)</td><td>+0.03 (±0.15)</td><td>-3.35 (±0.91)</td><td>-6.94 (±0.25)</td><td>-6.96 (±0.99)</td><td>-7.91 (±0.81)</td><td>-6.85 (±0.78)</td></tr><tr><td>No assignment entropy</td><td>-0.28 (±0.35)</td><td>-0.03 (±0.10)</td><td>+2.45 (±1.28)</td><td>+1.65 (±1.10)</td><td>-0.87 (±0.12)</td><td>+1.28 (±0.28)</td><td>+2.16 (±1.29)</td></tr><tr><td>No orthogonality</td><td>-0.11 (±0.05)</td><td>-0.01 (±0.09)</td><td>-0.87 (±0.27)</td><td>-1.25 (±0.44)</td><td>-0.50 (±0.87)</td><td>-1.25 (±1.24)</td><td>-1.20 (±0.83)</td></tr><tr><td>No equivariance</td><td>-0.28 (±0.28)</td><td>+0.00 (±0.07)</td><td>+0.34 (±0.72)</td><td>-0.85 (±1.11)</td><td>-4.76 (±1.38)</td><td>-2.14 (±1.14)</td><td>-1.07 (±0.75)</td></tr><tr><td>No selected-region presence</td><td>-0.06 (±0.54)</td><td>+0.07 (±0.03)</td><td>+1.28 (±0.64)</td><td>+0.66 (±0.44)</td><td>-2.70 (±0.91)</td><td>-0.58 (±1.62)</td><td>+0.44 (±1.24)</td></tr><tr><td>No usage diversity</td><td>+0.01 (±0.31)</td><td>+0.11 (±0.05)</td><td>+0.55 (±0.94)</td><td>-3.60 (±0.46)</td><td>-7.04 (±1.25)</td><td>-3.50 (±0.90)</td><td>-4.82 (±0.50)</td></tr><tr><td>No bank presence</td><td>-0.12 (±0.11)</td><td>-0.03 (±0.07)</td><td>-0.16 (±0.65)</td><td>-0.16 (±0.20)</td><td>-0.68 (±1.65)</td><td>-0.69 (±0.66)</td><td>-0.19 (±0.59)</td></tr></table>

Figure 20: Individual auxiliary or regularization loss removals under the bilinear stage-1 control, retaining the global decoder loss. Objectives are defined in Appendix C.5. Gray entries are absolute reference scores, and colored entries are differences in pp, with SDs as defined in Appendix E.

## F ADDITIONAL RESULTS

This section gives the full versions of the main-paper tables, in their order (Sections F.1–F.6), explains how we visualize quantized attributes (Section F.7), and reports the user-study details and results (Section F.8).

## F.1 LARGE-SCALE CLASSIFICATION IN FULL

Table 15 lists every method of the comparison summarized in Figure 5 (left) of the main paper, with the same properties and the same ranking.

Figure 5 (left) keeps one supervised method and the three vision-language methods with the highest ImageNet accuracy, preferring methods that are also baselines elsewhere in the paper; Table 15 lists all of them. In the supervised block, InfoDisent and ProtoQuant train their heads on a frozen, label-supervised ViT-B/16. The other rows report a linear probe on the backbone or on the concept representation, except DCLIP, which scores classes by their descriptions (Menon & Vondrick, 2023). Following CFM (Wittenmayer et al., 2026), LF-CBM (Oikarinen et al., 2023), LaBo (Yang et al., 2023), CDM (Panousis et al., 2023), DCLIP, DN-CBM (Rao et al., 2024) and D-CBM (Prasse et al., 2025) learn global concepts, CF-CBM (Panousis et al., 2024) region-level ones, and InfoDisent (Struski et al., 2024), SALF-CBM (Benou & Riklin-Raviv, 2025), ProtoQuant (Janusz et al., 2026) and CFM spatial ones.

Table 15: Large-scale classification, all methods (linear probe, top-1). ✓ yes, (✓) partially, × no, as in Table 1; ∅: not applicable; —: not reported. Best and second best among self-supervised methods; gray rows are unranked references. Vision-language results follow Fig. 6 of CFM (Wittenmayer et al., 2026); supervised ViT-B/16 results are from Table 4 of ProtoQuant (Janusz et al., 2026), using the CLS backbone baseline. <sup>†</sup>DCLIP uses description-based scoring, not a linear probe.
<table><tr><td rowspan="2">Method</td><td colspan="2">Concept representation</td><td colspan="2">Top-1 accuracy</td></tr><tr><td>Spatial grounding</td><td>Discrete</td><td>bottleneck ImageNet-1k↑ Places ↑</td><td></td></tr><tr><td colspan="5">Fully supervised, ViT-B/16 (reference)</td></tr><tr><td>Backbone</td><td>∅</td><td>∅</td><td>81.1</td><td></td></tr><tr><td>InfoDisent (Struski et al., 2024)</td><td>(√)</td><td>×</td><td>79.2</td><td></td></tr><tr><td>ProtoQuant (Janusz et al., 2026)</td><td>(√)</td><td>(√)</td><td>80.1</td><td></td></tr><tr><td colspan="5">Vision-language self-supervised, CLIP ViT-B/16</td></tr><tr><td>Backbone</td><td>∅</td><td>∅</td><td>80.2</td><td>55.1</td></tr><tr><td>LF-CBM (Oikarinen et al., 2023)</td><td>× × × ×</td><td>X</td><td>75.4</td><td>50.6</td></tr><tr><td>LaBo (Yang et al., 2023)</td><td></td><td>X</td><td>78.9</td><td></td></tr><tr><td>CDM (Panousis et al., 2023)</td><td></td><td>(√)</td><td>79.3</td><td>52.6</td></tr><tr><td>DCLIP† (Menon &amp; Vondrick, 2023)</td><td></td><td>X</td><td>68.0</td><td>40.3</td></tr><tr><td>DN-CBM (Rao et al., 2024)</td><td>×</td><td>X</td><td>79.5</td><td>55.1</td></tr><tr><td>D-CBM (Prasse et al., 2025)</td><td>X</td><td>X</td><td>70.5</td><td>50.9</td></tr><tr><td>CF-CBM (Panousis et al., 2024)</td><td>(√)</td><td>(√)</td><td>78.5</td><td></td></tr><tr><td>SALF-CBM (Benou &amp; Riklin-Raviv, 2025)</td><td>(√)</td><td>X</td><td>76.3</td><td>49.4</td></tr><tr><td>CFM (Wittenmayer et al., 2026)</td><td>(√)</td><td>X</td><td>78.9</td><td>55.4</td></tr><tr><td colspan="5">Vision-only self-supervised, DINOv2-B/14</td></tr><tr><td>Backbone</td><td>∅</td><td>∅</td><td>83.4</td><td>55.4</td></tr><tr><td>DisParQ (ours)</td><td></td><td></td><td>83.2</td><td>55.3</td></tr></table>

Properties. The two property columns follow Table 1. Spatial grounding is full when every location carries exactly one concept, partial for region-level or overlapping concepts, and absent for global concepts. A discrete bottleneck is full when discrete indices alone define what the predictor reads, and partial when a discrete selection or codebook is combined with continuous scores, as in CDM, CF-CBM and ProtoQuant. DisParQ is the only method with both. The columns describe how each method is built, not how well it localizes.

## F.2 FULL PARTIMAGENET COMPARISON

We extend Table 2 by reporting every evaluated configuration under the same protocol, as differences to DisParQ on DINOv2-B/14. Figure 21 covers the supervised methods: all part counts K of PDiscoFormer, released and retrained, and PIP-Net and ProtoQuant with their classifier stage on every backbone we trained them on. Figure 22 covers the self-supervised methods: every concept budget of CFM including its own unbudgeted setting, the sparse-autoencoder baselines across dictionary sizes, budgets and backbones, and the label-free first stage of PIP-Net and ProtoQuant. Figure 23 shows the Matryoshka SAE across dictionary sizes, concept budgets, loss weightings and its own backbone. DisParQ on the other backbones in Figure 8, with the backbone study of Appendix E.1.

<table><tr><td colspan="16" rowspan="1">Downstream              Part quality               Grounding   ImpurityBackbone          K/N      LP↑ kNN↑  Purfg ↑NMIm ↑ ARIm ↑ mIoU↑ FG-ARI↑  Loc. ↑Cons. ↑   Imp.↓DisParQ (ours): absolute values. Every row below is that row minus this one.</td></tr><tr><td colspan="3" rowspan="1">DINOv2-B/14       16/1024</td><td colspan="2" rowspan="1">89.2  88.7</td><td></td><td colspan="5" rowspan="1">81.5  69.8  62.3  35.9  26.3</td><td></td><td colspan="2" rowspan="1">36.4  16.3</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">0.385</td></tr><tr><td colspan="1" rowspan="1">PDiscoFormer, re</td><td colspan="2" rowspan="1">leased checkpoints</td><td colspan="2" rowspan="1"></td><td></td><td colspan="5" rowspan="1"></td><td></td><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="2" rowspan="1">8/8</td><td colspan="1" rowspan="1">-0.9</td><td colspan="1" rowspan="1">-0.5</td><td></td><td colspan="1" rowspan="1">-65.6</td><td colspan="1" rowspan="1">-56.0</td><td colspan="1" rowspan="1">-57.5</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-3.0</td><td></td><td colspan="1" rowspan="1">-11.7</td><td colspan="1" rowspan="1">-14.6</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.163</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="2" rowspan="1">16/16</td><td colspan="1" rowspan="1">+0.4</td><td colspan="1" rowspan="1">-0.2</td><td></td><td colspan="1" rowspan="1">-41.7</td><td colspan="1" rowspan="1">-28.9</td><td colspan="1" rowspan="1">-37.2</td><td colspan="1" rowspan="1">-7.7</td><td colspan="1" rowspan="1">+4.0</td><td></td><td colspan="1" rowspan="1">-4.5</td><td colspan="1" rowspan="1">-14.3</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.156</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">25/25</td><td colspan="1" rowspan="1">+0.3</td><td colspan="1" rowspan="1">-1.2</td><td></td><td colspan="1" rowspan="1">-31.8</td><td colspan="1" rowspan="1">-26.8</td><td colspan="1" rowspan="1">-29.8</td><td colspan="1" rowspan="1">-7.8</td><td colspan="1" rowspan="1">+3.1</td><td></td><td colspan="1" rowspan="1">-2.7</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.114</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">50/50</td><td colspan="1" rowspan="1">+0.6</td><td colspan="1" rowspan="1">-0.6</td><td></td><td colspan="1" rowspan="1">-26.6</td><td colspan="1" rowspan="1">-21.4</td><td colspan="1" rowspan="1">-24.4</td><td colspan="1" rowspan="1">-6.8</td><td colspan="1" rowspan="1">+2.5</td><td></td><td colspan="1" rowspan="1">-1.5</td><td colspan="1" rowspan="1">-14.3</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.120</td></tr><tr><td colspan="1" rowspan="1">PDiscoFormer†, ou</td><td colspan="2" rowspan="1">r retraining of the authors&amp;</td><td colspan="1" rowspan="1">#x27; code</td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">8/8</td><td colspan="1" rowspan="1">-0.6</td><td colspan="1" rowspan="1">-0.6</td><td></td><td colspan="1" rowspan="1">-60.1</td><td colspan="1" rowspan="1">-56.6</td><td colspan="1" rowspan="1">-56.4</td><td colspan="1" rowspan="1">-9.8</td><td colspan="1" rowspan="1">-1.5</td><td></td><td colspan="1" rowspan="1">-7.3</td><td colspan="1" rowspan="1">-14.4</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.198</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/16</td><td colspan="1" rowspan="1">-1.0</td><td colspan="1" rowspan="1">-1.1</td><td></td><td colspan="1" rowspan="1">-50.6</td><td colspan="1" rowspan="1">-51.7</td><td colspan="1" rowspan="1">-53.9</td><td colspan="1" rowspan="1">-7.7</td><td colspan="1" rowspan="1">+0.7</td><td></td><td colspan="1" rowspan="1">-4.0</td><td colspan="1" rowspan="1">-14.3</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.178</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">25/25</td><td colspan="1" rowspan="1">-1.6</td><td colspan="1" rowspan="1">-1.2</td><td></td><td colspan="1" rowspan="1">-54.4</td><td colspan="1" rowspan="1">-52.7</td><td colspan="1" rowspan="1">-54.0</td><td colspan="1" rowspan="1">-6.8</td><td colspan="1" rowspan="1">-0.6</td><td></td><td colspan="1" rowspan="1">-2.8</td><td colspan="1" rowspan="1">-14.4</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.151</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">50/50</td><td colspan="1" rowspan="1">-0.6</td><td colspan="1" rowspan="1">-0.7</td><td></td><td colspan="1" rowspan="1">-46.0</td><td colspan="1" rowspan="1">-43.4</td><td colspan="1" rowspan="1">-49.0</td><td colspan="1" rowspan="1">-6.2</td><td colspan="1" rowspan="1">+5.7</td><td></td><td colspan="1" rowspan="1">-0.3</td><td colspan="1" rowspan="1">-14.0</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.075</td></tr><tr><td colspan="3" rowspan="1">PDiscoFormer†, with the position, CLS and register tokens frozen as well</td><td colspan="1" rowspan="1">d register</td><td colspan="1" rowspan="1">tokens fr</td><td></td><td colspan="1" rowspan="1">n as well</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="3" rowspan="1">DINOv2-B/14        25/25</td><td colspan="1" rowspan="1">-1.1</td><td colspan="1" rowspan="1">-1.4</td><td></td><td colspan="1" rowspan="1">-54.1</td><td colspan="1" rowspan="1">-57.3</td><td colspan="1" rowspan="1">-57.0</td><td colspan="1" rowspan="1">-8.7</td><td colspan="1" rowspan="1">-2.4</td><td></td><td colspan="1" rowspan="1">-4.0</td><td colspan="1" rowspan="1">-14.6</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.156</td></tr><tr><td colspan="3" rowspan="1">PIP-Net, stage 2 (self-supervised pretraining</td><td colspan="1" rowspan="1">, then its c</td><td colspan="1" rowspan="1">lassifier)</td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/768</td><td colspan="1" rowspan="1">-1.7</td><td colspan="1" rowspan="1">-1.9</td><td></td><td colspan="1" rowspan="1">-22.2</td><td colspan="1" rowspan="1">-32.5</td><td colspan="1" rowspan="1">-50.8</td><td colspan="1" rowspan="1">-11.0</td><td colspan="1" rowspan="1">-7.4</td><td></td><td colspan="1" rowspan="1">-11.2</td><td colspan="1" rowspan="1">-9.9</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.052</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-1.4</td><td colspan="1" rowspan="1">-1.6</td><td></td><td colspan="1" rowspan="1">-25.6</td><td colspan="1" rowspan="1">-30.9</td><td colspan="1" rowspan="1">-50.4</td><td colspan="1" rowspan="1">-10.6</td><td colspan="1" rowspan="1">-6.9</td><td></td><td colspan="1" rowspan="1">-9.8</td><td colspan="1" rowspan="1">-9.6</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.047</td></tr><tr><td colspan="1" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/384</td><td colspan="1" rowspan="1">-3.8</td><td colspan="1" rowspan="1">-5.4</td><td></td><td colspan="1" rowspan="1">-44.5</td><td colspan="1" rowspan="1">-42.4</td><td colspan="1" rowspan="1">-57.1</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1">-8.2</td><td></td><td colspan="1" rowspan="1">-11.2</td><td colspan="1" rowspan="1">-11.2</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">-0.012</td></tr><tr><td colspan="1" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-4.3</td><td colspan="1" rowspan="1">-4.1</td><td></td><td colspan="1" rowspan="1">-37.1</td><td colspan="1" rowspan="1">-37.7</td><td colspan="1" rowspan="1">-55.3</td><td colspan="1" rowspan="1">-12.9</td><td colspan="1" rowspan="1">-7.4</td><td></td><td colspan="1" rowspan="1">-9.4</td><td colspan="1" rowspan="1">-9.7</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">+0.020</td></tr><tr><td colspan="1" rowspan="1">ConvNeXt-T</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/768</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-12.8</td><td></td><td colspan="1" rowspan="1">-63.8</td><td colspan="1" rowspan="1">-63.8</td><td colspan="1" rowspan="1">-62.3</td><td colspan="1" rowspan="1">-25.0</td><td colspan="1" rowspan="1">-22.2</td><td></td><td colspan="1" rowspan="1">-21.3</td><td colspan="1" rowspan="1">-13.4</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">-0.087</td></tr><tr><td colspan="1" rowspan="1">ResNet-50</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-27.8</td><td colspan="1" rowspan="1">-69.0</td><td></td><td colspan="1" rowspan="1">-61.1</td><td colspan="1" rowspan="1">-66.6</td><td colspan="1" rowspan="1">-62.3</td><td colspan="1" rowspan="1">-19.9</td><td colspan="1" rowspan="1">-15.2</td><td></td><td colspan="1" rowspan="1">-21.3</td><td colspan="1" rowspan="1">-14.8</td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1">-0.108</td></tr><tr><td colspan="1" rowspan="1">ProtoQuant, st</td><td colspan="1" rowspan="1">ge 2 (code</td><td colspan="1" rowspan="1">oook, then its cl</td><td colspan="1" rowspan="1">ssifier), f</td><td colspan="1" rowspan="1">ozen bac</td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-1.3</td><td colspan="1" rowspan="1">+0.0</td><td></td><td colspan="1" rowspan="1">-11.3</td><td colspan="1" rowspan="1">-17.7</td><td colspan="1" rowspan="1">-38.8</td><td colspan="1" rowspan="1">-10.7</td><td colspan="1" rowspan="1">-8.2</td><td></td><td colspan="1" rowspan="1">-23.2</td><td colspan="1" rowspan="1">-10.8</td><td></td><td colspan="14" rowspan="1">+0.108</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-1.4</td><td colspan="1" rowspan="1">+0.8</td><td></td><td colspan="1" rowspan="1">-9.8</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1">-32.3</td><td colspan="1" rowspan="1">-9.4</td><td colspan="1" rowspan="1">-6.8</td><td></td><td colspan="1" rowspan="1">-23.3</td><td colspan="1" rowspan="1">-11.5</td><td></td><td colspan="14" rowspan="1">+0.128</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/4096</td><td colspan="1" rowspan="1">-1.6</td><td colspan="1" rowspan="1">+0.0</td><td></td><td colspan="1" rowspan="1">-7.8</td><td colspan="1" rowspan="1">-11.5</td><td colspan="1" rowspan="1">-27.5</td><td colspan="1" rowspan="1">-8.3</td><td colspan="1" rowspan="1">-6.1</td><td></td><td colspan="1" rowspan="1">-23.3</td><td colspan="1" rowspan="1">-12.3</td><td></td><td colspan="14" rowspan="1">+0.166</td></tr><tr><td colspan="1" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-4.1</td><td colspan="1" rowspan="1">-4.5</td><td></td><td colspan="1" rowspan="1">-8.8</td><td colspan="1" rowspan="1">-12.1</td><td colspan="1" rowspan="1">-27.2</td><td colspan="1" rowspan="1">-8.1</td><td colspan="1" rowspan="1">-5.2</td><td></td><td colspan="1" rowspan="1">-22.4</td><td colspan="1" rowspan="1">-10.9</td><td></td><td colspan="14" rowspan="1">+0.145</td></tr><tr><td colspan="2" rowspan="1">DeiT-S/16</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-2.1</td><td colspan="1" rowspan="1">-3.9</td><td></td><td colspan="1" rowspan="1">-16.6</td><td colspan="1" rowspan="1">-18.8</td><td colspan="1" rowspan="1">-38.3</td><td colspan="1" rowspan="1">-10.3</td><td colspan="1" rowspan="1">-10.3</td><td></td><td colspan="1" rowspan="1">-23.3</td><td colspan="1" rowspan="1">-11.6</td><td></td><td colspan="14" rowspan="1">+0.264</td></tr><tr><td colspan="2" rowspan="1">ProtoQuantft, stage 1, on</td><td colspan="1" rowspan="1"> backbone first</td><td colspan="1" rowspan="1">ne-tune</td><td colspan="1" rowspan="1">on the l</td><td></td><td colspan="1" rowspan="1">ls (its ow</td><td colspan="1" rowspan="1">protoco</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-0.1</td><td colspan="1" rowspan="1">+0.0</td><td></td><td colspan="1" rowspan="1">-20.3</td><td colspan="1" rowspan="1">-61.7</td><td colspan="1" rowspan="1">-62.3</td><td colspan="1" rowspan="1">-26.6</td><td colspan="1" rowspan="1">-21.7</td><td></td><td colspan="1" rowspan="1">-27.7</td><td colspan="1" rowspan="1">-13.5</td><td></td><td colspan="14" rowspan="3">+0.222+0.214+0.226</td></tr><tr><td colspan="2" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-3.5</td><td colspan="1" rowspan="1">-4.0</td><td></td><td colspan="1" rowspan="1">-14.8</td><td colspan="1" rowspan="1">-37.5</td><td colspan="1" rowspan="1">-59.8</td><td colspan="1" rowspan="1">-18.9</td><td colspan="1" rowspan="1">-14.2</td><td></td><td colspan="1" rowspan="1">-27.6</td><td colspan="1" rowspan="1">-12.6</td><td colspan="14"></td></tr><tr><td colspan="2" rowspan="1">DeiT-S/16</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">+0.8</td><td colspan="1" rowspan="1">+1.3</td><td></td><td colspan="1" rowspan="1">-18.2</td><td colspan="1" rowspan="1">-29.1</td><td colspan="1" rowspan="1">-55.6</td><td colspan="1" rowspan="1">-16.3</td><td colspan="1" rowspan="1">-15.1</td><td></td><td colspan="1" rowspan="1">-23.9</td><td colspan="1" rowspan="1">-11.6</td><td colspan="14"></td></tr><tr><td colspan="2" rowspan="1">ProtoQuantft, stage 2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-1.1</td><td colspan="1" rowspan="1">+0.3</td><td></td><td colspan="1" rowspan="1">-18.7</td><td colspan="1" rowspan="1">-49.4</td><td colspan="1" rowspan="1">-61.2</td><td colspan="1" rowspan="1">-22.6</td><td colspan="1" rowspan="1">-18.6</td><td></td><td colspan="1" rowspan="1">-27.3</td><td colspan="1" rowspan="1">-12.8</td><td></td><td colspan="14" rowspan="3">+0.197+0.204+0.211</td></tr><tr><td colspan="2" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-3.6</td><td colspan="1" rowspan="1">-2.0</td><td></td><td colspan="1" rowspan="1">-19.7</td><td colspan="1" rowspan="1">-58.1</td><td colspan="1" rowspan="1">-62.3</td><td colspan="1" rowspan="1">-24.6</td><td colspan="1" rowspan="1">-20.2</td><td></td><td colspan="1" rowspan="1">-27.5</td><td colspan="1" rowspan="1">-12.4</td><td colspan="14"></td></tr><tr><td colspan="2" rowspan="1">DeiT-S/16</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">+1.0</td><td colspan="1" rowspan="1">+1.8</td><td></td><td colspan="1" rowspan="1">-18.7</td><td colspan="1" rowspan="1">-35.8</td><td colspan="1" rowspan="1">-59.4</td><td colspan="1" rowspan="1">-17.4</td><td colspan="1" rowspan="1">-15.6</td><td></td><td colspan="1" rowspan="1">-26.6</td><td colspan="1" rowspan="1">-12.7</td><td colspan="14"></td></tr><tr><td colspan="2" rowspan="1">PCA, class-wise (the ground</td><td colspan="1" rowspan="1">-truth label sel</td><td colspan="1" rowspan="1">ects one</td><td colspan="1" rowspan="1">PCA per class)</td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="14" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">4/632</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">-26.3</td><td colspan="1" rowspan="1">-18.8</td><td colspan="1" rowspan="1">-30.8</td><td colspan="1" rowspan="1">-24.5</td><td colspan="1" rowspan="1">-17.9</td><td></td><td colspan="1" rowspan="1">-16.6</td><td colspan="1" rowspan="1">-12.6</td><td></td><td colspan="14" rowspan="2">+0.275+0.260</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">8/1264</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">-24.7</td><td colspan="1" rowspan="1">-17.3</td><td colspan="1" rowspan="1">-28.9</td><td colspan="1" rowspan="1">-22.9</td><td colspan="1" rowspan="1">-15.8</td><td></td><td colspan="1" rowspan="1">-15.0</td><td colspan="1" rowspan="1">-12.4</td><td colspan="14"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/2528</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">-23.6</td><td colspan="1" rowspan="1">-16.4</td><td colspan="1" rowspan="1">-27.9</td><td colspan="1" rowspan="1">-21.7</td><td colspan="1" rowspan="1">-14.1</td><td></td><td colspan="1" rowspan="1">-13.5</td><td colspan="1" rowspan="1">-12.0</td><td></td><td colspan="14" rowspan="1">+0.243</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">32/5056</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">-23.1</td><td colspan="1" rowspan="1">-16.0</td><td colspan="1" rowspan="1">-27.0</td><td colspan="1" rowspan="1">-21.5</td><td colspan="1" rowspan="1">-14.0</td><td></td><td colspan="1" rowspan="1">-13.3</td><td colspan="1" rowspan="1">-12.1</td><td></td><td colspan="14" rowspan="1">+0.240</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">64/10112</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">-23.0</td><td colspan="1" rowspan="1">-15.8</td><td colspan="1" rowspan="1">-26.9</td><td colspan="1" rowspan="1">-21.5</td><td colspan="1" rowspan="1">-14.0</td><td></td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1">-12.0</td><td></td><td colspan="14" rowspan="1">+0.241</td></tr><tr><td colspan="16" rowspan="1">±4.3 pp                 ±57.3 pp                ±23.3 pp     ±0.243</td></tr><tr><td colspan="16" rowspan="2">Downstream               Part quality                Grounding    ImpurityBackbone           K/N       LP↑  kNN↑   Purfg ↑NMIm ↑ ARIm ↑mIoU↑ FG-ARI↑  Loc. ↑ Cons. ↑   Imp.↓DisParQ (ours): absolute values. Every row below is that row minus this one.DINOv2-B/14        16/1024                                                          36.4  16.3    0.385</td></tr><tr><td colspan="1" rowspan="1">89.2</td><td colspan="2" rowspan="1"></td><td colspan="6" rowspan="1"></td><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">0.385</td></tr><tr><td colspan="3" rowspan="1">PCA, per image</td><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="1"></td><td colspan="6" rowspan="1"></td><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="3" rowspan="1">DINOv2-B/14         16/16</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="3" rowspan="1">-66.6  -69.3  -62.1</td><td colspan="1" rowspan="1">-22.3</td><td colspan="1" rowspan="1">-15.7</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-17.1</td><td colspan="1" rowspan="1">-15.0</td><td></td><td colspan="1" rowspan="1">-0.056</td></tr><tr><td colspan="3" rowspan="1">CFM, released 8192-latent dictionary</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="3" rowspan="1">CLIP-DI-B/16        16/8192</td><td colspan="1" rowspan="1">-16.6</td><td colspan="1" rowspan="1">-19.5</td><td></td><td colspan="1" rowspan="1">-26.8</td><td colspan="1" rowspan="1">-18.4</td><td colspan="1" rowspan="1">-37.8</td><td colspan="1" rowspan="1">-13.2</td><td colspan="1" rowspan="1">-13.8</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-13.5</td><td colspan="1" rowspan="1">-3.9</td><td></td><td colspan="1" rowspan="1">-0.007</td></tr><tr><td colspan="3" rowspan="1">CLIP-DI-B/16        32/8192</td><td colspan="1" rowspan="1">-16.3</td><td colspan="1" rowspan="1">-19.5</td><td></td><td colspan="1" rowspan="1">-26.4</td><td colspan="1" rowspan="1">-17.5</td><td colspan="1" rowspan="1">-36.4</td><td colspan="1" rowspan="1">-12.1</td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-6.5</td><td colspan="1" rowspan="1">-2.8</td><td></td><td colspan="1" rowspan="1">+0.016</td></tr><tr><td colspan="2" rowspan="1">CLIP-DI-B/16</td><td colspan="1" rowspan="1">64/8192</td><td colspan="1" rowspan="1">-17.3</td><td colspan="1" rowspan="1">-19.5</td><td></td><td colspan="1" rowspan="1">-26.4</td><td colspan="1" rowspan="1">-17.5</td><td colspan="1" rowspan="1">-36.4</td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">+2.4</td><td colspan="1" rowspan="1">-2.1</td><td></td><td colspan="1" rowspan="1">+0.010</td></tr><tr><td colspan="2" rowspan="1">CLIP-DI-B/16</td><td colspan="1" rowspan="1">128/8192</td><td colspan="1" rowspan="1">-15.7</td><td colspan="1" rowspan="1">-19.5</td><td></td><td colspan="1" rowspan="1">-26.4</td><td colspan="1" rowspan="1">-17.4</td><td colspan="1" rowspan="1">-36.2</td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">+5.3</td><td colspan="1" rowspan="1">-2.4</td><td></td><td colspan="1" rowspan="1">-0.020</td></tr><tr><td colspan="3" rowspan="1">CLIP-DI-B/16        ~85/8192</td><td colspan="1" rowspan="1">-16.0</td><td colspan="1" rowspan="1">-19.5</td><td></td><td colspan="1" rowspan="1">-26.4</td><td colspan="1" rowspan="1">-17.4</td><td colspan="1" rowspan="1">-36.2</td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">+5.3</td><td colspan="1" rowspan="1">-2.3</td><td></td><td colspan="1" rowspan="1">-0.025</td></tr><tr><td colspan="3" rowspan="1">TopK SAE on DINOv2-B/14: dictionary size, then concepts per image</td><td colspan="1" rowspan="1">e, then co</td><td colspan="1" rowspan="1">ncepts pe</td><td></td><td colspan="1" rowspan="1">nage</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1">-15.7</td><td></td><td colspan="1" rowspan="1">-13.3</td><td colspan="1" rowspan="1">-14.3</td><td colspan="1" rowspan="1">-31.0</td><td colspan="1" rowspan="1">-11.7</td><td colspan="1" rowspan="1">-12.0</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-16.0</td><td colspan="1" rowspan="1">-6.8</td><td></td><td colspan="1" rowspan="1">+0.063</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-8.7</td><td colspan="1" rowspan="1">-15.2</td><td></td><td colspan="1" rowspan="1">-12.7</td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-26.3</td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1">-10.2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-15.8</td><td colspan="1" rowspan="1">-6.5</td><td></td><td colspan="1" rowspan="1">+0.077</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/4096</td><td colspan="1" rowspan="1">-8.2</td><td colspan="1" rowspan="1">-15.2</td><td></td><td colspan="1" rowspan="1">-12.4</td><td colspan="1" rowspan="1">-13.8</td><td colspan="1" rowspan="1">-30.4</td><td colspan="1" rowspan="1">-11.0</td><td colspan="1" rowspan="1">-10.7</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-15.8</td><td colspan="1" rowspan="1">-7.5</td><td></td><td colspan="1" rowspan="1">+0.074</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/8192</td><td colspan="1" rowspan="1">-9.3</td><td colspan="1" rowspan="1">-15.0</td><td></td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-15.1</td><td colspan="1" rowspan="1">-33.4</td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1">-10.4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-15.9</td><td colspan="1" rowspan="1">-7.5</td><td></td><td colspan="1" rowspan="1">+0.075</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">32/1024</td><td colspan="1" rowspan="1">-7.4</td><td colspan="1" rowspan="1">-14.4</td><td></td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1">-13.3</td><td colspan="1" rowspan="1">-31.0</td><td colspan="1" rowspan="1">-10.0</td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-12.7</td><td colspan="1" rowspan="1">-4.1</td><td></td><td colspan="1" rowspan="1">+0.075</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">64/1024</td><td colspan="1" rowspan="1">-8.0</td><td colspan="1" rowspan="1">-13.1</td><td></td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-14.4</td><td colspan="1" rowspan="1">-32.3</td><td colspan="1" rowspan="1">-10.2</td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-9.0</td><td colspan="1" rowspan="1">-2.7</td><td></td><td colspan="1" rowspan="1">+0.078</td></tr><tr><td colspan="3" rowspan="1">DINOv2-B/14        128/1024</td><td colspan="1" rowspan="1">-5.9</td><td colspan="1" rowspan="1">-13.3</td><td></td><td colspan="1" rowspan="1">-12.6</td><td colspan="1" rowspan="1">-16.7</td><td colspan="1" rowspan="1">-36.3</td><td colspan="1" rowspan="1">-11.5</td><td colspan="1" rowspan="1">-12.4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-5.1</td><td colspan="1" rowspan="1">-5.4</td><td></td><td colspan="1" rowspan="1">+0.075</td></tr><tr><td colspan="3" rowspan="1">TopK SAE on the other backbones</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv3-B/16</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-13.6</td><td colspan="1" rowspan="1">-22.7</td><td></td><td colspan="1" rowspan="1">-11.2</td><td colspan="1" rowspan="1">-3.6</td><td colspan="1" rowspan="1">-10.4</td><td colspan="1" rowspan="1">-8.7</td><td colspan="1" rowspan="1">-8.4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-14.6</td><td colspan="1" rowspan="1">-5.9</td><td></td><td colspan="1" rowspan="1">+0.112</td></tr><tr><td colspan="2" rowspan="1">TIPSv2-B/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-22.2</td><td></td><td colspan="1" rowspan="1">-14.1</td><td colspan="1" rowspan="1">-9.1</td><td colspan="1" rowspan="1">-21.1</td><td colspan="1" rowspan="1">-10.2</td><td colspan="1" rowspan="1">-11.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-15.4</td><td colspan="1" rowspan="1">-5.7</td><td></td><td colspan="1" rowspan="1">+0.063</td></tr><tr><td colspan="2" rowspan="1">CLIP-B/16</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-23.3</td><td colspan="1" rowspan="1">-36.0</td><td></td><td colspan="1" rowspan="1">-19.1</td><td colspan="1" rowspan="1">-15.6</td><td colspan="1" rowspan="1">-31.6</td><td colspan="1" rowspan="1">-13.0</td><td colspan="1" rowspan="1">-14.3</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-17.6</td><td colspan="1" rowspan="1">-6.0</td><td></td><td colspan="1" rowspan="1">+0.102</td></tr><tr><td colspan="2" rowspan="1">SigLIP2-B/16</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-20.6</td><td></td><td colspan="1" rowspan="1">-23.2</td><td colspan="1" rowspan="1">-31.7</td><td colspan="1" rowspan="1">-52.0</td><td colspan="1" rowspan="1">-15.2</td><td colspan="1" rowspan="1">-13.5</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-20.5</td><td colspan="1" rowspan="1">-7.9</td><td></td><td colspan="1" rowspan="1">+0.199</td></tr><tr><td colspan="2" rowspan="1">CLIP-DI-B/16</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-21.1</td><td colspan="1" rowspan="1">-36.7</td><td></td><td colspan="1" rowspan="1">-18.9</td><td colspan="1" rowspan="1">-6.7</td><td colspan="1" rowspan="1">-15.4</td><td colspan="1" rowspan="1">-13.2</td><td colspan="1" rowspan="1">-13.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-14.5</td><td colspan="1" rowspan="1">-6.5</td><td></td><td colspan="1" rowspan="1">+0.065</td></tr><tr><td colspan="2" rowspan="1">PatchSAE (authors' code</td><td colspan="1" rowspan="1">nd hyperparam</td><td colspan="1" rowspan="1">ters, our</td><td colspan="1" rowspan="1">features):</td><td></td><td colspan="1" rowspan="1">ctionary</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-6.3</td><td colspan="1" rowspan="1">-12.2</td><td></td><td colspan="1" rowspan="1">-15.8</td><td colspan="1" rowspan="1">-18.9</td><td colspan="1" rowspan="1">-41.5</td><td colspan="1" rowspan="1">-15.6</td><td colspan="1" rowspan="1">-15.3</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-14.9</td><td colspan="1" rowspan="1">-7.3</td><td></td><td colspan="1" rowspan="1">-0.009</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-6.4</td><td colspan="1" rowspan="1">-12.2</td><td></td><td colspan="1" rowspan="1">-13.0</td><td colspan="1" rowspan="1">-16.3</td><td colspan="1" rowspan="1">-39.4</td><td colspan="1" rowspan="1">-14.4</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-13.8</td><td colspan="1" rowspan="1">-8.7</td><td></td><td colspan="1" rowspan="1">-0.042</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/4096</td><td colspan="1" rowspan="1">-5.7</td><td colspan="1" rowspan="1">-11.7</td><td></td><td colspan="1" rowspan="1">-11.8</td><td colspan="1" rowspan="1">-15.9</td><td colspan="1" rowspan="1">-37.7</td><td colspan="1" rowspan="1">-13.6</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-13.5</td><td colspan="1" rowspan="1">-10.2</td><td></td><td colspan="1" rowspan="1">-0.052</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/8192</td><td colspan="1" rowspan="1">-5.2</td><td colspan="1" rowspan="1">-11.4</td><td></td><td colspan="1" rowspan="1">-9.8</td><td colspan="1" rowspan="1">-13.5</td><td colspan="1" rowspan="1">-34.5</td><td colspan="1" rowspan="1">-12.6</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-12.7</td><td colspan="1" rowspan="1">-11.2</td><td></td><td colspan="1" rowspan="1">-0.027</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/49152</td><td colspan="1" rowspan="1">-5.1</td><td colspan="1" rowspan="1">-11.5</td><td></td><td colspan="1" rowspan="1">-5.3</td><td colspan="1" rowspan="1">-9.7</td><td colspan="1" rowspan="1">-27.4</td><td colspan="1" rowspan="1">-10.9</td><td colspan="1" rowspan="1">-9.3</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-11.7</td><td colspan="1" rowspan="1">-11.9</td><td></td><td colspan="1" rowspan="1">+0.070</td></tr><tr><td colspan="1" rowspan="1">MSAE (authors</td><td colspan="1" rowspan="1">code and</td><td colspan="1" rowspan="1">hyperparameter</td><td colspan="1" rowspan="1">, our fea</td><td colspan="1" rowspan="1">ures), def</td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/6144</td><td colspan="1" rowspan="1">-4.0</td><td colspan="1" rowspan="1">-9.7</td><td></td><td colspan="1" rowspan="1">-22.1</td><td colspan="1" rowspan="1">-21.4</td><td colspan="1" rowspan="1">-42.4</td><td colspan="1" rowspan="1">-16.0</td><td colspan="1" rowspan="1">-15.1</td><td></td><td colspan="1" rowspan="1">-18.8</td><td colspan="1" rowspan="1">-8.6</td><td></td><td colspan="1" rowspan="1">+0.085</td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/6144</td><td colspan="1" rowspan="1">-3.8</td><td colspan="1" rowspan="1">-9.3</td><td></td><td colspan="1" rowspan="1">-23.0</td><td colspan="1" rowspan="1">-26.8</td><td colspan="1" rowspan="1">-50.0</td><td colspan="1" rowspan="1">-16.6</td><td colspan="1" rowspan="1">-16.0</td><td></td><td colspan="1" rowspan="1">-19.9</td><td colspan="1" rowspan="1">-8.8</td><td></td><td colspan="1" rowspan="1">+0.056</td></tr><tr><td colspan="1" rowspan="1">PIP-Net, stage 1 (s</td><td colspan="1" rowspan="1">elf-superv</td><td colspan="1" rowspan="1">ised pretrainin</td><td colspan="1" rowspan="1">g only)</td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/768</td><td colspan="1" rowspan="1">-12.3</td><td colspan="1" rowspan="1">-17.2</td><td></td><td colspan="1" rowspan="1">-13.0</td><td colspan="1" rowspan="1">-15.6</td><td colspan="1" rowspan="1">-28.0</td><td colspan="1" rowspan="1">-6.9</td><td colspan="1" rowspan="1">-7.2</td><td></td><td colspan="1" rowspan="1">-5.9</td><td colspan="1" rowspan="1">-6.1</td><td></td><td colspan="1" rowspan="1">-0.009</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-8.2</td><td colspan="1" rowspan="1">-16.1</td><td></td><td colspan="1" rowspan="1">-13.3</td><td colspan="1" rowspan="1">-14.2</td><td colspan="1" rowspan="1">-24.4</td><td colspan="1" rowspan="1">-5.2</td><td colspan="1" rowspan="1">-4.8</td><td></td><td colspan="1" rowspan="1">-3.6</td><td colspan="1" rowspan="1">-4.8</td><td></td><td colspan="1" rowspan="1">-0.009</td></tr><tr><td colspan="2" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1">16/384</td><td colspan="1" rowspan="1">-22.5</td><td colspan="1" rowspan="1">-32.0</td><td></td><td colspan="1" rowspan="1">-25.5</td><td colspan="1" rowspan="1">-26.1</td><td colspan="1" rowspan="1">-49.1</td><td colspan="1" rowspan="1">-12.2</td><td colspan="1" rowspan="1">-12.1</td><td></td><td colspan="1" rowspan="1">-8.3</td><td colspan="1" rowspan="1">-6.5</td><td></td><td colspan="1" rowspan="1">-0.035</td></tr><tr><td colspan="2" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-15.3</td><td colspan="1" rowspan="1">-22.8</td><td></td><td colspan="1" rowspan="1">-18.5</td><td colspan="1" rowspan="1">-16.6</td><td colspan="1" rowspan="1">-28.6</td><td colspan="1" rowspan="1">-7.8</td><td colspan="1" rowspan="1">-7.2</td><td></td><td colspan="1" rowspan="1">-5.1</td><td colspan="1" rowspan="1">-4.5</td><td></td><td colspan="1" rowspan="1">-0.011</td></tr><tr><td colspan="2" rowspan="1">ConvNeXt-T</td><td colspan="1" rowspan="1">16/768</td><td colspan="1" rowspan="1">-71.7</td><td colspan="1" rowspan="1">-85.5</td><td></td><td colspan="1" rowspan="1">-65.7</td><td colspan="1" rowspan="1">-69.6</td><td colspan="1" rowspan="1">-62.3</td><td colspan="1" rowspan="1">-28.0</td><td colspan="1" rowspan="1">-24.8</td><td></td><td colspan="1" rowspan="1">-23.1</td><td colspan="1" rowspan="1">-15.3</td><td></td><td colspan="1" rowspan="1">-0.170</td></tr><tr><td colspan="2" rowspan="1">ResNet-50</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-73.1</td><td colspan="1" rowspan="1">-81.2</td><td></td><td colspan="1" rowspan="1">-54.5</td><td colspan="1" rowspan="1">-63.6</td><td colspan="1" rowspan="1">-62.4</td><td colspan="1" rowspan="1">-18.6</td><td colspan="1" rowspan="1">-13.1</td><td></td><td colspan="1" rowspan="1">-20.8</td><td colspan="1" rowspan="1">-14.6</td><td></td><td colspan="1" rowspan="1">+0.252</td></tr><tr><td colspan="2" rowspan="1">ProtoQuant, stage 1 (codeb</td><td colspan="1" rowspan="1">ook only), froze</td><td colspan="1" rowspan="1">n backbone</td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/1024</td><td colspan="1" rowspan="1">-5.4</td><td colspan="1" rowspan="1">-8.4</td><td></td><td colspan="1" rowspan="1">-2.3</td><td colspan="1" rowspan="1">-3.2</td><td colspan="1" rowspan="1">-9.0</td><td colspan="1" rowspan="1">-3.3</td><td colspan="1" rowspan="1">+1.6</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-18.1</td><td colspan="1" rowspan="1">-8.0</td><td></td><td colspan="1" rowspan="1">+0.102</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-4.0</td><td colspan="1" rowspan="1">-4.8</td><td></td><td colspan="1" rowspan="1">+0.6</td><td colspan="1" rowspan="1">-1.1</td><td colspan="1" rowspan="1">-6.0</td><td colspan="1" rowspan="1">-2.2</td><td colspan="1" rowspan="1">+2.1</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-18.3</td><td colspan="1" rowspan="1">-9.0</td><td></td><td colspan="1" rowspan="1">+0.129</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">16/4096</td><td colspan="1" rowspan="1">-3.5</td><td colspan="1" rowspan="1">-4.0</td><td></td><td colspan="1" rowspan="1">+2.9</td><td colspan="1" rowspan="1">+0.3</td><td colspan="1" rowspan="1">-5.4</td><td colspan="1" rowspan="1">-2.7</td><td colspan="1" rowspan="1">+0.8</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-19.2</td><td colspan="1" rowspan="1">-10.7</td><td></td><td colspan="1" rowspan="1">+0.174</td></tr><tr><td colspan="1" rowspan="1">DINOv2-S/14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-8.8</td><td colspan="1" rowspan="1">-12.8</td><td></td><td colspan="1" rowspan="1">+0.8</td><td colspan="1" rowspan="1">+0.8</td><td colspan="1" rowspan="1">-2.0</td><td colspan="1" rowspan="1">-1.9</td><td colspan="1" rowspan="1">+2.5</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-17.5</td><td colspan="1" rowspan="1">-8.6</td><td></td><td colspan="1" rowspan="1">+0.111</td></tr><tr><td colspan="1" rowspan="1">DeiT-S/16</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16/2048</td><td colspan="1" rowspan="1">-5.1</td><td colspan="1" rowspan="1">-8.8</td><td></td><td colspan="1" rowspan="1">-12.0</td><td colspan="1" rowspan="1">-23.3</td><td colspan="1" rowspan="1">-49.6</td><td colspan="1" rowspan="1">-12.6</td><td colspan="1" rowspan="1">-8.8</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-23.6</td><td colspan="1" rowspan="1">-11.9</td><td></td><td colspan="1" rowspan="1">+0.282</td></tr><tr><td colspan="2" rowspan="1">Frozen backbone without </td><td colspan="1" rowspan="1">ny bottleneck (ti</td><td colspan="1" rowspan="1">e ceiling</td><td colspan="1" rowspan="1">for LP a</td><td></td><td colspan="1" rowspan="1">kNN)</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">CLS token</td><td colspan="1" rowspan="1">+0.1</td><td colspan="1" rowspan="1">+0.2</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td></tr><tr><td colspan="2" rowspan="1">DINOv2-B/14</td><td colspan="1" rowspan="1">patch mean</td><td colspan="1" rowspan="1">-3.5</td><td colspan="1" rowspan="1">-9.4</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td></tr><tr><td colspan="2" rowspan="1">CLIP-DI-B/16</td><td colspan="1" rowspan="1">CLS token</td><td colspan="1" rowspan="1">-5.9</td><td colspan="1" rowspan="1">-7.5</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td></tr><tr><td colspan="2" rowspan="1">CLIP-DI-B/16</td><td colspan="1" rowspan="1">patch mean</td><td colspan="1" rowspan="1">-15.4</td><td colspan="1" rowspan="1">-30.9</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td><td colspan="1" rowspan="1">∅</td><td></td><td colspan="1" rowspan="1">∅</td></tr><tr><td colspan="16" rowspan="1">±23.3 pp                  ±36.4 pp                  ±18.1 pp     ±0.170</td></tr></table>

Figure 21: PartImageNet: supervised methods against DisParQ. The gray row gives DisParQ on DINOv2-B/14 in absolute numbers (protocol and columns of Table 2); every other row is that method minus DisParQ, in percentage points (raw units for Imp.). Blue: better than DisParQ; red: worse, so for Imp. (↓) a negative difference is blue. Color scales are per panel and clipped at the 90th percentile of |∆|; ∅ is undefined. Every method here trains with class labels. <sup>†</sup> is our retraining of the authors’ code. ConvNeXt-T and ResNet-50 are PIP-Net’s original CNNs under its published schedule. ProtoQuant<sup>ft</sup> first fine-tunes the backbone on the labels, which is why its stage 1 is listed here. For PCA, $\bar { K } / N$ is components per class over components in total.

Figure 22: PartImageNet: self-supervised methods against DisParQ. Reading as in Figure 21. ∼85/8192 is CFM without a per-image budget, its own setting (every firing concept is kept). TopK SAE, PatchSAE and MSAE are trained on the same frozen patch tokens as DisParQ; PatchSAE and MSAE run their authors’ code and hyperparameters (49152 and 6144 are their published widths; MSAE’s other settings are in Figure 23) and, like CFM, receive the 16-concept budget only at scoring time. LP/kNN read what each method hands downstream: the pooled reconstruction for the SAEs, CFM’s [max |mean]-pooled concept activations. The last block gives the frozen backbone itself: its CLS token, which DisParQ predicts, and its mean patch token, the ceiling for a pooledreconstruction probe.

![](images/bbebb0f821f0a131ec5b9d831ba8197913c18a16faee53e9192b7f619f29324a.jpg)  
Figure 23: PartImageNet: the Matryoshka SAE across its settings, against DisParQ. Reading as in Figure 21. The authors’ code and hyperparameters, trained on the frozen patch tokens. 6144, 12288 and 24576 latents are its published expansion factors 8, 16 and 32 on the 768-d DINOv2-B/14 tokens (8192 is expansion 8 on CLIP ViT-L/14, the model it was published on); 1024 to 8192 are the sizes the other sparse autoencoders were swept over. The per-image concept budget is imposed at scoring time, so the 32-, 64- and 128-concept rows read the dictionary of the 16/6144 row.

## F.3 FULL CUB CONCEPT METRICS

Table 16 lists every concept metric we measure on CUB. Since CUB marks parts with keypoints, the clustering metrics are computed on the cells that hold a visible keypoint, and mIoU, FG-ARI and the grounding metrics are undefined (Appendix D.6). DisParQ leads the self-supervised methods on every column, by 3.5 to 8.5 points on the clustering metrics and by 17.2 on PIP purity. The supervised PDiscoFormer, trained with class labels and evaluated at $5 1 8 \times 5 1 8 ,$ , scores higher on the clustering metrics for $K \geq 8 .$ . Its CUB checkpoints read $5 1 8 \times 5 1 8$ images, so we resample their $3 7 \times 3 { \bar { 7 } }$ part maps to the $3 2 \times 3 2$ grid and scale the PIP purity window from 32 to 74 pixels (Appendix D.6). It also has a far lower keypoint error at every K (4.44, 2.93 and 2.87 for $K { = } 4 .$ 8 and 16, against 9.36 for DisParQ), since its background channel concentrates its parts on the bird and it reads $5 1 8 \times 5 1 8$ images. Its PIP purity is 64.3 at K=4 and 8 and 40.4 at $K { = } 1 6 ,$ against 54.5 for DisParQ with 16 concepts.

Table 16: All concept metrics on CUB-200-2011 (test split, frozen DINOv2-B/14; CFM: CLIP-DINOiser; $\mathrm { M S A E _ { C L I P - L / 1 4 } } \colon$ MSAE on the model it was published on; PDiscoFormer: the released checkpoints, at 518×518); ×100 except Kp NME. CUB is annotated with keypoints, not part masks, so mIoU, FG-ARI and the grounding triple of Table 2 are undefined. Marks, subscripts and gray rows as in Table 3; ∅: undefined for that method, —: not measured.
<table><tr><td rowspan="2">Method</td><td rowspan="2">K/N</td><td colspan="2">Downstream</td><td colspan="5">Concept quality</td></tr><tr><td>LP↑</td><td>kNN↑</td><td> $\mathrm { P u r } _ { \mathrm { f g } } \uparrow$ </td><td>NMIm↑</td><td></td><td>ARIm ↑ PIP pur. ↑</td><td>KpNME↓</td></tr><tr><td colspan="8">Supervised (reference)</td></tr><tr><td>PDiscoFormer</td><td>4/4</td><td></td><td></td><td>28.6</td><td>55.1</td><td>27.3</td><td>64.3</td><td>4.44</td></tr><tr><td>PDiscoFormer</td><td>8/8</td><td></td><td></td><td>51.9</td><td>67.2</td><td>42.5</td><td>64.3</td><td>2.93</td></tr><tr><td>PDiscoFormer</td><td>16/16</td><td>86.8</td><td>74.8</td><td>53.6</td><td>65.6</td><td>46.0</td><td>40.4</td><td>2.87</td></tr><tr><td>PCA</td><td>16/3200</td><td>∅</td><td>∅</td><td>14.3</td><td>3.5</td><td>0.4</td><td></td><td></td></tr><tr><td> $\mathrm { P I P - N e t } _ { \mathrm { s t a g e } 2 }$ </td><td>16/768</td><td>88.9</td><td>88.9</td><td>42.0</td><td>37.0</td><td>21.3</td><td>47.0</td><td>10.16</td></tr><tr><td> $\mathrm { P r o t o Q u a n t } _ { \mathrm { s t a g e } 2 }$ </td><td>16/4000</td><td>84.9</td><td>83.0</td><td>41.0</td><td>33.3</td><td>21.2</td><td>25.2</td><td>12.16</td></tr><tr><td colspan="9">Frozen backbone, no concept bottleneck (reference)</td></tr><tr><td>CLS token</td><td></td><td>90.4</td><td>86.7</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Self-supervised</td></tr><tr><td>CFM</td><td>16/8192</td><td>54.8</td><td>42.8</td><td>12.5</td><td>4.8</td><td>0.6</td><td>16.1</td><td>12.84</td></tr><tr><td>TopK SAE</td><td>16/2048</td><td>36.9</td><td>24.3</td><td>21.2</td><td>17.0</td><td>7.4</td><td>8.7</td><td>10.92</td></tr><tr><td>PatchSAE</td><td>16/49152</td><td>62.0</td><td>33.4</td><td>18.3</td><td>5.0</td><td>2.3</td><td>6.9</td><td>19.12</td></tr><tr><td>MSAE</td><td>16/6144</td><td>73.5</td><td>42.2</td><td>13.8</td><td>3.2</td><td>1.1</td><td>15.1</td><td>12.21</td></tr><tr><td> $\mathrm { M S A E } _ { \mathrm { C L I P - L / 1 4 } }$ </td><td>16/8192</td><td>71.5</td><td>36.5</td><td>18.3</td><td>15.2</td><td>4.4</td><td>8.8</td><td>11.44</td></tr><tr><td> $\mathrm { P I P - N e t _ { s t a g e 1 } }$ </td><td>16/768</td><td>77.8</td><td>68.4</td><td>39.7</td><td>32.7</td><td>16.4</td><td>37.3</td><td>9.87</td></tr><tr><td> $\mathrm { P r o t o Q u a n t _ { s t a g e 1 } }$ </td><td>16/4000</td><td>80.1</td><td>67.7</td><td>44.0</td><td>36.4</td><td>24.3</td><td>24.6</td><td>11.59</td></tr><tr><td>DisParQ (ours)</td><td>16/1024</td><td> $\mathbf { 8 9 . 4 } _ { + 9 . 3 }$ </td><td> $8 5 . 5 _ { + 1 7 . 1 }$ </td><td> $4 7 . 5 _ { + 3 . 5 }$ </td><td> ${ \bf 4 4 . 9 } _ { + 8 . 5 }$ </td><td> $\mathbf { 2 9 . 6 } _ { + 5 . 3 }$  一</td><td> $5 4 . 5 _ { + 1 7 . 2 }$  一</td><td> $\mathbf { 9 . } 3 6 _ { - 0 . 5 1 }$  </td></tr></table>

## F.4 PDISCOFORMER ON CARS AND DOGS

PDiscoFormer releases no checkpoint for Stanford Cars or Stanford Dogs, so Table 3 leaves those cells empty. Table 17 gives our retraining with the authors’ code and recipe (Appendix D.4), at $2 2 4 \times 2 2 4$ with K=16 parts, on the same training splits as DisParQ. As in Table 2, the probes read the mean of its modulated part embeddings. Trained with class labels, PDiscoFormer matches DisParQ on the linear probe on Cars (91.1 vs. 91.0) and trails it by 2.3 to 4.9 p.p. on the other three columns. Its recipe keeps the snapshot with the best test accuracy, which is that of epoch 27 of 28 on both datasets, so this selection hardly touches the test split.

## F.5 PROTOQUANT AND PIP-NET: FROZEN VS. FINE-TUNED BACKBONE

Two of our baselines fine-tune the backbone in their own protocols. ProtoQuant trains it for classification on the target dataset and then freezes it, so the per-backbone accuracies its Table 3 reports are those of a fine-tuned network, and PIP-Net’s recipe trains its backbone end to end. We report both on the frozen DINOv2-B/14 that every other method reads (Appendix D.4), and we also ran their protocols as published. Table 18 gives the pair, for completeness and as a control on that choice.

Table 17: PDiscoFormer reproduced on Cars and Dogs (test, ×100): the authors’ code retrained at 224 × 224 with K=16. The other rows are from Table 3.
<table><tr><td rowspan="2">Method</td><td rowspan="2">K/N</td><td colspan="2">Stanford Cars</td><td colspan="2">Stanford Dogs</td></tr><tr><td>LP↑</td><td>kNN↑</td><td>LP↑</td><td>kNN↑</td></tr><tr><td>PDiscoFormer, reproduced</td><td>16/16</td><td>91.1</td><td>79.9</td><td>85.5</td><td>80.9</td></tr><tr><td>CLS token</td><td></td><td>93.2</td><td>85.6</td><td>88.6</td><td>86.8</td></tr><tr><td>DisParQ (ours)</td><td>16/1024</td><td>91.0</td><td>83.4</td><td>87.8</td><td>85.8</td></tr></table>

Fine-tuning improves classification accuracy and hurts part quality. ProtoQuant’s stage-1 linear probe on PartImageNet rises from 85.2 to 89.1 and its kNN from 83.9 to 88.7, while its foreground purity falls from 82.1 to 61.2, its part mIoU from 33.7 to 9.3 and its part-to-part R@1 from 79.2 to 30.7. On CUB its purity falls from 44.0 to 16.3 and its keypoint error rises from 11.59 to 14.47, against the 15.50 of predicting the mean keypoint. PIP-Net loses the same part columns and, unlike ProtoQuant, loses accuracy with them: its stage-1 probe drops from 77.8 to 57.6 and its kNN from 70.6 to 31.8. What the backbone is fine-tuned on accounts for the difference. ProtoQuant’s first stage trains it with class labels, so its features are class-discriminative before any codebook i learned, while PIP-Net trains it through its own label-free objective, which supplies no such signal and degrades the pretrained features in this configuration.

One column improves: PIP-Net’s keypoint error on CUB, from 9.91 to 9.07. Kp NME regresses keypoints from part centroids, so it rewards parts that lie on the bird even when those parts are less pure, and it is the one column in the table that does not read a part as a region. Further arms are left out of the table because they move more than one thing at a time. Using PIP-Net’s official protocol, which unfreezes the last two blocks after a frozen warm-up and scores a bare softmax over the backbone’s channels, yields lower part scores on PartImageNet: foreground purity 20.8 and mIoU 12.5 at stage 1 on DINOv2-B/14, against 50.7 and 21.8 frozen, and 15.8 and 7.9 on the ConvNeXt and 27.0 and 17.3 on the ResNet from its own paper. Of the fine-tuned ProtoQuant arms we ran with three seeds, on their DeiT-S, accuracy varies by at most 0.6, which is why the table carries one seed per cell.

Thus, the frozen rows of the main tables are not a weakened form of either method. They are the stronger arm on the columns the paper measures, and they keep the comparison to one variable, since a fine-tuned arm sees the target dataset’s labels through its backbone as well as through its classifier. The fine-tuned runs are usable as a control because they reproduce their authors’ reported accuracy: ProtoQuant’s fine-tuned DeiT-S reaches a top-1 of 87.0 on bounding-box-cropped CUB, against the 86.9 of their Table 3.

## F.6 RETRIEVAL RESULTS

Table 19 is Table 4 with CUB and the part → image setting added, plus two rows for: PCA (perimage and class-wise), and DisParQ described by something other than its concept identities. The attribute head quantizes a residual, the normalized region feature minus its normalized prototype, and each concept owns a private codebook, so an attribute vector carries no concept identity by construction, and its codes are only comparable between occurrences of the same concept. The three descriptor rows show what follows: the codes alone retrieve a same-part neighbor about as often as the concept ids do (R@1 78.9 against 79.7 on PartImageNet) but rank the rest of the list far worse (mAP 16.3 against 30.5); concatenating the two changes nothing; and using the attribute to order candidates that already share a concept raises mAP to 57.3 on PartImageNet and from 9.3 to 28.1 on CUB. Appearance is orthogonal to part identity, and only useful once identity has selected the candidates. No baseline has an attribute, so these rows are ours alone and unranked.

Table 19 also holds the rows that Table 4 leaves out. The category-only oracle and uniform chance are two reference rankers that use no concepts (Appendix D.6). The oracle uses ground-truth classes for accuracy and image-to-image retrieval, solving those columns by construction. For part retrieval, it uses only part-label co-occurrence groups and reaches an R@1 of 43.9, against 79.7 for DisParQ. The components of per-image PCA (Appendix D.4) differ from image to image, so part-to-part retrieval with them is close to chance (R@1 5.3, against 4.6). Class-wise PCA, which selects its components with the class label and is therefore a supervised reference, shows why R@1 must be read beside mAP: ranking by category, it reaches an R@1 of 52.4 on its own query set at an mAP of only 8.9.

Table 18: A frozen against a fine-tuned backbone, for the two baselines whose own protocols fine-tune it (DINOv2-B/14; ×100 except Kp NME; stage 2 rows are gray, as they use class labels). The main paper reports the frozen arms, so that every method reads the same features; these rows reproduce them. Fine-tuning raises classification accuracy on every dataset and lowers part quality throughout, with one exception: PIP-Net’s keypoint error on CUB, which improves. PIP-Net’s finetuned arms exist only at the native 16 × 16 grid, so its pair is the second and third rows of its block and the first row is the AnyUp arm of Tables 2 and 3. CUB carries $\mathrm { P u r _ { f g } }$ and $\mathrm { K p }$ NME rather than PIP purity, whose window rule changed after the fine-tuned arms were scored. —: not measured.
<table><tr><td></td><td></td><td></td><td colspan="5">PartImageNet</td><td colspan="3">CUB</td><td>Cars</td><td>Dogs</td><td>Flowers</td></tr><tr><td>Method</td><td>Backbone</td><td>Stage</td><td>LP↑</td><td>kNN↑</td><td>Purfg ↑</td><td>mIoU↑</td><td>p2p R@1↑</td><td>LP↑</td><td>Purfg ↑</td><td>KpNME↓</td><td>LP↑</td><td>LP↑</td><td>LP↑</td></tr><tr><td>PIP-Net</td><td>frozen, AnyUp</td><td></td><td>76.9</td><td>71.5</td><td>68.5</td><td>29.0</td><td>66.6</td><td>77.8</td><td>39.7</td><td>9.87</td><td>82.3</td><td>77.8</td><td>99.1</td></tr><tr><td>PIP-Net</td><td>frozen, AnyUp</td><td></td><td>87.5</td><td>86.8</td><td>59.3</td><td>24.9</td><td>65.4</td><td>88.9</td><td>42.0</td><td>10.16</td><td>91.4</td><td>87.1</td><td>99.6</td></tr><tr><td>PIP-Net</td><td>frozen</td><td>1</td><td>77.8</td><td>70.6</td><td>50.7</td><td>21.8</td><td>61.3</td><td>79.0</td><td>38.8</td><td>9.91</td><td>85.3</td><td>76.4</td><td>99.0</td></tr><tr><td>PIP-Net</td><td>frozen</td><td>2</td><td>87.2</td><td>87.8</td><td>37.1</td><td>20.5</td><td>51.5</td><td>88.8</td><td>39.4</td><td>10.23</td><td>91.5</td><td>86.8</td><td>99.6</td></tr><tr><td>PIP-Net</td><td>fine-tuned</td><td>1</td><td>57.6</td><td>31.8</td><td>26.4</td><td>17.2</td><td>31.7</td><td>67.6</td><td>33.5</td><td>9.07</td><td>70.8</td><td>51.8</td><td>91.4</td></tr><tr><td>PIP-Net</td><td>fine-tuned</td><td>2</td><td>88.2</td><td>85.6</td><td>22.1</td><td>15.3</td><td>29.1</td><td>90.0</td><td>37.2</td><td>8.42</td><td>93.2</td><td>86.0</td><td>99.5</td></tr><tr><td>ProtoQuant</td><td>frozen</td><td>1</td><td>85.2</td><td>83.9</td><td>82.1</td><td>33.7</td><td>79.2</td><td>80.1</td><td>44.0</td><td>11.59</td><td>87.0</td><td>82.2</td><td>99.7</td></tr><tr><td>ProtoQuant</td><td>frozen</td><td>2</td><td>87.8</td><td>89.5</td><td>71.7</td><td>26.5</td><td>62.6</td><td>84.9</td><td>41.0</td><td>12.16</td><td>91.3</td><td>86.0</td><td>99.6</td></tr><tr><td>ProtoQuant</td><td>fine-tuned</td><td>1</td><td>89.1</td><td>88.7</td><td>61.2</td><td>9.3</td><td>30.7</td><td>89.1</td><td>16.3</td><td>14.47</td><td>93.8</td><td>87.2</td><td>99.5</td></tr><tr><td>ProtoQuant</td><td>fine-tuned</td><td>2</td><td>88.1</td><td>89.0</td><td>62.8</td><td>13.3</td><td>35.2</td><td>89.4</td><td>21.2</td><td>14.85</td><td>94.1</td><td>86.5</td><td>99.5</td></tr></table>

PDiscoFormer is scored at its native feature resolution, whose 16 × 16 grid gives a query set of 3,111 regions and an oracle R@1 of 46.2, against 3,524 regions and 43.9 on the 32 × 32 grid (Appendix D.6). Resampling its part maps to our 32 × 32 grid, as we do for its CUB models, puts it on our query set and changes its scores by at most 3.5: part-to-part R@1 is 45.8, 63.5 and 71.8 for K=16, 25 and 50 (natively 49.3, 63.6 and 71.8), and mAP is 32.1, 40.8 and 43.9 (natively 34.2, 41.9 and 45.9). On either grid, PDiscoFormer trails DisParQ on R@1 (79.7) and exceeds it on mAP (30.5).

On CUB, DisParQ leads part-to-part R@1 (36.8) and R@5 (73.7), against an R@1 of 22.7 for the oracle and 7.1 for chance, and trails the TopK SAE by 0.2 on mAP. Part-to-image retrieval is at chance on CUB, since nearly every image contains nearly every keypoint, so it is omitted (∅). Appendix D.6 defines the regions, the oracle and the query sets.

MSAE is trained under two loss weightings. The MSAE rows of Table 19 use the reverse-weighted dictionary, the configuration selected on PartImageNet (Table 2), which is the only one scored on this benchmark. Table 3 and Table 16 instead report the uniform-weighted dictionary selected jointly across the four fine-grained datasets, whose CUB linear probe and k-NN read 73.5 and 42.2 against the 72.5 and 41.7 reported there.

Table 19: Retrieval on PartImageNet and CUB (×100). Among self-supervised methods, DisParQ has the best part-to-part R@1 on both datasets. Conventions follow Table 2. A double dagger marks a different grid and oracle (Section F.6).
<table><tr><td></td><td></td><td colspan="2">Accuracy</td><td colspan="3">Image → image</td><td colspan="3">Part → image</td><td colspan="3">Part → part</td></tr><tr><td>Method</td><td>K/N</td><td>LP↑</td><td>kNN↑</td><td>R@1↑</td><td>R@5↑</td><td>mAP↑</td><td>R@1↑ R@5↑</td><td></td><td>mAP↑</td><td>R@1↑</td><td>R@5↑ mAP↑</td><td></td></tr><tr><td colspan="9">PartImageNet</td><td></td><td></td><td></td></tr><tr><td colspan="9">Supervised (reference)</td><td></td><td>78.3</td><td>34.2</td></tr><tr><td>PDiscoFormer‡</td><td>16/16</td><td>89.6</td><td>88.5</td><td>78.8</td><td>92.9</td><td>70.2</td><td>38.8</td><td>63.5</td><td>36.3</td><td>49.3</td><td></td><td>41.9</td></tr><tr><td>PDiscoFormer‡</td><td>25/25 50/50</td><td>89.5</td><td>87.5 88.1</td><td>78.4</td><td>93.4</td><td>69.1</td><td>52.0</td><td>74.6</td><td>42.5</td><td>63.6 71.8</td><td>86.2 89.0</td><td>45.9</td></tr><tr><td>PDiscoFormer‡ PCA, class-wise‡</td><td>16/2528</td><td>89.8 ∅</td><td>∅</td><td>78.4</td><td>93.4</td><td>69.3</td><td>61.9 83.0</td><td>80.4</td><td>51.4 20.1</td><td>52.4</td><td>88.1</td><td>8.9</td></tr><tr><td>PIP-Netstage 2</td><td>16/768</td><td>87.5</td><td>86.8</td><td>∅ 77.6</td><td>∅ 91.9</td><td>∅ 64.2</td><td>77.5</td><td>98.4 93.8</td><td>45.9</td><td>65.4</td><td>85.2</td><td>21.1</td></tr><tr><td>ProtoQuantstage 2</td><td>16/2048</td><td>87.8</td><td>89.5</td><td>82.3</td><td>93.8</td><td>76.6</td><td>86.2</td><td>98.5</td><td>33.0</td><td>62.6</td><td>89.6</td><td>15.5</td></tr><tr><td colspan="9">Frozen backbone, no concept bottleneck (reference)</td><td></td><td></td><td></td></tr><tr><td>CLS token</td><td></td><td>89.3</td><td>88.9</td><td></td><td>92.9</td><td>75.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Patch tokens, mean-pooled</td><td></td><td>85.7</td><td>79.3</td><td>81.0 57.5</td><td>82.2</td><td>44.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ground-truth controls category-only oracle</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>82.2</td><td>91.5</td><td>81.5</td><td>43.9</td><td>89.0</td><td>44.7</td></tr><tr><td>uniform chance</td><td></td><td></td><td></td><td>0.6</td><td>2.9</td><td>0.6</td><td>13.7</td><td>47.7</td><td>14.1</td><td>4.6</td><td>20.1</td><td>4.9</td></tr><tr><td colspan="9">Self-supervised</td><td></td><td></td><td></td></tr><tr><td>PCA, per-image</td><td>16/16</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>11.0</td><td>39.0</td><td>14.5</td><td>5.3</td><td>23.6</td><td>5.2</td></tr><tr><td>CFM</td><td>16/8192</td><td>72.6</td><td>69.2</td><td>47.0</td><td>75.4</td><td>34.6</td><td>63.4</td><td>74.3</td><td>40.8</td><td>31.6</td><td>68.8</td><td>16.0</td></tr><tr><td>TopK SAE</td><td>16/1024</td><td>78.1</td><td>73.0</td><td>52.6</td><td>79.2</td><td>42.7</td><td>84.7</td><td>97.4</td><td>34.9</td><td>56.2</td><td>86.5</td><td>17.2</td></tr><tr><td>TopK SAE</td><td>16/2048</td><td>80.5</td><td>73.5</td><td>53.8</td><td>77.5</td><td>43.3</td><td>85.6</td><td>98.0</td><td>35.7</td><td>58.3</td><td>86.9</td><td>18.1</td></tr><tr><td>PatchSAE</td><td>16/1024</td><td>82.9</td><td>76.5</td><td>56.8</td><td>80.8</td><td>44.6</td><td>83.3</td><td>97.4</td><td>29.3</td><td>41.9</td><td>83.6</td><td>11.8</td></tr><tr><td>PatchSAE</td><td>16/49152</td><td>84.1</td><td>77.2</td><td>56.8</td><td>81.1</td><td>44.4</td><td>85.0</td><td>95.5</td><td>20.6</td><td>48.6</td><td>82.9</td><td>8.0</td></tr><tr><td>MSAE</td><td>16/1024</td><td>85.2</td><td>79.9</td><td>58.9</td><td>81.7</td><td>46.1</td><td>82.9</td><td>95.8</td><td>30.9</td><td>41.4</td><td>82.2</td><td>12.5</td></tr><tr><td>MSAE</td><td>16/8192</td><td>85.6</td><td>79.5</td><td>58.3</td><td>82.3</td><td>45.4</td><td>79.8</td><td>93.4</td><td>30.3</td><td>41.2</td><td>79.9</td><td>12.6</td></tr><tr><td>PIP-Netstage 1</td><td>16/768</td><td>76.9</td><td>71.5</td><td>49.1</td><td>78.6</td><td>36.3</td><td>77.9</td><td>94.6</td><td>47.6</td><td>66.6</td><td>87.7</td><td>27.0</td></tr><tr><td>ProtoQuantstage 1</td><td>16/2048</td><td>85.2</td><td>83.9</td><td>72.5</td><td>90.4</td><td>61.8</td><td>89.8</td><td>98.8</td><td>40.3</td><td>79.2</td><td>92.8</td><td>22.7</td></tr><tr><td colspan="9">DisParQ (ours) 16/1024 89.2+3.6 88.7+4.8</td><td>79.7+0.5</td><td>92.4-0.4</td><td>30.5+3.5</td></tr><tr><td>DisParQ, the same model described differently (ours only, unranked)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>attribute codes only</td><td>16/1024 16/1024</td><td>89.2 89.2</td><td>88.7 88.7</td><td>80.6 80.6</td><td>92.9</td><td>76.1</td><td>89.8</td><td>98.1</td><td>32.1</td><td>78.9 79.9</td><td>93.8</td><td>16.3 30.5</td></tr><tr><td>concept id + attribute concept id + same-concept appearance</td><td>16/1024</td><td>89.2</td><td>88.7</td><td>80.6</td><td>92.9 92.9</td><td>76.1 76.1</td><td>86.9 74.8</td><td>97.8 91.9</td><td>48.5 37.7</td><td>82.4</td><td>92.4 93.2</td><td>57.3</td></tr><tr><td colspan="9">CUB-200-2011</td><td></td><td></td><td></td></tr><tr><td>Supervised (reference)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PDiscoFormer</td><td>16/16</td><td>86.8</td><td>74.8</td><td>69.8</td><td>89.5</td><td>38.2</td><td>∅</td><td>∅</td><td>∅</td><td>45.8</td><td>84.6</td><td>38.6</td></tr><tr><td>PCA, class-wise</td><td>16/3200</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>∅</td><td>9.0</td><td>34.0</td><td>7.1</td></tr><tr><td>PIP-Netstage 2 ProtoQuantstage 2</td><td>16/768 16/4000</td><td>88.9 84.9</td><td>88.9 83.0</td><td>85.6 76.8</td><td>94.9 93.5</td><td>75.0 52.5</td><td>∅ ∅</td><td>∅ ∅</td><td>∅ ∅</td><td>32.3</td><td>69.8</td><td>8.5 8.4</td></tr><tr><td colspan="9"></td><td>30.1</td><td>65.9</td><td></td></tr><tr><td>Frozen backbone, no concept bottleneck (reference) CLS token</td><td></td><td>90.4</td><td>86.7</td><td>85.1</td><td>95.4</td><td>65.9</td><td>∅</td><td>∅</td></table>

![](images/86fbcef583927c5d0b25645dded38ca0988fcd2af0830b66c29bb43d0991737d.jpg)  
Figure 24: Example questions shown in the survey briefing for Q2: part retrieval.

## F.7 WHAT THE QUANTIZED ATTRIBUTES KEEP

For a selected concept $i _ { k }$ in an image, the attribute encoder sees the residual $r _ { i _ { k } } = \nu ( \bar { z } _ { i _ { k } } ) - \nu ( w _ { i _ { k } } )$ between the normalized region mean $\bar { z } _ { i _ { k } }$ and the normalized concept prototype $w _ { i _ { k } }$ (Section 3.3), so $\nu ( w _ { i _ { k } } ) + r _ { i _ { k } }$ is the normalized region mean. Stage 2 projects the residual to $y _ { i _ { k } } = W _ { \mathrm { P Q } } r _ { i _ { k } }$ and encodes it as $G { = } 1 6$ indices $j _ { i _ { k } , 1 : G }$ into the concept-specific codebooks ${ { B } _ { { { i } _ { k } } } }$ , with $M { = } 6 4$ codewords per group (Appendix C.3). The attribute $q _ { i _ { k } }$ that the decoder receives is learned for decoding and is not trained to reconstruct $y _ { i _ { k } }$

To see what the code keeps, we therefore decode it back into feature space: each index $j _ { i _ { k } , g }$ is replaced by the mean of group g of $y _ { i _ { k } }$ over the training occurrences of concept $i _ { k }$ that received it, and a ridge regression fitted on the training split maps the result back to feature space, giving $r _ { i _ { k } } ^ { \mathrm { d e c } }$ The images of Figures 44–48 (Appendix G.6) are held out from both.

The quantized attribute also keeps which class an occurrence of a concept comes from: in $\mathsf { A p - }$ pendix G.7, the occurrences of one concept in the evaluation images group by the class of their image (Figures 49–52).

## F.8 USER STUDY DETAILS AND RESULTS

Setup. We recruited 120 native English speakers through the clickworker.com platform, balanced in terms of gender and age (18-64). The study was fully anonymous, no personal data was collected, and it followed our institution’s guidelines for research involving human participants. The median completion time was 8 minutes 42 seconds. Before starting, participants read a short briefing describing both question types, with an annotated example for each (Fig. 25 and Fig. 24). Each participant then answered 42 questions: 21 part-retrieval questions (Q2) and 21 part-consistency questions (Q1). For both question types, the 21 questions contained 7 prototypes from each of DisParQ, PIP-Net, and CFM, drawn from all three datasets, presented in random order and with the method identity hidden. We used three survey versions with different prototype sets, each completed by 40 participants. For Q1, the per-method CUB/Cars/PartImageNet counts were 3/3/1, 3/1/3 and 1/3/3 across versions. Prototypes were selected by sampling random test images and taking one of the four most important prototypes for the image’s prediction. Here, four specifies the candidate set for stimulus selection, not the model’s concept budget. The same images were used for all three methods, and all prototypes were visualized in the same way, by their top activating image patches. To ensure data quality, we screened responses for inconsistent answering and excluded 4% of participants, recruiting replacements until 120 valid responses were collected.

The images below show one concept learned by the model. How consistent is this concept. i.e., how clearly do all images show the same visual feature?  
![](images/3c5cab64305c7160eb890925f302ea470b5320a9d7ffbdfe21e2b225980e97ed.jpg)  
Figure 25: Example questions shown in the survey briefing for Q1: part consistency.

![](images/c3fce254cf6f875e5dea53bdb42ff0ec0897b3f43f69dd2633ef2e1bb7cb4dd0.jpg)  
Figure 26: Part consistency ratings per dataset. Each point is one participant’s mean rating (1: not consistent at all, 5: fully consistent) over the prototypes of a given method shown from that dataset (n=120). Violins show the distribution, diamonds the mean, and black bars the interquartile range. Brackets mark comparisons in which DisParQ is rated significantly higher than a baseline (twosided paired Wilcoxon signed-rank test, Holm-corrected): $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ unmarked comparisons are not significant.

Q1: Part consistency. Participants were shown the visualization of a single prototype and asked: “The images below show one concept learned by the model. How consistent is this concept, i.e., how clearly do all images show the same visual feature?” They answered on a 5-point scale (1: not consistent at all, 2: slightly, 3: moderately, 4: very, 5: fully consistent) and were told that there are no right or wrong answers. For each participant, we average ratings per method, overall and per dataset. We compare DisParQ with each baseline using two-sided paired Wilcoxon signed-rank tests, with Holm correction across all eight comparisons (two baselines, overall and three datasets).

Participants rated DisParQ prototypes as the most consistent overall, with a significant advantage over both PIP-Net $( p < 0 . 0 0 1 )$ and CFM $( p < 0 . 0 0 1 )$ . Per dataset (Fig. 26), the gap is largest on CUB, where DisParQ is significantly better than both baselines. On Cars it significantly outperforms CFM, while on PartImageNet all three methods are rated similarly low, suggesting that consistent parts are harder to judge on this more diverse dataset.

Table 20: Part retrieval accuracy (%). Participants chose which of four model-identified regions matches the visualized concept, or “None of the above” (chance: 25% or lower); 120 participants. All methods are significantly above chance $( p < 1 0 ^ { - 4 }$ , one-sided Wilcoxon). Stars mark baselines that DisParQ significantly outperforms (paired Wilcoxon signed-rank test on per-participant accuracy, Holm-corrected): $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$
<table><tr><td>Method</td><td>All</td><td>CUB</td><td>Cars</td><td>PartImageNet</td></tr><tr><td>Random</td><td>25.00</td><td>25.00</td><td>25.00</td><td>25.00</td></tr><tr><td>CFM</td><td>51.50***</td><td>50.00**</td><td> $5 6 . 6 7 ^ { * * * }$ </td><td>48.57**</td></tr><tr><td>PIP-Net</td><td>60.12***</td><td>63.57</td><td>63.57*</td><td>53.21***</td></tr><tr><td>DisParQ</td><td>69.05</td><td>65.00</td><td>72.86</td><td>69.29</td></tr></table>

Q2: Part retrieval. Inspired by HIVE (Kim et al., 2022), this task measures how well humans can predict the model’s reasoning from its explanations. Participants were shown the visualization of a prototype on the left and a new image with four lettered regions on the right, all identified by the model, and asked: “The images on the left show one concept learned by the model. Which region of the image on the right does this concept correspond $t o ? ^ { \dag }$ Participants could select one of the four regions or “None of the above”. The correct answer is the region in which the model localizes the prototype. For each participant, we compute the accuracy per method, overall and per dataset, and compare DisParQ with each baseline using a two-sided paired Wilcoxon signed-rank test with Holm correction. We test each method against a chance level of 25% with a one-sided Wilcoxon signed-rank test; this is conservative, as five answer options were available.

All methods lead to above-chance accuracy, but participants were most accurate with DisParQ prototypes on every dataset, reaching 69% overall compared to 60% for PIP-Net and 52% for CFM (Tab. 20). The advantage is significant against both baselines overall and on every dataset except CUB, where DisParQ and PIP-Net perform comparably. It is most pronounced on PartImageNet, where DisParQ outperforms PIP-Net by 16 percentage points. A logistic regression with standard errors clustered by participant confirms the result: with DisParQ, the odds of a correct answer are 1.5 and 2.1 times higher than with PIP-Net and CFM, respectively $( p < 1 0 ^ { - 4 } )$

## G QUALITATIVE RESULTS

This section first describes our interactive concept visualizer to browse concepts from DisParQ (Sections G.1). It then compares the concept maps of DisParQ with those of the baselines (Sections G.2– G.4), and shows what the description of DisParQ contains: the concepts behind a prediction (Section G.5) and their attributes (Sections G.6 and G.7).

## G.1 INTERACTIVE CONCEPT VISUALIZER

We provide an interactive concept visualizer at https://disparq.gmum.net/ that allows browsing concepts from DisParQ. The visualizer includes concept banks for five datasets (PartImageNet, CUB, Dogs, Cars, and Flowers), along with an image gallery that shows what concepts activate for an image and where. We describe the two galleries below:

Image gallery. This consists of three images per class (Figure 27) for each dataset. The gallery view (Figure 27, top) shows images with a concept overlay, and can be filtered by class or concept ID. Each image can be selected to open the per-image view, that shows the top-16 concepts, where they activate, and what they are, characterized by other exemplars from the dataset that contain the same concept (Figure 27, bottom). The visualizer also contains an option to view the per-image view of a random image from the selected dataset.

Concept gallery. This shows the full set of concepts that activate on at least one image in the dataset (Figure 28). The gallery view (Figure 28, top) shows all concepts from the dataset that are found in at least one image. Each concept card shows exemplars of images and where the concept activates on it, along with the number of images the concept activates on the primary classes they belong to. The per-concept view (Figure 28, bottom) shows the images and regions within them that correspond to the strongest, typical, and weakest activations for the concept in the dataset. It also shows the top classes the concept appears in, and other concepts that most frequently co-occur with the concept.

![](images/183364877e2b59c38acfa990207284933fbbb38086de278a5758c52ac223d931.jpg)  
Figure 27: Visualizer: the image gallery and one image. Top: The training images of one dataset, filtered by class or by concept, with an overlay of each image’s concept map. Bottom: One image, opened from the gallery: its concept map in the middle, its concepts numbered by match strength. Each concept’s row shows the training images where it matches most strongly, and an arrow joins the row to the concept’s region.

![](images/e81ffbccb74d6209d1b7f4670071ab8fe24518de682697bd25963532fd6d06db.jpg)

## Concepts

![](images/5e0d29ed06ad4ac2455fa52e10d8894149f56df02cb6538a7d7941779d41af26.jpg)

![](images/3dfc53e8f184cd49e7ee8d515d6c4b9b89766f7933e749084d97064ee63c7e2d.jpg)  
Figure 28: Visualizer: the concept gallery and one concept. Top: Every concept used on a dataset, each shown by its four strongest matches. Bottom: One concept, opened from the gallery: its strongest, typical and weakest matches, the classes it appears in with examples, and the concepts it most often appears with.

(b) One concept on different images GT part (outlined): boat sail

## G.2 PARTIMAGENET

This subsection extends Figure 6. All figures use the PartImageNet validation split and the models of Table 2; every method sees the same center-cropped 224×224 input, and the ground-truth mask is cropped identically. Colors follow the rule of Figure 6. Figure 29 shows five further images and, for two parts, the one concept of each method that is most consistent with that part over the validation split.

Input (a)  
GT parts  
DisParQ  
ProtoQuant  
![](images/0e84f3481607b9d4ed8130a202bd79bce130e6d288d9c52b21bd0bfd2e72e001.jpg)  
CFM  
PDiscoFormer†

![](images/18830fbca50938940f1c434f15c47e411799bd75621c245a20bf8e3934f3bd59.jpg)  
head/sail body foot hand/wing tail further segment within a part on the object, within no single part on background

Figure 29: Part maps, and one concept across images, on PartImageNet. (a) Each method’s hard concept map is cut into segments, the segments are matched one-to-one to the ground-truth parts of that image, and a matched segment takes its part’s color, so agreement with the GT parts column reads as matching colors. Hatched segments lie on the object but match no part (over-segmentation); they take a part’s color when at least 60% of the segment lies within that part, and are gray otherwise. Pale outlined segments lie on the background. <sup>†</sup> trained with class labels. (b) For a ground-truth part (outlined), the one concept of each method that is most consistent with that part over the validation split, shown on the same five images. The concept of DisParQ marks the part in every image; a baseline’s best concept is absent from most of them, or covers the whole object.

Figures 31 and 32 compare DisParQ with five baselines: ProtoQuant (Janusz et al., 2026), MSAE (Zaigrajew et al., 2025), PDiscoFormer (Aniraj et al., 2024), PIP-Net (Nauta et al., 2023) and CFM (Wittenmayer et al., 2026). DisParQ follows the annotated parts and stops at the object boundary: it separates the hull of a sailboat from its sails, and the head and flippers of a turtle from its shell. The baselines merge parts into one concept (MSAE, PIP-Net and CFM cover a turtle with the color of its shell) or spread into the background (MSAE and PIP-Net fill the sea and sky around a sailboat). On random images (Figure 32), DisParQ still recovers small parts such as a snake’s head, while MSAE marks the background and CFM covers the object with a single concept. Figure 30 follows one concept of each method across five images: the concept of DisParQ marks the sail, the hull and the bottle mouth in all five images and the tire, the wing and the fish head in most, whereas the best-matching concept of a baseline is missing from several images or covers the whole object.

Where DisParQ fails. Figure 33 shows the images with the largest per-image deficit against the best baseline. They share one cause: close-ups in which a single annotated part (almost always head) fills the frame. DisParQ describes such an image with many finer concepts like snout, eyes, ears, fur, of which only one can be matched to the single ground-truth part, while a method with one coarse concept per object matches it trivially. We see this as a limitation of the annotation granularity of human-labeled datasets rather than of the concepts: PartImageNet labels a head as one part, so a finer decomposition that is consistent across images is scored as over-segmentation.

One concept on different images GT part (outlined): boat sail  
GT part (outlined): bicycle tire  
GT part (outlined): bird wing  
![](images/7660bcb55bea0a6e3cf7d963d86ba1704e6a932fad1f860783a200c52868d567.jpg)  
Figure 30: One concept, different images. Each row shows, for the outlined ground-truth part, the concept of that method most consistent with it over the validation split. <sup>†</sup> trained with class labels.

![](images/3880e33962f07b69e8bf16e9aec5c501a2ad2558b84822f5cf709dcd771a6100.jpg)  
head/sail body engine/foot fin/wing further segment within a part on the object, within no single part on background

Figure 31: Further examples on PartImageNet, all five baselines. Maps and colors as in Figure 6.   
<sup>†</sup> trained with class labels.

![](images/a71e935384163250b411298e6e801a8d1f82d6ce77f3bb1437547624529af329.jpg)  
head body foot/tire wing seat/tail further segment within a part on the object, within no single part on background

Figure 32: Random examples from the PartImageNet validation split. Maps and colors as in Figure 6. <sup>†</sup> trained with class labels.

![](images/9a54953a56c2c170a8ac684110f6d36fa4d2077f29da3477ceab299f84338242.jpg)  
head body further segment within a part on the object, within no single part on background

Figure 33: Failure cases: images on which a baseline reaches a higher per-image mIoU than DisParQ. Maps and colors as in Figure 6. <sup>†</sup> trained with class labels.

## G.3 FINE-GRAINED DATASETS

This subsection repeats Figure 6 on four fine-grained datasets: CUB (Wah et al., 2011), Stanford Cars (Krause et al., 2013), Stanford Dogs (Khosla et al., 2011) and Oxford Flowers (Nilsback & Zisserman, 2008), with the baselines trained on each dataset (Figures 34–37). All images come from the test split of each dataset. These datasets have no part masks, so a segment is colored by where it sits on the object, by the same rule in every column; the reference column shows the foreground and, on CUB, the annotated keypoints. Panel (b) follows two concepts of DisParQ, each across five images, beside each baseline’s best-matching concept. The maps of DisParQ stay on the object and divide it into regions that recur across images, while MSAE also covers the background. The concept of DisParQ marks the same part in all five images, for example the tail or the head of a bird, while the best-matching concept of MSAE covers whole birds. PDiscoFormer is trained with class labels and shown for reference.

![](images/1cd037fc554307beae9f9795710b1719d4f3b7cc6b7f3d209cb33cdf3fd0f021.jpg)  
color = where the segment sits on the object (same rule in every column; no part meaning) GT keypoint on background

Figure 34: CUB: DisParQ and baselines. (a) Hard concept maps, colored by where a segment sits on the object (same rule in every column, no part meaning); pale outlined segments lie on the background. (b) Two concepts of DisParQ (outlined), each beside the concept of every baseline that agrees best with it. <sup>†</sup> trained with class labels.  
Input (a)  
Foreground  
DisParQ  
ProtoQuant  
MSAE  
PDiscoFormer†  
![](images/4ab7070f670ae294dd7e0a219017f1db235ba9dc457a753141953ed908add957.jpg)  
color = where the segment sits on the object (same rule in every column; no part meaning) on background

(b) One concept on different images

Figure 35: Stanford Cars: DisParQ and baselines. (a) Hard concept maps, colored by where a segment sits on the object (same rule in every column, no part meaning); pale outlined segments lie on the background. (b) Two concepts of DisParQ (outlined), each beside the concept of every baseline that agrees best with it. <sup>†</sup> trained with class labels.

![](images/e991e7e15fc56412887b4022abfebc21331ba998e98b287278062b5e2670d373.jpg)  
PDiscoFormer†

(b) One concept on different images DisParQ concept (outlined): paw  
![](images/c1d4ae5d0f41e9f78c0f5cc2e52e911f613c8e46acac220cae274700895c0d0d.jpg)  
color = where the segment sits on the object (same rule in every column; no part meaning) on background  
Figure 36: Stanford Dogs: DisParQ and baselines. (a) Hard concept maps, colored by where a segment sits on the object (same rule in every column, no part meaning); pale outlined segments lie on the background. (b) Two concepts of DisParQ (outlined), each beside the concept of every baseline that agrees best with it. <sup>†</sup> trained with class labels.

Input (a)  
Foreground  
DisParQ  
![](images/98177787ee2e97186f04693780435f17c2faa085011375f64c9e17a22c75ed27.jpg)  
MSAE  
PDiscoFormer†  
PIP-Net

(b) One concept on different images  
![](images/e0e30c23923132cdf05e4d0c7016b7f1e7129f0cc56b0ab1af98b3c9be93eb65.jpg)  
color = where the segment sits on the object (same rule in every column; no part meaning) on background

Figure 37: Oxford Flowers: DisParQ and baselines. (a) Hard concept maps, colored by where a segment sits on the object (same rule in every column, no part meaning); pale outlined segments lie on the background. (b) Two concepts of DisParQ (outlined), each beside the concept of every baseline that agrees best with it. <sup>†</sup> trained with class labels.

## G.4 PART-TO-PART RETRIEVAL

This subsection shows part-to-part retrieval, the benchmark of Table 4, defined in Appendix D.6, on all five datasets (Figure 38). For a query region (left), each row gives the four regions of other images that a method ranks highest by the concept histogram of the region. On PartImageNet and CUB, a green border marks a retrieval that carries the query’s annotated part and a red border a miss. Queries and retrievals come from held-out images (the PartImageNet validation split and the test splits of the other datasets), and all methods describe the same regions. A region is an annotated part mask on PartImageNet and a keypoint group on CUB; on Cars, Dogs and Flowers, which have no part annotation, it is a concept of DisParQ.

All four retrievals of DisParQ carry the query’s part in each of the six scored panels, while at most two of the four retrievals of ProtoQuant or MSAE do, as for the snake head and for the head and leg of a bird. PDiscoFormer, trained with class labels, also retrieves the bird parts. On Cars, Dogs and Flowers, which have no part annotation, DisParQ retrieves windshields, bumpers, grilles, muzzles and petals like the query, while the baselines return regions of mixed appearance.

![](images/d1b90f50bd0950d3e0fd1a445e2a670f4dd26dc54e519306dae9e82b3edec20b.jpg)  
Figure 38: Part-to-part retrieval on five datasets. Each panel is one query: the region on the left (dark border) and, per method (rows, DisParQ first), the four highest-ranked regions of other images. Green or red border: the retrieval does or does not carry the query’s annotated part (PartImageNet and CUB); Cars, Dogs and Flowers are unscored and their region names are ours. <sup>†</sup> trained with class labels.

## G.5 CLASS EVIDENCE OF DISPARQ AND INTERPRETABILITY

This subsection extends Figure 1 to more images, for DisParQ on DINOv2-B/14 (Oquab et al., 2024) (Figures 39–43). Predictions use the separate explanation classifier in Appendix D.6. The left panels show the four concepts with the largest positive drops in the predicted-class logit. The right panels show their nearest training occurrences of the same concept, ranked by cosine similarity between attributes. The images on the left come from the PartImageNet validation split and the test splits of the other datasets.

On PartImageNet (Figure 39), the concepts for a Jeep, a warplane or a Komodo dragon retrieve the same part in other images and classes. On CUB (Figure 40), the green head of a mallard retrieves the heads of other mallards, the head and wing of a blue jay account for 81% of its positive logit drops, and the yellow wing bar of a European goldfinch retrieves wings with the same bar. On Stanford Cars (Figure 41), the evidence lies on parts a person would name: the number plate and tail lights of a BMW X5 retrieve number plates and tail lights of other cars, and the headlight and wheel of a Volkswagen Beetle retrieve round headlights and wheel rims. On Stanford Dogs (Figure 42), it lies on the face and the coat, and the face concept of a Blenheim spaniel retrieves the faces of other spaniels. On Oxford Flowers (Figure 43), the corona and stamens of a passion flower account for 79% of its positive logit drops, and the dark disc of a sunflower retrieves other sunflower discs.

Interpretability. Two regularities hold across the 25 examples. Positive logit drops are concentrated in a few concepts. The four largest contributions account for 56 to 100% of their total and the largest alone for 17 to 52%. These normalized occlusion effects measure relative support under the intervention. The concepts are also shared across images, as each concept retrieves the same part in other images, often of other classes, so predictions for different classes are explained with one vocabulary of parts, and a prediction reads as a short list of recognizable parts with their weights. Appendix D.6 (Concept contribution) defines how the contribution is measured.

✓ American egret p=0.96

![](images/a22eee17effebe82bacc7dced93c92e3887331b62ad94eb0154d1def95f30697.jpg)  
Figure 39: PartImageNet: evidence concepts and what they retrieve. Left: the four concepts with the largest contribution to the predicted-class logit. Right: each one cropped from this image (thick border) beside its nearest training occurrences.

![](images/5ed621c6103792d6667edc13eb2fcf5d5e4188e32526a458a0462ed1c4caf1b9.jpg)  
Figure 40: CUB: evidence concepts and what they retrieve. Left: the four concepts with the largest contribution to the predicted-class logit. Right: each one cropped from this image (thick border) beside its nearest training occurrences.

![](images/13c67691d32c98d96a801c302426548243aebee55efa3f772d5cf56545081ce6.jpg)  
✓ Hyundai Sonata Sedan 2012 p=0.73

Figure 41: Stanford Cars: evidence concepts and what they retrieve. Left: the four concepts with the largest contribution to the predicted-class logit. Right: each one cropped from this image (thick border) beside its nearest training occurrences.

Concept nearest neighbors  
![](images/44c947123d5b858382da773174c63d7ab23589db9dcc6701bf30a6206d9b6057.jpg)  
Figure 42: Stanford Dogs: evidence concepts and what they retrieve. Left: the four concepts with the largest contribution to the predicted-class logit. Right: each one cropped from this image (thick border) beside its nearest training occurrences.

![](images/56763271215c8e0f1a85fe5b87de55df21989d8b2d1830a5374db2e38aae8b95.jpg)  
✓ Orange dahlia p=0.94

Figure 43: Oxford Flowers: evidence concepts and what they retrieve. Left: the four concepts with the largest contribution to the predicted-class logit. Right: each one cropped from this image (thick border) beside its nearest training occurrences.

## G.6 WHAT QUANTIZATION KEEPS

This subsection shows what the product-quantized attribute (Jegou et al.´ , 2011) keeps of the continuous one (Figures 44–48). To draw a code, we decode it back into feature space as $r _ { i _ { k } } ^ { \mathrm { d e c } }$ using statistics of the training split (Appendix F.7). The images shown come from the PartImageNet validation split and the test splits of the other datasets. Top: one PCA basis, fitted on the backbone’s foreground patch features, maps features to colors, so colors compare across columns. The columns show the backbone features and each concept region drawn in three ways: as its concept key $\nu ( w _ { i _ { k } } )$ , as key plus continuous attribute $\nu ( w _ { i _ { k } } ) + r _ { i _ { k } }$ , and as key plus quantized attribute $\nu ( w _ { i _ { k } } ) + r _ { i _ { k } } ^ { \mathrm { d e c } }$ . Bottom: for two concepts, the attributes of their occurrences before (left) and after (right) quantization, drawn as image crops at their coordinates in one PCA plane. The two panels are scaled independently, so what matters is which crops stay next to each other.

Top rows. Drawn as its concept key alone, a region has the same color in every image: the key says which concept a region belongs to, not how the region looks. Adding the continuous attribute gives each region back the color of its own backbone features, and the quantized attribute gives nearly the same colors. Bottom panels. Quantization keeps the neighborhoods of the attribute space: 55 to 78% of the 5-nearest-neighbor relations (the share in each panel title) are the same before and after. The code therefore keeps which occurrences of a concept look alike, which Section G.7 examines over all occurrences of a concept.

![](images/6fe1b9ccfef909294a609f5d1b1891a3e37c967e3438a7df7df152be5f8df14a.jpg)

bird wing: continuous attribute  
![](images/4ecbf8b548408562d5c7cfeb584c4d3fa623b4771fc96663ba7063571f0c7dee.jpg)

quantized attribute — 78% of 5-NN kept  
![](images/6bc5e7ff830e8db564fab6f58b17fab51a098766229fa957967fc4d234536221.jpg)

fish head: continuous attribute  
![](images/5fc39516117f0009ee49f10eef76d9d9b3a625f4e681bb98c4149f51a4366c8c.jpg)

![](images/04db9fbbfc910fdbcf8b860e395c9ac24eee30b05eeb9d2996435932c86e409a.jpg)  
Figure 44: PartImageNet: what quantization keeps. Top: backbone features, and each concept region as its key, key + continuous attribute and key + quantized attribute, in one PCA basis. Bottom: two concepts’ attribute spaces before and after quantization (same crops and basis; panels scaled independently).

![](images/0059e1271e9c95904756cec3610cae10af08b48c6f6e76df752ac363b544ac2e.jpg)

![](images/7b415556f3476da77a6a8d3609cca1e97b7eeaee4effd355d701a70573cc666d.jpg)

![](images/57327154408add1d722d08fd3fd62fa190c1eee09b4a3da6568a9888755e511f.jpg)

head: continuous attribute  
![](images/8182e910e5dae1504ba1a446659e5e88f7bde94ce993626cbc40a31414126d90.jpg)

![](images/659f3c531936397bc1a84ca19de6e9def44b8b17f4edeff0e9c7cc1bd28cc62e.jpg)  
Figure 45: CUB: what quantization keeps. Top: backbone features, and each concept region as its key, key + continuous attribute and key + quantized attribute, in one PCA basis. Bottom: two concepts’ attribute spaces before and after quantization (same crops and basis; panels scaled independently).

![](images/3d5694f27f143b7802fbdd552d6f0811d93c42c2ea4c93d964c0ff6b2e65b4a2.jpg)  
Figure 46: Stanford Cars: what quantization keeps. Top: backbone features, and each concept region as its key, key + continuous attribute and key + quantized attribute, in one PCA basis. Bottom: two concepts’ attribute spaces before and after quantization (same crops and basis; panels scaled independently).

Input  
Backbone features  
Concepts only  
+ continuous attr.  
+ quantized attr.  
![](images/c091ef50f6169a0605fd3fd3e54613dc857973499927745841ea92c20c1b895e.jpg)

paw: continuous attribute  
![](images/c490e90ad7f0fc6b3073198dfa78328553a3b90637abd93ed837ebe4f1c0d8be.jpg)

quantized attribute — 66% of 5-NN kept  
![](images/8773749d457160de038a20516b9f7000869deb7057ed17ce3d73baaaf640237b.jpg)

hind leg: continuous attribute  
![](images/cfd58d60b368872452dd9239d3532699322285474e08e8f22bd13eaf5cbb3b0b.jpg)

![](images/622273c7ce057b5e61ac4efe26b768d91ef915bc4ed54bd2de503f8603fb31b3.jpg)  
Figure 47: Stanford Dogs: what quantization keeps. Top: backbone features, and each concept region as its key, key + continuous attribute and key + quantized attribute, in one PCA basis. Bottom: two concepts’ attribute spaces before and after quantization (same crops and basis; panels scaled independently).

Backbone features

quantized attribute — 62% of 5-NN kept

\+ quantized attr.

\+ continuous attr.

quantized attribute — 78% of 5-NN kept  
Input  
Concepts only  
![](images/e8ae7fe02c1a7c90c2c9a33061b0085397da2f5a3911abceebb19787888539de.jpg)

flower center: continuous attribute  
![](images/5ada95b34c453d58e540cfde9e613bf3651e920f745ee5c2b50db7cc8bd6e035.jpg)

![](images/06e04db72cfc024f21b4b775d3cf945c1793684d392617bfc9198ed384084b35.jpg)

petals: continuous attribute  
![](images/9ce6c80cf0cae3572e745afab424884ef9b1f1ee64c1acce689fe017f0195148.jpg)

![](images/6672f72f69fbc8c56b27953a21f144b3b6ee57cfd690f555d0135c49db42094c.jpg)  
Figure 48: Oxford Flowers: what quantization keeps. Top: backbone features, and each concept region as its key, key + continuous attribute and key + quantized attribute, in one PCA basis. Bottom: two concepts’ attribute spaces before and after quantization (same crops and basis; panels scaled independently).

## G.7 HOW ONE CONCEPT VARIES ACROSS IMAGES

This subsection shows how one concept varies across images (Figures 49–52). Two occurrences of a concept share its prototype and differ in their attribute codes (Section 3.3). A concept occurs in an image when the model selects it among the K=16 concepts of that image and gives it at least 1.2% of the 32 × 32 grid. Each map holds the quantized attributes $q _ { i _ { k } }$ (Equation 10) of all occurrences of one concept in images not used in training: the PartImageNet validation and test splits (3,614 images) and the test splits of CUB (5,794), Stanford Cars (8,041) and Stanford Dogs (8,580). At evaluation, $q _ { i _ { k } }$ is the concatenation of the G=16 selected codewords, so it follows from the discrete code alone.

We fit a PCA to these vectors and draw each occurrence at its coordinates, as a crop with the concept’s region outlined or as a dot. The plane holds 12 to 20% of the variance, so positions are approximate; the row below each map gives the exact neighbors of one occurrence (the input, the large crop), its four nearest and four farthest by cosine over the full 768-dimensional attribute. Unlike the bottom panels of Figures 44–48, which compare continuous and quantized attributes on a small gallery, these maps show the quantized attribute of every occurrence.

The frame of a crop gives the class of its image (the ImageNet class on PartImageNet, the species on CUB, the make on Stanford Cars and the breed on Stanford Dogs), with the five most frequent classes of the concept colored and the others gray; the outline color gives the distance to the input, from vermilion (nearest) to blue (farthest). The model never sees these labels during training. They are used here only to color the frames.

Within a concept, occurrences group by the kind of object, although no class label is used in training. The PartImageNet concept on the quadruped head separates tigers, leopards, cougars, domestic cats and cheetahs, and the four nearest occurrences of the input leopard are leopards; the four nearest beaks to an evening grosbeak’s belong to evening grosbeaks, the nearest grilles to a Mercedes-Benz C-Class come from the same model, and the nearest muzzles to a bull mastiff’s are those of bull mastiffs.

This is the division of labor the model is built for. The concept identity is shared across classes and says which part is present, such as a grille; the attribute says which kind of that part it is, down to the species, the make or the breed. A description by concepts and attributes can therefore separate fine-grained classes that share all their parts, which is consistent with the small gap to the backbone on the fine-grained datasets (Table 3).

What this does not show. The attribute is computed from a frozen DINOv2 region feature, which already separates classes, so part of this structure comes from the backbone. The figures show that the discrete code preserves it within a single concept; they do not show that quantization adds it.

quadruped head  
![](images/435dc15e7c3680b5f4b7466669b19b2af1e74aa7e4a695ddbba912d84206a603.jpg)  
Figure 49: PartImageNet, quadruped head (64 images) and quadruped body (60 images): how a concept of DisParQ varies across images. Map: PCA of the quantized attributes of all occurrences of the concept; each is a crop with its region outlined, or a dot; the dark-framed crop is the input. Frame color: the ImageNet class of the image; outline color: distance to the input. Row: the input, its four nearest and four farthest occurrences. On PartImageNet and CUB, a concept is named after the annotated part it falls on.

beak  
![](images/1270d60f16fffd8fc63cfca7b8623201b4b8b72c3b51c3f5378bc42a4aa3f337.jpg)

![](images/9cb181d58ff4d7354766b82a34adb4a65504ead82e8a90afdea31acf78e27ef2.jpg)

![](images/7fb40948c87b4e5c2f006cc74fcf6263a1fd4381875b4fb8b19d6b7515cbe336.jpg)

![](images/d482a064c946bc6ead0a1c5c8738305d281844f0c68b3e2c29dfcd59cc51cc68.jpg)  
Figure 50: CUB, beak (73 images) and wing (191 images): how a concept of DisParQ varies across images. Map: PCA of the quantized attributes of all occurrences of the concept; each is a crop with its region outlined, or a dot; the dark-framed crop is the input. Frame color: the species of the image; outline color: distance to the input. Row: the input, its four nearest and four farthest occurrences. On PartImageNet and CUB, a concept is named after the annotated part it falls on.

![](images/5096fa01e81526f2414633d635e7236f5aa201121d102092f87398d44c45dc99.jpg)  
Figure 51: Stanford Cars, two concepts (288 and 500 images): how a concept of DisParQ varies across images. Map: PCA of the quantized attributes of all occurrences of the concept; each is a crop with its region outlined, or a dot; the dark-framed crop is the input. Frame color: the make of the image; outline color: distance to the input. Row: the input, its four nearest and four farthest occurrences. On PartImageNet and CUB, a concept is named after the annotated part it falls on.

![](images/5ef290bc41a47602034831b9008ed8469e171743137f919456a938fc2b6adbb2.jpg)  
Figure 52: Stanford Dogs, two concepts (197 and 212 images): how a concept of DisParQ varies across images. Map: PCA of the quantized attributes of all occurrences of the concept; each is a crop with its region outlined, or a dot; the dark-framed crop is the input. Frame color: the breed of the image; outline color: distance to the input. Row: the input, its four nearest and four farthest occurrences. On PartImageNet and CUB, a concept is named after the annotated part it falls on.

## H AUTOMATIC ALIGNMENT OF CONCEPT NAMES

DisParQ learns spatially grounded concepts without text supervision (Section 3). In this section, we investigate whether these concepts can be named after training using language-aligned features and a fixed vocabulary, and whether their quantized attributes help describe individual occurrences. We first align the frozen concept descriptions with a language-aligned visual representation, then retrieve names from a fixed vocabulary. Finally, we assess whether those names recover the object or part labels of their regions. All naming methods use the same regions within each model and evaluation dataset, so their comparison isolates the contribution of the naming rule.

## H.1 CONCEPT DESCRIPTIONS AND ALIGNMENT TARGETS

Frozen concept models. We use three frozen DisParQ models, two based on DINOv2 and trained on ImageNet-1k and PartImageNet (PIN), respectively, and one based on CLIP-DINOiser and trained on PartImageNet. Following the backbone interfaces in Appendix E.1, the DINOv2 models use $H = W = 3 2$ , while CLIP-DINOiser uses $H = W = 2 8$ and omits layer fusion.

For each image x, we extract the selected concepts and hard identity map p of Section 3.2. A concept occurrence consists of the locations assigned to a selected bank identity $n = i _ { k }$ , namely those with $p _ { u } = n$ . Its description combines the fixed prototype $w _ { n }$ with the image-dependent attribute $q _ { n }$ from Equation 10. Naming therefore uses the same quantized attributes as the decoder. As in the method section, dependence on the current image is implicit.

Language-aligned targets. To relate these descriptions to text, we use a frozen CLIP-DINOiser (Wysoczanska et al.´ , 2024) teacher. Its projected patch features follow the languagealigned interface described in Appendix E.1. We bilinearly resample them to the concept model’s $H \times W$ grid, average over locations with $p _ { u } = n ,$ , and normalize the resulting vector to unit length. The resulting target $\varphi _ { n } \in \mathbb { R } ^ { 5 1 2 }$ describes the image content assigned to concept n. Every comparison uses the same teacher and its OpenCLIP ViT-B/16 text encoder (Cherti et al., 2023), trained on LAION-2B (Schuhmann et al., 2022).

Data and fitting split. For each concept model, we fit alignment independently on PartImageNet (He et al., 2022) and VOC2012 (Everingham et al., 2012). We split the 20,466 PartImageNet training images or 5,717 VOC2012 Main training images into 90% for fitting and 10% for adapter selection, using image-level split seed 45. Only the fitting subset determines adapter parameters, input statistics, and mean target vectors. No segmentation labels are used in fitting or selection. Evaluation uses the 1,206 PartImageNet validation images or 1,449 VOC2012 validation images. Each image undergoes a single shorter-side resize and center crop to $2 2 4 \times 2 2 4$ , without sliding windows or multiple crops. Annotation masks undergo the paired transformation with nearest-neighbor interpolation. An occurrence used for fitting or adapter selection must occupy at least four assignment-grid cells.

## H.2 CONCEPT DESCRIPTIONS TO NAMES

A fixed name can describe what occurrences of a concept have in common, while an attributedependent name can also reflect differences between occurrences. We compare both approaches in the teacher’s 512-dimensional space using the same vocabulary.

Fixed concept descriptions. The region mean averages $\varphi _ { n }$ over fitting images for each concept and normalizes the mean to unit length. It provides a fixed description without learning an adapter. The linear key and MLP key methods instead learn a mapping from the prototype $w _ { n }$ to the teacher space. Here $\mathrm { i \hbar \ k e y ^ { \prime } }$ refers to the matching prototype $w _ { n }$ . The MLP has one hidden layer of width 512 with GELU. Both mappings produce one normalized vector per concept and therefore one fixed name. We also evaluate the normalized prototype directly (direct key) for CLIP-DINOiser, whose prototypes occupy the compatible projected CLIP space. DINOv2 prototypes have no such direct correspondence to text.

Attribute-dependent descriptions. To let names reflect individual occurrences, we add $q _ { n }$ to the adapter input. Before fitting the adapters, we compute a mean and standard deviation for each input dimension across the occurrences in the fitting split, separately for prototypes and attributes. We subtract the corresponding mean and divide by the standard deviation for each adapter input. The additive adapter applies a learned linear projection to each standardized input, adds the two outputs and a learned bias, and normalizes the result to unit length. The joint MLP instead concatenates the standardized inputs and passes them through a 512-unit hidden layer with GELU, followed by a projection to the teacher space and unit-length normalization. This allows the attribute’s contribution to depend on the prototype.

Adapter fitting. Adapters minimize cosine distance to $\varphi _ { n }$ , with occurrences weighted inversely by their concept’s fitting frequency. We use AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 512, and at most 100 epochs. We select the adapter with the highest holdout cosine similarity, averaged equally over concepts with at least five qualifying fitting-image occurrences. These are the supported concepts used below. Early stopping uses patience 15 and minimum improvement $1 0 ^ { - 5 }$

The pooled teacher reference names each validation occurrence directly from $\varphi _ { n }$ , which is computed by running the teacher on that validation image and pooling its features over the concept region. The adapters instead predict a vector in the teacher’s space from $w _ { n }$ and, for the attribute-dependent methods, $q _ { n }$ . The reference therefore shows what names can be retrieved from the teacher’s features on the same regions.

Vocabulary retrieval. We use the filtered vocabulary supplied by the authors of CFM (Wittenmayer et al., 2026), retaining 102,171 distinct strings after removing the malformed entry (#. Each string has three normalized text embeddings, obtained by averaging the supplied noun, adjective, and verb prompt sets separately. Given a normalized visual description, we select the string with the largest cosine similarity across all strings and prompt groups. Every method uses this same vocabulary and prompt recipe, without CFM’s hierarchy-specific naming correction. Dataset class names do not restrict the choice of a name.

## H.3 EVALUATING THE RETRIEVED NAMES

Name-to-class retrieval. We assess the selected name by asking whether it retrieves the annotated label of its region. We re-encode the name and the dataset class names with a shared prompt ensemble: the teacher’s 80 ImageNet templates for VOC, and “a photo of $\{ \} . \ ' / \Updot { \mathrm { ~ a ~ } }$ close-up photo of {}.” for PartImageNet. We then rank class embeddings by cosine similarity to the name embedding. The candidates are the 20 VOC object classes or 40 PartImageNet part classes, plus background. This evaluates the retrieved text through its embedding, rather than matching strings exactly or classifying directly from the adapter output.

Eligible regions and metrics. We evaluate validation occurrences of supported concepts that occupy at least four assignment-grid cells. We upsample each hard region to the 224 × 224 annotation grid by nearest neighbor and ignore void pixels. For inclusion in the primary metric, the dominant ground-truth label must be foreground and cover at least 75% of the region’s valid pixels. This focuses evaluation on regions with a clear annotated label.

Let $R _ { n } ( x )$ be the rank of that label when retrieved from the predicted name, and let ${ \mathcal { O } } _ { n }$ contain eligible validation images for concept n. We average first over occurrences of each concept and then over the set C of concepts with at least one eligible occurrence:

$$
\begin{array} { c } { \mathrm { A c c } _ { \mathrm { n a m e } } = \displaystyle \frac { 1 } { | \mathcal C | } \sum _ { n \in \mathcal C } \frac { 1 } { | \mathcal O _ { n } | } \sum _ { \boldsymbol x \in \mathcal O _ { n } } \mathcal k [ { \boldsymbol R } _ { n } ( { \boldsymbol x } ) = 1 ] , } \\ { \mathrm { M R R } = \displaystyle \frac { 1 } { | \mathcal C | } \sum _ { n \in \mathcal C } \frac { 1 } { | \mathcal O _ { n } | } \sum _ { \boldsymbol x \in \mathcal O _ { n } } \frac { 1 } { { \boldsymbol R } _ { n } ( { \boldsymbol x } ) } . } \end{array}\tag{22}
$$

Top-1 accuracy measures how often the correct label ranks first, while mean reciprocal rank (MRR) also credits labels retrieved below first place. Tables 21 and 22 report $1 0 0 \mathrm { A c c } _ { \mathrm { n a m e } }$ and MRR, together with the eligible region and concept counts. Those subsets differ across concept models, including for the pooled-teacher reference. No human or language-model judge is used. The text encoder both selects names and scores their correspondence to class labels, so the metric is not an independent semantic judgment. The confidence intervals measure variation across sampled concepts, not across training seeds.

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet / DINOv2</td><td colspan="2">PIN / DINOv2</td><td colspan="2">PIN / CLIP-DINOiser</td></tr><tr><td>Top-1</td><td>MRR</td><td>Top-1</td><td>MRR</td><td>Top-1</td><td>MRR</td></tr><tr><td>Region mean</td><td>15.28</td><td>0.2764</td><td>23.18</td><td>0.3623</td><td>26.88</td><td>0.4133</td></tr><tr><td>Direct key</td><td></td><td></td><td></td><td></td><td>31.95</td><td>0.4486</td></tr><tr><td>Linear key</td><td>12.29</td><td>0.2468</td><td>22.95</td><td>0.3598</td><td>27.40</td><td>0.4245</td></tr><tr><td>MLP key</td><td>14.82</td><td>0.2722</td><td>22.29</td><td>0.3544</td><td>26.30</td><td>0.4074</td></tr><tr><td>Additive</td><td>16.04</td><td>0.2905</td><td>22.53</td><td>0.3651</td><td>30.71</td><td>0.4534</td></tr><tr><td>Joint MLP</td><td>23.74</td><td>0.3691</td><td>21.74</td><td>0.3553</td><td>26.86</td><td>0.4136</td></tr><tr><td>Pooled teacher (reference)</td><td>29.75</td><td>0.4314</td><td>26.02</td><td>0.3959</td><td>30.97</td><td>0.4544</td></tr></table>

Table 21: PartImageNet name-to-class retrieval on validation regions. Top-1 is a percentage, while MRR lies in [0, 1]. Both are macro-averaged over concepts. Column groups identify concepttraining dataset / backbone. Adapters are fitted independently on PartImageNet training images. Eligible region / concept counts are 5,541 / 467, 4,758 / 416, 1,208 / 171, respectively. The teacher reference pools features from each model’s own regions.
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet / DINOv2</td><td colspan="2">PIN / DINOv2</td><td colspan="2">PIN / CLIP-DINOiser</td></tr><tr><td>Top-1</td><td>MRR</td><td>Top-1</td><td>MRR</td><td>Top-1</td><td>MRR</td></tr><tr><td>Region mean</td><td>30.09</td><td>0.4643</td><td>41.28</td><td>0.5490</td><td>39.96</td><td>0.5531</td></tr><tr><td>Direct key</td><td></td><td></td><td></td><td></td><td>44.97</td><td>0.5878</td></tr><tr><td>Linear key</td><td>27.90</td><td>0.4384</td><td>41.45</td><td>0.5515</td><td>44.32</td><td>0.5794</td></tr><tr><td>MLP key</td><td>29.13</td><td>0.4537</td><td>41.54</td><td>0.5512</td><td>39.40</td><td>0.5481</td></tr><tr><td>Additive</td><td>41.56</td><td>0.5515</td><td>54.07</td><td>0.6396</td><td>58.50</td><td>0.6895</td></tr><tr><td>Joint MLP</td><td>53.14</td><td>0.6438</td><td>61.54</td><td>0.7004</td><td>58.17</td><td>0.6936</td></tr><tr><td>Pooled teacher (reference)</td><td>73.82</td><td>0.8128</td><td>80.21</td><td>0.8630</td><td>73.33</td><td>0.8103</td></tr></table>

Table 22: VOC2012 name-to-class retrieval on validation regions. Top-1 is a percentage, while MRR lies in [0, 1]. Both are macro-averaged over concepts. Column groups identify concept-training dataset / backbone. Adapters are fitted independently on VOC training images. Eligible region / concept counts are 7,567 / 633, 7,088 / 368, 3,576 / 190, respectively. The teacher reference pools features from each model’s own regions.

## H.4 RESULTS

Object names on VOC. The joint MLP improves top-1 retrieval over region means by 23.05, 20.26, and 18.21 percentage points for ImageNet/DINOv2, PartImageNet/DINOv2, and PartImageNet/CLIP-DINOiser, respectively (Table 22). Paired concept-bootstrap 95% intervals are [20.35, 25.48], [16.86, 23.56], and [13.75, 22.33] points, using 1,000 resamples. The additive adapter performs slightly better than the joint MLP for PartImageNet/CLIP-DINOiser (58.50 versus 58.17). Attribute-dependent adapters therefore improve object-name retrieval over fixed region-mean names in all three settings.

Part names on PartImageNet. The gains do not extend uniformly to part labels (Table 21). The joint MLP improves over region means by 8.46 points for ImageNet/DINOv2 (95% interval [5.96, 10.83]), but changes PartImageNet/DINOv2 and PartImageNet/CLIP-DINOiser by −1.44 and −0.02 points, with intervals [−3.22, 0.40] and [−3.53, 2.98]. Direct CLIP-DINOiser prototypes remain competitive. Better agreement with the teacher also need not yield better part names: for PartImageNet/DINOv2, joint-MLP cosine similarity to the pooled teacher increases from the region mean’s 0.8169 to 0.8615, while top-1 retrieval falls from 23.18 to 21.74. The pooled teacher itself achieves only 26 − −31% top-1 on PartImageNet part labels.

## H.5 QUALITATIVE EXAMPLES

Gallery construction. For each supported concept, the gallery contains its closest qualifying validation image and up to nine additional, distinct images sampled reproducibly. A displayed occurrence must be among the image’s K selected concepts and occupy at least 1% of the assignment grid. The closest image minimizes the prototype distance $\lVert z _ { u } - \bar { w } _ { n } \rVert _ { 2 } ^ { 2 }$ from Equation 1 over image-grid locations. The gallery separates fixed names from attribute-dependent names and includes pooledteacher predictions, with five candidate names available per method. The 1% threshold applies only to visualization. Quantitative evaluation retains the four-cell threshold above.

Selection for the figures. Figures 53 and 54 each show ten concepts. To select them, we exclude background-associated concepts by requiring at least 85% mean foreground overlap across their saved gallery examples and at least 75% in each of the five displayed examples. We then inspect contact sheets to select concepts with diverse classes, excluding rows with potentially inappropriate displayed labels. Each row retains the closest example and four random examples. The figures illustrate differences between methods rather than estimating naming accuracy. All displayed names are unedited top-one predictions.

Additional selected examples. Figures 55 and 56 apply the same support, foreground-overlap, label-screening rules to two additional model–dataset pairs. Each row retains the closest saved occurrence and four visually similar examples chosen manually from the remaining saved gallery images.

<table><tr><td>Concept-level names</td><td>Closest</td><td>Random 1</td><td>Random 2</td><td>Random 3</td><td>Random 4</td></tr><tr><td>Concept 437 Direct: N/A</td><td><img src="images/b06c8c380cbfcd577882c3025b22867fc757e07fa347bd7f8466af457a4f2e67.jpg"/></td><td><img src="images/70a5662301843448b3a59e776911d581948648dde9ec256e8763c468afb057e5.jpg"/></td><td><img src="images/74046bd69251c11b9dfae19fbac62b26d2b9d24a93ba93051579c69b1b648b93.jpg"/></td><td><img src="images/3af54cd3fbab3c14bf112c76a92c536f25600630c816322895a74fd0eba22451.jpg"/></td><td><img src="images/ff765c2c9b13d32941e01ac9806232f94395361e1ad021c0447791f5a105b1a7.jpg"/></td></tr><tr><td>Mean: car images</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear: car images</td><td>A: car images</td><td>A: car images</td><td>A: car images</td><td>A: car images</td><td>A: car images</td></tr><tr><td>MLP: car images</td><td>J: car images T: alloy wheels</td><td>J: car images T: car images</td><td>J: car images T: alloy wheels</td><td>J: car images T: alloy wheels</td><td>J: motorcycle wheel T: wheel motorcycle</td></tr><tr><td>Concept 600</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct: N/A</td><td><img src="images/68ff4456c98307e6990c14b6685c33c30a1b4f720913d5333b3043a492b51699.jpg"/></td><td><img src="images/de1d4efcebf3c843b5fa76d9b07f843ff1da48e4e3a738e4b2bda69707a071ba.jpg"/></td><td><img src="images/a8c537e1a8d3eb2ee2b0fbf8237b0c4e38ad1856d7db13d4fac4ca6bcc27518f.jpg"/></td><td><img src="images/28a0684810f667141635a391a761a438128463dc9d07362d5d7e18f7918d9603.jpg"/></td><td></td></tr><tr><td>Mean: bus service</td><td></td><td></td><td></td><td></td><td><img src="images/22f3c8f5c46a0b297dcd2cad0cbc223fa0aea5e64c56e6789d1f612759f02239.jpg"/></td></tr><tr><td>Linear: bus route</td><td>A: courtesy bus</td><td>A: city bus</td><td>A: train at</td><td>A: city bus</td><td>A: train at</td></tr><tr><td>MLP: bus service</td><td>J: courtesy bus T: courtesy bus</td><td>J: city bus T: trolley bus</td><td>J: train at T: train at</td><td>J: bus service T: city bus</td><td>J: train at T: train at</td></tr><tr><td>Concept 467</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct: N/A</td><td><img src="images/b920b2c4990255f3876aa99a2de2f9922d81b27efd72ecba03e33610b554170c.jpg"/></td><td><img src="images/a06f1f80569a8d2b673d4f147b7a886087160c61171686cd565d65fac3062927.jpg"/></td><td><img src="images/7c5e824f758923166e75e5b6197b77c5bbada5c522c730a22cc46d8f2ce5a700.jpg"/></td><td><img src="images/4bb013c2027f8b2201c1896400b382d48ec2465fed509da1a98a8f89f75e73c1.jpg"/></td><td><img src="images/385924bbe9059acf89e1fdd5d5eeb04590328fd67be63f34c96d1e8ce750daab.jpg"/></td></tr><tr><td>Mean: train at</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear: train at</td><td>A: train at</td><td>A: train at</td><td>A: train at</td><td>A: train at</td><td>A: train at</td></tr><tr><td>MLP: train at</td><td>J: railroad train T: rail vehicle</td><td>J: railroad train T: railroad train</td><td>J: railroad train T: railroad company</td><td>J: train at</td><td>J: railroad train T: railroad train</td></tr><tr><td>Concept 313</td><td></td><td></td><td></td><td>T: train at</td><td></td></tr><tr><td>Direct: N/A</td><td><img src="images/e85f9cf9db3f57e83f2663bfb2afcdb43447d6038d69efee6d38cf574d9db0ef.jpg"/></td><td><img src="images/9cafee3aee72a9ebe4ef19b100336664c1c26fd23fafc482f6a5f796ef6f53d3.jpg"/></td><td><img src="images/df9c4e0a8f3bc71e938e0106a5223120b52a2be105eebb437c769f29bb861c1c.jpg"/></td><td></td><td></td></tr><tr><td>Mean: these</td><td></td><td></td><td></td><td><img src="images/e4f35a51760f2c1ef7d94a42e35884cc928c602d4c02f43496a39d9c45a830b4.jpg"/></td><td><img src="images/c40f036020b0f3e40997f5849f014c2673adfacd1136c64e4b7c9c8958611b3e.jpg"/></td></tr><tr><td>Linear: these</td><td>A: recent J: latest</td><td>A: these</td><td>A: bottle out of</td><td>A: these</td><td>A: these</td></tr><tr><td>MLP: these</td><td>T: bottle label</td><td>J: orig T: bottle out of</td><td>J: bottle out of T: drink driver</td><td>J: these T: bottle out of</td><td>J: bottle out of T: bottle out of</td></tr><tr><td>Concept 608</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct: N/A Mean: recent</td><td><img src="images/583e5795496c867ae21bfc49639cd07f7ff8c49648080b5019e6c1499473aa2d.jpg"/></td><td><img src="images/95576b7b65905c05c846faf233eb2059857ba1f773797dff328edda0da55dd28.jpg"/></td><td><img src="images/cb6b1b60a95677234ab7bd4331b004272d2df182640aec0464697ac3fea96438.jpg"/></td><td><img src="images/f503e4fb21ccea8eed46994e8b59e66fde6f9aeb544762219b1fca4d85c161da.jpg"/></td><td><img src="images/684bfff4191c45164442f17f037b3ff41632b5d0b673a1c65fb3017097ed3479.jpg"/></td></tr><tr><td>Linear: haired</td><td>A: baby hat</td><td>A: baby hat</td><td></td><td></td><td></td></tr><tr><td>MLP: recent</td><td>J: recent T: asian couple</td><td>J: recent T: frat</td><td>A: baby hat J: recent</td><td>A: baby hat J: recent</td><td>A: baby hat J: recent</td></tr><tr><td>Concept 559</td><td></td><td></td><td>T: two sitter sofa</td><td>T: baby bed</td><td>T: toddler girl</td></tr><tr><td>Direct: N/A</td><td><img src="images/1e5a60fc256d1138a28b8d423c9edd6cd1c5f77775e41767af2089ad832ca8f0.jpg"/></td><td><img src="images/d487adf3bd6bfbb986a9f0c24c8d59a5342aaeee8d76988c8c667965949fd745.jpg"/></td><td><img src="images/fccfd94efd8c219d47d4a718765dccd7549814a73b7048fe87116bc4bd99a39a.jpg"/></td><td><img src="images/075e0fbd57c130cbf090fa9389eb471e361c3c2d2a45aec8017746676e44a658.jpg"/></td><td><img src="images/5e92972757745d00191fe922903ffc67686fc2e5f398461c350e4fcbde8ae964.jpg"/></td></tr><tr><td>Mean: house cat Linear: house cat</td><td>A: cat</td><td>A: house cat</td><td>A: house cat</td><td>A: house cat</td><td>A: house cat</td></tr><tr><td>MLP: house cat</td><td>J: house cat T: burmese cat</td><td>J: house cat T: house cat</td><td>J: house cat T: house cat</td><td>J: house cat T: house cat</td><td>J: house cat T: house cat</td></tr><tr><td>Concept 562 Direct: N/A</td><td><img src="images/78968d5a8c269bffab1e4ce7b87c22d16b37e089fde65ebdafc5f60ca453a00e.jpg"/></td><td><img src="images/ef59c701f4694f16946ee92300c674352990fa56c156b471a403f2774dc0e6d0.jpg"/></td><td><img src="images/4618ac4df63117d8a7baaac22b1827796b3bee4923cafeb754eb54f149f8a756.jpg"/></td><td><img src="images/28f186ffe4d28e5ec0f28a8001c4bc56cfe4c9979956010b59b0dce7e08ac896.jpg"/></td><td><img src="images/6d4de672df2b1bb689556338cf16f1ed5d625b11bce44cdcc1fa506d32fa28d1.jpg"/></td></tr><tr><td>Mean: dog breed Linear: dog breed</td><td>A: dog breed</td><td></td><td></td><td></td><td></td></tr><tr><td>MLP: dog breed</td><td>J: french bulldog T: pug</td><td>A: dog breed J: dog breeding T: dog breeding</td><td>A: dog breed J: dog breed T: dog breed</td><td>A: dog breeding J: dog breeding</td><td>A: dog breed J: dog breeding</td></tr><tr><td>Concept 659</td><td></td><td></td><td></td><td>T: english bulldog</td><td>T: shih</td></tr><tr><td>Direct: N/A</td><td><img src="images/67ba30ddb24fa2e59dfcbb3ded4c369ea97af5801e33b6f67ab9a1b75df8bd2e.jpg"/></td><td><img src="images/21ec2e3bc6d1345afda587668d1a94e2ff5bb6d4ec82d3ce0a72882d64ac391a.jpg"/></td><td><img src="images/3d1d4c82c3ce4d01af5efcd80fbbee6d2ed85410f0db7429f0b8a6cc6e7b9c23.jpg"/></td><td><img src="images/0a7976a985203eb367862f00dc5fe6169a0acca252ca1febfeafe5538f776e54.jpg"/></td><td><img src="images/f094a92658a6a528c18e753fdc4dc32d78f6c58c41c3b540b60c09d25ce2ca5d.jpg"/></td></tr><tr><td>Mean: farm animal Linear: dog breeding</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP: farm animal</td><td>A: cattle breeding J: farm horse</td><td>A: bovine J: farm animal</td><td>A: dog breeding J: dog breeding</td><td>A: dog breed</td><td>A: farm animal</td></tr><tr><td></td><td>T: horse&#x27;s mouth</td><td>T: cow head</td><td>T: dog breed</td><td>J: dog breed T: dog breed</td><td>J: farm animal T: farm animal</td></tr><tr><td>Concept 1022</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct: N/A</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td><img src="images/795554982beda8bbc92acfac01e005a556febf2e2da3d58aa5d9d7d72e17b901.jpg"/></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td><img src="images/8fa9d2ffbecce8fb4dd647e921e5c4791902851bae8096003f59b985461fbd13.jpg"/></td><td><img src="images/77ec2d4d15845bbb0616ac6ece06c8bfbff15d652526961b69cb75fc5f255faf.jpg"/></td><td></td><td></td></tr><tr><td>Mean: farm animals</td><td></td><td></td><td></td><td><img src="images/29e7351fe34dc4e1307b25ce6a4153390872b66c129432d25cbf69da095b00f2.jpg"/></td><td><img src="images/b22d8eefbd760425f84df7dfea730f88834c05c73d3d662a9b53ad145674f374.jpg"/></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear: cattle breeding</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>A: farm animals</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>A: sheep</td></tr><tr><td></td><td></td><td></td><td>A: farm horse</td><td></td><td></td></tr><tr><td></td><td></td><td>A: farm horse</td><td></td><td></td><td></td></tr><tr><td>MLP: farm animal</td><td></td><td></td><td></td><td>J: baby sheep</td><td></td></tr><tr><td></td><td></td><td></td><td>J: farm horse</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>J: lamb wool</td></tr><tr><td></td><td>A: farm horse J: farm horse</td><td>J: farm horse T: pony trekking</td><td>T: farm horse</td><td>T: sheep pen</td><td>T: sheep</td></tr><tr><td></td></table>

Figure 53: PartImageNet-trained DINOv2 concepts named on VOC (selected examples). Each row shows the closest occurrence and four saved random examples. Left: fixed direct-key, regionmean, linear-key, and MLP-key names. Direct CLIP naming is undefined for DINOv2 (N/A). Below each image: A, additive, J, joint MLP, gray T, teacher.

A: american eagle J: american eagle T: american eagle

A: mouth watering J: mouth-watering T: bigmouthed

## Concept-level names

Concept 676 Direct: googly eyes Mean: googly eyes Linear: photo shows MLP: googly eyes

Concept 653 Direct: loud mouth Mean: loud mouth Linear: loud mouth MLP: loud mouth

## Concept 419

Direct: puffer fish Mean: fish species Linear: fish species MLP: fish species

Concept 884 Direct: food fish Mean: sport fish Linear: sport fish MLP: sport fish

Concept 445   
Direct: tortoiseshell turtle Mean: tortoiseshell turtle Linear: turtles   
MLP: tortoiseshell turtle

Concept 615 Direct: bald eagle Mean: american eagle Linear: american eagle MLP: american eagle

Concept 109 Direct: squirrel Mean: squirrel Linear: squirrel MLP: squirrel

Concept 190 Direct: leopardprint Mean: leopardprint Linear: leopardprint MLP: leopardprint

Concept 166 Direct: tire rim Mean: car images Linear: car images MLP: car images

## Concept 76

Direct: farm machinery   
Mean: tractor   
Linear: tractor   
MLP: tractor

![](images/241e959101fa698f784052debc99f3c90e8c54facfa2004cfb2382450b1953b4.jpg)

A: googly eyesJ: googly eyesT: googly eyes

![](images/0a3e7397ba6cc368730fb9fa304ef0d40414360f543f1ef78500360d87fd31e1.jpg)

A: mouth watering J: mouth-watering T: loud mouth

![](images/bad77c1a80ee86f015a20a93a09b3f2d2fcfa0af5d8e1cc53dd5243d2ae8b66a.jpg)

A: fish species J: fish species T: fish species

![](images/85e793494eebb314ee14282632af569104cda91bcde33ab72d5306ed02a9bc24.jpg)

A: fish for J: sport fish T: sport fish

![](images/e55ad2b9dd743ce32e6e7559f0b2dcd325e57a8307bfa5942b8db6aa2b6bad9f.jpg)  
A: tortoiseshell turtle J: tortoiseshell turtle T: turtles

![](images/0cb3e0de947ce1198721e1cd13f41363a34e54f94b327882c8a2c88ef528446d.jpg)

A: bald eagle J: american eagle T: american eagle

![](images/1f26a771d7fe5c596deba1a73ea66a18ab7d611d6a80a830c978d0eb65800d5b.jpg)

A: red squirrel J: red squirrel T: squirrel

A: car images J: car images T: car

![](images/1fd9ae864021397ed2dd9946981e114abf0044f8e6da6fa2ef2b5e933337c6f5.jpg)  
A: tractor J: tractor T: tractor

## Random 1

![](images/e29a2a1437d105f94d0cfcb96e526968cc9dc3035177ed4311e2af91a9b3c9ef.jpg)

A: leopardprint J: leopardprint T: leopardprint

A: googly eyesJ: googly eyesT: set eyes on

Random 2  
![](images/b0c7b29dc10caa8f118246f775384b15831e9728c951909f9e7bb2497350b519.jpg)

![](images/6f2928d2dda62da0a2a779c99e4163bdbf2fd1b5a67e3233998726e2f1f46f3a.jpg)

![](images/3c5d6656614e5bd25361bcf6f0aa3043215dac33262b017e84173677b6f7b2ee.jpg)

A: wolf dog J: crying wolf T: loud mouth

A: with an eye to J: with an eye to T: googly eyes

![](images/91e02e36a3b247579760a4a00d116518f70a7f238920ef1a00848296d4c3703e.jpg)

![](images/6fa030443aa9e61505b4e0699bcacfa6a69d86e82575714bcbccd7f1b44f7afe.jpg)

A: fish species J: fish species T: fish species

![](images/b8bf027b8d419b56de3c4aeaff93f1ac8a55fdad067b3964e1b21b00df20545e.jpg)  
A: tortoiseshell turtle J: tortoiseshell turtle T: soft-shelled turtle

A: mouth watering J: loud mouth T: dog breed

![](images/cfea23df1436201ecce916fbdd6cb70f4383cf860bc6e49a0fb04b7f24c64142.jpg)

A: sport fish J: fish for T: rainbow trout

![](images/c848815847facddd49fd60c9ad3b637c2a131b02583a7ccb7e32a7065025a7b9.jpg)

![](images/0d55b4e5f2fb51ff68dfd0dd496aeef1a0d24626ff35bda8d97235eed9856d2e.jpg)

![](images/b2dec08186872faea1bfdc94e76063aaee8ad18ee9e2353555dd40c5ab86565f.jpg)  
A: turtles J: water turtle T: soft-shelled turtle

A: fish species J: fish species T: sea toad

![](images/b64fb6ba50f7df52daa1b819a2fc4b88ac7db0dd244d9ed1d6ee1407ddbcc71d.jpg)

A: freshwater fish J: freshwater fish T: trout

![](images/0cc15fddbd4b571d1e193d005c65c21a8636cc3a6eeb3a9a60ffd2c7fa1c6b29.jpg)

![](images/ebdff18f8e1d8d44d1165c2467adaa8b04f5415d3775ac11fea8e2cf38832d8b.jpg)  
A: have my eye on J: hairy eyeball T: with an eye to

![](images/da6355e5400692fc656d3a8282c7d180e7d3d3e690c165c1154e12896325d42f.jpg)

![](images/83a04ab20841c91abffb40c8ac44e1fe4b47ef2f580c96387c48ccb376318682.jpg)  
A: truck farming J: truck wiring T: plow blade

A: bird of prey J: eagle nest T: fish eagle

A: crucifix fish J: fish for T: fish meal

## Random 3

A: freshwater fish J: freshwater fish T: fish for

![](images/2345e12d255a989811d44bfc063a99cd71760a466aa011e7745f5d427708e624.jpg)

![](images/33f8250480a2777c6db5f97dbbb8832af07d310f007a0281eec4a7433896f1e1.jpg)

![](images/5e0db29a4e3b3c5e402a4b83404807cb680926302f8194b1c2972be826e07876.jpg)

![](images/cae32e0cb026ebe7314a779c4a08223f01852bce697471939a6da86d32c1c362.jpg)

![](images/386a06da34e3c620428138e640954272b467195408e101502ea2cc4e539a474a.jpg)

A: car images J: car images T: car images

A: fox squirrel J: squirrel T: squirrel

![](images/0b916000b6125a5a4555a10805470a36eb3680eeb94c3b5eafc2e417c2f7e5eb.jpg)

![](images/d6081323b4a6bc683c8709253854b3a14bad8431722eccdbb0ec841693c824d4.jpg)  
A: leatherback turtle J: sea turtle T: leatherback turtle

![](images/e9870a377e22316d307418de475a8ff5b7a6f2c3eb51225263f4ad943641e4d3.jpg)  
A: soft-shelled turtle J: sea turtle T: female of the species is more deadly than the male

A: leopardprint J: leopardprint T: leopard print

![](images/b07be9ccc3335a0c96482050b254c862918caff5305023ba0e17e643c083c602.jpg)

![](images/5d49cd9e60638731135ad4738ed3fed11eab8f1a7c26cf054b8171b5c1da789f.jpg)  
A: truck farming J: truck farming T: truck farming

A: freshwater fish J: fish for T: lake trout

![](images/a0c64ce7d086e5ff1b2bfe9fa573f275a89cb055efd786a9b15ff950197d3df6.jpg)

A: squirrel J: squirrel T: tree squirrel

A: bald eagle J: bald eagle T: bald eagle

![](images/3c3957947b57c22bdbd83b45df9c8f6d24ac0f89f1eca6d8a754a6530ac2145e.jpg)

![](images/b62d35ec33170b80176d34c95b388144356dadbc6171819ba97eca7e6274dc1c.jpg)

![](images/637e711d0413b3dd6b3db06c87d068e6fec812c3fe2b04ae053dce48ae914b47.jpg)

![](images/cb58a5a052dde83c3ad523af329ce0a4d90166b46457332373cc4a0ec0d660fd.jpg)  
A: american eagle J: american eagle T: american eagle

A: crucifix fish J: fish for T: fish for

![](images/405c67735af0a3150f067d30d8cfda4c327bd76bc1cff1dc5ee1a4b4d4eb2221.jpg)

A: big cat J: leopardprint T: big cat

A: car images J: car images T: suv

![](images/f46eeaedfb3ec4dcbd870036860241e92389d58f1acd0e07ce7608776b01908b.jpg)  
A: mouth watering J: mouth watering T: dog breed

## Random 4

A: squirrel J: tree squirrel T: squirrel

![](images/a267dcfb466da2cb2affa3c8c09fe5801226610128f75dd305a88880f0e2083b.jpg)

![](images/4e87c986b732d1e9ba950e3191ea6724df6ca2c892f06f69aa60284860cd9c13.jpg)  
A: with an eye to J: rare bird T: game bird

A: leopard J: leopard T: leopard

![](images/83a10d52a47658052b0d38c5428b1dacdc96b9ce31b36acff6ca86340e17d0f1.jpg)

![](images/d5a1b6751b51ae96d3e28763dc6d4e83180de086efd09c700ff17783b9dc8317.jpg)

A: car images J: car images T: car images

![](images/d2858efb2e28637dd17406250c0237378f443bad521c0f3e271d0472e56d96e4.jpg)

![](images/8eff224b43c37670d2cbf368168b714a40a6c1ad75b747e376682eb6c91b046d.jpg)  
A: truck farming J: truck farming T: truck farming

A: squirrel J: squirrel T: squirrel

![](images/d9de483ae85c794067f0a9b43e9af15da5e4977ae4f6986d3113e0c0cd3dad18.jpg)  
A: big cat J: big cat T: leopardprint

![](images/2aabb12120469472512dbb85ef8ec7cf04357af27d86f5b03c67f006711e83ba.jpg)

A: car images J: car images T: alloy wheels

![](images/a1ac2cdf4957491c22a5fb0d8dee25880c3511ace191643365c488afaf4ec18f.jpg)  
A: two-wheeled tractor J: two-wheeled tractor T: two-wheeled tractor

Figure 54: PartImageNet-trained CLIP-DINOiser concepts named on PartImageNet (selected examples). Each row shows the closest occurrence and four saved random examples. Direct names use the compatible projected CLIP prototype space. Labels denote A, additive, J, joint MLP, gray T, teacher.

Similar 2

A: farm horse J: farm horse T: farm horse

A: peta J: dog breeding T: dairy cow

A: house cat J: house cat T: cat food

## Concept-level names

Concept 166 Direct: tire rim Mean: car images Linear: car images MLP: car images

## Concept 432

Direct: trolley bus Mean: trolley bus Linear: trolley bus MLP: trolley bus

Concept 67 Direct: bull nose Mean: dog breeding Linear: dog breed MLP: dog breed

Concept 75 Direct: house cat Mean: house cat Linear: house cat MLP: house cat

Concept 261 Direct: cattle breeding Mean: cattle breeding Linear: cattle breeding MLP: cattle breeding

Concept 92 Direct: sweatshirt Mean: outerwear Linear: outerwear MLP: outerwear

Concept 374 Direct: avian Mean: avian Linear: avian MLP: avian

Concept 138 Direct: baby sheep Mean: sheep Linear: sheep MLP: sheep

## Concept 23

Direct: trail riding Mean: horse riding Linear: horse riding MLP: horse riding

## Concept 1009

Direct: cycling jersey Mean: maillot Linear: maillot MLP: maillot

![](images/24786441eb495fcdb5906211c3a86c3b83a6498ea8660c9f80dde63e75f17047.jpg)

A: car images J: car images T: car images

![](images/1ccb5b61e517b48bd9d8c59bba9603da1c1efdde06135f0e3443a83e5af0c7b8.jpg)

A: buses J: bus service T: bus service

![](images/4e1060f9f00e0360ea1f7d8cccb28341236f3624c93713dda26b84b4d40bc3e2.jpg)  
A: dog breed J: dog breed T: dog breeding

![](images/392a4d3796cabdfa08c0befc73668cc467acc242d1ea9f1b49beb83a969e1f01.jpg)

![](images/1e1b51b51f1d06e106ec24f8ab7f84f8c9c0b9647dfc955a1dcc3c6d219e08d3.jpg)  
A: cattle breeding J: cattle breeding T: cattle breeding

![](images/a5bb801d2ffedd27d64984c13771284a7d24a7a73738cf75043d031bebc13e34.jpg)

A: outerwear J: winter jacket T: winter jacket

![](images/3cc001e015cbda929c24fd4e79076c5f24f8b447142e96c7a4dca0da19c4cf9f.jpg)  
A: suvs J: sport utility vehicle T: suv

A: avian J: avian T: myna bird

A: baby sheep J: sheep T: baby sheep

![](images/c8159f94342319d6e2557ddc676ecaa74e22ba8b5184338902fd76266758bbc4.jpg)

A: horseback riding J: horse riding T: ride horseback

![](images/4ccc027d559b446f1f71b05f239eb51d1cce8c70a6b832c0c0bfe61d2a9ea1b4.jpg)  
A: cycling jersey J: maillot T: cycling jersey

![](images/f2952a7e947a20748501033cea777f596572314c2c40730ca6b8add94bb29108.jpg)

![](images/2756fadfeb8c06e1598d596eb7fc945ea32f4d291abc2bb223dfe60807a4c1e7.jpg)  
A: car images J: car images T: car images

![](images/ee1c5e7bbd51cfea5fd0cfa984968999565fb4761f668fbfc26f0a7655f84a98.jpg)

A: trolley bus J: city bus T: articulated bus

![](images/89616558c72b92012eda11463c842d55dd3a91bff99880115108db69eca7d62a.jpg)

![](images/16ac8e1892f83af65d2d218d6b81764f6654bd8e55e7dcd15663e0fba61afc9a.jpg)

![](images/cf0aacb6cb159d7a7f7d9e85021f0d9e86292a65048b7d1d347162895c7e435c.jpg)

![](images/07b1a7839ac43efb56c72e0df3a53e921284ab173927065234dc7037508b2735.jpg)  
A: cattle farm J: cattle farm T: cattle farm

A: bus service J: bus service T: city bus

![](images/9bbee034ed2f26183ccea1069b242c93bbd510b80cd2e00de893b0e473f257bf.jpg)

![](images/37d4c95630664ab2b3faf1677c4987cbf14bce50cfb15a6bf111619965a0aa9e.jpg)

![](images/a1b9f645fd121b4accfb201069c9b08ddcd7df80c5866e36da3e5ec811b04992.jpg)

A: dog breed J: dog breed T: dogfriendly

A: outerwear J: outerwear T: body warmer

A: house cat J: house cat T: house cat

![](images/c1a4d16ef2921ef787aa9c351ac3f546c0a9e025f2f91587561a3690ed65b111.jpg)

![](images/df6e3fb135f79f9060a317290ece5ef1d096cfe500e895f960929b06f51c31f7.jpg)  
A: riding school J: pack riding T: livery driver

A: bird food J: avian T: mallard

A: ride horseback J: horse riding T: cattle ranch

![](images/ad561073d195383a66dc9121858e814cbf0e63c2644f9d5e1f414ebaacaff8c3.jpg)

A: winter jacket J: outerwear T: sweatshirt

![](images/18bae08ab6295ed0691089e65f1568006746de76481511164194c729f62d81e7.jpg)

![](images/994a1f05d32d73d89c9540058edcfd97e90a79e4b26de054a0e8c6c2a3b9706f.jpg)  
A: car images J: car images T: alloy wheels

A: lamb wool J: lamb wool T: lamb wool

![](images/9323481992608e08547697df57e105f4bc61ade208250fad271bcfe260523b60.jpg)

![](images/3786cb44141bfa07f637ed1a902783d5afb5fb73fe6ebb2f77fe87f5586f4a4b.jpg)

A: rare bird J: macaw T: macaws

![](images/2fad7c5629cd3c34d5184fc4bdf79bb850e2fc509249e43c1e6611adebdc8a3e.jpg)

![](images/c8e8604dd9e49c921ede3655ea48bdf40ec23ae96a84e0ce2b8d96e29f776925.jpg)

A: trolley bus J: city bus T: city bus

![](images/56cb4add38c121a4757629e2122713d265f85a1cf28e5f8c2fb8573e8e8833cf.jpg)

A: sheep   
J: sheep   
T: merino sheep

![](images/34ca79e21065004a87840a296853eb4622c75be9450752355eb8189c7a9282d0.jpg)

A: horseback riding J: horse riding T: race horse

![](images/f28bb20f83c06cb446325f4f38b6f682fb9d3702743832c048725f03ba2251db.jpg)

![](images/fecbc7bedcc617de661579a8c5fc451aa367a21ae836da19e7fb40a14f9fcb2c.jpg)  
A: cycling jersey J: these T: cycling jersey

A: dog breed J: dog breed T: dog breed

![](images/4e3587d258240a1540f6f8959d440a010d5028dab865d4ca7065343eb2352830.jpg)

A: burmese cat J: house cat T: burmese cat

![](images/497a191af0270a7765ac0b42e32b47a36da9a3bb13b433d04f140658c112dca0.jpg)

A: beef cattle J: bovine T: beef plant

![](images/702fe4ddc3c59634d1c3b1f380a234775a79366fad01f95fdd8724cb0f000584.jpg)

A: winter jacket J: outerwear T: waterproof jacket

![](images/3f903ba5fc66add3290728f40e03a1f86a11795e57d8c81be2548a778143bd5e.jpg)

![](images/5b771ef543055e6d5afec7a62631fe4a9d636e0ec102e5f6c5f42962b34505d7.jpg)

A: rare bird J: parrots T: cute birds

![](images/e8e483455874e5a3f5c4eabcb631afea7cfa62639dd0fef94beab1bcaa58a4d7.jpg)

A: sheepsJ: sheepsT: sheeps

![](images/a14cdf0d28c0a28947b50c31e5f2aa12ecacbd0eebe456292d2823b3f05e4cef.jpg)

A: horse riding J: horse riding T: horseback riding

![](images/2f2ea8d701f5958b658b2a9515931c53e92482d4a8319b74611e2bd9713b68f8.jpg)  
A: cycling jersey J: cycling jersey T: cycling jersey

## Similar 4

![](images/b33a18bcdc92088d7b7e8178324a3b075daff37495b7e0e6aea5f5511578b1b8.jpg)  
A: car images J: car images T: car images

![](images/60d8050e78a4c32a621ef819867ad166579d811eb5aa3d72116bfe39816253c3.jpg)  
A: bus service J: bus service T: bus service

![](images/43163a6248cca315473a6fe081d636d6c191ad86b71886b664852a4379ed7f7b.jpg)

A: dog breed J: dog breed T: dog breed

![](images/8bc06b27d71370eca226ca0d36ec36493cc09e00974ca208b461f9a44e4f369b.jpg)

A: bear cat J: cat T: cute cat

![](images/b528985976341c9aa439d804c724c1f759c7cf86b2a9fbb379f8f33f0dafb42c.jpg)  
A: beef cattle J: beef cattle T: beef cattle

![](images/7aa06bf11a9b9948288d27b58fab0d9dbee100539119bec77fa073ac7b90649d.jpg)  
A: winter jacket J: outerwear T: toddler boy

![](images/dbb735c937b01b9a0adbe7eac49353b2bbb267e3ecde1a40863f60d9fcd7f38e.jpg)

A: rare bird J: rare bird T: avian

![](images/9fa74fb12de70b44d72f01e04884351817b5bdfa7f3ea52fddd94a4711036bf5.jpg)  
A: baby sheep J: sheep T: lamb wool

![](images/d38cfe9ff1c590be37f7c0cfd090bee20ce486f3b10fca884dea6a1e8fffc268.jpg)  
A: horse riding J: horse riding T: race horse

![](images/5205341e78b1b924475f50fa0d29c426475c42da696702f99fd2fde05a3affa4.jpg)  
A: jerseys J: jerseys T: cycling jersey

Figure 55: PartImageNet-trained CLIP-DINOiser concepts named on VOC (curated examples). Each row shows the closest occurrence and four visually similar saved examples. Fixed names appear at left; per-image labels are A, additive, J, joint MLP, and gray T, teacher.

<table><tr><td>Concept-level names</td><td>Closest</td><td>Similar 1</td><td>Similar 2</td><td>Similar 3</td><td>Similar 4</td></tr><tr><td>Concept 82 Direct: N/A</td><td><img src="images/42b5e6747e09652c0988603be2376bc2a729a8b214dad9437eac4658d7fd3630.jpg"/></td><td><img src="images/b32dd3464123a75482cd2ceae1eefe7a83212ce72e52ba34f70ba0f6b9b6eec6.jpg"/></td><td><img src="images/34ff5e72ddaf4418db9eba71110b636242a3ea1d7c6ceb87c25b8ed6cb83cec4.jpg"/></td><td><img src="images/f719326fa4759dd393aadd46f026cd9ec11a48151343252a943696c37f36521f.jpg"/></td><td><img src="images/90bedb72d4c7b54c0ccfe564a3beb83655059cc9b7845b3dc24d315d35609fe5.jpg"/></td></tr><tr><td>Mean: dog breed</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear: dog breed MLP: dog breed</td><td>A: cute J: cute</td><td>A: dog breed J: dog breeding</td><td>A: dog breed J: dog breed</td><td>A: dog breed J: dog breeding</td><td>A: dog breed J: dog breed</td></tr><tr><td>Concept 231</td><td>T: dog breeding</td><td>T: dog breed</td><td>T: dog breeding</td><td>T: dog breed</td><td>T: dogfriendly</td></tr><tr><td>Direct: N/A Mean: dog breed Linear: dog breed MLP: dog breed</td><td><img src="images/925df9afe4e9050d05f85e4aa16dfdc9f9500752e6e037b2f300cb078207b513.jpg"/> A: dog breed</td><td><img src="images/9237ec8a7fb54a57db14ed2f68a342a8695ea54ac4207c8bf3e10f516198c79f.jpg"/> A: dog breed</td><td><img src="images/9830c5ee6204c8525e70dccec90cdbf6c93d1a676eb9f810f091790d2e70919b.jpg"/> A: dog breed</td><td><img src="images/b270c172e063de2a24abaa9068a2d4efa1114c83947b4dac9b0befe7c357a781.jpg"/> A: dog breed</td><td><img src="images/c08137303e9db0e57653dae99694e986d9670f298c254ef982d99de472d65107.jpg"/> A: dog breed</td></tr><tr><td>Concept 672 Direct: N/A</td><td>J: dog breed T: dog breed <img src="images/68276721ece9e11c8be76fc35f62f85a9d06e7c8138eba73a8d19d827b771879.jpg"/></td><td>J: dog breed T: dog breed <img src="images/4ce96de62bdccece5c8d196940d042b181f7a7b092e9ba4e78b19e59356e01b8.jpg"/></td><td>J: dog breed T: companion dog</td><td>J: dog breed T: dog breed</td><td>J: dog breed T: dog breed</td></tr><tr><td>Mean: monkey Linear: monkey MLP: monkey</td><td>A: animal lover J: monkey T: howler monkey</td><td>A: monkey J: monkey jacket T: baboon</td><td><img src="images/f830a110be0940f9a6fb7bef701bf7f5ffd697c90439e68938b1eb332efe8667.jpg"/> A: monkey J: monkey</td><td><img src="images/738c7f16fce6fe2bfdee41e8bab57927fd64e2dbc15836394676ee1be42c46e9.jpg"/> A: howler monkey J: howler monkey</td><td><img src="images/c9c6660aa30d5ce3a26757624ea50c4c649ef35d090e3677b9fd3d4475173ce5.jpg"/> A: monkey J: monkey jacket</td></tr><tr><td>Concept 400 Direct: N/A Mean: tortoiseshell turtle Linear: tortoiseshell turtle</td><td><img src="images/2d118a2b9316b777f9704300efe505d79f529c52b6ed915efea129a44d3d4304.jpg"/></td><td><img src="images/0193046dc1053202e758081eee9955d414d9939e5f49426a52d5361a5a6ba7db.jpg"/></td><td>T: endangered species <img src="images/1189284b532920173218265e4213fc83507ba47174a7e2d6e9147fa718b7458c.jpg"/></td><td>T: howler monkey <img src="images/02c26d8b5877acc0202ba8b503f2e8ed4a7e9a3695f568927b51c23934376700.jpg"/></td><td>T: monkey jacket <img src="images/4d0f67d3423bccbc8b6d7b7fea1fd25422271ce074919acb3bab5971713f12a8.jpg"/></td></tr><tr><td>MLP: tortoiseshell turtle Concept 507 Direct: N/A</td><td>A: tortoise shell J: turtle shell T: tortoise shell <img src="images/3a90bcdd1f617807933c54814613158585fa3a3a3013e8e2296e354e2334a5f8.jpg"/></td><td>A: tortoiseshell turtle J: tortoiseshell turtle T: tortoiseshell turtle</td><td>A: sea turtle J: sea turtle T: sea turtle</td><td>A: tortoiseshell turtle J: tortoiseshell turtle T: tortoiseshell turtle</td><td>A: tortoiseshell turtle J: tortoiseshell turtle T: tortoiseshell turtle</td></tr><tr><td>Mean: car images Linear: car images MLP: car images Concept 965</td><td>A: car images J: truck wiring T: veteran car</td><td><img src="images/f953b63bd33b7978585f3cac59e5898e839f8034d49c55ce027dfb9bfa827e70.jpg"/> A: car images J: truck wiring T: ford f</td><td><img src="images/72079db20919921239f5ee0d84d61ca509f5b07f2223552b66715c4782ee3810.jpg"/> A: courtesy car J: car images T: muscle car</td><td><img src="images/0d1fd2c6b36496706734360fb3c256e049ef2f5e5bf0fce13c5667aed1df5229.jpg"/> A: car images J: autos post T: palace car</td><td><img src="images/95a17f60f1fee5758b9c3deccc1371eb22fc109da425940139cfcc54fc7424c8.jpg"/> A: car images J: truck wiring T: truck wiring</td></tr><tr><td>Direct: N/A Mean: car images Linear: car images MLP: car images</td><td><img src="images/913b657c95e34461500b164bb7dcf893601d63a3df33f9dd79088a05f3ff2066.jpg"/> A: car images</td><td><img src="images/c1682df2b1f121896ba52fec1970ad3fd71d8a38b95c63d9f07e57c9bf2559fd.jpg"/> A: car images</td><td><img src="images/34a5ab69e11e8b5a68280917ad1d4e1cc9bd5483e08c80ccfe17704280251fc6.jpg"/></td><td><img src="images/96d2ec77870071b1944b1d167c872c97a5351e606978a901c50e7470ca329404.jpg"/></td><td><img src="images/78d5811e93dac02af2c8eef3ed38cacf58d1e3cc9146aac28869b886b9245538.jpg"/></td></tr><tr><td>Concept 623 Direct: N/A Mean: car images</td><td>J: car images T: car images <img src="images/9e8c2e48a578e4c046120ae25a5b3a7169b65cbfc6e0e555239bf154a14902cc.jpg"/></td><td>J: car images T: car images</td><td>A: car images J: car images T: car images</td><td>A: car images J: car images T: camper van</td><td>A: car images J: car images T: police van</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td><img src="images/ec96bd72d5bee83a5b1d19ecb2323bd61e8e3e868cff13dcc17a069ba8ed3d6f.jpg"/></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td><img src="images/a7392f1540f63085b21eca4adbba13e3a7098fb849827b52506f88f83fc22c41.jpg"/></td><td></td><td></td></tr><tr><td>Linear: car images</td><td></td><td></td><td></td><td><img src="images/d8bbe4eaeabe548ad1bcbf99145364550d093d9e3f6a1d0442d91aa609465be4.jpg"/></td><td><img src="images/81fba434bf4ab4bb721e1eb5ebd264d00de3a43ba1a99bf3e2e2a87bbac2c64f.jpg"/></td></tr><tr><td>MLP: car images</td><td>A: car images</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>J: car images</td><td>A: car images J: car manufacturer</td><td>A: car images</td><td>A: car images</td><td>A: car images</td></tr><tr><td></td><td>T: headlight</td><td>T: sports car</td><td>J: car images</td><td>J: car images T: car images</td><td>J: car images T: car images</td></tr><tr><td>Concept 617</td><td></td><td></td><td>T: car manufacturer</td><td></td><td></td></tr><tr><td>Direct: N/A</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mean: avian</td><td><img src="images/ddb29a1734ea283e1dc07d13ddbdfa1c6df4b4af7ba8e57f043aa622a86095d9.jpg"/></td><td><img src="images/2e7f3b9a5516ad9baad33262a9af4f92f97b5c47d6509c8907c9d1a924a980ef.jpg"/></td><td><img src="images/352c36a36d2dbe99669255f7c8757f5a79a573b5cab8a7a28a5ec1994eb2d99e.jpg"/></td><td><img src="images/022297265de886ba63f9cce7acacd4a540015ee32e9a3bb629fb43b5c3d5ddb2.jpg"/></td><td><img src="images/45d70ed04873a252bec83c0aff14d7a1508fed318c06f1df891fb91b87df00ad.jpg"/></td></tr><tr><td>Linear: avian</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP: avian</td><td>A: rare bird</td><td>A: avian</td><td>A: avian</td><td>A: rare bird</td><td>A: avian</td></tr><tr><td></td><td>J: rare bird T: avian</td><td>J: bird of prey</td><td>J: golden eagle</td><td>J: rubythroated</td><td>J: bald eagle</td></tr><tr><td></td><td></td><td>T: maori hen</td><td>T: bird of prey</td><td>T: bird poop</td><td>T: bald eagle</td></tr><tr><td>Concept 478</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct: N/A</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td><img src="images/9fb59ba64a8e39df7c5063f1b48edb52e038852e203fcbe8bbdc5cdd08febc81.jpg"/></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td><img src="images/89446ac2528c98f3f750a5441844286e105facd45c7780ae92fee71e8a905c09.jpg"/></td><td><img src="images/7675ba2f9a1032782a58f994d4f502ba043e089ebcfb3a54ddf616d62e500bc0.jpg"/></td><td><img src="images/b1e43453519b9312e1f2c0f2cffc0e596eb69765d8d0cf8422eeea69bc73982e.jpg"/></td><td></td></tr><tr><td>Mean: avian</td><td></td><td></td><td></td><td></td><td><img src="images/9567368bef8a3c089c384610766a846708c473ea2b750e62694736e5345ceaa2.jpg"/></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear: avian</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>A: bird of prey</td><td></td><td></td></tr><tr><td></td><td></td><td>A: game bird</td><td></td><td>A: bird watching</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>A: avian</td></tr><tr><td>MLP: avian</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>J: shorebird</td><td>J: golden eagle</td><td>J: shorebird</td><td>J: rare bird</td></tr><tr><td></td><td>A: bird watching</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>T: golden eagle</td><td></td><td></td></tr><tr><td></td><td></td><td>T: game bird</td><td></td><td>T: sandpiper</td><td>T: rubythroated</td></tr><tr><td></td><td>J: rare bird</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>T: bird watching</td><td></td><td></td><td></td><td></td></tr><tr><td>Concept 1006</td></table>

Figure 56: ImageNet-trained DINOv2 concepts named on PartImageNet (curated examples). Each row shows the closest occurrence and four visually similar examples selected from the saved gallery. Fixed names appear at left, and the names below each image are A, additive, J, joint MLP, and gray T, teacher.