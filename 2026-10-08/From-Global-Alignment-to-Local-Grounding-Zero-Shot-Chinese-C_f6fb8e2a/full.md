# From Global Alignment to Local Grounding: Zero-Shot Chinese Character Recognition with Radical Verification

Yu-Heng Shih<sup>1</sup>, Bing-Chen Wu<sup>1</sup>, Tsz-To Wong<sup>1</sup>, Ting-En Yen<sup>1</sup>, Hong-Han Shuai<sup>1</sup> , Bin-Hua Hsieh<sup>2</sup> , Chien-An Chen<sup>2</sup>, Yi-Ren Yeh<sup>3</sup> , and Ching-Chun Huang<sup>1</sup>

<sup>1</sup> National Yang Ming Chiao Tung University, Taiwan   
{ra890927.cs12,evan20010126.cs12,tsztowong.cs13, tny0302.en12,hhshuai,chingchun}@nycu.edu.tw 2 E.SUN Financial Holding Co., Ltd., Taiwan {jeff-23314,lukechen-15953}@esunbank.com.tw 3 National Kaohsiung Normal University, Taiwan yryeh@nknu.edu.tw

Abstract. Zero-shot Chinese character recognition (ZS-CCR) aims to recognize characters whose categories are never observed during training, and typically relies on the compositional structure shared between seen and unseen characters. Recent CLIP-style methods represent this structure with the Ideographic Description Sequence (IDS) and align it with glyph images in a shared embedding space. However, they rely on a single global image–IDS similarity that discards the spatial layout of radicals and, being learned only implicitly from seen classes, generalizes poorly to unseen ones; moreover, global matching often retrieves the correct character within the top candidates yet fails to rank it first when characters difer only in subtle local radicals. To address these issues, we propose a global-to-local two-stage framework. In the first stage, STG-CLIP augments the IDS with explicit tree-position and radical-level geometric priors, yielding a spatial-aware prototype that provides a consistent spatial description across seen and unseen categories for high-recall global retrieval. In the second stage, the Radical Verification Module (RVM) uses the radical instances of each retrieved candidate as queries to verify whether the corresponding radicals can be matched to spatially compatible regions in the input glyph. A margin-based gating rule activates the RVM only when the leading global candidates receive similar similarity scores. Experiments on the ICDAR2013 benchmark demonstrate that our method achieves state-of-the-art performance under the characterlevel zero-shot setting, obtaining 83.06% top-1 accuracy with 2,755 seen classes. Ablation studies further show that the explicit geometric priors and radical-level verification provide complementary improvements.

Keywords: Zero-Shot Learning · Chinese Character Recognition · Radical Verification

## 1 Introduction

Unlike alphabetic scripts, Chinese contains an enormous number of character categories, and rare characters absent from the training data are routinely encountered in everyday handwriting, historical documents, and personal or place names. Since collecting handwritten samples for every category is practically impossible, zero-shot Chinese character recognition (ZS-CCR), which recognizes characters without any handwritten training samples, provides a practical direction for handling rare and unseen characters.

In ZS-CCR, the vocabulary is split into disjoint seen classes, which have handwritten training samples, and unseen classes, which have none. Every class is described by its Ideographic Description Sequence (IDS), which decomposes a character into radical and structural-operator tokens; e.g., an character 好 decomposes into 女 and 子 in a left–right structure, both of which also appear in seen characters. During training, the model learns to match handwritten images of seen classes to the IDS of their classes. At inference, it takes a handwritten image of an unseen class and outputs the class whose IDS best matches it. No handwritten image, glyph, radical mask, or class description of an unseen class enters optimization.

Recognition is possible because Chinese characters are strongly compositional: radicals and their spatial arrangements are shared across categories, so knowledge transfers from seen to unseen classes. Recent ZS-CCR methods adopt CLIP-style frameworks [1, 5, 11] that learn a shared embedding space between glyph images and IDS sequences. Relying on a single global image–IDS similarity, however, has two limitations for fine-grained recognition.

First, the IDS is flattened into a one-dimensional sequence before text encoding, which discards the parent–child and sibling relationships of its tree topology. A prior work, FT-CLIP [5], encodes only the parent relation of each node. More importantly, existing methods lack explicit geometric descriptors of where each radical resides and how much space it occupies, so the model must implicitly infer these spatial distributions from seen characters. When transferred to unseen characters with novel radical combinations and layouts, the implicitly learned positional prior extrapolates poorly, causing a seen-to-unseen spatial generalization gap.

Second, a single global similarity score is frequently insuficient for finegrained discrimination. Characters with similar overall layouts often difer by only a single local radical. Global alignment efectively retrieves the correct category within the top-k candidates but struggles to rank it first. Global matching thus suits candidate retrieval but not the final decision.

To address these limitations, we propose a coarse-to-fine two-stage recognition framework grounded in a dual-role interpretation of the IDS: the IDS is encoded into a spatial-aware class prototype for global retrieval, while the same decomposition serves as radical-level hypotheses to be verified locally. Rather than leaving radical positions to be inferred from data, the first stage, Spatial Tree-Geometric CLIP (STG-CLIP), injects tree-position embeddings (depth and sibling indices) and canonical radical geometric priors (normalized centroid, bounding box, and area ratio) into the IDS representation as explicit signals. Because these priors are derived from canonical glyphs and remain independent of category visibility, STG-CLIP establishes a consistent spatial description across seen and unseen categories; the resulting prototypes are used for global image–prototype alignment to retrieve top-k candidates at high recall. The second stage, the Radical Verification Module (RVM), treats the IDS decomposition of each retrieved candidate as a set of radical hypotheses. Through cross-attention between radical queries and localized image features, the RVM extracts radical-level visual evidence and evaluates each candidate by radical presence and glyph coverage. A margin-based gating rule invokes the RVM only when the leading global candidates are indistinguishable.

Our main contributions are summarized as follows:

Spatial-Aware IDS Representation We introduce a tree- and geometryaware IDS representation that injects depth and sibling embeddings alongside canonical geometry (centroid, bounding box, and area ratio) into IDS tokens. This explicit class-side information mitigates the seen-to-unseen spatial gap.

– Coarse-to-Fine Framework with Margin Gating We develop a margingated coarse-to-fine inference framework that separates global candidate retrieval from local verification, triggering verification only for ambiguous decisions to balance eficiency and accuracy.

– Radical Verification Module (RVM) The RVM advances ZS-CCR from global similarity matching to candidate-conditioned local verification. By evaluating radical presence and glyph coverage, the RVM leverages localized visual evidence to resolve subtle ambiguities among visually similar characters.

## 2 Related Work

## 2.1 Contrastive Learning-based Methods for ZS-CCR

Contrastive learning frameworks, particularly CLIP-inspired methods [8], show strong potential in ZS-CCR by aligning visual features with class-level semantic text descriptions. CCR-CLIP [11] pioneers this paradigm by matching printed images with their 1D Ideographic Description Sequences (IDS) in a shared space. However, flattening the 2D spatial layout into a 1D sequence discards the hierarchical tree topology. To capture this, FT-CLIP [5] introduces a tree encoder operating on a novel Formation Tree, while Cai and Zhu [1] align both local components and global features to enrich cross-modal representations.

Although these methods improve structural modeling via symbolic or relative component relationships, they lack absolute geometric awareness. They fail to associate radical instances with continuous, character-dependent geometric attributes (e.g., centroid, bounding-box size, and area ratio). Conversely, our STG-CLIP augments tokens with explicit tree-position information and canonical geometry, while separating global retrieval from candidate-specific verification.

## 2.2 Generative Sequence Recognition Methods

Another paradigm treats ZS-CCR as an image-to-sequence translation task. Methods like DenseRAN [9] and SD/SLD [3] employ visual encoders with autoregressive (AR) decoders to sequentially generate radical or stroke sequences, enabling the recognition of unseen classes via reusable components. However, AR decoding is prone to error accumulation, where early errors propagate through the sequence, and requires complex post-processing to parse valid labels. In contrast, our method bypasses sequence generation entirely, treating the IDS as established class-side information to directly retrieve character classes through image–prototype alignment.

## 2.3 Radical Detection and Localization Methods

To enhance spatial grounding, several studies formulate radical recognition as a detection or localization problem. JRED [6] employs learnable radical embeddings as sliding detectors over multi-scale feature maps to generate explicit response maps. Similarly, LERRNet [7] introduces attribute hint vectors initialized via Word2Vec to guide the decoder toward specific radical regions, proving that local evidence efectively complements holistic representations. Nevertheless, these end-to-end models predict component sequences directly rather than verifying candidate hypotheses. Our Radical Verification Module (RVM) instead acts as a candidate-conditioned verifier, extracting localized visual evidence to resolve global prediction ambiguities.

## 2.4 Radical Representation Learning

Representing individual radical identities inherently before structural aggregation is also critical. Early methods represent radicals using discrete multi-hot or random encodings, ignoring visual properties and distorting the encoding space topology. To alleviate this, FaRE [13] introduces a feature-aware encoding strategy that extracts implicit visual features from rendered font images. While FaRE improves radical category embeddings, existing IDS-based methods still treat radicals as static semantic tokens, ignoring their character-dependent location, scale, or aspect ratio. Since the same radical exhibits distinct physical variations across diferent characters, our method models the role of a radical \*instance\* by attaching canonical geometric attributes to the corresponding IDS tokens, resolving their spatial blindness.

## 3 Proposed Method

## 3.1 Problem Formulation and Framework Overview

In zero-shot Chinese character recognition (ZS-CCR), the objective is to recognize handwritten character instances from categories that are not observed during training. Let $\mathcal { V } _ { \mathrm { s e e n } }$ and $\mathcal { V } _ { \mathrm { u n s e e n } }$ denote the sets of seen and unseen character categories, respectively, where $\mathcal { V } _ { \mathrm { s e e n } } \cap \mathcal { V } _ { \mathrm { u n s e e n } } = \emptyset$ . During training, handwritten character images are available only for categories in $\mathcal { \mathrm { V } } _ { \mathrm { s e e n } }$ . During inference, given a handwritten query image I<sup>∗</sup> from an unseen category, the model predicts its label $\hat { y } \in \mathcal { V } _ { \mathrm { u n s e e n } }$

![](images/1de74703f856167ce6bdff15c959e95147f0d153f7dd384921d21fe8f7c4d0d4.jpg)  
Fig. 1: Overview of our framework. STG-CLIP (Left): the first stage learns a shared image–text space by constructing tree- and geometry-aware IDS prototypes. RVM (Top-right): the RVM uses the radical occurrences of each candidate as queries to verify whether they are grounded in the input image. Two-stage Inference (Bottomright): STG-CLIP first retrieves the Top-K candidates by global similarity; the RVM then verifies the radicals of each candidate against the input glyph and selects the final prediction when the leading candidates are globally ambiguous.

In our setting, IDS descriptions and canonical geometric descriptors are treated as class-side side information available for all categories, whereas printed glyph images and radical masks belonging to unseen classes are strictly excluded from model optimization.

Following the dual-role interpretation of IDS, the proposed framework separates global candidate retrieval from local radical verification, as illustrated in Figure 1. The first stage, Spatial Tree-Geometric CLIP (STG-CLIP), augments IDS tokens with explicit tree-position and radical-geometry information to learn a shared image–prototype embedding space. Given a handwritten query, STG-CLIP performs global alignment to retrieve a compact top-K candidate shortlist $\mathcal { C } _ { K } \subset \mathcal { V } _ { \mathrm { u n s e e n } }$ at high recall.

The second stage, the Radical Verification Module (RVM), treats the IDS decomposition of each retrieved candidate as a specific set of radical hypotheses. Through cross-attention between radical queries and localized visual features, the RVM evaluates whether the queried radicals are visually present and whether their predicted regions spatially support the observed glyph foreground. Finally, a margin-based gating rule decides whether to invoke this local verification to re-rank the candidates or safely trust the global top-1 prediction.

## 3.2 Tree- and Geometry-Aware IDS Representation

For each character category $y$ (either seen or unseen), we construct a geometrically augmented IDS representation $S _ { \mathrm { g e o } } ( y )$ and encode it into a class prototype $\mathbf { t } _ { y } .$ This representation enriches the original IDS token identities by injecting lightweight tree-position information and canonical radical-level geometry.

IDS Tree-Position Encoding The text encoder in CLIP-style frameworks is inherently designed for sequential inputs, meaning its self-attention mechanism does not presuppose any hierarchical tree structures. When the IDS tree is flattened into a one-dimensional sequence, standard sequential encoders struggle to distinguish sibling components at the same hierarchical level from nested components across diferent levels.

To preserve the structural relations implicit in the tree topology, we parse the IDS into a tree structure and assign each token a depth index $d _ { i }$ and a sibling index $s _ { i }$ . The depth index denotes the token’s distance from the root node, while the sibling index captures the left-to-right order among nodes sharing the same parent. These attributes provide lightweight hierarchical cues to guide the sequence encoding.

Radical Geometry Augmentation While the tree-position encoding captures the symbolic organization of the IDS tree, it lacks absolute physical awareness regarding where each component resides and how much space it occupies. To address this spatial blindness, we augment the tokens with continuous geometric descriptors computed from canonical printed glyph masks.

For a radical token, its corresponding stroke paths are grouped and rendered onto a binary canvas to obtain its component mask. For a structural operator token, its geometric attribute is derived from the spatial aggregation (the enclosing bounding box and union of masks) of its corresponding subtree. From the resulting mask of each token $i ,$ we extract a five-dimensional geometric prior vector:

$$
\mathbf { g } _ { i } = [ c _ { x } , c _ { y } , b _ { w } , b _ { h } , r _ { \mathrm { a r e a } } ] \in \mathbb { R } ^ { 5 } ,\tag{1}
$$

where $( c _ { x } , c _ { y } )$ is the normalized centroid within the character canvas, $( b _ { w } , b _ { h } )$ represents the normalized bounding-box width and height, and $r _ { \mathrm { a r e a } }$ denotes the foreground area ratio computed specifically within its own bounding box rather than the whole glyph plane. These descriptors serve as stable class-level spatial priors independent of handwriting styles.

Given the original token embedding $\mathbf { x } _ { i }$ , we form a fused token representation by adding the tree-position and projected geometric embeddings:

$$
\mathbf { x } _ { i } ^ { \mathrm { g e o } } = \mathbf { x } _ { i } + E _ { \mathrm { d e p t h } } ( d _ { i } ) + E _ { \mathrm { s i b l i n g } } ( s _ { i } ) + \mathrm { M L P } ( \mathbf { g } _ { i } ) ,\tag{2}
$$

where $E _ { \mathrm { d e p t h } } ( \cdot )$ and $E _ { \mathrm { s i b l i n g } } ( \cdot )$ are learnable tree-position embeddings, and $\mathrm { M L P ( \cdot ) }$ projects the geometric descriptor into the token embedding dimension. The final augmented sequence for category $y$ is denoted as $S _ { \mathrm { g e o } } ( y ) = \{ \mathbf { x } _ { i } ^ { \mathrm { g e o } } \} _ { i = 1 } ^ { L _ { y } }$ , where $L _ { y }$ is the sequence length.

IDS Prototype Encoding The spatial-aware token sequence is fed into a text encoder $\varPsi ( \cdot )$ to generate the final class prototype vector:

$$
\mathbf { t } _ { y } = \varPsi ( S _ { \mathrm { g e o } } ( y ) ) .\tag{3}
$$

When instantiated with a CLIP-style Transformer, the hidden state at the endof-sequence (EOS) token is projected into the shared embedding space as $\mathbf { t } _ { y } .$ Notably, this representation serves a dual purpose: $\mathbf { t } _ { y }$ acts as a global retrieval target in STG-CLIP, while the underlying symbolic radical decomposition is passed to the RVM as structural hypotheses for local verification.

## 3.3 Global Image–Prototype Alignment (STG-CLIP)

The first stage, STG-CLIP, learns a shared embedding space between handwritten character images and the spatially enriched IDS prototypes to perform coarse retrieval.

Image and Prototype Representation Given a handwritten image $I _ { \mathrm { h w } }$ and its corresponding canonical printed glyph $I _ { \mathrm { s t d } }$ from a seen category, a shared image encoder $\varPhi _ { \mathrm { i m g } } ( \cdot )$ extracts their visual embeddings:

$$
\begin{array} { r } { \mathbf { v } _ { \mathrm { h w } } = \varPhi _ { \mathrm { i m g } } ( I _ { \mathrm { h w } } ) , \qquad \mathbf { v } _ { \mathrm { s t d } } = \varPhi _ { \mathrm { i m g } } ( I _ { \mathrm { s t d } } ) . } \end{array}\tag{4}
$$

The printed-glyph branch acts purely as an auxiliary regularizer during training and is completely omitted at inference. All visual and text embeddings are $L _ { 2 ^ { - } }$ normalized before calculating similarity scores.

Contrastive Alignment with Style Regularization A symmetric InfoNCE objective is then applied to the distinct image–prototype pairs $\{ ( \bar { \mathbf { v } } _ { c } , \mathbf { t } _ { c } ) \} _ { c = 1 } ^ { C }$ within the mini-batch:

$$
\mathcal { L } _ { \mathrm { i t c } } = - \frac { 1 } { 2 C } \sum _ { c = 1 } ^ { C } \left[ \log \frac { \exp ( \sin ( \bar { \mathbf { v } } _ { c } , \mathbf { t } _ { c } ) / \tau ) } { \sum _ { j = 1 } ^ { C } \exp ( \sin ( \bar { \mathbf { v } } _ { c } , \mathbf { t } _ { j } ) / \tau ) } + \log \frac { \exp ( \sin ( \mathbf { t } _ { c } , \bar { \mathbf { v } } _ { c } ) / \tau ) } { \sum _ { j = 1 } ^ { C } \exp ( \sin ( \mathbf { t } _ { c } , \bar { \mathbf { v } } _ { j } ) / \tau ) } \right] ,\tag{5}
$$

where sim(·, ·) denotes cosine similarity and τ is a temperature parameter.

To mitigate the visual domain gap caused by diverse handwriting styles, we introduce an auxiliary style regularization term that encourages the handwritten embedding to align closely with the corresponding canonical printed glyph:

$$
\mathcal { L } _ { \mathrm { s t y l e } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ 1 - \sin ( { \mathbf v } _ { \mathrm { h w } , i } , { \mathbf v } _ { \mathrm { s t d } , i } ) \right] .\tag{6}
$$

The overall objective for global alignment is formulated as:

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = \mathcal { L } _ { \mathrm { i t c } } + \lambda _ { \mathrm { s t y l e } } \mathcal { L } _ { \mathrm { s t y l e } } ,\tag{7}
$$

where $\lambda _ { \mathrm { s t y l e } }$ controls the weight of the style regularization term.

Top-K Candidate Retrieval During inference, given a query handwritten image $I ^ { * }$ , its visual feature vector $\mathbf { v } ^ { * } = \varPhi _ { \mathrm { i m g } } ( I ^ { * } )$ is matched against all possible character categories in the entire recognition vocabulary to compute global similarity scores:

$$
s _ { \mathrm { g l o b a l } } ( y ) = \sin ( { \mathbf { v } } ^ { * } , { \mathbf { t } } _ { y } ) , \qquad y \in \mathcal { Y } _ { \mathrm { v o c a b u l a r y } } .\tag{8}
$$

The coarse shortlist of candidates is then retrieved by selecting the top-K scoring classes:

$$
\begin{array} { r } { \mathcal { C } _ { K } = \mathrm { T o p K } _ { y \in \mathcal { V } _ { \mathrm { v o c a b u l a r y } } } \ s _ { \mathrm { g l o b a l } } ( y ) . } \end{array}\tag{9}
$$

## 3.4 Radical Verification Module

To resolve fine-grained ambiguities among visually similar characters that share a nearly identical global layout but difer by a single radical, we introduce the Radical Verification Module (RVM). The RVM shifts the focus from holistic semantic matching to localized structural alignment, explicitly verifying whether a candidate’s radicals are visually grounded in the input image.

Localized Visual Features To extract rich local details, a separate CNN backbone $\varPhi _ { \mathrm { l o c a l } } ( \cdot )$ —specifically ResNet-50 [4]—is dedicated to the RVM and trained independently of STG-CLIP. The generated visual feature maps are flattened and projected into a sequence of patch tokens $F = \{ \mathbf { f } _ { 1 } , \dots , \mathbf { f } _ { N } \} \in \mathbb { R } ^ { N \times d }$ . Standard 2D sinusoidal position embeddings are added to $F$ to preserve the spatial layout of the character.

Decoder-based Structural Alignment For an evaluated candidate $y ,$ its IDS decomposition yields an ordered sequence of radical occurrences $\mathcal { R } ( y ) =$ $( r _ { 1 } , \ldots , r _ { Q _ { y } } )$ . Each radical occurrence is initialized as a learnable embedding to serve as a decoder query:

$$
R ^ { ( 0 ) } = \left( E _ { \mathrm { r a d } } ( r _ { 1 } ) , \ldots , E _ { \mathrm { r a d } } ( r _ { Q _ { y } } ) \right) \in \mathbb { R } ^ { Q _ { y } \times d } .\tag{10}
$$

Unlike STG-CLIP text representations, these queries do not encode canonical geometric priors, as handwritten and printed radicals are not pixel-aligned.

These queries are fed into a cascade of L transformer decoder layers to attend to the localized visual features $F .$ . At the l-th layer, the representations are refined via cross-attention:

$$
\widetilde { R } ^ { ( l ) } = \mathrm { L a y e r N o r m } \left( R ^ { ( l - 1 ) } + \mathrm { C r o s s A t t n } \left( Q = R ^ { ( l - 1 ) } , K = F , V = F \right) \right) ,\tag{11}
$$

$$
R ^ { ( l ) } = \mathrm { L a y e r N o r m } \left( \widetilde { R } ^ { ( l ) } + \mathrm { F F N } \left( \widetilde { R } ^ { ( l ) } \right) \right) .\tag{12}
$$

The terminal contextualized radical representations $R ^ { ( L ) } \in \mathbb { R } ^ { Q _ { y } \times d }$ are subsequently forwarded to two parallel prediction heads. The match head applies

a linear projection followed by a sigmoid function to output an existence probability $\hat { s } _ { q } \in [ 0 , 1 ]$ for each radical query, quantifying whether the component is visually supported:

$$
\hat { s } _ { q } = \sigma \left( h _ { \mathrm { m a t c h } } \left( R _ { q } ^ { ( L ) } \right) \right) .\tag{13}
$$

Simultaneously, the mask head projects each token into a spatial response map, which is reshaped and bilinearly upsampled to the input resolution, yielding a soft attention mask $\hat { M } _ { q } \in [ 0 , 1 ] ^ { \overset { \vartriangle } { H } \times \hat { W } }$ that dynamically delineates the predicted location of the q-th radical:

$$
\hat { M } _ { q } = h _ { \mathrm { m a s k } } \left( R _ { q } ^ { ( L ) } \right) .\tag{14}
$$

Multi-Task Optimization During training, positive queries are formed by the radicals inherently present in the character, while negative queries are randomly sampled from absent categories.

Radical existence is supervised via a binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { m a t c h } } = - \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \left[ s _ { q } \log ( \hat { s } _ { q } ) + ( 1 - s _ { q } ) \log ( 1 - \hat { s } _ { q } ) \right] ,\tag{15}
$$

where $s _ { q } \in \{ 0 , 1 \}$ represents the ground-truth presence.

For spatial localization, since radical strokes are extremely sparse on the image lattice, we optimize positive queries using a combination of Focal loss (to suppress easy background pixels) and Dice loss (to optimize region overlap):

$$
\mathcal { L } _ { \mathrm { f o c a l } } ^ { ( q ) } = - \frac { 1 } { H W } \sum _ { i = 1 } ^ { H W } \alpha _ { t } ( 1 - p _ { q , i } ) ^ { \gamma } \log ( p _ { q , i } ) ,\tag{16}
$$

where $p _ { q , i } = { \hat { m } } _ { q , i }$ if the ground-truth mask pixel $m _ { q , i } = 1$ , and $p _ { q , i } = 1 - \hat { m } _ { q , i }$ otherwise; $\alpha _ { t }$ balances foreground/background pixel distributions. The Dice loss is defined as:

$$
\mathcal { L } _ { \mathrm { d i c e } } ^ { ( q ) } = 1 - \frac { 2 \langle \hat { M } _ { q } , M _ { q } \rangle + \epsilon } { \Vert \hat { M } _ { q } \Vert _ { 1 } + \Vert M _ { q } \Vert _ { 1 } + \epsilon } ,\tag{17}
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product and ϵ is a small constant. Negative queries bypass the spatial losses. The total multi-task objective is:

$$
\mathcal { L } _ { \mathrm { R V M } } = \mathcal { L } _ { \mathrm { m a t c h } } + \lambda _ { 1 } \frac { 1 } { \vert \mathcal { Q } ^ { + } \vert } \sum _ { q \in \mathcal { Q } ^ { + } } \mathcal { L } _ { \mathrm { f o c a l } } ^ { ( q ) } + \lambda _ { 2 } \frac { 1 } { \vert \mathcal { Q } ^ { + } \vert } \sum _ { q \in \mathcal { Q } ^ { + } } \mathcal { L } _ { \mathrm { d i c e } } ^ { ( q ) } ,\tag{18}
$$

where $\mathcal { Q } ^ { + }$ is the set of positive queries, and $\lambda _ { 1 } , \lambda _ { 2 }$ are balancing weights.

## 3.5 Training and Inference Pipeline

To ensure robust radical grounding and eficient deployment, we design a twostage training strategy followed by a gated inference mechanism.

Two-Stage Training Strategy The training procedure consists of a printed pretraining phase followed by a handwritten fine-tuning phase. In the printed pretraining stage (strong supervision), the RVM is optimized on standard, clean printed glyphs from the seen character classes. Because printed glyphs ofer precise, pixel-level radical mask annotations, all three loss functions $( { \mathcal { L } } _ { \mathrm { m a t c h } } , { \mathcal { L } } _ { \mathrm { f o c a l } } .$ and $\mathcal { L } _ { \mathrm { d i c e } } )$ are fully activated. This stage forces the RVM queries to firmly anchor onto their corresponding visual regions and layouts. In the subsequent handwritten fine-tuning stage (weak supervision), the model transitions to real-world handwritten images from the seen classes. Since real handwritten data lacks radical-level pixel annotations, the mask head is frozen to preserve the anchoring capabilities learned from printed glyphs. The remaining parts of the module are optimized using exclusively the character-level occurrence loss ${ \mathcal { L } } _ { \mathrm { m a t c h } }$ , allowing the local backbone and cross-attention blocks to adapt to handwritten deformations without requiring dense spatial labels.

Gated Coarse-to-Fine Inference At inference, the RVM serves as a plug-and-play verifier for the retrieved candidate shortlist $\mathcal { C } _ { K }$ . For each candidate $y _ { k } \in \mathcal { C } _ { K }$ , its constituent radicals are queried to output existence scores $\hat { s } _ { j }$ and soft masks $\hat { M _ { j } }$ . We evaluate each candidate through a two-fold local verification score. The first component is radical presence $( s _ { \mathrm { p r e s } } )$ , which computes the average existence probability of all queried radicals to penalize extraneous components:

$$
s _ { \mathrm { p r e s } } ( y _ { k } ) = \frac { 1 } { | \mathcal { R } ( y _ { k } ) | } \sum _ { j = 1 } ^ { | \mathcal { R } ( y _ { k } ) | } \hat { s } _ { j } .\tag{19}
$$

The second component is glyph coverage $\left( { { s _ { \mathrm { { c o v } } } } } \right)$ , which measures whether the union of the predicted radical masks covers the actual observed strokes in the handwritten foreground mask $M _ { \mathrm { f g } }$ obtained by simple image binarization:

$$
M _ { \mathrm { u n i o n } } ( y _ { k } ) = \bigcup _ { j = 1 } ^ { | \mathcal { R } ( y _ { k } ) | } \mathbb { I } \left( \hat { M } _ { j } > \theta \right) ,\tag{20}
$$

$$
s _ { \mathrm { c o v } } ( y _ { k } ) = \frac { \left. M _ { \mathrm { u n i o n } } ( y _ { k } ) \cap M _ { \mathrm { f g } } \right. _ { 1 } } { \left. M _ { \mathrm { f g } } \right. _ { 1 } + \epsilon } ,\tag{21}
$$

where $\mathbb { I } ( \cdot )$ binarizes the soft masks at threshold θ. The joint local verification score is formulated as:

$$
s _ { \mathrm { l o c a l } } ( y _ { k } ) = s _ { \mathrm { p r e s } } ( y _ { k } ) \cdot s _ { \mathrm { c o v } } ( y _ { k } ) .\tag{22}
$$

To maintain retrieval eficiency and avoid unnecessary computations, a marginbased gating rule dictates the final prediction. Let $y _ { ( 1 ) }$ and $y _ { ( 2 ) }$ be the top-1 and top-2 candidates ranked by the first-stage global score s<sub>global</sub>. We calculate the confidence margin:

$$
\varDelta = s _ { \mathrm { g l o b a l } } ( y _ { ( 1 ) } ) - s _ { \mathrm { g l o b a l } } ( y _ { ( 2 ) } ) .\tag{23}
$$

The final predicted label $\hat { y }$ is determined by:

$$
\hat { y } = \left\{ \begin{array} { l l } { \arg \operatorname* { m a x } _ { y _ { k } \in \mathcal { C } _ { K } } s _ { \mathrm { l o c a l } } ( y _ { k } ) , } & { \mathrm { i f } ~ \varDelta \leq \delta , } \\ { y _ { ( 1 ) } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{24}
$$

where $\delta$ is a predefined threshold tuned on the validation set. When the global confidence margin $\varDelta$ is large $\left( \varDelta > \delta \right)$ , the global top-1 prediction is directly trusted for maximum eficiency. Conversely, when the top two candidates are neck-and-neck $( \varDelta \leq \delta )$ , indicating a fine-grained structural ambiguity, the gate is activated, and the local verification score $s _ { \mathrm { l o c a l } }$ is leveraged to resolve the ambiguity based on explicit local evidence.

## 4 Experiments

## 4.1 Dataset and Implementation Details

Dataset. We conduct experiments on two ofline handwritten Chinese character datasets, CASIA-HWDB1.0-1.1 [14] and ICDAR2013 [10]. CASIA-HWDB1.0-1.1 aggregates 2,678,424 isolated character samples over 3,881 classes, among which 3,755 belong to the level-1 commonly used character set of GB2312-80, and we use these samples for training. ICDAR2013 is adopted as the test set; it contains 224,419 handwritten samples collected from 60 writers that are disjoint from those in the training set, while covering the identical 3,755 level-1 categories.

To supervise the RVM at the radical level, we additionally construct groundtruth radical masks from printed glyphs, since handwritten samples provide no pixel-level radical annotations. Specifically, we render characters from makemeahanzi and, following the IDS decomposition, group the strokes belonging to each radical and rasterize them onto the glyph canvas to obtain a binary mask for that radical. These printed-glyph masks are used only in the RVM pre-training stage; during handwritten fine-tuning, the mask head is frozen and no mask annotations are required. Illustrative examples are provided in the supplementary material.

Implementation Details. Our framework is implemented in $\mathrm { P y }$ Torch and evaluated across two processing granularities. For the global retrieval stage, the STG-CLIP image encoder operates on $3 2 \times 3 2$ inputs, and the style-regularization weight is set to $\lambda _ { \mathrm { s t y l e } } { = } 1$ . For the local verification stage, the Radical Verification Module (RVM) adopts a dedicated backbone operating on 64 × 64 inputs, and the balancing weights for multi-task optimization are set to $\lambda _ { 1 } { = } 5$ and $\lambda _ { 2 } { = } 1$ . At inference time, the gating threshold $\delta$ is adaptively selected per setting on the validation set. All remaining hardware configurations, training objectives, and optimization hyperparameters are deferred to the Supplementary Material.

Table 1: Character-level and radical-level zero-shot recognition accuracy on handwritten datasets under diferent numbers of seen training classes.
<table><tr><td></td><td colspan="5">Character Zero-shot</td><td colspan="5">Radical Zero-shot</td></tr><tr><td>Method</td><td>500</td><td>1000</td><td>1500</td><td>2000</td><td>2755</td><td>50</td><td>40</td><td>30</td><td>20</td><td>10</td></tr><tr><td>DenseRAN [9]</td><td>1.70%</td><td>8.44%</td><td>14.71%</td><td>19.51%</td><td>30.68%</td><td>0.21%</td><td>0.29%</td><td>0.25%</td><td>0.42%</td><td>0.69%</td></tr><tr><td>HDE [2]</td><td>4.90%</td><td>12.77%</td><td>19.25%</td><td>25.13%</td><td>33.49%</td><td>3.26%</td><td>4.29%</td><td>6.33%</td><td>7.64%</td><td>9.33%</td></tr><tr><td>SD [3]</td><td>5.60%</td><td>13.85%</td><td>22.88%</td><td>25.73%</td><td>37.91%</td><td>5.28%</td><td>6.87%</td><td>9.02%</td><td>14.67%</td><td>15.83%</td></tr><tr><td>STAR [12]</td><td>7.54%</td><td>19.47%</td><td>27.79%</td><td>35.53%</td><td>43.86%</td><td>6.95%</td><td>12.28%</td><td>14.74%</td><td>18.37%</td><td>23.23%</td></tr><tr><td>FaRE [13]</td><td>7.21%</td><td>21.78%</td><td>36.58%</td><td>47.33%</td><td>57.17%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JRED [6]</td><td>25.19%</td><td>36.21%</td><td>41.93%</td><td>42.05%</td><td>56.26%</td><td>10.21%</td><td>8.30%</td><td>12.67%</td><td>19.13%</td><td>27.08%</td></tr><tr><td>CCR-CLIP [11]</td><td>21.79%</td><td>42.99%</td><td>55.86%</td><td>62.99%</td><td>72.98%</td><td>11.15%</td><td>13.85%</td><td>16.01%</td><td>16.76%</td><td>15.96%</td></tr><tr><td>LERRNet [7]</td><td></td><td>32.73% 55.36% 66.01%</td><td></td><td>70.81%</td><td>73.59%</td><td>8.35%</td><td>11.22%</td><td>14.81%</td><td>15.98%</td><td>19.37%</td></tr><tr><td>FT-CLIP [5]</td><td>26.17%</td><td>50.68%</td><td>62.25%</td><td>70.39%</td><td>78.96%</td><td>12.34%</td><td>16.66%</td><td>21.51%</td><td>24.65%</td><td>27.80%</td></tr><tr><td>STG-CLIP</td><td>21.25%</td><td>39.38%</td><td>59.91%</td><td>69.96%</td><td>82.26%</td><td>8.89%</td><td>14.26%</td><td>21.62% 27.63%37.30%</td><td></td><td></td></tr><tr><td>STG-CLIP + RVM 22.12%</td><td></td><td>40.52%</td><td>63.93%</td><td></td><td>71.48% 83.06%</td><td>8.74%</td><td>14.22%</td><td>21.44%</td><td>27.52%37.76%</td><td></td></tr></table>

## 4.2 Experiments on Unseen Character Recognition

Zero-shot learning refers to the ability to generalize to unseen categories. In Chinese character recognition, this setting is particularly important because many characters may be absent from the training set, while their structures can still be described by IDS. We conduct experiments under two zero-shot settings, including character zero-shot and radical zero-shot. During testing, all categories appearing in the corresponding experiment are used as candidates.

Character Zero-shot Settings. The selection of categories in the training and testing sets has a significant impact on zero-shot Chinese character recognition. For handwritten characters, we first sort the 3755 level-1 commonly used Chinese characters according to the order of GB2312. Samples from CASIA-HWDB whose labels belong to the first m characters are used as the training set, where m ∈ {500, 1000, 1500, 2000, 2755}. Samples from ICDAR2013 whose labels belong to the last 1000 characters are used as the test set.

Radical Zero-shot Settings. In the radical zero-shot setting, characters are split according to the frequency of their radicals. A character is assigned to the test set if it contains at least one radical whose frequency is lower than a threshold n; otherwise, it is assigned to the training set. We evaluate five thresholds, where n ∈ {50, 40, 30, 20, 10}. This setting evaluates whether the model can generalize to characters containing rare radical compositions.

## 4.3 Comparison with Existing Methods

Table 1 reports accuracy under the character and radical zero-shot settings. CLIP-based methods clearly outperform radical-based ones, and among them our method achieves the best results on most settings. On the character setting, STG-CLIP+RVM reaches 83.06% at 2755 seen classes, surpassing the previous best FT-CLIP by 4.10 points and CCR-CLIP by 10.08. RVM consistently improves STG-CLIP across all seen-class counts, most notably by 4.02 points at 1500, confirming that local radical verification resolves the fine-grained ambiguities left by the global stage. Under the radical setting, however, RVM yields little gain: since this split partitions data by radicals, the test characters contain radicals unseen during training, which RVM cannot reliably verify.

Table 2: Cumulative ablation study of the proposed components under the character zero-shot setting with $2 , 7 5 5$ seen classes. Each row adds one component to the previous setting.
<table><tr><td>Cumulative setting</td><td>Top-1 Acc.</td><td>Δ</td></tr><tr><td>Baseline</td><td>76.27</td><td></td></tr><tr><td>+ IDS Prototype Encoding</td><td>77.78</td><td>+1.51</td></tr><tr><td>+ Radical Geometry Augmentation</td><td>82.26</td><td>+4.48</td></tr><tr><td>+ RVM</td><td>83.06</td><td>+0.80</td></tr></table>

STG-CLIP is less competitive under small training data, trailing CCR-CLIP at 500 and 1000 seen classes. We attribute this to our ViT-based image encoder, which lacks the local inductive biases of CCR-CLIP’s convolutional encoder and thus needs more data to train well. The same crossover appears in both settings at a comparable data scale, indicating that ViT is more data-hungry but scales better once data is suficient.

## 4.4 Ablation Studies

Table 2 evaluates the cumulative impact of our components. Compared to a plain CLIP baseline (76.27%), reinjecting tree topology (depth and sibling indices) lifts accuracy to 77.78% (+1.51%). Further augmenting tokens with explicit canonical geometry yields the most substantial improvement to 82.26% (+4.48%), suggesting that explicit physical attributes mitigate the seen-to-unseen spatial gap. Lastly, incorporating the RVM provides an additional boost to 83.06% (+0.80%), demonstrating that localized structural verification resolves fine-grained global ambiguities. Note that per-component gains depend on the order of addition, and this ablation does not fully separate the encoding design from the added geometric information.

Selection of K and δ We select the shortlist size K and gating threshold δ using a held-out validation set. As illustrated in Figure 2, a smaller shortlist consistently yields better accuracy $( K { = } 2 > K { = } 3 > K { = } 4 )$ . Although expanding K improves raw retrieval recall, it also admits negative distractors that trigger re-ranking errors. For K=2, the model exhibits high robustness to the threshold, with performance fluctuating by less than 0.4% across all evaluated δ. Accordingly, the validation optimums (K=2 and δ=0.03) are selected and fixed for final evaluation.

![](images/1384653a90f9f32ce2d9f9e672432bc9d9259734e8a81d01792aa9fb0e6e198b.jpg)  
Fig. 2: Hyperparameter selection for shortlist size K and gating threshold δ on the validation set (character zero-shot setting). The optimal configuration (K=2, δ=0.03) is fixed at test time.

## 5 Conclusion

In this paper, we propose a coarse-to-fine framework for zero-shot Chinese character recognition. By introducing a spatial-aware IDS representation that injects tree topology and continuous geometric priors, we establish a consistent spatial description across categories, efectively mitigating the seen-to-unseen spatial gap. Our framework coordinates STG-CLIP for high-recall global retrieval and a Radical Verification Module (RVM) for local candidate re-ranking, selectively triggered via a margin-gated rule. Extensive experiments demonstrate that our method achieves state-of-the-art accuracy in character zero-shot scenarios. Our analysis also reveals scaling properties of ViT encoders and outlines limitations in radical zero-shot settings. Explicitly modeling radical geometry and local verification provides a robust paradigm for compositional recognition, and extending the RVM to entirely unseen radicals remains an important direction for future work.

## Acknowledgements

This work was financially supported in part (project number: 112UA10019) by the Co-creation Platform of the Industry Academia Innovation School, NYCU, under the framework of the National Key Fields Industry-University Cooperation and Skilled Personnel Training Act, from the Ministry of Education (MOE) and industry partners in Taiwan. It also supported in part by the National Science and Technology Council, Taiwan, under Grant NSTC-115-2634-F-A49-011-, NSTC-114-2218-E-A49-024-, Grant NSTC-115-2425-H-A49-001, Grant NSTC-114-2622-E-A49-027, Grant NSTC-115-2221-E-A49 -124 -MY3, Grant NSTC-115-2218-E-A49 -017 and in part by the Higher Education Sprout Project of the National Yang Ming Chiao Tung University and the Ministry of Education (MOE), Taiwan. It is also partly supported by E.SUN Financial Holding Co Ltd. (Corresponding author: Ching-Chun Huang.)

## References

1. Cai, H., Zhu, A.: Cross-modal alignment of local and global features for zero-shot chinese character recognition. In: 2024 IEEE International Conference on Image Processing (ICIP). pp. 2041–2047 (2024). https://doi.org/10.1109/ICIP51287. 2024.10647599

2. Cao, Z., Lu, J., Cui, S., Zhang, C.: Zero-shot handwritten chinese character recognition with hierarchical decomposition embedding. Pattern Recognition 107, 107488 (2020). https://doi.org/https://doi.org/10.1016/j.patcog.2020.107488

3. Chen, J., Li, B., Xue, X.: Zero-shot chinese character recognition with stroke-level decomposition. In: Zhou, Z.H. (ed.) Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21. pp. 615–621. International Joint Conferences on Artificial Intelligence Organization (8 2021). https://doi. org/10.24963/ijcai.2021/85, main Track

4. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

5. Hong, Y., Qiao, X., Li, Y., Li, R., Zhang, J.: Improving chinese character representation with formation tree. Neurocomputing 638, 130098 (2025). https: //doi.org/https://doi.org/10.1016/j.neucom.2025.130098

6. Luo, G.F., Wang, D.H., Zhang, X.Y., Lin, Z.H., Zhu, S.: Joint radical embedding and detection for zero-shot chinese character recognition. Pattern Recognition 161, 111286 (2025). https://doi.org/https://doi.org/10.1016/j.patcog.2024. 111286

7. Pan, S.L., Wang, D.H., Jiang, N., Zhang, X.Y., Zhu, S.: Learning explicit radical representations for zero-shot chinese character recognition. In: Antonacopoulos, A., Chaudhuri, S., Chellappa, R., Liu, C.L., Bhattacharya, S., Pal, U. (eds.) Pattern Recognition. pp. 303–317. Springer Nature Switzerland, Cham (2025)

8. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/radford21a.html

9. Wang, W., Zhang, J., Du, J., Wang, Z.R., Zhu, Y.: DenseRAN for ofline handwritten chinese character recognition. In: 2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR). pp. 104–109 (2018). https: //doi.org/10.1109/ICFHR-2018.2018.00027

10. Yin, F., Wang, Q., Zhang, X., Liu, C.: ICDAR 2013 chinese handwriting recognition competition. Proceedings of the International Conference on Document Analysis and Recognition, ICDAR pp. 1464–1470 (2013). https://doi.org/10.1109/ ICDAR.2013.218, 12th International Conference on Document Analysis and Recognition, ICDAR 2013

11. Yu, H., Wang, X., Li, B., Xue, X.: Chinese text recognition with a pre-trained CLIP-like model through image-IDS aligning. pp. 11909–11918 (2023)

12. Zeng, J., Xu, R., Wu, Y., Li, H., Lu, J.: Zero-shot chinese character recognition with stroke-and radical-level decompositions. In: 2023 International Joint Conference on Neural Networks (IJCNN). pp. 1–9. IEEE (2023)

13. Zhan, H., Li, Y., Xiong, Y.j., Lu, Y.: FaRE: A feature-aware radical encoding strategy for zero-shot chinese character recognition. In: Proceedings of the Asian Conference on Computer Vision. pp. 390–401 (2024)

14. Zhang, X.Y., Bengio, Y., Liu, C.L.: Online and ofline handwritten chinese character recognition: A comprehensive study and new benchmark (2016), https: //api.semanticscholar.org/CorpusID:78363