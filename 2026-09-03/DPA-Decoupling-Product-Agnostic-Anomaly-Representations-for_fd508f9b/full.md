# DPA: Decoupling Product-Agnostic Anomaly Representations for Zero-shot Anomaly Generation

Hang Yao, Yansheng Fu, Ming Liu, Zifei Yan, Yanli Ji, Hongzhi Zhang, Wangmeng Zuo

![](images/4103c992db3481d3f18bba725377dc89bbd0e81800ccf64273827abb6431db8d.jpg)  
Fig. 1: Comparison of zero-shot anomaly generation setting with different anomaly source. (a) shows the real target-product anomalies. Texture pasting-based setting in (b) uses generic texture images as the anomaly source and often generate unrealistic anomaly patterns. Text description-based setting in (c) uses text descriptions as anomaly source. However, text cannot full specify diverse anomaly appearances, and pre-trained anomaly concepts may diverge from real manufacturing anomalies. Our anomaly transfer-based setting in (d) instead learns a anomaly concept from real anomalies of other product and transfers it to a normal target image. This enables generating anomalies that closely resemble the real anomalies in (a).

Abstract—Industrial anomaly detection benefits from anomaly samples, yet newly deployed products typically provide only normal images, making anomaly samples difficult to collect. Zero-shot anomaly generation offers a promising solution which avoids collection of target-product anomalies. However, existing methods mainly rely on texture images or text descriptions as anomaly sources, which often produce unrealistic anomalies. Observing that similar anomalies can recur across different products, we propose anomaly transfer-based zero-shot generation, which reuses real anomalies from existing source products, making target-product anomalies no longer necessary to generate realistic anomalious samples for unseen target products. Since not every anomaly type suits the target product, an anomaly type filtering mechanism first selects plausible source types. To transfer selected anomaly, we propose DPA, a diffusion-based framework that decouples product-agnostic anomaly representations. Instead of directly extracting anomaly representations, DPA learns product-irrelevant anomaly embeddings through training with the mismatched data pair, enabling transferable anomaly concept learning across products. Furthermore, we design an adaptive mask-guided pipeline that leverages adaptive masks to control the positional and geometric plausibility of generated anomalies during generation. A training-free anomaly labeling module is further introduced to produce pixel-level annotations aligned with generated anomalies. Extensive experiments on MVTec-AD, VisA, and a dedicated anomaly-transfer benchmark demonstrate that the proposed setting and DPA generate

more realistic anomalies and significantly improve downstream anomaly detection performance under both zero-shot and fewshot settings. Source code and models will be released.

Index Terms—Anomaly detection, anomaly generation, zeroshot learning.

## I. INTRODUCTION

NDUSTRIAL anomaly detection (IAD) [1]–[3], as a critical step of quality control, is drawing increasing attention in the intelligent industry era. Meanwhile, the rapid development of modern manufacturing has led to diversified demands and constant product iteration. When deploying IAD models [4]–[6] to these continuously introduced new products, both normal and anomalous samples are required for training is typically. In practice, however, while normal samples are readily obtainable, anomalous samples are extremely scarce. To alleviate this data scarcity, a straightforward and practical solution is to generate sufficient and diverse anomalous samples using only normal data, i.e., zero-shot anomaly generation.

Despite their diversity, zero-shot anomaly generation methods follow the same general formulation: an anomaly source is combined with a normal target image to produce a target anomalous image and its annotation. Within this formulation, the most critical element is the anomaly source, that is, the information that specifies what the anomaly should look like. The real target anomalies shown in Fig.1(a) represent the desired output and, ideally, the most reliable anomaly source. However, they are inaccessible under zero-shot setting. Early texture pasting-based methods [7]–[9] use external texture images as anomaly source and insert them into randomly sampled regions of normal images. Though efficient, they produce anomalies that are often unnatural and deviate significantly from real anomaly patterns, as illustrated in Fig.1(b). Recent, text description-base methods [10] use text descriptions as the source and prompt a pre-trained diffusion model (Fig.1(c)). However, text cannot fully specify the appearance of diverse industrial anomalies, and the anomaly concepts learned by the pre-trained generation model diverge from real manufacturing anomalies.Therefore, this anomaly source still lack authenticity, resulting in a substantial gap in visual appearance between generated anomalies and the expected target anomalies.

Although anomalous data are difficult to collect for a newly deployed product, factories often retain real anomaly samples from existing production. Many anomaly concepts, such as scratches, cracks, discoloration, and bent components, recur across related product families and exhibit similar visual patterns. This cross-product commonality makes existing anomalies a more faithful anomaly source than generic textures or text descriptions, while avoiding the costly collection of targetproduct anomalies. Motivated by this observation, we propose anomaly transfer-based zero-shot generation, illustrated in Fig.1(d). In this setting, anomaly concepts extracted from real anomalies of source products are transferred to unseen target products, with no real target anomaly samples required.

However, not every source anomaly is meaningful for the given target product. For example, a bent-lead anomaly is applicable only to products containing leads, whereas transferring it to a bottle would create semantically invalid training data. We therefore introduce a product-aware anomaly type filtering mechanism before transfer. For each target product, semantically incompatible source anomaly types are discarded, and only reasonable types are retained for generation. In practice, we use the vision-language model QWen3-VL to make this compatibility decision before generation, after which the compatibility decision is cached and reused for all subsequent samples. Importantly, the vision-language model only decides whether an anomaly is suitable for the product. The anomaly concept itself is learned from the selected source anomaly.

A straightforward approach to achieve the transfer of selected anomaly types, is to use existing anomaly generation methods [11], [12] to extract anomaly representations from the source anomaly and apply it to a normal target-product image <sup>1</sup>. However, such direct adaptation either fails to inject the anomaly into the target image or produces obvious artifacts, as shown in Fig.2(b) and (c). We attribute such failures to the impure anomaly representations extracted with these methods, where the vanilla feature extraction module inevitably couples anomaly representations with product representations.

![](images/3d939277fdb9ac0239a73d40eccc356d8556e7a70dd9b96ab55400e69b36df66.jpg)  
Fig. 2: Results of direct cross-product anomaly transfer and the proposed DPA. (a) A source anomalous image contains both anomaly and source-product appearance. Directly applying existing anomaly generators to cross-product transfer, as illustrated by (b) AnomalyDiffusion [11] and (c) AnoGen [12], may transfer source-product information together with the anomaly and produce randomly located or misaligned masks. (d) DPA transfers a decoupled anomaly concept to a reasonable location of the target product and produces a mask aligned with the generated anomaly.

This entanglement motivates our Anomaly Concept Decoupling (ACD). First, ACD introduces two learnable embeddings: a product embedding shared across normal and anomalous samples to represent product appearance, and an anomaly embedding dedicated to anomaly semantics. And then, to assign complementary responsibilities to these embeddings, we construct three types of input-target pairs for embeddings training: masked anomaly → masked anomaly, anomaly → normal, and normal → anomaly. Guided by the corresponding embeddings, these pairs require the frozen diffusion model to perform three tasks: preserving, removing, and inserting anomalies, respectively. The preserve pair directly learns from the anomalous region to capture visual anomaly appearance, whereas the remove and insert pairs encourage the embedding to understand abstract anomaly concepts by erasing and adding anomalies. Together, these complementary tasks encourage it to decouple an abstract and transferable anomaly concept, rather than memorizing source-product appearance.

Another problem of these generation methods is the unreasonable mask, which is also shown in Fig.2(b) and (c). Existing methods typically use predefined masks as both generation constraints and ground-truth annotations. On the one hand, the masks are generated via random algorithms, which means that the mask regions may not be the place where a specific anomaly occurs. On the other hand, the generated anomalies are often unaligned with the masks (typically a circle or square), leading to imprecise annotations.

To address the mask issues, we propose an adaptive maskguided pipeline comprising three modules: Initial Mask Generation (IMG), Mask Refinement(MR), and Anomaly Labeling(AL), which operate before, during, and after generation, respectively. Before generation, IMG determines the anomaly location via attention maps between anomaly embeddings and normal image features, and sets the size by sampling from the prior size distribution of existing anomalies, yielding a circular coarse mask as an initial estimate. During generation, MR dynamically refines this mask, aligning it with the evolving anomaly and guiding the anomaly shape toward more plausible forms. After generation, to further produce a high-resolution mask, AL incorporates the refined masks, attention maps, and multi-scale VAE decoder features of normal and anomalous samples, to produce pixel-level precise annotations. These three modules jointly ensure the generated anomalies have reasonable spatial locations and shapes with accurate labels.

To verify the effectiveness of our anomaly transfer-based generation setting and generation pipeline, extensive experiments are conducted on two widely-used datasets (i.e., MVTec-AD [13] and VisA [14], where anomaly embeddings are extracted from one of these two datasets and evaluated on the other one. We also integrate a dedicated dataset for anomaly transfer, comprising both source data for extracting anomaly and target data for anomaly generation. Furthermore, we evaluate our model under the few-shot anomaly generation setting (where a small number of target anomaly samples are allowed), demonstrating the generality and effectiveness of our anomaly generation method. The contributions of this paper are summarized as follows.

• We unify zero-shot anomaly generation as the composition of an anomaly source and a normal target image, and formulate anomaly transfer-based zero-shot generation setting, which uses real anomalies from existing products as a more faithful source.

• A anomaly concept decoupling method is proposed, which decouples product appearance and anomaly concept to two learnable embeddings through complementary preserve, remove, and add tasks.

• We introduce an adaptive mask-guided generation and labeling pipeline which considers location and shape of generated anomalies, and produces high-precision annotations aligned with the generated anomalies.

• Experiments on three datasets, multiple downstream detection methods, and both zero-shot and few-shot settings consistently validate the proposed setting and method.

## II. RELATED WORK

## A. Industrial Anomaly Detection

Due to the scarcity of anomaly data, mainstream IAD algorithms adopt an unsupervised paradigm [2], [9], [15], [16], relying solely on normal samples for detection. Some of these methods design complex network architectures to fit the normal data distribution and treat anomalies as outliers [17]–[21]. Although these algorithms achieve good performance, their complex network structures lead to slow inference speed. In contrast to these architecture-oriented approaches, another line of unsupervised IAD focuses on data synthesis [7], [22], [23]. These methods design simple anomaly synthesis strategies and combine them with lightweight detection networks, achieving decent detection performance. Specifically, they mainly paste texture data onto normal samples to simulate anomalies. These methods are highly efficient and well-suited for practical production deployment. However, due to the significant gap between synthetic anomalies and real anomalies, there is room for further performance improvement. Our goal is to design a zero-shot anomaly generation method that produces more realistic anomalies than the aforementioned synthesis strategies. By using the generated anomalies to train such lightweight detection models, we aim to maintain their low computational cost while further improving detection performance, enabling fast and accurate anomaly detection.

## B. Diffusion model

Based on principles of nonequilibrium thermodynamics [24], diffusion model (DM) [25]–[27] is proposed for image generation, and utilized in a variety of downstream tasks [28]–[31]. Denoising diffusion implicit model (DDIM) [32] considers the reverse process of DM as a non-Markovian process, which speeds up the inference greatly. The latent diffusion model (LDM) [33] conducts training and inference in the latent space, further reducing the resource and time costs. Recently, Text inversion [34] and Dreambooth [35] learn the concept of subjects in a given reference set and synthesize novel renditions of them in different contexts for customized generation. However, these methods heavily rely on real anomaly data, requiring the collection and annotation of targetspecific anomalous samples, which significantly hinders the practical deployment of anomaly generation models.

## C. Anomaly Generation

The scarcity of anomaly data has drawn considerable research attention, prompting the development of various anomaly generation methods to address this challenge. Conventional methods [7], [23], [36] do not rely on target anomaly data and can be regarded as zero-shot anomaly generation approaches. These methods simulate anomalies by replicating normal regions, texture patches, or adding noise perturbations. Although capable of synthesizing anomalies, they suffer from limited realism. Inspired by diffusion-based customized generation approaches [34], [35], few-shot anomaly generation methods [11], [12], [37]–[39] learn anomaly concepts from target anomaly samples to generate anomalies with enhanced realism. However, such methods require prior collection and annotation of anomalous samples, which is difficult under the premise of anomaly data scarcity. Recent zero-shot anomaly generation methods [10], [40] employ detailed textual descriptions to guide pretrained generation models for zeroshot anomaly generation. Nevertheless, for common pretrained generation models, the textual descriptions of certain anomaly concepts often misalign with the desired visual characteristics. In summary, existing methods struggle to simultaneously achieve realistic anomaly generation and freedom from collecting target anomaly data, leading to significant limitations in practical applications.

## III. METHODOLOGY

We first define a common paradigm for zero-shot anomaly generation:

$$
\underbrace { \mathcal { S } } _ { \mathrm { a n o m a l y ~ s o u r c e } } + \underbrace { I _ { p } ^ { n } } _ { \mathrm { t a r g e t ~ n o r m a l } } \xrightarrow { \mathcal { G } } ( \hat { I } _ { p } ^ { a } , \hat { m } _ { p } ^ { a } ) ,\tag{1}
$$

where $s$ provides the anomaly prior, $I _ { p } ^ { n }$ is a normal image of target product $p ,$ and $\mathcal { G }$ produces a target anomalous image $\hat { I } _ { p } ^ { a }$ and its label $\hat { m } _ { p } ^ { a } .$ . Texture-pasting methods instantiate the paradigm as $\mathcal { G } ( \mathcal { T } , I _ { p } ^ { n } )$ , where T is a generic texture image; text-guided methods use $\mathcal { G } ( d ^ { a } , I _ { p } ^ { n } )$ , where $d ^ { a }$ is an anomaly description. DPA instead uses $\mathcal { G } ( \mathcal { D } _ { s } ^ { a } , I _ { p } ^ { n } )$ , where $\mathcal { D } _ { s } ^ { a }$ contains real anomalous images, providing a substantially more faithful anomaly source.

![](images/86b639af93aabbb6df4a91e479a722b8782c773c8055432a092c62f8d8db725f.jpg)  
Fig. 3: Decoupling training with three input–target pairs. The preserve, remove, and add tasks force the anomaly embeddings to learn pure anomaly concept form product appearance.

Accordingly, DPA follows a training-to-inference pipeline. During training, Anomaly Concept Decoupling (ACD) learns a decoupled anomaly embedding from real source anomalies, by trained with the preserve, remove, and add training pairs. Once the anomaly representations have been decoupled, inference begins with product-aware anomaly type filtering. For each target product, DPA evaluates the available source anomaly types and retains only those that are semantically compatible with the product. For every retained anomaly-product pair, the decoupled anomaly is combined with the target-product to construct a joint text prompt, e.g., “A [bent lead] [pcb3].” This prompt is encoded as learned embeddings, and conditions the frozen diffusion model to transfer the selected anomaly to a target normal image. The subsequent Adaptive Mask-Guided Generation determines a plausible location and size and progressively controls the anomaly shape with adaptive masks. After generation, a training-free Anomaly Labeling converts the guided mask into an aligned high-resolution mask.

## A. Anomaly Concept Decoupling

Although a real source anomaly provides a faithful visual prior in 1, it contains two inseparable factors: the source product and its anomaly. To obtain the product-irrelevant anomaly concept, we decouple the anomaly concept from product information. The design of the training strategy is illustrated in Fig.3. We first introduce two learnable embeddings: a product embedding shared by both normal and anomalous samples, alongside a dedicated anomaly embedding activated only for abnormal samples. Guided by different embeddings, the diffusion model is required to complete three tasks by training with intentionally mismatched input-target pairs: retaining the anomaly, removing real anomalies and adding nonexistent ones. This training strategy enhances the semantic representation of the anomaly embedding beyond pixel-level reconstruction, effectively decoupling anomaly patterns from product-specific information.

1) Decoupling Training Strategy.: Given a spatially aligned anomalous–normal pair $( I ^ { a } , I ^ { n } )$ (the normal counterpart $I ^ { n }$ is obtained by completing the masked anomalous region with inpainting model [33]), we encode them as latent features $x ^ { a }$ and $x ^ { n }$ using a pre-trained VAE. Let $e ^ { p }$ and $e ^ { a }$ denote the learnable product and anomaly embeddings, respectively. As shown in Fig.3, the input feature $x ^ { i }$ , target feature $x ^ { t }$ , and conditioning embedding e are sampled from<sup>2</sup>

$$
\begin{array} { c } { { ( x ^ { i } , x ^ { t } , e ) \in \{ ( m ^ { a } \cdot x ^ { a } , m ^ { a } \cdot x ^ { a } , e ^ { a } ) , } } \\ { { ( x ^ { a } , x ^ { n } , e ^ { p } ) , ( x ^ { n } , x ^ { a } , [ e ^ { p } , e ^ { a } ] ) \} , } } \end{array}\tag{2}
$$

where $m ^ { a }$ is the source anomaly mask. The three configurations respectively require the model to (1) preserve the masked anomaly pattern under $e ^ { a } , ( 2 )$ remove the anomaly from $x ^ { a }$ under the product embedding $e ^ { p }$ , and (3) add the anomaly to $x ^ { n }$ under the joint condition $[ e ^ { p } , e ^ { a } ]$ . Because the add and remove tasks share the same product embedding, this construction encourages source-product information to be captured by $e ^ { p }$ and the anomaly concept by $e ^ { a }$

2) Diffusion loss for mismatched pairs: The standard diffusion objective assumes identical input and target features and therefore does not directly provide the correction required by the add and remove pairs above. We adopt the Anomalyoriented Training Paradigm (ATP) loss [9] for training, which supports reconstruction toward a different target. Let $\ v { x } _ { t } ^ { i }$ be the noisy input latent at diffusion timestep t. The ACD objective is

$$
\begin{array} { r } { L _ { A C D } = \| ( \epsilon - \frac { \sqrt { \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { t } } } ( x ^ { i } - x ^ { t } ) ) - \epsilon _ { \theta } ( x _ { t } ^ { i } , t , e ) \| _ { 2 } . } \end{array}\tag{3}
$$

where ϵ is the Gaussian noise that is added to the input feature $x _ { t } ^ { i } .$ , and alpha is the diffusion coefficient at timestep t.

This loss renders the same denoiser compatible with preservation, remove, and add pairs. Importantly, the novelty of ACD lies not in designing a new diffusion loss, but in a training strategy that constructs three input–target pairs to endow the two embeddings with complementary and decoupled semantic concepts.

## B. Adaptive Mask-Guided Generation

After obtaining the decoupled anomaly embeddings, we transfer the learned anomaly concepts to normal images of target products. As illustrated in Fig. 4, we introduce an adaptive mask-guided process for anomaly generation. in which we control the generated content with from two aspects, and spatial attributes with adaptive masks, respectively.

![](images/fca889f6f64e077258cd0c6b6cf53a87b322d208597caf451f8511836cbbcbcc.jpg)  
Fig. 4: Overview of the cross-product anomaly transfer pipeline in DPA. The Adaptive mask-guide generation process proceeds through follows a remove–insert–refine process to generate anomaly. Each stage is controlled by a distinct text prompt and corresponding adaptive masks. For text prompt, we first filter a reasonable anomaly type and build corresponding text prompt. For adaptive masks, (a) initial mask generation first generates an initial mask $m _ { i n i t } .$ , which provides location and size priors. (b) mask refinement then derives three different adaptive masks from $m _ { i n i t }$ to govern the spatial attributes of the three stages, respectively. Finally, (b) an anomaly labeling module fuses refined mask, attention maps and multi-scale VAE feature differences to produce pixel-level labels $m _ { g t }$ precisely aligned with the generated anomalies.

Regarding anomaly content, not every source anomaly type is applicable to given target product. Therefore, before generation, we employ a product-aware anomaly type filtering mechanism to filter anomaly type. Then, each reasonable type is combined with the target-product category to construct a text prompt, which is subsequently encoded into embeddings as the text condition.

To control the spatial attributes of the anomaly, we introduce an adaptive mask-guided process to manipulate these attributes in a stage-wise manner. There are two key modules are involved, including Initial Mask Generation and Mask Refinement. Initial Mask Generation first generates initial mask for controlling plausible location and size. Mask Refinement further refines initial mask to produce three adaptive masks for removing conflicting normal content, inserting anomaly concept and refining the anomaly shape.

Furthermore, after generation, a Generated-Sample Verification is utilized for discarding failed outputs.

1) Product-Aware Anomaly Type Filtering: The availability of real source anomalies does not imply that all of them should be transferred to every target. Let $\left( { \cal I } _ { s } ^ { a } , m _ { s } ^ { a } , c _ { s } \right)$ denote a source anomalous image, its mask, and its anomaly type, and let $( I _ { p } ^ { n } , c _ { p } )$ denote a normal exemplar and the category of target product $p .$ . Before generation, Qwen3-VL [41] receives $( I _ { s } ^ { a } , m _ { s } ^ { a } , c _ { s } , I _ { p } ^ { n } , c _ { p } )$ and judges whether anomaly type $c _ { s }$ could plausibly occur on product $c _ { p } .$ . If the answer is negative, DPA skips this anomaly type and tests the next candidate. If the answer is positive, DPA accepts the source-anomaly–targetproduct pair and executes the subsequent transfer.

This compatibility decision is performed offline once for each source-anomaly-type–target-product candidate and cached for all subsequent samples. Qwen is therefore not invoked during diffusion sampling or downstream detector training and inference.

2) Initial Mask Generation: Initial Mask Generation provides the coarse spatial constraint that determines the anomaly location and size. We construct a circular mask $m _ { i n i t }$ whose center $C$ controls location and whose radius R controls size. The circle is only an initialization and its shape is refined during diffusion rather than used as the final annotation.

Anomalies usually occur in regions that are semantically relevant to the anomaly type. For example, a bent-lead anomaly can only appear in lead regions rather than arbitrary background areas. Therefore, instead of randomly sampling anomaly locations, we first identify potential anomaly regions according to the semantic relationship between the target product and the transferred anomaly concept.

Specifically, we compute a cross-attention map $a _ { i n i t } \in$ $\mathbb { R } ^ { r \times r }$ between anomaly text features $t ^ { a } \in \mathbb { R } ^ { Z \times C _ { 1 } }$ and targetnormal image features $v ^ { n } \in \mathbb { R } ^ { r \times r \times C _ { 2 } }$ from the Stable Diffusion UNet [33]. The text transformer maps the anomaly embedding $e ^ { a }$ to Z token features. We aggregate attention over these tokens and over UNet layers k through L:

$$
a _ { i n i t } = \frac { 1 } { Z ( L - k ) } \sum _ { z = 0 } ^ { Z } \sum _ { l = k } ^ { L } \mathrm { S o f t m a x } ( \frac { Q _ { l } K _ { l } ^ { \top } } { \sqrt { d } } ) ,\tag{4}
$$

$$
Q = \phi _ { q } ( v ^ { n } ) , K = \phi _ { k } ( t ^ { a } ) .\tag{5}
$$

Here, $Q$ and K are the projected query and key features, and d is their channel dimension. Because ACD suppresses sourceproduct appearance in $e ^ { a }$ , high responses in $a _ { i n i t }$ indicate target regions semantically associated with the anomaly rather than regions resembling the source product.

The attention map is subsequently binarized and constrained within the object region mask $m _ { o }$ to obtain the potential anomaly region mask $m _ { n }$

$$
m _ { n } = { \mathrm { B i n a r i z e } } ( m _ { o } \cdot a _ { i n i t } ) .\tag{6}
$$

The anomaly center is randomly sampled from the potential region:

$$
C = S a m p l e ( C _ { i } \mid C _ { i } ( m _ { n } ) = 1 ) .\tag{7}
$$

After determining the location, we control anomaly size with statistics from the source anomaly data. We compute the range $[ R _ { m i n } , R _ { m a x } ]$ of circumcircle radii and sample

$$
R = S a m p l e ( [ R _ { m i n } , R _ { m a x } ] ) .\tag{8}
$$

The sampled center and radius define $m _ { i n i t } \in \mathbb { R } ^ { r \times r } ,$ providing semantic location and empirical size priors while retaining diversity across generated samples.

3) Mask Refinement: The coarse masks can only roughly constrain the position and size of generated anomalies. However, there are two finer masks are required to remove exiting normal contents and refine the shape of generated anomalies, respectively. For these two purposes, mask refinement refines the initial masks under different conditions.

First, although $m _ { i n i t }$ specifies the generation region, normal product structures may still remain within this area and interfere with subsequent anomaly generation. For example, when generating a bent-lead anomaly, the original normal lead should be removed before introducing the anomaly. Therefore, a dedicated removal mask is required to identify the normal product regions that should be erased and to guide unconditional generation for normal-content removal. We obtain such a mask $m _ { r m }$ by intersecting the initial mask $m _ { i n i t }$ with the potential anomaly-prone region $m _ { n } \colon$

$$
m _ { r m } = m _ { i n i t } \cdot m _ { n } ,\tag{9}
$$

The removal mask $m _ { r m }$ therefore erases only the normal structure that occupies the intended anomaly region, preparing it for anomaly insertion while preserving the surrounding target content.

Second, the coarse mask itself cannot accurately align the shape of the generated anomaly. Consequently, it provides only limited spatial guidance and may restrict precise shape control during the later stages of generation. To obtain a mask that better aligns with the generated anomaly, we further refine the initial mask according to the anomaly-related attention map:

$$
m _ { r e f } = \mathrm { B i n a r i z e } ( m _ { i n i t } \cdot a _ { r e f } ) ,\tag{10}
$$

where $\boldsymbol { a } _ { r e f }$ is recomputed according to Eq.4 after anomaly content has begun to emerge. Unlike the fixed circle, $m _ { r e f }$ follows the current anomaly response and provides a shapeaware constraint for the final diffusion stage.

Thus, $m _ { r m }$ and $m _ { r e f }$ serve different purposes: the former clears conflicting normal content, whereas the latter aligns subsequent generation with the emerging anomaly shape.

4) Generation with Adaptive Masks: During generation, we start from a noisy normal latent and generate anomaly with different masks, wihch are activated at different stages of the remove–insert–refine process to progressively control anomaly generation. Specifically, the removal mask $m _ { r m }$ is first used to remove residual normal content within the designated region by unconditional generation. Subsequently, under the guidance of the learned embedding, anomaly content is generated within the coarse mask $m _ { i n i t }$ , which provides a reasonable constraint on anomaly location and size. Finally, the refined mask $m _ { r e f }$ is employed to provide more accurate shape-aware spatial constraints, enabling finer control over anomaly shapes and producing more realistic anomaly structures.

Formally, the mask-guided latent update at timestep t is defined as

$$
x _ { t } ^ { a } = m \cdot \hat { x } _ { t } ^ { a } + ( 1 - m ) \cdot x _ { t } ^ { n } ,\tag{11}
$$

where $\hat { x } _ { t } ^ { a }$ denotes the denoised latent feature at timestep t, and $\boldsymbol { x } _ { t } ^ { n }$ represents the noisy latent feature of the normal sample. The mask $m$ corresponds to the active adaptive mask at the current generation stage, namely $m _ { r m } , m _ { i n i t } .$ , or $m _ { r e f }$

Progressively switching from $m _ { r m }$ to $m _ { i n i t }$ and then $m _ { r e f }$ mirrors the required operations: remove conflicting normal content, insert the anomaly at a plausible position and scale, and refine its final shape.

5) Generated-Sample Verification: Product-aware filtering decides whether an anomaly type is semantically suitable before generation, but diffusion may still fail to express an accepted concept in an sample. We therefore apply a separate output-level check that rejects failed samples. We define the anomaly saliency score s as the difference between the mean attention inside and outside $m _ { r e f } \colon$

$$
s = \mathrm { M e a n } ( m _ { r e f } \cdot a _ { f n l } ) - \mathrm { M e a n } ( ( 1 - m _ { r e f } ) \cdot a _ { f n l } ) .\tag{12}
$$

If s is below a predefined threshold τ , the generated sample is discarded. If more than one third of an inference batch fails this test, we stop the remaining generation for that source– target pair. This is an operational safeguard for synthesis failure and is distinct from the semantic compatibility decision made before transfer.

## C. Anomaly Labeling

The generation masks control diffusion but are not yet suitable as segmentation labels. In particular, $m _ { r e f }$ is only

$6 4 \times 6 4$ and remains a spatial constraint rather than a pixelaccurate description of the generated appearance. We therefore derive the final label from the actual change between the normal and anomalous outputs.

The VAE decoder exposes this change at progressively higher resolutions. Because mask-guided generation preserves content outside the controlled region, discrepancies between the normal and generated features are concentrated within the anomaly regions. Comparing the two multi-scale decoder features therefore complements the semantic but low-resolution attention masks.

Specifically, we consider the latent features and decoded images as the first and final layers of the decoder, denoted by $F ^ { a } = [ x ^ { a } , D _ { 0 } ^ { a } , D _ { 1 } ^ { a } , D _ { 2 } ^ { a } , \hat { I } ^ { a } ]$ and $F ^ { n } = [ x ^ { n } , D _ { 0 } ^ { n } , D _ { 1 } ^ { n } , D _ { 2 } ^ { n } , { \hat { I } } ^ { { \dot { n } } } ]$ for the anomalous and normal samples, respectively. For each scale $d ,$ we compute the cosine distance between $F _ { d } ^ { a }$ and $F _ { d } ^ { n }$ multiply it by $m _ { r e f }$ , normalize, and upsample to obtain multiscale masks:

$$
m _ { g e n } ^ { d } = \mathrm { U p } ( \mathrm { N o r m } ( m _ { r e f } \cdot \mathrm { C o s } ( F _ { d } ^ { a } , F _ { d } ^ { n } ) ) ) .\tag{13}
$$

Finally, the multi-scale masks are integrated with the attention maps $a _ { f n l }$ to produce the final high-resolution groundtruth mask:

$$
m _ { g t } = \mathrm { B i n a r i z e } ( \mathrm { N o r m } ( \mathrm { U p } ( m _ { r e f } \cdot a _ { f n l } ) + { \frac { 1 } { D } } \sum _ { d = 1 } ^ { D } m _ { g e n } ^ { d } ) ) .\tag{14}
$$

The fusion uses $m _ { r e f }$ to suppress irrelevant changes, $a _ { f n l }$ to retain anomaly semantics, and the multi-scale feature distances to recover boundaries. The resulting $m _ { g t }$ is therefore aligned with the generated anomaly rather than inherited from the initial circle.

## IV. EXPERIMENTS

## A. Datasets

We conduct experiments on two widely used industrial anomaly detection datasets, MVTec-AD [13] and VisA [14]. To test cross-dataset transfer, each dataset serves as the source of real anomaly concepts when the other serves as the unseen target; no anomalous image from the target dataset is used for generation training.

Morerover, we further construct a dedicated benchmark, Anomaly Transfer-based Anomaly Detection (ATAD), to isolate the transfer setting. ATAD integrates product categories from MVTec-AD, VisA, and two in-the-wild datasets, Real-IAD [42] and MANTA [43]. Its source and target partitions contain 13 and 16 product categories, respectively. Every target anomaly type has a semantically corresponding source type, but the source and target products differ. This correspondence is metadata inherent to the benchmark and is reported only to characterize its coverage; neither DPA nor the Qwen prompt accesses the mapping. We learn anomaly concepts only from the source partition, generate training data from normal images in the target partition, and evaluate anomaly detection on held-out target images. Supplementary material provides the category mapping and construction details.

## B. Setting

We evaluate two anomaly generation settings:

1) Zero-shot: source normal/anomalous images and target normal training images are available, but no target anomalous image is observed. The complete target test set is reserved for downstream evaluation.

2) Few-shot: a subset of target anomalies is additionally available. Following AnomalyDiffusion [11], we split target anomalous samples by using one third to train the generator and reserving the remaining two thirds for downstream testing.

## C. Implementation Details

We use Stable Diffusion 1.5 [33] as the generator. Only the newly introduced text-encoder embeddings are optimized; other pre-trained components remain frozen. We use Qwen3-VL-8B to filter semantically compatible sourceanomaly/target-product pairs in an offline preprocessing step, and each decision is cached and reused for all images generated from that pair. Generation conditions are formed only for the accepted pairs, and Qwen is not part of per-image generation or downstream inference. For each target category, we generate 2,000 anomalous images. If N anomaly types are compatible with the category, each type contributes $2 0 0 0 / N$ samples. We set the initial noise timestep to $T = 9 0 0$ and use 10 inference steps. Unless stated otherwise, DRAEM [7] is the common downstream detector, so differences reflect the generated training data rather than the detection architecture. All baselines use their official code and released models, and all experiments run on one NVIDIA RTX A6000 GPU. Additional hyperparameters are provided in the supplementary material.

## D. Evaluation Metrics

We evaluate image-level detection and pixel-level localization with Area Under the Receiver Operating Characteristic Curve (AUROC), Average Precision (AP), and maximum F1-score (F1). The prefixes I- and P- denote image- and pixel-level metrics, respectively. Per-Region Overlap (PRO) additionally measures region-level localization quality. The reported mean averages the seven metrics over the evaluated datasets. More evaluation results are available in the supplementary material.

## E. Results of Zero-shot Anomaly Generation

1) Comparison with Zero-shot Generation Methods: We first compare DPA with zero-shot generation methods, including DRAEM [7], CPR [22], RealNet [36], GLASS [23] and AnomalyAny [10]. Here, “DRAEM” denotes its native generation module. Every method supplies generated image– mask pairs to the same DRAEM detector under an identical training protocol, isolating generation quality as the experimental variable.

As reported in Tab I, DPA obtains the best overall mean of 84.91, exceeding the strongest baseline, AnomalyAny (81.44), by 3.47 points. The advantage is particularly clear for localization: relative to the best competing value, DPA improves

TABLE I: Comparison with different zero-shot anomaly generation methods. Each method supplies generated data to the same DRAEM detector. “DRAEM” denotes its native generation module. Bold indicates the best result.
<table><tr><td>AG Methods</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ATAD P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>MVTec-AD P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>VisA P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>Mean</td></tr><tr><td>DRAEM</td><td>93.73</td><td>91.32</td><td>86.91</td><td>96.54</td><td>47.40</td><td>49.23</td><td>85.63</td><td>98.72</td><td>99.22</td><td>97.61</td><td>97.07</td><td>67.01</td><td>65.52</td><td>90.73</td><td>96.88</td><td>97.49</td><td>94.18</td><td>97.33</td><td>29.82</td><td></td><td>35.96</td><td>86.32 81.17</td></tr><tr><td>CPR</td><td>94.37</td><td>92.41</td><td>87.81</td><td>96.16</td><td>44.14</td><td>47.25</td><td>83.67</td><td>98.82</td><td>99.32</td><td>97.76</td><td>96.48</td><td>67.27</td><td>65.86</td><td>89.54</td><td>96.22</td><td>97.00</td><td>92.64</td><td>96.30</td><td>29.70</td><td>38.12</td><td>83.87</td><td>80.70</td></tr><tr><td>RealNet</td><td>93.12</td><td>89.50</td><td>85.16</td><td>95.54</td><td>42.94</td><td>46.61</td><td>84.53</td><td>98.76</td><td>99.25</td><td>97.60</td><td>96.92</td><td>67.09</td><td>64.63</td><td>90.62</td><td>96.93</td><td>97.33</td><td>93.88</td><td>96.69</td><td>29.95</td><td>36.99</td><td>87.21</td><td>80.54</td></tr><tr><td>GLASS</td><td>94.81</td><td>92.79</td><td>89.53</td><td>96.30</td><td>36.46</td><td>43.04</td><td>86.74</td><td>98.89</td><td>99.30</td><td>97.95</td><td>97.15</td><td>67.36</td><td>65.92</td><td>90.82</td><td>97.28</td><td>97.52</td><td>94.50</td><td>97.32</td><td>25.81</td><td>35.43</td><td>89.06</td><td>80.67</td></tr><tr><td>AnomalyAny</td><td>93.87</td><td>91.30</td><td>86.50</td><td>97.81</td><td>42.85</td><td>48.19</td><td>88.17</td><td>98.60</td><td>98.86</td><td>97.86</td><td>97.65</td><td>66.95</td><td>64.51</td><td>91.25</td><td>96.63</td><td>97.50 98.50</td><td>92.52</td><td>98.50</td><td>32.06</td><td>40.15</td><td>88.48</td><td>81.44</td></tr><tr><td>Ours</td><td>96.65</td><td>94.66</td><td>90.80</td><td>98.89</td><td>55.42</td><td>57.00</td><td>91.67</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>98.23</td><td></td><td>95.79</td><td>98.47</td><td>38.33</td><td>43.35</td><td>89.17</td><td>84.91</td></tr></table>

TABLE II: Combination with different synthesis-based anomaly detection methods. Bold represents optimal results.
<table><tr><td>AD Methods</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ATAD P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>MVTec-AD | P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>I-AUROC</td><td>I-AP I-F1</td><td>VisA P-AUROC</td><td></td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>Mean</td></tr><tr><td>DRAEM +Ours</td><td>93.73 96.65</td><td>91.32 94.66</td><td>86.91 90.80</td><td>96.54 98.89</td><td>47.40 55.42</td><td>49.23 57.00</td><td>85.63 91.67</td><td>98.72 99.25</td><td>99.22 99.60</td><td>97.61 98.34</td><td>97.07 98.06</td><td>67.01 75.44</td><td>65.52 71.40</td><td>90.73 94.10</td><td>96.88 98.23</td><td>97.49 98.50</td><td>94.18 95.79</td><td>97.33 98.47</td><td>29.82</td><td>35.96</td><td>86.32</td><td>81.17</td></tr><tr><td>CPR</td><td>96.12</td><td>94.94</td><td>91.30</td><td>99.01</td><td>54.96</td><td>55.62</td><td>93.69</td><td>99.65</td><td>99.87</td><td>99.34</td><td>99.19</td><td>81.91</td><td>75.41</td><td>97.73</td><td>97.02</td><td>97.46</td><td>93.98</td><td>99.12</td><td>38.33 51.92</td><td>43.35 53.03</td><td>89.17 94.23</td><td>84.91 86.93</td></tr><tr><td>+Ours</td><td>96.65</td><td>95.69</td><td>91.98</td><td>99.23</td><td>57.66</td><td>57.00</td><td>95.13</td><td>99.68</td><td>99.89</td><td>99.34</td><td>99.21</td><td>82.34</td><td>75.84</td><td>97.83</td><td>97.37</td><td>97.71</td><td>94.83</td><td>99.25</td><td>54.40</td><td>55.17</td><td>94.69</td><td>87.66</td></tr><tr><td>GLASS +Ours</td><td>95.98 96.42</td><td>93.78 94.85</td><td>90.43 90.93</td><td>99.18 99.32</td><td>39.17 41.52</td><td>49.28 50.89</td><td>94.61 95.88</td><td>99.71 99.73</td><td>99.89 99.91</td><td>99.54 99.33</td><td>99.11 99.10</td><td>69.32 70.47</td><td>70.96 71.44</td><td>95.28 95.41</td><td>97.78 97.79</td><td>98.08 98.08</td><td>95.14 95.24</td><td>98.82 98.85</td><td>43.72 45.93</td><td>47.58 48.21</td><td>91.77 93.32</td><td>84.24 84.89</td></tr></table>

P-AP by 8.02 points on ATAD and 8.08 points on MVTec-AD, and improves P-F1 by 3.20 points on VisA. These gains support effectiveness of DPA.

2) Combination with Different Detection Frameworks: The preceding comparison fixes the detector to isolate generation quality. We next replace the native synthesized data of DRAEM [7], CPR [22], and GLASS [23] with DPAgenerated data while retaining each detection framework. As shown in Tab II, the overall mean increases from 81.17 to 84.91 for DRAEM, from 86.93 to 87.66 for CPR, and from 84.24 to 84.89 for GLASS. The consistent improvements observed across three different detection pipelines suggest that the performance gains of DPA originate from the generated anomaly data, rather than from reliance on any specific detection method.

3) Qualitative Evaluation: Fig.5 presents qualitative comparisons of different zero-shot anomaly generation methods, which use different anomaly-source. Across zipper, pill, and cashew targets, DRAEM pastes fragments of generic texture images that do not reproduce the realistic anomalies. AnomalyAny struggles to describe diverse abnormal appearances such as tooth separation in detail, and its understanding of ’crack is inconsistent with the real concept of cracks. Consequently, there remains a certain gap between the generated anomalies and real-world anomalies. DPA instead conditions generation on existing source anomaly data to produce more realistic anomalies. Additional examples are provided in the supplementary material.

## F. Results of few-shot Anomaly Generation

1) Performance Against Anomaly Generation Methods: Although our method is primarily designed for zero-shot anomaly generation, it can be naturally applied to few-shot settings by using a small number of target anomaly samples for training. We compare our method with several representative few-shot generation approaches, including AnomalyDiffusion [11], AnoGen [12], and TF-IDG [39].

As shown in Tab III, DPA ranks first on ATAD (82.22), MVTec-AD (91.28), and VisA (82.37), yielding an overall mean of 85.29. This is 1.16 points above AnomalyDiffusion, the strongest baseline at 84.13. The same spatial-control and labeling mechanisms therefore remain useful even when the anomaly embeddings are learned directly from target anomaly data. Thus, DPA is not limited to the zero-shot generation setting.

![](images/7faa7414c852a64d819793333828013124caf567013a7f7bd1e088ee3ab8453b.jpg)  
Fig. 5: Qualitative comparison of different zero-shot anomaly generation methods. The first column shows real target anomalies. DRAEM uses the texture images as its anomaly source, AnomalyAny uses the displayed text descriptions, and DPA uses real source-product anomalies. Each generated image includes its predicted mask in the bottom-right inset.

TABLE III: Comparison with different few-shot anomaly generation methods. Results for each dataset represent the mean of all metrics. Bold represents optimal results.
<table><tr><td>AG Methods</td><td>ATAD</td><td>MVTec-AD</td><td>VisA</td><td>Mean</td></tr><tr><td>AnomalyDiffusion</td><td>81.74</td><td>90.94</td><td>79.72</td><td>84.13</td></tr><tr><td>AnoGen</td><td>79.79</td><td>89.93</td><td>79.60</td><td>83.10</td></tr><tr><td>TF-IDG</td><td>79.10</td><td>88.73</td><td>78.41</td><td>82.08</td></tr><tr><td>Ours</td><td>82.22</td><td>91.28</td><td>82.37</td><td>85.29</td></tr></table>

![](images/cba897233003170231df5f55ba3ffd41cce0dd60f5018b79fb824335d7a0c4d9.jpg)  
Fig. 6: Qualitative comparison under the few-shot setting. The first column shows real target anomalies, and the remaining columns show results from AnomalyDiffusion, AnoGen, TF-IDG, and DPA. The bottom-right inset of each generated image is its mask.

2) Qualitative Evaluation: Fig.6 shows the generation results under the few-shot generation setting. AnomalyDiffusion and TF-IDG can place masks on background regions, causing failures for scratch\_neck on screw and bent\_lead on pcb1. AnoGen restricts generation to the foreground product, but this constraint is insufficient, Defects such as bent\_lead may appear in positions without any leads. Our method produces physically plausible anomalies accompanied by well-aligned annotations. Additional results are provided in the supplementary material.

## G. Ablation Study

We now isolate the contribution of each module and test the assumptions behind cross-product transfer. Additional ablations are provided in the supplementary material.

1) Effectiveness of Each Proposed Module: We train DRAEM on MVTec-AD with each variant’s generated samples. Without ACD, anomaly embeddings are learned directly from source anomalies; without IMG, the initial circle is placed randomly; without MR, the initial mask remains fixed throughout generation; and without AL, the low-resolution $m _ { r e f }$ is used as the segmentation label. In Tab IV, the complete model reaches a mean of 90.88, 2.90 points above the native DRAEM-generation baseline. Removing any module reduces the performance of anoomaly detection. Each module is therefore necessary for anomaly transfer-based anomaly generation.

TABLE IV: Ablation of DPA modules on MVTec-AD. “Baseline” uses DRAEM’s native generator; “w/o” removes the indicated DPA module. Bold denotes the best result.
<table><tr><td>Methods</td><td>I-AUROC</td><td>I-AP</td><td>I-F1-max</td><td>P-AUROC</td><td>P-AP</td><td>P-F1-max</td><td>PRO</td><td>Mean</td></tr><tr><td>baseline</td><td>98.72</td><td>99.22</td><td>97.61</td><td>97.07</td><td>67.01</td><td>65.52</td><td>90.73</td><td>87.98</td></tr><tr><td>w/o ACD</td><td>99.18</td><td>99.48</td><td>97.91</td><td>98.14</td><td>72.97</td><td>69.85</td><td>93.67</td><td>90.17</td></tr><tr><td>w/o IMG</td><td>99.20</td><td>99.55</td><td>98.09</td><td>97.70</td><td>72.57</td><td>69.25</td><td>93.70</td><td>90.01</td></tr><tr><td>w/o MR</td><td>98.98</td><td>99.49</td><td>97.73</td><td>97.71</td><td>72.88</td><td>68.77</td><td>92.65</td><td>89.74</td></tr><tr><td>w/o AL</td><td>99.14</td><td>99.50</td><td>98.33</td><td>98.12</td><td>73.39</td><td>69.34</td><td>93.67</td><td>90.21</td></tr><tr><td>ours</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>90.88</td></tr></table>

![](images/6059f348e3cde1fe396a348016851a86bcb5150b14dd2bae26a3130c6eae251f.jpg)  
Fig. 7: Cross-attention maps generated in IMG with and without ACD. After decoupling, the anomaly representation localizes semantically corresponding regions across different products, indicating a more transferable concept that is less dominated by source-product appearance.

2) Qualitative Analysis of Anomaly Concept Decoupling: To show the affect of ACD, Fig.7 visualizes the cross attenion map in IMG, when transfer bent lead from source product pcb1 to pcb2 and transistor. Without ACD, attention is scattered across structures resembling the source product and fails to reliably localize the target leads. With ACD enabled, the same anomaly embedding attends to the lead regions on both targets, even though the transistor differs substantially in appearance from the source PCB. This visualization thus demonstrates that our ACD helps learn anomaly concepts that are disentangled from product-specific features.

3) Impact of Transferring Unreasonable Anomalies: In real-world anomaly transfer scenarios, not all source anomalies are semantically compatible with a given target product. To investigate the effect of transferring such unreasonable anomalies, we consider three settings: (1) transferring only unreasonable anomaly types, which are semantically incompatible with the target product, (2) transferring only reasonable anomaly types, which are compatible and used in our default setting, and (3) transferring all anomaly types without filtering.

As shown in Tab V, transferring only reasonable anomaly types yields better performance compared with transferring unreasonable anomalies. Furthermore, the intermediate result for all types also demonstrates that DPA is not catastrophically sensitive to occasional selection errors, even though reasonable types provide the optimal generated results. In addition, Fig.8 presents several transfer results for unreasonable anomaly types, whose generated outputs are physically unrealistic.

TABLE V: Impact of transferring reasonable and unreasonable anomaly types. Bold represents optimal results.
<table><tr><td>transferred anomaly</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>Mean</td></tr><tr><td>Unreasonable</td><td>99.07</td><td>99.33</td><td>98.12</td><td>97.91</td><td>72.83</td><td>69.39</td><td>93.33</td><td>89.99</td></tr><tr><td>Reasonable type (Ours)</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>90.88</td></tr><tr><td>All type</td><td>99.23</td><td>99.59</td><td>98.21</td><td>98.15</td><td>73.96</td><td>70.25</td><td>93.70</td><td>90.44</td></tr></table>

![](images/2783b5c0aaf8c14385701457f21f9c17d1f05de1b05cd98bda82e9b16717c9cc.jpg)  
Fig. 8: Results of transferring unreasonable anomalies.

4) Impact of Target Anomaly Coverage: The proposed anomaly transfer paradigm aims to transfer generic and common anomaly types across products that are prevalent in realworld manufacturing scenarios (e.g., scratches, cracks, bent lead), rather than product-specific anomalies. Consequently, some target anomalies may not have corresponding counterparts in the source domain. To investigate the impact of such cases, we compare two experimental results: (1) transferring anomaly types that include the target anomaly category, and (2) transferring only anomaly types that exclude the target anomaly category.

As shown in Tab VI, source coverage of the target anomaly type raises the performance of anomaly detection. The gap confirms that a matching source concept is useful, but the model remains competitive without one because other transferred anomalies still teach the detector to separate normal from abnormal appearance. Thus, anomaly transfer is most effective when source coverage is broad, yet it does not require a one-to-one source counterpart for every future target anomaly. If a specific uncovered type is essential, the few-shot protocol provides a natural extension.

## H. Computational Cost

Tab VII reports the cost of generating 2,000 images for each product of MVTec-AD dataset. DPA has a total runtime of 76.4 hours, which is lower than AnomalyDiffusion (205.4 hours), TF-IDG (124 hours) and AnomalyAny (291.5 hours). Although AnoGen is faster, it only generates images at a resolution of $2 5 6 \times 2 5 6$ , whereas DPA operates at $5 1 2 \times 5 1 2$ Furthermore, DPA outperforms AnoGen in inference speed, which yields greater advantages when numerous generated samples are required.

TABLE VI: Impact of target anomaly type coverage. Bold represents the best result.
<table><tr><td>transferred anomaly</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>Mean</td></tr><tr><td>w/o target type</td><td>99.20</td><td>99.50</td><td>98.30</td><td>97.99</td><td>72.83</td><td>69.86</td><td>94.01</td><td>90.24</td></tr><tr><td>with target type e (Ours)</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>90.88</td></tr></table>

TABLE VII: Train and test time cost on MVTec-AD dataset.
<table><tr><td>Method</td><td>Resolution</td><td>Train/hours</td><td>Test/hours</td><td>Total/hours</td></tr><tr><td>DRAEM</td><td>=</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td>AnomalyDiffusion</td><td>256</td><td>153.7</td><td>51.7</td><td>205.4</td></tr><tr><td>AnoGen</td><td>256</td><td>30.5</td><td>12.7</td><td>43.2</td></tr><tr><td>TF-IDG</td><td>512</td><td></td><td>124</td><td>124</td></tr><tr><td>AnomalyAny</td><td>512</td><td></td><td>291.5</td><td>291.5</td></tr><tr><td>Ours</td><td>512</td><td>70.9</td><td>5.5</td><td>76.4</td></tr></table>

## V. CONCLUSION

We introduced anomaly transfer-based zero-shot generation, which converts anomaly scarcity on a new product to a crossproduct transfer problem. The real anomalies from existing products can provide the visual prior, while no anomalous target image is required. Because not every anomaly is suitable for every product, DPA first uses product-aware filtering to retain only reasonable source anomaly types. It then decouples anomaly concept from source-product appearance, and generates the anomaly with adaptive masks, finally deriving a high-resolution label from the generated anomaly. Results on MVTec-AD, VisA, and ATAD show that the resulting data outperform existing zero-shot generate methods, improve multiple downstream detectors, and remain effective under the few-shot generation setting.

## REFERENCES

[1] H. Zhu, Z. Liu, Z. Sun, W. Dong, X. Xiao, Z. Zhang, and Y. Xu, “Anomaly or characteristic: Memory-based coarse-to-fine feature fusion for industrial anomaly detection,” IEEE Transactions on Multimedia, 2026.

[2] B.-B. Gao, “Dual-masked and discriminative reconstruction for unified vision anomaly detection,” IEEE Transactions on Image Processing, 2026.

[3] T. Zhang, W. Pang, and X. Lu, “Tunclip: Adaptive multi-scale zero-shot anomaly detection via vision-language feature infiltration and semantic enhancement,” IEEE Transactions on Multimedia, 2026.

[4] L. Gao, J. Zhang, C. Yang, and Y. Zhou, “Cas-vswin transformer: A variant swin transformer for surface-defect detection,” Computers in Industry, vol. 140, p. 103689, 2022.

[5] Q. Zou, Z. Zhang, Q. Li, X. Qi, Q. Wang, and S. Wang, “Deepcrack: Learning hierarchical convolutional features for crack detection,” IEEE transactions on image processing, vol. 28, no. 3, pp. 1498–1512, 2018.

[6] H. Li, J. Wu, D. Liu, L. Y. Wu, H. Chen, and C. Shen, “Accurate industrial anomaly detection and localization using weakly-supervised residual transformers,” IEEE Transactions on Image Processing, 2026.

[7] V. Zavrtanik, M. Kristan, and D. Skocaj, “Draem-a discriminativelyˇ trained reconstruction embedding for surface anomaly detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8330–8339.

[8] H. Zhang, Z. Wang, D. Zeng, Z. Wu, and Y.-G. Jiang, “Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[9] H. Yao, M. Liu, Z. Yin, Z. Yan, X. Hong, and W. Zuo, “Glad: Towards better reconstruction with global and local adaptive diffusion models for unsupervised anomaly detection,” in European Conference on Computer Vision. Springer, 2024, pp. 1–17.

[10] H. Sun, Y. Cao, H. Dong, and O. Fink, “Unseen visual anomaly generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 25 508–25 517.

[11] T. Hu, J. Zhang, R. Yi, Y. Du, X. Chen, L. Liu, Y. Wang, and C. Wang, “Anomalydiffusion: Few-shot anomaly image generation with diffusion model,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 8, 2024, pp. 8526–8534.

[12] G. Gui, B.-B. Gao, J. Liu, C. Wang, and Y. Wu, “Few-shot anomalydriven generation for anomaly classification and segmentation,” in European Conference on Computer Vision. Springer, 2024, pp. 210– 226.

[13] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9592–9600.

[14] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-thedifference self-supervised pre-training for anomaly detection and segmentation,” in European conference on computer vision. Springer, 2022, pp. 392–408.

[15] F. Yang, P. Jing, W. Wang, F. L. Wang, and Y. Su, “Padnet: Progressivedifference-aware feature reconstruction mechanism for anomaly detection,” IEEE Transactions on Multimedia, 2025.

[16] Z. Yang, T. Zheng, X. Ni, Z. Yan, S. Liu, Y. Yu, H. Chen, Y. Wang, L. Fang, and W. Hao, “Open set industrial surface defect recognition with high frequency feature enhancement and class mutual-information constraint,” IEEE Transactions on Multimedia, 2026.

[17] A. Mousakhan, T. Brox, and J. Tayyub, “Anomaly detection with conditioned denoising diffusion models,” arXiv preprint arXiv:2305.15956, 2023.

[18] J. Guo, S. Lu, W. Zhang, F. Chen, H. Li, and H. Liao, “Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 20 405–20 415.

[19] W. Luo, Y. Cao, H. Yao, X. Zhang, J. Lou, Y. Cheng, W. Shen, and W. Yu, “Exploring intrinsic normal prototypes within a single image for universal anomaly detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9974–9983.

[20] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “Padim: a patch distribution modeling framework for anomaly detection and localization,” in International Conference on Pattern Recognition. Springer, 2021, pp. 475–489.

[21] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler,¨ “Towards total recall in industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 318–14 328.

[22] H. Li, J. Hu, B. Li, H. Chen, Y. Zheng, and C. Shen, “Target before shooting: Accurate anomaly detection and localization under one millisecond via cascade patch retrieval,” IEEE Transactions on Image Processing, 2024.

[23] Q. Chen, H. Luo, C. Lv, and Z. Zhang, “A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization,” in European Conference on Computer Vision. Springer, 2024, pp. 37–54.

[24] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning. PMLR, 2015, pp. 2256–2265.

[25] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[26] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International conference on machine learning. PMLR, 2021, pp. 8162–8171.

[27] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[28] X. Liu, Y. Wei, M. Liu, X. Lin, P. Ren, X. Xie, and W. Zuo, “Smartcontrol: Enhancing controlnet for handling rough visual conditions,” arXiv preprint arXiv:2404.06451, 2024.

[29] Y. Wei, Y. Zhang, Z. Ji, J. Bai, L. Zhang, and W. Zuo, “Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15 943–15 953.

[30] Y. Zhang, Y. Wei, D. Jiang, X. ZHANG, W. Zuo, and Q. Tian, “Controlvideo: Training-free controllable text-to-video generation,” in The Twelfth International Conference on Learning Representations, 2023.

[31] Y. Zhang, Y. Wei, X. Lin, Z. Hui, P. Ren, X. Xie, X. Ji, and W. Zuo, “Videoelevator: Elevating video generation quality with versatile textto-image diffusion models,” arXiv preprint arXiv:2403.05438, 2024.

[32] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2020.

[33] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[34] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or, “An image is worth one word: Personalizing text-to-image generation using textual inversion,” arXiv preprint arXiv:2208.01618, 2022.

[35] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 500–22 510.

[36] X. Zhang, M. Xu, and X. Zhou, “Realnet: A feature selection network with realistic synthetic anomaly for anomaly detection,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 16 699–16 708.

[37] Z. Dai, S. Zeng, H. Liu, X. Li, F. Xue, and Y. Zhou, “Seas: few-shot industrial anomaly image generation with separation and sharing finetuning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 23 135–23 144.

[38] Y. Jin, J. Peng, Q. He, T. Hu, J. Wu, H. Chen, H. Wang, W. Zhu, M. Chi, J. Liu et al., “Dual-interrelated diffusion model for few-shot anomaly image generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 30 420–30 429.

[39] R. Xu, Y.-T. Chiu, T.-I. Chen, O. Chew, Y.-Y. Chuang, and W.-H. Cheng, “Training-free industrial defect generation with diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 24 214–24 223.

[40] Z. Lai, Y. Lu, X. Li, J. Lin, Y. Qu, L. Cao, M. Li, and R. Ji, “Anomalypainter: Vision-language-diffusion synergy for zero-shot realistic and diverse industrial anomaly synthesis,” arXiv preprint arXiv:2503.07253, 2025.

[41] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[42] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, and L. Ma, “Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 883–22 892.

[43] L. Fan, D. Fan, Z. Hu, Y. Ding, D. Di, K. Yi, M. Pagnucco, and Y. Song, “Manta: A large-scale multi-view and visual-text anomaly detection dataset for tiny objects,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 25 518–25 527.

# Supplementary Material

## I. APPENDIX

The content of this supplementary material is organized as follows:

• More implementation details in section II.

• Additional Metrics in section III.

• ATAD dataset in section IV.

• Additional Ablation Study in section V.

• Discussion on Inpainting Image Quality in section VI.

• Results of Anomaly Type Filtering in section VII.

• Additional Results of Anomaly Generation in section VIII

• Additional Results of Anomaly Detection in section IX.

• Additional Visualization in section X.

## II. MORE IMPLEMENTATION DETAILS

## A. More Training Details

In our method, anomaly types and product categories are represented as bracketed text, such as “a [bent lead]”, “a [pcb1]”, and “a [pcb1] with [bent lead]”, respectively. To learn these concepts, we adopt Textual Inversion [1], which adds token ids and corresponding learnable embeddings for each new word in the tokenizer and text encoder. Each product category is represented by 8 learnable embeddings, while each anomaly type is represented by 4 learnable embeddings. These embeddings are jointly optimized during training and serve as the textual representations of products and anomalies throughout the generation process.

During the decoupling training phase, the batch size is set to 6, and the learning rate is 0.032 without scaling. We train the learning embeddings for 3000 steps per anomaly type. In the zero-shot setting, we additionally train the product embeddings with the same training parameters as in the decoupling phase, except for a batch size of 16. In the few-shot setting, we directly use the trained decoupled product embeddings without further training.

For downstream anomaly detection, we directly adopt parameters of the original algorithm.

## B. More inference details

During generation, we load the decoupled anomaly embeddings and the additionally trained target-product embeddings into the text encoder. The prompt combines the target product with the selected anomaly type. Starting from a noisy normal sample at timestep $T = 9 0 0 \mathrm { { \ : } }$ , the diffusion model performs 10 denoising steps to generate an anomalous image. The adaptive mask is switched progressively during these 10 steps. Specifically, step 0 uses the removal mask $m _ { r m }$ to remove conflicting normal content, steps 1–6 use the initial mask $m _ { i n i t }$ to insert the anomaly with plausible location and size, and steps 7–9 use the refined mask $m _ { r e f }$ to refine the anomaly shape. In the Generated-Sample Verification, the anomaly saliency threshold is set to $\tau = 0 . 0 5$

## III. ADDITIONAL METRICS

Besides the standard anomaly detection metrics, we additionally report Accuracy (ACC) and Intersection over Union (IoU) for methods with binary segmentation outputs of DRAEM [2] in section IX. These metrics are particularly relevant in practical industrial applications.

In addition, to evaluate the quality of generated anomalies, we adopt three commonly used metrics. For zero-shot generation, since there is no strict target category sample, we evaluate the Inception Score (IS). For few-shot generation, we use the commonly used LPIPS (IC-LPIPS) [3] and Kernel Inception Distance (KID) [4].

## IV. ATAD DATASET

To facilitate research on anomaly transfer, we construct a new benchmark named the Anomaly Transfer-based Anomaly Detection dataset (ATAD) by integrating product categories and anomaly types from four widely used industrial anomaly detection datasets, including the manually curated datasets MVTec-AD [5] and VisA [6], as well as the in-the-wild datasets Real-IAD [7] and MANTA [8].

ATAD is divided into a source subset that provides reference anomalies and a target subset used for generation and detection. Each target anomaly category has at least one semantically corresponding source type. Example correspondences are shown in fig. 1, and the complete many-to-many mapping is provided in ATAD\_anomaly\_type\_match.csv. The mapping is metadata inherent to the benchmark and is provided only to document its anomaly-type coverage. It is not an input to DPA, the Qwen prompt, prompt construction, training, or hyperparameter selection. Our experiments instead use Qwen3-VL-8B to select plausible source–target pairs without consulting the mapping or any target anomaly label.

Detailed dataset statistics are summarized in table I. The source subset contains 13 product categories, including 6,308 normal training images, 2,170 normal test images, and 1,886 anomalous test images. The target subset contains 16 product categories, including 8,555 normal training images, 2,858 normal test images, and 1,993 anomalous test images. The selected products cover diverse industrial scenarios, ranging from single-instance objects to multi-instance products (e.g., candles, capsules, and macaroni2) and structurally complex products (e.g., pcb2 and pcb3), providing a challenging and diverse benchmark for anomaly transfer research.

## V. ADDITIONAL ABLATION STUDY

Influences of Threshold in Mask Binarization. We further investigate the impact of different threshold values in the mask binarization process during generation. The results presented in the table II indicate that a threshold of 0.55 yields the optimal detection performance overall.

![](images/d585644cf7f49e0c9372b3b2fcb740d8869bc48eeda81575f8f1c9e088e6b415.jpg)  
Fig. 1. Similarity between source and target anomalies in ATAD dataset.

TABLE I  
STATISTICAL OVERVIEW OF THE ATAD DATASET. FOR EACH CATEGORY, THE NUMBER OF TRAINING SAMPLES, NORMAL TEST SAMPLES, ANOMALOUS TEST SAMPLES AND ANOMALY TYPES IS PROVIDED.
<table><tr><td>dataset</td><td>category</td><td>train (normal)</td><td>test (normal)</td><td>test (anomalous)</td><td>anomaly type</td></tr><tr><td rowspan="9">source sub-dataset</td><td rowspan="3">bottle_cap capsule_mvtec fire_hood hazelnut macaronil oblong_tab t pcb1 pcb4</td><td rowspan="9">370 219 418 391 900 509</td><td rowspan="9">369 23 418 40 100</td><td>263</td><td>3 5</td></tr><tr><td>109 169</td><td></td></tr><tr><td></td><td>4</td></tr><tr><td></td><td>70 100</td><td>4 5</td></tr><tr><td></td><td>36</td><td>4</td></tr><tr><td></td><td>100</td><td>5</td></tr><tr><td>904 904</td><td>100</td><td>6</td></tr><tr><td>578</td><td></td><td></td></tr><tr><td></td><td>38 469</td><td>3 3</td></tr><tr><td rowspan="3"></td><td>wood</td><td>362 247</td><td>361 19</td><td>280 49</td><td>4 4</td></tr><tr><td>zipper_mvtec total</td><td>240</td><td>32</td><td>103</td><td>6</td></tr><tr><td></td><td>6308</td><td>2170</td><td>1886</td><td>56</td></tr><tr><td rowspan="10">target sub-dataset</td><td rowspan="9">cable candle capsule_manta capsules cashew eraser</td><td rowspan="9">224 900</td><td>58</td><td>35</td><td>3</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>90</td><td>4</td></tr><tr><td></td><td>40</td><td>2</td></tr><tr><td></td><td>50 78</td><td>3 4</td></tr><tr><td>50 389</td><td>235</td><td>4</td></tr><tr><td></td><td>80</td><td>4</td></tr><tr><td>450 900</td><td>100</td><td>5</td></tr><tr><td></td><td>86</td><td>4</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="9">pcb3 pill</td><td>905 267</td><td>101</td><td>77</td><td>4</td></tr><tr><td></td><td>26</td><td>96</td><td>4</td></tr><tr><td>442</td><td>442</td><td>118</td><td>4</td></tr><tr><td>371 620</td><td>370</td><td>261</td><td>4</td></tr><tr><td>toy_brick</td><td>120</td><td>31</td><td>2</td></tr><tr><td>woodstick</td><td>442</td><td>116</td><td>4</td></tr><tr><td>zipper_realiad</td><td>250</td><td>500</td><td>4</td></tr><tr><td>total</td><td></td><td></td><td></td></tr><tr><td></td><td>2858</td><td>1993</td><td>59</td></tr></table>

TABLE II  
COMPARISON OF DIFFERENT THRESHOLDS. BOLD REPRESENTS OPTIMAL RESULTS.
<table><tr><td>Threshold</td><td>I-AUROC</td><td>I-AP</td><td>I-F1-max</td><td>P-AUROC</td><td>P-AP</td><td>P-F1-max</td><td>PRO</td><td>Mean</td></tr><tr><td>0.45</td><td>99.07</td><td>99.43</td><td>98.11</td><td>98.16</td><td>73.00</td><td>68.83</td><td>93.26</td><td>89.98</td></tr><tr><td>0.50</td><td>99.12</td><td>99.40</td><td>97.83</td><td>98.18</td><td>73.53</td><td>69.67</td><td>93.59</td><td>90.19</td></tr><tr><td>0.55 (Ours)</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>90.88</td></tr><tr><td>0.60</td><td>99.13</td><td>99.48</td><td>98.10</td><td>97.67</td><td>71.18</td><td>68.57</td><td>93.22</td><td>89.62</td></tr></table>

TABLE III  
ABLATION OF NORMAL REGION REMOVAL (NRR) AND GENERATED-SAMPLE VERIFICATION (GSF). BOLD DENOTES THE BEST RESULT.
<table><tr><td></td><td>I-AUROC</td><td>I-AP</td><td>I-F1-max</td><td>P-AUROC</td><td>P-AP</td><td>P-F1-max</td><td>PRO</td><td>Mean</td></tr><tr><td>w/o GSF</td><td>98.99</td><td>99.41</td><td>97.78</td><td>97.79</td><td>72.40</td><td>68.73</td><td>93.57</td><td>89.81</td></tr><tr><td>w/o NRR</td><td>99.09</td><td>99.46</td><td>98.13</td><td>97.90</td><td>73.50</td><td>69.36</td><td>93.84</td><td>90.18</td></tr><tr><td>ours</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>90.88</td></tr></table>

TABLE IV

QUALITY OF THE COMPLETED INPAINTING-NORMAL COUNTERPARTS ON MVTEC-AD. LOWER KID AND FID INDICATE A DISTRIBUTION CLOSER TO THE REAL NORMAL DATA.
<table><tr><td>Computed with real normal images</td><td>KID↓</td><td>FID↓</td></tr><tr><td>real anomalous images</td><td>0.0956</td><td>114.64</td></tr><tr><td>inpainting-normal images</td><td>0.0517</td><td>71.83</td></tr></table>

Impact of Normal Region Removal and Generated-Sample Verification.

As shown in table III, introducing the proposed Normal Region Removal (NRR) module and Generated-Sample Verification (GSF) module consistently improves all evaluation metrics. The NRR module removes normal content within the mask region via unconditional generation before anomaly generation, preventing interference from original structures. GSF removes generated samples that lack visible anomalies despite having anomaly labels. Without such filtering, these mislabeled normal samples would pollute the training set and harm downstream detection. The results confirm the importance of both modules for anomaly detection performance.

## VI. DISCUSSION ON INPAINTING IMAGE QUALITY

To rule out the possibility that poor inpainting-normal images degrade anomaly generation, we adopt two complementary measures. First, during training, we paste the original anomaly back onto the inpainted image with mask to further eliminate residual differences in the normal regions. This ensures that even if the inpainted image contains minor artifacts, they will not be mistakenly learned as anomaly patterns. Second, we evaluate the distributional fidelity of the inpainted results. Using KID and FID on MVTec-AD, we compare the distance from inpainted images to real normal samples against that from raw anomalies to normal samples. The significantly smaller distance of inpainted images confirms that the inpainting model produces satisfying counterparts. fig. 1 further show no severe artifacts in the completed regions. This ensures that subsequent training benefits from reliable paired data.

## VII. RESULTS OF ANOMALY TYPE FILTERING

Our method transfers source anomalies to target products, which inherently involves the selection of anomaly types. To facilitate anomaly transfer, we systematically split anomaly data and renamed anomaly types. Comprehensive details are included in the additional supplementary materials MVTec-AD\_anomaly.csv (for MVTec-AD dataset), VisA\_anomaly.csv (for VisA dataset), ATAD\_source\_anomaly.csv and ATAD\_target\_anomaly.csv (for ATAD dataset).

![](images/b8e5487b04be13f8295212292d912231f98d573a108202672965c71d0d65d916.jpg)  
Fig. 2. Qualitative examples of inpainted pseudo-normal images.

Because semantically incompatible transfers can degrade detection, Qwen3-VL filters source–target pairs before generation. It receives the source anomalous image, its mask and type, together with a normal exemplar and the category of the target product, then judges compatibility under a fixed instruction. This is a one-time offline decision for each source-anomaly-type–target-product candidate. The result is cached and reused for all generated samples, so Qwen is not called during diffusion sampling or downstream detector training and inference. The complete filtered outputs are provided in the additional supplementary materials:VisA2MVTec-AD\_filter.csv for MVTec-AD, MVTec-AD2VisA\_filter.csv for VisA, and source2target\_filter.csv for ATAD.

## VIII. ADDITIONAL RESULTS OF ANOMALY GENERATION

Tables V and VI quantify generation quality. In the zeroshot generation setting, target anomaly types are unavailable during training, so generated and real defects cannot be paired by type; we therefore report IS but not IC-LPIPS or KID. In the few-shot generation setting, real target anomalies provide valid same-type references, allowing IC-LPIPS and KID to measure diversity. DPA obtains the best mean in both tab s.

## IX. ADDITIONAL RESULTS OF ANOMALY DETECTION

Deterministic Detection Performance. Since DRAEM produces binary segmentation outputs, its predictions can be interpreted as deterministic metrics that explicitly identify the presence and location of anomalies within input images. In practical industrial deployment, we require such deterministic outputs to make reliable decisions about whether to accept or reject tested products. As shown in table VII, under both zero-shot and few-shot generation settings, our method brings significant improvements in deterministic metrics, which holds substantial practical importance for real-world industrial applications.

Performance Against Unsupervised IAD Methods. The zero-shot generation setting uses no target anomalous image during training and is therefore comparable to unsupervised IAD. We additionally compare with SimpleNet [9],

TABLE V  
IS COMPARISON UNDER THE ZERO-SHOT SETTING ON MVTEC-AD. BOLD DENOTES THE BEST RESULT.
<table><tr><td>Class</td><td>AnomalyAny</td><td>DPA</td></tr><tr><td>bottle</td><td>2.17</td><td>1.47</td></tr><tr><td>cable</td><td>2.95</td><td>1.89</td></tr><tr><td>capsule</td><td>1.71</td><td>1.90</td></tr><tr><td>carpet</td><td>1.24</td><td>1.26</td></tr><tr><td>grid</td><td>2.32</td><td>2.49</td></tr><tr><td>hazelnut</td><td>2.19</td><td>2.06</td></tr><tr><td>leather</td><td>1.37</td><td>3.04</td></tr><tr><td>metal_nut</td><td>1.53</td><td>1.53</td></tr><tr><td>pill</td><td>2.40</td><td>1.91</td></tr><tr><td>screw</td><td>1.20</td><td>1.69</td></tr><tr><td>tile</td><td>3.44</td><td>2.54</td></tr><tr><td>toothbrush</td><td>1.59</td><td>1.51</td></tr><tr><td>transistor</td><td>1.50</td><td>1.67</td></tr><tr><td>wood</td><td>1.97</td><td>3.02</td></tr><tr><td>zipper</td><td>1.88</td><td>1.75</td></tr><tr><td>mean</td><td>1.96</td><td>1.98</td></tr></table>

TABLE VI  
IC-LPIPS AND KID COMPARISONUNDER THE FEW-SHOT SETTING ON MVTEC-AD DATASET. BOLD REPRESENT OPTIMAL RESULTS.
<table><tr><td rowspan="2">class</td><td colspan="2">Anomalydiffusion</td><td colspan="2">AnoGen</td><td colspan="2">DPA</td></tr><tr><td>IC-L↑</td><td>KID↓</td><td>IC-L↑</td><td>KID↓</td><td>IC-L↑</td><td>KID↓</td></tr><tr><td>bottle</td><td>0.15</td><td>0.07</td><td>0.12</td><td>0.12</td><td>0.15</td><td>0.09</td></tr><tr><td>cable</td><td>0.38</td><td>0.07</td><td>0.36</td><td>0.11</td><td>0.38</td><td>0.04</td></tr><tr><td>capsule</td><td>0.16</td><td>0.04</td><td>0.17</td><td>0.05</td><td>0.19</td><td>0.11</td></tr><tr><td>carpet</td><td>0.21</td><td>0.13</td><td>0.22</td><td>0.12</td><td>0.25</td><td>0.10</td></tr><tr><td>grid</td><td>0.39</td><td>0.08</td><td>0.41</td><td>0.09</td><td>0.34</td><td>0.05</td></tr><tr><td>hazelnut</td><td>0.27</td><td>0.02</td><td>0.29</td><td>0.02</td><td>0.29</td><td>0.02</td></tr><tr><td>leather</td><td>0.33</td><td>0.15</td><td>0.35</td><td>0.13</td><td>0.32</td><td>0.10</td></tr><tr><td>metal_nut</td><td>0.24</td><td>0.08</td><td>0.21</td><td>0.12</td><td>0.23</td><td>0.08</td></tr><tr><td>pill</td><td>0.23</td><td>0.03</td><td>0.23</td><td>0.06</td><td>0.25</td><td>0.04</td></tr><tr><td>screw</td><td>0.25</td><td>0.01</td><td>0.26</td><td>0.02</td><td>0.27</td><td>0.02</td></tr><tr><td>tile</td><td>0.48</td><td>0.28</td><td>0.46</td><td>0.29</td><td>0.44</td><td>0.16</td></tr><tr><td>toothbrush</td><td>0.15</td><td>0.03</td><td>0.16</td><td>0.05</td><td>0.16</td><td>0.03</td></tr><tr><td>transistor</td><td>0.29</td><td>0.15</td><td>0.27</td><td>0.16</td><td>0.27</td><td>0.11</td></tr><tr><td>wood</td><td>0.33</td><td>0.11</td><td>0.34</td><td>0.07</td><td>0.36</td><td>0.07</td></tr><tr><td>zipper</td><td>0.23</td><td>0.11</td><td>0.23</td><td>0.13</td><td>0.20</td><td>0.04</td></tr><tr><td>mean</td><td>0.27</td><td>0.09</td><td>0.27</td><td>0.10</td><td>0.28</td><td>0.07</td></tr></table>

Dinomaly [10], and INP-Former [11]. For a fair evaluation, center-crop augmentation is disabled because it can remove anomalous regions. Table VIII shows that detection models trained with DPA-generated data remain competitive with these unsupervised approaches.

Detailed Detection and Segmentation Results. In addition, we also report the detailed detection and segmentation results for each category on the MVTec-AD, VisA and ATAD datasets under zero-shot setting. table IX, table X and table XI to presents the results of two representative methods (DRAEM [2] and AnomalyAny [12]) and our proposed approach.

![](images/087175f43121e90ebc6af462100ef412a09c31cf4d577e1a7b401f51a114b28a.jpg)

![](images/df8001a956675774853b545fd7b96aaa95bd746dae7d3cc3edbd468651e85012.jpg)

![](images/2c30b95d2c65e854a5027f36c0979ebc4c9dd279b36ae9b94ce1743b0ad7e998.jpg)  
Fig. 3. Anomaly localization results of DRAEM, which is trained with data from different generation methods under zero-shot generation setting. Each group shows the input anomaly, the predictions, and the ground truth.

![](images/fad403e0f2568e3bcc6d9e7317a5421d2f45ff2cb9903b130c2e9ee63ef2dfa2.jpg)

![](images/8cf7c6906ec19de1f2db69d4d5498a04b2022f0d7414e5ee441ca65609c9b899.jpg)

![](images/c56d07513f8b710c8beb37182eddbd0ab142494513f65efde11414d80dccaa09.jpg)  
Fig. 4. Anomaly localization results of DRAEM, which is trained with data from different generation methods under few-shot generation setting. Each group shows the input anomaly, the predictions, and the ground truth.

TABLE VII  
COMPARISON OF DETERMINISTIC METRICS UNDER ZERO-SHOT AND FEW-SHOT SETTINGS. BOLD DENOTES THE BEST RESULT.
<table><tr><td rowspan="2">AD Methods</td><td colspan="2">ATAD</td><td colspan="2">MVTec-AD</td><td colspan="2">VisA</td></tr><tr><td>ACC</td><td>IoU</td><td>ACC</td><td>IoU</td><td>ACC</td><td>IoU</td></tr><tr><td colspan="7">zero-shot setting</td></tr><tr><td>DRAEM</td><td>64.85 66.01</td><td>58.62</td><td>68.81</td><td>54.14</td><td>56.67</td><td>55.14</td></tr><tr><td>CPR</td><td>64.02</td><td>60.08</td><td>60.19 69.46</td><td>50.28</td><td>55.78</td><td>52.06 54.98</td></tr><tr><td>RealNet</td><td>60.09</td><td>60.73</td><td></td><td>54.79</td><td>53.82</td><td></td></tr><tr><td>GLASS</td><td>66.47</td><td>55.89</td><td>67.08</td><td>55.36</td><td>52.17</td><td>52.82</td></tr><tr><td>AnomalyAny</td><td></td><td>57.09</td><td>67.14</td><td>49.56</td><td>52.47</td><td>49.36</td></tr><tr><td>Ours</td><td>70.15</td><td>61.96</td><td>75.45</td><td>59.26</td><td>62.93</td><td>57.90</td></tr><tr><td colspan="7">few-shot setting</td></tr><tr><td>AnomalyDiffusion</td><td>73.37</td><td>65.69</td><td>74.55</td><td>62.87</td><td>62.46</td><td>61.44</td></tr><tr><td>AnoGen</td><td>74.07</td><td>66.08</td><td>77.09</td><td>61.22</td><td>65.68</td><td>61.77</td></tr><tr><td>TF-IDG</td><td>70.69</td><td>66.93</td><td>65.22</td><td>55.16</td><td>65.96</td><td>59.64</td></tr><tr><td>Ours</td><td>74.72</td><td>70.71</td><td>77.66</td><td>63.33</td><td>67.16</td><td>63.72</td></tr></table>

## X. ADDITIONAL VISUALIZATION

Figures 3 and 4 compare localization under the zero-shot and few-shot protocols, respectively. In every case, the same DRAEM anomaly detection model is trained with data produced by the listed generation methods, so differences reflect the generated images and labels.

Additional zero-shot generation results are presented in fig. 5, while few-shot generation results are provided in fig. 6.

## REFERENCES

[1] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or, “An image is worth one word: Personalizing text-to-image generation using textual inversion,” arXiv preprint arXiv:2208.01618, 2022.

[2] V. Zavrtanik, M. Kristan, and D. Skocaj, “Draem-a discriminativelyˇ trained reconstruction embedding for surface anomaly detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8330–8339.

[3] U. Ojha, Y. Li, J. Lu, A. A. Efros, Y. J. Lee, E. Shechtman, and R. Zhang, “Few-shot image generation via cross-domain correspondence,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 10 743–10 752.

[4] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demysti-´ fying mmd gans,” arXiv preprint arXiv:1801.01401, 2018.

[5] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9592–9600.

[6] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-thedifference self-supervised pre-training for anomaly detection and segmentation,” in European conference on computer vision. Springer, 2022, pp. 392–408.

[7] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, and L. Ma, “Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 883–22 892.

![](images/d866d2f147c039151b58e787c9419ec3f98aba13804424864cfe4056b58106ff.jpg)

![](images/cf2806f843caae1dec42a91bbcbeedbcf447be55c404facaa31fd770431ea394.jpg)

![](images/255c60b5986b759f141dbae8e76729f29bb8f52bd5d304982a547c637caadb68.jpg)  
Fig. 5. Additional zero-shot anomaly generation results for DRAEM, AnomalyAny, and DPA.

![](images/bcc42a411bdde8e11a4c714f616eefeb79aba61455860b4c178321d4af4c21f7.jpg)

![](images/3f95e176186de577af8e35a5115c8d9765ea3bb12ae256e2cde98d881bb9a838.jpg)  
Fig. 6. Additional few-shot generation results for AnomalyDiffusion, AnoGen, TF-IDG, and DPA.

![](images/1f8f0d62a45f4e8f79a05091a296b9ddcf11f31bfdb8af7797182c7c8eda277f.jpg)

![](images/464803030a3f537045c024ea1cb15ef69502fb991e8f34996d98c951f3415f16.jpg)

TABLE VIII  
COMPARISON WITH SOTA UNSUPERVISED ANOMALY DETECTION METHODS. BOLD REPRESENTS OPTIMAL RESULTS.
<table><tr><td>AD Methods</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ATAD P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>I-AUROC</td><td>I-AP I-F1</td><td>MVTec-AD</td><td>P-AUROC</td><td>P-AP</td><td>P-F1 PRO</td><td></td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>VisA P-AUROC</td><td>P-AP</td><td>P-F1</td><td>PRO</td><td>Mean</td></tr><tr><td>DRAEM</td><td>93.73</td><td>91.32</td><td>86.91</td><td>96.54</td><td>47.40</td><td>49.23</td><td>85.63</td><td>98.72</td><td>99.22</td><td>97.61</td><td>97.07</td><td>67.01</td><td>65.52</td><td>90.73</td><td>96.88</td><td>97.49</td><td>94.18</td><td>97.33</td><td></td><td>29.82</td><td>35.96</td><td>81.17</td></tr><tr><td>SimpleNet</td><td>92.34</td><td>91.07</td><td>85.24</td><td>98.26</td><td>30.89</td><td>36.62</td><td>93.32</td><td>98.92</td><td>99.55</td><td>98.86</td><td>97.93</td><td>51.94</td><td>54.33</td><td>92.25</td><td>96.16</td><td>96.91</td><td>92.64</td><td>98.46</td><td>33.16</td><td>37.11</td><td>86.32 92.29</td><td>79.44</td></tr><tr><td>CPR</td><td>96.12</td><td>94.94</td><td>91.30</td><td>99.01</td><td>54.96</td><td>55.62</td><td>93.69</td><td>99.65</td><td>99.87</td><td>99.34</td><td>99.19</td><td>81.91</td><td>75.41</td><td>97.73</td><td>97.23</td><td>97.62</td><td>94.71</td><td>99.13</td><td>51.46</td><td>52.99</td><td>93.89</td><td>86.94</td></tr><tr><td>GLASS</td><td>95.98</td><td>93.78</td><td>90.43</td><td>99.18</td><td>39.17</td><td>49.28</td><td>94.61</td><td>99.71</td><td>99.89</td><td>99.54</td><td>99.11</td><td>69.32</td><td>70.96</td><td>95.28</td><td>97.78</td><td>98.08</td><td>95.14</td><td>98.82</td><td>43.72</td><td>47.58</td><td>91.77</td><td>84.24</td></tr><tr><td>Dinamaly</td><td>96.63</td><td>95.25</td><td>91.60</td><td>99.30</td><td>47.63</td><td>51.37</td><td>96.29</td><td>99.69</td><td>99.87</td><td>99.21</td><td>98.55</td><td>66.80</td><td>67.51</td><td>95.37</td><td>98.54</td><td>98.79</td><td>95.48</td><td>99.10</td><td>49.24</td><td>52.88</td><td>94.78</td><td>85.42</td></tr><tr><td>INP-former</td><td>96.50</td><td>94.81</td><td>91.51</td><td>99.11</td><td>53.01</td><td>55.54</td><td>95.69</td><td>99.47</td><td>99.82</td><td>98.98</td><td>98.52</td><td>68.23</td><td>67.95</td><td>96.38</td><td>98.02</td><td>98.29</td><td>95.06</td><td>98.60</td><td>49.61</td><td>52.73</td><td>93.71</td><td>85.79</td></tr><tr><td>DRAEM+Ours</td><td>96.65</td><td>94.66</td><td>90.80</td><td>98.89</td><td>55.42</td><td>57.00</td><td>91.67</td><td>99.25</td><td>99.60</td><td>98.34</td><td>98.06</td><td>75.44</td><td>71.40</td><td>94.10</td><td>98.23</td><td>98.50</td><td>95.79</td><td>98.47</td><td>38.33</td><td>43.35</td><td>89.17</td><td>84.91</td></tr><tr><td>CPR+Ours</td><td>96.65</td><td>95.69</td><td>91.98</td><td>99.23</td><td>57.66</td><td>57.00</td><td>95.13</td><td>99.68</td><td>99.89</td><td>99.34</td><td>99.21</td><td>82.34</td><td>75.84</td><td>97.83</td><td>97.37</td><td>97.71</td><td>94.83</td><td>99.25</td><td>54.40</td><td>55.17</td><td>94.69</td><td>87.66</td></tr><tr><td>GLASS+Ours</td><td>96.42</td><td>94.85</td><td>90.93</td><td>99.32</td><td>41.52</td><td>50.89</td><td>95.88</td><td>99.73</td><td>99.91</td><td>99.33</td><td>99.10</td><td>70.47</td><td>71.44</td><td>95.41</td><td>97.79</td><td>98.08</td><td>95.24</td><td>98.85</td><td>45.93</td><td>48.21</td><td>93.32</td><td>84.89</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE IX  
COMPARISON UNDER THE ZERO-SHOT GENERATION SETTING WITH DRAEM ON ATAD DATASETS. BOLD REPRESENT OPTIMAL RESULTS.
<table><tr><td>Class</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ACC</td><td>DREAM P-AUROC</td><td>P-AP</td><td>P-F1 IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ACC</td><td>AnomalyAny |P-AUROC</td><td>P-AP</td><td>P-F1</td><td>IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>DPA ACC</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>IOU</td></tr><tr><td>cable</td><td>94.19</td><td>92.90</td><td>87.88</td><td>78.49</td><td>98.14</td><td>78.76</td><td>70.66</td><td></td><td>94.83</td><td></td><td></td><td></td><td></td><td></td><td>69.63</td><td>60.37</td><td>87.46</td><td>98.87</td><td>98.31</td><td>94.44</td><td>91.40</td><td>99.62</td><td>88.11</td><td>78.63</td><td>55.19 94.27</td></tr><tr><td>candle</td><td>88.69</td><td>87.52</td><td>81.65</td><td>53.68</td><td>93.06</td><td>18.38 28.50</td><td>54.86 31.35</td><td>86.69 80.34</td><td>96.37</td><td>93.18 96.47</td><td>85.71 89.73</td><td>83.87 53.16</td><td>98.16 97.05</td><td>78.72 19.51</td><td>28.40</td><td>55.19</td><td>89.89</td><td>97.82</td><td>97.35</td><td>93.55</td><td>72.63</td><td>99.12</td><td>31.78</td><td>37.88 60.96</td><td>84.43</td></tr><tr><td>capsules</td><td>91.60</td><td>93.04</td><td>85.98</td><td>56.36</td><td>94.89</td><td>24.61</td><td>30.28 55.61</td><td>88.49</td><td>92.43</td><td>92.51</td><td>85.11</td><td>70.91</td><td>99.57</td><td>40.00</td><td>53.52</td><td>61.26 52.03</td><td>93.35</td><td>95.90</td><td>95.58</td><td>90.20</td><td>70.91 88.75</td><td>99.73</td><td>73.18</td><td>75.98 65.25 75.76</td><td>95.54</td></tr><tr><td>capsule_mantan cashew</td><td>100.00 94.36</td><td>100.00 96.33</td><td>100.00 91.03</td><td>99.58</td><td>91.47</td><td>53.61</td><td>56.62 79.23</td><td>82.79</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.58</td><td>98.17 96.27</td><td>67.83 5.25</td><td>79.20 12.45</td><td>94.87 36.67</td><td></td><td>100.00 97.56</td><td>100.00 98.33</td><td>100.00 94.41</td><td>39.84</td><td>98.09 97.66</td><td>80.86 40.34</td><td>28.36 43.08 44.72</td><td>91.38 86.83</td></tr><tr><td>eraser</td><td>86.25</td><td>86.76</td><td>80.82</td><td>39.06 62.98</td><td>92.13 98.35</td><td>32.55 49.51</td><td>42.42 43.78 54.62 65.46</td><td>79.72 85.75</td><td>95.38 90.09</td><td>96.98 89.54</td><td>92.68 80.86</td><td>40.62 63.78</td><td>97.19</td><td>51.60</td><td>54.23</td><td>82.46 71.24 81.53</td><td></td><td>91.80</td><td>91.19</td><td>83.49</td><td>64.58</td><td>99.18</td><td>60.07</td><td>72.65</td><td>91.31</td></tr><tr><td>fryum</td><td>96.00</td><td>97.16</td><td>94.61</td><td>40.00</td><td>94.29</td><td>27.33</td><td>37.97 42.31</td><td>64.99</td><td>94.95</td><td>96.05</td><td>91.61</td><td>41.54</td><td>98.09</td><td>36.31</td><td>44.69</td><td>48.79 81.92</td><td></td><td>99.48</td><td>99.66</td><td>98.16</td><td>60.00</td><td>99.23</td><td>53.90</td><td>58.80 52.37 44.01</td><td>72.76</td></tr><tr><td>macaroni2</td><td>95.80</td><td>96.52</td><td>89.36</td><td>50.00</td><td>98.30</td><td>9.13 22.87</td><td>67.89</td><td>91.80</td><td>92.54</td><td>92.64</td><td>85.45</td><td>50.00</td><td>99.69</td><td>19.08</td><td>28.49</td><td>56.48 95.82</td><td></td><td>98.44</td><td>98.40</td><td>94.69</td><td>50.00</td><td>99.92</td><td>12.67</td><td>25.31 59.63</td><td>98.02</td></tr><tr><td>pcb2</td><td>98.21</td><td>97.80</td><td>93.92</td><td>68.28</td><td>83.91</td><td>0.93 3.83</td><td>56.91</td><td>60.01</td><td>98.45</td><td>98.11</td><td>94.32</td><td>74.73</td><td>95.57</td><td>19.20</td><td>23.97</td><td>47.41 84.39</td><td></td><td>98.62</td><td>98.10</td><td>96.09</td><td>61.29</td><td>98.96</td><td>29.42</td><td>37.26 65.14</td><td>86.55</td></tr><tr><td>pcb3 pill</td><td>97.40 97.40</td><td>96.07</td><td>94.74</td><td>56.74</td><td>94.27</td><td>23.93 38.81</td><td>56.97</td><td>88.71</td><td>99.13</td><td>98.72</td><td>96.20</td><td>58.43</td><td>98.54</td><td>22.53</td><td>36.18</td><td>66.71 90.30</td><td></td><td>98.95</td><td>98.61</td><td>93.24</td><td>58.99</td><td>99.06</td><td>9.78</td><td>21.77 68.34 74.27</td><td>89.81</td></tr><tr><td>plastic_nut</td><td>87.81</td><td>99.33 70.37</td><td>95.79 67.20</td><td>29.51 79.29</td><td>96.21 99.22</td><td>47.57 52.04</td><td>49.36 35.26 51.07 80.04</td><td>91.71 85.78</td><td>96.31 84.29</td><td>99.00 61.16</td><td>96.34 59.53</td><td>40.16 80.00</td><td>98.55 99.53</td><td>61.47 50.12</td><td>61.70 50.46</td><td>39.60 95.27 57.08 91.02</td><td></td><td>98.84 91.33</td><td>99.70 76.60</td><td>97.94 71.89</td><td>63.11 80.00</td><td>99.64 99.87</td><td>78.37 70.17</td><td>58.44 81.02</td><td>97.43 93.89</td></tr><tr><td>toy_brick</td><td>75.65</td><td>73.61</td><td>67.06</td><td>60.54</td><td>98.19</td><td>50.47</td><td>51.32 60.36</td><td>82.30</td><td>80.34</td><td>80.42</td><td>69.94</td><td>58.64</td><td>98.11</td><td>43.41</td><td>49.16</td><td>57.49 76.29</td><td></td><td>87.64</td><td>87.78</td><td>78.08</td><td>69.57</td><td>97.24</td><td>63.42</td><td>64.73</td><td>81.14</td></tr><tr><td>type_c</td><td>91.59 94.05</td><td>83.83 83.41</td><td>80.65 77.45</td><td>84.11</td><td>90.57</td><td>20.80 22.30</td><td>29.86 63.17 36.09</td><td>69.55</td><td>93.17</td><td>79.78</td><td>77.97</td><td>81.46 82.97</td><td>92.25</td><td>22.16</td><td>32.56 76.79</td><td>75.75</td><td>98.17</td><td></td><td>91.77</td><td>86.96 82.46</td><td>91.39</td><td>98.42 99.83</td><td>42.70 46.21</td><td>62.34 79.13 86.31</td><td>92.71 96.15</td></tr><tr><td>woodstick zipper_realiad</td><td>99.97</td><td>99.99</td><td>99.60</td><td>82.08 54.13</td><td>99.76 93.79</td><td>49.04</td><td>80.89 49.59 56.34</td><td>95.87 93.45</td><td>93.64 99.98</td><td>86.30 99.99</td><td>78.97 99.60</td><td>83.60</td><td>99.54 98.65</td><td>86.97 61.44</td><td>81.84 72.88 64.63 53.50</td><td>93.40 97.00</td><td>95.04 99.72</td><td></td><td>88.18 99.87</td><td>84.05 98.39 75.87</td><td></td><td>97.69</td><td>89.48 72.40</td><td>83.92 70.16 57.50</td><td>96.23</td></tr><tr><td></td><td>93.06</td><td></td><td></td><td></td><td></td><td></td><td>40.87 58.15</td><td></td><td></td><td></td><td>86.50</td><td>66.47</td><td></td><td>42.85</td><td>48.19</td><td></td><td></td><td></td><td>94.96</td><td>90.87</td><td>70.15</td><td>98.95</td><td>56.04</td><td>56.80</td><td>90.53</td></tr><tr><td>mean</td><td></td><td>90.92</td><td>86.73</td><td>62.18</td><td>94.78</td><td>35.06</td><td></td><td>82.99</td><td>93.87</td><td>91.30</td><td></td><td></td><td>97.81</td><td></td><td>57.09</td><td>88.17</td><td>96.76</td><td></td><td></td><td></td><td></td><td></td><td></td><td>61.96</td><td></td></tr></table>

TABLE X  
COMPARISON UNDER THE ZERO-SHOT GENERATION SETTING WITH DRAEM ON MVTEC-AD DATASETS. BOLD REPRESENT OPTIMAL RESULTS.
<table><tr><td>Class</td><td>I-AUROC</td><td></td><td>I-AP I-Fl</td><td>ACC</td><td>DRAEM</td><td>P-AUROC</td><td>P-AP P-Fl</td><td>IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>ACC</td><td>AnomalyAny P-AUROC</td><td>P-AP</td><td>P-F1</td><td>IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-Fl ACC</td><td>DPA P-AUROC</td><td>P-AP</td><td>P-Fl</td><td>IOU</td><td>PRO</td></tr><tr><td>capsule</td><td>98.68</td><td>99.74</td><td>98.17</td><td>29.55</td><td>96.78</td><td></td><td></td><td></td><td>93.04</td><td></td><td>99.50</td><td></td><td></td><td>97.94</td><td></td><td>52.79 52.09 31.44</td><td>36.11</td><td>92.65</td><td>97.53</td><td>99.52 99.56</td><td>96.74 97.52</td><td>28.03 56.25</td><td>98.37</td><td>58.63</td><td>54.82</td><td>37.63</td><td>93.66</td></tr><tr><td>screw</td><td>96.41 98.72</td><td>98.58</td><td>96.33 97.14</td><td>27.50 77.78</td><td>98.44 97.88</td><td></td><td>49.88</td><td></td><td>91.37 92.31</td><td>100.00</td><td>100.00</td><td>100.00</td><td>31.25</td><td></td><td></td><td>23.26 71.22</td><td>24.31</td><td>88.37</td><td>98.77</td><td></td><td></td><td></td><td></td><td>99.41</td><td>63.92 78.59</td><td>61.77 55.22</td><td>95.19 95.12</td></tr><tr><td>carpet cable</td><td>95.65</td><td>99.62 97.38</td><td>92.06</td><td>87.33</td><td>95.05</td><td></td><td>65.06 56.61</td><td>53.87 39.52</td><td>81.22</td><td>99.32</td><td>99.79 96.43</td><td>97.70 90.62</td><td>77.78 70.67</td><td>98.50</td><td></td><td>67.42 57.54</td><td>55.60 45.62</td><td>93.96 81.89</td><td>99.08 97.32</td><td>99.73 98.50</td><td>97.73 93.92</td><td>74.36 92.67</td><td>99.19 97.13</td><td>71.37</td><td>72.43 66.34</td><td>56.09 46.11</td><td>90.59</td></tr><tr><td>pill</td><td>97.95</td><td>99.62</td><td>97.53</td><td>34.73</td><td>98.25</td><td></td><td>71.39</td><td>31.47</td><td>88.96</td><td>94.40 97.44</td><td>99.53</td><td>97.51</td><td>41.32</td><td>94.68 99.21</td><td>62.66 85.84</td><td>80.85</td><td>36.07</td><td>94.47</td><td>99.51</td><td>99.91</td><td>98.58</td><td>52.10</td><td>99.13</td><td>81.02</td><td>74.47</td><td>47.12</td><td>95.49</td></tr><tr><td>transistor</td><td>93.92</td><td>93.45</td><td>84.62</td><td>78.00</td><td></td><td>80.96</td><td>39.40</td><td>59.06</td><td>72.00</td><td>90.79</td><td>87.87</td><td>88.61</td><td>79.00</td><td>88.14</td><td>39.09</td><td>45.11</td><td>62.69</td><td>69.18</td><td>97.33</td><td>96.86</td><td>92.11</td><td>88.00</td><td>85.18</td><td>43.89</td><td>46.82</td><td>67.67</td><td>79.14</td></tr><tr><td>bottle</td><td>99.76</td><td>99.93</td><td>99.20</td><td>95.18</td><td>98.36</td><td></td><td>78.04</td><td>63.24</td><td>93.72</td><td>99.92</td><td>99.98</td><td>99.21</td><td>90.36</td><td>97.63</td><td>80.36</td><td>72.14</td><td>46.94</td><td>91.76</td><td>99.92</td><td>99.98</td><td>99.21</td><td>93.98</td><td>99.11</td><td>87.37</td><td>79.30</td><td>60.92</td><td>94.65</td></tr><tr><td>hazelnut metal_nut</td><td>100.00</td><td>100.00</td><td>100.00</td><td>66.36</td><td>99.27</td><td>85.49 77.19</td><td>70.80</td><td>56.74</td><td>95.64</td><td>99.68</td><td>99.82</td><td>97.90</td><td>69.09</td><td>99.49</td><td>82.38</td><td>74.62</td><td>48.80</td><td>96.12</td><td>100.00</td><td>100.00</td><td>100.00</td><td>80.00</td><td>99.84</td><td>92.96</td><td>86.58</td><td>74.84</td><td>96.58</td></tr><tr><td>toothbrush</td><td>100.00 100.00</td><td>100.00 100.00</td><td>100.00 100.00</td><td>95.65 76.19</td><td>99.52 99.03</td><td>95.51 53.70</td><td>90.70 55.70</td><td>77.74 56.92</td><td>96.74 93.10</td><td>99.90 100.00</td><td>99.98 100.00</td><td>99.47 100.00</td><td>94.78 78.57</td><td>97.05 98.61</td><td>78.73 44.14</td><td>73.28</td><td>53.97</td><td>94.98</td><td>100.00 100.00</td><td>100.00</td><td>100.00</td><td>94.78</td><td>98.73</td><td>91.53</td><td>86.01</td><td>69.69</td><td>96.31 91.17</td></tr><tr><td>grid</td><td>100.00</td><td>100.00</td><td>100.00</td><td>50.00</td><td>99.74</td><td>66.00</td><td>64.43</td><td>64.59</td><td>97.83</td><td>100.00</td><td>100.00</td><td>100.00</td><td>61.54</td><td>99.71</td><td>66.42</td><td>50.83 63.37</td><td>50.89 62.95</td><td>90.14 97.27</td><td>100.00</td><td>100.00 100.00</td><td>100.00 100.00</td><td>83.33 62.82</td><td>99.14 99.79</td><td>60.54 67.03</td><td>60.05 66.06</td><td>57.35 65.47</td><td>98.05</td></tr><tr><td>leather</td><td>100.00</td><td>100.00</td><td>100.00</td><td>75.00</td><td>98.82</td><td></td><td>60.24</td><td>60.60</td><td>95.03</td><td>100.00</td><td>100.00</td><td>100.00</td><td>54.03</td><td>99.15</td><td>64.93</td><td>65.22</td><td>61.18</td><td>95.29</td><td>100.00</td><td>100.00</td><td>100.00</td><td>79.03</td><td>99.58</td><td>68.35</td><td>65.77</td><td>62.71</td><td>97.59</td></tr><tr><td>tile zipper</td><td>100.00 100.00</td><td>100.00 100.00</td><td>100.00 100.00</td><td>89.74 93.38</td><td>99.74 98.98</td><td></td><td>61.45 97.24 91.17 78.32 73.71</td><td>78.77 49.33</td><td>96.83 92.63</td><td>100.00 99.92</td><td>100.00</td><td>100.00 99.58</td><td>88.89 86.09</td><td>99.68 99.11</td><td>97.20</td><td>91.18</td><td>77.40</td><td>97.55</td><td>100.00</td><td>100.00</td><td>100.00</td><td>89.74</td><td>99.67</td><td>96.83</td><td>90.96</td><td>78.12</td><td>96.85</td></tr><tr><td>wood</td><td>99.82</td><td>99.95</td><td>99.16</td><td>55.70</td><td>95.26</td><td></td><td>70.89 66.15</td><td>44.30</td><td>80.54</td><td>100.00</td><td>99.98 100.00</td><td>100.00</td><td>59.49</td><td>97.15</td><td>78.57 76.69</td><td>73.58 69.04</td><td>36.93 43.96</td><td>95.05 90.09</td><td>100.00 100.00</td><td>100.00 100.00</td><td>100.00 100.00</td><td>93.38 63.29</td><td>99.52 97.53</td><td>83.11 81.04</td><td>77.39 74.67</td><td>59.67 50.23</td><td>96.95 90.48</td></tr><tr><td>mean</td><td>98.73</td><td>99.22</td><td>97.61</td><td>68.81</td><td>97.07</td><td></td><td>67.01 65.52</td><td>54.14</td><td>90.73</td><td>98.60</td><td>98.86</td><td>97.86</td><td>67.14</td><td>97.65</td><td>66.95</td><td>64.51</td><td>49.56</td><td>91.25</td><td>99.30</td><td>99.60</td><td>98.39</td><td>75.45</td><td>98.09</td><td>75.08</td><td>70.90</td><td></td><td>93.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>59.26</td><td></td></tr></table>

TABLE XI

COMPARISON UNDER THE ZERO-SHOT GENERATION SETTING WITH DRAEM ON VISA DATASETS. BOLD REPRESENT OPTIMAL RESULTS.
<table><tr><td>Class</td><td>I-AUROC</td><td>I-AP</td><td>I-Fl</td><td>DRAEM ACC</td><td>| P-AUROC</td><td>P-AP P-Fl</td><td>IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-Fl</td><td>AnomalyAny ACC</td><td> P-AUROC</td><td>P-AP</td><td>P-F1 IOU</td><td>PRO</td><td>I-AUROC</td><td>I-AP</td><td>I-Fl</td><td>DPA ACC</td><td>P-AUROC</td><td>P-AP</td><td>P-Fl</td><td>IOU</td></tr><tr><td>candle</td><td>95.23</td><td>96.19</td><td>90.43</td><td>95.95</td><td></td><td>38.60</td><td>61.34</td><td>89.06</td><td>96.15</td><td>96.49</td><td>89.55 91.92</td><td>50.50 62.50</td><td>96.95 99.56</td><td>18.77 44.16</td><td>27.76</td><td>52.42 89.16</td><td>97.01</td><td>97.53</td><td>92.86</td><td>63.50</td><td>99.38 99.81</td><td>35.49</td><td>39.71</td><td>56.88 91.19</td></tr><tr><td>capsules</td><td>96.57</td><td>97.75</td><td>93.75</td><td>98.60</td><td>32.44 34.15</td><td>41.55</td><td>45.62</td><td>87.33</td><td>95.15</td><td>97.21</td><td></td><td></td><td></td><td>52.35</td><td>52.19</td><td>95.11</td><td>98.05</td><td>98.78</td><td>96.00</td><td>62.50 94.67</td><td>95.08</td><td>56.95 65.87</td><td>58.29</td><td>94.46</td></tr><tr><td>cashew chewinggum</td><td>92.78 98.62</td><td>96.07 99.41</td><td>91.08 96.48</td><td>92.15 98.61</td><td>22.43 35.59</td><td>26.37 48.17</td><td>12.49 58.03</td><td>77.37 79.56</td><td>95.42 95.24</td><td>97.65 97.89</td><td>93.20 38.00 48.67</td><td>98.27 98.32</td><td>44.89 37.94</td><td>48.21 45.47</td><td>32.49 45.84</td><td>82.07 75.81</td><td>98.90 98.92</td><td>99.42 99.50</td><td>97.56 96.97</td><td>69.33</td><td>99.25</td><td>24.69 25.15 39.55 50.20</td><td>29.34 67.46</td><td>85.70 79.14</td></tr><tr><td>fryum</td><td>92.28</td><td>96.52</td><td>89.95</td><td>68.00 45.33 94.20</td><td>38.08</td><td>40.96</td><td>48.76</td><td>77.14</td><td>95.96</td><td>97.70</td><td>91.19 93.33</td><td>98.93</td><td></td><td>62.46 64.53</td><td>42.37</td><td>81.96</td><td>98.66</td><td>99.33</td><td>96.52</td><td>58.00</td><td>94.99</td><td>40.09 45.10</td><td>53.57</td><td>91.19</td></tr><tr><td>macaronil macaroni2</td><td>98.77 99.50</td><td>98.71</td><td>96.52</td><td>51.50 99.46 50.00</td><td>23.10</td><td>26.60</td><td>75.32</td><td>92.60</td><td>97.80</td><td>97.66</td><td>92.38 51.50 89.36</td><td>99.80</td><td>11.78</td><td>24.73</td><td>59.17</td><td>94.41</td><td>99.55</td><td>99.53</td><td>97.54</td><td>50.50</td><td>99.96</td><td>23.08 32.44</td><td>66.52</td><td>96.63</td></tr><tr><td>pcb1</td><td>96.47</td><td>99.50 96.51</td><td>96.62 91.08</td><td>99.95 99.61</td><td>12.50 70.03</td><td>20.12 68.25</td><td>71.08 58.21</td><td>97.16 88.67</td><td>95.77 97.46</td><td>96.49 97.63 91.54</td><td>50.00 63.50</td><td>98.29 98.89</td><td>9.12 21.64</td><td>22.85 34.50</td><td>67.87 46.57</td><td>91.80 90.29</td><td>99.23 99.02</td><td>99.29 98.91</td><td>95.61 50.00 96.55</td><td>62.50</td><td>99.93 99.55</td><td>14.89 24.06 66.62 61.98</td><td>59.26 61.48</td><td>97.24 89.39</td></tr><tr><td>pcb2</td><td>99.33</td><td>99.32</td><td>97.03</td><td>98.37</td><td>7.68</td><td>16.43</td><td>57.53</td><td>73.79</td><td>98.38</td><td>98.31 94.58 96.12</td><td>74.00</td><td>98.26</td><td>23.36</td><td>30.62</td><td>44.58</td><td>82.65</td><td>99.40</td><td>99.40</td><td>96.62</td><td>61.50</td><td>98.46</td><td>14.25 21.56</td><td>60.43</td><td>82.13</td></tr><tr><td>pcb3 pcb4</td><td>98.23 98.80</td><td>96.30 95.61</td><td>96.04 97.46</td><td>96.87 98.74</td><td>16.80 43.25</td><td>25.80</td><td>66.54</td><td>89.25</td><td>98.95</td><td>98.84</td><td>56.22 62.69</td><td>97.27 98.83</td><td>29.09 34.18</td><td>41.74</td><td>62.75</td><td>90.08</td><td>99.02</td><td>98.80</td><td>96.62 58.21</td><td>98.62 98.95</td><td></td><td>30.82 37.05 52.06</td><td>64.14</td><td>91.00 92.76</td></tr><tr><td>pipe_fryum</td><td>95.92</td><td>97.97</td><td>93.66</td><td>70.65 38.67 95.43</td><td>21.83</td><td>47.91 30.82</td><td>61.60 45.22</td><td>93.18 90.75</td><td>99.55 96.94</td><td>99.52 98.43</td><td>97.09 93.84 35.33</td><td>97.22</td><td>37.41</td><td>39.38 44.03</td><td>57.84 39.60</td><td>93.54 90.89</td><td>98.51 96.84</td><td>95.34 98.31</td><td>96.48 83.08 95.00 41.33</td><td>96.72</td><td></td><td>45.18 48.80 51.94</td><td>62.78 54.60</td><td>93.74</td></tr><tr><td>mean</td><td>96.88</td><td>97.49</td><td>94.18</td><td>56.67 97.33</td><td>29.82</td><td>35.96</td><td>55.14</td><td>86.32</td><td>96.63</td><td>97.50 92.52</td><td>52.47</td><td>98.50</td><td>32.06</td><td>40.15</td><td>49.36</td><td>88.48</td><td>98.59</td><td>98.68</td><td>96.19 62.93</td><td>98.39</td><td>36.70</td><td>42.26</td><td>57.90</td><td>90.38</td></tr></table>

[8] L. Fan, D. Fan, Z. Hu, Y. Ding, D. Di, K. Yi, M. Pagnucco, and Y. Song, “Manta: A large-scale multi-view and visual-text anomaly detection dataset for tiny objects,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 25 518–25 527.

[9] Z. Liu, Y. Zhou, Y. Xu, and Z. Wang, “Simplenet: A simple network for image anomaly detection and localization,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 20 402–20 411.

[10] J. Guo, S. Lu, W. Zhang, F. Chen, H. Li, and H. Liao, “Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 20 405–20 415.

[11] W. Luo, Y. Cao, H. Yao, X. Zhang, J. Lou, Y. Cheng, W. Shen, and W. Yu, “Exploring intrinsic normal prototypes within a single image for universal anomaly detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9974–9983.

[12] H. Sun, Y. Cao, H. Dong, and O. Fink, “Unseen visual anomaly generation,” in Proceedings of the Computer Vision and Pattern Recognition

Conference, 2025, pp. 25 508–25 517.